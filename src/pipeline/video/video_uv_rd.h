// Chroma coefficient-level RD and lambda calibration (AV2ENC_RD_UVLEVEL).
//
// Milestone 5 of the RD-engine program splits the chroma gap into four parts:
//
//   * TX size / type / partition: **not legal**.  §5.20.6.3 derives the chroma
//     transform size from the chroma residual block (`get_tx_size(plane > 0,
//     ...)` ignores the luma partition; `Enable_Chroma_DCTOnly = 1` in this
//     writer), and TX partitioning is luma-only, so every chroma leaf has
//     exactly one DCT_DCT transform per plane.  There is no candidate set to
//     search and this header does not invent one.
//   * coefficient levels: the leaf used the scalar `compute_tx` rounding.  This
//     file supplies the missing trellis candidates (scalar, and the RDOQ full
//     coefficient DP at three rate weights), each priced on the *same* J:
//
//         J = D_perc + lambda * (mode_symbol_bits + exact_coefficient_bits)
//
//     D_perc is `eff_perc::weighted_sse` of the quantised reconstruction (the
//     same CSF/chroma weight the leaf charges), and the coefficient bits are
//     the static §8.3 EOB / txb_skip / coeff_base / coeff_br / escape / sign
//     information bits (`txtsearch::scan_coefficient_bits`).  The trellis only
//     proposes levels; the exact J accepts or rejects them, so a trellis model
//     error can never win a block.
//   * lambda: chroma is miscalibrated until measured.  `uv_lambda_multiplier()`
//     is the env-only calibration knob (default 1.0); the measured value is
//     documented in docs/rd-lane5.md and set as the default once the sweep is
//     closed.
//   * CfL: the derived-alpha hypothesis is already in the candidate set.  The
//     explicit-alpha syntax is unreachable in this writer (`code_uv_mode()`
//     codes `cfl_alpha_signs/u/v` only under `Enable_MHCCP`), which the nightly
//     recipe does not enable; see docs/rd-lane5.md.
//
// Interface (documented for the integrator):
//
//   LevelParams    geometry + quantiser + §8.3 qctx of one chroma plane.
//   make_variant() builds one candidate `TxPlan` for a prediction.  Variant 0
//                  is the caller's scalar plan (copied), variants 1..N are the
//                  RDOQ trellis at `trellis_scale * mul`.
//
// The plans it returns are writer-ready: scalar levels, §5.20.7.30 raster
// levels, EOB, coded-size reconstruction.  No syntax changes, no CLI, no CMake.
//
// Scope: the scalar-quantiser path only (QM off, dc == ac).  The caller falls
// back to its own `compute_tx` for QM / split DC-AC, where the trellis would
// need a per-coefficient step the current RDOQ Config cannot express.  Default
// off: nothing reads this file unless AV2ENC_RD_UVLEVEL=1.
#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "codec/rate/rdo/rdo.h"
#include "codec/rate/rdoq/rdoq.h"
#include "codec/rate/rdoq/rdoq_av2_model.h"
#include "codec/transform/tx_partition/tx_scan.h"
#include "codec/transform/variants/tx_variants.h"

namespace av2enc::video::uv_rd {

// AV2ENC_RD_UVLEVEL: master gate for chroma coefficient-level RD.  Requires
// AV2ENC_RD_UVFULL (the mode search owns the prediction set).  Env only.
inline bool enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_RD_UVLEVEL");
    if (env == nullptr || *env == '\0') return false;
    if (std::strcmp(env, "0") == 0) return false;
    const auto equal = [env](const char* word) {
      for (std::size_t i = 0; word[i] != '\0'; ++i) {
        if (std::tolower(static_cast<unsigned char>(env[i])) != word[i]) {
          return false;
        }
      }
      return env[std::strlen(word)] == '\0';
    };
    return !(equal("false") || equal("off") || equal("no"));
  }();
  return value;
}

inline double env_positive(const char* name, double fallback) {
  const char* env = std::getenv(name);
  if (env == nullptr || *env == '\0') return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(env, &end);
  if (end == env || !std::isfinite(parsed) || parsed <= 0.0) return fallback;
  return parsed;
}

// AV2ENC_RD_UVLAMBDA: multiplier on the chroma *mode*-symbol / candidate
// lambda.  1.0 = the leaf lambda the M4 search used (measured as calibrated).
inline double lambda_multiplier() {
  static const double value = env_positive("AV2ENC_RD_UVLAMBDA", 1.0);
  return value;
}

// AV2ENC_RD_UVLEVEL_LAMBDA: multiplier on the *level*-selection lambda.  The
// static coefficient model overprices chroma levels relative to the perceptual
// D, so this site is calibrated separately.  Measured 0.125 on the 240p30 /
// 100 kbps point (docs/rd-lane5.md); this is the default when the gate is on.
inline double level_lambda_multiplier() {
  static const double value =
      env_positive("AV2ENC_RD_UVLEVEL_LAMBDA", 0.125);
  return value;
}

// AV2ENC_RD_UVLEVEL_SCALE: RDOQ SSE scale (`lambda = scale * step^2`).  The
// candidate is re-scored on the exact J, so this only shapes the proposals.
inline double trellis_scale() {
  static const double value = env_positive("AV2ENC_RD_UVLEVEL_SCALE", 0.10);
  return value;
}

// AV2ENC_RD_UVLEVEL_TOPN: stage-2 modes re-scored when the level stage is on.
// Unlimited re-scoring (8) moved K by +0.03 inside the ripple at 4x the encode
// time, so the default stays at the M4 value (3); the knob exists for
// experiments.
inline int level_top_n() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_RD_UVLEVEL_TOPN");
    if (env == nullptr || *env == '\0') return 3;
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed < 0 || parsed > 16) return 3;
    return static_cast<int>(parsed);
  }();
  return value;
}

// AV2ENC_RD_UVLEVEL_INTER: also run the level stage on the inter chroma
// residual (the P-frame chroma bulk).  On by default when the master gate is
// on; an explicit 0/false/off keeps the M4 scalar plan for a key-only
// ablation.
inline bool inter_enabled() {
  static const bool value = [] {
    if (!enabled()) return false;
    const char* env = std::getenv("AV2ENC_RD_UVLEVEL_INTER");
    if (env == nullptr || *env == '\0') return true;
    if (std::strcmp(env, "0") == 0) return false;
    const auto equal = [env](const char* word) {
      for (std::size_t i = 0; word[i] != '\0'; ++i) {
        if (std::tolower(static_cast<unsigned char>(env[i])) != word[i]) {
          return false;
        }
      }
      return env[std::strlen(word)] == '\0';
    };
    return !(equal("false") || equal("off") || equal("no"));
  }();
  return value;
}

// AV2ENC_RD_UVLEVEL_STATS: one-line exit summary.  Diagnostic only.
// Wavefront workers can touch the level stage concurrently, so the diagnostic
// counters are atomic; they never feed a decision.
struct Counters {
  std::atomic<unsigned long long> blocks{0};  // chroma leaves
  std::atomic<unsigned long long> plans{0};   // proposals built
  std::atomic<unsigned long long> trellis_runs{0};
  std::atomic<unsigned long long> trellis_failures{0};
  std::atomic<unsigned long long> flips{0};   // plane decisions won
};

inline Counters& counters() {
  static Counters value;
  return value;
}

inline void print_stats();

inline bool stats_enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_RD_UVLEVEL_STATS");
    if (env == nullptr || *env == '\0') return false;
    if (std::strcmp(env, "0") == 0) return false;
    std::atexit(&print_stats);
    return true;
  }();
  return value;
}

inline void print_stats() {
  if (!enabled()) return;
  const Counters& c = counters();
  std::fprintf(stderr,
               "uv-level: blocks=%llu plans=%llu trellis=%llu failed=%llu "
               "flips=%llu\n",
               static_cast<unsigned long long>(c.blocks.load()),
               static_cast<unsigned long long>(c.plans.load()),
               static_cast<unsigned long long>(c.trellis_runs.load()),
               static_cast<unsigned long long>(c.trellis_failures.load()),
               static_cast<unsigned long long>(c.flips.load()));
}

// One chroma plane's geometry and quantiser.  `dc`/`ac` are the §7.14.2
// plane steps (`PlaneQuantStep`); the trellis runs on the scalar quantiser
// (dc == ac, any step).  Split DC/AC and QM keep the caller's compute_tx.
struct LevelParams {
  int w = 0;
  int h = 0;
  int plane = 1;
  int dc = txv::kQuantStep;
  int ac = txv::kQuantStep;
  int qctx = 0;
  // §8.3.2 coefficient CDF select: the inter residual path uses the inter
  // rows for the U txb_skip proposal; the exact cost is scored by the caller.
  bool is_inter = false;
  // Visible rows of the chroma block (--video-edge-mask); -1 = all rows.
  int mask_row = -1;
};

inline bool supported(const LevelParams& p) {
  return p.w > 0 && p.h > 0 && p.w * p.h <= 4096 && p.dc == p.ac &&
         p.ac > 0 && (p.plane == 1 || p.plane == 2);
}

// Builds one trellis candidate from a tightly packed source block and the
// candidate prediction.  Returns false when the geometry/quantiser is outside
// the supported scalar path; the caller then keeps its scalar plan.  `plan`
// is left untouched on failure.
inline bool trellis_plan(const LevelParams& p, const std::uint8_t* src,
                         const std::uint8_t* pred, double sse_scale,
                         TxPlan& plan, std::string& error) {
  if (!supported(p) || src == nullptr || pred == nullptr || sse_scale <= 0.0) {
    return false;
  }
  const auto params_opt = txv::make_tx_params(p.w, p.h, txv::TxType::kDctDct);
  if (!params_opt.has_value()) return false;
  // Same plane-step scaling as compute_tx(): the forward target is
  // `lround(coeff * quant_factor * 528 / ac)` and the reconstruction divides
  // the 528 dequant by the plane step.  dc == ac is enforced by supported().
  txv::TxParams params = *params_opt;
  if (p.ac != txv::kQuantStep) {
    params.quant_factor *=
        static_cast<double>(txv::kQuantStep) / static_cast<double>(p.ac);
  }
  const int coded_pels = p.w * p.h;
  const int area = params.w * params.h;

  std::vector<std::int16_t> residual_coded(
      static_cast<std::size_t>(coded_pels));
  for (int y = 0; y < p.h; ++y) {
    const bool hidden = p.mask_row >= 0 && y >= p.mask_row;
    for (int x = 0; x < p.w; ++x) {
      const std::size_t i =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(p.w) +
          static_cast<std::size_t>(x);
      residual_coded[i] = static_cast<std::int16_t>(
          hidden ? 0 : static_cast<int>(src[i]) - static_cast<int>(pred[i]));
    }
  }
  std::vector<std::int16_t> residual_adj(
      static_cast<std::size_t>(area));
  txv::downsample_residual(residual_coded.data(), params,
                           residual_adj.data());
  std::vector<double> coeff(static_cast<std::size_t>(area));
  txv::forward_block(residual_adj.data(), params, coeff.data());

  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  if (!::av2enc::video::txpartition::get_scan(
          params.tx_size,
          ::av2enc::video::txpartition::TxScanClass::k2D, scan)) {
    return false;
  }
  std::vector<std::int32_t> scalar_scan(static_cast<std::size_t>(area), 0);
  std::vector<av2enc::rdoq::Coefficient> input(
      static_cast<std::size_t>(area));
  for (int c = 0; c < area; ++c) {
    const int pos = scan[static_cast<std::size_t>(c)];
    const std::int32_t scalar = static_cast<std::int32_t>(
        std::lround(coeff[static_cast<std::size_t>(pos)] *
                    params.quant_factor));
    scalar_scan[static_cast<std::size_t>(c)] = scalar;
    input[static_cast<std::size_t>(c)].original =
        coeff[static_cast<std::size_t>(pos)];
    input[static_cast<std::size_t>(c)].scalar_level = scalar;
    input[static_cast<std::size_t>(c)].level_context = 0;
  }

  av2enc::rdoq::av2::BlockInfo info;
  info.plane = p.plane;
  info.params = params;
  info.txb_skip_context = 0;
  info.dc_sign_context = 0;
  info.qctx = p.qctx;
  info.is_inter = p.is_inter;
  av2enc::rdoq::av2::Model model;
  if (!model.init(info, scalar_scan, &error)) return false;

  av2enc::rdoq::Config config;
  config.quant_step = 1.0 / params.quant_factor;
  config.lambda =
      av2enc::rdo::lambda_sse_from_quant_step(config.quant_step, sse_scale);
  config.max_coefficients = area;
  av2enc::rdoq::RdoqResult result;
  if (!av2enc::rdoq::rdoq_quantize(input, config, model.rate_model(), result,
                                   &error)) {
    return false;
  }

  plan = TxPlan{};
  plan.tx_size = params.tx_size;
  plan.adj_w = params.w;
  plan.adj_h = params.h;
  plan.levels.assign(static_cast<std::size_t>(area), 0);
  for (int c = 0; c < result.eob; ++c) {
    plan.levels[scan[static_cast<std::size_t>(c)]] =
        result.levels[static_cast<std::size_t>(c)];
  }
  plan.eob = result.eob;

  std::vector<std::int32_t> dequant(static_cast<std::size_t>(area));
  std::vector<std::int16_t> recon_residual(
      static_cast<std::size_t>(coded_pels));
  // §7.14.4 dequantise with the plane step, exactly like the dc == ac branch of
  // compute_tx(), then the shared inverse transform.
  for (int i = 0; i < area; ++i) {
    const std::int32_t qc = plan.levels[static_cast<std::size_t>(i)];
    const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
    const std::int64_t high = (aq * p.ac) & 0xFFFFFF;
    const std::int32_t dq =
        txv::round2(static_cast<std::int32_t>(high), txv::kQuantTableBits) /
        params.dq_denom;
    dequant[static_cast<std::size_t>(i)] =
        std::max(-(1 << 15), std::min((1 << 15) - 1, qc < 0 ? -dq : dq));
  }
  txv::inverse_block(dequant.data(), params, recon_residual.data());
  plan.recon.resize(static_cast<std::size_t>(coded_pels));
  for (int y = 0; y < p.h; ++y) {
    for (int x = 0; x < p.w; ++x) {
      const std::size_t i =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(p.w) +
          static_cast<std::size_t>(x);
      const int value = static_cast<int>(pred[i]) +
                        static_cast<int>(recon_residual[i]);
      plan.recon[i] = static_cast<std::uint8_t>(
          value < 0 ? 0 : (value > 255 ? 255 : value));
    }
  }
  return true;
}

// Number of trellis proposals per plane.  0 disables the level stage (the
// scalar plan is still returned).  Three weights bracket the proposal space:
// a lighter and a heavier rate weight around the calibrated scale.
inline int trellis_variants() {
  static const int value = [] {
    if (!enabled()) return 0;
    const char* env = std::getenv("AV2ENC_RD_UVLEVEL_TRIES");
    if (env == nullptr || *env == '\0') return 3;
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed < 0 || parsed > 6) return 3;
    return static_cast<int>(parsed);
  }();
  return value;
}

// Weight multiplier of variant `v` (1-based; variant 0 is the scalar plan).
inline double variant_weight(int v) {
  static const double kWeights[6] = {0.5, 1.0, 2.0, 0.25, 4.0, 8.0};
  const int index = v - 1;
  if (index < 0) return 1.0;
  return kWeights[index < 6 ? index : 5];
}

}  // namespace av2enc::video::uv_rd
