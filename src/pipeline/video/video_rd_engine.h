// One RD cost engine for every video decision (docs/rd-engine.md, lane L1).
//
//   J = D_perc + lambda(qindex, site) * R_exact
//
// The historical paths each carry their own proxy: the key-intra leaf prices
// its TX partition on plain SSE + a constant 3-bit partition estimate, the
// inter leaf on SSE + `estimate_tx_bits`, chroma on a U-only SSE.  This file
// replaces the *cost*, not the candidate set: one perceptual distortion, one
// exact §8.2 information rate split by symbol family, one lambda per site.
//
// `D_perc` is the AV2ENC_EFF_PERC distortion (`video_eff_perc.h`): per-picture
// Q8 source weights (3x3/7x7 CSF gains, visible-row clip, source-extent
// clamp) applied to the reconstruction error.  When the eff-perc gate is off
// the weight map is invalid and `distortion()` degrades to the same weighted
// SSE with neutral weights, i.e. plain SSE + the caller's CSF gains.
//
// `R_exact` is the §8.2 information cost.  Coefficient families (txb_skip,
// EOB, coeff_base, coeff_br, coeff_mag, sign) come from the *same* §8.3 model
// the TX search and the writer use (`codec/rate/rdoq/rdoq_av2_model.h`) driven
// through its SymbolSink hook: every emitted symbol is classified by the CDF
// row it was priced against, so the family split sums to the model total by
// construction.  TX type / TX partition / partition / mode-MV have their own
// `-log2 p` helpers below; the caller passes the row it will actually code
// against (live §8.3 CDF image or the static §9.3 slice).
//
// The model is header-only on purpose: the video pipeline's source list is an
// explicit CMake target owned by the integrator, and a lane may not edit it.
// `video_rd_engine.cpp` is the translation unit CMake will eventually list;
// until then it is also the standalone self-check for the family split.
//
// Gate (env only, no CLI flag, default off):
//   AV2ENC_RD_ENGINE=1                 master gate
//   AV2ENC_RD_ENGINE_KEY=<x>           key-intra lambda scale (calibrated)
//   AV2ENC_RD_ENGINE_INTER=<x>         inter lambda scale (calibrated)
//   AV2ENC_RD_ENGINE_CHROMA=<x>        chroma lambda scale (calibrated)
//   AV2ENC_RD_ENGINE_AUDIT=<path>      shadow -log2(p) audit against
//                                      AV2ENC_BIT_CENSUS (JSON totals row)
//
// Unset (or 0/false/off/no) does no work and every pinned stream keeps its
// bytes.  The engine never writes bits and never changes a syntax decision on
// its own; a caller must opt in at a site.
#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "codec/entropy/qctx/qctx.h"
#include "codec/rate/rdoq/rdoq_av2_model.h"
#include "codec/transform/tx_partition/tx_scan.h"
#include "codec/transform/variants/tx_variants.h"
#include "pipeline/video/video_eff_perc.h"

namespace av2enc::video::rd_engine {

namespace av2 = av2enc::rdoq::av2;
namespace qctx = av2enc::still::qctx;
namespace txpart = av2enc::video::txpartition;
namespace txv = av2enc::txv;

inline bool parse_off(const char* value) {
  if (value == nullptr || *value == '\0') return true;
  if (std::strcmp(value, "0") == 0) return true;
  auto equal = [value](const char* word) {
    for (std::size_t i = 0; word[i] != '\0'; ++i) {
      if (std::tolower(static_cast<unsigned char>(value[i])) != word[i]) {
        return false;
      }
    }
    return value[std::strlen(word)] == '\0';
  };
  return equal("false") || equal("off") || equal("no");
}

inline double env_positive(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || !std::isfinite(parsed) || parsed <= 0.0) return fallback;
  return parsed;
}

inline double env_unit(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || !std::isfinite(parsed) || parsed < 0.0) return fallback;
  return parsed;
}

// ---------------------------------------------------------------------------
// Rate families
// ---------------------------------------------------------------------------
// The names follow the census `fine` labels (video_syntax_writer.h) so the
// shadow audit is a direct comparison, plus the coarse families the census
// tracks separately (partition, intra mode / inter mode-MV, header rows).
enum Family : int {
  kFamPartition = 0,
  kFamModeMv,
  kFamTxbSkip,
  kFamEob,
  kFamCoeffBase,
  kFamCoeffBr,
  kFamCoeffMag,
  kFamSign,
  kFamTxType,
  kFamTxPartition,
  kFamHeader,
  kFamOther,
  kFamCount
};

inline const char* family_name(int family) {
  switch (family) {
    case kFamPartition: return "partition";
    case kFamModeMv: return "mode_mv";
    case kFamTxbSkip: return "txb_skip";
    case kFamEob: return "eob";
    case kFamCoeffBase: return "coeff_base";
    case kFamCoeffBr: return "coeff_br";
    case kFamCoeffMag: return "coeff_mag";
    case kFamSign: return "sign";
    case kFamTxType: return "tx_type";
    case kFamTxPartition: return "tx_partition";
    case kFamHeader: return "header";
    default: return "other";
  }
}

// `-log2 p` of one symbol under an AV2 forward-CDF row (n-1 boundaries plus
// the rate/count pair).  This is the §8.2 ideal code length; the range coder
// is within a fraction of a bit per block of it.
inline double symbol_bits(const std::uint16_t* cdf, int n, int symbol) {
  if (cdf == nullptr || n < 2 || symbol < 0 || symbol >= n) return 0.0;
  const int lo = symbol == 0 ? 0 : static_cast<int>(cdf[symbol - 1]);
  const int hi = symbol == n - 1 ? 32768 : static_cast<int>(cdf[symbol]);
  const int freq = hi - lo;
  if (freq <= 0) return 0.0;
  return std::log2(32768.0 / static_cast<double>(freq));
}

// One candidate's exact rate, split by family.  `total()` is the sum, so a
// caller can never drift from the model it is charging.
struct RateBreakdown {
  double bits[kFamCount] = {};

  void add(int family, double value) {
    if (family >= 0 && family < kFamCount) bits[family] += value;
  }
  double get(int family) const {
    return family >= 0 && family < kFamCount ? bits[family] : 0.0;
  }
  double total() const {
    double sum = 0.0;
    for (double value : bits) sum += value;
    return sum;
  }
  // Everything a coefficient/TX plan owns: the residual families plus the TX
  // type and TX partition symbols.
  double residual_total() const {
    return get(kFamTxbSkip) + get(kFamEob) + get(kFamCoeffBase) +
           get(kFamCoeffBr) + get(kFamCoeffMag) + get(kFamSign) +
           get(kFamTxType) + get(kFamTxPartition);
  }
  void operator+=(const RateBreakdown& other) {
    for (int i = 0; i < kFamCount; ++i) bits[i] += other.bits[i];
  }
};

// ---------------------------------------------------------------------------
// Per-family exact-rate API
// ---------------------------------------------------------------------------
// TXB_SKIP / EOB / coefficient families are priced by `block_rate()` below,
// which needs the whole quantised block because the contexts depend on
// already-coded neighbours.  The helpers here price the syntax it does not
// cover, from whatever CDF row the caller will code against.

inline double tx_type_bits(const std::uint16_t* cdf, int n, int symbol) {
  return symbol_bits(cdf, n, symbol);
}

inline double tx_partition_bits(const std::uint16_t* cdf, int n, int symbol) {
  return symbol_bits(cdf, n, symbol);
}

inline double partition_bits(const std::uint16_t* cdf, int n, int symbol) {
  return symbol_bits(cdf, n, symbol);
}

inline double mode_bits(const std::uint16_t* cdf, int n, int symbol) {
  return symbol_bits(cdf, n, symbol);
}

// §5.20.7.28 read_quant escape length (q-length + golomb + literal), the
// `coeff_mag` family for a level at or above the base/br maximum.  Identical
// arithmetic to the §8.3 model, exposed for a per-level diagnostic.
inline double coeff_mag_bits(std::int32_t abs_value, std::int32_t pre_level,
                             std::int32_t predictor) {
  if (abs_value <= pre_level) return 0.0;
  const std::int32_t x = abs_value - pre_level;
  std::int32_t m = 0;
  if (predictor > 0) {
    std::uint32_t value = static_cast<std::uint32_t>(predictor);
    while (value > 1U) {
      value >>= 1U;
      ++m;
    }
  }
  if (m < 1) m = 1;
  if (m > 6) m = 6;
  const std::int32_t k = m + 1;
  const std::int32_t c_max = std::min(m + 4, 6);
  double bits = 0.0;
  if (x < (c_max << m)) {
    bits += static_cast<double>((x >> m) + 1);  // q zero bits + terminating 1
    bits += static_cast<double>(m);             // literal remainder
    return bits;
  }
  bits += static_cast<double>(c_max);
  for (std::int32_t g = 0; g <= 31; ++g) {
    const std::int32_t length = k + g;
    const std::int32_t x_base = (c_max << m) + (1 << length) - (1 << k);
    if (x < x_base + (1 << length)) {
      bits += static_cast<double>(g + 1);
      bits += static_cast<double>(length);
      return bits;
    }
  }
  return bits + 40.0;
}

// One transform block in the engine's own view: §5.20.7.30 scan-order levels
// over the adjusted grid, plus the geometry the writer will code.
//
// `txb_skip_context` / `dc_sign_context` are the *live* §5.20.7.27 neighbour
// contexts (above/left cumulative level and DC category).  Leaving both at 0
// reproduces the historical frozen-context model; a caller that tracks the
// writer's above/left state must pass the real values, because the frozen rows
// are 1.2-2.3x too expensive (docs/rd-rate.md §2).
struct BlockPlan {
  int tx_size = 0;
  int tx_type = 0;
  int plane = 0;
  int area = 0;  // txpart::tx_scan_area(tx_size); levels_scan has `area`
  const std::int32_t* levels_scan = nullptr;
  int txb_skip_context = 0;
  int dc_sign_context = 0;
};

namespace detail {

// Classifies every §8.3 symbol by the CDF row the model priced it against and
// charges the same -log2(p) back, so the family split and the model total are
// the same number.  Literals are sequenced by the model's fixed order
// (EOB escapes first, then per coefficient: escape, then sign), which lets
// the mag/sign split be exact without touching the model.
class FamilySink : public av2::SymbolSink {
 public:
  FamilySink(RateBreakdown* out, const qctx::CoeffCdfSlice& slice)
      : out_(out), slice_(&slice) {}

  double symbol(const std::uint16_t* cdf, int n, int symbol) override {
    const double bits = symbol_bits(cdf, n, symbol);
    const int family = classify(cdf);
    out_->add(family, bits);
    if (family == kFamCoeffBase || family == kFamCoeffBr) {
      coeff_phase_ = true;
    }
    return bits;
  }

  double literal(double bits) override {
    // The writer's census tags every §5.20.7.28 escape *and* the AC sign
    // literal as `coeff_mag` (RawScope in code_tx_view); EOB escape/extra
    // literals sit inside write_eob's `eob` scope.  Mirror that so the shadow
    // audit compares like for like.  `sign` is therefore the §7.13.3.9 DC sign
    // symbols only.
    out_->add(coeff_phase_ ? kFamCoeffMag : kFamEob, bits);
    return bits;
  }

 private:
  static bool in(const std::uint16_t* p, const std::uint16_t* base,
                 std::size_t count) {
    return p >= base && p < base + count;
  }

  int classify(const std::uint16_t* p) const {
    const auto flat = [](const auto* row) {
      return reinterpret_cast<const std::uint16_t*>(row);
    };
    const qctx::CoeffCdfSlice& s = *slice_;
    if (in(p, flat(s.txb_skip), 2 * 5 * 10 * 3) ||
        in(p, flat(s.v_txb_skip), 12 * 3)) {
      return kFamTxbSkip;
    }
    if (in(p, flat(s.eob_pt_16), 3 * 6) ||
        in(p, flat(s.eob_pt_32), 3 * 7) ||
        in(p, flat(s.eob_pt_64), 3 * 8) ||
        in(p, flat(s.eob_pt_128), 3 * 9) ||
        in(p, flat(s.eob_pt_256), 3 * 9) ||
        in(p, flat(s.eob_pt_512), 3 * 9) ||
        in(p, flat(s.eob_pt_1024), 3 * 9) ||
        in(p, flat(s.eob_extra), 3)) {
      return kFamEob;
    }
    if (in(p, flat(s.coeff_base), 5 * 20 * 2 * 5) ||
        in(p, flat(s.coeff_base_lf), 5 * 33 * 2 * 7) ||
        in(p, flat(s.coeff_base_eob), 5 * 4 * 4) ||
        in(p, flat(s.coeff_base_lf_eob), 5 * 4 * 6) ||
        in(p, flat(s.coeff_base_uv), 12 * 5) ||
        in(p, flat(s.coeff_base_lf_uv), 12 * 7) ||
        in(p, flat(s.coeff_base_eob_uv), 4 * 4) ||
        in(p, flat(s.coeff_base_lf_eob_uv), 4 * 6) ||
        in(p, flat(s.coeff_base_bob), 3 * 3 * 4) ||
        in(p, flat(s.coeff_base_idtx), 3 * 7 * 5)) {
      return kFamCoeffBase;
    }
    if (in(p, flat(s.coeff_br), 7 * 5) ||
        in(p, flat(s.coeff_br_lf), 14 * 5) ||
        in(p, flat(s.coeff_br_uv), 4 * 5) ||
        in(p, flat(s.coeff_br_idtx), 3 * 7 * 5)) {
      return kFamCoeffBr;
    }
    if (in(p, flat(s.dc_sign), 2 * 2 * 3 * 3) ||
        in(p, flat(s.idtx_sign), 3 * 9 * 3)) {
      return kFamSign;
    }
    return kFamOther;
  }

  RateBreakdown* out_;
  const qctx::CoeffCdfSlice* slice_;
  bool coeff_phase_ = false;
};

}  // namespace detail

// Exact §8.3 family split for one block.  Returns false when the block is
// outside the model (1-D transform class, unsupported geometry).
inline bool block_rate(const BlockPlan& block, bool is_inter, int qctx_slice,
                       const qctx::CoeffCdfSlice* cdf, RateBreakdown* out) {
  if (out == nullptr || block.levels_scan == nullptr) return false;
  const auto params_opt = txv::make_tx_params(
      static_cast<txv::TxSize>(block.tx_size),
      static_cast<txv::TxType>(block.tx_type));
  if (!params_opt.has_value()) return false;
  const txv::TxParams& params = *params_opt;
  if (txpart::tx_class_of(block.tx_type) != txpart::TxScanClass::k2D) {
    return false;
  }
  const int scan_area = txpart::tx_scan_area(block.tx_size);
  if (block.area != scan_area || block.area > params.w * params.h) return false;
  const qctx::CoeffCdfSlice& slice =
      cdf != nullptr ? *cdf : qctx::tables_for_slice(qctx_slice);

  // `av2::Model::init` takes the block in §5.20.7.30 *scan* order and walks it
  // with its own scan: `levels_scan[c]` must be scan position c and
  // `raster[scan[c]]` the same coefficient.  Passing a raster array here (as
  // the lane-1 engine did) re-labels every coefficient with the raster index
  // of the scan position, which mismatches every base/BR context and the EOB
  // it derives; that was the whole 21-30 % overprice on eob/base/br.
  const std::span<const std::int32_t> scan_levels(
      block.levels_scan, static_cast<std::size_t>(block.area));

  av2::BlockInfo info;
  info.plane = block.plane;
  info.params = params;
  info.txb_skip_context = block.txb_skip_context;
  info.dc_sign_context = block.dc_sign_context;
  info.qctx = std::clamp(qctx_slice, 0, 3);
  info.is_inter = is_inter;
  info.cdf_override = cdf;
  av2::Model model;
  std::string error;
  if (!model.init(info, scan_levels, &error)) return false;
  detail::FamilySink sink(out, slice);
  model.set_symbol_sink(&sink);
  model.scalar_bits();  // drives the sink; the split is in `out`
  return true;
}

// Whole plan: the sum of the per-block splits plus the non-coefficient syntax
// the caller already priced (TX type and TX partition symbols).  The extra
// type bits are charged to the TX-type family.  The result is the *raw*
// static-CDF model; `apply_calibration()` maps it onto the live §8.3 cost.
inline RateBreakdown plan_rate(const std::vector<BlockPlan>& blocks,
                               bool is_inter, int qctx_slice,
                               const qctx::CoeffCdfSlice* cdf,
                               double extra_type_bits = 0.0,
                               double extra_partition_bits = 0.0) {
  RateBreakdown rate;
  for (const BlockPlan& block : blocks) {
    RateBreakdown block_bits;
    if (block_rate(block, is_inter, qctx_slice, cdf, &block_bits)) {
      rate += block_bits;
    }
  }
  rate.add(kFamTxType, extra_type_bits);
  rate.add(kFamTxPartition, extra_partition_bits);
  return rate;
}

// ---------------------------------------------------------------------------
// Static-CDF calibration
// ---------------------------------------------------------------------------
// The §8.3 model in block_rate() is the *static* §9.3 slice with context 0,
// while the writer charges the *live adapted* rows with the real neighbours.
// The engine's shadow audit (AV2ENC_RD_ENGINE_AUDIT vs AV2ENC_BIT_CENSUS)
// measures the gap per family on the emitted key picture; these are the
// measured operating-point ratios at QP120 (fit on 120, validated at 119).
// The raw bits are exact: the coefficient families are the static model's
// number, and `coeff_mag` (raw escape/sign literals) and the DC `sign`
// symbols already match the writer, so their ratio is 1.
struct Calibration {
  double scale[kFamCount] = {};
};

inline bool calibration_enabled() {
  static const bool value = [] {
    // On by default: the correction is the measured static->live ratio, and
    // `AV2ENC_RD_ENGINE_CAL=0` is the raw-model ablation.
    const char* env = std::getenv("AV2ENC_RD_ENGINE_CAL");
    if (env == nullptr || *env == '\0') return true;
    return !parse_off(env);
  }();
  return value;
}

inline const Calibration& calibration() {
  static const Calibration value = [] {
    Calibration cal;
    for (double& scale : cal.scale) scale = 1.0;
    // Measured QP119 key picture, 240p, after the §5.20.7.30 scan-order fix
    // (docs/rd-rate.md §2, "ctx" column: live contexts with the §9.3 static
    // rows).  The frozen-context model is 1.264x/1.120x/1.086x/1.067x on
    // txb_skip/eob/coeff_base/coeff_br, so the inverse is the static-model
    // correction this driver needs.  The live-row model needs no correction
    // (1.0015x), but it requires the writer's adapted rows and neighbour
    // contexts; the driver's measured leaves only have the static slice.
    cal.scale[kFamTxbSkip] = env_positive("AV2ENC_RD_ENGINE_CAL_TXB", 0.791);
    cal.scale[kFamEob] = env_positive("AV2ENC_RD_ENGINE_CAL_EOB", 0.893);
    cal.scale[kFamCoeffBase] =
        env_positive("AV2ENC_RD_ENGINE_CAL_BASE", 0.921);
    cal.scale[kFamCoeffBr] = env_positive("AV2ENC_RD_ENGINE_CAL_BR", 0.937);
    cal.scale[kFamCoeffMag] = env_positive("AV2ENC_RD_ENGINE_CAL_MAG", 1.000);
    cal.scale[kFamSign] = env_positive("AV2ENC_RD_ENGINE_CAL_SIGN", 1.004);
    return cal;
  }();
  return value;
}

inline void apply_calibration(RateBreakdown* rate) {
  if (rate == nullptr || !calibration_enabled()) return;
  const Calibration& cal = calibration();
  for (int family = 0; family < kFamCount; ++family) {
    rate->bits[family] *= cal.scale[family];
  }
}

// ---------------------------------------------------------------------------
// D_perc
// ---------------------------------------------------------------------------

// Perceptual distortion of one block extent.  `weights` is the frame Q8 map
// (eff_perc::FrameWeights); null or invalid selects neutral weights.  The
// CSF gains are the caller's site values (`eff_perc::key_csf_gain()` on the
// key picture).  `rows` clips the hidden bottom rows exactly like
// sse_against_source().
inline double distortion(const eff_perc::FrameWeights& weights, int x, int y,
                         int w, int h, const std::uint8_t* src, int src_stride,
                         int sx, int sy, const std::uint8_t* rec, int rec_stride,
                         int rx, int ry, int rows, double csf_gain,
                         double csf2_gain) {
  std::vector<std::uint16_t> leaf;
  const std::uint16_t* pix = nullptr;
  if (weights.valid()) {
    eff_perc::fill_leaf_weights(weights, x, y, w, h, &leaf);
    pix = leaf.data();
  }
  return eff_perc::weighted_sse(pix, w, h, src, src_stride, sx, sy, rec,
                                rec_stride, rx, ry, rows, csf_gain, csf2_gain);
}

// ---------------------------------------------------------------------------
// Lambda
// ---------------------------------------------------------------------------
enum class Site : int {
  kKeyIntra = 0,
  kInter,
  kChroma,
};

// The operating point is 240p / 100 kbps, QP ~119/120.  The existing sites
// already carry a QP-fitted base lambda (key_tx_rd_lambda / leaf_rd_lambda);
// the engine applies one calibrated scale per site on top.  The scales are
// measured at that operating point (docs/rd-lane1.md); the default is the
// measured best for the key-intra site, and 1.0 for the sites this milestone
// does not wire, so enabling the gate never silently re-tunes them.
inline double site_scale(Site site) {
  switch (site) {
    case Site::kKeyIntra:
      return env_positive("AV2ENC_RD_ENGINE_KEY", 1.0);
    case Site::kInter:
      return env_positive("AV2ENC_RD_ENGINE_INTER", 1.0);
    case Site::kChroma:
      return env_positive("AV2ENC_RD_ENGINE_CHROMA", 1.0);
  }
  return 1.0;
}

// The distortion the engine charges at a site.  `AV2ENC_RD_ENGINE_CSF=0` is
// the plain-SSE ablation (the eff-perc CSF gains are the site's D, not a
// separate tool), so a negative D-only result can be separated from the
// corrected-R mechanism.
inline bool csf_enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_RD_ENGINE_CSF");
    if (env == nullptr || *env == '\0') return true;
    return !parse_off(env);
  }();
  return value;
}

// ---------------------------------------------------------------------------
// J
// ---------------------------------------------------------------------------
struct RdCost {
  double distortion = 0.0;
  double lambda = 0.0;
  double rate = 0.0;
  double j = 0.0;
  bool valid() const { return std::isfinite(j); }
};

inline RdCost cost(double d_perc, double lambda, double r_exact) {
  RdCost value;
  value.distortion = d_perc;
  value.lambda = lambda;
  value.rate = r_exact;
  value.j = d_perc + lambda * r_exact;
  return value;
}

inline RdCost cost(double d_perc, double lambda, const RateBreakdown& rate) {
  return cost(d_perc, lambda, rate.total());
}

// ---------------------------------------------------------------------------
// Candidate helper
// ---------------------------------------------------------------------------
// One evaluated candidate.  A cheap screen may fill `distortion` and `rate`
// with a proxy, but `exact == false` candidates cannot win: only a candidate
// whose real transform was priced can be selected.
struct Candidate {
  double distortion = 0.0;
  RateBreakdown rate;
  bool exact = false;
  int index = 0;
};

inline double candidate_j(const Candidate& candidate, double lambda) {
  return cost(candidate.distortion, lambda, candidate.rate).j;
}

// Returns the index of the cheapest exact candidate, or -1 when none has an
// exact cost.  Strict `<` keeps the first (lowest enumeration order) on ties,
// matching the historical tie-break.
inline int best_candidate(const std::vector<Candidate>& candidates,
                          double lambda) {
  int best = -1;
  double best_j = 0.0;
  for (const Candidate& candidate : candidates) {
    if (!candidate.exact) continue;
    const double j = candidate_j(candidate, lambda);
    if (best < 0 || j < best_j) {
      best = candidate.index;
      best_j = j;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Gate + shadow audit
// ---------------------------------------------------------------------------
struct Counters {
  std::atomic<unsigned long long> key_site_calls{0};
  std::atomic<unsigned long long> inter_site_calls{0};
  std::atomic<unsigned long long> chroma_site_calls{0};
  std::atomic<unsigned long long> blocks_priced{0};
  std::atomic<unsigned long long> blocks_fallback{0};
  std::atomic<unsigned long long> audit_blocks{0};
};

inline Counters& counters() {
  // Leaked on purpose: print_stats() runs from atexit, i.e. after function
  // statics with destructors would already have been torn down.
  static Counters* value = new Counters();
  return *value;
}

struct Audit {
  std::mutex mutex;
  double bits[4][kFamCount] = {};  // site 0 key, 1 inter, 2 chroma, 3 all
  unsigned long long blocks = 0;
  bool active = false;
  bool written = false;
  std::string path;
};

inline Audit& audit() {
  // Leaked on purpose; see counters().
  static Audit* value = new Audit();
  return *value;
}

inline bool audit_active() {
  static const bool value = [] {
    const char* path = std::getenv("AV2ENC_RD_ENGINE_AUDIT");
    if (parse_off(path)) return false;
    Audit& state = audit();
    state.active = true;
    state.path = path;
    return true;
  }();
  return value;
}

inline int audit_site_index(Site site) {
  switch (site) {
    case Site::kKeyIntra: return 0;
    case Site::kInter: return 1;
    case Site::kChroma: return 2;
  }
  return 3;
}

// Accumulate one emitted block's family split.  Called only from the audit
// path, i.e. after a real decision has been made for that block.
inline void audit_block(Site site, int plane, int tx_size, bool is_inter,
                        const RateBreakdown& rate) {
  if (!audit_active()) return;
  Audit& state = audit();
  const int index = audit_site_index(site);
  std::lock_guard<std::mutex> lock(state.mutex);
  for (int family = 0; family < kFamCount; ++family) {
    state.bits[index][family] += rate.get(family);
    state.bits[3][family] += rate.get(family);
  }
  ++state.blocks;
  (void)plane;
  (void)tx_size;
  (void)is_inter;
  ++counters().audit_blocks;
}

inline bool enabled();

inline void write_audit_totals() {
  Audit& state = audit();
  if (!state.active || state.written) return;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.written) return;
  state.written = true;
  std::FILE* file = std::fopen(state.path.c_str(), "a");
  if (file == nullptr) return;
  std::fprintf(file,
               "{\"kind\":\"rd_engine_audit_totals\",\"blocks\":%llu,"
               "\"sites\":{",
               state.blocks);
  bool first_site = true;
  for (int index = 0; index < 4; ++index) {
    double total = 0.0;
    for (double value : state.bits[index]) total += value;
    if (total == 0.0) continue;
    std::fprintf(file, "%s\"%s\":{", first_site ? "" : ",",
                 index == 0 ? "key"
                            : (index == 1 ? "inter"
                                          : (index == 2 ? "chroma" : "all")));
    first_site = false;
    bool first = true;
    for (int family = 0; family < kFamCount; ++family) {
      if (state.bits[index][family] == 0.0) continue;
      std::fprintf(file, "%s\"%s\":%.4f", first ? "" : ",",
                   family_name(family), state.bits[index][family]);
      first = false;
    }
    std::fprintf(file, "}");
  }
  std::fprintf(file, "}}\n");
  std::fclose(file);
}

inline void print_stats() {
  if (!enabled()) return;
  const Counters& c = counters();
  std::fprintf(stderr,
               "rd-engine: key=%llu inter=%llu chroma=%llu blocks=%llu "
               "fallback=%llu audit_blocks=%llu key_scale=%.3f "
               "inter_scale=%.3f chroma_scale=%.3f\n",
               static_cast<unsigned long long>(c.key_site_calls.load()),
               static_cast<unsigned long long>(c.inter_site_calls.load()),
               static_cast<unsigned long long>(c.chroma_site_calls.load()),
               static_cast<unsigned long long>(c.blocks_priced.load()),
               static_cast<unsigned long long>(c.blocks_fallback.load()),
               static_cast<unsigned long long>(c.audit_blocks.load()),
               site_scale(Site::kKeyIntra), site_scale(Site::kInter),
               site_scale(Site::kChroma));
  if (audit_active()) {
    Audit& state = audit();
    std::fprintf(stderr, "rd-engine-audit: blocks=%llu\n", state.blocks);
    for (int index = 0; index < 4; ++index) {
      double total = 0.0;
      for (double value : state.bits[index]) total += value;
      if (total == 0.0) continue;
      std::fprintf(stderr, "rd-engine-audit: site=%s total=%.1f",
                   index == 0 ? "key"
                              : (index == 1 ? "inter"
                                            : (index == 2 ? "chroma" : "all")),
                   total);
      for (int family = 0; family < kFamCount; ++family) {
        if (state.bits[index][family] == 0.0) continue;
        std::fprintf(stderr, " %s=%.1f", family_name(family),
                     state.bits[index][family]);
      }
      std::fprintf(stderr, "\n");
    }
    write_audit_totals();
  }
}

inline bool enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_RD_ENGINE");
    return !parse_off(env);
  }();
  static const bool registered = [] {
    if (value) std::atexit(&print_stats);
    return true;
  }();
  (void)registered;
  return value;
}

}  // namespace av2enc::video::rd_engine
