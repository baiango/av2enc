// Host-emulation parity test for the CUDA GME device core.
//
// The same `gme_device.cuh` that the CUDA kernels include is compiled here by
// the host compiler and compared against the CPU reference in
// `src/encoder/inter/gme/gme.cpp`.  No GPU and no nvcc are needed: `gme_plan.cpp`
// and `av2enc::cuda::Arena` are linked against a tiny host CUDA runtime stub.
//
// Coverage:
//   * the portable 128-bit integer against the host __int128 arithmetic used
//     by the CPU reference (add/sub/mul/shift/div/ratio);
//   * §7.12.2.2 projection, full MV range and every block size 4x4..64x64,
//     all four precisions, GLOBALMV usability/warp selection;
//   * §7.13.3.21 shear validity;
//   * §5.18.9 quantisation, codability, validation and degradation;
//   * fits from correspondences, block MVs and warps: translation, ROTZOOM and
//     AFFINE, with noise, weights, outlier passes and all options;
//   * determinism under reversed input order;
//   * the bilinear quality proxy against the CPU helper;
//   * synthetic pan/zoom/rotate frames through a real block matcher;
//   * a real 720p pair when testdata/shine_1280x720_60f.y4m is present;
//   * required_device_bytes / allocate_checked / mark / rewind /
//     kArenaTooSmall / stage / download.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cuda/codec/inter/gme/gme.h"
#include "codec/inter/gme/gme.h"

namespace dev = av2enc::cuda::gme;
namespace cpu = av2enc::inter::gme;

namespace {

int g_checks = 0;
int g_failures = 0;

void Check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 40) std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

void CheckEqI64(long long actual, long long expected, const char* what) {
  ++g_checks;
  if (actual != expected) {
    ++g_failures;
    if (g_failures <= 40) {
      std::fprintf(stderr, "FAIL: %s (actual %lld, expected %lld)\n", what,
                   actual, expected);
    }
  }
}

void CheckClose(double actual, double expected, double tolerance,
                const char* what) {
  ++g_checks;
  const double scale = std::max(1.0, std::fabs(expected));
  if (!(std::fabs(actual - expected) <= tolerance * scale)) {
    ++g_failures;
    if (g_failures <= 40) {
      std::fprintf(stderr, "FAIL: %s (actual %.12g, expected %.12g)\n", what,
                   actual, expected);
    }
  }
}

// ---------------------------------------------------------------------------
// CPU <-> device POD conversion.
// ---------------------------------------------------------------------------

dev::GmeParams to_dev_params(const cpu::GmParams& p) {
  dev::GmeParams out;
  out.type = static_cast<dev::GmeI32>(p.type);
  for (int i = 0; i < 6; ++i) out.p[i] = p.p[static_cast<std::size_t>(i)];
  return out;
}

dev::GmePoint to_dev_point(const cpu::GmPoint& p) {
  dev::GmePoint out;
  out.src_x = p.src_x;
  out.src_y = p.src_y;
  out.dst_x = p.dst_x;
  out.dst_y = p.dst_y;
  out.weight = p.weight;
  return out;
}

dev::GmeBlockMv to_dev_block(const cpu::GmBlockMv& b) {
  dev::GmeBlockMv out;
  out.x = b.x;
  out.y = b.y;
  out.mv_x = b.mv_x;
  out.mv_y = b.mv_y;
  out.weight = b.weight;
  return out;
}

dev::GmeWarpSample to_dev_warp(const cpu::GmWarpSample& w) {
  dev::GmeWarpSample out;
  out.x = w.x;
  out.y = w.y;
  out.radius = w.radius;
  out.model = to_dev_params(w.model);
  out.weight = w.weight;
  return out;
}

dev::GmeOptions to_dev_options(const cpu::GmOptions& o) {
  dev::GmeOptions d;
  d.model = static_cast<dev::GmeI32>(o.model);
  d.min_points = o.min_points;
  d.max_points = o.max_points;
  d.outlier_passes = o.outlier_passes;
  d.outlier_threshold_q8 = o.outlier_threshold_q8;
  d.allow_rotzoom_degrade = o.allow_rotzoom_degrade ? 1 : 0;
  d.require_shear_valid = o.require_shear_valid ? 1 : 0;
  d.rotzoom_mismatch_q16 = o.rotzoom_mismatch_q16;
  d.min_scale_q16 = o.min_scale_q16;
  d.max_scale_q16 = o.max_scale_q16;
  d.max_corner_displacement_q8 = o.max_corner_displacement_q8;
  d.force_integer_mv = o.force_integer_mv ? 1 : 0;
  d.quantize = o.quantize ? 1 : 0;
  d.min_gain_percent = o.min_gain_percent;
  d.frame_width = o.frame_width;
  d.frame_height = o.frame_height;
  return d;
}

int reason_code(const std::string& reason) {
  if (reason.empty()) return dev::kGmeReasonNone;
  if (reason == "no points") return dev::kGmeReasonNoPoints;
  if (reason == "too few weighted points") return dev::kGmeReasonTooFewWeighted;
  if (reason == "degenerate point distribution") {
    return dev::kGmeReasonDegenerate;
  }
  if (reason == "collinear points (det <= 0)") return dev::kGmeReasonCollinear;
  if (reason == "too few correspondences") {
    return dev::kGmeReasonTooFewCorrespondences;
  }
  if (reason == "too many correspondences") {
    return dev::kGmeReasonTooManyCorrespondences;
  }
  if (reason == "block MV correspondence exceeds int32") {
    return dev::kGmeReasonBlockMvOverflow;
  }
  if (reason == "warp sample coordinate exceeds int32") {
    return dev::kGmeReasonWarpOverflow;
  }
  if (reason == "IDENTITY with non-default parameters") {
    return dev::kGmeReasonIdentityNonDefault;
  }
  if (reason == "translation not on the 1/8-pel grid") {
    return dev::kGmeReasonTransGrid;
  }
  if (reason == "translation outside GM_TRANS range") {
    return dev::kGmeReasonTransRange;
  }
  if (reason == "alpha not on the 1/1024 grid") {
    return dev::kGmeReasonAlphaGrid;
  }
  if (reason == "alpha outside GM_ALPHA range") {
    return dev::kGmeReasonAlphaRange;
  }
  if (reason == "ROTZOOM requires p4 = -p3 and p5 = p2") {
    return dev::kGmeReasonRotzoomSymmetry;
  }
  if (reason == "unknown GmType") return dev::kGmeReasonUnknownType;
  if (reason == "model scale/determinant out of range") {
    return dev::kGmeReasonScaleOutOfRange;
  }
  if (reason == "shear model is not valid") return dev::kGmeReasonShearInvalid;
  if (reason == "corner projection out of range") {
    return dev::kGmeReasonCornerOutOfRange;
  }
  if (reason == "affine close to similarity; signalled ROTZOOM") {
    return dev::kGmeReasonAffineToRotzoom;
  }
  if (reason == "degraded to ROTZOOM after AFFINE rejection") {
    return dev::kGmeReasonDegradedAfterReject;
  }
  if (reason == "model rejected") return dev::kGmeReasonModelRejected;
  if (reason.rfind("global motion gain below threshold", 0) == 0) {
    return dev::kGmeReasonGainBelowThreshold;
  }
  return -1;
}

void compare_estimate(const cpu::GmEstimate& reference,
                      const dev::GmeEstimate& actual, const char* label) {
  ++g_checks;
  if (reference.params.type !=
      static_cast<cpu::GmType>(actual.params.type)) {
    ++g_failures;
    if (g_failures <= 40) {
      std::fprintf(stderr, "FAIL: %s type %d != %d\n", label,
                   static_cast<int>(reference.params.type),
                   actual.params.type);
    }
  }
  for (int i = 0; i < 6; ++i) {
    CheckEqI64(actual.params.p[i],
               reference.params.p[static_cast<std::size_t>(i)], label);
  }
  CheckEqI64(actual.diag.ok, reference.diag.ok ? 1 : 0, label);
  CheckEqI64(actual.diag.accepted, reference.diag.accepted ? 1 : 0, label);
  CheckEqI64(actual.diag.codable, reference.diag.codable ? 1 : 0, label);
  CheckEqI64(actual.diag.shear_valid, reference.diag.shear_valid ? 1 : 0,
             label);
  CheckEqI64(actual.diag.degraded, reference.diag.degraded ? 1 : 0, label);
  CheckEqI64(actual.diag.identity, reference.diag.identity ? 1 : 0, label);
  CheckEqI64(actual.diag.points_in, reference.diag.points_in, label);
  CheckEqI64(actual.diag.points_used, reference.diag.points_used, label);
  CheckEqI64(actual.diag.outlier_passes_run,
             reference.diag.outlier_passes_run, label);
  CheckEqI64(actual.diag.identity_sse_q8, reference.diag.identity_sse_q8,
             label);
  CheckEqI64(actual.diag.model_sse_q8, reference.diag.model_sse_q8, label);
  CheckEqI64(actual.diag.candidate_model_sse_q8,
             reference.diag.candidate_model_sse_q8, label);
  CheckEqI64(actual.diag.shear_mismatch_q16,
             reference.diag.shear_mismatch_q16, label);
  CheckClose(actual.diag.gain_db, reference.diag.gain_db, 1e-6, label);
  CheckClose(actual.diag.model_rmse_q8, reference.diag.model_rmse_q8, 1e-6,
             label);
  CheckClose(actual.diag.candidate_gain_db, reference.diag.candidate_gain_db,
             1e-6, label);
  const int expected_reason = reason_code(reference.diag.reason);
  CheckEqI64(actual.diag.reason, expected_reason, label);
}

// ---------------------------------------------------------------------------
// 128-bit self-test against the host compiler's __int128.
// ---------------------------------------------------------------------------

std::uint64_t g_rng = 0x9e3779b97f4a7c15ULL;

std::uint64_t Next() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 7;
  g_rng ^= g_rng << 17;
  return g_rng;
}

__int128 ToRef(dev::GmeI128 value) {
  const unsigned __int128 bits =
      (static_cast<unsigned __int128>(static_cast<std::uint64_t>(value.hi))
       << 64) |
      value.lo;
  __int128 result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

dev::GmeI128 RandomI128(int high_bits) {
  std::uint64_t hi = Next();
  if (high_bits < 64) hi &= ((1ULL << high_bits) - 1ULL);
  if ((Next() & 1ULL) != 0) hi |= ~0ULL << high_bits;
  else if (high_bits < 64) hi &= ~(~0ULL << high_bits);
  const std::uint64_t lo = Next();
  dev::GmeI128 result;
  result.lo = lo;
  result.hi = static_cast<std::int64_t>(hi);
  return result;
}

std::int64_t RefDivRound(__int128 numerator, __int128 denominator) {
  if (denominator <= 0) return 0;
  const bool negative = numerator < 0;
  const unsigned __int128 a =
      negative ? static_cast<unsigned __int128>(-numerator)
               : static_cast<unsigned __int128>(numerator);
  const unsigned __int128 d = static_cast<unsigned __int128>(denominator);
  unsigned __int128 q = a / d;
  const unsigned __int128 r = a % d;
  if (r * 2 >= d) ++q;
  const unsigned __int128 imax =
      static_cast<unsigned __int128>(std::numeric_limits<std::int64_t>::max());
  if (q > imax) {
    return negative ? std::numeric_limits<std::int64_t>::min()
                    : std::numeric_limits<std::int64_t>::max();
  }
  const std::int64_t result = static_cast<std::int64_t>(q);
  return negative ? -result : result;
}

std::int64_t RefRatioQ16(__int128 numerator, __int128 denominator) {
  if (denominator == 0) return 0;
  const bool negative = (numerator < 0) != (denominator < 0);
  const unsigned __int128 n =
      numerator < 0 ? static_cast<unsigned __int128>(-numerator)
                    : static_cast<unsigned __int128>(numerator);
  const unsigned __int128 d =
      denominator < 0 ? static_cast<unsigned __int128>(-denominator)
                      : static_cast<unsigned __int128>(denominator);
  unsigned __int128 integer = n / d;
  unsigned __int128 remainder = n % d;
  std::uint64_t frac = 0;
  for (int bit = 0; bit < 16; ++bit) {
    remainder <<= 1;
    frac <<= 1;
    if (remainder >= d) {
      remainder -= d;
      frac |= 1U;
    }
  }
  remainder <<= 1;
  if (remainder >= d) ++frac;
  if (frac == (1U << 16)) {
    frac = 0;
    ++integer;
  }
  const std::int64_t i32_max = 2147483647LL;
  if (integer > static_cast<unsigned __int128>(i32_max >> 16)) {
    return negative ? -2147483647LL - 1LL : i32_max;
  }
  std::int64_t result = static_cast<std::int64_t>(integer) * 65536 +
                        static_cast<std::int64_t>(frac);
  if (negative) result = -result;
  if (result < -2147483647LL - 1LL) result = -2147483647LL - 1LL;
  if (result > i32_max) result = i32_max;
  return result;
}

void TestI128() {
  for (int iteration = 0; iteration < 4000; ++iteration) {
    const dev::GmeI128 a = RandomI128(40);
    const dev::GmeI128 b = RandomI128(40);
    const __int128 ra = ToRef(a);
    const __int128 rb = ToRef(b);

    Check(ToRef(dev::gme_i128_add(a, b)) == ra + rb, "i128 add");
    Check(ToRef(dev::gme_i128_sub(a, b)) == ra - rb, "i128 sub");
    Check(ToRef(dev::gme_i128_neg(a)) == -ra, "i128 neg");
    Check(ToRef(dev::gme_i128_mul(a, b)) ==
              static_cast<__int128>(static_cast<unsigned __int128>(ra) *
                                     static_cast<unsigned __int128>(rb)),
          "i128 mul");
    const int cmp = dev::gme_i128_cmp(a, b);
    Check((cmp < 0) == (ra < rb) && (cmp > 0) == (ra > rb) &&
              (cmp == 0) == (ra == rb),
          "i128 cmp");

    const int shift = static_cast<int>(Next() % 128U);
    Check(ToRef(dev::gme_i128_shr(a, shift)) == (ra >> shift), "i128 shr");
    Check(ToRef(dev::gme_i128_shl(a, shift)) ==
              static_cast<__int128>(static_cast<unsigned __int128>(ra)
                                    << shift),
          "i128 shl");

    const std::int64_t sat = dev::gme_i128_to_i64_sat(a);
    const __int128 imax = std::numeric_limits<std::int64_t>::max();
    const __int128 imin = std::numeric_limits<std::int64_t>::min();
    const std::int64_t expected_sat =
        static_cast<std::int64_t>(ra > imax ? imax : (ra < imin ? imin : ra));
    CheckEqI64(sat, expected_sat, "i128 sat i64");
  }

  for (int iteration = 0; iteration < 4000; ++iteration) {
    const dev::GmeI128 a = RandomI128(63);
    dev::GmeI128 b = RandomI128(63);
    if (b.hi == 0 && b.lo == 0) b.lo = 1;
    const __int128 ra = ToRef(a);
    const __int128 rb = ToRef(b);
    CheckEqI64(dev::gme_i128_div_round(a, b), RefDivRound(ra, rb),
               "i128 div round");
    CheckEqI64(dev::gme_i128_ratio_q16(a, b), RefRatioQ16(ra, rb),
               "i128 ratio q16");
  }

  // Targeted 128/128 division with the denominator's bit 127 set: the
  // 129-bit remainder carry is what a naive shift-subtract gets wrong.
  for (int iteration = 0; iteration < 500; ++iteration) {
    dev::GmeU128 n;
    n.hi = Next();
    n.lo = Next();
    dev::GmeU128 d;
    d.hi = (Next() | (1ULL << 63));
    d.lo = Next();
    if (d.hi == 0 && d.lo == 0) d.lo = 1;
    const unsigned __int128 rn =
        (static_cast<unsigned __int128>(n.hi) << 64) | n.lo;
    const unsigned __int128 rd =
        (static_cast<unsigned __int128>(d.hi) << 64) | d.lo;
    const dev::GmeUDiv qr = dev::gme_u128_divmod(n, d);
    const unsigned __int128 rq =
        (static_cast<unsigned __int128>(qr.q.hi) << 64) | qr.q.lo;
    const unsigned __int128 rr =
        (static_cast<unsigned __int128>(qr.r.hi) << 64) | qr.r.lo;
    Check(rq == rn / rd, "u128 high-bit divmod quotient");
    Check(rr == rn % rd, "u128 high-bit divmod remainder");
  }

  // Unsigned 128/128 division against the compiler's unsigned division.
  for (int iteration = 0; iteration < 2000; ++iteration) {
    dev::GmeU128 n;
    n.hi = Next();
    n.lo = Next();
    dev::GmeU128 d;
    d.hi = Next();
    d.lo = Next();
    if (d.hi == 0 && d.lo == 0) d.lo = 1;
    const unsigned __int128 rn =
        (static_cast<unsigned __int128>(n.hi) << 64) | n.lo;
    const unsigned __int128 rd =
        (static_cast<unsigned __int128>(d.hi) << 64) | d.lo;
    const dev::GmeUDiv qr = dev::gme_u128_divmod(n, d);
    const unsigned __int128 rq =
        (static_cast<unsigned __int128>(qr.q.hi) << 64) | qr.q.lo;
    const unsigned __int128 rr =
        (static_cast<unsigned __int128>(qr.r.hi) << 64) | qr.r.lo;
    Check(rq == rn / rd, "u128 divmod quotient");
    Check(rr == rn % rd, "u128 divmod remainder");
  }
}

// ---------------------------------------------------------------------------
// Projection, shear and finalisation.
// ---------------------------------------------------------------------------

cpu::GmParams Similarity(double scale, double angle_deg, double tx,
                         double ty) {
  const double radians = angle_deg * 3.14159265358979323846 / 180.0;
  const double c = std::cos(radians) * scale;
  const double s = std::sin(radians) * scale;
  cpu::GmParams params;
  params.type = cpu::GmType::kRotZoom;
  params.p[0] = static_cast<std::int32_t>(std::llround(tx * 65536.0));
  params.p[1] = static_cast<std::int32_t>(std::llround(ty * 65536.0));
  params.p[2] = static_cast<std::int32_t>(std::llround(c * 65536.0));
  params.p[3] = static_cast<std::int32_t>(std::llround(-s * 65536.0));
  params.p[4] = static_cast<std::int32_t>(std::llround(s * 65536.0));
  params.p[5] = static_cast<std::int32_t>(std::llround(c * 65536.0));
  return params;
}

cpu::GmParams Affine(double p2, double p3, double p4, double p5, double tx,
                     double ty) {
  cpu::GmParams params;
  params.type = cpu::GmType::kAffine;
  params.p[0] = static_cast<std::int32_t>(std::llround(tx * 65536.0));
  params.p[1] = static_cast<std::int32_t>(std::llround(ty * 65536.0));
  params.p[2] = static_cast<std::int32_t>(std::llround(p2 * 65536.0));
  params.p[3] = static_cast<std::int32_t>(std::llround(p3 * 65536.0));
  params.p[4] = static_cast<std::int32_t>(std::llround(p4 * 65536.0));
  params.p[5] = static_cast<std::int32_t>(std::llround(p5 * 65536.0));
  return params;
}

void TestProjection() {
  std::vector<cpu::GmParams> models;
  models.push_back(cpu::make_identity_gm());
  models.push_back(Similarity(1.0, 0.0, 9.5, -5.25));
  models.push_back(Similarity(1.0, 0.0, -2047.875, 2047.875));
  models.push_back(Similarity(1.06, 0.0, 0.0, 0.0));
  models.push_back(Similarity(0.33, 17.0, 4.0, -3.0));
  models.push_back(Similarity(3.0, -31.0, -100.0, 200.0));
  models.push_back(Affine(1.04, 0.02, -0.03, 0.98, 4.0, -3.0));
  models.push_back(Affine(0.2, 0.4, 0.1, 1.4, 1000.0, -1000.0));
  models.push_back(Affine(1.0, 0.0, 0.0, -1.0, 0.0, 0.0));  // reflection
  models.push_back(Affine(1.0, 0.0, 0.0, 1.0, 0.0, 0.0));

  int count = 0;
  for (const cpu::GmParams& model : models) {
    const dev::GmeParams device = to_dev_params(model);
    for (int mi_cols = 1; mi_cols <= 40; mi_cols += 13) {
      for (int mi_rows = 1; mi_rows <= 40; mi_rows += 17) {
        for (int bw4 = 1; bw4 <= 16; ++bw4) {
          for (int bh4 = 1; bh4 <= 16; bh4 += 3) {
            for (int mi_col = 0; mi_col < mi_cols; mi_col += 7) {
              for (int mi_row = 0; mi_row < mi_rows; mi_row += 9) {
                for (int precision = 4; precision <= 7; ++precision) {
                  int cpu_mv[2] = {0, 0};
                  int dev_mv[2] = {0, 0};
                  cpu::get_warp_motion_vector(
                      model, mi_col, mi_row, bw4, bh4, mi_cols, mi_rows,
                      static_cast<cpu::MvPrecision>(precision), cpu_mv);
                  dev::gme_get_warp_motion_vector(
                      device, mi_col, mi_row, bw4, bh4, mi_cols, mi_rows,
                      static_cast<dev::GmeMvPrecision>(precision), dev_mv);
                  CheckEqI64(dev_mv[0], cpu_mv[0], "projection mv_y");
                  CheckEqI64(dev_mv[1], cpu_mv[1], "projection mv_x");
                  ++count;
                }
              }
            }
          }
        }
      }
    }
    // GLOBALMV helper.
    for (int bw4 = 1; bw4 <= 16; bw4 += 5) {
      for (int bh4 = 1; bh4 <= 16; bh4 += 7) {
        const cpu::GmGlobalMvUse cpu_use =
            cpu::global_mv_for_block(model, 2, 3, 5, bw4, bh4, 40, 30);
        const dev::GmeGlobalMvUse dev_use =
            dev::gme_global_mv_for_block(device, 2, 3, 5, bw4, bh4, 40, 30);
        CheckEqI64(dev_use.usable, cpu_use.usable ? 1 : 0, "global usable");
        CheckEqI64(dev_use.use_warp, cpu_use.use_warp ? 1 : 0, "global warp");
        CheckEqI64(dev_use.ref, cpu_use.ref, "global ref");
        CheckEqI64(dev_use.mv[0], cpu_use.mv[0], "global mv_y");
        CheckEqI64(dev_use.mv[1], cpu_use.mv[1], "global mv_x");
      }
    }
    // project_gm_point.
    for (int i = 0; i < 200; ++i) {
      const std::int32_t x = static_cast<std::int32_t>(Next() % 4000001ULL) -
                             2000000;
      const std::int32_t y = static_cast<std::int32_t>(Next() % 4000001ULL) -
                             2000000;
      std::int32_t cpu_x = 0;
      std::int32_t cpu_y = 0;
      std::int32_t dev_x = 0;
      std::int32_t dev_y = 0;
      cpu::project_gm_point(model, x, y, &cpu_x, &cpu_y);
      dev::gme_project_gm_point(device, x, y, &dev_x, &dev_y);
      CheckEqI64(dev_x, cpu_x, "project_gm_point x");
      CheckEqI64(dev_y, cpu_y, "project_gm_point y");
    }
  }
  // Full MV range: a raw (unquantised) model far outside the syntax grid and a
  // huge MI grid make the MV_LOW/MV_UPP clamp, not the border clamp, bind.
  {
    cpu::GmParams extreme = cpu::make_identity_gm();
    extreme.type = cpu::GmType::kRotZoom;
    extreme.p[0] = 2000000000;
    extreme.p[1] = -2000000000;
    extreme.p[2] = 200000;
    extreme.p[3] = -50000;
    extreme.p[4] = 50000;
    extreme.p[5] = 200000;
    const dev::GmeParams device = to_dev_params(extreme);
    for (int bw4 = 1; bw4 <= 16; ++bw4) {
      for (int bh4 = 1; bh4 <= 16; ++bh4) {
        for (int precision = 4; precision <= 7; ++precision) {
          int cpu_mv[2] = {0, 0};
          int dev_mv[2] = {0, 0};
          cpu::get_warp_motion_vector(
              extreme, 500000, 500000, bw4, bh4, 1000001, 1000001,
              static_cast<cpu::MvPrecision>(precision), cpu_mv);
          dev::gme_get_warp_motion_vector(
              device, 500000, 500000, bw4, bh4, 1000001, 1000001,
              static_cast<dev::GmeMvPrecision>(precision), dev_mv);
          CheckEqI64(dev_mv[0], cpu_mv[0], "full-range mv_y");
          CheckEqI64(dev_mv[1], cpu_mv[1], "full-range mv_x");
          cpu::get_warp_motion_vector(
              extreme, 0, 0, bw4, bh4, 1000001, 1000001,
              static_cast<cpu::MvPrecision>(precision), cpu_mv);
          dev::gme_get_warp_motion_vector(
              device, 0, 0, bw4, bh4, 1000001, 1000001,
              static_cast<dev::GmeMvPrecision>(precision), dev_mv);
          CheckEqI64(dev_mv[0], cpu_mv[0], "full-range border mv_y");
          CheckEqI64(dev_mv[1], cpu_mv[1], "full-range border mv_x");
        }
      }
    }
  }
  std::printf("projection cases: %d\n", count);

  // Shear validity over random and grid-aligned models.
  for (int i = 0; i < 4000; ++i) {
    std::int32_t p[6];
    for (int j = 0; j < 6; ++j) {
      if ((Next() & 1ULL) != 0) {
        p[j] = static_cast<std::int32_t>(Next() % 131072ULL) - 65536;
      } else {
        p[j] = static_cast<std::int32_t>(Next() % 2000001ULL) - 1000000;
      }
    }
    if (i < 100) {
      p[2] = 65536;
      p[5] = 65536;
      p[3] = 0;
      p[4] = 0;
    }
    Check(dev::gme_setup_shear_valid(p) == cpu::setup_shear_valid(p),
          "shear validity");
  }
}

void TestFinalize() {
  cpu::GmOptions options;
  std::vector<cpu::GmParams> models;
  models.push_back(cpu::make_identity_gm());
  models.push_back(Similarity(1.02, 0.5, 3.0, -2.0));
  models.push_back(Similarity(1.0, 0.0, 0.25, -0.125));
  models.push_back(Affine(1.04, 0.02, -0.03, 0.98, 4.0, -3.0));
  models.push_back(Affine(1.0, 0.0, 0.0, 1.0, 0.0, 0.0));
  models.push_back(Affine(1.0, 0.0001, -0.0001, 1.0, 0.0, 0.0));
  models.push_back(Affine(0.2, 0.0, 0.0, 0.2, 0.0, 0.0));
  models.push_back(Affine(4.0, 0.0, 0.0, 4.0, 0.0, 0.0));
  models.push_back(Affine(1.0, 0.0, 0.0, -1.0, 0.0, 0.0));
  models.push_back(Affine(-1.0, 0.0, 0.0, -1.0, 0.0, 0.0));

  // Random raw models, including off-grid values.
  for (int i = 0; i < 3000; ++i) {
    cpu::GmParams raw;
    raw.type = static_cast<cpu::GmType>(Next() % 3ULL);
    for (int j = 0; j < 6; ++j) {
      if ((Next() & 3ULL) == 0) {
        raw.p[static_cast<std::size_t>(j)] =
            static_cast<std::int32_t>(Next() % 2000001ULL) - 1000000;
      } else if (j < 2) {
        raw.p[static_cast<std::size_t>(j)] =
            static_cast<std::int32_t>(Next() % 32769ULL) * 8192 - 16384 * 8192;
      } else {
        raw.p[static_cast<std::size_t>(j)] =
            static_cast<std::int32_t>(Next() % 1025ULL) * 64 - 512 * 64;
      }
    }
    models.push_back(raw);
  }

  int checked = 0;
  for (const cpu::GmParams& model : models) {
    for (int variant = 0; variant < 8; ++variant) {
      cpu::GmOptions local = options;
      local.quantize = (variant & 1) != 0;
      local.force_integer_mv = (variant & 2) != 0;
      local.allow_rotzoom_degrade = (variant & 4) != 0;
      local.require_shear_valid = (variant & 8) != 0;
      if ((variant & 1) == 0) local.frame_width = 640;
      if ((variant & 2) == 0) local.frame_height = 360;
      const dev::GmeOptions device_options = to_dev_options(local);
      const cpu::GmEstimate reference =
          cpu::finalize_gm_params(model, local);
      const dev::GmeEstimate actual =
          dev::gme_finalize_gm_params(to_dev_params(model), device_options);
      compare_estimate(reference, actual, "finalize");
      ++checked;
    }
  }
  std::printf("finalize cases: %d\n", checked);
}

// ---------------------------------------------------------------------------
// Fit parity.
// ---------------------------------------------------------------------------

void ProjectQ8(const cpu::GmParams& params, int x_q8, int y_q8, int* rx_q8,
               int* ry_q8) {
  std::int32_t rx = 0;
  std::int32_t ry = 0;
  cpu::project_gm_point(params, x_q8 << 13, y_q8 << 13, &rx, &ry);
  *rx_q8 = rx >> 13;
  *ry_q8 = ry >> 13;
}

std::vector<cpu::GmPoint> GridPoints(const cpu::GmParams& params, int width,
                                     int height, int step_px, int noise_seed,
                                     int noise_amplitude) {
  std::vector<cpu::GmPoint> points;
  std::uint32_t seed = static_cast<std::uint32_t>(noise_seed) * 2654435761U + 1U;
  auto next = [&]() {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
  };
  for (int y = 8; y < height - 8; y += step_px) {
    for (int x = 8; x < width - 8; x += step_px) {
      const int sx = x * 8;
      const int sy = y * 8;
      int rx = 0;
      int ry = 0;
      ProjectQ8(params, sx, sy, &rx, &ry);
      if (noise_amplitude > 0) {
        rx += static_cast<int>(next() % static_cast<std::uint32_t>(
                                             2 * noise_amplitude + 1)) -
              noise_amplitude;
        ry += static_cast<int>(next() % static_cast<std::uint32_t>(
                                             2 * noise_amplitude + 1)) -
              noise_amplitude;
      }
      cpu::GmPoint point;
      point.src_x = sx;
      point.src_y = sy;
      point.dst_x = rx;
      point.dst_y = ry;
      point.weight = 1;
      points.push_back(point);
    }
  }
  return points;
}

void RunPointsCase(const std::vector<cpu::GmPoint>& points,
                   const cpu::GmOptions& options, const char* label) {
  std::vector<dev::GmePoint> device_points;
  device_points.reserve(points.size());
  for (const cpu::GmPoint& point : points) {
    device_points.push_back(to_dev_point(point));
  }
  std::vector<dev::GmePoint> active(device_points.size() + 1U);
  std::vector<dev::GmeI64> residuals(device_points.size() + 1U);
  std::vector<dev::GmeU8> keep(device_points.size() + 1U);

  const cpu::GmEstimate reference = cpu::estimate_gm_points(points, options);
  const dev::GmeEstimate actual = dev::gme_estimate_points_core(
      device_points.data(), static_cast<int>(device_points.size()),
      to_dev_options(options), active.data(), residuals.data(), keep.data());
  compare_estimate(reference, actual, label);

  // Order independence: reverse the input.  Integer sums and the median
  // selection are order-independent, so the model must be identical.
  std::vector<cpu::GmPoint> reversed(points.rbegin(), points.rend());
  std::vector<dev::GmePoint> device_reversed;
  device_reversed.reserve(reversed.size());
  for (const cpu::GmPoint& point : reversed) {
    device_reversed.push_back(to_dev_point(point));
  }
  const dev::GmeEstimate reversed_actual = dev::gme_estimate_points_core(
      device_reversed.data(), static_cast<int>(device_reversed.size()),
      to_dev_options(options), active.data(), residuals.data(), keep.data());
  Check(reversed_actual.params.type == actual.params.type,
        "reverse type determinism");
  for (int i = 0; i < 6; ++i) {
    CheckEqI64(reversed_actual.params.p[i], actual.params.p[i],
               "reverse params determinism");
  }
}

void RunBlockCase(const std::vector<cpu::GmBlockMv>& blocks,
                  const cpu::GmOptions& options, const char* label) {
  std::vector<dev::GmeBlockMv> device_blocks;
  device_blocks.reserve(blocks.size());
  for (const cpu::GmBlockMv& block : blocks) {
    device_blocks.push_back(to_dev_block(block));
  }
  std::vector<dev::GmePoint> points(device_blocks.size() + 1U);
  std::vector<dev::GmePoint> active(device_blocks.size() + 1U);
  std::vector<dev::GmeI64> residuals(device_blocks.size() + 1U);
  std::vector<dev::GmeU8> keep(device_blocks.size() + 1U);

  const cpu::GmEstimate reference = cpu::estimate_gm_block_mvs(blocks, options);
  const dev::GmeEstimate actual = dev::gme_estimate_blocks_core(
      device_blocks.data(), static_cast<int>(device_blocks.size()),
      to_dev_options(options), points.data(), active.data(), residuals.data(),
      keep.data());
  compare_estimate(reference, actual, label);
}

void RunWarpCase(const std::vector<cpu::GmWarpSample>& warps,
                 const cpu::GmOptions& options, const char* label) {
  std::vector<dev::GmeWarpSample> device_warps;
  device_warps.reserve(warps.size());
  for (const cpu::GmWarpSample& warp : warps) {
    device_warps.push_back(to_dev_warp(warp));
  }
  std::vector<dev::GmePoint> points(warps.size() * 5U + 1U);
  std::vector<dev::GmePoint> active(warps.size() * 5U + 1U);
  std::vector<dev::GmeI64> residuals(warps.size() * 5U + 1U);
  std::vector<dev::GmeU8> keep(warps.size() * 5U + 1U);

  const cpu::GmEstimate reference = cpu::estimate_gm_warps(warps, options);
  const dev::GmeEstimate actual = dev::gme_estimate_warps_core(
      device_warps.data(), static_cast<int>(device_warps.size()),
      to_dev_options(options), points.data(), active.data(), residuals.data(),
      keep.data());
  compare_estimate(reference, actual, label);
}

void TestFits() {
  const int width = 640;
  const int height = 360;
  const cpu::GmParams models[] = {
      Similarity(1.0, 0.0, 0.0, 0.0),
      Similarity(1.0, 0.0, 9.5, -5.25),
      Similarity(1.0, 0.0, -37.25, 22.5),
      Similarity(1.06, 0.0, 0.0, 0.0),
      Similarity(0.94, 0.0, 0.0, 0.0),
      Similarity(1.0, 2.0, 0.0, 0.0),
      Similarity(1.0, -7.5, 4.0, -3.0),
      Similarity(1.04, 1.5, 4.0, -3.0),
      Similarity(1.18, 4.0, -12.0, 7.0),
      Affine(1.04, 0.02, -0.03, 0.98, 4.0, -3.0),
      Affine(0.97, 0.04, 0.015, 1.03, -6.0, 2.0),
  };
  const cpu::GmFitModel fit_models[] = {cpu::GmFitModel::kTranslation,
                                        cpu::GmFitModel::kRotZoom,
                                        cpu::GmFitModel::kAffine};
  int cases = 0;
  for (const cpu::GmParams& model : models) {
    for (cpu::GmFitModel fit_model : fit_models) {
      for (int noise = 0; noise <= 2; noise += 2) {
        for (int seed = 0; seed < 3; ++seed) {
          std::vector<cpu::GmPoint> points = GridPoints(
              model, width, height, 24, seed + 1, noise * 2);
          // Add 8% gross outliers for a second variant.
          if (seed == 2) {
            for (std::size_t i = 0; i < points.size(); i += 13U) {
              points[i].dst_x += 400;
              points[i].dst_y -= 350;
            }
          }
          for (int pass = 0; pass <= 2; ++pass) {
            cpu::GmOptions options;
            options.model = fit_model;
            options.frame_width = width;
            options.frame_height = height;
            options.outlier_passes = pass;
            options.outlier_threshold_q8 = 8 + pass * 8;
            options.min_gain_percent = seed == 0 ? 0 : 1;
            RunPointsCase(points, options, "points fit");
            ++cases;
          }
        }
      }
    }
  }

  // Weighted points and force_integer_mv / quantize-off variants.
  {
    std::vector<cpu::GmPoint> points =
        GridPoints(Similarity(1.03, 1.0, 5.0, -4.0), width, height, 16, 7, 1);
    std::uint32_t seed = 12345U;
    for (cpu::GmPoint& point : points) {
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;
      point.weight = static_cast<std::int32_t>(seed % 255ULL) + 1;
    }
    for (int variant = 0; variant < 4; ++variant) {
      cpu::GmOptions options;
      options.outlier_passes = 3;
      options.outlier_threshold_q8 = 16;
      options.force_integer_mv = (variant & 1) != 0;
      options.quantize = (variant & 2) == 0;
      options.min_gain_percent = 0;
      RunPointsCase(points, options, "weighted fit");
      ++cases;
    }
  }

  // Degenerate / rejection cases.
  {
    std::vector<cpu::GmPoint> points;
    cpu::GmOptions options;
    RunPointsCase(points, options, "no points");
    cpu::GmPoint one{};
    one.weight = 1;
    points.push_back(one);
    RunPointsCase(points, options, "one point");
    for (int i = 0; i < 20; ++i) {
      cpu::GmPoint point;
      point.src_x = i * 8;
      point.src_y = 0;
      point.dst_x = i * 8 + 8;
      point.dst_y = 0;
      point.weight = 1;
      points.push_back(point);
    }
    options.model = cpu::GmFitModel::kAffine;
    RunPointsCase(points, options, "collinear");
    options.model = cpu::GmFitModel::kRotZoom;
    RunPointsCase(points, options, "degenerate rotzoom");
    for (cpu::GmPoint& point : points) point.weight = 0;
    RunPointsCase(points, options, "zero weights");
    options.max_points = 10;
    RunPointsCase(points, options, "too many");
    options = cpu::GmOptions{};
    options.min_points = 100;
    RunPointsCase(points, options, "too few");
  }

  // Block MV fits.
  {
    const cpu::GmParams block_models[] = {
        Similarity(1.0, 0.0, 9.5, -5.25),
        Similarity(1.06, 0.0, 0.0, 0.0),
        Similarity(1.0, 2.0, 0.0, 0.0),
        Affine(1.04, 0.02, -0.03, 0.98, 4.0, -3.0),
    };
    const int block_sizes[] = {4, 8, 16, 24, 32, 48, 64};
    for (const cpu::GmParams& model : block_models) {
      for (int block : block_sizes) {
        std::vector<cpu::GmBlockMv> blocks;
        for (int y = block / 2; y + block <= height; y += block) {
          for (int x = block / 2; x + block <= width; x += block) {
            const int centre_x = x * 8;
            const int centre_y = y * 8;
            int rx = 0;
            int ry = 0;
            ProjectQ8(model, centre_x, centre_y, &rx, &ry);
            cpu::GmBlockMv mv;
            mv.x = centre_x;
            mv.y = centre_y;
            mv.mv_x = rx - centre_x;
            mv.mv_y = ry - centre_y;
            mv.weight = 16;
            blocks.push_back(mv);
          }
        }
        // Zero-weight entries and one overflow entry are exercised separately.
        cpu::GmBlockMv skipped{};
        skipped.weight = 0;
        blocks.push_back(skipped);
        cpu::GmOptions options;
        options.frame_width = width;
        options.frame_height = height;
        options.outlier_passes = 2;
        options.outlier_threshold_q8 = 16;
        RunBlockCase(blocks, options, "block fit");
        ++cases;
      }
    }
    std::vector<cpu::GmBlockMv> overflow(1);
    overflow[0].x = 2147483647;
    overflow[0].mv_x = 1;
    overflow[0].weight = 1;
    cpu::GmOptions options;
    RunBlockCase(overflow, options, "block overflow");
    ++cases;
  }

  // Warp fits, including radius / weight variants.
  {
    const cpu::GmParams warp_models[] = {
        Similarity(1.03, 1.0, 5.0, -4.0),
        Similarity(1.0, 2.0, 0.0, 0.0),
        Similarity(0.95, -1.0, -2.0, 3.0),
        Affine(1.04, 0.02, -0.03, 0.98, 4.0, -3.0),
    };
    const int radii[] = {8, 16, 32, 64};
    for (const cpu::GmParams& model : warp_models) {
      for (int radius : radii) {
        std::vector<cpu::GmWarpSample> warps;
        for (int y = 32; y + 32 <= height; y += 48) {
          for (int x = 32; x + 32 <= width; x += 48) {
            cpu::GmWarpSample warp;
            warp.x = x * 8;
            warp.y = y * 8;
            warp.radius = radius;
            warp.model = model;
            warp.weight = 7 + (x / 48) % 17;
            warps.push_back(warp);
          }
        }
        cpu::GmWarpSample skipped{};
        skipped.weight = 0;
        warps.push_back(skipped);
        cpu::GmOptions options;
        options.frame_width = width;
        options.frame_height = height;
        options.outlier_passes = 2;
        options.outlier_threshold_q8 = 16;
        RunWarpCase(warps, options, "warp fit");
        ++cases;
      }
    }
    std::vector<cpu::GmWarpSample> overflow(1);
    overflow[0].x = 2147483647;
    overflow[0].y = 0;
    overflow[0].radius = 16;
    overflow[0].model = cpu::make_identity_gm();
    overflow[0].weight = 1;
    cpu::GmOptions options;
    RunWarpCase(overflow, options, "warp overflow");
    ++cases;
  }

  std::printf("fit cases: %d\n", cases);
}

// ---------------------------------------------------------------------------
// Quality proxy.
// ---------------------------------------------------------------------------

struct Plane {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> data;

  void Init(int width, int height) {
    w = width;
    h = height;
    data.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  }
  std::uint8_t& At(int x, int y) {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(x)];
  }
  std::uint8_t At(int x, int y) const {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(x)];
  }
};

std::uint8_t TextureSample(int x, int y) {
  const double fx = static_cast<double>(x);
  const double fy = static_cast<double>(y);
  const double value = 128.0 + 58.0 * std::sin(fx * 0.057) +
                       48.0 * std::cos(fy * 0.071) +
                       34.0 * std::sin((fx + fy) * 0.031) +
                       24.0 * std::cos((fx - fy) * 0.043) +
                       18.0 * std::sin(std::sqrt((fx - 320.0) * (fx - 320.0) +
                                                 (fy - 180.0) * (fy - 180.0)) *
                                        0.11);
  return static_cast<std::uint8_t>(
      std::clamp(static_cast<int>(std::llround(value)), 0, 255));
}

Plane MakeTexture(int w, int h) {
  Plane plane;
  plane.Init(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) plane.At(x, y) = TextureSample(x, y);
  }
  return plane;
}

void TestQualityProxy() {
  const int w = 128;
  const int h = 96;
  const Plane source = MakeTexture(w, h);
  const cpu::GmParams models[] = {
      cpu::make_identity_gm(),
      Similarity(1.0, 0.0, 9.5, -5.25),
      Similarity(1.06, 0.0, 0.0, 0.0),
      Similarity(1.0, 2.0, 0.0, 0.0),
      Affine(1.04, 0.02, -0.03, 0.98, 4.0, -3.0),
  };
  for (const cpu::GmParams& model : models) {
    Plane cpu_dst;
    cpu_dst.Init(w, h);
    const bool ok = cpu::warp_plane_bilinear(
        model, source.data.data(), w, h, w, cpu_dst.data.data(), w, h, w, 8);
    Check(ok, "CPU warp plane");
    Plane dev_dst;
    dev_dst.Init(w, h);
    const bool dev_ok = dev::gme_warp_plane_bilinear(
        to_dev_params(model), source.data.data(), w, h, w,
        dev_dst.data.data(), w, h, w, 8);
    Check(dev_ok, "device warp plane");
    Check(cpu_dst.data == dev_dst.data, "warp plane pixel parity");

    const cpu::GmParams identity = cpu::make_identity_gm();
    Plane identity_dst;
    identity_dst.Init(w, h);
    cpu::warp_plane_bilinear(identity, source.data.data(), w, h, w,
                             identity_dst.data.data(), w, h, w, 8);
    const std::int64_t cpu_identity_sse = cpu::warp_plane_sse(
        identity, source.data.data(), w, h, w, cpu_dst.data.data(), w, h, w, 8);
    const std::int64_t cpu_model_sse = cpu::warp_plane_sse(
        model, source.data.data(), w, h, w, cpu_dst.data.data(), w, h, w, 8);
    const std::int64_t dev_identity_sse = dev::gme_warp_plane_sse(
        to_dev_params(identity), source.data.data(), w, h, w,
        cpu_dst.data.data(), w, h, w, 8);
    const std::int64_t dev_model_sse = dev::gme_warp_plane_sse(
        to_dev_params(model), source.data.data(), w, h, w,
        cpu_dst.data.data(), w, h, w, 8);
    CheckEqI64(dev_identity_sse, cpu_identity_sse, "identity SSE proxy");
    CheckEqI64(dev_model_sse, cpu_model_sse, "model SSE proxy");
  }
  // Invalid arguments.
  Check(!dev::gme_warp_plane_bilinear(to_dev_params(cpu::make_identity_gm()),
                                      source.data.data(), w, h, w, nullptr, w,
                                      h, w, 8),
        "bilinear rejects null output");
  std::vector<std::uint8_t> dummy(static_cast<std::size_t>(w) * h);
  Check(!dev::gme_warp_plane_bilinear(to_dev_params(cpu::make_identity_gm()),
                                      source.data.data(), w, h, w,
                                      dummy.data(), w, h, w, 10),
        "bilinear rejects 10-bit");
}

// ---------------------------------------------------------------------------
// Synthetic / real frames through a block matcher.
// ---------------------------------------------------------------------------

Plane WarpPlane(const cpu::GmParams& params, const Plane& src, int w, int h) {
  Plane dst;
  dst.Init(w, h);
  const bool ok = cpu::warp_plane_bilinear(params, src.data.data(), src.w,
                                           src.h, src.w, dst.data.data(), w, h,
                                           w, 8);
  Check(ok, "warp plane");
  return dst;
}

std::int64_t SadAt(const Plane& ref, const Plane& cur, int bx, int by,
                   int block, int dx, int dy, int sample_step) {
  std::int64_t sad = 0;
  for (int y = 0; y < block; y += sample_step) {
    for (int x = 0; x < block; x += sample_step) {
      const int rx = bx + x + dx;
      const int ry = by + y + dy;
      sad += std::abs(static_cast<int>(cur.At(bx + x, by + y)) -
                      static_cast<int>(ref.At(rx, ry)));
    }
  }
  return sad;
}

struct MatchOptions {
  int block = 16;
  int search = 16;
  int sample_step = 2;
  int min_variance = 24;
};

std::vector<cpu::GmBlockMv> MatchBlocks(const Plane& ref, const Plane& cur,
                                         const MatchOptions& options) {
  std::vector<cpu::GmBlockMv> out;
  for (int by = 0; by + options.block <= cur.h; by += options.block) {
    for (int bx = 0; bx + options.block <= cur.w; bx += options.block) {
      std::int64_t sum = 0;
      std::int64_t sum2 = 0;
      int count = 0;
      for (int y = 0; y < options.block; y += options.sample_step) {
        for (int x = 0; x < options.block; x += options.sample_step) {
          const int v = cur.At(bx + x, by + y);
          sum += v;
          sum2 += v * v;
          ++count;
        }
      }
      const double mean = static_cast<double>(sum) / count;
      const double variance = static_cast<double>(sum2) / count - mean * mean;
      if (variance < options.min_variance) continue;

      std::int64_t best = std::numeric_limits<std::int64_t>::max();
      int best_dx = 0;
      int best_dy = 0;
      for (int dy = -options.search; dy <= options.search; ++dy) {
        for (int dx = -options.search; dx <= options.search; ++dx) {
          if (bx + dx < 0 || by + dy < 0 ||
              bx + dx + options.block > ref.w ||
              by + dy + options.block > ref.h) {
            continue;
          }
          const std::int64_t sad =
              SadAt(ref, cur, bx, by, options.block, dx, dy,
                    options.sample_step);
          if (sad < best) {
            best = sad;
            best_dx = dx;
            best_dy = dy;
          }
        }
      }
      if (best == std::numeric_limits<std::int64_t>::max()) continue;

      auto subpel = [&](int delta_x, int delta_y) -> double {
        if (bx + delta_x < 0 || by + delta_y < 0 ||
            bx + delta_x + options.block > ref.w ||
            by + delta_y + options.block > ref.h) {
          return static_cast<double>(best);
        }
        return static_cast<double>(SadAt(ref, cur, bx, by, options.block,
                                         delta_x, delta_y,
                                         options.sample_step));
      };
      const double best_sad = static_cast<double>(best);
      const double smx = subpel(best_dx - 1, best_dy);
      const double spx = subpel(best_dx + 1, best_dy);
      const double smy = subpel(best_dx, best_dy - 1);
      const double spy = subpel(best_dx, best_dy + 1);
      const double denx = smx - 2.0 * best_sad + spx;
      const double deny = smy - 2.0 * best_sad + spy;
      double sub_x = denx > 0.0 ? 0.5 * (smx - spx) / denx : 0.0;
      double sub_y = deny > 0.0 ? 0.5 * (smy - spy) / deny : 0.0;
      sub_x = std::clamp(sub_x, -0.5, 0.5);
      sub_y = std::clamp(sub_y, -0.5, 0.5);

      cpu::GmBlockMv block;
      block.x = (bx + options.block / 2) * 8;
      block.y = (by + options.block / 2) * 8;
      block.mv_x = best_dx * 8 + static_cast<int>(std::llround(sub_x * 8.0));
      block.mv_y = best_dy * 8 + static_cast<int>(std::llround(sub_y * 8.0));
      block.weight =
          static_cast<int>(std::clamp(variance / 8.0, 1.0, 255.0));
      out.push_back(block);
    }
  }
  return out;
}

void TestSyntheticFrames() {
  const int w = 640;
  const int h = 360;
  const Plane source = MakeTexture(w, h);
  cpu::GmOptions options;
  options.frame_width = w;
  options.frame_height = h;
  options.outlier_passes = 3;
  options.outlier_threshold_q8 = 16;

  struct Case {
    const char* name;
    cpu::GmParams model;
  };
  const Case cases[] = {
      {"pan", Similarity(1.0, 0.0, 9.5, -5.25)},
      {"zoom", Similarity(1.06, 0.0, 0.0, 0.0)},
      {"rotate", Similarity(1.0, 2.0, 0.0, 0.0)},
      {"pan+zoom+rotate", Similarity(1.04, 1.5, 4.0, -3.0)},
  };
  MatchOptions match;
  match.block = 24;
  match.search = 24;
  match.sample_step = 2;
  for (const Case& item : cases) {
    const Plane current = WarpPlane(item.model, source, w, h);
    const std::vector<cpu::GmBlockMv> blocks = MatchBlocks(source, current, match);
    Check(blocks.size() > 100U, "synthetic matcher produced blocks");
    RunBlockCase(blocks, options, item.name);

    const std::int64_t identity_sse = cpu::warp_plane_sse(
        cpu::make_identity_gm(), source.data.data(), w, h, w,
        current.data.data(), w, h, w, 8);
    const dev::GmeEstimate actual = [&]() {
      std::vector<dev::GmeBlockMv> device_blocks;
      device_blocks.reserve(blocks.size());
      for (const cpu::GmBlockMv& block : blocks) {
        device_blocks.push_back(to_dev_block(block));
      }
      std::vector<dev::GmePoint> points(device_blocks.size() + 1U);
      std::vector<dev::GmePoint> active(device_blocks.size() + 1U);
      std::vector<dev::GmeI64> residuals(device_blocks.size() + 1U);
      std::vector<dev::GmeU8> keep(device_blocks.size() + 1U);
      return dev::gme_estimate_blocks_core(
          device_blocks.data(), static_cast<int>(device_blocks.size()),
          to_dev_options(options), points.data(), active.data(),
          residuals.data(), keep.data());
    }();
    const std::int64_t model_sse = dev::gme_warp_plane_sse(
        actual.params, source.data.data(), w, h, w, current.data.data(), w, h,
        w, 8);
    Check(identity_sse >= 0 && model_sse >= 0, "synthetic proxy SSE valid");
    Check(model_sse <= identity_sse, "synthetic proxy improves");
    std::printf(
        "  synthetic %-16s blocks=%zu used=%d type=%d gain=%.2f dB sse "
        "%lld -> %lld\n",
        item.name, blocks.size(), actual.diag.points_used,
        actual.params.type, actual.diag.gain_db,
        static_cast<long long>(identity_sse),
        static_cast<long long>(model_sse));
  }
}

bool LoadY4mFrames(const std::string& path, int max_frames,
                   std::vector<Plane>* out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  std::string line;
  if (!std::getline(input, line) || line.rfind("YUV4MPEG2", 0) != 0) {
    return false;
  }
  std::istringstream header(line);
  std::string token;
  int width = 0;
  int height = 0;
  bool c420 = false;
  while (header >> token) {
    if (token.size() > 1) {
      const char tag = token[0];
      const std::string value = token.substr(1);
      if (tag == 'W') width = std::stoi(value);
      if (tag == 'H') height = std::stoi(value);
      if (tag == 'C') {
        c420 = value == "420" || value == "420jpeg" || value == "420mpeg2";
      }
    }
  }
  if (width <= 0 || height <= 0 || !c420) return false;
  const int cw = (width + 1) / 2;
  const int ch = (height + 1) / 2;
  std::vector<std::uint8_t> ybuf(static_cast<std::size_t>(width) * height);
  std::vector<std::uint8_t> ubuf(static_cast<std::size_t>(cw) * ch);
  std::vector<std::uint8_t> vbuf(static_cast<std::size_t>(cw) * ch);
  for (int frame = 0; frame < max_frames; ++frame) {
    if (!std::getline(input, line) || line.rfind("FRAME", 0) != 0) break;
    if (!input.read(reinterpret_cast<char*>(ybuf.data()),
                    static_cast<std::streamsize>(ybuf.size())) ||
        !input.read(reinterpret_cast<char*>(ubuf.data()),
                    static_cast<std::streamsize>(ubuf.size())) ||
        !input.read(reinterpret_cast<char*>(vbuf.data()),
                    static_cast<std::streamsize>(vbuf.size()))) {
      break;
    }
    Plane plane;
    plane.Init(width, height);
    plane.data = ybuf;
    out->push_back(std::move(plane));
  }
  return !out->empty();
}

void TestReal720p(const std::string& repo_root, bool quick) {
  const std::string path = repo_root + "/testdata/shine_1280x720_60f.y4m";
  std::vector<Plane> frames;
  if (!LoadY4mFrames(path, 11, &frames) || frames.size() < 2) {
    std::printf("  real 720p: skipped (no testdata)\n");
    return;
  }
  cpu::GmOptions options;
  options.frame_width = frames[0].w;
  options.frame_height = frames[0].h;
  options.outlier_passes = 3;
  options.outlier_threshold_q8 = 24;

  auto analyze = [&](const Plane& reference, const Plane& current,
                     const char* label, int search) {
    MatchOptions match;
    match.block = 16;
    match.search = search;
    match.sample_step = 2;
    const std::vector<cpu::GmBlockMv> blocks =
        MatchBlocks(reference, current, match);
    RunBlockCase(blocks, options, label);
    std::printf("  real 720p %-16s blocks=%zu\n", label, blocks.size());
  };

  analyze(frames[0], frames[1], "pair 0/1", 16);
  if (!quick && frames.size() >= 11) {
    analyze(frames[0], frames[10], "pair 0/10", 32);
  }

  MatchOptions match;
  match.block = 16;
  match.search = 32;
  match.sample_step = 2;
  const cpu::GmParams known = Similarity(1.02, 0.5, 3.0, -2.0);
  const Plane warped = WarpPlane(known, frames[0], frames[0].w, frames[0].h);
  const std::vector<cpu::GmBlockMv> warped_blocks =
      MatchBlocks(frames[0], warped, match);
  RunBlockCase(warped_blocks, options, "real 720p known warp");
  std::printf("  real 720p known warp: blocks=%zu\n", warped_blocks.size());
}

// ---------------------------------------------------------------------------
// Arena and staging.
// ---------------------------------------------------------------------------

void TestArenaAndStaging() {
  const dev::GmeConfig config{17, 9, 5, 2};
  const std::uint64_t need = dev::required_device_bytes(config);
  Check(need > 0, "required_device_bytes positive");
  const auto up = [](std::uint64_t value) {
    return (value + 255ULL) & ~static_cast<std::uint64_t>(255ULL);
  };
  const std::uint64_t jobs = 2;
  const std::uint64_t point_total = jobs * 17;
  const std::uint64_t block_total = jobs * 9;
  const std::uint64_t warp_total = jobs * 5;
  const std::uint64_t expected =
      up(point_total * sizeof(dev::GmePoint)) +
      up(point_total * sizeof(dev::GmePoint)) +
      up(point_total * sizeof(dev::GmeI64)) +
      up(point_total * sizeof(dev::GmeU8)) +
      up(block_total * sizeof(dev::GmeBlockMv)) +
      up(warp_total * sizeof(dev::GmeWarpSample)) +
      up(jobs * sizeof(dev::GmeEstimate)) +
      up(jobs * sizeof(dev::GmeJobControl));
  CheckEqI64(static_cast<long long>(need),
             static_cast<long long>(expected), "required layout");
  Check(dev::required_device_bytes(dev::GmeConfig{0, 1, 1, 1}) == 0,
        "invalid max_points");
  Check(dev::required_device_bytes(dev::GmeConfig{1, 0, 1, 1}) == 0,
        "invalid max_blocks");
  Check(dev::required_device_bytes(dev::GmeConfig{1, 1, 0, 1}) == 0,
        "invalid max_warps");
  Check(dev::required_device_bytes(dev::GmeConfig{1, 1, 1, 0}) == 0,
        "invalid max_jobs");

  {
    av2enc::cuda::Arena arena;
    Check(arena.init(need + av2enc::cuda::Arena::kAlignment) ==
              av2enc::cuda::Status::kOk,
          "arena init need+slack");
    dev::GmeBuffers buffers;
    std::uint64_t reported = 0;
    std::string error;
    Check(dev::allocate_checked(arena, config, &buffers, &reported, &error) ==
              dev::GmeStatus::kOk,
          "allocate exact");
    CheckEqI64(static_cast<long long>(reported),
               static_cast<long long>(need), "reported need");
    CheckEqI64(static_cast<long long>(arena.used()),
               static_cast<long long>(need), "used == need");
    Check(buffers.persistent_mark != dev::kGmeNoArenaMark, "mark recorded");

    // Stage points / blocks / warps and read them back through the host stub.
    std::vector<dev::GmePoint> points(17);
    for (int i = 0; i < 17; ++i) {
      points[static_cast<std::size_t>(i)].src_x = i;
      points[static_cast<std::size_t>(i)].src_y = i * 2;
      points[static_cast<std::size_t>(i)].dst_x = i + 3;
      points[static_cast<std::size_t>(i)].dst_y = i * 2 - 4;
      points[static_cast<std::size_t>(i)].weight = i + 1;
    }
    Check(dev::stage_points(buffers, 0, points.data(), 17) ==
              dev::GmeStatus::kOk,
          "stage points");
    for (int i = 0; i < 17; ++i) {
      CheckEqI64(buffers.d_points[i].dst_x, points[static_cast<std::size_t>(i)].dst_x,
                 "staged point");
    }
    std::vector<dev::GmeBlockMv> blocks(9);
    for (int i = 0; i < 9; ++i) {
      blocks[static_cast<std::size_t>(i)].x = i * 8;
      blocks[static_cast<std::size_t>(i)].weight = 3;
    }
    Check(dev::stage_blocks(buffers, 1, blocks.data(), 9) ==
              dev::GmeStatus::kOk,
          "stage blocks");
    CheckEqI64(buffers.d_blocks[9].x, 0, "second job block offset");
    CheckEqI64(buffers.d_blocks[9 + 4].x, 32, "second job block value");
    std::vector<dev::GmeWarpSample> warps(5);
    for (int i = 0; i < 5; ++i) {
      warps[static_cast<std::size_t>(i)].x = i;
      warps[static_cast<std::size_t>(i)].weight = 5;
    }
    Check(dev::stage_warps(buffers, 0, warps.data(), 5) ==
              dev::GmeStatus::kOk,
          "stage warps");
    CheckEqI64(buffers.d_warps[4].x, 4, "staged warp");

    // Download path: write a result by hand (host stub) and read it back.
    buffers.d_results[0] = dev::gme_make_identity_estimate();
    buffers.d_results[0].params.p[0] = 12345;
    dev::GmeEstimate downloaded;
    Check(dev::download_estimate(buffers, 0, &downloaded) ==
              dev::GmeStatus::kOk,
          "download estimate");
    CheckEqI64(downloaded.params.p[0], 12345, "downloaded param");

    // Two staging calls leave the arena flat.
    const std::uint64_t used = arena.used();
    const std::uint64_t peak = arena.peak_used();
    Check(dev::stage_points(buffers, 0, points.data(), 17) ==
              dev::GmeStatus::kOk,
          "second stage points");
    CheckEqI64(static_cast<long long>(arena.used()),
               static_cast<long long>(used), "no growth after stage");
    CheckEqI64(static_cast<long long>(arena.peak_used()),
               static_cast<long long>(peak), "no peak growth after stage");

    // mark / rewind bracket.
    const std::uint64_t mark = dev::gme_mark(arena);
    void* probe = nullptr;
    Check(arena.allocate_checked(av2enc::cuda::Arena::kAlignment, &probe) ==
              av2enc::cuda::Status::kOk,
          "probe allocation");
    Check(dev::gme_rewind(arena, mark), "rewind probe");
    CheckEqI64(static_cast<long long>(arena.used()),
               static_cast<long long>(used), "rewind restores used");

    dev::deallocate(arena, buffers);
    CheckEqI64(static_cast<long long>(arena.used()), 0, "deallocate zero");
    Check(buffers.persistent_mark == dev::kGmeNoArenaMark,
          "deallocate clears mark");
  }

  {
    av2enc::cuda::Arena arena;
    Check(arena.init(need - 1) == av2enc::cuda::Status::kOk,
          "short arena init");
    dev::GmeBuffers buffers;
    std::uint64_t reported = 0;
    std::string error;
    Check(dev::allocate_checked(arena, config, &buffers, &reported, &error) ==
              dev::GmeStatus::kArenaTooSmall,
          "one byte short -> kArenaTooSmall");
    CheckEqI64(static_cast<long long>(reported),
               static_cast<long long>(need), "short reports need");
    CheckEqI64(static_cast<long long>(arena.used()), 0,
               "short arena bump unchanged");
    Check(buffers.d_points == nullptr, "short arena leaves no buffers");
  }

  {
    av2enc::cuda::Arena arena;
    Check(arena.init(need * 2) == av2enc::cuda::Status::kOk,
          "double arena init");
    dev::GmeBuffers first;
    dev::GmeBuffers second;
    Check(dev::allocate_checked(arena, config, &first) == dev::GmeStatus::kOk,
          "first allocation");
    Check(dev::allocate_checked(arena, config, &second) == dev::GmeStatus::kOk,
          "second allocation");
    Check(second.persistent_mark > first.persistent_mark,
          "second mark above first");
    dev::deallocate(arena, second);
    CheckEqI64(static_cast<long long>(arena.used()),
               static_cast<long long>(need), "rewind to first");
    Check(first.d_points != nullptr, "first still valid");
    dev::deallocate(arena, first);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string repo_root = argc > 1 ? argv[1] : ".";
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--quick") quick = true;
  }
  std::printf("gme_emulation_test: CUDA GME host parity\n");
  TestI128();
  TestProjection();
  TestFinalize();
  TestFits();
  TestQualityProxy();
  TestSyntheticFrames();
  TestReal720p(repo_root, quick);
  TestArenaAndStaging();
  std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
