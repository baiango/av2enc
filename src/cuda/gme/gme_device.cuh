// AV2 v1.0.0 global motion estimation — shared host/device core.
//
// This header is the CUDA-side transcription of the integer parts of
// `src/encoder/inter/gme/gme.{h,cpp}` and `gme_core.h`:
//
//   * fit from correspondences (weighted integer least squares),
//   * fit from block MVs and per-block warps,
//   * deterministic median outlier rejection,
//   * §5.18.9 quantisation, AFFINE -> ROTZOOM -> IDENTITY degradation,
//   * §7.13.3.21 shear validity, §7.13.3.23 model reduction,
//   * §7.12.2.2 projection, MV clamp and precision lowering,
//   * the bilinear quality proxy used by the CPU reference.
//
// The CPU reference accumulates in `__int128`.  Device code cannot rely on a
// 128-bit integer type, so this header implements a small portable signed
// 128-bit integer (`GmeI128`) with the same two's-complement semantics and the
// same round-to-nearest division used by the reference.  It is compiled by:
//
//   * nvcc as `__host__ __device__` code for the kernels, and
//   * the host compiler (no __CUDACC__) for the no-nvcc parity test.
//
// Every free function is marked AV2ENC_GME_HD.  There are no namespace-scope
// variable definitions, no `std::` algorithms and no host-only symbols in any
// path reachable from device code (see README "Device-visibility audit").
#ifndef AV2ENC_CUDA_GME_GME_DEVICE_CUH_
#define AV2ENC_CUDA_GME_GME_DEVICE_CUH_

#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define AV2ENC_GME_HD __host__ __device__ inline
#else
#define AV2ENC_GME_HD inline
#endif

namespace av2enc::cuda::gme {

using GmeI32 = std::int32_t;
using GmeI64 = std::int64_t;
using GmeU64 = std::uint64_t;
using GmeU8 = std::uint8_t;

// ---------------------------------------------------------------------------
// Constants.  Enumerators are compile-time constants in both host and device
// code and never require a device symbol.
// ---------------------------------------------------------------------------

enum : int {
  kGmeMiSize = 4,
  kGmeMvBorder = 128,
  kGmeMvLow = -(1 << 16),
  kGmeMvUpp = 1 << 16,

  kGmeWarpedModelPrecBits = 16,
  kWarpedModelPrecBitsAlias = 16,
  kGmeWarpParamReduceBits = 6,
  kGmeGmTransPrecBits = 3,
  kGmeGmAlphaPrecBits = 10,
  kGmeGmTransPrecDiff = kGmeWarpedModelPrecBits - kGmeGmTransPrecBits,
  kGmeGmAlphaPrecDiff = kGmeWarpedModelPrecBits - kGmeGmAlphaPrecBits,
  kGmeGmAbsTransBits = 14,
  kGmeGmAbsAlphaBits = 9,
  kGmeGmTransMax = (1 << kGmeGmAbsTransBits) - 1,
  kGmeGmAlphaMax = (1 << kGmeGmAbsAlphaBits) - 1,
  kGmeGmTransMin = -kGmeGmTransMax,
  kGmeGmAlphaMin = -kGmeGmAlphaMax,
  kGmeGmTransStep = 1 << kGmeGmTransPrecDiff,
  kGmeGmAlphaStep = 1 << kGmeGmAlphaPrecDiff,
  kGmeGmOne = 1 << kGmeWarpedModelPrecBits,

  kGmeDivLutBits = 7,
  kGmeDivLutPrecBits = 9,
  kGmeDivLutNum = 129,

  kGmeMaxRefsPerFrame = 7,
  kGmeNumRefFrames = 16,
};

// §5.18.9 GmType values.
enum GmeGmType : GmeI32 {
  kGmeIdentity = 0,
  kGmeRotZoom = 1,
  kGmeAffine = 2,
};

// GmFitModel values (CPU `GmFitModel`).
enum GmeFitModel : GmeI32 {
  kGmeFitTranslation = 0,
  kGmeFitRotZoom = 1,
  kGmeFitAffine = 2,
};

// MvPrecision values (§6.19.7.x).
enum GmeMvPrecision : GmeI32 {
  kGmeOnePel = 4,
  kGmeHalfPel = 5,
  kGmeQuarterPel = 6,
  kGmeEighthPel = 7,
};

// Diagnostics reason codes.  The CPU reference carries a std::string; the
// device result carries a stable code so no allocation is needed.  The tests
// map the CPU strings to these codes (see gme_emulation_test.cpp).
enum GmeReason : int {
  kGmeReasonNone = 0,
  kGmeReasonNoPoints,
  kGmeReasonTooFewWeighted,
  kGmeReasonDegenerate,
  kGmeReasonCollinear,
  kGmeReasonTooFewCorrespondences,
  kGmeReasonTooManyCorrespondences,
  kGmeReasonBlockMvOverflow,
  kGmeReasonWarpOverflow,
  kGmeReasonIdentityNonDefault,
  kGmeReasonTransGrid,
  kGmeReasonTransRange,
  kGmeReasonAlphaGrid,
  kGmeReasonAlphaRange,
  kGmeReasonRotzoomSymmetry,
  kGmeReasonUnknownType,
  kGmeReasonScaleOutOfRange,
  kGmeReasonShearInvalid,
  kGmeReasonCornerOutOfRange,
  kGmeReasonAffineToRotzoom,
  kGmeReasonDegradedAfterReject,
  kGmeReasonModelRejected,
  kGmeReasonGainBelowThreshold,
  kGmeReasonCount,
};

// ---------------------------------------------------------------------------
// Small deterministic integer helpers (host float environment must not leak
// into fitting).  Names are prefixed to avoid collisions with the CPU header.
// ---------------------------------------------------------------------------

AV2ENC_GME_HD constexpr GmeI32 gme_clip3(GmeI32 low, GmeI32 high, GmeI32 value) {
  return value < low ? low : (value > high ? high : value);
}

AV2ENC_GME_HD constexpr GmeI32 gme_abs_int(GmeI32 value) {
  return value < 0 ? -value : value;
}

AV2ENC_GME_HD constexpr int gme_floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

AV2ENC_GME_HD constexpr int gme_round2(int value, int n) {
  if (n <= 0) return value;
  return (value + (1 << (n - 1))) >> n;
}

AV2ENC_GME_HD constexpr int gme_round2_signed(int value, int n) {
  return value >= 0 ? gme_round2(value, n) : -gme_round2(-value, n);
}

AV2ENC_GME_HD GmeI64 gme_clamp_i64(GmeI64 value, GmeI64 low, GmeI64 high) {
  return value < low ? low : (value > high ? high : value);
}

AV2ENC_GME_HD GmeI64 gme_abs_i64(GmeI64 value) {
  return value < 0 ? -value : value;
}

AV2ENC_GME_HD GmeI64 gme_max_i64(GmeI64 a, GmeI64 b) {
  return a > b ? a : b;
}

AV2ENC_GME_HD GmeI64 gme_min_i64(GmeI64 a, GmeI64 b) {
  return a < b ? a : b;
}

AV2ENC_GME_HD constexpr GmeI64 gme_i32_min() {
  return static_cast<GmeI64>(-2147483647LL - 1LL);
}
AV2ENC_GME_HD constexpr GmeI64 gme_i32_max() { return 2147483647LL; }
AV2ENC_GME_HD constexpr GmeI64 gme_i64_max() {
  return 9223372036854775807LL;
}
AV2ENC_GME_HD constexpr GmeI64 gme_i64_min() {
  return -9223372036854775807LL - 1LL;
}

AV2ENC_GME_HD GmeI64 gme_round2_signed64(GmeI64 value, int n) {
  if (n <= 0) return value;
  const GmeI64 half = static_cast<GmeI64>(1) << (n - 1);
  return value >= 0 ? ((value + half) >> n) : -(((-value) + half) >> n);
}

// ---------------------------------------------------------------------------
// Portable signed 128-bit integer.
// ---------------------------------------------------------------------------

struct GmeU128 {
  GmeU64 lo;
  GmeU64 hi;
};

struct GmeI128 {
  GmeU64 lo;
  GmeI64 hi;
};

AV2ENC_GME_HD GmeI128 gme_i128_from_i64(GmeI64 value) {
  GmeI128 out;
  out.lo = static_cast<GmeU64>(value);
  out.hi = value < 0 ? -1 : 0;
  return out;
}

AV2ENC_GME_HD GmeI128 gme_i128_zero() {
  GmeI128 out;
  out.lo = 0;
  out.hi = 0;
  return out;
}

AV2ENC_GME_HD bool gme_i128_is_zero(GmeI128 value) {
  return value.lo == 0 && value.hi == 0;
}

AV2ENC_GME_HD GmeI128 gme_i128_add(GmeI128 a, GmeI128 b) {
  GmeI128 out;
  out.lo = a.lo + b.lo;
  const GmeU64 carry = (out.lo < a.lo) ? 1ULL : 0ULL;
  out.hi = static_cast<GmeI64>(static_cast<GmeU64>(a.hi) +
                               static_cast<GmeU64>(b.hi) + carry);
  return out;
}

AV2ENC_GME_HD GmeI128 gme_i128_sub(GmeI128 a, GmeI128 b) {
  GmeI128 out;
  out.lo = a.lo - b.lo;
  const GmeU64 borrow = (a.lo < b.lo) ? 1ULL : 0ULL;
  out.hi = static_cast<GmeI64>(static_cast<GmeU64>(a.hi) -
                               static_cast<GmeU64>(b.hi) - borrow);
  return out;
}

AV2ENC_GME_HD GmeI128 gme_i128_neg(GmeI128 value) {
  return gme_i128_sub(gme_i128_zero(), value);
}

AV2ENC_GME_HD int gme_i128_cmp(GmeI128 a, GmeI128 b) {
  if (a.hi < b.hi) return -1;
  if (a.hi > b.hi) return 1;
  if (a.lo < b.lo) return -1;
  if (a.lo > b.lo) return 1;
  return 0;
}

AV2ENC_GME_HD GmeI128 gme_i128_max(GmeI128 a, GmeI128 b) {
  return gme_i128_cmp(a, b) >= 0 ? a : b;
}

AV2ENC_GME_HD GmeU128 gme_i128_abs_u(GmeI128 value) {
  GmeU128 out;
  if (value.hi >= 0) {
    out.lo = value.lo;
    out.hi = static_cast<GmeU64>(value.hi);
    return out;
  }
  // Two's-complement negation of the magnitude.
  out.lo = ~value.lo + 1ULL;
  const GmeU64 carry = (out.lo == 0) ? 1ULL : 0ULL;
  out.hi = ~static_cast<GmeU64>(value.hi) + carry;
  return out;
}

AV2ENC_GME_HD GmeI128 gme_i128_from_u(GmeU128 value, bool negative) {
  GmeI128 out;
  if (!negative) {
    out.lo = value.lo;
    out.hi = static_cast<GmeI64>(value.hi);
    return out;
  }
  out.lo = ~value.lo + 1ULL;
  const GmeU64 carry = (out.lo == 0) ? 1ULL : 0ULL;
  out.hi = static_cast<GmeI64>(~value.hi + carry);
  return out;
}

// 64x64 -> 128 unsigned low product.
AV2ENC_GME_HD GmeU128 gme_mul64(GmeU64 a, GmeU64 b) {
  const GmeU64 a0 = a & 0xffffffffULL;
  const GmeU64 a1 = a >> 32;
  const GmeU64 b0 = b & 0xffffffffULL;
  const GmeU64 b1 = b >> 32;
  const GmeU64 p00 = a0 * b0;
  const GmeU64 p01 = a0 * b1;
  const GmeU64 p10 = a1 * b0;
  const GmeU64 p11 = a1 * b1;
  const GmeU64 mid = (p00 >> 32) + (p01 & 0xffffffffULL) +
                     (p10 & 0xffffffffULL);
  GmeU128 out;
  out.lo = (mid << 32) | (p00 & 0xffffffffULL);
  out.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
  return out;
}

// Low 128 bits of a*b.  Values used by the fitter fit comfortably in the
// signed 128-bit range, so the low half is the exact result.
AV2ENC_GME_HD GmeI128 gme_i128_mul(GmeI128 a, GmeI128 b) {
  const bool negative = (a.hi < 0) != (b.hi < 0);
  const GmeU128 ua = gme_i128_abs_u(a);
  const GmeU128 ub = gme_i128_abs_u(b);
  const GmeU128 p00 = gme_mul64(ua.lo, ub.lo);
  const GmeU128 p01 = gme_mul64(ua.lo, ub.hi);
  const GmeU128 p10 = gme_mul64(ua.hi, ub.lo);
  GmeU128 result;
  result.lo = p00.lo;
  result.hi = p00.hi + p01.lo + p10.lo;
  return gme_i128_from_u(result, negative);
}

AV2ENC_GME_HD GmeI128 gme_i128_mul_i64(GmeI128 a, GmeI64 b) {
  return gme_i128_mul(a, gme_i128_from_i64(b));
}

AV2ENC_GME_HD GmeI128 gme_i128_shl(GmeI128 value, int shift) {
  if (shift <= 0) return value;
  GmeI128 out;
  if (shift >= 128) {
    out.lo = 0;
    out.hi = 0;
    return out;
  }
  if (shift >= 64) {
    out.lo = 0;
    out.hi = static_cast<GmeI64>(value.lo << (shift - 64));
    return out;
  }
  out.lo = value.lo << shift;
  out.hi = static_cast<GmeI64>(
      (static_cast<GmeU64>(value.hi) << shift) |
      (value.lo >> (64 - shift)));
  return out;
}

// Arithmetic right shift (sign extension), matching the CPU `__int128 >> n`.
AV2ENC_GME_HD GmeI128 gme_i128_shr(GmeI128 value, int shift) {
  if (shift <= 0) return value;
  if (shift >= 128) {
    GmeI128 out;
    out.lo = value.hi < 0 ? ~static_cast<GmeU64>(0) : 0ULL;
    out.hi = value.hi < 0 ? -1 : 0;
    return out;
  }
  GmeI128 out;
  if (shift == 64) {
    out.lo = static_cast<GmeU64>(value.hi);
    out.hi = value.hi < 0 ? -1 : 0;
    return out;
  }
  if (shift > 64) {
    out.lo = static_cast<GmeU64>(value.hi) >> (shift - 64);
    if (value.hi < 0) {
      out.lo |= (~static_cast<GmeU64>(0)) << (128 - shift);
    }
    out.hi = value.hi < 0 ? -1 : 0;
    return out;
  }
  out.lo = (value.lo >> shift) |
           (static_cast<GmeU64>(value.hi) << (64 - shift));
  out.hi = value.hi >> shift;
  return out;
}

AV2ENC_GME_HD bool gme_u128_lt(GmeU128 a, GmeU128 b) {
  return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

AV2ENC_GME_HD GmeU128 gme_u128_sub(GmeU128 a, GmeU128 b) {
  GmeU128 out;
  out.lo = a.lo - b.lo;
  out.hi = a.hi - b.hi - ((a.lo < b.lo) ? 1ULL : 0ULL);
  return out;
}

AV2ENC_GME_HD GmeU128 gme_u128_add(GmeU128 a, GmeU128 b) {
  GmeU128 out;
  out.lo = a.lo + b.lo;
  out.hi = a.hi + b.hi + ((out.lo < a.lo) ? 1ULL : 0ULL);
  return out;
}

AV2ENC_GME_HD GmeU128 gme_u128_shl1(GmeU128 value) {
  GmeU128 out;
  out.lo = value.lo << 1;
  out.hi = (value.hi << 1) | (value.lo >> 63);
  return out;
}

struct GmeUDiv {
  GmeU128 q;
  GmeU128 r;
};

// Unsigned 128/128 division by binary long division.  Deterministic and exact.
AV2ENC_GME_HD GmeUDiv gme_u128_divmod(GmeU128 numerator,
                                             GmeU128 denominator) {
  GmeUDiv out;
  out.q.lo = 0;
  out.q.hi = 0;
  out.r.lo = 0;
  out.r.hi = 0;
  if (denominator.hi == 0 && denominator.lo == 0) {
    return out;
  }
  // 129-bit shift-subtract remainder: `carry` is bit 128.  Keeping it
  // explicit makes the routine exact for any 128-bit denominator, including
  // values with bit 127 set (the fitter's denominators are smaller, but the
  // helper is also used by the emulation self-test).
  GmeU64 remainder_hi = 0;
  GmeU64 remainder_lo = 0;
  for (int bit = 127; bit >= 0; --bit) {
    const bool carry = (remainder_hi >> 63) != 0;
    remainder_hi = (remainder_hi << 1) | (remainder_lo >> 63);
    const GmeU64 value_bit =
        (bit >= 64) ? ((numerator.hi >> (bit - 64)) & 1ULL)
                    : ((numerator.lo >> bit) & 1ULL);
    remainder_lo = (remainder_lo << 1) | value_bit;
    GmeU128 current;
    current.lo = remainder_lo;
    current.hi = remainder_hi;
    if (carry || !gme_u128_lt(current, denominator)) {
      const GmeU64 old_lo = remainder_lo;
      remainder_lo -= denominator.lo;
      remainder_hi = remainder_hi - denominator.hi -
                     ((old_lo < denominator.lo) ? 1ULL : 0ULL);
      if (bit >= 64) {
        out.q.hi |= (1ULL << (bit - 64));
      } else {
        out.q.lo |= (1ULL << bit);
      }
    }
  }
  out.r.lo = remainder_lo;
  out.r.hi = remainder_hi;
  return out;
}

// CPU `to_i64_sat`.
AV2ENC_GME_HD GmeI64 gme_i128_to_i64_sat(GmeI128 value) {
  if (value.hi > 0) return gme_i64_max();
  if (value.hi < -1) return gme_i64_min();
  return static_cast<GmeI64>(value.lo);
}

// CPU `saturate_i32`.
AV2ENC_GME_HD GmeI32 gme_i128_sat_i32(GmeI128 value) {
  const GmeI64 wide = gme_i128_to_i64_sat(value);
  if (wide > gme_i32_max()) return static_cast<GmeI32>(gme_i32_max());
  if (wide < gme_i32_min()) return static_cast<GmeI32>(gme_i32_min());
  return static_cast<GmeI32>(wide);
}

// CPU `div_round_i128`: round(num/den) to nearest, ties away from zero.
AV2ENC_GME_HD GmeI64 gme_i128_div_round(GmeI128 numerator,
                                               GmeI128 denominator) {
  if (denominator.hi < 0 ||
      (denominator.hi == 0 && denominator.lo == 0)) {
    return 0;
  }
  const bool negative = numerator.hi < 0;
  const GmeU128 n = gme_i128_abs_u(numerator);
  const GmeU128 d = gme_i128_abs_u(denominator);
  GmeUDiv qr = gme_u128_divmod(n, d);
  const GmeU128 twice = gme_u128_add(qr.r, qr.r);
  if (!gme_u128_lt(twice, d)) {
    qr.q = gme_u128_add(qr.q, GmeU128{1ULL, 0ULL});
  }
  if (qr.q.hi != 0 || qr.q.lo > static_cast<GmeU64>(gme_i64_max())) {
    return negative ? gme_i64_min() : gme_i64_max();
  }
  const GmeI64 result = static_cast<GmeI64>(qr.q.lo);
  return negative ? -result : result;
}

// CPU `round_shift_i128`.
AV2ENC_GME_HD GmeI64 gme_i128_round_shift(GmeI128 value, int shift) {
  if (shift <= 0) return gme_i128_to_i64_sat(value);
  return gme_i128_div_round(
      value, gme_i128_from_i64(static_cast<GmeI64>(1) << shift));
}

// CPU `ratio_q16`: exact Q16 ratio with round-to-nearest, integer long
// division only.
AV2ENC_GME_HD GmeI64 gme_i128_ratio_q16(GmeI128 numerator,
                                               GmeI128 denominator) {
  if (denominator.hi == 0 && denominator.lo == 0) return 0;
  const bool negative = (numerator.hi < 0) != (denominator.hi < 0);
  const GmeU128 n = gme_i128_abs_u(numerator);
  const GmeU128 d = gme_i128_abs_u(denominator);
  GmeUDiv qr = gme_u128_divmod(n, d);
  GmeU128 integer = qr.q;
  GmeU128 remainder = qr.r;
  GmeU64 frac = 0;
  for (int bit = 0; bit < 16; ++bit) {
    remainder = gme_u128_shl1(remainder);
    frac <<= 1;
    if (!gme_u128_lt(remainder, d)) {
      remainder = gme_u128_sub(remainder, d);
      frac |= 1ULL;
    }
  }
  remainder = gme_u128_shl1(remainder);
  if (!gme_u128_lt(remainder, d)) ++frac;
  if (frac == (1ULL << 16)) {
    frac = 0;
    integer = gme_u128_add(integer, GmeU128{1ULL, 0ULL});
  }
  if (integer.hi != 0 || integer.lo > 32767ULL) {
    return negative ? gme_i32_min() : gme_i32_max();
  }
  GmeI64 result = static_cast<GmeI64>(integer.lo) * 65536 +
                  static_cast<GmeI64>(frac);
  if (negative) result = -result;
  return gme_clamp_i64(result, gme_i32_min(), gme_i32_max());
}

// ---------------------------------------------------------------------------
// §7.13.3.22 resolve divisor (also used by §5.18.9.1 scale_warp_model).
// ---------------------------------------------------------------------------

AV2ENC_GME_HD constexpr int gme_div_lut_entry(int f) {
  const int den = 128 + f;
  return (65536 + den / 2) / den;
}

struct GmeDivScaleShift {
  int shift;
  int factor;
};

AV2ENC_GME_HD GmeDivScaleShift gme_resolve_divisor(int d) {
  const int n = gme_floor_log2(d);
  const int e = d - (1 << n);
  const int f = (n > kGmeDivLutBits)
                    ? gme_round2(e, n - kGmeDivLutBits)
                    : (e << (kGmeDivLutBits - n));
  GmeDivScaleShift out;
  out.shift = n + kGmeDivLutPrecBits;
  out.factor = gme_div_lut_entry(f);
  return out;
}

// ---------------------------------------------------------------------------
// §7.13.3.21 setup shear validity, boolean form.
// ---------------------------------------------------------------------------

AV2ENC_GME_HD bool gme_setup_shear_valid(const GmeI32 p[6]) {
  if (p[2] < -(1 << 20) || p[2] > (1 << 20) || p[5] < -(1 << 20) ||
      p[5] > (1 << 20) || p[3] < -(1 << 20) || p[3] > (1 << 20) ||
      p[4] < -(1 << 20) || p[4] > (1 << 20)) {
    return false;
  }
  const int max_value = 32767 - (1 << (kGmeWarpParamReduceBits - 1));
  const GmeI64 one = kGmeGmOne;
  const GmeI64 alpha0 = static_cast<GmeI64>(p[2]) - one;
  const GmeI64 beta0 = p[3];
  const GmeDivScaleShift ds = gme_resolve_divisor(p[2] > 0 ? p[2] : 1);
  const GmeI64 v = static_cast<GmeI64>(p[4]) << kGmeWarpedModelPrecBits;
  const GmeI64 w = static_cast<GmeI64>(p[3]) * p[4];
  const GmeI64 gamma0 =
      static_cast<GmeI64>(p[5]) -
      gme_round2_signed64(w * ds.factor, ds.shift) - one;

  const int alpha = gme_round2_signed(
                        gme_clip3(-32768, max_value,
                                  static_cast<int>(gme_clamp_i64(
                                      alpha0, gme_i32_min(), gme_i32_max()))),
                        kGmeWarpParamReduceBits)
                    << kGmeWarpParamReduceBits;
  const int beta = gme_round2_signed(
                       gme_clip3(-32768, max_value,
                                 static_cast<int>(gme_clamp_i64(
                                     beta0, gme_i32_min(), gme_i32_max()))),
                       kGmeWarpParamReduceBits)
                   << kGmeWarpParamReduceBits;
  const int gamma = gme_round2_signed(
                        gme_clip3(-32768, max_value,
                                  static_cast<int>(
                                      gme_round2_signed64(v * ds.factor,
                                                          ds.shift))),
                        kGmeWarpParamReduceBits)
                    << kGmeWarpParamReduceBits;
  const int delta = gme_round2_signed(
                        gme_clip3(-32768, max_value,
                                  static_cast<int>(gme_clamp_i64(
                                      gamma0, gme_i32_min(), gme_i32_max()))),
                        kGmeWarpParamReduceBits)
                    << kGmeWarpParamReduceBits;

  if (4 * gme_abs_int(alpha) + 7 * gme_abs_int(beta) >=
      (3 << kGmeWarpedModelPrecBits)) {
    return false;
  }
  if (4 * gme_abs_int(gamma) + 4 * gme_abs_int(delta) >=
      (3 << kGmeWarpedModelPrecBits)) {
    return false;
  }
  return true;
}

// §7.13.3.23 reduce_warp_model.
AV2ENC_GME_HD void gme_reduce_warp_model_in_place(GmeI32 p[6]) {
  const int max_value =
      (1 << (kGmeWarpedModelPrecBits - 1)) - (1 << kGmeWarpParamReduceBits);
  const int min_value = -max_value;
  for (int i = 2; i < 6; ++i) {
    const int offset = (i == 2 || i == 5) ? kGmeGmOne : 0;
    const GmeI64 widened = static_cast<GmeI64>(p[i]) - offset;
    const int original =
        static_cast<int>(gme_clamp_i64(widened, gme_i32_min(), gme_i32_max()));
    const int clamped = gme_clip3(min_value, max_value, original);
    const int rounded =
        gme_round2_signed(clamped, kGmeWarpParamReduceBits)
        << kGmeWarpParamReduceBits;
    p[i] = rounded + offset;
  }
}

// ---------------------------------------------------------------------------
// Types shared with the public API.
// ---------------------------------------------------------------------------

struct GmeParams {
  GmeI32 type;
  GmeI32 p[6];
};

struct GmePoint {
  GmeI32 src_x;
  GmeI32 src_y;
  GmeI32 dst_x;
  GmeI32 dst_y;
  GmeI32 weight;
};

struct GmeBlockMv {
  GmeI32 x;
  GmeI32 y;
  GmeI32 mv_x;
  GmeI32 mv_y;
  GmeI32 weight;
};

struct GmeWarpSample {
  GmeI32 x;
  GmeI32 y;
  GmeI32 radius;
  GmeParams model;
  GmeI32 weight;
};

struct GmeOptions {
  GmeI32 model;
  GmeI32 min_points;
  GmeI32 max_points;
  GmeI32 outlier_passes;
  GmeI64 outlier_threshold_q8;
  GmeI32 allow_rotzoom_degrade;
  GmeI32 require_shear_valid;
  GmeI32 rotzoom_mismatch_q16;
  GmeI32 min_scale_q16;
  GmeI32 max_scale_q16;
  GmeI64 max_corner_displacement_q8;
  GmeI32 force_integer_mv;
  GmeI32 quantize;
  GmeI32 min_gain_percent;
  GmeI32 frame_width;
  GmeI32 frame_height;
};

struct GmeDiag {
  GmeI32 ok;
  GmeI32 accepted;
  GmeI32 codable;
  GmeI32 shear_valid;
  GmeI32 degraded;
  GmeI32 identity;
  GmeI32 points_in;
  GmeI32 points_used;
  GmeI32 outlier_passes_run;
  GmeI64 identity_sse_q8;
  GmeI64 model_sse_q8;
  double gain_db;
  double model_rmse_q8;
  double max_residual_q8;
  GmeI32 shear_mismatch_q16;
  double candidate_gain_db;
  GmeI64 candidate_model_sse_q8;
  GmeI32 reason;
  GmeI32 reserved;
};

struct GmeEstimate {
  GmeParams params;
  GmeDiag diag;
};

struct GmeQuality {
  GmeI64 identity_sse_q8;
  GmeI64 model_sse_q8;
  GmeI64 total_weight;
  double identity_rmse_q8;
  double model_rmse_q8;
  double gain_db;
  GmeI32 improves;
};

AV2ENC_GME_HD GmeParams gme_make_identity_params() {
  GmeParams out;
  out.type = kGmeIdentity;
  out.p[0] = 0;
  out.p[1] = 0;
  out.p[2] = kGmeGmOne;
  out.p[3] = 0;
  out.p[4] = 0;
  out.p[5] = kGmeGmOne;
  return out;
}

AV2ENC_GME_HD bool gme_is_identity_model(const GmeParams& params) {
  return params.p[0] == 0 && params.p[1] == 0 &&
         params.p[2] == kGmeGmOne && params.p[3] == 0 && params.p[4] == 0 &&
         params.p[5] == kGmeGmOne;
}

AV2ENC_GME_HD GmeDiag gme_make_diag() {
  GmeDiag out;
  out.ok = 0;
  out.accepted = 0;
  out.codable = 0;
  out.shear_valid = 1;
  out.degraded = 0;
  out.identity = 1;
  out.points_in = 0;
  out.points_used = 0;
  out.outlier_passes_run = 0;
  out.identity_sse_q8 = 0;
  out.model_sse_q8 = 0;
  out.gain_db = 0.0;
  out.model_rmse_q8 = 0.0;
  out.max_residual_q8 = 0.0;
  out.shear_mismatch_q16 = 0;
  out.candidate_gain_db = 0.0;
  out.candidate_model_sse_q8 = 0;
  out.reason = kGmeReasonNone;
  out.reserved = 0;
  return out;
}

AV2ENC_GME_HD GmeEstimate gme_make_identity_estimate() {
  GmeEstimate out;
  out.params = gme_make_identity_params();
  out.diag = gme_make_diag();
  return out;
}

// CPU `identity_estimate(reason)`: ok=false, codable=true, identity=true.
AV2ENC_GME_HD GmeEstimate gme_make_identity_estimate_reason(
    int reason) {
  GmeEstimate out = gme_make_identity_estimate();
  out.diag.codable = 1;
  out.diag.reason = reason;
  return out;
}

AV2ENC_GME_HD GmeOptions gme_default_options() {
  GmeOptions out;
  out.model = kGmeFitAffine;
  out.min_points = 3;
  out.max_points = 8192;
  out.outlier_passes = 2;
  out.outlier_threshold_q8 = 32;
  out.allow_rotzoom_degrade = 1;
  out.require_shear_valid = 0;
  out.rotzoom_mismatch_q16 = 1 << 12;
  out.min_scale_q16 = kGmeGmOne / 3;
  out.max_scale_q16 = 3 * kGmeGmOne;
  out.max_corner_displacement_q8 = 2048LL * 8;
  out.force_integer_mv = 0;
  out.quantize = 1;
  out.min_gain_percent = 1;
  out.frame_width = 0;
  out.frame_height = 0;
  return out;
}

AV2ENC_GME_HD const char* gme_reason_name(int reason) {
  switch (reason) {
    case kGmeReasonNone:
      return "";
    case kGmeReasonNoPoints:
      return "no points";
    case kGmeReasonTooFewWeighted:
      return "too few weighted points";
    case kGmeReasonDegenerate:
      return "degenerate point distribution";
    case kGmeReasonCollinear:
      return "collinear points (det <= 0)";
    case kGmeReasonTooFewCorrespondences:
      return "too few correspondences";
    case kGmeReasonTooManyCorrespondences:
      return "too many correspondences";
    case kGmeReasonBlockMvOverflow:
      return "block MV correspondence exceeds int32";
    case kGmeReasonWarpOverflow:
      return "warp sample coordinate exceeds int32";
    case kGmeReasonIdentityNonDefault:
      return "IDENTITY with non-default parameters";
    case kGmeReasonTransGrid:
      return "translation not on the 1/8-pel grid";
    case kGmeReasonTransRange:
      return "translation outside GM_TRANS range";
    case kGmeReasonAlphaGrid:
      return "alpha not on the 1/1024 grid";
    case kGmeReasonAlphaRange:
      return "alpha outside GM_ALPHA range";
    case kGmeReasonRotzoomSymmetry:
      return "ROTZOOM requires p4 = -p3 and p5 = p2";
    case kGmeReasonUnknownType:
      return "unknown GmType";
    case kGmeReasonScaleOutOfRange:
      return "model scale/determinant out of range";
    case kGmeReasonShearInvalid:
      return "shear model is not valid";
    case kGmeReasonCornerOutOfRange:
      return "corner projection out of range";
    case kGmeReasonAffineToRotzoom:
      return "affine close to similarity; signalled ROTZOOM";
    case kGmeReasonDegradedAfterReject:
      return "degraded to ROTZOOM after AFFINE rejection";
    case kGmeReasonModelRejected:
      return "model rejected";
    case kGmeReasonGainBelowThreshold:
      return "global motion gain below threshold";
    default:
      return "unknown reason";
  }
}

// ---------------------------------------------------------------------------
// Projection and quality proxy helpers.
// ---------------------------------------------------------------------------

AV2ENC_GME_HD GmeI32 gme_sat_i32_from_i128(GmeI128 value) {
  return gme_i128_sat_i32(value);
}

AV2ENC_GME_HD void gme_project_gm_point(const GmeParams& params,
                                               GmeI32 x_q16, GmeI32 y_q16,
                                               GmeI32* ref_x_q16,
                                               GmeI32* ref_y_q16) {
  const GmeI128 x = gme_i128_from_i64(x_q16);
  const GmeI128 y = gme_i128_from_i64(y_q16);
  const GmeI128 rx = gme_i128_add(
      gme_i128_shr(gme_i128_add(gme_i128_mul_i64(x, params.p[2]),
                                gme_i128_mul_i64(y, params.p[3])),
                   16),
      gme_i128_from_i64(params.p[0]));
  const GmeI128 ry = gme_i128_add(
      gme_i128_shr(gme_i128_add(gme_i128_mul_i64(x, params.p[4]),
                                gme_i128_mul_i64(y, params.p[5])),
                   16),
      gme_i128_from_i64(params.p[1]));
  if (ref_x_q16 != nullptr) *ref_x_q16 = gme_sat_i32_from_i128(rx);
  if (ref_y_q16 != nullptr) *ref_y_q16 = gme_sat_i32_from_i128(ry);
}

AV2ENC_GME_HD GmeI64 gme_project_x_q8(const GmeParams& params,
                                             GmeI64 x_q8, GmeI64 y_q8) {
  const GmeI128 x_q16 = gme_i128_mul_i64(gme_i128_from_i64(x_q8), 8192);
  const GmeI128 y_q16 = gme_i128_mul_i64(gme_i128_from_i64(y_q8), 8192);
  const GmeI128 value = gme_i128_shr(
      gme_i128_add(gme_i128_mul(gme_i128_from_i64(params.p[2]), x_q16),
                   gme_i128_mul(gme_i128_from_i64(params.p[3]), y_q16)),
      16);
  return gme_i128_to_i64_sat(gme_i128_shr(
      gme_i128_add(value, gme_i128_from_i64(params.p[0])), 13));
}

AV2ENC_GME_HD GmeI64 gme_project_y_q8(const GmeParams& params,
                                             GmeI64 x_q8, GmeI64 y_q8) {
  const GmeI128 x_q16 = gme_i128_mul_i64(gme_i128_from_i64(x_q8), 8192);
  const GmeI128 y_q16 = gme_i128_mul_i64(gme_i128_from_i64(y_q8), 8192);
  const GmeI128 value = gme_i128_shr(
      gme_i128_add(gme_i128_mul(gme_i128_from_i64(params.p[4]), x_q16),
                   gme_i128_mul(gme_i128_from_i64(params.p[5]), y_q16)),
      16);
  return gme_i128_to_i64_sat(gme_i128_shr(
      gme_i128_add(value, gme_i128_from_i64(params.p[1])), 13));
}

AV2ENC_GME_HD GmeI64 gme_point_residual(const GmeParams& params,
                                               const GmePoint& point) {
  const GmeI128 px = gme_i128_from_i64(
      gme_project_x_q8(params, point.src_x, point.src_y));
  const GmeI128 py = gme_i128_from_i64(
      gme_project_y_q8(params, point.src_x, point.src_y));
  const GmeI128 dx = gme_i128_sub(px, gme_i128_from_i64(point.dst_x));
  const GmeI128 dy = gme_i128_sub(py, gme_i128_from_i64(point.dst_y));
  const GmeI128 sse = gme_i128_add(gme_i128_mul(dx, dx),
                                   gme_i128_mul(dy, dy));
  return gme_i128_to_i64_sat(sse);
}

// CPU `lower_mv_precision`.
AV2ENC_GME_HD int gme_lower_mv_precision_value(int value,
                                                      GmeMvPrecision precision) {
  const int bits = 7 - static_cast<int>(precision);
  if (bits <= 0) return value;
  const int radix = 1 << bits;
  const int a = gme_abs_int(value);
  const int a_int = (a + (1 << (bits - 1)) - 1) >> bits;
  int result = value >= 0 ? (a_int << bits) : (-a_int << bits);
  if ((a_int << bits) != a) {
    result = gme_clip3(kGmeMvLow + radix, kGmeMvUpp - radix, result);
  }
  return result;
}

// §7.12.2.2 get_warp_motion_vector, including the §5.20.9.4/.5 border clamp
// and lower_mv_precision().
AV2ENC_GME_HD void gme_get_warp_motion_vector(
    const GmeParams& params, int mi_col, int mi_row, int bw4, int bh4,
    int mi_cols, int mi_rows, GmeMvPrecision precision, int mv[2]) {
  const int bw = bw4 * kGmeMiSize;
  const int bh = bh4 * kGmeMiSize;
  const int x = mi_col * kGmeMiSize + bw / 2 - 1;
  const int y = mi_row * kGmeMiSize + bh / 2 - 1;
  const GmeI64 xc = static_cast<GmeI64>(params.p[2] - kGmeGmOne) * x +
                    static_cast<GmeI64>(params.p[3]) * y + params.p[0];
  const GmeI64 yc = static_cast<GmeI64>(params.p[4]) * x +
                    static_cast<GmeI64>(params.p[5] - kGmeGmOne) * y +
                    params.p[1];
  GmeI64 mv_y = 0;
  GmeI64 mv_x = 0;
  if (precision == kGmeEighthPel) {
    mv_y = gme_round2_signed64(yc, kGmeWarpedModelPrecBits - 3);
    mv_x = gme_round2_signed64(xc, kGmeWarpedModelPrecBits - 3);
  } else {
    mv_y = gme_round2_signed64(yc, kGmeWarpedModelPrecBits - 2) * 2;
    mv_x = gme_round2_signed64(xc, kGmeWarpedModelPrecBits - 2) * 2;
  }
  mv_y = gme_clip3(kGmeMvLow + 1, kGmeMvUpp - 1,
                   static_cast<int>(gme_clamp_i64(mv_y, gme_i32_min(),
                                                   gme_i32_max())));
  mv_x = gme_clip3(kGmeMvLow + 1, kGmeMvUpp - 1,
                   static_cast<int>(gme_clamp_i64(mv_x, gme_i32_min(),
                                                   gme_i32_max())));
  const int low_row = -(mi_row + bh4) * kGmeMiSize * 8 - kGmeMvBorder;
  const int high_row = (mi_rows - mi_row) * kGmeMiSize * 8 + kGmeMvBorder;
  const int low_col = -(mi_col + bw4) * kGmeMiSize * 8 - kGmeMvBorder;
  const int high_col = (mi_cols - mi_col) * kGmeMiSize * 8 + kGmeMvBorder;
  mv_y = gme_clip3(low_row, high_row, static_cast<int>(mv_y));
  mv_x = gme_clip3(low_col, high_col, static_cast<int>(mv_x));
  if (precision < kGmeHalfPel) {
    mv_y = gme_lower_mv_precision_value(static_cast<int>(mv_y), precision);
    mv_x = gme_lower_mv_precision_value(static_cast<int>(mv_x), precision);
  }
  mv[0] = static_cast<int>(mv_y);
  mv[1] = static_cast<int>(mv_x);
}

struct GmeGlobalMvUse {
  GmeI32 usable;
  GmeI32 use_warp;
  GmeI32 ref;
  GmeI32 mv[2];
};

AV2ENC_GME_HD GmeGlobalMvUse gme_global_mv_for_block(
    const GmeParams& params, int ref, int mi_col, int mi_row, int bw4,
    int bh4, int mi_cols, int mi_rows) {
  GmeGlobalMvUse out;
  out.ref = ref;
  out.usable = params.type != kGmeIdentity ? 1 : 0;
  out.use_warp = 0;
  out.mv[0] = 0;
  out.mv[1] = 0;
  if (!out.usable) return out;
  const int bw = bw4 * kGmeMiSize;
  const int bh = bh4 * kGmeMiSize;
  out.use_warp = gme_min_i64(bw, bh) >= 8 ? 1 : 0;
  gme_get_warp_motion_vector(params, mi_col, mi_row, bw4, bh4, mi_cols,
                             mi_rows, kGmeEighthPel, out.mv);
  return out;
}

// ---------------------------------------------------------------------------
// Quantisation, codability, validation and degradation.
// ---------------------------------------------------------------------------

AV2ENC_GME_HD GmeI32 gme_quantize_step(GmeI64 value, GmeI64 step,
                                              GmeI64 min_value,
                                              GmeI64 max_value) {
  GmeI64 quantised = gme_i128_div_round(
                         gme_i128_from_i64(value),
                         gme_i128_from_i64(step)) *
                     step;
  quantised = gme_clamp_i64(quantised, min_value, max_value);
  return static_cast<GmeI32>(quantised);
}

AV2ENC_GME_HD GmeParams gme_quantize_model(const GmeParams& input,
                                                  const GmeOptions& options) {
  GmeParams out = input;
  const GmeI64 trans_step = options.force_integer_mv ? kGmeGmOne
                                                     : kGmeGmTransStep;
  const GmeI64 trans_raw_limit =
      static_cast<GmeI64>(kGmeGmTransMax) * kGmeGmTransStep;
  const GmeI64 trans_limit = (trans_raw_limit / trans_step) * trans_step;
  out.p[0] = gme_quantize_step(input.p[0], trans_step, -trans_limit,
                               trans_limit);
  out.p[1] = gme_quantize_step(input.p[1], trans_step, -trans_limit,
                               trans_limit);

  const GmeI64 alpha_limit =
      static_cast<GmeI64>(kGmeGmAlphaMax) * kGmeGmAlphaStep;
  if (out.type == kGmeRotZoom) {
    const GmeI64 a =
        (static_cast<GmeI64>(input.p[2]) + input.p[5]) / 2;
    const GmeI64 b =
        (static_cast<GmeI64>(input.p[3]) - input.p[4]) / 2;
    const GmeI32 qa =
        gme_quantize_step(a - kGmeGmOne, kGmeGmAlphaStep, -alpha_limit,
                          alpha_limit) +
        kGmeGmOne;
    const GmeI32 qb = gme_quantize_step(b, kGmeGmAlphaStep, -alpha_limit,
                                        alpha_limit);
    out.p[2] = qa;
    out.p[3] = qb;
    out.p[4] = -qb;
    out.p[5] = qa;
  } else if (out.type == kGmeAffine) {
    out.p[2] = gme_quantize_step(static_cast<GmeI64>(input.p[2]) - kGmeGmOne,
                                 kGmeGmAlphaStep, -alpha_limit,
                                 alpha_limit) +
               kGmeGmOne;
    out.p[3] = gme_quantize_step(input.p[3], kGmeGmAlphaStep, -alpha_limit,
                                 alpha_limit);
    out.p[4] = gme_quantize_step(input.p[4], kGmeGmAlphaStep, -alpha_limit,
                                 alpha_limit);
    out.p[5] = gme_quantize_step(static_cast<GmeI64>(input.p[5]) - kGmeGmOne,
                                 kGmeGmAlphaStep, -alpha_limit,
                                 alpha_limit) +
               kGmeGmOne;
  } else {
    out = gme_make_identity_params();
  }
  if (gme_is_identity_model(out)) out.type = kGmeIdentity;
  return out;
}

AV2ENC_GME_HD bool gme_model_scale_ok(const GmeParams& params,
                                             const GmeOptions& options) {
  if (params.type == kGmeIdentity) return true;
  if (params.p[2] < options.min_scale_q16 ||
      params.p[2] > options.max_scale_q16) {
    return false;
  }
  if (params.type == kGmeAffine) {
    if (params.p[5] < options.min_scale_q16 ||
        params.p[5] > options.max_scale_q16) {
      return false;
    }
    const GmeI64 det = static_cast<GmeI64>(params.p[2]) * params.p[5] -
                       static_cast<GmeI64>(params.p[3]) * params.p[4];
    if (det <= 0) return false;
  }
  return true;
}

AV2ENC_GME_HD bool gme_corner_projection_ok(
    const GmeParams& params, const GmeOptions& options) {
  if (options.frame_width <= 0 || options.frame_height <= 0) return true;
  const GmeI64 corners[4][2] = {
      {0, 0},
      {static_cast<GmeI64>(options.frame_width - 1) << 16, 0},
      {0, static_cast<GmeI64>(options.frame_height - 1) << 16},
      {static_cast<GmeI64>(options.frame_width - 1) << 16,
       static_cast<GmeI64>(options.frame_height - 1) << 16},
  };
  for (int index = 0; index < 4; ++index) {
    GmeI32 rx = 0;
    GmeI32 ry = 0;
    gme_project_gm_point(params,
                         gme_i128_sat_i32(gme_i128_from_i64(corners[index][0])),
                         gme_i128_sat_i32(gme_i128_from_i64(corners[index][1])),
                         &rx, &ry);
    const GmeI64 dx = static_cast<GmeI64>(rx) - corners[index][0];
    const GmeI64 dy = static_cast<GmeI64>(ry) - corners[index][1];
    const GmeI64 limit = options.max_corner_displacement_q8 * 8192;
    if (gme_abs_i64(dx) > limit || gme_abs_i64(dy) > limit) return false;
  }
  return true;
}

AV2ENC_GME_HD bool gme_params_codable(const GmeParams& params,
                                             int* reason) {
  for (int i = 0; i < 6; ++i) {
    const GmeI32 value = params.p[i];
    if (i < 2) {
      if ((value % kGmeGmTransStep) != 0) {
        if (reason != nullptr) *reason = kGmeReasonTransGrid;
        return false;
      }
      const GmeI64 limit =
          static_cast<GmeI64>(kGmeGmTransMax) * kGmeGmTransStep;
      if (value < -limit || value > limit) {
        if (reason != nullptr) *reason = kGmeReasonTransRange;
        return false;
      }
    } else {
      if ((value % kGmeGmAlphaStep) != 0) {
        if (reason != nullptr) *reason = kGmeReasonAlphaGrid;
        return false;
      }
      const GmeI64 offset =
          (i == 2 || i == 5) ? static_cast<GmeI64>(value) - kGmeGmOne
                             : static_cast<GmeI64>(value);
      const GmeI64 limit =
          static_cast<GmeI64>(kGmeGmAlphaMax) * kGmeGmAlphaStep;
      if (offset < -limit || offset > limit) {
        if (reason != nullptr) *reason = kGmeReasonAlphaRange;
        return false;
      }
    }
  }
  if (params.type == kGmeIdentity) {
    if (!gme_is_identity_model(params)) {
      if (reason != nullptr) *reason = kGmeReasonIdentityNonDefault;
      return false;
    }
    return true;
  }
  if (params.type == kGmeRotZoom) {
    if (params.p[4] != -params.p[3] || params.p[5] != params.p[2]) {
      if (reason != nullptr) *reason = kGmeReasonRotzoomSymmetry;
      return false;
    }
  } else if (params.type != kGmeAffine) {
    if (reason != nullptr) *reason = kGmeReasonUnknownType;
    return false;
  }
  return true;
}

AV2ENC_GME_HD bool gme_validate_gm_params(const GmeParams& params,
                                                 const GmeOptions& options,
                                                 GmeDiag* diag) {
  GmeDiag local = gme_make_diag();
  local.ok = 1;
  local.identity = gme_is_identity_model(params) ? 1 : 0;
  int reason = kGmeReasonNone;
  local.codable = gme_params_codable(params, &reason) ? 1 : 0;
  local.shear_valid = gme_setup_shear_valid(params.p) ? 1 : 0;
  local.shear_mismatch_q16 =
      static_cast<GmeI32>(gme_max_i64(gme_abs_int(params.p[4] + params.p[3]),
                                      gme_abs_int(params.p[5] - params.p[2])));
  if (!local.codable) {
    local.accepted = 0;
    local.reason = reason;
  } else if (local.identity) {
    local.accepted = 1;
  } else if (!gme_model_scale_ok(params, options)) {
    local.accepted = 0;
    local.reason = kGmeReasonScaleOutOfRange;
  } else if (options.require_shear_valid && !local.shear_valid) {
    local.accepted = 0;
    local.reason = kGmeReasonShearInvalid;
  } else if (!gme_corner_projection_ok(params, options)) {
    local.accepted = 0;
    local.reason = kGmeReasonCornerOutOfRange;
  } else {
    local.accepted = 1;
  }
  if (diag != nullptr) *diag = local;
  return local.accepted != 0;
}

AV2ENC_GME_HD GmeEstimate gme_finalize_once(
    const GmeParams& candidate, const GmeOptions& options, bool degraded,
    int reason) {
  GmeEstimate out;
  out.params = options.quantize ? gme_quantize_model(candidate, options)
                                : candidate;
  out.diag = gme_make_diag();
  out.diag.degraded = degraded ? 1 : 0;
  out.diag.reason = reason;
  out.diag.identity = gme_is_identity_model(out.params) ? 1 : 0;
  out.diag.shear_valid = gme_setup_shear_valid(out.params.p) ? 1 : 0;
  out.diag.shear_mismatch_q16 = static_cast<GmeI32>(
      gme_max_i64(gme_abs_int(out.params.p[3] + out.params.p[4]),
                  gme_abs_int(out.params.p[5] - out.params.p[2])));
  out.diag.codable =
      gme_params_codable(out.params, &out.diag.reason) ? 1 : 0;
  out.diag.ok = 1;
  if (out.diag.identity) {
    out.diag.accepted = 1;
    out.diag.codable = 1;
    return out;
  }
  if (!out.diag.codable) {
    out.diag.accepted = 0;
    return out;
  }
  if (!gme_model_scale_ok(out.params, options)) {
    out.diag.accepted = 0;
    out.diag.reason = kGmeReasonScaleOutOfRange;
    return out;
  }
  if (options.require_shear_valid && !out.diag.shear_valid) {
    out.diag.accepted = 0;
    out.diag.reason = kGmeReasonShearInvalid;
    return out;
  }
  if (!gme_corner_projection_ok(out.params, options)) {
    out.diag.accepted = 0;
    out.diag.reason = kGmeReasonCornerOutOfRange;
    return out;
  }
  out.diag.accepted = 1;
  return out;
}

AV2ENC_GME_HD GmeEstimate gme_finalize_gm_params(
    const GmeParams& raw, const GmeOptions& options) {
  GmeEstimate out;
  GmeParams candidate = raw;
  bool degraded = false;
  int reason = kGmeReasonNone;
  if (candidate.type == kGmeAffine) {
    const GmeI32 mismatch = static_cast<GmeI32>(
        gme_max_i64(gme_abs_int(candidate.p[4] + candidate.p[3]),
                    gme_abs_int(candidate.p[5] - candidate.p[2])));
    if (options.allow_rotzoom_degrade &&
        mismatch <= options.rotzoom_mismatch_q16) {
      const GmeI64 a =
          (static_cast<GmeI64>(candidate.p[2]) + candidate.p[5]) / 2;
      const GmeI64 b =
          (static_cast<GmeI64>(candidate.p[3]) - candidate.p[4]) / 2;
      candidate.type = kGmeRotZoom;
      candidate.p[2] = static_cast<GmeI32>(a);
      candidate.p[3] = static_cast<GmeI32>(b);
      candidate.p[4] = static_cast<GmeI32>(-b);
      candidate.p[5] = static_cast<GmeI32>(a);
      degraded = true;
      reason = kGmeReasonAffineToRotzoom;
    }
  }
  if (candidate.type == kGmeRotZoom) {
    const GmeI64 a =
        (static_cast<GmeI64>(candidate.p[2]) + candidate.p[5]) / 2;
    const GmeI64 b =
        (static_cast<GmeI64>(candidate.p[3]) - candidate.p[4]) / 2;
    candidate.p[2] = static_cast<GmeI32>(a);
    candidate.p[3] = static_cast<GmeI32>(b);
    candidate.p[4] = static_cast<GmeI32>(-b);
    candidate.p[5] = static_cast<GmeI32>(a);
  }

  out = gme_finalize_once(candidate, options, degraded, reason);
  if (out.diag.accepted) return out;

  if (raw.type == kGmeAffine && options.allow_rotzoom_degrade) {
    GmeParams rot = raw;
    const GmeI64 a = (static_cast<GmeI64>(raw.p[2]) + raw.p[5]) / 2;
    const GmeI64 b = (static_cast<GmeI64>(raw.p[3]) - raw.p[4]) / 2;
    rot.type = kGmeRotZoom;
    rot.p[2] = static_cast<GmeI32>(a);
    rot.p[3] = static_cast<GmeI32>(b);
    rot.p[4] = static_cast<GmeI32>(-b);
    rot.p[5] = static_cast<GmeI32>(a);
    GmeEstimate rot_out = gme_finalize_once(
        rot, options, true, kGmeReasonDegradedAfterReject);
    if (rot_out.diag.accepted) {
      rot_out.diag.points_in = out.diag.points_in;
      rot_out.diag.points_used = out.diag.points_used;
      return rot_out;
    }
  }

  GmeEstimate identity = gme_finalize_once(
      gme_make_identity_params(), options, true,
      out.diag.reason != kGmeReasonNone ? out.diag.reason
                                        : kGmeReasonModelRejected);
  identity.diag.ok = out.diag.ok;
  identity.diag.points_in = out.diag.points_in;
  identity.diag.points_used = out.diag.points_used;
  identity.diag.degraded = 1;
  return identity;
}

// ---------------------------------------------------------------------------
// Weighted quality proxy.
// ---------------------------------------------------------------------------

AV2ENC_GME_HD double gme_sqrt(double value) {
#if defined(__CUDACC__)
  return ::sqrt(value);
#else
  return value <= 0.0 ? 0.0 : __builtin_sqrt(value);
#endif
}

AV2ENC_GME_HD double gme_log10(double value) {
#if defined(__CUDACC__)
  return ::log10(value);
#else
  return __builtin_log10(value);
#endif
}

AV2ENC_GME_HD GmeQuality gme_evaluate_quality(
    const GmeParams& params, const GmePoint* points, int count) {
  GmeI128 identity_sse = gme_i128_zero();
  GmeI128 model_sse = gme_i128_zero();
  GmeI128 weight = gme_i128_zero();
  for (int i = 0; i < count; ++i) {
    const GmePoint& point = points[i];
    if (point.weight <= 0) continue;
    const GmeI128 w = gme_i128_from_i64(point.weight);
    const GmeI128 idx =
        gme_i128_sub(gme_i128_from_i64(point.dst_x),
                     gme_i128_from_i64(point.src_x));
    const GmeI128 idy =
        gme_i128_sub(gme_i128_from_i64(point.dst_y),
                     gme_i128_from_i64(point.src_y));
    identity_sse = gme_i128_add(
        identity_sse,
        gme_i128_mul(w, gme_i128_add(gme_i128_mul(idx, idx),
                                     gme_i128_mul(idy, idy))));
    const GmeI128 px = gme_i128_from_i64(
        gme_project_x_q8(params, point.src_x, point.src_y));
    const GmeI128 py = gme_i128_from_i64(
        gme_project_y_q8(params, point.src_x, point.src_y));
    const GmeI128 dx = gme_i128_sub(px, gme_i128_from_i64(point.dst_x));
    const GmeI128 dy = gme_i128_sub(py, gme_i128_from_i64(point.dst_y));
    model_sse = gme_i128_add(
        model_sse,
        gme_i128_mul(w, gme_i128_add(gme_i128_mul(dx, dx),
                                     gme_i128_mul(dy, dy))));
    weight = gme_i128_add(weight, w);
  }
  GmeQuality quality;
  quality.identity_sse_q8 = gme_i128_to_i64_sat(identity_sse);
  quality.model_sse_q8 = gme_i128_to_i64_sat(model_sse);
  quality.total_weight = gme_i128_to_i64_sat(weight);
  const double w = quality.total_weight > 0
                       ? static_cast<double>(quality.total_weight)
                       : 1.0;
  quality.identity_rmse_q8 =
      gme_sqrt(static_cast<double>(quality.identity_sse_q8) / w);
  quality.model_rmse_q8 =
      gme_sqrt(static_cast<double>(quality.model_sse_q8) / w);
  quality.improves = quality.model_sse_q8 < quality.identity_sse_q8 ? 1 : 0;
  quality.gain_db = 0.0;
  if (quality.model_sse_q8 > 0 && quality.identity_sse_q8 > 0) {
    quality.gain_db =
        10.0 * gme_log10(static_cast<double>(quality.identity_sse_q8) /
                         static_cast<double>(quality.model_sse_q8));
  } else if (quality.model_sse_q8 == 0) {
    quality.gain_db = 99.0;
  }
  return quality;
}

// CPU `median_of` is nth_element(...)[size/2]; this binary selection returns
// the same order statistic for non-negative residuals without sorting.
AV2ENC_GME_HD GmeI64 gme_kth_smallest(const GmeI64* values, int count,
                                             int k) {
  if (count <= 0) return 0;
  if (k < 0) k = 0;
  if (k >= count) k = count - 1;
  GmeU64 result = 0;
  int remaining = k;
  for (int bit = 62; bit >= 0; --bit) {
    int count_zero = 0;
    for (int i = 0; i < count; ++i) {
      const GmeU64 value = static_cast<GmeU64>(values[i]);
      if ((value >> (bit + 1)) == (result >> (bit + 1))) {
        if (((value >> bit) & 1ULL) == 0) ++count_zero;
      }
    }
    if (remaining < count_zero) {
      // Current bit is 0.
    } else {
      result |= (1ULL << bit);
      remaining -= count_zero;
    }
  }
  return static_cast<GmeI64>(result);
}

// ---------------------------------------------------------------------------
// Correspondence extraction (CPU fit_gm_block_mvs / fit_gm_warps).
// Returns 1/0 for "written/skipped" and -1 for the CPU's int32 overflow path.
// ---------------------------------------------------------------------------

AV2ENC_GME_HD int gme_extract_point_from_block(const GmeBlockMv& block,
                                                      GmePoint* out) {
  if (block.weight <= 0) return 0;
  const GmeI64 dst_x = static_cast<GmeI64>(block.x) + block.mv_x;
  const GmeI64 dst_y = static_cast<GmeI64>(block.y) + block.mv_y;
  if (dst_x < gme_i32_min() || dst_x > gme_i32_max() ||
      dst_y < gme_i32_min() || dst_y > gme_i32_max()) {
    return -1;
  }
  out->src_x = block.x;
  out->src_y = block.y;
  out->dst_x = static_cast<GmeI32>(dst_x);
  out->dst_y = static_cast<GmeI32>(dst_y);
  out->weight = block.weight;
  return 1;
}

AV2ENC_GME_HD int gme_extract_points_from_warp(
    const GmeWarpSample& warp, GmePoint out[5]) {
  if (warp.weight <= 0) return 0;
  const GmeI32 base = warp.weight / 5;
  GmeI32 remainder = warp.weight % 5;
  const GmeI32 offsets[5][2] = {
      {0, 0},
      {warp.radius, 0},
      {-warp.radius, 0},
      {0, warp.radius},
      {0, -warp.radius},
  };
  for (int index = 0; index < 5; ++index) {
    const GmeI64 sx = static_cast<GmeI64>(warp.x) + offsets[index][0];
    const GmeI64 sy = static_cast<GmeI64>(warp.y) + offsets[index][1];
    if (sx < gme_i32_min() || sx > gme_i32_max() ||
        sy < gme_i32_min() || sy > gme_i32_max()) {
      return -1;
    }
    GmeI32 rx = 0;
    GmeI32 ry = 0;
    // The CPU reference passes `(int32)t * 8192` (q8 -> q16) to
    // project_gm_point; coordinates in a coded frame are far below the int32
    // wrap point, so the cast matches the reference.
    gme_project_gm_point(
        warp.model,
        static_cast<GmeI32>(sx * 8192),
        static_cast<GmeI32>(sy * 8192), &rx, &ry);
    GmePoint point;
    point.src_x = static_cast<GmeI32>(sx);
    point.src_y = static_cast<GmeI32>(sy);
    point.dst_x = static_cast<GmeI32>(static_cast<GmeI64>(rx) >> 13);
    point.dst_y = static_cast<GmeI32>(static_cast<GmeI64>(ry) >> 13);
    point.weight = base > 0 ? base : 1;
    if (remainder > 0) {
      point.weight += 1;
      --remainder;
    }
    out[index] = point;
  }
  return 5;
}

// ---------------------------------------------------------------------------
// Integer least squares.
// ---------------------------------------------------------------------------

struct GmeFitCore {
  GmeParams params;
  GmeI32 ok;
  GmeI32 reason;
  GmeI32 points_used;
};

AV2ENC_GME_HD GmeFitCore gme_fit_core(const GmePoint* points,
                                             int count, GmeFitModel model,
                                             int min_points) {
  GmeFitCore out;
  out.params = gme_make_identity_params();
  out.ok = 0;
  out.reason = kGmeReasonNone;
  out.points_used = 0;
  if (count <= 0) {
    out.reason = kGmeReasonNoPoints;
    return out;
  }

  GmeI128 weight_sum = gme_i128_zero();
  GmeI128 sum_x = gme_i128_zero();
  GmeI128 sum_y = gme_i128_zero();
  GmeI128 sum_u = gme_i128_zero();
  GmeI128 sum_v = gme_i128_zero();
  int used = 0;
  GmeI64 max_abs = 0;
  for (int i = 0; i < count; ++i) {
    const GmePoint& point = points[i];
    if (point.weight <= 0) continue;
    const GmeI128 w = gme_i128_from_i64(point.weight);
    weight_sum = gme_i128_add(weight_sum, w);
    sum_x = gme_i128_add(sum_x, gme_i128_mul_i64(w, point.src_x));
    sum_y = gme_i128_add(sum_y, gme_i128_mul_i64(w, point.src_y));
    sum_u = gme_i128_add(sum_u, gme_i128_mul_i64(w, point.dst_x));
    sum_v = gme_i128_add(sum_v, gme_i128_mul_i64(w, point.dst_y));
    max_abs = gme_max_i64(
        max_abs,
        gme_max_i64(gme_max_i64(gme_abs_i64(point.src_x),
                                gme_abs_i64(point.src_y)),
                    gme_max_i64(gme_abs_i64(point.dst_x),
                                gme_abs_i64(point.dst_y))));
    ++used;
  }
  if (used < min_points || gme_i128_cmp(weight_sum, gme_i128_zero()) <= 0) {
    out.reason = kGmeReasonTooFewWeighted;
    return out;
  }

  const GmeI64 mean_x = gme_i128_div_round(sum_x, weight_sum);
  const GmeI64 mean_y = gme_i128_div_round(sum_y, weight_sum);
  const GmeI64 mean_u = gme_i128_div_round(sum_u, weight_sum);
  const GmeI64 mean_v = gme_i128_div_round(sum_v, weight_sum);

  int shift = 0;
  while ((max_abs >> shift) > 16384) ++shift;

  GmeI128 sxx = gme_i128_zero();
  GmeI128 sxy = gme_i128_zero();
  GmeI128 syy = gme_i128_zero();
  GmeI128 sxu = gme_i128_zero();
  GmeI128 syu = gme_i128_zero();
  GmeI128 sxv = gme_i128_zero();
  GmeI128 syv = gme_i128_zero();
  GmeI128 den = gme_i128_zero();
  GmeI128 num_a = gme_i128_zero();
  GmeI128 num_b = gme_i128_zero();

  for (int i = 0; i < count; ++i) {
    const GmePoint& point = points[i];
    if (point.weight <= 0) continue;
    const GmeI128 w = gme_i128_from_i64(point.weight);
    const GmeI64 cx =
        (static_cast<GmeI64>(point.src_x) - mean_x) >> shift;
    const GmeI64 cy =
        (static_cast<GmeI64>(point.src_y) - mean_y) >> shift;
    const GmeI64 cu =
        (static_cast<GmeI64>(point.dst_x) - mean_u) >> shift;
    const GmeI64 cv =
        (static_cast<GmeI64>(point.dst_y) - mean_v) >> shift;
    const GmeI128 x = gme_i128_from_i64(cx);
    const GmeI128 y = gme_i128_from_i64(cy);
    const GmeI128 u = gme_i128_from_i64(cu);
    const GmeI128 v = gme_i128_from_i64(cv);
    if (model != kGmeFitTranslation) {
      sxx = gme_i128_add(sxx, gme_i128_mul(w, gme_i128_mul(x, x)));
      sxy = gme_i128_add(sxy, gme_i128_mul(w, gme_i128_mul(x, y)));
      syy = gme_i128_add(syy, gme_i128_mul(w, gme_i128_mul(y, y)));
      sxu = gme_i128_add(sxu, gme_i128_mul(w, gme_i128_mul(x, u)));
      syu = gme_i128_add(syu, gme_i128_mul(w, gme_i128_mul(y, u)));
      sxv = gme_i128_add(sxv, gme_i128_mul(w, gme_i128_mul(x, v)));
      syv = gme_i128_add(syv, gme_i128_mul(w, gme_i128_mul(y, v)));
      den = gme_i128_add(
          den, gme_i128_mul(w, gme_i128_add(gme_i128_mul(x, x),
                                            gme_i128_mul(y, y))));
      num_a = gme_i128_add(
          num_a, gme_i128_mul(w, gme_i128_add(gme_i128_mul(x, u),
                                              gme_i128_mul(y, v))));
      num_b = gme_i128_add(
          num_b, gme_i128_mul(w, gme_i128_sub(gme_i128_mul(x, v),
                                              gme_i128_mul(y, u))));
    }
  }

  GmeI64 a = kGmeGmOne;
  GmeI64 b = 0;
  GmeI64 c = 0;
  GmeI64 d = kGmeGmOne;

  if (model == kGmeFitRotZoom) {
    if (gme_i128_is_zero(den)) {
      out.reason = kGmeReasonDegenerate;
      return out;
    }
    a = gme_i128_ratio_q16(num_a, den);
    b = gme_i128_ratio_q16(num_b, den);
    c = -b;
    d = a;
  } else if (model == kGmeFitAffine) {
    const GmeI128 det = gme_i128_sub(gme_i128_mul(sxx, syy),
                                     gme_i128_mul(sxy, sxy));
    if (gme_i128_cmp(det, gme_i128_zero()) <= 0) {
      out.reason = kGmeReasonCollinear;
      return out;
    }
    a = gme_i128_ratio_q16(
        gme_i128_sub(gme_i128_mul(sxu, syy), gme_i128_mul(syu, sxy)), det);
    b = gme_i128_ratio_q16(
        gme_i128_sub(gme_i128_mul(syu, sxx), gme_i128_mul(sxu, sxy)), det);
    c = gme_i128_ratio_q16(
        gme_i128_sub(gme_i128_mul(sxv, syy), gme_i128_mul(syv, sxy)), det);
    d = gme_i128_ratio_q16(
        gme_i128_sub(gme_i128_mul(syv, sxx), gme_i128_mul(sxv, sxy)), det);
  }

  const GmeI64 tx = mean_u - gme_i128_round_shift(
                                  gme_i128_add(gme_i128_mul_i64(
                                                   gme_i128_from_i64(a),
                                                   mean_x),
                                               gme_i128_mul_i64(
                                                   gme_i128_from_i64(b),
                                                   mean_y)),
                                  16);
  const GmeI64 ty = mean_v - gme_i128_round_shift(
                                  gme_i128_add(gme_i128_mul_i64(
                                                   gme_i128_from_i64(c),
                                                   mean_x),
                                               gme_i128_mul_i64(
                                                   gme_i128_from_i64(d),
                                                   mean_y)),
                                  16);

  out.params.type =
      model == kGmeFitAffine ? kGmeAffine : kGmeRotZoom;
  if (model == kGmeFitTranslation) {
    out.params.type = kGmeRotZoom;
    out.params = gme_make_identity_params();
    out.params.type = kGmeRotZoom;
  } else {
    out.params.p[2] = gme_i128_sat_i32(gme_i128_from_i64(a));
    out.params.p[3] = gme_i128_sat_i32(gme_i128_from_i64(b));
    out.params.p[4] = gme_i128_sat_i32(gme_i128_from_i64(c));
    out.params.p[5] = gme_i128_sat_i32(gme_i128_from_i64(d));
  }
  out.params.p[0] =
      gme_i128_sat_i32(gme_i128_mul_i64(gme_i128_from_i64(tx), 8192));
  out.params.p[1] =
      gme_i128_sat_i32(gme_i128_mul_i64(gme_i128_from_i64(ty), 8192));
  out.ok = 1;
  out.points_used = used;
  return out;
}

// Fit + outlier rejection + finalisation + gain gate + quality.  This is the
// direct transcription of `estimate_points_internal()` plus `fit_gm_points()`.
// `active`, `residuals` and `keep` are caller-provided scratch arrays of at
// least `count` entries (per job; no allocation happens here).
AV2ENC_GME_HD GmeEstimate gme_estimate_points_core(
    const GmePoint* points, int count, const GmeOptions& options,
    GmePoint* active, GmeI64* residuals, GmeU8* keep) {
  GmeEstimate out = gme_make_identity_estimate();
  const int min_points = options.min_points > 1 ? options.min_points : 1;
  const int max_points = options.max_points > 1 ? options.max_points : 1;
  const int original_count = count;

  if (count < min_points) {
    out.diag.reason = kGmeReasonTooFewCorrespondences;
    return out;
  }
  if (count > max_points) {
    out.diag.reason = kGmeReasonTooManyCorrespondences;
    return out;
  }

  for (int i = 0; i < count; ++i) active[i] = points[i];

  GmeFitCore fit =
      gme_fit_core(active, count, static_cast<GmeFitModel>(options.model),
                   options.min_points);
  int passes = 0;
  if (options.outlier_passes > 0 && fit.ok) {
    const GmeI128 threshold_floor = gme_i128_mul_i64(
        gme_i128_from_i64(options.outlier_threshold_q8),
        options.outlier_threshold_q8);
    for (int pass = 0; pass < options.outlier_passes; ++pass) {
      int residual_count = 0;
      for (int i = 0; i < count; ++i) {
        if (active[i].weight <= 0) continue;
        residuals[residual_count++] =
            gme_point_residual(fit.params, active[i]);
      }
      if (residual_count <= options.min_points) break;
      const GmeI64 median =
          gme_kth_smallest(residuals, residual_count, residual_count / 2);
      const GmeI128 threshold = gme_i128_max(
          threshold_floor, gme_i128_mul_i64(gme_i128_from_i64(median), 4));
      int kept_count = 0;
      for (int i = 0; i < count; ++i) {
        if (active[i].weight <= 0) {
          keep[i] = 0;
          continue;
        }
        const bool kept =
            gme_i128_cmp(
                gme_i128_from_i64(
                    gme_point_residual(fit.params, active[i])),
                threshold) <= 0;
        keep[i] = kept ? 1 : 0;
        if (kept) ++kept_count;
      }
      ++passes;
      if (kept_count < options.min_points) break;
      if (kept_count == count) break;
      int write = 0;
      for (int i = 0; i < count; ++i) {
        if (keep[i]) active[write++] = active[i];
      }
      count = write;
      fit = gme_fit_core(active, count,
                         static_cast<GmeFitModel>(options.model),
                         options.min_points);
      if (!fit.ok) break;
    }
  }

  out.params = fit.params;
  out.diag.ok = fit.ok;
  out.diag.points_in = original_count;
  out.diag.points_used = fit.points_used;
  out.diag.outlier_passes_run = passes;
  out.diag.reason = fit.reason;
  out.diag.identity = gme_is_identity_model(fit.params) ? 1 : 0;
  if (!fit.ok) return out;

  GmeEstimate final = gme_finalize_gm_params(fit.params, options);
  final.diag.points_in = original_count;
  final.diag.points_used = fit.points_used;
  final.diag.outlier_passes_run = passes;
  final.diag.shear_mismatch_q16 = static_cast<GmeI32>(
      gme_max_i64(gme_abs_int(final.params.p[3] + final.params.p[4]),
                  gme_abs_int(final.params.p[5] - final.params.p[2])));

  const GmeQuality quality =
      gme_evaluate_quality(final.params, points, original_count);
  final.diag.identity_sse_q8 = quality.identity_sse_q8;
  final.diag.model_sse_q8 = quality.model_sse_q8;
  final.diag.gain_db = quality.gain_db;
  final.diag.model_rmse_q8 = quality.model_rmse_q8;
  final.diag.identity = gme_is_identity_model(final.params) ? 1 : 0;

  if (!final.diag.identity && final.diag.accepted) {
    final.diag.candidate_gain_db = quality.gain_db;
    final.diag.candidate_model_sse_q8 = quality.model_sse_q8;
    const int min_gain =
        options.min_gain_percent > 0 ? options.min_gain_percent : 0;
    const bool enough_gain =
        quality.identity_sse_q8 <= 0 ||
        gme_i128_cmp(
            gme_i128_mul_i64(
                gme_i128_from_i64(quality.model_sse_q8), 100),
            gme_i128_mul_i64(
                gme_i128_from_i64(quality.identity_sse_q8),
                100 - min_gain)) <= 0;
    if (!enough_gain) {
      final.params = gme_make_identity_params();
      final.diag.identity = 1;
      final.diag.accepted = 1;
      final.diag.degraded = 1;
      final.diag.gain_db = 0.0;
      final.diag.model_sse_q8 = quality.identity_sse_q8;
      final.diag.reason = kGmeReasonGainBelowThreshold;
    }
  }
  return final;
}

// Host-side correspondence collection followed by the shared estimator.  The
// GPU kernels implement the same collection with a parallel prefix scan; the
// emulation test uses these serial forms.
AV2ENC_GME_HD GmeEstimate gme_estimate_blocks_core(
    const GmeBlockMv* blocks, int count, const GmeOptions& options,
    GmePoint* points, GmePoint* active, GmeI64* residuals, GmeU8* keep) {
  int point_count = 0;
  for (int i = 0; i < count; ++i) {
    const int result = gme_extract_point_from_block(blocks[i], &points[point_count]);
    if (result < 0) {
      return gme_make_identity_estimate_reason(kGmeReasonBlockMvOverflow);
    }
    if (result > 0) ++point_count;
  }
  return gme_estimate_points_core(points, point_count, options, active,
                                  residuals, keep);
}

AV2ENC_GME_HD GmeEstimate gme_estimate_warps_core(
    const GmeWarpSample* warps, int count, const GmeOptions& options,
    GmePoint* points, GmePoint* active, GmeI64* residuals, GmeU8* keep) {
  int point_count = 0;
  for (int i = 0; i < count; ++i) {
    GmePoint samples[5];
    const int result = gme_extract_points_from_warp(warps[i], samples);
    if (result < 0) {
      return gme_make_identity_estimate_reason(kGmeReasonWarpOverflow);
    }
    for (int j = 0; j < result; ++j) points[point_count++] = samples[j];
  }
  return gme_estimate_points_core(points, point_count, options, active,
                                  residuals, keep);
}

// ---------------------------------------------------------------------------
// Bilinear quality proxy (not the normative §7.13.3.19 predictor).
// ---------------------------------------------------------------------------

AV2ENC_GME_HD GmeU8 gme_bilinear_sample(const GmeU8* plane, int width,
                                               int height, int stride,
                                               GmeI64 x_q16, GmeI64 y_q16,
                                               int max_value) {
  const GmeI64 fx = x_q16 & 0xFFFF;
  const GmeI64 fy = y_q16 & 0xFFFF;
  const GmeI64 ix = x_q16 >> 16;
  const GmeI64 iy = y_q16 >> 16;
  const int x0 = static_cast<int>(gme_clamp_i64(ix, 0, width - 1));
  const int y0 = static_cast<int>(gme_clamp_i64(iy, 0, height - 1));
  const int x1 = static_cast<int>(gme_clamp_i64(ix + 1, 0, width - 1));
  const int y1 = static_cast<int>(gme_clamp_i64(iy + 1, 0, height - 1));
  const int p00 = plane[static_cast<GmeI64>(y0) * stride + x0];
  const int p01 = plane[static_cast<GmeI64>(y0) * stride + x1];
  const int p10 = plane[static_cast<GmeI64>(y1) * stride + x0];
  const int p11 = plane[static_cast<GmeI64>(y1) * stride + x1];
  const GmeI64 top = (p00 * (65536 - fx) + p01 * fx) >> 16;
  const GmeI64 bottom = (p10 * (65536 - fx) + p11 * fx) >> 16;
  const GmeI64 value = (top * (65536 - fy) + bottom * fy) >> 16;
  return static_cast<GmeU8>(
      gme_clip3(0, max_value, static_cast<int>(value)));
}

AV2ENC_GME_HD GmeU8 gme_warp_sample(const GmeParams& params,
                                           const GmeU8* src, int src_w,
                                           int src_h, int src_stride, int x,
                                           int y, int bit_depth) {
  const GmeI64 x_q16 = static_cast<GmeI64>(x) << 16;
  const GmeI64 y_q16 = static_cast<GmeI64>(y) << 16;
  const GmeI128 rx = gme_i128_add(
      gme_i128_shr(gme_i128_add(gme_i128_mul_i64(
                                    gme_i128_from_i64(params.p[2]), x_q16),
                                gme_i128_mul_i64(
                                    gme_i128_from_i64(params.p[3]), y_q16)),
                   16),
      gme_i128_from_i64(params.p[0]));
  const GmeI128 ry = gme_i128_add(
      gme_i128_shr(gme_i128_add(gme_i128_mul_i64(
                                    gme_i128_from_i64(params.p[4]), x_q16),
                                gme_i128_mul_i64(
                                    gme_i128_from_i64(params.p[5]), y_q16)),
                   16),
      gme_i128_from_i64(params.p[1]));
  return gme_bilinear_sample(src, src_w, src_h, src_stride,
                             gme_i128_to_i64_sat(rx),
                             gme_i128_to_i64_sat(ry),
                             (1 << bit_depth) - 1);
}

AV2ENC_GME_HD bool gme_warp_plane_bilinear(
    const GmeParams& params, const GmeU8* src, int src_w, int src_h,
    int src_stride, GmeU8* dst, int dst_w, int dst_h, int dst_stride,
    int bit_depth) {
  if (src == nullptr || dst == nullptr || src_w <= 0 || src_h <= 0 ||
      dst_w <= 0 || dst_h <= 0 || src_stride < src_w || dst_stride < dst_w ||
      bit_depth != 8) {
    return false;
  }
  for (int y = 0; y < dst_h; ++y) {
    for (int x = 0; x < dst_w; ++x) {
      dst[static_cast<GmeI64>(y) * dst_stride + x] =
          gme_warp_sample(params, src, src_w, src_h, src_stride, x, y,
                          bit_depth);
    }
  }
  return true;
}

AV2ENC_GME_HD GmeI64 gme_warp_plane_sse(
    const GmeParams& params, const GmeU8* ref, int ref_w, int ref_h,
    int ref_stride, const GmeU8* cur, int cur_w, int cur_h, int cur_stride,
    int bit_depth) {
  if (ref == nullptr || cur == nullptr || ref_w <= 0 || ref_h <= 0 ||
      cur_w <= 0 || cur_h <= 0 || ref_stride < ref_w ||
      cur_stride < cur_w || bit_depth < 1 || bit_depth > 16) {
    return -1;
  }
  const int max_value = (1 << bit_depth) - 1;
  GmeI64 sse = 0;
  for (int y = 0; y < cur_h; ++y) {
    for (int x = 0; x < cur_w; ++x) {
      const GmeI64 x_q16 = static_cast<GmeI64>(x) << 16;
      const GmeI64 y_q16 = static_cast<GmeI64>(y) << 16;
      const GmeI128 rx = gme_i128_add(
          gme_i128_shr(gme_i128_add(gme_i128_mul_i64(
                                        gme_i128_from_i64(params.p[2]),
                                        x_q16),
                                    gme_i128_mul_i64(
                                        gme_i128_from_i64(params.p[3]),
                                        y_q16)),
                       16),
          gme_i128_from_i64(params.p[0]));
      const GmeI128 ry = gme_i128_add(
          gme_i128_shr(gme_i128_add(gme_i128_mul_i64(
                                        gme_i128_from_i64(params.p[4]),
                                        x_q16),
                                    gme_i128_mul_i64(
                                        gme_i128_from_i64(params.p[5]),
                                        y_q16)),
                       16),
          gme_i128_from_i64(params.p[1]));
      const int predicted = static_cast<int>(gme_bilinear_sample(
          ref, ref_w, ref_h, ref_stride, gme_i128_to_i64_sat(rx),
          gme_i128_to_i64_sat(ry), max_value));
      const int source = cur[static_cast<GmeI64>(y) * cur_stride + x];
      const int delta = predicted - source;
      sse += static_cast<GmeI64>(delta * delta);
    }
  }
  return sse;
}

}  // namespace av2enc::cuda::gme

#endif  // AV2ENC_CUDA_GME_GME_DEVICE_CUH_
