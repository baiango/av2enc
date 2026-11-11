#ifndef AV2ENC_ENCODER_INTER_GME_GME_CORE_H_
#define AV2ENC_ENCODER_INTER_GME_GME_CORE_H_

// Shared integer arithmetic for AV2 global motion (AV2 v1.0.0 §7.12.2,
// §5.18.9, §7.13.3.21). Header-only so a CUDA port can include the same
// transcription.
//
// Conventions:
//   * Warp model parameters are Q16 (§9.2 WARPEDMODEL_PREC_BITS = 16).
//   * Correspondences and block motion vectors are in 1/8 luma samples
//     (q8).  One luma sample = 8 q8 units.
//   * Parameter order matches §7.12.2.2 / §7.13.3.19: p[0] = x/column
//     translation, p[1] = y/row translation, p[2..5] = affine coefficients.
//   * Motion vectors are (mv_y, mv_x) in 1/8 sample units.

#include <cstdint>

namespace av2enc::inter::gme {

inline constexpr int kMiSize = 4;
inline constexpr int kMvBorder = 128;
inline constexpr int kMvLow = -(1 << 16);
inline constexpr int kMvUpp = 1 << 16;

// §9.2.
inline constexpr int kWarpedModelPrecBits = 16;
inline constexpr int kWarpParamReduceBits = 6;
inline constexpr int kWarpedModelTransClamp = 1 << 27;
inline constexpr int kGmTransPrecBits = 3;
inline constexpr int kGmAlphaPrecBits = 10;
inline constexpr int kGmTransPrecDiff = kWarpedModelPrecBits - kGmTransPrecBits;
inline constexpr int kGmAlphaPrecDiff = kWarpedModelPrecBits - kGmAlphaPrecBits;
inline constexpr int kGmAbsTransBits = 14;
inline constexpr int kGmAbsAlphaBits = 9;
inline constexpr int kGmTransMax = (1 << kGmAbsTransBits) - 1;
inline constexpr int kGmAlphaMax = (1 << kGmAbsAlphaBits) - 1;
inline constexpr int kGmTransMin = -kGmTransMax;
inline constexpr int kGmAlphaMin = -kGmAlphaMax;
inline constexpr int kGmTransStep = 1 << kGmTransPrecDiff;      // 8192
inline constexpr int kGmAlphaStep = 1 << kGmAlphaPrecDiff;      // 64
inline constexpr int kGmOne = 1 << kWarpedModelPrecBits;        // 65536

// §5.18.9.1 Default_Warp_Params.
inline constexpr int kDefaultWarpParams[6] = {0, 0, kGmOne, 0, 0, kGmOne};

// §5.18.3.1 sentinel.
inline constexpr int kRestrictedOrderHint = -1;

// ---------------------------------------------------------------------------
// §4.8 integer forms.
// ---------------------------------------------------------------------------

constexpr int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

constexpr int abs_int(int value) { return value < 0 ? -value : value; }

constexpr int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// Round2(x, n) for x >= 0.
constexpr int round2(int value, int n) {
  if (n <= 0) return value;
  return (value + (1 << (n - 1))) >> n;
}

constexpr int round2_signed(int value, int n) {
  if (value >= 0) return round2(value, n);
  return -round2(-value, n);
}

inline std::int64_t round2_signed64(std::int64_t value, int n) {
  if (n <= 0) return value;
  const std::int64_t half = std::int64_t{1} << (n - 1);
  return value >= 0 ? ((value + half) >> n) : -(((-value) + half) >> n);
}

// ---------------------------------------------------------------------------
// §7.13.3.22 resolve divisor (also used by §5.18.9.1 scale_warp_model).
// ---------------------------------------------------------------------------

inline constexpr int kDivLutBits = 7;
inline constexpr int kDivLutPrecBits = 9;
inline constexpr int kDivLutNum = 129;

constexpr int div_lut_entry(int f) {
  const int den = 128 + f;
  return (65536 + den / 2) / den;
}

struct DivScaleShift {
  int shift = 0;
  int factor = 0;
};

inline DivScaleShift resolve_divisor(int d) {
  const int n = floor_log2(d);
  const int e = d - (1 << n);
  const int f = (n > kDivLutBits) ? round2(e, n - kDivLutBits) : (e << (kDivLutBits - n));
  return DivScaleShift{n + kDivLutPrecBits, div_lut_entry(f)};
}

// §5.18.3.1 get_relative_dist.
inline int get_relative_dist(int a, int b) {
  if (a == kRestrictedOrderHint && b == kRestrictedOrderHint) return 0;
  if (a == kRestrictedOrderHint) return 127;
  if (b == kRestrictedOrderHint) return -127;
  return clip3(-127, 127, a - b);
}

// §7.13.3.21 setup shear validity, boolean form. Full shear values are only
// needed by the predictor; validation and model degradation need the two
// conformance inequalities.
inline bool setup_shear_valid(const int p[6]) {
  // Keep the int64 products below well inside their range. Global motion
  // syntax already limits these values; this guard only protects direct API
  // callers with an unquantised fit.
  if (p[2] < -(1 << 20) || p[2] > (1 << 20) || p[5] < -(1 << 20) ||
      p[5] > (1 << 20) || p[3] < -(1 << 20) || p[3] > (1 << 20) ||
      p[4] < -(1 << 20) || p[4] > (1 << 20)) {
    return false;
  }
  const int max_value = 32767 - (1 << (kWarpParamReduceBits - 1));
  const int one = kGmOne;
  const std::int64_t alpha0 = static_cast<std::int64_t>(p[2]) - one;
  const std::int64_t beta0 = p[3];
  const DivScaleShift ds = resolve_divisor(p[2] > 0 ? p[2] : 1);
  const std::int64_t v = static_cast<std::int64_t>(p[4]) << kWarpedModelPrecBits;
  const std::int64_t w = static_cast<std::int64_t>(p[3]) * p[4];
  const std::int64_t gamma0 =
      static_cast<std::int64_t>(p[5]) - round2_signed64(w * ds.factor, ds.shift) - one;
  const int alpha = round2_signed(static_cast<int>(clip3(
                                      -32768, max_value,
                                      static_cast<int>(alpha0 > 2147483647LL
                                                           ? 2147483647LL
                                                           : (alpha0 < -2147483648LL
                                                                  ? -2147483648LL
                                                                  : alpha0)))),
                                  kWarpParamReduceBits)
                    << kWarpParamReduceBits;
  const int beta = round2_signed(static_cast<int>(clip3(
                                     -32768, max_value,
                                     static_cast<int>(beta0 > 2147483647LL
                                                          ? 2147483647LL
                                                          : (beta0 < -2147483648LL
                                                                 ? -2147483648LL
                                                                 : beta0)))),
                                 kWarpParamReduceBits)
                   << kWarpParamReduceBits;
  const int gamma = round2_signed(
                        static_cast<int>(clip3(
                            -32768, max_value,
                            static_cast<int>(round2_signed64(v * ds.factor, ds.shift)))),
                        kWarpParamReduceBits)
                    << kWarpParamReduceBits;
  const int delta = round2_signed(
                        static_cast<int>(clip3(
                            -32768, max_value,
                            static_cast<int>(gamma0 > 2147483647LL
                                                 ? 2147483647LL
                                                 : (gamma0 < -2147483648LL ? -2147483648LL
                                                                           : gamma0)))),
                        kWarpParamReduceBits)
                    << kWarpParamReduceBits;
  if (4 * abs_int(alpha) + 7 * abs_int(beta) >= (3 << kWarpedModelPrecBits)) return false;
  if (4 * abs_int(gamma) + 4 * abs_int(delta) >= (3 << kWarpedModelPrecBits)) return false;
  return true;
}

// §7.13.3.23 reduce_warp_model (used when an affine fit is clamped before
// being handed to the predictor).
inline void reduce_warp_model_in_place(int p[6]) {
  const int max_value = (1 << (kWarpedModelPrecBits - 1)) - (1 << kWarpParamReduceBits);
  const int min_value = -max_value;
  for (int i = 2; i < 6; ++i) {
    const int offset = (i == 2 || i == 5) ? kGmOne : 0;
    const std::int64_t widened = static_cast<std::int64_t>(p[i]) - offset;
    const int original = static_cast<int>(
        widened > 2147483647LL ? 2147483647LL
                               : (widened < -2147483648LL ? -2147483648LL : widened));
    const int clamped = clip3(min_value, max_value, original);
    const int rounded = round2_signed(clamped, kWarpParamReduceBits) << kWarpParamReduceBits;
    p[i] = rounded + offset;
  }
}

}  // namespace av2enc::inter::gme

#endif  // AV2ENC_ENCODER_INTER_GME_GME_CORE_H_
