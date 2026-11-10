#ifndef AV2ENC_ENCODER_CFL_COMMON_H_
#define AV2ENC_ENCODER_CFL_COMMON_H_

// Chroma-from-luma (CfL) shared definitions — AV2 v1.0.0 §7.13.5.
//
// This header is deliberately free of the standard library so the same POD
// block descriptor and the same integer arithmetic compile for the CPU
// reference (src/encoder/cfl/cfl.cpp) and the CUDA kernel
// (src/cuda/cfl/cfl.cu). Bit-exactness between the two paths comes from
// sharing these functions, not from two transcriptions of the spec.
//
// Spec references:
//   §5.20.7.32  read_cfl_alphas()            (syntax)
//   §6.19.7.25  Read CFL alphas semantics    (sign table, ranges)
//   §7.13.5     Predict chroma from luma     (prediction / derived alpha)
//   §4.8        Round2 / Round2Signed
//   §7.13.3.22  resolve_divisor / Div_Lut    (used by resolve_division)
//
// Only 8-bit 4:2:0 is a supported encoder format for M1 (see
// docs/tool-roadmap.md); 4:2:2 and 4:4:4 code paths are implemented because
// the spec text is uniform, but they are reference-only and untested beyond
// the arithmetic checks in cfl_test.cpp.

#include <cstdint>

#if defined(__CUDACC__)
#define AV2ENC_CFL_HD __host__ __device__
#else
#define AV2ENC_CFL_HD
#endif

namespace av2enc::cfl {

// ---------------------------------------------------------------------------
// §9.2 constants
// ---------------------------------------------------------------------------

// Number of values for cfl_alpha_u and cfl_alpha_v: |CflAlpha| = 1..8.
inline constexpr int kAlphabetSize = 8;
// Number of values for cfl_alpha_signs.
inline constexpr int kJointSigns = 8;
// Number of contexts for is_cfl (entropy integration, not used here).
inline constexpr int kContexts = 3;
// Number of contexts for cfl_alpha_u / cfl_alpha_v (entropy integration).
inline constexpr int kAlphaContexts = 6;
// Number of above/left samples used by the derived-alpha correlation.
inline constexpr int kNumRefSamCfl = 8;
// cfl_index values (§6.19.7.25 Table 6.26).
inline constexpr int kIndexExplicit = 0;
inline constexpr int kIndexDerivedAlpha = 1;
inline constexpr int kIndexMulti = 2;
// CFL_SIGN_* values (§6.19.7.25 Table 6.27).
inline constexpr int kSignZero = 0;
inline constexpr int kSignNeg = 1;
inline constexpr int kSignPos = 2;
// Explicit alpha is signalled in 1/8 units: CflAlpha*32 == alpha << 5.
inline constexpr int kExplicitAlphaShift = 5;
inline constexpr int kExplicitAlphaQ8 = kAlphabetSize << kExplicitAlphaShift;
// Round2Signed(alpha * (L - lumaAvg), 11).
inline constexpr int kScaleShift = 11;
// Largest chroma block extent handled by one CfL block (spec: <= 64x64).
inline constexpr int kMaxDim = 64;
// Div_Lut parameters (§9.2 / §7.13.3.22).
inline constexpr int kDivLutBits = 7;
inline constexpr int kDivLutPrecBits = 9;
inline constexpr int kDivLutNum = 129;

// ---------------------------------------------------------------------------
// §4.8 mathematical functions (integer forms used by the spec text)
// ---------------------------------------------------------------------------

AV2ENC_CFL_HD inline constexpr int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

AV2ENC_CFL_HD inline constexpr int clip1(int value, int bit_depth) {
  return clip3(0, (1 << bit_depth) - 1, value);
}

AV2ENC_CFL_HD inline constexpr int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// Round2(x, n) = floor((x + 2^(n-1)) / 2^n). The spec notes that this uses
// real division; with non-negative integer inputs the shift form below is the
// equivalent integer definition given in §4.8.
AV2ENC_CFL_HD inline constexpr int round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

// Round2Signed(x, n): Round2 for x >= 0, -Round2(-x, n) for x < 0.
AV2ENC_CFL_HD inline constexpr int round2_signed(int value, int n) {
  if (value >= 0) {
    return round2(value, n);
  }
  return -round2(-value, n);
}

// Div_Lut[f] = lround(65536 / (128 + f)), the table published in §7.13.3.22.
// cfl_test.cpp checks this closed form against the normative table entries.
AV2ENC_CFL_HD inline constexpr int div_lut_entry(int f) {
  const int den = 128 + f;
  return (65536 + den / 2) / den;
}

// resolve_divisor(d) for d > 0, returning (shift, factor) as used by
// approx_divide. §7.13.3.22.
struct DivScaleShift {
  int shift;
  int factor;
};

AV2ENC_CFL_HD inline DivScaleShift resolve_divisor(int d) {
  const int n = floor_log2(d);
  const int e = d - (1 << n);
  const int f = (n > kDivLutBits) ? round2(e, n - kDivLutBits)
                                  : (e << (kDivLutBits - n));
  const int factor = div_lut_entry(f);
  return DivScaleShift{n + kDivLutPrecBits, factor};
}

// approx_divide(num, den) = Round2(num * scale, shift). §7.13.2.11.
AV2ENC_CFL_HD inline int approx_divide(int num, int den) {
  const DivScaleShift ds = resolve_divisor(den);
  return round2(num * ds.factor, ds.shift);
}

// resolve_division(N, D, shift) approximates (N << shift) / D. §7.13.3.22.
AV2ENC_CFL_HD inline int resolve_division(int n_value, int d_value, int shift) {
  const bool sign_n = n_value < 0;
  int n = sign_n ? -n_value : n_value;
  const int shift_n = floor_log2(n);
  const int shift_d = floor_log2(d_value);
  const int e_d = d_value - (1 << shift_d);
  const int f_d = (shift_d > kDivLutBits)
                      ? round2(e_d, shift_d - kDivLutBits)
                      : (e_d << (kDivLutBits - shift_d));
  const int f_n = (shift_n > kDivLutBits)
                      ? round2(n, shift_n - kDivLutBits)
                      : (n << (kDivLutBits - shift_n));
  const int shift_add = shift_d - shift_n - shift;
  int ret = 0;
  if (shift_add <= 1) {
    const int shift0 = kDivLutPrecBits + kDivLutBits + shift_add;
    if (shift0 >= 0) {
      ret = (div_lut_entry(f_d) * f_n) >> shift0;
    } else {
      ret = (2 << shift) - 1;
    }
  }
  ret = ret < ((2 << shift) - 1) ? ret : ((2 << shift) - 1);
  if (sign_n) {
    ret = -ret;
  }
  return ret;
}

// ---------------------------------------------------------------------------
// §7.13.5 down-sampling filters.
//
// Cfl_Filters_420[3][3][3] and Cfl_Filters_422[3][3] are reproduced as weight
// functions so they are constexpr and device-callable. cfl_test.cpp checks
// every entry against the literal spec tables.
//   Cfl_Filters_420 = {{{0,0,0},{0,2,2},{0,2,2}},
//                      {{0,0,0},{1,2,1},{1,2,1}},
//                      {{0,1,0},{1,4,1},{0,1,0}}}
//   Cfl_Filters_422 = {{0,4,4},{2,4,2},{0,8,0}}
// `filter` is the already-normalised index (3 -> 0 handled by the caller).
// ---------------------------------------------------------------------------

AV2ENC_CFL_HD inline constexpr int ds_weight_420(int filter, int dy, int dx) {
  if (filter == 0) {
    return (dy >= 0 && dx >= 0) ? 2 : 0;
  }
  if (filter == 1) {
    if (dy < 0) {
      return 0;
    }
    return dx == 0 ? 2 : 1;
  }
  // filter == 2
  if (dy == 0 && dx == 0) {
    return 4;
  }
  if (dy == 0 || dx == 0) {
    return 1;
  }
  return 0;
}

// §7.13.5 indexes Cfl_Filters_422 by the horizontal tap (dx + subX), not by
// the vertical offset; `dx` here is the offset in [-1, 1].
AV2ENC_CFL_HD inline constexpr int ds_weight_422(int filter, int dx) {
  if (filter == 0) {
    return dx < 0 ? 0 : 4;
  }
  if (filter == 1) {
    return dx == 0 ? 4 : 2;
  }
  return dx == 0 ? 8 : 0;
}

// One tap of the CfL down-sampling filter for the current chroma format.
// `sub_x`/`sub_y` are the 4:2:0/4:2:2/4:4:4 SubsamplingX/Y values.
AV2ENC_CFL_HD inline constexpr int ds_weight(int filter, int dy, int dx,
                                             int sub_x, int sub_y) {
  if (sub_x != 0 && sub_y != 0) {
    return ds_weight_420(filter, dy, dx);
  }
  if (sub_x != 0) {
    return dy == 0 ? ds_weight_422(filter, dx) : 0;
  }
  // Both the 4:4:4 and the (unsupported) 4:4:0 case take the spec's
  // `else { t = 8 * v; }` branch.
  return (dx == 0 && dy == 0) ? 8 : 0;
}

// ---------------------------------------------------------------------------
// Geometry helpers for the two descriptor fields that are easy to get wrong.
// M1 uses MI_SIZE = 4 luma samples and a 64x64 superblock.
// ---------------------------------------------------------------------------

// §7.13.5 sbTop = (MiRow >> log2SbH) << (log2SbH + MI_SIZE_LOG2) for a 64x64
// superblock, expressed in luma rows.
AV2ENC_CFL_HD inline constexpr int cfl_sb_top_y(int chroma_y, int sub_y) {
  return ((chroma_y << sub_y) / 64) * 64;
}

// §5.20.4.1 AvailUChroma = is_inside(ChromaMiRow - 1, ChromaMiCol), where
// ChromaMiRow = (luma_y / MI_SIZE) >> sub_y. For 4:2:0 the first chroma mi
// row spans chroma rows 0..3, so this is false until chroma_y >= 4 — even
// though chroma rows 1..3 have a sample row above them.
AV2ENC_CFL_HD inline constexpr bool cfl_avail_above(int chroma_y, int sub_y) {
  return (((chroma_y << sub_y) / 4) >> sub_y) > 0;
}

AV2ENC_CFL_HD inline constexpr bool cfl_avail_left(int chroma_x, int sub_x) {
  return (((chroma_x << sub_x) / 4) >> sub_x) > 0;
}

// ---------------------------------------------------------------------------
// POD descriptors shared with the CUDA kernel.
// ---------------------------------------------------------------------------

// A signed alpha in explicit signalling units: |alpha| = 1..8 eighths, and
// sign == kSignZero implies magnitude == 0.
struct Alpha {
  std::int8_t sign = kSignZero;
  std::int8_t magnitude = 0;
};

// cfl_alpha_signs, cfl_alpha_u, cfl_alpha_v as read/written by
// read_cfl_alphas() (§5.20.7.32). `alpha_u`/`alpha_v` are the signalled
// magnitude-minus-one symbols (0..7) and are unused when the sign is zero.
struct Signalling {
  int cfl_index = kIndexExplicit;
  int alpha_signs = 0;
  int alpha_u = 0;
  int alpha_v = 0;
};

// One chroma transform block plus the reconstructed luma it predicts from.
//
// Coordinates are chroma samples. `chroma` holds the DC-predicted chroma
// samples on input and receives the CfL prediction on output. The luma origin
// is (x << sub_x, y << sub_y).
//
// Bounds:
//   * luma_max_x / luma_max_y are the inclusive bounds of the *reconstructed*
//     luma plane. Pass the padded reconstruction bounds (M0 pads planes to a
//     64-sample multiple), not the visible picture size: a partial bottom /
//     right block reconstructs into the padding and the decoder reads those
//     samples back for CfL.
//   * coded_width / coded_height are MiCols*MI_SIZE>>sub_x and
//     MiRows*MI_SIZE>>sub_y. They drive the §7.13.5 right/bottom neighbour
//     replication in the luma-average collection.
//   * have_above / have_left are AvailUChroma / AvailLChroma, derived at the
//     chroma mi granularity (§5.20.4.1).
struct Block {
  const std::uint8_t* luma = nullptr;
  int luma_stride = 0;
  int luma_max_x = 0;
  int luma_max_y = 0;
  int sb_top_y = 0;  // top luma row of the enclosing superblock
  std::uint8_t* chroma = nullptr;
  int chroma_stride = 0;
  int x = 0;  // chroma origin
  int y = 0;
  int w = 0;  // chroma block size (= transform size for the plane)
  int h = 0;
  int coded_width = 0;
  int coded_height = 0;
  int have_above = 0;
  int have_left = 0;
  int bit_depth = 8;
  int ds_filter = 0;  // cfl_ds_filter_index, 0..3 (3 behaves as 0)
  int sub_x = 1;
  int sub_y = 1;
};

// Per-plane alpha pair as signalled by cfl_alpha_signs.
struct AlphaPair {
  Alpha u{};
  Alpha v{};
};

// ---------------------------------------------------------------------------
// Alpha helpers.
// ---------------------------------------------------------------------------

// Signed Q8 value used by the prediction: CflAlphaU * 32 (§7.13.5).
AV2ENC_CFL_HD inline constexpr int alpha_q8(Alpha a) {
  if (a.sign == kSignZero || a.magnitude == 0) {
    return 0;
  }
  const int magnitude = a.magnitude;
  return a.sign == kSignNeg ? -magnitude * (1 << kExplicitAlphaShift)
                            : magnitude * (1 << kExplicitAlphaShift);
}

// Snaps a Q8 value to the nearest signallable explicit alpha (ties round away
// from zero, magnitude clamped to 1..8). Used by the encoder alpha search; it
// is not a bitstream operation.
AV2ENC_CFL_HD inline Alpha snap_alpha_q8(int q8) {
  if (q8 == 0) {
    return Alpha{kSignZero, 0};
  }
  const int magnitude_q8 = q8 < 0 ? -q8 : q8;
  int magnitude = (magnitude_q8 + (1 << (kExplicitAlphaShift - 1))) >>
                  kExplicitAlphaShift;
  magnitude = clip3(1, kAlphabetSize, magnitude);
  return Alpha{static_cast<std::int8_t>(q8 < 0 ? kSignNeg : kSignPos),
               static_cast<std::int8_t>(magnitude)};
}

// cfl_alpha_signs = 3 * signU + signV - 1 (§5.20.7.32). Returns -1 when the
// pair is not signallable.
AV2ENC_CFL_HD inline constexpr int joint_signs(AlphaPair p) {
  const int sign_u = p.u.sign;
  const int sign_v = p.v.sign;
  if (sign_u < kSignZero || sign_u > kSignPos || sign_v < kSignZero ||
      sign_v > kSignPos) {
    return -1;
  }
  if (sign_u == kSignZero && sign_v == kSignZero) {
    return -1;  // redundant with DC intra; §6.19.7.25
  }
  return 3 * sign_u + sign_v - 1;
}

// True when one plane's alpha is representable.
AV2ENC_CFL_HD inline constexpr bool alpha_signalable(Alpha a) {
  if (a.sign == kSignZero) {
    return a.magnitude == 0;
  }
  if (a.sign != kSignNeg && a.sign != kSignPos) {
    return false;
  }
  return a.magnitude >= 1 && a.magnitude <= kAlphabetSize;
}

// True when the pair can be represented by read_cfl_alphas() syntax.
AV2ENC_CFL_HD inline constexpr bool signalable(AlphaPair p) {
  return alpha_signalable(p.u) && alpha_signalable(p.v) &&
         joint_signs(p) >= 0;
}

// Packs an explicit pair into the syntax elements a future entropy pass must
// code. No bitstream code lives in this component.
AV2ENC_CFL_HD inline Signalling explicit_signalling(AlphaPair p) {
  const int signs = joint_signs(p);
  Signalling s{};
  s.cfl_index = kIndexExplicit;
  s.alpha_signs = signs < 0 ? 0 : signs;
  s.alpha_u = p.u.sign == kSignZero ? 0 : p.u.magnitude - 1;
  s.alpha_v = p.v.sign == kSignZero ? 0 : p.v.magnitude - 1;
  return s;
}

AV2ENC_CFL_HD inline Signalling derived_signalling() {
  Signalling s{};
  s.cfl_index = kIndexDerivedAlpha;
  return s;
}

// Inverse of explicit_signalling() for tests and round-trips.
AV2ENC_CFL_HD inline AlphaPair pair_from_signalling(Signalling s) {
  const int signs = s.alpha_signs;
  const int sign_u = (signs + 1) / 3;
  const int sign_v = (signs + 1) % 3;
  AlphaPair p{};
  p.u.sign = static_cast<std::int8_t>(sign_u);
  p.u.magnitude = static_cast<std::int8_t>(sign_u == kSignZero ? 0 : s.alpha_u + 1);
  p.v.sign = static_cast<std::int8_t>(sign_v);
  p.v.magnitude = static_cast<std::int8_t>(sign_v == kSignZero ? 0 : s.alpha_v + 1);
  return p;
}

}  // namespace av2enc::cfl

#endif  // AV2ENC_ENCODER_CFL_COMMON_H_
