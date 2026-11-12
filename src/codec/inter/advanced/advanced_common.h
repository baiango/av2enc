#ifndef AV2ENC_INTER_ADVANCED_COMMON_H_
#define AV2ENC_INTER_ADVANCED_COMMON_H_

// Shared definitions for the advanced inter components — AV2 v1.0.0.
//
// This header is standard-library free so the exact same integer arithmetic
// compiles for the CPU reference (src/encoder/inter/advanced/*.cpp) and the
// CUDA kernels (src/cuda/inter/advanced/*.cu). Bit-exactness between the two
// paths therefore comes from sharing one transcription of the spec, not from
// two independent ones.
//
// Scope (see README.md for the full status table):
//   §7.13.3.13/.14  compound prediction: AVERAGE (+CWP and implicit mask),
//                   WEDGE, DIFFWTD, INTRA (inter-intra)
//   §7.13.3.19/.20  block warp (8x8) and extended warp (4x4)
//   §7.13.3.21–.24  setup shear, warp estimation, model reduction
//   §7.13.3.25      block adaptive weighted prediction (BAWP)
//   §7.13.3.27–.30  wedge / diffwtd / intra-mode masks and mask blend
//   §7.7, §7.8      reference list selection, past/future/cur lists
//   §5.18.3.1       get_relative_dist, display order hint reconstruction
//
// No bitstream syntax is written here: every function produces prediction
// samples, a mask, a parameter set, or the symbol values a future entropy pass
// must code.
//
// Conventions:
//   * Sample coordinates are (row, col) — `[y][x]` — matching §7.13.3.
//   * Motion vectors are `mv[0] = row`, `mv[1] = col`, 1/8 luma sample units
//     unless `prescaled` says 1/16 (§7.13.3.17).
//   * Filtered inter predictions that feed a compound blend live at 1/16
//     sample precision (4 fractional bits): `Preds ~= sample << 4`.
//     Single-reference predictions are rounded to integer samples.

#include <cstddef>
#include <cstdint>

#include "tables/warp_filters.h"
#include "tables/wedge_tables.h"

#if defined(__CUDACC__)
#define AV2ENC_ADV_HD __host__ __device__
#else
#define AV2ENC_ADV_HD
#endif

namespace av2enc::inter::adv {

// ---------------------------------------------------------------------------
// §9.2 constants used by this component.
// ---------------------------------------------------------------------------

inline constexpr int kMiSize = 4;
inline constexpr int kMiSizeLog2 = 2;
inline constexpr int kNumRefFrames = 16;
inline constexpr int kRefsPerFrame = 7;
inline constexpr int kRestrictedOrderHint = -1;  // RESTRICTED_OH

inline constexpr int kFilterBits = 7;
inline constexpr int kSubpelBits = 4;
inline constexpr int kScaleSubpelBits = 10;
inline constexpr int kRefScaleShift = 14;
inline constexpr int kMvLow = -(1 << 16);
inline constexpr int kMvUpp = 1 << 16;

inline constexpr int kInterRound0 = 3;  // after the horizontal pass
inline constexpr int kInterRound1Single = 11;
inline constexpr int kInterRound1Compound = 7;

inline constexpr int kWarpedModelPrecBits = 16;
inline constexpr int kWarpParamReduceBits = 6;
inline constexpr int kWarpedDiffPrecBits = 10;
inline constexpr int kWarpedPixelPrecShifts = 64;  // 1 << 6
inline constexpr int kWarpedModelTransClamp = 1 << 27;
inline constexpr int kExtWarpTaps = 6;
inline constexpr int kExtWarpPhases = 64;
inline constexpr int kExtWarpRoundBits = kWarpedModelPrecBits - 6;  // EXT_WARP_PHASES_LOG2 = 6

inline constexpr int kLsMvMax = 256;
inline constexpr int kLeastSquaresSamplesMax = 8;
inline constexpr int kMaxWarpRefCandidates = 4;

inline constexpr int kMaxCwpNum = 5;
inline constexpr int kCwpEqual = 8;
inline constexpr int kCompoundTypes = 4;
inline constexpr int kInterIntraModes = 4;

inline constexpr int kMaskMasterSize = 128;
inline constexpr int kWedgeAngles = tables::kWedgeAngles;
inline constexpr int kWedgeTypes = tables::kWedgeTypes;
inline constexpr int kNumWedgeDist = tables::kNumWedgeDist;
inline constexpr int kWedgeBldLutSize = tables::kWedgeBldLutSize;
inline constexpr int kWedgeBoundarySharp = 0;
inline constexpr int kWedgeBoundarySmooth = 1;
inline constexpr int kHwEdgeAngles = 10;

// §6.19.7.15 Table: compound_type syntax values.
enum class CompoundType : std::int32_t {
  kWedge = 0,
  kDiffWtd = 1,
  kAverage = 2,
  kIntra = 3,
};

// §6.19.7.13 motion modes (MOTION_MODES = 5).
enum class MotionMode : std::int32_t {
  kSimple = 0,
  kInterIntra = 1,
  kLocalWarp = 2,
  kDeltaWarp = 3,
  kExtendWarp = 4,
  // Not an AV2 motion mode. Encoder-only overlapped MC experiment; see obmc.h.
  kObmcExperimental = 5,
};

// §6.19.7.11 Table 6.24 interintra_mode values.
enum class InterIntraMode : std::int32_t {
  kDc = 0,
  kVertical = 1,
  kHorizontal = 2,
  kSmooth = 3,
};

// `Cwp_Weighting_Factor[is_same_side][idx]` from §5.20.7.6. `idx` is the number
// of `cwp_idx == 0` bits that were read; 0..MAX_CWP_NUM-1.
AV2ENC_ADV_HD inline constexpr int cwp_weighting_factor(int index, bool same_side) {
  constexpr std::int8_t kOpposite[kMaxCwpNum] = {8, 12, 4, 10, 6};
  constexpr std::int8_t kSame[kMaxCwpNum] = {8, 12, 4, 20, -4};
  return same_side ? kSame[index] : kOpposite[index];
}

// ---------------------------------------------------------------------------
// §4.8 mathematical functions, integer forms.
// ---------------------------------------------------------------------------

AV2ENC_ADV_HD inline constexpr int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

AV2ENC_ADV_HD inline constexpr int clip1(int value, int bit_depth) {
  return clip3(0, (1 << bit_depth) - 1, value);
}

AV2ENC_ADV_HD inline constexpr int abs_int(int value) { return value < 0 ? -value : value; }

AV2ENC_ADV_HD inline constexpr int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// Round2(x, n) = floor((x + 2^(n-1)) / 2^n) for x >= 0 (§4.8).
AV2ENC_ADV_HD inline constexpr int round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

// Round2Signed(x, n) = Round2(x, n) for x >= 0 else -Round2(-x, n).
AV2ENC_ADV_HD inline constexpr int round2_signed(int value, int n) {
  if (value >= 0) {
    return round2(value, n);
  }
  return -round2(-value, n);
}

// 64-bit form of Round2Signed, used where a product can exceed 32 bits before
// the spec's Clip3().
AV2ENC_ADV_HD inline std::int64_t round2_signed64(std::int64_t value, int n) {
  if (n <= 0) {
    return value;
  }
  const std::int64_t half = std::int64_t{1} << (n - 1);
  return value >= 0 ? ((value + half) >> n) : -(((-value) + half) >> n);
}

// Arithmetic right shift. C++ leaves the sign of >> on negative values
// implementation-defined; every target compiler (gcc/clang, nvcc host and
// device) defines it as an arithmetic shift, which is what the spec's `>>`
// means. Wrapping it keeps the intent explicit.
AV2ENC_ADV_HD inline constexpr int asr(int value, int n) { return value >> n; }

// ---------------------------------------------------------------------------
// §7.13.3.22 resolve divisor / resolve_division (also used by BAWP and warp).
// Verified against the Div_Lut closed form in the unit tests.
// ---------------------------------------------------------------------------

inline constexpr int kDivLutBits = 7;
inline constexpr int kDivLutPrecBits = 9;
inline constexpr int kDivLutNum = 129;

AV2ENC_ADV_HD inline constexpr int div_lut_entry(int f) {
  const int den = 128 + f;
  return (65536 + den / 2) / den;
}

struct DivScaleShift {
  int shift;
  int factor;
};

// §7.13.3.22: n = FloorLog2(d), e = d - (1 << n),
// f = (n > DIV_LUT_BITS) ? Round2(e, n - DIV_LUT_BITS)
//                        : (e << (DIV_LUT_BITS - n)).
AV2ENC_ADV_HD inline DivScaleShift resolve_divisor(int d) {
  const int n = floor_log2(d);
  const int e = d - (1 << n);
  const int f = (n > kDivLutBits) ? round2(e, n - kDivLutBits) : (e << (kDivLutBits - n));
  return DivScaleShift{n + kDivLutPrecBits, div_lut_entry(f)};
}

// resolve_division(N, D, shift) approximates (N << shift) / D.
// Returns 0 for D <= 0 (the spec only calls this with D > 0; a zero divisor
// yields det == 0 handling in the callers, which check first).
AV2ENC_ADV_HD inline int resolve_division(int n_value, int d_value, int shift) {
  if (d_value <= 0) {
    return 1 << shift;
  }
  const bool sign_n = n_value < 0;
  int n = sign_n ? -n_value : n_value;
  const int shift_n = floor_log2(n);
  const int shift_d = floor_log2(d_value);
  const int e_d = d_value - (1 << shift_d);
  const int f_d = (shift_d > kDivLutBits) ? round2(e_d, shift_d - kDivLutBits)
                                          : (e_d << (kDivLutBits - shift_d));
  const int f_n =
      (shift_n > kDivLutBits) ? round2(n, shift_n - kDivLutBits) : (n << (kDivLutBits - shift_n));
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

AV2ENC_ADV_HD inline int approx_divide(int num, int den) {
  if (den <= 0) {
    return 0;
  }
  const DivScaleShift ds = resolve_divisor(den);
  return round2(num * ds.factor, ds.shift);
}

// ---------------------------------------------------------------------------
// Plane descriptors.
// ---------------------------------------------------------------------------

// A read-only plane with inclusive visible bounds. Reference reads are clamped
// to [0, max_x] x [0, max_y] by the warp processes (§7.13.3.19/.20).
struct PlaneView {
  const std::uint8_t* data = nullptr;
  int stride = 0;
  int max_x = 0;
  int max_y = 0;
};

// A writable plane. Prediction writes are clipped to the visible bounds by the
// caller; these processes assume the block lies inside the frame.
struct MutablePlaneView {
  std::uint8_t* data = nullptr;
  int stride = 0;
  int max_x = 0;
  int max_y = 0;
};

// A plane of compound predictions (1/16 sample precision) or masks.
struct PredPlane {
  const std::int16_t* data = nullptr;
  int stride = 0;
};

// Block/section geometry in plane samples.
struct BlockRect {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

// ---------------------------------------------------------------------------
// §7.13.3.27 wedge masks, computed on the fly.
//
// `initialise_wedge_mask_table()` materialises a 2x20x128x128 MasterMask
// (2.6 MB) and 68 masks per block size. The functions below evaluate the same
// expression for one sample instead: only w*h <= 64*64 values are ever needed
// for one block, and a CUDA kernel cannot index a 2.6 MB constant table.
// wedge_test compares every generated mask against a literal transcription of
// initialise_wedge_mask_table().
// ---------------------------------------------------------------------------

// MasterMask[boundary][angle][n][m] for 0 <= n,m < MASK_MASTER_SIZE.
AV2ENC_ADV_HD inline int wedge_master_value(int boundary, int angle, int n, int m) {
  const int y = ((n << 1) - kMaskMasterSize + 1) * tables::kWedgeSinLut[angle];
  int d = ((m << 1) - kMaskMasterSize + 1) * tables::kWedgeCosLut[angle] + y;
  if (boundary == kWedgeBoundarySharp) {
    d *= 2;
  }
  const int clamp_d = clip3(-31, 31, d);
  const int w =
      clamp_d >= 0 ? tables::kPosDist2BldWeight[clamp_d] : tables::kNegDist2BldWeight[-clamp_d];
  return w << 2;
}

// §7.13.3.27 wedge mask for one block. `bsize` selects the boundary type
// (bsize <= BLOCK_16X16 is sharp) and the block dimensions.
AV2ENC_ADV_HD inline int wedge_mask_value(int bsize, int wedge_sign, int wedge_index, int i,
                                          int j) {
  const int w = tables::kWedgeBlockWidth[bsize];
  const int h = tables::kWedgeBlockHeight[bsize];
  const int boundary = (bsize <= tables::kBlock16x16) ? kWedgeBoundarySharp : kWedgeBoundarySmooth;
  const int dir = tables::kWedgeCodebook[wedge_index][0];
  const int xoff = kMaskMasterSize / 2 - ((tables::kWedgeCodebook[wedge_index][1] * w) >> 3);
  const int yoff = kMaskMasterSize / 2 - ((tables::kWedgeCodebook[wedge_index][2] * h) >> 3);
  const int master = wedge_master_value(boundary, dir, yoff + i, xoff + j);
  return wedge_sign ? (64 - master) : master;
}

AV2ENC_ADV_HD inline bool bsize_has_wedge(int bsize) {
  return bsize >= 0 && bsize < tables::kBlockSizes && tables::kWedgeBits[bsize] > 0;
}

// §7.13.3.27: `boundary = bsize <= BLOCK_16X16 ? SHARP : SMOOTH`.
AV2ENC_ADV_HD inline int wedge_boundary_for_bsize(int bsize) {
  return bsize <= tables::kBlock16x16 ? kWedgeBoundarySharp : kWedgeBoundarySmooth;
}

// ---------------------------------------------------------------------------
// §7.13.3.30 mask blend, split out so both the CPU reference and the kernel can
// call it per sample. `m` is in 0..64; `pred0`/`pred1` are the two compound
// predictions (1/16 sample precision).
// ---------------------------------------------------------------------------

AV2ENC_ADV_HD inline int mask_blend_sample(int m, int pred0, int pred1, int inter_post_round) {
  return clip1(round2(m * pred0 + (64 - m) * pred1, 6 + inter_post_round), 8);
}

// Inter-intra variant: both inputs are integer samples (intra prediction and
// the already-clipped inter prediction) and no post-round is applied (§7.13.3.30
// first branch). Round2 of a non-negative value cannot exceed 255.
AV2ENC_ADV_HD inline int inter_intra_blend_sample(int m, int intra_pred, int inter_pred) {
  return round2(m * intra_pred + (64 - m) * inter_pred, 6);
}

// §7.13.3.28 difference weight mask value.
AV2ENC_ADV_HD inline int diffwtd_mask_value(int pred0, int pred1, int mask_type,
                                            int inter_post_round, int bit_depth) {
  int diff = abs_int(pred0 - pred1);
  diff = round2(diff, (bit_depth - 8) + inter_post_round);
  const int m = clip3(0, 64, 38 + diff / 16);
  return mask_type ? (64 - m) : m;
}

// §7.13.3.29 intra-mode variant mask value.
AV2ENC_ADV_HD inline int intra_mode_mask_value(InterIntraMode mode, int w, int h, int i, int j) {
  const int size_scale = 128 / (h > w ? h : w);
  switch (mode) {
    case InterIntraMode::kVertical:
      return tables::kIiWeights1d[i * size_scale];
    case InterIntraMode::kHorizontal:
      return tables::kIiWeights1d[j * size_scale];
    case InterIntraMode::kSmooth:
      return tables::kIiWeights1d[(i < j ? i : j) * size_scale];
    case InterIntraMode::kDc:
    default:
      return 32;
  }
}

// §7.13.3.30 chroma down-sampling of a luma mask.
AV2ENC_ADV_HD inline int mask_subsample(int m00, int m01, int m10, int m11, int sub_x, int sub_y) {
  if (!sub_x) {
    return m00;
  }
  if (!sub_y) {
    return round2(m00 + m01, 1);
  }
  return round2(m00 + m01 + m10 + m11, 2);
}

// ---------------------------------------------------------------------------
// §7.13.3.25 BAWP: the explicit alpha derivation and the sample adaptation.
// The template collection lives in bawp_common.h.
// ---------------------------------------------------------------------------

// BAWP_SCALES_CTX_COUNT = 3. `list_index` is
//   (YMode == NEARMV) ? 0 : (YMode == NEWMV && use_amvd) ? 1 : 2.
AV2ENC_ADV_HD inline int bawp_explicit_alpha(int list_index, int first_ref_dist,
                                             bool explicit_bawp_scale) {
  int scale = list_index + 1;
  if (first_ref_dist > 4) {
    scale += 1;
  }
  if (!explicit_bawp_scale) {
    scale = -scale;
  }
  return 256 + 16 * scale;
}

// `orig * alpha + beta` is signed; the spec writes `>> shift`.
AV2ENC_ADV_HD inline int bawp_apply_sample(int orig, int alpha, int beta, int shift) {
  return clip1(asr(orig * alpha + beta, shift), 8);
}

// to_fullmv (§7.13.3.25 helper): 1/8-pel to integer luma samples.
AV2ENC_ADV_HD inline int to_fullmv(int mv) { return (mv + 3 + ((mv >= 0) ? 1 : 0)) >> 3; }

// ---------------------------------------------------------------------------
// §5.18.3.1 get_relative_dist.
// ---------------------------------------------------------------------------

AV2ENC_ADV_HD inline int get_relative_dist(int a, int b) {
  if (a == kRestrictedOrderHint && b == kRestrictedOrderHint) {
    return 0;
  }
  if (a == kRestrictedOrderHint) {
    return 127;
  }
  if (b == kRestrictedOrderHint) {
    return -127;
  }
  return clip3(-127, 127, a - b);
}

}  // namespace av2enc::inter::adv

#endif  // AV2ENC_INTER_ADVANCED_COMMON_H_
