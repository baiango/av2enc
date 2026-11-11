// Bit-depth-dependent sample ranges and arithmetic for AV2 10-bit streams.
//
// AV2 v1.0.0 signals exactly two sample depths: `bit_depth_idc = 0` is 10-bit
// and `= 1` is 8-bit (§6.4.1 Table 6.3).  A value above 1 is reserved, so this
// component models 10-bit only and refuses to emit 12-bit.
//
// The helpers here are the *edge / CfL / deblock sample ranges* the 8-bit
// M1/M2 cores hard-code at 8 bits.  They are deliberately header-only-ish (a
// small .cpp) so both the still writer and an eventual integrated M1/M2 path
// can call one source of truth:
//
//   §4.8          Clip1(x) = Clip3(0, 2^BitDepth - 1, x)               -> 1023
//   §7.13.2        AboveRow/LeftCol unavailable fills:
//                      AboveRow = (1 << (BitDepth - 1)) - 1            ->  511
//                      LeftCol  = (1 << (BitDepth - 1)) + 1            ->  513
//                      corner   = 1 << (BitDepth - 1)                  ->  512
//   §7.13.2.10     DC prediction neutral value 1 << (BitDepth - 1)     ->  512
//   §7.13.5        CfL luma average = 8 << (BitDepth - 1)               -> 4096
//                  CfL luma average clamp = (8 << BitDepth) - 1         -> 8191
//                  scaledLuma = Round2Signed(alpha*(L-lumaAvg), 11)
//   §7.14.2        qlookup(q) (depth-independent step table, q <= MaxQ)
//   §7.14.3/§7.14.4 Dequant clip = +/-(1 << (7 + BitDepth))             -> +-131072
//   §7.15.2.1      inverse row clip  +/-(1 << (BitDepth + 7))           -> +-131072
//                  inverse col clip  +/-(1 << BitDepth)                 -> +-1024
//   §7.17          deblock q index offset 24*(BitDepth-8)               -> 48
//                  side = Max(Side_Thresholds[qInd] + (1 << (12-BitDepth)), 0)
//                         >> (13 - BitDepth)
//   §6.4.1         MaxQ = 255 (8-bit) / 303 (10-bit)
//   §5.18.6.1      base_q_idx is f(8) at 8-bit and f(9) at 10-bit
//   §8.3.1         coefficient-CDF slice from raw base_q_idx:
//                      <=90 -> 0, <=140 -> 1, <=190 -> 2, else 3
//
// NOTE (10-bit vs the 8-bit code): the 8-bit M1 path uses Clip1 = 255, DC bias
// 128, dequant clip +/-(1<<15), inverse col clip +/-(1<<8), and no deblock q
// offset.  None of those values are valid at 10-bit.  In particular
// `txv::dequantize` / `txv::inverse_block` clip at 8-bit bounds and must not be
// used for a 10-bit stream; use `dequantize()` / `inverse_block()` here, which
// delegate the arithmetic to `src/cuda/m1/hbd/hbd_cpu.h` (the verified 10-bit
// oracle used by M10).
#pragma once

#include <cstdint>

namespace av2enc::hbd::depth {

inline constexpr int kBitDepth = 10;
inline constexpr int kSampleMax = (1 << kBitDepth) - 1;  // 1023

// Everything that changes with the signalled depth, in one struct so a caller
// can print/diff the full set (tests pin every field).
struct SampleRanges {
  int bit_depth = kBitDepth;
  int sample_max = kSampleMax;
  int dc_bias = 1 << (kBitDepth - 1);
  int intra_above_default = (1 << (kBitDepth - 1)) - 1;
  int intra_left_default = (1 << (kBitDepth - 1)) + 1;
  int intra_corner_default = 1 << (kBitDepth - 1);
  int dequant_min = -(1 << (7 + kBitDepth));
  int dequant_max = (1 << (7 + kBitDepth)) - 1;
  int inverse_row_min = -(1 << (kBitDepth + 7));
  int inverse_row_max = (1 << (kBitDepth + 7)) - 1;
  int inverse_col_min = -(1 << kBitDepth);
  int inverse_col_max = (1 << kBitDepth) - 1;
  // §7.14.3 conformance: Residual shall fit a signed integer with
  // 1 + BitDepth bits.
  int residual_min = -(1 << kBitDepth);
  int residual_max = (1 << kBitDepth) - 1;
  // §7.17: baseFilterLevel - 24 * (BitDepth - 8).
  int deblock_qindex_offset = 24 * (kBitDepth - 8);
  // §7.13.5 CfL: L and lumaAvg carry three fractional bits.
  int cfl_frac_bits = 3;
  int cfl_luma_avg_bias = 8 << (kBitDepth - 1);
  int cfl_luma_avg_max = (8 << kBitDepth) - 1;
  int cfl_shift = 11;
};

constexpr SampleRanges sample_ranges() { return SampleRanges{}; }

// ---------------------------------------------------------------------------
// §4.8 / §7.14.4 / §7.15.2.1 clipping
// ---------------------------------------------------------------------------

int clip1(std::int64_t value);
int clip_dequant(std::int64_t value);
int clip_inverse_row(std::int64_t value);
int clip_inverse_col(std::int64_t value);

// True when a coded residual sample is representable with 1 + BitDepth bits.
bool residual_representable(std::int64_t value);

// ---------------------------------------------------------------------------
// §7.14.2 quantizer step and §8.3.1 coefficient-CDF context
// ---------------------------------------------------------------------------

// MaxQ for the signalled depth: 255 at 8-bit, 303 at 10-bit.
int max_q(int bit_depth);

// qlookup(q).  §7.14.2 is depth independent; the caller clamps q to
// [1, max_q(bit_depth)] (get_q() does this with Clip3).  q == 0 is the
// lossless special case and returns Ac_Qlookup[0] = 64.
int qlookup(int qindex);

// get_q(qindex, delta) for the 10-bit MaxQ.  `qindex == 0 && delta <= 0`
// returns Ac_Qlookup[0] as required by §7.14.2.
int get_q(int qindex, int delta);

// §8.3.1 coefficient-CDF slice index from the *raw* frame base_q_idx.  Never
// derive this from a delta-Q or segmentation qindex, and never from the depth.
int qctx_slice(int base_q_idx);

// Number of bits syntax-codes base_q_idx with (§5.18.6.1): 9 at 10-bit.
int base_q_idx_bits(int bit_depth);

// ---------------------------------------------------------------------------
// §7.17 deblock sample ranges
// ---------------------------------------------------------------------------

struct DeblockStrength {
  int q_thr = 0;
  int side = 0;
  int q_ind = 0;
};

// §7.17.5 + §7.17.6 for one plane/pass at a given bit depth.  `plane_delta`
// is DeltaQ*Ac + BaseUVAcDeltaQ (0 for luma, 0 for the M1/M2 default at
// chroma).  `df_delta_q` is DfDeltaQ[i].  The 10-bit path subtracts 48 from
// the side-table index and rounds/shifts side by (12-BitDepth)/(13-BitDepth).
DeblockStrength deblock_strength(int qindex, int bit_depth, int plane_delta,
                                 int df_delta_q);

// ---------------------------------------------------------------------------
// §7.13.5 CfL sample ranges
// ---------------------------------------------------------------------------

// lumaAvg = 8 << (BitDepth-1) when nothing is available; otherwise
// Min((8 << BitDepth) - 1, approx_divide(luma_sum, luma_count)).  `L` and
// lumaAvg carry three fractional bits.
int cfl_luma_average(std::int64_t luma_sum, int luma_count);

// scaledLuma = Round2Signed(alpha_q8 * (L - luma_avg), 11), where alpha_q8 is
// CflAlphaU/V * 32 (the signed Q8 alpha the bitstream signals).
int cfl_scaled_luma(int alpha_q8, int luma, int luma_avg);

// Clip1(dc + scaledLuma), the §7.13.5 prediction write.
int cfl_apply(int dc, int alpha_q8, int luma, int luma_avg);

}  // namespace av2enc::hbd::depth
