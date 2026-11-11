#include "encoder/hbd/hbd_depth.h"

#include <algorithm>
#include <cstdint>

#include "cuda/m1/hbd/hbd_cpu.h"
#include "encoder/cfl/cfl_common.h"
#include "encoder/filters/deblock/deblock_core.h"
#include "encoder/format/format.h"

namespace av2enc::hbd::depth {
namespace {

int round2(int value, int n) {
  if (n <= 0) return value;
  return (value + (1 << (n - 1))) >> n;
}

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

}  // namespace

// ---------------------------------------------------------------------------
// Clipping
// ---------------------------------------------------------------------------

int clip1(std::int64_t value) {
  if (value < 0) return 0;
  if (value > kSampleMax) return kSampleMax;
  return static_cast<int>(value);
}

int clip_dequant(std::int64_t value) {
  // §7.14.4 step 8: Clip3(-(1 << (7+BitDepth)), (1 << (7+BitDepth)) - 1, dq2).
  return hbd::clip_dequant(value, kBitDepth);
}

int clip_inverse_row(std::int64_t value) {
  // §7.15.2.1: Clip3(-(1 << (BitDepth+7)), (1 << (BitDepth+7)) - 1, ...).
  return hbd::clip_inverse_row(value, kBitDepth);
}

int clip_inverse_col(std::int64_t value) {
  // §7.15.2.1: Clip3(-(1 << BitDepth), (1 << BitDepth) - 1, ...).
  return hbd::clip_inverse_col(value, kBitDepth);
}

bool residual_representable(std::int64_t value) {
  const SampleRanges r = sample_ranges();
  return value >= r.residual_min && value <= r.residual_max;
}

// ---------------------------------------------------------------------------
// Quantizer step / CDF context
// ---------------------------------------------------------------------------

int max_q(int bit_depth) {
  // §6.4.1 Table 6.3: MAXQ_8_BITS + 2 * MAXQ_OFFSET for higher depths; the
  // format component is the single source of truth.
  if (bit_depth == 8) return fmt::max_quantizer(fmt::BitDepth::k8);
  if (bit_depth == 10) return fmt::max_quantizer(fmt::BitDepth::k10);
  return -1;  // 12-bit is reserved in AV2 v1.0.0
}

int qlookup(int qindex) {
  // §7.14.2.  q is clipped to [1, MaxQ] by get_q(); q == 0 (lossless) returns
  // Ac_Qlookup[0] through the special case in get_q().
  if (qindex < 0) return 0;
  if (qindex < 25) return deblock::kAcQlookup[qindex < 0 ? 0 : qindex];
  return deblock::kAcQlookup[((qindex - 1) % 24) + 1] << ((qindex - 1) / 24);
}

int get_q(int qindex, int delta) {
  // §7.14.2 get_q().  The 10-bit MaxQ is 303.
  if (qindex == 0 && delta <= 0) return deblock::kAcQlookup[0];
  const int qc = clip3(1, max_q(kBitDepth), qindex + delta);
  return qlookup(qc);
}

int qctx_slice(int base_q_idx) {
  // §8.3.1 init_coeff_cdfs(), from the raw frame base_q_idx.
  if (base_q_idx <= 90) return 0;
  if (base_q_idx <= 140) return 1;
  if (base_q_idx <= 190) return 2;
  return 3;
}

int base_q_idx_bits(int bit_depth) {
  // §5.18.6.1: f(8) at 8-bit, f(9) otherwise.
  return bit_depth == 8 ? 8 : 9;
}

// ---------------------------------------------------------------------------
// Deblock ranges
// ---------------------------------------------------------------------------

DeblockStrength deblock_strength(int qindex, int bit_depth, int plane_delta,
                                 int df_delta_q) {
  DeblockStrength out;
  const int maxq = max_q(bit_depth);
  if (maxq <= 0) return out;  // reserved depth: no conforming stream

  // §7.17.6 q_clamped().
  const int qc =
      (qindex == 0 && plane_delta <= 0) ? 0 : clip3(1, maxq, qindex + plane_delta);
  const int lvl = qc + 8 * df_delta_q;  // DF_DELTA_SCALE == 8

  // §7.17.5: qThr = Round2(get_q(lvl, 0), QUANT_TABLE_BITS) >> 6.
  out.q_thr = round2(get_q(lvl, 0), 3) >> 6;

  // The side table index and the side value both depend on BitDepth.  At
  // 10-bit the index is `lvl - 48` and side is rounded by 4 and shifted by 3.
  out.q_ind = clip3(0, deblock::kMaxSideTable - 1,
                    lvl - 24 * (bit_depth - 8));
  const int bias = 1 << (12 - bit_depth);
  const int shift = 13 - bit_depth;
  out.side = std::max(deblock::kSideThresholds[out.q_ind] + bias, 0) >> shift;
  return out;
}

// ---------------------------------------------------------------------------
// CfL ranges
// ---------------------------------------------------------------------------

int cfl_luma_average(std::int64_t luma_sum, int luma_count) {
  const SampleRanges r = sample_ranges();
  int average = r.cfl_luma_avg_bias;
  if (luma_count > 0) {
    average = cfl::approx_divide(static_cast<int>(luma_sum), luma_count);
    if (average > r.cfl_luma_avg_max) average = r.cfl_luma_avg_max;
  }
  return average;
}

int cfl_scaled_luma(int alpha_q8, int luma, int luma_avg) {
  return cfl::round2_signed(alpha_q8 * (luma - luma_avg), 11);
}

int cfl_apply(int dc, int alpha_q8, int luma, int luma_avg) {
  return clip1(static_cast<std::int64_t>(dc) +
               cfl_scaled_luma(alpha_q8, luma, luma_avg));
}

}  // namespace av2enc::hbd::depth
