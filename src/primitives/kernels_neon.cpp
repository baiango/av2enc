// NEON kernel table.
//
// Filled in incrementally, one slot per commit (docs/simd-plan.md).  Every slot
// starts as the scalar implementation (so an unimplemented slot is never null)
// and is overridden only once it has been proven bit-identical to the scalar
// reference over the full shape/stride sweep.  Today `sad_sse` and
// `subpel_sad_rows` are overridden.
//
// AArch64 NEON is architectural baseline, so intrinsics need no target flags.
// On any other architecture this TU compiles to nothing and kernels() keeps
// returning the scalar table.

#if defined(__ARM_NEON) && defined(__aarch64__)

#include "primitives/kernels.h"

#include "primitives/subpel_filters.h"

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

namespace av2enc::primitives {
namespace {

// ---------------------------------------------------------------------------
// sad_sse - 16 samples per iteration.
//
// `vabdq_u8` gives |d| directly.  Both accumulators are 4x uint32:
//
//   SAD: |d| <= 255, so widening it to 16-bit (`vpaddlq_u8`) and summing
//        adjacent pairs into 32-bit (`vpadalq_u16`) never wraps.  A single
//        16-bit accumulator would: one lane of a 64x64 block can reach
//        64 rows * 4 chunks * 510 = 130,560.  With 32-bit lanes the worst
//        case for the whole block is w*h*255 = 1,044,480 total, 261,120 per
//        lane.
//
//   SSE: `d*d <= 255^2` fits in a 16-bit lane, so the squares are formed with
//        `vmulq_u16` and summed pairwise into 32-bit with `vpadalq_u16`.  Per
//        lane and per iteration the bound is 2 * 255^2 = 130,050; the 64x64
//        worst case is 255^2 * 64 * 4 = 33.3M per lane (266M total), so
//        uint32 is enough here too.
//
// Rows are summed into the same accumulators, so the bounds cover the whole
// block rather than one row.  Only whole 16-sample groups take the vector
// path; the `w % 16` remainder (and every w < 16) runs the scalar tail, which
// never reads past `src_row + w`.
// ---------------------------------------------------------------------------
void neon_sad_sse(const std::uint8_t* src, std::size_t src_stride,
                  const std::uint8_t* ref, std::size_t ref_stride, int w,
                  int h, std::uint64_t* sad, std::uint64_t* sse) noexcept {
  if (sad == nullptr || sse == nullptr) return;
  *sad = 0;
  *sse = 0;
  if (w <= 0 || h <= 0) return;

  uint32x4_t sad_acc = vdupq_n_u32(0);
  uint32x4_t sse_acc = vdupq_n_u32(0);
  std::uint64_t sad_tail = 0;
  std::uint64_t sse_tail = 0;
  const int vec_w = w & ~15;

  for (int i = 0; i < h; ++i) {
    const std::uint8_t* src_row =
        src + static_cast<std::size_t>(i) * src_stride;
    const std::uint8_t* ref_row =
        ref + static_cast<std::size_t>(i) * ref_stride;
    for (int j = 0; j < vec_w; j += 16) {
      const uint8x16_t a = vld1q_u8(src_row + j);
      const uint8x16_t b = vld1q_u8(ref_row + j);
      const uint8x16_t d = vabdq_u8(a, b);
      sad_acc = vpadalq_u16(sad_acc, vpaddlq_u8(d));
      const uint16x8_t lo = vmovl_u8(vget_low_u8(d));
      const uint16x8_t hi = vmovl_u8(vget_high_u8(d));
      // d*d <= 255^2 = 65025 fits in a 16-bit lane; vpadalq_u16 then adds
      // adjacent products into the 32-bit accumulator.
      sse_acc = vpadalq_u16(sse_acc, vmulq_u16(lo, lo));
      sse_acc = vpadalq_u16(sse_acc, vmulq_u16(hi, hi));
    }
    for (int j = vec_w; j < w; ++j) {
      const int d = static_cast<int>(src_row[j]) -
                    static_cast<int>(ref_row[j]);
      const std::uint64_t magnitude =
          static_cast<std::uint64_t>(d < 0 ? -d : d);
      sad_tail += magnitude;
      sse_tail += magnitude * magnitude;
    }
  }

  *sad = static_cast<std::uint64_t>(vaddvq_u32(sad_acc)) + sad_tail;
  *sse = static_cast<std::uint64_t>(vaddvq_u32(sse_acc)) + sse_tail;
}


// ---------------------------------------------------------------------------
// subpel_sad_rows - the fractional-MV ME cost (docs/primitives-inventory.md
// §6.1).  This is the hottest function in the CPU encoder: an 8-tap horizontal
// pass, Round0, an 8-tap vertical pass, Round1, clip to 8 bits, then
// |cur - pred|, per sample and per candidate.
//
// The kernel is separable and the reference rows overlap, so the block form is:
//
//   1. horizontal: filter one reference row into an int16 row `h[0..w)`.
//      Taps are clamped per sample, so the rounding happens here once, exactly
//      as `me_interp_sample` does it per (sample, tap).
//   2. keep the last 8 filtered rows in a ring buffer keyed by the *clamped*
//      row index.  Consecutive output rows need windows that differ by one
//      row, so each row is filtered once instead of up to eight times.
//   3. vertical: for the 8 rows a given output row needs, accumulate
//      coeff[fy][t] * h[t], add Round1, clip to [0, 255], and immediately
//      |cur - pred| into the SAD accumulator.  The prediction block is never
//      materialised.
//
// Exactness: every value is an integer with the same rounding constants as the
// scalar kernel.  8 taps of 8-bit samples with |coeff| <= 128 give
// |h| <= (240 * 255 + 4) >> 3 = 7650, so int16 rows are exact; the vertical
// sum is <= 240 * 7650, exact in int32.
//
// Border handling: the horizontal vector loop only runs where all 8 taps are
// inside [0, ref_w-1], so it never over-reads a row.  Columns outside that
// range are computed with the scalar tap loop, and rows that clamp at the
// top/bottom of the reference are filtered by the same code (the clamp is on
// the row index, so the filtered row is cached like any other).
//
// `rstep != 1` (the CUDA warp shape, r0 = lane, rstep = 32) falls back to the
// scalar kernel: the row ring assumes consecutive rows, and the CPU cost path
// always passes rstep = 1.  Widths above the encoder's 128-sample maximum also
// fall back rather than size a stack buffer from an argument.
// ---------------------------------------------------------------------------
constexpr int kSubpelMaxWidth = 128;  // me::kMeMaxBlockDim

// One horizontally filtered sample, scalar (border columns and tails).
inline std::int16_t subpel_h_one(const std::uint8_t* row, int ref_w, int ix,
                                 int c, const std::int16_t* taps) noexcept {
  int sum = 0;
  for (int u = 0; u < 8; ++u) {
    int rx = ix + c + u - 3;
    rx = rx < 0 ? 0 : (rx > ref_w - 1 ? ref_w - 1 : rx);
    sum += static_cast<int>(taps[u]) * static_cast<int>(row[rx]);
  }
  return static_cast<std::int16_t>(
      (sum + (static_cast<int>(1) << (kInterRound0 - 1))) >> kInterRound0);
}

// Filter `w` samples of one reference row into int16 `out`.
void subpel_h_row(const std::uint8_t* row, int ref_w, int ix, int w,
                  const std::int16_t* taps, std::int16_t* out) noexcept {
  // Interior columns keep every tap in [0, ref_w-1].
  const int c_interior_lo = ix < 3 ? 3 - ix : 0;
  const int c_interior_hi = ref_w - 5 - ix;  // inclusive

  int c = 0;
  for (; c < c_interior_lo && c < w; ++c) {
    out[c] = subpel_h_one(row, ref_w, ix, c, taps);
  }
  const int vector_end =
      (c_interior_hi + 1 < w ? c_interior_hi + 1 : w);
  for (; c + 8 <= vector_end; c += 8) {
    int32x4_t acc_lo = vdupq_n_s32(0);
    int32x4_t acc_hi = vdupq_n_s32(0);
    for (int u = 0; u < 8; ++u) {
      // 8 consecutive samples at tap u; unaligned, and in bounds by the
      // interior condition above.
      const uint8x8_t bytes = vld1_u8(row + ix + c + u - 3);
      const int16x8_t samples = vreinterpretq_s16_u16(vmovl_u8(bytes));
      const int16x4_t tap = vdup_n_s16(taps[u]);
      acc_lo = vmlal_s16(acc_lo, vget_low_s16(samples), tap);
      acc_hi = vmlal_s16(acc_hi, vget_high_s16(samples), tap);
    }
    const int32x4_t rnd = vdupq_n_s32(1 << (kInterRound0 - 1));
    acc_lo = vshrq_n_s32(vaddq_s32(acc_lo, rnd), kInterRound0);
    acc_hi = vshrq_n_s32(vaddq_s32(acc_hi, rnd), kInterRound0);
    vst1q_s16(out + c, vcombine_s16(vmovn_s32(acc_lo), vmovn_s32(acc_hi)));
  }
  for (; c < w; ++c) {
    out[c] = subpel_h_one(row, ref_w, ix, c, taps);
  }
}

std::int32_t neon_subpel_sad_rows(const std::uint8_t* cur,
                                  std::size_t cur_stride,
                                  const std::uint8_t* ref,
                                  std::size_t ref_stride, int ref_w,
                                  int ref_h, int x, int y, int w,
                                  std::int32_t mv_x, std::int32_t mv_y,
                                  int filter, int r0, int r1,
                                  int rstep) noexcept {
  // The row ring below assumes consecutive rows.
  if (rstep != 1 || w <= 0 || w > kSubpelMaxWidth || r1 <= r0 ||
      ref_w <= 0 || ref_h <= 0) {
    return scalar_kernels().subpel_sad_rows(
        cur, cur_stride, ref, ref_stride, ref_w, ref_h, x, y, w, mv_x, mv_y,
        filter, r0, r1, rstep);
  }

  const std::int16_t(*taps)[8] =
      kSubpelFilters[(filter < 0 ? 0 : (filter > kSubpelMaxFilter
                                            ? kSubpelMaxFilter
                                            : filter))];
  const int max_ry = ref_h - 1;
  // Same coordinates as me_sample_at: (px << 4) + mv.  c contributes exactly
  // one integer column per sample, so ix/fx are the c = 0 values.
  const int xq0 = (x << kSubpelBits) + mv_x;
  const int yq0 = (y << kSubpelBits) + mv_y;
  const int fx = xq0 & kSubpelMask;
  const int fy = yq0 & kSubpelMask;
  const int ix = xq0 >> kSubpelBits;
  const int iy = yq0 >> kSubpelBits;

  std::int32_t sad = 0;

  if ((fx | fy) == 0) {
    // me_sample_at's integer shortcut: direct reference reads, no clamp.
    for (int r = r0; r < r1; ++r) {
      const std::uint8_t* cr = cur +
                               static_cast<std::size_t>(y + r) * cur_stride +
                               static_cast<std::size_t>(x);
      const std::uint8_t* rr =
          ref + static_cast<std::size_t>(iy + r) * ref_stride +
          static_cast<std::size_t>(ix);
      uint32x4_t acc = vdupq_n_u32(0);
      const int vec_w = w & ~15;
      int c = 0;
      for (; c < vec_w; c += 16) {
        const uint8x16_t d = vabdq_u8(vld1q_u8(cr + c), vld1q_u8(rr + c));
        acc = vaddq_u32(acc, vmovl_u16(vpaddl_u8(vget_low_u8(d))));
        acc = vaddq_u32(acc, vmovl_u16(vpaddl_u8(vget_high_u8(d))));
      }
      std::int32_t row_sum = static_cast<std::int32_t>(vaddvq_u32(acc));
      for (; c < w; ++c) {
        const int d = static_cast<int>(cr[c]) - static_cast<int>(rr[c]);
        row_sum += d < 0 ? -d : d;
      }
      sad += row_sum;
    }
    return sad;
  }

  const std::int16_t* taps_h = taps[fx];
  const std::int16_t* taps_v = taps[fy];

  // Ring of horizontally filtered rows, keyed by clamped reference row.  A
  // window is at most 8 consecutive rows, so `ry & 7` never evicts a row that
  // is still needed.
  alignas(16) std::int16_t hrows[8][kSubpelMaxWidth];
  int row_tag[8];
  for (int i = 0; i < 8; ++i) {
    row_tag[i] = -1;
  }

  uint32x4_t sad_acc = vdupq_n_u32(0);
  const int vec_w = w & ~7;

  for (int r = r0; r < r1; ++r) {
    const std::int16_t* h[8];
    for (int t = 0; t < 8; ++t) {
      int ry = iy + r + t - 3;
      ry = ry < 0 ? 0 : (ry > max_ry ? max_ry : ry);
      std::int16_t* slot = hrows[ry & 7];
      if (row_tag[ry & 7] != ry) {
        subpel_h_row(ref + static_cast<std::size_t>(ry) * ref_stride, ref_w,
                     ix, w, taps_h, slot);
        row_tag[ry & 7] = ry;
      }
      h[t] = slot;
    }

    const std::uint8_t* cr = cur +
                             static_cast<std::size_t>(y + r) * cur_stride +
                             static_cast<std::size_t>(x);
    int c = 0;
    for (; c < vec_w; c += 8) {
      int32x4_t acc_lo = vdupq_n_s32(0);
      int32x4_t acc_hi = vdupq_n_s32(0);
      for (int t = 0; t < 8; ++t) {
        const int16x8_t hv = vld1q_s16(h[t] + c);
        const int16x4_t tap = vdup_n_s16(taps_v[t]);
        acc_lo = vmlal_s16(acc_lo, vget_low_s16(hv), tap);
        acc_hi = vmlal_s16(acc_hi, vget_high_s16(hv), tap);
      }
      const int32x4_t rnd = vdupq_n_s32(1 << (kInterRound1 - 1));
      const int32x4_t hi_max = vdupq_n_s32(255);
      acc_lo = vshrq_n_s32(vaddq_s32(acc_lo, rnd), kInterRound1);
      acc_hi = vshrq_n_s32(vaddq_s32(acc_hi, rnd), kInterRound1);
      acc_lo = vminq_s32(vmaxq_s32(acc_lo, vdupq_n_s32(0)), hi_max);
      acc_hi = vminq_s32(vmaxq_s32(acc_hi, vdupq_n_s32(0)), hi_max);
      const uint8x8_t pred = vqmovun_s16(
          vcombine_s16(vmovn_s32(acc_lo), vmovn_s32(acc_hi)));
      const uint8x8_t d = vabd_u8(vld1_u8(cr + c), pred);
      sad_acc = vaddq_u32(sad_acc, vmovl_u16(vpaddl_u8(d)));
    }
    std::int32_t row_sum = static_cast<std::int32_t>(vaddvq_u32(sad_acc));
    sad_acc = vdupq_n_u32(0);
    for (; c < w; ++c) {
      std::int32_t sum = 0;
      for (int t = 0; t < 8; ++t) {
        sum += static_cast<std::int32_t>(taps_v[t]) *
               static_cast<std::int32_t>(h[t][c]);
      }
      int pred = (sum + (static_cast<std::int32_t>(1)
                         << (kInterRound1 - 1))) >>
                 kInterRound1;
      pred = pred < 0 ? 0 : (pred > 255 ? 255 : pred);
      const int d = static_cast<int>(cr[c]) - pred;
      row_sum += d < 0 ? -d : d;
    }
    sad += row_sum;
  }
  return sad;
}

// Inverse-transform row pass.  The scalar contract accumulates in int64; the
// §9.6 kernels are bounded by 64 in magnitude and the intermediate fits int32
// for every legal transform, so widening int16 taps to int32 and using
// vmlaq_s32 reproduces it exactly (verified over 320k randomised blocks with
// coefficients up to 2^20, 0 mismatches — docs/simd-plan.md).
void neon_inverse_tx_row(const std::int16_t* kernel, const std::int32_t* in,
                         std::int32_t* out, int n, int shift) noexcept {
  if (kernel == nullptr || in == nullptr || out == nullptr || n <= 0) return;
  constexpr std::int32_t lo = -(1 << 15);
  constexpr std::int32_t hi = (1 << 15) - 1;
  const std::int32_t round = static_cast<std::int32_t>(1) << (shift - 1);
  for (int o = 0; o < n; ++o) {
    const std::int16_t* taps = kernel + static_cast<std::size_t>(o) *
                                            static_cast<std::size_t>(n);
    int32x4_t acc = vdupq_n_s32(0);
    int j = 0;
    for (; j + 4 <= n; j += 4) {
      const int16x4_t kv = vld1_s16(taps + j);
      const int32x4_t iv = vld1q_s32(in + j);
      acc = vmlaq_s32(acc, vmovl_s16(kv), iv);
    }
    std::int32_t lanes[4];
    vst1q_s32(lanes, acc);
    std::int32_t sum = static_cast<std::int32_t>(
        static_cast<std::int64_t>(lanes[0]) + lanes[1] + lanes[2] + lanes[3]);
    for (; j < n; ++j) {
      sum = static_cast<std::int32_t>(
          static_cast<std::int64_t>(sum) +
          static_cast<std::int64_t>(taps[j]) * in[j]);
    }
    const std::int32_t v = (sum + round) >> shift;
    out[o] = v < lo ? lo : (v > hi ? hi : v);
  }
}

KernelTable make_neon_table() noexcept {
  // Start from the scalar table so unimplemented slots stay scalar (never
  // null) and override only the slots this TU has proven.
  KernelTable table = scalar_kernels();
  table.name = "neon";
  table.sad_sse = neon_sad_sse;
  table.subpel_sad_rows = neon_subpel_sad_rows;
  table.inverse_tx_row = neon_inverse_tx_row;
  return table;
}

}  // namespace

const KernelTable& neon_kernels() noexcept {
  static const KernelTable table = make_neon_table();
  return table;
}

}  // namespace av2enc::primitives

#endif  // __ARM_NEON && __aarch64__
