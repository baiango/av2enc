// Shared host/device pixel arithmetic for the format video core.
//
// The core serves 10-bit 4:2:0/4:2:2/4:4:4/4:0:0 and 8-bit 4:2:2/4:4:4/4:0:0
// streams.  For those formats the CPU reference (`video_hbd.cpp`) and the CUDA
// backend (`src/cuda/pipeline/video/hbd/`) must make *identical* decisions and
// produce identical samples.  Everything that touches a pixel therefore lives
// in this header as an inline function compiled for both host and device.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (commit 38f28e7):
//   §7.13.2.10  DC intra prediction (approx_divide / resolve_divisor)
//   §7.13.3.17  subpel sample location derivation (reuses `mc_core.h`)
//   §7.13.3.18  two-pass subpel interpolation, Clip1 at BitDepth
//   §7.14.2     qlookup
//
// Sample storage is always `uint16_t`, LSB-aligned to the signalled depth, so
// the 8-bit formats use the same container.  All arithmetic is integer; there
// is no FP or FMA sensitivity.
#pragma once

#include <cstdint>

#include "codec/inter/mc/mc_core.h"
#include "formats/format.h"

#if defined(__CUDACC__)
#define AV2ENC_HBD_HD __host__ __device__ __forceinline__
#else
#define AV2ENC_HBD_HD inline
#endif

namespace av2enc::video::hbd {

AV2ENC_HBD_HD constexpr int clip3_value(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

AV2ENC_HBD_HD constexpr int clip1_value(int value, int bit_depth) {
  return clip3_value(0, (1 << bit_depth) - 1, value);
}

AV2ENC_HBD_HD constexpr int round2_value(int value, int shift) {
  if (shift <= 0) return value;
  return (value + (1 << (shift - 1))) >> shift;
}

AV2ENC_HBD_HD constexpr int floor_log2_value(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// §7.13.2.10 / §7.13.2.11 divisor tables.  Kept identical to the intra
// component (`src/codec/intra/modes/intra_modes.cpp`) so 8-bit 4:2:0 predictors
// match that reference exactly and 10-bit chroma block sums (w + h = 48 at
// 4:2:2) use the full Div_Lut form.
constexpr int kHbdDivLutBits = 7;      // DIV_LUT_BITS
constexpr int kHbdDivLutPrecBits = 9;  // DIV_LUT_PREC_BITS

// Div_Lut from the spec attachment; the values are the intra component's
// `kDivLut` (test-pinned there).  Namespace scope because CUDA's C++20 mode
// rejects a `static constexpr` local inside a constexpr function.
#if defined(__CUDA_ARCH__)
#define AV2ENC_HBD_DIV_TABLE __device__ constexpr
#else
#define AV2ENC_HBD_DIV_TABLE inline constexpr
#endif
AV2ENC_HBD_DIV_TABLE int kHbdDivLut[129] = {
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465,
    462, 458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 426, 423,
    420, 417, 415, 412, 410, 407, 405, 402, 400, 397, 395, 392, 390, 388,
    386, 383, 381, 379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358,
    356, 354, 352, 350, 349, 347, 345, 343, 341, 340, 338, 336, 334, 333,
    331, 329, 328, 326, 324, 323, 321, 320, 318, 317, 315, 314, 312, 311,
    309, 308, 306, 305, 303, 302, 301, 299, 298, 297, 295, 294, 293, 291,
    290, 289, 287, 286, 285, 284, 282, 281, 280, 279, 278, 277, 275, 274,
    273, 272, 271, 270, 269, 267, 266, 265, 264, 263, 262, 261, 260, 259,
    258, 257, 256,
};
#undef AV2ENC_HBD_DIV_TABLE

AV2ENC_HBD_HD constexpr int hbd_resolve_divisor(int divisor) {
  const int magnitude = divisor < 0 ? -divisor : divisor;
  const int n = floor_log2_value(magnitude);
  const int e = magnitude - (1 << n);
  const int f = n > kHbdDivLutBits ? round2_value(e, n - kHbdDivLutBits)
                                   : (e << (kHbdDivLutBits - n));
  return kHbdDivLut[f] * (divisor < 0 ? -1 : 1);
}

AV2ENC_HBD_HD constexpr int hbd_approx_divide(int numerator,
                                              int denominator) {
  return round2_value(numerator * hbd_resolve_divisor(denominator),
                      floor_log2_value(denominator < 0 ? -denominator
                                                       : denominator) +
                          kHbdDivLutPrecBits);
}

// One DC prediction sample set (§7.13.2.10).  `above`/`left` hold w/h
// reconstructed samples; `dst` receives w*h samples at `dst_stride`.
AV2ENC_HBD_HD void hbd_dc_predict(const uint16_t* above, const uint16_t* left,
                                  int w, int h, bool have_above,
                                  bool have_left, int bit_depth,
                                  uint16_t* dst, int dst_stride) {
  int value = 0;
  if (have_left && have_above) {
    int sum = 0;
    for (int k = 0; k < h; ++k) sum += left[k];
    for (int k = 0; k < w; ++k) sum += above[k];
    value = clip1_value(hbd_approx_divide(sum, w + h), bit_depth);
  } else if (have_left) {
    int sum = 0;
    for (int k = 0; k < h; ++k) sum += left[k];
    value = round2_value(sum, floor_log2_value(h));
  } else if (have_above) {
    int sum = 0;
    for (int k = 0; k < w; ++k) sum += above[k];
    value = round2_value(sum, floor_log2_value(w));
  } else {
    value = 1 << (bit_depth - 1);
  }
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_stride) +
          static_cast<std::size_t>(x)] =
          static_cast<std::uint16_t>(value);
    }
  }
}

// §7.13.3.18 prediction for one sample.  `ref` is the MI-aligned reference
// plane; taps clip to [first_x,last_x] / [first_y,last_y].  `mv` is in 1/8
// luma samples; `sub_x`/`sub_y` are the plane subsampling.
AV2ENC_HBD_HD int hbd_mc_sample(const std::uint16_t* ref, int stride,
                                int first_x, int first_y, int last_x,
                                int last_y, int mv_y, int mv_x, int sub_x,
                                int sub_y, int block_w, int block_h,
                                int bit_depth, int r, int c) {
  using namespace av2enc::inter::mc;
  // r/c are block-relative output coordinates; the sample location derivation
  // is linear in them, so derive it once at the block origin (0,0).
  const McSampleLocation loc =
      mc_derive_sample_location(0, 0, Mv{mv_y, mv_x}, sub_x, sub_y);
  const std::int32_t p_x = loc.start_x + loc.step_x * c;
  const std::int32_t p_y = loc.start_y + loc.step_y * r;

  const int col = (p_x >> kScaleSubpelBits) - 3;
  const int phase_h = static_cast<int>((p_x >> 6) & kSubpelMask);
  const int phase_v = static_cast<int>((p_y >> 6) & kSubpelMask);

  // §7.13.3.18 filter downgrade: a pass whose extent is <= 4 samples uses the
  // four-tap table (the 8-bit MC component applies the same rule).
  const InterpFilter filter =
      mc_pass_filter(InterpFilter::kEighttap, /*small=*/false);
  const InterpFilter filter_h =
      mc_pass_filter(filter, block_w <= 4);
  const InterpFilter filter_v =
      mc_pass_filter(filter, block_h <= 4);
  const std::int16_t* fh =
      &kSubpelFilters[static_cast<int>(filter_h)][phase_h][0];
  const std::int16_t* fv =
      &kSubpelFilters[static_cast<int>(filter_v)][phase_v][0];

  std::int32_t sum_v = 0;
  for (int t = 0; t < 8; ++t) {
    const int row =
        clip3_value(first_y, last_y, (static_cast<int>(p_y >> kScaleSubpelBits)
                                          ) + t - 3);
    std::int32_t hsum = 0;
    for (int k = 0; k < 8; ++k) {
      const int cx = clip3_value(first_x, last_x, col + k);
      hsum += static_cast<std::int32_t>(fh[k]) *
              static_cast<std::int32_t>(
                  ref[static_cast<std::size_t>(row) *
                          static_cast<std::size_t>(stride) +
                      static_cast<std::size_t>(cx)]);
    }
    // §7.13.3.18 stores the intermediate row at 16-bit precision; only the
    // final prediction is Clip1'd at BitDepth.
    const std::int32_t intermediate = round2_value(hsum, kInterRound0);
    sum_v += static_cast<std::int32_t>(fv[t]) * intermediate;
  }
  return clip1_value(round2_value(sum_v, kInterRound1), bit_depth);
}

// Whole-block MC.  Writes w*h samples at `dst_stride`.  `w`/`h` are in the
// plane's own samples.
AV2ENC_HBD_HD void hbd_mc_block(const std::uint16_t* ref, int stride,
                                int first_x, int first_y, int last_x,
                                int last_y, int mv_y, int mv_x, int sub_x,
                                int sub_y, int bit_depth, int x, int y, int w,
                                int h, std::uint16_t* dst, int dst_stride) {
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      dst[static_cast<std::size_t>(r) * static_cast<std::size_t>(dst_stride) +
          static_cast<std::size_t>(c)] = static_cast<std::uint16_t>(
          hbd_mc_sample(ref, stride, first_x, first_y, last_x, last_y, mv_y,
                        mv_x, sub_x, sub_y, w, h, bit_depth, y + r, x + c));
    }
  }
}

// Sum of squared differences between two same-sized blocks.
AV2ENC_HBD_HD std::int64_t hbd_sse(const std::uint16_t* a, int a_stride,
                                   const std::uint16_t* b, int b_stride,
                                   int w, int h) {
  std::int64_t sse = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int d = static_cast<int>(
                        a[static_cast<std::size_t>(y) *
                              static_cast<std::size_t>(a_stride) +
                          static_cast<std::size_t>(x)]) -
                    static_cast<int>(
                        b[static_cast<std::size_t>(y) *
                              static_cast<std::size_t>(b_stride) +
                          static_cast<std::size_t>(x)]);
      sse += static_cast<std::int64_t>(d) * d;
    }
  }
  return sse;
}

// Integer-pel SAD at a candidate MV, clipped to the reference extent.
AV2ENC_HBD_HD std::int64_t hbd_sad_at(const std::uint16_t* cur, int cur_stride,
                                      const std::uint16_t* ref, int ref_stride,
                                      int x, int y, int w, int h, int mv_y,
                                      int mv_x, int ref_w, int ref_h) {
  std::int64_t sad = 0;
  for (int r = 0; r < h; ++r) {
    const int ry = clip3_value(0, ref_h - 1, y + r + mv_y);
    for (int c = 0; c < w; ++c) {
      const int rx = clip3_value(0, ref_w - 1, x + c + mv_x);
      const int a = static_cast<int>(
          cur[static_cast<std::size_t>(y + r) *
                  static_cast<std::size_t>(cur_stride) +
              static_cast<std::size_t>(x + c)]);
      const int b = static_cast<int>(
          ref[static_cast<std::size_t>(ry) * static_cast<std::size_t>(ref_stride) +
              static_cast<std::size_t>(rx)]);
      sad += a > b ? a - b : b - a;
    }
  }
  return sad;
}

// Deterministic diamond search plus half/quarter/eighth refinement.  `stages`
// is a bitmask: 1 = half, 2 = quarter, 4 = eighth.  The result is in 1/8 luma
// samples.  The CPU backend and the CUDA kernel call this exact function.
AV2ENC_HBD_HD void hbd_me_search(const std::uint16_t* cur, int cur_stride,
                                  const std::uint16_t* ref, int ref_stride,
                                  int x, int y, int w, int h, int ref_w,
                                  int ref_h, int range, int stages,
                                  int bit_depth, int* out_mv_y,
                                  int* out_mv_x) {
  // MVs are in 1/8 luma samples throughout: an integer-pel offset is 8 units.
  int best_y = 0;
  int best_x = 0;
  std::int64_t best = hbd_sad_at(cur, cur_stride, ref, ref_stride, x, y, w, h,
                                 0, 0, ref_w, ref_h);
  static constexpr int kDiamond[8][2] = {{-1, 0}, {1, 0},  {0, -1}, {0, 1},
                                         {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  bool moved = true;
  for (int iteration = 0; moved && iteration < 32; ++iteration) {
    moved = false;
    int next_y = best_y;
    int next_x = best_x;
    for (int i = 0; i < 8; ++i) {
      const int cy = best_y + kDiamond[i][0] * 8;
      const int cx = best_x + kDiamond[i][1] * 8;
      if (cy < -range * 8 || cy > range * 8 || cx < -range * 8 ||
          cx > range * 8) {
        continue;
      }
      const std::int64_t cost = hbd_sad_at(cur, cur_stride, ref, ref_stride, x,
                                           y, w, h, cy / 8, cx / 8, ref_w,
                                           ref_h);
      if (cost < best) {
        best = cost;
        next_y = cy;
        next_x = cx;
      }
    }
    if (next_y != best_y || next_x != best_x) {
      best_y = next_y;
      best_x = next_x;
      moved = true;
    }
  }

  // Fractional refinement.  Each stage evaluates the eight neighbours plus the
  // centre using the normative two-pass filter, so the chosen MV is the best
  // measured sample position, not a sign-of-SAD guess.
  const int steps[3] = {4, 2, 1};  // half, quarter, eighth (1/8-sample units)
  const auto block_sse = [&](int mv_y, int mv_x) {
    std::int64_t sse = 0;
    for (int r = 0; r < h; ++r) {
      for (int c = 0; c < w; ++c) {
        const int pred = hbd_mc_sample(ref, ref_stride, 0, 0, ref_w - 1,
                                       ref_h - 1, mv_y, mv_x, 0, 0, w, h,
                                       bit_depth, y + r, x + c);
        const int src = static_cast<int>(
            cur[static_cast<std::size_t>(y + r) *
                    static_cast<std::size_t>(cur_stride) +
                static_cast<std::size_t>(x + c)]);
        const int d = src - pred;
        sse += static_cast<std::int64_t>(d) * d;
      }
    }
    return sse;
  };
  for (int s = 0; s < 3; ++s) {
    if ((stages & (1 << s)) == 0) continue;
    std::int64_t stage_best = block_sse(best_y, best_x);
    int stage_y = best_y;
    int stage_x = best_x;
    for (int i = -1; i <= 1; ++i) {
      for (int j = -1; j <= 1; ++j) {
        if (i == 0 && j == 0) continue;
        const int cy = best_y + i * steps[s];
        const int cx = best_x + j * steps[s];
        std::int64_t sse = 0;
        for (int r = 0; r < h; ++r) {
          for (int c = 0; c < w; ++c) {
            // The source block origin is (x, y), so the prediction must be
            // evaluated at the same absolute position.
            const int pred = hbd_mc_sample(
                ref, ref_stride, 0, 0, ref_w - 1, ref_h - 1, cy, cx, 0, 0, w,
                h, bit_depth, y + r, x + c);
            const int src = static_cast<int>(
                cur[static_cast<std::size_t>(y + r) *
                        static_cast<std::size_t>(cur_stride) +
                    static_cast<std::size_t>(x + c)]);
            const int d = src - pred;
            sse += static_cast<std::int64_t>(d) * d;
          }
        }
        if (sse < stage_best) {
          stage_best = sse;
          stage_y = cy;
          stage_x = cx;
        }
      }
    }
    best_y = stage_y;
    best_x = stage_x;
    best = stage_best;
  }
  *out_mv_y = best_y;
  *out_mv_x = best_x;
}

// Block variance cost used by the partition decision.  Identical formula to
// the 8-bit core's block_variance_cost_at().
AV2ENC_HBD_HD std::int64_t hbd_variance_cost(const std::uint16_t* src,
                                             int stride, int x, int y, int w,
                                             int h) {
  if (w <= 0 || h <= 0) return 0;
  std::int64_t sum = 0;
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      sum += src[static_cast<std::size_t>(y + r) *
                     static_cast<std::size_t>(stride) +
                 static_cast<std::size_t>(x + c)];
    }
  }
  const double mean =
      static_cast<double>(sum) / static_cast<double>(w * h);
  double sse = 0.0;
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      const double d =
          static_cast<double>(src[static_cast<std::size_t>(y + r) *
                                      static_cast<std::size_t>(stride) +
                                  static_cast<std::size_t>(x + c)]) -
          mean;
      sse += d * d;
    }
  }
  return static_cast<std::int64_t>(sse);
}

}  // namespace av2enc::video::hbd
