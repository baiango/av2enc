// AV2 §7.13.3.18 inter prediction — shared host/device arithmetic.
//
// This header carries the normative integer arithmetic of single-reference
// translational motion compensation: §7.13.3.17 sample-location derivation and
// §7.13.3.18 two-pass subpel filtering.  The CPU reference (mc.cpp) uses the
// spec's intermediate-array form; the CUDA kernel (src/cuda/inter/mc) evaluates
// the same horizontal sums per output sample.  Both call the inline arithmetic
// below, so they agree bit-exactly.
//
// Scope: 8-bit 4:2:0, single reference, no compound / warp / OBMC / refinement.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §7.13.3.17, §7.13.3.18.

#ifndef AV2ENC_ENCODER_INTER_MC_MC_CORE_H_
#define AV2ENC_ENCODER_INTER_MC_MC_CORE_H_

#include <cstddef>
#include <cstdint>

#include "codec/inter/mc/mc_subpel_tables.h"
#include "codec/inter/mc/mc_types.h"

#if defined(__CUDACC__)
#define AV2ENC_MC_HD __host__ __device__ __forceinline__
#else
#define AV2ENC_MC_HD inline
#endif

namespace av2enc::inter::mc {

// §4.7 Round2: floor((x + 2^(n-1)) / 2^n).  C++20 guarantees arithmetic right
// shift for signed values, which is the spec's mathematical floor.
AV2ENC_MC_HD constexpr std::int32_t mc_round2(std::int32_t x, int n) {
  if (n == 0) {
    return x;
  }
  return (x + (1 << (n - 1))) >> n;
}

// §4.7 Round2Signed.  Only used for the reference-scaling path, which is
// outside the unscaled M2 default but part of the normative formula.
AV2ENC_MC_HD constexpr std::int32_t mc_round2_signed(std::int32_t x, int n) {
  return x >= 0 ? mc_round2(x, n) : -mc_round2(-x, n);
}

// 64-bit variant for §7.13.3.17: origX * (1 << 14) exceeds int32 at 8K and
// for the full 17-bit MV range, so the spec derivation must not use int32.
AV2ENC_MC_HD constexpr std::int64_t mc_round2_signed64(std::int64_t x, int n) {
  if (n == 0) {
    return x;
  }
  return x >= 0 ? ((x + (static_cast<std::int64_t>(1) << (n - 1))) >> n)
                : -(((-x) + (static_cast<std::int64_t>(1) << (n - 1))) >> n);
}

AV2ENC_MC_HD constexpr std::int32_t mc_clip3(std::int32_t low,
                                             std::int32_t high,
                                             std::int32_t value) {
  return value < low ? low : (value > high ? high : value);
}

// §7.13.3.18 stores predictions through Clip1 before writing CurrFrame.
AV2ENC_MC_HD constexpr std::uint8_t mc_clip1(std::int32_t value) {
  return static_cast<std::uint8_t>(mc_clip3(0, 255, value));
}

// §7.13.3.18 filter-downgrade rule: 8-tap regular/sharp become the four-tap
// regular table, smooth becomes the four-tap smooth table, for a pass whose
// extent is <= 4 samples.
AV2ENC_MC_HD constexpr InterpFilter mc_pass_filter(InterpFilter filter,
                                                   bool small) {
  if (!small) {
    return filter;
  }
  if (filter == InterpFilter::kEighttap ||
      filter == InterpFilter::kEighttapSharp) {
    return InterpFilter::kFourtap;
  }
  if (filter == InterpFilter::kEighttapSmooth) {
    return InterpFilter::kFourtapSmooth;
  }
  return filter;
}

AV2ENC_MC_HD constexpr const std::int16_t* mc_filter_taps(InterpFilter filter,
                                                          int phase) {
  return kSubpelFilters[static_cast<int>(filter)][phase & kSubpelMask];
}

// §7.13.3.17 motion vector scaling / sampling-location derivation.
//
// `sub_x`/`sub_y` are the plane subsampling (0 for luma, 1 for 4:2:0 chroma).
// `mv` is in 1/8 luma samples.  Outputs are 1/1024-sample locations.
//
// The derivation keeps the spec's xScale/yScale form, but this component only
// accepts unscaled references (xScale == yScale == 1 << REF_SCALE_SHIFT), which
// is the M2 baseline.  Scaled references would require the current frame's
// dimensions and are deliberately rejected at integration time.
struct McSampleLocation {
  std::int32_t start_x = 0;
  std::int32_t start_y = 0;
  std::int32_t step_x = 0;
  std::int32_t step_y = 0;
};

AV2ENC_MC_HD McSampleLocation mc_derive_sample_location(int x, int y, Mv mv,
                                                        int sub_x, int sub_y) {
  McSampleLocation loc;

  constexpr std::int64_t kScale = 1 << kRefScaleShift;  // unscaled reference
  constexpr std::int64_t kHalfSample = 1 << (kSubpelBits - 1);
  constexpr std::int64_t kOff = (1 << (kScaleSubpelBits - kSubpelBits)) / 2;

  const std::int64_t orig_x =
      (static_cast<std::int64_t>(x) << kSubpelBits) +
      ((2 * mv.x) >> sub_x) + kHalfSample;
  const std::int64_t orig_y =
      (static_cast<std::int64_t>(y) << kSubpelBits) +
      ((2 * mv.y) >> sub_y) + kHalfSample;

  const std::int64_t base_x =
      orig_x * kScale - (kHalfSample << kRefScaleShift);
  const std::int64_t base_y =
      orig_y * kScale - (kHalfSample << kRefScaleShift);

  constexpr int kShift =
      kRefScaleShift + kSubpelBits - kScaleSubpelBits;  // 14 + 4 - 10 = 8
  loc.start_x =
      static_cast<std::int32_t>(mc_round2_signed64(base_x, kShift) + kOff);
  loc.start_y =
      static_cast<std::int32_t>(mc_round2_signed64(base_y, kShift) + kOff);
  loc.step_x = static_cast<std::int32_t>(kScale >> (kRefScaleShift - kScaleSubpelBits));
  loc.step_y = loc.step_x;
  return loc;
}

// Number of intermediate rows required by §7.13.3.18:
//   (((h - 1) * yStep + (1 << 10) - 1) >> 10) + 8
AV2ENC_MC_HD constexpr int mc_intermediate_rows(int h, int step_y) {
  return (((h - 1) * step_y + (1 << kScaleSubpelBits) - 1) >>
          kScaleSubpelBits) +
         8;
}

// One horizontal 8-tap convolution at an already-derived integer column.
// `col` is the position of tap 0 (i.e. (p >> 10) - 3); taps are clipped to
// [first_x, last_x].  Returns the unrounded sum (Round2(.., InterRound0) is
// applied by the caller so the CPU can store the intermediate array).
AV2ENC_MC_HD std::int32_t mc_horizontal_sum(const std::uint8_t* ref,
                                            int stride, int row, int col,
                                            int first_x, int last_x,
                                            const std::int16_t* filter) {
  std::int32_t sum = 0;
  for (int t = 0; t < 8; ++t) {
    const int cx = mc_clip3(first_x, last_x, col + t);
    sum += static_cast<std::int32_t>(filter[t]) *
           static_cast<std::int32_t>(ref[static_cast<std::size_t>(row) *
                                             static_cast<std::size_t>(stride) +
                                         static_cast<std::size_t>(cx)]);
  }
  return sum;
}

// The complete per-sample two-dimensional filter used by the CUDA kernel and
// by the CPU cross-check.  `intermediate` is not materialised: the eight
// horizontal sums are recomputed for each output sample.
//
// `r`/`c` are output sample coordinates; `p_x`/`p_y` are the 1/1024-sample
// positions from §7.13.3.18.
AV2ENC_MC_HD std::uint8_t mc_predict_sample(
    const std::uint8_t* ref, int stride, int first_x, int first_y, int last_x,
    int last_y, std::int32_t start_x, std::int32_t start_y, std::int32_t step_x,
    std::int32_t step_y, const std::int16_t* filter_h,
    const std::int16_t* filter_v, int r, int c) {
  const std::int32_t p_x = start_x + step_x * c;
  const std::int32_t col = (p_x >> kScaleSubpelBits) - 3;
  const int phase_h = static_cast<int>((p_x >> 6) & kSubpelMask);

  const std::int32_t p_y = (start_y & 1023) + step_y * r;
  const int row_base = static_cast<int>(p_y >> kScaleSubpelBits);
  const int phase_v = static_cast<int>((p_y >> 6) & kSubpelMask);

  const std::int16_t* fh = filter_h + phase_h * kSubpelTaps;
  const std::int16_t* fv = filter_v + phase_v * kSubpelTaps;

  std::int32_t sum_v = 0;
  for (int t = 0; t < 8; ++t) {
    const int row = mc_clip3(first_y, last_y, (start_y >> 10) + row_base + t -
                                                   3);
    const std::int32_t hsum =
        mc_horizontal_sum(ref, stride, row, col, first_x, last_x, fh);
    const std::int32_t intermediate = mc_round2(hsum, kInterRound0);
    sum_v += static_cast<std::int32_t>(fv[t]) * intermediate;
  }
  return mc_clip1(mc_round2(sum_v, kInterRound1));
}

}  // namespace av2enc::inter::mc

#endif  // AV2ENC_ENCODER_INTER_MC_MC_CORE_H_
