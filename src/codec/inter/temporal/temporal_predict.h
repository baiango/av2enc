// AV2 v1.0.0 temporal inter tools — prediction helpers.
//
// Two-pass subpel prediction (§7.13.3.18) used by TIP and refine-MV.  The
// arithmetic is the same as src/encoder/inter/mc/ (shared filter table and
// §7.13.3.17 sample-location derivation), but this version is parameterised by
// the rounding stage:
//
//   InterRound1 = 11 -> integer 8-bit prediction (single reference, refine-MV
//                      SAD search, TIP weight 16)
//   InterRound1 = 7  -> 1/16 sample precision (compound / TIP blend / OPFL)
//
// Reference reads are clipped to an inclusive area.  That is the §7.13.3.4
// `useRefArea` behaviour used by refine-MV; pass the full plane for ordinary
// prediction.

#ifndef AV2ENC_ENCODER_INTER_TEMPORAL_TEMPORAL_PREDICT_H_
#define AV2ENC_ENCODER_INTER_TEMPORAL_TEMPORAL_PREDICT_H_

#include <cstdint>

#include "codec/inter/advanced/warp_common.h"
#include "codec/inter/mc/mc_core.h"
#include "codec/inter/temporal/temporal_math.h"
#include "codec/inter/temporal/temporal_types.h"

namespace av2enc::inter::temporal {

// One output sample.  `first_*`/`last_*` are the inclusive reference area.
template <int kRound1, bool kClip>
inline int32_t predict_sample(const PlaneView& ref, int first_x, int first_y,
                              int last_x, int last_y, std::int32_t start_x,
                              std::int32_t start_y, std::int32_t step_x,
                              std::int32_t step_y, InterpFilter filter_h,
                              InterpFilter filter_v, int r, int c) {
  const std::int32_t p_x = start_x + step_x * c;
  const int col = static_cast<int>(p_x >> kScaleSubpelBits) - 3;
  const int phase_h = static_cast<int>((p_x >> 6) & (mc::kSubpelMask));
  const std::int16_t* fh =
      &mc::kSubpelFilters[static_cast<int>(filter_h)][phase_h][0];

  const std::int32_t p_y = (start_y & 1023) + step_y * r;
  const int row_base = static_cast<int>(p_y >> kScaleSubpelBits);
  const int phase_v = static_cast<int>((p_y >> 6) & (mc::kSubpelMask));
  const std::int16_t* fv =
      &mc::kSubpelFilters[static_cast<int>(filter_v)][phase_v][0];

  std::int32_t sum_v = 0;
  for (int t = 0; t < 8; ++t) {
    const int row =
        clip3(first_y, last_y,
              static_cast<int>(start_y >> kScaleSubpelBits) + row_base + t - 3);
    const std::int32_t hsum = mc::mc_horizontal_sum(
        ref.data, ref.stride, row, col, first_x, last_x, fh);
    const std::int32_t intermediate = round2(hsum, kInterRound0);
    sum_v += static_cast<std::int32_t>(fv[t]) * intermediate;
  }
  const std::int32_t result = round2(sum_v, kRound1);
  return kClip ? static_cast<std::int32_t>(clip1(result)) : result;
}

// Predicts a rectangle into an int16 plane at 1/16 sample precision.
// `out_stride` is in samples.  Returns false for invalid geometry.
inline bool predict_plane_1_16(const PlaneView& ref, int x, int y, int w,
                               int h, Mv mv, InterpFilter filter, int sub_x,
                               int sub_y, bool prescaled, int ref_w,
                               int ref_h, int frame_w, int frame_h,
                               const RefArea& area, std::int16_t* out,
                               int out_stride) {
  if (ref.data == nullptr || out == nullptr || w <= 0 || h <= 0 ||
      out_stride < w || ref_w <= 0 || ref_h <= 0 || frame_w <= 0 ||
      frame_h <= 0 || area.first_x < 0 || area.first_y < 0 ||
      area.last_x >= ref_w || area.last_y >= ref_h ||
      area.first_x > area.last_x || area.first_y > area.last_y) {
    return false;
  }
  const int mv_pair[2] = {mv.row, mv.col};
  // §7.13.3.17 scales with the LUMA reference/frame dimensions.  `ref_w` and
  // `ref_h` are the plane extent; recover the MI-aligned luma dimensions from
  // the chroma subsampling.
  const int ref_luma_w = ref_w << sub_x;
  const int ref_luma_h = ref_h << sub_y;
  const adv::MvScaleOutput loc = adv::motion_vector_scaling(
      sub_x > 0 ? 1 : 0, sub_x, sub_y, x, y, mv_pair, prescaled, ref_luma_w,
      ref_luma_h, frame_w, frame_h);
  const InterpFilter filter_h = mc::mc_pass_filter(filter, w <= 4);
  const InterpFilter filter_v = mc::mc_pass_filter(filter, h <= 4);
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      const std::int32_t value =
          predict_sample<kInterRound1Compound, false>(
              ref, area.first_x, area.first_y, area.last_x, area.last_y,
              loc.start_x, loc.start_y, loc.step_x, loc.step_y, filter_h,
              filter_v, r, c);
      out[static_cast<std::size_t>(r) * static_cast<std::size_t>(out_stride) +
          static_cast<std::size_t>(c)] = static_cast<std::int16_t>(value);
    }
  }
  return true;
}

// Predicts a rectangle as clipped 8-bit samples (single-reference rounding).
inline bool predict_plane_u8(const PlaneView& ref, int x, int y, int w, int h,
                             Mv mv, InterpFilter filter, int sub_x, int sub_y,
                             bool prescaled, int ref_w, int ref_h,
                             int frame_w, int frame_h, const RefArea& area,
                             std::uint8_t* out, int out_stride) {
  if (ref.data == nullptr || out == nullptr || w <= 0 || h <= 0 ||
      out_stride < w || ref_w <= 0 || ref_h <= 0 || frame_w <= 0 ||
      frame_h <= 0 || area.first_x < 0 || area.first_y < 0 ||
      area.last_x >= ref_w || area.last_y >= ref_h ||
      area.first_x > area.last_x || area.first_y > area.last_y) {
    return false;
  }
  const int mv_pair[2] = {mv.row, mv.col};
  // §7.13.3.17 scales with the LUMA reference/frame dimensions.  `ref_w` and
  // `ref_h` are the plane extent; recover the MI-aligned luma dimensions from
  // the chroma subsampling.
  const int ref_luma_w = ref_w << sub_x;
  const int ref_luma_h = ref_h << sub_y;
  const adv::MvScaleOutput loc = adv::motion_vector_scaling(
      sub_x > 0 ? 1 : 0, sub_x, sub_y, x, y, mv_pair, prescaled, ref_luma_w,
      ref_luma_h, frame_w, frame_h);
  const InterpFilter filter_h = mc::mc_pass_filter(filter, w <= 4);
  const InterpFilter filter_v = mc::mc_pass_filter(filter, h <= 4);
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      const std::int32_t value = predict_sample<kInterRound1Single, true>(
          ref, area.first_x, area.first_y, area.last_x, area.last_y,
          loc.start_x, loc.start_y, loc.step_x, loc.step_y, filter_h, filter_v,
          r, c);
      out[static_cast<std::size_t>(r) * static_cast<std::size_t>(out_stride) +
          static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(value);
    }
  }
  return true;
}

// §7.13.3.1 compound average for COMPOUND_AVERAGE:
//   Clip1(Round2(cwp*Preds[0] + (16-cwp)*Preds[1], 4 + InterPostRound))
// with InterPostRound = 4 for compound predictions.
inline void compound_average(const std::int16_t* p0, int p0_stride,
                             const std::int16_t* p1, int p1_stride, int w,
                             int h, int cwp, std::uint8_t* dst,
                             int dst_stride) {
  const int shift = 4 + kInterPostRoundCompound;  // 8
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      const std::int32_t v0 =
          p0[static_cast<std::size_t>(r) * static_cast<std::size_t>(p0_stride) +
             static_cast<std::size_t>(c)];
      const std::int32_t v1 =
          p1[static_cast<std::size_t>(r) * static_cast<std::size_t>(p1_stride) +
             static_cast<std::size_t>(c)];
      const std::int32_t mixed =
          static_cast<std::int32_t>(cwp) * v0 +
          static_cast<std::int32_t>(16 - cwp) * v1;
      dst[static_cast<std::size_t>(r) * static_cast<std::size_t>(dst_stride) +
          static_cast<std::size_t>(c)] =
          static_cast<std::uint8_t>(clip1(round2(mixed, shift)));
    }
  }
}

// SAD used by the refine-MV search: even rows only, full width, 8-bit
// precision, then >> (BitDepth - 8) (a no-op for 8-bit).
inline std::int64_t refine_sad(const std::uint8_t* p0, int p0_stride,
                               const std::uint8_t* p1, int p1_stride, int w,
                               int h) {
  std::int64_t sad = 0;
  for (int r = 0; r < h; r += 2) {
    for (int c = 0; c < w; ++c) {
      const int a =
          p0[static_cast<std::size_t>(r) * static_cast<std::size_t>(p0_stride) +
             static_cast<std::size_t>(c)];
      const int b =
          p1[static_cast<std::size_t>(r) * static_cast<std::size_t>(p1_stride) +
             static_cast<std::size_t>(c)];
      sad += a > b ? a - b : b - a;
    }
  }
  return sad;  // >> (BitDepth - 8) is zero for 8-bit
}

}  // namespace av2enc::inter::temporal

#endif  // AV2ENC_ENCODER_INTER_TEMPORAL_TEMPORAL_PREDICT_H_
