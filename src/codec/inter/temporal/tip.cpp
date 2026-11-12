// AV2 v1.0.0 temporal inter tools — TIP block prediction.

#include "codec/inter/temporal/tip.h"

#include "codec/inter/temporal/temporal_predict.h"

namespace av2enc::inter::temporal {

TemporalStatus tip_predict_block(const TipPredictInput& in) {
  if (in.dst == nullptr || in.dst_stride < in.w || in.w <= 0 || in.h <= 0 ||
      in.frame_w <= 0 || in.frame_h <= 0 || in.num_refs < 1 ||
      in.num_refs > 2 || in.ref[0].data == nullptr) {
    return TemporalStatus::kInvalidArgument;
  }
  const bool single = in.num_refs == 1 || in.cwp_weight == 16;
  if (single) {
    const RefArea area = in.use_ref_area
                             ? in.area[0]
                             : RefArea{0, 0, in.ref[0].width - 1,
                                       in.ref[0].height - 1};
    if (!predict_plane_u8(in.ref[0], in.x, in.y, in.w, in.h, in.mv[0],
                          in.filter, in.sub_x, in.sub_y, false,
                          in.ref[0].width, in.ref[0].height, in.frame_w,
                          in.frame_h, area, in.dst, in.dst_stride)) {
      return TemporalStatus::kInvalidArgument;
    }
    return TemporalStatus::kOk;
  }
  if (in.ref[1].data == nullptr || in.scratch0 == nullptr ||
      in.scratch1 == nullptr || in.scratch_stride < in.w) {
    return TemporalStatus::kInvalidArgument;
  }
  const RefArea area0 = in.use_ref_area
                            ? in.area[0]
                            : RefArea{0, 0, in.ref[0].width - 1,
                                      in.ref[0].height - 1};
  const RefArea area1 = in.use_ref_area
                            ? in.area[1]
                            : RefArea{0, 0, in.ref[1].width - 1,
                                      in.ref[1].height - 1};
  if (!predict_plane_1_16(in.ref[0], in.x, in.y, in.w, in.h, in.mv[0],
                          in.filter, in.sub_x, in.sub_y, false,
                          in.ref[0].width, in.ref[0].height, in.frame_w,
                          in.frame_h, area0, in.scratch0,
                          in.scratch_stride)) {
    return TemporalStatus::kInvalidArgument;
  }
  if (!predict_plane_1_16(in.ref[1], in.x, in.y, in.w, in.h, in.mv[1],
                          in.filter, in.sub_x, in.sub_y, false,
                          in.ref[1].width, in.ref[1].height, in.frame_w,
                          in.frame_h, area1, in.scratch1,
                          in.scratch_stride)) {
    return TemporalStatus::kInvalidArgument;
  }
  compound_average(in.scratch0, in.scratch_stride, in.scratch1,
                   in.scratch_stride, in.w, in.h, in.cwp_weight, in.dst,
                   in.dst_stride);
  return TemporalStatus::kOk;
}

}  // namespace av2enc::inter::temporal
