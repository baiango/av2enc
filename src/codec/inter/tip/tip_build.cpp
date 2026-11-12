// AV2 v1.0.0 TIP build — motion field to samples (§7.10.6, §7.22).

#include "codec/inter/tip/tip_build.h"

#include <algorithm>
#include <cstddef>

#include "codec/inter/temporal/motion_field.h"
#include "codec/inter/temporal/temporal_math.h"
#include "codec/inter/temporal/tip.h"

namespace av2enc::inter::tip {
namespace {

int count_valid(const MotionField& field) {
  int n = 0;
  const std::size_t cells = static_cast<std::size_t>(field.w8) *
                            static_cast<std::size_t>(field.h8);
  for (std::size_t i = 0; i < cells; ++i) {
    n += field.valid[i] != 0 ? 1 : 0;
  }
  return n;
}

bool valid_plane(const PlaneView& p) {
  return p.data != nullptr && p.stride > 0 && p.width > 0 && p.height > 0;
}

}  // namespace

void fill_tip_closest_refs(const MotionFieldConfig& config,
                           const MotionFieldResult& result, TipConfig* tip) {
  if (tip == nullptr) return;
  tip->num_past_refs = config.num_past_refs;
  tip->num_future_refs = config.num_future_refs;
  tip->order_hint = config.order_hint;
  if (result.closest_past >= 0 &&
      result.closest_past < temporal::kRefsPerFrame) {
    tip->closest_past_order_hint = config.ref_order_hint[result.closest_past];
  } else {
    tip->closest_past_order_hint = temporal::kRestrictedOrderHint;
  }
  if (result.closest_future >= 0 &&
      result.closest_future < temporal::kRefsPerFrame) {
    tip->closest_future_order_hint =
        config.ref_order_hint[result.closest_future];
  } else {
    tip->closest_future_order_hint = temporal::kRestrictedOrderHint;
  }
}

TemporalStatus predict_tip_plan(const TipPlan& plan, const TipConfig& tip,
                                const MotionFieldConfig& config,
                                const PlaneView ref[3][2],
                                std::uint8_t* const dst[3],
                                const int dst_stride[3],
                                const int dst_width[3],
                                const int dst_height[3],
                                const int sub_x[3], const int sub_y[3],
                                std::int16_t* pred_scratch,
                                std::uint8_t* block_scratch,
                                TipBuildOutput* out) {
  if (tip.mode == temporal::TipFrameMode::kDisabled || plan.blocks == nullptr ||
      dst[0] == nullptr || pred_scratch == nullptr ||
      block_scratch == nullptr) {
    return TemporalStatus::kInvalidArgument;
  }
  if (config.mi_cols <= 0 || config.mi_rows <= 0) {
    return TemporalStatus::kInvalidArgument;
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (dst[plane] == nullptr || dst_stride[plane] < dst_width[plane] ||
        dst_width[plane] <= 0 || dst_height[plane] <= 0 ||
        sub_x[plane] < 0 || sub_x[plane] > 1 || sub_y[plane] < 0 ||
        sub_y[plane] > 1 || !valid_plane(ref[plane][0]) ||
        !valid_plane(ref[plane][1])) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  const int frame_w = config.mi_cols * temporal::kMiSize;
  const int frame_h = config.mi_rows * temporal::kMiSize;
  for (int b = 0; b < plan.num_blocks; ++b) {
    const temporal::TipBlock& block = plan.blocks[b];
    for (int plane = 0; plane < 3; ++plane) {
      const int w = (block.bw4 * temporal::kMiSize) >> sub_x[plane];
      const int h = (block.bh4 * temporal::kMiSize) >> sub_y[plane];
      const int x = (block.mi_col * temporal::kMiSize) >> sub_x[plane];
      const int y = (block.mi_row * temporal::kMiSize) >> sub_y[plane];
      const int vis_w = std::min(w, dst_width[plane] - x);
      const int vis_h = std::min(h, dst_height[plane] - y);
      if (vis_w <= 0 || vis_h <= 0) continue;
      temporal::TipPredictInput in;
      in.ref[0] = ref[plane][0];
      in.ref[1] = ref[plane][1];
      in.num_refs = 2;
      in.x = x;
      in.y = y;
      in.w = w;
      in.h = h;
      in.sub_x = sub_x[plane];
      in.sub_y = sub_y[plane];
      in.frame_w = frame_w;
      in.frame_h = frame_h;
      in.mv[0] = block.mv[0];
      in.mv[1] = block.mv[1];
      in.cwp_weight = plan.weight;
      in.filter = tip.interp_filter;
      in.use_ref_area = false;
      in.dst = block_scratch;
      in.dst_stride = w;
      in.scratch0 = pred_scratch;
      in.scratch1 = pred_scratch + 16 * 16;
      in.scratch_stride = w;
      const TemporalStatus status = temporal::tip_predict_block(in);
      if (status != TemporalStatus::kOk) return status;
      for (int r = 0; r < vis_h; ++r) {
        std::copy_n(block_scratch + static_cast<std::size_t>(r) * w, vis_w,
                    dst[plane] +
                        static_cast<std::ptrdiff_t>(y + r) * dst_stride[plane] +
                        x);
      }
      if (out != nullptr) {
        out->predicted_samples[plane] +=
            static_cast<std::int64_t>(vis_w) * vis_h;
      }
    }
    if (out != nullptr) ++out->predicted_blocks;
  }
  if (out != nullptr) out->num_blocks = plan.num_blocks;
  return TemporalStatus::kOk;
}

TemporalStatus build_tip_frame(const TipBuildInput& in, TipBuildOutput* out) {
  if (in.field == nullptr || in.plan == nullptr || in.tip.mode !=
                                                   temporal::TipFrameMode::kAsOutput) {
    return TemporalStatus::kInvalidArgument;
  }
  if (in.tip.allow_tip_hole_fill &&
      (in.scratch == nullptr || in.scratch->avg_mv == nullptr ||
       in.scratch->avg_valid == nullptr)) {
    return TemporalStatus::kInvalidArgument;
  }
  if (out != nullptr) *out = TipBuildOutput{};

  MotionFieldResult result;
  TemporalStatus status =
      temporal::motion_field_estimation(in.config, in.field, nullptr, &result);
  if (status != TemporalStatus::kOk) return status;
  // §7.10.1: TIP output needs both ClosestPast and ClosestFuture.
  if (result.closest_past < 0 || result.closest_future < 0) {
    return TemporalStatus::kInvalidArgument;
  }
  const int estimation_valid = count_valid(*in.field);

  TipConfig tip = in.tip;
  fill_tip_closest_refs(in.config, result, &tip);
  status = temporal::setup_tip_motion_field(in.config, tip, in.field, nullptr,
                                            in.scratch, &result);
  if (status != TemporalStatus::kOk) return status;
  const int setup_valid = count_valid(*in.field);

  status = temporal::build_tip_plan(in.config, tip, *in.field, in.plan);
  if (status != TemporalStatus::kOk) return status;

  TemporalStatus predict = predict_tip_plan(
      *in.plan, tip, in.config, in.ref, in.dst, in.dst_stride, in.dst_width,
      in.dst_height, in.sub_x, in.sub_y, in.pred_scratch,
      in.block_scratch, out);
  if (predict != TemporalStatus::kOk) return predict;

  if (out != nullptr) {
    out->field_result = result;
    out->field_valid_after_estimation = estimation_valid;
    out->field_valid_after_setup = setup_valid;
    out->hole_filled = std::max(0, setup_valid - estimation_valid);
    out->field_invalid = in.config.mi_cols * in.config.mi_rows / 4 -
                         setup_valid;
    out->num_blocks = in.plan->num_blocks;
  }
  return TemporalStatus::kOk;
}

TemporalStatus store_tip_frame_motion_field(const TipPlan& plan,
                                            const TipStoreInput& in,
                                            temporal::RefMvField* out) {
  if (out == nullptr || out->ref == nullptr || out->mv == nullptr ||
      in.mi_cols <= 0 || in.mi_rows <= 0 ||
      in.mi_cols != out->w8 * 2 || in.mi_rows != out->h8 * 2 ||
      plan.blocks == nullptr || plan.mv_grid == nullptr ||
      plan.mv_grid1 == nullptr || in.closest_past < 0 ||
      in.closest_past >= temporal::kRefsPerFrame || in.closest_future < 0 ||
      in.closest_future >= temporal::kRefsPerFrame) {
    return TemporalStatus::kInvalidArgument;
  }
  for (int b = 0; b < plan.num_blocks; ++b) {
    const temporal::TipBlock& block = plan.blocks[b];
    // §7.22 clamps the block extent to the frame before deriving the 8x8
    // loop bounds; the last block can be 2 MI rows tall on a 360-line frame.
    const int bw4 = std::min(block.bw4, in.mi_cols - block.mi_col);
    const int bh4 = std::min(block.bh4, in.mi_rows - block.mi_row);
    if (bw4 <= 0 || bh4 <= 0) continue;
    const int rows8 = (bh4 + 1) >> 1;  // Round2(bh4, 1)
    const int cols8 = (bw4 + 1) >> 1;
    for (int i8 = 0; i8 < rows8; ++i8) {
      for (int j8 = 0; j8 < cols8; ++j8) {
        temporal::MotionFieldStoreEntry entry;
        entry.refs[0] = in.closest_past;
        entry.refs[1] = in.weight16 ? temporal::kRefNone : in.closest_future;
        const int row = block.mi_row + (i8 << 1);
        const int col = block.mi_col + (j8 << 1);
        const std::size_t grid =
            static_cast<std::size_t>(row) *
                static_cast<std::size_t>(in.mi_cols) +
            static_cast<std::size_t>(col);
        entry.mv[0] = plan.mv_grid[grid];
        entry.mv[1] = plan.mv_grid1[grid];
        entry.is_wedge = false;
        entry.mask = nullptr;
        entry.mask_stride = 0;
        entry.i8 = 0;
        entry.j8 = 0;
        entry.order_hint = in.order_hint;
        entry.ref_order_hint[0] = in.closest_past_order_hint;
        entry.ref_order_hint[1] = in.closest_future_order_hint;
        const int y8 = (block.mi_row >> 1) + i8;
        const int x8 = (block.mi_col >> 1) + j8;
        const TemporalStatus status =
            temporal::store_motion_field_entry(entry, out, y8, x8);
        if (status != TemporalStatus::kOk) return status;
      }
    }
  }
  return TemporalStatus::kOk;
}

}  // namespace av2enc::inter::tip
