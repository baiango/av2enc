// AV2 v1.0.0 temporal inter tools — refine-MV implementation.

#include "codec/inter/temporal/refinemv.h"

#include <algorithm>

#include "codec/inter/temporal/temporal_math.h"
#include "codec/inter/temporal/temporal_predict.h"

namespace av2enc::inter::temporal {
namespace {

// Refinemv_Neighbors[24][2] from §7.13.3.6, in (dy, dx) order.
constexpr int kRefinemvNeighbors[24][2] = {
    {-2, -2}, {-2, -1}, {-2, 0}, {-2, 1}, {-2, 2}, {-1, -2},
    {-1, -1}, {-1, 0},  {-1, 1}, {-1, 2}, {0, -2},  {0, -1},
    {0, 1},   {0, 2},   {1, -2}, {1, -1}, {1, 0},   {1, 1},
    {1, 2},   {2, -2},  {2, -1}, {2, 0},  {2, 1},   {2, 2}};

int scaled_start_x(const RefAreaInput& in, int list, Mv mv) {
  const int mv_pair[2] = {mv.row, mv.col};
  const adv::MvScaleOutput loc = adv::motion_vector_scaling(
      0, in.sub_x, in.sub_y, in.x, in.y, mv_pair, false, in.ref_luma_w[list],
      in.ref_luma_h[list], in.frame_w, in.frame_h);
  return loc.start_x;
}
int scaled_step_x(const RefAreaInput& in, int list, Mv mv) {
  const int mv_pair[2] = {mv.row, mv.col};
  const adv::MvScaleOutput loc = adv::motion_vector_scaling(
      0, in.sub_x, in.sub_y, in.x, in.y, mv_pair, false, in.ref_luma_w[list],
      in.ref_luma_h[list], in.frame_w, in.frame_h);
  return loc.step_x;
}
int scaled_start_y(const RefAreaInput& in, int list, Mv mv) {
  const int mv_pair[2] = {mv.row, mv.col};
  const adv::MvScaleOutput loc = adv::motion_vector_scaling(
      0, in.sub_x, in.sub_y, in.x, in.y, mv_pair, false, in.ref_luma_w[list],
      in.ref_luma_h[list], in.frame_w, in.frame_h);
  return loc.start_y;
}
int scaled_step_y(const RefAreaInput& in, int list, Mv mv) {
  const int mv_pair[2] = {mv.row, mv.col};
  const adv::MvScaleOutput loc = adv::motion_vector_scaling(
      0, in.sub_x, in.sub_y, in.x, in.y, mv_pair, false, in.ref_luma_w[list],
      in.ref_luma_h[list], in.frame_w, in.frame_h);
  return loc.step_y;
}

}  // namespace

bool refine_mv_allowed_mode(const RefineMvConfig& ctx) {
  if (ctx.skip_mode || ctx.is_global_global_mv || !ctx.motion_mode_simple) {
    return false;
  }
  if (ctx.opfl_refine_type == RefineType::kSwitchable && ctx.has_newmv &&
      !ctx.use_optflow) {
    return false;
  }
  return true;
}

bool refine_mv_switchable(const RefineMvConfig& ctx) {
  if (ctx.y_mode_near_near_mv ||
      (ctx.y_mode_joint_new_mv && ctx.use_optflow &&
       ctx.opfl_refine_type == RefineType::kSwitchable)) {
    return false;
  }
  return true;
}

bool refine_mv_allowed_reference(const RefineMvConfig& ctx) {
  if (ctx.frame_is_switch || ctx.ref0_scaled || ctx.ref1_scaled) {
    return false;
  }
  const int d0 = get_relative_dist(ctx.order_hint, ctx.ref0_order_hint);
  const int d1 = get_relative_dist(ctx.order_hint, ctx.ref1_order_hint);
  return d0 != 0 && d0 == -d1;
}

TemporalStatus refine_mv_syntax(const RefineMvConfig& ctx, RefineMvSyntax* out) {
  if (out == nullptr) {
    return TemporalStatus::kInvalidArgument;
  }
  out->use_refinemv = false;
  out->decided_against_refinemv = false;
  out->switchable = false;
  const bool size_ok =
      ctx.block_width >= 8 && ctx.block_height >= 8 &&
      (ctx.block_width >= 16 || ctx.block_height >= 16);
  if (!ctx.enable_refinemv || !ctx.is_compound || !size_ok ||
      !refine_mv_allowed_mode(ctx) || !refine_mv_allowed_reference(ctx)) {
    return TemporalStatus::kOk;
  }
  out->switchable = refine_mv_switchable(ctx);
  if (out->switchable) {
    out->use_refinemv = ctx.use_refinemv_bit;
  } else {
    out->use_refinemv = true;
  }
  return TemporalStatus::kOk;
}

MvPair offset_refinemv(const MvPair& src, int dx, int dy) {
  MvPair out;
  out.mv[0].row = src.mv[0].row + dy * 8;
  out.mv[0].col = src.mv[0].col + dx * 8;
  out.mv[1].row = src.mv[1].row - dy * 8;
  out.mv[1].col = src.mv[1].col - dx * 8;
  return out;
}

TemporalStatus compute_ref_area(const RefAreaInput& in, RefArea out[2]) {
  if (out == nullptr || in.w <= 0 || in.h <= 0 || in.frame_w <= 0 ||
      in.frame_h <= 0 || in.num_refs < 1 || in.num_refs > 2) {
    return TemporalStatus::kInvalidArgument;
  }
  for (int list = 0; list < in.num_refs; ++list) {
    const int last_x =
        ((in.ref_mi_cols[list] * kMiSize) >> in.sub_x) - 1;
    const int last_y =
        ((in.ref_mi_rows[list] * kMiSize) >> in.sub_y) - 1;
    if (last_x < 0 || last_y < 0) {
      return TemporalStatus::kInvalidArgument;
    }
    const Mv mv = in.cand_mvs[list];
    const int start_x = scaled_start_x(in, list, mv);
    const int step_x = scaled_step_x(in, list, mv);
    const int start_y = scaled_start_y(in, list, mv);
    const int step_y = scaled_step_y(in, list, mv);
    if (in.w == 4) {
      out[list].first_x = clip3(0, last_x, (start_x >> 10) - 1);
      out[list].last_x =
          clip3(0, last_x, ((start_x + step_x * (in.w - 1)) >> 10) + 2);
    } else {
      out[list].first_x = clip3(0, last_x, (start_x >> 10) - 3);
      out[list].last_x =
          clip3(0, last_x, ((start_x + step_x * (in.w - 1)) >> 10) + 4);
    }
    if (in.h == 4) {
      out[list].first_y = clip3(0, last_y, (start_y >> 10) - 1);
      out[list].last_y =
          clip3(0, last_y, ((start_y + step_y * (in.h - 1)) >> 10) + 2);
    } else {
      out[list].first_y = clip3(0, last_y, (start_y >> 10) - 3);
      out[list].last_y =
          clip3(0, last_y, ((start_y + step_y * (in.h - 1)) >> 10) + 4);
    }
  }
  return TemporalStatus::kOk;
}

namespace {

std::int64_t sad_refinemv(const RefineSearchInput& in, int dx, int dy,
                          int x, int y, int w, int h,
                          const RefArea area[2]) {
  const MvPair mvs = offset_refinemv(MvPair{in.cand_mvs[0], in.cand_mvs[1]},
                                     dx, dy);
  RefArea a0 = area[0];
  RefArea a1 = area[1];
  if (!in.use_ref_area) {
    a0 = RefArea{0, 0, in.ref[0].width - 1, in.ref[0].height - 1};
    a1 = RefArea{0, 0, in.ref[1].width - 1, in.ref[1].height - 1};
  }
  // A scale source of 0 means "the caller only supplied the plane extent"
  // (the historical component/test form); derive the MI-aligned luma size.
  const int lw0 = in.ref_luma_w[0] > 0 ? in.ref_luma_w[0] : in.ref[0].width;
  const int lh0 = in.ref_luma_h[0] > 0 ? in.ref_luma_h[0] : in.ref[0].height;
  const int lw1 = in.ref_luma_w[1] > 0 ? in.ref_luma_w[1] : in.ref[1].width;
  const int lh1 = in.ref_luma_h[1] > 0 ? in.ref_luma_h[1] : in.ref[1].height;
  if (!predict_plane_u8_ex(in.ref[0], x, y, w, h, mvs.mv[0],
                           InterpFilter::kBilinear, 0, 0, false, lw0, lh0,
                           in.ref[0].width, in.ref[0].height, in.frame_w,
                           in.frame_h, a0, in.scratch0, in.scratch_stride)) {
    return -1;
  }
  if (!predict_plane_u8_ex(in.ref[1], x, y, w, h, mvs.mv[1],
                           InterpFilter::kBilinear, 0, 0, false, lw1, lh1,
                           in.ref[1].width, in.ref[1].height, in.frame_w,
                           in.frame_h, a1, in.scratch1, in.scratch_stride)) {
    return -1;
  }
  return refine_sad(in.scratch0, in.scratch_stride, in.scratch1,
                    in.scratch_stride, w, h);
}

}  // namespace

TemporalStatus search_refinemv(const RefineSearchInput& in,
                               RefineSearchResult* out) {
  if (out == nullptr) {
    return TemporalStatus::kInvalidArgument;
  }
  out->dx = 0;
  out->dy = 0;
  out->offset_mvs = MvPair{in.cand_mvs[0], in.cand_mvs[1]};
  out->best_sad = 0;
  if (in.w <= 0 || in.h <= 0 || in.num_refs < 2 || in.ref[0].data == nullptr ||
      in.ref[1].data == nullptr || in.scratch0 == nullptr ||
      in.scratch1 == nullptr || in.scratch_stride < in.w + 4) {
    return TemporalStatus::kInvalidArgument;
  }
  for (int list = 0; list < 2; ++list) {
    const Mv mv = in.cand_mvs[list];
    if (mv.row - 4 * 8 < kMvLow + 1 || mv.row + 2 * 8 > kMvUpp - 1 ||
        mv.col - 4 * 8 < kMvLow + 1 || mv.col + 2 * 8 > kMvUpp - 1) {
      return TemporalStatus::kOk;
    }
  }

  RefArea area[2] = {in.area[0], in.area[1]};
  if (!in.use_ref_area) {
    area[0] = RefArea{0, 0, in.ref[0].width - 1, in.ref[0].height - 1};
    area[1] = RefArea{0, 0, in.ref[1].width - 1, in.ref[1].height - 1};
  }
  const int x = in.x - 2;
  const int y = in.y - 2;
  const int w = in.w + 4;
  const int h = in.h + 4;
  const bool allow_centre = in.tip_pred || !in.switchable;
  int best_dy = 0;
  int best_dx = 0;
  std::int64_t best_sad = 0;
  const std::int64_t th =
      static_cast<std::int64_t>(w) * static_cast<std::int64_t>(h) * 2;

  if (allow_centre) {
    best_sad = sad_refinemv(in, 0, 0, x, y, w, h, area);
    if (best_sad < 0) {
      return TemporalStatus::kInvalidArgument;
    }
    best_sad -= best_sad >> 3;
    if (best_sad < th) {
      out->best_sad = best_sad;
      return TemporalStatus::kOk;
    }
  }
  for (int idx = 0; idx < 24; ++idx) {
    const int try_dy = kRefinemvNeighbors[idx][0];
    const int try_dx = kRefinemvNeighbors[idx][1];
    const std::int64_t sad =
        sad_refinemv(in, try_dx, try_dy, x, y, w, h, area);
    if (sad < 0) {
      return TemporalStatus::kInvalidArgument;
    }
    if ((idx == 0 && !allow_centre) || sad < best_sad) {
      best_dy = try_dy;
      best_dx = try_dx;
      best_sad = sad;
    }
  }
  out->dx = best_dx;
  out->dy = best_dy;
  out->offset_mvs = offset_refinemv(
      MvPair{in.cand_mvs[0], in.cand_mvs[1]}, best_dx, best_dy);
  out->best_sad = best_sad;
  return TemporalStatus::kOk;
}

TemporalStatus refine_predict_block(const RefinePredictInput& in) {
  if (in.dst == nullptr || in.dst_stride < in.w || in.w <= 0 || in.h <= 0 ||
      in.num_refs < 2 || in.ref[0].data == nullptr ||
      in.ref[1].data == nullptr || in.scratch0 == nullptr ||
      in.scratch1 == nullptr || in.scratch_stride < in.w) {
    return TemporalStatus::kInvalidArgument;
  }
  RefArea a0 = in.area[0];
  RefArea a1 = in.area[1];
  if (!in.use_ref_area) {
    a0 = RefArea{0, 0, in.ref[0].width - 1, in.ref[0].height - 1};
    a1 = RefArea{0, 0, in.ref[1].width - 1, in.ref[1].height - 1};
  }
  const int lw0 = in.ref_luma_w[0] > 0 ? in.ref_luma_w[0]
                                       : (in.ref[0].width << in.sub_x);
  const int lh0 = in.ref_luma_h[0] > 0 ? in.ref_luma_h[0]
                                       : (in.ref[0].height << in.sub_y);
  const int lw1 = in.ref_luma_w[1] > 0 ? in.ref_luma_w[1]
                                       : (in.ref[1].width << in.sub_x);
  const int lh1 = in.ref_luma_h[1] > 0 ? in.ref_luma_h[1]
                                       : (in.ref[1].height << in.sub_y);
  if (!predict_plane_1_16_ex(in.ref[0], in.x, in.y, in.w, in.h,
                             in.offset_mvs.mv[0], InterpFilter::kEighttapSharp,
                             in.sub_x, in.sub_y, false, lw0, lh0,
                             in.ref[0].width, in.ref[0].height, in.frame_w,
                             in.frame_h, a0, in.scratch0, in.scratch_stride)) {
    return TemporalStatus::kInvalidArgument;
  }
  if (!predict_plane_1_16_ex(in.ref[1], in.x, in.y, in.w, in.h,
                             in.offset_mvs.mv[1], InterpFilter::kEighttapSharp,
                             in.sub_x, in.sub_y, false, lw1, lh1,
                             in.ref[1].width, in.ref[1].height, in.frame_w,
                             in.frame_h, a1, in.scratch1, in.scratch_stride)) {
    return TemporalStatus::kInvalidArgument;
  }
  compound_average(in.scratch0, in.scratch_stride, in.scratch1,
                   in.scratch_stride, in.w, in.h, in.cwp_weight, in.dst,
                   in.dst_stride);
  return TemporalStatus::kOk;
}

}  // namespace av2enc::inter::temporal
