#include "codec/inter/drl_reorder/video_stack_adapter.h"

#include <algorithm>

namespace av2enc::inter::drl {

Mode mode_from_m2(merge::DrlReorder reorder) {
  switch (reorder) {
    case merge::DrlReorder::kDisabled:
      return Mode::kDisabled;
    case merge::DrlReorder::kConstraint:
      return Mode::kConstraint;
    case merge::DrlReorder::kAlways:
      return Mode::kAlways;
  }
  return Mode::kDisabled;
}

merge::DrlReorder mode_to_m2(Mode mode) {
  switch (mode) {
    case Mode::kDisabled:
      return merge::DrlReorder::kDisabled;
    case Mode::kConstraint:
      return merge::DrlReorder::kConstraint;
    case Mode::kAlways:
      return merge::DrlReorder::kAlways;
  }
  return merge::DrlReorder::kDisabled;
}

void load_video_stack(const merge::MvStack& in, Stack* out) {
  if (out == nullptr) {
    return;
  }
  *out = Stack{};
  out->count = std::min(in.count, kMaxRefMvStackSize);
  out->num_nearest = std::min(in.num_nearest, out->count);
  for (int i = 0; i < out->count; ++i) {
    const merge::StackCandidate& src = in.cand[i];
    StackCandidate& dst = out->cand[i];
    dst.mv[0][0] = src.mv[0].y;
    dst.mv[0][1] = src.mv[0].x;
    dst.mv[1][0] = src.mv[1].y;
    dst.mv[1][1] = src.mv[1].x;
    dst.weight = src.weight;
    dst.row_offset = src.row_offset;
    dst.col_offset = src.col_offset;
    dst.cwp = src.cwp;
  }
}

void store_video_stack(const Stack& in, merge::MvStack* out) {
  if (out == nullptr) {
    return;
  }
  const int count = std::min(in.count, kMaxRefMvStackSize);
  out->count = count;
  out->num_nearest = std::min(in.num_nearest, count);
  for (int i = 0; i < count; ++i) {
    const StackCandidate& src = in.cand[i];
    merge::StackCandidate& dst = out->cand[i];
    dst.mv[0] = inter::mc::Mv{src.mv[0][0], src.mv[0][1]};
    dst.mv[1] = inter::mc::Mv{src.mv[1][0], src.mv[1][1]};
    dst.weight = src.weight;
    dst.row_offset = src.row_offset;
    dst.col_offset = src.col_offset;
    dst.cwp = static_cast<std::int8_t>(src.cwp);
  }
}

int reorder_video_stack(merge::MvStack* stack, Mode mode,
                     bool use_temporal_first) {
  if (stack == nullptr) {
    return 0;
  }
  Stack view;
  load_video_stack(*stack, &view);
  const int swapped = reorder_stack(&view, mode, use_temporal_first);
  if (swapped == 0) {
    return 0;
  }
  store_video_stack(view, stack);
  return 1;
}

merge::Status build_reordered_stack(const merge::StackParams& params,
                                    Mode mode, bool use_temporal_first,
                                    merge::MvStack* stack) {
  if (stack == nullptr) {
    return merge::Status::kInvalidArgument;
  }
  merge::StackParams build = params;
  // Never let the M2 layer apply its selection sort; only this module's
  // normative transform may reorder RefStackMv.
  build.reorder = merge::DrlReorder::kDisabled;
  const merge::Status status = merge::build_mv_stack(build, stack);
  if (status != merge::Status::kOk) {
    return status;
  }
  reorder_video_stack(stack, mode, use_temporal_first);
  return merge::Status::kOk;
}

int signal_index_for_pre(const merge::MvStack& stack, Mode mode,
                         bool use_temporal_first, int pre_index) {
  Stack view;
  load_video_stack(stack, &view);
  return drl::signal_index_for_pre(view.cand, view.count, view.num_nearest,
                                   mode, use_temporal_first, pre_index);
}

}  // namespace av2enc::inter::drl
