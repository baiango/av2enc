// §7.12.2.7/.8 temporal MV-stack adapter for the video merge layer.
//
// `av2enc::video::merge` deliberately has no link dependency on the temporal
// component (`src/codec/inter/temporal/`): its stack builder calls a
// caller-supplied function pointer at the normative step.  This adapter is the
// only bridge; `video_frame.cpp` (which also owns the §7.9 estimation) attaches
// it to every non-key frame when `enable_ref_frame_mvs` is on.
//
// Keeping the adapter in a header included by video_frame.cpp means
// `video_syntax.cpp` can rebuild the identical stack by copying the function
// pointer through `syntax::FramePlan` without pulling motion_field.cpp into
// the standalone gate builds.
#pragma once

#include "codec/inter/merge/merge.h"
#include "codec/inter/temporal/motion_field.h"
#include "codec/inter/temporal/ref_mv_bank.h"

namespace av2enc::video {

// Converts the merge stack to the temporal component's StackMv view, runs
// §7.12.2.7 temporal_scan() (which internally calls §7.12.2.8), and copies the
// candidates/weights back.  PruneCount is the builder's shared budget.
inline void video_temporal_scan(const merge::StackParams& p,
                                merge::MvStack* stack, int* prune_count) {
  if (stack == nullptr || prune_count == nullptr) return;
  inter::temporal::StackMv view[merge::kMaxRefMvStackSize];
  const int before = stack->count;
  for (int i = 0; i < before; ++i) {
    view[i].mv[0] = inter::temporal::Mv{stack->cand[i].mv[0].y,
                                        stack->cand[i].mv[0].x};
    view[i].mv[1] = inter::temporal::Mv{stack->cand[i].mv[1].y,
                                        stack->cand[i].mv[1].x};
    view[i].weight = stack->cand[i].weight;
    view[i].cwp = inter::temporal::kCwpEqual;
  }
  inter::temporal::TemporalScanInput in;
  in.mi_row = p.mi_row;
  in.mi_col = p.mi_col;
  in.mi_cols = p.mi_cols;
  in.mi_rows = p.mi_rows;
  in.tile_mi_col_start = p.tile_mi_col_start;
  in.tile_mi_col_end = p.tile_mi_col_end;
  in.tile_mi_row_start = p.tile_mi_row_start;
  in.tile_mi_row_end = p.tile_mi_row_end;
  in.bw4 = p.bw4;
  in.bh4 = p.bh4;
  in.ref0 = p.ref_frame[0];
  in.ref1 = p.ref_frame[1];
  in.order_hint = p.order_hint;
  in.is_compound = p.is_compound;
  in.use_traj = false;
  in.field = p.motion_field;
  in.num_refs = p.num_refs;
  if (p.order_hints != nullptr) {
    if (in.ref0 >= 0 && in.ref0 < p.num_refs) {
      in.ref0_order_hint = p.order_hints[in.ref0];
    }
    if (in.ref1 >= 0 && in.ref1 < p.num_refs) {
      in.ref1_order_hint = p.order_hints[in.ref1];
    }
  }
  int count = before;
  int prune = *prune_count;
  if (inter::temporal::temporal_scan(in, view, &count, &prune) !=
      inter::temporal::TemporalStatus::kOk) {
    return;
  }
  *prune_count = prune;
  for (int i = 0; i < count; ++i) {
    merge::StackCandidate& c = stack->cand[i];
    c.mv[0] = inter::mc::Mv{view[i].mv[0].row, view[i].mv[0].col};
    c.weight = view[i].weight;
    if (i < before) continue;  // existing candidate: weight only
    c.mv[1] = p.is_compound
                  ? inter::mc::Mv{view[i].mv[1].row, view[i].mv[1].col}
                  : inter::mc::Mv{0, 0};
    c.row_offset = 0;
    c.col_offset = 0;
    c.cwp = inter::temporal::kCwpEqual;
  }
  stack->count = count;
}

// Attaches the adapter to a stack-params object.  Only callers that link the
// temporal component (video_frame.cpp) may do this.
inline void attach_video_temporal_scan(merge::StackParams* params) {
  if (params != nullptr && params->use_ref_frame_mvs) {
    params->temporal_scan = &video_temporal_scan;
  }
}

// §7.12.2.21 Fill mvp from ref mv bank adapter.
inline void video_refmvbank_scan(const merge::StackParams& p,
                                 merge::MvStack* stack, int* prune_count) {
  if (stack == nullptr || prune_count == nullptr || p.refmv_bank == nullptr) {
    return;
  }
  inter::temporal::StackMv view[merge::kMaxRefMvStackSize];
  const int before = stack->count;
  for (int i = 0; i < before; ++i) {
    view[i].mv[0] = inter::temporal::Mv{stack->cand[i].mv[0].y,
                                        stack->cand[i].mv[0].x};
    view[i].mv[1] = inter::temporal::Mv{stack->cand[i].mv[1].y,
                                        stack->cand[i].mv[1].x};
    view[i].cwp = stack->cand[i].cwp;
  }
  int num_found = stack->count;
  const int ref_frames[2] = {
      static_cast<int>(p.ref_frame[0]),
      p.is_compound ? static_cast<int>(p.ref_frame[1])
                    : static_cast<int>(merge::kRefNone)};
  inter::temporal::fill_mvp_from_ref_mv_bank(
      *p.refmv_bank, ref_frames, p.is_compound, p.use_intrabc,
      p.max_drl_bits_minus_1, p.mi_row, p.mi_col, p.mi_cols, p.mi_rows,
      p.bw4 * 4, p.bh4 * 4, merge::kMaxRefMvStackSize, view, &num_found,
      prune_count);
  for (int i = before; i < num_found; ++i) {
    merge::StackCandidate& c = stack->cand[i];
    c.mv[0] = inter::mc::Mv{view[i].mv[0].row, view[i].mv[0].col};
    c.mv[1] = inter::mc::Mv{view[i].mv[1].row, view[i].mv[1].col};
    c.weight = 1;
    c.row_offset = 0;
    c.col_offset = 0;
    c.cwp = static_cast<std::int8_t>(view[i].cwp);
    c.from_refmvbank = true;
    ++stack->count;
  }
  if (p.refmvbank_added != nullptr) {
    *p.refmvbank_added += stack->count - before;
  }
}

inline void attach_video_refmvbank(merge::StackParams* params) {
  if (params != nullptr && params->refmv_bank != nullptr) {
    params->refmv_bank_scan = &video_refmvbank_scan;
  }
}

}  // namespace av2enc::video
