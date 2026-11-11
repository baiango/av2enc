// Bridge between the spec-exact DRL reorder transform and the M2 merge stack.
//
// This header is the *only* place in the module that includes the M2 merge
// layer.  It does not modify src/encoder/m2/merge; callers that own that code
// switch their stack build to build_reordered_stack() and their index choice
// to signal_index_for_pre().
//
// Why a bridge is needed
// ----------------------
// m2::merge::build_mv_stack() implements DrlReorder by calling its own
// `sort_nearest()`, which is a full selection sort.  The normative process is a
// single move-to-front (§7.12.2.19), so using that path with reordering enabled
// is the historical desync.  build_reordered_stack() therefore always builds
// with kDisabled and applies drl::reorder_stack() itself.

#pragma once

#include "encoder/inter/drl_reorder/drl_reorder.h"
#include "encoder/m2/merge/merge.h"

namespace av2enc::inter::drl {

namespace merge = av2enc::m2::merge;

Mode mode_from_m2(merge::DrlReorder reorder);
merge::DrlReorder mode_to_m2(Mode mode);

// Copies the candidates and the numNearest snapshot.
void load_m2_stack(const merge::MvStack& in, Stack* out);
void store_m2_stack(const Stack& in, merge::MvStack* out);

// Applies the normative §7.12.2.19 transform to [0, num_nearest).  Returns 1
// when a swap happened.
int reorder_m2_stack(merge::MvStack* stack, Mode mode,
                     bool use_temporal_first);

// Spec-exact stack build for a reordering-enabled bitstream:
//   1. copy `params` and force merge::DrlReorder::kDisabled;
//   2. call merge::build_mv_stack();
//   3. apply drl::reorder_stack() with `mode` / `use_temporal_first`.
// On success the returned stack is exactly the decoder's RefStackMv order.
merge::Status build_reordered_stack(const merge::StackParams& params,
                                    Mode mode, bool use_temporal_first,
                                    merge::MvStack* stack);

// Encoder-side signalled index for a candidate identified by its pre-reorder
// position.  Returns -1 when the position is outside the stack.
int signal_index_for_pre(const merge::MvStack& stack, Mode mode,
                         bool use_temporal_first, int pre_index);

}  // namespace av2enc::inter::drl
