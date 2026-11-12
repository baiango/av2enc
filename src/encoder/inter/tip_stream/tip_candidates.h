// AV2 v1.0.0 TIP-derived MV candidates.
//
// Scope:
//   §7.12.2.15  derive ref mv candidate from tip mode process
//   §7.12.2.17  derive single ref mv candidate from tip mode process
//   §7.12.2.18  TIP add derived process
//
// These are the stack-side processes a decoder runs when it sees a neighbour
// coded as a TIP block (RefFrames[..][0] == TIP_FRAME) and when it expands the
// TIP SubMvs into a derived single-list candidate.  The existing M2 stack
// builder (`inter/m2/merge/merge.cpp`) drops TIP neighbours outright, so a
// stream that mixes TIP-AS-REF blocks with NEARMV/NEWMV blocks needs these.
//
// Conventions match inter/temporal:
//   * MVs are 1/8 luma samples, `mv[0] = row`, `mv[1] = column`.
//   * `StackMv::cwp` is CWP_EQUAL unless the caller seeded it otherwise.
//   * `prune_count` maps to PruneCount; `derived->prune` to DerivedPruneCount.

#ifndef AV2ENC_ENCODER_INTER_TIP_STREAM_TIP_CANDIDATES_H_
#define AV2ENC_ENCODER_INTER_TIP_STREAM_TIP_CANDIDATES_H_

#include "encoder/inter/temporal/temporal_types.h"

namespace av2enc::inter::tip {

using temporal::Mv;
using temporal::StackMv;
using temporal::TemporalStatus;
using temporal::TipCandInput;
using temporal::TipConfig;

// §7.12.2.18 output stack: DerivedStackMv / DerivedMvCount.
struct DerivedMvStack {
  Mv mv[temporal::kMaxDrStackSize];
  int count = 0;
  int prune = 0;  // DerivedPruneCount
};

// §7.12.2.15.  `cand` carries the neighbour location plus the motion field and
// the caller-owned block-grid state that get_tip_cand() needs.
TemporalStatus derive_ref_mv_candidate_from_tip_mode(const TipCandInput& cand,
                                                     const TipConfig& tip,
                                                     int weight, StackMv* stack,
                                                     int* num_mv_found,
                                                     int* prune_count);

// §7.12.2.17.  `cand_list` selects which list of the TIP candidate pair to
// insert.  Only RefStackMv[][0] is written, exactly as the spec does.
TemporalStatus derive_single_ref_mv_candidate_from_tip_mode(
    const TipCandInput& cand, const TipConfig& tip, int cand_list, int weight,
    StackMv* stack, int* num_mv_found, int* prune_count);

// §7.12.2.18.  `sub_mvs[list]` is SubMvs[mvRow][mvCol][list] for the TIP
// neighbour.  Adds the derived single-list MV to the derived stack.
TemporalStatus tip_add_derived(const Mv sub_mvs[2], const TipConfig& tip,
                               DerivedMvStack* derived);

}  // namespace av2enc::inter::tip

#endif  // AV2ENC_ENCODER_INTER_TIP_STREAM_TIP_CANDIDATES_H_
