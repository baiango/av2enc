// AV2 v1.0.0 TIP-derived MV candidates (§7.12.2.15/.17/.18).

#include "encoder/inter/tip_stream/tip_candidates.h"

#include "encoder/inter/temporal/motion_field.h"
#include "encoder/inter/temporal/temporal_math.h"

namespace av2enc::inter::tip {
namespace {

bool check_stack(const StackMv* stack, const int* num, const int* prune) {
  return stack != nullptr && num != nullptr && prune != nullptr && *num >= 0 &&
         *num <= temporal::kMaxRefMvStackSize && *prune >= 0;
}

}  // namespace

TemporalStatus derive_ref_mv_candidate_from_tip_mode(const TipCandInput& cand,
                                                     const TipConfig& tip,
                                                     int weight, StackMv* stack,
                                                     int* num_mv_found,
                                                     int* prune_count) {
  if (!check_stack(stack, num_mv_found, prune_count) || weight <= 0) {
    return TemporalStatus::kInvalidArgument;
  }
  const temporal::MvPair cand_mvs = temporal::get_tip_cand(cand, tip);
  bool found = false;
  if (*prune_count < temporal::kMaxPrNum) {
    for (int idx = 0; idx < *num_mv_found; ++idx) {
      ++(*prune_count);
      if (cand_mvs.mv[0] == stack[idx].mv[0] &&
          cand_mvs.mv[1] == stack[idx].mv[1]) {
        stack[idx].weight += weight;
        found = true;
        break;
      }
    }
  }
  if (!found && *num_mv_found < temporal::kMaxRefMvStackSize) {
    const int dst = *num_mv_found;
    stack[dst].mv[0] = cand_mvs.mv[0];
    stack[dst].mv[1] = cand_mvs.mv[1];
    stack[dst].cwp = temporal::kCwpEqual;
    stack[dst].weight = weight;
    ++(*num_mv_found);
  }
  return TemporalStatus::kOk;
}

TemporalStatus derive_single_ref_mv_candidate_from_tip_mode(
    const TipCandInput& cand, const TipConfig& tip, int cand_list, int weight,
    StackMv* stack, int* num_mv_found, int* prune_count) {
  if (!check_stack(stack, num_mv_found, prune_count) || weight <= 0 ||
      cand_list < 0 || cand_list > 1) {
    return TemporalStatus::kInvalidArgument;
  }
  const temporal::MvPair cand_mvs = temporal::get_tip_cand(cand, tip);
  bool found = false;
  if (*prune_count < temporal::kMaxPrNum) {
    for (int idx = 0; idx < *num_mv_found; ++idx) {
      ++(*prune_count);
      if (cand_mvs.mv[cand_list] == stack[idx].mv[0]) {
        stack[idx].weight += weight;
        found = true;
        break;
      }
    }
  }
  if (!found && *num_mv_found < temporal::kMaxRefMvStackSize) {
    const int dst = *num_mv_found;
    stack[dst].mv[0] = cand_mvs.mv[cand_list];
    stack[dst].weight = weight;
    stack[dst].cwp = temporal::kCwpEqual;
    ++(*num_mv_found);
  }
  return TemporalStatus::kOk;
}

TemporalStatus tip_add_derived(const Mv sub_mvs[2], const TipConfig& tip,
                               DerivedMvStack* derived) {
  if (sub_mvs == nullptr || derived == nullptr || derived->count < 0 ||
      derived->count > temporal::kMaxDrStackSize || derived->prune < 0) {
    return TemporalStatus::kInvalidArgument;
  }
  const Mv linear{sub_mvs[0].row - sub_mvs[1].row,
                  sub_mvs[0].col - sub_mvs[1].col};
  const temporal::TipOffsets offsets = temporal::get_tip_offsets(tip);
  const Mv proj = temporal::get_mv_projection(linear, offsets.past_offset,
                                              offsets.ref_offset);
  Mv out;
  out.row = temporal::clip3(temporal::kMvLow + 1, temporal::kMvUpp - 1,
                            sub_mvs[0].row - proj.row);
  out.col = temporal::clip3(temporal::kMvLow + 1, temporal::kMvUpp - 1,
                            sub_mvs[0].col - proj.col);
  if (derived->prune < temporal::kMaxDrPrNum) {
    for (int i = 0; i < derived->count; ++i) {
      ++derived->prune;
      if (derived->mv[i] == out) {
        return TemporalStatus::kOk;
      }
    }
  }
  if (derived->count < temporal::kMaxDrStackSize) {
    derived->mv[derived->count] = out;
    ++derived->count;
  }
  return TemporalStatus::kOk;
}

}  // namespace av2enc::inter::tip
