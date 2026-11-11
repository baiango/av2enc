// Multi-reference MV stack / merge candidates.
//
// AV2 builds the spatial MV stack once per reference list (§7.12.2).  For a
// compound block with two distinct references, the modes that use two DRL
// indices (NEAR_NEARMV, NEAR_NEWMV, NEW_NEARMV, JOINT_NEWMV) derive list-0
// candidates from RefStack0Mvs and list-1 candidates from RefStack1Mvs, exactly
// as §5.20.7.6's has_second_drl() path does.  skip_mode instead uses the
// two-list stack entry at a single index.
//
// This module wraps the M2-owned spec-exact stack in src/encoder/m2/merge/:
//   * one single-reference stack per list (has_second_drl);
//   * one compound two-list stack (skip_mode);
//   * a deterministic compound merge candidate list.
//
// The older shared `inter::mc::build_mv_stack` is deliberately not used here:
// it inserts §7.12.2.20 MVP-combination candidates without the block-size gate.
// `merge::build_mv_stack` applies the gate, so every DRL index this module
// returns also exists on the decoder stack.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "encoder/inter/mc/mc.h"
#include "encoder/m2/merge/merge.h"

namespace av2enc::m2::multiref {

namespace mc = av2enc::inter::mc;
namespace merge = av2enc::m2::merge;

struct RefStackPair {
  // Single-reference stacks (one per RefFrame index).
  merge::MvStack list[2];
  // Two-list stack used by skip_mode.
  merge::MvStack compound;
  int num_nearest[2] = {0, 0};
  bool ok[2] = {false, false};
  bool compound_ok = false;
};

// Builds the two single-reference stacks and the two-list stack.  `params`
// supplies the MI grid, block geometry, global MVs, DRL reorder and (when
// available) FrameDistance[]/NumRefs for reference projection.  `ref0`/`ref1`
// are RefFrame indices 0..NumTotalRefs-1.
bool build_pair_stacks(const merge::StackParams& params, std::int8_t ref0,
                       std::int8_t ref1, RefStackPair& out,
                       std::string& error);

struct CompoundMergeCandidate {
  mc::Mv mv[2] = {};
  int drl_index[2] = {0, 0};
  int weight = 0;
  // True for the single-index form (skip_mode / RefStackMv[i]).
  bool single_index = false;
};

struct CompoundMergeSet {
  std::vector<CompoundMergeCandidate> candidates;
};

// Fixed order:
//   1. single-index candidates i = 0..compound.count-1 (skip_mode form);
//   2. two-index pairs (i, j), i != j, lexicographic over the per-list counts.
// Duplicate MV pairs are dropped, keeping the first occurrence.
CompoundMergeSet build_compound_merge_candidates(const RefStackPair& stacks);

struct MergeScore {
  int candidate_index = -1;
  mc::Mv mv[2] = {};
  int drl_index[2] = {0, 0};
  std::int64_t sse = 0;
};

// Predicts each candidate's luma block (average compound of the two
// references at the candidate MVs) and returns the lowest SSE.  Ties keep the
// earlier candidate.
bool select_compound_merge_candidate(const CompoundMergeSet& set,
                                     const mc::RefFrame refs[2],
                                     const mc::BlockRect& luma_rect,
                                     int bsize, int mi_cols, int mi_rows,
                                     const std::uint8_t* src, int src_stride,
                                     std::uint8_t* scratch,
                                     int scratch_stride, MergeScore& out,
                                     std::string& error);

// Single-reference merge selection: lowest SAD over both lists' stack
// candidates (NEARMV / merge form).
struct RefMergeScore {
  int ref = -1;
  int drl_index = -1;
  std::int64_t sad = 0;
};
bool select_best_ref_merge(const RefStackPair& stacks,
                           const mc::RefFrame refs[2],
                           const mc::BlockRect& luma_rect,
                           mc::InterpFilter filter, const std::uint8_t* src,
                           int src_stride, std::uint8_t* scratch,
                           int scratch_stride, RefMergeScore& out,
                           std::string& error);

}  // namespace av2enc::m2::multiref
