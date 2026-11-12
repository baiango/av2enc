#include "codec/inter/multiref/mvstack_multi.h"

#include <algorithm>
#include <limits>
#include <set>
#include <utility>

#include "codec/inter/multiref/compound_predict.h"

namespace av2enc::video::multiref {
namespace {

using MvKey = std::pair<std::int32_t, std::int32_t>;

}  // namespace

bool build_pair_stacks(const merge::StackParams& params, std::int8_t ref0,
                       std::int8_t ref1, RefStackPair& out,
                       std::string& error) {
  error.clear();
  out = RefStackPair{};
  const std::int8_t refs[2] = {ref0, ref1};
  for (int list = 0; list < 2; ++list) {
    merge::StackParams local = params;
    local.ref_frame[0] = refs[list];
    local.ref_frame[1] = merge::kRefNone;
    local.is_compound = false;
    const merge::Status status = merge::build_mv_stack(local, &out.list[list]);
    out.ok[list] = status == merge::Status::kOk;
    if (!out.ok[list]) {
      error = std::string("build_mv_stack failed for list ") +
              std::to_string(list) + ": " + merge::status_string(status);
      return false;
    }
    out.num_nearest[list] = out.list[list].num_nearest;
  }
  merge::StackParams compound = params;
  compound.ref_frame[0] = ref0;
  compound.ref_frame[1] = ref1;
  compound.is_compound = true;
  const merge::Status status = merge::build_mv_stack(compound, &out.compound);
  out.compound_ok = status == merge::Status::kOk;
  if (!out.compound_ok) {
    error = std::string("compound build_mv_stack failed: ") +
            merge::status_string(status);
    return false;
  }
  return true;
}

CompoundMergeSet build_compound_merge_candidates(const RefStackPair& stacks) {
  CompoundMergeSet set;
  if (!stacks.ok[0] || !stacks.ok[1]) return set;
  std::set<std::pair<MvKey, MvKey>> seen;
  const auto push = [&](int i0, int i1, mc::Mv mv0, mc::Mv mv1,
                        int weight0, int weight1) {
    CompoundMergeCandidate candidate;
    candidate.mv[0] = mv0;
    candidate.mv[1] = mv1;
    candidate.drl_index[0] = i0;
    candidate.drl_index[1] = i1;
    candidate.weight = weight0 + weight1;
    candidate.single_index = i0 == i1;
    const std::pair<MvKey, MvKey> key{{mv0.y, mv0.x}, {mv1.y, mv1.x}};
    if (!seen.insert(key).second) return;
    set.candidates.push_back(candidate);
  };

  if (stacks.compound_ok) {
    for (int i = 0; i < stacks.compound.count; ++i) {
      const merge::StackCandidate& candidate = stacks.compound.cand[i];
      push(i, i, candidate.mv[0], candidate.mv[1], candidate.weight,
           candidate.weight);
    }
  }
  for (int i = 0; i < stacks.list[0].count; ++i) {
    for (int j = 0; j < stacks.list[1].count; ++j) {
      if (i == j) continue;
      const merge::StackCandidate& a = stacks.list[0].cand[i];
      const merge::StackCandidate& b = stacks.list[1].cand[j];
      push(i, j, a.mv[0], b.mv[0], a.weight, b.weight);
    }
  }
  return set;
}

bool select_compound_merge_candidate(const CompoundMergeSet& set,
                                     const mc::RefFrame refs[2],
                                     const mc::BlockRect& luma_rect,
                                     int bsize, int mi_cols, int mi_rows,
                                     const std::uint8_t* src, int src_stride,
                                     std::uint8_t* scratch,
                                     int scratch_stride, MergeScore& out,
                                     std::string& error) {
  error.clear();
  out = MergeScore{};
  if (refs == nullptr || src == nullptr || scratch == nullptr ||
      src_stride <= 0 || scratch_stride <= 0) {
    error = "merge selection input missing";
    return false;
  }
  std::int64_t best = std::numeric_limits<std::int64_t>::max();
  bool found = false;
  for (std::size_t i = 0; i < set.candidates.size(); ++i) {
    CompoundPrediction prediction;
    if (!build_compound_prediction(refs[0], refs[1], luma_rect, bsize,
                                   set.candidates[i].mv[0],
                                   set.candidates[i].mv[1],
                                   mc::InterpFilter::kEighttap, mi_cols,
                                   mi_rows, prediction, error)) {
      return false;
    }
    CompoundModeParams average;
    average.type = adv::CompoundType::kAverage;
    average.cwp_weight = kCwpEqual;
    const std::int64_t sse =
        compound_luma_sse(prediction, average, src, src_stride, scratch,
                          scratch_stride);
    if (sse == std::numeric_limits<std::int64_t>::max()) {
      error = "merge candidate prediction failed";
      return false;
    }
    if (!found || sse < best) {
      best = sse;
      out.candidate_index = static_cast<int>(i);
      out.mv[0] = set.candidates[i].mv[0];
      out.mv[1] = set.candidates[i].mv[1];
      out.drl_index[0] = set.candidates[i].drl_index[0];
      out.drl_index[1] = set.candidates[i].drl_index[1];
      out.sse = sse;
      found = true;
    }
  }
  if (!found) {
    error = "no compound merge candidate";
    return false;
  }
  return true;
}

bool select_best_ref_merge(const RefStackPair& stacks,
                           const mc::RefFrame refs[2],
                           const mc::BlockRect& luma_rect,
                           mc::InterpFilter filter, const std::uint8_t* src,
                           int src_stride, std::uint8_t* scratch,
                           int scratch_stride, RefMergeScore& out,
                           std::string& error) {
  error.clear();
  out = RefMergeScore{};
  if (refs == nullptr || src == nullptr || scratch == nullptr ||
      src_stride <= 0 || scratch_stride <= 0) {
    error = "single-ref merge input missing";
    return false;
  }
  std::int64_t best = std::numeric_limits<std::int64_t>::max();
  bool found = false;
  for (int list = 0; list < 2; ++list) {
    if (!stacks.ok[list]) continue;
    const merge::MvStack& stack = stacks.list[list];
    for (int drl = 0; drl < stack.count; ++drl) {
      // NEARMV/merge uses RefStackMv[drl][0] directly.  The references are
      // unscaled and share the current frame dimensions, so no reference
      // scaling applies.
      const mc::Mv mv = stack.cand[drl].mv[0];
      const mc::McStatus mc_status =
          mc::predict_plane(refs[list], 0, luma_rect, mv, filter, scratch,
                            scratch_stride);
      if (mc_status != mc::McStatus::kOk) {
        error = std::string("predict_plane failed: ") +
                mc::mc_status_string(mc_status);
        return false;
      }
      const std::int64_t sad =
          mc::sad_block(scratch, scratch_stride, src, src_stride,
                        luma_rect.w, luma_rect.h);
      if (!found || sad < best) {
        best = sad;
        out.ref = list;
        out.drl_index = drl;
        out.sad = sad;
        found = true;
      }
    }
  }
  if (!found) {
    error = "no reference merge candidate";
    return false;
  }
  return true;
}

}  // namespace av2enc::video::multiref
