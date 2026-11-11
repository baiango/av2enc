#include "search.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace av2enc::m1::rdo_loop {
namespace {

using partition::BlockSize;
using partition::PartitionType;

constexpr double kInfiniteCost = std::numeric_limits<double>::infinity();

int search_order_index(PartitionType partition) noexcept {
  for (int i = 0; i < kSearchOrderCount; ++i) {
    if (kSearchOrder[i] == partition) {
      return i;
    }
  }
  return kSearchOrderCount;
}

bool partition_sane(PartitionType partition) noexcept {
  return search_order_index(partition) < kSearchOrderCount;
}

}  // namespace

bool search_options_valid(const SearchOptions& options) noexcept {
  return options.min_block_size_px >= partition::kMinBlockPixels &&
         options.max_depth >= 0 && options.max_leaf_probes_per_superblock >= 0 &&
         std::isfinite(options.candidate_sse_margin) &&
         options.candidate_sse_margin >= 0.0;
}

bool validate_search_options(const SearchOptions& options, std::string* error) {
  if (options.min_block_size_px < partition::kMinBlockPixels) {
    if (error != nullptr) {
      *error = "SearchOptions.min_block_size_px must be >= 4";
    }
    return false;
  }
  if (options.max_depth < 0) {
    if (error != nullptr) *error = "SearchOptions.max_depth must be >= 0";
    return false;
  }
  if (options.max_leaf_probes_per_superblock < 0) {
    if (error != nullptr) {
      *error = "SearchOptions.max_leaf_probes_per_superblock must be >= 0";
    }
    return false;
  }
  if (!std::isfinite(options.candidate_sse_margin) ||
      options.candidate_sse_margin < 0.0) {
    if (error != nullptr) {
      *error = "SearchOptions.candidate_sse_margin must be finite and >= 0";
    }
    return false;
  }
  return true;
}

double default_partition_signal_bits(const NodeInfo& info,
                                     PartitionType partition) noexcept {
  if (info.forced) {
    return 0.0;
  }
  switch (partition) {
    case PartitionType::kNone:
      // do_split = 0 (one binary symbol).
      return 1.0;
    case PartitionType::kHorz:
    case PartitionType::kVert:
      // do_split = 1 plus rect_type.
      return 2.0;
    case PartitionType::kSplit:
      // do_split = 1 plus the square-split flag.
      return 1.5;
  }
  return kInfiniteCost;
}

bool describe_node(const partition::FrameGeometry& geometry, int mi_row,
                   int mi_col, BlockSize size, int depth, bool has_chroma,
                   bool chroma_offset, NodeInfo* out, std::string* error) {
  if (out == nullptr) {
    if (error != nullptr) *error = "describe_node: null output";
    return false;
  }
  *out = NodeInfo{};
  if (!partition::is_valid_block_size(size)) {
    if (error != nullptr) *error = "describe_node: invalid block size";
    return false;
  }
  if (mi_row < 0 || mi_col < 0 || mi_row >= partition::mi_rows(geometry) ||
      mi_col >= partition::mi_cols(geometry)) {
    if (error != nullptr) *error = "describe_node: node origin outside the frame";
    return false;
  }
  out->mi_row = mi_row;
  out->mi_col = mi_col;
  out->size = size;
  out->depth = depth;
  out->has_chroma = has_chroma;
  out->chroma_offset = chroma_offset;
  out->boundary = partition::partition_implied(geometry, mi_row, mi_col, size);
  out->allowed = partition::init_allowed_partitions(
      geometry, mi_row, mi_col, size, has_chroma, chroma_offset);

  out->forced = false;
  out->forced_partition = PartitionType::kNone;
  if (out->boundary.implied &&
      out->allowed.is_allowed(out->boundary.implied_partition)) {
    out->forced = true;
    out->forced_partition = out->boundary.implied_partition;
  } else if (out->allowed.count == 1) {
    out->forced = true;
    for (int index = 0; index < partition::kPartitionTypeCount; ++index) {
      if (out->allowed.allowed[index]) {
        out->forced_partition = static_cast<PartitionType>(index);
        break;
      }
    }
  }
  return true;
}

bool policy_allows(const SearchOptions& options, int depth, BlockSize size,
                   PartitionType partition) noexcept {
  if (partition == PartitionType::kNone) {
    return true;
  }
  if (partition == PartitionType::kSplit && !options.allow_split) {
    return false;
  }
  if ((partition == PartitionType::kHorz ||
       partition == PartitionType::kVert) &&
      !options.allow_rect) {
    return false;
  }
  if (depth + 1 > options.max_depth) {
    return false;
  }
  const BlockSize subsize = partition::partition_subsize(partition, size);
  if (!partition::is_valid_block_size(subsize)) {
    return false;
  }
  const int floor_px =
      std::max(options.min_block_size_px, partition::kMinBlockPixels);
  return partition::block_width(subsize) >= floor_px &&
         partition::block_height(subsize) >= floor_px;
}

int enumerate_node_children(const partition::FrameGeometry& geometry,
                            const NodeInfo& info, PartitionType partition,
                            ChildRef slots[4], std::string* error) {
  if (slots == nullptr) {
    if (error != nullptr) *error = "enumerate_node_children: null slots";
    return 0;
  }
  if (partition == PartitionType::kNone) {
    return 0;
  }
  const partition_ext::PartitionType ext_partition =
      partition_ext::from_base(partition);
  if (!partition_ext::is_valid_partition(ext_partition)) {
    if (error != nullptr) *error = "enumerate_node_children: invalid partition";
    return 0;
  }
  partition_ext::ChildSlot ext_slots[4];
  partition_ext::ChromaBlock inherited;
  const int raw = partition_ext::enumerate_children(
      geometry, info.mi_row, info.mi_col, info.size, ext_partition,
      info.has_chroma, info.chroma_offset, inherited, ext_slots);
  if (raw <= 0) {
    return 0;
  }
  int count = 0;
  for (int i = 0; i < raw; ++i) {
    const partition_ext::ChildSlot& slot = ext_slots[i];
    if (!partition_ext::child_is_coded(geometry, slot)) {
      continue;
    }
    ChildRef& out = slots[count++];
    out.mi_row = slot.mi_row;
    out.mi_col = slot.mi_col;
    out.size = slot.size;
    out.has_chroma = slot.has_chroma;
    out.chroma_offset = slot.chroma_offset;
  }
  return count;
}

void NodeDecisionLoop::reset(const NodeInfo& info,
                             const SearchOptions& options) noexcept {
  info_ = info;
  options_ = options;
  decision_ = NodeDecision{};
  lambda_ = 0.0;
}

bool NodeDecisionLoop::add(const CandidateCost& candidate, std::string* error) {
  if (!candidate.valid || !partition_sane(candidate.partition)) {
    ++decision_.rejected_count;
    if (error != nullptr) *error = "NodeDecisionLoop::add: invalid candidate";
    return false;
  }
  if (!cost_terms_valid(candidate.cost.terms)) {
    ++decision_.rejected_count;
    if (error != nullptr) {
      *error = "NodeDecisionLoop::add: candidate terms are not finite";
    }
    return false;
  }
  if (decision_.candidate_count >= kMaxNodeCandidates) {
    ++decision_.rejected_count;
    if (error != nullptr) *error = "NodeDecisionLoop::add: too many candidates";
    return false;
  }

  NodeCost cost = make_node_cost(candidate.cost.terms, lambda_,
                                 options_.rate_aware, candidate.cost.leaves);
  if (!node_cost_valid(cost)) {
    ++decision_.rejected_count;
    if (error != nullptr) {
      *error = "NodeDecisionLoop::add: candidate cost is not finite";
    }
    return false;
  }

  const int slot = decision_.candidate_count;
  NodeCandidate& stored = decision_.candidates[slot];
  stored = NodeCandidate{};
  stored.partition = candidate.partition;
  stored.cost = cost;
  stored.child_count = std::max(0, std::min(candidate.child_count,
                                            kMaxNodeCandidates));
  for (int i = 0; i < stored.child_count; ++i) {
    stored.terms[i] = candidate.child_terms[i];
  }
  stored.valid = true;
  ++decision_.candidate_count;

  if (!decision_.found ||
      node_cost_better(cost, candidate.partition, slot, decision_.cost,
                       decision_.partition, decision_.candidate_index)) {
    decision_.found = true;
    decision_.partition = candidate.partition;
    decision_.cost = cost;
    decision_.candidate_index = slot;
  }
  return true;
}

bool NodeDecisionLoop::add_terms(PartitionType partition,
                                 const CostTerms& terms, int leaves,
                                 std::string* error) {
  CandidateCost candidate;
  candidate.partition = partition;
  candidate.cost = make_node_cost(terms, lambda_, options_.rate_aware, leaves);
  candidate.child_count = 0;
  candidate.valid = true;
  return add(candidate, error);
}

bool NodeDecisionLoop::add_partition(PartitionType partition,
                                     std::span<const NodeCost> children,
                                     std::string* error) {
  if (children.empty() || children.size() > kMaxNodeCandidates) {
    ++decision_.rejected_count;
    if (error != nullptr) {
      *error = "NodeDecisionLoop::add_partition: invalid child count";
    }
    return false;
  }
  CostTerms terms;
  int leaves = 0;
  for (const NodeCost& child : children) {
    if (!node_cost_valid(child)) {
      ++decision_.rejected_count;
      if (error != nullptr) {
        *error = "NodeDecisionLoop::add_partition: invalid child cost";
      }
      return false;
    }
    terms.distortion_sse += child.terms.distortion_sse;
    terms.mode_bits += child.terms.mode_bits;
    terms.coefficient_bits += child.terms.coefficient_bits;
    terms.tx_type_bits += child.terms.tx_type_bits;
    terms.signal_bits += child.terms.signal_bits;
    leaves += child.leaves;
  }
  double signal = default_partition_signal_bits(info_, partition);
  if (!std::isfinite(signal)) {
    ++decision_.rejected_count;
    if (error != nullptr) {
      *error = "NodeDecisionLoop::add_partition: no signal cost";
    }
    return false;
  }
  // A spec-forced partition codes no symbol.  NONE at an unforced node codes
  // do_split = 0, which is already in `signal`.
  terms.signal_bits += signal;

  CandidateCost candidate;
  candidate.partition = partition;
  candidate.cost = make_node_cost(terms, lambda_, options_.rate_aware, leaves);
  candidate.child_count = static_cast<int>(children.size());
  for (std::size_t i = 0; i < children.size(); ++i) {
    candidate.child_terms[i] = children[i].terms;
  }
  candidate.valid = true;
  return add(candidate, error);
}

bool NodeDecisionLoop::cannot_beat_best(const NodeCost& accumulated) const
    noexcept {
  if (!options_.exact_branch_and_bound || !decision_.found) {
    return false;
  }
  if (!std::isfinite(accumulated.cost)) {
    return true;
  }
  // Remaining children have non-negative distortion and rate, so equality can
  // still win on the distortion/rate tie-break; only a strict excess prunes.
  return accumulated.cost > decision_.cost.cost;
}

bool record_node_trace(const NodeInfo& info, const NodeDecision& decision,
                       CostTrace& trace) noexcept {
  if (!decision.found) {
    return false;
  }
  NodeTraceEntry entry;
  entry.mi_row = info.mi_row;
  entry.mi_col = info.mi_col;
  entry.size = static_cast<std::uint16_t>(info.size);
  entry.depth = static_cast<std::int8_t>(info.depth);
  entry.partition = static_cast<std::uint8_t>(decision.partition);
  entry.forced = info.forced ? 1 : 0;
  entry.rate_aware = decision.cost.rate_aware ? 1 : 0;
  const int winner = decision.candidate_index;
  entry.child_count =
      static_cast<std::uint8_t>(decision.candidates[winner].child_count);
  entry.lambda = decision.cost.lambda;
  entry.terms = decision.cost.terms;
  entry.cost = decision.cost.cost;
  entry.leaves = decision.cost.leaves;
  return trace.record(entry);
}

int prune_candidates(std::span<const CandidateScore> candidates,
                     std::span<int> keep, std::span<int> scratch,
                     const PruneConfig& config) noexcept {
  if (keep.size() < candidates.size()) {
    return 0;
  }
  if (candidates.empty()) {
    return 0;
  }
  const int count = static_cast<int>(candidates.size());
  int max_keep = config.max_keep;
  if (max_keep <= 0 || max_keep > count) {
    max_keep = count;
  }
  if (scratch.size() < static_cast<std::size_t>(max_keep)) {
    return 0;
  }

  // Deterministic top-K by (SSE, index) using insertion into a fixed array.
  int selected = 0;
  for (int i = 0; i < count; ++i) {
    const CandidateScore& candidate = candidates[static_cast<std::size_t>(i)];
    if (!std::isfinite(candidate.sse) || candidate.sse < 0.0) {
      continue;
    }
    int position = selected;
    while (position > 0) {
      const CandidateScore& previous =
          candidates[static_cast<std::size_t>(
              scratch[static_cast<std::size_t>(position - 1)])];
      const bool candidate_better =
          candidate.sse < previous.sse ||
          (candidate.sse == previous.sse && candidate.index < previous.index);
      if (!candidate_better) {
        break;
      }
      scratch[static_cast<std::size_t>(position)] =
          scratch[static_cast<std::size_t>(position - 1)];
      --position;
    }
    if (selected < max_keep) {
      scratch[static_cast<std::size_t>(position)] = i;
      ++selected;
    } else if (position < max_keep) {
      scratch[static_cast<std::size_t>(position)] = i;
    }
  }
  if (selected == 0) {
    return 0;
  }
  // `scratch` holds *candidate array positions*, not the caller indices.  The
  // margin is applied against the best selected SSE.
  const int best_pos = scratch[0];
  const double best_sse =
      candidates[static_cast<std::size_t>(best_pos)].sse;
  const double threshold =
      config.sse_margin > 0.0 ? best_sse * config.sse_margin
                              : std::numeric_limits<double>::infinity();

  int kept = 0;
  for (int i = 0; i < count; ++i) {
    const CandidateScore& candidate = candidates[static_cast<std::size_t>(i)];
    if (!std::isfinite(candidate.sse) || candidate.sse < 0.0) {
      continue;
    }
    bool in_top = false;
    for (int s = 0; s < selected; ++s) {
      if (scratch[static_cast<std::size_t>(s)] == i) {
        in_top = true;
        break;
      }
    }
    const bool keep_forced =
        config.keep_index >= 0 && candidate.index == config.keep_index;
    if (keep_forced || (in_top && candidate.sse <= threshold)) {
      keep[static_cast<std::size_t>(kept++)] = candidate.index;
    }
  }
  return kept;
}

}  // namespace av2enc::m1::rdo_loop
