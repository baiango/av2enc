// AV2 extended-partition geometry, allowed sets, contexts and decision solver.
// See partition_ext.h for the clause map and scope.

#include "partition_ext.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

namespace av2enc::partition_ext {
namespace {

constexpr int kSentinelLog2 = 6;  // BLOCK_256X256: never "smaller than current"

bool geometry_ready(const FrameGeometry& geometry) {
  return geometry.width > 0 && geometry.height > 0 &&
         (geometry.sb_size == 64 || geometry.sb_size == 128) &&
         geometry.max_pb_aspect_ratio >= 1;
}

int log2_or_sentinel(const MiSizeGrid& grid, int mi_row, int mi_col,
                     bool width) {
  const BlockSize size = grid.at(mi_row, mi_col);
  if (!partition::is_valid_block_size(size)) {
    return kSentinelLog2;
  }
  return width ? partition::mi_width_log2(size)
               : partition::mi_height_log2(size);
}

bool cost_is_sane(double value) {
  return !std::isnan(value) && value >= 0.0;
}

bool all_costs_sane(const NodeCosts& costs) {
  const double values[] = {
      costs.leaf,          costs.none_signal,  costs.split_signal,
      costs.horz_signal,   costs.vert_signal,  costs.horz3_signal,
      costs.vert3_signal,  costs.horz4a_signal, costs.horz4b_signal,
      costs.vert4a_signal, costs.vert4b_signal,
  };
  for (double value : values) {
    if (!cost_is_sane(value)) return false;
  }
  return true;
}

PartitionType base_to_ext(partition::PartitionType partition) {
  switch (partition) {
    case partition::PartitionType::kNone:
      return PartitionType::kNone;
    case partition::PartitionType::kSplit:
      return PartitionType::kSplit;
    case partition::PartitionType::kHorz:
      return PartitionType::kHorz;
    case partition::PartitionType::kVert:
      return PartitionType::kVert;
  }
  return PartitionType::kNone;
}

bool is_base_partition(PartitionType partition) {
  return partition == PartitionType::kNone ||
         partition == PartitionType::kHorz ||
         partition == PartitionType::kVert ||
         partition == PartitionType::kSplit;
}

bool fail(std::string& error, std::string message) {
  error = std::move(message);
  return false;
}

// The extended-row `is_partition_allowed` body (§5.20.3.2).  Base partitions
// are delegated to partition::is_partition_allowed().
bool is_ext_partition_allowed(const FrameGeometry& geometry, int mi_row,
                              int mi_col, BlockSize size,
                              PartitionType partition, bool has_chroma,
                              bool chroma_offset, const Tools& tools) {
  if (!geometry_ready(geometry) ||
      !partition::is_valid_block_size(size) || mi_row < 0 || mi_col < 0) {
    return false;
  }
  const BlockSize subsize = partition_subsize(partition, size);
  if (!partition::is_valid_block_size(subsize)) {
    return false;
  }

  // rect_type_implied_by_bsize(): a HORZ_* candidate is impossible when the
  // block can only be cut vertically, and vice versa.
  const partition::RectTypeImplied rect =
      partition::rect_type_implied_by_bsize(size);
  if (rect.implied) {
    if (rect.partition == partition::PartitionType::kVert &&
        is_horizontal(partition)) {
      return false;
    }
    if (rect.partition == partition::PartitionType::kHorz &&
        is_vertical(partition)) {
      return false;
    }
  }

  // Aspect-ratio guard, evaluated on the *subsize* (the 1/4 or 1/8 block).
  const int bw = partition::block_width(subsize);
  const int bh = partition::block_height(subsize);
  const int aspect = geometry.max_pb_aspect_ratio;
  if (bw > bh * aspect || bh > bw * aspect) {
    if (bw >= bh * 8 || bh >= bw * 8) {
      return false;
    }
  }

  bool local_offset = chroma_offset;
  if (has_chroma && !local_offset) {
    local_offset = chroma_offset_for_partition(geometry, partition, size);
  }
  if (has_chroma && !local_offset) {
    if (partition::chroma_plane_residual_size(geometry, subsize) ==
        BlockSize::kInvalid) {
      return false;
    }
  }

  // is_ext_partition_allowed() / is_uneven_4way_partition_allowed() restricted
  // to TreeType == SHARED_PART, where only the tool flags matter.
  switch (partition) {
    case PartitionType::kHorz3:
    case PartitionType::kVert3:
      if (!tools.enable_ext_partitions) return false;
      break;
    case PartitionType::kHorz4A:
    case PartitionType::kHorz4B:
    case PartitionType::kVert4A:
    case PartitionType::kVert4B:
      if (!tools.enable_ext_partitions ||
          !tools.enable_uneven_4way_partitions) {
        return false;
      }
      break;
    default:
      return false;
  }

  if (has_chroma && local_offset) {
    const int w4 = partition::num_4x4_blocks_wide(size);
    const int h4 = partition::num_4x4_blocks_high(size);
    const int half_w = w4 >> 1;
    const int half_h = h4 >> 1;
    switch (partition) {
      case PartitionType::kHorz3:
        return partition::block_coded(geometry, mi_row + 3 * (half_h >> 1),
                                      mi_col);
      case PartitionType::kVert3:
        return partition::block_coded(geometry, mi_row,
                                      mi_col + 3 * (half_w >> 1));
      case PartitionType::kHorz4A:
      case PartitionType::kHorz4B: {
        const int h4_sub = partition::num_4x4_blocks_high(subsize);
        return partition::block_coded(geometry, mi_row + 7 * h4_sub, mi_col);
      }
      case PartitionType::kVert4A:
      case PartitionType::kVert4B: {
        const int w4_sub = partition::num_4x4_blocks_wide(subsize);
        return partition::block_coded(geometry, mi_row, mi_col + 7 * w4_sub);
      }
      default:
        break;
    }
  }
  return true;
}

// Candidate iteration order for a deterministic solver.
void candidate_order(TieBreakOrder order, PartitionType out[kPartitionCount]) {
  if (order == TieBreakOrder::kRectCompatOrder) {
    const PartitionType seq[kPartitionCount] = {
        PartitionType::kNone,    PartitionType::kSplit,
        PartitionType::kHorz,    PartitionType::kVert,
        PartitionType::kHorz3,   PartitionType::kVert3,
        PartitionType::kHorz4A,  PartitionType::kHorz4B,
        PartitionType::kVert4A,  PartitionType::kVert4B,
    };
    for (int i = 0; i < kPartitionCount; ++i) out[i] = seq[i];
    return;
  }
  for (int i = 0; i < kPartitionCount; ++i) {
    out[i] = static_cast<PartitionType>(i);
  }
}

bool policy_allows(BlockSize size, PartitionType partition, int depth,
                   const DecisionOptions& options) {
  if (partition == PartitionType::kNone) return true;
  if (partition == PartitionType::kSplit && !options.allow_split) return false;
  if ((partition == PartitionType::kHorz || partition == PartitionType::kVert) &&
      !options.allow_rect) {
    return false;
  }
  if (is_three_way(partition) && !options.allow_3way) return false;
  if (is_uneven_4way(partition) && !options.allow_uneven4) return false;
  if (depth + 1 > options.max_depth) return false;
  const BlockSize subsize = partition_subsize(partition, size);
  if (!partition::is_valid_block_size(subsize)) return false;
  const int floor_px = std::max(options.min_block_size_px, kMinBlockPixels);
  return partition::block_width(subsize) >= floor_px &&
         partition::block_height(subsize) >= floor_px;
}

struct SolveResult {
  DecisionStatus status = DecisionStatus::kOk;
  double cost = 0.0;
  std::string error;
};

SolveResult solve_node(const FrameGeometry& geometry, int mi_row, int mi_col,
                       BlockSize size, int depth, bool has_chroma,
                       bool chroma_offset, const ChromaBlock& inherited,
                       const CostFunction& cost_fn, const Tools& tools,
                       const DecisionOptions& options, PartitionNode& node);

SolveResult solve_candidate(const FrameGeometry& geometry, int mi_row,
                            int mi_col, BlockSize size, int depth,
                            bool has_chroma, bool chroma_offset,
                            const ChromaBlock& inherited,
                            PartitionType partition, double signal_cost,
                            const CostFunction& cost_fn, const Tools& tools,
                            const DecisionOptions& options,
                            std::vector<PartitionNode>& children,
                            double& total_cost) {
  SolveResult result;
  total_cost = signal_cost;

  ChildSlot slots[4];
  const int count = enumerate_children(geometry, mi_row, mi_col, size,
                                       partition, has_chroma, chroma_offset,
                                       inherited, slots);
  if (count <= 0) {
    result.status = DecisionStatus::kNoValidPartition;
    result.error = "partition has no child slots";
    return result;
  }

  children.clear();
  children.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    const ChildSlot& slot = slots[index];
    if (!child_is_coded(geometry, slot)) continue;
    PartitionNode child;
    SolveResult child_result =
        solve_node(geometry, slot.mi_row, slot.mi_col, slot.size, depth + 1,
                   slot.has_chroma, slot.chroma_offset, slot.chroma, cost_fn,
                   tools, options, child);
    if (child_result.status != DecisionStatus::kOk) {
      result = std::move(child_result);
      return result;
    }
    total_cost += child_result.cost;
    children.push_back(std::move(child));
  }
  if (!std::isfinite(total_cost)) {
    result.status = DecisionStatus::kNoValidPartition;
    result.error = "candidate subtree cost is not finite";
    return result;
  }
  return result;
}

SolveResult solve_node(const FrameGeometry& geometry, int mi_row, int mi_col,
                       BlockSize size, int depth, bool has_chroma,
                       bool chroma_offset, const ChromaBlock& inherited,
                       const CostFunction& cost_fn, const Tools& tools,
                       const DecisionOptions& options, PartitionNode& node) {
  SolveResult result;
  if (!geometry_ready(geometry) ||
      !partition::is_valid_block_size(size) || mi_row < 0 || mi_col < 0 ||
      mi_row >= partition::mi_rows(geometry) ||
      mi_col >= partition::mi_cols(geometry)) {
    result.status = DecisionStatus::kInvalidGeometry;
    result.error = "invalid node geometry";
    return result;
  }
  if (!cost_fn) {
    result.status = DecisionStatus::kInvalidGeometry;
    result.error = "cost function is empty";
    return result;
  }

  NodeInfo info;
  info.mi_row = mi_row;
  info.mi_col = mi_col;
  info.size = size;
  info.depth = depth;
  info.has_chroma = has_chroma;
  info.chroma_offset = chroma_offset;
  info.boundary = implied_partition(geometry, mi_row, mi_col, size);
  info.allowed = init_allowed_partitions(geometry, mi_row, mi_col, size,
                                         has_chroma, chroma_offset, tools);
  const ChromaState chroma =
      resolve_chroma(geometry, mi_row, mi_col, size, PartitionType::kNone,
                     has_chroma, chroma_offset, inherited);
  info.chroma = chroma.block;

  const NodeCosts costs = cost_fn(info);
  if (!all_costs_sane(costs)) {
    result.status = DecisionStatus::kInvalidCost;
    result.error = "cost function returned NaN or a negative cost";
    return result;
  }

  bool forced = false;
  PartitionType forced_partition = PartitionType::kNone;
  if (info.boundary.implied &&
      info.allowed.is_allowed(info.boundary.implied_partition)) {
    forced = true;
    forced_partition = info.boundary.implied_partition;
  } else if (info.allowed.count == 1) {
    forced = true;
    PartitionType order[kPartitionCount];
    candidate_order(TieBreakOrder::kSpecOrder, order);
    for (PartitionType candidate : order) {
      if (info.allowed.is_allowed(candidate)) {
        forced_partition = candidate;
        break;
      }
    }
  }

  double best_cost = kInfiniteCost;
  PartitionType best_partition = PartitionType::kNone;
  std::vector<PartitionNode> best_children;
  bool found = false;
  bool saw_invalid_cost = false;
  std::string last_error;

  PartitionType order[kPartitionCount];
  candidate_order(options.tie_break, order);
  for (PartitionType partition : order) {
    if (forced && partition != forced_partition) continue;
    if (!info.allowed.is_allowed(partition)) continue;
    if (!policy_allows(size, partition, depth, options)) {
      if (forced) {
        result.status = DecisionStatus::kNoValidPartition;
        result.error = "spec-forced partition conflicts with DecisionOptions";
        return result;
      }
      continue;
    }

    const double signal = forced ? 0.0 : partition_signal_cost(costs, partition);
    if (!std::isfinite(signal)) continue;

    if (partition == PartitionType::kNone) {
      if (!std::isfinite(costs.leaf)) continue;
      const double total = costs.leaf + signal;
      if (!std::isfinite(total)) continue;
      if (!found || total < best_cost) {
        found = true;
        best_cost = total;
        best_partition = partition;
        best_children.clear();
      }
      continue;
    }

    std::vector<PartitionNode> children;
    double total = 0.0;
    SolveResult candidate = solve_candidate(
        geometry, mi_row, mi_col, size, depth, has_chroma, chroma_offset,
        inherited, partition, signal, cost_fn, tools, options, children, total);
    if (candidate.status != DecisionStatus::kOk) {
      if (candidate.status == DecisionStatus::kInvalidCost) {
        saw_invalid_cost = true;
      }
      if (!candidate.error.empty()) last_error = std::move(candidate.error);
      continue;
    }
    if (!found || total < best_cost) {
      found = true;
      best_cost = total;
      best_partition = partition;
      best_children = std::move(children);
    }
  }

  if (!found) {
    result.status = saw_invalid_cost ? DecisionStatus::kInvalidCost
                                     : DecisionStatus::kNoValidPartition;
    result.error = last_error.empty()
                       ? "no partition candidate has a finite cost"
                       : std::move(last_error);
    return result;
  }

  node.mi_row = mi_row;
  node.mi_col = mi_col;
  node.size = size;
  node.partition = best_partition;
  node.depth = depth;
  node.has_chroma = has_chroma;
  node.chroma_offset = chroma_offset;
  node.chroma = info.chroma;
  node.boundary = info.boundary;
  node.allowed = info.allowed;
  node.forced = forced;
  node.cost = best_cost;
  node.children = std::move(best_children);

  result.status = DecisionStatus::kOk;
  result.cost = best_cost;
  return result;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

std::string node_label(const PartitionNode& node) {
  std::ostringstream out;
  out << partition::block_size_name(node.size) << '@' << '(' << node.mi_row
      << ',' << node.mi_col << ") depth " << node.depth;
  return out.str();
}

struct ValidateState {
  const FrameGeometry& geometry;
  const Tools& tools;
  int root_mi_row = 0;
  int root_mi_col = 0;
  int sb4 = 0;
  std::vector<std::uint8_t> covered;
  int expected_cells = 0;
  std::string error;
};

bool same_set(const PartitionSet& a, const PartitionSet& b) {
  if (a.count != b.count) return false;
  for (int i = 0; i < kPartitionCount; ++i) {
    if (a.allowed[i] != b.allowed[i]) return false;
  }
  return true;
}

bool validate_node(ValidateState& state, const PartitionNode& node,
                   BlockSize expected_size, int expected_row, int expected_col,
                   int expected_depth, bool expected_has_chroma,
                   bool expected_offset, const ChromaBlock& expected_inherited) {
  if (node.size != expected_size || node.mi_row != expected_row ||
      node.mi_col != expected_col) {
    std::ostringstream out;
    out << "node position/size mismatch: got " << node_label(node)
        << ", expected " << partition::block_size_name(expected_size) << '@'
        << '(' << expected_row << ',' << expected_col << ')';
    return fail(state.error, out.str());
  }
  if (node.depth != expected_depth) {
    return fail(state.error,
                "node " + node_label(node) + " has wrong depth");
  }
  if (node.has_chroma != expected_has_chroma ||
      node.chroma_offset != expected_offset) {
    return fail(state.error,
                "node " + node_label(node) + " has wrong chroma state");
  }
  if (!partition::is_inside(state.geometry, node.mi_row, node.mi_col)) {
    return fail(state.error,
                "node " + node_label(node) + " starts outside frame");
  }

  const Boundary boundary = implied_partition(
      state.geometry, node.mi_row, node.mi_col, node.size);
  const PartitionSet allowed = init_allowed_partitions(
      state.geometry, node.mi_row, node.mi_col, node.size, node.has_chroma,
      node.chroma_offset, state.tools);
  if (node.boundary.has_rows != boundary.has_rows ||
      node.boundary.has_cols != boundary.has_cols ||
      node.boundary.implied != boundary.implied ||
      node.boundary.implied_partition != boundary.implied_partition ||
      !same_set(node.allowed, allowed)) {
    return fail(state.error,
                "node " + node_label(node) +
                    " has stale boundary/allowed fields");
  }
  if (!allowed.is_allowed(node.partition)) {
    return fail(state.error,
                "node " + node_label(node) +
                    " uses a partition the spec forbids");
  }

  const ChromaState chroma = resolve_chroma(
      state.geometry, node.mi_row, node.mi_col, node.size, node.partition,
      node.has_chroma, node.chroma_offset, expected_inherited);
  if (node.chroma != chroma.block) {
    return fail(state.error,
                "node " + node_label(node) + " has a stale chroma block");
  }

  bool spec_forced = false;
  PartitionType spec_partition = PartitionType::kNone;
  if (boundary.implied && allowed.is_allowed(boundary.implied_partition)) {
    spec_forced = true;
    spec_partition = boundary.implied_partition;
  } else if (allowed.count == 1) {
    spec_forced = true;
    for (int index = 0; index < kPartitionCount; ++index) {
      const PartitionType candidate = static_cast<PartitionType>(index);
      if (allowed.is_allowed(candidate)) {
        spec_partition = candidate;
        break;
      }
    }
  }
  if (spec_forced) {
    if (!node.forced || node.partition != spec_partition) {
      return fail(state.error,
                  "node " + node_label(node) +
                      " does not use the spec-forced partition");
    }
  } else if (node.forced) {
    return fail(state.error,
                "node " + node_label(node) + " claims to be forced but is not");
  }

  if (node.partition == PartitionType::kNone) {
    if (!node.children.empty()) {
      return fail(state.error,
                  "PARTITION_NONE node " + node_label(node) + " has children");
    }
    const int w4 = partition::num_4x4_blocks_wide(node.size);
    const int h4 = partition::num_4x4_blocks_high(node.size);
    const int row_end = std::min(node.mi_row + h4,
                                 partition::mi_rows(state.geometry));
    const int col_end = std::min(node.mi_col + w4,
                                 partition::mi_cols(state.geometry));
    for (int row = node.mi_row; row < row_end; ++row) {
      for (int col = node.mi_col; col < col_end; ++col) {
        const int local_row = row - state.root_mi_row;
        const int local_col = col - state.root_mi_col;
        if (local_row < 0 || local_row >= state.sb4 || local_col < 0 ||
            local_col >= state.sb4) {
          return fail(state.error,
                      "leaf " + node_label(node) + " escapes its superblock");
        }
        const std::size_t index =
            static_cast<std::size_t>(local_row) *
                static_cast<std::size_t>(state.sb4) +
            static_cast<std::size_t>(local_col);
        if (state.covered[index] != 0) {
          return fail(state.error, "overlapping leaves at mi (" +
                                       std::to_string(row) + ',' +
                                       std::to_string(col) + ')');
        }
        state.covered[index] = 1;
      }
    }
    return true;
  }

  ChildSlot slots[4];
  const int slot_count = enumerate_children(
      state.geometry, node.mi_row, node.mi_col, node.size, node.partition,
      node.has_chroma, node.chroma_offset, expected_inherited, slots);
  if (slot_count == 0) {
    return fail(state.error,
                "node " + node_label(node) + " has no child slots");
  }

  int coded_slots = 0;
  for (int index = 0; index < slot_count; ++index) {
    if (child_is_coded(state.geometry, slots[index])) ++coded_slots;
  }
  if (node.children.size() != static_cast<std::size_t>(coded_slots)) {
    std::ostringstream out;
    out << partition_name(node.partition) << " node " << node_label(node)
        << " has " << node.children.size() << " children, expected "
        << coded_slots << " in-frame children";
    return fail(state.error, out.str());
  }

  std::size_t child_index = 0;
  for (int index = 0; index < slot_count; ++index) {
    const ChildSlot& slot = slots[index];
    if (!child_is_coded(state.geometry, slot)) continue;
    if (child_index >= node.children.size()) {
      return fail(state.error,
                  "node " + node_label(node) + " is missing a child");
    }
    if (!validate_node(state, node.children[child_index], slot.size,
                       slot.mi_row, slot.mi_col, node.depth + 1,
                       slot.has_chroma, slot.chroma_offset, slot.chroma)) {
      return false;
    }
    ++child_index;
  }
  if (child_index != node.children.size()) {
    return fail(state.error,
                "node " + node_label(node) + " has an unexpected extra child");
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Partition enum / tables
// ---------------------------------------------------------------------------

const char* partition_name(PartitionType partition) {
  switch (partition) {
    case PartitionType::kNone:
      return "NONE";
    case PartitionType::kHorz:
      return "HORZ";
    case PartitionType::kVert:
      return "VERT";
    case PartitionType::kHorz3:
      return "HORZ_3";
    case PartitionType::kVert3:
      return "VERT_3";
    case PartitionType::kHorz4A:
      return "HORZ_4A";
    case PartitionType::kHorz4B:
      return "HORZ_4B";
    case PartitionType::kVert4A:
      return "VERT_4A";
    case PartitionType::kVert4B:
      return "VERT_4B";
    case PartitionType::kSplit:
      return "SPLIT";
  }
  return "INVALID";
}

bool is_valid_partition(PartitionType partition) {
  const int value = static_cast<int>(partition);
  return value >= 0 && value < kPartitionCount;
}

bool is_extended(PartitionType partition) {
  return partition >= PartitionType::kHorz3 && partition <= PartitionType::kVert4B;
}

bool is_three_way(PartitionType partition) {
  return partition == PartitionType::kHorz3 || partition == PartitionType::kVert3;
}

bool is_uneven_4way(PartitionType partition) {
  return partition == PartitionType::kHorz4A ||
         partition == PartitionType::kHorz4B ||
         partition == PartitionType::kVert4A ||
         partition == PartitionType::kVert4B;
}

bool is_four_way(PartitionType partition) {
  return is_three_way(partition) || is_uneven_4way(partition);
}

bool is_horizontal(PartitionType partition) {
  return partition == PartitionType::kHorz ||
         partition == PartitionType::kHorz3 ||
         partition == PartitionType::kHorz4A ||
         partition == PartitionType::kHorz4B;
}

bool is_vertical(PartitionType partition) {
  return partition == PartitionType::kVert ||
         partition == PartitionType::kVert3 ||
         partition == PartitionType::kVert4A ||
         partition == PartitionType::kVert4B;
}

PartitionType rect_direction(PartitionType partition) {
  if (is_horizontal(partition)) return PartitionType::kHorz;
  if (is_vertical(partition)) return PartitionType::kVert;
  return PartitionType::kNone;
}

partition::PartitionType to_base(PartitionType partition) {
  switch (partition) {
    case PartitionType::kNone:
      return partition::PartitionType::kNone;
    case PartitionType::kHorz:
      return partition::PartitionType::kHorz;
    case PartitionType::kVert:
      return partition::PartitionType::kVert;
    case PartitionType::kHorz3:
    case PartitionType::kHorz4A:
    case PartitionType::kHorz4B:
      return partition::PartitionType::kHorz;
    case PartitionType::kVert3:
    case PartitionType::kVert4A:
    case PartitionType::kVert4B:
      return partition::PartitionType::kVert;
    case PartitionType::kSplit:
      return partition::PartitionType::kSplit;
  }
  return partition::PartitionType::kNone;
}

PartitionType from_base(partition::PartitionType partition) {
  return base_to_ext(partition);
}

bool validate_tools(const Tools& tools, std::string& error) {
  if (tools.enable_uneven_4way_partitions && !tools.enable_ext_partitions) {
    return fail(error,
                "enable_uneven_4way_partitions requires enable_ext_partitions "
                "(§5.4.3)");
  }
  error.clear();
  return true;
}

BlockSize partition_subsize(PartitionType partition, BlockSize size) {
  if (!is_valid_partition(partition) ||
      !partition::is_valid_block_size(size)) {
    return BlockSize::kInvalid;
  }
  const int value = static_cast<int>(kPartitionSubsize
                                         [static_cast<int>(partition)]
                                         [static_cast<int>(size)]);
  if (value >= kExtBlockSizeCount) return BlockSize::kInvalid;
  return static_cast<BlockSize>(value);
}

BlockSize h_partition_midsize(BlockSize size) {
  if (!partition::is_valid_block_size(size)) return BlockSize::kInvalid;
  const int value = static_cast<int>(
      kHPartitionMidsize[static_cast<std::size_t>(size)]);
  if (value >= kExtBlockSizeCount) return BlockSize::kInvalid;
  return static_cast<BlockSize>(value);
}

int size_adjust(BlockSize size) {
  if (!partition::is_valid_block_size(size)) return 0;
  return static_cast<int>(kPartitionSizeAdjust[static_cast<std::size_t>(size)]);
}

int size_adjust_rect_type(BlockSize size) {
  if (!partition::is_valid_block_size(size)) return 0;
  return static_cast<int>(
      kPartitionSizeAdjustRectType[static_cast<std::size_t>(size)]);
}

// ---------------------------------------------------------------------------
// Boundary / implied
// ---------------------------------------------------------------------------

Boundary implied_partition(const FrameGeometry& geometry, int mi_row,
                           int mi_col, BlockSize size) {
  Boundary out;
  const partition::PartitionBoundary boundary =
      partition::partition_implied(geometry, mi_row, mi_col, size);
  out.has_rows = boundary.has_rows;
  out.has_cols = boundary.has_cols;
  out.implied = boundary.implied;
  out.implied_partition = base_to_ext(boundary.implied_partition);
  return out;
}

// ---------------------------------------------------------------------------
// Chroma
// ---------------------------------------------------------------------------

bool operator==(const ChromaBlock& a, const ChromaBlock& b) {
  return a.mi_row == b.mi_row && a.mi_col == b.mi_col && a.size == b.size;
}
bool operator!=(const ChromaBlock& a, const ChromaBlock& b) { return !(a == b); }

bool chroma_offset_for_subsize(const FrameGeometry& geometry,
                                  BlockSize subsize) {
  return partition::is_chroma_offset_for_subsize(geometry, subsize);
}

bool chroma_offset_for_partition(const FrameGeometry& geometry,
                                    PartitionType partition, BlockSize size) {
  if (!partition::is_valid_block_size(size)) return false;
  const BlockSize subsize = partition_subsize(partition, size);
  if (!partition::is_valid_block_size(subsize)) return false;
  if (chroma_offset_for_subsize(geometry, subsize)) return true;
  if (partition == PartitionType::kHorz3) {
    const bool middle_chroma =
        size == BlockSize::k8x32 && geometry.chroma_subsampling_x;
    if (!middle_chroma) {
      if (chroma_offset_for_subsize(geometry, h_partition_midsize(size))) {
        return true;
      }
    }
  }
  return false;
}

ChromaState resolve_chroma(const FrameGeometry& geometry, int mi_row,
                           int mi_col, BlockSize size,
                           PartitionType partition, bool has_chroma,
                           bool chroma_offset, const ChromaBlock& inherited) {
  ChromaState state;
  state.block = inherited;
  state.chroma_offset = chroma_offset;
  if (has_chroma && !chroma_offset) {
    state.block.mi_row = mi_row;
    state.block.mi_col = mi_col;
    state.block.size = size;
    state.chroma_offset =
        chroma_offset_for_partition(geometry, partition, size);
  }
  return state;
}

// ---------------------------------------------------------------------------
// Child enumeration
// ---------------------------------------------------------------------------

bool child_is_coded(const FrameGeometry& geometry, const ChildSlot& slot) {
  return slot.mi_row < partition::mi_rows(geometry) &&
         slot.mi_col < partition::mi_cols(geometry);
}

int enumerate_children(const FrameGeometry& geometry, int mi_row, int mi_col,
                       BlockSize size, PartitionType partition,
                       bool has_chroma, bool chroma_offset,
                       const ChromaBlock& inherited, ChildSlot slots[4]) {
  if (partition == PartitionType::kNone) return 0;
  const BlockSize subsize = partition_subsize(partition, size);
  if (!partition::is_valid_block_size(subsize)) return 0;

  const int w4 = partition::num_4x4_blocks_wide(size);
  const int h4 = partition::num_4x4_blocks_high(size);
  const int half_w = w4 >> 1;
  const int half_h = h4 >> 1;
  const int quarter_w = half_w >> 1;
  const int quarter_h = half_h >> 1;

  const ChromaState chroma = resolve_chroma(
      geometry, mi_row, mi_col, size, partition, has_chroma, chroma_offset,
      inherited);
  const bool first_has_chroma = has_chroma && !chroma.chroma_offset;

  // HORZ_3@8x32 and VERT_3@32x8: the middle pair shares a chroma block whose
  // size is Partition_Subsize[PARTITION_HORZ|VERT][bSize].
  bool middle_chroma = false;
  ChromaBlock middle_block = chroma.block;
  if (partition == PartitionType::kHorz3 && size == BlockSize::k8x32 &&
      has_chroma && geometry.chroma_subsampling_x) {
    middle_chroma = true;
    middle_block.mi_row = mi_row + quarter_h;
    middle_block.mi_col = mi_col;
    middle_block.size = partition_subsize(PartitionType::kHorz, size);
  } else if (partition == PartitionType::kVert3 && size == BlockSize::k32x8 &&
             has_chroma && geometry.chroma_subsampling_y) {
    middle_chroma = true;
    middle_block.mi_row = mi_row;
    middle_block.mi_col = mi_col + quarter_w;
    middle_block.size = partition_subsize(PartitionType::kVert, size);
  }

  auto slot = [&](int index, int row, int col, BlockSize child_size,
                  bool child_has_chroma, bool child_offset,
                  const ChromaBlock& child_chroma) {
    slots[index].mi_row = row;
    slots[index].mi_col = col;
    slots[index].size = child_size;
    slots[index].has_chroma = child_has_chroma;
    slots[index].chroma_offset = child_offset;
    slots[index].chroma = child_chroma;
  };

  switch (partition) {
    case PartitionType::kNone:
      return 0;
    case PartitionType::kSplit:
      slot(0, mi_row, mi_col, subsize, has_chroma, false, chroma.block);
      slot(1, mi_row, mi_col + half_w, subsize, has_chroma, false,
           chroma.block);
      slot(2, mi_row + half_h, mi_col, subsize, has_chroma, false,
           chroma.block);
      slot(3, mi_row + half_h, mi_col + half_w, subsize, has_chroma, false,
           chroma.block);
      return 4;
    case PartitionType::kHorz:
      slot(0, mi_row, mi_col, subsize, first_has_chroma, chroma.chroma_offset,
           chroma.block);
      slot(1, mi_row + half_h, mi_col, subsize, has_chroma,
           chroma.chroma_offset, chroma.block);
      return 2;
    case PartitionType::kVert:
      slot(0, mi_row, mi_col, subsize, first_has_chroma, chroma.chroma_offset,
           chroma.block);
      slot(1, mi_row, mi_col + half_w, subsize, has_chroma,
           chroma.chroma_offset, chroma.block);
      return 2;
    case PartitionType::kHorz3: {
      const BlockSize midsize = h_partition_midsize(size);
      if (!partition::is_valid_block_size(midsize)) return 0;
      const ChromaBlock mid = middle_chroma ? middle_block : chroma.block;
      const ChromaBlock tail = middle_chroma ? middle_block : chroma.block;
      slot(0, mi_row, mi_col, subsize, first_has_chroma,
           chroma.chroma_offset, chroma.block);
      slot(1, mi_row + quarter_h, mi_col, midsize,
           first_has_chroma && !middle_chroma,
           chroma.chroma_offset || middle_chroma, mid);
      slot(2, mi_row + quarter_h, mi_col + half_w, midsize,
           first_has_chroma, chroma.chroma_offset || middle_chroma, mid);
      slot(3, mi_row + 3 * quarter_h, mi_col, subsize, has_chroma,
           chroma.chroma_offset, tail);
      return 4;
    }
    case PartitionType::kVert3: {
      const BlockSize midsize = h_partition_midsize(size);
      if (!partition::is_valid_block_size(midsize)) return 0;
      const ChromaBlock mid = middle_chroma ? middle_block : chroma.block;
      const ChromaBlock tail = middle_chroma ? middle_block : chroma.block;
      slot(0, mi_row, mi_col, subsize, first_has_chroma,
           chroma.chroma_offset, chroma.block);
      slot(1, mi_row, mi_col + quarter_w, midsize,
           first_has_chroma && !middle_chroma,
           chroma.chroma_offset || middle_chroma, mid);
      slot(2, mi_row + half_h, mi_col + quarter_w, midsize,
           first_has_chroma, chroma.chroma_offset || middle_chroma, mid);
      slot(3, mi_row, mi_col + 3 * quarter_w, subsize, has_chroma,
           chroma.chroma_offset, tail);
      return 4;
    }
    case PartitionType::kHorz4A:
    case PartitionType::kHorz4B: {
      const BlockSize big =
          partition_subsize(PartitionType::kHorz, size);
      const BlockSize med = partition_subsize(PartitionType::kHorz, big);
      if (!partition::is_valid_block_size(big) ||
          !partition::is_valid_block_size(med)) {
        return 0;
      }
      const int unit = h4 >> 3;
      const bool a = partition == PartitionType::kHorz4A;
      slot(0, mi_row, mi_col, subsize, first_has_chroma,
           chroma.chroma_offset, chroma.block);
      slot(1, mi_row + unit, mi_col, a ? med : big, first_has_chroma,
           chroma.chroma_offset, chroma.block);
      slot(2, mi_row + (a ? 3 : 5) * unit, mi_col, a ? big : med,
           first_has_chroma, chroma.chroma_offset, chroma.block);
      slot(3, mi_row + 7 * unit, mi_col, subsize, has_chroma,
           chroma.chroma_offset, chroma.block);
      return 4;
    }
    case PartitionType::kVert4A:
    case PartitionType::kVert4B: {
      const BlockSize big = partition_subsize(PartitionType::kVert, size);
      const BlockSize med = partition_subsize(PartitionType::kVert, big);
      if (!partition::is_valid_block_size(big) ||
          !partition::is_valid_block_size(med)) {
        return 0;
      }
      const int unit = w4 >> 3;
      const bool a = partition == PartitionType::kVert4A;
      slot(0, mi_row, mi_col, subsize, first_has_chroma,
           chroma.chroma_offset, chroma.block);
      slot(1, mi_row, mi_col + unit, a ? med : big, first_has_chroma,
           chroma.chroma_offset, chroma.block);
      slot(2, mi_row, mi_col + (a ? 3 : 5) * unit, a ? big : med,
           first_has_chroma, chroma.chroma_offset, chroma.block);
      slot(3, mi_row, mi_col + 7 * unit, subsize, has_chroma,
           chroma.chroma_offset, chroma.block);
      return 4;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Allowed set
// ---------------------------------------------------------------------------

bool PartitionSet::is_allowed(PartitionType partition) const {
  if (!is_valid_partition(partition)) return false;
  return allowed[static_cast<int>(partition)];
}

bool is_partition_allowed(const FrameGeometry& geometry, int mi_row,
                          int mi_col, BlockSize size,
                          PartitionType partition, bool has_chroma,
                          bool chroma_offset, const Tools& tools) {
  if (is_base_partition(partition)) {
    return partition::is_partition_allowed(
        geometry, mi_row, mi_col, size, to_base(partition), has_chroma,
        chroma_offset);
  }
  return is_ext_partition_allowed(geometry, mi_row, mi_col, size, partition,
                                  has_chroma, chroma_offset, tools);
}

PartitionSet init_allowed_partitions(const FrameGeometry& geometry, int mi_row,
                                     int mi_col, BlockSize size,
                                     bool has_chroma, bool chroma_offset,
                                     const Tools& tools) {
  PartitionSet set;
  for (int index = 0; index < kPartitionCount; ++index) {
    const PartitionType partition = static_cast<PartitionType>(index);
    const bool good = is_partition_allowed(geometry, mi_row, mi_col, size,
                                           partition, has_chroma, chroma_offset,
                                           tools);
    set.allowed[index] = good;
    if (good) ++set.count;
  }
  if (set.count == 0) {
    set.allowed[static_cast<int>(PartitionType::kNone)] = true;
    set.count = 1;
  }
  return set;
}

// ---------------------------------------------------------------------------
// Contexts
// ---------------------------------------------------------------------------

partition::PartitionContexts partition_contexts(const FrameGeometry& geometry,
                                                int mi_row, int mi_col,
                                                BlockSize size,
                                                const MiSizeGrid& sizes) {
  return partition::partition_contexts(geometry, mi_row, mi_col, size, sizes);
}

int ext_partition_context(const FrameGeometry& geometry, int mi_row,
                          int mi_col, BlockSize size, bool horizontal,
                          const MiSizeGrid& luma_sizes) {
  if (!partition::is_valid_block_size(size)) return 0;
  const int adjust = size_adjust(size);
  int ctx1 = 0;
  int ctx2 = 0;
  if (horizontal) {
    const int bsh = std::max(partition::mi_height_log2(size) - 1, 1);
    const int h4 = partition::num_4x4_blocks_high(size);
    ctx1 = log2_or_sentinel(luma_sizes, mi_row, mi_col - 1, /*width=*/false) < bsh
               ? 1
               : 0;
    ctx2 = log2_or_sentinel(luma_sizes, mi_row + (h4 >> 1), mi_col - 1,
                            /*width=*/false) < bsh
               ? 1
               : 0;
  } else {
    const int bsw = std::max(partition::mi_width_log2(size) - 1, 1);
    const int w4 = partition::num_4x4_blocks_wide(size);
    ctx1 = log2_or_sentinel(luma_sizes, mi_row - 1, mi_col, /*width=*/true) < bsw
               ? 1
               : 0;
    ctx2 = log2_or_sentinel(luma_sizes, mi_row - 1, mi_col + (w4 >> 1),
                            /*width=*/true) < bsw
               ? 1
               : 0;
  }
  (void)geometry;
  return adjust * 4 + ctx1 * 2 + ctx2;
}

// ---------------------------------------------------------------------------
// Decision API
// ---------------------------------------------------------------------------

NodeCosts from_partition_costs(const partition::NodeCosts& costs) {
  NodeCosts out;
  out.leaf = costs.leaf;
  out.none_signal = costs.none_signal;
  out.split_signal = costs.split_signal;
  out.horz_signal = costs.horz_signal;
  out.vert_signal = costs.vert_signal;
  out.horz3_signal = kInfiniteCost;
  out.vert3_signal = kInfiniteCost;
  out.horz4a_signal = kInfiniteCost;
  out.horz4b_signal = kInfiniteCost;
  out.vert4a_signal = kInfiniteCost;
  out.vert4b_signal = kInfiniteCost;
  return out;
}

double partition_signal_cost(const NodeCosts& costs, PartitionType partition) {
  switch (partition) {
    case PartitionType::kNone:
      return costs.none_signal;
    case PartitionType::kSplit:
      return costs.split_signal;
    case PartitionType::kHorz:
      return costs.horz_signal;
    case PartitionType::kVert:
      return costs.vert_signal;
    case PartitionType::kHorz3:
      return costs.horz3_signal;
    case PartitionType::kVert3:
      return costs.vert3_signal;
    case PartitionType::kHorz4A:
      return costs.horz4a_signal;
    case PartitionType::kHorz4B:
      return costs.horz4b_signal;
    case PartitionType::kVert4A:
      return costs.vert4a_signal;
    case PartitionType::kVert4B:
      return costs.vert4b_signal;
  }
  return kInfiniteCost;
}

const char* decision_status_name(DecisionStatus status) {
  switch (status) {
    case DecisionStatus::kOk:
      return "ok";
    case DecisionStatus::kInvalidGeometry:
      return "invalid-geometry";
    case DecisionStatus::kInvalidTools:
      return "invalid-tools";
    case DecisionStatus::kInvalidCost:
      return "invalid-cost";
    case DecisionStatus::kNoValidPartition:
      return "no-valid-partition";
  }
  return "unknown";
}

PartitionDecision decide_superblock(const FrameGeometry& geometry,
                                    int sb_mi_row, int sb_mi_col,
                                    const CostFunction& costs,
                                    const Tools& tools,
                                    const DecisionOptions& options) {
  PartitionDecision decision;
  std::string error;
  if (!partition::validate_geometry(geometry, error)) {
    decision.status = DecisionStatus::kInvalidGeometry;
    decision.error = std::move(error);
    return decision;
  }
  if (!validate_tools(tools, error)) {
    decision.status = DecisionStatus::kInvalidTools;
    decision.error = std::move(error);
    return decision;
  }
  const int size_mi = partition::sb_size_mi(geometry);
  if (size_mi <= 0 || sb_mi_row < 0 || sb_mi_col < 0 ||
      sb_mi_row >= partition::mi_rows(geometry) ||
      sb_mi_col >= partition::mi_cols(geometry) ||
      (sb_mi_row % size_mi) != 0 || (sb_mi_col % size_mi) != 0) {
    decision.status = DecisionStatus::kInvalidGeometry;
    decision.error = "superblock origin is outside or unaligned";
    return decision;
  }

  const BlockSize root_size = partition::superblock_size(geometry.sb_size);
  ChromaBlock inherited;
  inherited.mi_row = sb_mi_row;
  inherited.mi_col = sb_mi_col;
  inherited.size = root_size;
  SolveResult result = solve_node(
      geometry, sb_mi_row, sb_mi_col, root_size, /*depth=*/0,
      geometry.has_chroma, /*chroma_offset=*/false, inherited, costs, tools,
      options, decision.root);
  decision.status = result.status;
  decision.error = std::move(result.error);
  if (decision.status == DecisionStatus::kOk) {
    decision.cost = result.cost;
    decision.nodes = count_nodes(decision.root);
    decision.leaves = count_leaves(decision.root);
  }
  return decision;
}

FrameDecision decide_frame(const FrameGeometry& geometry,
                           const CostFunction& costs, const Tools& tools,
                           const DecisionOptions& options) {
  FrameDecision frame;
  std::string error;
  if (!partition::validate_geometry(geometry, error)) {
    frame.status = DecisionStatus::kInvalidGeometry;
    frame.error = std::move(error);
    return frame;
  }
  if (!validate_tools(tools, error)) {
    frame.status = DecisionStatus::kInvalidTools;
    frame.error = std::move(error);
    return frame;
  }
  if (!costs) {
    frame.status = DecisionStatus::kInvalidGeometry;
    frame.error = "cost function is empty";
    return frame;
  }

  const int size_mi = partition::sb_size_mi(geometry);
  for (int row = 0; row < partition::mi_rows(geometry); row += size_mi) {
    for (int col = 0; col < partition::mi_cols(geometry); col += size_mi) {
      PartitionDecision decision =
          decide_superblock(geometry, row, col, costs, tools, options);
      if (decision.status != DecisionStatus::kOk) {
        std::ostringstream out;
        out << "superblock (" << row << ',' << col
            << ") failed: " << decision.error;
        frame.status = decision.status;
        frame.error = out.str();
        frame.superblocks.clear();
        frame.cost = 0.0;
        frame.nodes = 0;
        frame.leaves = 0;
        return frame;
      }
      frame.cost += decision.cost;
      frame.nodes += decision.nodes;
      frame.leaves += decision.leaves;
      frame.superblocks.push_back(std::move(decision.root));
    }
  }
  frame.status = DecisionStatus::kOk;
  return frame;
}

void collect_leaves(const PartitionNode& node, std::vector<LeafBlock>& leaves) {
  if (node.partition == PartitionType::kNone) {
    LeafBlock leaf;
    leaf.mi_row = node.mi_row;
    leaf.mi_col = node.mi_col;
    leaf.size = node.size;
    leaf.depth = node.depth;
    leaf.has_chroma = node.has_chroma;
    leaf.chroma_offset = node.chroma_offset;
    leaf.chroma = node.chroma;
    leaves.push_back(leaf);
    return;
  }
  for (const PartitionNode& child : node.children) {
    collect_leaves(child, leaves);
  }
}

int count_nodes(const PartitionNode& node) {
  int count = 1;
  for (const PartitionNode& child : node.children) {
    count += count_nodes(child);
  }
  return count;
}

int count_leaves(const PartitionNode& node) {
  if (node.partition == PartitionType::kNone) return 1;
  int count = 0;
  for (const PartitionNode& child : node.children) {
    count += count_leaves(child);
  }
  return count;
}

// ---------------------------------------------------------------------------
// Validation entry points
// ---------------------------------------------------------------------------

bool validate_tree(const FrameGeometry& geometry, const Tools& tools,
                   const PartitionNode& root, std::string& error) {
  std::string geometry_error;
  if (!partition::validate_geometry(geometry, geometry_error)) {
    return fail(error, std::move(geometry_error));
  }
  if (!validate_tools(tools, geometry_error)) {
    return fail(error, std::move(geometry_error));
  }
  if (root.size != partition::superblock_size(geometry.sb_size)) {
    return fail(error, "root is not a superblock-sized node");
  }
  const int size_mi = partition::sb_size_mi(geometry);
  if (root.mi_row < 0 || root.mi_col < 0 || (root.mi_row % size_mi) != 0 ||
      (root.mi_col % size_mi) != 0) {
    return fail(error, "root origin is not superblock-aligned");
  }
  if (!partition::is_inside(geometry, root.mi_row, root.mi_col)) {
    return fail(error, "root origin is outside the frame");
  }

  ValidateState state{geometry, tools, root.mi_row, root.mi_col, size_mi, {},
                      0, {}};
  state.covered.assign(
      static_cast<std::size_t>(size_mi) * static_cast<std::size_t>(size_mi),
      0);
  const int row_end = std::min(root.mi_row + size_mi,
                               partition::mi_rows(geometry));
  const int col_end = std::min(root.mi_col + size_mi,
                               partition::mi_cols(geometry));
  state.expected_cells = (row_end - root.mi_row) * (col_end - root.mi_col);

  ChromaBlock inherited;
  inherited.mi_row = root.mi_row;
  inherited.mi_col = root.mi_col;
  inherited.size = root.size;
  if (!validate_node(state, root, root.size, root.mi_row, root.mi_col, 0,
                     geometry.has_chroma, /*expected_offset=*/false,
                     inherited)) {
    error = std::move(state.error);
    return false;
  }

  int covered = 0;
  for (std::uint8_t cell : state.covered) {
    covered += cell != 0 ? 1 : 0;
  }
  if (covered != state.expected_cells) {
    std::ostringstream out;
    out << "tree covers " << covered << " of " << state.expected_cells
        << " in-frame cells";
    return fail(error, out.str());
  }
  error.clear();
  return true;
}

bool validate_frame(const FrameGeometry& geometry, const Tools& tools,
                    const FrameDecision& frame, std::string& error) {
  std::string geometry_error;
  if (!partition::validate_geometry(geometry, geometry_error)) {
    return fail(error, std::move(geometry_error));
  }
  if (!validate_tools(tools, geometry_error)) {
    return fail(error, std::move(geometry_error));
  }
  if (frame.status != DecisionStatus::kOk) {
    return fail(error, "frame decision status is not ok");
  }
  const std::size_t expected =
      static_cast<std::size_t>(partition::sb_rows(geometry)) *
      static_cast<std::size_t>(partition::sb_cols(geometry));
  if (frame.superblocks.size() != expected) {
    std::ostringstream out;
    out << "frame has " << frame.superblocks.size() << " superblocks, expected "
        << expected;
    return fail(error, out.str());
  }

  const int size_mi = partition::sb_size_mi(geometry);
  std::size_t index = 0;
  for (int row = 0; row < partition::mi_rows(geometry); row += size_mi) {
    for (int col = 0; col < partition::mi_cols(geometry); col += size_mi) {
      const PartitionNode& root = frame.superblocks[index];
      if (root.mi_row != row || root.mi_col != col) {
        std::ostringstream out;
        out << "superblock " << index << " is at (" << root.mi_row << ','
            << root.mi_col << "), expected (" << row << ',' << col << ')';
        return fail(error, out.str());
      }
      std::string tree_error;
      if (!validate_tree(geometry, tools, root, tree_error)) {
        std::ostringstream out;
        out << "superblock (" << row << ',' << col << "): " << tree_error;
        return fail(error, out.str());
      }
      ++index;
    }
  }
  error.clear();
  return true;
}

bool validate_geometry(const FrameGeometry& geometry, std::string& error) {
  return partition::validate_geometry(geometry, error);
}

}  // namespace av2enc::partition_ext
