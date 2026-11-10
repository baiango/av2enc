// CPU reference for AV2 v1.0.0 recursive partitioning.  See partition.h for
// the scope and the spec clauses implemented.

#include "partition.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

namespace av2enc::partition {
namespace {

// ---------------------------------------------------------------------------
// §9.2 tables.  Numeric values are the spec's enum_block_size / enum_tx values;
// 255 is BLOCK_INVALID.
// ---------------------------------------------------------------------------

constexpr std::uint8_t kInvalid = 255;

constexpr std::uint8_t kMiWidthLog2[kBlockSizeCount] = {
    0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 0,
    2, 1, 3, 2, 4, 0, 3, 1, 4,
};

constexpr std::uint8_t kMiHeightLog2[kBlockSizeCount] = {
    0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 2,
    0, 3, 1, 4, 2, 3, 0, 4, 1,
};

constexpr std::uint8_t kPartitionSizeAdjust[kBlockSizeCount] = {
    0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0,
};

constexpr std::uint8_t kPartitionSizeAdjustRect[kBlockSizeCount] = {
    0, 0, 0, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 13, 14, 13, 14, 0, 0, 0, 0,
};

// Partition_Subsize rows for PARTITION_HORZ (partition enum 1), PARTITION_VERT
// (2), and PARTITION_SPLIT (9).  Rows 3..8 are the extended partitions, which
// M1 does not use.
constexpr std::uint8_t kHorzSubsize[kBlockSizeCount] = {
    kInvalid, 0,  kInvalid, 2,  3,  20, 5,  6,  22, 8,
    9,        24, 11,       12, kInvalid, 14, 15, kInvalid, 17, 1,
    kInvalid, 4,  26,       7,  28, kInvalid, kInvalid, kInvalid, kInvalid,
};

constexpr std::uint8_t kVertSubsize[kBlockSizeCount] = {
    kInvalid, kInvalid, 0,  1,  19, 3,  4,  21, 6,  7,
    23,       9,        10, kInvalid, 12, 13, kInvalid, 15, 16, kInvalid,
    2,        25,       5,  27, 8,  kInvalid, kInvalid, kInvalid, kInvalid,
};

constexpr std::uint8_t kSplitSubsize[kBlockSizeCount] = {
    kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
    kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
    kInvalid, 12, kInvalid, kInvalid, 15, kInvalid, kInvalid, kInvalid,
    kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
};

// Subsampled_Size[subsize][subx=1][suby=1] for 4:2:0 chroma.
constexpr std::uint8_t kChroma420Subsize[kBlockSizeCount] = {
    0,  0,  0,  0,  1,  2,  3,  4,  5,  6,
    7,  8,  9,  10, 11, 12, 13, 14, 15, 1,
    2,  19, 20, 21, 22, 19, 20, 25, 26,
};

constexpr const char* kBlockNames[kBlockSizeCount] = {
    "4x4",   "4x8",   "8x4",   "8x8",   "8x16",  "16x8",  "16x16",
    "16x32", "32x16", "32x32", "32x64", "64x32", "64x64", "64x128",
    "128x64", "128x128", "128x256", "256x128", "256x256", "4x16",
    "16x4",  "8x32",  "32x8",  "16x64", "64x16", "4x32",  "32x4",
    "8x64",  "64x8",
};

constexpr int kSentinelLog2 = 6;  // BLOCK_256X256: never smaller than current

bool geometry_ready(const FrameGeometry& geometry) {
  return geometry.width > 0 && geometry.height > 0 &&
         (geometry.sb_size == 64 || geometry.sb_size == 128) &&
         geometry.max_pb_aspect_ratio >= 1;
}

bool cost_is_sane(double value) {
  return !std::isnan(value) && value >= 0.0;
}

bool all_costs_sane(const NodeCosts& costs) {
  return cost_is_sane(costs.leaf) && cost_is_sane(costs.none_signal) &&
         cost_is_sane(costs.split_signal) && cost_is_sane(costs.horz_signal) &&
         cost_is_sane(costs.vert_signal);
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
  }
  return kInfiniteCost;
}

int log2_or_sentinel(BlockSize size, bool width) {
  if (!is_valid_block_size(size)) {
    return kSentinelLog2;
  }
  return width ? mi_width_log2(size) : mi_height_log2(size);
}

// One child slot produced by a partition.  Mirrors the recursion arguments in
// decode_partition() §5.20.3.1 exactly.
struct ChildSlot {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  bool has_chroma = false;
  bool chroma_offset = false;
};

int enumerate_children(const FrameGeometry& geometry, int mi_row, int mi_col,
                       BlockSize size, PartitionType partition,
                       bool has_chroma, bool chroma_offset,
                       ChildSlot slots[4]) {
  if (partition == PartitionType::kNone) {
    return 0;
  }
  const BlockSize subsize = partition_subsize(partition, size);
  if (!is_valid_block_size(subsize)) {
    return 0;
  }
  const int w4 = num_4x4_blocks_wide(size);
  const int h4 = num_4x4_blocks_high(size);
  const int half_w = w4 >> 1;
  const int half_h = h4 >> 1;

  bool next_offset = chroma_offset;
  if (has_chroma && !next_offset) {
    next_offset = is_chroma_offset_for_partition(geometry, partition, size);
  }

  switch (partition) {
    case PartitionType::kSplit:
      // decode_partition passes chromaOffset = 0 for all four children and
      // leaves hasChroma unchanged.
      slots[0] = {mi_row, mi_col, subsize, has_chroma, false};
      slots[1] = {mi_row, mi_col + half_w, subsize, has_chroma, false};
      slots[2] = {mi_row + half_h, mi_col, subsize, has_chroma, false};
      slots[3] = {mi_row + half_h, mi_col + half_w, subsize, has_chroma,
                  false};
      return 4;
    case PartitionType::kHorz:
      slots[0] = {mi_row, mi_col, subsize, has_chroma && !next_offset,
                  next_offset};
      slots[1] = {mi_row + half_h, mi_col, subsize, has_chroma, next_offset};
      return 2;
    case PartitionType::kVert:
      slots[0] = {mi_row, mi_col, subsize, has_chroma && !next_offset,
                  next_offset};
      slots[1] = {mi_row, mi_col + half_w, subsize, has_chroma, next_offset};
      return 2;
    case PartitionType::kNone:
      break;
  }
  return 0;
}

bool child_is_coded(const FrameGeometry& geometry, const ChildSlot& slot) {
  return slot.mi_row < mi_rows(geometry) && slot.mi_col < mi_cols(geometry);
}

bool policy_allows(const FrameGeometry& geometry, BlockSize size,
                   PartitionType partition, int depth,
                   const DecisionOptions& options) {
  (void)geometry;
  if (partition == PartitionType::kNone) {
    return true;
  }
  if (partition == PartitionType::kSplit && !options.allow_split) {
    return false;
  }
  if ((partition == PartitionType::kHorz || partition == PartitionType::kVert) &&
      !options.allow_rect) {
    return false;
  }
  if (depth + 1 > options.max_depth) {
    return false;
  }
  const BlockSize subsize = partition_subsize(partition, size);
  if (!is_valid_block_size(subsize)) {
    return false;
  }
  const int floor_px = std::max(options.min_block_size_px, kMinBlockPixels);
  return block_width(subsize) >= floor_px && block_height(subsize) >= floor_px;
}

struct SolveResult {
  DecisionStatus status = DecisionStatus::kOk;
  double cost = 0.0;
  std::string error;
};

SolveResult solve_node(const FrameGeometry& geometry, int mi_row, int mi_col,
                       BlockSize size, int depth, bool has_chroma,
                       bool chroma_offset, const CostFunction& cost_fn,
                       const DecisionOptions& options, PartitionNode& node);

SolveResult solve_partition_candidate(
    const FrameGeometry& geometry, int mi_row, int mi_col, BlockSize size,
    int depth, bool has_chroma, bool chroma_offset, PartitionType partition,
    double signal_cost, const CostFunction& cost_fn,
    const DecisionOptions& options, std::vector<PartitionNode>& children,
    double& total_cost) {
  SolveResult result;
  total_cost = signal_cost;

  ChildSlot slots[4];
  const int count =
      enumerate_children(geometry, mi_row, mi_col, size, partition, has_chroma,
                         chroma_offset, slots);
  if (count <= 0) {
    result.status = DecisionStatus::kNoValidPartition;
    result.error = "partition has no child slots";
    return result;
  }

  children.clear();
  children.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    const ChildSlot& slot = slots[index];
    if (!child_is_coded(geometry, slot)) {
      // decode_partition returns before coding a child whose origin is
      // outside the frame; no syntax and no pixels.
      continue;
    }
    PartitionNode child;
    SolveResult child_result =
        solve_node(geometry, slot.mi_row, slot.mi_col, slot.size, depth + 1,
                   slot.has_chroma, slot.chroma_offset, cost_fn, options,
                   child);
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
                       bool chroma_offset, const CostFunction& cost_fn,
                       const DecisionOptions& options, PartitionNode& node) {
  SolveResult result;
  if (!geometry_ready(geometry) || !is_valid_block_size(size) || mi_row < 0 ||
      mi_col < 0 || mi_row >= mi_rows(geometry) ||
      mi_col >= mi_cols(geometry)) {
    result.status = DecisionStatus::kInvalidGeometry;
    result.error = "invalid node geometry";
    return result;
  }
  if (!cost_fn) {
    result.status = DecisionStatus::kInvalidGeometry;
    result.error = "cost function is empty";
    return result;
  }

  PartitionNodeInfo info;
  info.mi_row = mi_row;
  info.mi_col = mi_col;
  info.size = size;
  info.depth = depth;
  info.has_chroma = has_chroma;
  info.chroma_offset = chroma_offset;
  info.boundary = partition_implied(geometry, mi_row, mi_col, size);
  info.allowed = init_allowed_partitions(geometry, mi_row, mi_col, size,
                                         has_chroma, chroma_offset);

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
    for (int index = 0; index < kPartitionTypeCount; ++index) {
      if (info.allowed.allowed[index]) {
        forced_partition = static_cast<PartitionType>(index);
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

  for (int index = 0; index < kPartitionTypeCount; ++index) {
    const PartitionType partition = static_cast<PartitionType>(index);
    if (forced && partition != forced_partition) {
      continue;
    }
    if (!info.allowed.is_allowed(partition)) {
      continue;
    }
    if (!policy_allows(geometry, size, partition, depth, options)) {
      if (forced) {
        result.status = DecisionStatus::kNoValidPartition;
        result.error = "spec-forced partition conflicts with DecisionOptions";
        return result;
      }
      continue;
    }

    const double signal = forced ? 0.0 : partition_signal_cost(costs, partition);
    if (!std::isfinite(signal)) {
      continue;
    }

    if (partition == PartitionType::kNone) {
      if (!std::isfinite(costs.leaf)) {
        continue;
      }
      const double total = costs.leaf + signal;
      if (!std::isfinite(total)) {
        continue;
      }
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
    SolveResult candidate =
        solve_partition_candidate(geometry, mi_row, mi_col, size, depth,
                                  has_chroma, chroma_offset, partition, signal,
                                  cost_fn, options, children, total);
    if (candidate.status != DecisionStatus::kOk) {
      if (candidate.status == DecisionStatus::kInvalidCost) {
        saw_invalid_cost = true;
      }
      if (!candidate.error.empty()) {
        last_error = std::move(candidate.error);
      }
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
// Validation helpers
// ---------------------------------------------------------------------------

bool fail(std::string& error, std::string message) {
  error = std::move(message);
  return false;
}

struct ValidateState {
  const FrameGeometry& geometry;
  int root_mi_row = 0;
  int root_mi_col = 0;
  int sb4 = 0;
  std::vector<std::uint8_t> covered;
  int expected_cells = 0;
  std::string error;
};

std::string node_label(const PartitionNode& node) {
  std::ostringstream out;
  out << block_size_name(node.size) << "@(" << node.mi_row << ',' << node.mi_col
      << ") depth " << node.depth;
  return out.str();
}

bool validate_node(ValidateState& state, const PartitionNode& node,
                   BlockSize expected_size, int expected_row, int expected_col,
                   int expected_depth, bool expected_has_chroma,
                   bool expected_offset) {
  if (node.size != expected_size || node.mi_row != expected_row ||
      node.mi_col != expected_col) {
    std::ostringstream out;
    out << "node position/size mismatch: got " << node_label(node)
        << ", expected " << block_size_name(expected_size) << "@("
        << expected_row << ',' << expected_col << ')';
    return fail(state.error, out.str());
  }
  if (node.depth != expected_depth) {
    return fail(state.error, "node " + node_label(node) +
                                 " has wrong depth");
  }
  if (node.has_chroma != expected_has_chroma ||
      node.chroma_offset != expected_offset) {
    return fail(state.error, "node " + node_label(node) +
                                 " has wrong chroma state");
  }
  if (!is_inside(state.geometry, node.mi_row, node.mi_col)) {
    return fail(state.error, "node " + node_label(node) + " starts outside frame");
  }

  const PartitionBoundary boundary = partition_implied(
      state.geometry, node.mi_row, node.mi_col, node.size);
  const PartitionSet allowed =
      init_allowed_partitions(state.geometry, node.mi_row, node.mi_col,
                              node.size, node.has_chroma, node.chroma_offset);
  if (node.boundary.has_rows != boundary.has_rows ||
      node.boundary.has_cols != boundary.has_cols ||
      node.boundary.implied != boundary.implied ||
      node.boundary.implied_partition != boundary.implied_partition ||
      node.allowed.count != allowed.count ||
      node.allowed.allowed[0] != allowed.allowed[0] ||
      node.allowed.allowed[1] != allowed.allowed[1] ||
      node.allowed.allowed[2] != allowed.allowed[2] ||
      node.allowed.allowed[3] != allowed.allowed[3]) {
    return fail(state.error, "node " + node_label(node) +
                                 " has stale boundary/allowed fields");
  }
  if (!allowed.is_allowed(node.partition)) {
    return fail(state.error, "node " + node_label(node) +
                                 " uses a partition the spec forbids");
  }

  bool spec_forced = false;
  PartitionType spec_partition = PartitionType::kNone;
  if (boundary.implied && allowed.is_allowed(boundary.implied_partition)) {
    spec_forced = true;
    spec_partition = boundary.implied_partition;
  } else if (allowed.count == 1) {
    spec_forced = true;
    for (int index = 0; index < kPartitionTypeCount; ++index) {
      if (allowed.allowed[index]) {
        spec_partition = static_cast<PartitionType>(index);
        break;
      }
    }
  }
  if (spec_forced) {
    if (!node.forced || node.partition != spec_partition) {
      return fail(state.error, "node " + node_label(node) +
                                   " does not use the spec-forced partition");
    }
  } else if (node.forced) {
    return fail(state.error, "node " + node_label(node) +
                                 " claims to be forced but is not");
  }

  if (node.partition == PartitionType::kNone) {
    if (!node.children.empty()) {
      return fail(state.error, "PARTITION_NONE node " + node_label(node) +
                                   " has children");
    }
    // Mark coverage, clipped to the frame and to this superblock.
    const int w4 = num_4x4_blocks_wide(node.size);
    const int h4 = num_4x4_blocks_high(node.size);
    const int row_end = std::min(node.mi_row + h4, mi_rows(state.geometry));
    const int col_end = std::min(node.mi_col + w4, mi_cols(state.geometry));
    for (int row = node.mi_row; row < row_end; ++row) {
      for (int col = node.mi_col; col < col_end; ++col) {
        const int local_row = row - state.root_mi_row;
        const int local_col = col - state.root_mi_col;
        if (local_row < 0 || local_row >= state.sb4 || local_col < 0 ||
            local_col >= state.sb4) {
          return fail(state.error, "leaf " + node_label(node) +
                                       " escapes its superblock");
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
      node.has_chroma, node.chroma_offset, slots);
  if (slot_count == 0) {
    return fail(state.error, "node " + node_label(node) +
                                 " has no child slots");
  }

  int coded_slots = 0;
  for (int index = 0; index < slot_count; ++index) {
    if (child_is_coded(state.geometry, slots[index])) {
      ++coded_slots;
    }
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
    if (!child_is_coded(state.geometry, slot)) {
      continue;
    }
    if (child_index >= node.children.size()) {
      return fail(state.error, "node " + node_label(node) +
                                   " is missing a child");
    }
    if (!validate_node(state, node.children[child_index], slot.size,
                       slot.mi_row, slot.mi_col, node.depth + 1,
                       slot.has_chroma, slot.chroma_offset)) {
      return false;
    }
    ++child_index;
  }
  if (child_index != node.children.size()) {
    return fail(state.error, "node " + node_label(node) +
                                 " has an unexpected extra child");
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Block geometry / tables
// ---------------------------------------------------------------------------

bool is_valid_block_size(BlockSize size) {
  const int value = static_cast<int>(size);
  return value >= 0 && value < kBlockSizeCount;
}

int mi_width_log2(BlockSize size) {
  if (!is_valid_block_size(size)) {
    return 0;
  }
  return static_cast<int>(kMiWidthLog2[static_cast<std::size_t>(size)]);
}

int mi_height_log2(BlockSize size) {
  if (!is_valid_block_size(size)) {
    return 0;
  }
  return static_cast<int>(kMiHeightLog2[static_cast<std::size_t>(size)]);
}

int block_width(BlockSize size) {
  if (!is_valid_block_size(size)) {
    return 0;
  }
  return 4 << mi_width_log2(size);
}

int block_height(BlockSize size) {
  if (!is_valid_block_size(size)) {
    return 0;
  }
  return 4 << mi_height_log2(size);
}

int num_4x4_blocks_wide(BlockSize size) {
  if (!is_valid_block_size(size)) {
    return 0;
  }
  return 1 << mi_width_log2(size);
}

int num_4x4_blocks_high(BlockSize size) {
  if (!is_valid_block_size(size)) {
    return 0;
  }
  return 1 << mi_height_log2(size);
}

bool is_square(BlockSize size) {
  return is_valid_block_size(size) && mi_width_log2(size) == mi_height_log2(size);
}

BlockSize superblock_size(int sb_size_pixels) {
  if (sb_size_pixels == 64) {
    return BlockSize::k64x64;
  }
  if (sb_size_pixels == 128) {
    return BlockSize::k128x128;
  }
  return BlockSize::kInvalid;
}

const char* block_size_name(BlockSize size) {
  if (!is_valid_block_size(size)) {
    return "invalid";
  }
  return kBlockNames[static_cast<std::size_t>(size)];
}

const char* partition_name(PartitionType partition) {
  switch (partition) {
    case PartitionType::kNone:
      return "NONE";
    case PartitionType::kSplit:
      return "SPLIT";
    case PartitionType::kHorz:
      return "HORZ";
    case PartitionType::kVert:
      return "VERT";
  }
  return "INVALID";
}

int spec_partition_index(PartitionType partition) {
  switch (partition) {
    case PartitionType::kNone:
      return 0;
    case PartitionType::kSplit:
      return 9;
    case PartitionType::kHorz:
      return 1;
    case PartitionType::kVert:
      return 2;
  }
  return -1;
}

BlockSize partition_subsize(PartitionType partition, BlockSize size) {
  if (!is_valid_block_size(size)) {
    return BlockSize::kInvalid;
  }
  const std::size_t index = static_cast<std::size_t>(size);
  switch (partition) {
    case PartitionType::kNone:
      return size;
    case PartitionType::kHorz:
      return static_cast<BlockSize>(kHorzSubsize[index]);
    case PartitionType::kVert:
      return static_cast<BlockSize>(kVertSubsize[index]);
    case PartitionType::kSplit:
      return static_cast<BlockSize>(kSplitSubsize[index]);
  }
  return BlockSize::kInvalid;
}

RectTypeImplied rect_type_implied_by_bsize(BlockSize size) {
  RectTypeImplied out;
  switch (size) {
    case BlockSize::k4x8:
    case BlockSize::k64x128:
    case BlockSize::k128x256:
    case BlockSize::k4x16:
      out.implied = true;
      out.partition = PartitionType::kHorz;
      return out;
    case BlockSize::k8x4:
    case BlockSize::k128x64:
    case BlockSize::k256x128:
    case BlockSize::k16x4:
      out.implied = true;
      out.partition = PartitionType::kVert;
      return out;
    default:
      return out;
  }
}

int partition_size_adjust(BlockSize size) {
  if (!is_valid_block_size(size)) {
    return 0;
  }
  return static_cast<int>(kPartitionSizeAdjust[static_cast<std::size_t>(size)]);
}

int partition_size_adjust_rect_type(BlockSize size) {
  if (!is_valid_block_size(size)) {
    return 0;
  }
  return static_cast<int>(
      kPartitionSizeAdjustRect[static_cast<std::size_t>(size)]);
}

// ---------------------------------------------------------------------------
// Geometry / availability
// ---------------------------------------------------------------------------

bool validate_geometry(const FrameGeometry& geometry, std::string& error) {
  if (geometry.width <= 0 || geometry.height <= 0) {
    return fail(error, "frame width and height must be positive");
  }
  if (geometry.sb_size != 64 && geometry.sb_size != 128) {
    return fail(error, "superblock size must be 64 or 128");
  }
  if (geometry.max_pb_aspect_ratio < 1) {
    return fail(error, "max_pb_aspect_ratio must be at least 1");
  }
  if (geometry.max_pb_aspect_ratio != 2 &&
      geometry.max_pb_aspect_ratio != 4 &&
      geometry.max_pb_aspect_ratio != 8) {
    return fail(error,
                "max_pb_aspect_ratio must be 2, 4, or 8 (AV2 §5.4.3)");
  }
  if (geometry.chroma_subsampling_x != geometry.chroma_subsampling_y &&
      geometry.has_chroma) {
    return fail(error, "4:2:2 chroma is not modelled; use 4:2:0 or 4:4:4");
  }
  error.clear();
  return true;
}

int mi_cols(const FrameGeometry& geometry) {
  // §5.18.4.4 compute_image_size():
  //   MiCols = 2 * ((FrameWidth + 7) >> 3)
  // This rounds up to an even number of 4x4 units (8-pixel granularity), not
  // merely ceil(width / 4).
  return 2 * ((geometry.width + 7) >> 3);
}

int mi_rows(const FrameGeometry& geometry) {
  return 2 * ((geometry.height + 7) >> 3);
}

int sb_size_mi(const FrameGeometry& geometry) {
  return geometry.sb_size / 4;
}

int sb_cols(const FrameGeometry& geometry) {
  const int size_mi = sb_size_mi(geometry);
  return size_mi > 0 ? (mi_cols(geometry) + size_mi - 1) / size_mi : 0;
}

int sb_rows(const FrameGeometry& geometry) {
  const int size_mi = sb_size_mi(geometry);
  return size_mi > 0 ? (mi_rows(geometry) + size_mi - 1) / size_mi : 0;
}

bool is_inside(const FrameGeometry& geometry, int mi_row, int mi_col) {
  return mi_row >= 0 && mi_col >= 0 && mi_row < mi_rows(geometry) &&
         mi_col < mi_cols(geometry);
}

bool block_coded(const FrameGeometry& geometry, int mi_row, int mi_col) {
  return mi_row >= 0 && mi_col >= 0 && mi_row < mi_rows(geometry) &&
         mi_col < mi_cols(geometry);
}

BlockRect block_rect(const FrameGeometry& geometry, int mi_row, int mi_col,
                     BlockSize size) {
  BlockRect rect;
  rect.x = mi_col * 4;
  rect.y = mi_row * 4;
  rect.width = block_width(size);
  rect.height = block_height(size);
  rect.visible_width = std::max(0, std::min(rect.width, geometry.width - rect.x));
  rect.visible_height =
      std::max(0, std::min(rect.height, geometry.height - rect.y));
  return rect;
}

NeighborAvailability neighbor_availability(const FrameGeometry& geometry,
                                           int mi_row, int mi_col,
                                           BlockSize size) {
  NeighborAvailability out;
  const int w4 = num_4x4_blocks_wide(size);
  const int h4 = num_4x4_blocks_high(size);
  out.left = is_inside(geometry, mi_row, mi_col - 1);
  out.above = is_inside(geometry, mi_row - 1, mi_col);
  out.above_left = is_inside(geometry, mi_row - 1, mi_col - 1);
  out.above_right = is_inside(geometry, mi_row - 1, mi_col + w4);
  out.below_left = is_inside(geometry, mi_row + h4, mi_col - 1);
  return out;
}

// ---------------------------------------------------------------------------
// Boundary implied / allowed partitions
// ---------------------------------------------------------------------------

PartitionBoundary partition_implied_at_boundary(const FrameGeometry& geometry,
                                                int mi_row, int mi_col,
                                                BlockSize size) {
  PartitionBoundary out;
  if (!geometry_ready(geometry) || !is_valid_block_size(size) || mi_row < 0 ||
      mi_col < 0) {
    return out;
  }

  const int w4 = num_4x4_blocks_wide(size);
  const int h4 = num_4x4_blocks_high(size);
  out.has_rows = (mi_row + (h4 >> 1)) < mi_rows(geometry);
  out.has_cols = (mi_col + (w4 >> 1)) < mi_cols(geometry);
  if (out.has_rows && out.has_cols) {
    return out;
  }

  PartitionType implied = PartitionType::kNone;
  if (w4 == h4) {
    implied = out.has_rows ? PartitionType::kVert : PartitionType::kHorz;
  } else if (h4 > w4) {
    if (!out.has_rows) {
      implied = PartitionType::kHorz;
    } else {
      const bool sub_has_cols = (mi_col + (w4 >> 2)) < mi_cols(geometry);
      if (w4 >= 4 && !sub_has_cols) {
        implied = PartitionType::kHorz;
      }
    }
  } else {
    if (!out.has_cols) {
      implied = PartitionType::kVert;
    } else {
      const bool sub_has_rows = (mi_row + (h4 >> 2)) < mi_rows(geometry);
      if (h4 >= 4 && !sub_has_rows) {
        implied = PartitionType::kVert;
      }
    }
  }

  out.implied = implied != PartitionType::kNone;
  out.implied_partition = implied;
  return out;
}

PartitionBoundary partition_implied(const FrameGeometry& geometry, int mi_row,
                                    int mi_col, BlockSize size) {
  PartitionBoundary out = partition_implied_at_boundary(geometry, mi_row,
                                                        mi_col, size);
  if (!is_valid_block_size(size)) {
    return out;
  }
  const int value = static_cast<int>(size);
  // §5.20.3.2: thin blocks and 4x4 always infer PARTITION_NONE.
  if (size == BlockSize::k4x4 || value >= static_cast<int>(BlockSize::k4x32)) {
    out.implied = true;
    out.implied_partition = PartitionType::kNone;
  }
  return out;
}

bool PartitionSet::is_allowed(PartitionType partition) const {
  const int index = static_cast<int>(partition);
  if (index < 0 || index >= kPartitionTypeCount) {
    return false;
  }
  return allowed[index];
}

bool is_chroma_offset_for_subsize(const FrameGeometry& geometry,
                                  BlockSize subsize) {
  if (!is_valid_block_size(subsize)) {
    return false;
  }
  if (geometry.chroma_subsampling_y && mi_height_log2(subsize) == 0) {
    return true;
  }
  if (geometry.chroma_subsampling_x && mi_width_log2(subsize) == 0) {
    return true;
  }
  return false;
}

bool is_chroma_offset_for_partition(const FrameGeometry& geometry,
                                    PartitionType partition, BlockSize size) {
  if (partition == PartitionType::kNone) {
    return is_chroma_offset_for_subsize(geometry, size);
  }
  return is_chroma_offset_for_subsize(geometry,
                                      partition_subsize(partition, size));
}

BlockSize chroma_plane_residual_size(const FrameGeometry& geometry,
                                     BlockSize luma_size) {
  if (!is_valid_block_size(luma_size)) {
    return BlockSize::kInvalid;
  }
  if (!geometry.chroma_subsampling_x && !geometry.chroma_subsampling_y) {
    return luma_size;
  }
  if (geometry.chroma_subsampling_x && geometry.chroma_subsampling_y) {
    return static_cast<BlockSize>(
        kChroma420Subsize[static_cast<std::size_t>(luma_size)]);
  }
  return BlockSize::kInvalid;  // 4:2:2 not modelled
}

bool is_partition_allowed(const FrameGeometry& geometry, int mi_row,
                          int mi_col, BlockSize size,
                          PartitionType partition, bool has_chroma,
                          bool chroma_offset) {
  if (!geometry_ready(geometry) || !is_valid_block_size(size) ||
      mi_row < 0 || mi_col < 0) {
    return false;
  }
  const BlockSize subsize = partition_subsize(partition, size);
  if (!is_valid_block_size(subsize)) {
    return false;
  }

  // rect_type_implied_by_bsize() (§5.20.3.2).
  const RectTypeImplied rect = rect_type_implied_by_bsize(size);
  if (rect.implied) {
    if (rect.partition == PartitionType::kVert &&
        (partition == PartitionType::kHorz)) {
      return false;
    }
    if (rect.partition == PartitionType::kHorz &&
        (partition == PartitionType::kVert)) {
      return false;
    }
  }

  // Aspect-ratio guard.  MaxPbAspectRatio defaults to 8.
  const int bw = block_width(subsize);
  const int bh = block_height(subsize);
  const int aspect = geometry.max_pb_aspect_ratio;
  if (bw > bh * aspect || bh > bw * aspect) {
    if (partition == PartitionType::kNone) {
      return false;
    }
    if (bw >= bh * 8 || bh >= bw * 8) {
      return false;
    }
  }

  bool local_offset = chroma_offset;
  if (has_chroma && !local_offset) {
    local_offset = is_chroma_offset_for_partition(geometry, partition, size);
  }
  if (has_chroma && !local_offset) {
    if (chroma_plane_residual_size(geometry, subsize) == BlockSize::kInvalid) {
      return false;
    }
  }
  // check_chroma(bSize) is false because this module only models SHARED_PART.

  const int w4 = num_4x4_blocks_wide(size);
  const int h4 = num_4x4_blocks_high(size);
  const int half_w = w4 >> 1;
  const int half_h = h4 >> 1;

  if (partition == PartitionType::kNone) {
    const bool has_rows = (mi_row + half_h) < mi_rows(geometry);
    const bool has_cols = (mi_col + half_w) < mi_cols(geometry);
    if (!has_rows || !has_cols) {
      return false;
    }
  }

  if (has_chroma && local_offset) {
    if (partition == PartitionType::kHorz) {
      return block_coded(geometry, mi_row + half_h, mi_col);
    }
    if (partition == PartitionType::kVert) {
      return block_coded(geometry, mi_row, mi_col + half_w);
    }
  }
  return true;
}

PartitionSet init_allowed_partitions(const FrameGeometry& geometry,
                                     int mi_row, int mi_col, BlockSize size,
                                     bool has_chroma, bool chroma_offset) {
  PartitionSet set;
  for (int index = 0; index < kPartitionTypeCount; ++index) {
    const PartitionType partition = static_cast<PartitionType>(index);
    const bool good = is_partition_allowed(geometry, mi_row, mi_col, size,
                                           partition, has_chroma, chroma_offset);
    set.allowed[index] = good;
    if (good) {
      ++set.count;
    }
  }
  if (set.count == 0) {
    set.allowed[static_cast<int>(PartitionType::kNone)] = true;
    set.count = 1;
  }
  return set;
}

// ---------------------------------------------------------------------------
// Neighbour size grid / CDF contexts
// ---------------------------------------------------------------------------

MiSizeGrid::MiSizeGrid(const FrameGeometry& geometry) { reset(geometry); }

void MiSizeGrid::reset(const FrameGeometry& geometry) {
  rows_ = mi_rows(geometry);
  cols_ = mi_cols(geometry);
  const std::size_t count =
      static_cast<std::size_t>(std::max(0, rows_)) *
      static_cast<std::size_t>(std::max(0, cols_));
  cells_.assign(count, BlockSize::kInvalid);
}

bool MiSizeGrid::in_bounds(int mi_row, int mi_col) const {
  return mi_row >= 0 && mi_col >= 0 && mi_row < rows_ && mi_col < cols_;
}

BlockSize MiSizeGrid::at(int mi_row, int mi_col) const {
  if (!in_bounds(mi_row, mi_col)) {
    return BlockSize::kInvalid;
  }
  return cells_[static_cast<std::size_t>(mi_row) *
                   static_cast<std::size_t>(cols_) +
               static_cast<std::size_t>(mi_col)];
}

void MiSizeGrid::set(int mi_row, int mi_col, BlockSize size) {
  if (!in_bounds(mi_row, mi_col)) {
    return;
  }
  cells_[static_cast<std::size_t>(mi_row) *
             static_cast<std::size_t>(cols_) +
         static_cast<std::size_t>(mi_col)] = size;
}

void MiSizeGrid::fill_block(int mi_row, int mi_col, BlockSize size) {
  if (!is_valid_block_size(size)) {
    return;
  }
  const int w4 = num_4x4_blocks_wide(size);
  const int h4 = num_4x4_blocks_high(size);
  for (int row = mi_row; row < mi_row + h4; ++row) {
    for (int col = mi_col; col < mi_col + w4; ++col) {
      set(row, col, size);
    }
  }
}

PartitionContexts partition_contexts(const FrameGeometry& geometry,
                                     int mi_row, int mi_col, BlockSize size,
                                     const MiSizeGrid& sizes) {
  PartitionContexts out;
  if (!is_valid_block_size(size)) {
    return out;
  }

  const int bsw = std::max(mi_width_log2(size), 1);
  const int bsh = std::max(mi_height_log2(size), 1);

  const BlockSize left = sizes.at(mi_row, mi_col - 1);
  const BlockSize above = sizes.at(mi_row - 1, mi_col);
  const int left_h = log2_or_sentinel(left, /*width=*/false);
  const int above_w = log2_or_sentinel(above, /*width=*/true);

  const int ctx1 = left_h < bsh ? 1 : 0;
  const int ctx2 = above_w < bsw ? 1 : 0;
  out.do_split = partition_size_adjust(size) * 4 + ctx1 * 2 + ctx2;
  out.rect_type = partition_size_adjust_rect_type(size) * 4 + ctx1 * 2 + ctx2;

  // do_square_split uses raw Mi log2 values and explicit availability.
  const int raw_bsw = mi_width_log2(size);
  const int raw_bsh = mi_height_log2(size);
  const bool avail_left = is_inside(geometry, mi_row, mi_col - 1);
  const bool avail_above = is_inside(geometry, mi_row - 1, mi_col);
  const bool left_smaller =
      avail_left && (log2_or_sentinel(left, /*width=*/false) < raw_bsh);
  const bool above_smaller =
      avail_above && (log2_or_sentinel(above, /*width=*/true) < raw_bsw);
  out.do_square_split = 0;
  if (size == BlockSize::k256x256) {
    out.do_square_split += 4;
  }
  if (left_smaller) {
    out.do_square_split += 2;
  }
  if (above_smaller) {
    out.do_square_split += 1;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Decision API
// ---------------------------------------------------------------------------

const char* decision_status_name(DecisionStatus status) {
  switch (status) {
    case DecisionStatus::kOk:
      return "ok";
    case DecisionStatus::kInvalidGeometry:
      return "invalid-geometry";
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
                                    const DecisionOptions& options) {
  PartitionDecision decision;
  std::string geometry_error;
  if (!validate_geometry(geometry, geometry_error)) {
    decision.status = DecisionStatus::kInvalidGeometry;
    decision.error = std::move(geometry_error);
    return decision;
  }
  const int size_mi = sb_size_mi(geometry);
  if (size_mi <= 0 || sb_mi_row < 0 || sb_mi_col < 0 ||
      sb_mi_row >= mi_rows(geometry) || sb_mi_col >= mi_cols(geometry) ||
      (sb_mi_row % size_mi) != 0 || (sb_mi_col % size_mi) != 0) {
    decision.status = DecisionStatus::kInvalidGeometry;
    decision.error = "superblock origin is outside or unaligned";
    return decision;
  }

  const BlockSize root_size = superblock_size(geometry.sb_size);
  SolveResult result =
      solve_node(geometry, sb_mi_row, sb_mi_col, root_size, /*depth=*/0,
                 geometry.has_chroma, /*chroma_offset=*/false, costs, options,
                 decision.root);
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
                           const CostFunction& costs,
                           const DecisionOptions& options) {
  FrameDecision frame;
  std::string geometry_error;
  if (!validate_geometry(geometry, geometry_error)) {
    frame.status = DecisionStatus::kInvalidGeometry;
    frame.error = std::move(geometry_error);
    return frame;
  }
  if (!costs) {
    frame.status = DecisionStatus::kInvalidGeometry;
    frame.error = "cost function is empty";
    return frame;
  }

  const int size_mi = sb_size_mi(geometry);
  const int rows_mi = mi_rows(geometry);
  const int cols_mi = mi_cols(geometry);
  for (int row = 0; row < rows_mi; row += size_mi) {
    for (int col = 0; col < cols_mi; col += size_mi) {
      PartitionDecision decision =
          decide_superblock(geometry, row, col, costs, options);
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
  if (node.partition == PartitionType::kNone) {
    return 1;
  }
  int count = 0;
  for (const PartitionNode& child : node.children) {
    count += count_leaves(child);
  }
  return count;
}

bool validate_tree(const FrameGeometry& geometry, const PartitionNode& root,
                   std::string& error) {
  std::string geometry_error;
  if (!validate_geometry(geometry, geometry_error)) {
    return fail(error, std::move(geometry_error));
  }
  if (root.size != superblock_size(geometry.sb_size)) {
    return fail(error, "root is not a superblock-sized node");
  }
  const int size_mi = sb_size_mi(geometry);
  if (root.mi_row < 0 || root.mi_col < 0 ||
      (root.mi_row % size_mi) != 0 || (root.mi_col % size_mi) != 0) {
    return fail(error, "root origin is not superblock-aligned");
  }
  if (!is_inside(geometry, root.mi_row, root.mi_col)) {
    return fail(error, "root origin is outside the frame");
  }

  ValidateState state{geometry, root.mi_row, root.mi_col, size_mi, {}, 0, {}};
  state.covered.assign(
      static_cast<std::size_t>(size_mi) * static_cast<std::size_t>(size_mi),
      0);

  const int row_end = std::min(root.mi_row + size_mi, mi_rows(geometry));
  const int col_end = std::min(root.mi_col + size_mi, mi_cols(geometry));
  state.expected_cells = (row_end - root.mi_row) * (col_end - root.mi_col);

  if (!validate_node(state, root, root.size, root.mi_row, root.mi_col, 0,
                     geometry.has_chroma, /*expected_offset=*/false)) {
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

bool validate_frame(const FrameGeometry& geometry, const FrameDecision& frame,
                    std::string& error) {
  std::string geometry_error;
  if (!validate_geometry(geometry, geometry_error)) {
    return fail(error, std::move(geometry_error));
  }
  if (frame.status != DecisionStatus::kOk) {
    return fail(error, "frame decision status is not ok");
  }
  const std::size_t expected = static_cast<std::size_t>(sb_rows(geometry)) *
                               static_cast<std::size_t>(sb_cols(geometry));
  if (frame.superblocks.size() != expected) {
    std::ostringstream out;
    out << "frame has " << frame.superblocks.size() << " superblocks, expected "
        << expected;
    return fail(error, out.str());
  }

  const int size_mi = sb_size_mi(geometry);
  std::size_t index = 0;
  for (int row = 0; row < mi_rows(geometry); row += size_mi) {
    for (int col = 0; col < mi_cols(geometry); col += size_mi) {
      const PartitionNode& root = frame.superblocks[index];
      if (root.mi_row != row || root.mi_col != col) {
        std::ostringstream out;
        out << "superblock " << index << " is at (" << root.mi_row << ','
            << root.mi_col << "), expected (" << row << ',' << col << ')';
        return fail(error, out.str());
      }
      std::string tree_error;
      if (!validate_tree(geometry, root, tree_error)) {
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

}  // namespace av2enc::partition
