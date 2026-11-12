// AV2 semi-decoupled partitioning (SDP), CPU reference implementation.
// See sdp.h for the clause map and scope.

#include "sdp.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

namespace av2enc::sdp {
namespace {

bool geometry_ready(const FrameGeometry& geometry) {
  return geometry.width > 0 && geometry.height > 0 &&
         (geometry.sb_size == 64 || geometry.sb_size == 128) &&
         geometry.max_pb_aspect_ratio >= 1;
}

bool fail(std::string& error, std::string message) {
  error = std::move(message);
  return false;
}

bool cost_is_sane(double value) {
  return !std::isnan(value) && value >= 0.0;
}

bool all_costs_sane(const NodeCosts& costs) {
  const double values[] = {
      costs.leaf,           costs.none_signal,   costs.split_signal,
      costs.horz_signal,    costs.vert_signal,   costs.horz3_signal,
      costs.vert3_signal,   costs.horz4a_signal, costs.horz4b_signal,
      costs.vert4a_signal,  costs.vert4b_signal, costs.region_type[0],
      costs.region_type[1],
  };
  for (double value : values) {
    if (!cost_is_sane(value)) return false;
  }
  return true;
}

int num_planes(const FrameGeometry& geometry) {
  return geometry.has_chroma ? 3 : 1;
}

// Converts the rect-only enum from src/encoder/partition to the 10-value enum.
PartitionType from_base(partition::PartitionType value) {
  switch (value) {
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

bool sdp_is_three_way(PartitionType p) {
  return p == PartitionType::kHorz3 || p == PartitionType::kVert3;
}

bool sdp_is_uneven_4way(PartitionType p) {
  return p == PartitionType::kHorz4A || p == PartitionType::kHorz4B ||
         p == PartitionType::kVert4A || p == PartitionType::kVert4B;
}

bool rect_is_horizontal(PartitionType p) {
  return p == PartitionType::kHorz || p == PartitionType::kHorz3 ||
         p == PartitionType::kHorz4A || p == PartitionType::kHorz4B;
}

bool rect_is_vertical(PartitionType p) {
  return p == PartitionType::kVert || p == PartitionType::kVert3 ||
         p == PartitionType::kVert4A || p == PartitionType::kVert4B;
}

bool policy_allows(BlockSize size, PartitionType partition, int depth,
                   const DecisionOptions& options) {
  if (partition == PartitionType::kNone) return true;
  if (partition == PartitionType::kSplit && !options.allow_split) return false;
  if ((partition == PartitionType::kHorz || partition == PartitionType::kVert) &&
      !options.allow_rect) {
    return false;
  }
  if (sdp_is_three_way(partition) && !options.allow_3way) return false;
  if (sdp_is_uneven_4way(partition) && !options.allow_uneven4) return false;
  if (depth + 1 > options.max_depth) return false;
  const BlockSize subsize = partition_ext::partition_subsize(partition, size);
  if (!partition::is_valid_block_size(subsize)) return false;
  const int floor_px = std::max(options.min_block_size_px, kMinBlockPixels);
  return partition::block_width(subsize) >= floor_px &&
         partition::block_height(subsize) >= floor_px;
}

}  // namespace

// ---------------------------------------------------------------------------
// Tools / geometry
// ---------------------------------------------------------------------------

const char* tree_type_name(TreeType type) {
  switch (type) {
    case TreeType::kSharedPart:
      return "SHARED_PART";
    case TreeType::kLumaPart:
      return "LUMA_PART";
    case TreeType::kChromaPart:
      return "CHROMA_PART";
  }
  return "?";
}

bool is_valid_tree_type(TreeType type) {
  return type == TreeType::kSharedPart || type == TreeType::kLumaPart ||
         type == TreeType::kChromaPart;
}

int plane_start(TreeType type) {
  return type == TreeType::kChromaPart ? 1 : 0;
}

const char* region_type_name(RegionType type) {
  return type == RegionType::kIntraRegion ? "INTRA_REGION" : "MIXED_REGION";
}

bool sdp_validate_geometry(const FrameGeometry& geometry, std::string& error) {
  std::string base_error;
  if (!partition::validate_geometry(geometry, base_error)) {
    return fail(error, base_error);
  }
  if (geometry.sb_size != 64 && geometry.sb_size != 128) {
    return fail(error, "SDP reference supports 64x64 and 128x128 superblocks");
  }
  error.clear();
  return true;
}

bool validate_tools(const Tools& tools, std::string& error) {
  if (tools.enable_extended_sdp && !tools.enable_sdp) {
    return fail(error, "enable_extended_sdp requires enable_sdp");
  }
  if (tools.enable_extended_sdp && tools.single_picture_header_flag) {
    return fail(error,
                "enable_extended_sdp is inferred 0 when "
                "single_picture_header_flag is 1");
  }
  if (tools.enable_uneven_4way_partitions && !tools.enable_ext_partitions) {
    return fail(error,
                "enable_uneven_4way_partitions requires enable_ext_partitions");
  }
  error.clear();
  return true;
}

bool bru_mode_active(const Tools& tools) {
  // §5.18.4.2: IsBridge -> INACTIVE; else use_bru ? INACTIVE : ACTIVE.
  if (tools.is_bridge) return false;
  return !tools.use_bru;
}

bool is_bsize_allowed_for_extended_sdp(BlockSize size,
                                       PartitionType partition) {
  const int bw = partition::block_width(size);
  const int bh = partition::block_height(size);
  return bw <= kInterSdpMaxBlockSize && bh <= kInterSdpMaxBlockSize &&
         bw >= 8 && bh >= 8 && partition < PartitionType::kHorz4A &&
         partition != PartitionType::kNone;
}

// ---------------------------------------------------------------------------
// Implied / rect / allowed
// ---------------------------------------------------------------------------

Boundary sdp_implied_at_boundary(const FrameGeometry& geometry,
                                       int mi_row, int mi_col,
                                       BlockSize size) {
  Boundary out;
  const partition::PartitionBoundary base =
      partition::partition_implied_at_boundary(geometry, mi_row, mi_col, size);
  out.has_rows = base.has_rows;
  out.has_cols = base.has_cols;
  out.implied = base.implied;
  out.implied_partition = from_base(base.implied_partition);
  return out;
}

Boundary sdp_implied(const FrameGeometry& geometry, int mi_row,
                           int mi_col, BlockSize size, TreeType tree_type,
                           bool chroma_partition_known,
                           PartitionType luma_partition) {
  if (!partition::is_valid_block_size(size)) {
    return Boundary{};
  }
  Boundary out = sdp_implied_at_boundary(geometry, mi_row, mi_col, size);
  const int value = static_cast<int>(size);
  if (size == BlockSize::k4x4 ||
      value >= static_cast<int>(BlockSize::k4x32)) {
    out.implied = true;
    out.implied_partition = PartitionType::kNone;
    return out;
  }
  if (tree_type == TreeType::kChromaPart && size == BlockSize::k8x8) {
    out.implied = true;
    out.implied_partition = PartitionType::kNone;
    return out;
  }
  if (tree_type == TreeType::kChromaPart && size == BlockSize::k64x64 &&
      chroma_partition_known) {
    out.implied = true;
    out.implied_partition = luma_partition;
    return out;
  }
  return out;
}

partition::RectTypeImplied sdp_rect_implied(BlockSize size,
                                                      TreeType tree_type) {
  partition::RectTypeImplied out =
      partition::rect_type_implied_by_bsize(size);
  if (!out.implied && tree_type == TreeType::kChromaPart) {
    if (size == BlockSize::k8x16 || size == BlockSize::k8x32) {
      out.implied = true;
      out.partition = partition::PartitionType::kHorz;
    } else if (size == BlockSize::k16x8 || size == BlockSize::k32x8) {
      out.implied = true;
      out.partition = partition::PartitionType::kVert;
    }
  }
  return out;
}

BlockSize sdp_subsize(PartitionType partition, BlockSize size) {
  return partition_ext::partition_subsize(partition, size);
}

BlockSize sdp_midsize(BlockSize size) {
  return partition_ext::h_partition_midsize(size);
}

int sdp_size_adjust(BlockSize size) {
  return partition_ext::size_adjust(size);
}

int sdp_size_adjust_rect(BlockSize size) {
  return partition_ext::size_adjust_rect_type(size);
}

bool sdp_chroma_offset_for_subsize(const FrameGeometry& geometry,
                                  BlockSize subsize) {
  return partition::is_chroma_offset_for_subsize(geometry, subsize);
}

bool sdp_chroma_offset_for_partition(const FrameGeometry& geometry,
                                    PartitionType partition, BlockSize size) {
  return partition_ext::chroma_offset_for_partition(geometry, partition, size);
}

bool sdp_ext_allowed(const FrameGeometry& geometry, BlockSize size,
                              bool horizontal, TreeType tree_type,
                              const Tools& tools) {
  (void)geometry;
  if (!tools.enable_ext_partitions) return false;
  if (tree_type != TreeType::kChromaPart) return true;
  const int w = partition::block_width(size);
  const int h = partition::block_height(size);
  if (horizontal) return h > 16 && w > 8;
  return w > 16 && h > 8;
}

bool sdp_uneven4_allowed(const FrameGeometry& geometry,
                                      BlockSize size, bool horizontal,
                                      TreeType tree_type,
                                      const Tools& tools) {
  (void)geometry;
  if (!tools.enable_uneven_4way_partitions) return false;
  if (tree_type != TreeType::kChromaPart) return true;
  if (horizontal) return partition::block_height(size) == 64;
  return partition::block_width(size) == 64;
}

bool sdp_check_chroma(const FrameGeometry& geometry, BlockSize size,
                  TreeType tree_type) {
  if (partition::chroma_plane_residual_size(geometry, size) ==
      BlockSize::kInvalid) {
    return false;
  }
  return tree_type == TreeType::kLumaPart &&
         partition::block_width(size) >= 64 &&
         partition::block_height(size) >= 64;
}

bool sdp_partition_allowed(const FrameGeometry& geometry, int mi_row, int mi_col,
                          BlockSize size, PartitionType partition,
                          TreeType tree_type, bool has_chroma,
                          bool chroma_offset, RegionType region_type,
                          const Tools& tools) {
  if (!geometry_ready(geometry) || !partition::is_valid_block_size(size) ||
      mi_row < 0 || mi_col < 0 || !is_valid_tree_type(tree_type)) {
    return false;
  }
  const BlockSize subsize = sdp_subsize(partition, size);
  if (!partition::is_valid_block_size(subsize)) {
    return false;
  }
  if (!tools.frame_is_intra && region_type == RegionType::kMixedRegion &&
      subsize == BlockSize::k4x4) {
    return false;
  }

  const partition::RectTypeImplied implied =
      sdp_rect_implied(size, tree_type);
  if (implied.implied) {
    if (implied.partition == partition::PartitionType::kVert &&
        rect_is_horizontal(partition)) {
      return false;
    }
    if (implied.partition == partition::PartitionType::kHorz &&
        rect_is_vertical(partition)) {
      return false;
    }
  }

  const int bw = partition::block_width(subsize);
  const int bh = partition::block_height(subsize);
  const int aspect = geometry.max_pb_aspect_ratio;
  if (bw > bh * aspect || bh > bw * aspect) {
    if (partition == PartitionType::kNone) return false;
    if (bw >= bh * 8 || bh >= bw * 8) return false;
  }

  bool local_offset = chroma_offset;
  if (has_chroma && tree_type != TreeType::kChromaPart) {
    if (!local_offset) {
      local_offset = sdp_chroma_offset_for_partition(geometry, partition, size);
    }
  }
  const bool need_residual_check =
      (has_chroma && !local_offset && tree_type != TreeType::kLumaPart) ||
      sdp_check_chroma(geometry, size, tree_type);
  if (need_residual_check) {
    if (partition::chroma_plane_residual_size(geometry, subsize) ==
        BlockSize::kInvalid) {
      return false;
    }
  }

  switch (partition) {
    case PartitionType::kHorz3:
      if (!sdp_ext_allowed(geometry, size, /*horizontal=*/true,
                                    tree_type, tools)) {
        return false;
      }
      break;
    case PartitionType::kVert3:
      if (!sdp_ext_allowed(geometry, size, /*horizontal=*/false,
                                    tree_type, tools)) {
        return false;
      }
      break;
    case PartitionType::kHorz4A:
    case PartitionType::kHorz4B:
      if (!sdp_ext_allowed(geometry, size, true, tree_type, tools) ||
          !sdp_uneven4_allowed(geometry, size, true, tree_type,
                                            tools)) {
        return false;
      }
      break;
    case PartitionType::kVert4A:
    case PartitionType::kVert4B:
      if (!sdp_ext_allowed(geometry, size, false, tree_type, tools) ||
          !sdp_uneven4_allowed(geometry, size, false, tree_type,
                                            tools)) {
        return false;
      }
      break;
    case PartitionType::kNone: {
      const int w4 = partition::num_4x4_blocks_wide(size);
      const int h4 = partition::num_4x4_blocks_high(size);
      const bool has_rows = (mi_row + (h4 >> 1)) < partition::mi_rows(geometry);
      const bool has_cols = (mi_col + (w4 >> 1)) < partition::mi_cols(geometry);
      if ((tree_type != TreeType::kChromaPart || size != BlockSize::k8x8) &&
          (!has_rows || !has_cols)) {
        return false;
      }
      break;
    }
    default:
      break;
  }

  if (has_chroma && tree_type != TreeType::kLumaPart &&
      num_planes(geometry) > 1) {
    const int w4 = partition::num_4x4_blocks_wide(size);
    const int h4 = partition::num_4x4_blocks_high(size);
    const int half_w = w4 >> 1;
    const int half_h = h4 >> 1;
    if (local_offset) {
      switch (partition) {
        case PartitionType::kHorz:
          return partition::block_coded(geometry, mi_row + half_h, mi_col);
        case PartitionType::kVert:
          return partition::block_coded(geometry, mi_row, mi_col + half_w);
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
  }
  return true;
}

PartitionSet sdp_init_allowed(const FrameGeometry& geometry, int mi_row,
                                     int mi_col, BlockSize size,
                                     TreeType tree_type, bool has_chroma,
                                     bool chroma_offset,
                                     RegionType region_type,
                                     const Tools& tools) {
  PartitionSet set;
  for (int i = 0; i < kPartitionCount; ++i) {
    const auto p = static_cast<PartitionType>(i);
    const bool good =
        sdp_partition_allowed(geometry, mi_row, mi_col, size, p, tree_type,
                             has_chroma, chroma_offset, region_type, tools);
    set.allowed[i] = good;
    if (good) ++set.count;
  }
  if (set.count == 0) {
    set.allowed[static_cast<int>(PartitionType::kNone)] = true;
    set.count = 1;
  }
  return set;
}

// ---------------------------------------------------------------------------
// Chroma state / SDP bookkeeping
// ---------------------------------------------------------------------------

ChromaState sdp_resolve_chroma(const FrameGeometry& geometry, int mi_row,
                           int mi_col, BlockSize size,
                           PartitionType partition, bool has_chroma,
                           bool chroma_offset, const ChromaBlock& inherited) {
  ChromaState state;
  state.block = inherited;
  state.chroma_offset = chroma_offset;
  if (!chroma_offset && has_chroma) {
    state.block.mi_row = mi_row;
    state.block.mi_col = mi_col;
    state.block.size = size;
    state.chroma_offset =
        sdp_chroma_offset_for_partition(geometry, partition, size);
  }
  return state;
}

bool sdp_is_horizontal(PartitionType partition) { return rect_is_horizontal(partition); }
bool sdp_is_vertical(PartitionType partition) { return rect_is_vertical(partition); }

bool chroma_follows_luma(PartitionType root_partition,
                         const std::vector<PartitionType>& child_partitions) {
  const bool top_horz = root_partition == PartitionType::kHorz ||
                        root_partition == PartitionType::kHorz3;
  const bool top_vert = root_partition == PartitionType::kVert ||
                        root_partition == PartitionType::kVert3;
  bool follows = root_partition == PartitionType::kNone || top_horz || top_vert;
  if (!follows) return false;
  for (PartitionType child : child_partitions) {
    if (child == PartitionType::kNone || (top_horz && rect_is_horizontal(child)) ||
        (top_vert && rect_is_vertical(child))) {
      follows = false;
      break;
    }
  }
  return follows;
}

bool cfl_allowed_in_sdp(PartitionType luma_partition, bool chroma_follows,
                        PartitionType chroma_partition) {
  if (chroma_follows || chroma_partition == PartitionType::kNone) return true;
  const bool top_horz = luma_partition == PartitionType::kHorz ||
                        luma_partition == PartitionType::kHorz3;
  const bool top_vert = luma_partition == PartitionType::kVert ||
                        luma_partition == PartitionType::kVert3;
  const bool top_uneven_horz = luma_partition == PartitionType::kHorz4A ||
                              luma_partition == PartitionType::kHorz4B;
  const bool top_uneven_vert = luma_partition == PartitionType::kVert4A ||
                              luma_partition == PartitionType::kVert4B;
  if ((top_horz || top_uneven_horz) && rect_is_horizontal(chroma_partition)) {
    return true;
  }
  if ((top_vert || top_uneven_vert) && rect_is_vertical(chroma_partition)) {
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Decision solver (greedy: luma tree first, then chroma tree)
// ---------------------------------------------------------------------------

namespace {

struct SolveResult {
  DecisionStatus status = DecisionStatus::kOk;
  double cost = 0.0;
  std::string error;
};

struct Solver {
  const FrameGeometry& geometry;
  const Tools& tools;
  const CostFunction& cost_fn;
  const DecisionOptions& options;

  SolveResult solve(int mi_row, int mi_col, BlockSize size, int depth,
                    TreeType tree, bool has_chroma, bool chroma_offset,
                    const ChromaBlock& inherited, bool ext_allowed,
                    RegionType region_type, bool chroma_known,
                    PartitionType luma_partition, Node& node) const;

  SolveResult solve_sdp_split(int mi_row, int mi_col, int depth,
                              bool has_chroma, bool chroma_offset,
                              const ChromaBlock& inherited,
                              Node& node) const;

  SolveResult solve_partition_children(
      int mi_row, int mi_col, BlockSize size, int depth, PartitionType partition,
      TreeType child_tree, bool has_chroma, bool chroma_offset,
      const ChromaBlock& inherited, bool next_ext_allowed,
      RegionType child_region, std::vector<Node>& children,
      double& total) const;

  SolveResult solve_normal(int mi_row, int mi_col, BlockSize size, int depth,
                           TreeType tree, bool has_chroma, bool chroma_offset,
                           const ChromaBlock& inherited, bool ext_allowed,
                           RegionType region_type, bool chroma_known,
                           PartitionType luma_partition, Node& node) const;
};

bool is_sdp_intra_split(const FrameGeometry& geometry, const Tools& tools,
                        const DecisionOptions& options, TreeType tree,
                        BlockSize size) {
  return options.allow_sdp && tools.enable_sdp && tools.frame_is_intra &&
         geometry.has_chroma && tree == TreeType::kSharedPart &&
         size == BlockSize::k64x64;
}

SolveResult Solver::solve_partition_children(
    int mi_row, int mi_col, BlockSize size, int depth, PartitionType partition,
    TreeType child_tree, bool has_chroma, bool chroma_offset,
    const ChromaBlock& inherited, bool next_ext_allowed,
    RegionType child_region, std::vector<Node>& children,
    double& total) const {
  SolveResult result;
  total = 0.0;
  partition_ext::ChildSlot slots[4];
  const int count = partition_ext::enumerate_children(
      geometry, mi_row, mi_col, size, partition, has_chroma, chroma_offset,
      inherited, slots);
  if (count <= 0) {
    result.status = DecisionStatus::kNoValidPartition;
    result.error = "partition has no child slots";
    return result;
  }
  children.clear();
  children.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const partition_ext::ChildSlot& slot = slots[i];
    if (!partition_ext::child_is_coded(geometry, slot)) continue;
    Node child;
    SolveResult child_result =
        solve(slot.mi_row, slot.mi_col, slot.size, depth + 1, child_tree,
              slot.has_chroma, slot.chroma_offset, slot.chroma, next_ext_allowed,
              child_region, /*chroma_known=*/false, PartitionType::kNone, child);
    if (child_result.status != DecisionStatus::kOk) {
      return child_result;
    }
    total += child_result.cost;
    children.push_back(std::move(child));
  }
  if (!std::isfinite(total)) {
    result.status = DecisionStatus::kNoValidPartition;
    result.error = "candidate subtree cost is not finite";
  }
  return result;
}

SolveResult Solver::solve_sdp_split(int mi_row, int mi_col, int depth,
                                    bool has_chroma, bool chroma_offset,
                                    const ChromaBlock& inherited,
                                    Node& node) const {
  SolveResult result;
  Node luma;
  SolveResult luma_result =
      solve(mi_row, mi_col, BlockSize::k64x64, depth + 1, TreeType::kLumaPart,
            /*has_chroma=*/true, /*chroma_offset=*/false, inherited,
            /*ext_allowed=*/false, RegionType::kMixedRegion,
            /*chroma_known=*/false, PartitionType::kNone, luma);
  if (luma_result.status != DecisionStatus::kOk) return luma_result;

  std::vector<PartitionType> direct_children;
  direct_children.reserve(luma.children.size());
  for (const Node& child : luma.children) {
    direct_children.push_back(child.partition);
  }
  const bool follows = chroma_follows_luma(luma.partition, direct_children);
  luma.chroma_follows_luma = follows;

  Node chroma;
  SolveResult chroma_result =
      solve(mi_row, mi_col, BlockSize::k64x64, depth + 1,
            TreeType::kChromaPart, /*has_chroma=*/true,
            /*chroma_offset=*/false, inherited, /*ext_allowed=*/false,
            RegionType::kMixedRegion, /*chroma_known=*/follows, luma.partition,
            chroma);
  if (chroma_result.status != DecisionStatus::kOk) return chroma_result;
  chroma.chroma_partition_known = follows;
  chroma.implied_luma_partition = luma.partition;
  chroma.cfl_allowed_in_sdp =
      cfl_allowed_in_sdp(luma.partition, follows, chroma.partition);

  node.mi_row = mi_row;
  node.mi_col = mi_col;
  node.size = BlockSize::k64x64;
  node.depth = depth;
  node.tree_type = TreeType::kSharedPart;
  node.partition = PartitionType::kNone;
  node.sdp_split = true;
  node.has_chroma = has_chroma;
  node.chroma_offset = chroma_offset;
  node.chroma = inherited;
  node.extended_sdp_allowed = false;
  node.boundary = sdp_implied(geometry, mi_row, mi_col,
                                    BlockSize::k64x64, TreeType::kSharedPart);
  node.allowed = PartitionSet{};
  node.forced = true;
  node.codes_chroma = false;
  node.cost = luma_result.cost + chroma_result.cost;
  node.children.clear();
  node.children.push_back(std::move(luma));
  node.children.push_back(std::move(chroma));
  result.cost = node.cost;
  return result;
}

SolveResult Solver::solve_normal(int mi_row, int mi_col, BlockSize size,
                                 int depth, TreeType tree, bool has_chroma,
                                 bool chroma_offset,
                                 const ChromaBlock& inherited,
                                 bool ext_allowed, RegionType region_type,
                                 bool chroma_known,
                                 PartitionType luma_partition,
                                 Node& node) const {
  SolveResult result;
  const bool bru_active = bru_mode_active(tools);

  NodeInfo info;
  info.mi_row = mi_row;
  info.mi_col = mi_col;
  info.size = size;
  info.depth = depth;
  info.tree_type = tree;
  info.has_chroma = has_chroma;
  info.chroma_offset = chroma_offset;
  info.chroma = inherited;
  info.extended_sdp_allowed = ext_allowed;
  info.chroma_partition_known = chroma_known;
  info.implied_luma_partition = luma_partition;
  info.boundary = sdp_implied(geometry, mi_row, mi_col, size, tree,
                                    chroma_known, luma_partition);
  info.allowed = sdp_init_allowed(geometry, mi_row, mi_col, size, tree,
                                         has_chroma, chroma_offset, region_type,
                                         tools);

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
    for (int i = 0; i < kPartitionCount; ++i) {
      if (info.allowed.allowed[i]) {
        forced_partition = static_cast<PartitionType>(i);
        break;
      }
    }
  } else if (!bru_active) {
    forced = true;
    forced_partition = PartitionType::kNone;
  }

  const bool can_ext_sdp =
      options.allow_extended_sdp && tools.enable_extended_sdp &&
      !tools.frame_is_intra && tree == TreeType::kSharedPart && bru_active &&
      size != partition::superblock_size(geometry.sb_size);

  double best_cost = kInfiniteCost;
  PartitionType best_partition = PartitionType::kNone;
  bool best_extended = false;
  RegionType best_region = RegionType::kMixedRegion;
  std::vector<Node> best_children;
  bool found = false;
  bool saw_invalid_cost = false;
  std::string last_error;

  // Pass 0 honours DecisionOptions.  Pass 1 relaxes the policy limits when
  // the spec forbids PARTITION_NONE at an edge and every remaining candidate
  // was policy-blocked: a partial superblock must still be codable.
  for (int pass = 0; pass < 2 && !found; ++pass) {
  for (int index = 0; index < kPartitionCount; ++index) {
    const auto partition = static_cast<PartitionType>(index);
    if (forced && partition != forced_partition) continue;
    if (!info.allowed.is_allowed(partition) &&
        !(forced && !bru_active && partition == PartitionType::kNone)) {
      continue;
    }
    // A spec-forced partition is mandatory: DecisionOptions policy limits
    // apply only to optional splits (this is what lets `max_depth 0` encode a
    // partial-superblock edge).
    if (!forced && pass == 0 && !policy_allows(size, partition, depth, options)) {
      continue;
    }
    const double signal =
        forced ? 0.0 : partition_signal_cost(costs, partition);
    if (!std::isfinite(signal)) continue;

    if (partition == PartitionType::kNone) {
      if (!std::isfinite(costs.leaf)) continue;
      const double total = costs.leaf + signal;
      if (!std::isfinite(total)) continue;
      if (!found || total < best_cost) {
        found = true;
        best_cost = total;
        best_partition = partition;
        best_extended = false;
        best_region = region_type;
        best_children.clear();
      }
      continue;
    }

    const BlockSize subsize = sdp_subsize(partition, size);
    bool next_ext = ext_allowed && partition::block_width(subsize) > 4 &&
                    partition::block_height(subsize) > 4;
    if (partition == PartitionType::kHorz3 ||
        partition == PartitionType::kVert3) {
      const BlockSize mid = sdp_midsize(size);
      next_ext = next_ext && partition::block_width(mid) > 4 &&
                 partition::block_height(mid) > 4;
    }
    const bool region_eligible =
        can_ext_sdp && is_bsize_allowed_for_extended_sdp(size, partition);

    // Variant 1: no SDP / MIXED region (children keep the current tree type).
    if (region_eligible) {
      if (std::isfinite(costs.region_type[1])) {
        std::vector<Node> children;
        double child_total = 0.0;
        SolveResult candidate = solve_partition_children(
            mi_row, mi_col, size, depth, partition, tree, has_chroma,
            chroma_offset, inherited, next_ext, RegionType::kMixedRegion,
            children, child_total);
        if (candidate.status == DecisionStatus::kOk) {
          const double total = signal + costs.region_type[1] + child_total;
          if (std::isfinite(total) && (!found || total < best_cost)) {
            found = true;
            best_cost = total;
            best_partition = partition;
            best_extended = false;
            best_region = RegionType::kMixedRegion;
            best_children = std::move(children);
          }
        } else if (candidate.status == DecisionStatus::kInvalidCost) {
          saw_invalid_cost = true;
        } else if (!candidate.error.empty()) {
          last_error = candidate.error;
        }
      }
    }

    // Variant 2: region_type == INTRA_REGION (luma tree + one chroma block).
    if (region_eligible) {
      NodeInfo chroma_info;
      chroma_info.mi_row = mi_row;
      chroma_info.mi_col = mi_col;
      chroma_info.size = size;
      chroma_info.depth = depth + 1;
      chroma_info.tree_type = TreeType::kChromaPart;
      chroma_info.sdp_chroma_block = true;
      chroma_info.has_chroma = true;
      chroma_info.chroma_offset = true;
      chroma_info.chroma.mi_row = mi_row;
      chroma_info.chroma.mi_col = mi_col;
      chroma_info.chroma.size = size;
      chroma_info.boundary = sdp_implied(
          geometry, mi_row, mi_col, size, TreeType::kChromaPart);
      chroma_info.allowed = sdp_init_allowed(
          geometry, mi_row, mi_col, size, TreeType::kChromaPart, true, true,
          RegionType::kIntraRegion, tools);
      const NodeCosts chroma_costs = cost_fn(chroma_info);
      if (all_costs_sane(chroma_costs) && std::isfinite(chroma_costs.leaf) &&
          std::isfinite(costs.region_type[0])) {
        std::vector<Node> children;
        double child_total = 0.0;
        SolveResult candidate = solve_partition_children(
            mi_row, mi_col, size, depth, partition, TreeType::kLumaPart,
            has_chroma, chroma_offset, inherited, next_ext,
            RegionType::kIntraRegion, children, child_total);
        if (candidate.status == DecisionStatus::kOk) {
          Node chroma;
          chroma.mi_row = mi_row;
          chroma.mi_col = mi_col;
          chroma.size = size;
          chroma.depth = depth + 1;
          chroma.tree_type = TreeType::kChromaPart;
          chroma.partition = PartitionType::kNone;
          chroma.sdp_chroma_block = true;
          chroma.has_chroma = true;
          chroma.chroma_offset = true;
          chroma.chroma.mi_row = mi_row;
          chroma.chroma.mi_col = mi_col;
          chroma.chroma.size = size;
          chroma.boundary = chroma_info.boundary;
          chroma.allowed = chroma_info.allowed;
          chroma.forced = true;
          chroma.codes_chroma = true;
          chroma.cost = chroma_costs.leaf;
          const double total = signal + costs.region_type[0] + child_total +
                               chroma_costs.leaf;
          if (std::isfinite(total) && (!found || total < best_cost)) {
            found = true;
            best_cost = total;
            best_partition = partition;
            best_extended = true;
            best_region = RegionType::kIntraRegion;
            best_children = std::move(children);
            best_children.push_back(std::move(chroma));
          }
        } else if (candidate.status == DecisionStatus::kInvalidCost) {
          saw_invalid_cost = true;
        } else if (!candidate.error.empty()) {
          last_error = candidate.error;
        }
      }
    }

    if (!region_eligible) {
      std::vector<Node> children;
      double child_total = 0.0;
      SolveResult candidate = solve_partition_children(
          mi_row, mi_col, size, depth, partition, tree, has_chroma,
          chroma_offset, inherited, next_ext, region_type, children,
          child_total);
      if (candidate.status != DecisionStatus::kOk) {
        if (candidate.status == DecisionStatus::kInvalidCost) {
          saw_invalid_cost = true;
        }
        if (!candidate.error.empty()) last_error = candidate.error;
        continue;
      }
      const double total = signal + child_total;
      if (std::isfinite(total) && (!found || total < best_cost)) {
        found = true;
        best_cost = total;
        best_partition = partition;
        best_extended = false;
        best_region = region_type;
        best_children = std::move(children);
      }
    }
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
  node.depth = depth;
  node.tree_type = tree;
  node.partition = best_partition;
  node.sdp_split = false;
  node.extended_sdp = best_extended;
  node.region_type = best_region;
  node.region_type_coded = can_ext_sdp &&
                           is_bsize_allowed_for_extended_sdp(size,
                                                             best_partition);
  node.has_chroma = has_chroma;
  node.chroma_offset = chroma_offset;
  const ChromaState resolved =
      sdp_resolve_chroma(geometry, mi_row, mi_col, size, best_partition,
                     has_chroma, chroma_offset, inherited);
  node.chroma = resolved.block;
  node.extended_sdp_allowed = ext_allowed;
  node.boundary = info.boundary;
  node.allowed = info.allowed;
  node.forced = forced;
  node.codes_chroma =
      best_partition == PartitionType::kNone && has_chroma &&
      geometry.has_chroma && tree != TreeType::kLumaPart;
  node.cost = best_cost;
  node.children = std::move(best_children);
  result.cost = best_cost;
  return result;
}

SolveResult Solver::solve(int mi_row, int mi_col, BlockSize size, int depth,
                          TreeType tree, bool has_chroma, bool chroma_offset,
                          const ChromaBlock& inherited, bool ext_allowed,
                          RegionType region_type, bool chroma_known,
                          PartitionType luma_partition, Node& node) const {
  if (!geometry_ready(geometry) || !partition::is_valid_block_size(size) ||
      mi_row < 0 || mi_col < 0 || mi_row >= partition::mi_rows(geometry) ||
      mi_col >= partition::mi_cols(geometry)) {
    SolveResult result;
    result.status = DecisionStatus::kInvalidGeometry;
    result.error = "invalid node geometry";
    return result;
  }
  if (is_sdp_intra_split(geometry, tools, options, tree, size)) {
    return solve_sdp_split(mi_row, mi_col, depth, has_chroma, chroma_offset,
                           inherited, node);
  }
  return solve_normal(mi_row, mi_col, size, depth, tree, has_chroma,
                      chroma_offset, inherited, ext_allowed, region_type,
                      chroma_known, luma_partition, node);
}

}  // namespace

// ---------------------------------------------------------------------------
// Public decision API
// ---------------------------------------------------------------------------

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
    case DecisionStatus::kInvalidTree:
      return "invalid-tree";
  }
  return "unknown";
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

PartitionDecision decide_superblock(const FrameGeometry& geometry,
                                    const Tools& tools, int sb_mi_row,
                                    int sb_mi_col, const CostFunction& costs,
                                    const DecisionOptions& options) {
  PartitionDecision decision;
  std::string error;
  if (!sdp_validate_geometry(geometry, error) || !validate_tools(tools, error)) {
    decision.status = DecisionStatus::kInvalidGeometry;
    decision.error = std::move(error);
    return decision;
  }
  if (tools.enable_sdp && !geometry.has_chroma) {
    decision.status = DecisionStatus::kInvalidTools;
    decision.error = "enable_sdp requires chroma (Monochrome infers it 0)";
    return decision;
  }
  if (!costs) {
    decision.status = DecisionStatus::kInvalidCost;
    decision.error = "cost function is empty";
    return decision;
  }
  const int sb4 = partition::sb_size_mi(geometry);
  if (sb_mi_row < 0 || sb_mi_col < 0 ||
      sb_mi_row >= partition::mi_rows(geometry) ||
      sb_mi_col >= partition::mi_cols(geometry) ||
      (sb_mi_row % sb4) != 0 || (sb_mi_col % sb4) != 0) {
    decision.status = DecisionStatus::kInvalidGeometry;
    decision.error = "superblock origin is outside or unaligned";
    return decision;
  }

  Solver solver{geometry, tools, costs, options};
  ChromaBlock inherited;
  inherited.mi_row = sb_mi_row;
  inherited.mi_col = sb_mi_col;
  inherited.size = partition::superblock_size(geometry.sb_size);
  const SolveResult result = solver.solve(
      sb_mi_row, sb_mi_col, partition::superblock_size(geometry.sb_size),
      /*depth=*/0, TreeType::kSharedPart, /*has_chroma=*/true,
      /*chroma_offset=*/false, inherited,
      /*ext_allowed=*/tools.enable_extended_sdp && !tools.frame_is_intra,
      RegionType::kMixedRegion, /*chroma_known=*/false,
      PartitionType::kNone, decision.root);
  if (result.status != DecisionStatus::kOk) {
    decision.status = result.status;
    decision.error = result.error;
    return decision;
  }
  decision.status = DecisionStatus::kOk;
  decision.cost = result.cost;
  decision.nodes = count_nodes(decision.root);
  decision.leaves = count_leaves(decision.root);
  std::string validate_error;
  if (!validate_tree(geometry, tools, decision.root, validate_error)) {
    decision.status = DecisionStatus::kInvalidTree;
    decision.error = std::move(validate_error);
    return decision;
  }
  return decision;
}

FrameDecision decide_frame(const FrameGeometry& geometry, const Tools& tools,
                           const CostFunction& costs,
                           const DecisionOptions& options) {
  FrameDecision frame;
  std::string error;
  if (!sdp_validate_geometry(geometry, error) || !validate_tools(tools, error)) {
    frame.status = DecisionStatus::kInvalidGeometry;
    frame.error = std::move(error);
    return frame;
  }
  if (tools.enable_sdp && !geometry.has_chroma) {
    frame.status = DecisionStatus::kInvalidTools;
    frame.error = "enable_sdp requires chroma (Monochrome infers it 0)";
    return frame;
  }
  const int sb4 = partition::sb_size_mi(geometry);
  for (int r = 0; r < partition::mi_rows(geometry); r += sb4) {
    for (int c = 0; c < partition::mi_cols(geometry); c += sb4) {
      if (r >= partition::mi_rows(geometry) ||
          c >= partition::mi_cols(geometry)) {
        continue;
      }
      PartitionDecision sb =
          decide_superblock(geometry, tools, r, c, costs, options);
      if (sb.status != DecisionStatus::kOk) {
        frame.status = sb.status;
        frame.error = "superblock (" + std::to_string(r) + "," +
                      std::to_string(c) + "): " + sb.error;
        return frame;
      }
      frame.cost += sb.cost;
      frame.nodes += sb.nodes;
      frame.leaves += sb.leaves;
      frame.superblocks.push_back(std::move(sb.root));
    }
  }
  frame.status = DecisionStatus::kOk;
  std::string validate_error;
  if (!validate_frame(geometry, tools, frame, validate_error)) {
    frame.status = DecisionStatus::kInvalidTree;
    frame.error = std::move(validate_error);
  }
  return frame;
}

void collect_leaves(const Node& node, std::vector<LeafBlock>& leaves) {
  if (node.sdp_split) {
    for (const Node& child : node.children) collect_leaves(child, leaves);
    return;
  }
  if (node.partition == PartitionType::kNone) {
    LeafBlock leaf;
    leaf.mi_row = node.mi_row;
    leaf.mi_col = node.mi_col;
    leaf.size = node.size;
    leaf.depth = node.depth;
    leaf.tree_type = node.tree_type;
    leaf.has_chroma = node.has_chroma;
    leaf.chroma_offset = node.chroma_offset;
    leaf.chroma = node.chroma;
    leaf.codes_chroma = node.codes_chroma;
    leaf.sdp_chroma_block = node.sdp_chroma_block;
    leaves.push_back(leaf);
    return;
  }
  for (const Node& child : node.children) {
    collect_leaves(child, leaves);
  }
}

int count_nodes(const Node& node) {
  int count = 1;
  for (const Node& child : node.children) count += count_nodes(child);
  return count;
}

int count_leaves(const Node& node) {
  if (node.sdp_split) {
    int count = 0;
    for (const Node& child : node.children) count += count_leaves(child);
    return count;
  }
  if (node.partition == PartitionType::kNone) return 1;
  int count = 0;
  for (const Node& child : node.children) count += count_leaves(child);
  return count;
}

}  // namespace av2enc::sdp
