// M1 GPU host-side frame/block plan builder.  See plan.h for the contract.

#include "cuda/pipeline/still/plan/plan.h"

#include "codec/partition/tree/partition.h"
#include "codec/partition/ext/partition_ext.h"
#include "codec/partition/sdp/sdp.h"
#include "codec/transform/variants/tx_variants.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <utility>

namespace av2enc::still::plan {
namespace {

using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionType;

// Same superblock-local decoded map as still_encoder.cpp.
struct DecodedMap {
  int cols = 0;
  int rows = 0;
  std::vector<std::uint8_t> cells;

  void reset(int c, int r) {
    cols = c;
    rows = r;
    cells.assign(static_cast<std::size_t>(c) * static_cast<std::size_t>(r), 0);
  }
  bool at(int x, int y) const {
    if (x < 0 || y < 0 || x >= cols || y >= rows) return false;
    return cells[static_cast<std::size_t>(y) * static_cast<std::size_t>(cols) +
                 static_cast<std::size_t>(x)] != 0;
  }
  void fill(int x, int y, int w, int h) {
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        if (x + i >= 0 && y + j >= 0 && x + i < cols && y + j < rows) {
          cells[static_cast<std::size_t>(y + j) *
                    static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(x + i)] = 1;
        }
      }
    }
  }
};

// Copied from still_encoder.cpp::block_variance_cost_at(): sum over the *visible*
// part of the block, double accumulation, truncated to int64.
std::int64_t block_variance_cost_at(const FrameGeometry& geometry,
                                    std::span<const std::uint8_t> luma,
                                    int mi_row, int mi_col, BlockSize size) {
  const partition::BlockRect rect =
      partition::block_rect(geometry, mi_row, mi_col, size);
  const int w = rect.visible_width;
  const int h = rect.visible_height;
  if (w <= 0 || h <= 0) return 0;
  const int stride = geometry.width;
  std::int64_t sum = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      sum += luma[static_cast<std::size_t>(rect.y + y) *
                      static_cast<std::size_t>(stride) +
                  static_cast<std::size_t>(rect.x + x)];
    }
  }
  const double mean =
      static_cast<double>(sum) / static_cast<double>(w * h);
  double sse = 0.0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const double d =
          static_cast<double>(luma[static_cast<std::size_t>(rect.y + y) *
                                       static_cast<std::size_t>(stride) +
                                   static_cast<std::size_t>(rect.x + x)]) -
          mean;
      sse += d * d;
    }
  }
  return static_cast<std::int64_t>(sse);
}

// Mirrors still_encoder.cpp::count_top_right().
int count_top_right(int plane, int abs_x4, int abs_y4, int w4, int sb_mi_row,
                    int sb_mi_col, int sb_cols_plane, int mi_cols_v,
                    const DecodedMap& decoded) {
  const int sub = plane == 0 ? 0 : 1;
  const int sb_c0 = sb_mi_col >> sub;
  const int sb_r0 = sb_mi_row >> sub;
  const int rel_x = abs_x4 - sb_c0;
  const int rel_y = abs_y4 - sb_r0;
  const int plane_cols = plane == 0 ? mi_cols_v : mi_cols_v / 2;
  const int border_w = std::min(2 * (16 >> sub), plane_cols - sb_c0);
  int count = 0;
  for (int i = 0; i < w4; ++i) {
    const int x = rel_x + w4 + i;
    if (x >= (rel_y == 0 ? border_w : sb_cols_plane)) break;
    if (rel_y - 1 >= 0 &&
        !decoded.at(sb_c0 + x, sb_r0 + rel_y - 1)) {
      break;
    }
    ++count;
  }
  return count;
}

// Mirrors still_encoder.cpp::count_bottom_left().
int count_bottom_left(int plane, int abs_x4, int abs_y4, int h4, int sb_mi_row,
                      int sb_mi_col, int sb_rows_plane,
                      const DecodedMap& decoded) {
  const int sub = plane == 0 ? 0 : 1;
  const int sb_c0 = sb_mi_col >> sub;
  const int sb_r0 = sb_mi_row >> sub;
  const int rel_x = abs_x4 - sb_c0;
  const int rel_y = abs_y4 - sb_r0;
  int count = 0;
  for (int i = 0; i < h4; ++i) {
    const int y = rel_y + h4 + i;
    if (y >= sb_rows_plane) break;
    if (rel_x - 1 >= 0 && !decoded.at(sb_c0 + rel_x - 1, sb_r0 + y)) break;
    ++count;
  }
  return count;
}

NeighborAvailability compute_availability(
    int plane, int origin_x, int origin_y, int abs_x4, int abs_y4, int w4,
    int h4, int sb_mi_row, int sb_mi_col, int sb_cols_plane,
    int sb_rows_plane, int mi_cols_v, const DecodedMap& decoded) {
  NeighborAvailability out;
  // prepare_edges() uses sample-coordinate tests, not 4x4 indices.
  out.above = origin_y > 0;
  out.left = origin_x > 0;
  out.above_left = out.above && out.left;
  const int top_right =
      count_top_right(plane, abs_x4, abs_y4, w4, sb_mi_row, sb_mi_col,
                      sb_cols_plane, mi_cols_v, decoded);
  const int bottom_left =
      count_bottom_left(plane, abs_x4, abs_y4, h4, sb_mi_row, sb_mi_col,
                        sb_rows_plane, decoded);
  out.num4_above_right = static_cast<std::uint8_t>(top_right);
  out.num4_below_left = static_cast<std::uint8_t>(bottom_left);
  out.above_right = top_right > 0;
  out.below_left = bottom_left > 0;
  return out;
}

bool make_plane_tx(int coded_w, int coded_h, const char* plane_name,
                   PlaneTxPlan& out, std::string& error) {
  const auto params =
      txv::make_tx_params(coded_w, coded_h, txv::TxType::kDctDct);
  if (!params.has_value()) {
    std::ostringstream os;
    os << "unsupported transform size " << coded_w << 'x' << coded_h
       << " on " << plane_name;
    error = os.str();
    return false;
  }
  out.coded_w = coded_w;
  out.coded_h = coded_h;
  out.tx_size = static_cast<int>(params->tx_size);
  out.adj_w = static_cast<int>(params->w);
  out.adj_h = static_cast<int>(params->h);
  out.tx_type = static_cast<std::uint8_t>(params->tx_type);
  out.coeff_count = static_cast<std::uint32_t>(out.adj_w * out.adj_h);
  return true;
}

int clipped_extent(int size, int limit, int origin) {
  return std::max(0, std::min(size, limit - origin));
}

// ChromaMiRow / ChromaMiCol / ChromaMiSize (§5.20.3.1) captured at the last
// ancestor whose chromaOffset was 0.  A sub-8 leaf does not always own the
// chroma block it inherits.
struct ChromaInfo {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  bool valid = false;
};

// §5.18.1: an intra frame uses BLOCK_128X128 even when the sequence signals
// use_256x256_superblock.  The still planner is always intra.
int effective_sb_size(int sequence_sb_size) {
  return sequence_sb_size == 256 ? 128 : sequence_sb_size;
}

bool sequence_sb_size_ok(int sb_size) {
  return sb_size == 64 || sb_size == 128 || sb_size == 256;
}

struct LeafView {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  int depth = 0;
  bool has_chroma = false;
  bool chroma_offset = false;
  ChromaInfo chroma;
  std::uint8_t tree_type = 0;
  bool sdp_chroma_block = false;
};

// Neutral view of one decided partition node.  The rect-only, extended and SDP
// solvers all map onto this so tree export, block flattening and validation
// share one walk.
struct NodeView {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  int depth = 0;
  partition::PartitionType base_partition = partition::PartitionType::kNone;
  partition_ext::PartitionType ext_partition = partition_ext::PartitionType::kNone;
  bool has_chroma = false;
  bool chroma_offset = false;
  ChromaInfo chroma;
  bool forced = false;
  std::uint8_t allowed_mask = 0;
  std::uint16_t allowed_mask_ext = 0;
  bool codes_chroma = false;
  double cost = 0.0;
  std::uint8_t tree_type = 0;
  bool sdp_split = false;
  bool sdp_chroma_block = false;
  std::vector<NodeView> children;

  bool is_leaf() const { return children.empty(); }
};

partition_ext::PartitionType ext_from_base(partition::PartitionType type) {
  switch (type) {
    case partition::PartitionType::kNone:
      return partition_ext::PartitionType::kNone;
    case partition::PartitionType::kSplit:
      return partition_ext::PartitionType::kSplit;
    case partition::PartitionType::kHorz:
      return partition_ext::PartitionType::kHorz;
    case partition::PartitionType::kVert:
      return partition_ext::PartitionType::kVert;
  }
  return partition_ext::PartitionType::kNone;
}

partition::PartitionType base_from_ext(partition_ext::PartitionType type) {
  using PE = partition_ext::PartitionType;
  switch (type) {
    case PE::kNone:
      return partition::PartitionType::kNone;
    case PE::kSplit:
      return partition::PartitionType::kSplit;
    case PE::kHorz:
    case PE::kHorz3:
    case PE::kHorz4A:
    case PE::kHorz4B:
      return partition::PartitionType::kHorz;
    case PE::kVert:
    case PE::kVert3:
    case PE::kVert4A:
    case PE::kVert4B:
      return partition::PartitionType::kVert;
  }
  return partition::PartitionType::kNone;
}

std::uint16_t ext_mask_from_base(const partition::PartitionSet& allowed) {
  std::uint16_t mask = 0;
  if (allowed.allowed[static_cast<int>(partition::PartitionType::kNone)]) {
    mask = static_cast<std::uint16_t>(
        mask | (1U << static_cast<int>(partition_ext::PartitionType::kNone)));
  }
  if (allowed.allowed[static_cast<int>(partition::PartitionType::kSplit)]) {
    mask = static_cast<std::uint16_t>(
        mask | (1U << static_cast<int>(partition_ext::PartitionType::kSplit)));
  }
  if (allowed.allowed[static_cast<int>(partition::PartitionType::kHorz)]) {
    mask = static_cast<std::uint16_t>(
        mask | (1U << static_cast<int>(partition_ext::PartitionType::kHorz)));
  }
  if (allowed.allowed[static_cast<int>(partition::PartitionType::kVert)]) {
    mask = static_cast<std::uint16_t>(
        mask | (1U << static_cast<int>(partition_ext::PartitionType::kVert)));
  }
  return mask;
}

std::uint8_t base_mask_from_ext(const partition_ext::PartitionSet& allowed) {
  std::uint8_t mask = 0;
  using PE = partition_ext::PartitionType;
  if (allowed.is_allowed(PE::kNone)) {
    mask = static_cast<std::uint8_t>(
        mask | (1U << static_cast<int>(partition::PartitionType::kNone)));
  }
  if (allowed.is_allowed(PE::kSplit)) {
    mask = static_cast<std::uint8_t>(
        mask | (1U << static_cast<int>(partition::PartitionType::kSplit)));
  }
  if (allowed.is_allowed(PE::kHorz) || allowed.is_allowed(PE::kHorz3) ||
      allowed.is_allowed(PE::kHorz4A) || allowed.is_allowed(PE::kHorz4B)) {
    mask = static_cast<std::uint8_t>(
        mask | (1U << static_cast<int>(partition::PartitionType::kHorz)));
  }
  if (allowed.is_allowed(PE::kVert) || allowed.is_allowed(PE::kVert3) ||
      allowed.is_allowed(PE::kVert4A) || allowed.is_allowed(PE::kVert4B)) {
    mask = static_cast<std::uint8_t>(
        mask | (1U << static_cast<int>(partition::PartitionType::kVert)));
  }
  return mask;
}

// Same recursion as still_encoder.cpp::chroma_step(): the node updates
// ChromaMi when HasChroma is set and chromaOffset is still 0, then children
// inherit it.
void convert_base_node(const partition::PartitionNode& node,
                       const ChromaInfo& inherited, NodeView& out) {
  out.mi_row = node.mi_row;
  out.mi_col = node.mi_col;
  out.size = node.size;
  out.depth = node.depth;
  out.base_partition = node.partition;
  out.ext_partition = ext_from_base(node.partition);
  out.has_chroma = node.has_chroma;
  out.chroma_offset = node.chroma_offset;
  out.forced = node.forced;
  out.cost = node.cost;
  ChromaInfo current = inherited;
  if (node.has_chroma && !node.chroma_offset) {
    current = ChromaInfo{node.mi_row, node.mi_col, node.size, true};
  }
  out.chroma = current;
  out.codes_chroma = node.has_chroma;
  std::uint8_t mask = 0;
  for (int p = 0; p < partition::kPartitionTypeCount; ++p) {
    if (node.allowed.allowed[p]) {
      mask = static_cast<std::uint8_t>(mask | (1U << p));
    }
  }
  out.allowed_mask = mask;
  out.allowed_mask_ext = ext_mask_from_base(node.allowed);
  out.children.resize(node.children.size());
  for (std::size_t i = 0; i < node.children.size(); ++i) {
    convert_base_node(node.children[i], current, out.children[i]);
  }
}

void convert_ext_node(const partition_ext::PartitionNode& node, NodeView& out) {
  out.mi_row = node.mi_row;
  out.mi_col = node.mi_col;
  out.size = node.size;
  out.depth = node.depth;
  out.base_partition = base_from_ext(node.partition);
  out.ext_partition = node.partition;
  out.has_chroma = node.has_chroma;
  out.chroma_offset = node.chroma_offset;
  out.forced = node.forced;
  out.cost = node.cost;
  out.chroma = ChromaInfo{node.chroma.mi_row, node.chroma.mi_col,
                          node.chroma.size,
                          partition::is_valid_block_size(node.chroma.size)};
  out.codes_chroma = node.has_chroma;
  out.allowed_mask = base_mask_from_ext(node.allowed);
  std::uint16_t mask = 0;
  for (int p = 0; p < partition_ext::kPartitionCount; ++p) {
    if (node.allowed.allowed[p]) {
      mask = static_cast<std::uint16_t>(mask | (1U << p));
    }
  }
  out.allowed_mask_ext = mask;
  out.children.resize(node.children.size());
  for (std::size_t i = 0; i < node.children.size(); ++i) {
    convert_ext_node(node.children[i], out.children[i]);
  }
}

void convert_sdp_node(const sdp::Node& node, NodeView& out) {
  out.mi_row = node.mi_row;
  out.mi_col = node.mi_col;
  out.size = node.size;
  out.depth = node.depth;
  out.base_partition = base_from_ext(node.partition);
  out.ext_partition = node.partition;
  out.has_chroma = node.has_chroma;
  out.chroma_offset = node.chroma_offset;
  out.forced = node.forced;
  out.cost = node.cost;
  out.tree_type = static_cast<std::uint8_t>(node.tree_type);
  out.sdp_split = node.sdp_split;
  out.sdp_chroma_block = node.sdp_chroma_block;
  out.chroma = ChromaInfo{node.chroma.mi_row, node.chroma.mi_col,
                          node.chroma.size,
                          partition::is_valid_block_size(node.chroma.size)};
  out.codes_chroma = node.codes_chroma;
  out.allowed_mask = base_mask_from_ext(node.allowed);
  std::uint16_t mask = 0;
  for (int p = 0; p < partition_ext::kPartitionCount; ++p) {
    if (node.allowed.allowed[p]) {
      mask = static_cast<std::uint16_t>(mask | (1U << p));
    }
  }
  out.allowed_mask_ext = mask;
  out.children.resize(node.children.size());
  for (std::size_t i = 0; i < node.children.size(); ++i) {
    convert_sdp_node(node.children[i], out.children[i]);
  }
}

void collect_leaf_views(const NodeView& node, std::vector<LeafView>& out) {
  if (node.is_leaf()) {
    LeafView leaf;
    leaf.mi_row = node.mi_row;
    leaf.mi_col = node.mi_col;
    leaf.size = node.size;
    leaf.depth = node.depth;
    leaf.has_chroma = node.codes_chroma;
    leaf.chroma_offset = node.chroma_offset;
    leaf.chroma = node.chroma;
    leaf.tree_type = node.tree_type;
    leaf.sdp_chroma_block = node.sdp_chroma_block;
    out.push_back(leaf);
    return;
  }
  for (const NodeView& child : node.children) {
    collect_leaf_views(child, out);
  }
}

// Appends `node` and its whole subtree to the preorder tree array.
void append_tree_node(const NodeView& node, std::uint32_t sb_index,
                      FramePlan& plan, std::uint32_t& leaf_cursor) {
  const std::uint32_t index = static_cast<std::uint32_t>(plan.tree.size());
  TreeNode entry;
  entry.index = index;
  entry.sb_index = sb_index;
  entry.mi_row = node.mi_row;
  entry.mi_col = node.mi_col;
  entry.block_size = static_cast<std::uint8_t>(node.size);
  entry.partition = static_cast<std::uint8_t>(node.base_partition);
  entry.depth = static_cast<std::uint8_t>(node.depth);
  entry.allowed_mask = node.allowed_mask;
  entry.allowed_mask_ext = node.allowed_mask_ext;
  entry.ext_partition = static_cast<std::uint8_t>(node.ext_partition);
  entry.tree_type = node.tree_type;
  entry.sdp_split = node.sdp_split;
  entry.sdp_chroma_block = node.sdp_chroma_block;
  entry.forced = node.forced;
  entry.sb_root = node.depth == 0;
  entry.cost = node.cost;
  const bool leaf = node.is_leaf();
  entry.first_child = leaf ? kNoTreeNode : index + 1;
  plan.tree.push_back(entry);

  if (leaf) {
    plan.tree[index].leaf_index = leaf_cursor;
    ++leaf_cursor;
    return;
  }
  std::uint8_t child_count = 0;
  for (const NodeView& child : node.children) {
    append_tree_node(child, sb_index, plan, leaf_cursor);
    ++child_count;
  }
  plan.tree[index].child_count = child_count;
  plan.tree[index].subtree =
      static_cast<std::uint32_t>(plan.tree.size() - index - 1);
}

}  // namespace

void apply_debug_env(PlanOptions& options) {
  if (std::getenv("AV2ENC_STILL_SQUARE_ONLY") != nullptr) {
    options.square_only = true;
  }
  if (std::getenv("AV2ENC_STILL_LEAF32") != nullptr) {
    options.leaf32 = true;
  }
}

bool build_frame_plan_impl(int width, int height,
                           std::span<const std::uint8_t> luma,
                           const PlanOptions& options,
                           const LeafCostFunction& leaf_cost,
                           FramePlan& plan, std::string& error) {
  error.clear();
  plan = FramePlan{};
  if (width <= 0 || height <= 0) {
    error = "frame dimensions must be positive";
    return false;
  }
  if (!sequence_sb_size_ok(options.sb_size)) {
    error = "sb_size must be 64, 128, or 256";
    return false;
  }
  if (options.min_block_size < 1) {
    error = "min_block_size must be positive";
    return false;
  }
  if (options.max_partition_depth < 0) {
    error = "max_partition_depth must be non-negative";
    return false;
  }
  if (options.enable_uneven_4way_partitions &&
      !options.enable_ext_partitions) {
    error = "uneven 4-way partitions require enable_ext_partitions";
    return false;
  }
  if (options.sdp && (options.enable_ext_partitions || options.sub8)) {
    error = "SDP cannot be combined with extended partitions or sub-8 search";
    return false;
  }
  const std::size_t luma_size = static_cast<std::size_t>(width) *
                                static_cast<std::size_t>(height);
  if (luma.size() < luma_size) {
    error = "luma plane is smaller than width*height";
    return false;
  }

  FrameGeometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = effective_sb_size(options.sb_size);
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;

  std::string geometry_error;
  if (!partition::validate_geometry(geometry, geometry_error)) {
    error = geometry_error;
    return false;
  }

  PlanOptions resolved = options;
  // Same lambda / signal cost as the CPU M1 encoder.
  const double lambda =
      0.85 * std::pow(2.0, (static_cast<int>(resolved.base_q_idx) - 12) / 6.0);
  const double signal_cost = lambda * 2.0;

  const auto leaf_cost_px = [&](int mi_row, int mi_col, BlockSize size) {
    const std::int64_t variance =
        leaf_cost ? leaf_cost(NodeGeometry{
                        mi_row, mi_col, static_cast<std::uint8_t>(size)})
                  : block_variance_cost_at(geometry, luma, mi_row, mi_col,
                                           size);
    double leaf = static_cast<double>(variance);
    // The M1 writer codes one TX_MODE_LARGEST transform block per plane, so a
    // coding block larger than 64x64 is not codable.  A 128x128 root therefore
    // never wins as PARTITION_NONE.
    const int bw = partition::block_width(size);
    const int bh = partition::block_height(size);
    if (bw > 64 || bh > 64) leaf += 1.0e18;
    if (resolved.square_only && bw != bh) leaf += 1.0e18;
    if (resolved.leaf32 && (bw > 32 || bh > 32)) leaf += 1.0e18;
    return leaf;
  };

  std::vector<NodeView> roots;
  double frame_cost = 0.0;
  int frame_nodes = 0;
  int frame_leaves = 0;

  if (resolved.sdp) {
    sdp::Tools tools;
    tools.enable_sdp = true;
    tools.enable_extended_sdp = false;
    tools.enable_ext_partitions = false;
    tools.enable_uneven_4way_partitions = false;
    tools.single_picture_header_flag = true;
    tools.frame_is_intra = true;
    sdp::DecisionOptions sopt;
    // The CPU product path's first decoder-gated SDP configuration: every
    // intra 64x64 split carries a forced PARTITION_NONE luma tree and a forced
    // PARTITION_NONE chroma tree.
    sopt.min_block_size_px = 64;
    sopt.max_depth = 0;
    sopt.allow_rect = false;
    sopt.allow_split = false;
    sopt.allow_3way = false;
    sopt.allow_uneven4 = false;
    sopt.allow_sdp = true;
    sopt.allow_extended_sdp = false;
    const auto cost_fn = [&](const sdp::NodeInfo& info) {
      sdp::NodeCosts costs;
      costs.leaf = info.tree_type == sdp::TreeType::kChromaPart
                       ? 0.0
                       : leaf_cost_px(info.mi_row, info.mi_col, info.size);
      costs.none_signal = 0.0;
      costs.split_signal = sdp::kInfiniteCost;
      costs.horz_signal = sdp::kInfiniteCost;
      costs.vert_signal = sdp::kInfiniteCost;
      costs.horz3_signal = sdp::kInfiniteCost;
      costs.vert3_signal = sdp::kInfiniteCost;
      costs.horz4a_signal = sdp::kInfiniteCost;
      costs.horz4b_signal = sdp::kInfiniteCost;
      costs.vert4a_signal = sdp::kInfiniteCost;
      costs.vert4b_signal = sdp::kInfiniteCost;
      costs.region_type[0] = sdp::kInfiniteCost;
      costs.region_type[1] = sdp::kInfiniteCost;
      return costs;
    };
    const sdp::FrameDecision decision =
        sdp::decide_frame(geometry, tools, cost_fn, sopt);
    if (decision.status != sdp::DecisionStatus::kOk) {
      error = "SDP partition decision failed: " + decision.error;
      return false;
    }
    if (!sdp::validate_frame(geometry, tools, decision, error)) {
      error = "SDP partition tree invalid: " + error;
      return false;
    }
    frame_cost = decision.cost;
    frame_nodes = decision.nodes;
    frame_leaves = decision.leaves;
    roots.resize(decision.superblocks.size());
    for (std::size_t i = 0; i < decision.superblocks.size(); ++i) {
      convert_sdp_node(decision.superblocks[i], roots[i]);
    }
  } else if (resolved.enable_ext_partitions) {
    partition_ext::Tools tools;
    tools.enable_ext_partitions = true;
    tools.enable_uneven_4way_partitions =
        resolved.enable_uneven_4way_partitions;
    partition_ext::DecisionOptions eopt;
    // The extended families reach the 1/8 subsizes; the CPU product path uses
    // a 4-sample floor for this solver.
    eopt.min_block_size_px = 4;
    eopt.max_depth = resolved.sub8
                         ? std::max(resolved.max_partition_depth, 8)
                         : resolved.max_partition_depth;
    eopt.allow_rect = true;
    eopt.allow_split = geometry.sb_size == 128;
    eopt.allow_3way = true;
    eopt.allow_uneven4 = resolved.enable_uneven_4way_partitions;
    // Reproduces partition::decide_frame's tie-breaks when ext is unavailable.
    eopt.tie_break = partition_ext::TieBreakOrder::kRectCompatOrder;
    if (resolved.force_single_block) {
      eopt.max_depth = 0;
      eopt.min_block_size_px = 64;
    }
    const auto cost_fn = [&](const partition_ext::NodeInfo& info) {
      partition_ext::NodeCosts costs;
      costs.leaf = leaf_cost_px(info.mi_row, info.mi_col, info.size);
      costs.none_signal = 0.0;
      costs.split_signal = signal_cost;
      costs.horz_signal = signal_cost;
      costs.vert_signal = signal_cost;
      const int w = partition::block_width(info.size);
      const int h = partition::block_height(info.size);
      costs.horz3_signal =
          (h >= 32 && w >= 8) ? signal_cost : partition_ext::kInfiniteCost;
      costs.vert3_signal =
          (w >= 32 && h >= 8) ? signal_cost : partition_ext::kInfiniteCost;
      const double h4 = (h >= 32 && w >= 8) ? signal_cost
                                            : partition_ext::kInfiniteCost;
      // Vertical uneven 4-way with a 4-wide 1/8 child still desyncs the pinned
      // decoder at the bottom edge (docs/negative-results.md); keep it at
      // 8-wide subsizes.
      const double v4 = (w >= 64 && h >= 8) ? signal_cost
                                            : partition_ext::kInfiniteCost;
      costs.horz4a_signal = h4;
      costs.horz4b_signal = h4;
      costs.vert4a_signal = v4;
      costs.vert4b_signal = v4;
      return costs;
    };
    const partition_ext::FrameDecision decision =
        partition_ext::decide_frame(geometry, cost_fn, tools, eopt);
    if (decision.status != partition_ext::DecisionStatus::kOk) {
      error = "extended partition decision failed: " + decision.error;
      return false;
    }
    if (!partition_ext::validate_frame(geometry, tools, decision, error)) {
      error = "extended partition tree invalid: " + error;
      return false;
    }
    frame_cost = decision.cost;
    frame_nodes = decision.nodes;
    frame_leaves = decision.leaves;
    roots.resize(decision.superblocks.size());
    for (std::size_t i = 0; i < decision.superblocks.size(); ++i) {
      convert_ext_node(decision.superblocks[i], roots[i]);
    }
  } else {
    partition::DecisionOptions poptions;
    poptions.min_block_size_px = resolved.sub8 ? 4 : resolved.min_block_size;
    poptions.max_depth = resolved.sub8
                             ? std::max(resolved.max_partition_depth, 8)
                             : resolved.max_partition_depth;
    poptions.allow_rect = resolved.allow_rect;
    // PARTITION_SPLIT exists only at 128x128 / 256x256.  The CPU still path
    // enables it whenever the frame SbSize is 128.
    poptions.allow_split = resolved.allow_split || geometry.sb_size == 128;
    if (resolved.force_single_block) {
      poptions.max_depth = 0;
      poptions.min_block_size_px = 64;
    }
    const auto cost_fn = [&](const partition::PartitionNodeInfo& info) {
      partition::NodeCosts costs;
      costs.leaf = leaf_cost_px(info.mi_row, info.mi_col, info.size);
      costs.none_signal = 0.0;
      costs.split_signal = signal_cost;
      costs.horz_signal = signal_cost;
      costs.vert_signal = signal_cost;
      return costs;
    };
    const partition::FrameDecision decision =
        partition::decide_frame(geometry, cost_fn, poptions);
    if (decision.status != partition::DecisionStatus::kOk) {
      error = "partition decision failed: " + decision.error;
      return false;
    }
    if (!partition::validate_frame(geometry, decision, error)) {
      error = "partition tree invalid: " + error;
      return false;
    }
    frame_cost = decision.cost;
    frame_nodes = decision.nodes;
    frame_leaves = decision.leaves;
    roots.resize(decision.superblocks.size());
    for (std::size_t i = 0; i < decision.superblocks.size(); ++i) {
      convert_base_node(decision.superblocks[i], ChromaInfo{}, roots[i]);
    }
  }

  const int mi_cols_v = partition::mi_cols(geometry);
  const int mi_rows_v = partition::mi_rows(geometry);
  const int chroma_width = (width + 1) / 2;
  const int chroma_height = (height + 1) / 2;

  plan.width = width;
  plan.height = height;
  plan.chroma_width = chroma_width;
  plan.chroma_height = chroma_height;
  plan.mi_cols = mi_cols_v;
  plan.mi_rows = mi_rows_v;
  plan.sb_size = geometry.sb_size;
  plan.sequence_sb_size = options.sb_size;
  plan.sb_cols = partition::sb_cols(geometry);
  plan.sb_rows = partition::sb_rows(geometry);
  plan.coded_luma_w = mi_cols_v * 4;
  plan.coded_luma_h = mi_rows_v * 4;
  plan.coded_chroma_w = mi_cols_v * 2;
  plan.coded_chroma_h = mi_rows_v * 2;
  plan.luma_stride = (plan.coded_luma_w + 63) & ~63;
  plan.luma_padded_height = (plan.coded_luma_h + 63) & ~63;
  plan.chroma_stride = plan.luma_stride / 2;
  plan.chroma_padded_height = plan.luma_padded_height / 2;
  plan.leaves = frame_leaves;
  plan.nodes = frame_nodes;
  plan.cost = frame_cost;
  plan.options = resolved;
  plan.blocks.reserve(static_cast<std::size_t>(frame_leaves));

  // Superblock-local decoded maps, reset exactly like clear_block_decoded_flags.
  DecodedMap decoded[3];
  for (int plane = 0; plane < 3; ++plane) {
    decoded[plane].reset(plane == 0 ? mi_cols_v : mi_cols_v / 2,
                         plane == 0 ? mi_rows_v : mi_rows_v / 2);
  }

  std::vector<LeafView> leaves;
  for (int sbi = 0; sbi < static_cast<int>(roots.size()); ++sbi) {
    const NodeView& sb = roots[static_cast<std::size_t>(sbi)];
    for (int plane = 0; plane < 3; ++plane) {
      decoded[plane].reset(plane == 0 ? mi_cols_v : mi_cols_v / 2,
                           plane == 0 ? mi_rows_v : mi_rows_v / 2);
    }
    int sb_cols_plane[3] = {0, 0, 0};
    int sb_rows_plane[3] = {0, 0, 0};
    const int sb_mi = geometry.sb_size / 4;
    sb_cols_plane[0] = std::min(sb_mi, mi_cols_v - sb.mi_col);
    sb_rows_plane[0] = std::min(sb_mi, mi_rows_v - sb.mi_row);
    sb_cols_plane[1] =
        std::min(sb_mi / 2, mi_cols_v / 2 - sb.mi_col / 2);
    sb_rows_plane[1] =
        std::min(sb_mi / 2, mi_rows_v / 2 - sb.mi_row / 2);
    sb_cols_plane[2] = sb_cols_plane[1];
    sb_rows_plane[2] = sb_rows_plane[1];

    leaves.clear();
    collect_leaf_views(sb, leaves);

    for (const LeafView& leaf : leaves) {
      BlockPlan block;
      block.index = static_cast<std::uint32_t>(plan.blocks.size());
      block.sb_index = static_cast<std::uint32_t>(sbi);
      block.sb_mi_row = sb.mi_row;
      block.sb_mi_col = sb.mi_col;
      block.mi_row = leaf.mi_row;
      block.mi_col = leaf.mi_col;
      block.x = leaf.mi_col * 4;
      block.y = leaf.mi_row * 4;
      block.chroma_x = block.x / 2;
      block.chroma_y = block.y / 2;
      const bool codes_luma = leaf.tree_type != 2;  // CHROMA_PART has no luma
      block.tree_type = leaf.tree_type;
      block.block_size = static_cast<std::uint8_t>(leaf.size);
      block.depth = static_cast<std::uint8_t>(leaf.depth);
      block.has_chroma = leaf.has_chroma;
      block.chroma_offset = leaf.chroma_offset;
      if (codes_luma) {
        block.luma_w = partition::block_width(leaf.size);
        block.luma_h = partition::block_height(leaf.size);
        block.visible_luma_w =
            clipped_extent(block.luma_w, width, block.x);
        block.visible_luma_h =
            clipped_extent(block.luma_h, height, block.y);
      } else {
        block.luma_w = 0;
        block.luma_h = 0;
        block.visible_luma_w = 0;
        block.visible_luma_h = 0;
      }

      if (leaf.chroma.valid) {
        block.chroma_mi_row = leaf.chroma.mi_row;
        block.chroma_mi_col = leaf.chroma.mi_col;
        block.chroma_mi_size = static_cast<std::uint8_t>(leaf.chroma.size);
      } else {
        block.chroma_mi_row = block.mi_row;
        block.chroma_mi_col = block.mi_col;
        block.chroma_mi_size = static_cast<std::uint8_t>(leaf.size);
      }

      if (leaf.has_chroma) {
        const BlockSize chroma_luma_size = leaf.chroma.valid
                                               ? leaf.chroma.size
                                               : leaf.size;
        const BlockSize chroma_size =
            partition::chroma_plane_residual_size(geometry, chroma_luma_size);
        if (!partition::is_valid_block_size(chroma_size)) {
          error = "invalid chroma residual size";
          return false;
        }
        block.chroma_w = partition::block_width(chroma_size);
        block.chroma_h = partition::block_height(chroma_size);
        block.chroma_x = block.chroma_mi_col * 2;
        block.chroma_y = block.chroma_mi_row * 2;
        block.visible_chroma_w = clipped_extent(
            block.chroma_w, chroma_width, block.chroma_x);
        block.visible_chroma_h = clipped_extent(
            block.chroma_h, chroma_height, block.chroma_y);
      } else {
        // A non-owning sub-8 leaf codes only its luma payload.
        block.chroma_w = 0;
        block.chroma_h = 0;
        block.chroma_x = 0;
        block.chroma_y = 0;
        block.visible_chroma_w = 0;
        block.visible_chroma_h = 0;
      }

      if (codes_luma &&
          !make_plane_tx(block.luma_w, block.luma_h, "luma", block.luma,
                         error)) {
        return false;
      }
      if (block.chroma_w > 0 &&
          block.chroma_h > 0) {
        if (!make_plane_tx(block.chroma_w, block.chroma_h, "U", block.u,
                           error) ||
            !make_plane_tx(block.chroma_w, block.chroma_h, "V", block.v,
                           error)) {
          return false;
        }
      }

      block.luma.coeff_offset =
          codes_luma ? plan.luma_coeff_count : 0;
      block.u.coeff_offset = leaf.has_chroma ? plan.u_coeff_count : 0;
      block.v.coeff_offset = leaf.has_chroma ? plan.v_coeff_count : 0;

      const int luma_w4 = block.luma_w / 4;
      const int luma_h4 = block.luma_h / 4;
      const int chroma_w4 = block.chroma_w / 4;
      const int chroma_h4 = block.chroma_h / 4;
      if (codes_luma) {
        block.luma_avail = compute_availability(
            0, block.x, block.y, block.mi_col, block.mi_row, luma_w4, luma_h4,
            sb.mi_row, sb.mi_col, sb_cols_plane[0], sb_rows_plane[0],
            mi_cols_v, decoded[0]);
      }
      if (leaf.has_chroma) {
        block.chroma_avail = compute_availability(
            1, block.chroma_x, block.chroma_y, block.chroma_mi_col / 2,
            block.chroma_mi_row / 2, chroma_w4, chroma_h4, sb.mi_row,
            sb.mi_col, sb_cols_plane[1], sb_rows_plane[1], mi_cols_v,
            decoded[1]);
      }

      // Coefficient layout: one contiguous per-plane buffer.
      if (codes_luma) plan.luma_coeff_count += block.luma.coeff_count;
      if (leaf.has_chroma) {
        plan.u_coeff_count += block.u.coeff_count;
        plan.v_coeff_count += block.v.coeff_count;
      }

      // Fill decoded maps after the block, exactly like encode_block().
      if (codes_luma) {
        decoded[0].fill(block.mi_col, block.mi_row, luma_w4, luma_h4);
      }
      if (leaf.has_chroma) {
        decoded[1].fill(block.chroma_mi_col / 2, block.chroma_mi_row / 2,
                        chroma_w4, chroma_h4);
        decoded[2].fill(block.chroma_mi_col / 2, block.chroma_mi_row / 2,
                        chroma_w4, chroma_h4);
      }

      plan.blocks.push_back(block);
    }
  }

  // Preorder tree view.  Leaves are assigned in the same order as `blocks`.
  plan.tree.reserve(static_cast<std::size_t>(frame_nodes));
  std::uint32_t leaf_cursor = 0;
  for (int sbi = 0; sbi < static_cast<int>(roots.size()); ++sbi) {
    append_tree_node(roots[static_cast<std::size_t>(sbi)],
                     static_cast<std::uint32_t>(sbi), plan, leaf_cursor);
  }
  if (leaf_cursor != static_cast<std::uint32_t>(plan.blocks.size())) {
    error = "tree leaf count does not match the block list";
    return false;
  }

  if (!validate_frame_plan(plan, error)) {
    error = "frame plan invalid: " + error;
    return false;
  }
  return true;
}

bool build_frame_plan(int width, int height, std::span<const std::uint8_t> luma,
                      const PlanOptions& options, FramePlan& plan,
                      std::string& error) {
  return build_frame_plan_impl(width, height, luma, options,
                               LeafCostFunction{}, plan, error);
}

bool build_frame_plan_with_leaf_cost(
    int width, int height, std::span<const std::uint8_t> luma,
    const PlanOptions& options, const LeafCostFunction& leaf_cost,
    FramePlan& plan, std::string& error) {
  if (!leaf_cost) {
    error = "leaf cost provider is empty";
    return false;
  }
  return build_frame_plan_impl(width, height, luma, options, leaf_cost, plan,
                               error);
}

namespace {

bool validate_tree_node(const FramePlan& plan, std::size_t index,
                        std::uint32_t& leaf_cursor, std::size_t& next,
                        std::string& error) {
  if (index >= plan.tree.size()) {
    error = "tree node index is outside the array";
    return false;
  }
  const TreeNode& node = plan.tree[index];
  if (node.index != index) {
    error = "tree node index is not its array position";
    return false;
  }
  if (!partition::is_valid_block_size(
          static_cast<BlockSize>(node.block_size))) {
    error = "tree node has an invalid block size";
    return false;
  }
  if (node.partition > static_cast<std::uint8_t>(partition::PartitionType::kVert)) {
    error = "tree node has an invalid partition type";
    return false;
  }
  if ((node.allowed_mask & ~0x0FU) != 0) {
    error = "tree node has an invalid allowed mask";
    return false;
  }
  if (node.ext_partition > 9 ||
      (node.allowed_mask_ext & ~0x3FFU) != 0) {
    error = "tree node has an invalid extended-partition field";
    return false;
  }
  if (node.ext_partition != 0 &&
      base_from_ext(static_cast<partition_ext::PartitionType>(
          node.ext_partition)) !=
          static_cast<partition::PartitionType>(node.partition)) {
    error = "tree node extended and base partitions disagree";
    return false;
  }
  if (node.tree_type > 2) {
    error = "tree node has an invalid TreeType";
    return false;
  }
  if (node.sdp_chroma_block && node.tree_type != 2) {
    error = "synthetic SDP chroma block is not a CHROMA_PART node";
    return false;
  }
  if (node.sb_root != (node.depth == 0)) {
    error = "tree node superblock-root flag disagrees with its depth";
    return false;
  }
  if (node.child_count == 0) {
    if (node.first_child != kNoTreeNode || node.subtree != 0) {
      error = "tree leaf has child metadata";
      return false;
    }
    if (node.partition !=
        static_cast<std::uint8_t>(partition::PartitionType::kNone)) {
      error = "tree leaf has a non-NONE partition";
      return false;
    }
    if (node.leaf_index != leaf_cursor) {
      error = "tree leaf order does not match the block array";
      return false;
    }
    if (node.leaf_index >= plan.blocks.size()) {
      error = "tree leaf index is outside the block array";
      return false;
    }
    const BlockPlan& block = plan.blocks[node.leaf_index];
    if (block.mi_row != node.mi_row || block.mi_col != node.mi_col ||
        block.tree_type != node.tree_type ||
        (block.luma_w != 0 && block.block_size != node.block_size)) {
      error = "tree leaf geometry disagrees with its block entry";
      return false;
    }
    ++leaf_cursor;
    next = index + 1;
    return true;
  }

  if (node.child_count == 0 || node.child_count > 4) {
    error = "tree internal node has an invalid child count";
    return false;
  }
  if (node.partition ==
          static_cast<std::uint8_t>(partition::PartitionType::kNone) &&
      !node.sdp_split) {
    error = "tree internal NONE node is not an SDP split";
    return false;
  }
  if (node.sdp_split &&
      (node.tree_type != 0 || node.child_count != 2)) {
    error = "SDP split node must be SHARED_PART with two tree roots";
    return false;
  }
  if (node.first_child != index + 1) {
    error = "tree children are not contiguous in preorder";
    return false;
  }
  std::size_t cursor = index + 1;
  for (int child = 0; child < static_cast<int>(node.child_count); ++child) {
    std::size_t after = 0;
    if (!validate_tree_node(plan, cursor, leaf_cursor, after, error)) {
      return false;
    }
    if (after <= cursor) {
      error = "tree child walk did not advance";
      return false;
    }
    cursor = after;
  }
  if (cursor != index + 1 + node.subtree) {
    error = "tree subtree count disagrees with its children";
    return false;
  }
  next = cursor;
  return true;
}

}  // namespace

bool validate_frame_plan(const FramePlan& plan, std::string& error) {
  error.clear();
  if (plan.width <= 0 || plan.height <= 0) {
    error = "frame dimensions must be positive";
    return false;
  }
  if (plan.sb_size != 64 && plan.sb_size != 128) {
    error = "effective superblock size must be 64 or 128";
    return false;
  }
  if (!sequence_sb_size_ok(plan.sequence_sb_size)) {
    error = "sequence superblock size must be 64, 128, or 256";
    return false;
  }
  if (plan.sequence_sb_size == 256 && plan.sb_size != 128) {
    error = "a 256x256 sequence flag clamps an intra frame to 128x128";
    return false;
  }
  if (plan.sequence_sb_size != 256 && plan.sequence_sb_size != plan.sb_size) {
    error = "sequence and frame superblock sizes disagree";
    return false;
  }
  FrameGeometry geometry;
  geometry.width = plan.width;
  geometry.height = plan.height;
  geometry.sb_size = plan.sb_size;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;
  if (plan.mi_cols != partition::mi_cols(geometry) ||
      plan.mi_rows != partition::mi_rows(geometry)) {
    error = "MiCols/MiRows do not match the frame dimensions";
    return false;
  }
  if (plan.leaves != static_cast<int>(plan.blocks.size())) {
    error = "leaf count does not match the block list";
    return false;
  }

  const std::size_t cell_count = static_cast<std::size_t>(plan.mi_cols) *
                                 static_cast<std::size_t>(plan.mi_rows);
  std::vector<std::uint8_t> covered(cell_count, 0);
  std::uint32_t expected_luma_coeff = 0;
  std::uint32_t expected_u_coeff = 0;
  std::uint32_t expected_v_coeff = 0;
  const int sb_size_mi = plan.sb_size / 4;

  for (std::size_t i = 0; i < plan.blocks.size(); ++i) {
    const BlockPlan& block = plan.blocks[i];
    if (block.index != i) {
      error = "block index is not its position in the syntax-order list";
      return false;
    }
    if (!partition::is_valid_block_size(
            static_cast<BlockSize>(block.block_size))) {
      error = "block has an invalid block size";
      return false;
    }
    const BlockSize size = static_cast<BlockSize>(block.block_size);
    const bool codes_luma = block.luma_w != 0 || block.luma_h != 0;
    if (codes_luma) {
      if (partition::block_width(size) != block.luma_w ||
          partition::block_height(size) != block.luma_h) {
        error = "luma dimensions do not match the block size";
        return false;
      }
    } else {
      if (block.luma_w != 0 || block.luma_h != 0 ||
          block.visible_luma_w != 0 || block.visible_luma_h != 0 ||
          block.luma.coeff_count != 0 || block.tree_type != 2) {
        error = "leaf without a luma payload is not a CHROMA_PART block";
        return false;
      }
    }
    if (block.x != block.mi_col * 4 || block.y != block.mi_row * 4) {
      error = "block origin is inconsistent";
      return false;
    }
    if (block.mi_row < 0 || block.mi_col < 0 || block.mi_row >= plan.mi_rows ||
        block.mi_col >= plan.mi_cols) {
      error = "block origin is outside the coded frame";
      return false;
    }
    if (block.sb_mi_row != (block.mi_row / sb_size_mi) * sb_size_mi ||
        block.sb_mi_col != (block.mi_col / sb_size_mi) * sb_size_mi) {
      error = "block does not belong to its stored superblock";
      return false;
    }
    // max_partition_depth is a search cap, not a structural rule.  A partial
    // superblock can force (or boundary-relax) the tree below the cap because
    // the spec forbids PARTITION_NONE there; only enforce the cap inside fully
    // coded superblocks.
    const bool partial_sb =
        block.sb_mi_row + sb_size_mi > plan.mi_rows ||
        block.sb_mi_col + sb_size_mi > plan.mi_cols;
    if (block.depth > plan.options.max_partition_depth && !partial_sb &&
        !plan.options.sub8 && !plan.options.sdp) {
      error = "block depth exceeds max_partition_depth";
      return false;
    }
    if (codes_luma && plan.options.square_only &&
        block.luma_w != block.luma_h) {
      error = "square-only plan contains a rectangular block";
      return false;
    }
    if (codes_luma && plan.options.leaf32 &&
        (block.luma_w > 32 || block.luma_h > 32)) {
      error = "leaf32 plan contains a block larger than 32";
      return false;
    }

    if (block.visible_luma_w !=
            clipped_extent(block.luma_w, plan.width, block.x) ||
        block.visible_luma_h !=
            clipped_extent(block.luma_h, plan.height, block.y)) {
      error = "visible block extents are inconsistent";
      return false;
    }

    const auto check_tx = [&](const PlaneTxPlan& tx, int coded_w, int coded_h,
                              std::uint32_t expected_offset,
                              std::uint32_t& next_offset) {
      const auto params =
          txv::make_tx_params(coded_w, coded_h, txv::TxType::kDctDct);
      if (!params.has_value()) return false;
      if (tx.coded_w != coded_w || tx.coded_h != coded_h ||
          tx.tx_size != static_cast<int>(params->tx_size) ||
          tx.adj_w != static_cast<int>(params->w) ||
          tx.adj_h != static_cast<int>(params->h) ||
          tx.tx_type != static_cast<std::uint8_t>(params->tx_type) ||
          tx.coeff_count !=
              static_cast<std::uint32_t>(tx.adj_w * tx.adj_h) ||
          tx.coeff_offset != expected_offset) {
        return false;
      }
      next_offset = expected_offset + tx.coeff_count;
      return true;
    };
    if ((codes_luma &&
         !check_tx(block.luma, block.luma_w, block.luma_h,
                   expected_luma_coeff, expected_luma_coeff)) ||
        (block.has_chroma &&
         (!check_tx(block.u, block.chroma_w, block.chroma_h,
                    expected_u_coeff, expected_u_coeff) ||
          !check_tx(block.v, block.chroma_w, block.chroma_h,
                    expected_v_coeff, expected_v_coeff)))) {
      error = "transform plan or coefficient offsets are inconsistent";
      return false;
    }

    const int w4 = block.luma_w / 4;
    const int h4 = block.luma_h / 4;
    if (block.luma_avail.num4_above_right > w4 ||
        block.luma_avail.num4_below_left > h4) {
      error = "luma availability count exceeds the block size";
      return false;
    }
    if (block.has_chroma) {
      if (block.chroma_mi_row < 0 || block.chroma_mi_col < 0 ||
          block.chroma_mi_row >= plan.mi_rows ||
          block.chroma_mi_col >= plan.mi_cols) {
        error = "owned chroma block origin is outside the coded frame";
        return false;
      }
      const BlockSize chroma_size =
          partition::chroma_plane_residual_size(
              geometry, static_cast<BlockSize>(block.chroma_mi_size));
      if (!partition::is_valid_block_size(chroma_size)) {
        error = "block has an invalid chroma residual size";
        return false;
      }
      if (partition::block_width(chroma_size) != block.chroma_w ||
          partition::block_height(chroma_size) != block.chroma_h) {
        error = "chroma dimensions do not match the residual size rules";
        return false;
      }
      if (block.chroma_x != block.chroma_mi_col * 2 ||
          block.chroma_y != block.chroma_mi_row * 2 ||
          block.visible_chroma_w != clipped_extent(block.chroma_w,
                                                   plan.chroma_width,
                                                   block.chroma_x) ||
          block.visible_chroma_h != clipped_extent(block.chroma_h,
                                                   plan.chroma_height,
                                                   block.chroma_y)) {
        error = "owned chroma block origin or visible extent is inconsistent";
        return false;
      }
      const int cw4 = block.chroma_w / 4;
      const int ch4 = block.chroma_h / 4;
      if (block.chroma_avail.num4_above_right > cw4 ||
          block.chroma_avail.num4_below_left > ch4) {
        error = "chroma availability count exceeds the block size";
        return false;
      }
    } else {
      if (block.chroma_w != 0 || block.chroma_h != 0 ||
          block.chroma_x != 0 || block.chroma_y != 0 ||
          block.visible_chroma_w != 0 || block.visible_chroma_h != 0 ||
          block.u.coeff_count != 0 || block.v.coeff_count != 0) {
        error = "non-owning leaf carries a chroma payload";
        return false;
      }
    }

    const int row_end = codes_luma
                            ? std::min(block.mi_row + h4, plan.mi_rows)
                            : 0;
    const int col_end = codes_luma
                            ? std::min(block.mi_col + w4, plan.mi_cols)
                            : 0;
    for (int row = block.mi_row; row < row_end; ++row) {
      for (int col = block.mi_col; col < col_end; ++col) {
        const std::size_t idx = static_cast<std::size_t>(row) *
                                    static_cast<std::size_t>(plan.mi_cols) +
                                static_cast<std::size_t>(col);
        if (covered[idx] != 0) {
          error = "overlapping leaf blocks";
          return false;
        }
        covered[idx] = 1;
      }
    }
  }

  for (std::size_t i = 0; i < covered.size(); ++i) {
    if (covered[i] == 0) {
      error = "block list does not cover the coded frame";
      return false;
    }
  }

  // Chroma ownership must cover the half-resolution plane exactly once.  A
  // sub-8 or extended-partition leaf that does not own the inherited block
  // codes no chroma, so it does not fill cells here.
  const int chroma_mi_cols = plan.mi_cols / 2;
  const int chroma_mi_rows = plan.mi_rows / 2;
  std::vector<std::uint8_t> chroma_covered(
      static_cast<std::size_t>(chroma_mi_cols) *
          static_cast<std::size_t>(chroma_mi_rows),
      0);
  bool any_owner = false;
  for (const BlockPlan& block : plan.blocks) {
    if (!block.has_chroma) continue;
    any_owner = true;
    const int cw4 = block.chroma_w / 4;
    const int ch4 = block.chroma_h / 4;
    const int origin_row = block.chroma_mi_row / 2;
    const int origin_col = block.chroma_mi_col / 2;
    const int row_end = std::min(origin_row + ch4, chroma_mi_rows);
    const int col_end = std::min(origin_col + cw4, chroma_mi_cols);
    for (int row = origin_row; row < row_end; ++row) {
      for (int col = origin_col; col < col_end; ++col) {
        const std::size_t idx = static_cast<std::size_t>(row) *
                                    static_cast<std::size_t>(chroma_mi_cols) +
                                static_cast<std::size_t>(col);
        if (chroma_covered[idx] != 0) {
          error = "overlapping owned chroma blocks";
          return false;
        }
        chroma_covered[idx] = 1;
      }
    }
  }
  if (any_owner) {
    int uncovered = 0;
    for (std::size_t i = 0; i < chroma_covered.size(); ++i) {
      if (chroma_covered[i] == 0) ++uncovered;
    }
    if (uncovered != 0) {
      error = "owned chroma blocks do not cover the chroma plane (" +
              std::to_string(uncovered) + " cells of " +
              std::to_string(chroma_covered.size()) + ")";
      return false;
    }
  }

  if (plan.tree.size() != static_cast<std::size_t>(plan.nodes)) {
    error = "tree size does not match the decided node count";
    return false;
  }
  std::size_t tree_index = 0;
  std::uint32_t leaf_cursor = 0;
  int sb_index = 0;
  while (tree_index < plan.tree.size()) {
    const TreeNode& root = plan.tree[tree_index];
    if (!root.sb_root || root.depth != 0) {
      error = "tree preorder does not start a node at a superblock root";
      return false;
    }
    if (root.sb_index != static_cast<std::uint32_t>(sb_index)) {
      error = "tree superblock index is out of order";
      return false;
    }
    const int expect_row = (sb_index / plan.sb_cols) * sb_size_mi;
    const int expect_col = (sb_index % plan.sb_cols) * sb_size_mi;
    if (root.mi_row != expect_row || root.mi_col != expect_col) {
      error = "tree superblock origin is out of raster order";
      return false;
    }
    std::size_t next = 0;
    if (!validate_tree_node(plan, tree_index, leaf_cursor, next, error)) {
      return false;
    }
    if (next <= tree_index) {
      error = "tree preorder walk did not advance";
      return false;
    }
    tree_index = next;
    ++sb_index;
  }
  if (sb_index != plan.sb_cols * plan.sb_rows) {
    error = "tree does not contain every superblock";
    return false;
  }
  if (leaf_cursor != plan.blocks.size()) {
    error = "tree leaves do not cover the block list";
    return false;
  }

  if (expected_luma_coeff != plan.luma_coeff_count ||
      expected_u_coeff != plan.u_coeff_count ||
      expected_v_coeff != plan.v_coeff_count) {
    error = "plane coefficient totals do not match the block list";
    return false;
  }
  return true;
}

const char* block_size_name(int block_size) {
  return partition::block_size_name(static_cast<BlockSize>(block_size));
}

namespace {

void tsv_bool(std::ostringstream& out, bool value) { out << (value ? 1 : 0); }

void tsv_tx(std::ostringstream& out, const PlaneTxPlan& tx) {
  out << tx.tx_size << '\t' << tx.adj_w << '\t' << tx.adj_h << '\t'
      << tx.coeff_offset;
}

}  // namespace

std::string plan_to_tsv(const FramePlan& plan) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "# av2enc m1 FramePlan v1\n";
  out << "# width=" << plan.width << " height=" << plan.height
      << " chroma=" << plan.chroma_width << 'x' << plan.chroma_height
      << " mi=" << plan.mi_cols << 'x' << plan.mi_rows << " sb=" << plan.sb_size
      << " seq_sb=" << plan.sequence_sb_size
      << " leaves=" << plan.leaves << " nodes=" << plan.nodes << " cost="
      << std::fixed << std::setprecision(6) << plan.cost << '\n';
  out << "index\tsb_index\tsb_mi_row\tsb_mi_col\tmi_row\tmi_col\tsize\tdepth"
         "\tluma_w\tluma_h\tvisible_luma_w\tvisible_luma_h\tchroma_w\t"
         "chroma_h\tvisible_chroma_w\tvisible_chroma_h\thas_chroma\t"
         "chroma_offset\tluma_tx_size\tluma_adj_w\tluma_adj_h\t"
         "luma_coeff_offset\tu_tx_size\tu_adj_w\tu_adj_h\tu_coeff_offset\t"
         "v_tx_size\tv_adj_w\tv_adj_h\tv_coeff_offset\tabove\tleft\t"
         "above_right\tbelow_left\tnum4_above_right\tnum4_below_left\t"
         "chroma_above\tchroma_left\tchroma_above_right\tchroma_below_left\t"
         "chroma_num4_above_right\tchroma_num4_below_left\t"
         "chroma_mi_row\tchroma_mi_col\tchroma_mi_size\ttree_type\n";
  for (const BlockPlan& block : plan.blocks) {
    out << block.index << '\t' << block.sb_index << '\t' << block.sb_mi_row
        << '\t' << block.sb_mi_col << '\t' << block.mi_row << '\t'
        << block.mi_col << '\t' << block_size_name(block.block_size) << '\t'
        << static_cast<int>(block.depth) << '\t' << block.luma_w << '\t'
        << block.luma_h << '\t' << block.visible_luma_w << '\t'
        << block.visible_luma_h << '\t' << block.chroma_w << '\t'
        << block.chroma_h << '\t' << block.visible_chroma_w << '\t'
        << block.visible_chroma_h << '\t';
    tsv_bool(out, block.has_chroma);
    out << '\t';
    tsv_bool(out, block.chroma_offset);
    out << '\t';
    tsv_tx(out, block.luma);
    out << '\t';
    tsv_tx(out, block.u);
    out << '\t';
    tsv_tx(out, block.v);
    out << '\t';
    tsv_bool(out, block.luma_avail.above);
    out << '\t';
    tsv_bool(out, block.luma_avail.left);
    out << '\t';
    tsv_bool(out, block.luma_avail.above_right);
    out << '\t';
    tsv_bool(out, block.luma_avail.below_left);
    out << '\t' << static_cast<int>(block.luma_avail.num4_above_right) << '\t'
        << static_cast<int>(block.luma_avail.num4_below_left) << '\t';
    tsv_bool(out, block.chroma_avail.above);
    out << '\t';
    tsv_bool(out, block.chroma_avail.left);
    out << '\t';
    tsv_bool(out, block.chroma_avail.above_right);
    out << '\t';
    tsv_bool(out, block.chroma_avail.below_left);
    out << '\t' << static_cast<int>(block.chroma_avail.num4_above_right)
        << '\t' << static_cast<int>(block.chroma_avail.num4_below_left)
        << '\t' << block.chroma_mi_row << '\t' << block.chroma_mi_col << '\t'
        << static_cast<int>(block.chroma_mi_size) << '\t'
        << static_cast<int>(block.tree_type) << '\n';
  }
  return out.str();
}

std::string plan_tree_to_tsv(const FramePlan& plan) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "# av2enc m1 FramePlan tree v1\n";
  out << "# width=" << plan.width << " height=" << plan.height
      << " mi=" << plan.mi_cols << 'x' << plan.mi_rows << " sb=" << plan.sb_size
      << " seq_sb=" << plan.sequence_sb_size
      << " tree_nodes=" << plan.tree.size() << " leaves=" << plan.leaves
      << '\n';
  out << "index\tsb_index\tmi_row\tmi_col\tsize\tpartition\tpartition_id\t"
         "depth\tallowed_mask\tforced\tsb_root\tchild_count\tfirst_child\t"
         "subtree\tleaf_index\text_partition\text_partition_id\t"
         "allowed_mask_ext\ttree_type\tsdp_split\tsdp_chroma_block\tcost\n";
  for (const TreeNode& node : plan.tree) {
    out << node.index << '\t' << node.sb_index << '\t' << node.mi_row << '\t'
        << node.mi_col << '\t' << block_size_name(node.block_size) << '\t'
        << partition::partition_name(
               static_cast<PartitionType>(node.partition))
        << '\t' << static_cast<int>(node.partition) << '\t'
        << static_cast<int>(node.depth) << '\t'
        << static_cast<int>(node.allowed_mask) << '\t';
    tsv_bool(out, node.forced);
    out << '\t';
    tsv_bool(out, node.sb_root);
    out << '\t' << static_cast<int>(node.child_count) << '\t';
    if (node.first_child == kNoTreeNode) {
      out << -1;
    } else {
      out << node.first_child;
    }
    out << '\t' << node.subtree << '\t';
    if (node.leaf_index == kNoTreeNode) {
      out << -1;
    } else {
      out << node.leaf_index;
    }
    out << '\t'
        << partition_ext::partition_name(
               static_cast<partition_ext::PartitionType>(
                   node.ext_partition))
        << '\t' << static_cast<int>(node.ext_partition) << '\t'
        << static_cast<int>(node.allowed_mask_ext) << '\t'
        << static_cast<int>(node.tree_type) << '\t';
    tsv_bool(out, node.sdp_split);
    out << '\t';
    tsv_bool(out, node.sdp_chroma_block);
    out << '\t' << std::setprecision(6) << node.cost << '\n';
  }
  return out.str();
}

std::string plan_to_json(const FramePlan& plan) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "{\"kind\":\"av2enc-m1-frame-plan\",\"version\":1";
  out << ",\"width\":" << plan.width << ",\"height\":" << plan.height;
  out << ",\"chroma_width\":" << plan.chroma_width
      << ",\"chroma_height\":" << plan.chroma_height;
  out << ",\"mi_cols\":" << plan.mi_cols << ",\"mi_rows\":" << plan.mi_rows;
  out << ",\"sb_size\":" << plan.sb_size << ",\"sb_cols\":" << plan.sb_cols
      << ",\"sb_rows\":" << plan.sb_rows;
  out << ",\"sequence_sb_size\":" << plan.sequence_sb_size;
  out << ",\"coded_luma_w\":" << plan.coded_luma_w
      << ",\"coded_luma_h\":" << plan.coded_luma_h
      << ",\"coded_chroma_w\":" << plan.coded_chroma_w
      << ",\"coded_chroma_h\":" << plan.coded_chroma_h;
  out << ",\"luma_stride\":" << plan.luma_stride
      << ",\"luma_padded_height\":" << plan.luma_padded_height
      << ",\"chroma_stride\":" << plan.chroma_stride
      << ",\"chroma_padded_height\":" << plan.chroma_padded_height;
  out << ",\"leaves\":" << plan.leaves << ",\"nodes\":" << plan.nodes;
  out << ",\"tree_nodes\":" << plan.tree.size();
  out << ",\"cost\":" << std::setprecision(17) << plan.cost;
  out << ",\"luma_coeff_count\":" << plan.luma_coeff_count
      << ",\"u_coeff_count\":" << plan.u_coeff_count
      << ",\"v_coeff_count\":" << plan.v_coeff_count;
  out << ",\"options\":{\"sb_size\":" << plan.options.sb_size
      << ",\"base_q_idx\":"
      << static_cast<int>(plan.options.base_q_idx)
      << ",\"min_block_size\":" << plan.options.min_block_size
      << ",\"max_partition_depth\":" << plan.options.max_partition_depth
      << ",\"force_single_block\":"
      << (plan.options.force_single_block ? "true" : "false")
      << ",\"sub8\":" << (plan.options.sub8 ? "true" : "false")
      << ",\"enable_ext_partitions\":"
      << (plan.options.enable_ext_partitions ? "true" : "false")
      << ",\"enable_uneven_4way_partitions\":"
      << (plan.options.enable_uneven_4way_partitions ? "true" : "false")
      << ",\"sdp\":" << (plan.options.sdp ? "true" : "false")
      << ",\"square_only\":" << (plan.options.square_only ? "true" : "false")
      << ",\"leaf32\":" << (plan.options.leaf32 ? "true" : "false") << '}';

  const auto tx_json = [&](const PlaneTxPlan& tx) {
    out << "{\"coded_w\":" << tx.coded_w << ",\"coded_h\":" << tx.coded_h
        << ",\"tx_size\":" << tx.tx_size << ",\"adj_w\":" << tx.adj_w
        << ",\"adj_h\":" << tx.adj_h
        << ",\"tx_type\":" << static_cast<int>(tx.tx_type)
        << ",\"coeff_offset\":" << tx.coeff_offset
        << ",\"coeff_count\":" << tx.coeff_count << '}';
  };
  const auto avail_json = [&](const NeighborAvailability& avail) {
    out << "{\"above\":" << (avail.above ? "true" : "false")
        << ",\"left\":" << (avail.left ? "true" : "false")
        << ",\"above_left\":" << (avail.above_left ? "true" : "false")
        << ",\"above_right\":" << (avail.above_right ? "true" : "false")
        << ",\"below_left\":" << (avail.below_left ? "true" : "false")
        << ",\"num4_above_right\":"
        << static_cast<int>(avail.num4_above_right)
        << ",\"num4_below_left\":"
        << static_cast<int>(avail.num4_below_left) << '}';
  };

  out << ",\"blocks\":[";
  for (std::size_t i = 0; i < plan.blocks.size(); ++i) {
    const BlockPlan& block = plan.blocks[i];
    if (i != 0) out << ',';
    out << "{\"index\":" << block.index << ",\"sb_index\":" << block.sb_index
        << ",\"sb_mi_row\":" << block.sb_mi_row
        << ",\"sb_mi_col\":" << block.sb_mi_col
        << ",\"mi_row\":" << block.mi_row << ",\"mi_col\":" << block.mi_col
        << ",\"x\":" << block.x << ",\"y\":" << block.y
        << ",\"chroma_x\":" << block.chroma_x
        << ",\"chroma_y\":" << block.chroma_y << ",\"size\":\""
        << block_size_name(block.block_size) << '"'
        << ",\"size_enum\":" << static_cast<int>(block.block_size)
        << ",\"depth\":" << static_cast<int>(block.depth)
        << ",\"luma_w\":" << block.luma_w << ",\"luma_h\":" << block.luma_h
        << ",\"visible_luma_w\":" << block.visible_luma_w
        << ",\"visible_luma_h\":" << block.visible_luma_h
        << ",\"chroma_w\":" << block.chroma_w
        << ",\"chroma_h\":" << block.chroma_h
        << ",\"visible_chroma_w\":" << block.visible_chroma_w
        << ",\"visible_chroma_h\":" << block.visible_chroma_h
        << ",\"has_chroma\":" << (block.has_chroma ? "true" : "false")
        << ",\"chroma_offset\":" << (block.chroma_offset ? "true" : "false")
        << ",\"chroma_mi_row\":" << block.chroma_mi_row
        << ",\"chroma_mi_col\":" << block.chroma_mi_col
        << ",\"chroma_mi_size\":" << static_cast<int>(block.chroma_mi_size)
        << ",\"tree_type\":" << static_cast<int>(block.tree_type)
        << ",\"luma\":";
    tx_json(block.luma);
    out << ",\"u\":";
    tx_json(block.u);
    out << ",\"v\":";
    tx_json(block.v);
    out << ",\"luma_avail\":";
    avail_json(block.luma_avail);
    out << ",\"chroma_avail\":";
    avail_json(block.chroma_avail);
    out << '}';
  }
  out << "],\"tree\":[";
  for (std::size_t i = 0; i < plan.tree.size(); ++i) {
    const TreeNode& node = plan.tree[i];
    if (i != 0) out << ',';
    out << "{\"index\":" << node.index << ",\"sb_index\":" << node.sb_index
        << ",\"mi_row\":" << node.mi_row << ",\"mi_col\":" << node.mi_col
        << ",\"size\":\"" << block_size_name(node.block_size) << '"'
        << ",\"size_enum\":" << static_cast<int>(node.block_size)
        << ",\"partition\":\""
        << partition::partition_name(
               static_cast<PartitionType>(node.partition))
        << '"' << ",\"partition_id\":" << static_cast<int>(node.partition)
        << ",\"depth\":" << static_cast<int>(node.depth)
        << ",\"allowed_mask\":" << static_cast<int>(node.allowed_mask)
        << ",\"forced\":" << (node.forced ? "true" : "false")
        << ",\"sb_root\":" << (node.sb_root ? "true" : "false")
        << ",\"child_count\":" << static_cast<int>(node.child_count)
        << ",\"first_child\":"
        << (node.first_child == kNoTreeNode
                ? -1
                : static_cast<int>(node.first_child))
        << ",\"subtree\":" << node.subtree << ",\"leaf_index\":"
        << (node.leaf_index == kNoTreeNode
                ? -1
                : static_cast<int>(node.leaf_index))
        << ",\"ext_partition\":\""
        << partition_ext::partition_name(
               static_cast<partition_ext::PartitionType>(node.ext_partition))
        << '"' << ",\"ext_partition_id\":"
        << static_cast<int>(node.ext_partition)
        << ",\"allowed_mask_ext\":"
        << static_cast<int>(node.allowed_mask_ext)
        << ",\"tree_type\":" << static_cast<int>(node.tree_type)
        << ",\"sdp_split\":" << (node.sdp_split ? "true" : "false")
        << ",\"sdp_chroma_block\":"
        << (node.sdp_chroma_block ? "true" : "false")
        << ",\"cost\":" << std::setprecision(17) << node.cost << '}';
  }
  out << "]}\n";
  return out.str();
}

}  // namespace av2enc::still::plan
