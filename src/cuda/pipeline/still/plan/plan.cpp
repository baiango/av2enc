// M1 GPU host-side frame/block plan builder.  See plan.h for the contract.

#include "cuda/pipeline/still/plan/plan.h"

#include "codec/partition/tree/partition.h"
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

// Appends `node` and its whole subtree to the preorder tree array.
void append_tree_node(const partition::PartitionNode& node,
                      std::uint32_t sb_index, FramePlan& plan,
                      std::uint32_t& leaf_cursor) {
  const std::uint32_t index = static_cast<std::uint32_t>(plan.tree.size());
  TreeNode entry;
  entry.index = index;
  entry.sb_index = sb_index;
  entry.mi_row = node.mi_row;
  entry.mi_col = node.mi_col;
  entry.block_size = static_cast<std::uint8_t>(node.size);
  entry.partition = static_cast<std::uint8_t>(node.partition);
  entry.depth = static_cast<std::uint8_t>(node.depth);
  std::uint8_t mask = 0;
  for (int p = 0; p < partition::kPartitionTypeCount; ++p) {
    if (node.allowed.allowed[p]) {
      mask = static_cast<std::uint8_t>(mask | (1U << p));
    }
  }
  entry.allowed_mask = mask;
  entry.forced = node.forced;
  entry.sb_root = node.depth == 0;
  const bool leaf = node.partition == partition::PartitionType::kNone;
  entry.first_child = leaf ? kNoTreeNode : index + 1;
  plan.tree.push_back(entry);

  if (leaf) {
    plan.tree[index].leaf_index = leaf_cursor;
    ++leaf_cursor;
    return;
  }
  std::uint8_t child_count = 0;
  for (const partition::PartitionNode& child : node.children) {
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

bool build_frame_plan(int width, int height, std::span<const std::uint8_t> luma,
                      const PlanOptions& options, FramePlan& plan,
                      std::string& error) {
  error.clear();
  plan = FramePlan{};
  if (width <= 0 || height <= 0) {
    error = "frame dimensions must be positive";
    return false;
  }
  if (options.sb_size != 64) {
    error = "M1 frame plan supports a 64x64 superblock only";
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
  const std::size_t luma_size = static_cast<std::size_t>(width) *
                                static_cast<std::size_t>(height);
  if (luma.size() < luma_size) {
    error = "luma plane is smaller than width*height";
    return false;
  }

  FrameGeometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = options.sb_size;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;

  std::string geometry_error;
  if (!partition::validate_geometry(geometry, geometry_error)) {
    error = geometry_error;
    return false;
  }

  PlanOptions resolved = options;
  partition::DecisionOptions poptions;
  poptions.min_block_size_px = resolved.min_block_size;
  poptions.max_depth = resolved.max_partition_depth;
  poptions.allow_rect = resolved.allow_rect;
  poptions.allow_split = resolved.allow_split;
  if (resolved.force_single_block) {
    poptions.max_depth = 0;
    poptions.min_block_size_px = 64;
  }

  // Same lambda / signal cost as the CPU M1 encoder.
  const double lambda =
      0.85 * std::pow(2.0, (static_cast<int>(resolved.base_q_idx) - 12) / 6.0);
  const double signal_cost = lambda * 2.0;

  const auto cost_fn = [&](const partition::PartitionNodeInfo& info) {
    partition::NodeCosts costs;
    costs.leaf = static_cast<double>(
        block_variance_cost_at(geometry, luma, info.mi_row, info.mi_col,
                               info.size));
    const int bw = partition::block_width(info.size);
    const int bh = partition::block_height(info.size);
    if (resolved.square_only && bw != bh) {
      costs.leaf += 1.0e18;
    }
    if (resolved.leaf32 && (bw > 32 || bh > 32)) {
      costs.leaf += 1.0e18;
    }
    costs.none_signal = 0.0;
    costs.split_signal = signal_cost;
    costs.horz_signal = signal_cost;
    costs.vert_signal = signal_cost;
    return costs;
  };

  const partition::FrameDecision frame =
      partition::decide_frame(geometry, cost_fn, poptions);
  if (frame.status != partition::DecisionStatus::kOk) {
    error = "partition decision failed: " + frame.error;
    return false;
  }
  if (!partition::validate_frame(geometry, frame, error)) {
    error = "partition tree invalid: " + error;
    return false;
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
  plan.sb_size = options.sb_size;
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
  plan.leaves = frame.leaves;
  plan.nodes = frame.nodes;
  plan.cost = frame.cost;
  plan.options = resolved;
  plan.blocks.reserve(static_cast<std::size_t>(frame.leaves));

  // Superblock-local decoded maps, reset exactly like clear_block_decoded_flags.
  DecodedMap decoded[3];
  for (int plane = 0; plane < 3; ++plane) {
    decoded[plane].reset(plane == 0 ? mi_cols_v : mi_cols_v / 2,
                         plane == 0 ? mi_rows_v : mi_rows_v / 2);
  }

  std::vector<partition::LeafBlock> leaves;
  for (int sbi = 0; sbi < static_cast<int>(frame.superblocks.size()); ++sbi) {
    const partition::PartitionNode& sb =
        frame.superblocks[static_cast<std::size_t>(sbi)];
    for (int plane = 0; plane < 3; ++plane) {
      decoded[plane].reset(plane == 0 ? mi_cols_v : mi_cols_v / 2,
                           plane == 0 ? mi_rows_v : mi_rows_v / 2);
    }
    int sb_cols_plane[3] = {0, 0, 0};
    int sb_rows_plane[3] = {0, 0, 0};
    sb_cols_plane[0] = std::min(16, mi_cols_v - sb.mi_col);
    sb_rows_plane[0] = std::min(16, mi_rows_v - sb.mi_row);
    sb_cols_plane[1] = std::min(8, mi_cols_v / 2 - sb.mi_col / 2);
    sb_rows_plane[1] = std::min(8, mi_rows_v / 2 - sb.mi_row / 2);
    sb_cols_plane[2] = sb_cols_plane[1];
    sb_rows_plane[2] = sb_rows_plane[1];

    leaves.clear();
    partition::collect_leaves(sb, leaves);

    for (const partition::LeafBlock& leaf : leaves) {
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
      block.block_size = static_cast<std::uint8_t>(leaf.size);
      block.depth = static_cast<std::uint8_t>(leaf.depth);
      block.has_chroma = leaf.has_chroma;
      block.chroma_offset = leaf.chroma_offset;
      block.luma_w = partition::block_width(leaf.size);
      block.luma_h = partition::block_height(leaf.size);
      block.visible_luma_w =
          clipped_extent(block.luma_w, width, block.x);
      block.visible_luma_h =
          clipped_extent(block.luma_h, height, block.y);

      const BlockSize chroma_size =
          partition::chroma_plane_residual_size(geometry, leaf.size);
      if (!partition::is_valid_block_size(chroma_size)) {
        error = "invalid chroma residual size";
        return false;
      }
      block.chroma_w = partition::block_width(chroma_size);
      block.chroma_h = partition::block_height(chroma_size);
      block.visible_chroma_w =
          clipped_extent(block.chroma_w, chroma_width, block.chroma_x);
      block.visible_chroma_h =
          clipped_extent(block.chroma_h, chroma_height, block.chroma_y);

      if (!make_plane_tx(block.luma_w, block.luma_h, "luma", block.luma,
                         error)) {
        return false;
      }
      if (!make_plane_tx(block.chroma_w, block.chroma_h, "U", block.u,
                         error)) {
        return false;
      }
      if (!make_plane_tx(block.chroma_w, block.chroma_h, "V", block.v,
                         error)) {
        return false;
      }

      block.luma.coeff_offset = plan.luma_coeff_count;
      block.u.coeff_offset = plan.u_coeff_count;
      block.v.coeff_offset = plan.v_coeff_count;

      const int luma_w4 = block.luma_w / 4;
      const int luma_h4 = block.luma_h / 4;
      const int chroma_w4 = block.chroma_w / 4;
      const int chroma_h4 = block.chroma_h / 4;
      block.luma_avail = compute_availability(
          0, block.x, block.y, block.mi_col, block.mi_row, luma_w4, luma_h4,
          sb.mi_row, sb.mi_col, sb_cols_plane[0], sb_rows_plane[0], mi_cols_v,
          decoded[0]);
      block.chroma_avail = compute_availability(
          1, block.chroma_x, block.chroma_y, block.chroma_x >> 2,
          block.chroma_y >> 2, chroma_w4, chroma_h4, sb.mi_row, sb.mi_col,
          sb_cols_plane[1], sb_rows_plane[1], mi_cols_v, decoded[1]);

      // Coefficient layout: one contiguous per-plane buffer.
      plan.luma_coeff_count += block.luma.coeff_count;
      plan.u_coeff_count += block.u.coeff_count;
      plan.v_coeff_count += block.v.coeff_count;

      // Fill decoded maps after the block, exactly like encode_block().
      decoded[0].fill(block.mi_col, block.mi_row, luma_w4, luma_h4);
      decoded[1].fill(block.mi_col / 2, block.mi_row / 2, chroma_w4, chroma_h4);
      decoded[2].fill(block.mi_col / 2, block.mi_row / 2, chroma_w4, chroma_h4);

      plan.blocks.push_back(block);
    }
  }

  // Preorder tree view.  Leaves are assigned in the same order as `blocks`.
  plan.tree.reserve(static_cast<std::size_t>(frame.nodes));
  std::uint32_t leaf_cursor = 0;
  for (int sbi = 0; sbi < static_cast<int>(frame.superblocks.size()); ++sbi) {
    append_tree_node(frame.superblocks[static_cast<std::size_t>(sbi)],
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
  if (node.sb_root != (node.depth == 0)) {
    error = "tree node superblock-root flag disagrees with its depth";
    return false;
  }
  if (node.partition ==
      static_cast<std::uint8_t>(partition::PartitionType::kNone)) {
    if (node.child_count != 0 || node.first_child != kNoTreeNode ||
        node.subtree != 0) {
      error = "tree leaf has child metadata";
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
        block.block_size != node.block_size) {
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
  if (plan.sb_size != 64) {
    error = "superblock size must be 64";
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
    if (partition::block_width(size) != block.luma_w ||
        partition::block_height(size) != block.luma_h) {
      error = "luma dimensions do not match the block size";
      return false;
    }
    if (block.x != block.mi_col * 4 || block.y != block.mi_row * 4 ||
        block.chroma_x != block.x / 2 || block.chroma_y != block.y / 2) {
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
    if (block.depth > plan.options.max_partition_depth && !partial_sb) {
      error = "block depth exceeds max_partition_depth";
      return false;
    }
    if (plan.options.square_only && block.luma_w != block.luma_h) {
      error = "square-only plan contains a rectangular block";
      return false;
    }
    if (plan.options.leaf32 && (block.luma_w > 32 || block.luma_h > 32)) {
      error = "leaf32 plan contains a block larger than 32";
      return false;
    }

    const BlockSize chroma_size =
        partition::chroma_plane_residual_size(geometry, size);
    if (!partition::is_valid_block_size(chroma_size)) {
      error = "block has an invalid chroma residual size";
      return false;
    }
    if (partition::block_width(chroma_size) != block.chroma_w ||
        partition::block_height(chroma_size) != block.chroma_h) {
      error = "chroma dimensions do not match the residual size rules";
      return false;
    }
    if (block.visible_luma_w !=
            clipped_extent(block.luma_w, plan.width, block.x) ||
        block.visible_luma_h !=
            clipped_extent(block.luma_h, plan.height, block.y) ||
        block.visible_chroma_w != clipped_extent(block.chroma_w,
                                                 plan.chroma_width,
                                                 block.chroma_x) ||
        block.visible_chroma_h != clipped_extent(block.chroma_h,
                                                 plan.chroma_height,
                                                 block.chroma_y)) {
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
    if (!check_tx(block.luma, block.luma_w, block.luma_h, expected_luma_coeff,
                  expected_luma_coeff) ||
        !check_tx(block.u, block.chroma_w, block.chroma_h, expected_u_coeff,
                  expected_u_coeff) ||
        !check_tx(block.v, block.chroma_w, block.chroma_h, expected_v_coeff,
                  expected_v_coeff)) {
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
    const int cw4 = block.chroma_w / 4;
    const int ch4 = block.chroma_h / 4;
    if (block.chroma_avail.num4_above_right > cw4 ||
        block.chroma_avail.num4_below_left > ch4) {
      error = "chroma availability count exceeds the block size";
      return false;
    }

    const int row_end = std::min(block.mi_row + h4, plan.mi_rows);
    const int col_end = std::min(block.mi_col + w4, plan.mi_cols);
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
         "chroma_num4_above_right\tchroma_num4_below_left\n";
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
        << '\n';
  }
  return out.str();
}

std::string plan_tree_to_tsv(const FramePlan& plan) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "# av2enc m1 FramePlan tree v1\n";
  out << "# width=" << plan.width << " height=" << plan.height
      << " mi=" << plan.mi_cols << 'x' << plan.mi_rows << " sb=" << plan.sb_size
      << " tree_nodes=" << plan.tree.size() << " leaves=" << plan.leaves
      << '\n';
  out << "index\tsb_index\tmi_row\tmi_col\tsize\tpartition\tpartition_id\t"
         "depth\tallowed_mask\tforced\tsb_root\tchild_count\tfirst_child\t"
         "subtree\tleaf_index\n";
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
    out << '\n';
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
  out << ",\"options\":{\"base_q_idx\":"
      << static_cast<int>(plan.options.base_q_idx)
      << ",\"min_block_size\":" << plan.options.min_block_size
      << ",\"max_partition_depth\":" << plan.options.max_partition_depth
      << ",\"force_single_block\":"
      << (plan.options.force_single_block ? "true" : "false")
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
        << '}';
  }
  out << "]}\n";
  return out.str();
}

}  // namespace av2enc::still::plan
