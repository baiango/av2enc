// M1 GPU -> syntax bridge.  See still_stream_bridge.h.
#include "pipeline/still/stream/still_stream_bridge.h"

#include "cuda/pipeline/still/plan/plan.h"
#include "pipeline/still/still_tables.h"
#include "codec/partition/ext/partition_ext.h"
#include "codec/partition/tree/partition.h"
#include "codec/transform/variants/tx_variants.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::still::stream {
namespace {

using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionType;

constexpr int kMaxTreeDepth = 32;

// Neutral frame geometry: both POD sets (cuda::still_gpu and av2enc::still::plan) map onto it.
struct FrameInfo {
  int width = 0;
  int height = 0;
  int sb_size = 64;
  std::uint32_t coeff_count[3] = {0, 0, 0};
  // Optional filter decision grids (see FilterGrids).
  const std::int8_t* cdef_idx = nullptr;
  int cdef_n64_cols = 0;
  int cdef_n64_rows = 0;
  const std::uint8_t* ccso_blks[3] = {nullptr, nullptr, nullptr};
  int ccso_blk_cols[3] = {0, 0, 0};
  int ccso_blk_rows[3] = {0, 0, 0};
};

// What the packer needs per leaf, independent of which POD set described it.
struct LeafView {
  int mi_row = 0;
  int mi_col = 0;
  int bsize = 0;
  int has_chroma = 0;
  int chroma_offset = 0;
  int chroma_mi_row = 0;
  int chroma_mi_col = 0;
  int chroma_bsize = 0;
  int tree_type = 0;
  int luma_w = 0;
  int luma_h = 0;
  int chroma_w = 0;
  int chroma_h = 0;
  int tx_w = 0;
  int tx_h = 0;
  int chroma_tx_w = 0;
  int chroma_tx_h = 0;
  std::uint32_t coeff_offset[3] = {0, 0, 0};
  int eob[3] = {0, 0, 0};
  int y_mode = 0;
  int y_angle_delta = 0;
  int uv_mode = 0;
  int is_cfl = 0;
  int tx_type[3] = {0, 0, 0};
  // §5.20.8.2/§7.15.3 intra secondary transform (luma only).
  int sec_tx_type = 0;
  int stx_set = 0;
  int cctx_type = 0;
  // §5.20.5.5 MRL and §5.20.7.9 DIP decisions.
  int mrl_index = 0;
  int mrl_sec_index = 0;
  int use_dip = 0;
  int dip_mode = 0;
  int dip_transpose = 0;
  // §5.20.7.32 CfL signalling.
  int cfl_index = 1;
  int cfl_alpha_signs = 0;
  int cfl_alpha_u_syntax = 0;
  int cfl_alpha_v_syntax = 0;
  int cfl_mh_dir = 0;
  // §5.20.5.7 segment id from a per-SB SEG_LVL_ALT_Q plan (0 when off).
  int segment_id = 0;
  // §5.20.8 / §5.20.5.3-.4 screen-content decisions.
  int is_palette = 0;
  std::uint32_t palette_index = 0;
  int is_ibc = 0;
  int ibc_skip = 1;
  int ibc_mode = 1;
  int ibc_ref_mv_idx = 0;
  std::int32_t ibc_bv_row = 0;
  std::int32_t ibc_bv_col = 0;
};

FrameGeometry make_geometry(const FrameInfo& info) {
  FrameGeometry geometry;
  geometry.width = info.width;
  geometry.height = info.height;
  geometry.sb_size = info.sb_size != 0 ? info.sb_size : 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;
  return geometry;
}

// Spec-derived allowed/forced flags for one node, mirroring
// partition::solve_node() so a plan codes exactly the partition symbols the
// decoder expects.  `forced_partition` is -1 when the node codes syntax.
bool derive_node_flags(const FrameGeometry& geometry, int mi_row, int mi_col,
                       int bsize, bool has_chroma, bool chroma_offset,
                       std::uint8_t* mask, std::uint8_t* flags,
                       int* forced_partition, std::string& error) {
  const auto size = static_cast<BlockSize>(bsize);
  if (!partition::is_valid_block_size(size)) {
    error = "invalid block size in the partition tree";
    return false;
  }
  if (mi_row < 0 || mi_col < 0 || mi_row >= partition::mi_rows(geometry) ||
      mi_col >= partition::mi_cols(geometry)) {
    error = "partition tree node is outside the frame";
    return false;
  }
  const partition::PartitionBoundary boundary =
      partition::partition_implied(geometry, mi_row, mi_col, size);
  const partition::PartitionSet allowed = partition::init_allowed_partitions(
      geometry, mi_row, mi_col, size, has_chroma, chroma_offset);
  std::uint8_t m = 0;
  for (int p = 0; p < partition::kPartitionTypeCount; ++p) {
    if (allowed.allowed[p]) m = static_cast<std::uint8_t>(m | (1U << p));
  }
  int forced = -1;
  if (boundary.implied && allowed.is_allowed(boundary.implied_partition)) {
    forced = static_cast<int>(boundary.implied_partition);
  } else if (allowed.count == 1) {
    for (int p = 0; p < partition::kPartitionTypeCount; ++p) {
      if (allowed.allowed[p]) {
        forced = p;
        break;
      }
    }
  }
  *mask = m;
  *flags = forced >= 0 ? syntax::kFlagForced : 0;
  *forced_partition = forced;
  return true;
}

// Chroma recursion state for one child slot (§5.20.3.2 item 6,
// `enumerate_children`).  SPLIT keeps has_chroma and clears chromaOffset for
// all four children; HORZ/VERT propagate the offset to both, and the top child
// loses has_chroma when it is chroma-offset.
struct ChildState {
  bool has_chroma = true;
  bool chroma_offset = false;
};

ChildState child_state(const FrameGeometry& geometry, PartitionType partition,
                       BlockSize size, bool has_chroma, bool chroma_offset,
                       int child) {
  bool next = chroma_offset;
  if (geometry.has_chroma && !next) {
    next = partition::is_chroma_offset_for_partition(geometry, partition, size);
  }
  ChildState state;
  if (partition == PartitionType::kSplit) {
    state.has_chroma = has_chroma;
    state.chroma_offset = false;
    return state;
  }
  state.chroma_offset = next;
  state.has_chroma = (child == 0) ? (has_chroma && !next) : has_chroma;
  return state;
}

void adapt_gpu_leaf(const cuda::still_gpu::BlockPlan& block,
                    const cuda::still_gpu::BlockDecision& decision,
                    LeafView& leaf) {
  leaf.mi_row = block.mi_row;
  leaf.mi_col = block.mi_col;
  leaf.has_chroma = block.has_chroma;
  leaf.chroma_offset = block.chroma_offset;
  leaf.chroma_mi_row = block.chroma_mi_row;
  leaf.chroma_mi_col = block.chroma_mi_col;
  leaf.chroma_bsize = block.chroma_mi_size;
  leaf.tree_type = block.tree_type;
  leaf.luma_w = block.luma_w;
  leaf.luma_h = block.luma_h;
  leaf.chroma_w = block.chroma_w;
  leaf.chroma_h = block.chroma_h;
  leaf.tx_w = block.tx_w;
  leaf.tx_h = block.tx_h;
  leaf.chroma_tx_w = block.chroma_tx_w;
  leaf.chroma_tx_h = block.chroma_tx_h;
  leaf.coeff_offset[0] = block.coeff_offset_y;
  leaf.coeff_offset[1] = block.coeff_offset_u;
  leaf.coeff_offset[2] = block.coeff_offset_v;
  leaf.eob[0] = decision.eob_y;
  leaf.eob[1] = decision.eob_u;
  leaf.eob[2] = decision.eob_v;
  leaf.y_mode = decision.luma_mode;
  leaf.y_angle_delta = decision.luma_angle_delta;
  leaf.uv_mode = decision.uv_mode;
  leaf.is_cfl = decision.is_cfl;
  leaf.tx_type[0] = decision.tx_type_y;
  leaf.tx_type[1] = decision.tx_type_u;
  leaf.tx_type[2] = decision.tx_type_v;
  leaf.sec_tx_type = decision.sec_tx_type;
  leaf.stx_set = decision.stx_set;
  leaf.cctx_type = decision.cctx_type;
  leaf.mrl_index = decision.mrl_index;
  leaf.mrl_sec_index = decision.mrl_sec_index;
  leaf.use_dip = decision.use_dip;
  leaf.dip_mode = decision.dip_mode;
  leaf.dip_transpose = decision.dip_transpose;
  leaf.cfl_index = decision.cfl_index;
  leaf.cfl_alpha_signs = decision.cfl_alpha_signs;
  leaf.cfl_alpha_u_syntax = decision.cfl_alpha_u_syntax;
  leaf.cfl_alpha_v_syntax = decision.cfl_alpha_v_syntax;
  leaf.cfl_mh_dir = decision.cfl_mh_dir;
  leaf.segment_id = decision.segment_id;
  leaf.is_palette = decision.is_palette;
  leaf.palette_index = decision.palette_index;
  leaf.is_ibc = decision.is_ibc;
  leaf.ibc_skip = decision.ibc_skip;
  leaf.ibc_mode = decision.ibc_mode;
  leaf.ibc_ref_mv_idx = decision.ibc_ref_mv_idx;
  leaf.ibc_bv_row = decision.ibc_bv_row;
  leaf.ibc_bv_col = decision.ibc_bv_col;
}

bool adapt_plan_leaf(const plan::BlockPlan& block,
                     const cuda::still_gpu::BlockDecision& decision,
                     LeafView& leaf) {
  leaf.mi_row = block.mi_row;
  leaf.mi_col = block.mi_col;
  leaf.bsize = block.block_size;
  leaf.has_chroma = block.has_chroma ? 1 : 0;
  leaf.chroma_offset = block.chroma_offset ? 1 : 0;
  leaf.chroma_mi_row = block.chroma_mi_row;
  leaf.chroma_mi_col = block.chroma_mi_col;
  leaf.chroma_bsize = block.chroma_mi_size;
  leaf.tree_type = block.tree_type;
  leaf.luma_w = block.luma_w;
  leaf.luma_h = block.luma_h;
  leaf.chroma_w = block.chroma_w;
  leaf.chroma_h = block.chroma_h;
  leaf.tx_w = block.luma.adj_w;
  leaf.tx_h = block.luma.adj_h;
  leaf.chroma_tx_w = block.u.adj_w;
  leaf.chroma_tx_h = block.u.adj_h;
  leaf.coeff_offset[0] = block.luma.coeff_offset;
  leaf.coeff_offset[1] = block.u.coeff_offset;
  leaf.coeff_offset[2] = block.v.coeff_offset;
  leaf.eob[0] = decision.eob_y;
  leaf.eob[1] = decision.eob_u;
  leaf.eob[2] = decision.eob_v;
  leaf.y_mode = decision.luma_mode;
  leaf.y_angle_delta = decision.luma_angle_delta;
  leaf.uv_mode = decision.uv_mode;
  leaf.is_cfl = decision.is_cfl;
  leaf.tx_type[0] = decision.tx_type_y;
  leaf.tx_type[1] = decision.tx_type_u;
  leaf.tx_type[2] = decision.tx_type_v;
  leaf.segment_id = decision.segment_id;
  leaf.sec_tx_type = decision.sec_tx_type;
  leaf.stx_set = decision.stx_set;
  leaf.cctx_type = decision.cctx_type;
  return true;
}

class Packer {
 public:
  Packer(const FrameInfo& info, std::span<const LeafView> leaves,
         const GpuCoefficients& coeffs, PackedFrame& out)
      : info_(info),
        geometry_(make_geometry(info)),
        leaves_(leaves),
        coeffs_(coeffs),
        out_(out) {}

  bool run(std::span<const TreeNode> tree, std::string& error) {
    out_.blocks.clear();
    out_.levels.clear();
    out_.plan.width = info_.width;
    out_.plan.height = info_.height;
    out_.plan.sb_size = geometry_.sb_size;
    for (int plane = 0; plane < 3; ++plane) {
      out_.plan.ccso_blks[plane] = info_.ccso_blks[plane];
      out_.plan.ccso_blk_cols[plane] = info_.ccso_blk_cols[plane];
      out_.plan.ccso_blk_rows[plane] = info_.ccso_blk_rows[plane];
    }

    const int sb_mi = partition::sb_size_mi(geometry_);
    const int mi_cols = partition::mi_cols(geometry_);
    const int mi_rows = partition::mi_rows(geometry_);
    const int sb_cols = (mi_cols + sb_mi - 1) / sb_mi;
    const int sb_rows = (mi_rows + sb_mi - 1) / sb_mi;
    for (int plane = 0; plane < 3; ++plane) {
      plane_ranges_[plane].clear();
    }
    if (tree.empty()) {
      error = "partition tree is empty";
      return false;
    }
    std::size_t index = 0;
    int sb_index = 0;
    while (index < tree.size()) {
      std::size_t next = 0;
      if (!pack_node(tree, index, /*is_root=*/true, /*depth=*/0, &next,
                     error)) {
        return false;
      }
      index = next;
      ++sb_index;
    }
    if (sb_index != sb_cols * sb_rows) {
      error = "partition tree covers " + std::to_string(sb_index) + " of " +
              std::to_string(sb_cols * sb_rows) + " superblocks";
      return false;
    }
    return validate_plane_layout(error);
  }

 private:
  bool pack_node(std::span<const TreeNode> tree, std::size_t tree_index,
                 bool is_root, int depth, std::size_t* next,
                 std::string& error) {
    if (depth > kMaxTreeDepth) {
      error = "partition tree is deeper than the M1 limit";
      return false;
    }
    if (tree_index >= tree.size()) {
      error = "partition tree ended inside a subtree";
      return false;
    }
    const TreeNode& node = tree[tree_index];
    const auto bsize = static_cast<BlockSize>(node.bsize);
    const auto partition = static_cast<PartitionType>(node.partition);
    const bool is_ext = node.partition_is_ext != 0;
    if (!partition::is_valid_block_size(bsize)) {
      error = "partition tree node has an invalid block size";
      return false;
    }
    if (!is_ext && node.partition >= partition::kPartitionTypeCount) {
      error = "partition tree node has an invalid partition type";
      return false;
    }
    if (is_ext && node.ext_partition >= partition_ext::kPartitionCount) {
      error = "partition tree node has an invalid extended partition type";
      return false;
    }
    const int code_partition =
        is_ext ? static_cast<int>(node.ext_partition)
               : static_cast<int>(node.partition);
    const std::uint32_t code_mask =
        is_ext ? node.allowed_mask_ext : node.allowed_mask;
    if ((node.flags & syntax::kFlagForced) == 0 &&
        (code_mask & (1u << static_cast<unsigned>(code_partition))) == 0) {
      error = "partition is not in the node's allowed set";
      return false;
    }

    const std::size_t block_index = out_.blocks.size();
    out_.blocks.emplace_back();
    if (node.sdp_split != 0) {
      // §6.19.3.1 SDP split node: codes no partition syntax; its two children
      // are the LUMA_PART and CHROMA_PART tree roots.
      syntax::BlockSyntax& entry = out_.blocks.back();
      entry.mi_row = node.mi_row;
      entry.mi_col = node.mi_col;
      entry.bsize = node.bsize;
      entry.partition = 0;
      entry.allowed_mask = 0;
      entry.flags = static_cast<std::uint8_t>(syntax::kFlagSdpSplit);
      if (is_root) {
        entry.flags =
            static_cast<std::uint8_t>(entry.flags | syntax::kFlagSbRoot);
      }
      if (node.child_count == 0 || node.child_count > 4) {
        error = "SDP split node has an invalid child count";
        return false;
      }
      std::size_t cursor = tree_index + 1;
      for (int child = 0; child < static_cast<int>(node.child_count); ++child) {
        if (cursor >= tree.size()) {
          error = "SDP split node ended before all children were listed";
          return false;
        }
        std::size_t after = 0;
        if (!pack_node(tree, cursor, /*is_root=*/false, depth + 1, &after,
                       error)) {
          return false;
        }
        cursor = after;
      }
      const std::size_t subtree = out_.blocks.size() - block_index - 1;
      if (subtree > 0xFFFFU) {
        error = "partition subtree too large for the syntax stream";
        return false;
      }
      out_.blocks[block_index].subtree = static_cast<std::uint16_t>(subtree);
      *next = out_.blocks.size();
      return true;
    }
    {
      syntax::BlockSyntax& entry = out_.blocks.back();
      entry.mi_row = node.mi_row;
      entry.mi_col = node.mi_col;
      entry.bsize = node.bsize;
      entry.partition = static_cast<std::uint8_t>(code_partition);
      entry.allowed_mask = static_cast<std::uint16_t>(code_mask);
      entry.tree_type = node.tree_type;
      entry.flags = static_cast<std::uint8_t>(node.flags & syntax::kFlagForced);
      if (is_root) {
        entry.flags =
            static_cast<std::uint8_t>(entry.flags | syntax::kFlagSbRoot);
      }
    }

    const bool is_leaf = code_partition == 0;
    if (is_leaf) {
      if (node.child_count != 0) {
        error = "leaf node reports children";
        return false;
      }
      if (!pack_leaf(node, block_index, error)) return false;
    } else {
      if (node.child_count == 0 || node.child_count > 4) {
        error = "non-leaf node reports an invalid child count";
        return false;
      }
      std::size_t cursor = tree_index + 1;
      for (int child = 0; child < static_cast<int>(node.child_count);
           ++child) {
        if (cursor >= tree.size()) {
          error = "partition tree ended before all children were listed";
          return false;
        }
        std::size_t after = 0;
        if (!pack_node(tree, cursor, /*is_root=*/false, depth + 1, &after,
                       error)) {
          return false;
        }
        cursor = after;
      }
    }

    const std::size_t subtree = out_.blocks.size() - block_index - 1;
    if (subtree > 0xFFFFU) {
      error = "partition subtree too large for the syntax stream";
      return false;
    }
    out_.blocks[block_index].subtree = static_cast<std::uint16_t>(subtree);
    *next = out_.blocks.size();
    return true;
  }

  bool pack_leaf(const TreeNode& node, std::size_t block_index,
                 std::string& error) {
    if (node.leaf_index >= leaves_.size()) {
      error = "leaf_index is outside the GPU block/decision arrays";
      return false;
    }
    const LeafView& leaf = leaves_[node.leaf_index];
    const auto bsize = static_cast<BlockSize>(node.bsize);

    if (leaf.mi_row != static_cast<int>(node.mi_row) ||
        leaf.mi_col != static_cast<int>(node.mi_col)) {
      error = "block position does not match the partition tree node";
      return false;
    }
    // M1 codes U and V for every leaf, so the partition recursion's
    // `has_chroma` proxy (CHROMA_PART bookkeeping) does not gate residual
    // syntax here.  The chroma residual size is validated instead.
    const bool has_luma = leaf.luma_w != 0 || leaf.luma_h != 0;
    const int luma_w = has_luma ? partition::block_width(bsize) : 0;
    const int luma_h = has_luma ? partition::block_height(bsize) : 0;
    if (leaf.luma_w != luma_w || leaf.luma_h != luma_h) {
      error = "block luma size does not match the tree node block size";
      return false;
    }
    const bool has_chroma = leaf.has_chroma != 0;
    const BlockSize chroma_mi_size =
        leaf.chroma_bsize != 0 ? static_cast<BlockSize>(leaf.chroma_bsize)
                               : bsize;
    const BlockSize chroma_size =
        partition::chroma_plane_residual_size(geometry_, chroma_mi_size);
    if (!partition::is_valid_block_size(chroma_size)) {
      error = "tree node has no valid 4:2:0 chroma residual size";
      return false;
    }
    const int chroma_w = has_chroma ? partition::block_width(chroma_size) : 0;
    const int chroma_h = has_chroma ? partition::block_height(chroma_size) : 0;
    if (leaf.chroma_w != chroma_w || leaf.chroma_h != chroma_h) {
      error = "block chroma size does not match the 4:2:0 residual size";
      return false;
    }

    // TX_MODE_LARGEST: one transform per plane spanning the coded block, so
    // the declared transform size is derived from the coded dimensions.
    std::optional<txv::TxParams> luma_tx =
        has_luma ? txv::make_tx_params(luma_w, luma_h)
                 : std::optional<txv::TxParams>{};
    std::optional<txv::TxParams> chroma_tx =
        has_chroma ? txv::make_tx_params(chroma_w, chroma_h) : luma_tx;
    if ((has_luma && !luma_tx.has_value()) || !chroma_tx.has_value()) {
      error = "unsupported M1 transform size";
      return false;
    }
    if ((has_luma && (leaf.tx_w != luma_tx->w || leaf.tx_h != luma_tx->h)) ||
        (has_chroma && (leaf.chroma_tx_w != chroma_tx->w ||
                        leaf.chroma_tx_h != chroma_tx->h))) {
      error = "declared adjusted TX size does not match the transform";
      return false;
    }
    // §5.20.7.27 intra TX_TYPE: the adjusted size is type-independent, so the
    // declared tx_w/tx_h check above covers every signalled type.  The writer
    // validates that the type is codable for the block's set/mode.
    if (has_luma && (leaf.tx_type[0] < 0 || leaf.tx_type[0] >= 16)) {
      error = "luma tx_type outside the §9.6 TX_TYPE range";
      return false;
    }
    if (leaf.y_mode < 0 || leaf.y_mode >= 13 || leaf.uv_mode < 0 ||
        leaf.uv_mode >= 13) {
      error = "mode index outside the AV2 intra mode range";
      return false;
    }
    if (leaf.y_angle_delta < -3 || leaf.y_angle_delta > 3) {
      error = "AngleDeltaY outside [-3, 3]";
      return false;
    }

    syntax::BlockSyntax& entry = out_.blocks[block_index];
    entry.y_mode = static_cast<std::uint8_t>(leaf.y_mode);
    entry.y_angle_delta = static_cast<std::int8_t>(leaf.y_angle_delta);
    entry.uv_mode = static_cast<std::uint8_t>(leaf.uv_mode);
    entry.flags |= syntax::kFlagHasChromaInfo;
    entry.tree_type = static_cast<std::uint8_t>(leaf.tree_type);
    entry.has_chroma = has_chroma ? 1 : 0;
    entry.chroma_offset = static_cast<std::uint8_t>(leaf.chroma_offset);
    entry.chroma_bsize = static_cast<std::uint8_t>(chroma_mi_size);
    entry.chroma_mi_row = static_cast<std::uint16_t>(
        leaf.chroma_bsize != 0 ? leaf.chroma_mi_row : leaf.mi_row);
    entry.chroma_mi_col = static_cast<std::uint16_t>(
        leaf.chroma_bsize != 0 ? leaf.chroma_mi_col : leaf.mi_col);
    entry.mrl_index = static_cast<std::int8_t>(leaf.mrl_index);
    entry.mrl_sec_index = static_cast<std::uint8_t>(leaf.mrl_sec_index);
    entry.use_dip = static_cast<std::uint8_t>(leaf.use_dip);
    entry.dip_mode = static_cast<std::uint8_t>(leaf.dip_mode);
    entry.dip_transpose = static_cast<std::uint8_t>(leaf.dip_transpose);
    entry.cfl_index = static_cast<std::uint8_t>(leaf.cfl_index);
    entry.cfl_alpha_signs = static_cast<std::uint8_t>(leaf.cfl_alpha_signs);
    entry.cfl_alpha_u = static_cast<std::uint8_t>(leaf.cfl_alpha_u_syntax);
    entry.cfl_alpha_v = static_cast<std::uint8_t>(leaf.cfl_alpha_v_syntax);
    entry.cfl_mh_dir = static_cast<std::uint8_t>(leaf.cfl_mh_dir);
    entry.segment_id = static_cast<std::uint8_t>(leaf.segment_id);
    if (leaf.is_cfl != 0) entry.flags |= syntax::kFlagIsCfl;
    // §5.20.8 palette / §5.20.5.3-.4 intra block copy.  The syntax writer codes
    // the mode-info form and (for IBC) the block vector from these fields.
    if (leaf.is_palette != 0) {
      if (leaf.is_ibc != 0) {
        error = "leaf is both palette and intra-block-copy";
        return false;
      }
      entry.flags |= syntax::kFlagIsPalette;
      entry.palette_index = leaf.palette_index;
    }
    if (leaf.is_ibc != 0) {
      if (leaf.is_palette != 0 || leaf.is_cfl != 0 || leaf.use_dip != 0 ||
          leaf.mrl_index != 0) {
        error = "IBC leaf carries incompatible intra mode state";
        return false;
      }
      if (leaf.eob[0] != 0 || leaf.eob[1] != 0 || leaf.eob[2] != 0) {
        error = "IBC leaf must code no residual";
        return false;
      }
      entry.flags |= syntax::kFlagIsIbc;
      entry.ibc_skip = static_cast<std::uint8_t>(leaf.ibc_skip);
      entry.ibc_mode = static_cast<std::uint8_t>(leaf.ibc_mode);
      entry.ibc_ref_mv_idx = static_cast<std::uint8_t>(leaf.ibc_ref_mv_idx);
      entry.ibc_bv_row = leaf.ibc_bv_row;
      entry.ibc_bv_col = leaf.ibc_bv_col;
    }
    // §5.20.10.1 cdef_idx for the 64x64 block containing this leaf's origin.
    // The writer ignores repeats, so every leaf carries its block's value.
    if (info_.cdef_idx != nullptr) {
      // §7.18: one CDEF block is 64x64 luma = 16 MI cells.
      constexpr int kCdefSizeMi = 16;
      const int n64_row = leaf.mi_row / kCdefSizeMi;
      const int n64_col = leaf.mi_col / kCdefSizeMi;
      if (n64_col >= info_.cdef_n64_cols || n64_row >= info_.cdef_n64_rows ||
          n64_col < 0 || n64_row < 0) {
        error = "cdef_idx grid does not cover a coding block";
        return false;
      }
      entry.cdef_idx =
          info_.cdef_idx[static_cast<std::size_t>(n64_row) *
                             static_cast<std::size_t>(info_.cdef_n64_cols) +
                         static_cast<std::size_t>(n64_col)];
    }

    const txv::TxParams* params[3] = {has_luma ? &*luma_tx : nullptr,
                                      &*chroma_tx, &*chroma_tx};
    const int plane_count = has_chroma ? 3 : (has_luma ? 1 : 0);
    const std::span<const std::int32_t> planes[3] = {coeffs_.y, coeffs_.u,
                                                     coeffs_.v};

    entry.levels_offset = static_cast<std::uint32_t>(out_.levels.size());
    for (int plane = 0; plane < plane_count; ++plane) {
      if (params[plane] == nullptr) continue;
      const txv::TxParams& tx = *params[plane];
      const int area = tx.w * tx.h;
      const int eob = leaf.eob[plane];
      if (eob < 0 || eob > area) {
        error = "eob outside the adjusted transform area";
        return false;
      }
      entry.tx_size[plane] = static_cast<std::uint8_t>(tx.tx_size);
      entry.tx_type[plane] = static_cast<std::uint8_t>(leaf.tx_type[plane]);
      if (plane == 0) {
        entry.sec_tx_type = static_cast<std::uint8_t>(leaf.sec_tx_type);
        entry.most_probable_stx_set = static_cast<std::uint8_t>(leaf.stx_set);
      }
      if (plane == 1) {
        entry.cctx_type = static_cast<std::uint8_t>(leaf.cctx_type);
      }
      entry.eob[plane] = static_cast<std::uint32_t>(eob);
      if (eob == 0) continue;
      const std::uint64_t offset = leaf.coeff_offset[plane];
      const std::uint64_t area_u = static_cast<std::uint64_t>(area);
      if (offset + area_u > planes[plane].size()) {
        error = "coefficient plane is too small for the block's transform";
        return false;
      }
      if (info_.coeff_count[plane] != 0 &&
          offset + area_u > info_.coeff_count[plane]) {
        error = "coefficient offset is outside the plan's coeff_count";
        return false;
      }
      plane_ranges_[plane].emplace_back(offset, area_u);
      std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
      syntax::build_scan(tx.w, tx.h, scan);
      for (int c = 0; c < eob; ++c) {
        out_.levels.push_back(
            planes[plane][static_cast<std::size_t>(offset) +
                          static_cast<std::size_t>(
                              scan[static_cast<std::size_t>(c)])]);
      }
    }
    return true;
  }

  // The GPU hands over three independent per-plane coefficient buffers.  Each
  // block declares its own element range in every plane; those ranges must not
  // overlap, otherwise two blocks would code the same coefficients.  Checked
  // per plane over all blocks with eob > 0, independent of block order.
  bool validate_plane_layout(std::string& error) {
    for (int plane = 0; plane < 3; ++plane) {
      std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges =
          plane_ranges_[plane];
      std::sort(ranges.begin(), ranges.end());
      for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first < ranges[i - 1].first + ranges[i - 1].second) {
          error = "coefficient ranges overlap between blocks on plane " +
                  std::to_string(plane);
          return false;
        }
      }
    }
    return true;
  }

  const FrameInfo& info_;
  FrameGeometry geometry_;
  std::span<const LeafView> leaves_;
  GpuCoefficients coeffs_;
  PackedFrame& out_;
  // (offset, adjusted area) per plane, for the non-overlap check.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> plane_ranges_[3];
};

bool pack_frame(const FrameInfo& info, std::span<const TreeNode> tree,
                std::span<const LeafView> leaves,
                const GpuCoefficients& coeffs, PackedFrame& out,
                std::string& error) {
  error.clear();
  if (info.width <= 0 || info.height <= 0) {
    error = "GPU frame plan has no size";
    return false;
  }
  if ((info.width & 1) != 0 || (info.height & 1) != 0) {
    error = "M1 4:2:0 requires even dimensions";
    return false;
  }
  Packer packer(info, leaves, coeffs, out);
  return packer.run(tree, error);
}

bool assemble_packed(
    const PackedFrame& packed, const StillConfig& config,
    std::vector<std::uint8_t>& out, std::string& error,
    std::size_t* tile_bytes,
    std::span<const palette::PaletteResult> palette_results = {}) {
  return syntax::assemble_still_frame(packed.plan, packed.blocks, packed.levels,
                                   config, out, error, tile_bytes,
                                   palette_results);
}

}  // namespace

bool plan_tree_to_nodes(const plan::FramePlan& plan,
                        std::vector<TreeNode>& tree, std::string& error) {
  error.clear();
  tree.clear();
  const std::vector<plan::TreeNode>& src = plan.tree;
  if (src.empty()) {
    error = "plan has no partition tree";
    return false;
  }
  if (plan.nodes != 0 && static_cast<int>(src.size()) != plan.nodes) {
    error = "plan.tree size does not match plan.nodes";
    return false;
  }
  tree.assign(src.size(), TreeNode{});

  struct Walker {
    const std::vector<plan::TreeNode>& src;
    std::vector<TreeNode>& dst;
    const plan::FramePlan& plan;
    const FrameGeometry& geometry;
    std::string* error;
    int leaf_count = 0;

    // `has_chroma` / `chroma_offset` are the §5.20.3.2 recursion state for this
    // node; they select the allowed-partition set, so the plan's stored
    // allowed_mask/forced can be checked against the spec here.
    bool build(std::size_t i, std::size_t* next, int depth, bool has_chroma,
               bool chroma_offset) {
      if (i >= src.size()) {
        *error = "partition tree ended inside a subtree";
        return false;
      }
      if (depth > kMaxTreeDepth) {
        *error = "partition tree is deeper than the M1 limit";
        return false;
      }
      const plan::TreeNode& node = src[i];
      if (node.index != i) {
        *error = "plan tree node index is not its preorder position";
        return false;
      }
      std::uint8_t spec_mask = 0;
      std::uint8_t spec_flags = 0;
      int forced_partition = -1;
      if (!derive_node_flags(geometry, node.mi_row, node.mi_col,
                             node.block_size, has_chroma, chroma_offset,
                             &spec_mask, &spec_flags, &forced_partition,
                             *error)) {
        return false;
      }
      if (node.allowed_mask != spec_mask) {
        *error = "plan tree allowed_mask disagrees with the spec at mi (" +
                 std::to_string(node.mi_row) + "," +
                 std::to_string(node.mi_col) + ")";
        return false;
      }
      if ((node.forced ? syntax::kFlagForced : 0) != spec_flags) {
        *error = "plan tree forced flag disagrees with the spec at mi (" +
                 std::to_string(node.mi_row) + "," +
                 std::to_string(node.mi_col) + ")";
        return false;
      }
      const auto partition = static_cast<PartitionType>(node.partition );
      if (node.partition >= partition::kPartitionTypeCount) {
        *error = "plan tree node has an invalid partition type";
        return false;
      }
      if (forced_partition >= 0 && forced_partition != node.partition) {
        *error = "plan tree partition disagrees with the spec-forced partition";
        return false;
      }
      if ((spec_mask >> node.partition) == 0) {
        *error = "plan tree partition is not in its allowed set";
        return false;
      }

      TreeNode& out = dst[i];
      out.mi_row = static_cast<std::uint16_t>(node.mi_row);
      out.mi_col = static_cast<std::uint16_t>(node.mi_col);
      out.bsize = node.block_size;
      out.partition = node.partition;
      out.allowed_mask = node.allowed_mask;
      out.child_count = node.child_count;
      out.flags = 0;
      if (node.forced) out.flags |= syntax::kFlagForced;
      if (node.sb_root) out.flags |= syntax::kFlagSbRoot;

      if (partition == PartitionType::kNone) {
        if (node.child_count != 0 || node.subtree != 0 ||
            node.first_child != plan::kNoTreeNode) {
          *error = "plan tree leaf reports children";
          return false;
        }
        if (node.leaf_index == plan::kNoTreeNode) {
          *error = "plan tree leaf has no leaf_index";
          return false;
        }
        if (!plan.blocks.empty() && node.leaf_index >= plan.blocks.size()) {
          *error = "plan tree leaf_index is outside plan.blocks";
          return false;
        }
        // Leaves must reference blocks in syntax order.
        if (static_cast<int>(node.leaf_index) != leaf_count) {
          *error = "plan tree leaves are not in plan.blocks order";
          return false;
        }
        if (!plan.blocks.empty()) {
          const plan::BlockPlan& block = plan.blocks[node.leaf_index];
          if (block.has_chroma != has_chroma) {
            *error = "plan leaf has_chroma disagrees with the recursion";
            return false;
          }
          if (block.chroma_offset != chroma_offset) {
            *error = "plan leaf chroma_offset disagrees with the recursion";
            return false;
          }
        }
        ++leaf_count;
        out.leaf_index = node.leaf_index;
        *next = i + 1;
        return true;
      }

      if (node.child_count == 0 || node.child_count > 4) {
        *error = "plan tree node has an invalid child count";
        return false;
      }
      if (node.first_child != i + 1) {
        *error = "plan tree children are not contiguous in preorder";
        return false;
      }
      std::size_t cursor = i + 1;
      for (int child = 0; child < static_cast<int>(node.child_count);
           ++child) {
        const ChildState state =
            child_state(geometry, partition,
                        static_cast<BlockSize>(node.block_size), has_chroma,
                        chroma_offset, child);
        std::size_t after = 0;
        if (!build(cursor, &after, depth + 1, state.has_chroma,
                   state.chroma_offset)) {
          return false;
        }
        cursor = after;
      }
      if (node.subtree != cursor - i - 1) {
        *error = "plan tree subtree count disagrees with its children";
        return false;
      }
      *next = cursor;
      return true;
    }
  };

  const FrameGeometry geometry =
      make_geometry(FrameInfo{plan.width, plan.height,
                              plan.sb_size != 0 ? plan.sb_size : 64, {0, 0, 0}});
  Walker walker{src, tree, plan, geometry, &error};
  const int sb_mi = partition::sb_size_mi(geometry);
  const int sb_cols = (partition::mi_cols(geometry) + sb_mi - 1) / sb_mi;
  const int sb_rows = (partition::mi_rows(geometry) + sb_mi - 1) / sb_mi;

  std::size_t cursor = 0;
  int sb_index = 0;
  while (cursor < src.size()) {
    if (!src[cursor].sb_root) {
      error = "plan tree top-level node is not a superblock root";
      return false;
    }
    if (src[cursor].sb_index != static_cast<std::uint32_t>(sb_index)) {
      error = "plan tree superblocks are not in raster order";
      return false;
    }
    if (!walker.build(cursor, &cursor, 0, /*has_chroma=*/true,
                      /*chroma_offset=*/false)) {
      return false;
    }
    ++sb_index;
  }
  if (sb_index != sb_cols * sb_rows) {
    error = "plan tree covers " + std::to_string(sb_index) + " of " +
            std::to_string(sb_cols * sb_rows) + " superblocks";
    return false;
  }
  if (plan.leaves != 0 && walker.leaf_count != plan.leaves) {
    error = "plan tree leaf count disagrees with plan.leaves";
    return false;
  }
  return true;
}

bool pack_gpu_frame(const cuda::still_gpu::FramePlan& plan,
                    std::span<const TreeNode> tree,
                    std::span<const cuda::still_gpu::BlockPlan> blocks,
                    std::span<const cuda::still_gpu::BlockDecision> decisions,
                    const GpuCoefficients& coeffs, PackedFrame& out,
                    std::string& error, const FilterGrids& grids,
                    std::span<const palette::PaletteResult> palette_results) {
  (void)palette_results;
  error.clear();
  if (blocks.size() != decisions.size()) {
    error = "GPU block and decision arrays have different lengths";
    return false;
  }
  FrameInfo info;
  info.width = static_cast<int>(plan.width);
  info.height = static_cast<int>(plan.height);
  info.sb_size = plan.sb_size != 0 ? plan.sb_size : 64;
  info.coeff_count[0] = plan.coeff_count_y;
  info.coeff_count[1] = plan.coeff_count_u;
  info.coeff_count[2] = plan.coeff_count_v;
  info.cdef_idx = grids.cdef_idx;
  info.cdef_n64_cols = grids.cdef_n64_cols;
  info.cdef_n64_rows = grids.cdef_n64_rows;
  for (int plane = 0; plane < 3; ++plane) {
    info.ccso_blks[plane] = grids.ccso_blks[plane];
    info.ccso_blk_cols[plane] = grids.ccso_blk_cols[plane];
    info.ccso_blk_rows[plane] = grids.ccso_blk_rows[plane];
  }
  std::vector<LeafView> leaves(blocks.size());
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    adapt_gpu_leaf(blocks[i], decisions[i], leaves[i]);
  }
  return pack_frame(info, tree, leaves, coeffs, out, error);
}

bool pack_gpu_raster_frame(const cuda::still_gpu::FramePlan& plan,
                           std::span<const cuda::still_gpu::BlockPlan> blocks,
                           std::span<const cuda::still_gpu::BlockDecision> decisions,
                           const GpuCoefficients& coeffs, PackedFrame& out,
                           std::string& error) {
  error.clear();
  FrameInfo info;
  info.width = static_cast<int>(plan.width);
  info.height = static_cast<int>(plan.height);
  info.sb_size = plan.sb_size != 0 ? plan.sb_size : 64;
  const FrameGeometry geometry = make_geometry(info);
  const int sb_mi = partition::sb_size_mi(geometry);
  const int mi_cols = partition::mi_cols(geometry);
  const int mi_rows = partition::mi_rows(geometry);
  if (sb_mi <= 0) {
    error = "invalid superblock geometry";
    return false;
  }
  const int sb_cols = (mi_cols + sb_mi - 1) / sb_mi;
  const int sb_rows = (mi_rows + sb_mi - 1) / sb_mi;
  const std::size_t sb_count =
      static_cast<std::size_t>(sb_cols) * static_cast<std::size_t>(sb_rows);
  if (blocks.size() != sb_count) {
    error = "raster plan expects exactly one block per superblock (" +
            std::to_string(sb_count) + "), got " +
            std::to_string(blocks.size());
    return false;
  }

  std::vector<TreeNode> tree(sb_count);
  for (int row = 0; row < sb_rows; ++row) {
    for (int col = 0; col < sb_cols; ++col) {
      const std::size_t index =
          static_cast<std::size_t>(row) * static_cast<std::size_t>(sb_cols) +
          static_cast<std::size_t>(col);
      const int mi_row = row * sb_mi;
      const int mi_col = col * sb_mi;
      TreeNode& node = tree[index];
      node.mi_row = static_cast<std::uint16_t>(mi_row);
      node.mi_col = static_cast<std::uint16_t>(mi_col);
      const BlockSize size = partition::superblock_size(geometry.sb_size);
      node.bsize = static_cast<std::uint8_t>(size);
      node.partition = static_cast<std::uint8_t>(PartitionType::kNone);
      node.child_count = 0;
      node.leaf_index = static_cast<std::uint32_t>(index);
      if (blocks[index].mi_row != node.mi_row ||
          blocks[index].mi_col != node.mi_col) {
        error = "raster plan block " + std::to_string(index) +
                " is not at its superblock origin";
        return false;
      }
      std::uint8_t mask = 0;
      std::uint8_t flags = 0;
      int forced_partition = -1;
      if (!derive_node_flags(geometry, mi_row, mi_col,
                             static_cast<int>(size), /*has_chroma=*/true,
                             /*chroma_offset=*/false, &mask, &flags,
                             &forced_partition, error)) {
        return false;
      }
      if (forced_partition >= 0 &&
          forced_partition != static_cast<int>(PartitionType::kNone)) {
        error = "superblock position cannot be coded as PARTITION_NONE";
        return false;
      }
      node.allowed_mask = mask;
      node.flags = flags;
    }
  }
  return pack_gpu_frame(plan, tree, blocks, decisions, coeffs, out, error);
}

bool pack_plan_frame(const plan::FramePlan& plan,
                     std::span<const plan::BlockPlan> blocks,
                     std::span<const cuda::still_gpu::BlockDecision> decisions,
                     const GpuCoefficients& coeffs, PackedFrame& out,
                     std::string& error) {
  error.clear();
  if (blocks.size() != decisions.size()) {
    error = "plan block and decision arrays have different lengths";
    return false;
  }
  if (!plan.blocks.empty() && plan.blocks.size() != blocks.size()) {
    error = "leaf array does not match plan.blocks";
    return false;
  }
  if (!plan.blocks.empty() && blocks.data() != plan.blocks.data()) {
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      if (blocks[i].index != plan.blocks[i].index ||
          blocks[i].sb_index != plan.blocks[i].sb_index ||
          blocks[i].mi_row != plan.blocks[i].mi_row ||
          blocks[i].mi_col != plan.blocks[i].mi_col) {
        error = "leaf array order disagrees with plan.blocks";
        return false;
      }
    }
  }
  std::vector<TreeNode> tree;
  if (!plan_tree_to_nodes(plan, tree, error)) return false;
  FrameInfo info;
  info.width = plan.width;
  info.height = plan.height;
  info.sb_size = plan.sb_size != 0 ? plan.sb_size : 64;
  info.coeff_count[0] = plan.luma_coeff_count;
  info.coeff_count[1] = plan.u_coeff_count;
  info.coeff_count[2] = plan.v_coeff_count;
  std::vector<LeafView> leaves(blocks.size());
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    adapt_plan_leaf(blocks[i], decisions[i], leaves[i]);
  }
  return pack_frame(info, tree, leaves, coeffs, out, error);
}

bool assemble_gpu_frame(const cuda::still_gpu::FramePlan& plan,
                        std::span<const TreeNode> tree,
                        std::span<const cuda::still_gpu::BlockPlan> blocks,
                        std::span<const cuda::still_gpu::BlockDecision> decisions,
                        const GpuCoefficients& coeffs,
                        const StillConfig& config, std::vector<std::uint8_t>& out,
                        std::string& error, std::size_t* tile_bytes,
                        const FilterGrids& grids,
                        std::span<const palette::PaletteResult> palette_results) {
  PackedFrame packed;
  if (!pack_gpu_frame(plan, tree, blocks, decisions, coeffs, packed, error,
                      grids, palette_results)) {
    return false;
  }
  return assemble_packed(packed, config, out, error, tile_bytes,
                         palette_results);
}

bool assemble_gpu_raster_frame(
    const cuda::still_gpu::FramePlan& plan,
    std::span<const cuda::still_gpu::BlockPlan> blocks,
    std::span<const cuda::still_gpu::BlockDecision> decisions,
    const GpuCoefficients& coeffs, const StillConfig& config,
    std::vector<std::uint8_t>& out, std::string& error,
    std::size_t* tile_bytes) {
  PackedFrame packed;
  if (!pack_gpu_raster_frame(plan, blocks, decisions, coeffs, packed, error)) {
    return false;
  }
  return assemble_packed(packed, config, out, error, tile_bytes);
}

bool assemble_plan_frame(const plan::FramePlan& plan,
                         std::span<const plan::BlockPlan> blocks,
                         std::span<const cuda::still_gpu::BlockDecision> decisions,
                         const GpuCoefficients& coeffs, const StillConfig& config,
                         std::vector<std::uint8_t>& out, std::string& error,
                         std::size_t* tile_bytes) {
  PackedFrame packed;
  if (!pack_plan_frame(plan, blocks, decisions, coeffs, packed, error)) {
    return false;
  }
  return assemble_packed(packed, config, out, error, tile_bytes);
}

}  // namespace av2enc::still::stream
