// M1 GPU host-side frame plan: partition tree, leaf geometry, syntax order.
//
// Spec-only.  This is the CPU host half of workstream 2 in docs/m1-gpu.md:
// it runs the same `partition::decide_frame` search as the CPU M1 reference
// (`src/encoder/m1/still_encoder.cpp`) and flattens the resulting tree into a plain
// block list in exact entropy/syntax order.  The CUDA stages consume this list;
// they never run the partition search themselves.
//
// Contract
// --------
//   * Cost model is copied verbatim from the CPU M1 path: the leaf cost is the
//     truncated block variance of the visible source luma, and every partition
//     signal costs `lambda * 2` with `lambda = 0.85 * 2^((qp - 12) / 6)`.
//   * Depth 4, min block size 8 by default.  `force_single_block` forces one
//     64x64 PARTITION_NONE superblock exactly like StillOptions.
//   * The debug gates AV2ENC_STILL_SQUARE_ONLY and AV2ENC_STILL_LEAF32 are honoured
//     through `apply_debug_env()` with the same +1e18 leaf penalty.
//   * Blocks are emitted in decode/syntax order: superblocks in raster order,
//     then each tree depth-first in child order (NONE = one block).
//
// Geometry notes
// --------------
//   * Luma block dimensions are the coded block size, not the clipped visible
//     size.  A bottom/right edge block can extend past MiCols*4 / MiRows*4 and
//     is still coded in full; `visible_*` carries the clamped extent.
//   * Chroma residual size comes from `partition::chroma_plane_residual_size`
//     (§9.2 Subsampled_Size for 4:2:0).
//   * `PlaneTxPlan::tx_size` is the AV2 enum_tx_size index (0..24) and
//     `adj_w * adj_h` is the adjusted transform area the M1 transform layer
//     (`txv::make_tx_params`) works on.  M1 always uses DCT_DCT.
//   * `coeff_offset` is an element offset (int32 levels) into a per-plane
//     coefficient buffer; `coeff_count` elements follow.  U and V have separate
//     buffers but identical layouts in 4:2:0.
//
// Availability
// ------------
//   * `NeighborAvailability` mirrors what the CPU M1 encoder feeds to
//     `intra::prepare_edges`: above = origin sample row > 0, left = origin
//     sample column > 0, and num4_above_right / num4_below_left are the 4x4
//     counts after the superblock-local `clear_block_decoded_flags` reset.
//     The boolean above_right / below_left bits are `count > 0`.
//   * Chroma availability is reported separately because its 4x4 grid origin
//     is luma_origin / 2.
//
// This header is host-side and CPU-only; it has no CUDA dependency.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace av2enc::still::plan {

// M1 partition search options.  Defaults match av2enc::still::StillOptions.
struct PlanOptions {
  // Sequence-level superblock flag: 64, 128, or 256.  A still picture is an
  // intra frame, and §5.18.1 clamps its frame `SbSize` to 128 even when
  // use_256x256_superblock is signalled, so a 256 request plans 128x128 roots.
  int sb_size = 64;
  std::uint8_t base_q_idx = 90;
  int min_block_size = 8;
  int max_partition_depth = 4;
  bool allow_rect = true;
  bool allow_split = false;
  bool force_single_block = false;
  // Sub-8 coding blocks: the CPU path lowers the partition floor to 4 and
  // raises the rect recursion depth to at least 8.
  bool sub8 = false;
  // §5.4.3 / §5.20.3.2 extended partition families.  `enable_uneven_4way`
  // requires ext partitions just like the sequence header.
  bool enable_ext_partitions = false;
  bool enable_uneven_4way_partitions = false;
  // Semi-decoupled partitioning.  The CPU product path currently gates the
  // forced-PARTITION_NONE luma/chroma trees of every intra 64x64 split; the
  // planner reproduces exactly that configuration.
  bool sdp = false;
  // Debug gates; normally set by apply_debug_env().
  bool square_only = false;
  bool leaf32 = false;
};

// Sets square_only / leaf32 from AV2ENC_STILL_SQUARE_ONLY / AV2ENC_STILL_LEAF32.
// Never clears a flag, so callers may force a gate explicitly as well.
void apply_debug_env(PlanOptions& options);

// One transform block on one plane.
struct PlaneTxPlan {
  int coded_w = 0;   // residual block width handed to the transform
  int coded_h = 0;
  int tx_size = -1;  // enum_tx_size index (§6.19.6.1); -1 = unsupported
  int adj_w = 0;     // adjusted transform width  (4..32)
  int adj_h = 0;     // adjusted transform height (4..32)
  std::uint8_t tx_type = 0;  // TX_TYPE; M1 uses DCT_DCT (0)
  std::uint32_t coeff_offset = 0;  // int32 element offset in the plane buffer
  std::uint32_t coeff_count = 0;   // adj_w * adj_h
};

// Per-block neighbour availability, matching intra::NeighborRequest.
struct NeighborAvailability {
  bool above = false;
  bool left = false;
  bool above_left = false;
  bool above_right = false;
  bool below_left = false;
  std::uint8_t num4_above_right = 0;
  std::uint8_t num4_below_left = 0;
};

inline constexpr std::uint32_t kNoTreeNode = 0xFFFFFFFFu;

// One partition-tree node in preorder (superblocks raster, each node followed
// by its children in decode order).  This is the tree the syntax writer needs;
// `FramePlan::blocks` remains the leaf-only view in the same order.
//
// Children are contiguous: the first child is `first_child` (always
// `index + 1` for a non-leaf), and `child_count` entries follow.  `subtree` is
// the number of descendant entries.  A leaf has partition == kNone
// (PartitionType 0), child_count == 0, first_child == kNoTreeNode and points at
// its `FramePlan::blocks` entry through `leaf_index`.
struct TreeNode {
  std::uint32_t index = 0;
  std::uint32_t sb_index = 0;
  int mi_row = 0;
  int mi_col = 0;
  std::uint8_t block_size = 255;  // partition::BlockSize enum value
  std::uint8_t partition = 0;     // partition::PartitionType enum value
  std::uint8_t depth = 0;
  std::uint8_t allowed_mask = 0;  // bit p => allowed[PartitionType p]
  bool forced = false;            // no partition syntax coded at this node
  bool sb_root = false;           // first node of a superblock (depth 0)
  std::uint8_t child_count = 0;
  std::uint32_t first_child = kNoTreeNode;
  std::uint32_t subtree = 0;      // descendant entries
  std::uint32_t leaf_index = kNoTreeNode;  // FramePlan::blocks index for leaves
  // Extended-partition view (§6.19.3 spec enum 0..9).  `partition` above keeps
  // the four-value M1 enum for existing consumers; an extended row is folded
  // to HORZ/VERT there and stays exact here.
  std::uint8_t ext_partition = 0;
  std::uint16_t allowed_mask_ext = 0;  // bit p => allowed[spec PartitionType p]
  double cost = 0.0;                   // decided subtree cost
  // SDP (§6.19.3.1 TreeType): 0 = SHARED_PART, 1 = LUMA_PART,
  // 2 = CHROMA_PART.  `sdp_split` marks the intra luma/chroma split node; it
  // codes no partition syntax and its two children are the tree roots.
  std::uint8_t tree_type = 0;
  bool sdp_split = false;
  bool sdp_chroma_block = false;
};

// One leaf coding block.  Plain data; upload as an array.
struct BlockPlan {
  std::uint32_t index = 0;
  std::uint32_t sb_index = 0;
  int sb_mi_row = 0;
  int sb_mi_col = 0;
  int mi_row = 0;
  int mi_col = 0;
  int x = 0;  // mi_col * 4 (luma sample origin)
  int y = 0;  // mi_row * 4
  int chroma_x = 0;  // x / 2
  int chroma_y = 0;  // y / 2
  std::uint8_t block_size = 255;  // partition::BlockSize enum value
  std::uint8_t depth = 0;
  int luma_w = 0;
  int luma_h = 0;
  int visible_luma_w = 0;
  int visible_luma_h = 0;
  int chroma_w = 0;
  int chroma_h = 0;
  int visible_chroma_w = 0;
  int visible_chroma_h = 0;
  // Partition recursion state.  The M1 CPU reference codes U/V for every leaf,
  // so `u`/`v` and their offsets are always populated even when this is false
  // (only reachable below the default 8-pixel floor).
  bool has_chroma = false;
  bool chroma_offset = false;
  // ChromaMi* (luma 4x4 units) for the block this leaf owns.  A sub-8 leaf
  // that does not own the inherited 8x8 chroma block has has_chroma == false
  // and codes no U/V payload; chroma_mi_* still records the shared block.
  int chroma_mi_row = 0;
  int chroma_mi_col = 0;
  std::uint8_t chroma_mi_size = 255;
  // 0 = SHARED_PART, 1 = LUMA_PART, 2 = CHROMA_PART (SDP).  Zero for the
  // shared-tree rect/extended paths.
  std::uint8_t tree_type = 0;
  PlaneTxPlan luma;
  PlaneTxPlan u;
  PlaneTxPlan v;
  NeighborAvailability luma_avail;
  NeighborAvailability chroma_avail;
};

// Whole-frame plan.  `blocks` is the syntax-order block list.
struct FramePlan {
  int width = 0;
  int height = 0;
  int chroma_width = 0;
  int chroma_height = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  // Effective frame SbSize (§5.18.1): 64 or 128.  A still picture clamps a
  // 256x256 sequence flag to 128x128.
  int sb_size = 0;
  // Requested sequence-level superblock flag (PlanOptions::sb_size).
  int sequence_sb_size = 64;
  int sb_cols = 0;
  int sb_rows = 0;
  // MiCols*4 / MiRows*4 coded plane and the 64-aligned backing store size used
  // by the CPU M1 encoder.
  int coded_luma_w = 0;
  int coded_luma_h = 0;
  int coded_chroma_w = 0;
  int coded_chroma_h = 0;
  int luma_stride = 0;
  int luma_padded_height = 0;
  int chroma_stride = 0;
  int chroma_padded_height = 0;
  int leaves = 0;
  int nodes = 0;
  double cost = 0.0;
  std::uint32_t luma_coeff_count = 0;
  std::uint32_t u_coeff_count = 0;
  std::uint32_t v_coeff_count = 0;
  PlanOptions options;
  // Leaf-only view, exact syntax order.  Unchanged by the tree export.
  std::vector<BlockPlan> blocks;
  // Full partition tree, preorder.  `tree.size() == nodes`; leaves reference
  // the entries in `blocks` through TreeNode::leaf_index.
  std::vector<TreeNode> tree;
};

// Builds the plan.  `luma` is a tightly packed 8-bit plane of width*height
// samples (the same data the CPU M1 variance cost reads).
bool build_frame_plan(int width, int height, std::span<const std::uint8_t> luma,
                      const PlanOptions& options, FramePlan& plan,
                      std::string& error);

// One candidate node handed to a caller-supplied leaf-cost provider.
struct NodeGeometry {
  int mi_row = 0;
  int mi_col = 0;
  std::uint8_t block_size = 255;  // partition::BlockSize enum value
};

// Leaf variance in the same units as block_variance_cost_at(): the truncated
// SSE of the visible luma part, as an int64.  The planner adds the >64 and
// debug-gate penalties; the provider only computes the variance.
using LeafCostFunction = std::function<std::int64_t(const NodeGeometry&)>;

// Same plan, but leaf variances come from `leaf_cost`.  A device cost kernel
// (plan_cost.cu) can fill these without changing tree/leaf ordering.
bool build_frame_plan_with_leaf_cost(
    int width, int height, std::span<const std::uint8_t> luma,
    const PlanOptions& options, const LeafCostFunction& leaf_cost,
    FramePlan& plan, std::string& error);

// Structural check of a built plan: block geometry, chroma residual rules,
// coefficient-offset contiguity, availability counts, and exact coverage of
// MiCols x MiRows cells.  Does not re-run the partition decision.
bool validate_frame_plan(const FramePlan& plan, std::string& error);

// Deterministic text dumps.  Both end with a trailing newline.
//   TSV:  header lines beginning with '#', one tab-separated data row per block
//   JSON: one object, blocks in the same order
std::string plan_to_tsv(const FramePlan& plan);
std::string plan_to_json(const FramePlan& plan);

// Preorder tree dump (same block/leaf order, one row per partition node).
std::string plan_tree_to_tsv(const FramePlan& plan);

// partition::block_size_name() for a stored BlockPlan::block_size value.
const char* block_size_name(int block_size);

}  // namespace av2enc::still::plan
