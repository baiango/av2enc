// AV2 sub-8 coding blocks: spec rules, chroma ownership, contexts, TX choice.
//
// Scope
// -----
// This module owns the *reference rules* for luma coding blocks whose width or
// height is 4 samples (BLOCK_4X4, 4X8, 8X4, 4X16, 16X4, 4X32, 32X4).  In AV2
// a "sub-8" block is one where min(width, height) == 4, i.e. exactly the sizes
// whose chroma block can no longer follow the luma block at 4:2:0.  8x8 is the
// smallest square that maps 1:1 to a 4:4 chroma block, which is why 8x8 is the
// "normal floor": below it the chroma block is inherited from a larger
// ancestor instead of being derived from the luma size.
//
// BLOCK_8X64 / BLOCK_64X8 are also terminal (`partition_implied()` returns
// NONE) and share the 4:2:0 chroma-offset behaviour, but their minimum side is
// 8, so is_sub8() is false for them.  They are covered by the partition and
// chroma tests as thin blocks.
//
// It is additive: src/encoder/m1/m1_*, src/encoder/partition/,
// src/main.cpp and CMakeLists.txt are not modified.  See README.md for the
// exact M1 writer call sites that still need the `has_chroma` /
// `ChromaMi*` plumbing.
//
// Spec: AV2 Bitstream & Decoding Process Specification v1.0.0.
//   §5.20.3.1 decode_partition (hasChroma / chromaOffset / ChromaMi*)
//   §5.20.3.2 read_partition, is_partition_allowed, partition_implied
//   §5.20.5.3 intra_frame_mode_info, §5.20.5.6 read_intra_uv_mode
//   §5.20.7.23 residual, §5.20.7.26 get_plane_residual_size
//   §8.3.2 CDF context derivation (do_split, rect_type, all_zero)
//   §9.2 Partition_Size_Adjust, Max_Tx_Size_Rect, Subsampled_Size
//
// No AVM, SVT-AV1/AV2, clave or av2-turbo code was consulted.
#pragma once

#include "encoder/partition/partition.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace av2enc::m1::sub8 {

using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionType;

// ---------------------------------------------------------------------------
// Geometry predicates
// ---------------------------------------------------------------------------

// True when min(block_width, block_height) == 4.  The seven 4xN / Nx4 sizes
// that are valid AV2 block sizes (4x4, 4x8, 8x4, 4x16, 16x4, 4x32, 32x4).
bool is_sub8(BlockSize size) noexcept;

// Smallest side of the block in samples (4, 8, 16, ...), 0 when invalid.
int min_dim_px(BlockSize size) noexcept;

const char* block_name(BlockSize size) noexcept;

// ---------------------------------------------------------------------------
// Partition tree / syntax validity (§5.20.3.2)
// ---------------------------------------------------------------------------

// Everything `read_partition()` derives before coding a symbol.
struct PartitionPlan {
  partition::PartitionSet allowed;
  partition::PartitionBoundary implied;   // partition_implied()
  partition::RectTypeImplied rect_type;   // rect_type_implied_by_bsize()
  // True when the spec codes no partition symbol at this node: the boundary
  // implies a partition that is allowed, or only one partition is allowed.
  bool forced = false;
  PartitionType forced_partition = PartitionType::kNone;
  // Symbol presence for the chosen partition (meaningful only when !forced).
  bool codes_do_split = false;
  bool codes_do_square_split = false;
  bool codes_rect_type = false;
  // PARTITION_NONE is legal and the caller may choose it.
  bool none_allowed = false;
  // "allowHorz"/"allowVert" as used by read_partition(); equal to the single
  // allowed rect family when rect_type is implied.
  bool allow_horz = false;
  bool allow_vert = false;
};

// Re-derives allowed/implied from the geometry.  `has_chroma`/`chroma_offset`
// are the decode_partition() recursion arguments.
PartitionPlan plan_partition(const FrameGeometry& geometry, int mi_row,
                             int mi_col, BlockSize size, bool has_chroma,
                             bool chroma_offset);

// True when `partition` is spec-valid for this node (delegates to
// partition::is_partition_allowed and mirrors the PARTITION_NONE fallback).
bool partition_allowed(const FrameGeometry& geometry, int mi_row, int mi_col,
                       BlockSize size, PartitionType partition,
                       bool has_chroma, bool chroma_offset);

// ---------------------------------------------------------------------------
// Chroma ownership along the decode recursion
// ---------------------------------------------------------------------------

// The §5.20.3.1 `ChromaMi*` state.  `mi_row`/`mi_col` are luma 4x4 units.
struct ChromaInfo {
  bool valid = false;
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;  // ChromaMiSize
};

// One node of a partition tree with enough state to derive chroma ownership
// and to drive the minimal syntax writer.  `children` are in decode order and
// contain in-frame children only, exactly like partition::PartitionNode.
struct Node {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  PartitionType partition = PartitionType::kNone;
  bool forced = false;
  bool has_chroma = false;      // HasChroma for this node
  bool chroma_offset = false;   // chromaOffset for this node
  ChromaInfo chroma;            // active ChromaMi* at this node
  std::vector<Node> children;
};

// Chooses a partition at a non-forced node.  Return a valid, allowed
// partition; the builder validates it.
using ChoosePartition = std::function<PartitionType(const Node&, const PartitionPlan&)>;

// Builds one superblock subtree.  `sb_mi_row`/`sb_mi_col` must be aligned to
// the superblock size and in frame.  Only the four M1 partitions are modelled
// and the superblock must be 64x64 (M1 configuration); 128x128 (where
// PARTITION_SPLIT exists) is rejected instead of guessed.
bool build_superblock(const FrameGeometry& geometry, int sb_mi_row,
                      int sb_mi_col, const ChoosePartition& choose, Node& out,
                      std::string& error);

// Recursively derives has_chroma/chroma_offset/chroma for every node using the
// exact decode_partition() recursion.  Called by build_superblock(); exposed
// so tests and callers that build a Node tree themselves can reuse it.
void assign_chroma(const FrameGeometry& geometry, bool has_chroma,
                   bool chroma_offset, const ChromaInfo& inherited, Node& node);

// Leaves in decode order (SPLIT TL,TR,BL,BR; HORZ top,bottom; VERT left,right).
void collect_leaves(const Node& node, std::vector<const Node*>& leaves);

// ---------------------------------------------------------------------------
// Mode info (§5.20.5.3 / §5.20.5.6)
// ---------------------------------------------------------------------------

struct ModeInfoPlan {
  // read_intra_y_mode() runs for every SHARED_PART leaf, sub-8 included.
  bool codes_y_mode = true;
  // read_intra_uv_mode() runs iff HasChroma (the chroma-owning leaf).
  bool codes_uv_mode = false;
  bool cfl_allowed = false;
  // Palette is only allowed for MiSize >= BLOCK_8X8, so it is never present on
  // a sub-8 leaf; M1 does not enable screen-content tools either.
  bool palette_allowed = false;
  // get_plane_residual_size(ChromaMiSize, plane), the sizes the chroma mode
  // list / CfL availability and the chroma residual use.
  BlockSize uv_residual_size = BlockSize::kInvalid;
};

ModeInfoPlan mode_info_plan(const FrameGeometry& geometry, const Node& leaf,
                            bool enable_cfl_intra);

// ---------------------------------------------------------------------------
// Coefficient contexts (§8.3.2 all_zero) and transform selection
// ---------------------------------------------------------------------------

// `plane_bsize` is get_plane_residual_size(plane > 0 ? ChromaMiSize : MiSize,
// plane).  `tx_lw`/`tx_lh` are Tx_Width_Log2/Tx_Height_Log2.  `above`/`left`
// are the already-OR'd neighbour contexts for the transform footprint.
// `u_has_cf` is true when the U plane had a non-zero EOB (EobU != 0).
int txb_skip_context(int plane, BlockSize plane_bsize, int tx_lw, int tx_lh,
                     int above, int left, bool u_has_cf);

// `get_plane_residual_size(ChromaMiSize, 1)` — the chroma residual block.
BlockSize chroma_residual_size(const FrameGeometry& geometry,
                               const ChromaInfo& chroma);

struct TxPlan {
  int luma_tx = 0;    // tvx::TxSize index for TX_MODE_LARGEST
  int chroma_tx = 0;  // tvx::TxSize index for TX_MODE_LARGEST
  int luma_eob_multisize = 0;
  int chroma_eob_multisize = 0;
  // transform_type() forces DCT_DCT when eob <= 1.
  bool luma_tx_type_coded = false;
  // get_tx_set() for an intra 4xN/Nx4 / 4x4 block: TX_SET_INTRA_1 (7 symbols)
  // whenever the type is coded at all.
  bool luma_tx_type_set1 = false;
  // TX_MODE_SELECT (tx_mode_select == 1) would code tx partition syntax on a
  // 4x8/8x4; M1 uses TX_MODE_LARGEST so no transform-tree symbols appear.
  bool codes_tx_partition = false;
};

// Applies §9.2 Max_Tx_Size_Rect and §8.3.2 get_tx_set to one leaf.  `eob_luma`
// is the luma plane EOB (0 means "all skip": no TX type symbol).
TxPlan tx_plan(const FrameGeometry& geometry, const Node& leaf, int eob_luma);

// eobMultisize (§5.20.7.27): Min(Tx_Width_Log2,5) + Min(Tx_Height_Log2,5) - 4.
int eob_multisize(int tx_size);

// True when the transform type set for this intra transform is TX_SET_INTRA_1.
bool tx_type_set1(int tx_size);

}  // namespace av2enc::m1::sub8
