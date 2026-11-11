// AV2 M1 still-picture syntax writer.
//
// This module owns *only* bitstream syntax: partition symbols, mode info,
// transform-tree/EOB/coefficient coding, and OBU assembly.  It does not decide
// anything and it does not reconstruct anything: it consumes a block-result
// stream (BlockSyntax + a flat coefficient-level array) produced by _either_
// the CPU decision/reconstruction path (src/encoder/m1/m1_still.cpp) or the
// GPU pipeline (src/cuda/m1/, packed by src/encoder/m1/stream/).
//
// Design contract
// ---------------
//   * The stream is a *preorder* list of partition nodes, superblocks first, in
//     raster superblock order.  A node is a leaf when partition == kNone, in
//     which case it also carries the leaf payload (modes, TX sizes, EOBs, and
//     the offset of its coefficient levels).
//   * Every byte of syntax context (do_split / rect_type / Y-mode set /
//     is_cfl / is_long_side / TXB skip / DC sign / coefficient magnitude) is
//     *derived here* from the block stream in coding order, exactly as a
//     decoder derives it.  The producer therefore never has to send CDF or
//     neighbour state: above_mi/left_mi, IntraJointModes, uvCfL, culLevel and
//     DC-sign contexts are rebuilt from the stream itself.
//   * Coefficient levels are stored in *scan order*: level[offset + c] is the
//     quantised coefficient at scan position c, for c in [0, eob).  Planes are
//     contiguous in the order Y, U, V.  This is the natural output of a GPU
//     transform/quant stage and keeps the plan compact.
//   * Levels are per plane `eob` entries; `levels_count` is the sum over the
//     three planes and is validated against the flat array bounds.
//
// Bit-exactness: for the same decisions, assemble_m1_frame() emits exactly the
// bytes the pre-refactor inline writer in m1_still.cpp emitted.  See
// src/encoder/m1/stream/README.md for the ordering and CDF-context rules a GPU
// orchestrator must satisfy.
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0.
//   §5.20.3.2 partition syntax, §5.20.5 mode info, §5.20.7 residual coding,
//   §8.3.2 partition contexts, §8.2 symbol coder (src/entropy/av2).
#pragma once

#include "encoder/m1/m1_headers.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::m1::syntax {

inline constexpr int kPlaneCount = 3;

// BlockSyntax::flags bits.
enum BlockFlags : std::uint8_t {
  // The partition solver reports that the spec codes no partition syntax at
  // this node (boundary-implied or single candidate).  Mirrors
  // partition::PartitionNode::forced.
  kFlagForced = 1u << 0,
  // This leaf codes chroma with CfL (CFL_DERIVED_ALPHA).  The is_cfl symbol
  // and cfl_index are emitted only when set / when CfL is enabled.
  kFlagIsCfl = 1u << 1,
  // This node is the root of a superblock (depth 0).  Required so the writer
  // can clear the left partition context at the start of every superblock row,
  // exactly like m1_still's `if (sb.mi_col == 0) clear_left_context()`.
  kFlagSbRoot = 1u << 2,
};

// One node of the partition tree, plus the leaf payload when it is a leaf.
//
// Leaf payload fields are ignored when partition != kNone.  `subtree` makes the
// preorder stream self-describing: it is the number of following entries that
// belong to this node's children (0 for a leaf).  It lets the writer walk the
// tree without duplicating the spec's subsize/boundary rules, and it is
// validated against those rules on every frame.
struct BlockSyntax {
  // --- tree ---
  std::uint16_t mi_row = 0;
  std::uint16_t mi_col = 0;
  std::uint8_t bsize = 0;         // partition::BlockSize
  std::uint8_t partition = 0;     // partition::PartitionType
  std::uint8_t allowed_mask = 0;  // bit p = allowed[PartitionType p]
  std::uint8_t flags = 0;         // BlockFlags
  std::uint16_t subtree = 0;      // entries in the child subtrees

  // --- leaf mode info ---
  std::uint8_t y_mode = 0;        // AV2 YMode 0..12 (§6.19.5.5)
  std::int8_t y_angle_delta = 0;  // AngleDeltaY -3..3
  std::uint8_t uv_mode = 0;       // AV2 UV mode (kDcPred when is_cfl)

  // --- leaf residual ---
  std::uint8_t tx_size[kPlaneCount] = {0, 0, 0};  // AV2 TX_SIZE 0..24
  // AV2 TX_TYPE (txv::TxType).  Only luma can signal a non-DCT type; chroma is
  // derived from the DCT-only rule.  Ignored for eob <= 1 (spec forces DCT).
  std::uint8_t tx_type[kPlaneCount] = {0, 0, 0};
  std::uint32_t eob[kPlaneCount] = {0, 0, 0};     // scan-position EOB per plane
  std::uint32_t levels_offset = 0;  // first level index for this leaf
  // §5.20.10.1 cdef_idx for the 64x64 block containing this leaf's origin.
  // Only the first coding block of each 64x64 origin codes the symbol; the
  // writer ignores repeats.  Ignored when CDEF is disabled.
  std::int8_t cdef_idx = 0;
};

// Frame geometry and ordering metadata.  `blocks` (passed to
// assemble_m1_frame) must contain exactly superblock_count() node subtrees, in
// raster superblock order, each rooted by a kFlagSbRoot entry.
struct FramePlan {
  int width = 0;   // visible luma width  (before MiCols rounding)
  int height = 0;  // visible luma height (before MiRows rounding)
  int sb_size = 64;
  // CCSO (§5.20.10.2 read_ccso) per-plane unit-enable grids.  Null/zero when
  // CCSO is disabled.  Values are the ccso_blk symbols (0/1) in unit order.
  const std::uint8_t* ccso_blks[kPlaneCount] = {nullptr, nullptr, nullptr};
  int ccso_blk_cols[kPlaneCount] = {0, 0, 0};
  int ccso_blk_rows[kPlaneCount] = {0, 0, 0};

  int sb_cols() const;
  int sb_rows() const;
  int superblock_count() const;
};

// Emit the complete temporal unit (TD + sequence header + CLK OBUs).
//
// `blocks`            preorder partition nodes as described above.
// `levels`            flat scan-order coefficient levels (Y, U, V per leaf).
// `config`            sequence/frame header configuration (m1_headers.h).
// `out`               receives the frame bytes.
// `error`             set on failure.
// `tile_bytes`        optional: receives the §8.2 tile payload size.
//
// Returns false and sets `error` on any stream/plan inconsistency (bad
// subtree, out-of-range mode, EOB outside the transform, level array overflow,
// entropy-coder failure).
bool assemble_m1_frame(const FramePlan& plan,
                       std::span<const BlockSyntax> blocks,
                       std::span<const std::int32_t> levels,
                       const M1Config& config, std::vector<std::uint8_t>& out,
                       std::string& error,
                       std::size_t* tile_bytes = nullptr);

// IntraJointMode for a (YMode, AngleDeltaY) pair: the reordered non-directional
// index 0..4, or 5 + directional index for directional modes (§5.20.5.5).  The
// decision path needs this to update its own IntraJointModes grid; it is the
// same value the syntax writer derives.
int y_mode_joint(int av2_mode, int angle_delta);

// Diagonal scan (§7.16.3-style up-right) over an adjusted transform size.
// Exported so the decision path and the GPU bridge produce scan-order levels
// with the same order the writer consumes.
void build_scan(int w, int h, std::span<std::uint16_t> scan);

// Number of entries the scan covers (== w * h).
inline int scan_size(int w, int h) { return w * h; }

// One in-frame child slot of a partition (§9.2 Partition_Subsize).  Children
// whose origin lies outside MiRows/MiCols are not coded and are not listed.
struct ChildSlot {
  std::uint16_t mi_row = 0;
  std::uint16_t mi_col = 0;
  std::uint8_t bsize = 0;  // partition::BlockSize
};

// Enumerate the in-frame children of a partition node in decode order
// (SPLIT: TL/TR/BL/BR, HORZ: top/bottom, VERT: left/right).  Returns 0 when the
// partition is not defined for that block size (e.g. HORZ/VERT at 64x8, SPLIT
// below 128x128).  This is the single implementation of the rule; the writer
// validates against it and the GPU bridge builds trees with it.
int partition_child_slots(int mi_row, int mi_col, int bsize, int partition,
                          int mi_rows, int mi_cols, ChildSlot out[4]);

}  // namespace av2enc::m1::syntax
