// AV2 M1 still-picture syntax writer.
//
// This module owns *only* bitstream syntax: partition symbols, mode info,
// transform-tree/EOB/coefficient coding, and OBU assembly.  It does not decide
// anything and it does not reconstruct anything: it consumes a block-result
// stream (BlockSyntax + a flat coefficient-level array) produced by _either_
// the CPU decision/reconstruction path (src/encoder/m1/still_encoder.cpp) or the
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
// Bit-exactness: for the same decisions, assemble_still_frame() emits exactly the
// bytes the pre-refactor inline writer in still_encoder.cpp emitted.  See
// src/encoder/m1/stream/README.md for the ordering and CDF-context rules a GPU
// orchestrator must satisfy.
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0.
//   §5.20.3.2 partition syntax, §5.20.5 mode info, §5.20.7 residual coding,
//   §8.3.2 partition contexts, §8.2 symbol coder (src/entropy/av2).
#pragma once

#include "pipeline/still/still_headers.h"
#include "codec/entropy/avg_cdf/avg_cdf_bank.h"
#include "codec/intra/palette/core/palette.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::still::syntax {

inline constexpr int kPlaneCount = 3;
inline constexpr int kMaxLumaTxParts = 4;

// One luma transform block of a TX_MODE_SELECT coding block (§5.20.6.3).
struct LumaTxPart {
  std::uint8_t tx_size = 0;      // AV2 TX_SIZE 0..24
  std::uint8_t tx_type = 0;      // txv::TxType
  std::uint8_t sec_tx_type = 0;  // 0 = no IST
  std::uint8_t most_probable_stx_set = 0;
  std::uint32_t eob = 0;         // scan-position EOB
};

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
  // exactly like still_encoder's `if (sb.mi_col == 0) clear_left_context()`.
  kFlagSbRoot = 1u << 2,
  // This leaf is coded with a luma palette (§5.20.8).  The extra syntax lives
  // at the palette position (after UV mode, before DIP) and is driven by the
  // side-array entry at BlockSyntax::palette_index.
  kFlagIsPalette = 1u << 3,
  // The chroma-ownership fields (has_chroma / chroma_bsize / chroma_mi_row /
  // chroma_mi_col) are valid.  Producers that predate sub-8 blocks (the GPU
  // bridge, the basic path) leave this clear, which keeps the historical
  // "derive chroma from the luma leaf" behaviour byte-identical.
  kFlagHasChromaInfo = 1u << 4,
  // Intra SDP luma/chroma split node (§5.20.3.1).  It codes no partition
  // syntax; its two children are the luma root and then the chroma root.
  kFlagSdpSplit = 1u << 5,
  // This leaf is an intra-block-copy block (§5.20.5.3/.4).  No Y/UV mode or
  // palette symbol is coded; use_intrabc + skip_flag + intrabc_info replace
  // them.  The block vector is in 1/8 luma samples.
  kFlagIsIbc = 1u << 6,
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
  std::uint8_t partition = 0;     // spec PartitionType value 0..9
  // bit p = allowed[PartitionType p] (10-bit spec enum).  The rect-only path
  // only sets bits 0..3; extended partitions use bits 3..8 and 9 (SPLIT).
  std::uint16_t allowed_mask = 0;
  std::uint8_t flags = 0;         // BlockFlags
  std::uint16_t subtree = 0;      // entries in the child subtrees
  // §5.20.5.7 intra_segment_id().  Always 0 unless the frame signals
  // segmentation with active features; the real per-SB path fills it from the
  // planner.
  std::uint8_t segment_id = 0;

  // --- chroma ownership (§5.20.3.1, sub-8 leaves) ---
  // Valid only when kFlagHasChromaInfo is set.  `has_chroma` is the decoded
  // HasChroma value for this leaf (false on the first children of a sub-8
  // split, which code no UV mode/residual).  When true the leaf owns the whole
  // inherited chroma block described by ChromaMiRow/Col/Size; `chroma_bsize`
  // is ChromaMiSize in luma 4x4 units, so the chroma residual size is
  // partition::chroma_plane_residual_size(chroma_bsize).
  std::uint8_t has_chroma = 1;
  std::uint8_t chroma_offset = 0;     // decoded chromaOffset recursion arg
  std::uint8_t chroma_bsize = 0;      // partition::BlockSize (ChromaMiSize)
  std::uint16_t chroma_mi_row = 0;    // luma 4x4 units
  std::uint16_t chroma_mi_col = 0;
  // §6.19.3.1 TreeType for an SDP frame (0 SHARED, 1 LUMA, 2 CHROMA).
  // CHROMA_PART codes partition symbols from PlaneStart 1 and skips luma mode,
  // luma residual, TX size, CDEF/CCSO/delta-q and palette/DIP.
  std::uint8_t tree_type = 0;

  // --- leaf mode info ---
  std::uint8_t y_mode = 0;        // AV2 YMode 0..12 (§6.19.5.5)
  std::int8_t y_angle_delta = 0;  // AngleDeltaY -3..3
  // §5.20.5.5 mrl_index / mrl_sec_index.  Signals only for directional luma
  // modes when the sequence header enables MRL; ignored/zero otherwise.
  std::int8_t mrl_index = 0;      // 0..3
  std::uint8_t mrl_sec_index = 0; // 0/1; requires mrl_index > 0 and not 4x4
  // §5.20.7.32 read_cfl_alphas() when the block is CfL: cfl_index 0 explicit /
  // 1 derived, and the explicit sign/magnitude symbols.
  std::uint8_t cfl_index = 1;
  std::uint8_t cfl_alpha_signs = 0;
  std::uint8_t cfl_alpha_u = 0;
  std::uint8_t cfl_alpha_v = 0;
  // §5.20.7.32 cfl_mh_dir for cfl_index == CFL_MULTI (MHCCP).
  std::uint8_t cfl_mh_dir = 0;
  // §5.20.7.9 dip_mode_info().  Signalled only for luma DC blocks when the
  // sequence enables DIP and the size/palette predicate holds.
  std::uint8_t use_dip = 0;
  std::uint8_t dip_transpose = 0;
  std::uint8_t dip_mode = 0;      // 0..5
  std::uint8_t uv_mode = 0;       // AV2 UV mode (kDcPred when is_cfl)

  // --- leaf residual ---
  std::uint8_t tx_size[kPlaneCount] = {0, 0, 0};  // AV2 TX_SIZE 0..24
  // AV2 TX_TYPE (txv::TxType).  Only luma can signal a non-DCT type; chroma is
  // derived from the DCT-only rule.  Ignored for eob <= 1 (spec forces DCT).
  std::uint8_t tx_type[kPlaneCount] = {0, 0, 0};
  std::uint32_t eob[kPlaneCount] = {0, 0, 0};     // scan-position EOB per plane
  std::uint32_t levels_offset = 0;  // first level index for this leaf
  // §5.20.8 palette side-array index (ignored unless kFlagIsPalette).  The
  // PaletteResult is too large to inline in this POD used by the GPU bridge.
  std::uint32_t palette_index = 0;
  // §5.20.10.1 cdef_idx for the 64x64 block containing this leaf's origin.
  // Only the first coding block of each 64x64 origin codes the symbol; the
  // writer ignores repeats.  Ignored when CDEF is disabled.
  std::int8_t cdef_idx = 0;
  // §5.20.5.4 intrabc_info() for kFlagIsIbc leaves.  `ibc_mode` is 0 = NEWMV
  // (MVD present) / 1 = NEARMV (RefStackMv only); `ibc_ref_mv_idx` is the DRL
  // index; `ibc_bv_*` is the absolute block vector in 1/8 luma samples.
  // `ibc_skip` = 1 codes no residual (exact copy), 0 codes a normal residual.
  std::uint8_t ibc_skip = 1;
  std::uint8_t ibc_mode = 1;
  std::uint8_t ibc_ref_mv_idx = 0;
  std::int32_t ibc_bv_row = 0;
  std::int32_t ibc_bv_col = 0;
  // §5.20.8.2 secondary transform (IST), luma only.  sec_tx_type 0 means no
  // IST; the set is 0..6 (0..3 for a large ADST_ADST block).  Both fields are
  // ignored when enable_intra_ist is 0 or the block is not IST-applicable.
  std::uint8_t sec_tx_type = 0;
  std::uint8_t most_probable_stx_set = 0;
  // §5.20.7.27 cctx_type (0 = CCTX_NONE) for plane 1.  Coded after the U EOB
  // and before the coefficients when enable_cctx is on and the U eob is
  // neither 0 nor 1; otherwise it must be 0.
  std::uint8_t cctx_type = 0;

  // --- §5.20.6.1-.3 TX_MODE_SELECT luma parts ---
  // tx_part_count == 0 keeps the historical single-transform layout and reads
  // tx_size[0]/tx_type[0]/eob[0].  When tx_mode_select is on, tx_part_count is
  // 1..4 and the luma coefficients of this leaf are the parts' eobs, in part
  // order (decode order), starting at levels_offset.  Chroma remains a single
  // transform per plane after the luma levels.
  std::uint8_t tx_part_count = 0;
  std::uint8_t tx_partition = 0;  // txpartition::TxPartition value
  LumaTxPart tx_parts[kMaxLumaTxParts];
};

// Frame geometry and ordering metadata.  `blocks` (passed to
// assemble_still_frame) must contain exactly superblock_count() node subtrees, in
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
// `config`            sequence/frame header configuration (still_headers.h).
// `out`               receives the frame bytes.
// `error`             set on failure.
// `tile_bytes`        optional: receives the §8.2 tile payload size.
// `palette_results`   optional side array indexed by BlockSyntax::palette_index
//                     for leaves with kFlagIsPalette; required when the config
//                     enables palette and a leaf sets that flag.
//
// Returns false and sets `error` on any stream/plan inconsistency (bad
// subtree, out-of-range mode, EOB outside the transform, level array overflow,
// entropy-coder failure).
bool assemble_still_frame(const FramePlan& plan,
                       std::span<const BlockSyntax> blocks,
                       std::span<const std::int32_t> levels,
                       const StillConfig& config, std::vector<std::uint8_t>& out,
                       std::string& error,
                       std::size_t* tile_bytes = nullptr,
                       std::span<const palette::PaletteResult> palette_results =
                           {},
                       std::span<const std::uint16_t> cdf_seed = {},
                       std::vector<std::uint16_t>* cdf_final = nullptr);

// ---------------------------------------------------------------------------
// Opaque CDF image: the §8.2 CDF arrays of one tile, packed without the
// writer's neighbour/context state.  This is the seed/export surface the
// average-CDF bank (§8.2.2 frame->tile copy, §8.2.4 Saved* average, §7.5
// frame-end update) needs; it deliberately carries no partition/mode context.
// ---------------------------------------------------------------------------

// Number of uint16_t entries in one image.
std::size_t cdf_image_words();

// §9.3 defaults for the coefficient-CDF slice selected by the raw base_q_idx.
bool cdf_image_defaults_for_base_q_idx(int base_q_idx,
                                       std::vector<std::uint16_t>& out,
                                       std::string* error = nullptr);

// Builds the avg_cdf::CdfBank table list over four equal-sized images.  The
// bank may then drive `load_tile_cdfs()` / `save_tile_cdfs()` /
// `average_at_frame_end()`; the caller copies `tile` into/out of the syntax
// writer through the `cdf_seed` / `cdf_final` arguments above.
bool cdf_image_tables(std::vector<std::uint16_t>& frame,
                      std::vector<std::uint16_t>& tile,
                      std::vector<std::uint16_t>& saved,
                      std::vector<std::uint16_t>& defaults,
                      std::vector<entropy::avg_cdf::CdfTable>& tables,
                      std::string* error = nullptr);

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

}  // namespace av2enc::still::syntax
