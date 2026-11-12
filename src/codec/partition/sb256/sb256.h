// AV2 v1.0.0 256x256-superblock reference (CPU correctness oracle).
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//   §5.4.3   sequence_partition_config(): use_256x256_superblock
//   §5.18.1  frame header: SbSize = FrameIsIntra ? 128 : 256 when the flag is 1
//   §5.18.4.4 compute_image_size(): MiCols / MiRows
//   §5.18.7.2 tile_info(), §5.18.7.6 get_seq_sb_size()
//   §5.20.3.1 decode_partition()
//   §5.20.3.2 read_partition(), partition_implied(), init_allowed_partitions()
//   §6.19.3.1 / §9.2 block and partition tables
//   §8.3.2   do_split / do_square_split / rect_type contexts
//   §5.20.9.1 is_inside, §5.20.10.1/2 CDEF and CCSO unit geometry
//
// The two superblock sizes are deliberately not the same thing:
//
//   * get_seq_sb_size() is a *sequence-level* value: 256x256 when the flag is 1.
//     It feeds seq_tile_info_present_flag parsing and the CCSO/GDF unit rules.
//   * SbSize in the frame header is 128x128 for every intra frame, even when
//     use_256x256_superblock is 1 (§5.18.1).  256x256 superblocks exist only in
//     inter frames.
//
// So a still-picture stream can (and must) gate the sequence flag, but its
// partition tree is a 128x128 tree.  This module models both: the full
// 256x256 recursion for inter-frame planning/verification, and the intra
// 128x128 fallback that the minimal stream writer emits.
//
// Scope
// -----
//   * 256x256, 128x128 and 64x64 superblocks (the latter two for cross-checks)
//   * PARTITION_NONE / PARTITION_SPLIT / PARTITION_HORZ / PARTITION_VERT
//   * the 256x256 -> 128x128 and 128x128 -> 64x64 PARTITION_SPLIT entries
//   * boundary / implied-partition recursion for partial superblocks
//   * MiCols/MiRows, superblock grid, block and chroma geometry
//   * §8.3.2 partition contexts, including the +4 do_square_split row that
//     only BLOCK_256X256 uses
//   * CDEF / CCSO / entropy context sizing that depends on SbSize
//   * a deterministic square-leaf reference tree (8x8..64x64) and a minimal
//     intra stream carrying use_256x256_superblock = 1 (sb256_stream.h)
//
// Deliberately not here: inter prediction, SDP, extended partitions, transform
// coefficient coding.  The reference tree uses only DC_PRED / no-residual
// square leaves so the superblock syntax can be gated in isolation.
//
// No AVM, SVT-AV1/AV2, clave, or av2-turbo encoder code was consulted.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::sb256 {

// ---------------------------------------------------------------------------
// §6.19.3.1 Table 6.22 enum_block_size.  Numeric values match the spec.
// ---------------------------------------------------------------------------
enum class BlockSize : std::uint8_t {
  k4x4 = 0,
  k4x8 = 1,
  k8x4 = 2,
  k8x8 = 3,
  k8x16 = 4,
  k16x8 = 5,
  k16x16 = 6,
  k16x32 = 7,
  k32x16 = 8,
  k32x32 = 9,
  k32x64 = 10,
  k64x32 = 11,
  k64x64 = 12,
  k64x128 = 13,
  k128x64 = 14,
  k128x128 = 15,
  k128x256 = 16,
  k256x128 = 17,
  k256x256 = 18,
  k4x16 = 19,
  k16x4 = 20,
  k8x32 = 21,
  k32x8 = 22,
  k16x64 = 23,
  k64x16 = 24,
  k4x32 = 25,
  k32x4 = 26,
  k8x64 = 27,
  k64x8 = 28,
  kInvalid = 255,
};

inline constexpr int kBlockSizeCount = 29;
inline constexpr int kSbSize = 256;
inline constexpr int kSbSizeMi = kSbSize / 4;

bool is_valid_block_size(BlockSize size);
int mi_width_log2(BlockSize size);
int mi_height_log2(BlockSize size);
int num_4x4_blocks_wide(BlockSize size);
int num_4x4_blocks_high(BlockSize size);
int block_width(BlockSize size);
int block_height(BlockSize size);
bool is_square(BlockSize size);
BlockSize superblock_size(int sb_size_pixels);
const char* block_size_name(BlockSize size);

// ---------------------------------------------------------------------------
// §5.4.3 / §5.18.1 superblock configuration
// ---------------------------------------------------------------------------
//
// `use_128x128_superblock` is only present when `use_256x256_superblock` is 0
// (§5.4.3); setting both is a configuration error, not a legal bitstream.
// `frame_is_intra` is the §5.18.1 switch that clamps an intra frame back to a
// 128x128 SbSize.
struct SbConfig {
  bool use_256x256_superblock = false;
  bool use_128x128_superblock = false;
  bool frame_is_intra = true;
};

bool validate_sb_config(const SbConfig& config, std::string& error);

// §5.18.7.6 get_seq_sb_size(): BLOCK_256X256 / BLOCK_128X128 / BLOCK_64X64.
BlockSize get_seq_sb_size(bool use_256x256_superblock,
                          bool use_128x128_superblock);

// §5.18.1: SbSize.  An intra frame with use_256x256_superblock = 1 uses
// BLOCK_128X128; only an inter frame uses BLOCK_256X256.
BlockSize frame_sb_size(const SbConfig& config);
int frame_sb_size_pixels(const SbConfig& config);

// ---------------------------------------------------------------------------
// Partition types
// ---------------------------------------------------------------------------
//
// PARTITION_* compact enum for the four base partitions.  Spec values are
// NONE=0, HORZ=1, VERT=2, ... SPLIT=9; use spec_partition_index() when
// indexing a spec table or a CDF row.
enum class PartitionType : std::uint8_t {
  kNone = 0,
  kSplit = 1,
  kHorz = 2,
  kVert = 3,
};

inline constexpr int kPartitionTypeCount = 4;
const char* partition_name(PartitionType partition);
int spec_partition_index(PartitionType partition);

// §9.2 Partition_Subsize[partition][bSize].  kInvalid when the table entry is
// BLOCK_INVALID.
//   256x256 SPLIT -> 128x128; 128x128 SPLIT -> 64x64; no smaller SPLIT.
//   256x256 HORZ  -> 256x128; 256x256 VERT -> 128x256.
//   256x128 HORZ is invalid; 128x256 VERT is invalid (rect_type implied).
BlockSize partition_subsize(PartitionType partition, BlockSize size);

// §5.20.3.2 rect_type_implied_by_bsize().  Only SHARED_PART rows apply here.
struct RectTypeImplied {
  bool implied = false;
  PartitionType partition = PartitionType::kNone;  // kHorz or kVert
};
RectTypeImplied rect_type_implied_by_bsize(BlockSize size);

// §9.2 Partition_Size_Adjust / Partition_Size_Adjust_Rect_Type.
//   256x256 -> 9 / 12, 256x128 -> 8 / 11, 128x256 -> 7 / 10.
int partition_size_adjust(BlockSize size);
int partition_size_adjust_rect_type(BlockSize size);

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------
//
// One 8-bit 4:2:0 still picture, one tile.  `sb_size` is the *frame* SbSize:
// 256 for a 256-root inter tree, 128 for the intra fallback, 64 for
// cross-checks.  Width/height are visible luma samples, not padded.
struct Geometry {
  int width = 0;
  int height = 0;
  int sb_size = kSbSize;
  bool has_chroma = true;
  bool chroma_subsampling_x = true;
  bool chroma_subsampling_y = true;
  int max_pb_aspect_ratio = 8;  // MaxPbAspectRatio: 2, 4 or 8 (§5.4.3)
};

bool validate_geometry(const Geometry& geometry, std::string& error);

// §5.18.4.4 compute_image_size():
//   MiCols = 2 * ((FrameWidth + 7) >> 3)
//   MiRows = 2 * ((FrameHeight + 7) >> 3)
int mi_cols(const Geometry& geometry);
int mi_rows(const Geometry& geometry);
int sb_size_mi(const Geometry& geometry);
int sb_cols(const Geometry& geometry);
int sb_rows(const Geometry& geometry);

bool is_inside(const Geometry& geometry, int mi_row, int mi_col);
bool block_coded(const Geometry& geometry, int mi_row, int mi_col);

// Visible block rectangle clipped to the frame.
struct BlockRect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
  int visible_width = 0;
  int visible_height = 0;
};
BlockRect block_rect(const Geometry& geometry, int mi_row, int mi_col,
                     BlockSize size);

// Chroma residual size for 4:2:0 or 4:4:4.  BLOCK_INVALID when chroma would be
// smaller than 4x4.
BlockSize chroma_plane_residual_size(const Geometry& geometry,
                                     BlockSize luma_size);

// ChromaMiRow / ChromaMiCol / ChromaMiSize state (§5.20.3.1).
struct ChromaBlock {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
};
bool operator==(const ChromaBlock& a, const ChromaBlock& b);
bool operator!=(const ChromaBlock& a, const ChromaBlock& b);

bool is_chroma_offset_for_subsize(const Geometry& geometry, BlockSize subsize);
bool is_chroma_offset_for_partition(const Geometry& geometry,
                                    PartitionType partition, BlockSize size);

struct ChromaState {
  bool chroma_offset = false;
  ChromaBlock block;
};
ChromaState resolve_chroma(const Geometry& geometry, int mi_row, int mi_col,
                           BlockSize size, PartitionType partition,
                           bool has_chroma, bool chroma_offset,
                           const ChromaBlock& inherited);

// ---------------------------------------------------------------------------
// Boundary / implied partition / allowed set
// ---------------------------------------------------------------------------
struct Boundary {
  bool has_rows = false;
  bool has_cols = false;
  bool implied = false;
  PartitionType implied_partition = PartitionType::kNone;
};

// §5.20.3.2 partition_implied_at_boundary(): the half-block test.
Boundary partition_implied_at_boundary(const Geometry& geometry, int mi_row,
                                       int mi_col, BlockSize size);

// §5.20.3.2 partition_implied(): adds the BLOCK_4X4 / thin-block NONE rule.
Boundary partition_implied(const Geometry& geometry, int mi_row, int mi_col,
                           BlockSize size);

struct PartitionSet {
  bool allowed[kPartitionTypeCount] = {false, false, false, false};
  int count = 0;
  bool is_allowed(PartitionType partition) const;
};

// §5.20.3.2 is_partition_allowed() for TreeType == SHARED_PART and the four
// base partitions.  `chroma_offset` is the recursion argument.
bool is_partition_allowed(const Geometry& geometry, int mi_row, int mi_col,
                          BlockSize size, PartitionType partition,
                          bool has_chroma, bool chroma_offset);

// §5.20.3.2 init_allowed_partitions().  If the spec allows nothing, NONE is
// forced, exactly like the decoder.
PartitionSet init_allowed_partitions(const Geometry& geometry, int mi_row,
                                     int mi_col, BlockSize size,
                                     bool has_chroma, bool chroma_offset);

// ---------------------------------------------------------------------------
// §8.3.2 contexts
// ---------------------------------------------------------------------------
class MiSizeGrid {
 public:
  MiSizeGrid() = default;
  explicit MiSizeGrid(const Geometry& geometry);

  void reset(const Geometry& geometry);
  bool in_bounds(int mi_row, int mi_col) const;
  BlockSize at(int mi_row, int mi_col) const;
  void set(int mi_row, int mi_col, BlockSize size);
  void fill_block(int mi_row, int mi_col, BlockSize size);

  int rows() const { return rows_; }
  int cols() const { return cols_; }

 private:
  int rows_ = 0;
  int cols_ = 0;
  std::vector<BlockSize> cells_;
};

struct Contexts {
  int do_split = 0;
  int do_square_split = 0;
  int rect_type = 0;
};

// §8.3.2 contexts for the luma/shared plane (PlaneStart == 0):
//   do_split        = Partition_Size_Adjust[bSize]*4 + left*2 + above
//   do_square_split = (bSize == 256x256 ? 4 : 0) + left*2 + above
//   rect_type       = Partition_Size_Adjust_Rect_Type[bSize]*4 + ...
// At a 256x256 root with no decided neighbours: 36 / 4 / 48.
Contexts partition_contexts(const Geometry& geometry, int mi_row, int mi_col,
                            BlockSize size, const MiSizeGrid& sizes);

// Default CDF rows.  The [2] dimension is PlaneStart (0 = luma/shared,
// 1 = chroma); SHARED_PART always uses row 0.  The 256 SB flag adds no CDF
// row; it changes the *context* and moves SPLIT to do_square_split.
const std::uint16_t* do_split_cdf_row(int plane, int context);
const std::uint16_t* do_square_split_cdf_row(int plane, int context);
const std::uint16_t* rect_type_cdf_row(int plane, int context);

// ---------------------------------------------------------------------------
// SbSize-dependent unit sizing
// ---------------------------------------------------------------------------
//
// CDEF is always a 64x64 unit (§5.20.10.1, cdefSize4 from BLOCK_64X64); a
// 256x256 superblock therefore contains 4x4 = 16 CDEF units.  The
// cdef_index0 context's above-row test uses Mi_Width_Log2[SbSize] (6 at 256).
int cdef_unit_size_pixels();
int cdef_units_per_sb(int sb_size_pixels);   // 1 / 2 / 4
int cdef_sb_width_log2(int sb_size_pixels);  // 4 / 5 / 6

// CCSO (§5.18.7.12): if ccso_unit_matches_sb_size, the unit is SbSize;
// otherwise it is 64/128/256 chosen from the tile-start alignment.
int ccso_luma_size_log2(int sb_size_pixels, bool ccso_unit_matches_sb_size,
                        unsigned tile_start_or);
int ccso_luma_size_log2_matching_sb(int sb_size_pixels);  // 6 / 7 / 8
int ccso_luma_size_log2_from_tile_alignment(unsigned tile_start_or);
int ccso_unit_mi_shift(int sb_size_pixels);  // log2(unit) - 2 => 4 / 5 / 6

// §6.19.2.1 tile-level entropy context extents.  LeftMiSizes/LeftDcContext are
// sized numSbs * sbSize4; Above* are sized the same way across MiCols.
int entropy_left_context_rows(const Geometry& geometry);
int entropy_above_context_cols(const Geometry& geometry);

// ---------------------------------------------------------------------------
// Partition syntax plan
// ---------------------------------------------------------------------------
enum class PartitionSymbolKind : std::uint8_t {
  kDoSplit = 0,
  kDoSquareSplit = 1,
  kRectType = 2,
};
struct PartitionSymbol {
  PartitionSymbolKind kind = PartitionSymbolKind::kDoSplit;
  int value = 0;
  int context = 0;
};

struct Node;

// §5.20.3.2 read_partition() symbol sequence for one node.  Returns no symbols
// when the partition is implied or only one partition is allowed.  The
// caller supplies the §8.3.2 contexts.
bool plan_partition_syntax(const Node& node, const Contexts& contexts,
                           std::vector<PartitionSymbol>& out,
                           std::string& error);

// ---------------------------------------------------------------------------
// Reference partition tree
// ---------------------------------------------------------------------------
struct Node {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  PartitionType partition = PartitionType::kNone;
  int depth = 0;
  bool has_chroma = false;
  bool chroma_offset = false;
  bool forced = false;
  Boundary boundary;
  PartitionSet allowed;
  ChromaBlock chroma;
  std::vector<Node> children;
};

struct FrameTree {
  Geometry geometry;
  std::vector<Node> superblocks;  // raster order
  int nodes = 0;
  int leaves = 0;
};

struct LeafBlock {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  int depth = 0;
  bool has_chroma = false;
  bool chroma_offset = false;
  ChromaBlock chroma;
};

// Deterministic reference tree.  Full roots use PARTITION_SPLIT down to
// 64x64; boundary-forced roots use the implied HORZ/VERT, and rectangular
// blocks are split toward a square leaf.  Leaves are square, 8x8..64x64 for
// the geometries the minimal stream supports; arbitrary partial sizes may
// recurse deeper in the geometry-only reference.
bool build_reference_frame(const Geometry& geometry, FrameTree& out,
                           std::string& error);

bool validate_frame(const Geometry& geometry, const FrameTree& frame,
                    std::string& error);

void collect_leaves(const Node& node, std::vector<LeafBlock>& out);
int count_nodes(const Node& node);
int count_leaves(const Node& node);

}  // namespace av2enc::sb256
