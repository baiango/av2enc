// AV2 v1.0.0 128x128-superblock reference (CPU correctness oracle).
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//   §5.4.3   sequence_partition_config(): use_128x128_superblock
//   §5.18.4.4 compute_image_size(): MiCols / MiRows
//   §5.18.7.6 get_seq_sb_size()
//   §5.20.3.1 decode_partition()
//   §5.20.3.2 read_partition(), partition_implied(), init_allowed_partitions()
//   §6.19.3.1 / §9.2 block and partition tables
//   §8.3.2   do_split / do_square_split / rect_type contexts
//
// Scope
// -----
//   * 128x128 and 64x64 superblocks (the latter only for cross-checks)
//   * PARTITION_NONE / PARTITION_SPLIT / PARTITION_HORZ / PARTITION_VERT
//   * the 128x128 -> 64x64 PARTITION_SPLIT entry from §9.2
//   * boundary / implied-partition recursion for partial superblocks
//   * MiCols/MiRows, superblock grid, block and chroma geometry
//   * §8.3.2 partition contexts, including do_square_split, which the M1
//     writer currently does not code (see README.md call sites)
//   * a deterministic all-64x64-leaf reference tree and a minimal 128-SB
//     stream (sb128_stream.h)
//
// Deliberately not here: inter prediction, SDP, extended partitions, 256x256
// superblocks, transform coefficient coding.  The reference tree uses only
// DC_PRED / no-residual leaves so the SB syntax can be gated in isolation.
//
// No AVM, SVT-AV1/AV2, clave, or av2-turbo encoder code was consulted.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::sb128 {

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
inline constexpr int kSbSize = 128;
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

// PARTITION_* compact enum for the four base partitions.  The spec enum
// values are NONE=0, HORZ=1, VERT=2, ... SPLIT=9; use spec_partition_index()
// when indexing a spec table or a CDF row.
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
// BLOCK_INVALID.  The 128x128 SPLIT entry is BLOCK_64X64; SPLIT below 128 is
// invalid, which is why a 64x64 tree cannot use do_square_split.
BlockSize partition_subsize(PartitionType partition, BlockSize size);

// §5.20.3.2 rect_type_implied_by_bsize().  Only the SHARED_PART rows apply
// here (no CHROMA_PART).
struct RectTypeImplied {
  bool implied = false;
  PartitionType partition = PartitionType::kNone;  // kHorz or kVert
};
RectTypeImplied rect_type_implied_by_bsize(BlockSize size);

// §9.2 Partition_Size_Adjust / Partition_Size_Adjust_Rect_Type.  These are the
// CDF row multipliers that move 128x128 from context 12 (the 64x64 row) to
// context 24, and SPLIT onto the separate do_square_split table.
int partition_size_adjust(BlockSize size);
int partition_size_adjust_rect_type(BlockSize size);

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

// One 8-bit 4:2:0 still picture, one tile.  `sb_size` is 128 for this module;
// 64 is accepted only so tests can compare the two trees.  Width/height are
// visible luma samples, not padded.
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

// Visible block rectangle clipped to the frame.  x/y/width/height are the
// coded block; visible_width/visible_height are clipped to the picture.
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

// Chroma block at half resolution for 4:2:0.  `chroma_plane_residual_size`
// returns BLOCK_INVALID when the chroma block would be smaller than 4x4.
BlockSize chroma_plane_residual_size(const Geometry& geometry,
                                     BlockSize luma_size);

// ChromaMiRow / ChromaMiCol / ChromaMiSize state (§5.20.3.1).  A leaf that
// codes chroma uses ChromaMiSize, which can differ from its own luma size.
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

// §5.20.3.2 partition_implied_at_boundary().  The half-block test is
//   has_rows = (r + (Num_4x4_Blocks_High[bSize] >> 1)) < MiRows
//   has_cols = (c + (Num_4x4_Blocks_Wide[bSize] >> 1)) < MiCols
Boundary partition_implied_at_boundary(const Geometry& geometry, int mi_row,
                                       int mi_col, BlockSize size);

// §5.20.3.2 partition_implied().  Adds the BLOCK_4X4 / thin-block NONE rule.
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

// Per-4x4-cell map of already-decided block sizes.  Outside cells and unset
// cells behave like the spec's BLOCK_256X256 sentinel (never "smaller").
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
//   do_split        = Partition_Size_Adjust[bSize]*4 + left_smaller*2 + above_smaller
//   do_square_split = (bSize == 256x256 ? 4 : 0) + left*2 + above
//   rect_type       = Partition_Size_Adjust_Rect_Type[bSize]*4 + ...
// `sizes` must contain every block decided before this node.
Contexts partition_contexts(const Geometry& geometry, int mi_row, int mi_col,
                            BlockSize size, const MiSizeGrid& sizes);

// Default CDF rows.  The [2] dimension is PlaneStart (0 = luma/shared,
// 1 = chroma); SHARED_PART always uses row 0.  The 128 SB flag does not add a
// CDF row; it changes the *context* via Partition_Size_Adjust and moves SPLIT
// to the do_square_split table.  These return nullptr for an invalid index.
const std::uint16_t* do_split_cdf_row(int plane, int context);
const std::uint16_t* do_square_split_cdf_row(int plane, int context);
const std::uint16_t* rect_type_cdf_row(int plane, int context);

// Context-derivation helpers whose value changes with a 128 SB:
//   CDEF §8.3.2 cdef_index0 uses shift = Mi_Width_Log2[SbSize] (5, not 4).
//   CCSO §5.18.7.12 CcsoLumaSizeLog2 = 6 for 64 SB, 7 for 128 SB.
int cdef_sb_width_log2(int sb_size_pixels);
int ccso_luma_size_log2(int sb_size_pixels);
int ccso_unit_mi_shift(int sb_size_pixels);

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

// Build the deterministic reference tree: 128x128 SBs, PARTITION_SPLIT at a
// full root, boundary-forced HORZ/VERT at a partial root, 128x64 -> VERT and
// 64x128 -> HORZ, and PARTITION_NONE 64x64 leaves.  This is the tree the
// minimal stream writes.
bool build_reference_frame(const Geometry& geometry, FrameTree& out,
                           std::string& error);

bool validate_frame(const Geometry& geometry, const FrameTree& frame,
                    std::string& error);

void collect_leaves(const Node& node, std::vector<LeafBlock>& out);
int count_nodes(const Node& node);
int count_leaves(const Node& node);

}  // namespace av2enc::sb128
