// CPU reference for the AV2 v1.0.0 recursive partition tree.
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §5.20.3.1 (decode partition syntax),
// §5.20.3.2 (read partition syntax), §6.19.3.1 (partition semantics),
// and §9.2 (block / partition tables).  No AVM, SVT, clave, or av2-turbo
// encoder code was consulted.
//
// Scope
// -----
//   * PARTITION_NONE / PARTITION_SPLIT / PARTITION_HORZ / PARTITION_VERT
//   * block sizes 4x4 through 64x64, plus 128x128 superblock geometry
//   * rectangular sizes produced by those splits (4x8, 8x4, ... 64x8)
//   * boundary geometry, neighbour availability, partition_implied*,
//     and the spec's allowed-partition derivation
//   * a top-down decision API that accepts caller-supplied candidate costs
//     and returns a deterministic tree
//
// Deliberately not here
// ---------------------
//   * entropy coding / CDF assembly / bitstream writing
//   * extended 3-way and 4-way partitions, SDP, BRU
//   * chroma prediction or chroma-only partition trees
//
// Spec fact that surprises most readers: in AV2 v1.0.0, Table 9.x
// (Partition_Subsize) only defines PARTITION_SPLIT for BLOCK_128X128 and
// BLOCK_256X256.  A 64x64 block cannot signal PARTITION_SPLIT.  A square
// 32x32 decomposition below a 64x64 block is expressed as HORZ then VERT
// (or VERT then HORZ), which is exactly what the M0 reference stream does.
// PARTITION_SPLIT therefore only does useful work with a 128x128 superblock.
// This module reports SPLIT as *not allowed* wherever the spec says so; do
// not "fix" that or the decoder will reject the stream.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::partition {

// Infinite cost means "this candidate is unavailable".  Costs may not be
// negative or NaN; the decision API rejects those as programming errors.
inline constexpr double kInfiniteCost =
    std::numeric_limits<double>::infinity();

// Smallest luma block the M1 encoder will create.  The spec also allows 4x4;
// callers may raise the floor (for example 8 for a fast-decode profile) but
// never below 4.
inline constexpr int kMinBlockPixels = 4;

// AV2 superblock sizes in scope.  256x256 uses the sequence-header flag and is
// coded as a 256 root only on inter frames (§5.18.1 clamps intra SbSize to 128).
inline constexpr int kMinSuperblockPixels = 64;
inline constexpr int kMaxSuperblockPixels = 256;

// AV2 Table 6.22 (enum_block_size).  The numeric values are the spec's enum
// values, so spec tables can be indexed directly.  They are NOT a compact
// ordinal; kInvalid is 255.
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

bool is_valid_block_size(BlockSize size);
int block_width(BlockSize size);
int block_height(BlockSize size);
int mi_width_log2(BlockSize size);   // Mi_Width_Log2 (§9.2)
int mi_height_log2(BlockSize size);  // Mi_Height_Log2 (§9.2)
int num_4x4_blocks_wide(BlockSize size);
int num_4x4_blocks_high(BlockSize size);
bool is_square(BlockSize size);
BlockSize superblock_size(int sb_size_pixels);
const char* block_size_name(BlockSize size);

// PARTITION_NONE / SPLIT / HORZ / VERT.  This is a compact enum for the M1
// subset, not the spec's 10-value enum.  spec_partition_index() maps to the
// spec values (NONE=0, HORZ=1, VERT=2, ... SPLIT=9).
enum class PartitionType : std::uint8_t {
  kNone = 0,
  kSplit = 1,
  kHorz = 2,
  kVert = 3,
};

inline constexpr int kPartitionTypeCount = 4;

const char* partition_name(PartitionType partition);
int spec_partition_index(PartitionType partition);

// Partition_Subsize[partition][bSize] (§9.2).  Returns BlockSize::kInvalid
// when the table entry is BLOCK_INVALID.
BlockSize partition_subsize(PartitionType partition, BlockSize size);

// rect_type_implied_by_bsize() (§5.20.3.2).  `implied` is false when
// rect_type is RECT_INVALID and therefore must be coded (subject to the
// allowed-partition set).
struct RectTypeImplied {
  bool implied = false;
  PartitionType partition = PartitionType::kNone;  // kHorz or kVert
};
RectTypeImplied rect_type_implied_by_bsize(BlockSize size);

// Partition_Size_Adjust / Partition_Size_Adjust_Rect_Type (§9.2).
int partition_size_adjust(BlockSize size);
int partition_size_adjust_rect_type(BlockSize size);

// One 8-bit 4:2:0 still picture, one tile.  Width/height are luma samples and
// are not padded.  `sb_size` is 64, 128 or 256.  The default 4:2:0 subsampling and
// has_chroma=true are the M1 configuration; monochrome and 4:4:4 callers can
// flip the flags, but 4:2:2 is not modelled.
struct FrameGeometry {
  int width = 0;
  int height = 0;
  int sb_size = 64;
  bool has_chroma = true;
  bool chroma_subsampling_x = true;
  bool chroma_subsampling_y = true;
  int max_pb_aspect_ratio = 8;  // MaxPbAspectRatio: 2, 4, or 8 (§5.4.3)
};

bool validate_geometry(const FrameGeometry& geometry, std::string& error);

int mi_cols(const FrameGeometry& geometry);  // §5.18.4.4, even 4x4 units
int mi_rows(const FrameGeometry& geometry);  // §5.18.4.4, even 4x4 units
int sb_size_mi(const FrameGeometry& geometry);
int sb_cols(const FrameGeometry& geometry);
int sb_rows(const FrameGeometry& geometry);

// Tile-relative availability (§5.20.9.1 is_inside).  One tile in M1, so this
// is frame-relative.  A future tile implementation narrows these bounds; the
// partition geometry itself always uses MiRows/MiCols, as the spec does.
// MiRows/MiCols are rounded to 8 pixels (§5.18.4.4), so a neighbour can be
// "available" for context while still lying outside the visible width/height;
// pixel reads must clip to width/height separately.
bool is_inside(const FrameGeometry& geometry, int mi_row, int mi_col);
bool block_coded(const FrameGeometry& geometry, int mi_row, int mi_col);

struct BlockRect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
  int visible_width = 0;
  int visible_height = 0;
};
BlockRect block_rect(const FrameGeometry& geometry, int mi_row, int mi_col,
                     BlockSize size);

struct NeighborAvailability {
  bool left = false;         // (mi_row, mi_col - 1)
  bool above = false;        // (mi_row - 1, mi_col)
  bool above_left = false;   // (mi_row - 1, mi_col - 1)
  // First 4x4 cell beyond the block edges, for intra sample availability:
  bool above_right = false;  // (mi_row - 1, mi_col + numWide4x4)
  bool below_left = false;   // (mi_row + numHigh4x4, mi_col - 1)
};
NeighborAvailability neighbor_availability(const FrameGeometry& geometry,
                                           int mi_row, int mi_col,
                                           BlockSize size);

// partition_implied_at_boundary() (§5.20.3.2 helper).  has_rows/has_cols use
// the half-block test:
//   has_rows = (mi_row + (numHigh4x4 >> 1)) < MiRows
//   has_cols = (mi_col + (numWide4x4 >> 1)) < MiCols
struct PartitionBoundary {
  bool has_rows = false;
  bool has_cols = false;
  bool implied = false;
  PartitionType implied_partition = PartitionType::kNone;
};
PartitionBoundary partition_implied_at_boundary(const FrameGeometry& geometry,
                                                int mi_row, int mi_col,
                                                BlockSize size);

// partition_implied() (§5.20.3.2).  Adds the spec's early cases:
//   BLOCK_4X4 or bSize >= BLOCK_4X32  ->  implied PARTITION_NONE
// otherwise it is partition_implied_at_boundary().  CHROMA_PART-specific
// cases do not apply because this module models SHARED_PART.
PartitionBoundary partition_implied(const FrameGeometry& geometry,
                                    int mi_row, int mi_col, BlockSize size);

struct PartitionSet {
  bool allowed[kPartitionTypeCount] = {false, false, false, false};
  int count = 0;
  bool is_allowed(PartitionType partition) const;
};

// init_allowed_partitions() (§5.20.3.2).  If no partition is spec-valid the
// spec forces PARTITION_NONE; that fallback is mirrored here.
PartitionSet init_allowed_partitions(const FrameGeometry& geometry,
                                     int mi_row, int mi_col, BlockSize size,
                                     bool has_chroma, bool chroma_offset);

// is_partition_allowed() (§5.20.3.2), restricted to the four M1 partitions and
// SHARED_PART.  enable_ext_partitions is treated as 0.
//   has_chroma    = hasChroma recursion parameter
//   chroma_offset = chromaOffset recursion parameter
bool is_partition_allowed(const FrameGeometry& geometry, int mi_row,
                          int mi_col, BlockSize size, PartitionType partition,
                          bool has_chroma, bool chroma_offset);

// is_chroma_offset_for_subsize / _partition (§5.20.3.2).  Only the six
// non-extended partitions exist here, so no HORZ_3/VERT_3 middle-block case.
bool is_chroma_offset_for_subsize(const FrameGeometry& geometry,
                                  BlockSize subsize);
bool is_chroma_offset_for_partition(const FrameGeometry& geometry,
                                    PartitionType partition, BlockSize size);

// get_plane_residual_size(subsize, plane=1) for 4:2:0.  Returns
// BlockSize::kInvalid when the chroma block would be smaller than 4x4.
// Returns the luma size unchanged for non-subsampled chroma.  4:2:2 is not
// modelled and returns kInvalid.
BlockSize chroma_plane_residual_size(const FrameGeometry& geometry,
                                     BlockSize luma_size);

// Per-4x4-cell map of decided block sizes.  Used for the do_split /
// do_square_split / rect_type contexts (§8.3.2).  Unset cells and cells
// outside the frame behave like the spec's BLOCK_256X256 sentinel, i.e. they
// are never "smaller than" the current block.
class MiSizeGrid {
 public:
  MiSizeGrid() = default;
  explicit MiSizeGrid(const FrameGeometry& geometry);

  void reset(const FrameGeometry& geometry);
  bool in_bounds(int mi_row, int mi_col) const;
  BlockSize at(int mi_row, int mi_col) const;  // kInvalid when unset/outside
  void set(int mi_row, int mi_col, BlockSize size);
  void fill_block(int mi_row, int mi_col, BlockSize size);

  int rows() const { return rows_; }
  int cols() const { return cols_; }

 private:
  int rows_ = 0;
  int cols_ = 0;
  std::vector<BlockSize> cells_;
};

struct PartitionContexts {
  int do_split = 0;
  int do_square_split = 0;
  int rect_type = 0;
};

// Context derivation for the three partition syntax elements (§8.3.2).
// The caller must have filled every already-decided leaf into `sizes`.
PartitionContexts partition_contexts(const FrameGeometry& geometry,
                                     int mi_row, int mi_col, BlockSize size,
                                     const MiSizeGrid& sizes);

// Candidate costs supplied by the caller (RDO, SATD, or a test model).
//
//   leaf          cost of coding this node as one PARTITION_NONE block,
//                 including mode/transform/coefficient bits, but excluding
//                 the partition flag
//   none_signal   cost of coding do_split = 0 (ignored when the partition is
//                 forced by the spec and no syntax is coded)
//   split_signal  cost of coding PARTITION_SPLIT
//   horz_signal   cost of coding PARTITION_HORZ
//   vert_signal   cost of coding PARTITION_VERT
//
// Use kInfiniteCost for an unavailable candidate.  Negative values and NaN
// are rejected by the decision API.
struct NodeCosts {
  double leaf = kInfiniteCost;
  double none_signal = 0.0;
  double split_signal = 0.0;
  double horz_signal = 0.0;
  double vert_signal = 0.0;
};

struct PartitionNodeInfo {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  int depth = 0;
  bool has_chroma = false;
  bool chroma_offset = false;
  PartitionBoundary boundary;
  PartitionSet allowed;
};

using CostFunction = std::function<NodeCosts(const PartitionNodeInfo&)>;

struct DecisionOptions {
  // Minimum luma width and height of a block created by an *optional* split.
  // These are search-policy limits, not geometry: a spec-forced partition is
  // always honoured, and a node whose half-size leaves the frame may relax the
  // limits because the spec forbids PARTITION_NONE there.
  int min_block_size_px = kMinBlockPixels;
  // Maximum partition recursion depth for optional splits.  Root superblock
  // is depth 0.  Boundary-forced partitions may exceed it.
  int max_depth = 32;
  bool allow_rect = true;   // HORZ / VERT
  bool allow_split = true;  // PARTITION_SPLIT (only legal for 128 SB roots)
};

struct PartitionNode {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  PartitionType partition = PartitionType::kNone;
  int depth = 0;
  bool has_chroma = false;
  bool chroma_offset = false;
  // Spec-derived flags, copied from the node info the cost callback saw.
  // The integration pass uses these to decide whether do_split,
  // do_square_split, and rect_type are coded.
  PartitionBoundary boundary;
  PartitionSet allowed;
  // True when the spec codes no partition syntax at this node: either the
  // boundary implies the partition or only one partition is allowed.
  bool forced = false;
  double cost = 0.0;
  // In-frame children only, in decode order.  A child whose origin is outside
  // MiRows/MiCols is skipped by decode_partition and is not represented.
  std::vector<PartitionNode> children;
};

enum class DecisionStatus : std::uint8_t {
  kOk = 0,
  kInvalidGeometry,
  kInvalidCost,
  kNoValidPartition,
};
const char* decision_status_name(DecisionStatus status);

struct PartitionDecision {
  DecisionStatus status = DecisionStatus::kInvalidGeometry;
  PartitionNode root;
  double cost = 0.0;
  int nodes = 0;
  int leaves = 0;
  std::string error;
};

struct FrameDecision {
  DecisionStatus status = DecisionStatus::kInvalidGeometry;
  std::vector<PartitionNode> superblocks;  // raster order, row-major
  double cost = 0.0;
  int nodes = 0;
  int leaves = 0;
  std::string error;
};

struct LeafBlock {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  int depth = 0;
  bool has_chroma = false;
  bool chroma_offset = false;
};

// Decide one superblock.  `sb_mi_row` / `sb_mi_col` must be aligned to the
// superblock size and inside the frame.
PartitionDecision decide_superblock(const FrameGeometry& geometry,
                                    int sb_mi_row, int sb_mi_col,
                                    const CostFunction& costs,
                                    const DecisionOptions& options);

// Decide every superblock in raster order.  Deterministic for identical
// geometry, costs, and options.
FrameDecision decide_frame(const FrameGeometry& geometry,
                           const CostFunction& costs,
                           const DecisionOptions& options);

// Depth-first, decode-order leaf walk: SPLIT visits TL, TR, BL, BR; HORZ
// visits top then bottom; VERT visits left then right.
void collect_leaves(const PartitionNode& node, std::vector<LeafBlock>& leaves);

int count_nodes(const PartitionNode& node);
int count_leaves(const PartitionNode& node);

// Structural / spec validation of a tree against the geometry.  Checks:
//   * node origin and block size match the parent's partition
//   * the partition is spec-allowed and respects the implied / forced choice
//   * child count, order, positions, and sizes
//   * hasChroma / chromaOffset propagation
//   * stored boundary/allowed fields are not stale
//   * every in-frame cell in the superblock is covered exactly once
// Does not check costs or DecisionOptions policy.
bool validate_tree(const FrameGeometry& geometry, const PartitionNode& root,
                   std::string& error);

// Validates every superblock and the frame-level SB layout.
bool validate_frame(const FrameGeometry& geometry, const FrameDecision& frame,
                    std::string& error);

}  // namespace av2enc::partition
