// AV2 v1.0.0 extended partitions (3-way and uneven 4-way), CPU reference.
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//   §5.20.3.1 decode_partition      (child geometry for all 10 partitions)
//   §5.20.3.2 read_partition        (symbol order, implied rules, allowed set)
//   §5.20.3.2 is_partition_allowed / is_ext_partition_allowed /
//             is_uneven_4way_partition_allowed / is_chroma_offset_for_partition
//   §6.19.3.1/2 partition semantics (PartitionType enum, Rect_Part_Table)
//   §8.3.2 partition contexts       (do_ext_partition / do_uneven_4way)
//   §9.2 Partition_Subsize, H_Partition_Midsize, Partition_Size_Adjust
//
// The rect-only tree lives in src/encoder/partition and is *not* modified here.
// This module reuses its FrameGeometry / BlockSize / MiSizeGrid / block tables
// and adds the six extended partitions on top:
//
//     PARTITION_HORZ_3   (1:2:1 horizontal, middle split vertically)
//     PARTITION_VERT_3   (1:2:1 vertical,   middle split horizontally)
//     PARTITION_HORZ_4A  (1/8, 1/4, 1/2, 1/8 from the top)
//     PARTITION_HORZ_4B  (1/8, 1/2, 1/4, 1/8 from the top)
//     PARTITION_VERT_4A  (1/8, 1/4, 1/2, 1/8 from the left)
//     PARTITION_VERT_4B  (1/8, 1/2, 1/4, 1/8 from the left)
//
// Scope: 8-bit 4:2:0 (also 4:4:4 / monochrome geometry), 64x64 and 128x128
// superblocks, one tile, TreeType = SHARED_PART (no SDP, no BRU, no chroma-only
// partition trees).  `enable_ext_partitions` / `enable_uneven_4way_partitions`
// are explicit inputs; nothing here emits a bitstream (see ext_syntax.h).
//
// Deliberate non-goals: entropy coding, pixel processing, RDO rate models.
// Costs are caller-supplied, exactly like partition::NodeCosts.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "codec/partition/tree/partition.h"
#include "codec/partition/ext/ext_tables_generated.h"

namespace av2enc::partition_ext {

using partition::BlockSize;
using partition::FrameGeometry;
using partition::MiSizeGrid;

inline constexpr int kPartitionCount = 10;
inline constexpr int kMinBlockPixels = partition::kMinBlockPixels;
inline constexpr double kInfiniteCost = partition::kInfiniteCost;

// §6.19.3 PartitionType enum.  The numeric value is the spec value and the
// CDF/context index; do not renumber.
enum class PartitionType : std::uint8_t {
  kNone = 0,
  kHorz = 1,
  kVert = 2,
  kHorz3 = 3,
  kVert3 = 4,
  kHorz4A = 5,
  kHorz4B = 6,
  kVert4A = 7,
  kVert4B = 8,
  kSplit = 9,
};

const char* partition_name(PartitionType partition);
constexpr int spec_partition_index(PartitionType partition) {
  return static_cast<int>(partition);
}

bool is_valid_partition(PartitionType partition);
bool is_extended(PartitionType partition);  // 3..8 (3-way or uneven 4-way)
bool is_three_way(PartitionType partition);
bool is_uneven_4way(PartitionType partition);
bool is_four_way(PartitionType partition);  // 3..8
bool is_horizontal(PartitionType partition);
bool is_vertical(PartitionType partition);
// Base rect of a partition, or kNone for NONE/SPLIT.  Used for rect_type.
PartitionType rect_direction(PartitionType partition);
// Mapping to/from the rect-only enum in src/encoder/partition.
partition::PartitionType to_base(PartitionType partition);
PartitionType from_base(partition::PartitionType partition);

// §5.4.3 sequence_partition_config() tools.  `enable_uneven_4way_partitions`
// is only read when enable_ext_partitions is 1; a caller that sets it without
// ext is a programming error (validate_tools() rejects it).
struct Tools {
  // Matches the sequence-header default: nothing extended unless requested.
  bool enable_ext_partitions = false;
  bool enable_uneven_4way_partitions = false;
};
bool validate_tools(const Tools& tools, std::string& error);

// §9.2 Partition_Subsize.  Returns kInvalid when the table entry is
// BLOCK_INVALID.
BlockSize partition_subsize(PartitionType partition, BlockSize size);
// §9.2 H_Partition_Midsize.
BlockSize h_partition_midsize(BlockSize size);
// §9.2 Partition_Size_Adjust / _Rect_Type.
int size_adjust(BlockSize size);
int size_adjust_rect_type(BlockSize size);

// §5.20.3.2 implied_partition().  Extended partitions are never implied: the
// implied partition is always NONE, HORZ, or VERT.  Thin-block / 4x4 NONE
// implication is included.
struct Boundary {
  bool has_rows = false;
  bool has_cols = false;
  bool implied = false;
  PartitionType implied_partition = PartitionType::kNone;
};
Boundary implied_partition(const FrameGeometry& geometry, int mi_row,
                           int mi_col, BlockSize size);

// §5.20.3.2 chroma_offset_for_partition(), including the HORZ_3 middle
// special case.
bool chroma_offset_for_subsize(const FrameGeometry& geometry,
                                  BlockSize subsize);
bool chroma_offset_for_partition(const FrameGeometry& geometry,
                                    PartitionType partition, BlockSize size);

struct PartitionSet {
  bool allowed[kPartitionCount] = {};
  int count = 0;
  bool is_allowed(PartitionType partition) const;
};

// §5.20.3.2 is_partition_allowed().  Base partitions (NONE/HORZ/VERT/SPLIT)
// delegate to partition::is_partition_allowed() so the rect-only behaviour is
// shared bit-for-bit.  Extended rows add the tool flags and the spec's
// chromaOffset / block_coded cases.
bool is_partition_allowed(const FrameGeometry& geometry, int mi_row, int mi_col,
                          BlockSize size, PartitionType partition,
                          bool has_chroma, bool chroma_offset,
                          const Tools& tools);

// §5.20.3.2 init_allowed_partitions().  Forces PARTITION_NONE when the spec
// allows nothing else.
PartitionSet init_allowed_partitions(const FrameGeometry& geometry, int mi_row,
                                     int mi_col, BlockSize size,
                                     bool has_chroma, bool chroma_offset,
                                     const Tools& tools);

// The spec's ChromaMiRow/ChromaMiCol/ChromaMiSize state.  A leaf that codes
// chroma uses `size` (the *chroma* block), not necessarily its luma size: the
// HORZ_3@8x32 and VERT_3@32x8 middle pair share one chroma block.
struct ChromaBlock {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
};
bool operator==(const ChromaBlock& a, const ChromaBlock& b);
bool operator!=(const ChromaBlock& a, const ChromaBlock& b);

// decode_partition()'s chroma bookkeeping:
//   if (!chromaOffset && hasChroma) {
//       chromaOffset = chroma_offset_for_partition(p, bSize);
//       ChromaMi*    = (r, c, bSize);
//   }
struct ChromaState {
  bool chroma_offset = false;
  ChromaBlock block;
};
ChromaState resolve_chroma(const FrameGeometry& geometry, int mi_row,
                           int mi_col, BlockSize size,
                           PartitionType partition, bool has_chroma,
                           bool chroma_offset, const ChromaBlock& inherited);

// One child call produced by decode_partition().  `has_chroma` /
// `chroma_offset` are the recursion arguments; `chroma` is the ChromaMi* value
// in effect when the child runs (the child may overwrite it if it transitions).
struct ChildSlot {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  bool has_chroma = false;
  bool chroma_offset = false;
  ChromaBlock chroma;
};
bool child_is_coded(const FrameGeometry& geometry, const ChildSlot& slot);

// §5.20.3.1 child enumeration in decode order.  Returns 0 for PARTITION_NONE
// or an undefined Partition_Subsize.  The caller filters with
// child_is_coded(); `slots` always receives the full spec list.
int enumerate_children(const FrameGeometry& geometry, int mi_row, int mi_col,
                       BlockSize size, PartitionType partition,
                       bool has_chroma, bool chroma_offset,
                       const ChromaBlock& inherited, ChildSlot slots[4]);

// ---------------------------------------------------------------------------
// Contexts (§8.3.2)
// ---------------------------------------------------------------------------

// do_split / rect_type / do_square_split, identical to the rect-only module.
partition::PartitionContexts partition_contexts(const FrameGeometry& geometry,
                                                int mi_row, int mi_col,
                                                BlockSize size,
                                                const MiSizeGrid& sizes);

// §8.3.2 do_ext_partition and do_uneven_4way_partition context.  Both share
// one derivation:
//   RECT_HORZ: bsh = max(Height_Log2 - 1, 1)
//              ctx1 = Height_Log2(LeftMiSizes[r])                 < bsh
//              ctx2 = Height_Log2(LeftMiSizes[r + h4/2])          < bsh
//   RECT_VERT: bsw = max(Width_Log2 - 1, 1)
//              ctx1 = Width_Log2(AboveMiSizes[c])                 < bsw
//              ctx2 = Width_Log2(AboveMiSizes[c + w4/2])          < bsw
//   ctx = Partition_Size_Adjust[bSize] * 4 + ctx1 * 2 + ctx2
// Unset / outside cells behave like the spec's BLOCK_256X256 sentinel.
int ext_partition_context(const FrameGeometry& geometry, int mi_row,
                          int mi_col, BlockSize size, bool horizontal,
                          const MiSizeGrid& luma_sizes);

// ---------------------------------------------------------------------------
// Decision API (same shape as partition::decide_frame)
// ---------------------------------------------------------------------------

struct NodeCosts {
  double leaf = kInfiniteCost;
  double none_signal = 0.0;
  double split_signal = 0.0;
  double horz_signal = 0.0;
  double vert_signal = 0.0;
  double horz3_signal = 0.0;
  double vert3_signal = 0.0;
  double horz4a_signal = 0.0;
  double horz4b_signal = 0.0;
  double vert4a_signal = 0.0;
  double vert4b_signal = 0.0;
};
NodeCosts from_partition_costs(const partition::NodeCosts& costs);

double partition_signal_cost(const NodeCosts& costs, PartitionType partition);

struct NodeInfo {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  int depth = 0;
  bool has_chroma = false;
  bool chroma_offset = false;
  ChromaBlock chroma;
  Boundary boundary;
  PartitionSet allowed;
};

using CostFunction = std::function<NodeCosts(const NodeInfo&)>;

// Tie-break order.  kSpecOrder walks the §6.19.3 enum (0..9); kRectCompatOrder
// walks NONE, SPLIT, HORZ, VERT first (the rect-only module's order) and then
// the extended partitions.  Costs being equal, the first candidate wins.
enum class TieBreakOrder : std::uint8_t {
  kSpecOrder = 0,
  kRectCompatOrder = 1,
};

struct DecisionOptions {
  int min_block_size_px = kMinBlockPixels;
  int max_depth = 32;
  bool allow_rect = true;     // HORZ / VERT
  bool allow_split = true;    // PARTITION_SPLIT
  bool allow_3way = true;     // HORZ_3 / VERT_3
  bool allow_uneven4 = true;  // HORZ_4A/B, VERT_4A/B
  TieBreakOrder tie_break = TieBreakOrder::kSpecOrder;
};

struct PartitionNode {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  PartitionType partition = PartitionType::kNone;
  int depth = 0;
  bool has_chroma = false;
  bool chroma_offset = false;
  ChromaBlock chroma;  // ChromaMi* in effect at this node
  Boundary boundary;
  PartitionSet allowed;
  bool forced = false;
  double cost = 0.0;
  std::vector<PartitionNode> children;  // in-frame children, decode order
};

enum class DecisionStatus : std::uint8_t {
  kOk = 0,
  kInvalidGeometry,
  kInvalidTools,
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
  std::vector<PartitionNode> superblocks;
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
  ChromaBlock chroma;
};

PartitionDecision decide_superblock(const FrameGeometry& geometry,
                                    int sb_mi_row, int sb_mi_col,
                                    const CostFunction& costs,
                                    const Tools& tools,
                                    const DecisionOptions& options);

FrameDecision decide_frame(const FrameGeometry& geometry,
                           const CostFunction& costs, const Tools& tools,
                           const DecisionOptions& options);

// Depth-first decode-order leaf walk.
void collect_leaves(const PartitionNode& node, std::vector<LeafBlock>& leaves);
int count_nodes(const PartitionNode& node);
int count_leaves(const PartitionNode& node);

// Structural / spec validation, including chroma-block propagation and exact
// in-frame coverage.  Does not check costs or DecisionOptions policy.
bool validate_tree(const FrameGeometry& geometry, const Tools& tools,
                   const PartitionNode& root, std::string& error);
bool validate_frame(const FrameGeometry& geometry, const Tools& tools,
                    const FrameDecision& frame, std::string& error);

// Geometry helper used by the tests and the syntax writer.
bool validate_geometry(const FrameGeometry& geometry, std::string& error);

}  // namespace av2enc::partition_ext
