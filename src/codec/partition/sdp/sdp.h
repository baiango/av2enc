// AV2 semi-decoupled partitioning (SDP), CPU reference.
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//   §5.4.3  sequence_partition_config()  (enable_sdp, enable_extended_sdp)
//   §5.20.3.1 decode_partition()         (TreeType transitions, usingSdp)
//   §5.20.3.2 read_partition() + partition_implied / sdp_partition_allowed /
//             sdp_ext_allowed / sdp_uneven4_allowed /
//             sdp_chroma_offset_for_partition / sdp_check_chroma
//   §6.4.3  sequence partition config semantics
//   §6.19.3 partition semantics          (TreeType, ChromaFollowsLuma,
//             ChromaPartitionKnown, ChromaMiRow/Col/Size, LumaPartitions)
//   §8.3.2  partition contexts           (PlaneStart selects the CDF plane)
//   §9.2    Partition_Subsize / H_Partition_Midsize / Partition_Size_Adjust
//   §9.3    Default_Region_Type_Cdf
//
// This module owns ONLY the reference model: geometry, the luma/chroma tree
// split, allowed sets, chroma bookkeeping, contexts and a symbol-order plan.
// It does not encode pixels or write a bitstream.  It reuses:
//   * encoder/partition      BlockSize, FrameGeometry, geometry tables
//   * encoder/partition_ext  the 10-value PartitionType, extended subsize
//                            tables, child/chroma enumeration
// No file outside src/encoder/sdp/ is modified.
//
// Deliberate non-goals: inter prediction, BRU, 256x256 superblocks, 4:2:2,
// tiles (availability is frame-relative), and pixel reconstruction.  The
// `region_type` half of extended SDP IS modelled structurally (it is only
// reachable for inter frames), but no inter encoder drives it yet.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "codec/partition/tree/partition.h"
#include "codec/partition/ext/partition_ext.h"

namespace av2enc::sdp {

using partition::BlockSize;
using partition::FrameGeometry;
using partition::MiSizeGrid;
using partition_ext::ChromaBlock;
using partition_ext::PartitionSet;
using partition_ext::PartitionType;

inline constexpr int kPartitionCount = partition_ext::kPartitionCount;
inline constexpr int kNumPlanes = 3;
inline constexpr int kMinBlockPixels = partition::kMinBlockPixels;
inline constexpr double kInfiniteCost = partition::kInfiniteCost;
// INTER_SDP_MAX_BLOCK_SIZE (§3).
inline constexpr int kInterSdpMaxBlockSize = 64;

// §6.19.3.1 TreeType.  Numeric values are the spec's; they are NOT a CDF
// index.  The CDF plane is PlaneStart (0 for SHARED/LUMA, 1 for CHROMA).
enum class TreeType : std::uint8_t {
  kSharedPart = 0,
  kLumaPart = 1,
  kChromaPart = 2,
};
const char* tree_type_name(TreeType type);
bool is_valid_tree_type(TreeType type);
// §8.3.2: do_split / do_square_split / rect_type / do_ext_partition select
// Tile*Cdf[PlaneStart].  CHROMA_PART codes from plane 1.
int plane_start(TreeType type);

// §6.19.3.1 RegionType.  INTRA_REGION=0, MIXED_REGION=1 (constants §3).
enum class RegionType : std::uint8_t {
  kIntraRegion = 0,
  kMixedRegion = 1,
};
const char* region_type_name(RegionType type);

// §5.4.3 sequence_partition_config().  enable_extended_sdp is only signalled
// when enable_sdp==1 and single_picture_header_flag==0; otherwise it is
// inferred 0.  enable_uneven_4way_partitions is only read when
// enable_ext_partitions==1.
struct Tools {
  bool enable_sdp = false;
  bool enable_extended_sdp = false;
  bool enable_ext_partitions = false;
  bool enable_uneven_4way_partitions = false;
  bool single_picture_header_flag = true;
  bool frame_is_intra = true;
  // §5.18.4.2: bru_mode = IsBridge ? INACTIVE : use_bru ? INACTIVE : ACTIVE.
  // In a normal frame this is ACTIVE, which is what enables region_type and the
  // syntax path in read_partition().  A per-SB BruTileActive symbol must be
  // resolved by the caller into `use_bru`/`is_bridge` before building a tree.
  bool use_bru = false;
  bool is_bridge = false;
};
bool sdp_validate_geometry(const FrameGeometry& geometry, std::string& error);
bool validate_tools(const Tools& tools, std::string& error);
// The per-superblock bru_mode: false means read_partition() returns
// PARTITION_NONE after the implied/numAllowed checks and codes no symbols.
bool bru_mode_active(const Tools& tools);
// §5.20.3.1 is_bsize_allowed_for_extended_sdp().
bool is_bsize_allowed_for_extended_sdp(BlockSize size,
                                       PartitionType partition);

struct Boundary {
  bool has_rows = false;
  bool has_cols = false;
  bool implied = false;
  PartitionType implied_partition = PartitionType::kNone;
};

// §5.20.3.2 partition_implied_at_boundary() (TreeType-independent).
Boundary sdp_implied_at_boundary(const FrameGeometry& geometry, int mi_row,
                                 int mi_col, BlockSize size);

// §5.20.3.2 partition_implied().  Adds the thin-block/4x4 NONE cases, the
// CHROMA_PART 8x8 NONE case and the CHROMA_PART 64x64 ChromaPartitionKnown
// case.  `luma_partition` is LumaPartitions[r][c] for the latter.
Boundary sdp_implied(const FrameGeometry& geometry, int mi_row, int mi_col,
                     BlockSize size, TreeType tree_type,
                     bool chroma_partition_known = false,
                     PartitionType luma_partition = PartitionType::kNone);

// §5.20.3.2 sdp_rect_implied(), including the CHROMA_PART-only
// 8x16/8x32 (HORZ) and 16x8/32x8 (VERT) cases.
partition::RectTypeImplied sdp_rect_implied(BlockSize size,
                                            TreeType tree_type);

// §9.2 tables (delegates).
BlockSize sdp_subsize(PartitionType partition, BlockSize size);
BlockSize sdp_midsize(BlockSize size);
int sdp_size_adjust(BlockSize size);
int sdp_size_adjust_rect(BlockSize size);

// §5.20.3.2 sdp_chroma_offset_for_subsize / _partition.
bool sdp_chroma_offset_for_subsize(const FrameGeometry& geometry,
                                   BlockSize subsize);
bool sdp_chroma_offset_for_partition(const FrameGeometry& geometry,
                                     PartitionType partition, BlockSize size);

// §5.20.3.2 sdp_ext_allowed() / sdp_uneven4_allowed().
// `horizontal` is the rectType (RECT_HORZ when true).
bool sdp_ext_allowed(const FrameGeometry& geometry, BlockSize size,
                     bool horizontal, TreeType tree_type, const Tools& tools);
bool sdp_uneven4_allowed(const FrameGeometry& geometry, BlockSize size,
                         bool horizontal, TreeType tree_type,
                         const Tools& tools);

// §5.20.3.2 sdp_check_chroma(): true only for a LUMA_PART block of at least 64x64
// whose plane-residual size is representable.
bool sdp_check_chroma(const FrameGeometry& geometry, BlockSize size,
                      TreeType tree_type);

// §5.20.3.2 sdp_partition_allowed().  `chroma_offset` is the recursion
// argument.  The function reproduces the spec's TreeType gating exactly:
//   * the local chromaOffset update is skipped for CHROMA_PART;
//   * the chromaOffset/block_coded checks are skipped for LUMA_PART;
//   * the PARTITION_NONE half-block test has the CHROMA_PART@8x8 exception;
//   * the !FrameIsIntra && RegionType==MIXED_REGION && subSize==4x4 reject.
bool sdp_partition_allowed(const FrameGeometry& geometry, int mi_row,
                           int mi_col, BlockSize size, PartitionType partition,
                           TreeType tree_type, bool has_chroma,
                           bool chroma_offset, RegionType region_type,
                           const Tools& tools);

// §5.20.3.2 sdp_init_allowed().  Forces PARTITION_NONE when nothing is
// allowed.
PartitionSet sdp_init_allowed(const FrameGeometry& geometry, int mi_row,
                              int mi_col, BlockSize size, TreeType tree_type,
                              bool has_chroma, bool chroma_offset,
                              RegionType region_type, const Tools& tools);

struct ChromaState {
  bool chroma_offset = false;
  ChromaBlock block;
};

// §5.20.3.1 ChromaMi* update.  The update happens for every TreeType whenever
// !chromaOffset && hasChroma.  `partition` is the partition just read.
ChromaState sdp_resolve_chroma(const FrameGeometry& geometry, int mi_row,
                               int mi_col, BlockSize size,
                               PartitionType partition, bool has_chroma,
                               bool chroma_offset,
                               const ChromaBlock& inherited);

// ---------------------------------------------------------------------------
// Luma/chroma decoupling data
// ---------------------------------------------------------------------------

// §5.20.3.1 ChromaFollowsLuma for one LUMA_PART 64x64 root.  The spec sets an
// initial value from the root partition and lets each *direct child* clear it.
// Reproduced verbatim; do not "fix" the asymmetry.
bool chroma_follows_luma(PartitionType root_partition,
                         const std::vector<PartitionType>& child_partitions);

// §5.20.3.1 CflAllowedInSdp at a CHROMA_PART 64x64 root.
bool cfl_allowed_in_sdp(PartitionType luma_partition, bool chroma_follows,
                        PartitionType chroma_partition);

bool sdp_is_horizontal(PartitionType partition);
bool sdp_is_vertical(PartitionType partition);

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

// One node of the SDP-aware partition tree.
//
// For an intra SDP split (enable_sdp, SHARED_PART, 64x64, FrameIsIntra):
//   sdp_split == true, partition == kNone, children == {luma root, chroma
//   root}, both of size 64x64.
//
// For extended SDP (region_type == INTRA_REGION on an inter frame):
//   extended_sdp == true, region_type == kIntraRegion, `children` are the
//   luma partition children (tree_type == LUMA_PART) followed by exactly one
//   synthetic chroma block (tree_type == CHROMA_PART, sdp_chroma_block == true,
//   partition == kNone, forced == true).
//
// `codes_chroma` is the leaf's HasChroma (only meaningful on a NONE leaf).
struct Node {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  int depth = 0;
  TreeType tree_type = TreeType::kSharedPart;
  PartitionType partition = PartitionType::kNone;

  bool sdp_split = false;
  bool extended_sdp = false;
  bool sdp_chroma_block = false;
  bool region_type_coded = false;
  RegionType region_type = RegionType::kMixedRegion;

  bool has_chroma = false;
  bool chroma_offset = false;
  // ChromaMi* in effect *after* this node's partition update, i.e. the block
  // children inherit and the block a codes_chroma leaf owns.  For an SDP split
  // node no update happens; it stays the inherited value.
  ChromaBlock chroma;
  bool extended_sdp_allowed = false;

  // Luma root bookkeeping (valid when tree_type == LUMA_PART).
  bool chroma_follows_luma = false;
  // Chroma root bookkeeping (valid when tree_type == CHROMA_PART and size is
  // 64x64): ChromaPartitionKnown[r][c] and LumaPartitions[r][c].
  bool chroma_partition_known = false;
  PartitionType implied_luma_partition = PartitionType::kNone;
  bool cfl_allowed_in_sdp = false;

  Boundary boundary;
  PartitionSet allowed;
  bool forced = false;
  bool codes_chroma = false;
  double cost = 0.0;
  std::vector<Node> children;
};

struct LeafBlock {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  int depth = 0;
  TreeType tree_type = TreeType::kSharedPart;
  bool has_chroma = false;
  bool chroma_offset = false;
  ChromaBlock chroma;
  // True when this leaf codes one chroma block (HasChroma == 1).  Its chroma
  // extent is `chroma` (ChromaMiSize, in luma 4x4 units).
  bool codes_chroma = false;
  bool sdp_chroma_block = false;
};

// ---------------------------------------------------------------------------
// Decision API
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
  // region_type symbol cost: [0] = INTRA_REGION, [1] = MIXED_REGION.
  double region_type[2] = {kInfiniteCost, kInfiniteCost};
};
double partition_signal_cost(const NodeCosts& costs, PartitionType partition);

struct NodeInfo {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  int depth = 0;
  TreeType tree_type = TreeType::kSharedPart;
  bool sdp_split = false;
  bool extended_sdp = false;
  bool sdp_chroma_block = false;
  bool region_type_coded = false;
  RegionType region_type = RegionType::kMixedRegion;
  bool has_chroma = false;
  bool chroma_offset = false;
  ChromaBlock chroma;
  bool extended_sdp_allowed = false;
  bool chroma_partition_known = false;
  PartitionType implied_luma_partition = PartitionType::kNone;
  Boundary boundary;
  PartitionSet allowed;
};

using CostFunction = std::function<NodeCosts(const NodeInfo&)>;

struct DecisionOptions {
  int min_block_size_px = kMinBlockPixels;
  int max_depth = 32;
  bool allow_rect = true;
  bool allow_split = true;
  bool allow_3way = true;
  bool allow_uneven4 = true;
  // Policy switch for the mandatory intra SDP split.  When false, a 64x64
  // SHARED node is solved as a plain shared node even if tools.enable_sdp is
  // set (used for parity tests and staged integration).
  bool allow_sdp = true;
  // Policy switch for region_type (inter frames only).
  bool allow_extended_sdp = true;
};

enum class DecisionStatus : std::uint8_t {
  kOk = 0,
  kInvalidGeometry,
  kInvalidTools,
  kInvalidCost,
  kNoValidPartition,
  kInvalidTree,
};
const char* decision_status_name(DecisionStatus status);

struct PartitionDecision {
  DecisionStatus status = DecisionStatus::kInvalidGeometry;
  Node root;
  double cost = 0.0;
  int nodes = 0;
  int leaves = 0;
  std::string error;
};

struct FrameDecision {
  DecisionStatus status = DecisionStatus::kInvalidGeometry;
  std::vector<Node> superblocks;
  double cost = 0.0;
  int nodes = 0;
  int leaves = 0;
  std::string error;
};

PartitionDecision decide_superblock(const FrameGeometry& geometry,
                                    const Tools& tools,
                                    int sb_mi_row, int sb_mi_col,
                                    const CostFunction& costs,
                                    const DecisionOptions& options);
FrameDecision decide_frame(const FrameGeometry& geometry, const Tools& tools,
                           const CostFunction& costs,
                           const DecisionOptions& options);

void collect_leaves(const Node& node, std::vector<LeafBlock>& leaves);
int count_nodes(const Node& node);
int count_leaves(const Node& node);

// Structural validation; re-derives every spec rule and checks exact in-frame
// coverage of both planes.  Costs and DecisionOptions are not checked.
bool validate_tree(const FrameGeometry& geometry, const Tools& tools,
                   const Node& root, std::string& error);
bool validate_frame(const FrameGeometry& geometry, const Tools& tools,
                    const FrameDecision& frame, std::string& error);

// Deterministic canonical string (ignores cost).  Used by tests and the
// syntax trace.
std::string tree_signature(const Node& node);

// ---------------------------------------------------------------------------
// Contexts and syntax plan
// ---------------------------------------------------------------------------

struct SymbolContexts {
  int do_split = 0;
  int do_square_split = 0;
  int rect_type = 0;
  int do_ext = 0;
  int region_type = 0;
};

// §8.3.2 contexts for one node.  `plane_sizes` must contain the already
// decoded neighbours for the node's plane (luma for PlaneStart 0, chroma for
// PlaneStart 1); unset cells behave like BLOCK_256X256.
SymbolContexts node_contexts(const FrameGeometry& geometry, const Node& node,
                             const MiSizeGrid& plane_sizes);

// §8.3.2 region_type context (network of numSamples groups).
int region_type_context(BlockSize size);

enum class SymbolKind : std::uint8_t {
  kDoSplit = 0,
  kDoSquareSplit = 1,
  kRectType = 2,
  kDoExtPartition = 3,
  kDoUneven4wayPartition = 4,
  kUneven4wayType = 5,  // L(1) literal
  kRegionType = 6,
};
const char* symbol_name(SymbolKind kind);
int symbol_alphabet(SymbolKind kind);   // 0 for the literal
int max_symbol_context(SymbolKind kind);

struct Symbol {
  SymbolKind kind = SymbolKind::kDoSplit;
  int value = 0;
  int context = 0;
  int plane = 0;  // PlaneStart / Tile*Cdf plane index
  bool literal = false;
};

// TreeType transition positions.  A switch carries no bits; it records where
// the decoder changes PlaneStart between the luma and chroma trees.
enum class SwitchKind : std::uint8_t {
  kSdpIntraLumaStart = 0,   // SHARED -> LUMA at a 64x64 intra node
  kSdpIntraChromaStart = 1,  // LUMA -> CHROMA
  kSdpIntraEnd = 2,          // CHROMA -> SHARED
  kSdpExtendedLumaStart = 3,  // SHARED -> LUMA after region_type==INTRA
  kSdpExtendedChromaStart = 4,
  kSdpExtendedEnd = 5,
};
const char* switch_kind_name(SwitchKind kind);

struct TreeSwitch {
  SwitchKind kind = SwitchKind::kSdpIntraLumaStart;
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
};

// §5.20.3.2 read_partition() symbol sequence for a known partition.  Emits
// nothing for the implied / single-candidate / BRU-inactive cases and validates
// that `partition` is the one the symbols encode.  `plane` is used for all
// partition symbols; `contexts` must be derived with the same plane.
struct SymbolPlan {
  bool ok = true;
  bool forced = false;
  std::string error;
  std::vector<Symbol> symbols;
};
SymbolPlan plan_partition_symbols(const Boundary& boundary,
                                  const PartitionSet& allowed, BlockSize size,
                                  TreeType tree_type, PartitionType partition,
                                  const SymbolContexts& contexts,
                                  bool bru_active);

// Full superblock plan: partition symbols, region_type, and the luma/chroma
// tree-switch positions, in decode order.
struct SuperblockPlan {
  bool ok = true;
  std::string error;
  std::vector<Symbol> symbols;
  std::vector<TreeSwitch> switches;
};
SuperblockPlan plan_superblock_symbols(const FrameGeometry& geometry,
                                       const Tools& tools, const Node& root,
                                       MiSizeGrid& luma_sizes,
                                       MiSizeGrid& chroma_sizes);

// Independent decoder over a symbol plan.  It re-derives implied/allowed sets
// and ChromaFollowsLuma from the symbols and rebuilds the tree.  Returns false
// on a malformed plan; the result passes validate_tree() on success.  The
// caller-supplied grids carry cross-superblock context and are filled in
// decode order.
bool decode_superblock_symbols(const FrameGeometry& geometry,
                               const Tools& tools, int sb_mi_row,
                               int sb_mi_col,
                               const std::vector<Symbol>& symbols,
                               MiSizeGrid& luma_sizes,
                               MiSizeGrid& chroma_sizes, Node& root,
                               std::vector<TreeSwitch>* switches,
                               std::string& error);

// All tree-type switches of a decided frame, in decode order.
std::vector<TreeSwitch> tree_type_events(const FrameDecision& frame);

// Plane coverage grids, for the integration pass.  `luma` receives every luma
// leaf; `chroma` receives ChromaMiSize at ChromaMiRow/Col for every leaf that
// codes chroma.
void build_plane_grids(const FrameGeometry& geometry, const Node& root,
                       MiSizeGrid& luma, MiSizeGrid& chroma);

}  // namespace av2enc::sdp
