// M1 <-> partition_ext adapter: extended partition streams for the M1 writer.
//
// This module is the *only* place that maps the AV2 extended-partition solver
// (src/encoder/partition_ext/) onto the M1 syntax writer's node shape.  It is
// additive: it does not modify src/encoder/m1/still_*.{h,cpp}, partition_ext, the
// CUDA tree, or the build.
//
// What it provides
// ----------------
//   * StillNode / BlockSyntaxV2: the 10-value spec PartitionType, a 16-bit
//     `allowed_mask`, the forced/SB-root flags, the preorder `subtree` count,
//     and the per-leaf `has_chroma` + ChromaBlock that HORZ_3@8x32 and
//     HORZ_4A@32x32 need.  This is the shape still_syntax::BlockSyntax must grow
//     into (see README.md, "M1 writer contract").
//   * TilePartitionCdfs: the five §9.3 partition CDF arrays, including the two
//     the M1 writer does not have yet (TileDoExtPartitionCdf,
//     TileDoUneven4wayPartitionCdf) with §8.3.2 context derivation that matches
//     the writer's left_mi()/above_mi() arrays exactly.
//   * plan_node(): calls partition_ext::plan_partition_symbols() with the
//     boundary/allowed set re-derived from the node, so do_ext_partition,
//     do_uneven_4way_partition and the raw L(1) uneven_4way_partition_type bit
//     are emitted with the spec's implication rules (not the writer's current
//     `allowed[HORZ] || allowed[VERT]` shortcut).
//   * emit_symbols(): the §8.2 emission for a plan, into a caller-owned
//     av2::Encoder and caller-owned per-tile CDF copies.
//   * build_frame_stream()/build_superblock_stream(): validated preorder node
//     streams (plus leaf chroma records) from a partition_ext decision.
//   * encode_frame_partitions()/decode_frame_partitions(): a reference encoder
//     and an independent decoder for a whole frame of partition symbols.  Used
//     by the round-trip tests and the minimal dav2d gate.
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0
// §5.20.3.1/2, §8.2, §8.3.2, §9.2, §9.3.  No AVM/SVT/clave/av2-turbo code.
#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "pipeline/still/still_syntax.h"
#include "codec/partition/tree/partition.h"
#include "codec/partition/ext/ext_syntax.h"
#include "codec/partition/ext/partition_ext.h"
#include "codec/entropy/av2/av2_entropy.h"

namespace av2enc::still::partition_stream {

namespace pe = partition_ext;
namespace base = partition;

using pe::BlockSize;
using pe::PartitionType;

inline constexpr int kPartitionCount = pe::kPartitionCount;
// Bit p is allowed[PartitionType p]; values 0..9, so 10 bits.
inline constexpr std::uint16_t kFullAllowedMask = (1u << kPartitionCount) - 1u;
// The M1 writer's current BlockSyntax::allowed_mask is uint8.  Values 8
// (VERT_4B) and 9 (SPLIT) do not fit; see mask_fits_writer_u8().
inline constexpr std::uint16_t kWriterMaskBits = 0x00FFu;

// ---------------------------------------------------------------------------
// CDFs (§9.3 defaults, one mutable copy per tile)
// ---------------------------------------------------------------------------

// Exactly the five partition arrays a SHARED_PART tile needs.  Plane 0 is the
// only plane M1 uses (PlaneStart == 0); the [2] dimension is kept so the
// layout and the §9.3 copy stay byte-identical to partition_ext::PartitionCdfs
// and the M0/M1 code that flattens `[plane][context][3]`.
//
// M1 naming:
//   ext.do_split                 -> TileDoSplitCdf
//   ext.do_square_split          -> TileDoSquareSplitCdf
//   ext.rect_type                -> TileRectTypeCdf
//   ext.do_ext_partition         -> TileDoExtPartitionCdf          (new)
//   ext.do_uneven_4way_partition -> TileDoUneven4wayPartitionCdf   (new)
struct TilePartitionCdfs {
  pe::PartitionCdfs ext{};
};

// Copies the §9.3 defaults from ext_tables_generated.h.  Call once per tile.
void init_tile_partition_cdfs(TilePartitionCdfs& cdfs);

// CDF row for a symbol.  `context` must be in [0, max_symbol_context(kind)];
// literals (kUneven4wayType) have no CDF and return nullptr with `error` set.
std::uint16_t* cdf_row(TilePartitionCdfs& cdfs, pe::SymbolKind kind,
                       int context, std::string& error);

// ---------------------------------------------------------------------------
// allowed_mask
// ---------------------------------------------------------------------------

// 10-bit mask from a partition_ext allowed set.
std::uint16_t to_allowed_mask(const pe::PartitionSet& allowed);
bool mask_allows(std::uint16_t mask, PartitionType partition);
int mask_count(std::uint16_t mask);

// The base-rect view the *current* M1 writer understands:
//   bit 0 (kNone)  = allowed[NONE]
//   bit 1 (kSplit) = allowed[SPLIT]
//   bit 2 (kHorz)  = HORZ || HORZ_3 || HORZ_4A || HORZ_4B
//   bit 3 (kVert)  = VERT || VERT_3 || VERT_4A || VERT_4B
// This reproduces the spec's `allowHorz` / `allowVert` test that decides
// whether rect_type is coded, so an unmodified writer's base symbols stay
// correct while the adapter emits the extended symbols.  It is NOT the full
// allowed set: use StillNode::allowed_mask for that.
std::uint8_t to_writer_mask8(std::uint16_t full_mask);

// True when every set bit fits the current uint8 mask (i.e. no VERT_4B/SPLIT).
bool mask_fits_writer_u8(std::uint16_t full_mask);

// ---------------------------------------------------------------------------
// Contexts (§8.3.2)
// ---------------------------------------------------------------------------

// The four §8.3.2 contexts for one node.  For NONE/SPLIT, `do_ext` is unused.
struct NodeContexts {
  int do_split = 0;
  int do_square_split = 0;
  int rect_type = 0;
  int do_ext = 0;
};

// Neighbour probes, expressed exactly like the M1 writer's
// left_mi(row) / above_mi(col) arrays: Mi_Height_Log2 / Mi_Width_Log2 of the
// block that covers the probe cell, with 6 (BLOCK_256X256) for unset cells.
// `*_mid` are the §8.3.2 second probes at r + h4/2 and c + w4/2.
struct NeighbourContext {
  int left_log2 = 6;
  int left_mid_log2 = 6;
  int above_log2 = 6;
  int above_mid_log2 = 6;
  bool left_available = false;
  bool above_available = false;
};

// Derive the four contexts from the writer's neighbour arrays.  `partition`
// selects the do_ext rect direction (NONE/SPLIT never code do_ext).
NodeContexts contexts_from_neighbours(BlockSize size, PartitionType partition,
                                      const NeighbourContext& neighbours);

// Same probes, read from a partition_ext::MiSizeGrid (CPU decision path and
// tests).  Equivalent to partition_ext::node_contexts() for every node that
// validate_tree() accepts; tests pin the equality.
NeighbourContext neighbours_from_grid(const base::FrameGeometry& geometry,
                                      int mi_row, int mi_col, BlockSize size,
                                      const base::MiSizeGrid& grid);

// M1-writer convenience: read the four probes straight out of the writer's
// left_mi / above_mi arrays (Mi_*_Log2 values, 6 sentinel, +16 padding).  Out
// of range entries behave like the sentinel.  This is the exact bridge for
// still_syntax::Writer::left_mi()/above_mi().
NeighbourContext neighbours_from_arrays(std::span<const int> left_mi,
                                        std::span<const int> above_mi,
                                        int mi_row, int mi_col, BlockSize size,
                                        bool left_available,
                                        bool above_available);

// The §8.3.2 do_ext_partition / do_uneven_4way_partition context alone.
int ext_partition_context(BlockSize size, bool horizontal,
                          const NeighbourContext& neighbours);

// ---------------------------------------------------------------------------
// M1 node shape
// ---------------------------------------------------------------------------

// One preorder entry in the M1 partition stream.  This is BlockSyntax's tree
// part with two additions the extended rows require:
//   * `allowed_mask` is 16 bits (10 spec partition values);
//   * `partition` is the 10-value spec enum, not partition::PartitionType.
// Leaves also carry the chroma arguments and the ChromaMi* block in effect, so
// the chroma/CfL coder can code the shared 8x16/16x8 middle block of HORZ_3 and
// skip chroma for the first HORZ_4A/4B children.
//
// flags use syntax::kFlagForced / syntax::kFlagSbRoot, so the M1 writer's
// existing flag handling carries over unchanged.
struct StillNode {
  std::uint16_t mi_row = 0;
  std::uint16_t mi_col = 0;
  std::uint8_t bsize = 0;          // base::BlockSize
  std::uint8_t partition = 0;      // pe::PartitionType (spec value 0..9)
  std::uint8_t flags = 0;          // syntax::kFlagForced | kFlagSbRoot
  std::uint16_t subtree = 0;       // preorder entries below this node
  std::uint16_t allowed_mask = 0;  // bit p = allowed[PartitionType p]
  bool has_chroma = false;
  bool chroma_offset = false;
  pe::ChromaBlock chroma{};
};

// A ChromaBlock converted to the coordinate system the M1 chroma/CfL grid
// uses: 4x4 units of the *chroma plane* (luma MiRow >> subsampling).  The
// block size is the chroma residual size (`chroma_plane_residual_size`), not
// the luma size.  For the HORZ_3@8x32 middle pair this is (1,0) 4x8; for the
// HORZ_4A@32x32 last child it is (0,0) 16x16.
struct ChromaPlaneBlock {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize residual_size = BlockSize::kInvalid;
};
ChromaPlaneBlock chroma_plane_block(const base::FrameGeometry& geometry,
                                    const pe::ChromaBlock& block);

// Leaf payload the current BlockSyntax has no field for.  `node_index` is the
// index of the leaf in the preorder node vector.  Emitted in decode order.
struct LeafChroma {
  std::uint32_t node_index = 0;
  bool has_chroma = false;
  bool chroma_offset = false;
  pe::ChromaBlock chroma{};
};

// A whole frame in preorder, superblocks in raster order.  `nodes` is exactly
// the order still_syntax::Writer::write_node() walks; `leaves` is exactly the
// order partition_ext::collect_leaves() returns.
struct FrameStream {
  std::vector<StillNode> nodes;
  std::vector<LeafChroma> leaves;
  int node_count = 0;
  int leaf_count = 0;
};

// Proposed replacement for syntax::BlockSyntax's tree part.  The M1 writer
// needs every field here; a drop-in widening is described in README.md.
struct BlockSyntaxV2 {
  std::uint16_t mi_row = 0;
  std::uint16_t mi_col = 0;
  std::uint8_t bsize = 0;          // base::BlockSize
  std::uint8_t partition = 0;      // pe::PartitionType
  std::uint16_t allowed_mask = 0;  // 10 bits
  std::uint8_t flags = 0;
  std::uint16_t subtree = 0;
  std::uint8_t has_chroma = 0;
  std::uint8_t chroma_bsize = 0;   // pe::BlockSize of ChromaMiSize
  std::uint16_t chroma_mi_row = 0;
  std::uint16_t chroma_mi_col = 0;
};

BlockSyntaxV2 to_block_syntax_v2(const StillNode& node);

// Build a validated preorder stream for one superblock (flags gains
// kFlagSbRoot) or for a whole frame.  Both call the partition_ext validators
// and fail rather than emit a stale tree.
bool build_superblock_stream(const base::FrameGeometry& geometry,
                             const pe::Tools& tools,
                             const pe::PartitionNode& root,
                             FrameStream& out, std::string& error);
bool build_frame_stream(const base::FrameGeometry& geometry,
                        const pe::Tools& tools,
                        const pe::FrameDecision& frame, FrameStream& out,
                        std::string& error);

// ---------------------------------------------------------------------------
// Symbol planning and emission
// ---------------------------------------------------------------------------

struct NodePlan {
  bool ok = true;
  bool forced = false;
  pe::Boundary boundary{};
  pe::PartitionSet allowed{};
  pe::SymbolPlan symbols;
  std::string error;
};

// Re-derive boundary/allowed for `node`, check the node against them, and run
// partition_ext::plan_partition_symbols().  Overload 2 computes the contexts
// from the M1 writer's neighbour arrays; overload 3 from a MiSizeGrid.
NodePlan plan_node(const base::FrameGeometry& geometry, const pe::Tools& tools,
                   const StillNode& node, const NodeContexts& contexts);
NodePlan plan_node(const base::FrameGeometry& geometry, const pe::Tools& tools,
                   const StillNode& node, const NeighbourContext& neighbours);
NodePlan plan_node(const base::FrameGeometry& geometry, const pe::Tools& tools,
                   const StillNode& node, const base::MiSizeGrid& grid);

// Emit a symbol plan into `encoder`.  The caller owns the CDF copies and must
// set encoder.disable_cdf_update before the first symbol.  Literals use
// encode_bool() and adapt no CDF (§5.20.3.2 uneven_4way_partition_type).
bool emit_symbols(const pe::SymbolPlan& plan, TilePartitionCdfs& cdfs,
                  av2::Encoder& encoder, pe::SyntaxStats* stats,
                  std::string& error);

// ---------------------------------------------------------------------------
// Whole-frame reference codec for partition symbols
// ---------------------------------------------------------------------------

// Encode one superblock's partition symbols into a standalone payload.  Used
// by the tests to compare byte-for-byte against
// partition_ext::encode_tree_symbols().  `root` must be superblock-aligned.
bool encode_superblock_partitions(const base::FrameGeometry& geometry,
                                  const pe::Tools& tools,
                                  const pe::PartitionNode& root,
                                  TilePartitionCdfs& cdfs,
                                  bool disable_cdf_update,
                                  std::vector<std::uint8_t>& out,
                                  pe::SyntaxStats* stats, std::string& error);

// Encode `frame`'s partition symbols for every superblock in raster order into
// one tile payload.  The grid is filled as the walk proceeds, exactly like the
// decoder; `frame` must already pass partition_ext::validate_frame().
bool encode_frame_partitions(const base::FrameGeometry& geometry,
                             const pe::Tools& tools,
                             const pe::FrameDecision& frame,
                             TilePartitionCdfs& cdfs, bool disable_cdf_update,
                             std::vector<std::uint8_t>& out,
                             pe::SyntaxStats* stats, std::string& error);

// Independent decoder: reads one frame of partition symbols and rebuilds the
// trees.  Per-SB roots are raster-ordered; `frame.superblocks` matches
// partition_ext::decide_frame()'s output order.
bool decode_frame_partitions(const base::FrameGeometry& geometry,
                             const pe::Tools& tools, TilePartitionCdfs& cdfs,
                             av2::Decoder& decoder, bool disable_cdf_update,
                             pe::FrameDecision& frame, pe::SyntaxStats* stats,
                             std::string& error);

// ---------------------------------------------------------------------------
// Cost model adapter
// ---------------------------------------------------------------------------

// Per-family signal costs for the extended rows, in the caller's rate unit.
// kInfiniteCost disables a family (the solver never selects it).
struct ExtSignalCosts {
  double three_way = pe::kInfiniteCost;     // HORZ_3 / VERT_3
  double uneven_4way = pe::kInfiniteCost;   // HORZ_4A/B, VERT_4A/B
};

// Lift an M1 rect-only cost callback to partition_ext::CostFunction.  The
// caller's six-way costs are reused verbatim through
// partition_ext::from_partition_costs(); the extended signals are added from
// `ext`.  This is the incremental migration path: tools off reproduces the
// rect-only solver's choices (kRectCompatOrder).
pe::CostFunction lift_rect_costs(
    const base::CostFunction& rect_costs,
    const ExtSignalCosts& ext = ExtSignalCosts{});

}  // namespace av2enc::still::partition_stream
