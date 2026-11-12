// Partition syntax model for AV2 extended partitions.
//
// This is the bridge between the geometry solver in partition_ext.h and the M1
// entropy writer.  It implements:
//   * the exact symbol sequence of §5.20.3.2 read_partition(),
//     including every implied / no-syntax case;
//   * the §8.3.2 CDF-row derivations for do_split, do_square_split, rect_type,
//     do_ext_partition and do_uneven_4way_partition;
//   * the §8.3.2 `L(1)` literal for uneven_4way_partition_type;
//   * a reference encoder and an independent decoder over the normative §8.2
//     symbol coder in src/entropy/av2/.
//
// What the M1 writer must add is listed in README.md ("M1 writer contract").
// No file outside src/encoder/partition_ext/ is modified.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "codec/partition/ext/partition_ext.h"

namespace av2enc::av2 {
struct Encoder;
struct Decoder;
}  // namespace av2enc::av2

namespace av2enc::partition_ext {

// Per-tile mutable copy of the five §9.3 partition CDF arrays.  Plane 0 is
// luma (PlaneStart == 0, always true for SHARED_PART); plane 1 is only used by
// chroma-only partition trees, which M1 does not produce.
struct PartitionCdfs {
  std::uint16_t do_split[2][64][3];
  std::uint16_t do_square_split[2][8][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t do_ext_partition[2][64][3];
  std::uint16_t do_uneven_4way_partition[2][64][3];
};
void init_default_cdfs(PartitionCdfs& cdfs);

enum class SymbolKind : std::uint8_t {
  kDoSplit = 0,
  kDoSquareSplit = 1,
  kRectType = 2,
  kDoExtPartition = 3,
  kDoUneven4wayPartition = 4,
  kUneven4wayType = 5,  // L(1) literal, not a CDF
};
const char* symbol_name(SymbolKind kind);
// CDF alphabet size for the symbol, or 0 for the literal.
int symbol_alphabet(SymbolKind kind);
// Largest legal context value for the symbol's CDF (0 for the literal).
int max_symbol_context(SymbolKind kind);

struct PartitionSymbol {
  SymbolKind kind = SymbolKind::kDoSplit;
  int value = 0;
  int context = 0;
  bool literal = false;
};

// §8.3.2 contexts for one node.  `do_ext` is shared by do_ext_partition and
// do_uneven_4way_partition (the spec reuses the derivation).
struct SymbolContexts {
  int do_split = 0;
  int do_square_split = 0;
  int rect_type = 0;
  int do_ext = 0;
};

// Compute all four contexts.  `partition` selects the rect direction for
// do_ext (NONE/SPLIT never code do_ext, so their value is unused).
SymbolContexts node_contexts(const FrameGeometry& geometry,
                             const PartitionNode& node,
                             const MiSizeGrid& luma_sizes);

// §5.20.3.2 read_partition() symbol sequence for a known partition.
// `forced == true` means the spec codes nothing at this node.
struct SymbolPlan {
  bool ok = true;
  bool forced = false;
  std::string error;
  std::vector<PartitionSymbol> symbols;
};
SymbolPlan plan_partition_symbols(const Boundary& boundary,
                                  const PartitionSet& allowed,
                                  BlockSize size, PartitionType partition,
                                  const SymbolContexts& contexts);

// Per-symbol bit costs from the caller's rate model, in bits (or any common
// unit).  `node_signal_cost()` reduces these to the NodeCosts::*_signal fields
// so an RDO loop can stay exactly in sync with read_partition(): a symbol that
// the spec implies costs nothing and is never emitted.
struct SymbolCosts {
  double do_split[2] = {0.0, 0.0};
  double do_square_split[2] = {0.0, 0.0};
  double rect_type[2] = {0.0, 0.0};
  double do_ext_partition[2] = {0.0, 0.0};
  double do_uneven_4way_partition[2] = {0.0, 0.0};
  double uneven_4way_type[2] = {0.0, 0.0};
};

// Cost of the partition syntax that would be emitted for `partition` at a node
// with this boundary/allowed set.  Returns kInfiniteCost when the plan is not
// reachable (or a symbol is not in the model).
double node_signal_cost(const Boundary& boundary, const PartitionSet& allowed,
                        BlockSize size, PartitionType partition,
                        const SymbolCosts& costs);

struct SyntaxStats {
  int symbols = 0;
  int literal_bits = 0;
  int nodes = 0;
  int leaves = 0;
};

// Encode the whole partition tree of one superblock.  `grid` is the caller's
// MiSizeGrid and must already contain every leaf decoded before this
// superblock (raster order); the function fills it as it walks.  It recomputes
// boundary / allowed / chroma and fails on any stale node field.
bool encode_tree_symbols(const FrameGeometry& geometry, const Tools& tools,
                         const PartitionNode& root, PartitionCdfs& cdfs,
                         MiSizeGrid& grid, av2::Encoder& encoder,
                         bool disable_cdf_update, SyntaxStats* stats,
                         std::string& error,
                         std::vector<PartitionSymbol>* trace = nullptr);

// Independent decoder: reads the same symbol sequence and rebuilds the tree.
// The produced tree passes validate_tree() when the stream is well formed.
bool decode_tree_symbols(const FrameGeometry& geometry, const Tools& tools,
                         PartitionNode& root, PartitionCdfs& cdfs,
                         MiSizeGrid& grid, av2::Decoder& decoder,
                         SyntaxStats* stats, std::string& error);

// Convenience: a canonical string of a tree, used by tests.
std::string tree_signature(const PartitionNode& node);

}  // namespace av2enc::partition_ext
