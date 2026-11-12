// §5.20.3.2 partition symbol model and reference codec for extended partitions.

#include "ext_syntax.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <sstream>

#include "codec/entropy/av2/av2_entropy.h"

namespace av2enc::partition_ext {
namespace {

bool fail(std::string& error, std::string message) {
  error = std::move(message);
  return false;
}

// Rect_Part_Table[do_ext][do_uneven][type][rect] (§5.20.3.2).
PartitionType rect_part_table(bool do_ext, bool do_uneven, int type,
                              bool horizontal) {
  if (!do_ext) return horizontal ? PartitionType::kHorz : PartitionType::kVert;
  if (!do_uneven) {
    return horizontal ? PartitionType::kHorz3 : PartitionType::kVert3;
  }
  if (horizontal) {
    return type == 0 ? PartitionType::kHorz4A : PartitionType::kHorz4B;
  }
  return type == 0 ? PartitionType::kVert4A : PartitionType::kVert4B;
}

PartitionType implied_rect_type(BlockSize size) {
  const partition::RectTypeImplied implied =
      partition::rect_type_implied_by_bsize(size);
  if (!implied.implied) return PartitionType::kNone;
  return from_base(implied.partition);
}

PartitionType three_way_for(bool horizontal) {
  return horizontal ? PartitionType::kHorz3 : PartitionType::kVert3;
}

PartitionType four_way_a_for(bool horizontal) {
  return horizontal ? PartitionType::kHorz4A : PartitionType::kVert4A;
}

bool horizontal_family_allowed(const PartitionSet& allowed) {
  return allowed.is_allowed(PartitionType::kHorz) ||
         allowed.is_allowed(PartitionType::kHorz3) ||
         allowed.is_allowed(PartitionType::kHorz4A) ||
         allowed.is_allowed(PartitionType::kHorz4B);
}

bool vertical_family_allowed(const PartitionSet& allowed) {
  return allowed.is_allowed(PartitionType::kVert) ||
         allowed.is_allowed(PartitionType::kVert3) ||
         allowed.is_allowed(PartitionType::kVert4A) ||
         allowed.is_allowed(PartitionType::kVert4B);
}

struct ExtFlags {
  bool non_ext = false;
  bool ext3 = false;
  bool ext4 = false;
};

ExtFlags ext_flags(const PartitionSet& allowed, bool horizontal) {
  ExtFlags flags;
  if (horizontal) {
    flags.non_ext = allowed.is_allowed(PartitionType::kHorz);
    flags.ext3 = allowed.is_allowed(PartitionType::kHorz3);
    flags.ext4 = allowed.is_allowed(PartitionType::kHorz4A) ||
                 allowed.is_allowed(PartitionType::kHorz4B);
  } else {
    flags.non_ext = allowed.is_allowed(PartitionType::kVert);
    flags.ext3 = allowed.is_allowed(PartitionType::kVert3);
    flags.ext4 = allowed.is_allowed(PartitionType::kVert4A) ||
                 allowed.is_allowed(PartitionType::kVert4B);
  }
  return flags;
}

std::uint16_t* cdf_row(PartitionCdfs& cdfs, SymbolKind kind, int context,
                       std::string& error) {
  if (context < 0 || context > max_symbol_context(kind)) {
    fail(error, "symbol context out of range");
    return nullptr;
  }
  switch (kind) {
    case SymbolKind::kDoSplit:
      return &cdfs.do_split[0][context][0];
    case SymbolKind::kDoSquareSplit:
      return &cdfs.do_square_split[0][context][0];
    case SymbolKind::kRectType:
      return &cdfs.rect_type[0][context][0];
    case SymbolKind::kDoExtPartition:
      return &cdfs.do_ext_partition[0][context][0];
    case SymbolKind::kDoUneven4wayPartition:
      return &cdfs.do_uneven_4way_partition[0][context][0];
    case SymbolKind::kUneven4wayType:
      break;
  }
  fail(error, "literal symbol has no CDF");
  return nullptr;
}

struct EncodeState {
  const FrameGeometry& geometry;
  const Tools& tools;
  PartitionCdfs& cdfs;
  MiSizeGrid& grid;
  av2::Encoder& encoder;
  SyntaxStats stats;
  std::string error;
  std::vector<PartitionSymbol>* trace = nullptr;
};

struct DecodeState {
  const FrameGeometry& geometry;
  const Tools& tools;
  PartitionCdfs& cdfs;
  MiSizeGrid& grid;
  av2::Decoder& decoder;
  SyntaxStats stats;
  std::string error;
};

bool emit_symbol(EncodeState& state, const PartitionSymbol& symbol) {
  if (symbol.literal) {
    const av2::Status status = state.encoder.encode_bool(symbol.value != 0);
    if (status != av2::Status::kOk) {
      return fail(state.error,
                  std::string("encode_bool failed: ") +
                      av2::status_name(status));
    }
    ++state.stats.literal_bits;
    return true;
  }
  std::uint16_t* cdf =
      cdf_row(state.cdfs, symbol.kind, symbol.context, state.error);
  if (cdf == nullptr) return false;
  const av2::Status status = state.encoder.encode_symbol(cdf, 2, symbol.value);
  if (status != av2::Status::kOk) {
    return fail(state.error,
                std::string("encode_symbol failed: ") +
                    av2::status_name(status));
  }
  ++state.stats.symbols;
  return true;
}

bool read_cdf_symbol(DecodeState& state, SymbolKind kind, int context,
                     int& value) {
  std::uint16_t* cdf = cdf_row(state.cdfs, kind, context, state.error);
  if (cdf == nullptr) return false;
  const av2::Status status = state.decoder.decode_symbol(cdf, 2, &value);
  if (status != av2::Status::kOk) {
    return fail(state.error,
                std::string("decode_symbol failed: ") +
                    av2::status_name(status));
  }
  ++state.stats.symbols;
  return true;
}

bool read_literal_bit(DecodeState& state, int& value) {
  std::uint32_t raw = 0;
  const av2::Status status = state.decoder.read_literal(1, &raw);
  if (status != av2::Status::kOk) {
    return fail(state.error,
                std::string("read_literal failed: ") +
                    av2::status_name(status));
  }
  ++state.stats.literal_bits;
  value = static_cast<int>(raw);
  return true;
}

// read_partition() for a non-forced node.
bool read_partition_symbols(DecodeState& state, const PartitionSet& allowed,
                            int mi_row, int mi_col, BlockSize size,
                            PartitionType& partition) {
  const partition::PartitionContexts base = partition::partition_contexts(
      state.geometry, mi_row, mi_col, size, state.grid);

  SymbolContexts contexts;
  contexts.do_split = base.do_split;
  contexts.do_square_split = base.do_square_split;
  contexts.rect_type = base.rect_type;

  int value = 0;
  if (allowed.is_allowed(PartitionType::kNone)) {
    if (!read_cdf_symbol(state, SymbolKind::kDoSplit, contexts.do_split,
                         value)) {
      return false;
    }
    if (value == 0) {
      partition = PartitionType::kNone;
      return true;
    }
  }
  if (allowed.is_allowed(PartitionType::kSplit)) {
    if (!read_cdf_symbol(state, SymbolKind::kDoSquareSplit,
                         contexts.do_square_split, value)) {
      return false;
    }
    if (value != 0) {
      partition = PartitionType::kSplit;
      return true;
    }
  }

  PartitionType rect = implied_rect_type(size);
  if (rect == PartitionType::kNone) {
    const bool allow_horz = horizontal_family_allowed(allowed);
    const bool allow_vert = vertical_family_allowed(allowed);
    if (!allow_horz) {
      rect = PartitionType::kVert;
    } else if (!allow_vert) {
      rect = PartitionType::kHorz;
    }
  }
  if (rect == PartitionType::kNone) {
    if (!read_cdf_symbol(state, SymbolKind::kRectType, contexts.rect_type,
                         value)) {
      return false;
    }
    rect = value != 0 ? PartitionType::kVert : PartitionType::kHorz;
  }
  const bool horizontal = rect == PartitionType::kHorz;
  const ExtFlags flags = ext_flags(allowed, horizontal);

  bool do_ext = false;
  if (flags.non_ext && (flags.ext3 || flags.ext4)) {
    if (!read_cdf_symbol(state, SymbolKind::kDoExtPartition,
                         ext_partition_context(state.geometry, mi_row, mi_col,
                                               size, horizontal, state.grid),
                         value)) {
      return false;
    }
    do_ext = value != 0;
  } else {
    do_ext = flags.ext3 || flags.ext4;
  }

  bool do_uneven = false;
  int type = 0;
  if (do_ext) {
    if (flags.ext3 && flags.ext4) {
      if (!read_cdf_symbol(state, SymbolKind::kDoUneven4wayPartition,
                           ext_partition_context(state.geometry, mi_row, mi_col,
                                                 size, horizontal, state.grid),
                           value)) {
        return false;
      }
      do_uneven = value != 0;
    } else {
      do_uneven = flags.ext4;
    }
    if (do_uneven) {
      if (!read_literal_bit(state, type)) return false;
    }
  }

  partition = rect_part_table(do_ext, do_uneven, type, horizontal);
  return true;
}

bool encode_node(EncodeState& state, const PartitionNode& node, int depth,
                 bool has_chroma, bool chroma_offset,
                 const ChromaBlock& inherited) {
  const int r = node.mi_row;
  const int c = node.mi_col;
  const BlockSize size = node.size;
  if (!partition::is_valid_block_size(size)) {
    return fail(state.error, "node has an invalid block size");
  }
  const Boundary boundary = implied_partition(state.geometry, r, c, size);
  const PartitionSet allowed = init_allowed_partitions(
      state.geometry, r, c, size, has_chroma, chroma_offset, state.tools);
  if (boundary.has_rows != node.boundary.has_rows ||
      boundary.has_cols != node.boundary.has_cols ||
      boundary.implied != node.boundary.implied ||
      boundary.implied_partition != node.boundary.implied_partition ||
      allowed.count != node.allowed.count) {
    return fail(state.error, "node has stale boundary/allowed fields");
  }
  for (int i = 0; i < kPartitionCount; ++i) {
    if (allowed.allowed[i] != node.allowed.allowed[i]) {
      return fail(state.error, "node has stale allowed set");
    }
  }
  if (!allowed.is_allowed(node.partition)) {
    return fail(state.error, "node uses a partition that is not allowed");
  }
  if (node.depth != depth || node.has_chroma != has_chroma ||
      node.chroma_offset != chroma_offset) {
    return fail(state.error, "node depth/chroma arguments do not match the walk");
  }

  const ChromaState chroma =
      resolve_chroma(state.geometry, r, c, size, node.partition, has_chroma,
                     chroma_offset, inherited);
  if (chroma.block != node.chroma) {
    return fail(state.error, "node has a stale chroma block");
  }

  const SymbolContexts contexts = node_contexts(state.geometry, node, state.grid);
  const SymbolPlan plan = plan_partition_symbols(boundary, allowed, size,
                                                 node.partition, contexts);
  if (!plan.ok) return fail(state.error, plan.error);
  for (const PartitionSymbol& symbol : plan.symbols) {
    if (!emit_symbol(state, symbol)) return false;
    if (state.trace != nullptr) state.trace->push_back(symbol);
  }
  ++state.stats.nodes;

  if (node.partition == PartitionType::kNone) {
    if (!node.children.empty()) {
      return fail(state.error, "PARTITION_NONE node has children");
    }
    state.grid.fill_block(r, c, size);
    ++state.stats.leaves;
    return true;
  }

  ChildSlot slots[4];
  const int slot_count = enumerate_children(
      state.geometry, r, c, size, node.partition, has_chroma, chroma_offset,
      inherited, slots);
  if (slot_count == 0) {
    return fail(state.error, "partition has no child slots");
  }
  std::size_t child_index = 0;
  for (int index = 0; index < slot_count; ++index) {
    const ChildSlot& slot = slots[index];
    if (!child_is_coded(state.geometry, slot)) continue;
    if (child_index >= node.children.size()) {
      return fail(state.error, "node is missing a child");
    }
    const PartitionNode& child = node.children[child_index];
    if (child.mi_row != slot.mi_row || child.mi_col != slot.mi_col ||
        child.size != slot.size) {
      return fail(state.error,
                  "child position/size does not match the partition slots");
    }
    if (!encode_node(state, child, depth + 1, slot.has_chroma,
                     slot.chroma_offset, slot.chroma)) {
      return false;
    }
    ++child_index;
  }
  if (child_index != node.children.size()) {
    return fail(state.error, "node has an unexpected extra child");
  }
  return true;
}

bool decode_node(DecodeState& state, int r, int c, BlockSize size, int depth,
                 bool has_chroma, bool chroma_offset,
                 const ChromaBlock& inherited, PartitionNode& node) {
  const Boundary boundary = implied_partition(state.geometry, r, c, size);
  const PartitionSet allowed = init_allowed_partitions(
      state.geometry, r, c, size, has_chroma, chroma_offset, state.tools);

  PartitionType partition = PartitionType::kNone;
  if (boundary.implied && allowed.is_allowed(boundary.implied_partition)) {
    partition = boundary.implied_partition;
  } else if (allowed.count == 1) {
    for (int index = 0; index < kPartitionCount; ++index) {
      const PartitionType candidate = static_cast<PartitionType>(index);
      if (allowed.is_allowed(candidate)) {
        partition = candidate;
        break;
      }
    }
  } else if (!read_partition_symbols(state, allowed, r, c, size,
                                     partition)) {
    return false;
  }

  const ChromaState chroma = resolve_chroma(
      state.geometry, r, c, size, partition, has_chroma, chroma_offset,
      inherited);

  node.mi_row = r;
  node.mi_col = c;
  node.size = size;
  node.partition = partition;
  node.depth = depth;
  node.has_chroma = has_chroma;
  node.chroma_offset = chroma_offset;
  node.chroma = chroma.block;
  node.boundary = boundary;
  node.allowed = allowed;
  node.forced = (boundary.implied && allowed.is_allowed(boundary.implied_partition)) ||
                allowed.count == 1;
  node.cost = 0.0;
  node.children.clear();
  ++state.stats.nodes;

  if (partition == PartitionType::kNone) {
    state.grid.fill_block(r, c, size);
    ++state.stats.leaves;
    return true;
  }

  ChildSlot slots[4];
  const int slot_count = enumerate_children(
      state.geometry, r, c, size, partition, has_chroma, chroma_offset,
      inherited, slots);
  if (slot_count == 0) {
    return fail(state.error, "partition has no child slots");
  }
  for (int index = 0; index < slot_count; ++index) {
    const ChildSlot& slot = slots[index];
    if (!child_is_coded(state.geometry, slot)) continue;
    PartitionNode child;
    if (!decode_node(state, slot.mi_row, slot.mi_col, slot.size, depth + 1,
                     slot.has_chroma, slot.chroma_offset, slot.chroma,
                     child)) {
      return false;
    }
    node.children.push_back(std::move(child));
  }
  return true;
}

}  // namespace

void init_default_cdfs(PartitionCdfs& cdfs) {
  std::memcpy(cdfs.do_split, kDoSplitCdf, sizeof(cdfs.do_split));
  std::memcpy(cdfs.do_square_split, kDoSquareSplitCdf,
              sizeof(cdfs.do_square_split));
  std::memcpy(cdfs.rect_type, kRectTypeCdf, sizeof(cdfs.rect_type));
  std::memcpy(cdfs.do_ext_partition, kDoExtPartitionCdf,
              sizeof(cdfs.do_ext_partition));
  std::memcpy(cdfs.do_uneven_4way_partition, kDoUneven4wayPartitionCdf,
              sizeof(cdfs.do_uneven_4way_partition));
}

const char* symbol_name(SymbolKind kind) {
  switch (kind) {
    case SymbolKind::kDoSplit:
      return "do_split";
    case SymbolKind::kDoSquareSplit:
      return "do_square_split";
    case SymbolKind::kRectType:
      return "rect_type";
    case SymbolKind::kDoExtPartition:
      return "do_ext_partition";
    case SymbolKind::kDoUneven4wayPartition:
      return "do_uneven_4way_partition";
    case SymbolKind::kUneven4wayType:
      return "uneven_4way_partition_type";
  }
  return "invalid";
}

int symbol_alphabet(SymbolKind kind) {
  return kind == SymbolKind::kUneven4wayType ? 0 : 2;
}

int max_symbol_context(SymbolKind kind) {
  switch (kind) {
    case SymbolKind::kDoSquareSplit:
      return 7;
    case SymbolKind::kUneven4wayType:
      return 0;
    default:
      return 63;
  }
}

SymbolContexts node_contexts(const FrameGeometry& geometry,
                             const PartitionNode& node,
                             const MiSizeGrid& luma_sizes) {
  const partition::PartitionContexts base = partition::partition_contexts(
      geometry, node.mi_row, node.mi_col, node.size, luma_sizes);
  SymbolContexts contexts;
  contexts.do_split = base.do_split;
  contexts.do_square_split = base.do_square_split;
  contexts.rect_type = base.rect_type;
  const PartitionType direction = rect_direction(node.partition);
  if (direction != PartitionType::kNone) {
    contexts.do_ext = ext_partition_context(
        geometry, node.mi_row, node.mi_col, node.size,
        direction == PartitionType::kHorz, luma_sizes);
  }
  return contexts;
}

SymbolPlan plan_partition_symbols(const Boundary& boundary,
                                  const PartitionSet& allowed, BlockSize size,
                                  PartitionType partition,
                                  const SymbolContexts& contexts) {
  SymbolPlan plan;
  auto push = [&plan](SymbolKind kind, int value, int context) {
    PartitionSymbol symbol;
    symbol.kind = kind;
    symbol.value = value;
    symbol.context = context;
    symbol.literal = kind == SymbolKind::kUneven4wayType;
    plan.symbols.push_back(symbol);
  };

  if (boundary.implied && allowed.is_allowed(boundary.implied_partition)) {
    plan.forced = true;
    if (partition != boundary.implied_partition) {
      plan.ok = false;
      plan.error = "chosen partition is not the boundary-implied partition";
    }
    return plan;
  }
  if (allowed.count == 1) {
    plan.forced = true;
    for (int index = 0; index < kPartitionCount; ++index) {
      const PartitionType candidate = static_cast<PartitionType>(index);
      if (allowed.is_allowed(candidate)) {
        if (partition != candidate) {
          plan.ok = false;
          plan.error = "chosen partition is not the single allowed one";
        }
        return plan;
      }
    }
    plan.ok = false;
    plan.error = "allowed set is empty";
    return plan;
  }

  if (!allowed.is_allowed(partition)) {
    plan.ok = false;
    plan.error = "chosen partition is not allowed";
    return plan;
  }

  if (allowed.is_allowed(PartitionType::kNone)) {
    const int value = partition == PartitionType::kNone ? 0 : 1;
    push(SymbolKind::kDoSplit, value, contexts.do_split);
    if (value == 0) return plan;
  } else if (partition == PartitionType::kNone) {
    plan.ok = false;
    plan.error = "PARTITION_NONE is not allowed";
    return plan;
  }

  if (allowed.is_allowed(PartitionType::kSplit)) {
    const int value = partition == PartitionType::kSplit ? 1 : 0;
    push(SymbolKind::kDoSquareSplit, value, contexts.do_square_split);
    if (value == 1) return plan;
  } else if (partition == PartitionType::kSplit) {
    plan.ok = false;
    plan.error = "PARTITION_SPLIT is not allowed";
    return plan;
  }

  PartitionType rect = implied_rect_type(size);
  if (rect == PartitionType::kNone) {
    const bool allow_horz = horizontal_family_allowed(allowed);
    const bool allow_vert = vertical_family_allowed(allowed);
    if (!allow_horz) {
      rect = PartitionType::kVert;
    } else if (!allow_vert) {
      rect = PartitionType::kHorz;
    }
  }
  if (rect == PartitionType::kNone) {
    const PartitionType direction = rect_direction(partition);
    if (direction == PartitionType::kNone) {
      plan.ok = false;
      plan.error = "chosen partition has no rect direction";
      return plan;
    }
    rect = direction;
    push(SymbolKind::kRectType,
         direction == PartitionType::kVert ? 1 : 0, contexts.rect_type);
  }
  const bool horizontal = rect == PartitionType::kHorz;
  if (rect_direction(partition) != rect) {
    plan.ok = false;
    plan.error = "chosen partition does not match the rect direction";
    return plan;
  }

  const ExtFlags flags = ext_flags(allowed, horizontal);
  const bool chosen_four_way = is_four_way(partition);
  bool do_ext = false;
  if (flags.non_ext && (flags.ext3 || flags.ext4)) {
    do_ext = chosen_four_way;
    push(SymbolKind::kDoExtPartition, do_ext ? 1 : 0, contexts.do_ext);
  } else {
    do_ext = flags.ext3 || flags.ext4;
    if (do_ext != chosen_four_way) {
      plan.ok = false;
      plan.error = "do_ext_partition is implied but the partition disagrees";
      return plan;
    }
  }

  if (!do_ext) {
    if (partition != rect) {
      plan.ok = false;
      plan.error = "2-way partition is not the rect direction";
    }
    return plan;
  }

  const bool chosen_uneven = is_uneven_4way(partition);
  bool do_uneven = false;
  if (flags.ext3 && flags.ext4) {
    do_uneven = chosen_uneven;
    push(SymbolKind::kDoUneven4wayPartition, do_uneven ? 1 : 0,
         contexts.do_ext);
  } else {
    do_uneven = flags.ext4;
    if (do_uneven != chosen_uneven) {
      plan.ok = false;
      plan.error =
          "do_uneven_4way_partition is implied but the partition disagrees";
      return plan;
    }
  }

  if (!do_uneven) {
    if (partition != three_way_for(horizontal)) {
      plan.ok = false;
      plan.error = "3-way partition does not match the rect direction";
    }
    return plan;
  }

  if (partition != PartitionType::kHorz4A && partition != PartitionType::kHorz4B &&
      partition != PartitionType::kVert4A && partition != PartitionType::kVert4B) {
    plan.ok = false;
    plan.error = "uneven 4-way partition expected";
    return plan;
  }
  const bool b = partition == PartitionType::kHorz4B ||
                 partition == PartitionType::kVert4B;
  if (partition != four_way_a_for(horizontal) && !b) {
    plan.ok = false;
    plan.error = "uneven 4-way partition does not match the rect direction";
    return plan;
  }
  push(SymbolKind::kUneven4wayType, b ? 1 : 0, 0);
  return plan;
}

double node_signal_cost(const Boundary& boundary, const PartitionSet& allowed,
                        BlockSize size, PartitionType partition,
                        const SymbolCosts& costs) {
  SymbolContexts contexts;
  const SymbolPlan plan =
      plan_partition_symbols(boundary, allowed, size, partition, contexts);
  if (!plan.ok) return kInfiniteCost;
  double total = 0.0;
  for (const PartitionSymbol& symbol : plan.symbols) {
    double value = 0.0;
    switch (symbol.kind) {
      case SymbolKind::kDoSplit:
        value = costs.do_split[symbol.value];
        break;
      case SymbolKind::kDoSquareSplit:
        value = costs.do_square_split[symbol.value];
        break;
      case SymbolKind::kRectType:
        value = costs.rect_type[symbol.value];
        break;
      case SymbolKind::kDoExtPartition:
        value = costs.do_ext_partition[symbol.value];
        break;
      case SymbolKind::kDoUneven4wayPartition:
        value = costs.do_uneven_4way_partition[symbol.value];
        break;
      case SymbolKind::kUneven4wayType:
        value = costs.uneven_4way_type[symbol.value];
        break;
    }
    if (!std::isfinite(value) || value < 0.0) return kInfiniteCost;
    total += value;
  }
  return total;
}

bool encode_tree_symbols(const FrameGeometry& geometry, const Tools& tools,
                         const PartitionNode& root, PartitionCdfs& cdfs,
                         MiSizeGrid& grid, av2::Encoder& encoder,
                         bool disable_cdf_update, SyntaxStats* stats,
                         std::string& error,
                         std::vector<PartitionSymbol>* trace) {
  std::string tools_error;
  if (!validate_tools(tools, tools_error)) {
    return fail(error, std::move(tools_error));
  }
  if (root.size != partition::superblock_size(geometry.sb_size)) {
    return fail(error, "root is not a superblock-sized node");
  }
  const int root_mi = partition::sb_size_mi(geometry);
  if (root.mi_row < 0 || root.mi_col < 0 || (root.mi_row % root_mi) != 0 ||
      (root.mi_col % root_mi) != 0) {
    return fail(error, "root origin is not superblock-aligned");
  }

  encoder.disable_cdf_update = disable_cdf_update ? 1u : 0u;
  EncodeState state{geometry, tools, cdfs, grid, encoder, {}, {}, trace};
  ChromaBlock inherited;
  inherited.mi_row = root.mi_row;
  inherited.mi_col = root.mi_col;
  inherited.size = root.size;
  if (!encode_node(state, root, 0, geometry.has_chroma, false, inherited)) {
    error = std::move(state.error);
    return false;
  }
  if (stats != nullptr) *stats = state.stats;
  error.clear();
  return true;
}

bool decode_tree_symbols(const FrameGeometry& geometry, const Tools& tools,
                         PartitionNode& root, PartitionCdfs& cdfs,
                         MiSizeGrid& grid, av2::Decoder& decoder,
                         SyntaxStats* stats, std::string& error) {
  std::string tools_error;
  if (!validate_tools(tools, tools_error)) {
    return fail(error, std::move(tools_error));
  }
  if (root.size != partition::superblock_size(geometry.sb_size)) {
    return fail(error, "root is not a superblock-sized node");
  }
  const int size_mi = partition::sb_size_mi(geometry);
  if (root.mi_row < 0 || root.mi_col < 0 ||
      (root.mi_row % size_mi) != 0 || (root.mi_col % size_mi) != 0) {
    return fail(error, "root origin is not superblock-aligned");
  }

  DecodeState state{geometry, tools, cdfs, grid, decoder, {}, {}};
  ChromaBlock inherited;
  inherited.mi_row = root.mi_row;
  inherited.mi_col = root.mi_col;
  inherited.size = root.size;
  if (!decode_node(state, root.mi_row, root.mi_col, root.size, 0,
                   geometry.has_chroma, false, inherited, root)) {
    error = std::move(state.error);
    return false;
  }
  if (stats != nullptr) *stats = state.stats;
  error.clear();
  return true;
}

std::string tree_signature(const PartitionNode& node) {
  std::ostringstream out;
  out << partition::block_size_name(node.size) << '@' << node.mi_row << ','
      << node.mi_col << ':' << partition_name(node.partition) << ':'
      << (node.forced ? 'F' : 'S') << '(';
  for (const PartitionNode& child : node.children) {
    out << tree_signature(child);
  }
  out << ')';
  return out.str();
}

}  // namespace av2enc::partition_ext
