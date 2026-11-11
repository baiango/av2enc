// SDP partition contexts and the §5.20.3.2 symbol plan / independent decoder.

#include "encoder/sdp/sdp.h"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include "encoder/sdp/sdp_internal.h"

namespace av2enc::sdp {
namespace {

using detail::log2_or_sentinel;
using detail::rect_is_horizontal;
using detail::rect_is_vertical;
using detail::rect_part_table;
using detail::same_set;
using detail::set_allows;

bool can_code_region_type(const FrameGeometry& geometry, const Tools& tools,
                          TreeType tree, BlockSize size,
                          PartitionType partition) {
  return tools.enable_extended_sdp && !tools.frame_is_intra &&
         tree == TreeType::kSharedPart && bru_mode_active(tools) &&
         size != partition::superblock_size(geometry.sb_size) &&
         is_bsize_allowed_for_extended_sdp(size, partition);
}

bool next_extended_sdp_allowed(const FrameGeometry& geometry, BlockSize size,
                               PartitionType partition, bool current) {
  (void)geometry;
  const BlockSize subsize = sdp_subsize(partition, size);
  if (!partition::is_valid_block_size(subsize)) return false;
  bool next = current && partition::block_width(subsize) > 4 &&
              partition::block_height(subsize) > 4;
  if (partition == PartitionType::kHorz3 ||
      partition == PartitionType::kVert3) {
    const BlockSize mid = sdp_midsize(size);
    next = next && partition::block_width(mid) > 4 &&
           partition::block_height(mid) > 4;
  }
  return next;
}

void fill_leaf_grids(const Node& node, MiSizeGrid& luma, MiSizeGrid& chroma) {
  if (node.tree_type != TreeType::kChromaPart) {
    luma.fill_block(node.mi_row, node.mi_col, node.size);
  }
  if (node.codes_chroma) {
    chroma.fill_block(node.chroma.mi_row, node.chroma.mi_col, node.chroma.size);
  }
}

}  // namespace

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
    case SymbolKind::kRegionType:
      return "region_type";
  }
  return "?";
}

int symbol_alphabet(SymbolKind kind) {
  return kind == SymbolKind::kUneven4wayType ? 0 : 2;
}

int max_symbol_context(SymbolKind kind) {
  switch (kind) {
    case SymbolKind::kDoSplit:
    case SymbolKind::kRectType:
    case SymbolKind::kDoExtPartition:
    case SymbolKind::kDoUneven4wayPartition:
      return 63;
    case SymbolKind::kDoSquareSplit:
      return 7;
    case SymbolKind::kRegionType:
      return 3;
    case SymbolKind::kUneven4wayType:
      return 0;
  }
  return 0;
}

int region_type_context(BlockSize size) {
  const int num_samples =
      partition::block_width(size) * partition::block_height(size);
  if (num_samples <= 128) return 0;
  if (num_samples <= 512) return 1;
  if (num_samples <= 1024) return 2;
  return 3;
}

namespace {

// do_ext_partition / do_uneven_4way_partition context.  `horizontal` is the
// rectType chosen by read_partition(), so this must be called after the
// partition is known (the decoder recomputes it at that point).
int do_ext_context(BlockSize size, bool horizontal,
                   const MiSizeGrid& plane_sizes, int mi_row, int mi_col) {
  const int w4 = partition::num_4x4_blocks_wide(size);
  const int h4 = partition::num_4x4_blocks_high(size);
  int ext1 = 0;
  int ext2 = 0;
  if (horizontal) {
    const int ext_bsh = std::max(partition::mi_height_log2(size) - 1, 1);
    ext1 = log2_or_sentinel(plane_sizes, mi_row, mi_col - 1, false) < ext_bsh
               ? 1
               : 0;
    ext2 = log2_or_sentinel(plane_sizes, mi_row + (h4 >> 1), mi_col - 1,
                            false) < ext_bsh
               ? 1
               : 0;
  } else {
    const int ext_bsw = std::max(partition::mi_width_log2(size) - 1, 1);
    ext1 = log2_or_sentinel(plane_sizes, mi_row - 1, mi_col, true) < ext_bsw
               ? 1
               : 0;
    ext2 = log2_or_sentinel(plane_sizes, mi_row - 1, mi_col + (w4 >> 1),
                            true) < ext_bsw
               ? 1
               : 0;
  }
  return sdp_size_adjust(size) * 4 + ext1 * 2 + ext2;
}

}  // namespace

SymbolContexts node_contexts(const FrameGeometry& geometry, const Node& node,
                             const MiSizeGrid& plane_sizes) {
  SymbolContexts out;
  if (!partition::is_valid_block_size(node.size)) return out;
  const int bsw = std::max(partition::mi_width_log2(node.size), 1);
  const int bsh = std::max(partition::mi_height_log2(node.size), 1);
  const int ctx1 =
      log2_or_sentinel(plane_sizes, node.mi_row, node.mi_col - 1, false) < bsh
          ? 1
          : 0;
  const int ctx2 =
      log2_or_sentinel(plane_sizes, node.mi_row - 1, node.mi_col, true) < bsw
          ? 1
          : 0;
  out.do_split = sdp_size_adjust(node.size) * 4 + ctx1 * 2 + ctx2;
  out.rect_type =
      sdp_size_adjust_rect(node.size) * 4 + ctx1 * 2 + ctx2;

  const int raw_bsw = partition::mi_width_log2(node.size);
  const int raw_bsh = partition::mi_height_log2(node.size);
  const bool avail_left =
      partition::is_inside(geometry, node.mi_row, node.mi_col - 1);
  const bool avail_above =
      partition::is_inside(geometry, node.mi_row - 1, node.mi_col);
  const bool left_smaller =
      avail_left &&
      (log2_or_sentinel(plane_sizes, node.mi_row, node.mi_col - 1, false) <
       raw_bsh);
  const bool above_smaller =
      avail_above &&
      (log2_or_sentinel(plane_sizes, node.mi_row - 1, node.mi_col, true) <
       raw_bsw);
  out.do_square_split = (node.size == BlockSize::k256x256 ? 4 : 0) +
                        (left_smaller ? 2 : 0) + (above_smaller ? 1 : 0);

  out.do_ext = do_ext_context(node.size, rect_is_horizontal(node.partition),
                              plane_sizes, node.mi_row, node.mi_col);
  out.region_type = region_type_context(node.size);
  return out;
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

SymbolPlan plan_partition_symbols(const Boundary& boundary,
                                  const PartitionSet& allowed, BlockSize size,
                                  TreeType tree_type, PartitionType partition,
                                  const SymbolContexts& contexts,
                                  bool bru_active) {
  SymbolPlan plan;
  const int plane = plane_start(tree_type);
  auto add = [&plan, plane](SymbolKind kind, int value, int context,
                            bool literal = false) {
    Symbol symbol;
    symbol.kind = kind;
    symbol.value = value;
    symbol.context = context;
    symbol.plane = plane;
    symbol.literal = literal;
    plan.symbols.push_back(symbol);
  };
  auto fail_plan = [&plan](std::string message) {
    plan.ok = false;
    plan.error = std::move(message);
    return plan;
  };

  if (boundary.implied && set_allows(allowed, boundary.implied_partition)) {
    plan.forced = true;
    if (partition != boundary.implied_partition) {
      return fail_plan("chosen partition is not the implied partition");
    }
    return plan;
  }
  if (allowed.count == 1) {
    plan.forced = true;
    for (int i = 0; i < kPartitionCount; ++i) {
      if (allowed.allowed[i]) {
        if (partition != static_cast<PartitionType>(i)) {
          return fail_plan("chosen partition is not the only allowed one");
        }
        break;
      }
    }
    return plan;
  }
  if (!bru_active) {
    plan.forced = true;
    if (partition != PartitionType::kNone) {
      return fail_plan("BRU-inactive read_partition() forces PARTITION_NONE");
    }
    return plan;
  }

  if (set_allows(allowed, PartitionType::kNone)) {
    const bool split = partition != PartitionType::kNone;
    add(SymbolKind::kDoSplit, split ? 1 : 0, contexts.do_split);
    if (!split) {
      if (partition != PartitionType::kNone) {
        return fail_plan("do_split=0 but the partition is not NONE");
      }
      return plan;
    }
  } else if (partition == PartitionType::kNone) {
    return fail_plan("PARTITION_NONE chosen while do_split is implied");
  }

  if (partition == PartitionType::kSplit) {
    if (!set_allows(allowed, PartitionType::kSplit)) {
      return fail_plan("PARTITION_SPLIT is not allowed");
    }
    add(SymbolKind::kDoSquareSplit, 1, contexts.do_square_split);
    return plan;
  }
  if (set_allows(allowed, PartitionType::kSplit)) {
    add(SymbolKind::kDoSquareSplit, 0, contexts.do_square_split);
  }

  const partition::RectTypeImplied implied =
      sdp_rect_implied(size, tree_type);
  bool horizontal = true;
  if (implied.implied) {
    horizontal = implied.partition == partition::PartitionType::kHorz;
  } else {
    const bool allow_h = set_allows(allowed, PartitionType::kHorz) ||
                         set_allows(allowed, PartitionType::kHorz3) ||
                         set_allows(allowed, PartitionType::kHorz4A) ||
                         set_allows(allowed, PartitionType::kHorz4B);
    const bool allow_v = set_allows(allowed, PartitionType::kVert) ||
                         set_allows(allowed, PartitionType::kVert3) ||
                         set_allows(allowed, PartitionType::kVert4A) ||
                         set_allows(allowed, PartitionType::kVert4B);
    if (!allow_h) {
      horizontal = false;
    } else if (!allow_v) {
      horizontal = true;
    } else {
      horizontal = rect_is_horizontal(partition);
      add(SymbolKind::kRectType, horizontal ? 0 : 1, contexts.rect_type);
    }
  }
  if (horizontal ? !rect_is_horizontal(partition)
                 : !rect_is_vertical(partition)) {
    return fail_plan("chosen partition does not match the coded rect_type");
  }

  const bool non_ext = horizontal ? set_allows(allowed, PartitionType::kHorz)
                                  : set_allows(allowed, PartitionType::kVert);
  const bool ext3 = horizontal ? set_allows(allowed, PartitionType::kHorz3)
                               : set_allows(allowed, PartitionType::kVert3);
  const bool ext4 =
      horizontal ? (set_allows(allowed, PartitionType::kHorz4A) ||
                    set_allows(allowed, PartitionType::kHorz4B))
                 : (set_allows(allowed, PartitionType::kVert4A) ||
                    set_allows(allowed, PartitionType::kVert4B));

  bool do_ext = false;
  if (non_ext && (ext3 || ext4)) {
    do_ext = detail::is_three_way(partition) ||
             detail::is_uneven_4way(partition);
    add(SymbolKind::kDoExtPartition, do_ext ? 1 : 0, contexts.do_ext);
  } else {
    do_ext = ext3 || ext4;
  }
  if (!do_ext && (detail::is_three_way(partition) ||
                  detail::is_uneven_4way(partition))) {
    return fail_plan("chosen extended partition is not available");
  }

  bool do_uneven = false;
  int type = 0;
  if (do_ext) {
    if (ext3 && ext4) {
      do_uneven = detail::is_uneven_4way(partition);
      add(SymbolKind::kDoUneven4wayPartition, do_uneven ? 1 : 0,
          contexts.do_ext);
    } else {
      do_uneven = ext4;
    }
    if (do_uneven) {
      type = (partition == PartitionType::kHorz4B ||
              partition == PartitionType::kVert4B)
                 ? 1
                 : 0;
      add(SymbolKind::kUneven4wayType, type, 0, /*literal=*/true);
    } else if (detail::is_uneven_4way(partition)) {
      return fail_plan("uneven 4-way chosen but not available");
    }
  }

  if (rect_part_table(do_ext, do_uneven, type, horizontal) != partition) {
    return fail_plan("symbol plan does not encode the chosen partition");
  }
  return plan;
}

namespace {

struct Planner {
  const FrameGeometry& geometry;
  const Tools& tools;
  MiSizeGrid& luma;
  MiSizeGrid& chroma;
  SuperblockPlan& plan;

  bool node(const Node& current, int plane) {
    if (current.sdp_split) {
      if (current.children.size() != 2) {
        return fail("SDP split node has " +
                    std::to_string(current.children.size()) + " children");
      }
      switch_event(SwitchKind::kSdpIntraLumaStart, current);
      if (!node(current.children[0], 0)) return false;
      switch_event(SwitchKind::kSdpIntraChromaStart, current);
      if (!node(current.children[1], 1)) return false;
      switch_event(SwitchKind::kSdpIntraEnd, current);
      return true;
    }
    if (current.sdp_chroma_block) {
      fill_leaf_grids(current, luma, chroma);
      return true;
    }

    const MiSizeGrid& plane_sizes = plane == 0 ? luma : chroma;
    const SymbolContexts contexts = node_contexts(geometry, current, plane_sizes);
    const SymbolPlan symbols = plan_partition_symbols(
        current.boundary, current.allowed, current.size, current.tree_type,
        current.partition, contexts, bru_mode_active(tools));
    if (!symbols.ok) {
      return fail("node " + tree_signature(current) + ": " + symbols.error);
    }
    plan.symbols.insert(plan.symbols.end(), symbols.symbols.begin(),
                        symbols.symbols.end());

    if (current.region_type_coded) {
      Symbol symbol;
      symbol.kind = SymbolKind::kRegionType;
      symbol.value = current.extended_sdp ? 0 : 1;
      symbol.context = contexts.region_type;
      symbol.plane = 0;
      plan.symbols.push_back(symbol);
    }

    if (current.partition == PartitionType::kNone) {
      fill_leaf_grids(current, luma, chroma);
      return true;
    }

    if (current.extended_sdp) {
      if (current.children.empty()) {
        return fail("extended-SDP node has no children");
      }
      switch_event(SwitchKind::kSdpExtendedLumaStart, current);
      for (std::size_t i = 0; i + 1 < current.children.size(); ++i) {
        if (!node(current.children[i], 0)) return false;
      }
      switch_event(SwitchKind::kSdpExtendedChromaStart, current);
      if (!node(current.children.back(), 1)) return false;
      switch_event(SwitchKind::kSdpExtendedEnd, current);
      return true;
    }
    for (const Node& child : current.children) {
      if (!node(child, plane)) return false;
    }
    return true;
  }

  void switch_event(SwitchKind kind, const Node& current) {
    TreeSwitch event;
    event.kind = kind;
    event.mi_row = current.mi_row;
    event.mi_col = current.mi_col;
    event.size = current.size;
    plan.switches.push_back(event);
  }

  bool fail(std::string message) {
    plan.ok = false;
    plan.error = std::move(message);
    return false;
  }
};

}  // namespace

SuperblockPlan plan_superblock_symbols(const FrameGeometry& geometry,
                                       const Tools& tools, const Node& root,
                                       MiSizeGrid& luma_sizes,
                                       MiSizeGrid& chroma_sizes) {
  SuperblockPlan plan;
  Planner planner{geometry, tools, luma_sizes, chroma_sizes, plan};
  if (!planner.node(root, 0)) {
    plan.ok = false;
    if (plan.error.empty()) plan.error = "symbol plan failed";
    return plan;
  }
  plan.ok = true;
  return plan;
}

// ---------------------------------------------------------------------------
// Independent decoder
// ---------------------------------------------------------------------------

namespace {

struct Decoder {
  const FrameGeometry& geometry;
  const Tools& tools;
  MiSizeGrid& luma;
  MiSizeGrid& chroma;
  const std::vector<Symbol>& symbols;
  std::size_t cursor = 0;
  std::vector<TreeSwitch> switches;
  std::string error;

  bool fail(std::string message) {
    error = std::move(message);
    return false;
  }

  bool read(SymbolKind kind, int plane, int context, bool literal, int& value) {
    if (cursor >= symbols.size()) {
      return fail(std::string("unexpected end of symbols reading ") +
                  symbol_name(kind));
    }
    const Symbol& symbol = symbols[cursor];
    if (symbol.kind != kind || symbol.plane != plane ||
        symbol.context != context || symbol.literal != literal) {
      std::ostringstream out;
      out << "symbol mismatch: got " << symbol_name(symbol.kind)
          << "(plane " << symbol.plane << ", ctx " << symbol.context
          << (symbol.literal ? ", literal" : "") << "), expected "
          << symbol_name(kind) << "(plane " << plane << ", ctx " << context
          << (literal ? ", literal" : "") << ")";
      return fail(out.str());
    }
    if (!literal && (symbol.value < 0 || symbol.value > 1)) {
      return fail("binary symbol has an out-of-range value");
    }
    value = symbol.value;
    ++cursor;
    return true;
  }

  PartitionType read_partition(const Boundary& boundary,
                               const PartitionSet& allowed, BlockSize size,
                               TreeType tree, const SymbolContexts& contexts,
                               const MiSizeGrid& plane_sizes, int mi_row,
                               int mi_col) {
    const int plane = plane_start(tree);
    if (boundary.implied && set_allows(allowed, boundary.implied_partition)) {
      return boundary.implied_partition;
    }
    if (allowed.count == 1) {
      for (int i = 0; i < kPartitionCount; ++i) {
        if (allowed.allowed[i]) return static_cast<PartitionType>(i);
      }
    }
    if (!bru_mode_active(tools)) return PartitionType::kNone;

    int value = 0;
    if (set_allows(allowed, PartitionType::kNone)) {
      if (!read(SymbolKind::kDoSplit, plane, contexts.do_split, false, value)) {
        return PartitionType::kNone;
      }
      if (value == 0) return PartitionType::kNone;
    }
    if (set_allows(allowed, PartitionType::kSplit)) {
      if (!read(SymbolKind::kDoSquareSplit, plane, contexts.do_square_split,
                false, value)) {
        return PartitionType::kNone;
      }
      if (value != 0) return PartitionType::kSplit;
    }

    const partition::RectTypeImplied implied =
        sdp_rect_implied(size, tree);
    bool horizontal = true;
    if (implied.implied) {
      horizontal = implied.partition == partition::PartitionType::kHorz;
    } else {
      const bool allow_h = set_allows(allowed, PartitionType::kHorz) ||
                           set_allows(allowed, PartitionType::kHorz3) ||
                           set_allows(allowed, PartitionType::kHorz4A) ||
                           set_allows(allowed, PartitionType::kHorz4B);
      const bool allow_v = set_allows(allowed, PartitionType::kVert) ||
                           set_allows(allowed, PartitionType::kVert3) ||
                           set_allows(allowed, PartitionType::kVert4A) ||
                           set_allows(allowed, PartitionType::kVert4B);
      if (!allow_h) {
        horizontal = false;
      } else if (!allow_v) {
        horizontal = true;
      } else {
        if (!read(SymbolKind::kRectType, plane, contexts.rect_type, false,
                  value)) {
          return PartitionType::kNone;
        }
        horizontal = value == 0;
      }
    }

    const bool non_ext = horizontal ? set_allows(allowed, PartitionType::kHorz)
                                    : set_allows(allowed, PartitionType::kVert);
    const bool ext3 = horizontal ? set_allows(allowed, PartitionType::kHorz3)
                                 : set_allows(allowed, PartitionType::kVert3);
    const bool ext4 =
        horizontal ? (set_allows(allowed, PartitionType::kHorz4A) ||
                      set_allows(allowed, PartitionType::kHorz4B))
                   : (set_allows(allowed, PartitionType::kVert4A) ||
                      set_allows(allowed, PartitionType::kVert4B));

    bool do_ext = false;
    const int ext_ctx =
        do_ext_context(size, horizontal, plane_sizes, mi_row, mi_col);
    if (non_ext && (ext3 || ext4)) {
      if (!read(SymbolKind::kDoExtPartition, plane, ext_ctx, false,
                value)) {
        return PartitionType::kNone;
      }
      do_ext = value != 0;
    } else {
      do_ext = ext3 || ext4;
    }

    bool do_uneven = false;
    int type = 0;
    if (do_ext) {
      if (ext3 && ext4) {
        if (!read(SymbolKind::kDoUneven4wayPartition, plane, ext_ctx,
                  false, value)) {
          return PartitionType::kNone;
        }
        do_uneven = value != 0;
      } else {
        do_uneven = ext4;
      }
      if (do_uneven) {
        if (!read(SymbolKind::kUneven4wayType, plane, 0, true, type)) {
          return PartitionType::kNone;
        }
        if (type < 0 || type > 1) {
          fail("invalid uneven_4way_partition_type literal");
          return PartitionType::kNone;
        }
      }
    }
    return rect_part_table(do_ext, do_uneven, type, horizontal);
  }

  void switch_event(SwitchKind kind, const Node& current) {
    TreeSwitch event;
    event.kind = kind;
    event.mi_row = current.mi_row;
    event.mi_col = current.mi_col;
    event.size = current.size;
    switches.push_back(event);
  }

  bool node(Node& out, TreeType tree, int mi_row, int mi_col, BlockSize size,
            int depth, bool has_chroma, bool chroma_offset,
            const ChromaBlock& inherited, bool ext_allowed,
            RegionType incoming_region, bool chroma_known,
            PartitionType luma_partition) {
    if (!partition::is_valid_block_size(size)) {
      return fail("invalid block size");
    }
    out = Node{};
    out.mi_row = mi_row;
    out.mi_col = mi_col;
    out.size = size;
    out.depth = depth;
    out.tree_type = tree;
    out.has_chroma = has_chroma;
    out.chroma_offset = chroma_offset;
    out.extended_sdp_allowed = ext_allowed;

    const bool should_split =
        tools.enable_sdp && tools.frame_is_intra && geometry.has_chroma &&
        tree == TreeType::kSharedPart && size == BlockSize::k64x64;
    if (should_split) {
      out.sdp_split = true;
      out.partition = PartitionType::kNone;
      out.forced = true;
      out.boundary = sdp_implied(geometry, mi_row, mi_col, size, tree);
      out.chroma = inherited;
      out.children.resize(2);
      switch_event(SwitchKind::kSdpIntraLumaStart, out);
      if (!node(out.children[0], TreeType::kLumaPart, mi_row, mi_col, size,
                depth + 1, true, false, inherited, false,
                RegionType::kMixedRegion, false, PartitionType::kNone)) {
        return false;
      }
      Node& luma_node = out.children[0];
      std::vector<PartitionType> direct;
      for (const Node& child : luma_node.children) {
        direct.push_back(child.partition);
      }
      luma_node.chroma_follows_luma =
          chroma_follows_luma(luma_node.partition, direct);
      switch_event(SwitchKind::kSdpIntraChromaStart, out);
      if (!node(out.children[1], TreeType::kChromaPart, mi_row, mi_col, size,
                depth + 1, true, false, inherited, false,
                RegionType::kMixedRegion, luma_node.chroma_follows_luma,
                luma_node.partition)) {
        return false;
      }
      Node& chroma_node = out.children[1];
      chroma_node.chroma_partition_known = luma_node.chroma_follows_luma;
      chroma_node.implied_luma_partition = luma_node.partition;
      chroma_node.cfl_allowed_in_sdp =
          cfl_allowed_in_sdp(luma_node.partition,
                             luma_node.chroma_follows_luma,
                             chroma_node.partition);
      switch_event(SwitchKind::kSdpIntraEnd, out);
      return true;
    }

    Boundary boundary = sdp_implied(geometry, mi_row, mi_col, size, tree,
                                          chroma_known, luma_partition);
    PartitionSet allowed = sdp_init_allowed(
        geometry, mi_row, mi_col, size, tree, has_chroma, chroma_offset,
        incoming_region, tools);
    out.boundary = boundary;
    out.allowed = allowed;
    const bool implied_forced =
        boundary.implied && set_allows(allowed, boundary.implied_partition);
    out.forced = implied_forced || allowed.count == 1 ||
                 !bru_mode_active(tools);

    const MiSizeGrid& plane_sizes =
        tree == TreeType::kChromaPart ? chroma : luma;
    const SymbolContexts contexts =
        node_contexts(geometry, out, plane_sizes);
    const PartitionType partition =
        read_partition(boundary, allowed, size, tree, contexts, plane_sizes,
                       mi_row, mi_col);
    if (!error.empty()) return false;
    out.partition = partition;

    const bool region_coded =
        can_code_region_type(geometry, tools, tree, size, partition);
    out.region_type_coded = region_coded;
    RegionType region = incoming_region;
    if (region_coded) {
      int value = 1;
      if (!read(SymbolKind::kRegionType, 0, contexts.region_type, false,
                value)) {
        return false;
      }
      if (value == 0) {
        region = RegionType::kIntraRegion;
        out.extended_sdp = true;
      } else {
        region = RegionType::kMixedRegion;
      }
    }
    out.region_type = region;

    const ChromaState resolved = sdp_resolve_chroma(
        geometry, mi_row, mi_col, size, partition, has_chroma, chroma_offset,
        inherited);
    out.chroma = resolved.block;
    out.codes_chroma =
        partition == PartitionType::kNone && has_chroma && geometry.has_chroma &&
        tree != TreeType::kLumaPart;

    if (partition == PartitionType::kNone) {
      fill_leaf_grids(out, luma, chroma);
      return true;
    }

    partition_ext::ChildSlot slots[4];
    const int count = partition_ext::enumerate_children(
        geometry, mi_row, mi_col, size, partition, has_chroma, chroma_offset,
        inherited, slots);
    if (count <= 0) return fail("decoded partition has no children");
    std::vector<partition_ext::ChildSlot> coded;
    for (int i = 0; i < count; ++i) {
      if (partition_ext::child_is_coded(geometry, slots[i])) {
        coded.push_back(slots[i]);
      }
    }
    const bool next_ext =
        next_extended_sdp_allowed(geometry, size, partition, ext_allowed);

    if (out.extended_sdp) {
      out.children.resize(coded.size() + 1);
      switch_event(SwitchKind::kSdpExtendedLumaStart, out);
      for (std::size_t i = 0; i < coded.size(); ++i) {
        if (!node(out.children[i], TreeType::kLumaPart, coded[i].mi_row,
                  coded[i].mi_col, coded[i].size, depth + 1,
                  coded[i].has_chroma, coded[i].chroma_offset,
                  coded[i].chroma, next_ext, RegionType::kIntraRegion, false,
                  PartitionType::kNone)) {
          return false;
        }
      }
      switch_event(SwitchKind::kSdpExtendedChromaStart, out);
      Node& chroma_node = out.children.back();
      chroma_node = Node{};
      chroma_node.mi_row = mi_row;
      chroma_node.mi_col = mi_col;
      chroma_node.size = size;
      chroma_node.depth = depth + 1;
      chroma_node.tree_type = TreeType::kChromaPart;
      chroma_node.partition = PartitionType::kNone;
      chroma_node.sdp_chroma_block = true;
      chroma_node.has_chroma = true;
      chroma_node.chroma_offset = true;
      chroma_node.chroma.mi_row = mi_row;
      chroma_node.chroma.mi_col = mi_col;
      chroma_node.chroma.size = size;
      chroma_node.forced = true;
      chroma_node.codes_chroma = true;
      chroma_node.boundary = sdp_implied(
          geometry, mi_row, mi_col, size, TreeType::kChromaPart);
      chroma_node.allowed = sdp_init_allowed(
          geometry, mi_row, mi_col, size, TreeType::kChromaPart, true, true,
          RegionType::kIntraRegion, tools);
      fill_leaf_grids(chroma_node, luma, chroma);
      switch_event(SwitchKind::kSdpExtendedEnd, out);
      return true;
    }

    out.children.resize(coded.size());
    for (std::size_t i = 0; i < coded.size(); ++i) {
      if (!node(out.children[i], tree, coded[i].mi_row, coded[i].mi_col,
                coded[i].size, depth + 1, coded[i].has_chroma,
                coded[i].chroma_offset, coded[i].chroma, next_ext, region,
                false, PartitionType::kNone)) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace

bool decode_superblock_symbols(const FrameGeometry& geometry,
                               const Tools& tools, int sb_mi_row,
                               int sb_mi_col,
                               const std::vector<Symbol>& symbols,
                               MiSizeGrid& luma_sizes,
                               MiSizeGrid& chroma_sizes, Node& root,
                               std::vector<TreeSwitch>* switches,
                               std::string& error) {
  Decoder decoder{geometry, tools, luma_sizes, chroma_sizes, symbols,
                  /*cursor=*/0, /*switches=*/{}, /*error=*/{}};
  ChromaBlock inherited;
  inherited.mi_row = sb_mi_row;
  inherited.mi_col = sb_mi_col;
  inherited.size = partition::superblock_size(geometry.sb_size);
  if (!decoder.node(root, TreeType::kSharedPart, sb_mi_row, sb_mi_col,
                    inherited.size, 0, true, false, inherited,
                    tools.enable_extended_sdp && !tools.frame_is_intra,
                    RegionType::kMixedRegion, false, PartitionType::kNone)) {
    error = decoder.error;
    return false;
  }
  if (decoder.cursor != symbols.size()) {
    error = "unused symbols after the superblock";
    return false;
  }
  if (switches != nullptr) *switches = std::move(decoder.switches);
  error.clear();
  return true;
}

namespace {

void collect_events(const Node& node, std::vector<TreeSwitch>& events) {
  auto push = [&events, &node](SwitchKind kind) {
    TreeSwitch event;
    event.kind = kind;
    event.mi_row = node.mi_row;
    event.mi_col = node.mi_col;
    event.size = node.size;
    events.push_back(event);
  };
  if (node.sdp_split) {
    push(SwitchKind::kSdpIntraLumaStart);
    for (std::size_t i = 0; i < node.children.size(); ++i) {
      if (i == 1) push(SwitchKind::kSdpIntraChromaStart);
      collect_events(node.children[i], events);
    }
    push(SwitchKind::kSdpIntraEnd);
    return;
  }
  if (node.extended_sdp) {
    push(SwitchKind::kSdpExtendedLumaStart);
    for (std::size_t i = 0; i < node.children.size(); ++i) {
      if (i + 1 == node.children.size()) {
        push(SwitchKind::kSdpExtendedChromaStart);
      }
      collect_events(node.children[i], events);
    }
    push(SwitchKind::kSdpExtendedEnd);
    return;
  }
  for (const Node& child : node.children) collect_events(child, events);
}

}  // namespace

std::vector<TreeSwitch> tree_type_events(const FrameDecision& frame) {
  std::vector<TreeSwitch> events;
  for (const Node& sb : frame.superblocks) {
    collect_events(sb, events);
  }
  return events;
}

const char* switch_kind_name(SwitchKind kind) {
  switch (kind) {
    case SwitchKind::kSdpIntraLumaStart:
      return "SDP_INTRA_LUMA_START";
    case SwitchKind::kSdpIntraChromaStart:
      return "SDP_INTRA_CHROMA_START";
    case SwitchKind::kSdpIntraEnd:
      return "SDP_INTRA_END";
    case SwitchKind::kSdpExtendedLumaStart:
      return "SDP_EXTENDED_LUMA_START";
    case SwitchKind::kSdpExtendedChromaStart:
      return "SDP_EXTENDED_CHROMA_START";
    case SwitchKind::kSdpExtendedEnd:
      return "SDP_EXTENDED_END";
  }
  return "?";
}

}  // namespace av2enc::sdp
