// AV2 M2 P-frame tile syntax writer: block syntax.  Partition tree
// (§5.20.3.2), intra mode info (§5.20.5), the intra leaf (palette/DIP/IBC),
// the per-unit filter syntax (§5.20.10) and the leaf dispatch write_block().
#include "pipeline/video/video_syntax_writer.h"

namespace av2enc::video::syntax {
namespace detail {

bool Writer::code_do_split(PartitionType partition, int r, int c, BlockSize size) {
  const int bsw = std::max(kMiWidthLog2[static_cast<int>(size)], 1);
  const int bsh = std::max(kMiHeightLog2[static_cast<int>(size)], 1);
  const int ctx = kPartitionSizeAdjust[static_cast<int>(size)] * 4 +
                  (left_mi(r) < bsh ? 2 : 0) +
                  (above_mi(c) < bsw ? 1 : 0);
  return sym(&cdfs_.do_split[0][ctx][0], 2,
             partition == PartitionType::kNone ? 0 : 1);
}

bool Writer::code_do_square_split(PartitionType partition, int r, int c,
                          BlockSize size) {
  const int bsw = kMiWidthLog2[static_cast<int>(size)];
  const int bsh = kMiHeightLog2[static_cast<int>(size)];
  int ctx = (left_mi(r) < bsh ? 2 : 0) +
            (above_mi(c) < bsw ? 1 : 0);
  if (size == BlockSize::k256x256) ctx += 4;
  return sym(&cdfs_.do_square_split[0][ctx][0], 2,
             partition == PartitionType::kSplit ? 1 : 0);
}

bool Writer::code_rect_type(PartitionType partition, int r, int c, BlockSize size) {
  const int bsw = std::max(kMiWidthLog2[static_cast<int>(size)], 1);
  const int bsh = std::max(kMiHeightLog2[static_cast<int>(size)], 1);
  const int ctx = kPartitionSizeAdjustRect[static_cast<int>(size)] * 4 +
                  (left_mi(r) < bsh ? 2 : 0) +
                  (above_mi(c) < bsw ? 1 : 0);
  return sym(&cdfs_.rect_type[0][ctx][0], 2,
             partition == PartitionType::kVert ? kRectVert : kRectHorz);
}

// §5.20.3.2 partition syntax for one SDP node.  The producer stores the
// reference planner's exact symbol sequence (implied / single-candidate
// partitions code nothing), including the PlaneStart and context derived
// from the luma/chroma MiSize grids, so the writer cannot disagree with the
// solver about the tree.
bool Writer::code_sdp_partition(const BlockSyntax& node, std::string& error) {
  for (int i = 0; i < static_cast<int>(node.sdp_symbol_count); ++i) {
    const BlockSyntax::SdpSymbol& symbol = node.sdp_symbols[i];
    switch (static_cast<sdpp::SymbolKind>(symbol.kind)) {
      case sdpp::SymbolKind::kDoSplit:
        if (!sym(&cdfs_.do_split[symbol.plane][symbol.context][0], 2,
                 symbol.value)) {
          return false;
        }
        break;
      case sdpp::SymbolKind::kRectType:
        if (!sym(&cdfs_.rect_type[symbol.plane][symbol.context][0], 2,
                 symbol.value)) {
          return false;
        }
        break;
      case sdpp::SymbolKind::kDoSquareSplit:
      case sdpp::SymbolKind::kDoExtPartition:
      case sdpp::SymbolKind::kDoUneven4wayPartition:
      case sdpp::SymbolKind::kUneven4wayType:
      case sdpp::SymbolKind::kRegionType:
        error = "SDP frame codes " +
                std::string(sdpp::symbol_name(
                    static_cast<sdpp::SymbolKind>(symbol.kind))) +
                " at a 64x64 superblock stream; only rect-only sub-trees "
                "are wired";
        return false;
    }
  }
  return true;
}

// SDP writer (--video-sdp on the intra keyframe).  The 64x64 split node
// codes no partition syntax and carries the luma root followed by the chroma
// root.  Each sub-tree node codes its own PlaneStart-selected partition
// symbols; leaf blocks gate the tree-type-dependent mode/residual syntax in
// write_block().
bool Writer::write_node_sdp(std::size_t index, std::size_t* next,
                    std::string& error) {
  const BlockSyntax& node = blocks_[index];
  const auto size = static_cast<BlockSize>(node.bsize);
  const auto partition = static_cast<PartitionType>(node.partition);
  const std::size_t subtree_end = index + 1 + node.subtree;
  if (subtree_end > blocks_.size()) {
    error = "node " + std::to_string(index) + " has a subtree past the end";
    return false;
  }
  if ((node.flags & kFlagSdpSplit) != 0) {
    if (node.subtree == 0) {
      error = "SDP split node " + std::to_string(index) +
              " has no luma/chroma children";
      return false;
    }
    std::size_t cursor = index + 1;
    while (cursor < subtree_end) {
      std::size_t after = 0;
      if (!write_node_sdp(cursor, &after, error)) return false;
      if (after > subtree_end) {
        error = "SDP node " + std::to_string(index) +
                " has an inconsistent subtree count";
        return false;
      }
      cursor = after;
    }
    if (cursor != subtree_end) {
      error = "SDP node " + std::to_string(index) +
              " has an inconsistent subtree count";
      return false;
    }
    *next = subtree_end;
    return true;
  }
  if (!partition::is_valid_block_size(size)) {
    error = "node " + std::to_string(index) + " has an invalid block size";
    return false;
  }
  if (!inside_mi(node.mi_row, node.mi_col)) {
    error = "node " + std::to_string(index) + " is outside the frame";
    return false;
  }
  if (!code_sdp_partition(node, error)) return false;

  if (partition == PartitionType::kNone) {
    if (node.subtree != 0) {
      error = "leaf node " + std::to_string(index) +
              " has a non-zero subtree";
      return false;
    }
    if (!write_block(node, error)) return false;
    *next = index + 1;
    return true;
  }

  // Descend into the children (preorder).  The tree solver produced one
  // child per in-frame slot, so the subtree count is authoritative.
  std::size_t cursor = index + 1;
  while (cursor < subtree_end) {
    std::size_t after = 0;
    if (!write_node_sdp(cursor, &after, error)) return false;
    if (after > subtree_end) {
      error = "node " + std::to_string(index) +
              " has an inconsistent subtree count";
      return false;
    }
    cursor = after;
  }
  if (cursor != subtree_end) {
    error = "node " + std::to_string(index) +
            " has an inconsistent subtree count";
    return false;
  }
  *next = subtree_end;
  return true;
}

// Emit the syntax for the node at `index` (and, for non-leaves, its whole
// subtree); `next` receives the index one past the subtree.
bool Writer::write_node(std::size_t index, std::size_t* next, std::string& error) {
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
  if (config_.enable_ext_partitions) {
    return write_node_ext(index, next, error);
  }
#endif
  if (plan_.frame_sdp) {
    return write_node_sdp(index, next, error);
  }
  // R19 diagnostic: every coded leaf runs through this wrapper so the report
  // can compare the model's prediction with the symbols the writer charged.
  const auto write_leaf = [&](const BlockSyntax& leaf) -> bool {
    if (!rate_interval::detail::stats_enabled()) {
      return write_block(leaf, error);
    }
    double before[kCostTagCount];
    for (int tag = 0; tag < kCostTagCount; ++tag) {
      before[tag] = cost_meter_.bits[tag];
    }
    const std::uint64_t prefix_before = enc_.prefix_bits;
    if (!write_block(leaf, error)) return false;
    record_rate_interval_leaf(leaf, before, prefix_before);
    return true;
  };
  const BlockSyntax& node = blocks_[index];
  const auto size = static_cast<BlockSize>(node.bsize);
  const auto partition = static_cast<PartitionType>(node.partition);
  const std::size_t subtree_end = index + 1 + node.subtree;
  if (subtree_end > blocks_.size()) {
    error = "node " + std::to_string(index) + " has a subtree past the end";
    return false;
  }
  if (!partition::is_valid_block_size(size)) {
    error = "node " + std::to_string(index) + " has an invalid block size";
    return false;
  }
  const int r = node.mi_row;
  const int c = node.mi_col;
  if (r < 0 || c < 0 || r >= mi_rows_ || c >= mi_cols_) {
    error = "node " + std::to_string(index) + " is outside the frame";
    return false;
  }
  const bool forced = (node.flags & kFlagForced) != 0;
  const auto allowed = [&node](PartitionType p) {
    return (node.allowed_mask & (1u << static_cast<unsigned>(p))) != 0;
  };

  if (trace_) {
    std::cerr << "m1 node r=" << r << " c=" << c
              << " size=" << partition::block_size_name(size)
              << " part=" << partition::partition_name(partition)
              << " forced=" << (forced ? 1 : 0) << " allowed="
              << allowed(PartitionType::kNone) << allowed(PartitionType::kSplit)
              << allowed(PartitionType::kHorz)
              << allowed(PartitionType::kVert) << "\n";
  }

  if (!forced) {
    if (allowed(PartitionType::kNone)) {
      if (!code_do_split(partition, r, c, size)) {
        error = error_;
        return false;
      }
      if (partition == PartitionType::kNone) {
        if (node.subtree != 0) {
          error = "leaf node " + std::to_string(index) +
                  " has a non-zero subtree";
          return false;
        }
        if (!write_leaf(node)) return false;
        *next = index + 1;
        return true;
      }
    }
    if (partition == PartitionType::kNone) {
      // No do_split symbol was coded (NONE is the implied partition), but
      // the tree solver only reaches this for a forced node.
      if (node.subtree != 0) {
        error = "leaf node " + std::to_string(index) +
                " has a non-zero subtree";
        return false;
      }
      if (!write_leaf(node)) return false;
      *next = index + 1;
      return true;
    }
    // §5.20.3.2: do_square_split follows do_split and precedes rect_type.
    // It is present only when PARTITION_SPLIT is allowed, which can happen
    // only at a 128x128 (or 256x256) node; the 64x64 path never reaches it.
    if (allowed(PartitionType::kSplit)) {
      if (!code_do_square_split(partition, r, c, size)) {
        error = error_;
        return false;
      }
    }
    const partition::RectTypeImplied implied =
        partition::rect_type_implied_by_bsize(size);
    bool need_rect = partition != PartitionType::kSplit && !implied.implied;
    if (need_rect) {
      const bool allow_h = allowed(PartitionType::kHorz);
      const bool allow_v = allowed(PartitionType::kVert);
      if (!allow_h || !allow_v) need_rect = false;
    }
    if (need_rect) {
      if (!code_rect_type(partition, r, c, size)) {
        error = error_;
        return false;
      }
    }
  }
  if (partition == PartitionType::kNone) {
    if (node.subtree != 0) {
      error = "leaf node " + std::to_string(index) +
              " has a non-zero subtree";
      return false;
    }
    if (!write_leaf(node)) return false;
    *next = index + 1;
    return true;
  }

  // Descend into the children (preorder) and validate them against the
  // Partition_Subsize / boundary rules.
  SlotInfo slots[4];
  const int slot_count = enumerate_slots(r, c, size, partition, mi_rows_,
                                         mi_cols_, slots);
  if (slot_count == 0) {
    error = "node " + std::to_string(index) + " has no in-frame children";
    return false;
  }
  std::size_t cursor = index + 1;
  int slot_cursor = 0;
  while (cursor < subtree_end) {
    const BlockSyntax& child = blocks_[cursor];
    while (slot_cursor < slot_count &&
           (slots[slot_cursor].mi_row != static_cast<int>(child.mi_row) ||
            slots[slot_cursor].mi_col != static_cast<int>(child.mi_col))) {
      ++slot_cursor;
    }
    if (slot_cursor >= slot_count) {
      error = "node " + std::to_string(cursor) +
              " does not match a partition subsize slot of node " +
              std::to_string(index);
      return false;
    }
    if (child.bsize != static_cast<std::uint8_t>(slots[slot_cursor].size)) {
      error = "node " + std::to_string(cursor) +
              " has the wrong block size for its partition";
      return false;
    }
    ++slot_cursor;
    std::size_t after = 0;
    if (!write_node(cursor, &after, error)) return false;
    if (after > subtree_end) {
      error = "node " + std::to_string(index) +
              " has an inconsistent subtree count";
      return false;
    }
    cursor = after;
  }
  if (cursor != subtree_end) {
    error = "node " + std::to_string(index) +
            " has an inconsistent subtree count";
    return false;
  }
  *next = subtree_end;
  return true;
}

#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
// ---------------------------------------------------------------------
// Extended-partition writer (--video-ext-partitions).  The symbol sequence
// comes from partition_stream/partition_ext so the implication rules for
// do_ext_partition / do_uneven_4way_partition and the raw L(1) type bit are
// the reference module's, not re-derived here.
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// §8.2.2 extended-partition CDF image bridge
//
// `ext_cdfs_` is the still-stream planner's per-tile copy of the five
// partition rows.  It is the model the extended writer actually adapts, so
// it must be the same state the §8.2.4 / §7.4.4 carry image stores.  These
// helpers keep the two in lockstep: one copy in after the seed is imported,
// one copy out before the tile image is exported.  With the tool off the
// rows are never coded, so both directions are a no-op.
// ---------------------------------------------------------------------
void Writer::import_ext_cdfs_from_image() {
  std::memcpy(ext_cdfs_.ext.do_split, cdfs_.do_split,
              sizeof(cdfs_.do_split));
  std::memcpy(ext_cdfs_.ext.do_square_split, cdfs_.do_square_split,
              sizeof(cdfs_.do_square_split));
  std::memcpy(ext_cdfs_.ext.rect_type, cdfs_.rect_type,
              sizeof(cdfs_.rect_type));
  std::memcpy(ext_cdfs_.ext.do_ext_partition, cdfs_.do_ext_partition,
              sizeof(cdfs_.do_ext_partition));
  std::memcpy(ext_cdfs_.ext.do_uneven_4way_partition,
              cdfs_.do_uneven_4way_partition,
              sizeof(cdfs_.do_uneven_4way_partition));
}

void Writer::export_ext_cdfs_to_image() {
  std::memcpy(cdfs_.do_split, ext_cdfs_.ext.do_split,
              sizeof(cdfs_.do_split));
  std::memcpy(cdfs_.do_square_split, ext_cdfs_.ext.do_square_split,
              sizeof(cdfs_.do_square_split));
  std::memcpy(cdfs_.rect_type, ext_cdfs_.ext.rect_type,
              sizeof(cdfs_.rect_type));
  std::memcpy(cdfs_.do_ext_partition, ext_cdfs_.ext.do_ext_partition,
              sizeof(cdfs_.do_ext_partition));
  std::memcpy(cdfs_.do_uneven_4way_partition,
              ext_cdfs_.ext.do_uneven_4way_partition,
              sizeof(cdfs_.do_uneven_4way_partition));
}

bool Writer::write_node_ext(std::size_t index, std::size_t* next,
                    std::string& error) {
  const BlockSyntax& node = blocks_[index];
  const auto size = static_cast<partition::BlockSize>(node.bsize);
  const auto partition = static_cast<pe::PartitionType>(node.partition);
  const std::size_t subtree_end = index + 1 + node.subtree;
  if (subtree_end > blocks_.size()) {
    error = "node " + std::to_string(index) + " has a subtree past the end";
    return false;
  }
  if (!partition::is_valid_block_size(size)) {
    error = "node " + std::to_string(index) + " has an invalid block size";
    return false;
  }
  const int r = node.mi_row;
  const int c = node.mi_col;
  if (!inside_mi(r, c)) {
    error = "node " + std::to_string(index) + " is outside the frame";
    return false;
  }
  if (partition != pe::PartitionType::kNone && node.subtree == 0) {
    error = "node " + std::to_string(index) +
            " has no children for a split";
    return false;
  }

  ps::StillNode sn;
  sn.mi_row = node.mi_row;
  sn.mi_col = node.mi_col;
  sn.bsize = node.bsize;
  sn.partition = node.partition;
  sn.flags = node.flags;
  sn.subtree = node.subtree;
  sn.allowed_mask = node.allowed_mask;
  // The video leaf coder owns its chroma block; collect_node_ext() rejects a
  // tree that needs shared ChromaMi ownership.
  sn.has_chroma = true;
  sn.chroma_offset = false;
  sn.chroma = pe::ChromaBlock{r, c, size};

  const ps::NeighbourContext nc = ps::neighbours_from_arrays(
      std::span<const int>(left_mi_), std::span<const int>(above_mi_), r, c,
      size, inside_mi(r, c - 1), inside_mi(r - 1, c));
  const ps::NodePlan plan = ps::plan_node(geometry_, tools_, sn, nc);
  if (!plan.ok) {
    error = "node " + std::to_string(index) + ": " + plan.error;
    return false;
  }
  if (!ps::emit_symbols(plan.symbols, ext_cdfs_, enc_, nullptr, error)) {
    return false;
  }

  if (partition == pe::PartitionType::kNone) {
    if (node.subtree != 0) {
      error = "leaf node " + std::to_string(index) +
              " has a non-zero subtree";
      return false;
    }
    if (!write_block(node, error)) return false;
    *next = index + 1;
    return true;
  }

  pe::ChildSlot raw[4];
  const int raw_count = pe::enumerate_children(
      geometry_, r, c, size, partition, /*has_chroma=*/true,
      /*chroma_offset=*/false, pe::ChromaBlock{r, c, size}, raw);
  pe::ChildSlot slots[4];
  int slot_count = 0;
  for (int i = 0; i < raw_count; ++i) {
    if (pe::child_is_coded(geometry_, raw[i])) slots[slot_count++] = raw[i];
  }
  if (slot_count == 0) {
    error = "node " + std::to_string(index) + " has no in-frame children";
    return false;
  }
  std::size_t cursor = index + 1;
  int slot_cursor = 0;
  while (cursor < subtree_end) {
    const BlockSyntax& child = blocks_[cursor];
    while (slot_cursor < slot_count &&
           (slots[slot_cursor].mi_row != static_cast<int>(child.mi_row) ||
            slots[slot_cursor].mi_col != static_cast<int>(child.mi_col))) {
      ++slot_cursor;
    }
    if (slot_cursor >= slot_count) {
      error = "node " + std::to_string(cursor) +
              " does not match a partition child slot of node " +
              std::to_string(index);
      return false;
    }
    if (child.bsize != static_cast<std::uint8_t>(slots[slot_cursor].size)) {
      error = "node " + std::to_string(cursor) +
              " has the wrong block size for its partition";
      return false;
    }
    ++slot_cursor;
    std::size_t after = 0;
    if (!write_node_ext(cursor, &after, error)) return false;
    if (after > subtree_end) {
      error = "node " + std::to_string(index) +
              " has an inconsistent subtree count";
      return false;
    }
    cursor = after;
  }
  if (cursor != subtree_end) {
    error = "node " + std::to_string(index) +
            " has an inconsistent subtree count";
    return false;
  }
  *next = subtree_end;
  return true;
}
#endif  // AV2ENC_VIDEO_EXT_PARTITIONS

int Writer::joint_context() const {
  if (std::getenv("AV2ENC_STILL_YMODE_CTX0") != nullptr) return 0;
  if (std::getenv("AV2ENC_STILL_YMODE_CTX2") != nullptr) return 2;
  // Uses the block-start MiRow/MiCol of the current node.
  return (get_joint_mode(cur_r_, cur_c_, cur_w4_, cur_h4_, 0) >= 5 ? 1 : 0) +
         (get_joint_mode(cur_r_, cur_c_, cur_w4_, cur_h4_, 1) >= 5 ? 1 : 0);
}

// §5.20.5.5 get_intra_y_mode_set: the signalled mode index is resolved
// through a neighbour-dependent list.  Directional candidates are seeded with
// the IntraJointModes of the below-left (dir=0) and above-right (dir=1)
// neighbours, then expanded around them for blocks larger than 8x8/64 pels,
// and finally filled from Default_Mode_List_Y.  The encoder must invert that
// list; writing the static-list position makes dependent blocks decode a
// different direction.
int Writer::y_mode_index_for(int joint, bool sub8_leaf) const {
  if (joint < 5) {
    return joint;
  }
  const int target = joint - 5;  // directional index 0..55
  int mode_index = 5;
  bool selected[56] = {};
  int dir_modes[2] = {0, 0};
  int count = 0;
  const int w = cur_w4_ * 4;
  const int h = cur_h4_ * 4;
  // §5.20.5.5 gates the neighbour-derived list on `MiSize >= BLOCK_8X8`,
  // not on the pixel dimensions: rectangular sizes such as 32x4/4x32 are
  // >= BLOCK_8X8 in the size enum and therefore use the dynamic list.
  if (!sub8_leaf) {
    for (int dir = 0; dir < 2; ++dir) {
      int mode = get_joint_mode(cur_r_, cur_c_, cur_w4_, cur_h4_, dir);
      if (mode >= 5) {
        mode -= 5;
        if (count == 0 || mode != dir_modes[0]) {
          if (mode == target) return mode_index;
          ++mode_index;
          selected[mode] = true;
          dir_modes[count++] = mode;
        }
      }
    }
    if (w * h > 64) {
      for (int i = 1; i <= 4; ++i) {
        for (int j = 0; j < count; ++j) {
          for (int sgn = -1; sgn <= 1; sgn += 2) {
            int mode = dir_modes[j] + i * sgn;
            if (mode < 0) {
              mode += 56;
            } else if (mode >= 56) {
              mode -= 56;
            }
            if (!selected[mode]) {
              if (mode == target) return mode_index;
              ++mode_index;
              selected[mode] = true;
            }
          }
        }
      }
    }
  }
  for (int i = 0; i < 56; ++i) {
    const int mode = kDefaultModeListY[i];
    if (!selected[mode]) {
      if (mode == target) return mode_index;
      ++mode_index;
    }
  }
  return mode_index;
}

bool Writer::code_y_mode(const BlockSyntax& node) {
  const YModeCode code =
      y_mode_code(node.y_mode, node.y_angle_delta);
  // Directional modes use the dynamic set; non-directional indices are the
  // reordered mode itself and are unaffected by neighbours.
  const bool sub8_leaf =
      static_cast<BlockSize>(node.bsize) < BlockSize::k8x8;
  const int mode_index =
      code.joint < 5 ? code.mode_index
                     : y_mode_index_for(code.joint, sub8_leaf);
  const int ctx = joint_context();
  if (mode_index < kIntraModesCount) {
    if (!sym(&cdfs_.y_mode_set[0], 4, 0)) return false;
    if (mode_index <= 6) {
      return sym(&cdfs_.y_mode_index[ctx][0], 8, mode_index);
    }
    if (!sym(&cdfs_.y_mode_index[ctx][0], 8, 7)) return false;
    return sym(&cdfs_.y_mode_offset[ctx][0], 6, mode_index - 7);
  }
  const int set = 1 + (mode_index - kIntraModesCount) / 16;
  const int second = (mode_index - kIntraModesCount) % 16;
  if (!sym(&cdfs_.y_mode_set[0], 4, set)) return false;
  return write_literal(4, static_cast<std::uint32_t>(second));
}

// is_cfl context: neighbouring chroma CfL flags (§8.3.x).
int Writer::cfl_context() const {
  const int cr = cur_chroma_r_;
  const int cc = cur_chroma_c_;
  const int cmi_cols = chroma_mi_cols();
  int ctx = 0;
  if (cr > 0 && cc < cmi_cols &&
      uvcfl_[static_cast<std::size_t>(cr - 1) *
                 static_cast<std::size_t>(cmi_cols) +
             static_cast<std::size_t>(cc)] != 0) {
    ++ctx;
  }
  if (cc > 0 && cr < chroma_mi_rows() &&
      uvcfl_[static_cast<std::size_t>(cr) *
                 static_cast<std::size_t>(cmi_cols) +
             static_cast<std::size_t>(cc - 1)] != 0) {
    ++ctx;
  }
  return ctx;
}

bool Writer::code_uv_mode(const BlockSyntax& node, int cw, int ch) {
  const bool is_cfl = (node.flags & kFlagIsCfl) != 0;
  const int uv_mode = node.uv_mode;
  const int y_mode = node.y_mode;
  if (is_cfl) {
    // is_cfl derives its context from neighbouring chroma CfL flags.
    if (!sym(&cdfs_.is_cfl[cfl_context()][0], 2, 1)) return false;
    if (config_.enable_mhccp) {
      // §5.20.7.32 with MHCCP: cfl_mhccp, cfl_index, cfl_mh_dir and the
      // explicit alpha symbols all live in the gated adapter.
      const bool mhccp_allowed = cfl::mhccp::allowed(
          cw, ch, config_.enable_mhccp, /*is_chroma_part=*/false,
          /*frame_is_intra=*/true, /*cfl_allowed_in_sdp=*/false,
          /*lossless=*/false);
      if (node.cfl_index == 2 && !mhccp_allowed) {
        error_ = "CFL_MULTI signalled for a block where MHCCP is not allowed";
        return false;
      }
      cfl::mhccp::CflSyntax syntax;
      syntax.cfl_mhccp = node.cfl_index == 2 ? 1 : 0;
      syntax.cfl_index = node.cfl_index;
      syntax.cfl_mh_dir = node.cfl_mh_dir;
      syntax.cfl_alpha_signs = node.cfl_alpha_signs;
      syntax.cfl_alpha_u = node.cfl_alpha_u;
      syntax.cfl_alpha_v = node.cfl_alpha_v;
      std::string alpha_error;
      // §8.3: TileCflMhDirCdf[Size_Group[MiSize]] uses the coding block.
      const int mh_size_group =
          node.bsize < 29 ? cfl::mhccp::kSizeGroup[node.bsize] : -1;
      if (mh_size_group < 0) {
        error_ = "invalid block size for the cfl_mh_dir context";
        return false;
      }
      if (!cfl::mhccp::write_cfl_alphas(
              enc_, cdfs_.mh_cdfs, syntax, config_.enable_cfl_intra,
              mhccp_allowed, mh_size_group, &alpha_error)) {
        error_ = "read_cfl_alphas: " + alpha_error;
        return false;
      }
      return true;
    }
    // CFL_DERIVED_ALPHA = 1 (the non-MHCCP path).
    return sym(&cdfs_.cfl_index[0], 2, node.cfl_index);
  }
  const int ctx = y_mode_is_directional(y_mode) ? 1 : 0;
  // is_cfl = 0 is only coded when CfL is allowed.
  if (config_.enable_cfl_intra) {
    if (!sym(&cdfs_.is_cfl[cfl_context()][0], 2, 0)) return false;
  }
  const int idx = uv_mode_index(uv_mode, y_mode);
  if (idx <= 6) {
    return sym(&cdfs_.uv_mode[ctx][0], 8, idx);
  }
  if (!sym(&cdfs_.uv_mode[ctx][0], 8, 7)) return false;
  return write_literal(3, static_cast<std::uint32_t>(idx - 7));
}

// `is_bv_valid` only needs the plane geometry, so the writer builds a view
// with null sample pointers; it never copies pixels.
ibc::FrameView Writer::ibc_frame_view() const {
  ibc::FrameView view;
  view.y = ibc::PlaneView{nullptr, mi_cols_ * 4, mi_cols_ * 4,
                          mi_rows_ * 4};
  view.u = ibc::PlaneView{nullptr, chroma_mi_cols() * 4,
                          chroma_mi_cols() * 4, chroma_mi_rows() * 4};
  view.v = view.u;
  view.has_chroma = has_chroma_;
  view.sub_x = sub_x_;
  view.sub_y = sub_y_;
  return view;
}

// §5.20.5.3 use_intrabc (+ read_skip when set).  Present for every leaf when
// allow_intrabc is on; only eligible block sizes carry the symbol.
bool Writer::write_keyframe_mode_info(const BlockSyntax& node, bool is_ibc,
                              std::string& error) {
  if (!plan_.allow_intrabc) {
    if (is_ibc) {
      error = "IBC leaf supplied but the frame disables intrabc";
      return false;
    }
    return true;
  }
  if (!ibcs::intrabc_available(plan_.allow_intrabc, node.bsize)) {
    if (is_ibc) {
      error = "IBC leaf is not signalable at this block size";
      return false;
    }
    return true;
  }
  if (is_ibc) {
    // read_intrabc_info() forces DC luma/chroma and clears palette/CfL.
    if (node.y_mode != kDcPred || (node.flags & kFlagIsCfl) != 0) {
      error = "IBC leaf carries intra/palette mode state";
      return false;
    }
  }
  const int ctx = std::min(
      ibcs::ibc_symbol_context(ibc_grid_, mi_cols_, mi_rows_, node.mi_row,
                               node.mi_col, node.bsize, plan_.sb_size),
      2);
  if (!ibcs::write_use_intrabc(enc_, ibc_cdfs_, ctx, is_ibc, error)) {
    return false;
  }
  if (!is_ibc) return true;
  const int skip_ctx = std::min(
      ibcs::skip_symbol_context(ibc_skip_grid_, mi_cols_, mi_rows_,
                                node.mi_row, node.mi_col, node.bsize,
                                plan_.sb_size),
      2);
  return ibcs::write_skip_flag(enc_, ibc_cdfs_, skip_ctx,
                               node.ibc_skip != 0, error);
}

// §5.20.5.4 intrabc_info() (intrabc_mode/DRL/assign_mv) and, when the
// sequence enables it, morph_pred.  The BV stack is rebuilt from the same
// neighbour map and ring buffer the decoder maintains.
bool Writer::write_ibc_body(const BlockSyntax& node, std::string& error) {
  const ibc::FrameView frame = ibc_frame_view();
  const ibcs::BvStack stack = ibcs::build_bv_stack(
      frame, ibc_config_, ibc_buffer_, node.mi_row, node.mi_col, node.bsize,
      ibc_neighbours_);
  ibcs::BvSyntax syntax;
  syntax.mode = node.ibc_mode == 1 ? ibcs::IntrabcMode::kNearMv
                                   : ibcs::IntrabcMode::kNewMv;
  syntax.ref_mv_idx = node.ibc_ref_mv_idx;
  syntax.intrabc_precision_quarter = false;
  const ibc::BlockVector target{node.ibc_bv_row, node.ibc_bv_col};
  if (!ibcs::write_intrabc_info(enc_, ibc_cdfs_, ibc_content_, stack, syntax,
                                target, error)) {
    return false;
  }
  if (plan_.morph_pred_present) {
    const int w4 = std::max(1, cur_w4_);
    const int h4 = std::max(1, cur_h4_);
    const int ctx = morph::morph_symbol_context(
        morph_grid_.data(), mi_cols_, mi_rows_, node.mi_row, node.mi_col, w4,
        h4, plan_.sb_size);
    if (!morph::write_morph_pred(enc_, morph_cdfs_, ctx,
                                 node.morph_pred != 0, error)) {
      return false;
    }
  } else if (node.morph_pred != 0) {
    error = "morph_pred set but the frame does not signal it";
    return false;
  }
  if (node.ibc_skip != 0) {
    // §5.20.5.1: an IBC skip block codes no residual, but the decoder still
    // resets the level/DC contexts over the block.
    reset_coeff_contexts(node.mi_row, node.mi_col, cur_w4_, cur_h4_);
  }
  return true;
}

// §5.20.8 palette_mode_info() for a keyframe leaf.  When screen-content tools
// are enabled the symbol is present for every DC block that satisfies the
// §5.20.5.5 predicate, so a non-palette leaf still codes has_palette_y = 0.
bool Writer::write_palette(const BlockSyntax& node, int w, int h,
                   std::string& error) {
  if (!plan_.allow_screen_content_tools) return true;
  if ((node.flags & kFlagIsIbc) != 0) return true;
  if (!palette_writer_.has_value()) {
    error = "palette writer is not initialised: " + palette_reset_error_;
    return false;
  }
  const int onscreen_w = std::min(w, mi_cols_ * 4 - node.mi_col * 4);
  const int onscreen_h = std::min(h, mi_rows_ * 4 - node.mi_row * 4);
  const palette::PaletteResult* result = nullptr;
  if (node.palette_index != BlockSyntax::kNoPalette) {
    if (node.palette_index >= plan_.palettes.size()) {
      error = "palette index is outside the frame palette list";
      return false;
    }
    result = &plan_.palettes[node.palette_index];
    if (!result->has_palette) {
      error = "palette leaf carries an empty palette result";
      return false;
    }
  }
  pals::PaletteModeInfo info;
  std::string palette_error;
  if (!palette_writer_->write(enc_, palette_cdfs_, node.mi_row, node.mi_col,
                              node.bsize, w, h, onscreen_w, onscreen_h,
                              result, node.y_mode, &info, palette_error)) {
    error = "palette syntax: " + palette_error;
    return false;
  }
  const bool want_palette = result != nullptr;
  if (info.has_palette_y != want_palette) {
    error = "palette syntax did not match the supplied decision";
    return false;
  }
  return true;
}

// §5.20.7.9 dip_mode_info(): after the intra Y/UV mode and palette syntax,
// before the residual.  Only luma DC blocks that satisfy the size predicate
// can signal it.  The context is the number of §5.20.4.1 neighbours whose
// UseDip flag is 1 (DIP_CTXS = 3), so the writer keeps the same per-MI grid
// the decoder's decode_block() maintains.
bool Writer::write_dip(const BlockSyntax& node, int w, int h, std::string& error) {
  const bool palette = node.palette_index != BlockSyntax::kNoPalette;
  const bool present = config_.enable_dip && node.y_mode == kDcPred &&
                       !palette && w > 4 && h > 4 && w * h >= 128;
  if (!present) {
    if (node.use_dip != 0) {
      error = "use_dip is not signalable for this block";
      return false;
    }
    return true;
  }
  if (usedip_.empty()) {
    error = "DIP use-dip grid is not initialised";
    return false;
  }
  const int r = node.mi_row;
  const int c = node.mi_col;
  const mrl_stream::MrlBlockSyntax mrl_block{r, c, w >> 2, h >> 2};
  mrl_stream::NeighborPos npos[2];
  const int nnum = mrl_stream::collect_npos(
      mrl_block, mi_rows_, mi_cols_, partition::sb_size_mi(geometry_), npos);
  int use_dip_ctx = 0;
  for (int i = 0; i < nnum; ++i) {
    if (usedip_[static_cast<std::size_t>(npos[i].row) *
                    static_cast<std::size_t>(mi_cols_) +
                static_cast<std::size_t>(npos[i].col)] != 0) {
      ++use_dip_ctx;
    }
  }
  ::av2enc::dip::SyntaxContext context;
  context.enable_dip = true;
  context.y_mode = node.y_mode;
  context.palette_size_y = 0;
  context.tree_is_chroma_part = false;
  context.w = w;
  context.h = h;
  context.use_dip_ctx = use_dip_ctx;
  ::av2enc::dip::DipModeInfo info;
  info.use_dip = node.use_dip != 0;
  info.dip_transpose = node.dip_transpose != 0;
  info.dip_mode = node.dip_mode;
  std::string dip_error;
  if (!::av2enc::dip::write_dip_mode_info(context, info, enc_, cdfs_.dip_cdfs,
                                          &dip_error)) {
    error = "dip_mode_info: " + dip_error;
    return false;
  }
  return true;
}

void Writer::update_usedip(const BlockSyntax& node, int w4, int h4) {
  if (!config_.enable_dip || usedip_.empty()) return;
  const std::uint8_t value = node.use_dip != 0 ? 1 : 0;
  for (int yy = 0; yy < h4; ++yy) {
    const int rr = node.mi_row + yy;
    if (rr >= mi_rows_) break;
    for (int xx = 0; xx < w4; ++xx) {
      const int cc = node.mi_col + xx;
      if (cc >= mi_cols_) break;
      usedip_[static_cast<std::size_t>(rr) *
                  static_cast<std::size_t>(mi_cols_) +
              static_cast<std::size_t>(cc)] = value;
    }
  }
}

bool Writer::write_intra_leaf(const BlockSyntax& node, int w, int h, int cw, int ch,
                      std::string& error) {
  const int tree_type = node.tree_type;
  const bool chroma_part = tree_type == 2;
  const bool chroma_info = (node.flags & kFlagHasChromaInfo) != 0;
  // §5.20.7.26: a monochrome sequence has no chroma residual at all, so the
  // Subsampled_Size lookup must not run for it.
  const bool node_has_chroma =
      fmt::has_chroma(chroma_) && (!chroma_info || node.has_chroma != 0);
  const int chroma_mi_row =
      chroma_info ? static_cast<int>(node.chroma_mi_row) : node.mi_row;
  const int chroma_mi_col =
      chroma_info ? static_cast<int>(node.chroma_mi_col) : node.mi_col;
  // Validate the declared transform sizes against TX_MODE_LARGEST geometry
  // before emitting anything.  Under TX_MODE_SELECT the luma size is carried
  // by the per-part list (validated against the signalled partition below);
  // chroma keeps the TX_MODE_LARGEST geometry.  A CHROMA_PART leaf has no
  // luma transform; a non-owning luma leaf has no chroma transforms.
  const int coded_w[kPlaneCount] = {w, cw, cw};
  const int coded_h[kPlaneCount] = {h, ch, ch};
  const int plane_count =
      node_has_chroma ? kPlaneCount : (chroma_part ? 0 : 1);
  for (int plane = plan_.tx_mode_select ? 1 : 0; plane < plane_count;
       ++plane) {
    if ((plane == 0 && chroma_part) || (plane > 0 && !node_has_chroma)) {
      continue;
    }
    const auto params_opt = txv::make_tx_params(
        static_cast<txv::TxSize>(node.tx_size[plane]),
        txv::TxType::kDctDct);
    if (!params_opt.has_value() ||
        params_opt->coded_w != coded_w[plane] ||
        params_opt->coded_h != coded_h[plane]) {
      error = "tx_size on plane " + std::to_string(plane) + " (" +
              std::to_string(node.tx_size[plane]) + ") does not match the " +
              "TX_MODE_LARGEST block size " +
              std::to_string(coded_w[plane]) + "x" +
              std::to_string(coded_h[plane]) + " tree=" +
              std::to_string(node.tree_type) + " bsize=" +
              std::to_string(node.bsize) + " chroma_bsize=" +
              std::to_string(node.chroma_bsize) + " has_chroma=" +
              std::to_string(node.has_chroma);
      return false;
    }
  }

  const std::size_t levels_end = levels_.size();
  std::size_t level_base = node.levels_offset;
  if (!chroma_part) {
    if (!code_y_mode(node)) {
      error = error_;
      return false;
    }
    // §5.20.5.5: after the Y mode/angle delta, every luma coding block with
    // Block_Width/Height <= FSC_MAX codes fsc_mode when enable_idtx_intra is
    // 1.  This encoder never uses FSC (enable_fsc = 0), so it codes 0 and all
    // neighbours stay 0 (CDF context [0][group]).
    if (config_.enable_idtx_intra && w <= 32 && h <= 32) {
      const int group = kFscBsizeGroups[node.bsize];
      if (!sym(&cdfs_.fsc_mode[0][group][0], 2, 0)) return false;
    }
    if (config_.enable_mrls) {
      const mrl_stream::MrlBlockSyntax mrl_block{
          node.mi_row, node.mi_col, w >> 2, h >> 2};
      const mrl_stream::MrlDecision mrl_decision{
          static_cast<int>(node.mrl_index), node.mrl_sec_index != 0};
      const mrl_stream::MrlSyntaxStatus status = mrl_stream::write_mrl_syntax(
          enc_, cdfs_.mrl, mrl_context_, mrl_block, node.y_mode, mrl_decision,
          &error_);
      if (status != mrl_stream::MrlSyntaxStatus::kOk) {
        error = "mrl syntax: " +
                std::string(mrl_stream::mrl_syntax_status_name(status));
        if (!error_.empty()) error += ": " + error_;
        return false;
      }
    } else if (node.mrl_index != 0 || node.mrl_sec_index != 0) {
      error = "MRL decisions supplied but the sequence header disables MRL";
      return false;
    }
  }
  if (node_has_chroma && !code_uv_mode(node, cw, ch)) {
    error = error_;
    return false;
  }
  if (!chroma_part && !write_palette(node, w, h, error)) return false;
  if (!chroma_part && !write_dip(node, w, h, error)) return false;
  if (!chroma_part) {
    if (!code_luma_residual(node, w, h, levels_end, false, level_base,
                            error)) {
      return false;
    }
    level_base += static_cast<std::size_t>(luma_level_count(node));
  }
  if (!node_has_chroma) {
    if (error.empty() && !error_.empty()) error = error_;
    return true;
  }
  const TxView u_view{node.tx_size[1], node.tx_type[1],
                      static_cast<int>(node.eob[1]), chroma_mi_row,
                      chroma_mi_col};
  if (!code_tx_view(node, 1, u_view, level_base, cw, ch, levels_end, false,
                    error)) {
    return false;
  }
  level_base += node.eob[1];
  const TxView v_view{node.tx_size[2], node.tx_type[2],
                      static_cast<int>(node.eob[2]), chroma_mi_row,
                      chroma_mi_col};
  if (!code_tx_view(node, 2, v_view, level_base, cw, ch, levels_end, false,
                    error)) {
    return false;
  }
  if (error.empty() && !error_.empty()) error = error_;
  return true;
}

// §5.20.10.1 read_cdef().  A no-op for CdefStrengths == 1, for a 64x64
// whose origin was already coded, and for a skipped inter block when
// cdef_on_skip_txfm is off (the decoder returns before reading cdef_idx).
// §5.20.10.4 read_lr(): once per superblock, before read_partition().  The
// unit grid is at least one superblock, so only a superblock origin starts
// a restoration unit.
bool Writer::write_lr_superblock(const BlockSyntax& node, std::string& error) {
  if (filters_ == nullptr || !filters_->lr.enabled) return true;
  if (filters_->lr.pc_wiener) {
    // Luma PC-Wiener units first (plane order 0,1,2 in read_lr()).
    if (!pc_ready_) return error.empty() ? true : false;
    namespace pcw = ::av2enc::filters::pc_wiener_stream;
    const post::VideoLrPlane& cfg = filters_->lr.plane[0];
    if (cfg.enabled) {
      const int unit_size = pc_state_.unit_size[0];
      const int origin_x = node.mi_col * 4;
      const int origin_y = node.mi_row * 4;
      if (unit_size > 0 && origin_x % unit_size == 0 &&
          origin_y % unit_size == 0) {
        const int unit_row = origin_y / unit_size;
        const int unit_col = origin_x / unit_size;
        if (unit_row < cfg.unit_rows && unit_col < cfg.unit_cols) {
          const std::size_t index =
              static_cast<std::size_t>(unit_row) *
                  static_cast<std::size_t>(cfg.unit_cols) +
              static_cast<std::size_t>(unit_col);
          if (index >= cfg.unit_type.size()) {
            error = "PC-Wiener unit decision vector is too small";
            return false;
          }
          pcw::UnitDecision decision;
          decision.type = cfg.unit_type[index];
          if (!pcw::WriteUnit(enc_, pc_state_, 0, unit_row, unit_col,
                              decision, &error)) {
            return false;
          }
        }
      }
    }
    if (lr_chroma_ready_) {
      std::vector<filters_stream::LrUnitDecision> decisions;
      if (!filters_stream::lr_build_decisions(
              lr_state_, node.mi_row, node.mi_col, /*w4=*/16, /*h4=*/16,
              /*tile_mi_row_start=*/0, /*tile_mi_col_start=*/0, &decisions,
              &error)) {
        return false;
      }
      if (!filters_stream::lr_write_block(
              enc_, lr_state_, node.mi_row, node.mi_col, /*w4=*/16,
              /*h4=*/16, /*tile_mi_row_start=*/0, /*tile_mi_col_start=*/0,
              decisions, &error)) {
        return false;
      }
    }
    return true;
  }
  std::vector<filters_stream::LrUnitDecision> decisions;
  if (!filters_stream::lr_build_decisions(lr_state_, node.mi_row,
                                          node.mi_col, /*w4=*/16, /*h4=*/16,
                                          /*tile_mi_row_start=*/0,
                                          /*tile_mi_col_start=*/0,
                                          &decisions, &error)) {
    return false;
  }
  if (!filters_stream::lr_write_block(
          enc_, lr_state_, node.mi_row, node.mi_col, /*w4=*/16, /*h4=*/16,
          /*tile_mi_row_start=*/0, /*tile_mi_col_start=*/0, decisions,
          &error)) {
    return false;
  }
  return true;
}

bool Writer::write_cdef_index(const BlockSyntax& node, std::string& error) {
  if (filters_ == nullptr || !filters_->cdef.enabled) return true;
  if (plan_.inter_frame && node.is_inter && node.skip_flag != 0 &&
      !filters_->cdef.on_skip_txfm) {
    return true;
  }
  const int br = (node.mi_row + tile_mi_row_) >> 4;
  const int bc = (node.mi_col + tile_mi_col_) >> 4;
  if (br < 0 || bc < 0 || br >= filters_->cdef.n64_rows ||
      bc >= filters_->cdef.n64_cols) {
    error = "cdef_idx grid does not cover a coding block";
    return false;
  }
  const int idx = filters_->cdef.index[static_cast<std::size_t>(br) *
                                          static_cast<std::size_t>(
                                              filters_->cdef.n64_cols) +
                                      static_cast<std::size_t>(bc)];
  const std::uint64_t prefix_before =
      cost_meter_.active ? enc_.prefix_bits : 0;
  if (!filters_stream::cdef_write_block(
          enc_, cdef_state_, /*tile_mi_col_start=*/0,
          /*tile_mi_row_start=*/0, node.mi_row, node.mi_col, idx, &error)) {
    return false;
  }
  if (cost_meter_.active) {
    cost_meter_.add_prefix(kCostCdef, prefix_before, enc_.prefix_bits);
  }
  return true;
}

// §5.20.10.2 read_ccso().  Called for every leaf; the shared writer is a
// no-op unless the leaf origin is a CCSO unit origin.
bool Writer::write_ccso_index(const BlockSyntax& node, std::string& error) {
  if (filters_ == nullptr || !filters_->ccso.enabled) return true;
  int blk[3] = {0, 0, 0};
  const int shift = filters_->ccso.ccso_luma_size_log2 - 2;
  for (int plane = 0; plane < 3; ++plane) {
    const post::VideoCcsoPlane& src = filters_->ccso.plane[plane];
    if (!src.enabled || shift < 0) continue;
    const int row = (node.mi_row + tile_mi_row_) >> shift;
    const int col = (node.mi_col + tile_mi_col_) >> shift;
    if (row < 0 || col < 0 || row >= src.blk_rows || col >= src.blk_cols) {
      continue;
    }
    blk[plane] = src.blks[static_cast<std::size_t>(row) *
                              static_cast<std::size_t>(src.blk_cols) +
                          static_cast<std::size_t>(col)];
  }
  const std::uint64_t prefix_before =
      cost_meter_.active ? enc_.prefix_bits : 0;
  if (!filters_stream::ccso_write_block(enc_, ccso_state_, node.mi_row,
                                        node.mi_col, blk, &error)) {
    return false;
  }
  if (cost_meter_.active) {
    cost_meter_.add_prefix(kCostCcso, prefix_before, enc_.prefix_bits);
  }
  return true;
}

// R19 diagnostic (AV2ENC_RATE_INTERVAL_STATS=1).  Re-runs the interval model
// on the leaf the writer just emitted and compares it with the information
// cost the live CDF actually charged (`AV2ENC_SYMBOL_COST` meter).  Predicted
// and actual cover the same symbol groups:
//   * inter leaf: is_inter + skip + inter mode (mode), residual syntax;
//   * intra leaf: residual only (the neighbour-derived y_mode inversion lives
//     here in the writer and is not part of the model).
// The exact (R05) column is the same symbol sequence through a real §8.2
// encoder, reported for the error/cost trade-off.
void Writer::record_rate_interval_leaf(const BlockSyntax& leaf,
                                       const double* meter_before,
                                       std::uint64_t prefix_before) {
  using namespace rate_interval;
  if ((leaf.flags & kFlagIsIbc) != 0 || leaf.palette_index != BlockSyntax::kNoPalette) {
    return;
  }
  const auto delta = [&](int tag) {
    return cost_meter_.bits[tag] - meter_before[tag];
  };
  LeafSample sample;
  sample.bsize = leaf.bsize;
  sample.is_inter = leaf.is_inter != 0 ? 1 : 0;
  sample.inter_frame = plan_.inter_frame ? 1 : 0;
  sample.skip = leaf.skip_flag != 0 ? 1 : 0;
  sample.actual_mode =
      delta(kCostIsInter) + delta(kCostSkip) + delta(kCostInterMode);
  sample.actual_intra_mode = delta(kCostIntra);
  sample.actual_residual = delta(kCostResidual);
  (void)prefix_before;

  LeafRateInput in;
  LeafLevelStorage storage;
  const bool adapt = config_.disable_cdf_update == 0;
  if (!build_from_syntax(leaf, levels_, plan_.tx_mode_select, adapt,
                         static_cast<int>(config_.base_q_idx), &in, &storage)) {
    return;
  }
  // Historical heuristic on the same leaf: the error the model must reduce.
  double legacy = 0.0;
  for (int i = 0; i < in.part_count; ++i) {
    const LeafPart& part = in.parts[i];
    if (part.eob == 0) {
      legacy += 1.0;
      continue;
    }
    int nonzero = 0;
    for (int c = 0; c < part.eob; ++c) {
      if (part.levels[static_cast<std::size_t>(c)] != 0) ++nonzero;
    }
    legacy += 8.0 + (part.plane == 0 && in.is_inter ? 4.0 : 0.0) +
              2.5 * static_cast<double>(nonzero) +
              0.5 * static_cast<double>(part.eob);
  }
  sample.legacy_residual = legacy;

  using clock = std::chrono::steady_clock;
  const clock::time_point t0 = clock::now();
  const LeafRate rate = estimate(in, rate_store(), Mode::kInterval, false);
  const clock::time_point t1 = clock::now();
  const LeafRate exact = estimate(in, rate_store(), Mode::kExact, false);
  const clock::time_point t2 = clock::now();
  sample.predicted_mode = rate.mode_bits;
  sample.predicted_residual = rate.residual_bits;
  sample.exact_mode = exact.mode_bits;
  // In exact mode `residual_bits` is still the interval number (the shipped
  // model); the trial encoder reports its unmaterialized prefix bits here.
  sample.exact_residual = exact.exact_prefix_bits;
  sample.exact_valid = !exact.failed;
  sample.ns_interval =
      std::chrono::duration<double, std::micro>(t1 - t0).count();
  sample.ns_exact =
      std::chrono::duration<double, std::micro>(t2 - t1).count();
  static const bool trace = std::getenv("AV2ENC_RATE_INTERVAL_TRACE") != nullptr;
  if (trace) {
    std::fprintf(stderr,
                 "ri_leaf mi=%d,%d bsize=%d inter=%d skip=%d parts=%d eob0=%d "
                 "eob1=%d eob2=%d pred=%.1f/%.1f exact=%.1f/%.1f "
                 "actual=%.1f/%.1f intra=%.1f\n",
                 static_cast<int>(leaf.mi_row), static_cast<int>(leaf.mi_col),
                 static_cast<int>(leaf.bsize), sample.is_inter, sample.skip,
                 in.part_count, static_cast<int>(leaf.eob[0]),
                 static_cast<int>(leaf.eob[1]), static_cast<int>(leaf.eob[2]),
                 sample.predicted_mode, sample.predicted_residual,
                 sample.exact_mode, sample.exact_residual, sample.actual_mode,
                 sample.actual_residual, sample.actual_intra_mode);
  }
  record(sample);
  stats().exact_runs.fetch_add(sample.exact_valid ? 1 : 0,
                               std::memory_order_relaxed);
}

bool Writer::write_block(const BlockSyntax& node, std::string& error) {
  if (decision_dump_active_) dump_decision(node);
#if defined(AV2ENC_VIDEO_DELTAQ)
  // Mirrors src/codec/entropy/deltaq/video/video_deltaq.cpp::
  // write_superblock_delta_q(): verify the planned tile state before the
  // symbol, emit §5.20.5.11 (or no symbol for a full-SB skip), then verify
  // the achieved qindex.  §5.20.5.11 codes nothing when
  // MiSize == SbSize && skip_flag, so such a superblock can only realize a
  // zero step; the planner must have produced one.
  const auto write_delta_step = [this](const rc::DeltaQPlanStep& step,
                                       const BlockSyntax& leaf,
                                       bool skip_flag,
                                       std::string& step_error) -> bool {
    std::string dq_error;
    if (!av2enc::deltaq::check_plan_step(dqp_.tile_qp, step, &dq_error)) {
      step_error = "delta-Q plan: " + dq_error;
      return false;
    }
    if (!video::deltaq::write_delta_qindex(dqp_, enc_, leaf, skip_flag,
                                           step.reduced_delta, &dq_error)) {
      step_error = "delta_qindex: " + dq_error;
      return false;
    }
    if (dqp_.tile_qp.current_q_index != step.qindex_after) {
      step_error =
          "delta_qindex: superblock (" + std::to_string(step.sb_y) + "," +
          std::to_string(step.sb_x) + ") did not reach the planned qindex " +
          std::to_string(step.qindex_after) + " (got " +
          std::to_string(dqp_.tile_qp.current_q_index) +
          "); a full-SB skip cannot code a delta";
      return false;
    }
    return true;
  };
#endif
  const int r = node.mi_row;
  const int c = node.mi_col;
  const auto size = static_cast<BlockSize>(node.bsize);
  const int w = partition::block_width(size);
  const int h = partition::block_height(size);
  const int w4 = w >> 2;
  const int h4 = h >> 2;
  cur_r_ = r;
  cur_c_ = c;
  cur_w4_ = w4;
  cur_h4_ = h4;
  // §6.19.3.1 TreeType.  A CHROMA_PART leaf codes no luma mode, TX size,
  // CDEF/CCSO/delta-q or luma residual; a LUMA_PART leaf codes no chroma.
  const int tree_type = node.tree_type;
  const bool chroma_part = tree_type == 2;
  const bool chroma_info = (node.flags & kFlagHasChromaInfo) != 0;
  // §5.20.7.26: a monochrome sequence has no chroma residual at all, so the
  // Subsampled_Size lookup must not run for it.
  const bool node_has_chroma =
      fmt::has_chroma(chroma_) && (!chroma_info || node.has_chroma != 0);
  if (chroma_part && !node_has_chroma) {
    // §5.20.3.1: a chroma-tree leaf with HasChroma == 0 codes nothing.
    return true;
  }
  // §5.20.4.1 decode_block(): update_ibc_buffers() runs at the start of the
  // block, before mode_info() builds the BV stack.
  const bool is_ibc_block = (node.flags & kFlagIsIbc) != 0;
  if (plan_.allow_intrabc) {
    ibcs::begin_ibc_block(ibc_buffer_, r, c);
  }

  int cw = 0;
  int ch = 0;
  const auto chroma_size = chroma_info
                               ? fmt::chroma_residual_size(
                                     partition::block_width(
                                         static_cast<BlockSize>(
                                             node.chroma_bsize)),
                                     partition::block_height(
                                         static_cast<BlockSize>(
                                             node.chroma_bsize)),
                                     chroma_)
                               : fmt::chroma_residual_size(w, h, chroma_);
  if (node_has_chroma) {      // §5.20.7.26 get_plane_residual_size() via the Subsampled_Size table.
    // `fmt::chroma_residual_size()` models 4:2:0/4:2:2/4:4:4; the 4:2:0
    // values are the historical `block_width/2` pair.  For an SDP chroma
    // leaf the owned ChromaMi block, not the luma leaf, selects the size.
    if (!chroma_size.has_value()) {
      error = "invalid chroma residual size";
      return false;
    }
    cw = fmt::block_width(*chroma_size);
    ch = fmt::block_height(*chroma_size);
  }
  cur_chroma_r_ =
      chroma_info ? (node.chroma_mi_row >> 1) : (r >> sub_y_);
  cur_chroma_c_ =
      chroma_info ? (node.chroma_mi_col >> 1) : (c >> sub_x_);

  // R09 report: per-leaf range-coder bit snapshots around the syntax groups
  // the model prices.  `act_is_inter_skip` = is_inter + skip_flag,
  // `act_mode_info` = read_ref_frames + mode/DRL/MVD, `act_residual` = the
  // transform/coefficient syntax.  CDEF/CCSO/delta-Q bits sit between the
  // groups and belong to neither.
  const bool rate_report =
      rate_tags_active_ && node.rate_diag.valid != 0 && plan_.inter_frame;
  std::uint32_t rate_bit0 = 0;
  std::uint32_t rate_bit2 = 0;
  std::uint32_t rate_bit3 = 0;
  std::uint32_t rate_bit4 = 0;
  std::uint32_t rate_bit5 = 0;
  if (rate_report) {
    clear_rate_tags();
    rate_bit0 = enc_.prefix_bits;
  }

#if defined(AV2ENC_VIDEO_DELTAQ)
  // §5.20.5.7/.8 / §5.20.7.2 / §5.20.5.11 ordering state for this leaf.
  // `first_leaf` is set at each superblock root: only that leaf calls
  // read_delta_qindex(), and read_delta_qindex() clears ReadDeltas whether
  // or not it coded a symbol.
  const bool dqp = dqp_active_;
  const bool seg_pre_skip = dqp && dqp_.segmentation.seg_id_pre_skip;
  const int segment_id = dqp ? static_cast<int>(node.segment_id) : 0;
  // §5.20.7.3/.6/.10: SEG_LVL_SKIP/GLOBALMV force YMode = GLOBALMV and
  // SkipSegFrame; SKIP also infers skip_flag = 1.  The pinned dav2d still
  // reads is_inter for a GLOBALMV block, so only the ref-frames/mode/skip
  // bits are inferred (docs/negative-results.md).
  const bool seg_skip =
      dqp && plan_.inter_frame && dqp_.segmentation.skip_active(segment_id);
  const bool seg_globalmv =
      dqp && plan_.inter_frame &&
      dqp_.segmentation.globalmv_active(segment_id);
  const bool seg_forced_inter = seg_skip || seg_globalmv;
  if (trace_ && dqp) {
    std::cerr << "wseg r=" << node.mi_row << " c=" << node.mi_col
              << " size=" << static_cast<int>(node.bsize)
              << " seg=" << segment_id
              << " inter=" << static_cast<int>(node.is_inter)
              << " skip=" << static_cast<int>(node.skip_flag)
              << " gmv=" << (seg_globalmv ? 1 : 0)
              << " skp=" << (seg_skip ? 1 : 0)
              << " preskip=" << (seg_pre_skip ? 1 : 0)
              << " temporal="
              << (dqp_.segmentation.temporal_update ? 1 : 0) << "\n";
  }
  const rc::DeltaQPlanStep* delta_step = sb_first_leaf_ ? sb_step_ : nullptr;
  sb_first_leaf_ = false;
  std::string dq_error;
#endif

  if (plan_.inter_frame) {
#if defined(AV2ENC_VIDEO_DELTAQ)
    if (dqp && seg_pre_skip &&
        !video::deltaq::write_inter_segment_id_pre(
            dqp_, enc_, node, segment_id, /*skip_flag=*/false, &dq_error)) {
      error = "inter_segment_id(preSkip): " + dq_error;
      return false;
    }
#endif
#if defined(AV2ENC_VIDEO_DELTAQ)
    if (seg_globalmv && node.is_inter == 0) {
      error = "SEG_LVL_GLOBALMV leaf is not an inter block";
      return false;
    }
#endif
    set_rate_tag(rate_hybrid::report::kTagIsInter);
    if (!code_is_inter(node.is_inter != 0)) {
      set_rate_tag(-1);
      error = error_;
      return false;
    }
    if (node.is_inter) {
#if defined(AV2ENC_VIDEO_DELTAQ)
      if (seg_skip && node.skip_flag == 0) {
        error = "SEG_LVL_SKIP leaf must have skip_flag = 1";
        return false;
      }
      if (!seg_skip)
#endif
      set_rate_tag(rate_hybrid::report::kTagSkip);
      if (!code_skip_flag(node.skip_flag != 0)) {
        set_rate_tag(-1);
        error = error_;
        return false;
      }
      set_rate_tag(-1);
      if (rate_report) rate_bit2 = enc_.prefix_bits;
#if defined(AV2ENC_VIDEO_DELTAQ)
      if (dqp && !seg_pre_skip &&
          !video::deltaq::write_inter_segment_id_post(
              dqp_, enc_, node, segment_id, node.skip_flag != 0,
              &dq_error)) {
        error = "inter_segment_id: " + dq_error;
        return false;
      }
#endif
      if (!write_cdef_index(node, error)) return false;
      if (!write_ccso_index(node, error)) return false;
#if defined(AV2ENC_VIDEO_DELTAQ)
      if (dqp && delta_step != nullptr &&
          !write_delta_step(*delta_step, node, node.skip_flag != 0, error)) {
        return false;
      }
#endif
      if (rate_report) rate_bit3 = enc_.prefix_bits;
#if defined(AV2ENC_VIDEO_DELTAQ)
      // §5.20.7.6/.10: the forced GLOBALMV path codes ref_frames, mode,
      // BAWP, motion mode and DRL as inferred values (no bits).
      if (seg_forced_inter) {
        if (node.compound || node.inter_mode != kInterGlobalMv ||
            node.ref_index != 0 || node.mv_y != 0 || node.mv_x != 0 ||
            node.use_bawp != 0 || node.motion_mode != 0) {
          error = "segmentation-forced GLOBALMV leaf carries mode syntax";
          return false;
        }
      } else
#endif
      if (!write_inter_mode_info(node, error)) return false;
      if (rate_report) {
        rate_bit4 = enc_.prefix_bits;
        rate_bit5 = enc_.prefix_bits;
      }
      if (node.skip_flag) {
        reset_coeff_contexts(r, c, w4, h4);
      } else {
        if (rate_report) set_rate_tag(rate_hybrid::report::kTagResid);
        if (!write_inter_residual(node, w, h, cw, ch, error)) {
          set_rate_tag(-1);
          return false;
        }
        if (rate_report) {
          set_rate_tag(-1);
          rate_bit5 = enc_.prefix_bits;
        }
      }
    } else {
      if (rate_report) rate_bit2 = enc_.prefix_bits;
#if defined(AV2ENC_VIDEO_DELTAQ)
      if (dqp && !seg_pre_skip &&
          !video::deltaq::write_inter_segment_id_post(
              dqp_, enc_, node, segment_id, /*skip_flag=*/false, &dq_error)) {
        error = "inter_segment_id: " + dq_error;
        return false;
      }
#endif
      if (!write_cdef_index(node, error)) return false;
      if (!write_ccso_index(node, error)) return false;
#if defined(AV2ENC_VIDEO_DELTAQ)
      if (dqp && delta_step != nullptr &&
          !write_delta_step(*delta_step, node, /*skip_flag=*/false, error)) {
        return false;
      }
#endif
      if (!write_intra_leaf(node, w, h, cw, ch, error)) return false;
    }
  } else {
    // Intra keyframe: intra_frame_mode_info() does not code is_inter.
    const bool is_ibc = (node.flags & kFlagIsIbc) != 0;
    if (node.is_inter) {
      error = "keyframe block marked inter";
      return false;
    }
#if defined(AV2ENC_VIDEO_DELTAQ)
    // §5.20.5.3: intra_segment_id() precedes the keyframe mode info.
    if (dqp && !chroma_part &&
        !video::deltaq::write_intra_segment_id(
            dqp_, enc_, node, segment_id, /*skip_flag=*/false,
            &dq_error)) {
      error = "intra_segment_id: " + dq_error;
      return false;
    }
#endif
    if (!chroma_part &&
        !write_keyframe_mode_info(node, is_ibc, error)) {
      return false;
    }
    // §5.20.10.1 read_cdef(): after the skip group, before the modes.  An
    // IBC skip block returns before cdef_idx unless CDEF filters skip blocks.
    // §5.20.10.2 read_ccso() is a separate reader with no such early return,
    // so the CCSO unit symbol is always coded here.  A CHROMA_PART leaf
    // codes no CDEF/CCSO.
    const bool cdef_skip =
        chroma_part ||
        (is_ibc && node.ibc_skip != 0 && filters_ != nullptr &&
         filters_->cdef.enabled && !filters_->cdef.on_skip_txfm);
    if (!cdef_skip) {
      if (!write_cdef_index(node, error)) return false;
    }
    if (!write_ccso_index(node, error)) return false;
#if defined(AV2ENC_VIDEO_DELTAQ)
    if (dqp && !chroma_part && delta_step != nullptr &&
        !write_delta_step(*delta_step, node, /*skip_flag=*/false, error)) {
      return false;
    }
#endif
    if (chroma_part) {
      if (!write_intra_leaf(node, w, h, cw, ch, error)) return false;
    } else if (is_ibc) {
      if (!write_ibc_body(node, error)) return false;
    } else if (!write_intra_leaf(node, w, h, cw, ch, error)) {
      return false;
    }
  }

#if defined(AV2ENC_VIDEO_DELTAQ)
  // §5.20.5.7: record SegmentIds for the following leaves' prediction.
  if (dqp &&
      !video::deltaq::commit_segment_block(dqp_, node, segment_id,
                                           &dq_error)) {
    error = "commit_segment_block: " + dq_error;
    return false;
  }
#endif

  // Update partition context.  A CHROMA_PART leaf updates only the chroma
  // plane grid; a LUMA_PART/shared leaf updates the luma grid and, when it
  // owns chroma, the chroma grid.
  if (!chroma_part) {
    for (int yy = 0; yy < h4; ++yy) {
      left_mi(r + yy) = kMiHeightLog2[static_cast<int>(size)];
    }
    for (int xx = 0; xx < w4; ++xx) {
      above_mi(c + xx) = kMiWidthLog2[static_cast<int>(size)];
    }
  }

  if (!chroma_part) update_mode_grids(node);
  if (!chroma_part) update_usedip(node, w4, h4);

  if (!chroma_part) {
    if (!node.is_inter) {
      const YModeCode ycode = y_mode_code(node.y_mode, node.y_angle_delta);
      for (int yy = 0; yy < h4; ++yy) {
        const int rr = r + yy;
        if (rr >= mi_rows_) break;
        for (int xx = 0; xx < w4; ++xx) {
          const int cc = c + xx;
          if (cc >= mi_cols_) break;
          intra_joint(rr, cc) = ycode.joint;
        }
      }
    } else {
      for (int yy = 0; yy < h4; ++yy) {
        const int rr = r + yy;
        if (rr >= mi_rows_) break;
        for (int xx = 0; xx < w4; ++xx) {
          const int cc = c + xx;
          if (cc >= mi_cols_) break;
          intra_joint(rr, cc) = 0;  // inter_block_mode_info: DC_PRED
        }
      }
    }
  }

  if (!node_has_chroma) return true;
  const int cr = cur_chroma_r_;
  const int cc0 = cur_chroma_c_;
  const int cw4 = cw / 4;
  const int ch4 = ch / 4;
  const int chroma_rows = chroma_mi_rows();
  const int chroma_cols = chroma_mi_cols();
  const bool is_cfl = !node.is_inter && (node.flags & kFlagIsCfl) != 0;
  for (int yy = 0; yy < ch4; ++yy) {
    const int rr = cr + yy;
    if (rr >= chroma_rows) break;
    for (int xx = 0; xx < cw4; ++xx) {
      const int cc = cc0 + xx;
      if (cc >= chroma_cols) break;
      uvcfl(rr, cc) = is_cfl ? 1 : 0;
    }
  }
  if (plan_.allow_intrabc) {
    // §5.20.4.1: MorphPreds = use_intrabc && morph_pred and UseIntrabc over
    // every 4x4 of the decoded block.  The BV neighbour map/ring must be
    // advanced for every leaf, IBC or not.
    const std::uint8_t ibc_value = is_ibc_block ? 1 : 0;
    const std::uint8_t skip_value =
        (is_ibc_block && node.ibc_skip != 0) ? 1 : 0;
    const std::uint8_t morph_value =
        (is_ibc_block && node.morph_pred != 0) ? 1 : 0;
    for (int yy = 0; yy < h4; ++yy) {
      const int rr = r + yy;
      if (rr >= mi_rows_) break;
      for (int xx = 0; xx < w4; ++xx) {
        const int cc = c + xx;
        if (cc >= mi_cols_) break;
        const std::size_t idx = cell_index(rr, cc);
        ibc_grid_[idx] = ibc_value;
        ibc_skip_grid_[idx] = skip_value;
        if (plan_.morph_pred_present) morph_grid_[idx] = morph_value;
      }
    }
    ibcs::finish_ibc_block(
        ibc_buffer_, ibc_neighbours_, r, c, node.bsize, is_ibc_block,
        ibc::BlockVector{node.ibc_bv_row, node.ibc_bv_col});
  }
  if (rate_report && node.rate_diag.fallback == 0) {
    rate_hybrid::report::Sample sample;
    sample.is_inter = 1;
    sample.inter_mode = node.inter_mode;
    sample.bsize = node.bsize;
    sample.skip = node.skip_flag;
    sample.pred_syntax_old = node.rate_diag.syntax_old;
    sample.pred_syntax_new = node.rate_diag.syntax_new;
    sample.pred_coeff_old = node.rate_diag.coeff_old;
    sample.pred_coeff_new = node.rate_diag.coeff_new;
    for (int i = 0; i < 6; ++i) sample.pred_term[i] = node.rate_diag.term[i];
    sample.pred_bawp = node.rate_diag.bawp;
    sample.eob = node.rate_diag.eob;
    sample.nonzero = node.rate_diag.nonzero;
    sample.sum_abs = node.rate_diag.sum_abs;
    sample.act_is_inter_skip =
        static_cast<float>(rate_bit2 - rate_bit0);
    sample.act_mode_info = static_cast<float>(rate_bit4 - rate_bit3);
    sample.act_residual = static_cast<float>(rate_bit5 - rate_bit4);
    for (int tag = 0; tag < rate_hybrid::report::kTagCount; ++tag) {
      sample.exact[tag] = static_cast<float>(rate_tag_bits_[tag]);
    }
    rate_hybrid::report::add(sample);
  }
  return true;
}

}  // namespace detail
}  // namespace av2enc::video::syntax
