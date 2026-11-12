// SDP tree validation and plane grids.

#include "codec/partition/sdp/sdp.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "codec/partition/sdp/sdp_internal.h"

namespace av2enc::sdp {
namespace {

using detail::log2_or_sentinel;
using detail::num_planes;
using detail::same_set;
using detail::set_allows;

enum class CoverKind { kLuma, kChroma };

std::string node_label(const Node& node) {
  std::ostringstream out;
  out << tree_type_name(node.tree_type) << ' '
      << partition::block_size_name(node.size) << '@' << '(' << node.mi_row
      << ',' << node.mi_col << ") depth " << node.depth;
  return out.str();
}

bool expected_codes_chroma(const FrameGeometry& geometry, TreeType tree,
                           bool has_chroma) {
  return has_chroma && geometry.has_chroma && tree != TreeType::kLumaPart;
}

bool same_block(const ChromaBlock& a, const ChromaBlock& b) {
  return a.mi_row == b.mi_row && a.mi_col == b.mi_col && a.size == b.size;
}

bool can_code_region_type(const FrameGeometry& geometry, const Tools& tools,
                          TreeType tree, BlockSize size,
                          PartitionType partition) {
  return tools.enable_extended_sdp && !tools.frame_is_intra &&
         tree == TreeType::kSharedPart && bru_mode_active(tools) &&
         size != partition::superblock_size(geometry.sb_size) &&
         is_bsize_allowed_for_extended_sdp(size, partition);
}

struct Validator {
  const FrameGeometry& geometry;
  const Tools& tools;
  int rows = 0;
  int cols = 0;
  std::vector<int> luma_cover;
  std::vector<int> chroma_cover;
  std::string error;

  Validator(const FrameGeometry& g, const Tools& t) : geometry(g), tools(t) {
    rows = partition::mi_rows(g);
    cols = partition::mi_cols(g);
    luma_cover.assign(static_cast<std::size_t>(std::max(0, rows)) *
                          static_cast<std::size_t>(std::max(0, cols)),
                      0);
    chroma_cover = luma_cover;
  }

  int& cell(std::vector<int>& grid, int r, int c) {
    return grid[static_cast<std::size_t>(r) *
                    static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(c)];
  }

  bool mark(CoverKind kind, int r, int c, BlockSize size) {
    if (!partition::is_valid_block_size(size)) {
      return fail("invalid block size in coverage");
    }
    std::vector<int>& grid =
        kind == CoverKind::kLuma ? luma_cover : chroma_cover;
    const int w4 = partition::num_4x4_blocks_wide(size);
    const int h4 = partition::num_4x4_blocks_high(size);
    for (int y = r; y < r + h4; ++y) {
      for (int x = c; x < c + w4; ++x) {
        if (y < 0 || x < 0 || y >= rows || x >= cols) continue;
        ++cell(grid, y, x);
      }
    }
    return true;
  }

  bool fail(std::string message) {
    error = std::move(message);
    return false;
  }

  bool validate(const Node& node, TreeType expected_tree,
                BlockSize expected_size, int expected_row, int expected_col,
                int expected_depth, bool expected_has_chroma,
                bool expected_offset, const ChromaBlock& inherited,
                bool expected_ext_allowed, RegionType incoming_region,
                bool chroma_known, PartitionType luma_partition);

  bool validate_children(const Node& node, BlockSize size,
                         const ChromaBlock& inherited);
};

bool Validator::validate_children(const Node& node, BlockSize size,
                                  const ChromaBlock& inherited) {
  const BlockSize subsize = sdp_subsize(node.partition, size);
  if (!partition::is_valid_block_size(subsize)) {
    return fail("node " + node_label(node) + " has an invalid subsize");
  }
  bool next_ext = node.extended_sdp_allowed &&
                  partition::block_width(subsize) > 4 &&
                  partition::block_height(subsize) > 4;
  if (node.partition == PartitionType::kHorz3 ||
      node.partition == PartitionType::kVert3) {
    const BlockSize mid = sdp_midsize(size);
    next_ext = next_ext && partition::block_width(mid) > 4 &&
               partition::block_height(mid) > 4;
  }

  partition_ext::ChildSlot slots[4];
  const int count = partition_ext::enumerate_children(
      geometry, node.mi_row, node.mi_col, size, node.partition,
      node.has_chroma, node.chroma_offset, inherited, slots);
  if (count <= 0) {
    return fail("node " + node_label(node) + " has no child slots");
  }
  std::vector<partition_ext::ChildSlot> coded;
  for (int i = 0; i < count; ++i) {
    if (partition_ext::child_is_coded(geometry, slots[i])) {
      coded.push_back(slots[i]);
    }
  }

  if (node.extended_sdp) {
    // Last child is the single CHROMA_PART block at the node's own size.
    if (node.children.size() != coded.size() + 1) {
      return fail("extended-SDP node " + node_label(node) +
                  " has the wrong child count");
    }
    for (std::size_t i = 0; i < coded.size(); ++i) {
      const Node& child = node.children[i];
      if (!validate(child, TreeType::kLumaPart, coded[i].size,
                    coded[i].mi_row, coded[i].mi_col, node.depth + 1,
                    coded[i].has_chroma, coded[i].chroma_offset,
                    coded[i].chroma, next_ext, RegionType::kIntraRegion,
                    false, PartitionType::kNone)) {
        return false;
      }
    }
    const Node& chroma = node.children.back();
    if (chroma.size != size || chroma.mi_row != node.mi_row ||
        chroma.mi_col != node.mi_col ||
        chroma.tree_type != TreeType::kChromaPart ||
        !chroma.sdp_chroma_block || chroma.partition != PartitionType::kNone ||
        !chroma.codes_chroma || !chroma.has_chroma ||
        chroma.mi_row != chroma.chroma.mi_row ||
        chroma.mi_col != chroma.chroma.mi_col ||
        chroma.chroma.size != size) {
      return fail("extended-SDP node " + node_label(node) +
                  " has a malformed chroma block");
    }
    if (chroma.depth != node.depth + 1) {
      return fail("extended-SDP chroma block has the wrong depth");
    }
    return mark(CoverKind::kChroma, chroma.chroma.mi_row,
                chroma.chroma.mi_col, chroma.chroma.size);
  }

  if (node.children.size() != coded.size()) {
    return fail("node " + node_label(node) + " has the wrong child count");
  }
  for (std::size_t i = 0; i < coded.size(); ++i) {
    const Node& child = node.children[i];
    if (!validate(child, node.tree_type, coded[i].size, coded[i].mi_row,
                  coded[i].mi_col, node.depth + 1, coded[i].has_chroma,
                  coded[i].chroma_offset, coded[i].chroma, next_ext,
                  node.region_type, false, PartitionType::kNone)) {
      return false;
    }
  }
  return true;
}

bool Validator::validate(const Node& node, TreeType expected_tree,
                         BlockSize expected_size, int expected_row,
                         int expected_col, int expected_depth,
                         bool expected_has_chroma, bool expected_offset,
                         const ChromaBlock& inherited,
                         bool expected_ext_allowed, RegionType incoming_region,
                         bool chroma_known, PartitionType luma_partition) {
  if (node.size != expected_size || node.mi_row != expected_row ||
      node.mi_col != expected_col) {
    return fail("node position/size mismatch: got " + node_label(node));
  }
  if (node.depth != expected_depth) {
    return fail("node " + node_label(node) + " has the wrong depth");
  }
  if (node.tree_type != expected_tree) {
    return fail("node " + node_label(node) + " has the wrong TreeType");
  }
  if (node.has_chroma != expected_has_chroma ||
      node.chroma_offset != expected_offset) {
    return fail("node " + node_label(node) + " has the wrong chroma state");
  }
  if (node.extended_sdp_allowed != expected_ext_allowed) {
    return fail("node " + node_label(node) +
                " has a stale extendedSdpAllowed argument");
  }
  if (!partition::is_valid_block_size(node.size)) {
    return fail("node " + node_label(node) + " has an invalid block size");
  }
  if (node.mi_row < 0 || node.mi_col < 0 ||
      node.mi_row >= partition::mi_rows(geometry) ||
      node.mi_col >= partition::mi_cols(geometry)) {
    return fail("node " + node_label(node) + " starts outside the frame");
  }

  const bool sdp_should_split =
      tools.enable_sdp && tools.frame_is_intra && geometry.has_chroma &&
      expected_tree == TreeType::kSharedPart &&
      node.size == BlockSize::k64x64;
  if (node.sdp_split != sdp_should_split) {
    return fail("node " + node_label(node) +
                (node.sdp_split ? " must not use SDP"
                                : " must split into luma/chroma trees"));
  }

  if (node.sdp_chroma_block) {
    if (expected_tree != TreeType::kChromaPart ||
        node.partition != PartitionType::kNone || !node.forced ||
        !node.codes_chroma || !node.has_chroma || !node.chroma_offset ||
        node.mi_row != node.chroma.mi_row ||
        node.mi_col != node.chroma.mi_col || node.chroma.size != node.size) {
      return fail("synthetic chroma block " + node_label(node) +
                  " is malformed");
    }
    return mark(CoverKind::kChroma, node.chroma.mi_row, node.chroma.mi_col,
                node.chroma.size);
  }

  if (node.sdp_split) {
    if (node.partition != PartitionType::kNone || !node.forced ||
        node.children.size() != 2) {
      return fail("SDP split node " + node_label(node) + " is malformed");
    }    const Node& luma = node.children[0];
    const Node& chroma = node.children[1];
    if (!validate(luma, TreeType::kLumaPart, BlockSize::k64x64, node.mi_row,
                  node.mi_col, node.depth + 1, true, false, inherited, false,
                  RegionType::kMixedRegion, false, PartitionType::kNone)) {
      return false;
    }
    if (!validate(chroma, TreeType::kChromaPart, BlockSize::k64x64,
                  node.mi_row, node.mi_col, node.depth + 1, true, false,
                  inherited, false, RegionType::kMixedRegion, luma.chroma_follows_luma,
                  luma.partition)) {
      return false;
    }
    if (luma.chroma_follows_luma !=
        chroma_follows_luma(luma.partition,
                            [&luma]() {
                              std::vector<PartitionType> out;
                              for (const Node& c : luma.children) {
                                out.push_back(c.partition);
                              }
                              return out;
                            }())) {
      return fail("SDP split node " + node_label(node) +
                  " has a stale ChromaFollowsLuma");
    }
    if (chroma.chroma_partition_known != luma.chroma_follows_luma ||
        chroma.implied_luma_partition != luma.partition) {
      return fail("SDP chroma root " + node_label(chroma) +
                  " has stale ChromaPartitionKnown/LumaPartitions");
    }
    if (chroma.cfl_allowed_in_sdp !=
        cfl_allowed_in_sdp(luma.partition, luma.chroma_follows_luma,
                           chroma.partition)) {
      return fail("SDP chroma root " + node_label(chroma) +
                  " has a stale CflAllowedInSdp");
    }
    return true;
  }

  const Boundary boundary =
      sdp_implied(geometry, node.mi_row, node.mi_col, node.size,
                        expected_tree, chroma_known, luma_partition);
  PartitionSet allowed = sdp_init_allowed(
      geometry, node.mi_row, node.mi_col, node.size, expected_tree,
      node.has_chroma, node.chroma_offset, incoming_region, tools);
  if (node.boundary.has_rows != boundary.has_rows ||
      node.boundary.has_cols != boundary.has_cols ||
      node.boundary.implied != boundary.implied ||
      node.boundary.implied_partition != boundary.implied_partition) {
    return fail("node " + node_label(node) + " has a stale boundary");
  }
  if (!same_set(node.allowed, allowed)) {
    return fail("node " + node_label(node) + " has a stale allowed set");
  }

  bool expected_forced = false;
  PartitionType expected_partition = PartitionType::kNone;
  if (boundary.implied && set_allows(allowed, boundary.implied_partition)) {
    expected_forced = true;
    expected_partition = boundary.implied_partition;
  } else if (allowed.count == 1) {
    expected_forced = true;
    for (int i = 0; i < kPartitionCount; ++i) {
      if (allowed.allowed[i]) {
        expected_partition = static_cast<PartitionType>(i);
        break;
      }
    }
  } else if (!bru_mode_active(tools)) {
    expected_forced = true;
    expected_partition = PartitionType::kNone;
  }
  if (node.forced != expected_forced) {
    return fail("node " + node_label(node) + " has a stale forced flag");
  }
  if (expected_forced && node.partition != expected_partition) {
    return fail("node " + node_label(node) +
                " does not use the spec-forced partition");
  }
  if (!expected_forced && !set_allows(allowed, node.partition)) {
    return fail("node " + node_label(node) +
                " uses a partition the spec does not allow");
  }

  const bool region_coded = can_code_region_type(
      geometry, tools, expected_tree, node.size, node.partition);
  if (node.region_type_coded != region_coded) {
    return fail("node " + node_label(node) +
                " has a stale region_type_coded flag");
  }
  RegionType expected_region = incoming_region;
  if (node.extended_sdp) {
    expected_region = RegionType::kIntraRegion;
  } else if (region_coded) {
    expected_region = RegionType::kMixedRegion;
  }
  if (node.region_type != expected_region) {
    return fail("node " + node_label(node) + " has a stale RegionType");
  }
  if (node.extended_sdp) {
    if (tools.frame_is_intra || !tools.enable_extended_sdp ||
        !region_coded) {
      return fail("node " + node_label(node) +
                  " uses extended SDP where the spec forbids it");
    }
    if (node.partition == PartitionType::kNone) {
      return fail("extended-SDP node cannot be PARTITION_NONE");
    }
  } else if (node.sdp_chroma_block) {
    return fail("synthetic chroma block outside an extended-SDP node");
  }

  const ChromaState resolved_node =
      sdp_resolve_chroma(geometry, node.mi_row, node.mi_col, node.size,
                         node.partition, node.has_chroma, node.chroma_offset,
                         inherited);
  if (!same_block(node.chroma, resolved_node.block)) {
    return fail("node " + node_label(node) + " has a stale ChromaMiSize");
  }

  if (node.partition == PartitionType::kNone) {
    if (!node.children.empty()) {
      return fail("leaf node " + node_label(node) + " has children");
    }
    const bool codes =
        expected_codes_chroma(geometry, expected_tree, node.has_chroma);
    if (node.codes_chroma != codes) {
      return fail("leaf node " + node_label(node) +
                  " has a stale HasChroma");
    }
    const ChromaState resolved =
        sdp_resolve_chroma(geometry, node.mi_row, node.mi_col, node.size,
                       node.partition, node.has_chroma, node.chroma_offset,
                       inherited);
    if (!same_block(node.chroma, resolved.block)) {
      return fail("leaf node " + node_label(node) +
                  " has a stale ChromaMiSize");
    }
    if (expected_tree != TreeType::kChromaPart) {
      if (!mark(CoverKind::kLuma, node.mi_row, node.mi_col, node.size)) {
        return false;
      }
    }
    if (codes) {
      if (!mark(CoverKind::kChroma, node.chroma.mi_row, node.chroma.mi_col,
                node.chroma.size)) {
        return false;
      }
    }
    return true;
  }

  if (node.codes_chroma) {
    return fail("internal node " + node_label(node) +
                " incorrectly claims to code chroma");
  }
  return validate_children(node, node.size, inherited);
}

bool which_plane(const FrameGeometry& geometry, const std::vector<int>& cover,
                 const char* name, int root_row, int root_col,
                 BlockSize root_size, std::string& error) {
  const int rows = partition::mi_rows(geometry);
  const int cols = partition::mi_cols(geometry);
  const int w4 = partition::num_4x4_blocks_wide(root_size);
  const int h4 = partition::num_4x4_blocks_high(root_size);
  const int end_row = std::min(root_row + h4, rows);
  const int end_col = std::min(root_col + w4, cols);
  for (int r = root_row; r < end_row; ++r) {
    for (int c = root_col; c < end_col; ++c) {
      const int count =
          cover[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(c)];
      if (count != 1) {
        std::ostringstream out;
        out << name << " plane cell (" << r << "," << c << ") covered "
            << count << " times";
        error = out.str();
        return false;
      }
    }
  }
  return true;
}

}  // namespace

bool validate_tree(const FrameGeometry& geometry, const Tools& tools,
                   const Node& root, std::string& error) {
  std::string message;
  if (!sdp_validate_geometry(geometry, message) || !validate_tools(tools, message)) {
    error = std::move(message);
    return false;
  }
  if (tools.enable_sdp && !geometry.has_chroma) {
    error = "enable_sdp requires chroma";
    return false;
  }
  Validator validator(geometry, tools);
  ChromaBlock inherited;
  inherited.mi_row = root.mi_row;
  inherited.mi_col = root.mi_col;
  inherited.size = root.size;
  if (!validator.validate(root, TreeType::kSharedPart,
                          partition::superblock_size(geometry.sb_size),
                          root.mi_row, root.mi_col, 0, true, false, inherited,
                          tools.enable_extended_sdp && !tools.frame_is_intra,
                          RegionType::kMixedRegion, false,
                          PartitionType::kNone)) {
    error = validator.error;
    return false;
  }
  if (!which_plane(geometry, validator.luma_cover, "luma", root.mi_row,
                   root.mi_col, root.size, error)) {
    return false;
  }
  if (!which_plane(geometry, validator.chroma_cover, "chroma", root.mi_row,
                   root.mi_col, root.size, error)) {
    return false;
  }
  error.clear();
  return true;
}

bool validate_frame(const FrameGeometry& geometry, const Tools& tools,
                    const FrameDecision& frame, std::string& error) {
  std::string message;
  if (!sdp_validate_geometry(geometry, message) || !validate_tools(tools, message)) {
    error = std::move(message);
    return false;
  }
  const int sb4 = partition::sb_size_mi(geometry);
  const BlockSize sb_size = partition::superblock_size(geometry.sb_size);
  std::size_t index = 0;
  for (int r = 0; r < partition::mi_rows(geometry); r += sb4) {
    for (int c = 0; c < partition::mi_cols(geometry); c += sb4) {
      if (r >= partition::mi_rows(geometry) ||
          c >= partition::mi_cols(geometry)) {
        continue;
      }
      if (index >= frame.superblocks.size()) {
        error = "frame is missing a superblock";
        return false;
      }
      const Node& sb = frame.superblocks[index++];
      if (sb.mi_row != r || sb.mi_col != c || sb.size != sb_size ||
          sb.tree_type != TreeType::kSharedPart || sb.depth != 0) {
        std::ostringstream out;
        out << "superblock layout mismatch at (" << r << "," << c << ")";
        error = out.str();
        return false;
      }
      if (!validate_tree(geometry, tools, sb, error)) {
        error = "superblock (" + std::to_string(r) + "," + std::to_string(c) +
                "): " + error;
        return false;
      }
    }
  }
  if (index != frame.superblocks.size()) {
    error = "frame has extra superblocks";
    return false;
  }
  error.clear();
  return true;
}

std::string tree_signature(const Node& node) {
  std::string out;
  out += tree_type_name(node.tree_type);
  out += '@';
  out += std::to_string(node.mi_row);
  out += ',';
  out += std::to_string(node.mi_col);
  out += ':';
  out += partition::block_size_name(node.size);
  out += ':';
  out += partition_ext::partition_name(node.partition);
  out += node.forced ? ":F" : ":S";
  if (node.sdp_split) out += ":SDP";
  if (node.extended_sdp) out += ":XREG";
  if (node.sdp_chroma_block) out += ":CB";
  if (node.chroma_follows_luma) out += ":CF";
  if (node.chroma_partition_known) out += ":CK";
  if (node.codes_chroma) out += ":UC";
  out += '(';
  for (const Node& child : node.children) {
    out += tree_signature(child);
  }
  out += ')';
  return out;
}

void build_plane_grids(const FrameGeometry& geometry, const Node& root,
                       MiSizeGrid& luma, MiSizeGrid& chroma) {
  luma.reset(geometry);
  chroma.reset(geometry);
  std::vector<LeafBlock> leaves;
  collect_leaves(root, leaves);
  for (const LeafBlock& leaf : leaves) {
    if (leaf.tree_type != TreeType::kChromaPart) {
      luma.fill_block(leaf.mi_row, leaf.mi_col, leaf.size);
    }
    if (leaf.codes_chroma) {
      chroma.fill_block(leaf.chroma.mi_row, leaf.chroma.mi_col,
                        leaf.chroma.size);
    }
  }
}

}  // namespace av2enc::sdp
