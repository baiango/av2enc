// 256x256-superblock CPU reference.  See sb256.h for the spec clauses.

#include "sb256.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <utility>

#include "encoder/m0_cdf_tables.h"

namespace av2enc::sb256 {
namespace {

constexpr std::uint8_t kInvalid = 255;

// §9.2: Mi_Width_Log2 / Mi_Height_Log2, indexed by enum_block_size.
constexpr std::uint8_t kMiWidthLog2[kBlockSizeCount] = {
    0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 0,
    2, 1, 3, 2, 4, 0, 3, 1, 4,
};
constexpr std::uint8_t kMiHeightLog2[kBlockSizeCount] = {
    0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 2,
    0, 3, 1, 4, 2, 3, 0, 4, 1,
};

// §9.2 Partition_Size_Adjust / Partition_Size_Adjust_Rect_Type.
constexpr std::uint8_t kPartitionSizeAdjust[kBlockSizeCount] = {
    0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0,
};
constexpr std::uint8_t kPartitionSizeAdjustRect[kBlockSizeCount] = {
    0, 0, 0, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 13, 14, 13, 14, 0, 0, 0, 0,
};

// §9.2 Partition_Subsize rows for the base partitions.
constexpr std::uint8_t kHorzSubsize[kBlockSizeCount] = {
    kInvalid, 0,  kInvalid, 2,  3,  20, 5,  6,  22, 8,
    9,        24, 11,       12, kInvalid, 14, 15, kInvalid, 17, 1,
    kInvalid, 4,  26,       7,  28, kInvalid, kInvalid, kInvalid, kInvalid,
};
constexpr std::uint8_t kVertSubsize[kBlockSizeCount] = {
    kInvalid, kInvalid, 0,  1,  19, 3,  4,  21, 6,  7,
    23,       9,        10, kInvalid, 12, 13, kInvalid, 15, 16, kInvalid,
    2,        25,       5,  27, 8,  kInvalid, kInvalid, kInvalid, kInvalid,
};
// The only valid SPLIT entries are BLOCK_256X256 -> BLOCK_128X128 and
// BLOCK_128X128 -> BLOCK_64X64.
constexpr std::uint8_t kSplitSubsize[kBlockSizeCount] = {
    kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
    kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
    kInvalid, 12, kInvalid, kInvalid, 15, kInvalid, kInvalid, kInvalid,
    kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
};

// Subsampled_Size[subsize][subx=1][suby=1] for 4:2:0 chroma.
constexpr std::uint8_t kChroma420Subsize[kBlockSizeCount] = {
    0,  0,  0,  0,  1,  2,  3,  4,  5,  6,
    7,  8,  9,  10, 11, 12, 13, 14, 15, 1,
    2,  19, 20, 21, 22, 19, 20, 25, 26,
};

constexpr const char* kBlockNames[kBlockSizeCount] = {
    "4x4",   "4x8",   "8x4",   "8x8",   "8x16",  "16x8",  "16x16",
    "16x32", "32x16", "32x32", "32x64", "64x32", "64x64", "64x128",
    "128x64", "128x128", "128x256", "256x128", "256x256", "4x16",
    "16x4",  "8x32",  "32x8",  "16x64", "64x16",  "4x32",  "32x4",
    "8x64",  "64x8",
};

constexpr int kSentinelLog2 = 6;  // BLOCK_256X256

int log2_or_sentinel(BlockSize size, bool width) {
  if (!is_valid_block_size(size)) {
    return kSentinelLog2;
  }
  return width ? mi_width_log2(size) : mi_height_log2(size);
}

bool fail(std::string& error, std::string message) {
  error = std::move(message);
  return false;
}

struct ChildSlot {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  bool has_chroma = false;
  bool chroma_offset = false;
  ChromaBlock chroma;
};

int enumerate_children(const Geometry& geometry, int mi_row, int mi_col,
                       BlockSize size, PartitionType partition,
                       bool has_chroma, bool chroma_offset,
                       const ChromaBlock& inherited, ChildSlot slots[4]) {
  if (partition == PartitionType::kNone) {
    return 0;
  }
  const BlockSize subsize = partition_subsize(partition, size);
  if (!is_valid_block_size(subsize)) {
    return 0;
  }
  const int w4 = num_4x4_blocks_wide(size);
  const int h4 = num_4x4_blocks_high(size);
  const int half_w = w4 >> 1;
  const int half_h = h4 >> 1;

  bool next_offset = chroma_offset;
  if (has_chroma && !next_offset) {
    next_offset = is_chroma_offset_for_partition(geometry, partition, size);
  }

  switch (partition) {
    case PartitionType::kSplit:
      // decode_partition() passes chromaOffset = 0 to every SPLIT child.
      slots[0] = {mi_row, mi_col, subsize, has_chroma, false, inherited};
      slots[1] = {mi_row, mi_col + half_w, subsize, has_chroma, false,
                  inherited};
      slots[2] = {mi_row + half_h, mi_col, subsize, has_chroma, false,
                  inherited};
      slots[3] = {mi_row + half_h, mi_col + half_w, subsize, has_chroma, false,
                  inherited};
      return 4;
    case PartitionType::kHorz:
      slots[0] = {mi_row, mi_col, subsize, has_chroma && !next_offset,
                  next_offset, inherited};
      slots[1] = {mi_row + half_h, mi_col, subsize, has_chroma, next_offset,
                  inherited};
      return 2;
    case PartitionType::kVert:
      slots[0] = {mi_row, mi_col, subsize, has_chroma && !next_offset,
                  next_offset, inherited};
      slots[1] = {mi_row, mi_col + half_w, subsize, has_chroma, next_offset,
                  inherited};
      return 2;
    case PartitionType::kNone:
      break;
  }
  return 0;
}

bool child_is_coded(const Geometry& geometry, const ChildSlot& slot) {
  return slot.mi_row < mi_rows(geometry) && slot.mi_col < mi_cols(geometry);
}

// Deterministic reference policy:
//   * full 256x256 roots SPLIT to 128x128, full 128x128 roots SPLIT to 64x64;
//   * boundary-forced partitions are taken verbatim;
//   * a non-square block splits toward a square;
//   * a square block at or below 64x64 is a PARTITION_NONE leaf.
PartitionType choose_partition(const Geometry& geometry, int mi_row,
                               int mi_col, BlockSize size,
                               const Boundary& boundary,
                               const PartitionSet& allowed) {
  (void)geometry;
  (void)mi_row;
  (void)mi_col;
  if (boundary.implied && allowed.is_allowed(boundary.implied_partition)) {
    return boundary.implied_partition;
  }
  if (is_square(size) && block_width(size) > 64) {
    if (allowed.is_allowed(PartitionType::kSplit)) {
      return PartitionType::kSplit;
    }
  }
  if (is_square(size)) {
    if (allowed.is_allowed(PartitionType::kNone)) {
      return PartitionType::kNone;
    }
  } else if (block_height(size) > block_width(size)) {
    if (allowed.is_allowed(PartitionType::kHorz)) {
      return PartitionType::kHorz;
    }
    if (allowed.is_allowed(PartitionType::kVert)) {
      return PartitionType::kVert;
    }
  } else {
    if (allowed.is_allowed(PartitionType::kVert)) {
      return PartitionType::kVert;
    }
    if (allowed.is_allowed(PartitionType::kHorz)) {
      return PartitionType::kHorz;
    }
  }
  if (allowed.is_allowed(PartitionType::kNone)) {
    return PartitionType::kNone;
  }
  for (int index = 0; index < kPartitionTypeCount; ++index) {
    const auto candidate = static_cast<PartitionType>(index);
    if (allowed.is_allowed(candidate)) {
      return candidate;
    }
  }
  return PartitionType::kNone;
}

bool build_node(const Geometry& geometry, int mi_row, int mi_col,
                BlockSize size, int depth, bool has_chroma,
                bool chroma_offset, const ChromaBlock& inherited, Node& node,
                std::string& error) {
  if (!is_valid_block_size(size) || mi_row < 0 || mi_col < 0 ||
      !is_inside(geometry, mi_row, mi_col)) {
    return fail(error, "reference tree: invalid node geometry");
  }

  node.mi_row = mi_row;
  node.mi_col = mi_col;
  node.size = size;
  node.depth = depth;
  node.has_chroma = has_chroma;
  node.boundary = partition_implied(geometry, mi_row, mi_col, size);
  node.allowed = init_allowed_partitions(geometry, mi_row, mi_col, size,
                                         has_chroma, chroma_offset);

  bool forced = false;
  PartitionType partition = PartitionType::kNone;
  if (node.boundary.implied &&
      node.allowed.is_allowed(node.boundary.implied_partition)) {
    forced = true;
    partition = node.boundary.implied_partition;
  } else if (node.allowed.count == 1) {
    forced = true;
    for (int index = 0; index < kPartitionTypeCount; ++index) {
      if (node.allowed.allowed[index]) {
        partition = static_cast<PartitionType>(index);
        break;
      }
    }
  } else {
    partition =
        choose_partition(geometry, mi_row, mi_col, size, node.boundary,
                         node.allowed);
  }
  if (!node.allowed.is_allowed(partition)) {
    return fail(error, "reference tree: chosen partition is not allowed");
  }

  const ChromaState chroma = resolve_chroma(
      geometry, mi_row, mi_col, size, partition, has_chroma, chroma_offset,
      inherited);
  node.chroma = chroma.block;
  // Stored like partition_ext::Node: the recursion argument, not the value
  // after this node's own is_chroma_offset_for_partition() update.
  node.chroma_offset = chroma_offset;
  node.partition = partition;
  node.forced = forced;

  if (partition == PartitionType::kNone) {
    return true;
  }

  ChildSlot slots[4];
  const int count = enumerate_children(geometry, mi_row, mi_col, size,
                                       partition, has_chroma,
                                       chroma.chroma_offset, chroma.block,
                                       slots);
  if (count == 0) {
    return fail(error, "reference tree: partition has no children");
  }
  node.children.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    const ChildSlot& slot = slots[index];
    if (!child_is_coded(geometry, slot)) {
      continue;
    }
    Node child;
    if (!build_node(geometry, slot.mi_row, slot.mi_col, slot.size, depth + 1,
                    slot.has_chroma, slot.chroma_offset, slot.chroma, child,
                    error)) {
      return false;
    }
    node.children.push_back(std::move(child));
  }
  return true;
}

struct ValidateState {
  const Geometry& geometry;
  int root_row = 0;
  int root_col = 0;
  int sb_mi = 0;
  std::vector<std::uint8_t> covered;
  std::string error;
};

std::string node_label(const Node& node) {
  std::ostringstream out;
  out << block_size_name(node.size) << "@(" << node.mi_row << ','
      << node.mi_col << ") depth " << node.depth;
  return out.str();
}

bool validate_node(ValidateState& state, const Node& node,
                   BlockSize expected_size, int expected_row, int expected_col,
                   int expected_depth, bool expected_has_chroma,
                   bool expected_offset) {
  if (node.size != expected_size || node.mi_row != expected_row ||
      node.mi_col != expected_col || node.depth != expected_depth) {
    return fail(state.error, "node geometry mismatch at " + node_label(node));
  }
  if (node.has_chroma != expected_has_chroma ||
      node.chroma_offset != expected_offset) {
    return fail(state.error, "node chroma state mismatch at " +
                                 node_label(node));
  }
  if (!is_inside(state.geometry, node.mi_row, node.mi_col)) {
    return fail(state.error, "node starts outside the frame: " +
                                 node_label(node));
  }

  const Boundary boundary =
      partition_implied(state.geometry, node.mi_row, node.mi_col, node.size);
  const PartitionSet allowed = init_allowed_partitions(
      state.geometry, node.mi_row, node.mi_col, node.size, node.has_chroma,
      node.chroma_offset);
  if (node.boundary.has_rows != boundary.has_rows ||
      node.boundary.has_cols != boundary.has_cols ||
      node.boundary.implied != boundary.implied ||
      node.boundary.implied_partition != boundary.implied_partition ||
      node.allowed.count != allowed.count ||
      node.allowed.allowed[0] != allowed.allowed[0] ||
      node.allowed.allowed[1] != allowed.allowed[1] ||
      node.allowed.allowed[2] != allowed.allowed[2] ||
      node.allowed.allowed[3] != allowed.allowed[3]) {
    return fail(state.error, "stale boundary/allowed set at " +
                                 node_label(node));
  }
  if (!allowed.is_allowed(node.partition)) {
    return fail(state.error, "spec forbids the partition at " +
                                 node_label(node));
  }

  bool spec_forced = false;
  PartitionType spec_partition = PartitionType::kNone;
  if (boundary.implied && allowed.is_allowed(boundary.implied_partition)) {
    spec_forced = true;
    spec_partition = boundary.implied_partition;
  } else if (allowed.count == 1) {
    spec_forced = true;
    for (int index = 0; index < kPartitionTypeCount; ++index) {
      if (allowed.allowed[index]) {
        spec_partition = static_cast<PartitionType>(index);
        break;
      }
    }
  }
  if (spec_forced) {
    if (!node.forced || node.partition != spec_partition) {
      return fail(state.error, "node does not use the spec-forced partition: " +
                                   node_label(node));
    }
  } else if (node.forced) {
    return fail(state.error, "node claims to be forced but is not: " +
                                 node_label(node));
  }

  if (node.partition == PartitionType::kNone) {
    if (!node.children.empty()) {
      return fail(state.error, "PARTITION_NONE node has children: " +
                                   node_label(node));
    }
    const int w4 = num_4x4_blocks_wide(node.size);
    const int h4 = num_4x4_blocks_high(node.size);
    const int row_end = std::min(node.mi_row + h4, mi_rows(state.geometry));
    const int col_end = std::min(node.mi_col + w4, mi_cols(state.geometry));
    for (int row = node.mi_row; row < row_end; ++row) {
      for (int col = node.mi_col; col < col_end; ++col) {
        const int local_row = row - state.root_row;
        const int local_col = col - state.root_col;
        if (local_row < 0 || local_row >= state.sb_mi || local_col < 0 ||
            local_col >= state.sb_mi) {
          return fail(state.error, "leaf escapes its superblock: " +
                                       node_label(node));
        }
        const std::size_t index =
            static_cast<std::size_t>(local_row) *
                static_cast<std::size_t>(state.sb_mi) +
            static_cast<std::size_t>(local_col);
        if (state.covered[index] != 0) {
          return fail(state.error, "overlapping leaves at mi (" +
                                       std::to_string(row) + ',' +
                                       std::to_string(col) + ')');
        }
        state.covered[index] = 1;
      }
    }
    return true;
  }

  const ChromaState resolved = resolve_chroma(
      state.geometry, node.mi_row, node.mi_col, node.size, node.partition,
      node.has_chroma, node.chroma_offset, node.chroma);
  if (resolved.block != node.chroma) {
    return fail(state.error, "stale chroma block at " + node_label(node));
  }
  ChildSlot slots[4];
  const int count = enumerate_children(
      state.geometry, node.mi_row, node.mi_col, node.size, node.partition,
      node.has_chroma, resolved.chroma_offset, node.chroma, slots);
  if (count == 0) {
    return fail(state.error, "partition has no child slots: " +
                                 node_label(node));
  }
  int coded_slots = 0;
  for (int index = 0; index < count; ++index) {
    if (child_is_coded(state.geometry, slots[index])) {
      ++coded_slots;
    }
  }
  if (node.children.size() != static_cast<std::size_t>(coded_slots)) {
    return fail(state.error, "wrong child count at " + node_label(node));
  }
  std::size_t child_index = 0;
  for (int index = 0; index < count; ++index) {
    const ChildSlot& slot = slots[index];
    if (!child_is_coded(state.geometry, slot)) {
      continue;
    }
    if (!validate_node(state, node.children[child_index], slot.size,
                       slot.mi_row, slot.mi_col, node.depth + 1,
                       slot.has_chroma, slot.chroma_offset)) {
      return false;
    }
    ++child_index;
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Block tables
// ---------------------------------------------------------------------------

bool is_valid_block_size(BlockSize size) {
  const int value = static_cast<int>(size);
  return value >= 0 && value < kBlockSizeCount;
}

int mi_width_log2(BlockSize size) {
  if (!is_valid_block_size(size)) return 0;
  return static_cast<int>(kMiWidthLog2[static_cast<std::size_t>(size)]);
}

int mi_height_log2(BlockSize size) {
  if (!is_valid_block_size(size)) return 0;
  return static_cast<int>(kMiHeightLog2[static_cast<std::size_t>(size)]);
}

int num_4x4_blocks_wide(BlockSize size) { return 1 << mi_width_log2(size); }

int num_4x4_blocks_high(BlockSize size) { return 1 << mi_height_log2(size); }

int block_width(BlockSize size) { return 4 * num_4x4_blocks_wide(size); }

int block_height(BlockSize size) { return 4 * num_4x4_blocks_high(size); }

bool is_square(BlockSize size) {
  return is_valid_block_size(size) &&
         mi_width_log2(size) == mi_height_log2(size);
}

BlockSize superblock_size(int sb_size_pixels) {
  if (sb_size_pixels == 64) return BlockSize::k64x64;
  if (sb_size_pixels == 128) return BlockSize::k128x128;
  if (sb_size_pixels == 256) return BlockSize::k256x256;
  return BlockSize::kInvalid;
}

const char* block_size_name(BlockSize size) {
  if (!is_valid_block_size(size)) return "invalid";
  return kBlockNames[static_cast<std::size_t>(size)];
}

const char* partition_name(PartitionType partition) {
  switch (partition) {
    case PartitionType::kNone:
      return "NONE";
    case PartitionType::kSplit:
      return "SPLIT";
    case PartitionType::kHorz:
      return "HORZ";
    case PartitionType::kVert:
      return "VERT";
  }
  return "INVALID";
}

int spec_partition_index(PartitionType partition) {
  switch (partition) {
    case PartitionType::kNone:
      return 0;
    case PartitionType::kHorz:
      return 1;
    case PartitionType::kVert:
      return 2;
    case PartitionType::kSplit:
      return 9;
  }
  return -1;
}

BlockSize partition_subsize(PartitionType partition, BlockSize size) {
  if (!is_valid_block_size(size)) return BlockSize::kInvalid;
  const std::size_t index = static_cast<std::size_t>(size);
  switch (partition) {
    case PartitionType::kNone:
      return size;
    case PartitionType::kHorz:
      return static_cast<BlockSize>(kHorzSubsize[index]);
    case PartitionType::kVert:
      return static_cast<BlockSize>(kVertSubsize[index]);
    case PartitionType::kSplit:
      return static_cast<BlockSize>(kSplitSubsize[index]);
  }
  return BlockSize::kInvalid;
}

RectTypeImplied rect_type_implied_by_bsize(BlockSize size) {
  RectTypeImplied out;
  switch (size) {
    case BlockSize::k4x8:
    case BlockSize::k64x128:
    case BlockSize::k128x256:
    case BlockSize::k4x16:
      out.implied = true;
      out.partition = PartitionType::kHorz;
      return out;
    case BlockSize::k8x4:
    case BlockSize::k128x64:
    case BlockSize::k256x128:
    case BlockSize::k16x4:
      out.implied = true;
      out.partition = PartitionType::kVert;
      return out;
    default:
      return out;
  }
}

int partition_size_adjust(BlockSize size) {
  if (!is_valid_block_size(size)) return 0;
  return static_cast<int>(kPartitionSizeAdjust[static_cast<std::size_t>(size)]);
}

int partition_size_adjust_rect_type(BlockSize size) {
  if (!is_valid_block_size(size)) return 0;
  return static_cast<int>(
      kPartitionSizeAdjustRect[static_cast<std::size_t>(size)]);
}

// ---------------------------------------------------------------------------
// Superblock configuration
// ---------------------------------------------------------------------------

bool validate_sb_config(const SbConfig& config, std::string& error) {
  if (config.use_256x256_superblock && config.use_128x128_superblock) {
    return fail(error,
                "use_128x128_superblock is not present when "
                "use_256x256_superblock = 1 (§5.4.3)");
  }
  error.clear();
  return true;
}

BlockSize get_seq_sb_size(bool use_256x256_superblock,
                          bool use_128x128_superblock) {
  if (use_256x256_superblock) return BlockSize::k256x256;
  if (use_128x128_superblock) return BlockSize::k128x128;
  return BlockSize::k64x64;
}

BlockSize frame_sb_size(const SbConfig& config) {
  if (config.use_256x256_superblock) {
    return config.frame_is_intra ? BlockSize::k128x128 : BlockSize::k256x256;
  }
  if (config.use_128x128_superblock) return BlockSize::k128x128;
  return BlockSize::k64x64;
}

int frame_sb_size_pixels(const SbConfig& config) {
  return block_width(frame_sb_size(config));
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

bool validate_geometry(const Geometry& geometry, std::string& error) {
  if (geometry.width <= 0 || geometry.height <= 0) {
    return fail(error, "frame width and height must be positive");
  }
  if (geometry.sb_size != 64 && geometry.sb_size != 128 &&
      geometry.sb_size != 256) {
    return fail(error, "superblock size must be 64, 128 or 256");
  }
  if (geometry.max_pb_aspect_ratio != 2 &&
      geometry.max_pb_aspect_ratio != 4 &&
      geometry.max_pb_aspect_ratio != 8) {
    return fail(error, "max_pb_aspect_ratio must be 2, 4 or 8 (AV2 §5.4.3)");
  }
  if (geometry.has_chroma && geometry.chroma_subsampling_x !=
                                 geometry.chroma_subsampling_y) {
    return fail(error, "4:2:2 chroma is not modelled; use 4:2:0 or 4:4:4");
  }
  error.clear();
  return true;
}

int mi_cols(const Geometry& geometry) {
  // §5.18.4.4 compute_image_size(): 2 * ((FrameWidth + 7) >> 3).
  return 2 * ((geometry.width + 7) >> 3);
}

int mi_rows(const Geometry& geometry) {
  return 2 * ((geometry.height + 7) >> 3);
}

int sb_size_mi(const Geometry& geometry) { return geometry.sb_size / 4; }

int sb_cols(const Geometry& geometry) {
  const int size_mi = sb_size_mi(geometry);
  return size_mi > 0 ? (mi_cols(geometry) + size_mi - 1) / size_mi : 0;
}

int sb_rows(const Geometry& geometry) {
  const int size_mi = sb_size_mi(geometry);
  return size_mi > 0 ? (mi_rows(geometry) + size_mi - 1) / size_mi : 0;
}

bool is_inside(const Geometry& geometry, int mi_row, int mi_col) {
  return mi_row >= 0 && mi_col >= 0 && mi_row < mi_rows(geometry) &&
         mi_col < mi_cols(geometry);
}

bool block_coded(const Geometry& geometry, int mi_row, int mi_col) {
  return is_inside(geometry, mi_row, mi_col);
}

BlockRect block_rect(const Geometry& geometry, int mi_row, int mi_col,
                     BlockSize size) {
  BlockRect rect;
  rect.x = mi_col * 4;
  rect.y = mi_row * 4;
  rect.width = block_width(size);
  rect.height = block_height(size);
  rect.visible_width =
      std::max(0, std::min(rect.width, geometry.width - rect.x));
  rect.visible_height =
      std::max(0, std::min(rect.height, geometry.height - rect.y));
  return rect;
}

BlockSize chroma_plane_residual_size(const Geometry& geometry,
                                     BlockSize luma_size) {
  if (!is_valid_block_size(luma_size)) return BlockSize::kInvalid;
  if (!geometry.chroma_subsampling_x && !geometry.chroma_subsampling_y) {
    return luma_size;
  }
  if (geometry.chroma_subsampling_x && geometry.chroma_subsampling_y) {
    return static_cast<BlockSize>(
        kChroma420Subsize[static_cast<std::size_t>(luma_size)]);
  }
  return BlockSize::kInvalid;  // 4:2:2 not modelled
}

bool operator==(const ChromaBlock& a, const ChromaBlock& b) {
  return a.mi_row == b.mi_row && a.mi_col == b.mi_col && a.size == b.size;
}

bool operator!=(const ChromaBlock& a, const ChromaBlock& b) {
  return !(a == b);
}

bool is_chroma_offset_for_subsize(const Geometry& geometry, BlockSize subsize) {
  if (!is_valid_block_size(subsize)) return false;
  if (geometry.chroma_subsampling_y && mi_height_log2(subsize) == 0) {
    return true;
  }
  if (geometry.chroma_subsampling_x && mi_width_log2(subsize) == 0) {
    return true;
  }
  return false;
}

bool is_chroma_offset_for_partition(const Geometry& geometry,
                                    PartitionType partition, BlockSize size) {
  if (partition == PartitionType::kNone) {
    return is_chroma_offset_for_subsize(geometry, size);
  }
  return is_chroma_offset_for_subsize(geometry,
                                      partition_subsize(partition, size));
}

ChromaState resolve_chroma(const Geometry& geometry, int mi_row, int mi_col,
                           BlockSize size, PartitionType partition,
                           bool has_chroma, bool chroma_offset,
                           const ChromaBlock& inherited) {
  ChromaState state;
  state.chroma_offset = chroma_offset;
  state.block = inherited;
  if (!chroma_offset && has_chroma) {
    // §5.20.3.1: the block's own size becomes ChromaMiSize when the partition
    // itself does not trigger a chroma offset.
    state.chroma_offset =
        is_chroma_offset_for_partition(geometry, partition, size);
    state.block.mi_row = mi_row;
    state.block.mi_col = mi_col;
    state.block.size = size;
  }
  return state;
}

// ---------------------------------------------------------------------------
// Boundary / allowed set
// ---------------------------------------------------------------------------

Boundary partition_implied_at_boundary(const Geometry& geometry, int mi_row,
                                       int mi_col, BlockSize size) {
  Boundary out;
  if (!is_valid_block_size(size)) return out;
  const int numWide = num_4x4_blocks_wide(size);
  const int numHigh = num_4x4_blocks_high(size);
  out.has_rows = (mi_row + (numHigh >> 1)) < mi_rows(geometry);
  out.has_cols = (mi_col + (numWide >> 1)) < mi_cols(geometry);
  if (out.has_rows && out.has_cols) {
    return out;
  }
  PartitionType implied = PartitionType::kNone;
  if (numWide == numHigh) {
    implied = out.has_rows ? PartitionType::kVert : PartitionType::kHorz;
  } else if (numHigh > numWide) {
    if (!out.has_rows) {
      implied = PartitionType::kHorz;
    } else {
      const bool sub_has_cols = (mi_col + (numWide >> 2)) < mi_cols(geometry);
      if (numWide >= 4 && !sub_has_cols) implied = PartitionType::kHorz;
    }
  } else {
    if (!out.has_cols) {
      implied = PartitionType::kVert;
    } else {
      const bool sub_has_rows = (mi_row + (numHigh >> 2)) < mi_rows(geometry);
      if (numHigh >= 4 && !sub_has_rows) implied = PartitionType::kVert;
    }
  }
  out.implied_partition = implied;
  out.implied = implied != PartitionType::kNone;
  return out;
}

Boundary partition_implied(const Geometry& geometry, int mi_row, int mi_col,
                           BlockSize size) {
  Boundary out;
  if (!is_valid_block_size(size)) return out;
  const int value = static_cast<int>(size);
  // BLOCK_4X4 and everything at or above BLOCK_4X32 imply PARTITION_NONE.
  if (size == BlockSize::k4x4 || value >= static_cast<int>(BlockSize::k4x32)) {
    out.implied = true;
    out.implied_partition = PartitionType::kNone;
    out.has_rows = (mi_row + (num_4x4_blocks_high(size) >> 1)) <
                   mi_rows(geometry);
    out.has_cols = (mi_col + (num_4x4_blocks_wide(size) >> 1)) <
                   mi_cols(geometry);
    return out;
  }
  return partition_implied_at_boundary(geometry, mi_row, mi_col, size);
}

bool PartitionSet::is_allowed(PartitionType partition) const {
  const int index = static_cast<int>(partition);
  return index >= 0 && index < kPartitionTypeCount && allowed[index];
}

bool is_partition_allowed(const Geometry& geometry, int mi_row, int mi_col,
                          BlockSize size, PartitionType partition,
                          bool has_chroma, bool chroma_offset) {
  if (!is_valid_block_size(size) || mi_row < 0 || mi_col < 0) return false;
  const BlockSize subsize = partition_subsize(partition, size);
  if (!is_valid_block_size(subsize)) return false;

  const RectTypeImplied rect = rect_type_implied_by_bsize(size);
  if (rect.implied) {
    if (rect.partition == PartitionType::kVert &&
        partition == PartitionType::kHorz) {
      return false;
    }
    if (rect.partition == PartitionType::kHorz &&
        partition == PartitionType::kVert) {
      return false;
    }
  }

  const int bw = block_width(subsize);
  const int bh = block_height(subsize);
  const int aspect = geometry.max_pb_aspect_ratio;
  if (bw > bh * aspect || bh > bw * aspect) {
    if (partition == PartitionType::kNone) return false;
    if (bw >= bh * 8 || bh >= bw * 8) return false;
  }

  bool local_offset = chroma_offset;
  if (has_chroma && !local_offset) {
    local_offset = is_chroma_offset_for_partition(geometry, partition, size);
  }
  if (has_chroma && !local_offset &&
      chroma_plane_residual_size(geometry, subsize) == BlockSize::kInvalid) {
    return false;
  }

  const int w4 = num_4x4_blocks_wide(size);
  const int h4 = num_4x4_blocks_high(size);
  const int half_w = w4 >> 1;
  const int half_h = h4 >> 1;

  if (partition == PartitionType::kNone) {
    const bool has_rows = (mi_row + half_h) < mi_rows(geometry);
    const bool has_cols = (mi_col + half_w) < mi_cols(geometry);
    if (!has_rows || !has_cols) return false;
  }

  if (has_chroma && local_offset) {
    if (partition == PartitionType::kHorz) {
      return block_coded(geometry, mi_row + half_h, mi_col);
    }
    if (partition == PartitionType::kVert) {
      return block_coded(geometry, mi_row, mi_col + half_w);
    }
  }
  return true;
}

PartitionSet init_allowed_partitions(const Geometry& geometry, int mi_row,
                                     int mi_col, BlockSize size,
                                     bool has_chroma, bool chroma_offset) {
  PartitionSet set;
  for (int index = 0; index < kPartitionTypeCount; ++index) {
    const auto partition = static_cast<PartitionType>(index);
    const bool good = is_partition_allowed(geometry, mi_row, mi_col, size,
                                           partition, has_chroma,
                                           chroma_offset);
    set.allowed[index] = good;
    if (good) ++set.count;
  }
  if (set.count == 0) {
    set.allowed[static_cast<int>(PartitionType::kNone)] = true;
    set.count = 1;
  }
  return set;
}

// ---------------------------------------------------------------------------
// Contexts / CDF rows
// ---------------------------------------------------------------------------

MiSizeGrid::MiSizeGrid(const Geometry& geometry) { reset(geometry); }

void MiSizeGrid::reset(const Geometry& geometry) {
  rows_ = mi_rows(geometry);
  cols_ = mi_cols(geometry);
  cells_.assign(static_cast<std::size_t>(rows_) *
                    static_cast<std::size_t>(cols_),
                BlockSize::kInvalid);
}

bool MiSizeGrid::in_bounds(int mi_row, int mi_col) const {
  return mi_row >= 0 && mi_col >= 0 && mi_row < rows_ && mi_col < cols_;
}

BlockSize MiSizeGrid::at(int mi_row, int mi_col) const {
  if (!in_bounds(mi_row, mi_col)) return BlockSize::kInvalid;
  return cells_[static_cast<std::size_t>(mi_row) *
                    static_cast<std::size_t>(cols_) +
                static_cast<std::size_t>(mi_col)];
}

void MiSizeGrid::set(int mi_row, int mi_col, BlockSize size) {
  if (!in_bounds(mi_row, mi_col)) return;
  cells_[static_cast<std::size_t>(mi_row) *
             static_cast<std::size_t>(cols_) +
         static_cast<std::size_t>(mi_col)] = size;
}

void MiSizeGrid::fill_block(int mi_row, int mi_col, BlockSize size) {
  const int h4 = num_4x4_blocks_high(size);
  const int w4 = num_4x4_blocks_wide(size);
  for (int row = mi_row; row < mi_row + h4; ++row) {
    for (int col = mi_col; col < mi_col + w4; ++col) {
      set(row, col, size);
    }
  }
}

Contexts partition_contexts(const Geometry& geometry, int mi_row, int mi_col,
                            BlockSize size, const MiSizeGrid& sizes) {
  Contexts out;
  if (!is_valid_block_size(size)) return out;

  const int bsw = std::max(mi_width_log2(size), 1);
  const int bsh = std::max(mi_height_log2(size), 1);
  const int left_h = log2_or_sentinel(sizes.at(mi_row, mi_col - 1), false);
  const int above_w = log2_or_sentinel(sizes.at(mi_row - 1, mi_col), true);
  const int ctx1 = left_h < bsh ? 1 : 0;
  const int ctx2 = above_w < bsw ? 1 : 0;

  out.do_split = partition_size_adjust(size) * 4 + ctx1 * 2 + ctx2;
  out.rect_type = partition_size_adjust_rect_type(size) * 4 + ctx1 * 2 + ctx2;

  const int raw_bsw = mi_width_log2(size);
  const int raw_bsh = mi_height_log2(size);
  const bool avail_left = is_inside(geometry, mi_row, mi_col - 1);
  const bool avail_above = is_inside(geometry, mi_row - 1, mi_col);
  int square = 0;
  if (size == BlockSize::k256x256) square += 4;
  if (avail_left && left_h < raw_bsh) square += 2;
  if (avail_above && above_w < raw_bsw) square += 1;
  out.do_square_split = square;
  return out;
}

const std::uint16_t* do_split_cdf_row(int plane, int context) {
  if (plane < 0 || plane >= 2 || context < 0 || context >= 64) return nullptr;
  return &av2enc::m0::kDoSplitCdf[static_cast<std::size_t>(plane)]
                                [static_cast<std::size_t>(context)][0];
}

const std::uint16_t* do_square_split_cdf_row(int plane, int context) {
  if (plane < 0 || plane >= 2 || context < 0 || context >= 8) return nullptr;
  return &av2enc::m0::kDoSquareSplitCdf[static_cast<std::size_t>(plane)]
                                      [static_cast<std::size_t>(context)][0];
}

const std::uint16_t* rect_type_cdf_row(int plane, int context) {
  if (plane < 0 || plane >= 2 || context < 0 || context >= 64) return nullptr;
  return &av2enc::m0::kRectTypeCdf[static_cast<std::size_t>(plane)]
                                 [static_cast<std::size_t>(context)][0];
}

// ---------------------------------------------------------------------------
// SbSize-dependent unit sizing
// ---------------------------------------------------------------------------

int cdef_unit_size_pixels() { return 64; }

int cdef_units_per_sb(int sb_size_pixels) {
  if (sb_size_pixels != 64 && sb_size_pixels != 128 &&
      sb_size_pixels != 256) {
    return 0;
  }
  return sb_size_pixels / cdef_unit_size_pixels();
}

int cdef_sb_width_log2(int sb_size_pixels) {
  switch (sb_size_pixels) {
    case 64:
      return 4;
    case 128:
      return 5;
    case 256:
      return 6;
    default:
      return 0;
  }
}

int ccso_luma_size_log2_matching_sb(int sb_size_pixels) {
  const int shift = cdef_sb_width_log2(sb_size_pixels);
  return shift > 0 ? shift + 2 : 0;  // Mi_Width_Log2[SbSize] + MI_SIZE_LOG2
}

int ccso_luma_size_log2_from_tile_alignment(unsigned tile_start_or) {
  if ((tile_start_or & 63U) == 0U) return 8;
  if ((tile_start_or & 31U) == 0U) return 7;
  return 6;
}

int ccso_luma_size_log2(int sb_size_pixels, bool ccso_unit_matches_sb_size,
                        unsigned tile_start_or) {
  if (ccso_unit_matches_sb_size) {
    return ccso_luma_size_log2_matching_sb(sb_size_pixels);
  }
  return ccso_luma_size_log2_from_tile_alignment(tile_start_or);
}

int ccso_unit_mi_shift(int sb_size_pixels) {
  const int log2_size = ccso_luma_size_log2_matching_sb(sb_size_pixels);
  return log2_size > 2 ? log2_size - 2 : 0;
}

int entropy_left_context_rows(const Geometry& geometry) {
  return sb_rows(geometry) * sb_size_mi(geometry);
}

int entropy_above_context_cols(const Geometry& geometry) {
  return sb_cols(geometry) * sb_size_mi(geometry);
}

// ---------------------------------------------------------------------------
// Partition syntax plan
// ---------------------------------------------------------------------------

bool plan_partition_syntax(const Node& node, const Contexts& contexts,
                           std::vector<PartitionSymbol>& out,
                           std::string& error) {
  out.clear();
  if (node.forced) return true;

  const bool allow_none = node.allowed.is_allowed(PartitionType::kNone);
  if (allow_none) {
    out.push_back(
        {PartitionSymbolKind::kDoSplit,
         node.partition == PartitionType::kNone ? 0 : 1, contexts.do_split});
    if (node.partition == PartitionType::kNone) return true;
  } else if (node.partition == PartitionType::kNone) {
    return fail(error, "partition NONE is not allowed but was selected");
  }

  const bool allow_split = node.allowed.is_allowed(PartitionType::kSplit);
  if (allow_split) {
    const int value = node.partition == PartitionType::kSplit ? 1 : 0;
    out.push_back({PartitionSymbolKind::kDoSquareSplit, value,
                   contexts.do_square_split});
    if (node.partition == PartitionType::kSplit) return true;
  } else if (node.partition == PartitionType::kSplit) {
    return fail(error, "PARTITION_SPLIT is not allowed at this node");
  }

  const RectTypeImplied implied = rect_type_implied_by_bsize(node.size);
  PartitionType rect = implied.partition;
  bool need_rect = !implied.implied;
  if (need_rect) {
    const bool allow_h = node.allowed.is_allowed(PartitionType::kHorz);
    const bool allow_v = node.allowed.is_allowed(PartitionType::kVert);
    if (!allow_h && !allow_v) {
      return fail(error, "no rectangular partition is allowed");
    }
    if (!allow_h) {
      rect = PartitionType::kVert;
      need_rect = false;
    } else if (!allow_v) {
      rect = PartitionType::kHorz;
      need_rect = false;
    }
  }
  if (need_rect) {
    if (node.partition != PartitionType::kHorz &&
        node.partition != PartitionType::kVert) {
      return fail(error, "non-rectangular partition reached rect_type syntax");
    }
    const int value = node.partition == PartitionType::kVert ? 1 : 0;
    out.push_back({PartitionSymbolKind::kRectType, value, contexts.rect_type});
    rect = node.partition;
  }

  if (node.partition == PartitionType::kHorz && rect != PartitionType::kHorz) {
    return fail(error, "HORZ partition contradicts the implied rect_type");
  }
  if (node.partition == PartitionType::kVert && rect != PartitionType::kVert) {
    return fail(error, "VERT partition contradicts the implied rect_type");
  }
  return true;
}

// ---------------------------------------------------------------------------
// Reference tree
// ---------------------------------------------------------------------------

bool build_reference_frame(const Geometry& geometry, FrameTree& out,
                           std::string& error) {
  if (!validate_geometry(geometry, error)) return false;
  const BlockSize root_size = superblock_size(geometry.sb_size);
  if (!is_valid_block_size(root_size)) {
    return fail(error, "invalid superblock size");
  }
  const int size_mi = sb_size_mi(geometry);
  const int rows = sb_rows(geometry);
  const int cols = sb_cols(geometry);

  FrameTree frame;
  frame.geometry = geometry;
  frame.superblocks.clear();
  frame.superblocks.reserve(static_cast<std::size_t>(rows) *
                            static_cast<std::size_t>(cols));
  for (int row = 0; row < mi_rows(geometry); row += size_mi) {
    for (int col = 0; col < mi_cols(geometry); col += size_mi) {
      Node root;
      const ChromaBlock inherited{row, col, root_size};
      if (!build_node(geometry, row, col, root_size, 0, geometry.has_chroma,
                      false, inherited, root, error)) {
        return false;
      }
      frame.superblocks.push_back(std::move(root));
    }
  }
  frame.nodes = 0;
  frame.leaves = 0;
  for (const Node& root : frame.superblocks) {
    frame.nodes += count_nodes(root);
    frame.leaves += count_leaves(root);
  }
  if (!validate_frame(geometry, frame, error)) return false;
  out = std::move(frame);
  return true;
}

bool validate_frame(const Geometry& geometry, const FrameTree& frame,
                    std::string& error) {
  if (!validate_geometry(geometry, error)) return false;
  const int size_mi = sb_size_mi(geometry);
  const int rows = sb_rows(geometry);
  const int cols = sb_cols(geometry);
  if (frame.superblocks.size() !=
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) {
    return fail(error, "frame has the wrong superblock count");
  }
  const BlockSize root_size = superblock_size(geometry.sb_size);
  std::size_t index = 0;
  for (int row = 0; row < mi_rows(geometry); row += size_mi) {
    for (int col = 0; col < mi_cols(geometry); col += size_mi) {
      ValidateState state{geometry, row, col, size_mi, {}, {}};
      state.covered.assign(static_cast<std::size_t>(size_mi) *
                               static_cast<std::size_t>(size_mi),
                           0);
      const ChromaBlock expected{row, col, root_size};
      const Node& root = frame.superblocks[index++];
      if (root.chroma != expected) {
        return fail(error, "superblock chroma block mismatch");
      }
      if (!validate_node(state, root, root_size, row, col, 0,
                         geometry.has_chroma, false)) {
        return false;
      }
      const int visible_rows =
          std::min(size_mi, mi_rows(geometry) - row);
      const int visible_cols =
          std::min(size_mi, mi_cols(geometry) - col);
      for (int y = 0; y < visible_rows; ++y) {
        for (int x = 0; x < visible_cols; ++x) {
          const std::size_t cell = static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(size_mi) +
                                   static_cast<std::size_t>(x);
          if (state.covered[cell] == 0) {
            return fail(error, "superblock has uncovered cells at mi (" +
                                   std::to_string(row + y) + ',' +
                                   std::to_string(col + x) + ')');
          }
        }
      }
    }
  }
  return true;
}

void collect_leaves(const Node& node, std::vector<LeafBlock>& out) {
  if (node.partition == PartitionType::kNone) {
    LeafBlock leaf;
    leaf.mi_row = node.mi_row;
    leaf.mi_col = node.mi_col;
    leaf.size = node.size;
    leaf.depth = node.depth;
    leaf.has_chroma = node.has_chroma;
    leaf.chroma_offset = node.chroma_offset;
    leaf.chroma = node.chroma;
    out.push_back(leaf);
    return;
  }
  for (const Node& child : node.children) {
    collect_leaves(child, out);
  }
}

int count_nodes(const Node& node) {
  int count = 1;
  for (const Node& child : node.children) count += count_nodes(child);
  return count;
}

int count_leaves(const Node& node) {
  if (node.partition == PartitionType::kNone) return 1;
  int count = 0;
  for (const Node& child : node.children) count += count_leaves(child);
  return count;
}

}  // namespace av2enc::sb256
