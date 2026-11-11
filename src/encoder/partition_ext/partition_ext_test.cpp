// Tests for the AV2 extended-partition geometry / allowed-set / decision
// reference.  No entropy coder here; ext_syntax_test.cpp covers the symbols.

#include "test_util.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

int g_failures = 0;

namespace {

using pe::BlockSize;
using pe::PartitionType;

constexpr PartitionType kNone = PartitionType::kNone;
constexpr PartitionType kHorz = PartitionType::kHorz;
constexpr PartitionType kVert = PartitionType::kVert;
constexpr PartitionType kHorz3 = PartitionType::kHorz3;
constexpr PartitionType kVert3 = PartitionType::kVert3;
constexpr PartitionType kHorz4A = PartitionType::kHorz4A;
constexpr PartitionType kHorz4B = PartitionType::kHorz4B;
constexpr PartitionType kVert4A = PartitionType::kVert4A;
constexpr PartitionType kVert4B = PartitionType::kVert4B;
constexpr PartitionType kSplit = PartitionType::kSplit;

void test_enum_and_tables() {
  CHECK_EQ(pe::kPartitionCount, 10);
  CHECK_STR(pe::partition_name(kNone), "NONE");
  CHECK_STR(pe::partition_name(kHorz3), "HORZ_3");
  CHECK_STR(pe::partition_name(kVert4B), "VERT_4B");
  CHECK_STR(pe::partition_name(kSplit), "SPLIT");
  for (int i = 0; i < 10; ++i) {
    const PartitionType p = static_cast<PartitionType>(i);
    CHECK_EQ(pe::spec_partition_index(p), i);
    CHECK_EQ(pe::is_valid_partition(p), 1);
  }
  CHECK(!pe::is_extended(kNone));
  CHECK(!pe::is_extended(kSplit));
  CHECK(pe::is_extended(kHorz3));
  CHECK(pe::is_extended(kVert4B));
  CHECK(pe::is_three_way(kHorz3));
  CHECK(!pe::is_three_way(kHorz4A));
  CHECK(pe::is_uneven_4way(kHorz4A));
  CHECK(pe::is_four_way(kVert3));
  CHECK(pe::is_horizontal(kHorz4B));
  CHECK(pe::is_vertical(kVert3));
  CHECK(pe::rect_direction(kHorz3) == kHorz);
  CHECK(pe::rect_direction(kVert4A) == kVert);
  CHECK(pe::rect_direction(kNone) == kNone);
  CHECK(pe::to_base(kHorz3) == p::PartitionType::kHorz);
  CHECK(pe::to_base(kVert4B) == p::PartitionType::kVert);
  CHECK(pe::to_base(kSplit) == p::PartitionType::kSplit);

  // §9.2 Partition_Subsize for the extended rows.
  CHECK(pe::partition_subsize(kHorz3, BlockSize::k64x64) == BlockSize::k64x16);
  CHECK(pe::partition_subsize(kHorz3, BlockSize::k8x32) == BlockSize::k8x8);
  CHECK(pe::partition_subsize(kVert3, BlockSize::k64x64) == BlockSize::k16x64);
  CHECK(pe::partition_subsize(kVert3, BlockSize::k32x8) == BlockSize::k8x8);
  CHECK(pe::partition_subsize(kHorz4A, BlockSize::k64x64) == BlockSize::k64x8);
  CHECK(pe::partition_subsize(kHorz4A, BlockSize::k32x32) == BlockSize::k32x4);
  CHECK(pe::partition_subsize(kHorz4A, BlockSize::k16x16) ==
        BlockSize::kInvalid);
  CHECK(pe::partition_subsize(kVert4B, BlockSize::k32x32) == BlockSize::k4x32);
  CHECK(pe::partition_subsize(kHorz4B, BlockSize::k8x32) == BlockSize::k8x4);
  CHECK(pe::partition_subsize(kVert4A, BlockSize::k32x16) == BlockSize::k4x16);
  CHECK(pe::partition_subsize(kSplit, BlockSize::k128x128) ==
        BlockSize::k64x64);
  CHECK(pe::partition_subsize(kSplit, BlockSize::k64x64) == BlockSize::kInvalid);
  CHECK(pe::partition_subsize(kHorz3, BlockSize::k128x128) ==
        BlockSize::kInvalid);
  CHECK(pe::partition_subsize(kHorz3, BlockSize::k4x16) == BlockSize::kInvalid);

  CHECK(pe::h_partition_midsize(BlockSize::k64x64) == BlockSize::k32x32);
  CHECK(pe::h_partition_midsize(BlockSize::k8x32) == BlockSize::k4x16);
  CHECK(pe::h_partition_midsize(BlockSize::k32x8) == BlockSize::k16x4);
  CHECK(pe::h_partition_midsize(BlockSize::k8x16) == BlockSize::k4x8);
  CHECK(pe::h_partition_midsize(BlockSize::k4x4) == BlockSize::kInvalid);

  CHECK_EQ(pe::size_adjust(BlockSize::k64x64), 3);
  CHECK_EQ(pe::size_adjust(BlockSize::k16x16), 1);
  CHECK_EQ(pe::size_adjust(BlockSize::k4x4), 0);
  CHECK_EQ(pe::size_adjust_rect_type(BlockSize::k64x64), 6);
  CHECK_EQ(pe::size_adjust_rect_type(BlockSize::k32x32), 3);

  std::string error;
  CHECK(pe::validate_tools(ext_tools(true, true), error));
  CHECK(pe::validate_tools(ext_tools(false, false), error));
  CHECK(!pe::validate_tools(ext_tools(false, true), error));
  CHECK(!error.empty());
}

void test_implied() {
  const p::FrameGeometry frame = geometry_420(64, 64, 64);
  pe::Boundary b = pe::implied_partition(frame, 0, 0, BlockSize::k64x64);
  CHECK(!b.implied);

  // 4x4 and thin blocks always imply PARTITION_NONE.
  b = pe::implied_partition(frame, 0, 0, BlockSize::k4x4);
  CHECK(b.implied);
  CHECK(b.implied_partition == kNone);
  b = pe::implied_partition(frame, 0, 0, BlockSize::k4x32);
  CHECK(b.implied);
  CHECK(b.implied_partition == kNone);

  // 520x128 with a 128x128 superblock: the last root is forced VERT.
  const p::FrameGeometry edge = geometry_420(520, 128, 128);
  b = pe::implied_partition(edge, 0, 128, BlockSize::k128x128);
  CHECK(b.implied);
  CHECK(b.implied_partition == kVert);
  CHECK(!b.has_cols);

  // 1080p bottom 128x128 root: forced HORZ.
  const p::FrameGeometry full = geometry_420(1920, 1080, 128);
  b = pe::implied_partition(full, 256, 0, BlockSize::k128x128);
  CHECK(b.implied);
  CHECK(b.implied_partition == kHorz);
  CHECK(!b.has_rows);

  // 640x360 bottom 64x64 root is not implied: its half-square fits.
  const p::FrameGeometry hd = geometry_420(640, 360, 64);
  b = pe::implied_partition(hd, 80, 0, BlockSize::k64x64);
  CHECK(!b.implied);
}

pe::PartitionSet allowed_at(const p::FrameGeometry& geometry, int r, int c,
                            BlockSize size, bool has_chroma,
                            const pe::Tools& tools) {
  return pe::init_allowed_partitions(geometry, r, c, size, has_chroma,
                                     /*chroma_offset=*/false, tools);
}

int count_allowed(const pe::PartitionSet& set) { return set.count; }

bool allowed_equals_base(const pe::PartitionSet& ext,
                         const p::PartitionSet& base) {
  if (ext.count != base.count) return false;
  // partition::PartitionType order is NONE, SPLIT, HORZ, VERT.
  if (ext.is_allowed(kNone) != base.allowed[0]) return false;
  if (ext.is_allowed(kSplit) != base.allowed[1]) return false;
  if (ext.is_allowed(kHorz) != base.allowed[2]) return false;
  if (ext.is_allowed(kVert) != base.allowed[3]) return false;
  return true;
}

void test_allowed_sets() {
  const p::FrameGeometry frame = geometry_420(64, 64, 64);
  const pe::Tools on = ext_tools(true, true);

  pe::PartitionSet set = allowed_at(frame, 0, 0, BlockSize::k64x64, true, on);
  CHECK_EQ(count_allowed(set), 9);
  CHECK(!set.is_allowed(kSplit));
  for (PartitionType p :
       {kNone, kHorz, kVert, kHorz3, kVert3, kHorz4A, kHorz4B, kVert4A,
        kVert4B}) {
    CHECK(set.is_allowed(p));
  }

  set = allowed_at(frame, 0, 0, BlockSize::k32x32, true, on);
  CHECK_EQ(count_allowed(set), 9);

  set = allowed_at(frame, 0, 0, BlockSize::k16x16, true, on);
  CHECK_EQ(count_allowed(set), 5);
  CHECK(set.is_allowed(kHorz3));
  CHECK(set.is_allowed(kVert3));
  CHECK(!set.is_allowed(kHorz4A));

  set = allowed_at(frame, 0, 0, BlockSize::k8x16, true, on);
  CHECK_EQ(count_allowed(set), 4);
  CHECK(set.is_allowed(kHorz3));
  CHECK(!set.is_allowed(kVert3));

  set = allowed_at(frame, 0, 0, BlockSize::k16x8, true, on);
  CHECK_EQ(count_allowed(set), 4);
  CHECK(set.is_allowed(kVert3));

  set = allowed_at(frame, 0, 0, BlockSize::k8x32, true, on);
  CHECK_EQ(count_allowed(set), 6);
  CHECK(set.is_allowed(kHorz3));
  CHECK(set.is_allowed(kHorz4A));
  CHECK(set.is_allowed(kHorz4B));

  set = allowed_at(frame, 0, 0, BlockSize::k32x8, true, on);
  CHECK_EQ(count_allowed(set), 6);
  CHECK(set.is_allowed(kVert3));
  CHECK(set.is_allowed(kVert4A));
  CHECK(set.is_allowed(kVert4B));

  // Tool flags.
  const pe::Tools ext_only = ext_tools(true, false);
  set = allowed_at(frame, 0, 0, BlockSize::k64x64, true, ext_only);
  CHECK_EQ(count_allowed(set), 5);
  CHECK(set.is_allowed(kHorz3));
  CHECK(set.is_allowed(kVert3));
  CHECK(!set.is_allowed(kHorz4A));
  CHECK(!set.is_allowed(kVert4B));

  const pe::Tools off = ext_tools(false, false);
  set = allowed_at(frame, 0, 0, BlockSize::k64x64, true, off);
  CHECK_EQ(count_allowed(set), 3);
  CHECK(set.is_allowed(kNone));
  CHECK(set.is_allowed(kHorz));
  CHECK(set.is_allowed(kVert));
  CHECK(!set.is_allowed(kHorz3));

  // Extended-partition rejection when MaxPbAspectRatio is reduced to 4:
  // 32x32 4-way uses a 32x4 subsize (aspect 8).
  p::FrameGeometry narrow = frame;
  narrow.max_pb_aspect_ratio = 4;
  set = allowed_at(narrow, 0, 0, BlockSize::k32x32, true, on);
  CHECK(!set.is_allowed(kHorz4A));
  CHECK(!set.is_allowed(kHorz4B));
  CHECK(set.is_allowed(kHorz3));
  narrow.max_pb_aspect_ratio = 2;
  set = allowed_at(narrow, 0, 0, BlockSize::k32x32, true, on);
  // §5.20.3.2 rejects a non-NONE split only when the subsize ratio reaches 8;
  // 32x8 (ratio 4) is still allowed at MaxPbAspectRatio = 2.
  CHECK(set.is_allowed(kHorz3));
  CHECK(set.is_allowed(kHorz));    // 32x16 aspect 2
  CHECK(set.is_allowed(kVert));

  // Monochrome: no chroma args, ext partitions unchanged.
  set = allowed_at(frame, 0, 0, BlockSize::k64x64, /*has_chroma=*/false, on);
  CHECK_EQ(count_allowed(set), 9);

  // Cross-check every base decision against the rect-only module.
  const p::FrameGeometry frames[] = {
      geometry_420(64, 64, 64),   geometry_420(128, 128, 128),
      geometry_420(640, 360, 64), geometry_420(1920, 1080, 128),
      geometry_420(520, 128, 128), geometry_420(8, 8, 64),
  };
  const BlockSize sizes[] = {
      BlockSize::k64x64, BlockSize::k32x32, BlockSize::k16x16,
      BlockSize::k8x8,   BlockSize::k8x16,  BlockSize::k16x8,
      BlockSize::k8x32,  BlockSize::k32x8,  BlockSize::k4x16,
      BlockSize::k16x4,  BlockSize::k4x8,   BlockSize::k8x4,
  };
  for (const p::FrameGeometry& g : frames) {
    for (int r = 0; r < p::mi_rows(g); r += 4) {
      for (int c = 0; c < p::mi_cols(g); c += 4) {
        for (BlockSize size : sizes) {
          if (c + p::num_4x4_blocks_wide(size) > p::mi_cols(g) + 16) continue;
          for (int hc = 0; hc < 2; ++hc) {
            for (int co = 0; co < 2; ++co) {
              const bool has_chroma = hc != 0;
              const bool chroma_offset = co != 0;
              const pe::PartitionSet ext = pe::init_allowed_partitions(
                  g, r, c, size, has_chroma, chroma_offset, off);
              const p::PartitionSet base = p::init_allowed_partitions(
                  g, r, c, size, has_chroma, chroma_offset);
              if (!allowed_equals_base(ext, base)) {
                ++g_failures;
                std::cerr << "allowed mismatch at " << r << ',' << c << ' '
                          << p::block_size_name(size) << " hc=" << has_chroma
                          << " co=" << chroma_offset << '\n';
              }
            }
          }
        }
      }
    }
  }
}

void test_chroma_rules() {
  const p::FrameGeometry frame = geometry_420(64, 64, 64);

  CHECK(pe::chroma_offset_for_subsize(frame, BlockSize::k8x4));
  CHECK(pe::chroma_offset_for_subsize(frame, BlockSize::k4x16));
  CHECK(!pe::chroma_offset_for_subsize(frame, BlockSize::k8x8));

  // HORZ_3@8x32: subsize 8x8 is not an offset, and the middle special case
  // suppresses the midsize (4x16) offset check.
  CHECK(!pe::chroma_offset_for_partition(frame, kHorz3, BlockSize::k8x32));
  // HORZ_3@8x16: subsize 8x4 is an offset.
  CHECK(pe::chroma_offset_for_partition(frame, kHorz3, BlockSize::k8x16));
  // VERT_3@32x8: the spec has no midsize special case, and the subsize 8x8
  // is not an offset.
  CHECK(!pe::chroma_offset_for_partition(frame, kVert3, BlockSize::k32x8));
  // 4-way at 32x32 uses a 32x4 / 4x32 subsize: offset.
  CHECK(pe::chroma_offset_for_partition(frame, kHorz4A, BlockSize::k32x32));
  CHECK(pe::chroma_offset_for_partition(frame, kVert4B, BlockSize::k32x32));
  // 4-way at 64x64 uses 64x8: not an offset.
  CHECK(!pe::chroma_offset_for_partition(frame, kHorz4A, BlockSize::k64x64));

  // 4:4:4 never offsets.
  p::FrameGeometry g444 = frame;
  g444.chroma_subsampling_x = false;
  g444.chroma_subsampling_y = false;
  CHECK(!pe::chroma_offset_for_partition(g444, kHorz3, BlockSize::k8x16));
}

void expect_slot(const pe::ChildSlot& slot, int row, int col, BlockSize size,
                 bool has_chroma, bool chroma_offset, int chroma_row,
                 int chroma_col, BlockSize chroma_size) {
  CHECK_EQ(slot.mi_row, row);
  CHECK_EQ(slot.mi_col, col);
  CHECK(slot.size == size);
  CHECK_EQ(slot.has_chroma, has_chroma);
  CHECK_EQ(slot.chroma_offset, chroma_offset);
  CHECK_EQ(slot.chroma.mi_row, chroma_row);
  CHECK_EQ(slot.chroma.mi_col, chroma_col);
  CHECK(slot.chroma.size == chroma_size);
}

void test_child_geometry() {
  const p::FrameGeometry frame = geometry_420(64, 64, 64);
  pe::ChildSlot slots[4];
  pe::ChromaBlock inherited{0, 0, BlockSize::k64x64};

  int n = pe::enumerate_children(frame, 0, 0, BlockSize::k64x64, kHorz3, true,
                                 false, inherited, slots);
  CHECK_EQ(n, 4);
  expect_slot(slots[0], 0, 0, BlockSize::k64x16, true, false, 0, 0,
              BlockSize::k64x64);
  expect_slot(slots[1], 4, 0, BlockSize::k32x32, true, false, 0, 0,
              BlockSize::k64x64);
  expect_slot(slots[2], 4, 8, BlockSize::k32x32, true, false, 0, 0,
              BlockSize::k64x64);
  expect_slot(slots[3], 12, 0, BlockSize::k64x16, true, false, 0, 0,
              BlockSize::k64x64);

  n = pe::enumerate_children(frame, 0, 0, BlockSize::k64x64, kVert3, true,
                             false, inherited, slots);
  CHECK_EQ(n, 4);
  expect_slot(slots[0], 0, 0, BlockSize::k16x64, true, false, 0, 0,
              BlockSize::k64x64);
  expect_slot(slots[1], 0, 4, BlockSize::k32x32, true, false, 0, 0,
              BlockSize::k64x64);
  expect_slot(slots[2], 8, 4, BlockSize::k32x32, true, false, 0, 0,
              BlockSize::k64x64);
  expect_slot(slots[3], 0, 12, BlockSize::k16x64, true, false, 0, 0,
              BlockSize::k64x64);

  // HORZ_4A / 4B at 32x32.
  n = pe::enumerate_children(frame, 0, 0, BlockSize::k32x32, kHorz4A, true,
                             false, inherited, slots);
  CHECK_EQ(n, 4);
  expect_slot(slots[0], 0, 0, BlockSize::k32x4, false, true, 0, 0,
              BlockSize::k32x32);
  expect_slot(slots[1], 1, 0, BlockSize::k32x8, false, true, 0, 0,
              BlockSize::k32x32);
  expect_slot(slots[2], 3, 0, BlockSize::k32x16, false, true, 0, 0,
              BlockSize::k32x32);
  expect_slot(slots[3], 7, 0, BlockSize::k32x4, true, true, 0, 0,
              BlockSize::k32x32);

  n = pe::enumerate_children(frame, 0, 0, BlockSize::k32x32, kHorz4B, true,
                             false, inherited, slots);
  CHECK_EQ(n, 4);
  CHECK(slots[1].size == BlockSize::k32x16);
  CHECK_EQ(slots[1].mi_row, 1);
  CHECK(slots[2].size == BlockSize::k32x8);
  CHECK_EQ(slots[2].mi_row, 5);
  CHECK(slots[3].size == BlockSize::k32x4);
  CHECK_EQ(slots[3].mi_row, 7);

  n = pe::enumerate_children(frame, 0, 0, BlockSize::k32x32, kVert4B, true,
                             false, inherited, slots);
  CHECK_EQ(n, 4);
  CHECK(slots[0].size == BlockSize::k4x32);
  CHECK(slots[1].size == BlockSize::k16x32);
  CHECK_EQ(slots[1].mi_col, 1);
  CHECK(slots[2].size == BlockSize::k8x32);
  CHECK_EQ(slots[2].mi_col, 5);
  CHECK(slots[3].size == BlockSize::k4x32);
  CHECK_EQ(slots[3].mi_col, 7);

  // HORZ_3@8x32 shares one middle chroma block.
  n = pe::enumerate_children(frame, 0, 0, BlockSize::k8x32, kHorz3, true,
                             false, inherited, slots);
  CHECK_EQ(n, 4);
  expect_slot(slots[0], 0, 0, BlockSize::k8x8, true, false, 0, 0,
              BlockSize::k8x32);
  expect_slot(slots[1], 2, 0, BlockSize::k4x16, false, true, 2, 0,
              BlockSize::k8x16);
  expect_slot(slots[2], 2, 1, BlockSize::k4x16, true, true, 2, 0,
              BlockSize::k8x16);
  expect_slot(slots[3], 6, 0, BlockSize::k8x8, true, false, 2, 0,
              BlockSize::k8x16);

  // VERT_3@32x8.
  n = pe::enumerate_children(frame, 0, 0, BlockSize::k32x8, kVert3, true,
                             false, inherited, slots);
  CHECK_EQ(n, 4);
  expect_slot(slots[0], 0, 0, BlockSize::k8x8, true, false, 0, 0,
              BlockSize::k32x8);
  expect_slot(slots[1], 0, 2, BlockSize::k16x4, false, true, 0, 2,
              BlockSize::k16x8);
  expect_slot(slots[2], 1, 2, BlockSize::k16x4, true, true, 0, 2,
              BlockSize::k16x8);
  expect_slot(slots[3], 0, 6, BlockSize::k8x8, true, false, 0, 2,
              BlockSize::k16x8);

  // SPLIT resets chromaOffset for every child (128x128 root only).
  const p::FrameGeometry big = geometry_420(128, 128, 128);
  n = pe::enumerate_children(big, 0, 0, BlockSize::k128x128, kSplit, true,
                             true, inherited, slots);
  CHECK_EQ(n, 4);
  for (int i = 0; i < 4; ++i) {
    CHECK_EQ(slots[i].has_chroma, 1);
    CHECK_EQ(slots[i].chroma_offset, 0);
    CHECK(slots[i].size == BlockSize::k64x64);
  }

  // Out-of-frame children are flagged by child_is_coded.
  const p::FrameGeometry edge = geometry_420(64, 64, 64);
  n = pe::enumerate_children(edge, 14, 0, BlockSize::k16x16, kHorz, true,
                             false, inherited, slots);
  CHECK_EQ(n, 2);
  CHECK(pe::child_is_coded(edge, slots[0]));
  CHECK(!pe::child_is_coded(edge, slots[1]));
}

std::string ext_signature(const pe::PartitionNode& node) {
  std::string out = p::block_size_name(node.size);
  out += '@';
  out += std::to_string(node.mi_row);
  out += ',';
  out += std::to_string(node.mi_col);
  out += ':';
  out += pe::partition_name(node.partition);
  out += ':';
  out += node.forced ? 'F' : 'S';
  out += '(';
  for (const pe::PartitionNode& child : node.children) {
    out += ext_signature(child);
  }
  out += ')';
  return out;
}

std::string frame_signature(const pe::FrameDecision& frame) {
  std::string out;
  for (const pe::PartitionNode& sb : frame.superblocks) {
    out += ext_signature(sb);
  }
  return out;
}

void test_decision_ext_trees() {
  const pe::Tools tools = ext_tools(true, true);
  pe::DecisionOptions options;

  // 64x64 root -> HORZ_3.
  {
    const p::FrameGeometry frame = geometry_420(64, 64, 64);
    pe::CostFunction costs = choose_partition([](const pe::NodeInfo& info) {
      return info.depth == 0 ? kHorz3 : kNone;
    });
    const pe::PartitionDecision decision =
        pe::decide_superblock(frame, 0, 0, costs, tools, options);
    CHECK(decision.status == pe::DecisionStatus::kOk);
    CHECK(decision.root.partition == kHorz3);
    CHECK_EQ(decision.leaves, 4);
    CHECK_EQ(decision.nodes, 5);
    CHECK(!decision.root.forced);
    std::string error;
    CHECK(pe::validate_tree(frame, tools, decision.root, error));
    if (!error.empty()) std::cerr << error << '\n';

    // Leaves carry the spec chroma blocks.
    std::vector<pe::LeafBlock> leaves;
    pe::collect_leaves(decision.root, leaves);
    CHECK_EQ(leaves.size(), 4);
    CHECK(leaves[0].chroma.size == BlockSize::k64x16);
    CHECK(leaves[1].chroma.size == BlockSize::k32x32);
    CHECK(leaves[3].chroma.size == BlockSize::k64x16);
  }

  // 640x360 frame, HORZ_3 at every 64x64/32x32/16x16 node it can use.
  {
    const p::FrameGeometry frame = geometry_420(640, 360, 64);
    pe::CostFunction costs = choose_partition([](const pe::NodeInfo& info) {
      if (info.size == BlockSize::k64x64 ||
          info.size == BlockSize::k32x32 ||
          info.size == BlockSize::k16x16) {
        return kHorz3;
      }
      return kNone;
    });
    const pe::FrameDecision a = pe::decide_frame(frame, costs, tools, options);
    const pe::FrameDecision b = pe::decide_frame(frame, costs, tools, options);
    CHECK(a.status == pe::DecisionStatus::kOk);
    CHECK(b.status == pe::DecisionStatus::kOk);
    CHECK_STR(frame_signature(a), frame_signature(b));
    std::string error;
    CHECK(pe::validate_frame(frame, tools, a, error));
    if (!error.empty()) std::cerr << error << '\n';
    CHECK(a.nodes > a.leaves);
    CHECK_EQ(a.superblocks.size(), static_cast<std::size_t>(10 * 6));
  }

  // 1080p, 128x128 superblocks: SPLIT root, HORZ_4A at 64x64.
  {
    const p::FrameGeometry frame = geometry_420(1920, 1080, 128);
    pe::CostFunction costs = choose_partition([](const pe::NodeInfo& info) {
      if (info.size == BlockSize::k128x128) return kSplit;
      if (info.size == BlockSize::k64x64) return kHorz4A;
      return kNone;
    });
    const pe::FrameDecision decision =
        pe::decide_frame(frame, costs, tools, options);
    CHECK(decision.status == pe::DecisionStatus::kOk);
    std::string error;
    CHECK(pe::validate_frame(frame, tools, decision, error));
    if (!error.empty()) std::cerr << error << '\n';
    CHECK_EQ(decision.superblocks.size(), static_cast<std::size_t>(15 * 9));

    // The bottom SB row root is spec-forced HORZ, overriding the chooser.
    const pe::PartitionNode& bottom = decision.superblocks[15 * 8];
    CHECK(bottom.partition == kHorz);
    CHECK(bottom.forced);
  }

  // 520x128: last 128x128 root is forced VERT.
  {
    const p::FrameGeometry frame = geometry_420(520, 128, 128);
    pe::CostFunction costs = choose_partition([](const pe::NodeInfo& info) {
      return info.size == BlockSize::k64x64 ? kVert4B : kNone;
    });
    const pe::FrameDecision decision =
        pe::decide_frame(frame, costs, tools, options);
    CHECK(decision.status == pe::DecisionStatus::kOk);
    std::string error;
    CHECK(pe::validate_frame(frame, tools, decision, error));
    if (!error.empty()) std::cerr << error << '\n';
    CHECK(decision.superblocks[4].partition == kVert);
    CHECK(decision.superblocks[4].forced);
  }

  // Monochrome and 4:4:4 geometries carry the same partition geometry; only
  // the chroma bookkeeping changes.
  for (int variant = 0; variant < 2; ++variant) {
    p::FrameGeometry frame = geometry_420(64, 64, 64);
    if (variant == 0) {
      frame.has_chroma = false;
    } else {
      frame.chroma_subsampling_x = false;
      frame.chroma_subsampling_y = false;
    }
    pe::CostFunction costs = choose_partition([](const pe::NodeInfo& info) {
      return info.depth == 0 ? kHorz3 : kNone;
    });
    const pe::PartitionDecision decision = pe::decide_superblock(
        frame, 0, 0, costs, tools, pe::DecisionOptions());
    CHECK(decision.status == pe::DecisionStatus::kOk);
    CHECK(decision.root.partition == kHorz3);
    std::string error;
    CHECK(pe::validate_tree(frame, tools, decision.root, error));
    if (!error.empty()) std::cerr << error << '\n';
    std::vector<pe::LeafBlock> leaves;
    pe::collect_leaves(decision.root, leaves);
    CHECK_EQ(leaves.size(), 4);
    for (const pe::LeafBlock& leaf : leaves) {
      CHECK_EQ(leaf.has_chroma, variant == 0 ? 0 : 1);
    }
  }

  // All four uneven variants are reachable.
  for (PartitionType want :
       {kHorz4A, kHorz4B, kVert4A, kVert4B}) {
    const p::FrameGeometry frame = geometry_420(64, 64, 64);
    pe::CostFunction costs = choose_partition([want](const pe::NodeInfo& info) {
      return info.depth == 0 ? want : kNone;
    });
    pe::DecisionOptions uneven;
    const pe::PartitionDecision decision =
        pe::decide_superblock(frame, 0, 0, costs, tools, uneven);
    CHECK(decision.status == pe::DecisionStatus::kOk);
    CHECK(decision.root.partition == want);
    CHECK_EQ(decision.leaves, 4);
    std::string error;
    CHECK(pe::validate_tree(frame, tools, decision.root, error));
    if (!error.empty()) std::cerr << error << '\n';
  }
  // VERT_4A/4B at 32x32 through a 64x64 VERT root.
  for (PartitionType want : {kVert4A, kVert4B}) {
    const p::FrameGeometry frame = geometry_420(64, 64, 64);
    pe::CostFunction costs = choose_partition([want](const pe::NodeInfo& info) {
      if (info.depth == 0) return kVert;
      if (info.size == BlockSize::k32x64) return kHorz;
      if (info.size == BlockSize::k32x32) return want;
      return kNone;
    });
    const pe::PartitionDecision decision =
        pe::decide_superblock(frame, 0, 0, costs, tools, pe::DecisionOptions());
    CHECK(decision.status == pe::DecisionStatus::kOk);
    CHECK(decision.root.partition == kVert);
    CHECK(decision.root.children.size() == 2);
    CHECK(decision.root.children[0].partition == kHorz);
    CHECK(decision.root.children[0].children[0].partition == want);
    std::string error;
    CHECK(pe::validate_tree(frame, tools, decision.root, error));
    if (!error.empty()) std::cerr << error << '\n';
  }
}

void test_decision_policy() {
  const pe::Tools tools = ext_tools(true, true);
  const p::FrameGeometry frame = geometry_420(64, 64, 64);

  // Family switches: NONE and one extended split are both affordable, so the
  // split wins normally and NONE wins when the family is disabled.
  auto both_cost = [](pe::PartitionType split) {
    return [split](const pe::NodeInfo& info) {
      const double w = static_cast<double>(p::block_width(info.size));
      const double h = static_cast<double>(p::block_height(info.size));
      pe::NodeCosts costs;
      costs.leaf = w * w * h * h;
      costs.none_signal = 0.0;
      costs.split_signal = pe::kInfiniteCost;
      costs.horz_signal = pe::kInfiniteCost;
      costs.vert_signal = pe::kInfiniteCost;
      costs.horz3_signal = pe::kInfiniteCost;
      costs.vert3_signal = pe::kInfiniteCost;
      costs.horz4a_signal = pe::kInfiniteCost;
      costs.horz4b_signal = pe::kInfiniteCost;
      costs.vert4a_signal = pe::kInfiniteCost;
      costs.vert4b_signal = pe::kInfiniteCost;
      if (info.depth == 0) {
        if (split == kHorz3) costs.horz3_signal = 0.0;
        if (split == kHorz4A) costs.horz4a_signal = 0.0;
      }
      return costs;
    };
  };
  pe::CostFunction h3 = both_cost(kHorz3);
  pe::DecisionOptions no3;
  no3.allow_3way = false;
  pe::PartitionDecision decision =
      pe::decide_superblock(frame, 0, 0, h3, tools, no3);
  CHECK(decision.status == pe::DecisionStatus::kOk);
  CHECK(decision.root.partition == kNone);

  pe::DecisionOptions no4;
  no4.allow_uneven4 = false;
  pe::CostFunction h4 = both_cost(kHorz4A);
  decision = pe::decide_superblock(frame, 0, 0, h4, tools, no4);
  CHECK(decision.status == pe::DecisionStatus::kOk);
  CHECK(decision.root.partition == kNone);

  // max_depth 1 allows the root split and stops the children.
  pe::DecisionOptions depth1;
  depth1.max_depth = 1;
  decision = pe::decide_superblock(frame, 0, 0, h3, tools, depth1);
  CHECK(decision.status == pe::DecisionStatus::kOk);
  CHECK(decision.root.partition == kHorz3);
  CHECK_EQ(decision.leaves, 4);

  // min_block_size_px 8 rejects the 32x4 / 64x8 4-way subsize at 32x32.
  const p::FrameGeometry frame32 = frame;  // root is 64x64
  pe::CostFunction v4 = choose_partition([](const pe::NodeInfo& info) {
    if (info.depth == 0) return kVert;
    if (info.size == BlockSize::k32x64) return kVert4A;
    return kNone;
  });
  pe::DecisionOptions floor8;
  floor8.min_block_size_px = 8;
  decision = pe::decide_superblock(frame32, 0, 0, v4, tools, floor8);
  CHECK(decision.status == pe::DecisionStatus::kOk);
  // 64x64 -> 32x64, then 32x64 VERT_4A has a 4x16 subsize (width 4 < 8).
  CHECK(decision.root.partition == kVert);
  CHECK(decision.root.children[0].partition == kNone);

  // A forced partition that conflicts with the policy is an error.
  const p::FrameGeometry edge = geometry_420(520, 128, 128);
  pe::DecisionOptions no_depth;
  no_depth.max_depth = 0;
  pe::CostFunction none = choose_partition(
      [](const pe::NodeInfo&) { return kNone; });
  const pe::PartitionDecision forced =
      pe::decide_superblock(edge, 0, 128, none, tools, no_depth);
  CHECK(forced.status == pe::DecisionStatus::kNoValidPartition);

  // Invalid tools.
  const pe::Tools bad = ext_tools(false, true);
  decision = pe::decide_superblock(frame, 0, 0, h3, bad, pe::DecisionOptions());
  CHECK(decision.status == pe::DecisionStatus::kInvalidTools);

  // Invalid costs.
  pe::CostFunction nan_cost = [](const pe::NodeInfo&) {
    pe::NodeCosts costs;
    costs.leaf = std::numeric_limits<double>::quiet_NaN();
    return costs;
  };
  decision =
      pe::decide_superblock(frame, 0, 0, nan_cost, tools, pe::DecisionOptions());
  CHECK(decision.status == pe::DecisionStatus::kInvalidCost);

  pe::CostFunction negative = [](const pe::NodeInfo&) {
    pe::NodeCosts costs;
    costs.leaf = 1.0;
    costs.horz3_signal = -1.0;
    return costs;
  };
  decision = pe::decide_superblock(frame, 0, 0, negative, tools,
                                   pe::DecisionOptions());
  CHECK(decision.status == pe::DecisionStatus::kInvalidCost);

  // Invalid superblock origin.
  decision = pe::decide_superblock(frame, 8, 0, h3, tools, pe::DecisionOptions());
  CHECK(decision.status == pe::DecisionStatus::kInvalidGeometry);
}

void test_rect_compat() {
  // With the extended tools disabled and the rect-only tie-break order, the
  // solver must reproduce partition::decide_frame exactly.
  const p::FrameGeometry frames[] = {
      geometry_420(64, 64, 64),
      geometry_420(640, 360, 64),
      geometry_420(128, 128, 128),
      geometry_420(520, 128, 128),
      geometry_420(8, 8, 64),
  };
  const pe::Tools off = ext_tools(false, false);
  pe::DecisionOptions options;
  options.tie_break = pe::TieBreakOrder::kRectCompatOrder;
  p::DecisionOptions base_options;
  base_options.max_depth = options.max_depth;

  for (unsigned seed = 0; seed < 6; ++seed) {
    auto base_cost = [seed](const p::PartitionNodeInfo& info) {
      // Reuse the ext random model with the same inputs.
      pe::NodeInfo converted;
      converted.mi_row = info.mi_row;
      converted.mi_col = info.mi_col;
      converted.size = info.size;
      converted.depth = info.depth;
      const pe::CostFunction model = random_cost(seed);
      const pe::NodeCosts costs = model(converted);
      p::NodeCosts base;
      base.leaf = costs.leaf;
      base.none_signal = costs.none_signal;
      base.split_signal = costs.split_signal;
      base.horz_signal = costs.horz_signal;
      base.vert_signal = costs.vert_signal;
      return base;
    };
    // Ext-side cost must be derived from the identical base values.
    auto ext_cost = [base_cost](const pe::NodeInfo& info) {
      p::PartitionNodeInfo converted;
      converted.mi_row = info.mi_row;
      converted.mi_col = info.mi_col;
      converted.size = info.size;
      converted.depth = info.depth;
      return pe::from_partition_costs(base_cost(converted));
    };

    for (const p::FrameGeometry& frame : frames) {
      const p::FrameDecision base = p::decide_frame(frame, base_cost, base_options);
      const pe::FrameDecision ext =
          pe::decide_frame(frame, ext_cost, off, options);
      CHECK_EQ(base.status == p::DecisionStatus::kOk, 1);
      CHECK(ext.status == pe::DecisionStatus::kOk);
      std::string base_sig;
      for (const p::PartitionNode& sb : base.superblocks) {
        base_sig += base_signature(sb);
      }
      CHECK_STR(frame_signature(ext), base_sig);
      if (frame_signature(ext) != base_sig) {
        std::cerr << "seed " << seed << " frame " << frame.width << 'x'
                  << frame.height << " sb " << frame.sb_size << '\n';
      }
    }
  }

  // All-equal costs exercise the tie-break order explicitly.
  auto equal_base = [](const p::PartitionNodeInfo&) {
    p::NodeCosts costs;
    costs.leaf = 100.0;
    costs.none_signal = 0.0;
    costs.split_signal = 0.0;
    costs.horz_signal = 0.0;
    costs.vert_signal = 0.0;
    return costs;
  };
  auto equal_ext = [](const pe::NodeInfo&) {
    pe::NodeCosts costs;
    costs.leaf = 100.0;
    costs.none_signal = 0.0;
    costs.split_signal = 0.0;
    costs.horz_signal = 0.0;
    costs.vert_signal = 0.0;
    costs.horz3_signal = pe::kInfiniteCost;
    costs.vert3_signal = pe::kInfiniteCost;
    costs.horz4a_signal = pe::kInfiniteCost;
    costs.horz4b_signal = pe::kInfiniteCost;
    costs.vert4a_signal = pe::kInfiniteCost;
    costs.vert4b_signal = pe::kInfiniteCost;
    return costs;
  };
  for (const p::FrameGeometry& frame : frames) {
    const p::FrameDecision base =
        p::decide_frame(frame, equal_base, base_options);
    const pe::FrameDecision ext =
        pe::decide_frame(frame, equal_ext, off, options);
    std::string base_sig;
    for (const p::PartitionNode& sb : base.superblocks) {
      base_sig += base_signature(sb);
    }
    CHECK_STR(frame_signature(ext), base_sig);
  }
}

void test_validation_rejects() {
  const pe::Tools tools = ext_tools(true, true);
  const p::FrameGeometry frame = geometry_420(64, 64, 64);
  pe::CostFunction costs = choose_partition([](const pe::NodeInfo& info) {
    if (info.depth == 0) return kHorz3;
    if (info.depth == 1 && info.size == BlockSize::k32x32) return kVert3;
    return kNone;
  });
  const pe::PartitionDecision decision =
      pe::decide_superblock(frame, 0, 0, costs, tools, pe::DecisionOptions());
  CHECK(decision.status == pe::DecisionStatus::kOk);
  std::string error;
  CHECK(pe::validate_tree(frame, tools, decision.root, error));

  {
    pe::PartitionNode bad = decision.root;
    bad.partition = kSplit;
    CHECK(!pe::validate_tree(frame, tools, bad, error));
  }
  {
    pe::PartitionNode bad = decision.root;
    bad.children[1].size = BlockSize::k64x32;
    CHECK(!pe::validate_tree(frame, tools, bad, error));
  }
  {
    pe::PartitionNode bad = decision.root;
    bad.allowed.allowed[static_cast<int>(kHorz3)] = false;
    bad.allowed.count -= 1;
    CHECK(!pe::validate_tree(frame, tools, bad, error));
  }
  {
    pe::PartitionNode bad = decision.root;
    bad.forced = true;
    CHECK(!pe::validate_tree(frame, tools, bad, error));
  }
  {
    pe::PartitionNode bad = decision.root;
    bad.children.clear();
    CHECK(!pe::validate_tree(frame, tools, bad, error));
  }
  {
    pe::PartitionNode bad = decision.root;
    bad.children[1].chroma.size = BlockSize::k8x8;
    CHECK(!pe::validate_tree(frame, tools, bad, error));
  }
  {
    pe::PartitionNode bad = decision.root;
    bad.boundary.has_rows = !bad.boundary.has_rows;
    CHECK(!pe::validate_tree(frame, tools, bad, error));
  }
  {
    pe::PartitionNode bad = decision.root;
    bad.children[0].has_chroma = !bad.children[0].has_chroma;
    CHECK(!pe::validate_tree(frame, tools, bad, error));
  }
}

void test_extended_leaf_chroma() {
  // Force HORZ_3@8x32 via a 32x32 root path: 64x64 -> VERT(32x64) ->
  // HORZ at 32x64 gives 32x32; VERT at 32x32 gives 16x32; HORZ_3 there is
  // 16x32, not 8x32.  Build the 8x32 case directly with enumerate_children
  // (above) and check the leaf-level chroma through a decoded tree instead.
  //
  // 64x64 -> SPLIT is illegal, so use 128x128 -> SPLIT -> 64x64 -> HORZ_3?
  // HORZ_3 at 64x64 does not hit the middle special case.  The 8x32 / 32x8
  // special cases are instead exercised in ext_syntax_test.cpp by a direct
  // node walk.
  CHECK(true);
}

}  // namespace

int main() {
  test_enum_and_tables();
  test_implied();
  test_allowed_sets();
  test_chroma_rules();
  test_child_geometry();
  test_decision_ext_trees();
  test_decision_policy();
  test_rect_compat();
  test_validation_rejects();
  test_extended_leaf_chroma();

  if (g_failures != 0) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "partition_ext_test: all checks passed\n";
  return 0;
}
