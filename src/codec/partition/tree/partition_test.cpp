// Tests for the CPU AV2 partition-tree reference.
//
// Run with ./build_and_test.sh.  No bitstream, decoder, or entropy integration
// is exercised here; the tests cover geometry, spec validity, boundary rules,
// deterministic decisions, and partial-superblock edge cases.

#include "partition.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace p = av2enc::partition;

namespace {

int g_failures = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      ++g_failures;                                                          \
      std::cerr << __FILE__ << ':' << __LINE__                               \
                << ": CHECK failed: " #condition << '\n';                    \
    }                                                                        \
  } while (false)

#define CHECK_EQ_INT(actual, expected)                                       \
  do {                                                                       \
    const long long actual_value = static_cast<long long>(actual);           \
    const long long expected_value = static_cast<long long>(expected);       \
    if (actual_value != expected_value) {                                    \
      ++g_failures;                                                          \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK_EQ failed: "      \
                << #actual << " = " << actual_value << ", expected "         \
                << expected_value << '\n';                                   \
    }                                                                        \
  } while (false)

p::CostFunction none_only_cost(double leaf_cost) {
  return [leaf_cost](const p::PartitionNodeInfo&) {
    p::NodeCosts costs;
    costs.leaf = leaf_cost;
    costs.none_signal = 0.0;
    costs.split_signal = p::kInfiniteCost;
    costs.horz_signal = p::kInfiniteCost;
    costs.vert_signal = p::kInfiniteCost;
    return costs;
  };
}

// Leaf cost proportional to area squared.  Splitting always reduces the total,
// so this forces the tree to the smallest blocks the spec and options allow.
p::CostFunction area_squared_cost() {
  return [](const p::PartitionNodeInfo& info) {
    const double width = static_cast<double>(p::block_width(info.size));
    const double height = static_cast<double>(p::block_height(info.size));
    const double area = width * height;
    p::NodeCosts costs;
    costs.leaf = area * area;
    costs.none_signal = 0.0;
    costs.split_signal = 0.0;
    costs.horz_signal = 0.0;
    costs.vert_signal = 0.0;
    return costs;
  };
}

p::FrameGeometry geometry_420(int width, int height, int sb_size) {
  p::FrameGeometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = sb_size;
  return geometry;
}

void append_signature(const p::PartitionNode& node, std::string& out) {
  out += p::block_size_name(node.size);
  out += ':';
  out += std::to_string(node.mi_row);
  out += ',';
  out += std::to_string(node.mi_col);
  out += ':';
  out += p::partition_name(node.partition);
  out += ':';
  out += node.forced ? 'F' : 'S';
  out += '(';
  for (const p::PartitionNode& child : node.children) {
    append_signature(child, out);
  }
  out += ')';
}

std::string tree_signature(const p::PartitionNode& node) {
  std::string out;
  append_signature(node, out);
  return out;
}

void test_block_tables() {
  CHECK_EQ_INT(p::block_width(p::BlockSize::k4x4), 4);
  CHECK_EQ_INT(p::block_width(p::BlockSize::k64x64), 64);
  CHECK_EQ_INT(p::block_width(p::BlockSize::k128x128), 128);
  CHECK_EQ_INT(p::block_height(p::BlockSize::k16x32), 32);
  CHECK_EQ_INT(p::num_4x4_blocks_wide(p::BlockSize::k64x32), 16);
  CHECK_EQ_INT(p::num_4x4_blocks_high(p::BlockSize::k64x32), 8);
  CHECK_EQ_INT(p::mi_width_log2(p::BlockSize::k128x128), 5);
  CHECK_EQ_INT(p::mi_height_log2(p::BlockSize::k32x64), 4);
  CHECK(p::is_square(p::BlockSize::k8x8));
  CHECK(!p::is_square(p::BlockSize::k8x4));
  CHECK(!p::is_valid_block_size(p::BlockSize::kInvalid));
  CHECK(p::superblock_size(64) == p::BlockSize::k64x64);
  CHECK(p::superblock_size(128) == p::BlockSize::k128x128);
  CHECK(p::superblock_size(32) == p::BlockSize::kInvalid);
  CHECK(std::string(p::block_size_name(p::BlockSize::k32x16)) == "32x16");
  CHECK(std::string(p::block_size_name(p::BlockSize::kInvalid)) == "invalid");

  CHECK_EQ_INT(p::spec_partition_index(p::PartitionType::kNone), 0);
  CHECK_EQ_INT(p::spec_partition_index(p::PartitionType::kHorz), 1);
  CHECK_EQ_INT(p::spec_partition_index(p::PartitionType::kVert), 2);
  CHECK_EQ_INT(p::spec_partition_index(p::PartitionType::kSplit), 9);
  CHECK(std::string(p::partition_name(p::PartitionType::kSplit)) == "SPLIT");

  // Table 9.x: Partition_Subsize.
  CHECK(p::partition_subsize(p::PartitionType::kNone, p::BlockSize::k64x64) ==
        p::BlockSize::k64x64);
  CHECK(p::partition_subsize(p::PartitionType::kHorz, p::BlockSize::k8x8) ==
        p::BlockSize::k8x4);
  CHECK(p::partition_subsize(p::PartitionType::kVert, p::BlockSize::k8x8) ==
        p::BlockSize::k4x8);
  CHECK(p::partition_subsize(p::PartitionType::kHorz, p::BlockSize::k4x8) ==
        p::BlockSize::k4x4);
  CHECK(p::partition_subsize(p::PartitionType::kVert, p::BlockSize::k4x8) ==
        p::BlockSize::kInvalid);
  CHECK(p::partition_subsize(p::PartitionType::kHorz, p::BlockSize::k128x128) ==
        p::BlockSize::k128x64);
  CHECK(p::partition_subsize(p::PartitionType::kVert, p::BlockSize::k128x128) ==
        p::BlockSize::k64x128);
  CHECK(p::partition_subsize(p::PartitionType::kHorz, p::BlockSize::k64x128) ==
        p::BlockSize::k64x64);
  CHECK(p::partition_subsize(p::PartitionType::kVert, p::BlockSize::k128x64) ==
        p::BlockSize::k64x64);
  CHECK(p::partition_subsize(p::PartitionType::kHorz, p::BlockSize::k4x4) ==
        p::BlockSize::kInvalid);
  CHECK(p::partition_subsize(p::PartitionType::kVert, p::BlockSize::k4x4) ==
        p::BlockSize::kInvalid);

  // AV2-specific: SPLIT is only defined for 128x128 and 256x256.
  CHECK(p::partition_subsize(p::PartitionType::kSplit, p::BlockSize::k8x8) ==
        p::BlockSize::kInvalid);
  CHECK(p::partition_subsize(p::PartitionType::kSplit, p::BlockSize::k32x32) ==
        p::BlockSize::kInvalid);
  CHECK(p::partition_subsize(p::PartitionType::kSplit, p::BlockSize::k64x64) ==
        p::BlockSize::kInvalid);
  CHECK(p::partition_subsize(p::PartitionType::kSplit, p::BlockSize::k128x128) ==
        p::BlockSize::k64x64);
  CHECK(p::partition_subsize(p::PartitionType::kSplit, p::BlockSize::k256x256) ==
        p::BlockSize::k128x128);

  p::RectTypeImplied rect = p::rect_type_implied_by_bsize(p::BlockSize::k4x8);
  CHECK(rect.implied);
  CHECK(rect.partition == p::PartitionType::kHorz);
  rect = p::rect_type_implied_by_bsize(p::BlockSize::k8x4);
  CHECK(rect.implied);
  CHECK(rect.partition == p::PartitionType::kVert);
  CHECK(!p::rect_type_implied_by_bsize(p::BlockSize::k64x64).implied);

  CHECK_EQ_INT(p::partition_size_adjust(p::BlockSize::k32x32), 2);
  CHECK_EQ_INT(p::partition_size_adjust_rect_type(p::BlockSize::k32x32), 3);
  CHECK_EQ_INT(p::partition_size_adjust(p::BlockSize::k128x128), 6);
  CHECK_EQ_INT(p::partition_size_adjust_rect_type(p::BlockSize::k128x128), 9);
}

void test_geometry() {
  p::FrameGeometry geometry = geometry_420(640, 360, 64);
  std::string error;
  CHECK(p::validate_geometry(geometry, error));
  CHECK_EQ_INT(p::mi_cols(geometry), 160);
  CHECK_EQ_INT(p::mi_rows(geometry), 90);
  CHECK_EQ_INT(p::sb_cols(geometry), 10);
  CHECK_EQ_INT(p::sb_rows(geometry), 6);
  CHECK_EQ_INT(p::sb_size_mi(geometry), 16);

  geometry.sb_size = 128;
  CHECK(p::validate_geometry(geometry, error));
  CHECK_EQ_INT(p::sb_cols(geometry), 5);
  CHECK_EQ_INT(p::sb_rows(geometry), 3);

  geometry = geometry_420(1920, 1080, 64);
  CHECK_EQ_INT(p::mi_cols(geometry), 480);
  CHECK_EQ_INT(p::mi_rows(geometry), 270);
  CHECK_EQ_INT(p::sb_cols(geometry), 30);
  CHECK_EQ_INT(p::sb_rows(geometry), 17);
  geometry.sb_size = 128;
  CHECK_EQ_INT(p::sb_cols(geometry), 15);
  CHECK_EQ_INT(p::sb_rows(geometry), 9);

  geometry = geometry_420(520, 128, 128);
  CHECK_EQ_INT(p::mi_cols(geometry), 130);
  CHECK_EQ_INT(p::mi_rows(geometry), 32);
  CHECK_EQ_INT(p::sb_cols(geometry), 5);
  CHECK_EQ_INT(p::sb_rows(geometry), 1);

  // §5.18.4.4 rounds MiCols/MiRows up to an even number of 4x4 units.
  p::FrameGeometry odd = geometry_420(4, 4, 64);
  CHECK_EQ_INT(p::mi_cols(odd), 2);
  CHECK_EQ_INT(p::mi_rows(odd), 2);
  odd = geometry_420(9, 17, 64);
  CHECK_EQ_INT(p::mi_cols(odd), 4);   // 2 * ((9 + 7) >> 3)
  CHECK_EQ_INT(p::mi_rows(odd), 6);   // 2 * ((17 + 7) >> 3)
  odd = geometry_420(12, 12, 64);
  CHECK_EQ_INT(p::mi_cols(odd), 4);

  p::FrameGeometry bad = geometry_420(0, 10, 64);
  CHECK(!p::validate_geometry(bad, error));
  bad = geometry_420(10, 10, 32);
  CHECK(!p::validate_geometry(bad, error));
  bad = geometry_420(10, 10, 64);
  bad.chroma_subsampling_x = true;
  bad.chroma_subsampling_y = false;
  CHECK(!p::validate_geometry(bad, error));
  bad = geometry_420(10, 10, 64);
  bad.max_pb_aspect_ratio = 3;
  CHECK(!p::validate_geometry(bad, error));

  geometry = geometry_420(640, 360, 64);
  p::BlockRect rect =
      p::block_rect(geometry, 80, 128, p::BlockSize::k64x64);
  CHECK_EQ_INT(rect.x, 512);
  CHECK_EQ_INT(rect.y, 320);
  CHECK_EQ_INT(rect.width, 64);
  CHECK_EQ_INT(rect.height, 64);
  CHECK_EQ_INT(rect.visible_width, 64);
  CHECK_EQ_INT(rect.visible_height, 40);

  geometry = geometry_420(1920, 1080, 128);
  rect = p::block_rect(geometry, 256, 448, p::BlockSize::k128x128);
  CHECK_EQ_INT(rect.x, 1792);
  CHECK_EQ_INT(rect.y, 1024);
  CHECK_EQ_INT(rect.visible_width, 128);
  CHECK_EQ_INT(rect.visible_height, 56);
}

void test_boundary_implied() {
  const p::FrameGeometry hd = geometry_420(640, 360, 64);
  p::PartitionBoundary boundary =
      p::partition_implied_at_boundary(hd, 0, 0, p::BlockSize::k64x64);
  CHECK(boundary.has_rows);
  CHECK(boundary.has_cols);
  CHECK(!boundary.implied);

  boundary =
      p::partition_implied_at_boundary(hd, 80, 128, p::BlockSize::k64x64);
  CHECK(boundary.has_rows);   // 80 + 8 = 88 < 90
  CHECK(boundary.has_cols);   // 128 + 8 = 136 < 160
  CHECK(!boundary.implied);

  // A 64x32 wider-than-high block at the bottom edge is implied VERT.
  boundary =
      p::partition_implied_at_boundary(hd, 88, 128, p::BlockSize::k64x32);
  CHECK(!boundary.has_rows);  // 88 + 4 = 92 >= 90
  CHECK(boundary.has_cols);
  CHECK(boundary.implied);
  CHECK(boundary.implied_partition == p::PartitionType::kVert);

  // A 32x16 block at the same origin has no implied partition because its
  // quarter-height test still lands inside the frame.
  boundary =
      p::partition_implied_at_boundary(hd, 88, 128, p::BlockSize::k32x16);
  CHECK(!boundary.has_rows);
  CHECK(boundary.has_cols);
  CHECK(!boundary.implied);

  boundary =
      p::partition_implied_at_boundary(hd, 86, 128, p::BlockSize::k32x16);
  CHECK(boundary.has_rows);
  CHECK(!boundary.implied);

  // 1080p with 128x128 superblocks: the last row is partial.
  const p::FrameGeometry uhd = geometry_420(1920, 1080, 128);
  boundary =
      p::partition_implied_at_boundary(uhd, 256, 448, p::BlockSize::k128x128);
  CHECK(!boundary.has_rows);  // 256 + 16 = 272 >= 270
  CHECK(boundary.has_cols);
  CHECK(boundary.implied);
  CHECK(boundary.implied_partition == p::PartitionType::kHorz);

  // The same origin with a 64x64 block is inside enough to code NONE.
  const p::FrameGeometry uhd64 = geometry_420(1920, 1080, 64);
  boundary =
      p::partition_implied_at_boundary(uhd64, 256, 448, p::BlockSize::k64x64);
  CHECK(boundary.has_rows);  // 256 + 8 = 264 < 270
  CHECK(!boundary.implied);

  // Partial right edge implies VERT for a square block.
  const p::FrameGeometry narrow = geometry_420(520, 128, 128);
  boundary =
      p::partition_implied_at_boundary(narrow, 0, 128, p::BlockSize::k128x128);
  CHECK(boundary.has_rows);
  CHECK(!boundary.has_cols);  // 128 + 16 = 144 >= 130
  CHECK(boundary.implied);
  CHECK(boundary.implied_partition == p::PartitionType::kVert);

  // partition_implied() adds the 4x4 / thin-block PARTITION_NONE cases.
  boundary = p::partition_implied(hd, 10, 10, p::BlockSize::k4x4);
  CHECK(boundary.implied);
  CHECK(boundary.implied_partition == p::PartitionType::kNone);

  boundary = p::partition_implied(hd, 10, 10, p::BlockSize::k4x32);
  CHECK(boundary.implied);
  CHECK(boundary.implied_partition == p::PartitionType::kNone);

  boundary = p::partition_implied(hd, 10, 10, p::BlockSize::k8x64);
  CHECK(boundary.implied);
  CHECK(boundary.implied_partition == p::PartitionType::kNone);

  // Non-thin blocks inside the frame are not implied.
  boundary = p::partition_implied(hd, 8, 8, p::BlockSize::k32x8);
  CHECK(!boundary.implied);
}

void test_allowed_partitions() {
  const p::FrameGeometry hd64 = geometry_420(640, 360, 64);
  p::PartitionSet allowed =
      p::init_allowed_partitions(hd64, 0, 0, p::BlockSize::k64x64, true, false);
  CHECK_EQ_INT(allowed.count, 3);
  CHECK(allowed.is_allowed(p::PartitionType::kNone));
  CHECK(allowed.is_allowed(p::PartitionType::kHorz));
  CHECK(allowed.is_allowed(p::PartitionType::kVert));
  CHECK(!allowed.is_allowed(p::PartitionType::kSplit));

  const p::FrameGeometry hd128 = geometry_420(640, 360, 128);
  allowed = p::init_allowed_partitions(hd128, 0, 0, p::BlockSize::k128x128,
                                       true, false);
  CHECK_EQ_INT(allowed.count, 4);
  CHECK(allowed.is_allowed(p::PartitionType::kSplit));

  allowed =
      p::init_allowed_partitions(hd64, 0, 0, p::BlockSize::k8x8, true, false);
  CHECK(allowed.is_allowed(p::PartitionType::kHorz));
  CHECK(allowed.is_allowed(p::PartitionType::kVert));
  CHECK(!allowed.is_allowed(p::PartitionType::kSplit));

  allowed =
      p::init_allowed_partitions(hd64, 0, 0, p::BlockSize::k4x4, true, false);
  CHECK_EQ_INT(allowed.count, 1);
  CHECK(allowed.is_allowed(p::PartitionType::kNone));

  allowed =
      p::init_allowed_partitions(hd64, 0, 0, p::BlockSize::k4x8, true, false);
  CHECK(allowed.is_allowed(p::PartitionType::kNone));
  CHECK(allowed.is_allowed(p::PartitionType::kHorz));
  CHECK(!allowed.is_allowed(p::PartitionType::kVert));

  allowed =
      p::init_allowed_partitions(hd128, 0, 0, p::BlockSize::k64x128, true,
                                 false);
  CHECK(allowed.is_allowed(p::PartitionType::kNone));
  CHECK(allowed.is_allowed(p::PartitionType::kHorz));
  CHECK(!allowed.is_allowed(p::PartitionType::kVert));

  allowed =
      p::init_allowed_partitions(hd128, 0, 0, p::BlockSize::k128x64, true,
                                 false);
  CHECK(allowed.is_allowed(p::PartitionType::kNone));
  CHECK(!allowed.is_allowed(p::PartitionType::kHorz));
  CHECK(allowed.is_allowed(p::PartitionType::kVert));

  // Bottom edge: PARTITION_NONE is forbidden; HORZ and VERT remain legal.
  allowed = p::init_allowed_partitions(hd64, 0, 0, p::BlockSize::k64x64, true,
                                       false);
  CHECK(allowed.is_allowed(p::PartitionType::kNone));
  const p::FrameGeometry short_frame = geometry_420(64, 32, 64);
  allowed = p::init_allowed_partitions(short_frame, 0, 0,
                                       p::BlockSize::k64x64, true, false);
  CHECK(!allowed.is_allowed(p::PartitionType::kNone));
  CHECK(allowed.is_allowed(p::PartitionType::kHorz));
  CHECK(allowed.is_allowed(p::PartitionType::kVert));

  // A 64x32 block at the bottom edge cannot code NONE.
  allowed = p::init_allowed_partitions(hd64, 88, 128, p::BlockSize::k64x32,
                                       true, false);
  CHECK(!allowed.is_allowed(p::PartitionType::kNone));
  CHECK(allowed.is_allowed(p::PartitionType::kHorz));
  CHECK(allowed.is_allowed(p::PartitionType::kVert));

  // 4:2:0 chroma offset lets 8x8 split into 8x4 / 4x8 without a chroma issue.
  CHECK(p::is_partition_allowed(hd64, 0, 0, p::BlockSize::k8x8,
                                p::PartitionType::kHorz, true, false));
  CHECK(p::is_partition_allowed(hd64, 0, 0, p::BlockSize::k8x8,
                                p::PartitionType::kVert, true, false));
  CHECK(p::is_partition_allowed(hd64, 0, 0, p::BlockSize::k4x8,
                                p::PartitionType::kHorz, true, false));

  // Monochrome relaxes the chroma checks but not the geometry.
  CHECK(p::is_partition_allowed(hd64, 0, 0, p::BlockSize::k8x8,
                                p::PartitionType::kHorz, false, false));

  CHECK(p::chroma_plane_residual_size(hd64, p::BlockSize::k8x8) ==
        p::BlockSize::k4x4);
  CHECK(p::chroma_plane_residual_size(hd64, p::BlockSize::k4x4) ==
        p::BlockSize::k4x4);
}

void test_neighbor_availability() {
  const p::FrameGeometry hd = geometry_420(640, 360, 64);
  p::NeighborAvailability n =
      p::neighbor_availability(hd, 0, 0, p::BlockSize::k64x64);
  CHECK(!n.left);
  CHECK(!n.above);
  CHECK(!n.above_left);

  n = p::neighbor_availability(hd, 16, 16, p::BlockSize::k64x64);
  CHECK(n.left);
  CHECK(n.above);
  CHECK(n.above_left);
  CHECK(n.above_right);   // (15, 32) is inside
  CHECK(n.below_left);    // (32, 15) is inside

  n = p::neighbor_availability(hd, 0, 128, p::BlockSize::k64x64);
  CHECK(n.left);          // (0, 127) is inside
  CHECK(!n.above);
  CHECK(!n.above_right);
  CHECK(n.below_left);    // (16, 127) is inside

  n = p::neighbor_availability(hd, 80, 128, p::BlockSize::k64x64);
  CHECK(n.left);
  CHECK(n.above);
  CHECK(n.above_right);   // (79, 144) is inside
  CHECK(!n.below_left);   // (96, 127) is outside MiRows=90
}

void test_contexts() {
  const p::FrameGeometry geometry = geometry_420(64, 64, 64);
  p::MiSizeGrid grid(geometry);
  CHECK_EQ_INT(grid.rows(), 16);
  CHECK_EQ_INT(grid.cols(), 16);
  CHECK(grid.at(-1, 0) == p::BlockSize::kInvalid);
  CHECK(grid.at(0, 0) == p::BlockSize::kInvalid);

  grid.fill_block(0, 0, p::BlockSize::k32x32);
  CHECK(grid.at(7, 7) == p::BlockSize::k32x32);
  CHECK(grid.at(8, 8) == p::BlockSize::kInvalid);

  p::PartitionContexts contexts =
      p::partition_contexts(geometry, 0, 8, p::BlockSize::k32x32, grid);
  // left is 32x32 (log2 height 3 == bsh 3), above is unavailable.
  CHECK_EQ_INT(contexts.do_split, 2 * 4);
  CHECK_EQ_INT(contexts.rect_type, 3 * 4);
  CHECK_EQ_INT(contexts.do_square_split, 0);

  grid.reset(geometry);
  grid.fill_block(0, 0, p::BlockSize::k8x8);
  contexts = p::partition_contexts(geometry, 0, 2, p::BlockSize::k32x32, grid);
  // 8x8 is smaller than 32x32, so the left context is 1.
  CHECK_EQ_INT(contexts.do_split, 2 * 4 + 2);
  CHECK_EQ_INT(contexts.rect_type, 3 * 4 + 2);
  CHECK_EQ_INT(contexts.do_square_split, 2);

  grid.reset(geometry);
  grid.fill_block(0, 0, p::BlockSize::k16x16);
  contexts = p::partition_contexts(geometry, 4, 0, p::BlockSize::k32x32, grid);
  // above is 16x16 (log2 width 2 < bsw 3), left is unavailable.
  CHECK_EQ_INT(contexts.do_split, 2 * 4 + 1);
  CHECK_EQ_INT(contexts.do_square_split, 1);
}

void test_decision_basics() {
  const p::FrameGeometry geometry = geometry_420(64, 64, 64);
  p::DecisionOptions options;
  options.min_block_size_px = 4;

  p::PartitionDecision decision =
      p::decide_superblock(geometry, 0, 0, none_only_cost(1.0), options);
  CHECK(decision.status == p::DecisionStatus::kOk);
  CHECK(decision.root.partition == p::PartitionType::kNone);
  CHECK_EQ_INT(decision.nodes, 1);
  CHECK_EQ_INT(decision.leaves, 1);
  CHECK(decision.cost == 1.0);

  std::string error;
  CHECK(p::validate_tree(geometry, decision.root, error));

  // Area-squared forces a deep tree.  At 64x64 SPLIT is illegal, so the first
  // choice is HORZ (ties are broken NONE, SPLIT, HORZ, VERT).
  decision =
      p::decide_superblock(geometry, 0, 0, area_squared_cost(), options);
  CHECK(decision.status == p::DecisionStatus::kOk);
  CHECK(decision.root.partition == p::PartitionType::kHorz);
  CHECK(!decision.root.forced);
  CHECK(decision.leaves > 1);
  CHECK(p::validate_tree(geometry, decision.root, error));
  std::vector<p::LeafBlock> leaves;
  p::collect_leaves(decision.root, leaves);
  CHECK_EQ_INT(leaves.size(), decision.leaves);
  for (const p::LeafBlock& leaf : leaves) {
    CHECK(p::block_width(leaf.size) >= 4);
    CHECK(p::block_height(leaf.size) >= 4);
  }

  // Policy floor of 32x32 stops the same cost model at 32x32 leaves.
  options.min_block_size_px = 32;
  decision =
      p::decide_superblock(geometry, 0, 0, area_squared_cost(), options);
  CHECK(decision.status == p::DecisionStatus::kOk);
  CHECK(p::validate_tree(geometry, decision.root, error));
  leaves.clear();
  p::collect_leaves(decision.root, leaves);
  CHECK_EQ_INT(leaves.size(), 4);
  for (const p::LeafBlock& leaf : leaves) {
    CHECK(leaf.size == p::BlockSize::k32x32);
  }

  // Depth 0 forces NONE when a leaf cost exists.
  options = p::DecisionOptions{};
  options.max_depth = 0;
  decision =
      p::decide_superblock(geometry, 0, 0, none_only_cost(2.0), options);
  CHECK(decision.status == p::DecisionStatus::kOk);
  CHECK(decision.root.partition == p::PartitionType::kNone);
  CHECK(decision.cost == 2.0);

  // Depth 0 with no leaf cost has no valid tree.
  auto split_only_cost = [](const p::PartitionNodeInfo&) {
    p::NodeCosts costs;
    costs.leaf = p::kInfiniteCost;
    costs.split_signal = 0.0;
    costs.horz_signal = 0.0;
    costs.vert_signal = 0.0;
    return costs;
  };
  decision = p::decide_superblock(geometry, 0, 0, split_only_cost, options);
  CHECK(decision.status == p::DecisionStatus::kNoValidPartition);

  // Rectangular splits disabled: at a 64x64 root only NONE remains.
  options = p::DecisionOptions{};
  options.allow_rect = false;
  decision = p::decide_superblock(geometry, 0, 0, none_only_cost(3.0), options);
  CHECK(decision.status == p::DecisionStatus::kOk);
  CHECK(decision.root.partition == p::PartitionType::kNone);

  // Invalid costs are rejected.
  auto negative_cost = [](const p::PartitionNodeInfo&) {
    p::NodeCosts costs;
    costs.leaf = -1.0;
    return costs;
  };
  decision = p::decide_superblock(geometry, 0, 0, negative_cost, options);
  CHECK(decision.status == p::DecisionStatus::kInvalidCost);

  auto nan_cost = [](const p::PartitionNodeInfo&) {
    p::NodeCosts costs;
    costs.leaf = std::numeric_limits<double>::quiet_NaN();
    return costs;
  };
  decision = p::decide_superblock(geometry, 0, 0, nan_cost, options);
  CHECK(decision.status == p::DecisionStatus::kInvalidCost);
}

void test_tie_break() {
  const p::FrameGeometry geometry = geometry_420(128, 128, 128);
  p::DecisionOptions options;
  options.min_block_size_px = 64;

  auto cost = [](const p::PartitionNodeInfo& info) {
    p::NodeCosts costs;
    if (info.size == p::BlockSize::k128x128) {
      costs.leaf = 10.0;
      costs.split_signal = 0.0;
      costs.horz_signal = p::kInfiniteCost;
      costs.vert_signal = p::kInfiniteCost;
    } else {
      costs.leaf = 2.5;
      costs.split_signal = p::kInfiniteCost;
      costs.horz_signal = p::kInfiniteCost;
      costs.vert_signal = p::kInfiniteCost;
    }
    return costs;
  };

  // NONE = 10, SPLIT = 4 * 2.5 = 10.  Tie goes to NONE.
  p::PartitionDecision decision =
      p::decide_superblock(geometry, 0, 0, cost, options);
  CHECK(decision.status == p::DecisionStatus::kOk);
  CHECK(decision.root.partition == p::PartitionType::kNone);
  CHECK(decision.cost == 10.0);

  // Make one quadrant cheaper: SPLIT = 2.0 + 3 * 2.5 = 9.5 < 10.
  auto cheaper_split = [](const p::PartitionNodeInfo& info) {
    p::NodeCosts costs;
    if (info.size == p::BlockSize::k128x128) {
      costs.leaf = 10.0;
      costs.split_signal = 0.0;
      costs.horz_signal = p::kInfiniteCost;
      costs.vert_signal = p::kInfiniteCost;
    } else {
      costs.leaf = (info.mi_row == 0 && info.mi_col == 0) ? 2.0 : 2.5;
      costs.split_signal = p::kInfiniteCost;
      costs.horz_signal = p::kInfiniteCost;
      costs.vert_signal = p::kInfiniteCost;
    }
    return costs;
  };
  decision = p::decide_superblock(geometry, 0, 0, cheaper_split, options);
  CHECK(decision.status == p::DecisionStatus::kOk);
  CHECK(decision.root.partition == p::PartitionType::kSplit);
  CHECK_EQ_INT(decision.root.children.size(), 4);
  CHECK(decision.cost == 9.5);
  CHECK_EQ_INT(decision.root.children[0].mi_row, 0);
  CHECK_EQ_INT(decision.root.children[0].mi_col, 0);
  CHECK_EQ_INT(decision.root.children[1].mi_row, 0);
  CHECK_EQ_INT(decision.root.children[1].mi_col, 16);
  CHECK_EQ_INT(decision.root.children[2].mi_row, 16);
  CHECK_EQ_INT(decision.root.children[2].mi_col, 0);
  CHECK_EQ_INT(decision.root.children[3].mi_row, 16);
  CHECK_EQ_INT(decision.root.children[3].mi_col, 16);
  for (const p::PartitionNode& child : decision.root.children) {
    CHECK(child.size == p::BlockSize::k64x64);
  }
}

void test_implied_override() {
  // 64x32 frame: the 64x64 superblock has only half a row, so HORZ is forced.
  const p::FrameGeometry geometry = geometry_420(64, 32, 64);
  p::DecisionOptions options;

  auto cost = [](const p::PartitionNodeInfo& info) {
    p::NodeCosts costs;
    if (info.size == p::BlockSize::k64x64) {
      costs.leaf = p::kInfiniteCost;
      costs.split_signal = p::kInfiniteCost;
      costs.horz_signal = 0.0;
      costs.vert_signal = 0.0;
    } else if (info.size == p::BlockSize::k64x16) {
      costs.leaf = 100.0;
    } else {
      costs.leaf = 1.0;
    }
    return costs;
  };

  p::PartitionDecision decision =
      p::decide_superblock(geometry, 0, 0, cost, options);
  CHECK(decision.status == p::DecisionStatus::kOk);
  CHECK(decision.root.forced);
  CHECK(decision.root.partition == p::PartitionType::kHorz);
  std::string error;
  CHECK(p::validate_tree(geometry, decision.root, error));

  // A hand-built tree that codes VERT at this boundary must be rejected.
  p::PartitionNode bad;
  bad.mi_row = 0;
  bad.mi_col = 0;
  bad.size = p::BlockSize::k64x64;
  bad.partition = p::PartitionType::kVert;
  bad.depth = 0;
  bad.has_chroma = true;
  bad.chroma_offset = false;
  bad.boundary = p::partition_implied(geometry, 0, 0, p::BlockSize::k64x64);
  bad.allowed = p::init_allowed_partitions(geometry, 0, 0,
                                           p::BlockSize::k64x64, true, false);
  bad.forced = false;
  p::PartitionNode child0;
  child0.mi_row = 0;
  child0.mi_col = 0;
  child0.size = p::BlockSize::k32x64;
  child0.partition = p::PartitionType::kNone;
  child0.depth = 1;
  child0.has_chroma = true;
  child0.chroma_offset = false;
  child0.forced = false;
  bad.children.push_back(child0);
  p::PartitionNode child1 = child0;
  child1.mi_col = 8;
  bad.children.push_back(child1);
  CHECK(!p::validate_tree(geometry, bad, error));
  CHECK(!error.empty());
}

void test_128_split() {
  const p::FrameGeometry geometry = geometry_420(256, 256, 128);
  p::DecisionOptions options;
  options.min_block_size_px = 64;

  auto cost = [](const p::PartitionNodeInfo& info) {
    p::NodeCosts costs;
    if (info.size == p::BlockSize::k128x128) {
      costs.leaf = p::kInfiniteCost;
      costs.split_signal = 0.0;
      costs.horz_signal = p::kInfiniteCost;
      costs.vert_signal = p::kInfiniteCost;
    } else {
      costs.leaf = 1.0;
      costs.split_signal = p::kInfiniteCost;
      costs.horz_signal = p::kInfiniteCost;
      costs.vert_signal = p::kInfiniteCost;
    }
    return costs;
  };

  p::PartitionDecision decision =
      p::decide_superblock(geometry, 0, 0, cost, options);
  CHECK(decision.status == p::DecisionStatus::kOk);
  CHECK(decision.root.partition == p::PartitionType::kSplit);
  CHECK_EQ_INT(decision.root.children.size(), 4);
  std::string error;
  CHECK(p::validate_tree(geometry, decision.root, error));
  std::vector<p::LeafBlock> leaves;
  p::collect_leaves(decision.root, leaves);
  CHECK_EQ_INT(leaves.size(), 4);

  // The whole frame with area-squared: SPLIT at the root, then rectangular
  // splits below because 64x64 cannot signal SPLIT.
  const p::FrameGeometry one_sb = geometry_420(128, 128, 128);
  options = p::DecisionOptions{};
  p::FrameDecision frame =
      p::decide_frame(one_sb, area_squared_cost(), options);
  CHECK(frame.status == p::DecisionStatus::kOk);
  CHECK_EQ_INT(frame.superblocks.size(), 1);
  CHECK(frame.superblocks[0].partition == p::PartitionType::kSplit);
  CHECK(p::validate_frame(one_sb, frame, error));
}

void test_edge_cases() {
  p::DecisionOptions options;
  std::string error;

  // 640x360, 64x64 superblocks: 10x6 SBs with a partial bottom row.
  p::FrameGeometry geometry = geometry_420(640, 360, 64);
  p::FrameDecision first =
      p::decide_frame(geometry, area_squared_cost(), options);
  CHECK(first.status == p::DecisionStatus::kOk);
  CHECK_EQ_INT(first.superblocks.size(), 60);
  CHECK(p::validate_frame(geometry, first, error));
  CHECK(first.leaves > 0);

  // Every 640x360 superblock, including the bottom-right one, can code
  // PARTITION_NONE.  This is the M0 edge behavior the pinned 640x360 stream
  // relies on; VERT is never forced here.
  p::FrameDecision all_none =
      p::decide_frame(geometry_420(640, 360, 64), none_only_cost(1.0), options);
  CHECK(all_none.status == p::DecisionStatus::kOk);
  CHECK_EQ_INT(all_none.superblocks.size(), 60);
  CHECK(p::validate_frame(geometry_420(640, 360, 64), all_none, error));
  for (const p::PartitionNode& root : all_none.superblocks) {
    CHECK(root.partition == p::PartitionType::kNone);
  }

  // Determinism: same inputs produce the identical tree signature.
  p::FrameDecision second =
      p::decide_frame(geometry, area_squared_cost(), options);
  CHECK(second.status == p::DecisionStatus::kOk);
  CHECK_EQ_INT(first.nodes, second.nodes);
  CHECK_EQ_INT(first.leaves, second.leaves);
  for (std::size_t index = 0; index < first.superblocks.size(); ++index) {
    CHECK(tree_signature(first.superblocks[index]) ==
          tree_signature(second.superblocks[index]));
  }

  // 640x360 with 128x128 superblocks: 5x3 SBs.
  geometry = geometry_420(640, 360, 128);
  p::FrameDecision frame =
      p::decide_frame(geometry, area_squared_cost(), options);
  CHECK(frame.status == p::DecisionStatus::kOk);
  CHECK_EQ_INT(frame.superblocks.size(), 15);
  CHECK(p::validate_frame(geometry, frame, error));

  // 1080p, 64x64 superblocks: 30x17 SBs.
  geometry = geometry_420(1920, 1080, 64);
  frame = p::decide_frame(geometry, area_squared_cost(), options);
  CHECK(frame.status == p::DecisionStatus::kOk);
  CHECK_EQ_INT(frame.superblocks.size(), 510);
  CHECK(p::validate_frame(geometry, frame, error));

  // 1080p, 128x128 superblocks: 15x9 SBs, partial bottom row.
  geometry = geometry_420(1920, 1080, 128);
  frame = p::decide_frame(geometry, area_squared_cost(), options);
  CHECK(frame.status == p::DecisionStatus::kOk);
  CHECK_EQ_INT(frame.superblocks.size(), 135);
  CHECK(p::validate_frame(geometry, frame, error));
  // Bottom-left SB at (256, 0) must be forced HORZ by the boundary.
  const p::PartitionNode& bottom_left = frame.superblocks[8 * 15];
  CHECK_EQ_INT(bottom_left.mi_row, 256);
  CHECK_EQ_INT(bottom_left.mi_col, 0);
  CHECK(bottom_left.forced);
  CHECK(bottom_left.partition == p::PartitionType::kHorz);

  // Partial right SB: 520x128 with 128 SB, last column root at mi_col 128.
  geometry = geometry_420(520, 128, 128);
  frame = p::decide_frame(geometry, area_squared_cost(), options);
  CHECK(frame.status == p::DecisionStatus::kOk);
  CHECK_EQ_INT(frame.superblocks.size(), 5);
  CHECK(p::validate_frame(geometry, frame, error));
  const p::PartitionNode& right_edge = frame.superblocks[4];
  CHECK_EQ_INT(right_edge.mi_col, 128);
  CHECK(right_edge.forced);
  CHECK(right_edge.partition == p::PartitionType::kVert);
}

void test_tiny_frame() {
  p::DecisionOptions options;
  std::string error;

  // An 8x8 frame is a 2x2-mi superblock region.  Boundary rules force a
  // chain of splits down to 4x4 PARTITION_NONE leaves.  (MiCols/MiRows are
  // rounded to 8 pixels, so a 4x4 picture also has a 2x2-mi nominal grid;
  // that rounding is covered in test_geometry.)
  const p::FrameGeometry geometry = geometry_420(8, 8, 64);
  p::FrameDecision frame =
      p::decide_frame(geometry, area_squared_cost(), options);
  CHECK(frame.status == p::DecisionStatus::kOk);
  CHECK_EQ_INT(frame.superblocks.size(), 1);
  CHECK(p::validate_frame(geometry, frame, error));
  std::vector<p::LeafBlock> leaves;
  p::collect_leaves(frame.superblocks[0], leaves);
  CHECK_EQ_INT(leaves.size(), 4);
  for (const p::LeafBlock& leaf : leaves) {
    CHECK(leaf.size == p::BlockSize::k4x4);
  }

  // 640x360 bottom-right SB: root 64x64 NONE is legal, and the solver can
  // choose it when the leaf is cheap.
  const p::FrameGeometry bottom_right = geometry_420(640, 360, 64);
  p::PartitionDecision decision = p::decide_superblock(
      bottom_right, 80, 128, none_only_cost(1.0), options);
  CHECK(decision.status == p::DecisionStatus::kOk);
  CHECK(decision.root.partition == p::PartitionType::kNone);
  CHECK(p::validate_tree(bottom_right, decision.root, error));
}

void test_geometry_sweep() {
  // Small deterministic sweep over partial-SB shapes, including dimensions
  // that are not multiples of 8 (MiCols/MiRows rounding) to guard regressions
  // in boundary handling.
  const int widths[] = {8, 16, 40, 72, 136, 200, 13, 29};
  const int heights[] = {8, 24, 72, 40, 88, 120, 29, 13};
  p::DecisionOptions options;
  std::string error;
  for (int sb_size : {64, 128}) {
    for (std::size_t index = 0; index < 8; ++index) {
      const p::FrameGeometry geometry =
          geometry_420(widths[index], heights[index], sb_size);
      p::FrameDecision frame =
          p::decide_frame(geometry, area_squared_cost(), options);
      if (frame.status != p::DecisionStatus::kOk) {
        ++g_failures;
        std::cerr << "geometry sweep " << widths[index] << 'x'
                  << heights[index] << " sb " << sb_size << ": "
                  << frame.error << '\n';
        continue;
      }
      if (!p::validate_frame(geometry, frame, error)) {
        ++g_failures;
        std::cerr << "geometry sweep " << widths[index] << 'x'
                  << heights[index] << " sb " << sb_size << ": " << error
                  << '\n';
      }
    }
  }
}

void test_partial_sb_depth_matrix() {
  // Partial-superblock geometry must never conflict with the M1 policy hints
  // (min_block_size = 8, allow_rect, no SPLIT).  A block whose half-size
  // leaves the frame has no legal PARTITION_NONE, so the solver must keep
  // subdividing even when max_depth or the block floor would normally stop it.
  const int sizes[][2] = {
      {136, 88},  {1280, 720}, {416, 208}, {518, 322}, {532, 368},
      {64, 64},   {640, 360},  {1920, 1080}, {100, 70},
  };
  std::string error;
  for (const auto& size : sizes) {
    const int width = size[0];
    const int height = size[1];
    const p::FrameGeometry geometry = geometry_420(width, height, 64);
    for (int depth = 0; depth <= 8; ++depth) {
      p::DecisionOptions options;
      options.min_block_size_px = 8;
      options.max_depth = depth;
      options.allow_rect = true;
      options.allow_split = false;
      const p::FrameDecision first =
          p::decide_frame(geometry, area_squared_cost(), options);
      if (first.status != p::DecisionStatus::kOk) {
        ++g_failures;
        std::cerr << "partial-SB " << width << 'x' << height << " depth "
                  << depth << ": " << first.error << '\n';
        continue;
      }
      if (!p::validate_frame(geometry, first, error)) {
        ++g_failures;
        std::cerr << "partial-SB " << width << 'x' << height << " depth "
                  << depth << ": " << error << '\n';
      }
      int leaves = 0;
      for (const p::PartitionNode& sb : first.superblocks) {
        std::vector<p::LeafBlock> sb_leaves;
        p::collect_leaves(sb, sb_leaves);
        leaves += static_cast<int>(sb_leaves.size());
        for (const p::LeafBlock& leaf : sb_leaves) {
          // With min_block_size = 8 the interior tree must never reach 4x4.
          CHECK(p::block_width(leaf.size) >= 8);
          CHECK(p::block_height(leaf.size) >= 8);
        }
      }
      CHECK_EQ_INT(leaves, first.leaves);

      // Identical inputs must produce the identical tree.
      const p::FrameDecision second =
          p::decide_frame(geometry, area_squared_cost(), options);
      CHECK(second.status == p::DecisionStatus::kOk);
      CHECK_EQ_INT(second.nodes, first.nodes);
      CHECK_EQ_INT(second.leaves, first.leaves);
      if (second.superblocks.size() == first.superblocks.size()) {
        for (std::size_t index = 0; index < first.superblocks.size(); ++index) {
          CHECK(tree_signature(first.superblocks[index]) ==
                tree_signature(second.superblocks[index]));
        }
      }
    }

    // force_single_block (--still-single) also has to cope with a partial SB,
    // where the spec makes a single root block impossible.
    p::DecisionOptions single;
    single.min_block_size_px = 64;
    single.max_depth = 0;
    single.allow_rect = true;
    single.allow_split = false;
    const p::FrameDecision single_frame =
        p::decide_frame(geometry, area_squared_cost(), single);
    if (single_frame.status != p::DecisionStatus::kOk) {
      ++g_failures;
      std::cerr << "partial-SB " << width << 'x' << height
                << " single: " << single_frame.error << '\n';
    } else {
      CHECK(p::validate_frame(geometry, single_frame, error));
    }
  }
}

void test_min_block_floor_depth8() {
  // Depth 8 is deep enough to reach 4x4, but the default M1 floor is 8.
  const int sizes[][2] = {{64, 64}, {640, 360}, {1920, 1080}};
  std::string error;
  for (const auto& size : sizes) {
    const p::FrameGeometry geometry = geometry_420(size[0], size[1], 64);
    p::DecisionOptions options;
    options.min_block_size_px = 8;
    options.max_depth = 8;
    options.allow_rect = true;
    options.allow_split = false;
    const p::FrameDecision frame =
        p::decide_frame(geometry, area_squared_cost(), options);
    CHECK(frame.status == p::DecisionStatus::kOk);
    CHECK(p::validate_frame(geometry, frame, error));
    bool saw_8x8 = false;
    for (const p::PartitionNode& sb : frame.superblocks) {
      std::vector<p::LeafBlock> leaves;
      p::collect_leaves(sb, leaves);
      for (const p::LeafBlock& leaf : leaves) {
        CHECK(p::block_width(leaf.size) >= 8);
        CHECK(p::block_height(leaf.size) >= 8);
        if (leaf.size == p::BlockSize::k8x8) saw_8x8 = true;
      }
    }
    CHECK(saw_8x8);
  }

  // Lowering the floor to 4 does allow the 4x4 leaves depth 8 can reach.
  const p::FrameGeometry geometry = geometry_420(64, 64, 64);
  p::DecisionOptions options;
  options.min_block_size_px = 4;
  options.max_depth = 8;
  const p::FrameDecision frame =
      p::decide_frame(geometry, area_squared_cost(), options);
  CHECK(frame.status == p::DecisionStatus::kOk);
  std::vector<p::LeafBlock> leaves;
  p::collect_leaves(frame.superblocks[0], leaves);
  bool saw_4x4 = false;
  for (const p::LeafBlock& leaf : leaves) {
    if (leaf.size == p::BlockSize::k4x4) saw_4x4 = true;
  }
  CHECK(saw_4x4);
}

void test_validation_rejects() {
  const p::FrameGeometry geometry = geometry_420(64, 64, 64);
  p::DecisionOptions options;
  std::string error;

  p::PartitionDecision good =
      p::decide_superblock(geometry, 0, 0, area_squared_cost(), options);
  CHECK(good.status == p::DecisionStatus::kOk);
  CHECK(p::validate_tree(geometry, good.root, error));

  // SPLIT is illegal at 64x64.
  p::PartitionNode bad = good.root;
  bad.partition = p::PartitionType::kSplit;
  CHECK(!p::validate_tree(geometry, bad, error));

  // Removing children from a split node is invalid.
  bad = good.root;
  bad.children.clear();
  CHECK(!p::validate_tree(geometry, bad, error));

  // Claiming to be forced when the spec does not force the node.
  bad = good.root;
  bad.forced = true;
  CHECK(!p::validate_tree(geometry, bad, error));

  // Stale spec-derived fields are rejected.
  bad = good.root;
  bad.allowed.count = 0;
  CHECK(!p::validate_tree(geometry, bad, error));
  bad = good.root;
  bad.boundary.implied = !bad.boundary.implied;
  CHECK(!p::validate_tree(geometry, bad, error));

  // A child at the wrong position.
  bad = good.root;
  if (!bad.children.empty()) {
    bad.children[0].mi_col += 1;
    CHECK(!p::validate_tree(geometry, bad, error));
  }

  // Frame validation checks the superblock layout too.
  p::FrameDecision frame =
      p::decide_frame(geometry, area_squared_cost(), options);
  CHECK(frame.status == p::DecisionStatus::kOk);
  CHECK(p::validate_frame(geometry, frame, error));
  frame.superblocks[0].mi_col = 4;
  CHECK(!p::validate_frame(geometry, frame, error));
}

void test_decision_errors() {
  const p::FrameGeometry geometry = geometry_420(64, 64, 64);
  p::DecisionOptions options;

  // Unaligned superblock origins are rejected.
  p::PartitionDecision decision =
      p::decide_superblock(geometry, 4, 0, none_only_cost(1.0), options);
  CHECK(decision.status == p::DecisionStatus::kInvalidGeometry);

  // Empty cost function is rejected by decide_frame.
  p::FrameDecision frame = p::decide_frame(geometry, p::CostFunction(), options);
  CHECK(frame.status == p::DecisionStatus::kInvalidGeometry);

  // Unsupported geometry.
  p::FrameGeometry bad = geometry;
  bad.sb_size = 32;
  frame = p::decide_frame(bad, none_only_cost(1.0), options);
  CHECK(frame.status == p::DecisionStatus::kInvalidGeometry);
}

}  // namespace

int main() {
  test_block_tables();
  test_geometry();
  test_boundary_implied();
  test_allowed_partitions();
  test_neighbor_availability();
  test_contexts();
  test_decision_basics();
  test_tie_break();
  test_implied_override();
  test_128_split();
  test_edge_cases();
  test_tiny_frame();
  test_geometry_sweep();
  test_partial_sb_depth_matrix();
  test_min_block_floor_depth8();
  test_validation_rejects();
  test_decision_errors();

  if (g_failures != 0) {
    std::cerr << "partition_test: " << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "partition_test: PASS\n";
  return 0;
}
