// Tests for the 128x128-superblock reference and minimal stream.
//
// Compile with partition.cpp so every table/boundary/allowed/context result is
// cross-checked against the independent M1 rect-partition module.

#include "test_common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "encoder/m0_cdf_tables.h"
#include "encoder/partition/partition.h"
#include "encoder/sb128/sb128.h"
#include "encoder/sb128/sb128_stream.h"

namespace {

namespace p = av2enc::partition;
using av2enc::BitWriter;
using av2enc::sb128::BlockRect;
using av2enc::sb128::BlockSize;
using av2enc::sb128::Boundary;
using av2enc::sb128::Contexts;
using av2enc::sb128::Geometry;
using av2enc::sb128::PartitionType;

constexpr PartitionType kNone = PartitionType::kNone;
constexpr PartitionType kSplit = PartitionType::kSplit;
constexpr PartitionType kHorz = PartitionType::kHorz;
constexpr PartitionType kVert = PartitionType::kVert;

Geometry make_geometry(int width, int height, int sb_size = 128) {
  Geometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = sb_size;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;
  geometry.max_pb_aspect_ratio = 8;
  return geometry;
}

p::FrameGeometry to_partition(const Geometry& geometry) {
  p::FrameGeometry out;
  out.width = geometry.width;
  out.height = geometry.height;
  out.sb_size = geometry.sb_size;
  out.has_chroma = geometry.has_chroma;
  out.chroma_subsampling_x = geometry.chroma_subsampling_x;
  out.chroma_subsampling_y = geometry.chroma_subsampling_y;
  out.max_pb_aspect_ratio = geometry.max_pb_aspect_ratio;
  return out;
}

void test_block_tables() {
  CHECK(av2enc::sb128::is_valid_block_size(BlockSize::k128x128));
  CHECK(!av2enc::sb128::is_valid_block_size(BlockSize::kInvalid));
  CHECK(av2enc::sb128::block_width(BlockSize::k128x128) == 128);
  CHECK(av2enc::sb128::block_height(BlockSize::k128x128) == 128);
  CHECK(av2enc::sb128::num_4x4_blocks_wide(BlockSize::k128x128) == 32);
  CHECK(av2enc::sb128::num_4x4_blocks_high(BlockSize::k128x128) == 32);
  CHECK(av2enc::sb128::superblock_size(128) == BlockSize::k128x128);
  CHECK(av2enc::sb128::superblock_size(64) == BlockSize::k64x64);
  CHECK(av2enc::sb128::superblock_size(256) == BlockSize::kInvalid);
  CHECK(std::string(av2enc::sb128::block_size_name(BlockSize::k128x128)) ==
        "128x128");
  CHECK(av2enc::sb128::spec_partition_index(kSplit) == 9);
  CHECK(av2enc::sb128::spec_partition_index(kHorz) == 1);
  CHECK(av2enc::sb128::spec_partition_index(kVert) == 2);
  CHECK(av2enc::sb128::spec_partition_index(kNone) == 0);

  CHECK(av2enc::sb128::partition_size_adjust(BlockSize::k128x128) == 6);
  CHECK(av2enc::sb128::partition_size_adjust(BlockSize::k64x64) == 3);
  CHECK(av2enc::sb128::partition_size_adjust_rect_type(
            BlockSize::k128x128) == 9);
  CHECK(av2enc::sb128::partition_size_adjust_rect_type(BlockSize::k64x64) ==
        6);
}

void test_partition_subsize() {
  using av2enc::sb128::partition_subsize;
  // The whole point of a 128 SB: SPLIT is legal at 128x128 and lowers to
  // 64x64, while a 64x64 block cannot signal SPLIT.
  CHECK(partition_subsize(kSplit, BlockSize::k128x128) == BlockSize::k64x64);
  CHECK(partition_subsize(kSplit, BlockSize::k64x64) == BlockSize::kInvalid);
  CHECK(partition_subsize(kHorz, BlockSize::k128x128) == BlockSize::k128x64);
  CHECK(partition_subsize(kVert, BlockSize::k128x128) == BlockSize::k64x128);
  // 128x64 has no 128x32 block, so HORZ is invalid and VERT wins.
  CHECK(partition_subsize(kHorz, BlockSize::k128x64) == BlockSize::kInvalid);
  CHECK(partition_subsize(kVert, BlockSize::k128x64) == BlockSize::k64x64);
  CHECK(partition_subsize(kHorz, BlockSize::k64x128) == BlockSize::k64x64);
  CHECK(partition_subsize(kVert, BlockSize::k64x128) == BlockSize::kInvalid);
  CHECK(partition_subsize(kNone, BlockSize::k128x128) ==
        BlockSize::k128x128);

  const auto horz = av2enc::sb128::rect_type_implied_by_bsize(
      BlockSize::k128x64);
  CHECK(horz.implied && horz.partition == kVert);
  const auto vert = av2enc::sb128::rect_type_implied_by_bsize(
      BlockSize::k64x128);
  CHECK(vert.implied && vert.partition == kHorz);
  CHECK(!av2enc::sb128::rect_type_implied_by_bsize(BlockSize::k128x128)
             .implied);
}

void test_geometry() {
  struct Case {
    int width;
    int height;
    int mi_cols;
    int mi_rows;
    int sb_cols;
    int sb_rows;
  };
  const Case cases[] = {
      {128, 128, 32, 32, 1, 1},
      {256, 256, 64, 64, 2, 2},
      {640, 360, 160, 90, 5, 3},
      {1920, 1080, 480, 270, 15, 9},
      {200, 136, 50, 34, 2, 2},
      {8, 8, 2, 2, 1, 1},
  };
  for (const Case& c : cases) {
    const Geometry geometry = make_geometry(c.width, c.height);
    std::string error;
    CHECK(av2enc::sb128::validate_geometry(geometry, error));
    CHECK(av2enc::sb128::mi_cols(geometry) == c.mi_cols);
    CHECK(av2enc::sb128::mi_rows(geometry) == c.mi_rows);
    CHECK(av2enc::sb128::sb_size_mi(geometry) == 32);
    CHECK(av2enc::sb128::sb_cols(geometry) == c.sb_cols);
    CHECK(av2enc::sb128::sb_rows(geometry) == c.sb_rows);
  }

  // 64 SB remains geometrically representable for cross-checks; only the
  // stream builder insists on 128.
  const Geometry g64 = make_geometry(640, 360, 64);
  CHECK(av2enc::sb128::sb_cols(g64) == 10);
  CHECK(av2enc::sb128::sb_rows(g64) == 6);
  CHECK(av2enc::sb128::cdef_sb_width_log2(64) == 4);
  CHECK(av2enc::sb128::cdef_sb_width_log2(128) == 5);
  CHECK(av2enc::sb128::ccso_luma_size_log2(64) == 6);
  CHECK(av2enc::sb128::ccso_luma_size_log2(128) == 7);
  CHECK(av2enc::sb128::ccso_unit_mi_shift(64) == 4);
  CHECK(av2enc::sb128::ccso_unit_mi_shift(128) == 5);

  std::string error;
  Geometry bad = make_geometry(128, 128);
  bad.sb_size = 256;
  CHECK(!av2enc::sb128::validate_geometry(bad, error));
  bad = make_geometry(128, 128);
  bad.max_pb_aspect_ratio = 3;
  CHECK(!av2enc::sb128::validate_geometry(bad, error));
  bad = make_geometry(128, 128);
  bad.chroma_subsampling_y = false;
  CHECK(!av2enc::sb128::validate_geometry(bad, error));
}

void test_boundary_implied() {
  // 1080p: the last SB row starts at mi 256 with MiRows 270.  The half-block
  // test fails (256 + 16 >= 270), and the square root implies HORZ.
  const Geometry g1080 = make_geometry(1920, 1080);
  const Boundary bottom =
      av2enc::sb128::partition_implied(g1080, 256, 0, BlockSize::k128x128);
  CHECK(!bottom.has_rows);
  CHECK(bottom.has_cols);
  CHECK(bottom.implied && bottom.implied_partition == kHorz);

  // A full root codes partition syntax (no implication).
  const Boundary full =
      av2enc::sb128::partition_implied(g1080, 0, 0, BlockSize::k128x128);
  CHECK(full.has_rows && full.has_cols && !full.implied);

  // 192x136: the right SB column at mi 32 is partial (MiCols 48).  The half
  // width test fails and the square root implies VERT.
  const Geometry g192 = make_geometry(192, 136);
  const Boundary right =
      av2enc::sb128::partition_implied(g192, 0, 32, BlockSize::k128x128);
  CHECK(right.has_rows);
  CHECK(!right.has_cols);
  CHECK(right.implied && right.implied_partition == kVert);

  // 64x64 with fewer than 8 mi rows left: NONE is no longer legal and the
  // implied partition is HORZ.  (640x360's last leaf starts at mi 80 with 10
  // rows left, so it keeps NONE; mi 88 is the forced case.)
  const Geometry g360 = make_geometry(640, 360);
  const Boundary leaf =
      av2enc::sb128::partition_implied(g360, 88, 0, BlockSize::k64x64);
  CHECK(!leaf.has_rows);
  CHECK(leaf.implied && leaf.implied_partition == kHorz);

  // Thin blocks always imply NONE (BLOCK_4X32 and above).
  const Boundary thin =
      av2enc::sb128::partition_implied(g1080, 0, 0, BlockSize::k4x32);
  CHECK(thin.implied && thin.implied_partition == kNone);
}

void test_allowed_and_forced() {
  const Geometry g1080 = make_geometry(1920, 1080);

  auto allowed = av2enc::sb128::init_allowed_partitions(
      g1080, 0, 0, BlockSize::k128x128, true, false);
  CHECK(allowed.count == 4);
  CHECK(allowed.is_allowed(kNone));
  CHECK(allowed.is_allowed(kSplit));
  CHECK(allowed.is_allowed(kHorz));
  CHECK(allowed.is_allowed(kVert));

  // Partial bottom root: NONE fails the half-block test but SPLIT is still in
  // the allowed set.  read_partition returns the implied HORZ first, so the
  // reference tree is forced to HORZ.
  allowed = av2enc::sb128::init_allowed_partitions(
      g1080, 256, 0, BlockSize::k128x128, true, false);
  CHECK(!allowed.is_allowed(kNone));
  CHECK(allowed.is_allowed(kSplit));
  CHECK(allowed.is_allowed(kHorz));

  // 128x64 forbids HORZ by rect_type_implied_by_bsize().
  allowed = av2enc::sb128::init_allowed_partitions(
      g1080, 256, 0, BlockSize::k128x64, true, false);
  CHECK(allowed.is_allowed(kNone));
  CHECK(allowed.is_allowed(kVert));
  CHECK(!allowed.is_allowed(kHorz));
  CHECK(!allowed.is_allowed(kSplit));

  // 64x64 cannot SPLIT.
  allowed = av2enc::sb128::init_allowed_partitions(
      g1080, 0, 0, BlockSize::k64x64, true, false);
  CHECK(!allowed.is_allowed(kSplit));
}

void test_contexts_and_cdf_rows() {
  const Geometry g = make_geometry(640, 360);
  av2enc::sb128::MiSizeGrid grid(g);
  Contexts ctx =
      av2enc::sb128::partition_contexts(g, 0, 0, BlockSize::k128x128, grid);
  CHECK(ctx.do_split == 24);       // size_adjust 6 * 4
  CHECK(ctx.do_square_split == 0); // no left/above
  CHECK(ctx.rect_type == 36);      // rect adjust 9 * 4

  // A 64x64 leaf to the left/above moves the 128 root to context 26/25/27 and
  // the do_square_split context to 2/1/3.
  grid.fill_block(0, 16, BlockSize::k64x64);
  ctx = av2enc::sb128::partition_contexts(g, 0, 32, BlockSize::k128x128, grid);
  CHECK(ctx.do_split == 26);
  CHECK(ctx.do_square_split == 2);

  av2enc::sb128::MiSizeGrid grid2(g);
  grid2.fill_block(16, 0, BlockSize::k64x64);
  ctx = av2enc::sb128::partition_contexts(g, 32, 0, BlockSize::k128x128, grid2);
  CHECK(ctx.do_split == 25);
  CHECK(ctx.do_square_split == 1);

  grid2.fill_block(32, 16, BlockSize::k64x64);  // left neighbour of (32,32)
  grid2.fill_block(16, 32, BlockSize::k64x64);  // above neighbour of (32,32)
  ctx = av2enc::sb128::partition_contexts(g, 32, 32, BlockSize::k128x128,
                                          grid2);
  CHECK(ctx.do_split == 27);
  CHECK(ctx.do_square_split == 3);

  // 128x64 uses the next Partition_Size_Adjust row (5), not 6.
  av2enc::sb128::MiSizeGrid empty(g);
  ctx = av2enc::sb128::partition_contexts(g, 0, 0, BlockSize::k128x64, empty);
  CHECK(ctx.do_split == 20);
  CHECK(ctx.rect_type == 32);

  // CDF rows: the [2] dimension is PlaneStart, not the SB flag.  SHARED_PART
  // uses plane 0.  Context bounds are the table shapes.
  CHECK(av2enc::sb128::do_split_cdf_row(0, 24) ==
        &av2enc::m0::kDoSplitCdf[0][24][0]);
  CHECK(av2enc::sb128::do_split_cdf_row(1, 63) ==
        &av2enc::m0::kDoSplitCdf[1][63][0]);
  CHECK(av2enc::sb128::do_split_cdf_row(0, 64) == nullptr);
  CHECK(av2enc::sb128::do_split_cdf_row(2, 0) == nullptr);
  CHECK(av2enc::sb128::do_square_split_cdf_row(0, 0) ==
        &av2enc::m0::kDoSquareSplitCdf[0][0][0]);
  CHECK(av2enc::sb128::do_square_split_cdf_row(0, 7) ==
        &av2enc::m0::kDoSquareSplitCdf[0][7][0]);
  CHECK(av2enc::sb128::do_square_split_cdf_row(0, 8) == nullptr);
  CHECK(av2enc::sb128::rect_type_cdf_row(0, 36) ==
        &av2enc::m0::kRectTypeCdf[0][36][0]);
  CHECK(av2enc::sb128::rect_type_cdf_row(0, 64) == nullptr);
}

void test_chroma_and_edges() {
  const Geometry g = make_geometry(640, 360);
  CHECK(av2enc::sb128::chroma_plane_residual_size(g, BlockSize::k128x128) ==
        BlockSize::k64x64);
  CHECK(av2enc::sb128::chroma_plane_residual_size(g, BlockSize::k128x64) ==
        BlockSize::k64x32);
  CHECK(av2enc::sb128::chroma_plane_residual_size(g, BlockSize::k64x128) ==
        BlockSize::k32x64);
  CHECK(av2enc::sb128::chroma_plane_residual_size(g, BlockSize::k64x64) ==
        BlockSize::k32x32);

  const BlockRect bottom = av2enc::sb128::block_rect(
      g, 80, 0, BlockSize::k64x64);  // y = 320, 40 visible rows
  CHECK(bottom.x == 0 && bottom.y == 320);
  CHECK(bottom.width == 64 && bottom.height == 64);
  CHECK(bottom.visible_width == 64);
  CHECK(bottom.visible_height == 40);

  const BlockRect right = av2enc::sb128::block_rect(
      g, 0, 144, BlockSize::k64x64);  // x = 576, 64 visible columns
  CHECK(right.x == 576 && right.visible_width == 64);

  // Chroma offset recursion at the 1080p boundary: the implied HORZ root sets
  // ChromaMiSize to its own 128x128, then 128x64 -> VERT -> 64x64 leaves.
  const Geometry g1080 = make_geometry(1920, 1080);
  const av2enc::sb128::ChromaBlock inherited{256, 0, BlockSize::k128x128};
  const auto root = av2enc::sb128::resolve_chroma(
      g1080, 256, 0, BlockSize::k128x128, kHorz, true, false, inherited);
  CHECK(root.block.size == BlockSize::k128x128);
  CHECK(!root.chroma_offset);
  const auto mid = av2enc::sb128::resolve_chroma(
      g1080, 256, 0, BlockSize::k128x64, kVert, true, root.chroma_offset,
      root.block);
  CHECK(mid.block.size == BlockSize::k128x64);
  const auto leaf = av2enc::sb128::resolve_chroma(
      g1080, 256, 0, BlockSize::k64x64, kNone, true, mid.chroma_offset,
      mid.block);
  CHECK(leaf.block.size == BlockSize::k64x64);
  CHECK(av2enc::sb128::chroma_plane_residual_size(g1080, leaf.block.size) ==
        BlockSize::k32x32);
}

std::string serialize_node(const av2enc::sb128::Node& node) {
  std::string out;
  out += std::to_string(node.mi_row) + "," + std::to_string(node.mi_col) + ",";
  out += std::to_string(static_cast<int>(node.size)) + "," +
         std::to_string(static_cast<int>(node.partition)) + "," +
         std::to_string(node.forced) + "," +
         std::to_string(node.has_chroma) + "," +
         std::to_string(node.chroma_offset) + "," +
         std::to_string(static_cast<int>(node.chroma.size)) + ";";
  for (const auto& child : node.children) out += serialize_node(child);
  return out;
}

std::string serialize_frame(const av2enc::sb128::FrameTree& frame) {
  std::string out;
  for (const auto& root : frame.superblocks) out += serialize_node(root);
  return out;
}

void test_reference_tree() {
  struct Case {
    int width;
    int height;
    int superblocks;
    int nodes;
    int leaves;
  };
  const Case cases[] = {
      {128, 128, 1, 5, 4},
      {256, 256, 4, 20, 16},
      {640, 360, 15, 75, 60},
      {1920, 1080, 135, 660, 510},
  };
  for (const Case& c : cases) {
    const Geometry geometry = make_geometry(c.width, c.height);
    av2enc::sb128::FrameTree frame;
    std::string error;
    CHECK(av2enc::sb128::build_reference_frame(geometry, frame, error));
    CHECK(frame.superblocks.size() ==
          static_cast<std::size_t>(c.superblocks));
    CHECK(frame.nodes == c.nodes);
    CHECK(frame.leaves == c.leaves);

    std::vector<av2enc::sb128::LeafBlock> leaves;
    for (const auto& root : frame.superblocks) {
      av2enc::sb128::collect_leaves(root, leaves);
    }
    CHECK(leaves.size() == static_cast<std::size_t>(c.leaves));
    for (const auto& leaf : leaves) {
      CHECK(leaf.size == BlockSize::k64x64);
      CHECK(leaf.has_chroma);
    }

    // Determinism: same geometry -> byte-identical tree serialization.
    av2enc::sb128::FrameTree again;
    CHECK(av2enc::sb128::build_reference_frame(geometry, again, error));
    CHECK(serialize_frame(frame) == serialize_frame(again));
  }

  // 1080p's last SB row is boundary-forced HORZ, not SPLIT.
  const Geometry g1080 = make_geometry(1920, 1080);
  av2enc::sb128::FrameTree frame;
  std::string error;
  CHECK(av2enc::sb128::build_reference_frame(g1080, frame, error));
  const auto& last = frame.superblocks.back();
  CHECK(last.mi_row == 256);
  CHECK(last.forced);
  CHECK(last.partition == kHorz);
  CHECK(last.children.size() == 1);
  CHECK(last.children[0].size == BlockSize::k128x64);
  CHECK(last.children[0].partition == kVert);
  CHECK(last.children[0].children.size() == 2);

  // 640x360's last SB row is a full root (half-block test passes) and uses
  // SPLIT; only the bottom 64x64 leaves are clipped by the picture.
  const Geometry g360 = make_geometry(640, 360);
  CHECK(av2enc::sb128::build_reference_frame(g360, frame, error));
  const auto& bottom = frame.superblocks.back();
  CHECK(bottom.mi_row == 64);
  CHECK(!bottom.forced);
  CHECK(bottom.partition == kSplit);
  CHECK(bottom.children.size() == 4);

  CHECK(av2enc::sb128::validate_frame(g360, frame, error));
}

void test_cross_check_partition_module() {
  // Every §9.2 block/partition row, not just the sizes the tree reaches.
  for (int value = 0; value < av2enc::sb128::kBlockSizeCount; ++value) {
    const auto size = static_cast<BlockSize>(value);
    const auto psize = static_cast<p::BlockSize>(value);
    CHECK(av2enc::sb128::mi_width_log2(size) == p::mi_width_log2(psize));
    CHECK(av2enc::sb128::mi_height_log2(size) == p::mi_height_log2(psize));
    CHECK(av2enc::sb128::block_width(size) == p::block_width(psize));
    CHECK(av2enc::sb128::block_height(size) == p::block_height(psize));
    CHECK(av2enc::sb128::partition_size_adjust(size) ==
          p::partition_size_adjust(psize));
    CHECK(av2enc::sb128::partition_size_adjust_rect_type(size) ==
          p::partition_size_adjust_rect_type(psize));
    const auto implied = av2enc::sb128::rect_type_implied_by_bsize(size);
    const auto pimplied = p::rect_type_implied_by_bsize(psize);
    CHECK(implied.implied == pimplied.implied);
    CHECK(static_cast<int>(implied.partition) ==
          static_cast<int>(pimplied.partition));
    for (const PartitionType part : {kNone, kSplit, kHorz, kVert}) {
      CHECK(static_cast<int>(av2enc::sb128::partition_subsize(part, size)) ==
            static_cast<int>(p::partition_subsize(
                static_cast<p::PartitionType>(part), psize)));
    }
  }

  const Geometry geometries[] = {
      make_geometry(128, 128),   make_geometry(256, 256),
      make_geometry(640, 360),   make_geometry(1920, 1080),
      make_geometry(200, 136),   make_geometry(136, 200),
  };
  const BlockSize sizes[] = {
      BlockSize::k128x128, BlockSize::k128x64, BlockSize::k64x128,
      BlockSize::k64x64,   BlockSize::k32x32,  BlockSize::k16x16,
      BlockSize::k8x8,     BlockSize::k4x8,
  };
  for (const Geometry& geometry : geometries) {
    const p::FrameGeometry pg = to_partition(geometry);
    CHECK(p::mi_cols(pg) == av2enc::sb128::mi_cols(geometry));
    CHECK(p::mi_rows(pg) == av2enc::sb128::mi_rows(geometry));
    CHECK(p::sb_cols(pg) == av2enc::sb128::sb_cols(geometry));
    CHECK(p::sb_rows(pg) == av2enc::sb128::sb_rows(geometry));

    for (int row = 0; row < av2enc::sb128::mi_rows(geometry); row += 3) {
      for (int col = 0; col < av2enc::sb128::mi_cols(geometry); col += 3) {
        for (const BlockSize size : sizes) {
          const auto psize = static_cast<p::BlockSize>(size);
          for (const PartitionType part : {kNone, kSplit, kHorz, kVert}) {
            const auto ppart = static_cast<p::PartitionType>(part);
            CHECK(static_cast<int>(av2enc::sb128::partition_subsize(part, size)) ==
                  static_cast<int>(p::partition_subsize(ppart, psize)));
            CHECK(av2enc::sb128::is_partition_allowed(geometry, row, col, size,
                                                      part, true, false) ==
                  p::is_partition_allowed(pg, row, col, psize, ppart, true,
                                          false));
          }
          const Boundary ours =
              av2enc::sb128::partition_implied(geometry, row, col, size);
          const p::PartitionBoundary theirs =
              p::partition_implied(pg, row, col, psize);
          CHECK(ours.has_rows == theirs.has_rows);
          CHECK(ours.has_cols == theirs.has_cols);
          CHECK(ours.implied == theirs.implied);
          CHECK(static_cast<int>(ours.implied_partition) ==
                static_cast<int>(theirs.implied_partition));

          const auto our_set = av2enc::sb128::init_allowed_partitions(
              geometry, row, col, size, true, false);
          const auto their_set =
              p::init_allowed_partitions(pg, row, col, psize, true, false);
          CHECK(our_set.count == their_set.count);
          for (int i = 0; i < 4; ++i) {
            CHECK(our_set.allowed[i] == their_set.allowed[i]);
          }
        }
      }
    }
  }
}

void test_cross_check_contexts() {
  const Geometry geometries[] = {
      make_geometry(128, 128), make_geometry(640, 360),
      make_geometry(1920, 1080),
  };
  const BlockSize sizes[] = {
      BlockSize::k128x128, BlockSize::k128x64, BlockSize::k64x128,
      BlockSize::k64x64,   BlockSize::k32x32,  BlockSize::k8x8,
  };
  for (const Geometry& geometry : geometries) {
    const p::FrameGeometry pg = to_partition(geometry);
    av2enc::sb128::MiSizeGrid ours(geometry);
    p::MiSizeGrid theirs(pg);
    for (int row = 0; row < av2enc::sb128::mi_rows(geometry); ++row) {
      for (int col = 0; col < av2enc::sb128::mi_cols(geometry); ++col) {
        const BlockSize size =
            ((row / 4 + col / 4) & 1) != 0 ? BlockSize::k64x64
                                           : BlockSize::k32x32;
        ours.set(row, col, size);
        theirs.set(row, col, static_cast<p::BlockSize>(size));
      }
    }
    for (int row = 0; row < av2enc::sb128::mi_rows(geometry); row += 5) {
      for (int col = 0; col < av2enc::sb128::mi_cols(geometry); col += 5) {
        for (const BlockSize size : sizes) {
          const Contexts a = av2enc::sb128::partition_contexts(
              geometry, row, col, size, ours);
          const p::PartitionContexts b = p::partition_contexts(
              pg, row, col, static_cast<p::BlockSize>(size), theirs);
          CHECK(a.do_split == b.do_split);
          CHECK(a.do_square_split == b.do_square_split);
          CHECK(a.rect_type == b.rect_type);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Header / stream tests
// ---------------------------------------------------------------------------

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  bool read_bit() {
    if (pos_ >= bytes_.size() * 8U) {
      ok_ = false;
      return false;
    }
    const std::uint8_t byte = bytes_[pos_ / 8U];
    const int shift = 7 - static_cast<int>(pos_ % 8U);
    ++pos_;
    return ((byte >> shift) & 1U) != 0U;
  }

  std::uint32_t read_bits(unsigned count) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < count; ++i) {
      value = (value << 1U) | (read_bit() ? 1U : 0U);
    }
    return value;
  }

  std::uint32_t uvlc() {
    unsigned leading_zeros = 0;
    while (!read_bit()) {
      ++leading_zeros;
      if (leading_zeros > 31) {
        ok_ = false;
        return 0;
      }
    }
    const std::uint32_t base = (1U << leading_zeros) - 1U;
    return base + (leading_zeros == 0 ? 0 : read_bits(leading_zeros));
  }

  bool ok() const { return ok_; }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t pos_ = 0;
  bool ok_ = true;
};

bool sequence_use_128(const av2enc::sb128::StreamConfig& config) {
  BitWriter writer;
  av2enc::sb128::write_sequence_header(writer, config);
  const std::vector<std::uint8_t> bytes = writer.take_bytes();
  BitReader reader(bytes);
  CHECK(reader.uvlc() == 0);    // seq_header_id
  CHECK(reader.read_bits(5) == 0);  // seq_profile_idc
  CHECK(reader.read_bit());     // single_picture_header_flag
  reader.read_bits(5);          // seq_level_idx
  CHECK(reader.uvlc() == 0);    // chroma_format_idc
  CHECK(reader.uvlc() == 1);    // bit_depth_idc
  const unsigned width_bits = reader.read_bits(4) + 1;
  const unsigned height_bits = reader.read_bits(4) + 1;
  reader.read_bits(width_bits);
  reader.read_bits(height_bits);
  CHECK(!reader.read_bit());  // seq_cropping_window_present_flag
  CHECK(!reader.read_bit());  // use_256x256_superblock
  const bool use_128 = reader.read_bit();
  CHECK(reader.ok());
  return use_128;
}

void test_headers() {
  av2enc::sb128::StreamConfig config;
  config.width = 128;
  config.height = 128;
  CHECK(sequence_use_128(config));
  config.use_128x128_superblock = false;
  CHECK(!sequence_use_128(config));
  config.use_128x128_superblock = true;

  struct Case {
    int width;
    int height;
    std::size_t bits;
  };
  // 128x128 with a 128 SB has one tile, so no tile increment bits: 22 bits.
  // The same frame with a 64 SB would have 2x2 SBs and 24 bits.
  const Case cases[] = {
      {128, 128, 22},
      {256, 256, 24},
      {640, 360, 24},
      {1920, 1080, 24},
  };
  for (const Case& c : cases) {
    av2enc::sb128::StreamConfig cfg;
    cfg.width = static_cast<std::uint16_t>(c.width);
    cfg.height = static_cast<std::uint16_t>(c.height);
    BitWriter writer;
    const std::size_t bits = av2enc::sb128::write_frame_header(writer, cfg);
    CHECK(bits == c.bits);
  }

  av2enc::sb128::StreamConfig cfg64;
  cfg64.width = 128;
  cfg64.height = 128;
  cfg64.use_128x128_superblock = false;
  BitWriter writer;
  CHECK(av2enc::sb128::write_frame_header(writer, cfg64) == 24);
}

std::vector<int> obu_types(const std::vector<std::uint8_t>& bytes) {
  std::vector<int> types;
  std::size_t pos = 0;
  while (pos < bytes.size()) {
    std::uint32_t size = 0;
    unsigned shift = 0;
    while (pos < bytes.size()) {
      const std::uint8_t byte = bytes[pos++];
      size |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
      if ((byte & 0x80U) == 0U) break;
      shift += 7;
    }
    if (pos >= bytes.size() || pos + size > bytes.size()) {
      types.push_back(-1);
      break;
    }
    const std::uint8_t header = bytes[pos];
    types.push_back(static_cast<int>((header >> 2U) & 0x1FU));
    pos += size;
  }
  return types;
}

void test_stream_builder() {
  struct Case {
    int width;
    int height;
    int leaves;
  };
  const Case cases[] = {
      {128, 128, 4},
      {256, 256, 16},
      {640, 360, 60},
      {1920, 1080, 510},
      // Partial-bottom variants: h = 26 (SPLIT) and h = 10 (implied HORZ).
      {128, 232, 8},
      {128, 104, 4},
      {128, 168, 6},
  };
  for (const Case& c : cases) {
    av2enc::sb128::StreamConfig config;
    config.width = static_cast<std::uint16_t>(c.width);
    config.height = static_cast<std::uint16_t>(c.height);
    av2enc::sb128::StreamResult first;
    std::string error;
    CHECK(av2enc::sb128::make_sb128_stream(config, first, error));
    CHECK(first.leaves == c.leaves);
    CHECK(!first.frame.empty());
    CHECK(!first.tile.empty());
    CHECK(first.symbols > 0);

    const std::vector<int> types = obu_types(first.frame);
    CHECK(types.size() == 3);
    CHECK(types.size() == 3 && types[0] == 2 && types[1] == 1 &&
          types[2] == 4);

    av2enc::sb128::StreamResult second;
    CHECK(av2enc::sb128::make_sb128_stream(config, second, error));
    CHECK(first.frame == second.frame);
    CHECK(first.tile == second.tile);
  }

  // Adaptive CDFs are also supported (the decoder updates in lockstep).
  av2enc::sb128::StreamConfig adaptive;
  adaptive.width = 128;
  adaptive.height = 128;
  adaptive.disable_cdf_update = false;
  av2enc::sb128::StreamResult a;
  std::string error;
  CHECK(av2enc::sb128::make_sb128_stream(adaptive, a, error));
  av2enc::sb128::StreamResult b;
  CHECK(av2enc::sb128::make_sb128_stream(adaptive, b, error));
  CHECK(a.frame == b.frame);
  adaptive.disable_cdf_update = true;
  av2enc::sb128::StreamResult frozen;
  CHECK(av2enc::sb128::make_sb128_stream(adaptive, frozen, error));
  CHECK(a.tile != frozen.tile);

  // Geometry support: right edge must be SB-aligned; the bottom remainder
  // must leave 64x64 leaves.
  struct Support {
    int width;
    int height;
    bool supported;
  };
  const Support support[] = {
      {128, 128, true},   {256, 256, true},   {640, 360, true},
      {1920, 1080, true}, {128, 232, true},   {128, 104, true},
      {128, 168, true},   {200, 128, false},  {128, 136, false},
      {128, 200, false},  {128, 96, false},   {128, 160, false},
  };
  for (const Support& s : support) {
    const Geometry geometry = make_geometry(s.width, s.height);
    std::string why;
    CHECK(av2enc::sb128::stream_geometry_supported(geometry, why) ==
          s.supported);
  }

  // Geometry accepted by the tree builder but rejected by the minimal leaf
  // writer must fail cleanly, not emit a desynchronised stream.
  av2enc::sb128::StreamConfig odd;
  odd.width = 200;
  odd.height = 128;
  av2enc::sb128::StreamResult bad;
  std::string error_message;
  CHECK(!av2enc::sb128::make_sb128_stream(odd, bad, error_message));
  CHECK(!error_message.empty());
}

void test_sb128_geometry_supported() {
  const Geometry g = make_geometry(1920, 1080);
  std::string error;
  CHECK(av2enc::sb128::stream_geometry_supported(g, error));
}

}  // namespace

int main() {
  test_block_tables();
  test_partition_subsize();
  test_geometry();
  test_boundary_implied();
  test_allowed_and_forced();
  test_contexts_and_cdf_rows();
  test_chroma_and_edges();
  test_reference_tree();
  test_cross_check_partition_module();
  test_cross_check_contexts();
  test_headers();
  test_stream_builder();
  test_sb128_geometry_supported();
  return test::finish("sb128_test");
}
