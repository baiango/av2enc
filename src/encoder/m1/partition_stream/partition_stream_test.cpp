// Tests for the M1 extended-partition adapter.
//
// Coverage:
//   * §9.3 CDF copies, including the two new rows and both CDF modes;
//   * allowed_mask (10-bit, writer family view, uint8 fit);
//   * §8.3.2 context derivation from the M1 writer's neighbour arrays,
//     including the r + h4/2 / c + w4/2 second probes;
//   * plan_partition_symbols() output: do_ext_partition,
//     do_uneven_4way_partition, and the L(1) uneven_4way_partition_type bit;
//   * preorder stream shape (64x64 and 128x128 superblocks, tool flags on/off);
//   * per-leaf has_chroma / ChromaBlock for HORZ_4A@32x32 (chroma-less first
//     children) and HORZ_3@8x32 (shared middle chroma);
//   * encode -> independent decoder round trips, byte-identical to
//     partition_ext::encode_tree_symbols(), in both CDF modes.
//
// Run: ./build_and_test.sh

#include "partition_stream.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "encoder/m0_cdf_tables.h"
#include "encoder/m1/m1_tables.h"
#include "encoder/partition_ext/test_util.h"

int g_failures = 0;

namespace ps = av2enc::m1::partition_stream;
namespace av2 = av2enc::av2;

namespace {

constexpr pe::PartitionType kNone = pe::PartitionType::kNone;
constexpr pe::PartitionType kHorz = pe::PartitionType::kHorz;
constexpr pe::PartitionType kVert = pe::PartitionType::kVert;
constexpr pe::PartitionType kHorz3 = pe::PartitionType::kHorz3;
constexpr pe::PartitionType kVert3 = pe::PartitionType::kVert3;
constexpr pe::PartitionType kHorz4A = pe::PartitionType::kHorz4A;
constexpr pe::PartitionType kHorz4B = pe::PartitionType::kHorz4B;
constexpr pe::PartitionType kVert4B = pe::PartitionType::kVert4B;
constexpr pe::PartitionType kSplit = pe::PartitionType::kSplit;


pe::DecisionOptions default_options() {
  pe::DecisionOptions options;
  options.min_block_size_px = 4;
  options.max_depth = 32;
  return options;
}

pe::FrameDecision decide(const p::FrameGeometry& geometry,
                         const pe::Tools& tools,
                         const std::function<pe::PartitionType(
                             const pe::NodeInfo&)>& chooser) {
  return pe::decide_frame(geometry, choose_partition(chooser), tools,
                          default_options());
}

void test_cdfs() {
  ps::TilePartitionCdfs cdfs;
  ps::init_tile_partition_cdfs(cdfs);

  // Base rows are byte-identical to the shipping M0/M1 tables.
  CHECK(std::memcmp(cdfs.ext.do_split, av2enc::m0::kDoSplitCdf,
                    sizeof(cdfs.ext.do_split)) == 0);
  CHECK(std::memcmp(cdfs.ext.do_square_split,
                    av2enc::m0::kDoSquareSplitCdf,
                    sizeof(cdfs.ext.do_square_split)) == 0);
  CHECK(std::memcmp(cdfs.ext.rect_type, av2enc::m0::kRectTypeCdf,
                    sizeof(cdfs.ext.rect_type)) == 0);
  // The two rows the M1 writer is missing come from the §9.3 tables.
  CHECK(std::memcmp(cdfs.ext.do_ext_partition, pe::kDoExtPartitionCdf,
                    sizeof(cdfs.ext.do_ext_partition)) == 0);
  CHECK(std::memcmp(cdfs.ext.do_uneven_4way_partition,
                    pe::kDoUneven4wayPartitionCdf,
                    sizeof(cdfs.ext.do_uneven_4way_partition)) == 0);
  CHECK_EQ(cdfs.ext.do_ext_partition[0][0][0], 16384);
  CHECK_EQ(cdfs.ext.do_uneven_4way_partition[0][0][0], 16384);

  // The adapter adds no storage: it is exactly the five partition arrays.
  static_assert(sizeof(ps::TilePartitionCdfs) == sizeof(pe::PartitionCdfs));
  static_assert(offsetof(ps::TilePartitionCdfs, ext) == 0);

  std::string error;
  CHECK(ps::cdf_row(cdfs, pe::SymbolKind::kDoExtPartition, 63, error) ==
        &cdfs.ext.do_ext_partition[0][63][0]);
  CHECK(ps::cdf_row(cdfs, pe::SymbolKind::kDoUneven4wayPartition, 17, error) ==
        &cdfs.ext.do_uneven_4way_partition[0][17][0]);
  CHECK(ps::cdf_row(cdfs, pe::SymbolKind::kDoExtPartition, 64, error) ==
        nullptr);
  CHECK(ps::cdf_row(cdfs, pe::SymbolKind::kUneven4wayType, 0, error) ==
        nullptr);
}

// The adapter derives §8.3.2 contexts with partition_ext's tables; the M1
// writer's code_do_split/code_rect_type use m1_tables.h.  Pin them together so
// a table regeneration cannot silently desynchronise the CDF contexts.
void test_table_parity() {
  for (int i = 0; i < p::kBlockSizeCount; ++i) {
    const auto size = static_cast<p::BlockSize>(i);
    CHECK_EQ(p::mi_width_log2(size), av2enc::m1::kMiWidthLog2[i]);
    CHECK_EQ(p::mi_height_log2(size), av2enc::m1::kMiHeightLog2[i]);
    CHECK_EQ(pe::size_adjust(size), av2enc::m1::kPartitionSizeAdjust[i]);
    CHECK_EQ(pe::size_adjust_rect_type(size),
             av2enc::m1::kPartitionSizeAdjustRect[i]);
  }
}

void test_allowed_mask() {
  const p::FrameGeometry g = geometry_420(64, 64, 64);
  pe::Tools off;
  pe::Tools ext;
  ext.enable_ext_partitions = true;
  pe::Tools both;
  both.enable_ext_partitions = true;
  both.enable_uneven_4way_partitions = true;

  const pe::PartitionSet s_off =
      pe::init_allowed_partitions(g, 0, 0, p::BlockSize::k64x64, true, false,
                                  off);
  const std::uint16_t m_off = ps::to_allowed_mask(s_off);
  CHECK_EQ(m_off, (1u << 0) | (1u << 1) | (1u << 2));
  CHECK_EQ(ps::mask_count(m_off), s_off.count);
  CHECK(ps::mask_allows(m_off, kNone));
  CHECK(ps::mask_allows(m_off, kHorz));
  CHECK(ps::mask_allows(m_off, kVert));
  CHECK(!ps::mask_allows(m_off, kHorz3));
  CHECK_EQ(ps::to_writer_mask8(m_off), 0x0Du);  // NONE | HORZ | VERT
  CHECK(ps::mask_fits_writer_u8(m_off));

  const pe::PartitionSet s_ext =
      pe::init_allowed_partitions(g, 0, 0, p::BlockSize::k64x64, true, false,
                                  ext);
  const std::uint16_t m_ext = ps::to_allowed_mask(s_ext);
  CHECK_EQ(ps::mask_count(m_ext), s_ext.count);
  CHECK(ps::mask_allows(m_ext, kHorz3));
  CHECK(ps::mask_allows(m_ext, kVert3));
  CHECK(!ps::mask_allows(m_ext, kHorz4A));
  CHECK_EQ(ps::to_writer_mask8(m_ext), 0x0Du);
  CHECK(ps::mask_fits_writer_u8(m_ext));

  const pe::PartitionSet s_both =
      pe::init_allowed_partitions(g, 0, 0, p::BlockSize::k64x64, true, false,
                                  both);
  const std::uint16_t m_both = ps::to_allowed_mask(s_both);
  CHECK_EQ(ps::mask_count(m_both), s_both.count);
  CHECK(ps::mask_allows(m_both, kVert4B));  // bit 8: does not fit uint8
  CHECK(!ps::mask_fits_writer_u8(m_both));

  // 128x128 root: SPLIT (bit 9) is allowed but no extended row is.
  const p::FrameGeometry big = geometry_420(128, 128, 128);
  const pe::PartitionSet s_big =
      pe::init_allowed_partitions(big, 0, 0, p::BlockSize::k128x128, true,
                                  false, both);
  const std::uint16_t m_big = ps::to_allowed_mask(s_big);
  CHECK(ps::mask_allows(m_big, kSplit));
  CHECK(!ps::mask_allows(m_big, kHorz3));
  CHECK(!ps::mask_fits_writer_u8(m_big));
  CHECK_EQ(ps::to_writer_mask8(m_big), 0x0Fu);

  // Unknown/high bits never produce a partition.
  CHECK(!ps::mask_allows(static_cast<std::uint16_t>(1u << 10), kNone));
  CHECK_EQ(ps::mask_count(static_cast<std::uint16_t>(0xFC00u)), 0);
}

// Compare the adapter's neighbour-based contexts with partition_ext's grid
// derivation over a whole decided frame.
void check_contexts_frame(const p::FrameGeometry& geometry,
                          const pe::Tools& tools,
                          const pe::CostFunction& costs) {
  const pe::FrameDecision frame =
      pe::decide_frame(geometry, costs, tools, default_options());
  CHECK(frame.status == pe::DecisionStatus::kOk);
  if (frame.status != pe::DecisionStatus::kOk) {
    std::cerr << "frame decision: " << frame.error << '\n';
    return;
  }
  p::MiSizeGrid grid(geometry);
  int mismatches = 0;
  std::function<void(const pe::PartitionNode&)> walk =
      [&](const pe::PartitionNode& node) {
        const ps::NeighbourContext neighbours = ps::neighbours_from_grid(
            geometry, node.mi_row, node.mi_col, node.size, grid);
        const ps::NodeContexts got = ps::contexts_from_neighbours(
            node.size, node.partition, neighbours);
        const pe::SymbolContexts want =
            pe::node_contexts(geometry, node, grid);
        if (got.do_split != want.do_split ||
            got.do_square_split != want.do_square_split ||
            got.rect_type != want.rect_type || got.do_ext != want.do_ext) {
          ++mismatches;
          std::cerr << "context mismatch at " << p::block_size_name(node.size)
                    << '@' << node.mi_row << ',' << node.mi_col << " part "
                    << pe::partition_name(node.partition) << ": got "
                    << got.do_split << '/' << got.do_square_split << '/'
                    << got.rect_type << '/' << got.do_ext << " want "
                    << want.do_split << '/' << want.do_square_split << '/'
                    << want.rect_type << '/' << want.do_ext << '\n';
        }
        if (node.partition == kNone) {
          grid.fill_block(node.mi_row, node.mi_col, node.size);
          return;
        }
        for (const pe::PartitionNode& child : node.children) walk(child);
      };
  for (const pe::PartitionNode& sb : frame.superblocks) walk(sb);
  CHECK_EQ(mismatches, 0);
}

void test_contexts() {
  pe::Tools off;
  pe::Tools ext;
  ext.enable_ext_partitions = true;
  pe::Tools both;
  both.enable_ext_partitions = true;
  both.enable_uneven_4way_partitions = true;

  const auto chooser = [](const pe::NodeInfo& info) {
    if (info.size == p::BlockSize::k64x64) return kHorz3;
    if (info.size == p::BlockSize::k32x32) return kVert3;
    if (info.size == p::BlockSize::k16x16) return kHorz4A;
    if (info.size == p::BlockSize::k8x8) return kVert4B;
    return kNone;
  };

  check_contexts_frame(geometry_420(64, 64, 64), off,
                       choose_partition(chooser));
  check_contexts_frame(geometry_420(64, 64, 64), both,
                       choose_partition(chooser));
  check_contexts_frame(geometry_420(640, 360, 64), both,
                       choose_partition(chooser));
  check_contexts_frame(geometry_420(192, 128, 128), both,
                       choose_partition(chooser));

  // Direct second-probe check: a synthetic grid where the start probe and the
  // midpoint probe disagree.  This is the bug the README warns about
  // ("must be called with r + h4/2 / c + w4/2").
  const p::FrameGeometry g = geometry_420(128, 128, 64);
  p::MiSizeGrid grid(g);
  grid.fill_block(8, 0, p::BlockSize::k32x16);   // left of the node at row 8
  grid.fill_block(12, 0, p::BlockSize::k64x32);  // left of the node at row 16
  grid.fill_block(4, 8, p::BlockSize::k16x16);   // above the node at col 8
  grid.fill_block(4, 16, p::BlockSize::k32x32);  // above the node at col 16

  const p::BlockSize size = p::BlockSize::k64x64;
  const ps::NeighbourContext nc =
      ps::neighbours_from_grid(g, 8, 8, size, grid);
  CHECK_EQ(nc.left_log2, 2);       // 32x16 -> Mi_Height_Log2 2
  CHECK_EQ(nc.left_mid_log2, 3);   // 64x32 at row 16
  CHECK_EQ(nc.above_log2, 2);      // 16x16 -> Mi_Width_Log2 2
  CHECK_EQ(nc.above_mid_log2, 3);  // 32x32 at col 16

  for (bool horizontal : {true, false}) {
    CHECK_EQ(ps::ext_partition_context(size, horizontal, nc),
             pe::ext_partition_context(g, 8, 8, size, horizontal, grid));
  }
  // The writer's left_mi/above_mi arrays hold one Mi_*_Log2 per row/column.
  std::vector<int> left(static_cast<std::size_t>(p::mi_rows(g)) + 16, 6);
  std::vector<int> above(static_cast<std::size_t>(p::mi_cols(g)) + 16, 6);
  for (int row = 0; row < p::mi_rows(g); ++row) {
    const p::BlockSize bs = grid.at(row, 7);
    left[static_cast<std::size_t>(row)] =
        p::is_valid_block_size(bs) ? p::mi_height_log2(bs) : 6;
  }
  for (int col = 0; col < p::mi_cols(g); ++col) {
    const p::BlockSize bs = grid.at(7, col);
    above[static_cast<std::size_t>(col)] =
        p::is_valid_block_size(bs) ? p::mi_width_log2(bs) : 6;
  }
  const ps::NeighbourContext from_arrays =
      ps::neighbours_from_arrays(left, above, 8, 8, size, true, true);
  CHECK_EQ(from_arrays.left_log2, nc.left_log2);
  CHECK_EQ(from_arrays.left_mid_log2, nc.left_mid_log2);
  CHECK_EQ(from_arrays.above_log2, nc.above_log2);
  CHECK_EQ(from_arrays.above_mid_log2, nc.above_mid_log2);

  const ps::NodeContexts got =
      ps::contexts_from_neighbours(size, kHorz3, nc);
  CHECK_EQ(got.do_ext, ps::ext_partition_context(size, true, nc));
  const pe::SymbolContexts want = pe::node_contexts(
      g, [&] {
        pe::PartitionNode node;
        node.mi_row = 8;
        node.mi_col = 8;
        node.size = size;
        node.partition = kHorz3;
        return node;
      }(),
      grid);
  CHECK_EQ(got.do_split, want.do_split);
  CHECK_EQ(got.rect_type, want.rect_type);
  CHECK_EQ(got.do_ext, want.do_ext);
}

ps::M1Node make_m1_node(const p::FrameGeometry& geometry,
                        const pe::Tools& tools, int mi_row, int mi_col,
                        p::BlockSize size, pe::PartitionType partition,
                        bool has_chroma = true, bool chroma_offset = false) {
  ps::M1Node node;
  node.mi_row = static_cast<std::uint16_t>(mi_row);
  node.mi_col = static_cast<std::uint16_t>(mi_col);
  node.bsize = static_cast<std::uint8_t>(size);
  node.partition = static_cast<std::uint8_t>(partition);
  node.has_chroma = has_chroma;
  node.chroma_offset = chroma_offset;
  const pe::PartitionSet allowed = pe::init_allowed_partitions(
      geometry, mi_row, mi_col, size, has_chroma, chroma_offset, tools);
  node.allowed_mask = ps::to_allowed_mask(allowed);
  const pe::Boundary boundary =
      pe::implied_partition(geometry, mi_row, mi_col, size);
  node.flags = 0;
  if ((boundary.implied &&
       allowed.is_allowed(boundary.implied_partition)) ||
      allowed.count == 1) {
    node.flags |= av2enc::m1::syntax::kFlagForced;
  }
  return node;
}

void expect_kinds(const ps::NodePlan& plan,
                  std::initializer_list<pe::SymbolKind> kinds) {
  CHECK(plan.ok);
  if (!plan.ok) {
    std::cerr << "plan error: " << plan.error << '\n';
    return;
  }
  CHECK_EQ(plan.symbols.symbols.size(), kinds.size());
  if (plan.symbols.symbols.size() != kinds.size()) return;
  std::size_t i = 0;
  for (pe::SymbolKind kind : kinds) {
    CHECK(plan.symbols.symbols[i].kind == kind);
    ++i;
  }
}

void test_plan_symbols() {
  const p::FrameGeometry g = geometry_420(64, 64, 64);
  pe::Tools off;
  pe::Tools ext;
  ext.enable_ext_partitions = true;
  pe::Tools both;
  both.enable_ext_partitions = true;
  both.enable_uneven_4way_partitions = true;

  // Root contexts on an empty grid: adjust * 4 for both CDFs.
  const p::MiSizeGrid empty(g);
  const ps::NeighbourContext root_nc =
      ps::neighbours_from_grid(g, 0, 0, p::BlockSize::k64x64, empty);
  CHECK_EQ(root_nc.left_log2, 6);
  CHECK_EQ(root_nc.above_log2, 6);
  const ps::NodeContexts root_ctx = ps::contexts_from_neighbours(
      p::BlockSize::k64x64, kHorz3, root_nc);
  CHECK_EQ(root_ctx.do_split, 12);     // Partition_Size_Adjust[64x64] = 3
  CHECK_EQ(root_ctx.rect_type, 24);    // Size_Adjust_Rect_Type[64x64] = 6
  CHECK_EQ(root_ctx.do_ext, 12);

  // HORZ_3 at 64x64, both tools on: do_split=1, rect_type=0, do_ext=1,
  // do_uneven=0 (3-way only).
  {
    const ps::M1Node node = make_m1_node(g, both, 0, 0,
                                         p::BlockSize::k64x64, kHorz3);
    const ps::NodePlan plan = ps::plan_node(g, both, node, root_nc);
    expect_kinds(plan,
                 {pe::SymbolKind::kDoSplit, pe::SymbolKind::kRectType,
                  pe::SymbolKind::kDoExtPartition,
                  pe::SymbolKind::kDoUneven4wayPartition});
    CHECK_EQ(plan.symbols.symbols[0].value, 1);
    CHECK_EQ(plan.symbols.symbols[1].value, 0);  // HORZ
    CHECK_EQ(plan.symbols.symbols[2].value, 1);
    CHECK_EQ(plan.symbols.symbols[3].value, 0);
    CHECK_EQ(plan.symbols.symbols[1].context, 24);
    CHECK_EQ(plan.symbols.symbols[2].context, 12);
    CHECK_EQ(plan.symbols.symbols[3].context, 12);
  }

  // VERT_3 at 64x64 with ext only: do_ext and do_uneven are both implied.
  {
    const ps::M1Node node = make_m1_node(g, ext, 0, 0,
                                         p::BlockSize::k64x64, kVert3);
    const ps::NodePlan plan = ps::plan_node(g, ext, node, root_nc);
    expect_kinds(plan, {pe::SymbolKind::kDoSplit,
                        pe::SymbolKind::kRectType,
                        pe::SymbolKind::kDoExtPartition});
    CHECK_EQ(plan.symbols.symbols[1].value, 1);  // VERT
    CHECK_EQ(plan.symbols.symbols[2].value, 1);
  }

  // HORZ_4A at 32x32: the uneven type is an L(1) literal with value 0.
  {
    const p::MiSizeGrid grid(g);
    const ps::M1Node node = make_m1_node(g, both, 0, 0,
                                         p::BlockSize::k32x32, kHorz4A);
    const ps::NodePlan plan = ps::plan_node(g, both, node, grid);
    expect_kinds(plan,
                 {pe::SymbolKind::kDoSplit, pe::SymbolKind::kRectType,
                  pe::SymbolKind::kDoExtPartition,
                  pe::SymbolKind::kDoUneven4wayPartition,
                  pe::SymbolKind::kUneven4wayType});
    const pe::PartitionSymbol& literal = plan.symbols.symbols.back();
    CHECK(literal.literal);
    CHECK_EQ(literal.value, 0);
    CHECK_EQ(literal.context, 0);
  }

  // VERT_4B at 32x32 is the same sequence with literal value 1.
  {
    const ps::M1Node node = make_m1_node(g, both, 0, 0,
                                         p::BlockSize::k32x32, kVert4B);
    const ps::NodePlan plan = ps::plan_node(g, both, node, empty);
    expect_kinds(plan,
                 {pe::SymbolKind::kDoSplit, pe::SymbolKind::kRectType,
                  pe::SymbolKind::kDoExtPartition,
                  pe::SymbolKind::kDoUneven4wayPartition,
                  pe::SymbolKind::kUneven4wayType});
    CHECK_EQ(plan.symbols.symbols[1].value, 1);  // VERT
    CHECK_EQ(plan.symbols.symbols.back().value, 1);
  }

  // Tools off: only the rect family exists.
  {
    const ps::M1Node node =
        make_m1_node(g, off, 0, 0, p::BlockSize::k64x64, kHorz);
    const ps::NodePlan plan = ps::plan_node(g, off, node, root_nc);
    expect_kinds(plan, {pe::SymbolKind::kDoSplit,
                        pe::SymbolKind::kRectType});
    CHECK_EQ(plan.symbols.symbols[1].value, 0);
  }

  // 128x128 root SPLIT codes do_split then do_square_split.
  {
    const p::FrameGeometry big = geometry_420(128, 128, 128);
    const p::MiSizeGrid grid(big);
    const ps::M1Node node = make_m1_node(big, both, 0, 0,
                                         p::BlockSize::k128x128, kSplit);
    const ps::NodePlan plan = ps::plan_node(big, both, node, grid);
    expect_kinds(plan, {pe::SymbolKind::kDoSplit,
                        pe::SymbolKind::kDoSquareSplit});
    CHECK_EQ(plan.symbols.symbols[1].value, 1);
  }

  // A stale mask is rejected instead of silently emitting the wrong syntax.
  {
    ps::M1Node node =
        make_m1_node(g, both, 0, 0, p::BlockSize::k64x64, kHorz3);
    node.allowed_mask =
        static_cast<std::uint16_t>(node.allowed_mask & ~(1u << 3));
    const ps::NodePlan plan = ps::plan_node(g, both, node, root_nc);
    CHECK(!plan.ok);
  }

  // Boundary-implied nodes are forced and code nothing.
  {
    const p::FrameGeometry edge = geometry_420(64, 64, 64);
    const ps::M1Node node = make_m1_node(edge, both, 0, 0,
                                         p::BlockSize::k4x32, kNone);
    const ps::NodePlan plan = ps::plan_node(edge, both, node, empty);
    CHECK(plan.ok);
    CHECK(plan.forced);
    CHECK(plan.symbols.symbols.empty());
  }
}

void check_stream_against_tree(const ps::FrameStream& stream,
                               const pe::FrameDecision& frame) {
  int want_nodes = 0;
  int want_leaves = 0;
  for (const pe::PartitionNode& sb : frame.superblocks) {
    want_nodes += pe::count_nodes(sb);
    want_leaves += pe::count_leaves(sb);
  }
  CHECK_EQ(stream.node_count, want_nodes);
  CHECK_EQ(stream.leaf_count, want_leaves);

  std::vector<pe::LeafBlock> want;
  for (const pe::PartitionNode& sb : frame.superblocks) {
    pe::collect_leaves(sb, want);
  }
  CHECK_EQ(stream.leaves.size(), want.size());
  const std::size_t count = std::min(stream.leaves.size(), want.size());
  for (std::size_t i = 0; i < count; ++i) {
    const ps::LeafChroma& got = stream.leaves[i];
    const pe::LeafBlock& expected = want[i];
    CHECK_EQ(got.has_chroma, expected.has_chroma);
    CHECK_EQ(got.chroma_offset, expected.chroma_offset);
    CHECK(got.chroma == expected.chroma);
    CHECK(stream.nodes[got.node_index].bsize ==
          static_cast<std::uint8_t>(expected.size));
    CHECK_EQ(stream.nodes[got.node_index].mi_row, expected.mi_row);
    CHECK_EQ(stream.nodes[got.node_index].mi_col, expected.mi_col);
  }

  // Validate the preorder forest: every root's subtree covers exactly the
  // following entries, and there is one root per superblock.
  std::size_t cursor = 0;
  int roots = 0;
  while (cursor < stream.nodes.size()) {
    CHECK(cursor + 1u + stream.nodes[cursor].subtree <= stream.nodes.size());
    cursor = cursor + 1u + stream.nodes[cursor].subtree;
    ++roots;
  }
  CHECK_EQ(cursor, stream.nodes.size());
  CHECK_EQ(roots, static_cast<int>(frame.superblocks.size()));
}

void test_stream_shape() {
  const p::FrameGeometry g64 = geometry_420(64, 64, 64);
  pe::Tools both;
  both.enable_ext_partitions = true;
  both.enable_uneven_4way_partitions = true;

  // 64x64 superblock: HORZ_3 root, 5 nodes / 4 leaves.
  {
    const pe::FrameDecision frame = decide(
        g64, both, [](const pe::NodeInfo& info) {
          return info.depth == 0 ? kHorz3 : kNone;
        });
    CHECK(frame.status == pe::DecisionStatus::kOk);
    ps::FrameStream stream;
    std::string error;
    CHECK(ps::build_frame_stream(g64, both, frame, stream, error));
    if (!error.empty()) std::cerr << error << '\n';
    CHECK_EQ(stream.nodes.size(), 5);
    CHECK_EQ(stream.leaves.size(), 4);
    const ps::M1Node& root = stream.nodes[0];
    CHECK_EQ(root.subtree, 4);
    CHECK_EQ(root.partition, static_cast<std::uint8_t>(kHorz3));
    CHECK((root.flags & av2enc::m1::syntax::kFlagSbRoot) != 0);
    CHECK((root.flags & av2enc::m1::syntax::kFlagForced) == 0);
    CHECK(ps::mask_allows(root.allowed_mask, kHorz3));
    CHECK(ps::mask_allows(root.allowed_mask, kVert4B));
    CHECK(!ps::mask_fits_writer_u8(root.allowed_mask));
    CHECK_EQ(ps::to_writer_mask8(root.allowed_mask), 0x0Du);
    // The leaf chroma blocks are the spec's: 64x16, 32x32, 32x32, 64x16.
    CHECK(stream.nodes[stream.leaves[0].node_index].bsize ==
          static_cast<std::uint8_t>(p::BlockSize::k64x16));
    CHECK(stream.nodes[stream.leaves[1].node_index].bsize ==
          static_cast<std::uint8_t>(p::BlockSize::k32x32));
    CHECK(stream.nodes[stream.leaves[3].node_index].bsize ==
          static_cast<std::uint8_t>(p::BlockSize::k64x16));
    check_stream_against_tree(stream, frame);
  }

  // 128x128 superblock: SPLIT root (bit 9), HORZ_3 only below it.
  {
    const p::FrameGeometry g = geometry_420(128, 128, 128);
    const pe::FrameDecision frame = decide(
        g, both, [](const pe::NodeInfo& info) {
          if (info.depth == 0) return kSplit;
          if (info.size == p::BlockSize::k64x64 && info.mi_row == 0 &&
              info.mi_col == 0) {
            return kHorz3;
          }
          return kNone;
        });
    CHECK(frame.status == pe::DecisionStatus::kOk);
    ps::FrameStream stream;
    std::string error;
    CHECK(ps::build_frame_stream(g, both, frame, stream, error));
    CHECK_EQ(frame.superblocks.size(), 1u);
    const ps::M1Node& root = stream.nodes[0];
    CHECK_EQ(root.partition, static_cast<std::uint8_t>(kSplit));
    CHECK(ps::mask_allows(root.allowed_mask, kSplit));
    CHECK(!ps::mask_allows(root.allowed_mask, kHorz3));
    CHECK((root.flags & av2enc::m1::syntax::kFlagSbRoot) != 0);
    CHECK_EQ(stream.nodes.size(),
             static_cast<std::size_t>(pe::count_nodes(frame.superblocks[0])));
    check_stream_against_tree(stream, frame);
  }

  // Multiple 64x64 superblocks (two columns) and an edge superblock.
  {
    const p::FrameGeometry g = geometry_420(96, 64, 64);
    const pe::FrameDecision frame = decide(
        g, both, [](const pe::NodeInfo& info) {
          if (info.size == p::BlockSize::k64x64 && info.mi_col == 0) {
            return kVert3;
          }
          return kNone;
        });
    CHECK(frame.status == pe::DecisionStatus::kOk);
    ps::FrameStream stream;
    std::string error;
    CHECK(ps::build_frame_stream(g, both, frame, stream, error));
    CHECK_EQ(frame.superblocks.size(), 2u);
    int roots = 0;
    for (const ps::M1Node& node : stream.nodes) {
      if ((node.flags & av2enc::m1::syntax::kFlagSbRoot) != 0) ++roots;
    }
    CHECK_EQ(roots, 2);
    CHECK_EQ(stream.nodes[0].mi_row, 0);
    CHECK_EQ(stream.nodes[0].mi_col, 0);
    check_stream_against_tree(stream, frame);
  }
}

void test_chroma_carry() {
  pe::Tools both;
  both.enable_ext_partitions = true;
  both.enable_uneven_4way_partitions = true;

  // HORZ_4A@32x32: first three children code no chroma; the last 32x4 child
  // owns the whole 32x32 chroma block (16x16 at 4:2:0).
  {
    const p::FrameGeometry g = geometry_420(64, 64, 64);
    const pe::FrameDecision frame = decide(
        g, both, [](const pe::NodeInfo& info) {
          if (info.depth == 0) return kHorz;
          if (info.size == p::BlockSize::k64x32 && info.mi_col == 0) {
            return kVert;
          }
          if (info.size == p::BlockSize::k32x32 && info.mi_row == 0 &&
              info.mi_col == 0) {
            return kHorz4A;
          }
          return kNone;
        });
    CHECK(frame.status == pe::DecisionStatus::kOk);
    ps::FrameStream stream;
    std::string error;
    CHECK(ps::build_frame_stream(g, both, frame, stream, error));
    check_stream_against_tree(stream, frame);

    // Find the four children of the HORZ_4A node (preorder: node index 3).
    std::size_t horz4a = stream.nodes.size();
    for (std::size_t i = 0; i < stream.nodes.size(); ++i) {
      if (stream.nodes[i].partition == static_cast<std::uint8_t>(kHorz4A)) {
        horz4a = i;
        break;
      }
    }
    CHECK(horz4a + 4 < stream.nodes.size());
    if (horz4a + 4 < stream.nodes.size()) {
      const bool expected_chroma[4] = {false, false, false, true};
      for (int i = 0; i < 4; ++i) {
        const ps::M1Node& child = stream.nodes[horz4a + 1u + static_cast<std::size_t>(i)];
        CHECK_EQ(child.has_chroma, expected_chroma[i]);
        CHECK_EQ(child.chroma_offset, 1);
        CHECK(child.chroma.size == p::BlockSize::k32x32);
        CHECK_EQ(child.chroma.mi_row, 0);
        CHECK_EQ(child.chroma.mi_col, 0);
      }
      CHECK(stream.nodes[horz4a + 4].bsize ==
            static_cast<std::uint8_t>(p::BlockSize::k32x4));
      const ps::ChromaPlaneBlock chroma =
          ps::chroma_plane_block(g, stream.nodes[horz4a + 4].chroma);
      CHECK_EQ(chroma.mi_row, 0);
      CHECK_EQ(chroma.mi_col, 0);
      CHECK(chroma.residual_size == p::BlockSize::k16x16);
    }
  }

  // HORZ_3@8x32: the middle pair (4x16 + 4x16) shares one 8x16 chroma block,
  // owned by the mid-right child.  The top and bottom 8x8 leaves keep their
  // own chroma.
  {
    const p::FrameGeometry g = geometry_420(64, 64, 64);
    const pe::FrameDecision frame = decide(
        g, both, [](const pe::NodeInfo& info) {
          if (info.depth == 0) return kVert;                        // 32x64
          if (info.size == p::BlockSize::k32x64 && info.mi_col == 0)
            return kVert;                                           // 16x64
          if (info.size == p::BlockSize::k16x64 && info.mi_row == 0)
            return kHorz;                                           // 16x32
          if (info.size == p::BlockSize::k16x32 && info.mi_col == 0)
            return kVert;                                           // 8x32
          if (info.size == p::BlockSize::k8x32 && info.mi_row == 0 &&
              info.mi_col == 0) {
            return kHorz3;
          }
          return kNone;
        });
    CHECK(frame.status == pe::DecisionStatus::kOk);
    ps::FrameStream stream;
    std::string error;
    CHECK(ps::build_frame_stream(g, both, frame, stream, error));
    if (!error.empty()) std::cerr << error << '\n';
    check_stream_against_tree(stream, frame);

    std::size_t horz3 = stream.nodes.size();
    for (std::size_t i = 0; i < stream.nodes.size(); ++i) {
      if (stream.nodes[i].partition == static_cast<std::uint8_t>(kHorz3)) {
        horz3 = i;
        break;
      }
    }
    CHECK(horz3 + 4 < stream.nodes.size());
    if (horz3 + 4 < stream.nodes.size()) {
      const bool expected_chroma[4] = {true, false, true, true};
      // These are the blocks each leaf actually codes: the top and bottom
      // 8x8 leaves overwrite the inherited ChromaMi* (has_chroma && !offset),
      // while the mid-right 4x16 leaf keeps the shared 8x16 block.
      const p::BlockSize expected_size[4] = {
          p::BlockSize::k8x8, p::BlockSize::k8x16, p::BlockSize::k8x16,
          p::BlockSize::k8x8};
      const int expected_row[4] = {0, 2, 2, 6};
      for (int i = 0; i < 4; ++i) {
        const ps::M1Node& child = stream.nodes[horz3 + 1u + static_cast<std::size_t>(i)];
        CHECK_EQ(child.has_chroma, expected_chroma[i]);
        CHECK(child.chroma.size == expected_size[i]);
        CHECK_EQ(child.chroma.mi_row, expected_row[i]);
        CHECK_EQ(child.chroma.mi_col, 0);
      }
      CHECK_EQ(stream.nodes[horz3 + 2].chroma_offset, 1);
      CHECK_EQ(stream.nodes[horz3 + 3].chroma_offset, 1);
      CHECK_EQ(stream.nodes[horz3 + 4].chroma_offset, 0);

      // Chroma-plane coordinates: the shared middle 8x16 block starts at luma
      // mi row 2, i.e. chroma mi row 1, and its residual is 4x8.
      const ps::ChromaPlaneBlock mid =
          ps::chroma_plane_block(g, stream.nodes[horz3 + 3].chroma);
      CHECK_EQ(mid.mi_row, 1);
      CHECK_EQ(mid.mi_col, 0);
      CHECK(mid.residual_size == p::BlockSize::k4x8);
      // The top 8x8 leaf codes its own 8x8 chroma (4x4 residual).
      const ps::ChromaPlaneBlock top =
          ps::chroma_plane_block(g, stream.nodes[horz3 + 1].chroma);
      CHECK_EQ(top.mi_row, 0);
      CHECK_EQ(top.mi_col, 0);
      CHECK(top.residual_size == p::BlockSize::k4x4);
    }
  }

  // VERT_3@32x8: the mirror case for the shared 16x8 middle chroma block.
  {
    const p::FrameGeometry g = geometry_420(64, 64, 64);
    const pe::FrameDecision frame = decide(
        g, both, [](const pe::NodeInfo& info) {
          if (info.depth == 0) return kHorz;                         // 64x32
          if (info.size == p::BlockSize::k64x32 && info.mi_row == 0)
            return kHorz;                                            // 64x16
          if (info.size == p::BlockSize::k64x16 && info.mi_row == 0)
            return kVert;                                            // 32x16
          if (info.size == p::BlockSize::k32x16 && info.mi_col == 0)
            return kHorz;                                            // 32x8
          if (info.size == p::BlockSize::k32x8 && info.mi_row == 0 &&
              info.mi_col == 0) {
            return kVert3;
          }
          return kNone;
        });
    CHECK(frame.status == pe::DecisionStatus::kOk);
    ps::FrameStream stream;
    std::string error;
    CHECK(ps::build_frame_stream(g, both, frame, stream, error));
    if (!error.empty()) std::cerr << error << '\n';
    check_stream_against_tree(stream, frame);

    std::size_t vert3 = stream.nodes.size();
    for (std::size_t i = 0; i < stream.nodes.size(); ++i) {
      if (stream.nodes[i].partition == static_cast<std::uint8_t>(kVert3)) {
        vert3 = i;
        break;
      }
    }
    CHECK(vert3 + 4 < stream.nodes.size());
    if (vert3 + 4 < stream.nodes.size()) {
      const bool expected_chroma[4] = {true, false, true, true};
      const p::BlockSize expected_size[4] = {
          p::BlockSize::k8x8, p::BlockSize::k16x8, p::BlockSize::k16x8,
          p::BlockSize::k8x8};
      for (int i = 0; i < 4; ++i) {
        const ps::M1Node& child =
            stream.nodes[vert3 + 1u + static_cast<std::size_t>(i)];
        CHECK_EQ(child.has_chroma, expected_chroma[i]);
        CHECK(child.chroma.size == expected_size[i]);
      }
      CHECK_EQ(stream.nodes[vert3 + 2].chroma_offset, 1);
      CHECK_EQ(stream.nodes[vert3 + 3].chroma_offset, 1);
      const ps::ChromaPlaneBlock mid =
          ps::chroma_plane_block(g, stream.nodes[vert3 + 3].chroma);
      CHECK_EQ(mid.mi_row, 0);
      CHECK_EQ(mid.mi_col, 1);
      CHECK(mid.residual_size == p::BlockSize::k8x4);
    }
  }
}

std::vector<pe::LeafBlock> frame_leaves(const pe::FrameDecision& frame) {
  std::vector<pe::LeafBlock> leaves;
  for (const pe::PartitionNode& sb : frame.superblocks) {
    pe::collect_leaves(sb, leaves);
  }
  return leaves;
}

void check_roundtrip_costs(const p::FrameGeometry& geometry,
                           const pe::Tools& tools, const pe::CostFunction& costs,
                           bool adaptive, const char* label) {
  const pe::FrameDecision frame =
      pe::decide_frame(geometry, costs, tools, default_options());
  if (frame.status != pe::DecisionStatus::kOk) {
    ++g_failures;
    std::cerr << label << ": decision failed: " << frame.error << '\n';
    return;
  }
  std::string error;
  if (!pe::validate_frame(geometry, tools, frame, error)) {
    ++g_failures;
    std::cerr << label << ": validate failed: " << error << '\n';
    return;
  }

  ps::FrameStream stream;
  if (!ps::build_frame_stream(geometry, tools, frame, stream, error)) {
    ++g_failures;
    std::cerr << label << ": stream failed: " << error << '\n';
    return;
  }

  ps::TilePartitionCdfs enc_cdfs;
  ps::init_tile_partition_cdfs(enc_cdfs);
  std::vector<std::uint8_t> bytes;
  pe::SyntaxStats enc_stats{};
  const bool disable_cdf_update = !adaptive;
  if (!ps::encode_frame_partitions(geometry, tools, frame, enc_cdfs,
                                   disable_cdf_update, bytes, &enc_stats,
                                   error)) {
    ++g_failures;
    std::cerr << label << ": encode failed: " << error << '\n';
    return;
  }
  CHECK(enc_stats.nodes == frame.nodes);
  CHECK(enc_stats.leaves == frame.leaves);

  // Byte-identical to partition_ext's reference encoder for a single SB.
  if (frame.superblocks.size() == 1) {
    ps::TilePartitionCdfs a;
    ps::TilePartitionCdfs b;
    ps::init_tile_partition_cdfs(a);
    ps::init_tile_partition_cdfs(b);
    std::vector<std::uint8_t> adapter_bytes;
    if (!ps::encode_superblock_partitions(
            geometry, tools, frame.superblocks[0], a, disable_cdf_update,
            adapter_bytes, nullptr, error)) {
      ++g_failures;
      std::cerr << label << ": per-SB encode failed: " << error << '\n';
    } else {
      std::vector<std::uint8_t> ref_buffer(1u << 20, 0);
      av2::Encoder ref;
      ref.init_symbol(ref_buffer.data(), ref_buffer.size());
      p::MiSizeGrid ref_grid(geometry);
      pe::SyntaxStats ref_stats{};
      if (!pe::encode_tree_symbols(geometry, tools, frame.superblocks[0], b.ext,
                                   ref_grid, ref, disable_cdf_update,
                                   &ref_stats, error)) {
        ++g_failures;
        std::cerr << label << ": reference encode failed: " << error << '\n';
      } else {
        std::size_t ref_size = 0;
        ref.exit_symbol(&ref_size);
        std::vector<std::uint8_t> ref_bytes(
            ref_buffer.begin(),
            ref_buffer.begin() + static_cast<std::ptrdiff_t>(ref_size));
        CHECK(adapter_bytes == ref_bytes);
        CHECK_EQ(enc_stats.symbols, ref_stats.symbols);
        CHECK_EQ(enc_stats.literal_bits, ref_stats.literal_bits);
      }
    }
  }

  // Independent decoder.
  ps::TilePartitionCdfs dec_cdfs;
  ps::init_tile_partition_cdfs(dec_cdfs);
  av2::Decoder decoder;
  if (decoder.init_symbol(bytes.data(), bytes.size(), bytes.size()) !=
      av2::Status::kOk) {
    ++g_failures;
    std::cerr << label << ": decoder init failed\n";
    return;
  }
  pe::FrameDecision decoded;
  pe::SyntaxStats dec_stats{};
  if (!ps::decode_frame_partitions(geometry, tools, dec_cdfs, decoder,
                                   disable_cdf_update, decoded, &dec_stats,
                                   error)) {
    ++g_failures;
    std::cerr << label << ": decode failed: " << error << '\n';
    return;
  }
  CHECK_EQ(dec_stats.symbols, enc_stats.symbols);
  CHECK_EQ(dec_stats.literal_bits, enc_stats.literal_bits);
  CHECK_EQ(decoded.superblocks.size(), frame.superblocks.size());
  const std::size_t count =
      std::min(decoded.superblocks.size(), frame.superblocks.size());
  for (std::size_t i = 0; i < count; ++i) {
    CHECK_STR(pe::tree_signature(decoded.superblocks[i]),
              pe::tree_signature(frame.superblocks[i]));
  }
  const std::vector<pe::LeafBlock> want = frame_leaves(frame);
  const std::vector<pe::LeafBlock> got = frame_leaves(decoded);
  CHECK_EQ(got.size(), want.size());
  const std::size_t leaf_count = std::min(got.size(), want.size());
  for (std::size_t i = 0; i < leaf_count; ++i) {
    CHECK(got[i].size == want[i].size);
    CHECK_EQ(got[i].mi_row, want[i].mi_row);
    CHECK_EQ(got[i].mi_col, want[i].mi_col);
    CHECK_EQ(got[i].has_chroma, want[i].has_chroma);
    CHECK_EQ(got[i].chroma_offset, want[i].chroma_offset);
    CHECK(got[i].chroma == want[i].chroma);
  }

  // Decoded trees rebuild the same preorder stream (including leaf chroma).
  ps::FrameStream decoded_stream;
  if (ps::build_frame_stream(geometry, tools, decoded, decoded_stream,
                             error)) {
    CHECK_EQ(decoded_stream.nodes.size(), stream.nodes.size());
    CHECK_EQ(decoded_stream.leaves.size(), stream.leaves.size());
    const std::size_t n =
        std::min(decoded_stream.leaves.size(), stream.leaves.size());
    for (std::size_t i = 0; i < n; ++i) {
      CHECK(decoded_stream.leaves[i].chroma == stream.leaves[i].chroma);
      CHECK_EQ(decoded_stream.leaves[i].has_chroma,
               stream.leaves[i].has_chroma);
    }
  } else {
    ++g_failures;
    std::cerr << label << ": decoded stream failed: " << error << '\n';
  }
}

void check_roundtrip(const p::FrameGeometry& geometry, const pe::Tools& tools,
                     const std::function<pe::PartitionType(
                         const pe::NodeInfo&)>& chooser,
                     bool adaptive, const char* label) {
  check_roundtrip_costs(geometry, tools, choose_partition(chooser), adaptive,
                        label);
}

void test_random_roundtrips() {
  pe::Tools off;
  pe::Tools ext;
  ext.enable_ext_partitions = true;
  pe::Tools both;
  both.enable_ext_partitions = true;
  both.enable_uneven_4way_partitions = true;
  const pe::Tools* variants[] = {&off, &ext, &both};
  const char* names[] = {"off", "ext", "both"};
  const p::FrameGeometry geometries[] = {
      geometry_420(64, 64, 64),   geometry_420(96, 64, 64),
      geometry_420(80, 48, 64),   geometry_420(128, 128, 128),
      geometry_420(192, 96, 128), geometry_420(64, 128, 64),
  };
  for (std::size_t v = 0; v < 3; ++v) {
    for (const p::FrameGeometry& geometry : geometries) {
      for (unsigned seed = 1; seed <= 3; ++seed) {
        for (bool adaptive : {false, true}) {
          const std::string label =
              std::string("random/") + names[v] + '/' +
              std::to_string(geometry.width) + 'x' +
              std::to_string(geometry.height) + "/sb" +
              std::to_string(geometry.sb_size) + "/seed" +
              std::to_string(seed) + (adaptive ? "/adaptive" : "/frozen");
          check_roundtrip_costs(geometry, *variants[v],
                                ::random_cost(static_cast<unsigned>(seed * 7u + v)), adaptive,
                                label.c_str());
        }
      }
    }
  }
}

void test_roundtrips() {
  pe::Tools off;
  pe::Tools ext;
  ext.enable_ext_partitions = true;
  pe::Tools both;
  both.enable_ext_partitions = true;
  both.enable_uneven_4way_partitions = true;

  const auto chooser = [](const pe::NodeInfo& info) {
    if (info.depth == 0 && info.size == p::BlockSize::k128x128) return kSplit;
    if (info.size == p::BlockSize::k64x64) {
      return (info.mi_col / 16 + info.mi_row / 16) % 2 == 0 ? kHorz3
                                                             : kVert3;
    }
    if (info.size == p::BlockSize::k32x32) {
      return (info.mi_row / 8) % 2 == 0 ? kHorz4A : kVert4B;
    }
    if (info.size == p::BlockSize::k16x16) return kHorz;
    return kNone;
  };

  struct Case {
    p::FrameGeometry geometry;
    const pe::Tools* tools;
    const char* label;
  };
  const Case cases[] = {
      {geometry_420(64, 64, 64), &off, "64sb/off"},
      {geometry_420(64, 64, 64), &ext, "64sb/ext"},
      {geometry_420(64, 64, 64), &both, "64sb/both"},
      {geometry_420(640, 360, 64), &both, "64sb-640x360/both"},
      {geometry_420(96, 64, 64), &both, "64sb-edge/both"},
      {geometry_420(128, 128, 128), &off, "128sb/off"},
      {geometry_420(128, 128, 128), &ext, "128sb/ext"},
      {geometry_420(128, 128, 128), &both, "128sb/both"},
      {geometry_420(192, 192, 128), &both, "128sb-192x192/both"},
      {geometry_420(320, 180, 128), &both, "128sb-320x180/both"},
  };

  for (const Case& test : cases) {
    for (bool adaptive : {false, true}) {
      const std::string label =
          std::string(test.label) + (adaptive ? "/adaptive" : "/frozen");
      check_roundtrip(test.geometry, *test.tools, chooser, adaptive,
                      label.c_str());
    }
  }
}

void test_cost_adapter() {
  const p::FrameGeometry g = geometry_420(64, 64, 64);
  bool saw_base_info = false;
  p::PartitionSet saw_allowed{};
  p::CostFunction rect = [&](const p::PartitionNodeInfo& info) {
    saw_base_info = true;
    saw_allowed = info.allowed;
    p::NodeCosts costs;
    const double w = static_cast<double>(p::block_width(info.size));
    const double h = static_cast<double>(p::block_height(info.size));
    costs.leaf = w * h * w * h;
    costs.none_signal = 0.0;
    costs.split_signal = p::kInfiniteCost;
    costs.horz_signal = 1.0;
    costs.vert_signal = 1.0;
    return costs;
  };
  ps::ExtSignalCosts ext;
  ext.three_way = 0.5;
  ext.uneven_4way = pe::kInfiniteCost;
  const pe::CostFunction lifted = ps::lift_rect_costs(rect, ext);

  pe::NodeInfo info;
  info.mi_row = 0;
  info.mi_col = 0;
  info.size = p::BlockSize::k64x64;
  info.has_chroma = true;
  info.allowed = pe::init_allowed_partitions(
      g, 0, 0, info.size, true, false, [] {
        pe::Tools t;
        t.enable_ext_partitions = true;
        return t;
      }());
  const pe::NodeCosts costs = lifted(info);
  CHECK(saw_base_info);
  // The compact base enum is mapped from spec values, not shifted blindly.
  CHECK(saw_allowed.is_allowed(p::PartitionType::kNone));
  CHECK(saw_allowed.is_allowed(p::PartitionType::kHorz));
  CHECK(saw_allowed.is_allowed(p::PartitionType::kVert));
  CHECK(!saw_allowed.is_allowed(p::PartitionType::kSplit));
  CHECK_EQ(saw_allowed.count, 3);
  CHECK_EQ(costs.leaf, 64.0 * 64.0 * 64.0 * 64.0);
  CHECK_EQ(costs.none_signal, 0.0);
  CHECK_EQ(costs.horz_signal, 1.0);
  CHECK_EQ(costs.horz3_signal, 0.5);
  CHECK_EQ(costs.vert3_signal, 0.5);
  CHECK(costs.horz4a_signal == pe::kInfiniteCost);
  CHECK(costs.vert4b_signal == pe::kInfiniteCost);

  // End to end: the lifted cost model drives a HORZ_3 decision.
  pe::DecisionOptions options = default_options();
  options.max_depth = 1;
  pe::Tools tools;
  tools.enable_ext_partitions = true;
  const pe::FrameDecision frame =
      pe::decide_frame(g, lifted, tools, options);
  CHECK(frame.status == pe::DecisionStatus::kOk);
  CHECK(frame.superblocks[0].partition == kHorz3);
  CHECK_EQ(frame.leaves, 4);
}

void test_block_syntax_v2() {
  ps::M1Node node;
  node.mi_row = 3;
  node.mi_col = 5;
  node.bsize = static_cast<std::uint8_t>(p::BlockSize::k32x32);
  node.partition = static_cast<std::uint8_t>(kHorz4B);
  node.allowed_mask = 0x1FFu;
  node.flags = av2enc::m1::syntax::kFlagForced;
  node.subtree = 4;
  node.has_chroma = false;
  node.chroma_offset = true;
  node.chroma.mi_row = 2;
  node.chroma.mi_col = 4;
  node.chroma.size = p::BlockSize::k32x32;
  const ps::BlockSyntaxV2 v2 = ps::to_block_syntax_v2(node);
  CHECK_EQ(v2.mi_row, 3);
  CHECK_EQ(v2.mi_col, 5);
  CHECK_EQ(v2.bsize, static_cast<int>(p::BlockSize::k32x32));
  CHECK_EQ(v2.partition, static_cast<int>(kHorz4B));
  CHECK_EQ(v2.allowed_mask, 0x1FF);
  CHECK_EQ(v2.flags, av2enc::m1::syntax::kFlagForced);
  CHECK_EQ(v2.subtree, 4);
  CHECK_EQ(v2.has_chroma, 0);
  CHECK_EQ(v2.chroma_bsize, static_cast<int>(p::BlockSize::k32x32));
  CHECK_EQ(v2.chroma_mi_row, 2);
  CHECK_EQ(v2.chroma_mi_col, 4);
}

}  // namespace

int main() {
  test_cdfs();
  test_table_parity();
  test_allowed_mask();
  test_contexts();
  test_plan_symbols();
  test_stream_shape();
  test_chroma_carry();
  test_roundtrips();
  test_random_roundtrips();
  test_cost_adapter();
  test_block_syntax_v2();

  if (g_failures != 0) {
    std::cerr << "partition_stream_test: " << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "partition_stream_test: PASS\n";
  return 0;
}
