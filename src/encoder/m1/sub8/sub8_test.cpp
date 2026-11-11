// Tests for the sub-8 coding-block reference rules and the minimal syntax
// writer/reader.  No external framework: a failure prints and sets a counter,
// the process exits non-zero.
#include "encoder/m1/sub8/sub8.h"
#include "encoder/m1/sub8/sub8_minimal.h"
#include "encoder/m1/sub8/sub8_syntax.h"

#include "encoder/m1/m1_tables.h"
#include "encoder/partition/partition.h"
#include "encoder/tx_variants/tx_variants.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace av2enc::m1::sub8;
using av2enc::partition::BlockSize;
using av2enc::partition::PartitionType;

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

#define CHECK_EQ(a, b)                                                        \
  do {                                                                        \
    ++g_checks;                                                               \
    const auto lhs_ = (a);                                                    \
    const auto rhs_ = (b);                                                    \
    if (!(lhs_ == rhs_)) {                                                    \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s\n", __FILE__, __LINE__, #a, \
                   #b);                                                       \
      ++g_failures;                                                           \
    }                                                                         \
  } while (0)

FrameGeometry make_geometry(int width, int height) {
  FrameGeometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;
  return geometry;
}

// --------------------------------------------------------------------------
// Predicates
// --------------------------------------------------------------------------
void test_predicates() {
  const BlockSize sub8[] = {BlockSize::k4x4,  BlockSize::k4x8,
                            BlockSize::k8x4,  BlockSize::k4x16,
                            BlockSize::k16x4, BlockSize::k4x32,
                            BlockSize::k32x4};
  for (BlockSize size : sub8) {
    CHECK(is_sub8(size));
    CHECK_EQ(min_dim_px(size), 4);
  }
  const BlockSize not_sub8[] = {BlockSize::k8x8,   BlockSize::k8x16,
                                BlockSize::k16x8,  BlockSize::k16x16,
                                BlockSize::k32x8,  BlockSize::k8x32,
                                BlockSize::k64x16, BlockSize::k16x64,
                                BlockSize::k8x64,  BlockSize::k64x8};
  for (BlockSize size : not_sub8) {
    CHECK(!is_sub8(size));
    CHECK(min_dim_px(size) >= 8);
  }
  CHECK(!is_sub8(BlockSize::kInvalid));
}

// --------------------------------------------------------------------------
// Partition validity sweep against the partition module
// --------------------------------------------------------------------------
void test_partition_sweep() {
  const int frames[][2] = {{8, 8},   {12, 8},  {8, 12},
                           {16, 16}, {20, 12}, {12, 20}};
  for (const auto& frame : frames) {
    const FrameGeometry geometry = make_geometry(frame[0], frame[1]);
    const int mi_rows = av2enc::partition::mi_rows(geometry);
    const int mi_cols = av2enc::partition::mi_cols(geometry);
    for (int size_index = 0; size_index < av2enc::partition::kBlockSizeCount;
         ++size_index) {
      const auto size = static_cast<BlockSize>(size_index);
      if (!av2enc::partition::is_valid_block_size(size)) continue;
      for (int mi_row = 0; mi_row < mi_rows; ++mi_row) {
        for (int mi_col = 0; mi_col < mi_cols; ++mi_col) {
          for (int chroma = 0; chroma < 4; ++chroma) {
            const bool has_chroma = (chroma & 1) != 0;
            const bool chroma_offset = (chroma & 2) != 0;
            const PartitionPlan plan = plan_partition(
                geometry, mi_row, mi_col, size, has_chroma, chroma_offset);
            const auto allowed = av2enc::partition::init_allowed_partitions(
                geometry, mi_row, mi_col, size, has_chroma, chroma_offset);
            const auto implied = av2enc::partition::partition_implied(
                geometry, mi_row, mi_col, size);
            const auto rect =
                av2enc::partition::rect_type_implied_by_bsize(size);
            CHECK_EQ(plan.allowed.count, allowed.count);
            for (int p = 0; p < av2enc::partition::kPartitionTypeCount; ++p) {
              CHECK_EQ(plan.allowed.allowed[p], allowed.allowed[p]);
            }
            CHECK_EQ(plan.implied.has_rows, implied.has_rows);
            CHECK_EQ(plan.implied.has_cols, implied.has_cols);
            CHECK_EQ(plan.implied.implied, implied.implied);
            CHECK_EQ(plan.implied.implied_partition,
                     implied.implied_partition);
            CHECK_EQ(plan.rect_type.implied, rect.implied);
            CHECK_EQ(plan.rect_type.partition, rect.partition);
            CHECK_EQ(plan.none_allowed,
                     allowed.is_allowed(PartitionType::kNone));
            CHECK_EQ(plan.allow_horz,
                     allowed.is_allowed(PartitionType::kHorz));
            CHECK_EQ(plan.allow_vert,
                     allowed.is_allowed(PartitionType::kVert));

            const bool spec_forced =
                (implied.implied &&
                 allowed.is_allowed(implied.implied_partition)) ||
                allowed.count == 1;
            CHECK_EQ(plan.forced, spec_forced);
            if (plan.forced) {
              CHECK(!plan.codes_do_split);
              CHECK(!plan.codes_do_square_split);
              CHECK(!plan.codes_rect_type);
            } else {
              CHECK_EQ(plan.codes_do_split, plan.none_allowed);
              CHECK_EQ(plan.codes_do_square_split,
                       allowed.is_allowed(PartitionType::kSplit));
              CHECK_EQ(plan.codes_rect_type,
                       !rect.implied && plan.allow_horz && plan.allow_vert);
            }
            for (int p = 0; p < av2enc::partition::kPartitionTypeCount; ++p) {
              const auto partition = static_cast<PartitionType>(p);
              CHECK_EQ(partition_allowed(geometry, mi_row, mi_col, size,
                                         partition, has_chroma, chroma_offset),
                       allowed.is_allowed(partition));
            }
          }
        }
      }
    }
  }
}

// --------------------------------------------------------------------------
// Sub-8 specifics: implied rules, aspect, boundary fallbacks
// --------------------------------------------------------------------------
void test_sub8_partition_cases() {
  // 4x4 is always terminal: partition_implied() returns NONE.
  {
    const FrameGeometry geometry = make_geometry(16, 16);
    const PartitionPlan plan = plan_partition(geometry, 1, 1, BlockSize::k4x4,
                                              true, false);
    CHECK(plan.forced);
    CHECK_EQ(plan.forced_partition, PartitionType::kNone);
    CHECK(!plan.codes_do_split);
  }
  // 4x8 only supports HORZ; rect_type is implied, not coded.
  {
    const FrameGeometry geometry = make_geometry(16, 16);
    const PartitionPlan plan = plan_partition(geometry, 0, 0, BlockSize::k4x8,
                                              true, false);
    CHECK(!plan.forced);
    CHECK(plan.allow_horz);
    CHECK(!plan.allow_vert);
    CHECK(!plan.codes_rect_type);
    CHECK(!plan.allowed.is_allowed(PartitionType::kVert));
    CHECK(!plan.allowed.is_allowed(PartitionType::kSplit));
  }
  // 8x4 mirrors it.
  {
    const FrameGeometry geometry = make_geometry(16, 16);
    const PartitionPlan plan = plan_partition(geometry, 0, 0, BlockSize::k8x4,
                                              true, false);
    CHECK(!plan.forced);
    CHECK(!plan.allow_horz);
    CHECK(plan.allow_vert);
    CHECK(!plan.codes_rect_type);
  }
  // 4x16 / 16x4 are still splittable in the implied direction.
  {
    const FrameGeometry geometry = make_geometry(32, 32);
    const PartitionPlan horz = plan_partition(geometry, 0, 0,
                                              BlockSize::k4x16, true, false);
    CHECK(horz.codes_do_split);
    CHECK(horz.allow_horz);
    CHECK(!horz.allow_vert);
    const PartitionPlan vert = plan_partition(geometry, 0, 0,
                                              BlockSize::k16x4, true, false);
    CHECK(vert.codes_do_split);
    CHECK(!vert.allow_horz);
    CHECK(vert.allow_vert);
  }
  // >= BLOCK_4X32 is terminal (partition_implied -> NONE).
  const BlockSize terminal[] = {BlockSize::k4x32, BlockSize::k32x4,
                                BlockSize::k8x64, BlockSize::k64x8};
  for (BlockSize size : terminal) {
    const FrameGeometry geometry = make_geometry(128, 128);
    const PartitionPlan plan =
        plan_partition(geometry, 0, 0, size, true, false);
    CHECK(plan.forced);
    CHECK_EQ(plan.forced_partition, PartitionType::kNone);
  }
  // A 4x8 at the bottom coded row with chroma can only be a leaf: HORZ needs
  // its second child in frame, and NONE violates the half-block test.
  {
    const FrameGeometry geometry = make_geometry(8, 8);
    CHECK_EQ(av2enc::partition::mi_rows(geometry), 2);
    const PartitionPlan plan = plan_partition(geometry, 1, 0, BlockSize::k4x8,
                                              true, true);
    CHECK(plan.allowed.count == 1);
    CHECK(plan.forced);
    CHECK_EQ(plan.forced_partition, PartitionType::kNone);
    CHECK(!plan.allowed.is_allowed(PartitionType::kHorz));
  }
  // A 4x8 at the right edge of a 12x8 frame (MiCols = 4) is still codable.
  {
    const FrameGeometry geometry = make_geometry(12, 8);
    CHECK_EQ(av2enc::partition::mi_cols(geometry), 4);
    const PartitionPlan plan = plan_partition(geometry, 0, 3, BlockSize::k4x8,
                                              true, true);
    CHECK(plan.codes_do_split);
    CHECK(plan.allow_horz);
    CHECK_EQ(av2enc::partition::partition_subsize(PartitionType::kHorz,
                                                  BlockSize::k4x8),
             BlockSize::k4x4);
  }
  // MaxPbAspectRatio constrains the subsize, not the current block.  A 64x16
  // node can split VERT (32x16, ratio 2) but HORZ (64x8, ratio 8) is rejected
  // when the sequence header lowers the limit to 2.
  {
    FrameGeometry geometry = make_geometry(128, 128);
    geometry.max_pb_aspect_ratio = 2;
    const PartitionPlan plan = plan_partition(geometry, 0, 0,
                                              BlockSize::k64x16, true, false);
    CHECK(plan.allow_vert);
    CHECK(!plan.allow_horz);
  }
  // The default M1 limit is 8, where 64x8 is exactly at the limit.
  {
    const FrameGeometry geometry = make_geometry(128, 128);
    const PartitionPlan plan = plan_partition(geometry, 0, 0,
                                              BlockSize::k64x16, true, false);
    CHECK(plan.allow_horz);
    CHECK(plan.allow_vert);
  }
  // A 4x4 in the bottom-right coded cell is still a forced leaf.
  {
    const FrameGeometry geometry = make_geometry(8, 8);
    const PartitionPlan plan = plan_partition(geometry, 1, 1, BlockSize::k4x4,
                                              true, true);
    CHECK(plan.forced);
    CHECK_EQ(plan.forced_partition, PartitionType::kNone);
  }
  // 8x8 is the first size that can code both rect directions and therefore
  // rect_type when both are allowed.
  {
    const FrameGeometry geometry = make_geometry(16, 16);
    const PartitionPlan plan = plan_partition(geometry, 0, 0, BlockSize::k8x8,
                                              true, false);
    CHECK(!plan.forced);
    CHECK(plan.allow_horz && plan.allow_vert);
    CHECK(plan.codes_rect_type);
    CHECK(!plan.allowed.is_allowed(PartitionType::kSplit));
  }
}

// --------------------------------------------------------------------------
// Chroma ownership
// --------------------------------------------------------------------------
struct ShapeCase {
  const char* name;
  int width;
  int height;
  const char* mode;  // N/V/H/VS/HS/VF/VL/HF/HL
  bool edge16;
};

PartitionType choose_shape(const Node& node, const PartitionPlan&,
                           const char* mode, bool edge16) {
  if (edge16) {
    if (node.size == BlockSize::k16x16) return PartitionType::kHorz;
    if (node.size == BlockSize::k16x8) return PartitionType::kVert;
    if (node.size == BlockSize::k8x8) return PartitionType::kHorz;
    if (node.size == BlockSize::k8x4) return PartitionType::kVert;
    return PartitionType::kNone;
  }
  const bool first = mode[1] == 'F' || mode[1] == 'S';
  const bool last = mode[1] == 'L' || mode[1] == 'S';
  if (node.size == BlockSize::k8x8) {
    if (mode[0] == 'V') return PartitionType::kVert;
    if (mode[0] == 'H') return PartitionType::kHorz;
    return PartitionType::kNone;
  }
  if (node.size == BlockSize::k4x8 && first && node.mi_col == 0) {
    return PartitionType::kHorz;
  }
  if (node.size == BlockSize::k4x8 && last && node.mi_col != 0) {
    return PartitionType::kHorz;
  }
  if (node.size == BlockSize::k8x4 && first && node.mi_row == 0) {
    return PartitionType::kVert;
  }
  if (node.size == BlockSize::k8x4 && last && node.mi_row != 0) {
    return PartitionType::kVert;
  }
  return PartitionType::kNone;
}

bool build_case(const ShapeCase& shape_case, Node& root, std::string& error) {
  const FrameGeometry geometry =
      make_geometry(shape_case.width, shape_case.height);
  const std::string mode = shape_case.mode;
  const bool edge16 = shape_case.edge16;
  const auto chooser = [mode, edge16](const Node& node,
                                      const PartitionPlan& plan) {
    return choose_shape(node, plan, mode.c_str(), edge16);
  };
  return build_superblock(geometry, 0, 0, chooser, root, error);
}

// Converts a sub8::Node into a partition::PartitionNode so the partition
// module's own validate_tree() can re-derive hasChroma/chromaOffset.
av2enc::partition::PartitionNode to_partition_node(
    const FrameGeometry& geometry, const Node& node, int depth) {
  av2enc::partition::PartitionNode out;
  out.mi_row = node.mi_row;
  out.mi_col = node.mi_col;
  out.size = node.size;
  out.partition = node.partition;
  out.depth = depth;
  out.has_chroma = node.has_chroma;
  out.chroma_offset = node.chroma_offset;
  const PartitionPlan plan =
      plan_partition(geometry, node.mi_row, node.mi_col, node.size,
                     node.has_chroma, node.chroma_offset);
  out.boundary = plan.implied;
  out.allowed = plan.allowed;
  out.forced = plan.forced;
  for (const Node& child : node.children) {
    out.children.push_back(to_partition_node(geometry, child, depth + 1));
  }
  return out;
}

void test_chroma_ownership() {
  const ShapeCase cases[] = {
      {"baseline_none", 8, 8, "N", false},
      {"vert_leaf4x8", 8, 8, "V", false},
      {"vert_first4x4", 8, 8, "VF", false},
      {"vert_last4x4", 8, 8, "VL", false},
      {"vert_all4x4", 8, 8, "VS", false},
      {"horz_leaf8x4", 8, 8, "H", false},
      {"horz_first4x4", 8, 8, "HF", false},
      {"horz_last4x4", 8, 8, "HL", false},
      {"horz_all4x4", 8, 8, "HS", false},
      {"edge16_all4x4", 16, 16, "", true},
  };
  for (const ShapeCase& shape_case : cases) {
    Node root;
    std::string error;
    CHECK(build_case(shape_case, root, error));
    if (!error.empty()) {
      std::fprintf(stderr, "  %s: %s\n", shape_case.name, error.c_str());
      continue;
    }
    const FrameGeometry geometry =
        make_geometry(shape_case.width, shape_case.height);
    const auto partition_root = to_partition_node(geometry, root, 0);
    CHECK(av2enc::partition::validate_tree(geometry, partition_root, error));
    if (!error.empty()) std::fprintf(stderr, "  validate: %s\n", error.c_str());

    std::vector<const Node*> leaves;
    collect_leaves(root, leaves);
    int owners = 0;
    for (const Node* leaf : leaves) {
      if (!leaf->has_chroma) continue;
      ++owners;
      CHECK(leaf->chroma.valid);
      CHECK(av2enc::partition::is_valid_block_size(leaf->chroma.size));
      const BlockSize residual = chroma_residual_size(geometry, leaf->chroma);
      CHECK(av2enc::partition::is_valid_block_size(residual));
      CHECK(av2enc::partition::block_width(residual) >= 4);
      CHECK(av2enc::partition::block_height(residual) >= 4);
    }
    CHECK(owners > 0);
  }

  // The exact ownership pattern of an 8x8 VERT+HORZ tree: only the last leaf.
  {
    Node root;
    std::string error;
    CHECK(build_case({"vert_all4x4", 8, 8, "VS", false}, root, error));
    std::vector<const Node*> leaves;
    collect_leaves(root, leaves);
    CHECK_EQ(leaves.size(), std::size_t{4});
    const int expect_rows[] = {0, 1, 0, 1};
    const int expect_cols[] = {0, 0, 1, 1};
    for (std::size_t i = 0; i < leaves.size(); ++i) {
      CHECK_EQ(leaves[i]->mi_row, expect_rows[i]);
      CHECK_EQ(leaves[i]->mi_col, expect_cols[i]);
      CHECK_EQ(leaves[i]->size, BlockSize::k4x4);
      CHECK_EQ(leaves[i]->has_chroma, i == leaves.size() - 1);
    }
    for (const Node* leaf : leaves) {
      CHECK_EQ(leaf->chroma.mi_row, 0);
      CHECK_EQ(leaf->chroma.mi_col, 0);
      CHECK_EQ(leaf->chroma.size, BlockSize::k8x8);
    }
  }

  // 16x8 VERT -> two 8x8, each HORZ -> 8x4 pair.  Each 8x8 group owns its own
  // chroma (the 16x8 subsize 8x8 is not a chroma offset), so there are two
  // owners at (0,0) and (0,1), each a 4x4 chroma residual.
  {
    const FrameGeometry geometry = make_geometry(16, 8);
    const auto chooser = [](const Node& node, const PartitionPlan&) {
      if (node.size == BlockSize::k16x8) return PartitionType::kVert;
      if (node.size == BlockSize::k8x8) return PartitionType::kHorz;
      return PartitionType::kNone;
    };
    Node root;
    std::string error;
    CHECK(build_superblock(geometry, 0, 0, chooser, root, error));
    std::vector<const Node*> leaves;
    collect_leaves(root, leaves);
    int owners = 0;
    for (const Node* leaf : leaves) {
      if (!leaf->has_chroma) continue;
      ++owners;
      CHECK_EQ(leaf->size, BlockSize::k8x4);
      CHECK_EQ(leaf->chroma.size, BlockSize::k8x8);
      CHECK_EQ(leaf->chroma.mi_row, 0);
      CHECK_EQ(chroma_residual_size(geometry, leaf->chroma), BlockSize::k4x4);
    }
    CHECK_EQ(owners, 2);
  }

  // Rejects a caller that picks a forbidden partition.
  {
    const FrameGeometry geometry = make_geometry(8, 8);
    const auto bad = [](const Node& node, const PartitionPlan&) {
      if (node.size == BlockSize::k8x8) return PartitionType::kVert;
      if (node.size == BlockSize::k4x8) return PartitionType::kVert;
      return PartitionType::kNone;
    };
    Node root;
    std::string error;
    CHECK(!build_superblock(geometry, 0, 0, bad, root, error));
    CHECK(!error.empty());
  }
}

// --------------------------------------------------------------------------
// Mode info plan
// --------------------------------------------------------------------------
void test_mode_info() {
  Node root;
  std::string error;
  CHECK(build_case({"vert_all4x4", 8, 8, "VS", false}, root, error));
  std::vector<const Node*> leaves;
  collect_leaves(root, leaves);
  const FrameGeometry geometry = make_geometry(8, 8);
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    const ModeInfoPlan plan = mode_info_plan(geometry, *leaves[i], true);
    CHECK(plan.codes_y_mode);
    CHECK_EQ(plan.codes_uv_mode, leaves[i]->has_chroma);
    CHECK(!plan.palette_allowed);  // sub-8: palette is never allowed
    if (leaves[i]->has_chroma) {
      CHECK_EQ(plan.uv_residual_size, BlockSize::k4x4);
      CHECK(plan.cfl_allowed);
    } else {
      CHECK_EQ(plan.uv_residual_size, BlockSize::kInvalid);
    }
  }
  // CfL disabled -> no is_cfl symbol but uv_mode is still present.
  const ModeInfoPlan dc = mode_info_plan(geometry, *leaves.back(), false);
  CHECK(dc.codes_uv_mode);
  CHECK(!dc.cfl_allowed);

  // An 8x8 leaf also never uses palette in the M1 configuration.
  Node root8;
  CHECK(build_case({"baseline_none", 8, 8, "N", false}, root8, error));
  std::vector<const Node*> single;
  collect_leaves(root8, single);
  CHECK_EQ(single.size(), std::size_t{1});
  const ModeInfoPlan plan8 = mode_info_plan(geometry, *single[0], true);
  CHECK(plan8.codes_uv_mode);
  CHECK(!plan8.palette_allowed);
  CHECK_EQ(plan8.uv_residual_size, BlockSize::k4x4);
}

// --------------------------------------------------------------------------
// Coefficient contexts
// --------------------------------------------------------------------------
void test_coefficient_contexts() {
  // TX_MODE_LARGEST: plane residual size == transform size, luma ctx 0.
  CHECK_EQ(txb_skip_context(0, BlockSize::k4x4, 2, 2, 0, 0, false), 0);
  CHECK_EQ(txb_skip_context(0, BlockSize::k4x8, 2, 3, 0, 0, false), 0);
  CHECK_EQ(txb_skip_context(0, BlockSize::k8x4, 3, 2, 0, 0, false), 0);
  CHECK_EQ(txb_skip_context(0, BlockSize::k8x8, 3, 3, 0, 0, false), 0);
  // A transform smaller than the block uses the magnitude context.
  CHECK_EQ(txb_skip_context(0, BlockSize::k8x8, 2, 2, 1, 0, false),
           (1 + 0 + 3) >> 1);
  CHECK_EQ(txb_skip_context(0, BlockSize::k16x16, 2, 3, 2, 3, false),
           (2 + 3 + 3) >> 1);

  // Chroma U: +6, plus the neighbour OR.
  CHECK_EQ(txb_skip_context(1, BlockSize::k4x4, 2, 2, 0, 0, false), 6);
  CHECK_EQ(txb_skip_context(1, BlockSize::k4x4, 2, 2, 1, 0, false), 7);
  CHECK_EQ(txb_skip_context(1, BlockSize::k4x4, 2, 2, 6, 6, false), 8);
  // Chroma V: 6 * u_has_cf + 3 * (block larger than tx) + neighbours.
  CHECK_EQ(txb_skip_context(2, BlockSize::k4x4, 2, 2, 0, 0, false), 0);
  CHECK_EQ(txb_skip_context(2, BlockSize::k4x4, 2, 2, 0, 0, true), 6);
  CHECK_EQ(txb_skip_context(2, BlockSize::k4x4, 2, 2, 1, 1, false), 2);
  CHECK_EQ(txb_skip_context(2, BlockSize::k4x4, 2, 2, 1, 1, true), 8);
  CHECK_EQ(txb_skip_context(2, BlockSize::k8x4, 2, 2, 0, 0, false), 3);
  CHECK_EQ(txb_skip_context(2, BlockSize::k8x4, 2, 2, 0, 0, true), 9);
  CHECK_EQ(txb_skip_context(2, BlockSize::k8x8, 2, 2, 0, 0, false), 3);
}

// --------------------------------------------------------------------------
// Transform selection
// --------------------------------------------------------------------------
void test_tx_plan() {
  const FrameGeometry geometry = make_geometry(16, 16);
  struct Expect {
    BlockSize size;
    int tx;
    int multisize;
  };
  const Expect expects[] = {
      {BlockSize::k4x4, 0, 0},   {BlockSize::k4x8, 5, 1},
      {BlockSize::k8x4, 6, 1},   {BlockSize::k8x8, 1, 2},
      {BlockSize::k4x16, 13, 2}, {BlockSize::k16x4, 14, 2},
      {BlockSize::k8x16, 7, 3},  {BlockSize::k16x8, 8, 3},
  };
  for (const Expect& expect : expects) {
    Node leaf;
    leaf.size = expect.size;
    leaf.has_chroma = false;
    const TxPlan plan = tx_plan(geometry, leaf, 0);
    CHECK_EQ(plan.luma_tx, expect.tx);
    CHECK_EQ(plan.luma_eob_multisize, expect.multisize);
    CHECK_EQ(eob_multisize(plan.luma_tx), expect.multisize);
  }
  // Chroma follows ChromaMiSize, not the luma leaf.
  {
    Node leaf;
    leaf.size = BlockSize::k4x4;
    leaf.has_chroma = true;
    leaf.chroma.valid = true;
    leaf.chroma.mi_row = 0;
    leaf.chroma.mi_col = 0;
    leaf.chroma.size = BlockSize::k8x8;
    const TxPlan plan = tx_plan(geometry, leaf, 0);
    CHECK_EQ(plan.chroma_tx, 0);  // TX_4X4
    CHECK_EQ(plan.chroma_eob_multisize, 0);
    CHECK(!plan.codes_tx_partition);
  }
  // transform_type() is implied DCT for eob <= 1; above that, 4x4/4x8/8x4 all
  // use TX_SET_INTRA_1.
  {
    Node leaf;
    leaf.size = BlockSize::k4x8;
    leaf.has_chroma = false;
    CHECK(!tx_plan(geometry, leaf, 0).luma_tx_type_coded);
    CHECK(!tx_plan(geometry, leaf, 1).luma_tx_type_coded);
    CHECK(tx_plan(geometry, leaf, 2).luma_tx_type_coded);
  }
  CHECK(tx_type_set1(0));   // 4x4
  CHECK(tx_type_set1(5));   // 4x8
  CHECK(tx_type_set1(6));   // 8x4
  CHECK(tx_type_set1(1));   // 8x8
  CHECK(tx_type_set1(2));   // 16x16
  CHECK(!tx_type_set1(3));  // 32x32 -> TX_SET_DCTONLY for intra
}

// --------------------------------------------------------------------------
// Syntax writer/reader
// --------------------------------------------------------------------------
struct Golden {
  const char* name;
  int width;
  int height;
  const char* mode;
  bool edge16;
  const char* hex;
};

const Golden kGoldens[] = {
    {"baseline_none", 8, 8, "N", false, "0f0000"},
    {"vert_leaf4x8", 8, 8, "V", false, "f1460000"},
    {"vert_first4x4", 8, 8, "VF", false, "fe228c0000"},
    {"vert_last4x4", 8, 8, "VL", false, "f2b87c0000"},
    {"vert_all4x4", 8, 8, "VS", false, "fe26baa00000"},
    {"horz_leaf8x4", 8, 8, "H", false, "dfe30000"},
    {"horz_first4x4", 8, 8, "HF", false, "ea523c0000"},
    {"horz_last4x4", 8, 8, "HL", false, "e111140000"},
    {"horz_all4x4", 8, 8, "HS", false, "ea55a1a00000"},
    {"edge16_all4x4", 16, 16, "", true,
     "de32c59ee6e30710c4698c33890000"},
};

std::vector<std::uint8_t> hex_to_bytes(const char* hex) {
  std::vector<std::uint8_t> out;
  const std::size_t size = std::strlen(hex);
  for (std::size_t i = 0; i + 1 < size; i += 2) {
    const auto nibble = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      return 0;
    };
    out.push_back(static_cast<std::uint8_t>((nibble(hex[i]) << 4) |
                                            nibble(hex[i + 1])));
  }
  return out;
}

void test_syntax_goldens_and_round_trip() {
  for (const Golden& golden : kGoldens) {
    const ShapeCase shape_case{golden.name, golden.width, golden.height,
                               golden.mode, golden.edge16};
    Node root;
    std::string error;
    CHECK(build_case(shape_case, root, error));
    if (!error.empty()) {
      std::fprintf(stderr, "  %s: %s\n", golden.name, error.c_str());
      continue;
    }
    const FrameGeometry geometry = make_geometry(golden.width, golden.height);
    SyntaxOptions options;
    std::vector<std::uint8_t> tile;
    CHECK(write_tile(geometry, std::span<const Node>(&root, 1), options, tile,
                     error));
    if (!error.empty()) {
      std::fprintf(stderr, "  write %s: %s\n", golden.name, error.c_str());
      continue;
    }
    const std::vector<std::uint8_t> expect = hex_to_bytes(golden.hex);
    CHECK_EQ(tile.size(), expect.size());
    if (tile.size() == expect.size()) {
      CHECK(std::memcmp(tile.data(), expect.data(), tile.size()) == 0);
      if (std::memcmp(tile.data(), expect.data(), tile.size()) != 0) {
        std::fprintf(stderr, "  golden %s: got ", golden.name);
        for (std::uint8_t byte : tile) std::fprintf(stderr, "%02x", byte);
        std::fprintf(stderr, "\n");
      }
    }
    // Independent reader round trip.
    std::vector<DecodedLeaf> leaves;
    CHECK(read_tile(geometry, tile, options, leaves, error));
    if (!error.empty()) {
      std::fprintf(stderr, "  read %s: %s\n", golden.name, error.c_str());
      continue;
    }
    std::vector<const Node*> plant;
    collect_leaves(root, plant);
    CHECK_EQ(leaves.size(), plant.size());
    for (std::size_t i = 0; i < leaves.size() && i < plant.size(); ++i) {
      CHECK_EQ(leaves[i].mi_row, plant[i]->mi_row);
      CHECK_EQ(leaves[i].mi_col, plant[i]->mi_col);
      CHECK_EQ(leaves[i].size, plant[i]->size);
      CHECK_EQ(leaves[i].has_chroma, plant[i]->has_chroma);
      CHECK_EQ(leaves[i].y_mode, 0);
      if (leaves[i].has_chroma) {
        CHECK_EQ(leaves[i].uv_mode, 0);
        CHECK(leaves[i].luma_all_zero);
        CHECK(leaves[i].u_all_zero);
        CHECK(leaves[i].v_all_zero);
        CHECK_EQ(leaves[i].chroma.size, plant[i]->chroma.size);
        CHECK_EQ(leaves[i].chroma.mi_row, plant[i]->chroma.mi_row);
        CHECK_EQ(leaves[i].chroma.mi_col, plant[i]->chroma.mi_col);
      }
    }
    // Round-trip through the complete frame wrapper too.
    std::vector<std::uint8_t> frame;
    std::vector<DecodedLeaf> frame_leaves;
    CHECK(make_minimal_frame(geometry, std::span<const Node>(&root, 1),
                             options, frame, &frame_leaves, error));
    CHECK(!frame.empty());
    CHECK_EQ(frame_leaves.size(), leaves.size());
  }
}

void test_cdf_mode_round_trip() {
  // Adaptive CDFs must round-trip as well as the frozen default.
  Node root;
  std::string error;
  CHECK(build_case({"vert_all4x4", 8, 8, "VS", false}, root, error));
  const FrameGeometry geometry = make_geometry(8, 8);
  for (bool disable : {true, false}) {
    SyntaxOptions options;
    options.disable_cdf_update = disable;
    std::vector<std::uint8_t> tile;
    CHECK(write_tile(geometry, std::span<const Node>(&root, 1), options, tile,
                     error));
    std::vector<DecodedLeaf> leaves;
    CHECK(read_tile(geometry, tile, options, leaves, error));
    CHECK_EQ(leaves.size(), std::size_t{4});
  }
}

void test_reader_rejects_garbage() {
  const FrameGeometry geometry = make_geometry(8, 8);
  std::vector<std::uint8_t> garbage(8, 0);
  std::vector<DecodedLeaf> leaves;
  std::string error;
  const bool ok = read_tile(geometry, garbage, SyntaxOptions{}, leaves, error);
  CHECK(!ok);
  CHECK(!error.empty());
}


// --------------------------------------------------------------------------
// Multi-superblock frame: two full 64x64 SBs decomposed to 4x4 leaves
// --------------------------------------------------------------------------
PartitionType choose_deep(const Node& node, const PartitionPlan&) {
  if (node.size == BlockSize::k64x64) return PartitionType::kHorz;
  if (node.size == BlockSize::k64x32) return PartitionType::kVert;
  if (node.size == BlockSize::k32x32) return PartitionType::kHorz;
  if (node.size == BlockSize::k32x16) return PartitionType::kVert;
  if (node.size == BlockSize::k16x16) return PartitionType::kHorz;
  if (node.size == BlockSize::k16x8) return PartitionType::kVert;
  if (node.size == BlockSize::k8x8) return PartitionType::kHorz;
  if (node.size == BlockSize::k8x4) return PartitionType::kVert;
  return PartitionType::kNone;
}

void test_multi_superblock() {
  const FrameGeometry geometry = make_geometry(128, 64);
  const int sb_mi = 16;
  std::vector<Node> superblocks;
  std::string error;
  const int mi_rows = av2enc::partition::mi_rows(geometry);
  const int mi_cols = av2enc::partition::mi_cols(geometry);
  for (int row = 0; row * sb_mi < mi_rows; ++row) {
    for (int col = 0; col * sb_mi < mi_cols; ++col) {
      Node sb;
      CHECK(build_superblock(geometry, row * sb_mi, col * sb_mi, choose_deep,
                             sb, error));
      superblocks.push_back(std::move(sb));
    }
  }
  CHECK_EQ(superblocks.size(), std::size_t{2});

  SyntaxOptions options;
  std::vector<std::uint8_t> tile;
  CHECK(write_tile(geometry, superblocks, options, tile, error));
  const std::vector<std::uint8_t> expect = hex_to_bytes(
      "d6dc64c6902dd8fc67fa7e5c3a97121c686e358ff7763201584a65c981dad6a3"
      "509b5a70d63bb2da30b54adf8983559b48f52e74aa50082ab9726f38d3de1c22"
      "b9a67a318a6b548cf665adf135f1cfc70555dc1571dd6e16525469255b56cc35"
      "1a2e62945d9b48f688ee46854f753ad00d28c4415620057f677aa039a50cf7a6"
      "231518cf63c80cd16ef8d58d97d271a7bb9bd0229452c5ea7a46a45035ddbcb0"
      "8d8c84ca7bb713a54fc0229452c5ea7a46a45035ddbcb08d8c84ca7bb713a55d"
      "7891070354bb33c26be39f8d77b74319920734450c1b74a3f471900ac2547a74"
      "a9ad467488840afecf10f5de774247d6d30f8d6f15ba60be2a79764081346f5c"
      "8809b3f1a9140e1158ef086022af632015849df008a514b17a9e91a9140dfc66"
      "3bc21808abd8c80561277c0229452c5ea7a46a450384563bc21808abd8c80561"
      "277c0229452c5ea7a46a450351e8b7f723921329eedc4e946bc4d7c73f1c1557"
      "7055c78828b7f723921329eedc4e946bc4d7c73f1c15577055c7600000");
  CHECK_EQ(tile.size(), expect.size());
  if (tile.size() == expect.size()) {
    CHECK(std::memcmp(tile.data(), expect.data(), tile.size()) == 0);
  }

  std::vector<DecodedLeaf> leaves;
  CHECK(read_tile(geometry, tile, options, leaves, error));
  CHECK_EQ(leaves.size(), std::size_t{512});
  int sub8_count = 0;
  for (const DecodedLeaf& leaf : leaves) {
    if (is_sub8(leaf.size)) ++sub8_count;
  }
  CHECK_EQ(sub8_count, 512);

  std::vector<std::uint8_t> frame;
  std::vector<DecodedLeaf> frame_leaves;
  CHECK(make_minimal_frame(geometry, superblocks, options, frame,
                           &frame_leaves, error));
  CHECK(!frame.empty());
  CHECK_EQ(frame_leaves.size(), std::size_t{512});
}

}  // namespace

int main() {
  test_predicates();
  test_partition_sweep();
  test_sub8_partition_cases();
  test_chroma_ownership();
  test_mode_info();
  test_coefficient_contexts();
  test_tx_plan();
  test_syntax_goldens_and_round_trip();
  test_multi_superblock();
  test_cdf_mode_round_trip();
  test_reader_rejects_garbage();
  if (g_failures != 0) {
    std::fprintf(stderr, "sub8_test: %d/%d checks failed\n", g_failures,
                 g_checks);
    return 1;
  }
  std::printf("sub8_test: %d checks passed\n", g_checks);
  return 0;
}
