// CPU tests for the M1 GPU host-side FramePlan builder.
//
// Run with ./build_and_test.sh.  The target is Linux + CUDA; this binary is a
// correctness oracle for the plan geometry, ordering, and determinism only.

#include "cuda/m1/plan/plan.h"

#include "encoder/partition/partition.h"
#include "io/y4m.h"

#if __has_include("cuda/m1/m1_gpu.h")
#include "cuda/m1/plan/plan_gpu_bridge.h"
#define M1_PLAN_HAVE_GPU_PODS 1
#if __has_include("encoder/m1/stream/m1_stream_bridge.h")
#define M1_PLAN_HAVE_STREAM_TREE 1
#else
#define M1_PLAN_HAVE_STREAM_TREE 0
#endif
#else
#define M1_PLAN_HAVE_GPU_PODS 0
#define M1_PLAN_HAVE_STREAM_TREE 0
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef PLAN_REPO_ROOT
#define PLAN_REPO_ROOT "."
#endif

namespace m1p = av2enc::m1::plan;
namespace partition = av2enc::partition;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(condition)) {                                                      \
      ++g_failures;                                                          \
      std::cerr << __FILE__ << ':' << __LINE__                               \
                << ": CHECK failed: " #condition << '\n';                    \
    }                                                                        \
  } while (false)

#define CHECK_EQ_INT(actual, expected)                                       \
  do {                                                                       \
    ++g_checks;                                                              \
    const long long actual_value = static_cast<long long>(actual);           \
    const long long expected_value = static_cast<long long>(expected);       \
    if (actual_value != expected_value) {                                    \
      ++g_failures;                                                          \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK_EQ failed: "      \
                << #actual << " = " << actual_value << ", expected "         \
                << expected_value << '\n';                                   \
    }                                                                        \
  } while (false)

std::vector<std::uint8_t> constant_luma(int width, int height,
                                        std::uint8_t value) {
  return std::vector<std::uint8_t>(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height), value);
}

// Piecewise-constant tiles.  The between-tile variance keeps the partition
// search splitting until a leaf fits inside one tile, which exercises deep
// trees without changing the production cost model.
std::vector<std::uint8_t> mosaic_luma(int width, int height, int tile) {
  std::vector<std::uint8_t> luma(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::uint32_t tx = static_cast<std::uint32_t>(x / tile);
      const std::uint32_t ty = static_cast<std::uint32_t>(y / tile);
      std::uint32_t h = tx * 0x9e3779b9u ^ ty * 0x85ebca6bu ^ 0xc2b2ae35u;
      h ^= h >> 16;
      h *= 0x7feb352du;
      h ^= h >> 15;
      const int value = 16 + static_cast<int>((h >> 8) % 224u);
      luma[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(value);
    }
  }
  return luma;
}

m1p::PlanOptions base_options() {
  m1p::PlanOptions options;
  options.min_block_size = 8;
  options.max_partition_depth = 4;
  return options;
}

bool build(int width, int height, const std::vector<std::uint8_t>& luma,
           const m1p::PlanOptions& options, m1p::FramePlan& plan) {
  std::string error;
  const bool ok = m1p::build_frame_plan(width, height, luma, options, plan,
                                        error);
  if (!ok) {
    std::cerr << "build_frame_plan failed: " << error << '\n';
  }
  return ok;
}

bool read_fixture(const std::string& name, int& width, int& height,
                  std::vector<std::uint8_t>& luma, std::string& error) {
  const std::filesystem::path path =
      std::filesystem::path(PLAN_REPO_ROOT) / "tests" / "data" /
      (name + ".y4m");
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult result = reader.read_frame(frame, error);
  if (result != av2enc::Y4mReadResult::kFrame) {
    if (error.empty()) error = "fixture has no frames";
    return false;
  }
  width = static_cast<int>(frame.width());
  height = static_cast<int>(frame.height());
  const std::span<const std::uint8_t> y = frame.y();
  luma.assign(y.begin(), y.end());
  return true;
}

// One line per CPU trace item in the first decision pass:
//   N <mi_row> <mi_col> <size> <partition> <forced> <allowed4>
//   B <mi_row> <mi_col> <size>          (immediately after its leaf node)
std::string trace_order_signature(const m1p::FramePlan& plan) {
  std::ostringstream out;
  for (const m1p::TreeNode& node : plan.tree) {
    out << "N " << node.mi_row << ' ' << node.mi_col << ' '
        << m1p::block_size_name(node.block_size) << ' '
        << partition::partition_name(
               static_cast<partition::PartitionType>(node.partition))
        << ' ' << (node.forced ? 1 : 0) << ' ';
    for (int p = 0; p < partition::kPartitionTypeCount; ++p) {
      out << ((node.allowed_mask >> p) & 1U);
    }
    out << '\n';
    if (node.partition ==
            static_cast<std::uint8_t>(partition::PartitionType::kNone) &&
        node.leaf_index != m1p::kNoTreeNode) {
      const m1p::BlockPlan& block = plan.blocks[node.leaf_index];
      out << "B " << block.mi_row << ' ' << block.mi_col << ' '
          << m1p::block_size_name(block.block_size) << '\n';
    }
  }
  return out.str();
}

bool read_expected_signature(const std::string& name, std::string& out,
                             std::string& error) {
  const std::filesystem::path path =
      std::filesystem::path(PLAN_REPO_ROOT) / "src" / "cuda" / "m1" / "plan" /
      "tests" / ("trace_order_" + name + ".txt");
  std::ifstream file(path);
  if (!file) {
    error = "cannot open expected trace file " + path.string();
    return false;
  }
  std::ostringstream filtered;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line[0] == '#') continue;
    if (line.empty()) continue;
    filtered << line << '\n';
  }
  out = filtered.str();
  return true;
}

bool check_fixture_trace(const std::string& name, int expected_leaves,
                         int expected_nodes) {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> luma;
  std::string error;
  if (!read_fixture(name, width, height, luma, error)) {
    std::cerr << "read_fixture(" << name << "): " << error << '\n';
    return false;
  }
  m1p::FramePlan plan;
  if (!build(width, height, luma, base_options(), plan)) return false;
  CHECK_EQ_INT(plan.leaves, expected_leaves);
  CHECK_EQ_INT(plan.nodes, expected_nodes);
  CHECK_EQ_INT(plan.tree.size(), static_cast<std::size_t>(expected_nodes));
  std::string expected;
  if (!read_expected_signature(name, expected, error)) {
    std::cerr << error << '\n';
    return false;
  }
  const std::string actual = trace_order_signature(plan);
  if (actual != expected) {
    std::size_t line = 1;
    std::size_t pos = 0;
    while (pos < actual.size() && pos < expected.size() &&
           actual[pos] == expected[pos]) {
      if (actual[pos] == '\n') ++line;
      ++pos;
    }
    std::cerr << "trace signature mismatch for " << name << " at line "
              << line << '\n';
    std::cerr << "  expected: "
              << expected.substr(pos, expected.find('\n', pos) - pos)
              << '\n';
    std::cerr << "  actual:   "
              << actual.substr(pos, actual.find('\n', pos) - pos) << '\n';
    return false;
  }
  return true;
}

// Every block must be preceded by the blocks that supply its above row and
// left column of reconstructed samples.
bool verify_decode_order(const m1p::FramePlan& plan, std::string& error) {
  std::vector<int> owner(static_cast<std::size_t>(plan.mi_cols) *
                             static_cast<std::size_t>(plan.mi_rows),
                         -1);
  for (const m1p::BlockPlan& block : plan.blocks) {
    const int w4 = block.luma_w / 4;
    const int h4 = block.luma_h / 4;
    const int row_end = std::min(block.mi_row + h4, plan.mi_rows);
    const int col_end = std::min(block.mi_col + w4, plan.mi_cols);
    for (int row = block.mi_row; row < row_end; ++row) {
      for (int col = block.mi_col; col < col_end; ++col) {
        owner[static_cast<std::size_t>(row) *
                  static_cast<std::size_t>(plan.mi_cols) +
              static_cast<std::size_t>(col)] = static_cast<int>(block.index);
      }
    }
  }
  for (const m1p::BlockPlan& block : plan.blocks) {
    const int w4 = block.luma_w / 4;
    const int h4 = block.luma_h / 4;
    const int row_end = std::min(block.mi_row + h4, plan.mi_rows);
    const int col_end = std::min(block.mi_col + w4, plan.mi_cols);
    if (block.mi_row > 0) {
      for (int col = block.mi_col; col < col_end; ++col) {
        const int above = owner[static_cast<std::size_t>(block.mi_row - 1) *
                                   static_cast<std::size_t>(plan.mi_cols) +
                               static_cast<std::size_t>(col)];
        if (above < 0 || above >= static_cast<int>(block.index)) {
          error = "above neighbour is not coded first";
          return false;
        }
      }
    }
    if (block.mi_col > 0) {
      for (int row = block.mi_row; row < row_end; ++row) {
        const int left = owner[static_cast<std::size_t>(row) *
                                   static_cast<std::size_t>(plan.mi_cols) +
                               static_cast<std::size_t>(block.mi_col - 1)];
        if (left < 0 || left >= static_cast<int>(block.index)) {
          error = "left neighbour is not coded first";
          return false;
        }
      }
    }
  }
  return true;
}

void test_constant_64x64() {
  const auto luma = constant_luma(64, 64, 128);
  m1p::FramePlan plan;
  CHECK(build(64, 64, luma, base_options(), plan));
  CHECK_EQ_INT(plan.mi_cols, 16);
  CHECK_EQ_INT(plan.mi_rows, 16);
  CHECK_EQ_INT(plan.sb_cols, 1);
  CHECK_EQ_INT(plan.sb_rows, 1);
  CHECK_EQ_INT(plan.leaves, 1);
  CHECK_EQ_INT(plan.blocks.size(), 1u);
  if (plan.blocks.size() == 1) {
    const m1p::BlockPlan& b = plan.blocks[0];
    CHECK_EQ_INT(b.mi_row, 0);
    CHECK_EQ_INT(b.mi_col, 0);
    CHECK_EQ_INT(b.luma_w, 64);
    CHECK_EQ_INT(b.luma_h, 64);
    CHECK_EQ_INT(b.chroma_w, 32);
    CHECK_EQ_INT(b.chroma_h, 32);
    CHECK_EQ_INT(b.luma.tx_size, 4);   // TX_64X64
    CHECK_EQ_INT(b.luma.adj_w, 32);    // §7.15.4 adjusted transform
    CHECK_EQ_INT(b.luma.adj_h, 32);
    CHECK_EQ_INT(b.u.tx_size, 3);      // TX_32X32
    CHECK_EQ_INT(b.luma.coeff_offset, 0);
    CHECK_EQ_INT(b.u.coeff_offset, 0);
    CHECK_EQ_INT(b.luma.coeff_count, 1024);
    CHECK_EQ_INT(b.u.coeff_count, 1024);
    CHECK(!b.luma_avail.above);
    CHECK(!b.luma_avail.left);
    CHECK(!b.luma_avail.above_right);
    CHECK(!b.luma_avail.below_left);
    CHECK(b.has_chroma);
    CHECK(!b.chroma_offset);
  }
  CHECK_EQ_INT(plan.luma_coeff_count, 1024);
  CHECK_EQ_INT(plan.u_coeff_count, 1024);
  CHECK_EQ_INT(plan.v_coeff_count, 1024);
  CHECK_EQ_INT(plan.luma_stride, 64);
  CHECK_EQ_INT(plan.chroma_stride, 32);
  CHECK_EQ_INT(plan.tree.size(), 1u);
  if (plan.tree.size() == 1) {
    const m1p::TreeNode& root = plan.tree[0];
    CHECK_EQ_INT(root.index, 0);
    CHECK_EQ_INT(root.sb_index, 0);
    CHECK_EQ_INT(root.mi_row, 0);
    CHECK_EQ_INT(root.mi_col, 0);
    CHECK_EQ_INT(root.block_size,
                 static_cast<int>(av2enc::partition::BlockSize::k64x64));
    CHECK_EQ_INT(root.partition,
                 static_cast<int>(av2enc::partition::PartitionType::kNone));
    CHECK(root.sb_root);
    CHECK(!root.forced);
    CHECK_EQ_INT(root.allowed_mask, 13);  // NONE | HORZ | VERT
    CHECK_EQ_INT(root.child_count, 0);
    CHECK_EQ_INT(root.first_child, m1p::kNoTreeNode);
    CHECK_EQ_INT(root.subtree, 0);
    CHECK_EQ_INT(root.leaf_index, 0);
  }
  std::string error;
  CHECK(m1p::validate_frame_plan(plan, error));
}

void test_mosaic_64x64_deep() {
  const auto luma = mosaic_luma(64, 64, 4);
  m1p::FramePlan plan;
  CHECK(build(64, 64, luma, base_options(), plan));
  CHECK(plan.leaves > 1);
  std::string error;
  CHECK(m1p::validate_frame_plan(plan, error));
  for (const m1p::BlockPlan& b : plan.blocks) {
    CHECK(b.luma_w <= 64);
    CHECK(b.luma_h <= 64);
    CHECK(b.depth <= 4);
  }
}

void test_min_block_4() {
  // Depth 4 only reaches 16x16 (AV2 has no SPLIT at 64); depth 8 and a 4-pixel
  // floor are needed to exercise 4x4 leaves.  This mirrors --m1-depth 8.
  m1p::PlanOptions options = base_options();
  options.min_block_size = 4;
  options.max_partition_depth = 8;
  const auto luma = mosaic_luma(64, 64, 4);
  m1p::FramePlan plan;
  CHECK(build(64, 64, luma, options, plan));
  std::string error;
  CHECK(m1p::validate_frame_plan(plan, error));
  bool saw_4x4 = false;
  for (const m1p::BlockPlan& b : plan.blocks) {
    if (b.luma_w == 4 && b.luma_h == 4) {
      saw_4x4 = true;
      CHECK_EQ_INT(b.chroma_w, 4);
      CHECK_EQ_INT(b.chroma_h, 4);
      CHECK_EQ_INT(b.luma.tx_size, 0);  // TX_4X4
      CHECK_EQ_INT(b.luma.adj_w, 4);
      CHECK_EQ_INT(b.luma.adj_h, 4);
    }
  }
  CHECK(saw_4x4);
}

void test_640x360() {
  const auto luma = mosaic_luma(640, 360, 8);
  m1p::FramePlan plan;
  CHECK(build(640, 360, luma, base_options(), plan));
  CHECK_EQ_INT(plan.mi_cols, 160);
  CHECK_EQ_INT(plan.mi_rows, 90);
  CHECK_EQ_INT(plan.sb_cols, 10);
  CHECK_EQ_INT(plan.sb_rows, 6);
  CHECK_EQ_INT(plan.coded_luma_w, 640);
  CHECK_EQ_INT(plan.coded_luma_h, 360);
  CHECK_EQ_INT(plan.luma_stride, 640);
  CHECK_EQ_INT(plan.luma_padded_height, 384);
  CHECK_EQ_INT(plan.chroma_stride, 320);
  CHECK_EQ_INT(plan.chroma_padded_height, 192);
  CHECK(plan.leaves > 10);
  std::string error;
  CHECK(m1p::validate_frame_plan(plan, error));
  CHECK(verify_decode_order(plan, error));
  CHECK_EQ_INT(plan.tree.size(), static_cast<std::size_t>(plan.nodes));
  std::uint32_t next_leaf = 0;
  for (const m1p::TreeNode& node : plan.tree) {
    if (node.partition ==
        static_cast<std::uint8_t>(av2enc::partition::PartitionType::kNone)) {
      CHECK_EQ_INT(node.leaf_index, next_leaf);
      ++next_leaf;
      if (node.leaf_index < plan.blocks.size()) {
        const m1p::BlockPlan& block = plan.blocks[node.leaf_index];
        CHECK_EQ_INT(block.mi_row, node.mi_row);
        CHECK_EQ_INT(block.mi_col, node.mi_col);
        CHECK_EQ_INT(block.block_size, node.block_size);
      }
    } else {
      CHECK_EQ_INT(node.first_child, node.index + 1);
      CHECK(node.child_count >= 1 && node.child_count <= 4);
    }
  }
  CHECK_EQ_INT(next_leaf, static_cast<std::uint32_t>(plan.blocks.size()));

  // The bottom superblock starts at mi_row 80 and has 10 in-frame mi rows.
  bool saw_bottom_sb = false;
  bool saw_crossing_block = false;
  for (const m1p::BlockPlan& b : plan.blocks) {
    CHECK(b.visible_luma_w >= 0 && b.visible_luma_w <= b.luma_w);
    CHECK(b.visible_luma_h >= 0 && b.visible_luma_h <= b.luma_h);
    if (b.sb_mi_row == 80) {
      saw_bottom_sb = true;
      if (b.mi_row + b.luma_h / 4 > 90) {
        saw_crossing_block = true;
        CHECK(b.visible_luma_h < b.luma_h);
      } else {
        CHECK_EQ_INT(b.visible_luma_h, b.luma_h);
      }
    }
    if (b.mi_row == 0) CHECK(!b.luma_avail.above);
    if (b.mi_col == 0) CHECK(!b.luma_avail.left);
    CHECK(b.luma_avail.num4_above_right <= b.luma_w / 4);
    CHECK(b.luma_avail.num4_below_left <= b.luma_h / 4);
    CHECK(b.chroma_avail.num4_above_right <= b.chroma_w / 4);
    CHECK(b.chroma_avail.num4_below_left <= b.chroma_h / 4);
  }
  CHECK(saw_bottom_sb);
  CHECK(saw_crossing_block);
}

void test_1080p() {
  const auto luma = mosaic_luma(1920, 1080, 8);
  m1p::FramePlan plan;
  CHECK(build(1920, 1080, luma, base_options(), plan));
  CHECK_EQ_INT(plan.mi_cols, 480);
  CHECK_EQ_INT(plan.mi_rows, 270);
  CHECK_EQ_INT(plan.sb_cols, 30);
  CHECK_EQ_INT(plan.sb_rows, 17);  // 16 full rows + a 14-mi bottom row
  CHECK(plan.leaves > 100);
  std::string error;
  CHECK(m1p::validate_frame_plan(plan, error));
  CHECK(verify_decode_order(plan, error));
  bool saw_bottom_sb = false;
  for (const m1p::BlockPlan& b : plan.blocks) {
    if (b.sb_mi_row == 256) {
      saw_bottom_sb = true;
      CHECK(b.visible_luma_h <= 1080 - b.y);
    }
  }
  CHECK(saw_bottom_sb);
  CHECK_EQ_INT(plan.coded_luma_h, 1080);
  CHECK_EQ_INT(plan.luma_padded_height, 1088);
}

void test_partial_right_and_bottom() {
  // 300x300: MiCols/MiRows = 76, so the last superblock row and column both
  // cover only 12 of 16 mi cells.  (100x70 fails inside partition::decide_frame
  // with the default depth/min floor; that is a reference limitation shared
  // with the CPU M1 path, not a plan-builder bug.)
  const auto luma = mosaic_luma(300, 300, 4);
  m1p::FramePlan plan;
  CHECK(build(300, 300, luma, base_options(), plan));
  CHECK_EQ_INT(plan.mi_cols, 76);
  CHECK_EQ_INT(plan.mi_rows, 76);
  CHECK_EQ_INT(plan.sb_cols, 5);
  CHECK_EQ_INT(plan.sb_rows, 5);
  CHECK_EQ_INT(plan.coded_luma_w, 304);
  CHECK_EQ_INT(plan.coded_luma_h, 304);
  std::string error;
  CHECK(m1p::validate_frame_plan(plan, error));
  CHECK(verify_decode_order(plan, error));
  bool saw_right = false;
  bool saw_bottom = false;
  for (const m1p::BlockPlan& b : plan.blocks) {
    if (b.sb_mi_col == 64) {
      saw_right = true;
      CHECK(b.visible_luma_w <= 300 - b.x);
    }
    if (b.sb_mi_row == 64) {
      saw_bottom = true;
      CHECK(b.visible_luma_h <= 300 - b.y);
    }
  }
  CHECK(saw_right);
  CHECK(saw_bottom);
}

void test_force_single_block() {
  m1p::PlanOptions options = base_options();
  options.force_single_block = true;
  const auto luma = mosaic_luma(640, 360, 8);
  m1p::FramePlan plan;
  CHECK(build(640, 360, luma, options, plan));
  CHECK_EQ_INT(plan.leaves, 60);
  CHECK_EQ_INT(plan.blocks.size(), 60u);
  int expected = 0;
  for (const m1p::BlockPlan& b : plan.blocks) {
    CHECK_EQ_INT(b.luma_w, 64);
    CHECK_EQ_INT(b.luma_h, 64);
    CHECK_EQ_INT(b.mi_col, (expected % 10) * 16);
    CHECK_EQ_INT(b.mi_row, (expected / 10) * 16);
    ++expected;
  }
  const m1p::BlockPlan& first = plan.blocks[0];
  CHECK(!first.luma_avail.above);
  CHECK(!first.luma_avail.left);
  const m1p::BlockPlan& right = plan.blocks[1];
  CHECK(!right.luma_avail.above);
  CHECK(right.luma_avail.left);
  // §6.19.2.3 clear_block_decoded_flags: the top border is one for up to two
  // superblocks, so the first 64-wide block of a superblock row sees 16
  // above-right 4x4 cells even when the block is not at the frame's left edge.
  CHECK_EQ_INT(right.luma_avail.num4_above_right, 16);
  CHECK(right.luma_avail.above_right);
  const m1p::BlockPlan& below = plan.blocks[10];
  CHECK(below.luma_avail.above);
  CHECK(!below.luma_avail.left);
  // Blocks in the bottom SB extend past the 92-mi coded plane.
  const m1p::BlockPlan& bottom = plan.blocks[50];
  CHECK_EQ_INT(bottom.mi_row, 80);
  CHECK(bottom.luma_h == 64);
  CHECK_EQ_INT(bottom.visible_luma_h, 40);
  CHECK_EQ_INT(bottom.visible_luma_w, 64);
}

void test_debug_gates() {
  const auto luma = mosaic_luma(640, 360, 8);
  m1p::PlanOptions square = base_options();
  square.square_only = true;
  m1p::FramePlan square_plan;
  CHECK(build(640, 360, luma, square, square_plan));
  for (const m1p::BlockPlan& b : square_plan.blocks) {
    CHECK_EQ_INT(b.luma_w, b.luma_h);
  }
  std::string error;
  CHECK(m1p::validate_frame_plan(square_plan, error));

  m1p::PlanOptions leaf32 = base_options();
  leaf32.leaf32 = true;
  m1p::FramePlan leaf32_plan;
  CHECK(build(640, 360, luma, leaf32, leaf32_plan));
  for (const m1p::BlockPlan& b : leaf32_plan.blocks) {
    CHECK(b.luma_w <= 32);
    CHECK(b.luma_h <= 32);
  }
  CHECK(m1p::validate_frame_plan(leaf32_plan, error));
}

void test_determinism_and_dump() {
  const auto luma = mosaic_luma(640, 360, 8);
  const m1p::PlanOptions options = base_options();
  m1p::FramePlan a;
  m1p::FramePlan b;
  CHECK(build(640, 360, luma, options, a));
  CHECK(build(640, 360, luma, options, b));
  const std::string tsv_a = m1p::plan_to_tsv(a);
  const std::string tsv_b = m1p::plan_to_tsv(b);
  CHECK(tsv_a == tsv_b);
  const std::string tree_tsv_a = m1p::plan_tree_to_tsv(a);
  CHECK(tree_tsv_a == m1p::plan_tree_to_tsv(b));
  CHECK(m1p::plan_to_json(a) == m1p::plan_to_json(b));

  std::size_t tsv_rows = 0;
  std::size_t pos = 0;
  while (pos < tsv_a.size()) {
    const std::size_t end = tsv_a.find('\n', pos);
    const std::string line =
        tsv_a.substr(pos, end == std::string::npos ? std::string::npos
                                                   : end - pos);
    if (!line.empty() && line[0] != '#') ++tsv_rows;  // column header + rows
    if (end == std::string::npos) break;
    pos = end + 1;
  }
  CHECK_EQ_INT(tsv_rows, static_cast<std::size_t>(a.blocks.size()) + 1);

  std::size_t tree_rows = 0;
  pos = 0;
  while (pos < tree_tsv_a.size()) {
    const std::size_t end = tree_tsv_a.find('\n', pos);
    const std::string line =
        tree_tsv_a.substr(pos, end == std::string::npos ? std::string::npos
                                                        : end - pos);
    if (!line.empty() && line[0] != '#') ++tree_rows;
    if (end == std::string::npos) break;
    pos = end + 1;
  }
  CHECK_EQ_INT(tree_rows, a.tree.size() + 1);

  const std::string json = m1p::plan_to_json(a);
  CHECK(json.find("\"blocks\":[") != std::string::npos);
  CHECK(json.find("],\"tree\":[") != std::string::npos);
  CHECK(!json.empty() && json.back() == '\n');
}

void test_fixture_trace_order() {
  CHECK(check_fixture_trace("m0_64x64_1f", 11, 21));
  CHECK(check_fixture_trace("m0_640x360_1f", 218, 376));
}

void test_errors() {
  std::vector<std::uint8_t> tiny(10, 0);
  m1p::FramePlan plan;
  std::string error;
  CHECK(!m1p::build_frame_plan(16, 16, tiny, base_options(), plan, error));
  CHECK(!error.empty());

  const auto luma = constant_luma(64, 64, 0);
  m1p::PlanOptions options = base_options();
  options.sb_size = 128;
  CHECK(!m1p::build_frame_plan(64, 64, luma, options, plan, error));
  CHECK(error.find("64x64 superblock") != std::string::npos);

  options = base_options();
  options.max_partition_depth = -1;
  CHECK(!m1p::build_frame_plan(64, 64, luma, options, plan, error));
  CHECK(!error.empty());

  options = base_options();
  options.min_block_size = 0;
  CHECK(!m1p::build_frame_plan(64, 64, luma, options, plan, error));
  CHECK(!error.empty());
}

void test_env_gates() {
  m1p::PlanOptions options;
  unsetenv("AV2ENC_M1_SQUARE_ONLY");
  unsetenv("AV2ENC_M1_LEAF32");
  m1p::apply_debug_env(options);
  CHECK(!options.square_only);
  CHECK(!options.leaf32);
  setenv("AV2ENC_M1_SQUARE_ONLY", "1", 1);
  setenv("AV2ENC_M1_LEAF32", "1", 1);
  m1p::apply_debug_env(options);
  CHECK(options.square_only);
  CHECK(options.leaf32);
  unsetenv("AV2ENC_M1_SQUARE_ONLY");
  unsetenv("AV2ENC_M1_LEAF32");
}

#if M1_PLAN_HAVE_GPU_PODS

void test_gpu_bridge() {
  const auto luma = mosaic_luma(640, 360, 8);
  std::vector<av2enc::cuda::m1gpu::BlockPlan> blocks;
  av2enc::cuda::m1gpu::FramePlan frame;
  std::string error;
  CHECK(m1p::build_gpu_frame_plan(640, 360, luma, base_options(), blocks, frame,
                                  error));
  CHECK(!blocks.empty());
  CHECK_EQ_INT(frame.block_count, blocks.size());
  CHECK_EQ_INT(frame.leaf_count, blocks.size());
  CHECK_EQ_INT(frame.width, 640);
  CHECK_EQ_INT(frame.height, 360);
  CHECK_EQ_INT(frame.padded_width, 640);
  CHECK_EQ_INT(frame.padded_height, 384);
  CHECK_EQ_INT(frame.y_stride, 640);
  CHECK_EQ_INT(frame.u_stride, 320);
  CHECK_EQ_INT(frame.v_stride, 320);
  CHECK_EQ_INT(frame.mi_cols, 160);
  CHECK_EQ_INT(frame.mi_rows, 90);
  CHECK_EQ_INT(frame.sb_cols, 10);
  CHECK_EQ_INT(frame.sb_rows, 6);
  CHECK_EQ_INT(frame.chroma_shift_x, 1);
  CHECK_EQ_INT(frame.chroma_shift_y, 1);
  CHECK_EQ_INT(frame.is_placeholder, 0);
  CHECK_EQ_INT(frame.blocks.size(), blocks.size());
  CHECK(frame.blocks.data() == blocks.data());

  m1p::FramePlan reference;
  CHECK(build(640, 360, luma, base_options(), reference));
  CHECK_EQ_INT(frame.coeff_count_y, reference.luma_coeff_count);
  CHECK_EQ_INT(frame.coeff_count_u, reference.u_coeff_count);
  CHECK_EQ_INT(frame.coeff_count_v, reference.v_coeff_count);
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const m1p::BlockPlan& src = reference.blocks[i];
    const av2enc::cuda::m1gpu::BlockPlan& dst = blocks[i];
    CHECK_EQ_INT(dst.mi_row, src.mi_row);
    CHECK_EQ_INT(dst.mi_col, src.mi_col);
    CHECK_EQ_INT(dst.luma_w, src.luma_w);
    CHECK_EQ_INT(dst.luma_h, src.luma_h);
    CHECK_EQ_INT(dst.chroma_w, src.chroma_w);
    CHECK_EQ_INT(dst.chroma_h, src.chroma_h);
    CHECK_EQ_INT(dst.tx_w, src.luma.adj_w);
    CHECK_EQ_INT(dst.tx_h, src.luma.adj_h);
    CHECK_EQ_INT(dst.chroma_tx_w, src.u.adj_w);
    CHECK_EQ_INT(dst.chroma_tx_h, src.u.adj_h);
    CHECK_EQ_INT(dst.tx_upscale_w, src.luma_w / src.luma.adj_w);
    CHECK_EQ_INT(dst.tx_upscale_h, src.luma_h / src.luma.adj_h);
    CHECK_EQ_INT(dst.has_chroma, src.has_chroma ? 1 : 0);
    CHECK_EQ_INT(dst.tree_depth, src.depth);
    CHECK_EQ_INT(dst.plane_offset_y,
                 src.y * reference.luma_stride + src.x);
    CHECK_EQ_INT(dst.plane_offset_u,
                 src.chroma_y * reference.chroma_stride + src.chroma_x);
    CHECK_EQ_INT(dst.coeff_offset_y, src.luma.coeff_offset);
    CHECK_EQ_INT(dst.coeff_offset_u, src.u.coeff_offset);
    CHECK_EQ_INT(dst.coeff_offset_v, src.v.coeff_offset);
    const int expected_avail =
        (src.luma_avail.left ? av2enc::cuda::m1gpu::kAvailLeft : 0) |
        (src.luma_avail.above ? av2enc::cuda::m1gpu::kAvailAbove : 0) |
        (src.luma_avail.above_left ? av2enc::cuda::m1gpu::kAvailAboveLeft : 0) |
        (src.luma_avail.above_right ? av2enc::cuda::m1gpu::kAvailAboveRight
                                    : 0) |
        (src.luma_avail.below_left ? av2enc::cuda::m1gpu::kAvailBelowLeft : 0);
    CHECK_EQ_INT(dst.avail, expected_avail);
  }

#if M1_PLAN_HAVE_STREAM_TREE
  std::vector<av2enc::m1::stream::TreeNode> stream_tree;
  std::vector<av2enc::cuda::m1gpu::BlockPlan> tree_blocks;
  av2enc::cuda::m1gpu::FramePlan tree_frame;
  CHECK(m1p::build_gpu_frame_plan(640, 360, luma, base_options(), tree_blocks,
                                  tree_frame, stream_tree, error));
  CHECK_EQ_INT(stream_tree.size(), reference.tree.size());
  for (std::size_t i = 0; i < stream_tree.size(); ++i) {
    const m1p::TreeNode& src = reference.tree[i];
    const av2enc::m1::stream::TreeNode& dst = stream_tree[i];
    CHECK_EQ_INT(dst.mi_row, src.mi_row);
    CHECK_EQ_INT(dst.mi_col, src.mi_col);
    CHECK_EQ_INT(dst.bsize, src.block_size);
    CHECK_EQ_INT(dst.partition, src.partition);
    CHECK_EQ_INT(dst.allowed_mask, src.allowed_mask);
    CHECK_EQ_INT(dst.child_count, src.child_count);
    const int expected_flags =
        src.forced ? av2enc::m1::syntax::kFlagForced : 0;
    CHECK_EQ_INT(dst.flags & av2enc::m1::syntax::kFlagForced, expected_flags);
    if (src.partition ==
        static_cast<std::uint8_t>(av2enc::partition::PartitionType::kNone)) {
      CHECK_EQ_INT(dst.leaf_index, src.leaf_index);
    }
  }
#endif

  // Below the default 8-pixel floor a leaf can have has_chroma = 0.  The M1
  // reference still codes chroma for every leaf, so offsets must stay
  // contiguous in the bridge too.
  m1p::PlanOptions deep = base_options();
  deep.min_block_size = 4;
  deep.max_partition_depth = 8;
  const auto deep_luma = mosaic_luma(64, 64, 4);
  std::vector<av2enc::cuda::m1gpu::BlockPlan> deep_blocks;
  av2enc::cuda::m1gpu::FramePlan deep_frame;
  CHECK(m1p::build_gpu_frame_plan(64, 64, deep_luma, deep, deep_blocks,
                                  deep_frame, error));
  bool saw_no_chroma = false;
  for (std::size_t i = 0; i < deep_blocks.size(); ++i) {
    if (deep_blocks[i].has_chroma == 0) saw_no_chroma = true;
    if (i == 0) continue;
    CHECK(deep_blocks[i].coeff_offset_y >= deep_blocks[i - 1].coeff_offset_y);
    CHECK(deep_blocks[i].coeff_offset_u >= deep_blocks[i - 1].coeff_offset_u);
    CHECK(deep_blocks[i].coeff_offset_v >= deep_blocks[i - 1].coeff_offset_v);
  }
  CHECK(saw_no_chroma);
}

#endif  // M1_PLAN_HAVE_GPU_PODS

}  // namespace

int main() {
  test_constant_64x64();
  test_mosaic_64x64_deep();
  test_min_block_4();
  test_640x360();
  test_1080p();
  test_partial_right_and_bottom();
  test_force_single_block();
  test_debug_gates();
  test_determinism_and_dump();
  test_fixture_trace_order();
  test_errors();
  test_env_gates();
#if M1_PLAN_HAVE_GPU_PODS
  test_gpu_bridge();
#endif

  if (g_failures != 0) {
    std::cerr << g_failures << " of " << g_checks << " checks failed\n";
    return 1;
  }
  std::cout << "all " << g_checks << " checks passed\n";
  return 0;
}
