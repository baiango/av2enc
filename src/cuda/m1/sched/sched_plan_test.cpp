// Host-only tests for the M1 wavefront plan: coverage, exact directional-edge
// availability, dependency levels, syntax-order equivalence, TX class spans and
// launch accounting.  No CUDA required; runs on the macOS dev host.
//
// Build/run: src/cuda/m1/sched/build_sched_test.sh --plan-only

#include "cuda/m1/sched/sched.h"
#include "cuda/m1/sched/sched_bridge.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace {

using av2enc::cuda::m1::sched::BlockPlan;
using av2enc::cuda::m1::sched::FramePlan;
using av2enc::cuda::m1::sched::Plan;
using av2enc::cuda::m1::sched::PlaneJob;

int failures = 0;

void expect(bool condition, const char* what) {
  if (condition) {
    std::printf("  ok    %s\n", what);
  } else {
    std::printf("  FAIL  %s\n", what);
    ++failures;
  }
}

void expect_eq(long long got, long long want, const char* what) {
  if (got == want) {
    std::printf("  ok    %s (%lld)\n", what, got);
  } else {
    std::printf("  FAIL  %s: got %lld want %lld\n", what, got, want);
    ++failures;
  }
}

FramePlan make_frame(int width, int height) {
  FramePlan f;
  f.width = width;
  f.height = height;
  f.chroma_width = (width + 1) / 2;
  f.chroma_height = (height + 1) / 2;
  f.mi_cols = 2 * ((width + 7) >> 3);
  f.mi_rows = 2 * ((height + 7) >> 3);
  const int coded_w = f.mi_cols * 4;
  const int coded_h = f.mi_rows * 4;
  f.padded_luma_width = (coded_w + 63) & ~63;
  f.padded_luma_height = (coded_h + 63) & ~63;
  f.padded_chroma_width = f.padded_luma_width / 2;
  f.padded_chroma_height = f.padded_luma_height / 2;
  return f;
}

BlockPlan make_leaf(int mi_row, int mi_col, int w, int h, int mode_seed) {
  BlockPlan b;
  b.mi_row = mi_row;
  b.mi_col = mi_col;
  b.luma_w = w;
  b.luma_h = h;
  b.has_chroma = (w >= 8 && h >= 8);
  if (b.has_chroma) {
    b.chroma_w = w / 2;
    b.chroma_h = h / 2;
  }
  b.luma_mode = mode_seed % 13;
  b.luma_angle = (mode_seed % 7) - 3;
  b.chroma_mode = (mode_seed % 3 == 0) ? 9 : 0;
  b.chroma_angle = 0;
  b.is_cfl = (b.has_chroma && (mode_seed % 2) == 0) ? 1 : 0;
  return b;
}

// Full raster tiling with uniform square leaves.
std::vector<BlockPlan> tile_uniform(const FramePlan& f, int size) {
  std::vector<BlockPlan> out;
  const int step = size / 4;
  int seed = 0;
  // CPU syntax order: superblock raster first, then raster inside each SB.
  for (int sbr = 0; sbr < f.mi_rows; sbr += 16) {
    for (int sbc = 0; sbc < f.mi_cols; sbc += 16) {
      const int r_end = std::min(sbr + 16, f.mi_rows);
      const int c_end = std::min(sbc + 16, f.mi_cols);
      for (int r = sbr; r < r_end; r += step) {
        for (int c = sbc; c < c_end; c += step) {
          out.push_back(make_leaf(r, c, size, size, seed++));
        }
      }
    }
  }
  return out;
}

// Mixed square tree: deterministic splits, forced split at partial SBs.
std::vector<BlockPlan> tile_mixed(const FramePlan& f) {
  std::vector<BlockPlan> out;
  int seed = 0;
  const auto split_p = [](int mi_row, int mi_col, int px) {
    if (px == 64) {
      return ((mi_row / 16) + (mi_col / 16)) % 2 == 0;
    }
    if (px == 32) {
      return ((mi_row / 16) * 3 + (mi_col / 16)) % 3 == 0;
    }
    if (px == 16) {
      return ((mi_row / 16) + (mi_col / 16)) % 4 == 0;
    }
    return false;
  };
  const auto tile = [&](auto&& self, int mi_row, int mi_col, int px) -> void {
    const int mi = px / 4;
    const bool fits = (mi_row + mi <= f.mi_rows) && (mi_col + mi <= f.mi_cols);
    if (!fits) {
      if (px <= 8) {
        return;
      }
      const int h = px / 2;
      self(self, mi_row, mi_col, h);
      self(self, mi_row, mi_col + h / 4, h);
      self(self, mi_row + h / 4, mi_col, h);
      self(self, mi_row + h / 4, mi_col + h / 4, h);
      return;
    }
    if (px > 8 && split_p(mi_row, mi_col, px)) {
      const int h = px / 2;
      self(self, mi_row, mi_col, h);
      self(self, mi_row, mi_col + h / 4, h);
      self(self, mi_row + h / 4, mi_col, h);
      self(self, mi_row + h / 4, mi_col + h / 4, h);
      return;
    }
    out.push_back(make_leaf(mi_row, mi_col, px, px, seed++));
  };
  for (int r = 0; r < f.mi_rows; r += 16) {
    for (int c = 0; c < f.mi_cols; c += 16) {
      tile(tile, r, c, 64);
    }
  }
  return out;
}

// Independent transcription of m1_still.cpp's count_top_right /
// count_bottom_left with plain full-frame maps.  Deliberately not shared with
// the plan implementation.
struct RefDecoded {
  int cols = 0;
  int rows = 0;
  std::vector<std::uint8_t> cells;
  void reset(int c, int r) {
    cols = c;
    rows = r;
    cells.assign(static_cast<std::size_t>(c) * static_cast<std::size_t>(r), 0);
  }
  bool at(int x, int y) const {
    if (x < 0 || y < 0 || x >= cols || y >= rows) {
      return false;
    }
    return cells[static_cast<std::size_t>(y) * static_cast<std::size_t>(cols) +
                  static_cast<std::size_t>(x)] != 0;
  }
  void fill(int x, int y, int w, int h) {
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        if (x + i >= 0 && y + j >= 0 && x + i < cols && y + j < rows) {
          cells[static_cast<std::size_t>(y + j) *
                    static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(x + i)] = 1;
        }
      }
    }
  }
};

struct RefN4 {
  int above_right = 0;
  int below_left = 0;
};

RefN4 ref_n4(const RefDecoded& decoded, int plane_cols, int cell_x,
             int cell_y, int w4, int h4, int sb_c0, int sb_r0, int sb_w,
             int sb_h) {
  RefN4 out;
  const int sub_border = 2 * sb_w;
  const int border_w = std::min(sub_border, plane_cols - sb_c0);
  const int rel_x = cell_x - sb_c0;
  const int rel_y = cell_y - sb_r0;
  for (int i = 0; i < w4; ++i) {
    const int x = rel_x + w4 + i;
    if (x >= (rel_y == 0 ? border_w : sb_w)) {
      break;
    }
    if (rel_y - 1 >= 0 && !decoded.at(sb_c0 + x, sb_r0 + rel_y - 1)) {
      break;
    }
    ++out.above_right;
  }
  for (int i = 0; i < h4; ++i) {
    const int y = rel_y + h4 + i;
    if (y >= sb_h) {
      break;
    }
    if (rel_x - 1 >= 0 && !decoded.at(sb_c0 + rel_x - 1, sb_r0 + y)) {
      break;
    }
    ++out.below_left;
  }
  return out;
}

// Recomputes n4 for every job of the plan from the block list and compares.
int check_n4(const FramePlan& frame, const Plan& plan) {
  int mismatches = 0;
  const int luma_cols = frame.mi_cols;
  const int luma_rows = frame.mi_rows;
  const int chroma_cols = frame.mi_cols / 2;
  const int chroma_rows = frame.mi_rows / 2;
  RefDecoded luma;
  RefDecoded chroma;
  luma.reset(luma_cols, luma_rows);
  chroma.reset(chroma_cols, chroma_rows);
  int cur_sb_r = -1;
  int cur_sb_c = -1;
  // Jobs are stored level-major; walk blocks in syntax order instead and find
  // the job by (block, plane).
  std::vector<int> job_of(static_cast<std::size_t>(frame.block_count) * 3, -1);
  for (int i = 0; i < static_cast<int>(plan.jobs().size()); ++i) {
    const PlaneJob& j = plan.job(i);
    job_of[static_cast<std::size_t>(3 * j.block + j.plane)] = i;
  }
  for (int bi = 0; bi < frame.block_count; ++bi) {
    const BlockPlan& b = frame.blocks[bi];
    const int sb_r = (b.mi_row / 16) * 16;
    const int sb_c = (b.mi_col / 16) * 16;
    if (sb_r != cur_sb_r || sb_c != cur_sb_c) {
      luma.cells.assign(luma.cells.size(), 0);
      chroma.cells.assign(chroma.cells.size(), 0);
      cur_sb_r = sb_r;
      cur_sb_c = sb_c;
    }
    const int ly = job_of[static_cast<std::size_t>(3 * bi + 0)];
    const RefN4 y = ref_n4(luma, luma_cols, b.mi_col, b.mi_row, b.luma_w / 4,
                           b.luma_h / 4, sb_c, sb_r,
                           std::min(16, luma_cols - sb_c),
                           std::min(16, luma_rows - sb_r));
    if (plan.job(ly).n4_above_right != y.above_right ||
        plan.job(ly).n4_below_left != y.below_left) {
      ++mismatches;
    }
    if (b.has_chroma) {
      const int cc = job_of[static_cast<std::size_t>(3 * bi + 1)];
      const int csc = sb_c / 2;
      const int csr = sb_r / 2;
      const RefN4 c = ref_n4(chroma, chroma_cols, b.mi_col / 2, b.mi_row / 2,
                             b.chroma_w / 4, b.chroma_h / 4, csc, csr,
                             std::min(8, chroma_cols - csc),
                             std::min(8, chroma_rows - csr));
      if (plan.job(cc).n4_above_right != c.above_right ||
          plan.job(cc).n4_below_left != c.below_left) {
        ++mismatches;
      }
      const int cv = job_of[static_cast<std::size_t>(3 * bi + 2)];
      if (plan.job(cv).n4_above_right != c.above_right ||
          plan.job(cv).n4_below_left != c.below_left) {
        ++mismatches;
      }
    }
    luma.fill(b.mi_col, b.mi_row, b.luma_w / 4, b.luma_h / 4);
    if (b.has_chroma) {
      chroma.fill(b.mi_col / 2, b.mi_row / 2, b.chroma_w / 4, b.chroma_h / 4);
    }
  }
  return mismatches;
}

void verify_plan(const char* name, const FramePlan& frame,
                 const std::vector<BlockPlan>& blocks, int expect_levels) {
  std::printf("== %s (%dx%d, %zu leaves) ==\n", name, frame.width, frame.height,
              blocks.size());
  FramePlan f = frame;
  f.blocks = blocks.data();
  f.block_count = static_cast<int>(blocks.size());
  Plan plan;
  std::string error;
  if (!plan.build(f, error)) {
    std::printf("  FAIL  build: %s\n", error.c_str());
    ++failures;
    return;
  }
  const auto& c = plan.counters();
  expect_eq(c.dependency_violations, 0, "dependency violations");
  expect_eq(c.syntax_order_violations, 0, "syntax-order violations");
  if (expect_levels >= 0) {
    expect_eq(c.levels, expect_levels, "wavefront levels");
  } else {
    std::printf("  ok    wavefront levels (%d)\n", c.levels);
  }
  expect_eq(check_n4(f, plan), 0, "n4 availability mismatches");
  expect_eq(plan.total_coeffs(),
            static_cast<long long>(blocks.size()) * 0 +
                [&] {
                  long long sum = 0;
                  for (const BlockPlan& b : blocks) {
                    sum += static_cast<long long>(b.luma_w) * b.luma_h;
                    if (b.has_chroma) {
                      sum += 2ll * b.chroma_w * b.chroma_h;
                    }
                  }
                  return sum;
                }(),
            "canonical coefficient elements");

  // Every edge must point backwards in level and never forwards in CPU order.
  bool levels_ok = true;
  bool order_ok = true;
  int reported = 0;
  for (int i = 0; i < static_cast<int>(plan.jobs().size()); ++i) {
    for (int e = plan.dependency_begin(i); e < plan.dependency_end(i); ++e) {
      const int d = plan.dependency_indices()[static_cast<std::size_t>(e)];
      if (plan.job(d).level >= plan.job(i).level) {
        levels_ok = false;
      }
      const PlaneJob& jd = plan.job(d);
      const PlaneJob& ji = plan.job(i);
      if (!(jd.syntax < ji.syntax ||
            (jd.syntax == ji.syntax && jd.plane < ji.plane))) {
        order_ok = false;
        if (reported < 8) {
          std::printf("        order edge p%d blk%d(%d,%d) <- p%d blk%d(%d,%d)\n",
                      ji.plane, ji.block, ji.x, ji.y, jd.plane, jd.block, jd.x,
                      jd.y);
          ++reported;
        }
      }
    }
  }
  expect(levels_ok, "every edge crosses a level");
  expect(order_ok, "CPU syntax order is topological");
  expect(plan.job(0).level == 0, "first job at level 0");

  // Level ranges and class spans.
  bool ranges_ok = true;
  bool spans_ok = true;
  int cursor = 0;
  for (int lv = 0; lv < plan.levels(); ++lv) {
    const int begin = plan.level_begin(lv);
    const int end = plan.level_end(lv);
    if (begin != cursor || end < begin) {
      ranges_ok = false;
    }
    for (int i = begin; i < end; ++i) {
      if (plan.job(i).level != lv) {
        ranges_ok = false;
      }
    }
    cursor = end;
    for (int si = plan.level_batch(lv).class_begin;
         si < plan.level_batch(lv).class_end; ++si) {
      const auto& span = plan.class_spans()[static_cast<std::size_t>(si)];
      if (span.level != lv || span.job_begin < begin || span.job_end > end ||
          span.job_end <= span.job_begin) {
        spans_ok = false;
      }
      for (int i = span.job_begin; i < span.job_end; ++i) {
        if (plan.job(i).cls != span.cls) {
          spans_ok = false;
        }
      }
    }
  }
  expect(ranges_ok, "level ranges are contiguous and monotonic");
  expect(spans_ok, "class spans cover their levels exactly");
  expect(c.max_jobs_per_level > 0, "non-empty levels");
  expect(c.scheduled_kernel_launches < c.naive_block_calls ||
             blocks.size() <= 4,
         "batched launches fewer than per-block calls");
}

void test_degenerate() {
  std::printf("== validation ==\n");
  FramePlan f = make_frame(64, 64);
  BlockPlan b = make_leaf(0, 0, 64, 64, 0);
  f.blocks = &b;
  f.block_count = 1;
  Plan plan;
  std::string error;
  expect(plan.build(f, error), "single 64x64 leaf builds");
  expect_eq(plan.levels(), 2, "single leaf: Y level 0, chroma level 1");

  FramePlan bad = make_frame(64, 64);
  BlockPlan overlap[2] = {make_leaf(0, 0, 32, 32, 1),
                          make_leaf(0, 0, 32, 32, 2)};
  bad.blocks = overlap;
  bad.block_count = 2;
  Plan p2;
  expect(!p2.build(bad, error), "overlapping leaves rejected");

  FramePlan gap = make_frame(64, 64);
  BlockPlan one = make_leaf(0, 0, 32, 32, 3);
  gap.blocks = &one;
  gap.block_count = 1;
  Plan p3;
  expect(!p3.build(gap, error), "incomplete tiling rejected");

  FramePlan odd = make_frame(64, 64);
  odd.width = 65;
  BlockPlan leaf = make_leaf(0, 0, 64, 64, 4);
  odd.blocks = &leaf;
  odd.block_count = 1;
  Plan p4;
  expect(!p4.build(odd, error), "odd picture width rejected");
}

void test_bridge() {
  std::printf("== shared contract bridge ==\n");
  av2enc::cuda::m1gpu::FramePlan shared;
  shared.width = 64;
  shared.height = 64;
  shared.padded_width = 64;
  shared.padded_height = 64;
  shared.y_stride = 64;
  shared.u_stride = 32;
  shared.v_stride = 32;
  shared.mi_cols = 16;
  shared.mi_rows = 16;
  shared.sb_cols = 1;
  shared.sb_rows = 1;
  shared.block_count = 1;
  shared.leaf_count = 1;
  std::vector<av2enc::cuda::m1gpu::BlockPlan> leaves(1);
  leaves[0].mi_row = 0;
  leaves[0].mi_col = 0;
  leaves[0].luma_w = 64;
  leaves[0].luma_h = 64;
  leaves[0].chroma_w = 32;
  leaves[0].chroma_h = 32;
  leaves[0].tx_w = 32;
  leaves[0].tx_h = 32;
  leaves[0].chroma_tx_w = 32;
  leaves[0].chroma_tx_h = 32;
  leaves[0].has_chroma = 1;
  shared.blocks = std::span<const av2enc::cuda::m1gpu::BlockPlan>(leaves);
  av2enc::cuda::m1gpu::BlockDecision decision;
  decision.luma_mode = 3;
  decision.luma_angle_delta = 1;
  decision.uv_mode = 0;
  decision.is_cfl = 1;
  const Plan bridged =
      av2enc::cuda::m1::sched::build_m1_plan(shared, &decision);
  expect(bridged.counters().coding_blocks == 1 &&
             bridged.counters().plane_jobs == 3,
         "bridge expands one leaf into three planes");
  expect(bridged.counters().levels == 2 &&
             bridged.counters().cfl_jobs == 2,
         "bridge keeps Y before chroma and CfL flags");
  expect(bridged.job(0).luma_mode == 3 && bridged.job(0).angle == 1,
         "bridge copies luma decisions");
}

}  // namespace

int main() {
  test_degenerate();
  test_bridge();

  {
    FramePlan f = make_frame(64, 64);
    const auto blocks = tile_uniform(f, 8);
    // Levels are deeper than the plain anti-diagonal (16 / 125 / 375) because
    // the §7.13.2.1 above-right extension is read while building the edge
    // arrays even for DC, and it can span up to two superblocks along the top
    // row of an SB.  Values cross-checked with an independent simulation.
    verify_plan("64x64 all 8x8", f, blocks, 23);
  }
  {
    FramePlan f = make_frame(640, 360);
    const auto blocks = tile_uniform(f, 8);
    verify_plan("640x360 all 8x8", f, blocks, 197);
  }
  {
    FramePlan f = make_frame(640, 360);
    const auto blocks = tile_mixed(f);
    verify_plan("640x360 mixed tree", f, blocks, -1);
  }
  {
    FramePlan f = make_frame(648, 360);
    const auto blocks = tile_mixed(f);
    verify_plan("648x360 mixed tree (right edge)", f, blocks, -1);
  }
  {
    FramePlan f = make_frame(1920, 1080);
    const auto blocks = tile_uniform(f, 8);
    verify_plan("1920x1080 all 8x8", f, blocks, 599);
  }

  if (failures == 0) {
    std::printf("\nPASS: M1 scheduler plan tests\n");
    return 0;
  }
  std::printf("\nFAIL: %d check(s) failed\n", failures);
  return 1;
}
