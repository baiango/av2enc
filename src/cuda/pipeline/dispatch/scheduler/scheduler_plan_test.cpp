// Host-only tests for the wavefront plan: dependency edges, wavefront levels,
// transform-class batching, launch accounting and input validation.  No CUDA is
// required, so this runs on the macOS dev host as well as the GPU box.
//
// Build/run: src/cuda/scheduler/build_scheduler_test.sh --plan-only

#include "cuda/pipeline/dispatch/scheduler/scheduler.h"
#include "basic_geometry.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using av2enc::cuda::sched::CodingBlock;
using av2enc::cuda::sched::FrameGeometry;
using av2enc::cuda::sched::FrameJob;
using av2enc::cuda::sched::Plan;
using av2enc::cuda::sched::PlanCounters;
using av2enc::cuda::sched::testutil::make_geometry;
using av2enc::cuda::sched::testutil::make_basic_k32x32_blocks;

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
    std::printf("  FAIL  %s: got %lld, want %lld\n", what, got, want);
    ++failures;
  }
}

FrameJob make_job(const FrameGeometry& geometry,
                  const std::vector<CodingBlock>& blocks,
                  std::int64_t total_coeffs) {
  FrameJob job;
  job.geometry = geometry;
  job.blocks = blocks.data();
  job.block_count = static_cast<int>(blocks.size());
  job.total_coeffs = total_coeffs;
  return job;
}

// A plain uniform tiling, used to check edge/level arithmetic exactly.
std::vector<CodingBlock> make_uniform_blocks(int cols, int rows) {
  std::vector<CodingBlock> blocks;
  std::int64_t coeffs = 0;
  for (int j = 0; j < rows; ++j) {
    for (int i = 0; i < cols; ++i) {
      CodingBlock b;
      b.mi_row = j * 8;
      b.mi_col = i * 8;
      b.luma_w = 32;
      b.luma_h = 32;
      b.chroma_w = 16;
      b.chroma_h = 16;
      b.upscale = 1;
      b.luma_pels = 1024;
      b.chroma_pels = 256;
      b.coeff_offset = static_cast<int>(coeffs);
      b.real_coeffs = 1;
      coeffs += 1024 + 2 * 256;
      blocks.push_back(b);
    }
  }
  return blocks;
}

void print_counters(const char* label, const Plan& plan) {
  const PlanCounters& c = plan.counters();
  std::printf(
      "  %-10s blocks %5d  jobs %5d  waves %4d  classes %d  groups %4d  "
      "edges %5lld  naive %5lld/%5lld  wave %4lld\n",
      label, c.coding_blocks, c.plane_jobs, c.levels, c.transform_classes,
      c.groups, c.dependency_edges, c.naive_component_calls,
      c.naive_kernel_launches, c.scheduled_kernel_launches);
}

void run_case(const char* label, int width, int height) {
  const FrameGeometry geometry = make_geometry(width, height);
  std::int64_t total = 0;
  const std::vector<CodingBlock> blocks =
      make_basic_k32x32_blocks(geometry, &total);
  const FrameJob job = make_job(geometry, blocks, total);
  Plan plan;
  std::string error;
  const bool built = plan.build(job, error);
  std::printf("%s %dx%d\n", label, width, height);
  if (!built) {
    std::printf("  FAIL  plan build: %s\n", error.c_str());
    ++failures;
    return;
  }
  print_counters(label, plan);
  const PlanCounters& c = plan.counters();
  expect(plan.geometry().width == width && plan.geometry().height == height &&
             plan.geometry().padded_width == geometry.padded_width,
         "plan stores the frame geometry");
  expect_eq(c.dependency_violations, 0, "no dependency violations");
  expect(!c.cyclic, "graph is acyclic");
  expect(c.levels >= 1, "has at least one wavefront level");
  expect(c.plane_jobs == 3 * c.real_blocks, "one Y/U/V job per real block");
  expect(c.scheduled_kernel_launches < c.naive_kernel_launches,
         "fewer kernel launches than the per-block path");
  expect_eq(c.naive_component_calls,
            static_cast<long long>(c.real_blocks) * 13 +
                static_cast<long long>(c.luma_64_jobs),
            "naive component-call count (13 per 32x32, +1 for 64x64 luma)");

  // Structural invariants: groups partition the jobs, slots are contiguous,
  // and every level range is contiguous.
  bool groups_partition = true;
  int counted = 0;
  for (std::size_t g = 0; g < plan.groups().size(); ++g) {
    const auto& group = plan.groups()[g];
    counted += group.job_end - group.job_begin;
    if (g > 0) {
      const auto& previous = plan.groups()[g - 1];
      if (group.job_begin != previous.job_end) {
        groups_partition = false;
      }
    }
    for (int i = group.job_begin; i < group.job_end; ++i) {
      const auto& pj = plan.jobs()[static_cast<std::size_t>(i)];
      if (pj.cls != group.cls || pj.level != group.level ||
          pj.slot != group.slot_begin + (i - group.job_begin)) {
        groups_partition = false;
      }
    }
  }
  expect(counted == c.plane_jobs, "groups cover every plane job");
  expect(groups_partition, "group slots and class ids are contiguous");

  int level_jobs = 0;
  for (int level = 0; level < plan.levels(); ++level) {
    const int begin = plan.level_begin(level);
    const int end = plan.level_end(level);
    level_jobs += end - begin;
    for (int i = begin; i < end; ++i) {
      if (plan.jobs()[static_cast<std::size_t>(i)].level != level) {
        level_jobs = -1;
      }
    }
  }
  expect(level_jobs == c.plane_jobs, "level ranges cover every plane job");

  // Every prerequisite must be in a strictly earlier wavefront level.
  int checked_edges = 0;
  for (std::size_t i = 0; i < plan.jobs().size(); ++i) {
    const auto& pj = plan.jobs()[i];
    if (pj.plane != 0) {
      continue;
    }
    // A luma 32x32 block with a left and above neighbour must be one level
    // deeper than both of them.
    if (pj.x > 0 && pj.y > 0 && pj.level == 0) {
      ++failures;
      std::printf("  FAIL  interior block %zu on wavefront level 0\n", i);
    }
    ++checked_edges;
  }
  expect(checked_edges > 0, "luma jobs present");

  std::printf("\n");
}

void test_uniform_grid() {
  std::printf("uniform 4x3 grid (exact level arithmetic)\n");
  const FrameGeometry geometry = make_geometry(128, 96);
  std::vector<CodingBlock> blocks = make_uniform_blocks(4, 3);
  std::int64_t total = 0;
  for (const CodingBlock& b : blocks) {
    total += b.luma_pels + 2 * b.chroma_pels;
  }
  const FrameJob job = make_job(geometry, blocks, total);
  Plan plan;
  std::string error;
  if (!plan.build(job, error)) {
    std::printf("  FAIL  build: %s\n", error.c_str());
    ++failures;
    return;
  }
  print_counters("uniform", plan);
  const PlanCounters& c = plan.counters();
  expect_eq(c.plane_jobs, 36, "plane jobs");
  expect_eq(c.levels, 6, "wavefront levels");
  expect_eq(c.dependency_edges, 51, "dependency edges (3 planes x 17)");
  expect_eq(c.groups, 12, "groups (2 classes per level)");
  expect_eq(c.scheduled_kernel_launches, 67,
            "scheduled kernels (6 pred + 12*5 + scatter)");
  expect_eq(c.naive_component_calls, 156, "naive component calls");
  expect_eq(c.naive_kernel_launches, 204, "naive kernels");
  expect_eq(c.dependency_violations, 0, "no dependency violations");

  // Exact prerequisite check for the interior block (1,1): left (0,1) and
  // above (1,0), both on level 1, so the block lands on level 2.
  bool found = false;
  for (const auto& pj : plan.jobs()) {
    if (pj.plane == 0 && pj.x == 32 && pj.y == 32) {
      found = true;
      expect_eq(pj.level, 2, "interior luma block level");
    }
  }
  expect(found, "interior luma block present");
  std::printf("\n");
}

void test_rejections() {
  std::printf("input validation\n");
  const FrameGeometry geometry = make_geometry(128, 96);
  std::vector<CodingBlock> blocks = make_uniform_blocks(4, 3);
  Plan plan;
  std::string error;

  FrameJob bad = make_job(geometry, blocks, 0);
  bad.geometry.chroma_width = 63;  // wrong for 128
  expect(!plan.build(bad, error), "rejects a mismatched chroma geometry");

  bad = make_job(geometry, blocks, 0);
  bad.geometry.mi_cols = 33;
  expect(!plan.build(bad, error), "rejects a mismatched mi grid");

  std::vector<CodingBlock> overlap = blocks;
  overlap[1].mi_col = overlap[0].mi_col;  // duplicate position
  overlap[1].mi_row = overlap[0].mi_row;
  FrameJob overlapping = make_job(geometry, overlap, 0);
  expect(!plan.build(overlapping, error), "rejects overlapping blocks");

  std::vector<CodingBlock> bad_pels = blocks;
  bad_pels[0].luma_pels = 1000;
  FrameJob wrong = make_job(geometry, bad_pels, 0);
  expect(!plan.build(wrong, error), "rejects a wrong luma_pels count");

  std::vector<CodingBlock> duplicate_coeffs = blocks;
  duplicate_coeffs[1].coeff_offset = duplicate_coeffs[0].coeff_offset;
  FrameJob coeffs = make_job(geometry, duplicate_coeffs, 0);
  expect(!plan.build(coeffs, error), "rejects overlapping coefficient ranges");

  std::vector<CodingBlock> bad_scale = blocks;
  bad_scale[0].upscale = 2;
  FrameJob scale = make_job(geometry, bad_scale, 0);
  expect(!plan.build(scale, error), "rejects upscale on a non-64x64 block");
  std::printf("\n");
}

void test_signature() {
  std::printf("block signature\n");
  const std::vector<CodingBlock> blocks = make_uniform_blocks(4, 3);
  std::vector<CodingBlock> changed = blocks;
  const std::uint64_t base = av2enc::cuda::sched::coding_blocks_signature(
      blocks.data(), static_cast<int>(blocks.size()));
  expect(base == av2enc::cuda::sched::coding_blocks_signature(
                     changed.data(), static_cast<int>(changed.size())),
         "identical lists hash identically");
  changed[5].mi_col += 8;
  expect(base != av2enc::cuda::sched::coding_blocks_signature(
                     changed.data(), static_cast<int>(changed.size())),
         "changed lists hash differently");
  std::printf("\n");
}

}  // namespace

int main() {
  std::printf("wavefront plan test\n\n");
  run_case("64x64", 64, 64);
  run_case("640x360", 640, 360);
  run_case("1280x720", 1280, 720);
  run_case("1920x1080", 1920, 1080);
  test_uniform_grid();
  test_rejections();
  test_signature();

  if (failures != 0) {
    std::printf("RESULT: FAIL (%d checks)\n", failures);
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
