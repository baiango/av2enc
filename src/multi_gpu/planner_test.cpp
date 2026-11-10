// Standalone tests for the multi-GPU tile assignment / VRAM / transfer planner.
// No CUDA, no devices, no bitstream. Every timing number here is a model.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "planner.h"

namespace {

int g_failures = 0;

void check(bool condition, const char* expression, const char* file, int line) {
  if (!condition) {
    ++g_failures;
    std::cerr << file << ':' << line << ": CHECK failed: " << expression << '\n';
  }
}

#define CHECK(expression) \
  ::check((expression), #expression, __FILE__, __LINE__)

bool near(double a, double b, double tolerance = 1e-6) {
  return std::fabs(a - b) <= tolerance;
}

av2enc::tiles::TileLayout make_layout(int width, int height,
                                     av2enc::tiles::TileLayoutPreset preset) {
  const auto dims = av2enc::tiles::preset_dims(preset);
  const auto result = av2enc::tiles::plan_tile_layout(width, height, dims.cols,
                                                      dims.rows, 64, 31, 0);
  if (!result.ok()) {
    std::cerr << "layout failed: " << result.error << '\n';
    std::exit(EXIT_FAILURE);
  }
  return result.layout;
}

av2enc::multigpu::PlannerConfig default_config(int gpu_count) {
  av2enc::multigpu::PlannerConfig config;
  config.gpu_count = gpu_count;
  return config;
}

std::vector<std::vector<int>> expect_balanced(int width, int height,
                                              av2enc::tiles::TileLayoutPreset preset,
                                              int gpu_count) {
  const auto layout = make_layout(width, height, preset);
  av2enc::multigpu::GpuTileAssignment assignment;
  std::string error;
  CHECK(av2enc::multigpu::assign_tiles(
      layout, gpu_count, av2enc::multigpu::AssignmentPolicy::kBalancedLpt,
      assignment, error));
  CHECK(error.empty());
  return assignment.tile_indices;
}

void test_documented_assignments() {
  using av2enc::tiles::TileLayoutPreset;

  CHECK(av2enc::multigpu::default_layout_for_gpus(1) ==
        TileLayoutPreset::k4x2);
  CHECK(av2enc::multigpu::default_layout_for_gpus(2) ==
        TileLayoutPreset::k8x2);
  CHECK(av2enc::multigpu::default_layout_for_gpus(4) ==
        TileLayoutPreset::k16x2);

  CHECK(expect_balanced(1920, 1080, TileLayoutPreset::k4x2, 1) ==
        std::vector<std::vector<int>>({{0, 1, 2, 3, 4, 5, 6, 7}}));

  CHECK(expect_balanced(1920, 1080, TileLayoutPreset::k8x2, 2) ==
        std::vector<std::vector<int>>(
            {{0, 2, 4, 6, 8, 10, 12, 14}, {1, 3, 5, 7, 9, 11, 13, 15}}));

  CHECK(expect_balanced(1920, 1080, TileLayoutPreset::k16x2, 4) ==
        std::vector<std::vector<int>>(
            {{4, 8, 12, 14, 16, 20, 24, 28},
             {5, 9, 13, 15, 17, 21, 25, 29},
             {0, 2, 6, 10, 18, 22, 26, 30},
             {1, 3, 7, 11, 19, 23, 27, 31}}));

  CHECK(expect_balanced(3840, 2160, TileLayoutPreset::k8x2, 2) ==
        std::vector<std::vector<int>>(
            {{0, 2, 4, 6, 8, 10, 12, 14}, {1, 3, 5, 7, 9, 11, 13, 15}}));

  CHECK(expect_balanced(3840, 2160, TileLayoutPreset::k16x2, 4) ==
        std::vector<std::vector<int>>(
            {{0, 4, 8, 12, 16, 20, 24, 28},
             {1, 5, 9, 13, 17, 21, 25, 29},
             {2, 6, 10, 14, 18, 22, 26, 30},
             {3, 7, 11, 15, 19, 23, 27, 31}}));
}

void test_assignment_invariants() {
  using av2enc::tiles::TileLayoutPreset;
  using av2enc::multigpu::AssignmentPolicy;
  using av2enc::multigpu::assign_tiles;

  for (int gpus : {1, 2, 4}) {
    for (auto preset : {TileLayoutPreset::k4x2, TileLayoutPreset::k8x2,
                        TileLayoutPreset::k16x2}) {
      const auto layout = make_layout(1920, 1080, preset);
      if (layout.tile_count() < gpus) {
        continue;
      }
      av2enc::multigpu::GpuTileAssignment assignment;
      std::string error;
      CHECK(assign_tiles(layout, gpus, AssignmentPolicy::kBalancedLpt,
                         assignment, error));
      std::vector<int> flat;
      for (const auto& group : assignment.tile_indices) {
        CHECK(!group.empty());
        for (int index : group) {
          flat.push_back(index);
        }
      }
      std::sort(flat.begin(), flat.end());
      CHECK(flat.size() == static_cast<std::size_t>(layout.tile_count()));
      for (int i = 0; i < layout.tile_count(); ++i) {
        CHECK(flat[static_cast<std::size_t>(i)] == i);
      }
      CHECK(assignment.tile_counts.size() == static_cast<std::size_t>(gpus));
      CHECK(near(assignment.area_imbalance, 1.0, 0.02));
      CHECK(assignment.cost_imbalance >= 1.0);
      CHECK(assignment.load_min_ratio <= 1.0);
      CHECK(assignment.load_cv >= 0.0);
    }
  }

  // A layout with fewer tiles than GPUs is an explicit failure, not a drop.
  const auto layout = make_layout(1920, 1080, TileLayoutPreset::k4x2);
  av2enc::multigpu::GpuTileAssignment assignment;
  std::string error;
  CHECK(!assign_tiles(layout, 16, AssignmentPolicy::kBalancedLpt, assignment,
                      error));
  CHECK(!error.empty());
}

void test_contiguous_assignment() {
  using av2enc::tiles::TileLayoutPreset;
  using av2enc::multigpu::AssignmentPolicy;
  using av2enc::multigpu::assign_tiles;

  const auto layout1080 = make_layout(1920, 1080, TileLayoutPreset::k8x2);
  av2enc::multigpu::GpuTileAssignment assignment;
  std::string error;
  CHECK(assign_tiles(layout1080, 2, AssignmentPolicy::kContiguousRanges,
                     assignment, error));
  CHECK(assignment.tile_indices ==
        std::vector<std::vector<int>>(
            {{0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14, 15}}));
  // The 1080p row split is inherently uneven: 512 px vs 568 px.
  CHECK(assignment.area_imbalance > 1.04);
  CHECK(assignment.area_imbalance < 1.06);

  const auto layout4k = make_layout(3840, 2160, TileLayoutPreset::k8x2);
  CHECK(assign_tiles(layout4k, 2, AssignmentPolicy::kContiguousRanges,
                     assignment, error));
  CHECK(assignment.area_imbalance < 1.01);

  const auto layout1080_4 = make_layout(1920, 1080, TileLayoutPreset::k16x2);
  CHECK(assign_tiles(layout1080_4, 4, AssignmentPolicy::kContiguousRanges,
                     assignment, error));
  CHECK(assignment.tile_indices ==
        std::vector<std::vector<int>>(
            {{0, 1, 2, 3, 4, 5, 6, 7},
             {8, 9, 10, 11, 12, 13, 14, 15, 16},
             {17, 18, 19, 20, 21, 22, 23},
             {24, 25, 26, 27, 28, 29, 30, 31}}));
}

void test_vram_estimates() {
  using av2enc::tiles::TileLayoutPreset;
  using av2enc::multigpu::plan_multi_gpu;

  struct Case {
    int width;
    int height;
    TileLayoutPreset preset;
    int gpus;
    double expected_live_mb;
  };
  const Case cases[] = {
      {1920, 1080, TileLayoutPreset::k4x2, 1, 614.88},
      {1920, 1080, TileLayoutPreset::k8x2, 2, 588.25},
      {1920, 1080, TileLayoutPreset::k16x2, 4, 574.94},
      {3840, 2160, TileLayoutPreset::k4x2, 1, 851.75},
      {3840, 2160, TileLayoutPreset::k8x2, 2, 762.50},
      {3840, 2160, TileLayoutPreset::k16x2, 4, 711.50},
  };
  for (const Case& item : cases) {
    const auto layout = make_layout(item.width, item.height, item.preset);
    av2enc::multigpu::MultiGpuPlan plan;
    std::string error;
    CHECK(plan_multi_gpu(layout, default_config(item.gpus), plan, error));
    CHECK(plan.valid);
    CHECK(plan.vram_fits);
    CHECK(plan.gpus.size() == static_cast<std::size_t>(item.gpus));
    CHECK(near(plan.vram_live_max_mb, item.expected_live_mb, 0.5));
    CHECK(plan.vram_live_max_mb < plan.config.vram_cap_mb);
    for (const auto& gpu : plan.gpus) {
      CHECK(gpu.vram.fits);
      CHECK(near(gpu.vram.live_mb,
                 gpu.vram.persistent_mb + gpu.vram.scratch_mb +
                     gpu.vram.candidate_mb + gpu.vram.headroom_mb,
                 1e-6));
      CHECK(near(gpu.vram.headroom_mb, 512.0));
      CHECK(gpu.tile_count > 0);
    }
    // Reference duplication: refs=4, 1080p frame = 2.97 MiB, 4K = 11.87 MiB.
    const double frame_mb = plan.memory.frame_mb;
    CHECK(near(plan.memory.ref_pool_mb_per_gpu, 4.0 * frame_mb, 1e-6));
    CHECK(near(plan.memory.ref_extra_mb,
               static_cast<double>(item.gpus - 1) * 4.0 * frame_mb, 1e-6));
  }

  // Splitting the frame reduces scratch but not the persistent reference pool.
  const auto layout1 = make_layout(1920, 1080, TileLayoutPreset::k4x2);
  const auto layout4 = make_layout(1920, 1080, TileLayoutPreset::k16x2);
  av2enc::multigpu::MultiGpuPlan plan1;
  av2enc::multigpu::MultiGpuPlan plan4;
  std::string error;
  CHECK(plan_multi_gpu(layout1, default_config(1), plan1, error));
  CHECK(plan_multi_gpu(layout4, default_config(4), plan4, error));
  CHECK(near(plan1.gpus.front().vram.persistent_mb,
             plan4.gpus.front().vram.persistent_mb, 1e-6));
  CHECK(plan4.gpus.front().vram.scratch_mb <
        plan1.gpus.front().vram.scratch_mb);
}

void test_reference_duplication() {
  using av2enc::tiles::TileLayoutPreset;
  using av2enc::multigpu::frame_memory;

  const auto layout = make_layout(1920, 1080, TileLayoutPreset::k8x2);
  CHECK(layout.width == 1920 && layout.height == 1080);
  const auto memory = frame_memory(layout, 2, 4, 8);
  CHECK(memory.frame_bytes == 1920ull * 1080ull * 3ull / 2ull);
  CHECK(near(memory.frame_mb, 2.96630859375, 1e-9));
  CHECK(near(memory.ref_pool_mb_per_gpu, 11.865234375, 1e-9));
  CHECK(near(memory.ref_extra_mb, 11.865234375, 1e-9));
  CHECK(near(memory.ref_total_mb, 23.73046875, 1e-9));

  const auto memory4 = frame_memory(layout, 4, 4, 8);
  CHECK(near(memory4.ref_extra_mb, 35.595703125, 1e-9));

  // 10-bit doubles the reference pool.
  const auto memory10 = frame_memory(layout, 1, 4, 10);
  CHECK(near(memory10.frame_mb, 2.0 * memory.frame_mb, 1e-9));
}

void test_host_transfer() {
  using av2enc::tiles::TileLayoutPreset;
  using av2enc::multigpu::plan_host_transfer;
  using av2enc::multigpu::PlannerConfig;

  const auto layout1080 = make_layout(1920, 1080, TileLayoutPreset::k8x2);
  PlannerConfig config = default_config(2);
  config.adaptive_cdf = false;
  auto transfer = plan_host_transfer(layout1080, config);
  CHECK(near(transfer.effective_gbps, 26.7784, 0.001));
  CHECK(near(transfer.input_upload_mb, 2.0 * 2.96630859375, 1e-6));
  CHECK(near(transfer.recon_gather_mb, 2.96630859375, 1e-6));
  CHECK(near(transfer.ref_fill_mb, 2.96630859375, 1e-6));
  CHECK(near(transfer.tile_payload_mb, 0.11920928955078125, 1e-6));
  CHECK(near(transfer.cdf_gather_mb, 0.0));
  CHECK(transfer.phases == 4);
  CHECK(near(transfer.total_mb, 11.9835, 0.01));
  CHECK(transfer.total_ms < 1.0);
  CHECK(transfer.pct_of_frame_budget < 0.1);

  config.adaptive_cdf = true;
  transfer = plan_host_transfer(layout1080, config);
  CHECK(transfer.phases == 5);
  CHECK(near(transfer.cdf_gather_mb, 8.0 * 46.7 * 1024.0 / (1024.0 * 1024.0),
             1e-6));
  CHECK(near(transfer.total_mb, 12.3481, 0.01));

  const auto layout4k = make_layout(3840, 2160, TileLayoutPreset::k16x2);
  config = default_config(4);
  config.adaptive_cdf = true;
  transfer = plan_host_transfer(layout4k, config);
  CHECK(near(transfer.total_mb, 95.4087, 0.02));
  CHECK(transfer.pct_of_frame_budget < 0.5);

  config = default_config(1);
  transfer = plan_host_transfer(layout1080, config);
  CHECK(transfer.phases == 2);  // upload + payload only
  CHECK(near(transfer.recon_gather_mb, 0.0));
  CHECK(near(transfer.ref_fill_mb, 0.0));
}

void test_scaling_model_is_labelled() {
  using av2enc::tiles::TileLayoutPreset;
  using av2enc::multigpu::plan_multi_gpu;

  const auto layout = make_layout(1920, 1080, TileLayoutPreset::k8x2);
  av2enc::multigpu::PlannerConfig config = default_config(2);
  config.content_skew = 0.15;
  av2enc::multigpu::MultiGpuPlan plan;
  std::string error;
  CHECK(plan_multi_gpu(layout, config, plan, error));
  CHECK(plan.valid);
  CHECK(plan.scaling.measured == false);
  CHECK(plan.measured == false);
  CHECK(!plan.disclaimer.empty());
  CHECK(plan.scaling.speedup > 1.0);
  CHECK(plan.scaling.speedup < 2.0);
  CHECK(plan.scaling.compute_ms_1gpu > 900.0);

  // One GPU is the identity in the model.
  av2enc::multigpu::MultiGpuPlan single;
  CHECK(plan_multi_gpu(layout, default_config(1), single, error));
  CHECK(near(single.scaling.speedup, 1.0, 1e-9));
}

void test_determinism() {
  using av2enc::tiles::TileLayoutPreset;
  using av2enc::multigpu::plan_multi_gpu;

  const auto layout = make_layout(3840, 2160, TileLayoutPreset::k16x2);
  av2enc::multigpu::PlannerConfig config = default_config(4);
  config.content_skew = 0.15;
  av2enc::multigpu::MultiGpuPlan a;
  av2enc::multigpu::MultiGpuPlan b;
  std::string error;
  CHECK(plan_multi_gpu(layout, config, a, error));
  CHECK(plan_multi_gpu(layout, config, b, error));
  CHECK(a.assignment.tile_indices == b.assignment.tile_indices);
  CHECK(near(a.assignment.area_imbalance, b.assignment.area_imbalance));
  CHECK(near(a.assignment.cost_imbalance, b.assignment.cost_imbalance));
  CHECK(near(a.vram_live_max_mb, b.vram_live_max_mb));
  CHECK(near(a.transfer.total_mb, b.transfer.total_mb));
  CHECK(near(a.scaling.speedup, b.scaling.speedup));
  for (std::size_t i = 0; i < a.gpus.size(); ++i) {
    CHECK(a.gpus[i].tile_indices == b.gpus[i].tile_indices);
    CHECK(near(a.gpus[i].vram.live_mb, b.gpus[i].vram.live_mb));
  }
}

void test_rejections() {
  using av2enc::tiles::TileLayoutPreset;
  using av2enc::multigpu::plan_multi_gpu;

  const auto layout = make_layout(1920, 1080, TileLayoutPreset::k8x2);
  av2enc::multigpu::MultiGpuPlan plan;
  std::string error;

  auto config = default_config(0);
  CHECK(!plan_multi_gpu(layout, config, plan, error));

  config = default_config(2);
  config.reference_frames = 0;
  CHECK(!plan_multi_gpu(layout, config, plan, error));

  config = default_config(2);
  config.bit_depth = 12;
  CHECK(!plan_multi_gpu(layout, config, plan, error));

  config = default_config(2);
  config.pcie_gen = 2;
  CHECK(!plan_multi_gpu(layout, config, plan, error));

  config = default_config(2);
  config.pcie_efficiency = 1.5;
  CHECK(!plan_multi_gpu(layout, config, plan, error));

  config = default_config(2);
  config.serial_fraction = 1.0;
  CHECK(!plan_multi_gpu(layout, config, plan, error));

  // 4:2:0 model requires even dimensions.
  const auto odd = av2enc::tiles::plan_tile_layout(641, 361, 1, 1, 64, 31, 0);
  CHECK(odd.ok());
  config = default_config(1);
  CHECK(!plan_multi_gpu(odd.layout, config, plan, error));

  // A cap below the estimate is reported, not silently accepted.
  const auto large = make_layout(3840, 2160, TileLayoutPreset::k4x2);
  config = default_config(1);
  config.vram_cap_mb = 600.0;
  CHECK(plan_multi_gpu(large, config, plan, error));
  CHECK(plan.valid);
  CHECK(!plan.vram_fits);
}

}  // namespace

int main() {
  test_documented_assignments();
  test_assignment_invariants();
  test_contiguous_assignment();
  test_vram_estimates();
  test_reference_duplication();
  test_host_transfer();
  test_scaling_model_is_labelled();
  test_determinism();
  test_rejections();

  if (g_failures != 0) {
    std::cerr << "multi_gpu_planner_test: " << g_failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "multi_gpu_planner_test: PASS\n";
  return EXIT_SUCCESS;
}
