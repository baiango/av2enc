// CPU-vs-GPU plan parity over device-computed leaf variances.
//
// Builds the same frame plan twice:
//   * CPU reference: block_variance_cost_at() in the planner;
//   * GPU costs: plan_cost.cu's kernel fills one variance per candidate node,
//     and the planner's LeafCostFunction reads the table.
// Requires identical tree shape, leaf order, partition symbols/fields and
// per-node/frame costs.
//
//   ./build_plan_cuda_test.sh
//   OUT=/tmp/plan_cost_test ./build_plan_cuda_test.sh

#include "cuda/pipeline/still/plan/plan.h"
#include "cuda/pipeline/still/plan/plan_cost.h"

#include "codec/partition/tree/partition.h"
#include "cuda/arena.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace still_plan = av2enc::still::plan;
namespace partition = av2enc::partition;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(condition)                                                    \
  do {                                                                      \
    ++g_checks;                                                             \
    if (!(condition)) {                                                     \
      ++g_failures;                                                         \
      std::cerr << __FILE__ << ':' << __LINE__                              \
                << ": CHECK failed: " #condition << '\n';                   \
    }                                                                       \
  } while (false)

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

struct CostTable {
  int width = 0;
  int height = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  std::vector<still_plan::PlanCostRect> rects;
  std::vector<std::int64_t> costs;

  std::size_t index(int mi_row, int mi_col, std::uint8_t size) const {
    return (static_cast<std::size_t>(mi_row) *
                static_cast<std::size_t>(mi_cols) +
            static_cast<std::size_t>(mi_col)) *
               static_cast<std::size_t>(partition::kBlockSizeCount) +
           static_cast<std::size_t>(size);
  }
};

bool build_table(int width, int height, CostTable& table, std::string& error) {
  partition::FrameGeometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = 64;
  table.width = width;
  table.height = height;
  table.mi_cols = partition::mi_cols(geometry);
  table.mi_rows = partition::mi_rows(geometry);
  const std::size_t cells = static_cast<std::size_t>(table.mi_cols) *
                            static_cast<std::size_t>(table.mi_rows) *
                            static_cast<std::size_t>(
                                partition::kBlockSizeCount);
  table.rects.assign(cells, still_plan::PlanCostRect{});
  table.costs.assign(cells, std::numeric_limits<std::int64_t>::min());
  for (int mi_row = 0; mi_row < table.mi_rows; ++mi_row) {
    for (int mi_col = 0; mi_col < table.mi_cols; ++mi_col) {
      for (int s = 0; s < partition::kBlockSizeCount; ++s) {
        const auto size = static_cast<partition::BlockSize>(s);
        if (!partition::is_valid_block_size(size)) continue;
        const partition::BlockRect rect =
            partition::block_rect(geometry, mi_row, mi_col, size);
        const std::size_t idx =
            table.index(mi_row, mi_col, static_cast<std::uint8_t>(size));
        table.rects[idx].x = rect.x;
        table.rects[idx].y = rect.y;
        table.rects[idx].w = static_cast<std::uint16_t>(rect.visible_width);
        table.rects[idx].h = static_cast<std::uint16_t>(rect.visible_height);
      }
    }
  }
  (void)error;
  return true;
}

// Fills table.costs from a device kernel run.  All device memory comes from a
// single Arena, matching the runtime contract.
bool fill_table_gpu(const CostTable& source, const std::vector<std::uint8_t>& luma,
                    CostTable& table, std::string& error) {
  table = source;
  av2enc::cuda::Arena arena;
  if (!av2enc::cuda::is_ok(arena.init_mb(64))) {
    error = "arena init failed";
    return false;
  }
  auto* d_luma = arena.allocate_array<std::uint8_t>(luma.size());
  auto* d_rects = arena.allocate_array<still_plan::PlanCostRect>(
      table.rects.size());
  auto* d_costs =
      arena.allocate_array<std::int64_t>(table.costs.size());
  if (d_luma == nullptr || d_rects == nullptr || d_costs == nullptr) {
    error = "arena allocation failed";
    return false;
  }
  if (cudaMemcpy(d_luma, luma.data(), luma.size(), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    error = "luma upload failed";
    return false;
  }
  if (cudaMemcpy(d_rects, table.rects.data(),
                 table.rects.size() * sizeof(still_plan::PlanCostRect),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    error = "rect upload failed";
    return false;
  }
  if (!still_plan::plan_cost_launch_variance(
          d_luma, table.width, d_rects, d_costs,
          static_cast<int>(table.rects.size()), nullptr, &error)) {
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    error = "variance kernel sync failed";
    return false;
  }
  if (cudaMemcpy(table.costs.data(), d_costs,
                 table.costs.size() * sizeof(std::int64_t),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    error = "cost download failed";
    return false;
  }
  return true;
}

bool same_plan(const still_plan::FramePlan& cpu,
               const still_plan::FramePlan& gpu, std::string& error) {
  if (cpu.leaves != gpu.leaves || cpu.nodes != gpu.nodes ||
      cpu.blocks.size() != gpu.blocks.size() ||
      cpu.tree.size() != gpu.tree.size()) {
    error = "plan sizes differ";
    return false;
  }
  if (cpu.cost != gpu.cost) {
    std::ostringstream os;
    os.precision(17);
    os << "frame cost differs: cpu=" << cpu.cost << " gpu=" << gpu.cost;
    error = os.str();
    return false;
  }
  for (std::size_t i = 0; i < cpu.tree.size(); ++i) {
    const still_plan::TreeNode& a = cpu.tree[i];
    const still_plan::TreeNode& b = gpu.tree[i];
    if (a.mi_row != b.mi_row || a.mi_col != b.mi_col ||
        a.block_size != b.block_size || a.partition != b.partition ||
        a.ext_partition != b.ext_partition ||
        a.allowed_mask != b.allowed_mask ||
        a.allowed_mask_ext != b.allowed_mask_ext || a.forced != b.forced ||
        a.tree_type != b.tree_type || a.sdp_split != b.sdp_split ||
        a.sdp_chroma_block != b.sdp_chroma_block ||
        a.child_count != b.child_count || a.leaf_index != b.leaf_index ||
        a.cost != b.cost) {
      error = "tree node differs at index " + std::to_string(i);
      return false;
    }
  }
  for (std::size_t i = 0; i < cpu.blocks.size(); ++i) {
    const still_plan::BlockPlan& a = cpu.blocks[i];
    const still_plan::BlockPlan& b = gpu.blocks[i];
    if (a.mi_row != b.mi_row || a.mi_col != b.mi_col ||
        a.block_size != b.block_size || a.tree_type != b.tree_type ||
        a.luma_w != b.luma_w || a.luma_h != b.luma_h ||
        a.chroma_w != b.chroma_w || a.chroma_h != b.chroma_h ||
        a.chroma_mi_row != b.chroma_mi_row ||
        a.chroma_mi_col != b.chroma_mi_col ||
        a.chroma_mi_size != b.chroma_mi_size ||
        a.has_chroma != b.has_chroma ||
        a.luma.coeff_offset != b.luma.coeff_offset ||
        a.u.coeff_offset != b.u.coeff_offset ||
        a.v.coeff_offset != b.v.coeff_offset) {
      error = "block differs at index " + std::to_string(i);
      return false;
    }
  }
  if (still_plan::plan_to_tsv(cpu) != still_plan::plan_to_tsv(gpu) ||
      still_plan::plan_tree_to_tsv(cpu) != still_plan::plan_tree_to_tsv(gpu)) {
    error = "plan dumps differ";
    return false;
  }
  return true;
}

struct Case {
  const char* name;
  still_plan::PlanOptions options;
};

bool run_case(const Case& item, const std::vector<std::uint8_t>& luma,
              const CostTable& table, int width, int height, int& leaves,
              int& nodes) {
  still_plan::FramePlan cpu;
  still_plan::FramePlan gpu;
  std::string error;
  if (!still_plan::build_frame_plan(width, height, luma, item.options, cpu,
                                    error)) {
    std::cerr << item.name << ": CPU plan failed: " << error << '\n';
    ++g_failures;
    return false;
  }
  const auto provider = [&](const still_plan::NodeGeometry& node) {
    const std::size_t idx =
        table.index(node.mi_row, node.mi_col, node.block_size);
    return table.costs[idx];
  };
  if (!still_plan::build_frame_plan_with_leaf_cost(
          width, height, luma, item.options, provider, gpu, error)) {
    std::cerr << item.name << ": GPU-cost plan failed: " << error << '\n';
    ++g_failures;
    return false;
  }
  if (!same_plan(cpu, gpu, error)) {
    std::cerr << item.name << ": plan parity failed: " << error << '\n';
    ++g_failures;
    return false;
  }
  leaves += cpu.leaves;
  nodes += cpu.nodes;
  return true;
}

}  // namespace

int main() {
  constexpr int kWidth = 640;
  constexpr int kHeight = 360;
  const auto luma = mosaic_luma(kWidth, kHeight, 4);

  CostTable table;
  std::string error;
  if (!build_table(kWidth, kHeight, table, error)) {
    std::cerr << "cost table failed: " << error << '\n';
    return 1;
  }
  CostTable gpu_table;
  if (!fill_table_gpu(table, luma, gpu_table, error)) {
    std::cerr << "GPU cost fill failed: " << error << '\n';
    return 1;
  }
  bool saw_marker = false;
  for (std::int64_t cost : gpu_table.costs) {
    if (cost == std::numeric_limits<std::int64_t>::min()) saw_marker = true;
  }
  CHECK(!saw_marker);

  std::vector<Case> cases;
  {
    Case base{"base64", still_plan::PlanOptions{}};
    base.options.min_block_size = 8;
    base.options.max_partition_depth = 4;
    cases.push_back(base);

    Case sb128{"sb128", base.options};
    sb128.options.sb_size = 128;
    cases.push_back(sb128);

    Case sb256{"sb256", base.options};
    sb256.options.sb_size = 256;
    cases.push_back(sb256);

    Case ext{"ext", base.options};
    ext.options.enable_ext_partitions = true;
    ext.options.enable_uneven_4way_partitions = true;
    cases.push_back(ext);

    Case sub8{"sub8", base.options};
    sub8.options.sub8 = true;
    cases.push_back(sub8);

    Case sdp{"sdp", base.options};
    sdp.options.sdp = true;
    cases.push_back(sdp);
  }

  int leaves = 0;
  int nodes = 0;
  int passed = 0;
  for (const Case& item : cases) {
    if (run_case(item, luma, gpu_table, kWidth, kHeight, leaves, nodes)) {
      ++passed;
    }
  }

  // Negative control: a null device pointer must be rejected, not launched.
  CHECK(!still_plan::plan_cost_launch_variance(nullptr, kWidth, nullptr,
                                               nullptr, 1, nullptr, &error));
  CHECK(!error.empty());

  std::cout << "plan_cost: " << passed << "/" << cases.size()
            << " cases CPU==GPU, " << leaves << " leaves, " << nodes
            << " nodes, " << g_checks << " checks\n";
  if (g_failures != 0) {
    std::cerr << g_failures << " of " << g_checks << " checks failed\n";
    return 1;
  }
  return 0;
}
