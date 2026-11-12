#include "planner.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::multigpu {
namespace {

constexpr double kMib = 1024.0 * 1024.0;
constexpr double kCdfKib = 46.7;  // full per-tile CDF working set
constexpr int kMaxGpuCount = 8;
constexpr int kMaxContiguousTiles = 512;

// Raw PCIe payload per lane per direction, GB/s, before the efficiency factor.
double pcie_per_lane_gbps(int gen) {
  switch (gen) {
    case 3:
      return 0.985;
    case 4:
      return 1.969;
    case 5:
      return 3.938;
    default:
      return 0.0;
  }
}

std::uint64_t frame_bytes_for(const TileLayout& layout, int bit_depth) {
  const std::uint64_t samples = static_cast<std::uint64_t>(layout.width) *
                                static_cast<std::uint64_t>(layout.height);
  const std::uint64_t bytes_per_sample = bit_depth > 8 ? 2u : 1u;
  // 4:2:0: luma + two half-resolution chroma planes = 1.5 samples per pixel.
  return samples * 3u / 2u * bytes_per_sample;
}

// Stable 64-bit splitmix hash. Used only for the synthetic content-skew prior;
// an integer hash keeps the planner deterministic across libm versions.
std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31);
}

double unit_from_hash(std::uint64_t value) {
  return static_cast<double>(splitmix64(value) >> 11) *
         (1.0 / 9007199254740992.0);  // 2^53
}

std::vector<double> content_factors(int count, double skew) {
  std::vector<double> factors(static_cast<std::size_t>(count), 1.0);
  if (skew <= 0.0 || count <= 0) {
    return factors;
  }
  double sum = 0.0;
  for (int i = 0; i < count; ++i) {
    const double u = unit_from_hash(static_cast<std::uint64_t>(i + 1));
    factors[static_cast<std::size_t>(i)] = 1.0 + skew * (2.0 * u - 1.0);
    sum += factors[static_cast<std::size_t>(i)];
  }
  const double mean = sum / static_cast<double>(count);
  if (mean > 0.0) {
    for (double& factor : factors) {
      factor /= mean;
    }
  }
  return factors;
}

bool validate_frame_geometry(const TileLayout& layout, std::string& error) {
  if (!tiles::validate_tile_layout(layout, error)) {
    return false;
  }
  if (layout.width % 2 != 0 || layout.height % 2 != 0) {
    error = "multi-GPU planner models 4:2:0 and needs even frame dimensions";
    return false;
  }
  return true;
}

struct LoadMetrics {
  double area_imbalance = 1.0;
  double cost_imbalance = 1.0;
  double load_min_ratio = 1.0;
  double load_cv = 0.0;
};

LoadMetrics compute_metrics(const TileLayout& layout,
                            const std::vector<std::vector<int>>& groups,
                            const std::vector<double>& factors) {
  LoadMetrics metrics;
  if (groups.empty()) {
    return metrics;
  }
  std::vector<double> area(static_cast<std::size_t>(groups.size()), 0.0);
  std::vector<double> cost(static_cast<std::size_t>(groups.size()), 0.0);
  for (std::size_t g = 0; g < groups.size(); ++g) {
    for (int index : groups[g]) {
      const tiles::TileRect& tile =
          layout.tiles[static_cast<std::size_t>(index)];
      const double pixels = static_cast<double>(tile.pixels());
      area[g] += pixels;
      cost[g] += pixels * factors[static_cast<std::size_t>(index)];
    }
  }
  const double count = static_cast<double>(groups.size());
  const double area_sum = std::accumulate(area.begin(), area.end(), 0.0);
  const double cost_sum = std::accumulate(cost.begin(), cost.end(), 0.0);
  const double area_mean = area_sum / count;
  const double cost_mean = cost_sum / count;
  if (area_mean > 0.0) {
    metrics.area_imbalance =
        *std::max_element(area.begin(), area.end()) / area_mean;
  }
  if (cost_mean > 0.0) {
    metrics.cost_imbalance =
        *std::max_element(cost.begin(), cost.end()) / cost_mean;
    metrics.load_min_ratio =
        *std::min_element(cost.begin(), cost.end()) / cost_mean;
    double variance = 0.0;
    for (double value : cost) {
      const double delta = value - cost_mean;
      variance += delta * delta;
    }
    metrics.load_cv = std::sqrt(variance / count) / cost_mean;
  }
  return metrics;
}

void fill_group_totals(const TileLayout& layout,
                       const std::vector<std::vector<int>>& groups,
                       const std::vector<double>& factors,
                       GpuTileAssignment& assignment) {
  const std::size_t count = groups.size();
  assignment.tile_indices = groups;
  assignment.tile_counts.assign(count, 0);
  assignment.area_pixels.assign(count, 0);
  assignment.area_sb_cells.assign(count, 0);
  assignment.cost_units.assign(count, 0.0);
  for (std::size_t g = 0; g < count; ++g) {
    assignment.tile_counts[g] = static_cast<int>(groups[g].size());
    for (int index : groups[g]) {
      const tiles::TileRect& tile =
          layout.tiles[static_cast<std::size_t>(index)];
      assignment.area_pixels[g] += tile.pixels();
      assignment.area_sb_cells[g] += tile.sb_area();
      assignment.cost_units[g] +=
          static_cast<double>(tile.pixels()) *
          factors[static_cast<std::size_t>(index)];
    }
  }
}

bool assign_balanced_lpt(const TileLayout& layout, int gpu_count,
                         const std::vector<double>& factors,
                         std::vector<std::vector<int>>& groups,
                         std::string& error) {
  groups.assign(static_cast<std::size_t>(gpu_count), {});
  std::vector<std::uint64_t> loads(static_cast<std::size_t>(gpu_count), 0);
  std::vector<int> order;
  order.reserve(layout.tiles.size());
  for (int i = 0; i < layout.tile_count(); ++i) {
    order.push_back(i);
  }
  std::sort(order.begin(), order.end(), [&layout](int a, int b) {
    const std::uint64_t area_a = layout.tiles[static_cast<std::size_t>(a)].pixels();
    const std::uint64_t area_b = layout.tiles[static_cast<std::size_t>(b)].pixels();
    if (area_a != area_b) {
      return area_a > area_b;
    }
    return a < b;
  });

  for (int index : order) {
    int best_gpu = 0;
    for (int g = 1; g < gpu_count; ++g) {
      if (loads[static_cast<std::size_t>(g)] <
          loads[static_cast<std::size_t>(best_gpu)]) {
        best_gpu = g;
      }
    }
    groups[static_cast<std::size_t>(best_gpu)].push_back(index);
    loads[static_cast<std::size_t>(best_gpu)] +=
        layout.tiles[static_cast<std::size_t>(index)].pixels();
  }
  for (std::vector<int>& group : groups) {
    std::sort(group.begin(), group.end());
  }
  (void)factors;
  (void)error;
  return true;
}

bool assign_contiguous_dp(const TileLayout& layout, int gpu_count,
                          std::vector<std::vector<int>>& groups,
                          std::string& error) {
  const int n = layout.tile_count();
  if (n > kMaxContiguousTiles) {
    std::ostringstream out;
    out << "contiguous assignment supports at most " << kMaxContiguousTiles
        << " tiles; use balanced LPT";
    error = out.str();
    return false;
  }
  std::vector<std::uint64_t> prefix(static_cast<std::size_t>(n) + 1, 0);
  for (int i = 0; i < n; ++i) {
    prefix[static_cast<std::size_t>(i) + 1] =
        prefix[static_cast<std::size_t>(i)] +
        layout.tiles[static_cast<std::size_t>(i)].pixels();
  }
  const std::uint64_t inf = std::numeric_limits<std::uint64_t>::max();
  const std::size_t rows = static_cast<std::size_t>(gpu_count) + 1;
  std::vector<std::vector<std::uint64_t>> dp(
      rows, std::vector<std::uint64_t>(static_cast<std::size_t>(n) + 1, inf));
  std::vector<std::vector<int>> cut(
      rows, std::vector<int>(static_cast<std::size_t>(n) + 1, 0));
  dp[0][0] = 0;
  for (int g = 1; g <= gpu_count; ++g) {
    for (int i = g; i <= n; ++i) {
      for (int j = g - 1; j < i; ++j) {
        const std::uint64_t segment =
            prefix[static_cast<std::size_t>(i)] -
            prefix[static_cast<std::size_t>(j)];
        const std::uint64_t candidate =
            std::max(dp[static_cast<std::size_t>(g) - 1]
                       [static_cast<std::size_t>(j)],
                     segment);
        if (candidate < dp[static_cast<std::size_t>(g)]
                           [static_cast<std::size_t>(i)]) {
          dp[static_cast<std::size_t>(g)][static_cast<std::size_t>(i)] = candidate;
          cut[static_cast<std::size_t>(g)][static_cast<std::size_t>(i)] = j;
        }
      }
    }
  }
  groups.assign(static_cast<std::size_t>(gpu_count), {});
  int end = n;
  for (int g = gpu_count; g >= 1; --g) {
    const int start = cut[static_cast<std::size_t>(g)][static_cast<std::size_t>(end)];
    groups[static_cast<std::size_t>(g) - 1].assign(
        static_cast<std::size_t>(end - start), 0);
    for (int i = start; i < end; ++i) {
      groups[static_cast<std::size_t>(g) - 1]
            [static_cast<std::size_t>(i - start)] = i;
    }
    end = start;
  }
  return true;
}

}  // namespace

const char* assignment_policy_name(AssignmentPolicy policy) {
  switch (policy) {
    case AssignmentPolicy::kBalancedLpt:
      return "balanced-lpt";
    case AssignmentPolicy::kContiguousRanges:
      return "contiguous-ranges";
  }
  return "?";
}

const char* scale_mode_name(ScaleMode mode) {
  switch (mode) {
    case ScaleMode::kQuality:
      return "quality";
    case ScaleMode::kThroughput:
      return "throughput";
  }
  return "?";
}

const char* evidence_name(Evidence evidence) {
  switch (evidence) {
    case Evidence::kEstimated:
      return "estimated";
    case Evidence::kClaimed:
      return "claimed";
    case Evidence::kMeasured:
      return "measured";
  }
  return "?";
}

tiles::TileLayoutPreset default_layout_for_gpus(int gpu_count) {
  if (gpu_count <= 1) {
    return tiles::TileLayoutPreset::k4x2;
  }
  if (gpu_count == 2) {
    return tiles::TileLayoutPreset::k8x2;
  }
  return tiles::TileLayoutPreset::k16x2;
}

bool assign_tiles(const TileLayout& layout, int gpu_count,
                  AssignmentPolicy policy, GpuTileAssignment& assignment,
                  std::string& error) {
  error.clear();
  if (!validate_frame_geometry(layout, error)) {
    return false;
  }
  if (gpu_count < 1 || gpu_count > kMaxGpuCount) {
    std::ostringstream out;
    out << "GPU count must be 1.." << kMaxGpuCount;
    error = out.str();
    return false;
  }
  if (layout.tile_count() < gpu_count) {
    std::ostringstream out;
    out << "layout has " << layout.tile_count() << " tiles for " << gpu_count
        << " GPUs; use more tiles or fewer GPUs";
    error = out.str();
    return false;
  }

  const std::vector<double> factors =
      content_factors(layout.tile_count(), 0.0);  // assignment is area-based
  std::vector<std::vector<int>> groups;
  if (policy == AssignmentPolicy::kBalancedLpt) {
    if (!assign_balanced_lpt(layout, gpu_count, factors, groups, error)) {
      return false;
    }
  } else if (!assign_contiguous_dp(layout, gpu_count, groups, error)) {
    return false;
  }

  fill_group_totals(layout, groups, factors, assignment);
  const LoadMetrics metrics = compute_metrics(layout, groups, factors);
  assignment.area_imbalance = metrics.area_imbalance;
  assignment.cost_imbalance = metrics.cost_imbalance;
  assignment.load_min_ratio = metrics.load_min_ratio;
  assignment.load_cv = metrics.load_cv;
  return true;
}

FrameMemory frame_memory(const TileLayout& layout, int gpu_count,
                         int reference_frames, int bit_depth) {
  FrameMemory memory;
  const int gpus = std::max(1, gpu_count);
  const int refs = std::max(0, reference_frames);
  memory.frame_bytes = frame_bytes_for(layout, bit_depth);
  memory.frame_mb = static_cast<double>(memory.frame_bytes) / kMib;
  memory.ref_pool_mb_per_gpu = static_cast<double>(refs) * memory.frame_mb;
  memory.ref_total_mb = static_cast<double>(gpus) * memory.ref_pool_mb_per_gpu;
  memory.ref_extra_mb =
      static_cast<double>(gpus - 1) * memory.ref_pool_mb_per_gpu;
  return memory;
}

VramBreakdown estimate_vram(const TileLayout& layout,
                            std::span<const int> local_tile_indices,
                            int reference_frames, int lookahead_frames,
                            double rdo_multiplier, double headroom_mb,
                            int bit_depth, double vram_cap_mb) {
  VramBreakdown vram;
  vram.cap_mb = vram_cap_mb;
  const std::uint64_t frame_bytes = frame_bytes_for(layout, bit_depth);
  const double frame_mb = static_cast<double>(frame_bytes) / kMib;
  const double state_mb =
      std::max(1.0, static_cast<double>(layout.width) *
                        static_cast<double>(layout.height) / 4.0 / kMib);
  vram.persistent_mb =
      static_cast<double>(reference_frames + lookahead_frames + 2) * frame_mb +
      state_mb;

  std::uint64_t max_local_pixels = 0;
  for (int index : local_tile_indices) {
    if (index < 0 || index >= layout.tile_count()) {
      continue;
    }
    max_local_pixels =
        std::max(max_local_pixels,
                 layout.tiles[static_cast<std::size_t>(index)].pixels());
  }
  const std::size_t local_tiles = local_tile_indices.size();
  const double scratch_per_tile =
      static_cast<double>(max_local_pixels) * 24.0 + 2.0 * kMib;
  vram.scratch_mb =
      static_cast<double>(local_tiles) * scratch_per_tile / kMib;
  vram.candidate_mb = frame_mb * rdo_multiplier;
  vram.headroom_mb = headroom_mb;
  vram.live_mb = vram.persistent_mb + vram.scratch_mb + vram.candidate_mb +
                 vram.headroom_mb;
  vram.fits = vram.live_mb <= vram.cap_mb;
  return vram;
}

double pcie_effective_gbps(int pcie_gen, int pcie_width, double efficiency) {
  const double per_lane = pcie_per_lane_gbps(pcie_gen);
  return per_lane * static_cast<double>(pcie_width) * efficiency;
}

HostTransferPlan plan_host_transfer(const TileLayout& layout,
                                    const PlannerConfig& config) {
  HostTransferPlan plan;
  const int gpus = std::max(1, config.gpu_count);
  plan.adaptive_cdf = config.adaptive_cdf;
  plan.effective_gbps =
      pcie_effective_gbps(config.pcie_gen, config.pcie_width,
                          config.pcie_efficiency);
  const std::uint64_t frame_bytes = frame_bytes_for(layout, config.bit_depth);
  const double payload_bytes =
      config.target_fps > 0.0
          ? config.bitrate_mbps * 1.0e6 / 8.0 / config.target_fps
          : 0.0;

  std::array<double, 5> parts{};
  parts[0] = static_cast<double>(gpus) *
             static_cast<double>(frame_bytes);  // full-frame upload upper bound
  parts[1] = payload_bytes;
  if (gpus > 1) {
    parts[2] = static_cast<double>(frame_bytes);  // recon gather, once
    parts[3] = static_cast<double>(gpus - 1) *
               static_cast<double>(frame_bytes);  // reference fill
  }
  if (gpus > 1 && config.adaptive_cdf) {
    const int cdf_tiles = std::min(8, layout.tile_count());
    parts[4] = static_cast<double>(cdf_tiles) * kCdfKib * 1024.0;
  }
  plan.input_upload_mb = parts[0] / kMib;
  plan.tile_payload_mb = parts[1] / kMib;
  plan.recon_gather_mb = parts[2] / kMib;
  plan.ref_fill_mb = parts[3] / kMib;
  plan.cdf_gather_mb = parts[4] / kMib;

  double total_bytes = 0.0;
  for (double part : parts) {
    if (part > 0.0) {
      total_bytes += part;
      ++plan.phases;
    }
  }
  plan.total_mb = total_bytes / kMib;
  plan.total_ms = plan.effective_gbps > 0.0
                      ? total_bytes / (plan.effective_gbps * 1.0e9) * 1000.0 +
                            0.05 * static_cast<double>(plan.phases)
                      : 0.0;
  if (config.target_fps > 0.0) {
    plan.pct_of_frame_budget =
        100.0 * plan.total_ms / (1000.0 / config.target_fps);
  }
  return plan;
}

ScalingModel estimate_scaling_model(const TileLayout& layout,
                                    const PlannerConfig& config,
                                    const GpuTileAssignment& assignment,
                                    double transfer_ms) {
  ScalingModel model;
  const int gpus = std::max(1, config.gpu_count);
  const double megapixels =
      static_cast<double>(layout.width) * static_cast<double>(layout.height) /
      1.0e6;
  model.compute_ms_1gpu = megapixels * config.ms_per_megapixel;

  PlannerConfig single = config;
  single.gpu_count = 1;
  const HostTransferPlan single_transfer = plan_host_transfer(layout, single);
  model.baseline_ms = model.compute_ms_1gpu + single_transfer.total_ms;

  model.frame_ms =
      config.serial_fraction * model.compute_ms_1gpu +
      (1.0 - config.serial_fraction) * assignment.cost_imbalance *
          model.compute_ms_1gpu / static_cast<double>(gpus) +
      transfer_ms;
  if (model.frame_ms > 0.0) {
    model.speedup = model.baseline_ms / model.frame_ms;
    model.throughput_fps = 1000.0 / model.frame_ms;
  }
  model.efficiency = model.speedup / static_cast<double>(gpus);
  model.measured = false;
  return model;
}

bool plan_multi_gpu(const TileLayout& layout, const PlannerConfig& config,
                    MultiGpuPlan& plan, std::string& error) {
  error.clear();
  plan = MultiGpuPlan{};
  plan.layout = layout;
  plan.config = config;
  plan.measured = false;
  plan.disclaimer =
      "Planning model only. No multi-GPU hardware was used; every timing and "
      "scaling figure is simulated, not measured. VRAM numbers are estimates, "
      "not allocations.";
  plan.tile_count = layout.tile_count();

  if (!validate_frame_geometry(layout, error)) {
    return false;
  }
  if (config.gpu_count < 1 || config.gpu_count > kMaxGpuCount) {
    error = "GPU count must be 1..8";
    return false;
  }
  if (config.reference_frames < 1 || config.reference_frames > 16) {
    error = "reference frame count must be 1..16";
    return false;
  }
  if (config.lookahead_frames < 0 || config.lookahead_frames > 64) {
    error = "lookahead frame count must be 0..64";
    return false;
  }
  if (config.bit_depth != 8 && config.bit_depth != 10) {
    error = "only 8-bit and 10-bit 4:2:0 are modelled";
    return false;
  }
  if (!(config.vram_cap_mb > 0.0)) {
    error = "VRAM cap must be positive";
    return false;
  }
  if (!(config.headroom_mb >= 0.0) || !(config.rdo_multiplier >= 0.0)) {
    error = "headroom and RDO multiplier must be non-negative";
    return false;
  }
  if (!(config.target_fps > 0.0) || config.bitrate_mbps < 0.0) {
    error = "target fps must be positive and bitrate non-negative";
    return false;
  }
  if (pcie_per_lane_gbps(config.pcie_gen) <= 0.0 || config.pcie_width < 1 ||
      config.pcie_width > 16 || !(config.pcie_efficiency > 0.0) ||
      config.pcie_efficiency > 1.0) {
    error = "invalid PCIe generation, width, or efficiency";
    return false;
  }
  if (config.serial_fraction < 0.0 || config.serial_fraction >= 1.0) {
    error = "serial fraction must be in [0, 1)";
    return false;
  }
  if (!(config.ms_per_megapixel > 0.0)) {
    error = "ms per megapixel must be positive";
    return false;
  }
  if (config.content_skew < 0.0 || config.content_skew > 1.0) {
    error = "content skew must be in [0, 1]";
    return false;
  }

  GpuTileAssignment assignment;
  if (!assign_tiles(layout, config.gpu_count, config.assignment, assignment,
                    error)) {
    return false;
  }
  plan.assignment = assignment;
  plan.memory = frame_memory(layout, config.gpu_count, config.reference_frames,
                             config.bit_depth);
  plan.transfer = plan_host_transfer(layout, config);

  // Content-weighted metrics for the analytic prior.
  const std::vector<double> factors =
      content_factors(layout.tile_count(), config.content_skew);
  const LoadMetrics metrics = compute_metrics(layout, assignment.tile_indices,
                                              factors);
  plan.assignment.cost_imbalance = metrics.cost_imbalance;
  plan.assignment.load_min_ratio = metrics.load_min_ratio;
  plan.assignment.load_cv = metrics.load_cv;

  plan.gpus.clear();
  plan.gpus.reserve(static_cast<std::size_t>(config.gpu_count));
  plan.vram_live_max_mb = 0.0;
  plan.vram_fits = true;
  for (int g = 0; g < config.gpu_count; ++g) {
    GpuPlan gpu;
    gpu.gpu_index = g;
    gpu.tile_indices = assignment.tile_indices[static_cast<std::size_t>(g)];
    gpu.tile_count = static_cast<int>(gpu.tile_indices.size());
    gpu.tile_area_pixels = assignment.area_pixels[static_cast<std::size_t>(g)];
    for (int index : gpu.tile_indices) {
      gpu.max_tile_pixels =
          std::max(gpu.max_tile_pixels,
                   layout.tiles[static_cast<std::size_t>(index)].pixels());
    }
    gpu.vram = estimate_vram(layout, gpu.tile_indices,
                             config.reference_frames, config.lookahead_frames,
                             config.rdo_multiplier, config.headroom_mb,
                             config.bit_depth, config.vram_cap_mb);
    plan.vram_live_max_mb = std::max(plan.vram_live_max_mb, gpu.vram.live_mb);
    plan.vram_fits = plan.vram_fits && gpu.vram.fits;
    plan.gpus.push_back(std::move(gpu));
  }

  plan.scaling = estimate_scaling_model(layout, config, plan.assignment,
                                        plan.transfer.total_ms);

  // Deterministic per-tile host dispatch.  Raster slot == TileNum from
  // plan_tile_layout(); a worker's payload must be written into that slot and
  // the coordinator merges slots in ascending order.
  plan.dispatch.clear();
  plan.dispatch.reserve(layout.tiles.size());
  std::vector<int> owner(static_cast<std::size_t>(layout.tile_count()), -1);
  for (int g = 0; g < config.gpu_count; ++g) {
    for (int index : assignment.tile_indices[static_cast<std::size_t>(g)]) {
      owner[static_cast<std::size_t>(index)] = g;
    }
  }
  for (int t = 0; t < layout.tile_count(); ++t) {
    const tiles::TileRect& rect = layout.tiles[static_cast<std::size_t>(t)];
    TileDispatchEntry entry;
    entry.raster_slot = t;
    entry.tile_index = t;
    entry.gpu_index = owner[static_cast<std::size_t>(t)];
    entry.sb_col_start = rect.sb_col_start;
    entry.sb_row_start = rect.sb_row_start;
    entry.sb_cols = rect.sb_cols;
    entry.sb_rows = rect.sb_rows;
    entry.x0 = rect.x0;
    entry.y0 = rect.y0;
    entry.x1 = rect.x1;
    entry.y1 = rect.y1;
    plan.dispatch.push_back(entry);
  }

  plan.measured = false;
  plan.evidence = Evidence::kEstimated;
  plan.scaling.evidence = Evidence::kEstimated;
  plan.valid = true;
  return true;
}

}  // namespace av2enc::multigpu
