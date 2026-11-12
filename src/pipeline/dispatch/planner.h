// Deterministic tile-to-GPU planning for av2enc.
//
// Status: planning model only. There is no multi-GPU host, no measured
// scaling, and no bitstream output. This module computes:
//   * a deterministic tile -> GPU assignment for 1/2/4 GPUs,
//   * duplicated reference-pool sizes (VRAM is not pooled),
//   * host-mediated transfer parts and an estimated transfer time,
//   * a per-GPU `compat8g` VRAM estimate against the 7,500 MB cap,
//   * an analytic scaling prior, explicitly labelled as a model.
//
// See docs/multi-gpu.md and bench/multi_gpu_plan/. The default assignment is
// longest-processing-time-first (LPT) on tile pixel area with ties broken by
// lower tile index and then lower GPU index. That reproduces the documented
// 1080p/4K sets for 4x2, 8x2, and 16x2. The bitstream merge order is always
// raster TileNum, never completion order, so a GPU's set need not be
// contiguous.
//
// No encoder or CUDA code is called. No `cudaMalloc`, streams, events, or
// devices are touched.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "../tiles/tile_layout.h"

namespace av2enc::multigpu {

using tiles::TileLayout;

enum class AssignmentPolicy : std::uint8_t {
  kBalancedLpt = 0,
  kContiguousRanges = 1,
};

enum class ScaleMode : std::uint8_t {
  kQuality = 0,
  kThroughput = 1,
};

const char* assignment_policy_name(AssignmentPolicy policy);
const char* scale_mode_name(ScaleMode mode);

// Evidence attached to every timing/scaling number this module produces.
// kEstimated: analytic prior (the only value reachable today).
// kClaimed: a number a caller explicitly asserts without a measurement.
// kMeasured: backed by a recorded run with the same clip/QP/layout.
enum class Evidence : std::uint8_t {
  kEstimated = 0,
  kClaimed = 1,
  kMeasured = 2,
};
const char* evidence_name(Evidence evidence);

// docs/multi-gpu.md default: 1 GPU -> 4x2, 2 GPUs -> 8x2, 3+ GPUs -> 16x2.
// 16x2 is not decoder-verified yet; callers must gate it with dav2d first.
tiles::TileLayoutPreset default_layout_for_gpus(int gpu_count);

struct PlannerConfig {
  int gpu_count = 1;
  AssignmentPolicy assignment = AssignmentPolicy::kBalancedLpt;
  ScaleMode scale_mode = ScaleMode::kQuality;

  // compat8g defaults. 8-bit 4:2:0 only; 10-bit is estimated but not a target.
  int reference_frames = 4;
  int lookahead_frames = 4;
  double rdo_multiplier = 1.0;
  double headroom_mb = 512.0;
  double vram_cap_mb = 7500.0;
  int bit_depth = 8;

  // Host transfer model.
  bool adaptive_cdf = false;   // default bitstream is disable_cdf_update=1
  double bitrate_mbps = 1.0;   // encoded payload at target_fps
  double target_fps = 1.0;
  int pcie_gen = 4;            // 3, 4, or 5
  int pcie_width = 16;         // lanes
  double pcie_efficiency = 0.85;

  // Analytic scaling prior (not a measurement).
  double serial_fraction = 0.035;
  double ms_per_megapixel = 481.0;
  double content_skew = 0.0;   // 0 = pure area; deterministic synthetic prior otherwise
};

struct FrameMemory {
  std::uint64_t frame_bytes = 0;
  double frame_mb = 0.0;
  double ref_pool_mb_per_gpu = 0.0;
  double ref_total_mb = 0.0;
  double ref_extra_mb = 0.0;  // (G-1) * ref_pool_mb_per_gpu
};

struct GpuTileAssignment {
  // Per GPU, raster-sorted tile indices. Empty only for an invalid plan.
  std::vector<std::vector<int>> tile_indices;
  std::vector<int> tile_counts;
  std::vector<std::uint64_t> area_pixels;
  std::vector<std::uint64_t> area_sb_cells;
  std::vector<double> cost_units;

  double area_imbalance = 1.0;  // max_gpu_area / mean_gpu_area
  double cost_imbalance = 1.0;  // max_gpu_cost / mean_gpu_cost
  double load_min_ratio = 1.0;  // min_gpu_cost / mean_gpu_cost
  double load_cv = 0.0;         // coefficient of variation of cost loads
};

struct HostTransferPlan {
  double input_upload_mb = 0.0;   // G * frame_bytes (conservative upper bound)
  double tile_payload_mb = 0.0;   // compressed payload D2H
  double recon_gather_mb = 0.0;   // each GPU's local reconstruction, once
  double ref_fill_mb = 0.0;       // missing reference regions H2D
  double cdf_gather_mb = 0.0;     // only with opt-in adaptive CDFs
  double total_mb = 0.0;

  double effective_gbps = 0.0;
  double total_ms = 0.0;
  double pct_of_frame_budget = 0.0;  // at target_fps
  int phases = 0;
  bool adaptive_cdf = false;
};

struct VramBreakdown {
  double persistent_mb = 0.0;
  double scratch_mb = 0.0;
  double candidate_mb = 0.0;
  double headroom_mb = 0.0;
  double live_mb = 0.0;
  double cap_mb = 7500.0;
  bool fits = false;
};

struct GpuPlan {
  int gpu_index = 0;
  std::vector<int> tile_indices;
  int tile_count = 0;
  std::uint64_t tile_area_pixels = 0;
  std::uint64_t max_tile_pixels = 0;
  VramBreakdown vram;
};

// One unit of host-side dispatch work.  `raster_slot` is the merge position
// in the tile-group OBU (always raster TileNum, never completion order);
// `gpu_index` is the deterministic owner.  A worker must write its payload
// into `raster_slot`, and the coordinator merges slots in that order.
struct TileDispatchEntry {
  int raster_slot = 0;
  int tile_index = 0;
  int gpu_index = 0;
  int sb_col_start = 0;
  int sb_row_start = 0;
  int sb_cols = 0;
  int sb_rows = 0;
  int x0 = 0;  // visible luma rect
  int y0 = 0;
  int x1 = 0;
  int y1 = 0;
};

// Analytic only. `measured` is always false and there is no speedup claim
// until the validation sequence in docs/multi-gpu.md §10 runs.
struct ScalingModel {
  double compute_ms_1gpu = 0.0;
  double frame_ms = 0.0;
  double baseline_ms = 0.0;
  double speedup = 0.0;
  double efficiency = 0.0;
  double throughput_fps = 0.0;
  bool measured = false;
  Evidence evidence = Evidence::kEstimated;
};

struct MultiGpuPlan {
  bool valid = false;
  std::string error;
  TileLayout layout;
  PlannerConfig config;
  int tile_count = 0;
  GpuTileAssignment assignment;
  FrameMemory memory;
  HostTransferPlan transfer;
  std::vector<GpuPlan> gpus;
  std::vector<TileDispatchEntry> dispatch;  // raster order, one per tile
  ScalingModel scaling;
  double vram_live_max_mb = 0.0;
  bool vram_fits = false;
  bool measured = false;
  Evidence evidence = Evidence::kEstimated;
  std::string disclaimer;
};

// LPT / optimal contiguous partition. `every tile <= 1`.
bool assign_tiles(const TileLayout& layout, int gpu_count,
                  AssignmentPolicy policy, GpuTileAssignment& assignment,
                  std::string& error);

FrameMemory frame_memory(const TileLayout& layout, int gpu_count,
                         int reference_frames, int bit_depth);

VramBreakdown estimate_vram(const TileLayout& layout,
                            std::span<const int> local_tile_indices,
                            int reference_frames, int lookahead_frames,
                            double rdo_multiplier, double headroom_mb,
                            int bit_depth, double vram_cap_mb);

HostTransferPlan plan_host_transfer(const TileLayout& layout,
                                    const PlannerConfig& config);

double pcie_effective_gbps(int pcie_gen, int pcie_width,
                           double efficiency);

ScalingModel estimate_scaling_model(const TileLayout& layout,
                                    const PlannerConfig& config,
                                    const GpuTileAssignment& assignment,
                                    double transfer_ms);

bool plan_multi_gpu(const TileLayout& layout, const PlannerConfig& config,
                    MultiGpuPlan& plan, std::string& error);

}  // namespace av2enc::multigpu
