// Host-side contract between the frame pipeline planner and the CUDA workers.
//
// Status: host-only integration layer. No CUDA header is included, no device
// is queried, and no allocation is performed. It turns a
// av2enc::pipeline::FramePipelinePlan into exactly what one worker per GPU
// consumes, and owns the device-sized arena policy that both the planner and
// the runtime (src/cuda/arena.*) must agree on.
//
// Why this lives under src/cuda/: the arena rule and the per-GPU work split are
// CUDA-runtime policy, even though the data is plain PODs. The existing
// src/cuda/m1/plan/ module is host-only for the same reason.
//
// Ownership: this file does not include cuda_runtime.h, so it can be built and
// tested on a host without a CUDA toolkit. Wiring it to Arena::init() is a
// one-line call in the orchestrator:
//
//   Arena arena;
//   arena.init(budget.capacity_bytes);
//   ... and pass the same bytes to build_gpu_dispatch().
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "encoder/pipeline/pipeline.h"

namespace av2enc::cuda::pipeline {

inline constexpr std::uint64_t kMiB = 1024ull * 1024ull;
inline constexpr std::uint64_t kDefaultHeadroomBytes = 512ull * kMiB;

// Device-sized arena budget (§AGENTS.md / docs/vram.md):
//
//   capacity = min(total, free) - headroom
//   capacity = min(capacity, user_cap)     when user_cap != 0
//
// There is no fixed VRAM preset. When the runtime cannot query the device
// (total == free == 0) an explicit user cap is required; otherwise the plan
// fails instead of guessing a device size.
struct ArenaBudget {
  std::uint64_t device_total_bytes = 0;
  std::uint64_t device_free_bytes = 0;
  std::uint64_t headroom_bytes = kDefaultHeadroomBytes;
  std::uint64_t requested_cap_bytes = 0;
  std::uint64_t capacity_bytes = 0;
  bool device_known = false;
  bool capped_by_request = false;
};

// Returns false with a message when the inputs cannot produce a usable arena.
// `total` and `free` must both be zero (unknown) or both positive (known).
bool compute_arena_budget(std::uint64_t total_bytes, std::uint64_t free_bytes,
                          std::uint64_t headroom_bytes,
                          std::uint64_t cap_bytes, ArenaBudget& budget,
                          std::string* error = nullptr);

// One tile's dispatch record. All geometry is the validated TileRect; the
// worker must not recompute tile starts. qindex is the effective per-tile QP
// and delta_q_reduced is the value to code at the tile's first superblock
// (kDeltaQ only; 0 otherwise).
struct GpuTileWork {
  int tile_index = 0;
  int row = 0;
  int col = 0;
  int x0 = 0;
  int y0 = 0;
  int x1 = 0;
  int y1 = 0;
  int sb_col_start = 0;
  int sb_row_start = 0;
  int sb_cols = 0;
  int sb_rows = 0;
  int target_qindex = 0;
  int qindex = 0;
  int alt_q = 0;
  int segment_id = -1;
  int delta_q_reduced = 0;
};

// What one GPU worker owns for one frame.
struct GpuWorkerPlan {
  int gpu_index = 0;
  std::uint64_t arena_capacity_bytes = 0;
  double vram_live_mb = 0.0;
  double vram_cap_mb = 0.0;
  bool vram_fits = false;
  std::uint64_t tile_area_pixels = 0;
  std::vector<GpuTileWork> tiles;  // raster TileNum order within this worker
};

// Splits a validated frame plan into per-GPU worker plans. Validates that
// every tile is assigned exactly once, that the GPU index matches the
// assignment, and that the per-GPU VRAM estimate fits the arena when the plan
// was built with require_vram_fit = false (callers may still inspect fits).
bool build_gpu_dispatch(const av2enc::pipeline::FramePipelinePlan& plan,
                        std::uint64_t arena_capacity_bytes,
                        std::vector<GpuWorkerPlan>& workers,
                        std::string* error = nullptr);

}  // namespace av2enc::cuda::pipeline
