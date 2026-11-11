#include "cuda/pipeline/pipeline_gpu.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::cuda::pipeline {
namespace {

void set_error(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

}  // namespace

bool compute_arena_budget(std::uint64_t total_bytes,
                          std::uint64_t free_bytes,
                          std::uint64_t headroom_bytes,
                          std::uint64_t cap_bytes, ArenaBudget& budget,
                          std::string* error) {
  budget = ArenaBudget{};
  budget.device_total_bytes = total_bytes;
  budget.device_free_bytes = free_bytes;
  budget.headroom_bytes = headroom_bytes;
  budget.requested_cap_bytes = cap_bytes;

  if (total_bytes == 0 && free_bytes == 0) {
    budget.device_known = false;
    if (cap_bytes == 0) {
      set_error(error,
                "arena budget: device capacity is unknown; provide total/free "
                "bytes or an explicit VRAM cap");
      return false;
    }
    budget.capacity_bytes = cap_bytes;
    budget.capped_by_request = true;
    return true;
  }

  if (total_bytes == 0 || free_bytes == 0) {
    set_error(error,
              "arena budget: provide both device total and free bytes, or "
              "neither");
    return false;
  }

  budget.device_known = true;
  const std::uint64_t usable = std::min(total_bytes, free_bytes);
  if (usable <= headroom_bytes) {
    set_error(error,
              "arena budget: device capacity does not exceed the headroom");
    return false;
  }
  const std::uint64_t device_capacity = usable - headroom_bytes;
  if (cap_bytes != 0 && cap_bytes < device_capacity) {
    budget.capacity_bytes = cap_bytes;
    budget.capped_by_request = true;
  } else {
    budget.capacity_bytes = device_capacity;
    budget.capped_by_request = false;
  }
  return true;
}

bool build_gpu_dispatch(const av2enc::pipeline::FramePipelinePlan& plan,
                        std::uint64_t arena_capacity_bytes,
                        std::vector<GpuWorkerPlan>& workers,
                        std::string* error) {
  workers.clear();
  if (error != nullptr) {
    error->clear();
  }
  if (!plan.valid) {
    set_error(error, "gpu dispatch: frame plan is not valid");
    return false;
  }
  if (arena_capacity_bytes == 0) {
    set_error(error, "gpu dispatch: arena capacity must be positive");
    return false;
  }
  if (plan.tiles.empty()) {
    set_error(error, "gpu dispatch: frame plan has no tiles");
    return false;
  }
  if (plan.gpu.gpus.empty()) {
    set_error(error, "gpu dispatch: frame plan has no GPU assignment");
    return false;
  }

  const std::size_t worker_count = plan.gpu.gpus.size();
  workers.resize(worker_count);
  std::vector<int> assignment(plan.tiles.size(), -1);
  for (std::size_t g = 0; g < worker_count; ++g) {
    GpuWorkerPlan& worker = workers[g];
    const auto& gpu = plan.gpu.gpus[g];
    worker.gpu_index = gpu.gpu_index;
    worker.arena_capacity_bytes = arena_capacity_bytes;
    worker.vram_live_mb = gpu.vram.live_mb;
    worker.vram_cap_mb = gpu.vram.cap_mb;
    worker.vram_fits = gpu.vram.fits;
    worker.tile_area_pixels = gpu.tile_area_pixels;
    worker.tiles.reserve(gpu.tile_indices.size());
    for (int index : gpu.tile_indices) {
      if (index < 0 || static_cast<std::size_t>(index) >= plan.tiles.size()) {
        set_error(error, "gpu dispatch: tile index out of range");
        return false;
      }
      if (assignment[static_cast<std::size_t>(index)] != -1) {
        set_error(error, "gpu dispatch: tile assigned to more than one GPU");
        return false;
      }
      assignment[static_cast<std::size_t>(index)] = gpu.gpu_index;

      const av2enc::pipeline::TilePlan& tile =
          plan.tiles[static_cast<std::size_t>(index)];
      if (tile.gpu_index != gpu.gpu_index) {
        set_error(error,
                  "gpu dispatch: tile GPU index disagrees with the assignment");
        return false;
      }
      GpuTileWork work;
      work.tile_index = tile.index;
      work.row = tile.row;
      work.col = tile.col;
      work.x0 = tile.rect.x0;
      work.y0 = tile.rect.y0;
      work.x1 = tile.rect.x1;
      work.y1 = tile.rect.y1;
      work.sb_col_start = tile.rect.sb_col_start;
      work.sb_row_start = tile.rect.sb_row_start;
      work.sb_cols = tile.rect.sb_cols;
      work.sb_rows = tile.rect.sb_rows;
      work.target_qindex = tile.target_qindex;
      work.qindex = tile.qindex;
      work.alt_q = tile.alt_q;
      work.segment_id = tile.segment_id;
      work.delta_q_reduced = tile.delta_q_reduced;
      worker.tiles.push_back(work);
    }
  }

  for (std::size_t i = 0; i < assignment.size(); ++i) {
    if (assignment[i] < 0) {
      set_error(error, "gpu dispatch: tile is not assigned to any GPU");
      return false;
    }
  }
  return true;
}

}  // namespace av2enc::cuda::pipeline
