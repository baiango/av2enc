// Deterministic tile dispatch and raster-order merge.
//
// The planner (`../planner.h`) decides which tile belongs to which GPU. This
// header turns that assignment into two indexed views:
//
//   by_tile[tile_index]  -> owner slot + local index
//   by_gpu[slot][local]  -> tile index (always raster sorted)
//
// and merges payloads strictly in raster TileNum order, independent of the
// order in which GPUs or tiles completed. Completion order is never an input
// to the bitstream order.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "../planner.h"

namespace av2enc::multigpu::runtime {

struct DispatchEntry {
  int tile_index = -1;
  int gpu_slot = -1;
  int ordinal = -1;
  int local_index = -1;  // index inside by_gpu[gpu_slot]
};

struct DispatchTable {
  bool valid = false;
  std::string error;
  int tile_count = 0;
  std::vector<int> gpu_ordinals;  // slot -> physical ordinal
  std::vector<DispatchEntry> by_tile;               // raster order
  std::vector<std::vector<DispatchEntry>> by_gpu;   // GPU-local raster order
};

// `gpu_ordinals` must have one entry per plan GPU slot. The table is rejected
// if any tile is missing, duplicated, out of range, or assigned to a GPU whose
// plan tile list does not match the assignment block.
DispatchTable build_dispatch_table(const MultiGpuPlan& plan,
                                   std::span<const int> gpu_ordinals,
                                   std::string* error = nullptr);

struct TilePayloadView {
  int tile_index = -1;
  std::span<const std::uint8_t> bytes;
};

struct MergedFrame {
  bool valid = false;
  std::string error;
  int tile_count = 0;
  // Strict raster TileNum order, regardless of the input order.
  std::vector<std::vector<std::uint8_t>> tiles;
  std::vector<std::uint32_t> tile_sizes;
  std::uint64_t total_bytes = 0;
};

// Requires exactly one payload of at least 2 bytes per tile in
// [0, tile_count). Duplicate
// or missing tiles abort the whole frame; a partial merge is never returned.
bool merge_payloads(const MultiGpuPlan& plan,
                    std::span<const TilePayloadView> payloads,
                    MergedFrame& out, std::string* error = nullptr);

}  // namespace av2enc::multigpu::runtime
