#include "dispatch.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <utility>

namespace av2enc::multigpu::runtime {
namespace {

void set_error(std::string* error, const std::string& text) {
  if (error != nullptr) {
    *error = text;
  }
}

}  // namespace

DispatchTable build_dispatch_table(const MultiGpuPlan& plan,
                                   std::span<const int> gpu_ordinals,
                                   std::string* error) {
  DispatchTable table;
  if (!plan.valid) {
    table.error = "cannot dispatch an invalid planner result";
    set_error(error, table.error);
    return table;
  }
  const int gpu_count = static_cast<int>(plan.gpus.size());
  if (gpu_count < 1) {
    table.error = "planner returned no GPUs";
    set_error(error, table.error);
    return table;
  }
  if (static_cast<int>(gpu_ordinals.size()) != gpu_count) {
    table.error = "GPU ordinal list does not match planner GPU count";
    set_error(error, table.error);
    return table;
  }

  const int tile_count = plan.layout.tile_count();
  if (tile_count < 1) {
    table.error = "planner layout has no tiles";
    set_error(error, table.error);
    return table;
  }

  table.tile_count = tile_count;
  table.gpu_ordinals.assign(gpu_ordinals.begin(), gpu_ordinals.end());
  table.by_tile.assign(static_cast<std::size_t>(tile_count), DispatchEntry{});
  table.by_gpu.assign(static_cast<std::size_t>(gpu_count), {});

  std::vector<int> seen(static_cast<std::size_t>(tile_count), 0);
  for (int slot = 0; slot < gpu_count; ++slot) {
    const GpuPlan& gpu = plan.gpus[static_cast<std::size_t>(slot)];
    if (gpu.gpu_index != slot) {
      table.error = "planner GPU index does not match its slot";
      set_error(error, table.error);
      return table;
    }
    for (int local = 0; local < static_cast<int>(gpu.tile_indices.size());
         ++local) {
      const int tile = gpu.tile_indices[static_cast<std::size_t>(local)];
      if (tile < 0 || tile >= tile_count) {
        table.error = "planner assigned a tile outside the layout";
        set_error(error, table.error);
        return table;
      }
      if (local > 0 &&
          tile <= gpu.tile_indices[static_cast<std::size_t>(local) - 1]) {
        table.error = "planner GPU tile list is not raster sorted";
        set_error(error, table.error);
        return table;
      }
      if (seen[static_cast<std::size_t>(tile)] != 0) {
        table.error = "planner assigned a tile twice";
        set_error(error, table.error);
        return table;
      }
      seen[static_cast<std::size_t>(tile)] = 1;

      DispatchEntry entry;
      entry.tile_index = tile;
      entry.gpu_slot = slot;
      entry.ordinal = gpu_ordinals[static_cast<std::size_t>(slot)];
      entry.local_index = local;
      table.by_tile[static_cast<std::size_t>(tile)] = entry;
      table.by_gpu[static_cast<std::size_t>(slot)].push_back(entry);
    }
  }
  for (int tile = 0; tile < tile_count; ++tile) {
    if (seen[static_cast<std::size_t>(tile)] == 0) {
      std::ostringstream out;
      out << "planner left tile " << tile << " unassigned";
      table.error = out.str();
      set_error(error, table.error);
      return table;
    }
  }

  table.valid = true;
  if (error != nullptr) {
    error->clear();
  }
  return table;
}

bool merge_payloads(const MultiGpuPlan& plan,
                    std::span<const TilePayloadView> payloads, MergedFrame& out,
                    std::string* error) {
  out = MergedFrame{};
  if (!plan.valid) {
    out.error = "cannot merge an invalid planner result";
    set_error(error, out.error);
    return false;
  }
  const int tile_count = plan.layout.tile_count();
  if (tile_count < 1) {
    out.error = "planner layout has no tiles";
    set_error(error, out.error);
    return false;
  }

  std::vector<const TilePayloadView*> by_tile(
      static_cast<std::size_t>(tile_count), nullptr);
  for (const TilePayloadView& payload : payloads) {
    if (payload.tile_index < 0 || payload.tile_index >= tile_count) {
      out.error = "payload tile index is outside the layout";
      set_error(error, out.error);
      return false;
    }
    const std::size_t index = static_cast<std::size_t>(payload.tile_index);
    if (by_tile[index] != nullptr) {
      std::ostringstream text;
      text << "duplicate payload for tile " << payload.tile_index;
      out.error = text.str();
      set_error(error, out.error);
      return false;
    }
    if (payload.bytes.size() < 2) {
      std::ostringstream text;
      text << "tile " << payload.tile_index
           << " has a payload smaller than the 2-byte minimum";
      out.error = text.str();
      set_error(error, out.error);
      return false;
    }
    if (payload.bytes.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      out.error = "tile payload exceeds the 32-bit tile-size field";
      set_error(error, out.error);
      return false;
    }
    by_tile[index] = &payload;
  }

  out.tiles.resize(static_cast<std::size_t>(tile_count));
  out.tile_sizes.assign(static_cast<std::size_t>(tile_count), 0u);
  out.total_bytes = 0;
  for (int tile = 0; tile < tile_count; ++tile) {
    const TilePayloadView* payload = by_tile[static_cast<std::size_t>(tile)];
    if (payload == nullptr) {
      std::ostringstream text;
      text << "missing payload for tile " << tile;
      out.error = text.str();
      set_error(error, out.error);
      out.tiles.clear();
      out.tile_sizes.clear();
      return false;
    }
    out.tiles[static_cast<std::size_t>(tile)] =
        std::vector<std::uint8_t>(payload->bytes.begin(), payload->bytes.end());
    out.tile_sizes[static_cast<std::size_t>(tile)] =
        static_cast<std::uint32_t>(payload->bytes.size());
    out.total_bytes += static_cast<std::uint64_t>(payload->bytes.size());
  }

  out.tile_count = tile_count;
  out.valid = true;
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

}  // namespace av2enc::multigpu::runtime
