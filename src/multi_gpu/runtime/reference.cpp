#include "reference.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <sstream>

namespace av2enc::multigpu::runtime {
namespace {

struct PlaneGeometry {
  int width = 0;
  int height = 0;
  std::uint64_t offset = 0;
  std::size_t bytes_per_sample = 1;
};

std::size_t sample_bytes(const tiles::TileLayout& layout, int bit_depth) {
  (void)layout;
  return bit_depth > 8 ? 2u : 1u;
}

bool frame_layout(const tiles::TileLayout& layout, int bit_depth,
                  PlaneGeometry planes[3], std::string* error) {
  if (layout.width <= 0 || layout.height <= 0 || layout.width % 2 != 0 ||
      layout.height % 2 != 0) {
    if (error != nullptr) {
      *error = "reference exchange needs a positive even 4:2:0 frame";
    }
    return false;
  }
  const std::size_t bps = sample_bytes(layout, bit_depth);
  const std::uint64_t luma_samples =
      static_cast<std::uint64_t>(layout.width) *
      static_cast<std::uint64_t>(layout.height);
  const std::uint64_t chroma_samples =
      static_cast<std::uint64_t>(layout.width / 2) *
      static_cast<std::uint64_t>(layout.height / 2);

  planes[0].width = layout.width;
  planes[0].height = layout.height;
  planes[0].offset = 0;
  planes[0].bytes_per_sample = bps;

  planes[1].width = layout.width / 2;
  planes[1].height = layout.height / 2;
  planes[1].offset = luma_samples * bps;
  planes[1].bytes_per_sample = bps;

  planes[2].width = layout.width / 2;
  planes[2].height = layout.height / 2;
  planes[2].offset = (luma_samples + chroma_samples) * bps;
  planes[2].bytes_per_sample = bps;
  return true;
}

struct PlaneRect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
  std::uint64_t offset = 0;
  std::size_t bytes_per_sample = 1;
  int plane_stride = 0;
};

bool tile_plane_rects(const tiles::TileLayout& layout,
                      const tiles::TileRect& rect, int bit_depth,
                      PlaneRect out[3], std::string* error) {
  PlaneGeometry planes[3];
  if (!frame_layout(layout, bit_depth, planes, error)) {
    return false;
  }
  for (int plane = 0; plane < 3; ++plane) {
    const int shift = plane == 0 ? 0 : 1;
    const int width = rect.pixel_width();
    const int height = rect.pixel_height();
    if (width <= 0 || height <= 0) {
      if (error != nullptr) {
        *error = "tile rectangle is empty";
      }
      return false;
    }
    if (plane != 0 && ((width % 2) != 0 || (height % 2) != 0 ||
                       (rect.x0 % 2) != 0 || (rect.y0 % 2) != 0)) {
      if (error != nullptr) {
        *error = "chroma tile region is not 2x2 aligned";
      }
      return false;
    }
    out[plane].x = rect.x0 >> shift;
    out[plane].y = rect.y0 >> shift;
    out[plane].width = width >> shift;
    out[plane].height = height >> shift;
    out[plane].offset = planes[plane].offset;
    out[plane].bytes_per_sample = planes[plane].bytes_per_sample;
    out[plane].plane_stride = planes[plane].width;
  }
  return true;
}

}  // namespace

std::uint64_t tile_region_bytes(const tiles::TileLayout& layout,
                                const tiles::TileRect& rect, int bit_depth) {
  PlaneRect planes[3];
  if (!tile_plane_rects(layout, rect, bit_depth, planes, nullptr)) {
    return 0;
  }
  std::uint64_t total = 0;
  for (const PlaneRect& plane : planes) {
    total += static_cast<std::uint64_t>(plane.width) *
             static_cast<std::uint64_t>(plane.height) *
             static_cast<std::uint64_t>(plane.bytes_per_sample);
  }
  return total;
}

Status enqueue_tile_region_copy(DeviceBackend& backend, CopyDirection direction,
                                StreamHandle stream, std::uint8_t* dst_base,
                                const std::uint8_t* src_base,
                                std::uint64_t frame_offset,
                                const tiles::TileLayout& layout,
                                const tiles::TileRect& rect, int bit_depth) {
  if (stream == kInvalidStream || dst_base == nullptr || src_base == nullptr) {
    return Status::kInvalidArgument;
  }
  PlaneRect planes[3];
  if (!tile_plane_rects(layout, rect, bit_depth, planes, nullptr)) {
    return Status::kInvalidArgument;
  }
  for (const PlaneRect& plane : planes) {
    const std::size_t row_bytes = static_cast<std::size_t>(plane.width) *
                                  plane.bytes_per_sample;
    if (row_bytes == 0) {
      continue;
    }
    for (int row = 0; row < plane.height; ++row) {
      const std::uint64_t row_offset =
          frame_offset + plane.offset +
          (static_cast<std::uint64_t>(plane.y) +
           static_cast<std::uint64_t>(row)) *
              static_cast<std::uint64_t>(plane.plane_stride) *
              plane.bytes_per_sample +
          static_cast<std::uint64_t>(plane.x) * plane.bytes_per_sample;
      std::uint8_t* dst = dst_base + row_offset;
      const std::uint8_t* src = src_base + row_offset;
      Status status = Status::kOk;
      switch (direction) {
        case CopyDirection::kHostToDevice:
          status = backend.copy_h2d_async(stream, dst, src, row_bytes);
          break;
        case CopyDirection::kDeviceToHost:
          status = backend.copy_d2h_async(stream, dst, src, row_bytes);
          break;
        case CopyDirection::kDeviceToDevice:
          status = backend.copy_d2d_async(stream, dst, src, row_bytes);
          break;
      }
      if (!is_ok(status)) {
        return status;
      }
    }
  }
  return Status::kOk;
}

Status copy_tile_region_host(std::uint8_t* dst_frame,
                             const std::uint8_t* src_frame,
                             const tiles::TileLayout& layout,
                             const tiles::TileRect& rect, int bit_depth) {
  if (dst_frame == nullptr || src_frame == nullptr) {
    return Status::kInvalidArgument;
  }
  PlaneRect planes[3];
  if (!tile_plane_rects(layout, rect, bit_depth, planes, nullptr)) {
    return Status::kInvalidArgument;
  }
  for (const PlaneRect& plane : planes) {
    const std::size_t row_bytes = static_cast<std::size_t>(plane.width) *
                                  plane.bytes_per_sample;
    for (int row = 0; row < plane.height; ++row) {
      const std::uint64_t row_offset =
          plane.offset +
          (static_cast<std::uint64_t>(plane.y) +
           static_cast<std::uint64_t>(row)) *
              static_cast<std::uint64_t>(plane.plane_stride) *
              plane.bytes_per_sample +
          static_cast<std::uint64_t>(plane.x) * plane.bytes_per_sample;
      std::memcpy(dst_frame + row_offset, src_frame + row_offset, row_bytes);
    }
  }
  return Status::kOk;
}

ReferenceSchedule plan_reference_schedule(
    const MultiGpuPlan& plan, std::span<const int> gpu_ordinals,
    std::span<const std::uint64_t> payload_bytes_per_gpu) {
  ReferenceSchedule schedule;
  if (!plan.valid) {
    schedule.error = "cannot schedule an invalid planner result";
    return schedule;
  }
  const int gpu_count = static_cast<int>(plan.gpus.size());
  const int tile_count = plan.layout.tile_count();
  if (gpu_count < 1 || tile_count < 1) {
    schedule.error = "planner returned an empty GPU or tile set";
    return schedule;
  }
  if (!gpu_ordinals.empty() &&
      static_cast<int>(gpu_ordinals.size()) != gpu_count) {
    schedule.error = "GPU ordinal list does not match GPU count";
    return schedule;
  }
  if (!payload_bytes_per_gpu.empty() &&
      static_cast<int>(payload_bytes_per_gpu.size()) != gpu_count) {
    schedule.error = "payload byte list does not match GPU count";
    return schedule;
  }

  auto ordinal_for = [&](int slot) {
    return gpu_ordinals.empty()
               ? plan.gpus[static_cast<std::size_t>(slot)].gpu_index
               : gpu_ordinals[static_cast<std::size_t>(slot)];
  };

  schedule.gpu_count = gpu_count;
  schedule.tile_count = tile_count;
  schedule.reference_frames = plan.config.reference_frames;

  const std::uint64_t frame_bytes = plan.memory.frame_bytes;
  const int bit_depth = plan.config.bit_depth;

  // Slot -> tile owner, and a raster owner array.
  std::vector<int> owner(static_cast<std::size_t>(tile_count), -1);
  for (int slot = 0; slot < gpu_count; ++slot) {
    for (int tile : plan.gpus[static_cast<std::size_t>(slot)].tile_indices) {
      if (tile < 0 || tile >= tile_count) {
        schedule.error = "planner tile index is outside the layout";
        return schedule;
      }
      owner[static_cast<std::size_t>(tile)] = slot;
    }
  }
  for (int tile = 0; tile < tile_count; ++tile) {
    if (owner[static_cast<std::size_t>(tile)] < 0) {
      schedule.error = "planner left a tile unassigned";
      return schedule;
    }
  }

  for (int slot = 0; slot < gpu_count; ++slot) {
    TransferOp op;
    op.phase = TransferPhase::kSourceUpload;
    op.direction = TransferDirection::kHostToDevice;
    op.gpu_slot = slot;
    op.ordinal = ordinal_for(slot);
    op.tile_index = -1;
    op.bytes = frame_bytes;
    op.stream_role = 1;
    op.depends_on = -1;
    schedule.ops.push_back(op);
    schedule.source_upload_bytes += op.bytes;
  }

  for (int slot = 0; slot < gpu_count; ++slot) {
    const std::uint64_t bytes = payload_bytes_per_gpu.empty()
                                    ? 0
                                    : payload_bytes_per_gpu[static_cast<std::size_t>(slot)];
    if (bytes == 0) {
      continue;
    }
    TransferOp op;
    op.phase = TransferPhase::kPayloadDownload;
    op.direction = TransferDirection::kDeviceToHost;
    op.gpu_slot = slot;
    op.ordinal = ordinal_for(slot);
    op.tile_index = -1;
    op.bytes = bytes;
    op.stream_role = 1;
    op.depends_on = -1;
    schedule.ops.push_back(op);
    schedule.payload_bytes += bytes;
  }

  for (int tile = 0; tile < tile_count; ++tile) {
    const int slot = owner[static_cast<std::size_t>(tile)];
    const tiles::TileRect& rect =
        plan.layout.tiles[static_cast<std::size_t>(tile)];
    const std::uint64_t bytes = tile_region_bytes(plan.layout, rect, bit_depth);
    if (bytes == 0) {
      schedule.error = "tile region has no bytes";
      return schedule;
    }
    TransferOp op;
    op.phase = TransferPhase::kReconGather;
    op.direction = TransferDirection::kDeviceToHost;
    op.gpu_slot = slot;
    op.ordinal = ordinal_for(slot);
    op.tile_index = tile;
    op.bytes = bytes;
    op.stream_role = 1;
    op.depends_on = -1;
    schedule.ops.push_back(op);
    schedule.recon_gather_bytes += bytes;
  }

  for (int slot = 0; slot < gpu_count; ++slot) {
    for (int tile = 0; tile < tile_count; ++tile) {
      const tiles::TileRect& rect =
          plan.layout.tiles[static_cast<std::size_t>(tile)];
      const std::uint64_t bytes =
          tile_region_bytes(plan.layout, rect, bit_depth);
      TransferOp op;
      op.phase = TransferPhase::kReferenceFill;
      op.gpu_slot = slot;
      op.ordinal = ordinal_for(slot);
      op.tile_index = tile;
      op.bytes = bytes;
      op.stream_role = 1;
      op.depends_on = -1;
      if (owner[static_cast<std::size_t>(tile)] == slot) {
        op.direction = TransferDirection::kDeviceToDevice;
        schedule.reference_d2d_bytes += bytes;
      } else {
        op.direction = TransferDirection::kHostToDevice;
        schedule.reference_fill_bytes += bytes;
      }
      schedule.ops.push_back(op);
    }
  }

  schedule.host_tx_bytes = schedule.source_upload_bytes +
                           schedule.payload_bytes +
                           schedule.recon_gather_bytes +
                           schedule.reference_fill_bytes;
  if (plan.transfer.effective_gbps > 0.0) {
    schedule.estimated_ms = static_cast<double>(schedule.host_tx_bytes) /
                            (plan.transfer.effective_gbps * 1.0e9) * 1000.0;
  }
  schedule.valid = true;
  return schedule;
}

}  // namespace av2enc::multigpu::runtime
