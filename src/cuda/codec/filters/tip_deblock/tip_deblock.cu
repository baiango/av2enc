// AV2 §7.16 TIP deblocking filter — CUDA launch and arena plumbing.
//
// The arithmetic is entirely in tip_deblock_device.h.  This file only maps
// edges onto threads, validates the device frame, and manages the persistent
// arena.  There is no scratch pass: §7.16 is in-place, so the two launches
// operate directly on the caller's planes.

#include "tip_deblock.h"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::tip_deblock {
namespace {

static_assert(kTipArenaAlignment == Arena::kAlignment,
              "device plan and arena alignment must agree");

constexpr int kThreads = 256;

// Per-plane edge counts for one pass.  Each thread handles one §7.16 edge,
// i.e. one 4-sample segment:
//   pass 0 (vertical):   x = k*sw (k >= 1), y = 4*m
//   pass 1 (horizontal): x = 4*m,         y = k*sh (k >= 1)
struct PlaneGrid {
  int cols = 0;
  int rows = 0;
};

PlaneGrid plane_grid(const TipDeblockDeviceFrame& frame, int plane,
                     int tip_size, int pass) {
  const int sub_x = (plane == 0) ? 0 : frame.subsampling_x;
  const int sub_y = (plane == 0) ? 0 : frame.subsampling_y;
  const int w = (frame.mi_cols * kTipMiSize) >> sub_x;
  const int h = (frame.mi_rows * kTipMiSize) >> sub_y;
  PlaneGrid grid;
  if (pass == 0) {
    const int sw = tip_size >> sub_x;
    grid.cols = (sw > 0) ? (w - 1) / sw : 0;
    grid.rows = (h + 3) / 4;
  } else {
    const int sh = tip_size >> sub_y;
    grid.cols = (w + 3) / 4;
    grid.rows = (sh > 0) ? (h - 1) / sh : 0;
  }
  if (grid.cols < 0) {
    grid.cols = 0;
  }
  if (grid.rows < 0) {
    grid.rows = 0;
  }
  return grid;
}

template <typename Sample, int BitDepth>
__global__ void tip_pass_kernel(TipDeblockFrameView<Sample> frame,
                                TipDeblockDeviceOptions options, int pass,
                                int count0, int count1, int count2, int cols0,
                                int cols1, int cols2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int plane = 0;
  int local = 0;
  int cols = 0;
  if (idx < count0) {
    plane = 0;
    local = idx;
    cols = cols0;
  } else if (idx < count0 + count1) {
    plane = 1;
    local = idx - count0;
    cols = cols1;
  } else if (idx < count0 + count1 + count2) {
    plane = 2;
    local = idx - count0 - count1;
    cols = cols2;
  } else {
    return;
  }
  if (cols <= 0) {
    return;
  }

  const int sub_x = (plane == 0) ? 0 : frame.sub_x;
  const int sub_y = (plane == 0) ? 0 : frame.sub_y;
  int x = 0;
  int y = 0;
  if (pass == 0) {
    const int sw = options.tip_size >> sub_x;
    x = (local % cols + 1) * sw;
    y = (local / cols) * 4;
  } else {
    const int sh = options.tip_size >> sub_y;
    x = (local % cols) * 4;
    y = (local / cols + 1) * sh;
  }
  (void)tip_edge_view<true, Sample, BitDepth>(frame, options, plane, pass, x,
                                              y);
}

template <typename Sample>
TipDeblockFrameView<Sample> make_view(const TipDeblockDeviceFrame& frame) {
  TipDeblockFrameView<Sample> view{};
  for (int plane = 0; plane < 3; ++plane) {
    view.plane[plane] = static_cast<Sample*>(frame.plane[plane]);
    view.stride[plane] = frame.stride[plane];
  }
  view.mi_cols = frame.mi_cols;
  view.mi_rows = frame.mi_rows;
  view.sub_x = frame.subsampling_x;
  view.sub_y = frame.subsampling_y;
  view.tile_col_starts = frame.tile_col_starts;
  view.num_tile_cols = frame.num_tile_cols;
  view.tile_row_starts = frame.tile_row_starts;
  view.num_tile_rows = frame.num_tile_rows;
  return view;
}

bool validate_config(const TipDeblockConfig& config) {
  if (config.bit_depth != 8 && config.bit_depth != 10) {
    return false;
  }
  if (config.subsampling_x != 1 || config.subsampling_y != 1) {
    return false;
  }
  if (config.mi_cols <= 0 || config.mi_rows <= 0) {
    return false;
  }
  if ((config.mi_cols & 1) != 0 || (config.mi_rows & 1) != 0) {
    return false;
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (tip_deblock_plane_height(config, plane) <= 0 ||
        tip_deblock_plane_width(config, plane) <= 0) {
      return false;
    }
    if (config.stride[plane] != 0 &&
        config.stride[plane] < tip_deblock_plane_width(config, plane)) {
      return false;
    }
  }
  if (config.num_tile_cols < 0 || config.num_tile_rows < 0) {
    return false;
  }
  return true;
}

bool validate_frame(const TipDeblockDeviceFrame& frame,
                    const Options& options) {
  if (options.bit_depth != 8 && options.bit_depth != 10) {
    return false;
  }
  if (frame.subsampling_x != 1 || frame.subsampling_y != 1) {
    return false;
  }
  if (frame.mi_cols <= 0 || frame.mi_rows <= 0) {
    return false;
  }
  // Even MI grid: keeps 4:2:0 chroma integral and aligns the literal 4-sample
  // groups with the plane extent.
  if ((frame.mi_cols & 1) != 0 || (frame.mi_rows & 1) != 0) {
    return false;
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (frame.plane[plane] == nullptr) {
      return false;
    }
    const int sub_x = (plane == 0) ? 0 : frame.subsampling_x;
    const int width = (frame.mi_cols * kTipMiSize) >> sub_x;
    if (frame.stride[plane] < width) {
      return false;
    }
  }
  if ((frame.num_tile_cols > 0 && frame.tile_col_starts == nullptr) ||
      (frame.num_tile_rows > 0 && frame.tile_row_starts == nullptr)) {
    return false;
  }
  if (frame.num_tile_cols < 0 || frame.num_tile_rows < 0) {
    return false;
  }
  const int tip_size = static_cast<int>(
      av2enc::tip_deblock::tip_deblock_tip_size(options));
  if (tip_size != 8 && tip_size != 16) {
    return false;
  }
  // apply_deblocking_filter_tip is codable only for TIP_FRAME_AS_OUTPUT with
  // enable_df_sub_pu && allow_df_sub_pu.
  if (options.apply_deblocking_filter_tip &&
      !(av2enc::tip_deblock::tip_deblock_flag_signallable(options) &&
        options.allow_df_sub_pu)) {
    return false;
  }
  return true;
}

template <typename Sample, int BitDepth>
cudaError_t launch_passes(const TipDeblockDeviceFrame& frame,
                          const Options& options, int only_plane, int only_pass,
                          cudaStream_t stream) {
  const TipDeblockDeviceOptions device_options =
      tip_deblock_device_options(options);
  const TipDeblockFrameView<Sample> view = make_view<Sample>(frame);
  const int first_pass = (only_pass < 0) ? 0 : only_pass;
  const int last_pass = (only_pass < 0) ? 1 : only_pass;
  for (int pass = first_pass; pass <= last_pass; ++pass) {
    int counts[3] = {0, 0, 0};
    int cols[3] = {0, 0, 0};
    int total = 0;
    for (int plane = 0; plane < 3; ++plane) {
      if (only_plane >= 0 && plane != only_plane) {
        continue;
      }
      const PlaneGrid grid =
          plane_grid(frame, plane, device_options.tip_size, pass);
      cols[plane] = grid.cols;
      counts[plane] = grid.cols * grid.rows;
      total += counts[plane];
    }
    if (total <= 0) {
      continue;
    }
    const int blocks = (total + kThreads - 1) / kThreads;
    tip_pass_kernel<Sample, BitDepth><<<blocks, kThreads, 0, stream>>>(
        view, device_options, pass, counts[0], counts[1], counts[2], cols[0],
        cols[1], cols[2]);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
  }
  return cudaSuccess;
}

cudaError_t launch_frame_typed(const TipDeblockDeviceFrame& frame,
                               const Options& options, int only_plane,
                               int only_pass, cudaStream_t stream) {
  if (options.bit_depth == 8) {
    return launch_passes<std::uint8_t, 8>(frame, options, only_plane,
                                          only_pass, stream);
  }
  return launch_passes<std::uint16_t, 10>(frame, options, only_plane, only_pass,
                                          stream);
}

}  // namespace

cudaError_t tip_deblock_frame(const TipDeblockDeviceFrame& frame,
                              const Options& options, cudaStream_t stream) {
  if (!validate_frame(frame, options)) {
    return cudaErrorInvalidValue;
  }
  if (!av2enc::tip_deblock::tip_deblock_is_active(options)) {
    return cudaSuccess;
  }
  return launch_frame_typed(frame, options, -1, -1, stream);
}

cudaError_t tip_deblock_plane_pass(const TipDeblockDeviceFrame& frame,
                                   const Options& options, int plane, int pass,
                                   cudaStream_t stream) {
  if (plane < 0 || plane > 2 || pass < 0 || pass > 1) {
    return cudaErrorInvalidValue;
  }
  if (!validate_frame(frame, options)) {
    return cudaErrorInvalidValue;
  }
  if (!av2enc::tip_deblock::tip_deblock_is_active(options)) {
    return cudaSuccess;
  }
  return launch_frame_typed(frame, options, plane, pass, stream);
}

Status tip_deblock_arena_frame_init(Arena& arena,
                                    const TipDeblockConfig& config,
                                    TipDeblockArenaFrame* out,
                                    std::size_t* required_bytes) {
  if (out == nullptr) {
    return Status::kInvalidArgument;
  }
  *out = TipDeblockArenaFrame{};
  out->config = config;
  out->required_bytes = tip_deblock_required_device_bytes(config);
  if (required_bytes != nullptr) {
    *required_bytes = out->required_bytes;
  }
  if (!validate_config(config)) {
    return Status::kInvalidArgument;
  }
  if (!arena.initialized()) {
    return Status::kNotInitialized;
  }

  const std::uint64_t start = arena.mark();
  for (int plane = 0; plane < 3; ++plane) {
    const std::size_t bytes = tip_deblock_plane_bytes(config, plane);
    std::uint8_t* memory = arena.allocate_array<std::uint8_t>(bytes);
    if (memory == nullptr) {
      (void)arena.rewind(start);
      return Status::kArenaExhausted;
    }
    out->frame.plane[plane] = memory;
    out->frame.stride[plane] = tip_deblock_plane_stride(config, plane);
  }
  if (config.num_tile_cols > 0) {
    out->tile_cols = arena.allocate_array<std::uint16_t>(
        static_cast<std::size_t>(config.num_tile_cols));
    if (out->tile_cols == nullptr) {
      (void)arena.rewind(start);
      return Status::kArenaExhausted;
    }
  }
  if (config.num_tile_rows > 0) {
    out->tile_rows = arena.allocate_array<std::uint16_t>(
        static_cast<std::size_t>(config.num_tile_rows));
    if (out->tile_rows == nullptr) {
      (void)arena.rewind(start);
      return Status::kArenaExhausted;
    }
  }
  out->frame.mi_cols = config.mi_cols;
  out->frame.mi_rows = config.mi_rows;
  out->frame.subsampling_x = config.subsampling_x;
  out->frame.subsampling_y = config.subsampling_y;
  out->frame.tile_col_starts = out->tile_cols;
  out->frame.num_tile_cols = config.num_tile_cols;
  out->frame.tile_row_starts = out->tile_rows;
  out->frame.num_tile_rows = config.num_tile_rows;
  out->persistent_bytes =
      static_cast<std::size_t>(arena.used() - start);
  return Status::kOk;
}

namespace {

Status run_arena_bracketed(TipDeblockArenaFrame& frame, const Options& options,
                           int only_plane, int only_pass, Arena& arena,
                           cudaStream_t stream) {
  if (!validate_frame(frame.frame, options)) {
    return Status::kInvalidArgument;
  }
  if (frame.config.bit_depth != options.bit_depth) {
    return Status::kInvalidArgument;
  }
  // The filter is in-place, so scratch is 0 bytes; the bracket still reserves
  // and releases it so the arena cannot silently drift if that changes.
  const std::uint64_t mark = arena.mark();
  const std::size_t scratch = tip_deblock_scratch_bytes(frame.config);
  if (scratch > 0) {
    void* scratch_memory = nullptr;
    const Status reserved = arena.allocate_checked(scratch, &scratch_memory);
    if (reserved != Status::kOk) {
      (void)arena.rewind(mark);
      return reserved;
    }
    (void)scratch_memory;
  }
  if (!av2enc::tip_deblock::tip_deblock_is_active(options)) {
    (void)arena.rewind(mark);
    return Status::kOk;
  }
  const cudaError_t error =
      launch_frame_typed(frame.frame, options, only_plane, only_pass, stream);
  (void)arena.rewind(mark);
  return status_from_cuda(error);
}

}  // namespace

Status tip_deblock_frame_arena(TipDeblockArenaFrame& frame,
                               const Options& options, Arena& arena,
                               cudaStream_t stream) {
  return run_arena_bracketed(frame, options, -1, -1, arena, stream);
}

Status tip_deblock_plane_pass_arena(TipDeblockArenaFrame& frame,
                                    const Options& options, int plane, int pass,
                                    Arena& arena, cudaStream_t stream) {
  if (plane < 0 || plane > 2 || pass < 0 || pass > 1) {
    return Status::kInvalidArgument;
  }
  return run_arena_bracketed(frame, options, plane, pass, arena, stream);
}

}  // namespace av2enc::cuda::tip_deblock
