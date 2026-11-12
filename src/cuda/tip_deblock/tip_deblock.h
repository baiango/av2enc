// AV2 §7.16 Deblocking filter for TIP — CUDA API.
//
// The filter is strictly in-place on three device-resident planes.  It needs no
// per-launch scratch: §7.16 has no MI grid, no statistics pass and no
// dependencies outside the plane samples and the tile-start arrays.  Two
// kernel launches per frame (all planes, vertical boundaries; then all planes,
// horizontal boundaries) preserve the normative vertical-before-horizontal
// order per plane; the three planes are independent.
//
// Device memory policy (runtime contract):
//   * The kernels allocate nothing.  Plane and tile buffers are either
//     caller-owned device pointers or allocated once from the persistent
//     av2enc::cuda::Arena by tip_deblock_arena_frame_init().
//   * tip_deblock_required_device_bytes(config) reports the arena footprint
//     of one frame handle (three planes + tile-start arrays + scratch, which
//     is 0 bytes for this filter).
//   * tip_deblock_arena_frame_init() allocates the persistent buffers with
//     Arena::allocate_array and rewinds on failure; it reports
//     Status::kArenaExhausted (and the required size) when the arena is short.
//   * tip_deblock_frame_arena() brackets the launch with Arena::mark() /
//     Arena::rewind(); no allocation survives the call and repeated frames do
//     not grow the arena.
//
// Scope: 4:2:0 planar, 8- and 10-bit, even luma MI grid (frame padded to a
// multiple of 8).  No bitstream integration, no hidden synchronisation, no
// global state.
//
// Ordering relative to §7.17: §7.16 replaces the regular deblocking filter for
// TipFrameMode == TIP_FRAME_AS_OUTPUT and is followed by no CDEF/CCSO/LR.
// The CPU header (encoder/filters/tip_deblock/tip_deblock.h) owns the flag /
// quantizer contract; this layer consumes the same TipDeblockOptions POD.

#ifndef AV2ENC_CUDA_TIP_DEBLOCK_TIP_DEBLOCK_H_
#define AV2ENC_CUDA_TIP_DEBLOCK_TIP_DEBLOCK_H_

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "../arena.h"
#include "tip_deblock_device.h"

#include "encoder/filters/tip_deblock/tip_deblock.h"

namespace av2enc::cuda::tip_deblock {

using Options = av2enc::tip_deblock::TipDeblockOptions;

// Device-resident frame description.  Every pointer must be a device pointer.
// Plane extents follow the CPU rule:
//   luma:   mi_cols*4 x mi_rows*4
//   chroma: mi_cols*2 x mi_rows*2
// with 8-bit planes as uint8_t and 10-bit planes as little-endian uint16_t.
struct TipDeblockDeviceFrame {
  void* plane[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};  // samples
  int mi_cols = 0;
  int mi_rows = 0;
  int subsampling_x = 1;
  int subsampling_y = 1;
  // Device pointers to MiColStarts / MiRowStarts (luma MI units).  Contents
  // are device memory and are not read on the host.
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

// Maps the CPU options onto the kernel POD.  tip_size comes from the shared
// select_tip_size() so the device never re-derives it.
inline TipDeblockDeviceOptions tip_deblock_device_options(
    const Options& options) {
  TipDeblockDeviceOptions device;
  device.base_q_idx = options.base_q_idx;
  device.delta_q_u_ac = options.delta_q_u_ac;
  device.delta_q_v_ac = options.delta_q_v_ac;
  device.base_uv_ac_delta_q = options.base_uv_ac_delta_q;
  device.bit_depth = options.bit_depth;
  device.tip_size = static_cast<int>(
      av2enc::tip_deblock::tip_deblock_tip_size(options));
  device.disable_loopfilters_across_tiles =
      options.disable_loopfilters_across_tiles ? 1 : 0;
  return device;
}

// Builds the config used for arena sizing / standalone frame allocation.
// `bit_depth` comes from the options.  `num_tile_cols`/`num_tile_rows` size the
// tile-start arrays; the caller uploads the values afterwards.
inline TipDeblockConfig tip_deblock_config_from_options(const Options& options,
                                                        int mi_cols,
                                                        int mi_rows,
                                                        int num_tile_cols,
                                                        int num_tile_rows) {
  TipDeblockConfig config;
  config.mi_cols = mi_cols;
  config.mi_rows = mi_rows;
  config.subsampling_x = 1;
  config.subsampling_y = 1;
  config.bit_depth = options.bit_depth;
  config.num_tile_cols = num_tile_cols;
  config.num_tile_rows = num_tile_rows;
  return config;
}

// Applies §7.16 in place using caller-owned device buffers.  No allocation.
// Returns cudaErrorInvalidValue for unsupported geometry/bit depth, a missing
// plane/tile pointer, a short stride, an odd MI grid, an unsupported
// subsampling mode, an invalid tip size, or an apply flag that cannot be
// signalled.  Frames for which the process is not active are a successful
// no-op (matching the CPU kNotApplicable result).
cudaError_t tip_deblock_frame(const TipDeblockDeviceFrame& frame,
                              const Options& options,
                              cudaStream_t stream = nullptr);

// One plane/pass only (pass 0 = vertical boundaries, pass 1 = horizontal).
// Same no-op / validation contract as tip_deblock_frame().
cudaError_t tip_deblock_plane_pass(const TipDeblockDeviceFrame& frame,
                                   const Options& options, int plane, int pass,
                                   cudaStream_t stream = nullptr);

// Persistent arena-backed frame handle.
struct TipDeblockArenaFrame {
  TipDeblockDeviceFrame frame{};
  TipDeblockConfig config{};
  std::size_t required_bytes = 0;
  std::size_t persistent_bytes = 0;
  // Mutable device aliases so the caller can upload MiColStarts/MiRowStarts.
  std::uint16_t* tile_cols = nullptr;
  std::uint16_t* tile_rows = nullptr;
};

// Allocates the three planes and the tile-start arrays once from `arena`.
// On success `persistent_bytes` is the bump delta and `required_bytes` is
// tip_deblock_required_device_bytes(config).  On any failure the arena is
// rewound to its entry mark and *required_bytes (when non-null) is still
// written, so a caller that sees Status::kArenaExhausted can grow the arena
// and retry.
Status tip_deblock_arena_frame_init(Arena& arena,
                                    const TipDeblockConfig& config,
                                    TipDeblockArenaFrame* out,
                                    std::size_t* required_bytes = nullptr);

// Arena-bracketed frame/plane-pass entry points.  They reserve the scratch
// region (0 bytes today), launch, and always rewind to the entry mark.  The
// frame must come from tip_deblock_arena_frame_init() with the same config.
Status tip_deblock_frame_arena(TipDeblockArenaFrame& frame,
                               const Options& options, Arena& arena,
                               cudaStream_t stream = nullptr);

Status tip_deblock_plane_pass_arena(TipDeblockArenaFrame& frame,
                                    const Options& options, int plane, int pass,
                                    Arena& arena,
                                    cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::tip_deblock

#endif  // AV2ENC_CUDA_TIP_DEBLOCK_TIP_DEBLOCK_H_
