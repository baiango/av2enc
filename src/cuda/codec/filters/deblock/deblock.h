// AV2 §7.17 deblocking filter — CUDA API.
//
// The kernel calls the exact same host/device core as the CPU reference
// (src/encoder/filters/deblock/deblock_core.h), so the device output is
// bit-exact with the CPU given identical plane contents and metadata.
//
// No allocations, no hidden synchronisation, no global state.  The caller owns
// all device buffers.  Frames must be 8-bit 4:2:0 with an even luma MI grid
// (frame dimensions padded to a multiple of 8).

#ifndef AV2ENC_CUDA_DEBLOCK_DEBLOCK_H_
#define AV2ENC_CUDA_DEBLOCK_DEBLOCK_H_

#include <cstdint>

#include <cuda_runtime.h>

#include "../../../../codec/filters/deblock/deblock.h"

namespace av2enc::cuda {

// Device-resident frame description.  Every pointer in this struct must be a
// device pointer.  Plane extents are derived from mi_cols/mi_rows:
//   luma:   mi_cols*4      x mi_rows*4
//   chroma: mi_cols*2      x mi_rows*2
// and `stride` must be at least the plane width.
struct DeblockDeviceFrame {
  std::uint8_t* plane[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};
  const deblock::DeblockMi* mi[3] = {nullptr, nullptr, nullptr};
  int mi_cols = 0;
  int mi_rows = 0;
  int subsampling_x = 1;
  int subsampling_y = 1;
  // Device pointers to MiColStarts / MiRowStarts (luma MI units).  May be null
  // when the corresponding count is zero.  Contents are not validated on the
  // host (they are device memory); the caller must keep them in range.
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

// Applies the deblocking filter in place.  Two kernel launches (vertical edges,
// then horizontal edges), no synchronisation.  Returns
// cudaErrorInvalidValue for bad geometry, or the launch error from
// cudaGetLastError().
cudaError_t deblock_frame(const DeblockDeviceFrame& frame,
                          const deblock::DeblockOptions& options,
                          cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_DEBLOCK_DEBLOCK_H_
