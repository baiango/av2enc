// AV2 §7.18 CDEF — CUDA device API.
//
// Device twin of the CPU reference in src/encoder/filters/cdef/.  The kernel
// calls the same host/device core functions (cdef_core.h), so the output is
// bit-exact with the CPU for identical input planes and metadata.
//
// Scope: 8-bit 4:2:0, MI-aligned planes, separate source and destination
// buffers.  No allocations, no hidden synchronisation, no global state.

#ifndef AV2ENC_CUDA_CDEF_CDEF_H_
#define AV2ENC_CUDA_CDEF_CDEF_H_

#include <cuda_runtime.h>

#include <cstdint>

#include "codec/filters/cdef/cdef.h"

namespace av2enc::cuda {

// Device-resident CDEF inputs.  All grids are in luma MI units; the tile and
// metadata arrays must already be on the device.  `src` is the unfiltered
// CurrFrame and `dst` the CdefFrame; they must not alias.
struct CdefDeviceFrame {
  const std::uint8_t* src[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* dst[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};

  int mi_cols = 0;
  int mi_rows = 0;
  int subsampling_x = 1;
  int subsampling_y = 1;

  const std::uint8_t* skip = nullptr;
  const std::uint8_t* lossless = nullptr;
  const std::uint8_t* lossless_chroma = nullptr;

  const std::int8_t* cdef_idx = nullptr;
  int n64_cols = 0;
  int n64_rows = 0;

  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

// Validates host-side geometry/options/pointers and launches one kernel over
// all 8x8 CDEF blocks.  Returns cudaErrorInvalidValue for bad arguments.
cudaError_t cdef_frame(const CdefDeviceFrame& frame,
                       const av2enc::cdef::CdefOptions& options,
                       cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_CDEF_CDEF_H_
