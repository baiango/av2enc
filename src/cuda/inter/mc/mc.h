// AV2 §7.13.3 inter motion compensation — CUDA API.
//
// The kernel evaluates the same §7.13.3.18 arithmetic as the CPU reference
// (src/encoder/inter/mc/mc_core.h) and must be bit-exact with it.  One thread
// produces one output sample: the eight horizontal 8-tap sums are recomputed
// per sample instead of materialising the spec's intermediate array.  That is
// arithmetically identical (each horizontal tap sum is rounded with Round2(..,
// InterRound0) before the vertical accumulation) and needs no shared memory.
//
// No allocations, no hidden synchronisation, no global state.  The caller owns
// every device buffer.  Scope is single-reference 8-bit 4:2:0 translational MC
// with no reference scaling, matching the M2 baseline.

#ifndef AV2ENC_CUDA_INTER_MC_MC_H_
#define AV2ENC_CUDA_INTER_MC_MC_H_

#include <cstdint>

#include <cuda_runtime.h>

#include "../../../encoder/inter/mc/mc.h"

namespace av2enc::cuda {

// Device-resident reference frame, 8-bit 4:2:0.  Every pointer must be a
// device pointer.  Plane extents are derived from the MI grid:
//   luma:   mi_cols*4 x mi_rows*4
//   chroma: mi_cols*2 x mi_rows*2
// and each stride must be at least the plane width.
struct McDeviceFrame {
  const std::uint8_t* plane[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};
  int mi_cols = 0;
  int mi_rows = 0;
};

// Predicts one plane of one block.  Returns cudaErrorInvalidValue for bad
// geometry/pointers, otherwise the launch error from cudaGetLastError().
cudaError_t mc_predict_plane(const McDeviceFrame& ref, int plane,
                             const inter::mc::BlockRect& rect,
                             inter::mc::Mv mv,
                             inter::mc::InterpFilter filter, std::uint8_t* dst,
                             int dst_stride, cudaStream_t stream = nullptr);

// 4:2:0 convenience wrapper: one launch per plane (three asynchronous
// launches).  `luma_rect` must be even in x/y/w/h.
cudaError_t mc_predict_block_420(const McDeviceFrame& ref,
                                 const inter::mc::BlockRect& luma_rect,
                                 inter::mc::Mv mv,
                                 inter::mc::InterpFilter filter,
                                 std::uint8_t* dst[3], int dst_stride[3],
                                 cudaStream_t stream = nullptr);

// Batched MC job.  All pointers are device pointers.  `plane` is 0/1/2 and
// `rect` is in that plane's samples (for chroma, the caller passes the
// already-halved rectangle, as mc_predict_block_420 does).
struct McBlockJob {
  std::uint8_t* dst = nullptr;
  int dst_stride = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int plane = 0;
  inter::mc::Mv mv{};
  inter::mc::InterpFilter filter = inter::mc::InterpFilter::kEighttap;
};

// Predicts `count` independent blocks in one launch (one CUDA block per job).
// Use this in the encoder: a launch per prediction block is launch-bound at
// video scale.  `device_jobs` is a device array of count entries; the caller
// is responsible for job validity (device memory cannot be validated here).
cudaError_t mc_predict_batch(const McDeviceFrame& ref,
                             const McBlockJob* device_jobs, int count,
                             cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_INTER_MC_MC_H_
