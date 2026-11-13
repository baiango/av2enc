// Device body for the M2 video intra candidate generator.
//
// Included by `video_intra.cu` (real kernels) and by
// `video_intra_emulation_test.cpp` (host emulation, one thread).  The
// prediction itself is `intra_modes::intra_run_block_device()` from the shared
// `intra_predict_device.cuh`, i.e. exactly the code `intra_modes.cu` runs.
// This file only fills the descriptor, batches it, and reduces the SSE.
#ifndef AV2ENC_CUDA_PIPELINE_VIDEO_INTRA_VIDEO_INTRA_DEVICE_CUH_
#define AV2ENC_CUDA_PIPELINE_VIDEO_INTRA_VIDEO_INTRA_DEVICE_CUH_

#include "cuda/codec/intra/modes/intra_predict_device.cuh"
#include "cuda/pipeline/video/intra/video_intra.h"

namespace av2enc::cuda::video::intra {
namespace detail {

// One (block, candidate) pair in the device launch.  Built on the host by
// `Workspace::evaluate()`; the kernel index is the descriptor index.
struct CandidateDesc {
  std::int32_t job_index = 0;     // index into the staged BlockJob array
  std::int32_t result_index = 0;  // index into the CandidateResult array
  std::int32_t pred_offset = 0;   // element offset into the packed pred buffer
  std::int32_t mode = 0;          // intra_modes::IntraMode value
  std::int32_t angle_delta = 0;   // -3..3 for directional modes
};

// Threads per candidate.  The shared predictor is written for a caller-chosen
// block size; 128 matches `intra_modes`' batch kernel.
inline constexpr int kCandidateThreads = 128;

// Predicts one candidate into `pred[desc.pred_offset ..]`, optionally writes
// the coded-size residual (source - pred) for the TX stage, and reduces the SSE
// against the (clamped) source.
__device__ __forceinline__ void run_candidate(
    const BlockJob* __restrict__ jobs,
    const CandidateDesc* __restrict__ descs, int desc_index,
    std::uint8_t* __restrict__ pred, std::int16_t* __restrict__ residual,
    CandidateResult* __restrict__ results) {
  const CandidateDesc desc = descs[desc_index];
  const BlockJob job = jobs[desc.job_index];

  av2enc::cuda::intra_modes::IntraBlock block{};
  block.recon_plane = job.recon_plane;
  block.recon_stride = job.recon_stride;
  block.x0 = job.x;
  block.y0 = job.y;
  block.width = job.w;
  block.height = job.h;
  block.max_x = job.max_x;
  block.max_y = job.max_y;
  block.have_left = job.have_left;
  block.have_above = job.have_above;
  block.num4_above_right = job.num4_above_right;
  block.num4_below_left = job.num4_below_left;
  block.plane = job.plane;
  block.mode =
      static_cast<av2enc::cuda::intra_modes::IntraMode>(desc.mode);
  block.angle_delta = desc.angle_delta;
  block.filter_type_above = job.filter_type_above;
  block.filter_type_left = job.filter_type_left;
  block.enable_edge_filter = job.enable_edge_filter;
  block.source_plane = job.source_plane;
  block.source_stride = job.source_stride;
  block.source_max_x = job.source_max_x;
  block.source_max_y = job.source_max_y;

  std::uint8_t* out = pred + desc.pred_offset;
  std::int16_t* resid =
      residual != nullptr ? residual + desc.pred_offset : nullptr;

  __shared__ av2enc::cuda::intra_modes::SharedScratch scratch;
  av2enc::cuda::intra_modes::intra_run_block_device(block, out, resid,
                                                    &scratch);

  // SSE reduction.  Reading `out` back (instead of duplicating the predictor's
  // sample expression) keeps one source of truth for the prediction.
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  __shared__ long long partial[kCandidateThreads];
  const int n = job.w * job.h;
  const std::size_t source_stride =
      static_cast<std::size_t>(job.source_stride);
  long long local = 0;
  for (int i = tid; i < n; i += nthr) {
    const int px = i % job.w;
    const int py = i / job.w;
    int sx = job.x + px;
    int sy = job.y + py;
    if (sx > job.source_max_x) sx = job.source_max_x;
    if (sx < 0) sx = 0;
    if (sy > job.source_max_y) sy = job.source_max_y;
    if (sy < 0) sy = 0;
    const int sample =
        static_cast<int>(job.source_plane[static_cast<std::size_t>(sy) *
                                              source_stride +
                                          static_cast<std::size_t>(sx)]);
    const int diff = sample - static_cast<int>(out[i]);
    local += static_cast<long long>(diff) * diff;
  }
  partial[tid] = local;
  __syncthreads();

  if (tid == 0) {
    long long total = 0;
    for (int i = 0; i < nthr; ++i) {
      total += partial[i];
    }
    CandidateResult result{};
    result.sse = total;
    result.pred_offset = desc.pred_offset;
    result.block = desc.job_index;
    result.mode = static_cast<std::uint8_t>(desc.mode);
    result.angle_delta = static_cast<std::int8_t>(desc.angle_delta);
    result.plane = static_cast<std::uint8_t>(job.plane);
    results[desc.result_index] = result;
  }
}

}  // namespace detail
}  // namespace av2enc::cuda::video::intra

#endif  // AV2ENC_CUDA_PIPELINE_VIDEO_INTRA_VIDEO_INTRA_DEVICE_CUH_
