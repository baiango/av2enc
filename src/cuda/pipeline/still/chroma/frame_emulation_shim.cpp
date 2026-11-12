// Host-emulation shim for still_chroma_test.cu.
//
// The real module launches CUDA kernels. This shim provides the same
// chroma_decide / chroma_decide_batch symbols and runs the exact device search
// body (still_chroma_device.cuh) with one emulated thread. It lets the full
// real-frame CPU-vs-GPU comparison run on a machine without nvcc.
//
// Test-only: it is not part of the GPU build.

#include <cstdint>

#define __device__
#define __host__
#define __forceinline__ inline

struct EmuDim {
  unsigned x = 0;
  unsigned y = 0;
  unsigned z = 0;
};
static EmuDim threadIdx;
static EmuDim blockIdx;
static EmuDim blockDim;
static inline void __syncthreads() {}
// One emulated lane: every shuffle source outside lane 0 contributes nothing.
static inline long long __shfl_down_sync(unsigned, long long, int) { return 0; }

#include "cuda/pipeline/still/chroma/still_chroma_device.cuh"

namespace av2enc::cuda::still {

cudaError_t chroma_decide(const StillChromaJob& job, const StillChromaOptions& options,
                          StillChromaResult* d_result, cudaStream_t) {
  if (d_result == nullptr) return cudaErrorInvalidValue;
  // Match the production single-block validation closely enough for the
  // policy checks in still_chroma_test.cu.
  if (job.w < 4 || job.h < 4 || job.w > 64 || job.h > 64 ||
      (job.w & (job.w - 1)) != 0 || (job.h & (job.h - 1)) != 0) {
    return cudaErrorInvalidValue;
  }
  if (options.enable_uv_full != 0 || options.enable_uv_directional != 0) {
    if ((job.filter_type_above & ~1) != 0 ||
        (job.filter_type_left & ~1) != 0 ||
        (job.enable_edge_filter & ~1) != 0 ||
        job.y_mode < 0 || job.y_mode >= 13 || job.y_angle_delta < -3 ||
        job.y_angle_delta > 3) {
      return cudaErrorInvalidValue;
    }
  }
  blockDim.x = 1;
  threadIdx.x = 0;
  static ChromaUvShared uv;
  static ChromaDcCflShared dc;
  if (options.enable_uv_full != 0 || options.enable_uv_directional != 0) {
    chroma_uv_full_search(job, options, &uv, d_result);
  } else {
    chroma_dc_cfl_search(job, options, 0, 1, &dc, d_result);
  }
  return cudaSuccess;
}

cudaError_t chroma_decide_batch(const StillChromaJob* d_jobs, int job_count,
                                const StillChromaOptions& options,
                                StillChromaResult* d_results,
                                cudaStream_t stream) {
  if (job_count <= 0) return cudaSuccess;
  if (d_jobs == nullptr || d_results == nullptr) return cudaErrorInvalidValue;
  for (int i = 0; i < job_count; ++i) {
    const cudaError_t status =
        chroma_decide(d_jobs[i], options, d_results + i, stream);
    if (status != cudaSuccess) return status;
  }
  return cudaSuccess;
}

}  // namespace av2enc::cuda::still
