// CUDA MHCCP kernel (§7.13.6-§7.13.8).
//
// Integer-only, shared with the CPU reference through mhccp_common.h. The
// kernel body differs from the reference only in that thread 0 prepares the
// per-block CflRef/parameters and all threads evaluate one sample each.

#include "mhccp_gpu.h"

#include "mhccp_device.cuh"

namespace av2enc::cuda::mhccp {
namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ void run_job(const Block& block,
                                        const MultiParams& explicit_params,
                                        int cfl_mh_dir, bool derived,
                                        MultiParams* params_out) {
  __shared__ av2enc::cfl::mhccp::Refs refs;
  __shared__ MultiParams params;
  __shared__ int ok;

  if (threadIdx.x == 0) {
    ok = av2enc::cfl::mhccp::device::prepare(
             block, cfl_mh_dir, derived, explicit_params, &refs, &params)
             ? 1
             : 0;
    if (ok != 0 && params_out != nullptr) {
      *params_out = params;
    }
  }
  __syncthreads();
  if (ok == 0) {
    return;
  }

  const int samples = block.w * block.h;
  for (int index = static_cast<int>(threadIdx.x); index < samples;
       index += static_cast<int>(blockDim.x)) {
    const int i = index / block.w;
    const int j = index % block.w;
    const int value = av2enc::cfl::mhccp::device::sample(
        block, refs, params, cfl_mh_dir, i, j);
    const std::size_t offset =
        static_cast<std::size_t>(block.y + i) *
            static_cast<std::size_t>(block.chroma_stride) +
        static_cast<std::size_t>(block.x + j);
    block.chroma[offset] = static_cast<std::uint8_t>(value);
  }
}

__global__ void predict_kernel(Block block, MultiParams params, int cfl_mh_dir,
                               int derived, MultiParams* params_out) {
  run_job(block, params, cfl_mh_dir, derived != 0, params_out);
}

__global__ void predict_batch_kernel(const Job* jobs, MultiParams* params_out) {
  const int job = static_cast<int>(blockIdx.x);
  const Job& item = jobs[job];
  run_job(item.block, item.params, item.cfl_mh_dir, item.derived != 0,
          params_out != nullptr ? &params_out[job] : nullptr);
}

}  // namespace

cudaError_t mhccp_predict(const Block& block, const MultiParams& params,
                          int cfl_mh_dir, bool derived,
                          MultiParams* d_params_out, cudaStream_t stream) {
  predict_kernel<<<1, kThreads, 0, stream>>>(block, params, cfl_mh_dir,
                                             derived ? 1 : 0, d_params_out);
  return cudaGetLastError();
}

cudaError_t mhccp_predict_batch(const Job* d_jobs, int job_count,
                                MultiParams* d_params_out,
                                cudaStream_t stream) {
  if (job_count <= 0) {
    return cudaSuccess;
  }
  if (d_jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  predict_batch_kernel<<<job_count, kThreads, 0, stream>>>(d_jobs,
                                                           d_params_out);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::mhccp
