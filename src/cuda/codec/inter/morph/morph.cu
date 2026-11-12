// CUDA kernels for the AV2 morphological prediction process.
//
// One CUDA block per predicted block. Thread 0 performs the sequential
// template fit; the barrier, then all threads apply the adjustment and write
// the optional per-sample field. No allocation, no host round trip, and no
// synchronization other than the single barrier.

#include "cuda/codec/inter/morph/morph.h"

#include "cuda/codec/inter/morph/morph_device.cuh"

namespace av2enc::cuda::morph {
namespace {

constexpr int kThreads = 128;

__global__ void morph_predict_kernel(MorphJob job) {
  morph_run_job_device(job, nullptr, nullptr);
}

__global__ void morph_batch_kernel(const MorphJob* __restrict__ jobs,
                                   MorphFit* __restrict__ fits,
                                   int* __restrict__ statuses, int job_count) {
  const int index = static_cast<int>(blockIdx.x);
  if (index >= job_count) {
    return;
  }
  const MorphJob job = jobs[index];
  morph_run_job_device(job, fits == nullptr ? nullptr : fits + index,
                       statuses == nullptr ? nullptr : statuses + index);
}

}  // namespace

cudaError_t morph_predict(const MorphJob& job, cudaStream_t stream) {
  morph_predict_kernel<<<1, kThreads, 0, stream>>>(job);
  return cudaGetLastError();
}

cudaError_t morph_predict_batch_persist(const MorphJob* d_jobs,
                                        MorphFit* d_fits, int* d_statuses,
                                        int job_count, cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && d_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) {
    return cudaSuccess;
  }
  morph_batch_kernel<<<job_count, kThreads, 0, stream>>>(
      d_jobs, d_fits, d_statuses, job_count);
  return cudaGetLastError();
}

cudaError_t morph_predict_batch(const MorphJob* d_jobs, int job_count,
                                cudaStream_t stream) {
  return morph_predict_batch_persist(d_jobs, nullptr, nullptr, job_count,
                                     stream);
}

}  // namespace av2enc::cuda::morph
