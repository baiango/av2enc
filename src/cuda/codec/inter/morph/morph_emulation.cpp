// CPU emulation of the morph CUDA entry points, for the no-nvcc parity test.
//
// This is not the GPU path. It forwards each launch to the shared
// morph_device.cuh implementation with blockDim = 1, which validates the
// arithmetic, the shared-memory broadcast and the API/arena plumbing. The real
// bit-exactness gate is build_gpu_test.sh on the sm_86 box, which links
// morph.cu instead of this file.

#include "cuda_emulation_prefix.h"

#include "cuda/codec/inter/morph/morph_device.cuh"

namespace av2enc::cuda::morph {
namespace {

void run_one(const MorphJob& job) {
  threadIdx.x = 0;
  blockDim.x = 1;
  morph_run_job_device(job, nullptr, nullptr);
}

}  // namespace

cudaError_t morph_predict(const MorphJob& job, cudaStream_t /*stream*/) {
  run_one(job);
  return cudaSuccess;
}

cudaError_t morph_predict_batch_persist(const MorphJob* d_jobs,
                                        MorphFit* d_fits, int* d_statuses,
                                        int job_count, cudaStream_t /*stream*/) {
  if (job_count < 0 || (job_count > 0 && d_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  threadIdx.x = 0;
  blockDim.x = 1;
  for (int i = 0; i < job_count; ++i) {
    const MorphJob job = d_jobs[i];
    morph_run_job_device(job, d_fits == nullptr ? nullptr : d_fits + i,
                         d_statuses == nullptr ? nullptr : d_statuses + i);
  }
  return cudaSuccess;
}

cudaError_t morph_predict_batch(const MorphJob* d_jobs, int job_count,
                                cudaStream_t stream) {
  return morph_predict_batch_persist(d_jobs, nullptr, nullptr, job_count,
                                     stream);
}

}  // namespace av2enc::cuda::morph
