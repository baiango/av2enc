// Kernels and launchers for the §7.14.4 DC/AC delta-Q quantizer.
//
// One thread per transform block; each thread derives the plane's DC/AC
// quantizers, validates its q2 matrix, and runs the exact integer/floating
// arithmetic in `deltaq_device.cuh`.

#include "cuda/codec/transform/delta_q/deltaq_gpu.h"

#include "cuda/codec/transform/delta_q/deltaq_device.cuh"

namespace av2enc::cuda::deltaq {
namespace {

__device__ __forceinline__ bool job_ready(
    const Job& job, const std::int32_t* batch_q2, int* dc_quant,
    int* ac_quant) {
  if (!core::valid_job(job)) return false;
  core::plane_quant(job, dc_quant, ac_quant);
  return core::q2_ready(job, batch_q2, *dc_quant, *ac_quant);
}

__global__ void quantize_kernel(Batch batch) {
  const std::int32_t job_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (job_index >= batch.num_jobs) return;
  const Job job = batch.jobs[job_index];
  int dc_quant = 0;
  int ac_quant = 0;
  if (!job_ready(job, batch.q2, &dc_quant, &ac_quant)) {
    if (batch.status != nullptr) batch.status[job_index] = -1;
    return;
  }
  core::quantize_job(job, batch.coeff + job.in_offset,
                     batch.out + job.out_offset, batch.q2);
  if (batch.status != nullptr) batch.status[job_index] = 0;
}

__global__ void dequantize_kernel(Batch batch) {
  const std::int32_t job_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (job_index >= batch.num_jobs) return;
  const Job job = batch.jobs[job_index];
  int dc_quant = 0;
  int ac_quant = 0;
  if (!job_ready(job, batch.q2, &dc_quant, &ac_quant)) {
    if (batch.status != nullptr) batch.status[job_index] = -1;
    return;
  }
  core::dequantize_job(job, batch.levels + job.in_offset,
                       batch.out + job.out_offset, batch.q2);
  if (batch.status != nullptr) batch.status[job_index] = 0;
}

constexpr int kThreads = 256;

}  // namespace

cudaError_t quantize_batch(const Batch& batch, cudaStream_t stream) {
  if (batch.num_jobs <= 0) return cudaSuccess;
  if (batch.coeff == nullptr || batch.out == nullptr ||
      batch.jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (batch.num_jobs + kThreads - 1) / kThreads;
  quantize_kernel<<<blocks, kThreads, 0, stream>>>(batch);
  return cudaGetLastError();
}

cudaError_t dequantize_batch(const Batch& batch, cudaStream_t stream) {
  if (batch.num_jobs <= 0) return cudaSuccess;
  if (batch.levels == nullptr || batch.out == nullptr ||
      batch.jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (batch.num_jobs + kThreads - 1) / kThreads;
  dequantize_kernel<<<blocks, kThreads, 0, stream>>>(batch);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::deltaq
