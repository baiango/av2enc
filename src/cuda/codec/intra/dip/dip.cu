// AV2 DIP CUDA predictor.  See dip.h and dip_device.cuh.

#include "dip.h"

#include "dip_device.cuh"

namespace av2enc::cuda::dip {
namespace {

constexpr int kThreads = 128;

__global__ void dip_predict_kernel(DipBlock block, DipParams params,
                                   std::uint8_t* pred) {
  __shared__ DipSharedScratch scratch;
  dip_run_block_device(block, params, pred, &scratch);
}

__global__ void dip_predict_batch_kernel(const DipJob* jobs) {
  __shared__ DipSharedScratch scratch;
  const DipJob job = jobs[blockIdx.x];
  dip_run_block_device(job.block, job.params, job.pred, &scratch);
}

bool valid_dim(int v) {
  return v == 4 || v == 8 || v == 16 || v == 32 || v == 64;
}

bool valid_block(const DipBlock& block, const DipParams& params) {
  if (block.recon == nullptr || block.recon_stride <= 0) return false;
  if (!valid_dim(block.w) || !valid_dim(block.h)) return false;
  if (block.w <= 4 || block.h <= 4 || block.w * block.h < 128) return false;
  if (block.x < 0 || block.y < 0 || block.max_x < block.x ||
      block.max_y < block.y) {
    return false;
  }
  if (block.recon_stride < block.max_x + 1) return false;
  if (block.have_left < 0 || block.have_above < 0) return false;
  if (block.have_left != 0 && block.x == 0) return false;
  if (block.have_above != 0 && block.y == 0) return false;
  if (block.num4_above_right < 0 || block.num4_below_left < 0) return false;
  if (params.mode < 0 || params.mode >= kNumModes) return false;
  if (params.transpose < 0 || params.transpose > 1) return false;
  return true;
}

}  // namespace

cudaError_t dip_predict(const DipBlock& block, const DipParams& params,
                        std::uint8_t* pred, cudaStream_t stream) {
  if (pred == nullptr || !valid_block(block, params)) {
    return cudaErrorInvalidValue;
  }
  dip_predict_kernel<<<1, kThreads, 0, stream>>>(block, params, pred);
  return cudaGetLastError();
}

cudaError_t dip_predict_batch(const DipJob* d_jobs, int job_count,
                              cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && d_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) return cudaSuccess;
  dip_predict_batch_kernel<<<job_count, kThreads, 0, stream>>>(d_jobs);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::dip
