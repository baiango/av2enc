// CUDA dequantization with AV2 quantizer matrices.
//
// Mirrors `av2enc::qm::dequantize_block` by calling the same
// `av2enc::qm::core::dequantize_block` integer helper.  There is no FP math
// here, so -fmad is irrelevant; the build still uses it for consistency with
// the other component tests.

#include "cuda/codec/transform/qm/qm_gpu.h"

#include "codec/transform/qm/qm_core.h"

namespace av2enc::cuda::qm {
namespace {

__device__ __forceinline__ bool valid_job(const Job& job) {
  if (!av2enc::qm::core::valid_tx_size(job.tx_size)) return false;
  if (job.plane < 0 || job.plane > 2) return false;
  if (job.seg_qm_level < 0 || job.seg_qm_level > 15) return false;
  if (job.qm_level < 0 || job.qm_level > 15) return false;
  if (job.dc_quant <= 0 || job.ac_quant <= 0) return false;
  if (job.bit_depth < 8 || job.bit_depth > 12) return false;
  return true;
}

__global__ void dequantize_kernel(Batch batch) {
  const std::int32_t job_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (job_index >= batch.num_jobs) return;
  if (batch.status != nullptr) batch.status[job_index] = -1;

  const Job job = batch.jobs[job_index];
  if (!valid_job(job)) return;

  av2enc::qm::core::CoeffParams params;
  params.tx_size = job.tx_size;
  params.plane = job.plane;
  params.tx_type = job.tx_type;
  params.seg_qm_level = job.seg_qm_level;
  params.qm_level = job.qm_level;
  params.dc_quant = job.dc_quant;
  params.ac_quant = job.ac_quant;
  params.bit_depth = job.bit_depth;
  params.using_qmatrix = job.using_qmatrix != 0;
  params.qm_data_present = job.qm_data_present != 0;
  params.allow_tcq = job.allow_tcq != 0;
  params.lossless = job.lossless != 0;
  params.use_fsc = job.use_fsc != 0;
  params.qtable = batch.qtable;
  params.user_qm = batch.user_qm;
  const std::int32_t* levels = batch.levels + job.level_offset;
  std::int32_t* dequant = batch.dequant + job.dequant_offset;
  av2enc::qm::core::dequantize_block(params, levels, dequant);
  if (batch.status != nullptr) batch.status[job_index] = 0;
}

}  // namespace

cudaError_t dequantize_batch(const Batch& batch, cudaStream_t stream) {
  if (batch.num_jobs <= 0) return cudaSuccess;
  if (batch.levels == nullptr || batch.dequant == nullptr ||
      batch.jobs == nullptr || batch.qtable == nullptr) {
    return cudaErrorInvalidValue;
  }
  constexpr int kThreads = 256;
  const int blocks = (batch.num_jobs + kThreads - 1) / kThreads;
  dequantize_kernel<<<blocks, kThreads, 0, stream>>>(batch);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::qm
