// AV2 MRL CUDA predictor. See mrl.h and mrl_device.cuh.

#include "mrl.h"

#include "mrl_device.cuh"

namespace av2enc::cuda::mrl {
namespace {

constexpr int kThreads = 128;

__global__ void mrl_predict_kernel(MrlBlock block, MrlParams params,
                                   std::uint8_t* pred) {
  __shared__ MrlSharedScratch scratch;
  mrl_run_block_device(block, params, pred, &scratch);
}

__global__ void mrl_predict_batch_kernel(const MrlJob* jobs) {
  __shared__ MrlSharedScratch scratch;
  const MrlJob job = jobs[blockIdx.x];
  mrl_run_block_device(job.block, job.params, job.pred, &scratch);
}

bool valid_dim(int v) {
  return v == 4 || v == 8 || v == 16 || v == 32 || v == 64;
}

bool valid_block(const MrlBlock& b, const MrlParams& p) {
  if (b.recon == nullptr || b.recon_stride <= 0) return false;
  if (!valid_dim(b.w) || !valid_dim(b.h)) return false;
  if (b.x < 0 || b.y < 0 || b.max_x < b.x || b.max_y < b.y) return false;
  if (b.recon_stride < b.max_x + 1) return false;
  if (b.have_left < 0 || b.have_above < 0) return false;
  if (b.have_left != 0 && b.x == 0) return false;
  if (b.have_above != 0 && b.y == 0) return false;
  if (b.num4_above_right < 0 || b.num4_below_left < 0) return false;
  if (b.plane != 0) return false;  // MRL is luma-only
  if (b.sb_height <= 0 || (b.sb_height & (b.sb_height - 1)) != 0) return false;

  if (p.mode < 0 || p.mode >= av2enc::cuda::intra_modes::kIntraModes) {
    return false;
  }
  if (!av2enc::cuda::intra_modes::is_directional_mode(p.mode)) return false;
  if (p.angle_delta < -kMaxAngleDelta || p.angle_delta > kMaxAngleDelta) {
    return false;
  }
  if (p.mrl_index < 0 || p.mrl_index > kMaxMrlIndex) return false;
  if ((p.filter_type_above & ~1) != 0 || (p.filter_type_left & ~1) != 0 ||
      (p.enable_edge_filter & ~1) != 0 || (p.mrl_sec_index & ~1) != 0) {
    return false;
  }
  if (p.mrl_index == 0 && p.mrl_sec_index != 0) return false;
  if (p.mrl_sec_index != 0 && b.w == 4 && b.h == 4) return false;

  if (b.have_left != 0 && b.x - 1 - p.mrl_index < 0) return false;
  const int above_mrl =
      (b.y & (b.sb_height - 1)) == 0 ? 0 : p.mrl_index;
  if (b.have_above != 0 && b.y - 1 - above_mrl < 0) return false;
  return true;
}

}  // namespace

cudaError_t mrl_predict(const MrlBlock& block, const MrlParams& params,
                        std::uint8_t* pred, cudaStream_t stream) {
  if (pred == nullptr || !valid_block(block, params)) {
    return cudaErrorInvalidValue;
  }
  mrl_predict_kernel<<<1, kThreads, 0, stream>>>(block, params, pred);
  return cudaGetLastError();
}

cudaError_t mrl_predict_batch(const MrlJob* d_jobs, int job_count,
                              cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && d_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) return cudaSuccess;
  mrl_predict_batch_kernel<<<job_count, kThreads, 0, stream>>>(d_jobs);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::mrl
