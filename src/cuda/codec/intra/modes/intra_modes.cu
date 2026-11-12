// AV2 v1.0.0 full intra prediction kernels (M1).
//
// Integer only, mirroring the spec arithmetic exactly (Round2 / Clip1 /
// resolve_divisor). See intra_modes.h for the normative section map and the
// API contract.
//
// The device prediction body lives in intra_predict_device.cuh so the M1 GPU
// chroma full-UV loop can run exactly the same code. One CUDA block predicts
// one transform block. The above-row / left-column reference arrays live in
// shared memory with two slots of headroom so that the spec's AboveRow[-1],
// AboveRow[-2] and the directional extension indices are addressable without
// branches.

#include "intra_modes.h"

#include "intra_predict_device.cuh"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::intra_modes {
namespace {

constexpr int kBatchThreads = 128;
constexpr int kSingleMaxThreads = 256;

__global__ void im_predict_kernel(IntraBlock b,
                                  std::uint8_t* __restrict__ pred,
                                  std::int16_t* __restrict__ residual) {
  __shared__ SharedScratch s;
  intra_run_block_device(b, pred, residual, &s);
}

__global__ void im_predict_batch_kernel(const IntraJob* __restrict__ jobs) {
  __shared__ SharedScratch s;
  const IntraJob job = jobs[blockIdx.x];
  intra_run_block_device(job.block, job.pred, job.residual, &s);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

int im_threads(int work_items) {
  int threads = ((work_items + 31) / 32) * 32;
  if (threads < 32) {
    threads = 32;
  }
  if (threads > kSingleMaxThreads) {
    threads = kSingleMaxThreads;
  }
  return threads;
}

bool im_valid_dim(int v) {
  return v == 4 || v == 8 || v == 16 || v == 32 || v == 64;
}

bool im_valid_block(const IntraBlock& b, bool require_source) {
  if (!im_valid_dim(b.width) || !im_valid_dim(b.height)) {
    return false;
  }
  if (b.x0 < 0 || b.y0 < 0 || b.max_x < b.x0 || b.max_y < b.y0) {
    return false;
  }
  if (b.angle_delta < -kMaxAngleDelta || b.angle_delta > kMaxAngleDelta) {
    return false;
  }
  if (!is_directional_mode(static_cast<int>(b.mode)) && b.angle_delta != 0) {
    return false;
  }
  if (static_cast<int>(b.mode) < 0 || static_cast<int>(b.mode) >= kIntraModes) {
    return false;
  }
  if ((b.filter_type_above & ~1) != 0 || (b.filter_type_left & ~1) != 0) {
    return false;
  }
  if (b.num4_above_right < 0 || b.num4_below_left < 0) {
    return false;
  }
  if (b.plane < 0) {
    return false;
  }
  if (b.have_left != 0 || b.have_above != 0) {
    if (b.recon_plane == nullptr || b.recon_stride <= 0) {
      return false;
    }
  }
  if (b.have_left != 0 && b.x0 < 1) {
    return false;
  }
  if (b.have_above != 0 && b.y0 < 1) {
    return false;
  }
  if (require_source) {
    if (b.source_plane == nullptr || b.source_stride <= 0 ||
        b.source_max_x < b.x0 || b.source_max_y < b.y0) {
      return false;
    }
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

cudaError_t intra_predict(const IntraBlock& block, std::uint8_t* pred,
                          cudaStream_t stream) {
  if (!im_valid_block(block, false) || pred == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int work = block.width * block.height;
  im_predict_kernel<<<1, im_threads(work), 0, stream>>>(block, pred, nullptr);
  return cudaGetLastError();
}

cudaError_t intra_predict_residual(const IntraBlock& block, std::uint8_t* pred,
                                   std::int16_t* residual,
                                   cudaStream_t stream) {
  if (!im_valid_block(block, true) || pred == nullptr || residual == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int work = block.width * block.height;
  im_predict_kernel<<<1, im_threads(work), 0, stream>>>(block, pred, residual);
  return cudaGetLastError();
}

cudaError_t intra_predict_batch(const IntraJob* d_jobs, int job_count,
                                cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && d_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) {
    return cudaSuccess;
  }
  im_predict_batch_kernel<<<job_count, kBatchThreads, 0, stream>>>(d_jobs);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::intra_modes
