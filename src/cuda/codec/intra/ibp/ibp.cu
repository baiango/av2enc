// AV2 intra bi-prediction (IBP) — CUDA kernels.
//
// Integer only.  The device body lives in ibp_predict_device.cuh so the
// standalone kernel and the host-emulation parity test run the same code.

#include "ibp.h"

#include "ibp_predict_device.cuh"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::ibp {
namespace {

constexpr int kBatchThreads = 128;
constexpr int kSingleMaxThreads = 256;

__global__ void ibp_predict_kernel(IbpBlock block, std::uint8_t* __restrict__ pred) {
  __shared__ SharedScratch s;
  ibp_run_block_device(block, pred, &s);
}

__global__ void ibp_predict_batch_kernel(const IbpJob* __restrict__ jobs) {
  __shared__ SharedScratch s;
  const IbpJob job = jobs[blockIdx.x];
  ibp_run_block_device(job.block, job.pred, &s);
}

int ibp_threads(int work_items) {
  int threads = ((work_items + 31) / 32) * 32;
  if (threads < 32) {
    threads = 32;
  }
  if (threads > kSingleMaxThreads) {
    threads = kSingleMaxThreads;
  }
  return threads;
}

bool ibp_valid_dim(int v) {
  return v == 4 || v == 8 || v == 16 || v == 32 || v == 64;
}

bool ibp_valid_decision_mode(IntraMode mode) {
  return mode == intra_modes::kDcPred ||
         intra_modes::is_directional_mode(static_cast<int>(mode));
}

bool ibp_valid_block(const IbpBlock& job, bool require_pred) {
  const IntraBlock& b = job.intra;
  if (!ibp_valid_dim(b.width) || !ibp_valid_dim(b.height)) {
    return false;
  }
  if (b.x0 < 0 || b.y0 < 0 || b.max_x < b.x0 || b.max_y < b.y0) {
    return false;
  }
  if (b.angle_delta < -intra_modes::kMaxAngleDelta ||
      b.angle_delta > intra_modes::kMaxAngleDelta) {
    return false;
  }
  const int mode = static_cast<int>(b.mode);
  if (mode < 0 || mode >= intra_modes::kIntraModes) {
    return false;
  }
  if (!intra_modes::is_directional_mode(mode) && b.angle_delta != 0) {
    return false;
  }
  if ((b.filter_type_above & ~1) != 0 || (b.filter_type_left & ~1) != 0) {
    return false;
  }
  if (b.num4_above_right < 0 || b.num4_below_left < 0 || b.plane < 0) {
    return false;
  }
  if ((job.enable_ibp & ~1) != 0 || (job.is_inter & ~1) != 0 ||
      (job.use_dip & ~1) != 0 || (job.uv_is_cfl & ~1) != 0) {
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
  if (require_pred && b.source_plane != nullptr) {
    if (b.source_stride <= 0 || b.source_max_x < b.x0 ||
        b.source_max_y < b.y0) {
      return false;
    }
  }
  return true;
}

}  // namespace

cudaError_t ibp_predict(const IbpBlock& block, std::uint8_t* pred,
                        cudaStream_t stream) {
  if (!ibp_valid_block(block, false) || pred == nullptr) {
    return cudaErrorInvalidValue;
  }
  // IBP has no effect on these modes; use the existing intra predictor.
  if (!ibp_valid_decision_mode(block.intra.mode)) {
    return intra_modes::intra_predict(block.intra, pred, stream);
  }
  const int work = block.intra.width * block.intra.height;
  ibp_predict_kernel<<<1, ibp_threads(work), 0, stream>>>(block, pred);
  return cudaGetLastError();
}

cudaError_t ibp_predict_batch(const IbpJob* d_jobs, int job_count,
                              cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && d_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) {
    return cudaSuccess;
  }
  ibp_predict_batch_kernel<<<job_count, kBatchThreads, 0, stream>>>(d_jobs);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::ibp
