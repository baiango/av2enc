// CUDA kernels and launchers for the AV2 lossless inter residual path.
//
// One CUDA block per job for the block routines (the job descriptor is uniform
// across the block); one thread per 4-sample vector for the standalone WHT.
// All device memory is owned by the caller's `Workspace`, which `init()`
// allocates once from `av2enc::cuda::Arena`.  No launcher allocates memory.

#include "lossless_inter.h"

#include "lossless_inter_device.cuh"
#include "lossless_inter_math.h"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::lossless_inter {
namespace {

constexpr int kThreads = 128;

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

bool valid_block_dims(int w, int h) noexcept {
  return w > 0 && h > 0 && w <= kMaxTransformDim && h <= kMaxTransformDim;
}

bool valid_transform_dims(int w, int h, int dct_dct) noexcept {
  if (!valid_block_dims(w, h)) return false;
  if (dct_dct == 0) return true;
  return li_pow2_dim(w) && li_pow2_dim(h);
}

Status check_batch(const Workspace& workspace, int count) noexcept {
  if (!workspace.valid()) return Status::kNotInitialized;
  if (count <= 0 || count > workspace.max_jobs) {
    return Status::kInvalidArgument;
  }
  return Status::kOk;
}

template <typename T>
Status upload(const Workspace& workspace, T* destination, const T* source,
              int count, cudaStream_t stream) noexcept {
  (void)workspace;
  if (source == nullptr || destination == nullptr) {
    return Status::kInvalidArgument;
  }
  const std::size_t bytes = sizeof(T) * static_cast<std::size_t>(count);
  return status_from_cuda(
      cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice,
                      stream));
}

template <typename T>
Status download(const T* source, T* destination, int count,
                cudaStream_t stream) noexcept {
  if (destination == nullptr) return Status::kOk;
  if (source == nullptr) return Status::kInvalidArgument;
  const std::size_t bytes = sizeof(T) * static_cast<std::size_t>(count);
  return status_from_cuda(cudaMemcpyAsync(destination, source, bytes,
                                          cudaMemcpyDeviceToHost, stream));
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void wht4_forward_kernel(const WhtJob* jobs, int count) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) return;
  li_wht4_forward(jobs[index].in, jobs[index].out);
}

__global__ void wht4_inverse_kernel(const WhtJob* jobs, int count) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) return;
  li_wht4_inverse(jobs[index].in, jobs[index].shift, jobs[index].out);
}

__global__ void forward_block_kernel(const ForwardJob* jobs, int count) {
  __shared__ int scratch[kMaxTransformSamples];
  const int index = static_cast<int>(blockIdx.x);
  if (index >= count) return;
  li_forward_block_device(jobs[index], static_cast<int>(threadIdx.x),
                          static_cast<int>(blockDim.x), scratch);
}

__global__ void inverse_block_kernel(const InverseJob* jobs, int count) {
  __shared__ int scratch[kMaxTransformSamples];
  const int index = static_cast<int>(blockIdx.x);
  if (index >= count) return;
  const InverseJob& job = jobs[index];
  li_inverse_residual_device(job.levels, job.residual, job.w, job.h,
                             job.dct_dct, job.dpcm, job.vertical,
                             static_cast<int>(threadIdx.x),
                             static_cast<int>(blockDim.x), scratch);
}

__global__ void dpcm_difference_kernel(const DpcmJob* jobs, int count) {
  const int index = static_cast<int>(blockIdx.x);
  if (index >= count) return;
  li_dpcm_difference_device(jobs[index], static_cast<int>(threadIdx.x),
                            static_cast<int>(blockDim.x));
}

__global__ void dpcm_cumsum_kernel(const DpcmJob* jobs, int count) {
  const int index = static_cast<int>(blockIdx.x);
  if (index >= count) return;
  li_dpcm_cumsum_device(jobs[index], static_cast<int>(threadIdx.x),
                        static_cast<int>(blockDim.x));
}

__global__ void reconstruct_block_kernel(const ReconJob* jobs, int count) {
  __shared__ int scratch[kMaxTransformSamples];
  __shared__ int residual[kMaxTransformSamples];
  const int index = static_cast<int>(blockIdx.x);
  if (index >= count) return;
  li_reconstruct_block_device(jobs[index], static_cast<int>(threadIdx.x),
                              static_cast<int>(blockDim.x), scratch, residual);
}

__global__ void reconstruct_segmented_kernel(
    const ReconJob* jobs, int count, const FrameLossless* segment_qp,
    std::uint8_t* status) {
  __shared__ int scratch[kMaxTransformSamples];
  __shared__ int residual[kMaxTransformSamples];
  const int index = static_cast<int>(blockIdx.x);
  if (index >= count) return;
  const ReconJob& job = jobs[index];

  int segment = job.segment_id;
  if (segment < 0) segment = 0;
  if (segment >= kMaxSegments) segment = kMaxSegments - 1;

  // All threads of the block read the same record, so the branch is uniform.
  bool lossless = true;
  if (segment_qp != nullptr) {
    int frame = job.frame_index;
    if (frame < 0) frame = 0;
    const FrameLossless qp = segment_qp[frame];
    lossless = qp.lossless[segment];
  }

  if (!lossless) {
    if (threadIdx.x == 0 && status != nullptr) status[index] = 0;
    return;
  }
  if (threadIdx.x == 0 && status != nullptr) status[index] = 1;
  li_reconstruct_block_device(job, static_cast<int>(threadIdx.x),
                              static_cast<int>(blockDim.x), scratch, residual);
}

__global__ void select_tx_size_kernel(const TxSizeRequest* requests,
                                      TxSizeDecision* decisions, int count) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) return;
  decisions[index] = li_select_tx_size(requests[index]);
}

__global__ void derive_lossless_kernel(const FrameQuantInputs* inputs,
                                       FrameLossless* output, int count) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) return;
  output[index] = li_derive_lossless(inputs[index]);
}

int thread_blocks(int count) {
  return (count + kThreads - 1) / kThreads;
}

}  // namespace

// ---------------------------------------------------------------------------
// WHT
// ---------------------------------------------------------------------------

Status wht4_forward(Workspace& workspace, const WhtJob* jobs, int count,
                    cudaStream_t stream) {
  Status status = check_batch(workspace, count);
  if (!is_ok(status)) return status;
  for (int i = 0; i < count; ++i) {
    if (jobs[i].in == nullptr || jobs[i].out == nullptr) {
      return Status::kInvalidArgument;
    }
  }
  status = upload(workspace, workspace.wht_jobs, jobs, count, stream);
  if (!is_ok(status)) return status;
  wht4_forward_kernel<<<thread_blocks(count), kThreads, 0, stream>>>(
      workspace.wht_jobs, count);
  return last_error();
}

Status wht4_inverse(Workspace& workspace, const WhtJob* jobs, int count,
                    cudaStream_t stream) {
  Status status = check_batch(workspace, count);
  if (!is_ok(status)) return status;
  for (int i = 0; i < count; ++i) {
    if (jobs[i].in == nullptr || jobs[i].out == nullptr) {
      return Status::kInvalidArgument;
    }
    if (jobs[i].shift < 0 || jobs[i].shift > 30) {
      return Status::kInvalidArgument;
    }
  }
  status = upload(workspace, workspace.wht_jobs, jobs, count, stream);
  if (!is_ok(status)) return status;
  wht4_inverse_kernel<<<thread_blocks(count), kThreads, 0, stream>>>(
      workspace.wht_jobs, count);
  return last_error();
}

// ---------------------------------------------------------------------------
// DPCM
// ---------------------------------------------------------------------------

Status dpcm_difference(Workspace& workspace, const DpcmJob* jobs, int count,
                       cudaStream_t stream) {
  Status status = check_batch(workspace, count);
  if (!is_ok(status)) return status;
  for (int i = 0; i < count; ++i) {
    if (jobs[i].in == nullptr || jobs[i].out == nullptr) {
      return Status::kInvalidArgument;
    }
    if (!valid_block_dims(jobs[i].w, jobs[i].h)) {
      return Status::kInvalidArgument;
    }
  }
  status = upload(workspace, workspace.dpcm_jobs, jobs, count, stream);
  if (!is_ok(status)) return status;
  dpcm_difference_kernel<<<count, kThreads, 0, stream>>>(
      workspace.dpcm_jobs, count);
  return last_error();
}

Status dpcm_cumsum(Workspace& workspace, const DpcmJob* jobs, int count,
                   cudaStream_t stream) {
  Status status = check_batch(workspace, count);
  if (!is_ok(status)) return status;
  for (int i = 0; i < count; ++i) {
    if (jobs[i].in == nullptr || jobs[i].out == nullptr) {
      return Status::kInvalidArgument;
    }
    if (!valid_block_dims(jobs[i].w, jobs[i].h)) {
      return Status::kInvalidArgument;
    }
  }
  status = upload(workspace, workspace.dpcm_jobs, jobs, count, stream);
  if (!is_ok(status)) return status;
  dpcm_cumsum_kernel<<<count, kThreads, 0, stream>>>(workspace.dpcm_jobs,
                                                     count);
  return last_error();
}

// ---------------------------------------------------------------------------
// Block forward / inverse / reconstruction
// ---------------------------------------------------------------------------

Status forward_blocks(Workspace& workspace, const ForwardJob* jobs, int count,
                      cudaStream_t stream) {
  Status status = check_batch(workspace, count);
  if (!is_ok(status)) return status;
  for (int i = 0; i < count; ++i) {
    if (jobs[i].residual == nullptr || jobs[i].levels == nullptr) {
      return Status::kInvalidArgument;
    }
    if (!valid_transform_dims(jobs[i].w, jobs[i].h, jobs[i].dct_dct)) {
      return Status::kInvalidArgument;
    }
  }
  status = upload(workspace, workspace.forward_jobs, jobs, count, stream);
  if (!is_ok(status)) return status;
  forward_block_kernel<<<count, kThreads, 0, stream>>>(
      workspace.forward_jobs, count);
  return last_error();
}

Status inverse_blocks(Workspace& workspace, const InverseJob* jobs, int count,
                      cudaStream_t stream) {
  Status status = check_batch(workspace, count);
  if (!is_ok(status)) return status;
  for (int i = 0; i < count; ++i) {
    if (jobs[i].levels == nullptr || jobs[i].residual == nullptr) {
      return Status::kInvalidArgument;
    }
    if (!valid_transform_dims(jobs[i].w, jobs[i].h, jobs[i].dct_dct)) {
      return Status::kInvalidArgument;
    }
  }
  status = upload(workspace, workspace.inverse_jobs, jobs, count, stream);
  if (!is_ok(status)) return status;
  inverse_block_kernel<<<count, kThreads, 0, stream>>>(
      workspace.inverse_jobs, count);
  return last_error();
}

Status reconstruct_blocks(Workspace& workspace, const ReconJob* jobs,
                          int count, cudaStream_t stream) {
  Status status = check_batch(workspace, count);
  if (!is_ok(status)) return status;
  for (int i = 0; i < count; ++i) {
    if (jobs[i].levels == nullptr || jobs[i].pred == nullptr ||
        jobs[i].recon == nullptr) {
      return Status::kInvalidArgument;
    }
    if (!valid_transform_dims(jobs[i].w, jobs[i].h, jobs[i].dct_dct)) {
      return Status::kInvalidArgument;
    }
    if (jobs[i].pred_stride < jobs[i].w || jobs[i].recon_stride < jobs[i].w) {
      return Status::kInvalidArgument;
    }
  }
  status = upload(workspace, workspace.recon_jobs, jobs, count, stream);
  if (!is_ok(status)) return status;
  reconstruct_block_kernel<<<count, kThreads, 0, stream>>>(
      workspace.recon_jobs, count);
  return last_error();
}

Status reconstruct_blocks_segmented(Workspace& workspace, const ReconJob* jobs,
                                    int count,
                                    const FrameLossless* segment_qp,
                                    cudaStream_t stream) {
  Status status = check_batch(workspace, count);
  if (!is_ok(status)) return status;
  if (segment_qp == nullptr && workspace.lossless == nullptr) {
    return Status::kInvalidArgument;
  }
  for (int i = 0; i < count; ++i) {
    if (jobs[i].levels == nullptr || jobs[i].pred == nullptr ||
        jobs[i].recon == nullptr) {
      return Status::kInvalidArgument;
    }
    if (!valid_transform_dims(jobs[i].w, jobs[i].h, jobs[i].dct_dct)) {
      return Status::kInvalidArgument;
    }
    if (jobs[i].pred_stride < jobs[i].w || jobs[i].recon_stride < jobs[i].w) {
      return Status::kInvalidArgument;
    }
  }
  status = upload(workspace, workspace.recon_jobs, jobs, count, stream);
  if (!is_ok(status)) return status;
  const FrameLossless* qp =
      (segment_qp != nullptr) ? segment_qp : workspace.lossless;
  reconstruct_segmented_kernel<<<count, kThreads, 0, stream>>>(
      workspace.recon_jobs, count, qp, workspace.recon_status);
  return last_error();
}

// ---------------------------------------------------------------------------
// Host-facing derivations (device batch + optional download)
// ---------------------------------------------------------------------------

Status select_tx_size_batch(Workspace& workspace, const TxSizeRequest* requests,
                            int count, TxSizeDecision* host_out,
                            cudaStream_t stream) {
  Status status = check_batch(workspace, count);
  if (!is_ok(status)) return status;
  if (requests == nullptr) return Status::kInvalidArgument;
  status = upload(workspace, workspace.tx_requests, requests, count, stream);
  if (!is_ok(status)) return status;
  select_tx_size_kernel<<<thread_blocks(count), kThreads, 0, stream>>>(
      workspace.tx_requests, workspace.tx_decisions, count);
  status = last_error();
  if (!is_ok(status)) return status;
  return download(workspace.tx_decisions, host_out, count, stream);
}

Status derive_lossless_batch(Workspace& workspace,
                             const FrameQuantInputs* inputs, int count,
                             FrameLossless* host_out, cudaStream_t stream) {
  Status status = check_batch(workspace, count);
  if (!is_ok(status)) return status;
  if (inputs == nullptr) return Status::kInvalidArgument;
  status = upload(workspace, workspace.quant_inputs, inputs, count, stream);
  if (!is_ok(status)) return status;
  derive_lossless_kernel<<<thread_blocks(count), kThreads, 0, stream>>>(
      workspace.quant_inputs, workspace.lossless, count);
  status = last_error();
  if (!is_ok(status)) return status;
  return download(workspace.lossless, host_out, count, stream);
}

}  // namespace av2enc::cuda::lossless_inter
