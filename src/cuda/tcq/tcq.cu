// CUDA kernels and launchers for the TCQ / CCTX / parity-hiding batch.
//
// One thread owns one complete coefficient block.  The per-job body is
// `detail::tcq_run_job`, `detail::tcq_run_cctx_job` and
// `detail::tcq_run_parity_job` from tcq_device.cuh, so the GPU and the host
// emulation test run the same code.
//
// There is no allocation, no synchronization, no shared memory and no global
// mutable state here.  Build with -fmad=false so the DP's `a + b * c` does not
// contract into an FMA and stays bit-identical to the CPU reference compiled
// with -ffp-contract=off.

#include "cuda/tcq/tcq.h"

#include "cuda/tcq/tcq_device.cuh"

namespace av2enc::cuda::tcq {
namespace {

enum : int { kThreads = 128 };

__global__ void tcq_kernel(TcqDeviceBatch batch) {
  const std::int32_t job_index =
      static_cast<std::int32_t>(blockIdx.x) *
          static_cast<std::int32_t>(blockDim.x) +
      static_cast<std::int32_t>(threadIdx.x);
  if (job_index >= batch.num_jobs) {
    return;
  }
  detail::tcq_run_job(batch, job_index);
}

__global__ void cctx_kernel(CctxDeviceBatch batch) {
  const std::int32_t job_index =
      static_cast<std::int32_t>(blockIdx.x) *
          static_cast<std::int32_t>(blockDim.x) +
      static_cast<std::int32_t>(threadIdx.x);
  if (job_index >= batch.num_jobs) {
    return;
  }
  detail::tcq_run_cctx_job(batch, job_index);
}

__global__ void parity_kernel(ParityDeviceBatch batch) {
  const std::int32_t job_index =
      static_cast<std::int32_t>(blockIdx.x) *
          static_cast<std::int32_t>(blockDim.x) +
      static_cast<std::int32_t>(threadIdx.x);
  if (job_index >= batch.num_jobs) {
    return;
  }
  detail::tcq_run_parity_job(batch, job_index);
}

bool workspace_ready(const TcqWorkspace& workspace) {
  return workspace.required_bytes != 0 && workspace.jobs != nullptr &&
         workspace.original != nullptr && workspace.pre_max != nullptr &&
         workspace.normal_rate != nullptr &&
         workspace.eob_coeff_rate != nullptr &&
         workspace.block_eob_rate != nullptr && workspace.tail != nullptr &&
         workspace.path_base != nullptr && workspace.base_levels != nullptr &&
         workspace.quant_levels != nullptr && workspace.states != nullptr &&
         workspace.eob != nullptr && workspace.status != nullptr &&
         workspace.cctx_jobs != nullptr && workspace.u_coeff != nullptr &&
         workspace.v_coeff != nullptr &&
         workspace.cctx_u_levels != nullptr &&
         workspace.cctx_v_levels != nullptr &&
         workspace.cctx_status != nullptr &&
         workspace.parity_jobs != nullptr && workspace.scan != nullptr &&
         workspace.parity_levels != nullptr &&
         workspace.parity_status != nullptr;
}

}  // namespace

cudaError_t tcq_quantize_batch(TcqWorkspace& workspace,
                               const TcqJob* host_jobs, int job_count,
                               cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && host_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) return cudaSuccess;
  if (!workspace_ready(workspace) || job_count > workspace.max_jobs) {
    return cudaErrorInvalidValue;
  }
  const cudaError_t copy =
      cudaMemcpy(workspace.jobs, host_jobs,
                 static_cast<std::size_t>(job_count) * sizeof(TcqJob),
                 cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) return copy;
  const int blocks = (job_count + kThreads - 1) / kThreads;
  tcq_kernel<<<blocks, kThreads, 0, stream>>>(tcq_batch(workspace, job_count));
  return cudaGetLastError();
}

cudaError_t cctx_search_batch(TcqWorkspace& workspace,
                              const CctxJob* host_jobs, int job_count,
                              cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && host_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) return cudaSuccess;
  if (!workspace_ready(workspace) || job_count > workspace.max_jobs) {
    return cudaErrorInvalidValue;
  }
  const cudaError_t copy =
      cudaMemcpy(workspace.cctx_jobs, host_jobs,
                 static_cast<std::size_t>(job_count) * sizeof(CctxJob),
                 cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) return copy;
  const int blocks = (job_count + kThreads - 1) / kThreads;
  cctx_kernel<<<blocks, kThreads, 0, stream>>>(
      cctx_batch(workspace, job_count));
  return cudaGetLastError();
}

cudaError_t parity_hide_batch(TcqWorkspace& workspace,
                              const ParityJob* host_jobs, int job_count,
                              cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && host_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) return cudaSuccess;
  if (!workspace_ready(workspace) || job_count > workspace.max_jobs) {
    return cudaErrorInvalidValue;
  }
  const cudaError_t copy =
      cudaMemcpy(workspace.parity_jobs, host_jobs,
                 static_cast<std::size_t>(job_count) * sizeof(ParityJob),
                 cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) return copy;
  const int blocks = (job_count + kThreads - 1) / kThreads;
  parity_kernel<<<blocks, kThreads, 0, stream>>>(
      parity_batch(workspace, job_count));
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::tcq
