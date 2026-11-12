// CUDA intra-full kernels and arena-backed launchers.
//
// See intra_full.h for the API and the memory contract, and
// intra_full_device.cuh for the bit-exact device bodies.  This translation unit
// contains no cudaMalloc / cudaFree / cudaMallocHost: all device memory is
// either the caller's arena allocation or the workspace staged out of it.
#include "intra_full.h"

#include "intra_full_device.cuh"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::intra_full {
namespace {

constexpr int kThreads = 128;
constexpr int kMaxTxElems = 32 * 32;  // §7.15.4 adjusted size cap

bool valid_tx_dim(int v) {
  return v == 4 || v == 8 || v == 16 || v == 32;
}

bool valid_tx_job(const TxJob& job) {
  const av2enc::txv::TxParams& p = job.params;
  if (!is_intra_identity_type(static_cast<int>(p.tx_type))) {
    return false;
  }
  int row = -1;
  int col = -1;
  if (!intra_identity_1d(static_cast<int>(p.tx_type), &row, &col)) {
    return false;
  }
  if (static_cast<int>(p.row_type) != row ||
      static_cast<int>(p.col_type) != col) {
    return false;
  }
  if (!valid_tx_dim(p.w) || !valid_tx_dim(p.h)) {
    return false;
  }
  const int elems = static_cast<int>(p.w) * static_cast<int>(p.h);
  if (elems > kMaxTxElems) {
    return false;
  }
  if (!av2enc::txv::tx_size_supported(p.coded_w, p.coded_h)) {
    return false;
  }
  if (p.coded_w % p.w != 0 || p.coded_h % p.h != 0) {
    return false;
  }
  if (p.row_shift <= 0 || p.row_shift >= 31 || p.col_shift <= 0 ||
      p.col_shift >= 31) {
    return false;
  }
  if (p.dq_denom != 1 && p.dq_denom != 2 && p.dq_denom != 4) {
    return false;
  }
  if (!(p.scale_w > 0.0) || !(p.scale_h > 0.0) ||
      !(p.quant_factor > 0.0) || !(p.rect_prescale > 0.0)) {
    return false;
  }
  return job.in_offset >= 0 && job.out_offset >= 0;
}

bool valid_dc_job(const DcJob& job) {
  const DcSubsampledDesc& in = job.in;
  if (in.w < 1 || in.w > 64 || in.h < 1 || in.h > 64) {
    return false;
  }
  if (in.bit_depth < 8 || in.bit_depth > 12) {
    return false;
  }
  if (in.have_above != 0 && (in.have_above != 1 || in.above == nullptr)) {
    return false;
  }
  if (in.have_left != 0 && (in.have_left != 1 || in.left == nullptr)) {
    return false;
  }
  return job.dst != nullptr && job.dst_stride >= in.w;
}

bool valid_corner_job(const CornerJob& job) {
  if (job.above == nullptr || job.left == nullptr) {
    return false;
  }
  return job.w >= 1 && job.w <= 64 && job.h >= 1 && job.h <= 64 &&
         job.p_angle >= 0 && job.p_angle <= 360;
}

bool valid_chroma_query(const ChromaQuery& q) {
  if (q.plane < 1 || q.plane > 2) {
    return false;
  }
  if (q.enable_chroma_dctonly != 0 && q.enable_chroma_dctonly != 1) {
    return false;
  }
  if (q.uv_mode < 0 || q.uv_mode > 13) {
    return false;
  }
  if (q.uv_angle_delta < -3 || q.uv_angle_delta > 3) {
    return false;
  }
  return av2enc::txv::tx_size_supported(q.coded_w, q.coded_h);
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void if_dc_kernel(const DcJob* __restrict__ jobs,
                             DcResult* __restrict__ traces,
                             int count) {
  if (static_cast<int>(blockIdx.x) >= count) {
    return;
  }
  const DcJob job = jobs[blockIdx.x];
  intra_full_dc_subsampled_dev(job.in, job.dst, job.dst_stride,
                               traces + blockIdx.x);
}

__global__ void if_corner_kernel(const CornerJob* __restrict__ jobs,
                                 CornerResult* __restrict__ results,
                                 int count) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) {
    return;
  }
  const CornerJob job = jobs[i];
  CornerResult result;
  result.applied = intra_full_filter_corner_apply_dev(
      job.w, job.h, job.p_angle, job.above, job.left);
  results[i] = result;
}

__global__ void if_chroma_kernel(const ChromaQuery* __restrict__ queries,
                                 ChromaResult* __restrict__ results,
                                 int count) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) {
    return;
  }
  ChromaResult result;
  (void)intra_full_derive_chroma_type_dev(queries[i], &result);
  results[i] = result;
}

__global__ void if_forward_kernel(const TxJob* __restrict__ jobs,
                                  const std::int16_t* __restrict__ residual,
                                  double* __restrict__ coeff) {
  extern __shared__ double s_tmp[];
  const TxJob job = jobs[blockIdx.x];
  intra_full_forward_block_dev(
      residual + static_cast<std::ptrdiff_t>(job.in_offset),
      coeff + static_cast<std::ptrdiff_t>(job.in_offset), job.params, s_tmp);
}

__global__ void if_inverse_kernel(const TxJob* __restrict__ jobs,
                                  const std::int32_t* __restrict__ dequant,
                                  std::int16_t* __restrict__ coded) {
  extern __shared__ std::int32_t s_inter[];
  const TxJob job = jobs[blockIdx.x];
  intra_full_inverse_block_dev(
      dequant + static_cast<std::ptrdiff_t>(job.in_offset),
      coded + static_cast<std::ptrdiff_t>(job.out_offset), job.params, s_inter);
}

// ---------------------------------------------------------------------------
// Staging helpers
// ---------------------------------------------------------------------------

template <typename T>
T* stage_jobs(Arena& arena, int count) {
  return arena.allocate_array<T>(static_cast<std::size_t>(count));
}

int grid_for(int count) { return (count + kThreads - 1) / kThreads; }

}  // namespace

// ---------------------------------------------------------------------------
// §7.13.2.11 DC subsampled
// ---------------------------------------------------------------------------

Status Workspace::dc_subsampled_predict(const DcJob* host_jobs, int count,
                                        DcResult* host_results,
                                        cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_dc_jobs ||
      (count > 0 && host_jobs == nullptr)) {
    return Status::kInvalidArgument;
  }
  for (int i = 0; i < count; ++i) {
    if (!valid_dc_job(host_jobs[i])) {
      return Status::kInvalidArgument;
    }
  }
  if (count == 0) {
    return Status::kOk;
  }

  Arena& arena = *arena_;
  const std::uint64_t mark = arena.mark();
  DcJob* d_jobs = stage_jobs<DcJob>(arena, count);
  if (d_jobs == nullptr) {
    arena.rewind(mark);
    return Status::kArenaExhausted;
  }
  Status status = status_from_cuda(cudaMemcpyAsync(
      d_jobs, host_jobs, sizeof(DcJob) * static_cast<std::size_t>(count),
      cudaMemcpyHostToDevice, stream));
  if (is_ok(status)) {
    if_dc_kernel<<<count, kThreads, 0, stream>>>(d_jobs, d_dc_results_, count);
    status = last_error();
  }
  if (is_ok(status) && host_results != nullptr) {
    status = status_from_cuda(cudaMemcpyAsync(
        host_results, d_dc_results_,
        sizeof(DcResult) * static_cast<std::size_t>(count),
        cudaMemcpyDeviceToHost, stream));
  }
  if (is_ok(status)) {
    status = status_from_cuda(cudaStreamSynchronize(stream));
  }
  arena.rewind(mark);
  return status;
}

// ---------------------------------------------------------------------------
// §7.13.2.14 filter corner
// ---------------------------------------------------------------------------

Status Workspace::filter_corner(const CornerJob* host_jobs, int count,
                                CornerResult* host_results,
                                cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_corner_jobs ||
      (count > 0 && host_jobs == nullptr)) {
    return Status::kInvalidArgument;
  }
  for (int i = 0; i < count; ++i) {
    if (!valid_corner_job(host_jobs[i])) {
      return Status::kInvalidArgument;
    }
  }
  if (count == 0) {
    return Status::kOk;
  }

  Arena& arena = *arena_;
  const std::uint64_t mark = arena.mark();
  CornerJob* d_jobs = stage_jobs<CornerJob>(arena, count);
  if (d_jobs == nullptr) {
    arena.rewind(mark);
    return Status::kArenaExhausted;
  }
  Status status = status_from_cuda(cudaMemcpyAsync(
      d_jobs, host_jobs, sizeof(CornerJob) * static_cast<std::size_t>(count),
      cudaMemcpyHostToDevice, stream));
  if (is_ok(status)) {
    if_corner_kernel<<<grid_for(count), kThreads, 0, stream>>>(
        d_jobs, d_corner_results_, count);
    status = last_error();
  }
  if (is_ok(status) && host_results != nullptr) {
    status = status_from_cuda(cudaMemcpyAsync(
        host_results, d_corner_results_,
        sizeof(CornerResult) * static_cast<std::size_t>(count),
        cudaMemcpyDeviceToHost, stream));
  }
  if (is_ok(status)) {
    status = status_from_cuda(cudaStreamSynchronize(stream));
  }
  arena.rewind(mark);
  return status;
}

// ---------------------------------------------------------------------------
// §5.20.7.29 chroma type derivation
// ---------------------------------------------------------------------------

Status Workspace::derive_chroma_type(const ChromaQuery* host_queries, int count,
                                     ChromaResult* host_results,
                                     cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_chroma_jobs ||
      (count > 0 && host_queries == nullptr) ||
      (count > 0 && host_results == nullptr)) {
    return Status::kInvalidArgument;
  }
  for (int i = 0; i < count; ++i) {
    if (!valid_chroma_query(host_queries[i])) {
      return Status::kInvalidArgument;
    }
  }
  if (count == 0) {
    return Status::kOk;
  }

  Arena& arena = *arena_;
  const std::uint64_t mark = arena.mark();
  ChromaQuery* d_queries = stage_jobs<ChromaQuery>(arena, count);
  if (d_queries == nullptr) {
    arena.rewind(mark);
    return Status::kArenaExhausted;
  }
  Status status = status_from_cuda(cudaMemcpyAsync(
      d_queries, host_queries,
      sizeof(ChromaQuery) * static_cast<std::size_t>(count),
      cudaMemcpyHostToDevice, stream));
  if (is_ok(status)) {
    if_chroma_kernel<<<grid_for(count), kThreads, 0, stream>>>(
        d_queries, d_chroma_results_, count);
    status = last_error();
  }
  if (is_ok(status)) {
    status = status_from_cuda(cudaMemcpyAsync(
        host_results, d_chroma_results_,
        sizeof(ChromaResult) * static_cast<std::size_t>(count),
        cudaMemcpyDeviceToHost, stream));
  }
  if (is_ok(status)) {
    status = status_from_cuda(cudaStreamSynchronize(stream));
  }
  arena.rewind(mark);
  return status;
}

// ---------------------------------------------------------------------------
// Identity transforms
// ---------------------------------------------------------------------------

Status Workspace::identity_forward(const TxJob* host_jobs, int count,
                                   const std::int16_t* residual, double* coeff,
                                   cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_tx_blocks ||
      (count > 0 && (host_jobs == nullptr || residual == nullptr ||
                     coeff == nullptr))) {
    return Status::kInvalidArgument;
  }
  int max_elems = 1;
  for (int i = 0; i < count; ++i) {
    if (!valid_tx_job(host_jobs[i])) {
      return Status::kInvalidArgument;
    }
    const int elems =
        static_cast<int>(host_jobs[i].params.w) * host_jobs[i].params.h;
    if (elems > max_elems) {
      max_elems = elems;
    }
  }
  if (count == 0) {
    return Status::kOk;
  }

  Arena& arena = *arena_;
  const std::uint64_t mark = arena.mark();
  TxJob* d_jobs = stage_jobs<TxJob>(arena, count);
  if (d_jobs == nullptr) {
    arena.rewind(mark);
    return Status::kArenaExhausted;
  }
  Status status = status_from_cuda(cudaMemcpyAsync(
      d_jobs, host_jobs, sizeof(TxJob) * static_cast<std::size_t>(count),
      cudaMemcpyHostToDevice, stream));
  if (is_ok(status)) {
    const std::size_t smem =
        static_cast<std::size_t>(max_elems) * sizeof(double);
    if_forward_kernel<<<count, max_elems, smem, stream>>>(d_jobs, residual,
                                                          coeff);
    status = last_error();
  }
  if (is_ok(status)) {
    status = status_from_cuda(cudaStreamSynchronize(stream));
  }
  arena.rewind(mark);
  return status;
}

Status Workspace::identity_inverse(const TxJob* host_jobs, int count,
                                   const std::int32_t* dequant,
                                   std::int16_t* coded, cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_tx_blocks ||
      (count > 0 && (host_jobs == nullptr || dequant == nullptr ||
                     coded == nullptr))) {
    return Status::kInvalidArgument;
  }
  int max_elems = 1;
  for (int i = 0; i < count; ++i) {
    if (!valid_tx_job(host_jobs[i])) {
      return Status::kInvalidArgument;
    }
    const int elems =
        static_cast<int>(host_jobs[i].params.w) * host_jobs[i].params.h;
    if (elems > max_elems) {
      max_elems = elems;
    }
  }
  if (count == 0) {
    return Status::kOk;
  }

  Arena& arena = *arena_;
  const std::uint64_t mark = arena.mark();
  TxJob* d_jobs = stage_jobs<TxJob>(arena, count);
  if (d_jobs == nullptr) {
    arena.rewind(mark);
    return Status::kArenaExhausted;
  }
  Status status = status_from_cuda(cudaMemcpyAsync(
      d_jobs, host_jobs, sizeof(TxJob) * static_cast<std::size_t>(count),
      cudaMemcpyHostToDevice, stream));
  if (is_ok(status)) {
    const std::size_t smem =
        static_cast<std::size_t>(max_elems) * sizeof(std::int32_t);
    if_inverse_kernel<<<count, max_elems, smem, stream>>>(d_jobs, dequant,
                                                          coded);
    status = last_error();
  }
  if (is_ok(status)) {
    status = status_from_cuda(cudaStreamSynchronize(stream));
  }
  arena.rewind(mark);
  return status;
}

}  // namespace av2enc::cuda::intra_full
