// CUDA inter DDT (Data Dependent Transform) kernels.
//
// See ddt_gpu.h for the contract and ddt_device.cuh for the bit-exact
// numerical body shared with the host emulation test.
//
// Allocation: every device byte used by the host-facing batch entry points is
// bump-allocated from the caller's av2enc::cuda::Arena between mark() and
// rewind().  There is no cudaMalloc/cudaFree in this file.
#include "cuda/codec/transform/ddt/ddt_gpu.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "cuda/arena.h"
#include "cuda/codec/transform/ddt/ddt_device.cuh"
#include "cuda/error.h"

namespace av2enc::cuda::ddt {
namespace {

using namespace dev;

constexpr int kThreads = 1024;

// ---------------------------------------------------------------------------
// Kernels: one CUDA block per transform block, one thread per adjusted-size
// element.  Every output element is an independent exact int64 dot product, so
// the parallel schedule is bit-identical to the CPU's row/column loops.
// ---------------------------------------------------------------------------

__global__ void forward_kernel(const std::int16_t* __restrict__ residual_adj,
                               std::int32_t* __restrict__ coeff,
                               std::int32_t* __restrict__ levels,
                               const DdtJob* __restrict__ jobs) {
  const DdtJob job = jobs[blockIdx.x];
  const TxParams p = job.params;
  const bool enable = job.enable_inter_ddt != 0;
  const bool is_inter = job.is_inter != 0;
  const bool intrabc = job.use_intrabc != 0;
  const Plan plan = make_plan(p, enable, is_inter, intrabc);
  const bool odd_rect = ((p.log2_w ^ p.log2_h) & 1) != 0;
  const int w = p.w;
  const int h = p.h;
  const int elems = w * h;

  __shared__ long long s_temp[kMaxElems];

  const int tid = static_cast<int>(threadIdx.x);
  if (tid < elems) {
    const int i = tid / w;
    const int j = tid - i * w;
    s_temp[tid] = forward_row_elem(residual_adj + job.in_offset + i * w, p,
                                   plan.row, j);
  }
  __syncthreads();

  if (tid < elems) {
    const int i = tid / w;
    const int j = tid - i * w;
    const std::int32_t value = forward_finish(
        forward_col_elem(s_temp, w, p, plan.col, i, j), odd_rect);
    coeff[job.out_offset + tid] = value;
    if (levels != nullptr) {
      levels[job.out_offset + tid] = level_from_coeff(value, p.dq_denom);
    }
  }
}

__global__ void inverse_kernel(const std::int32_t* __restrict__ input,
                               std::int16_t* __restrict__ residual_coded,
                               const DdtJob* __restrict__ jobs,
                               bool input_is_levels) {
  const DdtJob job = jobs[blockIdx.x];
  const TxParams p = job.params;
  const bool enable = job.enable_inter_ddt != 0;
  const bool is_inter = job.is_inter != 0;
  const bool intrabc = job.use_intrabc != 0;
  const Plan plan = make_plan(p, enable, is_inter, intrabc);
  const bool odd_rect = ((p.log2_w ^ p.log2_h) & 1) != 0;
  const int bit_depth = job.bit_depth;
  const int w = p.w;
  const int h = p.h;
  const int elems = w * h;

  __shared__ std::int32_t s_dq[kMaxElems];
  __shared__ std::int32_t s_inter[kMaxElems];

  const int tid = static_cast<int>(threadIdx.x);
  if (tid < elems) {
    const std::int32_t value = input[job.in_offset + tid];
    s_dq[tid] = input_is_levels ? dequantize_level(value, p.dq_denom) : value;
  }
  __syncthreads();

  if (tid < elems) {
    const int i = tid / w;
    const int j = tid - i * w;
    s_inter[tid] = inverse_row_elem(s_dq + i * w, p, plan.row, bit_depth,
                                    odd_rect, j);
  }
  __syncthreads();

  if (tid < elems) {
    const int i = tid / w;
    const int j = tid - i * w;
    const std::int32_t value =
        inverse_col_elem(s_inter, w, p, plan.col, bit_depth, i, j);
    store_coded(residual_coded + job.out_offset, p.coded_w, p.coded_h, w, h, i,
                j, static_cast<std::int16_t>(value));
  }
}

// ---------------------------------------------------------------------------
// Host helpers.
// ---------------------------------------------------------------------------

struct ArenaRewind {
  explicit ArenaRewind(Arena& arena) : arena_(arena), mark_(arena.mark()) {}
  ~ArenaRewind() { arena_.rewind(mark_); }
  ArenaRewind(const ArenaRewind&) = delete;
  ArenaRewind& operator=(const ArenaRewind&) = delete;

  Arena& arena_;
  std::uint64_t mark_;
};

bool fail(std::string* error, const std::string& what) {
  if (error != nullptr) *error = what;
  return false;
}

bool check_cuda(cudaError_t result, const char* what, std::string* error) {
  if (result == cudaSuccess) return true;
  if (error != nullptr) {
    *error = std::string("DDT GPU: ") + what + ": " + cuda_error_name(result);
  }
  (void)cudaGetLastError();
  return false;
}

int host_log2_dim(int n) {
  switch (n) {
    case 4:
      return 2;
    case 8:
      return 3;
    case 16:
      return 4;
    case 32:
      return 5;
    case 64:
      return 6;
    default:
      return -1;
  }
}

bool valid_coded_dim(int n) {
  return n == 4 || n == 8 || n == 16 || n == 32 || n == 64;
}

BatchStatus allocate(Arena& arena, std::uint64_t bytes, void** out,
                     std::string* error) {
  *out = nullptr;
  if (bytes == 0) return BatchStatus::kOk;
  void* raw = nullptr;
  const Status status = arena.allocate_checked(
      static_cast<std::size_t>(bytes), &raw, Arena::kAlignment);
  if (status == Status::kArenaExhausted) return BatchStatus::kArenaTooSmall;
  if (!is_ok(status)) {
    fail(error, std::string("DDT GPU: arena allocation failed: ") +
                    status_name(status));
    return BatchStatus::kInvalidArgument;
  }
  *out = raw;
  return BatchStatus::kOk;
}

struct Spans {
  std::uint64_t input = 0;
  std::uint64_t output = 0;
  bool ok = true;
};

Spans compute_spans(const DdtJob* jobs, int job_count, Direction direction) {
  Spans spans;
  for (int b = 0; b < job_count; ++b) {
    const DdtJob& job = jobs[b];
    const std::uint64_t elems =
        static_cast<std::uint64_t>(job.params.w) *
        static_cast<std::uint64_t>(job.params.h);
    const std::uint64_t out_elems =
        (direction == Direction::kForward)
            ? elems
            : static_cast<std::uint64_t>(job.params.coded_w) *
                  static_cast<std::uint64_t>(job.params.coded_h);
    const std::uint64_t in_end =
        static_cast<std::uint64_t>(job.in_offset) + elems;
    const std::uint64_t out_end =
        static_cast<std::uint64_t>(job.out_offset) + out_elems;
    if (in_end > spans.input) spans.input = in_end;
    if (out_end > spans.output) spans.output = out_end;
  }
  return spans;
}

}  // namespace

const char* batch_status_name(BatchStatus status) {
  switch (status) {
    case BatchStatus::kOk:
      return "ok";
    case BatchStatus::kInvalidArgument:
      return "invalid argument";
    case BatchStatus::kArenaTooSmall:
      return "arena too small";
    case BatchStatus::kCudaError:
      return "CUDA error";
  }
  return "unknown";
}

bool valid_job(const DdtJob& job, Direction direction, std::string* error) {
  (void)direction;
  const TxParams& p = job.params;
  if (!valid_coded_dim(p.coded_w) || !valid_coded_dim(p.coded_h)) {
    return fail(error, "DDT GPU: coded size is not an AV2 transform size");
  }
  const int expected_w = p.coded_w > kMaxDim ? kMaxDim : p.coded_w;
  const int expected_h = p.coded_h > kMaxDim ? kMaxDim : p.coded_h;
  if (p.w != expected_w || p.h != expected_h) {
    return fail(error, "DDT GPU: adjusted size does not match coded size");
  }
  if (p.coded_w % p.w != 0 || p.coded_h % p.h != 0) {
    return fail(error, "DDT GPU: coded size is not a multiple of adjusted size");
  }
  if (p.log2_w != host_log2_dim(p.coded_w) ||
      p.log2_h != host_log2_dim(p.coded_h)) {
    return fail(error, "DDT GPU: log2 size mismatch");
  }
  if (p.row_type < 0 || p.row_type > 3 || p.col_type < 0 ||
      p.col_type > 3) {
    return fail(error, "DDT GPU: 1D transform type out of range");
  }
  if (p.row_shift <= 0 || p.row_shift >= 31 || p.col_shift <= 0 ||
      p.col_shift >= 31) {
    return fail(error, "DDT GPU: transform shift out of range");
  }
  const int pels = static_cast<int>(p.coded_w) * p.coded_h;
  const int expected_denom = 1 << ((pels > 256 ? 1 : 0) + (pels > 1024 ? 1 : 0));
  if (p.dq_denom != expected_denom) {
    return fail(error, "DDT GPU: dq_denom does not match the coded area");
  }
  if (job.bit_depth != 8 && job.bit_depth != 10) {
    return fail(error, "DDT GPU: bit_depth must be 8 or 10");
  }
  if (job.in_offset < 0 || job.out_offset < 0) {
    return fail(error, "DDT GPU: negative element offset");
  }
  return true;
}

cudaError_t forward_batch_device(const std::int16_t* d_residual_adj,
                                 std::int32_t* d_coeff, std::int32_t* d_levels,
                                 const DdtJob* d_jobs, int job_count,
                                 cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && (d_residual_adj == nullptr ||
                                          d_coeff == nullptr ||
                                          d_jobs == nullptr))) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) return cudaSuccess;
  forward_kernel<<<static_cast<unsigned>(job_count), kThreads, 0, stream>>>(
      d_residual_adj, d_coeff, d_levels, d_jobs);
  return cudaGetLastError();
}

cudaError_t inverse_batch_device(const std::int32_t* d_input,
                                 std::int16_t* d_residual_coded,
                                 const DdtJob* d_jobs, int job_count,
                                 bool input_is_levels, cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && (d_input == nullptr ||
                                          d_residual_coded == nullptr ||
                                          d_jobs == nullptr))) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) return cudaSuccess;
  inverse_kernel<<<static_cast<unsigned>(job_count), kThreads, 0, stream>>>(
      d_input, d_residual_coded, d_jobs, input_is_levels);
  return cudaGetLastError();
}

BatchStatus forward_batch(Arena& arena, const DdtJob* jobs, int job_count,
                          const std::int16_t* host_residual_adj,
                          std::int32_t* host_coeff, std::int32_t* host_levels,
                          cudaStream_t stream, std::uint64_t* required_bytes,
                          std::string* error) {
  const BatchConfig config{Direction::kForward, job_count,
                           host_levels != nullptr, false};
  const std::uint64_t need = required_device_bytes(config);
  if (required_bytes != nullptr) *required_bytes = need;
  if (job_count < 0) {
    fail(error, "DDT GPU: negative batch size");
    return BatchStatus::kInvalidArgument;
  }
  if (job_count == 0) return BatchStatus::kOk;
  if (jobs == nullptr || host_residual_adj == nullptr || host_coeff == nullptr) {
    fail(error, "DDT GPU: null forward batch buffer");
    return BatchStatus::kInvalidArgument;
  }
  for (int b = 0; b < job_count; ++b) {
    if (!valid_job(jobs[b], Direction::kForward, error)) {
      return BatchStatus::kInvalidArgument;
    }
  }
  const Spans spans = compute_spans(jobs, job_count, Direction::kForward);
  const std::uint64_t in_capacity =
      1024ull * static_cast<std::uint64_t>(job_count);
  const std::uint64_t out_capacity =
      1024ull * static_cast<std::uint64_t>(job_count);
  if (spans.input > in_capacity || spans.output > out_capacity) {
    fail(error, "DDT GPU: job offsets exceed the batch buffer capacity");
    return BatchStatus::kInvalidArgument;
  }

  ArenaRewind rewind(arena);
  void* raw_jobs = nullptr;
  void* raw_in = nullptr;
  void* raw_coeff = nullptr;
  void* raw_levels = nullptr;
  const detail::WorkspaceBytes bytes = detail::workspace_bytes(config);
  BatchStatus status = allocate(arena, bytes.jobs, &raw_jobs, error);
  if (status == BatchStatus::kOk) {
    status = allocate(arena, bytes.input, &raw_in, error);
  }
  if (status == BatchStatus::kOk) {
    status = allocate(arena, bytes.coeff, &raw_coeff, error);
  }
  if (status == BatchStatus::kOk) {
    status = allocate(arena, bytes.levels, &raw_levels, error);
  }
  if (status != BatchStatus::kOk) return status;

  auto* d_jobs = static_cast<DdtJob*>(raw_jobs);
  auto* d_in = static_cast<std::int16_t*>(raw_in);
  auto* d_coeff = static_cast<std::int32_t*>(raw_coeff);
  auto* d_levels = static_cast<std::int32_t*>(raw_levels);

  if (!check_cuda(cudaMemcpyAsync(d_jobs, jobs,
                                  static_cast<std::size_t>(job_count) *
                                      sizeof(DdtJob),
                                  cudaMemcpyHostToDevice, stream),
                  "upload jobs", error)) {
    return BatchStatus::kCudaError;
  }
  if (!check_cuda(cudaMemcpyAsync(
                      d_in, host_residual_adj,
                      static_cast<std::size_t>(spans.input) * sizeof(std::int16_t),
                      cudaMemcpyHostToDevice, stream),
                  "upload residual", error)) {
    return BatchStatus::kCudaError;
  }
  if (!check_cuda(forward_batch_device(d_in, d_coeff, d_levels, d_jobs,
                                       job_count, stream),
                  "launch forward", error)) {
    return BatchStatus::kCudaError;
  }
  if (!check_cuda(cudaMemcpyAsync(
                      host_coeff, d_coeff,
                      static_cast<std::size_t>(spans.output) * sizeof(std::int32_t),
                      cudaMemcpyDeviceToHost, stream),
                  "download coefficients", error)) {
    return BatchStatus::kCudaError;
  }
  if (d_levels != nullptr) {
    if (!check_cuda(cudaMemcpyAsync(
                        host_levels, d_levels,
                        static_cast<std::size_t>(spans.output) *
                            sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost, stream),
                    "download levels", error)) {
      return BatchStatus::kCudaError;
    }
  }
  if (!check_cuda(cudaStreamSynchronize(stream), "synchronize", error)) {
    return BatchStatus::kCudaError;
  }
  return BatchStatus::kOk;
}

BatchStatus inverse_batch(Arena& arena, const DdtJob* jobs, int job_count,
                          const std::int32_t* host_input,
                          std::int16_t* host_residual_coded,
                          bool input_is_levels, cudaStream_t stream,
                          std::uint64_t* required_bytes, std::string* error) {
  const BatchConfig config{Direction::kInverse, job_count, false,
                           input_is_levels};
  const std::uint64_t need = required_device_bytes(config);
  if (required_bytes != nullptr) *required_bytes = need;
  if (job_count < 0) {
    fail(error, "DDT GPU: negative batch size");
    return BatchStatus::kInvalidArgument;
  }
  if (job_count == 0) return BatchStatus::kOk;
  if (jobs == nullptr || host_input == nullptr ||
      host_residual_coded == nullptr) {
    fail(error, "DDT GPU: null inverse batch buffer");
    return BatchStatus::kInvalidArgument;
  }
  for (int b = 0; b < job_count; ++b) {
    if (!valid_job(jobs[b], Direction::kInverse, error)) {
      return BatchStatus::kInvalidArgument;
    }
  }
  const Spans spans = compute_spans(jobs, job_count, Direction::kInverse);
  const std::uint64_t in_capacity =
      1024ull * static_cast<std::uint64_t>(job_count);
  const std::uint64_t out_capacity =
      4096ull * static_cast<std::uint64_t>(job_count);
  if (spans.input > in_capacity || spans.output > out_capacity) {
    fail(error, "DDT GPU: job offsets exceed the batch buffer capacity");
    return BatchStatus::kInvalidArgument;
  }

  ArenaRewind rewind(arena);
  void* raw_jobs = nullptr;
  void* raw_in = nullptr;
  void* raw_out = nullptr;
  const detail::WorkspaceBytes bytes = detail::workspace_bytes(config);
  BatchStatus status = allocate(arena, bytes.jobs, &raw_jobs, error);
  if (status == BatchStatus::kOk) {
    status = allocate(arena, bytes.input, &raw_in, error);
  }
  if (status == BatchStatus::kOk) {
    status = allocate(arena, bytes.coded, &raw_out, error);
  }
  if (status != BatchStatus::kOk) return status;

  auto* d_jobs = static_cast<DdtJob*>(raw_jobs);
  auto* d_in = static_cast<std::int32_t*>(raw_in);
  auto* d_out = static_cast<std::int16_t*>(raw_out);

  if (!check_cuda(cudaMemcpyAsync(d_jobs, jobs,
                                  static_cast<std::size_t>(job_count) *
                                      sizeof(DdtJob),
                                  cudaMemcpyHostToDevice, stream),
                  "upload jobs", error)) {
    return BatchStatus::kCudaError;
  }
  if (!check_cuda(cudaMemcpyAsync(
                      d_in, host_input,
                      static_cast<std::size_t>(spans.input) * sizeof(std::int32_t),
                      cudaMemcpyHostToDevice, stream),
                  "upload input", error)) {
    return BatchStatus::kCudaError;
  }
  if (!check_cuda(inverse_batch_device(d_in, d_out, d_jobs, job_count,
                                       input_is_levels, stream),
                  "launch inverse", error)) {
    return BatchStatus::kCudaError;
  }
  if (!check_cuda(cudaMemcpyAsync(
                      host_residual_coded, d_out,
                      static_cast<std::size_t>(spans.output) *
                          sizeof(std::int16_t),
                      cudaMemcpyDeviceToHost, stream),
                  "download residual", error)) {
    return BatchStatus::kCudaError;
  }
  if (!check_cuda(cudaStreamSynchronize(stream), "synchronize", error)) {
    return BatchStatus::kCudaError;
  }
  return BatchStatus::kOk;
}

}  // namespace av2enc::cuda::ddt
