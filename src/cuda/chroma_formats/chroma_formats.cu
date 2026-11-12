// CUDA kernels and arena-backed launchers for the chroma-format component.
//
// One CUDA block per job.  The kernels call the shared `__device__` bodies in
// `chroma_formats_device.cuh`; every device buffer is supplied by the caller
// (normally the Workspace accessors, which are arena-backed).
//
// Memory contract: the launchers below are the only code here that touches the
// arena.  Each stages its descriptor array in arena scratch with
// `Arena::allocate_checked`, launches, synchronises, then rewinds.  There is
// no cudaMalloc/cudaFree anywhere in the component.
#include "cuda/chroma_formats/chroma_formats.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "cuda/chroma_formats/chroma_formats_device.cuh"

namespace av2enc::cuda::chroma {
namespace {

constexpr int kKernelThreads = 1024;

// Host-side job validation.  These are ordinary host functions (never called
// from device code); device helpers all live in chroma_formats_device.cuh.
bool shape_ok(const TxShape& shape) {
  if (shape.adj_w < 1 || shape.adj_w > 32 || shape.adj_h < 1 ||
      shape.adj_h > 32) {
    return false;
  }
  if (shape.upscale_x < 1 || shape.upscale_x > 2 || shape.upscale_y < 1 ||
      shape.upscale_y > 2) {
    return false;
  }
  if (shape.nominal_w != shape.adj_w * shape.upscale_x ||
      shape.nominal_h != shape.adj_h * shape.upscale_y) {
    return false;
  }
  return true;
}

bool depth_ok(int bit_depth) { return bit_depth == 8 || bit_depth == 10; }

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void predict_kernel(const PredictJob* __restrict__ jobs) {
  const PredictJob job = jobs[blockIdx.x];
  __shared__ int s_pred;
  if (threadIdx.x == 0) {
    s_pred = detail::dc_predict_dev(job.recon, job.stride, job.max_x,
                                    job.max_y, job.x0, job.y0, job.nominal_w,
                                    job.nominal_h, job.have_left,
                                    job.have_above, job.bit_depth);
  }
  __syncthreads();
  const int count = job.nominal_w * job.nominal_h;
  for (int i = static_cast<int>(threadIdx.x); i < count;
       i += static_cast<int>(blockDim.x)) {
    job.pred[i] = static_cast<std::uint16_t>(s_pred);
  }
}

__global__ void residual_kernel(const ResidualJob* __restrict__ jobs) {
  const ResidualJob job = jobs[blockIdx.x];
  const int count = job.shape.adj_w * job.shape.adj_h;
  for (int i = static_cast<int>(threadIdx.x); i < count;
       i += static_cast<int>(blockDim.x)) {
    const int j = i / job.shape.adj_w;
    const int col = i - j * job.shape.adj_w;
    job.residual_adj[i] = detail::residual_sample_dev(
        job.src, job.src_stride, job.max_x, job.max_y, job.x0, job.y0,
        job.pred, job.shape, col, j);
  }
}

__global__ void forward_kernel(const ForwardJob* __restrict__ jobs) {
  const ForwardJob job = jobs[blockIdx.x];
  const int w = job.shape.adj_w;
  const int h = job.shape.adj_h;
  const int elems = w * h;
  __shared__ double s_tmp[1024];
  const int tid = static_cast<int>(threadIdx.x);
  const std::int16_t* kw = detail::dct_kernel_dev(w);
  const std::int16_t* kh = detail::dct_kernel_dev(h);

  if (tid < elems) {
    const int row = (tid / w) * w;
    const int v = tid - row;
    double sum = 0.0;
    for (int j = 0; j < w; ++j) {
      sum = detail::fp_add(
          sum, detail::fp_mul(static_cast<double>(kw[v * w + j]),
                              static_cast<double>(job.residual_adj[row + j])));
    }
    s_tmp[tid] = detail::fp_mul(sum, job.shape.scale_w);
  }
  __syncthreads();

  if (tid < elems) {
    const int u = tid / w;
    const int v = tid - u * w;
    double sum = 0.0;
    for (int i = 0; i < h; ++i) {
      sum = detail::fp_add(
          sum, detail::fp_mul(static_cast<double>(kh[u * h + i]),
                              s_tmp[i * w + v]));
    }
    double out = detail::fp_mul(sum, job.shape.scale_h);
    if (job.shape.odd_rect) {
      out = detail::fp_mul(out, job.shape.rect_prescale);
    }
    if (job.coeff != nullptr) {
      job.coeff[tid] = out;
    }
    job.levels[tid] =
        detail::lround_dev(detail::fp_mul(out, job.shape.quant_factor));
  }
}

__global__ void dequant_kernel(const DequantJob* __restrict__ jobs) {
  const DequantJob job = jobs[blockIdx.x];
  const int count = job.shape.adj_w * job.shape.adj_h;
  for (int i = static_cast<int>(threadIdx.x); i < count;
       i += static_cast<int>(blockDim.x)) {
    job.dequant[i] =
        detail::dequantize_level_dev(job.levels[i], job.shape.dq_denom,
                                     job.bit_depth);
  }
}

__global__ void inverse_kernel(const InverseJob* __restrict__ jobs) {
  const InverseJob job = jobs[blockIdx.x];
  const int w = job.shape.adj_w;
  const int h = job.shape.adj_h;
  const int elems = w * h;
  const int row_shift = job.shape.row_shift;
  const int col_shift = job.shape.col_shift;
  const bool odd_rect = job.shape.odd_rect;
  __shared__ std::int32_t s_inter[1024];
  const int tid = static_cast<int>(threadIdx.x);
  const std::int16_t* kw = detail::dct_kernel_dev(w);
  const std::int16_t* kh = detail::dct_kernel_dev(h);

  if (tid < elems) {
    const int row = (tid / w) * w;
    const int oj = tid - row;
    std::int64_t sum = 0;
    for (int j = 0; j < w; ++j) {
      sum += static_cast<std::int64_t>(kw[j * w + oj]) *
             detail::prescale_dev(job.dequant[row + j], odd_rect);
    }
    s_inter[tid] = detail::clip_inverse_row_dev(
        detail::round2_64_dev(sum, row_shift), job.bit_depth);
  }
  __syncthreads();

  if (tid < elems) {
    const int oi = tid / w;
    const int col = tid - oi * w;
    std::int64_t sum = 0;
    for (int i = 0; i < h; ++i) {
      sum += static_cast<std::int64_t>(kh[i * h + oi]) *
             s_inter[i * w + col];
    }
    job.residual_adj[tid] = static_cast<std::int16_t>(
        detail::clip_inverse_col_dev(
            detail::round2_64_dev(sum, col_shift), job.bit_depth));
  }
}

__global__ void reconstruct_kernel(const ReconstructJob* __restrict__ jobs) {
  const ReconstructJob job = jobs[blockIdx.x];
  const int elems = job.shape.adj_w * job.shape.adj_h;
  for (int i = static_cast<int>(threadIdx.x); i < elems;
       i += static_cast<int>(blockDim.x)) {
    const int j = i / job.shape.adj_w;
    const int col = i - j * job.shape.adj_w;
    const std::uint16_t value = detail::reconstruct_value_dev(
        job.pred, job.residual_adj[i], job.bit_depth);
    for (int dy = 0; dy < job.shape.upscale_y; ++dy) {
      const int py = job.y0 + j * job.shape.upscale_y + dy;
      if (py < 0 || py > job.max_y) continue;
      for (int dx = 0; dx < job.shape.upscale_x; ++dx) {
        const int px = job.x0 + col * job.shape.upscale_x + dx;
        if (px < 0 || px > job.max_x) continue;
        job.dst[py * job.dst_stride + px] = value;
      }
    }
  }
}

__global__ void copy_kernel(const CopyJob* __restrict__ jobs) {
  const CopyJob job = jobs[blockIdx.x];
  const int count = job.width * job.height;
  for (int i = static_cast<int>(threadIdx.x); i < count;
       i += static_cast<int>(blockDim.x)) {
    const int y = i / job.width;
    const int x = i - y * job.width;
    job.dst[y * job.dst_stride + x] = static_cast<std::uint16_t>(
        detail::clip1_dev(
            static_cast<int>(job.src[y * job.src_stride + x]),
            job.bit_depth));
  }
}

// ---------------------------------------------------------------------------
// Kernel thunks (one per job type)
// ---------------------------------------------------------------------------

void launch_predict(const PredictJob* jobs, int count, cudaStream_t stream) {
  predict_kernel<<<count, kKernelThreads, 0, stream>>>(jobs);
}
void launch_residual(const ResidualJob* jobs, int count, cudaStream_t stream) {
  residual_kernel<<<count, kKernelThreads, 0, stream>>>(jobs);
}
void launch_forward(const ForwardJob* jobs, int count, cudaStream_t stream) {
  forward_kernel<<<count, kKernelThreads, 0, stream>>>(jobs);
}
void launch_dequant(const DequantJob* jobs, int count, cudaStream_t stream) {
  dequant_kernel<<<count, kKernelThreads, 0, stream>>>(jobs);
}
void launch_inverse(const InverseJob* jobs, int count, cudaStream_t stream) {
  inverse_kernel<<<count, kKernelThreads, 0, stream>>>(jobs);
}
void launch_reconstruct(const ReconstructJob* jobs, int count,
                        cudaStream_t stream) {
  reconstruct_kernel<<<count, kKernelThreads, 0, stream>>>(jobs);
}
void launch_copy(const CopyJob* jobs, int count, cudaStream_t stream) {
  copy_kernel<<<count, kKernelThreads, 0, stream>>>(jobs);
}

// ---------------------------------------------------------------------------
// Generic arena-bracketed batch launcher
// ---------------------------------------------------------------------------

template <typename Job, typename Launch>
ChromaStatus run_batch(Arena& arena, const Config& config, const Job* jobs,
                       int count, cudaStream_t stream, const char* what,
                       Launch launch, std::string* error) {
  const std::uint64_t required = required_device_bytes(config);
  if (!config_valid(config)) {
    if (error != nullptr) *error = std::string(what) + ": invalid config";
    return ChromaStatus{ChromaStatus::Code::kInvalidArgument, 0};
  }
  if (jobs == nullptr || count < 1 || count > config.max_blocks) {
    if (error != nullptr) {
      *error = std::string(what) + ": job count outside [1, max_blocks]";
    }
    return ChromaStatus{ChromaStatus::Code::kInvalidArgument, required};
  }

  const std::uint64_t mark = arena.mark();
  void* raw = nullptr;
  const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(Job);
  const Status allocated = arena.allocate_checked(bytes, &raw);
  if (allocated != Status::kOk) {
    arena.rewind(mark);
    if (error != nullptr) {
      *error = std::string(what) + ": arena too small for " +
               std::to_string(bytes) + " descriptor bytes (need " +
               std::to_string(required) + ")";
    }
    return ChromaStatus{ChromaStatus::Code::kArenaTooSmall, required};
  }

  cudaError_t status = cudaMemcpyAsync(raw, jobs, bytes,
                                       cudaMemcpyHostToDevice, stream);
  if (status == cudaSuccess) {
    launch(static_cast<const Job*>(raw), count, stream);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) {
    status = cudaStreamSynchronize(stream);
  }
  arena.rewind(mark);

  if (status != cudaSuccess) {
    if (error != nullptr) {
      *error = std::string(what) + ": " + cudaGetErrorString(status);
    }
    return ChromaStatus{ChromaStatus::Code::kCudaError, required};
  }
  return ChromaStatus{};
}

}  // namespace

ChromaStatus Workspace::launch_error(const char* what, cudaError_t status,
                                     std::string* error) const {
  if (error != nullptr) {
    *error = std::string("chroma_formats ") + what + ": " +
             cudaGetErrorString(status);
  }
  return ChromaStatus{ChromaStatus::Code::kCudaError,
                      required_device_bytes(config_)};
}

// ---------------------------------------------------------------------------
// Batch entry points
// ---------------------------------------------------------------------------

ChromaStatus Workspace::predict(const PredictJob* host_jobs, int count,
                                cudaStream_t stream, std::string* error) {
  if (!initialized()) {
    return ChromaStatus{ChromaStatus::Code::kNotInitialized, 0};
  }
  if (host_jobs == nullptr || count < 1 || count > config_.max_blocks) {
    if (error != nullptr) {
      *error = "chroma_formats: job count outside [1, max_blocks]";
    }
    return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                        required_device_bytes(config_)};
  }
  for (int i = 0; i < count; ++i) {
    const PredictJob& job = host_jobs[i];
    if (job.recon == nullptr || job.pred == nullptr || job.stride <= 0 ||
        job.nominal_w < 4 || job.nominal_w > 64 || job.nominal_h < 4 ||
        job.nominal_h > 64 || !depth_ok(job.bit_depth) ||
        (job.have_left != 0 && job.x0 < 1) ||
        (job.have_above != 0 && job.y0 < 1) || job.x0 > job.max_x ||
        job.y0 > job.max_y) {
      if (error != nullptr) *error = "chroma_formats predict: invalid job";
      return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                          required_device_bytes(config_)};
    }
  }
  return run_batch(*arena_, config_, host_jobs, count, stream, "predict",
                   &launch_predict, error);
}

ChromaStatus Workspace::make_residuals(const ResidualJob* host_jobs, int count,
                                       cudaStream_t stream,
                                       std::string* error) {
  if (!initialized()) {
    return ChromaStatus{ChromaStatus::Code::kNotInitialized, 0};
  }
  if (host_jobs == nullptr || count < 1 || count > config_.max_blocks) {
    if (error != nullptr) {
      *error = "chroma_formats: job count outside [1, max_blocks]";
    }
    return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                        required_device_bytes(config_)};
  }
  for (int i = 0; i < count; ++i) {
    const ResidualJob& job = host_jobs[i];
    if (job.src == nullptr || job.residual_adj == nullptr ||
        job.src_stride <= 0 || !shape_ok(job.shape) ||
        !depth_ok(job.bit_depth)) {
      if (error != nullptr) {
        *error = "chroma_formats make_residuals: invalid job";
      }
      return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                          required_device_bytes(config_)};
    }
  }
  return run_batch(*arena_, config_, host_jobs, count, stream,
                   "make_residuals", &launch_residual, error);
}

ChromaStatus Workspace::forward(const ForwardJob* host_jobs, int count,
                                cudaStream_t stream, std::string* error) {
  if (!initialized()) {
    return ChromaStatus{ChromaStatus::Code::kNotInitialized, 0};
  }
  if (host_jobs == nullptr || count < 1 || count > config_.max_blocks) {
    if (error != nullptr) {
      *error = "chroma_formats: job count outside [1, max_blocks]";
    }
    return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                        required_device_bytes(config_)};
  }
  for (int i = 0; i < count; ++i) {
    const ForwardJob& job = host_jobs[i];
    if (job.residual_adj == nullptr || job.levels == nullptr ||
        !shape_ok(job.shape)) {
      if (error != nullptr) *error = "chroma_formats forward: invalid job";
      return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                          required_device_bytes(config_)};
    }
  }
  return run_batch(*arena_, config_, host_jobs, count, stream, "forward",
                   &launch_forward, error);
}

ChromaStatus Workspace::dequantize(const DequantJob* host_jobs, int count,
                                   cudaStream_t stream, std::string* error) {
  if (!initialized()) {
    return ChromaStatus{ChromaStatus::Code::kNotInitialized, 0};
  }
  if (host_jobs == nullptr || count < 1 || count > config_.max_blocks) {
    if (error != nullptr) {
      *error = "chroma_formats: job count outside [1, max_blocks]";
    }
    return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                        required_device_bytes(config_)};
  }
  for (int i = 0; i < count; ++i) {
    const DequantJob& job = host_jobs[i];
    if (job.levels == nullptr || job.dequant == nullptr ||
        !shape_ok(job.shape) || !depth_ok(job.bit_depth)) {
      if (error != nullptr) *error = "chroma_formats dequantize: invalid job";
      return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                          required_device_bytes(config_)};
    }
  }
  return run_batch(*arena_, config_, host_jobs, count, stream, "dequantize",
                   &launch_dequant, error);
}

ChromaStatus Workspace::inverse(const InverseJob* host_jobs, int count,
                                cudaStream_t stream, std::string* error) {
  if (!initialized()) {
    return ChromaStatus{ChromaStatus::Code::kNotInitialized, 0};
  }
  if (host_jobs == nullptr || count < 1 || count > config_.max_blocks) {
    if (error != nullptr) {
      *error = "chroma_formats: job count outside [1, max_blocks]";
    }
    return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                        required_device_bytes(config_)};
  }
  for (int i = 0; i < count; ++i) {
    const InverseJob& job = host_jobs[i];
    if (job.dequant == nullptr || job.residual_adj == nullptr ||
        !shape_ok(job.shape) || !depth_ok(job.bit_depth)) {
      if (error != nullptr) *error = "chroma_formats inverse: invalid job";
      return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                          required_device_bytes(config_)};
    }
  }
  return run_batch(*arena_, config_, host_jobs, count, stream, "inverse",
                   &launch_inverse, error);
}

ChromaStatus Workspace::reconstruct(const ReconstructJob* host_jobs, int count,
                                    cudaStream_t stream, std::string* error) {
  if (!initialized()) {
    return ChromaStatus{ChromaStatus::Code::kNotInitialized, 0};
  }
  if (host_jobs == nullptr || count < 1 || count > config_.max_blocks) {
    if (error != nullptr) {
      *error = "chroma_formats: job count outside [1, max_blocks]";
    }
    return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                        required_device_bytes(config_)};
  }
  for (int i = 0; i < count; ++i) {
    const ReconstructJob& job = host_jobs[i];
    if (job.residual_adj == nullptr || job.dst == nullptr ||
        job.dst_stride <= 0 || !shape_ok(job.shape) ||
        !depth_ok(job.bit_depth)) {
      if (error != nullptr) *error = "chroma_formats reconstruct: invalid job";
      return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                          required_device_bytes(config_)};
    }
  }
  return run_batch(*arena_, config_, host_jobs, count, stream, "reconstruct",
                   &launch_reconstruct, error);
}

ChromaStatus Workspace::copy_planes(const CopyJob* host_jobs, int count,
                                    cudaStream_t stream, std::string* error) {
  if (!initialized()) {
    return ChromaStatus{ChromaStatus::Code::kNotInitialized, 0};
  }
  if (host_jobs == nullptr || count < 1 || count > config_.max_blocks) {
    if (error != nullptr) {
      *error = "chroma_formats: job count outside [1, max_blocks]";
    }
    return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                        required_device_bytes(config_)};
  }
  for (int i = 0; i < count; ++i) {
    const CopyJob& job = host_jobs[i];
    if (job.src == nullptr || job.dst == nullptr || job.src_stride <= 0 ||
        job.dst_stride <= 0 || job.width < 0 || job.height < 0 ||
        !depth_ok(job.bit_depth)) {
      if (error != nullptr) *error = "chroma_formats copy_planes: invalid job";
      return ChromaStatus{ChromaStatus::Code::kInvalidArgument,
                          required_device_bytes(config_)};
    }
  }
  return run_batch(*arena_, config_, host_jobs, count, stream, "copy_planes",
                   &launch_copy, error);
}

}  // namespace av2enc::cuda::chroma
