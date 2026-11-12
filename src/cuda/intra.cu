// M0 DC intra prediction / residual / reconstruction kernels.
//
// Integer only, so the CUDA path is bit-exact with the scalar CPU reference in
// src/encoder/basic_encoder.cpp. See intra.h for the arithmetic contract.

#include "intra.h"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda {
namespace {

constexpr int kMaxWarps = 8;  // 256 threads per CUDA block

// ---------------------------------------------------------------------------
// Scalar arithmetic matching §4.8 and §7.13.2.11 as used by the CPU reference.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int clip3i(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ int clip1i(int value) {
  return clip3i(0, 255, value);
}

__device__ __forceinline__ int round2i(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

__device__ __forceinline__ int floor_log2i(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

__device__ __forceinline__ int approx_dividei(int numerator, int denominator) {
  const int shift = 9 + floor_log2i(denominator);
  return round2i(numerator * 512, shift);
}

// CPU: DC_PRED with no neighbours is 128; two neighbours use approx_divide,
// one neighbour uses Round2 with floor_log2(width|height).
__device__ __forceinline__ int dc_pred_from_sum(int sum, const DcBlock& b) {
  if (b.have_left != 0 && b.have_above != 0) {
    return clip1i(approx_dividei(sum, b.block_w + b.block_h));
  }
  if (b.have_left != 0) {
    return round2i(sum, floor_log2i(b.block_h));
  }
  if (b.have_above != 0) {
    return round2i(sum, floor_log2i(b.block_w));
  }
  return 128;
}

// ---------------------------------------------------------------------------
// Block-wide integer sum over the left column and above row, with edge
// clamping to the visible plane bounds.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int neighbour_sum(const DcBlock& b) {
  int sum = 0;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  if (b.have_left != 0) {
    const std::size_t stride = static_cast<std::size_t>(b.recon_stride);
    const std::size_t col = static_cast<std::size_t>(b.x0 - 1);
    for (int i = tid; i < b.block_h; i += nthr) {
      const int yy = clip3i(0, b.plane_max_y, b.y0 + i);
      sum += static_cast<int>(
          b.recon_plane[static_cast<std::size_t>(yy) * stride + col]);
    }
  }
  if (b.have_above != 0) {
    const std::size_t stride = static_cast<std::size_t>(b.recon_stride);
    const std::size_t row = static_cast<std::size_t>(b.y0 - 1);
    for (int j = tid; j < b.block_w; j += nthr) {
      const int xx = clip3i(0, b.plane_max_x, b.x0 + j);
      sum += static_cast<int>(
          b.recon_plane[row * stride + static_cast<std::size_t>(xx)]);
    }
  }
  return sum;
}

__device__ __forceinline__ int block_reduce_sum(int value, int* s_warp) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  if (lane == 0) {
    s_warp[warp] = value;
  }
  __syncthreads();
  const int warps = (static_cast<int>(blockDim.x) + 31) >> 5;
  value = static_cast<int>(threadIdx.x) < warps ? s_warp[threadIdx.x] : 0;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
  }
  return value;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void dc_pred_value_kernel(DcBlock b, int* __restrict__ d_pred) {
  __shared__ int s_warp[kMaxWarps];
  const int sum = block_reduce_sum(neighbour_sum(b), s_warp);
  if (threadIdx.x == 0) {
    *d_pred = dc_pred_from_sum(sum, b);
  }
}

__global__ void dc_predict_kernel(DcBlock b,
                                  std::uint8_t* __restrict__ pred_block) {
  __shared__ int s_warp[kMaxWarps];
  __shared__ int s_pred;
  const int sum = block_reduce_sum(neighbour_sum(b), s_warp);
  if (threadIdx.x == 0) {
    s_pred = dc_pred_from_sum(sum, b);
  }
  __syncthreads();
  const int n = b.block_w * b.block_h;
  const int nthr = static_cast<int>(blockDim.x);
  for (int i = static_cast<int>(threadIdx.x); i < n; i += nthr) {
    pred_block[i] = static_cast<std::uint8_t>(s_pred);
  }
}

__device__ __forceinline__ void predict_residual_body(
    const DcBlock& b, std::uint8_t* __restrict__ pred_block,
    std::int16_t* __restrict__ residual_block, int* s_warp, int* s_pred) {
  const int sum = block_reduce_sum(neighbour_sum(b), s_warp);
  if (threadIdx.x == 0) {
    *s_pred = dc_pred_from_sum(sum, b);
  }
  __syncthreads();

  const int n = b.block_w * b.block_h;
  const int nthr = static_cast<int>(blockDim.x);
  const std::size_t source_stride = static_cast<std::size_t>(b.source_stride);
  for (int i = static_cast<int>(threadIdx.x); i < n; i += nthr) {
    const int px = i % b.block_w;
    const int py = i / b.block_w;
    const int sx = clip3i(0, b.source_max_x, b.x0 + px);
    const int sy = clip3i(0, b.source_max_y, b.y0 + py);
    const int pred = *s_pred;
    const std::uint8_t sample =
        b.source_plane[static_cast<std::size_t>(sy) * source_stride +
                       static_cast<std::size_t>(sx)];
    if (pred_block != nullptr) {
      pred_block[i] = static_cast<std::uint8_t>(pred);
    }
    residual_block[i] = static_cast<std::int16_t>(static_cast<int>(sample) - pred);
  }
}

__global__ void dc_predict_residual_kernel(
    DcBlock b, std::uint8_t* __restrict__ pred_block,
    std::int16_t* __restrict__ residual_block) {
  __shared__ int s_warp[kMaxWarps];
  __shared__ int s_pred;
  predict_residual_body(b, pred_block, residual_block, s_warp, &s_pred);
}

__global__ void dc_residual_kernel(DcBlock b,
                                   const std::uint8_t* __restrict__ pred_block,
                                   std::int16_t* __restrict__ residual_block) {
  __shared__ int s_warp[kMaxWarps];
  __shared__ int s_pred;
  const int sum = block_reduce_sum(neighbour_sum(b), s_warp);
  if (threadIdx.x == 0) {
    s_pred = dc_pred_from_sum(sum, b);
  }
  __syncthreads();

  const int n = b.block_w * b.block_h;
  const int nthr = static_cast<int>(blockDim.x);
  const std::size_t source_stride = static_cast<std::size_t>(b.source_stride);
  for (int i = static_cast<int>(threadIdx.x); i < n; i += nthr) {
    const int px = i % b.block_w;
    const int py = i / b.block_w;
    const int sx = clip3i(0, b.source_max_x, b.x0 + px);
    const int sy = clip3i(0, b.source_max_y, b.y0 + py);
    const int pred = pred_block != nullptr ? static_cast<int>(pred_block[i]) : s_pred;
    const std::uint8_t sample =
        b.source_plane[static_cast<std::size_t>(sy) * source_stride +
                       static_cast<std::size_t>(sx)];
    residual_block[i] = static_cast<std::int16_t>(static_cast<int>(sample) - pred);
  }
}

__global__ void dc_predict_residual_batch_kernel(
    const DcBlockJob* __restrict__ jobs) {
  __shared__ int s_warp[kMaxWarps];
  __shared__ int s_pred;
  const DcBlockJob job = jobs[blockIdx.x];
  predict_residual_body(job.block, job.pred_block, job.residual_block, s_warp,
                        &s_pred);
}

__global__ void add_residual_clip_kernel(
    const std::uint8_t* __restrict__ pred_block,
    const std::int16_t* __restrict__ residual_block,
    std::uint8_t* __restrict__ recon_block, int n) {
  const int nthr = static_cast<int>(blockDim.x);
  for (int i = static_cast<int>(threadIdx.x); i < n; i += nthr) {
    const int value =
        static_cast<int>(pred_block[i]) + static_cast<int>(residual_block[i]);
    recon_block[i] = static_cast<std::uint8_t>(clip1i(value));
  }
}

__global__ void add_residual_clip_to_plane_kernel(
    const std::uint8_t* __restrict__ pred_block,
    const std::int16_t* __restrict__ residual_block,
    std::uint8_t* __restrict__ plane, int stride, int x0, int y0, int block_w,
    int block_h, int plane_max_x, int plane_max_y) {
  const int n = block_w * block_h;
  const int nthr = static_cast<int>(blockDim.x);
  const std::size_t row_stride = static_cast<std::size_t>(stride);
  for (int i = static_cast<int>(threadIdx.x); i < n; i += nthr) {
    const int x = x0 + i % block_w;
    const int y = y0 + i / block_w;
    if (x > plane_max_x || y > plane_max_y) {
      continue;
    }
    const int value =
        static_cast<int>(pred_block[i]) + static_cast<int>(residual_block[i]);
    plane[static_cast<std::size_t>(y) * row_stride +
          static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(clip1i(value));
  }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

int intra_threads(int work_items) {
  int threads = ((work_items + 31) / 32) * 32;
  if (threads < 32) {
    threads = 32;
  }
  if (threads > 256) {
    threads = 256;
  }
  return threads;
}

bool valid_block(const DcBlock& b, bool require_source) {
  if (b.block_w <= 0 || b.block_w > kIntraMaxBlockDim || b.block_h <= 0 ||
      b.block_h > kIntraMaxBlockDim) {
    return false;
  }
  if (b.x0 < 0 || b.y0 < 0 || b.plane_max_x < b.x0 ||
      b.plane_max_y < b.y0) {
    return false;
  }
  if (b.have_left != 0 &&
      (b.recon_plane == nullptr || b.recon_stride <= 0 || b.x0 < 1)) {
    return false;
  }
  if (b.have_above != 0 &&
      (b.recon_plane == nullptr || b.recon_stride <= 0 || b.y0 < 1)) {
    return false;
  }
  if (b.recon_plane != nullptr && b.recon_stride <= 0) {
    return false;
  }
  if (require_source) {
    if (b.source_plane == nullptr || b.source_stride <= 0 ||
        b.source_max_x < 0 || b.source_max_y < 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

cudaError_t dc_pred_value(const DcBlock& block, int* d_pred,
                          cudaStream_t stream) {
  if (!valid_block(block, false) || d_pred == nullptr) {
    return cudaErrorInvalidValue;
  }
  dc_pred_value_kernel<<<1, intra_threads(block.block_w * block.block_h), 0,
                         stream>>>(block, d_pred);
  return cudaGetLastError();
}

cudaError_t dc_predict(const DcBlock& block, std::uint8_t* pred_block,
                       cudaStream_t stream) {
  if (!valid_block(block, false) || pred_block == nullptr) {
    return cudaErrorInvalidValue;
  }
  dc_predict_kernel<<<1, intra_threads(block.block_w * block.block_h), 0,
                      stream>>>(block, pred_block);
  return cudaGetLastError();
}

cudaError_t dc_residual(const DcBlock& block, const std::uint8_t* pred_block,
                        std::int16_t* residual_block, cudaStream_t stream) {
  if (!valid_block(block, true) || residual_block == nullptr) {
    return cudaErrorInvalidValue;
  }
  dc_residual_kernel<<<1, intra_threads(block.block_w * block.block_h), 0,
                       stream>>>(block, pred_block, residual_block);
  return cudaGetLastError();
}

cudaError_t dc_predict_residual(const DcBlock& block,
                                std::uint8_t* pred_block,
                                std::int16_t* residual_block,
                                cudaStream_t stream) {
  if (!valid_block(block, true) || pred_block == nullptr ||
      residual_block == nullptr) {
    return cudaErrorInvalidValue;
  }
  dc_predict_residual_kernel<<<1, intra_threads(block.block_w * block.block_h),
                               0, stream>>>(block, pred_block, residual_block);
  return cudaGetLastError();
}

cudaError_t dc_predict_residual_batch(const DcBlockJob* d_jobs, int job_count,
                                      cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && d_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) {
    return cudaSuccess;
  }
  dc_predict_residual_batch_kernel<<<job_count, 256, 0, stream>>>(d_jobs);
  return cudaGetLastError();
}

cudaError_t add_residual_clip(const std::uint8_t* pred_block,
                              const std::int16_t* residual_block,
                              std::uint8_t* recon_block, int block_w,
                              int block_h, cudaStream_t stream) {
  if (pred_block == nullptr || residual_block == nullptr ||
      recon_block == nullptr || block_w <= 0 ||
      block_w > kIntraMaxBlockDim || block_h <= 0 ||
      block_h > kIntraMaxBlockDim) {
    return cudaErrorInvalidValue;
  }
  const int n = block_w * block_h;
  add_residual_clip_kernel<<<1, intra_threads(n), 0, stream>>>(
      pred_block, residual_block, recon_block, n);
  return cudaGetLastError();
}

cudaError_t add_residual_clip_to_plane(
    const std::uint8_t* pred_block, const std::int16_t* residual_block,
    std::uint8_t* plane, int stride, int x0, int y0, int block_w, int block_h,
    int plane_max_x, int plane_max_y, cudaStream_t stream) {
  if (pred_block == nullptr || residual_block == nullptr || plane == nullptr ||
      stride <= 0 || x0 < 0 || y0 < 0 || block_w <= 0 ||
      block_w > kIntraMaxBlockDim || block_h <= 0 ||
      block_h > kIntraMaxBlockDim || plane_max_x < x0 || plane_max_y < y0) {
    return cudaErrorInvalidValue;
  }
  const int n = block_w * block_h;
  add_residual_clip_to_plane_kernel<<<1, intra_threads(n), 0, stream>>>(
      pred_block, residual_block, plane, stride, x0, y0, block_w, block_h,
      plane_max_x, plane_max_y);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
