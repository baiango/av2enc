// CUDA implementation of the high-bit-depth sample path.
//
// See high_bitdepth.h.  No Blackwell-only features; plain sm_86-compatible
// kernels with one thread per sample/padding position.

#include "high_bitdepth.h"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::hb {
namespace {

__global__ void expand_8_to_16_kernel(const std::uint8_t* src,
                                      std::uint16_t* dst, int width,
                                      int height, int src_stride_bytes,
                                      int dst_stride_samples, int shift) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }
  const std::uint32_t value =
      static_cast<std::uint32_t>(src[static_cast<std::size_t>(y) *
                                         static_cast<std::size_t>(src_stride_bytes) +
                                     static_cast<std::size_t>(x)]);
  dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_stride_samples) +
      static_cast<std::size_t>(x)] =
      static_cast<std::uint16_t>(value << static_cast<unsigned>(shift));
}

__global__ void reduce_16_to_8_kernel(const std::uint16_t* src,
                                      std::uint8_t* dst, int width, int height,
                                      int src_stride_samples,
                                      int dst_stride_bytes, int bit_depth) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }
  const std::uint32_t value =
      src[static_cast<std::size_t>(y) * static_cast<std::size_t>(src_stride_samples) +
          static_cast<std::size_t>(x)];
  std::uint32_t rounded = value;
  if (bit_depth > 8) {
    const int shift = bit_depth - 8;
    const std::uint32_t bias = 1U << static_cast<unsigned>(shift - 1);
    rounded = (value + bias) >> static_cast<unsigned>(shift);
  }
  dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_stride_bytes) +
      static_cast<std::size_t>(x)] =
      static_cast<std::uint8_t>(rounded > 255U ? 255U : rounded);
}

__global__ void replicate_borders_16_kernel(std::uint16_t* plane, int stride,
                                            int coded_width, int coded_height,
                                            int visible_width,
                                            int visible_height) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= coded_width || y >= coded_height) {
    return;
  }
  if (x < visible_width && y < visible_height) {
    return;
  }
  const int source_x = x < visible_width ? x : visible_width - 1;
  const int source_y = y < visible_height ? y : visible_height - 1;
  plane[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
        static_cast<std::size_t>(x)] =
      plane[static_cast<std::size_t>(source_y) * static_cast<std::size_t>(stride) +
            static_cast<std::size_t>(source_x)];
}

dim3 grid_for(int width, int height) {
  constexpr int kBlockX = 32;
  constexpr int kBlockY = 8;
  const int blocks_x = (width + kBlockX - 1) / kBlockX;
  const int blocks_y = (height + kBlockY - 1) / kBlockY;
  return dim3(static_cast<unsigned>(blocks_x > 0 ? blocks_x : 1),
              static_cast<unsigned>(blocks_y > 0 ? blocks_y : 1));
}

}  // namespace

cudaError_t expand_8_to_16(const std::uint8_t* src, std::uint16_t* dst,
                           int width, int height, int src_stride_bytes,
                           int dst_stride_samples, int shift,
                           cudaStream_t stream) {
  if (width <= 0 || height <= 0) {
    return cudaSuccess;
  }
  if (shift < 0 || shift > 4 || src_stride_bytes < width ||
      dst_stride_samples < width) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(32, 8);
  const dim3 grid = grid_for(width, height);
  expand_8_to_16_kernel<<<grid, block, 0, stream>>>(
      src, dst, width, height, src_stride_bytes, dst_stride_samples, shift);
  return cudaGetLastError();
}

cudaError_t reduce_16_to_8(const std::uint16_t* src, std::uint8_t* dst,
                           int width, int height, int src_stride_samples,
                           int dst_stride_bytes, int bit_depth,
                           cudaStream_t stream) {
  if (width <= 0 || height <= 0) {
    return cudaSuccess;
  }
  if (bit_depth < 8 || bit_depth > 12 || src_stride_samples < width ||
      dst_stride_bytes < width) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(32, 8);
  const dim3 grid = grid_for(width, height);
  reduce_16_to_8_kernel<<<grid, block, 0, stream>>>(
      src, dst, width, height, src_stride_samples, dst_stride_bytes, bit_depth);
  return cudaGetLastError();
}

cudaError_t replicate_borders_16(std::uint16_t* plane, int stride,
                                 int coded_width, int coded_height,
                                 int visible_width, int visible_height,
                                 cudaStream_t stream) {
  if (coded_width <= 0 || coded_height <= 0) {
    return cudaSuccess;
  }
  if (visible_width <= 0 || visible_height <= 0) {
    return cudaSuccess;
  }
  if (stride < coded_width || visible_width > coded_width ||
      visible_height > coded_height) {
    return cudaErrorInvalidValue;
  }
  if (visible_width == coded_width && visible_height == coded_height) {
    return cudaSuccess;
  }
  const dim3 block(32, 8);
  const dim3 grid = grid_for(coded_width, coded_height);
  replicate_borders_16_kernel<<<grid, block, 0, stream>>>(
      plane, stride, coded_width, coded_height, visible_width, visible_height);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::hb
