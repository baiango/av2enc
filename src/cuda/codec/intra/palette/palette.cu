// AV2 palette intra — CUDA kernels.  See palette.h for the contract.
#include "cuda/codec/intra/palette/palette.h"

#include "codec/intra/palette/core/palette_core.h"

namespace av2enc::cuda {
namespace {

using namespace av2enc::palette;

__global__ void histogram_kernel(const std::uint8_t* src, int stride, int w,
                                 int h, std::uint32_t* hist) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const std::uint8_t v =
      src[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
          static_cast<std::size_t>(x)];
  atomicAdd(&hist[v], 1u);
}

__global__ void build_map_kernel(PaletteDeviceBlock block) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= block.block_w || y >= block.block_h) return;
  std::uint8_t value = 0;
  if (x < block.visible_w && y < block.visible_h) {
    value = block.src[static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(block.src_stride) +
                      static_cast<std::size_t>(x)];
    if (block.dither) {
      // Ordered 4x4 dither, matching the CPU reference.
      const int kDither[4][4] = {
          {0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};
      const int v = static_cast<int>(value) +
                    (kDither[y & 3][x & 3] - 8) / 8;
      value = static_cast<std::uint8_t>(palette_clip1(v));
    }
    block.map[static_cast<std::size_t>(y) *
                  static_cast<std::size_t>(block.map_stride) +
              static_cast<std::size_t>(x)] =
        static_cast<std::uint8_t>(
            palette_nearest_color(block.colors, block.size, value));
    return;
  }
}

// §5.20.8.4 replication runs as a second kernel. Doing it in the first kernel
// would read map cells written by other threads in the same launch.
__global__ void replicate_map_kernel(PaletteDeviceBlock block) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= block.block_w || y >= block.block_h) return;
  if (x < block.visible_w && y < block.visible_h) return;
  const int sx = x < block.visible_w ? x : block.visible_w - 1;
  const int sy = y < block.visible_h ? y : block.visible_h - 1;
  block.map[static_cast<std::size_t>(y) *
                static_cast<std::size_t>(block.map_stride) +
            static_cast<std::size_t>(x)] =
      block.map[static_cast<std::size_t>(sy) *
                    static_cast<std::size_t>(block.map_stride) +
                static_cast<std::size_t>(sx)];
}

__global__ void predict_kernel(PaletteDeviceBlock block) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= block.block_w || y >= block.block_h) return;
  const int idx = block.map[static_cast<std::size_t>(y) *
                                static_cast<std::size_t>(block.map_stride) +
                            static_cast<std::size_t>(x)];
  block.dst[static_cast<std::size_t>(y) *
                static_cast<std::size_t>(block.dst_stride) +
            static_cast<std::size_t>(x)] = block.colors[idx];
}

__global__ void context_kernel(PaletteContextDevice block) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= block.width || y >= block.height) return;
  std::uint8_t order[8];
  const int hash = palette_color_context(block.map, block.map_stride, y, x,
                                         block.size, order);
  const std::size_t idx = static_cast<std::size_t>(y) *
                              static_cast<std::size_t>(block.width) +
                          static_cast<std::size_t>(x);
  block.hash[idx] = static_cast<std::uint8_t>(hash);
  for (int i = 0; i < 8; ++i) {
    block.order[idx * static_cast<std::size_t>(block.order_stride) +
                static_cast<std::size_t>(i)] = order[i];
  }
}

dim3 grid_for(int w, int h) {
  constexpr int kThreads = 16;
  return dim3(static_cast<unsigned>((w + kThreads - 1) / kThreads),
              static_cast<unsigned>((h + kThreads - 1) / kThreads), 1);
}

cudaError_t validate_block(const PaletteDeviceBlock& block) {
  if (block.src == nullptr || block.colors == nullptr || block.map == nullptr ||
      block.size < 2 || block.size > kPaletteMaxSize || block.block_w <= 0 ||
      block.block_h <= 0 || block.visible_w <= 0 || block.visible_h <= 0 ||
      block.visible_w > block.block_w || block.visible_h > block.block_h) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t palette_histogram(const std::uint8_t* src, int stride, int w,
                              int h, std::uint32_t* hist,
                              cudaStream_t stream) {
  if (src == nullptr || hist == nullptr || w <= 0 || h <= 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(16, 16, 1);
  const dim3 grid = grid_for(w, h);
  histogram_kernel<<<grid, block, 0, stream>>>(src, stride, w, h, hist);
  return cudaGetLastError();
}

cudaError_t palette_build_map(const PaletteDeviceBlock& block,
                              cudaStream_t stream) {
  const cudaError_t status = validate_block(block);
  if (status != cudaSuccess) return status;
  const dim3 thread(16, 16, 1);
  const dim3 grid = grid_for(block.block_w, block.block_h);
  build_map_kernel<<<grid, thread, 0, stream>>>(block);
  cudaError_t status2 = cudaGetLastError();
  if (status2 != cudaSuccess) return status2;
  replicate_map_kernel<<<grid, thread, 0, stream>>>(block);
  return cudaGetLastError();
}

cudaError_t palette_predict(const PaletteDeviceBlock& block,
                            cudaStream_t stream) {
  const cudaError_t status = validate_block(block);
  if (status != cudaSuccess) return status;
  if (block.dst == nullptr) return cudaErrorInvalidValue;
  const dim3 thread(16, 16, 1);
  const dim3 grid = grid_for(block.block_w, block.block_h);
  predict_kernel<<<grid, thread, 0, stream>>>(block);
  return cudaGetLastError();
}

cudaError_t palette_color_contexts(const PaletteContextDevice& block,
                                   cudaStream_t stream) {
  if (block.map == nullptr || block.hash == nullptr || block.order == nullptr ||
      block.width <= 0 || block.height <= 0 || block.size < 2 ||
      block.size > kPaletteMaxSize || block.order_stride < 8) {
    return cudaErrorInvalidValue;
  }
  const dim3 thread(16, 16, 1);
  const dim3 grid = grid_for(block.width, block.height);
  context_kernel<<<grid, thread, 0, stream>>>(block);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
