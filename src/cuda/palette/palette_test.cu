// AV2 palette intra — CPU/CUDA bit-exactness test.
//
// The kernels and the CPU reference call the same palette_core.h functions, so
// this test checks indexing, strides, edge replication and context hashing.
//
// Build/run on the GPU box:
//   nvcc -std=c++17 -arch=sm_86 -I../../../src palette_test.cu \
//        ../../../src/encoder/palette/palette.cpp \
//        ../../../src/entropy/av2/av2_entropy.cpp -o palette_gpu_test
//   ./palette_gpu_test

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/palette/palette.h"
#include "encoder/palette/palette.h"

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
    }                                                                      \
  } while (0)

#define CUDA_TRY(expr, msg)                                                \
  do {                                                                     \
    const cudaError_t err__ = (expr);                                      \
    ++g_checks;                                                            \
    if (err__ != cudaSuccess) {                                            \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__,    \
                   (msg), cudaGetErrorString(err__));                      \
    }                                                                      \
  } while (0)

std::vector<std::uint8_t> make_pattern(int w, int h, int colours) {
  std::vector<std::uint8_t> y(static_cast<std::size_t>(w) *
                                  static_cast<std::size_t>(h),
                              0);
  static constexpr std::uint8_t kColors[8] = {0, 32, 90, 160, 200, 230, 250,
                                              255};
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      const int band = (r * colours) / h;
      y[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) +
        static_cast<std::size_t>(c)] = kColors[band % 8];
      if (r > h / 2 && c > w / 2) {
        y[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) +
          static_cast<std::size_t>(c)] = kColors[(band + 1) % colours];
      }
    }
  }
  return y;
}

template <typename T>
T* upload(const std::vector<T>& host) {
  void* dev = nullptr;
  if (cudaMalloc(&dev, host.size() * sizeof(T)) != cudaSuccess) return nullptr;
  if (cudaMemcpy(dev, host.data(), host.size() * sizeof(T),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(dev);
    return nullptr;
  }
  return static_cast<T*>(dev);
}

void test_block(int w, int h, int visible_w, int visible_h, int colours,
                bool dither) {
  const auto src = make_pattern(w, h, colours);
  av2enc::palette::PaletteParams params;
  params.dither = dither;
  av2enc::palette::PaletteResult cpu;
  std::string error;
  CHECK(av2enc::palette::analyze_palette(
            src.data(), w, w, h, visible_w, visible_h, nullptr, 0, params, cpu,
            error),
        "CPU analyze");

  std::uint8_t* d_src = upload(src);
  std::uint8_t* d_colors = upload(std::vector<std::uint8_t>(
      cpu.colors.begin(), cpu.colors.begin() + cpu.size));
  std::uint8_t* d_map = nullptr;
  std::uint8_t* d_dst = nullptr;
  CUDA_TRY(cudaMalloc(&d_map, static_cast<std::size_t>(w) * h),
           "alloc map");
  CUDA_TRY(cudaMalloc(&d_dst, static_cast<std::size_t>(w) * h),
           "alloc dst");

  av2enc::cuda::PaletteDeviceBlock block;
  block.src = d_src;
  block.src_stride = w;
  block.block_w = w;
  block.block_h = h;
  block.visible_w = visible_w;
  block.visible_h = visible_h;
  block.colors = d_colors;
  block.size = cpu.size;
  block.map = d_map;
  block.map_stride = w;
  block.dst = d_dst;
  block.dst_stride = w;
  block.dither = dither;
  CUDA_TRY(av2enc::cuda::palette_build_map(block), "build_map");
  CUDA_TRY(cudaDeviceSynchronize(), "sync map");
  std::vector<std::uint8_t> gpu_map(static_cast<std::size_t>(w) * h);
  CUDA_TRY(cudaMemcpy(gpu_map.data(), d_map, gpu_map.size(),
                      cudaMemcpyDeviceToHost),
           "copy map");
  CHECK(gpu_map == cpu.color_map, "index map matches the CPU");

  CUDA_TRY(av2enc::cuda::palette_predict(block), "predict");
  CUDA_TRY(cudaDeviceSynchronize(), "sync predict");
  std::vector<std::uint8_t> gpu_pred(static_cast<std::size_t>(w) * h);
  CUDA_TRY(cudaMemcpy(gpu_pred.data(), d_dst, gpu_pred.size(),
                      cudaMemcpyDeviceToHost),
           "copy pred");
  std::vector<std::uint8_t> cpu_pred(static_cast<std::size_t>(w) * h);
  av2enc::palette::predict_palette_block(cpu, w, h, cpu_pred.data(), w);
  CHECK(gpu_pred == cpu_pred, "prediction matches the CPU");

  // Colour contexts.
  std::uint8_t* d_hash = nullptr;
  std::uint8_t* d_order = nullptr;
  CUDA_TRY(cudaMalloc(&d_hash, static_cast<std::size_t>(w) * h), "alloc hash");
  CUDA_TRY(cudaMalloc(&d_order, static_cast<std::size_t>(w) * h * 8),
           "alloc order");
  av2enc::cuda::PaletteContextDevice ctx;
  ctx.map = d_map;
  ctx.map_stride = w;
  ctx.width = w;
  ctx.height = h;
  ctx.size = cpu.size;
  ctx.hash = d_hash;
  ctx.order = d_order;
  ctx.order_stride = 8;
  CUDA_TRY(av2enc::cuda::palette_color_contexts(ctx), "contexts");
  CUDA_TRY(cudaDeviceSynchronize(), "sync contexts");
  std::vector<std::uint8_t> g_hash(static_cast<std::size_t>(w) * h);
  std::vector<std::uint8_t> g_order(static_cast<std::size_t>(w) * h * 8);
  CUDA_TRY(cudaMemcpy(g_hash.data(), d_hash, g_hash.size(),
                      cudaMemcpyDeviceToHost),
           "copy hash");
  CUDA_TRY(cudaMemcpy(g_order.data(), d_order, g_order.size(),
                      cudaMemcpyDeviceToHost),
           "copy order");
  int hash_diff = 0;
  int order_diff = 0;
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      std::uint8_t order[8];
      const int hash = av2enc::palette::palette_color_context(
          cpu.color_map.data(), w, r, c, cpu.size, order);
      const std::size_t i = static_cast<std::size_t>(r) * w + c;
      if (g_hash[i] != hash) ++hash_diff;
      for (int k = 0; k < 8; ++k) {
        if (g_order[i * 8 + static_cast<std::size_t>(k)] != order[k]) {
          ++order_diff;
        }
      }
    }
  }
  CHECK(hash_diff == 0, "colour-context hash matches the CPU");
  CHECK(order_diff == 0, "ColorOrder matches the CPU");

  // Histogram.
  std::uint32_t* d_hist = nullptr;
  CUDA_TRY(cudaMalloc(&d_hist, 256 * sizeof(std::uint32_t)), "alloc hist");
  CUDA_TRY(cudaMemset(d_hist, 0, 256 * sizeof(std::uint32_t)), "zero hist");
  CUDA_TRY(av2enc::cuda::palette_histogram(d_src, w, visible_w, visible_h, d_hist),
           "histogram");
  CUDA_TRY(cudaDeviceSynchronize(), "sync hist");
  std::vector<std::uint32_t> hist(256);
  CUDA_TRY(cudaMemcpy(hist.data(), d_hist, 256 * sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost),
           "copy hist");
  std::vector<std::uint32_t> cpu_hist(256, 0);
  for (int r = 0; r < visible_h; ++r) {
    for (int c = 0; c < visible_w; ++c) {
      ++cpu_hist[src[static_cast<std::size_t>(r) * w + c]];
    }
  }
  CHECK(hist == cpu_hist, "histogram matches the CPU");

  cudaFree(d_src);
  cudaFree(d_colors);
  cudaFree(d_map);
  cudaFree(d_dst);
  cudaFree(d_hash);
  cudaFree(d_order);
  cudaFree(d_hist);
}

}  // namespace

int main() {
  test_block(8, 8, 8, 8, 4, false);
  test_block(16, 16, 16, 16, 3, false);
  test_block(32, 32, 32, 32, 5, false);
  test_block(64, 64, 64, 64, 4, false);
  test_block(64, 32, 64, 32, 3, true);
  test_block(32, 64, 32, 64, 3, false);
  test_block(64, 64, 40, 20, 4, false);
  test_block(64, 64, 63, 17, 3, false);
  std::printf("palette_gpu_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
