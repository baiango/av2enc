// AV2 intra block copy — CPU/CUDA bit-exactness test.
//
// Build/run on the GPU box (sm_86):
//   nvcc -std=c++17 -arch=sm_86 -I../../../src ibc_test.cu \
//        ../../../src/encoder/ibc/ibc.cpp \
//        ../../../src/encoder/partition/partition.cpp -o ibc_gpu_test
//   ./ibc_gpu_test

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/ibc/ibc.h"
#include "encoder/ibc/ibc.h"

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

using av2enc::ibc::BlockVector;

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

void test_copy_and_sad() {
  auto screen = av2enc::ibc::make_synthetic_screen(384, 192);
  const auto view = screen.view();
  av2enc::cuda::IbcDeviceFrame dev;
  std::uint8_t* d_y = upload(screen.y);
  std::uint8_t* d_u = upload(screen.u);
  std::uint8_t* d_v = upload(screen.v);
  dev.y = d_y;
  dev.u = d_u;
  dev.v = d_v;
  dev.stride_y = view.y.stride;
  dev.stride_u = view.u.stride;
  dev.stride_v = view.v.stride;
  dev.width = view.y.width;
  dev.height = view.y.height;

  const int bsize = static_cast<int>(av2enc::partition::BlockSize::k64x32);
  const int mi_row = 8;
  const int mi_col = 80;
  const BlockVector bv{0, -320 * 8};
  std::vector<std::uint8_t> cpu_y(64 * 32);
  std::vector<std::uint8_t> cpu_u(32 * 16);
  std::vector<std::uint8_t> cpu_v(32 * 16);
  CHECK(av2enc::ibc::copy_block(view, mi_row, mi_col, bsize, bv, cpu_y.data(),
                                64, cpu_u.data(), 32, cpu_v.data(), 32, 64,
                                32),
        "CPU copy");
  std::uint8_t* d_dst_y = nullptr;
  std::uint8_t* d_dst_u = nullptr;
  std::uint8_t* d_dst_v = nullptr;
  CUDA_TRY(cudaMalloc(&d_dst_y, cpu_y.size()), "alloc dst y");
  CUDA_TRY(cudaMalloc(&d_dst_u, cpu_u.size()), "alloc dst u");
  CUDA_TRY(cudaMalloc(&d_dst_v, cpu_v.size()), "alloc dst v");
  CUDA_TRY(av2enc::cuda::ibc_copy_block(dev, mi_row, mi_col, bsize, bv, d_dst_y,
                                        64, d_dst_u, 32, d_dst_v, 32, 64, 32),
           "GPU copy");
  std::vector<std::uint8_t> gpu_y(cpu_y.size());
  std::vector<std::uint8_t> gpu_u(cpu_u.size());
  std::vector<std::uint8_t> gpu_v(cpu_v.size());
  CUDA_TRY(cudaMemcpy(gpu_y.data(), d_dst_y, gpu_y.size(),
                      cudaMemcpyDeviceToHost),
           "copy y");
  CUDA_TRY(cudaMemcpy(gpu_u.data(), d_dst_u, gpu_u.size(),
                      cudaMemcpyDeviceToHost),
           "copy u");
  CUDA_TRY(cudaMemcpy(gpu_v.data(), d_dst_v, gpu_v.size(),
                      cudaMemcpyDeviceToHost),
           "copy v");
  CHECK(gpu_y == cpu_y, "luma copy matches the CPU");
  CHECK(gpu_u == cpu_u, "U copy matches the CPU");
  CHECK(gpu_v == cpu_v, "V copy matches the CPU");

  // SAD batch parity.
  const std::vector<BlockVector> cands = {
      {0, -320 * 8}, {-64 * 8, 0}, {0, -256 * 8}, {0, -192 * 8}};
  std::vector<BlockVector> host_bvs = cands;
  BlockVector* d_bvs = upload(host_bvs);
  std::int64_t* d_sads = nullptr;
  CUDA_TRY(cudaMalloc(&d_sads, cands.size() * sizeof(std::int64_t)),
           "alloc sads");
  CUDA_TRY(av2enc::cuda::ibc_sad_batch(dev, mi_row, mi_col, bsize, d_bvs,
                                       static_cast<int>(cands.size()), d_sads),
           "sad batch");
  std::vector<std::int64_t> gpu_sads(cands.size());
  CUDA_TRY(cudaMemcpy(gpu_sads.data(), d_sads,
                      gpu_sads.size() * sizeof(std::int64_t),
                      cudaMemcpyDeviceToHost),
           "copy sads");
  int diff = 0;
  for (std::size_t i = 0; i < cands.size(); ++i) {
    const int dx = av2enc::ibc::plane_offset(cands[i].col, 0);
    const int dy = av2enc::ibc::plane_offset(cands[i].row, 0);
    const int sx = mi_col * 4 + dx;
    const int sy = mi_row * 4 + dy;
    if (sx < 0 || sy < 0) continue;
    const std::int64_t sad = av2enc::ibc::sad_block(
        view.y.data + static_cast<std::size_t>(sy) * view.y.stride +
            static_cast<std::size_t>(sx),
        view.y.stride,
        view.y.data + static_cast<std::size_t>(mi_row * 4) * view.y.stride +
            static_cast<std::size_t>(mi_col * 4),
        view.y.stride, 64, 32);
    if (sad != gpu_sads[i]) ++diff;
  }
  CHECK(diff == 0, "SAD batch matches the CPU");

  // Global validity parity.
  av2enc::cuda::IbcDeviceGeometry geom;
  geom.mi_cols = view.y.width / 4;
  geom.mi_rows = view.y.height / 4;
  geom.sb_size = 64;
  geom.allow_global = true;
  std::uint8_t* d_valid = nullptr;
  CUDA_TRY(cudaMalloc(&d_valid, cands.size()), "alloc valid");
  CUDA_TRY(av2enc::cuda::ibc_validate_global(
               geom, mi_row, mi_col, bsize, d_bvs,
               static_cast<int>(cands.size()), d_valid),
           "validate");
  std::vector<std::uint8_t> gpu_valid(cands.size());
  CUDA_TRY(cudaMemcpy(gpu_valid.data(), d_valid, gpu_valid.size(),
                      cudaMemcpyDeviceToHost),
           "copy valid");
  av2enc::ibc::IbcConfig config;
  config.allow_global = true;
  config.allow_local = false;
  av2enc::ibc::IbcBufferState state;
  state.reset(geom.mi_cols, geom.mi_rows);
  int valid_diff = 0;
  for (std::size_t i = 0; i < cands.size(); ++i) {
    const bool cpu = av2enc::ibc::is_bv_valid(view, config, state, mi_row,
                                              mi_col, bsize, cands[i]);
    if ((gpu_valid[i] != 0) != cpu) ++valid_diff;
  }
  CHECK(valid_diff == 0, "global BV validity matches the CPU");

  // Search parity (global only).
  av2enc::ibc::SearchOptions options;
  options.step = 4;
  const av2enc::ibc::Match cpu_match = av2enc::ibc::search_block(
      view, config, state, mi_row, mi_col, bsize, options);
  BlockVector gpu_bv;
  std::int64_t gpu_sad = 0;
  CUDA_TRY(av2enc::cuda::ibc_search(dev, geom, mi_row, mi_col, bsize, 4, &gpu_bv,
                                    &gpu_sad),
           "search");
  CHECK(cpu_match.found, "CPU search found a match");
  CHECK(gpu_sad == cpu_match.sad, "GPU search SAD matches the CPU");

  cudaFree(d_y);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_dst_y);
  cudaFree(d_dst_u);
  cudaFree(d_dst_v);
  cudaFree(d_bvs);
  cudaFree(d_sads);
  cudaFree(d_valid);
}

void test_plane_offset_table() {
  // The device and host core are the same inline function; this is a cheap
  // guard that the device pass did not reassociate the rounding.
  static constexpr int kCases[][3] = {
      {-2560, 0, -320}, {-2560, 1, -160}, {-2561, 0, -321},
      {-2561, 1, -161}, {0, 0, 0},        {8, 0, 1},
      {-8, 0, -1},      {24, 1, 1},
  };
  for (const auto& c : kCases) {
    CHECK(av2enc::ibc::plane_offset(c[0], c[1]) == c[2],
          "plane_offset table");
  }
}

}  // namespace

int main() {
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    std::fprintf(stderr, "no CUDA device\n");
    return 1;
  }
  std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  test_plane_offset_table();
  test_copy_and_sad();
  std::printf("ibc_gpu_test: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
