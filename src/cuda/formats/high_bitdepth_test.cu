// CUDA high-bit-depth test: CPU vs GPU bit-exact for expand/reduce/borders.
//
// Build/run on the CUDA box (sm_86):
//   ./build_high_bitdepth_test.sh
//   ./high_bitdepth_cuda_test
//
// The CPU reference is the real component in src/encoder/format/format.cpp,
// linked into this binary.  The test compares every output sample, not a
// checksum, and exercises non-tight strides.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cuda/formats/high_bitdepth.h"
#include "formats/format.h"

namespace {

namespace fmt = av2enc::fmt;
namespace hb = av2enc::cuda::hb;

int g_failures = 0;
long long g_checks = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      ++g_failures;                                                          \
      std::printf("%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,           \
                  #condition);                                               \
    }                                                                        \
    ++g_checks;                                                              \
  } while (false)

#define CUDA_TRY(expr)                                                       \
  do {                                                                       \
    const cudaError_t err_ = (expr);                                         \
    if (err_ != cudaSuccess) {                                               \
      std::printf("%s:%d: CUDA error: %s\n", __FILE__, __LINE__,             \
                  cudaGetErrorString(err_));                                 \
      std::exit(1);                                                          \
    }                                                                        \
  } while (false)

void test_expand_8_to_16() {
  constexpr int kWidth = 67;
  constexpr int kHeight = 41;
  constexpr int kSrcStride = kWidth + 3;
  constexpr int kDstStride = kWidth + 5;

  std::mt19937 rng(0x5EEDu);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::vector<std::uint8_t> src(static_cast<std::size_t>(kSrcStride) * kHeight);
  for (std::uint8_t& value : src) {
    value = static_cast<std::uint8_t>(byte_dist(rng));
  }

  for (const int shift : {0, 2, 4}) {
    std::vector<std::uint16_t> expected(
        static_cast<std::size_t>(kDstStride) * kHeight, 0);
    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        expected[static_cast<std::size_t>(y) * kDstStride + x] =
            static_cast<std::uint16_t>(
                static_cast<std::uint32_t>(
                    src[static_cast<std::size_t>(y) * kSrcStride + x])
                << static_cast<unsigned>(shift));
      }
    }

    std::uint8_t* dev_src = nullptr;
    std::uint16_t* dev_dst = nullptr;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev_src), src.size()));
    CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&dev_dst),
        static_cast<std::size_t>(kDstStride) * kHeight *
            sizeof(std::uint16_t)));
    CUDA_TRY(cudaMemcpy(dev_src, src.data(), src.size(),
                        cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemset(dev_dst, 0,
                        static_cast<std::size_t>(kDstStride) * kHeight *
                            sizeof(std::uint16_t)));
    CUDA_TRY(hb::expand_8_to_16(dev_src, dev_dst, kWidth, kHeight, kSrcStride,
                                kDstStride, shift));
    CUDA_TRY(cudaDeviceSynchronize());

    std::vector<std::uint16_t> actual(
        static_cast<std::size_t>(kDstStride) * kHeight, 0);
    CUDA_TRY(cudaMemcpy(actual.data(), dev_dst, actual.size() * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < expected.size(); ++i) {
      CHECK(actual[i] == expected[i]);
    }
    CUDA_TRY(cudaFree(dev_src));
    CUDA_TRY(cudaFree(dev_dst));
  }

  // The CPU helper must agree with the same shift semantics.
  for (int value = 0; value <= 255; ++value) {
    const auto sample = static_cast<std::uint16_t>(value);
    CHECK(fmt::expand_8_to_depth(static_cast<std::uint8_t>(value),
                                 fmt::BitDepth::k10) ==
          static_cast<std::uint16_t>(
              static_cast<std::uint32_t>(sample) << 2U));
    CHECK(fmt::expand_8_to_depth(static_cast<std::uint8_t>(value),
                                 fmt::BitDepth::k12) ==
          static_cast<std::uint16_t>(
              static_cast<std::uint32_t>(sample) << 4U));
  }
}

void test_reduce_16_to_8() {
  constexpr int kWidth = 65;
  constexpr int kHeight = 37;
  constexpr int kSrcStride = kWidth + 2;
  constexpr int kDstStride = kWidth + 7;

  std::mt19937 rng(0xC0FFEEu);
  for (const int depth : {8, 10, 12}) {
    const int max_value = (1 << depth) - 1;
    std::uniform_int_distribution<int> dist(0, max_value);
    std::vector<std::uint16_t> src(static_cast<std::size_t>(kSrcStride) * kHeight);
    for (std::uint16_t& value : src) {
      value = static_cast<std::uint16_t>(dist(rng));
    }

    std::vector<std::uint8_t> expected(
        static_cast<std::size_t>(kDstStride) * kHeight, 0);
    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        expected[static_cast<std::size_t>(y) * kDstStride + x] =
            fmt::reduce_to_8(src[static_cast<std::size_t>(y) * kSrcStride + x],
                             static_cast<fmt::BitDepth>(depth));
      }
    }

    std::uint16_t* dev_src = nullptr;
    std::uint8_t* dev_dst = nullptr;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev_src),
                        src.size() * sizeof(std::uint16_t)));
    CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&dev_dst),
        static_cast<std::size_t>(kDstStride) * kHeight));
    CUDA_TRY(cudaMemcpy(dev_src, src.data(), src.size() * sizeof(std::uint16_t),
                        cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemset(dev_dst, 0,
                        static_cast<std::size_t>(kDstStride) * kHeight));
    CUDA_TRY(hb::reduce_16_to_8(dev_src, dev_dst, kWidth, kHeight, kSrcStride,
                                 kDstStride, depth));
    CUDA_TRY(cudaDeviceSynchronize());

    std::vector<std::uint8_t> actual(
        static_cast<std::size_t>(kDstStride) * kHeight, 0);
    CUDA_TRY(cudaMemcpy(actual.data(), dev_dst, actual.size(),
                        cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < expected.size(); ++i) {
      CHECK(actual[i] == expected[i]);
    }
    CUDA_TRY(cudaFree(dev_src));
    CUDA_TRY(cudaFree(dev_dst));
  }
}

void test_replicate_borders() {
  fmt::PlaneBuffer host;
  host.width = 16;
  host.height = 16;
  host.visible_width = 7;
  host.visible_height = 5;
  host.stride = 16;
  host.samples.assign(16U * 16U, 0);
  for (std::uint32_t y = 0; y < host.visible_height; ++y) {
    for (std::uint32_t x = 0; x < host.visible_width; ++x) {
      host.at(x, y) = static_cast<std::uint16_t>(x * 10U + y * 100U + 1U);
    }
  }

  fmt::PlaneBuffer expected = host;
  fmt::replicate_borders(expected);

  std::uint16_t* device = nullptr;
  const std::size_t bytes = host.samples.size() * sizeof(std::uint16_t);
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&device), bytes));
  CUDA_TRY(cudaMemcpy(device, host.samples.data(), bytes,
                      cudaMemcpyHostToDevice));
  CUDA_TRY(hb::replicate_borders_16(device, static_cast<int>(host.stride),
                                    static_cast<int>(host.width),
                                    static_cast<int>(host.height),
                                    static_cast<int>(host.visible_width),
                                    static_cast<int>(host.visible_height)));
  CUDA_TRY(cudaDeviceSynchronize());

  fmt::PlaneBuffer actual = host;
  CUDA_TRY(cudaMemcpy(actual.samples.data(), device, bytes,
                      cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < expected.samples.size(); ++i) {
    CHECK(actual.samples[i] == expected.samples[i]);
  }
  CUDA_TRY(cudaFree(device));

  // Visible == coded is a no-op.
  fmt::PlaneBuffer full = host;
  full.visible_width = full.width;
  full.visible_height = full.height;
  fmt::PlaneBuffer full_expected = full;
  fmt::replicate_borders(full_expected);
  CHECK(full_expected.samples == full.samples);
}

void test_geometry_agreement() {
  // The CUDA path is fed by the same geometry code; check the plane dimensions
  // the kernels will see for the new formats.
  for (const fmt::ChromaFormat chroma :
       {fmt::ChromaFormat::k420, fmt::ChromaFormat::k422,
        fmt::ChromaFormat::k444}) {
    const fmt::Format format{641, 361, chroma, fmt::BitDepth::k10};
    const auto geometry = fmt::make_frame_geometry(format);
    CHECK(geometry.has_value());
    if (!geometry) {
      continue;
    }
    CHECK(geometry->num_planes == 3);
    for (int plane = 0; plane < 3; ++plane) {
      const fmt::PlaneGeometry& pg = geometry->planes[plane];
      CHECK(pg.stride == pg.coded_width);
      CHECK(pg.sample_count ==
            static_cast<std::size_t>(pg.coded_width) * pg.coded_height);
    }
  }
}

}  // namespace

int main() {
  test_expand_8_to_16();
  test_reduce_16_to_8();
  test_replicate_borders();
  test_geometry_agreement();

  if (g_failures != 0) {
    std::printf("high_bitdepth_cuda_test: %d failure(s), %lld checks\n",
                g_failures, g_checks);
    return 1;
  }
  std::printf("high_bitdepth_cuda_test: PASS (%lld checks)\n", g_checks);
  return 0;
}
