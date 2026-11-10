// AV2 §7.18 CDEF — CPU-vs-CUDA bit-exactness and timing test.
//
// The CUDA kernel and the CPU reference call the same host/device core
// (cdef_core.h), so this test is a regression check for host/device codegen,
// block indexing, thread mapping, tile regions and stride handling.
//
// Usage: ./cdef_test [repo_root]

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/cdef/cdef.h"
#include "encoder/filters/cdef/cdef.h"
#include "encoder/filters/cdef/cdef_test_common.h"

namespace {

using av2enc::cdef::CdefFrame;
using av2enc::cdef::CdefOptions;
using av2enc::cdef::CdefStatus;
using av2enc::cdef::CdefStrength;
using av2enc::cdef::test::DiffStats;
using av2enc::cdef::test::Pattern;
using av2enc::cdef::test::TestFrame;

int g_checks = 0;
int g_failures = 0;
bool g_quick = false;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
    }                                                                      \
  } while (0)

#define CUDA_TRY(expr, msg, okvar)                                         \
  do {                                                                     \
    const cudaError_t err__ = (expr);                                      \
    ++g_checks;                                                            \
    if (err__ != cudaSuccess) {                                            \
      ++g_failures;                                                        \
      okvar = false;                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__,    \
                   (msg), cudaGetErrorString(err__));                      \
    }                                                                      \
  } while (0)

struct DeviceFrame {
  av2enc::cuda::CdefDeviceFrame view{};
  std::uint8_t* src[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* dst[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* skip = nullptr;
  std::uint8_t* lossless = nullptr;
  std::uint8_t* lossless_chroma = nullptr;
  std::int8_t* idx = nullptr;
  std::uint16_t* tile_cols = nullptr;
  std::uint16_t* tile_rows = nullptr;

  ~DeviceFrame() { release(); }

  void release() {
    for (int plane = 0; plane < 3; ++plane) {
      if (src[plane] != nullptr) {
        cudaFree(src[plane]);
        src[plane] = nullptr;
      }
      if (dst[plane] != nullptr) {
        cudaFree(dst[plane]);
        dst[plane] = nullptr;
      }
    }
    if (skip != nullptr) {
      cudaFree(skip);
      skip = nullptr;
    }
    if (lossless != nullptr) {
      cudaFree(lossless);
      lossless = nullptr;
    }
    if (lossless_chroma != nullptr) {
      cudaFree(lossless_chroma);
      lossless_chroma = nullptr;
    }
    if (idx != nullptr) {
      cudaFree(idx);
      idx = nullptr;
    }
    if (tile_cols != nullptr) {
      cudaFree(tile_cols);
      tile_cols = nullptr;
    }
    if (tile_rows != nullptr) {
      cudaFree(tile_rows);
      tile_rows = nullptr;
    }
    view = av2enc::cuda::CdefDeviceFrame{};
  }

  bool upload(const TestFrame& frame) {
    bool ok = true;
    for (int plane = 0; plane < 3; ++plane) {
      const std::size_t bytes = frame.src[plane].size();
      CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&src[plane]), bytes),
               "cudaMalloc src", ok);
      CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dst[plane]), bytes),
               "cudaMalloc dst", ok);
      CUDA_TRY(cudaMemcpy(src[plane], frame.src[plane].data(), bytes,
                          cudaMemcpyHostToDevice),
               "upload src", ok);
      CUDA_TRY(cudaMemcpy(dst[plane], frame.dst[plane].data(), bytes,
                          cudaMemcpyHostToDevice),
               "upload dst guard", ok);
    }
    const std::size_t mi_bytes = frame.skip.size();
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&skip), mi_bytes),
             "cudaMalloc skip", ok);
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&lossless), mi_bytes),
             "cudaMalloc lossless", ok);
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&lossless_chroma), mi_bytes),
             "cudaMalloc lossless_chroma", ok);
    CUDA_TRY(cudaMemcpy(skip, frame.skip.data(), mi_bytes,
                        cudaMemcpyHostToDevice),
             "upload skip", ok);
    CUDA_TRY(cudaMemcpy(lossless, frame.lossless.data(), mi_bytes,
                        cudaMemcpyHostToDevice),
             "upload lossless", ok);
    CUDA_TRY(cudaMemcpy(lossless_chroma, frame.lossless_chroma.data(), mi_bytes,
                        cudaMemcpyHostToDevice),
             "upload lossless_chroma", ok);
    const std::size_t idx_bytes = frame.idx.size();
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&idx), idx_bytes),
             "cudaMalloc idx", ok);
    CUDA_TRY(cudaMemcpy(idx, frame.idx.data(), idx_bytes,
                        cudaMemcpyHostToDevice),
             "upload idx", ok);

    if (!frame.tile_cols.empty()) {
      const std::size_t bytes =
          frame.tile_cols.size() * sizeof(std::uint16_t);
      CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&tile_cols), bytes),
               "cudaMalloc tile cols", ok);
      CUDA_TRY(cudaMemcpy(tile_cols, frame.tile_cols.data(), bytes,
                          cudaMemcpyHostToDevice),
               "upload tile cols", ok);
    }
    if (!frame.tile_rows.empty()) {
      const std::size_t bytes =
          frame.tile_rows.size() * sizeof(std::uint16_t);
      CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&tile_rows), bytes),
               "cudaMalloc tile rows", ok);
      CUDA_TRY(cudaMemcpy(tile_rows, frame.tile_rows.data(), bytes,
                          cudaMemcpyHostToDevice),
               "upload tile rows", ok);
    }

    for (int plane = 0; plane < 3; ++plane) {
      view.src[plane] = src[plane];
      view.dst[plane] = dst[plane];
      view.stride[plane] = frame.stride[plane];
    }
    view.mi_cols = frame.mi_cols;
    view.mi_rows = frame.mi_rows;
    view.subsampling_x = 1;
    view.subsampling_y = 1;
    view.skip = skip;
    view.lossless = lossless;
    view.lossless_chroma = lossless_chroma;
    view.cdef_idx = idx;
    view.n64_cols = frame.n64_cols;
    view.n64_rows = frame.n64_rows;
    view.tile_col_starts = frame.tile_cols.empty() ? nullptr : tile_cols;
    view.num_tile_cols = static_cast<int>(frame.tile_cols.size());
    view.tile_row_starts = frame.tile_rows.empty() ? nullptr : tile_rows;
    view.num_tile_rows = static_cast<int>(frame.tile_rows.size());
    return ok;
  }

  bool download(TestFrame* frame) {
    bool ok = true;
    for (int plane = 0; plane < 3; ++plane) {
      const std::size_t bytes = frame->dst[plane].size();
      CUDA_TRY(cudaMemcpy(frame->dst[plane].data(), dst[plane], bytes,
                          cudaMemcpyDeviceToHost),
               "download dst", ok);
    }
    return ok;
  }
};

CdefOptions options_with(int damp, int num, CdefStrength s0,
                         CdefStrength s1 = CdefStrength{}) {
  CdefOptions o;
  o.damping = damp;
  o.num_strengths = num;
  o.strengths[0] = s0;
  o.strengths[1] = s1;
  return o;
}

int run_cpu(TestFrame& frame, const CdefOptions& options) {
  av2enc::cdef::test::reset_dst(frame);
  CdefFrame view = frame.view();
  const CdefStatus status = av2enc::cdef::cdef_frame(view, options);
  return status == CdefStatus::kOk ? 0 : 1;
}

// Runs the GPU kernel and writes the result into `out`.  Returns false on a
// CUDA error.
bool run_gpu(const TestFrame& frame, const CdefOptions& options,
             TestFrame* out) {
  DeviceFrame dev;
  if (!dev.upload(frame)) {
    return false;
  }
  const cudaError_t err = av2enc::cuda::cdef_frame(dev.view, options);
  ++g_checks;
  if (err != cudaSuccess) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: cdef_frame launch (%s)\n",
                 cudaGetErrorString(err));
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    ++g_checks;
    ++g_failures;
    std::fprintf(stderr, "FAIL: kernel execution\n");
    return false;
  }
  *out = frame;
  if (!dev.download(out)) {
    return false;
  }
  return true;
}

void compare(const TestFrame& cpu, const TestFrame& gpu, const char* what) {
  const std::size_t mismatches = av2enc::cdef::test::count_mismatch(cpu, gpu);
  ++g_checks;
  if (mismatches != 0) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s: %zu mismatched samples (max |delta| %d)\n",
                 what, mismatches, av2enc::cdef::test::max_abs_diff(cpu, gpu));
    for (int plane = 0; plane < 3; ++plane) {
      const int w = cpu.plane_width(plane);
      const int h = cpu.plane_height(plane);
      const int stride = cpu.stride[plane];
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const std::size_t o =
              av2enc::cdef::test::pix_index(y, stride, x);
          if (cpu.dst[plane][o] != gpu.dst[plane][o]) {
            std::fprintf(stderr, "  first mismatch plane %d (%d,%d): %d vs %d\n",
                         plane, x, y, cpu.dst[plane][o], gpu.dst[plane][o]);
            plane = 3;
            y = h;
            break;
          }
        }
      }
    }
  }
}

void test_synthetic() {
  struct Size {
    int mi_cols;
    int mi_rows;
  };
  const Size sizes[3] = {{8, 6}, {16, 10}, {32, 18}};
  CdefOptions options = options_with(4, 2, CdefStrength{10, 2, 8, 1},
                                     CdefStrength{14, 4, 11, 2});
  bool all_ok = true;
  for (int p = 0; p < 11; ++p) {
    for (const Size& size : sizes) {
      TestFrame frame =
          av2enc::cdef::test::make_frame(size.mi_cols, size.mi_rows);
      av2enc::cdef::test::fill_pattern(frame, static_cast<Pattern>(p));
      av2enc::cdef::test::set_uniform_metadata(frame, 0, false, false, false);
      TestFrame cpu = frame;
      TestFrame gpu = frame;
      all_ok = all_ok && run_cpu(cpu, options) == 0;
      all_ok = all_ok && run_gpu(frame, options, &gpu);
      compare(cpu, gpu, "synthetic pattern");
    }
  }
  CHECK(all_ok, "synthetic frames CPU/GPU exact");
}

void test_random() {
  std::mt19937 rng(0xC0FFEECDu);
  CdefStrength sets[3] = {{5, 1, 4, 1}, {11, 4, 9, 2}, {15, 4, 13, 4}};
  bool all_ok = true;
  std::size_t total_changed = 0;
  const int iterations = g_quick ? 8 : 40;
  for (int iter = 0; iter < iterations; ++iter) {
    const int mi_cols = 4 + 2 * static_cast<int>(rng() % 40);
    const int mi_rows = 4 + 2 * static_cast<int>(rng() % 24);
    TestFrame frame = av2enc::cdef::test::make_frame(mi_cols, mi_rows);
    av2enc::cdef::test::fill_pattern(frame,
                                     static_cast<Pattern>(rng() % 11));
    CdefOptions options;
    options.damping = 3 + static_cast<int>(rng() % 4);
    options.num_strengths = 1 + static_cast<int>(rng() % 3);
    for (int i = 0; i < options.num_strengths; ++i) {
      options.strengths[i] = sets[rng() % 3];
    }
    options.on_skip_txfm = (rng() & 1) != 0;
    options.disable_loopfilters_across_tiles = (rng() & 1) != 0;
    av2enc::cdef::test::set_random_metadata(frame, rng, options.num_strengths,
                                            true, true, true);
    TestFrame cpu = frame;
    TestFrame gpu = frame;
    all_ok = all_ok && run_cpu(cpu, options) == 0;
    all_ok = all_ok && run_gpu(frame, options, &gpu);
    compare(cpu, gpu, "random frame");
    all_ok = all_ok && av2enc::cdef::test::guard_intact(gpu);
    total_changed += av2enc::cdef::test::diff_stats(gpu).changed;
  }
  CHECK(all_ok, "random frames CPU/GPU exact");
  std::printf("  random frames: %zu changed samples\n", total_changed);
}

void test_real_frame(const std::string& root) {
  TestFrame frame;
  if (!av2enc::cdef::test::load_y4m_first_frame(root + "/decoded.y4m",
                                                &frame)) {
    std::printf("  real frame: %s not found, skipped\n",
                (root + "/decoded.y4m").c_str());
    return;
  }
  av2enc::cdef::test::set_uniform_metadata(frame, 0, false, false, false);
  CdefOptions options = options_with(3, 1, CdefStrength{9, 2, 7, 1});
  TestFrame cpu = frame;
  TestFrame gpu = frame;
  const bool ok = run_cpu(cpu, options) == 0 &&
                  run_gpu(frame, options, &gpu);
  CHECK(ok, "real frame runs");
  if (ok) {
    compare(cpu, gpu, "real reconstructed frame");
    const DiffStats stats = av2enc::cdef::test::diff_stats(gpu);
    std::printf(
        "  real frame %dx%d: %zu changed, mean |delta| %.4f, max %d\n",
        gpu.plane_width(0), gpu.plane_height(0), stats.changed, stats.mean_abs,
        stats.max_abs);
    CHECK(av2enc::cdef::test::guard_intact(gpu), "real frame guards intact");
  }
}

void test_validation() {
  TestFrame frame = av2enc::cdef::test::make_frame(8, 6);
  av2enc::cdef::test::fill_pattern(frame, Pattern::kNoise);
  av2enc::cdef::test::set_uniform_metadata(frame, 0, false, false, false);
  CdefOptions options = options_with(3, 1, CdefStrength{8, 2, 6, 1});

  DeviceFrame dev;
  CHECK(dev.upload(frame), "validation frame uploads");
  CHECK(av2enc::cuda::cdef_frame(dev.view, options) == cudaSuccess,
        "valid device frame accepted");

  {
    av2enc::cuda::CdefDeviceFrame bad = dev.view;
    bad.src[0] = nullptr;
    CHECK(av2enc::cuda::cdef_frame(bad, options) == cudaErrorInvalidValue,
          "null src rejected");
  }
  {
    av2enc::cuda::CdefDeviceFrame bad = dev.view;
    bad.dst[1] = nullptr;
    CHECK(av2enc::cuda::cdef_frame(bad, options) == cudaErrorInvalidValue,
          "null dst rejected");
  }
  {
    av2enc::cuda::CdefDeviceFrame bad = dev.view;
    bad.dst[0] = const_cast<std::uint8_t*>(bad.src[0]);
    CHECK(av2enc::cuda::cdef_frame(bad, options) == cudaErrorInvalidValue,
          "in-place rejected");
  }
  {
    av2enc::cuda::CdefDeviceFrame bad = dev.view;
    bad.stride[2] = 1;
    CHECK(av2enc::cuda::cdef_frame(bad, options) == cudaErrorInvalidValue,
          "short stride rejected");
  }
  {
    av2enc::cuda::CdefDeviceFrame bad = dev.view;
    bad.mi_cols = 7;
    CHECK(av2enc::cuda::cdef_frame(bad, options) == cudaErrorInvalidValue,
          "odd MI grid rejected");
  }
  {
    av2enc::cuda::CdefDeviceFrame bad = dev.view;
    bad.mi_rows = 0;
    CHECK(av2enc::cuda::cdef_frame(bad, options) == cudaErrorInvalidValue,
          "empty frame rejected");
  }
  {
    CdefOptions bad = options;
    bad.bit_depth = 10;
    CHECK(av2enc::cuda::cdef_frame(dev.view, bad) == cudaErrorInvalidValue,
          "10-bit rejected");
  }
  {
    av2enc::cuda::CdefDeviceFrame bad = dev.view;
    bad.subsampling_y = 0;
    CHECK(av2enc::cuda::cdef_frame(bad, options) == cudaErrorInvalidValue,
          "4:4:4 rejected");
  }
  {
    CdefOptions bad = options;
    bad.damping = 7;
    CHECK(av2enc::cuda::cdef_frame(dev.view, bad) == cudaErrorInvalidValue,
          "damping rejected");
  }
  {
    CdefOptions bad = options;
    bad.num_strengths = 0;
    CHECK(av2enc::cuda::cdef_frame(dev.view, bad) == cudaErrorInvalidValue,
          "zero strengths rejected");
  }
  {
    CdefOptions bad = options;
    bad.strengths[0].y_sec = 3;
    CHECK(av2enc::cuda::cdef_frame(dev.view, bad) == cudaErrorInvalidValue,
          "secondary strength 3 rejected");
  }
  {
    av2enc::cuda::CdefDeviceFrame bad = dev.view;
    bad.skip = nullptr;
    CHECK(av2enc::cuda::cdef_frame(bad, options) == cudaErrorInvalidValue,
          "null skip rejected");
  }
  {
    av2enc::cuda::CdefDeviceFrame bad = dev.view;
    bad.num_tile_cols = 2;
    CHECK(av2enc::cuda::cdef_frame(bad, options) == cudaErrorInvalidValue,
          "missing tile array rejected");
  }
}

void test_timing(const std::string& root) {
  auto time_gpu = [](const TestFrame& frame, const CdefOptions& options,
                     int iters) -> double {
    DeviceFrame dev;
    if (!dev.upload(frame)) {
      return -1.0;
    }
    // Warm-up.
    (void)av2enc::cuda::cdef_frame(dev.view, options);
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
      (void)av2enc::cuda::cdef_frame(dev.view, options);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(ms) / static_cast<double>(iters);
  };

  CdefOptions options = options_with(3, 1, CdefStrength{9, 2, 7, 1});

  TestFrame real;
  if (av2enc::cdef::test::load_y4m_first_frame(root + "/decoded.y4m", &real)) {
    av2enc::cdef::test::set_uniform_metadata(real, 0, false, false, false);
    const double ms = time_gpu(real, options, g_quick ? 5 : 100);
    if (ms >= 0.0) {
      std::printf("  GPU 640x360 : %.4f ms/frame\n", ms);
    }
  }

  TestFrame hd = av2enc::cdef::test::make_frame(1920 / 4, 1080 / 4);
  av2enc::cdef::test::fill_pattern(hd, Pattern::kNoise);
  av2enc::cdef::test::set_uniform_metadata(hd, 0, false, false, false);
  const double ms = time_gpu(hd, options, g_quick ? 5 : 100);
  if (ms >= 0.0) {
    std::printf("  GPU 1920x1080: %.4f ms/frame\n", ms);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string root = ".";
  if (argc > 1) {
    root = argv[1];
  }

  g_quick = std::getenv("CDEF_TEST_QUICK") != nullptr;
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "cdef_test: no CUDA device (%s)\n",
                 cudaGetErrorString(err));
    return 77;
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  std::printf("cdef CUDA test on %s (sm_%d%d, %d SMs)\n", prop.name,
              prop.major, prop.minor, prop.multiProcessorCount);

  test_synthetic();
  test_random();
  test_real_frame(root);
  test_validation();
  test_timing(root);

  std::printf("cdef_test (CUDA): %s (%d checks, %d failures)\n",
              g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
