// CPU-vs-CUDA bit-exactness and timing test for the AV2 §7.17 deblocking
// filter.
//
// The CUDA kernel and the CPU reference call the same host/device core
// (deblock_core.h), so this test is a regression check for host/device codegen,
// metadata indexing, thread mapping, and stride handling.
//
// Usage: ./deblock_test [repo_root]

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "cuda/codec/filters/deblock/deblock.h"
#include "codec/filters/deblock/deblock.h"
#include "codec/filters/deblock/deblock_test_common.h"

namespace {

using av2enc::deblock::DeblockOptions;
using av2enc::deblock::TxSize;
using namespace av2enc::deblock::test;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                     \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_failures;                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg));   \
    }                                                                        \
  } while (0)

#define CUDA_CHECK(expr, msg)                                                \
  do {                                                                       \
    const cudaError_t err__ = (expr);                                        \
    ++g_checks;                                                              \
    if (err__ != cudaSuccess) {                                              \
      ++g_failures;                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__,      \
                   (msg), cudaGetErrorString(err__));                        \
    }                                                                        \
  } while (0)

#define CUDA_TRY(expr, msg, okvar)                                           \
  do {                                                                       \
    const cudaError_t err__ = (expr);                                        \
    ++g_checks;                                                              \
    if (err__ != cudaSuccess) {                                              \
      ++g_failures;                                                          \
      okvar = false;                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__,      \
                   (msg), cudaGetErrorString(err__));                        \
    }                                                                        \
  } while (0)

struct DeviceFrame {
  av2enc::cuda::DeblockDeviceFrame view{};
  std::uint8_t* y = nullptr;
  std::uint8_t* u = nullptr;
  std::uint8_t* v = nullptr;
  av2enc::deblock::DeblockMi* mi[3] = {nullptr, nullptr, nullptr};
  std::uint16_t* tile_cols = nullptr;
  std::uint16_t* tile_rows = nullptr;

  ~DeviceFrame() { release(); }

  void release() {
    if (y) cudaFree(y);
    if (u) cudaFree(u);
    if (v) cudaFree(v);
    for (auto*& m : mi) {
      if (m) cudaFree(m);
      m = nullptr;
    }
    if (tile_cols) cudaFree(tile_cols);
    if (tile_rows) cudaFree(tile_rows);
    y = u = v = nullptr;
    tile_cols = tile_rows = nullptr;
    view = av2enc::cuda::DeblockDeviceFrame{};
  }

  bool upload(const TestFrame& f) {
    release();
    bool ok = true;
    const std::size_t y_bytes =
        static_cast<std::size_t>(f.y_stride) * f.y_height;
    const std::size_t u_bytes =
        static_cast<std::size_t>(f.u_stride) * f.u_height;
    const std::size_t v_bytes =
        static_cast<std::size_t>(f.v_stride) * f.v_height;
    const std::size_t mi_bytes =
        static_cast<std::size_t>(f.mi_cols) * f.mi_rows *
        sizeof(av2enc::deblock::DeblockMi);
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&y), y_bytes),
               "cudaMalloc y", ok);
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&u), u_bytes),
               "cudaMalloc u", ok);
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&v), v_bytes),
               "cudaMalloc v", ok);
    CUDA_TRY(cudaMemcpy(y, f.y.data(), y_bytes, cudaMemcpyHostToDevice),
               "upload y", ok);
    CUDA_TRY(cudaMemcpy(u, f.u.data(), u_bytes, cudaMemcpyHostToDevice),
               "upload u", ok);
    CUDA_TRY(cudaMemcpy(v, f.v.data(), v_bytes, cudaMemcpyHostToDevice),
               "upload v", ok);
    for (int p = 0; p < 3; ++p) {
      CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&mi[p]), mi_bytes),
                 "cudaMalloc mi", ok);
      CUDA_TRY(cudaMemcpy(mi[p], f.mi(p).data(), mi_bytes,
                            cudaMemcpyHostToDevice),
                 "upload mi", ok);
    }
    if (!f.tile_cols.empty()) {
      const std::size_t bytes =
          f.tile_cols.size() * sizeof(std::uint16_t);
      CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&tile_cols), bytes),
                 "cudaMalloc tile cols", ok);
      CUDA_TRY(cudaMemcpy(tile_cols, f.tile_cols.data(), bytes,
                            cudaMemcpyHostToDevice),
                 "upload tile cols", ok);
    }
    if (!f.tile_rows.empty()) {
      const std::size_t bytes =
          f.tile_rows.size() * sizeof(std::uint16_t);
      CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&tile_rows), bytes),
                 "cudaMalloc tile rows", ok);
      CUDA_TRY(cudaMemcpy(tile_rows, f.tile_rows.data(), bytes,
                            cudaMemcpyHostToDevice),
                 "upload tile rows", ok);
    }
    view.plane[0] = y;
    view.plane[1] = u;
    view.plane[2] = v;
    view.stride[0] = f.y_stride;
    view.stride[1] = f.u_stride;
    view.stride[2] = f.v_stride;
    view.mi[0] = mi[0];
    view.mi[1] = mi[1];
    view.mi[2] = mi[2];
    view.mi_cols = f.mi_cols;
    view.mi_rows = f.mi_rows;
    view.subsampling_x = 1;
    view.subsampling_y = 1;
    view.tile_col_starts = tile_cols;
    view.num_tile_cols = static_cast<int>(f.tile_cols.size());
    view.tile_row_starts = tile_rows;
    view.num_tile_rows = static_cast<int>(f.tile_rows.size());
    return ok;
  }

  bool download(TestFrame* f) const {
    bool ok = true;
    const std::size_t y_bytes =
        static_cast<std::size_t>(f->y_stride) * f->y_height;
    const std::size_t u_bytes =
        static_cast<std::size_t>(f->u_stride) * f->u_height;
    const std::size_t v_bytes =
        static_cast<std::size_t>(f->v_stride) * f->v_height;
    CUDA_TRY(cudaMemcpy(f->y.data(), y, y_bytes, cudaMemcpyDeviceToHost),
               "download y", ok);
    CUDA_TRY(cudaMemcpy(f->u.data(), u, u_bytes, cudaMemcpyDeviceToHost),
               "download u", ok);
    CUDA_TRY(cudaMemcpy(f->v.data(), v, v_bytes, cudaMemcpyDeviceToHost),
               "download v", ok);
    return ok;
  }
};

DeblockOptions base_options() {
  DeblockOptions o;
  o.apply_deblocking_filter[0] = true;
  o.apply_deblocking_filter[1] = true;
  o.apply_deblocking_filter[2] = true;
  o.apply_deblocking_filter[3] = true;
  return o;
}

bool run_cpu(TestFrame* f, const DeblockOptions& o) {
  av2enc::deblock::DeblockFrame view = f->view();
  return av2enc::deblock::deblock_frame(view, o) ==
         av2enc::deblock::DeblockStatus::kOk;
}

bool run_gpu(TestFrame* f, const DeblockOptions& o, DeviceFrame& d) {
  if (!d.upload(*f)) return false;
  CUDA_CHECK(av2enc::cuda::deblock_frame(d.view, o), "cuda deblock_frame");
  CUDA_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  return d.download(f);
}

bool frames_equal(const TestFrame& a, const TestFrame& b) {
  return a.y == b.y && a.u == b.u && a.v == b.v;
}

std::size_t compare_case(const char* name, TestFrame& host,
                         const DeblockOptions& o, DeviceFrame& d,
                         TestFrame* cpu_out = nullptr) {
  TestFrame cpu = host;
  TestFrame gpu = host;
  CHECK(run_cpu(&cpu, o), "CPU deblock failed");
  CHECK(run_gpu(&gpu, o, d), "GPU deblock failed");
  const std::size_t mismatch = count_mismatch(cpu, gpu);
  CHECK(mismatch == 0, name);
  if (mismatch != 0) {
    int p = 0, x = 0, y = 0, va = 0, vb = 0;
    if (first_mismatch(cpu, gpu, &p, &x, &y, &va, &vb)) {
      std::fprintf(stderr,
                   "    first mismatch plane=%d (%dx%d) at (%d,%d): cpu=%d "
                   "gpu=%d\n",
                   p, cpu.plane_width(p), cpu.plane_height(p), x, y, va, vb);
    }
  }
  CHECK(guard_intact(gpu), "GPU must not write outside the MI-aligned plane");
  if (cpu_out != nullptr) {
    *cpu_out = cpu;
  }
  const std::size_t changed = count_changed(host, gpu);
  std::cout << "  " << name << ": " << (mismatch ? "MISMATCH" : "exact")
            << " (changed=" << changed << ")\n";
  return changed;
}

void test_synthetic_patterns(DeviceFrame& d) {
  const Pattern patterns[] = {Pattern::kFlat,       Pattern::kVerticalStep,
                              Pattern::kHorizontalStep, Pattern::kRampX,
                              Pattern::kRampY,    Pattern::kChecker,
                              Pattern::kAlternating, Pattern::kNoise,
                              Pattern::kSoftEdge};
  const int sizes[][2] = {{16, 16}, {32, 24}, {80, 44}};
  for (const auto& sz : sizes) {
    for (int pi = 0; pi < 9; ++pi) {
      TestFrame f = make_frame(sz[0], sz[1]);
      set_uniform_blocks(f, 8, TxSize::k32x32, 140);
      fill_pattern(f, patterns[pi]);
      DeblockOptions o = base_options();
      char label[128];
      std::snprintf(label, sizeof(label), "pattern %d %dx%d", pi, sz[0], sz[1]);
      compare_case(label, f, o, d);
    }
  }
}

void test_random_metadata(DeviceFrame& d, std::mt19937& rng, bool quick) {
  const int sizes[][2] = {{20, 16}, {32, 24}, {48, 32}, {80, 44}, {160, 90}};
  std::size_t changed_total = 0;
  const int iterations = quick ? 10 : 120;
  for (int iter = 0; iter < iterations; ++iter) {
    const auto& sz = sizes[iter % 5];
    TestFrame f = make_frame(sz[0], sz[1]);
    const Pattern patterns[] = {Pattern::kFlat,        Pattern::kVerticalStep,
                                Pattern::kHorizontalStep, Pattern::kRampX,
                                Pattern::kChecker,     Pattern::kNoise,
                                Pattern::kSoftEdge};
    fill_pattern(f, patterns[iter % 7]);
    set_random_all_planes(f, rng);
    set_random_tiles(f, rng);

    DeblockOptions o = base_options();
    o.apply_deblocking_filter[0] = (rng() & 1) != 0;
    o.apply_deblocking_filter[1] = (rng() & 1) != 0;
    o.apply_deblocking_filter[2] = (rng() & 1) != 0;
    o.apply_deblocking_filter[3] = (rng() & 1) != 0;
    o.disable_loopfilters_across_tiles = (rng() & 1) != 0;
    o.allow_df_sub_pu = (rng() & 1) != 0;
    o.frame_is_intra = (rng() & 3) != 0;
    o.enable_sdp = (rng() & 1) != 0;
    o.df_delta_q[0] = static_cast<int>(rng() % 5) - 2;
    o.df_delta_q[1] = static_cast<int>(rng() % 5) - 2;
    o.df_delta_q[2] = static_cast<int>(rng() % 5) - 2;
    o.df_delta_q[3] = static_cast<int>(rng() % 5) - 2;
    o.delta_q_u_ac = static_cast<int>(rng() % 9) - 4;
    o.delta_q_v_ac = static_cast<int>(rng() % 9) - 4;
    o.base_uv_ac_delta_q = static_cast<int>(rng() % 5) - 2;

    char label[128];
    std::snprintf(label, sizeof(label), "random iter %d (%dx%d)", iter, sz[0],
                  sz[1]);
    changed_total += compare_case(label, f, o, d);
  }
  CHECK(changed_total > 0, "random corpus must exercise filtering");
  std::cout << "  random metadata: " << iterations << " frames, changed samples="
            << changed_total << "\n";
}

std::string repo_root_from_source() {
  std::string path = __FILE__;
  for (int i = 0; i < 4; ++i) {
    const std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) break;
    path.resize(slash);
  }
  return path;
}

void test_real_recon(const std::string& root, DeviceFrame& d) {
  const std::string candidates[] = {
      root + "/decoded.y4m",
      root + "/testdata/shine_256x256_1f.y4m",
      root + "/testdata/shine_1280x720_60f.y4m",
  };
  std::string path;
  for (const std::string& p : candidates) {
    std::ifstream in(p, std::ios::binary);
    if (in) {
      path = p;
      break;
    }
  }
  if (path.empty()) {
    std::cout << "  real recon: skipped (no Y4M found)\n";
    return;
  }
  TestFrame f;
  if (!load_y4m_first_frame(path, &f)) {
    std::cout << "  real recon: skipped (parse failed)\n";
    return;
  }
  set_uniform_blocks(f, 8, TxSize::k32x32, 120);
  DeblockOptions o = base_options();
  TestFrame before = f;
  TestFrame cpu_result;
  const std::size_t changed =
      compare_case("real recon", f, o, d, &cpu_result);

  // Determinism: a second GPU run must produce the same bytes.
  TestFrame again = before;
  CHECK(run_gpu(&again, o, d), "second GPU run failed");
  CHECK(again.y == cpu_result.y && again.u == cpu_result.u &&
            again.v == cpu_result.v,
        "GPU deblock must be deterministic");
  std::cout << "    " << path << " " << f.luma_width() << "x"
            << f.luma_height() << ", changed=" << changed
            << ", mean|d|=" << mean_abs_delta(before, cpu_result)
            << ", max|d|=" << max_abs_delta(before, cpu_result) << "\n";
}

void test_validation(DeviceFrame& d, const TestFrame& f) {
  (void)d.upload(f);
  DeblockOptions o = base_options();

  auto expect_bad = [&](av2enc::cuda::DeblockDeviceFrame v, const char* msg) {
    const cudaError_t err = av2enc::cuda::deblock_frame(v, o);
    CHECK(err == cudaErrorInvalidValue, msg);
  };

  av2enc::cuda::DeblockDeviceFrame v = d.view;
  v.plane[0] = nullptr;
  expect_bad(v, "null plane rejected");
  v = d.view;
  v.stride[1] = 1;
  expect_bad(v, "short stride rejected");
  v = d.view;
  v.mi_cols = 0;
  expect_bad(v, "zero geometry rejected");
  v = d.view;
  v.mi_cols += 1;  // odd MI grid
  expect_bad(v, "odd MI grid rejected");
  v = d.view;
  v.num_tile_cols = 1;
  v.tile_col_starts = nullptr;
  expect_bad(v, "missing tile starts rejected");
  v = d.view;
  v.subsampling_x = 0;
  v.subsampling_y = 0;
  expect_bad(v, "4:4:4 rejected");

  DeblockOptions bad = o;
  bad.bit_depth = 10;
  const cudaError_t err = av2enc::cuda::deblock_frame(d.view, bad);
  CHECK(err == cudaErrorInvalidValue, "10-bit rejected");
  std::cout << "  CUDA validation: OK\n";
}

void test_timing(DeviceFrame& d, bool quick) {
  // 1080p, 32x32 coding blocks / 32x32 transforms, natural-ish content.
  TestFrame f = make_frame(480, 270);
  set_uniform_blocks(f, 8, TxSize::k32x32, 120);
  fill_pattern(f, Pattern::kNoise);
  // Add broad steps so the filter actually does work.
  for (int y = 0; y < f.luma_height(); ++y) {
    for (int x = 0; x < f.luma_width(); ++x) {
      const int base = static_cast<int>(
          f.y[static_cast<std::size_t>(y) * f.y_stride + x]);
      const int step = ((x / 64 + y / 64) & 1) ? 24 : -24;
      f.y[static_cast<std::size_t>(y) * f.y_stride + x] =
          static_cast<std::uint8_t>(std::max(0, std::min(255, base + step)));
    }
  }
  for (int p = 1; p < 3; ++p) {
    const int w = f.plane_width(p);
    const int h = f.plane_height(p);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int base = static_cast<int>(
            f.plane(p)[static_cast<std::size_t>(y) * f.stride(p) + x]);
        const int step = ((x / 32 + y / 32) & 1) ? 20 : -20;
        f.plane(p)[static_cast<std::size_t>(y) * f.stride(p) + x] =
            static_cast<std::uint8_t>(std::max(0, std::min(255, base + step)));
      }
    }
  }

  DeblockOptions o = base_options();
  TestFrame cpu = f;
  Timer ct;
  CHECK(run_cpu(&cpu, o), "timing CPU run failed");
  const double cpu_ms = ct.elapsed_ms();

  CHECK(d.upload(f), "timing upload failed");
  // Warm up.
  for (int i = 0; i < 5; ++i) {
    CUDA_CHECK(av2enc::cuda::deblock_frame(d.view, o), "warmup launch");
  }
  CUDA_CHECK(cudaDeviceSynchronize(), "warmup sync");

  const int kIters = quick ? 2 : 200;
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start), "event create");
  CUDA_CHECK(cudaEventCreate(&stop), "event create");
  CUDA_CHECK(cudaEventRecord(start), "event record");
  for (int i = 0; i < kIters; ++i) {
    CUDA_CHECK(av2enc::cuda::deblock_frame(d.view, o), "timed launch");
  }
  CUDA_CHECK(cudaEventRecord(stop), "event record");
  CUDA_CHECK(cudaEventSynchronize(stop), "event sync");
  float gpu_ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop), "elapsed time");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  gpu_ms /= static_cast<float>(kIters);

  const std::size_t samples =
      static_cast<std::size_t>(f.luma_width()) * f.luma_height() +
      2 * (static_cast<std::size_t>(f.chroma_width()) * f.chroma_height());
  const double gsamples = static_cast<double>(samples) /
                          (static_cast<double>(gpu_ms) * 1.0e6);
  std::cout << "  timing 1920x1080: GPU " << gpu_ms << " ms/frame ("
            << gsamples << " Gsample/s, 2 launches), CPU reference "
            << cpu_ms << " ms/frame (not a perf target)\n";
}

}  // namespace

int main(int argc, char** argv) {
  const std::string root = (argc > 1) ? argv[1] : repo_root_from_source();
  std::mt19937 rng(0xDEB10C4u);
  const bool quick = std::getenv("DEBLOCK_TEST_QUICK") != nullptr;

  std::cout << "AV2 deblocking filter CPU/CUDA bit-exactness tests\n";
  std::cout << "repo root: " << root << "\n";

  DeviceFrame d;
  test_synthetic_patterns(d);
  test_random_metadata(d, rng, quick);

  TestFrame small = make_frame(16, 16);
  set_uniform_blocks(small, 4, TxSize::k16x16, 120);
  fill_pattern(small, Pattern::kSoftEdge);
  test_validation(d, small);

  test_real_recon(root, d);
  test_timing(d, quick);

  std::cout << "checks: " << g_checks << ", failures: " << g_failures << "\n";
  return g_failures == 0 ? 0 : 1;
}
