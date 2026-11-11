// CPU-vs-GPU bit-exactness for the explicit CfL alpha path.
//
// This file is host-only CUDA API code (no kernels), so the same translation
// unit builds with nvcc against the real src/cuda/cfl/cfl.cu and with a plain
// host compiler against cuda_emulation.cpp for local test-logic validation.
//
// The GPU path is the shared kernel from src/cuda/cfl, called with
// alpha_q8 = CflAlpha * 32. The CPU path is the shared core in ../cfl.h, reached
// through this component's search/selection. Both use the same integer helpers
// (cfl_common.h), so this checks the launch/thread mapping and the alpha
// plumbing, not a second spec transcription.
//
// Build/run on the box: ./build_cuda_test.sh
// Local emulation:       ./build_emulated_gpu_test.sh
//
// Memory contract: the per-case descriptors are passed by value and contain
// device pointers. The batch entry points additionally take a *device* array
// of descriptors, so run_batch() uploads its host vector with cudaMemcpy.
// Every case is synchronized and named, so an asynchronous fault reports the
// case instead of only a source line.

#include "../../../cuda/cfl/cfl.h"

#include "../cfl.h"
#include "alpha.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

namespace cuda = av2enc::cuda;
namespace cfl = av2enc::cfl;
namespace alpha = av2enc::cfl::alpha;

namespace {

constexpr int kLumaStride = 192;
constexpr int kLumaSize = 192;
constexpr int kChromaStride = 128;
constexpr int kChromaSize = 128;
constexpr std::size_t kLumaPlane =
    static_cast<std::size_t>(kLumaStride) * kLumaSize;
constexpr std::size_t kChromaPlane =
    static_cast<std::size_t>(kChromaStride) * kChromaSize;

long long g_samples = 0;
long long g_mismatches = 0;
long long g_alpha_checks = 0;
long long g_alpha_mismatches = 0;
int g_cases = 0;
const char* g_case_name = "<none>";
int g_case_number = 0;

// Every case is named and synchronized, so an asynchronous device fault is
// reported with the case that caused it instead of only a source line.
void begin_case(const char* name) {
  ++g_case_number;
  g_case_name = name;
  if (std::getenv("ALPHA_CUDA_VERBOSE") != nullptr) {
    std::printf("case %d [%s]\n", g_case_number, name);
    std::fflush(stdout);
  }
}

void sync_case(const char* stage) {
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                 "CUDA failure at case %d [%s] stage [%s]: %s (%d)\n",
                 g_case_number, g_case_name, stage, cudaGetErrorString(err),
                 static_cast<int>(err));
    std::exit(1);
  }
}

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    const cudaError_t err_ = (expr);                                         \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr,                                                   \
                   "CUDA error at case %d [%s]: %s (%d) from %s at %s:%d\n", \
                   g_case_number, g_case_name, cudaGetErrorString(err_),      \
                   static_cast<int>(err_), #expr, __FILE__, __LINE__);       \
      std::exit(1);                                                          \
    }                                                                        \
  } while (false)

std::size_t ix(int row, int col, int stride) {
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(stride) +
         static_cast<std::size_t>(col);
}

struct HostCase {
  std::vector<std::uint8_t> luma;
  std::vector<std::uint8_t> chroma_u;
  std::vector<std::uint8_t> chroma_v;
  std::vector<std::uint8_t> source_u;
  std::vector<std::uint8_t> source_v;
  cfl::Block u{};
  cfl::Block v{};

  HostCase(int coded_w, int coded_h, int w, int h, int x, int y)
      : luma(kLumaPlane, 128),
        chroma_u(kChromaPlane, 128),
        chroma_v(kChromaPlane, 128),
        source_u(kChromaPlane, 128),
        source_v(kChromaPlane, 128) {
    u.luma = luma.data();
    u.luma_stride = kLumaStride;
    u.luma_max_x = kLumaSize - 1;
    u.luma_max_y = kLumaSize - 1;
    u.sb_top_y = ((y * 2) / 64) * 64;
    u.chroma = chroma_u.data();
    u.chroma_stride = kChromaStride;
    u.x = x;
    u.y = y;
    u.w = w;
    u.h = h;
    u.coded_width = coded_w;
    u.coded_height = coded_h;
    u.have_above = y >= 4 ? 1 : 0;
    u.have_left = x >= 4 ? 1 : 0;
    u.bit_depth = 8;
    u.ds_filter = 0;
    u.sub_x = 1;
    u.sub_y = 1;
    v = u;
    v.chroma = chroma_v.data();
  }

  std::uint8_t& luma_at(int row, int col) {
    return luma[ix(row, col, kLumaStride)];
  }
  std::uint8_t& u_at(int row, int col) {
    return chroma_u[ix(row, col, kChromaStride)];
  }
  std::uint8_t& v_at(int row, int col) {
    return chroma_v[ix(row, col, kChromaStride)];
  }
  std::uint8_t& src_u_at(int row, int col) {
    return source_u[ix(row, col, kChromaStride)];
  }
  std::uint8_t& src_v_at(int row, int col) {
    return source_v[ix(row, col, kChromaStride)];
  }
};

void fill_ramp(HostCase& c) {
  for (int row = 0; row < kLumaSize; ++row) {
    for (int col = 0; col < kLumaSize; ++col) {
      c.luma_at(row, col) =
          static_cast<std::uint8_t>(40 + ((row * 7 + col * 11) % 161));
    }
  }
}

int ref_round2_signed(int value) {
  constexpr int kShift = 11;
  if (value >= 0) {
    return (value + (1 << (kShift - 1))) >> kShift;
  }
  return -(((-value) + (1 << (kShift - 1))) >> kShift);
}

void fill_source(HostCase& c, const cfl::Scratch& scratch,
                 cfl::AlphaPair pair) {
  for (int i = 0; i < c.u.h; ++i) {
    for (int j = 0; j < c.u.w; ++j) {
      const int luma_value = scratch.luma[i * c.u.w + j];
      const int du = ref_round2_signed(
          cfl::alpha_q8(pair.u) * (luma_value - scratch.luma_avg));
      const int dv = ref_round2_signed(
          cfl::alpha_q8(pair.v) * (luma_value - scratch.luma_avg));
      c.src_u_at(c.u.y + i, c.u.x + j) =
          static_cast<std::uint8_t>(std::clamp(128 + du, 4, 251));
      c.src_v_at(c.u.y + i, c.u.x + j) =
          static_cast<std::uint8_t>(std::clamp(128 + dv, 4, 251));
    }
  }
}

struct DevicePlanes {
  std::uint8_t* luma = nullptr;
  std::uint8_t* chroma_u = nullptr;
  std::uint8_t* chroma_v = nullptr;

  DevicePlanes() = default;
  DevicePlanes(const DevicePlanes&) = delete;
  DevicePlanes& operator=(const DevicePlanes&) = delete;
  ~DevicePlanes() {
    if (luma != nullptr) {
      cudaFree(luma);
    }
    if (chroma_u != nullptr) {
      cudaFree(chroma_u);
    }
    if (chroma_v != nullptr) {
      cudaFree(chroma_v);
    }
  }

  bool allocate() {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, kLumaPlane));
    luma = static_cast<std::uint8_t*>(p);
    CUDA_CHECK(cudaMalloc(&p, kChromaPlane));
    chroma_u = static_cast<std::uint8_t*>(p);
    CUDA_CHECK(cudaMalloc(&p, kChromaPlane));
    chroma_v = static_cast<std::uint8_t*>(p);
    return true;
  }
};

void upload(const HostCase& c, DevicePlanes& d) {
  CUDA_CHECK(cudaMemcpy(d.luma, c.luma.data(), kLumaPlane,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.chroma_u, c.chroma_u.data(), kChromaPlane,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.chroma_v, c.chroma_v.data(), kChromaPlane,
                        cudaMemcpyHostToDevice));
}

void compare_planes(const char* what, const std::vector<std::uint8_t>& expected,
                    const std::vector<std::uint8_t>& actual) {
  for (std::size_t i = 0; i < expected.size(); ++i) {
    ++g_samples;
    if (expected[i] != actual[i]) {
      if (g_mismatches < 12) {
        std::fprintf(stderr, "MISMATCH [%s] byte %zu expected %u got %u\n",
                     what, i, static_cast<unsigned>(expected[i]),
                     static_cast<unsigned>(actual[i]));
      }
      ++g_mismatches;
    }
  }
}

// One explicit-alpha case: CPU apply_prediction_pair vs GPU cfl_predict.
void run_explicit(const char* what, const HostCase& host_case, int alpha_u,
                  int alpha_v) {
  begin_case(what);
  std::vector<std::uint8_t> cpu_u = host_case.chroma_u;
  std::vector<std::uint8_t> cpu_v = host_case.chroma_v;
  cfl::Block cu = host_case.u;
  cfl::Block cv = host_case.v;
  cu.chroma = cpu_u.data();
  cv.chroma = cpu_v.data();
  cfl::Scratch scratch;
  cfl::compute_luma(cu, scratch);
  cfl::apply_prediction(cu, alpha_u, scratch);
  cfl::apply_prediction(cv, alpha_v, scratch);

  DevicePlanes device;
  device.allocate();
  upload(host_case, device);

  cfl::Block du = host_case.u;
  cfl::Block dv = host_case.v;
  du.luma = device.luma;
  du.chroma = device.chroma_u;
  dv.luma = device.luma;
  dv.chroma = device.chroma_v;
  CUDA_CHECK(cuda::cfl_predict(du, alpha_u));
  CUDA_CHECK(cuda::cfl_predict(dv, alpha_v));
  sync_case("explicit launch");

  std::vector<std::uint8_t> gpu_u(kChromaPlane);
  std::vector<std::uint8_t> gpu_v(kChromaPlane);
  CUDA_CHECK(cudaMemcpy(gpu_u.data(), device.chroma_u, kChromaPlane,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gpu_v.data(), device.chroma_v, kChromaPlane,
                        cudaMemcpyDeviceToHost));
  compare_planes(what, cpu_u, gpu_u);
  compare_planes(what, cpu_v, gpu_v);
  ++g_cases;
}

// Derived-alpha case: CPU derive_implicit_alpha + apply vs the GPU entry point,
// including the alpha value the kernel reports.
void run_derived(const char* what, const HostCase& host_case) {
  begin_case(what);
  std::vector<std::uint8_t> cpu_u = host_case.chroma_u;
  std::vector<std::uint8_t> cpu_v = host_case.chroma_v;
  cfl::Block cu = host_case.u;
  cfl::Block cv = host_case.v;
  cu.chroma = cpu_u.data();
  cv.chroma = cpu_v.data();
  cfl::Scratch scratch;
  cfl::compute_luma(cu, scratch);
  const int alpha_u = cfl::derive_implicit_alpha(cu, scratch);
  cfl::apply_prediction(cu, alpha_u, scratch);
  const int alpha_v = cfl::derive_implicit_alpha(cv, scratch);
  cfl::apply_prediction(cv, alpha_v, scratch);

  DevicePlanes device;
  device.allocate();
  upload(host_case, device);
  cfl::Block du = host_case.u;
  cfl::Block dv = host_case.v;
  du.luma = device.luma;
  du.chroma = device.chroma_u;
  dv.luma = device.luma;
  dv.chroma = device.chroma_v;

  int* d_alpha = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_alpha), sizeof(int)));
  CUDA_CHECK(cuda::cfl_predict_derived(du, d_alpha));
  int gpu_alpha_u = 0;
  CUDA_CHECK(cudaMemcpy(&gpu_alpha_u, d_alpha, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cuda::cfl_predict_derived(dv, d_alpha));
  int gpu_alpha_v = 0;
  CUDA_CHECK(cudaMemcpy(&gpu_alpha_v, d_alpha, sizeof(int),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_alpha);
  sync_case("derived launch");

  ++g_alpha_checks;
  if (gpu_alpha_u != alpha_u) {
    ++g_alpha_mismatches;
    std::fprintf(stderr, "MISMATCH [%s] alpha_u expected %d got %d\n", what,
                 alpha_u, gpu_alpha_u);
  }
  ++g_alpha_checks;
  if (gpu_alpha_v != alpha_v) {
    ++g_alpha_mismatches;
    std::fprintf(stderr, "MISMATCH [%s] alpha_v expected %d got %d\n", what,
                 alpha_v, gpu_alpha_v);
  }

  std::vector<std::uint8_t> gpu_u(kChromaPlane);
  std::vector<std::uint8_t> gpu_v(kChromaPlane);
  CUDA_CHECK(cudaMemcpy(gpu_u.data(), device.chroma_u, kChromaPlane,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gpu_v.data(), device.chroma_v, kChromaPlane,
                        cudaMemcpyDeviceToHost));
  compare_planes(what, cpu_u, gpu_u);
  compare_planes(what, cpu_v, gpu_v);
  ++g_cases;
}

// Batch entry point with independent U and V descriptors (2 * kJobs jobs).
void run_batch(const char* what, HostCase& host_case) {
  begin_case(what);
  constexpr int kJobs = 4;
  const int alphas[kJobs] = {32, 64, -96, -256};
  const std::size_t batch_plane =
      static_cast<std::size_t>(kChromaStride) * kChromaSize;

  std::vector<std::vector<std::uint8_t>> cpu_u(kJobs);
  std::vector<std::vector<std::uint8_t>> cpu_v(kJobs);
  std::vector<std::vector<std::uint8_t>> gpu_u(kJobs);
  std::vector<std::vector<std::uint8_t>> gpu_v(kJobs);
  cfl::Scratch scratch;
  cfl::compute_luma(host_case.u, scratch);

  std::uint8_t* d_luma = nullptr;
  std::uint8_t* d_chroma = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_luma), kLumaPlane));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_chroma),
                        batch_plane * kJobs * 2));
  CUDA_CHECK(cudaMemcpy(d_luma, host_case.luma.data(), kLumaPlane,
                        cudaMemcpyHostToDevice));

  std::vector<cuda::CflJob> jobs(static_cast<std::size_t>(2 * kJobs));
  for (int i = 0; i < kJobs; ++i) {
    const std::size_t job = static_cast<std::size_t>(i);
    cpu_u[job] = host_case.chroma_u;
    cpu_v[job] = host_case.chroma_v;
    cfl::Block cu = host_case.u;
    cfl::Block cv = host_case.v;
    cu.chroma = cpu_u[job].data();
    cv.chroma = cpu_v[job].data();
    cfl::apply_prediction(cu, alphas[i], scratch);
    cfl::apply_prediction(cv, alphas[i], scratch);

    std::uint8_t* u_dst = d_chroma + batch_plane * static_cast<std::size_t>(2 * i);
    std::uint8_t* v_dst =
        d_chroma + batch_plane * static_cast<std::size_t>(2 * i + 1);
    CUDA_CHECK(cudaMemcpy(u_dst, host_case.chroma_u.data(), batch_plane,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v_dst, host_case.chroma_v.data(), batch_plane,
                          cudaMemcpyHostToDevice));
    jobs[static_cast<std::size_t>(2 * i)].block = host_case.u;
    jobs[static_cast<std::size_t>(2 * i)].block.luma = d_luma;
    jobs[static_cast<std::size_t>(2 * i)].block.chroma = u_dst;
    jobs[static_cast<std::size_t>(2 * i)].alpha_q8 = alphas[i];
    jobs[static_cast<std::size_t>(2 * i + 1)].block = host_case.v;
    jobs[static_cast<std::size_t>(2 * i + 1)].block.luma = d_luma;
    jobs[static_cast<std::size_t>(2 * i + 1)].block.chroma = v_dst;
    jobs[static_cast<std::size_t>(2 * i + 1)].alpha_q8 = alphas[i];
  }
  // cfl_predict_batch() takes a DEVICE CflJob array. The descriptors carry
  // device luma/chroma pointers, so the host vector must be uploaded; passing
  // jobs.data() directly makes predict_batch_kernel dereference host memory on
  // the GPU (illegal access).
  static_assert(std::is_trivially_copyable_v<cuda::CflJob>,
                "CflJob crosses the host/device boundary by value");
  const int job_count = 2 * kJobs;
  const std::size_t jobs_bytes =
      sizeof(cuda::CflJob) * static_cast<std::size_t>(job_count);
  cuda::CflJob* d_jobs = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_jobs), jobs_bytes));
  CUDA_CHECK(cudaMemcpy(d_jobs, jobs.data(), jobs_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cuda::cfl_predict_batch(d_jobs, job_count));
  sync_case("batch launch");
  cudaFree(d_jobs);

  for (int i = 0; i < kJobs; ++i) {
    const std::size_t job = static_cast<std::size_t>(i);
    gpu_u[job].resize(kChromaPlane);
    gpu_v[job].resize(kChromaPlane);
    CUDA_CHECK(cudaMemcpy(
        gpu_u[job].data(),
        d_chroma + batch_plane * static_cast<std::size_t>(2 * i), kChromaPlane,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_v[job].data(),
        d_chroma + batch_plane * static_cast<std::size_t>(2 * i + 1),
        kChromaPlane, cudaMemcpyDeviceToHost));
    compare_planes(what, cpu_u[job], gpu_u[job]);
    compare_planes(what, cpu_v[job], gpu_v[job]);
  }
  cudaFree(d_luma);
  cudaFree(d_chroma);
  ++g_cases;
}

}  // namespace

int main() {
  // Every per-plane alpha (17) on an interior block, applied to U and V.
  for (int index = 0; index < alpha::kPerPlaneAlphas; ++index) {
    const cfl::Alpha a = alpha::per_plane_alpha(index);
    HostCase c(64, 64, 8, 8, 16, 16);
    fill_ramp(c);
    run_explicit("all_alphas", c, cfl::alpha_q8(a), -cfl::alpha_q8(a));
  }

  // U/V pairs across all joint signs, sampled magnitudes.
  for (int signs = 0; signs < cfl::kJointSigns; ++signs) {
    for (int mag_u = 1; mag_u <= 8; mag_u += 3) {
      for (int mag_v = 1; mag_v <= 8; mag_v += 3) {
        const cfl::AlphaPair pair =
            alpha::pair_from_joint(signs, mag_u, mag_v);
        HostCase c(64, 64, 8, 8, 24, 24);
        fill_ramp(c);
        run_explicit("joint_signs", c, cfl::alpha_q8(pair.u),
                     cfl::alpha_q8(pair.v));
      }
    }
  }

  // Frame edges / superblock top / partial coded grid.
  const int positions[4] = {0, 4, 28, 56};
  for (int py = 0; py < 4; ++py) {
    for (int px = 0; px < 4; ++px) {
      HostCase c(64, 64, 8, 8, positions[px], positions[py]);
      fill_ramp(c);
      run_explicit("edge", c, 32 * 3, -32 * 5);
    }
  }
  HostCase partial(6, 6, 8, 8, 2, 2);
  fill_ramp(partial);
  run_explicit("partial", partial, 32 * 8, -32 * 8);

  // Clipping at both ends.
  HostCase clip(64, 64, 4, 4, 0, 0);
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      clip.luma_at(row, col) = 255;
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      clip.u_at(i, j) = 250;
      clip.v_at(i, j) = 5;
    }
  }
  run_explicit("clip", clip, 32 * 8, -32 * 8);

  // Derived alpha, including the reported implicitAlpha.
  HostCase derived(64, 64, 8, 8, 8, 8);
  fill_ramp(derived);
  for (int i = 0; i < 64; ++i) {
    derived.u_at(7, i) = static_cast<std::uint8_t>(100 + i % 17);
    derived.u_at(i, 7) = static_cast<std::uint8_t>(100 + i % 13);
    derived.v_at(7, i) = static_cast<std::uint8_t>(150 - i % 17);
    derived.v_at(i, 7) = static_cast<std::uint8_t>(150 - i % 11);
  }
  run_derived("derived", derived);

  // Search + GPU: the candidate chosen by this component on the CPU must match
  // the GPU prediction byte for byte.
  HostCase search_case(64, 64, 8, 8, 16, 16);
  fill_ramp(search_case);
  cfl::Scratch scratch;
  cfl::compute_luma(search_case.u, scratch);
  const cfl::AlphaPair target =
      alpha::pair_from_joint(3, 6, 2);
  fill_source(search_case, scratch, target);
  alpha::Config config{};
  config.cost.use_rate = false;
  const alpha::CflCdfs cdfs = alpha::default_cfl_cdfs();
  const auto decision = alpha::decide_cfl_alpha(
      search_case.u, search_case.v,
      cfl::SourcePlane{search_case.source_u.data(), kChromaStride,
                       kChromaSize - 1, kChromaSize - 1},
      cfl::SourcePlane{search_case.source_v.data(), kChromaStride,
                       kChromaSize - 1, kChromaSize - 1},
      config, cdfs, scratch);
  if (!decision.has_value() || !decision->chosen.explicit_alpha) {
    std::fprintf(stderr, "search did not choose explicit alpha\n");
    return 1;
  }
  run_explicit("search_choice", search_case, decision->chosen.q8_u,
               decision->chosen.q8_v);

  // Batch entry point.
  HostCase batch_case(64, 64, 8, 8, 16, 16);
  fill_ramp(batch_case);
  run_batch("batch", batch_case);

#ifdef AV2ENC_CUDA_EMULATION
  // Contract regression for the emulation only: a host CflJob array must be
  // rejected, never dereferenced as device memory. On real CUDA this call would
  // fault, which is exactly the bug this check prevents.
  cuda::CflJob host_jobs[1] = {};
  if (cuda::cfl_predict_batch(host_jobs, 1) != cudaErrorInvalidValue) {
    std::fprintf(stderr, "emulation contract: host CflJob* was accepted\n");
    return 1;
  }
  cfl::Block host_blocks[1] = {};
  if (cuda::cfl_predict_derived_batch(host_blocks, nullptr, 1) !=
      cudaErrorInvalidValue) {
    std::fprintf(stderr, "emulation contract: host Block* was accepted\n");
    return 1;
  }
  std::printf("emulation contract: host descriptor arrays rejected\n");
#endif

  if (g_mismatches == 0 && g_alpha_mismatches == 0) {
    std::printf("cfl_alpha_cuda_test: %d cases, %lld samples, %lld alphas, all exact\n",
                g_cases, g_samples, g_alpha_checks);
    return 0;
  }
  std::fprintf(stderr,
               "cfl_alpha_cuda_test: %lld/%lld samples, %lld/%lld alphas "
               "MISMATCH\n",
               g_mismatches, g_samples, g_alpha_mismatches, g_alpha_checks);
  return 1;
}
