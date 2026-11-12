// CUDA CfL component test: CPU vs GPU byte-exactness.
//
// The CPU reference is src/encoder/cfl/cfl.cpp (the correctness oracle). The
// kernel shares cfl_common.h integer helpers with it. This test covers:
//   * explicit alpha, all cfl_ds_filter_index values 0..3
//   * derived alpha (CFL_DERIVED_ALPHA)
//   * 4x4 .. 32x32 chroma blocks, top-left / interior / SB-top / frame edge
//   * partial right/bottom blocks that reconstruct into the padded plane
//   * batch launches (one CUDA block per job)
//
// Build/run with build_cfl_test.sh on the sm_86 box.

#include "cfl.h"

#include "../../../../codec/intra/cfl/cfl.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <type_traits>
#include <vector>

namespace cuda = av2enc::cuda;
namespace cfl = av2enc::cfl;

// The block descriptor crosses the host/device boundary by value.
static_assert(std::is_trivially_copyable_v<cfl::Block>);
static_assert(std::is_standard_layout_v<cfl::Block>);
static_assert(std::is_trivially_copyable_v<cuda::CflJob>);

namespace {

constexpr int kLumaStride = 192;
constexpr int kLumaPlane = kLumaStride * 192;
constexpr int kChromaStride = 128;
constexpr int kChromaPlane = kChromaStride * 128;

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    const cudaError_t err_ = (expr);                                         \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error: %s (%d) from %s at %s:%d\n",         \
                   cudaGetErrorString(err_), static_cast<int>(err_), #expr,  \
                   __FILE__, __LINE__);                                      \
      std::exit(1);                                                          \
    }                                                                        \
  } while (false)

long long g_samples = 0;
long long g_mismatches = 0;
long long g_alpha_checks = 0;
long long g_alpha_mismatches = 0;
int g_printed = 0;

void note(const char* what, int index, int expected, int actual) {
  if (g_printed >= 16) {
    return;
  }
  std::printf("  MISMATCH [%s] index %d expected %d got %d\n", what, index,
              expected, actual);
  ++g_printed;
}

std::uint8_t* luma_at(std::vector<std::uint8_t>& plane, int row, int col) {
  return &plane[static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(kLumaStride) +
                static_cast<std::size_t>(col)];
}

std::uint8_t* chroma_at(std::vector<std::uint8_t>& plane, int row, int col) {
  return &plane[static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(kChromaStride) +
                static_cast<std::size_t>(col)];
}

struct HostCase {
  std::vector<std::uint8_t> luma;
  std::vector<std::uint8_t> chroma;
  cfl::Block block;
};

HostCase make_case(std::mt19937& rng, int coded_w, int coded_h, int w, int h,
                   int x, int y, int filter, bool random_values) {
  HostCase c;
  c.luma.assign(kLumaPlane, 128);
  c.chroma.assign(kChromaPlane, 128);
  if (random_values) {
    for (auto& value : c.luma) {
      value = static_cast<std::uint8_t>(rng() & 0xFF);
    }
    for (auto& value : c.chroma) {
      value = static_cast<std::uint8_t>(rng() & 0xFF);
    }
  } else {
    for (int row = 0; row < 192; ++row) {
      for (int col = 0; col < 192; ++col) {
        *luma_at(c.luma, row, col) =
            static_cast<std::uint8_t>((row * 7 + col * 11) & 0xFF);
      }
    }
    for (int row = 0; row < 128; ++row) {
      for (int col = 0; col < 128; ++col) {
        *chroma_at(c.chroma, row, col) =
            static_cast<std::uint8_t>((row * 3 + col * 5) & 0xFF);
      }
    }
  }
  cfl::Block& b = c.block;
  b.luma = c.luma.data();
  b.luma_stride = kLumaStride;
  b.luma_max_x = 191;
  b.luma_max_y = 191;
  b.sb_top_y = ((y * 2) / 64) * 64;
  b.chroma = c.chroma.data();
  b.chroma_stride = kChromaStride;
  b.x = x;
  b.y = y;
  b.w = w;
  b.h = h;
  b.coded_width = coded_w;
  b.coded_height = coded_h;
  b.have_above = y >= 4 ? 1 : 0;
  b.have_left = x >= 4 ? 1 : 0;
  b.bit_depth = 8;
  b.ds_filter = filter;
  b.sub_x = 1;
  b.sub_y = 1;
  return c;
}

// Copies the host case into the (reused) device planes and applies CfL on the
// GPU. Returns the device alpha for derived mode.
int run_gpu(const cfl::Block& host_block, const std::vector<std::uint8_t>& luma,
            const std::vector<std::uint8_t>& chroma, int alpha_q8,
            bool derived, std::uint8_t* d_luma, std::uint8_t* d_chroma) {
  CUDA_CHECK(cudaMemcpy(d_luma, luma.data(), kLumaPlane,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_chroma, chroma.data(), kChromaPlane,
                        cudaMemcpyHostToDevice));
  cfl::Block device_block = host_block;
  device_block.luma = d_luma;
  device_block.chroma = d_chroma;
  if (derived) {
    int* d_alpha = nullptr;
    CUDA_CHECK(cudaMalloc(&d_alpha, sizeof(int)));
    CUDA_CHECK(cuda::cfl_predict_derived(device_block, d_alpha));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&alpha_q8, d_alpha, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_alpha));
  } else {
    CUDA_CHECK(cuda::cfl_predict(device_block, alpha_q8));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  return alpha_q8;
}

void compare_case(const HostCase& c, int alpha_q8, bool derived,
                  std::uint8_t* d_luma, std::uint8_t* d_chroma) {
  std::vector<std::uint8_t> gpu_chroma = c.chroma;
  const int used_alpha = run_gpu(c.block, c.luma, c.chroma, alpha_q8, derived,
                                 d_luma, d_chroma);
  CUDA_CHECK(cudaMemcpy(gpu_chroma.data(), d_chroma, kChromaPlane,
                        cudaMemcpyDeviceToHost));

  // CPU reference on its own copy.
  cfl::Block cpu_block = c.block;
  std::vector<std::uint8_t> cpu_chroma = c.chroma;
  cpu_block.chroma = cpu_chroma.data();
  cfl::Scratch scratch;
  cfl::compute_luma(cpu_block, scratch);
  const int expected_alpha =
      derived ? cfl::derive_implicit_alpha(cpu_block, scratch) : alpha_q8;
  cfl::apply_prediction(cpu_block, expected_alpha, scratch);

  if (derived) {
    ++g_alpha_checks;
    if (used_alpha != expected_alpha) {
      ++g_alpha_mismatches;
      note("derived_alpha", 0, expected_alpha, used_alpha);
    }
  }

  // Only the coded block is compared for the GPU (it writes w*h samples); the
  // CPU path also writes only w*h, so full-plane comparison is valid.
  for (int i = 0; i < c.block.h; ++i) {
    for (int j = 0; j < c.block.w; ++j) {
      const std::size_t index =
          static_cast<std::size_t>(c.block.y + i) * kChromaStride +
          static_cast<std::size_t>(c.block.x + j);
      ++g_samples;
      if (gpu_chroma[index] != cpu_chroma[index]) {
        ++g_mismatches;
        note("prediction", static_cast<int>(index), cpu_chroma[index],
             gpu_chroma[index]);
      }
    }
  }
}

void run_case_set(std::uint8_t* d_luma, std::uint8_t* d_chroma) {
  // Deterministic geometry sweep.
  const int sizes[] = {4, 8, 16, 32};
  const int xs[] = {0, 4, 8, 16, 32, 64, 96};
  const int ys[] = {0, 4, 8, 16, 32, 64, 96};
  std::mt19937 rng(12345);
  for (int filter = 0; filter <= 3; ++filter) {
    for (int size : sizes) {
      for (int x : xs) {
        for (int y : ys) {
          if (x + size > kChromaStride || y + size > kChromaStride ||
              x + size > 128 || y + size > 128) {
            continue;
          }
          HostCase c = make_case(rng, 128, 128, size, size, x, y, filter, false);
          compare_case(c, 8 * 32, false, d_luma, d_chroma);
          compare_case(c, -5 * 32, false, d_luma, d_chroma);
          compare_case(c, 0, false, d_luma, d_chroma);
        }
      }
    }
  }

  // Partial right/bottom blocks: the coded grid is smaller than the coded
  // block, so luma-average replication and padded-plane reads are exercised.
  for (int filter = 0; filter <= 3; ++filter) {
    for (int size : sizes) {
      HostCase c = make_case(rng, 20, 20, size, size, 16, 16, filter, false);
      compare_case(c, 3 * 32, false, d_luma, d_chroma);
      compare_case(c, 0, true, d_luma, d_chroma);
    }
  }

  // Randomised geometry and alpha, explicit and derived.
  for (int iteration = 0; iteration < 300; ++iteration) {
    const int coded_w = 8 + static_cast<int>(rng() % 56);
    const int coded_h = 8 + static_cast<int>(rng() % 56);
    const int w = 1 + static_cast<int>(rng() % 32);
    const int h = 1 + static_cast<int>(rng() % 32);
    const int x = static_cast<int>(rng() % static_cast<unsigned>(coded_w));
    const int y = static_cast<int>(rng() % static_cast<unsigned>(coded_h));
    if (x + w > kChromaStride || y + h > kChromaStride) {
      continue;
    }
    const int filter = static_cast<int>(rng() % 4);
    HostCase c = make_case(rng, coded_w, coded_h, w, h, x, y, filter, true);
    const int alpha = static_cast<int>(rng() % 1023) - 511;
    compare_case(c, alpha, false, d_luma, d_chroma);
    compare_case(c, 0, true, d_luma, d_chroma);
  }
}

void run_batch(std::uint8_t* d_luma_base, std::uint8_t* d_chroma_base) {
  constexpr int kJobs = 96;
  std::mt19937 rng(999);
  std::vector<HostCase> host;
  std::vector<cuda::CflJob> jobs(static_cast<std::size_t>(kJobs));
  std::vector<cfl::Block> blocks(static_cast<std::size_t>(kJobs));
  std::vector<std::uint8_t> luma_all(
      static_cast<std::size_t>(kJobs) * kLumaPlane);
  std::vector<std::uint8_t> chroma_all(
      static_cast<std::size_t>(kJobs) * kChromaPlane);
  host.reserve(kJobs);

  for (int job = 0; job < kJobs; ++job) {
    const int coded_w = 8 + static_cast<int>(rng() % 56);
    const int coded_h = 8 + static_cast<int>(rng() % 56);
    const int w = 1 + static_cast<int>(rng() % 32);
    const int h = 1 + static_cast<int>(rng() % 32);
    const int x = static_cast<int>(rng() % static_cast<unsigned>(coded_w));
    const int y = static_cast<int>(rng() % static_cast<unsigned>(coded_h));
    const int filter = static_cast<int>(rng() % 4);
    HostCase c = make_case(rng, coded_w, coded_h, w, h, x, y, filter, true);
    const std::size_t luma_offset =
        static_cast<std::size_t>(job) * kLumaPlane;
    const std::size_t chroma_offset =
        static_cast<std::size_t>(job) * kChromaPlane;
    std::copy(c.luma.begin(), c.luma.end(), luma_all.begin() + luma_offset);
    std::copy(c.chroma.begin(), c.chroma.end(),
              chroma_all.begin() + chroma_offset);
    cfl::Block device_block = c.block;
    device_block.luma = d_luma_base + luma_offset;
    device_block.chroma = d_chroma_base + chroma_offset;
    jobs[static_cast<std::size_t>(job)].block = device_block;
    jobs[static_cast<std::size_t>(job)].alpha_q8 =
        (static_cast<int>(rng() % 17) - 8) * 32;
    blocks[static_cast<std::size_t>(job)] = device_block;
    host.push_back(std::move(c));
  }

  CUDA_CHECK(cudaMemcpy(d_luma_base, luma_all.data(),
                        luma_all.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice));

  cuda::CflJob* d_jobs = nullptr;
  CUDA_CHECK(cudaMalloc(&d_jobs, sizeof(cuda::CflJob) * kJobs));
  CUDA_CHECK(cudaMemcpy(d_jobs, jobs.data(), sizeof(cuda::CflJob) * kJobs,
                        cudaMemcpyHostToDevice));
  cfl::Block* d_blocks = nullptr;
  int* d_alpha = nullptr;
  CUDA_CHECK(cudaMalloc(&d_blocks, sizeof(cfl::Block) * kJobs));
  CUDA_CHECK(cudaMalloc(&d_alpha, sizeof(int) * kJobs));
  CUDA_CHECK(cudaMemcpy(d_blocks, blocks.data(), sizeof(cfl::Block) * kJobs,
                        cudaMemcpyHostToDevice));

  std::vector<std::uint8_t> gpu_chroma(chroma_all.size());
  std::vector<int> gpu_alpha(static_cast<std::size_t>(kJobs), 0);

  // Explicit batch.
  CUDA_CHECK(cudaMemcpy(d_chroma_base, chroma_all.data(),
                        chroma_all.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cuda::cfl_predict_batch(d_jobs, kJobs));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(gpu_chroma.data(), d_chroma_base,
                        chroma_all.size() * sizeof(std::uint8_t),
                        cudaMemcpyDeviceToHost));

  // Derived batch (re-upload the DC planes; the explicit run modified them).
  CUDA_CHECK(cudaMemcpy(d_chroma_base, chroma_all.data(),
                        chroma_all.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cuda::cfl_predict_derived_batch(d_blocks, d_alpha, kJobs));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint8_t> gpu_derived(chroma_all.size());
  CUDA_CHECK(cudaMemcpy(gpu_derived.data(), d_chroma_base,
                        chroma_all.size() * sizeof(std::uint8_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gpu_alpha.data(), d_alpha, sizeof(int) * kJobs,
                        cudaMemcpyDeviceToHost));

  for (int job = 0; job < kJobs; ++job) {
    HostCase& c = host[static_cast<std::size_t>(job)];
    const std::size_t chroma_offset =
        static_cast<std::size_t>(job) * kChromaPlane;
    const std::uint8_t* explicit_plane = gpu_chroma.data() + chroma_offset;
    const std::uint8_t* derived_plane = gpu_derived.data() + chroma_offset;
    const int alpha = jobs[static_cast<std::size_t>(job)].alpha_q8;

    cfl::Scratch scratch;
    cfl::Block cpu_block = c.block;

    std::vector<std::uint8_t> explicit_cpu = c.chroma;
    cpu_block.chroma = explicit_cpu.data();
    cfl::compute_luma(cpu_block, scratch);
    cfl::apply_prediction(cpu_block, alpha, scratch);

    std::vector<std::uint8_t> derived_cpu = c.chroma;
    cpu_block.chroma = derived_cpu.data();
    const int derived_alpha = cfl::derive_implicit_alpha(cpu_block, scratch);
    cfl::apply_prediction(cpu_block, derived_alpha, scratch);

    ++g_alpha_checks;
    if (gpu_alpha[static_cast<std::size_t>(job)] != derived_alpha) {
      ++g_alpha_mismatches;
      note("batch_derived_alpha", job, derived_alpha,
           gpu_alpha[static_cast<std::size_t>(job)]);
    }

    for (int i = 0; i < c.block.h; ++i) {
      for (int j = 0; j < c.block.w; ++j) {
        const std::size_t index =
            static_cast<std::size_t>(c.block.y + i) * kChromaStride +
            static_cast<std::size_t>(c.block.x + j);
        ++g_samples;
        if (explicit_plane[index] != explicit_cpu[index]) {
          ++g_mismatches;
          note("batch_explicit", job, explicit_cpu[index],
               explicit_plane[index]);
        }
        ++g_samples;
        if (derived_plane[index] != derived_cpu[index]) {
          ++g_mismatches;
          note("batch_derived", job, derived_cpu[index], derived_plane[index]);
        }
      }
    }
  }

  CUDA_CHECK(cudaFree(d_jobs));
  CUDA_CHECK(cudaFree(d_blocks));
  CUDA_CHECK(cudaFree(d_alpha));
}

}  // namespace

int main() {
  std::uint8_t* d_luma = nullptr;
  std::uint8_t* d_chroma = nullptr;
  std::uint8_t* d_luma_batch = nullptr;
  std::uint8_t* d_chroma_batch = nullptr;
  constexpr int kBatchJobs = 96;
  CUDA_CHECK(cudaMalloc(&d_luma, kLumaPlane));
  CUDA_CHECK(cudaMalloc(&d_chroma, kChromaPlane));
  CUDA_CHECK(cudaMalloc(&d_luma_batch,
                        static_cast<std::size_t>(kBatchJobs) * kLumaPlane));
  CUDA_CHECK(cudaMalloc(&d_chroma_batch,
                        static_cast<std::size_t>(kBatchJobs) * kChromaPlane));

  run_case_set(d_luma, d_chroma);
  run_batch(d_luma_batch, d_chroma_batch);

  CUDA_CHECK(cudaFree(d_luma));
  CUDA_CHECK(cudaFree(d_chroma));
  CUDA_CHECK(cudaFree(d_luma_batch));
  CUDA_CHECK(cudaFree(d_chroma_batch));

  if (g_mismatches == 0 && g_alpha_mismatches == 0) {
    std::printf("cfl_cuda_test: %lld samples, %lld derived alphas, all exact\n",
                g_samples, g_alpha_checks);
    return 0;
  }
  std::fprintf(stderr,
               "cfl_cuda_test: %lld/%lld samples and %lld/%lld alphas FAILED\n",
               g_mismatches, g_samples, g_alpha_mismatches, g_alpha_checks);
  return 1;
}
