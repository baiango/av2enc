// CUDA MHCCP component test: CPU vs GPU byte-exactness.
//
// Run on the Linux + CUDA box:
//   ./build_gpu_test.sh
//   ARCH=sm_120 ./build_gpu_test.sh
//
// The CPU reference is mhccp.cpp (the correctness oracle). The kernel and the
// reference share mhccp_common.h. Covers explicit + derived parameters,
// directions 0..2, 4x4..32x32, edge/partial blocks, and a batch launch with
// per-job parameter writeback.

#include "mhccp_gpu.h"
#include "mhccp.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace mh = av2enc::cfl::mhccp;
namespace cuda_mh = av2enc::cuda::mhccp;

static_assert(std::is_trivially_copyable_v<mh::Block>);
static_assert(std::is_standard_layout_v<mh::Block>);
static_assert(std::is_trivially_copyable_v<mh::MultiParams>);
static_assert(std::is_trivially_copyable_v<cuda_mh::Job>);

namespace {

constexpr int kLumaStride = 192;
constexpr int kLumaSize = kLumaStride * 192;
constexpr int kChromaStride = 96;
constexpr int kChromaSize = kChromaStride * 96;

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
long long g_param_mismatches = 0;
int g_printed = 0;

std::uint32_t g_rng = 0x2545f491u;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
int random_range(int low, int high) {
  return low + static_cast<int>(next_random() %
                                static_cast<std::uint32_t>(high - low + 1));
}

void note(const char* what, int index, int expected, int actual) {
  if (g_printed >= 12) return;
  std::printf("  MISMATCH [%s] index %d expected %d got %d\n", what, index,
              expected, actual);
  ++g_printed;
}

struct HostCase {
  std::vector<std::uint8_t> luma;
  std::vector<std::uint8_t> chroma;
  mh::Block block;
};

HostCase make_case(int w, int h, int x, int y, int filter) {
  HostCase c;
  c.luma.assign(kLumaSize, 128);
  c.chroma.assign(kChromaSize, 128);
  for (std::uint8_t& v : c.luma) v = static_cast<std::uint8_t>(next_random());
  for (std::uint8_t& v : c.chroma) {
    v = static_cast<std::uint8_t>(next_random());
  }
  mh::Block& b = c.block;
  b.luma = c.luma.data();
  b.luma_stride = kLumaStride;
  b.luma_max_x = 191;
  b.luma_max_y = 191;
  b.chroma = c.chroma.data();
  b.chroma_stride = kChromaStride;
  b.chroma_max_x = 95;
  b.chroma_max_y = 95;
  b.x = x;
  b.y = y;
  b.w = w;
  b.h = h;
  b.bit_depth = 8;
  b.ds_filter = filter;
  b.sub_x = 1;
  b.sub_y = 1;
  b.have_above = y >= 4 ? 1 : 0;
  b.have_left = x >= 4 ? 1 : 0;
  b.num4_above_right = random_range(0, w / 2 + 2);
  b.num4_below_left = random_range(0, h / 2 + 2);
  b.mi_col_end = 96 << 1;
  b.mi_row_end = 96 << 1;
  b.sb_luma_y = ((y << 1) / 64) * 64;
  return c;
}

void compare_case(HostCase& c, int dir, bool derived, std::uint8_t* d_luma,
                  std::uint8_t* d_chroma) {
  CUDA_CHECK(cudaMemcpy(d_luma, c.luma.data(), kLumaSize,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_chroma, c.chroma.data(), kChromaSize,
                        cudaMemcpyHostToDevice));
  mh::Block device_block = c.block;
  device_block.luma = d_luma;
  device_block.chroma = d_chroma;

  mh::MultiParams explicit_params{};
  explicit_params.p[0] = random_range(-2000, 2000);
  explicit_params.p[1] = random_range(-2000, 2000);
  explicit_params.p[2] = random_range(0, 1 << 17);

  mh::MultiParams* d_params = nullptr;
  CUDA_CHECK(cudaMalloc(&d_params, sizeof(mh::MultiParams)));
  CUDA_CHECK(cuda_mh::mhccp_predict(device_block, explicit_params, dir,
                                    derived, d_params));
  CUDA_CHECK(cudaDeviceSynchronize());
  mh::MultiParams gpu_params{};
  CUDA_CHECK(cudaMemcpy(&gpu_params, d_params, sizeof(gpu_params),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_params));

  std::vector<std::uint8_t> gpu_chroma = c.chroma;
  CUDA_CHECK(cudaMemcpy(gpu_chroma.data(), d_chroma, kChromaSize,
                        cudaMemcpyDeviceToHost));

  mh::Block cpu_block = c.block;
  std::vector<std::uint8_t> cpu_chroma = c.chroma;
  cpu_block.chroma = cpu_chroma.data();
  mh::Refs refs;
  if (!mh::capture_refs(cpu_block, refs)) {
    note("capture", 0, 1, 0);
    return;
  }
  const mh::MultiParams cpu_params =
      derived ? mh::derive_multi_params(cpu_block, refs, dir, nullptr)
              : explicit_params;

  if (derived) {
    for (int k = 0; k < 3; ++k) {
      if (gpu_params.p[k] != cpu_params.p[k]) {
        ++g_param_mismatches;
        if (g_param_mismatches <= 8) {
          std::printf("  PARAM MISMATCH dir %d p%d expected %d got %d\n", dir,
                      k, cpu_params.p[k], gpu_params.p[k]);
        }
      }
    }
  }

  int mismatches = 0;
  for (int i = 0; i < c.block.h; ++i) {
    for (int j = 0; j < c.block.w; ++j) {
      const std::size_t index =
          static_cast<std::size_t>(c.block.y + i) * kChromaStride +
          static_cast<std::size_t>(c.block.x + j);
      const int expected =
          mh::predict_sample(cpu_block, refs, cpu_params, dir, i, j);
      if (gpu_chroma[index] != expected) {
        if (mismatches == 0) note("prediction", static_cast<int>(index),
                                  expected, gpu_chroma[index]);
        ++mismatches;
      }
      ++g_samples;
    }
  }
  g_mismatches += mismatches;
}

void test_single() {
  int cases = 0;
  const int sizes[] = {4, 8, 16, 32};
  for (int w : sizes) {
    for (int h : sizes) {
      for (int y = 0; y <= 64; y += (y < 4 ? 4 : 16)) {
        for (int x = 0; x <= 64; x += (x < 4 ? 4 : 16)) {
          HostCase c = make_case(w, h, x, y, random_range(0, 3));
          std::uint8_t* d_luma = nullptr;
          std::uint8_t* d_chroma = nullptr;
          CUDA_CHECK(cudaMalloc(&d_luma, kLumaSize));
          CUDA_CHECK(cudaMalloc(&d_chroma, kChromaSize));
          for (int dir = 0; dir < 3; ++dir) {
            compare_case(c, dir, true, d_luma, d_chroma);
            compare_case(c, dir, false, d_luma, d_chroma);
          }
          CUDA_CHECK(cudaFree(d_luma));
          CUDA_CHECK(cudaFree(d_chroma));
          ++cases;
        }
      }
    }
  }
  std::printf("single: %d cases\n", cases);
}

void test_batch() {
  constexpr int kJobs = 24;
  std::vector<std::uint8_t> luma(kLumaSize);
  std::vector<std::uint8_t> chroma(kChromaSize);
  for (std::uint8_t& v : luma) v = static_cast<std::uint8_t>(next_random());
  for (std::uint8_t& v : chroma) {
    v = static_cast<std::uint8_t>(next_random());
  }

  std::vector<mh::Block> host_blocks(kJobs);
  std::vector<cuda_mh::Job> jobs(kJobs);
  std::vector<int> dirs(kJobs);
  std::vector<mh::MultiParams> explicit_params(kJobs);
  for (int n = 0; n < kJobs; ++n) {
    // 6x4 grid of 16x16 cells in the 96x96 test plane; every job writes a
    // disjoint 8x8 or 16x16 region, so a batch launch has no write races.
    const int w = (n % 3 == 0) ? 16 : 8;
    const int h = (n % 4 == 0) ? 16 : 8;
    const int x = (n % 6) * 16;
    const int y = (n / 6) * 16;
    mh::Block b{};
    b.luma = luma.data();
    b.luma_stride = kLumaStride;
    b.luma_max_x = 191;
    b.luma_max_y = 191;
    b.chroma = chroma.data();
    b.chroma_stride = kChromaStride;
    b.chroma_max_x = 95;
    b.chroma_max_y = 95;
    b.x = x;
    b.y = y;
    b.w = w;
    b.h = h;
    b.bit_depth = 8;
    b.ds_filter = n & 3;
    b.sub_x = 1;
    b.sub_y = 1;
    b.have_above = y >= 4 ? 1 : 0;
    b.have_left = x >= 4 ? 1 : 0;
    b.num4_above_right = 0;
    b.num4_below_left = 0;
    b.mi_col_end = 192;
    b.mi_row_end = 192;
    b.sb_luma_y = 0;
    host_blocks[n] = b;
    dirs[n] = n % 3;
    explicit_params[n].p[0] = 100 + n;
    explicit_params[n].p[1] = -50 + n;
    explicit_params[n].p[2] = 60000 + n;
    jobs[n].block = b;
    jobs[n].params = explicit_params[n];
    jobs[n].cfl_mh_dir = dirs[n];
    jobs[n].derived = 1;
  }

  std::uint8_t* d_luma = nullptr;
  std::uint8_t* d_chroma = nullptr;
  cuda_mh::Job* d_jobs = nullptr;
  mh::MultiParams* d_params = nullptr;
  CUDA_CHECK(cudaMalloc(&d_luma, kLumaSize));
  CUDA_CHECK(cudaMalloc(&d_chroma, kChromaSize));
  CUDA_CHECK(cudaMalloc(&d_jobs, sizeof(cuda_mh::Job) * kJobs));
  CUDA_CHECK(cudaMalloc(&d_params, sizeof(mh::MultiParams) * kJobs));
  CUDA_CHECK(cudaMemcpy(d_luma, luma.data(), kLumaSize,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_chroma, chroma.data(), kChromaSize,
                        cudaMemcpyHostToDevice));
  for (int n = 0; n < kJobs; ++n) {
    jobs[n].block.luma = d_luma;
    jobs[n].block.chroma = d_chroma;
  }
  CUDA_CHECK(cudaMemcpy(d_jobs, jobs.data(), sizeof(cuda_mh::Job) * kJobs,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cuda_mh::mhccp_predict_batch(d_jobs, kJobs, d_params));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint8_t> gpu_chroma(kChromaSize);
  CUDA_CHECK(cudaMemcpy(gpu_chroma.data(), d_chroma, kChromaSize,
                        cudaMemcpyDeviceToHost));
  std::vector<mh::MultiParams> gpu_params(kJobs);
  CUDA_CHECK(cudaMemcpy(gpu_params.data(), d_params,
                        sizeof(mh::MultiParams) * kJobs,
                        cudaMemcpyDeviceToHost));

  // Apply the CPU reference sequentially on a host copy. The batch jobs write
  // disjoint chroma regions, so the order does not matter.
  std::vector<std::uint8_t> cpu_chroma = chroma;
  for (int n = 0; n < kJobs; ++n) {
    mh::Block b = host_blocks[n];
    b.chroma = cpu_chroma.data();
    mh::Refs refs;
    if (!mh::capture_refs(b, refs)) continue;
    const mh::MultiParams p =
        mh::derive_multi_params(b, refs, dirs[n], nullptr);
    for (int k = 0; k < 3; ++k) {
      if (gpu_params[n].p[k] != p.p[k]) {
        ++g_param_mismatches;
      }
    }
    for (int i = 0; i < b.h; ++i) {
      for (int j = 0; j < b.w; ++j) {
        const std::size_t index = static_cast<std::size_t>(b.y + i) *
                                      kChromaStride +
                                  static_cast<std::size_t>(b.x + j);
        const int expected = mh::predict_sample(b, refs, p, dirs[n], i, j);
        if (gpu_chroma[index] != expected) {
          if (g_mismatches == 0) {
            note("batch prediction", static_cast<int>(index), expected,
                 gpu_chroma[index]);
          }
          ++g_mismatches;
        }
        ++g_samples;
      }
    }
  }

  CUDA_CHECK(cudaFree(d_luma));
  CUDA_CHECK(cudaFree(d_chroma));
  CUDA_CHECK(cudaFree(d_jobs));
  CUDA_CHECK(cudaFree(d_params));
  std::printf("batch: %d jobs\n", kJobs);
}

}  // namespace

int main() {
  test_single();
  test_batch();
  std::printf("mhccp_cuda_test: %lld samples, %lld sample mismatches, "
              "%lld parameter mismatches\n",
              g_samples, g_mismatches, g_param_mismatches);
  return (g_mismatches == 0 && g_param_mismatches == 0) ? 0 : 1;
}
