// Host emulation of the shared MHCCP device path.
//
// Compiles mhccp_device.cuh with CUDA keywords stubbed and compares its
// CflRef capture, Q16 parameters and every predicted sample against the CPU
// reference in mhccp.cpp. Runs without nvcc; the real GPU test is
// mhccp_gpu_test.cu.

#include <cstdint>
#include <cstdio>
#include <vector>

#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static

struct EmuDim {
  unsigned x = 0;
  unsigned y = 0;
  unsigned z = 0;
};
[[maybe_unused]] static EmuDim threadIdx;
[[maybe_unused]] static EmuDim blockIdx;
[[maybe_unused]] static EmuDim blockDim;
[[maybe_unused]] static inline void __syncthreads() {}

#include "mhccp_device.cuh"
#include "mhccp.h"

namespace mh = av2enc::cfl::mhccp;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 20) {
      std::printf("FAIL: %s\n", what);
    }
  }
}

std::uint32_t g_rng = 0x51ed270bu;
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

constexpr int kLumaStride = 192;
constexpr int kLumaSize = kLumaStride * 192;
constexpr int kChromaStride = 96;
constexpr int kChromaSize = kChromaStride * 96;

bool run_case(int w, int h, int x, int y, int filter, int dir) {
  std::vector<std::uint8_t> luma(kLumaSize);
  std::vector<std::uint8_t> chroma(kChromaSize);
  for (std::uint8_t& v : luma) v = static_cast<std::uint8_t>(next_random());
  for (std::uint8_t& v : chroma) {
    v = static_cast<std::uint8_t>(next_random());
  }

  mh::Block block{};
  block.luma = luma.data();
  block.luma_stride = kLumaStride;
  block.luma_max_x = 191;
  block.luma_max_y = 191;
  block.chroma = chroma.data();
  block.chroma_stride = kChromaStride;
  block.chroma_max_x = 95;
  block.chroma_max_y = 95;
  block.x = x;
  block.y = y;
  block.w = w;
  block.h = h;
  block.bit_depth = 8;
  block.ds_filter = filter;
  block.sub_x = 1;
  block.sub_y = 1;
  block.have_above = y >= 4 ? 1 : 0;
  block.have_left = x >= 4 ? 1 : 0;
  block.num4_above_right = random_range(0, w / 2 + 2);
  block.num4_below_left = random_range(0, h / 2 + 2);
  block.mi_col_end = 96 << 1;
  block.mi_row_end = 96 << 1;
  block.sb_luma_y = ((y << 1) / 64) * 64;

  if (!mh::valid(block)) {
    return false;
  }

  mh::Refs cpu_refs;
  if (!mh::capture_refs(block, cpu_refs)) {
    return false;
  }
  int count = 0;
  const mh::MultiParams cpu_params =
      mh::derive_multi_params(block, cpu_refs, dir, &count);

  mh::Refs device_refs;
  mh::MultiParams device_params;
  const bool ok = mh::device::prepare(block, dir, true, mh::MultiParams{},
                                      &device_refs, &device_params);
  check(ok, "device prepare");
  if (!ok) {
    return false;
  }
  check(device_refs.above == cpu_refs.above, "emulation above");
  check(device_refs.left == cpu_refs.left, "emulation left");
  check(device_refs.rows == cpu_refs.rows, "emulation rows");
  check(device_refs.cols == cpu_refs.cols, "emulation cols");
  check(device_refs.ref_h == cpu_refs.ref_h, "emulation ref_h");
  check(device_refs.ref_w == cpu_refs.ref_w, "emulation ref_w");
  check(device_refs.target_h == cpu_refs.target_h, "emulation target_h");
  check(device_refs.target_w == cpu_refs.target_w, "emulation target_w");

  int ref_mismatch = 0;
  for (int i = 0; i < mh::kMaxRefDim; ++i) {
    for (int j = 0; j < mh::kMaxRefDim; ++j) {
      if (device_refs.luma[i][j] != cpu_refs.luma[i][j] ||
          device_refs.chroma[i][j] != cpu_refs.chroma[i][j]) {
        ++ref_mismatch;
      }
    }
  }
  check(ref_mismatch == 0, "emulation CflRef samples");

  for (int k = 0; k < 3; ++k) {
    check(device_params.p[k] == cpu_params.p[k], "emulation parameters");
  }

  int sample_mismatch = 0;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int got = mh::device::sample(block, device_refs, device_params,
                                         dir, i, j);
      const int want = mh::predict_sample(block, cpu_refs, cpu_params, dir, i,
                                          j);
      if (got != want) {
        ++sample_mismatch;
      }
    }
  }
  check(sample_mismatch == 0, "emulation prediction");
  return true;
}

}  // namespace

int main() {
  int cases = 0;
  const int sizes[] = {4, 8, 16, 32};
  for (int w : sizes) {
    for (int h : sizes) {
      for (int y = 0; y <= 64; y += (y < 4 ? 4 : 16)) {
        for (int x = 0; x <= 64; x += (x < 4 ? 4 : 16)) {
          const int filter = random_range(0, 3);
          for (int dir = 0; dir < 3; ++dir) {
            if (run_case(w, h, x, y, filter, dir)) {
              ++cases;
            }
          }
        }
      }
    }
  }

  // Explicit-parameter path.
  {
    const int w = 16;
    const int h = 8;
    std::vector<std::uint8_t> luma(kLumaSize, 128);
    std::vector<std::uint8_t> chroma(kChromaSize, 128);
    mh::Block block{};
    block.luma = luma.data();
    block.luma_stride = kLumaStride;
    block.luma_max_x = 191;
    block.luma_max_y = 191;
    block.chroma = chroma.data();
    block.chroma_stride = kChromaStride;
    block.chroma_max_x = 95;
    block.chroma_max_y = 95;
    block.x = 16;
    block.y = 16;
    block.w = w;
    block.h = h;
    block.bit_depth = 8;
    block.have_above = 1;
    block.have_left = 1;
    block.mi_col_end = 192;
    block.mi_row_end = 192;
    mh::MultiParams params{};
    params.p[0] = 300;
    params.p[1] = -120;
    params.p[2] = 64000;
    mh::Refs refs;
    check(mh::capture_refs(block, refs), "explicit capture");

    mh::Refs device_refs;
    mh::MultiParams device_params;
    check(mh::device::prepare(block, 1, false, params, &device_refs,
                              &device_params),
          "explicit prepare");
    int mismatches = 0;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const int got =
            mh::device::sample(block, device_refs, device_params, 1, i, j);
        const int want = mh::predict_sample(block, refs, params, 1, i, j);
        if (got != want) ++mismatches;
      }
    }
    check(mismatches == 0, "explicit emulation prediction");
    ++cases;
  }

  std::printf("mhccp_emulation_test: %d cases, %d checks, %d failures\n", cases,
              g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
