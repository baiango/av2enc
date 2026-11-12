// Host emulation of the shared AV2 MRL device predictor.
//
// Compiles mrl_device.cuh with CUDA keywords stubbed and one emulated thread,
// then compares every prediction byte against the CPU reference in
// src/encoder/mrl/. Runs without nvcc; the real GPU test is mrl_test.cu.

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
static EmuDim threadIdx;
static EmuDim blockIdx;
static EmuDim blockDim;
static inline void __syncthreads() {}

#include "mrl_device.cuh"

#include "codec/intra/mrl/core/mrl.h"

namespace {

using av2enc::mrl::Block;
using av2enc::mrl::MrlStatus;
using av2enc::mrl::Params;

int g_checks = 0;
int g_failures = 0;

std::uint32_t g_rng = 0x7f4a7c15u;
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

constexpr int kModes[8] = {1, 2, 3, 4, 5, 6, 7, 8};
constexpr int kDeltas[7] = {-3, -2, -1, 0, 1, 2, 3};

void run_case(int w, int h, int x, int y, int mode, int delta, int mrl,
              bool sec, bool smooth_above, bool smooth_left) {
  const int stride = 512;
  std::vector<std::uint8_t> plane(static_cast<std::size_t>(stride) * 512);
  for (std::uint8_t& v : plane) {
    v = static_cast<std::uint8_t>(next_random() & 0xFFu);
  }
  const int max_x = 500;
  const int max_y = 500;
  const bool have_left = x > 0;
  const bool have_above = y > 0;
  const int num4_ar = random_range(0, w / 4 + 2);
  const int num4_bl = random_range(0, h / 4 + 2);
  const bool effective_sec = sec && !(w == 4 && h == 4);

  Block block;
  block.recon = av2enc::intra::PlaneView{plane.data(), stride};
  block.x = x;
  block.y = y;
  block.w = w;
  block.h = h;
  block.max_x = max_x;
  block.max_y = max_y;
  block.have_left = have_left;
  block.have_above = have_above;
  block.num4_above_right = num4_ar;
  block.num4_below_left = num4_bl;
  block.plane = 0;
  block.sb_height = 64;

  Params params;
  params.mode = static_cast<av2enc::intra::Mode>(mode);
  params.angle_delta = delta;
  params.mrl_index = mrl;
  params.mrl_sec_index = effective_sec;
  params.enable_edge_filter = true;
  params.above_neighbor_is_smooth = smooth_above;
  params.left_neighbor_is_smooth = smooth_left;

  std::vector<std::uint8_t> want(static_cast<std::size_t>(w) * h);
  if (av2enc::mrl::predict_block(block, 8, params, want.data(), w) !=
      MrlStatus::kOk) {
    std::printf("CPU mrl rejected mrl=%d mode=%d %dx%d x=%d y=%d\n", mrl, mode,
                w, h, x, y);
    ++g_checks;
    ++g_failures;
    return;
  }

  av2enc::cuda::mrl::MrlBlock gpu_block{};
  gpu_block.recon = plane.data();
  gpu_block.recon_stride = stride;
  gpu_block.x = x;
  gpu_block.y = y;
  gpu_block.w = w;
  gpu_block.h = h;
  gpu_block.max_x = max_x;
  gpu_block.max_y = max_y;
  gpu_block.have_left = have_left ? 1 : 0;
  gpu_block.have_above = have_above ? 1 : 0;
  gpu_block.num4_above_right = num4_ar;
  gpu_block.num4_below_left = num4_bl;
  gpu_block.plane = 0;
  gpu_block.sb_height = 64;

  av2enc::cuda::mrl::MrlParams gpu_params{};
  gpu_params.mode = mode;
  gpu_params.angle_delta = delta;
  gpu_params.mrl_index = mrl;
  gpu_params.mrl_sec_index = effective_sec ? 1 : 0;
  gpu_params.enable_edge_filter = 1;
  gpu_params.filter_type_above = smooth_above ? 1 : 0;
  gpu_params.filter_type_left = smooth_left ? 1 : 0;

  std::vector<std::uint8_t> got(static_cast<std::size_t>(w) * h);
  static av2enc::cuda::mrl::MrlSharedScratch scratch;
  blockDim.x = 1;
  threadIdx.x = 0;
  av2enc::cuda::mrl::mrl_run_block_device(gpu_block, gpu_params, got.data(),
                                          &scratch);
  blockDim.x = 0;

  for (std::size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      std::printf(
          "FAIL mrl=%d sec=%d mode=%d delta=%d %dx%d x=%d y=%d sample=%zu "
          "got=%u want=%u\n",
          mrl, effective_sec ? 1 : 0, mode, delta, w, h, x, y, i,
          static_cast<unsigned>(got[i]), static_cast<unsigned>(want[i]));
      ++g_checks;
      ++g_failures;
      return;
    }
  }
  ++g_checks;
}

}  // namespace

int main() {
  const int dims[] = {4, 8, 16, 32, 64};
  const int positions[][2] = {{0, 0},  {4, 4},   {8, 4},   {4, 8},
                              {12, 20}, {32, 32}, {64, 4},  {4, 64},
                              {60, 60}};
  for (int trial = 0; trial < 8000; ++trial) {
    const int w = dims[random_range(0, 4)];
    const int h = dims[random_range(0, 4)];
    const auto& pos = positions[random_range(
        0, static_cast<int>(sizeof(positions) / sizeof(positions[0])) - 1)];
    const int mode = kModes[random_range(0, 7)];
    const int delta = kDeltas[random_range(0, 6)];
    const int mrl = random_range(0, 3);
    const bool sec = mrl > 0 && random_range(0, 1) != 0;
    run_case(w, h, pos[0], pos[1], mode, delta, mrl, sec,
             random_range(0, 1) != 0, random_range(0, 1) != 0);
  }
  // Targeted full sweep on representative geometry.
  const int geo[][4] = {{0, 0, 4, 4},    {4, 4, 8, 8},   {8, 4, 16, 8},
                        {4, 8, 8, 16},   {12, 20, 32, 32}, {32, 32, 64, 4},
                        {4, 64, 4, 64},  {60, 60, 8, 8}};
  for (const auto& g : geo) {
    for (int mode : kModes) {
      for (int delta : kDeltas) {
        for (int mrl = 0; mrl <= 3; ++mrl) {
          for (int sec = 0; sec <= 1; ++sec) {
            if (mrl == 0 && sec != 0) continue;
            if (g[2] == 4 && g[3] == 4 && sec != 0) continue;
            run_case(g[2], g[3], g[0], g[1], mode, delta, mrl, sec != 0, false,
                     false);
            run_case(g[2], g[3], g[0], g[1], mode, delta, mrl, sec != 0, true,
                     false);
            run_case(g[2], g[3], g[0], g[1], mode, delta, mrl, sec != 0, false,
                     true);
          }
        }
      }
    }
  }

  std::printf("mrl_emulation_test: %s (%d checks)\n",
              g_failures == 0 ? "PASS" : "FAIL", g_checks);
  return g_failures == 0 ? 0 : 1;
}
