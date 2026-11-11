// Host emulation parity test for the shared AV2 DIP device predictor.
//
// Compiles src/cuda/dip/dip_device.cuh with the CUDA keywords stubbed and
// compares every prediction byte against the independent CPU reference in
// src/encoder/dip/.  Runs without nvcc; the real GPU test is dip_test.cu.

#include <algorithm>
#include <cstddef>
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
static EmuDim blockDim;
static inline void __syncthreads() {}

#include "dip_device.cuh"

#include "encoder/dip/dip.h"
#include "io/y4m.h"

namespace {

using av2enc::dip::Block;
using av2enc::dip::Params;
using av2enc::dip::PlaneView;
using av2enc::dip::Status;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL: %s\n", what);
  }
}

std::uint32_t g_rng = 0x5a17c0deu;
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

constexpr int kValidDims[][2] = {
    {8, 16},  {8, 32},  {8, 64},  {16, 8},  {16, 16},
    {16, 32}, {16, 64}, {32, 8},  {32, 16}, {32, 32},
    {32, 64}, {64, 8},  {64, 16}, {64, 32}, {64, 64},
};

void run_case(const std::vector<std::uint8_t>& plane, int stride, int x, int y,
              int w, int h, int mode, int transpose, bool have_left,
              bool have_above, int num4_ar, int num4_bl, int max_x, int max_y,
              unsigned threads) {
  const PlaneView recon = PlaneView::from_u8(plane.data(), stride);
  Block block;
  block.recon = recon;
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

  Params params;
  params.mode = mode;
  params.transpose = transpose != 0;

  std::vector<std::uint8_t> want(static_cast<std::size_t>(w) *
                                 static_cast<std::size_t>(h));
  const Status status =
      av2enc::dip::predict_block8(block, 8, params, want.data(),
                                  static_cast<std::size_t>(w));
  if (status != Status::kOk) {
    std::printf("CPU DIP rejected a conforming case (%dx%d mode=%d)\n", w, h,
                mode);
    ++g_checks;
    ++g_failures;
    return;
  }

  av2enc::cuda::dip::DipBlock gpu_block{};
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

  av2enc::cuda::dip::DipParams gpu_params{};
  gpu_params.mode = mode;
  gpu_params.transpose = transpose;

  std::vector<std::uint8_t> got(static_cast<std::size_t>(w) *
                                static_cast<std::size_t>(h));
  static av2enc::cuda::dip::DipSharedScratch scratch;
  blockDim.x = threads;
  threadIdx.x = 0;
  av2enc::cuda::dip::dip_run_block_device(gpu_block, gpu_params, got.data(),
                                          &scratch);
  blockDim.x = 0;

  for (std::size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      std::printf(
          "FAIL %dx%d x=%d y=%d mode=%d tr=%d avail=%d%d sample=%zu "
          "got=%u want=%u\n",
          w, h, x, y, mode, transpose, have_left ? 1 : 0, have_above ? 1 : 0,
          i, static_cast<unsigned>(got[i]), static_cast<unsigned>(want[i]));
      ++g_checks;
      ++g_failures;
      return;
    }
  }
  ++g_checks;
}

}  // namespace

int main(int argc, char** argv) {
  constexpr int kStride = 160;
  constexpr int kRows = 160;
  std::vector<std::uint8_t> plane(static_cast<std::size_t>(kStride) * kRows);
  for (std::uint8_t& v : plane) {
    v = static_cast<std::uint8_t>(next_random() & 0xFFu);
  }

  const int positions[][2] = {{0, 0},   {8, 8},   {16, 24},
                              {32, 16}, {64, 48}, {80, 80}};
  int cases = 0;
  for (const auto& dim : kValidDims) {
    const int w = dim[0];
    const int h = dim[1];
    for (const auto& pos : positions) {
      const int x = pos[0];
      const int y = pos[1];
      if (x + w > 120 || y + h > 120) continue;
      for (int avail = 0; avail < 4; ++avail) {
        const bool have_left = (avail & 1) != 0;
        const bool have_above = (avail & 2) != 0;
        if (have_left && x == 0) continue;
        if (have_above && y == 0) continue;
        const int max_x = std::min(120, x + w - 1) - ((avail == 3) ? 2 : 0);
        const int max_y = std::min(120, y + h - 1) - ((avail == 3) ? 3 : 0);
        if (max_x < x || max_y < y) continue;
        const int num4_ar = random_range(0, w / 4 + 3);
        const int num4_bl = random_range(0, h / 4 + 3);
        for (int mode = 0; mode < 6; ++mode) {
          for (int transpose = 0; transpose < 2; ++transpose) {
            run_case(plane, kStride, x, y, w, h, mode, transpose, have_left,
                     have_above, num4_ar, num4_bl, max_x, max_y, 1);
            ++cases;
          }
        }
      }
    }
  }

  // Real luma blocks: same shared device code against the CPU reference on
  // decoded fixture samples.
  if (argc > 1) {
    av2enc::Y4mReader reader{argv[1]};
    std::string error;
    av2enc::Y4mFrame frame;
    if (reader.open(error) &&
        reader.read_frame(frame, error) == av2enc::Y4mReadResult::kFrame) {
      const auto luma = frame.y();
      const int width = static_cast<int>(frame.width());
      const int height = static_cast<int>(frame.height());
      std::vector<std::uint8_t> real(luma.begin(), luma.end());
      for (const auto& dim : kValidDims) {
        const int w = dim[0];
        const int h = dim[1];
        if (w > width || h > height) continue;
        const int xs[] = {0, std::max(0, width - w)};
        const int ys[] = {0, std::max(0, height - h)};
        for (int x : xs) {
          for (int y : ys) {
            if (x + w > width || y + h > height) continue;
            for (int mode = 0; mode < 6; ++mode) {
              run_case(real, width, x, y, w, h, mode, mode & 1, x > 0,
                       y > 0, 4, 4, width - 1, height - 1, 1);
              ++cases;
            }
          }
        }
      }
    }
  }

  check(cases > 0, "cases exercised");
  std::printf("dip_emulation_test: %d cases, %d checks, %d failures\n", cases,
              g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
