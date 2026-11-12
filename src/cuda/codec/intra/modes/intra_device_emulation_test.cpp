// Host emulation of the shared device intra predictor.
//
// This TU compiles intra_predict_device.cuh with the CUDA keywords stubbed and
// one emulated thread, then compares every prediction byte against the CPU
// reference in src/encoder/intra_modes/. It runs on machines without nvcc; the
// real GPU test is src/cuda/intra_modes/intra_modes_test.cu.
//
// Build/run: ./build_emulation_test.sh

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#include "intra_predict_device.cuh"

#include "codec/intra/modes/intra_modes.h"

namespace {

using av2enc::cuda::intra_modes::IntraBlock;
using av2enc::cuda::intra_modes::IntraMode;
using av2enc::cuda::intra_modes::SharedScratch;
using av2enc::intra::IntraEdges;
using av2enc::intra::IntraStatus;
using av2enc::intra::Mode;
using av2enc::intra::NeighborRequest;
using av2enc::intra::PlaneView;
using av2enc::intra::PredictParams;

int g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, int line, const char* detail,
            int value) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 20) {
      std::printf("FAIL line %d: %s (%s=%d)\n", line, what, detail, value);
    }
  }
}

#define EXPECT(cond) expect((cond), #cond, __LINE__, "", 0)
#define EXPECT_EQ(a, b, what) \
  expect((a) == (b), #a " == " #b, __LINE__, what, static_cast<int>(a))

std::uint32_t g_rng = 0x1f123bb5u;
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

const Mode kModes[] = {
    Mode::kDc,      Mode::kV,       Mode::kH,      Mode::kD45,
    Mode::kD135,    Mode::kD113,    Mode::kD157,   Mode::kD203,
    Mode::kD67,     Mode::kSmooth,  Mode::kSmoothV, Mode::kSmoothH,
    Mode::kPaeth};

const int kDeltas[] = {-3, -2, -1, 0, 1, 2, 3};

void run_case(int w, int h, int x, int y, int plane, int mode_index,
              int angle_delta, int have_left, int have_above,
              int num4_above_right, int num4_below_left, int filter_above,
              int filter_left, int enable_edge_filter) {
  const int stride = w + x + 96;
  const int rows = h + y + 96;
  std::vector<std::uint8_t> plane_data(
      static_cast<std::size_t>(stride) * static_cast<std::size_t>(rows));
  for (std::uint8_t& v : plane_data) {
    v = static_cast<std::uint8_t>(next_random() & 0xFFu);
  }
  // Keep the block itself distinct so an accidental read of the current block
  // is visible in the comparison.
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      plane_data[static_cast<std::size_t>(y + i) * stride + x + j] =
          static_cast<std::uint8_t>((i * 7 + j * 13 + 17) & 0xFF);
    }
  }

  NeighborRequest request;
  request.recon = PlaneView{plane_data.data(), stride};
  request.x = x;
  request.y = y;
  request.w = w;
  request.h = h;
  request.max_x = x + w + 32;
  request.max_y = y + h + 32;
  request.have_left = have_left != 0;
  request.have_above = have_above != 0;
  request.num4_above_right = num4_above_right;
  request.num4_below_left = num4_below_left;
  request.color_plane = plane;

  IntraEdges edges;
  if (av2enc::intra::prepare_edges(request, 8, edges) != IntraStatus::kOk) {
    EXPECT(false);
    return;
  }
  PredictParams params;
  params.mode = kModes[mode_index];
  params.angle_delta = angle_delta;
  params.enable_edge_filter = enable_edge_filter != 0;
  params.above_neighbor_is_smooth = filter_above != 0;
  params.left_neighbor_is_smooth = filter_left != 0;

  std::vector<std::uint8_t> want(static_cast<std::size_t>(w) * h);
  if (av2enc::intra::predict_intra(edges, params, want.data(),
                                   static_cast<std::size_t>(w)) !=
      IntraStatus::kOk) {
    std::printf("CPU predict failed mode=%d delta=%d %dx%d x=%d y=%d plane=%d\n",
                static_cast<int>(params.mode), angle_delta, w, h, x, y, plane);
    EXPECT(false);
    return;
  }

  IntraBlock block{};
  block.recon_plane = plane_data.data();
  block.recon_stride = stride;
  block.x0 = x;
  block.y0 = y;
  block.width = w;
  block.height = h;
  block.max_x = x + w + 32;
  block.max_y = y + h + 32;
  block.have_left = have_left;
  block.have_above = have_above;
  block.num4_above_right = num4_above_right;
  block.num4_below_left = num4_below_left;
  block.plane = plane;
  block.mode = static_cast<IntraMode>(static_cast<int>(params.mode));
  block.angle_delta = angle_delta;
  block.filter_type_above = filter_above;
  block.filter_type_left = filter_left;
  block.enable_edge_filter = enable_edge_filter;

  std::vector<std::uint8_t> got(static_cast<std::size_t>(w) * h);
  static SharedScratch scratch;
  blockDim.x = 1;
  threadIdx.x = 0;
  av2enc::cuda::intra_modes::intra_run_block_device(block, got.data(), nullptr,
                                                    &scratch);
  blockDim.x = 0;

  for (std::size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      std::printf(
          "FAIL line 0: mode=%d delta=%d %dx%d x=%d y=%d plane=%d "
          "filter=%d/%d edge=%d sample=%zu got=%u want=%u\n",
          static_cast<int>(params.mode), angle_delta, w, h, x, y, plane,
          filter_above, filter_left, enable_edge_filter, i,
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
  for (int trial = 0; trial < 2600; ++trial) {
    const int w = dims[random_range(0, 4)];
    const int h = dims[random_range(0, 4)];
    const int x = random_range(0, 1) == 0 ? 0 : random_range(1, 80);
    const int y = random_range(0, 1) == 0 ? 0 : random_range(1, 80);
    const int have_left = x > 0 ? 1 : 0;
    const int have_above = y > 0 ? 1 : 0;
    const int plane = random_range(0, 1);
    const int mode_index = random_range(0, 12);
    const bool directional =
        mode_index >= 1 && mode_index <= 8;
    const int angle_delta =
        !directional || random_range(0, 1) == 0
            ? 0
            : kDeltas[random_range(0, 6)];
    const int num4_above_right = random_range(0, w / 4 + 2);
    const int num4_below_left = random_range(0, h / 4 + 2);
    const int filter_above = random_range(0, 1);
    const int filter_left = random_range(0, 1);
    const int enable_edge_filter = random_range(0, 3) == 0 ? 0 : 1;
    run_case(w, h, x, y, plane, mode_index, angle_delta, have_left, have_above,
             num4_above_right, num4_below_left, filter_above, filter_left,
             enable_edge_filter);
  }

  // Every mode and delta at a fixed block/edge state.
  for (int mode = 0; mode < 13; ++mode) {
    for (int delta : kDeltas) {
      const int effective_delta = (mode >= 1 && mode <= 8) ? delta : 0;
      run_case(16, 8, 8, 8, 0, mode, effective_delta, 1, 1, 2, 1, 0, 1, 1);
      run_case(8, 16, 8, 8, 1, mode, effective_delta, 1, 1, 1, 2, 1, 0, 1);
    }
  }

  std::printf("intra_device_emulation_test: %s (%d checks)\n",
              g_failures == 0 ? "PASS" : "FAIL", g_checks);
  return g_failures == 0 ? 0 : 1;
}
