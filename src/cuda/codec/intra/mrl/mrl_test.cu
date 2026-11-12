// CPU-vs-GPU parity test for the AV2 MRL predictor.
//
// Every case builds a random reconstructed luma plane, runs the CPU reference
// (src/encoder/mrl/) and the CUDA kernel on the same descriptor, and requires
// byte-for-byte equality. Includes mrl_index 0..3, all directional modes and
// angle deltas, secondary-line blends, frame/SB edges, smooth-neighbour flags
// and 4x4..64x64 blocks.
//
// Build/run: ./build_mrl_test.sh

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda/codec/intra/mrl/mrl.h"
#include "codec/intra/mrl/core/mrl.h"

namespace {

using av2enc::mrl::Block;
using av2enc::mrl::MrlStatus;
using av2enc::mrl::Params;
namespace gpu = av2enc::cuda::mrl;

int g_checks = 0;
int g_failures = 0;

void fail(const char* what, int detail = 0) {
  ++g_checks;
  ++g_failures;
  if (g_failures <= 25) {
    std::printf("FAIL: %s (%d)\n", what, detail);
  }
}

void expect(bool condition, const char* what) {
  ++g_checks;
  if (!condition) fail(what);
}

#define CUDA_OK(expr)                                                          \
  do {                                                                         \
    const cudaError_t status_ = (expr);                                        \
    ++g_checks;                                                                \
    if (status_ != cudaSuccess) {                                              \
      ++g_failures;                                                            \
      std::printf("CUDA error %d at line %d\n", status_, __LINE__);            \
    }                                                                          \
  } while (0)

std::uint32_t g_rng = 0x9e3779b9u;
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
constexpr int kPlaneStride = 512;
constexpr int kPlaneRows = 512;

struct DevicePlane {
  std::uint8_t* data = nullptr;
  DevicePlane() = default;
  DevicePlane(const DevicePlane&) = delete;
  DevicePlane& operator=(const DevicePlane&) = delete;
  ~DevicePlane() {
    if (data != nullptr) cudaFree(data);
  }
};

void run_case(const std::vector<std::uint8_t>& plane, int x, int y, int w,
              int h, int mode, int delta, int mrl, bool sec, bool smooth_above,
              bool smooth_left, int num4_ar, int num4_bl, DevicePlane& device,
              gpu::MrlJob* jobs, std::uint8_t* d_pred,
              std::uint8_t* d_pred2) {
  const int max_x = kPlaneStride - 40;
  const int max_y = kPlaneRows - 40;
  const bool have_left = x > 0;
  const bool have_above = y > 0;
  const bool effective_sec = sec && !(w == 4 && h == 4);

  Block block;
  block.recon = av2enc::intra::PlaneView{plane.data(), kPlaneStride};
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
    fail("CPU MRL rejected a conforming case", mrl);
    return;
  }

  gpu::MrlBlock gb{};
  gb.recon = device.data;
  gb.recon_stride = kPlaneStride;
  gb.x = x;
  gb.y = y;
  gb.w = w;
  gb.h = h;
  gb.max_x = max_x;
  gb.max_y = max_y;
  gb.have_left = have_left ? 1 : 0;
  gb.have_above = have_above ? 1 : 0;
  gb.num4_above_right = num4_ar;
  gb.num4_below_left = num4_bl;
  gb.plane = 0;
  gb.sb_height = 64;

  gpu::MrlParams gp{};
  gp.mode = mode;
  gp.angle_delta = delta;
  gp.mrl_index = mrl;
  gp.mrl_sec_index = effective_sec ? 1 : 0;
  gp.enable_edge_filter = 1;
  gp.filter_type_above = smooth_above ? 1 : 0;
  gp.filter_type_left = smooth_left ? 1 : 0;

  std::vector<std::uint8_t> got(want.size());
  gpu::MrlJob job;
  job.block = gb;
  job.params = gp;
  job.pred = d_pred;
  CUDA_OK(gpu::mrl_predict(gb, gp, d_pred, nullptr));
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(got.data(), d_pred, want.size(), cudaMemcpyDeviceToHost));
  const bool single_ok = memcmp(got.data(), want.data(), want.size()) == 0;
  if (!single_ok) {
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (got[i] != want[i]) {
        std::printf(
            "FAIL mrl=%d sec=%d mode=%d delta=%d %dx%d x=%d y=%d sample=%zu "
            "gpu=%u cpu=%u\n",
            mrl, effective_sec ? 1 : 0, mode, delta, w, h, x, y, i,
            static_cast<unsigned>(got[i]), static_cast<unsigned>(want[i]));
        break;
      }
    }
    fail("single-block prediction mismatch", mrl);
  }

  // Batch path: two independent descriptors with distinct destinations.
  // `jobs` is DEVICE memory (cudaMalloc); the descriptors must be staged in a
  // host array and copied. Writing jobs[0] from the host was the segfault:
  // cudaMalloc memory is not host-accessible.
  gpu::MrlJob host_jobs[2] = {job, job};
  host_jobs[0].pred = d_pred;
  host_jobs[1].pred = d_pred2;
  CUDA_OK(cudaMemcpy(jobs, host_jobs, sizeof(host_jobs),
                     cudaMemcpyHostToDevice));
  CUDA_OK(gpu::mrl_predict_batch(jobs, 2, nullptr));
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<std::uint8_t> batch0(want.size());
  std::vector<std::uint8_t> batch1(want.size());
  CUDA_OK(cudaMemcpy(batch0.data(), d_pred, want.size(),
                     cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(batch1.data(), d_pred2, want.size(),
                     cudaMemcpyDeviceToHost));
  if (memcmp(batch0.data(), want.data(), want.size()) != 0 ||
      memcmp(batch1.data(), want.data(), want.size()) != 0) {
    fail("batch prediction mismatch", mrl);
  }
}

void availability_cases() {
  const std::vector<std::uint8_t> plane(kPlaneStride * kPlaneRows, 100);
  DevicePlane device;
  CUDA_OK(cudaMalloc(&device.data, plane.size()));
  CUDA_OK(cudaMemcpy(device.data, plane.data(), plane.size(),
                     cudaMemcpyHostToDevice));
  gpu::MrlBlock b{};
  b.recon = device.data;
  b.recon_stride = kPlaneStride;
  b.x = 4;
  b.y = 4;
  b.w = 8;
  b.h = 8;
  b.max_x = 400;
  b.max_y = 400;
  b.have_left = 1;
  b.have_above = 1;
  b.plane = 0;
  b.sb_height = 64;
  gpu::MrlParams p{};
  p.mode = 3;
  p.mrl_index = 1;
  std::uint8_t* d_pred = nullptr;
  CUDA_OK(cudaMalloc(&d_pred, 64 * 64));

  gpu::MrlParams dc = p;
  dc.mode = 0;
  expect(gpu::mrl_predict(b, dc, d_pred) == cudaErrorInvalidValue,
         "non-directional mode rejected");
  gpu::MrlParams bad_index = p;
  bad_index.mrl_index = 4;
  expect(gpu::mrl_predict(b, bad_index, d_pred) == cudaErrorInvalidValue,
         "mrl_index 4 rejected");
  gpu::MrlParams sec0 = p;
  sec0.mrl_index = 0;
  sec0.mrl_sec_index = 1;
  expect(gpu::mrl_predict(b, sec0, d_pred) == cudaErrorInvalidValue,
         "secondary without MRL rejected");
  gpu::MrlBlock tiny = b;
  tiny.w = 4;
  tiny.h = 4;
  gpu::MrlParams sec = p;
  sec.mrl_sec_index = 1;
  expect(gpu::mrl_predict(tiny, sec, d_pred) == cudaErrorInvalidValue,
         "secondary at 4x4 rejected");
  gpu::MrlBlock chroma = b;
  chroma.plane = 1;
  expect(gpu::mrl_predict(chroma, p, d_pred) == cudaErrorInvalidValue,
         "chroma rejected");
  cudaFree(d_pred);
}

}  // namespace

int main() {
  availability_cases();

  DevicePlane device;
  CUDA_OK(cudaMalloc(&device.data, kPlaneStride * kPlaneRows));
  std::vector<std::uint8_t> plane(kPlaneStride * kPlaneRows);
  gpu::MrlJob* jobs = nullptr;
  CUDA_OK(cudaMalloc(&jobs, sizeof(gpu::MrlJob) * 2));
  std::uint8_t* d_pred = nullptr;
  std::uint8_t* d_pred2 = nullptr;
  CUDA_OK(cudaMalloc(&d_pred, 64 * 64));
  CUDA_OK(cudaMalloc(&d_pred2, 64 * 64));

  const int dims[] = {4, 8, 16, 32, 64};
  const int positions[][2] = {{0, 0},   {4, 4},   {8, 4},   {4, 8},
                              {12, 20}, {32, 32}, {64, 4},  {4, 64},
                              {60, 60}, {400, 400}, {440, 440}};
  for (int trial = 0; trial < 2500; ++trial) {
    for (std::uint8_t& v : plane) {
      v = static_cast<std::uint8_t>(next_random() & 0xFFu);
    }
    CUDA_OK(cudaMemcpy(device.data, plane.data(), plane.size(),
                       cudaMemcpyHostToDevice));
    const int w = dims[random_range(0, 4)];
    const int h = dims[random_range(0, 4)];
    const auto& pos = positions[random_range(
        0, static_cast<int>(sizeof(positions) / sizeof(positions[0])) - 1)];
    const int mode = kModes[random_range(0, 7)];
    const int delta = kDeltas[random_range(0, 6)];
    const int mrl = random_range(0, 3);
    const bool sec = mrl > 0 && random_range(0, 1) != 0;
    run_case(plane, pos[0], pos[1], w, h, mode, delta, mrl, sec,
             random_range(0, 1) != 0, random_range(0, 1) != 0,
             random_range(0, w / 4 + 2), random_range(0, h / 4 + 2), device,
             jobs, d_pred, d_pred2);
  }

  cudaFree(d_pred);
  cudaFree(d_pred2);
  cudaFree(jobs);
  std::printf("mrl_test: %s (%d checks)\n",
              g_failures == 0 ? "PASS" : "FAIL", g_checks);
  return g_failures == 0 ? 0 : 1;
}
