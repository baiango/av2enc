// CPU-vs-GPU parity test for the AV2 DIP predictor.
//
// Every case builds a random reconstructed luma plane, runs the CPU reference
// (src/encoder/dip/) and the CUDA kernel on the same descriptor, and requires
// byte-for-byte equality.  Includes all DIP-capable block sizes, all 6
// dip_modes, both dip_transpose values, frame/SB availability combinations,
// partial right/bottom blocks, single-block and batch entry points.
//
// Build/run: ./build_and_test.sh (needs nvcc + a CUDA device).

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

#include "cuda/codec/intra/dip/dip.h"
#include "codec/intra/dip/dip.h"

namespace {

using av2enc::dip::Block;
using av2enc::dip::Params;
using av2enc::dip::PlaneView;
using av2enc::dip::Status;
namespace gpu = av2enc::cuda::dip;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 25) std::printf("FAIL: %s\n", what);
  }
}

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

constexpr int kStride = 160;
constexpr int kRows = 160;
constexpr int kValidDims[][2] = {
    {8, 16},  {8, 32},  {8, 64},  {16, 8},  {16, 16},
    {16, 32}, {16, 64}, {32, 8},  {32, 16}, {32, 32},
    {32, 64}, {64, 8},  {64, 16}, {64, 32}, {64, 64},
};

struct Case {
  av2enc::dip::Block cpu_block;
  gpu::DipBlock gpu_block;
  gpu::DipParams gpu_params;
  int w = 0;
  int h = 0;
  int x = 0;
  int y = 0;
};

Case make_case(const std::vector<std::uint8_t>& plane, std::uint8_t* d_plane,
               int x, int y, int w, int h, int mode, int transpose,
               bool have_left, bool have_above, int num4_ar, int num4_bl,
               int max_x, int max_y) {
  Case c;
  c.w = w;
  c.h = h;
  c.x = x;
  c.y = y;
  c.cpu_block.recon = PlaneView::from_u8(plane.data(), kStride);
  c.cpu_block.x = x;
  c.cpu_block.y = y;
  c.cpu_block.w = w;
  c.cpu_block.h = h;
  c.cpu_block.max_x = max_x;
  c.cpu_block.max_y = max_y;
  c.cpu_block.have_left = have_left;
  c.cpu_block.have_above = have_above;
  c.cpu_block.num4_above_right = num4_ar;
  c.cpu_block.num4_below_left = num4_bl;
  c.gpu_block.recon = d_plane;
  c.gpu_block.recon_stride = kStride;
  c.gpu_block.x = x;
  c.gpu_block.y = y;
  c.gpu_block.w = w;
  c.gpu_block.h = h;
  c.gpu_block.max_x = max_x;
  c.gpu_block.max_y = max_y;
  c.gpu_block.have_left = have_left ? 1 : 0;
  c.gpu_block.have_above = have_above ? 1 : 0;
  c.gpu_block.num4_above_right = num4_ar;
  c.gpu_block.num4_below_left = num4_bl;
  c.gpu_params.mode = mode;
  c.gpu_params.transpose = transpose;
  return c;
}

}  // namespace

int main() {
  std::vector<std::uint8_t> plane(static_cast<std::size_t>(kStride) * kRows);
  for (std::uint8_t& v : plane) {
    v = static_cast<std::uint8_t>(next_random() & 0xFFu);
  }

  std::uint8_t* d_plane = nullptr;
  std::uint8_t* d_pred = nullptr;
  gpu::DipJob* d_jobs = nullptr;
  constexpr int kMaxJobs = 4096;
  check(cudaMalloc(&d_plane, plane.size()) == cudaSuccess, "cudaMalloc plane");
  check(cudaMalloc(&d_pred, 64 * 64) == cudaSuccess, "cudaMalloc pred");
  check(cudaMalloc(&d_jobs, sizeof(gpu::DipJob) * kMaxJobs) == cudaSuccess,
        "cudaMalloc jobs");
  check(cudaMemcpy(d_plane, plane.data(), plane.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess,
        "upload plane");

  const int positions[][2] = {{0, 0},   {8, 8},   {16, 24},
                              {32, 16}, {64, 48}, {80, 80}};
  int cases = 0;
  int batch_count = 0;
  std::vector<Case> batch_cases;
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
            const Case c = make_case(plane, d_plane, x, y, w, h, mode,
                                     transpose, have_left, have_above, num4_ar,
                                     num4_bl, max_x, max_y);
            Params params;
            params.mode = mode;
            params.transpose = transpose != 0;
            std::vector<std::uint8_t> want(static_cast<std::size_t>(w) *
                                           static_cast<std::size_t>(h));
            check(av2enc::dip::predict_block8(c.cpu_block, 8, params,
                                              want.data(),
                                              static_cast<std::size_t>(w)) ==
                      Status::kOk,
                  "CPU DIP");
            check(gpu::dip_predict(c.gpu_block, c.gpu_params, d_pred) ==
                      cudaSuccess,
                  "dip_predict");
            std::vector<std::uint8_t> got(want.size());
            check(cudaMemcpy(got.data(), d_pred, got.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess,
                  "download prediction");
            check(got == want, "single-block CPU/GPU parity");
            ++cases;

            if (batch_cases.size() < static_cast<std::size_t>(kMaxJobs)) {
              batch_cases.push_back(c);
              ++batch_count;
            }
          }
        }
      }
    }
  }

  // Batch entry point: one contiguous device output per job.
  if (!batch_cases.empty()) {
    std::vector<gpu::DipJob> jobs(batch_cases.size());
    std::vector<std::vector<std::uint8_t>> wants(batch_cases.size());
    std::uint8_t* d_batch_out = nullptr;
    const std::size_t batch_bytes = batch_cases.size() * 64 * 64;
    check(cudaMalloc(&d_batch_out, batch_bytes) == cudaSuccess,
          "cudaMalloc batch output");
    for (std::size_t i = 0; i < batch_cases.size(); ++i) {
      jobs[i] = gpu::DipJob{};
      jobs[i].block = batch_cases[i].gpu_block;
      jobs[i].params = batch_cases[i].gpu_params;
      jobs[i].pred = d_batch_out + i * 64 * 64;
      Params params;
      params.mode = batch_cases[i].gpu_params.mode;
      params.transpose = batch_cases[i].gpu_params.transpose != 0;
      wants[i].resize(static_cast<std::size_t>(batch_cases[i].w) *
                      static_cast<std::size_t>(batch_cases[i].h));
      check(av2enc::dip::predict_block8(
                batch_cases[i].cpu_block, 8, params, wants[i].data(),
                static_cast<std::size_t>(batch_cases[i].w)) == Status::kOk,
            "CPU batch DIP");
    }
    check(cudaMemcpy(d_jobs, jobs.data(), sizeof(gpu::DipJob) * jobs.size(),
                     cudaMemcpyHostToDevice) == cudaSuccess,
          "upload jobs");
    check(gpu::dip_predict_batch(d_jobs, static_cast<int>(jobs.size())) ==
              cudaSuccess,
          "dip_predict_batch");
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      const std::size_t size = wants[i].size();
      std::vector<std::uint8_t> got(size);
      check(cudaMemcpy(got.data(), d_batch_out + i * 64 * 64, size,
                       cudaMemcpyDeviceToHost) == cudaSuccess,
            "download batch output");
      check(got == wants[i], "batch CPU/GPU parity");
    }
    cudaFree(d_batch_out);
  }

  cudaFree(d_jobs);
  cudaFree(d_pred);
  cudaFree(d_plane);
  std::printf("dip_test: %d single cases, %d batch cases, %d checks, "
              "%d failures\n",
              cases, batch_count, g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
