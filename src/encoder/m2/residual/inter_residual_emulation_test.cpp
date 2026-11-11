// Host emulation parity test for the device-side inter-residual helpers.
//
// Compiles inter_residual_device.cuh with the CUDA keywords stubbed and one
// emulated thread, then compares every field against the CPU reference in
// inter_coeff_syntax.cpp / inter_residual.cpp.  Runs without nvcc; the real GPU
// kernels live in src/cuda/m2/ (the transforms are src/cuda/tx_variants/, which
// has its own CPU/GPU parity gate).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define __device__
#define __host__
#define __forceinline__ inline

#include "encoder/m2/residual/inter_residual_device.cuh"

#include "encoder/m2/m2_syntax.h"
#include "encoder/m2/residual/inter_coeff_syntax.h"
#include "encoder/m2/residual/inter_residual.h"
#include "encoder/tx_variants/tx_variants.h"

namespace {

using namespace av2enc::m2::residual;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
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

struct Size {
  int w;
  int h;
};

void test_residual_and_recon() {
  const Size sizes[] = {{4, 4}, {8, 8}, {16, 16}, {32, 32}, {64, 16}, {8, 32}};
  for (const Size& size : sizes) {
    const int count = size.w * size.h;
    std::vector<std::uint8_t> source(static_cast<std::size_t>(count));
    std::vector<std::uint8_t> pred(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
      source[static_cast<std::size_t>(i)] =
          static_cast<std::uint8_t>(random_range(0, 255));
      pred[static_cast<std::size_t>(i)] =
          static_cast<std::uint8_t>(random_range(0, 255));
    }
    std::vector<std::int16_t> device_residual(
        static_cast<std::size_t>(count));
    dev::ir_residual_coded(source.data(), pred.data(), size.w, size.h,
                           device_residual.data());
    bool ok = true;
    for (int i = 0; i < count; ++i) {
      const int expected = static_cast<int>(source[static_cast<std::size_t>(i)]) -
                           static_cast<int>(pred[static_cast<std::size_t>(i)]);
      if (device_residual[static_cast<std::size_t>(i)] != expected) ok = false;
    }
    check(ok, "device residual gather matches the CPU difference");

    std::vector<std::uint8_t> device_recon(static_cast<std::size_t>(count));
    const double device_sse = dev::ir_reconstruct(
        source.data(), pred.data(), device_residual.data(), size.w, size.h,
        device_recon.data());
    double expected_sse = 0.0;
    bool recon_ok = true;
    for (int i = 0; i < count; ++i) {
      const int value =
          std::clamp(static_cast<int>(pred[static_cast<std::size_t>(i)]) +
                         static_cast<int>(device_residual[static_cast<std::size_t>(i)]),
                     0, 255);
      if (device_recon[static_cast<std::size_t>(i)] != value) recon_ok = false;
      const double d = static_cast<double>(source[static_cast<std::size_t>(i)]) -
                       static_cast<double>(value);
      expected_sse += d * d;
    }
    check(recon_ok, "device reconstruction equals clip(pred + residual)");
    check(device_sse == expected_sse, "device SSE is exact");
  }
}

void test_eob_and_plan() {
  const Size sizes[] = {{4, 4},  {8, 8},   {16, 16}, {32, 32},
                        {4, 8},  {8, 4},   {16, 8},  {8, 16},
                        {16, 32}, {32, 16}, {64, 16}, {4, 16}};
  for (const Size& size : sizes) {
    const auto params =
        av2enc::txv::make_tx_params(size.w, size.h, av2enc::txv::TxType::kDctDct);
    if (!params.has_value()) {
      check(false, "unsupported test size");
      continue;
    }
    const int area = params->w * params->h;
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    av2enc::m2::syntax::build_scan(params->w, params->h, scan);
    std::vector<std::int32_t> scan_dev(static_cast<std::size_t>(area));
    for (int i = 0; i < area; ++i) {
      scan_dev[static_cast<std::size_t>(i)] = scan[static_cast<std::size_t>(i)];
    }

    for (int trial = 0; trial < 8; ++trial) {
      std::vector<std::int32_t> levels_raster(static_cast<std::size_t>(area), 0);
      // Sparse, sign-mixed levels with a few large magnitudes.
      const int nonzero = random_range(1, std::max(1, area / 3));
      for (int i = 0; i < nonzero; ++i) {
        const int pos = random_range(0, area - 1);
        const int magnitude = random_range(1, trial % 2 == 0 ? 3 : 24);
        levels_raster[static_cast<std::size_t>(pos)] =
            next_random() & 1U ? magnitude : -magnitude;
      }
      // Keep a non-zero DC sometimes so the DC sign path is exercised.
      if (trial % 3 == 0) levels_raster[0] = -7;

      const int cpu_eob = dev::ir_eob_scan(levels_raster.data(), scan.data(), area);
      int expected_eob = 0;
      for (int c = area - 1; c >= 0; --c) {
        if (levels_raster[static_cast<std::size_t>(
                scan[static_cast<std::size_t>(c)])] != 0) {
          expected_eob = c + 1;
          break;
        }
      }
      check(cpu_eob == expected_eob, "device EOB matches the scan-order EOB");
      if (cpu_eob == 0) continue;

      std::vector<std::int32_t> levels_scan(static_cast<std::size_t>(cpu_eob));
      for (int c = 0; c < cpu_eob; ++c) {
        levels_scan[static_cast<std::size_t>(c)] = levels_raster[
            static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])];
      }
      for (int plane = 0; plane < 3; ++plane) {
        const int dc_sign_ctx = random_range(0, 2);
        std::vector<CoeffPlanEntry> cpu_plan;
        std::string error;
        if (!build_coeff_plan(plane, cpu_eob, params->w, params->h,
                              levels_scan, dc_sign_ctx, &cpu_plan, &error)) {
          check(false, "cpu build_coeff_plan: " + error);
          continue;
        }
        std::vector<CoeffPlanEntry> device_plan(
            static_cast<std::size_t>(cpu_eob));
        std::vector<int> level_pre(static_cast<std::size_t>(area));
        dev::ir_build_coeff_plan(plane, cpu_eob, params->w, params->h,
                                 levels_scan.data(), scan.data(), dc_sign_ctx,
                                 device_plan.data(), level_pre.data());
        bool plan_ok = true;
        for (int c = 0; c < cpu_eob; ++c) {
          const CoeffPlanEntry& a = cpu_plan[static_cast<std::size_t>(c)];
          const CoeffPlanEntry& b = device_plan[static_cast<std::size_t>(c)];
          if (a.base_level != b.base_level || a.br_level != b.br_level ||
              a.sign != b.sign || a.is_eob != b.is_eob || a.is_lf != b.is_lf ||
              a.codes_br != b.codes_br || a.codes_dc_sign != b.codes_dc_sign ||
              a.codes_escape != b.codes_escape || a.base_ctx != b.base_ctx ||
              a.br_ctx != b.br_ctx || a.eob_ctx != b.eob_ctx ||
              a.dc_sign_ctx != b.dc_sign_ctx || a.pos != b.pos ||
              a.abs_level != b.abs_level || a.pre_level != b.pre_level ||
              a.escape_delta != b.escape_delta) {
            plan_ok = false;
            break;
          }
        }
        check(plan_ok, "device coefficient plan matches the CPU plan (plane " +
                           std::to_string(plane) + ", " +
                           std::to_string(size.w) + "x" +
                           std::to_string(size.h) + ")");
      }
    }
  }
}

void test_dc_sign_and_skip() {
  std::vector<std::uint8_t> above_dc;
  std::vector<std::uint8_t> left_dc;
  for (int trial = 0; trial < 64; ++trial) {
    const int max_x4 = random_range(1, 16);
    const int max_y4 = random_range(1, 16);
    above_dc.assign(static_cast<std::size_t>(max_x4 + 4), 0);
    left_dc.assign(static_cast<std::size_t>(max_y4 + 4), 0);
    for (std::size_t i = 0; i < above_dc.size(); ++i) {
      above_dc[i] = static_cast<std::uint8_t>(random_range(0, 2));
    }
    for (std::size_t i = 0; i < left_dc.size(); ++i) {
      left_dc[i] = static_cast<std::uint8_t>(random_range(0, 2));
    }
    const int x4 = random_range(0, max_x4 - 1);
    const int y4 = random_range(0, max_y4 - 1);
    const int w4 = random_range(1, std::min(4, max_x4 - x4));
    const int h4 = random_range(1, std::min(4, max_y4 - y4));
    CoeffContextView view;
    view.mi_cols = max_x4;
    view.mi_rows = max_y4;
    view.above_dc[0] = above_dc.data();
    view.left_dc[0] = left_dc.data();
    const int cpu = dc_sign_context(view, 0, x4, y4, w4, h4);
    const int device =
        dev::ir_dc_sign_context(above_dc.data(), left_dc.data(), x4, y4, w4, h4,
                                max_x4, max_y4);
    check(cpu == device, "device DC-sign context matches the CPU");
  }
  check(dev::ir_prefer_residual(10.0, 11.0) == 1, "residual preferred below skip cost");
  check(dev::ir_prefer_residual(11.0, 11.0) == 0, "ties skip");
  check(dev::ir_prefer_residual(-1.0, 11.0) == 0, "negative cost skips");
}

}  // namespace

int main() {
  test_residual_and_recon();
  test_eob_and_plan();
  test_dc_sign_and_skip();
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
