// Host-emulation parity test for the CUDA IST core.
//
// The real GPU test (tx_secondary_test.cu) needs nvcc.  This test compiles the
// exact same tx_secondary_device.cuh core with the host compiler and compares
// it bit-for-bit against the CPU reference.  It is the macOS fallback and an
// additional guard against device-only code drifting from the CPU path.
//
// Build/run:
//   ./build_and_test.sh
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "cuda/tx_secondary/tx_secondary_device.cuh"
#include "encoder/tx_secondary/tx_secondary.h"
#include "encoder/tx_secondary/tx_secondary_tables.h"
#include "encoder/tx_variants/tx_variants.h"

namespace {

using namespace av2enc;
using namespace av2enc::txsec;
using av2enc::txv::TxParams;
using av2enc::txv::TxType;

int g_failures = 0;
long long g_checks = 0;

void check(bool cond, const char* what, int line) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("FAIL [%d] %s\n", line, what);
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

std::mt19937 g_rng(0x1c0de123u);

int rand_range(int lo, int hi) {
  std::uniform_int_distribution<int> dist(lo, hi);
  return dist(g_rng);
}

av2enc::txsec::device::Tables host_tables() {
  av2enc::txsec::device::Tables t;
  t.ist4 = &kIst4x4Kernel[0][0][0][0];
  t.ist8 = &kIst8x8Kernel[0][0][0][0];
  t.scan_map = &kStxScanMap[0][0][0];
  t.tx_width = &kTxWidth[0];
  t.tx_height = &kTxHeight[0];
  t.tx_width_log2 = &kTxWidthLog2[0];
  t.scan_order4 = &kStxScanOrder4x4[0];
  t.scan_order8 = &kStxScanOrder8x8[0];
  return t;
}

std::vector<TxParams> all_sizes() {
  static const int kSizes[25][2] = {
      {4, 4},   {8, 8},   {16, 16}, {32, 32}, {64, 64}, {4, 8},
      {8, 4},   {8, 16},  {16, 8},  {16, 32}, {32, 16}, {32, 64},
      {64, 32}, {4, 16},  {16, 4},  {8, 32},  {32, 8},  {16, 64},
      {64, 16}, {4, 32},  {32, 4},  {8, 64},  {64, 8},  {4, 64},
      {64, 4},
  };
  std::vector<TxParams> out;
  for (const auto& s : kSizes) {
    auto p = txv::make_tx_params(s[0], s[1], TxType::kDctDct);
    CHECK(p.has_value());
    if (p.has_value()) out.push_back(*p);
  }
  return out;
}

void test_size_info() {
  const auto tables = host_tables();
  for (const TxParams& base : all_sizes()) {
    for (TxType type : {TxType::kDctDct, TxType::kAdstAdst}) {
      auto pv = txv::make_tx_params(base.coded_w, base.coded_h, type);
      CHECK(pv.has_value());
      if (!pv) continue;
      SecondarySizeInfo cpu;
      CHECK(secondary_size_info(*pv, type, &cpu));
      av2enc::txsec::device::SizeInfo gpu;
      CHECK(av2enc::txsec::device::size_info(
          tables, static_cast<int>(pv->tx_size), static_cast<int>(type), &gpu));
      CHECK(cpu.w == gpu.w);
      CHECK(cpu.h == gpu.h);
      CHECK(cpu.bwl == gpu.bwl);
      CHECK(cpu.large == gpu.large);
      CHECK(cpu.n == gpu.n);
      CHECK(cpu.n2 == gpu.n2);
      CHECK(cpu.scan_w == gpu.scan_w);
      CHECK(cpu.kernel_count == gpu.kernel_count);
    }
  }
}

void test_inverse_and_forward() {
  const auto tables = host_tables();
  long long cases = 0;
  for (const TxParams& base : all_sizes()) {
    for (TxType type : {TxType::kDctDct, TxType::kAdstAdst}) {
      auto pv = txv::make_tx_params(base.coded_w, base.coded_h, type);
      if (!pv) continue;
      const TxParams& p = *pv;
      SecondarySizeInfo info;
      CHECK(secondary_size_info(p, type, &info));
      const std::size_t elems =
          static_cast<std::size_t>(info.w) * static_cast<std::size_t>(info.h);
      std::uint16_t scan[1024];
      // Small independent scan for pattern placement.
      {
        int x = 0;
        int y = 0;
        for (int c = 0; c < info.w * info.h; ++c) {
          scan[c] = static_cast<std::uint16_t>(y * info.w + x);
          ++x;
          --y;
          if (y < 0 || x >= info.w) {
            ++x;
            const int s = std::min(x, info.h - 1 - y);
            x -= s;
            y += s;
          }
        }
      }
      std::vector<std::int32_t> base_in(elems, 0);
      std::vector<std::int32_t> cpu(elems);
      std::vector<std::int32_t> gpu(elems);
      for (int kernel = 0; kernel < info.kernel_count; ++kernel) {
        for (int sec = 1; sec < 4; ++sec) {
          for (int transpose = 0; transpose <= 1; ++transpose) {
            for (int pattern = 0; pattern < 5; ++pattern) {
              std::fill(base_in.begin(), base_in.end(), 0);
              for (int j = 0; j < info.n; ++j) {
                int value = 0;
                switch (pattern) {
                  case 0: value = 0; break;
                  case 1: value = (j & 1) ? 32767 : -32768; break;
                  case 2: value = (j & 1) ? 1 : -1; break;
                  case 3: value = rand_range(-40000, 40000); break;
                  default: value = (j == 0) ? 32767 : 0; break;
                }
                base_in[scan[j]] = value;
              }
              cpu = base_in;
              gpu = base_in;
              CHECK(inverse_secondary_dequant_kernel(
                  p, type, kernel, sec, transpose != 0, 8, cpu.data(),
                  cpu.size()));
              CHECK(av2enc::txsec::device::inverse(
                  tables, av2enc::txsec::device::SizeInfo{
                              info.w, info.h, info.bwl, info.n, info.n2,
                              info.scan_w, info.kernel_count, info.large},
                  kernel, sec, transpose, 8, gpu.data()));
              for (std::size_t i = 0; i < elems; ++i) CHECK(cpu[i] == gpu[i]);
              ++cases;

              // Forward projection parity.
              for (std::size_t i = 0; i < elems; ++i) {
                base_in[i] = rand_range(-2000, 2000);
              }
              cpu = base_in;
              gpu = base_in;
              CHECK(forward_secondary_coeffs_kernel(
                  p, type, kernel, sec, transpose != 0, 8, cpu.data(),
                  cpu.size()));
              CHECK(av2enc::txsec::device::forward(
                  tables, av2enc::txsec::device::SizeInfo{
                              info.w, info.h, info.bwl, info.n, info.n2,
                              info.scan_w, info.kernel_count, info.large},
                  kernel, sec, transpose, 8, gpu.data()));
              for (std::size_t i = 0; i < elems; ++i) CHECK(cpu[i] == gpu[i]);
              ++cases;
            }
          }
        }
      }
    }
  }
  std::printf("  emulation cases: %lld\n", cases);
}

}  // namespace

int main() {
  std::printf("av2enc AV2 secondary transform CUDA-core emulation test\n");
  test_size_info();
  test_inverse_and_forward();
  std::printf("%lld checks, %d failures\n", g_checks, g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
