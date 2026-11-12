// Host emulation parity test for the shared DDT device body.
//
// This TU compiles src/cuda/ddt/ddt_device.cuh with the CUDA keywords stubbed,
// so the exact integer code the kernels run is compared against the CPU
// reference in src/encoder/m2/ddt/.  It needs no GPU and no nvcc; the real GPU
// test is ddt_gpu_test.cu (built by build_gpu_test.sh).
//
// Coverage:
//   * all 320 Ddtx_Kernel8/16 entries against the §9.6 tables,
//   * get_transform_1d_type() over every flag/length combination,
//   * kernel selection, including the length-32 alias to Dct_Kernel32,
//   * 1D inverse/forward parity and the FDDT sample reversal,
//   * 2D block inverse (8/10-bit clips) and encoder forward for all 25 sizes
//     x 16 types, with the 100 legal DDT pairs counted separately,
//   * level_from_coeff / dequantize_level rounding.
//
// Build/run: ./build_and_test.sh

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static

#include "cuda/ddt/ddt_device.cuh"

#include "encoder/m2/ddt/ddt.h"
#include "encoder/m2/ddt/ddt_tables.h"
#include "encoder/tx_variants/tx_variants.h"

#include "cuda/ddt/ddt_test_pairs.h"

namespace {

using av2enc::m2::ddt::Context;
using av2enc::m2::ddt::Effective1d;
using av2enc::txv::Tx1dType;
using av2enc::txv::TxParams;
using av2enc::txv::TxSize;
using av2enc::txv::TxType;
namespace dev = av2enc::cuda::ddt::dev;
namespace txv = av2enc::txv;

int g_checks = 0;
int g_failures = 0;
std::uint64_t g_elements = 0;

void check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

// Elementwise comparison with one check per case and a running element count.
void expect_elements(const void* got, const void* expected, int count,
                     int element_size, const std::string& what) {
  ++g_checks;
  g_elements += static_cast<std::uint64_t>(count);
  if (std::memcmp(got, expected,
                  static_cast<std::size_t>(count) *
                      static_cast<std::size_t>(element_size)) != 0) {
    ++g_failures;
    if (g_failures <= 20) {
      std::printf("FAIL: %s (%d elements differ)\n", what.c_str(), count);
    }
  }
}

std::uint32_t g_rng = 0x9e3779b9U;
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

int effective_code(Effective1d type) { return static_cast<int>(type); }

int cpu_effective(Tx1dType signaled, int sz, const Context& ctx) {
  return effective_code(av2enc::m2::ddt::effective_1d(
      static_cast<Effective1d>(static_cast<int>(signaled)), sz, ctx));
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

void test_tables() {
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      check(dev::kDdtxKernel8[r][c] == av2enc::m2::ddt::kDdtxKernel8[r][c],
            "Ddtx_Kernel8[" + std::to_string(r) + "][" + std::to_string(c) +
                "] matches the §9.6 table");
    }
  }
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 16; ++c) {
      check(dev::kDdtxKernel16[r][c] == av2enc::m2::ddt::kDdtxKernel16[r][c],
            "Ddtx_Kernel16[" + std::to_string(r) + "][" + std::to_string(c) +
                "] matches the §9.6 table");
    }
  }
}

// ---------------------------------------------------------------------------
// get_transform_1d_type() and kernel selection
// ---------------------------------------------------------------------------

void test_effective_and_kernels() {
  const int signaled[4] = {0, 1, 2, 3};  // DCT, IDT, ADST, FDST
  const int sizes[4] = {4, 8, 16, 32};
  for (int s : signaled) {
    for (int sz : sizes) {
      for (int flags = 0; flags < 8; ++flags) {
        const bool enable = (flags & 1) != 0;
        const bool is_inter = (flags & 2) != 0;
        const bool intrabc = (flags & 4) != 0;
        Context ctx;
        ctx.enable_inter_ddt = enable;
        ctx.is_inter = is_inter;
        ctx.use_intrabc = intrabc;
        const int got =
            dev::effective_1d(s, sz, enable, is_inter, intrabc);
        const int expected = cpu_effective(static_cast<Tx1dType>(s), sz, ctx);
        check(got == expected,
              "effective_1d signaled=" + std::to_string(s) + " sz=" +
                  std::to_string(sz) + " flags=" + std::to_string(flags));
      }
    }
  }

  // Kernel tables: DDTX/FDDT at 8/16 are the DDT kernels; at 32 they alias
  // DCT32; at 4 they alias DCT4.  Compare contents, not addresses, because the
  // device tables are a separate (checked) copy.
  const int types[6] = {dev::kTypeDct,  dev::kTypeIdt,  dev::kTypeAdst,
                        dev::kTypeFdst, dev::kTypeDdtx, dev::kTypeFddt};
  for (int type : types) {
    for (int n : sizes) {
      const std::int16_t* device_kernel = dev::kernel_for(type, n);
      const std::int16_t* cpu_kernel = av2enc::m2::ddt::effective_kernel(
          static_cast<Effective1d>(type), n);
      check((device_kernel == nullptr) == (cpu_kernel == nullptr),
            "kernel null-ness type=" + std::to_string(type) + " n=" +
                std::to_string(n));
      if (device_kernel != nullptr && cpu_kernel != nullptr) {
        expect_elements(device_kernel, cpu_kernel, n * n,
                        static_cast<int>(sizeof(std::int16_t)),
                        "kernel contents type=" + std::to_string(type) +
                            " n=" + std::to_string(n));
      }
    }
  }

  // 32 alias-to-DCT: the DDTX/FDDT kernel at length 32 must be DCT32 exactly.
  const std::int16_t* dct32 = txv::tx_kernel(Tx1dType::kDct, 32);
  expect_elements(dev::kernel_for(dev::kTypeDdtx, 32), dct32, 32 * 32,
                  static_cast<int>(sizeof(std::int16_t)),
                  "DDTX at 32 aliases Dct_Kernel32");
  expect_elements(dev::kernel_for(dev::kTypeFddt, 32), dct32, 32 * 32,
                  static_cast<int>(sizeof(std::int16_t)),
                  "FDDT at 32 aliases Dct_Kernel32");
}

// ---------------------------------------------------------------------------
// 1D transforms
// ---------------------------------------------------------------------------

void test_inverse_1d() {
  const int sizes[4] = {4, 8, 16, 32};
  const int types[5] = {dev::kTypeDct, dev::kTypeAdst, dev::kTypeFdst,
                        dev::kTypeDdtx, dev::kTypeFddt};
  const int shifts[6] = {3, 6, 7, 10, 11, 13};
  for (int n : sizes) {
    for (int type : types) {
      for (int shift : shifts) {
        for (int col = 0; col < 2; ++col) {
          for (int depth : {8, 10}) {
            for (int trial = 0; trial < 3; ++trial) {
              std::int32_t src[32] = {};
              for (int i = 0; i < n; ++i) {
                src[i] = (trial == 0) ? random_range(-3000, 3000)
                                      : ((i & 1) ? 32767 : -32768);
              }
              std::int32_t got[32] = {};
              std::int32_t expected[32] = {};
              const bool got_ok = dev::inverse_1d(
                  src, n, shift, col != 0, type, depth, got);
              const bool expected_ok = av2enc::m2::ddt::inverse_1d(
                  src, n, shift, col != 0, static_cast<Effective1d>(type),
                  depth, expected);
              check(got_ok == expected_ok, "inverse_1d return value");
              expect_elements(got, expected, n,
                              static_cast<int>(sizeof(std::int32_t)),
                              "inverse_1d n=" + std::to_string(n) +
                                  " type=" + std::to_string(type) +
                                  " shift=" + std::to_string(shift) +
                                  " col=" + std::to_string(col) +
                                  " depth=" + std::to_string(depth));
            }
          }
        }
      }
    }
  }

  // IDT is not a matrix.
  std::int32_t src[8] = {};
  std::int32_t out[8] = {};
  check(!dev::inverse_1d(src, 8, 7, false, dev::kTypeIdt, 8, out),
        "inverse_1d rejects IDT");

  // FDDT is the sample-reversed DDTX for lengths 8 and 16.
  for (int n : {8, 16}) {
    for (int trial = 0; trial < 8; ++trial) {
      std::int32_t in[16] = {};
      std::int32_t ddtx[16] = {};
      std::int32_t fddt[16] = {};
      for (int i = 0; i < n; ++i) in[i] = random_range(-1500, 1500);
      dev::inverse_1d(in, n, 7, false, dev::kTypeDdtx, 8, ddtx);
      dev::inverse_1d(in, n, 7, false, dev::kTypeFddt, 8, fddt);
      for (int i = 0; i < n; ++i) {
        check(fddt[i] == ddtx[n - 1 - i],
              "FDDT" + std::to_string(n) +
                  " is the sample reversal of DDTX" + std::to_string(n));
      }
    }
  }
}

void test_forward_1d() {
  const int sizes[4] = {4, 8, 16, 32};
  const int types[5] = {dev::kTypeDct, dev::kTypeAdst, dev::kTypeFdst,
                        dev::kTypeDdtx, dev::kTypeFddt};
  const int shifts[6] = {6, 7, 10, 11, 12, 13};
  for (int n : sizes) {
    for (int type : types) {
      for (int shift : shifts) {
        for (int trial = 0; trial < 2; ++trial) {
          std::int32_t src[32] = {};
          for (int i = 0; i < n; ++i) src[i] = random_range(-1000, 1000);
          std::int32_t got[32] = {};
          std::int32_t expected[32] = {};
          const bool got_ok =
              dev::forward_1d(src, n, shift, type, got);
          const bool expected_ok = av2enc::m2::ddt::forward_1d(
              src, n, shift, static_cast<Effective1d>(type), expected);
          check(got_ok == expected_ok, "forward_1d return value");
          if (got_ok && expected_ok) {
            expect_elements(got, expected, n,
                            static_cast<int>(sizeof(std::int32_t)),
                            "forward_1d n=" + std::to_string(n) + " type=" +
                                std::to_string(type) + " shift=" +
                                std::to_string(shift));
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 2D block transforms
// ---------------------------------------------------------------------------

// Runs the shared device body exactly as the kernel maps it onto threads.
void emulate_inverse(const std::vector<std::int32_t>& dequant, const TxParams& p,
                     const Context& ctx, std::vector<std::int16_t>* coded) {
  const bool odd_rect = ((p.log2_w ^ p.log2_h) & 1) != 0;
  const dev::Plan plan = dev::make_plan(p, ctx.enable_inter_ddt, ctx.is_inter,
                                        ctx.use_intrabc);
  coded->assign(static_cast<std::size_t>(p.coded_w) *
                    static_cast<std::size_t>(p.coded_h), 0);
  std::vector<std::int32_t> inter(
      static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h), 0);
  for (int i = 0; i < p.h; ++i) {
    for (int j = 0; j < p.w; ++j) {
      inter[static_cast<std::size_t>(i) * static_cast<std::size_t>(p.w) +
            static_cast<std::size_t>(j)] = dev::inverse_row_elem(
          dequant.data() + static_cast<std::size_t>(i) *
                                 static_cast<std::size_t>(p.w), p, plan.row,
          ctx.bit_depth, odd_rect, j);
    }
  }
  for (int i = 0; i < p.h; ++i) {
    for (int j = 0; j < p.w; ++j) {
      const std::int32_t value = dev::inverse_col_elem(
          inter.data(), p.w, p, plan.col, ctx.bit_depth, i, j);
      dev::store_coded(coded->data(), p.coded_w, p.coded_h, p.w, p.h, i, j,
                       static_cast<std::int16_t>(value));
    }
  }
}

void emulate_forward(const std::vector<std::int16_t>& residual,
                     const TxParams& p, const Context& ctx,
                     std::vector<std::int32_t>* coeff) {
  const bool odd_rect = ((p.log2_w ^ p.log2_h) & 1) != 0;
  const dev::Plan plan = dev::make_plan(p, ctx.enable_inter_ddt, ctx.is_inter,
                                        ctx.use_intrabc);
  coeff->assign(
      static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h), 0);
  std::vector<long long> temp(
      static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h), 0);
  for (int i = 0; i < p.h; ++i) {
    for (int j = 0; j < p.w; ++j) {
      temp[static_cast<std::size_t>(i) * static_cast<std::size_t>(p.w) +
           static_cast<std::size_t>(j)] = dev::forward_row_elem(
          residual.data() + static_cast<std::size_t>(i) *
                                    static_cast<std::size_t>(p.w), p, plan.row, j);
    }
  }
  for (int i = 0; i < p.h; ++i) {
    for (int j = 0; j < p.w; ++j) {
      const long long value = dev::forward_col_elem(
          temp.data(), p.w, p, plan.col, i, j);
      (*coeff)[static_cast<std::size_t>(i) * static_cast<std::size_t>(p.w) +
               static_cast<std::size_t>(j)] =
          dev::forward_finish(value, odd_rect);
    }
  }
}

// §5.20.8.3 + Tx_Type_In_Set_Inter masks (shared with ddt_gpu_test.cu).
using PairRow = av2enc::cuda::ddt::test::SizeRow;
const PairRow* const kSizeRows = av2enc::cuda::ddt::test::kSizeRows;
constexpr int kSizeRowCount = 25;

void test_block_parity() {
  Context ctx;
  ctx.enable_inter_ddt = true;
  ctx.is_inter = true;

  const auto fill = [](int index, int w, int h, int shape) {
    const int x = index % w;
    const int y = index / w;
    switch (shape) {
      case 0:
        return 40;
      case 1:
        return x < w / 2 ? -80 : 70;
      case 2:
        return y < h / 2 ? -60 : 55;
      case 3:
        return ((x ^ y) & 1) ? 90 : -90;
      case 4:
        return (x * 7 + y * 3) % 121 - 60;
      default:
        return random_range(-120, 120);
    }
  };

  int active_pairs = 0;
  int inactive_pairs = 0;
  for (int row_index = 0; row_index < kSizeRowCount; ++row_index) {
    const PairRow& row = kSizeRows[row_index];
    for (int type = 0; type < 16; ++type) {
      const auto params_opt =
          txv::make_tx_params(row.size, static_cast<TxType>(type));
      check(params_opt.has_value(), "make_tx_params");
      if (!params_opt.has_value()) continue;
      const TxParams& p = *params_opt;
      const bool active = (row.ddt_mask & (1U << type)) != 0;
      if (active) {
        ++active_pairs;
      } else {
        ++inactive_pairs;
      }
      check(dev::plan_has_ddt(p, true, true, false) ==
                av2enc::m2::ddt::plan_has_ddt(p, ctx),
            "plan_has_ddt " + std::string(txv::tx_size_name(row.size)) +
                " type " + std::to_string(type));
      // plan_has_ddt() reports whether a DDT kernel actually changes the
      // block; that is the component mask (any ADST/FDST at adjusted length
      // 8/16), independent of whether Tx_Type_In_Set_Inter allows the type.
      check(dev::plan_has_ddt(p, true, true, false) ==
                ((row.component_mask & (1U << type)) != 0),
            "DDT activity matches the component mask " +
                std::string(txv::tx_size_name(row.size)) + " type " +
                std::to_string(type));

      const int count = p.w * p.h;
      for (int depth : {8, 10}) {
        Context local = ctx;
        local.bit_depth = depth;
        // Inverse: random dequant plus an extreme block for clipping parity.
        for (int shape = 0; shape < 3; ++shape) {
          std::vector<std::int32_t> dequant(static_cast<std::size_t>(count));
          for (int i = 0; i < count; ++i) {
            dequant[static_cast<std::size_t>(i)] =
                (shape == 2) ? ((i & 1) ? 32767 : -32768)
                             : random_range(-3000, 3000);
          }
          std::vector<std::int16_t> got;
          std::vector<std::int16_t> expected(
              static_cast<std::size_t>(p.coded_w) *
                  static_cast<std::size_t>(p.coded_h), 0);
          emulate_inverse(dequant, p, local, &got);
          check(av2enc::m2::ddt::inverse_block(dequant.data(), p, local,
                                               expected.data()),
                "CPU inverse_block");
          expect_elements(got.data(), expected.data(),
                          p.coded_w * p.coded_h,
                          static_cast<int>(sizeof(std::int16_t)),
                          "inverse_block " + std::string(txv::tx_size_name(
                                                 row.size)) +
                              " type " + std::to_string(type) + " depth " +
                              std::to_string(depth));
        }

        // Forward: two residual shapes plus random.
        for (int shape = 0; shape < 3; ++shape) {
          std::vector<std::int16_t> residual(static_cast<std::size_t>(count));
          for (int i = 0; i < count; ++i) {
            residual[static_cast<std::size_t>(i)] = static_cast<std::int16_t>(
                fill(i, p.w, p.h, shape == 2 ? 5 : shape));
          }
          std::vector<std::int32_t> got;
          std::vector<std::int32_t> expected(static_cast<std::size_t>(count), 0);
          emulate_forward(residual, p, local, &got);
          check(av2enc::m2::ddt::forward_block(residual.data(), p, local,
                                               expected.data()),
                "CPU forward_block");
          expect_elements(got.data(), expected.data(), count,
                          static_cast<int>(sizeof(std::int32_t)),
                          "forward_block " + std::string(txv::tx_size_name(
                                                 row.size)) +
                              " type " + std::to_string(type) + " depth " +
                              std::to_string(depth));
          for (int i = 0; i < count; ++i) {
            check(dev::level_from_coeff(got[static_cast<std::size_t>(i)],
                                        p.dq_denom) ==
                      av2enc::m2::ddt::level_from_coeff(
                          got[static_cast<std::size_t>(i)], p),
                  "level_from_coeff " +
                      std::string(txv::tx_size_name(row.size)) + " type " +
                      std::to_string(type));
          }
        }
      }
    }
  }
  check(active_pairs == 100,
        "100 legal (tx_size, tx_type) DDT pairs, got " +
            std::to_string(active_pairs));
  check(inactive_pairs == 300, "300 non-DDT pairs are also covered");
}

// ---------------------------------------------------------------------------
// Dequant rounding
// ---------------------------------------------------------------------------

void test_dequantize_level() {
  const auto params_opt = txv::make_tx_params(TxSize::k4x4, TxType::kDctDct);
  check(params_opt.has_value(), "dequantize_level params");
  if (!params_opt.has_value()) return;
  const int values[] = {0,      1,       -1,     7,      -7,      8,
                        -8,     9,       -9,     4095,   -4095,   100000,
                        -100000, 32767, -32768};
  for (int denom : {1, 2, 4}) {
    TxParams p = *params_opt;
    p.dq_denom = static_cast<std::uint8_t>(denom);
    for (int level_value : values) {
      std::int32_t levels[16] = {};
      std::int32_t expected[16] = {};
      for (int i = 0; i < 16; ++i) levels[i] = level_value + i;
      txv::dequantize(levels, p, expected);
      for (int i = 0; i < 16; ++i) {
        check(dev::dequantize_level(levels[i], denom) == expected[i],
              "dequantize_level level=" + std::to_string(levels[i]) +
                  " denom=" + std::to_string(denom));
      }
    }
  }
}

}  // namespace

int main() {
  test_tables();
  test_effective_and_kernels();
  test_inverse_1d();
  test_forward_1d();
  test_block_parity();
  test_dequantize_level();
  std::printf(
      "ddt device emulation: %d checks, %llu element comparisons, "
      "%d failures\n",
      g_checks, static_cast<unsigned long long>(g_elements), g_failures);
  return g_failures == 0 ? 0 : 1;
}
