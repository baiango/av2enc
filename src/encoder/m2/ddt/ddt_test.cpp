// CPU reference tests for the M2 inter-DDT component.
//
// Covers:
//   * pinned Ddtx_Kernel8/16 values and kernel properties,
//   * §7.15.4.1 get_transform_1d_type() over every flag/length combination,
//   * §7.15.2.1 1D DDTX/FDDT goldens, clipping and integer determinism,
//   * bit-exact equality with txv::inverse_block() for every non-DDT
//     (tx_size, tx_type) pair,
//   * forward+inverse round-trips over every legal DDT (tx_size, tx_type),
//     using real-image residuals and pinned real coefficient blocks,
//   * the constraint matrix (sizes/types/planes/tools) and the sequence flag
//     writer/reader with pinned goldens.
//
// Build/run with src/encoder/m2/ddt/build_and_test.sh.
#include "encoder/m2/ddt/ddt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "encoder/m2/ddt/ddt_constraints.h"
#include "encoder/m2/ddt/ddt_syntax.h"
#include "encoder/m2/ddt/ddt_tables.h"
#include "encoder/m2/residual/inter_tx_type.h"
#include "encoder/tx_variants/tx_variants.h"

namespace {

using namespace av2enc::m2::ddt;
namespace residual = av2enc::m2::residual;
namespace txv = av2enc::txv;
using av2enc::BitWriter;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

std::uint32_t g_rng = 0x12345678U;
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

// ---------------------------------------------------------------------------
// Kernel tables
// ---------------------------------------------------------------------------

void test_kernels() {
  const auto& k8 = kDdtxKernel8;
  const auto& k16 = kDdtxKernel16;
  check(k8[0][0] == 4 && k8[0][1] == 7 && k8[0][7] == 80,
        "Ddtx_Kernel8 first row");
  check(k8[7][0] == 56 && k8[7][7] == -6, "Ddtx_Kernel8 last row");
  check(k16[0][0] == 12 && k16[0][15] == 50, "Ddtx_Kernel16 first row");
  check(k16[15][0] == 44 && k16[15][2] == 124 && k16[15][3] == -126 &&
            k16[15][15] == 2,
        "Ddtx_Kernel16 last row");
  check(k16[9][2] == -99 && k16[9][9] == -60, "Ddtx_Kernel16 row 9");

  std::int64_t abs_sum = 0;
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) abs_sum += std::abs(k8[r][c]);
  }
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 16; ++c) abs_sum += std::abs(k16[r][c]);
  }
  check(abs_sum == 17999, "pinned kernel abs-sum");

  // Row norms: DDT kernels are scaled like DCT/ADST, 64*sqrt(n).
  for (int r = 0; r < 8; ++r) {
    std::int64_t norm = 0;
    for (int c = 0; c < 8; ++c) norm += k8[r][c] * k8[r][c];
    check(norm >= 32000 && norm <= 33000, "Ddtx_Kernel8 row norm");
  }
  for (int r = 0; r < 16; ++r) {
    std::int64_t norm = 0;
    for (int c = 0; c < 16; ++c) norm += k16[r][c] * k16[r][c];
    check(norm >= 64000 && norm <= 66000, "Ddtx_Kernel16 row norm");
  }
  check(ddt_kernel(4) == nullptr && ddt_kernel(32) == nullptr,
        "no DDT kernel for 4/32");
  check(ddt_kernel(8) == &k8[0][0] && ddt_kernel(16) == &k16[0][0],
        "ddt_kernel() addresses");
}

// ---------------------------------------------------------------------------
// get_transform_1d_type()
// ---------------------------------------------------------------------------

void test_effective_1d() {
  struct Case {
    Effective1d signaled;
    int sz;
    bool enable;
    bool is_inter;
    bool intrabc;
    Effective1d expected;
  };
  const Effective1d adst = Effective1d::kAdst;
  const Effective1d fdst = Effective1d::kFdst;
  const Effective1d ddtx = Effective1d::kDdtx;
  const Effective1d fddt = Effective1d::kFddt;
  const Case cases[] = {
      {adst, 8, true, true, false, ddtx},
      {fdst, 8, true, true, false, fddt},
      {adst, 16, true, true, false, ddtx},
      {fdst, 16, true, true, false, fddt},
      {adst, 8, false, true, false, adst},
      {adst, 8, true, false, false, adst},
      {adst, 8, true, true, true, adst},
      {adst, 4, true, true, false, adst},
      {fdst, 4, true, true, false, fdst},
      {Effective1d::kDct, 8, true, true, false, Effective1d::kDct},
      {Effective1d::kIdt, 16, true, true, false, Effective1d::kIdt},
      // 32 is not excluded by the predicate: DDTX/FDDT are derived, then
      // §7.15.2.1's sz == 32 branch aliases them to Dct_Kernel32.
      {adst, 32, true, true, false, ddtx},
      {fdst, 32, true, true, false, fddt},
  };
  for (const Case& c : cases) {
    Context ctx;
    ctx.enable_inter_ddt = c.enable;
    ctx.is_inter = c.is_inter;
    ctx.use_intrabc = c.intrabc;
    const Effective1d got = effective_1d(c.signaled, c.sz, ctx);
    check(got == c.expected,
          std::string("effective_1d ") + effective_1d_name(c.signaled) +
              " sz=" + std::to_string(c.sz) + " got " +
              effective_1d_name(got));
  }

  // At length 32 the kernel must alias DCT; at 8/16 it must be the DDT table.
  check(effective_kernel(ddtx, 32) == txv::tx_kernel(txv::Tx1dType::kDct, 32),
        "DDTX at 32 aliases DCT32");
  check(effective_kernel(fddt, 32) == txv::tx_kernel(txv::Tx1dType::kDct, 32),
        "FDDT at 32 aliases DCT32");
  check(effective_kernel(ddtx, 8) == ddt_kernel(8),
        "DDTX at 8 is Ddtx_Kernel8");
  check(effective_kernel(fddt, 16) == ddt_kernel(16),
        "FDDT at 16 is Ddtx_Kernel16 (mirrored by the transform)");
}

// ---------------------------------------------------------------------------
// 1D transforms
// ---------------------------------------------------------------------------

void test_inverse_1d_goldens() {
  const std::int32_t src8[8] = {37, -12, 205, -100, 8, 0, -41, 77};
  std::int32_t out8[8] = {};
  check(inverse_1d(src8, 8, 7, false, Effective1d::kDdtx, 8, out8),
        "inverse_1d DDTX8 succeeds");
  const std::int32_t golden8[8] = {6, -24, 132, 95, 129, -184, -18, 207};
  check(std::equal(out8, out8 + 8, golden8), "DDTX8 pinned golden");

  std::int32_t out_f8[8] = {};
  check(inverse_1d(src8, 8, 7, false, Effective1d::kFddt, 8, out_f8),
        "inverse_1d FDDT8 succeeds");
  for (int i = 0; i < 8; ++i) {
    check(out_f8[i] == out8[7 - i], "FDDT8 is the reverse of DDTX8");
  }

  const std::int32_t src16[16] = {5,  -60, 300, 41,   -7, 12, 90,   -210,
                                  77, 0,   13,  -14,  66, -3, -128, 42};
  std::int32_t out16[16] = {};
  check(inverse_1d(src16, 16, 6, false, Effective1d::kDdtx, 8, out16),
        "inverse_1d DDTX16 succeeds");
  const std::int32_t golden16[16] = {117, 214, 214, 474, 626,  59, 206, -281,
                                     389, -618, -717, -238, 358, 112, 34, 844};
  check(std::equal(out16, out16 + 16, golden16), "DDTX16 pinned golden");

  // Negative values must use §4.8 Round2's floor semantics.
  const std::int32_t neg[8] = {-37, 12, -205, 100, -8, 0, 41, -77};
  std::int32_t neg_out[8] = {};
  check(inverse_1d(neg, 8, 7, false, Effective1d::kDdtx, 8, neg_out),
        "inverse_1d negative succeeds");
  check(neg_out[0] == -6 && neg_out[1] == 24 && neg_out[7] == -207,
        "negative input golden");

  // Clipping: row clip is +-(1 << (BitDepth + 7)), column clip is
  // +-(1 << BitDepth) - 1.
  std::int32_t huge[8] = {};
  huge[0] = 30000;
  std::int32_t row_clip[8] = {};
  check(inverse_1d(huge, 8, 0, false, Effective1d::kDdtx, 8, row_clip),
        "inverse_1d row clip succeeds");
  check(row_clip[0] <= 32767 && row_clip[1] <= 32767, "row clip 8-bit");
  std::int32_t col_clip[8] = {};
  check(inverse_1d(huge, 8, 0, true, Effective1d::kDdtx, 8, col_clip),
        "inverse_1d col clip succeeds");
  check(col_clip[0] <= 255 && col_clip[1] <= 255, "column clip 8-bit");
  std::int32_t col_clip10[8] = {};
  check(inverse_1d(huge, 8, 0, true, Effective1d::kDdtx, 10, col_clip10),
        "inverse_1d col clip 10-bit succeeds");
  check(col_clip10[0] <= 1023, "column clip 10-bit");

  // Extreme dequant values at the smallest shifts exercise the int64-before-
  // clip path; the output must saturate, not wrap.
  for (int trial = 0; trial < 8; ++trial) {
    const int n = (trial & 1) ? 8 : 16;
    const int shift = (trial & 2) ? 6 : 3;
    const Effective1d type =
        (trial & 4) ? Effective1d::kFddt : Effective1d::kDdtx;
    std::int32_t src[16] = {};
    std::int32_t got[16] = {};
    for (int i = 0; i < n; ++i) src[i] = (i & 1) ? 32767 : -32768;
    check(inverse_1d(src, n, shift, false, type, 8, got),
          "inverse_1d extreme succeeds");
    for (int i = 0; i < n; ++i) {
      check(got[i] >= -32768 && got[i] <= 32767,
            "inverse_1d extreme row clip");
    }
    check(inverse_1d(src, n, shift, true, type, 8, got),
          "inverse_1d extreme column succeeds");
    for (int i = 0; i < n; ++i) {
      check(got[i] >= -256 && got[i] <= 255,
            "inverse_1d extreme column clip");
    }
  }

  // Independent naive reference for a random vector: same int64 accumulation
  // and Round2, written out here to catch indexing/rounding regressions.
  for (int trial = 0; trial < 40; ++trial) {
    const int n = (trial & 1) ? 8 : 16;
    const int shift = (trial % 3 == 0) ? 7 : (trial % 3 == 1 ? 11 : 13);
    const Effective1d type =
        (trial % 4 < 2) ? Effective1d::kDdtx : Effective1d::kFddt;
    const std::int16_t* kernel = ddt_kernel(n);
    std::int32_t src[16] = {};
    std::int32_t got[16] = {};
    for (int i = 0; i < n; ++i) src[i] = random_range(-2000, 2000);
    check(inverse_1d(src, n, shift, false, type, 8, got),
          "inverse_1d reference call");
    for (int i = 0; i < n; ++i) {
      const int sample =
          (type == Effective1d::kFddt) ? (n - 1 - i) : i;
      std::int64_t sum = 0;
      for (int j = 0; j < n; ++j) {
        sum += static_cast<std::int64_t>(kernel[j * n + sample]) * src[j];
      }
      const std::int64_t rounded =
          (sum + (std::int64_t{1} << (shift - 1))) >> shift;
      const std::int32_t clipped = static_cast<std::int32_t>(
          std::clamp<std::int64_t>(rounded, -(1 << 15), (1 << 15) - 1));
      check(got[i] == clipped, "inverse_1d matches naive reference");
    }
  }
}

void test_forward_inverse_1d() {
  const int shifts[] = {6, 7, 10, 11, 12, 13};
  for (int n : {8, 16}) {
    for (Effective1d type :
         {Effective1d::kDdtx, Effective1d::kFddt, Effective1d::kAdst,
          Effective1d::kFdst, Effective1d::kDct}) {
      for (int shift : shifts) {
        int max_error = 0;
        for (int trial = 0; trial < 16; ++trial) {
          std::int32_t src[16] = {};
          std::int32_t coeff[16] = {};
          std::int32_t back[16] = {};
          for (int i = 0; i < n; ++i) src[i] = random_range(-255, 255);
          check(forward_1d(src, n, shift, type, coeff),
                "forward_1d succeeds");
          check(inverse_1d(coeff, n, shift, false, type, 8, back),
                "inverse_1d after forward succeeds");
          for (int i = 0; i < n; ++i) {
            max_error = std::max(max_error, std::abs(back[i] - src[i]));
          }
        }
        check(max_error <= 20,
              std::string("1D forward/inverse ") + effective_1d_name(type) +
                  " n=" + std::to_string(n) + " shift=" +
                  std::to_string(shift) + " max err " +
                  std::to_string(max_error));
      }
    }
  }

  // Determinism: repeated calls produce bit-identical output.
  std::int32_t src[16] = {};
  for (int i = 0; i < 16; ++i) src[i] = random_range(-255, 255);
  std::int32_t a[16] = {};
  std::int32_t b[16] = {};
  forward_1d(src, 16, 13, Effective1d::kFddt, a);
  forward_1d(src, 16, 13, Effective1d::kFddt, b);
  check(std::memcmp(a, b, sizeof(a)) == 0, "1D forward determinism");
}

// ---------------------------------------------------------------------------
// 2D inverse matches txv exactly when DDT is off
// ---------------------------------------------------------------------------

void test_block_matches_txv() {
  Context ctx;
  ctx.enable_inter_ddt = false;
  ctx.is_inter = true;
  ctx.bit_depth = 8;
  for (int size = 0; size < 25; ++size) {
    for (int type = 0; type < 16; ++type) {
      const auto params_opt = txv::make_tx_params(
          static_cast<txv::TxSize>(size), static_cast<txv::TxType>(type));
      if (!params_opt.has_value()) continue;
      const txv::TxParams& p = *params_opt;
      std::vector<std::int32_t> dequant(
          static_cast<std::size_t>(p.w * p.h));
      for (std::int32_t& v : dequant) v = random_range(-3000, 3000);
      std::vector<std::int16_t> got(
          static_cast<std::size_t>(p.coded_w * p.coded_h), 0);
      std::vector<std::int16_t> expected(
          static_cast<std::size_t>(p.coded_w * p.coded_h), 0);
      check(inverse_block(dequant.data(), p, ctx, got.data()),
            "inverse_block succeeds");
      txv::inverse_block(dequant.data(), p, expected.data());
      check(std::memcmp(got.data(), expected.data(),
                        got.size() * sizeof(std::int16_t)) == 0,
            std::string("inverse_block == txv for ") +
                txv::tx_size_name(static_cast<txv::TxSize>(size)) + " " +
                txv::tx_type_name(static_cast<txv::TxType>(type)));
    }
  }
}

// ---------------------------------------------------------------------------
// Legal DDT configurations
// ---------------------------------------------------------------------------

struct SizeRow {
  const char* name;
  txv::TxSize size;
  std::uint16_t legal_mask;      // Tx_Type_In_Set_Inter, reduced_tx_set = 0
  std::uint16_t component_mask;  // any ADST/FDST at adjusted length 8/16
  std::uint16_t ddt_mask;        // legal_mask & component_mask
};

// §5.20.8.3 + Tx_Type_In_Set_Inter.  Masks are pinned from the spec tables.
const SizeRow kSizeRows[] = {
    {"4x4", txv::TxSize::k4x4, 0xffff, 0x0000, 0x0000},
    {"8x8", txv::TxSize::k8x8, 0xffff, 0xf1fe, 0xf1fe},
    {"16x16", txv::TxSize::k16x16, 0x0fff, 0xf1fe, 0x01fe},
    {"32x32", txv::TxSize::k32x32, 0x0201, 0x0000, 0x0000},
    {"64x64", txv::TxSize::k64x64, 0x0001, 0x0000, 0x0000},
    {"4x8", txv::TxSize::k4x8, 0xffff, 0x51da, 0x51da},
    {"8x4", txv::TxSize::k8x4, 0xffff, 0xa1ec, 0xa1ec},
    {"8x16", txv::TxSize::k8x16, 0xffff, 0xf1fe, 0xf1fe},
    {"16x8", txv::TxSize::k16x8, 0xffff, 0xf1fe, 0xf1fe},
    {"16x32", txv::TxSize::k16x32, 0xae25, 0xa1ec, 0xa024},
    {"32x16", txv::TxSize::k32x16, 0x5e13, 0x51da, 0x5012},
    {"32x64", txv::TxSize::k32x64, 0x0001, 0x0000, 0x0000},
    {"64x32", txv::TxSize::k64x32, 0x0001, 0x0000, 0x0000},
    {"4x16", txv::TxSize::k4x16, 0xffff, 0x51da, 0x51da},
    {"16x4", txv::TxSize::k16x4, 0xffff, 0xa1ec, 0xa1ec},
    {"8x32", txv::TxSize::k8x32, 0xae25, 0xa1ec, 0xa024},
    {"32x8", txv::TxSize::k32x8, 0x5e13, 0x51da, 0x5012},
    {"16x64", txv::TxSize::k16x64, 0x0425, 0xa1ec, 0x0024},
    {"64x16", txv::TxSize::k64x16, 0x0813, 0x51da, 0x0012},
    {"4x32", txv::TxSize::k4x32, 0xae25, 0x0000, 0x0000},
    {"32x4", txv::TxSize::k32x4, 0x5e13, 0x0000, 0x0000},
    {"8x64", txv::TxSize::k8x64, 0x0425, 0xa1ec, 0x0024},
    {"64x8", txv::TxSize::k64x8, 0x0813, 0x51da, 0x0012},
    {"4x64", txv::TxSize::k4x64, 0x0425, 0x0000, 0x0000},
    {"64x4", txv::TxSize::k64x4, 0x0813, 0x0000, 0x0000},
};

void test_plan_masks() {
  Context ctx;
  ctx.enable_inter_ddt = true;
  ctx.is_inter = true;
  int legal_ddt_pairs = 0;
  for (const SizeRow& row : kSizeRows) {
    std::uint16_t got = 0;
    for (int type = 0; type < 16; ++type) {
      const auto params = txv::make_tx_params(
          row.size, static_cast<txv::TxType>(type));
      if (!params.has_value()) continue;
      if (plan_has_ddt(*params, ctx)) {
        got |= static_cast<std::uint16_t>(1U << type);
      }
    }
    check(got == row.component_mask,
          std::string("DDT component mask ") + row.name);
    for (int type = 0; type < 16; ++type) {
      if ((row.ddt_mask & (1U << type)) != 0) ++legal_ddt_pairs;
    }
  }
  check(legal_ddt_pairs == 100, "100 legal (tx_size, tx_type) DDT pairs");

}

void test_block_ddt_roundtrip() {
  // Several residual shapes, so the test does not depend on random noise only.
  const auto fill = [](std::size_t index, int w, int h, int shape) {
    const int x = static_cast<int>(index % static_cast<std::size_t>(w));
    const int y = static_cast<int>(index / static_cast<std::size_t>(w));
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

  Context ctx;
  ctx.enable_inter_ddt = true;
  ctx.is_inter = true;
  ctx.bit_depth = 8;

  int active_cases = 0;
  for (const SizeRow& row : kSizeRows) {
    for (int type = 0; type < 16; ++type) {
      if ((row.ddt_mask & (1U << type)) == 0) continue;
      ++active_cases;
      const auto params = txv::make_tx_params(
          row.size, static_cast<txv::TxType>(type));
      if (!params.has_value()) {
        check(false, "make_tx_params for a pinned DDT case");
        continue;
      }
      const txv::TxParams& p = *params;
      const int count = p.w * p.h;
      for (int shape = 0; shape < 6; ++shape) {
        std::vector<std::int16_t> residual(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
          residual[static_cast<std::size_t>(i)] = static_cast<std::int16_t>(
              fill(static_cast<std::size_t>(i), p.w, p.h, shape));
        }
        std::vector<std::int32_t> coeff(static_cast<std::size_t>(count), 0);
        std::vector<std::int32_t> levels(static_cast<std::size_t>(count), 0);
        std::vector<std::int32_t> dequant(static_cast<std::size_t>(count), 0);
        std::vector<std::int16_t> coded(
            static_cast<std::size_t>(p.coded_w * p.coded_h), 0);
        check(forward_block(residual.data(), p, ctx, coeff.data()),
              "forward_block DDT succeeds");
        for (int i = 0; i < count; ++i) {
          levels[static_cast<std::size_t>(i)] =
              level_from_coeff(coeff[static_cast<std::size_t>(i)], p);
          dequant[static_cast<std::size_t>(i)] =
              (levels[static_cast<std::size_t>(i)] * 66) / p.dq_denom;
        }
        check(inverse_block(dequant.data(), p, ctx, coded.data()),
              "inverse_block DDT succeeds");
        // The adjusted-size residual is duplicated for 64-coded dimensions.
        int max_error = 0;
        std::int64_t sum_error = 0;
        for (int y = 0; y < p.coded_h; ++y) {
          for (int x = 0; x < p.coded_w; ++x) {
            const int ax = x / (p.coded_w / p.w);
            const int ay = y / (p.coded_h / p.h);
            const int source = residual[static_cast<std::size_t>(ay * p.w + ax)];
            const int error =
                std::abs(static_cast<int>(
                             coded[static_cast<std::size_t>(y * p.coded_w + x)]) -
                         source);
            max_error = std::max(max_error, error);
            sum_error += error;
          }
        }
        const double mean_error =
            static_cast<double>(sum_error) /
            static_cast<double>(p.coded_w * p.coded_h);
        check(max_error <= 96,
              std::string("DDT block max error ") + row.name + " type " +
                  std::to_string(type) + " shape " + std::to_string(shape) +
                  " = " + std::to_string(max_error));
        check(mean_error <= 24.0,
              std::string("DDT block mean error ") + row.name + " type " +
                  std::to_string(type));
      }

      // DDT must actually change the transform for at least one coefficient
      // pattern (a type with a DDT component but an all-zero response would
      // silently disable the feature).
      std::vector<std::int32_t> coeff(static_cast<std::size_t>(count), 0);
      for (int i = 0; i < count; ++i) {
        coeff[static_cast<std::size_t>(i)] = random_range(-500, 500);
      }
      std::vector<std::int16_t> with_ddt(
          static_cast<std::size_t>(p.coded_w * p.coded_h), 0);
      std::vector<std::int16_t> without_ddt(
          static_cast<std::size_t>(p.coded_w * p.coded_h), 0);
      check(inverse_block(coeff.data(), p, ctx, with_ddt.data()),
            "inverse_block DDT comparison");
      Context off = ctx;
      off.enable_inter_ddt = false;
      check(inverse_block(coeff.data(), p, off, without_ddt.data()),
            "inverse_block no-DDT comparison");
      check(std::memcmp(with_ddt.data(), without_ddt.data(),
                        with_ddt.size() * sizeof(std::int16_t)) != 0,
            std::string("DDT changes reconstruction for ") + row.name +
                " type " + std::to_string(type));
    }
  }
  check(active_cases == 100, "round-trip covers all 100 legal DDT pairs");
}

// Real coefficient blocks extracted from testdata/shine_256x256_1f.y4m:
// a 16x16 luma block, DC prediction, txv forward with ADST_ADST and the shared
// 528-step quantiser.  Kept as levels so the test exercises a real sparse
// spectrum rather than synthetic noise.  (dq_denom = 1 for 16x16.)
const std::int32_t kReal16x16Levels[16 * 16] = {
    -102,  -22,  -25,  -23,  -21,    2,  -13,    0,   24,   -9,   -3,    3,    4,   11,   11,   -8,
      50,    3,   31,    5,    8,   15,   -2,    6,   48,   -6,    6,    7,    9,   16,   19,   -8,
     -23,  -18,  -20,   -1,    4,    1,  -19,    0,   45,   -9,    6,    5,    1,   13,   18,   -5,
     -19,  -17,  -15,    1,    8,   -3,  -14,    4,   19,   -7,    4,    0,   -2,    5,   10,   -3,
      -8,   -5,   15,   -5,    1,    5,    1,    7,   -2,   -4,   -1,   -2,    4,    1,    2,   -4,
      20,    9,   21,   -6,   -4,   10,    7,   -2,   11,    0,    0,    1,    8,    4,    1,   -2,
     -19,    6,  -32,   -8,  -15,   -5,   -6,  -16,   21,    4,    1,    7,    0,    3,    2,    3,
     -20,    6,  -30,   -4,  -16,  -10,   -5,   -9,   15,    5,    1,    8,   -3,    3,    3,    3,
       9,    6,   14,   -4,   -5,    5,    7,    1,    2,   -1,    0,   -1,    4,    2,   -1,   -1,
      11,    0,   14,   -4,    4,   12,    6,   -3,    0,   -3,   -3,   -5,    5,   -1,   -2,   -1,
      -1,   -5,    7,   -2,    6,    7,    0,    1,   -3,   -1,   -5,   -3,    3,   -1,    0,   -2,
      12,    2,   19,   -1,    2,    6,    5,    5,   -4,   -1,   -1,   -2,    4,    0,    0,   -3,
       3,    3,    3,   -3,   -4,    1,    3,   -4,    1,   -2,    3,   -2,    3,    0,   -2,    0,
      -5,    0,   -9,   -3,   -6,   -3,   -1,   -5,    4,   -1,    3,   -1,    1,    0,   -1,    2,
       0,    1,    0,   -2,   -4,   -2,    1,    0,    3,    1,    2,   -1,    2,    2,    0,    0,
       3,    2,    6,    1,    2,    3,    3,    3,   -3,    0,   -3,    0,    0,    0,   -1,   -1,
};

// The source residual for the block above (pixel - 128), so the test can
// check an actual reconstruction error, not just non-zero output.
const std::int32_t kReal16x16Residual[16 * 16] = {
      84,  -26,  -29,  -40,  -88,   85,  -10,  -21,   95,   50, -118, -116,   63,   35,  -69,   65,
      70, -117, -124, -125,  -93,   83,  -89, -128,  -53,  104,  -83, -108,   80,  -70, -128,  -31,
      69, -118, -125, -124,  -92,   83,  -80, -119, -109,   82,  -41, -116,   78,  -55, -128,  -15,
      89,    4,    4,  -18,  -89,   82,  -80, -117, -118,   58,  -16, -128,   26,   76,   -1,   92,
     107,  115,  124,   40,  -98,   83,  -80, -117, -119,   51,  -10, -128,   -9,  125,  110,  120,
      77,  -72,  -76,  -98,  -94,   83,  -80, -117, -117,   63,  -22, -118,   78,    1, -100,   32,
      70, -113, -120, -119,  -92,   83,  -81, -121, -101,   91,  -54,  -98,   87,  -85, -128,  -55,
      70, -115, -122, -124,  -93,   83,  -88, -128,  -12,  100,  -97,  -98,   91,  -66, -128,  -32,
      99,   44,   41,   44,  -41,   86,   53,   52,  122,   16, -122, -115,   60,   76,   -4,   96,
      89,   98,  100,   89,  -62,   69,  100,   99,   32, -103, -115, -119,  -65,   85,  118,   66,
     -96,  -96,  -96,  -99, -111,  -98,  -96,  -98, -117, -115, -112, -112, -118, -102,  -80, -108,
    -115, -115, -115, -115, -113, -115, -115, -115, -113, -112, -112, -112, -112, -115, -117, -114,
    -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112,
    -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112,
    -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112,
    -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112,
};

// Encoder forward scale compatibility: our exact-transpose forward plus the
// 528/8 dequant step must reproduce txv's forward + quant_factor + dequant.
// txv::forward_block()/quantize() are the contract the M2 residual module
// already uses.  IDTX is excluded: txv forwards it unchanged (a 2x gain on the
// identity path), while ddt::forward_block compensates the identity scale.
void test_forward_scale_matches_txv() {
  Context ctx;
  ctx.enable_inter_ddt = true;
  ctx.is_inter = true;
  for (int size = 0; size < 25; ++size) {
    for (int type = 0; type < 16; ++type) {
      const auto params = txv::make_tx_params(
          static_cast<txv::TxSize>(size), static_cast<txv::TxType>(type));
      if (!params.has_value()) continue;
      const txv::TxParams& p = *params;
      const Tx1dType row = static_cast<Tx1dType>(p.row_type);
      const Tx1dType col = static_cast<Tx1dType>(p.col_type);
      if (row == Tx1dType::kIdt || col == Tx1dType::kIdt) continue;
      // txv's forward uses the signalled ADST/FDST kernel; when DDT rewrites
      // that direction the coefficient arrays are intentionally different
      // transforms, so only non-DDT blocks can be compared scale-for-scale.
      if (plan_has_ddt(p, ctx)) continue;
      const int count = p.w * p.h;
      std::vector<std::int16_t> residual(static_cast<std::size_t>(count));
      for (int i = 0; i < count; ++i) residual[static_cast<std::size_t>(i)] =
          static_cast<std::int16_t>(random_range(-160, 160));
      std::vector<double> txv_coeff(static_cast<std::size_t>(count));
      std::vector<std::int32_t> txv_levels(static_cast<std::size_t>(count));
      std::vector<std::int32_t> txv_dequant(static_cast<std::size_t>(count));
      txv::forward_transform(residual.data(), p, txv_coeff.data(),
                             txv_levels.data());
      txv::dequantize(txv_levels.data(), p, txv_dequant.data());
      std::vector<std::int32_t> coeff(static_cast<std::size_t>(count));
      std::vector<std::int32_t> levels(static_cast<std::size_t>(count));
      std::vector<std::int32_t> dequant(static_cast<std::size_t>(count));
      check(forward_block(residual.data(), p, ctx, coeff.data()),
            "forward_block scale succeeds");
      for (int i = 0; i < count; ++i) {
        levels[static_cast<std::size_t>(i)] =
            level_from_coeff(coeff[static_cast<std::size_t>(i)], p);
        dequant[static_cast<std::size_t>(i)] =
            (levels[static_cast<std::size_t>(i)] * 66) / p.dq_denom;
      }
      int max_delta = 0;
      for (int i = 0; i < count; ++i) {
        max_delta = std::max(
            max_delta,
            std::abs(dequant[static_cast<std::size_t>(i)] -
                     txv_dequant[static_cast<std::size_t>(i)]));
      }
      // A level step is 66/dq_denom dequant units; allow rounding across the
      // rational 2896 normalisation and the two quantiser paths.
      check(max_delta <= 70 / p.dq_denom,
            std::string("forward scale vs txv ") +
                txv::tx_size_name(static_cast<txv::TxSize>(size)) + " " +
                txv::tx_type_name(static_cast<txv::TxType>(type)) +
                " max delta " + std::to_string(max_delta));
    }
  }
}

void test_real_coefficients() {
  Context ctx;
  ctx.enable_inter_ddt = true;
  ctx.is_inter = true;
  ctx.bit_depth = 8;
  const auto params = txv::make_tx_params(txv::TxSize::k16x16,
                                          txv::TxType::kAdstAdst);
  check(params.has_value(), "real coefficient params");
  if (!params.has_value()) return;
  std::vector<std::int32_t> dequant(kReal16x16Levels,
                                    kReal16x16Levels + 16 * 16);
  for (std::int32_t& v : dequant) v *= 66;  // dq_denom == 1
  std::vector<std::int16_t> residual(16 * 16, 0);
  check(inverse_block(dequant.data(), *params, ctx, residual.data()),
        "real coefficient inverse");
  // Reconstruction error against the real source residual.
  std::int64_t sse = 0;
  int max_error = 0;
  std::int64_t abs_error = 0;
  for (int i = 0; i < 16 * 16; ++i) {
    const int error = std::abs(static_cast<int>(residual[static_cast<std::size_t>(i)]) -
                               kReal16x16Residual[i]);
    max_error = std::max(max_error, error);
    abs_error += error;
    sse += error * error;
  }
  check(max_error <= 20,
        "real coefficient max error = " + std::to_string(max_error));
  check(abs_error / (16 * 16) <= 5, "real coefficient mean abs error");
  check(sse <= 6000, "real coefficient SSE");
  // Determinism on the real block.
  std::vector<std::int16_t> again(16 * 16, 0);
  inverse_block(dequant.data(), *params, ctx, again.data());
  check(std::memcmp(residual.data(), again.data(),
                    residual.size() * sizeof(std::int16_t)) == 0,
        "real coefficient determinism");
  // FDDT is the mirror of DDTX per 1D pass; the 2D output must differ from the
  // ADST_ADST reconstruction.
  Context off = ctx;
  off.enable_inter_ddt = false;
  std::vector<std::int16_t> plain(16 * 16, 0);
  inverse_block(dequant.data(), *params, off, plain.data());
  check(std::memcmp(residual.data(), plain.data(),
                    residual.size() * sizeof(std::int16_t)) != 0,
        "real DDT coefficients differ from ADST_ADST");
}

// ---------------------------------------------------------------------------
// Constraint validation
// ---------------------------------------------------------------------------

void test_constraints() {
  // Legal DDT matrix for luma, reduced_tx_set = 0, inter, non-IBC.
  for (const SizeRow& row : kSizeRows) {
    for (int type = 0; type < 16; ++type) {
      BlockSignal signal;
      signal.tx_size = static_cast<int>(row.size);
      signal.tx_type = type;
      signal.plane = 0;
      signal.is_inter = true;
      signal.enable_inter_ddt = true;
      const Validation v = validate(signal);
      const bool expected = (row.ddt_mask & (1U << type)) != 0;
      check(v.ddt_active == expected,
            std::string("validate DDT-active ") + row.name + " type " +
                std::to_string(type) + " status " + status_name(v.status));
      if (expected) {
        check(v.status == Status::kOk,
              std::string("validate status ok ") + row.name);
        check(plan_has_ddt(*txv::make_tx_params(
                                  static_cast<txv::TxSize>(signal.tx_size),
                                  static_cast<txv::TxType>(type)),
                              [] {
                                Context c;
                                c.enable_inter_ddt = true;
                                c.is_inter = true;
                                return c;
                              }()),
              "validate plan changes kernel");
      } else {
        check(v.status == Status::kNoDdtComponent ||
                  v.status == Status::kTypeNotInSet,
              std::string("validate inactive status ") + row.name);
      }
    }
  }

  BlockSignal base;
  base.tx_size = static_cast<int>(txv::TxSize::k8x8);
  base.tx_type = static_cast<int>(txv::TxType::kAdstAdst);
  base.is_inter = true;
  base.enable_inter_ddt = true;

  {
    BlockSignal s = base;
    s.enable_inter_ddt = false;
    check(validate(s).status == Status::kDisabled, "disabled status");
  }
  {
    BlockSignal s = base;
    s.is_inter = false;
    check(validate(s).status == Status::kNotInter, "not-inter status");
  }
  {
    BlockSignal s = base;
    s.use_intrabc = true;
    check(validate(s).status == Status::kIntrabc, "intrabc status");
  }
  {
    BlockSignal s = base;
    s.lossless = true;
    check(validate(s).status == Status::kLossless, "lossless status");
  }
  {
    BlockSignal s = base;
    s.fsc_mode = true;
    check(validate(s).status == Status::kLossless, "fsc status");
  }
  {
    BlockSignal s = base;
    s.enable_chroma_dctonly = true;
    s.plane = 1;
    check(validate(s).status == Status::kChromaDctOnly,
          "chroma dctonly status");
  }
  {
    BlockSignal s = base;
    s.sec_tx_type = true;
    check(validate(s).status == Status::kIstConflict, "IST conflict status");
  }
  {
    BlockSignal s = base;
    s.reduced_tx_set = 1;
    check(validate(s).status == Status::kNoDdtComponent ||
              validate(s).status == Status::kTypeNotInSet,
          "reduced_tx_set 1 removes DDT");
    s.reduced_tx_set = 3;
    check(!validate(s).ddt_active, "reduced_tx_set 3 removes DDT");
  }
  {
    // Chroma: 16x16 luma -> 8x8 chroma.  ADST_ADST is in INTER_1 for 8x8 and
    // DDT applies on chroma when enable_chroma_dctonly = 0.
    BlockSignal s = base;
    s.tx_size = static_cast<int>(txv::TxSize::k8x8);
    s.plane = 1;
    s.enable_chroma_dctonly = false;
    check(validate(s).status == Status::kOk && validate(s).ddt_active,
          "chroma 8x8 DDT active");
    // 4x4 chroma cannot carry DDT.
    s.tx_size = static_cast<int>(txv::TxSize::k4x4);
    check(!validate(s).ddt_active, "4x4 chroma has no DDT");
    // A luma type outside the 16x16 chroma set aliases to DCT_DCT.  Type 12
    // (IDT_ADST) is in INTER_1 but not in INTER_2.
    s.tx_size = static_cast<int>(txv::TxSize::k16x16);
    s.tx_type = static_cast<int>(txv::TxType::kVadst);
    check(!validate(s).ddt_active, "chroma out-of-set type aliases to DCT");
  }
  {
    BlockSignal s = base;
    s.enable_chroma_dctonly = false;
    s.plane = 1;
    s.tx_type = static_cast<int>(txv::TxType::kIdtx);
    check(!validate(s).ddt_active, "chroma IDTX has no DDT");
  }
  {
    // TCQ and QM eligibility are properties of PlaneTxType, not of DDT.
    BlockSignal s = base;
    s.allow_tcq = true;
    s.using_qmatrix = true;
    Validation v = validate(s);
    check(v.tcq_eligible, "2D ADST is TCQ eligible");
    check(v.qm_eligible, "ADST is QM eligible");
    s.tx_type = static_cast<int>(txv::TxType::kVadst);
    s.tx_size = static_cast<int>(txv::TxSize::k8x16);
    v = validate(s);
    check(!v.tcq_eligible, "V_ADST is not TCQ eligible");
    // Tx_Type < IDTX is the QM condition; V_ADST (12) is above IDTX (9), so
    // 1D ADST types can carry DDT but never QM.
    check(!v.qm_eligible, "V_ADST is not QM eligible");
    s.tx_type = static_cast<int>(txv::TxType::kIdtx);
    v = validate(s);
    check(!v.qm_eligible, "IDTX is not QM eligible");
  }
}

// ---------------------------------------------------------------------------
// Syntax
// ---------------------------------------------------------------------------

void test_syntax() {
  {
    BitWriter writer;
    check(write_enable_inter_ddt(writer, true, true) == 0,
          "still picture writes no DDT bit");
    check(writer.bit_size() == 0, "still picture bit count");
  }
  {
    BitWriter writer;
    check(write_enable_inter_ddt(writer, false, false) == 1,
          "non-still writes one DDT bit");
    writer.align_to_byte();
    check(writer.bytes().size() == 1 && writer.bytes()[0] == 0x00,
          "DDT off golden 0x00");
  }
  {
    BitWriter writer;
    write_enable_inter_ddt(writer, false, true);
    writer.align_to_byte();
    check(writer.bytes().size() == 1 && writer.bytes()[0] == 0x80,
          "DDT on golden 0x80");
  }
  // Round-trip at every bit offset, with and without a prefix.
  for (int offset = 0; offset < 8; ++offset) {
    BitWriter writer;
    for (int i = 0; i < offset; ++i) writer.put_bit((i & 1) != 0);
    write_enable_inter_ddt(writer, false, (offset & 1) != 0);
    writer.align_to_byte();
    const auto bytes = writer.bytes();
    bool value = false;
    std::size_t next = 0;
    check(read_enable_inter_ddt(bytes, static_cast<std::size_t>(offset),
                                false, &value, &next),
          "read DDT bit");
    check(value == ((offset & 1) != 0), "DDT bit round-trip");
    check(next == static_cast<std::size_t>(offset + 1), "DDT next bit");
  }
  {
    const std::uint8_t data[1] = {0xFF};
    bool value = true;
    std::size_t next = 0;
    check(read_enable_inter_ddt(data, 0, true, &value, &next),
          "still picture read succeeds");
    check(!value && next == 0, "still picture infers DDT off");
    check(!read_enable_inter_ddt(data, 8, false, &value, &next),
          "out-of-range DDT read fails");
  }
  // Payload growth: syntax cost is 1 bit; the byte-aligned OBU payload grows by
  // 0 or 1 byte depending on the total header length.
  int grew = 0;
  int same = 0;
  for (std::size_t total = 0; total < 128; ++total) {
    const std::size_t delta = payload_bytes_added(total, false);
    check(delta <= 1, "payload growth at most one byte");
    (delta ? grew : same) += 1;
  }
  check(grew > 0 && same > 0, "payload growth is position dependent");
  // Pinned M2 gate header: 144 syntax bits without DDT, 145 with it; both fit
  // in the same 19-byte payload.
  check(payload_bytes_added(144, false) == 0, "gate header adds no payload byte");
  check(payload_bytes_added(150, false) == 1, "header can add one payload byte");
  for (std::size_t total = 0; total < 128; ++total) {
    check(payload_bytes_added(total, true) == 0,
          "still picture adds no payload byte");
  }
}

}  // namespace

int main() {
  test_kernels();
  test_effective_1d();
  test_inverse_1d_goldens();
  test_forward_inverse_1d();
  test_block_matches_txv();
  test_plan_masks();
  test_block_ddt_roundtrip();
  test_forward_scale_matches_txv();
  test_real_coefficients();
  test_constraints();
  test_syntax();
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
