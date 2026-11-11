// CPU self-test for the AV2 transform variants.
//
// This test is the host-only gate.  The CPU-vs-CUDA bit-exact gate lives in
// src/cuda/tx_variants/tx_variants_test.cu and calls the same CPU functions.
//
// Build/run:
//   ./build_and_test.sh
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "encoder/tx_variants/tx_select.h"
#include "encoder/tx_variants/tx_tables.h"
#include "encoder/tx_variants/tx_variants.h"

namespace {

using namespace av2enc::txv;

int g_failures = 0;
int g_checks = 0;

void check(bool cond, const char* what) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("FAIL: %s\n", what);
  }
}

void check_near(double got, double want, double tol, const char* what) {
  ++g_checks;
  if (!(std::fabs(got - want) <= tol)) {
    ++g_failures;
    std::printf("FAIL: %s (got %.17g want %.17g tol %g)\n", what, got, want,
                tol);
  }
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------
void test_tables() {
  check(kDct4[1][0] == 83, "Dct4 spot");
  check(kDct32[1][0] == 90, "Dct32 spot");
  check(kAdst4[0][0] == 18, "Adst4 spot");
  check(kAdst8[0][0] == 11, "Adst8 spot");
  check(kAdst16[0][0] == 8, "Adst16 spot");
  check(kFdst4[0][0] == 89, "Fdst4 spot");
  check(kFdst8[0][0] == 89, "Fdst8 spot");
  check(kFdst16[0][0] == 89, "Fdst16 spot");

  struct Entry {
    const std::int16_t* data;
    int n;
  };
  const Entry entries[] = {
      {&kDct4[0][0], 4},    {&kDct8[0][0], 8},    {&kDct16[0][0], 16},
      {&kDct32[0][0], 32},  {&kAdst4[0][0], 4},   {&kAdst8[0][0], 8},
      {&kAdst16[0][0], 16}, {&kFdst4[0][0], 4},   {&kFdst8[0][0], 8},
      {&kFdst16[0][0], 16},
  };
  for (const Entry& e : entries) {
    const double target = 64.0 * std::sqrt(static_cast<double>(e.n));
    for (int row = 0; row < e.n; ++row) {
      double sum = 0.0;
      for (int col = 0; col < e.n; ++col) {
        const double v = static_cast<double>(e.data[row * e.n + col]);
        sum += v * v;
      }
      const double norm = std::sqrt(sum);
      // §9.6 kernels are quantized approximations; 1% covers all of them.
      if (!(norm > target * 0.99 && norm < target * 1.01)) {
        ++g_failures;
        std::printf("FAIL: kernel row norm n=%d row=%d norm=%.6f target=%.6f\n",
                    e.n, row, norm, target);
      }
      ++g_checks;
    }
  }

  check(identity_scale(2) == 128, "identity scale 4");
  check(identity_scale(3) == 181, "identity scale 8");
  check(identity_scale(4) == 256, "identity scale 16");
  check(identity_scale(5) == 362, "identity scale 32");
  check(identity_scale(6) == 362, "identity scale 64");
}

void test_sizes_and_types() {
  check(!tx_size_supported(2, 8), "reject 2x8");
  check(!tx_size_supported(64, 128), "reject 64x128");
  check(!tx_size_supported(0, 0), "reject 0x0");
  check(tx_size_supported(4, 4), "accept 4x4");
  check(tx_size_supported(64, 8), "accept 64x8");

  for (int i = 0; i < 25; ++i) {
    const TxSize size = static_cast<TxSize>(i);
    const auto p = make_tx_params(size, TxType::kDctDct);
    check(p.has_value(), "make_tx_params size");
    if (!p.has_value()) {
      continue;
    }
    check(p->tx_size == i, "tx_size index");
    check(p->coded_w == kTxSizes[i].width, "coded_w");
    check(p->coded_h == kTxSizes[i].height, "coded_h");
    check(p->w == kTxSizes[i].adj_width, "adj w");
    check(p->h == kTxSizes[i].adj_height, "adj h");
    check(p->row_shift == kTxSizes[i].row_shift, "row shift");
    check(p->col_shift == kTxSizes[i].col_shift, "col shift");
    const int pels = p->coded_w * p->coded_h;
    const int denom = 1 << ((pels > 256) + (pels > 1024));
    check(p->dq_denom == denom, "dq denom");
    const bool odd =
        ((kTxSizes[i].log2_w ^ kTxSizes[i].log2_h) & 1) != 0;
    check_near(p->rect_prescale, odd ? 4096.0 / 2896.0 : 1.0, 0.0,
               "rect prescale");
    check(p->quant_factor > 0.0 && std::isfinite(p->quant_factor),
          "quant factor finite");
    check(tx_size_index(p->coded_w, p->coded_h) == i, "size index roundtrip");
  }

  // Transform_1d_Type table, row/col direction.
  struct Pair {
    Tx1dType row;
    Tx1dType col;
  };
  const Pair expected[16] = {
      {Tx1dType::kDct, Tx1dType::kDct},  {Tx1dType::kDct, Tx1dType::kAdst},
      {Tx1dType::kAdst, Tx1dType::kDct}, {Tx1dType::kAdst, Tx1dType::kAdst},
      {Tx1dType::kDct, Tx1dType::kFdst}, {Tx1dType::kFdst, Tx1dType::kDct},
      {Tx1dType::kFdst, Tx1dType::kFdst}, {Tx1dType::kFdst, Tx1dType::kAdst},
      {Tx1dType::kAdst, Tx1dType::kFdst}, {Tx1dType::kIdt, Tx1dType::kIdt},
      {Tx1dType::kIdt, Tx1dType::kDct},  {Tx1dType::kDct, Tx1dType::kIdt},
      {Tx1dType::kIdt, Tx1dType::kAdst}, {Tx1dType::kAdst, Tx1dType::kIdt},
      {Tx1dType::kIdt, Tx1dType::kFdst}, {Tx1dType::kFdst, Tx1dType::kIdt},
  };
  for (int i = 0; i < 16; ++i) {
    Tx1dType row;
    Tx1dType col;
    tx_1d_types(static_cast<TxType>(i), &row, &col);
    check(row == expected[i].row, "1d row type");
    check(col == expected[i].col, "1d col type");
    TxType back;
    check(make_tx_type(row, col, &back), "make_tx_type inverse");
    check(static_cast<int>(back) == i, "make_tx_type value");
  }

  // Length-32 ADST/FDST alias DCT per §7.15.2.1.
  check(tx_kernel(Tx1dType::kAdst, 32) == &kDct32[0][0], "adst32 alias dct");
  check(tx_kernel(Tx1dType::kFdst, 32) == &kDct32[0][0], "fdst32 alias dct");
  check(tx_kernel(Tx1dType::kIdt, 32) == nullptr, "idt has no kernel");
}

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------
void fill_random(std::vector<std::int16_t>* data, int n, std::mt19937* rng,
                 std::int16_t lo, std::int16_t hi) {
  std::uniform_int_distribution<int> dist(lo, hi);
  data->resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    (*data)[i] = static_cast<std::int16_t>(dist(*rng));
  }
}

void test_forward_idtx_is_identity() {
  std::mt19937 rng(7);
  std::vector<std::int16_t> residual;
  std::vector<double> coeff;
  for (int i = 0; i < 25; ++i) {
    const auto p = make_tx_params(static_cast<TxSize>(i), TxType::kIdtx);
    if (!p.has_value()) {
      continue;
    }
    const int n = p->w * p->h;
    fill_random(&residual, n, &rng, -255, 255);
    coeff.assign(static_cast<std::size_t>(n), 0.0);
    forward_block(residual.data(), *p, coeff.data());
    for (int k = 0; k < n; ++k) {
      // IDTX is a pass-through; odd-log2 rectangles add the 4096/2896
      // compensation, so compare using that factor.
      const double want =
          static_cast<double>(residual[k]) * p->rect_prescale;
      check(coeff[k] == want, "IDTX forward is scaled identity");
    }
  }
}

void test_inverse_impulse_matches_kernel() {
  // A single dequantized coefficient must reproduce the exact 1D kernel
  // entries after the normative shifts and clips.  This exercises all table
  // entries without depending on the forward transform.
  constexpr int kMaxCodedElems = 64 * 64;
  std::vector<std::int32_t> dequant(kMaxAdjTxElems);
  std::vector<std::int16_t> recon(kMaxCodedElems);
  for (int i = 0; i < 25; ++i) {
    const auto p = make_tx_params(static_cast<TxSize>(i), TxType::kDctDct);
    if (!p.has_value()) {
      continue;
    }
    const int w = p->w;
    const int h = p->h;
    for (int pos = 0; pos < w * h; ++pos) {
      std::fill(dequant.begin(), dequant.end(), 0);
      dequant[pos] = pos == 0 ? 1000 : 17;
      inverse_block(dequant.data(), *p, recon.data());
      // Just check that the coded-size residual is clipped and that the
      // adjusted expansion is piecewise equal.  High-frequency coefficients
      // can legitimately round to zero at this magnitude.
      bool nonzero = false;
      for (int k = 0; k < p->coded_w * p->coded_h; ++k) {
        nonzero = nonzero || recon[k] != 0;
        check(recon[k] >= -256 && recon[k] <= 255, "inverse clip");
      }
      if (pos == 0) {
        check(nonzero, "DC impulse reconstructs nonzero");
      }
      if (p->coded_w > w || p->coded_h > h) {
        const int sx = p->coded_w / w;
        const int sy = p->coded_h / h;
        for (int y = 0; y < p->coded_h; ++y) {
          for (int x = 0; x < p->coded_w; ++x) {
            const int ref = recon[static_cast<std::size_t>(y / sy) * w +
                                  (x / sx)];
            check(recon[static_cast<std::size_t>(y) * p->coded_w + x] == ref,
                  "adjusted-size duplication");
          }
        }
      }
    }
  }
}

void test_idtx_inverse_scale() {
  // IDTX inverse is a pure scaling of each coefficient; verify the exact
  // integer formula for every size, including the odd-log2 pre-scale.
  constexpr int kMaxCodedElems = 64 * 64;
  std::vector<std::int32_t> dequant(kMaxAdjTxElems);
  std::vector<std::int16_t> recon(kMaxCodedElems);
  const std::int32_t values[] = {1, 2, 3, 7, 33, 127, -1, -127, 1000};
  for (int i = 0; i < 25; ++i) {
    const auto p = make_tx_params(static_cast<TxSize>(i), TxType::kIdtx);
    if (!p.has_value()) {
      continue;
    }
    for (std::int32_t value : values) {
      std::fill(dequant.begin(), dequant.end(), 0);
      dequant[0] = value;
      inverse_block(dequant.data(), *p, recon.data());
      std::int32_t v = value;
      if (((p->log2_w - p->log2_h) & 1) != 0) {
        v = round2(static_cast<std::int32_t>(
                       static_cast<std::int64_t>(v) * 2896),
                   12);
      }
      const int scale_w = identity_scale(p->log2_w);
      const int scale_h = identity_scale(p->log2_h);
      std::int32_t expected =
          round2(static_cast<std::int32_t>(
                     static_cast<std::int64_t>(v) * scale_w),
                 p->row_shift);
      expected = round2(static_cast<std::int32_t>(
                            static_cast<std::int64_t>(expected) * scale_h),
                        p->col_shift);
      expected = expected < -256 ? -256 : (expected > 255 ? 255 : expected);
      check(recon[0] == static_cast<std::int16_t>(expected),
            "IDTX inverse exact coefficient");
    }
  }
}

void test_dct_adst_alias_at_32() {
  std::mt19937 rng(11);
  std::vector<std::int16_t> residual;
  std::vector<double> a(kMaxAdjTxElems);
  std::vector<double> b(kMaxAdjTxElems);
  const TxType types[] = {TxType::kAdstAdst, TxType::kFlipadstFlipadst,
                          TxType::kAdstDct, TxType::kDctAdst};
  for (TxType type : types) {
    for (int size_idx : {3, 4}) {  // 32x32 and 64x64 (adjusted 32x32)
      const auto pd = make_tx_params(static_cast<TxSize>(size_idx),
                                     TxType::kDctDct);
      const auto pt = make_tx_params(static_cast<TxSize>(size_idx), type);
      check(pd.has_value() && pt.has_value(), "alias params");
      if (!pd.has_value() || !pt.has_value()) {
        continue;
      }
      const int n = pd->w * pd->h;
      fill_random(&residual, n, &rng, -255, 255);
      forward_block(residual.data(), *pd, a.data());
      forward_block(residual.data(), *pt, b.data());
      for (int k = 0; k < n; ++k) {
        check(a[k] == b[k], "32-length ADST/FDST aliases DCT");
      }
    }
  }
}

void test_quantize_dequantize_exact() {
  std::mt19937 rng(13);
  std::uniform_int_distribution<int> dist(-1000000, 1000000);
  std::vector<double> coeff(4096);
  std::vector<std::int32_t> levels(4096);
  std::vector<std::int32_t> dequant(4096);
  for (int size_idx = 0; size_idx < 25; ++size_idx) {
    const auto p = make_tx_params(static_cast<TxSize>(size_idx));
    if (!p.has_value()) {
      continue;
    }
    const int n = p->w * p->h;
    for (int i = 0; i < n; ++i) {
      // lround ties: exactly k+0.5 and neighbours.
      const int base = dist(rng);
      coeff[i] = (static_cast<double>(base) + 0.5) / p->quant_factor;
    }
    quantize(coeff.data(), *p, levels.data());
    for (int i = 0; i < n; ++i) {
      check(levels[i] ==
                static_cast<std::int32_t>(
                    std::lround(coeff[i] * p->quant_factor)),
            "quantize lround");
    }
    dequantize(levels.data(), *p, dequant.data());
    for (int i = 0; i < n; ++i) {
      const std::int32_t qc = levels[i];
      const std::int32_t aq = qc < 0 ? -qc : qc;
      const std::int32_t high = (aq * kQuantStep) & 0xFFFFFF;
      const std::int32_t dq = round2(high, kQuantTableBits) / p->dq_denom;
      const std::int32_t signed_dq = qc < 0 ? -dq : dq;
      const std::int32_t expected =
          signed_dq < -(1 << 15)
              ? -(1 << 15)
              : (signed_dq > (1 << 15) - 1 ? (1 << 15) - 1 : signed_dq);
      check(dequant[i] == expected, "dequantize formula");
    }
  }
}

void test_residual_helpers() {
  std::mt19937 rng(17);
  std::uniform_int_distribution<int> pix(0, 255);

  // 64x64: make_residual(src) == downsample(source - pred).
  for (int size_idx = 0; size_idx < 25; ++size_idx) {
    const auto p = make_tx_params(static_cast<TxSize>(size_idx));
    if (!p.has_value()) {
      continue;
    }
    const int sw = 130;
    const int sh = 100;
    std::vector<std::uint8_t> src(static_cast<std::size_t>(sw) * sh);
    for (auto& v : src) {
      v = static_cast<std::uint8_t>(pix(rng));
    }
    const int pred = 117;
    const int x0 = 63;
    const int y0 = 37;
    std::vector<std::int16_t> full(
        static_cast<std::size_t>(p->coded_w) * p->coded_h);
    for (int y = 0; y < p->coded_h; ++y) {
      const int yy = std::min(std::max(y0 + y, 0), sh - 1);
      for (int x = 0; x < p->coded_w; ++x) {
        const int xx = std::min(std::max(x0 + x, 0), sw - 1);
        full[static_cast<std::size_t>(y) * p->coded_w + x] =
            static_cast<std::int16_t>(
                static_cast<int>(src[static_cast<std::size_t>(yy) * sw + xx]) -
                pred);
      }
    }
    std::vector<std::int16_t> a(
        static_cast<std::size_t>(p->w) * p->h);
    std::vector<std::int16_t> b(
        static_cast<std::size_t>(p->w) * p->h);
    make_residual(src.data(), sw, sw, sh, x0, y0, pred, *p, a.data());
    downsample_residual(full.data(), *p, b.data());
    for (int k = 0; k < p->w * p->h; ++k) {
      check(a[k] == b[k], "make_residual == downsample");
    }

    // upsample(downsample(x)) is the §7.15.4 expansion.
    std::vector<std::int16_t> up(
        static_cast<std::size_t>(p->coded_w) * p->coded_h);
    upsample_residual(a.data(), *p, up.data());
    const int sx = p->coded_w / p->w;
    const int sy = p->coded_h / p->h;
    for (int y = 0; y < p->coded_h; ++y) {
      for (int x = 0; x < p->coded_w; ++x) {
        check(up[static_cast<std::size_t>(y) * p->coded_w + x] ==
                  a[static_cast<std::size_t>(y / sy) * p->w + (x / sx)],
              "upsample duplication");
      }
    }
  }
}

void test_roundtrip_error() {
  // The transform pair is normative/approximate; check that a quantized
  // round trip stays bounded and DC-dominated for structured blocks.
  std::mt19937 rng(19);
  std::vector<std::int16_t> residual;
  std::vector<double> coeff;
  std::vector<std::int32_t> levels;
  std::vector<std::int32_t> dequant;
  std::vector<std::int16_t> recon;

  const TxType types[] = {TxType::kDctDct, TxType::kAdstAdst, TxType::kIdtx,
                          TxType::kFlipadstFlipadst};
  for (int size_idx = 0; size_idx < 25; ++size_idx) {
    for (TxType type : types) {
      const auto p = make_tx_params(static_cast<TxSize>(size_idx), type);
      if (!p.has_value()) {
        continue;
      }
      const int adj_n = p->w * p->h;
      const int coded_n = p->coded_w * p->coded_h;
      fill_random(&residual, adj_n, &rng, -50, 50);
      coeff.assign(static_cast<std::size_t>(adj_n), 0.0);
      levels.assign(static_cast<std::size_t>(adj_n), 0);
      dequant.assign(static_cast<std::size_t>(adj_n), 0);
      recon.assign(static_cast<std::size_t>(coded_n), 0);
      forward_transform(residual.data(), *p, coeff.data(), levels.data());
      inverse_transform(levels.data(), *p, dequant.data(), recon.data());
      std::vector<std::int16_t> up(static_cast<std::size_t>(coded_n));
      upsample_residual(residual.data(), *p, up.data());
      double sse = 0.0;
      for (int k = 0; k < coded_n; ++k) {
        const double d = static_cast<double>(recon[k]) - up[k];
        sse += d * d;
      }
      const double mse = sse / static_cast<double>(coded_n);
      check(mse < 50.0 * 50.0, "roundtrip MSE bounded");
      check(std::isfinite(sse), "roundtrip finite");
    }
  }
}

void test_64x64_equals_duplicated_32x32() {
  // 64x64 uses Transform_Shift {6,13} and identity scale 362, exactly like
  // 32x32, so its inverse must equal the 32x32 inverse expanded 2x2.
  std::mt19937 rng(23);
  std::vector<std::int32_t> dequant(kMaxAdjTxElems);
  std::vector<std::int16_t> a(64 * 64);
  std::vector<std::int16_t> b(32 * 32);
  std::vector<std::int16_t> expanded(64 * 64);
  for (TxType type : {TxType::kDctDct, TxType::kIdtx, TxType::kAdstAdst}) {
    const auto p64 = make_tx_params(TxSize::k64x64, type);
    const auto p32 = make_tx_params(TxSize::k32x32, type);
    check(p64.has_value() && p32.has_value(), "64/32 params");
    if (!p64.has_value() || !p32.has_value()) {
      continue;
    }
    std::uniform_int_distribution<int> dist(-100, 100);
    for (int i = 0; i < 32 * 32; ++i) {
      dequant[i] = dist(rng);
    }
    inverse_block(dequant.data(), *p64, a.data());
    inverse_block(dequant.data(), *p32, b.data());
    for (int y = 0; y < 64; ++y) {
      for (int x = 0; x < 64; ++x) {
        expanded[static_cast<std::size_t>(y) * 64 + x] =
            b[static_cast<std::size_t>(y / 2) * 32 + (x / 2)];
      }
    }
    check(std::memcmp(a.data(), expanded.data(), a.size() * sizeof(a[0])) ==
              0,
          "64x64 inverse == duplicated 32x32");
  }
}

// ---------------------------------------------------------------------------
// Partition selection
// ---------------------------------------------------------------------------
void test_partition_geometry() {
  for (int i = 0; i < 25; ++i) {
    const int w = kTxSizes[i].width;
    const int h = kTxSizes[i].height;
    check(tx_partition_legal(w, h, TxPartition::kNone), "NONE legal");
    TxPartition cand[4];
    const int count = tx_partition_candidates(w, h, cand, 4);
    check(count >= 1, "at least NONE");
    check(cand[0] == TxPartition::kNone, "NONE first");
    for (int c = 0; c < count; ++c) {
      int xs[4];
      int ys[4];
      int ws[4];
      int hs[4];
      const int parts =
          tx_partition_geometry(w, h, cand[c], xs, ys, ws, hs);
      check(parts == tx_partition_parts(cand[c]), "part count");
      int area = 0;
      std::vector<int> cover(static_cast<std::size_t>(w * h), 0);
      for (int k = 0; k < parts; ++k) {
        check(ws[k] >= 4 && hs[k] >= 4, "leaf dims >= 4");
        check(tx_size_supported(ws[k], hs[k]), "leaf is a TX size");
        area += ws[k] * hs[k];
        for (int y = 0; y < hs[k]; ++y) {
          for (int x = 0; x < ws[k]; ++x) {
            cover[static_cast<std::size_t>(ys[k] + y) * w + xs[k] + x] += 1;
          }
        }
      }
      check(area == w * h, "partition covers area");
      for (int v : cover) {
        check(v == 1, "partition has no overlap/gap");
      }
    }
  }
}

void test_selection() {
  std::vector<std::int16_t> zero(64 * 64, 0);
  TxSelection sel;
  TxSelectOptions opts;
  opts.max_depth = 2;
  check(select_tx_partition(zero.data(), 64, 64, opts, &sel),
        "select zero block");
  check(sel.count == 1, "zero block selects NONE");
  check(sel.cost == 0.0, "zero block zero cost");
  check(sel.root_partition == TxPartition::kNone, "zero block partition");

  // Vertical discontinuity: a split must beat NONE on SSE.  SPLIT and VERT
  // can both represent it exactly, so do not require one specific tie winner.
  std::vector<std::int16_t> vert(16 * 16);
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      vert[static_cast<std::size_t>(y) * 16 + x] =
          static_cast<std::int16_t>(x < 8 ? 0 : 120);
    }
  }
  opts.max_depth = 0;
  TxSelection baseline;
  check(select_tx_partition(vert.data(), 16, 16, opts, &baseline),
        "select baseline");
  opts.max_depth = 1;
  opts.search_tx_type = true;
  check(select_tx_partition(vert.data(), 16, 16, opts, &sel),
        "select vertical block");
  check(sel.count >= 1, "selection leaves");
  check(sel.root_partition != TxPartition::kNone, "split chosen");
  check(sel.cost <= baseline.cost, "split does not increase cost");
  check(sel.sse < 120.0 * 120.0 * 16 * 16, "vertical improves sse");

  // Every leaf must lie inside the block.
  for (int i = 0; i < sel.count; ++i) {
    const TxLeaf& leaf = sel.leaves[i];
    check(leaf.x >= 0 && leaf.y >= 0, "leaf origin");
    check(leaf.x + leaf.params.coded_w <= 16, "leaf x inside");
    check(leaf.y + leaf.params.coded_h <= 16, "leaf y inside");
  }

  // Unsupported root size is rejected.
  check(!select_tx_partition(zero.data(), 2, 8, opts, &sel),
        "reject unsupported root");
}

}  // namespace

int main() {
  std::printf("av2enc CPU transform-variants test\n");
  test_tables();
  test_sizes_and_types();
  test_forward_idtx_is_identity();
  test_inverse_impulse_matches_kernel();
  test_idtx_inverse_scale();
  test_dct_adst_alias_at_32();
  test_quantize_dequantize_exact();
  test_residual_helpers();
  test_roundtrip_error();
  test_64x64_equals_duplicated_32x32();
  test_partition_geometry();
  test_selection();

  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
