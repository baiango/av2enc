#include "cpu_reference.h"

#include "src/encoder/m0_tables.h"

#include <cmath>
#include <cstddef>

namespace ref {
namespace {

// Same rounding helpers as src/encoder/m0_still.cpp.
int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

void transform_shifts(int w, int h, int* row_shift, int* col_shift) {
  if (w == 32 && h == 32) {
    *row_shift = 6;
    *col_shift = 13;
  } else if (w == 16 && h == 16) {
    *row_shift = 6;
    *col_shift = 13;
  } else if ((w == 32 && h == 16) || (w == 16 && h == 32)) {
    *row_shift = 6;
    *col_shift = 12;
  } else if ((w == 32 && h == 8) || (w == 8 && h == 32)) {
    *row_shift = 6;
    *col_shift = 13;
  } else if ((w == 16 && h == 4) || (w == 4 && h == 16)) {
    *row_shift = 6;
    *col_shift = 12;
  } else if (w <= 16 && h <= 16) {
    *row_shift = 7;
    *col_shift = 11;
  } else {
    *row_shift = 6;
    *col_shift = 13;
  }
}

const std::int16_t* dct_kernel(int n) {
  return n == 16 ? &av2enc::m0::kDct16[0][0] : &av2enc::m0::kDct32[0][0];
}

}  // namespace

int clip1(int value) { return clip3(0, 255, value); }

int round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

BlockParams make_block_params(int n) {
  BlockParams p;
  p.n = n;
  int row_shift = 0;
  int col_shift = 0;
  transform_shifts(n, n, &row_shift, &col_shift);
  const int tx_pels = n * n;
  const int dq_denom = 1 << ((tx_pels > 256) + (tx_pels > 1024));
  p.dq_denom = dq_denom;

  // Exact expression order from m0_still.cpp lines 1270-1277.
  const double inv_scale =
      (64.0 * std::sqrt(static_cast<double>(n)) /
       static_cast<double>(1 << row_shift)) *
      (64.0 * std::sqrt(static_cast<double>(n)) /
       static_cast<double>(1 << col_shift));
  p.quant_factor = 8.0 * static_cast<double>(dq_denom) /
                   (inv_scale * static_cast<double>(kQuantStep));
  p.scale_w = 1.0 / (64.0 * std::sqrt(static_cast<double>(n)));
  p.scale_h = 1.0 / (64.0 * std::sqrt(static_cast<double>(n)));

  // Fixed-point row multiplier: row_mul ~= scale_w * 2^kQrow.
  const double row_mul_f = p.scale_w * static_cast<double>(1 << kQrow);
  p.row_mul = static_cast<std::int32_t>(std::llround(row_mul_f));
  const double row_eff =
      static_cast<double>(p.row_mul) /
      (static_cast<double>(1 << kQrow) * p.scale_w);  // 1 + delta

  // q = round(col_fixed * qf * scale_h / (2^kQrow * (1+delta)))
  const double quant_mul_f = p.quant_factor * p.scale_h /
                             (static_cast<double>(1 << kQrow) * row_eff) *
                             std::ldexp(1.0, kQshift);
  // ceil(), not round(): lround() rounds ties away from zero, and the
  // 16x16 coefficients are dyadic rationals that hit exact ties often.
  p.quant_mul = static_cast<std::int64_t>(std::ceil(quant_mul_f));
  const double quant_mul_naive_f = p.quant_factor * p.scale_h /
                                   static_cast<double>(1 << kQrow) *
                                   std::ldexp(1.0, kQshift);
  p.quant_mul_naive =
      static_cast<std::int64_t>(std::ceil(quant_mul_naive_f));
  return p;
}

void forward_dct_quant(const std::int16_t* block, int n,
                       const BlockParams& p, double* tmp, double* coeff,
                       std::int32_t* q) {
  const std::int16_t* k = dct_kernel(n);
  for (int i = 0; i < n; ++i) {
    for (int v = 0; v < n; ++v) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += static_cast<double>(k[v * n + j]) *
               static_cast<double>(block[static_cast<std::size_t>(i) * n + j]);
      }
      tmp[static_cast<std::size_t>(i) * n + v] = sum * p.scale_w;
    }
  }
  for (int u = 0; u < n; ++u) {
    for (int v = 0; v < n; ++v) {
      double sum = 0.0;
      for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(k[u * n + i]) *
               tmp[static_cast<std::size_t>(i) * n + v];
      }
      const double c = sum * p.scale_h;
      const std::size_t pos = static_cast<std::size_t>(u) * n + v;
      coeff[pos] = c;
      q[pos] = static_cast<std::int32_t>(std::lround(c * p.quant_factor));
    }
  }
}

void inverse_dct_from_q(const std::int32_t* q, int n, const BlockParams& p,
                        int* residual) {
  int row_shift = 0;
  int col_shift = 0;
  transform_shifts(n, n, &row_shift, &col_shift);
  const std::int16_t* k = dct_kernel(n);

  int dequant[32 * 32];
  for (int pos = 0; pos < n * n; ++pos) {
    const int qc = q[pos];
    if (qc == 0) {
      dequant[pos] = 0;
      continue;
    }
    const int dq_high = (std::abs(qc) * kQuantStep) & 0xFFFFFF;
    const int dq = round2(dq_high, kQuantTableBits);
    const int value = dq / p.dq_denom;
    dequant[pos] = qc < 0 ? -value : value;
  }

  int intermediate[32 * 32];
  for (int i = 0; i < n; ++i) {
    for (int oj = 0; oj < n; ++oj) {
      std::int64_t sum = 0;
      for (int j = 0; j < n; ++j) {
        sum += static_cast<std::int64_t>(k[j * n + oj]) *
               dequant[static_cast<std::size_t>(i) * n + j];
      }
      intermediate[static_cast<std::size_t>(i) * n + oj] = clip3(
          -(1 << 15), (1 << 15) - 1,
          round2(static_cast<int>(sum), row_shift));
    }
  }
  for (int oi = 0; oi < n; ++oi) {
    for (int j = 0; j < n; ++j) {
      std::int64_t sum = 0;
      for (int i = 0; i < n; ++i) {
        sum += static_cast<std::int64_t>(k[i * n + oi]) *
               intermediate[static_cast<std::size_t>(i) * n + j];
      }
      residual[static_cast<std::size_t>(oi) * n + j] =
          clip3(-256, 255, round2(static_cast<int>(sum), col_shift));
    }
  }
}

}  // namespace ref
