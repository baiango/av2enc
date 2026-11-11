#include "encoder/format/chroma/chroma_residual.h"

#include "encoder/m0_tables.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace av2enc::chroma {
namespace {

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

const std::int16_t* dct_kernel(int n) {
  switch (n) {
    case 4:
      return &m0::kDct4[0][0];
    case 8:
      return &m0::kDct8[0][0];
    case 16:
      return &m0::kDct16[0][0];
    default:
      return &m0::kDct32[0][0];
  }
}

constexpr int kDivLutBits = 7;
constexpr int kDivLutPrecBits = 9;
constexpr int kDivLutNum = 129;
constexpr int kDivLut[kDivLutNum] = {
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465, 462,
    458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 426, 423, 420, 417,
    415, 412, 410, 407, 405, 402, 400, 397, 395, 392, 390, 388, 386, 383, 381,
    379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358, 356, 354, 352, 350,
    349, 347, 345, 343, 341, 340, 338, 336, 334, 333, 331, 329, 328, 326, 324,
    323, 321, 320, 318, 317, 315, 314, 312, 311, 309, 308, 306, 305, 303, 302,
    301, 299, 298, 297, 295, 294, 293, 291, 290, 289, 287, 286, 285, 284, 282,
    281, 280, 279, 278, 277, 275, 274, 273, 272, 271, 270, 269, 267, 266, 265,
    264, 263, 262, 261, 260, 259, 258, 257, 256};

}  // namespace

bool make_tx_shape(int nominal_width, int nominal_height, TxShape& shape) {
  if (nominal_width < 4 || nominal_height < 4 || nominal_width > 64 ||
      nominal_height > 64) {
    return false;
  }
  // All AV2 TX dimensions are powers of two; reject anything else so the
  // duplication factor stays 1 or 2.
  if ((nominal_width & (nominal_width - 1)) != 0 ||
      (nominal_height & (nominal_height - 1)) != 0) {
    return false;
  }
  shape = TxShape{};
  shape.nominal_w = nominal_width;
  shape.nominal_h = nominal_height;
  shape.adj_w = std::min(nominal_width, 32);
  shape.adj_h = std::min(nominal_height, 32);
  shape.upscale_x = nominal_width / shape.adj_w;
  shape.upscale_y = nominal_height / shape.adj_h;
  shape.tx_size_ctx = tx_size_context(nominal_width, nominal_height);
  shape.eob_multisize =
      std::min(5, floor_log2(nominal_width)) +
      std::min(5, floor_log2(nominal_height)) - 4;
  shape.seg_eob = shape.adj_w * shape.adj_h;
  shape.tx_pels = nominal_width * nominal_height;
  shape.dq_denom = 1 << ((shape.tx_pels > 256) + (shape.tx_pels > 1024));
  transform_shifts(nominal_width, nominal_height, &shape.row_shift,
                   &shape.col_shift);
  shape.odd_rect = odd_rect_transform(nominal_width, nominal_height);
  return true;
}

int tx_size_context(int nominal_width, int nominal_height) {
  // Tx_Size_Sqr / Tx_Size_Sqr_Up are enum indices: TX_4X4 = 0, TX_8X8 = 1,
  // TX_16X16 = 2, TX_32X32 = 3, TX_64X64 = 4.  They are not raw log2 values.
  const int sqr = std::max(0, floor_log2(nominal_width) - 2);
  const int sqr_up = std::max(0, floor_log2(nominal_height) - 2);
  const int lo = std::min(sqr, sqr_up);
  const int hi = std::max(sqr, sqr_up);
  return (lo + hi + 1) >> 1;
}

bool odd_rect_transform(int width, int height) {
  return ((floor_log2(width) ^ floor_log2(height)) & 1) != 0;
}

int round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

int approx_divide(int numerator, int denominator) {
  const int abs_denominator = std::abs(denominator);
  if (abs_denominator == 0) {
    return 0;
  }
  const int n = floor_log2(abs_denominator);
  const int e = abs_denominator - (1 << n);
  const int f =
      n > kDivLutBits ? round2(e, n - kDivLutBits)
                      : e << (kDivLutBits - n);
  int scale = kDivLut[std::min(f, kDivLutNum - 1)];
  if (denominator < 0) {
    scale = -scale;
  }
  return round2(numerator * scale, n + kDivLutPrecBits);
}

void transform_shifts(int nominal_width, int nominal_height, int* row_shift,
                      int* col_shift) {
  // Transform_Shift[TX_SIZES_ALL][2] (§7.15.4.1).  Keyed by the nominal
  // width/height pair; the adjusted square must not be used here.
  const bool square = nominal_width == nominal_height;
  if (nominal_width == 4 && nominal_height == 4) {
    *row_shift = 7; *col_shift = 10;
  } else if (nominal_width == 8 && nominal_height == 8) {
    *row_shift = 7; *col_shift = 11;
  } else if (nominal_width == 16 && nominal_height == 16) {
    *row_shift = 6; *col_shift = 13;
  } else if (nominal_width == 32 && nominal_height == 32) {
    *row_shift = 6; *col_shift = 13;
  } else if (nominal_width == 64 && nominal_height == 64) {
    *row_shift = 6; *col_shift = 13;
  } else if (nominal_width == 4 && (nominal_height == 8 || nominal_height == 4)) {
    *row_shift = 7; *col_shift = 10;
  } else if (nominal_height == 4 && nominal_width == 8) {
    *row_shift = 7; *col_shift = 10;
  } else if ((nominal_width == 8 && nominal_height == 16) ||
             (nominal_width == 16 && nominal_height == 8)) {
    *row_shift = 7; *col_shift = 11;
  } else if ((nominal_width == 16 && nominal_height == 32) ||
             (nominal_width == 32 && nominal_height == 16)) {
    *row_shift = 6; *col_shift = 12;
  } else if ((nominal_width == 32 && nominal_height == 64) ||
             (nominal_width == 64 && nominal_height == 32)) {
    *row_shift = 6; *col_shift = 12;
  } else if ((nominal_width == 4 && nominal_height == 16) ||
             (nominal_width == 16 && nominal_height == 4)) {
    *row_shift = 6; *col_shift = 12;
  } else if ((nominal_width == 8 && nominal_height == 32) ||
             (nominal_width == 32 && nominal_height == 8)) {
    *row_shift = 6; *col_shift = 13;
  } else if ((nominal_width == 16 && nominal_height == 64) ||
             (nominal_width == 64 && nominal_height == 16)) {
    *row_shift = 6; *col_shift = 13;
  } else if ((nominal_width == 4 && nominal_height == 32) ||
             (nominal_width == 32 && nominal_height == 4)) {
    *row_shift = 7; *col_shift = 11;
  } else if ((nominal_width == 8 && nominal_height == 64) ||
             (nominal_width == 64 && nominal_height == 8)) {
    *row_shift = 6; *col_shift = 12;
  } else if ((nominal_width == 4 && nominal_height == 64) ||
             (nominal_width == 64 && nominal_height == 4)) {
    *row_shift = 6; *col_shift = 13;
  } else if (square) {
    *row_shift = 6; *col_shift = 13;
  } else {
    *row_shift = 6; *col_shift = 13;
  }
}

void forward_dct(const std::int16_t* residual, int adj_width, int adj_height,
                 bool odd_rect, double* coeff) {
  const int width = adj_width;
  const int height = adj_height;
  const std::int16_t* kw = dct_kernel(width);
  const std::int16_t* kh = dct_kernel(height);
  const double scale_w = 1.0 / (64.0 * std::sqrt(static_cast<double>(width)));
  const double scale_h = 1.0 / (64.0 * std::sqrt(static_cast<double>(height)));
  std::vector<double> tmp(static_cast<std::size_t>(width) *
                              static_cast<std::size_t>(height),
                          0.0);
  for (int i = 0; i < height; ++i) {
    for (int v = 0; v < width; ++v) {
      double sum = 0.0;
      for (int j = 0; j < width; ++j) {
        sum += static_cast<double>(kw[v * width + j]) *
               static_cast<double>(
                   residual[static_cast<std::size_t>(i) * width + j]);
      }
      tmp[static_cast<std::size_t>(i) * width + v] = sum * scale_w;
    }
  }
  for (int u = 0; u < height; ++u) {
    for (int v = 0; v < width; ++v) {
      double sum = 0.0;
      for (int i = 0; i < height; ++i) {
        sum += static_cast<double>(kh[u * height + i]) *
               tmp[static_cast<std::size_t>(i) * width + v];
      }
      coeff[static_cast<std::size_t>(u) * width + v] = sum * scale_h;
    }
  }
  if (odd_rect) {
    constexpr double kRectPrescale = 4096.0 / 2896.0;
    const std::size_t count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    for (std::size_t i = 0; i < count; ++i) {
      coeff[i] *= kRectPrescale;
    }
  }
}

void inverse_dct(const int* dequant, const TxShape& shape, int* residual) {
  const int width = shape.adj_w;
  const int height = shape.adj_h;
  const int row_shift = shape.row_shift;
  const int col_shift = shape.col_shift;
  const std::int16_t* kw = dct_kernel(width);
  const std::int16_t* kh = dct_kernel(height);
  const bool odd_rect = shape.odd_rect;
  const auto prescale = [odd_rect](int value) {
    return odd_rect ? round2(static_cast<int>(
                                static_cast<std::int64_t>(value) * 2896),
                            12)
                    : value;
  };
  std::vector<int> intermediate(static_cast<std::size_t>(width) *
                                    static_cast<std::size_t>(height),
                                0);
  for (int i = 0; i < height; ++i) {
    for (int oj = 0; oj < width; ++oj) {
      std::int64_t sum = 0;
      for (int j = 0; j < width; ++j) {
        sum += static_cast<std::int64_t>(kw[j * width + oj]) *
               prescale(dequant[static_cast<std::size_t>(i) * width + j]);
      }
      intermediate[static_cast<std::size_t>(i) * width + oj] = clip3(
          -(1 << 15), (1 << 15) - 1,
          round2(static_cast<int>(sum), row_shift));
    }
  }
  for (int oi = 0; oi < height; ++oi) {
    for (int j = 0; j < width; ++j) {
      std::int64_t sum = 0;
      for (int i = 0; i < height; ++i) {
        sum += static_cast<std::int64_t>(kh[i * height + oi]) *
               intermediate[static_cast<std::size_t>(i) * width + j];
      }
      residual[static_cast<std::size_t>(oi) * width + j] =
          clip3(-256, 255, round2(static_cast<int>(sum), col_shift));
    }
  }
}

void build_diagonal_scan(int width, int height,
                         std::vector<std::uint16_t>& scan) {
  scan.resize(static_cast<std::size_t>(width) *
              static_cast<std::size_t>(height));
  int x = 0;
  int y = 0;
  for (int c = 0; c < width * height; ++c) {
    scan[static_cast<std::size_t>(c)] =
        static_cast<std::uint16_t>(y * width + x);
    ++x;
    --y;
    if (y < 0 || x >= width) {
      ++x;
      const int s = std::min(x, height - 1 - y);
      x -= s;
      y += s;
    }
  }
}

void quantize_coefficients(const double* coeff, int count, double quant_factor,
                           int* levels) {
  for (int i = 0; i < count; ++i) {
    levels[i] = static_cast<int>(std::lround(coeff[i] * quant_factor));
  }
}

int dequantize_level(int level, int dq_denom) {
  if (level == 0) {
    return 0;
  }
  const int dq_high = (std::abs(level) * kQuantStep) & 0xFFFFFF;
  const int dq = round2(dq_high, kQuantTableBits);
  const int value = dq / dq_denom;
  return level < 0 ? -value : value;
}

void dequantize_block(const int* levels, int count, int dq_denom,
                      int* dequant) {
  for (int i = 0; i < count; ++i) {
    dequant[i] = dequantize_level(levels[i], dq_denom);
  }
}

double quant_factor_for(const TxShape& shape) {
  const double inv_scale =
      (64.0 * std::sqrt(static_cast<double>(shape.adj_w)) /
       static_cast<double>(1 << shape.row_shift)) *
      (64.0 * std::sqrt(static_cast<double>(shape.adj_h)) /
       static_cast<double>(1 << shape.col_shift));
  return 8.0 * static_cast<double>(shape.dq_denom) /
         (inv_scale * static_cast<double>(kQuantStep));
}

}  // namespace av2enc::chroma
