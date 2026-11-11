// CPU reference for the AV2 transform variants.  See tx_variants.h for the
// numeric contract and the list of deliberate differences from the M0 layer.
#include "encoder/tx_variants/tx_variants.h"

#include <cmath>
#include <cstddef>

#include "encoder/tx_variants/tx_tables.h"

namespace av2enc::txv {
namespace {

struct TypePair {
  Tx1dType row;
  Tx1dType col;
};

// §7.15.4.1 Transform_1d_Type[TX_TYPES][2].  dir 0 (row) is horizontal, dir 1
// (col) is vertical.
constexpr TypePair kTransform1dType[16] = {
    {Tx1dType::kDct, Tx1dType::kDct},    // DCT_DCT
    {Tx1dType::kDct, Tx1dType::kAdst},   // ADST_DCT
    {Tx1dType::kAdst, Tx1dType::kDct},   // DCT_ADST
    {Tx1dType::kAdst, Tx1dType::kAdst},  // ADST_ADST
    {Tx1dType::kDct, Tx1dType::kFdst},   // FLIPADST_DCT
    {Tx1dType::kFdst, Tx1dType::kDct},   // DCT_FLIPADST
    {Tx1dType::kFdst, Tx1dType::kFdst},  // FLIPADST_FLIPADST
    {Tx1dType::kFdst, Tx1dType::kAdst},  // FLIPADST_ADST
    {Tx1dType::kAdst, Tx1dType::kFdst},  // ADST_FLIPADST
    {Tx1dType::kIdt, Tx1dType::kIdt},    // IDTX
    {Tx1dType::kIdt, Tx1dType::kDct},    // V_DCT
    {Tx1dType::kDct, Tx1dType::kIdt},    // H_DCT
    {Tx1dType::kIdt, Tx1dType::kAdst},   // V_ADST
    {Tx1dType::kAdst, Tx1dType::kIdt},   // H_ADST
    {Tx1dType::kIdt, Tx1dType::kFdst},   // V_FLIPADST
    {Tx1dType::kFdst, Tx1dType::kIdt},   // H_FLIPADST
};

constexpr const char* kTxTypeNames[16] = {
    "DCT_DCT",         "ADST_DCT",       "DCT_ADST",      "ADST_ADST",
    "FLIPADST_DCT",    "DCT_FLIPADST",   "FLIPADST_FLIPADST",
    "FLIPADST_ADST",   "ADST_FLIPADST",  "IDTX",
    "V_DCT",           "H_DCT",          "V_ADST",        "H_ADST",
    "V_FLIPADST",      "H_FLIPADST",
};

constexpr const char* kTxSizeNames[25] = {
    "TX_4X4",   "TX_8X8",   "TX_16X16", "TX_32X32", "TX_64X64",
    "TX_4X8",   "TX_8X4",   "TX_8X16",  "TX_16X8",  "TX_16X32",
    "TX_32X16", "TX_32X64", "TX_64X32", "TX_4X16",  "TX_16X4",
    "TX_8X32",  "TX_32X8",  "TX_16X64", "TX_64X16", "TX_4X32",
    "TX_32X4",  "TX_8X64",  "TX_64X8",  "TX_4X64",  "TX_64X4",
};

inline std::int32_t clip3(std::int32_t low, std::int32_t high,
                          std::int32_t value) {
  return value < low ? low : (value > high ? high : value);
}

inline int log2_int(int value) {
  int n = 0;
  while ((1 << n) < value) {
    ++n;
  }
  return n;
}

}  // namespace

const char* tx_type_name(TxType type) {
  const int idx = static_cast<int>(type);
  return idx >= 0 && idx < 16 ? kTxTypeNames[idx] : "?";
}

const char* tx_size_name(TxSize size) {
  const int idx = static_cast<int>(size);
  return idx >= 0 && idx < 25 ? kTxSizeNames[idx] : "?";
}

int tx_size_index(int coded_w, int coded_h) {
  for (int i = 0; i < 25; ++i) {
    if (kTxSizes[i].width == coded_w && kTxSizes[i].height == coded_h) {
      return i;
    }
  }
  return -1;
}

bool tx_size_supported(int coded_w, int coded_h) {
  return tx_size_index(coded_w, coded_h) >= 0;
}

void tx_1d_types(TxType type, Tx1dType* row, Tx1dType* col) {
  const int idx = static_cast<int>(type);
  const TypePair pair =
      idx >= 0 && idx < 16 ? kTransform1dType[idx]
                           : TypePair{Tx1dType::kDct, Tx1dType::kDct};
  *row = pair.row;
  *col = pair.col;
}

bool make_tx_type(Tx1dType row, Tx1dType col, TxType* out) {
  for (int i = 0; i < 16; ++i) {
    if (kTransform1dType[i].row == row && kTransform1dType[i].col == col) {
      *out = static_cast<TxType>(i);
      return true;
    }
  }
  return false;
}

const std::int16_t* tx_kernel(Tx1dType type, int n) {
  switch (type) {
    case Tx1dType::kIdt:
      return nullptr;
    case Tx1dType::kAdst:
      if (n == 4) return &kAdst4[0][0];
      if (n == 8) return &kAdst8[0][0];
      if (n == 16) return &kAdst16[0][0];
      return &kDct32[0][0];  // §7.15.2.1: sz == 32 uses Dct_Kernel32.
    case Tx1dType::kFdst:
      if (n == 4) return &kFdst4[0][0];
      if (n == 8) return &kFdst8[0][0];
      if (n == 16) return &kFdst16[0][0];
      return &kDct32[0][0];
    case Tx1dType::kDct:
    default:
      if (n == 4) return &kDct4[0][0];
      if (n == 8) return &kDct8[0][0];
      if (n == 16) return &kDct16[0][0];
      return &kDct32[0][0];
  }
}

std::optional<TxParams> make_tx_params(int coded_w, int coded_h,
                                       TxType type) {
  const int size_index = tx_size_index(coded_w, coded_h);
  if (size_index < 0) {
    return std::nullopt;
  }
  const TxSizeInfo& info = kTxSizes[size_index];
  TxParams p;
  p.tx_size = static_cast<std::int8_t>(size_index);
  p.tx_type = static_cast<std::int8_t>(type);
  Tx1dType row = Tx1dType::kDct;
  Tx1dType col = Tx1dType::kDct;
  tx_1d_types(type, &row, &col);
  p.row_type = static_cast<std::int8_t>(row);
  p.col_type = static_cast<std::int8_t>(col);
  p.coded_w = static_cast<std::int16_t>(coded_w);
  p.coded_h = static_cast<std::int16_t>(coded_h);
  p.w = info.adj_width;
  p.h = info.adj_height;
  p.log2_w = info.log2_w;
  p.log2_h = info.log2_h;
  p.row_shift = info.row_shift;
  p.col_shift = info.col_shift;
  const int pels = coded_w * coded_h;
  p.dq_denom =
      static_cast<std::uint8_t>(1 << ((pels > 256) + (pels > 1024)));
  p.scale_w = 1.0 / (64.0 * std::sqrt(static_cast<double>(p.w)));
  p.scale_h = 1.0 / (64.0 * std::sqrt(static_cast<double>(p.h)));
  const bool odd_rect = ((p.log2_w ^ p.log2_h) & 1) != 0;
  p.rect_prescale = odd_rect ? 4096.0 / 2896.0 : 1.0;
  const double inv_scale =
      (64.0 * std::sqrt(static_cast<double>(p.w)) /
       static_cast<double>(1 << p.row_shift)) *
      (64.0 * std::sqrt(static_cast<double>(p.h)) /
       static_cast<double>(1 << p.col_shift));
  p.quant_factor =
      8.0 * static_cast<double>(p.dq_denom) / (inv_scale * kQuantStep);
  return p;
}

std::optional<TxParams> make_tx_params(TxSize size, TxType type) {
  const int idx = static_cast<int>(size);
  if (idx < 0 || idx >= 25) {
    return std::nullopt;
  }
  return make_tx_params(kTxSizes[idx].width, kTxSizes[idx].height, type);
}

void adjusted_size(const TxParams& p, int* w, int* h) {
  *w = p.w;
  *h = p.h;
}

std::int32_t round2(std::int32_t value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

void forward_block(const std::int16_t* residual_adjusted, const TxParams& p,
                   double* coeff) {
  const int w = p.w;
  const int h = p.h;
  const Tx1dType row_type = static_cast<Tx1dType>(p.row_type);
  const Tx1dType col_type = static_cast<Tx1dType>(p.col_type);
  const std::int16_t* kw = tx_kernel(row_type, w);
  const std::int16_t* kh = tx_kernel(col_type, h);
  const double scale_w = p.scale_w;
  const double scale_h = p.scale_h;
  double tmp[kMaxAdjTxElems] = {};

  for (int i = 0; i < h; ++i) {
    const int row = i * w;
    if (kw == nullptr) {
      for (int v = 0; v < w; ++v) {
        tmp[row + v] = static_cast<double>(residual_adjusted[row + v]);
      }
    } else {
      for (int v = 0; v < w; ++v) {
        double sum = 0.0;
        for (int j = 0; j < w; ++j) {
          sum += static_cast<double>(kw[v * w + j]) *
                 static_cast<double>(residual_adjusted[row + j]);
        }
        tmp[row + v] = sum * scale_w;
      }
    }
  }

  for (int u = 0; u < h; ++u) {
    const int out_row = u * w;
    if (kh == nullptr) {
      for (int v = 0; v < w; ++v) {
        coeff[out_row + v] = tmp[out_row + v];
      }
    } else {
      for (int v = 0; v < w; ++v) {
        double sum = 0.0;
        for (int i = 0; i < h; ++i) {
          sum += static_cast<double>(kh[u * h + i]) * tmp[i * w + v];
        }
        coeff[out_row + v] = sum * scale_h;
      }
    }
  }

  if (p.rect_prescale != 1.0) {
    const int n = w * h;
    for (int i = 0; i < n; ++i) {
      coeff[i] *= p.rect_prescale;
    }
  }
}

void quantize(const double* coeff, const TxParams& p, std::int32_t* levels) {
  const int n = p.w * p.h;
  for (int i = 0; i < n; ++i) {
    levels[i] =
        static_cast<std::int32_t>(std::lround(coeff[i] * p.quant_factor));
  }
}

void dequantize(const std::int32_t* levels, const TxParams& p,
                std::int32_t* dequant) {
  const int n = p.w * p.h;
  for (int i = 0; i < n; ++i) {
    const std::int32_t qc = levels[i];
    const std::int32_t abs_qc = qc < 0 ? -qc : qc;
    const std::int32_t dq_high = (abs_qc * kQuantStep) & 0xFFFFFF;
    const std::int32_t dq = round2(dq_high, kQuantTableBits);
    const std::int32_t value = dq / p.dq_denom;
    // §7.14.4: Dequant = Clip3(-(1 << (7 + BitDepth)), ...).
    dequant[i] = clip3(-(1 << 15), (1 << 15) - 1,
                       qc < 0 ? -value : value);
  }
}

void inverse_block(const std::int32_t* dequant, const TxParams& p,
                   std::int16_t* residual_coded) {
  const int w = p.w;
  const int h = p.h;
  const Tx1dType row_type = static_cast<Tx1dType>(p.row_type);
  const Tx1dType col_type = static_cast<Tx1dType>(p.col_type);
  const std::int16_t* kw = tx_kernel(row_type, w);
  const std::int16_t* kh = tx_kernel(col_type, h);
  const bool odd_rect = ((p.log2_w ^ p.log2_h) & 1) != 0;
  // §7.15.4.1 pre-scale for odd log2 difference; Round2(., 12).
  auto prescale = [&](std::int32_t value) -> std::int32_t {
    return odd_rect ? round2(static_cast<std::int32_t>(
                                static_cast<std::int64_t>(value) * 2896),
                             12)
                    : value;
  };
  const int identity_w = identity_scale(p.log2_w);
  const int identity_h = identity_scale(p.log2_h);

  std::int32_t intermediate[kMaxAdjTxElems] = {};

  // Row pass.
  for (int i = 0; i < h; ++i) {
    const int row = i * w;
    if (kw == nullptr) {
      for (int oj = 0; oj < w; ++oj) {
        const std::int64_t s =
            static_cast<std::int64_t>(prescale(dequant[row + oj])) *
            identity_w;
        intermediate[row + oj] = clip3(
            -(1 << 15), (1 << 15) - 1,
            round2(static_cast<std::int32_t>(s), p.row_shift));
      }
    } else {
      for (int oj = 0; oj < w; ++oj) {
        std::int64_t sum = 0;
        for (int j = 0; j < w; ++j) {
          sum += static_cast<std::int64_t>(kw[j * w + oj]) *
                 prescale(dequant[row + j]);
        }
        intermediate[row + oj] = clip3(
            -(1 << 15), (1 << 15) - 1,
            round2(static_cast<std::int32_t>(sum), p.row_shift));
      }
    }
  }

  // Column pass.
  for (int oi = 0; oi < h; ++oi) {
    const int out_row = oi * w;
    if (kh == nullptr) {
      for (int j = 0; j < w; ++j) {
        const std::int64_t s =
            static_cast<std::int64_t>(intermediate[out_row + j]) *
            identity_h;
        residual_coded[out_row + j] = static_cast<std::int16_t>(
            clip3(-256, 255,
                  round2(static_cast<std::int32_t>(s), p.col_shift)));
      }
    } else {
      for (int j = 0; j < w; ++j) {
        std::int64_t sum = 0;
        for (int i = 0; i < h; ++i) {
          sum += static_cast<std::int64_t>(kh[i * h + oi]) *
                 intermediate[i * w + j];
        }
        residual_coded[out_row + j] = static_cast<std::int16_t>(
            clip3(-256, 255,
                  round2(static_cast<std::int32_t>(sum), p.col_shift)));
      }
    }
  }

  // §7.15.4 adjusted-size expansion by sample duplication.
  if (w != p.coded_w || h != p.coded_h) {
    const int sx = p.coded_w / w;
    const int sy = p.coded_h / h;
    for (int y = p.coded_h - 1; y >= 0; --y) {
      const int ay = y / sy;
      for (int x = p.coded_w - 1; x >= 0; --x) {
        const int ax = x / sx;
        residual_coded[y * p.coded_w + x] =
            residual_coded[ay * w + ax];
      }
    }
  }
}

void forward_transform(const std::int16_t* residual_adjusted, const TxParams& p,
                       double* coeff_scratch, std::int32_t* levels) {
  forward_block(residual_adjusted, p, coeff_scratch);
  quantize(coeff_scratch, p, levels);
}

void inverse_transform(const std::int32_t* levels, const TxParams& p,
                       std::int32_t* dequant_scratch,
                       std::int16_t* residual_coded) {
  dequantize(levels, p, dequant_scratch);
  inverse_block(dequant_scratch, p, residual_coded);
}

void downsample_residual(const std::int16_t* residual_coded, const TxParams& p,
                         std::int16_t* residual_adjusted) {
  const int sx = p.coded_w / p.w;
  const int sy = p.coded_h / p.h;
  const int shift = log2_int(sx * sy);
  for (int j = 0; j < p.h; ++j) {
    for (int i = 0; i < p.w; ++i) {
      std::int32_t sum = 0;
      for (int dy = 0; dy < sy; ++dy) {
        const int y = j * sy + dy;
        for (int dx = 0; dx < sx; ++dx) {
          const int x = i * sx + dx;
          sum += residual_coded[y * p.coded_w + x];
        }
      }
      residual_adjusted[j * p.w + i] =
          static_cast<std::int16_t>(round2(sum, shift));
    }
  }
}

void make_residual(const std::uint8_t* src, int src_stride, int src_width,
                   int src_height, int x0, int y0, int pred,
                   const TxParams& p, std::int16_t* residual_adjusted) {
  const int sx = p.coded_w / p.w;
  const int sy = p.coded_h / p.h;
  const int shift = log2_int(sx * sy);
  for (int j = 0; j < p.h; ++j) {
    for (int i = 0; i < p.w; ++i) {
      std::int32_t sum = 0;
      for (int dy = 0; dy < sy; ++dy) {
        const int yy = y0 + j * sy + dy;
        const int yc = yy < 0 ? 0 : (yy >= src_height ? src_height - 1 : yy);
        for (int dx = 0; dx < sx; ++dx) {
          const int xx = x0 + i * sx + dx;
          const int xc = xx < 0 ? 0 : (xx >= src_width ? src_width - 1 : xx);
          sum += static_cast<std::int32_t>(
                     src[yc * src_stride + xc]) -
                 pred;
        }
      }
      residual_adjusted[j * p.w + i] =
          static_cast<std::int16_t>(round2(sum, shift));
    }
  }
}

void upsample_residual(const std::int16_t* residual_adjusted, const TxParams& p,
                       std::int16_t* residual_coded) {
  const int sx = p.coded_w / p.w;
  const int sy = p.coded_h / p.h;
  if (sx == 1 && sy == 1) {
    const int n = p.coded_w * p.coded_h;
    for (int i = 0; i < n; ++i) {
      residual_coded[i] = residual_adjusted[i];
    }
    return;
  }
  for (int y = 0; y < p.coded_h; ++y) {
    const int ay = y / sy;
    for (int x = 0; x < p.coded_w; ++x) {
      const int ax = x / sx;
      residual_coded[y * p.coded_w + x] =
          residual_adjusted[ay * p.w + ax];
    }
  }
}

void reconstruct_block(const std::int16_t* residual_adjusted, int pred,
                       const TxParams& p, std::uint8_t* dst, int dst_stride,
                       int dst_width, int dst_height, int x0, int y0) {
  const int sx = p.coded_w / p.w;
  const int sy = p.coded_h / p.h;
  for (int y = 0; y < p.coded_h; ++y) {
    const int yy = y0 + y;
    if (yy < 0 || yy >= dst_height) {
      continue;
    }
    const int ay = y / sy;
    for (int x = 0; x < p.coded_w; ++x) {
      const int xx = x0 + x;
      if (xx < 0 || xx >= dst_width) {
        continue;
      }
      const int ax = x / sx;
      const int value = pred + residual_adjusted[ay * p.w + ax];
      dst[yy * dst_stride + xx] =
          static_cast<std::uint8_t>(clip3(0, 255, value));
    }
  }
}

const char* tx_partition_name(TxPartition part) {
  switch (part) {
    case TxPartition::kNone:
      return "NONE";
    case TxPartition::kSplit:
      return "SPLIT";
    case TxPartition::kHorz:
      return "HORZ";
    case TxPartition::kVert:
      return "VERT";
  }
  return "?";
}

int tx_partition_parts(TxPartition part) {
  switch (part) {
    case TxPartition::kNone:
      return 1;
    case TxPartition::kHorz:
    case TxPartition::kVert:
      return 2;
    case TxPartition::kSplit:
      return 4;
  }
  return 0;
}

int tx_partition_geometry(int coded_w, int coded_h, TxPartition part, int* xs,
                          int* ys, int* ws, int* hs) {
  if (coded_w < 4 || coded_h < 4 || coded_w > 64 || coded_h > 64) {
    return 0;
  }
  switch (part) {
    case TxPartition::kNone:
      xs[0] = 0;
      ys[0] = 0;
      ws[0] = coded_w;
      hs[0] = coded_h;
      return 1;
    case TxPartition::kHorz: {
      if (coded_h < 8 || (coded_h & 1) != 0) {
        return 0;
      }
      const int hh = coded_h / 2;
      xs[0] = 0;
      ys[0] = 0;
      ws[0] = coded_w;
      hs[0] = hh;
      xs[1] = 0;
      ys[1] = hh;
      ws[1] = coded_w;
      hs[1] = hh;
      return 2;
    }
    case TxPartition::kVert: {
      if (coded_w < 8 || (coded_w & 1) != 0) {
        return 0;
      }
      const int hw = coded_w / 2;
      xs[0] = 0;
      ys[0] = 0;
      ws[0] = hw;
      hs[0] = coded_h;
      xs[1] = hw;
      ys[1] = 0;
      ws[1] = hw;
      hs[1] = coded_h;
      return 2;
    }
    case TxPartition::kSplit: {
      if (coded_w < 8 || coded_h < 8 || (coded_w & 1) != 0 ||
          (coded_h & 1) != 0) {
        return 0;
      }
      const int hw = coded_w / 2;
      const int hh = coded_h / 2;
      xs[0] = 0;
      ys[0] = 0;
      ws[0] = hw;
      hs[0] = hh;
      xs[1] = hw;
      ys[1] = 0;
      ws[1] = hw;
      hs[1] = hh;
      xs[2] = 0;
      ys[2] = hh;
      ws[2] = hw;
      hs[2] = hh;
      xs[3] = hw;
      ys[3] = hh;
      ws[3] = hw;
      hs[3] = hh;
      return 4;
    }
  }
  return 0;
}

bool tx_partition_legal(int coded_w, int coded_h, TxPartition part) {
  int xs[4];
  int ys[4];
  int ws[4];
  int hs[4];
  return tx_partition_geometry(coded_w, coded_h, part, xs, ys, ws, hs) > 0;
}

int tx_partition_candidates(int coded_w, int coded_h, TxPartition* out,
                            int cap) {
  static constexpr TxPartition kOrder[4] = {
      TxPartition::kNone, TxPartition::kSplit, TxPartition::kHorz,
      TxPartition::kVert};
  int count = 0;
  for (TxPartition part : kOrder) {
    if (count >= cap) {
      break;
    }
    if (tx_partition_legal(coded_w, coded_h, part)) {
      out[count++] = part;
    }
  }
  return count;
}

}  // namespace av2enc::txv
