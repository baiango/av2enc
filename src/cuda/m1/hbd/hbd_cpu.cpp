// CPU reference for the 10-bit M1 transform path.  See hbd_cpu.h.
#include "cuda/m1/hbd/hbd_cpu.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "encoder/tx_variants/tx_tables.h"

namespace av2enc::hbd {
namespace {

constexpr int kQuantStep = txv::kQuantStep;          // 528
constexpr int kQuantTableBits = txv::kQuantTableBits;  // 3

int clip3_int(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

int round2_64(std::int64_t value, int n) {
  if (n <= 0) {
    return static_cast<int>(value);
  }
  return static_cast<int>((value + (std::int64_t{1} << (n - 1))) >> n);
}

}  // namespace

// ---------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------

int dc_pred_bias(int bit_depth) {
  return bit_depth > 0 && bit_depth <= 12 ? (1 << (bit_depth - 1)) : 0;
}

int clip1(int value, int bit_depth) {
  const int high = bit_depth > 0 ? (1 << bit_depth) - 1 : 0;
  return clip3_int(0, high, value);
}

int clip_dequant(std::int64_t value, int bit_depth) {
  if (bit_depth <= 0) {
    return 0;
  }
  const std::int64_t high = std::int64_t{1} << (7 + bit_depth);
  if (value < -high) {
    return static_cast<int>(-high);
  }
  if (value > high - 1) {
    return static_cast<int>(high - 1);
  }
  return static_cast<int>(value);
}

int clip_inverse_row(std::int64_t value, int bit_depth) {
  if (bit_depth <= 0) {
    return 0;
  }
  const std::int64_t high = std::int64_t{1} << (bit_depth + 7);
  if (value < -high) {
    return static_cast<int>(-high);
  }
  if (value > high - 1) {
    return static_cast<int>(high - 1);
  }
  return static_cast<int>(value);
}

int clip_inverse_col(std::int64_t value, int bit_depth) {
  if (bit_depth <= 0) {
    return 0;
  }
  const std::int64_t high = std::int64_t{1} << bit_depth;
  if (value < -high) {
    return static_cast<int>(-high);
  }
  if (value > high - 1) {
    return static_cast<int>(high - 1);
  }
  return static_cast<int>(value);
}

// ---------------------------------------------------------------------------
// Scan / EOB
// ---------------------------------------------------------------------------

int scan_pos(int c, int w, int h) {
  int x = 0;
  int y = 0;
  for (int n = 0; n < c; ++n) {
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int s = std::min(x, h - 1 - y);
      x -= s;
      y += s;
    }
  }
  return y * w + x;
}

void build_scan(int w, int h, std::uint16_t* scan) {
  if (scan == nullptr || w <= 0 || h <= 0) {
    return;
  }
  int x = 0;
  int y = 0;
  for (int c = 0; c < w * h; ++c) {
    scan[c] = static_cast<std::uint16_t>(y * w + x);
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int s = std::min(x, h - 1 - y);
      x -= s;
      y += s;
    }
  }
}

int eob_from_levels(const std::int32_t* levels, int w, int h) {
  if (levels == nullptr || w <= 0 || h <= 0) {
    return 0;
  }
  const int area = w * h;
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  build_scan(w, h, scan.data());
  for (int c = area - 1; c >= 0; --c) {
    if (levels[scan[static_cast<std::size_t>(c)]] != 0) {
      return c + 1;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Core stages
// ---------------------------------------------------------------------------

void downsample(const std::int16_t* residual_coded, const TxParams& p,
                std::int16_t* residual_adjusted) {
  txv::downsample_residual(residual_coded, p, residual_adjusted);
}

void forward(const std::int16_t* residual_adjusted, const TxParams& p,
             double* coeff, std::int32_t* levels) {
  txv::forward_transform(residual_adjusted, p, coeff, levels);
}

void dequantize(const std::int32_t* levels, const TxParams& p,
                std::int32_t* dequant, int bit_depth) {
  const int n = p.w * p.h;
  for (int i = 0; i < n; ++i) {
    const std::int32_t qc = levels[i];
    const std::int32_t abs_qc = qc < 0 ? -qc : qc;
    const std::int32_t dq_high = (abs_qc * kQuantStep) & 0xFFFFFF;
    const std::int32_t dq = txv::round2(dq_high, kQuantTableBits) /
                            static_cast<std::int32_t>(p.dq_denom);
    dequant[i] = clip_dequant(qc < 0 ? -dq : dq, bit_depth);
  }
}

void inverse_block(const std::int32_t* dequant, const TxParams& p,
                   std::int16_t* residual_coded, int bit_depth) {
  const int w = p.w;
  const int h = p.h;
  const txv::Tx1dType row_type = static_cast<txv::Tx1dType>(p.row_type);
  const txv::Tx1dType col_type = static_cast<txv::Tx1dType>(p.col_type);
  const std::int16_t* kw = txv::tx_kernel(row_type, w);
  const std::int16_t* kh = txv::tx_kernel(col_type, h);
  const bool odd_rect = ((p.log2_w ^ p.log2_h) & 1) != 0;
  const int identity_w = txv::identity_scale(p.log2_w);
  const int identity_h = txv::identity_scale(p.log2_h);

  auto prescale = [odd_rect](std::int32_t value) -> std::int32_t {
    return odd_rect
               ? round2_64(static_cast<std::int64_t>(value) * 2896, 12)
               : value;
  };

  std::int32_t intermediate[txv::kMaxAdjTxElems] = {};

  // Row pass: clip ±(1 << (BitDepth + 7)) (§7.15.2.1).
  for (int i = 0; i < h; ++i) {
    const int row = i * w;
    if (kw == nullptr) {
      for (int oj = 0; oj < w; ++oj) {
        const std::int64_t s =
            static_cast<std::int64_t>(prescale(dequant[row + oj])) *
            identity_w;
        intermediate[row + oj] =
            clip_inverse_row(round2_64(s, p.row_shift), bit_depth);
      }
    } else {
      for (int oj = 0; oj < w; ++oj) {
        std::int64_t sum = 0;
        for (int j = 0; j < w; ++j) {
          sum += static_cast<std::int64_t>(kw[j * w + oj]) *
                 prescale(dequant[row + j]);
        }
        intermediate[row + oj] =
            clip_inverse_row(round2_64(sum, p.row_shift), bit_depth);
      }
    }
  }

  // Column pass: clip ±(1 << BitDepth) (§7.15.2.1).
  for (int oi = 0; oi < h; ++oi) {
    const int out_row = oi * w;
    if (kh == nullptr) {
      for (int j = 0; j < w; ++j) {
        const std::int64_t s =
            static_cast<std::int64_t>(intermediate[out_row + j]) *
            identity_h;
        residual_coded[out_row + j] = static_cast<std::int16_t>(
            clip_inverse_col(round2_64(s, p.col_shift), bit_depth));
      }
    } else {
      for (int j = 0; j < w; ++j) {
        std::int64_t sum = 0;
        for (int i = 0; i < h; ++i) {
          sum += static_cast<std::int64_t>(kh[i * h + oi]) *
                 intermediate[i * w + j];
        }
        residual_coded[out_row + j] = static_cast<std::int16_t>(
            clip_inverse_col(round2_64(sum, p.col_shift), bit_depth));
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

// ---------------------------------------------------------------------------
// Whole-block reference
// ---------------------------------------------------------------------------

void reconstruct(const std::int16_t* residual, const std::uint16_t* pred,
                 int count, int bit_depth, std::uint16_t* recon) {
  for (int i = 0; i < count; ++i) {
    const int value = static_cast<int>(pred[i]) + residual[i];
    recon[i] = static_cast<std::uint16_t>(clip1(value, bit_depth));
  }
}

bool encode_block_residual(const std::int16_t* residual_coded,
                           const std::uint16_t* pred, const TxParams& p,
                           int bit_depth, BlockResult* out) {
  if (residual_coded == nullptr || pred == nullptr || out == nullptr) {
    return false;
  }
  const int area = p.w * p.h;
  const int coded_pels = p.coded_w * p.coded_h;
  if (area <= 0 || coded_pels <= 0) {
    return false;
  }

  BlockResult result;
  result.levels.assign(static_cast<std::size_t>(area), 0);
  result.dequant.assign(static_cast<std::size_t>(area), 0);
  result.residual.assign(static_cast<std::size_t>(coded_pels), 0);
  result.recon.assign(static_cast<std::size_t>(coded_pels), 0);

  std::vector<std::int16_t> adjusted(static_cast<std::size_t>(area));
  downsample(residual_coded, p, adjusted.data());

  std::vector<double> coeff(static_cast<std::size_t>(area));
  forward(adjusted.data(), p, coeff.data(), result.levels.data());
  result.eob = eob_from_levels(result.levels.data(), p.w, p.h);

  dequantize(result.levels.data(), p, result.dequant.data(), bit_depth);
  inverse_block(result.dequant.data(), p, result.residual.data(), bit_depth);
  reconstruct(result.residual.data(), pred, coded_pels, bit_depth,
              result.recon.data());

  *out = std::move(result);
  return true;
}

bool encode_block(const std::uint16_t* src, int src_stride, int src_width,
                  int src_height, int x0, int y0, const std::uint16_t* pred,
                  const TxParams& p, int bit_depth, BlockResult* out) {
  if (src == nullptr || pred == nullptr || out == nullptr || src_stride <= 0 ||
      src_width <= 0 || src_height <= 0) {
    return false;
  }
  const int cw = p.coded_w;
  const int ch = p.coded_h;
  const int coded_pels = cw * ch;
  if (coded_pels <= 0) {
    return false;
  }

  std::vector<std::int16_t> residual(static_cast<std::size_t>(coded_pels));
  for (int y = 0; y < ch; ++y) {
    const int py = std::clamp(y0 + y, 0, src_height - 1);
    for (int x = 0; x < cw; ++x) {
      const int px = std::clamp(x0 + x, 0, src_width - 1);
      const int s = static_cast<int>(
          src[static_cast<std::size_t>(py) *
                  static_cast<std::size_t>(src_stride) +
              static_cast<std::size_t>(px)]);
      const int pr = static_cast<int>(
          pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
               static_cast<std::size_t>(x)]);
      residual[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
               static_cast<std::size_t>(x)] =
          static_cast<std::int16_t>(s - pr);
    }
  }
  return encode_block_residual(residual.data(), pred, p, bit_depth, out);
}

}  // namespace av2enc::hbd
