#include "transform_stream.h"

#include "pipeline/still/still_tables.h"
#include "codec/entropy/qctx/qctx.h"
#include "codec/rate/rdoq/rdoq_cdf_tables.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace av2enc::still::rdo_decide {
namespace {

constexpr double kCdfScale = 32768.0;
constexpr double kImpossibleBits = 1.0e6;
constexpr int kMaxSymbols = 8;

std::int32_t abs_i32(std::int32_t value) noexcept {
  return value < 0 ? -value : value;
}

std::int32_t floor_log2(std::uint32_t value) noexcept {
  std::int32_t result = 0;
  while (value > 1U) {
    value >>= 1U;
    ++result;
  }
  return result;
}

std::int32_t clip3(std::int32_t low, std::int32_t high,
                   std::int32_t value) noexcept {
  return value < low ? low : (value > high ? high : value);
}

// Ideal code length of one symbol under a static inverse CDF.  `n` is the
// symbol count; the array has n+1 entries.
double symbol_bits(const std::uint16_t* cdf, std::int32_t n,
                   std::int32_t symbol) {
  if (n <= 1) {
    return 0.0;
  }
  if (cdf == nullptr || n > kMaxSymbols || symbol < 0 || symbol >= n) {
    return kImpossibleBits;
  }
  std::int32_t probability = 0;
  if (symbol == 0) {
    probability = cdf[0];
  } else if (symbol == n - 1) {
    probability = static_cast<std::int32_t>(kCdfScale) - cdf[n - 2];
  } else {
    probability =
        static_cast<std::int32_t>(cdf[symbol]) - cdf[symbol - 1];
  }
  if (probability <= 0) {
    return kImpossibleBits;
  }
  return -std::log2(static_cast<double>(probability) / kCdfScale);
}

// Bits for the §5.20.7.28 read_quant escape (raw q-length + golomb + literal).
double escape_bits(std::int32_t x, std::int32_t predictor) {
  if (x < 0) {
    return kImpossibleBits;
  }
  const std::int32_t m = clip3(
      1, 6, predictor <= 0
                 ? 0
                 : floor_log2(static_cast<std::uint32_t>(predictor)));
  const std::int32_t k = m + 1;
  const std::int32_t c_max = std::min(m + 4, 6);
  double bits = 0.0;
  if (x < (c_max << m)) {
    const std::int32_t q = x >> m;
    bits += static_cast<double>(q + 1);
    bits += static_cast<double>(m);
    return bits;
  }
  bits += static_cast<double>(c_max);
  std::int32_t g = 0;
  for (;;) {
    const std::int32_t length = k + g;
    const std::int32_t x_base =
        (c_max << m) + (1 << length) - (1 << k);
    if (x < x_base + (1 << length)) {
      bits += static_cast<double>(g + 1 + length);
      return bits;
    }
    ++g;
    if (g > 30) {
      return kImpossibleBits;
    }
  }
}

std::int32_t pre_level_from_abs(std::int32_t abs_value,
                                std::int32_t base_levels,
                                std::int32_t base_max,
                                std::int32_t pre_max) noexcept {
  if (abs_value <= base_levels) {
    return abs_value;
  }
  const std::int32_t base = std::min(base_levels + 1, base_max);
  const std::int32_t pre = std::min(abs_value, pre_max);
  return base + (pre - base);
}

void build_diagonal_scan(int w, int h, std::int32_t* scan) noexcept {
  std::int32_t x = 0;
  std::int32_t y = 0;
  const std::int32_t count = w * h;
  for (std::int32_t c = 0; c < count; ++c) {
    scan[static_cast<std::size_t>(c)] = y * w + x;
    x += 1;
    y -= 1;
    if (y < 0 || x >= w) {
      x += 1;
      const std::int32_t s = std::min(x, h - 1 - y);
      x -= s;
      y += s;
    }
  }
}

}  // namespace

bool TransformStreamRate::init(const TransformStreamBlock& block,
                               std::string* error) {
  valid_ = false;
  if (error != nullptr) {
    error->clear();
  }
  if (block.plane < 0 || block.plane > 2) {
    if (error != nullptr) *error = "transform stream plane must be 0, 1 or 2";
    return false;
  }
  if (block.tx_size < 0 || block.tx_size >= 25) {
    if (error != nullptr) *error = "transform stream tx_size must be 0..24";
    return false;
  }
  if (block.w <= 0 || block.h <= 0 ||
      block.w * block.h > kMaxTransformArea) {
    if (error != nullptr) {
      *error = "transform stream geometry must be a non-empty <= 32x32 block";
    }
    return false;
  }
  const int area = block.w * block.h;
  if (block.levels_scan.size() != static_cast<std::size_t>(area)) {
    if (error != nullptr) {
      *error = "transform stream level span must equal w*h";
    }
    return false;
  }
  if (block.eob < 0 || block.eob > area) {
    if (error != nullptr) *error = "transform stream eob must be in [0, w*h]";
    return false;
  }
  const int skip_limit = block.plane == 2 ? 12 : 10;
  if (block.txb_skip_context < 0 ||
      block.txb_skip_context >= skip_limit) {
    if (error != nullptr) *error = "txb_skip context out of range";
    return false;
  }
  if (block.dc_sign_context < 0 || block.dc_sign_context > 2) {
    if (error != nullptr) *error = "DC sign context must be 0..2";
    return false;
  }

  plane_ = block.plane;
  tx_size_ = block.tx_size;
  w_ = block.w;
  h_ = block.h;
  area_ = area;
  eob_ = block.eob;
  txb_skip_context_ = block.txb_skip_context;
  dc_sign_context_ = block.dc_sign_context;
  qctx_ = block.qctx < 0 ? 0
                         : (block.qctx >= av2enc::still::qctx::kSliceCount
                                ? av2enc::still::qctx::kSliceCount - 1
                                : block.qctx);
  // The coded size only enters eob_pt's multisize, and min(5, log2(.)) is the
  // same for an adjusted 32-point dimension as for its 64-point coded source.
  coded_w_ = block.w;
  coded_h_ = block.h;

  for (int c = 0; c < area; ++c) {
    levels_[static_cast<std::size_t>(c)] =
        block.levels_scan[static_cast<std::size_t>(c)];
  }
  // Freeze all contexts from the scalar path.  Positions above the declared
  // EOB stay in the level array but are not coded by block_bits().
  build_diagonal_scan(w_, h_, scan_.data());
  for (int c = 0; c < area; ++c) {
    const std::int32_t pos = scan_[static_cast<std::size_t>(c)];
    const std::int32_t row = pos / w_;
    const std::int32_t col = pos % w_;
    const bool lf = (plane_ == 0 ? (row + col) < 4 : (row + col) < 1);
    const std::int32_t base_levels = lf ? 4 : 2;
    const std::int32_t base_max = lf ? 5 : 3;
    const std::int32_t br_max = (lf && plane_ > 0) ? 0 : 3;
    const std::int32_t pre_max = base_max + br_max;
    const std::int32_t abs_value =
        abs_i32(levels_[static_cast<std::size_t>(c)]);
    pre_levels_[static_cast<std::size_t>(pos)] =
        pre_level_from_abs(abs_value, base_levels, base_max, pre_max);
  }

  static constexpr int kSigRef[5][2] = {
      {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
  static constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

  for (int c = 0; c < area; ++c) {
    Position& p = positions_[static_cast<std::size_t>(c)];
    p.pos = scan_[static_cast<std::size_t>(c)];
    const std::int32_t row = p.pos / w_;
    const std::int32_t col = p.pos % w_;
    const bool lf = (plane_ == 0 ? (row + col) < 4 : (row + col) < 1);
    p.is_lf = lf ? 1 : 0;
    p.base_levels = lf ? 4 : 2;
    p.base_max = lf ? 5 : 3;
    const std::int32_t br_max = (lf && plane_ > 0) ? 0 : 3;
    p.pre_max = p.base_max + br_max;
    p.max_level = lf ? (plane_ == 0 ? 8 : 5) : 6;
    p.is_dc = (plane_ == 0 && p.pos == 0) ? 1 : 0;

    if (plane_ == 0) {
      const std::int32_t limit = lf ? 5 : 3;
      std::int32_t mag = 0;
      for (int k = 0; k < 5; ++k) {
        const std::int32_t rr = row + kSigRef[k][0];
        const std::int32_t cc = col + kSigRef[k][1];
        if (rr < h_ && cc < w_) {
          mag += std::min(
              pre_levels_[static_cast<std::size_t>(rr * w_ + cc)], limit);
        }
      }
      const std::int32_t mctx = (mag + 1) >> 1;
      if (lf) {
        if (c == 0) {
          p.base_ctx = std::min(mctx, 8);
        } else if (row + col < 2) {
          p.base_ctx = std::min(mctx, 6) + 9;
        } else {
          p.base_ctx = std::min(mctx, 4) + 16;
        }
      } else {
        const std::int32_t ctx2 = std::min(mctx, 4);
        p.base_ctx = ctx2;
        if (row + col >= 8) {
          p.base_ctx += 10;
        } else if (row + col >= 6) {
          p.base_ctx += 5;
        }
      }
    } else {
      const std::int32_t limit = lf ? 5 : 3;
      std::int32_t mag = 0;
      for (int k = 0; k < 3; ++k) {
        const std::int32_t rr = row + kSigRef[k][0];
        const std::int32_t cc = col + kSigRef[k][1];
        if (rr < h_ && cc < w_) {
          mag += std::min(
              pre_levels_[static_cast<std::size_t>(rr * w_ + cc)], limit);
        }
      }
      const std::int32_t mctx = (mag + 1) >> 1;
      const std::int32_t ctx2 = std::min(mctx, 3);
      p.base_ctx = plane_ == 1 ? ctx2 : ctx2 + 4;
    }

    if (c == 0) {
      p.eob_ctx = 0;
    } else if (c <= area_ / 8) {
      p.eob_ctx = 1;
    } else if (c <= area_ / 4) {
      p.eob_ctx = 2;
    } else {
      p.eob_ctx = 3;
    }

    std::int32_t br_mag = 0;
    for (int k = 0; k < 3; ++k) {
      const std::int32_t rr = row + kMagRef[k][0];
      const std::int32_t cc = col + kMagRef[k][1];
      if (rr < h_ && cc < w_) {
        br_mag += std::min(
            pre_levels_[static_cast<std::size_t>(rr * w_ + cc)], 5);
      }
    }
    br_mag = std::min((br_mag + 1) >> 1, 6);
    if (plane_ == 0) {
      p.br_ctx = lf ? br_mag + (p.pos != 0 ? 7 : 0) : br_mag;
    } else {
      p.br_ctx = std::min(br_mag, 3);
    }
  }

  std::int32_t hr_level_avg = 0;
  for (int c = area; c-- > 0;) {
    Position& p = positions_[static_cast<std::size_t>(c)];
    p.hr_pred = static_cast<double>(hr_level_avg);
    const std::int32_t abs_value =
        abs_i32(levels_[static_cast<std::size_t>(c)]);
    if (abs_value >= p.max_level) {
      const std::int32_t pre = pre_level_from_abs(
          abs_value, p.base_levels, p.base_max, p.pre_max);
      const std::int32_t x = abs_value - pre;
      hr_level_avg = (x + hr_level_avg) >> 1;
    }
  }

  // Validate against the CDF table dimensions.
  for (int c = 0; c < area; ++c) {
    const Position& p = positions_[static_cast<std::size_t>(c)];
    if (plane_ == 0) {
      if (p.is_lf != 0 ? p.base_ctx > 32 : p.base_ctx > 19) {
        if (error != nullptr) *error = "coeff_base context out of range";
        return false;
      }
      if (p.is_lf != 0 ? p.br_ctx > 13 : p.br_ctx > 6) {
        if (error != nullptr) *error = "coeff_br context out of range";
        return false;
      }
    } else if (p.base_ctx > 11) {
      if (error != nullptr) *error = "chroma coeff_base context out of range";
      return false;
    }
  }
  valid_ = true;
  return true;
}

const std::uint16_t* TransformStreamRate::txb_skip_cdf() const noexcept {
  const av2enc::still::qctx::CoeffCdfSlice& q = av2enc::still::qctx::tables_for_slice(qctx_);
  if (plane_ == 2) {
    return &(*q.v_txb_skip)[txb_skip_context_][0];
  }
  const int tsc =
      (av2enc::still::kTxSizeSqr[tx_size_] +
       av2enc::still::kTxSizeSqrUp[tx_size_] + 1) >> 1;
  return &(*q.txb_skip)[0][tsc][txb_skip_context_][0];
}

double TransformStreamRate::coefficient_bits(int scan_pos,
                                             std::int32_t signed_level,
                                             bool is_eob) const noexcept {
  const av2enc::still::qctx::CoeffCdfSlice& q = av2enc::still::qctx::tables_for_slice(qctx_);
  if (!valid_ || scan_pos < 0 || scan_pos >= area_) {
    return kImpossibleBits;
  }
  if (coefficient_override_ != nullptr) {
    const double bits = coefficient_override_(
        override_context_, scan_pos, signed_level, is_eob);
    return std::isfinite(bits) && bits >= 0.0 ? bits : kImpossibleBits;
  }
  const Position& p = positions_[static_cast<std::size_t>(scan_pos)];
  const int tsc =
      (av2enc::still::kTxSizeSqr[tx_size_] +
       av2enc::still::kTxSizeSqrUp[tx_size_] + 1) >> 1;
  const std::int32_t abs_value = abs_i32(signed_level);
  if (abs_value == 0) {
    if (is_eob) {
      return kImpossibleBits;
    }
    double bits = 0.0;
    if (p.is_lf != 0) {
      if (plane_ == 0) {
        bits += symbol_bits(&(*q.coeff_base_lf)[tsc][p.base_ctx][0][0], 6, 0);
      } else {
        bits += symbol_bits(&(*q.coeff_base_lf_uv)[p.base_ctx][0], 6, 0);
      }
    } else {
      if (plane_ == 0) {
        bits += symbol_bits(&(*q.coeff_base)[tsc][p.base_ctx][0][0], 4, 0);
      } else {
        bits += symbol_bits(&(*q.coeff_base_uv)[p.base_ctx][0], 4, 0);
      }
    }
    return bits;
  }

  const std::int32_t base_levels = p.base_levels;
  const std::int32_t base_max = p.base_max;
  std::int32_t base_level = 0;
  std::int32_t br_level = 0;
  if (abs_value <= base_levels) {
    base_level = abs_value;
  } else {
    base_level = std::min(base_levels + 1, base_max);
    const std::int32_t pre = std::min(abs_value, p.pre_max);
    br_level = pre - base_level;
  }
  const std::int32_t pre_level = base_level + br_level;

  double bits = 0.0;
  if (is_eob) {
    const std::int32_t symbol = base_level - 1;
    if (plane_ == 0) {
      if (p.is_lf != 0) {
        bits += symbol_bits(&(*q.coeff_base_lf_eob)[tsc][p.eob_ctx][0], 5,
                            symbol);
      } else {
        bits += symbol_bits(&(*q.coeff_base_eob)[tsc][p.eob_ctx][0], 3, symbol);
      }
    } else {
      if (p.is_lf != 0) {
        bits += symbol_bits(&(*q.coeff_base_lf_eob_uv)[p.eob_ctx][0], 5, symbol);
      } else {
        bits += symbol_bits(&(*q.coeff_base_eob_uv)[p.eob_ctx][0], 3, symbol);
      }
    }
  } else {
    if (plane_ == 0) {
      if (p.is_lf != 0) {
        bits += symbol_bits(&(*q.coeff_base_lf)[tsc][p.base_ctx][0][0], 6,
                            base_level);
      } else {
        bits += symbol_bits(&(*q.coeff_base)[tsc][p.base_ctx][0][0], 4,
                            base_level);
      }
    } else {
      if (p.is_lf != 0) {
        bits += symbol_bits(&(*q.coeff_base_lf_uv)[p.base_ctx][0], 6, base_level);
      } else {
        bits += symbol_bits(&(*q.coeff_base_uv)[p.base_ctx][0], 4, base_level);
      }
    }
  }

  if (pre_level > base_levels && !(p.is_lf != 0 && plane_ > 0)) {
    if (plane_ == 0) {
      if (p.is_lf != 0) {
        bits += symbol_bits(&(*q.coeff_br_lf)[p.br_ctx][0], 4, br_level);
      } else {
        bits += symbol_bits(&(*q.coeff_br)[p.br_ctx][0], 4, br_level);
      }
    } else {
      bits += symbol_bits(&(*q.coeff_br_uv)[p.br_ctx][0], 4, br_level);
    }
  }

  if (abs_value >= p.max_level) {
    bits += escape_bits(abs_value - pre_level,
                        static_cast<std::int32_t>(p.hr_pred));
  }

  if (p.is_dc != 0) {
    bits += symbol_bits(&(*q.dc_sign)[0][0][dc_sign_context_][0], 2,
                        signed_level < 0 ? 1 : 0);
  } else {
    bits += 1.0; // sign_bit literal
  }
  return bits;
}

double TransformStreamRate::eob_bits(int eob) const noexcept {
  const av2enc::still::qctx::CoeffCdfSlice& q = av2enc::still::qctx::tables_for_slice(qctx_);
  if (!valid_ || eob < 0 || eob > area_) {
    return kImpossibleBits;
  }
  if (eob_override_ != nullptr) {
    const double bits = eob_override_(override_context_, eob);
    return std::isfinite(bits) && bits >= 0.0 ? bits : kImpossibleBits;
  }
  double bits = symbol_bits(txb_skip_cdf(), 2, eob == 0 ? 1 : 0);
  if (eob == 0) {
    return bits;
  }

  std::int32_t eob_pt = 1;
  if (eob == 2) {
    eob_pt = 2;
  } else if (eob >= 3) {
    eob_pt = floor_log2(static_cast<std::uint32_t>(eob - 1)) + 2;
  }

  const std::int32_t log2_w =
      floor_log2(static_cast<std::uint32_t>(coded_w_));
  const std::int32_t log2_h =
      floor_log2(static_cast<std::uint32_t>(coded_h_));
  const std::int32_t multisize =
      std::min(5, log2_w) + std::min(5, log2_h) - 4;

  const std::uint16_t* cdf = nullptr;
  std::int32_t n = 0;
  std::int32_t escape_raw_bits = 0;
  switch (multisize) {
    case 0:
      cdf = &(*q.eob_pt_16)[0][0];
      n = 5;
      break;
    case 1:
      cdf = &(*q.eob_pt_32)[0][0];
      n = 6;
      break;
    case 2:
      cdf = &(*q.eob_pt_64)[0][0];
      n = 7;
      break;
    case 3:
      cdf = &(*q.eob_pt_128)[0][0];
      n = 8;
      break;
    case 4:
      cdf = &(*q.eob_pt_256)[0][0];
      n = 8;
      escape_raw_bits = 1;
      break;
    case 5:
      cdf = &(*q.eob_pt_512)[0][0];
      n = 8;
      escape_raw_bits = 2;
      break;
    default:
      cdf = &(*q.eob_pt_1024)[0][0];
      n = 8;
      escape_raw_bits = 2;
      break;
  }
  const std::int32_t eob_ctx = plane_ > 0 ? 2 : 0;
  cdf += static_cast<std::ptrdiff_t>(eob_ctx) * (n + 1);

  std::int32_t symbol = 0;
  if (eob_pt <= n) {
    symbol = eob_pt - 1;
  } else {
    symbol = n - 1;
  }
  bits += symbol_bits(cdf, n, symbol);
  if (symbol == n - 1 && escape_raw_bits > 0) {
    bits += static_cast<double>(escape_raw_bits);
  }

  if (eob_pt >= 3) {
    const std::int32_t base = (1 << (eob_pt - 2)) + 1;
    const std::int32_t extra = eob - base;
    const std::int32_t eob_extra = (extra >> (eob_pt - 3)) & 1;
    bits += symbol_bits(&(*q.eob_extra)[0], 2, eob_extra);
    for (std::int32_t i = eob_pt - 4; i >= 0; --i) {
      bits += 1.0;
    }
  }
  return bits;
}

double TransformStreamRate::block_bits() const noexcept {
  if (!valid_) {
    return kImpossibleBits;
  }
  double bits = eob_bits(eob_);
  for (int c = 0; c < eob_; ++c) {
    bits += coefficient_bits(c, levels_[static_cast<std::size_t>(c)],
                             c == eob_ - 1);
  }
  return bits;
}

av2enc::rdoq::RateModel TransformStreamRate::rdoq_rate_model() noexcept {
  av2enc::rdoq::RateModel model;
  model.context = this;
  model.coefficient_rate = &TransformStreamRate::coefficient_rate_callback;
  model.eob_rate = &TransformStreamRate::eob_rate_callback;
  return model;
}

double TransformStreamRate::coefficient_rate_callback(
    void* context, std::int32_t scan_pos, std::int32_t prev_abs_state,
    std::int32_t level, std::int32_t level_context, std::int32_t is_eob) {
  (void)prev_abs_state;
  (void)level_context;
  if (context == nullptr) {
    return kImpossibleBits;
  }
  return static_cast<const TransformStreamRate*>(context)->coefficient_bits(
      scan_pos, level, is_eob != 0);
}

double TransformStreamRate::eob_rate_callback(void* context,
                                              std::int32_t eob,
                                              std::int32_t num_coefficients) {
  if (context == nullptr) {
    return kImpossibleBits;
  }
  const TransformStreamRate* rate =
      static_cast<const TransformStreamRate*>(context);
  if (num_coefficients != rate->area()) {
    return kImpossibleBits;
  }
  return rate->eob_bits(eob);
}

}  // namespace av2enc::still::rdo_decide
