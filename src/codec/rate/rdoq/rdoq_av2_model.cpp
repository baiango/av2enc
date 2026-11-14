#include "rdoq_av2_model.h"

#include "pipeline/still/still_tables.h"
#include "codec/entropy/qctx/qctx.h"
#include "rdoq_cdf_tables.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::rdoq::av2 {
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
// symbol count; the array has n+1 entries (n-1 boundaries, rate, count).
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
    probability = static_cast<std::int32_t>(cdf[symbol]) - cdf[symbol - 1];
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
  const std::int32_t m =
      clip3(1, 6, predictor <= 0 ? 0 : floor_log2(static_cast<std::uint32_t>(predictor)));
  const std::int32_t k = m + 1;
  const std::int32_t c_max = std::min(m + 4, 6);
  double bits = 0.0;
  if (x < (c_max << m)) {
    const std::int32_t q = x >> m;
    bits += static_cast<double>(q + 1);  // q zero bits + the terminating one
    bits += static_cast<double>(m);      // literal remainder
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

}  // namespace

void build_scan(std::int32_t w, std::int32_t h,
                std::span<std::int32_t> scan) {
  if (w <= 0 || h <= 0 ||
      scan.size() < static_cast<std::size_t>(w) * static_cast<std::size_t>(h)) {
    return;
  }
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

bool Model::init(const BlockInfo& info,
                 std::span<const std::int32_t> scalar_levels_scan,
                 std::string* error) {
  info_ = info;
  qctx_ = info.qctx < 0
              ? 0
              : (info.qctx >= av2enc::still::qctx::kSliceCount ? av2enc::still::qctx::kSliceCount - 1
                                                    : info.qctx);
  positions_.clear();
  levels_scan_.clear();
  scalar_eob_ = 0;

  const txv::TxParams& params = info.params;
  const std::int32_t w = params.w;
  const std::int32_t h = params.h;
  if (params.coded_w <= 0 || params.coded_h <= 0 || w <= 0 || h <= 0) {
    if (error != nullptr) *error = "invalid transform geometry";
    return false;
  }
  const std::int32_t area = w * h;
  if (scalar_levels_scan.size() != static_cast<std::size_t>(area)) {
    if (error != nullptr) *error = "scalar level span size does not match the block";
    return false;
  }
  if (info.plane < 0 || info.plane > 2) {
    if (error != nullptr) *error = "plane must be 0, 1 or 2";
    return false;
  }
  if (info.params.tx_size < 0 || info.params.tx_size >= 25) {
    if (error != nullptr) *error = "tx_size must be in [0, 24]";
    return false;
  }
  if (info.dc_sign_context < 0 || info.dc_sign_context > 2) {
    if (error != nullptr) *error = "dc_sign_context must be in [0, 2]";
    return false;
  }
  if (info.plane == 2) {
    if (info.txb_skip_context < 0 || info.txb_skip_context >= 12) {
      if (error != nullptr) *error = "v_txb_skip context must be in [0, 11]";
      return false;
    }
  } else {
    if (info.txb_skip_context < 0 || info.txb_skip_context >= 10) {
      if (error != nullptr) *error = "txb_skip context must be in [0, 9]";
      return false;
    }
  }

  area_ = area;
  levels_scan_.assign(scalar_levels_scan.begin(), scalar_levels_scan.end());
  for (std::int32_t c = 0; c < area; ++c) {
    if (levels_scan_[static_cast<std::size_t>(c)] != 0) {
      scalar_eob_ = c + 1;
    }
  }

  std::vector<std::int32_t> scan(static_cast<std::size_t>(area), 0);
  build_scan(w, h, scan);

  // Raster view of the scalar levels and of their coded "pre-level"
  // (base + BR magnitude).  The AV2 contexts read already-coded neighbours,
  // which in reverse scan order are the coefficients below/right (higher scan
  // index).  Freezing them from the scalar path is the documented RDOQ
  // approximation.  Note that the neighbour values are magnitudes, not signed
  // levels.
  std::vector<std::int32_t> level_pre(static_cast<std::size_t>(area), 0);
  for (std::int32_t c = 0; c < area; ++c) {
    const std::int32_t pos = scan[static_cast<std::size_t>(c)];
    const std::int32_t row = pos / w;
    const std::int32_t col = pos % w;
    const bool lf = (info.plane == 0 ? (row + col) < 4 : (row + col) < 1);
    const std::int32_t base_levels = lf ? 4 : 2;
    const std::int32_t base_max = lf ? 5 : 3;
    const std::int32_t br_max = (lf && info.plane > 0) ? 0 : 3;
    const std::int32_t pre_max = base_max + br_max;
    const std::int32_t abs_value =
        abs_i32(levels_scan_[static_cast<std::size_t>(c)]);
    level_pre[static_cast<std::size_t>(pos)] =
        pre_level_from_abs(abs_value, base_levels, base_max, pre_max);
  }

  static constexpr int kSigRef[5][2] = {
      {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
  static constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

  positions_.resize(static_cast<std::size_t>(area));
  for (std::int32_t c = 0; c < area; ++c) {
    Position& p = positions_[static_cast<std::size_t>(c)];
    p.pos = scan[static_cast<std::size_t>(c)];
    p.row = p.pos / w;
    p.col = p.pos % w;
    p.is_lf = (info.plane == 0 ? (p.row + p.col) < 4
                               : (p.row + p.col) < 1)
                  ? 1
                  : 0;
    p.base_levels = p.is_lf ? 4 : 2;
    p.base_max = p.is_lf ? 5 : 3;
    p.br_max = (p.is_lf && info.plane > 0) ? 0 : 3;
    p.pre_max = p.base_max + p.br_max;
    p.max_level = p.is_lf ? (info.plane == 0 ? 8 : 5) : 6;
    p.is_dc = (info.plane == 0 && p.pos == 0) ? 1 : 0;

    const std::int32_t limit = p.is_lf ? 5 : 3;
    if (info.plane == 0) {
      std::int32_t mag = 0;
      for (int k = 0; k < 5; ++k) {
        const std::int32_t rr = p.row + kSigRef[k][0];
        const std::int32_t cc = p.col + kSigRef[k][1];
        if (rr < h && cc < w) {
          mag += std::min(
              level_pre[static_cast<std::size_t>(rr * w + cc)], limit);
        }
      }
      const std::int32_t mctx = (mag + 1) >> 1;
      if (p.is_lf != 0) {
        if (c == 0) {
          p.base_ctx = std::min(mctx, 8);
        } else if (p.row + p.col < 2) {
          p.base_ctx = std::min(mctx, 6) + 9;
        } else {
          p.base_ctx = std::min(mctx, 4) + 16;
        }
      } else {
        const std::int32_t ctx2 = std::min(mctx, 4);
        p.base_ctx = ctx2;
        if (p.row + p.col >= 8) {
          p.base_ctx += 10;
        } else if (p.row + p.col >= 6) {
          p.base_ctx += 5;
        }
      }
    } else {
      std::int32_t mag = 0;
      for (int k = 0; k < 3; ++k) {
        const std::int32_t rr = p.row + kSigRef[k][0];
        const std::int32_t cc = p.col + kSigRef[k][1];
        if (rr < h && cc < w) {
          mag += std::min(
              level_pre[static_cast<std::size_t>(rr * w + cc)], limit);
        }
      }
      const std::int32_t mctx = (mag + 1) >> 1;
      const std::int32_t ctx2 = std::min(mctx, 3);
      p.base_ctx = info.plane == 1 ? ctx2 : ctx2 + 4;
    }

    if (c == 0) {
      p.eob_ctx = 0;
    } else if (c <= area / 8) {
      p.eob_ctx = 1;
    } else if (c <= area / 4) {
      p.eob_ctx = 2;
    } else {
      p.eob_ctx = 3;
    }

    std::int32_t br_mag = 0;
    for (int k = 0; k < 3; ++k) {
      const std::int32_t rr = p.row + kMagRef[k][0];
      const std::int32_t cc = p.col + kMagRef[k][1];
      if (rr < h && cc < w) {
        br_mag += std::min(
            level_pre[static_cast<std::size_t>(rr * w + cc)], 5);
      }
    }
    br_mag = std::min((br_mag + 1) >> 1, 6);
    if (info.plane == 0) {
      if (p.is_lf != 0) {
        p.br_ctx = br_mag + (p.pos != 0 ? 7 : 0);
      } else {
        p.br_ctx = br_mag;
      }
    } else {
      p.br_ctx = std::min(br_mag, 3);
    }
  }

  // Escape predictor state from the scalar reverse scan.
  std::int32_t hr_level_avg = 0;
  for (std::int32_t c = area; c-- > 0;) {
    Position& p = positions_[static_cast<std::size_t>(c)];
    p.hr_pred = hr_level_avg;
    const std::int32_t abs_value =
        abs_i32(levels_scan_[static_cast<std::size_t>(c)]);
    if (abs_value >= p.max_level) {
      const std::int32_t pre = pre_level_from_abs(
          abs_value, p.base_levels, p.base_max, p.pre_max);
      const std::int32_t x = abs_value - pre;
      hr_level_avg = (x + hr_level_avg) >> 1;
    }
  }

  // Validate the context ranges against the table dimensions.
  for (const Position& p : positions_) {
    if (info.plane == 0) {
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
  return true;
}

RateModel Model::rate_model() noexcept {
  RateModel model;
  model.context = this;
  model.coefficient_rate = &Model::coefficient_rate;
  model.eob_rate = &Model::eob_rate;
  return model;
}

std::int32_t Model::scalar_level(std::int32_t scan_pos) const {
  if (scan_pos < 0 || scan_pos >= area_) {
    return 0;
  }
  return levels_scan_[static_cast<std::size_t>(scan_pos)];
}

const std::uint16_t* Model::txb_skip_cdf() const {
  const av2enc::still::qctx::CoeffCdfSlice& q = av2enc::still::qctx::tables_for_slice(qctx_);
  if (info_.plane == 2) {
    return &(*q.v_txb_skip)[info_.txb_skip_context][0];
  }
  const std::int32_t tsc =
      (av2enc::still::kTxSizeSqr[info_.params.tx_size] +
       av2enc::still::kTxSizeSqrUp[info_.params.tx_size] + 1) >> 1;
  return &(*q.txb_skip)[info_.is_inter ? 1 : 0][tsc]
                       [info_.txb_skip_context][0];
}

double Model::coefficient_rate(void* context,
                               std::int32_t scan_pos,
                               std::int32_t prev_abs_state,
                               std::int32_t level,
                               std::int32_t level_context,
                               std::int32_t is_eob) {
  (void)prev_abs_state;
  (void)level_context;
  if (context == nullptr) {
    return kImpossibleBits;
  }
  return static_cast<Model*>(context)->coefficient_bits(
      scan_pos, level, is_eob);
}

double Model::eob_rate(void* context,
                       std::int32_t eob,
                       std::int32_t num_coefficients) {
  if (context == nullptr) {
    return kImpossibleBits;
  }
  Model* model = static_cast<Model*>(context);
  if (num_coefficients != model->area_) {
    return kImpossibleBits;
  }
  return model->eob_bits(eob);
}

double Model::coefficient_bits(std::int32_t scan_pos,
                               std::int32_t signed_level,
                               std::int32_t is_eob) const {
  if (scan_pos < 0 || scan_pos >= area_) {
    return kImpossibleBits;
  }
  const av2enc::still::qctx::CoeffCdfSlice& q = av2enc::still::qctx::tables_for_slice(qctx_);
  const Position& p = positions_[static_cast<std::size_t>(scan_pos)];
  const std::int32_t tsc =
      (av2enc::still::kTxSizeSqr[info_.params.tx_size] +
       av2enc::still::kTxSizeSqrUp[info_.params.tx_size] + 1) >> 1;
  const std::int32_t abs_value = abs_i32(signed_level);
  if (abs_value == 0) {
    if (is_eob != 0) {
      return kImpossibleBits;  // an EOB coefficient is non-zero by definition
    }
    double bits = 0.0;
    if (p.is_lf != 0) {
      if (info_.plane == 0) {
        bits += symbol_bits(&(*q.coeff_base_lf)[tsc][p.base_ctx][0][0],
                            6, 0);
      } else {
        bits += symbol_bits(&(*q.coeff_base_lf_uv)[p.base_ctx][0], 6, 0);
      }
    } else {
      if (info_.plane == 0) {
        bits += symbol_bits(&(*q.coeff_base)[tsc][p.base_ctx][0][0],
                            4, 0);
      } else {
        bits += symbol_bits(&(*q.coeff_base_uv)[p.base_ctx][0], 4, 0);
      }
    }
    return bits;
  }

  const std::int32_t base_levels = p.base_levels;
  const std::int32_t base_max = p.base_max;
  const std::int32_t pre_max = p.pre_max;
  std::int32_t base_level = 0;
  std::int32_t br_level = 0;
  if (abs_value <= base_levels) {
    base_level = abs_value;
  } else {
    base_level = std::min(base_levels + 1, base_max);
    const std::int32_t pre = std::min(abs_value, pre_max);
    br_level = pre - base_level;
  }
  const std::int32_t pre_level = base_level + br_level;

  double bits = 0.0;
  if (is_eob != 0) {
    const std::int32_t symbol = base_level - 1;
    if (info_.plane == 0) {
      if (p.is_lf != 0) {
        bits += symbol_bits(
            &(*q.coeff_base_lf_eob)[tsc][p.eob_ctx][0], 5,
            symbol);
      } else {
        bits += symbol_bits(
            &(*q.coeff_base_eob)[tsc][p.eob_ctx][0], 3,
            symbol);
      }
    } else {
      if (p.is_lf != 0) {
        bits += symbol_bits(&(*q.coeff_base_lf_eob_uv)[p.eob_ctx][0], 5, symbol);
      } else {
        bits += symbol_bits(&(*q.coeff_base_eob_uv)[p.eob_ctx][0], 3, symbol);
      }
    }
  } else {
    if (info_.plane == 0) {
      if (p.is_lf != 0) {
        bits += symbol_bits(
            &(*q.coeff_base_lf)[tsc][p.base_ctx][0][0], 6,
            base_level);
      } else {
        bits += symbol_bits(
            &(*q.coeff_base)[tsc][p.base_ctx][0][0], 4,
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

  if (pre_level > base_levels && !(p.is_lf != 0 && info_.plane > 0)) {
    if (info_.plane == 0) {
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
    bits += escape_bits(abs_value - pre_level, p.hr_pred);
  }

  if (p.is_dc != 0) {
    bits += symbol_bits(&(*q.dc_sign)[0][0][info_.dc_sign_context][0], 2,
                        signed_level < 0 ? 1 : 0);
  } else {
    bits += 1.0;  // sign_bit literal
  }
  return bits;
}

double Model::eob_bits(std::int32_t eob) const {
  const av2enc::still::qctx::CoeffCdfSlice& q = av2enc::still::qctx::tables_for_slice(qctx_);
  if (eob < 0 || eob > area_) {
    return kImpossibleBits;
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
      floor_log2(static_cast<std::uint32_t>(info_.params.coded_w));
  const std::int32_t log2_h =
      floor_log2(static_cast<std::uint32_t>(info_.params.coded_h));
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
  // The eob plane context is 2 for chroma and 0/1 for intra/inter luma.
  const std::int32_t eob_ctx =
      info_.plane > 0 ? 2 : (info_.is_inter ? 1 : 0);
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

double Model::scalar_bits() const {
  double bits = eob_bits(scalar_eob_);
  for (std::int32_t c = 0; c < scalar_eob_; ++c) {
    const std::int32_t level = levels_scan_[static_cast<std::size_t>(c)];
    const std::int32_t is_eob = (c == scalar_eob_ - 1) ? 1 : 0;
    bits += coefficient_bits(c, level, is_eob);
  }
  return bits;
}

}  // namespace av2enc::rdoq::av2
