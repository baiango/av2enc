#include "parity_hiding.h"

#include "codec/entropy/qctx/qctx.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::tcq::parity {
namespace {

constexpr double kCdfScale = 32768.0;
constexpr double kImpossibleBits = 1.0e6;
constexpr int kMaxSymbols = 8;

// §5.20.7.27 Sig_Ref_Diff_Offset[3][5][2].
constexpr std::int32_t kSigRef[3][5][2] = {
    {{0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}},
    {{0, 1}, {1, 0}, {0, 2}, {0, 3}, {0, 4}},
    {{0, 1}, {1, 0}, {2, 0}, {3, 0}, {4, 0}},
};

// §5.20.7.27 Mag_Ref_Offset_With_Tx_Class[3][3][2].
constexpr std::int32_t kMagRef[3][3][2] = {
    {{0, 1}, {1, 0}, {1, 1}},
    {{0, 1}, {1, 0}, {0, 2}},
    {{0, 1}, {1, 0}, {2, 0}},
};

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

struct BaseDecision {
  CdfKind kind = CdfKind::kNone;
  std::int32_t ctx = 0;
  std::int32_t ctx2 = 0;  // TCQ CDF index (always 0 for parity hiding)
  std::int32_t symbols = 0;
};

// §5.20.7.27 coeff_base CDF selection.  `level_pre` is the first-pass
// Level[][] array in raster order; all reference positions are at higher scan
// indices and are already filled.  `c` is the scan index and `eob` the EOB.
BaseDecision choose_base(std::int32_t c, std::int32_t eob, std::int32_t pos,
                         std::int32_t row, std::int32_t col,
                         std::span<const std::int32_t> level_pre,
                         const Geometry& geometry, bool is_hidden,
                         bool is_lf) {
  BaseDecision out;
  const std::int32_t plane = geometry.plane;
  const TxClass tx_class = geometry.tx_class;

  if (c == eob - 1) {
    std::int32_t ctx = 0;
    if (c != 0) {
      const std::int32_t area = geometry.w * geometry.h;
      if (c <= area / 8) {
        ctx = 1;
      } else if (c <= area / 4) {
        ctx = 2;
      } else {
        ctx = 3;
      }
    }
    out.ctx = ctx;
    if (plane == 0) {
      out.kind = is_lf ? CdfKind::kCoeffBaseLfEob : CdfKind::kCoeffBaseEob;
      out.symbols = is_lf ? 5 : 3;
    } else {
      out.kind = is_lf ? CdfKind::kCoeffBaseLfEobUv : CdfKind::kCoeffBaseEobUv;
      out.symbols = is_lf ? 5 : 3;
    }
    return out;
  }

  const std::int32_t num = plane == 0 ? 5 : (tx_class == TxClass::k2D ? 3 : 2);
  std::int32_t mag = 0;
  for (std::int32_t idx = 0; idx < num; ++idx) {
    const std::int32_t rr = row + kSigRef[static_cast<int>(tx_class)][idx][0];
    const std::int32_t cc = col + kSigRef[static_cast<int>(tx_class)][idx][1];
    if (rr < geometry.h && cc < geometry.w) {
      const std::int32_t limit =
          (is_lf &&
           (tx_class == TxClass::k2D || idx < 2) &&
           !(is_hidden && c == 0))
              ? 5
              : 3;
      const std::int32_t value =
          level_pre[static_cast<std::size_t>(rr * geometry.w + cc)];
      mag += std::min(value, limit);
    }
  }
  const std::int32_t ctx = (mag + 1) >> 1;

  if (is_hidden && c == 0) {
    out.kind = CdfKind::kCoeffBasePh;
    out.ctx = std::min(ctx, kPhContexts - 1);
    out.symbols = 4;
    return out;
  }

  if (plane == 0) {
    if (is_lf) {
      out.kind = CdfKind::kCoeffBaseLf;
      out.symbols = 6;
      if (tx_class == TxClass::k2D) {
        if (c == 0) {
          out.ctx = std::min(ctx, 8);
        } else if (row + col < 2) {
          out.ctx = std::min(ctx, 6) + 9;
        } else {
          out.ctx = std::min(ctx, 4) + 16;
        }
      } else {
        const std::int32_t idx =
            tx_class == TxClass::kHoriz ? col : row;
        if (idx == 0) {
          out.ctx = 21 + std::min(ctx, 6);
        } else {
          out.ctx = 21 + 7 + std::min(ctx, 4);
        }
      }
    } else {
      out.kind = CdfKind::kCoeffBase;
      out.symbols = 4;
      const std::int32_t ctx2 = std::min(ctx, 4);
      if (tx_class == TxClass::k2D) {
        out.ctx = ctx2;
        if (row + col >= 8) {
          out.ctx += 10;
        } else if (row + col >= 6) {
          out.ctx += 5;
        }
      } else {
        // §5.20.7.27: 1D luma HF base contexts are ctx2 + 15 (15..19).
        out.ctx = ctx2 + 15;
      }
    }
  } else {
    out.kind = is_lf ? CdfKind::kCoeffBaseLfUv : CdfKind::kCoeffBaseUv;
    out.symbols = is_lf ? 6 : 4;
    const std::int32_t ctx2 = std::min(ctx, 3);
    out.ctx = tx_class != TxClass::k2D
                  ? ctx2 + 8
                  : (plane == 1 ? ctx2 : ctx2 + 4);
  }
  out.ctx2 = 0;
  (void)pos;
  return out;
}

// §5.20.7.27 coeff_br CDF selection.
BaseDecision choose_br(std::int32_t pos, std::int32_t row, std::int32_t col,
                       std::span<const std::int32_t> level_pre,
                       const Geometry& geometry, bool is_lf) {
  BaseDecision out;
  const std::int32_t plane = geometry.plane;
  const TxClass tx_class = geometry.tx_class;
  const std::int32_t num = (tx_class != TxClass::k2D && plane > 0) ? 2 : 3;
  std::int32_t mag = 0;
  for (std::int32_t idx = 0; idx < num; ++idx) {
    const std::int32_t rr = row + kMagRef[static_cast<int>(tx_class)][idx][0];
    const std::int32_t cc = col + kMagRef[static_cast<int>(tx_class)][idx][1];
    if (rr < geometry.h && cc < geometry.w) {
      mag += std::min(
          level_pre[static_cast<std::size_t>(rr * geometry.w + cc)],
          kMaxBaseBrRange - 1);
    }
  }
  mag = std::min((mag + 1) >> 1, 6);

  if (plane > 0) {
    out.kind = CdfKind::kCoeffBrUv;
    out.ctx = std::min(mag, 3);
  } else if (pos == 0) {
    out.kind = is_lf ? CdfKind::kCoeffBrLf : CdfKind::kCoeffBr;
    out.ctx = tx_class != TxClass::k2D ? mag + 7 : mag;
  } else {
    out.kind = is_lf ? CdfKind::kCoeffBrLf : CdfKind::kCoeffBr;
    out.ctx = is_lf ? mag + 7 : mag;
  }
  out.symbols = 4;
  return out;
}

struct EobDecision {
  CdfKind kind = CdfKind::kNone;
  std::int32_t ctx = 0;
  std::int32_t symbols = 0;
  std::int32_t escape_bits = 0;
};

EobDecision choose_eob_cdf(const Geometry& geometry, bool is_inter) {
  EobDecision out;
  const std::int32_t multisize = eob_multisize(geometry);
  out.ctx = geometry.plane > 0 ? 2 : (is_inter ? 1 : 0);
  switch (multisize) {
    case 0:
      out.kind = CdfKind::kEobPt16;
      out.symbols = 5;
      out.escape_bits = 0;
      break;
    case 1:
      out.kind = CdfKind::kEobPt32;
      out.symbols = 6;
      out.escape_bits = 0;
      break;
    case 2:
      out.kind = CdfKind::kEobPt64;
      out.symbols = 7;
      out.escape_bits = 0;
      break;
    case 3:
      out.kind = CdfKind::kEobPt128;
      out.symbols = 8;
      out.escape_bits = 0;
      break;
    case 4:
      out.kind = CdfKind::kEobPt256;
      out.symbols = 8;
      out.escape_bits = 1;
      break;
    case 5:
      out.kind = CdfKind::kEobPt512;
      out.symbols = 8;
      out.escape_bits = 2;
      break;
    default:
      out.kind = CdfKind::kEobPt1024;
      out.symbols = 8;
      out.escape_bits = 2;
      break;
  }
  return out;
}

std::int32_t eob_pt_for(std::int32_t eob) noexcept {
  if (eob <= 1) return 1;
  if (eob == 2) return 2;
  return floor_log2(static_cast<std::uint32_t>(eob - 1)) + 2;
}

// Fills the EOB syntax fields for `eob`.
void derive_eob(std::int32_t eob, const Geometry& geometry, bool is_inter,
                BlockSyntax* syntax) {
  syntax->eob = eob;
  if (eob == 0) {
    syntax->eob_kind = CdfKind::kNone;
    syntax->eob_ctx = 0;
    syntax->eob_symbols = 0;
    syntax->eob_symbol = 0;
    syntax->eob_pt = 0;
    syntax->eob_escape_bits = 0;
    syntax->eob_escape_value = 0;
    syntax->eob_extra_symbol = 0;
    syntax->eob_extra_lsb_bits = 0;
    syntax->eob_extra_lsb_value = 0;
    return;
  }
  const EobDecision decision = choose_eob_cdf(geometry, is_inter);
  const std::int32_t eob_pt = eob_pt_for(eob);
  syntax->eob_kind = decision.kind;
  syntax->eob_ctx = decision.ctx;
  syntax->eob_symbols = decision.symbols;
  syntax->eob_pt = eob_pt;
  std::int32_t escape = 0;
  if (eob_pt <= decision.symbols) {
    syntax->eob_symbol = eob_pt - 1;
  } else {
    syntax->eob_symbol = decision.symbols - 1;
    escape = eob_pt - decision.symbols;
  }
  // `m1`/AVM write the escape literal whenever the last CDF symbol is used,
  // including the `eob_pt == cdf_n` boundary where the literal is zero.
  syntax->eob_escape_bits =
      (syntax->eob_symbol == decision.symbols - 1) ? decision.escape_bits : 0;
  syntax->eob_escape_value = escape;
  syntax->eob_extra_symbol = 0;
  syntax->eob_extra_lsb_bits = 0;
  syntax->eob_extra_lsb_value = 0;
  if (eob_pt >= 3) {
    const std::int32_t base = (1 << (eob_pt - 2)) + 1;
    const std::int32_t extra = eob - base;
    syntax->eob_extra_symbol = (extra >> (eob_pt - 3)) & 1;
    syntax->eob_extra_lsb_bits = eob_pt - 3;
    syntax->eob_extra_lsb_value =
        extra & ((1 << (eob_pt - 3)) - 1);
  }
}

bool sign_uses_dc_cdf(std::int32_t plane, TxClass tx_class, std::int32_t pos,
                      std::int32_t row, std::int32_t col) noexcept {
  if (plane != 0) return false;
  if (pos == 0) return true;
  if (tx_class == TxClass::kHoriz && col == 0) return true;
  if (tx_class == TxClass::kVert && row == 0) return true;
  return false;
}

// Full-level -> (base, br, escape) decomposition for a non-hidden position.
void split_normal_level(std::int32_t abs_level, std::int32_t c,
                        std::int32_t eob, const LevelLimits& limits,
                        PositionSyntax* out) {
  const std::int32_t syntax_level = std::min(abs_level, limits.pre_max);
  std::int32_t level = 0;
  if (c == eob - 1) {
    const std::int32_t base_abs = std::min(syntax_level, limits.base_max);
    out->base_offset = 1;
    out->base_symbol = base_abs - 1;
    level = base_abs;
  } else {
    const std::int32_t base_abs = std::min(syntax_level, limits.base_max);
    out->base_offset = 0;
    out->base_symbol = base_abs;
    level = base_abs;
  }
  out->br_present = (level > limits.base_levels) &&
                    !(limits.is_lf && limits.br_max == 0);
  out->br_symbol = out->br_present ? (syntax_level - level) : 0;
  // `level` in §5.20.7.27's first pass is the value after `level += coeff_br`.
  out->level = level + out->br_symbol;
  out->max_level = limits.pre_max;
  out->escape_present = syntax_level >= limits.pre_max;
  out->escape_x = out->escape_present ? (abs_level - syntax_level) : 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public CDF decision wrappers
// ---------------------------------------------------------------------------

CdfDecision coeff_base_cdf_decision(
    std::int32_t c, std::int32_t eob, std::int32_t pos, std::int32_t row,
    std::int32_t col, std::span<const std::int32_t> level_pre,
    const Geometry& geometry, bool is_hidden, bool is_lf) {
  const BaseDecision decision = choose_base(c, eob, pos, row, col, level_pre,
                                            geometry, is_hidden, is_lf);
  CdfDecision out;
  out.kind = decision.kind;
  out.ctx = decision.ctx;
  out.ctx2 = decision.ctx2;
  out.symbols = decision.symbols;
  return out;
}

CdfDecision coeff_br_cdf_decision(
    std::int32_t pos, std::int32_t row, std::int32_t col,
    std::span<const std::int32_t> level_pre, const Geometry& geometry,
    bool is_lf) {
  const BaseDecision decision =
      choose_br(pos, row, col, level_pre, geometry, is_lf);
  CdfDecision out;
  out.kind = decision.kind;
  out.ctx = decision.ctx;
  out.ctx2 = decision.ctx2;
  out.symbols = decision.symbols;
  return out;
}

EobCdfDecision eob_cdf_decision(const Geometry& geometry, bool is_inter) {
  const EobDecision decision = choose_eob_cdf(geometry, is_inter);
  EobCdfDecision out;
  out.kind = decision.kind;
  out.ctx = decision.ctx;
  out.symbols = decision.symbols;
  out.escape_bits = decision.escape_bits;
  return out;
}

std::int32_t eob_pt_for_eob(std::int32_t eob) noexcept {
  return eob_pt_for(eob);
}

bool sign_is_dc_style(std::int32_t plane, TxClass tx_class, std::int32_t pos,
                      std::int32_t row, std::int32_t col) noexcept {
  return sign_uses_dc_cdf(plane, tx_class, pos, row, col);
}

std::int32_t eob_from_pt(std::int32_t eob_pt) noexcept {
  if (eob_pt < 2) return eob_pt;
  return (1 << (eob_pt - 2)) + 1;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const char* cdf_kind_name(CdfKind kind) noexcept {
  switch (kind) {
    case CdfKind::kNone: return "none";
    case CdfKind::kTxbSkip: return "txb_skip";
    case CdfKind::kEobPt16: return "eob_pt_16";
    case CdfKind::kEobPt32: return "eob_pt_32";
    case CdfKind::kEobPt64: return "eob_pt_64";
    case CdfKind::kEobPt128: return "eob_pt_128";
    case CdfKind::kEobPt256: return "eob_pt_256";
    case CdfKind::kEobPt512: return "eob_pt_512";
    case CdfKind::kEobPt1024: return "eob_pt_1024";
    case CdfKind::kEobExtra: return "eob_extra";
    case CdfKind::kCoeffBaseEob: return "coeff_base_eob";
    case CdfKind::kCoeffBaseLfEob: return "coeff_base_lf_eob";
    case CdfKind::kCoeffBaseUv: return "coeff_base_uv";
    case CdfKind::kCoeffBaseLfUv: return "coeff_base_lf_uv";
    case CdfKind::kCoeffBaseEobUv: return "coeff_base_eob_uv";
    case CdfKind::kCoeffBaseLfEobUv: return "coeff_base_lf_eob_uv";
    case CdfKind::kCoeffBase: return "coeff_base";
    case CdfKind::kCoeffBaseLf: return "coeff_base_lf";
    case CdfKind::kCoeffBasePh: return "coeff_base_ph";
    case CdfKind::kCoeffBr: return "coeff_br";
    case CdfKind::kCoeffBrLf: return "coeff_br_lf";
    case CdfKind::kCoeffBrUv: return "coeff_br_uv";
    case CdfKind::kDcSign: return "dc_sign";
  }
  return "?";
}

std::int32_t clip3(std::int32_t low, std::int32_t high,
                   std::int32_t value) noexcept {
  return value < low ? low : (value > high ? high : value);
}

std::int32_t get_msb(std::uint32_t value) noexcept {
  return floor_log2(value);
}

std::int32_t update_hr_level_avg(std::int32_t x, std::int32_t hr_level_avg,
                                 std::int32_t lvl_shift) noexcept {
  return ((x << lvl_shift) + hr_level_avg) >> 1;
}

std::int32_t dequantize_level(std::int32_t level, std::int32_t dq,
                              std::int32_t shift) noexcept {
  const std::int64_t aq =
      level < 0 ? -static_cast<std::int64_t>(level) : level;
  const std::int64_t high = (aq * dq) & 0xFFFFFF;
  const std::int32_t rounded =
      static_cast<std::int32_t>((high + (1 << (3 - 1))) >> 3);
  const std::int32_t value = rounded >> shift;
  return level < 0 ? -value : value;
}

bool make_geometry(std::int32_t tx_size, TxClass tx_class, std::int32_t plane,
                   Geometry* geometry, std::string* error) {
  if (geometry == nullptr) {
    if (error != nullptr) *error = "null geometry";
    return false;
  }
  if (tx_size < 0 || tx_size >= kTxSizesAll) {
    if (error != nullptr) *error = "tx_size must be in [0, 24]";
    return false;
  }
  if (plane < 0 || plane > 2) {
    if (error != nullptr) *error = "plane must be in [0, 2]";
    return false;
  }
  geometry->tx_size = tx_size;
  geometry->coded_w = kTxWidth[tx_size];
  geometry->coded_h = kTxHeight[tx_size];
  geometry->w = std::min(geometry->coded_w, kAdjustedMaxDim);
  geometry->h = std::min(geometry->coded_h, kAdjustedMaxDim);
  geometry->tx_class = tx_class;
  geometry->plane = plane;
  return true;
}

std::int32_t eob_multisize(const Geometry& geometry) noexcept {
  const std::int32_t lw = std::min(5, floor_log2(
      static_cast<std::uint32_t>(std::max<std::int32_t>(geometry.coded_w, 1))));
  const std::int32_t lh = std::min(5, floor_log2(
      static_cast<std::uint32_t>(std::max<std::int32_t>(geometry.coded_h, 1))));
  return lw + lh - 4;
}

bool parity_hiding_eligible(const Geometry& geometry,
                            const Eligibility& eligibility) noexcept {
  return eligibility.allow_parity_hiding && !eligibility.lossless &&
         !eligibility.allow_tcq && !eligibility.tx_type_is_idtx &&
         !eligibility.use_fsc && geometry.plane == 0;
}

bool is_low_frequency(std::int32_t row, std::int32_t col, TxClass tx_class,
                      std::int32_t plane) noexcept {
  if (tx_class == TxClass::k2D) {
    return plane == 0 ? ((row + col) < 4) : ((row + col) < 1);
  }
  if (tx_class == TxClass::kHoriz) {
    return plane == 0 ? (col < 2) : (col < 1);
  }
  return plane == 0 ? (row < 2) : (row < 1);
}

LevelLimits level_limits(std::int32_t row, std::int32_t col, TxClass tx_class,
                         std::int32_t plane) noexcept {
  LevelLimits out;
  out.is_lf = is_low_frequency(row, col, tx_class, plane);
  out.base_levels = out.is_lf ? kLfNumBaseLevels : kNumBaseLevels;
  out.base_max = out.is_lf ? 5 : 3;
  out.br_max = (out.is_lf && plane > 0) ? 0 : kCoeffBaseRange;
  out.pre_max = out.base_max + out.br_max;
  return out;
}

void build_scan(std::int32_t w, std::int32_t h, TxClass tx_class,
                std::span<std::int32_t> scan) noexcept {
  if (w <= 0 || h <= 0 ||
      scan.size() < static_cast<std::size_t>(w) * static_cast<std::size_t>(h)) {
    return;
  }
  std::int32_t c = 0;
  if (tx_class == TxClass::kVert) {
    for (std::int32_t y = 0; y < h; ++y) {
      for (std::int32_t x = 0; x < w; ++x) {
        scan[static_cast<std::size_t>(c++)] = y * w + x;
      }
    }
    return;
  }
  if (tx_class == TxClass::kHoriz) {
    for (std::int32_t x = 0; x < w; ++x) {
      for (std::int32_t y = 0; y < h; ++y) {
        scan[static_cast<std::size_t>(c++)] = y * w + x;
      }
    }
    return;
  }
  std::int32_t x = 0;
  std::int32_t y = 0;
  const std::int32_t count = w * h;
  for (std::int32_t i = 0; i < count; ++i) {
    scan[static_cast<std::size_t>(i)] = y * w + x;
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

bool build_scan(const Geometry& geometry, std::span<std::int32_t> scan,
                std::string* error) {
  const std::size_t area = static_cast<std::size_t>(geometry.w) *
                           static_cast<std::size_t>(geometry.h);
  if (scan.size() < area) {
    if (error != nullptr) *error = "scan span is smaller than w*h";
    return false;
  }
  build_scan(geometry.w, geometry.h, geometry.tx_class, scan);
  return true;
}

// ---------------------------------------------------------------------------
// `read_quant` escape length (§5.20.7.28)
// ---------------------------------------------------------------------------

std::int32_t escape_bit_count(std::int32_t x, std::int32_t predictor) noexcept {
  if (x < 0) return -1;
  const std::int32_t m =
      clip3(1, 6, predictor <= 0 ? 0 : get_msb(
                                  static_cast<std::uint32_t>(predictor)));
  const std::int32_t k = m + 1;
  const std::int32_t c_max = std::min(m + 4, 6);
  if (x < (c_max << m)) {
    const std::int32_t q = x >> m;
    return q + 1 + m;
  }
  std::int32_t bits = c_max;
  std::int32_t g = 0;
  for (;;) {
    const std::int32_t length = k + g;
    const std::int32_t x_base = (c_max << m) + (1 << length) - (1 << k);
    if (x < x_base + (1 << length)) {
      return bits + g + 1 + length;
    }
    ++g;
    if (g > 30) return -1;
  }
}

// ---------------------------------------------------------------------------
// Syntax derivation
// ---------------------------------------------------------------------------

bool derive_syntax(std::span<const std::int32_t> levels_scan,
                   const Geometry& geometry, const Eligibility& eligibility,
                   BlockSyntax* syntax, std::string* error) {
  if (syntax == nullptr) {
    if (error != nullptr) *error = "null syntax";
    return false;
  }
  const std::size_t area = static_cast<std::size_t>(geometry.w) *
                           static_cast<std::size_t>(geometry.h);
  if (levels_scan.size() != area) {
    if (error != nullptr) {
      *error = "levels span size does not match the adjusted block";
    }
    return false;
  }

  std::vector<std::int32_t> scan(area, 0);
  build_scan(geometry.w, geometry.h, geometry.tx_class, scan);

  std::int32_t eob = 0;
  for (std::int32_t c = static_cast<std::int32_t>(area) - 1; c >= 0; --c) {
    if (levels_scan[static_cast<std::size_t>(c)] != 0) {
      eob = c + 1;
      break;
    }
  }

  syntax->positions.clear();
  syntax->parity_hiding_eligible =
      parity_hiding_eligible(geometry, eligibility);
  syntax->is_hidden = false;
  syntax->sum_abs1 = 0;
  syntax->num_nz = 0;
  syntax->dc_ctx = 0;
  derive_eob(eob, geometry, eligibility.is_inter, syntax);
  if (eob == 0) return true;

  syntax->positions.resize(static_cast<std::size_t>(eob));

  std::vector<std::int32_t> level_pre(area, 0);
  const bool hidden_possible =
      syntax->parity_hiding_eligible && eob > kPhThresh;

  bool is_hidden = false;
  std::int32_t sum_abs1 = 0;
  std::int32_t num_nz = 0;

  // First pass over AC positions (c == eob-1 .. 1), exactly as §5.20.7.27.
  for (std::int32_t c = eob - 1; c >= 1; --c) {
    PositionSyntax& p = syntax->positions[static_cast<std::size_t>(c)];
    p.scan_index = c;
    p.pos = scan[static_cast<std::size_t>(c)];
    p.row = p.pos / geometry.w;
    p.col = p.pos % geometry.w;
    const LevelLimits limits =
        level_limits(p.row, p.col, geometry.tx_class, geometry.plane);
    p.is_lf = limits.is_lf;
    p.is_dc = false;
    const std::int32_t abs_level = abs_i32(
        levels_scan[static_cast<std::size_t>(c)]);
    split_normal_level(abs_level, c, eob, limits, &p);
    const BaseDecision base = choose_base(c, eob, p.pos, p.row, p.col,
                                          level_pre, geometry, is_hidden,
                                          p.is_lf);
    p.base_kind = base.kind;
    p.base_ctx = base.ctx;
    p.base_ctx2 = base.ctx2;
    p.base_symbols = base.symbols;
    if (p.br_present) {
      const BaseDecision br = choose_br(p.pos, p.row, p.col, level_pre,
                                        geometry, p.is_lf);
      p.br_kind = br.kind;
      p.br_ctx = br.ctx;
      p.br_symbols = br.symbols;
    }
    level_pre[static_cast<std::size_t>(p.pos)] = p.level;
    if (hidden_possible) {
      sum_abs1 ^= (std::min(p.level, kMaxBaseBrRange) & 1);
      if (p.level != 0) ++num_nz;
      is_hidden = num_nz >= kPhThresh;
    }
  }

  // DC position (c == 0).
  {
    PositionSyntax& p = syntax->positions[0];
    p.scan_index = 0;
    p.pos = scan[0];
    p.row = p.pos / geometry.w;
    p.col = p.pos % geometry.w;
    const LevelLimits limits =
        level_limits(p.row, p.col, geometry.tx_class, geometry.plane);
    p.is_lf = limits.is_lf;
    p.is_dc = true;
    const std::int32_t signed_level = levels_scan[0];
    const std::int32_t abs_level = abs_i32(signed_level);
    const std::int32_t dc_parity = abs_level & 1;

    if (is_hidden && dc_parity != sum_abs1) {
      if (error != nullptr) {
        *error = "hidden DC parity does not match sumAbs1";
      }
      return false;
    }

    if (is_hidden) {
      const BaseDecision base = choose_base(0, eob, p.pos, p.row, p.col,
                                            level_pre, geometry, true,
                                            p.is_lf);
      p.base_kind = base.kind;
      p.base_ctx = base.ctx;
      p.base_ctx2 = base.ctx2;
      p.base_symbols = base.symbols;
      p.base_offset = 0;
      const std::int32_t q_index = abs_level >> 1;
      p.base_symbol = std::min(q_index, 3);
      p.level = p.base_symbol;
      p.max_level = kNumBaseLevels + 1;  // 3
      p.escape_present = q_index > kNumBaseLevels;
      p.escape_x = p.escape_present ? (q_index - p.base_symbol) : 0;
      p.br_present = false;
      p.br_kind = CdfKind::kNone;
      p.br_symbol = 0;
      syntax->dc_ctx = p.base_ctx;
    } else {
      split_normal_level(abs_level, 0, eob, limits, &p);
      const BaseDecision base = choose_base(0, eob, p.pos, p.row, p.col,
                                            level_pre, geometry, false,
                                            p.is_lf);
      p.base_kind = base.kind;
      p.base_ctx = base.ctx;
      p.base_ctx2 = base.ctx2;
      p.base_symbols = base.symbols;
      if (p.br_present) {
        const BaseDecision br = choose_br(p.pos, p.row, p.col, level_pre,
                                          geometry, p.is_lf);
        p.br_kind = br.kind;
        p.br_ctx = br.ctx;
        p.br_symbols = br.symbols;
      }
    }
    level_pre[0] = p.level;
    syntax->is_hidden = is_hidden;
    syntax->sum_abs1 = sum_abs1;
    syntax->num_nz = num_nz;
  }

  // Second pass (c == eob-1 .. 0): signs and `read_quant` escapes.
  std::int32_t hr_level_avg = 0;
  for (std::int32_t c = eob - 1; c >= 0; --c) {
    PositionSyntax& p = syntax->positions[static_cast<std::size_t>(c)];
    const std::int32_t signed_level =
        levels_scan[static_cast<std::size_t>(c)];
    const bool hidden_dc = is_hidden && c == 0;
    p.sign_present = (p.level != 0) || (hidden_dc && sum_abs1 > 0);
    p.sign = signed_level < 0 ? 1 : 0;
    p.dc_style_sign =
        sign_uses_dc_cdf(geometry.plane, geometry.tx_class, p.pos, p.row,
                         p.col);

    if (p.escape_present) {
      const std::int32_t lvl_shift = hidden_dc ? 1 : 0;
      p.hr_predictor = hr_level_avg >> lvl_shift;
      p.escape_bits = escape_bit_count(p.escape_x, p.hr_predictor);
      if (p.escape_bits < 0) {
        if (error != nullptr) *error = "escape value is not codable";
        return false;
      }
      if (p.escape_x > (1 << 20)) {
        if (error != nullptr) *error = "escape value is too large";
        return false;
      }
      hr_level_avg =
          update_hr_level_avg(p.escape_x, hr_level_avg, lvl_shift);
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Normative inverse
// ---------------------------------------------------------------------------

bool decode_syntax(const BlockSyntax& syntax, const Geometry& geometry,
                   DecodedBlock* decoded, std::string* error) {
  if (decoded == nullptr) {
    if (error != nullptr) *error = "null decoded block";
    return false;
  }
  const std::size_t area = static_cast<std::size_t>(geometry.w) *
                           static_cast<std::size_t>(geometry.h);
  if (syntax.eob < 0 || static_cast<std::size_t>(syntax.eob) > area) {
    if (error != nullptr) *error = "eob outside the block";
    return false;
  }
  if (syntax.positions.size() !=
      static_cast<std::size_t>(syntax.eob)) {
    if (error != nullptr) *error = "syntax position count does not match eob";
    return false;
  }
  if (syntax.eob == 0) {
    decoded->quant_levels.assign(area, 0);
    decoded->eob = 0;
    decoded->is_hidden = false;
    decoded->sum_abs1 = 0;
    decoded->cul_level = 0;
    decoded->dc_category = 0;
    return true;
  }

  std::vector<std::int32_t> scan(area, 0);
  build_scan(geometry.w, geometry.h, geometry.tx_class, scan);

  // First pass: recompute isHidden / sumAbs1 from the coded base + br symbols.
  std::vector<std::int32_t> level_pre(area, 0);
  std::int32_t sum_abs1 = 0;
  std::int32_t num_nz = 0;
  bool is_hidden = false;
  const bool hidden_possible =
      syntax.parity_hiding_eligible && syntax.eob > kPhThresh;

  for (std::int32_t c = syntax.eob - 1; c >= 0; --c) {
    const PositionSyntax& p = syntax.positions[static_cast<std::size_t>(c)];
    const LevelLimits limits =
        level_limits(p.row, p.col, geometry.tx_class, geometry.plane);
    std::int32_t level = p.base_symbol + p.base_offset;
    if (level > limits.base_levels &&
        !(limits.is_lf && limits.br_max == 0)) {
      if (!p.br_present) {
        if (error != nullptr) *error = "coeff_br expected but not present";
        return false;
      }
      level += p.br_symbol;
      if (p.br_symbol < 0 || p.br_symbol > limits.br_max) {
        if (error != nullptr) *error = "coeff_br symbol outside [0, 3]";
        return false;
      }
    } else if (p.br_present) {
      if (error != nullptr) *error = "unexpected coeff_br";
      return false;
    }
    if (limits.is_lf && limits.br_max == 0 && level > limits.base_max) {
      if (error != nullptr) *error = "LF chroma level exceeds base_max";
      return false;
    }
    level_pre[static_cast<std::size_t>(p.pos)] = level;
    if (hidden_possible && c > 0) {
      sum_abs1 ^= (std::min(level, kMaxBaseBrRange) & 1);
      if (level != 0) ++num_nz;
      is_hidden = num_nz >= kPhThresh;
    }
  }

  if (is_hidden != syntax.is_hidden || sum_abs1 != syntax.sum_abs1) {
    if (error != nullptr) {
      *error = "decoded hidden state disagrees with the syntax record";
    }
    return false;
  }

  // Second pass: signs, escapes, hidden-DC mapping.
  decoded->quant_levels.assign(area, 0);
  decoded->dc_category = 0;
  decoded->cul_level = 0;
  std::int32_t hr_level_avg = 0;
  for (std::int32_t c = syntax.eob - 1; c >= 0; --c) {
    const PositionSyntax& p = syntax.positions[static_cast<std::size_t>(c)];
    const std::int32_t level = level_pre[static_cast<std::size_t>(p.pos)];
    const LevelLimits limits =
        level_limits(p.row, p.col, geometry.tx_class, geometry.plane);
    const bool hidden_dc = is_hidden && c == 0;
    const std::int32_t max_level =
        hidden_dc ? (kNumBaseLevels + 1) : limits.pre_max;
    const bool sign_present =
        (level != 0) || (hidden_dc && sum_abs1 > 0);
    if (sign_present != p.sign_present) {
      if (error != nullptr) *error = "sign presence disagrees with the syntax";
      return false;
    }

    std::int32_t quant = level;
    const bool escape_expected = level >= max_level;
    if (escape_expected != p.escape_present) {
      if (error != nullptr) *error = "escape presence disagrees with the syntax";
      return false;
    }
    if (p.escape_present) {
      const std::int32_t lvl_shift = hidden_dc ? 1 : 0;
      const std::int32_t predictor = hr_level_avg >> lvl_shift;
      if (predictor != p.hr_predictor) {
        if (error != nullptr) *error = "escape predictor disagrees";
        return false;
      }
      if (escape_bit_count(p.escape_x, predictor) != p.escape_bits) {
        if (error != nullptr) *error = "escape bit count disagrees";
        return false;
      }
      quant += p.escape_x;  // allowTcq == 0 for parity hiding
      hr_level_avg =
          update_hr_level_avg(p.escape_x, hr_level_avg, lvl_shift);
    }
    if (hidden_dc) quant = 2 * quant + sum_abs1;

    const std::int32_t signed_quant = p.sign ? -quant : quant;
    decoded->quant_levels[static_cast<std::size_t>(p.pos)] = signed_quant;
    if (c == 0 && quant > 0) {
      decoded->dc_category = p.sign ? 1 : 2;
    }
    decoded->cul_level = std::min(4, decoded->cul_level + quant);
  }
  decoded->eob = syntax.eob;
  decoded->is_hidden = is_hidden;
  decoded->sum_abs1 = sum_abs1;
  return true;
}

bool decoded_levels(const BlockSyntax& syntax, const Geometry& geometry,
                    std::span<std::int32_t> levels_scan, std::string* error) {
  const std::size_t area = static_cast<std::size_t>(geometry.w) *
                           static_cast<std::size_t>(geometry.h);
  if (levels_scan.size() < area) {
    if (error != nullptr) *error = "levels span is smaller than w*h";
    return false;
  }
  DecodedBlock decoded;
  if (!decode_syntax(syntax, geometry, &decoded, error)) return false;
  if (decoded.quant_levels.size() != area) {
    if (error != nullptr) *error = "decoded block size mismatch";
    return false;
  }
  std::vector<std::int32_t> scan(area, 0);
  build_scan(geometry.w, geometry.h, geometry.tx_class, scan);
  for (std::size_t c = 0; c < area; ++c) {
    levels_scan[c] = decoded.quant_levels[static_cast<std::size_t>(scan[c])];
  }
  return true;
}

// ---------------------------------------------------------------------------
// Static §9.3 rate model
// ---------------------------------------------------------------------------

namespace {

struct CdfView {
  const std::uint16_t* data = nullptr;
  std::int32_t symbols = 0;
};

CdfView base_cdf_view(const PositionSyntax& p, const Geometry& geometry,
                      const av2enc::still::qctx::CoeffCdfSlice& q) {
  CdfView out;
  const std::int32_t tsc = tx_sz_ctx(geometry.tx_size);
  switch (p.base_kind) {
    case CdfKind::kCoeffBaseEob:
      out.data = &(*q.coeff_base_eob)[tsc][p.base_ctx][0];
      out.symbols = 3;
      break;
    case CdfKind::kCoeffBaseLfEob:
      out.data = &(*q.coeff_base_lf_eob)[tsc][p.base_ctx][0];
      out.symbols = 5;
      break;
    case CdfKind::kCoeffBaseEobUv:
      out.data = &(*q.coeff_base_eob_uv)[p.base_ctx][0];
      out.symbols = 3;
      break;
    case CdfKind::kCoeffBaseLfEobUv:
      out.data = &(*q.coeff_base_lf_eob_uv)[p.base_ctx][0];
      out.symbols = 5;
      break;
    case CdfKind::kCoeffBase:
      out.data = &(*q.coeff_base)[tsc][p.base_ctx][p.base_ctx2][0];
      out.symbols = 4;
      break;
    case CdfKind::kCoeffBaseLf:
      out.data = &(*q.coeff_base_lf)[tsc][p.base_ctx][p.base_ctx2][0];
      out.symbols = 6;
      break;
    case CdfKind::kCoeffBaseUv:
      out.data = &(*q.coeff_base_uv)[p.base_ctx][0];
      out.symbols = 4;
      break;
    case CdfKind::kCoeffBaseLfUv:
      out.data = &(*q.coeff_base_lf_uv)[p.base_ctx][0];
      out.symbols = 6;
      break;
    case CdfKind::kCoeffBasePh:
      out.data = &(*q.coeff_base_ph)[p.base_ctx][0];
      out.symbols = 4;
      break;
    default:
      break;
  }
  return out;
}

CdfView br_cdf_view(const PositionSyntax& p,
                    const av2enc::still::qctx::CoeffCdfSlice& q) {
  CdfView out;
  switch (p.br_kind) {
    case CdfKind::kCoeffBr:
      out.data = &(*q.coeff_br)[p.br_ctx][0];
      out.symbols = 4;
      break;
    case CdfKind::kCoeffBrLf:
      out.data = &(*q.coeff_br_lf)[p.br_ctx][0];
      out.symbols = 4;
      break;
    case CdfKind::kCoeffBrUv:
      out.data = &(*q.coeff_br_uv)[p.br_ctx][0];
      out.symbols = 4;
      break;
    default:
      break;
  }
  return out;
}

CdfView eob_cdf_view(const BlockSyntax& syntax, const Geometry& geometry,
                     const av2enc::still::qctx::CoeffCdfSlice& q) {
  CdfView out;
  switch (syntax.eob_kind) {
    case CdfKind::kEobPt16:
      out.data = &(*q.eob_pt_16)[syntax.eob_ctx][0];
      out.symbols = 5;
      break;
    case CdfKind::kEobPt32:
      out.data = &(*q.eob_pt_32)[syntax.eob_ctx][0];
      out.symbols = 6;
      break;
    case CdfKind::kEobPt64:
      out.data = &(*q.eob_pt_64)[syntax.eob_ctx][0];
      out.symbols = 7;
      break;
    case CdfKind::kEobPt128:
      out.data = &(*q.eob_pt_128)[syntax.eob_ctx][0];
      out.symbols = 8;
      break;
    case CdfKind::kEobPt256:
      out.data = &(*q.eob_pt_256)[syntax.eob_ctx][0];
      out.symbols = 8;
      break;
    case CdfKind::kEobPt512:
      out.data = &(*q.eob_pt_512)[syntax.eob_ctx][0];
      out.symbols = 8;
      break;
    case CdfKind::kEobPt1024:
      out.data = &(*q.eob_pt_1024)[syntax.eob_ctx][0];
      out.symbols = 8;
      break;
    default:
      break;
  }
  (void)geometry;
  return out;
}

double symbol_bits(const std::uint16_t* cdf, std::int32_t n,
                   std::int32_t symbol) {
  if (n <= 1) return 0.0;
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
  if (probability <= 0) return kImpossibleBits;
  return -std::log2(static_cast<double>(probability) / kCdfScale);
}

}  // namespace

double block_rate_bits(const BlockSyntax& syntax, const Geometry& geometry,
                       const RateConfig& config) {
  if (config.qctx < 0 || config.qctx >= av2enc::still::qctx::kSliceCount) {
    return -1.0;
  }
  const av2enc::still::qctx::CoeffCdfSlice& q =
      av2enc::still::qctx::tables_for_slice(config.qctx);
  const std::int32_t tsc = tx_sz_ctx(geometry.tx_size);

  const std::uint16_t* skip_cdf = nullptr;
  if (geometry.plane == 2) {
    if (config.txb_skip_ctx < 0 || config.txb_skip_ctx > 11) return -1.0;
    skip_cdf = &(*q.v_txb_skip)[config.txb_skip_ctx][0];
  } else {
    if (config.txb_skip_ctx < 0 || config.txb_skip_ctx > 9) return -1.0;
    skip_cdf =
        &(*q.txb_skip)[config.is_inter ? 1 : 0][tsc][config.txb_skip_ctx][0];
  }
  double bits = symbol_bits(skip_cdf, 2, syntax.eob == 0 ? 1 : 0);
  if (syntax.eob == 0) return bits;

  const CdfView eob = eob_cdf_view(syntax, geometry, q);
  if (eob.data == nullptr) return -1.0;
  bits += symbol_bits(eob.data, eob.symbols, syntax.eob_symbol);
  bits += static_cast<double>(syntax.eob_escape_bits);
  if (syntax.eob_pt >= 3) {
    bits += symbol_bits(&(*q.eob_extra)[0], 2, syntax.eob_extra_symbol);
    bits += static_cast<double>(syntax.eob_extra_lsb_bits);
  }

  for (const PositionSyntax& p : syntax.positions) {
    const CdfView base = base_cdf_view(p, geometry, q);
    if (base.data == nullptr) return -1.0;
    bits += symbol_bits(base.data, base.symbols, p.base_symbol);
    if (p.br_present) {
      const CdfView br = br_cdf_view(p, q);
      if (br.data == nullptr) return -1.0;
      bits += symbol_bits(br.data, br.symbols, p.br_symbol);
    }
    bits += static_cast<double>(p.escape_bits);
  }

  for (const PositionSyntax& p : syntax.positions) {
    if (!p.sign_present) continue;
    if (p.dc_style_sign) {
      // `dc_sign` uses the neighbour-derived context; `dc_sign_horz_vert`
      // (a HORIZ/VERT coefficient with col/row zero that is not the true DC)
      // always uses context 0 (§5.20.7.27).
      const std::int32_t sign_ctx = (p.pos == 0) ? config.dc_sign_ctx : 0;
      const std::uint16_t* dc_sign =
          &(*q.dc_sign)[0][syntax.is_hidden ? 1 : 0][sign_ctx][0];
      bits += symbol_bits(dc_sign, 2, p.sign);
    } else {
      bits += 1.0;
    }
  }
  return bits;
}

// ---------------------------------------------------------------------------
// Encoder level choice
// ---------------------------------------------------------------------------

namespace {

struct Candidate {
  std::vector<std::int32_t> levels;
  std::int32_t adjusted_scan_index = -1;
  std::int32_t adjusted_delta = 0;
};

struct Evaluation {
  double distortion = std::numeric_limits<double>::infinity();
  double bits = std::numeric_limits<double>::infinity();
  double cost = std::numeric_limits<double>::infinity();
  BlockSyntax syntax;
};

std::int32_t syntax_magnitude(std::int32_t abs_level,
                              const LevelLimits& limits) noexcept {
  return std::min(abs_level, limits.pre_max);
}

std::int32_t contribution(std::int32_t abs_level,
                          const LevelLimits& limits) noexcept {
  return std::min(syntax_magnitude(abs_level, limits), kMaxBaseBrRange) & 1;
}

bool make_delta_level(std::int32_t level, std::int32_t delta,
                      std::int32_t* out) noexcept {
  const std::int32_t mag = abs_i32(level);
  const std::int32_t next = mag + delta;
  if (next < 0) return false;
  *out = level < 0 ? -next : next;
  return true;
}

bool analyze_ac(std::span<const std::int32_t> levels,
                const Geometry& geometry, std::int32_t* eob,
                std::int32_t* num_nz, std::int32_t* sum_abs1,
                std::int32_t* dc_parity) {
  const std::size_t area = static_cast<std::size_t>(geometry.w) *
                           static_cast<std::size_t>(geometry.h);
  if (levels.size() != area) return false;
  std::vector<std::int32_t> scan(area, 0);
  build_scan(geometry.w, geometry.h, geometry.tx_class, scan);
  std::int32_t last = 0;
  for (std::int32_t c = static_cast<std::int32_t>(area) - 1; c >= 0; --c) {
    if (levels[static_cast<std::size_t>(c)] != 0) {
      last = c + 1;
      break;
    }
  }
  std::int32_t sum = 0;
  std::int32_t count = 0;
  for (std::int32_t c = last - 1; c >= 1; --c) {
    const std::int32_t pos = scan[static_cast<std::size_t>(c)];
    const std::int32_t row = pos / geometry.w;
    const std::int32_t col = pos % geometry.w;
    const LevelLimits limits =
        level_limits(row, col, geometry.tx_class, geometry.plane);
    const std::int32_t abs_level =
        abs_i32(levels[static_cast<std::size_t>(c)]);
    sum ^= contribution(abs_level, limits);
    if (abs_level != 0) ++count;
  }
  *eob = last;
  *num_nz = count;
  *sum_abs1 = sum;
  *dc_parity = abs_i32(levels.empty() ? 0 : levels[0]) & 1;
  return true;
}

bool evaluate_candidate(std::span<const double> original,
                        const Candidate& candidate, const Geometry& geometry,
                        const Eligibility& eligibility,
                        const QuantizeConfig& config, Evaluation* out,
                        std::string* error) {
  out->syntax = BlockSyntax{};
  if (!derive_syntax(candidate.levels, geometry, eligibility, &out->syntax,
                     error)) {
    return false;
  }
  DecodedBlock decoded;
  if (!decode_syntax(out->syntax, geometry, &decoded, error)) return false;
  std::vector<std::int32_t> scan(candidate.levels.size(), 0);
  build_scan(geometry.w, geometry.h, geometry.tx_class, scan);
  for (std::size_t c = 0; c < candidate.levels.size(); ++c) {
    const std::int32_t expected = candidate.levels[c];
    const std::int32_t actual =
        decoded.quant_levels[static_cast<std::size_t>(scan[c])];
    if (expected != actual) {
      if (error != nullptr) *error = "encoder/decoder level mismatch";
      return false;
    }
  }
  const auto step_at = [&](std::size_t i) {
    return config.dequant_step.empty() ? config.quant_step
                                       : config.dequant_step[i];
  };
  double distortion = 0.0;
  for (std::size_t i = 0; i < candidate.levels.size(); ++i) {
    const double reconstructed =
        static_cast<double>(candidate.levels[i]) * step_at(i);
    const double err = original[i] - reconstructed;
    distortion += err * err;
  }
  const double bits = block_rate_bits(out->syntax, geometry, config.rate);
  if (bits < 0.0) {
    if (error != nullptr) *error = "rate model rejected the block";
    return false;
  }
  out->distortion = distortion;
  out->bits = bits;
  out->cost = config.mode == QuantizeMode::kMinDistortion
                  ? distortion
                  : distortion + config.lambda * bits;
  return true;
}

}  // namespace

bool quantize_config_valid(const QuantizeConfig& config) noexcept {
  if (!std::isfinite(config.quant_step) || !(config.quant_step > 0.0)) {
    return false;
  }
  if (!std::isfinite(config.lambda) || config.lambda < 0.0) return false;
  if (config.rate.qctx < 0 || config.rate.qctx >= av2enc::still::qctx::kSliceCount) {
    return false;
  }
  if (config.exhaustive_max_area < 0 || config.pruned_ac_candidates < 0) {
    return false;
  }
  return true;
}

bool quantize(std::span<const double> original,
              std::span<const std::int32_t> levels_seed,
              const Geometry& geometry, const QuantizeConfig& config,
              QuantizeResult* result, std::string* error) {
  if (result == nullptr) {
    if (error != nullptr) *error = "null result";
    return false;
  }
  const std::size_t area = static_cast<std::size_t>(geometry.w) *
                           static_cast<std::size_t>(geometry.h);
  if (original.size() != area || levels_seed.size() != area) {
    if (error != nullptr) {
      *error = "input spans must have exactly w*h entries";
    }
    return false;
  }
  if (!quantize_config_valid(config)) {
    if (error != nullptr) *error = "invalid quantize config";
    return false;
  }
  if (!config.dequant_step.empty() && config.dequant_step.size() != area) {
    if (error != nullptr) *error = "dequant_step span must have w*h entries";
    return false;
  }

  *result = QuantizeResult{};
  std::vector<std::int32_t> scan(area, 0);
  build_scan(geometry.w, geometry.h, geometry.tx_class, scan);

  std::int32_t eob = 0;
  std::int32_t num_nz = 0;
  std::int32_t sum_abs1 = 0;
  std::int32_t dc_parity = 0;
  if (!analyze_ac(levels_seed, geometry, &eob, &num_nz, &sum_abs1,
                  &dc_parity)) {
    if (error != nullptr) *error = "seed level analysis failed";
    return false;
  }

  const bool header_ok =
      parity_hiding_eligible(geometry, config.eligibility);
  const bool hidable =
      header_ok && eob > kPhThresh && num_nz >= kPhThresh;

  std::vector<std::int32_t> seed(levels_seed.begin(), levels_seed.end());
  std::vector<Candidate> candidates;

  auto add_candidate = [&](const std::vector<std::int32_t>& levels,
                           std::int32_t scan_index, std::int32_t delta) {
    Candidate candidate;
    candidate.levels = levels;
    candidate.adjusted_scan_index = scan_index;
    candidate.adjusted_delta = delta;
    candidates.push_back(std::move(candidate));
  };

  if (!hidable) {
    add_candidate(seed, -1, 0);
    result->hiding_considered = false;
    result->hidable = false;
  } else {
    const bool baseline_matches = dc_parity == sum_abs1;
    if (baseline_matches) {
      add_candidate(seed, -1, 0);
    } else {
      // DC candidates: exactly one sign of the magnitude step produces the
      // required parity (both are tried; the derivation rejects the other).
      for (std::int32_t delta : {1, -1}) {
        std::int32_t next = 0;
        if (!make_delta_level(seed[0], delta, &next)) continue;
        if ((abs_i32(next) & 1) != sum_abs1) continue;
        std::vector<std::int32_t> levels = seed;
        levels[0] = next;
        add_candidate(levels, 0, delta);
        // A zero DC has no coded sign yet; both signs are legal and can
        // differ in both distortion and the `dc_sign` CDF cost.
        if (seed[0] == 0) {
          levels[0] = -next;
          add_candidate(levels, 0, delta);
        }
      }

      // AC candidates: flip the capped parity of one nonzero AC level.
      struct AcCandidate {
        std::vector<std::int32_t> levels;
        std::int32_t scan_index = 0;
        std::int32_t delta = 0;
        double local_delta = 0.0;
      };
      std::vector<AcCandidate> ac_candidates;
      for (std::int32_t c = eob - 1; c >= 1; --c) {
        const std::int32_t old_level = seed[static_cast<std::size_t>(c)];
        if (old_level == 0) continue;
        const std::int32_t pos = scan[static_cast<std::size_t>(c)];
        const std::int32_t row = pos / geometry.w;
        const std::int32_t col = pos % geometry.w;
        const LevelLimits limits =
            level_limits(row, col, geometry.tx_class, geometry.plane);
        const std::int32_t old_abs = abs_i32(old_level);
        const std::int32_t old_contrib = contribution(old_abs, limits);
        const double step = config.dequant_step.empty()
                                ? config.quant_step
                                : config.dequant_step[static_cast<std::size_t>(c)];
        const double old_recon = static_cast<double>(old_level) * step;
        const double old_err = original[static_cast<std::size_t>(c)] - old_recon;
        const double old_dist = old_err * old_err;
        for (std::int32_t delta : {1, -1}) {
          std::int32_t next = 0;
          if (!make_delta_level(old_level, delta, &next)) continue;
          const std::int32_t new_contrib =
              contribution(abs_i32(next), limits);
          const bool flips = new_contrib != old_contrib;
          // Reducing the fourth AC non-zero to zero drops the block below
          // PHTHRESH and legally turns hiding off even when the capped
          // parity does not change.
          const bool drops_hiding = (next == 0 && num_nz == kPhThresh);
          if (!flips && !drops_hiding) continue;
          const double new_recon = static_cast<double>(next) * step;
          const double new_err =
              original[static_cast<std::size_t>(c)] - new_recon;
          AcCandidate ac;
          ac.levels = seed;
          ac.levels[static_cast<std::size_t>(c)] = next;
          ac.scan_index = c;
          ac.delta = delta;
          ac.local_delta = new_err * new_err - old_dist;
          ac_candidates.push_back(std::move(ac));
        }
      }
      if (area <= static_cast<std::size_t>(config.exhaustive_max_area)) {
        for (AcCandidate& ac : ac_candidates) {
          add_candidate(ac.levels, ac.scan_index, ac.delta);
        }
      } else {
        std::stable_sort(
            ac_candidates.begin(), ac_candidates.end(),
            [](const AcCandidate& a, const AcCandidate& b) {
              if (a.local_delta != b.local_delta) {
                return a.local_delta < b.local_delta;
              }
              return a.scan_index > b.scan_index;
            });
        const std::size_t keep = std::min<std::size_t>(
            ac_candidates.size(),
            static_cast<std::size_t>(config.pruned_ac_candidates));
        for (std::size_t i = 0; i < keep; ++i) {
          add_candidate(ac_candidates[i].levels,
                        ac_candidates[i].scan_index,
                        ac_candidates[i].delta);
        }
      }
    }

  }

  int best = -1;
  Evaluation best_eval;
  std::string candidate_error;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    Evaluation eval;
    candidate_error.clear();
    if (!evaluate_candidate(original, candidates[i], geometry,
                            config.eligibility, config, &eval,
                            &candidate_error)) {
      // A candidate that fails derivation is infeasible (normally: the
      // parity adjustment did not actually satisfy the hidden constraint).
      continue;
    }
    if (best < 0 || eval.cost < best_eval.cost) {
      best = static_cast<int>(i);
      best_eval = eval;
    }
  }
  if (best < 0) {
    if (error != nullptr) {
      *error = candidate_error.empty() ? "no representable candidate"
                                       : candidate_error;
    }
    return false;
  }

  const Candidate& chosen = candidates[static_cast<std::size_t>(best)];
  result->levels = chosen.levels;
  result->syntax = best_eval.syntax;
  result->eob = best_eval.syntax.eob;
  result->hidden = best_eval.syntax.is_hidden;
  result->hiding_considered = hidable;
  result->hidable = hidable;
  result->hiding_turned_off = hidable && !best_eval.syntax.is_hidden;
  result->adjusted_scan_index = chosen.adjusted_scan_index;
  result->adjusted_delta = chosen.adjusted_delta;
  result->distortion = best_eval.distortion;
  result->rate_bits = best_eval.bits;
  result->cost = best_eval.cost;
  return true;
}

bool quantize_scalar(std::span<const double> original,
                     const Geometry& geometry, const QuantizeConfig& config,
                     QuantizeResult* result, std::string* error) {
  const std::size_t area = static_cast<std::size_t>(geometry.w) *
                           static_cast<std::size_t>(geometry.h);
  if (original.size() != area) {
    if (error != nullptr) *error = "input span must have w*h entries";
    return false;
  }
  if (!quantize_config_valid(config)) {
    if (error != nullptr) *error = "invalid quantize config";
    return false;
  }
  if (!config.dequant_step.empty() && config.dequant_step.size() != area) {
    if (error != nullptr) *error = "dequant_step span must have w*h entries";
    return false;
  }
  std::vector<std::int32_t> levels(area, 0);
  for (std::size_t i = 0; i < area; ++i) {
    const double step = config.dequant_step.empty()
                            ? config.quant_step
                            : config.dequant_step[i];
    const double scaled = original[i] / step;
    if (!std::isfinite(scaled)) {
      if (error != nullptr) *error = "non-finite coefficient";
      return false;
    }
    const long long rounded = std::llround(scaled);
    const long long clamped =
        std::max<long long>(-(1LL << 20), std::min<long long>(1LL << 20,
                                                              rounded));
    levels[i] = static_cast<std::int32_t>(clamped);
  }
  return quantize(original, levels, geometry, config, result, error);
}

bool apply_parity_hiding(std::span<const double> original,
                         std::span<std::int32_t> levels,
                         const Geometry& geometry, const QuantizeConfig& config,
                         BlockSyntax* syntax, std::string* error) {
  const std::size_t area = static_cast<std::size_t>(geometry.w) *
                           static_cast<std::size_t>(geometry.h);
  if (levels.size() != area || original.size() != area) {
    if (error != nullptr) *error = "spans must have w*h entries";
    return false;
  }
  const std::vector<std::int32_t> seed(levels.begin(), levels.end());
  QuantizeResult result;
  if (!quantize(original, seed, geometry, config, &result, error)) {
    return false;
  }
  std::copy(result.levels.begin(), result.levels.end(), levels.begin());
  if (syntax != nullptr) *syntax = result.syntax;
  return true;
}

}  // namespace av2enc::tcq::parity


