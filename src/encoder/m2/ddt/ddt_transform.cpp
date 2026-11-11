#include "encoder/m2/ddt/ddt.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "encoder/m2/ddt/ddt_tables.h"

namespace av2enc::m2::ddt {
namespace {

using txv::Tx1dType;

// §4.8 Round2(x, n) = floor((x + 2^(n-1)) / 2^n).  Arithmetic right shift on a
// signed integer is floor division, so this matches the spec for negative x.
inline std::int64_t round2_64(std::int64_t value, int n) {
  if (n <= 0) return value;
  return (value + (std::int64_t{1} << (n - 1))) >> n;
}

// Round2 followed by the spec's Clip3, with the clip applied before any
// narrowing cast (a 32-length kernel sum at a small shift can exceed int32).
inline std::int32_t clip_round2(std::int64_t value, int n, std::int32_t low,
                                std::int32_t high) {
  const std::int64_t rounded = round2_64(value, n);
  if (rounded < low) return low;
  if (rounded > high) return high;
  return static_cast<std::int32_t>(rounded);
}

// Round-to-nearest integer division, symmetric around zero.  Used only by the
// encoder forward path and by the 4096/2896 compensation; the decoder process
// uses round2_64 exclusively.
inline std::int64_t round_div(std::int64_t value, std::int64_t divisor) {
  const std::int64_t half = divisor / 2;
  return value >= 0 ? (value + half) / divisor
                    : -((-value + half) / divisor);
}

int log2_of(int n);
bool kernel_is_flipped(Effective1d type, int n);

// Encoder forward with explicit fractional-bit bookkeeping.  `input_frac_bits`
// and `output_frac_bits` let the 2D pass carry precision between the row and
// column stages (Q-format), which keeps the integer forward scale-compatible
// with txv::forward_block() for 64-coded sizes whose Transform_Shift is small.
bool forward_stage(const std::int64_t* src, int n, int inverse_shift,
                   Effective1d type, int input_frac_bits, int output_frac_bits,
                   std::int64_t* out) {
  if (src == nullptr || out == nullptr) return false;
  const int log2_n = log2_of(n);
  if (log2_n < 0) return false;
  const std::int16_t* kernel = effective_kernel(type, n);
  if (kernel == nullptr) return false;
  const bool flipped = kernel_is_flipped(type, n);
  const int shift =
      12 + log2_n - inverse_shift + input_frac_bits - output_frac_bits;
  for (int r = 0; r < n; ++r) {
    std::int64_t sum = 0;
    for (int i = 0; i < n; ++i) {
      const int sample = flipped ? (n - 1 - i) : i;
      sum += static_cast<std::int64_t>(kernel[r * n + sample]) * src[i];
    }
    if (shift > 0) {
      out[r] = (sum + (std::int64_t{1} << (shift - 1))) >> shift;
    } else if (shift == 0) {
      out[r] = sum;
    } else {
      out[r] = sum << (-shift);
    }
  }
  return true;
}

int log2_of(int n) {
  switch (n) {
    case 4:
      return 2;
    case 8:
      return 3;
    case 16:
      return 4;
    case 32:
      return 5;
    default:
      return -1;
  }
}

// §7.15.4.1 get_identity_scale(log2Sz).
int identity_scale(int log2_sz) {
  switch (log2_sz) {
    case 2:
      return 128;
    case 3:
      return 181;
    case 4:
      return 256;
    default:
      return 362;
  }
}

// True when the kernel's second (sample) index is mirrored for this type.
inline bool kernel_is_flipped(Effective1d type, int n) {
  return type == Effective1d::kFddt && (n == 8 || n == 16);
}

Effective1d from_txv(Tx1dType type) {
  switch (type) {
    case Tx1dType::kDct:
      return Effective1d::kDct;
    case Tx1dType::kIdt:
      return Effective1d::kIdt;
    case Tx1dType::kAdst:
      return Effective1d::kAdst;
    case Tx1dType::kFdst:
    default:
      return Effective1d::kFdst;
  }
}

}  // namespace

const char* effective_1d_name(Effective1d type) {
  switch (type) {
    case Effective1d::kDct:
      return "DCT";
    case Effective1d::kIdt:
      return "IDT";
    case Effective1d::kAdst:
      return "ADST";
    case Effective1d::kFdst:
      return "FDST";
    case Effective1d::kDdtx:
      return "DDTX";
    case Effective1d::kFddt:
      return "FDDT";
  }
  return "?";
}

bool Plan::matches(const Plan& other) const {
  return row == other.row && col == other.col &&
         row_ddt == other.row_ddt && col_ddt == other.col_ddt;
}

Effective1d effective_1d(Effective1d signaled, int sz, const Context& ctx) {
  // §7.15.4.1 get_transform_1d_type(): t is ADST/FDST, sz is the adjusted
  // length, and sz == 4 is the one explicit exclusion.
  if (ctx.enable_inter_ddt && !ctx.use_intrabc && ctx.is_inter &&
      (signaled == Effective1d::kAdst || signaled == Effective1d::kFdst) &&
      sz != 4) {
    return signaled == Effective1d::kAdst ? Effective1d::kDdtx
                                          : Effective1d::kFddt;
  }
  return signaled;
}

Plan make_plan(const TxParams& params, const Context& ctx) {
  Plan plan;
  plan.row = effective_1d(from_txv(static_cast<Tx1dType>(params.row_type)),
                          params.w, ctx);
  plan.col = effective_1d(from_txv(static_cast<Tx1dType>(params.col_type)),
                          params.h, ctx);
  plan.row_ddt = plan.row == Effective1d::kDdtx || plan.row == Effective1d::kFddt;
  plan.col_ddt = plan.col == Effective1d::kDdtx || plan.col == Effective1d::kFddt;
  return plan;
}

const std::int16_t* effective_kernel(Effective1d type, int n) {
  switch (type) {
    case Effective1d::kIdt:
      return nullptr;
    case Effective1d::kDdtx:
    case Effective1d::kFddt:
      // §7.15.2.1: only sz == 8 and sz == 16 test DDTX/FDDT; the sz == 32
      // branch uses Dct_Kernel32, so DDTX/FDDT alias DCT at length 32.
      if (const std::int16_t* kernel = ddt_kernel(n)) return kernel;
      return txv::tx_kernel(Tx1dType::kDct, n);
    case Effective1d::kDct:
      return txv::tx_kernel(Tx1dType::kDct, n);
    case Effective1d::kAdst:
      return txv::tx_kernel(Tx1dType::kAdst, n);
    case Effective1d::kFdst:
    default:
      return txv::tx_kernel(Tx1dType::kFdst, n);
  }
}

bool inverse_1d(const std::int32_t* src, int n, int shift, bool col_tx,
                Effective1d type, int bit_depth, std::int32_t* out) {
  if (src == nullptr || out == nullptr) return false;
  if (type == Effective1d::kIdt) return false;  // identity is not a matrix
  if (log2_of(n) < 0) return false;
  const std::int16_t* kernel = effective_kernel(type, n);
  if (kernel == nullptr) return false;
  const bool flipped = kernel_is_flipped(type, n);
  const int clip_bits = bit_depth + (col_tx ? 0 : 7);
  const std::int32_t low = -(std::int32_t{1} << clip_bits);
  const std::int32_t high = (std::int32_t{1} << clip_bits) - 1;
  for (int i = 0; i < n; ++i) {
    const int sample = flipped ? (n - 1 - i) : i;
    std::int64_t sum = 0;
    for (int j = 0; j < n; ++j) {
      sum += static_cast<std::int64_t>(kernel[j * n + sample]) *
             static_cast<std::int64_t>(src[j]);
    }
    out[i] = clip_round2(sum, shift, low, high);
  }
  return true;
}

bool forward_1d(const std::int32_t* src, int n, int inverse_shift,
                Effective1d type, std::int32_t* out) {
  if (src == nullptr || out == nullptr) return false;
  if (type == Effective1d::kIdt) return false;
  const int log2_n = log2_of(n);
  if (log2_n < 0) return false;
  const std::int16_t* kernel = effective_kernel(type, n);
  if (kernel == nullptr) return false;
  const bool flipped = kernel_is_flipped(type, n);
  // Normalisation of the exact transpose pair: K K^T = (64^2 n) I, so the
  // matched forward shift is log2(64^2 n) - inverse_shift.
  const int shift = 12 + log2_n - inverse_shift;
  if (shift < 0) return false;
  for (int r = 0; r < n; ++r) {
    std::int64_t sum = 0;
    for (int i = 0; i < n; ++i) {
      const int sample = flipped ? (n - 1 - i) : i;
      sum += static_cast<std::int64_t>(kernel[r * n + sample]) *
             static_cast<std::int64_t>(src[i]);
    }
    out[r] = clip_round2(sum, shift, -(std::int32_t{1} << 30),
                         (std::int32_t{1} << 30) - 1);
  }
  return true;
}

bool inverse_block(const std::int32_t* dequant, const TxParams& params,
                   const Context& ctx, std::int16_t* residual_coded) {
  if (dequant == nullptr || residual_coded == nullptr) return false;
  const int w = params.w;
  const int h = params.h;
  if (w <= 0 || h <= 0) return false;
  const Plan plan = make_plan(params, ctx);
  const bool odd_rect = ((params.log2_w ^ params.log2_h) & 1) != 0;
  const int identity_w = identity_scale(params.log2_w);
  const int identity_h = identity_scale(params.log2_h);
  const int row_low = -(std::int32_t{1} << (ctx.bit_depth + 7));
  const int row_high = (std::int32_t{1} << (ctx.bit_depth + 7)) - 1;
  const int col_low = -(std::int32_t{1} << ctx.bit_depth);
  const int col_high = (std::int32_t{1} << ctx.bit_depth) - 1;

  auto prescale = [&](std::int32_t value) -> std::int32_t {
    if (!odd_rect) return value;
    return static_cast<std::int32_t>(
        round2_64(static_cast<std::int64_t>(value) * 2896, 12));
  };

  std::int32_t intermediate[txv::kMaxAdjTxElems] = {};
  std::int32_t column[txv::kMaxAdjTxDim] = {};
  std::int32_t row_in[txv::kMaxAdjTxDim] = {};

  // Row pass (§7.15.4.1).  The odd-log2 rectangular pre-scale is part of the
  // row input, before both the identity and the matrix branches.
  for (int i = 0; i < h; ++i) {
    const int row = i * w;
    for (int j = 0; j < w; ++j) row_in[j] = prescale(dequant[row + j]);
    if (plan.row == Effective1d::kIdt) {
      for (int j = 0; j < w; ++j) {
        const std::int64_t sum =
            static_cast<std::int64_t>(row_in[j]) * identity_w;
        intermediate[row + j] =
            clip_round2(sum, params.row_shift, row_low, row_high);
      }
    } else {
      if (!inverse_1d(row_in, w, params.row_shift, false, plan.row,
                      ctx.bit_depth, &intermediate[row])) {
        return false;
      }
    }
  }

  // Column pass (§7.15.4.1).
  for (int j = 0; j < w; ++j) {
    for (int i = 0; i < h; ++i) column[i] = intermediate[i * w + j];
    if (plan.col == Effective1d::kIdt) {
      for (int i = 0; i < h; ++i) {
        const std::int64_t sum =
            static_cast<std::int64_t>(column[i]) * identity_h;
        residual_coded[i * w + j] = static_cast<std::int16_t>(
            clip_round2(sum, params.col_shift, col_low, col_high));
      }
    } else {
      std::int32_t out[txv::kMaxAdjTxDim] = {};
      if (!inverse_1d(column, h, params.col_shift, true, plan.col,
                      ctx.bit_depth, out)) {
        return false;
      }
      for (int i = 0; i < h; ++i) {
        residual_coded[i * w + j] =
            static_cast<std::int16_t>(out[i]);
      }
    }
  }

  // §7.15.4 adjusted-size expansion by sample duplication.  The source and
  // destination are the same buffer, so walk backwards like txv does.
  if (w != params.coded_w || h != params.coded_h) {
    const int sx = params.coded_w / w;
    const int sy = params.coded_h / h;
    for (int y = params.coded_h - 1; y >= 0; --y) {
      const int ay = y / sy;
      for (int x = params.coded_w - 1; x >= 0; --x) {
        const int ax = x / sx;
        residual_coded[y * params.coded_w + x] =
            residual_coded[ay * w + ax];
      }
    }
  }
  return true;
}

bool forward_block(const std::int16_t* residual_adjusted,
                   const TxParams& params, const Context& ctx,
                   std::int32_t* coeff) {
  if (residual_adjusted == nullptr || coeff == nullptr) return false;
  const int w = params.w;
  const int h = params.h;
  if (w <= 0 || h <= 0) return false;
  const Plan plan = make_plan(params, ctx);
  const int identity_w = identity_scale(params.log2_w);
  const int identity_h = identity_scale(params.log2_h);

  // Eight fractional bits carried between the row and column passes.  This is
  // encoder-only precision management: the spec's inverse has no forward, and
  // without it the small Transform_Shift values of 64-coded sizes lose several
  // bits before the column gain.
  constexpr int kForwardFracBits = 8;
  constexpr std::int64_t kCoeffLimit = std::int64_t{1} << 30;

  std::int64_t temp[txv::kMaxAdjTxElems] = {};
  std::int64_t column[txv::kMaxAdjTxDim] = {};
  std::int64_t row_in[txv::kMaxAdjTxDim] = {};
  std::int64_t out[txv::kMaxAdjTxDim] = {};

  // Row pass: inverse of the decoder's row process, keeping Q8.
  for (int i = 0; i < h; ++i) {
    const int row = i * w;
    for (int j = 0; j < w; ++j) {
      row_in[j] = residual_adjusted[row + j];
    }
    if (plan.row == Effective1d::kIdt) {
      for (int j = 0; j < w; ++j) {
        temp[row + j] = round_div(
            row_in[j] << (params.row_shift + kForwardFracBits), identity_w);
      }
    } else if (!forward_stage(row_in, w, params.row_shift, plan.row, 0,
                              kForwardFracBits, &temp[row])) {
      return false;
    }
  }

  // Column pass: consume the Q8 row output.
  for (int j = 0; j < w; ++j) {
    for (int i = 0; i < h; ++i) column[i] = temp[i * w + j];
    if (plan.col == Effective1d::kIdt) {
      for (int i = 0; i < h; ++i) {
        out[i] = round_div(column[i] << params.col_shift,
                           static_cast<std::int64_t>(identity_h)
                               << kForwardFracBits);
      }
    } else if (!forward_stage(column, h, params.col_shift, plan.col,
                              kForwardFracBits, 0, out)) {
      return false;
    }
    for (int i = 0; i < h; ++i) {
      coeff[i * w + j] = static_cast<std::int32_t>(out[i]);
    }
  }

  const int count = w * h;
  // Compensate §7.15.4.1's odd-log2 rectangular pre-scale
  // (txRowIn = Round2(Dequant * 2896, 12)).
  const bool odd_rect = ((params.log2_w ^ params.log2_h) & 1) != 0;
  for (int i = 0; i < count; ++i) {
    std::int64_t value = coeff[i];
    if (odd_rect) value = round_div(value * 4096, 2896);
    value = std::clamp<std::int64_t>(value, -kCoeffLimit, kCoeffLimit);
    coeff[i] = static_cast<std::int32_t>(value);
  }
  return true;
}

double quant_factor(const TxParams& params) {
  return static_cast<double>(params.dq_denom) / 66.0;
}

std::int32_t level_from_coeff(std::int32_t coeff, const TxParams& params) {
  const std::int64_t value =
      static_cast<std::int64_t>(coeff) * params.dq_denom;
  return static_cast<std::int32_t>(round_div(value, 66));
}

bool plan_has_ddt(const TxParams& params, const Context& ctx) {
  const Plan plan = make_plan(params, ctx);
  // A DDTX/FDDT derived at adjusted length 32 is real per §7.15.4.1 but
  // §7.15.2.1's sz == 32 branch uses Dct_Kernel32 for *every* type, so the
  // reconstruction is identical to the non-DDT ADST/FDST alias.  Report only
  // the lengths where DDT changes the kernel.
  const bool row_changes =
      plan.row_ddt && (params.w == 8 || params.w == 16);
  const bool col_changes =
      plan.col_ddt && (params.h == 8 || params.h == 16);
  return row_changes || col_changes;
}

}  // namespace av2enc::m2::ddt
