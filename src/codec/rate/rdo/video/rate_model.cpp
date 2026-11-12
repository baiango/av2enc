#include "codec/rate/rdo/video/rate_model.h"

#include "codec/rate/decide/chroma_rdo.h"
#include "codec/rate/decide/intra_rdo.h"
#include "pipeline/video/video_cdf_tables.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace av2enc::video::rdo {
namespace {

int floor_log2_int(int value) {
  int log = 0;
  while (value > 1) {
    value >>= 1;
    ++log;
  }
  return log;
}

// Bit count of write_ns_bits(n, value) from §5.20.7.20 (literal bits only).
int ns_bit_count(std::uint32_t n, std::uint32_t value) {
  if (n <= 1) return 0;
  unsigned width = 0;
  while ((1U << width) < n) ++width;
  const std::uint32_t m = (1U << width) - n;
  return value < m ? static_cast<int>(width - 1U)
                   : static_cast<int>(width);
}

}  // namespace

double InterRateModel::cdf_bits(const std::uint16_t* cdf, int n, int symbol) {
  if (cdf == nullptr || n <= 1 || symbol < 0 || symbol >= n) {
    return symbol == 0 ? 0.0 : 16.0;
  }
  double probability = 0.0;
  constexpr double kScale = 32768.0;
  if (symbol == 0) {
    probability = static_cast<double>(cdf[0]) / kScale;
  } else if (symbol < n - 1) {
    probability = (static_cast<double>(cdf[symbol]) -
                   static_cast<double>(cdf[symbol - 1])) /
                  kScale;
  } else {
    probability = (kScale - static_cast<double>(cdf[n - 2])) / kScale;
  }
  if (!(probability > 0.0)) return 16.0;
  return -std::log2(probability);
}

double InterRateModel::default_is_inter_bits(int ctx, bool is_inter) {
  const int clamped = std::clamp(ctx, 0, 3);
  return cdf_bits(&kIsInterCdf[clamped][0], 2, is_inter ? 1 : 0);
}

double InterRateModel::default_skip_flag_bits(int ctx, bool skip_flag) {
  const int clamped = std::clamp(ctx, 0, 5);
  return cdf_bits(&kSkipCdf[clamped][0], 2, skip_flag ? 1 : 0);
}

double InterRateModel::default_single_mode_bits(int ctx, int single_mode) {
  if (single_mode < 0 || single_mode > 2) return 16.0;
  const int clamped = std::clamp(ctx, 0, 4);
  return cdf_bits(&kSingleModeCdf[clamped][0], 3, single_mode);
}

double InterRateModel::default_drl_bits(int ctx, int drl_index,
                                        int max_drl_bits) {
  const int m = std::max(1, max_drl_bits);
  if (drl_index < 0 || drl_index > m) return 16.0;
  const int clamped_ctx = std::clamp(ctx, 0, 4);
  double bits = 0.0;
  for (int idx = 0; idx < m; ++idx) {
    const int mode = idx < drl_index ? 1 : 0;
    bits += cdf_bits(&kDrlModeCdf[std::min(idx, 2)][clamped_ctx][0], 2, mode);
    if (mode == 0) break;
  }
  return bits;
}

double InterRateModel::default_mv_bits(int mv_y, int mv_x) {
  // Transcribed from video_syntax.cpp write_mv() (§5.20.7.20, MvPrecision = 1/8).
  const std::int32_t ay = std::abs(mv_y);
  const std::int32_t ax = std::abs(mv_x);
  const std::int32_t shell = ay + ax;
  if (shell >= (1 << 17)) return 64.0;
  const int cls = shell == 0 ? 0 : floor_log2_int(static_cast<int>(shell));
  const std::int32_t base = cls == 0 ? 0 : (1 << cls);
  const std::int32_t off = shell - base;

  const int setsym = cls >= 8 ? 1 : 0;
  const int class_sym = cls >= 8 ? cls - 8 : cls;
  if (class_sym < 0 || class_sym > 7) return 64.0;
  double bits = 0.0;
  bits += cdf_bits(&kJointShellSetCdf[0], 2, setsym);
  bits += cdf_bits(setsym ? &kJointShell6Class1Cdf[0]
                          : &kJointShell6Class0Cdf[0],
                   8, class_sym);
  if (setsym && class_sym == 7) {
    const int last = cls - 15;
    if (last < 0 || last > 1) return 64.0;
    bits += cdf_bits(&kJointShellLastTwoClassesCdf[0], 2, last);
  }

  if (cls < 2) {
    bits += cdf_bits(&kShellOffsetLowClassCdf[cls][0], 2,
                     static_cast<int>(off));
  } else if (cls == 2) {
    if (off == 0) {
      bits += cdf_bits(&kShellOffsetClass2Cdf[0], 2, 0);
    } else {
      bits += cdf_bits(&kShellOffsetClass2Cdf[0], 2, 1);
      if (off == 1) {
        bits += 1.0;
      } else {
        bits += 1.0 + 1.0;
      }
    }
  } else {
    if (cls > 15) return 64.0;
    for (int i = 0; i < cls; ++i) {
      bits += cdf_bits(&kShellOffsetOtherClassCdf[i][0], 2,
                       static_cast<int>((off >> i) & 1));
    }
  }

  if (shell > 0) {
    const std::int32_t max_pair = shell >> 1;
    bool skip_col_bit = false;
    std::int32_t col = 0;
    if (ax == max_pair && ((shell & 1) == 0)) {
      skip_col_bit = true;
      col = max_pair;
    } else if (ax <= max_pair) {
      col = ax;
    } else {
      col = ay;
    }

    if (max_pair > 0) {
      const int max_bits =
          std::min<std::int32_t>(max_pair, 2);
      const std::int32_t prefix_col = std::min<std::int32_t>(col, 2);
      for (int i = 0; i < max_bits; ++i) {
        const int greater = prefix_col > i ? 1 : 0;
        bits += cdf_bits(&kColMvGreaterCdf[i][0], 2, greater);
        if (greater == 0) break;
      }
      if (max_pair > 2 && prefix_col == 2) {
        bits += ns_bit_count(static_cast<std::uint32_t>(max_pair - 1),
                             static_cast<std::uint32_t>(col - 2));
      }
    }

    if (!skip_col_bit) {
      const int index = ax <= max_pair ? 0 : 1;
      const int row = std::min(cls, 3);
      bits += cdf_bits(&kColMvIndexCdf[row][0], 2, index);
    }
  }

  if (mv_y != 0) bits += 1.0;
  if (mv_x != 0) bits += 1.0;
  return bits;
}

double InterRateModel::default_mode_bits(void* context,
                                         const ModeRateInput& input) {
  (void)context;
  const RateContext& ctx = input.context;
  double bits = default_is_inter_bits(ctx.is_inter_ctx, input.is_inter);
  if (!input.is_inter) return bits;

  bits += default_skip_flag_bits(ctx.skip_ctx, input.skip_flag);
  const int single_mode = input.inter_mode == 0
                              ? 1
                              : (input.inter_mode == 1 ? 0 : 2);
  bits += default_single_mode_bits(ctx.new_mv_ctx, single_mode);
  if (input.inter_mode == 1 || input.inter_mode == 2) {
    const int drl = input.inter_mode == 2 ? 0 : input.drl_index;
    bits += default_drl_bits(ctx.new_mv_ctx, drl,
                             ctx.max_drl_bits_minus_1 + 1);
  }
  if (input.inter_mode == 2) {
    bits += default_mv_bits(input.mvd.y, input.mvd.x);
  }
  return bits;
}

double InterRateModel::default_residual_bits(
    void* context, const residual::BlockDecision& decision) {
  (void)context;
  return decision.skip ? decision.skip_rate_bits : decision.rate_bits;
}

double InterRateModel::default_intra_syntax_bits(void* context, int y_mode,
                                                 int y_angle_delta,
                                                 int uv_mode,
                                                 int is_inter_ctx) {
  (void)context;
  double bits = default_is_inter_bits(is_inter_ctx, false);
  bits += av2enc::still::rdo_decide::intra_syntax_bits(y_mode, y_angle_delta);
  bits += av2enc::still::rdo_decide::default_chroma_syntax_bits(
      false, uv_mode, 0);
  return bits;
}

double InterRateModel::mode_bits(const ModeRateInput& input) const {
  if (callbacks_.mode_bits != nullptr) {
    return callbacks_.mode_bits(callbacks_.context, input);
  }
  double bits = default_mode_bits(callbacks_.context, input);
  if (callbacks_.ref_bits != nullptr && input.is_inter) {
    bits += callbacks_.ref_bits(callbacks_.context, input.ref);
  }
  return bits;
}

double InterRateModel::residual_bits(
    const residual::BlockDecision& decision) const {
  if (callbacks_.residual_bits != nullptr) {
    return callbacks_.residual_bits(callbacks_.context, decision);
  }
  return default_residual_bits(callbacks_.context, decision);
}

double InterRateModel::intra_syntax_bits(int y_mode, int y_angle_delta,
                                         int uv_mode,
                                         int is_inter_ctx) const {
  if (callbacks_.intra_syntax_bits != nullptr) {
    return callbacks_.intra_syntax_bits(callbacks_.context, y_mode,
                                        y_angle_delta, uv_mode,
                                        is_inter_ctx);
  }
  return default_intra_syntax_bits(callbacks_.context, y_mode, y_angle_delta,
                                   uv_mode, is_inter_ctx);
}

}  // namespace av2enc::video::rdo
