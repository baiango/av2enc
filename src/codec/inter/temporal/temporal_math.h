// AV2 v1.0.0 temporal inter tools — shared integer arithmetic and projection.
//
// This header is standard-library-free and header-only so the same
// transcription of the spec runs on the CPU reference and (later) in a CUDA
// translation unit.  It reuses the §4.8 helpers from
// src/encoder/inter/advanced/advanced_common.h instead of duplicating them.
//
// Normative reference: AV2 v1.0.0 §7.9 (motion field estimation), §7.10 (TIP
// motion field), §7.13.3.8–.12 (OPFL), §4.8 (rounding).

#ifndef AV2ENC_ENCODER_INTER_TEMPORAL_TEMPORAL_MATH_H_
#define AV2ENC_ENCODER_INTER_TEMPORAL_TEMPORAL_MATH_H_

#include <cstdint>

#include "codec/inter/advanced/advanced_common.h"
#include "codec/inter/temporal/temporal_types.h"

namespace av2enc::inter::temporal {

using adv::abs_int;
using adv::clip3;
using adv::floor_log2;
using adv::get_relative_dist;
using adv::round2;
using adv::round2_signed;
using adv::round2_signed64;

// §4.8 Clip1 for the 8-bit reference path.
inline constexpr int clip1(int value) { return clip3(0, 255, value); }

// GetMsb(x) = FloorLog2(x) for x > 0, 0 for x == 0.  (The spec only calls it
// with non-negative values; the zero case appears for |suv| and |suw|.)
inline constexpr int get_msb(std::uint32_t value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}
inline constexpr int get_msb_signed(std::int32_t value) {
  return get_msb(static_cast<std::uint32_t>(value < 0 ? -value : value));
}

// §7.9.4 Div_Mult[32].
inline constexpr std::int32_t kDivMult[32] = {
    0,    16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638,
    1489, 1365,  1260, 1170, 1092, 1024, 963,  910,  862,  819,  780,
    744,  712,   682,  655,  630,  606,  585,  564,  546,  528};

// §7.9.4 get_mv_projection().  `numerator` and `denominator` are frame
// distances; the denominator is clipped to MAX_FRAME_DISTANCE and the
// numerator to ±MAX_FRAME_DISTANCE.  The 64-bit intermediate is required: the
// full 17-bit MV range times 31 times 16384 exceeds int32.
inline Mv get_mv_projection(Mv mv, int numerator, int denominator) {
  // The spec only applies Min(MAX_FRAME_DISTANCE, denominator); frame
  // distances are non-negative.  Clamp the low end too so a malformed
  // negative denominator cannot index Div_Mult out of bounds.
  const int clipped_denominator =
      denominator < 0 ? 0
                      : (denominator < kMaxFrameDistance ? denominator
                                                         : kMaxFrameDistance);
  const int clipped_numerator =
      clip3(-kMaxFrameDistance, kMaxFrameDistance, numerator);
  const std::int64_t scale = static_cast<std::int64_t>(kDivMult[clipped_denominator]);
  Mv out;
  out.row = static_cast<std::int32_t>(clip3(
      kMvLow + 1, kMvUpp - 1,
      static_cast<int>(round2_signed64(static_cast<std::int64_t>(mv.row) *
                                           clipped_numerator * scale,
                                       14))));
  out.col = static_cast<std::int32_t>(clip3(
      kMvLow + 1, kMvUpp - 1,
      static_cast<int>(round2_signed64(static_cast<std::int64_t>(mv.col) *
                                           clipped_numerator * scale,
                                       14))));
  return out;
}

// §7.9.5 get_mv_projection_clamp().
inline Mv get_mv_projection_clamp(Mv mv, int numerator, int denominator) {
  Mv out = get_mv_projection(mv, numerator, denominator);
  out.row = clip3(-kRefMvsLimit, kRefMvsLimit, out.row);
  out.col = clip3(-kRefMvsLimit, kRefMvsLimit, out.col);
  return out;
}

// §7.22 compression_mv(): pack an MV component into fewer bits for storage.
inline std::int16_t compression_mv(std::int32_t v) {
  const std::int32_t a = v < 0 ? -v : v;
  const int step_log2 = get_msb(static_cast<std::uint32_t>(a)) > 4
                            ? get_msb(static_cast<std::uint32_t>(a)) - 4
                            : 0;
  const std::int32_t c = (a >> step_log2) + (step_log2 << 4);
  return static_cast<std::int16_t>(v < 0 ? -c : c);
}

// §7.22 uncompression_mv().
inline std::int32_t uncompression_mv(std::int16_t v) {
  const std::int32_t c = v < 0 ? -static_cast<std::int32_t>(v)
                               : static_cast<std::int32_t>(v);
  const int step_log2 = (c >> 4) > 1 ? (c >> 4) - 1 : 0;
  const std::int32_t a = (c - (step_log2 << 4)) << step_log2;
  return v < 0 ? -a : a;
}

// §7.9.2 get_tmvp_shift().
inline constexpr int tmvp_shift(bool sb_64x64, int proj_step) {
  return (sb_64x64 || proj_step == 1) ? 3 : 4;
}
inline constexpr int tmvp_unit(int x8, bool sb_64x64, int proj_step) {
  return x8 >> tmvp_shift(sb_64x64, proj_step);
}
inline constexpr int tmvp_phase(int x8, bool sb_64x64, int proj_step) {
  return tmvp_unit(x8, sb_64x64, proj_step) % 3;
}
// §7.9.2 tmvp_avail(): both positions must be in the same TMVP unit.
inline constexpr bool tmvp_avail(int base8, int loc8, int max8, bool sb_64x64,
                                 int proj_step) {
  if (loc8 >= max8) {
    return false;
  }
  return tmvp_unit(base8, sb_64x64, proj_step) ==
         tmvp_unit(loc8, sb_64x64, proj_step);
}

// §7.9.7 project_no_constraint(), returning posValid through the caller.
inline int project_no_constraint(int v8, int32_t delta, int dst_sign, int max8,
                                 bool* pos_valid) {
  int32_t offset8;
  if (delta >= 0) {
    offset8 = delta >> (3 + 1 + kMiSizeLog2);
  } else {
    offset8 = -((-delta) >> (3 + 1 + kMiSizeLog2));
  }
  v8 += dst_sign * offset8;
  if (v8 < 0 || v8 >= max8) {
    *pos_valid = false;
  }
  return v8;
}

struct SampledPosition {
  bool valid = false;
  int x8 = 0;
  int y8 = 0;
};

// §7.9.6 get_sampled_position().
inline SampledPosition get_sampled_position(int x8, int y8, int dst_sign,
                                            Mv proj_mv, int mi_cols,
                                            int mi_rows, int proj_step) {
  SampledPosition out;
  out.valid = true;
  const int y = project_no_constraint(y8, proj_mv.row, dst_sign, mi_rows >> 1,
                                      &out.valid);
  const int x = project_no_constraint(x8, proj_mv.col, dst_sign, mi_cols >> 1,
                                      &out.valid);
  out.x8 = x;
  out.y8 = y;
  if (proj_step == 2) {
    out.x8 -= out.x8 & 1;
    out.y8 -= out.y8 & 1;
  }
  return out;
}

// §7.9.8 check_block_position().  Returns the updated posValid.
inline bool check_block_position(bool pos_valid, int base_x8, int base_y8,
                                 int pos_x8, int pos_y8, bool sb_64x64,
                                 int proj_step) {
  if (!pos_valid) {
    return false;
  }
  const int shift = tmvp_shift(sb_64x64, proj_step);
  for (int ord = 0; ord < 2; ++ord) {
    const int v8 = ord != 0 ? base_x8 : base_y8;
    const int sb_off8 = 1 << shift;
    const int max_off8 = proj_step > 1 ? (ord != 0 ? sb_off8 : 0)
                                       : (ord != 0 ? (sb_off8 >> 1) : 0);
    const int base8 = (v8 >> shift) << shift;
    const int pos8 = ord != 0 ? pos_x8 : pos_y8;
    if (pos8 < base8 - max_off8 || pos8 >= base8 + sb_off8 + max_off8) {
      return false;
    }
  }
  return true;
}

// TIP weight from `tip_global_wtd_index` (§7.13.3.1).  Returns false for an
// out-of-range index.
inline bool tip_weight(int index, int* weight) {
  if (index < 0 || index > 7) {
    return false;
  }
  *weight = kTipWeightingFactor[index];
  return true;
}

// §7.22/§7.13.3.2 get_tip_offsets().
inline TipOffsets get_tip_offsets(const TipConfig& cfg) {
  TipOffsets out;
  if (cfg.num_future_refs > 0 && cfg.num_past_refs > 0) {
    out.ref_offset =
        get_relative_dist(cfg.closest_future_order_hint,
                          cfg.closest_past_order_hint);
  } else {
    out.ref_offset =
        get_relative_dist(cfg.closest_past_order_hint,
                          cfg.closest_future_order_hint);
  }
  out.past_offset =
      get_relative_dist(cfg.order_hint, cfg.closest_past_order_hint);
  out.future_offset =
      get_relative_dist(cfg.order_hint, cfg.closest_future_order_hint);
  out.ref_offset = out.ref_offset < kMaxFrameDistance ? out.ref_offset
                                                      : kMaxFrameDistance;
  return out;
}

// §7.10.5 calc_avg().
inline int calc_avg(int n, int d) {
  if (d == 1) {
    return n;
  }
  if (d == 2) {
    return round2_signed(n, 1);
  }
  if (d == 3) {
    return round2_signed(n * 85, 8);
  }
  return round2_signed(n, 2);
}

// §7.13.3.9 distance reduction for OPFL.  Returns false when the process
// terminates immediately (either distance is zero).
inline bool opfl_scale_distances(int d0, int d1, int out[2]) {
  if (d0 == 0 || d1 == 0) {
    return false;
  }
  if (abs_int(d0) == abs_int(d1)) {
    out[0] = d0 < 0 ? -1 : 1;
    out[1] = d1 < 0 ? -1 : 1;
  } else if (abs_int(d0) > abs_int(d1)) {
    out[0] = d0 < 0 ? -2 : 2;
    out[1] = d1 < 0 ? -1 : 1;
  } else {
    out[0] = d0 < 0 ? -1 : 1;
    out[1] = d1 < 0 ? -2 : 2;
  }
  return true;
}

// §7.13.3.12 divide_and_round_array().  `den` must be > 0.
inline void divide_and_round_array(const int sol[2], int den, int shift,
                                   int result[2]) {
  int inv_den;
  int den_shift;
  if (den == 1) {
    inv_den = 1;
    den_shift = 0;
  } else {
    const adv::DivScaleShift ds = adv::resolve_divisor(den);
    den_shift = ds.shift;
    inv_den = ds.factor;
  }
  const int inv_den_msb = get_msb(static_cast<std::uint32_t>(inv_den < 0 ? -inv_den : inv_den));
  for (int i = 0; i < 2; ++i) {
    result[i] = 0;
    if (sol[i] == 0) {
      continue;
    }
    const bool sgn = sol[i] > 0;
    int tmp = sgn ? sol[i] : -sol[i];
    const int raw_red_bits =
        get_msb(static_cast<std::uint32_t>(tmp)) + inv_den_msb + 4 - kMaxLsBits;
    const int num_red_bits = raw_red_bits > 0 ? raw_red_bits : 0;
    if (num_red_bits > 0) {
      tmp = round2_signed(tmp, num_red_bits);
    }
    const int inc_bits = shift + num_red_bits - den_shift;
    std::int64_t value;
    if (inc_bits <= -31) {
      tmp = round2_signed(tmp, -inc_bits - 30);
      const std::int64_t mult = static_cast<std::int64_t>(tmp) * inv_den;
      value = round2_signed64(mult, 30);
    } else {
      const std::int64_t mult = static_cast<std::int64_t>(tmp) * inv_den;
      if (inc_bits >= 0) {
        value = mult << inc_bits;
      } else {
        value = round2_signed64(mult, -inc_bits);
      }
    }
    result[i] = sgn ? static_cast<int>(value) : -static_cast<int>(value);
  }
}

}  // namespace av2enc::inter::temporal

#endif  // AV2ENC_ENCODER_INTER_TEMPORAL_TEMPORAL_MATH_H_
