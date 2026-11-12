#ifndef AV2ENC_INTER_ADVANCED_WARP_COMMON_H_
#define AV2ENC_INTER_ADVANCED_WARP_COMMON_H_

// Warped motion compensation — AV2 v1.0.0 §7.13.3.19–.24 + §7.13.3.17.
//
// One transcription of the normative arithmetic, compiled for both the CPU
// reference and the CUDA kernel. The warp filter tables come from §9.5 through
// tables/warp_filters.h.
//
// Notes on exactness:
//   * The spec writes `Round2(sx, WARPEDDIFF_PREC_BITS)` on a *signed* value.
//     §4.8's equivalent integer form is `(x + (1 << (n-1))) >> n`, i.e. the
//     floor form (`round2()`), not `Round2Signed()`. The offsets below use
//     `round2()` deliberately.
//   * `motion_vector_scaling()` and `setup_shear()` use `Round2Signed()`.
//   * Every tap product fits in int32: the horizontal pass input is 8 bits and
//     the table sums to 128, so `s <= 128 * 255`; the vertical pass sees
//     `Round2(s, 3) <= 4080`, so `s <= 128 * 4080`.

#include "advanced_common.h"

namespace av2enc::inter::adv {

// ---------------------------------------------------------------------------
// §7.13.3.22 / §7.13.3.19 helper tables.
// ---------------------------------------------------------------------------

inline constexpr int kWarpedFilterRows = tables::kWarpedFilterRows;
inline constexpr int kWarpedFilterTaps = tables::kWarpedFilterTaps;
inline constexpr int kExtWarpFilterRows = tables::kExtWarpedFilterRows;

// Zero-phase row of Warped_Filters: 3 * WARPEDPIXEL_PREC_SHIFTS.
inline constexpr int kWarpZeroPhase = 3 * kWarpedPixelPrecShifts;

struct WarpParams {
  int p[6] = {0, 0, 1 << kWarpedModelPrecBits, 0, 0, 1 << kWarpedModelPrecBits};

  bool identity() const {
    return p[0] == 0 && p[1] == 0 && p[2] == (1 << kWarpedModelPrecBits) && p[3] == 0 &&
           p[4] == 0 && p[5] == (1 << kWarpedModelPrecBits);
  }
};

struct ShearParams {
  bool valid = false;
  int alpha = 0;
  int beta = 0;
  int gamma = 0;
  int delta = 0;
};

// ---------------------------------------------------------------------------
// §7.13.3.21 setup shear.
// ---------------------------------------------------------------------------

AV2ENC_ADV_HD inline ShearParams setup_shear(const WarpParams& params) {
  const int max_value = 32767 - (1 << (kWarpParamReduceBits - 1));
  const int one = 1 << kWarpedModelPrecBits;

  // All of these products can exceed 32 bits for a wild (but legal) input
  // model, so accumulate in 64 bits and clamp once, as the spec's Clip3() does.
  auto clip64_to_int = [](std::int64_t v) {
    return static_cast<int>(v > 2147483647 ? 2147483647 : (v < -2147483648 ? -2147483648 : v));
  };
  const int alpha0 = clip3(-32768, max_value, clip64_to_int(std::int64_t{params.p[2]} - one));
  const int beta0 = clip3(-32768, max_value, params.p[3]);
  const DivScaleShift ds = resolve_divisor(params.p[2] > 0 ? params.p[2] : 1);
  const std::int64_t v = std::int64_t{params.p[4]} << kWarpedModelPrecBits;
  const int gamma0 =
      clip3(-32768, max_value, clip64_to_int(round2_signed64(v * ds.factor, ds.shift)));
  const std::int64_t w = std::int64_t{params.p[3]} * params.p[4];
  const int delta0 = clip3(
      -32768, max_value,
      clip64_to_int(std::int64_t{params.p[5]} - round2_signed64(w * ds.factor, ds.shift) - one));

  ShearParams out{};
  out.alpha = round2_signed(alpha0, kWarpParamReduceBits) << kWarpParamReduceBits;
  out.beta = round2_signed(beta0, kWarpParamReduceBits) << kWarpParamReduceBits;
  out.gamma = round2_signed(gamma0, kWarpParamReduceBits) << kWarpParamReduceBits;
  out.delta = round2_signed(delta0, kWarpParamReduceBits) << kWarpParamReduceBits;

  bool valid = true;
  if (4 * abs_int(out.alpha) + 7 * abs_int(out.beta) >= (3 << kWarpedModelPrecBits)) {
    valid = false;
  }
  if (4 * abs_int(out.gamma) + 4 * abs_int(out.delta) >= (3 << kWarpedModelPrecBits)) {
    valid = false;
  }
  out.valid = valid;
  return out;
}

// ---------------------------------------------------------------------------
// §7.13.3.23 reduce_warp_model.
// ---------------------------------------------------------------------------

AV2ENC_ADV_HD inline WarpParams reduce_warp_model(const WarpParams& params) {
  const int max_value = (1 << (kWarpedModelPrecBits - 1)) - (1 << kWarpParamReduceBits);
  const int min_value = -max_value;
  WarpParams out = params;
  out.p[0] = params.p[0];
  out.p[1] = params.p[1];
  for (int i = 2; i < 6; ++i) {
    const int offset = (i == 2 || i == 5) ? (1 << kWarpedModelPrecBits) : 0;
    const std::int64_t widened = std::int64_t{params.p[i]} - offset;
    const int original = static_cast<int>(
        widened > 2147483647LL ? 2147483647LL
                               : (widened < -2147483648LL ? -2147483648LL : widened));
    const int clamped = clip3(min_value, max_value, original);
    const int rounded = round2_signed(clamped, kWarpParamReduceBits) << kWarpParamReduceBits;
    out.p[i] = rounded + offset;
  }
  return out;
}

// ---------------------------------------------------------------------------
// §7.13.3.23 get_warp_translation.
// `mid_x`/`mid_y` are luma samples; `block_mv` is in 1/8 sample units.
// Writes p[0] (row translation) and p[1] (col translation).
// ---------------------------------------------------------------------------

AV2ENC_ADV_HD inline void get_warp_translation(WarpParams& params, int mid_x, int mid_y,
                                               const int block_mv[2]) {
  const int one = 1 << kWarpedModelPrecBits;
  const int mv_x = block_mv[1];
  const int mv_y = block_mv[0];
  const std::int64_t vx =
      std::int64_t{mv_x} * (1LL << (kWarpedModelPrecBits - 3)) -
      (std::int64_t{mid_x} * (std::int64_t{params.p[2]} - one) + std::int64_t{mid_y} * params.p[3]);
  const std::int64_t vy =
      std::int64_t{mv_y} * (1LL << (kWarpedModelPrecBits - 3)) -
      (std::int64_t{mid_x} * params.p[4] + std::int64_t{mid_y} * (std::int64_t{params.p[5]} - one));
  const int clamp_high = kWarpedModelTransClamp - (1 << kWarpParamReduceBits);
  auto clamp_trans = [&](std::int64_t v) {
    const std::int64_t clamped =
        v > clamp_high ? clamp_high : (v < -kWarpedModelTransClamp ? -kWarpedModelTransClamp : v);
    return static_cast<int>(clamped);
  };
  params.p[0] = clamp_trans(vx);
  params.p[1] = clamp_trans(vy);
}

// ---------------------------------------------------------------------------
// §7.13.3.23 warp estimation.
//
// A `WarpSample` is one candidate correspondence in 1/8 luma sample units
// (`src_*` is the current-frame position, `dst_*` the reference position). The
// caller builds these from the §7.12.3 candidate list and the block MVs; this
// component does not own the candidate list.
// ---------------------------------------------------------------------------

struct WarpSample {
  int src_x = 0;
  int src_y = 0;
  int dst_x = 0;
  int dst_y = 0;
};

AV2ENC_ADV_HD inline int ls_product(int a, int b) { return ((a * b) >> 2) + (a + b); }

// `mid_x`/`mid_y` are the block centre in luma samples
// (MiCol*4 + w4*2 - 1, MiRow*4 + h4*2 - 1). `block_mv` is 1/8 sample units.
AV2ENC_ADV_HD inline WarpParams warp_estimate(const WarpSample* samples, int count, int mid_x,
                                              int mid_y, const int block_mv[2]) {
  int a00 = 0;
  int a01 = 0;
  int a11 = 0;
  int bx0 = 0;
  int bx1 = 0;
  int by0 = 0;
  int by1 = 0;

  const int suy = mid_y * 8;
  const int sux = mid_x * 8;
  const int duy = suy + block_mv[0];
  const int dux = sux + block_mv[1];

  for (int i = 0; i < count; ++i) {
    const int sy = samples[i].src_y - suy;
    const int sx = samples[i].src_x - sux;
    const int dy = samples[i].dst_y - duy;
    const int dx = samples[i].dst_x - dux;
    if (abs_int(sx - dx) < kLsMvMax && abs_int(sy - dy) < kLsMvMax) {
      a00 += ls_product(sx, sx) + 8;
      a01 += ls_product(sx, sy) + 4;
      a11 += ls_product(sy, sy) + 8;
      bx0 += ls_product(sx, dx) + 8;
      bx1 += ls_product(sy, dx) + 4;
      by0 += ls_product(sx, dy) + 4;
      by1 += ls_product(sy, dy) + 8;
    }
  }

  WarpParams params{};
  const std::int64_t det =
      static_cast<std::int64_t>(a00) * a11 - static_cast<std::int64_t>(a01) * a01;
  if (det == 0) {
    params.p[2] = 1 << kWarpedModelPrecBits;
    params.p[3] = 0;
    params.p[4] = 0;
    params.p[5] = 1 << kWarpedModelPrecBits;
    get_warp_translation(params, mid_x, mid_y, block_mv);
    return params;
  }

  DivScaleShift ds = resolve_divisor(static_cast<int>(det));
  int div_shift = ds.shift - kWarpedModelPrecBits;
  int div_factor = ds.factor;
  if (div_shift < 0) {
    div_factor <<= -div_shift;
    div_shift = 0;
  }

  auto diag64 = [&](std::int64_t v) -> int {
    // diag(v) = Clip3(INT32MIN, INT32MAX, Round2Signed(v * divFactor, divShift))
    const std::int64_t prod = v * div_factor;
    const std::int64_t rounded = round2_signed64(prod, div_shift);
    const std::int64_t clamped =
        rounded > 2147483647 ? 2147483647 : (rounded < -2147483648 ? -2147483648 : rounded);
    return static_cast<int>(clamped);
  };

  params.p[2] = diag64(static_cast<std::int64_t>(a11) * bx0 - static_cast<std::int64_t>(a01) * bx1);
  params.p[3] =
      diag64(static_cast<std::int64_t>(-a01) * bx0 + static_cast<std::int64_t>(a00) * bx1);
  params.p[4] = diag64(static_cast<std::int64_t>(a11) * by0 - static_cast<std::int64_t>(a01) * by1);
  params.p[5] =
      diag64(static_cast<std::int64_t>(-a01) * by0 + static_cast<std::int64_t>(a00) * by1);
  params = reduce_warp_model(params);
  get_warp_translation(params, mid_x, mid_y, block_mv);
  return params;
}

// ---------------------------------------------------------------------------
// §7.13.3.17 motion vector scaling. Outputs are in 1/1024 sample units.
// ---------------------------------------------------------------------------

struct MvScaleOutput {
  int start_x = 0;
  int start_y = 0;
  int step_x = 0;
  int step_y = 0;
};

// `prescaled == 0` means the MV is in 1/8 sample units, otherwise 1/16.
// `ref_w`/`ref_h` are the reference dimensions, `frame_w`/`frame_h` the current
// frame dimensions (both in luma samples).
AV2ENC_ADV_HD inline MvScaleOutput motion_vector_scaling(int /*plane*/, int sub_x, int sub_y, int x,
                                                         int y, const int mv[2], bool prescaled,
                                                         int ref_w, int ref_h, int frame_w,
                                                         int frame_h) {
  const int x_scale = ((ref_w << kRefScaleShift) + frame_w / 2) / frame_w;
  const int y_scale = ((ref_h << kRefScaleShift) + frame_h / 2) / frame_h;
  const int half_sample = 1 << (kSubpelBits - 1);
  int orig_x;
  int orig_y;
  if (prescaled) {
    orig_x = (x << kSubpelBits) + round2_signed(mv[1], sub_x) + half_sample;
    orig_y = (y << kSubpelBits) + round2_signed(mv[0], sub_y) + half_sample;
  } else {
    orig_x = (x << kSubpelBits) + ((2 * mv[1]) >> sub_x) + half_sample;
    orig_y = (y << kSubpelBits) + ((2 * mv[0]) >> sub_y) + half_sample;
  }
  const int base_x = orig_x * x_scale - (half_sample << kRefScaleShift);
  const int base_y = orig_y * y_scale - (half_sample << kRefScaleShift);
  const int off = (1 << (kScaleSubpelBits - kSubpelBits)) / 2;

  MvScaleOutput out{};
  out.start_x = round2_signed(base_x, kRefScaleShift + kSubpelBits - kScaleSubpelBits) + off;
  out.start_y = round2_signed(base_y, kRefScaleShift + kSubpelBits - kScaleSubpelBits) + off;
  out.step_x = round2_signed(x_scale, kRefScaleShift - kScaleSubpelBits);
  out.step_y = round2_signed(y_scale, kRefScaleShift - kScaleSubpelBits);
  return out;
}

AV2ENC_ADV_HD inline bool is_scaled_ref(int ref_w, int ref_h, int frame_w, int frame_h) {
  const int x_scale = ((ref_w << kRefScaleShift) + frame_w / 2) / frame_w;
  const int y_scale = ((ref_h << kRefScaleShift) + frame_h / 2) / frame_h;
  return x_scale != (1 << kRefScaleShift) || y_scale != (1 << kRefScaleShift);
}

// ---------------------------------------------------------------------------
// §7.13.3.20 helper get_sub_block_warp_mv (rnd = 0 form).
// ---------------------------------------------------------------------------

AV2ENC_ADV_HD inline void get_sub_block_warp_mv(const WarpParams& params, int /*plane*/, int sub_x,
                                                int sub_y, int x, int y, int w, int h, int rnd,
                                                int mv[2]) {
  const int src_x = (x + (w >> 1)) << sub_x;
  const int src_y = (y + (h >> 1)) << sub_y;
  const int dst_x = params.p[2] * src_x + params.p[3] * src_y + params.p[0];
  const int dst_y = params.p[4] * src_x + params.p[5] * src_y + params.p[1];
  const int shift = kWarpedModelPrecBits - 3;
  if (rnd) {
    mv[0] = round2_signed(dst_y - (src_y << kWarpedModelPrecBits), shift);
    mv[1] = round2_signed(dst_x - (src_x << kWarpedModelPrecBits), shift);
  } else {
    mv[0] = (dst_y - (src_y << kWarpedModelPrecBits)) >> shift;
    mv[1] = (dst_x - (src_x << kWarpedModelPrecBits)) >> shift;
  }
  mv[0] = clip3(kMvLow + 1, kMvUpp - 1, mv[0]);
  mv[1] = clip3(kMvLow + 1, kMvUpp - 1, mv[1]);
}

// §5.20.9.x clamp_mv_row / clamp_mv_col (MV_BORDER = 128, mi units of 4).
AV2ENC_ADV_HD inline int clamp_mv_row(int mvec, int mi_row, int bh4, int mi_rows) {
  const int low = -(mi_row + bh4) * kMiSize * 8 - 128;
  const int high = (mi_rows - mi_row) * kMiSize * 8 + 128;
  return clip3(low, high, mvec);
}

AV2ENC_ADV_HD inline int clamp_mv_col(int mvec, int mi_col, int bw4, int mi_cols) {
  const int low = -(mi_col + bw4) * kMiSize * 8 - 128;
  const int high = (mi_cols - mi_col) * kMiSize * 8 + 128;
  return clip3(low, high, mvec);
}

// ---------------------------------------------------------------------------
// §7.13.3.19 block warp, 8x8 section.
//
// `ref.max_x`/`ref.max_y` are the spec's `lastX`/`lastY`
// (`(RefMiCols * MI_SIZE >> subX) - 1`), i.e. the *padded* reference bounds.
// `x`/`y` are the block origin in `plane` samples; `i8`/`j8` pick the 8x8
// section. `block_w`/`block_h` clip the written section to the block.
// `inter_round1` is 7 for a compound prediction and 11 otherwise.
//
// `intermediate` must hold 15 * 8 ints. It is a parameter so the CUDA kernel can
// place it in shared memory instead of per-thread local memory.
// ---------------------------------------------------------------------------

inline constexpr int kBlockWarpRows = 15;
inline constexpr int kBlockWarpCols = 8;

struct BlockWarpState {
  ShearParams shear{};
  int ix4 = 0;
  int iy4 = 0;
  int sx4 = 0;
  int sy4 = 0;
};

// Projects the 8x8 section centre and derives the phase used by both passes.
AV2ENC_ADV_HD inline bool block_warp_prepare(const WarpParams& params, int x, int y, int i8, int j8,
                                             int sub_x, int sub_y, BlockWarpState* state) {
  const ShearParams shear = setup_shear(params);
  if (!shear.valid) {
    return false;
  }
  const int src_x = (x + j8 * 8 + 4) << sub_x;
  const int src_y = (y + i8 * 8 + 4) << sub_y;
  const int dst_x = params.p[2] * src_x + params.p[3] * src_y + params.p[0];
  const int dst_y = params.p[4] * src_x + params.p[5] * src_y + params.p[1];
  const int x4 = dst_x >> sub_x;
  const int y4 = dst_y >> sub_y;
  state->shear = shear;
  state->ix4 = x4 >> kWarpedModelPrecBits;
  state->sx4 = x4 & ((1 << kWarpedModelPrecBits) - 1);
  state->iy4 = y4 >> kWarpedModelPrecBits;
  state->sy4 = y4 & ((1 << kWarpedModelPrecBits) - 1);
  return true;
}

// One intermediate sample (horizontal pass). `intermediate` layout is
// [i1 + 7][i2 + 4] with row stride kBlockWarpCols.
AV2ENC_ADV_HD inline std::int32_t block_warp_intermediate_value(const PlaneView& ref,
                                                                const BlockWarpState& state, int i1,
                                                                int i2) {
  const ShearParams& shear = state.shear;
  const int ref_row = clip3(0, ref.max_y, state.iy4 + i1);
  const std::uint8_t* ref_line = ref.data + static_cast<std::ptrdiff_t>(ref_row) * ref.stride;
  const int sx = state.sx4 + shear.alpha * i2 + shear.beta * i1;
  const int offs = round2(sx, kWarpedDiffPrecBits) + kWarpZeroPhase;
  const int taps = offs >= 0 && offs < kWarpedFilterRows ? offs : kWarpZeroPhase;
  int s = 0;
  for (int i3 = 0; i3 < kWarpedFilterTaps; ++i3) {
    const int col = clip3(0, ref.max_x, state.ix4 + i2 - 3 + i3);
    s += tables::kWarpedFilters[taps][i3] * ref_line[col];
  }
  return round2(s, kInterRound0);
}

// Horizontal pass: fills `intermediate` (15 * 8 ints, row stride 8).
AV2ENC_ADV_HD inline void block_warp_intermediate(const PlaneView& ref, const BlockWarpState& state,
                                                  std::int32_t* intermediate) {
  for (int i1 = -7; i1 < 8; ++i1) {
    for (int i2 = -4; i2 < 4; ++i2) {
      intermediate[(i1 + 7) * kBlockWarpCols + (i2 + 4)] =
          block_warp_intermediate_value(ref, state, i1, i2);
    }
  }
}

// One output sample (vertical pass).
AV2ENC_ADV_HD inline std::int16_t block_warp_store_value(const BlockWarpState& state, int i1,
                                                         int i2, const std::int32_t* intermediate,
                                                         int inter_round1, bool clip_output) {
  const ShearParams& shear = state.shear;
  const int sy = state.sy4 + shear.gamma * i2 + shear.delta * i1;
  const int offs = round2(sy, kWarpedDiffPrecBits) + kWarpZeroPhase;
  const int taps = offs >= 0 && offs < kWarpedFilterRows ? offs : kWarpZeroPhase;
  int s = 0;
  for (int i3 = 0; i3 < kWarpedFilterTaps; ++i3) {
    s += tables::kWarpedFilters[taps][i3] * intermediate[(i1 + i3 + 4) * kBlockWarpCols + (i2 + 4)];
  }
  const int value = round2(s, inter_round1);
  return static_cast<std::int16_t>(clip_output ? clip1(value, 8) : value);
}

// Vertical pass: writes the 8x8 section, clipped to the block, at
// (out_x0 + local_x, out_y0 + local_y).
AV2ENC_ADV_HD inline void block_warp_store(const BlockWarpState& state, int i8, int j8, int block_w,
                                           int block_h, int inter_round1, bool clip_output,
                                           int out_x0, int out_y0, const std::int32_t* intermediate,
                                           std::int16_t* dst, int dst_stride) {
  for (int i1 = -4; i1 < 4; ++i1) {
    const int local_y = i8 * 8 + i1 + 4;
    if (local_y < 0 || local_y >= block_h) {
      continue;
    }
    std::int16_t* dst_row =
        dst + static_cast<std::ptrdiff_t>(out_y0 + local_y) * dst_stride + out_x0;
    for (int i2 = -4; i2 < 4; ++i2) {
      const int local_x = j8 * 8 + i2 + 4;
      if (local_x < 0 || local_x >= block_w) {
        continue;
      }
      dst_row[local_x] =
          block_warp_store_value(state, i1, i2, intermediate, inter_round1, clip_output);
    }
  }
}

AV2ENC_ADV_HD inline void block_warp_predict(const WarpParams& params, const PlaneView& ref, int x,
                                             int y, int i8, int j8, int block_w, int block_h,
                                             int sub_x, int sub_y, int inter_round1,
                                             bool clip_output, int out_x0, int out_y0,
                                             std::int32_t* intermediate, std::int16_t* dst,
                                             int dst_stride) {
  BlockWarpState state{};
  if (!block_warp_prepare(params, x, y, i8, j8, sub_x, sub_y, &state)) {
    return;
  }
  block_warp_intermediate(ref, state, intermediate);
  block_warp_store(state, i8, j8, block_w, block_h, inter_round1, clip_output, out_x0, out_y0,
                   intermediate, dst, dst_stride);
}

// ---------------------------------------------------------------------------
// §7.13.3.20 extended block warp, 4x4 section.
//
// `intermediate` must hold `ext_warp_intermediate_rows(...) * 4` ints.
// `ref_rows`/`ref_cols` select the lastX/lastY bounds in the reference plane.
// ---------------------------------------------------------------------------

inline constexpr int kExtWarpRows = 64;  // scaled 16x refs need <= 3*16+6
inline constexpr int kExtWarpCols = 4;

AV2ENC_ADV_HD inline int ext_warp_intermediate_rows(int y4, int step_y) {
  const int iy4 = y4 >> kWarpedModelPrecBits;
  return ((y4 + step_y * 3) >> kWarpedModelPrecBits) - iy4 + kExtWarpTaps;
}

struct ExtWarpContext {
  int plane = 0;
  int sub_x = 0;
  int sub_y = 0;
  int block_w = 0;
  int block_h = 0;
  int mi_col = 0;
  int mi_row = 0;
  int bw4 = 0;
  int bh4 = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  bool scaled = false;
  int ref_w = 0;
  int ref_h = 0;
  int frame_w = 0;
  int frame_h = 0;
  int inter_round1 = kInterRound1Single;
  int dst_x = 0;  // destination origin; may differ from the model origin
  int dst_y = 0;
  bool clip_output = true;
};

// Per-section state for the extended warp. `prepare` resolves the projected
// centre, the scaled-reference step and the reference bounding box; the two
// passes then consume it. Splitting them lets the CUDA kernel share the
// intermediate rows across a whole block.
struct ExtWarpSection {
  bool scaled = false;
  int first_x = 0;
  int first_y = 0;
  int bbox_last_x = 0;
  int bbox_last_y = 0;
  int ix4 = 0;
  int sx4 = 0;
  int iy4 = 0;
  int sy4 = 0;
  int out_x4 = 0;
  int out_y4 = 0;
  int step_x = 0;
  int step_y = 0;
  int rows = 0;
};

// Returns false when the section does not fit the intermediate buffer.
AV2ENC_ADV_HD inline bool ext_warp_prepare(const WarpParams& params, const PlaneView& ref,
                                           const ExtWarpContext& ctx, int x, int y, int i4, int j4,
                                           ExtWarpSection* out) {
  const int last_x = ref.max_x;
  const int last_y = ref.max_y;
  const int i8 = i4 >> 1;
  const int j8 = j4 >> 1;
  const int bbox_w = ctx.block_w < 8 ? ctx.block_w : 8;
  const int bbox_h = ctx.block_h < 8 ? ctx.block_h : 8;
  const int sub_x = ctx.sub_x;
  const int sub_y = ctx.sub_y;

  int mv[2] = {0, 0};
  get_sub_block_warp_mv(params, ctx.plane, sub_x, sub_y, x + j8 * 8, y + i8 * 8, bbox_w, bbox_h, 0,
                        mv);
  mv[0] = clamp_mv_row(mv[0], ctx.mi_row, ctx.bh4, ctx.mi_rows);
  mv[1] = clamp_mv_col(mv[1], ctx.mi_col, ctx.bw4, ctx.mi_cols);

  const MvScaleOutput scale =
      motion_vector_scaling(ctx.plane, sub_x, sub_y, x + j8 * 8, y + i8 * 8, mv, false, ctx.ref_w,
                            ctx.ref_h, ctx.frame_w, ctx.frame_h);

  int first_x = scale.start_x >> kScaleSubpelBits;
  int first_y = scale.start_y >> kScaleSubpelBits;
  first_x = clip3(0, last_x, first_x - 3);
  first_y = clip3(0, last_y, first_y - 3);
  const int bbox_last_x =
      clip3(0, last_x, ((scale.start_x + scale.step_x * (bbox_w - 1)) >> kScaleSubpelBits) + 4);
  const int bbox_last_y =
      clip3(0, last_y, ((scale.start_y + scale.step_y * (bbox_h - 1)) >> kScaleSubpelBits) + 4);

  const int src_x = (x + j4 * 4 + 2) << sub_x;
  const int src_y = (y + i4 * 4 + 2) << sub_y;
  const int dst_x = params.p[2] * src_x + params.p[3] * src_y + params.p[0];
  const int dst_y = params.p[4] * src_x + params.p[5] * src_y + params.p[1];

  out->scaled = ctx.scaled;
  out->first_x = first_x;
  out->first_y = first_y;
  out->bbox_last_x = bbox_last_x;
  out->bbox_last_y = bbox_last_y;
  out->out_x4 = dst_x >> sub_x;
  out->out_y4 = dst_y >> sub_y;
  out->step_x = 0;
  out->step_y = 0;

  if (ctx.scaled) {
    const int x_scale = ((ctx.ref_w << kRefScaleShift) + ctx.frame_w / 2) / ctx.frame_w;
    const int y_scale = ((ctx.ref_h << kRefScaleShift) + ctx.frame_h / 2) / ctx.frame_h;
    out->out_x4 -= 2 << kWarpedModelPrecBits;
    out->out_y4 -= 2 << kWarpedModelPrecBits;
    out->out_x4 = round2_signed(out->out_x4 * x_scale, kRefScaleShift);
    out->out_y4 = round2_signed(out->out_y4 * y_scale, kRefScaleShift);
    out->step_x = round2_signed(x_scale, kRefScaleShift - kScaleSubpelBits)
                  << (kWarpedModelPrecBits - kScaleSubpelBits);
    out->step_y = round2_signed(y_scale, kRefScaleShift - kScaleSubpelBits)
                  << (kWarpedModelPrecBits - kScaleSubpelBits);
  }

  out->iy4 = out->out_y4 >> kWarpedModelPrecBits;
  out->sy4 = out->out_y4 & ((1 << kWarpedModelPrecBits) - 1);
  out->ix4 = out->out_x4 >> kWarpedModelPrecBits;
  out->sx4 = out->out_x4 & ((1 << kWarpedModelPrecBits) - 1);
  out->rows = ctx.scaled ? ext_warp_intermediate_rows(out->out_y4, out->step_y) : 9;
  return out->rows > 0 && out->rows <= kExtWarpRows;
}

// One intermediate sample of the extended warp. `idx` enumerates
// k * 4 + l: for the unscaled path k in [-4, 5), l in [-2, 2); for the scaled
// path k in [0, rows), l in [0, 4).
AV2ENC_ADV_HD inline std::int32_t ext_warp_intermediate_value(const PlaneView& ref,
                                                              const ExtWarpSection& sec, int idx) {
  if (!sec.scaled) {
    const int k = idx / 4 - 4;
    const int l = idx % 4 - 2;
    const int offs_x = clip3(0, kExtWarpFilterRows - 1, round2(sec.sx4, kExtWarpRoundBits));
    const int int_y = clip3(sec.first_y, sec.bbox_last_y, sec.iy4 + k);
    const std::uint8_t* ref_line = ref.data + static_cast<std::ptrdiff_t>(int_y) * ref.stride;
    int s = 0;
    for (int m = 0; m < kExtWarpTaps; ++m) {
      const int col = clip3(sec.first_x, sec.bbox_last_x, sec.ix4 + l - 2 + m);
      s += tables::kExtWarpedFilters[offs_x][m] * ref_line[col];
    }
    return round2(s, kInterRound0);
  }
  const int k = idx / 4;
  const int l = idx % 4;
  const int int_y = clip3(sec.first_y, sec.bbox_last_y, sec.iy4 + k - 2);
  const std::uint8_t* ref_line = ref.data + static_cast<std::ptrdiff_t>(int_y) * ref.stride;
  const int pos = sec.out_x4 + sec.step_x * l;
  const int ix4 = pos >> kWarpedModelPrecBits;
  const int sx4 = pos & ((1 << kWarpedModelPrecBits) - 1);
  const int offs = clip3(0, kExtWarpFilterRows - 1, round2(sx4, kExtWarpRoundBits));
  int s = 0;
  for (int m = 0; m < kExtWarpTaps; ++m) {
    const int col = clip3(sec.first_x, sec.bbox_last_x, ix4 - 2 + m);
    s += tables::kExtWarpedFilters[offs][m] * ref_line[col];
  }
  return round2(s, kInterRound0);
}

// Horizontal pass. Writes `rows * 4` ints, row stride 4.
AV2ENC_ADV_HD inline void ext_warp_intermediate(const PlaneView& ref, const ExtWarpSection& sec,
                                                std::int32_t* intermediate) {
  for (int idx = 0; idx < sec.rows * kExtWarpCols; ++idx) {
    intermediate[idx] = ext_warp_intermediate_value(ref, sec, idx);
  }
}

// One output sample of the vertical pass. `k`/`l` are the section-local indices:
// unscaled k in [-2, 2), l in [-2, 2); scaled k in [0, 4), l in [0, 4).
AV2ENC_ADV_HD inline std::int16_t ext_warp_store_value(const ExtWarpSection& sec, int k, int l,
                                                       const std::int32_t* intermediate,
                                                       int inter_round1, bool clip_output) {
  int value;
  if (!sec.scaled) {
    const int offs_y = clip3(0, kExtWarpFilterRows - 1, round2(sec.sy4, kExtWarpRoundBits));
    int s = 0;
    for (int m = 0; m < kExtWarpTaps; ++m) {
      s +=
          tables::kExtWarpedFilters[offs_y][m] * intermediate[(k + m + 2) * kExtWarpCols + (l + 2)];
    }
    value = round2(s, inter_round1);
  } else {
    const int pos = sec.out_y4 + sec.step_y * k;
    const int iy_off = (pos >> kWarpedModelPrecBits) - sec.iy4;
    const int sy4 = pos & ((1 << kWarpedModelPrecBits) - 1);
    const int offs = clip3(0, kExtWarpFilterRows - 1, round2(sy4, kExtWarpRoundBits));
    int s = 0;
    for (int m = 0; m < kExtWarpTaps; ++m) {
      s += tables::kExtWarpedFilters[offs][m] * intermediate[(iy_off + m) * kExtWarpCols + l];
    }
    value = round2(s, inter_round1);
  }
  return static_cast<std::int16_t>(clip_output ? clip1(value, 8) : value);
}

// Vertical pass: writes the 4x4 section at (ctx.dst_x + j4*4, ctx.dst_y + i4*4).
AV2ENC_ADV_HD inline void ext_warp_store(const ExtWarpContext& ctx, const ExtWarpSection& sec,
                                         int i4, int j4, const std::int32_t* intermediate,
                                         std::int16_t* dst, int dst_stride) {
  const int k_begin = sec.scaled ? 0 : -2;
  const int l_begin = sec.scaled ? 0 : -2;
  for (int k = k_begin; k < k_begin + 4; ++k) {
    const int local_y = sec.scaled ? i4 * 4 + k : i4 * 4 + k + 2;
    if (local_y < 0 || local_y >= ctx.block_h) {
      continue;
    }
    for (int l = l_begin; l < l_begin + 4; ++l) {
      const int local_x = sec.scaled ? j4 * 4 + l : j4 * 4 + l + 2;
      if (local_x < 0 || local_x >= ctx.block_w) {
        continue;
      }
      dst[static_cast<std::ptrdiff_t>(ctx.dst_y + local_y) * dst_stride + ctx.dst_x + local_x] =
          ext_warp_store_value(sec, k, l, intermediate, ctx.inter_round1, ctx.clip_output);
    }
  }
}

AV2ENC_ADV_HD inline bool ext_block_warp_predict(const WarpParams& params, const PlaneView& ref,
                                                 const ExtWarpContext& ctx, int x, int y, int i4,
                                                 int j4, std::int32_t* intermediate,
                                                 std::int16_t* dst, int dst_stride) {
  ExtWarpSection sec{};
  if (!ext_warp_prepare(params, ref, ctx, x, y, i4, j4, &sec)) {
    return false;
  }
  ext_warp_intermediate(ref, sec, intermediate);
  ext_warp_store(ctx, sec, i4, j4, intermediate, dst, dst_stride);
  return true;
}

}  // namespace av2enc::inter::adv

#endif  // AV2ENC_INTER_ADVANCED_WARP_COMMON_H_
