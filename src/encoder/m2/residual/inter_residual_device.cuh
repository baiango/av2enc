// Device-side inter-residual helpers.
//
// This header is compiled twice: with nvcc for the real kernels and, with the
// CUDA keywords stubbed, by `inter_residual_emulation_test.cpp` on the host.
// That is how `src/cuda/mrl/mrl_device.cuh` is validated, and it keeps the CPU
// reference and the device path on the same integer code.
//
// What lives here is the part of the component a GPU implementation adds: the
// residual gather, the adjusted-size EOB search, the §8.3.2 coefficient context
// plan, and the skip-vs-residual cost comparison.  The forward/inverse
// transforms and quantisers are *not* reimplemented: a real kernel calls
// `src/cuda/tx_variants/` (`txv_forward_transform`, `txv_quantize`,
// `txv_inverse_transform`), whose CPU/GPU parity is already gated by that
// component.  See docs/INTEGRATION.md for the exact launch sequence.
#ifndef AV2ENC_M2_RESIDUAL_INTER_RESIDUAL_DEVICE_CUH_
#define AV2ENC_M2_RESIDUAL_INTER_RESIDUAL_DEVICE_CUH_

#include <cstdint>

#include "encoder/m2/residual/inter_coeff_plan_types.h"

namespace av2enc::m2::residual::dev {

__device__ __forceinline__ int ir_clip1(int value) {
  return value < 0 ? 0 : (value > 255 ? 255 : value);
}

__device__ __forceinline__ int ir_floor_log2(int value) {
  int log = 0;
  while (value > 1) {
    value >>= 1;
    ++log;
  }
  return log;
}

__device__ __forceinline__ int ir_abs(int value) {
  return value < 0 ? -value : value;
}

// residual_coded[y*w + x] = clip-residual(source - prediction), int16.
// Same signed 8-bit difference the CPU path stores (`src - pred`, both 0..255,
// so the range is [-255, 255] and always fits int16).
__device__ __forceinline__ void ir_residual_coded(const std::uint8_t* source,
                                                  const std::uint8_t* pred,
                                                  int w, int h,
                                                  std::int16_t* residual) {
  const int count = w * h;
  for (int i = 0; i < count; ++i) {
    residual[i] = static_cast<std::int16_t>(
        static_cast<int>(source[i]) - static_cast<int>(pred[i]));
  }
}

// Scan-order EOB from an adjusted-size raster level array.  `scan[c]` is the
// raster position of scan position c (see m2::syntax::build_scan).
__device__ __forceinline__ int ir_eob_scan(const std::int32_t* levels_raster,
                                           const std::uint16_t* scan,
                                           int area) {
  for (int c = area - 1; c >= 0; --c) {
    if (levels_raster[scan[c]] != 0) return c + 1;
  }
  return 0;
}

// §5.20.7.23 get_lf_limits() for TX_CLASS_2D.
__device__ __forceinline__ int ir_is_lf(int row, int col, int plane) {
  return (row + col) < (plane == 0 ? 4 : 1);
}

// §5.20.7.27 base/br/sign/escape split for one plane.
//
// `levels_scan[c]` is the signed level at scan position c (c < eob).
// `scan[c]` is the raster position of scan position c.
// `level_pre` is scratch of `adj_w*adj_h` ints (zeroed by this function).
// `plan` receives one entry per scan position [0, eob).
//
// Mirrors build_coeff_plan() in inter_coeff_syntax.cpp; the emulation test
// compares them entry by entry.
__device__ __forceinline__ void ir_build_coeff_plan(
    int plane, int eob, int adj_w, int adj_h, const std::int32_t* levels_scan,
    const std::uint16_t* scan, int dc_sign_ctx, CoeffPlanEntry* plan,
    int* level_pre) {
  const int area = adj_w * adj_h;
  for (int i = 0; i < area; ++i) level_pre[i] = 0;
  // Sig_Ref_Diff_Offset[TX_CLASS_2D].
  const int sig_ref[5][2] = {{0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
  // Mag_Ref_Offset_With_Tx_Class[TX_CLASS_2D].
  const int mag_ref[3][2] = {{0, 1}, {1, 0}, {1, 1}};
  constexpr int num_base_levels = 2;
  constexpr int coeff_base_range = 3;
  constexpr int lf_num_base_levels = 4;
  constexpr int max_base_br_range = 6;

  for (int c = eob - 1; c >= 0; --c) {
    const int pos = scan[c];
    const int row = pos / adj_w;
    const int col = pos % adj_w;
    const int abs_value = ir_abs(levels_scan[c]);
    const int lf = ir_is_lf(row, col, plane);
    const int base_levels = lf ? lf_num_base_levels : num_base_levels;
    const int base_max = base_levels + 1;
    const int br_max = (lf && plane > 0) ? 0 : coeff_base_range;
    const int pre_max = base_max + br_max;

    int base_level = 0;
    int br_level = 0;
    if (abs_value <= base_levels) {
      base_level = abs_value;
    } else {
      base_level = base_levels + 1 < base_max ? base_levels + 1 : base_max;
      const int capped = abs_value < pre_max ? abs_value : pre_max;
      br_level = capped - base_level;
    }

    int eob_ctx = 0;
    if (c != 0) {
      if (c <= area / 8) {
        eob_ctx = 1;
      } else if (c <= area / 4) {
        eob_ctx = 2;
      } else {
        eob_ctx = 3;
      }
    }

    int base_ctx = eob_ctx;
    if (c != eob - 1) {
      int mag = 0;
      const int num_ref = plane == 0 ? 5 : 3;
      for (int idx = 0; idx < num_ref; ++idx) {
        const int rr = row + sig_ref[idx][0];
        const int cc = col + sig_ref[idx][1];
        if (rr < adj_h && cc < adj_w) {
          const int limit = lf ? 5 : 3;
          const int value = level_pre[rr * adj_w + cc];
          mag += value < limit ? value : limit;
        }
      }
      const int sig_ctx = (mag + 1) >> 1;
      if (plane == 0) {
        if (lf) {
          if (c == 0) {
            base_ctx = sig_ctx < 8 ? sig_ctx : 8;
          } else if (row + col < 2) {
            base_ctx = (sig_ctx < 6 ? sig_ctx : 6) + 9;
          } else {
            base_ctx = (sig_ctx < 4 ? sig_ctx : 4) + 16;
          }
        } else {
          const int ctx2 = sig_ctx < 4 ? sig_ctx : 4;
          if (row + col < 6) {
            base_ctx = ctx2;
          } else if (row + col < 8) {
            base_ctx = ctx2 + 5;
          } else {
            base_ctx = ctx2 + 10;
          }
        }
      } else {
        const int ctx2 = sig_ctx < 3 ? sig_ctx : 3;
        base_ctx = plane == 1 ? ctx2 : ctx2 + 4;
      }
    }

    const int pre_level = base_level + br_level;
    const int codes_br = (pre_level > base_levels && !(lf && plane > 0)) ? 1 : 0;
    int br_ctx = 0;
    if (codes_br != 0) {
      int mag = 0;
      for (int idx = 0; idx < 3; ++idx) {
        const int rr = row + mag_ref[idx][0];
        const int cc = col + mag_ref[idx][1];
        if (rr < adj_h && cc < adj_w) {
          const int value = level_pre[rr * adj_w + cc];
          const int limit = max_base_br_range - 1;
          mag += value < limit ? value : limit;
        }
      }
      const int halved = (mag + 1) >> 1;
      mag = halved < max_base_br_range ? halved : max_base_br_range;
      if (plane == 0) {
        br_ctx = lf ? mag + 7 : mag;
      } else {
        br_ctx = mag < 3 ? mag : 3;
      }
    }

    level_pre[pos] = pre_level;

    CoeffPlanEntry& entry = plan[c];
    entry = CoeffPlanEntry{};
    entry.base_level = static_cast<std::uint8_t>(base_level);
    entry.br_level = static_cast<std::uint8_t>(br_level);
    entry.is_eob = c == eob - 1 ? 1 : 0;
    entry.is_lf = lf != 0 ? 1 : 0;
    entry.codes_br = static_cast<std::uint8_t>(codes_br);
    entry.base_ctx = static_cast<std::int16_t>(base_ctx);
    entry.br_ctx = static_cast<std::int16_t>(br_ctx);
    entry.eob_ctx = static_cast<std::int16_t>(eob_ctx);
    entry.pos = pos;
    entry.abs_level = abs_value;
    entry.pre_level = pre_level;
    entry.escape_delta = abs_value - pre_level;
  }

  for (int c = eob - 1; c >= 0; --c) {
    CoeffPlanEntry& entry = plan[c];
    const int row = entry.pos / adj_w;
    const int col = entry.pos % adj_w;
    const int level = levels_scan[c];
    entry.sign = level < 0 ? 1 : 0;
    entry.codes_dc_sign = (row == 0 && col == 0 && plane == 0) ? 1 : 0;
    entry.dc_sign_ctx = static_cast<std::int16_t>(dc_sign_ctx);
    int max_level = 0;
    if (entry.is_lf != 0) {
      max_level = plane == 0 ? lf_num_base_levels + coeff_base_range + 1
                             : lf_num_base_levels + 1;
    } else {
      max_level = num_base_levels + coeff_base_range + 1;
    }
    entry.codes_escape =
        (entry.abs_level >= max_level && entry.pre_level != 0) ? 1 : 0;
  }
}

// §8.3.2 dc_sign context from the (already coded) AboveDc/LeftDc grids.  The
// arrays are the plane's 4x4 grids; `x4`/`y4` are the block origin.
__device__ __forceinline__ int ir_dc_sign_context(const std::uint8_t* above_dc,
                                                  const std::uint8_t* left_dc,
                                                  int x4, int y4, int w4,
                                                  int h4, int max_x4,
                                                  int max_y4) {
  int dc_sign = 0;
  for (int k = 0; k < w4; ++k) {
    if (x4 + k < max_x4) {
      const int sign = above_dc[x4 + k];
      if (sign == 1) {
        --dc_sign;
      } else if (sign == 2) {
        ++dc_sign;
      }
    }
  }
  for (int k = 0; k < h4; ++k) {
    if (y4 + k < max_y4) {
      const int sign = left_dc[y4 + k];
      if (sign == 1) {
        --dc_sign;
      } else if (sign == 2) {
        ++dc_sign;
      }
    }
  }
  if (dc_sign < 0) return 1;
  if (dc_sign > 0) return 2;
  return 0;
}

// reconstruction = clip(prediction + residual) and its SSE against the source.
__device__ __forceinline__ double ir_reconstruct(const std::uint8_t* source,
                                                 const std::uint8_t* pred,
                                                 const std::int16_t* residual,
                                                 int w, int h,
                                                 std::uint8_t* recon) {
  const int count = w * h;
  double sse = 0.0;
  for (int i = 0; i < count; ++i) {
    const int value = ir_clip1(static_cast<int>(pred[i]) +
                               static_cast<int>(residual[i]));
    recon[i] = static_cast<std::uint8_t>(value);
    const double d =
        static_cast<double>(source[i]) - static_cast<double>(value);
    sse += d * d;
  }
  return sse;
}

// Lagrangian cost; lambda <= 0 selects the caller's derived value.
__device__ __forceinline__ double ir_cost(double distortion, double rate_bits,
                                          double lambda) {
  return distortion + lambda * rate_bits;
}

// Skip if the residual cannot beat the predicted-block cost.  The residual
// cost must already include every plane; a NaN/negative residual cost skips.
__device__ __forceinline__ int ir_prefer_residual(double residual_cost,
                                                  double skip_cost) {
  return (residual_cost >= 0.0 && residual_cost < skip_cost) ? 1 : 0;
}

}  // namespace av2enc::m2::residual::dev

#endif  // AV2ENC_M2_RESIDUAL_INTER_RESIDUAL_DEVICE_CUH_
