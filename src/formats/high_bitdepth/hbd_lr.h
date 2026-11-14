// §7.20 loop restoration planning and application for the 10-bit product
// still writer.
//
// Covered here:
//   * Wiener NS (§7.20.3) for luma and chroma at 10-bit, all product chroma
//     formats, including the chroma-from-luma taps;
//   * PC-Wiener (§7.20.4) for luma as RESTORE_PC_WIENER units;
// and, in hbd_gdf.h, GDF (§7.20.5).
//
// The pixel arithmetic is the shared `wiener_core.h` template.  The encoder
// fit is a least-squares solve over the same stripe-aware source sampler, and
// every coefficient is clipped to the normative Wiener_Ns_Taps range before
// the exact integer filter is evaluated.  Ties keep the unfiltered unit.
#ifndef AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_LR_H_
#define AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_LR_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/filters/loop_restoration/classes/wiener_tables.h"
#include "codec/filters/loop_restoration/restoration.h"
#include "codec/filters/loop_restoration/restoration_common.h"
#include "codec/filters/loop_restoration/wiener_core.h"
#include "codec/filters/pc_wiener/pc_wiener_stream.h"
#include "codec/filters/tables/luma_source.h"
#include "codec/filters/tables/pc_wiener.h"
#include "formats/format.h"
#include "formats/high_bitdepth/hbd_filter_config.h"

namespace av2enc::hbd {

struct HbdLrFrame16 {
  const std::uint16_t* curr[3] = {nullptr, nullptr, nullptr};
  const std::uint16_t* cdef[3] = {nullptr, nullptr, nullptr};
  const std::uint16_t* ref[3] = {nullptr, nullptr, nullptr};
  int curr_stride[3] = {0, 0, 0};
  int cdef_stride[3] = {0, 0, 0};
  int ref_stride[3] = {0, 0, 0};
  int visible_w[3] = {0, 0, 0};
  int visible_h[3] = {0, 0, 0};
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;
  int num_planes = 3;
  int bit_depth = 10;
  int base_q_idx = 90;
  // §5.4.5 cfl_ds_filter_index, used by the chroma-from-luma taps of the
  // chroma Wiener fit/apply.  0 is the historical 4:2:0 filter.
  int cfl_ds_filter_index = 0;
};

struct HbdLrPlan {
  HbdLrConfig config;
  std::int64_t sse_before = 0;
  std::int64_t sse_after = 0;
};

namespace lr_detail {

constexpr int kMaxCols = 18;

inline bool solve_system(int n, double a_in[kMaxCols][kMaxCols],
                         double b_in[kMaxCols], double x[kMaxCols]) {
  double a[kMaxCols][kMaxCols];
  double b[kMaxCols];
  double trace = 0.0;
  for (int i = 0; i < n; ++i) {
    trace += a_in[i][i];
    b[i] = b_in[i];
    for (int j = 0; j < n; ++j) a[i][j] = a_in[i][j];
  }
  const double ridge = 1e-6 * (trace / static_cast<double>(n) + 1.0);
  for (int i = 0; i < n; ++i) a[i][i] += ridge;
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double best = std::fabs(a[col][col]);
    for (int row = col + 1; row < n; ++row) {
      const double v = std::fabs(a[row][col]);
      if (v > best) {
        best = v;
        pivot = row;
      }
    }
    if (best < 1e-12) return false;
    if (pivot != col) {
      for (int j = col; j < n; ++j) std::swap(a[pivot][j], a[col][j]);
      std::swap(b[pivot], b[col]);
    }
    const double inv = 1.0 / a[col][col];
    for (int j = col; j < n; ++j) a[col][j] *= inv;
    b[col] *= inv;
    for (int row = 0; row < n; ++row) {
      if (row == col) continue;
      const double factor = a[row][col];
      if (factor == 0.0) continue;
      for (int j = col; j < n; ++j) a[row][j] -= factor * a[col][j];
      b[row] -= factor * b[col];
    }
  }
  for (int i = 0; i < n; ++i) x[i] = b[i];
  return true;
}

inline int quantize(double v) {
  if (!(v == v)) return 0;
  if (v > 1.0e9) v = 1.0e9;
  if (v < -1.0e9) v = -1.0e9;
  return v >= 0.0 ? static_cast<int>(std::floor(v + 0.5))
                  : -static_cast<int>(std::floor(-v + 0.5));
}

// Integer SSE of the §7.20.3 filter with `coeffs` over one unit.  Uses the
// exact shared sampler and rounding, so the value is what the decoder gets.
inline std::int64_t unit_sse(const av2enc::restoration::LrCoreFrame16& core,
                             int plane, int unit_row, int unit_col,
                             const std::int16_t* coeffs,
                             const std::uint16_t* target, int target_stride,
                             int visible_w, int visible_h) {
  std::int64_t sse = 0;
  for (int row = 0; row < core.mi_rows; ++row) {
    for (int col = 0; col < core.mi_cols; ++col) {
      av2enc::restoration::LrBlock b;
      if (!av2enc::restoration::lr_block_context_t(core, plane, row, col, &b)) {
        continue;
      }
      if (b.type != av2enc::restoration::kRestoreWienerNs) continue;
      if (b.unit_row != unit_row || b.unit_col != unit_col) continue;
      for (int r = 0; r < b.h; ++r) {
        for (int c = 0; c < b.w; ++c) {
          const int sx = b.x + c;
          const int sy = b.y + r;
          if (sx >= visible_w || sy >= visible_h) continue;
          const int m =
              av2enc::restoration::lr_get_source_sample_t(core, b, plane, sx,
                                                          sy);
          int s = m << av2enc::restoration::kWienerNsPrecBits;
          const int n_taps = plane == 0
                                 ? av2enc::restoration::kWienerNsTapsY
                                 : av2enc::restoration::kWienerNsTapsUv;
          const std::int8_t(*config)[3] =
              plane == 0 ? av2enc::restoration::kWienerNsConfigY
                         : av2enc::restoration::kWienerNsConfigUv;
          for (int i = 0; i < n_taps; ++i) {
            const int dy = config[i][0];
            const int dx = config[i][1];
            const int diff = av2enc::restoration::lr_get_source_sample_t(
                                 core, b, plane, sx + dx, sy + dy) -
                             m;
            s += diff * coeffs[config[i][2]];
          }
          if (plane > 0) {
            const int m_luma =
                av2enc::restoration::lr_get_luma_sample_t(core, b, sx, sy);
            for (int i = 0; i < n_taps; ++i) {
              const int dy = config[i][0];
              const int dx = config[i][1];
              const int luma_diff =
                  av2enc::restoration::lr_get_luma_sample_t(core, b, sx + dx,
                                                            sy + dy) -
                  m_luma;
              s += luma_diff *
                   coeffs[i + av2enc::restoration::kWienerNsShortCoeffs];
            }
          }
          const int v =
              av2enc::restoration::round2(s,
                                          av2enc::restoration::kWienerNsPrecBits);
          const int out = std::clamp(v, 0, (1 << core.bit_depth) - 1);
          const int diff =
              out - static_cast<int>(
                        target[static_cast<std::size_t>(sy) * target_stride + sx]);
          sse += static_cast<std::int64_t>(diff) * diff;
        }
      }
    }
  }
  return sse;
}

}  // namespace lr_detail

// Fits and decides Wiener NS units for one plane, then writes them into the
// plan.  `plane_cfg` must already carry unit_size/shift and the unit grid.
inline bool plan_lr_plane(const HbdLrFrame16& frame,
                          av2enc::restoration::LrCoreFrame16 core,
                          int plane, std::uint8_t frame_type,
                          HbdLrPlaneConfig* plane_cfg) {
  using namespace av2enc::restoration;
  const int n_cols = plane == 0 ? kWienerNsLumaCoeffs : kWienerNsChromaCoeffs;
  const std::int8_t(*config)[3] =
      plane == 0 ? kWienerNsConfigY : kWienerNsConfigUv;
  const int n_taps = plane == 0 ? kWienerNsTapsY : kWienerNsTapsUv;
  const int subset = plane == 0 ? 3 : 2;
  // Chroma-from-luma taps.  4:2:0 keeps the full §7.20.3 tap set.  The pinned
  // dav2d's CfL luma sampling for a chroma plane with `SubsamplingY == 0`
  // (4:2:2 and 4:4:4) diverges from §7.20.3 in two ways that a conforming
  // encoder cannot infer from the spec: both above-stripe luma lines collapse
  // onto `LumaStripeStartY - 2` (matched in `wiener_core.h`), and the luma
  // taps of the first column of each 64-sample block column read the
  // already-restored luma.  The second one has no spec-side equivalent, and a
  // third divergence appears when deblocking precedes the restoration.  Signal
  // the chroma-only subset (0) so the filter itself stays bit-exact; the
  // chroma coefficients are then fitted without the luma terms.
  const bool use_cfl = plane == 0 || (frame.sub_x == 1 && frame.sub_y == 1);
  const int max_cols = use_cfl ? n_cols : kWienerNsShortCoeffs;

  plane_cfg->frame_type = 0;
  plane_cfg->unit_type.assign(
      static_cast<std::size_t>(plane_cfg->unit_rows) *
          static_cast<std::size_t>(plane_cfg->unit_cols),
      static_cast<std::uint8_t>(kRestoreNone));
  plane_cfg->unit_coeffs.assign(
      static_cast<std::size_t>(plane_cfg->unit_rows) *
          static_cast<std::size_t>(plane_cfg->unit_cols) *
          static_cast<std::size_t>(kWienerNsCoeffStride),
      0);
  plane_cfg->unit_subset.assign(plane_cfg->unit_type.size(), 0);
  plane_cfg->unit_uv_sym.assign(plane_cfg->unit_type.size(), 0);

  const int visible_w = frame.visible_w[plane];
  const int visible_h = frame.visible_h[plane];
  for (int ur = 0; ur < plane_cfg->unit_rows; ++ur) {
    for (int uc = 0; uc < plane_cfg->unit_cols; ++uc) {
      double a[lr_detail::kMaxCols][lr_detail::kMaxCols] = {};
      double b[lr_detail::kMaxCols] = {};
      int samples = 0;
      std::int64_t identity_sse = 0;
      for (int row = 0; row < core.mi_rows; ++row) {
        for (int col = 0; col < core.mi_cols; ++col) {
          LrBlock blk;
          if (!lr_block_context_t(core, plane, row, col, &blk)) continue;
          if (blk.type != kRestoreWienerNs) continue;
          if (blk.unit_row != ur || blk.unit_col != uc) continue;
          for (int r = 0; r < blk.h; ++r) {
            for (int c = 0; c < blk.w; ++c) {
              const int sx = blk.x + c;
              const int sy = blk.y + r;
              if (sx >= visible_w || sy >= visible_h) continue;
              const int m = lr_get_source_sample_t(core, blk, plane, sx, sy);
              const int t = static_cast<int>(
                  frame.ref[plane][static_cast<std::size_t>(sy) *
                                       static_cast<std::size_t>(
                                           frame.ref_stride[plane]) +
                                   static_cast<std::size_t>(sx)]);
              double diff[lr_detail::kMaxCols] = {};
              for (int i = 0; i < n_taps; ++i) {
                const int dy = config[i][0];
                const int dx = config[i][1];
                const int d = lr_get_source_sample_t(core, blk, plane, sx + dx,
                                                      sy + dy) -
                              m;
                diff[config[i][2]] += static_cast<double>(d);
              }
              if (plane > 0 && use_cfl) {
                const int m_luma = lr_get_luma_sample_t(core, blk, sx, sy);
                for (int i = 0; i < n_taps; ++i) {
                  const int dy = config[i][0];
                  const int dx = config[i][1];
                  const int d = lr_get_luma_sample_t(core, blk, sx + dx,
                                                      sy + dy) -
                                m_luma;
                  diff[i + kWienerNsShortCoeffs] += static_cast<double>(d);
                }
              }
              const double tm = static_cast<double>(t - m);
              for (int j = 0; j < max_cols; ++j) {
                b[j] += tm * diff[j];
                for (int k = 0; k < max_cols; ++k) {
                  a[j][k] += diff[j] * diff[k];
                }
              }
              const int d0 = m - t;
              identity_sse += static_cast<std::int64_t>(d0) * d0;
              ++samples;
            }
          }
        }
      }
      if (samples == 0) continue;
      double x[lr_detail::kMaxCols] = {};
      if (!lr_detail::solve_system(max_cols, a, b, x)) continue;
      std::int16_t coeffs[kWienerNsCoeffStride] = {};
      for (int j = 0; j < max_cols; ++j) {
        coeffs[j] = static_cast<std::int16_t>(
            wiener_ns_clip_coeff(plane > 0, j, lr_detail::quantize(x[j])));
      }
      if (plane > 0 && use_cfl) {
        // §5.20.10.6 signals one value per mirrored pair for j >= 6.
        for (int j = kWienerNsShortCoeffs; j + 1 < kWienerNsCoeffStride;
             j += 2) {
          const int avg = (static_cast<int>(coeffs[j]) +
                           static_cast<int>(coeffs[j + 1]) + 1) /
                          2;
          coeffs[j] = static_cast<std::int16_t>(wiener_ns_clip_coeff(
              true, j, avg));
          coeffs[j + 1] = coeffs[j];
        }
      }
      const std::int64_t fitted_sse = lr_detail::unit_sse(
          core, plane, ur, uc, coeffs, frame.ref[plane],
          frame.ref_stride[plane], visible_w, visible_h);
      const std::size_t index =
          static_cast<std::size_t>(ur) *
              static_cast<std::size_t>(plane_cfg->unit_cols) +
          static_cast<std::size_t>(uc);
      if (fitted_sse < identity_sse) {
        plane_cfg->unit_type[index] = static_cast<std::uint8_t>(
            kRestoreWienerNs);
        for (int j = 0; j < kWienerNsCoeffStride; ++j) {
          plane_cfg->unit_coeffs[index *
                                     static_cast<std::size_t>(
                                         kWienerNsCoeffStride) +
                                 static_cast<std::size_t>(j)] = coeffs[j];
        }
        plane_cfg->unit_subset[index] =
            static_cast<std::uint8_t>(use_cfl ? subset : 0);
        plane_cfg->unit_uv_sym[index] = (plane > 0 && use_cfl) ? 1 : 0;
      }
    }
  }
  for (int ur = 0; ur < plane_cfg->unit_rows; ++ur) {
    for (int uc = 0; uc < plane_cfg->unit_cols; ++uc) {
      const std::size_t index =
          static_cast<std::size_t>(ur) *
              static_cast<std::size_t>(plane_cfg->unit_cols) +
          static_cast<std::size_t>(uc);
      if (plane_cfg->unit_type[index] == kRestoreWienerNs) {
        plane_cfg->enabled = true;
      }
    }
  }
  if (plane_cfg->enabled) plane_cfg->frame_type = frame_type;
  return true;
}

// Builds a `LrCoreFrame16` for the plan/apply stages.
inline av2enc::restoration::LrCoreFrame16 make_lr_core16(
    const HbdLrFrame16& frame, const HbdLrConfig& config,
    std::uint16_t* const lr[3], const int lr_stride[3]) {
  av2enc::restoration::LrCoreFrame16 core{};
  for (int p = 0; p < 3; ++p) {
    core.plane[p].curr = frame.curr[p];
    core.plane[p].curr_stride = frame.curr_stride[p];
    core.plane[p].cdef = frame.cdef[p];
    core.plane[p].cdef_stride = frame.cdef_stride[p];
    core.plane[p].dst = lr[p];
    core.plane[p].dst_stride = lr_stride[p];
    core.plane[p].unit_type = config.plane[p].unit_type.data();
    core.plane[p].unit_cols = config.plane[p].unit_cols;
    core.plane[p].unit_rows = config.plane[p].unit_rows;
    core.plane[p].unit_coeffs = config.plane[p].unit_coeffs.data();
  }
  core.mi_cols = frame.mi_cols;
  core.mi_rows = frame.mi_rows;
  core.sub_x = frame.sub_x;
  core.sub_y = frame.sub_y;
  core.bit_depth = frame.bit_depth;
  core.num_planes = frame.num_planes;
  for (int p = 0; p < 3; ++p) {
    core.unit_size[p] = config.plane[p].unit_size;
    core.frame_type[p] = config.plane[p].frame_type;
  }
  core.cfl_ds_filter_index = config.cfl_ds_filter_index;
  return core;
}

// Applies PC-Wiener to one unit of the luma plane into `lr_luma`.  The unit
// grid must mark the unit as RESTORE_PC_WIENER.
inline void pc_wiener_unit(
    const HbdLrFrame16& frame, const HbdLrConfig& config,
    const std::vector<std::uint8_t>& lr_tx_skip, int unit_row, int unit_col,
    std::uint16_t* lr_luma, int lr_stride) {
  using namespace av2enc::filters::spec;
  if (frame.num_planes < 1) return;
  std::vector<std::uint8_t> types(config.plane[0].unit_type.size(),
                                  av2enc::restoration::kRestorePcWiener);
  av2enc::restoration::LrCoreFrame16 core{};
  core.mi_cols = frame.mi_cols;
  core.mi_rows = frame.mi_rows;
  core.sub_x = frame.sub_x;
  core.sub_y = frame.sub_y;
  core.bit_depth = frame.bit_depth;
  core.plane[0].curr = frame.curr[0];
  core.plane[0].curr_stride = frame.curr_stride[0];
  core.plane[0].cdef = frame.cdef[0];
  core.plane[0].cdef_stride = frame.cdef_stride[0];
  core.plane[0].dst = lr_luma;
  core.plane[0].dst_stride = lr_stride;
  core.plane[0].unit_type = types.data();
  core.plane[0].unit_cols = config.plane[0].unit_cols;
  core.plane[0].unit_rows = config.plane[0].unit_rows;
  core.unit_size[0] = config.plane[0].unit_size;
  core.frame_type[0] = av2enc::restoration::kRestorePcWiener;

  for (int row = 0; row < core.mi_rows; ++row) {
    for (int col = 0; col < core.mi_cols; ++col) {
      av2enc::restoration::LrBlock blk;
      if (!av2enc::restoration::lr_block_context_t(core, 0, row, col, &blk)) {
        continue;
      }
      if (blk.unit_row != unit_row || blk.unit_col != unit_col) continue;
      PcWienerFrame pcf{};
      pcf.src.curr = frame.curr[0];
      pcf.src.cdef = frame.cdef[0];
      pcf.src.curr_stride = frame.curr_stride[0];
      pcf.src.cdef_stride = frame.cdef_stride[0];
      pcf.src.luma_start_x = blk.luma_start_x;
      pcf.src.luma_end_x = blk.luma_end_x;
      pcf.src.luma_start_y = blk.luma_start_y;
      pcf.src.luma_end_y = blk.luma_end_y;
      pcf.src.luma_stripe_start_y = blk.stripe_start_y;
      pcf.src.luma_stripe_end_y = blk.stripe_end_y;
      pcf.lr = lr_luma;
      pcf.lr_stride = lr_stride;
      PcWienerConfig cfg;
      cfg.base_q_idx = frame.base_q_idx;
      cfg.bit_depth = frame.bit_depth;
      cfg.mi_col_end = frame.mi_cols;
      cfg.mi_row_start = 0;
      cfg.mi_row_end = frame.mi_rows;
      cfg.lr_tx_skip = lr_tx_skip.data();
      cfg.lr_tx_skip_stride = frame.mi_cols;
      PcWienerProcess(pcf, cfg, blk.x, blk.y, blk.w, blk.h, false);
    }
  }
}

// Chooses RESTORE_PC_WIENER per luma restoration unit by exact SSE against the
// source.  `scratch` is a luma plane of the same stride/size as LrFrame.
inline void plan_pc_wiener16(const HbdLrFrame16& frame, HbdLrPlan* plan,
                             const std::vector<std::uint8_t>& lr_tx_skip,
                             std::uint16_t* scratch) {
  HbdLrConfig& cfg = plan->config;
  HbdLrPlaneConfig& luma = cfg.plane[0];
  const int stride = (frame.mi_cols * 4 + 63) & ~63;
  const int height = frame.mi_rows * 4;
  const int visible_w = frame.visible_w[0];
  const int visible_h = frame.visible_h[0];
  for (int ur = 0; ur < luma.unit_rows; ++ur) {
    for (int uc = 0; uc < luma.unit_cols; ++uc) {
      // Copy CdefFrame into the scratch plane.
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < stride; ++x) {
          scratch[static_cast<std::size_t>(y) * stride + x] =
              frame.cdef[0][static_cast<std::size_t>(y) *
                                frame.cdef_stride[0] + x];
        }
      }
      pc_wiener_unit(frame, cfg, lr_tx_skip, ur, uc, scratch, stride);
      const int unit_w = luma.unit_size;
      const int x0 = uc * unit_w;
      const int y0 = ur * unit_w;
      const int x1 = std::min(x0 + unit_w, visible_w);
      const int y1 = std::min(y0 + unit_w, visible_h);
      std::int64_t before = 0;
      std::int64_t after = 0;
      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
          const int ref = static_cast<int>(
              frame.ref[0][static_cast<std::size_t>(y) *
                               frame.ref_stride[0] + x]);
          const int cdef = static_cast<int>(
              frame.cdef[0][static_cast<std::size_t>(y) *
                                frame.cdef_stride[0] + x]);
          const int out = static_cast<int>(
              scratch[static_cast<std::size_t>(y) * stride + x]);
          const std::int64_t d0 = cdef - ref;
          const std::int64_t d1 = out - ref;
          before += d0 * d0;
          after += d1 * d1;
        }
      }
      if (after < before) {
        luma.unit_type[static_cast<std::size_t>(ur) *
                           static_cast<std::size_t>(luma.unit_cols) +
                       static_cast<std::size_t>(uc)] =
            av2enc::restoration::kRestorePcWiener;
        luma.enabled = true;
      }
    }
  }
  if (luma.enabled) {
    luma.frame_type = av2enc::restoration::kRestorePcWiener;
    cfg.enabled = true;
  } else {
    luma.frame_type = 0;
  }
}

// Applies PC-Wiener (§7.20.4) to the luma units whose LrType is
// RESTORE_PC_WIENER.  `lr_luma` must already contain the CdefFrame copy.
inline void apply_pc_wiener16(
    const HbdLrFrame16& frame, const HbdLrConfig& config,
    const std::vector<std::uint8_t>& lr_tx_skip, std::uint16_t* lr_luma,
    int lr_stride) {
  const HbdLrPlaneConfig& luma = config.plane[0];
  for (int ur = 0; ur < luma.unit_rows; ++ur) {
    for (int uc = 0; uc < luma.unit_cols; ++uc) {
      const std::size_t index =
          static_cast<std::size_t>(ur) *
              static_cast<std::size_t>(luma.unit_cols) +
          static_cast<std::size_t>(uc);
      if (index >= luma.unit_type.size()) continue;
      if (luma.unit_type[index] != av2enc::restoration::kRestorePcWiener) {
        continue;
      }
      pc_wiener_unit(frame, config, lr_tx_skip, ur, uc, lr_luma, lr_stride);
    }
  }
}

// Copies CdefFrame into LrFrame and runs Wiener NS for the units the plan
// marks RESTORE_WIENER_NONSEP.  PC-Wiener units stay as the CdefFrame copy and
// are filled by apply_pc_wiener16().
inline bool apply_wiener16(const HbdLrFrame16& frame,
                           const HbdLrConfig& config,
                           std::uint16_t* const lr[3],
                           const int lr_stride[3], std::string* error) {
  using namespace av2enc::restoration;
  // Copy step is unconditional (§7.20: LrFrame := CdefFrame).
  for (int p = 0; p < frame.num_planes; ++p) {
    const int w = (frame.mi_cols * 4) >> (p == 0 ? 0 : frame.sub_x);
    const int h = (frame.mi_rows * 4) >> (p == 0 ? 0 : frame.sub_y);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        lr[p][static_cast<std::size_t>(y) * lr_stride[p] + x] =
            frame.cdef[p][static_cast<std::size_t>(y) * frame.cdef_stride[p] +
                          x];
      }
    }
  }
  // Wiener units only: copy the config and blank PC-Wiener types so the core
  // does not reject them.
  HbdLrConfig wiener_only = config;
  for (int p = 0; p < frame.num_planes; ++p) {
    for (std::uint8_t& type : wiener_only.plane[p].unit_type) {
      if (type == kRestorePcWiener) type = kRestoreNone;
    }
    if (wiener_only.plane[p].frame_type == kRestorePcWiener) {
      // A PC-Wiener luma plane has no Wiener units; keep it disabled.
      wiener_only.plane[p].enabled = false;
    }
  }
  LrCoreFrame16 core = make_lr_core16(frame, wiener_only, lr, lr_stride);
  for (int p = 0; p < frame.num_planes; ++p) {
    if (wiener_only.plane[p].frame_type == kRestoreWienerNs) {
      lr_plane_t(core, p);
    }
  }
  (void)error;
  return true;
}

// Header-level decision for one plane: sets the frame type and unit grid from
// the requested flags.  A plane with no winning unit stays disabled.
inline bool plan_lr(const HbdLrFrame16& frame, bool luma, bool chroma,
                    bool pc_wiener, int base_q_idx, HbdLrPlan* plan,
                    std::string* error) {
  auto fail = [error](const std::string& text) {
    if (error != nullptr) *error = text;
    return false;
  };
  if (plan == nullptr) return fail("lr plan: null output");
  *plan = HbdLrPlan{};
  if (frame.mi_cols <= 0 || frame.mi_rows <= 0) {
    return fail("lr plan: invalid geometry");
  }
  // §5.18.7.11: LoopRestorationSize is 64 luma / 64 chroma.  Chroma must be at
  // least one stripe high (64 >> SubsamplingY).
  const int luma_unit = 64;
  const int chroma_unit = (512 >> std::max(frame.sub_x, frame.sub_y)) >>
                          (3 - std::max(frame.sub_x, frame.sub_y));
  plan->config.enabled = false;
  plan->config.base_q_idx = base_q_idx;
  plan->config.cfl_ds_filter_index = frame.cfl_ds_filter_index;
  plan->config.luma_shift = 3;
  plan->config.chroma_shift = 3 - std::max(frame.sub_x, frame.sub_y);

  HbdLrConfig& cfg = plan->config;
  for (int p = 0; p < frame.num_planes; ++p) {
    cfg.plane[p].unit_size = p == 0 ? luma_unit : chroma_unit;
    cfg.plane[p].shift = p == 0 ? 3 : plan->config.chroma_shift;
    cfg.plane[p].frame_type = 0;
  }

  av2enc::restoration::LrCoreFrame16 core{};
  core.mi_cols = frame.mi_cols;
  core.mi_rows = frame.mi_rows;
  core.sub_x = frame.sub_x;
  core.sub_y = frame.sub_y;
  core.bit_depth = frame.bit_depth;
  core.cfl_ds_filter_index = frame.cfl_ds_filter_index;
  core.num_planes = frame.num_planes;
  for (int p = 0; p < frame.num_planes; ++p) {
    core.plane[p].curr = frame.curr[p];
    core.plane[p].curr_stride = frame.curr_stride[p];
    core.plane[p].cdef = frame.cdef[p];
    core.plane[p].cdef_stride = frame.cdef_stride[p];
    core.unit_size[p] = cfg.plane[p].unit_size;
    core.frame_type[p] = av2enc::restoration::kRestoreWienerNs;
    const int sub_x = p == 0 ? 0 : frame.sub_x;
    const int sub_y = p == 0 ? 0 : frame.sub_y;
    const int pw = (frame.mi_cols * 4) >> sub_x;
    const int ph = (frame.mi_rows * 4) >> sub_y;
    cfg.plane[p].unit_cols =
        av2enc::restoration::lr_count_units(cfg.plane[p].unit_size, pw);
    cfg.plane[p].unit_rows =
        av2enc::restoration::lr_count_units(cfg.plane[p].unit_size, ph);
    core.plane[p].unit_cols = cfg.plane[p].unit_cols;
    core.plane[p].unit_rows = cfg.plane[p].unit_rows;
  }

  if (frame.num_planes > 0 && luma && !pc_wiener) {
    if (!plan_lr_plane(frame, core, 0, av2enc::restoration::kRestoreWienerNs,
                       &cfg.plane[0])) {
      return fail("lr plan: luma fit failed");
    }
  }
  if (frame.num_planes > 2 && chroma) {
    for (int p = 1; p < 3; ++p) {
      if (!plan_lr_plane(frame, core, p,
                         av2enc::restoration::kRestoreWienerNs,
                         &cfg.plane[p])) {
        return fail("lr plan: chroma fit failed");
      }
    }
  }
  // PC-Wiener is planned by hbd_still (it needs the LV values); this function
  // only allocates the luma unit grid.
  if (frame.num_planes > 0 && pc_wiener) {
    cfg.plane[0].unit_type.assign(
        static_cast<std::size_t>(cfg.plane[0].unit_rows) *
            static_cast<std::size_t>(cfg.plane[0].unit_cols),
        static_cast<std::uint8_t>(av2enc::restoration::kRestoreNone));
    cfg.plane[0].unit_coeffs.assign(
        cfg.plane[0].unit_type.size() *
            static_cast<std::size_t>(
                av2enc::restoration::kWienerNsCoeffStride),
        0);
    cfg.plane[0].unit_subset.assign(cfg.plane[0].unit_type.size(), 0);
    cfg.plane[0].unit_uv_sym.assign(cfg.plane[0].unit_type.size(), 0);
    cfg.plane[0].frame_type = av2enc::restoration::kRestorePcWiener;
  }
  for (int p = 0; p < frame.num_planes; ++p) {
    if (cfg.plane[p].frame_type != 0) cfg.enabled = true;
  }
  if (cfg.enabled) {
    plan->sse_before = 0;
    plan->sse_after = 0;
  }
  if (error != nullptr) error->clear();
  return true;
}

}  // namespace av2enc::hbd

#endif  // AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_LR_H_
