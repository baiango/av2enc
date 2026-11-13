// §7.19 CCSO planning and application for the 10-bit product still writer.
//
// The pixel process is the shared `ccso_core.h` template (`ccso_frame16`).
// This header only runs the encoder-side fit:
//
//   * classify every sample by the §7.19.1 luma band and edge classes using
//     the *pre-CDEF* luma (CurrFrame),
//   * fit one offset per cell as the mean (target - CdefFrame) rounded to the
//     signalled `Ccso_Offset * (ccso_scale_idx + 1)` ladder,
//   * keep the unit only when applying the fitted offsets lowers SSE,
//   * compare a band-offset-only candidate against edge-classified candidates
//     and pick the better frame.
//
// Offsets and classes are bit-depth generic: only the band shift
// (BitDepth - max_band_log2) and the Clip1 range change.
#ifndef AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_CCSO_H_
#define AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_CCSO_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/filters/loop_restoration/restoration.h"
#include "codec/filters/loop_restoration/restoration_common.h"
#include "formats/format.h"
#include "formats/high_bitdepth/hbd_filter_config.h"

namespace av2enc::hbd {

struct HbdCcsoPlan {
  HbdCcsoConfig config;
  std::int64_t sse_before = 0;
  std::int64_t sse_after = 0;
};

// Frame views for the fit.  `curr` is the post-deblock / pre-CDEF picture,
// `cdef` the post-CDEF picture, `ref` the visible source.
struct HbdCcsoFrame16 {
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
};

namespace ccso_detail {

inline int plane_sub_x(const HbdCcsoFrame16& f, int plane) {
  return plane == 0 ? 0 : f.sub_x;
}
inline int plane_sub_y(const HbdCcsoFrame16& f, int plane) {
  return plane == 0 ? 0 : f.sub_y;
}

inline int luma_at(const HbdCcsoFrame16& f, int x, int y) {
  const int max_x = f.mi_cols * 4 - 1;
  const int max_y = f.mi_rows * 4 - 1;
  x = x < 0 ? 0 : (x > max_x ? max_x : x);
  y = y < 0 ? 0 : (y > max_y ? max_y : y);
  return static_cast<int>(f.curr[0][y * f.curr_stride[0] + x]);
}

struct Candidate {
  bool bo_only = true;
  int scale_idx = 0;
  int quant_idx = 0;
  int ext_filter = 0;
  int edge_clf = 0;
  int max_band_log2 = 6;
};

struct FitResult {
  bool valid = false;
  int cells = 0;
  int interval = 1;
  int bands = 0;
  std::vector<std::int16_t> offsets;
  std::int64_t sse_before = 0;
  std::int64_t sse_after = 0;
  std::vector<std::uint8_t> blks;
};

// Nearest signalled offset index for a mean value.  Ties prefer the smaller
// magnitude so a zero mean stays at index 0.
inline int nearest_offset_idx(int mean, int scale) {
  int best = 0;
  int best_diff = 1 << 30;
  int best_magnitude = 1 << 30;
  for (int i = 0; i < restoration::kCcsoOffsetRange; ++i) {
    const int value = static_cast<int>(restoration::kCcsoOffset[i]) * scale;
    const int diff = mean - value;
    const int magnitude = diff < 0 ? -diff : diff;
    const int value_magnitude = value < 0 ? -value : value;
    if (magnitude < best_diff ||
        (magnitude == best_diff && value_magnitude < best_magnitude)) {
      best_diff = magnitude;
      best_magnitude = value_magnitude;
      best = i;
    }
  }
  return best;
}

// One full two-pass fit for `plane` with one candidate parameter set.
inline FitResult fit_plane(const HbdCcsoFrame16& f, int plane,
                           const Candidate& c, int ccso_luma_size_log2) {
  FitResult out;
  const int sub_x = plane_sub_x(f, plane);
  const int sub_y = plane_sub_y(f, plane);
  const int width = (f.mi_cols * 4) >> sub_x;
  const int height = (f.mi_rows * 4) >> sub_y;
  if (width <= 0 || height <= 0) return out;
  const int interval = c.bo_only ? 1 : (3 - c.edge_clf);
  if (interval <= 0) return out;
  const int bands = 1 << c.max_band_log2;
  const int cells = restoration::kCcsoOffsetEntries;
  out.valid = true;
  out.cells = cells;
  out.interval = interval;
  out.bands = bands;
  out.offsets.assign(static_cast<std::size_t>(cells), 0);
  std::vector<std::int64_t> sum(static_cast<std::size_t>(cells), 0);
  std::vector<std::int64_t> count(static_cast<std::size_t>(cells), 0);
  const int shift = f.bit_depth - c.max_band_log2;
  if (shift < 0 || shift > 24) {
    out.valid = false;
    return out;
  }
  const int quant_step =
      static_cast<int>(restoration::kCcsoQuantSz[c.scale_idx & 3]
                                                [c.quant_idx & 3]);
  const int ext = (c.ext_filter >= 0 && c.ext_filter < 7) ? c.ext_filter : 0;
  const int ey = restoration::kCcsoPos[ext][0];
  const int ex = restoration::kCcsoPos[ext][1];
  const std::size_t luma_stride =
      static_cast<std::size_t>(f.curr_stride[0]);
  const std::size_t dst_stride =
      static_cast<std::size_t>(f.cdef_stride[plane]);
  const std::size_t ref_stride =
      static_cast<std::size_t>(f.ref_stride[plane]);

  auto classify = [&](int x3, int y3, int* band_out, int* cls0_out,
                      int* cls1_out) {
    const int x_luma = x3 << sub_x;
    const int y_luma = y3 << sub_y;
    const int centre = static_cast<int>(
        f.curr[0][static_cast<std::size_t>(y_luma) * luma_stride + x_luma]);
    *band_out = centre >> shift;
    *cls0_out = 0;
    *cls1_out = 0;
    if (!c.bo_only) {
      const int plus = luma_at(f, x_luma + ex, y_luma + ey);
      const int minus = luma_at(f, x_luma - ex, y_luma - ey);
      *cls0_out = restoration::ccso_score(plus - centre, quant_step,
                                          c.edge_clf);
      *cls1_out = restoration::ccso_score(minus - centre, quant_step,
                                          c.edge_clf);
    }
  };

  // Pass 1: mean desired offset per cell over the visible samples.
  const int vis_w = std::min(f.visible_w[plane], width);
  const int vis_h = std::min(f.visible_h[plane], height);
  for (int y = 0; y < vis_h; ++y) {
    for (int x = 0; x < vis_w; ++x) {
      int band = 0;
      int cls0 = 0;
      int cls1 = 0;
      classify(x, y, &band, &cls0, &cls1);
      const std::size_t cell = static_cast<std::size_t>(
          restoration::ccso_offset_index(band, cls0, cls1));
      const int dst = static_cast<int>(
          f.cdef[plane][static_cast<std::size_t>(y) * dst_stride + x]);
      const int ref = static_cast<int>(
          f.ref[plane][static_cast<std::size_t>(y) * ref_stride + x]);
      sum[cell] += ref - dst;
      ++count[cell];
    }
  }
  const int scale = c.scale_idx + 1;
  for (int i = 0; i < cells; ++i) {
    if (count[static_cast<std::size_t>(i)] == 0) continue;
    const std::int64_t n = count[static_cast<std::size_t>(i)];
    int mean = static_cast<int>((sum[static_cast<std::size_t>(i)] >= 0
                                     ? sum[static_cast<std::size_t>(i)] + n / 2
                                     : sum[static_cast<std::size_t>(i)] - n / 2) /
                                n);
    const int idx = nearest_offset_idx(mean, scale);
    out.offsets[static_cast<std::size_t>(i)] = static_cast<std::int16_t>(
        static_cast<int>(restoration::kCcsoOffset[idx]) * scale);
  }

  // Pass 2: per-unit SSE before/after over the visible samples.
  const int unit_w = 1 << (ccso_luma_size_log2 - sub_x);
  const int unit_h = 1 << (ccso_luma_size_log2 - sub_y);
  if (unit_w <= 0 || unit_h <= 0) {
    out.valid = false;
    return out;
  }
  const int unit_cols = (width + unit_w - 1) / unit_w;
  const int unit_rows = (height + unit_h - 1) / unit_h;
  out.blks.assign(static_cast<std::size_t>(unit_cols) *
                      static_cast<std::size_t>(unit_rows),
                  0);
  std::vector<std::int64_t> before(static_cast<std::size_t>(unit_cols) *
                                       static_cast<std::size_t>(unit_rows),
                                   0);
  std::vector<std::int64_t> after = before;
  const int max_value = (1 << f.bit_depth) - 1;
  for (int y = 0; y < vis_h; ++y) {
    for (int x = 0; x < vis_w; ++x) {
      int band = 0;
      int cls0 = 0;
      int cls1 = 0;
      classify(x, y, &band, &cls0, &cls1);
      const std::size_t cell = static_cast<std::size_t>(
          restoration::ccso_offset_index(band, cls0, cls1));
      const int dst = static_cast<int>(
          f.cdef[plane][static_cast<std::size_t>(y) * dst_stride + x]);
      const int ref = static_cast<int>(
          f.ref[plane][static_cast<std::size_t>(y) * ref_stride + x]);
      const int filtered = std::clamp(
          dst + static_cast<int>(out.offsets[cell]), 0, max_value);
      const std::int64_t d0 = static_cast<std::int64_t>(dst - ref);
      const std::int64_t d1 = static_cast<std::int64_t>(filtered - ref);
      const std::size_t unit =
          static_cast<std::size_t>(y / unit_h) *
              static_cast<std::size_t>(unit_cols) +
          static_cast<std::size_t>(x / unit_w);
      before[unit] += d0 * d0;
      after[unit] += d1 * d1;
    }
  }
  out.sse_before = 0;
  out.sse_after = 0;
  for (std::size_t i = 0; i < before.size(); ++i) {
    out.sse_before += before[i];
    out.sse_after += after[i];
    out.blks[i] = after[i] < before[i] ? 1 : 0;
  }
  return out;
}

}  // namespace ccso_detail

// Builds the CCSO decision for one intra still frame.  Every product chroma
// format is supported; the offset table is per plane and the unit grid follows
// `ccso_luma_size_log2` (6 for a 64x64 superblock).
inline bool plan_ccso(const HbdCcsoFrame16& frame, int ccso_luma_size_log2,
                      bool edge_only, HbdCcsoPlan* plan, std::string* error) {
  auto fail = [error](const std::string& text) {
    if (error != nullptr) *error = text;
    return false;
  };
  if (plan == nullptr) return fail("ccso plan: null output");
  *plan = HbdCcsoPlan{};
  if (frame.mi_cols <= 0 || frame.mi_rows <= 0) {
    return fail("ccso plan: invalid geometry");
  }
  if (frame.num_planes < 1 || frame.num_planes > 3) {
    return fail("ccso plan: invalid plane count");
  }
  if (ccso_luma_size_log2 < 6 || ccso_luma_size_log2 > 8) {
    return fail("ccso plan: unit size outside 6..8");
  }
  plan->config.enabled = false;
  plan->config.ccso_luma_size_log2 = ccso_luma_size_log2;
  plan->config.unit_matches_sb_size = true;

  std::vector<ccso_detail::Candidate> candidates;
  if (!edge_only) {
    for (int scale = 0; scale <= 1; ++scale) {
      for (int bands : {4, 6}) {
        ccso_detail::Candidate c;
        c.bo_only = true;
        c.scale_idx = scale;
        c.max_band_log2 = bands;
        candidates.push_back(c);
      }
    }
  }
  for (int edge_clf = 0; edge_clf <= 1; ++edge_clf) {
    for (int scale = 0; scale <= 1; ++scale) {
      for (int quant : {1, 2}) {
        for (int ext : {0, 2}) {
          ccso_detail::Candidate c;
          c.bo_only = false;
          c.edge_clf = edge_clf;
          c.scale_idx = scale;
          c.quant_idx = quant;
          c.ext_filter = ext;
          c.max_band_log2 = 3;  // §5.18.7.12 cap for the edge form
          candidates.push_back(c);
        }
      }
    }
  }

  std::vector<ccso_detail::Candidate> best_candidates;
  std::vector<ccso_detail::FitResult> best_fits;
  for (int plane = 0; plane < frame.num_planes; ++plane) {
    ccso_detail::FitResult best;
    ccso_detail::Candidate best_candidate;
    for (const ccso_detail::Candidate& c : candidates) {
      ccso_detail::FitResult one =
          ccso_detail::fit_plane(frame, plane, c, ccso_luma_size_log2);
      if (!one.valid) continue;
      if (!best.valid || one.sse_after < best.sse_after) {
        best = std::move(one);
        best_candidate = c;
      }
    }
    if (!best.valid) {
      best_candidates.push_back(ccso_detail::Candidate{});
      best_fits.push_back(ccso_detail::FitResult{});
      continue;
    }
    best_candidates.push_back(best_candidate);
    const std::int64_t gain = best.sse_before - best.sse_after;
    if (gain <= 0 && !edge_only) {
      // No unit improves: leave the plane disabled (no header row, no tile
      // symbols, and the CdefFrame stays untouched).
      best_fits.push_back(ccso_detail::FitResult{});
      continue;
    }
    best_fits.push_back(std::move(best));
  }

  for (int plane = 0; plane < frame.num_planes; ++plane) {
    const ccso_detail::FitResult& fit = best_fits[static_cast<std::size_t>(plane)];
    if (!fit.valid || fit.blks.empty()) continue;
    const ccso_detail::Candidate& c =
        best_candidates[static_cast<std::size_t>(plane)];
    HbdCcsoPlaneConfig& plane_cfg = plan->config.plane[plane];
    plane_cfg.enabled = true;
    plane_cfg.bo_only = c.bo_only;
    plane_cfg.scale_idx = c.scale_idx;
    plane_cfg.quant_idx = c.quant_idx;
    plane_cfg.ext_filter = c.ext_filter;
    plane_cfg.edge_clf = c.edge_clf;
    plane_cfg.max_band_log2 = c.max_band_log2;
    const int interval = fit.interval;
    const int bands = fit.bands;
    plane_cfg.offset_idx.assign(static_cast<std::size_t>(bands) * interval *
                                    interval,
                                0);
    std::size_t k = 0;
    for (int d0 = 0; d0 < interval; ++d0) {
      for (int d1 = 0; d1 < interval; ++d1) {
        for (int band = 0; band < bands; ++band) {
          const int value =
              fit.offsets[static_cast<std::size_t>(
                  restoration::ccso_offset_index(band, d0, d1))];
          const int scale = c.scale_idx + 1;
          int idx = 0;
          for (int i = 0; i < restoration::kCcsoOffsetRange; ++i) {
            if (static_cast<int>(restoration::kCcsoOffset[i]) * scale ==
                value) {
              idx = i;
              break;
            }
          }
          plane_cfg.offset_idx[k++] = static_cast<std::uint8_t>(idx);
        }
      }
    }
    const int sub_x = ccso_detail::plane_sub_x(frame, plane);
    const int sub_y = ccso_detail::plane_sub_y(frame, plane);
    const int width = (frame.mi_cols * 4) >> sub_x;
    const int height = (frame.mi_rows * 4) >> sub_y;
    const int unit_w = 1 << (ccso_luma_size_log2 - sub_x);
    const int unit_h = 1 << (ccso_luma_size_log2 - sub_y);
    const int cols = (width + unit_w - 1) / unit_w;
    const int rows = (height + unit_h - 1) / unit_h;
    plan->config.blks[plane] = std::move(fit.blks);
    plan->config.blk_cols[plane] = cols;
    plan->config.blk_rows[plane] = rows;
    plan->config.enabled = true;
    plan->sse_before += fit.sse_before;
    plan->sse_after += fit.sse_after;
  }
  if (!plan->config.enabled) {
    // Nothing to signal; keep an all-zero config.
    *plan = HbdCcsoPlan{};
    plan->config.ccso_luma_size_log2 = ccso_luma_size_log2;
  }
  if (error != nullptr) error->clear();
  return true;
}

// Runs §7.19 on uint16 planes.  `frame.cdef` is modified in place and `curr`
// is read; both must be the same buffers the planner saw.  `format` selects
// the plane count and subsampling.
inline bool apply_ccso16(const HbdCcsoPlan& plan, const fmt::Format& format,
                         const std::uint16_t* const curr[3],
                         std::uint16_t* const cdef[3],
                         const int curr_stride[3],
                         const int cdef_stride[3], int mi_cols, int mi_rows,
                         std::string* error) {
  if (!plan.config.enabled) return true;
  const int num_planes = fmt::num_planes(format.chroma);
  restoration::RestorationFrame16 frame{};
  for (int p = 0; p < num_planes; ++p) {
    frame.curr[p].data = const_cast<std::uint16_t*>(curr[p]);
    frame.curr[p].stride = curr_stride[p];
    frame.cdef[p].data = cdef[p];
    frame.cdef[p].stride = cdef_stride[p];
  }
  frame.mi_cols = mi_cols;
  frame.mi_rows = mi_rows;
  frame.subsampling_x = fmt::subsampling_x(format.chroma);
  frame.subsampling_y = fmt::subsampling_y(format.chroma);
  frame.bit_depth = 10;
  frame.num_planes = num_planes;
  frame.sb_width = 64;
  frame.sb_height = 64;
  frame.sb_shift = 4;

  restoration::CcsoFrameParams params;
  params.ccso_luma_size_log2 = plan.config.ccso_luma_size_log2;
  for (int p = 0; p < num_planes; ++p) {
    const HbdCcsoPlaneConfig& src = plan.config.plane[p];
    restoration::CcsoParams& dst = params.plane[p];
    dst.enabled = src.enabled;
    dst.bo_only = src.bo_only;
    dst.scale_idx = src.scale_idx;
    dst.quant_idx = src.quant_idx;
    dst.ext_filter = src.ext_filter;
    dst.edge_clf = src.edge_clf;
    dst.max_band_log2 = src.max_band_log2;
    params.blks[p] = plan.config.blks[p].data();
    params.blk_cols[p] = plan.config.blk_cols[p];
    params.blk_rows[p] = plan.config.blk_rows[p];
    if (src.enabled) {
      if (!restoration::ccso_set_offsets(dst, src.offset_idx.data(),
                                         src.offset_idx.size())) {
        if (error != nullptr) *error = "ccso plan: bad offset table";
        return false;
      }
    }
  }
  const restoration::RestorationStatus status = restoration::ccso_frame16(
      frame, params);
  if (status != restoration::RestorationStatus::kOk) {
    if (error != nullptr) {
      *error = std::string("hbd ccso: ") +
               restoration::restoration_status_string(status);
    }
    return false;
  }
  return true;
}

}  // namespace av2enc::hbd

#endif  // AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_CCSO_H_
