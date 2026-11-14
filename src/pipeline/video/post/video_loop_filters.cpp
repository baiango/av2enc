#include "pipeline/video/post/video_loop_filters.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "codec/filters/cdef/cdef.h"
#include "codec/filters/cdef/syntax/video_cdef.h"
#include "codec/filters/loop_restoration/restoration.h"
#include "codec/filters/loop_restoration/restoration_common.h"
#include "codec/filters/stream/filters_stream.h"
#include "codec/filters/loop_restoration/classes/wiener_decision.h"
#include "codec/filters/pc_wiener/pc_wiener_stream.h"
#include "codec/filters/gdf/gdf_decision.h"

namespace av2enc::video::post {
namespace {

namespace rest = ::av2enc::restoration;

bool fail(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

int plane_shift(int plane) { return plane == 0 ? 0 : 1; }

bool validate_frame(const VideoLoopFilterFrame& frame, bool need_source,
                    std::string* error) {
  if (frame.width <= 0 || frame.height <= 0 || frame.mi_cols <= 0 ||
      frame.mi_rows <= 0) {
    return fail(error, "video loop filters: invalid frame geometry");
  }
  if ((frame.mi_cols & 1) != 0 || (frame.mi_rows & 1) != 0) {
    return fail(error, "video loop filters: the 4:2:0 MI grid must be even");
  }
  if (frame.width > frame.mi_cols * 4 || frame.height > frame.mi_rows * 4) {
    return fail(error,
                "video loop filters: visible area exceeds the MI-aligned "
                "planes");
  }
  for (int plane = 0; plane < 3; ++plane) {
    const int width = (frame.mi_cols * 4) >> plane_shift(plane);
    if (frame.plane[plane] == nullptr || frame.stride[plane] < width) {
      return fail(error, "video loop filters: null or short plane");
    }
    if (need_source &&
        (frame.source[plane] == nullptr ||
         frame.source_stride[plane] < width)) {
      return fail(error,
                  "video loop filters: null or short source plane");
    }
  }
  return true;
}

int plane_width(const VideoLoopFilterFrame& frame, int plane) {
  return (frame.mi_cols * 4) >> plane_shift(plane);
}

int plane_height(const VideoLoopFilterFrame& frame, int plane) {
  return (frame.mi_rows * 4) >> plane_shift(plane);
}

void copy_plane(const std::uint8_t* src, int src_stride, std::uint8_t* dst,
                int dst_stride, int width, int height) {
  for (int y = 0; y < height; ++y) {
    std::copy_n(src + static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(src_stride),
                width,
                dst + static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(dst_stride));
  }
}

// ---------------------------------------------------------------------------
// CDEF
// ---------------------------------------------------------------------------

cdef_adapter::FrameView make_cdef_view(const VideoLoopFilterFrame& frame,
                                       const VideoCdefPlan& plan) {
  cdef_adapter::FrameView view;
  view.width = frame.width;
  view.height = frame.height;
  view.mi_cols = frame.mi_cols;
  view.mi_rows = frame.mi_rows;
  for (int plane = 0; plane < 3; ++plane) {
    view.recon[plane].data = frame.plane[plane];
    view.recon[plane].stride = frame.stride[plane];
    view.source[plane].data = frame.source[plane];
    view.source[plane].stride = frame.source_stride[plane];
  }
  if (!plan.skip.empty()) view.skip = plan.skip.data();
  if (!plan.lossless.empty()) view.lossless = plan.lossless.data();
  if (!plan.lossless_chroma.empty()) {
    view.lossless_chroma = plan.lossless_chroma.data();
  }
  return view;
}

bool prepare_cdef(const VideoLoopFilterFrame& frame,
                  const VideoLoopFilterOptions& options,
                  VideoCdefPlan* plan, std::string* error) {
  *plan = VideoCdefPlan{};
  if (options.cdef_damping < 3 || options.cdef_damping > 6) {
    return fail(error, "video CDEF: damping must be in [3, 6]");
  }
  if (options.cdef_num_strengths < 1 || options.cdef_num_strengths > 8) {
    return fail(error, "video CDEF: strength count must be in [1, 8]");
  }
  if (options.cdef_on_skip_txfm_mode != 0) {
    return fail(error,
                "video CDEF: only CdefOnSkipTxfm DISABLED is wired on the M2 "
                "stream (the tile writer codes no skip flag on intra leaves)");
  }

  const std::size_t mi_cells = static_cast<std::size_t>(frame.mi_cols) *
                               static_cast<std::size_t>(frame.mi_rows);
  plan->skip.assign(mi_cells, 0);
  if (!frame.blocks.empty()) {
    for (const VideoLoopFilterBlock& block : frame.blocks) {
      if (!block.skip) continue;
      for (int yy = 0; yy < block.h4; ++yy) {
        for (int xx = 0; xx < block.w4; ++xx) {
          const int rr = block.mi_row + yy;
          const int cc = block.mi_col + xx;
          if (rr < 0 || cc < 0 || rr >= frame.mi_rows ||
              cc >= frame.mi_cols) {
            continue;
          }
          plan->skip[static_cast<std::size_t>(rr) *
                         static_cast<std::size_t>(frame.mi_cols) +
                     static_cast<std::size_t>(cc)] = 1;
        }
      }
    }
  } else if (frame.skip != nullptr) {
    std::copy_n(frame.skip, mi_cells, plan->skip.begin());
  }
  // §7.18.1 LosslessArray[SegmentIds[mi]]: the M2 baseline codes no lossless
  // segments, so every MI cell is non-lossless.
  plan->lossless.assign(mi_cells, 0);
  plan->lossless_chroma.assign(mi_cells, 0);

  cdef_adapter::FrameView view = make_cdef_view(frame, *plan);
  cdef_adapter::Options cdef_options;
  cdef_options.damping = options.cdef_damping;
  cdef_options.num_strengths = options.cdef_num_strengths;
  cdef_options.on_skip_txfm = false;
  cdef_options.cdef_on_skip_txfm = options.cdef_on_skip_txfm_mode;
  cdef_options.disable_loopfilters_across_tiles = false;
  // With >1 strength the all-zero set must exist so a block can be left
  // unfiltered: §5.20.10.1 has no "off" index outside the strength table.
  cdef_options.include_off = options.cdef_num_strengths > 1;

  cdef_adapter::Plan adapter_plan;
  if (!cdef_adapter::plan(view, cdef_options, &adapter_plan, error)) {
    return false;
  }
  // The search may prune unused strength sets, so the effective CdefStrengths
  // can be smaller than the requested count (and is 1 when only one set is
  // used, which codes no cdef_idx symbol at all).  The header carries the
  // effective count.
  if (adapter_plan.num_strengths < 1 || adapter_plan.num_strengths > 8) {
    return fail(error, "video CDEF: invalid effective strength count");
  }
  if (adapter_plan.num_strengths == 1) {
    for (std::int8_t idx : adapter_plan.cdef_idx) {
      if (idx != 0) {
        return fail(error,
                    "video CDEF: single strength requires cdef_idx == 0");
      }
    }
  }

  plan->enabled = true;
  plan->damping = adapter_plan.options.damping;
  plan->num_strengths = adapter_plan.num_strengths;
  plan->on_skip_txfm = adapter_plan.options.on_skip_txfm;
  for (int i = 0; i < plan->num_strengths; ++i) {
    const av2enc::cdef::CdefStrength& src = adapter_plan.options.strengths[i];
    plan->strength[i].y_pri = src.y_pri;
    plan->strength[i].y_sec = src.y_sec;
    plan->strength[i].uv_pri = src.uv_pri;
    plan->strength[i].uv_sec = src.uv_sec;
  }
  plan->n64_cols = adapter_plan.n64_cols;
  plan->n64_rows = adapter_plan.n64_rows;
  plan->index = adapter_plan.cdef_idx;
  return true;
}

bool apply_cdef_cpu(VideoLoopFilterFrame& frame, const VideoCdefPlan& plan,
                    std::string* error) {
  if (plan.damping < 3 || plan.damping > 6 || plan.num_strengths < 1 ||
      plan.num_strengths > 8) {
    return fail(error, "video CDEF: invalid plan");
  }
  if (static_cast<int>(plan.index.size()) != plan.n64_cols * plan.n64_rows) {
    return fail(error, "video CDEF: index grid size mismatch");
  }

  av2enc::cdef::CdefOptions options;
  const CdefFilterOptions derived = cdef_filter_options(plan);
  options.damping = derived.damping;
  options.num_strengths = derived.num_strengths;
  options.on_skip_txfm = derived.on_skip_txfm;
  options.disable_loopfilters_across_tiles = false;
  options.bit_depth = 8;
  for (int i = 0; i < derived.num_strengths; ++i) {
    options.strengths[i].y_pri = derived.strength[i].y_pri;
    options.strengths[i].y_sec = derived.strength[i].y_sec;
    options.strengths[i].uv_pri = derived.strength[i].uv_pri;
    options.strengths[i].uv_sec = derived.strength[i].uv_sec;
  }

  // CDEF reads CurrFrame and writes a separate CdefFrame: the destination is a
  // scratch copy so a block never observes another block's output.
  std::vector<std::uint8_t> dst_storage[3];
  av2enc::cdef::CdefFrame cdef_frame{};
  for (int plane = 0; plane < 3; ++plane) {
    const int w = plane_width(frame, plane);
    const int h = plane_height(frame, plane);
    dst_storage[plane].assign(
        static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
    cdef_frame.planes[plane].src = frame.plane[plane];
    cdef_frame.planes[plane].dst = dst_storage[plane].data();
    cdef_frame.planes[plane].stride = frame.stride[plane];
  }
  cdef_frame.mi_cols = frame.mi_cols;
  cdef_frame.mi_rows = frame.mi_rows;
  cdef_frame.subsampling_x = 1;
  cdef_frame.subsampling_y = 1;
  cdef_frame.skip = plan.skip.data();
  cdef_frame.lossless = plan.lossless.data();
  cdef_frame.lossless_chroma = plan.lossless_chroma.data();
  cdef_frame.cdef_idx = plan.index.data();
  cdef_frame.n64_cols = plan.n64_cols;
  cdef_frame.n64_rows = plan.n64_rows;

  const av2enc::cdef::CdefStatus status =
      av2enc::cdef::cdef_frame(cdef_frame, options);
  if (status != av2enc::cdef::CdefStatus::kOk) {
    return fail(error, std::string("video CDEF failed: ") +
                           av2enc::cdef::cdef_status_string(status));
  }
  for (int plane = 0; plane < 3; ++plane) {
    copy_plane(dst_storage[plane].data(), plane_width(frame, plane),
               frame.plane[plane], frame.stride[plane],
               plane_width(frame, plane), plane_height(frame, plane));
  }
  return true;
}

// ---------------------------------------------------------------------------
// CCSO
// ---------------------------------------------------------------------------

av2enc::still::filters_stream::CcsoParams make_ccso_header_params(
    const VideoCcsoPlan& plan) {
  av2enc::still::filters_stream::CcsoParams params;
  params.frame_flag = plan.enabled;
  params.ccso_luma_size_log2 = plan.ccso_luma_size_log2;
  for (int plane = 0; plane < 3; ++plane) {
    const VideoCcsoPlane& src = plan.plane[plane];
    av2enc::still::filters_stream::CcsoPlaneParams& dst = params.plane[plane];
    dst.enabled = src.enabled;
    dst.bo_only = src.bo_only;
    dst.scale_idx = src.scale_idx;
    dst.quant_idx = src.quant_idx;
    dst.ext_filter = src.ext_filter;
    dst.edge_clf = src.edge_clf;
    dst.max_band_log2 = src.max_band_log2;
    dst.offset_idx = src.offset_idx;
  }
  return params;
}

bool prepare_ccso(const VideoLoopFilterFrame& frame,
                  const VideoLoopFilterOptions& options, VideoCcsoPlan* plan,
                  std::string* error) {
  *plan = VideoCcsoPlan{};
  if (!options.ccso) return true;
  if (options.ccso_bands_log2 < 0 || options.ccso_bands_log2 > 6) {
    return fail(error, "video CCSO: band count log2 must be in [0, 6]");
  }
  const int bands_log2 = options.ccso_bands_log2;
  const int bands = 1 << bands_log2;
  if (bands > static_cast<int>(rest::kCcsoOffsetEntries)) {
    return fail(error, "video CCSO: too many bands for the offset table");
  }
  // §5.18.7.12 with ccso_unit_matches_sb_size = 1 and a 64x64 superblock.
  const int ccso_luma_log2 = 6;
  plan->ccso_luma_size_log2 = ccso_luma_log2;

  const std::uint8_t* curr_luma =
      frame.curr[0] != nullptr ? frame.curr[0] : frame.plane[0];
  const int curr_luma_stride =
      frame.curr[0] != nullptr ? frame.curr_stride[0] : frame.stride[0];
  const int luma_w = plane_width(frame, 0);
  const int luma_h = plane_height(frame, 0);

  int cols = 0;
  int rows = 0;
  rest::ccso_blk_grid_size(luma_w, luma_h, ccso_luma_log2, 0, 0, &cols, &rows);

  for (int plane = 0; plane < 3; ++plane) {
    VideoCcsoPlane& plane_plan = plan->plane[plane];
    const int sub = plane == 0 ? 0 : 1;
    const int pw = plane_width(frame, plane);
    const int ph = plane_height(frame, plane);
    const int band_shift = 8 - bands_log2;
    const std::uint8_t* rec = frame.plane[plane];
    const int rec_stride = frame.stride[plane];
    const std::uint8_t* src = frame.source[plane];
    const int src_stride = frame.source_stride[plane];

    std::vector<double> sum_e(static_cast<std::size_t>(bands), 0.0);
    std::vector<double> sum_e2(static_cast<std::size_t>(bands), 0.0);
    std::vector<double> count(static_cast<std::size_t>(bands), 0.0);
    for (int y = 0; y < ph; ++y) {
      const int lx_row = (y << sub) * curr_luma_stride;
      for (int x = 0; x < pw; ++x) {
        const int lx = x << sub;
        const int band = curr_luma[lx_row + lx] >> band_shift;
        const int src_value =
            src[static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(src_stride) +
                static_cast<std::size_t>(x)];
        const int d =
            src_value - rec[static_cast<std::size_t>(y) *
                                static_cast<std::size_t>(rec_stride) +
                            static_cast<std::size_t>(x)];
        sum_e[static_cast<std::size_t>(band)] += d;
        sum_e2[static_cast<std::size_t>(band)] +=
            static_cast<double>(d) * static_cast<double>(d);
        count[static_cast<std::size_t>(band)] += 1.0;
      }

    }

    plane_plan.enabled = true;
    plane_plan.bo_only = true;
    plane_plan.scale_idx = 0;
    plane_plan.quant_idx = 0;
    plane_plan.ext_filter = 0;
    plane_plan.edge_clf = 0;
    plane_plan.max_band_log2 = bands_log2;
    plane_plan.offset_idx.assign(static_cast<std::size_t>(bands), 0);
    if (options.ccso_rd) {
      // Rate-aware fit.  §5.18.7.12 codes each ccso_offset_idx as tu(7), so
      // index i costs i extra bits over the index-0 "off" symbol (index 7 is
      // the tu(7) escape and costs 6).  Pick the offset per band by
      // SSE + lambda * table bits, sweep the free 2-bit scale_idx and then do
      // one exact clamped pass per plane to decide the per-unit ccso_blk
      // symbol.  A disabled unit/plane is left untouched by both sides.
      const double lambda = options.ccso_lambda;
      auto unary_extra_bits = [](int idx) { return idx < 7 ? idx : 6; };
      double best_cost = std::numeric_limits<double>::infinity();
      int best_scale = 0;
      std::vector<std::uint8_t> best_idx(static_cast<std::size_t>(bands), 0);
      for (int scale_idx = 0; scale_idx < 4; ++scale_idx) {
        const int scale = scale_idx + 1;
        double cost = 0.0;
        std::vector<std::uint8_t> idx(static_cast<std::size_t>(bands), 0);
        for (int band = 0; band < bands; ++band) {
          int band_idx = 0;
          double band_cost = sum_e2[static_cast<std::size_t>(band)];
          for (int i = 0; i < 8; ++i) {
            const double off =
                static_cast<double>(rest::kCcsoOffset[i] * scale);
            const double sse =
                sum_e2[static_cast<std::size_t>(band)] -
                2.0 * off * sum_e[static_cast<std::size_t>(band)] +
                off * off * count[static_cast<std::size_t>(band)];
            const double c = sse + lambda * unary_extra_bits(i);
            if (c < band_cost - 1e-9) {
              band_cost = c;
              band_idx = i;
            }
          }
          cost += band_cost;
          idx[static_cast<std::size_t>(band)] =
              static_cast<std::uint8_t>(band_idx);
        }
        if (cost < best_cost - 1e-9) {
          best_cost = cost;
          best_scale = scale_idx;
          best_idx = std::move(idx);
        }
      }
      plane_plan.scale_idx = best_scale;
      plane_plan.offset_idx = best_idx;

      int pcols = 0;
      int prows = 0;
      rest::ccso_blk_grid_size(pw, ph, ccso_luma_log2, sub, sub, &pcols,
                               &prows);
      const int shift = ccso_luma_log2 - sub;
      std::vector<double> unit_off(
          static_cast<std::size_t>(pcols) * static_cast<std::size_t>(prows),
          0.0);
      std::vector<double> unit_on(unit_off.size(), 0.0);
      const int scale = best_scale + 1;
      for (int y = 0; y < ph; ++y) {
        const int lx_row = (y << sub) * curr_luma_stride;
        for (int x = 0; x < pw; ++x) {
          const int lx = x << sub;
          const int band = curr_luma[lx_row + lx] >> band_shift;
          const int off =
              rest::kCcsoOffset[best_idx[static_cast<std::size_t>(band)]] *
              scale;
          const int base =
              rec[static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(rec_stride) +
                  static_cast<std::size_t>(x)];
          const int src_value =
              src[static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(src_stride) +
                  static_cast<std::size_t>(x)];
          const int filtered = std::clamp(base + off, 0, 255);
          const std::size_t unit =
              static_cast<std::size_t>(y >> shift) *
                  static_cast<std::size_t>(pcols) +
              static_cast<std::size_t>(x >> shift);
          const double d_off = static_cast<double>(src_value - base);
          const double d_on = static_cast<double>(src_value - filtered);
          unit_off[unit] += d_off * d_off;
          unit_on[unit] += d_on * d_on;
        }
      }
      // The ccso_blk symbol is a 2-symbol adaptive CDF symbol, ~0.5-1 bit.
      constexpr double kUnitSymbolBits = 1.0;
      plane_plan.blk_cols = pcols;
      plane_plan.blk_rows = prows;
      plane_plan.blks.assign(unit_on.size(), 0);
      std::size_t enabled_units = 0;
      for (std::size_t u = 0; u < unit_on.size(); ++u) {
        if (unit_off[u] - unit_on[u] > lambda * kUnitSymbolBits) {
          plane_plan.blks[u] = 1;
          ++enabled_units;
        }
      }
      if (enabled_units == 0) {
        plane_plan.enabled = false;
        plane_plan.blks.clear();
      }
      continue;
    }
    int nonzero = 0;
    for (int band = 0; band < bands; ++band) {
      int best_idx = 0;
      double best_sse = sum_e2[static_cast<std::size_t>(band)];
      for (int i = 0; i < 8; ++i) {
        const double off = static_cast<double>(rest::kCcsoOffset[i]);
        const double sse =
            sum_e2[static_cast<std::size_t>(band)] -
            2.0 * off * sum_e[static_cast<std::size_t>(band)] +
            off * off * count[static_cast<std::size_t>(band)];
        if (sse < best_sse - 1e-9) {
          best_sse = sse;
          best_idx = i;
        }
      }
      plane_plan.offset_idx[static_cast<std::size_t>(band)] =
          static_cast<std::uint8_t>(best_idx);
      if (best_idx != 0) ++nonzero;
    }
    // CcsoBlks: one enabled unit per CCSO unit (the still band-offset path
    // signals the same all-ones grid).  Luma must have a non-zero offset set
    // or the plane is disabled; chroma planes cannot be disabled by §5.4.10
    // separately, so an all-zero chroma fit still costs only header bits.
    plane_plan.blk_cols = cols;
    plane_plan.blk_rows = rows;
    plane_plan.blks.assign(static_cast<std::size_t>(cols) *
                               static_cast<std::size_t>(rows),
                           1);
    if (nonzero == 0 && plane == 0) {
      plane_plan.enabled = false;
      plane_plan.blks.clear();
    }
  }

  bool any_plane = false;
  for (int plane = 0; plane < 3; ++plane) {
    any_plane = any_plane || plan->plane[plane].enabled;
  }
  plan->enabled = any_plane;
  if (!any_plane) return true;

  // Validate through the real header writer before touching any pixel.
  av2enc::still::filters_stream::CcsoHeaderContext context;
  context.coded_lossless = false;
  context.enable_ccso = true;
  context.single_picture_header_flag = false;
  context.frame_is_intra = frame.frame_is_intra;
  context.num_planes = 3;
  context.sb_size = 64;
  context.ccso_unit_matches_sb_size = true;
  context.tile_start_or = 0;
  av2enc::still::filters_stream::CcsoParams params =
      make_ccso_header_params(*plan);
  BitWriter probe;
  if (av2enc::still::filters_stream::write_ccso_params(probe, params, context,
                                                       error) == 0 &&
      (error != nullptr && !error->empty())) {
    return false;
  }
  return true;
}

bool apply_ccso_cpu(const VideoLoopFilterFrame& frame,
                    const VideoCcsoPlan& plan, std::string* error) {
  if (!plan.enabled) return true;
  rest::CcsoFrameParams params;
  params.ccso_luma_size_log2 = plan.ccso_luma_size_log2;
  for (int plane = 0; plane < 3; ++plane) {
    const VideoCcsoPlane& src = plan.plane[plane];
    rest::CcsoParams& dst = params.plane[plane];
    dst.enabled = src.enabled;
    dst.bo_only = src.bo_only;
    dst.scale_idx = src.scale_idx;
    dst.quant_idx = src.quant_idx;
    dst.ext_filter = src.ext_filter;
    dst.edge_clf = src.edge_clf;
    dst.max_band_log2 = src.max_band_log2;
    if (src.enabled) {
      if (!rest::ccso_set_offsets(dst, src.offset_idx.data(),
                                  src.offset_idx.size())) {
        return fail(error, "video CCSO: invalid offset index table");
      }
      params.blks[plane] = src.blks.data();
      params.blk_cols[plane] = src.blk_cols;
      params.blk_rows[plane] = src.blk_rows;
    }
  }

  rest::RestorationFrame rf{};
  for (int plane = 0; plane < 3; ++plane) {
    rf.curr[plane].data = const_cast<std::uint8_t*>(
        frame.curr[plane] != nullptr ? frame.curr[plane] : frame.plane[plane]);
    rf.curr[plane].stride = frame.curr[plane] != nullptr
                                ? frame.curr_stride[plane]
                                : frame.stride[plane];
    rf.cdef[plane].data = frame.plane[plane];
    rf.cdef[plane].stride = frame.stride[plane];
    rf.lr[plane].data = frame.plane[plane];
    rf.lr[plane].stride = frame.stride[plane];
  }
  rf.mi_cols = frame.mi_cols;
  rf.mi_rows = frame.mi_rows;
  rf.subsampling_x = 1;
  rf.subsampling_y = 1;
  rf.bit_depth = 8;
  rf.sb_width = 64;
  rf.sb_height = 64;
  rf.sb_shift = 4;
  const rest::RestorationStatus status = rest::ccso_frame(rf, params);
  if (status != rest::RestorationStatus::kOk) {
    return fail(error, std::string("video CCSO failed: ") +
                           rest::restoration_status_string(status));
  }
  return true;
}

// ---------------------------------------------------------------------------
// Loop restoration (§7.20 luma Wiener NS and chroma LR)
// ---------------------------------------------------------------------------

// §7.20.1 LrTxSkip = skip_flag || eob == 0 on the luma MI grid.  The decision
// and the filter application must use the identical grid the decoder derives.
std::vector<std::uint8_t> build_lr_tx_skip(const VideoLoopFilterFrame& frame) {
  std::vector<std::uint8_t> lr_tx_skip(
      static_cast<std::size_t>(frame.mi_cols) *
          static_cast<std::size_t>(frame.mi_rows),
      0);
  for (const VideoLoopFilterBlock& block : frame.blocks) {
    const bool tx_skip = block.skip || block.luma_eob_zero;
    for (int yy = 0; yy < block.h4; ++yy) {
      for (int xx = 0; xx < block.w4; ++xx) {
        const int rr = block.mi_row + yy;
        const int cc = block.mi_col + xx;
        if (rr < 0 || cc < 0 || rr >= frame.mi_rows || cc >= frame.mi_cols) {
          continue;
        }
        lr_tx_skip[static_cast<std::size_t>(rr) *
                       static_cast<std::size_t>(frame.mi_cols) +
                   static_cast<std::size_t>(cc)] = tx_skip ? 1 : 0;
      }
    }
  }
  return lr_tx_skip;
}

bool prepare_lr(const VideoLoopFilterFrame& frame,
                const VideoLoopFilterOptions& options, VideoLrPlan* plan,
                std::string* error) {
  *plan = VideoLrPlan{};
  if (!options.wiener && !options.chroma_lr && !options.pc_wiener) return true;
  if (options.pc_wiener && options.wiener) {
    return fail(error,
                "video LR: --video-pc-wiener replaces the luma Wiener NS "
                "decision; do not combine the two luma tools");
  }

  plan->base_q_idx = frame.base_q_idx;
  // §7.20.4 luma PC-Wiener: its own decision and §5.18.7.11 dispatch.
  if (options.pc_wiener) {
    namespace pcw = ::av2enc::filters::pc_wiener_stream;
    const int luma_w = frame.mi_cols * 4;
    const int luma_h = frame.mi_rows * 4;
    pcw::DecisionPlanes planes;
    planes.width = luma_w;
    planes.height = luma_h;
    planes.bit_depth = 8;
    planes.base_q_idx = frame.base_q_idx;
    planes.sub_x = 1;
    planes.sub_y = 1;
    planes.luma_shift = 3;
    planes.chroma_shift = 3;
    for (int plane = 0; plane < 3; ++plane) {
      planes.curr[plane] =
          frame.curr[plane] != nullptr ? frame.curr[plane] : frame.plane[plane];
      planes.curr_stride[plane] = frame.curr[plane] != nullptr
                                      ? frame.curr_stride[plane]
                                      : frame.stride[plane];
      planes.cdef[plane] = frame.plane[plane];
      planes.cdef_stride[plane] = frame.stride[plane];
      planes.target[plane] = frame.source[plane];
      planes.target_stride[plane] = frame.source_stride[plane];
    }
    std::vector<std::uint8_t> lr_tx_skip = build_lr_tx_skip(frame);
    planes.lr_tx_skip = lr_tx_skip.data();
    planes.lr_tx_skip_stride = frame.mi_cols;
    plan->tx_skip = lr_tx_skip;
    plan->tx_skip_cols = frame.mi_cols;
    plan->tx_skip_rows = frame.mi_rows;
    pcw::DecisionResult result;
    std::string pc_error;
    if (!pcw::DecidePerUnitPcWiener(planes, 0.0, &result, &pc_error)) {
      return fail(error, "video PC-Wiener decision failed: " + pc_error);
    }
    // The sequence header is emitted before the frame is coded, so the tool
    // is signalled up front.  Every unit then codes its use_pc_wiener symbol;
    // an all-off frame is a valid (identity) PC-Wiener plan.
    const int unit_size =
        pcw::kRestorationTileSizeMax >> result.params.luma_shift;
    VideoLrPlane& dst = plan->plane[0];
    dst.enabled = true;
    dst.restore_type = static_cast<std::uint8_t>(pcw::kRestorePcWiener);
    dst.unit_size = unit_size;
    dst.shift = result.params.luma_shift;
    dst.unit_rows = pcw::CountUnits(unit_size, luma_h);
    dst.unit_cols = pcw::CountUnits(unit_size, luma_w);
    dst.unit_type = result.unit_type[0];
    plan->pc_wiener = true;
    plan->luma_shift = result.params.luma_shift;
    plan->enabled = true;
  }
  if (!options.wiener && !options.chroma_lr && !plan->enabled) return true;

  namespace classes = ::av2enc::restoration::classes;
  classes::WienerDecisionInput input;
  for (int plane = 0; plane < 3; ++plane) {
    input.core.plane[plane].curr =
        frame.curr[plane] != nullptr ? frame.curr[plane] : frame.plane[plane];
    input.core.plane[plane].curr_stride =
        frame.curr[plane] != nullptr ? frame.curr_stride[plane]
                                     : frame.stride[plane];
    input.core.plane[plane].cdef = frame.plane[plane];
    input.core.plane[plane].cdef_stride = frame.stride[plane];
    input.target[plane] = frame.source[plane];
    input.target_stride[plane] = frame.source_stride[plane];
  }
  input.core.mi_cols = frame.mi_cols;
  input.core.mi_rows = frame.mi_rows;
  input.core.sub_x = 1;
  input.core.sub_y = 1;
  input.core.bit_depth = 8;
  // §7.20.1 sbShift = Mi_Width_Log2[SbSize] (4 for 64x64).
  input.core.sb_shift = 4;
  input.base_q_idx = frame.base_q_idx;

  classes::WienerDecisionOptions lr_options;
  lr_options.allow_luma = options.wiener && !options.pc_wiener;
  lr_options.allow_chroma = options.chroma_lr;
  lr_options.allow_per_unit = true;
  // Frame-level filters need an intra frame and a §7.20.4 class map; the
  // per-unit form is the gated path and composes with P frames.
  lr_options.allow_frame_filters = false;
  lr_options.max_num_filter_classes = 1;
  lr_options.lambda_bits = 0.0;
  // One unit per 64x64 superblock keeps read_lr()'s per-SB visit shape; the
  // chroma unit is then 32x32 for 4:2:0.
  lr_options.forced_luma_shift = 3;
  lr_options.forced_chroma_shift = 3;

  classes::WienerDecision decision;
  const classes::WienerDecisionStatus status =
      classes::wiener_decide(input, lr_options, &decision);
  if (status != classes::WienerDecisionStatus::kOk) {
    return fail(error, std::string("video LR decision failed: ") +
                           classes::wiener_decision_status_string(status));
  }

  if (!plan->pc_wiener) plan->luma_shift = decision.params.luma_shift;
  plan->chroma_shift = decision.params.chroma_shift;
  bool any_plane = false;
  for (int plane = 0; plane < 3; ++plane) {
    VideoLrPlane& dst = plan->plane[plane];
    const int type = decision.params.frame_type[plane];
    // §5.18.7.11 codes one LoopRestorationSize per chroma plane pair even when
    // a plane's FrameRestorationType is RESTORE_NONE; keep the derived unit
    // size on every plane so params.unit_size[1] == params.unit_size[2] and the
    // tile walk matches the plan vectors.
    dst.unit_size = decision.unit_size[plane];
    dst.shift = plane == 0 ? plan->luma_shift : plan->chroma_shift;
    dst.unit_rows = decision.unit_rows[plane];
    dst.unit_cols = decision.unit_cols[plane];
    if (type == rest::kRestoreNone) continue;
    if (type != rest::kRestoreWienerNs) {
      return fail(error,
                  "video LR: only RESTORE_WIENER_NONSEP is wired (PC-Wiener "
                  "is a separate tool)");
    }
    dst.enabled = true;
    dst.restore_type = static_cast<std::uint8_t>(rest::kRestoreWienerNs);
    dst.unit_type = decision.unit_type[plane];
    dst.unit_coeffs = decision.unit_coeffs[plane];
    if (plane > 0) {
      // The shared §5.20.10.6 chroma writer codes coefficient pairs once
      // (wiener_ns_uv_sym), so every chroma filter must be pair-symmetric.
      // Mirror the fitted pairs here so the signalled filter and the applied
      // filter are identical.
      const std::size_t units = static_cast<std::size_t>(dst.unit_rows) *
                                static_cast<std::size_t>(dst.unit_cols);
      for (std::size_t unit = 0; unit < units; ++unit) {
        std::int16_t* coeffs =
            dst.unit_coeffs.data() + unit * rest::kWienerNsCoeffStride;
        for (int j = rest::kWienerNsShortCoeffs; j + 1 < rest::kWienerNsChromaCoeffs;
             j += 2) {
          coeffs[j + 1] = coeffs[j];
        }
      }
    }
    any_plane = true;
  }
  plan->enabled = any_plane || plan->enabled;
  return true;
}

bool apply_lr_cpu(const VideoLoopFilterFrame& frame, const VideoLrPlan& plan,
                  std::string* error) {
  if (!plan.enabled) return true;
  std::vector<std::uint8_t> lr_storage[3];
  for (int plane = 0; plane < 3; ++plane) {
    const int w = plane_width(frame, plane);
    const int h = plane_height(frame, plane);
    lr_storage[plane].assign(
        static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  }

  // §7.20.4 luma PC-Wiener (RESTORE_PC_WIENER): per-unit on/off from the plan.
  // loop_restoration_frame() begins with LrFrame := CdefFrame for every plane;
  // keep the PC-Wiener luma so a chroma-only WNS pass cannot replace it.
  std::vector<std::uint8_t> pc_wiener_luma;
  if (plan.pc_wiener) {
    namespace pcw = ::av2enc::filters::pc_wiener_stream;
    const int luma_w = frame.mi_cols * 4;
    const int luma_h = frame.mi_rows * 4;
    pcw::DecisionPlanes planes;
    planes.width = luma_w;
    planes.height = luma_h;
    planes.bit_depth = 8;
    planes.base_q_idx = frame.base_q_idx;
    planes.sub_x = 1;
    planes.sub_y = 1;
    for (int plane = 0; plane < 3; ++plane) {
      planes.curr[plane] =
          frame.curr[plane] != nullptr ? frame.curr[plane] : frame.plane[plane];
      planes.curr_stride[plane] = frame.curr[plane] != nullptr
                                      ? frame.curr_stride[plane]
                                      : frame.stride[plane];
      planes.cdef[plane] = frame.plane[plane];
      planes.cdef_stride[plane] = frame.stride[plane];
    }
    const std::vector<std::uint8_t> lr_tx_skip = build_lr_tx_skip(frame);
    planes.lr_tx_skip = lr_tx_skip.data();
    planes.lr_tx_skip_stride = frame.mi_cols;
    const int filtered = pcw::ApplyPerUnitPcWiener(
        planes, plan.plane[0].unit_type, plan.plane[0].unit_size,
        lr_storage[0].data(), luma_w);
    if (filtered < 0) {
      return fail(error, "video PC-Wiener application failed");
    }
  }

  if (plan.pc_wiener) pc_wiener_luma = lr_storage[0];

  bool any_wns = false;
  rest::LoopRestorationParams params;
  for (int plane = 0; plane < 3; ++plane) {
    const VideoLrPlane& src = plan.plane[plane];
    const bool wns = src.enabled &&
                     src.restore_type ==
                         static_cast<std::uint8_t>(rest::kRestoreWienerNs);
    params.frame_type[plane] = wns ? rest::kRestoreWienerNs : rest::kRestoreNone;
    params.unit_size[plane] = src.unit_size;
    params.unit_type[plane] = wns ? src.unit_type.data() : nullptr;
    params.unit_rows[plane] = src.unit_rows;
    params.unit_cols[plane] = src.unit_cols;
    params.unit_coeffs[plane] = wns ? src.unit_coeffs.data() : nullptr;
    params.frame_filters_on[plane] = 0;
    any_wns = any_wns || wns;
  }

  if (any_wns) {
    rest::RestorationFrame rf{};
    for (int plane = 0; plane < 3; ++plane) {
      const int w = plane_width(frame, plane);
      rf.curr[plane].data = const_cast<std::uint8_t*>(
          frame.curr[plane] != nullptr ? frame.curr[plane] : frame.plane[plane]);
      rf.curr[plane].stride = frame.curr[plane] != nullptr
                                  ? frame.curr_stride[plane]
                                  : frame.stride[plane];
      rf.cdef[plane].data = frame.plane[plane];
      rf.cdef[plane].stride = frame.stride[plane];
      rf.lr[plane].data = lr_storage[plane].data();
      rf.lr[plane].stride = w;
    }
    rf.mi_cols = frame.mi_cols;
    rf.mi_rows = frame.mi_rows;
    rf.subsampling_x = 1;
    rf.subsampling_y = 1;
    rf.bit_depth = 8;
    rf.sb_width = 64;
    rf.sb_height = 64;
    rf.sb_shift = 4;
    const rest::RestorationStatus status =
        rest::loop_restoration_frame(rf, params);
    if (status != rest::RestorationStatus::kOk) {
      return fail(error, std::string("video LR failed: ") +
                             rest::restoration_status_string(status));
    }
  }
  if (plan.pc_wiener) {
    lr_storage[0] = std::move(pc_wiener_luma);
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (!plan.plane[plane].enabled) continue;
    copy_plane(lr_storage[plane].data(), plane_width(frame, plane),
               frame.plane[plane], frame.stride[plane],
               plane_width(frame, plane), plane_height(frame, plane));
  }
  return true;
}

// ---------------------------------------------------------------------------
// GDF (§7.20.5, luma only) — header-only mode (gdf_per_block = 0)
// ---------------------------------------------------------------------------

// Builds the §7.20.1 inputs at 16-bit sample width, padded to the MI-aligned
// luma plane (the source picture is edge-replicated past the visible area).
void make_gdf_planes(const VideoLoopFilterFrame& frame,
                     const std::uint8_t* const cdef_override[3],
                     std::vector<std::uint16_t>* curr16,
                     std::vector<std::uint16_t>* cdef16,
                     std::vector<std::uint16_t>* lr16,
                     std::vector<std::uint16_t>* original16) {
  const int w = plane_width(frame, 0);
  const int h = plane_height(frame, 0);
  const std::size_t size = static_cast<std::size_t>(w) *
                           static_cast<std::size_t>(h);
  curr16->assign(size, 0);
  cdef16->assign(size, 0);
  lr16->assign(size, 0);
  original16->assign(size, 0);
  const std::uint8_t* curr =
      frame.curr[0] != nullptr ? frame.curr[0] : frame.plane[0];
  const int curr_stride =
      frame.curr[0] != nullptr ? frame.curr_stride[0] : frame.stride[0];
  const std::uint8_t* cdef =
      cdef_override != nullptr && cdef_override[0] != nullptr
          ? cdef_override[0]
          : frame.plane[0];
  const int cdef_stride =
      cdef_override != nullptr && cdef_override[0] != nullptr
          ? frame.cdef_stride[0]
          : frame.stride[0];
  const int vis_w = frame.width;
  const int vis_h = frame.height;
  for (int y = 0; y < h; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(w);
    const int sy = y < vis_h ? y : (vis_h > 0 ? vis_h - 1 : 0);
    for (int x = 0; x < w; ++x) {
      (*curr16)[row + static_cast<std::size_t>(x)] = curr[
          static_cast<std::size_t>(y) * static_cast<std::size_t>(curr_stride) +
          static_cast<std::size_t>(x)];
      (*cdef16)[row + static_cast<std::size_t>(x)] = cdef[
          static_cast<std::size_t>(y) * static_cast<std::size_t>(cdef_stride) +
          static_cast<std::size_t>(x)];
      (*lr16)[row + static_cast<std::size_t>(x)] =
          static_cast<std::uint16_t>(
              frame.plane[0][static_cast<std::size_t>(y) *
                                 static_cast<std::size_t>(frame.stride[0]) +
                             static_cast<std::size_t>(x)]);
      const int sx = x < vis_w ? x : (vis_w > 0 ? vis_w - 1 : 0);
      (*original16)[row + static_cast<std::size_t>(x)] =
          frame.source[0][static_cast<std::size_t>(sy) *
                              static_cast<std::size_t>(frame.source_stride[0]) +
                          static_cast<std::size_t>(sx)];
    }
  }
}

::av2enc::filters::gdf::FrameGdfContext make_gdf_context(
    const VideoLoopFilterFrame& frame, bool enable) {
  ::av2enc::filters::gdf::FrameGdfContext context;
  context.enable_gdf = enable;
  context.gdf_unit_matches_sb_size = true;
  context.coded_lossless = false;
  context.single_picture_header_flag = false;
  context.sb_size = 64;
  context.width = frame.mi_cols * 4;
  context.height = frame.mi_rows * 4;
  context.mi_cols = frame.mi_cols;
  context.mi_rows = frame.mi_rows;
  context.disable_loopfilters_across_tiles = false;
  context.tile_cols = 1;
  context.tile_rows = 1;
  context.mi_col_starts.assign(1, 0);
  context.mi_row_starts.assign(1, 0);
  return context;
}

bool prepare_gdf(const VideoLoopFilterFrame& frame, VideoGdfPlan* plan,
                 std::string* error) {
  namespace gdf = ::av2enc::filters::gdf;
  *plan = VideoGdfPlan{};
  if (!frame.frame_is_intra && frame.base_q_idx == 0) {
    return fail(error, "video GDF: invalid base_q_idx");
  }
  std::vector<std::uint16_t> curr16;
  std::vector<std::uint16_t> cdef16;
  std::vector<std::uint16_t> lr16;
  std::vector<std::uint16_t> original16;
  make_gdf_planes(frame, frame.cdef, &curr16, &cdef16, &lr16, &original16);
  const int w = plane_width(frame, 0);
  const int h = plane_height(frame, 0);

  gdf::GdfSourceFrame source;
  source.curr = curr16.data();
  source.cdef = cdef16.data();
  source.stride = w;
  gdf::GdfPlane original;
  original.data = original16.data();
  // §7.20.1 runs over the coded (MI-aligned) picture, so the last unit's
  // stripe region is bounded by MiCols*4/MiRows*4, not the visible size.
  original.width = w;
  original.height = h;
  original.stride = w;

  const gdf::FrameGdfContext context = make_gdf_context(frame, true);
  gdf::GdfDecisionOptions options;
  options.filter.base_q_idx = frame.base_q_idx;
  options.filter.frame_is_intra = frame.frame_is_intra;
  options.filter.bit_depth = 8;
  options.filter.num_total_refs = frame.num_total_refs;
  options.filter.order_hints[0] = frame.ref_order_hints[0];
  options.filter.order_hints[1] = frame.ref_order_hints[1];
  options.filter.frame_distance[0] = frame.frame_distance[0];
  options.filter.frame_distance[1] = frame.frame_distance[1];
  options.filter.restricted_oh = frame.restricted_oh;
  options.disable_cdf_update = true;
  options.lambda_bits = 0.0;
  gdf::GdfDecision decision;
  if (!gdf::DecideGdf(source, original, context, options, &decision, error)) {
    return fail(error, "video GDF decision failed: " + *error);
  }
  if (!decision.params.frame_enable) return true;
  plan->enabled = true;
  plan->qc_idx = decision.params.pic_qc_idx;
  plan->scale_idx = decision.params.pic_scale_idx;
  plan->per_block = false;
  plan->frame_is_intra = frame.frame_is_intra;
  plan->base_q_idx = frame.base_q_idx;
  plan->num_total_refs = frame.num_total_refs;
  for (int r = 0; r < 2; ++r) {
    plan->order_hints[r] = frame.ref_order_hints[r];
    plan->frame_distance[r] = frame.frame_distance[r];
  }
  plan->restricted_oh = frame.restricted_oh;
  return true;
}

bool apply_gdf_cpu(const VideoLoopFilterFrame& frame, const VideoGdfPlan& plan,
                   std::string* error) {
  namespace gdf = ::av2enc::filters::gdf;
  if (!plan.enabled) return true;
  std::vector<std::uint16_t> curr16;
  std::vector<std::uint16_t> cdef16;
  std::vector<std::uint16_t> lr16;
  std::vector<std::uint16_t> original16;
  make_gdf_planes(frame, frame.cdef, &curr16, &cdef16, &lr16, &original16);
  const int w = plane_width(frame, 0);
  const int h = plane_height(frame, 0);
  gdf::GdfSourceFrame source;
  source.curr = curr16.data();
  source.cdef = cdef16.data();
  source.stride = w;
  const gdf::FrameGdfContext context = make_gdf_context(frame, true);
  gdf::GdfFilterInputs inputs;
  inputs.frame_is_intra = frame.frame_is_intra;
  inputs.base_q_idx = frame.base_q_idx;
  inputs.bit_depth = 8;
  inputs.num_total_refs = frame.num_total_refs;
  inputs.order_hints[0] = frame.ref_order_hints[0];
  inputs.order_hints[1] = frame.ref_order_hints[1];
  inputs.frame_distance[0] = frame.frame_distance[0];
  inputs.frame_distance[1] = frame.frame_distance[1];
  inputs.restricted_oh = frame.restricted_oh;
  gdf::FrameGdfParams params;
  params.frame_enable = true;
  params.per_block = false;
  params.pic_qc_idx = plan.qc_idx;
  params.pic_scale_idx = plan.scale_idx;
  // Derive GdfBlkSize / GdfBlks from the same §5.18.7.9 row the frame header
  // writes, so the applied unit grid matches the decoder's.
  {
    BitWriter scratch;
    std::string derive_error;
    gdf::WriteFrameGdfParams(scratch, context, &params, &derive_error);
    if (!derive_error.empty()) {
      return fail(error, "video GDF header derivation failed: " + derive_error);
    }
  }
  std::vector<std::uint16_t> out16(lr16);
  if (!gdf::ApplyGdfToFrame(source, inputs, context, params, {}, out16.data(),
                            w, error)) {
    return fail(error, "video GDF apply failed: " + *error);
  }
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      frame.plane[0][static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(frame.stride[0]) +
                     static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(
          out16[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(x)]);
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Stage driver
// ---------------------------------------------------------------------------

bool apply_stage(VideoFilterStage stage, VideoLoopFilterFrame& frame,
                 const VideoFilterPlan& plan, VideoLoopFilterHooks* hooks,
                 std::string* error) {
  if (hooks != nullptr) {
    std::vector<std::uint8_t> stage_out[3];
    VideoLoopFilterDeviceFrame device;
    for (int plane = 0; plane < 3; ++plane) {
      const int w = plane_width(frame, plane);
      const int h = plane_height(frame, plane);
      stage_out[plane].assign(
          static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
      device.in[plane] = frame.plane[plane];
      device.in_stride[plane] = frame.stride[plane];
      device.curr[plane] = frame.curr[plane];
      device.curr_stride[plane] = frame.curr_stride[plane];
      device.cdef[plane] = frame.cdef[plane];
      device.cdef_stride[plane] = frame.cdef_stride[plane];
      device.out[plane] = stage_out[plane].data();
      device.out_stride[plane] = w;
    }
    if (!hooks->apply_stage(stage, plan, device, error)) return false;
    for (int plane = 0; plane < 3; ++plane) {
      copy_plane(stage_out[plane].data(), plane_width(frame, plane),
                 frame.plane[plane], frame.stride[plane],
                 plane_width(frame, plane), plane_height(frame, plane));
    }
    return true;
  }
  switch (stage) {
    case VideoFilterStage::kCdef:
      return apply_cdef_cpu(frame, plan.cdef, error);
    case VideoFilterStage::kCcso:
      return apply_ccso_cpu(frame, plan.ccso, error);
    case VideoFilterStage::kLr:
      return apply_lr_cpu(frame, plan.lr, error);
    case VideoFilterStage::kGdf:
      return apply_gdf_cpu(frame, plan.gdf, error);
  }
  return fail(error, "video loop filters: unknown stage");
}

}  // namespace

const char* video_filter_stage_name(VideoFilterStage stage) noexcept {
  switch (stage) {
    case VideoFilterStage::kCdef:
      return "cdef";
    case VideoFilterStage::kCcso:
      return "ccso";
    case VideoFilterStage::kLr:
      return "lr";
    case VideoFilterStage::kGdf:
      return "gdf";
  }
  return "unknown";
}

CdefFilterOptions cdef_filter_options(const VideoCdefPlan& plan) {
  CdefFilterOptions options;
  options.damping = plan.damping;
  options.num_strengths = plan.num_strengths;
  options.on_skip_txfm = plan.on_skip_txfm;
  for (int i = 0; i < plan.num_strengths && i < 8; ++i) {
    options.strength[i].y_pri = plan.strength[i].y_pri;
    options.strength[i].y_sec = plan.strength[i].y_sec;
    options.strength[i].uv_pri = plan.strength[i].uv_pri;
    options.strength[i].uv_sec = plan.strength[i].uv_sec;
  }
  return options;
}

LrFilterOptions lr_filter_options(const VideoLrPlan& plan) {
  LrFilterOptions options;
  for (int plane = 0; plane < 3; ++plane) {
    const VideoLrPlane& src = plan.plane[plane];
    options.frame_type[plane] =
        src.enabled ? static_cast<rest::RestorationType>(src.restore_type)
                    : rest::kRestoreNone;
    options.unit_size[plane] = src.unit_size;
    options.unit_rows[plane] = src.unit_rows;
    options.unit_cols[plane] = src.unit_cols;
    options.unit_type[plane] = src.enabled ? src.unit_type.data() : nullptr;
    options.unit_coeffs[plane] =
        src.enabled ? src.unit_coeffs.data() : nullptr;
  }
  return options;
}

CcsoFilterOptions ccso_filter_options(const VideoCcsoPlan& plan) {
  CcsoFilterOptions options;
  options.ccso_luma_size_log2 = plan.ccso_luma_size_log2;
  for (int plane = 0; plane < 3; ++plane) {
    const VideoCcsoPlane& src = plan.plane[plane];
    CcsoFilterOptions::Plane& dst = options.plane[plane];
    dst.enabled = src.enabled;
    dst.bo_only = src.bo_only;
    dst.scale_idx = src.scale_idx;
    dst.quant_idx = src.quant_idx;
    dst.ext_filter = src.ext_filter;
    dst.edge_clf = src.edge_clf;
    dst.max_band_log2 = src.max_band_log2;
    if (!src.enabled) continue;
    rest::CcsoParams params;
    params.enabled = true;
    params.bo_only = src.bo_only;
    params.scale_idx = src.scale_idx;
    params.quant_idx = src.quant_idx;
    params.ext_filter = src.ext_filter;
    params.edge_clf = src.edge_clf;
    params.max_band_log2 = src.max_band_log2;
    if (rest::ccso_set_offsets(params, src.offset_idx.data(),
                               src.offset_idx.size())) {
      std::copy_n(params.offsets, rest::kCcsoOffsetEntries, dst.offsets);
    }
  }
  return options;
}

bool apply_video_filter_stage(VideoFilterStage stage,
                              VideoLoopFilterFrame& frame,
                              const VideoFilterPlan& plan,
                              std::string* error) {
  return apply_stage(stage, frame, plan, /*hooks=*/nullptr, error);
}

bool run_video_loop_filters(VideoLoopFilterFrame& frame,
                            const VideoLoopFilterOptions& options,
                            VideoFilterPlan* plan, VideoLoopFilterHooks* hooks,
                            std::string* error) {
  if (plan == nullptr) return fail(error, "video loop filters: null plan");
  *plan = VideoFilterPlan{};
  if (!options.any()) return true;
  if (!validate_frame(frame, /*need_source=*/true, error)) return false;

  // §7.19/§7.20 read the post-deblock (pre-CDEF) frame.  Keep a copy whenever a
  // later stage needs it, before CDEF overwrites the reconstruction.
  const bool need_curr = options.ccso || options.wiener || options.chroma_lr ||
                         options.pc_wiener || options.gdf;
  std::vector<std::uint8_t> curr_storage[3];
  if (need_curr) {
    for (int plane = 0; plane < 3; ++plane) {
      const int w = plane_width(frame, plane);
      const int h = plane_height(frame, plane);
      curr_storage[plane].assign(
          static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
      copy_plane(frame.plane[plane], frame.stride[plane],
                 curr_storage[plane].data(), w, w, h);
      frame.curr[plane] = curr_storage[plane].data();
      frame.curr_stride[plane] = w;
    }
  }

  if (options.cdef) {
    if (!prepare_cdef(frame, options, &plan->cdef, error)) return false;
    if (!apply_stage(VideoFilterStage::kCdef, frame, *plan, hooks, error)) {
      return false;
    }
  }
  if (options.ccso) {
    if (!prepare_ccso(frame, options, &plan->ccso, error)) return false;
    if (!apply_stage(VideoFilterStage::kCcso, frame, *plan, hooks, error)) {
      return false;
    }
  }
  // §7.20.5 GDF samples CdefFrame (post-CCSO, pre-LR); keep that snapshot
  // before loop restoration rewrites the reconstruction.
  std::vector<std::uint8_t> cdef_storage[3];
  if (options.gdf) {
    for (int plane = 0; plane < 3; ++plane) {
      const int w = plane_width(frame, plane);
      const int h = plane_height(frame, plane);
      cdef_storage[plane].assign(
          static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
      copy_plane(frame.plane[plane], frame.stride[plane],
                 cdef_storage[plane].data(), w, w, h);
      frame.cdef[plane] = cdef_storage[plane].data();
      frame.cdef_stride[plane] = w;
    }
  }
  if (options.wiener || options.chroma_lr || options.pc_wiener) {
    if (!prepare_lr(frame, options, &plan->lr, error)) return false;
    if (!apply_stage(VideoFilterStage::kLr, frame, *plan, hooks, error)) {
      return false;
    }
  }
  if (options.gdf) {
    if (!prepare_gdf(frame, &plan->gdf, error)) return false;
    if (!apply_stage(VideoFilterStage::kGdf, frame, *plan, hooks, error)) {
      return false;
    }
  }
  return true;
}

}  // namespace av2enc::video::post
