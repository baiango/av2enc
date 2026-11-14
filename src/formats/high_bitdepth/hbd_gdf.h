// §7.20.5 GDF planning/application for the 10-bit product still writer.
//
// The pixel filter and the header derivation are the recovered normative
// reference (`codec/filters/tables/gdf.*`) and the decision glue in
// `codec/filters/gdf/gdf_decision.*`.  This header only selects
// gdf_pic_qc_idx / gdf_pic_scale_idx and applies the header-only mode
// (gdf_per_block = 0, so no §5.20.10.3 tile symbol is coded).
#ifndef AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_GDF_H_
#define AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_GDF_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/filters/gdf/gdf_decision.h"
#include "codec/filters/gdf/gdf_syntax.h"
#include "formats/format.h"
#include "formats/high_bitdepth/hbd_filter_config.h"

namespace av2enc::hbd {

namespace gdf = av2enc::filters::gdf;

struct HbdGdfPlan {
  HbdGdfConfig config;
  gdf::FrameGdfContext context;
  gdf::FrameGdfParams params;
  std::int64_t sse_before = 0;
  std::int64_t sse_after = 0;
};

struct HbdGdfFrame16 {
  const std::uint16_t* curr = nullptr;  // CurrFrame[0] (post-deblock)
  const std::uint16_t* cdef = nullptr;  // CdefFrame[0] (post-CDEF/CCSO)
  const std::uint16_t* ref = nullptr;   // source luma
  int curr_stride = 0;
  int cdef_stride = 0;
  int ref_stride = 0;
  int width = 0;   // visible luma
  int height = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  int bit_depth = 10;
  int base_q_idx = 90;
  bool frame_is_intra = true;
};

inline gdf::GdfFilterInputs make_gdf_inputs(const HbdGdfFrame16& frame) {
  gdf::GdfFilterInputs inputs;
  inputs.frame_is_intra = frame.frame_is_intra;
  inputs.base_q_idx = frame.base_q_idx;
  inputs.bit_depth = frame.bit_depth;
  inputs.num_total_refs = 0;
  inputs.restricted_oh = -1;
  return inputs;
}

// Builds the §5.18.7.9 context for a single-picture header.
inline gdf::FrameGdfContext make_gdf_context(const HbdGdfFrame16& frame,
                                             std::string* error) {
  gdf::FrameGdfContext context;
  context.enable_gdf = true;
  context.gdf_unit_matches_sb_size = true;
  context.coded_lossless = false;
  context.single_picture_header_flag = true;
  context.sb_size = 64;
  // §7.20.1 runs over the coded (MI-aligned) picture: the last GDF unit's
  // stripe region is bounded by MiCols*4/MiRows*4, so the context size must be
  // the padded extent, not the visible width/height.
  context.width = frame.mi_cols * 4;
  context.height = frame.mi_rows * 4;
  context.mi_cols = frame.mi_cols;
  context.mi_rows = frame.mi_rows;
  context.disable_loopfilters_across_tiles = false;
  context.tile_cols = 1;
  context.tile_rows = 1;
  context.mi_col_starts.assign(1, 0);
  context.mi_row_starts.assign(1, 0);
  if (context.width <= 0 || context.height <= 0) {
    if (error != nullptr) *error = "gdf: invalid MI-aligned geometry";
  }
  return context;
}

// Chooses the best header-only GDF configuration by exact SSE against the
// source.  Returns an all-zero plan when no candidate helps.
inline bool plan_gdf(const HbdGdfFrame16& frame, HbdGdfPlan* plan,
                     std::string* error) {
  auto fail = [error](const std::string& text) {
    if (error != nullptr) *error = text;
    return false;
  };
  if (plan == nullptr) return fail("gdf plan: null output");
  *plan = HbdGdfPlan{};
  if (frame.mi_cols <= 0 || frame.mi_rows <= 0) {
    return fail("gdf plan: invalid geometry");
  }
  plan->context = make_gdf_context(frame, error);
  gdf::GdfSourceFrame source;
  source.curr = frame.curr;
  source.cdef = frame.cdef;
  source.stride = frame.curr_stride;
  const gdf::GdfFilterInputs inputs = make_gdf_inputs(frame);
  const std::size_t plane_size =
      static_cast<std::size_t>(frame.mi_cols * 4) *
      static_cast<std::size_t>(frame.mi_rows * 4);
  std::vector<std::uint16_t> out(plane_size);

  std::int64_t best_sse = 0;
  bool have_best = false;
  for (int qc = 0; qc < gdf::kGdfNumPicQcIdx; ++qc) {
    for (int scale = 0; scale < gdf::kGdfNumPicScaleIdx; ++scale) {
      gdf::FrameGdfParams params;
      params.frame_enable = true;
      params.per_block = false;
      params.pic_qc_idx = qc;
      params.pic_scale_idx = scale;
      params.pix_scale = 1 + scale;
      // GDF adds its delta to LrFrame, so the evaluation buffer must start as
      // the CdefFrame copy (the decoder's LrFrame := CdefFrame step).
      for (int y = 0; y < frame.mi_rows * 4; ++y) {
        for (int x = 0; x < frame.mi_cols * 4; ++x) {
          out[static_cast<std::size_t>(y) * frame.curr_stride + x] =
              frame.cdef[static_cast<std::size_t>(y) * frame.cdef_stride + x];
        }
      }
      if (!gdf::ApplyGdfToFrame(source, inputs, plan->context, params, {}, out.data(),
                                frame.curr_stride, error)) {
        return false;
      }
      std::int64_t sse = 0;
      for (int y = 0; y < frame.height; ++y) {
        for (int x = 0; x < frame.width; ++x) {
          const int ref = static_cast<int>(
              frame.ref[static_cast<std::size_t>(y) * frame.ref_stride + x]);
          const int value = static_cast<int>(
              out[static_cast<std::size_t>(y) * frame.curr_stride + x]);
          const int diff = value - ref;
          sse += static_cast<std::int64_t>(diff) * diff;
        }
      }
      if (!have_best || sse < best_sse) {
        best_sse = sse;
        have_best = true;
        plan->config.enabled = true;
        plan->config.pic_qc_idx = qc;
        plan->config.pic_scale_idx = scale;
        plan->config.base_q_idx = frame.base_q_idx;
        plan->config.frame_is_intra = frame.frame_is_intra;
        plan->params = params;
      }
    }
  }
  if (!have_best) return fail("gdf plan: no candidate");
  // Compare with the unfiltered CdefFrame.
  std::int64_t before = 0;
  for (int y = 0; y < frame.height; ++y) {
    for (int x = 0; x < frame.width; ++x) {
      const int ref = static_cast<int>(
          frame.ref[static_cast<std::size_t>(y) * frame.ref_stride + x]);
      const int value = static_cast<int>(
          frame.cdef[static_cast<std::size_t>(y) * frame.cdef_stride + x]);
      const int diff = value - ref;
      before += static_cast<std::int64_t>(diff) * diff;
    }
  }
  plan->sse_before = before;
  plan->sse_after = best_sse;
  if (best_sse >= before) {
    plan->config.enabled = false;
  }
  if (error != nullptr) error->clear();
  return true;
}

// Applies the planned GDF to `lr_luma` in place, reading CdefFrame through the
// §7.20.2 source.  `lr_luma` is the post-Wiener LrFrame luma.
inline bool apply_gdf16(const HbdGdfPlan& plan, const HbdGdfFrame16& frame,
                        std::uint16_t* lr_luma, int lr_stride,
                        std::string* error) {
  if (!plan.config.enabled) return true;
  gdf::GdfSourceFrame source;
  source.curr = frame.curr;
  source.cdef = frame.cdef;
  source.stride = frame.curr_stride;
  const gdf::GdfFilterInputs inputs = make_gdf_inputs(frame);
  gdf::FrameGdfParams params = plan.params;
  params.frame_enable = true;
  params.per_block = false;
  params.pix_scale = 1 + params.pic_scale_idx;
  if (!gdf::ApplyGdfToFrame(source, inputs, plan.context, params, {}, lr_luma,
                            lr_stride, error)) {
    return false;
  }
  return true;
}

}  // namespace av2enc::hbd

#endif  // AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_GDF_H_
