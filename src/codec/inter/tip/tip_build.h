// AV2 v1.0.0 TIP build — motion field to samples.
//
// Scope:
//   §7.9      motion field estimation (delegated to inter/temporal)
//   §7.10.2–.5 TIP temporal scale, hole fill, block average, sample-gap fill
//             (delegated to inter/temporal/setup_tip_motion_field)
//   §7.10.6   build TIP process: per-16x16/8x8 block MV derivation,
//             TipGlobalMv addition, 4x4 MV grid, per-plane prediction with
//             TipInterpFilter, COMPOUND_AVERAGE + Tip_Weighting_Factor blend,
//             and the §7.22 motion-field store of the produced MVs.
//
// The temporal component already implements the MV-field half and
// `tip_predict_block()`.  What was missing is the assembly: this file walks a
// TipPlan, predicts every plane, and writes the per-4x4 MV grid plus the
// stored reference field.  Without that, a TIP_FRAME_AS_OUTPUT frame cannot be
// reconstructed and a chained TIP frame has no motion field to project.
//
// Destination planes may be smaller than the TIP block grid: at the right and
// bottom edges of a frame the coded block extends past the visible extent
// (MiRows is a multiple of 2, not of 4, so a 360-line frame ends in a 10-MI
// row superblock).  Prediction of an individual sample is independent of the
// block rectangle, so the writer clips the rectangle to the destination.  This
// reproduces the decoder's samples for every visible pixel without requiring
// padding rows in the caller's buffer.

#ifndef AV2ENC_ENCODER_INTER_TIP_STREAM_TIP_BUILD_H_
#define AV2ENC_ENCODER_INTER_TIP_STREAM_TIP_BUILD_H_

#include <cstdint>

#include "codec/inter/temporal/motion_field.h"
#include "codec/inter/temporal/temporal_types.h"

namespace av2enc::inter::tip {

using temporal::MotionField;
using temporal::MotionFieldConfig;
using temporal::MotionFieldResult;
using temporal::MotionFieldScratch;
using temporal::PlaneView;
using temporal::TemporalStatus;
using temporal::TipConfig;
using temporal::TipPlan;

// Fills TipConfig::closest_*_order_hint and the past/future counts from the
// motion-field configuration and the estimation result.  `get_tip_offsets()`
// is meaningless without it.
void fill_tip_closest_refs(const MotionFieldConfig& config,
                           const MotionFieldResult& result, TipConfig* tip);

struct TipBuildInput {
  // Motion-field configuration.  `saved[]` must hold the decoded reference
  // fields; `ref_order_hint[]`, `num_total_refs`, `num_past_refs`,
  // `num_future_refs` and `enable_tip` come from the frame header.
  MotionFieldConfig config;
  TipConfig tip;  // tip.mode must be kAsOutput for frame construction
  // Caller-owned scratch, sized mi_cols>>1 by mi_rows>>1.
  MotionField* field = nullptr;
  MotionFieldScratch* scratch = nullptr;  // required with allow_tip_hole_fill
  // Caller-owned TIP plan: `num_blocks` = capacity, blocks/mv_grid/mv_grid1.
  TipPlan* plan = nullptr;
  // Decoded reference planes, [plane][list].  `width`/`height` are the
  // MI-aligned plane extents used by §7.13.3.17.
  PlaneView ref[3][2];
  // Destination planes.
  std::uint8_t* dst[3] = {nullptr, nullptr, nullptr};
  int dst_stride[3] = {0, 0, 0};
  int dst_width[3] = {0, 0, 0};
  int dst_height[3] = {0, 0, 0};
  int sub_x[3] = {0, 1, 1};
  int sub_y[3] = {0, 1, 1};
  // Two 1/16-precision block buffers, at least 2 * (16 * 16) samples.
  std::int16_t* pred_scratch = nullptr;
  // 8-bit block output buffer, at least 16 * 16 samples.  Prediction always
  // runs on the full TIP block (the decoder does too, writing past the visible
  // frame into its padded buffer), and only the visible part is copied to dst.
  // This keeps mc_pass_filter(w,h) and geometry identical at frame edges.
  std::uint8_t* block_scratch = nullptr;
};

struct TipBuildOutput {
  MotionFieldResult field_result{};
  int num_blocks = 0;
  int field_valid_after_estimation = 0;
  int field_valid_after_setup = 0;
  // Net increase in valid 8x8 entries across §7.10.3 (hole fill) and §7.10.4
  // (block average).  Positive means the TIP setup recovered holes.
  int hole_filled = 0;
  int field_invalid = 0;  // entries still invalid after §7.10.4
  int predicted_blocks = 0;
  std::int64_t predicted_samples[3] = {0, 0, 0};
};

// Complete §7.10 build: estimation -> TIP setup -> plan -> samples.
TemporalStatus build_tip_frame(const TipBuildInput& in, TipBuildOutput* out);

// §7.10.6 sample construction for an existing plan (no motion-field work).
TemporalStatus predict_tip_plan(const TipPlan& plan, const TipConfig& tip,
                                const MotionFieldConfig& config,
                                const PlaneView ref[3][2],
                                std::uint8_t* const dst[3],
                                const int dst_stride[3],
                                const int dst_width[3],
                                const int dst_height[3],
                                const int sub_x[3], const int sub_y[3],
                                std::int16_t* pred_scratch,
                                std::uint8_t* block_scratch,
                                TipBuildOutput* out);

// §7.22 motion-field motion vector storage for a TIP_FRAME_AS_OUTPUT frame.
struct TipStoreInput {
  int mi_cols = 0;
  int mi_rows = 0;
  int order_hint = 0;
  // Reference-list indices (§7.8 ClosestPast / ClosestFuture).
  int closest_past = 0;
  int closest_future = 1;
  int closest_past_order_hint = 0;
  int closest_future_order_hint = 0;
  // True when Tip_Weighting_Factor[tip_global_wtd_index] == 16 (only list 0 is
  // stored; §7.22 mirrors it into list 1).
  bool weight16 = false;
};

TemporalStatus store_tip_frame_motion_field(const TipPlan& plan,
                                            const TipStoreInput& in,
                                            temporal::RefMvField* out);

// §7.22 wrapper: view a stored field as the §7.9 input type.
inline temporal::SavedMvField saved_mv_view(
    const temporal::RefMvField& field) {
  temporal::SavedMvField out;
  out.w8 = field.w8;
  out.h8 = field.h8;
  out.mv = field.mv;
  out.ref = field.ref;
  return out;
}

}  // namespace av2enc::inter::tip

#endif  // AV2ENC_ENCODER_INTER_TIP_STREAM_TIP_BUILD_H_
