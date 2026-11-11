// AV2 v1.0.0 temporal inter tools — TIP prediction and weights.
//
// `build_tip_plan()` in motion_field.h produces the per-block structure; this
// header turns one block into samples.  It implements the COMPOUND_AVERAGE
// blend of §7.13.3.1 with `Tip_Weighting_Factor` and the EIGHTTAP_SHARP /
// configured TIP interpolation filter.

#ifndef AV2ENC_ENCODER_INTER_TEMPORAL_TIP_H_
#define AV2ENC_ENCODER_INTER_TEMPORAL_TIP_H_

#include <cstdint>

#include "encoder/inter/temporal/temporal_types.h"

namespace av2enc::inter::temporal {

// Build-TIP block size in 4x4 units: 2 for BLOCK_8X8, 4 for BLOCK_16X16.
inline int tip_block_size4(const TipConfig& tip) {
  return (tip.enable_tip_refinemv &&
          tip.interp_filter == InterpFilter::kEighttapSharp)
             ? 2
             : 4;
}

// `store_refined_mvs()` in §5.20.7.6.
inline bool tip_store_refined_mvs(const TipConfig& tip) {
  return tip.tip_global_wtd_index >= 0 &&
         kTipWeightingFactor[tip.tip_global_wtd_index] == kCwpEqual &&
         tip.enable_tip_refinemv && tip.num_future_refs > 0 &&
         tip.num_past_refs > 0;
}

// True when the TIP weight is CWP_EQUAL and both reference directions exist;
// this is the condition used by `LumaUseOptflowRefinement` in predict_inter().
// For build TIP (§7.10.6), additionally require
// `TipInterpFilter == EIGHTAP_SHARP`.
inline bool tip_luma_use_opfl(const TipConfig& tip) {
  return tip.mode != TipFrameMode::kDisabled &&
         tip.opfl_refine_type != RefineType::kNone &&
         tip.tip_global_wtd_index >= 0 &&
         kTipWeightingFactor[tip.tip_global_wtd_index] == kCwpEqual &&
         tip.num_past_refs > 0 && tip.num_future_refs > 0 &&
         tip.enable_tip_refinemv;
}

struct TipPredictInput {
  PlaneView ref[2];
  int num_refs = 1;  // 1 when Tip_Weighting_Factor == 16, else 2
  int x = 0;         // plane sample coordinates
  int y = 0;
  int w = 0;
  int h = 0;
  int sub_x = 0;  // 1 for 4:2:0 chroma
  int sub_y = 0;
  int frame_w = 0;  // luma frame dimensions (for §7.13.3.17 scaling)
  int frame_h = 0;
  Mv mv[2];         // 1/8 luma samples
  int cwp_weight = kCwpEqual;
  InterpFilter filter = InterpFilter::kEighttapSharp;
  bool use_ref_area = false;
  RefArea area[2]{};  // inclusive, per list; ignored when !use_ref_area
  std::uint8_t* dst = nullptr;
  int dst_stride = 0;
  std::int16_t* scratch0 = nullptr;  // w*h, 1/16-precision list 0
  std::int16_t* scratch1 = nullptr;  // w*h, 1/16-precision list 1
  int scratch_stride = 0;
};

// Predicts one TIP block.  For a weight of 16 (or one reference) this is a
// single-reference EIGHTTAP_SHARP prediction rounded to 8 bits; otherwise two
// 1/16-precision predictions are blended with
//   Clip1(Round2(cwp*Preds[0] + (16-cwp)*Preds[1], 8)).
TemporalStatus tip_predict_block(const TipPredictInput& in);

}  // namespace av2enc::inter::temporal

#endif  // AV2ENC_ENCODER_INTER_TEMPORAL_TIP_H_
