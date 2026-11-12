// AV2 v1.0.0 TIP output rules.

#include "encoder/inter/tip_stream/tip_output.h"

namespace av2enc::inter::tip {

bool tip_implicit_output_inferred(bool monotonic_output_order_flag) {
  // §5.18.2: implicit_output_frame is only coded when immediate_output_frame
  // and monotonic_output_order_flag are both 0; otherwise it is inferred 0.
  return !monotonic_output_order_flag;
}

TemporalStatus plan_tip_output(const TipOutputConfig& config,
                               TipOutputPlan* plan) {
  TipOutputPlan out;
  out.enable_tip_output = config.enable_tip && !config.disable_tip_output;
  out.output_now = config.immediate_output || config.implicit_output;

  if (config.is_tip_frame) {
    if (!out.enable_tip_output) {
      out.reason = "TIP OBU requires EnableTipOutput (disable_tip_output=1)";
      if (plan != nullptr) *plan = out;
      return TemporalStatus::kInvalidArgument;
    }
    if (config.mode != TipFrameMode::kAsOutput) {
      out.reason = "TIP OBU must resolve to TIP_FRAME_AS_OUTPUT";
      if (plan != nullptr) *plan = out;
      return TemporalStatus::kInvalidArgument;
    }
    if (!config.has_closest_past || !config.has_closest_future) {
      out.reason = "TIP frame requires ClosestPast and ClosestFuture";
      if (plan != nullptr) *plan = out;
      return TemporalStatus::kInvalidArgument;
    }
    out.tip_obu_allowed = true;
    out.non_output_tip = !out.output_now;
    if (out.non_output_tip && !config.refreshes_slot) {
      out.reason = "non-output TIP frame must refresh a reference slot";
      if (plan != nullptr) *plan = out;
      return TemporalStatus::kInvalidArgument;
    }
    out.reason = out.output_now ? "TIP frame output now"
                                : "non-output TIP frame (reference only)";
    if (plan != nullptr) *plan = out;
    return TemporalStatus::kOk;
  }

  if (config.mode == TipFrameMode::kAsOutput) {
    out.reason = "TIP_FRAME_AS_OUTPUT requires a TIP OBU";
    if (plan != nullptr) *plan = out;
    return TemporalStatus::kInvalidArgument;
  }
  out.tip_obu_allowed = false;
  out.non_output_tip = false;
  out.reason = out.output_now ? "regular frame output now"
                              : "regular non-output frame";
  if (plan != nullptr) *plan = out;
  return TemporalStatus::kOk;
}

}  // namespace av2enc::inter::tip
