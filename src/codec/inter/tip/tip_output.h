// AV2 v1.0.0 TIP output rules (§5.4.6 EnableTipOutput, §5.18.2, §7.21.1).
//
// A TIP frame is not tile-coded: frame_header(1) runs decode_frame_wrapup()
// immediately (§5.18.1).  Whether it produces a displayed picture is decided
// by the same output flags as any other coded frame:
//
//   * immediate_output_frame == 1 -> the frame is output at this temporal unit;
//   * else implicit_output_frame == 1 -> it becomes eligible for §7.21.4
//     implicit output;
//   * else it is a coded non-output frame unit: it may refresh reference
//     slots but is never displayed directly (it can be shown later through an
//     SEF or an implicit-output pass).
//
// Two encoder-side rules are easy to get wrong and are checked here:
//   1. `EnableTipOutput = enable_tip && !disable_tip_output` gates whether a
//      TIP-frame OBU may appear at all.  With disable_tip_output == 1 a
//      TIP OBU is non-conforming (§6.20: an OBU_REGULAR_TIP must resolve to
//      TIP_FRAME_AS_OUTPUT, which the TIP block only infers when
//      EnableTipOutput is 1).
//   2. A non-output TIP frame must still refresh at least one reference slot;
//      a frame that neither refreshes nor is output has no effect and the
//      spec's coded-frame-unit grammar has no use for it.

#ifndef AV2ENC_ENCODER_INTER_TIP_STREAM_TIP_OUTPUT_H_
#define AV2ENC_ENCODER_INTER_TIP_STREAM_TIP_OUTPUT_H_

#include <cstdint>

#include "codec/inter/tip/tip_signalling.h"
#include "codec/inter/temporal/temporal_types.h"

namespace av2enc::inter::tip {

struct TipOutputConfig {
  bool enable_tip = false;
  bool disable_tip_output = false;
  bool is_tip_frame = false;          // OBU_LEADING_TIP / OBU_REGULAR_TIP
  TipFrameMode mode = TipFrameMode::kDisabled;
  bool immediate_output = false;
  bool implicit_output = false;
  // refresh_frame_flags != 0 for this frame.
  bool refreshes_slot = false;
  // ClosestPast/ClosestFuture exist for this frame (§7.8).
  bool has_closest_past = false;
  bool has_closest_future = false;
};

struct TipOutputPlan {
  bool enable_tip_output = false;  // §5.4.6 EnableTipOutput
  bool tip_obu_allowed = false;    // may this frame be a TIP-frame OBU?
  bool output_now = false;         // displayed at this temporal unit
  bool non_output_tip = false;     // legal non-output TIP (reference only)
  const char* reason = "";
};

// Returns kOk for a conforming configuration; kInvalidArgument with a reason
// string otherwise.  `plan` is filled even on failure when non-null.
TemporalStatus plan_tip_output(const TipOutputConfig& config,
                               TipOutputPlan* plan);

// Convenience: resolve output flags from the sequence's monotonic flag.
//   monotonic_output_order_flag == 1 -> implicit_output_frame is inferred 0.
bool tip_implicit_output_inferred(bool monotonic_output_order_flag);

}  // namespace av2enc::inter::tip

#endif  // AV2ENC_ENCODER_INTER_TIP_STREAM_TIP_OUTPUT_H_
