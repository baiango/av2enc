// AV2 v1.0.0 temporal inter tools — optical-flow refinement (OPFL).
//
// Implements §7.13.3.8–.12: `get_optflow_based_mv()`, `optflow_diff()`,
// `compute_grad()`, and `compute_opfl_mv()`.
//
// Inputs are the two predictions formed by `make_inter_predictions()`
// (§7.13.3.13) with single-reference rounding: integer 8-bit samples.  The
// starting MVs and the order-hint distances of the two references follow.
// Outputs are the refined per-4x4 MVs (`OpflMvs`, 1/16 precision) and the
// deltas (`MvDeltas`, 1/8 precision), exactly the arrays a caller needs for the
// luma pass and for the chroma `predict_subblock()` path.

#ifndef AV2ENC_ENCODER_INTER_TEMPORAL_OPFL_H_
#define AV2ENC_ENCODER_INTER_TEMPORAL_OPFL_H_

#include <cstdint>

#include "encoder/inter/temporal/temporal_types.h"

namespace av2enc::inter::temporal {

// §5.20.7.6 opfl_allowed_for_refs().
bool opfl_allowed_for_refs(int d0, int d1, bool frame_is_switch,
                           bool ref0_scaled, bool ref1_scaled);

struct OpflJob {
  int w = 0;  // region width in luma samples, multiple of 4
  int h = 0;
  const std::uint8_t* p0 = nullptr;  // integer 8-bit list 0
  int p0_stride = 0;
  const std::uint8_t* p1 = nullptr;
  int p1_stride = 0;
  Mv mvs[2];  // 1/8 luma samples
  int order_hint = 0;
  int ref0_order_hint = 0;
  int ref1_order_hint = 0;
  int n = 8;      // optical-flow block size: 8 or 4
  int opt_x = 0;  // subblock offset within the inter block (samples)
  int opt_y = 0;
};

// Computes OpflMvs and MvDeltas.  `out.opfl_mvs`/`out.mv_deltas` must be
// allocated for the whole inter block (indexed by 4x4 with the given strides).
TemporalStatus opfl_compute(const OpflJob& job, const OpflOutput& out);

}  // namespace av2enc::inter::temporal

#endif  // AV2ENC_ENCODER_INTER_TEMPORAL_OPFL_H_
