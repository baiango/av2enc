// Residual (skip_flag) decision wrapper for the M2 P-frame RDO layer.
//
// The coefficient arithmetic already lives in `src/encoder/m2/residual/`:
// forward/inverse transform, quantisation, EOB, reconstruction, and the
// writer's own symbol-cost measurement.  This wrapper only fixes the lambda
// convention (M1 QP-90 step 66, SSE domain) and the M2 syntax constraints
// (DCT_DCT only, TX_MODE_LARGEST, no RDOQ, no inter DDT) so the decision
// cannot drift from the tile writer.

#pragma once

#include "encoder/m2/rdo/config.h"
#include "encoder/m2/residual/inter_residual.h"

#include <string>

namespace av2enc::m2::rdo {

// DecisionOptions with:
//   * lambda = lambda_sse(config)  (or the caller's explicit lambda if > 0)
//   * base_q_idx = 90
//   * DCT_DCT only, no RDOQ, no inter DDT, TX_MODE_LARGEST
// `allow_residual` is left true; callers turn it off for skip-only candidates.
residual::DecisionOptions make_residual_options(const RdoConfig& config);

// Runs the shared residual decision with this layer's lambda.  `force_residual`
// is the stress-gate behaviour (code a residual even when skip is cheaper);
// the reconstruction still comes from the decision and is decoder-exact.
bool decide_residual(const residual::BlockInput& input,
                     const RdoConfig& config, bool allow_residual,
                     bool force_residual, residual::BlockDecision& decision,
                     std::string* error = nullptr);

}  // namespace av2enc::m2::rdo
