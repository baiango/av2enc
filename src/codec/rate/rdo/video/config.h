// M2 P-frame rate-aware decision layer: shared configuration.
//
// This file owns no encoder state and writes no bits.  It fixes the cost
// convention for every M2 inter decision:
//
//     cost = distortion + lambda * rate_bits
//
// lambda comes from the M1 QP-90 calibration in
// `src/encoder/m1/rdo_decide/config.h`, not from a second copy of the
// calibration math:
//
//     lambda_sse  = sse_scale * 66^2       (66 = QP-90 reconstruction step)
//     lambda_sad  = sqrt(lambda_sse)        (L1-like distortion)
//     lambda_satd = sqrt(lambda_sse)
//
// The square-root relation is the one documented by `src/encoder/rdo/rdo.h`
// for SAD/SATD costs.  The M2 experiment runs the default base_q_idx = 90, so
// the calibrated config is valid there.  The current m2/residual decision is
// QP-90-only; an integration at another QP must set quant_step to
// qlookup(base_q_idx)/8 and extend the residual path first.
//
// Cost units: distortion is SSE, SAD or SATD depending on `metric`; rate is
// always bits, never bytes; lambda is in the same distortion units as the
// metric.  Mixing a SAD distortion with an SSE lambda is a bug, so callers
// must use `lambda_for_metric()` / `selector_config()` and never form lambda
// themselves.

#pragma once

#include "codec/rate/decide/config.h"
#include "codec/rate/rdo/rdo.h"

#include <string>

namespace av2enc::video::rdo {

// One config per frame.  `lambda` is the M1 QP-90 calibration (step 66,
// sse_scale 0.5); `metric` selects the distortion domain.
struct RdoConfig {
  av2enc::still::rdo_decide::LambdaConfig lambda;
  av2enc::rdo::DistortionMetric metric =
      av2enc::rdo::DistortionMetric::kSsd;
  av2enc::rdo::TieBreak tie_break =
      av2enc::rdo::TieBreak::kDistortionThenRate;
  // false disables the candidate-selection rate term (selector lambda = 0)
  // and the partition leaf-rate term, for the experiment's distortion-only
  // control.  The residual decision keeps its own SSE lambda; use
  // make_residual_options()/decide_residual() separately if a distortion-only
  // residual control is wanted.
  bool rate_aware = true;
};

bool config_valid(const RdoConfig& config) noexcept;
bool validate_config(const RdoConfig& config, std::string* error = nullptr);

// The M2 profile: M1's calibrated QP-90 config plus SSE metric.  `error` is
// set only when the M1 calibration is invalid (it is not for QP 90).
RdoConfig calibrated_video_config(std::string* error = nullptr);

// lambda_sse = av2enc::still::rdo_decide::lambda_sse(lambda config).
double lambda_sse(const RdoConfig& config) noexcept;

// lambda in the distortion units of `config.metric`: lambda_sse for SSE and
// sqrt(lambda_sse) for SAD/SATD (see the file comment).
double lambda_for_metric(const RdoConfig& config) noexcept;

// Config for `av2enc::rdo::CandidateSelector`.  lambda is 0 when
// `rate_aware` is false, which makes the shared selector distortion-only.
av2enc::rdo::RdoConfig selector_config(const RdoConfig& config) noexcept;

}  // namespace av2enc::video::rdo
