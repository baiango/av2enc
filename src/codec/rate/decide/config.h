// Shared configuration and Lagrangian arithmetic for the M1 RDO decision layer.
//
// The M1 reference encoder pins base_q_idx = 90 and uses the AV2 quantiser from
// src/encoder/tx_variants/.  At QP 90 the *reconstruction* (dequant) step in
// the sample-domain distortion units is 66:
//
//     dequant = Round2(level * 528, 3) / dq_denom
//
// For an 8x8 transform dq_denom is 1 and the step is 66.  For larger transforms
// dq_denom grows exactly as fast as the forward coefficient scale, so the
// pixel-domain step stays 66 for every M1 transform size.  The constant 528 in
// txv::kQuantStep is the *forward quantiser table* entry, not a step; feeding it
// to lambda_sse_from_quant_step() inflates lambda by 8^2 = 64x.  This layer
// therefore uses the dequant step by default and documents the calibration.
//
// Lambda calibration
// ------------------
//     cost  = SSE + lambda * rate_bits
//     lambda = sse_scale * quant_step^2
//
// `sse_scale` is not normative.  The value in calibrated_still_config() is the one
// measured by the CPU experiment in experiment/README.md: the 64x64 and
// 640x360 QP-90 fixtures, scanning sse_scale and comparing bytes / PSNR /
// CVVDP against the SSE-only M1 decisions.  Use the sweep result instead of the
// constant when the target corpus changes.

#pragma once

#include "codec/rate/rdo/rdo.h"

#include <cstdint>
#include <string>

namespace av2enc::still::rdo_decide {

// Reconstruction step at M1's pinned base_q_idx = 90 (see header comment).
inline constexpr double kM1Qp90QuantStep = 66.0;

// Calibrated sse_scale for the M1 QP-90 fixtures.  The experiment README
// records the sweep: 0.5 corresponds to lambda = 2178 at QP 90 and is the
// best CVVDP point on both repository fixtures (64x64: +0.0142 JOD; 640x360:
// +0.0548 JOD versus the DC baseline).  1.0 trades a little CVVDP for fewer
// bytes; 0.25 buys more quality at a higher rate.  This constant is a starting
// point, not a target-corpus guarantee.
inline constexpr double kCalibratedSseScale = 0.5;

// Cost units: distortion is SSE in the metric the caller supplies; rate is in
// bits, never bytes.  A caller holding byte costs must multiply by 8.
struct LambdaConfig {
  double quant_step = kM1Qp90QuantStep;
  double sse_scale = kCalibratedSseScale;
  // false reproduces an SSE-only decision.  It is kept explicit so the same
  // API can generate the experiment baseline.
  bool rate_aware = true;
};

bool lambda_config_valid(const LambdaConfig& config) noexcept;
bool validate_lambda_config(const LambdaConfig& config,
                            std::string* error = nullptr);

// lambda = sse_scale * quant_step^2 via the shared RDO framework.
double lambda_sse(const LambdaConfig& config) noexcept;

// The M1 QP-90 profile.  `base_q_idx` is accepted for diagnostics only: the M1
// syntax writer pins 90.  Any other value returns a zeroed config and sets
// `error`; callers should pass the real dequant step explicitly when they
// extend M1 beyond QP 90.
LambdaConfig calibrated_still_config(std::string* error = nullptr);

// Convenience: cost(distortion, rate_bits) with this layer's convention.
double block_cost(double distortion, double rate_bits,
                  const LambdaConfig& config) noexcept;

}  // namespace av2enc::still::rdo_decide
