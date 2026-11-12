// Shared lambda / cost configuration for the unified M1 RDO loop.
//
// The loop uses one Lagrangian convention for every decision:
//
//     cost = SSE + lambda * rate_bits
//     lambda = sse_scale * quant_step^2   (pixel domain, 8-bit SSE)
//
// The quantiser step and the base sse_scale come from
// src/encoder/m1/rdo_decide/config.h so partition, intra, chroma, TX type and
// RDOQ all share one calibration.  This header adds only the pieces the unified
// loop needs on top of that layer:
//
//   * a per-CTU lambda scale driven by source activity,
//   * the rate split used in the per-node cost breakdown,
//   * a deterministic clamp for the adaptive scale.
//
// At M1's pinned base_q_idx = 90 the reconstruction step is 66 sample-domain
// units; 528 is the forward quantiser-table entry and must not be used here.
// See rdo_decide/config.h for the derivation and the measured sweep.

#pragma once

#include "codec/rate/decide/config.h"

#include <cstdint>
#include <string>

namespace av2enc::still::rdo_loop {

// M1 pins base_q_idx = 90; the sample-domain dequant step is 66.
inline constexpr double kM1Qp90QuantStep =
    rdo_decide::kM1Qp90QuantStep;

// Base pixel-domain sse_scale for the joint loop.  Calibrated by the
// experiment in experiment/README.md; it is not the rdo_decide layer's 0.5.
//
// Why the loop needs a much smaller scale: the joint node cost compares *real
// coded coefficient bits* for a whole partition node (hundreds to thousands of
// bits), while the older per-decision partition search used a variance/SSE bit
// proxy and could keep a larger scale for mode-only decisions.  One lambda now
// governs partition splits, luma/chroma modes, TX type and RDOQ, so it has to
// be calibrated on the joint node cost.  The measured optimum on the 64x64,
// 640x360 and 1080p QP-90 fixtures is 0.0005:
//
//     lambda = 0.0005 * 66^2 = 2.178   (SSE units per bit)
//
// The sweep is in experiment/README.md.  A larger scale keeps the same
// decisions on 64x64 but collapses 640x360/1080p toward low-rate blocks at the
// fixed QP; do not copy the mode-layer constant here.
inline constexpr double kBaseSseScale = 0.0005;

// CTU activity adaptation is disabled by default.  The measured sweep in the
// experiment README sets the shipped default: a non-zero strength is only
// justified when it wins on both byte/PSNR and CVVDP at matched rate.
inline constexpr double kDefaultCtuActivityStrength = 0.0;

struct LambdaConfig {
  double quant_step = kM1Qp90QuantStep;
  double sse_scale = kBaseSseScale;
  // lambda multiplier = clamp(1 + strength * (activity/mean - 1), min, max).
  // 0 keeps a single frame lambda.
  double ctu_activity_strength = kDefaultCtuActivityStrength;
  double ctu_scale_min = 0.75;
  double ctu_scale_max = 1.5;
  // false reproduces an SSE-only decision path; the same API can then emit the
  // legacy control arm.  The unified loop always runs rate-aware by default.
  bool rate_aware = true;
};

bool lambda_config_valid(const LambdaConfig& config) noexcept;
bool validate_lambda_config(const LambdaConfig& config,
                            std::string* error = nullptr);

// Base lambda = sse_scale * quant_step^2 (via the shared RDO framework).
double lambda_sse(const LambdaConfig& config) noexcept;

// Per-CTU lambda.  `activity` and `mean_activity` are the CTU and frame mean
// absolute luma deviation, scaled to integers (e.g. Q8).  Only their ratio is
// used, so any fixed-point scale is valid.  mean_activity == 0 disables the
// adaptation for that CTU.  The result is deterministic: one multiply, one
// divide, and the same clamp on every host.
struct CtuLambda {
  double lambda = 0.0;
  double scale = 1.0;
};

CtuLambda ctu_lambda(const LambdaConfig& config, std::uint32_t activity,
                     std::uint32_t mean_activity) noexcept;

// The loop config in the rdo_decide layer's shape, so the existing
// intra/chroma/TX/RDOQ selectors can consume one calibration.
rdo_decide::LambdaConfig to_decide_config(const LambdaConfig& config) noexcept;

// The M1 QP-90 profile.  `base_q_idx` is accepted for diagnostics only: M1 pins
// 90.  Any other value returns a zeroed config and sets `error`.
LambdaConfig calibrated_still_config(std::string* error = nullptr);

// cost(distortion, rate_bits) under the same convention.
double block_cost(double distortion, double rate_bits,
                  const LambdaConfig& config) noexcept;

}  // namespace av2enc::still::rdo_loop
