#pragma once

// Rate-aware intra mode decision for the M1 full-intra search.
//
// The M1 encoder currently chooses the luma mode by minimum SSE. That ignores
// the bits spent on the mode/angle syntax and on the residual coefficients, so
// the extra directional/smooth candidates can lose to DC at fixed QP even when
// they predict better. This header adds the missing RD decision:
//
//     cost = SSE + lambda * (syntax_rate_bits + coefficient_rate_bits)
//
// The selector itself is a pure, deterministic, O(1)-memory reduction over a
// candidate list. It is suitable for the CPU reference path and for a GPU
// reduction kernel: candidates in, chosen mode out.
//
// Rate inputs are caller-supplied. The default helpers are documented
// placeholders; the entropy owner should replace the coefficient-rate callback
// with exact symbol costs before enabling rate-aware selection by default.
//
// Units:
//   - sse: sample-domain squared error.
//   - rate bits: bits, not bytes.
//   - lambda: SSE units per bit. Build it from the real quantiser step with
//     make_intra_rdo_config(); the scale is a documented placeholder.

#include "rdo.h"

#include <cstdint>
#include <span>
#include <string>

namespace av2enc::rdo {

inline constexpr int kIntraModeCount = 13;  // AV2 YMode 0..12
inline constexpr int kIntraMaxAngleDelta = 3;

// Default placeholder for lambda_sse = sse_scale * quant_step^2. The M1
// reference keeps the quantiser step fixed at txv::kQuantStep = 528. Calibrate
// this scale before using rate-aware selection as a default encoder policy.
inline constexpr double kIntraDefaultSseScale = 0.125;

// One intra candidate as it crosses the decision API boundary.
//   - mode: AV2 YMode 0..12.
//   - angle_delta: AngleDeltaY -3..3 (0 for non-directional modes).
//   - sse: distortion against the source.
//   - syntax_rate_bits: mode/angle signalling.
//   - coefficient_rate_bits: rate delta for the residual coefficients.
struct IntraCandidateCost {
  std::int32_t mode = 0;
  std::int32_t angle_delta = 0;
  double sse = 0.0;
  double syntax_rate_bits = 0.0;
  double coefficient_rate_bits = 0.0;
};

inline double intra_candidate_rate_bits(const IntraCandidateCost& candidate) {
  return candidate.syntax_rate_bits + candidate.coefficient_rate_bits;
}

struct IntraRdoConfig {
  double lambda = 0.0;
  // false reproduces the current SSE-only M1 decision; true adds the rate term.
  bool rate_aware = true;
  // On equal cost and SSE, prefer the lower-rate candidate. The final tie-break
  // is the fixed AV2 candidate order (13 base modes, then angle deltas
  // -3,-2,-1,1,2,3 per directional mode), independent of caller enumeration.
  bool prefer_lower_rate_on_tie = true;
};

struct IntraModeChoice {
  bool found = false;
  std::int32_t mode = 0;
  std::int32_t angle_delta = 0;
  double sse = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  std::int32_t input_index = 0;
  std::int32_t candidate_count = 0;  // valid candidates considered
  std::int32_t rejected_count = 0;   // invalid candidates ignored
};

bool intra_rdo_config_valid(const IntraRdoConfig& config) noexcept;
bool validate_intra_rdo_config(const IntraRdoConfig& config,
                               std::string* error = nullptr);

// Streaming selector. Invalid candidates (bad mode/angle, negative/non-finite
// SSE or rate) are rejected and counted. Memory use is constant.
class IntraModeSelector {
 public:
  explicit IntraModeSelector(const IntraRdoConfig& config) noexcept;

  bool add(const IntraCandidateCost& candidate, std::string* error = nullptr);

  const IntraModeChoice& choice() const noexcept { return choice_; }
  void reset() noexcept;

 private:
  IntraRdoConfig config_;
  IntraModeChoice choice_;
};

IntraModeChoice select_intra_mode(std::span<const IntraCandidateCost> candidates,
                                  const IntraRdoConfig& config,
                                  std::string* error = nullptr);

// Pure comparator for GPU reductions. Equal cost/SSE/rate/mode/angle compare
// equal (false), so a stable reduction needs no insertion index. The CPU
// selector uses input order only as the final tie-break.
bool intra_candidate_better(const IntraCandidateCost& a,
                            const IntraCandidateCost& b,
                            const IntraRdoConfig& config) noexcept;

// lambda = lambda_sse_from_quant_step(quant_step, sse_scale), using the shared
// RDO framework. The scale default is a calibration placeholder.
IntraRdoConfig make_intra_rdo_config(
    double quant_step,
    double sse_scale = kIntraDefaultSseScale,
    bool rate_aware = true);

// Same, with quant_step_from_qp(base_q_idx) as the step model. M1 itself keeps
// the step fixed at 528, so the integration should normally use the explicit
// quant_step overload with txv::kQuantStep.
IntraRdoConfig make_intra_rdo_config_from_qp(
    int base_q_idx,
    double sse_scale = kIntraDefaultSseScale,
    bool rate_aware = true);

// Placeholder mode/angle signalling model. Exact costs depend on the
// neighbour-dependent AV2 Y-mode set and must come from the entropy owner.
double default_intra_syntax_rate_bits(std::int32_t mode,
                                      std::int32_t angle_delta) noexcept;

// Placeholder coefficient-rate model from residual SSE. The experiment harness
// uses actual quantised levels via intra_coefficient_rate_bits_from_levels().
double default_intra_coefficient_rate_bits(double sse,
                                           std::int32_t block_w,
                                           std::int32_t block_h,
                                           double quant_step) noexcept;

// Entropy-shaped placeholder for already-quantised levels in scan order. It
// reuses the RDOQ bring-up rate model; it is not an AV2 symbol-cost estimate.
double intra_coefficient_rate_bits_from_levels(
    std::span<const std::int32_t> levels_scan,
    std::int32_t eob,
    std::int32_t num_coefficients) noexcept;

// Pluggable rate hook. Return false (or a non-finite/negative rate) to reject a
// candidate. `context` is caller-owned and may carry the entropy state.
struct IntraRateRequest {
  std::int32_t mode = 0;
  std::int32_t angle_delta = 0;
  std::int32_t block_w = 0;
  std::int32_t block_h = 0;
  std::int32_t base_q_idx = 90;
  double sse = 0.0;
  double quant_step = 528.0;  // M1 keeps txv::kQuantStep
};

struct IntraRateEstimate {
  double syntax_rate_bits = 0.0;
  double coefficient_rate_bits = 0.0;
};

using IntraRateEstimator = bool (*)(void* context,
                                    const IntraRateRequest& request,
                                    IntraRateEstimate& out);

// Calls the estimator, or fills the documented defaults when estimator is
// null. Rejects non-finite or negative rates.
bool estimate_intra_rates(IntraRateEstimator estimator,
                          void* context,
                          const IntraRateRequest& request,
                          IntraRateEstimate& out,
                          std::string* error = nullptr);

}  // namespace av2enc::rdo
