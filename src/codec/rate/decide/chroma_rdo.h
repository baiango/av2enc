// Chroma DC / CfL decision on the U+V score.
//
// The M1 default chroma path compares DC and CfL with SSE only, and the
// historical score uses U only for DC.  That is why a U-only win can pick a
// chroma mode that is worse for V.  This selector:
//   * scores both planes (u_sse + v_sse),
//   * charges the chroma mode/is_cfl syntax and the residual coefficient rate,
//   * keeps the documented deterministic tie-break.
//
// Units are the same as the rest of the layer: SSE in distortion units, rate in
// bits, lambda from config.h.

#pragma once

#include "config.h"

#include <cstdint>
#include <span>
#include <string>

namespace av2enc::still::rdo_decide {

struct ChromaRateRequest {
  bool is_cfl = false;
  std::int32_t mode = 0;        // AV2 UVMode 0..12
  std::int32_t angle_delta = 0; // derived AngleDeltaUV, -3..3
  std::int32_t block_w = 0;     // chroma residual width
  std::int32_t block_h = 0;
  double sse_u = 0.0;
  double sse_v = 0.0;
  double quant_step = kM1Qp90QuantStep;
};

struct ChromaRateEstimate {
  double syntax_rate_bits = 0.0;
  double coefficient_rate_bits = 0.0;
};

using ChromaRateEstimator = bool (*)(void* context,
                                     const ChromaRateRequest& request,
                                     ChromaRateEstimate& out);

bool estimate_chroma_rates(ChromaRateEstimator estimator, void* context,
                           const ChromaRateRequest& request,
                           ChromaRateEstimate& out,
                           std::string* error = nullptr);

struct ChromaCandidateCost {
  bool is_cfl = false;
  std::int32_t mode = 0;
  std::int32_t angle_delta = 0;
  // Geometry shared by every candidate for one chroma block.  Used by the
  // default rate model and passed to a pluggable estimator.
  std::int32_t block_w = 0;
  std::int32_t block_h = 0;
  double sse_u = 0.0;
  double sse_v = 0.0;
  double syntax_rate_bits = 0.0;
  double coefficient_rate_bits = 0.0;
};

inline double chroma_candidate_rate_bits(
    const ChromaCandidateCost& candidate) noexcept {
  return candidate.syntax_rate_bits + candidate.coefficient_rate_bits;
}

struct ChromaRdoConfig {
  double lambda = 0.0;
  bool rate_aware = true;
  bool prefer_lower_rate_on_tie = true;
};

struct ChromaChoice {
  bool found = false;
  bool is_cfl = false;
  std::int32_t mode = 0;
  std::int32_t angle_delta = 0;
  double sse_u = 0.0;
  double sse_v = 0.0;
  double sse_total = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  std::int32_t input_index = 0;
  std::int32_t candidate_count = 0;
  std::int32_t rejected_count = 0;
};

// Streaming, allocation-free selector.  Invalid candidates are rejected and
// counted, exactly like the luma selector.
class ChromaModeSelector {
 public:
  explicit ChromaModeSelector(const ChromaRdoConfig& config) noexcept;

  bool add(const ChromaCandidateCost& candidate,
           std::string* error = nullptr);

  const ChromaChoice& choice() const noexcept { return choice_; }
  void reset() noexcept;

 private:
  ChromaRdoConfig config_;
  ChromaChoice choice_;
};

ChromaChoice select_chroma_mode(std::span<const ChromaCandidateCost> candidates,
                                const ChromaRdoConfig& config,
                                std::string* error = nullptr);

// Convenience for the M1 experiment/integration: fill rates from a pluggable
// estimator into a caller-owned scratch array, then select.  Both spans are
// caller-owned; no allocation.
ChromaChoice decide_chroma(std::span<const ChromaCandidateCost> candidates,
                           std::span<ChromaCandidateCost> scratch,
                           const LambdaConfig& config,
                           ChromaRateEstimator estimator = nullptr,
                           void* estimator_context = nullptr,
                           std::string* error = nullptr);

// Placeholder default syntax cost (CfL flag / UV mode).  The entropy owner
// should replace it with the exact §5.20.5.7 / §5.20.5.8 costs.
double default_chroma_syntax_bits(bool is_cfl, std::int32_t mode,
                                  std::int32_t angle_delta) noexcept;

}  // namespace av2enc::still::rdo_decide
