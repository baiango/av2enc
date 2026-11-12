#include "chroma_rdo.h"

#include "codec/rate/rdo/intra_decision.h"

#include <cmath>
#include <limits>

namespace av2enc::still::rdo_decide {
namespace {

constexpr int kModeCount = 13;
constexpr int kMaxAngleDelta = 3;

bool is_directional(std::int32_t mode) noexcept {
  return mode >= 1 && mode <= 8;
}

bool candidate_valid(const ChromaCandidateCost& candidate) noexcept {
  if (candidate.mode < 0 || candidate.mode >= kModeCount) {
    return false;
  }
  if (candidate.angle_delta < -kMaxAngleDelta ||
      candidate.angle_delta > kMaxAngleDelta) {
    return false;
  }
  if (candidate.is_cfl) {
    // CfL has no separate UV angle syntax; the decoder derives it from YMode.
    if (candidate.angle_delta != 0) {
      return false;
    }
  } else if (candidate.angle_delta != 0 &&
             !is_directional(candidate.mode)) {
    return false;
  }
  return std::isfinite(candidate.sse_u) && candidate.sse_u >= 0.0 &&
         std::isfinite(candidate.sse_v) && candidate.sse_v >= 0.0 &&
         std::isfinite(candidate.syntax_rate_bits) &&
         candidate.syntax_rate_bits >= 0.0 &&
         std::isfinite(candidate.coefficient_rate_bits) &&
         candidate.coefficient_rate_bits >= 0.0;
}

struct Scored {
  ChromaCandidateCost candidate{};
  double sse_total = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  std::int32_t input_index = 0;
};

bool better(const Scored& a, const Scored& b,
            const ChromaRdoConfig& config) noexcept {
  if (a.cost != b.cost) {
    return a.cost < b.cost;
  }
  if (a.sse_total != b.sse_total) {
    return a.sse_total < b.sse_total;
  }
  if (config.rate_aware && config.prefer_lower_rate_on_tie &&
      a.rate_bits != b.rate_bits) {
    return a.rate_bits < b.rate_bits;
  }
  // Non-CfL (the DC path) before CfL at equal cost/SSE/rate.
  if (a.candidate.is_cfl != b.candidate.is_cfl) {
    return !a.candidate.is_cfl;
  }
  if (a.candidate.mode != b.candidate.mode) {
    return a.candidate.mode < b.candidate.mode;
  }
  if (a.candidate.angle_delta != b.candidate.angle_delta) {
    return a.candidate.angle_delta < b.candidate.angle_delta;
  }
  return a.input_index < b.input_index;
}

}  // namespace

bool estimate_chroma_rates(ChromaRateEstimator estimator, void* context,
                           const ChromaRateRequest& request,
                           ChromaRateEstimate& out, std::string* error) {
  out = ChromaRateEstimate{};
  if (estimator != nullptr) {
    if (!estimator(context, request, out)) {
      if (error != nullptr) {
        *error = "ChromaRateEstimator rejected the candidate";
      }
      return false;
    }
  } else {
    out.syntax_rate_bits = default_chroma_syntax_bits(
        request.is_cfl, request.mode, request.angle_delta);
    out.coefficient_rate_bits =
        av2enc::rdo::default_intra_coefficient_rate_bits(
            request.sse_u + request.sse_v, request.block_w, request.block_h,
            request.quant_step);
  }
  if (!std::isfinite(out.syntax_rate_bits) || out.syntax_rate_bits < 0.0 ||
      !std::isfinite(out.coefficient_rate_bits) ||
      out.coefficient_rate_bits < 0.0) {
    if (error != nullptr) {
      *error = "chroma rate estimate must be finite and >= 0";
    }
    return false;
  }
  return true;
}

ChromaModeSelector::ChromaModeSelector(const ChromaRdoConfig& config) noexcept
    : config_(config) {}

bool ChromaModeSelector::add(const ChromaCandidateCost& candidate,
                             std::string* error) {
  if (!std::isfinite(config_.lambda) || config_.lambda < 0.0) {
    ++choice_.rejected_count;
    if (error != nullptr) *error = "invalid ChromaRdoConfig.lambda";
    return false;
  }
  if (!candidate_valid(candidate)) {
    ++choice_.rejected_count;
    if (error != nullptr) {
      *error = "invalid chroma candidate (mode/angle/distortion/rate)";
    }
    return false;
  }

  Scored score;
  score.candidate = candidate;
  score.sse_total = candidate.sse_u + candidate.sse_v;
  score.rate_bits = chroma_candidate_rate_bits(candidate);
  score.cost = config_.rate_aware
                   ? score.sse_total + config_.lambda * score.rate_bits
                   : score.sse_total;
  score.input_index = choice_.candidate_count;
  if (!std::isfinite(score.cost)) {
    ++choice_.rejected_count;
    if (error != nullptr) *error = "chroma candidate cost is not finite";
    return false;
  }
  ++choice_.candidate_count;

  if (!choice_.found) {
    choice_.found = true;
    choice_.is_cfl = candidate.is_cfl;
    choice_.mode = candidate.mode;
    choice_.angle_delta = candidate.angle_delta;
    choice_.sse_u = candidate.sse_u;
    choice_.sse_v = candidate.sse_v;
    choice_.sse_total = score.sse_total;
    choice_.rate_bits = score.rate_bits;
    choice_.cost = score.cost;
    choice_.input_index = score.input_index;
    return true;
  }

  Scored current;
  current.candidate.is_cfl = choice_.is_cfl;
  current.candidate.mode = choice_.mode;
  current.candidate.angle_delta = choice_.angle_delta;
  current.candidate.sse_u = choice_.sse_u;
  current.candidate.sse_v = choice_.sse_v;
  current.sse_total = choice_.sse_total;
  current.rate_bits = choice_.rate_bits;
  current.cost = choice_.cost;
  current.input_index = choice_.input_index;

  if (better(score, current, config_)) {
    choice_.is_cfl = candidate.is_cfl;
    choice_.mode = candidate.mode;
    choice_.angle_delta = candidate.angle_delta;
    choice_.sse_u = candidate.sse_u;
    choice_.sse_v = candidate.sse_v;
    choice_.sse_total = score.sse_total;
    choice_.rate_bits = score.rate_bits;
    choice_.cost = score.cost;
    choice_.input_index = score.input_index;
  }
  return true;
}

void ChromaModeSelector::reset() noexcept { choice_ = ChromaChoice{}; }

ChromaChoice select_chroma_mode(
    std::span<const ChromaCandidateCost> candidates,
    const ChromaRdoConfig& config, std::string* error) {
  ChromaModeSelector selector(config);
  for (const ChromaCandidateCost& candidate : candidates) {
    std::string local_error;
    if (!selector.add(candidate, &local_error)) {
      if (error != nullptr && error->empty()) {
        *error = local_error;
      }
    }
  }
  return selector.choice();
}

ChromaChoice decide_chroma(std::span<const ChromaCandidateCost> candidates,
                           std::span<ChromaCandidateCost> scratch,
                           const LambdaConfig& config,
                           ChromaRateEstimator estimator,
                           void* estimator_context, std::string* error) {
  if (error != nullptr) error->clear();
  if (!validate_lambda_config(config, error)) {
    return ChromaChoice{};
  }
  ChromaRdoConfig selector_config;
  selector_config.lambda = lambda_sse(config);
  selector_config.rate_aware = config.rate_aware;

  if (estimator == nullptr) {
    return select_chroma_mode(candidates, selector_config, error);
  }
  if (scratch.size() < candidates.size()) {
    if (error != nullptr) {
      *error = "chroma rate-estimator scratch span is too small";
    }
    return ChromaChoice{};
  }
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const ChromaCandidateCost& source = candidates[i];
    ChromaCandidateCost& target = scratch[i];
    target = source;
    ChromaRateRequest request;
    request.is_cfl = source.is_cfl;
    request.mode = source.mode;
    request.angle_delta = source.angle_delta;
    request.block_w = source.block_w;
    request.block_h = source.block_h;
    request.sse_u = source.sse_u;
    request.sse_v = source.sse_v;
    request.quant_step = config.quant_step;
    ChromaRateEstimate estimate;
    std::string local_error;
    if (!estimate_chroma_rates(estimator, estimator_context, request, estimate,
                               &local_error)) {
      if (error != nullptr && error->empty()) *error = local_error;
      continue;
    }
    target.syntax_rate_bits = estimate.syntax_rate_bits;
    target.coefficient_rate_bits = estimate.coefficient_rate_bits;
  }
  return select_chroma_mode(scratch.first(candidates.size()), selector_config,
                            error);
}

double default_chroma_syntax_bits(bool is_cfl, std::int32_t mode,
                                  std::int32_t angle_delta) noexcept {
  if (is_cfl) {
    // is_cfl flag plus cfl_index (derived-alpha path); no angle syntax.
    return 1.5;
  }
  return av2enc::rdo::default_intra_syntax_rate_bits(mode, angle_delta);
}

}  // namespace av2enc::still::rdo_decide
