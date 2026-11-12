#include "codec/rate/rdo/video/inter_decision.h"

#include "codec/rate/rdo/video/residual_decision.h"

#include <array>
#include <cmath>
#include <utility>

namespace av2enc::video::rdo {
namespace {

bool finite_nonnegative(double value) {
  return std::isfinite(value) && value >= 0.0;
}

bool candidate_valid(const InterCandidate& candidate) {
  return finite_nonnegative(candidate.distortion) &&
         finite_nonnegative(candidate.rate_bits);
}

}  // namespace

bool score_inter_candidate(const InterCandidateSpec& spec,
                           const residual::BlockInput& block,
                           bool evaluate_residual,
                           const RateContext& rate_context,
                           const RdoConfig& config,
                           const InterRateModel& model,
                           ScoredInterCandidate& out,
                           std::string* error) {
  out = ScoredInterCandidate{};
  std::string local_error;
  if (!validate_config(config, &local_error)) {
    if (error != nullptr) *error = local_error;
    return false;
  }

  residual::BlockDecision decision;
  if (!decide_residual(block, config, evaluate_residual,
                       spec.force_residual, decision, &local_error)) {
    if (error != nullptr) *error = local_error;
    return false;
  }

  ModeRateInput mode;
  mode.inter_mode = spec.inter_mode;
  mode.drl_index = spec.drl_index;
  mode.mvd = spec.mvd;
  mode.ref = spec.ref;
  mode.skip_flag = decision.skip;
  mode.is_inter = true;
  mode.context = rate_context;

  const double mode_bits = model.mode_bits(mode);
  const double residual_bits = model.residual_bits(decision);
  const double rate_bits = mode_bits + residual_bits;
  const double distortion =
      spec.has_distortion_override
          ? spec.distortion_override
          : (decision.skip ? decision.skip_distortion : decision.distortion);
  if (!finite_nonnegative(mode_bits) || !finite_nonnegative(residual_bits) ||
      !finite_nonnegative(distortion)) {
    if (error != nullptr) *error = "inter candidate rate/distortion invalid";
    return false;
  }

  InterCandidate& candidate = out.candidate;
  candidate.kind = spec.kind;
  candidate.inter_mode = spec.inter_mode;
  candidate.drl_index = spec.drl_index;
  candidate.mv = spec.mv;
  candidate.mvd = spec.mvd;
  candidate.ref = spec.ref;
  candidate.skip_flag = decision.skip;
  candidate.distortion = distortion;
  candidate.mode_bits = mode_bits;
  candidate.residual_bits = residual_bits;
  candidate.rate_bits = rate_bits;
  candidate.cost = distortion + (config.rate_aware
                                     ? lambda_for_metric(config) * rate_bits
                                     : 0.0);
  candidate.valid = true;
  out.residual = std::move(decision);
  out.has_residual = evaluate_residual || spec.force_residual;
  out.valid = true;
  return true;
}

InterCandidate make_intra_candidate(double distortion, double coefficient_bits,
                                    int y_mode, int y_angle_delta, int uv_mode,
                                    const RateContext& rate_context,
                                    const RdoConfig& config,
                                    const InterRateModel& model) {
  InterCandidate candidate;
  candidate.kind = InterCandidateKind::kIntra;
  candidate.inter_mode = -1;
  candidate.drl_index = 0;
  candidate.skip_flag = false;
  candidate.distortion = distortion;
  candidate.mode_bits = model.intra_syntax_bits(
      y_mode, y_angle_delta, uv_mode, rate_context.is_inter_ctx);
  candidate.residual_bits = coefficient_bits;
  candidate.rate_bits = candidate.mode_bits + candidate.residual_bits;
  candidate.cost = distortion + (config.rate_aware
                                     ? lambda_for_metric(config) *
                                           candidate.rate_bits
                                     : 0.0);
  candidate.valid = finite_nonnegative(candidate.distortion) &&
                    finite_nonnegative(candidate.mode_bits) &&
                    finite_nonnegative(candidate.residual_bits);
  return candidate;
}

InterDecision decide_inter_candidates(
    std::span<const InterCandidate> candidates, const RdoConfig& config,
    std::string* error) {
  InterDecision result;
  std::string local_error;
  if (!validate_config(config, &local_error)) {
    if (error != nullptr) *error = local_error;
    return result;
  }
  if (candidates.empty()) {
    if (error != nullptr) *error = "no inter candidates";
    return result;
  }

  const av2enc::rdo::RdoConfig selector = selector_config(config);
  av2enc::rdo::CandidateSelector selector_state(selector);
  std::array<int, 16> accepted{};
  int accepted_count = 0;

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const InterCandidate& candidate = candidates[i];
    if (!candidate.valid || !candidate_valid(candidate)) continue;
    av2enc::rdo::CandidateCost input;
    input.kind = av2enc::rdo::CandidateKind::kInter;
    input.id = static_cast<std::int32_t>(candidate.kind);
    input.aux = candidate.drl_index * 2 + (candidate.skip_flag ? 0 : 1);
    input.distortion = candidate.distortion;
    input.rate_bits = candidate.rate_bits;
    if (!selector_state.add(input, &local_error)) {
      // A single malformed candidate must not discard the rest of the list;
      // the shared selector counted it as rejected.
      continue;
    }
    if (accepted_count >= static_cast<int>(accepted.size())) {
      if (error != nullptr) *error = "too many inter candidates";
      return result;
    }
    accepted[static_cast<std::size_t>(accepted_count)] = static_cast<int>(i);
    ++accepted_count;
  }

  const av2enc::rdo::RdoDecision& decision = selector_state.decision();
  if (!decision.found ||
      decision.best.input_index >= accepted_count) {
    if (error != nullptr) *error = "no valid inter candidate";
    return result;
  }
  result.found = true;
  result.best_index = accepted[static_cast<std::size_t>(
      decision.best.input_index)];
  result.candidate = candidates[static_cast<std::size_t>(result.best_index)];
  result.cost = decision.best.cost;
  result.margin = decision.margin();
  result.candidate_count = decision.candidate_count;
  return result;
}

}  // namespace av2enc::video::rdo
