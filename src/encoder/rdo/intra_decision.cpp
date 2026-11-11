#include "intra_decision.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>

namespace av2enc::rdo {
namespace {

bool is_directional_mode(std::int32_t mode) noexcept {
  return mode >= 1 && mode <= 8;
}

// Matches intra::build_mode_candidates(): all 13 base modes first, then the
// six non-zero deltas per directional mode in -3,-2,-1,1,2,3 order.
int mode_order(std::int32_t mode, std::int32_t angle_delta) noexcept {
  if (angle_delta == 0) {
    return mode;
  }
  const int delta_index =
      angle_delta < 0 ? angle_delta + 3 : angle_delta + 2;
  return kIntraModeCount + (mode - 1) * 6 + delta_index;
}

std::int32_t abs_i32(std::int32_t value) noexcept {
  if (value == std::numeric_limits<std::int32_t>::min()) {
    return std::numeric_limits<std::int32_t>::max();
  }
  return value < 0 ? -value : value;
}

struct ScoredCandidate {
  IntraCandidateCost candidate{};
  double cost = 0.0;
  double rate_bits = 0.0;
  std::int32_t input_index = 0;
};

bool better_candidate(const ScoredCandidate& a,
                      const ScoredCandidate& b,
                      const IntraRdoConfig& config) noexcept {
  if (a.cost != b.cost) {
    return a.cost < b.cost;
  }
  if (a.candidate.sse != b.candidate.sse) {
    return a.candidate.sse < b.candidate.sse;
  }
  if (config.rate_aware && config.prefer_lower_rate_on_tie &&
      a.rate_bits != b.rate_bits) {
    return a.rate_bits < b.rate_bits;
  }
  const int order_a = mode_order(a.candidate.mode, a.candidate.angle_delta);
  const int order_b = mode_order(b.candidate.mode, b.candidate.angle_delta);
  if (order_a != order_b) {
    return order_a < order_b;
  }
  return a.input_index < b.input_index;
}

bool candidate_valid(const IntraCandidateCost& candidate) noexcept {
  if (candidate.mode < 0 || candidate.mode >= kIntraModeCount) {
    return false;
  }
  if (candidate.angle_delta < -kIntraMaxAngleDelta ||
      candidate.angle_delta > kIntraMaxAngleDelta) {
    return false;
  }
  if (candidate.angle_delta != 0 &&
      !is_directional_mode(candidate.mode)) {
    return false;
  }
  return std::isfinite(candidate.sse) && candidate.sse >= 0.0 &&
         std::isfinite(candidate.syntax_rate_bits) &&
         candidate.syntax_rate_bits >= 0.0 &&
         std::isfinite(candidate.coefficient_rate_bits) &&
         candidate.coefficient_rate_bits >= 0.0;
}

}  // namespace

bool intra_rdo_config_valid(const IntraRdoConfig& config) noexcept {
  return std::isfinite(config.lambda) && config.lambda >= 0.0;
}

bool validate_intra_rdo_config(const IntraRdoConfig& config,
                               std::string* error) {
  if (!std::isfinite(config.lambda) || config.lambda < 0.0) {
    if (error != nullptr) {
      *error = "IntraRdoConfig.lambda must be finite and >= 0";
    }
    return false;
  }
  return true;
}

IntraModeSelector::IntraModeSelector(const IntraRdoConfig& config) noexcept
    : config_(config) {}

bool IntraModeSelector::add(const IntraCandidateCost& candidate,
                            std::string* error) {
  if (!intra_rdo_config_valid(config_)) {
    ++choice_.rejected_count;
    if (error != nullptr) {
      *error = "invalid IntraRdoConfig";
    }
    return false;
  }
  if (!candidate_valid(candidate)) {
    ++choice_.rejected_count;
    if (error != nullptr) {
      *error = "invalid intra candidate (mode/angle/distortion/rate)";
    }
    return false;
  }

  const double rate_bits = intra_candidate_rate_bits(candidate);
  ScoredCandidate score;
  score.candidate = candidate;
  score.rate_bits = rate_bits;
  score.cost = config_.rate_aware
                   ? candidate.sse + config_.lambda * rate_bits
                   : candidate.sse;
  if (!std::isfinite(score.cost)) {
    ++choice_.rejected_count;
    if (error != nullptr) {
      *error = "intra candidate cost is not finite";
    }
    return false;
  }
  score.input_index = choice_.candidate_count;
  ++choice_.candidate_count;

  if (!choice_.found) {
    choice_.found = true;
    choice_.mode = candidate.mode;
    choice_.angle_delta = candidate.angle_delta;
    choice_.sse = candidate.sse;
    choice_.rate_bits = rate_bits;
    choice_.cost = score.cost;
    choice_.input_index = score.input_index;
    return true;
  }

  ScoredCandidate current;
  current.candidate.mode = choice_.mode;
  current.candidate.angle_delta = choice_.angle_delta;
  current.candidate.sse = choice_.sse;
  current.rate_bits = choice_.rate_bits;
  current.cost = choice_.cost;
  current.input_index = choice_.input_index;

  if (better_candidate(score, current, config_)) {
    choice_.mode = candidate.mode;
    choice_.angle_delta = candidate.angle_delta;
    choice_.sse = candidate.sse;
    choice_.rate_bits = rate_bits;
    choice_.cost = score.cost;
    choice_.input_index = score.input_index;
  }
  return true;
}

void IntraModeSelector::reset() noexcept { choice_ = IntraModeChoice{}; }

IntraModeChoice select_intra_mode(std::span<const IntraCandidateCost> candidates,
                                  const IntraRdoConfig& config,
                                  std::string* error) {
  IntraModeSelector selector(config);
  for (const IntraCandidateCost& candidate : candidates) {
    std::string local_error;
    if (!selector.add(candidate, &local_error)) {
      if (error != nullptr && error->empty()) {
        *error = local_error;
      }
    }
  }
  return selector.choice();
}

bool intra_candidate_better(const IntraCandidateCost& a,
                            const IntraCandidateCost& b,
                            const IntraRdoConfig& config) noexcept {
  if (!intra_rdo_config_valid(config) || !candidate_valid(a) ||
      !candidate_valid(b)) {
    return false;
  }
  ScoredCandidate score_a;
  score_a.candidate = a;
  score_a.rate_bits = intra_candidate_rate_bits(a);
  score_a.cost = config.rate_aware
                     ? a.sse + config.lambda * score_a.rate_bits
                     : a.sse;
  ScoredCandidate score_b;
  score_b.candidate = b;
  score_b.rate_bits = intra_candidate_rate_bits(b);
  score_b.cost = config.rate_aware
                     ? b.sse + config.lambda * score_b.rate_bits
                     : b.sse;
  if (!std::isfinite(score_a.cost) || !std::isfinite(score_b.cost)) {
    return false;
  }
  return better_candidate(score_a, score_b, config);
}

IntraRdoConfig make_intra_rdo_config(double quant_step,
                                     double sse_scale,
                                     bool rate_aware) {
  IntraRdoConfig config;
  config.lambda = lambda_sse_from_quant_step(quant_step, sse_scale);
  config.rate_aware = rate_aware;
  return config;
}

IntraRdoConfig make_intra_rdo_config_from_qp(int base_q_idx,
                                             double sse_scale,
                                             bool rate_aware) {
  return make_intra_rdo_config(quant_step_from_qp(base_q_idx), sse_scale,
                               rate_aware);
}

double default_intra_syntax_rate_bits(std::int32_t mode,
                                      std::int32_t angle_delta) noexcept {
  double bits = 5.0;
  switch (mode) {
    case 0:  // DC
      bits = 2.0;
      break;
    case 1:  // V
    case 2:  // H
      bits = 3.0;
      break;
    case 3:  // D45
    case 4:  // D135
    case 5:  // D113
    case 6:  // D157
    case 7:  // D203
    case 8:  // D67
      bits = 4.0;
      break;
    case 9:  // SMOOTH
    case 10: // SMOOTH_V
    case 11: // SMOOTH_H
    case 12: // PAETH
      bits = 4.5;
      break;
    default:
      bits = 5.0;
      break;
  }
  const std::int64_t delta =
      angle_delta < 0 ? -static_cast<std::int64_t>(angle_delta)
                      : static_cast<std::int64_t>(angle_delta);
  return bits + 2.0 * static_cast<double>(delta);
}

double default_intra_coefficient_rate_bits(double sse,
                                           std::int32_t block_w,
                                           std::int32_t block_h,
                                           double quant_step) noexcept {
  if (!std::isfinite(sse) || sse < 0.0 || block_w <= 0 || block_h <= 0 ||
      !std::isfinite(quant_step) || quant_step <= 0.0) {
    return 0.0;
  }
  const double area = static_cast<double>(block_w) *
                      static_cast<double>(block_h);
  const double rms = std::sqrt(sse / area);
  const double significant =
      area * std::min(1.0, rms / (0.5 * quant_step));
  // Placeholder: one framing bit per four samples plus two bits per estimated
  // significant coefficient. This is deliberately simple and not calibrated.
  return 0.25 * area + 2.0 * significant;
}

double intra_coefficient_rate_bits_from_levels(
    std::span<const std::int32_t> levels_scan,
    std::int32_t eob,
    std::int32_t num_coefficients) noexcept {
  if (num_coefficients <= 0) {
    return 0.0;
  }
  const std::int32_t count = static_cast<std::int32_t>(
      std::min<std::size_t>(levels_scan.size(), static_cast<std::size_t>(
                                                   std::max(eob, 0))));
  double bits = 0.0;
  std::int32_t previous_abs = 0;
  for (std::int32_t i = 0; i < count; ++i) {
    const std::int32_t abs_level = abs_i32(levels_scan[static_cast<std::size_t>(i)]);
    bits += default_rdoq_coefficient_rate(nullptr, i, previous_abs, abs_level, 0);
    previous_abs = abs_level == 0 ? 0 : (abs_level == 1 ? 1 : 2);
  }
  bits += default_rdoq_eob_rate(nullptr, count, num_coefficients);
  return bits;
}

bool estimate_intra_rates(IntraRateEstimator estimator,
                          void* context,
                          const IntraRateRequest& request,
                          IntraRateEstimate& out,
                          std::string* error) {
  out = IntraRateEstimate{};
  if (estimator != nullptr) {
    if (!estimator(context, request, out)) {
      if (error != nullptr) {
        *error = "IntraRateEstimator rejected the candidate";
      }
      return false;
    }
  } else {
    out.syntax_rate_bits =
        default_intra_syntax_rate_bits(request.mode, request.angle_delta);
    out.coefficient_rate_bits =
        default_intra_coefficient_rate_bits(request.sse, request.block_w,
                                            request.block_h,
                                            request.quant_step);
  }
  if (!std::isfinite(out.syntax_rate_bits) ||
      out.syntax_rate_bits < 0.0 ||
      !std::isfinite(out.coefficient_rate_bits) ||
      out.coefficient_rate_bits < 0.0) {
    if (error != nullptr) {
      *error = "intra rate estimate must be finite and >= 0";
    }
    return false;
  }
  return true;
}

}  // namespace av2enc::rdo
