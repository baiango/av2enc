#include "intra_rdo.h"

#include <cmath>

namespace av2enc::still::rdo_decide {

av2enc::rdo::IntraRdoConfig make_intra_config(const LambdaConfig& config) {
  return av2enc::rdo::make_intra_rdo_config(
      config.quant_step, config.sse_scale, config.rate_aware);
}

av2enc::rdo::IntraModeChoice decide_luma_intra(
    std::span<const av2enc::rdo::IntraCandidateCost> candidates,
    std::span<av2enc::rdo::IntraCandidateCost> scratch,
    const LambdaConfig& config,
    av2enc::rdo::IntraRateEstimator estimator, void* estimator_context,
    std::string* error, const IntraBlockGeometry& geometry) {
  if (error != nullptr) {
    error->clear();
  }
  const av2enc::rdo::IntraRdoConfig selector_config = make_intra_config(config);
  if (!av2enc::rdo::validate_intra_rdo_config(selector_config, error)) {
    return av2enc::rdo::IntraModeChoice{};
  }
  if (estimator == nullptr) {
    return av2enc::rdo::select_intra_mode(candidates, selector_config, error);
  }
  if (scratch.size() < candidates.size()) {
    if (error != nullptr) {
      *error = "intra rate-estimator scratch span is too small";
    }
    return av2enc::rdo::IntraModeChoice{};
  }

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const av2enc::rdo::IntraCandidateCost& source = candidates[i];
    av2enc::rdo::IntraCandidateCost& target = scratch[i];
    target = source;
    av2enc::rdo::IntraRateRequest request;
    request.mode = source.mode;
    request.angle_delta = source.angle_delta;
    request.block_w = geometry.block_w;
    request.block_h = geometry.block_h;
    request.base_q_idx = geometry.base_q_idx;
    request.sse = source.sse;
    request.quant_step = config.quant_step;
    av2enc::rdo::IntraRateEstimate estimate;
    std::string local_error;
    if (!av2enc::rdo::estimate_intra_rates(
            estimator, estimator_context, request, estimate, &local_error)) {
      if (error != nullptr && error->empty()) {
        *error = local_error;
      }
      // Leave the supplied rates in place; the selector validates them.
      continue;
    }
    target.syntax_rate_bits = estimate.syntax_rate_bits;
    target.coefficient_rate_bits = estimate.coefficient_rate_bits;
  }
  return av2enc::rdo::select_intra_mode(
      scratch.first(candidates.size()), selector_config, error);
}

double intra_syntax_bits(std::int32_t mode,
                         std::int32_t angle_delta) noexcept {
  return av2enc::rdo::default_intra_syntax_rate_bits(mode, angle_delta);
}

double intra_coefficient_bits(const TransformStreamRate& rate) noexcept {
  return rate.block_bits();
}

}  // namespace av2enc::still::rdo_decide
