#include "encoder/m2/rdo/config.h"

#include <cmath>

namespace av2enc::m2::rdo {

bool config_valid(const RdoConfig& config) noexcept {
  if (!av2enc::m1::rdo_decide::lambda_config_valid(config.lambda)) {
    return false;
  }
  if (static_cast<unsigned>(config.metric) >
      static_cast<unsigned>(av2enc::rdo::DistortionMetric::kSatd)) {
    return false;
  }
  if (static_cast<unsigned>(config.tie_break) >
      static_cast<unsigned>(av2enc::rdo::TieBreak::kRateThenDistortion)) {
    return false;
  }
  const double lambda = lambda_for_metric(config);
  return std::isfinite(lambda) && lambda >= 0.0;
}

bool validate_config(const RdoConfig& config, std::string* error) {
  std::string lambda_error;
  if (!av2enc::m1::rdo_decide::validate_lambda_config(config.lambda,
                                                      &lambda_error)) {
    if (error != nullptr) *error = lambda_error;
    return false;
  }
  if (static_cast<unsigned>(config.metric) >
      static_cast<unsigned>(av2enc::rdo::DistortionMetric::kSatd)) {
    if (error != nullptr) *error = "RdoConfig.metric is not valid";
    return false;
  }
  if (static_cast<unsigned>(config.tie_break) >
      static_cast<unsigned>(av2enc::rdo::TieBreak::kRateThenDistortion)) {
    if (error != nullptr) *error = "RdoConfig.tie_break is not valid";
    return false;
  }
  const double lambda = lambda_for_metric(config);
  if (!std::isfinite(lambda) || lambda < 0.0) {
    if (error != nullptr) *error = "RdoConfig.lambda is not finite";
    return false;
  }
  return true;
}

RdoConfig calibrated_m2_config(std::string* error) {
  RdoConfig config;
  std::string m1_error;
  config.lambda = av2enc::m1::rdo_decide::calibrated_m1_config(&m1_error);
  if (!m1_error.empty()) {
    if (error != nullptr) *error = m1_error;
    return RdoConfig{};
  }
  config.metric = av2enc::rdo::DistortionMetric::kSsd;
  config.tie_break = av2enc::rdo::TieBreak::kDistortionThenRate;
  config.rate_aware = true;
  return config;
}

double lambda_sse(const RdoConfig& config) noexcept {
  return av2enc::m1::rdo_decide::lambda_sse(config.lambda);
}

double lambda_for_metric(const RdoConfig& config) noexcept {
  const double sse = lambda_sse(config);
  if (config.metric == av2enc::rdo::DistortionMetric::kSsd) {
    return sse;
  }
  return std::sqrt(sse);
}

av2enc::rdo::RdoConfig selector_config(const RdoConfig& config) noexcept {
  av2enc::rdo::RdoConfig out;
  out.metric = config.metric;
  out.tie_break = config.tie_break;
  out.lambda = config.rate_aware ? lambda_for_metric(config) : 0.0;
  return out;
}

}  // namespace av2enc::m2::rdo
