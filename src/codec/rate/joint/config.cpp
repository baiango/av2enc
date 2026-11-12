#include "config.h"

#include <cmath>

namespace av2enc::still::rdo_loop {

bool lambda_config_valid(const LambdaConfig& config) noexcept {
  return std::isfinite(config.quant_step) && config.quant_step > 0.0 &&
         std::isfinite(config.sse_scale) && config.sse_scale >= 0.0 &&
         std::isfinite(config.ctu_activity_strength) &&
         config.ctu_activity_strength >= 0.0 &&
         std::isfinite(config.ctu_scale_min) && config.ctu_scale_min > 0.0 &&
         std::isfinite(config.ctu_scale_max) &&
         config.ctu_scale_max >= config.ctu_scale_min;
}

bool validate_lambda_config(const LambdaConfig& config, std::string* error) {
  if (!std::isfinite(config.quant_step) || config.quant_step <= 0.0) {
    if (error != nullptr) {
      *error = "LambdaConfig.quant_step must be finite and > 0";
    }
    return false;
  }
  if (!std::isfinite(config.sse_scale) || config.sse_scale < 0.0) {
    if (error != nullptr) {
      *error = "LambdaConfig.sse_scale must be finite and >= 0";
    }
    return false;
  }
  if (!std::isfinite(config.ctu_activity_strength) ||
      config.ctu_activity_strength < 0.0) {
    if (error != nullptr) {
      *error = "LambdaConfig.ctu_activity_strength must be finite and >= 0";
    }
    return false;
  }
  if (!std::isfinite(config.ctu_scale_min) || config.ctu_scale_min <= 0.0 ||
      !std::isfinite(config.ctu_scale_max) ||
      config.ctu_scale_max < config.ctu_scale_min) {
    if (error != nullptr) {
      *error = "LambdaConfig CTU scale bounds must be finite, > 0, min <= max";
    }
    return false;
  }
  return true;
}

double lambda_sse(const LambdaConfig& config) noexcept {
  if (!lambda_config_valid(config)) {
    return 0.0;
  }
  return av2enc::rdo::lambda_sse_from_quant_step(config.quant_step,
                                                 config.sse_scale);
}

CtuLambda ctu_lambda(const LambdaConfig& config, std::uint32_t activity,
                     std::uint32_t mean_activity) noexcept {
  CtuLambda out;
  const double base = lambda_sse(config);
  out.lambda = base;
  out.scale = 1.0;
  if (config.ctu_activity_strength <= 0.0 || mean_activity == 0) {
    return out;
  }
  const double ratio = static_cast<double>(activity) /
                       static_cast<double>(mean_activity);
  double scale =
      1.0 + config.ctu_activity_strength * (ratio - 1.0);
  if (scale < config.ctu_scale_min) {
    scale = config.ctu_scale_min;
  } else if (scale > config.ctu_scale_max) {
    scale = config.ctu_scale_max;
  }
  out.scale = scale;
  out.lambda = base * scale;
  return out;
}

rdo_decide::LambdaConfig to_decide_config(const LambdaConfig& config) noexcept {
  rdo_decide::LambdaConfig out;
  out.quant_step = config.quant_step;
  out.sse_scale = config.sse_scale;
  out.rate_aware = config.rate_aware;
  return out;
}

LambdaConfig calibrated_still_config(std::string* error) {
  LambdaConfig config;
  if (error != nullptr) {
    error->clear();
  }
  return config;
}

double block_cost(double distortion, double rate_bits,
                  const LambdaConfig& config) noexcept {
  if (!config.rate_aware) {
    return distortion;
  }
  return distortion + lambda_sse(config) * rate_bits;
}

}  // namespace av2enc::still::rdo_loop
