#include "config.h"

#include <cmath>

namespace av2enc::still::rdo_decide {

bool lambda_config_valid(const LambdaConfig& config) noexcept {
  return std::isfinite(config.quant_step) && config.quant_step > 0.0 &&
         std::isfinite(config.sse_scale) && config.sse_scale >= 0.0;
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
  return true;
}

double lambda_sse(const LambdaConfig& config) noexcept {
  if (!lambda_config_valid(config)) {
    return 0.0;
  }
  return av2enc::rdo::lambda_sse_from_quant_step(config.quant_step,
                                                 config.sse_scale);
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

}  // namespace av2enc::still::rdo_decide
