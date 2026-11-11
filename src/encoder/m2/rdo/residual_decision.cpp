#include "encoder/m2/rdo/residual_decision.h"

namespace av2enc::m2::rdo {

residual::DecisionOptions make_residual_options(const RdoConfig& config) {
  residual::DecisionOptions options;
  options.base_q_idx = 90;
  const double lambda = lambda_sse(config);
  options.lambda = lambda;
  options.lambda_sse_scale = config.lambda.sse_scale;
  options.allow_residual = true;
  // The M2 tile writer emits TX_MODE_LARGEST with DCT_DCT only.  Searching
  // other inter TX types here would produce a decision the writer cannot
  // signal, so the search is disabled by construction.
  options.try_inter_tx_types = false;
  options.tx_type_sse_scale = 0.125;
  options.rdoq = false;
  options.rdoq_sse_scale = 0.125;
  options.enable_inter_ddt = false;
  options.reduced_tx_set = 0;
  options.enable_chroma_dctonly = true;
  return options;
}

bool decide_residual(const residual::BlockInput& input,
                     const RdoConfig& config, bool allow_residual,
                     bool force_residual, residual::BlockDecision& decision,
                     std::string* error) {
  residual::DecisionOptions options = make_residual_options(config);
  options.allow_residual = allow_residual || force_residual;
  std::string local_error;
  std::string& error_ref = error != nullptr ? *error : local_error;
  if (!residual::decide_inter_block(input, options, decision, error_ref)) {
    return false;
  }
  if (!force_residual || !decision.skip) return true;
  // Stress mode: keep the residual even when the rate-aware decision would
  // skip.  A residual that quantised to all zero reconstructs to the
  // prediction; charge the three txb_skip symbols plus skip_flag = 0.
  decision.skip = false;
  if (decision.zero_residual) {
    decision.distortion = decision.skip_distortion;
    decision.rate_bits = 1.0 + 3.0 * 1.2;
  }
  decision.cost = decision.distortion + lambda_sse(config) * decision.rate_bits;
  return true;
}

}  // namespace av2enc::m2::rdo
