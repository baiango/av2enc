#include "encoder/m2/rdo/partition_decision.h"

#include "encoder/m1/rdo_decide/partition_rdo.h"

#include <utility>

namespace av2enc::m2::rdo {

partition::FrameDecision decide_partition_rdo(
    const partition::FrameGeometry& geometry,
    const partition::DecisionOptions& options,
    const LeafCostFunction& leaf_costs, const PartitionRdoConfig& config,
    std::string* error) {
  if (error != nullptr) error->clear();
  if (!leaf_costs) {
    partition::FrameDecision empty;
    empty.status = partition::DecisionStatus::kInvalidCost;
    empty.error = "partition leaf cost callback is empty";
    if (error != nullptr) *error = empty.error;
    return empty;
  }

  const auto raw_costs =
      [&leaf_costs, &config](const partition::PartitionNodeInfo& info) {
        const LeafCost leaf = leaf_costs(info);
        av2enc::m1::rdo_decide::PartitionCostInputs raw;
        raw.leaf_distortion = leaf.distortion;
        raw.leaf_rate_bits = leaf.rate_bits;
        raw.none_signal_bits = config.none_signal_bits;
        raw.split_signal_bits = config.split_signal_bits;
        raw.horz_signal_bits = config.horz_signal_bits;
        raw.vert_signal_bits = config.vert_signal_bits;
        return raw;
      };

  av2enc::m1::rdo_decide::PartitionDecideConfig pconfig;
  pconfig.lambda = lambda_sse(config.rdo);
  pconfig.rate_aware = config.rdo.rate_aware;
  // Control mode only: same lambda, no leaf-rate term.  Charging two split
  // bits keeps the signal cost comparable to the rate-aware path.
  pconfig.sse_only_split_cost = 2.0 * pconfig.lambda;

  std::string local_error;
  av2enc::m1::rdo_decide::PartitionRdoDecision decision =
      av2enc::m1::rdo_decide::decide_frame_rdo(geometry, raw_costs, options,
                                               pconfig, &local_error);
  if (error != nullptr) {
    *error = decision.error.empty() ? local_error : decision.error;
  }
  return std::move(decision.frame);
}

double variance_leaf_rate_bits(double sse, int w, int h,
                               double quant_step) noexcept {
  return av2enc::m1::rdo_decide::estimate_leaf_rate_bits(sse, w, h,
                                                         quant_step);
}

}  // namespace av2enc::m2::rdo
