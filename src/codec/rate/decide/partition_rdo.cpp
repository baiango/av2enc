#include "partition_rdo.h"

#include "codec/rate/rdo/intra_decision.h"
#include "codec/rate/rdo/rdo.h"

#include <cmath>
#include <utility>

namespace av2enc::still::rdo_decide {
namespace {

bool rate_is_valid(double bits) noexcept {
  return std::isfinite(bits) && bits >= 0.0;
}

bool distortion_is_valid(double distortion) noexcept {
  return std::isinf(distortion)
             ? distortion > 0.0
             : (std::isfinite(distortion) && distortion >= 0.0);
}

bool inputs_valid(const PartitionCostInputs& inputs) noexcept {
  return distortion_is_valid(inputs.leaf_distortion) &&
         rate_is_valid(inputs.leaf_rate_bits) &&
         rate_is_valid(inputs.none_signal_bits) &&
         rate_is_valid(inputs.split_signal_bits) &&
         rate_is_valid(inputs.horz_signal_bits) &&
         rate_is_valid(inputs.vert_signal_bits);
}

}  // namespace

bool partition_config_valid(const PartitionDecideConfig& config) noexcept {
  return std::isfinite(config.lambda) && config.lambda >= 0.0 &&
         std::isfinite(config.sse_only_split_cost) &&
         config.sse_only_split_cost >= 0.0;
}

bool validate_partition_config(const PartitionDecideConfig& config,
                               std::string* error) {
  if (!std::isfinite(config.lambda) || config.lambda < 0.0) {
    if (error != nullptr) {
      *error = "PartitionDecideConfig.lambda must be finite and >= 0";
    }
    return false;
  }
  if (!std::isfinite(config.sse_only_split_cost) ||
      config.sse_only_split_cost < 0.0) {
    if (error != nullptr) {
      *error = "PartitionDecideConfig.sse_only_split_cost must be >= 0";
    }
    return false;
  }
  return true;
}

partition::NodeCosts lagrangian_node_costs(
    const PartitionCostInputs& inputs,
    const PartitionDecideConfig& config) {
  partition::NodeCosts costs;
  if (config.rate_aware) {
    const double lambda =
        std::isfinite(config.lambda) && config.lambda >= 0.0 ? config.lambda
                                                             : 0.0;
    costs.leaf = std::isinf(inputs.leaf_distortion)
                     ? partition::kInfiniteCost
                     : inputs.leaf_distortion + lambda * inputs.leaf_rate_bits;
    costs.none_signal = lambda * inputs.none_signal_bits;
    costs.split_signal = lambda * inputs.split_signal_bits;
    costs.horz_signal = lambda * inputs.horz_signal_bits;
    costs.vert_signal = lambda * inputs.vert_signal_bits;
  } else {
    costs.leaf = inputs.leaf_distortion;
    costs.none_signal = 0.0;
    costs.split_signal = config.sse_only_split_cost;
    costs.horz_signal = config.sse_only_split_cost;
    costs.vert_signal = config.sse_only_split_cost;
  }
  return costs;
}

PartitionRdoDecision decide_frame_rdo(const partition::FrameGeometry& geometry,
                                      const PartitionRawCostFunction& costs,
                                      const partition::DecisionOptions& options,
                                      const PartitionDecideConfig& config,
                                      std::string* error) {
  PartitionRdoDecision result;
  if (error != nullptr) error->clear();
  if (!validate_partition_config(config, error)) {
    result.status = partition::DecisionStatus::kInvalidCost;
    return result;
  }
  if (!costs) {
    result.status = partition::DecisionStatus::kInvalidCost;
    result.error = "partition raw cost callback is empty";
    if (error != nullptr) *error = result.error;
    return result;
  }

  bool bad_input = false;
  const partition::CostFunction cost_fn =
      [&](const partition::PartitionNodeInfo& info) {
        const PartitionCostInputs raw = costs(info);
        if (!inputs_valid(raw)) {
          bad_input = true;
          partition::NodeCosts invalid;
          invalid.leaf = partition::kInfiniteCost;
          return invalid;
        }
        return lagrangian_node_costs(raw, config);
      };

  result.frame = partition::decide_frame(geometry, cost_fn, options);
  result.status = result.frame.status;
  result.cost = result.frame.cost;
  result.nodes = result.frame.nodes;
  result.leaves = result.frame.leaves;
  result.error = result.frame.error;
  if (bad_input) {
    result.status = partition::DecisionStatus::kInvalidCost;
    result.error = "partition raw costs must be finite, >= 0, or +inf";
  }
  if (result.status != partition::DecisionStatus::kOk && result.error.empty()) {
    result.error = "partition decision failed";
  }
  if (result.status != partition::DecisionStatus::kOk && error != nullptr) {
    *error = result.error;
  }
  return result;
}

PartitionRdoDecision decide_superblock_rdo(
    const partition::FrameGeometry& geometry, int sb_mi_row, int sb_mi_col,
    const PartitionRawCostFunction& costs,
    const partition::DecisionOptions& options,
    const PartitionDecideConfig& config, std::string* error) {
  PartitionRdoDecision result;
  if (error != nullptr) error->clear();
  if (!validate_partition_config(config, error)) {
    result.status = partition::DecisionStatus::kInvalidCost;
    return result;
  }
  if (!costs) {
    result.status = partition::DecisionStatus::kInvalidCost;
    result.error = "partition raw cost callback is empty";
    if (error != nullptr) *error = result.error;
    return result;
  }

  bool bad_input = false;
  const partition::CostFunction cost_fn =
      [&](const partition::PartitionNodeInfo& info) {
        const PartitionCostInputs raw = costs(info);
        if (!inputs_valid(raw)) {
          bad_input = true;
          partition::NodeCosts invalid;
          invalid.leaf = partition::kInfiniteCost;
          return invalid;
        }
        return lagrangian_node_costs(raw, config);
      };

  const partition::PartitionDecision decision =
      partition::decide_superblock(geometry, sb_mi_row, sb_mi_col, cost_fn,
                                   options);
  result.status = decision.status;
  result.cost = decision.cost;
  result.nodes = decision.nodes;
  result.leaves = decision.leaves;
  result.error = decision.error;
  // Reuse FrameDecision as the one-tree carrier so the result type is shared.
  result.frame.status = decision.status;
  result.frame.cost = decision.cost;
  result.frame.nodes = decision.nodes;
  result.frame.leaves = decision.leaves;
  result.frame.error = decision.error;
  result.frame.superblocks.push_back(decision.root);
  if (bad_input) {
    result.status = partition::DecisionStatus::kInvalidCost;
    result.error = "partition raw costs must be finite, >= 0, or +inf";
  }
  if (result.status != partition::DecisionStatus::kOk && result.error.empty()) {
    result.error = "partition decision failed";
  }
  if (result.status != partition::DecisionStatus::kOk && error != nullptr) {
    *error = result.error;
  }
  return result;
}

double estimate_leaf_rate_bits(double sse, int w, int h,
                               double quant_step) noexcept {
  if (!std::isfinite(sse) || sse < 0.0 || w <= 0 || h <= 0 ||
      !std::isfinite(quant_step) || quant_step <= 0.0) {
    return 0.0;
  }
  // 3 bits buys a typical intra mode/angle; the coefficient term uses the RDO
  // framework's documented SSE-based placeholder.  This is deliberately a
  // ranking aid: the partition search cannot run a full transform before the
  // reconstruction order is known.
  return 3.0 + av2enc::rdo::default_intra_coefficient_rate_bits(
                   sse, w, h, quant_step);
}

}  // namespace av2enc::still::rdo_decide
