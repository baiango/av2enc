// Rate-aware partition NONE / split / rect decision.
//
// The geometry, spec legality and deterministic tie-break stay in
// src/encoder/partition/.  This layer converts raw (distortion, rate bits)
// inputs into the Lagrangian NodeCosts that component expects, then calls the
// existing solver.  That keeps one authority for what the decoder will accept:
// PARTITION_SPLIT is still illegal at 64x64 and below, and forced boundary
// partitions still ignore caller policy.
//
// Cost convention
// ---------------
//   rate_aware = true:
//       leaf_cost    = leaf_distortion + lambda * leaf_rate_bits
//       signal_cost  = lambda * signal_rate_bits
//   rate_aware = false:
//       leaf_cost    = leaf_distortion
//       signal_cost  = sse_only_split_cost for every non-NONE partition
//
// The second mode reproduces the pre-RDO M1 partition search (variance SSE plus
// a fixed signal cost) so an experiment can compare the two directly.

#pragma once

#include "config.h"
#include "codec/partition/tree/partition.h"

#include <functional>
#include <string>

namespace av2enc::still::rdo_decide {

struct PartitionCostInputs {
  double leaf_distortion = partition::kInfiniteCost;
  double leaf_rate_bits = 0.0;
  double none_signal_bits = 0.0;
  double split_signal_bits = 0.0;
  double horz_signal_bits = 0.0;
  double vert_signal_bits = 0.0;
};

struct PartitionDecideConfig {
  double lambda = 0.0;
  bool rate_aware = true;
  // Used only when rate_aware is false, in distortion units.
  double sse_only_split_cost = 0.0;
};

bool partition_config_valid(const PartitionDecideConfig& config) noexcept;
bool validate_partition_config(const PartitionDecideConfig& config,
                               std::string* error = nullptr);

// Converts one node's raw inputs to the partition module's cost form.
partition::NodeCosts lagrangian_node_costs(
    const PartitionCostInputs& inputs,
    const PartitionDecideConfig& config);

// Raw cost callback: distortion plus *bits*.  It must be deterministic; the
// solver calls it once per node candidate.
using PartitionRawCostFunction =
    std::function<PartitionCostInputs(const partition::PartitionNodeInfo&)>;

struct PartitionRdoDecision {
  partition::DecisionStatus status = partition::DecisionStatus::kInvalidGeometry;
  partition::FrameDecision frame;
  double cost = 0.0;
  int nodes = 0;
  int leaves = 0;
  std::string error;
};

// Frame-level convenience wrapper.  This is a frame/superblock operation, not
// the inner candidate loop; the returned tree owns its nodes.
PartitionRdoDecision decide_frame_rdo(const partition::FrameGeometry& geometry,
                                      const PartitionRawCostFunction& costs,
                                      const partition::DecisionOptions& options,
                                      const PartitionDecideConfig& config,
                                      std::string* error = nullptr);

// Single-superblock variant.
PartitionRdoDecision decide_superblock_rdo(
    const partition::FrameGeometry& geometry, int sb_mi_row, int sb_mi_col,
    const PartitionRawCostFunction& costs,
    const partition::DecisionOptions& options,
    const PartitionDecideConfig& config, std::string* error = nullptr);

// Documented proxy for a leaf's coefficient+syntax rate when the caller only
// has a distortion estimate (the M1 partition search runs before prediction).
// It is a ranking aid, not an entropy estimate.  `sse` is in sample-domain
// squared error, `w`/`h` are in samples.
double estimate_leaf_rate_bits(double sse, int w, int h,
                               double quant_step) noexcept;

}  // namespace av2enc::still::rdo_decide
