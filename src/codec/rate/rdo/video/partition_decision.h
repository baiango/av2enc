// Rate-aware partition NONE / HORZ / VERT decision for the M2 P-frame path.
//
// The tree geometry, spec legality and forced-boundary behaviour stay in
// `src/encoder/partition/`.  The Lagrange conversion stays in
// `src/encoder/m1/rdo_decide/partition_rdo.*`.  This wrapper only:
//   * supplies the M2 M1-QP-90 lambda (step 66),
//   * exposes a raw (distortion, bits) leaf callback, so the integration can
//     return a real inter leaf cost instead of a variance proxy,
//   * fixes the partition signal bits (2 bits for SPLIT/HORZ/VERT).
//
// The experiment uses `variance_leaf_rate_bits()`, the documented M1 leaf-rate
// proxy, because the M2 partition search runs before prediction.  A production
// integration that has a per-node inter cost should return it from the leaf
// callback; the solver and tie-breaks are unchanged.

#pragma once

#include "codec/rate/rdo/video/config.h"
#include "codec/partition/tree/partition.h"

#include <functional>
#include <string>

namespace av2enc::video::rdo {

struct LeafCost {
  double distortion = partition::kInfiniteCost;
  double rate_bits = 0.0;
};

using LeafCostFunction = std::function<LeafCost(
    const partition::PartitionNodeInfo& info)>;

struct PartitionRdoConfig {
  RdoConfig rdo;
  double none_signal_bits = 0.0;
  double split_signal_bits = 2.0;
  double horz_signal_bits = 2.0;
  double vert_signal_bits = 2.0;
};

// Runs the M1 rate-aware partition solver with the M2 lambda and the caller's
// leaf costs.  The returned tree is the same `partition::FrameDecision` the
// core validates and consumes.  On failure the frame's status/error describe
// the problem.
partition::FrameDecision decide_partition_rdo(
    const partition::FrameGeometry& geometry,
    const partition::DecisionOptions& options,
    const LeafCostFunction& leaf_costs, const PartitionRdoConfig& config,
    std::string* error = nullptr);

// Documented variance-based leaf-rate proxy (delegates to M1's
// `estimate_leaf_rate_bits()`).  It is a ranking aid, not an entropy estimate.
double variance_leaf_rate_bits(double sse, int w, int h,
                               double quant_step) noexcept;

}  // namespace av2enc::video::rdo
