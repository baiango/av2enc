// Luma intra mode + angle decision for the M1 full-intra search.
//
// This is the M1-facing wrapper around src/encoder/rdo/intra_decision.*.  It
// adds the shared lambda calibration (config.h) and an allocation-free way to
// plug a rate estimator into a fixed candidate array.
//
// The selector keeps the documented deterministic order:
//   1. lowest SSE + lambda * (syntax + coefficient) bits
//   2. lowest SSE
//   3. lowest rate
//   4. fixed AV2 candidate order (13 base modes, then angle deltas
//      -3,-2,-1,1,2,3 per directional mode)
//   5. input index
//
// No allocation happens here: candidates and scratch are caller-owned spans.

#pragma once

#include "config.h"
#include "encoder/rdo/intra_decision.h"
#include "transform_stream.h"

#include <cstdint>
#include <span>
#include <string>

namespace av2enc::m1::rdo_decide {

struct IntraBlockGeometry {
  std::int32_t block_w = 0;
  std::int32_t block_h = 0;
  std::int32_t base_q_idx = 90;
};

// Builds the rdo selector config from this layer's lambda calibration.
av2enc::rdo::IntraRdoConfig make_intra_config(const LambdaConfig& config);

// Selects a luma mode from already-costed candidates.  `scratch` is only used
// when `estimator` is non-null; it must hold at least `candidates.size()`
// entries.  When the estimator is null, the candidate's syntax/coefficient
// fields are used as supplied.  Both paths are allocation-free.
// `geometry` fills the estimator request; it is ignored when estimator is null.
av2enc::rdo::IntraModeChoice decide_luma_intra(
    std::span<const av2enc::rdo::IntraCandidateCost> candidates,
    std::span<av2enc::rdo::IntraCandidateCost> scratch,
    const LambdaConfig& config,
    av2enc::rdo::IntraRateEstimator estimator = nullptr,
    void* estimator_context = nullptr, std::string* error = nullptr,
    const IntraBlockGeometry& geometry = {});

// Default syntax cost for a mode/angle pair, in bits.  Placeholder in the RDO
// framework; the integrator should replace it with the neighbour-dependent
// §5.20.5.5 Y-mode set cost.
double intra_syntax_bits(std::int32_t mode,
                         std::int32_t angle_delta) noexcept;

// Coefficient-rate term from a candidate's real quantised levels through the
// transform-stream adapter.  `rate` must already be initialised for the block.
double intra_coefficient_bits(const TransformStreamRate& rate) noexcept;

}  // namespace av2enc::m1::rdo_decide
