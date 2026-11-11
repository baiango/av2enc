// M1 rate-aware decision layer.
//
// One include for the four decisions the M1 core can call:
//   * luma intra mode + angle      -> intra_rdo.h
//   * partition NONE/split/rect    -> partition_rdo.h
//   * TX type + RDOQ               -> tx_rdo.h
//   * chroma DC/CfL (U+V score)    -> chroma_rdo.h
//
// All decisions use the same convention:
//
//     cost = SSE + lambda * rate_bits
//     lambda = sse_scale * quant_step^2
//
// with lambda from av2enc::rdo::lambda_sse_from_quant_step() and the calibration
// documented in config.h.  Rate is in bits.  Candidate selectors are
// deterministic under exact floating-point comparison; build with
// -ffp-contract=off when CPU/GPU parity matters.
//
// The layer owns no encoder state, writes no bits, and is independent of the
// syntax writer.  It is safe to link into the CPU reference and into the GPU
// host planner; the TX-stream adapter is allocation-free by construction.

#pragma once

#include "config.h"
#include "intra_rdo.h"
#include "partition_rdo.h"
#include "tx_rdo.h"
#include "chroma_rdo.h"
#include "transform_stream.h"

namespace av2enc::m1::rdo_decide {

// Version tag for the experiment/integration logs.
inline constexpr const char* kLayerVersion = "m1-rdo-decide-1";

}  // namespace av2enc::m1::rdo_decide
