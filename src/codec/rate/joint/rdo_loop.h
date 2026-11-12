// Unified M1 rate-aware decision loop.
//
// One include for the four decisions and the shared cost machinery:
//
//   config.h    lambda calibration, per-CTU scale
//   cost.h      cost breakdown, node trace
//   search.h    partition/mode candidate policy, pruning, node selector
//   batch.h     GPU-ready candidate batch + deterministic reduction
//   activity.h  deterministic CTU activity for the lambda hint
//
// Everything here is host-side reference code.  It owns no encoder state,
// writes no bits and allocates nothing in its decision paths.  The encoder
// adapter (experiment/patch_still_rdo_loop.py in this directory) trial-encodes a
// node with the real M1 prediction/transform/RDOQ path and hands the measured
// CostTerms to search.h.  The CUDA port replaces the measurement with a kernel
// and keeps the same candidate order and comparator.
//
// Convention: distortion is SSE in 8-bit sample units, rate is bits, lambda is
// distortion units per bit.  Build with -ffp-contract=off for CPU/GPU parity.

#pragma once

#include "activity.h"
#include "batch.h"
#include "config.h"
#include "cost.h"
#include "search.h"

namespace av2enc::still::rdo_loop {

// Version tag for experiment/integration logs.
inline constexpr const char* kLoopVersion = "m1-rdo-loop-1";

}  // namespace av2enc::still::rdo_loop
