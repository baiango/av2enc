// Umbrella include for the M2 P-frame rate-aware decision layer.
//
// Components:
//   config.h              M1 QP-90 lambda (step 66), metric/tie-break config
//   rate_model.h          §9.3 CDF mode/MV/ref bits + pluggable callbacks
//   residual_decision.*   skip_flag decision on top of m2/residual
//   inter_decision.*      candidate cost + deterministic selection
//   intra_fallback.*      M1 TransformStreamRate for the DC fallback
//   partition_decision.*  M1 rate-aware partition solver wrapper
//
// This layer owns no encoder state, reads no bitstream state, and writes no
// bits.  It is the decision half of the P-frame path; the tile writer in
// `src/encoder/m2/m2_syntax.cpp` remains the only syntax authority.

#pragma once

#include "encoder/m2/rdo/config.h"
#include "encoder/m2/rdo/context.h"
#include "encoder/m2/rdo/inter_decision.h"
#include "encoder/m2/rdo/intra_fallback.h"
#include "encoder/m2/rdo/partition_decision.h"
#include "encoder/m2/rdo/rate_model.h"
#include "encoder/m2/rdo/residual_decision.h"
