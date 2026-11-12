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
// `src/encoder/m2/video_syntax.cpp` remains the only syntax authority.

#pragma once

#include "codec/rate/rdo/video/config.h"
#include "codec/rate/rdo/video/context.h"
#include "codec/rate/rdo/video/inter_decision.h"
#include "codec/rate/rdo/video/intra_fallback.h"
#include "codec/rate/rdo/video/partition_decision.h"
#include "codec/rate/rdo/video/rate_model.h"
#include "codec/rate/rdo/video/residual_decision.h"
