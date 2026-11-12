// Rate-context adapters for the M2 P-frame decision layer.
//
// The default `InterRateModel` uses neutral contexts.  A production
// integration that tracks neighbour modes can fill `RateContext` exactly by
// delegating to `src/encoder/m2/merge/`, which owns the §7.11.2 / §5.20.7
// derivations.  These helpers are thin adapters; they do not reimplement the
// context math.

#pragma once

#include "codec/inter/merge/merge.h"
#include "codec/rate/rdo/video/rate_model.h"

namespace av2enc::video::rdo {

// TileSkipCdf context from the two decoded neighbours (§5.20.5.10).  The M2
// baseline has skip_mode = 0.
int skip_context_from_neighbours(const merge::NeighborModes& neighbours,
                                 bool skip_mode = false);

// §7.11.2 NewMvContext via merge::derive_new_mv_context().  Returns false on
// an invalid parameter set (the merge status is not kOk).
bool new_mv_context_from_neighbours(
    const merge::NewMvContextParams& params, int* ctx);

// Bundles the contexts the rate model needs.  `is_inter_ctx` stays a caller
// argument because TileIsInterCdf selection is not owned by the merge layer.
// `neighbours` supplies the skip context; `new_mv` supplies the single-mode /
// DRL context.
RateContext make_rate_context(
    int is_inter_ctx, const merge::NeighborModes& neighbours,
    const merge::NewMvContextParams& new_mv,
    int max_drl_bits_minus_1 = 4);

}  // namespace av2enc::video::rdo
