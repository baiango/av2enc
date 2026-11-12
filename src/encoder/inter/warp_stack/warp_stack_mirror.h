#ifndef AV2ENC_INTER_WARP_STACK_WARP_STACK_MIRROR_H_
#define AV2ENC_INTER_WARP_STACK_WARP_STACK_MIRROR_H_

// Independent re-derivation of the warp machinery, used only by the tests.
//
// `warp_stack_mirror.cpp` is a second transcription of §7.11.4, §7.12.2.3/.4/
// .6/.9/.11/.12/.19/.20/.23 and §7.12.3 with a different internal structure:
//
//   * the primary builder is a fixed-array / callback-driven transcription of
//     the spec's in-place state machine; the mirror is a vector-based
//     "collect candidate events, then fold" implementation;
//   * the mirror never calls the primary builder, the primary scan helpers or
//     the primary corner/sample routines.  It shares only the public data
//     types and the `adv` arithmetic primitives.
//
// Randomized lockstep parity between the two is the evidence that the warp
// state machine is not accidentally correct for one particular wiring.  The
// gate uses the primary builder; the mirror is not compiled into the encoder.

#include "encoder/inter/warp_stack/warp_stack.h"

namespace av2enc::inter::warp::mirror {

Status build_stack(const WarpStackParams& params, WarpStack* stack);

bool warp_samples(const WarpSampleParams& params, WarpSampleSet* out);

}  // namespace av2enc::inter::warp::mirror

#endif  // AV2ENC_INTER_WARP_STACK_WARP_STACK_MIRROR_H_
