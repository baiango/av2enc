#ifndef AV2ENC_INTER_ADVANCED_OBMC_COMMON_H_
#define AV2ENC_INTER_ADVANCED_OBMC_COMMON_H_

// Overlapped block motion compensation — **not part of AV2 v1.0.0**.
//
// AV2 removed OBMC as a motion mode: `MOTION_MODES = 5` is
// SIMPLE/INTERINTRA/LOCALWARP/DELTAWARP/EXTENDWARP, and there is no OBMC
// predictor in §7.13.3. The closest normative tools are local warp (which
// models the same sub-block motion variation with a smooth affine field) and
// inter-intra for the occlusion case.
//
// This file exists because the wave-2 brief asked for an OBMC component and
// because an encoder-side "what would overlapped MC buy us" experiment is
// useful when deciding how much to invest in local warp. It must never be
// enabled in a conforming stream: doing so would desynchronise the decoder.
// It is deliberately isolated from the other advanced-inter components and is
// tested only for deterministic behaviour.
//
// The blend is the usual AV1-style ramp: the neighbour prediction owns the
// samples closest to the shared edge and fades linearly into the block's own
// prediction over `overlap` samples.

#include "advanced_common.h"

namespace av2enc::inter::adv {

// Neighbour weight in 1/64 units for sample `d` (0 = shared edge).
// w(d) = 64 * (overlap - d) / (overlap + 1), clipped to [0, 64].
AV2ENC_ADV_HD inline int obmc_weight(int d, int overlap) {
  if (overlap <= 0 || d < 0 || d >= overlap) {
    return 0;
  }
  return (64 * (overlap - d)) / (overlap + 1);
}

// out = Round2((64 - w) * own + w * neighbour, 6)
AV2ENC_ADV_HD inline int obmc_blend_sample(int own, int neighbour, int w) {
  return clip1(round2((64 - w) * own + w * neighbour, 6), 8);
}

// Sequential above-then-left blend for one sample. `above`/`left` are the
// neighbour predictions for this sample; unavailable neighbours are ignored by
// passing the same band limit for both (overlap == 0 skips everything).
AV2ENC_ADV_HD inline int obmc_sample_at(int own, int above, int left, int i, int j, int overlap) {
  int value = own;
  if (i < overlap) {
    value = obmc_blend_sample(value, above, obmc_weight(i, overlap));
  }
  if (j < overlap) {
    value = obmc_blend_sample(value, left, obmc_weight(j, overlap));
  }
  return value;
}

}  // namespace av2enc::inter::adv

#endif  // AV2ENC_INTER_ADVANCED_OBMC_COMMON_H_
