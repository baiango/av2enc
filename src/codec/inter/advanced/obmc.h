#ifndef AV2ENC_INTER_ADVANCED_OBMC_H_
#define AV2ENC_INTER_ADVANCED_OBMC_H_

// Encoder-only OBMC experiment. See obmc_common.h: AV2 v1.0.0 has no OBMC.
//
// The caller supplies the neighbour predictions *for this block's region*
// (i.e. the above neighbour's MV applied to the current block), which is what an
// encoder would generate with a second prediction call.
//
// Usage:
//   adv::ObmcNeighbourPred above{above_pred, stride};
//   adv::ObmcNeighbourPred left{left_pred, stride};
//   adv::obmc_apply(rect, /*overlap=*/4, dst, stride, &above, &left);
//
// Returns false when the inputs are inconsistent. Overlap is clamped to
// [0, min(w, h) - 1].

#include "obmc_common.h"

namespace av2enc::inter::adv {

struct ObmcNeighbourPred {
  const std::uint8_t* pred = nullptr;  // block-sized prediction, origin (0,0)
  int stride = 0;
};

struct ObmcConfig {
  BlockRect rect{};
  int overlap = 4;
  std::uint8_t* dst = nullptr;
  int dst_stride = 0;
};

// Overlap actually applied: clamped to [0, min(rect.w, rect.h) - 1].
AV2ENC_ADV_HD inline int obmc_overlap_for(const ObmcConfig& config) {
  const int min_dim = config.rect.w < config.rect.h ? config.rect.w : config.rect.h;
  const int max_overlap = min_dim - 1;
  return clip3(0, max_overlap < 0 ? 0 : max_overlap, config.overlap);
}

// Applies the above neighbour first, then the left neighbour (sequential, the
// same convention as AV1). `above`/`left` may be null when unavailable.
bool obmc_apply(const ObmcConfig& config, const ObmcNeighbourPred* above,
                const ObmcNeighbourPred* left);

// Total squared difference between an OBMC blend and the plain prediction,
// useful for judging whether the overlap would change anything. Returns -1 on
// invalid input.
std::int64_t obmc_prediction_delta(const ObmcConfig& config, const ObmcNeighbourPred* above,
                                   const ObmcNeighbourPred* left, const std::uint8_t* plain,
                                   int plain_stride);

}  // namespace av2enc::inter::adv

#endif  // AV2ENC_INTER_ADVANCED_OBMC_H_
