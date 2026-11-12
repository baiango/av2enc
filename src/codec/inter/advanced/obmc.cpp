#include "obmc.h"

#include <cstring>

namespace av2enc::inter::adv {
namespace {

bool config_ok(const ObmcConfig& c) {
  return c.dst != nullptr && c.dst_stride > 0 && c.rect.w > 0 && c.rect.h > 0;
}

bool neighbour_ok(const ObmcNeighbourPred* n) {
  return n == nullptr || (n->pred != nullptr && n->stride > 0);
}

}  // namespace

bool obmc_apply(const ObmcConfig& config, const ObmcNeighbourPred* above,
                const ObmcNeighbourPred* left) {
  if (!config_ok(config) || !neighbour_ok(above) || !neighbour_ok(left)) {
    return false;
  }
  const int overlap = obmc_overlap_for(config);
  if (overlap == 0) {
    return true;
  }

  if (above != nullptr) {
    for (int i = 0; i < overlap; ++i) {
      const int w = obmc_weight(i, overlap);
      std::uint8_t* row = config.dst +
                          static_cast<std::ptrdiff_t>(config.rect.y + i) * config.dst_stride +
                          config.rect.x;
      const std::uint8_t* nb = above->pred + static_cast<std::ptrdiff_t>(i) * above->stride;
      for (int j = 0; j < config.rect.w; ++j) {
        row[j] = static_cast<std::uint8_t>(obmc_blend_sample(row[j], nb[j], w));
      }
    }
  }

  if (left != nullptr) {
    for (int i = 0; i < config.rect.h; ++i) {
      std::uint8_t* row = config.dst +
                          static_cast<std::ptrdiff_t>(config.rect.y + i) * config.dst_stride +
                          config.rect.x;
      const std::uint8_t* nb = left->pred + static_cast<std::ptrdiff_t>(i) * left->stride;
      for (int j = 0; j < overlap; ++j) {
        const int w = obmc_weight(j, overlap);
        row[j] = static_cast<std::uint8_t>(obmc_blend_sample(row[j], nb[j], w));
      }
    }
  }
  return true;
}

std::int64_t obmc_prediction_delta(const ObmcConfig& config, const ObmcNeighbourPred* /*above*/,
                                   const ObmcNeighbourPred* /*left*/, const std::uint8_t* plain,
                                   int plain_stride) {
  if (!config_ok(config) || plain == nullptr || plain_stride <= 0) {
    return -1;
  }
  std::int64_t delta = 0;
  for (int i = 0; i < config.rect.h; ++i) {
    const std::uint8_t* row = config.dst +
                              static_cast<std::ptrdiff_t>(config.rect.y + i) * config.dst_stride +
                              config.rect.x;
    const std::uint8_t* ref = plain + static_cast<std::ptrdiff_t>(i) * plain_stride;
    for (int j = 0; j < config.rect.w; ++j) {
      const int d = static_cast<int>(row[j]) - static_cast<int>(ref[j]);
      delta += static_cast<std::int64_t>(d * d);
    }
  }
  return delta;
}

}  // namespace av2enc::inter::adv
