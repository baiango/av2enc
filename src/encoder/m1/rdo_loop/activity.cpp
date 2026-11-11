#include "activity.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace av2enc::m1::rdo_loop {

std::uint32_t luma_activity_q8(const std::uint8_t* plane, int stride, int x,
                               int y, int w, int h) noexcept {
  if (plane == nullptr || stride <= 0 || w <= 0 || h <= 0 || x < 0 || y < 0) {
    return 0;
  }
  const std::uint64_t count =
      static_cast<std::uint64_t>(w) * static_cast<std::uint64_t>(h);
  std::uint64_t sum = 0;
  for (int row = 0; row < h; ++row) {
    const std::uint8_t* line =
        plane + static_cast<std::size_t>(y + row) *
                    static_cast<std::size_t>(stride) +
        static_cast<std::size_t>(x);
    for (int col = 0; col < w; ++col) {
      sum += line[col];
    }
  }
  const std::uint32_t mean =
      static_cast<std::uint32_t>((sum + count / 2) / count);

  std::uint64_t deviation = 0;
  for (int row = 0; row < h; ++row) {
    const std::uint8_t* line =
        plane + static_cast<std::size_t>(y + row) *
                    static_cast<std::size_t>(stride) +
        static_cast<std::size_t>(x);
    for (int col = 0; col < w; ++col) {
      const int value = static_cast<int>(line[col]);
      const int diff = value - static_cast<int>(mean);
      deviation += static_cast<std::uint64_t>(diff < 0 ? -diff : diff);
    }
  }
  const std::uint64_t q8 = (deviation * 256U + count / 2) / count;
  return static_cast<std::uint32_t>(
      q8 > static_cast<std::uint64_t>(
               std::numeric_limits<std::uint32_t>::max())
          ? std::numeric_limits<std::uint32_t>::max()
          : q8);
}

std::uint32_t mean_activity_q8(const std::uint32_t* activities,
                               int count) noexcept {
  if (activities == nullptr || count <= 0) {
    return 0;
  }
  std::uint64_t sum = 0;
  for (int i = 0; i < count; ++i) {
    sum += activities[i];
  }
  return static_cast<std::uint32_t>(
      (sum + static_cast<std::uint64_t>(count) / 2U) /
      static_cast<std::uint64_t>(count));
}

}  // namespace av2enc::m1::rdo_loop
