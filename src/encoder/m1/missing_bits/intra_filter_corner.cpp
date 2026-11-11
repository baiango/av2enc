// AV2 v1.0.0 §7.13.2.14 — filter corner process.

#include "encoder/m1/missing_bits/intra_filter_corner.h"

namespace av2enc::m1::missing_bits {

void filter_corner(av2enc::intra::EdgeRow& above,
                   av2enc::intra::EdgeRow& left) noexcept {
  const int value = filter_corner_value(left[0], above[-1], above[0]);
  left[-1] = static_cast<std::int16_t>(value);
  above[-1] = static_cast<std::int16_t>(value);
}

bool filter_corner_if_applicable(int w, int h, int p_angle,
                                 av2enc::intra::EdgeRow& above,
                                 av2enc::intra::EdgeRow& left) noexcept {
  if (!filter_corner_applies(w, h, p_angle)) {
    return false;
  }
  filter_corner(above, left);
  return true;
}

}  // namespace av2enc::m1::missing_bits
