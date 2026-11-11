// AV2 v1.0.0 §7.13.2.14 — filter corner process.
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §7.13.2.14, p. 463, invoked from §7.13.2.7:
//
//   filter_corner() {
//     s = LeftCol[0] * 5 + AboveRow[-1] * 6 + AboveRow[0] * 5
//     return Round2(s, 4)              // i.e. (s + 8) >> 4
//   }
//   LeftCol[-1] = filter_corner()
//   AboveRow[-1] = LeftCol[-1]
//
// The §7.13.2.7 guard is:
//   ( applyIbp || (pAngle > 90 && pAngle < 180) ) && (w + h) >= 24
//
// This module implements the non-IBP form (`applyIbp == 0`), which is the
// only form reachable while `enable_ibp == 0`.  The 5/6/5 tap subset is the
// three taps above; the `Round2(..., 4)` shift is exactly `(s + 8) >> 4`.
//
// Production call site: `src/encoder/intra_modes/intra_modes.cpp:130-136`
// (`filter_corner`) and `:719-723` (`predict_intra`).  The reference below is
// byte-identical to it; `missing_bits_test.cpp` proves equality when the rule
// is inactive by pre-applying this reference to an `IntraEdges` copy and
// checking `predict_intra` output is unchanged.

#pragma once

#include <cstdint>

#include "encoder/intra_modes/intra_modes.h"

namespace av2enc::m1::missing_bits {

// §7.13.2.14.  `above_m1` is AboveRow[-1] *before* the corner is overwritten,
// i.e. the top-left sample produced by §7.13.2.1.  Returns Round2(s, 4).
constexpr int filter_corner_value(int left0, int above_m1, int above0) noexcept {
  const int s = left0 * 5 + above_m1 * 6 + above0 * 5;
  return (s + 8) >> 4;
}

// §7.13.2.7 guard for the non-IBP path: the corner is filtered only for
// pAngle in (90, 180) and w + h >= 24.
constexpr bool filter_corner_applies(int w, int h, int p_angle) noexcept {
  return p_angle > 90 && p_angle < 180 && (w + h) >= 24;
}

// Applies the reference value to both corner slots.
void filter_corner(av2enc::intra::EdgeRow& above,
                   av2enc::intra::EdgeRow& left) noexcept;

// Applies the reference when `filter_corner_applies(w, h, p_angle)`.  Returns
// true when the corner was modified.
bool filter_corner_if_applicable(int w, int h, int p_angle,
                                 av2enc::intra::EdgeRow& above,
                                 av2enc::intra::EdgeRow& left) noexcept;

}  // namespace av2enc::m1::missing_bits
