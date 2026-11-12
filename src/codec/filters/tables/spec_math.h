// §4.8 integer math shared by the recovered-table CPU references.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7).
//
// These helpers are deliberately integer-only so the CPU reference and any
// future device port produce identical results.

#ifndef AV2ENC_ENCODER_FILTERS_SPEC_TABLES_SPEC_MATH_H_
#define AV2ENC_ENCODER_FILTERS_SPEC_TABLES_SPEC_MATH_H_

#include <cstdint>

namespace av2enc::filters::spec {

// §4.8 Clip3(x, y, z).
constexpr int Clip3(int x, int y, int z) { return z < x ? x : (z > y ? y : z); }

// §4.8 Round2(x, n) = floor((x + 2^(n-1)) / 2^n).  An arithmetic right shift
// implements floor division for the signed values used by the filters.
constexpr std::int64_t Round2(std::int64_t x, int n) {
  return n == 0 ? x : (x + (std::int64_t{1} << (n - 1))) >> n;
}

// §4.8 Round2Signed(x, n).
constexpr std::int64_t Round2Signed(std::int64_t x, int n) {
  return x >= 0 ? Round2(x, n) : -Round2(-x, n);
}

constexpr int Abs(int x) { return x < 0 ? -x : x; }

// §4.8 Clip1(x) for the supplied bit depth.
constexpr int Clip1(int bit_depth, int x) {
  return Clip3(0, (1 << bit_depth) - 1, x);
}

// §7.14.2 MaxQ.  Only 8- and 10-bit streams are signalable in v1.0.0; 12-bit
// is kept internal for the CPU reference.
constexpr int MaxQ(int bit_depth) {
  return bit_depth == 8 ? 255 : (bit_depth == 10 ? 303 : 351);
}

}  // namespace av2enc::filters::spec

#endif  // AV2ENC_ENCODER_FILTERS_SPEC_TABLES_SPEC_MATH_H_
