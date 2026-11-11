// AV2 v1.0.0 §7.13.2.11 — DC intra prediction subsampled process.
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §7.13.2.11, pp. 460-461.
//
//   sum = 0; count = 0
//   if (haveLeft) {
//     stepH = h > 32 ? 2 : 1
//     for (k = 0; k < h; k += stepH) { sum += LeftCol[k];  count++ }
//   }
//   if (haveAbove) {
//     stepW = w > 32 ? 2 : 1
//     for (k = 0; k < w; k += stepW) { sum += AboveRow[k]; count++ }
//   }
//   if (count == 0) avg = 1 << (BitDepth - 1)
//   else            avg = Clip1(approx_divide(sum, count))
//
// `approx_divide(num, den)` is the §7.13.2.11 helper:
//   (shift, scale) = resolve_divisor(den); return Round2(num * scale, shift)
//
// This is the process the decoder runs *instead of* §7.13.2.10 plain DC when
// the prediction region exceeds 32 samples on an axis.  The production
// predictor (`src/encoder/intra_modes/intra_modes.cpp` `predict_dc`) only
// implements §7.13.2.10; the two are byte-identical for w <= 32 && h <= 32
// (see missing_bits_test.cpp), and differ exactly when a step is 2.
//
// The reference below is independent of the production predictor: it carries
// its own Div_Lut (§7.13.3.22) and its own rounding helpers.

#pragma once

#include <cstddef>
#include <cstdint>

namespace av2enc::m1::missing_bits {

// DIV_LUT_NUM (§3).  Exposed so the test can pin the table used here.
inline constexpr int kDivLutNum = 129;

// §7.13.3.22 Div_Lut[DIV_LUT_NUM], copied byte-for-byte from the spec.
inline constexpr int kDivLut[kDivLutNum] = {
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465,
    462, 458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 426, 423,
    420, 417, 415, 412, 410, 407, 405, 402, 400, 397, 395, 392, 390, 388,
    386, 383, 381, 379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358,
    356, 354, 352, 350, 349, 347, 345, 343, 341, 340, 338, 336, 334, 333,
    331, 329, 328, 326, 324, 323, 321, 320, 318, 317, 315, 314, 312, 311,
    309, 308, 306, 305, 303, 302, 301, 299, 298, 297, 295, 294, 293, 291,
    290, 289, 287, 286, 285, 284, 282, 281, 280, 279, 278, 277, 275, 274,
    273, 272, 271, 270, 269, 267, 266, 265, 264, 263, 262, 261, 260, 259,
    258, 257, 256};

// Inputs of §7.13.2.11.
//
// `above` points at AboveRow[0..w-1] and `left` at LeftCol[0..h-1] from the
// §7.13.2.1 edge preparation.  Values are 16-bit because the edge arrays are
// (the current 8-bit pipeline only fills 0..255, but the process is
// bit-depth-agnostic and the test covers 8/10/12-bit magnitudes).
struct DcSubsampledInput {
  const std::int16_t* above = nullptr;
  const std::int16_t* left = nullptr;
  int w = 0;
  int h = 0;
  int bit_depth = 8;
  bool have_above = false;
  bool have_left = false;
};

// Optional trace of the process, useful for goldens and for explaining the
// divergence from §7.13.2.10.
struct DcSubsampledTrace {
  int step_w = 0;
  int step_h = 0;
  int sum = 0;
  int count = 0;
  int avg = 0;
};

// stepW / stepH: 2 when the extent is greater than 32, else 1.
constexpr int dc_subsampled_step(int extent) noexcept {
  return extent > 32 ? 2 : 1;
}

// §7.13.2.11 approx_divide() using the Div_Lut above.
int approx_divide(int numerator, int denominator) noexcept;

// Full process.  `dst` is an h x w block with `dst_stride` samples per row.
// The output is 16-bit so 10/12-bit predictions are not truncated; the 8-bit
// caller narrows (the shipped pipeline is 8-bit).
// Returns false only for invalid geometry/inputs; the process itself has no
// failure mode.
bool dc_intra_subsampled(const DcSubsampledInput& in, std::uint16_t* dst,
                         std::size_t dst_stride,
                         DcSubsampledTrace* trace = nullptr) noexcept;

// Convenience overload writing into a tightly packed w*h buffer.
bool dc_intra_subsampled(const DcSubsampledInput& in, std::uint16_t* dst,
                         DcSubsampledTrace* trace = nullptr) noexcept;

}  // namespace av2enc::m1::missing_bits
