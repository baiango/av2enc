// AV2 v1.0.0 §7.13.2.11 — DC intra prediction subsampled process.

#include "encoder/m1/missing_bits/dc_intra_subsampled.h"

#include <algorithm>
#include <cstdlib>

namespace av2enc::m1::missing_bits {
namespace {

constexpr int kDivLutBits = 7;      // DIV_LUT_BITS
constexpr int kDivLutPrecBits = 9;  // DIV_LUT_PREC_BITS

int round2(int value, int shift) noexcept {
  if (shift <= 0) {
    return value;
  }
  return (value + (1 << (shift - 1))) >> shift;
}

int floor_log2(int value) noexcept {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// §7.13.3.22 resolve_divisor().  Kept local so this file is a literal
// transcription of the spec rather than a call into the production core.
int resolve_divisor(int divisor) noexcept {
  const int magnitude = std::abs(divisor);
  const int n = floor_log2(magnitude);
  const int e = magnitude - (1 << n);
  const int f =
      n > kDivLutBits ? round2(e, n - kDivLutBits) : (e << (kDivLutBits - n));
  return kDivLut[f] * (divisor < 0 ? -1 : 1);
}

int divisor_shift(int divisor) noexcept {
  return floor_log2(std::abs(divisor)) + kDivLutPrecBits;
}

int clip1(int value, int bit_depth) noexcept {
  const int high = (1 << bit_depth) - 1;
  return value < 0 ? 0 : (value > high ? high : value);
}

bool geometry_ok(const DcSubsampledInput& in) noexcept {
  if (in.w < 1 || in.h < 1 || in.w > 64 || in.h > 64) {
    return false;
  }
  if (in.bit_depth < 8 || in.bit_depth > 12) {
    return false;
  }
  if (in.have_above && in.above == nullptr) {
    return false;
  }
  if (in.have_left && in.left == nullptr) {
    return false;
  }
  return true;
}

}  // namespace

int approx_divide(int numerator, int denominator) noexcept {
  if (denominator == 0) {
    return 0;
  }
  return round2(numerator * resolve_divisor(denominator),
                divisor_shift(denominator));
}

bool dc_intra_subsampled(const DcSubsampledInput& in, std::uint16_t* dst,
                         std::size_t dst_stride,
                         DcSubsampledTrace* trace) noexcept {
  if (dst == nullptr || dst_stride < static_cast<std::size_t>(in.w) ||
      !geometry_ok(in)) {
    return false;
  }

  DcSubsampledTrace local;
  int sum = 0;
  int count = 0;

  if (in.have_left) {
    local.step_h = dc_subsampled_step(in.h);
    for (int k = 0; k < in.h; k += local.step_h) {
      sum += in.left[k];
      ++count;
    }
  }
  if (in.have_above) {
    local.step_w = dc_subsampled_step(in.w);
    for (int k = 0; k < in.w; k += local.step_w) {
      sum += in.above[k];
      ++count;
    }
  }

  int avg = 0;
  if (count == 0) {
    avg = 1 << (in.bit_depth - 1);
  } else {
    avg = clip1(approx_divide(sum, count), in.bit_depth);
  }

  for (int i = 0; i < in.h; ++i) {
    for (int j = 0; j < in.w; ++j) {
      dst[static_cast<std::size_t>(i) * dst_stride +
          static_cast<std::size_t>(j)] = static_cast<std::uint16_t>(avg);
    }
  }

  if (trace != nullptr) {
    local.sum = sum;
    local.count = count;
    local.avg = avg;
    *trace = local;
  }
  return true;
}

bool dc_intra_subsampled(const DcSubsampledInput& in, std::uint16_t* dst,
                         DcSubsampledTrace* trace) noexcept {
  return dc_intra_subsampled(in, dst, static_cast<std::size_t>(in.w), trace);
}

}  // namespace av2enc::m1::missing_bits
