// Normative AV2 v1.0.0 §8.2 symbol entropy coder — host-only helpers.
//
// The coder state machine itself lives in av2_entropy.h as AV2_ENTROPY_HD
// inline functions so the same code can be compiled for CUDA later.  This
// translation unit only contains:
//   * status_name()             (host diagnostics)
//   * copy_cdf()/average_cdf()  (§8.2.4 tile-group CDF bookkeeping, once per
//                                tile group; not in the per-symbol hot path)
//
// No allocations, no globals, no exceptions, no macOS-specific code.

#include "av2_entropy.h"

#include <cstring>

namespace av2enc::av2 {

const char* status_name(Status status) noexcept {
  switch (status) {
    case Status::kOk:
      return "ok";
    case Status::kInvalidArgument:
      return "invalid argument";
    case Status::kInvalidCdf:
      return "invalid CDF";
    case Status::kInvalidSymbol:
      return "invalid symbol";
    case Status::kInvalidState:
      return "invalid coder state";
    case Status::kBufferTooSmall:
      return "output buffer too small";
    case Status::kTruncatedInput:
      return "truncated input";
    case Status::kInvalidTrailingBits:
      return "invalid trailing bits";
    case Status::kSymbolMaxBitsUnderflow:
      return "SymbolMaxBits < -14";
    case Status::kAlreadyFinished:
      return "already finished";
    case Status::kEncoderFailed:
      return "encoder failed";
    case Status::kDecoderFailed:
      return "decoder failed";
  }
  return "unknown status";
}

Status copy_cdf(uint16_t* dst, const uint16_t* src, int n) noexcept {
  if (dst == nullptr || src == nullptr || n < 2 || n > kMaxSymbols) {
    return Status::kInvalidArgument;
  }
  std::memcpy(dst, src, static_cast<size_t>(n + 1) * sizeof(uint16_t));
  return Status::kOk;
}

Status average_cdf(uint16_t* cdf, const uint16_t* tile_cdf, int n,
                   uint32_t num_log2, uint32_t tile_num) noexcept {
  if (cdf == nullptr || tile_cdf == nullptr || n < 2 || n > kMaxSymbols) {
    return Status::kInvalidArgument;
  }
  if (num_log2 > 3u) {
    return Status::kInvalidArgument;
  }

  // avg_cdf(cdf, tilecdf, sz) with sz = n + 1.
  if (tile_num == 0u) {
    for (int i = 0; i < n - 1; ++i) {
      cdf[i] = static_cast<uint16_t>(kCdfScale);
    }
    cdf[n - 1] = tile_cdf[n - 1];
    cdf[n] = 0;
  }
  for (int i = 0; i < n - 1; ++i) {
    const uint32_t value = cdf[i];
    const uint32_t tile_value = tile_cdf[i];
    cdf[i] = static_cast<uint16_t>(
        value - ((kCdfScale - tile_value) >> num_log2));
  }
  cdf[n] = static_cast<uint16_t>(cdf[n] + (tile_cdf[n] >> num_log2));
  return Status::kOk;
}

}  // namespace av2enc::av2
