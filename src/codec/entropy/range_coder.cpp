#include "range_coder.h"

#include <algorithm>
#include <cassert>

namespace av2enc::entropy {
namespace {

// Renormalize whenever the range drops below 2^24.  This guarantees that after
// dividing by kProbTotal (2^15) the per-unit range is at least 2^9, so no
// symbol interval can collapse even when its frequency is 1.
constexpr uint32_t kTopValue = 1u << 24;

// finish() performs five byte shifts.  The total stream is R + 5 bytes, where
// R is the number of renormalization shifts performed while coding; the five
// shifts flush the 32-bit final low plus the leading carry-guard byte,
// including any 0xFF bytes held back for carry propagation.  The decoder
// consumes five bytes during init (the guard byte is shifted out of the
// 32-bit code word).
constexpr int kFlushByteCount = 5;

static_assert(kProbTotal <= 0xFFFFu,
              "CDF entries are stored as uint16_t");
static_assert(kMaxAlphabetSize < kProbTotal,
              "one frequency unit per symbol must fit in kProbTotal");
static_assert(kProbTotal <= kTopValue,
              "renormalization threshold must leave room for the total");

}  // namespace

const char* status_string(RangeCoderStatus status) {
  switch (status) {
    case RangeCoderStatus::kOk:
      return "ok";
    case RangeCoderStatus::kInvalidArgument:
      return "invalid argument";
    case RangeCoderStatus::kInvalidSymbol:
      return "invalid symbol";
    case RangeCoderStatus::kEncoderFinished:
      return "encoder already finished";
    case RangeCoderStatus::kDecoderNotInitialized:
      return "decoder not initialized";
    case RangeCoderStatus::kDecoderAlreadyInitialized:
      return "decoder already initialized";
    case RangeCoderStatus::kDecoderFailed:
      return "decoder previously failed";
    case RangeCoderStatus::kTruncatedInput:
      return "truncated input";
  }
  return "unknown status";
}

// ---------------------------------------------------------------------------
// AdaptiveCdf
// ---------------------------------------------------------------------------

bool AdaptiveCdf::valid_cumulative(
    std::span<const uint16_t> cumulative) noexcept {
  if (cumulative.size() < 2 ||
      cumulative.size() > kMaxAlphabetSize + 1) {
    return false;
  }
  if (cumulative.front() != 0 ||
      static_cast<uint32_t>(cumulative.back()) != kProbTotal) {
    return false;
  }
  for (size_t i = 1; i < cumulative.size(); ++i) {
    if (cumulative[i] <= cumulative[i - 1]) return false;
  }
  return true;
}

AdaptiveCdf::CdfArray AdaptiveCdf::apportion(
    const std::array<uint32_t, kMaxAlphabetSize>& weights,
    uint32_t num_symbols) noexcept {
  CdfArray cdf{};
  if (num_symbols == 0) return cdf;
  if (num_symbols == 1) {
    cdf[1] = static_cast<uint16_t>(kProbTotal);
    return cdf;
  }

  uint64_t total = 0;
  for (uint32_t i = 0; i < num_symbols; ++i) {
    total += weights[i];
  }
  assert(total > 0);

  // Reserve one frequency unit per symbol.  Spread the remaining units in
  // proportion to the weights with cumulative rounding.  Because the
  // accumulator carries the exact remainder into the next symbol, the shares
  // sum to kProbTotal - num_symbols exactly; the final CDF is exactly
  // kProbTotal, with no accumulated drift.
  const uint32_t available = kProbTotal - num_symbols;
  uint64_t accumulator = 0;
  uint32_t cumulative = 0;
  for (uint32_t i = 0; i < num_symbols; ++i) {
    accumulator += static_cast<uint64_t>(available) * weights[i];
    const uint32_t share = static_cast<uint32_t>(accumulator / total);
    accumulator %= total;
    cumulative += 1 + share;
    cdf[i + 1] = static_cast<uint16_t>(cumulative);
  }
  assert(cumulative == kProbTotal);
  return cdf;
}

std::optional<AdaptiveCdf> AdaptiveCdf::uniform(uint32_t num_symbols,
                                                uint32_t adaptation_rate) {
  if (num_symbols == 0 || num_symbols > kMaxAlphabetSize ||
      adaptation_rate > kMaxAdaptationRate) {
    return std::nullopt;
  }

  std::array<uint32_t, kMaxAlphabetSize> weights{};
  for (uint32_t i = 0; i < num_symbols; ++i) weights[i] = 1;

  AdaptiveCdf model;
  model.num_symbols_ = num_symbols;
  model.adaptation_rate_ = adaptation_rate;
  model.initial_cdf_ = apportion(weights, num_symbols);
  model.cdf_ = model.initial_cdf_;
  return model;
}

std::optional<AdaptiveCdf> AdaptiveCdf::from_frequencies(
    std::span<const uint32_t> frequencies, uint32_t adaptation_rate) {
  if (frequencies.empty() || frequencies.size() > kMaxAlphabetSize ||
      adaptation_rate > kMaxAdaptationRate) {
    return std::nullopt;
  }

  std::array<uint32_t, kMaxAlphabetSize> weights{};
  for (size_t i = 0; i < frequencies.size(); ++i) {
    if (frequencies[i] == 0) return std::nullopt;
    weights[i] = frequencies[i];
  }

  AdaptiveCdf model;
  model.num_symbols_ = static_cast<uint32_t>(frequencies.size());
  model.adaptation_rate_ = adaptation_rate;
  model.initial_cdf_ = apportion(weights, model.num_symbols_);
  model.cdf_ = model.initial_cdf_;
  return model;
}

std::optional<AdaptiveCdf> AdaptiveCdf::from_cumulative(
    std::span<const uint16_t> cumulative, uint32_t adaptation_rate) {
  if (adaptation_rate > kMaxAdaptationRate ||
      !valid_cumulative(cumulative)) {
    return std::nullopt;
  }

  AdaptiveCdf model;
  model.num_symbols_ = static_cast<uint32_t>(cumulative.size() - 1);
  model.adaptation_rate_ = adaptation_rate;
  for (size_t i = 0; i < cumulative.size(); ++i) {
    model.initial_cdf_[i] = cumulative[i];
  }
  model.cdf_ = model.initial_cdf_;
  return model;
}

bool AdaptiveCdf::set_adaptation_rate(uint32_t rate) noexcept {
  if (rate > kMaxAdaptationRate) return false;
  adaptation_rate_ = rate;
  return true;
}

bool AdaptiveCdf::update(uint32_t symbol) noexcept {
  if (symbol >= num_symbols_) return false;
  if (num_symbols_ < 2 || adaptation_rate_ == 0) return true;

  std::array<uint32_t, kMaxAlphabetSize> weights{};
  for (uint32_t i = 0; i < num_symbols_; ++i) {
    weights[i] = frequency(i);
  }

  const uint32_t increment =
      std::max<uint32_t>(1, kProbTotal >> adaptation_rate_);
  weights[symbol] += increment;

  cdf_ = apportion(weights, num_symbols_);
  return true;
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

void Encoder::shift_low() {
  const uint32_t low32 = static_cast<uint32_t>(low_);
  const uint32_t carry = static_cast<uint32_t>(low_ >> 32);

  // A top byte of 0xFF is held back because a future carry would turn it into
  // 0x00 and propagate further.  cache_size_ counts that cache byte plus all
  // pending 0xFF bytes.  The carry is added to every pending byte: the cache
  // byte is at most 0xFE, so it absorbs the +1, and the pending 0xFF bytes
  // wrap to 0x00 exactly as a binary carry would.
  if (low32 < 0xFF000000u || carry != 0) {
    const uint8_t carry_byte = static_cast<uint8_t>(carry);
    uint8_t byte = cache_;
    do {
      out_.push_back(static_cast<uint8_t>(byte + carry_byte));
      byte = 0xFF;
    } while (--cache_size_ != 0);
    cache_ = static_cast<uint8_t>(low32 >> 24);
  }

  ++cache_size_;
  low_ = static_cast<uint64_t>(static_cast<uint32_t>(low32 << 8));
}

RangeCoderStatus Encoder::encode(uint32_t symbol, AdaptiveCdf& model) {
  if (finished_) return RangeCoderStatus::kEncoderFinished;

  const uint32_t num_symbols = model.num_symbols();
  if (symbol >= num_symbols) return RangeCoderStatus::kInvalidSymbol;

  // A one-symbol alphabet carries no information and consumes no range.
  if (num_symbols == 1) {
    model.update(symbol);
    return RangeCoderStatus::kOk;
  }

  const uint16_t* cdf = model.cumulative();
  const uint32_t freq_low = cdf[symbol];
  const uint32_t width =
      static_cast<uint32_t>(cdf[symbol + 1]) - freq_low;

  const uint32_t unit = range_ / kProbTotal;
  assert(unit > 0);
  low_ += static_cast<uint64_t>(unit) * freq_low;
  range_ = unit * width;

  while (range_ < kTopValue) {
    shift_low();
    range_ <<= 8;
  }

  model.update(symbol);
  return RangeCoderStatus::kOk;
}

RangeCoderStatus Encoder::finish() {
  if (finished_) return RangeCoderStatus::kEncoderFinished;

  for (int i = 0; i < kFlushByteCount; ++i) {
    shift_low();
  }
  finished_ = true;
  return RangeCoderStatus::kOk;
}

void Encoder::reset() {
  low_ = 0;
  range_ = 0xFFFFFFFFu;
  cache_ = 0;
  cache_size_ = 1;
  out_.clear();
  finished_ = false;
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

RangeCoderStatus Decoder::init() {
  if (initialized_) return RangeCoderStatus::kDecoderAlreadyInitialized;
  if (data_.size() < static_cast<size_t>(kFlushByteCount)) {
    return RangeCoderStatus::kTruncatedInput;
  }

  range_ = 0xFFFFFFFFu;
  code_ = 0;
  pos_ = 0;
  // Five bytes: the leading carry-guard byte is shifted out of the 32-bit
  // code word, leaving the four bytes that seed the code value.
  for (int i = 0; i < kFlushByteCount; ++i) {
    code_ = (code_ << 8) | data_[pos_++];
  }
  initialized_ = true;
  failed_ = false;
  return RangeCoderStatus::kOk;
}

RangeCoderStatus Decoder::renormalize() {
  while (range_ < kTopValue) {
    if (pos_ >= data_.size()) return RangeCoderStatus::kTruncatedInput;
    code_ = (code_ << 8) | data_[pos_++];
    range_ <<= 8;
  }
  return RangeCoderStatus::kOk;
}

RangeCoderStatus Decoder::decode(AdaptiveCdf& model, uint32_t* symbol) {
  if (symbol == nullptr) return RangeCoderStatus::kInvalidArgument;
  if (!initialized_) return RangeCoderStatus::kDecoderNotInitialized;
  if (failed_) return RangeCoderStatus::kDecoderFailed;

  const uint32_t num_symbols = model.num_symbols();
  if (num_symbols == 1) {
    *symbol = 0;
    model.update(0);
    return RangeCoderStatus::kOk;
  }

  const uint32_t unit = range_ / kProbTotal;
  assert(unit > 0);

  // |value| may exceed the probability total when the code word sits at the
  // very top of the range; clamping is the standard range-decoder rule.
  uint32_t value = code_ / unit;
  if (value >= kProbTotal) value = kProbTotal - 1;

  const uint16_t* cdf = model.cumulative();
  uint32_t decoded = 0;
  while (decoded + 1 < num_symbols && cdf[decoded + 1] <= value) {
    ++decoded;
  }

  const uint32_t freq_low = cdf[decoded];
  const uint32_t width =
      static_cast<uint32_t>(cdf[decoded + 1]) - freq_low;
  assert(width > 0);

  code_ -= unit * freq_low;
  range_ = unit * width;

  const RangeCoderStatus status = renormalize();
  if (status != RangeCoderStatus::kOk) {
    failed_ = true;
    return status;
  }

  model.update(decoded);
  *symbol = decoded;
  return RangeCoderStatus::kOk;
}

void Decoder::reset(std::span<const uint8_t> data) noexcept {
  data_ = data;
  pos_ = 0;
  range_ = 0xFFFFFFFFu;
  code_ = 0;
  initialized_ = false;
  failed_ = false;
}

}  // namespace av2enc::entropy
