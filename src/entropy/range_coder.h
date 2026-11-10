// Adaptive multi-symbol range coder.
//
// This is a from-scratch, self-contained arithmetic coding foundation for the
// AV2 encoder.  It deliberately contains no AV2 syntax; it provides the
// primitive that a CDF adaptation layer can be built on.
//
// IMPORTANT: this is a generic 32-bit range coder, not the normative AV2
// v1.0.0 §8.2 symbol decoder.  AV2's decoder is a 15-bit, bit-oriented
// SymbolRange/SymbolValue coder with inverse CDFs and trailing-bit rules.  See
// README.md ("What must change for AV2") before using this module to produce a
// conformant bitstream.
//
// Design choice
// -------------
// 32-bit range, 8-bit (byte) renormalization, 15-bit probability precision
// (total frequency 32768).  15 bits is the natural precision for AV2-style CDF
// tables and keeps the coder's interval arithmetic comfortably inside 32 bits:
// after renormalization range >= 2^24, so even a symbol with frequency 1 gets
// an interval of at least 512.
//
// The encoder emits bytes as soon as their value is final, using a one-byte
// cache and a count of pending 0xFF bytes so that carries from future symbols
// can be propagated into already-emitted bytes.  This avoids any global state
// and keeps the output a simple byte string.
//
// See README.md in this directory for the exact bitstream format, the model
// invariant list, and the list of things that must change to match the
// normative AV2 entropy coder.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace av2enc::entropy {

// Probability precision.  CDFs are fixed-point values in [0, kProbTotal].
inline constexpr uint32_t kProbBits = 15;
inline constexpr uint32_t kProbTotal = 1u << kProbBits;  // 32768

// Largest alphabet accepted by this foundation.
inline constexpr uint32_t kMaxAlphabetSize = 16;

// Default adaptation shift.  rate == 0 disables adaptation; larger rates adapt
// more slowly.  See AdaptiveCdf::update().
inline constexpr uint32_t kDefaultAdaptationRate = 5;
inline constexpr uint32_t kMaxAdaptationRate = 15;

enum class RangeCoderStatus : uint8_t {
  kOk = 0,
  kInvalidArgument,          // null pointer or otherwise unusable argument
  kInvalidSymbol,            // symbol >= alphabet size
  kEncoderFinished,          // encode()/finish() after finish()
  kDecoderNotInitialized,    // decode() before init()
  kDecoderAlreadyInitialized,// init() called twice on one stream
  kDecoderFailed,            // decoder previously hit a fatal error
  kTruncatedInput,           // stream ended in init() or renormalization
};

const char* status_string(RangeCoderStatus status);

// Adaptive cumulative distribution over 1..kMaxAlphabetSize symbols.
//
// Representation: ascending cumulative frequencies in [0, kProbTotal].
//
//   cdf[0] == 0
//   cdf[n] == kProbTotal
//   cdf[i] < cdf[i+1]                     (every symbol has frequency >= 1)
//   frequency(i) == cdf[i+1] - cdf[i]
//
// Coding symbol s uses the half-open interval [cdf[s], cdf[s+1]).
//
// Adaptation (rate r in [1, 15], applied *after* a symbol is coded):
//
//   freq[s] += max(1, kProbTotal >> r)
//
// followed by an exact renormalization of all frequencies back to kProbTotal.
// The renormalization reserves one unit per symbol (so no symbol can acquire
// zero probability), then apportions the remaining kProbTotal - n units in
// proportion to the updated frequencies using cumulative rounding.  The
// resulting CDF is exact: no rounding drift accumulates.
//
// rate == 0 freezes the model (useful for static AV2 CDFs).  reset() restores
// the distribution the model was constructed with.
class AdaptiveCdf {
 public:
  using CdfArray = std::array<uint16_t, kMaxAlphabetSize + 1>;

  // Uniform distribution over num_symbols.  The remainders from integer
  // division are spread deterministically; the sum is always exactly
  // kProbTotal.
  static std::optional<AdaptiveCdf> uniform(
      uint32_t num_symbols,
      uint32_t adaptation_rate = kDefaultAdaptationRate);

  // Initial distribution proportional to |frequencies|.  Every frequency
  // must be > 0.  Performs exact apportionment to kProbTotal using cumulative
  // rounding (one unit is reserved per symbol first).
  static std::optional<AdaptiveCdf> from_frequencies(
      std::span<const uint32_t> frequencies,
      uint32_t adaptation_rate = kDefaultAdaptationRate);

  // Initial distribution copied from an explicit CDF.  |cumulative| must have
  // n + 1 entries (n in [1, kMaxAlphabetSize]) and satisfy the invariants
  // above.  This is the hook for AV2's normative CDF tables.
  static std::optional<AdaptiveCdf> from_cumulative(
      std::span<const uint16_t> cumulative,
      uint32_t adaptation_rate = kDefaultAdaptationRate);

  uint32_t num_symbols() const noexcept { return num_symbols_; }
  uint32_t adaptation_rate() const noexcept { return adaptation_rate_; }

  // Pointer to num_symbols() + 1 ascending CDF entries.
  const uint16_t* cumulative() const noexcept { return cdf_.data(); }

  // Precondition for the three accessors below: symbol < num_symbols().
  uint32_t low(uint32_t symbol) const noexcept { return cdf_[symbol]; }
  uint32_t high(uint32_t symbol) const noexcept { return cdf_[symbol + 1]; }
  uint32_t frequency(uint32_t symbol) const noexcept {
    return static_cast<uint32_t>(cdf_[symbol + 1]) - cdf_[symbol];
  }

  // Returns false if |rate| > kMaxAdaptationRate.
  bool set_adaptation_rate(uint32_t rate) noexcept;

  // Adapt the model after observing |symbol|.  Returns false if the symbol is
  // outside the alphabet.  A failed update leaves the model unchanged.
  bool update(uint32_t symbol) noexcept;

  // Restore the construction-time distribution.  The adaptation rate is kept.
  void reset() noexcept { cdf_ = initial_cdf_; }

 private:
  AdaptiveCdf() = default;

  static bool valid_cumulative(std::span<const uint16_t> cumulative) noexcept;
  static CdfArray apportion(
      const std::array<uint32_t, kMaxAlphabetSize>& weights,
      uint32_t num_symbols) noexcept;

  CdfArray initial_cdf_{};
  CdfArray cdf_{};
  uint32_t num_symbols_ = 0;
  uint32_t adaptation_rate_ = kDefaultAdaptationRate;
};

// Range encoder.  One instance encodes one contiguous sequence of symbols.
//
//   Encoder enc;
//   st = enc.encode(symbol, model);   // model is updated on success
//   st = enc.finish();
//   const std::vector<uint8_t>& bytes = enc.bitstream();
//
// The model passed to encode() must be the same object (same state) that the
// decoder will use, in the same order.  encode() advances the model itself so
// the two sides cannot drift accidentally.
class Encoder {
 public:
  Encoder() = default;

  // Codes |symbol| with the current CDF of |model| and then adapts |model|.
  // Returns kInvalidSymbol (state unchanged) if symbol >= model.num_symbols().
  RangeCoderStatus encode(uint32_t symbol, AdaptiveCdf& model);

  // Emits the final bytes (see README.md; a finished stream has
  // 5 + renormalization-count bytes).  Must be called exactly once before
  // bitstream() is considered complete.
  RangeCoderStatus finish();

  bool finished() const noexcept { return finished_; }

  // Bytes produced so far.  Complete only after finish() returns kOk.
  const std::vector<uint8_t>& bitstream() const noexcept { return out_; }

  // Discard all state and output.
  void reset();

 private:
  void shift_low();

  uint64_t low_ = 0;
  uint32_t range_ = 0xFFFFFFFFu;
  uint8_t cache_ = 0;
  uint64_t cache_size_ = 1;
  std::vector<uint8_t> out_;
  bool finished_ = false;
};

// Range decoder.  The caller supplies the complete byte range (for AV2 this is
// one tile payload) and decodes exactly as many symbols as were encoded.
//
// The stream is intentionally not self-delimiting: an arithmetic coder cannot
// know how many symbols the encoder wrote.  Callers must know the symbol count
// or use an end-of-sequence symbol in the syntax layer.  When exactly that
// count is decoded, a valid stream is consumed completely
// (bytes_remaining() == 0); truncation is reported as kTruncatedInput.
class Decoder {
 public:
  Decoder() = default;
  explicit Decoder(std::span<const uint8_t> data) noexcept : data_(data) {}

  // Reads the 5-byte initial code.  Fails with kTruncatedInput if fewer than
  // 5 bytes are available.
  RangeCoderStatus init();

  // Decodes one symbol into |symbol| and adapts |model| exactly as the encoder
  // did.  On a fatal error the decoder is poisoned and later calls return
  // kDecoderFailed (except kInvalidArgument, which is checked first).
  RangeCoderStatus decode(AdaptiveCdf& model, uint32_t* symbol);

  bool initialized() const noexcept { return initialized_; }
  bool failed() const noexcept { return failed_; }

  size_t bytes_consumed() const noexcept { return pos_; }
  size_t bytes_remaining() const noexcept { return data_.size() - pos_; }
  std::span<const uint8_t> data() const noexcept { return data_; }

  // Rebind to a new byte range.  Clears all decoder state.
  void reset(std::span<const uint8_t> data) noexcept;

 private:
  RangeCoderStatus renormalize();

  std::span<const uint8_t> data_{};
  size_t pos_ = 0;
  uint32_t range_ = 0xFFFFFFFFu;
  uint32_t code_ = 0;
  bool initialized_ = false;
  bool failed_ = false;
};

}  // namespace av2enc::entropy
