// Tests for the standalone adaptive multi-symbol range coder.
//
// Coverage:
//   * round trips over alphabets 2..16, many lengths and adaptation rates
//   * strongly skewed sources and constant sources
//   * one-symbol alphabet (no information, no coded bytes)
//   * >= 1,000,000 symbol round trip
//   * truncation, byte corruption, random garbage, and API misuse
//   * model invariants and encoder/decoder CDF synchronization
//   * a golden byte vector that pins the bitstream format

#include "range_coder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

using namespace av2enc::entropy;

namespace {

int g_checks = 0;
int g_failures = 0;

void check_impl(bool condition, const char* expression, const char* file,
                int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expression);
  }
}

#define CHECK(condition) \
  check_impl((condition), #condition, __FILE__, __LINE__)

#define CHECK_STATUS(expression, expected)                                  \
  do {                                                                      \
    const RangeCoderStatus status_value = (expression);                     \
    ++g_checks;                                                             \
    if (status_value != (expected)) {                                       \
      ++g_failures;                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %s (expected %s)\n",          \
                   __FILE__, __LINE__, #expression,                         \
                   status_string(status_value), status_string(expected));   \
    }                                                                       \
  } while (false)

bool model_is_valid(const AdaptiveCdf& model) {
  const uint32_t n = model.num_symbols();
  if (n < 1 || n > kMaxAlphabetSize) return false;
  const uint16_t* cdf = model.cumulative();
  if (cdf[0] != 0) return false;
  if (static_cast<uint32_t>(cdf[n]) != kProbTotal) return false;
  for (uint32_t i = 0; i < n; ++i) {
    if (cdf[i + 1] <= cdf[i]) return false;
  }
  return true;
}

bool models_equal(const AdaptiveCdf& a, const AdaptiveCdf& b) {
  if (a.num_symbols() != b.num_symbols()) return false;
  if (a.adaptation_rate() != b.adaptation_rate()) return false;
  for (uint32_t i = 0; i <= a.num_symbols(); ++i) {
    if (a.cumulative()[i] != b.cumulative()[i]) return false;
  }
  return true;
}

// Encodes |symbols| with |initial|, decodes them with an independent copy, and
// checks that both final models are identical.  Returns the stream size via
// |stream_size| when non-null.
bool roundtrip(std::span<const uint32_t> symbols, const AdaptiveCdf& initial,
               size_t* stream_size = nullptr) {
  AdaptiveCdf encoder_model = initial;
  AdaptiveCdf decoder_model = initial;

  Encoder encoder;
  for (uint32_t symbol : symbols) {
    if (encoder.encode(symbol, encoder_model) != RangeCoderStatus::kOk) {
      return false;
    }
  }
  if (encoder.finish() != RangeCoderStatus::kOk) return false;

  const std::vector<uint8_t>& stream = encoder.bitstream();
  if (stream_size != nullptr) *stream_size = stream.size();

  Decoder decoder{std::span<const uint8_t>(stream)};
  if (decoder.init() != RangeCoderStatus::kOk) return false;

  for (uint32_t expected : symbols) {
    uint32_t decoded = 0xFFFFFFFFu;
    if (decoder.decode(decoder_model, &decoded) != RangeCoderStatus::kOk) {
      return false;
    }
    if (decoded != expected) return false;
  }

  // A valid stream is consumed exactly by decoding the encoded symbol count.
  if (decoder.bytes_remaining() != 0) return false;

  return models_equal(encoder_model, decoder_model);
}

std::vector<uint32_t> random_symbols(std::mt19937_64* rng, uint32_t alphabet,
                                     size_t count) {
  std::uniform_int_distribution<uint32_t> distribution(0, alphabet - 1);
  std::vector<uint32_t> symbols(count);
  for (uint32_t& symbol : symbols) symbol = distribution(*rng);
  return symbols;
}

std::vector<uint32_t> skewed_symbols(std::mt19937_64* rng, uint32_t alphabet,
                                     size_t count, double zero_probability) {
  std::bernoulli_distribution pick_zero(zero_probability);
  std::uniform_int_distribution<uint32_t> pick_other(1, alphabet - 1);
  std::vector<uint32_t> symbols(count);
  for (uint32_t& symbol : symbols) {
    symbol = pick_zero(*rng) ? 0u : pick_other(*rng);
  }
  return symbols;
}

// ---------------------------------------------------------------------------

void test_uniform_roundtrips() {
  std::mt19937_64 rng(0x9E3779B97F4A7C15ull);
  constexpr std::array<size_t, 8> kLengths = {0, 1, 2, 3, 7, 64, 1000, 4096};
  constexpr std::array<uint32_t, 6> kRates = {0, 2, 4, 5, 8, 12};

  for (uint32_t alphabet = 2; alphabet <= kMaxAlphabetSize; ++alphabet) {
    for (uint32_t rate : kRates) {
      const auto initial = AdaptiveCdf::uniform(alphabet, rate);
      CHECK(initial.has_value());
      if (!initial) continue;
      CHECK(model_is_valid(*initial));

      for (size_t length : kLengths) {
        const std::vector<uint32_t> symbols =
            random_symbols(&rng, alphabet, length);
        size_t stream_size = 0;
        CHECK(roundtrip(symbols, *initial, &stream_size));
        CHECK(stream_size >= 5);
      }
    }
  }
}

void test_initialization_from_tables() {
  // from_frequencies: a skewed four-symbol source.
  const std::array<uint32_t, 4> frequencies = {1, 3, 12, 48};
  const auto from_freq = AdaptiveCdf::from_frequencies(frequencies, 4);
  CHECK(from_freq.has_value());
  if (from_freq) {
    CHECK(model_is_valid(*from_freq));
    CHECK(from_freq->num_symbols() == 4);
    CHECK(from_freq->frequency(0) < from_freq->frequency(1));
    CHECK(from_freq->frequency(1) < from_freq->frequency(2));
    CHECK(from_freq->frequency(2) < from_freq->frequency(3));
    std::mt19937_64 rng(1234);
    const std::vector<uint32_t> symbols =
        random_symbols(&rng, 4, 20000);
    CHECK(roundtrip(symbols, *from_freq));
  }

  // from_cumulative: explicit CDF with deliberately uneven widths.
  const std::array<uint16_t, 6> cdf = {0, 1, 2, 100, 30000, 32768};
  const auto from_cdf = AdaptiveCdf::from_cumulative(cdf, 3);
  CHECK(from_cdf.has_value());
  if (from_cdf) {
    CHECK(model_is_valid(*from_cdf));
    for (uint32_t i = 0; i <= 5; ++i) {
      CHECK(from_cdf->cumulative()[i] == cdf[i]);
    }
    std::mt19937_64 rng(5678);
    const std::vector<uint32_t> symbols = random_symbols(&rng, 5, 20000);
    CHECK(roundtrip(symbols, *from_cdf));
    AdaptiveCdf frozen = *from_cdf;
    CHECK(frozen.set_adaptation_rate(0));
    const std::vector<uint32_t> static_symbols =
        random_symbols(&rng, 5, 1000);
    CHECK(roundtrip(static_symbols, frozen));
  }

  // Single-symbol table forms are valid too.
  const std::array<uint16_t, 2> one_cdf = {0, 32768};
  const auto one = AdaptiveCdf::from_cumulative(one_cdf);
  CHECK(one.has_value());
  if (one) CHECK(model_is_valid(*one));
  const std::array<uint32_t, 1> one_frequency = {12345};
  CHECK(AdaptiveCdf::from_frequencies(one_frequency).has_value());
}

void test_skewed_sources() {
  std::mt19937_64 rng(0xD1B54A32D192ED03ull);
  constexpr std::array<double, 4> kProbabilities = {0.5, 0.9, 0.99, 0.999};

  for (uint32_t alphabet : {2u, 3u, 4u, 8u, 16u}) {
    for (double probability : kProbabilities) {
      const auto initial = AdaptiveCdf::uniform(alphabet, 4);
      CHECK(initial.has_value());
      if (!initial) continue;
      const std::vector<uint32_t> symbols =
          skewed_symbols(&rng, alphabet, 20000, probability);
      size_t stream_size = 0;
      CHECK(roundtrip(symbols, *initial, &stream_size));
      CHECK(stream_size < 20000);  // must beat one byte per symbol
    }
  }

  // Constant source: probability estimate should collapse toward symbol 0.
  for (uint32_t alphabet = 2; alphabet <= kMaxAlphabetSize; ++alphabet) {
    const auto initial = AdaptiveCdf::uniform(alphabet, 4);
    CHECK(initial.has_value());
    if (!initial) continue;
    const std::vector<uint32_t> symbols(10000, 0u);
    size_t stream_size = 0;
    CHECK(roundtrip(symbols, *initial, &stream_size));
    CHECK(stream_size < 32);
  }
}

void test_single_symbol_alphabet() {
  const auto initial = AdaptiveCdf::uniform(1, 5);
  CHECK(initial.has_value());
  if (!initial) return;

  AdaptiveCdf model = *initial;
  Encoder encoder;
  for (int i = 0; i < 1000; ++i) {
    CHECK_STATUS(encoder.encode(0, model), RangeCoderStatus::kOk);
  }
  CHECK_STATUS(encoder.finish(), RangeCoderStatus::kOk);

  // No information is emitted; only the five flush bytes exist.
  CHECK(encoder.bitstream().size() == 5);
  for (uint8_t byte : encoder.bitstream()) CHECK(byte == 0);
  CHECK(models_equal(model, *initial));

  for (size_t cut = 0; cut < 5; ++cut) {
    Decoder truncated(std::span<const uint8_t>(encoder.bitstream().data(), cut));
    CHECK_STATUS(truncated.init(), RangeCoderStatus::kTruncatedInput);
  }

  Decoder decoder(encoder.bitstream());
  CHECK_STATUS(decoder.init(), RangeCoderStatus::kOk);
  AdaptiveCdf decoder_model = *initial;
  for (int i = 0; i < 1000; ++i) {
    uint32_t symbol = 7;
    CHECK_STATUS(decoder.decode(decoder_model, &symbol),
                 RangeCoderStatus::kOk);
    CHECK(symbol == 0);
  }
  CHECK(models_equal(decoder_model, *initial));
  CHECK(decoder.bytes_remaining() == 0);
}

void test_long_sequence() {
  constexpr size_t kLength = 1000000;
  constexpr uint32_t kAlphabet = 16;

  const auto initial = AdaptiveCdf::uniform(kAlphabet, 5);
  CHECK(initial.has_value());
  if (!initial) return;

  std::mt19937_64 rng(0xA0761D6478BD642Full);
  std::vector<uint32_t> symbols = skewed_symbols(&rng, kAlphabet, kLength, 0.75);

  const auto start = std::chrono::steady_clock::now();
  size_t stream_size = 0;
  CHECK(roundtrip(symbols, *initial, &stream_size));
  const auto elapsed = std::chrono::steady_clock::now() - start;

  const double milliseconds =
      std::chrono::duration<double, std::milli>(elapsed).count();
  std::printf("  long sequence: %zu symbols -> %zu bytes, %.1f ms\n", kLength,
              stream_size, milliseconds);
  CHECK(stream_size > 5);
  CHECK(stream_size < kLength);
}

void test_model_invariants() {
  std::mt19937_64 rng(0xE7037ED1A0B428DBull);

  for (uint32_t alphabet : {1u, 2u, 3u, 5u, 16u}) {
    for (uint32_t rate : {0u, 1u, 5u, 15u}) {
      const auto initial = AdaptiveCdf::uniform(alphabet, rate);
      CHECK(initial.has_value());
      if (!initial) continue;
      AdaptiveCdf model = *initial;
      CHECK(model_is_valid(model));

      std::uniform_int_distribution<uint32_t> distribution(0, alphabet - 1);
      bool updates_ok = true;
      for (int i = 0; i < 200000; ++i) {
        const uint32_t symbol = distribution(rng);
        updates_ok = model.update(symbol) && updates_ok;
        if ((i & 0x3FF) == 0) CHECK(model_is_valid(model));
      }
      CHECK(updates_ok);
      CHECK(model_is_valid(model));

      model.reset();
      CHECK(models_equal(model, *initial));
    }
  }

  // Adaptation must move probability mass toward the observed symbol.
  const auto binary = AdaptiveCdf::uniform(2, 5);
  CHECK(binary.has_value());
  if (binary) {
    AdaptiveCdf model = *binary;
    const uint32_t before = model.frequency(0);
    for (int i = 0; i < 100; ++i) CHECK(model.update(0));
    CHECK(model.frequency(0) > before);
    CHECK(model.frequency(1) >= 1);
  }
}

void test_truncated_input() {
  constexpr uint32_t kAlphabet = 4;
  constexpr size_t kLength = 5000;
  const auto initial = AdaptiveCdf::uniform(kAlphabet, 5);
  CHECK(initial.has_value());
  if (!initial) return;

  std::mt19937_64 rng(42);
  const std::vector<uint32_t> symbols =
      random_symbols(&rng, kAlphabet, kLength);

  AdaptiveCdf encoder_model = *initial;
  Encoder encoder;
  for (uint32_t symbol : symbols) {
    CHECK_STATUS(encoder.encode(symbol, encoder_model), RangeCoderStatus::kOk);
  }
  CHECK_STATUS(encoder.finish(), RangeCoderStatus::kOk);
  const std::vector<uint8_t>& stream = encoder.bitstream();
  const size_t length = stream.size();
  CHECK(length > 100);

  // Fewer than five bytes cannot even initialize.
  for (size_t cut = 0; cut < 5; ++cut) {
    Decoder decoder(std::span<const uint8_t>(stream.data(), cut));
    CHECK_STATUS(decoder.init(), RangeCoderStatus::kTruncatedInput);
  }

  // Removing part of the payload must be reported as truncation while
  // decoding the expected number of symbols.  The decoder consumes exactly the
  // encoder's byte stream (see README), so the prefix still decodes correctly
  // and the missing byte is hit before the final symbol.
  const std::array<size_t, 7> cuts = {
      5, length / 4, length / 2, (3 * length) / 4, length - 16, length - 8,
      length - 6};
  for (size_t cut : cuts) {
    if (cut < 5 || cut >= length) continue;
    AdaptiveCdf decoder_model = *initial;
    Decoder decoder(std::span<const uint8_t>(stream.data(), cut));
    const RangeCoderStatus init_status = decoder.init();
    CHECK(init_status == RangeCoderStatus::kOk);
    if (init_status != RangeCoderStatus::kOk) continue;

    bool saw_truncation = false;
    for (size_t i = 0; i < kLength; ++i) {
      uint32_t symbol = 0;
      const RangeCoderStatus status = decoder.decode(decoder_model, &symbol);
      if (status != RangeCoderStatus::kOk) {
        CHECK(status == RangeCoderStatus::kTruncatedInput);
        saw_truncation = true;
        break;
      }
      CHECK(symbol == symbols[i]);
    }
    CHECK(saw_truncation);
  }

  // Exhaustive truncation test: every possible cut of a shorter stream must be
  // detected, and the decodable prefix must still match.
  {
    constexpr size_t kSmallLength = 600;
    const std::vector<uint32_t> small_symbols =
        random_symbols(&rng, kAlphabet, kSmallLength);
    AdaptiveCdf small_encoder_model = *initial;
    Encoder small_encoder;
    for (uint32_t symbol : small_symbols) {
      CHECK_STATUS(small_encoder.encode(symbol, small_encoder_model),
                   RangeCoderStatus::kOk);
    }
    CHECK_STATUS(small_encoder.finish(), RangeCoderStatus::kOk);
    const std::vector<uint8_t>& small_stream = small_encoder.bitstream();

    for (size_t cut = 0; cut < small_stream.size(); ++cut) {
      Decoder decoder(std::span<const uint8_t>(small_stream.data(), cut));
      const RangeCoderStatus init_status = decoder.init();
      if (init_status != RangeCoderStatus::kOk) {
        CHECK(init_status == RangeCoderStatus::kTruncatedInput);
        continue;
      }

      AdaptiveCdf decoder_model = *initial;
      bool saw_truncation = false;
      for (size_t i = 0; i < kSmallLength; ++i) {
        uint32_t symbol = 0;
        const RangeCoderStatus status = decoder.decode(decoder_model, &symbol);
        if (status != RangeCoderStatus::kOk) {
          CHECK(status == RangeCoderStatus::kTruncatedInput);
          saw_truncation = true;
          break;
        }
        CHECK(symbol == small_symbols[i]);
      }
      CHECK(saw_truncation);
    }
  }

  // A decoder that hit a fatal error stays failed.
  {
    Decoder decoder(std::span<const uint8_t>(stream.data(), 5));
    CHECK_STATUS(decoder.init(), RangeCoderStatus::kOk);
    AdaptiveCdf decoder_model = *initial;
    bool saw_error = false;
    for (size_t i = 0; i < 100000; ++i) {
      uint32_t symbol = 0;
      const RangeCoderStatus status = decoder.decode(decoder_model, &symbol);
      if (status != RangeCoderStatus::kOk) {
        saw_error = true;
        CHECK(status == RangeCoderStatus::kTruncatedInput);
        break;
      }
    }
    CHECK(saw_error);
    uint32_t symbol = 0;
    CHECK_STATUS(decoder.decode(decoder_model, &symbol),
                 RangeCoderStatus::kDecoderFailed);
  }
}

void test_corruption_and_garbage() {
  constexpr uint32_t kAlphabet = 8;
  constexpr size_t kLength = 512;
  const auto initial = AdaptiveCdf::uniform(kAlphabet, 5);
  CHECK(initial.has_value());
  if (!initial) return;

  std::mt19937_64 rng(0x8EBC6AF09C88C6E3ull);
  const std::vector<uint32_t> symbols =
      random_symbols(&rng, kAlphabet, kLength);

  AdaptiveCdf encoder_model = *initial;
  Encoder encoder;
  for (uint32_t symbol : symbols) {
    CHECK_STATUS(encoder.encode(symbol, encoder_model), RangeCoderStatus::kOk);
  }
  CHECK_STATUS(encoder.finish(), RangeCoderStatus::kOk);
  const std::vector<uint8_t> stream = encoder.bitstream();
  const size_t length = stream.size();
  CHECK(length > 64);

  // Flip one bit in bytes that are definitely part of the coded value.  The
  // first byte is the carry guard, which the decoder ignores, and a flip in
  // the last few bytes can still leave the code inside the same final
  // interval.  A raw arithmetic stream has no checksum, so "detected" means
  // either a decode error or a data mismatch.
  const std::array<size_t, 8> positions = {1,
                                           2,
                                           3,
                                           length / 4,
                                           length / 2,
                                           (3 * length) / 4,
                                           length - 12,
                                           length - 8};
  for (size_t position : positions) {
    if (position == 0 || position + 4 >= length) continue;
    std::vector<uint8_t> corrupted = stream;
    corrupted[position] ^= 0x01u;

    Decoder decoder(corrupted);
    CHECK_STATUS(decoder.init(), RangeCoderStatus::kOk);
    AdaptiveCdf decoder_model = *initial;
    bool detected = false;
    for (size_t i = 0; i < kLength; ++i) {
      uint32_t symbol = 0;
      const RangeCoderStatus status = decoder.decode(decoder_model, &symbol);
      if (status != RangeCoderStatus::kOk) {
        detected = true;
        break;
      }
      if (symbol != symbols[i]) {
        detected = true;
        break;
      }
    }
    CHECK(detected);
  }

  // Random bytes must never crash or emit an out-of-alphabet symbol, even
  // though they may decode to a perfectly "valid" arbitrary sequence.
  std::uniform_int_distribution<int> byte_distribution(0, 255);
  std::uniform_int_distribution<size_t> size_distribution(5, 256);
  for (int trial = 0; trial < 500; ++trial) {
    const size_t garbage_size = size_distribution(rng);
    std::vector<uint8_t> garbage(garbage_size);
    for (uint8_t& byte : garbage) {
      byte = static_cast<uint8_t>(byte_distribution(rng));
    }
    Decoder decoder(garbage);
    const RangeCoderStatus init_status = decoder.init();
    CHECK(init_status == RangeCoderStatus::kOk ||
          init_status == RangeCoderStatus::kTruncatedInput);
    if (init_status != RangeCoderStatus::kOk) continue;

    AdaptiveCdf model = *initial;
    for (int i = 0; i < 4096; ++i) {
      uint32_t symbol = 0;
      if (decoder.decode(model, &symbol) != RangeCoderStatus::kOk) break;
      CHECK(symbol < kAlphabet);
      CHECK(model_is_valid(model));
    }
  }
}

void test_api_misuse_and_model_validation() {
  // Invalid alphabet sizes and rates.
  CHECK(!AdaptiveCdf::uniform(0).has_value());
  CHECK(!AdaptiveCdf::uniform(kMaxAlphabetSize + 1).has_value());
  CHECK(!AdaptiveCdf::uniform(2, kMaxAdaptationRate + 1).has_value());
  CHECK(AdaptiveCdf::uniform(2, kMaxAdaptationRate).has_value());
  CHECK(AdaptiveCdf::uniform(1, 0).has_value());

  // Invalid frequencies.
  CHECK(!AdaptiveCdf::from_frequencies(
             std::span<const uint32_t>()).has_value());
  const std::array<uint32_t, 2> zero_frequency = {0, 1};
  CHECK(!AdaptiveCdf::from_frequencies(zero_frequency).has_value());
  const std::array<uint32_t, 17> too_many = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  CHECK(!AdaptiveCdf::from_frequencies(too_many).has_value());

  // Invalid cumulative tables.
  const std::array<uint16_t, 3> nonzero_start = {1, 100, 32768};
  const std::array<uint16_t, 3> wrong_total = {0, 100, 32767};
  const std::array<uint16_t, 3> not_increasing = {0, 100, 100};
  const std::array<uint16_t, 1> empty_cdf = {0};
  const std::array<uint16_t, 18> oversized_cdf = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32768};
  CHECK(!AdaptiveCdf::from_cumulative(nonzero_start).has_value());
  CHECK(!AdaptiveCdf::from_cumulative(wrong_total).has_value());
  CHECK(!AdaptiveCdf::from_cumulative(not_increasing).has_value());
  CHECK(!AdaptiveCdf::from_cumulative(empty_cdf).has_value());
  CHECK(!AdaptiveCdf::from_cumulative(oversized_cdf).has_value());
  CHECK(!AdaptiveCdf::from_cumulative(
             std::span<const uint16_t>()).has_value());

  // Invalid update / rate changes leave the model untouched.
  const auto initial = AdaptiveCdf::uniform(4, 5);
  CHECK(initial.has_value());
  if (initial) {
    AdaptiveCdf model = *initial;
    CHECK(!model.update(4));
    CHECK(models_equal(model, *initial));
    CHECK(!model.set_adaptation_rate(16));
    CHECK(model.adaptation_rate() == 5);
    CHECK(model.set_adaptation_rate(0));
    CHECK(model.adaptation_rate() == 0);
    const uint32_t before = model.frequency(0);
    CHECK(model.update(0));
    CHECK(model.frequency(0) == before);  // frozen
  }

  // Encoder misuse.
  Encoder encoder;
  const auto model = AdaptiveCdf::uniform(3, 5);
  CHECK(model.has_value());
  if (!model) return;
  AdaptiveCdf encoder_model = *model;
  CHECK_STATUS(encoder.encode(3, encoder_model),
               RangeCoderStatus::kInvalidSymbol);
  CHECK(models_equal(encoder_model, *model));  // failed encode is atomic
  CHECK_STATUS(encoder.encode(1, encoder_model), RangeCoderStatus::kOk);
  CHECK_STATUS(encoder.finish(), RangeCoderStatus::kOk);
  CHECK_STATUS(encoder.encode(0, encoder_model),
               RangeCoderStatus::kEncoderFinished);
  CHECK_STATUS(encoder.finish(), RangeCoderStatus::kEncoderFinished);

  const std::vector<uint8_t>& stream = encoder.bitstream();
  CHECK(stream.size() >= 5);

  // Decoder misuse.
  Decoder decoder{std::span<const uint8_t>(stream)};
  uint32_t symbol = 0;
  AdaptiveCdf decoder_model = *model;
  CHECK_STATUS(decoder.decode(decoder_model, &symbol),
               RangeCoderStatus::kDecoderNotInitialized);
  CHECK_STATUS(decoder.init(), RangeCoderStatus::kOk);
  CHECK_STATUS(decoder.init(), RangeCoderStatus::kDecoderAlreadyInitialized);
  CHECK_STATUS(decoder.decode(decoder_model, nullptr),
               RangeCoderStatus::kInvalidArgument);
  CHECK_STATUS(decoder.decode(decoder_model, &symbol), RangeCoderStatus::kOk);
  CHECK(symbol == 1);

  // A decoder can be rebound to another stream.
  std::vector<uint8_t> copy(stream);
  decoder.reset(std::span<const uint8_t>(copy));
  CHECK_STATUS(decoder.init(), RangeCoderStatus::kOk);
  CHECK_STATUS(decoder.decode(decoder_model, &symbol), RangeCoderStatus::kOk);
  CHECK(symbol == 1);
}

// Pins the byte stream format.  If the coder changes intentionally, this test
// and the format section of README.md must change together.
void test_golden_vector() {
  const auto initial = AdaptiveCdf::uniform(4, 5);
  CHECK(initial.has_value());
  if (!initial) return;

  const std::array<uint32_t, 10> symbols = {0, 1, 2, 3, 3, 2, 1, 0, 0, 2};
  AdaptiveCdf encoder_model = *initial;
  Encoder encoder;
  for (uint32_t symbol : symbols) {
    CHECK_STATUS(encoder.encode(symbol, encoder_model), RangeCoderStatus::kOk);
  }
  CHECK_STATUS(encoder.finish(), RangeCoderStatus::kOk);

  const std::array<uint8_t, 7> expected_bytes = {0x00, 0x1D, 0x38, 0xFD,
                                                 0x67, 0x7C, 0x34};
  const std::vector<uint8_t>& stream = encoder.bitstream();
  CHECK(stream.size() == expected_bytes.size());
  if (stream.size() == expected_bytes.size()) {
    for (size_t i = 0; i < expected_bytes.size(); ++i) {
      CHECK(stream[i] == expected_bytes[i]);
    }
  }

  // Deterministic model state after the ten updates.
  const std::array<uint16_t, 5> expected_cdf = {0, 8667, 16370, 25064, 32768};
  for (uint32_t i = 0; i <= 4; ++i) {
    CHECK(encoder_model.cumulative()[i] == expected_cdf[i]);
  }

  AdaptiveCdf decoder_model = *initial;
  Decoder decoder{std::span<const uint8_t>(stream)};
  CHECK_STATUS(decoder.init(), RangeCoderStatus::kOk);
  for (uint32_t expected : symbols) {
    uint32_t decoded = 0xFFFFFFFFu;
    CHECK_STATUS(decoder.decode(decoder_model, &decoded),
                 RangeCoderStatus::kOk);
    CHECK(decoded == expected);
  }
  CHECK(models_equal(encoder_model, decoder_model));
  CHECK(decoder.bytes_remaining() == 0);
}

// A sequence that leaves a 0xFF byte pending at finish.  This exercises the
// carry/cache flush path that the usual "finish adds five bytes" case misses.
void test_pending_ff_flush() {
  const auto initial = AdaptiveCdf::uniform(2, 0);
  CHECK(initial.has_value());
  if (!initial) return;

  // Symbol i is bit i of 0x0001: one 1 followed by fifteen 0s.
  const std::array<uint32_t, 16> symbols = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  AdaptiveCdf encoder_model = *initial;
  Encoder encoder;
  for (uint32_t symbol : symbols) {
    CHECK_STATUS(encoder.encode(symbol, encoder_model), RangeCoderStatus::kOk);
  }
  const size_t before_finish = encoder.bitstream().size();
  CHECK_STATUS(encoder.finish(), RangeCoderStatus::kOk);

  // One pending 0xFF byte makes finish emit six bytes; the total is still
  // renormalization_count + 5.
  CHECK(encoder.bitstream().size() - before_finish == 6);
  const std::array<uint8_t, 7> expected = {0x00, 0x7F, 0xFF, 0xC0,
                                           0x00, 0x00, 0x00};
  const std::vector<uint8_t>& stream = encoder.bitstream();
  CHECK(stream.size() == expected.size());
  if (stream.size() == expected.size()) {
    for (size_t i = 0; i < expected.size(); ++i) {
      CHECK(stream[i] == expected[i]);
    }
  }

  AdaptiveCdf decoder_model = *initial;
  Decoder decoder{std::span<const uint8_t>(stream)};
  CHECK_STATUS(decoder.init(), RangeCoderStatus::kOk);
  for (uint32_t expected_symbol : symbols) {
    uint32_t decoded = 0xFFFFFFFFu;
    CHECK_STATUS(decoder.decode(decoder_model, &decoded),
                 RangeCoderStatus::kOk);
    CHECK(decoded == expected_symbol);
  }
  CHECK(models_equal(encoder_model, decoder_model));
}

void run(const char* name, void (*test)()) {
  const int failures_before = g_failures;
  test();
  std::printf("%s %s\n", g_failures == failures_before ? "PASS" : "FAIL",
              name);
}

}  // namespace

int main() {
  run("uniform round trips (alphabets 2..16)", test_uniform_roundtrips);
  run("initialization from frequency/CDF tables",
      test_initialization_from_tables);
  run("skewed and constant sources", test_skewed_sources);
  run("single-symbol alphabet", test_single_symbol_alphabet);
  run("long sequence (>= 1e6 symbols)", test_long_sequence);
  run("model invariants", test_model_invariants);
  run("truncated input", test_truncated_input);
  run("corruption and random garbage", test_corruption_and_garbage);
  run("API misuse and model validation", test_api_misuse_and_model_validation);
  run("pending 0xFF flush", test_pending_ff_flush);
  run("golden vector", test_golden_vector);

  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
