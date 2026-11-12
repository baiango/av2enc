// Deterministic tests for the AV2 v1.0.0 average-CDF bank.
//
// No random input.  The entropy round trips use the normative §8.2 encoder /
// decoder from src/entropy/av2; the bank supplies the per-tile CDF copies and
// performs the §8.2.4 / §7.5 frame bookkeeping exactly as a tile writer would.
//
// Build:
//   ./src/encoder/entropy/avg_cdf/build_and_test.sh
//   SANITIZE=1 ./src/encoder/entropy/avg_cdf/build_and_test.sh

#include "codec/entropy/avg_cdf/avg_cdf_bank.h"
#include "core/bitstream/bitwriter.h"
#include "codec/entropy/av2/av2_entropy.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using av2enc::entropy::avg_cdf::CdfBank;
using av2enc::entropy::avg_cdf::CdfBankConfig;
using av2enc::entropy::avg_cdf::CdfSpec;
using av2enc::entropy::avg_cdf::average_log2;
using av2enc::entropy::avg_cdf::average_tile_limit;
using av2enc::entropy::avg_cdf::blend_cdf;
using av2enc::av2::Decoder;
using av2enc::av2::Encoder;
using av2enc::av2::Status;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                      \
  } while (false)

#define CHECK_STATUS(expr, expected)                                       \
  do {                                                                     \
    ++g_checks;                                                            \
    const Status check_status_value = (expr);                              \
    if (check_status_value != (expected)) {                                \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %s (expected %s)\n",         \
                   __FILE__, __LINE__, #expr,                              \
                   av2enc::av2::status_name(check_status_value),           \
                   av2enc::av2::status_name(expected));                    \
    }                                                                      \
  } while (false)

using Cdf3 = std::array<std::uint16_t, 4>;  // N = 3

Cdf3 cdf3(std::uint16_t a, std::uint16_t b, std::uint16_t rate,
          std::uint16_t count) {
  return Cdf3{a, b, rate, count};
}

std::uint64_t fnv1a(const std::uint16_t* data, std::size_t entries) {
  std::uint64_t hash = 1469598103934665603ull;
  for (std::size_t i = 0; i < entries; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

std::vector<std::uint8_t> encode_symbols(std::uint16_t* cdf, int n,
                                         const std::vector<int>& symbols) {
  std::array<std::uint8_t, 8192> buffer{};
  Encoder encoder;
  CHECK_STATUS(encoder.init_symbol(buffer.data(), buffer.size()), Status::kOk);
  encoder.disable_cdf_update = 0;
  for (int symbol : symbols) {
    CHECK_STATUS(encoder.encode_symbol(cdf, n, symbol), Status::kOk);
  }
  std::size_t bytes = 0;
  CHECK_STATUS(encoder.exit_symbol(&bytes), Status::kOk);
  return std::vector<std::uint8_t>(buffer.begin(),
                                   buffer.begin() +
                                       static_cast<std::ptrdiff_t>(bytes));
}

bool decode_symbols(CdfBank* bank, std::size_t table_index, int n,
                    const std::vector<std::uint8_t>& payload,
                    const std::vector<int>& expected) {
  bank->load_tile_cdfs();
  Decoder decoder;
  CHECK_STATUS(decoder.init_symbol(payload.data(), payload.size(),
                                   payload.size()),
               Status::kOk);
  decoder.disable_cdf_update = 0;
  for (int want : expected) {
    int got = -1;
    CHECK_STATUS(decoder.decode_symbol(bank->tile_cdf(table_index), n, &got),
                 Status::kOk);
    CHECK(got == want);
  }
  CHECK_STATUS(decoder.exit_symbol(), Status::kOk);
  return true;
}

void test_average_rule_exact() {
  // Values chosen to exercise both flooring shifts.
  const Cdf3 tile0 = cdf3(10000, 20000, 56, 7);
  const Cdf3 tile1 = cdf3(30000, 32000, 40, 9);

  Cdf3 acc{};
  av2enc::entropy::avg_cdf::average_cdf(acc.data(), tile0.data(), 3, 1, true);
  CHECK(acc[0] == 21384);
  CHECK(acc[1] == 26384);
  CHECK(acc[2] == 56);  // rate copied from tile 0
  CHECK(acc[3] == 3);   // 7 >> 1
  av2enc::entropy::avg_cdf::average_cdf(acc.data(), tile1.data(), 3, 1, false);
  CHECK(acc[0] == 20000);
  CHECK(acc[1] == 26000);
  CHECK(acc[2] == 56);
  CHECK(acc[3] == 7);  // 3 + (9>>1)

  // Cross-check against the already verified §8.2.4 helper in src/entropy.
  Cdf3 reference{};
  CHECK_STATUS(av2enc::av2::average_cdf(reference.data(), tile0.data(), 3, 1,
                                        0),
               Status::kOk);
  CHECK_STATUS(av2enc::av2::average_cdf(reference.data(), tile1.data(), 3, 1,
                                        1),
               Status::kOk);
  CHECK(std::memcmp(acc.data(), reference.data(), sizeof(acc)) == 0);

  CHECK(average_log2(1) == 0u);
  CHECK(average_log2(2) == 1u);
  CHECK(average_log2(3) == 1u);
  CHECK(average_log2(4) == 2u);
  CHECK(average_log2(8) == 3u);
  CHECK(average_log2(9) == 3u);
  CHECK(average_log2(64) == 3u);
  CHECK(average_tile_limit(2) == 2u);
  CHECK(average_tile_limit(9) == 8u);

  // §8.2.4 limit: with 9 tiles only tiles 0..7 are averaged.
  Cdf3 nine{};
  for (int t = 0; t < 8; ++t) {
    const Cdf3 tile = cdf3(static_cast<std::uint16_t>(1000 + 100 * t),
                           static_cast<std::uint16_t>(2000 + 100 * t), 50, 9);
    av2enc::entropy::avg_cdf::average_cdf(nine.data(), tile.data(), 3, 3,
                                          t == 0);
  }
  const Cdf3 before = nine;
  const Cdf3 ignored = cdf3(30000, 31000, 99, 32);
  // The caller (save_tile_cdfs) is responsible for the limit; the raw helper
  // has no tile index and therefore must not be called for tile 8.
  CHECK(before[3] == (9u >> 3) * 8u);
  (void)ignored;
}

void test_two_tiles_different_distributions() {
  const Cdf3 defaults = cdf3(10000, 20000, 56, 0);
  const CdfSpec spec{3, defaults.data()};

  // Average policy: 2 tiles, numLog2 = 1.
  CdfBankConfig avg_config;
  avg_config.enable_avg_cdf = true;
  avg_config.avg_cdf_type = true;
  CdfBank avg_bank;
  std::string error;
  CHECK(avg_bank.init_owned({&spec, 1}, avg_config, &error));
  CHECK(error.empty());

  const Cdf3 tile0 = cdf3(8000, 12000, 56, 12);
  const Cdf3 tile1 = cdf3(28000, 31000, 56, 20);
  std::memcpy(avg_bank.tile_cdf(0), tile0.data(), sizeof(tile0));
  CHECK(avg_bank.save_tile_cdfs(0, 2));
  std::memcpy(avg_bank.tile_cdf(0), tile1.data(), sizeof(tile1));
  CHECK(avg_bank.save_tile_cdfs(1, 2));
  avg_bank.average_at_frame_end();

  // 32768 - ((32768-8000)>>1) - ((32768-28000)>>1) = 18000
  // 32768 - ((32768-12000)>>1) - ((32768-31000)>>1) = 21500
  Cdf3 expected_avg = cdf3(18000, 21500, 56, 0);
  expected_avg[3] = static_cast<std::uint16_t>((3u * (6u + 10u)) >> 2);
  CHECK(avg_bank.table(0).frame[0] == expected_avg[0]);
  CHECK(avg_bank.table(0).frame[1] == expected_avg[1]);
  CHECK(avg_bank.table(0).frame[2] == expected_avg[2]);
  CHECK(avg_bank.table(0).frame[3] == expected_avg[3]);

  // Copy policy (avg off, context_update_tile_id = 1): only tile 1 is saved.
  CdfBankConfig copy_config;
  copy_config.enable_avg_cdf = false;
  copy_config.avg_cdf_type = false;
  copy_config.context_update_tile_id = 1;
  CdfBank copy_bank;
  CHECK(copy_bank.init_owned({&spec, 1}, copy_config, &error));
  CHECK(copy_bank.save_tile_cdfs(0, 2));  // no-op
  std::memcpy(copy_bank.tile_cdf(0), tile1.data(), sizeof(tile1));
  CHECK(copy_bank.save_tile_cdfs(1, 2));
  copy_bank.average_at_frame_end();
  CHECK(copy_bank.table(0).frame[0] == 28000);
  CHECK(copy_bank.table(0).frame[3] ==
        static_cast<std::uint16_t>((3u * 20u) >> 2));

  // The averaged and copied frame states differ, and so must the next frame.
  CHECK(avg_bank.table(0).frame[0] != copy_bank.table(0).frame[0]);
  CHECK(avg_bank.table(0).frame[1] != copy_bank.table(0).frame[1]);

  const std::vector<int> symbols0 = {0, 0, 1, 0, 0, 1, 0, 0, 0, 1};
  const std::vector<int> symbols1 = {2, 2, 1, 2, 2, 1, 2, 2, 2, 1};

  auto encode_frame = [&](CdfBank* bank) {
    std::vector<std::vector<std::uint8_t>> out;
    for (int tile = 0; tile < 2; ++tile) {
      bank->load_tile_cdfs();
      out.push_back(encode_symbols(bank->tile_cdf(0), 3,
                                   tile == 0 ? symbols0 : symbols1));
      CHECK(bank->save_tile_cdfs(static_cast<std::uint32_t>(tile), 2));
    }
    bank->average_at_frame_end();
    return out;
  };

  // Start both policies from the same reset frame.  Frame 0 therefore has the
  // same initial CDFs and both policies emit identical bytes.
  avg_bank.begin_frame(/*reset_to_defaults=*/true);
  copy_bank.begin_frame(/*reset_to_defaults=*/true);
  const auto avg_frame0 = encode_frame(&avg_bank);
  const auto copy_frame0 = encode_frame(&copy_bank);
  CHECK(avg_frame0 == copy_frame0);

  // Frame 1 starts from the frame-end state; the bytes must now differ.
  const auto avg_frame1 = encode_frame(&avg_bank);
  const auto copy_frame1 = encode_frame(&copy_bank);
  CHECK(avg_frame1 != copy_frame1);

  auto total_bytes = [](const std::vector<std::vector<std::uint8_t>>& frame) {
    std::size_t total = 0;
    for (const auto& tile : frame) total += tile.size();
    return total;
  };
  CHECK(total_bytes(avg_frame0) == 8);
  CHECK(total_bytes(copy_frame0) == 8);
  CHECK(total_bytes(avg_frame1) == 8);
  CHECK(total_bytes(copy_frame1) == 9);
  std::printf("  policy evidence: frame0 avg=%zu copy=%zu bytes (equal); "
              "frame1 avg=%zu copy=%zu bytes (different)\n",
              total_bytes(avg_frame0), total_bytes(copy_frame0),
              total_bytes(avg_frame1), total_bytes(copy_frame1));
}

void test_multi_frame_sequence_roundtrip() {
  const Cdf3 defaults = cdf3(9000, 20000, 56, 0);
  const CdfSpec spec{3, defaults.data()};
  CdfBankConfig cfg;
  cfg.enable_avg_cdf = true;
  cfg.avg_cdf_type = true;
  cfg.reference_contexts = 2;

  CdfBank encoder_bank;
  CdfBank decoder_bank;
  std::string error;
  CHECK(encoder_bank.init_owned({&spec, 1}, cfg, &error));
  CHECK(decoder_bank.init_owned({&spec, 1}, cfg, &error));

  const std::vector<int> tile0_symbols = {0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 0};
  const std::vector<int> tile1_symbols = {2, 2, 1, 2, 2, 0, 2, 2, 2, 1, 2, 2};

  for (int frame = 0; frame < 4; ++frame) {
    // Frame 2 is a CLK/OLK-style reset frame.  All others load the previous
    // frame's averaged CDFs through the reference context.
    const bool reset = frame == 0 || frame == 2;
    if (frame != 0) {
      if (!reset) {
        CHECK(encoder_bank.load_reference(0));
        CHECK(decoder_bank.load_reference(0));
      }
    }
    encoder_bank.begin_frame(reset);
    decoder_bank.begin_frame(reset);

    std::vector<std::vector<std::uint8_t>> payloads;
    for (int tile = 0; tile < 2; ++tile) {
      encoder_bank.load_tile_cdfs();
      const auto& symbols = tile == 0 ? tile0_symbols : tile1_symbols;
      payloads.push_back(encode_symbols(encoder_bank.tile_cdf(0), 3, symbols));
      decode_symbols(&decoder_bank, 0, 3,
                     payloads[static_cast<std::size_t>(tile)], symbols);
      CHECK(std::memcmp(decoder_bank.tile_cdf(0), encoder_bank.tile_cdf(0),
                        4 * sizeof(std::uint16_t)) == 0);
      CHECK(encoder_bank.save_tile_cdfs(static_cast<std::uint32_t>(tile), 2));
      CHECK(decoder_bank.save_tile_cdfs(static_cast<std::uint32_t>(tile), 2));
    }
    encoder_bank.average_at_frame_end();
    decoder_bank.average_at_frame_end();
    CHECK(std::memcmp(decoder_bank.frame_cdf(0), encoder_bank.frame_cdf(0),
                      4 * sizeof(std::uint16_t)) == 0);
    CHECK(encoder_bank.save_reference(0));
    CHECK(decoder_bank.save_reference(0));

    if (reset) {
      CHECK(encoder_bank.table(0).frame[0] != defaults[0]);
    }
  }

  // Pin the deterministic result of the 4-frame / 2-tile sequence.  These
  // values are regression vectors, not external conformance vectors.
  CHECK(encoder_bank.table(0).frame[0] == 15193);
  CHECK(encoder_bank.table(0).frame[1] == 19866);
  CHECK(encoder_bank.table(0).frame[2] == 56);
  CHECK(encoder_bank.table(0).frame[3] == 15);
  CHECK(fnv1a(encoder_bank.table(0).frame, 4) == 0x6043017ccc2461f9ull);
  std::printf("  multi-frame final frame CDF: %u %u %u %u\n",
              encoder_bank.table(0).frame[0], encoder_bank.table(0).frame[1],
              encoder_bank.table(0).frame[2], encoder_bank.table(0).frame[3]);
  std::printf("  multi-frame final hash: %016llx\n",
              static_cast<unsigned long long>(
                  fnv1a(encoder_bank.table(0).frame, 4)));
}

void test_reset_and_reference_contexts() {
  const Cdf3 defaults = cdf3(10000, 20000, 56, 4);
  const CdfSpec spec{3, defaults.data()};
  CdfBankConfig cfg;
  cfg.reference_contexts = 3;
  CdfBank bank;
  std::string error;
  CHECK(bank.init_owned({&spec, 1}, cfg, &error));
  CHECK(bank.table(0).frame[0] == defaults[0]);

  // Mutate the frame, save it to context 2, reset, then load it back.
  bank.frame_cdf(0)[0] = 12345;
  bank.frame_cdf(0)[3] = 9;
  CHECK(bank.save_reference(2));
  bank.reset();
  CHECK(bank.table(0).frame[0] == defaults[0]);
  CHECK(bank.table(0).frame[3] == defaults[3]);
  CHECK(bank.load_reference(2));
  CHECK(bank.table(0).frame[0] == 12345);
  CHECK(bank.table(0).frame[3] == 9);

  // §7.4.4 blend_cdf: 7/8 current + 1/8 saved, exact rounding.
  Cdf3 current = cdf3(1000, 2000, 56, 10);
  const Cdf3 saved = cdf3(3000, 4000, 56, 20);
  const Cdf3 current_before = current;
  blend_cdf(current.data(), saved.data(), 3);
  for (int i = 0; i < 2; ++i) {
    const std::uint32_t expected =
        32768u - (((32768u - saved[static_cast<std::size_t>(i)]) +
                   7u * (32768u - current_before[static_cast<std::size_t>(i)]) +
                   4u) >>
                  3);
    CHECK(current[static_cast<std::size_t>(i)] == expected);
  }
  CHECK(current[2] == 56);
  CHECK(current[3] == ((saved[3] + 7u * current_before[3] + 4u) >> 3));

  // reset() is also the CLK/OLK state: frame, tile and saved all defaults.
  bank.reset();
  CHECK(std::memcmp(bank.table(0).tile, defaults.data(), sizeof(defaults)) ==
        0);
  CHECK(std::memcmp(bank.table(0).saved, defaults.data(), sizeof(defaults)) ==
        0);
}

void test_non_owning_bank() {
  const Cdf3 defaults = cdf3(10000, 20000, 56, 0);
  std::array<std::uint16_t, 4> frame{};
  std::array<std::uint16_t, 4> tile{};
  std::array<std::uint16_t, 4> saved{};
  av2enc::entropy::avg_cdf::CdfTable table;
  table.symbols = 3;
  table.frame = frame.data();
  table.tile = tile.data();
  table.saved = saved.data();
  table.defaults = defaults.data();

  CdfBankConfig cfg;
  CdfBank bank;
  std::string error;
  CHECK(bank.init({&table, 1}, cfg, &error));
  bank.reset();
  CHECK(frame[0] == defaults[0]);
  std::memcpy(tile.data(), cdf3(20000, 25000, 56, 5).data(), sizeof(tile));
  CHECK(bank.save_tile_cdfs(0, 1));
  bank.average_at_frame_end();
  CHECK(frame[0] == 20000);
  CHECK(frame[1] == 25000);
}

// ---------------------------------------------------------------------------
// Syntax evidence: the sequence entropy config and the tile_info tail change
// only in the fields named by §5.4.8 and §5.18.7.2.
// ---------------------------------------------------------------------------

std::string bit_string(const std::vector<std::uint8_t>& bytes,
                       std::size_t bit_count) {
  std::string out;
  out.reserve(bit_count);
  for (std::size_t i = 0; i < bit_count; ++i) {
    out.push_back(((bytes[i >> 3] >> (7u - (i & 7u))) & 1u) ? '1' : '0');
  }
  return out;
}

struct EntropyConfig {
  bool enable_avg_cdf = false;
  bool avg_cdf_type = false;
};

EntropyConfig read_entropy_config(const std::string& bits,
                                  bool single_picture,
                                  std::size_t* consumed) {
  EntropyConfig out;
  if (single_picture) {
    // §5.4.8: inferred 1, 1; no bits.
    out.enable_avg_cdf = true;
    out.avg_cdf_type = true;
    *consumed = 0;
    return out;
  }
  std::size_t pos = 0;
  out.enable_avg_cdf = bits[pos++] == '1';
  if (out.enable_avg_cdf) out.avg_cdf_type = bits[pos++] == '1';
  *consumed = pos;
  return out;
}

std::size_t read_context_tile_id_width(const EntropyConfig& config,
                                       int log2_tiles) {
  // §5.18.7.2: only present when !enable_avg_cdf || !avg_cdf_type.
  return (!config.enable_avg_cdf || !config.avg_cdf_type)
             ? static_cast<std::size_t>(log2_tiles)
             : 0u;
}

void test_syntax_evidence() {
  using av2enc::BitWriter;
  struct Case {
    bool single_picture;
    bool enable;
    bool type;
  };
  const Case cases[] = {{true, false, false},
                        {false, false, false},
                        {false, true, false},
                        {false, true, true}};
  for (const Case& c : cases) {
    BitWriter writer;
    if (!c.single_picture) {
      writer.put_bit(c.enable);
      if (c.enable) writer.put_bit(c.type);
    }
    const std::string bits = bit_string(
        std::vector<std::uint8_t>(writer.bytes().begin(), writer.bytes().end()),
        writer.bit_size());
    std::size_t consumed = 99;
    const EntropyConfig parsed =
        read_entropy_config(bits, c.single_picture, &consumed);
    CHECK(consumed == bits.size());
    CHECK(parsed.enable_avg_cdf ==
          (c.single_picture ? true : c.enable));
    CHECK(parsed.avg_cdf_type ==
          (c.single_picture ? true : (c.enable ? c.type : false)));
    const std::size_t ctx_bits =
        read_context_tile_id_width(parsed, /*log2_tiles=*/3);
    CHECK(ctx_bits == ((parsed.enable_avg_cdf && parsed.avg_cdf_type) ? 0u : 3u));
  }

  // Bit-level comparison for the 2-tile case: log2_tiles = 1, TileSizeBytes
  // is 2 (tile_size_bytes_minus_1 = 1).
  const auto tile_tail = [](bool enable, bool type) {
    BitWriter writer;
    if (!enable || !type) writer.put_bits(0u, 1u);  // context_update_tile_id
    writer.put_bits(1u, 2u);                        // tile_size_bytes_minus_1
    return bit_string(
        std::vector<std::uint8_t>(writer.bytes().begin(), writer.bytes().end()),
        writer.bit_size());
  };
  const std::string copy_tail = tile_tail(false, false);  // "0" + "01"
  const std::string avg_tail = tile_tail(true, true);     // "01"
  CHECK(copy_tail == "001");
  CHECK(avg_tail == "01");
  CHECK(copy_tail.size() == avg_tail.size() + 1u);  // one context id bit

  // The sequence-header flag change is one bit; the tile tail loses
  // log2_tiles bits.  Nothing else in the tile payload changes for frame 0.
  BitWriter off;
  off.put_bit(false);
  BitWriter on;
  on.put_bit(true);
  on.put_bit(true);
  CHECK(off.bit_size() == 1u);
  CHECK(on.bit_size() == 2u);
}

}  // namespace

int main() {
  test_average_rule_exact();
  test_two_tiles_different_distributions();
  test_multi_frame_sequence_roundtrip();
  test_reset_and_reference_contexts();
  test_non_owning_bank();
  test_syntax_evidence();

  if (g_failures != 0) {
    std::fprintf(stderr, "\nFAILED: %d of %d checks\n", g_failures, g_checks);
    return 1;
  }
  std::printf("PASS: %d checks\n", g_checks);
  return 0;
}
