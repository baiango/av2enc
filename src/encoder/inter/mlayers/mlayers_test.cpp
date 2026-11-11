// Unit tests for the embedded-layer (mlayer) syntax and validation.
//
// 1. Pinned golden byte strings for §5.4.1 layer headers and §5.2.2 OBU headers,
//    generated independently by golden_reference.py.
// 2. Independent reader round-trip (fields, effective maps, presence map).
// 3. Exhaustive dependency-map validation over all lower-triangular maps for
//    max_mlayer_id = 3 and all per-layer tlayer maps up to max_tlayer_id = 2.
// 4. §7.3.6/§7.3.7 emission-order and availability rejection cases.

#include "encoder/inter/mlayers/mlayers.h"
#include "encoder/inter/mlayers/mlayers_syntax.h"
#include "encoder/m2/m2_headers.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace av2enc::inter::mlayers;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* expression, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::fprintf(stderr, "FAIL line %d: %s\n", line, expression);
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

std::vector<std::uint8_t> from_hex(const char* text) {
  std::vector<std::uint8_t> out;
  const std::string s(text);
  if (s.size() % 2 != 0) {
    std::fprintf(stderr, "bad golden hex length\n");
    std::exit(2);
  }
  for (std::size_t i = 0; i < s.size(); i += 2) {
    out.push_back(static_cast<std::uint8_t>(
        std::strtoul(s.substr(i, 2).c_str(), nullptr, 16)));
  }
  return out;
}

std::string to_hex(std::span<const std::uint8_t> bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  for (std::uint8_t byte : bytes) {
    out.push_back(kDigits[byte >> 4U]);
    out.push_back(kDigits[byte & 0x0FU]);
  }
  return out;
}

void check_bytes(std::span<const std::uint8_t> actual, const char* golden,
                 int line) {
  const std::vector<std::uint8_t> expected = from_hex(golden);
  ++g_checks;
  if (actual.size() != expected.size() ||
      !std::equal(actual.begin(), actual.end(), expected.begin())) {
    ++g_failures;
    std::fprintf(stderr, "FAIL line %d: golden mismatch\n  expected %s\n  actual   %s\n",
                 line, golden, to_hex(actual).c_str());
  }
}

#define CHECK_BYTES(actual, golden) check_bytes((actual), (golden), __LINE__)

// Pinned goldens from golden_reference.py.
constexpr const char* kGoldenBaseline64 =
    "13048075002abffe0028006e23800000101b8082";
constexpr const char* kGoldenM1Inferred64 =
    "1404807500755fff000a001b88e000000406e02080";
constexpr const char* kGoldenM1Explicit64 =
    "1404807500755fff1c028006e23800000101b80820";
constexpr const char* kGoldenM2T2Multi320 =
    "1704807504ac3cfecc7fffff00a001b88e000000406e0208";
constexpr const char* kGoldenM2T2Single320 =
    "1604807504ac3cfecc7f7c028006e23800000101b80820";
constexpr const char* kGoldenCompatM1_64 =
    "13048075006abffe0014003711c00000080dc041";
constexpr const char* kGoldenM7Inferred64 =
    "1404807501fd57ffc0028006e23800000101b80820";

void test_golden_sequence_headers() {
  std::string error;

  SequenceHeaderConfig baseline = baseline_sequence_header_config(64, 64, 90, 2);
  std::vector<std::uint8_t> out;
  CHECK(make_sequence_header_obu(baseline, SequenceSyntaxDialect::kSpecV1_0_0,
                                 out, error));
  CHECK(error.empty());
  CHECK_BYTES(out, kGoldenBaseline64);

  // Byte-identical to the M2 baseline writer for max_mlayer_id = 0.
  av2enc::m2::M2Config m2;
  m2.max_frame_width = 64;
  m2.max_frame_height = 64;
  m2.base_q_idx = 90;
  m2.order_hint_bits = 4;
  m2.num_ref_frames = 2;
  CHECK_BYTES(av2enc::m2::make_sequence_header_obu(m2), kGoldenBaseline64);

  SequenceHeaderConfig m1_inferred = baseline;
  m1_inferred.layers = default_sequence_layer_config(1, 0);
  CHECK(make_sequence_header_obu(m1_inferred,
                                 SequenceSyntaxDialect::kSpecV1_0_0, out,
                                 error));
  CHECK_BYTES(out, kGoldenM1Inferred64);

  SequenceHeaderConfig m1_explicit = m1_inferred;
  m1_explicit.layers.mlayer_dependency_present_flag = true;
  CHECK(make_sequence_header_obu(m1_explicit,
                                 SequenceSyntaxDialect::kSpecV1_0_0, out,
                                 error));
  CHECK_BYTES(out, kGoldenM1Explicit64);

  SequenceHeaderConfig m2t2 = baseline_sequence_header_config(320, 180, 90, 2);
  m2t2.layers = default_sequence_layer_config(2, 2);
  m2t2.layers.mlayer_dependency_present_flag = true;
  m2t2.layers.tlayer_dependency_present_flag = true;
  m2t2.layers.multi_tlayer_dependency_map_present_flag = true;
  CHECK(make_sequence_header_obu(m2t2, SequenceSyntaxDialect::kSpecV1_0_0, out,
                                 error));
  CHECK_BYTES(out, kGoldenM2T2Multi320);

  SequenceHeaderConfig m2t2_single = m2t2;
  m2t2_single.layers.multi_tlayer_dependency_map_present_flag = false;
  // With multi = 0, every embedded layer must copy TLayerDependencyMap[0].
  for (int m = 1; m <= 2; ++m) {
    m2t2_single.layers.dependencies.tlayer[static_cast<std::size_t>(m)] =
        m2t2_single.layers.dependencies.tlayer[0];
  }
  CHECK(make_sequence_header_obu(m2t2_single,
                                 SequenceSyntaxDialect::kSpecV1_0_0, out,
                                 error));
  CHECK_BYTES(out, kGoldenM2T2Single320);

  SequenceHeaderConfig compat = m1_inferred;
  CHECK(make_sequence_header_obu(compat,
                                 SequenceSyntaxDialect::kPinnedDav2dCompat, out,
                                 error));
  CHECK_BYTES(out, kGoldenCompatM1_64);

  SequenceHeaderConfig m7 = baseline;
  m7.layers = default_sequence_layer_config(7, 0);
  CHECK(make_sequence_header_obu(m7, SequenceSyntaxDialect::kSpecV1_0_0, out,
                                 error));
  CHECK_BYTES(out, kGoldenM7Inferred64);
}

void test_reader_round_trip() {
  std::string error;
  SequenceHeaderConfig config = baseline_sequence_header_config(320, 180, 90, 2);
  config.layers = default_sequence_layer_config(2, 2);
  config.layers.mlayer_dependency_present_flag = true;
  config.layers.tlayer_dependency_present_flag = true;
  config.layers.multi_tlayer_dependency_map_present_flag = true;
  // Non-default map: layer 2 depends only on layer 1 (plus itself), and layer 1
  // depends only on layer 0 (plus itself). This is transitive.
  config.layers.dependencies.mlayer[1][0] = true;
  config.layers.dependencies.mlayer[2][0] = true;
  config.layers.dependencies.mlayer[2][1] = false;
  // Per-embedded-layer tlayer maps: m1 uses a direct chain t0 <- t1 <- t2.
  config.layers.dependencies.tlayer[1][1][0] = false;
  config.layers.dependencies.tlayer[1][2][0] = false;
  config.layers.dependencies.tlayer[1][2][1] = true;

  std::vector<std::uint8_t> obu;
  CHECK(make_sequence_header_obu(config, SequenceSyntaxDialect::kSpecV1_0_0,
                                 obu, error));
  ParsedSequenceHeader parsed;
  CHECK(parse_sequence_header_obu(obu, SequenceSyntaxDialect::kSpecV1_0_0,
                                  parsed, error));
  CHECK(parsed.max_mlayer_id == 2);
  CHECK(parsed.max_tlayer_id == 2);
  CHECK(parsed.seq_max_mlayer_cnt == 3);
  CHECK(parsed.seq_max_mlayer_cnt_coded);
  CHECK(parsed.max_frame_width == 320);
  CHECK(parsed.max_frame_height == 180);
  CHECK(parsed.mlayer_dependency_present_flag);
  CHECK(parsed.tlayer_dependency_present_flag);
  CHECK(parsed.multi_tlayer_dependency_map_present_flag);
  CHECK(parsed.dependencies.mlayer == config.layers.dependencies.mlayer);
  CHECK(parsed.dependencies.tlayer == config.layers.dependencies.tlayer);
  const MLayerPresenceMap expected_presence =
      derive_mlayer_presence_map(config.layers.dependencies.mlayer, 2);
  CHECK(parsed.presence == expected_presence);
  CHECK(parsed.order_hint_bits == 4);
  CHECK(parsed.num_ref_frames == 2);
  // m0 presence: {0}; m1 depends on 0 -> {0,1}; m2 depends on 1 -> {0,1,2}
  CHECK(parsed.presence[0][0]);
  CHECK(!parsed.presence[0][1]);
  CHECK(parsed.presence[1][0]);
  CHECK(parsed.presence[2][0]);
  CHECK(!parsed.presence[2][1]);
  CHECK(parsed.presence[2][2]);

  // The compat dialect has no count field but still parses the same header
  // prefix.
  SequenceHeaderConfig compat = baseline_sequence_header_config(64, 64, 90, 2);
  compat.layers = default_sequence_layer_config(1, 0);
  CHECK(make_sequence_header_obu(compat,
                                 SequenceSyntaxDialect::kPinnedDav2dCompat, obu,
                                 error));
  CHECK(parse_sequence_header_obu(obu,
                                  SequenceSyntaxDialect::kPinnedDav2dCompat,
                                  parsed, error));
  CHECK(parsed.max_mlayer_id == 1);
  CHECK(!parsed.seq_max_mlayer_cnt_coded);
  CHECK(parsed.seq_max_mlayer_cnt == 1);
  CHECK(parsed.dialect == SequenceSyntaxDialect::kPinnedDav2dCompat);
}

void test_seq_max_mlayer_cnt_round_trip() {
  std::string error;
  // Exercise every legal (max_mlayer_id, SeqMaxMlayerCnt) pair and therefore
  // every n = CeilLog2(max_mlayer_id + 1) width.
  for (int max_m = 1; max_m <= kMaxMlayerId; ++max_m) {
    for (int count = 1; count <= max_m + 1; ++count) {
      SequenceHeaderConfig config =
          baseline_sequence_header_config(64, 64, 90, 2);
      config.layers = default_sequence_layer_config(max_m, 0);
      config.layers.seq_max_mlayer_cnt = count;
      CHECK(validate_sequence_layer_config(config.layers, error));
      std::vector<std::uint8_t> obu;
      CHECK(make_sequence_header_obu(config, SequenceSyntaxDialect::kSpecV1_0_0,
                                     obu, error));
      ParsedSequenceHeader parsed;
      CHECK(parse_sequence_header_obu(obu, SequenceSyntaxDialect::kSpecV1_0_0,
                                      parsed, error));
      CHECK(parsed.max_mlayer_id == max_m);
      CHECK(parsed.seq_max_mlayer_cnt_coded);
      CHECK(parsed.seq_max_mlayer_cnt == count);
    }
  }
}

void test_obu_headers() {
  std::string error;
  std::vector<std::uint8_t> out;
  CHECK(make_obu_header(1, 0, 0, 0, out, error));
  CHECK_BYTES(out, "04");
  CHECK(make_obu_header(2, 0, 0, 31, out, error));
  CHECK_BYTES(out, "08");
  CHECK(make_obu_header(20, 0, 0, 31, out, error));
  CHECK_BYTES(out, "50");
  CHECK(make_obu_header(4, 0, 1, 0, out, error));
  CHECK_BYTES(out, "9020");
  CHECK(make_obu_header(4, 0, 0, 5, out, error));
  CHECK_BYTES(out, "9005");
  CHECK(make_obu_header(8, 0, 0, 31, out, error));
  CHECK_BYTES(out, "a01f");

  // §5.2.2 inference with extension_flag = 0.
  ObuHeaderFields parsed;
  std::size_t header_bytes = 0;
  const std::vector<std::uint8_t> td = from_hex("08");
  CHECK(parse_obu_header(td, 0, parsed, header_bytes, error));
  CHECK(!parsed.extension_flag);
  CHECK(parsed.obu_type == 2);
  CHECK(parsed.mlayer_id == 0);
  CHECK(parsed.xlayer_id == kGlobalXLayerId);
  CHECK(header_bytes == 1);

  const std::vector<std::uint8_t> clk = from_hex("9020");
  CHECK(parse_obu_header(clk, 0, parsed, header_bytes, error));
  CHECK(parsed.extension_flag);
  CHECK(parsed.obu_type == 4);
  CHECK(parsed.mlayer_id == 1);
  CHECK(parsed.xlayer_id == 0);
  CHECK(header_bytes == 2);

  // Invalid combinations.
  CHECK(!make_obu_header(2, 0, 0, 0, out, error));   // TD must be global
  CHECK(!make_obu_header(1, 0, 1, 0, out, error));   // SEQ forced mlayer 0
  CHECK(!make_obu_header(4, 1, 0, 0, out, error));   // CLK forced tlayer 0
  CHECK(!make_obu_header(4, 0, 0, 31, out, error));  // global frame OBU
  // LCR with GLOBAL_XLAYER_ID and mlayer 0 / tlayer 0 is valid.
  CHECK(make_obu_header(16, 0, 0, 31, out, error));

  const ObuHeaderFields inferred_td = inferred_obu_header_fields(2, 0);
  CHECK(inferred_td.xlayer_id == kGlobalXLayerId);
  const ObuHeaderFields inferred_seq = inferred_obu_header_fields(1, 0);
  CHECK(inferred_seq.xlayer_id == 0);
}

// ---------------------------------------------------------------------------
// Exhaustive dependency-map validation
// ---------------------------------------------------------------------------

std::uint32_t triangle_bit_count(int maximum) {
  std::uint32_t bits = 0;
  for (int curr = 1; curr <= maximum; ++curr) {
    bits += static_cast<std::uint32_t>(curr + 1);
  }
  return bits;
}

MLayerDependencyMap mlayer_map_from_bits(int maximum, std::uint32_t bits) {
  MLayerDependencyMap map{};
  for (int curr = 0; curr <= maximum; ++curr) {
    map[static_cast<std::size_t>(curr)][static_cast<std::size_t>(curr)] = true;
  }
  for (int curr = 1; curr <= maximum; ++curr) {
    for (int ref = curr - 1; ref >= 0; --ref) {
      const bool value = (bits & 1U) != 0U;
      bits >>= 1U;
      map[static_cast<std::size_t>(curr)][static_cast<std::size_t>(ref)] =
          value;
    }
  }
  return map;
}

TLayerDependencyMap tlayer_map_from_bits(int maximum, std::uint32_t bits) {
  TLayerDependencyMap map{};
  for (int curr = 0; curr <= maximum; ++curr) {
    map[static_cast<std::size_t>(curr)][static_cast<std::size_t>(curr)] = true;
  }
  for (int curr = 1; curr <= maximum; ++curr) {
    for (int ref = curr - 1; ref >= 0; --ref) {
      const bool value = (bits & 1U) != 0U;
      bits >>= 1U;
      map[static_cast<std::size_t>(curr)][static_cast<std::size_t>(ref)] =
          value;
    }
  }
  return map;
}

bool transitive_mlayer(const MLayerDependencyMap& map, int maximum) {
  for (int a = 0; a <= maximum; ++a) {
    for (int b = 0; b <= maximum; ++b) {
      if (!map[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)]) {
        continue;
      }
      for (int c = 0; c <= maximum; ++c) {
        if (map[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)] &&
            !map[static_cast<std::size_t>(a)][static_cast<std::size_t>(c)]) {
          return false;
        }
      }
    }
  }
  return true;
}

bool transitive_tlayer(const TLayerDependencyMap& map, int maximum) {
  for (int a = 0; a <= maximum; ++a) {
    for (int b = 0; b <= maximum; ++b) {
      if (!map[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)]) {
        continue;
      }
      for (int c = 0; c <= maximum; ++c) {
        if (map[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)] &&
            !map[static_cast<std::size_t>(a)][static_cast<std::size_t>(c)]) {
          return false;
        }
      }
    }
  }
  return true;
}

MLayerPresenceMap brute_force_presence(const MLayerDependencyMap& map,
                                       int maximum) {
  MLayerPresenceMap presence{};
  // §5.4.1 p58 sets the identity for every mlayerId in 0..7, including layers
  // above max_mlayer_id; only dependency bits are limited by the maximum.
  for (int a = 0; a < kMlayerCount; ++a) {
    for (int b = 0; b < kMlayerCount; ++b) {
      presence[static_cast<std::size_t>(a)]
              [static_cast<std::size_t>(b)] =
          a == b || (a <= maximum && b <= maximum &&
                     map[static_cast<std::size_t>(a)]
                        [static_cast<std::size_t>(b)]);
    }
  }
  // Floyd-Warshall closure over the full 8x8 presence map.
  for (int k = 0; k < kMlayerCount; ++k) {
    for (int i = 0; i < kMlayerCount; ++i) {
      if (!presence[static_cast<std::size_t>(i)]
                   [static_cast<std::size_t>(k)]) {
        continue;
      }
      for (int j = 0; j < kMlayerCount; ++j) {
        if (presence[static_cast<std::size_t>(k)]
                    [static_cast<std::size_t>(j)]) {
          presence[static_cast<std::size_t>(i)]
                  [static_cast<std::size_t>(j)] = true;
        }
      }
    }
  }
  return presence;
}

void test_exhaustive_mlayer_maps() {
  constexpr int kMax = 3;
  const std::uint32_t bits = triangle_bit_count(kMax);
  CHECK(bits == 9);
  std::uint32_t valid = 0;
  for (std::uint32_t value = 0; value < (1U << bits); ++value) {
    const MLayerDependencyMap map = mlayer_map_from_bits(kMax, value);
    std::string error;
    const bool accepted =
        validate_mlayer_dependency_map(map, kMax, error);
    const bool expected = transitive_mlayer(map, kMax);
    CHECK(accepted == expected);
    if (accepted) ++valid;
    const MLayerPresenceMap presence =
        derive_mlayer_presence_map(map, kMax);
    const MLayerPresenceMap expected_presence =
        brute_force_presence(map, kMax);
    CHECK(presence == expected_presence);
  }
  CHECK(valid > 0);

  // Reflexivity failures are rejected.
  for (int curr = 0; curr <= kMax; ++curr) {
    MLayerDependencyMap map =
        default_dependency_maps(kMax, 0).mlayer;
    map[static_cast<std::size_t>(curr)][static_cast<std::size_t>(curr)] =
        false;
    std::string error;
    CHECK(!validate_mlayer_dependency_map(map, kMax, error));
  }
  // A dependency above max_mlayer_id is rejected.
  {
    MLayerDependencyMap map = default_dependency_maps(kMax, 0).mlayer;
    map[3][0] = true;
    std::string error;
    CHECK(!validate_mlayer_dependency_map(map, 2, error));
  }
  // An upper-triangular bit is rejected.
  {
    MLayerDependencyMap map = default_dependency_maps(kMax, 0).mlayer;
    map[0][1] = true;
    std::string error;
    CHECK(!validate_mlayer_dependency_map(map, kMax, error));
  }
}

void test_exhaustive_tlayer_maps() {
  // max_tlayer_id = 2 has 5 lower-triangular bits (curr=1: 2, curr=2: 3).
  constexpr int kMaxT = 2;
  const std::uint32_t bits = triangle_bit_count(kMaxT);
  CHECK(bits == 5);

  std::uint32_t valid = 0;
  for (std::uint32_t value = 0; value < (1U << bits); ++value) {
    const TLayerDependencyMap map = tlayer_map_from_bits(kMaxT, value);
    TLayerDependencyMaps maps{};
    maps[0] = map;
    std::string error;
    const bool accepted =
        validate_tlayer_dependency_maps(maps, 0, kMaxT, error);
    const bool expected = transitive_tlayer(map, kMaxT);
    CHECK(accepted == expected);
    if (accepted) ++valid;
  }
  CHECK(valid > 0);

  // Exhaustive cross-product for two embedded layers, max_mlayer_id = 1.
  constexpr int kMaxM = 1;
  std::uint32_t combinations = 0;
  std::uint32_t accepted_combinations = 0;
  for (std::uint32_t a = 0; a < (1U << bits); ++a) {
    for (std::uint32_t b = 0; b < (1U << bits); ++b) {
      TLayerDependencyMaps maps{};
      maps[0] = tlayer_map_from_bits(kMaxT, a);
      maps[1] = tlayer_map_from_bits(kMaxT, b);
      std::string error;
      const bool accepted =
          validate_tlayer_dependency_maps(maps, kMaxM, kMaxT, error);
      const bool expected =
          transitive_tlayer(maps[0], kMaxT) &&
          transitive_tlayer(maps[1], kMaxT);
      CHECK(accepted == expected);
      ++combinations;
      if (accepted) ++accepted_combinations;
    }
  }
  CHECK(combinations == (1U << (2U * bits)));

  // Exhaustive cross-product for three embedded layers, max_mlayer_id = 2.
  combinations = 0;
  for (std::uint32_t a = 0; a < (1U << bits); ++a) {
    for (std::uint32_t b = 0; b < (1U << bits); ++b) {
      for (std::uint32_t c = 0; c < (1U << bits); ++c) {
        TLayerDependencyMaps maps{};
        maps[0] = tlayer_map_from_bits(kMaxT, a);
        maps[1] = tlayer_map_from_bits(kMaxT, b);
        maps[2] = tlayer_map_from_bits(kMaxT, c);
        std::string error;
        const bool accepted =
            validate_tlayer_dependency_maps(maps, 2, kMaxT, error);
        const bool expected =
            transitive_tlayer(maps[0], kMaxT) &&
            transitive_tlayer(maps[1], kMaxT) &&
            transitive_tlayer(maps[2], kMaxT);
        CHECK(accepted == expected);
        ++combinations;
      }
    }
  }
  CHECK(combinations == (1U << (3U * bits)));
  CHECK(accepted_combinations > 0);

  // Reflexivity failure.
  {
    TLayerDependencyMaps maps = default_dependency_maps(1, kMaxT).tlayer;
    maps[1][2][2] = false;
    std::string error;
    CHECK(!validate_tlayer_dependency_maps(maps, 1, kMaxT, error));
  }
  // A map for an undeclared embedded layer is rejected.
  {
    TLayerDependencyMaps maps = default_dependency_maps(1, kMaxT).tlayer;
    maps[2][0][0] = true;
    std::string error;
    CHECK(!validate_tlayer_dependency_maps(maps, 1, kMaxT, error));
  }
}

void test_sequence_layer_config_validation() {
  std::string error;
  SequenceLayerConfig config = default_sequence_layer_config(2, 2);
  CHECK(validate_sequence_layer_config(config, error));

  config = default_sequence_layer_config(2, 2);
  config.seq_max_mlayer_cnt = 0;
  CHECK(!validate_sequence_layer_config(config, error));

  config = default_sequence_layer_config(2, 2);
  config.seq_max_mlayer_cnt = 4;
  CHECK(!validate_sequence_layer_config(config, error));

  config = default_sequence_layer_config(0, 0);
  config.seq_max_mlayer_cnt = 2;
  CHECK(!validate_sequence_layer_config(config, error));

  config = default_sequence_layer_config(0, 0);
  config.mlayer_dependency_present_flag = true;
  CHECK(!validate_sequence_layer_config(config, error));

  config = default_sequence_layer_config(0, 0);
  config.tlayer_dependency_present_flag = true;
  CHECK(!validate_sequence_layer_config(config, error));

  config = default_sequence_layer_config(1, 2);
  config.multi_tlayer_dependency_map_present_flag = true;
  CHECK(!validate_sequence_layer_config(config, error));

  config = default_sequence_layer_config(2, 2);
  config.mlayer_dependency_present_flag = false;
  config.dependencies.mlayer[2][0] = false;
  CHECK(!validate_sequence_layer_config(config, error));

  config = default_sequence_layer_config(2, 2);
  config.tlayer_dependency_present_flag = false;
  config.dependencies.tlayer[1][2][0] = false;
  CHECK(!validate_sequence_layer_config(config, error));

  config = default_sequence_layer_config(1, 2);
  config.tlayer_dependency_present_flag = true;
  config.multi_tlayer_dependency_map_present_flag = false;
  config.dependencies.tlayer[1][2][0] = false;
  CHECK(!validate_sequence_layer_config(config, error));

  config = default_sequence_layer_config(1, 2);
  config.tlayer_dependency_present_flag = true;
  config.multi_tlayer_dependency_map_present_flag = true;
  // A transitive direct chain for m1: t2 -> t1 -> t0, without t2 -> t0.
  config.dependencies.tlayer[1][1][0] = false;
  config.dependencies.tlayer[1][2][1] = true;
  config.dependencies.tlayer[1][2][0] = false;
  CHECK(validate_sequence_layer_config(config, error));

  // MaxMlayerCnt can be below max_mlayer_id + 1 (§6.4.1 p261 only caps the
  // number of distinct values).
  config = default_sequence_layer_config(7, 0);
  config.seq_max_mlayer_cnt = 1;
  CHECK(validate_sequence_layer_config(config, error));
}

// ---------------------------------------------------------------------------
// Stream ordering / availability
// ---------------------------------------------------------------------------

LayerObuRecord td_record() {
  LayerObuRecord r;
  r.obu_type = 2;
  r.layer = LayerId{31, 0, 0};
  return r;
}

LayerObuRecord seq_record(int max_mlayer, int max_tlayer) {
  LayerObuRecord r;
  r.obu_type = 1;
  r.layer = LayerId{0, 0, 0};
  r.seq_header_id = 0;
  r.header_max_mlayer_id = max_mlayer;
  r.header_max_tlayer_id = max_tlayer;
  return r;
}

LayerObuRecord clk_record(int mlayer, int order_hint = 0) {
  LayerObuRecord r;
  r.obu_type = 4;
  r.layer = LayerId{0, static_cast<std::uint8_t>(mlayer), 0};
  r.key_frame = true;
  r.coded_output_frame = true;
  r.order_hint = order_hint;
  r.seq_header_ref = 0;
  return r;
}

void test_stream_auditor() {
  std::string error;

  {
    LayerStreamAuditor audit(default_sequence_layer_config(1, 0));
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 0), error));
    CHECK(audit.observe(clk_record(0), error));
    CHECK(audit.observe(clk_record(1), error));
    CHECK(audit.finish(error));
  }

  // Descending embedded-layer order.
  {
    LayerStreamAuditor audit(default_sequence_layer_config(1, 0));
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 0), error));
    CHECK(audit.observe(clk_record(1), error));
    CHECK(audit.observe(clk_record(0), error));
    CHECK(!audit.finish(error));
  }

  // Missing sequence header.
  {
    LayerStreamAuditor audit(default_sequence_layer_config(1, 0));
    CHECK(audit.observe(td_record(), error));
    CHECK(!audit.observe(clk_record(0), error));
  }

  // Frame tlayer above max_tlayer_id.
  {
    LayerStreamAuditor audit(default_sequence_layer_config(1, 1));
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 1), error));
    LayerObuRecord frame = clk_record(0);
    frame.layer.tlayer_id = 2;
    CHECK(!audit.observe(frame, error));
  }

  // A frame OBU cannot use GLOBAL_XLAYER_ID.
  {
    LayerStreamAuditor audit(default_sequence_layer_config(1, 0));
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 0), error));
    LayerObuRecord frame = clk_record(0);
    frame.layer.xlayer_id = 31;
    CHECK(!audit.observe(frame, error));
  }

  // Two output frame units in the same embedded layer.
  {
    LayerStreamAuditor audit(default_sequence_layer_config(1, 0));
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 0), error));
    CHECK(audit.observe(clk_record(0), error));
    CHECK(audit.observe(clk_record(0), error));
    CHECK(!audit.finish(error));
  }

  // Output frame units in one extended layer unit must share OrderHint.
  {
    LayerStreamAuditor audit(default_sequence_layer_config(1, 0));
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 0), error));
    CHECK(audit.observe(clk_record(0, 0), error));
    CHECK(audit.observe(clk_record(1, 1), error));
    CHECK(!audit.finish(error));
  }

  // Lowest present embedded layer must start with CLK when the unit has a CLK.
  {
    LayerStreamAuditor audit(default_sequence_layer_config(1, 0));
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 0), error));
    LayerObuRecord frame = clk_record(0);
    frame.key_frame = false;
    frame.obu_type = 7;
    CHECK(audit.observe(frame, error));
    CHECK(audit.observe(clk_record(1), error));
    CHECK(!audit.finish(error));
  }

  // Non-output frame without an output frame in the same layer.
  {
    LayerStreamAuditor audit(default_sequence_layer_config(1, 0));
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 0), error));
    CHECK(audit.observe(clk_record(0), error));
    LayerObuRecord hidden = clk_record(1);
    hidden.coded_output_frame = false;
    CHECK(audit.observe(hidden, error));
    CHECK(!audit.finish(error));
  }

  // Inter-layer reference to a higher layer is rejected.
  {
    LayerStreamAuditor audit(default_sequence_layer_config(1, 0));
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 0), error));
    LayerObuRecord frame = clk_record(0);
    frame.has_inter_layer_ref = true;
    frame.inter_layer_ref = LayerId{0, 1, 0};
    CHECK(!audit.observe(frame, error));
  }

  // Inter-layer reference to a layer outside MLayerPresenceMap is rejected.
  {
    SequenceLayerConfig config = default_sequence_layer_config(1, 0);
    config.dependencies.mlayer[1][0] = false;
    config.mlayer_dependency_present_flag = true;
    LayerStreamAuditor audit(config);
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 0), error));
    CHECK(audit.observe(clk_record(0), error));
    LayerObuRecord frame = clk_record(1);
    frame.has_inter_layer_ref = true;
    frame.inter_layer_ref = LayerId{0, 0, 0};
    CHECK(!audit.observe(frame, error));
  }

  // Inter-layer reference needs an earlier output frame in the same TU.
  {
    LayerStreamAuditor audit(default_sequence_layer_config(1, 0));
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 0), error));
    LayerObuRecord frame = clk_record(1);
    frame.has_inter_layer_ref = true;
    frame.inter_layer_ref = LayerId{0, 0, 0};
    CHECK(!audit.observe(frame, error));
  }

  // SWITCH/RAS at mlayer > 0 must not depend on another embedded layer.
  {
    LayerStreamAuditor audit(default_sequence_layer_config(1, 0));
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 0), error));
    LayerObuRecord sw;
    sw.obu_type = 10;
    sw.layer = LayerId{0, 1, 0};
    sw.switch_or_ras = true;
    sw.coded_output_frame = true;
    sw.order_hint = 0;
    sw.seq_header_ref = 0;
    CHECK(!audit.observe(sw, error));
  }

  // Number of distinct embedded layers must fit SeqMaxMlayerCnt.
  {
    SequenceLayerConfig config = default_sequence_layer_config(1, 0);
    config.seq_max_mlayer_cnt = 1;
    LayerStreamAuditor audit(config);
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 0), error));
    CHECK(audit.observe(clk_record(0), error));
    CHECK(audit.observe(clk_record(1), error));
    CHECK(!audit.finish(error));
  }
  {
    SequenceLayerConfig config = default_sequence_layer_config(1, 0);
    config.seq_max_mlayer_cnt = 2;
    LayerStreamAuditor audit(config);
    CHECK(audit.observe(td_record(), error));
    CHECK(audit.observe(seq_record(1, 0), error));
    CHECK(audit.observe(clk_record(1), error));
    CHECK(audit.finish(error));
  }

  // An empty temporal unit is rejected.
  {
    LayerStreamAuditor audit(default_sequence_layer_config(1, 0));
    CHECK(audit.observe(td_record(), error));
    CHECK(!audit.observe(td_record(), error));
    CHECK(audit.finish(error));
  }
}

}  // namespace

int main() {
  test_golden_sequence_headers();
  test_reader_round_trip();
  test_seq_max_mlayer_cnt_round_trip();
  test_obu_headers();
  test_exhaustive_mlayer_maps();
  test_exhaustive_tlayer_maps();
  test_sequence_layer_config_validation();
  test_stream_auditor();

  std::printf("mlayers_test: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
