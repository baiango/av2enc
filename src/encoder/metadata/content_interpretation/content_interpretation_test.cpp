// Unit tests for the AV2 v1.0.0 content interpretation OBU.
//
// Build with src/encoder/metadata/content_interpretation/build.sh (this agent
// owns only this directory, so there is no CMake target). The byte goldens are
// pinned from the §5.15 / §5.2.1 / §5.4.12 syntax, computed independently of
// the implementation and cross-checked with the pinned dav2d build by
// run_ci_gate.sh.

#include "encoder/metadata/content_interpretation/content_interpretation.h"

#include "bitstream/bitwriter.h"
#include "bitstream/m0_headers.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace av2enc;
using namespace av2enc::ci;

int g_failures = 0;

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      ++g_failures;                                                           \
      std::cerr << __FILE__ << ':' << __LINE__                                \
                << ": CHECK failed: " #expression "\n";                       \
    }                                                                         \
  } while (false)

#define CHECK_HEX(got, expected)                                              \
  do {                                                                        \
    const std::string got_hex = to_hex(got);                                  \
    const std::string want_hex = (expected);                                  \
    if (got_hex != want_hex) {                                                \
      ++g_failures;                                                           \
      std::cerr << __FILE__ << ':' << __LINE__ << ": hex mismatch\n  got  "   \
                << got_hex << "\n  want " << want_hex << '\n';                \
    }                                                                         \
  } while (false)

std::string to_hex(std::span<const std::uint8_t> bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 3U);
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    const std::uint8_t byte = bytes[index];
    out.push_back(kDigits[byte >> 4U]);
    out.push_back(kDigits[byte & 0x0FU]);
    if (index + 1U != bytes.size()) {
      out.push_back(' ');
    }
  }
  return out;
}

// Independent payload builder. The module writer rejects invalid input, so the
// negative tests need their own encoder.
class ManualPayload {
 public:
  void put(std::uint32_t value, unsigned bits) {
    for (unsigned index = bits; index-- > 0U;) {
      bits_.push_back(((value >> index) & 1U) != 0U);
    }
  }
  void bit(bool value) { bits_.push_back(value); }
  void uvlc(std::uint32_t value) {
    unsigned leading_zeros = 0;
    while (static_cast<std::uint64_t>(value) >=
           ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
      ++leading_zeros;
    }
    put(0U, leading_zeros);
    bit(true);
    put(value - static_cast<std::uint32_t>((1ULL << leading_zeros) - 1ULL),
        leading_zeros);
  }
  void rg(std::uint32_t value, unsigned n) {
    const std::uint32_t prefix = value >> n;
    for (std::uint32_t index = 0; index < prefix; ++index) {
      bit(true);
    }
    bit(false);
    put(value & ((1UL << n) - 1UL), n);
  }
  std::vector<std::uint8_t> finish(bool good_trailing = true) {
    bit(false);          // obu_extension_flag
    bit(good_trailing);  // trailing_bits marker
    while ((bits_.size() & 7U) != 0U) {
      bit(false);
    }
    std::vector<std::uint8_t> out(bits_.size() / 8U, 0U);
    for (std::size_t index = 0; index < bits_.size(); ++index) {
      if (bits_[index]) {
        out[index / 8U] |= static_cast<std::uint8_t>(0x80U >> (index & 7U));
      }
    }
    return out;
  }

 private:
  std::vector<bool> bits_;
};

std::vector<std::uint8_t> make_obu(std::uint8_t type,
                                   std::span<const std::uint8_t> payload,
                                   bool extension = false,
                                   std::uint8_t xlayer_id = 0) {
  m0::ObuHeader header;
  header.type = static_cast<m0::ObuType>(type);
  header.tlayer_id = 0;
  header.extension_flag = extension;
  header.xlayer_id = xlayer_id;
  return m0::make_obu(header, payload);
}

std::vector<std::uint8_t> concat(
    std::initializer_list<std::span<const std::uint8_t>> parts) {
  std::vector<std::uint8_t> out;
  for (const auto part : parts) {
    out.insert(out.end(), part.begin(), part.end());
  }
  return out;
}

// Golden fixtures. Each entry is {description, params, payload hex, OBU hex}.
struct Golden {
  const char* name;
  ContentInterpretation params;
  const char* payload_hex;
  const char* obu_hex;
};

ContentInterpretation golden_a() {
  return default_content_interpretation();
}

ContentInterpretation golden_b() {
  ContentInterpretation params;
  params.scan_type = ScanType::kProgressive;
  params.has_color_description = true;
  params.color = derive_color_description(1);  // BT.709 SDR
  params.has_chroma_sample_position = true;
  params.chroma_top = ChromaSamplePosition::kCenter;
  params.chroma_bottom = ChromaSamplePosition::kCenter;
  params.has_aspect_ratio = true;
  params.aspect.idc = 1;  // 1:1
  params.aspect.sar_width = 1;
  params.aspect.sar_height = 1;
  params.has_timing_info = true;
  params.timing.num_units_in_display_tick = 1;
  params.timing.time_scale = 30;
  params.timing.equal_picture_interval = true;
  params.timing.num_ticks_per_picture_minus_1 = 0;
  return params;
}

ContentInterpretation golden_c() {
  ContentInterpretation params;
  params.scan_type = ScanType::kInterlaceField;
  params.has_color_description = true;
  params.color.idc = 0;  // explicit
  params.color.primaries = 9;
  params.color.transfer = 16;
  params.color.matrix = 9;
  params.color.full_range = true;
  params.has_chroma_sample_position = true;
  params.chroma_top = ChromaSamplePosition::kLeft;
  params.chroma_bottom = ChromaSamplePosition::kBottom;
  params.has_aspect_ratio = true;
  params.aspect.idc = 255;
  params.aspect.sar_width = 16;
  params.aspect.sar_height = 11;
  return params;
}

ContentInterpretation golden_d() {
  ContentInterpretation params;
  params.scan_type = ScanType::kInterlaceComplementaryPair;
  params.has_chroma_sample_position = true;
  params.chroma_top = ChromaSamplePosition::kTopLeft;
  params.chroma_bottom = ChromaSamplePosition::kTop;
  params.has_aspect_ratio = true;
  params.aspect.idc = 14;  // 4:3
  params.aspect.sar_width = 4;
  params.aspect.sar_height = 3;
  return params;
}

ContentInterpretation golden_e() {
  ContentInterpretation params;
  params.scan_type = ScanType::kProgressive;
  params.has_chroma_sample_position = true;
  params.chroma_top = ChromaSamplePosition::kCenter;
  params.chroma_bottom = ChromaSamplePosition::kCenter;
  return params;
}

ContentInterpretation golden_f() {
  ContentInterpretation params;
  params.scan_type = ScanType::kUnspecified;
  params.has_timing_info = true;
  params.timing.num_units_in_display_tick = 1001;
  params.timing.time_scale = 30000;
  params.timing.equal_picture_interval = false;
  return params;
}

ContentInterpretation golden_g() {
  ContentInterpretation params;
  params.scan_type = ScanType::kProgressive;
  params.has_color_description = true;
  params.color = derive_color_description(5);  // sYCC, exercises rg(2) q=1
  params.has_chroma_sample_position = true;
  params.chroma_top = ChromaSamplePosition::kCenter;
  params.chroma_bottom = ChromaSamplePosition::kCenter;
  params.has_aspect_ratio = true;
  params.aspect.idc = 255;
  params.aspect.sar_width = 160;
  params.aspect.sar_height = 99;
  params.has_timing_info = true;
  params.timing.num_units_in_display_tick = 1;
  params.timing.time_scale = 60;
  params.timing.equal_picture_interval = true;
  params.timing.num_ticks_per_picture_minus_1 = 1;
  return params;
}

// Reserved colour idc 7 is spec-legal to parse but reserved for AOMedia.
ContentInterpretation golden_h() {
  ContentInterpretation params;
  params.scan_type = ScanType::kUnspecified;
  params.has_color_description = true;
  params.color.idc = 7;
  params.color.primaries = 2;
  params.color.transfer = 2;
  params.color.matrix = 2;
  params.color.full_range = false;
  return params;
}

ContentInterpretation golden_i() {
  ContentInterpretation params;
  params.scan_type = ScanType::kInterlaceField;
  params.has_chroma_sample_position = true;
  params.chroma_top = ChromaSamplePosition::kLeft;
  params.chroma_bottom = ChromaSamplePosition::kBottomLeft;
  params.has_aspect_ratio = true;
  params.aspect.idc = 0;  // table row 0 is 0:0 -> unspecified
  return params;
}

const Golden kGoldens[] = {
    {"A minimal", golden_a(), "00 40", "03 60 00 40"},
    {"B progressive/colour/chroma/aspect/timing", golden_b(),
     "7c 24 02 00 00 00 02 00 00 00 3d a0",
     "0d 60 7c 24 02 00 00 00 02 00 00 00 3d a0"},
    {"C interlace explicit colour/SAR", golden_c(),
     "b8 01 22 01 39 bf c2 23 10",
     "0a 60 b8 01 22 01 39 bf c2 23 10"},
    {"D complementary pair", golden_d(), "d8 64 0e 40", "05 60 d8 64 0e 40"},
    {"E progressive chroma only", golden_e(), "50 48", "03 60 50 48"},
    {"F timing only", golden_f(),
     "04 00 00 03 e9 00 00 75 30 20",
     "0b 60 04 00 00 03 e9 00 00 75 30 20"},
    {"G sYCC + explicit SAR + ticks", golden_g(),
     "7c 92 ff 01 42 06 40 00 00 00 10 00 00 03 ca 40",
     "11 60 7c 92 ff 01 42 06 40 00 00 00 10 00 00 03 ca 40"},
    {"H reserved colour idc", golden_h(), "20 b2", "03 60 20 b2"},
    {"I interlace 4:2:0 positions", golden_i(), "98 94 01",
     "04 60 98 94 01"},
};

void test_goldens() {
  for (const Golden& golden : kGoldens) {
    std::string error;
    const std::vector<std::uint8_t> payload =
        serialize_content_interpretation_payload(golden.params, error);
    if (!error.empty()) {
      ++g_failures;
      std::cerr << golden.name << ": serialize error: " << error << '\n';
      continue;
    }
    CHECK_HEX(payload, golden.payload_hex);

    std::size_t payload_bits = 0;
    const std::vector<std::uint8_t> obu =
        make_content_interpretation_obu(golden.params, error, &payload_bits);
    CHECK(error.empty());
    CHECK_HEX(obu, golden.obu_hex);

    ContentInterpretationView view;
    if (!parse_content_interpretation(payload, view, error, true)) {
      ++g_failures;
      std::cerr << golden.name << ": parse failed: " << error << '\n';
      continue;
    }
    CHECK(view.params == golden.params);
    CHECK(view.reserved_bits == 0U);
    CHECK(view.syntax_bits <= payload_bits);

    const std::vector<std::uint8_t> reencoded =
        serialize_content_interpretation_payload(view.params, error);
    CHECK(error.empty());
    CHECK_HEX(reencoded, golden.payload_hex);
  }
}

void test_tables() {
  const std::array<std::uint32_t, 17> width{
      0, 1, 12, 10, 16, 40, 24, 20, 32, 80, 18, 15, 64, 160, 4, 3, 2};
  const std::array<std::uint32_t, 17> height{
      0, 1, 11, 11, 11, 33, 11, 11, 11, 33, 11, 11, 33, 99, 3, 2, 1};
  CHECK(kAspectRatioWidth == width);
  CHECK(kAspectRatioHeight == height);
  for (std::uint8_t idc = 0; idc <= 16; ++idc) {
    AspectRatio ratio;
    ratio.idc = idc;
    CHECK(ratio.resolved_width() == width[idc]);
    CHECK(ratio.resolved_height() == height[idc]);
  }

  // §6.14 4:2:0 offsets.
  const ChromaSampleOffset left420 =
      chroma_sample_offset(ChromaSamplePosition::kLeft, ChromaFormat::k420);
  CHECK(left420.applicable);
  CHECK(left420.horizontal == 0.0);
  CHECK(left420.vertical == 0.5);
  const ChromaSampleOffset left422 =
      chroma_sample_offset(ChromaSamplePosition::kLeft, ChromaFormat::k422);
  CHECK(left422.applicable);
  CHECK(left422.horizontal == 0.0);
  CHECK(left422.vertical == 0.0);
  const ChromaSampleOffset center420 =
      chroma_sample_offset(ChromaSamplePosition::kCenter, ChromaFormat::k420);
  CHECK(center420.horizontal == 0.5);
  CHECK(center420.vertical == 0.5);
  const ChromaSampleOffset center422 =
      chroma_sample_offset(ChromaSamplePosition::kCenter, ChromaFormat::k422);
  CHECK(center422.horizontal == 0.5);
  CHECK(center422.vertical == 0.0);
  const ChromaSampleOffset top_left =
      chroma_sample_offset(ChromaSamplePosition::kTopLeft, ChromaFormat::k420);
  CHECK(top_left.applicable);
  CHECK(top_left.horizontal == 0.0);
  CHECK(top_left.vertical == 0.0);
  const ChromaSampleOffset bottom =
      chroma_sample_offset(ChromaSamplePosition::kBottom, ChromaFormat::k420);
  CHECK(bottom.vertical == 1.0);
  // 4:2:2 has no vertical half-sample positions beyond the two rows.
  CHECK(!chroma_sample_offset(ChromaSamplePosition::kTopLeft,
                              ChromaFormat::k422)
             .applicable);
  CHECK(!chroma_sample_offset(ChromaSamplePosition::kTop, ChromaFormat::k422)
             .applicable);
  CHECK(!chroma_sample_offset(ChromaSamplePosition::kBottomLeft,
                              ChromaFormat::k422)
             .applicable);
  CHECK(!chroma_sample_offset(ChromaSamplePosition::kBottom,
                              ChromaFormat::k422)
             .applicable);
  CHECK(!chroma_sample_offset(ChromaSamplePosition::kUnspecified,
                              ChromaFormat::k420)
             .applicable);
}

void test_color_derivation() {
  const ColorDescription bt709 = derive_color_description(1);
  CHECK(bt709.primaries == 1 && bt709.transfer == 1 && bt709.matrix == 1);
  const ColorDescription pq = derive_color_description(2);
  CHECK(pq.primaries == 9 && pq.transfer == 16 && pq.matrix == 9);
  const ColorDescription hlg = derive_color_description(3);
  CHECK(hlg.primaries == 9 && hlg.transfer == 18 && hlg.matrix == 9);
  const ColorDescription srgb = derive_color_description(4);
  CHECK(srgb.primaries == 1 && srgb.transfer == 13 && srgb.matrix == 0);
  const ColorDescription sycc = derive_color_description(5);
  CHECK(sycc.primaries == 1 && sycc.transfer == 13 && sycc.matrix == 5);
  const ColorDescription reserved = derive_color_description(9);
  CHECK(reserved.primaries == 2 && reserved.transfer == 2 &&
        reserved.matrix == 2);
}

void test_validation() {
  // Hard conformance: chroma sample position > 5 when present.
  ContentInterpretation chroma = golden_e();
  chroma.chroma_top = ChromaSamplePosition::kUnspecified;
  chroma.chroma_bottom = ChromaSamplePosition::kUnspecified;
  CHECK(validate_content_interpretation(chroma).has_errors());
  std::string error;
  CHECK(make_content_interpretation_obu(chroma, error).empty());
  CHECK(!error.empty());

  // Aspect ratio idc 17..254 is forbidden.
  ContentInterpretation aspect = golden_a();
  aspect.has_aspect_ratio = true;
  aspect.aspect.idc = 17;
  CHECK(validate_content_interpretation(aspect).has_errors());
  CHECK(make_content_interpretation_obu(aspect, error).empty());

  // 255 is always allowed, but 0:0 is "unspecified" and warns.
  aspect.aspect.idc = 255;
  aspect.aspect.sar_width = 0;
  aspect.aspect.sar_height = 0;
  const ValidationResult sar_zero = validate_content_interpretation(aspect);
  CHECK(sar_zero.ok());
  CHECK(sar_zero.has_warnings());

  // One zero dimension warns; the ratio is unspecified.
  aspect.aspect.sar_width = 16;
  const ValidationResult sar_half = validate_content_interpretation(aspect);
  CHECK(sar_half.ok());
  CHECK(sar_half.has_warnings());

  // Progressive scan codes one chroma position; a differing bottom is an error
  // unless it is left unspecified (then it is derived from top).
  ContentInterpretation progressive = golden_e();
  progressive.chroma_bottom = ChromaSamplePosition::kBottom;
  CHECK(validate_content_interpretation(progressive).has_errors());
  progressive.chroma_bottom = ChromaSamplePosition::kUnspecified;
  CHECK(validate_content_interpretation(progressive).ok());

  // Timing ranges.
  ContentInterpretation timing = golden_f();
  timing.timing.num_units_in_display_tick = 0;
  CHECK(validate_content_interpretation(timing).has_errors());
  timing = golden_f();
  timing.timing.time_scale = 0;
  CHECK(validate_content_interpretation(timing).has_errors());
  timing = golden_f();
  timing.timing.equal_picture_interval = true;
  timing.timing.num_ticks_per_picture_minus_1 = 0xFFFFFFFFU;
  CHECK(validate_content_interpretation(timing).has_errors());
  timing.timing.num_ticks_per_picture_minus_1 = 0xFFFFFFFEU;
  CHECK(validate_content_interpretation(timing).ok());

  // Reserved colour codepoints are warnings, errors under strict.
  ContentInterpretation reserved = golden_h();
  const ValidationResult lenient = validate_content_interpretation(reserved);
  CHECK(lenient.ok());
  CHECK(lenient.has_warnings());
  CHECK(validate_content_interpretation(reserved, ChromaFormat::k420, true)
            .has_errors());

  // 4:2:2 cannot use CSP_TOPLEFT..CSP_BOTTOM.
  ContentInterpretation chroma422 = golden_d();
  const ValidationResult lenient422 =
      validate_content_interpretation(chroma422, ChromaFormat::k422);
  CHECK(lenient422.ok());
  CHECK(lenient422.has_warnings());
  CHECK(validate_content_interpretation(chroma422, ChromaFormat::k422, true)
            .has_errors());

  // Ignored fields produce warnings, not errors.
  ContentInterpretation ignored = golden_a();
  ignored.color.primaries = 9;
  const ValidationResult ignored_result = validate_content_interpretation(ignored);
  CHECK(ignored_result.ok());
  CHECK(ignored_result.has_warnings());
}

void test_reader_rejects_invalid() {
  std::string error;
  ContentInterpretationView view;

  // Chroma position 6 is not a conforming coded value.
  ManualPayload chroma6;
  chroma6.put(1U, 2U);  // scan = progressive
  chroma6.bit(false);   // colour description absent
  chroma6.bit(true);    // chroma present
  chroma6.bit(false);   // aspect absent
  chroma6.bit(false);   // timing absent
  chroma6.put(0U, 2U);  // reserved
  chroma6.uvlc(6);
  const std::vector<std::uint8_t> chroma6_payload = chroma6.finish();
  CHECK(!parse_content_interpretation(chroma6_payload, view, error, true));
  CHECK(!error.empty());
  // Structural parsing still works without the semantic checks.
  CHECK(parse_content_interpretation(chroma6_payload, view, error, false));
  CHECK(view.params.chroma_top == ChromaSamplePosition::kUnspecified);

  // Aspect ratio idc 17 with no explicit SAR.
  ManualPayload aspect17;
  aspect17.put(0U, 2U);
  aspect17.bit(false);
  aspect17.bit(false);
  aspect17.bit(true);
  aspect17.bit(false);
  aspect17.put(0U, 2U);
  aspect17.put(17U, 8U);
  const std::vector<std::uint8_t> aspect17_payload = aspect17.finish();
  CHECK(!parse_content_interpretation(aspect17_payload, view, error, true));
  CHECK(parse_content_interpretation(aspect17_payload, view, error, false));

  // Zero timing values are a conformance violation.
  ManualPayload bad_timing;
  bad_timing.put(0U, 2U);
  bad_timing.bit(false);
  bad_timing.bit(false);
  bad_timing.bit(false);
  bad_timing.bit(true);
  bad_timing.put(0U, 2U);
  bad_timing.put(0U, 32U);
  bad_timing.put(30U, 32U);
  bad_timing.bit(false);
  const std::vector<std::uint8_t> bad_timing_payload = bad_timing.finish();
  CHECK(!parse_content_interpretation(bad_timing_payload, view, error, true));

  // A chroma position above the enum range must not wrap through uint8_t.
  ManualPayload chroma300;
  chroma300.put(1U, 2U);
  chroma300.bit(false);
  chroma300.bit(true);
  chroma300.bit(false);
  chroma300.bit(false);
  chroma300.put(0U, 2U);
  chroma300.uvlc(300);
  const std::vector<std::uint8_t> chroma300_payload = chroma300.finish();
  CHECK(!parse_content_interpretation(chroma300_payload, view, error, false));

  // §4.11.10: 32 one bits make rg() return the -1 sentinel, which is a
  // descriptor conformance violation.
  ManualPayload rg_sentinel;
  rg_sentinel.put(0U, 2U);
  rg_sentinel.bit(true);   // colour description present
  rg_sentinel.bit(false);  // chroma absent
  rg_sentinel.bit(false);  // aspect absent
  rg_sentinel.bit(false);  // timing absent
  rg_sentinel.put(0U, 2U);
  for (unsigned index = 0; index < 32U; ++index) {
    rg_sentinel.bit(true);
  }
  const std::vector<std::uint8_t> rg_sentinel_payload = rg_sentinel.finish();
  CHECK(!parse_content_interpretation(rg_sentinel_payload, view, error, false));

  // Truncated payload.
  const std::vector<std::uint8_t> payload =
      serialize_content_interpretation_payload(golden_b(), error);
  CHECK(!payload.empty());
  std::vector<std::uint8_t> truncated(payload.begin(), payload.end() - 1);
  CHECK(!parse_content_interpretation(truncated, view, error, true));

  // Corrupted trailing bits.
  std::vector<std::uint8_t> bad_trailing = payload;
  bad_trailing.back() = 0x00;
  CHECK(!parse_content_interpretation(bad_trailing, view, error, true));

  // Non-zero padding after the trailing 1 bit.
  std::vector<std::uint8_t> bad_padding = payload;
  bad_padding.back() |= 0x01U;
  CHECK(!parse_content_interpretation(bad_padding, view, error, true));

  // Missing extension flag (payload ends exactly after the syntax).
  CHECK(!parse_content_interpretation(
      std::span<const std::uint8_t>(payload).first(11U), view, error, true));

  // ci_reserved_2bit is recorded but ignored by the decoder.
  ManualPayload reserved;
  reserved.put(0U, 2U);
  reserved.bit(false);
  reserved.bit(false);
  reserved.bit(false);
  reserved.bit(false);
  reserved.put(3U, 2U);
  const std::vector<std::uint8_t> reserved_payload = reserved.finish();
  CHECK(parse_content_interpretation(reserved_payload, view, error, true));
  CHECK(view.reserved_bits == 3U);

  // obu_extension_flag = 1 is not parsed by this module.
  ManualPayload extended;
  extended.put(0U, 2U);
  extended.bit(false);
  extended.bit(false);
  extended.bit(false);
  extended.bit(false);
  extended.put(0U, 2U);
  std::vector<std::uint8_t> extended_payload = extended.finish();
  extended_payload.back() |= 0x80U;  // set the extension flag instead
  CHECK(!parse_content_interpretation(extended_payload, view, error, true));
}

void test_obu_parser() {
  std::string error;
  const std::vector<std::uint8_t> payload =
      serialize_content_interpretation_payload(golden_b(), error);
  const std::vector<std::uint8_t> obu =
      make_obu(kObuTypeContentInterpretation, payload);
  std::vector<ObuView> obus;
  CHECK(parse_obu_sequence(obu, obus, error));
  CHECK(obus.size() == 1U);
  CHECK(obus[0].type == kObuTypeContentInterpretation);
  CHECK(obus[0].size == obu.size());
  CHECK(obus[0].payload.size() == payload.size());

  // Truncated leb128: the second byte is the OBU header, which has its top
  // bit clear, so the decoded size is far past the buffer.
  std::vector<std::uint8_t> bad = obu;
  bad[0] = 0x7FU;
  std::size_t offset = 0;
  ObuView view;
  CHECK(!parse_obu(bad, offset, view, error));

  // Zero size is invalid.
  const std::vector<std::uint8_t> zero{0x00U, 0x60U};
  offset = 0;
  CHECK(!parse_obu(zero, offset, view, error));
}

void test_insertion() {
  const std::vector<std::uint8_t> td_payload{};
  const std::vector<std::uint8_t> seq_payload{0x00U};
  const std::vector<std::uint8_t> frame_payload{0x11U, 0x22U};
  const std::vector<std::uint8_t> metadata_payload{0x00U};

  const std::vector<std::uint8_t> td = make_obu(2U, td_payload);
  const std::vector<std::uint8_t> seq = make_obu(1U, seq_payload);
  const std::vector<std::uint8_t> frame = make_obu(4U, frame_payload);
  const std::vector<std::uint8_t> metadata = make_obu(8U, metadata_payload);

  std::string error;
  const std::vector<std::uint8_t> ci_obu =
      make_content_interpretation_obu(golden_b(), error);
  CHECK(error.empty());

  // TD + SEQ + CLK -> TD + SEQ + CI + CLK.
  const std::vector<std::uint8_t> plain = concat({td, seq, frame});
  std::vector<std::uint8_t> plain_inserted;
  std::size_t inserted_at = 0;
  CHECK(insert_content_interpretation(plain, ci_obu, plain_inserted, error,
                                      &inserted_at));
  CHECK(inserted_at == td.size() + seq.size());
  CHECK(plain_inserted.size() == plain.size() + ci_obu.size());
  std::vector<ObuView> obus;
  CHECK(parse_obu_sequence(plain_inserted, obus, error));
  CHECK(obus.size() == 4U);
  CHECK(obus[0].type == 2U);
  CHECK(obus[1].type == 1U);
  CHECK(obus[2].type == kObuTypeContentInterpretation);
  CHECK(obus[3].type == 4U);

  // Prefix metadata must stay after the CI OBU (§7.3.3 order).
  const std::vector<std::uint8_t> with_metadata =
      concat({td, seq, metadata, frame});
  std::vector<std::uint8_t> metadata_inserted;
  CHECK(insert_content_interpretation(with_metadata, ci_obu, metadata_inserted,
                                      error));
  CHECK(parse_obu_sequence(metadata_inserted, obus, error));
  CHECK(obus.size() == 5U);
  CHECK(obus[2].type == kObuTypeContentInterpretation);
  CHECK(obus[3].type == 8U);
  CHECK(obus[4].type == 4U);

  // Scan reports the CI OBU before the first frame.
  ContentInterpretationScanResult scan;
  CHECK(scan_content_interpretation(plain_inserted, scan, error));
  CHECK(scan.ci_obus == 1U);
  CHECK(scan.invalid_ci_obus == 0U);
  CHECK(scan.has_sequence_header);
  CHECK(scan.has_first_frame);
  CHECK(scan.records.size() == 1U);
  CHECK(scan.records[0].before_first_frame);
  CHECK(scan.first_frame_offset == inserted_at + ci_obu.size());

  // A temporal unit may not carry two CI OBUs.
  std::vector<std::uint8_t> twice;
  CHECK(!insert_content_interpretation(plain_inserted, ci_obu, twice, error));

  // A CI OBU after the frame is reported by the scanner as misordered.
  const std::vector<std::uint8_t> late = concat({td, seq, frame, ci_obu});
  ContentInterpretationScanResult late_scan;
  CHECK(scan_content_interpretation(late, late_scan, error));
  CHECK(late_scan.ci_obus == 1U);
  CHECK(late_scan.invalid_ci_obus == 1U);
  CHECK(late_scan.records.size() == 1U);
  CHECK(!late_scan.records[0].before_first_frame);

  // Two CI OBUs in one temporal unit are a §7.3.3 violation.
  const std::vector<std::uint8_t> duplicate = concat({td, seq, ci_obu, ci_obu, frame});
  ContentInterpretationScanResult duplicate_scan;
  CHECK(scan_content_interpretation(duplicate, duplicate_scan, error));
  CHECK(duplicate_scan.ci_obus == 2U);
  CHECK(duplicate_scan.invalid_ci_obus == 1U);
  CHECK(duplicate_scan.records.size() == 2U);
  CHECK(duplicate_scan.records[0].valid);
  CHECK(duplicate_scan.records[1].valid);

  // No sequence header.
  const std::vector<std::uint8_t> no_seq = concat({td, frame});
  std::vector<std::uint8_t> rejected;
  CHECK(!insert_content_interpretation(no_seq, ci_obu, rejected, error));

  // No frame.
  const std::vector<std::uint8_t> no_frame = concat({td, seq});
  CHECK(!insert_content_interpretation(no_frame, ci_obu, rejected, error));

  // Two sequence headers with different obu_xlayer_id make placement
  // ambiguous.
  const std::vector<std::uint8_t> seq_x0 = make_obu(1U, seq_payload, true, 0U);
  const std::vector<std::uint8_t> seq_x1 = make_obu(1U, seq_payload, true, 1U);
  const std::vector<std::uint8_t> multi = concat({td, seq_x0, seq_x1, frame});
  CHECK(!insert_content_interpretation(multi, ci_obu, rejected, error));

  // The same xlayer twice is accepted (insert after the last sequence
  // header).
  const std::vector<std::uint8_t> multi_same =
      concat({td, seq_x0, seq_x0, frame});
  std::vector<std::uint8_t> multi_inserted;
  CHECK(insert_content_interpretation(multi_same, ci_obu, multi_inserted,
                                      error));
  CHECK(parse_obu_sequence(multi_inserted, obus, error));
  CHECK(obus.size() == 5U);
  CHECK(obus[2].type == 1U);
  CHECK(obus[3].type == kObuTypeContentInterpretation);

  // Global metadata before the coded extended layer unit is rejected.
  const std::vector<std::uint8_t> global_metadata =
      make_obu(8U, metadata_payload, true, 31U);
  const std::vector<std::uint8_t> global_tu =
      concat({td, global_metadata, seq, frame});
  CHECK(!insert_content_interpretation(global_tu, ci_obu, rejected, error));
}

void test_cvs_constancy() {
  ContentInterpretation params = golden_b();

  {
    const CiObservation observations[] = {
        {0U, true, true, &params},
        {0U, false, false, &params},
        {0U, false, false, &params},
    };
    const ValidationResult result =
        check_cvs_constancy(std::span<const CiObservation>(observations));
    CHECK(result.ok());
    CHECK(result.clean());
  }

  {
    ContentInterpretation changed = params;
    changed.chroma_top = ChromaSamplePosition::kLeft;
    changed.chroma_bottom = ChromaSamplePosition::kLeft;
    const CiObservation observations[] = {
        {0U, true, true, &params},
        {0U, false, false, &changed},
    };
    const ValidationResult result =
        check_cvs_constancy(std::span<const CiObservation>(observations));
    CHECK(result.has_errors());
  }

  {
    // Present in a later TU but not in the first TU of the CVS.
    const CiObservation observations[] = {
        {0U, true, true, nullptr},
        {0U, false, false, &params},
    };
    const ValidationResult result =
        check_cvs_constancy(std::span<const CiObservation>(observations));
    CHECK(result.has_errors());
  }

  {
    // Timing must match across embedded layers.
    ContentInterpretation layer1 = default_content_interpretation();
    layer1.has_timing_info = true;
    layer1.timing.num_units_in_display_tick = 1;
    layer1.timing.time_scale = 30;
    ContentInterpretation layer0 = layer1;
    layer0.timing.time_scale = 60;
    const CiObservation observations[] = {
        {0U, true, true, &layer0},
        {1U, true, true, &layer1},
    };
    const ValidationResult result =
        check_cvs_constancy(std::span<const CiObservation>(observations));
    CHECK(result.has_errors());
  }

  {
    // Different parameters in different layers are fine, only timing is
    // constrained across layers.
    ContentInterpretation layer1 = default_content_interpretation();
    layer1.scan_type = ScanType::kInterlaceField;
    const CiObservation observations[] = {
        {0U, true, true, &params},
        {1U, true, true, &layer1},
    };
    const ValidationResult result =
        check_cvs_constancy(std::span<const CiObservation>(observations));
    CHECK(result.ok());
  }

  {
    // A later random access point re-establishes parameters.
    ContentInterpretation changed = params;
    changed.has_timing_info = false;
    changed.timing = TimingInfo{};
    changed.scan_type = ScanType::kInterlaceComplementaryPair;
    const CiObservation observations[] = {
        {0U, true, true, &params},
        {0U, false, true, &changed},
    };
    const ValidationResult result =
        check_cvs_constancy(std::span<const CiObservation>(observations));
    CHECK(result.ok());
  }

  {
    // Layer state is per embedded layer: layer 1's first TU must not reset
    // layer 0's established parameters.
    ContentInterpretation layer0 = golden_b();
    ContentInterpretation layer1 = default_content_interpretation();
    const CiObservation observations[] = {
        {0U, true, true, &layer0},
        {1U, true, true, &layer1},
        {0U, false, false, &layer0},
        {1U, false, false, &layer1},
    };
    const ValidationResult result =
        check_cvs_constancy(std::span<const CiObservation>(observations));
    CHECK(result.ok());
  }

  {
    // A layer that appears after the first temporal unit of the CVS cannot be
    // validated.
    const CiObservation observations[] = {
        {0U, true, true, &params},
        {2U, false, false, &params},
    };
    const ValidationResult result =
        check_cvs_constancy(std::span<const CiObservation>(observations));
    CHECK(result.has_errors());
  }

  {
    // reset() starts a new CVS, where a different timing_info() is legal.
    CvsConstancyTracker tracker;
    std::string error;
    const CiObservation first{0U, true, true, &params};
    CHECK(tracker.observe(first, error));
    CHECK(tracker.timing_established());
    tracker.reset();
    CHECK(!tracker.timing_established());
    ContentInterpretation other = params;
    other.timing.time_scale = 60;
    const CiObservation new_cvs{0U, true, true, &other};
    CHECK(tracker.observe(new_cvs, error));
    CHECK(tracker.timing().time_scale == 60U);
  }

  {
    // Missing first observation is an API error.
    CvsConstancyTracker tracker;
    std::string error;
    const CiObservation observation{0U, false, false, &params};
    CHECK(!tracker.observe(observation, error));
    CHECK(!error.empty());
    tracker.reset();
    const CiObservation first{0U, true, true, &params};
    CHECK(tracker.observe(first, error));
    CHECK(tracker.established(0U));
    CHECK(tracker.timing_established());
  }

  {
    // Observing a CI OBU in a non-RAP TU with no previous establishment
    // establishes the parameters.
    CvsConstancyTracker tracker;
    std::string error;
    const CiObservation first{0U, true, true, nullptr};
    CHECK(tracker.observe(first, error));
    CHECK(!tracker.established(0U));
    const CiObservation second{0U, false, false, &params};
    CHECK(!tracker.observe(second, error));  // missing in the first TU
  }
}

void test_scan_invalid_ci() {
  const std::vector<std::uint8_t> empty_payload;
  const std::vector<std::uint8_t> seq_payload{0x00U};
  const std::vector<std::uint8_t> frame_payload{0x11U};
  const std::vector<std::uint8_t> td = make_obu(2U, empty_payload);
  const std::vector<std::uint8_t> seq = make_obu(1U, seq_payload);
  const std::vector<std::uint8_t> frame = make_obu(4U, frame_payload);

  ManualPayload chroma6;
  chroma6.put(1U, 2U);
  chroma6.bit(false);
  chroma6.bit(true);
  chroma6.bit(false);
  chroma6.bit(false);
  chroma6.put(0U, 2U);
  chroma6.uvlc(6);
  const std::vector<std::uint8_t> invalid_payload = chroma6.finish();
  const std::vector<std::uint8_t> invalid_ci =
      make_obu(kObuTypeContentInterpretation, invalid_payload);

  const std::vector<std::uint8_t> tu = concat({td, seq, invalid_ci, frame});
  ContentInterpretationScanResult scan;
  std::string error;
  CHECK(scan_content_interpretation(tu, scan, error));
  CHECK(scan.ci_obus == 1U);
  CHECK(scan.invalid_ci_obus == 1U);
  CHECK(scan.records.size() == 1U);
  CHECK(scan.records[0].before_first_frame);
}

}  // namespace

int main() {
  test_goldens();
  test_tables();
  test_color_derivation();
  test_validation();
  test_reader_rejects_invalid();
  test_obu_parser();
  test_insertion();
  test_cvs_constancy();
  test_scan_invalid_ci();

  if (g_failures == 0) {
    std::cout << "content_interpretation_test: all checks passed\n";
    return 0;
  }
  std::cerr << "content_interpretation_test: " << g_failures
            << " check(s) failed\n";
  return 1;
}
