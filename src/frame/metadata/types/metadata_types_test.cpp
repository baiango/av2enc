// Unit tests for the AV2 v1.0.0 non-HDR metadata unit types.
//
// Build with src/encoder/metadata/types/build.sh (no CMake target yet: this
// agent owns only src/encoder/metadata/types/). The expectations below are
// pinned bit goldens derived from the §5.17 syntax with an independent
// bit-packing reference, then cross-checked through the writers and readers.
//
// Covered: §5.17.4 / §5.17.7 / §5.17.8 / §5.17.9 / §5.17.10 / §5.17.11 /
// §5.17.12 / §5.17.13 syntax and §6.16.4 / §6.16.7-§6.16.13 semantics,
// OBU_METADATA_SHORT and OBU_METADATA_GROUP framing, cancel/persistence/
// necessity/layer flags, prefix/suffix placement, stream scanning, and
// malformed-input rejection.

#include "frame/metadata/types/metadata_types.h"

#include "core/bitstream/basic_headers.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace av2enc;
using namespace av2enc::metadata;

int g_failures = 0;

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      ++g_failures;                                                           \
      std::cerr << __FILE__ << ':' << __LINE__                                \
                << ": CHECK failed: " #expression "\n";                       \
    }                                                                         \
  } while (false)

#define REQUIRE(expression)                                                   \
  do {                                                                        \
    if (!(expression)) {                                                      \
      ++g_failures;                                                           \
      std::cerr << __FILE__ << ':' << __LINE__                                \
                << ": REQUIRE failed: " #expression "\n";                     \
      return;                                                                 \
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
  out.reserve(bytes.size() * 2U);
  for (const std::uint8_t byte : bytes) {
    out.push_back(kDigits[byte >> 4U]);
    out.push_back(kDigits[byte & 0x0FU]);
  }
  return out;
}

std::string to_hex(const std::vector<std::uint8_t>& bytes) {
  return to_hex(std::span<const std::uint8_t>(bytes));
}

int count_errors(const hdr::ValidationResult& result) {
  int count = 0;
  for (const auto& issue : result.issues()) {
    if (issue.severity == hdr::Severity::kError) {
      ++count;
    }
  }
  return count;
}

int count_warnings(const hdr::ValidationResult& result) {
  int count = 0;
  for (const auto& issue : result.issues()) {
    if (issue.severity == hdr::Severity::kWarning) {
      ++count;
    }
  }
  return count;
}

// ---------------------------------------------------------------------------
// Samples (the golden values pinned below)
// ---------------------------------------------------------------------------

ItutT35 sample_itut_t35() {
  ItutT35 value;
  value.country_code = 0xB5;  // United States
  value.payload = {0x00, 0x3C, 0x00, 0x01, 0x04};
  return value;
}

ItutT35 sample_itut_t35_extension() {
  ItutT35 value;
  value.country_code = 0xFF;
  value.country_code_extension_byte = 0x00;
  value.payload = {0x31, 0x32};
  return value;
}

Timecode sample_timecode() {
  Timecode value;
  value.counting_type = 0;
  value.full_timestamp_flag = true;
  value.discontinuity_flag = false;
  value.cnt_dropped_flag = false;
  value.n_frames = 42;
  value.seconds_value = 7;
  value.minutes_value = 3;
  value.hours_value = 1;
  // In the full-timestamp form the three presence flags are not coded; the
  // decoder reports them as true, so the canonical value sets them too.
  value.seconds_flag = true;
  value.minutes_flag = true;
  value.hours_flag = true;
  value.time_offset_length = 0;
  return value;
}

BandingHints sample_banding_hints() {
  BandingHints value;
  value.coding_banding_present_flag = true;
  value.source_banding_present_flag = false;
  value.banding_hints_flag = true;
  value.three_color_components_flag = false;
  value.components[0].banding_in_component_present_flag = true;
  value.components[0].max_band_width_minus_4 = 12;
  value.components[0].max_band_step_minus_1 = 3;
  value.band_units_information_present_flag = true;
  value.num_band_units_rows_minus_1 = 1;  // 2 rows
  value.num_band_units_cols_minus_1 = 2;  // 3 columns
  value.varying_size_band_units_flag = true;
  value.band_block_in_luma_samples = 1;  // 32 luma samples
  value.vert_size_in_band_blocks_minus_1 = {0, 1};
  value.horz_size_in_band_blocks_minus_1 = {0, 1, 2};
  value.banding_in_band_unit_present = {1, 0, 1, 0, 1, 1};
  return value;
}

IccProfile sample_icc_profile() {
  IccProfile value;
  // ICC header prefix with 'ADBE' magic and profile version 4.3.
  value.data = {0x00, 0x00, 0x01, 0x20, 0x41, 0x44, 0x42, 0x45,
                0x04, 0x30, 0x00, 0x00, 0x6D, 0x6E, 0x74, 0x72};
  return value;
}

ScanType sample_scan_type() {
  ScanType value;
  value.mps_pic_struct_type = 1;       // top field
  value.mps_source_scan_type_idc = 2;  // interlace field picture samples
  value.mps_duplicate_flag = false;
  return value;
}

TemporalPointInfo sample_temporal_point_info() {
  TemporalPointInfo value;
  value.frame_presentation_time = 1000;
  return value;
}

FrameHash hash_sequence(std::uint8_t start) {
  FrameHash hash{};
  for (std::size_t i = 0; i < hash.size(); ++i) {
    hash[i] = static_cast<std::uint8_t>(start + i);
  }
  return hash;
}

DecodedFrameHash sample_decoded_frame_hash() {
  DecodedFrameHash value;
  value.hash_type = 0;  // MD5
  value.per_plane = true;
  value.has_grain = false;
  value.is_monochrome = false;
  value.plane_hash[0] = hash_sequence(0x00);
  value.plane_hash[1] = hash_sequence(0x10);
  value.plane_hash[2] = hash_sequence(0x20);
  return value;
}

UserDataUnregistered sample_user_data() {
  UserDataUnregistered value;
  for (std::size_t i = 0; i < value.uuid.size(); ++i) {
    value.uuid[i] = static_cast<std::uint8_t>(i);
  }
  value.payload = {0xDE, 0xAD, 0xBE, 0xEF};
  return value;
}

// All eight units in canonical order (the group golden order).
std::vector<MetadataUnitPayload> all_units() {
  std::string error;
  std::vector<MetadataUnitPayload> units;
  const auto append = [&units](std::optional<MetadataUnitPayload> unit) {
    if (unit) {
      units.push_back(std::move(*unit));
    }
  };
  append(make_itut_t35_unit(sample_itut_t35(), error));
  append(make_timecode_unit(sample_timecode(), error));
  append(make_decoded_frame_hash_unit(sample_decoded_frame_hash(), error));
  append(make_banding_hints_unit(sample_banding_hints(), error));
  append(make_icc_profile_unit(sample_icc_profile(), error));
  append(make_scan_type_unit(sample_scan_type(), error));
  append(make_temporal_point_info_unit(sample_temporal_point_info(), error));
  append(make_user_data_unregistered_unit(sample_user_data(), error));
  return units;
}

// Group-eligible subset: METADATA_TYPE_TEMPORAL_POINT_INFO is short-only.
std::vector<GroupUnit> group_units() {
  std::vector<GroupUnit> groups;
  for (const auto& unit : all_units()) {
    if (unit.type == Type::kTemporalPointInfo) {
      continue;
    }
    GroupUnit group;
    group.payload = unit;
    group.layer = hdr::MetadataLayer::kCurrent;
    group.persistence = hdr::MetadataPersistence::kGlobal;
    group.priority = unit.type == Type::kDecodedFrameHash ? 1U : 0U;
    groups.push_back(std::move(group));
  }
  return groups;
}

// ---------------------------------------------------------------------------
// OBU helpers
// ---------------------------------------------------------------------------

std::vector<std::uint8_t> make_test_obu(std::uint8_t type,
                                        std::span<const std::uint8_t> payload) {
  av2enc::basic::ObuHeader header;
  header.type = static_cast<av2enc::basic::ObuType>(type);
  header.tlayer_id = 0;
  header.extension_flag = false;
  return av2enc::basic::make_obu(header, payload);
}

std::vector<std::uint8_t> make_test_temporal_unit() {
  std::vector<std::uint8_t> out = {0x01, 0x08};  // temporal delimiter
  const std::vector<std::uint8_t> seq(12, 0x5A);
  const std::vector<std::uint8_t> seq_obu = make_test_obu(1, seq);
  out.insert(out.end(), seq_obu.begin(), seq_obu.end());
  const std::vector<std::uint8_t> frame_payload = {0x80};
  const std::vector<std::uint8_t> frame = make_test_obu(4, frame_payload);
  out.insert(out.end(), frame.begin(), frame.end());
  return out;
}

bool parse_short_obu(std::span<const std::uint8_t> bytes,
                     hdr::MetadataShortView& view, std::string& error) {
  std::vector<hdr::ObuView> obus;
  if (!hdr::parse_obu_sequence(bytes, obus, error)) {
    return false;
  }
  if (obus.size() != 1U || obus[0].type != hdr::kObuTypeMetadataShort) {
    error = "expected exactly one OBU_METADATA_SHORT";
    return false;
  }
  return hdr::parse_metadata_short(obus[0].payload, view, error);
}

bool parse_group_obu(std::span<const std::uint8_t> bytes,
                     hdr::MetadataGroupView& view, std::string& error) {
  std::vector<hdr::ObuView> obus;
  if (!hdr::parse_obu_sequence(bytes, obus, error)) {
    return false;
  }
  if (obus.size() != 1U || obus[0].type != hdr::kObuTypeMetadataGroup) {
    error = "expected exactly one OBU_METADATA_GROUP";
    return false;
  }
  return hdr::parse_metadata_group(obus[0].payload, obus[0].xlayer_id, view,
                                   error);
}

// ---------------------------------------------------------------------------
// Pinned bit goldens
// ---------------------------------------------------------------------------

void test_unit_goldens() {
  std::string error;
  {
    const auto unit = make_itut_t35_unit(sample_itut_t35(), error);
    REQUIRE(unit.has_value());
    CHECK_HEX(unit->bytes, "b5003c000104");
  }
  {
    const auto unit = make_itut_t35_unit(sample_itut_t35_extension(), error);
    REQUIRE(unit.has_value());
    CHECK_HEX(unit->bytes, "ff003132");
  }
  {
    const auto unit = make_timecode_unit(sample_timecode(), error);
    REQUIRE(unit.has_value());
    CHECK_HEX(unit->bytes, "04150e1840");
  }
  {
    const auto unit = make_banding_hints_unit(sample_banding_hints(), error);
    REQUIRE(unit.has_value());
    CHECK_HEX(unit->bytes, "a98708a401004558");
  }
  {
    const auto unit = make_icc_profile_unit(sample_icc_profile(), error);
    REQUIRE(unit.has_value());
    CHECK_HEX(unit->bytes,
              "0000012041444245043000006d6e7472");
  }
  {
    const auto unit = make_scan_type_unit(sample_scan_type(), error);
    REQUIRE(unit.has_value());
    CHECK_HEX(unit->bytes, "0c");
  }
  {
    const auto unit = make_temporal_point_info_unit(
        sample_temporal_point_info(), error);
    REQUIRE(unit.has_value());
    CHECK_HEX(unit->bytes, "e807");
  }
  {
    const auto unit =
        make_decoded_frame_hash_unit(sample_decoded_frame_hash(), error);
    REQUIRE(unit.has_value());
    CHECK_HEX(unit->bytes,
              "08000102030405060708090a0b0c0d0e0f"
              "101112131415161718191a1b1c1d1e1f"
              "202122232425262728292a2b2c2d2e2f");
  }
  {
    const auto unit = make_user_data_unregistered_unit(sample_user_data(),
                                                       error);
    REQUIRE(unit.has_value());
    CHECK_HEX(unit->bytes, "000102030405060708090a0b0c0d0e0fdeadbeef");
  }
}

void test_short_obu_goldens() {
  std::string error;
  hdr::MetadataObuConfig config;  // prefix, layer current, persistence global

  const auto short_hex = [&](const MetadataUnitPayload& unit,
                             const hdr::MetadataObuConfig& cfg,
                             const char* expected) {
    const std::vector<std::uint8_t> obu = make_short_obu(unit, cfg, error);
    REQUIRE(!obu.empty());
    CHECK_HEX(obu, expected);
  };

  short_hex(all_units()[0], config,
            "0a202003b5003c00010480");
  short_hex(all_units()[1], config,
            "0920200404150e1840" "80");
  short_hex(all_units()[2], config,
            "3520200508000102030405060708090a0b0c0d0e0f"
            "101112131415161718191a1b1c1d1e1f"
            "202122232425262728292a2b2c2d2e2f80");
  short_hex(all_units()[3], config,
            "0c202006a98708a40100455880");
  short_hex(all_units()[4], config,
            "142020070000012041444245043000006d6e747280");
  short_hex(all_units()[5], config,
            "052020080c80");
  short_hex(all_units()[6], config,
            "06202009e80780");
  short_hex(all_units()[7], config,
            "1820200a000102030405060708090a0b0c0d0e0fdeadbeef80");

  hdr::MetadataObuConfig suffix;
  suffix.suffix = true;
  short_hex(all_units()[0], suffix,
            "0a20a003b5003c00010480");

  const std::vector<std::uint8_t> cancel =
      make_short_cancel_obu(Type::kItutT35, config, error);
  REQUIRE(!cancel.empty());
  CHECK_HEX(cancel, "0420280380");

  const auto ext = make_itut_t35_unit(sample_itut_t35_extension(), error);
  REQUIRE(ext.has_value());
  short_hex(*ext, config, "08202003ff00313280");
}

void test_group_obu_goldens() {
  std::string error;
  const std::vector<GroupUnit> units = group_units();
  CHECK(units.size() == 7U);

  hdr::MetadataObuConfig config;
  config.necessity = hdr::MetadataNecessity::kAdvisory;
  const std::vector<std::uint8_t> prefix =
      make_group_obu(units, config, error);
  REQUIRE(!prefix.empty());
  CHECK_HEX(prefix,
            "90012440060306064000b5003c000104040605400004150e1840"
            "050631400408000102030405060708090a0b0c0d0e0f"
            "101112131415161718191a1b1c1d1e1f"
            "202122232425262728292a2b2c2d2e2f"
            "0606084000a98708a401004558"
            "07061040000000012041444245043000006d6e7472"
            "08060140000c"
            "0a06144000000102030405060708090a0b0c0d0e0fdeadbeef80");

  hdr::MetadataObuConfig suffix = config;
  suffix.suffix = true;
  const std::vector<std::uint8_t> suffix_bytes =
      make_group_obu(units, suffix, error);
  REQUIRE(!suffix_bytes.empty());
  CHECK_HEX(suffix_bytes,
            "900124c0060306064000b5003c000104040605400004150e1840"
            "050631400408000102030405060708090a0b0c0d0e0f"
            "101112131415161718191a1b1c1d1e1f"
            "202122232425262728292a2b2c2d2e2f"
            "0606084000a98708a401004558"
            "07061040000000012041444245043000006d6e7472"
            "08060140000c"
            "0a06144000000102030405060708090a0b0c0d0e0fdeadbeef80");

  std::vector<GroupUnit> cancels;
  for (const auto& unit : units) {
    GroupUnit entry;
    entry.payload.type = unit.payload.type;
    entry.layer = hdr::MetadataLayer::kCurrent;
    entry.persistence = hdr::MetadataPersistence::kGlobal;
    entry.cancel = true;
    cancels.push_back(std::move(entry));
  }
  const std::vector<std::uint8_t> cancel_bytes =
      make_group_obu(cancels, config, error);
  REQUIRE(!cancel_bytes.empty());
  CHECK_HEX(cancel_bytes, "122440060301040105010601070108010a0180");
}

// ---------------------------------------------------------------------------
// Exact payload sizes
// ---------------------------------------------------------------------------

void test_unit_sizes() {
  CHECK(itut_t35_unit_size(sample_itut_t35()) == 6U);
  CHECK(itut_t35_unit_size(sample_itut_t35_extension()) == 4U);
  CHECK(timecode_unit_size(sample_timecode()) == 5U);
  CHECK(icc_profile_unit_size(sample_icc_profile()) == 16U);
  CHECK(decoded_frame_hash_unit_size(sample_decoded_frame_hash()) == 49U);
  CHECK(user_data_unregistered_unit_size(sample_user_data()) == 20U);
  CHECK(temporal_point_info_unit_size(sample_temporal_point_info()) == 2U);

  Timecode min_timecode;
  CHECK(timecode_unit_size(min_timecode) == kTimecodeMinUnitBytes);

  Timecode max_timecode;
  max_timecode.counting_type = 5;
  max_timecode.full_timestamp_flag = false;
  max_timecode.n_frames = 511;
  max_timecode.seconds_flag = true;
  max_timecode.seconds_value = 59;
  max_timecode.minutes_flag = true;
  max_timecode.minutes_value = 59;
  max_timecode.hours_flag = true;
  max_timecode.hours_value = 23;
  max_timecode.time_offset_length = 31;
  max_timecode.time_offset_value = 0x7FFFFFFFU;
  CHECK(timecode_syntax_bits(max_timecode) == 73U);
  CHECK(timecode_unit_size(max_timecode) == kTimecodeMaxUnitBytes);

  BandingHints min_banding;
  CHECK(banding_hints_unit_size(min_banding) == kBandingHintsMinUnitBytes);

  BandingHints max_banding;
  max_banding.coding_banding_present_flag = true;
  max_banding.source_banding_present_flag = true;
  max_banding.banding_hints_flag = true;
  max_banding.three_color_components_flag = true;
  for (auto& component : max_banding.components) {
    component.banding_in_component_present_flag = true;
    component.max_band_width_minus_4 = 63;
    component.max_band_step_minus_1 = 15;
  }
  max_banding.band_units_information_present_flag = true;
  max_banding.num_band_units_rows_minus_1 = 31;
  max_banding.num_band_units_cols_minus_1 = 31;
  max_banding.varying_size_band_units_flag = true;
  max_banding.band_block_in_luma_samples = 7;
  max_banding.vert_size_in_band_blocks_minus_1.assign(32, 31);
  max_banding.horz_size_in_band_blocks_minus_1.assign(32, 31);
  max_banding.banding_in_band_unit_present.assign(32U * 32U, 1);
  CHECK(banding_hints_syntax_bits(max_banding) == 1396U);
  CHECK(banding_hints_unit_size(max_banding) == kBandingHintsMaxUnitBytes);

  const std::pair<std::uint32_t, std::size_t> leb_cases[] = {
      {0U, 1U},       {127U, 1U},      {128U, 2U},      {16383U, 2U},
      {16384U, 3U},   {2097151U, 3U},  {2097152U, 4U},
      {268435455U, 4U}, {268435456U, 5U}, {0xFFFFFFFFU, 5U}};
  for (const auto& [value, size] : leb_cases) {
    TemporalPointInfo point;
    point.frame_presentation_time = value;
    CHECK(temporal_point_info_unit_size(point) == size);
    const std::vector<std::uint8_t> bytes =
        serialize_temporal_point_info_unit(point);
    CHECK(bytes.size() == size);
    TemporalPointInfo decoded;
    std::string error;
    CHECK(decode_temporal_point_info(bytes, decoded, error));
    CHECK(decoded.frame_presentation_time == value);
  }

  DecodedFrameHash whole;
  whole.per_plane = false;
  whole.frame_hash = hash_sequence(0xA0);
  CHECK(decoded_frame_hash_unit_size(whole) ==
        kDecodedFrameHashWholeUnitBytes);
  DecodedFrameHash mono;
  mono.per_plane = true;
  mono.is_monochrome = true;
  mono.plane_hash[0] = hash_sequence(0x00);
  CHECK(decoded_frame_hash_unit_size(mono) ==
        kDecodedFrameHashPerPlaneMonoUnitBytes);
}

// ---------------------------------------------------------------------------
// Round-trips
// ---------------------------------------------------------------------------

void test_round_trip() {
  std::string error;

  {
    const auto unit = make_itut_t35_unit(sample_itut_t35(), error);
    REQUIRE(unit.has_value());
    ItutT35 decoded;
    REQUIRE(decode_itut_t35(unit->bytes, decoded, error));
    CHECK(decoded == sample_itut_t35());
    CHECK(serialize_itut_t35_unit(decoded) == unit->bytes);
  }
  {
    const auto unit = make_itut_t35_unit(sample_itut_t35_extension(), error);
    REQUIRE(unit.has_value());
    ItutT35 decoded;
    REQUIRE(decode_itut_t35(unit->bytes, decoded, error));
    CHECK(decoded == sample_itut_t35_extension());
    CHECK(decoded.country_code_extension_byte == 0x00);
  }
  {
    const auto unit = make_timecode_unit(sample_timecode(), error);
    REQUIRE(unit.has_value());
    Timecode decoded;
    REQUIRE(decode_timecode(unit->bytes, decoded, error));
    CHECK(decoded == sample_timecode());
    CHECK(serialize_timecode_unit(decoded) == unit->bytes);
  }
  {
    const auto unit = make_banding_hints_unit(sample_banding_hints(), error);
    REQUIRE(unit.has_value());
    BandingHints decoded;
    REQUIRE(decode_banding_hints(unit->bytes, decoded, error));
    CHECK(decoded == sample_banding_hints());
    CHECK(serialize_banding_hints_unit(decoded) == unit->bytes);
  }
  {
    const auto unit = make_icc_profile_unit(sample_icc_profile(), error);
    REQUIRE(unit.has_value());
    IccProfile decoded;
    REQUIRE(decode_icc_profile(unit->bytes, decoded, error));
    CHECK(decoded == sample_icc_profile());
    CHECK(serialize_icc_profile_unit(decoded) == unit->bytes);
  }
  {
    const auto unit = make_scan_type_unit(sample_scan_type(), error);
    REQUIRE(unit.has_value());
    ScanType decoded;
    REQUIRE(decode_scan_type(unit->bytes, decoded, error));
    CHECK(decoded == sample_scan_type());
    CHECK(serialize_scan_type_unit(decoded) == unit->bytes);
  }
  {
    const auto unit =
        make_temporal_point_info_unit(sample_temporal_point_info(), error);
    REQUIRE(unit.has_value());
    TemporalPointInfo decoded;
    REQUIRE(decode_temporal_point_info(unit->bytes, decoded, error));
    CHECK(decoded == sample_temporal_point_info());
    CHECK(serialize_temporal_point_info_unit(decoded) == unit->bytes);
  }
  {
    const auto unit =
        make_decoded_frame_hash_unit(sample_decoded_frame_hash(), error);
    REQUIRE(unit.has_value());
    DecodedFrameHash decoded;
    REQUIRE(decode_decoded_frame_hash(unit->bytes, decoded, error));
    CHECK(decoded == sample_decoded_frame_hash());
    CHECK(serialize_decoded_frame_hash_unit(decoded) == unit->bytes);
  }
  {
    const auto unit =
        make_user_data_unregistered_unit(sample_user_data(), error);
    REQUIRE(unit.has_value());
    UserDataUnregistered decoded;
    REQUIRE(decode_user_data_unregistered(unit->bytes, decoded, error));
    CHECK(decoded == sample_user_data());
    CHECK(serialize_user_data_unregistered_unit(decoded) == unit->bytes);
  }

  // Whole-frame (per_plane == 0) hash form.
  DecodedFrameHash whole;
  whole.per_plane = false;
  whole.has_grain = true;
  whole.is_monochrome = false;
  whole.frame_hash = hash_sequence(0xA0);
  const auto unit = make_decoded_frame_hash_unit(whole, error);
  REQUIRE(unit.has_value());
  CHECK(unit->bytes.size() == 17U);
  DecodedFrameHash decoded;
  REQUIRE(decode_decoded_frame_hash(unit->bytes, decoded, error));
  CHECK(decoded == whole);

  // ICC version helper (§6.16.9): 4.3 in the sample.
  const auto version = icc_profile_version(sample_icc_profile());
  REQUIRE(version.has_value());
  CHECK(version->major == 4U);
  CHECK(version->minor == 3U);
  IccProfile short_profile;
  short_profile.data.assign(9, 0);
  CHECK(!icc_profile_version(short_profile).has_value());
}

void test_timecode_forms() {
  std::string error;
  const std::uint8_t offset_lengths[] = {0, 1, 5, 7, 8, 16, 31};
  for (const std::uint8_t offset_length : offset_lengths) {
    for (int form = 0; form < 5; ++form) {
      Timecode value;
      value.counting_type = static_cast<std::uint8_t>(form % 7);
      value.discontinuity_flag = (form & 1) != 0;
      value.cnt_dropped_flag = (form & 2) != 0;
      value.n_frames = 511;
      value.time_offset_length = offset_length;
      if (offset_length > 0) {
        value.time_offset_value =
            offset_length == 31 ? 0x7FFFFFFFU
                                : ((1U << offset_length) - 1U);
      }
      if (form == 0) {
        value.full_timestamp_flag = true;
        value.seconds_value = 59;
        value.minutes_value = 59;
        value.hours_value = 23;
        value.seconds_flag = true;
        value.minutes_flag = true;
        value.hours_flag = true;
      } else if (form == 1) {
        value.full_timestamp_flag = false;  // no timestamp fields
      } else {
        value.full_timestamp_flag = false;
        value.seconds_flag = true;
        value.seconds_value = 59;
        if (form >= 3) {
          value.minutes_flag = true;
          value.minutes_value = 59;
        }
        if (form >= 4) {
          value.hours_flag = true;
          value.hours_value = 23;
        }
      }
      const auto unit = make_timecode_unit(value, error);
      REQUIRE(unit.has_value());
      CHECK(unit->bytes.size() == timecode_unit_size(value));
      Timecode decoded;
      REQUIRE(decode_timecode(unit->bytes, decoded, error));
      CHECK(decoded == value);
    }
  }

  // Full timestamp sets the implicit presence flags in the decoder.
  Timecode full;
  full.full_timestamp_flag = true;
  full.seconds_value = 1;
  full.minutes_value = 2;
  full.hours_value = 3;
  const auto unit = make_timecode_unit(full, error);
  REQUIRE(unit.has_value());
  Timecode decoded;
  REQUIRE(decode_timecode(unit->bytes, decoded, error));
  CHECK(decoded.seconds_flag);
  CHECK(decoded.minutes_flag);
  CHECK(decoded.hours_flag);
}

void test_banding_forms() {
  std::string error;

  const auto check_round_trip = [&error](const BandingHints& value) {
    const auto unit = make_banding_hints_unit(value, error);
    REQUIRE(unit.has_value());
    CHECK(unit->bytes.size() == banding_hints_unit_size(value));
    BandingHints decoded;
    REQUIRE(decode_banding_hints(unit->bytes, decoded, error));
    CHECK(decoded == value);
  };

  check_round_trip(BandingHints{});

  BandingHints coding_only;
  coding_only.coding_banding_present_flag = true;
  coding_only.source_banding_present_flag = true;
  check_round_trip(coding_only);

  BandingHints hints_no_components;
  hints_no_components.coding_banding_present_flag = true;
  hints_no_components.banding_hints_flag = true;
  check_round_trip(hints_no_components);

  BandingHints three_components;
  three_components.coding_banding_present_flag = true;
  three_components.banding_hints_flag = true;
  three_components.three_color_components_flag = true;
  for (auto& component : three_components.components) {
    component.banding_in_component_present_flag = true;
    component.max_band_width_minus_4 = 63;
    component.max_band_step_minus_1 = 15;
  }
  check_round_trip(three_components);

  BandingHints uniform;
  uniform.coding_banding_present_flag = true;
  uniform.banding_hints_flag = true;
  uniform.band_units_information_present_flag = true;
  uniform.num_band_units_rows_minus_1 = 1;
  uniform.num_band_units_cols_minus_1 = 2;
  uniform.banding_in_band_unit_present = {0, 1, 0, 1, 0, 1};
  check_round_trip(uniform);

  check_round_trip(sample_banding_hints());

  BandingHints maximum;
  maximum.coding_banding_present_flag = true;
  maximum.source_banding_present_flag = true;
  maximum.banding_hints_flag = true;
  maximum.three_color_components_flag = true;
  for (auto& component : maximum.components) {
    component.banding_in_component_present_flag = true;
    component.max_band_width_minus_4 = 63;
    component.max_band_step_minus_1 = 15;
  }
  maximum.band_units_information_present_flag = true;
  maximum.num_band_units_rows_minus_1 = 31;
  maximum.num_band_units_cols_minus_1 = 31;
  maximum.varying_size_band_units_flag = true;
  maximum.band_block_in_luma_samples = 7;
  maximum.vert_size_in_band_blocks_minus_1.assign(32, 31);
  maximum.horz_size_in_band_blocks_minus_1.assign(32, 31);
  maximum.banding_in_band_unit_present.assign(32U * 32U, 1);
  check_round_trip(maximum);
}

// ---------------------------------------------------------------------------
// Framing matrix: type x prefix/suffix x layer/persistence/necessity
// ---------------------------------------------------------------------------

void test_framing_matrix() {
  std::string error;
  const std::vector<MetadataUnitPayload> units = all_units();
  CHECK(units.size() == 8U);

  const hdr::MetadataLayer layers[] = {
      hdr::MetadataLayer::kUnspecified, hdr::MetadataLayer::kGlobal,
      hdr::MetadataLayer::kCurrent};
  const hdr::MetadataPersistence persistence[] = {
      hdr::MetadataPersistence::kGlobal,
      hdr::MetadataPersistence::kBasic,
      hdr::MetadataPersistence::kNoPersistence,
      hdr::MetadataPersistence::kEnhanced};

  for (const auto& unit : units) {
    // Short OBU, every layer/persistence and both placements.
    for (const auto layer : layers) {
      for (const auto persist : persistence) {
        for (int suffix = 0; suffix < 2; ++suffix) {
          hdr::MetadataObuConfig config;
          config.layer = layer;
          config.persistence = persist;
          config.suffix = suffix != 0;
          const std::vector<std::uint8_t> obu =
              make_short_obu(unit, config, error);
          REQUIRE(!obu.empty());
          hdr::MetadataShortView view;
          REQUIRE(parse_short_obu(obu, view, error));
          CHECK(view.type == static_cast<std::uint32_t>(unit.type));
          CHECK(view.layer == layer);
          CHECK(view.persistence == persist);
          CHECK(view.is_suffix == (suffix != 0));
          CHECK(!view.cancel);
          CHECK(std::equal(view.unit.begin(), view.unit.end(),
                           unit.bytes.begin(), unit.bytes.end()));
        }
      }
    }

    // Short cancel OBU (no payload).
    {
      hdr::MetadataObuConfig config;
      config.layer = hdr::MetadataLayer::kCurrent;
      const std::vector<std::uint8_t> obu =
          make_short_cancel_obu(unit.type, config, error);
      REQUIRE(!obu.empty());
      hdr::MetadataShortView view;
      REQUIRE(parse_short_obu(obu, view, error));
      CHECK(view.cancel);
      CHECK(view.unit.empty());
      CHECK(view.type == static_cast<std::uint32_t>(unit.type));
    }

    if (unit.type == Type::kTemporalPointInfo) {
      continue;  // short-only, covered below by the rejection test
    }

    // Group OBU with per-unit layer/persistence/priority.
    for (const auto layer : layers) {
      for (const auto persist : persistence) {
        for (const std::uint8_t priority : {std::uint8_t{0},
                                            std::uint8_t{128},
                                            std::uint8_t{255}}) {
          for (int suffix = 0; suffix < 2; ++suffix) {
            GroupUnit group;
            group.payload = unit;
            group.layer = layer;
            group.persistence = persist;
            group.priority = priority;
            hdr::MetadataObuConfig config;
            config.suffix = suffix != 0;
            config.necessity = hdr::MetadataNecessity::kMixed;
            config.application = hdr::MetadataApplicationId::kWearable;
            const std::vector<std::uint8_t> obu =
                make_group_obu(std::span<const GroupUnit>(&group, 1), config,
                               error);
            REQUIRE(!obu.empty());
            hdr::MetadataGroupView view;
            REQUIRE(parse_group_obu(obu, view, error));
            CHECK(view.is_suffix == (suffix != 0));
            CHECK(view.necessity == hdr::MetadataNecessity::kMixed);
            CHECK(view.application == hdr::MetadataApplicationId::kWearable);
            REQUIRE(view.units.size() == 1U);
            CHECK(view.units[0].type ==
                  static_cast<std::uint32_t>(unit.type));
            CHECK(view.units[0].layer == layer);
            CHECK(view.units[0].persistence == persist);
            CHECK(view.units[0].priority == priority);
            CHECK(!view.units[0].cancel);
            CHECK(std::equal(view.units[0].unit.begin(),
                             view.units[0].unit.end(), unit.bytes.begin(),
                             unit.bytes.end()));
          }
        }
      }
    }

    // Group cancel entry.
    {
      GroupUnit group;
      group.payload.type = unit.type;
      group.layer = hdr::MetadataLayer::kGlobal;
      group.persistence = hdr::MetadataPersistence::kNoPersistence;
      group.cancel = true;
      hdr::MetadataObuConfig config;
      const std::vector<std::uint8_t> obu =
          make_group_obu(std::span<const GroupUnit>(&group, 1), config, error);
      REQUIRE(!obu.empty());
      hdr::MetadataGroupView view;
      REQUIRE(parse_group_obu(obu, view, error));
      REQUIRE(view.units.size() == 1U);
      CHECK(view.units[0].cancel);
      CHECK(view.units[0].unit.empty());
    }
  }

  // Every necessity idc and application id representable in 5 bits.
  for (int necessity = 0; necessity < 4; ++necessity) {
    for (const std::uint8_t application : {std::uint8_t{0}, std::uint8_t{5},
                                            std::uint8_t{16},
                                            std::uint8_t{31}}) {
      GroupUnit group;
      group.payload = units[0];
      group.layer = hdr::MetadataLayer::kCurrent;
      hdr::MetadataObuConfig config;
      config.necessity = static_cast<hdr::MetadataNecessity>(necessity);
      config.application =
          static_cast<hdr::MetadataApplicationId>(application);
      const std::vector<std::uint8_t> obu =
          make_group_obu(std::span<const GroupUnit>(&group, 1), config, error);
      REQUIRE(!obu.empty());
      hdr::MetadataGroupView view;
      REQUIRE(parse_group_obu(obu, view, error));
      CHECK(static_cast<int>(view.necessity) == necessity);
      CHECK(static_cast<std::uint8_t>(view.application) == application);
    }
  }
}

void test_extra_unit_bytes() {
  std::string error;
  const std::vector<MetadataUnitPayload> units = all_units();

  const std::size_t bit_units[] = {
      static_cast<std::size_t>(Type::kTimecode),
      static_cast<std::size_t>(Type::kBandingHints),
      static_cast<std::size_t>(Type::kScanType),
      static_cast<std::size_t>(Type::kTemporalPointInfo),
      static_cast<std::size_t>(Type::kDecodedFrameHash)};
  for (const auto& unit : units) {
    if (std::find(std::begin(bit_units), std::end(bit_units),
                  static_cast<std::size_t>(unit.type)) ==
        std::end(bit_units)) {
      continue;
    }
    Value original;
    REQUIRE(decode_unit(unit.type, unit.bytes, original, error));
    for (std::size_t extra = 1; extra <= 3; ++extra) {
      std::vector<std::uint8_t> padded = unit.bytes;
      padded.insert(padded.end(), extra, 0x00);
      Value padded_value;
      REQUIRE(decode_unit(unit.type, padded, padded_value, error));
      CHECK(padded_value == original);

      // The writer must reject non-canonical padding even though the reader
      // accepts it.
      MetadataUnitPayload padded_unit = unit;
      padded_unit.bytes = padded;
      hdr::MetadataObuConfig config;
      CHECK(make_short_obu(padded_unit, config, error).empty());
    }
  }
}

// ---------------------------------------------------------------------------
// Malformed input rejection
// ---------------------------------------------------------------------------

void test_malformed_inputs() {
  std::string error;

  const auto expect_itut_fail = [&error](std::vector<std::uint8_t> bytes) {
    ItutT35 value;
    CHECK(!decode_itut_t35(bytes, value, error));
  };
  expect_itut_fail({});
  expect_itut_fail({0xFF});  // extension byte missing

  {
    const std::vector<std::uint8_t> bytes =
        serialize_timecode_unit(sample_timecode());
    for (std::size_t n = 0; n < bytes.size(); ++n) {
      const std::vector<std::uint8_t> cut(bytes.begin(),
                                          bytes.begin() +
                                              static_cast<std::ptrdiff_t>(n));
      Timecode value;
      CHECK(!decode_timecode(cut, value, error));
    }
  }
  {
    const std::vector<std::uint8_t> bytes =
        serialize_banding_hints_unit(sample_banding_hints());
    for (std::size_t n = 0; n < bytes.size(); ++n) {
      const std::vector<std::uint8_t> cut(bytes.begin(),
                                          bytes.begin() +
                                              static_cast<std::ptrdiff_t>(n));
      BandingHints value;
      CHECK(!decode_banding_hints(cut, value, error));
    }
  }
  {
    BandingHints maximum = sample_banding_hints();
    maximum.num_band_units_rows_minus_1 = 31;
    maximum.num_band_units_cols_minus_1 = 31;
    maximum.varying_size_band_units_flag = false;
    maximum.band_block_in_luma_samples = 0;
    maximum.vert_size_in_band_blocks_minus_1.clear();
    maximum.horz_size_in_band_blocks_minus_1.clear();
    maximum.banding_in_band_unit_present.assign(32U * 32U, 1);
    const auto unit = make_banding_hints_unit(maximum, error);
    REQUIRE(unit.has_value());
    std::vector<std::uint8_t> cut = unit->bytes;
    cut.pop_back();
    BandingHints value;
    CHECK(!decode_banding_hints(cut, value, error));
  }
  {
    const std::vector<std::uint8_t> bytes =
        serialize_decoded_frame_hash_unit(sample_decoded_frame_hash());
    for (std::size_t n = 0; n < bytes.size(); ++n) {
      const std::vector<std::uint8_t> cut(bytes.begin(),
                                          bytes.begin() +
                                              static_cast<std::ptrdiff_t>(n));
      DecodedFrameHash value;
      CHECK(!decode_decoded_frame_hash(cut, value, error));
    }
    std::vector<std::uint8_t> reserved = bytes;
    reserved[0] = static_cast<std::uint8_t>(reserved[0] | 0x01U);
    DecodedFrameHash value;
    CHECK(!decode_decoded_frame_hash(reserved, value, error));
  }
  {
    ScanType value;
    CHECK(!decode_scan_type({}, value, error));
  }
  {
    TemporalPointInfo value;
    const std::vector<std::uint8_t> empty;
    const std::vector<std::uint8_t> truncated = {0x80};
    const std::vector<std::uint8_t> too_long = {0x80, 0x80, 0x80, 0x80,
                                                0x80};
    const std::vector<std::uint8_t> overflow = {0xFF, 0xFF, 0xFF, 0xFF, 0x10};
    CHECK(!decode_temporal_point_info(empty, value, error));
    CHECK(!decode_temporal_point_info(truncated, value, error));
    CHECK(!decode_temporal_point_info(too_long, value, error));
    CHECK(!decode_temporal_point_info(overflow, value, error));
  }
  {
    UserDataUnregistered value;
    for (std::size_t n = 0; n < kUserDataUnregisteredUuidBytes; ++n) {
      const std::vector<std::uint8_t> cut(n, 0);
      CHECK(!decode_user_data_unregistered(cut, value, error));
    }
    const std::vector<std::uint8_t> uuid_only(16, 0x5A);
    CHECK(decode_user_data_unregistered(uuid_only, value, error));
    CHECK(value.uuid[0] == 0x5AU);
    CHECK(value.payload.empty());
  }
  {
    // Timecode cut in the middle of the full-timestamp fields: the sample
    // needs 39 syntax bits, so the first 3 bytes are not enough.
    const std::vector<std::uint8_t> bytes = {0x04, 0x15, 0x0E};
    Timecode value;
    CHECK(!decode_timecode(bytes, value, error));
  }
}

// ---------------------------------------------------------------------------
// Semantic validation
// ---------------------------------------------------------------------------

void test_validation() {
  {
    ItutT35 value = sample_itut_t35();
    value.payload.clear();
    CHECK(count_errors(validate_itut_t35(value)) == 1);
    value = sample_itut_t35();
    value.country_code_extension_byte = 0x01;
    CHECK(count_errors(validate_itut_t35(value)) == 1);
    value.country_code = 0xFF;
    CHECK(count_errors(validate_itut_t35(value)) == 0);
  }
  {
    Timecode value = sample_timecode();
    value.counting_type = 7;
    CHECK(count_warnings(validate_timecode(value, false)) == 1);
    CHECK(count_errors(validate_timecode(value, false)) == 0);
    CHECK(count_errors(validate_timecode(value, true)) == 1);

    value = sample_timecode();
    value.seconds_value = 60;
    CHECK(count_errors(validate_timecode(value)) == 1);
    value.minutes_value = 60;
    CHECK(count_errors(validate_timecode(value)) >= 1);
    value = sample_timecode();
    value.hours_value = 24;
    CHECK(count_errors(validate_timecode(value)) == 1);

    value = sample_timecode();
    value.n_frames = 512;
    CHECK(count_errors(validate_timecode(value)) == 1);

    value = sample_timecode();
    value.time_offset_length = 4;
    value.time_offset_value = 16;
    CHECK(count_errors(validate_timecode(value)) == 1);

    Timecode partial;
    partial.minutes_flag = true;  // without seconds_flag
    CHECK(count_errors(validate_timecode(partial)) >= 1);

    partial = Timecode{};
    partial.seconds_flag = true;
    partial.hours_flag = true;  // without minutes_flag
    CHECK(count_errors(validate_timecode(partial)) >= 1);
  }
  {
    BandingHints value;
    value.banding_hints_flag = true;  // coding_banding_present_flag == 0
    CHECK(count_errors(validate_banding_hints(value)) == 1);

    value = BandingHints{};
    value.coding_banding_present_flag = true;
    value.banding_hints_flag = true;
    value.components[1].banding_in_component_present_flag = true;
    CHECK(count_errors(validate_banding_hints(value)) == 1);

    value = sample_banding_hints();
    value.vert_size_in_band_blocks_minus_1.pop_back();
    CHECK(count_errors(validate_banding_hints(value)) == 1);

    value = sample_banding_hints();
    value.banding_in_band_unit_present[0] = 2;
    CHECK(count_errors(validate_banding_hints(value)) == 1);

    value = sample_banding_hints();
    value.num_band_units_rows_minus_1 = 32;
    CHECK(count_errors(validate_banding_hints(value)) >= 1);
  }
  {
    IccProfile value;
    value.data.assign(9, 0);
    CHECK(count_warnings(validate_icc_profile(value, false)) == 1);
    CHECK(count_errors(validate_icc_profile(value, false)) == 0);
    CHECK(count_errors(validate_icc_profile(value, true)) == 1);
  }
  {
    ScanType value;
    value.mps_pic_struct_type = 12;
    CHECK(count_errors(validate_scan_type(value)) == 0);
    value.mps_pic_struct_type = 13;
    CHECK(count_errors(validate_scan_type(value)) == 1);
    value.mps_pic_struct_type = 0;
    value.mps_source_scan_type_idc = 3;
    CHECK(count_errors(validate_scan_type(value)) == 0);
  }
  {
    DecodedFrameHash value;
    value.per_plane = false;
    value.hash_type = 1;
    CHECK(count_warnings(validate_decoded_frame_hash(value, false)) == 1);
    CHECK(count_errors(validate_decoded_frame_hash(value, false)) == 0);
    CHECK(count_errors(validate_decoded_frame_hash(value, true)) == 1);

    value = DecodedFrameHash{};
    value.plane_hash[0] = hash_sequence(0x10);
    CHECK(count_errors(validate_decoded_frame_hash(value)) == 1);

    value = DecodedFrameHash{};
    value.per_plane = true;
    value.is_monochrome = true;
    value.plane_hash[1] = hash_sequence(0x10);
    CHECK(count_errors(validate_decoded_frame_hash(value)) == 1);
  }
  {
    CHECK(count_errors(validate_temporal_point_info(
              sample_temporal_point_info())) == 0);
    CHECK(count_errors(
              validate_user_data_unregistered(sample_user_data())) == 0);
  }
}

// ---------------------------------------------------------------------------
// Writer validation
// ---------------------------------------------------------------------------

void test_writer_validation() {
  std::string error;
  const std::vector<MetadataUnitPayload> units = all_units();
  REQUIRE(units.size() == 8U);

  {
    hdr::MetadataObuConfig config;
    config.layer = hdr::MetadataLayer::kValues;
    CHECK(make_short_obu(units[0], config, error).empty());
    CHECK(!error.empty());
  }
  {
    hdr::MetadataObuConfig config;
    config.persistence = static_cast<hdr::MetadataPersistence>(4);
    CHECK(make_short_obu(units[0], config, error).empty());
  }
  {
    // Short-form necessity/application are implicit 0 and ignored by the
    // short writer; they are group-only knobs.
    hdr::MetadataObuConfig config;
    config.necessity = hdr::MetadataNecessity::kAdvisory;
    config.application = hdr::MetadataApplicationId::kTv;
    CHECK(!make_short_obu(units[0], config, error).empty());
  }
  {
    // Group framing takes layer/persistence/priority from each entry;
    // config.layer is unused (pinned to LAYER_CURRENT internally).
    hdr::MetadataObuConfig config;
    config.layer = hdr::MetadataLayer::kValues;
    CHECK(!make_group_obu(group_units(), config, error).empty());
  }
  {
    hdr::MetadataObuConfig config;
    config.cancel = true;
    CHECK(make_short_obu(units[0], config, error).empty());
  }
  {
    hdr::MetadataObuConfig config;
    MetadataUnitPayload unknown;
    unknown.type = static_cast<Type>(11);
    unknown.bytes = {0x00};
    CHECK(make_short_obu(unknown, config, error).empty());
    CHECK(make_short_cancel_obu(static_cast<Type>(11), config, error)
              .empty());
  }
  {
    hdr::MetadataObuConfig config;
    MetadataUnitPayload padded = units[1];
    padded.bytes.push_back(0x00);
    CHECK(make_short_obu(padded, config, error).empty());
  }
  {
    hdr::MetadataObuConfig config;
    config.application = static_cast<hdr::MetadataApplicationId>(6);
    CHECK(make_group_obu(group_units(), config, error).empty());
    config.application = static_cast<hdr::MetadataApplicationId>(32);
    CHECK(make_group_obu(group_units(), config, error).empty());
  }
  {
    hdr::MetadataObuConfig config;
    config.cancel = true;
    CHECK(make_group_obu(group_units(), config, error).empty());
  }
  {
    hdr::MetadataObuConfig config;
    CHECK(make_group_obu({}, config, error).empty());
  }
  {
    // §6.16.3: metadata_unit_cnt_minus_1 < 16383, i.e. at most 16383 units.
    std::vector<GroupUnit> too_many(16384);
    for (auto& entry : too_many) {
      entry.payload.type = Type::kItutT35;
      entry.layer = hdr::MetadataLayer::kCurrent;
      entry.cancel = true;
    }
    hdr::MetadataObuConfig config;
    CHECK(make_group_obu(too_many, config, error).empty());
  }
  {
    hdr::MetadataObuConfig config;
    GroupUnit group;
    group.payload.type = Type::kTemporalPointInfo;
    group.payload.bytes = serialize_temporal_point_info_unit(
        sample_temporal_point_info());
    CHECK(make_group_obu(std::span<const GroupUnit>(&group, 1), config, error)
              .empty());
  }
  {
    hdr::MetadataObuConfig config;
    GroupUnit group;
    group.payload = units[0];
    group.cancel = true;  // cancel must not carry a payload
    CHECK(make_group_obu(std::span<const GroupUnit>(&group, 1), config, error)
              .empty());
  }
  {
    hdr::MetadataObuConfig config;
    GroupUnit group;
    group.payload = units[0];
    group.layer = hdr::MetadataLayer::kValues;
    CHECK(make_group_obu(std::span<const GroupUnit>(&group, 1), config, error)
              .empty());
  }
  {
    hdr::MetadataObuConfig config;
    GroupUnit group;
    group.payload = units[0];
    group.persistence = static_cast<hdr::MetadataPersistence>(4);
    CHECK(make_group_obu(std::span<const GroupUnit>(&group, 1), config, error)
              .empty());
  }
  {
    // Strict range validation reaches the framing writer.
    DecodedFrameHash value;
    value.hash_type = 1;
    const auto unit = make_decoded_frame_hash_unit(value, error, false);
    REQUIRE(unit.has_value());
    hdr::MetadataObuConfig lenient;
    CHECK(!make_short_obu(*unit, lenient, error).empty());
    hdr::MetadataObuConfig strict;
    strict.strict_ranges = true;
    CHECK(make_short_obu(*unit, strict, error).empty());
  }
}

// ---------------------------------------------------------------------------
// Group framing with HDR units (mixed group)
// ---------------------------------------------------------------------------

void test_mixed_group_with_hdr() {
  std::string error;
  GroupUnit cll;
  cll.payload.type = Type::kHdrCll;
  cll.payload.bytes = hdr::serialize_cll_unit(hdr::ContentLightLevel{1000, 400});
  cll.layer = hdr::MetadataLayer::kCurrent;

  GroupUnit mdcv;
  mdcv.payload.type = Type::kHdrMdcv;
  hdr::MasteringDisplayColorVolume volume;
  volume.primary_chromaticity_x = {8500, 6550, 35400};
  volume.primary_chromaticity_y = {39850, 2300, 14600};
  volume.white_point_chromaticity_x = 15635;
  volume.white_point_chromaticity_y = 16450;
  volume.luminance_max = 10000000;
  volume.luminance_min = 50;
  mdcv.payload.bytes = hdr::serialize_mdcv_unit(volume);

  GroupUnit timecode;
  const auto tc = make_timecode_unit(sample_timecode(), error);
  REQUIRE(tc.has_value());
  timecode.payload = *tc;

  const GroupUnit units[] = {cll, mdcv, timecode};
  hdr::MetadataObuConfig config;
  const std::vector<std::uint8_t> obu =
      make_group_obu(units, config, error);
  REQUIRE(!obu.empty());
  hdr::MetadataGroupView view;
  REQUIRE(parse_group_obu(obu, view, error));
  REQUIRE(view.units.size() == 3U);

  hdr::ContentLightLevel decoded_cll;
  REQUIRE(hdr::decode_cll(view.units[0].unit, decoded_cll, error));
  CHECK((decoded_cll == hdr::ContentLightLevel{1000, 400}));
  hdr::MasteringDisplayColorVolume decoded_mdcv;
  REQUIRE(hdr::decode_mdcv(view.units[1].unit, decoded_mdcv, error));
  CHECK(decoded_mdcv == volume);
  Timecode decoded_tc;
  REQUIRE(decode_timecode(view.units[2].unit, decoded_tc, error));
  CHECK(decoded_tc == sample_timecode());

  // The same generic short framing also accepts HDR units, and the scanner
  // decodes them through the HDR module.
  MetadataUnitPayload cll_payload;
  cll_payload.type = Type::kHdrCll;
  cll_payload.bytes =
      hdr::serialize_cll_unit(hdr::ContentLightLevel{1000, 400});
  hdr::MetadataObuConfig short_config;
  const std::vector<std::uint8_t> cll_obu =
      make_short_obu(cll_payload, short_config, error);
  REQUIRE(!cll_obu.empty());

  std::vector<std::uint8_t> with_cll;
  REQUIRE(hdr::insert_metadata_obus(
      make_test_temporal_unit(), cll_obu,
      hdr::MetadataPlacement::kBeforeFirstFrame, with_cll, error));
  ScanResult scan;
  REQUIRE(scan_metadata(with_cll, scan, error));
  CHECK(scan.supported_units == 1U);
  REQUIRE(scan.records.size() == 1U);
  CHECK(scan.records[0].metadata_type ==
        static_cast<std::uint32_t>(Type::kHdrCll));
  REQUIRE(std::holds_alternative<hdr::ContentLightLevel>(
      scan.records[0].value));
  CHECK(std::get<hdr::ContentLightLevel>(scan.records[0].value) ==
        (hdr::ContentLightLevel{1000, 400}));
}

// ---------------------------------------------------------------------------
// Stream scanning
// ---------------------------------------------------------------------------

void test_scan() {
  std::string error;
  const std::vector<MetadataUnitPayload> units = all_units();
  REQUIRE(units.size() == 8U);

  hdr::MetadataObuConfig prefix_config;
  std::vector<std::uint8_t> prefix_obus;
  for (const auto& unit : units) {
    const std::vector<std::uint8_t> obu =
        make_short_obu(unit, prefix_config, error);
    REQUIRE(!obu.empty());
    prefix_obus.insert(prefix_obus.end(), obu.begin(), obu.end());
  }

  hdr::MetadataObuConfig suffix_config;
  suffix_config.suffix = true;
  const std::vector<std::uint8_t> group =
      make_group_obu(group_units(), suffix_config, error);
  REQUIRE(!group.empty());

  const std::vector<std::uint8_t> base = make_test_temporal_unit();
  std::vector<std::uint8_t> with_prefix;
  REQUIRE(hdr::insert_metadata_obus(base, prefix_obus,
                                    hdr::MetadataPlacement::kBeforeFirstFrame,
                                    with_prefix, error));
  std::vector<std::uint8_t> with_both;
  REQUIRE(hdr::insert_metadata_obus(with_prefix, group,
                                    hdr::MetadataPlacement::kEndOfTemporalUnit,
                                    with_both, error));

  ScanResult scan;
  REQUIRE(scan_metadata(with_both, scan, error));
  CHECK(scan.metadata_obus == 9U);
  CHECK(scan.metadata_units == 15U);
  CHECK(scan.supported_units == 15U);
  CHECK(scan.unsupported_units == 0U);
  CHECK(scan.invalid_units == 0U);
  CHECK(scan.cancelled_units == 0U);
  REQUIRE(scan.records.size() == 15U);

  for (std::size_t i = 0; i < 8U; ++i) {
    const auto& record = scan.records[i];
    CHECK(record.metadata_type == static_cast<std::uint32_t>(units[i].type));
    CHECK(!record.in_group);
    CHECK(!record.is_suffix);
    CHECK(record.before_first_frame);
    CHECK(!record.cancel);
    CHECK(record.has_value);
    CHECK(record.layer == hdr::MetadataLayer::kCurrent);
    CHECK(record.persistence == hdr::MetadataPersistence::kGlobal);
    CHECK(record.priority == 0U);
  }
  CHECK(scan.records[0].value == Value{sample_itut_t35()});
  CHECK(scan.records[1].value == Value{sample_timecode()});
  CHECK(scan.records[2].value == Value{sample_decoded_frame_hash()});
  CHECK(scan.records[3].value == Value{sample_banding_hints()});
  CHECK(scan.records[4].value == Value{sample_icc_profile()});
  CHECK(scan.records[5].value == Value{sample_scan_type()});
  CHECK(scan.records[6].value == Value{sample_temporal_point_info()});
  CHECK(scan.records[7].value == Value{sample_user_data()});

  const std::vector<GroupUnit> groups = group_units();
  for (std::size_t i = 0; i < groups.size(); ++i) {
    const auto& record = scan.records[8U + i];
    CHECK(record.metadata_type ==
          static_cast<std::uint32_t>(groups[i].payload.type));
    CHECK(record.in_group);
    CHECK(record.is_suffix);
    CHECK(!record.before_first_frame);
    CHECK(!record.cancel);
    CHECK(record.has_value);
    CHECK(record.layer == hdr::MetadataLayer::kCurrent);
    CHECK(record.persistence == hdr::MetadataPersistence::kGlobal);
    CHECK(record.priority == groups[i].priority);
  }

  // Cancelled units are reported without a value.
  hdr::MetadataObuConfig cancel_config;
  cancel_config.suffix = true;
  const std::vector<std::uint8_t> cancel =
      make_short_cancel_obu(Type::kTimecode, cancel_config, error);
  REQUIRE(!cancel.empty());
  std::vector<std::uint8_t> with_cancel;
  REQUIRE(hdr::insert_metadata_obus(with_both, cancel,
                                    hdr::MetadataPlacement::kEndOfTemporalUnit,
                                    with_cancel, error));
  ScanResult cancel_scan;
  REQUIRE(scan_metadata(with_cancel, cancel_scan, error));
  CHECK(cancel_scan.metadata_units == 16U);
  CHECK(cancel_scan.cancelled_units == 1U);
  REQUIRE(cancel_scan.records.size() == 16U);
  CHECK(cancel_scan.records.back().cancel);
  CHECK(!cancel_scan.records.back().has_value);
  CHECK(cancel_scan.records.back().metadata_type ==
        static_cast<std::uint32_t>(Type::kTimecode));
}

void test_scan_unknown_invalid() {
  std::string error;
  std::vector<std::uint8_t> unit = make_test_temporal_unit();

  hdr::MetadataObuConfig suffix;
  suffix.suffix = true;

  // Reserved metadata_type 11: structurally valid, not decodable here.
  {
    const std::vector<std::uint8_t> obu =
        hdr::make_metadata_short_obu(
            static_cast<Type>(11), std::vector<std::uint8_t>{0x00}, suffix,
            error);
    REQUIRE(!obu.empty());
    std::vector<std::uint8_t> out;
    REQUIRE(hdr::insert_metadata_obus(unit, obu,
                                      hdr::MetadataPlacement::kEndOfTemporalUnit,
                                      out, error));
    unit = std::move(out);
  }
  // Known type with a truncated payload: invalid.
  {
    const std::vector<std::uint8_t> obu =
        hdr::make_metadata_short_obu(
            Type::kTimecode, std::vector<std::uint8_t>{0x00}, suffix, error);
    REQUIRE(!obu.empty());
    std::vector<std::uint8_t> out;
    REQUIRE(hdr::insert_metadata_obus(unit, obu,
                                      hdr::MetadataPlacement::kEndOfTemporalUnit,
                                      out, error));
    unit = std::move(out);
  }
  // A group carrying temporal point info (forbidden by §6.16.11): invalid.
  {
    hdr::MetadataGroupEntry entry;
    entry.type = Type::kTemporalPointInfo;
    entry.unit = {0x00};
    entry.layer = hdr::MetadataLayer::kCurrent;
    hdr::MetadataObuConfig config;
    config.suffix = true;
    const std::vector<std::uint8_t> obu =
        hdr::make_metadata_group_obu(std::span<const hdr::MetadataGroupEntry>(
                                         &entry, 1),
                                     config, error);
    REQUIRE(!obu.empty());
    std::vector<std::uint8_t> out;
    REQUIRE(hdr::insert_metadata_obus(unit, obu,
                                      hdr::MetadataPlacement::kEndOfTemporalUnit,
                                      out, error));
    unit = std::move(out);
  }

  ScanResult scan;
  REQUIRE(scan_metadata(unit, scan, error));
  CHECK(scan.metadata_obus == 3U);
  CHECK(scan.metadata_units == 3U);
  CHECK(scan.supported_units == 0U);
  CHECK(scan.unsupported_units == 1U);
  CHECK(scan.invalid_units == 2U);
  CHECK(scan.records.empty());
}

// ---------------------------------------------------------------------------
// Misc API
// ---------------------------------------------------------------------------

void test_api_helpers() {
  CHECK(is_known_type(Type::kHdrCll));
  CHECK(is_known_type(Type::kUserDataUnregistered));
  CHECK(!is_known_type(static_cast<Type>(0)));
  CHECK(!is_known_type(static_cast<Type>(11)));
  CHECK(is_group_allowed(Type::kItutT35));
  CHECK(!is_group_allowed(Type::kTemporalPointInfo));

  CHECK(std::string_view(type_name(Type::kItutT35)) == "itut_t35");
  CHECK(std::string_view(type_name(Type::kTimecode)) == "timecode");
  CHECK(std::string_view(type_name(Type::kDecodedFrameHash)) ==
        "decoded_frame_hash");
  CHECK(std::string_view(type_name(Type::kBandingHints)) == "banding_hints");
  CHECK(std::string_view(type_name(Type::kIccProfile)) == "icc_profile");
  CHECK(std::string_view(type_name(Type::kScanType)) == "scan_type");
  CHECK(std::string_view(type_name(Type::kTemporalPointInfo)) ==
        "temporal_point_info");
  CHECK(std::string_view(type_name(Type::kUserDataUnregistered)) ==
        "user_data_unregistered");
  CHECK(std::string_view(type_name(static_cast<Type>(11))) == "reserved");

  // §6.16.10 Table 6.18 restrictions.
  CHECK(scan_type_required_ci_scan_type_idc(0) == 1);
  CHECK(scan_type_required_ci_scan_type_idc(7) == 1);
  CHECK(scan_type_required_ci_scan_type_idc(8) == 1);
  for (const std::uint8_t value :
       {std::uint8_t{1}, std::uint8_t{2}, std::uint8_t{9},
        std::uint8_t{10}, std::uint8_t{11}, std::uint8_t{12}}) {
    CHECK(scan_type_required_ci_scan_type_idc(value) == 2);
  }
  for (const std::uint8_t value :
       {std::uint8_t{3}, std::uint8_t{4}, std::uint8_t{5},
        std::uint8_t{6}}) {
    CHECK(scan_type_required_ci_scan_type_idc(value) == 3);
  }
  CHECK(!scan_type_required_ci_scan_type_idc(13).has_value());
  CHECK(!scan_type_required_ci_scan_type_idc(31).has_value());
}

void test_canonical_unit_size() {
  const std::vector<MetadataUnitPayload> units = all_units();
  const std::size_t expected[] = {6, 5, 49, 8, 16, 1, 2, 20};
  REQUIRE(units.size() == 8U);
  for (std::size_t i = 0; i < units.size(); ++i) {
    Value value;
    std::string error;
    REQUIRE(decode_unit(units[i].type, units[i].bytes, value, error));
    CHECK(canonical_unit_size(value) == expected[i]);
    CHECK(units[i].bytes.size() == expected[i]);
  }
}

}  // namespace

int main() {
  const struct {
    const char* name;
    void (*run)();
  } tests[] = {
      {"unit_goldens", test_unit_goldens},
      {"short_obu_goldens", test_short_obu_goldens},
      {"group_obu_goldens", test_group_obu_goldens},
      {"unit_sizes", test_unit_sizes},
      {"round_trip", test_round_trip},
      {"timecode_forms", test_timecode_forms},
      {"banding_forms", test_banding_forms},
      {"framing_matrix", test_framing_matrix},
      {"extra_unit_bytes", test_extra_unit_bytes},
      {"malformed_inputs", test_malformed_inputs},
      {"validation", test_validation},
      {"writer_validation", test_writer_validation},
      {"mixed_group_with_hdr", test_mixed_group_with_hdr},
      {"scan", test_scan},
      {"scan_unknown_invalid", test_scan_unknown_invalid},
      {"api_helpers", test_api_helpers},
      {"canonical_unit_size", test_canonical_unit_size},
  };

  for (const auto& test : tests) {
    const int before = g_failures;
    test.run();
    std::cout << (g_failures == before ? "PASS " : "FAIL ") << test.name
              << '\n';
  }

  if (g_failures != 0) {
    std::cerr << "metadata_types_test: " << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "metadata_types_test: PASS\n";
  return 0;
}
