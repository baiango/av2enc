// Unit tests for src/encoder/metadata/gaps/.
//
// Covers §5.17.2/§5.17.3/§6.16.2/§6.16.3 gaps the shared metadata module does
// not write:
//   * muh_cancel_flag == 1 with the metadata_type signalled and no unit body;
//   * muh_layer_idc == LAYER_VALUES with muh_xlayer_map f(32) and one
//     muh_mlayer_map f(8) per set bit (GLOBAL_XLAYER_ID) or a single map;
//   * muh_header_extension_byte f(8);
//   * OBU header extension for obu_mlayer_id/obu_xlayer_id.
//
// Goldens are hand-derived from the syntax tables (each hex string is built
// field by field in the comment above it) and then round-tripped through the
// independent reader.

#include "encoder/metadata/gaps/metadata_gaps.h"

#include "bitstream/m0_headers.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace av2enc;
using namespace av2enc::metadata;
using namespace av2enc::metadata::gaps;

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

std::vector<std::uint8_t> wrap_raw(std::uint8_t type,
                                   const GapObuConfig& config,
                                   std::span<const std::uint8_t> payload) {
  m0::ObuHeader header;
  header.type = static_cast<m0::ObuType>(type);
  header.tlayer_id = config.tlayer_id;
  header.extension_flag =
      config.mlayer_id != 0U || config.xlayer_id != 0U;
  header.mlayer_id = config.mlayer_id;
  header.xlayer_id = config.xlayer_id;
  return m0::make_obu(header, payload);
}

// §5.17.5 CLL unit: max_cll = 1000 (0x03E8), max_fall = 400 (0x0190).
std::vector<std::uint8_t> cll_unit() {
  hdr::ContentLightLevel cll;
  cll.max_cll = 1000;
  cll.max_fall = 400;
  return hdr::serialize_cll_unit(cll);
}

MetadataGapUnit cll_gap_unit() {
  MetadataGapUnit unit;
  unit.header.type = Type::kHdrCll;
  unit.payload = cll_unit();
  return unit;
}

bool same_payload(std::span<const std::uint8_t> parsed,
                  const std::vector<std::uint8_t>& expected) {
  return std::equal(parsed.begin(), parsed.end(), expected.begin(),
                    expected.end());
}

// ---------------------------------------------------------------------------
// Pinned goldens
// ---------------------------------------------------------------------------

// Short cancel: options(1) metadata_type leb128(1) trailing_bits(1).
//   options = LAYER_CURRENT(2)<<4 | cancel(1)<<3 = 0x28
//   metadata_type = 4 (TIMEcode) -> 0x04
//   payload = 28 04 80 ; OBU header = type8<<2 = 0x20 ; size = 4
void test_short_cancel_golden() {
  GapObuConfig config;
  std::string error;
  const auto obu = make_short_cancel_obu(Type::kTimecode, config, error);
  REQUIRE(obu.size() == 5U);
  CHECK_HEX(obu, "0420280480");

  ParsedShortObu view;
  REQUIRE(parse_short_obu(obu, view, error));
  CHECK(view.cancel);
  CHECK(view.type == 4U);
  CHECK(view.layer == hdr::MetadataLayer::kCurrent);
  CHECK(view.persistence == hdr::MetadataPersistence::kGlobal);
  CHECK(view.payload.empty());
  CHECK(!view.is_suffix);
  std::cout << "PASS short_cancel_golden\n";
}

void test_short_cancel_suffix_golden() {
  GapObuConfig config;
  config.suffix = true;
  MetadataGapUnit unit;
  unit.header.type = Type::kHdrMdcv;
  unit.header.cancel = true;
  unit.header.layer = hdr::MetadataLayer::kGlobal;
  unit.header.persistence = hdr::MetadataPersistence::kNoPersistence;
  std::string error;
  const auto obu = make_short_obu(unit, config, error);
  // options = 0x80 | 1<<4 | 1<<3 | 2 = 0x9A
  CHECK_HEX(obu, "04209a0280");
  ParsedShortObu view;
  REQUIRE(parse_short_obu(obu, view, error));
  CHECK(view.is_suffix);
  CHECK(view.cancel);
  CHECK(view.type == 2U);
  CHECK(view.layer == hdr::MetadataLayer::kGlobal);
  CHECK(view.persistence == hdr::MetadataPersistence::kNoPersistence);
  std::cout << "PASS short_cancel_suffix_golden\n";
}

// Local LAYER_VALUES group:
//   23 options = necessity(1)<<5 | application 3 (TV)
//   00 unit_cnt_minus_1
//   01 metadata_type = HDR_CLL
//   08 muh_header_size = 4 (leb(payload)+2+1 map) << 1
//   04 muh_payload_size = 4
//   65 muh_layer_idc=3<<5 | persistence=1<<2 | priority>>6=1
//   68 (priority & 0x3f) << 2  (priority = 0x5A)
//   02 muh_mlayer_map
//   03 e8 01 90 metadata_unit
//   80 trailing_bits
void test_group_layer_values_local_golden() {
  GapObuConfig config;
  config.necessity = hdr::MetadataNecessity::kNecessary;
  config.application = hdr::MetadataApplicationId::kTv;
  config.xlayer_id = 0;
  config.mlayer_id = 0;

  MetadataGapUnit unit = cll_gap_unit();
  unit.header.layer = hdr::MetadataLayer::kValues;
  unit.header.persistence = hdr::MetadataPersistence::kBasic;
  unit.header.priority = 0x5A;
  unit.header.layer_values.mlayer_map = {0x02};

  std::string error;
  const auto obu = make_group_obu(std::span<const MetadataGapUnit>(&unit, 1),
                                  config, error);
  REQUIRE(!obu.empty());
  CHECK_HEX(obu, "0e24230001080465680203e8019080");

  ParsedGroupObu view;
  REQUIRE(parse_group_obu(obu, view, error));
  REQUIRE(view.units.size() == 1U);
  CHECK(view.necessity == hdr::MetadataNecessity::kNecessary);
  CHECK(view.application == hdr::MetadataApplicationId::kTv);
  CHECK(view.xlayer_id == 0U);
  CHECK(view.units[0].header.layer == hdr::MetadataLayer::kValues);
  CHECK(view.units[0].header.persistence ==
        hdr::MetadataPersistence::kBasic);
  CHECK(view.units[0].header.priority == 0x5AU);
  CHECK(view.units[0].header.layer_values.mlayer_map ==
        std::vector<std::uint8_t>{0x02});
  CHECK(same_payload(view.units[0].payload, cll_unit()));
  std::cout << "PASS group_layer_values_local_golden\n";
}

// Global LAYER_VALUES group (obu_header_extension_flag = 1):
//   OBU header a4 = 0x80 | type9<<2 ; extension 1f = mlayer 0<<5 | xlayer 31
//   23 00 01 12 04 65 68 00 00 00 05 01 04 03 e8 01 90 80
//   header_size = 9 = leb(4) + 2 + xlayer_map(4) + 2 mlayer maps
//   muh_xlayer_map = 5 (bits 0 and 2) -> mlayer maps 01 and 04
void test_group_layer_values_global_golden() {
  GapObuConfig config;
  config.necessity = hdr::MetadataNecessity::kNecessary;
  config.application = hdr::MetadataApplicationId::kTv;
  config.xlayer_id = kGlobalXlayerId;

  MetadataGapUnit unit = cll_gap_unit();
  unit.header.layer = hdr::MetadataLayer::kValues;
  unit.header.persistence = hdr::MetadataPersistence::kBasic;
  unit.header.priority = 0x5A;
  unit.header.layer_values.xlayer_map = 0x00000005U;
  unit.header.layer_values.mlayer_map = {0x01, 0x04};

  std::string error;
  const auto obu = make_group_obu(std::span<const MetadataGapUnit>(&unit, 1),
                                  config, error);
  REQUIRE(!obu.empty());
  CHECK_HEX(obu, "14a41f2300011204656800000005010403e8019080");

  ParsedGroupObu view;
  REQUIRE(parse_group_obu(obu, view, error));
  REQUIRE(view.units.size() == 1U);
  CHECK(view.xlayer_id == kGlobalXlayerId);
  CHECK(view.units[0].header.layer_values.xlayer_map == 0x05U);
  CHECK(view.units[0].header.layer_values.mlayer_map ==
        (std::vector<std::uint8_t>{0x01, 0x04}));
  CHECK(same_payload(view.units[0].payload, cll_unit()));
  std::cout << "PASS group_layer_values_global_golden\n";
}

// Header extension bytes follow the layer maps and are counted by
// muh_header_size: header_size = 4 + 2 = 6 -> 0x0C.
void test_group_header_extension_golden() {
  GapObuConfig config;
  config.necessity = hdr::MetadataNecessity::kNecessary;
  config.application = hdr::MetadataApplicationId::kTv;

  MetadataGapUnit unit = cll_gap_unit();
  unit.header.layer = hdr::MetadataLayer::kValues;
  unit.header.persistence = hdr::MetadataPersistence::kBasic;
  unit.header.priority = 0x5A;
  unit.header.layer_values.mlayer_map = {0x02};
  unit.header.header_extension = {0xAA, 0xBB};

  std::string error;
  const auto obu = make_group_obu(std::span<const MetadataGapUnit>(&unit, 1),
                                  config, error);
  REQUIRE(!obu.empty());
  CHECK_HEX(obu, "10242300010c04656802aabb03e8019080");

  ParsedGroupObu view;
  REQUIRE(parse_group_obu(obu, view, error));
  REQUIRE(view.units.size() == 1U);
  CHECK(view.units[0].header.header_extension ==
        (std::vector<std::uint8_t>{0xAA, 0xBB}));
  CHECK(view.units[0].header.layer_values.mlayer_map ==
        (std::vector<std::uint8_t>{0x02}));
  CHECK(same_payload(view.units[0].payload, cll_unit()));
  std::cout << "PASS group_header_extension_golden\n";
}

// Cancel inside a group: header_size = extension bytes only (2) -> 0x05.
void test_group_cancel_golden() {
  GapObuConfig config;
  config.necessity = hdr::MetadataNecessity::kNecessary;
  config.application = hdr::MetadataApplicationId::kTv;

  MetadataGapUnit unit;
  unit.header.type = Type::kTimecode;
  unit.header.cancel = true;
  unit.header.header_extension = {0x11, 0x22};

  std::string error;
  const auto obu = make_group_obu(std::span<const MetadataGapUnit>(&unit, 1),
                                  config, error);
  REQUIRE(!obu.empty());
  CHECK_HEX(obu, "082423000405112280");

  ParsedGroupObu view;
  REQUIRE(parse_group_obu(obu, view, error));
  REQUIRE(view.units.size() == 1U);
  CHECK(view.units[0].header.cancel);
  CHECK(view.units[0].header.type == Type::kTimecode);
  CHECK(view.units[0].payload.empty());
  CHECK(view.units[0].header.header_extension ==
        (std::vector<std::uint8_t>{0x11, 0x22}));
  std::cout << "PASS group_cancel_golden\n";
}

// Short OBU with an OBU header extension: xlayer 5, mlayer 1.
//   OBU header a0 = 0x80 | type8<<2 ; extension 25 = 1<<5 | 5
void test_short_obu_header_extension_golden() {
  GapObuConfig config;
  config.mlayer_id = 1;
  config.xlayer_id = 5;

  MetadataGapUnit unit = cll_gap_unit();
  std::string error;
  const auto obu = make_short_obu(unit, config, error);
  REQUIRE(!obu.empty());
  CHECK_HEX(obu, "09a025200103e8019080");

  ParsedShortObu view;
  REQUIRE(parse_short_obu(obu, view, error));
  CHECK(!view.cancel);
  CHECK(view.type == static_cast<std::uint32_t>(Type::kHdrCll));
  CHECK(same_payload(view.payload, cll_unit()));
  std::cout << "PASS short_obu_header_extension_golden\n";
}

// ---------------------------------------------------------------------------
// Cross-check against the shared HDR group writer
// ---------------------------------------------------------------------------

void test_plain_group_matches_hdr_writer() {
  hdr::MetadataObuConfig hdr_config;
  hdr_config.suffix = false;
  hdr_config.necessity = hdr::MetadataNecessity::kNecessary;
  hdr_config.application = hdr::MetadataApplicationId::kTv;

  hdr::MetadataGroupEntry entry;
  entry.type = hdr::MetadataType::kHdrCll;
  entry.unit = cll_unit();
  entry.layer = hdr::MetadataLayer::kCurrent;
  entry.persistence = hdr::MetadataPersistence::kGlobal;
  entry.priority = 0x5A;

  std::string hdr_error;
  const auto expected = hdr::make_metadata_group_obu(
      std::span<const hdr::MetadataGroupEntry>(&entry, 1), hdr_config,
      hdr_error);
  REQUIRE(!expected.empty());

  GapObuConfig config;
  config.necessity = hdr::MetadataNecessity::kNecessary;
  config.application = hdr::MetadataApplicationId::kTv;
  MetadataGapUnit unit = cll_gap_unit();
  unit.header.priority = 0x5A;
  std::string error;
  const auto got = make_group_obu(std::span<const MetadataGapUnit>(&unit, 1),
                                  config, error);
  REQUIRE(!got.empty());
  CHECK_HEX(got, to_hex(expected));

  ParsedGroupObu view;
  REQUIRE(parse_group_obu(expected, view, error));
  REQUIRE(view.units.size() == 1U);
  CHECK(view.units[0].header.priority == 0x5AU);
  CHECK(same_payload(view.units[0].payload, cll_unit()));
  std::cout << "PASS plain_group_matches_hdr_writer\n";
}

// ---------------------------------------------------------------------------
// Round trips
// ---------------------------------------------------------------------------

MetadataGapUnit make_unit(Type type, std::vector<std::uint8_t> payload) {
  MetadataGapUnit unit;
  unit.header.type = type;
  unit.payload = std::move(payload);
  return unit;
}

void test_short_round_trip_matrix() {
  Timecode timecode;
  timecode.full_timestamp_flag = true;
  timecode.n_frames = 42;
  timecode.seconds_value = 7;
  timecode.minutes_value = 3;
  timecode.hours_value = 1;

  ScanType scan;
  scan.mps_pic_struct_type = 1;
  scan.mps_source_scan_type_idc = 2;

  TemporalPointInfo tpi;
  tpi.frame_presentation_time = 1000;

  UserDataUnregistered user;
  user.uuid.fill(0x11);
  user.payload = {0xDE, 0xAD};

  const std::vector<MetadataGapUnit> units = {
      make_unit(Type::kHdrCll, cll_unit()),
      make_unit(Type::kTimecode, serialize_timecode_unit(timecode)),
      make_unit(Type::kScanType, serialize_scan_type_unit(scan)),
      make_unit(Type::kTemporalPointInfo,
                serialize_temporal_point_info_unit(tpi)),
      make_unit(Type::kUserDataUnregistered,
                serialize_user_data_unregistered_unit(user)),
      make_unit(Type::kHdrMdcv, [] {
        hdr::MasteringDisplayColorVolume mdcv;
        mdcv.primary_chromaticity_x = {1000, 2000, 3000};
        mdcv.primary_chromaticity_y = {4000, 5000, 6000};
        mdcv.white_point_chromaticity_x = 7000;
        mdcv.white_point_chromaticity_y = 8000;
        mdcv.luminance_max = 10000;
        mdcv.luminance_min = 50;
        return hdr::serialize_mdcv_unit(mdcv);
      }()),
  };

  std::size_t checked = 0;
  for (const auto& unit : units) {
    for (const bool suffix : {false, true}) {
      for (const auto layer :
           {hdr::MetadataLayer::kUnspecified, hdr::MetadataLayer::kGlobal,
            hdr::MetadataLayer::kCurrent}) {
        for (const auto persistence :
             {hdr::MetadataPersistence::kGlobal,
              hdr::MetadataPersistence::kBasic,
              hdr::MetadataPersistence::kNoPersistence,
              hdr::MetadataPersistence::kEnhanced}) {
          GapObuConfig config;
          config.suffix = suffix;
          MetadataGapUnit placed = unit;
          placed.header.layer = layer;
          placed.header.persistence = persistence;
          std::string error;
          const auto obu = make_short_obu(placed, config, error);
          REQUIRE(!obu.empty());
          ParsedShortObu view;
          REQUIRE(parse_short_obu(obu, view, error));
          CHECK(view.is_suffix == suffix);
          CHECK(view.layer == layer);
          CHECK(view.persistence == persistence);
          CHECK(!view.cancel);
          CHECK(view.type == static_cast<std::uint32_t>(unit.header.type));
          CHECK(same_payload(view.payload, unit.payload));
          ++checked;
        }
      }
    }
  }
  std::cout << "PASS short_round_trip_matrix (" << checked << " cases)\n";
}

void test_short_cancel_round_trip_all_types() {
  std::size_t checked = 0;
  for (std::uint32_t raw = 1; raw <= 10; ++raw) {
    const auto type = static_cast<Type>(raw);
    GapObuConfig config;
    MetadataGapUnit unit;
    unit.header.type = type;
    unit.header.cancel = true;
    unit.header.layer = hdr::MetadataLayer::kGlobal;
    unit.header.persistence = hdr::MetadataPersistence::kEnhanced;
    std::string error;
    const auto obu = make_short_obu(unit, config, error);
    REQUIRE(!obu.empty());
    ParsedShortObu view;
    REQUIRE(parse_short_obu(obu, view, error));
    CHECK(view.cancel);
    CHECK(view.type == raw);
    CHECK(view.layer == hdr::MetadataLayer::kGlobal);
    CHECK(view.persistence == hdr::MetadataPersistence::kEnhanced);
    CHECK(view.payload.empty());
    ++checked;
  }
  std::cout << "PASS short_cancel_round_trip_all_types (" << checked
            << " cases)\n";
}

void test_group_round_trip_mixed() {
  GapObuConfig config;
  config.suffix = true;
  config.necessity = hdr::MetadataNecessity::kMixed;
  config.application = hdr::MetadataApplicationId::kHmd;
  config.tlayer_id = 2;
  config.mlayer_id = 0;
  config.xlayer_id = kGlobalXlayerId;

  MetadataGapUnit cll = cll_gap_unit();
  cll.header.layer = hdr::MetadataLayer::kValues;
  cll.header.persistence = hdr::MetadataPersistence::kBasic;
  cll.header.priority = 0x12;
  cll.header.layer_values.xlayer_map = 0x80000003U & 0x7FFFFFFFU;
  cll.header.layer_values.mlayer_map = {0x01, 0x02};
  cll.header.header_extension = {0x7E};

  MetadataGapUnit cancel;
  cancel.header.type = Type::kUserDataUnregistered;
  cancel.header.cancel = true;
  cancel.header.header_extension = {0x42};

  MetadataGapUnit scan = make_unit(
      Type::kScanType, serialize_scan_type_unit(ScanType{}));
  scan.header.layer = hdr::MetadataLayer::kGlobal;
  scan.header.header_extension = {0x00, 0xFF};

  const std::vector<MetadataGapUnit> units = {cll, cancel, scan};
  std::string error;
  const auto obu = make_group_obu(units, config, error);
  REQUIRE(!obu.empty());

  ParsedGroupObu view;
  REQUIRE(parse_group_obu(obu, view, error));
  CHECK(view.is_suffix);
  CHECK(view.necessity == hdr::MetadataNecessity::kMixed);
  CHECK(view.application == hdr::MetadataApplicationId::kHmd);
  CHECK(view.tlayer_id == 2U);
  CHECK(view.xlayer_id == kGlobalXlayerId);
  REQUIRE(view.units.size() == 3U);

  CHECK(view.units[0].header.type == Type::kHdrCll);
  CHECK(view.units[0].header.layer == hdr::MetadataLayer::kValues);
  CHECK(view.units[0].header.persistence ==
        hdr::MetadataPersistence::kBasic);
  CHECK(view.units[0].header.priority == 0x12U);
  CHECK(view.units[0].header.layer_values.xlayer_map == 3U);
  CHECK(view.units[0].header.layer_values.mlayer_map ==
        (std::vector<std::uint8_t>{0x01, 0x02}));
  CHECK(view.units[0].header.header_extension ==
        std::vector<std::uint8_t>{0x7E});
  CHECK(same_payload(view.units[0].payload, cll_unit()));

  CHECK(view.units[1].header.cancel);
  CHECK(view.units[1].header.type == Type::kUserDataUnregistered);
  CHECK(view.units[1].header.layer == hdr::MetadataLayer::kUnspecified);
  CHECK(view.units[1].header.persistence ==
        hdr::MetadataPersistence::kGlobal);
  CHECK(view.units[1].header.priority == 0U);
  CHECK(view.units[1].header.layer_values.mlayer_map.empty());
  CHECK(view.units[1].header.header_extension ==
        std::vector<std::uint8_t>{0x42});
  CHECK(view.units[1].payload.empty());

  CHECK(view.units[2].header.layer == hdr::MetadataLayer::kGlobal);
  CHECK(view.units[2].header.header_extension ==
        (std::vector<std::uint8_t>{0x00, 0xFF}));
  CHECK(same_payload(view.units[2].payload,
                     serialize_scan_type_unit(ScanType{})));
  std::cout << "PASS group_round_trip_mixed\n";
}

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

void test_scan_records() {
  std::vector<std::uint8_t> tu = m0::make_temporal_delimiter_obu();

  GapObuConfig config;
  std::string error;
  const auto cancel = make_short_cancel_obu(Type::kTimecode, config, error);
  REQUIRE(!cancel.empty());
  tu.insert(tu.end(), cancel.begin(), cancel.end());

  MetadataGapUnit cll = cll_gap_unit();
  cll.header.layer = hdr::MetadataLayer::kValues;
  cll.header.layer_values.mlayer_map = {0x02};
  const auto group =
      make_group_obu(std::span<const MetadataGapUnit>(&cll, 1), config, error);
  REQUIRE(!group.empty());
  tu.insert(tu.end(), group.begin(), group.end());

  const auto frame = m0::make_obu(m0::ObuHeader{m0::ObuType::kClosedLoopKey},
                                  std::span<const std::uint8_t>{});
  tu.insert(tu.end(), frame.begin(), frame.end());

  GapObuConfig suffix;
  suffix.suffix = true;
  const auto after = make_short_cancel_obu(Type::kHdrCll, suffix, error);
  REQUIRE(!after.empty());
  tu.insert(tu.end(), after.begin(), after.end());

  GapScanResult scan;
  REQUIRE(scan_gap_metadata(tu, scan, error));
  CHECK(scan.metadata_obus == 3U);
  CHECK(scan.metadata_units == 3U);
  CHECK(scan.cancelled_units == 2U);
  CHECK(scan.supported_units == 1U);
  CHECK(scan.invalid_units == 0U);
  REQUIRE(scan.records.size() == 3U);

  CHECK(!scan.records[0].in_group);
  CHECK(scan.records[0].cancel);
  CHECK(scan.records[0].before_first_frame);
  CHECK(scan.records[0].type == static_cast<std::uint32_t>(Type::kTimecode));

  CHECK(scan.records[1].in_group);
  CHECK(!scan.records[1].cancel);
  CHECK(scan.records[1].layer == hdr::MetadataLayer::kValues);
  CHECK(scan.records[1].mlayer_map == std::vector<std::uint8_t>{0x02});
  CHECK(scan.records[1].has_value);
  CHECK(scan.records[1].before_first_frame);

  CHECK(scan.records[2].is_suffix);
  CHECK(!scan.records[2].before_first_frame);
  CHECK(scan.records[2].cancel);
  std::cout << "PASS scan_records\n";
}

// ---------------------------------------------------------------------------
// Invalid input
// ---------------------------------------------------------------------------

void test_invalid_writer_inputs() {
  std::string error;
  GapObuConfig config;

  // Short form cannot carry LAYER_VALUES.
  {
    MetadataGapUnit unit = cll_gap_unit();
    unit.header.layer = hdr::MetadataLayer::kValues;
    unit.header.layer_values.mlayer_map = {0x01};
    CHECK(make_short_obu(unit, config, error).empty());
    CHECK(error.find("LAYER_VALUES") != std::string::npos ||
          error.find("muh_layer_idc") != std::string::npos);
  }
  // Short form has no header extension bytes.
  {
    MetadataGapUnit unit = cll_gap_unit();
    unit.header.header_extension = {0x01};
    CHECK(make_short_obu(unit, config, error).empty());
    CHECK(error.find("header_extension") != std::string::npos ||
          error.find("muh_header_extension") != std::string::npos);
  }
  // LAYER_VALUES without a map.
  {
    MetadataGapUnit unit = cll_gap_unit();
    unit.header.layer = hdr::MetadataLayer::kValues;
    CHECK(make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), config,
                         error)
              .empty());
    CHECK(error.find("muh_mlayer_map") != std::string::npos);
  }
  // GLOBAL map count must equal popcount(xlayer_map).
  {
    config.xlayer_id = kGlobalXlayerId;
    MetadataGapUnit unit = cll_gap_unit();
    unit.header.layer = hdr::MetadataLayer::kValues;
    unit.header.layer_values.xlayer_map = 0x3;
    unit.header.layer_values.mlayer_map = {0x01};
    CHECK(make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), config,
                         error)
              .empty());
    CHECK(error.find("GLOBAL_XLAYER_ID") != std::string::npos ||
          error.find("muh_mlayer_map") != std::string::npos);
  }
  // xlayer_map bit 31 is reserved.
  {
    MetadataGapUnit unit = cll_gap_unit();
    unit.header.layer = hdr::MetadataLayer::kValues;
    unit.header.layer_values.xlayer_map = 0x80000000U;
    CHECK(make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), config,
                         error)
              .empty());
    CHECK(error.find("bit 31") != std::string::npos);
  }
  // Non-global xlayer must not code muh_xlayer_map.
  {
    config.xlayer_id = 0;
    MetadataGapUnit unit = cll_gap_unit();
    unit.header.layer = hdr::MetadataLayer::kValues;
    unit.header.layer_values.xlayer_map = 1;
    unit.header.layer_values.mlayer_map = {0x01};
    CHECK(make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), config,
                         error)
              .empty());
    CHECK(error.find("GLOBAL_XLAYER_ID") != std::string::npos);
  }
  // A map bit below obu_mlayer_id is forbidden.
  {
    GapObuConfig mlayer_config;
    mlayer_config.mlayer_id = 2;
    MetadataGapUnit unit = cll_gap_unit();
    unit.header.layer = hdr::MetadataLayer::kValues;
    unit.header.layer_values.mlayer_map = {0x02};
    CHECK(make_group_obu(std::span<const MetadataGapUnit>(&unit, 1),
                         mlayer_config, error)
              .empty());
    CHECK(error.find("below obu_mlayer_id") != std::string::npos);
  }
  // Cancel unit cannot carry a payload.
  {
    MetadataGapUnit unit;
    unit.header.type = Type::kHdrCll;
    unit.header.cancel = true;
    unit.payload = cll_unit();
    CHECK(make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), config,
                         error)
              .empty());
    CHECK(error.find("cancel") != std::string::npos);
  }
  // Non-canonical / empty payloads.
  {
    MetadataGapUnit unit = cll_gap_unit();
    unit.payload.push_back(0x00);
    CHECK(make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), config,
                         error)
              .empty());
    MetadataGapUnit empty = make_unit(Type::kHdrCll, {});
    CHECK(make_group_obu(std::span<const MetadataGapUnit>(&empty, 1), config,
                         error)
              .empty());
  }
  // Reserved enum values.
  {
    MetadataGapUnit unit = cll_gap_unit();
    unit.header.persistence =
        static_cast<hdr::MetadataPersistence>(4);
    CHECK(make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), config,
                         error)
              .empty());
    unit = cll_gap_unit();
    unit.header.layer = static_cast<hdr::MetadataLayer>(4);
    CHECK(make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), config,
                         error)
              .empty());
    unit = cll_gap_unit();
    unit.header.type = Type::kReserved;
    CHECK(make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), config,
                         error)
              .empty());
  }
  // Reserved application id.
  {
    GapObuConfig bad;
    bad.application = static_cast<hdr::MetadataApplicationId>(6);
    MetadataGapUnit unit = cll_gap_unit();
    CHECK(make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), bad,
                         error)
              .empty());
    CHECK(error.find("reserved") != std::string::npos);
  }
  // muh_header_size f(7) overflow.
  {
    MetadataGapUnit unit = cll_gap_unit();
    unit.header.header_extension.assign(200U, 0x00);
    CHECK(make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), config,
                         error)
              .empty());
    CHECK(error.find("muh_header_size") != std::string::npos);
  }
  // Empty group.
  {
    CHECK(make_group_obu(std::span<const MetadataGapUnit>{}, config, error)
              .empty());
  }
  // OBU header fields must fit their widths.
  {
    GapObuConfig bad;
    bad.tlayer_id = 4;
    MetadataGapUnit unit = cll_gap_unit();
    CHECK(make_short_obu(unit, bad, error).empty());
    bad = GapObuConfig{};
    bad.xlayer_id = 32;
    CHECK(make_short_obu(unit, bad, error).empty());
  }
  std::cout << "PASS invalid_writer_inputs\n";
}

void test_invalid_reader_inputs() {
  std::string error;
  GapObuConfig config;

  // Wrong OBU types.
  {
    MetadataGapUnit unit = cll_gap_unit();
    const auto group =
        make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), config,
                       error);
    REQUIRE(!group.empty());
    ParsedShortObu short_view;
    CHECK(!parse_short_obu(group, short_view, error));

    const auto short_obu = make_short_obu(unit, config, error);
    REQUIRE(!short_obu.empty());
    ParsedGroupObu group_view;
    CHECK(!parse_group_obu(short_obu, group_view, error));
  }

  // Truncated OBU and trailing bytes.
  {
    MetadataGapUnit unit = cll_gap_unit();
    const auto obu = make_short_obu(unit, config, error);
    REQUIRE(obu.size() > 2U);
    ParsedShortObu view;
    CHECK(!parse_short_obu(
        std::span<const std::uint8_t>(obu.data(), obu.size() - 1U), view,
        error));
    std::vector<std::uint8_t> extended = obu;
    extended.push_back(0x00);
    CHECK(!parse_short_obu(extended, view, error));
  }

  // Corrupt trailing_bits.
  {
    MetadataGapUnit unit = cll_gap_unit();
    auto obu = make_short_obu(unit, config, error);
    REQUIRE(!obu.empty());
    obu.back() = 0x00;
    ParsedShortObu view;
    CHECK(!parse_short_obu(obu, view, error));
    CHECK(error.find("trailing_bits") != std::string::npos);
  }

  // Hand-built short OBU with muh_layer_idc == LAYER_VALUES (reserved for the
  // short form by §6.16.2).
  {
    const std::vector<std::uint8_t> payload = {0x68, 0x04, 0x80};
    const auto obu = wrap_raw(kObuTypeMetadataShort, GapObuConfig{}, payload);
    ParsedShortObu view;
    CHECK(!parse_short_obu(obu, view, error));
    CHECK(error.find("muh_layer_idc") != std::string::npos);
  }

  // Hand-built group with muh_reserved_zero_2bits set.
  {
    MetadataGapUnit unit = cll_gap_unit();
    unit.header.priority = 0x5A;
    auto obu = make_group_obu(std::span<const MetadataGapUnit>(&unit, 1),
                              config, error);
    REQUIRE(obu.size() > 10U);
    // Locate the priority byte: leb size, header byte, options, count,
    // type, unit header byte, payload size, layer byte, priority byte.
    std::size_t pos = 0;
    while ((obu[pos++] & 0x80U) != 0U) {
    }
    pos += 1U;  // obu_header (no extension here)
    const std::size_t payload_start = pos;
    // options (1), count (1), type (1), header size (1), payload size (1),
    // layer (1) then priority.
    const std::size_t priority_index = payload_start + 6U;
    REQUIRE(priority_index < obu.size());
    auto bad = obu;
    bad[priority_index] |= 0x01U;
    ParsedGroupObu view;
    CHECK(!parse_group_obu(bad, view, error));
    CHECK(error.find("muh_reserved_zero_2bits") != std::string::npos);
  }

  // Hand-built group with muh_header_size smaller than its fixed fields.
  {
    MetadataGapUnit unit = cll_gap_unit();
    auto obu = make_group_obu(std::span<const MetadataGapUnit>(&unit, 1),
                              config, error);
    REQUIRE(!obu.empty());
    std::size_t pos = 0;
    while ((obu[pos++] & 0x80U) != 0U) {
    }
    ++pos;  // OBU header
    const std::size_t payload_start = pos;
    auto bad = obu;
    bad[payload_start + 3U] = 0x02U;  // header_size = 1
    ParsedGroupObu view;
    CHECK(!parse_group_obu(bad, view, error));
    CHECK(error.find("muh_header_size") != std::string::npos);
  }

  // Hand-built GLOBAL LAYER_VALUES group with bit 31 of muh_xlayer_map set.
  {
    GapObuConfig global;
    global.xlayer_id = kGlobalXlayerId;
    MetadataGapUnit unit = cll_gap_unit();
    unit.header.layer = hdr::MetadataLayer::kValues;
    unit.header.layer_values.xlayer_map = 1;
    unit.header.layer_values.mlayer_map = {0x01};
    auto obu = make_group_obu(std::span<const MetadataGapUnit>(&unit, 1),
                              global, error);
    REQUIRE(!obu.empty());
    std::size_t pos = 0;
    while ((obu[pos++] & 0x80U) != 0U) {
    }
    ++pos;  // OBU header byte
    ++pos;  // OBU header extension byte
    const std::size_t payload_start = pos;
    // options, count, type, header size, payload size, layer, priority,
    // then the 4-byte xlayer map.
    const std::size_t map_index = payload_start + 7U;
    REQUIRE(map_index + 3U < obu.size());
    auto bad = obu;
    bad[map_index] |= 0x80U;
    ParsedGroupObu view;
    CHECK(!parse_group_obu(bad, view, error));
    CHECK(error.find("bit 31") != std::string::npos);
  }

  // Hand-built group whose muh_header_size overruns the payload.
  {
    const std::vector<std::uint8_t> payload = {
        0x23, 0x00,              // options, count
        0x04, 0x0E,              // type TIMEcode, header_size 7
        0x00, 0x00, 0x00,        // payload size, layer, priority
        0x80,                    // trailing_bits (too few header bytes)
    };
    const auto obu = wrap_raw(kObuTypeMetadataGroup, GapObuConfig{}, payload);
    ParsedGroupObu view;
    CHECK(!parse_group_obu(obu, view, error));
  }

  // Hand-built group with a truncated map.
  {
    const std::vector<std::uint8_t> payload = {
        0x23, 0x00,        // options, count
        0x04, 0x06,        // type TIMEcode, header_size 3
        0x00, 0x60, 0x00,  // payload size 0, LAYER_VALUES, priority 0
        0x80,              // trailing_bits, map missing
    };
    const auto obu = wrap_raw(kObuTypeMetadataGroup, GapObuConfig{}, payload);
    ParsedGroupObu view;
    CHECK(!parse_group_obu(obu, view, error));
    CHECK(error.find("muh_mlayer_map") != std::string::npos);
  }

  std::cout << "PASS invalid_reader_inputs\n";
}

}  // namespace

int main() {
  test_short_cancel_golden();
  test_short_cancel_suffix_golden();
  test_group_layer_values_local_golden();
  test_group_layer_values_global_golden();
  test_group_header_extension_golden();
  test_group_cancel_golden();
  test_short_obu_header_extension_golden();
  test_plain_group_matches_hdr_writer();
  test_short_round_trip_matrix();
  test_short_cancel_round_trip_all_types();
  test_group_round_trip_mixed();
  test_scan_records();
  test_invalid_writer_inputs();
  test_invalid_reader_inputs();

  if (g_failures != 0) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "metadata_gaps_test: all checks passed\n";
  return 0;
}
