// Unit tests for the AV2 v1.0.0 HDR metadata OBU module.
//
// Build with src/encoder/hdr/build.sh (no CMake target yet: this agent owns
// only src/encoder/hdr/). The expectations below are pinned bit goldens from
// the §5.17.2/§5.17.3/§5.17.5/§5.17.6 syntax, cross-checked against dav2d.

#include "encoder/hdr/hdr_metadata.h"

#include "bitstream/m0_headers.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace av2enc;
using namespace av2enc::hdr;

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
  out.reserve(bytes.size() * 2U);
  for (const std::uint8_t byte : bytes) {
    out.push_back(kDigits[byte >> 4U]);
    out.push_back(kDigits[byte & 0x0FU]);
  }
  return out;
}

// BT.2020 primaries + D65, 1000 cd/m^2 max, 0.005 cd/m^2 min.
ContentLightLevel sample_cll() { return ContentLightLevel{1000, 400}; }

MasteringDisplayColorVolume sample_mdcv() {
  MasteringDisplayColorVolume mdcv;
  // §6.16.6 suggests i = 0 green, 1 blue, 2 red.
  mdcv.primary_chromaticity_x = {8500, 6550, 35400};    // 0.1700, 0.1310, 0.7080
  mdcv.primary_chromaticity_y = {39850, 2300, 14600};   // 0.7970, 0.0460, 0.2920
  mdcv.white_point_chromaticity_x = 15635;              // D65 0.3127
  mdcv.white_point_chromaticity_y = 16450;              // D65 0.3290
  mdcv.luminance_max = 10000000;                        // 1000.0 cd/m^2
  mdcv.luminance_min = 50;                              // 0.0050 cd/m^2
  return mdcv;
}

HdrMetadata sample_hdr() {
  HdrMetadata hdr;
  hdr.has_cll = true;
  hdr.cll = sample_cll();
  hdr.has_mdcv = true;
  hdr.mdcv = sample_mdcv();
  return hdr;
}

std::vector<std::uint8_t> make_obu(std::uint8_t type,
                                   std::span<const std::uint8_t> payload) {
  m0::ObuHeader header;
  header.type = static_cast<m0::ObuType>(type);
  header.tlayer_id = 0;
  header.extension_flag = false;
  return m0::make_obu(header, payload);
}

std::vector<std::uint8_t> make_temporal_delimiter() { return {0x01, 0x08}; }

std::vector<std::uint8_t> make_sequence_header_payload(std::uint8_t seed) {
  std::vector<std::uint8_t> payload(12, seed);
  return payload;
}

// TD + SEQ + CLK, i.e. the M1 still-picture temporal unit shape with a dummy
// tile payload (the HDR module never looks inside the frame OBU).
std::vector<std::uint8_t> make_test_temporal_unit(std::uint8_t seq_seed = 0x5A) {
  std::vector<std::uint8_t> out = make_temporal_delimiter();
  const std::vector<std::uint8_t> seq_payload =
      make_sequence_header_payload(seq_seed);
  const std::vector<std::uint8_t> seq = make_obu(1, seq_payload);
  out.insert(out.end(), seq.begin(), seq.end());
  const std::vector<std::uint8_t> frame_payload = {0x80};
  const std::vector<std::uint8_t> frame = make_obu(4, frame_payload);
  out.insert(out.end(), frame.begin(), frame.end());
  return out;
}

int count_errors(const ValidationResult& result) {
  int count = 0;
  for (const auto& issue : result.issues()) {
    if (issue.severity == Severity::kError) ++count;
  }
  return count;
}

int count_warnings(const ValidationResult& result) {
  int count = 0;
  for (const auto& issue : result.issues()) {
    if (issue.severity == Severity::kWarning) ++count;
  }
  return count;
}

// ---------------------------------------------------------------------------
// Pinned bit goldens
// ---------------------------------------------------------------------------

void test_cll_golden() {
  std::string error;
  MetadataObuConfig config;
  const std::vector<std::uint8_t> obu =
      make_cll_short_obu(sample_cll(), config, error);
  CHECK(error.empty());
  // 08                                  leb128 OBU size: 1 header + 7 payload
  //   20                                obu_header: ext=0, type=8, tlayer=0
  //     20                              is_suffix=0, layer=2, cancel=0, pers=0
  //       01                            leb128 metadata_type = HDR_CLL
  //         03e8                        max_cll  = 1000
  //             0190                    max_fall = 400
  //                 80                  trailing_bits
  CHECK_HEX(obu, "0820200103e8019080");
}

void test_mdcv_golden() {
  std::string error;
  MetadataObuConfig config;
  const std::vector<std::uint8_t> obu =
      make_mdcv_short_obu(sample_mdcv(), config, error);
  CHECK(error.empty());
  // 1c                                  leb128 OBU size: 1 + 27
  //   20                                obu_header type 8
  //     20                              §5.17.2 header byte
  //       02                            metadata_type = HDR_MDCV
  //         2134 9baa                   primary 0 (green) x/y
  //                   1996 08fc         primary 1 (blue)  x/y
  //                             8a48 3908  primary 2 (red) x/y
  //                                       3d13 4042  white point x/y
  //                                                 00989680  luminance_max
  //                                                           00000032  min
  //                                                                   80
  CHECK_HEX(obu,
            "1c202002"
            "21349baa199608fc8a483908"
            "3d134042"
            "00989680"
            "00000032"
            "80");
}

void test_group_golden() {
  std::string error;
  MetadataObuConfig config;
  const HdrMetadata hdr = sample_hdr();
  const std::vector<std::uint8_t> obu =
      make_hdr_metadata_obus(hdr, HdrObuPacking::kMetadataGroup, config, error);
  CHECK(error.empty());
  // 2a                                  leb128 OBU size: 1 + 41
  //   24                                obu_header type 9
  //     40                              suffix=0, necessity=2, app=0
  //       01                            metadata_unit_cnt_minus_1 = 1
  //         01 06 04 4000 03e80190      CLL unit (type 1, hdr_size 3, size 4)
  //                                   02 06 18 4000 <24-byte MDCV payload>
  //                                                              80
  CHECK_HEX(obu,
            "2a244001"
            "010604400003e80190"
            "0206184000"
            "21349baa199608fc8a483908"
            "3d134042"
            "00989680"
            "00000032"
            "80");
}

void test_suffix_and_cancel_goldens() {
  std::string error;
  MetadataObuConfig suffix;
  suffix.suffix = true;
  CHECK_HEX(make_cll_short_obu(sample_cll(), suffix, error),
            "0820a00103e8019080");

  MetadataObuConfig cancel;
  const std::vector<std::uint8_t> obu =
      make_cancel_short_obu(MetadataType::kHdrMdcv, cancel, error);
  CHECK(error.empty());
  // 04 20 28 02 80: size 4, type 8, suffix=0/layer=2/cancel=1/pers=0,
  // metadata_type = 2, trailing bits.
  CHECK_HEX(obu, "0420280280");

  // Group suffix flips bit 7 of the group header byte: 0xC0.
  MetadataObuConfig group_suffix;
  group_suffix.suffix = true;
  const std::vector<std::uint8_t> group = make_hdr_metadata_obus(
      sample_hdr(), HdrObuPacking::kMetadataGroup, group_suffix, error);
  CHECK(error.empty());
  CHECK(group.size() > 2U);
  CHECK(group[2] == 0xC0U);
}

void test_legacy_layout_goldens() {
  std::string error;
  const std::vector<std::uint8_t> cll = make_legacy_av1_hdr_metadata_obus(
      HdrMetadata{true, sample_cll(), false, {}}, error);
  CHECK(error.empty());
  CHECK_HEX(cll, "07200103e8019080");

  HdrMetadata only_mdcv;
  only_mdcv.has_mdcv = true;
  only_mdcv.mdcv = sample_mdcv();
  const std::vector<std::uint8_t> mdcv =
      make_legacy_av1_hdr_metadata_obus(only_mdcv, error);
  CHECK(error.empty());
  CHECK_HEX(mdcv,
            "1b200221349baa199608fc8a4839083d134042009896800000003280");

  // A spec reader must not mistake this for a §5.17.2 OBU. It parses the first
  // payload byte 0x01 as the short header: is_suffix=0, layer=0, cancel=0,
  // persistence=1, then metadata_type 0x03 (ITU-T T.35). This documents the
  // hazard; the scanner only ever sees spec-layout streams.
  ObuView view;
  std::size_t offset = 0;
  CHECK(parse_obu(cll, offset, view, error));
  MetadataShortView short_view;
  CHECK(parse_metadata_short(view.payload, short_view, error));
  CHECK(short_view.type == 3U);
  CHECK(!short_view.is_suffix);
  CHECK(!short_view.cancel);
}

// ---------------------------------------------------------------------------
// Round-trips
// ---------------------------------------------------------------------------

void test_round_trip() {
  std::string error;
  MetadataObuConfig config;
  const HdrMetadata hdr = sample_hdr();
  const std::vector<std::uint8_t> obus =
      make_hdr_metadata_obus(hdr, HdrObuPacking::kShortObus, config, error);
  CHECK(error.empty());

  std::vector<std::uint8_t> tu = make_test_temporal_unit();
  std::vector<std::uint8_t> with_metadata;
  CHECK(insert_metadata_obus(tu, obus, MetadataPlacement::kBeforeFirstFrame,
                             with_metadata, error));
  CHECK(error.empty());

  HdrScanResult scan;
  CHECK(scan_hdr_metadata(with_metadata, scan, error));
  CHECK(error.empty());
  CHECK(scan.metadata_obus == 2U);
  CHECK(scan.metadata_units == 2U);
  CHECK(scan.records.size() == 2U);
  CHECK(scan.unknown_units == 0U);
  CHECK(scan.invalid_units == 0U);
  if (scan.records.size() == 2U) {
    CHECK(scan.records[0].type == MetadataType::kHdrCll);
    CHECK(scan.records[0].cll == sample_cll());
    CHECK(scan.records[1].type == MetadataType::kHdrMdcv);
    CHECK(scan.records[1].mdcv == sample_mdcv());
    CHECK(scan.records[0].before_first_frame);
    CHECK(!scan.records[0].in_group);
  }

  // Group packing round-trips the same values through one OBU.
  const std::vector<std::uint8_t> group = make_hdr_metadata_obus(
      hdr, HdrObuPacking::kMetadataGroup, config, error);
  CHECK(error.empty());
  CHECK(insert_metadata_obus(tu, group, MetadataPlacement::kBeforeFirstFrame,
                             with_metadata, error));
  CHECK(scan_hdr_metadata(with_metadata, scan, error));
  CHECK(scan.metadata_obus == 1U);
  CHECK(scan.records.size() == 2U);
  CHECK(scan.records.size() == 2U && scan.records[0].cll == sample_cll());
  CHECK(scan.records.size() == 2U && scan.records[1].mdcv == sample_mdcv());
  CHECK(scan.records.size() == 2U && scan.records[0].in_group);
}

void test_extreme_values_round_trip() {
  std::string error;
  MetadataObuConfig config;

  ContentLightLevel cll;
  cll.max_cll = 0;
  cll.max_fall = 0xFFFF;
  const std::vector<std::uint8_t> cll_obu =
      make_cll_short_obu(cll, config, error);
  CHECK(error.empty());
  CHECK(cll_obu.size() == 9U);

  MasteringDisplayColorVolume mdcv;
  mdcv.primary_chromaticity_x = {5, 37000, 5};
  mdcv.primary_chromaticity_y = {5, 42000, 42000};
  mdcv.white_point_chromaticity_x = 37000;
  mdcv.white_point_chromaticity_y = 42000;
  mdcv.luminance_max = 100000000;  // upper bound
  mdcv.luminance_min = 50000;      // legal because max != 50000
  const std::vector<std::uint8_t> mdcv_obu =
      make_mdcv_short_obu(mdcv, config, error);
  CHECK(error.empty());

  const std::vector<std::uint8_t> obus = [&] {
    std::vector<std::uint8_t> all = cll_obu;
    all.insert(all.end(), mdcv_obu.begin(), mdcv_obu.end());
    return all;
  }();
  std::vector<std::uint8_t> tu = make_test_temporal_unit();
  std::vector<std::uint8_t> with_metadata;
  CHECK(insert_metadata_obus(tu, obus, MetadataPlacement::kBeforeFirstFrame,
                             with_metadata, error));

  HdrScanResult scan;
  CHECK(scan_hdr_metadata(with_metadata, scan, error));
  CHECK(scan.records.size() == 2U);
  if (scan.records.size() == 2U) {
    CHECK(scan.records[0].cll == cll);
    CHECK(scan.records[1].mdcv == mdcv);
  }

  // Zero/unknown MDCV: legal "unknown/unspecified" encoding, warned by default.
  MasteringDisplayColorVolume unknown;
  const ValidationResult result = validate_mdcv(unknown, false);
  CHECK(result.ok());
  CHECK(count_warnings(result) > 0);
  CHECK(make_mdcv_short_obu(unknown, config, error).size() == 29U);
  CHECK(error.empty());
}

void test_extra_unit_bytes() {
  // §5.17.1 permits metadata_unit_remaining_bit padding, so a CLL unit with
  // extra bytes must still decode to the same values.
  ContentLightLevel cll;
  std::string error;
  std::vector<std::uint8_t> unit = serialize_cll_unit(ContentLightLevel{321, 123});
  unit.push_back(0xDE);
  unit.push_back(0xAD);
  CHECK(decode_cll(unit, cll, error));
  CHECK(cll.max_cll == 321U);
  CHECK(cll.max_fall == 123U);

  std::vector<std::uint8_t> mdcv_unit = serialize_mdcv_unit(sample_mdcv());
  mdcv_unit.push_back(0xFF);
  MasteringDisplayColorVolume mdcv;
  CHECK(decode_mdcv(mdcv_unit, mdcv, error));
  CHECK(mdcv == sample_mdcv());
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

void test_validation() {
  // Canonical sample is fully in range.
  {
    const ValidationResult result = validate_mdcv(sample_mdcv(), false);
    CHECK(result.clean());
  }
  // Boundary values are valid.
  {
    MasteringDisplayColorVolume mdcv = sample_mdcv();
    mdcv.primary_chromaticity_x = {5, 37000, 5};
    mdcv.primary_chromaticity_y = {5, 42000, 42000};
    mdcv.white_point_chromaticity_x = 5;
    mdcv.white_point_chromaticity_y = 42000;
    mdcv.luminance_max = 50000;
    mdcv.luminance_min = 1;
    CHECK(validate_mdcv(mdcv, true).clean());
  }
  // Out-of-range chromaticity: warning by default, error in strict mode.
  {
    MasteringDisplayColorVolume mdcv = sample_mdcv();
    mdcv.primary_chromaticity_x[0] = 4;
    const ValidationResult relaxed = validate_mdcv(mdcv, false);
    CHECK(relaxed.ok());
    CHECK(count_warnings(relaxed) == 1);
    const ValidationResult strict = validate_mdcv(mdcv, true);
    CHECK(!strict.ok());
    CHECK(count_errors(strict) == 1);
  }
  // Y range differs from X: 37001 is fine for y, 42001 is not.
  {
    MasteringDisplayColorVolume mdcv = sample_mdcv();
    mdcv.primary_chromaticity_y[1] = 37001;
    CHECK(validate_mdcv(mdcv, true).clean());
    mdcv.primary_chromaticity_y[1] = 42001;
    CHECK(!validate_mdcv(mdcv, true).ok());
  }
  // Luminance bounds.
  {
    MasteringDisplayColorVolume mdcv = sample_mdcv();
    mdcv.luminance_max = 49999;
    CHECK(!validate_mdcv(mdcv, true).ok());
    CHECK(validate_mdcv(mdcv, false).ok());
    mdcv = sample_mdcv();
    mdcv.luminance_max = 100000001;
    CHECK(!validate_mdcv(mdcv, true).ok());
    mdcv = sample_mdcv();
    mdcv.luminance_min = 0;
    CHECK(!validate_mdcv(mdcv, true).ok());
    mdcv = sample_mdcv();
    mdcv.luminance_min = 50001;
    CHECK(!validate_mdcv(mdcv, true).ok());
  }
  // §6.16.6 normative constraint, always an error.
  {
    MasteringDisplayColorVolume mdcv = sample_mdcv();
    mdcv.luminance_max = 50000;
    mdcv.luminance_min = 50000;
    CHECK(!validate_mdcv(mdcv, false).ok());
    CHECK(!validate_mdcv(mdcv, true).ok());
  }
  // CLL: every u16 value is representable; zero means "not signalled".
  {
    CHECK(validate_cll(ContentLightLevel{0, 0}, true).clean());
    CHECK(validate_cll(ContentLightLevel{65535, 65535}, true).clean());
  }
}

void test_writer_validation() {
  std::string error;
  MetadataObuConfig config;

  MasteringDisplayColorVolume bad = sample_mdcv();
  bad.luminance_max = 50000;
  bad.luminance_min = 50000;
  CHECK(make_mdcv_short_obu(bad, config, error).empty());
  CHECK(!error.empty());

  // Out-of-range values are spec-legal "unknown" and are emitted by default.
  MasteringDisplayColorVolume unknown = sample_mdcv();
  unknown.primary_chromaticity_x[0] = 0;
  const std::vector<std::uint8_t> obu =
      make_mdcv_short_obu(unknown, config, error);
  CHECK(error.empty());
  CHECK(obu.size() == 29U);

  // strict_ranges refuses to encode a placeholder.
  MetadataObuConfig strict;
  strict.strict_ranges = true;
  CHECK(make_mdcv_short_obu(unknown, strict, error).empty());
  CHECK(!error.empty());

  // The convenience writer validates the MDCV unit before packing.
  HdrMetadata hdr = sample_hdr();
  hdr.mdcv.luminance_min = 0;
  CHECK(make_hdr_metadata_obus(hdr, HdrObuPacking::kShortObus, strict, error)
            .empty());
  CHECK(!error.empty());
}

// ---------------------------------------------------------------------------
// Multiple units / unknown units
// ---------------------------------------------------------------------------

void test_multiple_obus() {
  std::string error;
  MetadataObuConfig config;
  std::vector<std::uint8_t> tu = make_test_temporal_unit();

  // Two short OBUs in one temporal unit.
  const std::vector<std::uint8_t> short_obus = make_hdr_metadata_obus(
      sample_hdr(), HdrObuPacking::kShortObus, config, error);
  CHECK(error.empty());

  // Group carrying CLL + an unknown user-data unit; both are structurally
  // parsed, the unknown one is counted and skipped.
  std::vector<MetadataGroupEntry> entries;
  entries.push_back(MetadataGroupEntry{MetadataType::kHdrCll,
                                       serialize_cll_unit(sample_cll()),
                                       MetadataLayer::kCurrent,
                                       MetadataPersistence::kGlobal, 7, false});
  entries.push_back(MetadataGroupEntry{MetadataType::kUserDataUnregistered,
                                       {0x01, 0x02, 0x03, 0x04},
                                       MetadataLayer::kCurrent,
                                       MetadataPersistence::kGlobal, 200, false});
  const std::vector<std::uint8_t> mixed_group =
      make_metadata_group_obu(entries, config, error);
  CHECK(error.empty());

  // Priority survives the round-trip.
  ObuView obu;
  std::size_t offset = 0;
  CHECK(parse_obu(mixed_group, offset, obu, error));
  MetadataGroupView group;
  CHECK(parse_metadata_group(obu.payload, obu.xlayer_id, group, error));
  CHECK(group.units.size() == 2U);
  if (group.units.size() == 2U) {
    CHECK(group.units[0].priority == 7U);
    CHECK(group.units[1].priority == 200U);
    CHECK(group.units[1].type == 10U);
  }

  std::vector<std::uint8_t> combined = short_obus;
  combined.insert(combined.end(), mixed_group.begin(), mixed_group.end());
  std::vector<std::uint8_t> with_metadata;
  CHECK(insert_metadata_obus(tu, combined, MetadataPlacement::kBeforeFirstFrame,
                             with_metadata, error));
  HdrScanResult scan;
  CHECK(scan_hdr_metadata(with_metadata, scan, error));
  CHECK(scan.metadata_obus == 3U);   // 2 short + 1 group
  CHECK(scan.metadata_units == 4U);  // CLL, MDCV, CLL, user-data
  CHECK(scan.records.size() == 3U);  // CLL, MDCV, CLL
  CHECK(scan.unknown_units == 1U);
  CHECK(scan.invalid_units == 0U);
}

void test_cancel_units_are_not_records() {
  std::string error;
  MetadataObuConfig config;
  const std::vector<std::uint8_t> cancel =
      make_cancel_short_obu(MetadataType::kHdrCll, config, error);
  CHECK(error.empty());

  std::vector<std::uint8_t> tu = make_test_temporal_unit();
  std::vector<std::uint8_t> with_metadata;
  CHECK(insert_metadata_obus(tu, cancel, MetadataPlacement::kBeforeFirstFrame,
                             with_metadata, error));
  HdrScanResult scan;
  CHECK(scan_hdr_metadata(with_metadata, scan, error));
  CHECK(scan.metadata_obus == 1U);
  CHECK(scan.metadata_units == 1U);
  CHECK(scan.records.empty());
  CHECK(scan.unknown_units == 0U);
}

// ---------------------------------------------------------------------------
// Ordering relative to sequence/frame headers
// ---------------------------------------------------------------------------

void test_ordering() {
  std::string error;
  MetadataObuConfig prefix;
  MetadataObuConfig suffix;
  suffix.suffix = true;

  const std::vector<std::uint8_t> prefix_obus = make_hdr_metadata_obus(
      sample_hdr(), HdrObuPacking::kShortObus, prefix, error);
  const std::vector<std::uint8_t> suffix_obus = make_hdr_metadata_obus(
      sample_hdr(), HdrObuPacking::kShortObus, suffix, error);
  CHECK(error.empty());

  const std::vector<std::uint8_t> tu = make_test_temporal_unit();

  // Prefix: after the sequence header, before the frame.
  std::vector<std::uint8_t> inserted;
  std::size_t at = 0;
  CHECK(insert_metadata_obus(tu, prefix_obus,
                             MetadataPlacement::kBeforeFirstFrame, inserted,
                             error, &at));
  CHECK(error.empty());
  std::vector<ObuView> obus;
  CHECK(parse_obu_sequence(inserted, obus, error));
  CHECK(obus.size() == 5U);
  if (obus.size() == 5U) {
    CHECK(obus[0].type == 2U);  // TD
    CHECK(obus[1].type == 1U);  // SEQ
    CHECK(obus[2].type == 8U);  // CLL
    CHECK(obus[3].type == 8U);  // MDCV
    CHECK(obus[4].type == 4U);  // CLK
    CHECK(obus[2].offset == obus[1].offset + obus[1].size);
    CHECK(at == obus[2].offset);
  }
  HdrScanResult scan;
  CHECK(scan_hdr_metadata(inserted, scan, error));
  CHECK(scan.records.size() == 2U &&
        scan.records[0].before_first_frame &&
        !scan.records[0].is_suffix);

  // Suffix: after the frame.
  CHECK(insert_metadata_obus(tu, suffix_obus,
                             MetadataPlacement::kEndOfTemporalUnit, inserted,
                             error));
  CHECK(parse_obu_sequence(inserted, obus, error));
  CHECK(obus.size() == 5U);
  if (obus.size() == 5U) {
    CHECK(obus[2].type == 4U);
    CHECK(obus[3].type == 8U);
    CHECK(obus[4].type == 8U);
  }
  CHECK(scan_hdr_metadata(inserted, scan, error));
  CHECK(scan.records.size() == 2U && scan.records[0].is_suffix);
  CHECK(scan.records.size() == 2U && !scan.records[0].before_first_frame);

  // Mismatched suffix flag is rejected in both directions.
  CHECK(!insert_metadata_obus(tu, prefix_obus,
                              MetadataPlacement::kEndOfTemporalUnit, inserted,
                              error));
  CHECK(!error.empty());
  CHECK(!insert_metadata_obus(tu, suffix_obus,
                              MetadataPlacement::kBeforeFirstFrame, inserted,
                              error));

  // Prefix needs a sequence header before the frame.
  std::vector<std::uint8_t> no_seq = make_temporal_delimiter();
  const std::vector<std::uint8_t> frame = make_obu(4, std::vector<std::uint8_t>{0x80});
  no_seq.insert(no_seq.end(), frame.begin(), frame.end());
  CHECK(!insert_metadata_obus(no_seq, prefix_obus,
                              MetadataPlacement::kBeforeFirstFrame, inserted,
                              error));

  // Suffix needs a frame OBU as well.
  const std::vector<std::uint8_t> td_only = make_temporal_delimiter();
  CHECK(!insert_metadata_obus(td_only, suffix_obus,
                              MetadataPlacement::kEndOfTemporalUnit, inserted,
                              error));

  // Only metadata OBUs may be inserted.
  const std::vector<std::uint8_t> not_metadata = make_temporal_delimiter();
  CHECK(!insert_metadata_obus(tu, not_metadata,
                              MetadataPlacement::kBeforeFirstFrame, inserted,
                              error));
}

void test_bit_depth_independence() {
  // The HDR metadata syntax has no BitDepth-dependent field (§5.17.5/§5.17.6
  // are pure integers), so the same units must produce identical OBUs for
  // 8-bit and 10-bit sequence headers. The M1 encoder is 8-bit only today;
  // this asserts the module itself is format-agnostic.
  std::string error;
  MetadataObuConfig config;
  const std::vector<std::uint8_t> obus = make_hdr_metadata_obus(
      sample_hdr(), HdrObuPacking::kShortObus, config, error);
  CHECK(error.empty());

  const std::vector<std::uint8_t> tu_8bit = make_test_temporal_unit(0x5A);
  const std::vector<std::uint8_t> tu_10bit = make_test_temporal_unit(0x5B);

  std::vector<std::uint8_t> out_8bit;
  std::vector<std::uint8_t> out_10bit;
  CHECK(insert_metadata_obus(tu_8bit, obus,
                             MetadataPlacement::kBeforeFirstFrame, out_8bit,
                             error));
  CHECK(insert_metadata_obus(tu_10bit, obus,
                             MetadataPlacement::kBeforeFirstFrame, out_10bit,
                             error));

  HdrScanResult scan_8;
  HdrScanResult scan_10;
  CHECK(scan_hdr_metadata(out_8bit, scan_8, error));
  CHECK(scan_hdr_metadata(out_10bit, scan_10, error));
  CHECK(scan_8.records.size() == 2U && scan_10.records.size() == 2U);
  if (scan_8.records.size() == 2U && scan_10.records.size() == 2U) {
    CHECK(scan_8.records[0].cll == scan_10.records[0].cll);
    CHECK(scan_8.records[1].mdcv == scan_10.records[1].mdcv);

    // The metadata OBU bytes are byte-identical; only the sequence header
    // differs. Prefix metadata is inserted immediately before the frame.
    CHECK(out_8bit.size() == out_10bit.size());
    const std::size_t metadata_start =
        static_cast<std::size_t>(scan_8.records[0].obu_offset);
    const std::size_t metadata_end = metadata_start + 9U + 29U;
    CHECK(metadata_end <= out_8bit.size() && metadata_end <= out_10bit.size());
    if (metadata_end <= out_8bit.size() && metadata_end <= out_10bit.size()) {
      CHECK(std::equal(
          out_8bit.begin() + static_cast<std::ptrdiff_t>(metadata_start),
          out_8bit.begin() + static_cast<std::ptrdiff_t>(metadata_end),
          out_10bit.begin() + static_cast<std::ptrdiff_t>(metadata_start)));
    }
  }

  // Canonical unit sizes are fixed and independent of any bit depth.
  CHECK(serialize_cll_unit(sample_cll()).size() == kCllUnitBytes);
  CHECK(serialize_mdcv_unit(sample_mdcv()).size() == kMdcvUnitBytes);
}

// ---------------------------------------------------------------------------
// Invalid input
// ---------------------------------------------------------------------------

void test_invalid_inputs() {
  std::string error;
  ObuView obu;
  std::size_t offset = 0;

  // Truncated leb128 / truncated OBU / zero size.
  {
    const std::vector<std::uint8_t> data = {0x80};
    offset = 0;
    CHECK(!parse_obu(data, offset, obu, error));
  }
  {
    const std::vector<std::uint8_t> data = {0x05, 0x20};
    offset = 0;
    CHECK(!parse_obu(data, offset, obu, error));
  }
  {
    const std::vector<std::uint8_t> data = {0x00};
    offset = 0;
    CHECK(!parse_obu(data, offset, obu, error));
  }
  {
    // OBU size 0x0a but only 2 bytes present.
    const std::vector<std::uint8_t> data = {0x0a, 0x20};
    offset = 0;
    CHECK(!parse_obu(data, offset, obu, error));
  }
  {
    // leb128 longer than 5 bytes.
    const std::vector<std::uint8_t> data = {0x80, 0x80, 0x80, 0x80, 0x80, 0x00};
    offset = 0;
    CHECK(!parse_obu(data, offset, obu, error));
  }

  // Structural short-OBU failures.
  {
    MetadataShortView view;
    CHECK(!parse_metadata_short(std::vector<std::uint8_t>{0x20}, view, error));
    // muh_layer_idc == 3 is forbidden in the short form.
    CHECK(!parse_metadata_short(
        std::vector<std::uint8_t>{0x30, 0x01, 0x80}, view, error));
    // Missing / wrong trailing bits.
    CHECK(!parse_metadata_short(
        std::vector<std::uint8_t>{0x20, 0x01}, view, error));
    CHECK(!parse_metadata_short(
        std::vector<std::uint8_t>{0x20, 0x01, 0x00}, view, error));
    // Cancel OBU with a non-empty body is structurally invalid.
    CHECK(!parse_metadata_short(
        std::vector<std::uint8_t>{0x28, 0x01, 0x00, 0x80}, view, error));
  }

  // Typed decoders reject short units.
  {
    ContentLightLevel cll;
    CHECK(!decode_cll(std::vector<std::uint8_t>{0x03, 0xE8, 0x01}, cll, error));
    MasteringDisplayColorVolume mdcv;
    std::vector<std::uint8_t> unit(23, 0);
    CHECK(!decode_mdcv(unit, mdcv, error));
  }

  // Group structural failures.
  {
    MetadataGroupView group;
    CHECK(!parse_metadata_group(std::vector<std::uint8_t>{0x40}, 0, group,
                                error));
    // metadata_unit_cnt_minus_1 == 16383 is out of range.
    CHECK(!parse_metadata_group(
        std::vector<std::uint8_t>{0x40, 0xFF, 0x7F, 0x80}, 0, group, error));
    // header_size smaller than the fixed layer fields.
    CHECK(!parse_metadata_group(
        std::vector<std::uint8_t>{0x40, 0x00, 0x01, 0x02, 0x04, 0x40, 0x80},
        0, group, error));
    // Reserved muh_reserved_zero_2bits set.
    CHECK(!parse_metadata_group(
        std::vector<std::uint8_t>{0x40, 0x00, 0x01, 0x06, 0x04, 0x40, 0x01,
                                  0x03, 0xE8, 0x01, 0x90, 0x80},
        0, group, error));
    // Truncated unit payload.
    CHECK(!parse_metadata_group(
        std::vector<std::uint8_t>{0x40, 0x00, 0x01, 0x06, 0x04, 0x40, 0x00,
                                  0x03, 0xE8},
        0, group, error));
    // Missing trailing bits.
    CHECK(!parse_metadata_group(
        std::vector<std::uint8_t>{0x40, 0x00, 0x01, 0x06, 0x04, 0x40, 0x00,
                                  0x03, 0xE8, 0x01, 0x90},
        0, group, error));
    // Reserved muh_layer_idc (4).
    CHECK(!parse_metadata_group(
        std::vector<std::uint8_t>{0x40, 0x00, 0x01, 0x06, 0x04, 0x80, 0x00,
                                  0x03, 0xE8, 0x01, 0x90, 0x80},
        0, group, error));
  }

  // A structurally valid but too-short CLL unit is counted, not fatal.
  {
    std::vector<std::uint8_t> tu = make_test_temporal_unit();
    // type 1, header_size 3, payload size 3, layer/pers/prio, 3 payload bytes.
    const std::vector<std::uint8_t> group_payload = {
        0x40, 0x00, 0x01, 0x06, 0x03, 0x40, 0x00, 0x03, 0xE8, 0x01, 0x80};
    const std::vector<std::uint8_t> group = make_obu(9, group_payload);
    std::vector<std::uint8_t> with_metadata;
    CHECK(insert_metadata_obus(tu, group,
                               MetadataPlacement::kBeforeFirstFrame,
                               with_metadata, error));
    HdrScanResult scan;
    CHECK(scan_hdr_metadata(with_metadata, scan, error));
    CHECK(scan.metadata_obus == 1U);
    CHECK(scan.metadata_units == 1U);
    CHECK(scan.records.empty());
    CHECK(scan.invalid_units == 1U);
  }
}

void test_layer_values_group() {
  // §5.17.3 muh_layer_idc == LAYER_VALUES with obu_xlayer_id ==
  // GLOBAL_XLAYER_ID: one 32-bit xlayer map plus one mlayer byte per set bit.
  std::string error;
  std::vector<std::uint8_t> payload = {
      0x40, 0x00,              // group header, one unit
      0x01,                    // metadata_type = HDR_CLL
      0x10,                    // header_size = 8, cancel = 0
      0x04,                    // payload_size = 4
      0x60, 0x00,              // layer_idc = 3, persistence = 0, prio = 0
      0x00, 0x00, 0x00, 0x01,  // muh_xlayer_map: bit 0
      0x01,                    // muh_mlayer_map
      0x03, 0xE8, 0x01, 0x90,  // CLL unit
      0x80};                   // trailing_bits
  MetadataGroupView group;
  CHECK(parse_metadata_group(payload, av2enc::hdr::kGlobalXlayerId, group,
                             error));
  CHECK(group.units.size() == 1U);
  if (group.units.size() == 1U) {
    CHECK(group.units[0].layer == MetadataLayer::kValues);
    ContentLightLevel cll;
    CHECK(decode_cll(group.units[0].unit, cll, error));
    CHECK(cll == sample_cll());
  }

  // muh_xlayer_map bit 31 is forbidden.
  payload[7] = 0x80;
  CHECK(!parse_metadata_group(payload, av2enc::hdr::kGlobalXlayerId, group,
                              error));

  // Truncated map: header_size promises 5 map bytes but only 1 is present.
  const std::vector<std::uint8_t> truncated = {
      0x40, 0x00, 0x01, 0x10, 0x04, 0x60, 0x00, 0x00, 0x80};
  CHECK(!parse_metadata_group(truncated, av2enc::hdr::kGlobalXlayerId, group,
                              error));
}

// ---------------------------------------------------------------------------

struct TestCase {
  const char* name;
  void (*run)();
};

}  // namespace

int main() {
  const TestCase tests[] = {
      {"cll_golden", test_cll_golden},
      {"mdcv_golden", test_mdcv_golden},
      {"group_golden", test_group_golden},
      {"suffix_and_cancel_goldens", test_suffix_and_cancel_goldens},
      {"legacy_layout_goldens", test_legacy_layout_goldens},
      {"round_trip", test_round_trip},
      {"extreme_values_round_trip", test_extreme_values_round_trip},
      {"extra_unit_bytes", test_extra_unit_bytes},
      {"validation", test_validation},
      {"writer_validation", test_writer_validation},
      {"multiple_obus", test_multiple_obus},
      {"layer_values_group", test_layer_values_group},
      {"cancel_units_are_not_records", test_cancel_units_are_not_records},
      {"ordering", test_ordering},
      {"bit_depth_independence", test_bit_depth_independence},
      {"invalid_inputs", test_invalid_inputs},
  };

  for (const auto& test : tests) {
    const int before = g_failures;
    test.run();
    std::cout << (g_failures == before ? "PASS " : "FAIL ") << test.name
              << '\n';
  }

  if (g_failures != 0) {
    std::cerr << "hdr_metadata_test: " << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "hdr_metadata_test: PASS\n";
  return 0;
}
