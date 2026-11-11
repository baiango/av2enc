#pragma once

// AV2 v1.0.0 metadata unit types other than HDR CLL/MDCV.
//
// Normative reference: spec/AV2_Spec_v1.0.0.pdf. Section numbers below are
// from that document.
//
//   §5.17.1  metadata_unit(metadataPayloadSize), metadata_unit_remaining_bit
//   §5.17.2  metadata_short_obu(obuPayloadSize)
//   §5.17.3  metadata_group_obu()
//   §5.17.4  metadata_itut_t35(metadataPayloadSize)
//   §5.17.7  metadata_timecode()
//   §5.17.8  metadata_banding_hints()
//   §5.17.9  metadata_icc_profile(metadataPayloadSize)
//   §5.17.10 metadata_scan_type()
//   §5.17.11 metadata_temporal_point_info()
//   §5.17.12 metadata_decoded_frame_hash()
//   §5.17.13 metadata_user_data_unregistered(metadataPayloadSize)
//   §6.16.4  ITU-T T.35 semantics
//   §6.16.7  timecode semantics (counting_type, timestamp, time offset)
//   §6.16.8  banding hints semantics
//   §6.16.9  ICC profile semantics (ICCmajorVer / ICCminorVer)
//   §6.16.10 scan type semantics (Table 6.18)
//   §6.16.11 temporal point info semantics (short OBU only)
//   §6.16.12 user data unregistered semantics
//   §6.16.13 decoded frame hash semantics (MD5, per-plane / whole-frame)
//   §6.16.3  Table 6.17 metadata_type values, layer-specific status
//   §7.3.3   prefix/suffix metadata placement in coded output frame units
//
// OBU framing is delegated to av2enc::hdr: this module reuses its
// make_metadata_short_obu() / make_metadata_group_obu() writers, its
// parse_obu() / parse_metadata_short() / parse_metadata_group() readers and
// its insert_metadata_obus() placement helper. The HDR files are not modified.
//
// Design notes
// ------------
// * `serialize_*_unit()` emits the canonical metadata_unit() payload. For
//   non-byte-aligned syntax (timecode, banding hints) the final byte is
//   zero-padded with metadata_unit_remaining_bit. `*_unit_size()` reports the
//   exact canonical byte count, and the framing writers reject any payload
//   that is not exactly canonical, so the encoder never emits ambiguous
//   padding.
// * `decode_*()` accepts a payload at least as long as the syntax requires.
//   Extra bytes are §5.17.1 metadata_unit_remaining_bit and are ignored.
//   Truncated payloads and reserved syntax fields (e.g. decoded frame hash
//   `reserved`) are errors.
// * `validate_*()` reports semantic conformance. `strict == false` turns
//   reserved-but-parseable values into warnings; `strict == true` makes them
//   errors. Normative "shall" violations are always errors.
// * Layer/persistence/necessity/application/cancel flags live in the OBU
//   layer and are shared with the HDR module (hdr::MetadataObuConfig and
//   hdr::MetadataGroupEntry semantics).
// * METADATA_TYPE_TEMPORAL_POINT_INFO is short-OBU-only (§6.16.11); the group
//   writer rejects it and the scanner reports it as invalid.

#include "encoder/hdr/hdr_metadata.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace av2enc::metadata {

// §6.16.3 Table 6.17 values reused from the HDR module.
using Type = hdr::MetadataType;

inline constexpr std::size_t kItutT35MinUnitBytes = 1;   // country code
inline constexpr std::size_t kTimecodeMinUnitBytes = 3;  // 23 syntax bits
inline constexpr std::size_t kTimecodeMaxUnitBytes = 10;  // 73 syntax bits
inline constexpr std::size_t kBandingHintsMinUnitBytes = 1;  // 2 syntax bits
// 1396 syntax bits at the maximum 32x32 band-unit grid.
inline constexpr std::size_t kBandingHintsMaxUnitBytes = 175;
inline constexpr std::size_t kScanTypeUnitBytes = 1;
inline constexpr std::size_t kTemporalPointInfoMaxUnitBytes = 5;  // 32-bit leb128
inline constexpr std::size_t kFrameHashBytes = 16;
inline constexpr std::size_t kDecodedFrameHashWholeUnitBytes = 1 + kFrameHashBytes;
inline constexpr std::size_t kDecodedFrameHashPerPlaneMonoUnitBytes =
    1 + kFrameHashBytes;
inline constexpr std::size_t kDecodedFrameHashPerPlaneColorUnitBytes =
    1 + 3 * kFrameHashBytes;
inline constexpr std::size_t kUserDataUnregisteredUuidBytes = 16;
inline constexpr std::size_t kUserDataUnregisteredMinUnitBytes =
    kUserDataUnregisteredUuidBytes;
inline constexpr std::size_t kBandingMaxBandUnitsPerSide = 32;
inline constexpr std::uint8_t kItutT35ExtensionCountryCode = 0xFF;

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

// §5.17.4 metadata_itut_t35(metadataPayloadSize).
//
// The extension byte is coded if and only if country_code == 0xFF. The
// payload bytes are passed through unchanged (T.35 terminal provider code
// first, §6.16.4).
struct ItutT35 {
  std::uint8_t country_code = 0xB5;  // United States (ITU-T T.35 Annex A)
  std::uint8_t country_code_extension_byte = 0;
  std::vector<std::uint8_t> payload;

  friend bool operator==(const ItutT35&, const ItutT35&) = default;
};

// §5.17.7 metadata_timecode(). Field widths: counting_type 5, n_frames 9,
// seconds/minutes 6, hours 5, time_offset_length 5, time_offset_value
// time_offset_length (0..31).
struct Timecode {
  std::uint8_t counting_type = 0;  // 0..6 defined, 7..31 reserved
  bool full_timestamp_flag = false;
  bool discontinuity_flag = false;
  bool cnt_dropped_flag = false;
  std::uint16_t n_frames = 0;  // 9 bits

  // Present only when full_timestamp_flag == 1 (all three), or when the flags
  // below select them (full_timestamp_flag == 0).
  bool seconds_flag = false;
  std::uint8_t seconds_value = 0;  // 0..59
  bool minutes_flag = false;
  std::uint8_t minutes_value = 0;  // 0..59
  bool hours_flag = false;
  std::uint8_t hours_value = 0;  // 0..23

  std::uint8_t time_offset_length = 0;  // 0..31 bits
  std::uint32_t time_offset_value = 0;

  friend bool operator==(const Timecode&, const Timecode&) = default;
};

// §5.17.8 metadata_banding_hints().
struct BandingComponentHint {
  bool banding_in_component_present_flag = false;
  std::uint8_t max_band_width_minus_4 = 0;  // 6 bits
  std::uint8_t max_band_step_minus_1 = 0;   // 4 bits

  friend bool operator==(const BandingComponentHint&,
                         const BandingComponentHint&) = default;
};

struct BandingHints {
  bool coding_banding_present_flag = false;
  bool source_banding_present_flag = false;
  bool banding_hints_flag = false;  // present only when coding is set
  bool three_color_components_flag = false;
  // Lane 0 is always used when hints are present; lanes 1 and 2 only when
  // three_color_components_flag is set.
  std::array<BandingComponentHint, 3> components{};

  bool band_units_information_present_flag = false;
  std::uint8_t num_band_units_rows_minus_1 = 0;  // 5 bits
  std::uint8_t num_band_units_cols_minus_1 = 0;  // 5 bits
  bool varying_size_band_units_flag = false;
  std::uint8_t band_block_in_luma_samples = 0;  // 3 bits; size = 16 << value
  std::vector<std::uint8_t> vert_size_in_band_blocks_minus_1;  // rows entries
  std::vector<std::uint8_t> horz_size_in_band_blocks_minus_1;  // cols entries
  // Row-major r * cols + c, one byte per band unit with value 0 or 1.
  std::vector<std::uint8_t> banding_in_band_unit_present;

  friend bool operator==(const BandingHints&, const BandingHints&) = default;
};

// §5.17.9 metadata_icc_profile(metadataPayloadSize).
struct IccProfile {
  std::vector<std::uint8_t> data;

  friend bool operator==(const IccProfile&, const IccProfile&) = default;
};

// §6.16.9: ICCmajorVer = bytes[8], ICCminorVer = bytes[9] >> 4.
struct IccProfileVersion {
  std::uint8_t major = 0;
  std::uint8_t minor = 0;

  friend bool operator==(const IccProfileVersion&,
                         const IccProfileVersion&) = default;
};
std::optional<IccProfileVersion> icc_profile_version(const IccProfile& profile);

// §5.17.10 metadata_scan_type().
struct ScanType {
  std::uint8_t mps_pic_struct_type = 0;       // 5 bits, 0..12 defined
  std::uint8_t mps_source_scan_type_idc = 0;  // 2 bits, same as ci_scan_type_idc
  bool mps_duplicate_flag = false;

  friend bool operator==(const ScanType&, const ScanType&) = default;
};

// §6.16.10 Table 6.18: the ci_scan_type_idc value required for a given
// mps_pic_struct_type. nullopt for values above 12 (reserved).
std::optional<std::uint8_t> scan_type_required_ci_scan_type_idc(
    std::uint8_t mps_pic_struct_type);

// §5.17.11 metadata_temporal_point_info(). `leb128()` of the presentation time
// in clock ticks DispCT, relative to the previous random access point.
struct TemporalPointInfo {
  std::uint32_t frame_presentation_time = 0;

  friend bool operator==(const TemporalPointInfo&,
                         const TemporalPointInfo&) = default;
};

using FrameHash = std::array<std::uint8_t, kFrameHashBytes>;

// §5.17.12 metadata_decoded_frame_hash().
struct DecodedFrameHash {
  std::uint8_t hash_type = 0;  // 4 bits; 0 = MD5
  bool per_plane = false;
  bool has_grain = false;
  bool is_monochrome = false;
  std::array<FrameHash, 3> plane_hash{};  // valid when per_plane
  FrameHash frame_hash{};                 // valid when !per_plane

  friend bool operator==(const DecodedFrameHash&,
                         const DecodedFrameHash&) = default;
};

// §5.17.13 metadata_user_data_unregistered(metadataPayloadSize).
struct UserDataUnregistered {
  std::array<std::uint8_t, kUserDataUnregisteredUuidBytes> uuid{};
  std::vector<std::uint8_t> payload;

  friend bool operator==(const UserDataUnregistered&,
                         const UserDataUnregistered&) = default;
};

// Typed value of any metadata unit this module (or the HDR module) can decode.
using Value =
    std::variant<hdr::ContentLightLevel, hdr::MasteringDisplayColorVolume,
                 ItutT35, Timecode, BandingHints, IccProfile, ScanType,
                 TemporalPointInfo, DecodedFrameHash, UserDataUnregistered>;

// True for metadata_type values 1..10 (CLL/MDCV and the eight types above).
bool is_known_type(std::uint32_t type);
bool is_known_type(Type type);

// False for METADATA_TYPE_TEMPORAL_POINT_INFO, which is short-OBU-only.
bool is_group_allowed(Type type);

// "hdr_cll", "itut_t35", ...; "reserved" for unknown values.
const char* type_name(std::uint32_t type);
const char* type_name(Type type);

// ---------------------------------------------------------------------------
// Semantic validation
// ---------------------------------------------------------------------------

// §6.16.4: the payload shall start with an ITU-T T.35 terminal provider code.
// The extension byte is only coded when country_code == 0xFF, so a non-zero
// extension byte with any other country code would be silently lost.
hdr::ValidationResult validate_itut_t35(const ItutT35& value,
                                        bool strict = false);

// §6.16.7: value ranges for seconds/minutes/hours, n_frames (9 bits),
// time_offset_length (5 bits) and time_offset_value (fits the length).
// Reserved counting_type values warn unless strict.
hdr::ValidationResult validate_timecode(const Timecode& value,
                                        bool strict = false);

// §6.16.8: absent syntax fields must stay at their defaults so a round-trip is
// lossless; array lengths must match the signalled band-unit grid.
hdr::ValidationResult validate_banding_hints(const BandingHints& value,
                                             bool strict = false);

// §6.16.9: a profile shorter than 10 bytes has no ICCmajorVer/ICCminorVer.
// That is not a structural error (the payload is opaque), so it warns unless
// strict.
hdr::ValidationResult validate_icc_profile(const IccProfile& value,
                                           bool strict = false);

// §6.16.10 Table 6.18: mps_pic_struct_type above 12 is reserved and "shall not
// be present in bitstreams conforming to this specification" (always an
// error). mps_source_scan_type_idc is fully defined by its 2 bits.
hdr::ValidationResult validate_scan_type(const ScanType& value,
                                         bool strict = false);

// §6.16.11: frame_presentation_time is a 32-bit leb128; every value is legal.
hdr::ValidationResult validate_temporal_point_info(
    const TemporalPointInfo& value, bool strict = false);

// §6.16.13: hash_type 0 (MD5) is defined; other values are reserved for
// AOMedia use (warning unless strict).
hdr::ValidationResult validate_decoded_frame_hash(
    const DecodedFrameHash& value, bool strict = false);

// §6.16.12: UUID and payload bytes are opaque.
hdr::ValidationResult validate_user_data_unregistered(
    const UserDataUnregistered& value, bool strict = false);

// ---------------------------------------------------------------------------
// Canonical payload sizes and serialization
// ---------------------------------------------------------------------------

std::size_t itut_t35_unit_size(const ItutT35& value);
std::size_t timecode_syntax_bits(const Timecode& value);
std::size_t timecode_unit_size(const Timecode& value);
std::size_t banding_hints_syntax_bits(const BandingHints& value);
std::size_t banding_hints_unit_size(const BandingHints& value);
std::size_t icc_profile_unit_size(const IccProfile& value);
std::size_t temporal_point_info_unit_size(const TemporalPointInfo& value);
std::size_t decoded_frame_hash_unit_size(const DecodedFrameHash& value);
std::size_t user_data_unregistered_unit_size(const UserDataUnregistered& value);

// Canonical §5.17 payload bytes. Do not validate; use make_*_unit() for a
// validated path. Fixed-width fields are MSB-first; non-byte-aligned syntax is
// zero-padded to the next byte boundary (§5.17.1).
std::vector<std::uint8_t> serialize_itut_t35_unit(const ItutT35& value);
std::vector<std::uint8_t> serialize_timecode_unit(const Timecode& value);
std::vector<std::uint8_t> serialize_banding_hints_unit(
    const BandingHints& value);
std::vector<std::uint8_t> serialize_icc_profile_unit(const IccProfile& value);
std::vector<std::uint8_t> serialize_scan_type_unit(const ScanType& value);
std::vector<std::uint8_t> serialize_temporal_point_info_unit(
    const TemporalPointInfo& value);
std::vector<std::uint8_t> serialize_decoded_frame_hash_unit(
    const DecodedFrameHash& value);
std::vector<std::uint8_t> serialize_user_data_unregistered_unit(
    const UserDataUnregistered& value);

// A metadata_type plus its canonical metadata_unit() payload.
struct MetadataUnitPayload {
  Type type = Type::kItutT35;
  std::vector<std::uint8_t> bytes;
};

// Validate, then serialize. Returns nullopt and sets `error` when the value is
// not a legal bitstream value (always errors) or, under `strict`, when it uses
// reserved-but-parseable values. `config.strict_ranges` selects strict.
std::optional<MetadataUnitPayload> make_itut_t35_unit(const ItutT35& value,
                                                      std::string& error,
                                                      bool strict = false);
std::optional<MetadataUnitPayload> make_timecode_unit(const Timecode& value,
                                                      std::string& error,
                                                      bool strict = false);
std::optional<MetadataUnitPayload> make_banding_hints_unit(
    const BandingHints& value, std::string& error, bool strict = false);
std::optional<MetadataUnitPayload> make_icc_profile_unit(
    const IccProfile& value, std::string& error, bool strict = false);
std::optional<MetadataUnitPayload> make_scan_type_unit(const ScanType& value,
                                                       std::string& error,
                                                       bool strict = false);
std::optional<MetadataUnitPayload> make_temporal_point_info_unit(
    const TemporalPointInfo& value, std::string& error, bool strict = false);
std::optional<MetadataUnitPayload> make_decoded_frame_hash_unit(
    const DecodedFrameHash& value, std::string& error, bool strict = false);
std::optional<MetadataUnitPayload> make_user_data_unregistered_unit(
    const UserDataUnregistered& value, std::string& error,
    bool strict = false);

// Canonical unit size of an already serialized value (used by the framing
// writers to reject non-canonical padding).
std::size_t canonical_unit_size(const Value& value);

// ---------------------------------------------------------------------------
// OBU writers (framing delegated to av2enc::hdr)
// ---------------------------------------------------------------------------

// Builds one OBU_METADATA_SHORT carrying `unit`. Validates:
//   * the type is known and its payload is exactly canonical;
//   * muh_layer_idc < 3, muh_persistence_idc < 4;
//   * config.cancel must be false (use make_short_cancel_obu()).
// metadata_necessity_idc and metadata_application_id are implicitly 0 in the
// short form, so config.necessity/config.application are ignored here; set
// them only for make_group_obu().
std::vector<std::uint8_t> make_short_obu(const MetadataUnitPayload& unit,
                                         const hdr::MetadataObuConfig& config,
                                         std::string& error);

// A cancel OBU: metadata_type is signalled, no metadata_unit body
// (muh_cancel_flag == 1, §5.17.2).
std::vector<std::uint8_t> make_short_cancel_obu(
    Type type, const hdr::MetadataObuConfig& config, std::string& error);

// One unit inside a group OBU. For a cancel entry `payload.bytes` must be
// empty; the type is still signalled.
struct GroupUnit {
  MetadataUnitPayload payload;
  hdr::MetadataLayer layer = hdr::MetadataLayer::kCurrent;
  hdr::MetadataPersistence persistence = hdr::MetadataPersistence::kGlobal;
  std::uint8_t priority = 0;
  bool cancel = false;
};

// Builds one OBU_METADATA_GROUP (§5.17.3) with per-unit layer/persistence/
// priority. `config` supplies suffix, necessity and application only; the
// short-form cancel flag and per-unit layer/persistence do not come from it.
// Rejects METADATA_TYPE_TEMPORAL_POINT_INFO (§6.16.11), reserved application
// ids 6..15, reserved persistence values 4..7, LAYER_VALUES (no layer maps in
// this writer), non-canonical payloads and empty groups.
std::vector<std::uint8_t> make_group_obu(std::span<const GroupUnit> units,
                                         const hdr::MetadataObuConfig& config,
                                         std::string& error);

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

// Each decoder consumes the §5.17 payload (no OBU framing) and returns false
// on truncation or a reserved field that must be zero. Extra trailing bytes
// are accepted as §5.17.1 metadata_unit_remaining_bit.
bool decode_itut_t35(std::span<const std::uint8_t> unit, ItutT35& out,
                     std::string& error);
bool decode_timecode(std::span<const std::uint8_t> unit, Timecode& out,
                     std::string& error);
bool decode_banding_hints(std::span<const std::uint8_t> unit,
                          BandingHints& out, std::string& error);
bool decode_icc_profile(std::span<const std::uint8_t> unit, IccProfile& out,
                        std::string& error);
bool decode_scan_type(std::span<const std::uint8_t> unit, ScanType& out,
                      std::string& error);
bool decode_temporal_point_info(std::span<const std::uint8_t> unit,
                                TemporalPointInfo& out, std::string& error);
bool decode_decoded_frame_hash(std::span<const std::uint8_t> unit,
                               DecodedFrameHash& out, std::string& error);
bool decode_user_data_unregistered(std::span<const std::uint8_t> unit,
                                   UserDataUnregistered& out,
                                   std::string& error);

// Typed decode keyed by metadata_type. Supports CLL/MDCV through the HDR
// module. Returns false for reserved/unknown types and malformed payloads.
bool decode_unit(Type type, std::span<const std::uint8_t> unit, Value& out,
                 std::string& error);

// ---------------------------------------------------------------------------
// Stream scanning
// ---------------------------------------------------------------------------

struct MetadataRecord {
  std::size_t obu_offset = 0;  // offset of the metadata OBU size prefix
  bool in_group = false;
  bool is_suffix = false;
  bool before_first_frame = false;
  std::uint32_t metadata_type = 0;
  hdr::MetadataLayer layer = hdr::MetadataLayer::kUnspecified;
  hdr::MetadataPersistence persistence = hdr::MetadataPersistence::kGlobal;
  std::uint8_t priority = 0;
  bool cancel = false;
  bool has_value = false;
  Value value;
};

struct ScanResult {
  std::vector<MetadataRecord> records;  // stream order, incl. cancel records
  std::size_t metadata_obus = 0;        // short + group OBUs
  std::size_t metadata_units = 0;       // every unit, incl. cancelled
  std::size_t supported_units = 0;      // decoded known types (1..10)
  std::size_t unsupported_units = 0;    // reserved / future metadata_type
  std::size_t invalid_units = 0;        // known type, malformed payload
                                        // or group-carried temporal point info
  std::size_t cancelled_units = 0;
};

// Walks a complete temporal unit. Structural OBU/group errors are failures;
// malformed known-type payloads increment invalid_units and are skipped.
bool scan_metadata(std::span<const std::uint8_t> temporal_unit,
                   ScanResult& out, std::string& error);

}  // namespace av2enc::metadata
