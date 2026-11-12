#pragma once

// AV2 v1.0.0 metadata-OBU gaps: cancel flags, LAYER_VALUES layer maps,
// muh_header_extension_byte and a self-parser that exposes all three.
//
// Normative reference: spec/AV2_Spec_v1.0.0.pdf.
//
//   §5.17.2  metadata_short_obu(obuPayloadSize)
//   §5.17.3  metadata_group_obu()
//   §6.16.2  metadata short OBU semantics (muh_layer_idc < 3, cancel)
//   §6.16.3  metadata group OBU semantics (cancel, LAYER_VALUES maps,
//            muh_header_extension_byte, muh_reserved_zero_2bits, persistence)
//   §5.2.2   obu_header() (obu_header_extension_flag / mlayer / xlayer)
//
// Why this module exists
// ----------------------
// `src/encoder/metadata/types/` (owned by another agent) writes the canonical
// unit payloads and the short/group OBU shells, but its group writer rejects
// `muh_layer_idc == LAYER_VALUES` and never emits `muh_header_extension_byte`;
// its group reader parses the layer maps and then drops them. This module owns
// exactly the missing header surface:
//
//   * `muh_cancel_flag == 1` with the metadata_type still signalled and no
//     metadata_unit body, in both the short and group forms;
//   * `muh_layer_idc == LAYER_VALUES` (3): `muh_xlayer_map f(32)` followed by
//     one `muh_mlayer_map f(8)` per set bit when `obu_xlayer_id ==
//     GLOBAL_XLAYER_ID`, or a single `muh_mlayer_map f(8)` otherwise;
//   * `muh_header_extension_byte f(8)` bytes (opaque, decoder-ignored);
//   * the matching reader and temporal-unit scanner, including the exact
//     consumed header bytes.
//
// The existing HDR module (`hdr::make_metadata_group_obu`,
// `hdr::parse_metadata_group` and `hdr::insert_metadata_obus`) is reused for
// OBU framing/placement. For a group unit with no layer maps and no header
// extension this module's writer is byte-identical to the HDR writer; the test
// pins that equivalence so the two implementations cannot drift silently.
//
// Design notes
// ------------
// * Writers validate before emitting: a cancel unit carries no payload; a
//   non-cancel unit decodes to a known metadata_type at exactly its canonical
//   size; `muh_header_size <= 127` (f(7)); `muh_reserved_zero_2bits == 0`;
//   bit 31 of `muh_xlayer_map` is zero; and every `muh_mlayer_map` clears the
//   bits below `obu_mlayer_id` (§6.16.3).
// * The OBU header extension byte is emitted only when `obu_mlayer_id` or
//   `obu_xlayer_id` is non-zero, exactly as §5.2.2 requires. Metadata OBUs are
//   not in `is_extensible_obu()`, so the payload tail is plain `trailing_bits`
//   (a single 0x80 byte), never an `obu_extension_flag`.
// * Readers consume a complete Annex B.2 length-prefixed OBU, verify the OBU
//   type, the trailing bits and every header invariant, and return payload
//   spans into the caller's buffer.

#include "frame/metadata/types/metadata_types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::metadata::gaps {

using ::av2enc::metadata::Value;

// §5.2.2: GLOBAL_XLAYER_ID.
inline constexpr std::uint8_t kGlobalXlayerId = 31;

// §6.2.2 Table 6.1.
inline constexpr std::uint8_t kObuTypeMetadataShort = 8;
inline constexpr std::uint8_t kObuTypeMetadataGroup = 9;

// §5.17.3: muh_header_size f(7).
inline constexpr std::size_t kMaxMetadataHeaderBytes = 127;

// ---------------------------------------------------------------------------
// Header description
// ---------------------------------------------------------------------------

// §5.17.3 muh_xlayer_map f(32) + muh_mlayer_map f(8) sequence.
//
// When `obu_xlayer_id == GLOBAL_XLAYER_ID`, `xlayer_map` is coded and one
// `mlayer_map` byte is written for each set bit of `xlayer_map` in increasing
// bit order (bits 0..30; bit 31 must be zero). Otherwise `xlayer_map` is not
// coded and `mlayer_map` holds exactly one byte for the current xlayer.
struct LayerValueMaps {
  std::uint32_t xlayer_map = 0;
  std::vector<std::uint8_t> mlayer_map;

  friend bool operator==(const LayerValueMaps&, const LayerValueMaps&) =
      default;
};

// One §5.17.3 metadata-unit header. `muh_header_size` is derived, not stored.
struct MetadataUnitHeader {
  Type type = Type::kHdrCll;
  bool cancel = false;  // muh_cancel_flag
  hdr::MetadataLayer layer = hdr::MetadataLayer::kCurrent;  // muh_layer_idc
  hdr::MetadataPersistence persistence =
      hdr::MetadataPersistence::kGlobal;  // muh_persistence_idc
  std::uint8_t priority = 0;              // muh_priority f(8)
  LayerValueMaps layer_values;            // only when layer == kValues
  std::vector<std::uint8_t> header_extension;  // muh_header_extension_byte

  friend bool operator==(const MetadataUnitHeader&,
                         const MetadataUnitHeader&) = default;
};

// A metadata unit (header + §5.17.1 payload). For a cancel unit `payload` must
// be empty: only the type is signalled.
struct MetadataGapUnit {
  MetadataUnitHeader header;
  std::vector<std::uint8_t> payload;

  friend bool operator==(const MetadataGapUnit&, const MetadataGapUnit&) =
      default;
};

// OBU-level configuration. `suffix`, `necessity` and `application` are the
// §5.17.2/§5.17.3 OBU header options; `tlayer_id`/`mlayer_id`/`xlayer_id` are
// the §5.2.2 obu_header() fields.
struct GapObuConfig {
  bool suffix = false;
  hdr::MetadataNecessity necessity = hdr::MetadataNecessity::kAdvisory;
  hdr::MetadataApplicationId application =
      hdr::MetadataApplicationId::kUnspecified;
  std::uint8_t tlayer_id = 0;  // f(2)
  std::uint8_t mlayer_id = 0;  // f(3), OBU header extension only
  std::uint8_t xlayer_id = 0;  // f(5), OBU header extension only
};

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

// §5.17.2 OBU_METADATA_SHORT. Rejects LAYER_VALUES (the short form requires
// muh_layer_idc < 3), layer maps and header-extension bytes. A cancel unit
// emits the options byte and the metadata_type, then trailing_bits.
std::vector<std::uint8_t> make_short_obu(const MetadataGapUnit& unit,
                                         const GapObuConfig& config,
                                         std::string& error);

// Convenience: a short-form cancel OBU for `type` (no body).
std::vector<std::uint8_t> make_short_cancel_obu(Type type,
                                                const GapObuConfig& config,
                                                std::string& error);

// §5.17.3 OBU_METADATA_GROUP. Supports per-unit cancel, LAYER_VALUES maps and
// muh_header_extension_byte. The OBU header extension byte is emitted when
// config.mlayer_id/config.xlayer_id is non-zero.
std::vector<std::uint8_t> make_group_obu(std::span<const MetadataGapUnit> units,
                                         const GapObuConfig& config,
                                         std::string& error);

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

struct ParsedShortObu {
  bool is_suffix = false;
  hdr::MetadataLayer layer = hdr::MetadataLayer::kUnspecified;
  hdr::MetadataPersistence persistence = hdr::MetadataPersistence::kGlobal;
  bool cancel = false;
  std::uint32_t type = 0;
  std::size_t obu_bytes = 0;  // total bytes including the leb128 prefix
  std::span<const std::uint8_t> payload;  // empty when cancelled
};

struct ParsedUnit {
  MetadataUnitHeader header;
  std::span<const std::uint8_t> payload;  // empty when cancelled
};

struct ParsedGroupObu {
  bool is_suffix = false;
  hdr::MetadataNecessity necessity = hdr::MetadataNecessity::kUndefined;
  hdr::MetadataApplicationId application =
      hdr::MetadataApplicationId::kUnspecified;
  std::uint8_t tlayer_id = 0;
  std::uint8_t mlayer_id = 0;
  std::uint8_t xlayer_id = 0;
  std::size_t obu_bytes = 0;
  std::vector<ParsedUnit> units;
};

// `obu` is a complete Annex B.2 length-prefixed OBU. The payload spans point
// into `obu`; keep it alive for as long as the parsed view is used.
bool parse_short_obu(std::span<const std::uint8_t> obu, ParsedShortObu& out,
                     std::string& error);
bool parse_group_obu(std::span<const std::uint8_t> obu, ParsedGroupObu& out,
                     std::string& error);

// ---------------------------------------------------------------------------
// Temporal-unit scanning
// ---------------------------------------------------------------------------

struct GapRecord {
  std::size_t obu_offset = 0;  // offset of the leb128 size prefix
  bool in_group = false;
  bool is_suffix = false;
  bool before_first_frame = false;
  std::uint32_t type = 0;
  hdr::MetadataLayer layer = hdr::MetadataLayer::kUnspecified;
  hdr::MetadataPersistence persistence = hdr::MetadataPersistence::kGlobal;
  std::uint8_t priority = 0;
  bool cancel = false;
  std::uint32_t xlayer_map = 0;
  std::vector<std::uint8_t> mlayer_map;
  std::vector<std::uint8_t> header_extension;
  bool has_value = false;
  Value value;
};

struct GapScanResult {
  std::vector<GapRecord> records;  // stream order, including cancel records
  std::size_t metadata_obus = 0;
  std::size_t metadata_units = 0;
  std::size_t supported_units = 0;
  std::size_t unsupported_units = 0;
  std::size_t invalid_units = 0;
  std::size_t cancelled_units = 0;
};

// Structural OBU/header errors are failures. A known-type unit whose payload
// does not decode increments invalid_units and is skipped.
bool scan_gap_metadata(std::span<const std::uint8_t> temporal_unit,
                       GapScanResult& out, std::string& error);

}  // namespace av2enc::metadata::gaps
