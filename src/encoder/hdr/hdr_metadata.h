#pragma once

// AV2 v1.0.0 HDR metadata OBUs: high dynamic range content light level (CLL)
// and mastering display colour volume (MDCV).
//
// Normative reference: spec/AV2_Spec_v1.0.0.pdf. Section numbers below are
// from that document.
//
//   §5.2.1  open_bitstream_unit(): OBU payload/trailing_bits handling
//   §5.2.2  obu_header(): obu_header_extension_flag, obu_type, obu_tlayer_id
//   §5.17.1 metadata_unit(metadataPayloadSize)
//   §5.17.2 metadata_short_obu(obuPayloadSize)
//   §5.17.3 metadata_group_obu()
//   §5.17.5 metadata_hdr_cll():   f(16) max_cll, f(16) max_fall
//   §5.17.6 metadata_hdr_mdcv():  3x f(16) x/y primaries, f(16) white x/y,
//                                 f(32) luminance_max, f(32) luminance_min
//   §6.2.2  Table 6.1: OBU_METADATA_SHORT = 8, OBU_METADATA_GROUP = 9
//   §6.16.3 Table 6.17: METADATA_TYPE_HDR_CLL = 1, METADATA_TYPE_HDR_MDCV = 2
//   §6.16.5 CLL semantics, §6.16.6 MDCV semantics (value ranges)
//   §7.3.3  coded output frame unit ordering (prefix/suffix metadata)
//
// Only CLL and MDCV are HDR metadata units in the v1.0.0 spec. Colour
// description (primaries / transfer / matrix) is not part of metadata_unit();
// it lives in the LCR/OPS colour info and is out of scope here.
//
// Design notes
// ------------
// * Writers validate before emitting. Out-of-range MDCV values are legal in
//   the spec (they mean "unknown / unspecified", §6.16.6) so they are warned
//   about, not silently clamped. `strict_ranges` turns those warnings into
//   errors for encoder-side sanity. The one normative violation
//   (luminance_max == 50000 && luminance_min == 50000) is always rejected.
// * Readers are structural: a malformed OBU (bad leb128, truncated payload,
//   wrong trailing bits, reserved bits set) is an error. A structurally valid
//   metadata unit whose payload is too short for CLL/MDCV is reported as
//   invalid by the scanner rather than crashing.
// * `metadata_unit_remaining_bit` (§5.17.1) permits payloads longer than the
//   4/24 canonical bytes; decoders accept the extra bits, so `decode_*`
//   accepts them too. Writers always emit the canonical size.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::hdr {

// §6.2.2 Table 6.1.
inline constexpr std::uint8_t kObuTypeMetadataShort = 8;
inline constexpr std::uint8_t kObuTypeMetadataGroup = 9;

// §5.2.2: without obu_header_extension_flag, obu_xlayer_id is 0 (metadata is
// not MSDO/temporal delimiter). OBU_METADATA_* may also use GLOBAL_XLAYER_ID
// (31) with the extension flag, but only as a temporal-unit prefix ahead of
// the coded extended layer unit (§7.3.7); this module does not emit that form.
inline constexpr std::uint8_t kGlobalXlayerId = 31;

// §6.16.3 Table 6.17.
enum class MetadataType : std::uint32_t {
  kReserved = 0,
  kHdrCll = 1,
  kHdrMdcv = 2,
  kItutT35 = 3,
  kTimecode = 4,
  kDecodedFrameHash = 5,
  kBandingHints = 6,
  kIccProfile = 7,
  kScanType = 8,
  kTemporalPointInfo = 9,
  kUserDataUnregistered = 10,
};

// §6.16.3 metadata_necessity_idc (group OBU / short OBU is fixed to 0).
enum class MetadataNecessity : std::uint8_t {
  kUndefined = 0,
  kNecessary = 1,
  kAdvisory = 2,
  kMixed = 3,
};

// §6.16.3 Table 6.16.
enum class MetadataApplicationId : std::uint8_t {
  kUnspecified = 0,
  kMobileOrTv = 1,
  kMobile = 2,
  kTv = 3,
  kHmd = 4,
  kWearable = 5,
};

// §6.16.3 muh_persistence_idc.
enum class MetadataPersistence : std::uint8_t {
  kGlobal = 0,
  kBasic = 1,
  kNoPersistence = 2,
  kEnhanced = 3,
};

// §6.16.3 muh_layer_idc.
enum class MetadataLayer : std::uint8_t {
  kUnspecified = 0,
  kGlobal = 1,
  kCurrent = 2,
  kValues = 3,
};

// §5.17.6 metadata_hdr_cll(). Both fields are unsigned 16-bit; 0 means "no
// upper bound is signalled" (§6.16.5), not "zero nits".
struct ContentLightLevel {
  std::uint16_t max_cll = 0;   // cd/m^2 upper bound over all samples
  std::uint16_t max_fall = 0;  // cd/m^2 upper bound on frame average

  friend bool operator==(const ContentLightLevel&, const ContentLightLevel&) =
      default;
};

// §5.17.6 metadata_hdr_mdcv(). Units are the SMPTE ST 2086 integers:
// chromaticities in 0.00002, luminance in 0.0001 cd/m^2. Index 0/1/2 is
// suggested as green/blue/red (§6.16.6).
struct MasteringDisplayColorVolume {
  std::array<std::uint16_t, 3> primary_chromaticity_x{};
  std::array<std::uint16_t, 3> primary_chromaticity_y{};
  std::uint16_t white_point_chromaticity_x = 0;
  std::uint16_t white_point_chromaticity_y = 0;
  std::uint32_t luminance_max = 0;
  std::uint32_t luminance_min = 0;

  friend bool operator==(const MasteringDisplayColorVolume&,
                         const MasteringDisplayColorVolume&) = default;
};

// Convenience container: each unit is optional and emitted in CLL, MDCV order.
struct HdrMetadata {
  bool has_cll = false;
  ContentLightLevel cll;
  bool has_mdcv = false;
  MasteringDisplayColorVolume mdcv;
};

// §6.16.6 value ranges. Values outside these ranges are not conformance
// errors: they signal that the coordinate/luminance is unknown.
inline constexpr std::uint16_t kChromaticityXMin = 5;
inline constexpr std::uint16_t kChromaticityXMax = 37000;
inline constexpr std::uint16_t kChromaticityYMin = 5;
inline constexpr std::uint16_t kChromaticityYMax = 42000;
inline constexpr std::uint32_t kLuminanceMaxMin = 50000;
inline constexpr std::uint32_t kLuminanceMaxMax = 100000000;
inline constexpr std::uint32_t kLuminanceMinMin = 1;
inline constexpr std::uint32_t kLuminanceMinMax = 50000;

// Canonical metadata_unit() payload sizes (bytes).
inline constexpr std::size_t kCllUnitBytes = 4;
inline constexpr std::size_t kMdcvUnitBytes = 24;

enum class Severity : std::uint8_t { kError = 0, kWarning = 1 };

struct ValidationIssue {
  Severity severity = Severity::kError;
  std::string message;
};

class ValidationResult {
 public:
  void add(Severity severity, std::string message);
  const std::vector<ValidationIssue>& issues() const { return issues_; }
  bool has_errors() const;
  bool has_warnings() const;
  // True when there are no issues at all.
  bool clean() const { return issues_.empty(); }
  // True when no error-level issue is present.
  bool ok() const { return !has_errors(); }
  std::string summary() const;  // newline-joined "error: ..."/"warning: ..."

 private:
  std::vector<ValidationIssue> issues_;
};

// §6.16.5 has no reserved CLL values: every 16-bit value is encodable, and 0
// means "not signalled". `strict` has no effect; it exists for symmetry.
ValidationResult validate_cll(const ContentLightLevel& cll, bool strict = false);

// §6.16.6. With strict == false, values outside the documented ranges produce
// warnings (spec-legal "unknown"); with strict == true they produce errors.
// The normative constraint luminance_max == 50000 -> luminance_min != 50000 is
// always an error.
ValidationResult validate_mdcv(const MasteringDisplayColorVolume& mdcv,
                               bool strict = false);

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

enum class HdrObuPacking : std::uint8_t {
  kShortObus = 0,      // one OBU_METADATA_SHORT per unit (CLL then MDCV)
  kMetadataGroup = 1,  // one OBU_METADATA_GROUP carrying both units
};

struct MetadataObuConfig {
  // metadata_is_suffix: false = prefix (before frame data), true = suffix.
  // §7.3.3: prefix goes in the coded output frame unit before the frame OBUs,
  // suffix after them.
  bool suffix = false;
  // muh_layer_idc. The writer only emits LAYER_UNSPECIFIED/GLOBAL/CURRENT;
  // LAYER_VALUES needs explicit layer maps and is rejected.
  MetadataLayer layer = MetadataLayer::kCurrent;
  MetadataPersistence persistence = MetadataPersistence::kGlobal;
  // Short OBUs only. For a cancel OBU the caller must pass an empty unit.
  bool cancel = false;
  // Turn MDCV range warnings into errors; the normative luminance constraint
  // is rejected either way.
  bool strict_ranges = false;
  // Group OBU only.
  MetadataNecessity necessity = MetadataNecessity::kAdvisory;
  MetadataApplicationId application = MetadataApplicationId::kUnspecified;
};

// Canonical §5.17.5 / §5.17.6 metadata_unit() payloads (big-endian f(16)/f(32)).
std::vector<std::uint8_t> serialize_cll_unit(const ContentLightLevel& cll);
std::vector<std::uint8_t> serialize_mdcv_unit(
    const MasteringDisplayColorVolume& mdcv);

// Builds one OBU_METADATA_SHORT (type 8), including the Annex B.2 leb128 size
// prefix used by the M1 temporal unit. `metadata_unit` is the §5.17.1 payload.
// On failure returns an empty vector and sets `error`.
std::vector<std::uint8_t> make_metadata_short_obu(
    MetadataType type, std::span<const std::uint8_t> metadata_unit,
    const MetadataObuConfig& config, std::string& error);

std::vector<std::uint8_t> make_cll_short_obu(const ContentLightLevel& cll,
                                             const MetadataObuConfig& config,
                                             std::string& error);
std::vector<std::uint8_t> make_mdcv_short_obu(
    const MasteringDisplayColorVolume& mdcv, const MetadataObuConfig& config,
    std::string& error);

// A cancel OBU: metadata_type is signalled, no metadata_unit body.
std::vector<std::uint8_t> make_cancel_short_obu(MetadataType type,
                                                const MetadataObuConfig& config,
                                                std::string& error);

// One metadata unit inside a group OBU. `unit` is the §5.17.1 payload.
struct MetadataGroupEntry {
  MetadataType type = MetadataType::kHdrCll;
  std::vector<std::uint8_t> unit;
  MetadataLayer layer = MetadataLayer::kCurrent;
  MetadataPersistence persistence = MetadataPersistence::kGlobal;
  std::uint8_t priority = 0;
  bool cancel = false;
};

// Builds one OBU_METADATA_GROUP (type 9) with the given units, in order.
std::vector<std::uint8_t> make_metadata_group_obu(
    std::span<const MetadataGroupEntry> entries,
    const MetadataObuConfig& config, std::string& error);

// HDR convenience writer. Concatenates the OBUs for whichever units are
// present in `metadata`, in CLL -> MDCV order.
std::vector<std::uint8_t> make_hdr_metadata_obus(
    const HdrMetadata& metadata, HdrObuPacking packing,
    const MetadataObuConfig& config, std::string& error);

// --- Decoder-interop probe (NOT AV2 v1.0.0) --------------------------------
//
// The pinned dav2d snapshot (f9dafbc, 2026-11-10) parses OBU type 8 as the
// pre-1.0 AV1-style metadata OBU: leb128(metadata_type) immediately followed
// by the CLL/MDCV fields and trailing_bits, with no metadata_is_suffix /
// muh_layer_idc / muh_cancel_flag / muh_persistence_idc header. That layout is
// NOT §5.17.2 and must never be shipped in an AV2 stream. These helpers exist
// only so the decoder gate can prove the validator actively parses and
// validates HDR metadata rather than merely skipping it.
std::vector<std::uint8_t> make_legacy_av1_metadata_obu(
    MetadataType type, std::span<const std::uint8_t> unit, std::string& error);
std::vector<std::uint8_t> make_legacy_av1_hdr_metadata_obus(
    const HdrMetadata& metadata, std::string& error);

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

// One length-delimited OBU inside a temporal unit (Annex B.2 framing).
struct ObuView {
  std::uint8_t type = 0;
  std::uint8_t tlayer_id = 0;
  bool extension_flag = false;
  std::uint8_t mlayer_id = 0;
  std::uint8_t xlayer_id = 0;
  std::size_t offset = 0;       // offset of the leb128 size prefix
  std::size_t header_bytes = 0; // leb128 prefix + obu_header()
  std::size_t size = 0;         // total OBU bytes incl. prefix
  std::span<const std::uint8_t> payload;

  bool is_metadata() const {
    return type == kObuTypeMetadataShort || type == kObuTypeMetadataGroup;
  }
};

// Parses one leb128-prefixed OBU starting at `offset`; advances `offset` past
// it. Rejects truncated/overflowing leb128, size 0, and size beyond the buffer.
bool parse_obu(std::span<const std::uint8_t> data, std::size_t& offset,
               ObuView& out, std::string& error);

bool parse_obu_sequence(std::span<const std::uint8_t> data,
                        std::vector<ObuView>& out, std::string& error);

struct MetadataShortView {
  bool is_suffix = false;
  MetadataLayer layer = MetadataLayer::kUnspecified;
  MetadataPersistence persistence = MetadataPersistence::kGlobal;
  bool cancel = false;
  std::uint32_t type = 0;  // raw metadata_type; may be unknown to this module
  std::span<const std::uint8_t> unit;  // §5.17.1 payload (empty when cancelled)
};

// §5.17.2. `payload` is the OBU payload after obu_header().
bool parse_metadata_short(std::span<const std::uint8_t> payload,
                          MetadataShortView& out, std::string& error);

struct MetadataUnitView {
  std::uint32_t type = 0;
  MetadataLayer layer = MetadataLayer::kUnspecified;
  MetadataPersistence persistence = MetadataPersistence::kGlobal;
  std::uint8_t priority = 0;
  bool cancel = false;
  std::span<const std::uint8_t> unit;  // §5.17.1 payload
};

struct MetadataGroupView {
  bool is_suffix = false;
  MetadataNecessity necessity = MetadataNecessity::kUndefined;
  MetadataApplicationId application = MetadataApplicationId::kUnspecified;
  std::vector<MetadataUnitView> units;
};

// §5.17.3. `xlayer_id` is obu_xlayer_id, needed to size muh_xlayer_map when
// muh_layer_idc == LAYER_VALUES.
bool parse_metadata_group(std::span<const std::uint8_t> payload,
                          std::uint8_t xlayer_id, MetadataGroupView& out,
                          std::string& error);

// Typed unit decoders. Accept the canonical 4/24 bytes and any longer
// metadata_unit payload (§5.17.1 pads with metadata_unit_remaining_bit).
bool decode_cll(std::span<const std::uint8_t> unit, ContentLightLevel& out,
                std::string& error);
bool decode_mdcv(std::span<const std::uint8_t> unit,
                 MasteringDisplayColorVolume& out, std::string& error);

struct HdrMetadataRecord {
  std::size_t obu_offset = 0;  // offset of the metadata OBU size prefix
  bool in_group = false;
  bool is_suffix = false;
  bool before_first_frame = false;
  MetadataType type = MetadataType::kHdrCll;
  ContentLightLevel cll;                        // valid when type == kHdrCll
  MasteringDisplayColorVolume mdcv;             // valid when type == kHdrMdcv
};

struct HdrScanResult {
  std::vector<HdrMetadataRecord> records;  // CLL/MDCV units only, stream order
  std::size_t metadata_obus = 0;           // metadata short + group OBUs
  std::size_t metadata_units = 0;          // all units seen (incl. cancelled)
  std::size_t unknown_units = 0;           // metadata types not parsed here
  std::size_t invalid_units = 0;           // CLL/MDCV payload too short
};

// Walks a complete temporal unit. Structural OBU errors are failures; a unit
// with an invalid CLL/MDCV payload increments invalid_units and is skipped.
bool scan_hdr_metadata(std::span<const std::uint8_t> temporal_unit,
                       HdrScanResult& out, std::string& error);

// True for OBU types that carry coded frame data (§6.2.2 Table 6.1).
bool is_frame_obu_type(std::uint8_t type);

// ---------------------------------------------------------------------------
// Ordering
// ---------------------------------------------------------------------------

enum class MetadataPlacement : std::uint8_t {
  // Prefix metadata: insert immediately before the first frame OBU, i.e.
  // after the sequence header and any multi-frame-header/content-interpretation
  // OBUs. This is the §7.3.3 position: inside the coded output frame unit,
  // ahead of the coded frame.
  kBeforeFirstFrame = 0,
  // Suffix metadata: append after the last OBU of the temporal unit, i.e.
  // after the coded frame (§7.3.3).
  kEndOfTemporalUnit = 1,
};

// Inserts already-serialized metadata OBUs at `placement`. The metadata OBUs
// must be OBU_METADATA_SHORT/GROUP with metadata_is_suffix matching the
// placement. Fails if the temporal unit has no frame OBU or no sequence header
// before the frame. `inserted_at`, when non-null, receives the byte offset.
bool insert_metadata_obus(std::span<const std::uint8_t> temporal_unit,
                          std::span<const std::uint8_t> metadata_obus,
                          MetadataPlacement placement,
                          std::vector<std::uint8_t>& out, std::string& error,
                          std::size_t* inserted_at = nullptr);

}  // namespace av2enc::hdr
