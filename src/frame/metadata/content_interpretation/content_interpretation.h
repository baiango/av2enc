#pragma once

// AV2 v1.0.0 content interpretation OBU.
//
// Normative reference: spec/AV2_Spec_v1.0.0.pdf (commit 38f28e7). Section
// numbers below are from that document.
//
//   §5.2.1  open_bitstream_unit(): extensible-OBU payload, obu_extension_flag,
//           trailing_bits for OBU_CONTENT_INTERPRETATION
//   §5.2.2  obu_header(): obu_header_extension_flag, obu_type, obu_tlayer_id
//   §5.2.3  trailing_bits()
//   §5.4.12 timing_info(): f(32) num_units_in_display_tick, f(32) time_scale,
//           f(1) equal_picture_interval, uvlc() num_ticks_per_picture_minus_1
//   §5.15   content_interpretation_obu() and the Aspect_Ratio_*[17] tables
//   §6.2.2  Table 6.1: OBU_CONTENT_INTERPRETATION = 24
//   §6.4.12 timing_info() semantics (value ranges, per-CVS constancy)
//   §6.14   content interpretation OBU semantics (scan type, colour,
//           chroma sample position tables, aspect ratio, per-CVS rules)
//   §7.3.3  coded output frame unit: CI OBU is the first OBU of the unit
//   §7.3.6  coded extended layer unit (sequence header before the frame unit)
//   §7.3.7  temporal unit order
//   §7.3.8.10 Content interpretation OBU availability
//   §7.3.8.11 Content interpretation parameters initialization
//   §4.11.3 uvlc()
//   §4.11.10 rg(n) (used for ci_color_description_idc)
//
// Scope
// -----
// * Letten/field scan type, colour description (idc + explicit primaries /
//   transfer / matrix / full range), chroma sample position top/bottom with
//   the 4:2:0 and 4:2:2 interpretation tables, aspect ratio (table idc or
//   explicit SAR) and timing_info().
// * Writers emit the canonical form: extension flag 0, no extension data.
// * Readers are structural and, by default, also enforce the normative
//   conformance requirements (chroma position <= 5, aspect idc <= 16 or 255,
//   non-zero timing values). `strict == false` disables only those semantic
//   checks; framing errors are always rejected.
// * ci_reserved_2bit is written as 0. A reader records non-zero values but
//   accepts them, because §6.14 says the value "shall be ignored by a
//   decoder".
//
// §7.3.8.11 also lists `ci_extension_present_flag = 0` among the default
// parameter values even though §5.15 has no such syntax element. That is a
// spec text inconsistency; this module treats the payload extension as
// §5.2.1 defines it (obu_extension_flag / trailing_bits) and does not invent
// an extension flag.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace av2enc::ci {

// §6.2.2 Table 6.1.
inline constexpr std::uint8_t kObuTypeContentInterpretation = 24;

// Fixed part of content_interpretation_obu(): 2 + 1 + 1 + 1 + 1 + 2 bits.
inline constexpr std::size_t kContentInterpretationHeaderBits = 8;

// §5.15 Aspect_Ratio_Width / Aspect_Ratio_Height, index 0..16.
inline constexpr std::array<std::uint32_t, 17> kAspectRatioWidth{
    0, 1, 12, 10, 16, 40, 24, 20, 32, 80, 18, 15, 64, 160, 4, 3, 2};
inline constexpr std::array<std::uint32_t, 17> kAspectRatioHeight{
    0, 1, 11, 11, 11, 33, 11, 11, 11, 33, 11, 11, 33, 99, 3, 2, 1};

// §5.15 ci_scan_type_idc.
enum class ScanType : std::uint8_t {
  kUnspecified = 0,
  kProgressive = 1,                 // progressive frame picture samples
  kInterlaceField = 2,              // interlace field picture samples
  kInterlaceComplementaryPair = 3,  // complementary field-pair samples
};

// §6.14 chroma sample position. CSP_UNSPECIFIED (6) is a *derived* value for
// "not present"; it is not a conforming value when the syntax element is
// actually coded (the conformance limit is <= 5).
enum class ChromaSamplePosition : std::uint8_t {
  kLeft = 0,
  kCenter = 1,
  kTopLeft = 2,
  kTop = 3,
  kBottomLeft = 4,
  kBottom = 5,
  kUnspecified = 6,
};

// The 4:2:0 / 4:2:2 interpretation differs only in which positions are valid
// (CSP_TOPLEFT..CSP_BOTTOM are N/A for 4:2:2) and in the vertical offset.
enum class ChromaFormat : std::uint8_t {
  k420 = 0,
  k422 = 1,
};

// One row of the §6.14 chroma sample position tables. Offsets are in luma
// sample units from the (0,0) luma sample.
struct ChromaSampleOffset {
  bool applicable = false;
  double horizontal = 0.0;
  double vertical = 0.0;
};

ChromaSampleOffset chroma_sample_offset(ChromaSamplePosition position,
                                        ChromaFormat format);
const char* chroma_sample_position_name(ChromaSamplePosition position);
const char* scan_type_name(ScanType scan_type);

// §6.14 Table 6.13: ci_color_description_idc.
enum class ColorDescriptionIdc : std::uint8_t {
  kExplicit = 0,
  kBt709Sdr = 1,
  kBt2100Pq = 2,
  kBt2100Hlg = 3,
  kSrgb = 4,
  kSycc = 5,
};

// §5.15 colour description. When `idc` is non-zero it determines primaries,
// transfer and matrix; the explicit fields are then ignored by the syntax and
// carry the derived values in this struct.
struct ColorDescription {
  std::uint8_t idc = 0;  // ci_color_description_idc (rg(2), 0..127)
  std::uint8_t primaries = 2;  // CP_UNSPECIFIED
  std::uint8_t transfer = 2;   // TC_UNSPECIFIED
  std::uint8_t matrix = 2;     // MC_UNSPECIFIED
  bool full_range = false;

  friend bool operator==(const ColorDescription&,
                         const ColorDescription&) = default;
};

// Values of {primaries, transfer, matrix} implied by Table 6.13. Returns the
// input's explicit fields when idc == 0 and the unspecified triple for
// reserved idc values 6..127.
ColorDescription derive_color_description(std::uint8_t idc);

// §5.15 aspect ratio. When `idc != 255`, sar_width/height are the table
// values. `idc == 0` means the SAR is unspecified (table row 0 is 0:0).
struct AspectRatio {
  std::uint8_t idc = 0;
  std::uint32_t sar_width = 0;
  std::uint32_t sar_height = 0;

  std::uint32_t resolved_width() const;
  std::uint32_t resolved_height() const;
  // True when resolved_width/height are both non-zero.
  bool resolved() const;

  friend bool operator==(const AspectRatio&, const AspectRatio&) = default;
};

// §5.4.12 timing_info().
struct TimingInfo {
  std::uint32_t num_units_in_display_tick = 0;
  std::uint32_t time_scale = 0;
  bool equal_picture_interval = false;
  std::uint32_t num_ticks_per_picture_minus_1 = 0;

  friend bool operator==(const TimingInfo&, const TimingInfo&) = default;
};

// Complete content interpretation parameter set for one embedded layer.
struct ContentInterpretation {
  ScanType scan_type = ScanType::kUnspecified;  // ci_scan_type_idc
  bool has_color_description = false;
  ColorDescription color;
  bool has_chroma_sample_position = false;
  ChromaSamplePosition chroma_top = ChromaSamplePosition::kUnspecified;
  ChromaSamplePosition chroma_bottom = ChromaSamplePosition::kUnspecified;
  bool has_aspect_ratio = false;
  AspectRatio aspect;
  bool has_timing_info = false;
  TimingInfo timing;

  friend bool operator==(const ContentInterpretation&,
                         const ContentInterpretation&) = default;
};

// Default parameter values from §7.3.8.11.
ContentInterpretation default_content_interpretation();

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

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
  bool clean() const { return issues_.empty(); }
  bool ok() const { return !has_errors(); }
  std::string summary() const;  // newline-joined "error: ..."/"warning: ..."

 private:
  std::vector<ValidationIssue> issues_;
};

// Checks the parameter set against §6.14 / §6.4.12. `format` selects the
// chroma sample position table so an out-of-table position (CSP_TOPLEFT.. for
// 4:2:2) is reported. With `strict == false` reserved codepoints and
// "ignored field" mismatches are warnings; with `strict == true` they become
// errors. Hard conformance violations (chroma position > 5, aspect idc in
// 17..254, zero timing values, ticks out of range) are always errors.
ValidationResult validate_content_interpretation(
    const ContentInterpretation& params, ChromaFormat format = ChromaFormat::k420,
    bool strict = false);

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

// Serializes content_interpretation_obu() syntax plus the §5.2.1 extensible-
// OBU obu_extension_flag = 0 and trailing_bits(). This is the OBU payload
// (everything after obu_header()). On failure returns an empty vector and sets
// `error` to the first validation error.
std::vector<std::uint8_t> serialize_content_interpretation_payload(
    const ContentInterpretation& params, std::string& error,
    ChromaFormat format = ChromaFormat::k420, bool strict = false);

// Builds one Annex B.2 length-delimited OBU_CONTENT_INTERPRETATION:
//
//   leb128(obu_size) obu_header() payload
//
// obu_size counts the OBU header and payload but not the leb128 prefix. The
// OBU is written with obu_header_extension_flag = 0 and obu_tlayer_id = 0.
std::vector<std::uint8_t> make_content_interpretation_obu(
    const ContentInterpretation& params, std::string& error,
    std::size_t* payload_bits = nullptr, ChromaFormat format = ChromaFormat::k420,
    bool strict = false);

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
  std::size_t offset = 0;        // offset of the leb128 size prefix
  std::size_t header_bytes = 0;  // leb128 prefix + obu_header()
  std::size_t size = 0;          // total OBU bytes including the prefix
  std::span<const std::uint8_t> payload;  // after obu_header()

  bool is_content_interpretation() const {
    return type == kObuTypeContentInterpretation;
  }
};

// Parses one leb128-prefixed OBU starting at `offset`; advances `offset` past
// it. Rejects truncated/overflowing leb128, size 0, and size beyond the buffer.
bool parse_obu(std::span<const std::uint8_t> data, std::size_t& offset,
               ObuView& out, std::string& error);

bool parse_obu_sequence(std::span<const std::uint8_t> data,
                        std::vector<ObuView>& out, std::string& error);

struct ContentInterpretationView {
  ContentInterpretation params;
  std::uint8_t reserved_bits = 0;  // ci_reserved_2bit, informational
  // Number of payload bits used by content_interpretation_obu() itself
  // (before the §5.2.1 extension flag and trailing_bits).
  std::size_t syntax_bits = 0;
  // True when the scan type made chroma_bottom explicit; false when
  // progressive and bottom was copied from top.
  bool chroma_bottom_signalled = false;
};

// Parses the OBU payload (everything after obu_header()) of a
// OBU_CONTENT_INTERPRETATION. Enforces §5.2.1 framing (extension flag 0, one
// trailing 1 bit, zero padding) and, when `strict`, the hard §6.14/§6.4.12
// conformance limits. Reserved colour codepoints remain warnings because
// §6.14 requires decoders to ignore them.
bool parse_content_interpretation(
    std::span<const std::uint8_t> payload, ContentInterpretationView& out,
    std::string& error, bool strict = true,
    ChromaFormat format = ChromaFormat::k420);

struct ContentInterpretationRecord {
  std::size_t obu_offset = 0;      // offset of the leb128 size prefix
  std::size_t obu_size = 0;        // total OBU bytes including the prefix
  bool before_first_frame = false; // in the coded output frame unit prefix
  bool valid = false;              // payload parsed and passed conformance
  ContentInterpretationView view;
};

struct ContentInterpretationScanResult {
  std::vector<ContentInterpretationRecord> records;  // stream order
  std::size_t ci_obus = 0;          // OBU type 24 instances
  // OBUs that are misordered (after the first frame), duplicated within one
  // temporal unit, or fail the §5.15/§6.14 checks.
  std::size_t invalid_ci_obus = 0;
  std::size_t frame_obus = 0;
  bool has_sequence_header = false;
  // Byte offset of the first frame OBU in the temporal unit (0 when absent).
  std::size_t first_frame_offset = 0;
  bool has_first_frame = false;
};

// Walks a complete temporal unit. Structural OBU errors (bad leb128, truncated
// payload) are failures; a structurally valid CI OBU with an invalid payload
// increments invalid_ci_obus and is skipped. An invalid CI OBU does not make
// the whole scan fail, so a tool can report every instance.
bool scan_content_interpretation(std::span<const std::uint8_t> temporal_unit,
                                 ContentInterpretationScanResult& out,
                                 std::string& error, bool strict = true,
                                 ChromaFormat format = ChromaFormat::k420);

// True for OBU types that carry coded frame data (§6.2.2 Table 6.1).
bool is_frame_obu_type(std::uint8_t type);

// True for OBU types that precede the coded output frame unit inside a coded
// extended layer unit or temporal unit: TD, sequence header, MSDO, LCR, atlas
// segment, operating point set (§7.3.6, §7.3.7).
bool is_tu_prefix_obu_type(std::uint8_t type);

// ---------------------------------------------------------------------------
// Ordering
// ---------------------------------------------------------------------------

// Inserts an already-serialized CI OBU at the §7.3.3 position: the first OBU of
// the coded output frame unit, i.e. immediately after the last temporal-unit /
// extended-layer prefix OBU (temporal delimiter, sequence header, LCR, OPS,
// atlas segment, MSDO) and before any multi-frame-header, prefix metadata or
// frame OBU.
//
// Fails when the temporal unit has no sequence header, no frame OBU, or more
// than one sequence header (multiple coded extended layer units make a single
// insertion point ambiguous). `inserted_at`, when non-null, receives the byte
// offset of the inserted OBU.
bool insert_content_interpretation(
    std::span<const std::uint8_t> temporal_unit,
    std::span<const std::uint8_t> ci_obu, std::vector<std::uint8_t>& out,
    std::string& error, std::size_t* inserted_at = nullptr);

// ---------------------------------------------------------------------------
// Decoder-interop probe (NOT conforming)
// ---------------------------------------------------------------------------
//
// The pinned dav2d snapshot (f9dafbc, 2026-11-10) may or may not consume the
// content interpretation OBU. To distinguish "parsed and validated" from
// "skipped as an unknown OBU", run_ci_gate.sh feeds the decoder an OBU that
// violates a hard §6.14 rule: ci_chroma_sample_position_top = 6
// (CSP_UNSPECIFIED) while the syntax element is present. A decoder that
// parses content_interpretation_obu() and enforces the `<= 5` requirement
// should abort under --strict 1; a decoder that ignores this OBU decodes it
// unchanged.
//
// This OBU must never appear in a shipped stream. It exists only for the
// decoder gate, mirroring the legacy-AV1 probe in src/encoder/hdr/.
std::vector<std::uint8_t> make_nonconforming_chroma_probe_obu(
    std::string& error);

// ---------------------------------------------------------------------------
// Per-CVS constancy (§6.14, §7.3.8.10, §7.3.8.11)
// ---------------------------------------------------------------------------

// One temporal unit's CI observation for one embedded layer, in decode order.
struct CiObservation {
  std::uint8_t mlayer = 0;
  // First temporal unit of the coded video sequence for this layer.
  bool is_first_tu_of_cvs = false;
  // The temporal unit contains a CLK or OLK for this layer (random access
  // point for this layer).
  bool has_random_access = false;
  // nullptr means no CI OBU is present in this temporal unit.
  const ContentInterpretation* params = nullptr;
};

// Streaming tracker for one coded video sequence. Call observe() once per
// temporal unit (decode order), then reset() at a new CVS.
//
// State is kept per embedded layer; timing_info() is shared across layers.
//
// Enforced rules:
//  * all CI instances for a layer in a CVS carry identical parameters;
//  * if a CI OBU is present in any temporal unit, one is also present in the
//    first temporal unit of the CVS for that layer, with the same contents;
//  * a CI OBU in a non-random-access temporal unit matches the parameters
//    established at the most recent random access point;
//  * timing_info() is identical across all embedded layers in the CVS.
class CvsConstancyTracker {
 public:
  void reset();

  // Returns false and sets `error` on a violated rule.
  bool observe(const CiObservation& observation, std::string& error);

  // Per-layer state, default-constructed when `mlayer` is unknown.
  bool established(std::uint8_t mlayer) const;
  ContentInterpretation params(std::uint8_t mlayer) const;
  bool timing_established() const { return timing_established_; }
  const TimingInfo& timing() const { return timing_; }

 private:
  struct LayerState {
    bool established = false;
    ContentInterpretation params;
    bool first_tu_seen = false;
    bool first_tu_had_ci = false;
  };

  std::map<std::uint8_t, LayerState> layers_;
  bool timing_established_ = false;
  TimingInfo timing_;
};

// Convenience bulk check for one coded video sequence. `observations` must be
// in decode order; layers may be interleaved.
ValidationResult check_cvs_constancy(std::span<const CiObservation> observations);

}  // namespace av2enc::ci
