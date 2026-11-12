#pragma once

// AV2 v1.0.0 random-access switch (SWITCH) and random-access switch frame
// (RAS) syntax, reference-set derivation and conformance checks.
//
// Scope (AV2 Bitstream & Decoding Process Specification v1.0.0, commit
// 38f28e7):
//   * §6.2 (Table 6.1): OBU_SWITCH = 10, OBU_RAS_FRAME = 21, both
//     layer-specific and both constrained to obu_tlayer_id == 0.
//   * §5.18.2 frame_header_info(): the SWITCH/RAS branch
//     (restricted_prediction_switch, FrameType = SWITCH_FRAME), long-term
//     frame ids, num_key_ref_frames / ref_long_term_id, forced
//     frame_size_override_flag = 1, forced explicitRefFrameMap = 1, derived
//     RAS refresh_frame_flags, and the SWITCH/RAS omissions
//     (signal_primary_ref_frame, disable_cross_frame_cdf_init,
//     skip_mode_present).
//   * §6.17 frame header OBU semantics: restricted_prediction_switch,
//     ref_long_term_id, and the RAS reference constraint
//     long_term_id_in_use(RefLongTermId[ref_frame_idx[i]]) == 1.
//   * §7.4.5 Random Access Switch: RAS is inter-predicted but may only
//     reference long-term reference frames named in ref_long_term_id; it does
//     not start a new CVS; reference slots not holding those long-term frames
//     are refreshed.
//
// This component is self-contained and writes no M2 core files. The frame
// header writer/reader below target the same fixed sequence-header profile as
// the M2 P-frame path (64x64 superblocks, frozen CDFs, no CDEF/GDF/LR/CCSO,
// no TIP/BRU/BAWP/OPFL/global motion) and add long_term_frame_id_bits.
//
// Namespace: av2enc::inter::random_access.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"

namespace av2enc::inter::random_access {

// ---------------------------------------------------------------------------
// Constants (§3)
// ---------------------------------------------------------------------------

inline constexpr std::uint8_t kNumRefFrames = 16;  // NUM_REF_FRAMES
inline constexpr std::uint8_t kRefsPerFrame = 7;   // REFS_PER_FRAME
inline constexpr std::uint8_t kMaxMlayerId = 7;    // max_mlayer_id field width
inline constexpr std::uint8_t kMaxTlayerId = 3;    // max_tlayer_id field width

// FrameType values (§6.17).
enum class FrameType : std::uint8_t {
  kKey = 0,
  kInter = 1,
  kIntraOnly = 2,
  kSwitch = 3,
};

// OBU types used by this component (Table 6.1). The numeric values are the
// normative obu_type values; the M0 writer takes its own enum but only writes
// the 5-bit field, so this enum is the source of truth for 10 and 21.
enum class ObuType : std::uint8_t {
  kSequenceHeader = 1,
  kTemporalDelimiter = 2,
  kClosedLoopKey = 4,
  kRegularTileGroup = 7,
  kSwitch = 10,
  kRas = 21,
};

// Frame-type derivation for the OBU types this component writes (§5.18.2).
struct FrameTypeInfo {
  ObuType obu_type = ObuType::kSwitch;
  bool is_regular = false;    // IsRegular
  bool key_frame = false;     // keyFrame
  FrameType frame_type = FrameType::kSwitch;
  bool frame_is_intra = false;
  // Syntax-field presence/derivation flags used by the writer/reader.
  bool codes_restricted_prediction_switch = false;
  bool codes_frame_is_inter = false;
  bool codes_num_key_ref_frames = false;
  bool forces_explicit_ref_map = false;
  bool forces_frame_size_override = false;
  bool derives_refresh_frame_flags = false;
  bool skips_primary_ref = false;
  bool resets_qm = false;
};

// §5.18.2 / Table 6.1 / §7.4.5. `restricted_prediction_switch` is the bit that
// is read for both SWITCH and RAS; it is ignored by the derivation for other
// OBU types.
FrameTypeInfo derive_frame_type(ObuType obu_type,
                                bool restricted_prediction_switch);

const char* frame_type_name(FrameType type);
const char* obu_type_name(ObuType type);

// ---------------------------------------------------------------------------
// Sequence header profile
// ---------------------------------------------------------------------------

// The subset of sequence_header_obu() (§5.4) that this component writes. All
// tool switches match the M2 CPU P-frame profile; long_term_frame_id_bits is
// the only field M2 currently hard-codes to zero that RAS needs.
struct SequenceConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // level 5.1
  std::uint8_t order_hint_bits = 4;
  std::uint8_t num_ref_frames = 2;
  std::uint8_t max_drl_bits_minus_1 = 4;
  std::uint8_t long_term_frame_id_bits = 0;  // 3-bit field, §5.4.6
  // Frame-level defaults used by this writer (base_q_idx and
  // disable_cdf_update live in the frame header, not the sequence header).
  std::uint8_t base_q_idx = 90;
  bool disable_cdf_update = true;
  bool enable_cfl_intra = true;
  // Compatibility quirk: the pinned dav2d build (f9dafbc, 2026-11-10) does not
  // implement the §5.18.2 RAS refresh_frame_flags derivation. Its frame-header
  // parser falls through to the generic SWITCH_FRAME branch and reads
  // NumRefFrames refresh bits for OBU_RAS_FRAME as well (the source carries a
  // "FIXME special cases for bridge and ras frames" comment). The default is
  // the normative behaviour (no bits, derive from RefState). The gate sets this
  // true so the pinned decoder and this writer agree; streams then also match
  // the derived mask the spec would produce.
  bool ras_codes_refresh_frame_flags = false;

  std::uint32_t frame_width_bits() const;
  std::uint32_t frame_height_bits() const;
};

// Reads back the fields needed to parse frame headers. `supported_profile`
// is false when a leading tool switch differs from the fixed profile above;
// the reader still returns the parsed fields but a caller should not rely on
// the conditional frame-header branches in that case.
struct SequenceHeaderView {
  std::uint16_t max_frame_width = 0;
  std::uint16_t max_frame_height = 0;
  std::uint8_t seq_level_idx = 0;
  std::uint8_t order_hint_bits = 0;
  std::uint8_t num_ref_frames = 0;
  std::uint8_t max_drl_bits_minus_1 = 0;
  std::uint8_t long_term_frame_id_bits = 0;
  bool monotonic_output_order_flag = true;
  bool explicit_ref_frame_map = false;
  bool explicit_num_ref_frames = false;
  bool enable_cfl_intra = false;
  bool seq_force_screen_content_tools = false;
  bool enable_tip = false;
  bool enable_bru = false;
  bool enable_bawp = false;
  bool enable_global_motion = false;
  bool enable_short_refresh_frame_flags = false;
  bool disable_cross_frame_cdf_init_supported = true;
  bool supported_profile = false;
};

// ---------------------------------------------------------------------------
// Frame header
// ---------------------------------------------------------------------------

// One frame_header_info() (§5.18.2) for OBU_CLOSED_LOOP_KEY, OBU_SWITCH or
// OBU_RAS_FRAME. The writer emits exactly the field order in the spec; fields
// not present for a type are ignored (and flagged by validate_header()).
struct FrameHeader {
  ObuType obu_type = ObuType::kSwitch;
  std::uint32_t order_hint = 0;
  // SWITCH and RAS only.
  bool restricted_prediction_switch = false;
  // KEY_FRAME only. LongTermId = long_term_id_plus_1 - 1; values 0..2^bits-1
  // (all-ones is reserved).
  std::uint8_t long_term_id_plus_1 = 0;
  // RAS only. num_key_ref_frames is 3 bits.
  std::uint8_t num_key_ref_frames = 0;
  std::array<std::uint8_t, kRefsPerFrame> ref_long_term_id{};
  // SWITCH/RAS explicit reference map.
  std::uint8_t num_total_refs = 0;
  std::array<std::uint8_t, kRefsPerFrame> ref_frame_idx{};
  // SWITCH explicit refresh mask. RAS derives it (§5.18.2); the writer ignores
  // this value unless SequenceConfig::ras_codes_refresh_frame_flags is set (the
  // pinned dav2d compatibility path). It must match
  // derive_ras_refresh_flags() to pass validation.
  std::uint16_t refresh_frame_flags = 0;
  bool immediate_output_frame = true;
  // Frame dimensions, 0 means "use max_frame_*". SWITCH always codes them
  // because frame_size_override_flag is forced to 1.
  std::uint16_t frame_width = 0;
  std::uint16_t frame_height = 0;
};

// ---------------------------------------------------------------------------
// Reference state and allowed sets
// ---------------------------------------------------------------------------

struct RefSlot {
  bool valid = false;
  // §6.17 long-term id of the frame held in the slot; -1 when the slot holds a
  // short-term frame or was never refreshed.
  std::int32_t long_term_id = -1;
  std::uint16_t width = 0;
  std::uint16_t height = 0;
  // Set when a SWITCH with restricted_prediction_switch = 1 marked the slot.
  bool restricted = false;
};

struct RefState {
  std::array<RefSlot, kNumRefFrames> slots{};
};

// §6.17: long_term_id_in_use(longTermId) scans num_key_ref_frames entries.
bool long_term_id_in_use(std::int32_t long_term_id,
                         std::span<const std::uint8_t> ref_long_term_ids);

// Slots a frame of `obu_type` may legally reference.
//   * SWITCH: every valid slot (switch frames may reference any valid frame;
//     restricted_prediction_switch only restricts what future frames may use).
//   * RAS: valid slots whose RefLongTermId appears in ref_long_term_ids.
//   * KEY: empty.
std::uint16_t allowed_reference_mask(
    ObuType obu_type, const RefState& refs,
    std::span<const std::uint8_t> ref_long_term_ids);

// §5.18.2 RAS refresh derivation:
//   refresh_frame_flags = OR over i < NumRefFrames of (1 << i) when
//   !RefValid[i] || !long_term_id_in_use(RefLongTermId[i]).
std::uint16_t derive_ras_refresh_flags(
    const RefState& refs, std::span<const std::uint8_t> ref_long_term_ids,
    std::uint8_t num_ref_frames = kNumRefFrames);

// Applies the reference-frame update implied by a decoded frame to `refs`.
// `tile_is_intra` is informational (reference state does not change). The
// function is only used by the gate/tests to build the next reference state.
void update_reference_state(const SequenceConfig& config, const FrameHeader& header,
                            std::uint16_t actual_refresh_flags, RefState& refs);

// ---------------------------------------------------------------------------
// Conformance
// ---------------------------------------------------------------------------

enum class ViolationCode : std::uint8_t {
  kUnsupportedObuType,
  kTlayerNonZero,
  kBadLongTermIdBits,
  kKeyLongTermIdReserved,
  kBadRefLongTermId,
  kTooManyKeyRefFrames,
  kRasRefNotLongTerm,
  kRasWithoutLongTermIds,
  kRasOrderHintOutOfRange,
  kRasRefreshMismatch,
  kBadNumTotalRefs,
  kBadRefFrameIdx,
  kInvalidReference,
  kReferenceDimensions,
  kSwitchRestrictedOnly,
  kReservedFieldSet,
};

struct Violation {
  ViolationCode code = ViolationCode::kUnsupportedObuType;
  std::string message;
};

struct ValidationResult {
  bool ok = true;
  std::vector<Violation> violations;
};

// Checks the header against §5.18.2, §6.17 and §7.4.5. `tlayer_id` is the
// obu_tlayer_id from the OBU header; SWITCH and RAS require 0 (Table 6.1).
ValidationResult validate_header(const SequenceConfig& config,
                                 const FrameHeader& header, const RefState& refs,
                                 std::uint8_t tlayer_id = 0);

// ---------------------------------------------------------------------------
// Header writers
// ---------------------------------------------------------------------------

// Writes the sequence_header_obu() syntax bits for the fixed M2 profile.
// Returns the syntax-bit count (without the OBU extension flag, trailing bits,
// OBU header or leb128 framing).
std::size_t write_sequence_header(BitWriter& writer, const SequenceConfig& config);

// Annex B framed OBUs, byte-identical in layout to
// av2enc::basic::make_obu()/av2enc::video::make_sequence_header_obu().
std::vector<std::uint8_t> make_sequence_header_obu(const SequenceConfig& config);
std::vector<std::uint8_t> make_temporal_delimiter_obu();

// Writes frame_header_info() (§5.18.2) without the leading is_first_tile_group
// bit and without byte_alignment(). Returns the number of bits written.
std::size_t write_frame_header_info(BitWriter& writer, const SequenceConfig& config,
                                    const FrameHeader& header);

// §5.19 tile_group_obu() header: is_first_tile_group f(1),
// frame_header(isFirst = 1), byte_alignment(). Returns bits written.
std::size_t write_tile_group_header(BitWriter& writer, const SequenceConfig& config,
                                    const FrameHeader& header);

// Complete OBU of the type selected by header.obu_type, including the tile
// payload (which must end with the §8.2.4 exit_symbol trailing bits).
std::vector<std::uint8_t> make_frame_obu(const SequenceConfig& config,
                                         const FrameHeader& header,
                                         std::span<const std::uint8_t> tile_payload);

// ---------------------------------------------------------------------------
// Header readers
// ---------------------------------------------------------------------------

struct ObuView {
  // Raw 5-bit obu_type (Table 6.1). Kept as a byte so the same parser can walk
  // OBU_TEMPORAL_DELIMITER / OBU_SEQUENCE_HEADER when splicing a unit.
  std::uint8_t obu_type = 0;
  std::uint8_t tlayer_id = 0;
  std::uint8_t mlayer_id = 0;
  std::uint8_t xlayer_id = 0;
  std::span<const std::uint8_t> payload;
};

// Parses an Annex B length-delimited OBU (§5.2.1/§5.2.2, Annex B.2).
// `consumed` receives the number of bytes read including the leb128 prefix.
bool parse_obu(std::span<const std::uint8_t> bytes, ObuView& out,
               std::size_t& consumed, std::string& error);

// Reads sequence_header_obu() payload bits (without the OBU extension flag or
// trailing bits). Returns false with an error on truncation/unsupported bits.
bool read_sequence_header(std::span<const std::uint8_t> payload,
                          SequenceHeaderView& out, std::string& error);

// Reads one tile_group_obu() header from `payload` (the OBU payload, starting
// at is_first_tile_group) for a known OBU type. `bits_consumed` ends after
// byte_alignment(), i.e. at the first tile payload byte. On success the
// derived fields (refresh_frame_flags for RAS, frame dimensions) are filled.
// `ras_codes_refresh_frame_flags` mirrors
// SequenceConfig::ras_codes_refresh_frame_flags and must match the writer.
bool read_frame_header(std::span<const std::uint8_t> payload, ObuType obu_type,
                       const SequenceHeaderView& sequence, FrameHeader& out,
                       std::size_t& bits_consumed, std::string& error,
                       bool ras_codes_refresh_frame_flags = false);

// ---------------------------------------------------------------------------
// Encoder-side strategy: reuse a P-frame payload or re-encode
// ---------------------------------------------------------------------------

struct PFrameCandidate {
  // Reference slot the existing P tile payload predicts from (M2 uses
  // ref_frame_idx[0]).
  std::uint8_t reference_slot = 0;
  // Refresh mask the existing P header would have signalled.
  std::uint16_t refresh_frame_flags = 0;
  std::uint32_t order_hint = 0;
  // Informational: true when the payload codes at least one inter block. The
  // planner does not require inter blocks (an all-intra payload is legal in a
  // SWITCH_FRAME), but it reports the reuse path in its reason string.
  bool has_inter_blocks = false;
};

enum class PayloadDecision : std::uint8_t {
  kReusePFrame,
  kReencodeRequired,
};

struct PayloadPlan {
  PayloadDecision decision = PayloadDecision::kReencodeRequired;
  FrameHeader header;
  std::string reason;
};

// SWITCH: legal whenever the candidate's reference slot is valid. The header
// keeps the P payload's reference and order hint; the caller chooses whether
// restricted_prediction_switch is set.
PayloadPlan plan_switch_frame(const SequenceConfig& config, const RefState& refs,
                              const PFrameCandidate& candidate,
                              bool restricted_prediction_switch);

// RAS: reuses the candidate when its reference slot holds a long-term frame
// named in ras_long_term_ids. The RAS always declares that reference
// (num_total_refs = 1); the pinned dav2d build rejects NumTotalRefs == 0 with
// "Error parsing frame header", and §6.17's reference restriction requires the
// slot to be one of the named long-term frames. When no such slot exists the
// caller must re-encode against a long-term slot, or set
// long_term_frame_id_bits > 0 first.
PayloadPlan plan_ras_frame(const SequenceConfig& config, const RefState& refs,
                           const PFrameCandidate& candidate,
                           std::span<const std::uint8_t> ras_long_term_ids);

// ---------------------------------------------------------------------------
// Helpers shared with the gate
// ---------------------------------------------------------------------------

std::uint32_t ceil_log2(std::uint32_t value);
std::uint32_t frame_width_bits(const SequenceConfig& config);
std::uint32_t frame_height_bits(const SequenceConfig& config);

}  // namespace av2enc::inter::random_access
