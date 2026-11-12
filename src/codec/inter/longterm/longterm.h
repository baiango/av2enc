// Long-term reference frames, open-loop keys and random-access bookkeeping.
//
// Normative reference: AV2 v1.0.0 (2026-05-28, commit 38f28e7).
//   §5.4.6  sequence_inter_config(): long_term_frame_id_bits, NumRefFrames
//   §5.18.2 frame_header_info(): long_term_id_plus_1, num_key_ref_frames,
//           ref_long_term_id[], refresh_frame_flags, immediate/implicit output
//   §6.4.6  long_term_frame_id_bits / num_ref_frames_minus_1 semantics
//   §6.17   frame header OBU semantics (num_key_ref_frames, ref_long_term_id,
//           refresh_frame_flags, immediate_output_frame)
//   §7.3.9  availability of long-term reference frames
//   §7.4.2  random access with/without long-term reference frames
//   §7.4.3  closed random access (CLK)
//   §7.4.4  open random access (OLK)
//   §7.23   reference frame update process (RefValid / RefLongTermId)
//
// This module owns exactly three things:
//
//   1. the exact syntax bits for the long-term-reference fields of the
//      sequence and frame headers, plus a matching reader;
//   2. a decoder-side reference-pool state machine that mirrors the normative
//      RefValid / RefLongTermId / OlkEncountered / OlkRefresh variables, and
//      the conformance rules of §7.4.3/§7.4.4/§7.23 on top of it;
//   3. a stream assembler that wraps existing M2 tile payloads in the new
//      headers (the M2 core is untouched).
//
// The M2 path always codes one reference slot and one intra keyframe; this
// module adds the OBU_OPEN_LOOP_KEY / long-term-reference layer above it.
#ifndef AV2ENC_ENCODER_INTER_LONGTERM_LONGTERM_H_
#define AV2ENC_ENCODER_INTER_LONGTERM_LONGTERM_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"

namespace av2enc::inter::longterm {

// NUM_REF_FRAMES (§7.23 loops over 0..NUM_REF_FRAMES-1).
inline constexpr int kNumRefFrames = 16;
// REFS_PER_FRAME: number of reference list entries per inter frame.
inline constexpr int kRefsPerFrame = 7;

// ---------------------------------------------------------------------------
// §5.18.2 codes, for OBU_OPEN_LOOP_KEY with long_term_frame_id_bits != 0:
//
//   num_key_ref_frames   f(3)
//   ref_long_term_id[i]  f(long_term_frame_id_bits)
//
// The pinned validator (`dav2d` f9dafbc, 2026-11-10) does *not* read those
// fields for an OLK: any insertion point in the OLK header produces
// "Error parsing frame header", while the header without them decodes. The
// gate therefore needs an explicit layout switch; the default stays the
// v1.0.0 syntax.
enum class LongTermListSyntax : std::uint8_t {
  kSpecV1 = 0,           // AV2 v1.0.0 §5.18.2
  kOmitPinnedDav2d = 1,  // no num_key_ref_frames/ref_long_term_id for an OLK
};

// §5.4.6 sequence header fields owned by this module
// ---------------------------------------------------------------------------
struct SequenceConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // level 5.1
  std::uint8_t base_q_idx = 90;

  // §5.4.6 order_hint_bits_minus_1 + 1. OrderHint is coded with this width.
  std::uint8_t order_hint_bits = 4;
  // §5.4.6 explicit_num_ref_frames / num_ref_frames_minus_1 (1..16).
  std::uint8_t num_ref_frames = 4;
  // §5.4.6 long_term_frame_id_bits f(3): 0 disables long-term references.
  std::uint8_t long_term_frame_id_bits = 2;
  // §5.4.6 seq_max_drl_bits_minus_1: 4 -> m = 5 DRL indices are codable.
  std::uint8_t max_drl_bits_minus_1 = 4;

  bool disable_cdf_update = true;
  bool enable_cfl_intra = true;
  // Layout of the OLK long-term list (§5.18.2). See LongTermListSyntax.
  LongTermListSyntax olk_long_term_list = LongTermListSyntax::kSpecV1;

  // Tool switches this module always writes (frozen, matching M2).
  bool explicit_ref_frame_map = true;
  bool explicit_num_ref_frames = true;
  bool enable_short_refresh_frame_flags = false;
  bool monotonic_output_order_flag = true;
  // §5.4.1 layer maxima. The layer-dependency syntax required when either
  // maximum is non-zero (seq_max_mlayer_cnt_minus_1, mlayer_dependency_map,
  // tlayer_dependency_map) is not implemented by this writer, so
  // validate_sequence_header_config() rejects non-zero values instead of
  // emitting a header that omits mandatory bits.
  std::uint8_t max_tlayer_id = 0;
  std::uint8_t max_mlayer_id = 0;
};

// Small helpers shared by the writer, the reader and the pool.
unsigned ceil_log2(std::uint32_t value);
// Largest legal RefLongTermId: (1 << long_term_frame_id_bits) - 2.
// Returns -1 when long_term_frame_id_bits == 0 (no long-term references).
int max_long_term_id(const SequenceConfig& config);
// long_term_id is coded as long_term_id_plus_1 = long_term_id + 1.
int max_long_term_id_plus_1(const SequenceConfig& config);

// ---------------------------------------------------------------------------
// §5.18.2 frame header: the subset that decides random-access behaviour
// ---------------------------------------------------------------------------
// The enumerator values are the §6.2.2 obu_type values (Table 6.1), so a role
// can be written straight into the OBU header.
enum class FrameRole : std::uint8_t {
  kClosedLoopKey = 4,  // OBU_CLOSED_LOOP_KEY
  kOpenLoopKey = 5,    // OBU_OPEN_LOOP_KEY
  kLeading = 6,        // OBU_LEADING_TILE_GROUP (IsRegular == 0)
  kRegular = 7,        // OBU_REGULAR_TILE_GROUP (IsRegular == 1)
};

constexpr bool is_key_role(FrameRole role) {
  return role == FrameRole::kClosedLoopKey || role == FrameRole::kOpenLoopKey;
}
// IsRegular (§5.18.2): OLK and REGULAR_TILE_GROUP are regular; LEADING_* are not.
constexpr bool is_regular_role(FrameRole role) {
  return role == FrameRole::kOpenLoopKey || role == FrameRole::kRegular;
}
constexpr bool is_random_access_role(FrameRole role) {
  return role == FrameRole::kClosedLoopKey || role == FrameRole::kOpenLoopKey;
}
constexpr std::uint8_t obu_type_of(FrameRole role) {
  return static_cast<std::uint8_t>(role);
}

struct FrameHeader {
  FrameRole role = FrameRole::kRegular;

  // frame_is_inter f(1). Absent (FrameIsIntra) for CLK/OLK. For a regular or
  // leading frame, false selects INTRA_ONLY_FRAME.
  bool frame_is_inter = true;

  // order_hint f(OrderHintBits).
  std::uint32_t order_hint = 0;

  // long_term_id_plus_1 - 1 for KEY_FRAME (CLK/OLK); -1 = "not a long-term
  // reference". Absent for inter frames, which infer LongTermId = -1.
  std::int32_t long_term_id = -1;

  // refresh_frame_flags. For CLK with max_mlayer_id == 0 the syntax infers
  // allFrames and no bits are coded; effective_refresh_flags() returns it.
  std::uint32_t refresh_frame_flags = 0;

  // Inter frames only (§5.18.2 explicit ref frame map).
  std::uint8_t num_total_refs = 0;
  std::vector<std::uint8_t> ref_frame_idx;

  // OBU_OPEN_LOOP_KEY only: num_key_ref_frames f(3) then
  // ref_long_term_id[i] f(long_term_frame_id_bits).
  std::vector<std::uint8_t> ref_long_term_id;

  // immediate_output_frame. Coded for CLK and inter frames; inferred 0 for
  // OLK (§5.18.2). implicit_output_frame is 0 because
  // monotonic_output_order_flag == 1.
  bool immediate_output_frame = true;
};

// ---------------------------------------------------------------------------
// Header serialisation (§5.4.6, §5.18.2, §5.19)
// ---------------------------------------------------------------------------
// sequence_header_obu() payload bits for single_picture_header_flag = 0.
// Returns 0 without writing when validate_sequence_header_config() rejects the
// config.
std::size_t write_sequence_header(BitWriter& writer,
                                  const SequenceConfig& config);
// Rejects configs this writer cannot serialise exactly.  max_tlayer_id > 0 or
// max_mlayer_id > 0 would require the §5.4.1 layer-dependency syntax, which is
// not implemented here.
bool validate_sequence_header_config(const SequenceConfig& config,
                                     std::string* error);
std::vector<std::uint8_t> make_sequence_header_obu(const SequenceConfig&);
std::vector<std::uint8_t> make_temporal_delimiter_obu();

// frame_header_info() bits, without is_first_tile_group and without
// byte_alignment(). Returns the number of bits written.
std::size_t write_frame_header(BitWriter& writer, const SequenceConfig& config,
                               const FrameHeader& header);

// §5.19 tile_group_obu(): is_first_tile_group f(1), frame_header(), then
// byte_alignment(). The tile payload follows the returned bytes.
std::size_t write_tile_group_header(BitWriter& writer,
                                    const SequenceConfig& config,
                                    const FrameHeader& header);

// Complete tile-group OBU (CLK / OLK / leading / regular) including the tile
// payload. The payload must already end with §8.2.4 exit_symbol() bits.
std::vector<std::uint8_t> make_frame_obu(
    const SequenceConfig& config, const FrameHeader& header,
    std::span<const std::uint8_t> tile_payload);

// ---------------------------------------------------------------------------
// Reader (round-trip / bit-exactness checks)
// ---------------------------------------------------------------------------
struct ObuView {
  std::uint8_t type = 0;
  std::uint8_t tlayer_id = 0;
  std::span<const std::uint8_t> payload;
};

// Annex B.2 [leb128][obu_header()][payload]. `consumed` counts the whole OBU.
bool parse_obu(std::span<const std::uint8_t> bytes, ObuView* view,
               std::size_t* consumed, std::string* error);

struct ParsedFrameHeader {
  FrameHeader header;
  // Bits consumed by is_first_tile_group + frame_header_info() + alignment.
  std::size_t header_bits = 0;
  // Byte offset of the tile payload inside the tile_group payload.
  std::size_t tile_offset = 0;
};

// Parses a tile_group_obu() payload (the bytes after obu_header()).
bool read_tile_group_header(std::span<const std::uint8_t> payload,
                            const SequenceConfig& config,
                            ParsedFrameHeader* out, std::string* error);

// Parses the sequence header payload written by write_sequence_header() and
// checks it against `config`. Returns the number of bits consumed.
bool read_sequence_header(std::span<const std::uint8_t> payload,
                          const SequenceConfig& config, std::size_t* bits,
                          std::string* error);

// ---------------------------------------------------------------------------
// §5.18.2 / §7.4.3 / §7.4.4 / §7.23 reference pool
// ---------------------------------------------------------------------------
// Effective refresh_frame_flags: CLK with max_mlayer_id == 0 codes no bits and
// refreshes every slot; otherwise the coded field is used.
std::uint32_t effective_refresh_flags(const SequenceConfig& config,
                                      const FrameHeader& header);

struct SlotState {
  bool valid = false;              // RefValid
  std::int32_t long_term_id = -1;  // RefLongTermId
  std::uint32_t order_hint = 0;    // RefOrderHint
  bool key_frame = false;          // RefFrameType == KEY_FRAME
  bool intra_only = false;
  std::uint64_t frame_counter = 0;
};

struct PoolState {
  int num_ref_frames = 2;
  int long_term_frame_id_bits = 0;
  int order_hint_bits = 4;
  std::array<SlotState, kNumRefFrames> slots{};
  // §5.18.2 frame-level state.
  bool olk_encountered = false;
  std::uint32_t olk_refresh = 0;   // OlkRefresh[obu_mlayer_id]
  std::uint32_t olk_order_hint = 0;  // OlkTUOrderHint
  // Long-term ids named by the most recent OLK. They must still be held when
  // the first Regular frame of the next temporal unit arrives (§7.4.4).
  std::array<std::int32_t, kRefsPerFrame> pending_ltr_ids{};
  int pending_ltr_count = 0;
  std::uint64_t frame_counter = 0;
  bool has_frame = false;
  bool has_started = false;  // a CLK has been applied
};

struct PoolEvents {
  bool closed_random_access = false;  // startCVS
  bool open_random_access = false;
  bool invalidation_ran = false;  // §5.18.2 first-Regular-frame loop
  std::uint32_t invalidated_flags = 0;
  std::uint32_t refreshed_flags = 0;
  std::uint32_t valid_before = 0;
  std::uint32_t valid_after = 0;
  bool first_refreshed_slot_only = false;  // KEY_FRAME "first" rule
};

void reset_pool(const SequenceConfig& config, PoolState* pool);

// Static §5.18.2/§7.4.2 conformance rules that do not need pool state.
bool validate_frame_header(const SequenceConfig& config,
                           const FrameHeader& header, std::string* error);

// Applies one frame header exactly in the order of §5.18.2 (invalidation,
// refresh bookkeeping) followed by §7.23 (RefValid/RefLongTermId update).
// `first_picture_in_tu` is the TU boundary flag. Returns false with `error`
// set when the header violates a conformance rule or references an invalid
// slot. The pool is not modified on failure (except through PoolEvents).
bool apply_frame(const SequenceConfig& config, const FrameHeader& header,
                 bool first_picture_in_tu, PoolState* pool,
                 PoolEvents* events, std::string* error);

// §7.3.6/§7.3.7 temporal-unit rules: at most one random-access OBU and only as
// the first frame unit; CLK and OLK never mix; leading frames never mix with
// regular frames; at least one coded output frame unit and, when a non-output
// frame unit is present, at least one output frame unit; all output frame
// units share one OrderHint.
bool validate_temporal_unit(const SequenceConfig& config,
                            std::span<const FrameHeader> frames,
                            std::string* error);

// §7.4.3 closed random access: a CLK starts a new coded video sequence, so it
// must not depend on any prior reference state, and it never names long-term
// ids. `required_long_term_ids` are the ids later frames still need.
bool validate_closed_random_access(const SequenceConfig& config,
                                   const FrameHeader& clk,
                                   std::string* error);

// §7.4.4 open random access. `required_long_term_ids` is the set of ids that
// frames after the random access point will reference; every one of them must
// be named in the OLK's ref_long_term_id list, and every named id must be
// held in a valid reference slot right now.
bool validate_open_random_access(
    const SequenceConfig& config, const FrameHeader& olk,
    const PoolState& pool, std::span<const std::int32_t> required_long_term_ids,
    std::string* error);

// A frame decoded between an OLK and the next Regular frame belongs to the
// leading set when it is not itself regular and its output order precedes the
// OLK's. Leading frames are discarded on random access, so they may reference
// pre-OLK pictures but must never refresh a retained long-term slot.
bool is_leading_candidate(const PoolState& pool, std::uint32_t order_hint);

// ---------------------------------------------------------------------------
// Encoder-side planners
// ---------------------------------------------------------------------------
// §7.4.3: the CLK refreshes every slot (no bits) and may carry a long-term id.
bool plan_closed_loop_key(const SequenceConfig& config,
                          std::uint32_t order_hint, std::int32_t long_term_id,
                          FrameHeader* header, std::string* error);

// §7.4.4: the OLK occupies `olk_slot` (only that slot is refreshed, so every
// retained long-term slot survives) and names `retain_long_term_ids` in
// ref_long_term_id. Its picture is intra and never immediately output.
bool plan_open_loop_key(const SequenceConfig& config,
                        const PoolState& pool, std::uint32_t order_hint,
                        std::uint8_t olk_slot, std::int32_t olk_long_term_id,
                        std::span<const std::int32_t> retain_long_term_ids,
                        FrameHeader* header, std::string* error);

// Plans one P frame: `ref_slots` become ref_frame_idx (one entry per ref) and
// `refresh_slot` receives the decoded picture.
bool plan_inter_frame(const SequenceConfig& config, FrameRole role,
                      std::uint32_t order_hint,
                      std::span<const std::uint8_t> ref_slots,
                      std::uint8_t refresh_slot, bool immediate_output_frame,
                      FrameHeader* header, std::string* error);

}  // namespace av2enc::inter::longterm

#endif  // AV2ENC_ENCODER_INTER_LONGTERM_LONGTERM_H_
