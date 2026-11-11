#ifndef AV2ENC_INTER_BFRAMES_BFRAMES_H_
#define AV2ENC_INTER_BFRAMES_BFRAMES_H_

// Encoder-side B-frame reference / hierarchy layer for AV2 v1.0.0.
//
// This directory owns *scheduling and validation*, not block prediction and not
// bitstream emission. It reuses the decoder-exact bookkeeping in
// `src/encoder/inter/advanced/b_frame.*`:
//
//   §7.7   build_ref_frame_list()             automatic reference ranking
//   §7.8   get_past_future_cur_ref_lists()    past / future / current split
//   §7.23  refresh_reference_slots()          slot refresh rules
//   §5.18  display_order_hint(), order_hint_lsbs_for(), get_relative_dist()
//
// Added here:
//
//   * hierarchical GOP schedules (low-delay P, one-level IPBB, dyadic B);
//   * absolute display order hints plus the exact LSB value to signal;
//   * explicit reference maps (frame_explicit_ref_frame_map = 1);
//   * deterministic coding order and deterministic slot allocation;
//   * key-frame reset and leading-picture (display-before-key, decode-after-key)
//     schedules;
//   * a decoder simulation that walks the coding order, applies §7.23 and the
//     §7.21.4/.5/.6 output processes, and compares the decoder-visible output
//     order with the intended display order;
//   * validation that rejects references outside the defined range, references
//     to frames that are not yet decoded, duplicate/ambiguous references,
//     references across a key-frame reset, wrong refresh flags, and order hints
//     that cannot be reconstructed.
//
// Output model: one embedded layer (max_mlayer_id = 0), so
// `output_ordering()` reduces to the order hint. §7.21.4/.5/.6 are transcribed
// closely enough to decide *relative* output order; timing is not modelled.
//
// Non-goals (documented in README.md / M2_INTEGRATION.md):
//   * no OBU syntax, tile syntax, entropy coding or RDO;
//   * at most two explicit references per frame (the AV2 syntax supports up to
//     REFS_PER_FRAME = 7; B/P scheduling only needs two);
//   * no show_existing_frame, switch frames, bridge frames, long-term refs,
//     multi-layer, or cross-GOP references;
//   * leading pictures are validated at the schedule level. A real stream also
//     needs OBU_OPEN_LOOP_KEY / OBU_LEADING_TILE_GROUP; the M2 path currently
//     writes only OBU_CLOSED_LOOP_KEY and OBU_REGULAR_TILE_GROUP.

#include <cstdint>
#include <vector>

namespace av2enc::inter::bframes {

inline constexpr int kMaxExplicitRefs = 2;
inline constexpr int kMaxRefSlots = 16;

enum class GopPattern : std::uint8_t {
  // I, then a display-order P chain. Display order == coding order.
  kLowDelayP = 0,
  // Key, P anchor, then all interior frames as B referencing both anchors:
  // coding frame types "I P B B ...".
  kIpbb = 1,
  // Key, P anchor, then a dyadic B pyramid (BFS coding order).
  kHierarchicalB = 2,
};

enum class FrameKind : std::uint8_t {
  kKey = 0,
  kP = 1,
  kB = 2,
};

enum class PlanError : std::uint8_t {
  kOk = 0,
  kBadGopSize,               // gop_size outside 2..16, or not a power of two
  kBadPattern,               // unsupported pattern / inconsistent kind
  kBadSlotCount,             // num_slots outside 2..16
  kBadOrderHintBits,         // order_hint_bits outside 1..16
  kBadLeadingCount,          // negative leading picture count
  kNotEnoughSlots,           // no free reference slot for a frame
  kBadDisplayOrder,          // display indices/hints not strictly increasing
  kBadCodingOrder,           // coding_order is not a permutation of 0..N-1
  kMissingKeyFrame,          // first coded frame is not a key frame
  kKeyFrameHasRefs,          // a key frame carries an explicit reference map
  kTooManyRefs,              // more than kMaxExplicitRefs references
  kMissingReference,         // non-key frame without a reference
  kReferenceOutOfRange,      // ref_display outside [0, N)
  kReferenceNotDecoded,      // reference is later in coding order
  kReferenceSlotInvalid,     // slot invalid or holds a different frame
  kReferenceHintMismatch,    // ref_order_hint disagrees with the referenced frame
  kKeyFrameRefInvalidated,   // reference crosses a key-frame reset / GOP
  kLeadingPictureRule,       // leading/non-leading reference rule violated
  kDuplicateReference,       // same display or slot referenced twice
  kBadReferenceSplit,        // B/P reference is not one past + one future
  kRefreshFlagsInvalid,      // keyframe not allFrames, or bad per-frame flags
  kOutputFlagsInvalid,       // both/neither immediate_output/implicit_output
  kOrderHintReconstruct,     // signalled LSBs do not reconstruct the hint
  kOutputOrderMismatch,      // decoder-visible output order != display order
  kOutputMissingFrame,       // a decoded frame was never output
  kInternal,                 // builder/validator invariant broken
};

const char* plan_error_name(PlanError error);
const char* gop_pattern_name(GopPattern pattern);
const char* frame_kind_name(FrameKind kind);

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

struct GopConfig {
  // Number of display frames from the key frame through the P anchor inclusive.
  // kHierarchicalB requires a power of two; all patterns accept 2..16.
  int gop_size = 8;
  GopPattern pattern = GopPattern::kHierarchicalB;
  // Absolute order hint of display index 0. The key frame's hint is
  // start_order_hint + leading_pictures.
  int start_order_hint = 0;
  // §5.4.6 OrderHintBits. 4 covers a 16-frame GOP per unit; larger values are
  // allowed and are exercised by the tests.
  int order_hint_bits = 4;
  // §5.4.6 NumRefFrames, 2..16. Every frame is stored in one slot until it is
  // output (implicit output relies on the slot), so a closed unit normally
  // needs num_slots >= gop_size + leading_pictures.
  int num_slots = 16;
  int base_q_idx = 100;
  // Leading pictures: frames displayed before the key frame but decoded after
  // it. They are assigned display indices 0..L-1, the key frame is at display
  // L, and the GOP's P anchor is at display L + gop_size - 1.
  int leading_pictures = 0;
};

struct SequenceConfig {
  int num_gops = 2;
  int gop_size = 8;
  GopPattern pattern = GopPattern::kHierarchicalB;
  int start_order_hint = 0;
  int order_hint_bits = 4;
  int num_slots = 16;
  int base_q_idx = 100;
  // Applied to every GOP after the first. The first GOP starts at display 0
  // with its key frame, because there is nothing to display before it.
  int leading_pictures = 0;
};

// ---------------------------------------------------------------------------
// Output plan
// ---------------------------------------------------------------------------

struct FramePlan {
  int display_index = 0;          // position in output order, 0..N-1
  int display_order_hint = 0;     // absolute OrderHint, monotonic in display
  int coding_index = 0;           // 0-based encode order
  int gop_index = 0;
  int temporal_layer = 0;
  FrameKind kind = FrameKind::kP;
  bool is_key = false;
  bool is_anchor = false;         // P frame closing the GOP
  bool is_leading = false;        // display < its key frame's display

  // §5.18.2 frame header output flags. Neither flag is set only for hidden
  // frames, which this layer does not schedule.
  bool immediate_output_frame = false;
  bool implicit_output_frame = true;

  // §5.18.2 explicit reference map.
  bool frame_explicit_ref_frame_map = true;
  int num_total_refs = 0;
  int ref_display[kMaxExplicitRefs] = {-1, -1};
  int ref_slot[kMaxExplicitRefs] = {-1, -1};
  int ref_order_hint[kMaxExplicitRefs] = {-1, -1};

  // §5.18.2 refresh_frame_flags. The slot is `slot`; a key frame refreshes
  // allFrames and the first refreshed slot is slot 0.
  int slot = -1;
  std::uint32_t refresh_frame_flags = 0;

  // Value to write with f(OrderHintBits). The decoder reconstructs the absolute
  // hint with display_order_hint().
  int order_hint_lsbs = 0;

  bool is_inter() const { return !is_key; }
};

struct Plan {
  GopPattern pattern = GopPattern::kHierarchicalB;
  int gop_size = 8;
  int start_order_hint = 0;
  int order_hint_bits = 4;
  int num_slots = 16;
  int base_q_idx = 100;
  int leading_pictures = 0;
  bool is_sequence = false;

  // Indexed by display index.
  std::vector<FramePlan> frames;
  // Display indices in coding order. coding_order[i] has coding_index i.
  std::vector<int> coding_order;

  int display_count() const { return static_cast<int>(frames.size()); }
  const FramePlan* frame_at_display(int display_index) const;
};

// Exact §5.18.2 fields the M2 frame-header writer must code for one frame.
// `order_hint_lsbs` is *not* the absolute hint.
struct FrameHeaderFields {
  bool keyframe = false;
  std::uint32_t order_hint_lsbs = 0;
  std::uint32_t refresh_frame_flags = 0;
  bool frame_explicit_ref_frame_map = true;
  std::uint8_t num_total_refs = 0;
  std::uint8_t ref_frame_idx[kMaxExplicitRefs] = {0, 0};
  bool immediate_output_frame = true;
  bool implicit_output_frame = false;
  // §5.18.2 reference_select: 1 when a two-reference frame may choose single or
  // compound prediction at superblock level.
  bool reference_select = false;
};

FrameHeaderFields header_fields(const FramePlan& frame);

// Frame kinds in coding (encode/decoder) order. This is the "frame-type order"
// the decoder sees, independent of display order.
std::vector<FrameKind> coding_frame_types(const Plan& plan);

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

// Builds one closed GOP unit. Display indices are 0..gop_size+leading-1; the
// key frame is at display `leading_pictures`. The builder assigns slots, order
// hint LSBs and output flags, then runs validate_plan(). On failure, returns
// false and sets *error (when non-null).
bool build_gop(const GopConfig& config, Plan& plan, PlanError* error);

// Builds `num_gops` independent closed GOP units. The first unit has no leading
// pictures; later units use config.leading_pictures. No reference crosses a GOP
// boundary (each unit starts with a key frame). On failure, returns false and
// sets *error.
bool build_sequence(const SequenceConfig& config, Plan& plan, PlanError* error);

// ---------------------------------------------------------------------------
// Decoder simulation
// ---------------------------------------------------------------------------

struct DecodeFrameTrace {
  int display_index = -1;
  int coding_index = -1;
  int num_total_refs = 0;
  int ref_display[kMaxExplicitRefs] = {-1, -1};
  int ref_slot[kMaxExplicitRefs] = {-1, -1};

  // §7.8 classification over the explicit map (indices into ref_display[]).
  int num_past = 0;
  int num_future = 0;
  int num_cur = 0;
  int past[7] = {0, 0, 0, 0, 0, 0, 0};
  int future[7] = {0, 0, 0, 0, 0, 0, 0};
  int cur[7] = {0, 0, 0, 0, 0, 0, 0};

  // §7.7 automatic ranking over the decoder state *before* this frame is
  // refreshed. This is the decoder's non-explicit choice; the plan's explicit
  // map is allowed to differ. Present so callers/tests can cross-check and so
  // the ranking code is reused rather than duplicated.
  int auto_num_total_refs = 0;
  int auto_ref_slot[7] = {0, 0, 0, 0, 0, 0, 0};
  int auto_ref_order_hint[7] = {-1, -1, -1, -1, -1, -1, -1};
  int auto_scores_distance[7] = {0, 0, 0, 0, 0, 0, 0};

  bool ref_slots_held_intended_frames = false;
  bool past_future_ok = false;
  bool refresh_ok = false;
  bool output_immediate = false;
};

struct DecodeReport {
  bool ok = false;
  PlanError error = PlanError::kOk;
  int error_coding_index = -1;
  int error_display_index = -1;

  std::vector<DecodeFrameTrace> frames;  // indexed by coding order
  std::vector<int> decode_order;         // display indices, coding order
  std::vector<int> output_order;         // display indices, decoder-visible
  std::vector<int> expected_output_order;
  std::vector<int> slot_of_frame;        // slot refreshed by each display frame
  bool output_matches_display_order = false;
};

// Walks the coding order, checks every reference against the live slot state,
// applies §7.23, and runs the §7.21.4/.5/.6 output processes. Never crashes on a
// malformed plan: it reports an error.
bool simulate_decode(const Plan& plan, DecodeReport* report);

struct ValidationReport {
  bool ok = false;
  PlanError error = PlanError::kOk;
  int error_coding_index = -1;
  int error_display_index = -1;
  // Filled from the decoder simulation when it ran.
  std::vector<int> decode_order;
  std::vector<int> output_order;
  bool output_matches_display_order = false;
};

// Structural checks + simulate_decode(). This is the gate the M2 integration
// must call before writing any bits.
bool validate_plan(const Plan& plan, ValidationReport* report);

}  // namespace av2enc::inter::bframes

#endif  // AV2ENC_INTER_BFRAMES_BFRAMES_H_
