// AV2 v1.0.0 §7.7 get_ref_frames, §7.8 get_past_future_cur_ref_lists, the
// §5.18.2 display-order-hint derivation, and the §6.4.1 R77 conformance rules.
//
// This is the encoder-side *implicit* reference mapping (when
// `explicit_ref_frame_map == 0`) and the display-order-hint computation that
// goes with it.  The M2 core writes an explicit map, so this path has never
// been exercised; it is required both to emit an implicit-map stream and to
// prove R77 for either mode.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §5.18.2, §6.4.1, §7.7, §7.8.
#pragma once

#include <cstdint>
#include <span>
#include <string>

#include "core/bitstream/bitwriter.h"
#include "codec/inter/mv_stack/bit_reader.h"
#include "codec/inter/mv_stack/mv_stack_full.h"

namespace av2enc::inter::mv_stack_full {

inline constexpr int kRefsPerFrameConst = 7;    // REFS_PER_FRAME
inline constexpr int kMaxNumRefSlots = 16;      // explicit_num_ref_frames max
inline constexpr int kDecayDistCap = 6;         // DECAY_DIST_CAP
inline constexpr int kDistWeightBits = 6;       // DIST_WEIGHT_BITS
inline constexpr int kDisplayOrderHintBits = 30;  // DISPLAY_ORDER_HINT_BITS
inline constexpr int kRestrictedOrderHint = -1;   // RESTRICTED_OH

// One decoded reference slot (the fields the two §7.7 passes read).
struct RefSlot {
  bool valid = false;
  bool restricted = false;  // RefOrderHint == RESTRICTED_OH
  int order_hint = 0;
  int frame_width = 0;
  int frame_height = 0;
  int counter = 0;  // RefCounter: unique tag for the frame in the slot
  int mlayer = 0;
  int tlayer = 0;
  int base_q_idx = 0;
  bool implicit_output = false;
  bool immediate_output = false;
  bool is_bridge = false;
};

inline int ref_order_hint(const RefSlot& slot) {
  return slot.restricted ? kRestrictedOrderHint : slot.order_hint;
}

// TLayerDependencyMap[obu_mlayer_id][obu_tlayer_id][ref_tlayer] and
// MLayerDependencyMap[obu_mlayer_id][ref_mlayer], represented as bit masks
// over the referenced layer id.  Bit 0 is layer 0.
struct DependencyMaps {
  std::uint64_t tlayer_mask = ~std::uint64_t{0};
  std::uint64_t mlayer_mask = ~std::uint64_t{0};
};

struct RefMapParams {
  int num_ref_frames = 8;         // NumRefFrames
  int active_num_ref_frames = 7;  // ActiveNumRefFrames = Min(REFS_PER_FRAME,
                                  // NumRefFrames)
  int order_hint = 0;
  int order_hint_bits = 4;
  int obu_mlayer_id = 0;
  int obu_tlayer_id = 0;
  DependencyMaps deps{};
  std::uint32_t allowed_frames = ~std::uint32_t{0};
  bool is_bridge = false;
  int bridge_frame_ref_idx = 0;
  bool check_res = true;
  int frame_width = 0;
  int frame_height = 0;
  const RefSlot* slots = nullptr;  // num_ref_frames entries
};

// §7.23 reference frame update state.  This is the encoder-side mirror of the
// decoder variables §7.7 reads (RefValid, RefCounter, RefOrderHint,
// RefFrameWidth/Height, RefBaseQIdx, RefTLayerId, RefMLayerId).  Every coded
// frame must call refresh() with the same refresh_frame_flags the frame header
// carries, or the implicit mapping diverges at the next frame.
struct RefSlotBank {
  int num_ref_frames = 0;
  int frame_counter = -1;
  RefSlot slots[kMaxNumRefSlots] = {};

  // Applies §7.23 for one decoded frame (called after the frame is coded).
  // For a KEY_FRAME / SWITCH_FRAME only the first refreshed slot stays valid
  // (`RefValid[i] = first ? 1 : 0`); a regular frame refreshes every named slot
  // with `RefCounter = FrameCounter`.  `counter < 0` increments the stored
  // FrameCounter, which is the decoder behaviour.
  bool refresh(int refresh_frame_flags, int order_hint, int frame_width,
               int frame_height, int base_q_idx, bool key_frame,
               int tlayer_id = 0, int mlayer_id = 0,
               bool implicit_output = true, bool immediate_output = true,
               int counter = -1, std::string* error = nullptr);
};

struct RefMap {
  int num_total_refs = 0;
  int ref_frame_idx[kRefsPerFrameConst] = {};
  // NRanked can exceed REFS_PER_FRAME (one entry is then dropped), so the
  // ranking arrays are sized for the full NumRefFrames range.
  int scores_distance[kMaxNumRefSlots] = {};
  int scores_order_hint[kMaxNumRefSlots] = {};
  int scores_score[kMaxNumRefSlots] = {};
  int scores_index[kMaxNumRefSlots] = {};
  // Per-slot mapOrderHint (-1 when the slot is not a candidate).
  int map_order_hint[kMaxNumRefSlots] = {};

  // §7.8 classification (indices into ref_frame_idx[]).
  int num_past_refs = 0;
  int num_future_refs = 0;
  int num_cur_refs = 0;
  int closest_past = kRefNone;
  int closest_future = kRefNone;
  int furthest_future = kRefNone;
  int skip_seg_frame = 0;
  int cur_refs[kRefsPerFrameConst] = {};
};

// §7.7 get_ref_frames().  `out->num_total_refs/ref_frame_idx` are the final
// mapping for this pass.
bool get_ref_frames(const RefMapParams& params, RefMap* out,
                    std::string* error);

// §7.8 get_past_future_cur_ref_lists().  Requires a RefMap produced by
// get_ref_frames() with the same params.
bool get_past_future_cur_ref_lists(const RefMapParams& params, RefMap* map,
                                   std::string* error);

// §5.18.2 for an *explicit* map: after frame_size() the decoder sets
// `ScoresDistance[i] = get_relative_dist(OrderHint, RefOrderHint[ref_frame_idx[i]])`
// and then calls get_past_future_cur_ref_lists().  This fills the same fields
// as the implicit path (scores_*, past/future/cur, ClosestPast/Future,
// FurthestFuture, SkipSegFrame) so the encoder can use one model for both.
// The indices are validated against the live slot bank.
bool derive_explicit_ref_map(const RefMapParams& params, int num_total_refs,
                             const int* ref_frame_idx, RefMap* out,
                             std::string* error);

// Full implicit-mapping sequence for one frame: §5.18.2 calls get_ref_frames(0)
// before frame_size() and get_ref_frames(1) after it.  Both passes are
// returned; the second is normative for the block layer.
struct ImplicitRefMap {
  RefMap stage0;  // checkRes = 0
  RefMap stage1;  // checkRes = 1 (after the frame size is known)
};
bool derive_implicit_ref_map(const RefMapParams& params, ImplicitRefMap* out,
                             std::string* error);

// ---------------------------------------------------------------------------
// §5.18.2 get_disp_order_hint / get_max_disp_order_hint
// ---------------------------------------------------------------------------

struct DispOrderHintParams {
  bool obu_closed_loop_key = false;
  bool is_sef = false;
  bool frame_type_switch = false;
  bool restricted_prediction_switch = false;
  int order_hint_lsbs = 0;
  int order_hint_bits = 4;
  int obu_mlayer_id = 0;
  int obu_tlayer_id = 0;
  DependencyMaps deps{};
  int num_ref_frames = 8;
  const RefSlot* slots = nullptr;
};

// Returns the maximum RefOrderHint over the eligible slots (0 when none).
bool get_max_disp_order_hint(const DispOrderHintParams& params,
                             bool only_showable, int* max_disp,
                             std::string* error);

// Returns the derived display order hint (can exceed (1<<OrderHintBits)-1).
// Fails when the §5.18.2 conformance bound
// `value < (1 << (DISPLAY_ORDER_HINT_BITS - 1))` is violated.
bool get_disp_order_hint(const DispOrderHintParams& params, int* disp_order_hint,
                         std::string* error);

// ---------------------------------------------------------------------------
// R77 — §6.4.1 operating-point consistency
// ---------------------------------------------------------------------------
//
// One operating point's view of the current frame's reference configuration.
// `ref_identity` is the stable frame identity (the encoder supplies e.g.
// RefCounter; order hint alone is not unique in the presence of wraparound).
struct OperatingPointRefMap {
  bool explicit_map = false;
  int num_total_refs = 0;
  int ref_frame_idx[kRefsPerFrameConst] = {};  // buffer indices
  int ref_identity[kRefsPerFrameConst] = {};   // frame identity per RefFrame[]
  int disp_order_hint = 0;
};

struct R77Result {
  bool mapping_consistent = false;
  bool disp_order_hint_consistent = false;
  bool explicit_slot_consistent = false;
};

// Checks the three §6.4.1 requirements:
//   1. the display order hint of the frame is the same for every operating
//      point;
//   2. an implicit mapping produces exactly the same reference frames at the
//      same indices in every operating point;
//   3. an explicit map associates each buffer index with the same frame in
//      every operating point.
bool validate_r77(std::span<const OperatingPointRefMap> maps, R77Result* out,
                  std::string* error);

// ---------------------------------------------------------------------------
// Frame-header reference-map syntax (§5.18.2)
// ---------------------------------------------------------------------------
//
// The implicit path writes *no* reference-map bits:
//   if (FrameType==SWITCH_FRAME || IsBridge) explicitRefFrameMap = 1
//   else if (seq.explicit_ref_frame_map) frame_explicit_ref_frame_map f(1)
//   else explicitRefFrameMap = 0
//   if (!IsBridge && explicitRefFrameMap) { num_total_refs f(3);
//                                            ref_frame_idx f(CeilLog2(NumRefFrames)) }
//   else get_ref_frames(0)
//   ... frame_size ...
//   if (!explicitRefFrameMap) get_ref_frames(1)
struct FrameRefMapSyntax {
  bool seq_explicit_ref_frame_map = true;
  bool frame_is_switch = false;
  bool is_bridge = false;
  int bridge_frame_ref_idx = 0;
  bool explicit_ref_frame_map = false;  // frame_explicit_ref_frame_map
  int num_total_refs = 0;
  int ref_frame_idx[kRefsPerFrameConst] = {};
};

struct FrameRefMapResult {
  bool explicit_map = false;
  RefMap map;
};

// Writer.  Codes exactly the §5.18.2 frame-header reference-map syntax for the
// gates in `syntax`:
//   * sequence `explicit_ref_frame_map == 1` -> one frame_explicit_ref_frame_map
//     bit; if it is 1, `num_total_refs f(3)` and that many
//     `ref_frame_idx f(CeilLog2(NumRefFrames))` values follow (the count is
//     per frame, so the list length varies);
//   * sequence `explicit_ref_frame_map == 0` -> no bits at all, and the mapping
//     is the §7.7 result;
//   * a SWITCH frame (or a bridge) is forced to the explicit form.
// In implicit mode the supplied stage-1 map is re-derived and checked against
// `params`; `false` means the encoder's block syntax would not match the
// decoder's mapping.
bool write_frame_ref_map_syntax(av2enc::BitWriter* writer,
                                const FrameRefMapSyntax& syntax,
                                const RefMapParams& params,
                                const ImplicitRefMap& implicit,
                                std::string* error);

// Reader.  Mirrors the writer, including the frame_explicit_ref_frame_map bit.
// In explicit mode it also derives ScoresDistance and the §7.8 classification
// for the read map; in implicit mode it runs both get_ref_frames() passes over
// `params`.

bool read_frame_ref_map_syntax(BitReader* reader,
                               const FrameRefMapSyntax& gates,
                               const RefMapParams& params,
                               FrameRefMapResult* out, std::string* error);

}  // namespace av2enc::inter::mv_stack_full
