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

// Writer.  In implicit mode `syntax.explicit_ref_frame_map` must be false and
// the supplied `implicit` stage-1 map is checked against `params`.
bool write_frame_ref_map_syntax(av2enc::BitWriter* writer,
                                const FrameRefMapSyntax& syntax,
                                const RefMapParams& params,
                                const ImplicitRefMap& implicit,
                                std::string* error);

// Reader.  In explicit mode it reads num_total_refs and the indices; in
// implicit mode it runs both get_ref_frames() passes over `params`.
bool read_frame_ref_map_syntax(BitReader* reader,
                               const FrameRefMapSyntax& gates,
                               const RefMapParams& params,
                               FrameRefMapResult* out, std::string* error);

}  // namespace av2enc::inter::mv_stack_full
