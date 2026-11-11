#ifndef AV2ENC_INTER_ADVANCED_B_FRAME_H_
#define AV2ENC_INTER_ADVANCED_B_FRAME_H_

// B-frame / hierarchical reference structure for AV2 v1.0.0.
//
// Two halves:
//
//  1. **Decoder-side reference bookkeeping** — §7.7 `get_ref_frames` (score,
//     rank and drop reference slots), §7.8 `get_past_future_cur_ref_lists`
//     (classify the selected refs as past / future / current), §5.18.3.1
//     `get_relative_dist`, and `get_disp_order_hint()` (recover an absolute
//     order hint from the signalled LSBs). These decide which frames a B frame
//     may predict from, so the encoder must reproduce them exactly.
//
//  2. **Encoder-side GOP scheduling** — a dyadic hierarchical B pyramid with
//     temporal layers, display order hints, encode order, refresh flags and
//     per-frame reference slots. The GOP uses `explicit_ref_frame_map`
//     (§5.18.2) so the two anchors are chosen explicitly instead of relying on
//     the §7.7 ranking, which may prefer a temporally closer slot.
//
// No bitstream is written. `GopFrame` exposes exactly the header fields the
// frame-header writer must code: `order_hint`, `frame_explicit_ref_frame_map`,
// `num_total_refs`, `ref_frame_idx[]`, `refresh_frame_flags`,
// `enable_order_hint`/`reference_select` at superblock level.
//
// Structure produced for gop_size = 8 (display index -> coding order):
//
//   display:   0   1   2   3   4   5   6   7
//   coding:    0   3   4   2   6   5   7   1     (0 = key, 7 = P anchor)
//   tlayer:    0   2   3   1   3   2   3   0
//   refs:     --  0,3 1,3 0,7 3,5 3,7 5,7 0
//
// Every B frame references one already-coded past frame and one already-coded
// future frame; the anchor at the end of the GOP is a P frame.

#include <cstdint>
#include <vector>

#include "advanced_common.h"

namespace av2enc::inter::adv {

// ---------------------------------------------------------------------------
// Reference slot state (subset of the §7.23 reference frame store that §7.7
// needs).
// ---------------------------------------------------------------------------

struct RefSlot {
  bool valid = false;
  int counter = -1;                       // RefCounter
  int order_hint = kRestrictedOrderHint;  // RefOrderHint
  int width = 0;                          // RefFrameWidth
  int height = 0;                         // RefFrameHeight
  int base_q_idx = 0;                     // RefBaseQIdx
  int tlayer_id = 0;                      // RefTLayerId
  int mlayer_id = 0;                      // RefMLayerId
  bool implicit_output = true;            // RefImplicitOutputFrame
  bool immediate_output = true;           // RefImmediateOutputFrame
};

struct RefSlotState {
  RefSlot slots[kNumRefFrames];
  int frame_counter = -1;
};

// §7.23: store a decoded frame into every slot selected by `refresh_flags`.
// `counter` is the new FrameCounter value (or -1 to auto-increment).
//
// `key_frame` implements the spec's `RefValid[i] = (KEY || SWITCH) ? first : 1`,
// i.e. a key frame refreshes its first slot and *invalidates* every other slot
// it refreshes, so pre-key frames cannot be referenced.
void refresh_reference_slots(RefSlotState* state, std::uint32_t refresh_flags, int order_hint,
                             int width, int height, int base_q_idx, bool key_frame = false,
                             int tlayer_id = 0, int mlayer_id = 0, int counter = -1);

// ---------------------------------------------------------------------------
// §7.7 get ref frames
// ---------------------------------------------------------------------------

struct RefListInput {
  RefSlotState state{};
  int order_hint = 0;
  int num_ref_frames = kNumRefFrames;
  int active_num_ref_frames = kRefsPerFrame;
  int obu_mlayer_id = 0;
  int obu_tlayer_id = 0;
  int frame_width = 0;
  int frame_height = 0;
  bool check_res = false;
  bool is_bridge = false;
  // Layer dependency is accepted for all slots by default; a multi-layer
  // stream sets these to the layer's dependency table row.
  bool layer_dependency_ok = true;
};

struct RefListResult {
  int num_total_refs = 0;
  int ref_frame_idx[kRefsPerFrame] = {0, 0, 0, 0, 0, 0, 0};
  int scores_distance[kRefsPerFrame] = {0, 0, 0, 0, 0, 0, 0};
  int ref_order_hint[kRefsPerFrame] = {
      kRestrictedOrderHint, kRestrictedOrderHint, kRestrictedOrderHint, kRestrictedOrderHint,
      kRestrictedOrderHint, kRestrictedOrderHint, kRestrictedOrderHint};
};

// Implements §7.7 exactly: distinct-slot detection, the score (distance decay +
// base_q + resolution penalty), the drop rule when more than REFS_PER_FRAME
// candidates rank, bubble sort, and the trailing restricted slots.
RefListResult build_ref_frame_list(const RefListInput& in);

// ---------------------------------------------------------------------------
// §7.8 get past future cur ref lists
// ---------------------------------------------------------------------------

struct PastFutureLists {
  int num_past = 0;
  int num_future = 0;
  int num_cur = 0;
  int closest_past = -1;
  int closest_future = -1;
  int furthest_future = -1;
  int skip_seg_frame = -1;
  int orig_closest_past = -1;
  int orig_closest_future = -1;
  int past[kRefsPerFrame] = {0, 0, 0, 0, 0, 0, 0};
  int future[kRefsPerFrame] = {0, 0, 0, 0, 0, 0, 0};
  int cur[kRefsPerFrame] = {0, 0, 0, 0, 0, 0, 0};
};

// `ref_order_hint[i]` is RefOrderHint[ref_frame_idx[i]];
// `scores_distance[i]` is ScoresDistance[i] (already `get_relative_dist()`ed
// in display-order space with the §7.7 distance convention).
PastFutureLists get_past_future_cur_ref_lists(const int ref_order_hint[kRefsPerFrame],
                                              const int scores_distance[kRefsPerFrame],
                                              int num_total_refs);

// ---------------------------------------------------------------------------
// Order hint reconstruction (§5.18.2 / §5.18.3.1).
// ---------------------------------------------------------------------------

// get_max_disp_order_hint(onlyShowable).
int max_display_order_hint(const RefSlotState& state, int num_ref_frames, bool only_showable);

// get_disp_order_hint(): lifts the signalled `order_hint` LSBs to an absolute
// display order hint using the maximum hint currently in the reference store.
// `order_hint_bits` is OrderHintBits from the sequence header.
int display_order_hint(int order_hint_lsbs, int order_hint_bits, int max_disp_order_hint);

// Encoder-side: choose the LSBs to signal for `absolute_hint`, given the
// current store and OrderHintBits. Returns the value to write with f(bits).
int order_hint_lsbs_for(int absolute_hint, int order_hint_bits, const RefSlotState& state,
                        int num_ref_frames);

// ---------------------------------------------------------------------------
// Encoder-side hierarchical GOP
// ---------------------------------------------------------------------------

struct GopConfig {
  int gop_size = 8;          // power of two, 2..16
  int start_order_hint = 0;  // absolute order hint of display index 0
  int num_slots = kNumRefFrames;
  int key_slot = 0;
  int base_q_idx = 100;  // stamped into the slot state for simulation
};

struct GopFrame {
  int display_index = 0;       // 0..gop_size-1
  int display_order_hint = 0;  // absolute
  int coding_index = 0;        // 0-based encode order
  int temporal_layer = 0;
  bool is_key = false;
  bool is_anchor = false;  // P frame at the end of the GOP
  // Explicit reference map (frame_explicit_ref_frame_map = 1).
  int num_total_refs = 0;
  int ref_display[2] = {-1, -1};  // display indices of past/future anchors
  int ref_slot[2] = {-1, -1};
  int ref_order_hint[2] = {kRestrictedOrderHint, kRestrictedOrderHint};
  int slot = 0;  // slot this frame refreshes
  std::uint32_t refresh_frame_flags = 0;
};

// Builds the dyadic B pyramid described at the top of this file. Returns an
// empty vector for an invalid gop_size/num_slots.
std::vector<GopFrame> build_hierarchical_gop(const GopConfig& config);

struct GopValidation {
  bool ok = false;
  int first_bad_coding_index = -1;
  int first_bad_frame = -1;
  // Simulated per-frame classification (indexed by coding order).
  int num_frames = 0;
  bool refs_were_past_future[32] = {};
};

// Walks the GOP in coding order, applying §7.23 refresh and the §7.8
// classification to the explicit reference map. Fails if a referenced slot is
// not yet valid, if the intended past/future split does not hold, or if the two
// explicit refs are not distinct.
GopValidation validate_gop(const std::vector<GopFrame>& gop, const GopConfig& config);

}  // namespace av2enc::inter::adv

#endif  // AV2ENC_INTER_ADVANCED_B_FRAME_H_
