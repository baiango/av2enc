// AV2 multi-reference frame map and compound reference selection.
//
// This is the encoder-side model of the parts of §5.18.2 / §5.20.7.10-.11 that
// the M2 core does not carry yet:
//
//   * the decoded picture buffer (reference slots) and their order hints;
//   * the per-frame explicit reference map: NumTotalRefs, ref_frame_idx[];
//   * `reference_select` (single vs compound prediction per block);
//   * RefFrame[] indices and the derived OrderHints[] / FrameDistance[];
//   * the §5.20.7.11 read_compound_ref() mapping (which pair of RefFrame
//     indices a compound block can actually signal);
//   * is_comp_ref_allowed() / NumSameRefCompound.
//
// Everything here is pure bookkeeping: no prediction and no bitstream syntax.
// The values are exactly the ones a future frame-header writer must code and a
// tile writer must derive contexts from.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §5.18.2, §5.20.7.10, §5.20.7.11, §7.11.2.
//
// Ownership: src/encoder/m2/multiref/.  The M2 core (src/encoder/m2/*.cpp) is
// not modified by this component.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::m2::multiref {

// §9.2 constants.  M2 uses NumRefFrames = 2 slots; the map itself supports the
// full 7-entry REFS_PER_FRAME range so the same code serves longer GOPs.
inline constexpr int kNumRefFrames = 2;
inline constexpr int kRefsPerFrame = 7;
inline constexpr int kMaxNumTotalRefs = kRefsPerFrame;
inline constexpr int kRestrictedOrderHint = -1;

// A decoded reference slot.  M2 keeps the slot count fixed at two; the key
// frame refreshes all slots, later frames refresh one slot.
struct RefSlot {
  bool valid = false;
  bool key_frame = false;
  std::uint32_t order_hint = 0;
};

using RefSlotBank = std::array<RefSlot, kNumRefFrames>;

// Per-frame reference configuration.  `ref_frame_idx[i]` selects the slot used
// by RefFrame[i]; `order_hints[i]` is OrderHints[i] = RefOrderHint[slot].
struct RefFrameMap {
  int num_total_refs = 0;
  std::uint8_t ref_frame_idx[kMaxNumTotalRefs] = {};
  std::uint32_t order_hints[kMaxNumTotalRefs] = {};
  // FrameDistance[i] = get_relative_dist(OrderHint, OrderHints[i]).
  std::int32_t frame_distance[kMaxNumTotalRefs] = {};
  std::uint32_t order_hint = 0;
  int num_same_ref_compound = 0;
  // `reference_select` is the frame-header field that enables comp_mode.
  bool reference_select = false;
  bool explicit_map = true;

  bool comp_allowed_at(int block_w, int block_h) const;
  // is_same_side() over two RefFrame indices (§5.20.7.6 CWP selection).
  bool is_same_side(int ref0, int ref1) const;
  bool is_forward(int ref) const { return frame_distance[ref] >= 0; }
};

// §5.18.2 / §5.20.7.10: is_comp_ref_allowed().
bool is_comp_ref_allowed(int block_w, int block_h);

// §4.8 get_relative_dist with RESTRICTED_OH handling.
int get_relative_dist(int a, int b);
inline int restricted_order_hint() { return kRestrictedOrderHint; }

// Builds and validates the map.  `ref_frame_idx` holds one slot index per
// reference; the slots must be valid and (for M2) in range.
bool build_ref_frame_map(const RefSlotBank& slots,
                         std::span<const std::uint8_t> ref_frame_idx,
                         std::uint32_t order_hint, bool reference_select,
                         int num_same_ref_compound, RefFrameMap& out,
                         std::string& error);

bool validate_ref_frame_map(const RefFrameMap& map, std::string& error);

// Encoder policy used by the M2 multi-reference path: take the `count` most
// recent valid slots, newest first, and emit a compound-capable map.  This is
// an encoder decision, not a normative process.
bool make_recent_ref_map(const RefSlotBank& slots, int count,
                         std::uint32_t order_hint, bool reference_select,
                         RefFrameMap& out, std::string& error);

// ---------------------------------------------------------------------------
// Compound reference pairs
// ---------------------------------------------------------------------------

// A signallable compound reference assignment: RefFrame[0] = ref0,
// RefFrame[1] = ref1 (indices into ref_frame_idx).  `forced` marks a pair that
// read_compound_ref() emits without any comp_ref bit.
struct CompoundRefPair {
  int ref0 = 0;
  int ref1 = 0;
  bool forced = false;
  int slot0() const;
  int slot1() const;
};

// Enumerates every compound pair the §5.20.7.11 syntax can produce for this
// map, in decoder order.  For NumTotalRefs == 2 the list always contains the
// forced (0, 1) pair and no comp_ref bits are coded.
std::vector<CompoundRefPair> enumerate_compound_pairs(const RefFrameMap& map);

// Simulates read_compound_ref() for one bit pattern.  `bits` is consumed in
// order, one bit per coded comp_ref; `forced` bits are not consumed.  Returns
// false if the stream is malformed.
bool decode_compound_ref(const RefFrameMap& map, std::span<const int> bits,
                         CompoundRefPair& out, std::string& error);

// The frame-header fields this module expects a writer to emit.
struct ReferenceSelectSyntax {
  bool reference_select = false;
  int num_total_refs = 0;
  std::uint8_t ref_frame_idx[kMaxNumTotalRefs] = {};
  int reference_select_bits() const;
  int reference_map_bits(int num_ref_frame_bits) const;
};

ReferenceSelectSyntax reference_select_syntax(const RefFrameMap& map);

}  // namespace av2enc::m2::multiref
