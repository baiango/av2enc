#include "codec/inter/multiref/ref_map.h"

#include <algorithm>

namespace av2enc::video::multiref {
bool is_comp_ref_allowed(int block_w, int block_h) {
  // §5.20.7.3: (Min(w, h) >= 8) || is_thin_4xn_nx4_block().  The M2 partition
  // grid never produces the thin 4xn/nx4 sizes, so the second term is 0.
  return block_w >= 8 && block_h >= 8;
}

int get_relative_dist(int a, int b) {
  if (a == kRestrictedOrderHint && b == kRestrictedOrderHint) {
    return 0;
  }
  if (a == kRestrictedOrderHint) {
    return 127;
  }
  if (b == kRestrictedOrderHint) {
    return -127;
  }
  const int d = a - b;
  return d < -127 ? -127 : (d > 127 ? 127 : d);
}

bool RefFrameMap::comp_allowed_at(int block_w, int block_h) const {
  return reference_select && is_comp_ref_allowed(block_w, block_h) &&
         num_total_refs >= 2;
}

bool RefFrameMap::is_same_side(int ref0, int ref1) const {
  const bool f0 = frame_distance[ref0] >= 0;
  const bool f1 = frame_distance[ref1] >= 0;
  return f0 == f1;
}

int CompoundRefPair::slot0() const { return ref0; }
int CompoundRefPair::slot1() const { return ref1; }

bool build_ref_frame_map(const RefSlotBank& slots,
                         std::span<const std::uint8_t> ref_frame_idx,
                         std::uint32_t order_hint, bool reference_select,
                         int num_same_ref_compound, RefFrameMap& out,
                         std::string& error) {
  error.clear();
  out = RefFrameMap{};
  if (ref_frame_idx.empty() ||
      ref_frame_idx.size() > static_cast<std::size_t>(kMaxNumTotalRefs)) {
    error = "num_total_refs outside 1..7";
    return false;
  }
  if (num_same_ref_compound < 0 || num_same_ref_compound > kMaxNumTotalRefs) {
    error = "num_same_ref_compound outside 0..7";
    return false;
  }
  out.num_total_refs = static_cast<int>(ref_frame_idx.size());
  out.order_hint = order_hint;
  out.reference_select = reference_select;
  out.num_same_ref_compound = num_same_ref_compound;
  out.explicit_map = true;
  for (std::size_t i = 0; i < ref_frame_idx.size(); ++i) {
    const std::uint8_t slot = ref_frame_idx[i];
    if (slot >= static_cast<std::uint8_t>(kNumRefFrames)) {
      error = "ref_frame_idx[" + std::to_string(i) + "] = " +
              std::to_string(static_cast<int>(slot)) + " outside NumRefFrames";
      return false;
    }
    if (!slots[slot].valid) {
      error = "ref_frame_idx[" + std::to_string(i) + "] selects an empty slot";
      return false;
    }
    out.ref_frame_idx[i] = slot;
    out.order_hints[i] = slots[slot].order_hint;
    out.frame_distance[i] =
        get_relative_dist(static_cast<int>(order_hint),
                          static_cast<int>(slots[slot].order_hint));
  }
  return validate_ref_frame_map(out, error);
}

bool validate_ref_frame_map(const RefFrameMap& map, std::string& error) {
  error.clear();
  if (map.num_total_refs < 1 || map.num_total_refs > kMaxNumTotalRefs) {
    error = "num_total_refs outside 1..7";
    return false;
  }
  if (map.num_same_ref_compound < 0 ||
      map.num_same_ref_compound > map.num_total_refs) {
    // §5.18.2: NumSameRefCompound = Min(num_same_ref_compound, NumTotalRefs).
    error = "num_same_ref_compound exceeds num_total_refs (clamp with "
            "Min(num_same_ref_compound, NumTotalRefs))";
    return false;
  }
  for (int i = 0; i < map.num_total_refs; ++i) {
    if (map.ref_frame_idx[i] >=
        static_cast<std::uint8_t>(kNumRefFrames)) {
      error = "ref_frame_idx outside NumRefFrames";
      return false;
    }
    const std::int32_t expect =
        get_relative_dist(static_cast<int>(map.order_hint),
                          static_cast<int>(map.order_hints[i]));
    if (map.frame_distance[i] != expect) {
      error = "frame_distance does not match get_relative_dist()";
      return false;
    }
  }
  return true;
}

bool make_recent_ref_map(const RefSlotBank& slots, int count,
                         std::uint32_t order_hint, bool reference_select,
                         RefFrameMap& out, std::string& error) {
  error.clear();
  if (count < 1 || count > kNumRefFrames) {
    error = "requested reference count outside 1..NumRefFrames";
    return false;
  }
  // Newest first by order hint; ties keep the lower slot index.  Slot 0 is
  // never preferred over an equally-newer slot purely by index, so this is a
  // stable, deterministic policy.
  std::vector<int> order;
  order.reserve(kNumRefFrames);
  for (int s = 0; s < kNumRefFrames; ++s) {
    if (slots[static_cast<std::size_t>(s)].valid) order.push_back(s);
  }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    const std::uint32_t ha = slots[static_cast<std::size_t>(a)].order_hint;
    const std::uint32_t hb = slots[static_cast<std::size_t>(b)].order_hint;
    if (ha != hb) return ha > hb;
    return a < b;
  });
  if (static_cast<int>(order.size()) < count) {
    error = "not enough valid reference slots for the requested map";
    return false;
  }
  std::vector<std::uint8_t> idx(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    idx[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(order[static_cast<std::size_t>(i)]);
  }
  return build_ref_frame_map(slots, idx, order_hint, reference_select,
                             /*num_same_ref_compound=*/0, out, error);
}

// ---------------------------------------------------------------------------
// §5.20.7.11 read_compound_ref()
//
//   RefFrame[0] = RefFrame[1] = NumTotalRefs - 1
//   nFound = 0
//   for ref in 0 .. NumTotalRefs - 2 while nFound < 2:
//     if nFound == 0 && ref == 2:                     comp_ref = 1 (implied)
//     elif nFound == 0 && ref+1 >= NumSameRefCompound
//                      && ref+1 == NumTotalRefs-1:   comp_ref = 1 (implied)
//     else:                                           comp_ref = S()
//     if comp_ref: RefFrame[nFound++] = ref
//                  if ref < NumSameRefCompound: ref--
// ---------------------------------------------------------------------------
bool decode_compound_ref(const RefFrameMap& map, std::span<const int> bits,
                         CompoundRefPair& out, std::string& error) {
  error.clear();
  out = CompoundRefPair{};
  if (map.num_total_refs < 2) {
    error = "compound reference coding requires NumTotalRefs >= 2";
    return false;
  }
  int ref_frame[2] = {map.num_total_refs - 1, map.num_total_refs - 1};
  int n_found = 0;
  std::size_t bit_pos = 0;
  const int same_ref = std::min(map.num_same_ref_compound, map.num_total_refs);
  for (int ref = 0; ref < map.num_total_refs - 1 && n_found < 2; ++ref) {
    int comp_ref = 0;
    bool implied = false;
    if (n_found == 0 && ref == 2) {
      comp_ref = 1;
      implied = true;
    } else if (n_found == 0 && ref + 1 >= same_ref &&
               ref + 1 == map.num_total_refs - 1) {
      comp_ref = 1;
      implied = true;
    } else {
      if (bit_pos >= bits.size()) {
        error = "compound ref bitstream exhausted";
        return false;
      }
      comp_ref = bits[bit_pos++] != 0 ? 1 : 0;
    }
    if (comp_ref) {
      // Forced pairs are recorded as implied; a pair is `forced` only when
      // every coded bit was implied.
      ref_frame[n_found] = ref;
      ++n_found;
      if (ref < same_ref) --ref;
      out.forced = out.forced || implied;
    }
  }
  out.ref0 = ref_frame[0];
  out.ref1 = ref_frame[1];
  if (out.ref0 == out.ref1 && same_ref == 0) {
    error = "same-reference compound with NumSameRefCompound == 0";
    return false;
  }
  // `forced` describes the whole pair: it is only true when no comp_ref bit was
  // read (the NumTotalRefs == 2 case).
  out.forced = bit_pos == 0;
  return true;
}

std::vector<CompoundRefPair> enumerate_compound_pairs(const RefFrameMap& map) {
  std::vector<CompoundRefPair> pairs;
  if (map.num_total_refs < 2) return pairs;
  // Exhaustive walk of every comp_ref bit pattern.  read_compound_ref() can
  // repeat a reference when ref < NumSameRefCompound, so the bit count can
  // exceed NumTotalRefs - 1; 2 * (NumTotalRefs - 1) covers the repeats.
  const int max_bits = std::min(14, 2 * (map.num_total_refs - 1));
  const int max_patterns = 1 << max_bits;
  std::vector<int> seen;
  for (int pattern = 0; pattern < max_patterns; ++pattern) {
    std::vector<int> scratch;
    for (int i = 0; i < max_bits; ++i) {
      scratch.push_back((pattern >> i) & 1);
    }
    CompoundRefPair pair;
    std::string error;
    if (!decode_compound_ref(map, scratch, pair, error)) continue;
    const int key = pair.ref0 * 16 + pair.ref1;
    if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;
    seen.push_back(key);
    pairs.push_back(pair);
  }
  return pairs;
}

int ReferenceSelectSyntax::reference_select_bits() const {
  // Cost proxy for the tile-level decision this frame-header bit enables
  // (one comp_mode symbol per inter block).  It is not header syntax.
  return reference_select ? 1 : 0;
}

int ReferenceSelectSyntax::reference_map_bits(int num_ref_frame_bits) const {
  // frame_explicit_ref_frame_map (1) + num_total_refs (3) + ref_frame_idx[]
  int bits = 1 + 3 + num_total_refs * num_ref_frame_bits;
  // reference_select (1) is coded with the other inter frame-header tools.
  bits += 1;
  return bits;
}

ReferenceSelectSyntax reference_select_syntax(const RefFrameMap& map) {
  ReferenceSelectSyntax syntax;
  syntax.reference_select = map.reference_select;
  syntax.num_total_refs = map.num_total_refs;
  for (int i = 0; i < map.num_total_refs; ++i) {
    syntax.ref_frame_idx[i] = map.ref_frame_idx[i];
  }
  return syntax;
}

}  // namespace av2enc::video::multiref
