#include "codec/inter/tlayers/tlayers.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>

namespace av2enc::inter::tlayers {
namespace {

bool is_power_of_two(int value) { return value > 1 && (value & (value - 1)) == 0; }

// A block size may be 1 even though 1 is not a "power of two" under the helper
// above (which excludes 1 so that degenerate sizes are rejected elsewhere).  A
// 1-picture block is the legal case for a clip whose frame count is one more
// than a whole number of 8-frame blocks: build_b_block() handles
// `block_size == 1` as a key-only anchor with no interior B picture.  Keep the
// two notions separate rather than loosening is_power_of_two(), which other
// callers rely on to reject 1.
bool is_dyadic_block(int value) {
  return value == 1 || is_power_of_two(value);
}

// §7.23: a key frame becomes valid only in the first refreshed slot.
int first_refreshed_slot(std::uint32_t flags) {
  int slot = 0;
  while (slot < 32 && ((flags >> slot) & 1U) == 0U) {
    ++slot;
  }
  return slot;
}

// §5.18.2 get_disp_order_hint(): lifts the signalled LSBs to an absolute hint
// using the maximum hint currently held by a showable, dependency-mapped
// reference slot.
int display_order_hint_from_lsbs(int order_hint_lsbs, int order_hint_bits,
                                 int max_disp_order_hint) {
  if (order_hint_bits <= 0) {
    return order_hint_lsbs;
  }
  const int span = 1 << order_hint_bits;
  int disp = order_hint_lsbs;
  const int offset = max_disp_order_hint - (span >> 1) - order_hint_lsbs;
  if (offset >= 0) {
    disp += ((offset >> order_hint_bits) + 1) << order_hint_bits;
  }
  return disp;
}

void set_error(PlanError* out, PlanError value) {
  if (out != nullptr) {
    *out = value;
  }
}

bool checked_power_of_two(int n, PlanError* error) {
  if (!is_power_of_two(n)) {
    set_error(error, PlanError::kBadGopSize);
    return false;
  }
  return true;
}

// Builds the display -> coding-order schedule. `layer` and `refs` are filled
// per display index. A chained plan (`num_gops > 1` and/or a tail block)
// concatenates one hierarchical-P schedule per block under a single key.
bool build_schedule(const PlanConfig& config, std::vector<int>& coding_order,
                    std::vector<int>& layer, std::vector<std::vector<int>>& refs,
                    std::vector<int>& gop_starts, std::vector<int>& block_of,
                    PlanError* error) {
  const int n = config.gop_size;
  // A one-picture stream is a lone key: no B block, no interior picture, no
  // references.  It is the still-image case of the chained form (and what
  // --nightly-b reduces to for --frames 1), so accept it ahead of the dyadic
  // block checks, which exist to constrain past/future pyramids.
  if (config.num_gops == 1 && n == 1 && config.tail_gop_size == 0) {
    coding_order = {0};
    layer.assign(1, 0);
    refs.assign(1, {});
    block_of.assign(1, 0);
    // build_plan() asserts gop_starts.size() == num_gops + (tail ? 1 : 0), so
    // the lone-key plan contributes exactly one block start (at display 0).
    gop_starts.assign(1, 0);
    return true;
  }
  if (config.schedule == Schedule::kHierarchicalB) {
    if (!checked_power_of_two(n, error)) {
      return false;
    }
    if (n > 8) {
      // A 16-frame B pyramid needs temporal layer 4, but max_tlayer_id is
      // f(2) and can only represent 0..3.
      set_error(error, PlanError::kTooManyLayers);
      return false;
    }
    if (config.num_gops < 1) {
      set_error(error, PlanError::kBadGopSize);
      return false;
    }
    if (config.tail_gop_size != 0 &&
        (!is_dyadic_block(config.tail_gop_size) ||
         config.tail_gop_size > n)) {
      set_error(error, PlanError::kBadGopSize);
      return false;
    }
    const int total = n * config.num_gops + config.tail_gop_size;
    layer.assign(static_cast<std::size_t>(total), 0);
    refs.assign(static_cast<std::size_t>(total), {});
    block_of.assign(static_cast<std::size_t>(total), 0);
    gop_starts.clear();
    coding_order.clear();
    // One dyadic B block. The block's past anchor is local 0 (the key in the
    // first block); its future anchor is local block_size-1. In a chained
    // block the past anchor references the previous block's future anchor and
    // the future anchor references the past anchor, so the layer-0 chain is
    // key -> future0 -> past1 -> future1 -> ... with no CLK after the first.
    const auto build_b_block = [&](int block_size, int offset,
                                   int block_index) {
      const int future = block_size - 1;
      if (block_size == 1) {
        // A 1-picture block is a single anchor with no interior B picture:
        // there is no future anchor and no past/future pair, so the whole
        // block is the past anchor.  It must still reference the previous
        // block (never be a second key), which keeps the layer-0 chain going
        // and leaves every picture coded exactly once.
        block_of[static_cast<std::size_t>(offset)] = block_index;
        coding_order.push_back(offset);
        if (offset != 0) {
          refs[static_cast<std::size_t>(offset)] = {offset - 1};
        }
        return true;
      }
      for (int local = 0; local < block_size; ++local) {
        block_of[static_cast<std::size_t>(offset + local)] = block_index;
      }
      coding_order.push_back(offset);
      coding_order.push_back(offset + future);
      if (offset == 0) {
        // The only key: no references.
      } else {
        refs[static_cast<std::size_t>(offset)] = {offset - 1};
      }
      refs[static_cast<std::size_t>(offset + future)] = {offset};
      struct Interval {
        int lo;
        int hi;
        int layer;
      };
      std::vector<Interval> work;
      work.push_back(Interval{0, future, 1});
      for (std::size_t wi = 0; wi < work.size(); ++wi) {
        const Interval interval = work[wi];
        if (interval.hi - interval.lo <= 1) {
          continue;
        }
        const int mid = (interval.lo + interval.hi) / 2;
        layer[static_cast<std::size_t>(offset + mid)] = interval.layer;
        refs[static_cast<std::size_t>(offset + mid)] = {offset + interval.lo,
                                                        offset + interval.hi};
        coding_order.push_back(offset + mid);
        const bool splits_further =
            (mid - interval.lo > 1) || (interval.hi - mid > 1);
        if (splits_further && interval.layer + 1 > kMaxTemporalLayerId) {
          set_error(error, PlanError::kTooManyLayers);
          return false;
        }
        work.push_back(Interval{interval.lo, mid, interval.layer + 1});
        work.push_back(Interval{mid, interval.hi, interval.layer + 1});
      }
      return true;
    };
    for (int b = 0; b < config.num_gops; ++b) {
      gop_starts.push_back(b * n);
      if (!build_b_block(n, b * n, b)) {
        return false;
      }
    }
    if (config.tail_gop_size != 0) {
      gop_starts.push_back(config.num_gops * n);
      if (!build_b_block(config.tail_gop_size, config.num_gops * n,
                         config.num_gops)) {
        return false;
      }
    }
    return true;
  }

  if (config.schedule != Schedule::kHierarchicalP) {
    set_error(error, PlanError::kBadSchedule);
    return false;
  }
  if (config.num_gops < 1) {
    set_error(error, PlanError::kBadGopSize);
    return false;
  }
  if (!is_power_of_two(n) || n > 16) {
    set_error(error, PlanError::kBadGopSize);
    return false;
  }
  if (config.tail_gop_size != 0 &&
      (!is_power_of_two(config.tail_gop_size) ||
       config.tail_gop_size > config.gop_size)) {
    set_error(error, PlanError::kBadGopSize);
    return false;
  }

  // One coding order per block, concatenated in display order. The only key
  // is the first coded frame of the whole plan; every later block starts with
  // its layer-0 anchor.
  const auto build_block = [&](int block_size, int offset, int block_index) {
    std::vector<int> coding;
    std::vector<int> block_layer(static_cast<std::size_t>(block_size), 0);
    const int anchor = block_size / 2;
    coding.push_back(0);
    coding.push_back(anchor);
    struct Interval {
      int lo;
      int hi;  // exclusive
      int layer;
    };
    std::vector<Interval> work;
    work.push_back(Interval{0, anchor, 1});
    work.push_back(Interval{anchor, block_size, 1});
    for (std::size_t wi = 0; wi < work.size(); ++wi) {
      const Interval interval = work[wi];
      if (interval.hi - interval.lo <= 1) {
        continue;
      }
      const int mid = (interval.lo + interval.hi) / 2;
      block_layer[static_cast<std::size_t>(mid)] = interval.layer;
      coding.push_back(mid);
      const bool splits_further =
          (mid - interval.lo > 1) || (interval.hi - mid > 1);
      if (splits_further && interval.layer + 1 > kMaxTemporalLayerId) {
        set_error(error, PlanError::kTooManyLayers);
        return false;
      }
      work.push_back(Interval{interval.lo, mid, interval.layer + 1});
      work.push_back(Interval{mid, interval.hi, interval.layer + 1});
    }
    for (const int local : coding) {
      coding_order.push_back(local + offset);
      layer[static_cast<std::size_t>(local + offset)] =
          block_layer[static_cast<std::size_t>(local)];
      block_of[static_cast<std::size_t>(local + offset)] = block_index;
    }
    return true;
  };

  const int total = n * config.num_gops + config.tail_gop_size;
  layer.assign(static_cast<std::size_t>(total), 0);
  refs.assign(static_cast<std::size_t>(total), {});
  block_of.assign(static_cast<std::size_t>(total), 0);
  gop_starts.clear();
  coding_order.clear();
  for (int b = 0; b < config.num_gops; ++b) {
    gop_starts.push_back(b * n);
    if (!build_block(n, b * n, b)) {
      return false;
    }
  }
  if (config.tail_gop_size != 0) {
    gop_starts.push_back(config.num_gops * n);
    if (!build_block(config.tail_gop_size, config.num_gops * n,
                     config.num_gops)) {
      return false;
    }
  }
  if (config.ref_mode == RefMode::kSingle) {
    // Single-reference carrier: every frame references the most recently
    // coded picture whose temporal layer is <= its own. Inside a block that
    // is the immediately preceding coding-order frame; the first frame of a
    // later block picks that block's predecessor anchor (layer 0), which is
    // the only legal cross-block reference under TLayerDependencyMap.
    for (std::size_t c = 1; c < coding_order.size(); ++c) {
      const int display = coding_order[c];
      const int layer_id = layer[static_cast<std::size_t>(display)];
      for (std::ptrdiff_t back = static_cast<std::ptrdiff_t>(c) - 1; back >= 0;
           --back) {
        const int cand = coding_order[static_cast<std::size_t>(back)];
        if (layer[static_cast<std::size_t>(cand)] <= layer_id) {
          refs[static_cast<std::size_t>(display)] = {cand};
          break;
        }
      }
      if (refs[static_cast<std::size_t>(display)].empty()) {
        set_error(error, PlanError::kReferenceNotDecoded);
        return false;
      }
    }
    return true;
  }
  if (config.ref_mode != RefMode::kTwo) {
    set_error(error, PlanError::kBadSchedule);
    return false;
  }
  // Two-reference carrier.  For each frame pick the two nearest pictures that
  // are already in coding order and whose temporal layer is <= this frame's
  // layer.  Sorting by (layer, |display distance|, display) is deterministic
  // and keeps every reference inside the default TLayerDependencyMap.
  std::vector<int> coding_index(static_cast<std::size_t>(total), -1);
  for (std::size_t c = 0; c < coding_order.size(); ++c) {
    coding_index[static_cast<std::size_t>(coding_order[c])] =
        static_cast<int>(c);
  }
  for (std::size_t c = 1; c < coding_order.size(); ++c) {
    const int display = coding_order[c];
    const int layer_id = layer[static_cast<std::size_t>(display)];
    std::vector<int> candidates;
    for (int e = 0; e < total; ++e) {
      if (coding_index[static_cast<std::size_t>(e)] >=
              static_cast<int>(c) ||
          e == display) {
        continue;
      }
      if (layer[static_cast<std::size_t>(e)] > layer_id) {
        continue;
      }
      candidates.push_back(e);
    }
    std::sort(candidates.begin(), candidates.end(),
              [&](int a, int b) {
                const int la = layer[static_cast<std::size_t>(a)];
                const int lb = layer[static_cast<std::size_t>(b)];
                if (la != lb) return la < lb;
                const int da = std::abs(a - display);
                const int db = std::abs(b - display);
                if (da != db) return da < db;
                return a < b;
              });
    std::vector<int>& frame_refs = refs[static_cast<std::size_t>(display)];
    const std::size_t take = std::min<std::size_t>(2, candidates.size());
    // Write them nearest-first in display distance so ref_frame_idx[0] is the
    // closest picture.
    frame_refs.assign(candidates.begin(), candidates.begin() +
                                              static_cast<std::ptrdiff_t>(take));
    std::sort(frame_refs.begin(), frame_refs.end(), [&](int a, int b) {
      return std::abs(a - display) < std::abs(b - display);
    });
  }
  return true;
}

bool build_dependency_map(const PlanConfig& config, const Plan& plan,
                          std::array<std::array<bool, kMaxTLayers>, kMaxTLayers>& out,
                          PlanError* error) {
  out = {};
  if (config.dependency == DependencyMode::kAllLower) {
    for (int c = 0; c < kMaxTLayers; ++c) {
      for (int r = 0; r < kMaxTLayers; ++r) {
        out[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] =
            (r <= c && c <= plan.max_tlayer_id);
      }
    }
    return true;
  }
  if (config.dependency != DependencyMode::kDirect) {
    set_error(error, PlanError::kInternal);
    return false;
  }
  // The implied base-layer dependency (§5.4.1 initialisation).
  out[0][0] = true;
  for (const PlannedFrame& f : plan.frames) {
    for (int i = 0; i < f.num_refs; ++i) {
      const int ref_display = f.ref_display[i];
      if (ref_display < 0 || ref_display >= plan.display_count()) {
        set_error(error, PlanError::kReferenceOutOfRange);
        return false;
      }
      const int ref_layer = plan.frames[static_cast<std::size_t>(ref_display)].tlayer_id;
      out[static_cast<std::size_t>(f.tlayer_id)][static_cast<std::size_t>(ref_layer)] = true;
    }
  }
  return true;
}

// §7.21 output simulation (single embedded layer). The pinned decoder must
// emit every picture exactly once, in display order, through the §7.21.3
// successive-hint path plus the §7.21.6 frame_buffers() drain and the
// end-of-stream flush. Mirrors the bframes OutputSim, which gates the same
// process for the B schedules; the chained plan is the first temporal-layer
// plan whose store is smaller than its display count.
bool simulate_output_order(const Plan& plan, PlanError* error) {
  const int count = plan.display_count();
  struct SlotMeta {
    bool valid = false;
    int frame = -1;
    int hint = 0;
    bool implicit = false;
  };
  std::array<SlotMeta, kNumRefFrames> meta{};
  std::vector<char> output_done(static_cast<std::size_t>(count), 0);
  std::vector<int> order;
  order.reserve(static_cast<std::size_t>(count));

  const auto eligible = [&](int s) {
    const SlotMeta& m = meta[static_cast<std::size_t>(s)];
    return m.valid && m.frame >= 0 && m.frame < count && m.implicit &&
           output_done[static_cast<std::size_t>(m.frame)] == 0;
  };
  const auto mark_output = [&](int s) {
    if (s < 0 || s >= plan.num_slots) return;
    SlotMeta& m = meta[static_cast<std::size_t>(s)];
    if (!m.valid || m.frame < 0 || m.frame >= count ||
        output_done[static_cast<std::size_t>(m.frame)] != 0) {
      return;
    }
    output_done[static_cast<std::size_t>(m.frame)] = 1;
    order.push_back(m.frame);
  };
  const auto output_successive = [&](int hint) {
    int current = hint;
    for (;;) {
      const int target = current + 1;
      bool made_output = false;
      for (int s = 0; s < plan.num_slots; ++s) {
        if (eligible(s) && meta[static_cast<std::size_t>(s)].hint == target) {
          mark_output(s);
          made_output = true;
        }
      }
      if (!made_output) break;
      current = target;
    }
  };
  const auto output_slot = [&](int s) {
    if (s < 0 || s >= plan.num_slots) return;
    const int hint = meta[static_cast<std::size_t>(s)].hint;
    const int frame = meta[static_cast<std::size_t>(s)].frame;
    mark_output(s);
    if (frame >= 0 && frame < count &&
        output_done[static_cast<std::size_t>(frame)] != 0) {
      output_successive(hint);
    }
  };
  const auto frame_buffers = [&](int ref_idx, int current_frame,
                                 int current_hint) {
    const int output_hint =
        ref_idx >= 0 ? meta[static_cast<std::size_t>(ref_idx)].hint
                     : current_hint;
    for (;;) {
      int best = -1;
      int best_hint = 0;
      for (int s = 0; s < plan.num_slots; ++s) {
        if (!eligible(s)) continue;
        const int hint = meta[static_cast<std::size_t>(s)].hint;
        if (hint >= output_hint) continue;
        if (best < 0 || hint < best_hint) {
          best = s;
          best_hint = hint;
        }
      }
      if (best < 0) break;
      output_slot(best);
    }
    if (ref_idx >= 0) {
      output_slot(ref_idx);
    } else if (current_frame >= 0 && current_frame < count &&
               output_done[static_cast<std::size_t>(current_frame)] == 0) {
      output_done[static_cast<std::size_t>(current_frame)] = 1;
      order.push_back(current_frame);
      output_successive(output_hint);
    }
  };

  for (const int display : plan.coding_order) {
    const PlannedFrame& f = plan.frames[static_cast<std::size_t>(display)];
    // §7.23: an eligible picture in a slot that is about to be overwritten is
    // output first.
    for (int s = 0; s < plan.num_slots; ++s) {
      if (((f.refresh_frame_flags >> s) & 1U) == 0U) continue;
      if (eligible(s)) {
        frame_buffers(s, -1, -1);
      }
    }
    for (int s = 0; s < plan.num_slots; ++s) {
      if (((f.refresh_frame_flags >> s) & 1U) == 0U) continue;
      SlotMeta& m = meta[static_cast<std::size_t>(s)];
      // §7.23: a key/dependent-switch frame marks only the first refreshed
      // slot valid.
      m.valid = !f.is_key || s == first_refreshed_slot(f.refresh_frame_flags);
      m.frame = m.valid ? display : -1;
      m.hint = static_cast<int>(f.display_order_hint);
      m.implicit = f.implicit_output_frame;
    }
    if (f.immediate_output_frame) {
      frame_buffers(-1, display, static_cast<int>(f.display_order_hint));
    }
  }
  for (;;) {
    int best = -1;
    int best_hint = 0;
    for (int s = 0; s < plan.num_slots; ++s) {
      if (!eligible(s)) continue;
      const int hint = meta[static_cast<std::size_t>(s)].hint;
      if (best < 0 || hint < best_hint) {
        best = s;
        best_hint = hint;
      }
    }
    if (best < 0) break;
    output_slot(best);
  }
  for (int d = 0; d < count; ++d) {
    if (output_done[static_cast<std::size_t>(d)] == 0) {
      set_error(error, PlanError::kOutputMissingFrame);
      return false;
    }
  }
  for (std::size_t i = 0; i < order.size(); ++i) {
    if (order[i] != static_cast<int>(i)) {
      set_error(error, PlanError::kOutputOrderMismatch);
      return false;
    }
  }
  return true;
}

}  // namespace

std::vector<int> Plan::coding_order_up_to_layer(int max_layer) const {
  std::vector<int> out;
  for (const int display : coding_order) {
    if (layer_of(display) <= max_layer) {
      out.push_back(display);
    }
  }
  return out;
}

// Slot allocation for one plan under a fixed reference-store size.  The plan's
// frames are rewritten in place (slot / ref_slot / order_hint_lsbs); returns
// false when the store is too small for the schedule's live-reference set.
//
// Separated from build_plan() so the minimum store size can be found by asking
// this allocator directly (minimum_slot_requirement()) instead of restating its
// recycling rule in a second estimate that could drift from it.
bool allocate_slots(Plan& plan, int num_slots, PlanError* error) {
  const int total = plan.display_count();
    // Slot allocation: prefer an empty slot, otherwise recycle the lowest slot
    // whose frame is no longer referenced by any later-coded frame.
    std::vector<int> last_use(static_cast<std::size_t>(total), -1);
    for (std::size_t c = 0; c < plan.coding_order.size(); ++c) {
      const PlannedFrame& f = plan.frames[static_cast<std::size_t>(plan.coding_order[c])];
      for (int i = 0; i < f.num_refs; ++i) {
        const int d = f.ref_display[i];
        if (d >= 0 && d < total) {
          last_use[static_cast<std::size_t>(d)] =
              std::max(last_use[static_cast<std::size_t>(d)], static_cast<int>(c));
        }
      }
    }
    std::array<int, kNumRefFrames> occupant{};
    occupant.fill(-1);
    std::vector<int> slot_of_display(static_cast<std::size_t>(total), -1);
    // Display-order frontier: before coding position c, the smallest display
    // index that has not been coded yet. Recycling is only safe when the
    // occupant is strictly below the frontier: every lower picture is already
    // coded, so the §7.21 process can drain them in display order before the
    // overwrite. (The pinned dav2d does not run that drain for a non-key
    // refresh; avmdec does. See docs/validation.md.)
    std::vector<int> frontier(plan.coding_order.size(), 0);
    {
      std::vector<int> coding_index(static_cast<std::size_t>(total), -1);
      for (std::size_t c = 0; c < plan.coding_order.size(); ++c) {
        coding_index[static_cast<std::size_t>(plan.coding_order[c])] =
            static_cast<int>(c);
      }
      int next = 0;
      for (std::size_t c = 0; c < plan.coding_order.size(); ++c) {
        while (next < total &&
               coding_index[static_cast<std::size_t>(next)] <
                   static_cast<int>(c)) {
          ++next;
        }
        frontier[c] = next;
      }
    }
    for (std::size_t c = 0; c < plan.coding_order.size(); ++c) {
      const int display = plan.coding_order[c];
      PlannedFrame& f = plan.frames[static_cast<std::size_t>(display)];
      int chosen = -1;
      // Prefer an empty slot: an implicitly output frame must stay in the store
      // until the decoder emits it, so recycling is only used when the store is
      // full. In that case pick the lowest display index below the coding
      // frontier whose frame is no longer referenced by any later-coded frame.
      for (int s = 0; s < num_slots; ++s) {
        if (occupant[static_cast<std::size_t>(s)] == -1) {
          chosen = s;
          break;
        }
      }
      if (chosen < 0) {
        int best_display = -1;
        for (int s = 0; s < num_slots; ++s) {
          const int occ = occupant[static_cast<std::size_t>(s)];
          if (occ < 0 || occ >= frontier[c] ||
              last_use[static_cast<std::size_t>(occ)] > static_cast<int>(c)) {
            continue;
          }
          if (best_display < 0 || occ < best_display) {
            best_display = occ;
            chosen = s;
          }
        }
      }
      if (chosen < 0) {
        // The store is too small for this schedule's live-reference set.
        // Callers size it from minimum_slot_requirement(); report the deficit
        // here so a caller that did not does not have to bisect the count.
        set_error(error, PlanError::kNotEnoughSlots);
        return false;
      }
      f.slot = chosen;
      occupant[static_cast<std::size_t>(chosen)] = display;
      slot_of_display[static_cast<std::size_t>(display)] = chosen;

      for (int i = 0; i < f.num_refs; ++i) {
        const int d = f.ref_display[i];
        f.ref_slot[i] = slot_of_display[static_cast<std::size_t>(d)];
        if (f.ref_slot[i] < 0) {
          set_error(error, PlanError::kReferenceNotDecoded);
          return false;
        }
        f.ref_order_hint[i] =
            plan.frames[static_cast<std::size_t>(d)].display_order_hint;
      }

      if (f.is_key) {
        // CLK refresh invalidates every slot except the first refreshed one.
        occupant.fill(-1);
        occupant[static_cast<std::size_t>(chosen)] = display;
        f.refresh_frame_flags = (1U << num_slots) - 1U;
      } else {
        f.refresh_frame_flags = 1U << chosen;
      }
    }

    // Derive the order-hint LSBs with the §5.18.2 replay. The reference store is
    // the only state get_disp_order_hint() consults.
    {
      std::array<int, kNumRefFrames> hint{};
      std::array<int, kNumRefFrames> layer_of_slot{};
      std::array<bool, kNumRefFrames> valid{};
      std::array<bool, kNumRefFrames> showable{};
      for (std::size_t c = 0; c < plan.coding_order.size(); ++c) {
        PlannedFrame& f = plan.frames[static_cast<std::size_t>(plan.coding_order[c])];
        int max_disp = 0;
        if (!f.is_key) {
          for (int s = 0; s < num_slots; ++s) {
            if (valid[static_cast<std::size_t>(s)] &&
                showable[static_cast<std::size_t>(s)] &&
                plan.tlayer_dependency[static_cast<std::size_t>(f.tlayer_id)]
                                       [static_cast<std::size_t>(
                                           layer_of_slot[static_cast<std::size_t>(s)])]) {
              max_disp = std::max(max_disp, hint[static_cast<std::size_t>(s)]);
            }
          }
        }
        const int span = 1 << plan.config.order_hint_bits;
        const std::uint32_t absolute = f.display_order_hint;
        if (absolute >= static_cast<std::uint32_t>(span)) {
          set_error(error, PlanError::kBadOrderHintBits);
          return false;
        }
        std::uint32_t lsbs = absolute;
        if (!f.is_key) {
          lsbs = static_cast<std::uint32_t>(
              static_cast<int>(absolute) & (span - 1));
          if (display_order_hint_from_lsbs(static_cast<int>(lsbs),
                                           plan.config.order_hint_bits, max_disp) !=
              static_cast<int>(absolute)) {
            set_error(error, PlanError::kOrderHintReconstruct);
            return false;
          }
        }
        f.order_hint_lsbs = lsbs;
        if (f.is_key) {
          // CLK refresh: only the first refreshed slot stays valid.
          const int key_slot = first_refreshed_slot(f.refresh_frame_flags);
          valid.fill(false);
          hint[static_cast<std::size_t>(key_slot)] =
              static_cast<int>(f.display_order_hint);
          layer_of_slot[static_cast<std::size_t>(key_slot)] = f.tlayer_id;
          showable[static_cast<std::size_t>(key_slot)] = true;
          valid[static_cast<std::size_t>(key_slot)] = true;
        } else {
          hint[static_cast<std::size_t>(f.slot)] =
              static_cast<int>(f.display_order_hint);
          layer_of_slot[static_cast<std::size_t>(f.slot)] = f.tlayer_id;
          showable[static_cast<std::size_t>(f.slot)] =
              f.implicit_output_frame || f.immediate_output_frame;
          valid[static_cast<std::size_t>(f.slot)] = true;
        }
      }
    }
  return true;
}

int minimum_slot_requirement(const PlanConfig& config) {
  for (int slots = 1; slots <= kNumRefFrames; ++slots) {
    PlanConfig probe_config = config;
    probe_config.num_slots = slots;
    Plan probe;
    PlanError probe_error = PlanError::kOk;
    if (build_plan(probe_config, probe, &probe_error)) return slots;
  }
  return 0;
}


bool build_plan(const PlanConfig& config, Plan& plan, PlanError* error) {
  set_error(error, PlanError::kOk);
  plan = Plan{};
  plan.config = config;
  const int n = config.gop_size;
  // gop_size 1 is legal only for the one-picture (lone key) plan.
  const bool single = config.num_gops == 1 && n == 1 &&
                      config.tail_gop_size == 0;
  if ((!single && !is_power_of_two(n)) || n > 16) {
    set_error(error, PlanError::kBadGopSize);
    return false;
  }
  if (config.num_gops < 1) {
    set_error(error, PlanError::kBadGopSize);
    return false;
  }
  if (config.tail_gop_size != 0 &&
      (!is_dyadic_block(config.tail_gop_size) ||
       config.tail_gop_size > n)) {
    set_error(error, PlanError::kBadGopSize);
    return false;
  }
  if (config.order_hint_bits < 1 || config.order_hint_bits > 16) {
    set_error(error, PlanError::kBadOrderHintBits);
    return false;
  }
  if (config.start_order_hint < 0) {
    set_error(error, PlanError::kBadStartOrderHint);
    return false;
  }
  if (config.num_slots < 1 || config.num_slots > kNumRefFrames) {
    set_error(error, PlanError::kBadSlotCount);
    return false;
  }

  const int total = n * config.num_gops + config.tail_gop_size;
  std::vector<int> coding_order;
  std::vector<int> layer;
  std::vector<std::vector<int>> refs;
  std::vector<int> gop_starts;
  std::vector<int> block_of;
  if (!build_schedule(config, coding_order, layer, refs, gop_starts, block_of,
                      error)) {
    return false;
  }
  if (static_cast<int>(coding_order.size()) != total ||
      static_cast<int>(gop_starts.size()) !=
          config.num_gops + (config.tail_gop_size != 0 ? 1 : 0)) {
    set_error(error, PlanError::kInternal);
    return false;
  }

  plan.frames.assign(static_cast<std::size_t>(total), PlannedFrame{});
  plan.coding_order = coding_order;
  plan.num_slots = config.num_slots;
  plan.num_gops = config.num_gops;
  plan.tail_gop_size = config.tail_gop_size;
  plan.gop_starts = gop_starts;

  for (int c = 0; c < total; ++c) {
    const int display = coding_order[static_cast<std::size_t>(c)];
    PlannedFrame& f = plan.frames[static_cast<std::size_t>(display)];
    f.display_index = display;
    f.coding_index = c;
    f.tlayer_id = layer[static_cast<std::size_t>(display)];
    f.gop_index = block_of[static_cast<std::size_t>(display)];
    f.is_key = (c == 0);
    f.is_anchor = (f.tlayer_id == 0 && !f.is_key);
    f.display_order_hint =
        static_cast<std::uint32_t>(config.start_order_hint + display);
    f.num_refs = static_cast<int>(refs[static_cast<std::size_t>(display)].size());
    for (int i = 0; i < f.num_refs; ++i) {
      f.ref_display[i] = refs[static_cast<std::size_t>(display)][static_cast<std::size_t>(i)];
    }
    // Output flags. The first picture is immediate; every later picture is
    // implicit (monotonic_output_order_flag = 0). A chained plan is a normal
    // implicit-output stream: the §7.21 output process emits it in display
    // order. The pinned dav2d f9dafbc does not run the §7.21.6 drain for a
    // non-key refresh that overwrites an implicit picture (a single-key
    // 60-picture stream emits only the final store contents); chained streams
    // are therefore validated with avmdec (docs/validation.md) until the pin
    // moves. Do not make a block anchor immediate: that constrains the
    // output floor and costs schedule freedom for a decoder defect.
    f.immediate_output_frame = (c == 0);
    f.implicit_output_frame = (c != 0);
  }

  // Layer statistics and max_tlayer_id.
  plan.max_tlayer_id = 0;
  for (const PlannedFrame& f : plan.frames) {
    plan.max_tlayer_id = std::max(plan.max_tlayer_id, f.tlayer_id);
    plan.layer_frame_count[static_cast<std::size_t>(f.tlayer_id)] += 1;
  }
  plan.num_layers = plan.max_tlayer_id + 1;
  if (plan.max_tlayer_id > kMaxTemporalLayerId) {
    set_error(error, PlanError::kTooManyLayers);
    return false;
  }

  // Direct dependency map, then order-hint simulation (which needs the map).
  if (!build_dependency_map(config, plan, plan.tlayer_dependency, error)) {
    return false;
  }
  plan.tlayer_dependency_present_flag =
      (config.explicit_dependency_map && plan.max_tlayer_id > 0);
  if (!config.explicit_dependency_map) {
    // §5.4.1 default: TLayerDependencyMap[c][r] = (r <= c). The plan must be
    // drop-safe under exactly that map.
    for (int c = 0; c <= plan.max_tlayer_id; ++c) {
      for (int r = 0; r <= c; ++r) {
        if (!plan.tlayer_dependency[static_cast<std::size_t>(c)]
                                   [static_cast<std::size_t>(r)]) {
          set_error(error, PlanError::kDependencyMapIncomplete);
          return false;
        }
      }
    }
  }


  if (!allocate_slots(plan, plan.num_slots, error)) return false;

  PlanError validation_error = PlanError::kOk;
  int bad_display = -1;
  int bad_coding = -1;
  if (!validate_plan(plan, &validation_error, &bad_display, &bad_coding)) {
    set_error(error, validation_error);
    return false;
  }
  return true;
}

bool validate_plan(const Plan& plan, PlanError* error, int* bad_display_index,
                   int* bad_coding_index) {
  set_error(error, PlanError::kOk);
  if (bad_display_index != nullptr) {
    *bad_display_index = -1;
  }
  if (bad_coding_index != nullptr) {
    *bad_coding_index = -1;
  }
  const auto fail = [&](PlanError code, int display, int coding) {
    set_error(error, code);
    if (bad_display_index != nullptr) {
      *bad_display_index = display;
    }
    if (bad_coding_index != nullptr) {
      *bad_coding_index = coding;
    }
    return false;
  };

  const PlanConfig& c = plan.config;
  const int n = plan.display_count();
  const int expected_frames = c.gop_size * c.num_gops + c.tail_gop_size;
  const bool single_plan = c.num_gops == 1 && c.gop_size == 1 &&
                           c.tail_gop_size == 0;
  if ((!single_plan && !is_power_of_two(c.gop_size)) || c.gop_size > 16 ||
      n != expected_frames || c.num_gops < 1 ||
      (c.tail_gop_size != 0 &&
       (!is_dyadic_block(c.tail_gop_size) || c.tail_gop_size > c.gop_size)) ||
      plan.num_gops != c.num_gops || plan.tail_gop_size != c.tail_gop_size) {
    return fail(PlanError::kBadGopSize, -1, -1);
  }
  if (c.order_hint_bits < 1 || c.order_hint_bits > 16) {
    return fail(PlanError::kBadOrderHintBits, -1, -1);
  }
  if (c.start_order_hint < 0) {
    return fail(PlanError::kBadStartOrderHint, -1, -1);
  }
  if (plan.num_slots < 1 || plan.num_slots > kNumRefFrames ||
      c.num_slots != plan.num_slots) {
    return fail(PlanError::kBadSlotCount, -1, -1);
  }
  if (plan.max_tlayer_id < 0 || plan.max_tlayer_id > kMaxTemporalLayerId ||
      plan.num_layers != plan.max_tlayer_id + 1) {
    return fail(PlanError::kTooManyLayers, -1, -1);
  }
  if (static_cast<int>(plan.coding_order.size()) != n) {
    return fail(PlanError::kBadCodingOrder, -1, -1);
  }

  // Coding order must be a permutation and the layer frame counts consistent.
  std::vector<int> seen(static_cast<std::size_t>(n), 0);
  int layer_count[kMaxTLayers] = {0, 0, 0, 0};
  for (std::size_t ci = 0; ci < plan.coding_order.size(); ++ci) {
    const int d = plan.coding_order[ci];
    if (d < 0 || d >= n || seen[static_cast<std::size_t>(d)] != 0) {
      return fail(PlanError::kBadCodingOrder, d, static_cast<int>(ci));
    }
    seen[static_cast<std::size_t>(d)] = 1;
    const PlannedFrame& f = plan.frames[static_cast<std::size_t>(d)];
    if (f.display_index != d || f.coding_index != static_cast<int>(ci)) {
      return fail(PlanError::kBadDisplayOrder, d, static_cast<int>(ci));
    }
    if (f.display_order_hint !=
        static_cast<std::uint32_t>(c.start_order_hint + d)) {
      return fail(PlanError::kBadDisplayOrder, d, static_cast<int>(ci));
    }
    if (f.tlayer_id < 0 || f.tlayer_id > plan.max_tlayer_id) {
      return fail(PlanError::kReferenceLayerViolation, d, static_cast<int>(ci));
    }
    layer_count[static_cast<std::size_t>(f.tlayer_id)] += 1;
    if (ci == 0) {
      if (!f.is_key || f.num_refs != 0) {
        return fail(PlanError::kKeyFrameHasRefs, d, static_cast<int>(ci));
      }
    } else if (f.is_key) {
      return fail(PlanError::kMissingKeyFrame, d, static_cast<int>(ci));
    }
  }
  for (int l = 0; l < kMaxTLayers; ++l) {
    if (layer_count[l] != plan.layer_frame_count[l]) {
      return fail(PlanError::kInternal, -1, -1);
    }
  }

  // Dependency map shape: [0][0] is implied; no ref > curr.
  for (int curr = 0; curr < kMaxTLayers; ++curr) {
    for (int ref = 0; ref < kMaxTLayers; ++ref) {
      if (ref > curr && plan.tlayer_dependency[static_cast<std::size_t>(curr)]
                                                [static_cast<std::size_t>(ref)]) {
        return fail(PlanError::kDependencyMapIncomplete, -1, -1);
      }
    }
  }
  if (!plan.tlayer_dependency[0][0]) {
    return fail(PlanError::kDependencyMapIncomplete, -1, -1);
  }
  if (plan.max_tlayer_id == 0 && plan.tlayer_dependency_present_flag) {
    return fail(PlanError::kDependencyMapIncomplete, -1, -1);
  }
  if (plan.max_tlayer_id > 0 &&
      plan.tlayer_dependency_present_flag != c.explicit_dependency_map) {
    return fail(PlanError::kDependencyMapIncomplete, -1, -1);
  }
  if (!plan.tlayer_dependency_present_flag) {
    for (int curr = 0; curr <= plan.max_tlayer_id; ++curr) {
      for (int ref = 0; ref <= curr; ++ref) {
        if (!plan.tlayer_dependency[static_cast<std::size_t>(curr)]
                                   [static_cast<std::size_t>(ref)]) {
          return fail(PlanError::kDependencyMapIncomplete, -1, -1);
        }
      }
    }
  }
  if (plan.multi_tlayer_dependency_map_present_flag) {
    // max_mlayer_id == 0. §5.4.1 infers the flag as 0; it is never coded.
    return fail(PlanError::kInternal, -1, -1);
  }

  // Per-frame checks in coding order.
  std::vector<int> decoded(static_cast<std::size_t>(n), 0);
  std::array<int, kNumRefFrames> slot_display{};
  slot_display.fill(-1);
  for (std::size_t ci = 0; ci < plan.coding_order.size(); ++ci) {
    const int d = plan.coding_order[ci];
    const PlannedFrame& f = plan.frames[static_cast<std::size_t>(d)];
    if (f.refresh_frame_flags == 0U) {
      return fail(PlanError::kRefreshFlagsInvalid, d, static_cast<int>(ci));
    }
    if (f.is_key) {
      const std::uint32_t all_frames = (1U << plan.num_slots) - 1U;
      if (f.refresh_frame_flags != all_frames) {
        return fail(PlanError::kRefreshFlagsInvalid, d, static_cast<int>(ci));
      }
    } else {
      if ((f.refresh_frame_flags & (f.refresh_frame_flags - 1U)) != 0U) {
        return fail(PlanError::kRefreshFlagsInvalid, d, static_cast<int>(ci));
      }
      int bit = 0;
      while (((f.refresh_frame_flags >> bit) & 1U) == 0U) {
        ++bit;
      }
      if (f.slot != bit || bit >= plan.num_slots) {
        return fail(PlanError::kRefreshFlagsInvalid, d, static_cast<int>(ci));
      }
    }
    if (f.slot < 0 || f.slot >= plan.num_slots) {
      return fail(PlanError::kRefreshFlagsInvalid, d, static_cast<int>(ci));
    }
    if (f.immediate_output_frame == f.implicit_output_frame) {
      return fail(PlanError::kOutputFlagsInvalid, d, static_cast<int>(ci));
    }
    if (ci == 0 && !f.immediate_output_frame) {
      return fail(PlanError::kOutputFlagsInvalid, d, static_cast<int>(ci));
    }
    if (f.num_refs < 0 || f.num_refs > kMaxRefs) {
      return fail(PlanError::kDuplicateReference, d, static_cast<int>(ci));
    }
    for (int i = 0; i < f.num_refs; ++i) {
      const int rd = f.ref_display[i];
      if (rd < 0 || rd >= n) {
        return fail(PlanError::kReferenceOutOfRange, d, static_cast<int>(ci));
      }
      if (decoded[static_cast<std::size_t>(rd)] == 0) {
        return fail(PlanError::kReferenceNotDecoded, d, static_cast<int>(ci));
      }
      for (int j = 0; j < i; ++j) {
        if (f.ref_display[j] == rd || f.ref_slot[j] == f.ref_slot[i]) {
          return fail(PlanError::kDuplicateReference, d, static_cast<int>(ci));
        }
      }
      const int rl = plan.frames[static_cast<std::size_t>(rd)].tlayer_id;
      if (rl > f.tlayer_id) {
        return fail(PlanError::kReferenceLayerViolation, d, static_cast<int>(ci));
      }
      if (!plan.tlayer_dependency[static_cast<std::size_t>(f.tlayer_id)]
                                 [static_cast<std::size_t>(rl)]) {
        return fail(PlanError::kDependencyMapIncomplete, d, static_cast<int>(ci));
      }
      if (f.ref_slot[i] < 0 || f.ref_slot[i] >= plan.num_slots ||
          slot_display[static_cast<std::size_t>(f.ref_slot[i])] != rd) {
        return fail(PlanError::kReferenceSlotInvalid, d, static_cast<int>(ci));
      }
      if (f.ref_order_hint[i] !=
          plan.frames[static_cast<std::size_t>(rd)].display_order_hint) {
        return fail(PlanError::kOrderHintReconstruct, d, static_cast<int>(ci));
      }
    }
    // Apply §7.23.
    if (f.is_key) {
      slot_display.fill(-1);
      slot_display[static_cast<std::size_t>(
          first_refreshed_slot(f.refresh_frame_flags))] = d;
    } else {
      slot_display[static_cast<std::size_t>(f.slot)] = d;
    }
    decoded[static_cast<std::size_t>(d)] = 1;
  }

  // Full order-hint replay (recomputes the stored LSBs and compares).
  {
    // Recompute LSBs with the same replay used by build_plan and compare.
    std::array<int, kNumRefFrames> hint{};
    std::array<int, kNumRefFrames> layer_of_slot{};
    std::array<bool, kNumRefFrames> valid{};
    std::array<bool, kNumRefFrames> showable{};
    for (std::size_t ci = 0; ci < plan.coding_order.size(); ++ci) {
      const PlannedFrame& f = plan.frames[static_cast<std::size_t>(plan.coding_order[ci])];
      int max_disp = 0;
      if (!f.is_key) {
        for (int s = 0; s < plan.num_slots; ++s) {
          if (valid[static_cast<std::size_t>(s)] &&
              showable[static_cast<std::size_t>(s)] &&
              plan.tlayer_dependency[static_cast<std::size_t>(f.tlayer_id)]
                                     [static_cast<std::size_t>(
                                         layer_of_slot[static_cast<std::size_t>(s)])]) {
            max_disp = std::max(max_disp, hint[static_cast<std::size_t>(s)]);
          }
        }
      }
      const int span = 1 << plan.config.order_hint_bits;
      if (f.display_order_hint >= static_cast<std::uint32_t>(span)) {
        return fail(PlanError::kBadOrderHintBits, f.display_index,
                    f.coding_index);
      }
      std::uint32_t expected = f.display_order_hint;
      if (!f.is_key) {
        bool found = false;
        for (int cand = 0; cand < span; ++cand) {
          if (display_order_hint_from_lsbs(cand, plan.config.order_hint_bits,
                                           max_disp) ==
              static_cast<int>(f.display_order_hint)) {
            expected = static_cast<std::uint32_t>(cand);
            found = true;
            break;
          }
        }
        if (!found) {
          return fail(PlanError::kOrderHintReconstruct, f.display_index,
                      f.coding_index);
        }
      }
      if (f.order_hint_lsbs != expected) {
        return fail(PlanError::kOrderHintReconstruct, f.display_index,
                    f.coding_index);
      }
      if (f.is_key) {
        const int key_slot = first_refreshed_slot(f.refresh_frame_flags);
        valid.fill(false);
        hint[static_cast<std::size_t>(key_slot)] =
            static_cast<int>(f.display_order_hint);
        layer_of_slot[static_cast<std::size_t>(key_slot)] = f.tlayer_id;
        showable[static_cast<std::size_t>(key_slot)] = true;
        valid[static_cast<std::size_t>(key_slot)] = true;
      } else {
        hint[static_cast<std::size_t>(f.slot)] =
            static_cast<int>(f.display_order_hint);
        layer_of_slot[static_cast<std::size_t>(f.slot)] = f.tlayer_id;
        showable[static_cast<std::size_t>(f.slot)] =
            f.implicit_output_frame || f.immediate_output_frame;
        valid[static_cast<std::size_t>(f.slot)] = true;
      }
    }
  }

  for (int max_layer = 0; max_layer <= plan.max_tlayer_id; ++max_layer) {
    PlanError drop_error = PlanError::kOk;
    if (!validate_layer_drop(plan, max_layer, &drop_error)) {
      return fail(drop_error, -1, -1);
    }
  }
  // §7.21: every picture must be output exactly once, in display order. This
  // is what the chained-GOP plan adds to the single-GOP carrier (the store is
  // now smaller than the display count for 60-frame inputs).
  PlanError output_error = PlanError::kOk;
  if (!simulate_output_order(plan, &output_error)) {
    return fail(output_error, -1, -1);
  }
  return true;
}

bool validate_layer_drop(const Plan& plan, int max_layer, PlanError* error) {
  set_error(error, PlanError::kOk);
  if (max_layer < 0 || max_layer > plan.max_tlayer_id) {
    set_error(error, PlanError::kReferenceLayerViolation);
    return false;
  }
  for (const int d : plan.coding_order) {
    const PlannedFrame& f = plan.frames[static_cast<std::size_t>(d)];
    if (f.tlayer_id > max_layer) {
      continue;
    }
    for (int i = 0; i < f.num_refs; ++i) {
      const int rd = f.ref_display[i];
      if (rd < 0 || rd >= plan.display_count()) {
        set_error(error, PlanError::kReferenceOutOfRange);
        return false;
      }
      const PlannedFrame& ref = plan.frames[static_cast<std::size_t>(rd)];
      if (ref.tlayer_id > max_layer) {
        // A retained frame would need a dropped frame: the sub-stream is not
        // independently decodable.
        set_error(error, PlanError::kReferenceLayerViolation);
        return false;
      }
      if (ref.coding_index >= f.coding_index) {
        set_error(error, PlanError::kReferenceNotDecoded);
        return false;
      }
    }
  }
  return true;
}

const char* plan_error_name(PlanError error) {
  switch (error) {
    case PlanError::kOk:
      return "ok";
    case PlanError::kBadGopSize:
      return "bad gop size";
    case PlanError::kBadSchedule:
      return "bad schedule";
    case PlanError::kBadOrderHintBits:
      return "bad order hint bits";
    case PlanError::kBadStartOrderHint:
      return "bad start order hint";
    case PlanError::kTooManyLayers:
      return "too many temporal layers";
    case PlanError::kBadSlotCount:
      return "bad slot count";
    case PlanError::kNotEnoughSlots:
      return "not enough reference slots";
    case PlanError::kReferenceOutOfRange:
      return "reference out of range";
    case PlanError::kReferenceNotDecoded:
      return "reference not decoded yet";
    case PlanError::kReferenceLayerViolation:
      return "reference violates the temporal layer order";
    case PlanError::kReferenceSlotInvalid:
      return "reference slot does not hold the reference";
    case PlanError::kDuplicateReference:
      return "duplicate reference";
    case PlanError::kKeyFrameHasRefs:
      return "key frame carries references";
    case PlanError::kMissingKeyFrame:
      return "missing or misplaced key frame";
    case PlanError::kRefreshFlagsInvalid:
      return "invalid refresh_frame_flags";
    case PlanError::kOutputFlagsInvalid:
      return "invalid output flags";
    case PlanError::kOrderHintReconstruct:
      return "order hint LSBs do not reconstruct";
    case PlanError::kDependencyMapIncomplete:
      return "TLayerDependencyMap does not cover a reference";
    case PlanError::kBadCodingOrder:
      return "bad coding order";
    case PlanError::kBadDisplayOrder:
      return "bad display order";
    case PlanError::kOutputMissingFrame:
      return "output is missing a picture";
    case PlanError::kOutputOrderMismatch:
      return "output order is not display order";
    case PlanError::kInternal:
      return "internal invariant";
  }
  return "unknown";
}

const char* schedule_name(Schedule schedule) {
  switch (schedule) {
    case Schedule::kHierarchicalP:
      return "hierarchical-p";
    case Schedule::kHierarchicalB:
      return "hierarchical-b";
  }
  return "unknown";
}

const char* dependency_mode_name(DependencyMode mode) {
  switch (mode) {
    case DependencyMode::kDirect:
      return "direct";
    case DependencyMode::kAllLower:
      return "all-lower";
  }
  return "unknown";
}

bool chained_gops_enabled() {
  const char* value = std::getenv("AV2ENC_TLAYERS_CHAIN");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  // Default-off: only an explicit non-zero value enables the chained plan.
  return value[0] != '0';
}

}  // namespace av2enc::inter::tlayers
