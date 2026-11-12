#include "codec/inter/tlayers/tlayers.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace av2enc::inter::tlayers {
namespace {

bool is_power_of_two(int value) { return value > 1 && (value & (value - 1)) == 0; }

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
// per display index.
bool build_schedule(const PlanConfig& config, std::vector<int>& coding_order,
                    std::vector<int>& layer, std::vector<std::vector<int>>& refs,
                    PlanError* error) {
  const int n = config.gop_size;
  if (!checked_power_of_two(n, error)) {
    return false;
  }
  if (config.schedule == Schedule::kHierarchicalB) {
    if (n > 8) {
      // A 16-frame B pyramid needs temporal layer 4, but max_tlayer_id is
      // f(2) and can only represent 0..3.
      set_error(error, PlanError::kTooManyLayers);
      return false;
    }
    layer.assign(static_cast<std::size_t>(n), 0);
    refs.assign(static_cast<std::size_t>(n), {});
    coding_order.clear();
    coding_order.push_back(0);
    coding_order.push_back(n - 1);
    refs[static_cast<std::size_t>(n - 1)] = {0};
    struct Interval {
      int lo;
      int hi;
      int layer;
    };
    std::vector<Interval> work;
    work.push_back(Interval{0, n - 1, 1});
    for (std::size_t wi = 0; wi < work.size(); ++wi) {
      const Interval interval = work[wi];
      if (interval.hi - interval.lo <= 1) {
        continue;
      }
      const int mid = (interval.lo + interval.hi) / 2;
      layer[static_cast<std::size_t>(mid)] = interval.layer;
      refs[static_cast<std::size_t>(mid)] = {interval.lo, interval.hi};
      coding_order.push_back(mid);
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
  }

  if (config.schedule != Schedule::kHierarchicalP) {
    set_error(error, PlanError::kBadSchedule);
    return false;
  }
  if (n > 16) {
    set_error(error, PlanError::kTooManyLayers);
    return false;
  }
  layer.assign(static_cast<std::size_t>(n), 0);
  refs.assign(static_cast<std::size_t>(n), {});
  coding_order.clear();
  const int anchor = n / 2;
  coding_order.push_back(0);
  coding_order.push_back(anchor);
  refs[static_cast<std::size_t>(anchor)] = {0};
  struct Interval {
    int lo;
    int hi;  // exclusive; n is a valid one-past-the-end boundary
    int layer;
  };
  std::vector<Interval> work;
  work.push_back(Interval{0, anchor, 1});
  work.push_back(Interval{anchor, n, 1});
  for (std::size_t wi = 0; wi < work.size(); ++wi) {
    const Interval interval = work[wi];
    if (interval.hi - interval.lo <= 1) {
      continue;
    }
    const int mid = (interval.lo + interval.hi) / 2;
    layer[static_cast<std::size_t>(mid)] = interval.layer;
    coding_order.push_back(mid);
    const bool splits_further =
        (mid - interval.lo > 1) || (interval.hi - mid > 1);
    if (splits_further && interval.layer + 1 > kMaxTemporalLayerId) {
      set_error(error, PlanError::kTooManyLayers);
      return false;
    }
    work.push_back(Interval{interval.lo, mid, interval.layer + 1});
    work.push_back(Interval{mid, interval.hi, interval.layer + 1});
  }
  // Single-reference carrier: every frame after the anchor references the
  // immediately preceding frame in coding order. Lower layers are always
  // decoded before higher layers.
  for (std::size_t c = 2; c < coding_order.size(); ++c) {
    refs[static_cast<std::size_t>(coding_order[c])] = {coding_order[c - 1]};
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

bool build_plan(const PlanConfig& config, Plan& plan, PlanError* error) {
  set_error(error, PlanError::kOk);
  plan = Plan{};
  plan.config = config;
  const int n = config.gop_size;
  if (!is_power_of_two(n) || n > 16) {
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

  std::vector<int> coding_order;
  std::vector<int> layer;
  std::vector<std::vector<int>> refs;
  if (!build_schedule(config, coding_order, layer, refs, error)) {
    return false;
  }

  plan.frames.assign(static_cast<std::size_t>(n), PlannedFrame{});
  plan.coding_order = coding_order;
  plan.num_slots = config.num_slots;

  for (int c = 0; c < n; ++c) {
    const int display = coding_order[static_cast<std::size_t>(c)];
    PlannedFrame& f = plan.frames[static_cast<std::size_t>(display)];
    f.display_index = display;
    f.coding_index = c;
    f.tlayer_id = layer[static_cast<std::size_t>(display)];
    f.is_key = (c == 0);
    f.is_anchor = (f.tlayer_id == 0 && !f.is_key);
    f.display_order_hint =
        static_cast<std::uint32_t>(config.start_order_hint + display);
    f.num_refs = static_cast<int>(refs[static_cast<std::size_t>(display)].size());
    for (int i = 0; i < f.num_refs; ++i) {
      f.ref_display[i] = refs[static_cast<std::size_t>(display)][static_cast<std::size_t>(i)];
    }
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

  // Slot allocation: prefer an empty slot, otherwise recycle the lowest slot
  // whose frame is no longer referenced by any later-coded frame.
  std::vector<int> last_use(static_cast<std::size_t>(n), -1);
  for (std::size_t c = 0; c < plan.coding_order.size(); ++c) {
    const PlannedFrame& f = plan.frames[static_cast<std::size_t>(plan.coding_order[c])];
    for (int i = 0; i < f.num_refs; ++i) {
      const int d = f.ref_display[i];
      if (d >= 0 && d < n) {
        last_use[static_cast<std::size_t>(d)] =
            std::max(last_use[static_cast<std::size_t>(d)], static_cast<int>(c));
      }
    }
  }
  std::array<int, kNumRefFrames> occupant{};
  occupant.fill(-1);
  std::vector<int> slot_of_display(static_cast<std::size_t>(n), -1);
  for (std::size_t c = 0; c < plan.coding_order.size(); ++c) {
    const int display = plan.coding_order[c];
    PlannedFrame& f = plan.frames[static_cast<std::size_t>(display)];
    int chosen = -1;
    // Prefer an empty slot: an implicitly output frame must stay in the store
    // until the decoder emits it, so recycling is only used when the store is
    // full. In that case pick the lowest slot whose frame is no longer
    // referenced by any later-coded frame. Use num_slots >= display_count when
    // implicit output must retain every frame (the decoder gate does).
    for (int s = 0; s < plan.num_slots; ++s) {
      if (occupant[static_cast<std::size_t>(s)] == -1) {
        chosen = s;
        break;
      }
    }
    if (chosen < 0) {
      for (int s = 0; s < plan.num_slots; ++s) {
        const int occ = occupant[static_cast<std::size_t>(s)];
        if (occ >= 0 && last_use[static_cast<std::size_t>(occ)] <= static_cast<int>(c)) {
          chosen = s;
          break;
        }
      }
    }
    if (chosen < 0) {
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
      f.refresh_frame_flags = (1U << plan.num_slots) - 1U;
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
  if (!is_power_of_two(n) || n != c.gop_size || n > 16) {
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

}  // namespace av2enc::inter::tlayers
