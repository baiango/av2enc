#include "b_frame.h"

#include <algorithm>

namespace av2enc::inter::adv {
namespace {

inline constexpr int kDecayDistCap = 6;
inline constexpr std::int8_t kDistScoreLookup[kDecayDistCap + 1] = {0, 64, 96, 112, 120, 124, 126};
inline constexpr int kDistWeightBits = 6;
inline constexpr int kMaxInt = 0x7fffffff;

bool valid_ref_frame_size(const RefListInput& in, int slot) {
  if (!in.check_res) {
    return true;
  }
  const RefSlot& ref = in.state.slots[slot];
  return 2 * in.frame_width >= ref.width && 2 * in.frame_height >= ref.height &&
         in.frame_width <= 16 * ref.width && in.frame_height <= 16 * ref.height;
}

bool first_slot_with_ref(const RefSlotState& state, int i) {
  if (!state.slots[i].valid) {
    return false;
  }
  for (int j = 0; j < i; ++j) {
    if (state.slots[j].valid && state.slots[j].counter == state.slots[i].counter) {
      return false;
    }
  }
  return true;
}

int slot_layer_ok(const RefListInput& in, int slot) {
  // Single-layer streams accept everything. Multi-layer callers set the flag.
  if (!in.layer_dependency_ok) {
    return 0;
  }
  const RefSlot& ref = in.state.slots[slot];
  if (ref.mlayer_id > in.obu_mlayer_id) {
    return 0;
  }
  if (ref.tlayer_id > in.obu_tlayer_id && in.obu_tlayer_id >= 0) {
    // TLayerDependencyMap is not modelled beyond "no deeper temporal layer".
    return 0;
  }
  return 1;
}

struct RankedRef {
  int slot = 0;
  int score = 0;
  int order_hint = 0;
  int distance = 0;
  int base_q_idx = 0;
  int layer = 0;
};

}  // namespace

void refresh_reference_slots(RefSlotState* state, std::uint32_t refresh_flags, int order_hint,
                             int width, int height, int base_q_idx, bool key_frame, int tlayer_id,
                             int mlayer_id, int counter) {
  if (state == nullptr) {
    return;
  }
  const int new_counter = counter >= 0 ? counter : state->frame_counter + 1;
  state->frame_counter = new_counter;
  bool first = true;
  for (int i = 0; i < kNumRefFrames; ++i) {
    if (((refresh_flags >> i) & 1u) == 0u) {
      continue;
    }
    RefSlot& slot = state->slots[i];
    slot.valid = key_frame ? first : true;
    first = false;
    slot.counter = new_counter;
    slot.order_hint = order_hint;
    slot.width = width;
    slot.height = height;
    slot.base_q_idx = base_q_idx;
    slot.tlayer_id = tlayer_id;
    slot.mlayer_id = mlayer_id;
  }
}

RefListResult build_ref_frame_list(const RefListInput& in) {
  RefListResult out{};
  int map_order_hint[kNumRefFrames];
  int map_base_q[kNumRefFrames];
  int max_disp = 0;
  const int num_ref_frames =
      std::min(in.num_ref_frames > 0 ? in.num_ref_frames : kNumRefFrames, kNumRefFrames);
  for (int i = 0; i < kNumRefFrames; ++i) {
    map_order_hint[i] = -1;
    map_base_q[i] = 0;
  }
  for (int i = 0; i < num_ref_frames; ++i) {
    const RefSlot& slot = in.state.slots[i];
    if (!first_slot_with_ref(in.state, i) || slot.order_hint == kRestrictedOrderHint ||
        !slot_layer_ok(in, i) || (in.is_bridge && i != 0)) {
      continue;
    }
    if (valid_ref_frame_size(in, i)) {
      map_order_hint[i] = slot.order_hint;
    }
    map_base_q[i] = slot.base_q_idx;
    max_disp = std::max(max_disp, slot.order_hint);
  }

  RankedRef ranked[kNumRefFrames]{};
  int num_ranked = 0;
  int min_q = 0;
  int max_q = 0;
  for (int i = 0; i < num_ref_frames; ++i) {
    const int d = map_order_hint[i];
    if (d == -1) {
      continue;
    }
    const int q = map_base_q[i];
    const int disp_diff = get_relative_dist(in.order_hint, d);
    int t_dist = abs_int(disp_diff) + in.obu_mlayer_id - in.state.slots[i].mlayer_id;
    if (t_dist < 0) {
      t_dist = 0;
    }
    int score;
    if (max_disp > in.order_hint) {
      score = (t_dist << kDistWeightBits) + q;
    } else {
      const int decay_index = std::min(t_dist, kDecayDistCap);
      score = kDistScoreLookup[decay_index] + std::max(t_dist - kDecayDistCap, 0) + q;
    }
    const int ref_ratio =
        floor_log2(std::max(in.state.slots[i].width * in.state.slots[i].height, 1));
    score -= ref_ratio << 5;

    bool is_new = true;
    for (int r = 0; r < num_ranked; ++r) {
      if (ranked[r].order_hint == d && ranked[r].score == score &&
          ranked[r].layer == in.state.slots[i].mlayer_id) {
        is_new = false;
        break;
      }
    }
    if (!is_new) {
      continue;
    }
    RankedRef& entry = ranked[num_ranked];
    entry.slot = i;
    entry.score = score;
    entry.order_hint = d;
    entry.distance = disp_diff;
    entry.base_q_idx = q;
    entry.layer = in.state.slots[i].mlayer_id;
    if (num_ranked == 0) {
      min_q = q;
      max_q = q;
    } else {
      min_q = std::min(min_q, q);
      max_q = std::max(max_q, q);
    }
    ++num_ranked;
  }

  if (num_ranked > kRefsPerFrame) {
    const int q_thresh = (max_q + min_q + 1) / 2;
    int n_past = 0;
    int n_future = 0;
    int max_past_distance = 0;
    int max_future_distance = 0;
    int past_idx = 0;
    int future_idx = 0;
    for (int i = 0; i < num_ranked; ++i) {
      if (ranked[i].base_q_idx >= q_thresh) {
        const int d = ranked[i].distance;
        if (d > 0) {
          if (d > max_past_distance) {
            max_past_distance = d;
            past_idx = i;
          }
          ++n_past;
        } else if (d < 0) {
          if (-d > max_future_distance) {
            max_future_distance = -d;
            future_idx = i;
          }
          ++n_future;
        }
      }
    }
    int unmapped = -1;
    if (n_past > n_future) {
      unmapped = past_idx;
    } else if (n_past < n_future) {
      unmapped = future_idx;
    } else if (n_past > 0) {
      unmapped = max_past_distance >= max_future_distance ? past_idx : future_idx;
    }
    if (unmapped >= 0) {
      ranked[unmapped].score = kMaxInt;
    }
  }

  for (int i = num_ranked - 1; i > 0; --i) {
    for (int j = 0; j < i; ++j) {
      if (ranked[j].score > ranked[j + 1].score) {
        std::swap(ranked[j], ranked[j + 1]);
      }
    }
  }

  const int active = std::min(std::max(in.active_num_ref_frames, 0), kRefsPerFrame);
  out.num_total_refs = std::min(num_ranked, active);
  for (int i = 0; i < out.num_total_refs; ++i) {
    out.ref_frame_idx[i] = ranked[i].slot;
    out.scores_distance[i] = ranked[i].distance;
    out.ref_order_hint[i] = ranked[i].order_hint;
  }

  if (in.check_res && !in.is_bridge) {
    for (int i = 0; i < num_ref_frames && out.num_total_refs < active; ++i) {
      const RefSlot& slot = in.state.slots[i];
      if (slot.valid && slot.order_hint == kRestrictedOrderHint && slot_layer_ok(in, i)) {
        out.ref_frame_idx[out.num_total_refs] = i;
        out.scores_distance[out.num_total_refs] = 0;
        out.ref_order_hint[out.num_total_refs] = kRestrictedOrderHint;
        ++out.num_total_refs;
      }
    }
  }
  return out;
}

PastFutureLists get_past_future_cur_ref_lists(const int ref_order_hint[kRefsPerFrame],
                                              const int scores_distance[kRefsPerFrame],
                                              int num_total_refs) {
  PastFutureLists out{};
  const int count = std::min(std::max(num_total_refs, 0), kRefsPerFrame);
  for (int i = 0; i < count; ++i) {
    if (ref_order_hint[i] == kRestrictedOrderHint) {
      continue;
    }
    if (scores_distance[i] > 0) {
      out.past[out.num_past++] = i;
      if (out.closest_past == -1 || scores_distance[i] < scores_distance[out.closest_past]) {
        out.closest_past = i;
      }
    } else if (scores_distance[i] < 0) {
      out.future[out.num_future++] = i;
      if (out.furthest_future == -1 || ref_order_hint[out.furthest_future] < ref_order_hint[i]) {
        out.furthest_future = i;
      }
      if (out.closest_future == -1 || ref_order_hint[i] < ref_order_hint[out.closest_future]) {
        out.closest_future = i;
      }
    } else {
      out.cur[out.num_cur++] = i;
    }
  }
  out.skip_seg_frame = out.num_cur > 0 ? out.cur[0] : out.closest_past;
  if (out.skip_seg_frame == -1) {
    out.skip_seg_frame = 0;
  }
  out.orig_closest_past = out.closest_past;
  out.orig_closest_future = out.closest_future;
  return out;
}

int max_display_order_hint(const RefSlotState& state, int num_ref_frames, bool only_showable) {
  int max_disp = 0;
  const int count = std::min(std::max(num_ref_frames, 0), kNumRefFrames);
  for (int i = 0; i < count; ++i) {
    const RefSlot& slot = state.slots[i];
    if (!slot.valid || slot.order_hint == kRestrictedOrderHint) {
      continue;
    }
    if (only_showable && !slot.implicit_output && !slot.immediate_output) {
      continue;
    }
    max_disp = std::max(max_disp, slot.order_hint);
  }
  return max_disp;
}

int display_order_hint(int order_hint_lsbs, int order_hint_bits, int max_disp_order_hint_value) {
  if (order_hint_bits <= 0) {
    return order_hint_lsbs;
  }
  const int span = 1 << order_hint_bits;
  int disp = order_hint_lsbs;
  const int offset = max_disp_order_hint_value - (span >> 1) - order_hint_lsbs;
  if (offset >= 0) {
    disp += ((offset >> order_hint_bits) + 1) << order_hint_bits;
  }
  return disp;
}

int order_hint_lsbs_for(int absolute_hint, int order_hint_bits, const RefSlotState& state,
                        int num_ref_frames) {
  if (order_hint_bits <= 0) {
    return absolute_hint;
  }
  const int span = 1 << order_hint_bits;
  const int lsbs = absolute_hint & (span - 1);
  const int max_disp = max_display_order_hint(state, num_ref_frames, true);
  int candidate = lsbs;
  // The reconstruction window is at most one span wide; try the few candidates
  // that could map to `absolute_hint` and return the first that does.
  for (int k = -1; k <= 2; ++k) {
    candidate = lsbs + (k << order_hint_bits);
    if (candidate < 0) {
      continue;
    }
    if (display_order_hint(candidate & (span - 1), order_hint_bits, max_disp) == absolute_hint) {
      return candidate & (span - 1);
    }
  }
  return lsbs;
}

std::vector<GopFrame> build_hierarchical_gop(const GopConfig& config) {
  std::vector<GopFrame> frames;
  const int n = config.gop_size;
  if (n < 2 || n > kNumRefFrames || (n & (n - 1)) != 0) {
    return frames;
  }
  if (config.num_slots < 2 || config.num_slots > kNumRefFrames || config.key_slot < 0 ||
      config.key_slot >= config.num_slots) {
    return frames;
  }

  frames.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    frames[static_cast<std::size_t>(i)].display_index = i;
    frames[static_cast<std::size_t>(i)].display_order_hint = config.start_order_hint + i;
    frames[static_cast<std::size_t>(i)].slot =
        i == 0 ? config.key_slot : (config.key_slot + i) % config.num_slots;
    frames[static_cast<std::size_t>(i)].refresh_frame_flags =
        1u << frames[static_cast<std::size_t>(i)].slot;
  }

  int coding_index = 0;
  auto emit = [&](int display, int layer, int ref_a, int ref_b) {
    GopFrame& f = frames[static_cast<std::size_t>(display)];
    f.coding_index = coding_index++;
    f.temporal_layer = layer;
    if (ref_a >= 0) {
      f.ref_display[f.num_total_refs] = ref_a;
      f.ref_slot[f.num_total_refs] = frames[static_cast<std::size_t>(ref_a)].slot;
      f.ref_order_hint[f.num_total_refs] =
          frames[static_cast<std::size_t>(ref_a)].display_order_hint;
      ++f.num_total_refs;
    }
    if (ref_b >= 0) {
      f.ref_display[f.num_total_refs] = ref_b;
      f.ref_slot[f.num_total_refs] = frames[static_cast<std::size_t>(ref_b)].slot;
      f.ref_order_hint[f.num_total_refs] =
          frames[static_cast<std::size_t>(ref_b)].display_order_hint;
      ++f.num_total_refs;
    }
  };

  // Key frame, then the P anchor at the end of the GOP, then the B pyramid.
  emit(0, 0, -1, -1);
  frames[0].is_key = true;
  emit(n - 1, 0, 0, -1);
  frames[static_cast<std::size_t>(n - 1)].is_anchor = true;

  struct Interval {
    int a;
    int b;
    int layer;
  };
  std::vector<Interval> stack;
  stack.push_back(Interval{0, n - 1, 1});
  while (!stack.empty()) {
    const Interval iv = stack.back();
    stack.pop_back();
    if (iv.b - iv.a <= 1) {
      continue;
    }
    const int mid = (iv.a + iv.b) / 2;
    emit(mid, iv.layer, iv.a, iv.b);
    // Push the right half first so the left half is coded first (deterministic).
    stack.push_back(Interval{mid, iv.b, iv.layer + 1});
    stack.push_back(Interval{iv.a, mid, iv.layer + 1});
  }
  return frames;
}

GopValidation validate_gop(const std::vector<GopFrame>& gop, const GopConfig& config) {
  GopValidation result{};
  if (gop.empty()) {
    return result;
  }
  result.num_frames = static_cast<int>(gop.size());
  std::vector<const GopFrame*> order;
  order.reserve(gop.size());
  for (const GopFrame& frame : gop) {
    order.push_back(&frame);
  }
  std::sort(order.begin(), order.end(),
            [](const GopFrame* a, const GopFrame* b) { return a->coding_index < b->coding_index; });

  RefSlotState state{};
  bool ok = true;
  int bad_index = -1;
  int bad_frame = -1;
  for (const GopFrame* frame : order) {
    if (frame->coding_index < 0 || frame->coding_index >= static_cast<int>(order.size())) {
      ok = false;
      bad_index = frame->coding_index;
      bad_frame = frame->display_index;
      break;
    }
    if (!frame->is_key) {
      if (frame->num_total_refs < 1 || frame->num_total_refs > 2) {
        ok = false;
      }
      int ref_order[2] = {kRestrictedOrderHint, kRestrictedOrderHint};
      int ref_distance[2] = {0, 0};
      for (int r = 0; r < frame->num_total_refs; ++r) {
        const int slot = frame->ref_slot[r];
        if (slot < 0 || slot >= config.num_slots || !state.slots[slot].valid) {
          ok = false;
          break;
        }
        ref_order[r] = state.slots[slot].order_hint;
        ref_distance[r] = get_relative_dist(frame->display_order_hint, ref_order[r]);
      }
      const PastFutureLists lists =
          get_past_future_cur_ref_lists(ref_order, ref_distance, frame->num_total_refs);
      // The anchor is a P frame (one past reference); every other inter frame is
      // a B frame (one past, one future).
      const bool split_ok =
          frame->is_anchor ? (lists.num_past == 1 && lists.num_future == 0 && lists.num_cur == 0)
                           : (lists.num_past == 1 && lists.num_future == 1 && lists.num_cur == 0);
      if (static_cast<std::size_t>(frame->coding_index) < 32) {
        result.refs_were_past_future[frame->coding_index] = split_ok;
      }
      if (!split_ok) {
        ok = false;
      }
      if (frame->num_total_refs == 2 && frame->ref_slot[0] == frame->ref_slot[1]) {
        ok = false;
      }
    }
    if (!ok) {
      bad_index = frame->coding_index;
      bad_frame = frame->display_index;
      break;
    }
    refresh_reference_slots(&state, frame->refresh_frame_flags, frame->display_order_hint, 1920,
                            1080, config.base_q_idx, frame->is_key);
  }
  result.ok = ok;
  result.first_bad_coding_index = bad_index;
  result.first_bad_frame = bad_frame;
  return result;
}

}  // namespace av2enc::inter::adv
