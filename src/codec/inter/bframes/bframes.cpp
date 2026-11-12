#include "bframes.h"

#include "codec/inter/advanced/b_frame.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace av2enc::inter::bframes {
namespace {

using adv::kRefsPerFrame;
using adv::kRestrictedOrderHint;
using adv::RefListInput;
using adv::RefListResult;
using adv::RefSlotState;

constexpr int kReferenceWidth = 1920;
constexpr int kReferenceHeight = 1080;

bool is_power_of_two(int value) { return value > 0 && (value & (value - 1)) == 0; }

bool has_one_bit(std::uint32_t value) {
  if (value == 0) {
    return false;
  }
  return (value & (value - 1U)) == 0U;
}

int bit_index(std::uint32_t value) {
  int index = 0;
  while ((value & 1U) == 0U) {
    value >>= 1U;
    ++index;
  }
  return index;
}

std::uint32_t all_frames_mask(int num_slots) {
  return (std::uint32_t{1} << static_cast<unsigned>(num_slots)) - 1U;
}

void set_error(PlanError* error, PlanError value) {
  if (error != nullptr) {
    *error = value;
  }
}

bool valid_common_config(int gop_size, GopPattern pattern, int order_hint_bits, int num_slots,
                         int leading_pictures, int start_order_hint, PlanError* error) {
  if (gop_size < 2 || gop_size > kMaxRefSlots) {
    set_error(error, PlanError::kBadGopSize);
    return false;
  }
  if (pattern != GopPattern::kLowDelayP && pattern != GopPattern::kIpbb &&
      pattern != GopPattern::kHierarchicalB) {
    set_error(error, PlanError::kBadPattern);
    return false;
  }
  if (pattern == GopPattern::kHierarchicalB && !is_power_of_two(gop_size)) {
    set_error(error, PlanError::kBadGopSize);
    return false;
  }
  if (order_hint_bits < 1 || order_hint_bits > 16) {
    set_error(error, PlanError::kBadOrderHintBits);
    return false;
  }
  if (num_slots < 2 || num_slots > kMaxRefSlots) {
    set_error(error, PlanError::kBadSlotCount);
    return false;
  }
  if (leading_pictures < 0) {
    set_error(error, PlanError::kBadLeadingCount);
    return false;
  }
  if (start_order_hint < 0) {
    set_error(error, PlanError::kBadDisplayOrder);
    return false;
  }
  return true;
}

struct LocalUnit {
  std::vector<FramePlan> frames;
  std::vector<int> coding;  // local display indices in coding order
};

// Builds one unit with local display indices 0..leading+gop_size-1. The key
// frame is at local display `leading`; the P anchor is the last display frame.
LocalUnit build_unit(const GopConfig& config, int start_order_hint, int leading, int gop_index,
                     PlanError* error) {
  LocalUnit unit;
  const int key = leading;
  const int end = leading + config.gop_size - 1;
  const int count = end + 1;
  unit.frames.resize(static_cast<std::size_t>(count));
  for (int d = 0; d < count; ++d) {
    FramePlan& f = unit.frames[static_cast<std::size_t>(d)];
    f.display_index = d;
    f.display_order_hint = start_order_hint + d;
    f.gop_index = gop_index;
    f.is_key = d == key;
    f.is_leading = d < key;
    f.is_anchor = d == end && !f.is_key;
    f.kind = f.is_key ? FrameKind::kKey : FrameKind::kP;
  }

  int next_coding = 0;
  auto emit = [&](int display, int layer, const std::vector<int>& refs) {
    FramePlan& f = unit.frames[static_cast<std::size_t>(display)];
    f.coding_index = next_coding++;
    f.temporal_layer = layer;
    f.num_total_refs = static_cast<int>(refs.size());
    for (std::size_t r = 0; r < refs.size(); ++r) {
      f.ref_display[r] = refs[r];
      f.ref_order_hint[r] = start_order_hint + refs[r];
    }
    if (f.is_key) {
      f.kind = FrameKind::kKey;
    } else {
      f.kind = refs.size() >= 2 ? FrameKind::kB : FrameKind::kP;
    }
    unit.coding.push_back(display);
  };

  // Key frame first: it must be decoded before every other frame in the unit.
  emit(key, 0, {});

  // Leading pictures are decoded after the key frame, latest display first, so
  // each one may reference the key (future) and the already-decoded next
  // leading picture (also future). A key-frame reset means there is no valid
  // past reference for these frames.
  for (int d = key - 1; d >= 0; --d) {
    std::vector<int> refs;
    if (d + 1 < key) {
      refs.push_back(d + 1);
    }
    refs.push_back(key);
    emit(d, 1, refs);
  }

  if (config.gop_size == 1) {
    return unit;
  }

  switch (config.pattern) {
    case GopPattern::kLowDelayP: {
      for (int d = key + 1; d <= end; ++d) {
        emit(d, 0, {d - 1});
      }
      break;
    }
    case GopPattern::kIpbb: {
      emit(end, 0, {key});  // P anchor
      for (int d = key + 1; d < end; ++d) {
        emit(d, 1, {key, end});
      }
      break;
    }
    case GopPattern::kHierarchicalB: {
      emit(end, 0, {key});  // P anchor
      struct Interval {
        int a;
        int b;
        int depth;
      };
      std::vector<Interval> queue;
      queue.push_back(Interval{key, end, 1});
      std::size_t head = 0;
      while (head < queue.size()) {
        const Interval interval = queue[head++];
        if (interval.b - interval.a <= 1) {
          continue;
        }
        const int mid = (interval.a + interval.b) / 2;
        emit(mid, interval.depth, {interval.a, interval.b});
        // Left-to-right breadth-first: deterministic for a given gop_size.
        queue.push_back(Interval{interval.a, mid, interval.depth + 1});
        queue.push_back(Interval{mid, interval.b, interval.depth + 1});
      }
      break;
    }
    default:
      set_error(error, PlanError::kBadPattern);
      return LocalUnit{};
  }

  if (unit.coding.size() != static_cast<std::size_t>(count)) {
    set_error(error, PlanError::kInternal);
    return LocalUnit{};
  }
  return unit;
}

// Assigns one refresh slot per frame in coding order. Key frames refresh
// allFrames and put their content in the first refreshed slot (slot 0). A
// non-key frame prefers an empty slot, then the lowest slot whose frame is no
// longer referenced by any later frame. This is deterministic and guarantees
// that a reference is never overwritten while a later frame still needs it.
bool assign_slots(Plan& plan, PlanError* error) {
  const int count = plan.display_count();
  std::vector<int> uses(static_cast<std::size_t>(count), 0);
  for (const FramePlan& f : plan.frames) {
    for (int r = 0; r < f.num_total_refs; ++r) {
      const int d = f.ref_display[r];
      if (d >= 0 && d < count) {
        ++uses[static_cast<std::size_t>(d)];
      }
    }
  }

  std::vector<int> owner(static_cast<std::size_t>(plan.num_slots), -1);
  for (int coding = 0; coding < count; ++coding) {
    const int display = plan.coding_order[static_cast<std::size_t>(coding)];
    if (display < 0 || display >= count) {
      set_error(error, PlanError::kBadCodingOrder);
      return false;
    }
    FramePlan& f = plan.frames[static_cast<std::size_t>(display)];
    for (int r = 0; r < f.num_total_refs; ++r) {
      const int d = f.ref_display[r];
      if (d >= 0 && d < count && uses[static_cast<std::size_t>(d)] > 0) {
        --uses[static_cast<std::size_t>(d)];
      }
      if (d >= 0 && d < count) {
        f.ref_slot[r] = plan.frames[static_cast<std::size_t>(d)].slot;
      }
    }

    if (f.is_key) {
      std::fill(owner.begin(), owner.end(), -1);
      f.slot = 0;
      f.refresh_frame_flags = all_frames_mask(plan.num_slots);
      owner[0] = display;
      continue;
    }

    int pick = -1;
    for (int s = 0; s < plan.num_slots; ++s) {
      if (owner[static_cast<std::size_t>(s)] == -1) {
        pick = s;
        break;
      }
    }
    if (pick < 0) {
      for (int s = 0; s < plan.num_slots; ++s) {
        const int held = owner[static_cast<std::size_t>(s)];
        if (held >= 0 && uses[static_cast<std::size_t>(held)] == 0) {
          pick = s;
          break;
        }
      }
    }
    if (pick < 0) {
      set_error(error, PlanError::kNotEnoughSlots);
      return false;
    }
    f.slot = pick;
    f.refresh_frame_flags = std::uint32_t{1} << static_cast<unsigned>(pick);
    owner[static_cast<std::size_t>(pick)] = display;
  }
  return true;
}

// Computes the signalled order_hint LSBs from the decoder state at each frame
// and proves that display_order_hint() recovers the absolute hint.
bool assign_order_hints(Plan& plan, PlanError* error) {
  RefSlotState state{};
  for (int coding = 0; coding < plan.display_count(); ++coding) {
    const int display = plan.coding_order[static_cast<std::size_t>(coding)];
    FramePlan& f = plan.frames[static_cast<std::size_t>(display)];
    const int max_disp = adv::max_display_order_hint(state, plan.num_slots, true);
    const int lsbs = adv::order_hint_lsbs_for(f.display_order_hint, plan.order_hint_bits, state,
                                              plan.num_slots);
    const int reconstructed = adv::display_order_hint(lsbs, plan.order_hint_bits, max_disp);
    if (reconstructed != f.display_order_hint) {
      set_error(error, PlanError::kOrderHintReconstruct);
      return false;
    }
    f.order_hint_lsbs = lsbs;
    adv::refresh_reference_slots(&state, f.refresh_frame_flags, f.display_order_hint,
                                 kReferenceWidth, kReferenceHeight, plan.base_q_idx, f.is_key);
  }
  return true;
}

void assign_output_flags(Plan& plan) {
  for (FramePlan& f : plan.frames) {
    // The first displayed frame is output immediately; every later frame is
    // buffered by the §7.21 implicit-output process so reordering is possible.
    f.immediate_output_frame = f.display_index == 0;
    f.implicit_output_frame = !f.immediate_output_frame;
  }
}

// ---------------------------------------------------------------------------
// §7.21.4 / §7.21.5 / §7.21.6 output simulation (single embedded layer).
// ---------------------------------------------------------------------------

struct OutputSim {
  struct SlotMeta {
    bool valid = false;
    int frame = -1;
    int hint = 0;
    bool implicit = false;
    bool immediate = false;
  };

  int num_slots = 0;
  int num_frames = 0;
  std::array<SlotMeta, kMaxRefSlots> meta{};
  std::vector<char> output_done;
  std::vector<int> order;

  void init(int slots, int frames) {
    num_slots = slots;
    num_frames = frames;
    output_done.assign(static_cast<std::size_t>(frames), 0);
    order.clear();
  }

  bool eligible(int slot) const {
    if (slot < 0 || slot >= num_slots) {
      return false;
    }
    const SlotMeta& m = meta[static_cast<std::size_t>(slot)];
    return m.valid && m.frame >= 0 && m.frame < num_frames && m.implicit &&
           output_done[static_cast<std::size_t>(m.frame)] == 0 &&
           m.hint != kRestrictedOrderHint;
  }

  // §7.21.4 output_process(slot): mark and append without following the
  // successive-hint chain.
  void mark_output_slot(int slot) {
    if (slot < 0 || slot >= num_slots) {
      return;
    }
    SlotMeta& m = meta[static_cast<std::size_t>(slot)];
    if (!m.valid || m.frame < 0 || m.frame >= num_frames ||
        output_done[static_cast<std::size_t>(m.frame)] != 0) {
      return;
    }
    output_done[static_cast<std::size_t>(m.frame)] = 1;
    order.push_back(m.frame);
  }

  // §7.21.3: output implicit frames with successive order hints. Iterative so a
  // long display-order chain cannot exhaust the call stack.
  void output_successive(int hint) {
    int current = hint;
    for (;;) {
      const int target = current + 1;
      bool made_output = false;
      for (int s = 0; s < num_slots; ++s) {
        if (eligible(s) && meta[static_cast<std::size_t>(s)].hint == target) {
          mark_output_slot(s);
          made_output = true;
        }
      }
      if (!made_output) {
        break;
      }
      current = target;
    }
  }

  void output_slot(int slot) {
    if (slot < 0 || slot >= num_slots) {
      return;
    }
    const int hint = meta[static_cast<std::size_t>(slot)].hint;
    const int frame = meta[static_cast<std::size_t>(slot)].frame;
    mark_output_slot(slot);
    if (frame >= 0 && frame < num_frames &&
        output_done[static_cast<std::size_t>(frame)] != 0) {
      output_successive(hint);
    }
  }

  // §7.21.6: output any eligible frames with a lower order hint first, then
  // ref_idx (or the current frame when ref_idx == -1), then successive hints.
  void frame_buffers(int ref_idx, int current_frame, int current_hint) {
    const int output_hint = ref_idx >= 0 ? meta[static_cast<std::size_t>(ref_idx)].hint
                                        : current_hint;
    for (;;) {
      int best = -1;
      int best_hint = 0;
      for (int s = 0; s < num_slots; ++s) {
        if (!eligible(s)) {
          continue;
        }
        const int hint = meta[static_cast<std::size_t>(s)].hint;
        if (hint >= output_hint) {
          continue;
        }
        if (best < 0 || hint < best_hint) {
          best = s;
          best_hint = hint;
        }
      }
      if (best < 0) {
        break;
      }
      output_slot(best);
    }

    if (ref_idx >= 0) {
      output_slot(ref_idx);
    } else if (current_frame >= 0 && current_frame < num_frames &&
               output_done[static_cast<std::size_t>(current_frame)] == 0) {
      output_done[static_cast<std::size_t>(current_frame)] = 1;
      order.push_back(current_frame);
      output_successive(output_hint);
    }
  }

  // §7.21.5: flush all remaining eligible frames in output order.
  void flush() {
    for (;;) {
      int best = -1;
      int best_hint = 0;
      for (int s = 0; s < num_slots; ++s) {
        if (!eligible(s)) {
          continue;
        }
        const int hint = meta[static_cast<std::size_t>(s)].hint;
        if (best < 0 || hint < best_hint) {
          best = s;
          best_hint = hint;
        }
      }
      if (best < 0) {
        break;
      }
      output_slot(best);
    }
  }
};

bool is_duplicate_reference(const FramePlan& f) {
  if (f.num_total_refs < 2) {
    return false;
  }
  return f.ref_display[0] == f.ref_display[1] || f.ref_slot[0] == f.ref_slot[1];
}

}  // namespace

const char* plan_error_name(PlanError error) {
  switch (error) {
    case PlanError::kOk:
      return "ok";
    case PlanError::kBadGopSize:
      return "bad_gop_size";
    case PlanError::kBadPattern:
      return "bad_pattern";
    case PlanError::kBadSlotCount:
      return "bad_slot_count";
    case PlanError::kBadOrderHintBits:
      return "bad_order_hint_bits";
    case PlanError::kBadLeadingCount:
      return "bad_leading_count";
    case PlanError::kNotEnoughSlots:
      return "not_enough_slots";
    case PlanError::kBadDisplayOrder:
      return "bad_display_order";
    case PlanError::kBadCodingOrder:
      return "bad_coding_order";
    case PlanError::kMissingKeyFrame:
      return "missing_key_frame";
    case PlanError::kKeyFrameHasRefs:
      return "key_frame_has_refs";
    case PlanError::kTooManyRefs:
      return "too_many_refs";
    case PlanError::kMissingReference:
      return "missing_reference";
    case PlanError::kReferenceOutOfRange:
      return "reference_out_of_range";
    case PlanError::kReferenceNotDecoded:
      return "reference_not_decoded";
    case PlanError::kReferenceSlotInvalid:
      return "reference_slot_invalid";
    case PlanError::kReferenceHintMismatch:
      return "reference_hint_mismatch";
    case PlanError::kKeyFrameRefInvalidated:
      return "key_frame_ref_invalidated";
    case PlanError::kLeadingPictureRule:
      return "leading_picture_rule";
    case PlanError::kDuplicateReference:
      return "duplicate_reference";
    case PlanError::kBadReferenceSplit:
      return "bad_reference_split";
    case PlanError::kRefreshFlagsInvalid:
      return "refresh_flags_invalid";
    case PlanError::kOutputFlagsInvalid:
      return "output_flags_invalid";
    case PlanError::kOrderHintReconstruct:
      return "order_hint_reconstruct";
    case PlanError::kOutputOrderMismatch:
      return "output_order_mismatch";
    case PlanError::kOutputMissingFrame:
      return "output_missing_frame";
    case PlanError::kInternal:
      return "internal";
    default:
      return "unknown";
  }
}

const char* gop_pattern_name(GopPattern pattern) {
  switch (pattern) {
    case GopPattern::kLowDelayP:
      return "low_delay_p";
    case GopPattern::kIpbb:
      return "ipbb";
    case GopPattern::kHierarchicalB:
      return "hierarchical_b";
    default:
      return "unknown";
  }
}

const char* frame_kind_name(FrameKind kind) {
  switch (kind) {
    case FrameKind::kKey:
      return "key";
    case FrameKind::kP:
      return "p";
    case FrameKind::kB:
      return "b";
    default:
      return "unknown";
  }
}

const FramePlan* Plan::frame_at_display(int display_index) const {
  if (display_index < 0 || display_index >= display_count()) {
    return nullptr;
  }
  return &frames[static_cast<std::size_t>(display_index)];
}

FrameHeaderFields header_fields(const FramePlan& frame) {
  FrameHeaderFields h;
  h.keyframe = frame.is_key;
  h.order_hint_lsbs = static_cast<std::uint32_t>(frame.order_hint_lsbs);
  h.refresh_frame_flags = frame.refresh_frame_flags;
  h.frame_explicit_ref_frame_map = frame.frame_explicit_ref_frame_map;
  h.num_total_refs = static_cast<std::uint8_t>(frame.num_total_refs);
  for (int r = 0; r < kMaxExplicitRefs; ++r) {
    h.ref_frame_idx[r] =
        static_cast<std::uint8_t>(frame.ref_slot[r] < 0 ? 0 : frame.ref_slot[r]);
  }
  h.immediate_output_frame = frame.immediate_output_frame;
  h.implicit_output_frame = frame.implicit_output_frame;
  h.reference_select = frame.num_total_refs >= 2;
  return h;
}

std::vector<FrameKind> coding_frame_types(const Plan& plan) {
  std::vector<FrameKind> types;
  types.reserve(plan.coding_order.size());
  for (int display : plan.coding_order) {
    if (display >= 0 && display < plan.display_count()) {
      types.push_back(plan.frames[static_cast<std::size_t>(display)].kind);
    }
  }
  return types;
}

bool build_gop(const GopConfig& config, Plan& plan, PlanError* error) {
  set_error(error, PlanError::kOk);
  if (!valid_common_config(config.gop_size, config.pattern, config.order_hint_bits,
                           config.num_slots, config.leading_pictures, config.start_order_hint,
                           error)) {
    return false;
  }

  LocalUnit unit = build_unit(config, config.start_order_hint, config.leading_pictures, 0, error);
  if (unit.coding.empty() || unit.coding.size() != unit.frames.size()) {
    return false;
  }

  plan = Plan{};
  plan.pattern = config.pattern;
  plan.gop_size = config.gop_size;
  plan.start_order_hint = config.start_order_hint;
  plan.order_hint_bits = config.order_hint_bits;
  plan.num_slots = config.num_slots;
  plan.base_q_idx = config.base_q_idx;
  plan.leading_pictures = config.leading_pictures;
  plan.is_sequence = false;
  plan.frames = std::move(unit.frames);
  plan.coding_order = std::move(unit.coding);
  assign_output_flags(plan);

  if (!assign_slots(plan, error) || !assign_order_hints(plan, error)) {
    return false;
  }

  ValidationReport report;
  if (!validate_plan(plan, &report)) {
    set_error(error, report.error);
    return false;
  }
  return true;
}

bool build_sequence(const SequenceConfig& config, Plan& plan, PlanError* error) {
  set_error(error, PlanError::kOk);
  if (config.num_gops < 1) {
    set_error(error, PlanError::kBadGopSize);
    return false;
  }
  if (!valid_common_config(config.gop_size, config.pattern, config.order_hint_bits,
                           config.num_slots, config.leading_pictures, config.start_order_hint,
                           error)) {
    return false;
  }

  plan = Plan{};
  plan.pattern = config.pattern;
  plan.gop_size = config.gop_size;
  plan.start_order_hint = config.start_order_hint;
  plan.order_hint_bits = config.order_hint_bits;
  plan.num_slots = config.num_slots;
  plan.base_q_idx = config.base_q_idx;
  plan.leading_pictures = config.leading_pictures;
  plan.is_sequence = true;

  for (int gop = 0; gop < config.num_gops; ++gop) {
    const int leading = gop == 0 ? 0 : config.leading_pictures;
    const int display_offset = plan.display_count();
    GopConfig unit_config;
    unit_config.gop_size = config.gop_size;
    unit_config.pattern = config.pattern;
    unit_config.start_order_hint = config.start_order_hint + display_offset;
    unit_config.order_hint_bits = config.order_hint_bits;
    unit_config.num_slots = config.num_slots;
    unit_config.base_q_idx = config.base_q_idx;
    unit_config.leading_pictures = leading;
    LocalUnit unit = build_unit(unit_config, unit_config.start_order_hint, leading, gop, error);
    if (unit.coding.empty() || unit.coding.size() != unit.frames.size()) {
      return false;
    }

    for (FramePlan& f : unit.frames) {
      f.display_index += display_offset;
      f.display_order_hint = config.start_order_hint + f.display_index;
      f.gop_index = gop;
      for (int r = 0; r < f.num_total_refs; ++r) {
        f.ref_display[r] += display_offset;
        f.ref_order_hint[r] = config.start_order_hint + f.ref_display[r];
      }
      plan.frames.push_back(f);
    }
    for (int local : unit.coding) {
      const int display = local + display_offset;
      plan.frames[static_cast<std::size_t>(display)].coding_index =
          static_cast<int>(plan.coding_order.size());
      plan.coding_order.push_back(display);
    }
  }

  assign_output_flags(plan);
  if (!assign_slots(plan, error) || !assign_order_hints(plan, error)) {
    return false;
  }

  ValidationReport report;
  if (!validate_plan(plan, &report)) {
    set_error(error, report.error);
    return false;
  }
  return true;
}

bool simulate_decode(const Plan& plan, DecodeReport* report_out) {
  DecodeReport r;
  const int count = plan.display_count();
  if (count <= 0) {
    r.error = PlanError::kBadDisplayOrder;
    if (report_out != nullptr) {
      *report_out = std::move(r);
    }
    return false;
  }
  if (plan.num_slots < 1 || plan.num_slots > kMaxRefSlots) {
    r.error = PlanError::kBadSlotCount;
    if (report_out != nullptr) {
      *report_out = std::move(r);
    }
    return false;
  }
  if (plan.order_hint_bits < 1 || plan.order_hint_bits > 16) {
    r.error = PlanError::kBadOrderHintBits;
    if (report_out != nullptr) {
      *report_out = std::move(r);
    }
    return false;
  }
  if (static_cast<int>(plan.coding_order.size()) != count) {
    r.error = PlanError::kBadCodingOrder;
    if (report_out != nullptr) {
      *report_out = std::move(r);
    }
    return false;
  }

  auto fail = [&](PlanError error, int coding_index, int display_index) {
    r.ok = false;
    r.error = error;
    r.error_coding_index = coding_index;
    r.error_display_index = display_index;
    if (report_out != nullptr) {
      *report_out = std::move(r);
    }
    return false;
  };

  r.frames.resize(static_cast<std::size_t>(count));
  r.decode_order.reserve(static_cast<std::size_t>(count));
  r.slot_of_frame.assign(static_cast<std::size_t>(count), -1);
  r.expected_output_order.resize(static_cast<std::size_t>(count));
  for (int d = 0; d < count; ++d) {
    r.expected_output_order[static_cast<std::size_t>(d)] = d;
  }

  RefSlotState state{};
  OutputSim output;
  output.init(plan.num_slots, count);
  std::vector<char> decoded(static_cast<std::size_t>(count), 0);

  auto refresh_and_output = [&](const FramePlan& f, int display) {
    // §7.23: an eligible frame in a slot that is about to be overwritten is
    // output first.
    for (int s = 0; s < plan.num_slots; ++s) {
      if ((f.refresh_frame_flags & (std::uint32_t{1} << static_cast<unsigned>(s))) == 0U) {
        continue;
      }
      if (output.eligible(s)) {
        output.frame_buffers(s, -1, -1);
      }
    }

    adv::refresh_reference_slots(&state, f.refresh_frame_flags, f.display_order_hint,
                                 kReferenceWidth, kReferenceHeight, plan.base_q_idx, f.is_key);

    for (int s = 0; s < plan.num_slots; ++s) {
      if ((f.refresh_frame_flags & (std::uint32_t{1} << static_cast<unsigned>(s))) == 0U) {
        continue;
      }
      OutputSim::SlotMeta& m = output.meta[static_cast<std::size_t>(s)];
      m.valid = state.slots[s].valid;
      m.hint = f.display_order_hint;
      m.implicit = f.implicit_output_frame;
      m.immediate = f.immediate_output_frame;
      m.frame = m.valid ? display : -1;
    }

    if (f.immediate_output_frame) {
      output.frame_buffers(-1, display, f.display_order_hint);
    }
  };

  for (int coding = 0; coding < count; ++coding) {
    const int display = plan.coding_order[static_cast<std::size_t>(coding)];
    if (display < 0 || display >= count) {
      return fail(PlanError::kBadCodingOrder, coding, display);
    }
    const FramePlan& f = plan.frames[static_cast<std::size_t>(display)];
    if (f.coding_index != coding) {
      return fail(PlanError::kBadCodingOrder, coding, display);
    }
    if (decoded[static_cast<std::size_t>(display)] != 0) {
      return fail(PlanError::kBadCodingOrder, coding, display);
    }

    DecodeFrameTrace trace;
    trace.display_index = display;
    trace.coding_index = coding;
    trace.num_total_refs = f.num_total_refs;
    trace.output_immediate = f.immediate_output_frame;

    if (f.num_total_refs < 0 || f.num_total_refs > kMaxExplicitRefs) {
      return fail(PlanError::kTooManyRefs, coding, display);
    }
    if (!f.is_key && f.num_total_refs == 0) {
      return fail(PlanError::kMissingReference, coding, display);
    }
    if (f.is_key && f.num_total_refs != 0) {
      return fail(PlanError::kKeyFrameHasRefs, coding, display);
    }
    if (is_duplicate_reference(f)) {
      return fail(PlanError::kDuplicateReference, coding, display);
    }

    for (int k = 0; k < f.num_total_refs; ++k) {
      const int ref_display = f.ref_display[k];
      const int ref_slot = f.ref_slot[k];
      trace.ref_display[k] = ref_display;
      trace.ref_slot[k] = ref_slot;
      if (ref_display < 0 || ref_display >= count) {
        return fail(PlanError::kReferenceOutOfRange, coding, display);
      }
      if (decoded[static_cast<std::size_t>(ref_display)] == 0) {
        return fail(PlanError::kReferenceNotDecoded, coding, display);
      }
      if (ref_slot < 0 || ref_slot >= plan.num_slots) {
        return fail(PlanError::kReferenceSlotInvalid, coding, display);
      }
      if (!state.slots[ref_slot].valid ||
          output.meta[static_cast<std::size_t>(ref_slot)].frame != ref_display) {
        return fail(PlanError::kReferenceSlotInvalid, coding, display);
      }
      if (state.slots[ref_slot].order_hint != f.ref_order_hint[k]) {
        return fail(PlanError::kReferenceSlotInvalid, coding, display);
      }
    }

    int ref_order[adv::kRefsPerFrame];
    int ref_distance[adv::kRefsPerFrame];
    for (int i = 0; i < adv::kRefsPerFrame; ++i) {
      ref_order[i] = kRestrictedOrderHint;
      ref_distance[i] = 0;
    }
    for (int k = 0; k < f.num_total_refs; ++k) {
      ref_order[k] = state.slots[f.ref_slot[k]].order_hint;
      ref_distance[k] = adv::get_relative_dist(f.display_order_hint, ref_order[k]);
    }
    const adv::PastFutureLists split =
        adv::get_past_future_cur_ref_lists(ref_order, ref_distance, f.num_total_refs);
    trace.num_past = split.num_past;
    trace.num_future = split.num_future;
    trace.num_cur = split.num_cur;
    for (int i = 0; i < adv::kRefsPerFrame; ++i) {
      trace.past[i] = split.past[i];
      trace.future[i] = split.future[i];
      trace.cur[i] = split.cur[i];
    }

    bool split_ok = false;
    if (f.is_key) {
      split_ok = f.num_total_refs == 0;
    } else if (f.is_leading) {
      // Leading pictures follow a key-frame reset: no past reference exists,
      // so every reference must be in the future.
      split_ok = f.num_total_refs >= 1 && split.num_past == 0 &&
                 split.num_future == f.num_total_refs && split.num_cur == 0;
    } else if (f.kind == FrameKind::kP) {
      split_ok = split.num_past == 1 && split.num_future == 0 && split.num_cur == 0;
    } else if (f.kind == FrameKind::kB) {
      split_ok = split.num_past == 1 && split.num_future == 1 && split.num_cur == 0;
    }
    trace.past_future_ok = split_ok;
    if (!split_ok) {
      return fail(PlanError::kBadReferenceSplit, coding, display);
    }

    if (f.is_key) {
      if (f.refresh_frame_flags != all_frames_mask(plan.num_slots) || f.slot != 0) {
        return fail(PlanError::kRefreshFlagsInvalid, coding, display);
      }
    } else {
      if (!has_one_bit(f.refresh_frame_flags)) {
        return fail(PlanError::kRefreshFlagsInvalid, coding, display);
      }
      const int slot = bit_index(f.refresh_frame_flags);
      if (slot != f.slot || slot < 0 || slot >= plan.num_slots) {
        return fail(PlanError::kRefreshFlagsInvalid, coding, display);
      }
    }
    if (f.immediate_output_frame == f.implicit_output_frame) {
      return fail(PlanError::kOutputFlagsInvalid, coding, display);
    }

    // §7.7 ranking over the live decoder state. The explicit map is allowed to
    // differ, but running the ranking here proves the shared code is what the
    // decoder would use and gives tests a cross-check.
    RefListInput list_input;
    list_input.state = state;
    list_input.order_hint = f.display_order_hint;
    list_input.num_ref_frames = plan.num_slots;
    list_input.active_num_ref_frames = kRefsPerFrame;
    list_input.obu_mlayer_id = 0;
    list_input.obu_tlayer_id = 0;
    list_input.frame_width = kReferenceWidth;
    list_input.frame_height = kReferenceHeight;
    list_input.check_res = false;
    const RefListResult auto_list = adv::build_ref_frame_list(list_input);
    trace.auto_num_total_refs = auto_list.num_total_refs;
    for (int i = 0; i < auto_list.num_total_refs; ++i) {
      trace.auto_ref_slot[i] = auto_list.ref_frame_idx[i];
      trace.auto_ref_order_hint[i] = auto_list.ref_order_hint[i];
      trace.auto_scores_distance[i] = auto_list.scores_distance[i];
    }

    const int max_disp = adv::max_display_order_hint(state, plan.num_slots, true);
    const int expected_lsbs = adv::order_hint_lsbs_for(f.display_order_hint,
                                                       plan.order_hint_bits, state,
                                                       plan.num_slots);
    if (f.order_hint_lsbs != expected_lsbs) {
      return fail(PlanError::kOrderHintReconstruct, coding, display);
    }
    if (adv::display_order_hint(f.order_hint_lsbs, plan.order_hint_bits, max_disp) !=
        f.display_order_hint) {
      return fail(PlanError::kOrderHintReconstruct, coding, display);
    }

    trace.ref_slots_held_intended_frames = true;
    trace.refresh_ok = true;
    refresh_and_output(f, display);
    decoded[static_cast<std::size_t>(display)] = 1;
    r.slot_of_frame[static_cast<std::size_t>(display)] = f.slot;
    r.decode_order.push_back(display);
    r.frames[static_cast<std::size_t>(coding)] = trace;
  }

  output.flush();
  for (int d = 0; d < count; ++d) {
    if (decoded[static_cast<std::size_t>(d)] == 0) {
      return fail(PlanError::kOutputMissingFrame, -1, d);
    }
    if (output.output_done[static_cast<std::size_t>(d)] == 0) {
      return fail(PlanError::kOutputMissingFrame, -1, d);
    }
  }

  r.output_order = output.order;
  r.output_matches_display_order = r.output_order == r.expected_output_order;
  if (!r.output_matches_display_order) {
    return fail(PlanError::kOutputOrderMismatch, -1, -1);
  }

  r.ok = true;
  r.error = PlanError::kOk;
  if (report_out != nullptr) {
    *report_out = std::move(r);
  }
  return true;
}

bool validate_plan(const Plan& plan, ValidationReport* report_out) {
  ValidationReport v;
  auto fail = [&](PlanError error, int coding_index, int display_index) {
    v.ok = false;
    v.error = error;
    v.error_coding_index = coding_index;
    v.error_display_index = display_index;
    if (report_out != nullptr) {
      *report_out = v;
    }
    return false;
  };

  const int count = plan.display_count();
  if (count <= 0) {
    return fail(PlanError::kBadDisplayOrder, -1, -1);
  }
  if (plan.num_slots < 2 || plan.num_slots > kMaxRefSlots) {
    return fail(PlanError::kBadSlotCount, -1, -1);
  }
  if (plan.order_hint_bits < 1 || plan.order_hint_bits > 16) {
    return fail(PlanError::kBadOrderHintBits, -1, -1);
  }
  if (static_cast<int>(plan.coding_order.size()) != count) {
    return fail(PlanError::kBadCodingOrder, -1, -1);
  }

  std::vector<char> seen(static_cast<std::size_t>(count), 0);
  for (int coding = 0; coding < count; ++coding) {
    const int display = plan.coding_order[static_cast<std::size_t>(coding)];
    if (display < 0 || display >= count || seen[static_cast<std::size_t>(display)] != 0) {
      return fail(PlanError::kBadCodingOrder, coding, display);
    }
    seen[static_cast<std::size_t>(display)] = 1;
    if (plan.frames[static_cast<std::size_t>(display)].coding_index != coding) {
      return fail(PlanError::kBadCodingOrder, coding, display);
    }
  }

  int previous_hint = -1;
  int previous_gop = -1;
  for (int d = 0; d < count; ++d) {
    const FramePlan& f = plan.frames[static_cast<std::size_t>(d)];
    if (f.display_index != d) {
      return fail(PlanError::kBadDisplayOrder, f.coding_index, d);
    }
    if (f.display_order_hint <= previous_hint) {
      return fail(PlanError::kBadDisplayOrder, f.coding_index, d);
    }
    previous_hint = f.display_order_hint;
    if (f.gop_index < 0 || f.gop_index < previous_gop) {
      return fail(PlanError::kBadDisplayOrder, f.coding_index, d);
    }
    previous_gop = f.gop_index;
    if (f.num_total_refs < 0 || f.num_total_refs > kMaxExplicitRefs) {
      return fail(PlanError::kTooManyRefs, f.coding_index, d);
    }
    if (!f.frame_explicit_ref_frame_map) {
      // This layer only schedules explicit maps; the automatic §7.7 ranking is
      // simulated for cross-checks but is not selected by the plan.
      return fail(PlanError::kBadPattern, f.coding_index, d);
    }
    if (f.immediate_output_frame == f.implicit_output_frame) {
      return fail(PlanError::kOutputFlagsInvalid, f.coding_index, d);
    }
    if (f.is_key) {
      if (f.kind != FrameKind::kKey) {
        return fail(PlanError::kBadPattern, f.coding_index, d);
      }
      if (f.num_total_refs != 0) {
        return fail(PlanError::kKeyFrameHasRefs, f.coding_index, d);
      }
    } else {
      if (f.kind == FrameKind::kKey) {
        return fail(PlanError::kBadPattern, f.coding_index, d);
      }
      if (f.num_total_refs == 0) {
        return fail(PlanError::kMissingReference, f.coding_index, d);
      }
      if (f.kind == FrameKind::kP && f.num_total_refs != 1) {
        return fail(PlanError::kBadPattern, f.coding_index, d);
      }
      if (f.kind == FrameKind::kB && f.num_total_refs != 2) {
        return fail(PlanError::kBadPattern, f.coding_index, d);
      }
    }
    for (int r = 0; r < f.num_total_refs; ++r) {
      const int ref_display = f.ref_display[r];
      if (ref_display < 0 || ref_display >= count || ref_display == d) {
        return fail(PlanError::kReferenceOutOfRange, f.coding_index, d);
      }
      if (r > 0 && (f.ref_display[0] == ref_display || f.ref_slot[0] == f.ref_slot[r])) {
        return fail(PlanError::kDuplicateReference, f.coding_index, d);
      }
      const FramePlan& ref = plan.frames[static_cast<std::size_t>(ref_display)];
      if (f.ref_order_hint[r] != ref.display_order_hint) {
        return fail(PlanError::kReferenceHintMismatch, f.coding_index, d);
      }
    }
  }

  // GOP layout: exactly one key per gop_index, the first coded frame of each GOP
  // is that key, and no reference crosses a GOP/key-frame reset.
  const int gop_count = previous_gop + 1;
  std::vector<int> gop_base(static_cast<std::size_t>(gop_count), -1);
  std::vector<int> gop_key(static_cast<std::size_t>(gop_count), -1);
  for (int d = 0; d < count; ++d) {
    const FramePlan& f = plan.frames[static_cast<std::size_t>(d)];
    const int g = f.gop_index;
    if (g < 0 || g >= gop_count) {
      return fail(PlanError::kBadPattern, f.coding_index, d);
    }
    if (gop_base[static_cast<std::size_t>(g)] < 0) {
      gop_base[static_cast<std::size_t>(g)] = d;
    }
    if (f.is_key) {
      if (gop_key[static_cast<std::size_t>(g)] >= 0) {
        return fail(PlanError::kBadPattern, f.coding_index, d);
      }
      gop_key[static_cast<std::size_t>(g)] = d;
    }
  }
  for (int g = 0; g < gop_count; ++g) {
    if (gop_base[static_cast<std::size_t>(g)] < 0 ||
        gop_key[static_cast<std::size_t>(g)] < 0) {
      return fail(PlanError::kMissingKeyFrame, -1, -1);
    }
  }

  std::vector<int> gop_last(static_cast<std::size_t>(gop_count), -1);
  for (int d = 0; d < count; ++d) {
    const int g = plan.frames[static_cast<std::size_t>(d)].gop_index;
    gop_last[static_cast<std::size_t>(g)] = d;
  }
  for (int d = 0; d < count; ++d) {
    const FramePlan& f = plan.frames[static_cast<std::size_t>(d)];
    const bool last = d == gop_last[static_cast<std::size_t>(f.gop_index)];
    if (f.is_anchor) {
      if (f.is_key || f.kind != FrameKind::kP || f.num_total_refs != 1 || !last) {
        return fail(PlanError::kBadPattern, f.coding_index, d);
      }
      if (f.ref_display[0] >= d) {
        return fail(PlanError::kBadReferenceSplit, f.coding_index, d);
      }
    } else if (!f.is_key && last) {
      return fail(PlanError::kBadPattern, f.coding_index, d);
    }
  }

  std::vector<char> gop_started(static_cast<std::size_t>(gop_count), 0);
  for (int coding = 0; coding < count; ++coding) {
    const int d = plan.coding_order[static_cast<std::size_t>(coding)];
    const FramePlan& f = plan.frames[static_cast<std::size_t>(d)];
    const int g = f.gop_index;
    if (gop_started[static_cast<std::size_t>(g)] == 0) {
      if (!f.is_key) {
        return fail(PlanError::kMissingKeyFrame, coding, d);
      }
      gop_started[static_cast<std::size_t>(g)] = 1;
    }
  }

  for (int d = 0; d < count; ++d) {
    const FramePlan& f = plan.frames[static_cast<std::size_t>(d)];
    const int g = f.gop_index;
    const int base = gop_base[static_cast<std::size_t>(g)];
    const int key = gop_key[static_cast<std::size_t>(g)];
    for (int r = 0; r < f.num_total_refs; ++r) {
      const int rd = f.ref_display[r];
      const FramePlan& ref = plan.frames[static_cast<std::size_t>(rd)];
      if (rd < base) {
        return fail(PlanError::kKeyFrameRefInvalidated, f.coding_index, d);
      }
      if (f.is_leading) {
        // Leading pictures may reference the key or another leading picture,
        // never a regular frame that is displayed after the key.
        if (rd > key) {
          return fail(PlanError::kLeadingPictureRule, f.coding_index, d);
        }
        if (rd < key && !ref.is_leading) {
          return fail(PlanError::kKeyFrameRefInvalidated, f.coding_index, d);
        }
      } else {
        // Regular frames may not use a leading picture (the first regular
        // frame after a random-access point invalidates them in §5.18.1).
        if (rd < key) {
          if (ref.is_leading) {
            return fail(PlanError::kLeadingPictureRule, f.coding_index, d);
          }
          return fail(PlanError::kKeyFrameRefInvalidated, f.coding_index, d);
        }
        if (ref.is_leading) {
          return fail(PlanError::kLeadingPictureRule, f.coding_index, d);
        }
      }
    }
  }

  DecodeReport decode;
  if (!simulate_decode(plan, &decode)) {
    v.ok = false;
    v.error = decode.error;
    v.error_coding_index = decode.error_coding_index;
    v.error_display_index = decode.error_display_index;
    v.decode_order = decode.decode_order;
    v.output_order = decode.output_order;
    v.output_matches_display_order = decode.output_matches_display_order;
    if (report_out != nullptr) {
      *report_out = v;
    }
    return false;
  }

  v.ok = true;
  v.error = PlanError::kOk;
  v.decode_order = decode.decode_order;
  v.output_order = decode.output_order;
  v.output_matches_display_order = decode.output_matches_display_order;
  if (report_out != nullptr) {
    *report_out = v;
  }
  return true;
}

}  // namespace av2enc::inter::bframes
