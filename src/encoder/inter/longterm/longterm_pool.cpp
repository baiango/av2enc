// Reference-pool state machine for §5.18.2 / §7.23 and the random-access
// conformance rules of §7.4.3 (closed) and §7.4.4 (open).
//
// The pool keeps exactly the state the normative decoding process keeps for
// the reference slots plus the four variables that decide random-access
// behaviour: RefValid, RefLongTermId, OlkEncountered and OlkRefresh.
#include "encoder/inter/longterm/longterm.h"

#include <algorithm>
#include <cstdint>

namespace av2enc::inter::longterm {
namespace {

bool set_error(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return false;
}

std::string slot_list(std::uint32_t mask) {
  std::string out;
  for (int i = 0; i < kNumRefFrames; ++i) {
    if ((mask & (1U << i)) == 0U) continue;
    if (!out.empty()) out += ",";
    out += std::to_string(i);
  }
  return out.empty() ? "-" : out;
}

std::uint32_t slot_mask(const PoolState& pool) {
  std::uint32_t mask = 0;
  for (int i = 0; i < pool.num_ref_frames && i < kNumRefFrames; ++i) {
    if (pool.slots[static_cast<std::size_t>(i)].valid) mask |= 1U << i;
  }
  return mask;
}

bool named_long_term_id(const FrameHeader& header, std::int32_t id) {
  for (const std::uint8_t value : header.ref_long_term_id) {
    if (static_cast<std::int32_t>(value) == id) return true;
  }
  return false;
}

int find_slot_with_long_term_id(const PoolState& pool, std::int32_t id) {
  for (int i = 0; i < pool.num_ref_frames && i < kNumRefFrames; ++i) {
    const SlotState& slot = pool.slots[static_cast<std::size_t>(i)];
    if (slot.valid && slot.long_term_id == id) return i;
  }
  return -1;
}

// §7.4.4: "A long term reference frame shall be included in the
// ref_long_term_id list of an OLK, if and only if: when using sequential
// decoding, this long term reference frame is held in a reference frame buffer
// when the OLK is encountered, and ... when the first Regular frame ... is
// encountered after the OLK, and [same embedded layer]."
//
// With the §5.18.2 invalidation code (`RefValid == 0` for slots that are not
// in OlkRefresh and have RefLongTermId == -1) this is exactly:
//
//   * every named id is held in a valid slot when the OLK is encountered;
//   * the OLK does not refresh a slot that holds a named id (otherwise the
//     reference would not be held at the first Regular frame);
//   * every live long-term reference that the OLK does not refresh is named
//     (otherwise it would be dropped by the invalidation although the spec
//     says only named long-term references survive).
bool validate_olk_long_term_list(const SequenceConfig& config,
                                 const FrameHeader& olk,
                                 const PoolState& pool, std::string* error) {
  for (const std::uint8_t value : olk.ref_long_term_id) {
    const std::int32_t id = static_cast<std::int32_t>(value);
    if (find_slot_with_long_term_id(pool, id) < 0) {
      return set_error(error, "OLK names long-term id " + std::to_string(id) +
                                  " which is not held in a valid slot");
    }
  }
  // With the pinned-dav2d layout the list cannot be coded, so the "must name"
  // direction is not expressible; §5.18.2's invalidation still keeps slots
  // whose RefLongTermId != -1, which is what retains the reference.
  const bool can_name =
      config.olk_long_term_list == LongTermListSyntax::kSpecV1;
  for (int i = 0; i < pool.num_ref_frames && i < kNumRefFrames; ++i) {
    const SlotState& slot = pool.slots[static_cast<std::size_t>(i)];
    if (!slot.valid || slot.long_term_id < 0) continue;
    const bool refreshed = (olk.refresh_frame_flags & (1U << i)) != 0U;
    const bool named = named_long_term_id(olk, slot.long_term_id);
    if (refreshed && named) {
      return set_error(error,
                       "OLK refreshes slot " + std::to_string(i) +
                           " holding named long-term reference id " +
                           std::to_string(slot.long_term_id));
    }
    if (can_name && !refreshed && !named) {
      return set_error(error,
                       "OLK does not name long-term reference id " +
                           std::to_string(slot.long_term_id) + " in slot " +
                           std::to_string(i) +
                           " which survives the random access point");
    }
  }
  return true;
}

}  // namespace

std::uint32_t effective_refresh_flags(const SequenceConfig& config,
                                      const FrameHeader& header) {
  if (header.role == FrameRole::kClosedLoopKey && config.max_mlayer_id == 0U) {
    // §5.18.2: OBU_CLOSED_LOOP_KEY + max_mlayer_id == 0 infers allFrames.
    return (1U << config.num_ref_frames) - 1U;
  }
  return header.refresh_frame_flags;
}

void reset_pool(const SequenceConfig& config, PoolState* pool) {
  *pool = PoolState{};
  pool->num_ref_frames = config.num_ref_frames;
  pool->long_term_frame_id_bits = config.long_term_frame_id_bits;
  pool->order_hint_bits = config.order_hint_bits;
}

// ---------------------------------------------------------------------------
// Static frame-header conformance (§5.18.2, §6.4.6, §6.17)
// ---------------------------------------------------------------------------
bool validate_frame_header(const SequenceConfig& config,
                           const FrameHeader& header, std::string* error) {
  if (config.long_term_frame_id_bits > 7U) {
    return set_error(error, "long_term_frame_id_bits exceeds f(3)");
  }
  if (config.order_hint_bits < 1U || config.order_hint_bits > 16U) {
    return set_error(error, "OrderHintBits outside [1, 16]");
  }
  if (config.num_ref_frames < 1U || config.num_ref_frames > 16U) {
    return set_error(error, "NumRefFrames outside [1, 16]");
  }
  if (config.enable_short_refresh_frame_flags) {
    return set_error(error,
                     "enable_short_refresh_frame_flags == 1 is not modelled");
  }
  if (header.order_hint >= (1U << config.order_hint_bits)) {
    return set_error(error, "order_hint does not fit OrderHintBits");
  }

  const bool key = is_key_role(header.role);
  const bool intra = key || !header.frame_is_inter;
  const std::uint32_t all_frames = (1U << config.num_ref_frames) - 1U;

  if (header.refresh_frame_flags > all_frames) {
    return set_error(error, "refresh_frame_flags exceeds NumRefFrames");
  }
  if (header.role == FrameRole::kClosedLoopKey && config.max_mlayer_id == 0U &&
      header.refresh_frame_flags != all_frames) {
    return set_error(
        error,
        "CLK with max_mlayer_id == 0 infers refresh_frame_flags = allFrames");
  }
  // §6.17: a non-output frame must refresh at least one slot.
  if (!header.immediate_output_frame && effective_refresh_flags(config, header) == 0U) {
    return set_error(error,
                     "immediate_output_frame == 0 requires refresh_frame_flags "
                     "!= 0");
  }

  if (key) {
    if (header.long_term_id < -1 ||
        header.long_term_id > max_long_term_id(config)) {
      return set_error(error,
                       "long_term_id outside [-1, (1 << "
                       "long_term_frame_id_bits) - 2]");
    }
    if (header.num_total_refs != 0U || !header.ref_frame_idx.empty()) {
      return set_error(error, "a key frame codes no ref_frame_idx");
    }
  } else {
    if (header.long_term_id != -1) {
      return set_error(error, "only KEY_FRAME codes long_term_id_plus_1");
    }
    if (!header.ref_long_term_id.empty()) {
      return set_error(error,
                       "num_key_ref_frames is only present for OLK/RAS");
    }
  }

  if (header.role == FrameRole::kOpenLoopKey) {
    // §5.18.2: immediate_output_frame is inferred 0 for OLK.
    if (header.immediate_output_frame) {
      return set_error(error, "OLK infers immediate_output_frame = 0");
    }
    if (header.ref_long_term_id.size() > static_cast<std::size_t>(kRefsPerFrame)) {
      return set_error(error, "num_key_ref_frames exceeds f(3)");
    }
    if (config.long_term_frame_id_bits == 0U &&
        !header.ref_long_term_id.empty()) {
      return set_error(
          error, "ref_long_term_id is absent when long_term_frame_id_bits == 0");
    }
    if (config.olk_long_term_list == LongTermListSyntax::kOmitPinnedDav2d &&
        !header.ref_long_term_id.empty()) {
      return set_error(error,
                       "this layout does not code ref_long_term_id for an OLK");
    }
    for (const std::uint8_t id : header.ref_long_term_id) {
      // §6.17: ref_long_term_id[i] != (1 << long_term_frame_id_bits) - 1.
      if (static_cast<std::int32_t>(id) > max_long_term_id(config)) {
        return set_error(error, "ref_long_term_id is a reserved value");
      }
    }
  }

  // §6.17 also forbids refresh_frame_flags overlapping OlkRefresh for a
  // leading frame; the stateful part is enforced by apply_frame().

  if (!intra) {
    const int max_refs =
        std::min(kRefsPerFrame, static_cast<int>(config.num_ref_frames));
    if (static_cast<int>(header.num_total_refs) > max_refs) {
      return set_error(error, "num_total_refs exceeds ActiveNumRefFrames");
    }
    if (header.ref_frame_idx.size() != header.num_total_refs) {
      return set_error(error, "ref_frame_idx size != num_total_refs");
    }
    for (const std::uint8_t idx : header.ref_frame_idx) {
      if (idx >= config.num_ref_frames) {
        return set_error(error, "ref_frame_idx >= NumRefFrames");
      }
    }
  } else if (header.num_total_refs != 0U || !header.ref_frame_idx.empty()) {
    return set_error(error, "an intra frame codes no num_total_refs");
  }

  if (!key && !header.frame_is_inter && config.num_ref_frames > 1U &&
      header.refresh_frame_flags == all_frames) {
    // §6.17: INTRA_ONLY_FRAME may not refresh every slot.
    return set_error(error,
                     "INTRA_ONLY_FRAME cannot refresh every reference slot");
  }
  return true;
}

// ---------------------------------------------------------------------------
// §5.18.2 + §7.23
// ---------------------------------------------------------------------------
bool apply_frame(const SequenceConfig& config, const FrameHeader& header,
                 bool first_picture_in_tu, PoolState* pool,
                 PoolEvents* events, std::string* error) {
  if (pool == nullptr || events == nullptr) {
    return set_error(error, "apply_frame: null pool/events");
  }
  if (pool->num_ref_frames != config.num_ref_frames ||
      pool->long_term_frame_id_bits != config.long_term_frame_id_bits ||
      pool->order_hint_bits != config.order_hint_bits) {
    return set_error(error, "pool was not reset for this sequence config");
  }
  if (!validate_frame_header(config, header, error)) return false;

  // Work on a copy so a rejected frame leaves the pool unchanged.
  PoolState next = *pool;

  PoolEvents ev;
  ev.valid_before = slot_mask(next);
  const std::uint32_t refresh = effective_refresh_flags(config, header);

  // §5.18.2: startCVS invalidates every reference buffer.
  if (header.role == FrameRole::kClosedLoopKey && first_picture_in_tu) {
    for (int i = 0; i < kNumRefFrames; ++i) {
      next.slots[static_cast<std::size_t>(i)].valid = false;
    }
    next.olk_encountered = false;
    next.olk_refresh = 0;
    next.pending_ltr_count = 0;
    ev.closed_random_access = true;
  }

  // §5.18.2: at the first Regular frame after an OLK, every slot that the OLK
  // did not refresh and that does not hold a long-term reference is dropped.
  if (next.olk_encountered && is_regular_role(header.role) &&
      first_picture_in_tu) {
    std::uint32_t invalidated = 0;
    for (int i = 0; i < config.num_ref_frames; ++i) {
      SlotState& slot = next.slots[static_cast<std::size_t>(i)];
      if ((next.olk_refresh & (1U << i)) == 0U && slot.long_term_id == -1) {
        if (slot.valid) invalidated |= 1U << i;
        slot.valid = false;
      }
    }
    // §7.4.4: each named long-term reference must still be held here.
    for (int i = 0; i < next.pending_ltr_count; ++i) {
      const std::int32_t id =
          next.pending_ltr_ids[static_cast<std::size_t>(i)];
      if (find_slot_with_long_term_id(next, id) < 0) {
        return set_error(
            error,
            "long-term reference id " + std::to_string(id) +
                " is not held at the first Regular frame after the OLK");
      }
    }
    next.pending_ltr_count = 0;
    next.olk_encountered = false;
    next.olk_refresh = 0;
    ev.invalidation_ran = true;
    ev.invalidated_flags = invalidated;
  }

  if (!is_regular_role(header.role)) {
    // §6.17: a leading frame must not refresh a slot in OlkRefresh.
    if ((refresh & next.olk_refresh) != 0U) {
      return set_error(
          error,
          "leading frame refreshes slot(s) " +
              slot_list(refresh & next.olk_refresh) + " that the OLK owns");
    }
    // It must also not destroy a long-term reference the OLK named.
    for (int i = 0; i < config.num_ref_frames; ++i) {
      if ((refresh & (1U << i)) == 0U) continue;
      const SlotState& slot = next.slots[static_cast<std::size_t>(i)];
      if (!slot.valid || slot.long_term_id < 0) continue;
      for (int p = 0; p < next.pending_ltr_count; ++p) {
        if (next.pending_ltr_ids[static_cast<std::size_t>(p)] ==
            slot.long_term_id) {
          return set_error(error,
                           "leading frame would overwrite long-term reference "
                           "id " +
                               std::to_string(slot.long_term_id) +
                               " in slot " + std::to_string(i));
        }
      }
    }
  }

  if (header.role == FrameRole::kOpenLoopKey) {
    // §7.4.4 long-term reference list consistency (see the helper).
    if (!validate_olk_long_term_list(config, header, next, error)) return false;
    ev.open_random_access = first_picture_in_tu;
  }

  if (header.frame_is_inter) {
    for (std::size_t i = 0; i < header.ref_frame_idx.size(); ++i) {
      const std::uint8_t idx = header.ref_frame_idx[i];
      if (!next.slots[idx].valid) {
        return set_error(error, "ref_frame_idx[" + std::to_string(i) +
                                    "] = " + std::to_string(idx) +
                                    " is not RefValid");
      }
    }
  }

  // §7.23 reference frame update process.
  if (!next.has_frame) {
    next.frame_counter = 0;
  } else {
    ++next.frame_counter;
  }
  const bool key = is_key_role(header.role);
  bool first = true;
  int refreshed_count = 0;
  std::uint32_t refreshed = 0;
  for (int i = 0; i < config.num_ref_frames; ++i) {
    if ((refresh & (1U << i)) == 0U) continue;
    SlotState& slot = next.slots[static_cast<std::size_t>(i)];
    // RefValid = (FrameType == KEY_FRAME || SWITCH_FRAME) ? first : 1.
    slot.valid = key ? first : true;
    slot.long_term_id = key ? header.long_term_id : -1;
    slot.order_hint = header.order_hint;
    slot.key_frame = key;
    slot.intra_only = !header.frame_is_inter;
    slot.frame_counter = next.frame_counter;
    first = false;
    ++refreshed_count;
    refreshed |= 1U << i;
  }
  ev.first_refreshed_slot_only = key && refreshed_count > 1;
  ev.refreshed_flags = refreshed;

  if (header.role == FrameRole::kOpenLoopKey) {
    next.olk_encountered = true;
    next.olk_refresh = refresh;
    next.olk_order_hint = header.order_hint;
    next.pending_ltr_count = static_cast<int>(header.ref_long_term_id.size());
    for (int i = 0; i < next.pending_ltr_count; ++i) {
      next.pending_ltr_ids[static_cast<std::size_t>(i)] =
          static_cast<std::int32_t>(
              header.ref_long_term_id[static_cast<std::size_t>(i)]);
    }
  } else if (header.role == FrameRole::kClosedLoopKey) {
    next.olk_encountered = false;
    next.olk_refresh = 0;
    next.pending_ltr_count = 0;
  }

  next.has_frame = true;
  next.has_started =
      next.has_started || header.role == FrameRole::kClosedLoopKey;
  ev.valid_after = slot_mask(next);
  *pool = next;
  *events = ev;
  return true;
}

// ---------------------------------------------------------------------------
// §7.3.6 / §7.3.7 temporal-unit rules
// ---------------------------------------------------------------------------
bool validate_temporal_unit(const SequenceConfig& config,
                            std::span<const FrameHeader> frames,
                            std::string* error) {
  if (frames.empty()) return set_error(error, "empty temporal unit");
  for (std::size_t i = 0; i < frames.size(); ++i) {
    if (!validate_frame_header(config, frames[i], error)) return false;
  }

  int clk = 0;
  int olk = 0;
  int leading = 0;
  int output_units = 0;
  int non_output_units = 0;
  std::uint32_t output_order_hint = 0;
  bool have_output_hint = false;

  for (std::size_t i = 0; i < frames.size(); ++i) {
    const FrameHeader& f = frames[i];
    if (is_random_access_role(f.role) && i != 0) {
      return set_error(error,
                       "only the first coded frame unit can be a CLK/OLK");
    }
    if (f.role == FrameRole::kClosedLoopKey) ++clk;
    if (f.role == FrameRole::kOpenLoopKey) ++olk;
    if (f.role == FrameRole::kLeading) ++leading;
    const bool output = f.immediate_output_frame && f.role != FrameRole::kOpenLoopKey;
    if (output) {
      ++output_units;
      if (have_output_hint && f.order_hint != output_order_hint) {
        return set_error(
            error, "all coded output frame units in a TU share one OrderHint");
      }
      output_order_hint = f.order_hint;
      have_output_hint = true;
    } else {
      ++non_output_units;
    }
  }
  if (clk > 0 && olk > 0) {
    return set_error(error, "a coded extended layer unit cannot mix CLK and OLK");
  }
  if (leading > 0 && leading != static_cast<int>(frames.size())) {
    return set_error(error,
                     "a TU containing a leading frame contains only leading "
                     "frames");
  }
  if (output_units == 0) {
    return set_error(error,
                     "every temporal unit needs at least one coded output "
                     "frame unit" +
                         std::string(non_output_units > 0
                                         ? " (a non-output OLK must share its TU "
                                           "with an output Regular frame)"
                                         : ""));
  }
  return true;
}

bool validate_closed_random_access(const SequenceConfig& config,
                                   const FrameHeader& clk, std::string* error) {
  if (clk.role != FrameRole::kClosedLoopKey) {
    return set_error(error, "closed random access requires OBU_CLOSED_LOOP_KEY");
  }
  if (!validate_frame_header(config, clk, error)) return false;
  if (!clk.ref_long_term_id.empty()) {
    return set_error(error,
                     "a CLK never codes num_key_ref_frames/ref_long_term_id");
  }
  // §7.4.3: the closed random access process invalidates every reference
  // buffer, so a CLK never consumes prior state. apply_frame() performs it.
  return true;
}

bool validate_open_random_access(
    const SequenceConfig& config, const FrameHeader& olk,
    const PoolState& pool, std::span<const std::int32_t> required_long_term_ids,
    std::string* error) {
  if (olk.role != FrameRole::kOpenLoopKey) {
    return set_error(error, "open random access requires OBU_OPEN_LOOP_KEY");
  }
  if (!validate_frame_header(config, olk, error)) return false;
  if (olk.refresh_frame_flags == 0U) {
    return set_error(error, "an OLK must refresh at least one reference slot");
  }
  // §7.4.4 long-term reference condition: every id that later frames need is
  // named, and every named id is held in a valid slot right now.
  for (const std::int32_t id : required_long_term_ids) {
    if (!named_long_term_id(olk, id)) {
      return set_error(error,
                       "OLK does not name required long-term reference id " +
                           std::to_string(id));
    }
  }
  return validate_olk_long_term_list(config, olk, pool, error);
}

bool is_leading_candidate(const PoolState& pool, std::uint32_t order_hint) {
  return pool.olk_encountered && order_hint < pool.olk_order_hint;
}

// ---------------------------------------------------------------------------
// Planners
// ---------------------------------------------------------------------------
bool plan_closed_loop_key(const SequenceConfig& config,
                          std::uint32_t order_hint, std::int32_t long_term_id,
                          FrameHeader* header, std::string* error) {
  if (header == nullptr) return set_error(error, "null header");
  *header = FrameHeader{};
  header->role = FrameRole::kClosedLoopKey;
  header->frame_is_inter = false;
  header->order_hint = order_hint;
  header->long_term_id = long_term_id;
  header->refresh_frame_flags = (1U << config.num_ref_frames) - 1U;
  header->immediate_output_frame = true;
  return validate_closed_random_access(config, *header, error);
}

bool plan_open_loop_key(const SequenceConfig& config, const PoolState& pool,
                        std::uint32_t order_hint, std::uint8_t olk_slot,
                        std::int32_t olk_long_term_id,
                        std::span<const std::int32_t> retain_long_term_ids,
                        FrameHeader* header, std::string* error) {
  if (header == nullptr) return set_error(error, "null header");
  if (olk_slot >= config.num_ref_frames) {
    return set_error(error, "OLK refresh slot >= NumRefFrames");
  }
  *header = FrameHeader{};
  header->role = FrameRole::kOpenLoopKey;
  header->frame_is_inter = false;
  header->order_hint = order_hint;
  header->long_term_id = olk_long_term_id;
  header->refresh_frame_flags = 1U << olk_slot;
  header->immediate_output_frame = false;
  header->num_total_refs = 0;
  header->ref_long_term_id.reserve(retain_long_term_ids.size());
  for (const std::int32_t id : retain_long_term_ids) {
    header->ref_long_term_id.push_back(static_cast<std::uint8_t>(id));
  }
  if (!validate_open_random_access(config, *header, pool, retain_long_term_ids,
                                   error)) {
    return false;
  }
  // The OLK's own slot must not be one of the retained long-term slots.
  for (const std::uint8_t value : header->ref_long_term_id) {
    for (int i = 0; i < config.num_ref_frames; ++i) {
      if (i != olk_slot) continue;
      const SlotState& slot = pool.slots[static_cast<std::size_t>(i)];
      if (slot.valid && slot.long_term_id == static_cast<std::int32_t>(value)) {
        return set_error(error,
                         "OLK slot " + std::to_string(olk_slot) +
                             " is the retained long-term slot");
      }
    }
  }
  return true;
}

bool plan_inter_frame(const SequenceConfig& config, FrameRole role,
                      std::uint32_t order_hint,
                      std::span<const std::uint8_t> ref_slots,
                      std::uint8_t refresh_slot, bool immediate_output_frame,
                      FrameHeader* header, std::string* error) {
  if (header == nullptr) return set_error(error, "null header");
  if (role != FrameRole::kRegular && role != FrameRole::kLeading) {
    return set_error(error, "plan_inter_frame needs a regular/leading role");
  }
  if (refresh_slot >= config.num_ref_frames) {
    return set_error(error, "refresh slot >= NumRefFrames");
  }
  *header = FrameHeader{};
  header->role = role;
  header->frame_is_inter = true;
  header->order_hint = order_hint;
  header->long_term_id = -1;
  header->refresh_frame_flags = 1U << refresh_slot;
  header->immediate_output_frame = immediate_output_frame;
  header->num_total_refs = static_cast<std::uint8_t>(ref_slots.size());
  header->ref_frame_idx.assign(ref_slots.begin(), ref_slots.end());
  return validate_frame_header(config, *header, error);
}

}  // namespace av2enc::inter::longterm
