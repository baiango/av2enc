#include "frame/ref_fields/ref_fields.h"

#include <algorithm>
#include <cstdlib>
#include <limits>

namespace av2enc::frame::ref_fields {
namespace {

bool fail(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

std::string u32(std::uint32_t value) { return std::to_string(value); }

// §4.11.7 ns( n ) write.
bool put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value,
            std::string* error) {
  if (n < 2U) return fail(error, "ns(n): n must be >= 2");
  if (value >= n) {
    return fail(error, "ns(" + u32(n) + "): value " + u32(value) +
                           " is out of range");
  }
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
  return true;
}

std::uint32_t all_frames_mask(std::uint32_t num_ref_frames) {
  if (num_ref_frames >= 32U) return std::numeric_limits<std::uint32_t>::max();
  return (1U << num_ref_frames) - 1U;
}

std::int64_t get_relative_dist(std::int32_t a, std::int32_t b,
                               const FrameRefState& state) {
  if (!state.enable_order_hint) return 0;
  if (state.order_hint_bits == 0 || state.order_hint_bits >= 32U) {
    return static_cast<std::int64_t>(a) - b;
  }
  const std::int64_t diff = static_cast<std::int64_t>(a) - b;
  const std::int64_t m = static_cast<std::int64_t>(1) << (state.order_hint_bits - 1U);
  return (diff & (m - 1)) - (diff & m);
}

bool is_ref_better(std::int32_t ref_disp, std::int32_t best_disp,
                   std::int64_t ref_ratio, std::int64_t best_ratio,
                   std::int32_t current_hint, const FrameRefState& state) {
  const std::int64_t d0 =
      std::abs(get_relative_dist(current_hint, ref_disp, state)) -
      (ref_ratio << 1);
  const std::int64_t d1 =
      std::abs(get_relative_dist(current_hint, best_disp, state)) -
      (best_ratio << 1);
  if (d0 < d1) return true;
  if (d0 == d1 &&
      get_relative_dist(ref_disp, best_disp, state) > 0) {
    return true;
  }
  return false;
}

std::int64_t floor_log2_u64(std::uint64_t value) {
  std::int64_t result = 0;
  while (value > 1U) {
    value >>= 1U;
    ++result;
  }
  return result;
}

}  // namespace

const char* to_string(FrameType type) {
  switch (type) {
    case FrameType::kKey:
      return "KEY_FRAME";
    case FrameType::kInter:
      return "INTER_FRAME";
    case FrameType::kIntraOnly:
      return "INTRA_ONLY_FRAME";
    case FrameType::kSwitch:
      return "SWITCH_FRAME";
  }
  return "unknown";
}

unsigned ceil_log2(std::uint32_t value) {
  if (value <= 1U) return 0U;
  unsigned bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0U) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0U) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

// ---------------------------------------------------------------------------
// RefBitReader
// ---------------------------------------------------------------------------
RefBitReader::RefBitReader(std::span<const std::uint8_t> bytes,
                           std::size_t bit_position)
    : bytes_(bytes), bit_position_(bit_position) {}

bool RefBitReader::bit(bool* out) {
  if (bit_position_ >= total_bits()) return false;
  const std::size_t byte = bit_position_ >> 3U;
  const unsigned shift = 7U - static_cast<unsigned>(bit_position_ & 7U);
  *out = ((bytes_[byte] >> shift) & 1U) != 0U;
  ++bit_position_;
  return true;
}

bool RefBitReader::bits(unsigned count, std::uint32_t* out) {
  if (count > 32U) return false;
  if (remaining_bits() < count) return false;
  std::uint32_t value = 0;
  for (unsigned i = 0; i < count; ++i) {
    bool b = false;
    if (!bit(&b)) return false;
    value = (value << 1U) | (b ? 1U : 0U);
  }
  *out = value;
  return true;
}

bool RefBitReader::uvlc(std::uint32_t* out) {
  unsigned zeros = 0;
  bool done = false;
  while (true) {
    if (!bit(&done)) return false;
    if (done) break;
    ++zeros;
    if (zeros >= 32U) return false;
  }
  std::uint32_t payload = 0;
  if (zeros > 0 && !bits(zeros, &payload)) return false;
  *out = payload + (1U << zeros) - 1U;
  return true;
}

bool RefBitReader::ns(std::uint32_t n, std::uint32_t* out) {
  if (n < 2U) return false;
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  if (w > 32U) return false;
  const std::uint32_t m = (1U << w) - n;
  std::uint32_t value = 0;
  if (w > 1U && !bits(w - 1U, &value)) return false;
  if (value < m) {
    *out = value;
    return true;
  }
  bool extra = false;
  if (!bit(&extra)) return false;
  const std::uint32_t x = (value << 1U) | (extra ? 1U : 0U);
  *out = x - m;
  return true;
}

bool RefBitReader::su(unsigned count, std::int32_t* out) {
  if (count == 0 || count > 32U) return false;
  std::uint32_t value = 0;
  if (!bits(count, &value)) return false;
  if ((value & (1U << (count - 1U))) != 0U) {
    *out = static_cast<std::int32_t>(value) -
           static_cast<std::int32_t>(1U << count);
  } else {
    *out = static_cast<std::int32_t>(value);
  }
  return true;
}

std::size_t RefBitReader::remaining_bits() const {
  return total_bits() - std::min(bit_position_, total_bits());
}

bool RefBitReader::seek(std::size_t bit_position) {
  if (bit_position > total_bits()) return false;
  bit_position_ = bit_position;
  return true;
}

// ---------------------------------------------------------------------------
// Reference state
// ---------------------------------------------------------------------------
bool slot_constraints_ok(const FrameRefState& state, std::uint32_t slot,
                         std::string* error) {
  if (slot >= kNumRefFrames) {
    return fail(error, "reference slot " + u32(slot) +
                           " is outside NUM_REF_FRAMES");
  }
  if (!state.slot[slot].valid) {
    return fail(error, "reference slot " + u32(slot) + " is not RefValid");
  }
  return true;
}

// ---------------------------------------------------------------------------
// primary_ref_frame
// ---------------------------------------------------------------------------
void choose_primary_secondary_ref_frame(const RefList& refs,
                                        const FrameRefState& state,
                                        std::uint8_t base_q_idx,
                                        std::uint32_t* primary,
                                        std::uint32_t* secondary) {
  *primary = kPrimaryRefNone;
  *secondary = kPrimaryRefNone;
  std::uint32_t primary_qp_diff = 512;
  std::uint32_t secondary_qp_diff = 512;
  std::int32_t primary_d = 0;
  std::int32_t secondary_d = 0;
  std::int64_t primary_ratio = 0;
  std::int64_t secondary_ratio = 0;

  for (std::uint32_t i = 0; i < refs.num_total_refs; ++i) {
    const std::uint32_t idx = refs.index[i];
    if (idx >= kNumRefFrames) continue;
    const RefSlot& slot = state.slot[idx];
    // first_slot_with_ref(idx) is a decoder-internal dedup rule keyed on
    // RefCounter; this module does not model RefCounter, so every valid slot
    // is treated as the first slot that holds its frame.
    if (slot.frame_type != FrameType::kInter || !slot.valid ||
        slot.order_hint == kRestrictedOrderHint) {
      continue;
    }
    const std::uint32_t q = slot.base_q_idx;
    const std::int32_t d = slot.order_hint;
    const std::int64_t d_ratio = floor_log2_u64(
        static_cast<std::uint64_t>(slot.width) * slot.height);
    const std::uint32_t qp_diff =
        q > base_q_idx ? q - base_q_idx : base_q_idx - q;
    const bool better_primary =
        qp_diff < primary_qp_diff ||
        (qp_diff == primary_qp_diff &&
         is_ref_better(d, primary_d, d_ratio, primary_ratio,
                       state.order_hint, state));
    if (better_primary) {
      *secondary = *primary;
      secondary_qp_diff = primary_qp_diff;
      secondary_d = primary_d;
      secondary_ratio = primary_ratio;
      *primary = i;
      primary_qp_diff = qp_diff;
      primary_d = d;
      primary_ratio = d_ratio;
    } else if (qp_diff < secondary_qp_diff ||
               (qp_diff == secondary_qp_diff &&
                is_ref_better(d, secondary_d, d_ratio, secondary_ratio,
                              state.order_hint, state))) {
      *secondary = i;
      secondary_qp_diff = qp_diff;
      secondary_d = d;
      secondary_ratio = d_ratio;
    }
  }
}

bool write_primary_ref_signal(BitWriter& writer, FrameType frame_type,
                              bool is_tip, const PrimaryRefSignal& signal,
                              std::string* error) {
  if (error != nullptr) error->clear();
  if (frame_type == FrameType::kKey || frame_type == FrameType::kIntraOnly ||
      frame_type == FrameType::kSwitch) {
    if (signal.signal_primary_ref_frame) {
      return fail(error,
                  "primary_ref_frame cannot be signalled for " +
                      std::string(to_string(frame_type)));
    }
    return true;
  }
  if (signal.signal_primary_ref_frame) {
    if (signal.primary_ref_frame >= kPrimaryRefNone) {
      return fail(error, "primary_ref_frame " +
                             u32(signal.primary_ref_frame) +
                             " does not fit f(3) (must be < 7)");
    }
    writer.put_bit(true);  // signal_primary_ref_frame
    if (!is_tip) writer.put_bit(signal.disable_cross_frame_cdf_init);
    writer.put_bits(signal.primary_ref_frame, 3U);
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    if (!is_tip) writer.put_bit(signal.disable_cross_frame_cdf_init);
  }
  return true;
}

bool read_primary_ref_signal(RefBitReader& reader, FrameType frame_type,
                             bool is_tip, PrimaryRefSignal* signal,
                             std::string* error) {
  if (error != nullptr) error->clear();
  if (signal == nullptr) return fail(error, "null primary-ref output");
  *signal = PrimaryRefSignal{};
  if (frame_type == FrameType::kKey || frame_type == FrameType::kIntraOnly ||
      frame_type == FrameType::kSwitch) {
    signal->primary_ref_frame = kPrimaryRefNone;
    signal->disable_cross_frame_cdf_init = true;
    return true;
  }
  bool signalled = false;
  if (!reader.bit(&signalled)) {
    return fail(error, "truncated signal_primary_ref_frame");
  }
  signal->signal_primary_ref_frame = signalled;
  if (!is_tip) {
    bool disable = true;
    if (!reader.bit(&disable)) {
      return fail(error, "truncated disable_cross_frame_cdf_init");
    }
    signal->disable_cross_frame_cdf_init = disable;
  } else {
    // Absent for TIP frames; the §5.18.2 process keeps the initial value,
    // which is 0 unless a frame-header reset already forced 1.
    signal->disable_cross_frame_cdf_init = false;
  }
  if (signalled) {
    std::uint32_t value = 0;
    if (!reader.bits(3U, &value)) {
      return fail(error, "truncated primary_ref_frame");
    }
    if (value >= kPrimaryRefNone) {
      return fail(error, "primary_ref_frame " + u32(value) +
                             " is a reserved value (PRIMARY_REF_NONE is not "
                             "codable in f(3))");
    }
    signal->primary_ref_frame = value;
  } else {
    signal->primary_ref_frame = kPrimaryRefChoose;
  }
  return true;
}

bool resolve_primary_ref(FrameType frame_type, const PrimaryRefSignal& signal,
                         const RefList& refs, const FrameRefState& state,
                         std::uint8_t base_q_idx, PrimaryRefResolved* out,
                         std::string* error) {
  if (error != nullptr) error->clear();
  if (out == nullptr) return fail(error, "null primary-ref resolution");
  *out = PrimaryRefResolved{};
  out->coded_disable_cross_frame_cdf_init =
      signal.disable_cross_frame_cdf_init;

  if (frame_type == FrameType::kKey || frame_type == FrameType::kIntraOnly ||
      frame_type == FrameType::kSwitch) {
    if (signal.signal_primary_ref_frame) {
      return fail(error, "primary_ref_frame cannot be signalled for " +
                             std::string(to_string(frame_type)));
    }
    out->primary_ref_frame = kPrimaryRefNone;
    out->derived_primary_ref_frame = kPrimaryRefNone;
    out->disable_cross_frame_cdf_init = true;
    return true;
  }

  if (refs.num_total_refs > kRefsPerFrame) {
    return fail(error, "NumTotalRefs exceeds REFS_PER_FRAME");
  }
  for (std::uint32_t i = 0; i < refs.num_total_refs; ++i) {
    std::string slot_error;
    if (!slot_constraints_ok(state, refs.index[i], &slot_error)) {
      return fail(error, "ref_frame_idx[" + u32(i) + "]: " + slot_error);
    }
  }

  std::uint32_t primary = kPrimaryRefNone;
  std::uint32_t secondary = kPrimaryRefNone;
  choose_primary_secondary_ref_frame(refs, state, base_q_idx, &primary,
                                     &secondary);

  std::uint32_t chosen_primary = primary;
  std::uint32_t chosen_secondary = secondary;
  if (signal.signal_primary_ref_frame) {
    const std::uint32_t signalled = signal.primary_ref_frame;
    if (signalled >= kPrimaryRefNone) {
      return fail(error, "signalled primary_ref_frame " + u32(signalled) +
                             " is not a reference-list index");
    }
    if (signalled >= refs.num_total_refs) {
      return fail(error, "primary_ref_frame " + u32(signalled) +
                             " must be < NumTotalRefs (" +
                             u32(refs.num_total_refs) + ")");
    }
    const RefSlot& slot = state.slot[refs.index[signalled]];
    if (slot.order_hint == kRestrictedOrderHint) {
      return fail(error,
                  "primary_ref_frame selects a RESTRICTED_OH reference");
    }
    // §5.18.2 choose_primary_secondary_ref_frame(): an explicit signal
    // replaces the derived primary and demotes it to the secondary.
    if (signalled != chosen_primary) {
      if (chosen_secondary == kPrimaryRefNone ||
          chosen_secondary == signalled) {
        chosen_secondary = chosen_primary;
      }
      chosen_primary = signalled;
    }
  }

  out->derived_primary_ref_frame = chosen_primary;
  out->secondary_ref_frame = chosen_secondary;
  out->primary_ref_frame = chosen_primary;

  // §5.18.2: when the derived primary reference is PRIMARY_REF_NONE the
  // process forces primary_ref_frame = PRIMARY_REF_NONE and
  // disable_cross_frame_cdf_init = 1.
  if (chosen_primary == kPrimaryRefNone) {
    out->primary_ref_frame = kPrimaryRefNone;
    out->derived_primary_ref_frame = kPrimaryRefNone;
    out->disable_cross_frame_cdf_init = true;
    out->load_cdfs_from_reference = false;
    out->load_previous_state = false;
    return true;
  }

  out->disable_cross_frame_cdf_init = signal.disable_cross_frame_cdf_init;
  if (!out->disable_cross_frame_cdf_init) {
    // load_cdfs( ref_frame_idx[primary_ref_frame] ) and load_previous().
    // prevFrame is ref_frame_idx[DerivedPrimaryRefFrame]; with the forcing
    // above it equals the signalled/derived primary.
    const std::uint32_t primary_slot = refs.index[chosen_primary];
    if (!state.slot[primary_slot].valid) {
      return fail(error, "primary reference slot is not valid");
    }
    if (state.slot[primary_slot].order_hint == kRestrictedOrderHint) {
      return fail(error, "primary reference has RESTRICTED_OH");
    }
    out->load_cdfs_from_reference = true;
    out->cdf_reference_slot = static_cast<std::uint8_t>(primary_slot);
    out->load_previous_state = true;
  }
  return true;
}

// ---------------------------------------------------------------------------
// DRL fields
// ---------------------------------------------------------------------------
namespace {

struct DrlRules {
  bool allow_change = false;
  std::uint32_t seq_minus_1 = 0;
  std::uint32_t max_value = 0;  // inclusive bound on the resolved value
  const char* label = "max_drl_bits_minus_1";
};

bool validate_drl_config(const DrlRules& rules, std::string* error) {
  if (rules.seq_minus_1 > rules.max_value) {
    return fail(error, std::string(rules.label) +
                           ": sequence value " + u32(rules.seq_minus_1) +
                           " exceeds the stack bound " + u32(rules.max_value));
  }
  return true;
}

bool write_drl_impl(BitWriter& writer, const DrlRules& rules, bool change,
                    std::uint32_t value, bool* out_change,
                    std::uint32_t* out_value, std::string* error) {
  if (!validate_drl_config(rules, error)) return false;
  if (value > rules.max_value) {
    return fail(error, std::string(rules.label) + " " + u32(value) +
                           " exceeds the stack bound " + u32(rules.max_value));
  }
  if (change && !rules.allow_change) {
    return fail(error, std::string("change_") + rules.label +
                           " requires the matching sequence flag");
  }
  if (!rules.allow_change) {
    if (value != rules.seq_minus_1) {
      return fail(error, std::string(rules.label) +
                             " cannot change: the sequence header does not "
                             "allow it");
    }
    *out_change = false;
    *out_value = value;
    return true;
  }
  if (value == rules.seq_minus_1) {
    writer.put_bit(false);  // change_*
    *out_change = false;
    *out_value = value;
    return true;
  }
  // Invert: value = v >= seq ? v + 1 : v.
  std::uint32_t v = 0;
  if (value > rules.seq_minus_1) {
    v = value - 1U;
    if (v < rules.seq_minus_1) {
      return fail(error, std::string(rules.label) + " is not representable");
    }
  } else {
    v = value;
    if (v >= rules.seq_minus_1) {
      return fail(error, std::string(rules.label) + " is not representable");
    }
  }
  const std::uint32_t ns_n = rules.max_value;  // MAX_*_STACK_SIZE - 2
  if (v >= ns_n) {
    return fail(error, std::string(rules.label) + " change value " + u32(v) +
                           " is out of ns() range");
  }
  writer.put_bit(true);  // change_*
  if (!put_ns(writer, ns_n, v, error)) return false;
  *out_change = true;
  *out_value = value;
  return true;
}

bool read_drl_impl(RefBitReader& reader, const DrlRules& rules, bool* change,
                   std::uint32_t* value, std::string* error) {
  if (!validate_drl_config(rules, error)) return false;
  *change = false;
  *value = rules.seq_minus_1;
  if (!rules.allow_change) return true;
  bool changed = false;
  if (!reader.bit(&changed)) {
    return fail(error, std::string("truncated change_") + rules.label);
  }
  *change = changed;
  if (!changed) return true;
  const std::uint32_t ns_n = rules.max_value;
  std::uint32_t v = 0;
  if (!reader.ns(ns_n, &v)) {
    return fail(error, std::string("truncated ") + rules.label);
  }
  std::uint32_t resolved = v;
  if (resolved >= rules.seq_minus_1) resolved += 1U;
  if (resolved > rules.max_value) {
    return fail(error, std::string(rules.label) + " resolves to " +
                           u32(resolved) + ", above the stack bound " +
                           u32(rules.max_value));
  }
  *value = resolved;
  return true;
}

}  // namespace

bool write_drl(BitWriter& writer, const DrlConfig& config,
               const DrlSignal& signal, std::string* error) {
  if (error != nullptr) error->clear();
  DrlRules rules;
  rules.allow_change = config.allow_frame_max_drl_bits;
  rules.seq_minus_1 = config.seq_max_drl_bits_minus_1;
  rules.max_value = kMaxRefMvStackSize - 2U;
  rules.label = "max_drl_bits_minus_1";
  bool change = false;
  std::uint32_t value = 0;
  return write_drl_impl(writer, rules, signal.change_drl,
                        signal.max_drl_bits_minus_1, &change, &value, error);
}

bool read_drl(RefBitReader& reader, const DrlConfig& config, DrlSignal* signal,
              std::string* error) {
  if (error != nullptr) error->clear();
  if (signal == nullptr) return fail(error, "null DRL output");
  DrlRules rules;
  rules.allow_change = config.allow_frame_max_drl_bits;
  rules.seq_minus_1 = config.seq_max_drl_bits_minus_1;
  rules.max_value = kMaxRefMvStackSize - 2U;
  rules.label = "max_drl_bits_minus_1";
  return read_drl_impl(reader, rules, &signal->change_drl,
                       &signal->max_drl_bits_minus_1, error);
}

bool write_bvp_drl(BitWriter& writer, const BvDrlConfig& config,
                   const BvDrlSignal& signal, std::string* error) {
  if (error != nullptr) error->clear();
  DrlRules rules;
  rules.allow_change = config.allow_frame_max_bvp_drl_bits;
  rules.seq_minus_1 = config.seq_max_bvp_drl_bits_minus_1;
  rules.max_value = kMaxRefBvStackSize - 2U;
  rules.label = "max_bvp_drl_bits_minus_1";
  bool change = false;
  std::uint32_t value = 0;
  return write_drl_impl(writer, rules, signal.change_bvp_drl,
                        signal.max_bvp_drl_bits_minus_1, &change, &value,
                        error);
}

bool read_bvp_drl(RefBitReader& reader, const BvDrlConfig& config,
                  BvDrlSignal* signal, std::string* error) {
  if (error != nullptr) error->clear();
  if (signal == nullptr) return fail(error, "null BVP DRL output");
  DrlRules rules;
  rules.allow_change = config.allow_frame_max_bvp_drl_bits;
  rules.seq_minus_1 = config.seq_max_bvp_drl_bits_minus_1;
  rules.max_value = kMaxRefBvStackSize - 2U;
  rules.label = "max_bvp_drl_bits_minus_1";
  return read_drl_impl(reader, rules, &signal->change_bvp_drl,
                       &signal->max_bvp_drl_bits_minus_1, error);
}

// ---------------------------------------------------------------------------
// refresh_frame_flags
// ---------------------------------------------------------------------------
bool validate_refresh_frame_flags(FrameType frame_type,
                                  const RefreshConfig& config,
                                  const RefreshSignal& signal,
                                  bool immediate_output_frame,
                                  std::string* error) {
  if (config.num_ref_frames < 1U || config.num_ref_frames > kNumRefFrames) {
    return fail(error, "NumRefFrames must be in [1, NUM_REF_FRAMES]");
  }
  const std::uint32_t all = all_frames_mask(config.num_ref_frames);
  const bool key = frame_type == FrameType::kKey;
  const bool is_switch = frame_type == FrameType::kSwitch;

  if (frame_type == FrameType::kIntraOnly && config.num_ref_frames > 1U &&
      signal.refresh_frame_flags == all) {
    return fail(error,
                "INTRA_ONLY_FRAME must not refresh all reference slots");
  }
  if (!immediate_output_frame && signal.refresh_frame_flags == 0U) {
    return fail(error,
                "a frame that is not output immediately must refresh at least "
                "one slot");
  }
  if (key && config.obu_is_closed_loop_key && config.max_mlayer_id_zero) {
    if (signal.has_refresh_frame_flags) {
      return fail(error, "has_refresh_frame_flags is not coded for a CLK");
    }
    if (signal.refresh_frame_flags != all) {
      return fail(error,
                  "a CLK infers refresh_frame_flags = allFrames");
    }
    return true;
  }
  if (signal.refresh_frame_flags > all) {
    return fail(error, "refresh_frame_flags " +
                           u32(signal.refresh_frame_flags) +
                           " exceeds NumRefFrames");
  }
  const bool short_refresh = config.enable_short_refresh_frame_flags &&
                             !is_switch;
  if (short_refresh && !key) {
    if (signal.has_refresh_frame_flags) {
      if (signal.frame_to_refresh >= config.num_ref_frames) {
        return fail(error, "frame_to_refresh " +
                               u32(signal.frame_to_refresh) +
                               " must be < NumRefFrames");
      }
      if (signal.refresh_frame_flags != (1U << signal.frame_to_refresh)) {
        return fail(error,
                    "refresh_frame_flags must equal 1 << frame_to_refresh");
      }
    } else if (signal.refresh_frame_flags != 0U) {
      return fail(error,
                  "has_refresh_frame_flags == 0 infers "
                  "refresh_frame_flags == 0");
    }
    return true;
  }
  if (signal.has_refresh_frame_flags) {
    return fail(error,
                "has_refresh_frame_flags is only coded when "
                "enable_short_refresh_frame_flags is 1 and the frame is "
                "neither KEY nor SWITCH");
  }
  if (short_refresh && key) {
    if (signal.frame_to_refresh >= config.num_ref_frames) {
      return fail(error, "frame_to_refresh " +
                             u32(signal.frame_to_refresh) +
                             " must be < NumRefFrames");
    }
    if (signal.refresh_frame_flags != (1U << signal.frame_to_refresh)) {
      return fail(error,
                  "a short-refresh KEY_FRAME writes 1 << frame_to_refresh");
    }
  }
  return true;
}

namespace {

// Fills `signal->frame_to_refresh` from a single-bit refresh mask.
bool derive_frame_to_refresh(const RefreshConfig& config,
                             RefreshSignal* signal, std::string* error) {
  const std::uint32_t mask = signal->refresh_frame_flags;
  std::uint32_t index = 0;
  bool found = false;
  for (std::uint32_t i = 0; i < config.num_ref_frames; ++i) {
    if (mask == (1U << i)) {
      index = i;
      found = true;
      break;
    }
  }
  if (!found) {
    return fail(error, "refresh_frame_flags " + u32(mask) +
                           " is not a single refresh slot");
  }
  signal->frame_to_refresh = index;
  return true;
}

}  // namespace

bool write_refresh_frame_flags(BitWriter& writer, FrameType frame_type,
                               const RefreshConfig& config,
                               const RefreshSignal& signal,
                               bool immediate_output_frame,
                               std::string* error) {
  if (error != nullptr) error->clear();
  if (config.num_ref_frames < 1U || config.num_ref_frames > kNumRefFrames) {
    return fail(error, "NumRefFrames must be in [1, NUM_REF_FRAMES]");
  }
  RefreshSignal resolved = signal;
  const std::uint32_t all = all_frames_mask(config.num_ref_frames);
  const bool key = frame_type == FrameType::kKey;
  const bool is_switch = frame_type == FrameType::kSwitch;
  const unsigned n = ceil_log2(config.num_ref_frames);

  if (key && config.obu_is_closed_loop_key && config.max_mlayer_id_zero) {
    if (resolved.refresh_frame_flags == 0U) {
      resolved.refresh_frame_flags = all;  // inferred allFrames
    }
    if (!validate_refresh_frame_flags(frame_type, config, resolved,
                                      immediate_output_frame, error)) {
      return false;
    }
    return true;  // no bits
  }
  if (key && config.enable_short_refresh_frame_flags) {
    if (!derive_frame_to_refresh(config, &resolved, error)) return false;
    if (!validate_refresh_frame_flags(frame_type, config, resolved,
                                      immediate_output_frame, error)) {
      return false;
    }
    writer.put_bits(resolved.frame_to_refresh, n);
    return true;
  }
  if (!key && !is_switch && config.enable_short_refresh_frame_flags) {
    resolved.has_refresh_frame_flags = resolved.refresh_frame_flags != 0U;
    if (resolved.has_refresh_frame_flags) {
      if (!derive_frame_to_refresh(config, &resolved, error)) return false;
    } else {
      resolved.frame_to_refresh = 0;
    }
    if (!validate_refresh_frame_flags(frame_type, config, resolved,
                                      immediate_output_frame, error)) {
      return false;
    }
    writer.put_bit(resolved.has_refresh_frame_flags);
    if (resolved.has_refresh_frame_flags) {
      writer.put_bits(resolved.frame_to_refresh, n);
    }
    return true;
  }
  resolved.has_refresh_frame_flags = false;
  if (!validate_refresh_frame_flags(frame_type, config, resolved,
                                    immediate_output_frame, error)) {
    return false;
  }
  writer.put_bits(resolved.refresh_frame_flags, config.num_ref_frames);
  return true;
}

bool read_refresh_frame_flags(RefBitReader& reader, FrameType frame_type,
                              const RefreshConfig& config,
                              bool immediate_output_frame,
                              RefreshSignal* signal, std::string* error) {
  if (error != nullptr) error->clear();
  if (signal == nullptr) return fail(error, "null refresh output");
  *signal = RefreshSignal{};
  if (config.num_ref_frames < 1U || config.num_ref_frames > kNumRefFrames) {
    return fail(error, "NumRefFrames must be in [1, NUM_REF_FRAMES]");
  }
  const std::uint32_t all = all_frames_mask(config.num_ref_frames);
  const bool key = frame_type == FrameType::kKey;
  const bool is_switch = frame_type == FrameType::kSwitch;
  const unsigned n = ceil_log2(config.num_ref_frames);

  if (key && config.obu_is_closed_loop_key && config.max_mlayer_id_zero) {
    signal->refresh_frame_flags = all;
  } else if (key && config.enable_short_refresh_frame_flags) {
    std::uint32_t value = 0;
    if (!reader.bits(n, &value)) {
      return fail(error, "truncated frame_to_refresh");
    }
    if (value >= config.num_ref_frames) {
      return fail(error, "frame_to_refresh " + u32(value) +
                             " must be < NumRefFrames");
    }
    signal->frame_to_refresh = value;
    signal->refresh_frame_flags = 1U << value;
  } else if (!key && !is_switch && config.enable_short_refresh_frame_flags) {
    bool has = false;
    if (!reader.bit(&has)) {
      return fail(error, "truncated has_refresh_frame_flags");
    }
    signal->has_refresh_frame_flags = has;
    if (!has) {
      signal->refresh_frame_flags = 0;
    } else {
      std::uint32_t value = 0;
      if (!reader.bits(n, &value)) {
        return fail(error, "truncated frame_to_refresh");
      }
      if (value >= config.num_ref_frames) {
        return fail(error, "frame_to_refresh " + u32(value) +
                               " must be < NumRefFrames");
      }
      signal->frame_to_refresh = value;
      signal->refresh_frame_flags = 1U << value;
    }
  } else {
    std::uint32_t value = 0;
    if (!reader.bits(config.num_ref_frames, &value)) {
      return fail(error, "truncated refresh_frame_flags");
    }
    signal->refresh_frame_flags = value;
  }
  return validate_refresh_frame_flags(frame_type, config, *signal,
                                      immediate_output_frame, error);
}

// ---------------------------------------------------------------------------
// frame_size_with_refs / found_ref
// ---------------------------------------------------------------------------
namespace {

bool validate_size_config(const SizeConfig& config, const RefList& refs,
                          const FrameRefState& state, std::string* error) {
  if (refs.num_total_refs > kRefsPerFrame) {
    return fail(error, "NumTotalRefs exceeds REFS_PER_FRAME");
  }
  if (config.num_total_refs != refs.num_total_refs) {
    return fail(error, "SizeConfig::num_total_refs disagrees with RefList");
  }
  if (config.frame_width_bits == 0 || config.frame_width_bits > 32U ||
      config.frame_height_bits == 0 || config.frame_height_bits > 32U) {
    return fail(error, "frame width/height bits must be in [1, 32]");
  }
  if (config.max_frame_width != state.max_frame_width ||
      config.max_frame_height != state.max_frame_height) {
    return fail(error,
                "SizeConfig frame maximum disagrees with FrameRefState");
  }
  return true;
}

std::uint32_t compute_mi(std::uint32_t size) {
  return 2U * ((size + 7U) >> 3U);
}

}  // namespace

bool validate_frame_size_against_refs(const FrameSizeResolved& size,
                                      const RefList& refs,
                                      const FrameRefState& state,
                                      std::string* error) {
  if (size.width == 0U || size.height == 0U) {
    return fail(error, "frame size must be non-zero");
  }
  if (size.width > state.max_frame_width ||
      size.height > state.max_frame_height) {
    return fail(error, "frame size " + u32(size.width) + "x" +
                           u32(size.height) +
                           " exceeds the sequence maximum " +
                           u32(state.max_frame_width) + "x" +
                           u32(state.max_frame_height));
  }
  for (std::uint32_t i = 0; i < refs.num_total_refs; ++i) {
    const std::uint32_t idx = refs.index[i];
    if (idx >= kNumRefFrames) {
      return fail(error, "ref_frame_idx[" + u32(i) + "] is out of range");
    }
    const RefSlot& slot = state.slot[idx];
    if (!slot.valid) {
      return fail(error, "ref_frame_idx[" + u32(i) + "] is not RefValid");
    }
    if (slot.width == 0U || slot.height == 0U) {
      return fail(error, "reference " + u32(i) + " has no frame size");
    }
    if (2U * size.width < slot.width || 2U * size.height < slot.height) {
      return fail(error,
                  "frame size is more than 2x smaller than reference " +
                      u32(i));
    }
    if (size.width > 16U * slot.width ||
        size.height > 16U * slot.height) {
      return fail(error,
                  "frame size is more than 16x larger than reference " +
                      u32(i));
    }
  }
  return true;
}

bool write_frame_size_with_refs(BitWriter& writer, const SizeConfig& config,
                                const RefList& refs,
                                const FrameRefState& state,
                                const FrameSizeSignal& signal,
                                FrameSizeResolved* out, std::string* error) {
  if (error != nullptr) error->clear();
  if (out == nullptr) return fail(error, "null frame-size output");
  if (!validate_size_config(config, refs, state, error)) return false;
  *out = FrameSizeResolved{};
  if (signal.width == 0U || signal.height == 0U) {
    return fail(error, "frame size must be non-zero");
  }

  std::uint32_t inherited_from = kNoRef;
  if (signal.prefer_found_ref) {
    for (std::uint32_t i = 0; i < refs.num_total_refs; ++i) {
      const std::uint32_t idx = refs.index[i];
      if (idx >= kNumRefFrames) {
        return fail(error, "ref_frame_idx[" + u32(i) + "] is out of range");
      }
      const RefSlot& slot = state.slot[idx];
      if (!slot.valid) {
        return fail(error, "ref_frame_idx[" + u32(i) + "] is not RefValid");
      }
      if (slot.order_hint == kRestrictedOrderHint) {
        return fail(error, "ref_frame_idx[" + u32(i) +
                               "] has RESTRICTED_OH and cannot supply a "
                               "frame size");
      }
      if (slot.width == signal.width && slot.height == signal.height) {
        inherited_from = i;
        break;
      }
    }
  }

  FrameSizeResolved resolved;
  resolved.found_ref.fill(false);
  resolved.inherited_from = kNoRef;
  if (inherited_from != kNoRef) {
    resolved.found_ref[inherited_from] = true;
    resolved.inherited_from = inherited_from;
    resolved.width = state.slot[refs.index[inherited_from]].width;
    resolved.height = state.slot[refs.index[inherited_from]].height;
    resolved.explicit_size = false;
  } else {
    resolved.explicit_size = true;
    resolved.width = signal.width;
    resolved.height = signal.height;
  }
  resolved.mi_cols = compute_mi(resolved.width);
  resolved.mi_rows = compute_mi(resolved.height);

  if (!validate_frame_size_against_refs(resolved, refs, state, error)) {
    return false;
  }
  if (resolved.explicit_size) {
    if (config.frame_width_bits < 32U &&
        (resolved.width - 1U) >= (1U << config.frame_width_bits)) {
      return fail(error, "frame_width_minus_1 does not fit frame_width_bits");
    }
    if (config.frame_height_bits < 32U &&
        (resolved.height - 1U) >= (1U << config.frame_height_bits)) {
      return fail(error, "frame_height_minus_1 does not fit frame_height_bits");
    }
  }

  const std::uint32_t n = refs.num_total_refs;
  for (std::uint32_t i = 0; i < n; ++i) {
    const bool found = resolved.found_ref[i];
    writer.put_bit(found);
    if (found) break;
  }
  if (resolved.explicit_size) {
    writer.put_bits(resolved.width - 1U, config.frame_width_bits);
    writer.put_bits(resolved.height - 1U, config.frame_height_bits);
  }
  *out = resolved;
  return true;
}

bool read_frame_size_with_refs(RefBitReader& reader, const SizeConfig& config,
                               const RefList& refs,
                               const FrameRefState& state,
                               FrameSizeResolved* out, std::string* error) {
  if (error != nullptr) error->clear();
  if (out == nullptr) return fail(error, "null frame-size output");
  if (!validate_size_config(config, refs, state, error)) return false;
  *out = FrameSizeResolved{};

  std::uint32_t inherited_from = kNoRef;
  for (std::uint32_t i = 0; i < refs.num_total_refs; ++i) {
    bool found = false;
    if (!reader.bit(&found)) {
      return fail(error, "truncated found_ref");
    }
    out->found_ref[i] = found;
    if (found) {
      inherited_from = i;
      break;
    }
  }

  if (inherited_from != kNoRef) {
    const std::uint32_t idx = refs.index[inherited_from];
    if (idx >= kNumRefFrames) {
      return fail(error, "found_ref selects an out-of-range slot");
    }
    const RefSlot& slot = state.slot[idx];
    if (!slot.valid) {
      return fail(error, "found_ref selects a slot that is not RefValid");
    }
    if (slot.order_hint == kRestrictedOrderHint) {
      return fail(error,
                  "found_ref selects a reference with RESTRICTED_OH");
    }
    out->inherited_from = inherited_from;
    out->width = slot.width;
    out->height = slot.height;
    out->explicit_size = false;
  } else {
    std::uint32_t width_minus_1 = 0;
    std::uint32_t height_minus_1 = 0;
    if (!reader.bits(config.frame_width_bits, &width_minus_1) ||
        !reader.bits(config.frame_height_bits, &height_minus_1)) {
      return fail(error, "truncated frame_size()");
    }
    out->explicit_size = true;
    out->width = width_minus_1 + 1U;
    out->height = height_minus_1 + 1U;
  }
  out->mi_cols = compute_mi(out->width);
  out->mi_rows = compute_mi(out->height);
  return validate_frame_size_against_refs(*out, refs, state, error);
}

// ---------------------------------------------------------------------------
// frame_header_copy / header_bit[]
// ---------------------------------------------------------------------------
bool HeaderBits::bit(std::size_t index) const {
  if (index >= num_bits) return false;
  const std::size_t byte = index >> 3U;
  if (byte >= bytes.size()) return false;
  const unsigned shift = 7U - static_cast<unsigned>(index & 7U);
  return ((bytes[byte] >> shift) & 1U) != 0U;
}

HeaderBits capture_bits(std::span<const std::uint8_t> writer_bytes,
                        std::size_t start_bit, std::size_t end_bit) {
  HeaderBits out;
  if (end_bit < start_bit || end_bit > writer_bytes.size() * 8U) return out;
  out.num_bits = end_bit - start_bit;
  out.bytes.assign((out.num_bits + 7U) / 8U, 0U);
  for (std::size_t i = 0; i < out.num_bits; ++i) {
    const std::size_t source = start_bit + i;
    const std::size_t src_byte = source >> 3U;
    const unsigned src_shift = 7U - static_cast<unsigned>(source & 7U);
    if (((writer_bytes[src_byte] >> src_shift) & 1U) == 0U) continue;
    const std::size_t dst_byte = i >> 3U;
    const unsigned dst_shift = 7U - static_cast<unsigned>(i & 7U);
    out.bytes[dst_byte] |= static_cast<std::uint8_t>(1U << dst_shift);
  }
  return out;
}

std::size_t write_frame_header_copy(BitWriter& writer, const HeaderBits& bits,
                                    std::string* error) {
  if (error != nullptr) error->clear();
  if (bits.num_bits == 0) {
    fail(error, "frame_header_copy needs NumFrameHeaderBits > 0");
    return 0;
  }
  if (bits.bytes.size() != (bits.num_bits + 7U) / 8U) {
    if (error != nullptr) *error = "frame_header_copy bit buffer is truncated";
    return 0;
  }
  for (std::size_t i = 0; i < bits.num_bits; ++i) {
    writer.put_bit(bits.bit(i));
  }
  return bits.num_bits;
}

bool read_frame_header_copy(RefBitReader& reader, const HeaderBits& expected,
                            std::string* error) {
  if (error != nullptr) error->clear();
  if (expected.num_bits == 0) {
    return fail(error, "frame_header_copy needs NumFrameHeaderBits > 0");
  }
  for (std::size_t i = 0; i < expected.num_bits; ++i) {
    bool actual = false;
    if (!reader.bit(&actual)) {
      return fail(error, "truncated frame_header_copy at bit " + u32(
                             static_cast<std::uint32_t>(i)));
    }
    if (actual != expected.bit(i)) {
      return fail(error, "frame_header_copy differs from the first header at "
                         "bit " +
                             u32(static_cast<std::uint32_t>(i)));
    }
  }
  return true;
}

void FrameHeaderStore::reset() { bits_ = HeaderBits{}; }

void FrameHeaderStore::record(std::span<const std::uint8_t> writer_bytes,
                              std::size_t start_bit, std::size_t end_bit) {
  bits_ = capture_bits(writer_bytes, start_bit, end_bit);
}

std::size_t FrameHeaderStore::write_copy(BitWriter& writer,
                                         std::string* error) const {
  if (!present()) {
    if (error != nullptr) *error = "no recorded frame header to copy";
    return 0;
  }
  return write_frame_header_copy(writer, bits_, error);
}

bool FrameHeaderStore::verify_copy(RefBitReader& reader,
                                   std::string* error) const {
  return read_frame_header_copy(reader, bits_, error);
}

bool write_nonfirst_tile_group_header(BitWriter& writer,
                                      const FrameHeaderStore& store,
                                      bool frame_header_present,
                                      std::string* error) {
  if (error != nullptr) error->clear();
  if (frame_header_present && !store.present()) {
    return fail(error,
                "frame_header_present_flag = 1 needs a recorded first header");
  }
  writer.put_bit(frame_header_present);
  if (!frame_header_present) return true;
  return store.write_copy(writer, error) != 0;
}

bool read_nonfirst_tile_group_header(RefBitReader& reader,
                                     const FrameHeaderStore& store,
                                     bool* frame_header_present,
                                     std::string* error) {
  if (error != nullptr) error->clear();
  if (frame_header_present == nullptr) {
    return fail(error, "null frame_header_present_flag output");
  }
  bool present = false;
  if (!reader.bit(&present)) {
    return fail(error, "truncated frame_header_present_flag");
  }
  *frame_header_present = present;
  if (!present) return true;
  return store.verify_copy(reader, error);
}

}  // namespace av2enc::frame::ref_fields
