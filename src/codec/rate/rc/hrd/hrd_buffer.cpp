#include "codec/rate/rc/hrd/hrd_buffer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::hrd {
namespace {

__extension__ using u128 = unsigned __int128;

void set_error(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
}

constexpr std::uint64_t kTimebase = 90000U;

bool to_u64(u128 value, std::uint64_t& out, std::string* error,
            const char* what) {
  if (value > static_cast<u128>(std::numeric_limits<std::uint64_t>::max())) {
    set_error(error, std::string(what) + " exceeds 64-bit model range");
    return false;
  }
  out = static_cast<std::uint64_t>(value);
  return true;
}

u128 ceil_div(u128 numerator, u128 denominator) {
  return (numerator + denominator - 1U) / denominator;
}

u128 bits_to_units(std::uint64_t bits, std::uint64_t bit_rate,
                   std::uint32_t time_scale) {
  // ceil(bits * 90000 * time_scale / bit_rate)
  const u128 numerator =
      static_cast<u128>(bits) * kTimebase * static_cast<u128>(time_scale);
  return ceil_div(numerator, static_cast<u128>(bit_rate));
}

u128 delay_to_units(std::uint64_t delay_1_90000, std::uint32_t time_scale) {
  return static_cast<u128>(delay_1_90000) *
         static_cast<u128>(time_scale);
}

u128 capacity_units(std::uint64_t buffer_size_bits, std::uint64_t bit_rate,
                    std::uint32_t time_scale) {
  // floor(buffer_size_bits * 90000 * time_scale / bit_rate)
  const u128 numerator = static_cast<u128>(buffer_size_bits) * kTimebase *
                         static_cast<u128>(time_scale);
  return numerator / static_cast<u128>(bit_rate);
}

bool derive_units(const TimingInfo& timing, std::uint32_t nutdt,
                  u128& decoding_units, u128& interval_units,
                  std::uint64_t& ticks_per_picture, bool& exact,
                  std::string* error) {
  if (!timing_info_valid(timing, error)) return false;
  if (nutdt == 0U) {
    set_error(error, "num_units_in_decoding_tick shall be greater than 0");
    return false;
  }
  decoding_units =
      static_cast<u128>(nutdt) * static_cast<u128>(kTimebase);
  interval_units = static_cast<u128>(timing.ticks_per_picture()) *
                   static_cast<u128>(timing.num_units_in_display_tick) *
                   static_cast<u128>(kTimebase);
  if (interval_units == 0U) {
    set_error(error, "display interval is zero");
    return false;
  }
  const u128 ticks = ceil_div(interval_units, decoding_units);
  if (ticks > static_cast<u128>(std::numeric_limits<std::uint64_t>::max())) {
    set_error(error, "ticks per picture exceeds 64-bit range");
    return false;
  }
  ticks_per_picture = static_cast<std::uint64_t>(ticks);
  exact = (interval_units % decoding_units) == 0U;
  return true;
}

}  // namespace

bool derive_clock(const TimingInfo& timing,
                  std::uint32_t num_units_in_decoding_tick,
                  ClockDerivation& clock, std::string* error) {
  u128 decoding_units = 0;
  u128 interval_units = 0;
  std::uint64_t ticks = 0;
  bool exact = false;
  if (!derive_units(timing, num_units_in_decoding_tick, decoding_units,
                    interval_units, ticks, exact, error)) {
    return false;
  }
  ClockDerivation out;
  if (!to_u64(decoding_units, out.decoding_tick_units, error,
              "DecCT")) {
    return false;
  }
  if (!to_u64(interval_units, out.frame_interval_units, error,
              "frame interval")) {
    return false;
  }
  out.ticks_per_picture = ticks;
  out.exact_ticks_per_picture = exact;
  clock = out;
  return true;
}

bool hrd_config_from_twopass(const rc::twopass::TwoPassConfig& config,
                             const rc::twopass::VbvState& vbv,
                             std::uint32_t num_units_in_decoding_tick,
                             HrdBufferConfig& out, std::string* error) {
  out = HrdBufferConfig{};
  if (num_units_in_decoding_tick == 0U) {
    set_error(error, "num_units_in_decoding_tick shall be greater than 0");
    return false;
  }
  const std::int32_t frame_rate_num = config.rc.vbr.frame_rate_num;
  const std::int32_t frame_rate_den = config.rc.vbr.frame_rate_den;
  if (frame_rate_num <= 0 || frame_rate_den <= 0) {
    set_error(error,
              "two-pass frame_rate_num/frame_rate_den must both be positive");
    return false;
  }
  out.timing.num_units_in_display_tick =
      static_cast<std::uint32_t>(frame_rate_den);
  out.timing.time_scale = static_cast<std::uint32_t>(frame_rate_num);
  out.timing.equal_picture_interval = true;
  out.timing.num_ticks_per_picture_minus_1 = 0;
  out.num_units_in_decoding_tick = num_units_in_decoding_tick;

  double rate = config.vbv.max_rate_bps > 0.0
                    ? config.vbv.max_rate_bps
                    : config.rc.vbr.target_bitrate_bps;
  if (!std::isfinite(rate) || rate <= 0.0 || rate > 1.0e15) {
    set_error(error, "two-pass bit rate is not finite and positive");
    return false;
  }
  out.bit_rate = static_cast<std::uint64_t>(std::llround(rate));

  const double frame_duration =
      static_cast<double>(frame_rate_den) / static_cast<double>(frame_rate_num);
  const double average_frame_bits = rate * frame_duration;
  double buffer_bits = 0.0;
  if (vbv.size_bits > 0.0) {
    buffer_bits = vbv.size_bits;
  } else if (config.vbv.buffer_size_bits > 0.0) {
    buffer_bits = config.vbv.buffer_size_bits;
  } else {
    const double frames =
        config.vbv.buffer_frames > 0.0 ? config.vbv.buffer_frames : 2.0;
    buffer_bits = frames * average_frame_bits;
  }
  if (!std::isfinite(buffer_bits) || buffer_bits <= 0.0 ||
      buffer_bits > 9.0e18) {
    set_error(error, "two-pass buffer size is not finite and positive");
    return false;
  }
  out.buffer_size_bits =
      static_cast<std::uint64_t>(std::llround(buffer_bits));
  out.decoder_buffer_delay = 70000;
  out.encoder_buffer_delay = 20000;
  out.low_delay_mode_flag = false;
  out.allow_delay_adjustment = true;
  return true;
}

bool plan_buffer_removal(const HrdBufferConfig& config,
                         std::span<const FrameBits> frames,
                         BufferTimingPlan& plan, std::string* error) {
  plan = BufferTimingPlan{};
  if (frames.empty()) {
    set_error(error, "buffer plan needs at least one frame");
    return false;
  }
  if (!frames[0].random_access_point) {
    set_error(error, "the first frame must be a random access point");
    return false;
  }
  if (!timing_info_valid(config.timing, error)) return false;
  if (!config.timing.equal_picture_interval) {
    set_error(error,
              "buffer plan requires equal_picture_interval = 1 "
              "(a constant frame interval)");
    return false;
  }
  if (config.num_units_in_decoding_tick == 0U) {
    set_error(error, "num_units_in_decoding_tick shall be greater than 0");
    return false;
  }
  if (config.bit_rate == 0U) {
    set_error(error, "bit_rate shall be greater than 0");
    return false;
  }
  if (config.buffer_size_bits == 0U) {
    set_error(error, "buffer_size_bits shall be greater than 0");
    return false;
  }
  if (config.decoder_buffer_delay > kMaxUvlc ||
      config.encoder_buffer_delay > kMaxUvlc) {
    set_error(error, "buffer delays exceed uvlc() range");
    return false;
  }
  if (!seq_decoder_model_info_valid(
          SeqDecoderModelInfo{config.decoder_buffer_delay,
                              config.encoder_buffer_delay,
                              config.low_delay_mode_flag},
          error)) {
    return false;
  }

  u128 decoding_units = 0;
  u128 interval_units = 0;
  std::uint64_t ticks_per_picture = 0;
  bool exact = false;
  if (!derive_units(config.timing, config.num_units_in_decoding_tick,
                    decoding_units, interval_units, ticks_per_picture, exact,
                    error)) {
    return false;
  }

  const std::uint32_t time_scale = config.timing.time_scale;
  const u128 capacity = capacity_units(config.buffer_size_bits,
                                       config.bit_rate, time_scale);

  std::vector<u128> transmission(frames.size());
  u128 max_transmission = 0;
  for (std::size_t i = 0; i < frames.size(); ++i) {
    transmission[i] = bits_to_units(frames[i].coded_bits, config.bit_rate,
                                    time_scale);
    max_transmission = std::max(max_transmission, transmission[i]);
  }

  std::uint64_t dbd_delay = config.decoder_buffer_delay;
  std::uint64_t ebd_delay = config.encoder_buffer_delay;
  if (config.allow_delay_adjustment) {
    // Quantise the required window to whole 1/90000 s delay units.
    const u128 max_transmission_delay = ceil_div(max_transmission, time_scale);
    const u128 first_delay = ceil_div(transmission[0], time_scale);
    if (max_transmission_delay > static_cast<u128>(kMaxUvlc) ||
        first_delay > static_cast<u128>(kMaxUvlc)) {
      set_error(error,
                "a frame transmission interval exceeds the uvlc() delay "
                "range; raise bit_rate or lower the frame size");
      return false;
    }
    dbd_delay = std::max<std::uint64_t>(
        dbd_delay, static_cast<std::uint64_t>(first_delay));
    if (dbd_delay == 0U) dbd_delay = 1U;
    std::uint64_t total_delay =
        std::max<std::uint64_t>(dbd_delay + ebd_delay, dbd_delay);
    total_delay = std::max<std::uint64_t>(
        total_delay, static_cast<std::uint64_t>(max_transmission_delay));
    if (total_delay < dbd_delay) total_delay = dbd_delay;
    ebd_delay = total_delay - dbd_delay;
    if (dbd_delay > kMaxUvlc || ebd_delay > kMaxUvlc) {
      set_error(error, "adjusted buffer delays exceed uvlc() range");
      return false;
    }
    const u128 total_units =
        delay_to_units(dbd_delay + ebd_delay, time_scale);
    if (total_units > capacity) {
      set_error(error,
                "buffer cannot hold the largest frame at the configured "
                "bit_rate (T * BitRate > BufferSize)");
      return false;
    }
  } else {
    if (dbd_delay == 0U) {
      set_error(error,
                "decoder_buffer_delay shall not be 0 in decoding schedule mode");
      return false;
    }
    const u128 dbd_units = delay_to_units(dbd_delay, time_scale);
    const u128 total_units =
        delay_to_units(dbd_delay + ebd_delay, time_scale);
    if (!config.low_delay_mode_flag && dbd_units < transmission[0]) {
      set_error(error,
                "decoder_buffer_delay is too short for frame 0 at this "
                "bit_rate");
      return false;
    }
    if (!config.low_delay_mode_flag && total_units < max_transmission) {
      set_error(error,
                "decoder_buffer_delay + encoder_buffer_delay is too short "
                "for the largest frame at this bit_rate");
      return false;
    }
    if (total_units > capacity) {
      set_error(error,
                "T * BitRate exceeds BufferSize; overflow is unavoidable");
      return false;
    }
  }

  const u128 dbd_units = delay_to_units(dbd_delay, time_scale);
  const u128 ebd_units = delay_to_units(ebd_delay, time_scale);
  const u128 window_units = dbd_units + ebd_units;
  const u128 one_tick = decoding_units;

  plan.low_delay_mode = config.low_delay_mode_flag;
  plan.num_units_in_decoding_tick = config.num_units_in_decoding_tick;
  plan.decoder_buffer_delay = static_cast<std::uint32_t>(dbd_delay);
  plan.encoder_buffer_delay = static_cast<std::uint32_t>(ebd_delay);
  plan.delays_adjusted =
      (dbd_delay != config.decoder_buffer_delay) ||
      (ebd_delay != config.encoder_buffer_delay);
  plan.buffer_size_bits = config.buffer_size_bits;
  if (!derive_clock(config.timing, config.num_units_in_decoding_tick,
                    plan.clock, error)) {
    return false;
  }
  plan.frames.resize(frames.size());

  u128 previous_scheduled = 0;
  u128 previous_last_arrival = 0;
  u128 previous_actual_removal = 0;
  u128 previous_rap_scheduled = 0;
  u128 current_rap_scheduled = 0;
  u128 scheduled = 0;

  for (std::size_t i = 0; i < frames.size(); ++i) {
    std::uint32_t br_time = 0;
    if (i == 0U) {
      scheduled = dbd_units;
      br_time = 0;
    } else {
      u128 raw = previous_scheduled + interval_units;
      const u128 min_after_actual = previous_actual_removal + one_tick;
      if (min_after_actual > raw) raw = min_after_actual;
      const u128 min_after_arrival = previous_last_arrival + transmission[i];
      if (min_after_arrival > raw) raw = min_after_arrival;
      if (frames[i].random_access_point) {
        // §E.7.2: DecoderBufferDelay <= ceil(TimeDelta[i]) at a RAP.
        const u128 min_for_dbd = previous_last_arrival + dbd_units;
        if (min_for_dbd > raw) raw = min_for_dbd;
      }
      const u128 base =
          frames[i].random_access_point ? previous_rap_scheduled
                                        : current_rap_scheduled;
      const u128 delta = raw > base ? raw - base : 0U;
      u128 ticks = ceil_div(delta, decoding_units);
      if (delta != 0U && ticks == 0U) ticks = 1U;
      if (ticks > static_cast<u128>(kMaxBrTime)) {
        set_error(error,
                  "frame " + std::to_string(i) + " needs br_time " +
                      std::to_string(static_cast<std::uint64_t>(ticks)) +
                      ", which does not fit in rg(4) (max 511). Increase "
                      "num_units_in_decoding_tick or add a random access "
                      "point.");
        return false;
      }
      br_time = static_cast<std::uint32_t>(ticks);
      scheduled = base + ticks * decoding_units;
    }

    u128 first_arrival =
        scheduled > window_units ? scheduled - window_units : 0U;
    if (first_arrival < previous_last_arrival) {
      first_arrival = previous_last_arrival;
    }
    const u128 last_arrival = first_arrival + transmission[i];
    const bool underflow = scheduled < last_arrival;
    u128 actual_removal = scheduled;
    if (underflow) {
      if (!config.low_delay_mode_flag) {
        set_error(error,
                  "frame " + std::to_string(i) +
                      " underflows the smoothing buffer in strict mode "
                      "(ScheduledRemoval < LastBitArrival)");
        return false;
      }
      actual_removal = ceil_div(last_arrival, decoding_units) * decoding_units;
    }
    if (i > 0U && actual_removal < previous_actual_removal + one_tick) {
      set_error(error,
                "frame " + std::to_string(i) +
                    " cannot start decoding at least one DecCT tick after the "
                    "previous frame");
      return false;
    }

    RemovalEntry entry;
    entry.random_access_point = frames[i].random_access_point;
    entry.br_time = br_time;
    entry.scheduled_units = 0;
    entry.first_bit_arrival_units = 0;
    entry.last_bit_arrival_units = 0;
    entry.actual_removal_units = 0;
    if (!to_u64(scheduled, entry.scheduled_units, error, "scheduled removal")) {
      return false;
    }
    if (!to_u64(first_arrival, entry.first_bit_arrival_units, error,
                "first bit arrival")) {
      return false;
    }
    if (!to_u64(last_arrival, entry.last_bit_arrival_units, error,
                "last bit arrival")) {
      return false;
    }
    if (!to_u64(actual_removal, entry.actual_removal_units, error,
                "actual removal")) {
      return false;
    }
    entry.underflow = underflow;
    plan.frames[i] = entry;
    plan.underflow_seen = plan.underflow_seen || underflow;

    previous_scheduled = scheduled;
    previous_last_arrival = last_arrival;
    previous_actual_removal = actual_removal;
    if (frames[i].random_access_point) {
      previous_rap_scheduled = scheduled;
      current_rap_scheduled = scheduled;
    }
  }

  // Overflow simulation in the same timebase. Arrival intervals are
  // sequential (FirstBitArrival[i] >= LastBitArrival[i-1]), so at most one
  // frame is arriving at a time; removals may overlap the next arrival.
  struct Event {
    std::uint64_t time_units = 0;
    long double time = 0.0L;
    int order = 0;  // 0 = arrival end, 1 = arrival start/instant, 2 = removal
    std::size_t frame = 0;
  };
  std::vector<Event> events;
  events.reserve(frames.size() * 3U);
  for (std::size_t i = 0; i < plan.frames.size(); ++i) {
    events.push_back(Event{plan.frames[i].first_bit_arrival_units,
                           static_cast<long double>(
                               plan.frames[i].first_bit_arrival_units),
                           1, i});
    events.push_back(Event{plan.frames[i].last_bit_arrival_units,
                           static_cast<long double>(
                               plan.frames[i].last_bit_arrival_units),
                           0, i});
    events.push_back(Event{plan.frames[i].actual_removal_units,
                           static_cast<long double>(
                               plan.frames[i].actual_removal_units),
                           2, i});
  }
  std::stable_sort(events.begin(), events.end(),
                   [](const Event& a, const Event& b) {
                     if (a.time_units != b.time_units) {
                       return a.time_units < b.time_units;
                     }
                     return a.order < b.order;
                   });
  const long double units_per_second =
      static_cast<long double>(kTimebase) *
      static_cast<long double>(time_scale);
  const long double rate_per_unit =
      static_cast<long double>(config.bit_rate) / units_per_second;
  long double occupancy = 0.0L;
  long double peak = 0.0L;
  long double active_start = 0.0L;
  std::size_t active_frame = frames.size();
  for (std::size_t e = 0; e < events.size();) {
    const std::uint64_t now_units = events[e].time_units;
    const long double now = events[e].time;
    if (active_frame < frames.size() && now > active_start) {
      occupancy += rate_per_unit * (now - active_start);
      active_start = now;
    }
    if (occupancy < 0.0L) occupancy = 0.0L;
    peak = std::max(peak, occupancy);

    // Arrival ends, then starts/instant arrivals, then removals at this time.
    std::size_t end = e;
    while (end < events.size() && events[end].time_units == now_units &&
           events[end].order == 0) {
      if (events[end].frame == active_frame) active_frame = frames.size();
      ++end;
    }
    while (end < events.size() && events[end].time_units == now_units &&
           events[end].order == 1) {
      const std::size_t frame = events[end].frame;
      if (plan.frames[frame].last_bit_arrival_units ==
          plan.frames[frame].first_bit_arrival_units) {
        occupancy += static_cast<long double>(frames[frame].coded_bits);
      } else {
        active_frame = frame;
        active_start = now;
      }
      ++end;
    }
    peak = std::max(peak, occupancy);
    while (end < events.size() && events[end].time_units == now_units) {
      // order 2: removal
      occupancy -= static_cast<long double>(frames[events[end].frame].coded_bits);
      if (occupancy < 0.0L) occupancy = 0.0L;
      peak = std::max(peak, occupancy);
      ++end;
    }
    e = end;
  }
  const long double peak_ceil = std::ceil(peak);
  if (peak_ceil >
      static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
    set_error(error, "peak buffer occupancy exceeds 64-bit range");
    return false;
  }
  plan.max_buffer_occupancy_bits =
      static_cast<std::uint64_t>(peak_ceil);
  if (plan.max_buffer_occupancy_bits > config.buffer_size_bits) {
    plan.overflow_seen = true;
    set_error(error,
              "modelled smoothing-buffer occupancy " +
                  std::to_string(plan.max_buffer_occupancy_bits) +
                  " bits exceeds BufferSize " +
                  std::to_string(config.buffer_size_bits) + " bits");
    return false;
  }
  plan.conformant = true;
  return true;
}

BufferRemovalTiming br_time_for_frame(const BufferTimingPlan& plan,
                                      std::size_t frame_index) {
  BufferRemovalTiming brt;
  brt.br_ops_dependent_flag = false;
  if (frame_index < plan.frames.size()) {
    brt.br_time = plan.frames[frame_index].br_time;
  }
  return brt;
}

bool plan_ops_buffer_removal(const HrdBufferConfig& config,
                             std::uint8_t br_ops_id,
                             std::span<const OperatingPointTimeline> ops,
                             OpsBufferTimingPlan& plan, std::string* error) {
  plan = OpsBufferTimingPlan{};
  if (ops.empty() || ops.size() > kMaxOpsCnt) {
    set_error(error, "an OPS must have 1..7 operating points");
    return false;
  }
  if (br_ops_id > kMaxOpsId) {
    set_error(error, "br_ops_id is f(4)");
    return false;
  }
  const std::size_t frame_count = ops[0].frames.size();
  if (frame_count == 0U) {
    set_error(error, "operating point 0 has no frames");
    return false;
  }
  for (std::size_t i = 0; i < ops.size(); ++i) {
    if (ops[i].frames.size() != frame_count) {
      set_error(error,
                "operating point " + std::to_string(i) +
                    " has a different frame count; per-operating-point BRT "
                    "times must align with the coded frame units");
      return false;
    }
  }

  HrdBufferConfig shared = config;
  if (shared.allow_delay_adjustment) {
    // Pick one sequence-level (decoder, encoder) delay pair that is large
    // enough for every operating point, then re-plan with it fixed.
    std::uint64_t dbd = 0;
    std::uint64_t total = 0;
    for (const OperatingPointTimeline& op : ops) {
      BufferTimingPlan candidate;
      if (!plan_buffer_removal(shared, op.frames, candidate, error)) {
        return false;
      }
      dbd = std::max<std::uint64_t>(dbd, candidate.decoder_buffer_delay);
      total = std::max<std::uint64_t>(
          total, static_cast<std::uint64_t>(candidate.decoder_buffer_delay) +
                     candidate.encoder_buffer_delay);
    }
    if (dbd > kMaxUvlc || total - dbd > kMaxUvlc) {
      set_error(error, "common buffer delays exceed uvlc() range");
      return false;
    }
    shared.decoder_buffer_delay = static_cast<std::uint32_t>(dbd);
    shared.encoder_buffer_delay =
        static_cast<std::uint32_t>(total - dbd);
    shared.allow_delay_adjustment = false;
  }

  plan.br_ops_id = br_ops_id;
  plan.ops.resize(ops.size());
  for (std::size_t i = 0; i < ops.size(); ++i) {
    if (!plan_buffer_removal(shared, ops[i].frames, plan.ops[i], error)) {
      return false;
    }
  }
  plan.decoder_buffer_delay = plan.ops[0].decoder_buffer_delay;
  plan.encoder_buffer_delay = plan.ops[0].encoder_buffer_delay;
  plan.num_units_in_decoding_tick = plan.ops[0].num_units_in_decoding_tick;
  for (const BufferTimingPlan& op : plan.ops) {
    if (op.decoder_buffer_delay != plan.decoder_buffer_delay ||
        op.encoder_buffer_delay != plan.encoder_buffer_delay) {
      set_error(error, "operating points resolved different buffer delays");
      return false;
    }
    plan.underflow_seen = plan.underflow_seen || op.underflow_seen;
    plan.overflow_seen = plan.overflow_seen || op.overflow_seen;
  }

  plan.per_frame.resize(frame_count);
  for (std::size_t frame = 0; frame < frame_count; ++frame) {
    BufferRemovalTiming brt;
    brt.br_ops_dependent_flag = true;
    brt.br_ops_id = br_ops_id;
    brt.op_times.resize(ops.size());
    for (std::size_t op = 0; op < ops.size(); ++op) {
      brt.op_times[op].br_decoder_model_present_op_flag = true;
      brt.op_times[op].br_time = plan.ops[op].frames[frame].br_time;
    }
    if (!buffer_removal_timing_valid(brt, error)) return false;
    plan.per_frame[frame] = brt;
  }
  return true;
}

bool frames_from_twopass_plans(std::span<const rc::twopass::FramePlan> plans,
                               std::span<const std::int64_t> actual_bits,
                               TwopassBitSource source,
                               std::vector<FrameBits>& out,
                               std::string* error) {
  out.clear();
  if (plans.empty()) {
    set_error(error, "no two-pass frame plans supplied");
    return false;
  }
  if (source == TwopassBitSource::kActual &&
      actual_bits.size() != plans.size()) {
    set_error(error, "actual_bits must have one entry per frame plan");
    return false;
  }
  if (!actual_bits.empty() && actual_bits.size() != plans.size()) {
    set_error(error, "actual_bits must be empty or match the plan count");
    return false;
  }
  out.resize(plans.size());
  for (std::size_t i = 0; i < plans.size(); ++i) {
    double bits = 0.0;
    switch (source) {
      case TwopassBitSource::kTarget: bits = plans[i].target_bits; break;
      case TwopassBitSource::kAllocation: bits = plans[i].allocation_bits; break;
      case TwopassBitSource::kEstimated: bits = plans[i].estimated_bits; break;
      case TwopassBitSource::kActual:
        if (actual_bits[i] < 0) {
          set_error(error, "actual_bits contains a negative value");
          return false;
        }
        bits = static_cast<double>(actual_bits[i]);
        break;
    }
    if (!std::isfinite(bits) || bits < 0.0) {
      set_error(error, "two-pass bit estimate is not finite and non-negative");
      return false;
    }
    if (bits > 9.0e18) {
      set_error(error, "two-pass bit estimate exceeds 64-bit range");
      return false;
    }
    out[i].coded_bits = static_cast<std::uint64_t>(std::llround(bits));
    out[i].random_access_point = plans[i].keyframe;
  }
  return true;
}

}  // namespace av2enc::hrd
