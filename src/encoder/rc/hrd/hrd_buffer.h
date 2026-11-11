#pragma once

// Decoder-model buffer planner for the HRD/timing component.
//
// This is the bridge between the existing two-pass/VBV plan
// (src/encoder/rc/twopass/) and the §5.12 buffer_removal_timing_obu():
//
//   FramePlan / actual payload bits
//        -> FrameBits (coded bits + random-access-point flag)
//        -> plan_buffer_removal()  (Annex E.5.4 decoding schedule)
//        -> BufferRemovalTiming (br_time / br_time_op)
//        -> build_buffer_removal_timing_obu()
//
// The planner uses the spec's clock derivation exactly:
//
//   DecCT = num_units_in_decoding_tick / time_scale                (§6.4.13)
//   ScheduledRemovalTiming[0] = DecoderBufferDelay / 90000          (§E.5.4)
//   ScheduledRemovalTiming[i] = ScheduledRemovalTiming[PrevRap]
//                               + BufferRemovalTime[i] * DecCT       (§E.5.4)
//   FirstBitArrival[i] = max(LastBitArrival[i-1], LatestArrivalTime[i])
//   LatestArrivalTime[i] = ScheduledRemoval[i]
//                          - (EncoderBufferDelay + DecoderBufferDelay)/90000
//   LastBitArrival[i] = FirstBitArrival[i] + CodedBits[i] / BitRate
//   strict-mode underflow: ScheduledRemoval[i] < LastBitArrival[i]   (§E.7.4)
//
// Time is simulated in integer units of 1/(90000 * time_scale) seconds, so
// DecCT is exactly num_units_in_decoding_tick * 90000 units and a delay in
// 1/90000 seconds is exactly delay * time_scale units. All rounding is
// upward (conservative for arrival times).
//
// Only Annex E.5.4 decoding-schedule mode is modelled. Presentation times,
// frame-buffer reuse and TimeToDecode (E.5.6/E.5.7, E.6) are not modelled;
// the planner checks the removal/arrival constraints and reports the rest.

#include "encoder/rc/hrd/hrd.h"
#include "encoder/rc/twopass/twopass.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::hrd {

// ---------------------------------------------------------------------------
// Clock derivation
// ---------------------------------------------------------------------------

struct ClockDerivation {
  // 1/(90000 * time_scale) seconds per unit.
  std::uint64_t decoding_tick_units = 0;   // DecCT
  std::uint64_t frame_interval_units = 0;  // DispCT * ticks_per_picture
  // ceil(frame_interval_units / decoding_tick_units)
  std::uint64_t ticks_per_picture = 0;
  bool exact_ticks_per_picture = false;
};

bool derive_clock(const TimingInfo& timing,
                  std::uint32_t num_units_in_decoding_tick,
                  ClockDerivation& clock, std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

struct FrameBits {
  // Coded payload bits for the frame's decodable frame group (DFG). This is
  // the coded frame payload, not container or IVF bytes.
  std::uint64_t coded_bits = 0;
  bool random_access_point = false;
};

struct HrdBufferConfig {
  TimingInfo timing{};
  std::uint32_t num_units_in_decoding_tick = 1;
  std::uint32_t decoder_buffer_delay = 70000;  // 1/90000 s
  std::uint32_t encoder_buffer_delay = 20000;  // 1/90000 s
  bool low_delay_mode_flag = false;
  std::uint64_t bit_rate = 0;          // bits per second
  std::uint64_t buffer_size_bits = 0;  // level MaxBufferSize * profile factor
  // When true the planner may raise the delays so that frame 0 and every
  // frame's transmission interval fit, and reports the values to signal.
  // When false the configured delays are used unchanged and the plan fails
  // rather than emitting an underflowing br_time.
  bool allow_delay_adjustment = true;
};

// Derives the HRD clock, bit rate and buffer size from the existing two-pass
// configuration and VBV state:
//   * frame interval from rc.vbr.frame_rate_num / frame_rate_den
//     (num_units_in_display_tick = denominator, time_scale = numerator),
//   * bit rate from vbv.max_rate_bps, or rc.vbr.target_bitrate_bps when the
//     VBV rate is unset,
//   * buffer size from vbv.size_bits, otherwise vbv.buffer_size_bits,
//     otherwise vbv.buffer_frames * the nominal frame budget.
// The caller supplies num_units_in_decoding_tick (the DecCT granularity) and
// can still override any field before planning.
bool hrd_config_from_twopass(const rc::twopass::TwoPassConfig& config,
                             const rc::twopass::VbvState& vbv,
                             std::uint32_t num_units_in_decoding_tick,
                             HrdBufferConfig& out,
                             std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

struct RemovalEntry {
  bool random_access_point = false;
  // br_time / br_time_op, in DecCT ticks relative to the applicable random
  // access point (the previous RAP for a RAP frame, the current RAP
  // otherwise). Always <= 511 because it is coded with rg(4).
  std::uint32_t br_time = 0;
  // Absolute model times, in 1/(90000 * time_scale) seconds.
  std::uint64_t scheduled_units = 0;
  std::uint64_t first_bit_arrival_units = 0;
  std::uint64_t last_bit_arrival_units = 0;
  // In low-delay mode an underflow defers removal to the first decode clock
  // tick after the DFG is complete (§E.5.4). Equals scheduled_units in
  // strict mode.
  std::uint64_t actual_removal_units = 0;
  bool underflow = false;
};

struct BufferTimingPlan {
  bool conformant = false;
  bool low_delay_mode = false;
  std::uint32_t num_units_in_decoding_tick = 0;
  // The delays to signal in seq_decoder_model_info(). These may differ from
  // the config when allow_delay_adjustment raised them.
  std::uint32_t decoder_buffer_delay = 0;
  std::uint32_t encoder_buffer_delay = 0;
  bool delays_adjusted = false;
  ClockDerivation clock{};
  std::vector<RemovalEntry> frames;
  std::uint64_t buffer_size_bits = 0;
  std::uint64_t max_buffer_occupancy_bits = 0;
  bool underflow_seen = false;
  bool overflow_seen = false;
};

// Builds a removal schedule that satisfies the strict-mode constraint
// ScheduledRemoval[i] >= LastBitArrival[i] whenever low_delay_mode_flag is
// false. frames[0] must be a random access point. The schedule follows the
// nominal display interval where possible and pushes a frame later (in whole
// DecCT ticks) only when the arrival model requires it.
bool plan_buffer_removal(const HrdBufferConfig& config,
                         std::span<const FrameBits> frames,
                         BufferTimingPlan& plan,
                         std::string* error = nullptr);

// Converts one entry of a plan into the §5.12 non-operating-point-dependent
// form (br_ops_dependent_flag = 0).
BufferRemovalTiming br_time_for_frame(const BufferTimingPlan& plan,
                                      std::size_t frame_index);

// ---------------------------------------------------------------------------
// Per-operating-point BRT
// ---------------------------------------------------------------------------

// One operating point's frame timeline. `frames` is aligned with every other
// operating point passed to plan_ops_buffer_removal(): all operating points
// cover the same coded frames and only their coded sizes differ.
struct OperatingPointTimeline {
  std::vector<FrameBits> frames;
};

struct OpsBufferTimingPlan {
  std::uint8_t br_ops_id = 0;
  std::uint32_t decoder_buffer_delay = 0;
  std::uint32_t encoder_buffer_delay = 0;
  std::uint32_t num_units_in_decoding_tick = 0;
  std::vector<BufferTimingPlan> ops;
  // One BRT per coded frame, with br_ops_dependent_flag = 1 and one
  // br_time_op entry per operating point.
  std::vector<BufferRemovalTiming> per_frame;
  bool underflow_seen = false;
  bool overflow_seen = false;
};

// `ops[i]` describes operating point i; the vector order is the BRT
// br_time_op index. Every timeline must have the same number of frames and
// frame 0 must be a RAP. All operating points must select the same
// sequence-level delays (the planner errors if adjustment diverges).
bool plan_ops_buffer_removal(const HrdBufferConfig& config,
                             std::uint8_t br_ops_id,
                             std::span<const OperatingPointTimeline> ops,
                             OpsBufferTimingPlan& plan,
                             std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Two-pass adapter
// ---------------------------------------------------------------------------

enum class TwopassBitSource {
  kTarget,      // FramePlan::target_bits (effective budget after the VBV cap)
  kAllocation,  // FramePlan::allocation_bits (pre-VBV water-filling)
  kEstimated,   // FramePlan::estimated_bits (model estimate)
  kActual,      // the measured actual_bits vector
};

// Converts twopass frame plans into FrameBits. `actual_bits` may be empty for
// the non-kActual sources, otherwise it must have one non-negative entry per
// plan. Keyframes become random access points. Values are rounded to the
// nearest bit with deterministic round-half-up.
bool frames_from_twopass_plans(std::span<const rc::twopass::FramePlan> plans,
                               std::span<const std::int64_t> actual_bits,
                               TwopassBitSource source,
                               std::vector<FrameBits>& out,
                               std::string* error = nullptr);

}  // namespace av2enc::hrd
