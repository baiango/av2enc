// Standalone tests for the HRD/timing component.
//
// Coverage:
//   * descriptor round-trips (uvlc, rg(4)) and pinned bit goldens;
//   * timing_info / seq_decoder_model_info / sequence timing block bytes;
//   * BRT non-dependent and per-operating-point forms, full OBU framing;
//   * content interpretation OBU carrying timing_info;
//   * validation: OPS id/count correspondence, tick ranges, low-delay;
//   * clock derivation (DispCT/DecCT) and the decoding-schedule planner;
//   * two-pass FramePlan -> FrameBits adapter;
//   * sequence-header timing splice on the pinned M2 header;
//   * temporal-unit OBU ordering.
//
// No external dependencies and no decoder is required.

#include "encoder/rc/hrd/hrd.h"
#include "encoder/rc/hrd/hrd_buffer.h"

#include "bitstream/m0_headers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using av2enc::BitWriter;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* expression, const char* file, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::cerr << "FAIL " << file << ":" << line << ": " << expression << "\n";
  }
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

std::string hex(std::span<const std::uint8_t> bytes) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    if (i != 0) out.push_back(' ');
    out.push_back(digits[(bytes[i] >> 4U) & 0x0FU]);
    out.push_back(digits[bytes[i] & 0x0FU]);
  }
  return out;
}

bool bytes_equal(std::span<const std::uint8_t> got,
                 std::initializer_list<std::uint8_t> want) {
  if (got.size() != want.size()) return false;
  std::size_t i = 0;
  for (std::uint8_t value : want) {
    if (got[i++] != value) return false;
  }
  return true;
}

void check_bytes(const std::vector<std::uint8_t>& got,
                 std::initializer_list<std::uint8_t> want, const char* what) {
  ++g_checks;
  if (!bytes_equal(got, want)) {
    std::cerr << "FAIL golden " << what << ": got [" << hex(got)
              << "] want [";
    bool first = true;
    for (std::uint8_t value : want) {
      if (!first) std::cerr << " ";
      first = false;
      static const char* digits = "0123456789abcdef";
      std::cerr << digits[(value >> 4U) & 0x0FU] << digits[value & 0x0FU];
    }
    std::cerr << "]\n";
    ++g_failures;
  }
}

std::vector<std::uint8_t> take(BitWriter& writer) {
  return writer.take_bytes();
}

// Pinned AV2 v1.0.0 M2 sequence header OBU for a 64x64 stream, written by
// src/encoder/m2/m2_headers.cpp with decoder_model_info_present_flag = 0.
// 13 04 | 80 75 00 2a bf fe 00 28 00 6e 23 80 00 00 10 1b 80 82
constexpr std::uint8_t kM2SequenceHeader64[] = {
    0x13, 0x04, 0x80, 0x75, 0x00, 0x2a, 0xbf, 0xfe, 0x00, 0x28,
    0x00, 0x6e, 0x23, 0x80, 0x00, 0x00, 0x10, 0x1b, 0x80, 0x82};

// ---------------------------------------------------------------------------
// Descriptors and syntax goldens
// ---------------------------------------------------------------------------

void test_descriptors() {
  const std::uint32_t uvlc_values[] = {0, 1, 2, 3, 4, 5, 6, 7, 126, 127,
                                       128, 70000, 20000, 0xFFFFFFFEU};
  for (std::uint32_t value : uvlc_values) {
    BitWriter writer;
    // uvlc() is private to hrd.cpp; exercise it through timing_info with
    // equal_picture_interval = 1 and value = num_ticks_per_picture_minus_1.
    av2enc::hrd::TimingInfo timing;
    timing.num_units_in_display_tick = 1;
    timing.time_scale = 1;
    timing.equal_picture_interval = true;
    timing.num_ticks_per_picture_minus_1 = value;
    std::string error;
    CHECK(av2enc::hrd::write_timing_info(writer, timing, &error));
    av2enc::hrd::BitReader reader{
        std::span<const std::uint8_t>(writer.bytes())};
    av2enc::hrd::TimingInfo decoded;
    CHECK(av2enc::hrd::read_timing_info(reader, decoded, &error));
    CHECK(decoded.num_ticks_per_picture_minus_1 == value);
    CHECK(decoded.num_units_in_display_tick == 1);
    CHECK(decoded.time_scale == 1);
  }

  const std::uint32_t rg_values[] = {0, 1, 15, 16, 17, 255, 256, 510, 511};
  for (std::uint32_t value : rg_values) {
    av2enc::hrd::BufferRemovalTiming brt;
    brt.br_time = value;
    std::vector<std::uint8_t> bytes;
    std::string error;
    CHECK(av2enc::hrd::build_buffer_removal_timing_obu(brt, bytes, &error));
    av2enc::hrd::BufferRemovalTiming decoded;
    CHECK(av2enc::hrd::read_buffer_removal_timing_obu(bytes, decoded, &error));
    CHECK(!decoded.br_ops_dependent_flag);
    CHECK(decoded.br_time == value);
  }

  av2enc::hrd::BufferRemovalTiming too_large;
  too_large.br_time = 512;
  std::vector<std::uint8_t> bytes;
  std::string error;
  CHECK(!av2enc::hrd::build_buffer_removal_timing_obu(too_large, bytes,
                                                      &error));
  CHECK(!error.empty());
}

void test_timing_goldens() {
  av2enc::hrd::TimingInfo timing;
  timing.num_units_in_display_tick = 1000;
  timing.time_scale = 30000;
  timing.equal_picture_interval = true;
  timing.num_ticks_per_picture_minus_1 = 0;
  {
    BitWriter writer;
    CHECK(av2enc::hrd::write_timing_info(writer, timing));
    check_bytes(take(writer),
                {0x00, 0x00, 0x03, 0xe8, 0x00, 0x00, 0x75, 0x30, 0xc0},
                "timing_info 30fps");
  }
  timing.num_units_in_display_tick = 1001;
  timing.time_scale = 24000;
  {
    BitWriter writer;
    CHECK(av2enc::hrd::write_timing_info(writer, timing));
    check_bytes(take(writer),
                {0x00, 0x00, 0x03, 0xe9, 0x00, 0x00, 0x5d, 0xc0, 0xc0},
                "timing_info 24000/1001");
  }

  av2enc::hrd::SeqDecoderModelInfo info;
  info.decoder_buffer_delay = 70000;
  info.encoder_buffer_delay = 20000;
  info.low_delay_mode_flag = false;
  {
    BitWriter writer;
    CHECK(av2enc::hrd::write_seq_decoder_model_info(writer, info));
    check_bytes(take(writer),
                {0x00, 0x00, 0x88, 0xb8, 0x80, 0x01, 0x38, 0x84},
                "seq_decoder_model_info 70000/20000/0");
  }

  av2enc::hrd::SequenceTimingConfig config;
  config.seq_initial_display_delay_present_flag = true;
  config.seq_initial_display_delay_minus_1 = 2;
  config.decoder_model_info_present_flag = true;
  config.num_units_in_decoding_tick = 1000;
  config.seq_decoder_model_info_present_flag = true;
  config.decoder_model_info = info;
  {
    BitWriter writer;
    CHECK(av2enc::hrd::write_sequence_timing(writer, config));
    check_bytes(take(writer),
                {0x94, 0x00, 0x00, 0x0f, 0xa2, 0x00, 0x01,
                 0x11, 0x71, 0x00, 0x02, 0x71, 0x08},
                "sequence timing block");
  }

  av2enc::hrd::BufferRemovalTiming brt;
  brt.br_time = 0;
  {
    std::vector<std::uint8_t> obu;
    CHECK(av2enc::hrd::build_buffer_removal_timing_obu(brt, obu));
    check_bytes(obu, {0x02, 0x3c, 0x02}, "BRT non-dependent br_time=0");
  }
  brt.br_time = 511;
  {
    std::vector<std::uint8_t> obu;
    CHECK(av2enc::hrd::build_buffer_removal_timing_obu(brt, obu));
    check_bytes(obu, {0x06, 0x3c, 0x7f, 0xff, 0xff, 0xff, 0x7c},
                "BRT non-dependent br_time=511");
  }

  av2enc::hrd::BufferRemovalTiming dependent;
  dependent.br_ops_dependent_flag = true;
  dependent.br_ops_id = 3;
  dependent.op_times.resize(2);
  dependent.op_times[0].br_decoder_model_present_op_flag = true;
  dependent.op_times[0].br_time = 0;
  dependent.op_times[1].br_decoder_model_present_op_flag = true;
  dependent.op_times[1].br_time = 5;
  {
    std::vector<std::uint8_t> obu;
    CHECK(av2enc::hrd::build_buffer_removal_timing_obu(dependent, obu));
    check_bytes(obu, {0x04, 0x3c, 0x9a, 0x82, 0x58},
                "BRT dependent ops=3 cnt=2 times=0,5");
  }

  av2enc::hrd::ContentInterpretation ci;
  ci.ci_timing_info_present_flag = true;
  ci.timing = timing;  // 24000/1001, 1 tick
  {
    std::vector<std::uint8_t> obu;
    CHECK(av2enc::hrd::build_content_interpretation_obu(ci, obu));
    check_bytes(obu,
                {0x0b, 0x60, 0x04, 0x00, 0x00, 0x03, 0xe9, 0x00,
                 0x00, 0x5d, 0xc0, 0xd0},
                "CI OBU with timing_info");
  }
}

void test_round_trips() {
  std::string error;
  av2enc::hrd::TimingInfo timing;
  timing.num_units_in_display_tick = 1001;
  timing.time_scale = 30000;
  timing.equal_picture_interval = true;
  timing.num_ticks_per_picture_minus_1 = 3;
  {
    BitWriter writer;
    CHECK(av2enc::hrd::write_timing_info(writer, timing));
    av2enc::hrd::BitReader reader{
        std::span<const std::uint8_t>(writer.bytes())};
    av2enc::hrd::TimingInfo decoded;
    CHECK(av2enc::hrd::read_timing_info(reader, decoded, &error));
    CHECK(decoded.num_units_in_display_tick ==
          timing.num_units_in_display_tick);
    CHECK(decoded.time_scale == timing.time_scale);
    CHECK(decoded.equal_picture_interval);
    CHECK(decoded.num_ticks_per_picture_minus_1 ==
          timing.num_ticks_per_picture_minus_1);
  }

  av2enc::hrd::BufferRemovalTiming dependent;
  dependent.br_ops_dependent_flag = true;
  dependent.br_ops_id = 15;
  dependent.op_times.resize(7);
  for (std::size_t i = 0; i < dependent.op_times.size(); ++i) {
    dependent.op_times[i].br_decoder_model_present_op_flag = (i % 2U) == 0U;
    dependent.op_times[i].br_time =
        static_cast<std::uint32_t>(i * 70U + 1U);
  }
  std::vector<std::uint8_t> obu;
  CHECK(av2enc::hrd::build_buffer_removal_timing_obu(dependent, obu, &error));
  av2enc::hrd::BufferRemovalTiming decoded;
  CHECK(av2enc::hrd::read_buffer_removal_timing_obu(obu, decoded, &error));
  CHECK(decoded.br_ops_dependent_flag);
  CHECK(decoded.br_ops_id == 15);
  CHECK(decoded.op_times.size() == 7);
  for (std::size_t i = 0; i < decoded.op_times.size(); ++i) {
    CHECK(decoded.op_times[i].br_decoder_model_present_op_flag ==
          dependent.op_times[i].br_decoder_model_present_op_flag);
    if (dependent.op_times[i].br_decoder_model_present_op_flag) {
      CHECK(decoded.op_times[i].br_time == dependent.op_times[i].br_time);
    }
  }

  av2enc::hrd::ContentInterpretation ci;
  ci.ci_scan_type_idc = 2;
  ci.ci_color_description_present_flag = true;
  ci.ci_color_description_idc = 0;
  ci.ci_color_primaries = 9;
  ci.ci_transfer_characteristics = 16;
  ci.ci_matrix_coefficients = 9;
  ci.ci_full_range_flag = true;
  ci.ci_chroma_sample_position_present_flag = true;
  ci.ci_chroma_sample_position_top = 1;
  ci.ci_chroma_sample_position_bottom = 2;
  ci.ci_aspect_ratio_info_present_flag = true;
  ci.ci_aspect_ratio_idc = 255;
  ci.ci_sar_width = 4;
  ci.ci_sar_height = 3;
  ci.ci_timing_info_present_flag = true;
  ci.timing = timing;
  std::vector<std::uint8_t> ci_obu;
  CHECK(av2enc::hrd::build_content_interpretation_obu(ci, ci_obu, &error));
  av2enc::hrd::ContentInterpretation decoded_ci;
  CHECK(av2enc::hrd::read_content_interpretation_obu(ci_obu, decoded_ci,
                                                     &error));
  CHECK(decoded_ci.ci_scan_type_idc == ci.ci_scan_type_idc);
  CHECK(decoded_ci.ci_color_description_present_flag);
  CHECK(decoded_ci.ci_color_description_idc == 0);
  CHECK(decoded_ci.ci_color_primaries == 9);
  CHECK(decoded_ci.ci_transfer_characteristics == 16);
  CHECK(decoded_ci.ci_matrix_coefficients == 9);
  CHECK(decoded_ci.ci_full_range_flag);
  CHECK(decoded_ci.ci_chroma_sample_position_top == 1);
  CHECK(decoded_ci.ci_chroma_sample_position_bottom == 2);
  CHECK(decoded_ci.ci_aspect_ratio_idc == 255);
  CHECK(decoded_ci.ci_sar_width == 4);
  CHECK(decoded_ci.ci_sar_height == 3);
  CHECK(decoded_ci.ci_timing_info_present_flag);
  CHECK(decoded_ci.timing.num_ticks_per_picture_minus_1 == 3);

  // Truncated OBU and wrong-type OBU are rejected.
  std::vector<std::uint8_t> truncated(obu.begin(), obu.end() - 1);
  av2enc::hrd::BufferRemovalTiming ignored;
  CHECK(!av2enc::hrd::read_buffer_removal_timing_obu(truncated, ignored,
                                                     &error));
  CHECK(!av2enc::hrd::read_buffer_removal_timing_obu(ci_obu, ignored, &error));
}

void test_validation() {
  std::string error;
  av2enc::hrd::TimingInfo timing;
  timing.num_units_in_display_tick = 0;
  CHECK(!av2enc::hrd::timing_info_valid(timing, &error));
  timing.num_units_in_display_tick = 1;
  timing.time_scale = 0;
  CHECK(!av2enc::hrd::timing_info_valid(timing, &error));
  timing.time_scale = 1;

  av2enc::hrd::SequenceTimingConfig config;
  config.seq_initial_display_delay_minus_1 = 16;
  CHECK(!av2enc::hrd::sequence_timing_valid(config, &error));
  config.seq_initial_display_delay_minus_1 = 0;
  config.decoder_model_info_present_flag = false;
  config.seq_decoder_model_info_present_flag = true;
  CHECK(!av2enc::hrd::sequence_timing_valid(config, &error));
  config.seq_decoder_model_info_present_flag = false;
  config.decoder_model_info_present_flag = true;
  config.num_units_in_decoding_tick = 0;
  CHECK(!av2enc::hrd::sequence_timing_valid(config, &error));
  config.num_units_in_decoding_tick = 1;
  CHECK(av2enc::hrd::sequence_timing_valid(config, &error));

  av2enc::hrd::BufferRemovalTiming brt;
  brt.br_time = 512;
  CHECK(!av2enc::hrd::validate_brt_tick_ranges(brt, &error));
  brt.br_time = 511;
  CHECK(av2enc::hrd::validate_brt_tick_ranges(brt, &error));
  brt.br_ops_dependent_flag = true;
  brt.br_ops_id = 16;
  brt.op_times.resize(1);
  CHECK(!av2enc::hrd::buffer_removal_timing_valid(brt, &error));
  brt.br_ops_id = 0;
  brt.op_times.clear();
  CHECK(!av2enc::hrd::buffer_removal_timing_valid(brt, &error));
  brt.op_times.resize(8);
  CHECK(!av2enc::hrd::buffer_removal_timing_valid(brt, &error));

  av2enc::hrd::ContentInterpretation ci;
  ci.ci_scan_type_idc = 4;
  CHECK(!av2enc::hrd::content_interpretation_valid(ci, &error));
  ci.ci_scan_type_idc = 0;
  ci.ci_chroma_sample_position_present_flag = true;
  ci.ci_chroma_sample_position_top = 6;
  CHECK(!av2enc::hrd::content_interpretation_valid(ci, &error));
  ci.ci_chroma_sample_position_top = 5;
  ci.ci_aspect_ratio_info_present_flag = true;
  ci.ci_aspect_ratio_idc = 17;
  CHECK(!av2enc::hrd::content_interpretation_valid(ci, &error));
  ci.ci_aspect_ratio_idc = 16;
  CHECK(av2enc::hrd::content_interpretation_valid(ci, &error));
}

void test_ops_correspondence() {
  av2enc::hrd::BufferRemovalTiming brt;
  brt.br_ops_dependent_flag = true;
  brt.br_ops_id = 2;
  brt.op_times.resize(3);
  const av2enc::hrd::OperatingPointSetSummary ops[] = {
      {0, 2, 3},
      {0, 5, 1},
      {1, 2, 3},
  };
  std::string error;
  CHECK(av2enc::hrd::validate_brt_against_ops(
      brt, 0, std::span<const av2enc::hrd::OperatingPointSetSummary>(ops),
      &error));
  brt.op_times.resize(2);
  CHECK(!av2enc::hrd::validate_brt_against_ops(
      brt, 0, std::span<const av2enc::hrd::OperatingPointSetSummary>(ops),
      &error));
  brt.op_times.resize(3);
  brt.br_ops_id = 4;
  CHECK(!av2enc::hrd::validate_brt_against_ops(
      brt, 0, std::span<const av2enc::hrd::OperatingPointSetSummary>(ops),
      &error));
  brt.br_ops_id = 2;
  CHECK(!av2enc::hrd::validate_brt_against_ops(
      brt, 2, std::span<const av2enc::hrd::OperatingPointSetSummary>(ops),
      &error));

  const av2enc::hrd::OperatingPointSetSummary reset_ops[] = {{0, 2, 0}};
  brt.op_times.resize(1);
  CHECK(!av2enc::hrd::validate_brt_against_ops(
      brt, 0,
      std::span<const av2enc::hrd::OperatingPointSetSummary>(reset_ops),
      &error));

  av2enc::hrd::BufferRemovalTiming plain;
  plain.br_time = 7;
  CHECK(av2enc::hrd::validate_brt_against_ops(
      plain, 0, std::span<const av2enc::hrd::OperatingPointSetSummary>(ops),
      &error));
}

// ---------------------------------------------------------------------------
// Clock derivation
// ---------------------------------------------------------------------------

void test_clock() {
  std::string error;
  av2enc::hrd::TimingInfo timing;
  timing.num_units_in_display_tick = 1000;
  timing.time_scale = 30000;
  timing.equal_picture_interval = true;
  timing.num_ticks_per_picture_minus_1 = 0;
  av2enc::hrd::ClockDerivation clock;
  CHECK(av2enc::hrd::derive_clock(timing, 1000, clock, &error));
  CHECK(clock.decoding_tick_units == 90000000ULL);
  CHECK(clock.frame_interval_units == 90000000ULL);
  CHECK(clock.ticks_per_picture == 1U);
  CHECK(clock.exact_ticks_per_picture);

  timing.num_units_in_display_tick = 1001;
  timing.time_scale = 30000;
  CHECK(av2enc::hrd::derive_clock(timing, 1001, clock, &error));
  CHECK(clock.ticks_per_picture == 1U);
  CHECK(clock.exact_ticks_per_picture);

  timing.num_units_in_display_tick = 1000;
  timing.time_scale = 30000;
  CHECK(av2enc::hrd::derive_clock(timing, 900, clock, &error));
  CHECK(clock.decoding_tick_units == 81000000ULL);
  CHECK(clock.frame_interval_units == 90000000ULL);
  CHECK(clock.ticks_per_picture == 2U);
  CHECK(!clock.exact_ticks_per_picture);

  timing.num_units_in_display_tick = 0;
  CHECK(!av2enc::hrd::derive_clock(timing, 1000, clock, &error));
}

// ---------------------------------------------------------------------------
// Buffer planner
// ---------------------------------------------------------------------------

av2enc::hrd::HrdBufferConfig base_config() {
  av2enc::hrd::HrdBufferConfig config;
  config.timing.num_units_in_display_tick = 1000;
  config.timing.time_scale = 30000;
  config.timing.equal_picture_interval = true;
  config.timing.num_ticks_per_picture_minus_1 = 0;
  config.num_units_in_decoding_tick = 1000;  // DecCT = 1/30 s
  config.decoder_buffer_delay = 70000;       // 0.777... s
  config.encoder_buffer_delay = 20000;       // 0.222... s
  config.low_delay_mode_flag = false;
  config.bit_rate = 2000000;                 // 2 Mbit/s
  config.buffer_size_bits = 2000000;         // 1 s
  config.allow_delay_adjustment = true;
  return config;
}

std::vector<av2enc::hrd::FrameBits> make_frames(std::size_t count,
                                                std::uint64_t bits,
                                                std::size_t rap_every = 0) {
  std::vector<av2enc::hrd::FrameBits> frames(count);
  for (std::size_t i = 0; i < count; ++i) {
    frames[i].coded_bits = bits;
    frames[i].random_access_point =
        (i == 0U) || (rap_every != 0U && i % rap_every == 0U);
  }
  return frames;
}

void test_buffer_planner() {
  std::string error;
  {
    av2enc::hrd::HrdBufferConfig config = base_config();
    const std::vector<av2enc::hrd::FrameBits> frames =
        make_frames(30, 20000);
    av2enc::hrd::BufferTimingPlan plan;
    CHECK(av2enc::hrd::plan_buffer_removal(config, frames, plan, &error));
    CHECK(plan.conformant);
    CHECK(!plan.underflow_seen);
    CHECK(!plan.overflow_seen);
    CHECK(plan.frames.size() == 30);
    CHECK(plan.frames[0].br_time == 0U);
    for (std::size_t i = 1; i < plan.frames.size(); ++i) {
      CHECK(plan.frames[i].br_time == i);
      CHECK(plan.frames[i].scheduled_units >=
            plan.frames[i].last_bit_arrival_units);
      CHECK(plan.frames[i].br_time >= plan.frames[i - 1].br_time);
    }
    CHECK(plan.max_buffer_occupancy_bits <= config.buffer_size_bits);
    CHECK(av2enc::hrd::br_time_for_frame(plan, 5).br_time == 5U);
  }

  {
    // 100 kbit frames take 50 ms to transmit, longer than one DecCT (33 ms),
    // so the schedule must move later in whole DecCT ticks.
    av2enc::hrd::HrdBufferConfig config = base_config();
    const std::vector<av2enc::hrd::FrameBits> frames =
        make_frames(10, 100000);
    av2enc::hrd::BufferTimingPlan plan;
    CHECK(av2enc::hrd::plan_buffer_removal(config, frames, plan, &error));
    CHECK(!plan.underflow_seen);
    for (std::size_t i = 1; i < plan.frames.size(); ++i) {
      CHECK(plan.frames[i].scheduled_units >=
            plan.frames[i].last_bit_arrival_units);
      CHECK(plan.frames[i].br_time >= plan.frames[i - 1].br_time);
      CHECK(plan.frames[i].br_time >= 1U);
    }
  }

  {
    // Auto-adjustment must raise a too-short decoder delay for frame 0.
    av2enc::hrd::HrdBufferConfig config = base_config();
    config.decoder_buffer_delay = 10;
    config.encoder_buffer_delay = 10;
    config.allow_delay_adjustment = true;
    const std::vector<av2enc::hrd::FrameBits> frames =
        make_frames(5, 100000);
    av2enc::hrd::BufferTimingPlan plan;
    CHECK(av2enc::hrd::plan_buffer_removal(config, frames, plan, &error));
    CHECK(plan.delays_adjusted);
    CHECK(plan.decoder_buffer_delay > config.decoder_buffer_delay);
    CHECK(plan.decoder_buffer_delay + plan.encoder_buffer_delay >=
          config.decoder_buffer_delay + config.encoder_buffer_delay);
    CHECK(!plan.underflow_seen);
  }

  {
    // With adjustment disabled the same delay is rejected in strict mode.
    av2enc::hrd::HrdBufferConfig config = base_config();
    config.decoder_buffer_delay = 10;
    config.encoder_buffer_delay = 10;
    config.allow_delay_adjustment = false;
    const std::vector<av2enc::hrd::FrameBits> frames =
        make_frames(5, 100000);
    av2enc::hrd::BufferTimingPlan plan;
    CHECK(!av2enc::hrd::plan_buffer_removal(config, frames, plan, &error));
  }

  {
    // Low-delay mode permits the underflow and defers removal.
    av2enc::hrd::HrdBufferConfig config = base_config();
    config.decoder_buffer_delay = 10;
    config.encoder_buffer_delay = 10;
    config.allow_delay_adjustment = false;
    config.low_delay_mode_flag = true;
    const std::vector<av2enc::hrd::FrameBits> frames =
        make_frames(5, 100000);
    av2enc::hrd::BufferTimingPlan plan;
    const bool ok =
        av2enc::hrd::plan_buffer_removal(config, frames, plan, &error);
    CHECK(ok);
    if (!ok) return;
    CHECK(plan.low_delay_mode);
    CHECK(plan.underflow_seen);
    CHECK(!plan.frames.empty());
    CHECK(plan.frames[0].underflow);
    CHECK(plan.frames[0].actual_removal_units >
          plan.frames[0].scheduled_units);
  }

  {
    // Sustained overspend: 50 ms of data per 33 ms display interval. The
    // schedule eventually has to put more than one DecCT tick between frames.
    av2enc::hrd::HrdBufferConfig config = base_config();
    const std::vector<av2enc::hrd::FrameBits> frames =
        make_frames(60, 100000);
    av2enc::hrd::BufferTimingPlan plan;
    CHECK(av2enc::hrd::plan_buffer_removal(config, frames, plan, &error));
    CHECK(!plan.underflow_seen);
    bool saw_wide_gap = false;
    for (std::size_t i = 1; i < plan.frames.size(); ++i) {
      CHECK(plan.frames[i].scheduled_units >=
            plan.frames[i].last_bit_arrival_units);
      if (plan.frames[i].br_time > plan.frames[i - 1].br_time + 1U) {
        saw_wide_gap = true;
      }
    }
    CHECK(saw_wide_gap);
  }

  {
    // Buffer too small for the coded frames.
    av2enc::hrd::HrdBufferConfig config = base_config();
    config.buffer_size_bits = 1000;
    const std::vector<av2enc::hrd::FrameBits> frames =
        make_frames(3, 100000);
    av2enc::hrd::BufferTimingPlan plan;
    CHECK(!av2enc::hrd::plan_buffer_removal(config, frames, plan, &error));
  }

  {
    // RAP resets: br_time is relative to the previous RAP for the RAP itself
    // and to the current RAP for following frames.
    av2enc::hrd::HrdBufferConfig config = base_config();
    const std::vector<av2enc::hrd::FrameBits> frames =
        make_frames(8, 20000, 4);
    av2enc::hrd::BufferTimingPlan plan;
    CHECK(av2enc::hrd::plan_buffer_removal(config, frames, plan, &error));
    CHECK(plan.frames[4].random_access_point);
    CHECK(plan.frames[4].br_time == 4U);  // offset from RAP at frame 0
    CHECK(plan.frames[5].br_time == 1U);  // offset from RAP at frame 4
    CHECK(plan.frames[6].br_time == 2U);
  }

  {
    // Input validation: first frame must be a RAP, constant frame interval,
    // non-zero rate and buffer.
    av2enc::hrd::HrdBufferConfig config = base_config();
    std::vector<av2enc::hrd::FrameBits> frames = make_frames(3, 20000);
    frames[0].random_access_point = false;
    av2enc::hrd::BufferTimingPlan plan;
    CHECK(!av2enc::hrd::plan_buffer_removal(config, frames, plan, &error));

    frames = make_frames(3, 20000);
    config.timing.equal_picture_interval = false;
    CHECK(!av2enc::hrd::plan_buffer_removal(config, frames, plan, &error));
    config = base_config();
    config.bit_rate = 0;
    CHECK(!av2enc::hrd::plan_buffer_removal(config, frames, plan, &error));
    config = base_config();
    config.buffer_size_bits = 0;
    CHECK(!av2enc::hrd::plan_buffer_removal(config, frames, plan, &error));
  }

  {
    // rg(4) cannot express more than 511 ticks from a RAP.
    av2enc::hrd::HrdBufferConfig config = base_config();
    config.timing.num_units_in_display_tick = 1;
    config.timing.time_scale = 90000;
    config.num_units_in_decoding_tick = 1;  // DecCT = 1/90000 s
    const std::vector<av2enc::hrd::FrameBits> frames =
        make_frames(600, 1000);
    av2enc::hrd::BufferTimingPlan plan;
    CHECK(!av2enc::hrd::plan_buffer_removal(config, frames, plan, &error));
    CHECK(!error.empty());
  }
}

void test_ops_planner() {
  std::string error;
  av2enc::hrd::HrdBufferConfig config = base_config();
  std::vector<av2enc::hrd::OperatingPointTimeline> ops(2);
  ops[0].frames = make_frames(6, 20000);
  ops[1].frames = make_frames(6, 30000);
  av2enc::hrd::OpsBufferTimingPlan plan;
  CHECK(av2enc::hrd::plan_ops_buffer_removal(config, 3, ops, plan, &error));
  CHECK(plan.br_ops_id == 3);
  CHECK(plan.ops.size() == 2);
  CHECK(plan.per_frame.size() == 6);
  for (const av2enc::hrd::BufferRemovalTiming& brt : plan.per_frame) {
    CHECK(brt.br_ops_dependent_flag);
    CHECK(brt.br_ops_id == 3);
    CHECK(brt.op_times.size() == 2);
    CHECK(brt.op_times[0].br_decoder_model_present_op_flag);
    CHECK(brt.op_times[1].br_decoder_model_present_op_flag);
  }
  const av2enc::hrd::OperatingPointSetSummary summary[] = {{0, 3, 2}};
  CHECK(av2enc::hrd::validate_brt_against_ops(
      plan.per_frame[0], 0,
      std::span<const av2enc::hrd::OperatingPointSetSummary>(summary),
      &error));

  std::vector<av2enc::hrd::OperatingPointTimeline> misaligned(2);
  misaligned[0].frames = make_frames(6, 20000);
  misaligned[1].frames = make_frames(5, 30000);
  CHECK(!av2enc::hrd::plan_ops_buffer_removal(config, 0, misaligned, plan,
                                              &error));

  std::vector<av2enc::hrd::OperatingPointTimeline> too_many(8);
  for (auto& op : too_many) op.frames = make_frames(2, 1000);
  CHECK(!av2enc::hrd::plan_ops_buffer_removal(config, 0, too_many, plan,
                                              &error));
}

void test_twopass_adapter() {
  std::string error;
  std::vector<av2enc::rc::twopass::FramePlan> plans(3);
  plans[0].keyframe = true;
  plans[0].target_bits = 1000.5;
  plans[0].allocation_bits = 900.0;
  plans[0].estimated_bits = 800.0;
  plans[1].keyframe = false;
  plans[1].target_bits = 2000.4;
  plans[1].allocation_bits = 1900.0;
  plans[1].estimated_bits = 1800.0;
  plans[2].keyframe = false;
  plans[2].target_bits = 3000.6;
  plans[2].allocation_bits = 2900.0;
  plans[2].estimated_bits = 2800.0;

  std::vector<av2enc::hrd::FrameBits> bits;
  CHECK(av2enc::hrd::frames_from_twopass_plans(
      plans, {}, av2enc::hrd::TwopassBitSource::kTarget, bits, &error));
  CHECK(bits.size() == 3);
  CHECK(bits[0].coded_bits == 1001U);
  CHECK(bits[1].coded_bits == 2000U);
  CHECK(bits[2].coded_bits == 3001U);
  CHECK(bits[0].random_access_point);
  CHECK(!bits[1].random_access_point);

  CHECK(av2enc::hrd::frames_from_twopass_plans(
      plans, {}, av2enc::hrd::TwopassBitSource::kAllocation, bits, &error));
  CHECK(bits[0].coded_bits == 900U);
  CHECK(av2enc::hrd::frames_from_twopass_plans(
      plans, {}, av2enc::hrd::TwopassBitSource::kEstimated, bits, &error));
  CHECK(bits[2].coded_bits == 2800U);

  const std::int64_t actual[] = {100, 200, 300};
  CHECK(av2enc::hrd::frames_from_twopass_plans(
      plans, std::span<const std::int64_t>(actual),
      av2enc::hrd::TwopassBitSource::kActual, bits, &error));
  CHECK(bits[0].coded_bits == 100U && bits[2].coded_bits == 300U);

  CHECK(!av2enc::hrd::frames_from_twopass_plans(
      plans, {}, av2enc::hrd::TwopassBitSource::kActual, bits, &error));
  const std::int64_t negative[] = {-1, 2, 3};
  CHECK(!av2enc::hrd::frames_from_twopass_plans(
      plans, std::span<const std::int64_t>(negative),
      av2enc::hrd::TwopassBitSource::kActual, bits, &error));
  plans[1].target_bits = std::nan("");
  CHECK(!av2enc::hrd::frames_from_twopass_plans(
      plans, {}, av2enc::hrd::TwopassBitSource::kTarget, bits, &error));

  av2enc::rc::twopass::TwoPassConfig tp;
  tp.rc.vbr.target_bitrate_bps = 2000000.0;
  tp.rc.vbr.frame_rate_num = 30000;
  tp.rc.vbr.frame_rate_den = 1001;
  tp.vbv.enabled = true;
  tp.vbv.buffer_frames = 2.0;
  av2enc::rc::twopass::VbvState vbv;
  vbv.size_bits = 1000000.0;
  av2enc::hrd::HrdBufferConfig hrd;
  CHECK(av2enc::hrd::hrd_config_from_twopass(tp, vbv, 1001, hrd, &error));
  CHECK(hrd.timing.num_units_in_display_tick == 1001);
  CHECK(hrd.timing.time_scale == 30000);
  CHECK(hrd.timing.equal_picture_interval);
  CHECK(hrd.num_units_in_decoding_tick == 1001);
  CHECK(hrd.bit_rate == 2000000U);
  CHECK(hrd.buffer_size_bits == 1000000U);

  vbv.size_bits = 0.0;
  tp.vbv.buffer_size_bits = 500000.0;
  CHECK(av2enc::hrd::hrd_config_from_twopass(tp, vbv, 1001, hrd, &error));
  CHECK(hrd.buffer_size_bits == 500000U);

  tp.vbv.buffer_size_bits = 0.0;
  CHECK(av2enc::hrd::hrd_config_from_twopass(tp, vbv, 1001, hrd, &error));
  CHECK(hrd.buffer_size_bits == 133467U);

  CHECK(!av2enc::hrd::hrd_config_from_twopass(tp, vbv, 0, hrd, &error));
  tp.rc.vbr.frame_rate_num = 0;
  CHECK(!av2enc::hrd::hrd_config_from_twopass(tp, vbv, 1001, hrd, &error));
  tp.rc.vbr.frame_rate_num = 30000;
  tp.rc.vbr.target_bitrate_bps = 0.0;
  CHECK(!av2enc::hrd::hrd_config_from_twopass(tp, vbv, 1001, hrd, &error));
}

// ---------------------------------------------------------------------------
// Sequence header splice
// ---------------------------------------------------------------------------

void test_sequence_header_splice() {
  std::string error;
  const std::vector<std::uint8_t> baseline(
      kM2SequenceHeader64,
      kM2SequenceHeader64 + sizeof(kM2SequenceHeader64));
  av2enc::hrd::SequenceTimingConfig config;
  config.seq_initial_display_delay_present_flag = true;
  config.seq_initial_display_delay_minus_1 = 2;
  config.decoder_model_info_present_flag = true;
  config.num_units_in_decoding_tick = 1000;
  config.seq_decoder_model_info_present_flag = true;
  config.decoder_model_info.decoder_buffer_delay = 70000;
  config.decoder_model_info.encoder_buffer_delay = 20000;
  config.decoder_model_info.low_delay_mode_flag = false;
  std::vector<std::uint8_t> spliced;
  CHECK(av2enc::hrd::splice_sequence_timing(baseline, config, spliced, &error));
  CHECK(spliced.size() > baseline.size());
  check_bytes(spliced,
              {0x20, 0x04, 0x80, 0x75, 0x00, 0x2a, 0xbf, 0xfe,
               0x94, 0x00, 0x00, 0x0f, 0xa2, 0x00, 0x01, 0x11,
               0x71, 0x00, 0x02, 0x71, 0x08, 0x02, 0x80, 0x06,
               0xe2, 0x38, 0x00, 0x00, 0x01, 0x01, 0xb8, 0x08,
               0x20},
              "spliced 64x64 sequence header with decoder model info");

  av2enc::hrd::SequenceTimingConfig config_without_sdmi = config;
  config_without_sdmi.seq_decoder_model_info_present_flag = false;
  std::vector<std::uint8_t> spliced_without_sdmi;
  CHECK(av2enc::hrd::splice_sequence_timing(
      baseline, config_without_sdmi, spliced_without_sdmi, &error));
  check_bytes(spliced_without_sdmi,
              {0x18, 0x04, 0x80, 0x75, 0x00, 0x2a, 0xbf, 0xfe,
               0x94, 0x00, 0x00, 0x0f, 0xa0, 0x01, 0x40, 0x03,
               0x71, 0x1c, 0x00, 0x00, 0x00, 0x80, 0xdc, 0x04,
               0x10},
              "spliced 64x64 sequence header without seq_decoder_model_info");

  av2enc::hrd::SequenceTimingConfig decoded;
  CHECK(av2enc::hrd::read_sequence_header_timing(spliced, decoded, &error));
  CHECK(decoded.seq_initial_display_delay_present_flag);
  CHECK(decoded.seq_initial_display_delay_minus_1 == 2);
  CHECK(decoded.decoder_model_info_present_flag);
  CHECK(decoded.num_units_in_decoding_tick == 1000);
  CHECK(decoded.seq_decoder_model_info_present_flag);
  CHECK(decoded.decoder_model_info.decoder_buffer_delay == 70000);
  CHECK(decoded.decoder_model_info.encoder_buffer_delay == 20000);
  CHECK(!decoded.decoder_model_info.low_delay_mode_flag);

  av2enc::hrd::SequenceTimingConfig none;
  CHECK(av2enc::hrd::read_sequence_header_timing(baseline, none, &error));
  CHECK(!none.decoder_model_info_present_flag);

  // Splicing a header that already carries timing is rejected.
  std::vector<std::uint8_t> twice;
  CHECK(!av2enc::hrd::splice_sequence_timing(spliced, config, twice, &error));

  // Splitting the OBU list still shows one sequence header.
  std::vector<av2enc::hrd::ParsedObu> obus;
  CHECK(av2enc::hrd::parse_obus(spliced, obus, &error));
  CHECK(obus.size() == 1U);
  CHECK(obus[0].type == av2enc::hrd::ObuType::kSequenceHeader);
}

// ---------------------------------------------------------------------------
// OBU ordering
// ---------------------------------------------------------------------------

void test_obu_order() {
  std::string error;
  const std::vector<std::uint8_t> td =
      av2enc::m0::make_temporal_delimiter_obu();
  const std::vector<std::uint8_t> seq(kM2SequenceHeader64,
                                      kM2SequenceHeader64 +
                                          sizeof(kM2SequenceHeader64));
  av2enc::hrd::ContentInterpretation ci;
  ci.ci_timing_info_present_flag = true;
  ci.timing.num_units_in_display_tick = 1000;
  ci.timing.time_scale = 30000;
  ci.timing.equal_picture_interval = true;
  ci.timing.num_ticks_per_picture_minus_1 = 0;
  std::vector<std::uint8_t> ci_obu;
  CHECK(av2enc::hrd::build_content_interpretation_obu(ci, ci_obu, &error));
  const std::vector<std::uint8_t>& ci_ref = ci_obu;
  av2enc::hrd::BufferRemovalTiming brt;
  brt.br_time = 0;
  std::vector<std::uint8_t> brt_obu;
  CHECK(av2enc::hrd::build_buffer_removal_timing_obu(brt, brt_obu, &error));
  const std::vector<std::uint8_t>& brt_ref = brt_obu;

  av2enc::m0::ObuHeader frame_header;
  frame_header.type = av2enc::m0::ObuType::kRegularTileGroup;
  const std::vector<std::uint8_t> frame = av2enc::m0::make_obu(
      frame_header, std::span<const std::uint8_t>{});

  std::vector<std::uint8_t> unit;
  for (const auto* piece : {&td, &seq, &ci_ref, &brt_ref, &frame}) {
    unit.insert(unit.end(), piece->begin(), piece->end());
  }
  std::vector<av2enc::hrd::ParsedObu> obus;
  CHECK(av2enc::hrd::parse_obus(unit, obus, &error));
  CHECK(av2enc::hrd::validate_temporal_unit_order(obus, &error));

  // An OPS OBU is allowed before the sequence header (per-operating-point
  // BRT streams need it). The validator does not parse its payload.
  av2enc::m0::ObuHeader ops_header;
  ops_header.type = static_cast<av2enc::m0::ObuType>(18);
  const std::vector<std::uint8_t> ops = av2enc::m0::make_obu(
      ops_header, std::span<const std::uint8_t>{});
  std::vector<std::uint8_t> with_ops;
  for (const auto* piece : {&td, &ops, &seq, &ci_ref, &brt_ref, &frame}) {
    with_ops.insert(with_ops.end(), piece->begin(), piece->end());
  }
  CHECK(av2enc::hrd::parse_obus(with_ops, obus, &error));
  CHECK(av2enc::hrd::validate_temporal_unit_order(obus, &error));

  // BRT after the frame OBU.
  std::vector<std::uint8_t> bad;
  for (const auto* piece : {&td, &seq, &ci_ref, &frame, &brt_ref}) {
    bad.insert(bad.end(), piece->begin(), piece->end());
  }
  CHECK(av2enc::hrd::parse_obus(bad, obus, &error));
  CHECK(!av2enc::hrd::validate_temporal_unit_order(obus, &error));

  // Two BRT OBUs before the frame.
  bad.clear();
  for (const auto* piece : {&td, &brt_ref, &brt_ref, &frame}) {
    bad.insert(bad.end(), piece->begin(), piece->end());
  }
  CHECK(av2enc::hrd::parse_obus(bad, obus, &error));
  CHECK(!av2enc::hrd::validate_temporal_unit_order(obus, &error));

  // Missing temporal delimiter.
  bad.clear();
  for (const auto* piece : {&seq, &frame}) {
    bad.insert(bad.end(), piece->begin(), piece->end());
  }
  CHECK(av2enc::hrd::parse_obus(bad, obus, &error));
  CHECK(!av2enc::hrd::validate_temporal_unit_order(obus, &error));

  // CI before the sequence header.
  bad.clear();
  for (const auto* piece : {&td, &ci_ref, &seq, &frame}) {
    bad.insert(bad.end(), piece->begin(), piece->end());
  }
  CHECK(av2enc::hrd::parse_obus(bad, obus, &error));
  CHECK(!av2enc::hrd::validate_temporal_unit_order(obus, &error));
}

}  // namespace

int main() {
  test_descriptors();
  test_timing_goldens();
  test_round_trips();
  test_validation();
  test_ops_correspondence();
  test_clock();
  test_buffer_planner();
  test_ops_planner();
  test_twopass_adapter();
  test_sequence_header_splice();
  test_obu_order();
  std::cout << "hrd_test: " << g_checks << " checks, " << g_failures
            << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
