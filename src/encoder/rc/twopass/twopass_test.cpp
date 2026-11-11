// Deterministic synthetic tests for the two-pass / VBV rate-control policy.
//
// Build and run with ./build_and_test.sh. No external dependencies.

#include "twopass.h"

#include "../rc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace av2enc::rc;
using namespace av2enc::rc::twopass;

int g_failures = 0;
int g_checks = 0;

void check_impl(bool condition, const char* expression, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::cerr << "line " << line << ": CHECK failed: " << expression << '\n';
  }
}

#define CHECK(expression) check_impl((expression), #expression, __LINE__)

bool near(double a, double b, double epsilon = 1e-6) {
  return std::fabs(a - b) <= epsilon;
}

FrameComplexity complexity(int width,
                           int height,
                           double mse,
                           double intra_fraction = 0.5) {
  FrameComplexity c;
  c.width = width;
  c.height = height;
  c.residual_mse = mse;
  c.residual_mad = std::sqrt(std::max(mse, 0.0));
  c.intra_fraction = intra_fraction;
  return c;
}

FrameStats flat_stats(int width, int height, double mse, bool keyframe = false) {
  FrameStats stats;
  stats.complexity = complexity(width, height, mse);
  stats.keyframe = keyframe;
  return stats;
}

TwoPassConfig vbr_config(double bitrate, bool with_vbv = false) {
  TwoPassConfig config;
  config.rc.mode = RateControlMode::kVbr;
  config.rc.vbr.target_bitrate_bps = bitrate;
  config.rc.vbr.frame_rate_num = 30;
  config.rc.vbr.frame_rate_den = 1;
  config.vbv.enabled = with_vbv;
  return config;
}

TwoPassConfig crf_config(double crf) {
  TwoPassConfig config;
  config.rc.mode = RateControlMode::kCrf;
  config.rc.crf.crf = crf;
  config.rc.vbr.target_bitrate_bps = 100000.0;
  config.rc.vbr.frame_rate_num = 30;
  config.rc.vbr.frame_rate_den = 1;
  config.vbv.enabled = false;
  return config;
}

bool same_tile(const TilePlan& a, const TilePlan& b) {
  if (a.sb_cols != b.sb_cols || a.sb_rows != b.sb_rows ||
      a.sb_col_start != b.sb_col_start || a.sb_row_start != b.sb_row_start ||
      a.target_qindex != b.target_qindex) {
    return false;
  }
  if (a.sb_target_qindex != b.sb_target_qindex ||
      a.sb_achieved_qindex != b.sb_achieved_qindex ||
      a.sb_segment_id != b.sb_segment_id) {
    return false;
  }
  if (a.delta_q.steps.size() != b.delta_q.steps.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.delta_q.steps.size(); ++i) {
    const DeltaQPlanStep& x = a.delta_q.steps[i];
    const DeltaQPlanStep& y = b.delta_q.steps[i];
    if (x.sb_x != y.sb_x || x.sb_y != y.sb_y ||
        x.reduced_delta != y.reduced_delta ||
        x.qindex_before != y.qindex_before ||
        x.qindex_after != y.qindex_after) {
      return false;
    }
  }
  return a.delta_q.max_abs_reduced_delta == b.delta_q.max_abs_reduced_delta &&
         a.delta_q.final_qindex == b.delta_q.final_qindex;
}

bool same_plan(const FramePlan& a, const FramePlan& b) {
  if (a.frame_index != b.frame_index || a.keyframe != b.keyframe ||
      a.base_q_idx != b.base_q_idx ||
      a.coeff_cdf_qctx != b.coeff_cdf_qctx ||
      a.continuous_qp != b.continuous_qp || a.lambda_sse != b.lambda_sse ||
      a.allocation_bits != b.allocation_bits ||
      a.target_bits != b.target_bits || a.estimated_bits != b.estimated_bits ||
      a.rate_feedback != b.rate_feedback ||
      a.delta_q.present != b.delta_q.present ||
      a.delta_q.res != b.delta_q.res ||
      a.segmentation.enabled != b.segmentation.enabled ||
      a.enable_ext_seg != b.enable_ext_seg ||
      a.vbv_fill_before != b.vbv_fill_before ||
      a.vbv_fill_after_estimate != b.vbv_fill_after_estimate ||
      a.vbv_underflow_prevented != b.vbv_underflow_prevented ||
      a.vbv_overflow_risk != b.vbv_overflow_risk ||
      a.target_reachable != b.target_reachable ||
      a.clamped_by_slew != b.clamped_by_slew ||
      a.clamped_by_range != b.clamped_by_range ||
      a.spatial_clamped != b.spatial_clamped ||
      a.crf_quality_target != b.crf_quality_target) {
    return false;
  }
  if (a.tiles.size() != b.tiles.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.tiles.size(); ++i) {
    if (!same_tile(a.tiles[i], b.tiles[i])) {
      return false;
    }
  }
  return a.segmentation_map.mi_cols == b.segmentation_map.mi_cols &&
         a.segmentation_map.mi_rows == b.segmentation_map.mi_rows &&
         a.segmentation_map.segment_id == b.segmentation_map.segment_id;
}

// ---------------------------------------------------------------------------
// VBV model
// ---------------------------------------------------------------------------

void test_vbv_model() {
  VbvConfig config;
  config.buffer_frames = 2.0;
  config.initial_fill = 0.5;
  config.underflow_margin = 0.02;
  config.overflow_margin = 0.02;
  VbvState state;
  std::string error;
  CHECK(make_vbv_state(config, 1000.0, 1.0 / 30.0, state, &error));
  CHECK(state.enabled);
  CHECK(near(state.size_bits, 2000.0));
  CHECK(near(state.fill_bits, 1000.0));
  CHECK(near(state.drain_bits_per_frame, 1000.0));
  CHECK(near(state.underflow_margin_bits, 40.0));
  CHECK(near(vbv_max_bits(state), 1960.0));
  CHECK(near(vbv_min_bits(state), 40.0));

  bool underflow = false;
  bool overflow = false;
  CHECK(vbv_apply(state, 500.0, underflow, overflow));
  CHECK(!underflow && !overflow);
  CHECK(near(state.fill_bits, 1500.0));

  CHECK(vbv_apply(state, 3000.0, underflow, overflow));
  CHECK(underflow);
  CHECK(!overflow);
  CHECK(state.underflow_events == 1);
  CHECK(near(state.fill_bits, 40.0, 1e-9));

  state.fill_bits = 1900.0;
  CHECK(vbv_apply(state, 0.0, underflow, overflow));
  CHECK(overflow);
  CHECK(state.overflow_events == 1);
  CHECK(near(state.fill_bits, 1960.0, 1e-9));

  CHECK(!vbv_apply(state, -1.0, underflow, overflow));
  CHECK(!vbv_apply(state, std::numeric_limits<double>::quiet_NaN(),
                   underflow, overflow));

  VbvConfig disabled;
  disabled.enabled = false;
  CHECK(make_vbv_state(disabled, 1000.0, 1.0 / 30.0, state, &error));
  CHECK(!state.enabled);
  CHECK(std::isinf(vbv_max_bits(state)));
  CHECK(vbv_apply(state, 1.0e9, underflow, overflow));
  CHECK(!underflow && !overflow);

  VbvConfig invalid = config;
  invalid.initial_fill = 1.5;
  CHECK(!make_vbv_state(invalid, 1000.0, 1.0 / 30.0, state, &error));
  invalid = config;
  invalid.underflow_margin = 0.6;
  CHECK(!make_vbv_state(invalid, 1000.0, 1.0 / 30.0, state, &error));
  CHECK(!make_vbv_state(config, 0.0, 1.0 / 30.0, state, &error));
  CHECK(!make_vbv_state(config, 1000.0, 0.0, state, &error));

  VbrParams vbr;
  vbr.frame_rate_num = 60;
  vbr.frame_rate_den = 1;
  CHECK(near(frame_duration_seconds(vbr), 1.0 / 60.0));
  CHECK(near(nominal_frame_bits(vbr), 500000.0 / 60.0));
  vbr.frame_rate_num = 0;
  CHECK(near(frame_duration_seconds(vbr), 0.0));
  CHECK(near(nominal_frame_bits(vbr), 0.0));
}

// ---------------------------------------------------------------------------
// Allocation and deterministic reallocation
// ---------------------------------------------------------------------------

void test_allocation() {
  constexpr int kWidth = 640;
  constexpr int kHeight = 360;
  std::vector<FrameStats> stats;
  const double mse_values[] = {1.0, 4.0, 9.0, 25.0, 100.0, 400.0};
  for (int i = 0; i < 6; ++i) {
    stats.push_back(flat_stats(kWidth, kHeight, mse_values[i]));
  }
  TwoPassConfig config = vbr_config(1000000.0);
  AllocationResult result;
  std::string error;
  CHECK(allocate_frames(stats, config, 60000.0, VbvState{}, result, &error));
  CHECK(result.frames.size() == 6);
  CHECK(near(result.allocated_bits, 60000.0, 1e-6));
  CHECK(near(result.total_bits, 60000.0));
  CHECK(result.dropped_bits == 0.0);
  for (std::size_t i = 1; i < result.frames.size(); ++i) {
    CHECK(result.frames[i].weight >= result.frames[i - 1].weight);
    CHECK(result.frames[i].target_bits >= result.frames[i - 1].target_bits);
  }
  CHECK(result.frames.front().target_bits < result.frames.back().target_bits);

  // All-flat input splits evenly.
  std::vector<FrameStats> flat(4, flat_stats(kWidth, kHeight, 0.0));
  CHECK(allocate_frames(flat, config, 4000.0, VbvState{}, result, &error));
  for (const FrameAllocation& frame : result.frames) {
    CHECK(near(frame.target_bits, 1000.0, 1e-6));
  }

  // Per-frame clamps from min/max ratios.
  std::vector<FrameStats> skewed;
  skewed.push_back(flat_stats(kWidth, kHeight, 0.0001));
  skewed.push_back(flat_stats(kWidth, kHeight, 1000000.0));
  config.min_frame_bits_ratio = 0.2;
  config.max_frame_bits_ratio = 8.0;
  CHECK(allocate_frames(skewed, config, 10000.0, VbvState{}, result, &error));
  CHECK(result.frames[0].target_bits >= 1000.0 - 1e-6);
  CHECK(result.frames[1].target_bits <= 40000.0 + 1e-6);
  CHECK(near(result.frames[0].target_bits + result.frames[1].target_bits,
             10000.0, 1e-6));

  // VBV caps a frame that would spend more than the bucket can supply, then
  // redistributes the freed budget to later frames. The trajectory must stay
  // at or above the underflow margin.
  TwoPassConfig vbv_config = vbr_config(30000.0, true);
  vbv_config.vbv.buffer_size_bits = 3000.0;
  vbv_config.vbv.initial_fill = 0.5;
  vbv_config.vbv.underflow_margin = 0.02;
  vbv_config.max_frame_bits_ratio = 8.0;
  std::vector<FrameStats> burst;
  burst.push_back(flat_stats(kWidth, kHeight, 1.0e6));
  burst.push_back(flat_stats(kWidth, kHeight, 4.0));
  burst.push_back(flat_stats(kWidth, kHeight, 4.0));
  burst.push_back(flat_stats(kWidth, kHeight, 4.0));
  VbvState start;
  CHECK(make_vbv_state(vbv_config.vbv, 1000.0, 1.0 / 30.0, start, &error));
  CHECK(allocate_frames(burst, vbv_config, 4000.0, start, result, &error));
  CHECK(result.frames[0].vbv_capped);
  CHECK(result.underflow_prevented);
  CHECK(result.reallocations >= 1);
  CHECK(near(result.allocated_bits, 4000.0, 1e-6));
  CHECK(result.frames[0].target_bits <= 2440.0 + 1e-6);  // 1500 + 1000 - 60
  double fill = start.fill_bits;
  for (const FrameAllocation& frame : result.frames) {
    fill += start.drain_bits_per_frame - frame.target_bits;
    CHECK(fill >= start.underflow_margin_bits - 1e-6);
    CHECK(near(fill, frame.vbv_fill_after, 1e-6));
  }

  // Two calls are bit-identical.
  AllocationResult again;
  CHECK(allocate_frames(burst, vbv_config, 4000.0, start, again, &error));
  CHECK(result.frames.size() == again.frames.size());
  for (std::size_t i = 0; i < result.frames.size(); ++i) {
    CHECK(result.frames[i].target_bits == again.frames[i].target_bits);
    CHECK(result.frames[i].vbv_fill_after == again.frames[i].vbv_fill_after);
    CHECK(result.frames[i].vbv_capped == again.frames[i].vbv_capped);
  }
  CHECK(result.allocated_bits == again.allocated_bits);
  CHECK(result.dropped_bits == again.dropped_bits);

  // Infeasible: even the minimum per-frame allocation underflows.
  TwoPassConfig tiny = vbr_config(30000.0, true);
  tiny.vbv.buffer_size_bits = 500.0;
  tiny.vbv.initial_fill = 0.1;
  tiny.vbv.underflow_margin = 0.02;
  tiny.vbv.max_rate_bps = 1000.0;
  tiny.min_frame_bits_ratio = 0.9;
  tiny.max_frame_bits_ratio = 1.0;
  tiny.max_qindex = 255;
  VbvState tiny_state;
  CHECK(make_vbv_state(tiny.vbv, 1000.0, 1.0 / 30.0, tiny_state, &error));
  CHECK(allocate_frames(flat, tiny, 4000.0, tiny_state, result, &error));
  CHECK(!result.underflow_feasible);

  // Invalid inputs.
  CHECK(!allocate_frames(std::span<const FrameStats>(), config, 1000.0,
                         VbvState{}, result, &error));
  CHECK(!allocate_frames(flat, config, -1.0, VbvState{}, result, &error));
  config.min_qindex = 0;
  CHECK(!two_pass_config_valid(config, &error));
}

// ---------------------------------------------------------------------------
// Lookahead window and controller sequencing
// ---------------------------------------------------------------------------

void test_lookahead_and_sequencing() {
  std::vector<FrameStats> stats;
  for (int i = 0; i < 8; ++i) {
    stats.push_back(flat_stats(320, 240, 10.0 + 5.0 * i, i == 0));
  }
  TwoPassConfig config = vbr_config(500000.0);
  config.lookahead_frames = 3;
  TwoPassController controller;
  std::string error;
  CHECK(controller.configure(config, &error));

  // Cannot begin before first-pass data.
  FramePlan plan;
  CHECK(!controller.begin_frame(plan, &error));

  // Streaming window: only three frames are planned.
  CHECK(controller.set_lookahead(stats, &error));
  CHECK(controller.lookahead_size() == 8);
  CHECK(controller.allocation().frames.size() == 3);

  double actual = 0.0;
  for (int i = 0; i < 3; ++i) {
    CHECK(controller.begin_frame(plan, &error));
    CHECK(plan.frame_index == i);
    actual = plan.estimated_bits;
    CHECK(controller.end_frame(static_cast<std::int64_t>(actual), &error));
  }
  CHECK(!controller.begin_frame(plan, &error));

  // A fresh window continues at the next frame.
  CHECK(controller.set_lookahead(
      std::span<const FrameStats>(stats.data() + 3, stats.size() - 3),
      &error));
  CHECK(controller.begin_frame(plan, &error));
  CHECK(plan.frame_index == 3);
  CHECK(controller.end_frame(
      static_cast<std::int64_t>(plan.estimated_bits), &error));

  // Offline analyze of the same clip produces the same first-three targets
  // as a window over the whole clip.
  TwoPassController offline;
  TwoPassConfig offline_config = vbr_config(500000.0);
  CHECK(offline.configure(offline_config, &error));
  CHECK(offline.analyze(stats, &error));
  CHECK(offline.allocation().frames.size() == 8);
  TwoPassController windowed;
  CHECK(windowed.configure(offline_config, &error));
  CHECK(windowed.set_lookahead(stats, &error));
  CHECK(windowed.allocation().frames.size() == 8);
  for (std::size_t i = 0; i < 8; ++i) {
    CHECK(offline.allocation().frames[i].target_bits ==
          windowed.allocation().frames[i].target_bits);
  }

  // Sequence errors.
  CHECK(offline.begin_frame(plan, &error));
  CHECK(!offline.begin_frame(plan, &error));
  CHECK(!offline.analyze(stats, &error));
  CHECK(!offline.set_lookahead(stats, &error));
  CHECK(offline.end_frame(static_cast<std::int64_t>(plan.estimated_bits),
                          &error));
  CHECK(!offline.end_frame(0, &error));
  CHECK(offline.begin_frame(plan, &error));
  CHECK(!offline.end_frame(-1, &error));
  CHECK(offline.end_frame(static_cast<std::int64_t>(plan.estimated_bits),
                          &error));

  // reset() keeps the config and clears history.
  CHECK(offline.reset(&error));
  CHECK(offline.frame_count() == 0);
  CHECK(offline.lookahead_size() == 0);
  CHECK(offline.config().rc.mode == RateControlMode::kVbr);
}


// ---------------------------------------------------------------------------
// Streaming buffer credit
// ---------------------------------------------------------------------------

void test_streaming_buffer_credit() {
  std::vector<FrameStats> stats(4, flat_stats(640, 360, 100.0));
  TwoPassConfig config = vbr_config(30000.0, true);
  config.lookahead_frames = 2;
  config.vbv.buffer_frames = 4.0;
  config.vbv.initial_fill = 0.5;
  config.vbv.max_rate_bps = 60000.0;
  config.vbv.underflow_margin = 0.02;
  config.max_qindex_step = 255;

  std::string error;
  // Baseline: a fresh window with the buffer at its ideal fill.
  TwoPassController fresh;
  CHECK(fresh.configure(config, &error));
  CHECK(fresh.set_lookahead(stats, &error));
  const double fresh_target = fresh.allocation().frames[0].target_bits;

  // Underspend two frames: the next window gets a positive buffer credit and
  // a larger per-frame allocation.
  TwoPassController under;
  CHECK(under.configure(config, &error));
  CHECK(under.set_lookahead(stats, &error));
  for (int i = 0; i < 2; ++i) {
    FramePlan plan;
    CHECK(under.begin_frame(plan, &error));
    CHECK(under.end_frame(0, &error));
  }
  CHECK(under.vbv().fill_bits > 0.5 * under.vbv().size_bits);
  CHECK(under.set_lookahead(
      std::span<const FrameStats>(stats.data() + 2, 2), &error));
  const double under_target = under.allocation().frames[0].target_bits;
  CHECK(under_target > fresh_target * 1.5);

  // Overspend two frames: the next window gets a negative credit.
  TwoPassController over;
  CHECK(over.configure(config, &error));
  CHECK(over.set_lookahead(stats, &error));
  for (int i = 0; i < 2; ++i) {
    FramePlan plan;
    CHECK(over.begin_frame(plan, &error));
    CHECK(over.end_frame(3000, &error));
  }
  CHECK(over.set_lookahead(
      std::span<const FrameStats>(stats.data() + 2, 2), &error));
  const double over_target = over.allocation().frames[0].target_bits;
  CHECK(over_target < fresh_target * 0.75);
}

// ---------------------------------------------------------------------------
// CRF / VBR target tracking
// ---------------------------------------------------------------------------

struct LinearEstimator {
  double header = 100.0;
  double bits_per_qindex = 20.0;
  double scale = 1.0;
};

double linear_estimate(void* context,
                       const FrameComplexity& /*complexity*/,
                       int qindex) {
  const auto* estimator = static_cast<const LinearEstimator*>(context);
  return estimator->scale *
         (estimator->header +
          estimator->bits_per_qindex * static_cast<double>(255 - qindex));
}

void test_target_tracking() {
  std::vector<FrameStats> stats;
  for (int i = 0; i < 12; ++i) {
    stats.push_back(flat_stats(640, 360, 50.0));
  }
  std::string error;

  // VBR against the built-in model: feed the estimate back as the actual.
  TwoPassConfig config = vbr_config(100000.0);
  TwoPassController controller;
  CHECK(controller.configure(config, &error));
  CHECK(controller.analyze(stats, &error));
  int previous = 0;
  double total_actual = 0.0;
  for (int i = 0; i < 12; ++i) {
    FramePlan plan;
    CHECK(controller.begin_frame(plan, &error));
    if (i > 0) {
      CHECK(std::abs(plan.base_q_idx - previous) <= 2);
    }
    previous = plan.base_q_idx;
    total_actual += plan.estimated_bits;
    CHECK(controller.end_frame(
        static_cast<std::int64_t>(plan.estimated_bits), &error));
  }
  CHECK(near(total_actual, 12.0 * 100000.0 / 30.0, 12.0 * 100.0));

  // VBR against a fitted linear estimator: the target is hit almost exactly.
  LinearEstimator linear;
  TwoPassConfig linear_config = vbr_config(30000.0);
  linear_config.rc.estimator.context = &linear;
  linear_config.rc.estimator.estimate = &linear_estimate;
  std::vector<FrameStats> linear_stats(10, flat_stats(640, 360, 50.0));
  AllocationResult scratch;
  CHECK(allocate_frames(linear_stats, linear_config, 10000.0, VbvState{},
                        scratch, &error));
  TwoPassController linear_controller;
  CHECK(linear_controller.configure(linear_config, &error));
  CHECK(linear_controller.analyze(linear_stats, &error));
  for (int i = 0; i < 10; ++i) {
    FramePlan plan;
    CHECK(linear_controller.begin_frame(plan, &error));
    const double estimate = linear_estimate(&linear, stats[0].complexity,
                                            plan.base_q_idx);
    CHECK(estimate <= 1000.0 + linear.bits_per_qindex + 1e-6);
    CHECK(estimate >= 1000.0 - linear.bits_per_qindex - 1e-6);
    CHECK(linear_controller.end_frame(
        static_cast<std::int64_t>(estimate), &error));
  }

  // CRF: qcomp off is constant quality across complexity.
  TwoPassConfig crf = crf_config(32.0);
  std::vector<FrameStats> mixed;
  mixed.push_back(flat_stats(640, 360, 4.0));
  mixed.push_back(flat_stats(640, 360, 400.0));
  mixed.push_back(flat_stats(640, 360, 40000.0));
  TwoPassController crf_controller;
  CHECK(crf_controller.configure(crf, &error));
  CHECK(crf_controller.analyze(mixed, &error));
  int q0 = 0;
  int q2 = 0;
  for (int i = 0; i < 3; ++i) {
    FramePlan plan;
    CHECK(crf_controller.begin_frame(plan, &error));
    CHECK(plan.crf_quality_target);
    if (i == 0) q0 = plan.base_q_idx;
    if (i == 2) q2 = plan.base_q_idx;
    CHECK(crf_controller.end_frame(
        static_cast<std::int64_t>(plan.estimated_bits), &error));
  }
  CHECK(q0 == q2);

  // CRF qcomp: a harder frame gets a coarser qindex.
  crf.rc.crf.qcomp = 0.5;
  crf.rc.crf.complexity_reference = 20.0;
  TwoPassController qcomp_controller;
  CHECK(qcomp_controller.configure(crf, &error));
  CHECK(qcomp_controller.analyze(mixed, &error));
  int q_low = 0;
  int q_high = 0;
  for (int i = 0; i < 3; ++i) {
    FramePlan plan;
    CHECK(qcomp_controller.begin_frame(plan, &error));
    if (i == 0) q_low = plan.base_q_idx;
    if (i == 2) q_high = plan.base_q_idx;
    CHECK(qcomp_controller.end_frame(
        static_cast<std::int64_t>(plan.estimated_bits), &error));
  }
  CHECK(q_high > q_low);

  // VBV underflow override: a tiny bucket forces a coarser qindex than CRF.
  TwoPassConfig tight = crf_config(32.0);
  tight.vbv.enabled = true;
  tight.vbv.buffer_size_bits = 2000.0;
  tight.vbv.initial_fill = 0.05;
  tight.vbv.max_rate_bps = 30000.0;
  tight.vbv.underflow_margin = 0.02;
  std::vector<FrameStats> expensive(1, flat_stats(640, 360, 20000.0));
  TwoPassController tight_controller;
  CHECK(tight_controller.configure(tight, &error));
  CHECK(tight_controller.analyze(expensive, &error));
  FramePlan tight_plan;
  CHECK(tight_controller.begin_frame(tight_plan, &error));
  CHECK(tight_plan.base_q_idx > q0);
  CHECK(tight_plan.vbv_underflow_prevented);
  CHECK(tight_plan.target_bits <= vbv_max_bits(tight_controller.vbv()) + 1e-6);
  CHECK(tight_controller.end_frame(
      static_cast<std::int64_t>(tight_plan.estimated_bits), &error));

  // Slew limit: a cheap frame followed by an expensive frame moves by at most
  // max_qindex_step when the VBV is disabled.
  TwoPassConfig slew = vbr_config(100000.0);
  slew.max_qindex_step = 4;
  std::vector<FrameStats> step_stats;
  step_stats.push_back(flat_stats(640, 360, 0.01));
  step_stats.push_back(flat_stats(640, 360, 100000.0));
  TwoPassController slew_controller;
  CHECK(slew_controller.configure(slew, &error));
  CHECK(slew_controller.analyze(step_stats, &error));
  FramePlan p0;
  FramePlan p1;
  CHECK(slew_controller.begin_frame(p0, &error));
  CHECK(slew_controller.end_frame(
      static_cast<std::int64_t>(p0.estimated_bits), &error));
  CHECK(slew_controller.begin_frame(p1, &error));
  CHECK(p1.base_q_idx <= p0.base_q_idx + 4);
  CHECK(p1.base_q_idx >= p0.base_q_idx - 4);
  CHECK(p1.clamped_by_slew);
  CHECK(slew_controller.end_frame(
      static_cast<std::int64_t>(p1.estimated_bits), &error));

  // Rate feedback: a 2x biased estimator learns from actual/estimate.
  LinearEstimator biased;
  biased.scale = 2.0;
  TwoPassConfig feedback = vbr_config(100000.0);
  feedback.rc.estimator.context = &biased;
  feedback.rc.estimator.estimate = &linear_estimate;
  feedback.max_qindex_step = 255;
  std::vector<FrameStats> feedback_stats(10, flat_stats(640, 360, 50.0));
  TwoPassConfig no_feedback = feedback;
  no_feedback.rate_feedback_weight = 0.0;
  // Isolate the QP feedback from the bit-budget reallocation.
  feedback.redistribute_bits = false;
  no_feedback.redistribute_bits = false;

  TwoPassController with_fb;
  TwoPassController without_fb;
  CHECK(with_fb.configure(feedback, &error));
  CHECK(without_fb.configure(no_feedback, &error));
  CHECK(with_fb.analyze(feedback_stats, &error));
  CHECK(without_fb.analyze(feedback_stats, &error));
  int q_with = 0;
  int q_without = 0;
  double actual_with = 0.0;
  double actual_without = 0.0;
  for (int i = 0; i < 10; ++i) {
    FramePlan a;
    FramePlan b;
    CHECK(with_fb.begin_frame(a, &error));
    CHECK(without_fb.begin_frame(b, &error));
    q_with = a.base_q_idx;
    q_without = b.base_q_idx;
    // The real rate is half the biased estimate.
    const double true_bits = linear_estimate(&biased, feedback_stats[0].complexity,
                                             a.base_q_idx) / 2.0;
    const double true_bits_b = linear_estimate(&biased, feedback_stats[0].complexity,
                                               b.base_q_idx) / 2.0;
    actual_with += true_bits;
    actual_without += true_bits_b;
    CHECK(with_fb.end_frame(static_cast<std::int64_t>(true_bits), &error));
    CHECK(without_fb.end_frame(static_cast<std::int64_t>(true_bits_b), &error));
  }
  CHECK(with_fb.rate_feedback() > 0.4);
  CHECK(with_fb.rate_feedback() < 0.6);
  CHECK(q_with <= q_without);
  CHECK(actual_with > actual_without * 1.5);
}

// ---------------------------------------------------------------------------
// Per-tile and per-superblock QP assignment
// ---------------------------------------------------------------------------

FrameStats tiled_frame(double mse0,
                       double mse1,
                       double mse2,
                       double mse3,
                       bool weights) {
  constexpr int kWidth = 1280;
  constexpr int kHeight = 720;
  FrameStats stats;
  stats.complexity = complexity(kWidth, kHeight, 100.0, 0.5);
  const int tile_mse[4] = {static_cast<int>(mse0), static_cast<int>(mse1),
                           static_cast<int>(mse2), static_cast<int>(mse3)};
  const int xs[3] = {0, 10, 20};
  const int ys[3] = {0, 6, 12};
  for (int ty = 0; ty < 2; ++ty) {
    for (int tx = 0; tx < 2; ++tx) {
      TileStats tile;
      tile.sb_col_start = xs[tx];
      tile.sb_row_start = ys[ty];
      tile.sb_cols = xs[tx + 1] - xs[tx];
      tile.sb_rows = ys[ty + 1] - ys[ty];
      const int px0 = tile.sb_col_start * 64;
      const int py0 = tile.sb_row_start * 64;
      const int px1 = std::min(kWidth, (tile.sb_col_start + tile.sb_cols) * 64);
      const int py1 = std::min(kHeight, (tile.sb_row_start + tile.sb_rows) * 64);
      tile.complexity.width = px1 - px0;
      tile.complexity.height = py1 - py0;
      tile.complexity.residual_mse = tile_mse[ty * 2 + tx];
      tile.complexity.residual_mad =
          std::sqrt(tile.complexity.residual_mse);
      tile.complexity.intra_fraction = 0.5;
      if (weights) {
        const std::size_t count =
            static_cast<std::size_t>(tile.sb_cols) *
            static_cast<std::size_t>(tile.sb_rows);
        tile.sb_weights.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
          // Horizontal gradient: the rightmost SB of a tile is hardest.
          tile.sb_weights[i] =
              10.0 + static_cast<double>(i % static_cast<std::size_t>(tile.sb_cols)) *
                         100.0;
        }
      }
      stats.tiles.push_back(std::move(tile));
    }
  }
  return stats;
}

void test_per_tile() {
  std::string error;
  FrameStats stats = tiled_frame(900.0, 25.0, 100.0, 400.0, true);

  // Delta-Q only, res 0: achieved equals desired exactly, and every tile
  // starts from base_q_idx.
  TwoPassConfig config = vbr_config(2000000.0);
  config.spatial_qp_strength = 12;
  config.enable_delta_q = true;
  config.delta_q_res = 0;
  config.enable_segmentation = false;
  TwoPassController controller;
  CHECK(controller.configure(config, &error));
  std::vector<FrameStats> one(1, stats);
  CHECK(controller.analyze(one, &error));
  FramePlan plan;
  CHECK(controller.begin_frame(plan, &error));
  CHECK(plan.delta_q.present);
  CHECK(plan.delta_q.res == 0);
  CHECK(!plan.segmentation.enabled);
  CHECK(plan.tiles.size() == 4);
  for (const TilePlan& tile : plan.tiles) {
    CHECK(tile.delta_q.steps.size() ==
          static_cast<std::size_t>(tile.sb_cols * tile.sb_rows));
    CHECK(tile.delta_q.steps.front().qindex_before == plan.base_q_idx);
    for (std::size_t i = 0; i < tile.sb_achieved_qindex.size(); ++i) {
      CHECK(tile.sb_achieved_qindex[i] == tile.sb_target_qindex[i]);
    }
  }
  // The hardest tile gets finer quantisation than the easiest tile.
  const int q_tile0 = plan.tiles[0].target_qindex;
  const int q_tile1 = plan.tiles[1].target_qindex;
  CHECK(q_tile0 < q_tile1);
  CHECK(q_tile0 < plan.base_q_idx);
  CHECK(q_tile1 > plan.base_q_idx);
  // Within a tile, the high-activity side gets the lower qindex.
  CHECK(plan.tiles[0].sb_target_qindex.front() >
        plan.tiles[0].sb_target_qindex.back());
  CHECK(controller.end_frame(
      static_cast<std::int64_t>(plan.estimated_bits), &error));

  // Replanning a tile is independent of the other tiles (tile reset).
  DeltaQPlanConfig plan_config;
  plan_config.base_q_idx = plan.base_q_idx;
  plan_config.delta_q_res = 0;
  plan_config.cols = plan.tiles[2].sb_cols;
  plan_config.rows = plan.tiles[2].sb_rows;
  DeltaQPlan repeated;
  CHECK(plan_delta_q(plan_config, plan.tiles[2].sb_target_qindex, repeated,
                     &error));
  CHECK(repeated.steps.size() == plan.tiles[2].delta_q.steps.size());
  for (std::size_t i = 0; i < repeated.steps.size(); ++i) {
    CHECK(repeated.steps[i].qindex_after ==
          plan.tiles[2].delta_q.steps[i].qindex_after);
    CHECK(repeated.steps[i].reduced_delta ==
          plan.tiles[2].delta_q.steps[i].reduced_delta);
  }

  // Coarse delta-Q res: achieved stays within 2^res - 1 of the target.
  config.delta_q_res = 2;
  TwoPassController coarse;
  CHECK(coarse.configure(config, &error));
  CHECK(coarse.analyze(one, &error));
  FramePlan coarse_plan;
  CHECK(coarse.begin_frame(coarse_plan, &error));
  for (const TilePlan& tile : coarse_plan.tiles) {
    for (std::size_t i = 0; i < tile.sb_achieved_qindex.size(); ++i) {
      CHECK(std::abs(tile.sb_achieved_qindex[i] -
                     tile.sb_target_qindex[i]) <= 3);
    }
    CHECK(tile.delta_q.steps.front().qindex_before == coarse_plan.base_q_idx);
  }
  CHECK(coarse.end_frame(
      static_cast<std::int64_t>(coarse_plan.estimated_bits), &error));

  // Segmentation only: per-SB classes, no delta-Q. Achieved is base + ALT_Q.
  TwoPassConfig seg = vbr_config(2000000.0);
  seg.spatial_qp_strength = 8;
  seg.enable_delta_q = false;
  seg.enable_segmentation = true;
  seg.segment_levels = 2;
  seg.segment_qindex_step = 4;
  TwoPassController seg_controller;
  CHECK(seg_controller.configure(seg, &error));
  CHECK(seg_controller.analyze(one, &error));
  FramePlan seg_plan;
  CHECK(seg_controller.begin_frame(seg_plan, &error));
  CHECK(!seg_plan.delta_q.present);
  CHECK(seg_plan.segmentation.enabled);
  CHECK(seg_plan.segmentation.update_map);
  CHECK(seg_plan.segmentation_map.mi_cols == 320);
  CHECK(seg_plan.segmentation_map.mi_rows == 180);
  CHECK(seg_plan.segmentation_map.segment_id.size() ==
        static_cast<std::size_t>(320 * 180));
  for (const TilePlan& tile : seg_plan.tiles) {
    CHECK(tile.delta_q.steps.empty());
    for (std::size_t i = 0; i < tile.sb_achieved_qindex.size(); ++i) {
      const int id = static_cast<int>(tile.sb_segment_id[i]);
      const int alt = static_cast<int>(
          seg_plan.segmentation.segments[static_cast<std::size_t>(id)]
              .data[static_cast<std::size_t>(kSegLvlAltQ)]);
      CHECK(tile.sb_achieved_qindex[i] ==
            std::clamp(seg_plan.base_q_idx + alt, 0, 255));
    }
  }
  CHECK(seg_plan.tiles[0].target_qindex < seg_plan.tiles[1].target_qindex);
  CHECK(seg_controller.end_frame(
      static_cast<std::int64_t>(seg_plan.estimated_bits), &error));

  // Both mechanisms: effective qindex (get_qindex) matches the achieved
  // value for every superblock, and ignoreDeltaQ removes the delta step.
  TwoPassConfig both = config;
  both.delta_q_res = 1;
  both.enable_segmentation = true;
  both.segment_levels = 2;
  TwoPassController both_controller;
  CHECK(both_controller.configure(both, &error));
  CHECK(both_controller.analyze(one, &error));
  FramePlan both_plan;
  CHECK(both_controller.begin_frame(both_plan, &error));
  CHECK(both_plan.delta_q.present);
  CHECK(both_plan.segmentation.enabled);
  for (const TilePlan& tile : both_plan.tiles) {
    const std::size_t count =
        static_cast<std::size_t>(tile.sb_cols) *
        static_cast<std::size_t>(tile.sb_rows);
    for (std::size_t i = 0; i < count; ++i) {
      QpContext ctx;
      ctx.base_q_idx = both_plan.base_q_idx;
      ctx.delta_q = both_plan.delta_q;
      ctx.current_q_index = tile.delta_q.steps[i].qindex_after;
      ctx.segmentation = both_plan.segmentation;
      int effective = 0;
      CHECK(effective_qindex(ctx, static_cast<int>(tile.sb_segment_id[i]),
                             false, effective, &error));
      CHECK(effective == tile.sb_achieved_qindex[i]);
      int without_delta = 0;
      CHECK(effective_qindex(ctx, static_cast<int>(tile.sb_segment_id[i]),
                             true, without_delta, &error));
      const int id = static_cast<int>(tile.sb_segment_id[i]);
      const int alt = static_cast<int>(
          both_plan.segmentation.segments[static_cast<std::size_t>(id)]
              .data[static_cast<std::size_t>(kSegLvlAltQ)]);
      CHECK(without_delta ==
            std::clamp(both_plan.base_q_idx + alt, 0, 255));
    }
  }
  CHECK(both_controller.end_frame(
      static_cast<std::int64_t>(both_plan.estimated_bits), &error));

  // Spatial strength 0 is exactly frame-uniform even with delta-Q enabled.
  TwoPassConfig uniform = vbr_config(2000000.0);
  uniform.spatial_qp_strength = 0;
  uniform.enable_delta_q = true;
  TwoPassController uniform_controller;
  CHECK(uniform_controller.configure(uniform, &error));
  CHECK(uniform_controller.analyze(one, &error));
  FramePlan uniform_plan;
  CHECK(uniform_controller.begin_frame(uniform_plan, &error));
  for (const TilePlan& tile : uniform_plan.tiles) {
    for (const int q : tile.sb_target_qindex) {
      CHECK(q == uniform_plan.base_q_idx);
    }
    CHECK(tile.delta_q.max_abs_reduced_delta == 0);
  }
  CHECK(uniform_controller.end_frame(
      static_cast<std::int64_t>(uniform_plan.estimated_bits), &error));

  // Bad tile geometry is rejected.
  FrameStats broken = stats;
  broken.tiles.pop_back();
  CHECK(!normalize_frame_stats(broken, broken, &error));
  broken = stats;
  broken.tiles[0].sb_col_start = 5;
  CHECK(!normalize_frame_stats(broken, broken, &error));
}

// ---------------------------------------------------------------------------
// First pass over a luma plane
// ---------------------------------------------------------------------------

void test_first_pass() {
  constexpr int kWidth = 256;
  constexpr int kHeight = 256;
  std::vector<std::uint8_t> luma(
      static_cast<std::size_t>(kWidth) * kHeight, 0);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      std::uint8_t value = 0;
      if (x < kWidth / 2) {
        value = static_cast<std::uint8_t>(100 + (y / 16) % 2);
      } else {
        value = static_cast<std::uint8_t>(((x / 4) + (y / 4)) % 2 == 0 ? 0 : 255);
      }
      luma[static_cast<std::size_t>(y) * kWidth +
           static_cast<std::size_t>(x)] = value;
    }
  }
  FrameStats stats;
  std::string error;
  CHECK(first_pass_luma(luma, kWidth, kHeight, kWidth, 2, 1, true, stats,
                        &error));
  CHECK(stats.keyframe);
  CHECK(stats.sb_size == 64);
  CHECK(stats.tiles.size() == 2);
  CHECK(stats.complexity.residual_mse > 100.0);
  CHECK(stats.tiles[1].complexity.residual_mse >
        stats.tiles[0].complexity.residual_mse);
  for (const TileStats& tile : stats.tiles) {
    CHECK(tile.sb_weights.size() ==
          static_cast<std::size_t>(tile.sb_cols * tile.sb_rows));
    CHECK(tile.complexity.width > 0);
    CHECK(tile.complexity.height > 0);
  }
  // The frame moments are the pixel-weighted combination of the tiles.
  double sum_sq = 0.0;
  double sum_pixels = 0.0;
  for (const TileStats& tile : stats.tiles) {
    const double pixels = static_cast<double>(tile.complexity.pixel_count());
    sum_sq += tile.complexity.residual_mse * pixels;
    sum_pixels += pixels;
  }
  CHECK(near(sum_sq / sum_pixels, stats.complexity.residual_mse, 1e-9));

  FrameStats again;
  CHECK(first_pass_luma(luma, kWidth, kHeight, kWidth, 2, 1, true, again,
                        &error));
  CHECK(again.complexity.residual_mse == stats.complexity.residual_mse);
  CHECK(again.tiles[1].sb_weights == stats.tiles[1].sb_weights);

  // Padded stride is ignored outside the visible area.
  constexpr int kStride = kWidth + 24;
  std::vector<std::uint8_t> padded(
      static_cast<std::size_t>(kStride) * kHeight, 255);
  for (int y = 0; y < kHeight; ++y) {
    std::copy_n(luma.data() + static_cast<std::size_t>(y) * kWidth, kWidth,
                padded.data() + static_cast<std::size_t>(y) * kStride);
  }
  FrameStats padded_stats;
  CHECK(first_pass_luma(padded, kWidth, kHeight, kStride, 2, 1, true,
                        padded_stats, &error));
  CHECK(padded_stats.complexity.residual_mse == stats.complexity.residual_mse);
  CHECK(padded_stats.tiles[0].sb_weights == stats.tiles[0].sb_weights);

  // Invalid inputs.
  CHECK(!first_pass_luma(luma, kWidth, kHeight, kWidth, 9, 1, false, stats,
                         &error));
  CHECK(!first_pass_luma(luma, 0, kHeight, kWidth, 1, 1, false, stats,
                         &error));
  CHECK(!first_pass_luma(
      std::span<const std::uint8_t>(luma.data(), 10), kWidth, kHeight,
      kWidth, 1, 1, false, stats, &error));
  CHECK(!first_pass_luma(luma, kWidth, kHeight, kWidth - 1, 1, 1, false,
                         stats, &error));
}


// ---------------------------------------------------------------------------
// Failure recovery
// ---------------------------------------------------------------------------

struct FailingEstimator {
  int calls = 0;
  int fail_after = 1;
};

double failing_estimate(void* context,
                        const FrameComplexity& /*complexity*/,
                        int /*qindex*/) {
  auto* state = static_cast<FailingEstimator*>(context);
  const int call = state->calls++;
  if (call >= state->fail_after) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return 1000.0;
}

void test_failure_recovery() {
  std::string error;
  TwoPassConfig config = crf_config(32.0);
  FailingEstimator failing;
  config.rc.estimator.context = &failing;
  config.rc.estimator.estimate = &failing_estimate;
  std::vector<FrameStats> one(1, flat_stats(640, 360, 100.0));

  TwoPassController controller;
  CHECK(controller.configure(config, &error));
  CHECK(controller.analyze(one, &error));

  // The inner CRF controller's begin_frame consumes the first estimate; the
  // spatial plan's estimate fails. begin_frame() must roll the inner frame
  // back so a retry works.
  FramePlan plan;
  CHECK(!controller.begin_frame(plan, &error));
  CHECK(!error.empty());
  CHECK(!controller.frame_open());
  failing.fail_after = 1000;
  failing.calls = 0;
  CHECK(controller.begin_frame(plan, &error));
  CHECK(controller.frame_open());
  CHECK(controller.end_frame(
      static_cast<std::int64_t>(plan.estimated_bits), &error));
}

// ---------------------------------------------------------------------------
// Configuration and determinism
// ---------------------------------------------------------------------------

void test_config_and_determinism() {
  std::string error;
  TwoPassConfig config = vbr_config(500000.0, true);
  CHECK(two_pass_config_valid(config, &error));
  config.complexity_exponent = 3.0;
  CHECK(!two_pass_config_valid(config, &error));
  config = vbr_config(500000.0);
  config.lookahead_frames = -1;
  CHECK(!two_pass_config_valid(config, &error));
  config = vbr_config(500000.0);
  config.min_frame_bits_ratio = 0.0;
  CHECK(!two_pass_config_valid(config, &error));
  config = vbr_config(500000.0);
  config.max_frame_bits_ratio = 0.5;
  CHECK(!two_pass_config_valid(config, &error));
  config = vbr_config(500000.0);
  config.max_qindex = 256;
  CHECK(!two_pass_config_valid(config, &error));
  config = vbr_config(500000.0);
  config.max_qindex_step = 0;
  CHECK(!two_pass_config_valid(config, &error));
  config = vbr_config(500000.0);
  config.delta_q_res = 4;
  CHECK(!two_pass_config_valid(config, &error));
  config = vbr_config(500000.0);
  config.enable_segmentation = true;
  config.segment_levels = 8;
  CHECK(!two_pass_config_valid(config, &error));
  config.enable_ext_seg = true;
  CHECK(two_pass_config_valid(config, &error));
  config = vbr_config(500000.0);
  config.rate_feedback_weight = 1.5;
  CHECK(!two_pass_config_valid(config, &error));
  config = vbr_config(500000.0);
  config.rate_feedback_clamp = 0.5;
  CHECK(!two_pass_config_valid(config, &error));
  config = vbr_config(500000.0);
  config.vbv.underflow_margin = 0.6;
  CHECK(!two_pass_config_valid(config, &error));

  // Full deterministic run: same inputs and actuals produce bit-identical
  // plans, including per-SB targets and delta-Q steps.
  std::vector<FrameStats> stats;
  for (int i = 0; i < 10; ++i) {
    FrameStats frame = tiled_frame(
        100.0 + 10.0 * i, 25.0 + i, 400.0 - 10.0 * i, 900.0 - 20.0 * i, true);
    frame.keyframe = (i == 0 || i == 5);
    stats.push_back(std::move(frame));
  }
  TwoPassConfig full = vbr_config(1500000.0, true);
  full.spatial_qp_strength = 10;
  full.enable_delta_q = true;
  full.delta_q_res = 1;
  full.enable_segmentation = true;
  full.segment_levels = 2;
  full.vbv.buffer_size_bits = 300000.0;
  full.vbv.initial_fill = 0.4;
  full.vbv.max_rate_bps = 2500000.0;
  full.max_qindex_step = 32;

  std::vector<FramePlan> first_run;
  std::vector<FramePlan> second_run;
  for (int run = 0; run < 2; ++run) {
    TwoPassController controller;
    CHECK(controller.configure(full, &error));
    CHECK(controller.analyze(stats, &error));
    std::vector<FramePlan>& output = run == 0 ? first_run : second_run;
    for (std::size_t i = 0; i < stats.size(); ++i) {
      FramePlan plan;
      CHECK(controller.begin_frame(plan, &error));
      output.push_back(plan);
      const double actual =
          plan.estimated_bits * (1.0 + 0.05 * static_cast<double>(i % 3));
      CHECK(controller.end_frame(
          static_cast<std::int64_t>(actual), &error));
    }
  }
  CHECK(first_run.size() == second_run.size());
  for (std::size_t i = 0; i < first_run.size(); ++i) {
    CHECK(same_plan(first_run[i], second_run[i]));
  }

  // Two controllers in lockstep also agree.
  TwoPassController a;
  TwoPassController b;
  CHECK(a.configure(full, &error));
  CHECK(b.configure(full, &error));
  CHECK(a.analyze(stats, &error));
  CHECK(b.analyze(stats, &error));
  for (std::size_t i = 0; i < stats.size(); ++i) {
    FramePlan pa;
    FramePlan pb;
    CHECK(a.begin_frame(pa, &error));
    CHECK(b.begin_frame(pb, &error));
    CHECK(same_plan(pa, pb));
    const double actual = pa.estimated_bits * 1.02;
    CHECK(a.end_frame(static_cast<std::int64_t>(actual), &error));
    CHECK(b.end_frame(static_cast<std::int64_t>(actual), &error));
  }

  // Reset and reconfigure are deterministic too.
  CHECK(a.reset(&error));
  CHECK(a.analyze(stats, &error));
  FramePlan reset_plan;
  CHECK(a.begin_frame(reset_plan, &error));
  CHECK(reset_plan.frame_index == 0);
}

}  // namespace

int main() {
  test_vbv_model();
  test_allocation();
  test_lookahead_and_sequencing();
  test_streaming_buffer_credit();
  test_target_tracking();
  test_per_tile();
  test_first_pass();
  test_failure_recovery();
  test_config_and_determinism();

  std::cout << g_checks << " checks, " << g_failures << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
