#include "rc.h"

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

bool near(double a, double b, double epsilon = 1e-9) {
  return std::fabs(a - b) <= epsilon;
}

FrameComplexity synthetic(int width,
                          int height,
                          double mad,
                          double mse = 0.0,
                          double intra_fraction = 0.0) {
  FrameComplexity complexity;
  complexity.width = width;
  complexity.height = height;
  complexity.residual_mad = mad;
  complexity.residual_mse = mse;
  complexity.intra_fraction = intra_fraction;
  return complexity;
}

RateControlConfig cqp_config(int qindex) {
  RateControlConfig config;
  config.mode = RateControlMode::kCqp;
  config.cqp.base_q_idx = qindex;
  return config;
}

RateControlConfig crf_config(double crf) {
  RateControlConfig config;
  config.mode = RateControlMode::kCrf;
  config.crf.crf = crf;
  return config;
}

RateControlConfig vbr_config(double bitrate_bps, int max_step = kMaxQIndex) {
  RateControlConfig config;
  config.mode = RateControlMode::kVbr;
  config.vbr.target_bitrate_bps = bitrate_bps;
  config.vbr.frame_rate_num = 30;
  config.vbr.frame_rate_den = 1;
  config.vbr.max_qindex_step = max_step;
  return config;
}

bool decision_equal(const QpDecision& a, const QpDecision& b) {
  return a.base_q_idx == b.base_q_idx &&
         a.coeff_cdf_qctx == b.coeff_cdf_qctx && a.qp_slice == b.qp_slice &&
         a.continuous_qp == b.continuous_qp &&
         a.estimated_bits == b.estimated_bits &&
         a.lambda_sse == b.lambda_sse &&
         a.target_reachable == b.target_reachable &&
         a.clamped_by_step == b.clamped_by_step;
}

struct TestEstimator {
  double base_bits = 0.0;
  double slope_per_qindex = 10.0;
};

double test_linear_estimator(void* context,
                             const FrameComplexity& complexity,
                             int base_q_idx) {
  const auto* estimator = static_cast<const TestEstimator*>(context);
  const double pixels = static_cast<double>(complexity.pixel_count());
  return estimator->base_bits +
         estimator->slope_per_qindex *
             (static_cast<double>(kMaxQIndex + 1 - base_q_idx)) +
         pixels * 0.001;
}

double test_nan_estimator(void*, const FrameComplexity&, int) {
  return std::numeric_limits<double>::quiet_NaN();
}

double test_negative_estimator(void*, const FrameComplexity&, int) {
  return -1.0;
}

void test_quantizer_and_slices() {
  CHECK(near(quant_step_from_qindex(0), 64.0));
  CHECK(near(quant_step_from_qindex(1), 40.0));
  CHECK(near(quant_step_from_qindex(24), 78.0));
  CHECK(near(quant_step_from_qindex(25), 80.0));
  CHECK(near(quant_step_from_qindex(255), 61440.0));
  CHECK(near(quant_step_from_qindex(-10), 64.0));
  CHECK(near(quant_step_from_qindex(999), 61440.0));

  double previous = 0.0;
  for (int q = 1; q <= kMaxQIndex; ++q) {
    const double step = quant_step_from_qindex(q);
    CHECK(step >= previous);
    previous = step;
  }

  CHECK(qindex_from_qp(-3.0) == 0);
  CHECK(qindex_from_qp(0.0) == 0);
  CHECK(qindex_from_qp(90.4) == 90);
  CHECK(qindex_from_qp(90.5) == 91);
  CHECK(qindex_from_qp(255.6) == 255);
  CHECK(qindex_from_qp(1000.0) == 255);
  CHECK(qindex_from_qp(std::numeric_limits<double>::quiet_NaN()) == 0);

  CHECK(clamp_qindex(-1) == 0);
  CHECK(clamp_qindex(300) == 255);
  CHECK(clamp_base_qindex(-1) == kMinQIndex);
  CHECK(clamp_base_qindex(0) == kMinQIndex);
  CHECK(clamp_base_qindex(300) == kMaxQIndex);

  CHECK(near(lambda_sse_from_qindex(0), 0.0));
  CHECK(near(lambda_sse_from_qindex(-5), 0.0));
  CHECK(near(lambda_sse_from_qindex(1), 200.0));
  CHECK(near(lambda_sse_from_qindex(24), 0.125 * 78.0 * 78.0));
  CHECK(near(lambda_sse_from_qindex(24, -1.0), 0.0));

  CHECK(coeff_cdf_qctx(0) == 0);
  CHECK(coeff_cdf_qctx(90) == 0);
  CHECK(coeff_cdf_qctx(91) == 1);
  CHECK(coeff_cdf_qctx(140) == 1);
  CHECK(coeff_cdf_qctx(141) == 2);
  CHECK(coeff_cdf_qctx(190) == 2);
  CHECK(coeff_cdf_qctx(191) == 3);
  CHECK(coeff_cdf_qctx(255) == 3);
  CHECK(coeff_cdf_qctx(-4) == 0);
  CHECK(coeff_cdf_qctx(999) == 3);

  CHECK(kCoeffCdfQpSlices.size() == 4);
  for (std::size_t i = 0; i < kCoeffCdfQpSlices.size(); ++i) {
    const QpSlice& slice = kCoeffCdfQpSlices[i];
    CHECK(slice.index == static_cast<int>(i));
    CHECK(slice.min_qindex <= slice.max_qindex);
    CHECK(&coeff_cdf_slice(slice.min_qindex) == &slice);
    CHECK(&coeff_cdf_slice(slice.max_qindex) == &slice);
    if (i + 1 < kCoeffCdfQpSlices.size()) {
      CHECK(slice.max_qindex + 1 == kCoeffCdfQpSlices[i + 1].min_qindex);
    }
  }
  CHECK(kCoeffCdfQpSlices.front().min_qindex == 0);
  CHECK(kCoeffCdfQpSlices.back().max_qindex == kMaxQIndex);
}

void test_complexity() {
  std::vector<std::uint8_t> original(16, 0);
  std::vector<std::uint8_t> predicted(16, 0);
  original[0] = 3;

  FrameComplexity complexity =
      complexity_from_u8_planes(original, predicted, 4, 4, 4);
  CHECK(complexity.width == 4);
  CHECK(complexity.height == 4);
  CHECK(near(complexity.residual_mad, 3.0 / 16.0));
  CHECK(near(complexity.residual_mse, 9.0 / 16.0));
  CHECK(near(complexity.intra_fraction, 0.0));
  CHECK(frame_complexity_valid(complexity));

  // Padding inside the stride must not contribute.
  std::vector<std::uint8_t> padded_original(4 * 6, 0);
  std::vector<std::uint8_t> padded_predicted(4 * 6, 0);
  padded_original[0] = 3;
  padded_original[4] = 200;
  padded_original[5] = 250;
  padded_predicted[4] = 0;
  const FrameComplexity padded =
      complexity_from_u8_planes(padded_original, padded_predicted, 4, 4, 6);
  CHECK(near(padded.residual_mad, 3.0 / 16.0));
  CHECK(near(padded.residual_mse, 9.0 / 16.0));

  // Insufficient span or invalid dimensions yield an invalid default.
  CHECK(complexity_from_u8_planes(original, predicted, 4, 4, 8).width == 0);
  CHECK(complexity_from_u8_planes(original, predicted, 0, 4, 4).width == 0);
  CHECK(complexity_from_u8_planes(original, predicted, 4, 4, 3).width == 0);

  CHECK(frame_complexity_valid(synthetic(64, 64, 10.0, 100.0, 0.5)));
  CHECK(!frame_complexity_valid(synthetic(0, 64, 10.0)));
  CHECK(!frame_complexity_valid(synthetic(64, 64, -1.0)));
  CHECK(!frame_complexity_valid(synthetic(64, 64, 1.0, -1.0)));
  CHECK(!frame_complexity_valid(
      synthetic(64, 64, 1.0, 1.0, std::numeric_limits<double>::quiet_NaN())));
  CHECK(!frame_complexity_valid(synthetic(64, 64, 1.0, 1.0, 1.5)));
  CHECK(frame_complexity_valid(synthetic(64, 64, 1.0, 1.0, 1.0)));
  CHECK(synthetic(128, 72, 0.0).pixel_count() == 128 * 72);

  CHECK(near(frame_complexity_sigma(synthetic(64, 64, 20.0, 400.0)), 20.0));
  CHECK(near(frame_complexity_sigma(synthetic(64, 64, 8.0, 0.0)),
             8.0 * 1.2533));
  CHECK(near(frame_complexity_sigma(synthetic(64, 64, 8.0, 0.0), 2.0), 16.0));
  CHECK(near(frame_complexity_sigma(synthetic(64, 64, 8.0, 0.0), 0.0), 0.0));
  CHECK(near(frame_complexity_sigma(synthetic(0, 64, 8.0)), 0.0));
}

void test_bitrate_model() {
  const FrameComplexity flat = synthetic(64, 64, 0.0, 0.0);
  const BitrateModelParams params;
  CHECK(near(estimate_frame_bits_model(flat, 90), params.header_bits));

  const FrameComplexity textured = synthetic(64, 64, 20.0, 400.0);
  double previous = std::numeric_limits<double>::infinity();
  for (int q = kMinQIndex; q <= kMaxQIndex; ++q) {
    const double bits = estimate_frame_bits_model(textured, q);
    CHECK(bits <= previous + 1e-9);
    CHECK(bits >= params.header_bits);
    previous = bits;
  }
  CHECK(near(estimate_frame_bits_model(textured, 0),
             estimate_frame_bits_model(textured, 1)));

  const FrameComplexity noisier = synthetic(64, 64, 40.0, 1600.0);
  CHECK(estimate_frame_bits_model(noisier, 64) >
        estimate_frame_bits_model(textured, 64));

  BitrateModelParams bad = params;
  bad.header_bits = -1.0;
  CHECK(!bitrate_model_params_valid(bad));
  bad = params;
  bad.coefficient_bits_per_sample =
      std::numeric_limits<double>::quiet_NaN();
  CHECK(!bitrate_model_params_valid(bad));
  bad = params;
  bad.max_bits_per_pixel = 0.0;
  CHECK(!bitrate_model_params_valid(bad));
  bad = params;
  bad.mad_to_sigma = 0.0;
  CHECK(!bitrate_model_params_valid(bad));

  BitrateEstimator estimator;
  double bits = 0.0;
  std::string error;
  CHECK(estimate_frame_bits(estimator, textured, 90, bits, &error));
  CHECK(near(bits, estimate_frame_bits_model(textured, 90)));
  CHECK(error.empty());

  CHECK(!estimate_frame_bits(estimator, synthetic(0, 64, 1.0), 90, bits,
                             &error));
  CHECK(!error.empty());

  TestEstimator context{1000.0, 10.0};
  estimator.estimate = &test_linear_estimator;
  estimator.context = &context;
  CHECK(estimate_frame_bits(estimator, textured, 90, bits, &error));
  CHECK(bits > 1000.0);

  estimator.estimate = &test_nan_estimator;
  error.clear();
  CHECK(!estimate_frame_bits(estimator, textured, 90, bits, &error));
  CHECK(!error.empty());

  estimator.estimate = &test_negative_estimator;
  error.clear();
  CHECK(!estimate_frame_bits(estimator, textured, 90, bits, &error));
  CHECK(!error.empty());

  estimator.estimate = nullptr;
  bad = params;
  bad.max_bits_per_pixel = -1.0;
  estimator.params = bad;
  CHECK(!bitrate_estimator_valid(estimator));
}

void test_cqp() {
  RateController controller;
  std::string error;
  QpDecision decision;
  CHECK(!controller.begin_frame(synthetic(64, 64, 10.0), decision, &error));
  CHECK(!error.empty());

  RateControlConfig config = cqp_config(90);
  CHECK(controller.configure(config, &error));
  CHECK(controller.config().mode == RateControlMode::kCqp);

  const FrameComplexity complexity = synthetic(64, 64, 10.0, 100.0);
  CHECK(controller.begin_frame(complexity, decision, &error));
  CHECK(decision.base_q_idx == 90);
  CHECK(decision.coeff_cdf_qctx == 0);
  CHECK(decision.qp_slice == 0);
  CHECK(near(decision.continuous_qp, 90.0));
  CHECK(near(decision.lambda_sse, 0.125 * 528.0 * 528.0));
  CHECK(decision.estimated_bits > 160.0);
  CHECK(decision.target_reachable);
  CHECK(!decision.clamped_by_step);
  CHECK(controller.frame_open());

  CHECK(!controller.begin_frame(complexity, decision, &error));
  CHECK(!error.empty());

  CHECK(controller.end_frame(10000));
  CHECK(!controller.frame_open());
  CHECK(controller.frame_count() == 1);
  CHECK(near(controller.bits_ema(), 10000.0));
  CHECK(controller.previous_qindex() == 90);

  CHECK(!controller.end_frame(1));
  CHECK(controller.begin_frame(complexity, decision, &error));
  error.clear();
  CHECK(!controller.end_frame(-1, &error));
  CHECK(!error.empty());
  CHECK(controller.frame_open());  // failed end_frame must not close the frame
  CHECK(controller.end_frame(9000));

  CHECK(controller.configure(cqp_config(91), &error));
  CHECK(controller.frame_count() == 0);
  CHECK(controller.begin_frame(complexity, decision, &error));
  CHECK(decision.base_q_idx == 91);
  CHECK(decision.coeff_cdf_qctx == 1);
  CHECK(decision.qp_slice == 1);
  CHECK(controller.end_frame(8000));

  // Invalid configurations are rejected and leave the previous config intact.
  error.clear();
  CHECK(!controller.configure(cqp_config(0), &error));
  CHECK(!error.empty());
  CHECK(controller.config().cqp.base_q_idx == 91);

  RateControlConfig invalid = config;
  invalid.mode = static_cast<RateControlMode>(9);
  CHECK(!rate_control_config_valid(invalid));

  invalid = config;
  invalid.cqp.base_q_idx = 300;
  CHECK(!rate_control_config_valid(invalid));

  invalid = config;
  invalid.lambda_sse_scale = -1.0;
  CHECK(!rate_control_config_valid(invalid));

  CHECK(controller.configure(config, &error));
  CHECK(controller.begin_frame(complexity, decision, &error));
  controller.reset();
  CHECK(!controller.frame_open());
  CHECK(controller.frame_count() == 0);
  CHECK(controller.previous_qindex() == 0);
  CHECK(near(controller.buffer_bits(), 0.0));
}

void test_crf() {
  RateController controller;
  std::string error;
  QpDecision decision;
  const FrameComplexity complexity = synthetic(64, 64, 10.0, 100.0);

  CHECK(controller.configure(crf_config(0.0), &error));
  CHECK(controller.begin_frame(complexity, decision, &error));
  CHECK(decision.base_q_idx == 1);
  CHECK(controller.end_frame(1000));

  CHECK(controller.configure(crf_config(63.0), &error));
  CHECK(controller.begin_frame(complexity, decision, &error));
  CHECK(decision.base_q_idx == 255);
  CHECK(controller.end_frame(1000));

  CHECK(controller.configure(crf_config(32.0), &error));
  CHECK(controller.begin_frame(complexity, decision, &error));
  CHECK(decision.base_q_idx == 130);
  CHECK(controller.end_frame(1000));

  int previous = 0;
  for (int crf = 0; crf <= 63; ++crf) {
    CHECK(controller.configure(crf_config(static_cast<double>(crf)), &error));
    CHECK(controller.begin_frame(complexity, decision, &error));
    CHECK(decision.base_q_idx >= previous);
    previous = decision.base_q_idx;
  }

  // Default qcomp = 0: constant qindex across frame complexity.
  RateControlConfig adaptive = crf_config(32.0);
  CHECK(controller.configure(adaptive, &error));
  CHECK(controller.begin_frame(synthetic(64, 64, 2.0, 4.0), decision, &error));
  const int low_q = decision.base_q_idx;
  CHECK(controller.end_frame(1000));
  CHECK(controller.begin_frame(synthetic(64, 64, 40.0, 1600.0), decision,
                               &error));
  CHECK(decision.base_q_idx == low_q);
  CHECK(controller.end_frame(1000));

  // A bounded qcomp term shifts qindex with complexity but never past the cap.
  adaptive.crf.qcomp = 1.0;
  adaptive.crf.complexity_reference = 10.0;
  adaptive.crf.max_qindex_adjust = 12.0;
  CHECK(controller.configure(adaptive, &error));
  CHECK(controller.begin_frame(synthetic(64, 64, 5.0, 25.0), decision, &error));
  const int low_complexity_q = decision.base_q_idx;
  CHECK(controller.end_frame(1000));
  CHECK(controller.begin_frame(synthetic(64, 64, 20.0, 400.0), decision,
                               &error));
  const int high_complexity_q = decision.base_q_idx;
  CHECK(controller.end_frame(1000));
  CHECK(low_complexity_q == 118);
  CHECK(high_complexity_q == 142);
  CHECK(high_complexity_q > low_complexity_q);

  adaptive.crf.max_qindex_adjust = 0.0;
  CHECK(controller.configure(adaptive, &error));
  CHECK(controller.begin_frame(synthetic(64, 64, 40.0, 1600.0), decision,
                               &error));
  CHECK(decision.base_q_idx == 130);
  CHECK(controller.end_frame(1000));

  // qcomp requires a positive reference.
  adaptive.crf.max_qindex_adjust = 12.0;
  adaptive.crf.complexity_reference = 0.0;
  CHECK(!rate_control_config_valid(adaptive));

  // Equal endpoints are legal and give a constant qindex.
  RateControlConfig flat = crf_config(31.0);
  flat.crf.qindex_at_crf0 = 50.0;
  flat.crf.qindex_at_crf63 = 50.0;
  CHECK(controller.configure(flat, &error));
  CHECK(controller.begin_frame(complexity, decision, &error));
  CHECK(decision.base_q_idx == 50);
  CHECK(controller.end_frame(1000));

  RateControlConfig invalid = crf_config(64.0);
  CHECK(!rate_control_config_valid(invalid));
  invalid = crf_config(32.0);
  invalid.crf.qindex_at_crf0 = 200.0;
  invalid.crf.qindex_at_crf63 = 100.0;
  CHECK(!rate_control_config_valid(invalid));
}

void test_vbr() {
  RateController controller;
  std::string error;
  QpDecision decision;

  // 128x72 with MSE 900 => sigma 30.
  const FrameComplexity complexity = synthetic(128, 72, 20.0, 900.0);

  RateControlConfig config = vbr_config(300000.0);
  CHECK(rate_control_config_valid(config));
  CHECK(controller.configure(config, &error));
  CHECK(controller.begin_frame(complexity, decision, &error));
  const int medium_q = decision.base_q_idx;
  CHECK(medium_q >= kMinQIndex && medium_q <= kMaxQIndex);
  CHECK(decision.target_reachable);
  CHECK(!decision.clamped_by_step);
  CHECK(near(decision.estimated_bits,
             estimate_frame_bits_model(complexity, medium_q)));
  CHECK(controller.end_frame(9000));

  // A higher target buys a lower qindex (better quality).
  CHECK(controller.configure(vbr_config(600000.0), &error));
  CHECK(controller.begin_frame(complexity, decision, &error));
  const int high_target_q = decision.base_q_idx;
  CHECK(controller.end_frame(9000));
  CHECK(high_target_q <= medium_q);

  // A target above what the best quantizer needs is reachable at q = 1.
  CHECK(controller.configure(vbr_config(100000000.0), &error));
  CHECK(controller.begin_frame(complexity, decision, &error));
  CHECK(decision.base_q_idx == kMinQIndex);
  CHECK(decision.target_reachable);
  CHECK(controller.end_frame(9000));

  // An impossible target falls back to max qindex and is flagged.
  CHECK(controller.configure(vbr_config(100.0), &error));
  CHECK(controller.begin_frame(complexity, decision, &error));
  CHECK(decision.base_q_idx == kMaxQIndex);
  CHECK(!decision.target_reachable);
  CHECK(controller.end_frame(1));

  // Slew limit: an under-budget frame grows the virtual buffer, but the next
  // decision may move by at most max_qindex_step toward the new target.
  RateControlConfig limited = vbr_config(300000.0, 4);
  CHECK(controller.configure(limited, &error));
  CHECK(controller.begin_frame(complexity, decision, &error));
  const int first_q = decision.base_q_idx;
  CHECK(controller.end_frame(0));
  CHECK(controller.buffer_bits() > 0.0);
  CHECK(controller.begin_frame(complexity, decision, &error));
  CHECK(decision.base_q_idx <= first_q);
  CHECK(first_q - decision.base_q_idx <= 4);
  const int under_budget_q = decision.base_q_idx;
  CHECK(controller.end_frame(100000));
  CHECK(controller.buffer_bits() < 0.0);
  // An over-budget frame reverses the correction, still within the slew limit.
  CHECK(controller.begin_frame(complexity, decision, &error));
  CHECK(decision.base_q_idx >= under_budget_q);
  CHECK(decision.base_q_idx - under_budget_q <= 4);
  CHECK(controller.end_frame(100000));

  // The virtual buffer is clamped to +/- buffer_frames * target_bits.
  CHECK(controller.configure(vbr_config(300000.0), &error));
  for (int frame = 0; frame < 40; ++frame) {
    CHECK(controller.begin_frame(complexity, decision, &error));
    CHECK(controller.end_frame(0));
  }
  CHECK(near(controller.buffer_bits(), 30.0 * 10000.0));
  CHECK(controller.frame_count() == 40);

  // A custom estimator hook replaces the built-in model in the controller.
  TestEstimator linear{1000.0, 10.0};
  RateControlConfig hooked = vbr_config(90000.0);  // 3000 bits/frame
  hooked.estimator.estimate = &test_linear_estimator;
  hooked.estimator.context = &linear;
  CHECK(controller.configure(hooked, &error));
  CHECK(controller.begin_frame(complexity, decision, &error));
  CHECK(decision.base_q_idx == 57);
  CHECK(decision.estimated_bits <= 3000.0);
  CHECK(decision.target_reachable);
  double previous_bits = 0.0;
  CHECK(estimate_frame_bits(hooked.estimator, complexity,
                            decision.base_q_idx - 1, previous_bits, &error));
  CHECK(previous_bits > 3000.0);
  CHECK(controller.end_frame(3000));

  hooked.estimator.estimate = &test_nan_estimator;
  CHECK(controller.configure(hooked, &error));
  CHECK(!controller.begin_frame(complexity, decision, &error));
  CHECK(!controller.frame_open());

  // Determinism: identical inputs produce identical decisions and state.
  RateControlConfig deterministic = vbr_config(400000.0, 12);
  RateController a;
  RateController b;
  CHECK(a.configure(deterministic, &error));
  CHECK(b.configure(deterministic, &error));
  for (int frame = 0; frame < 10; ++frame) {
    const FrameComplexity sample =
        synthetic(160, 90, 5.0 + static_cast<double>(frame) * 3.0,
                  100.0 + static_cast<double>(frame) * 150.0,
                  0.25);
    QpDecision da;
    QpDecision db;
    CHECK(a.begin_frame(sample, da, &error));
    CHECK(b.begin_frame(sample, db, &error));
    CHECK(decision_equal(da, db));
    const std::int64_t actual = 4000 + frame * 700;
    CHECK(a.end_frame(actual, &error));
    CHECK(b.end_frame(actual, &error));
    CHECK(near(a.buffer_bits(), b.buffer_bits()));
    CHECK(near(a.bits_ema(), b.bits_ema()));
  }
  CHECK(a.frame_count() == b.frame_count());

  // Invalid VBR configs.
  CHECK(!rate_control_config_valid(vbr_config(0.0)));
  RateControlConfig invalid = vbr_config(300000.0);
  invalid.vbr.frame_rate_num = 0;
  CHECK(!rate_control_config_valid(invalid));
  invalid = vbr_config(300000.0);
  invalid.vbr.min_qindex = 200;
  invalid.vbr.max_qindex = 100;
  CHECK(!rate_control_config_valid(invalid));
  invalid = vbr_config(300000.0);
  invalid.vbr.max_qindex_step = 0;
  CHECK(!rate_control_config_valid(invalid));
  invalid = vbr_config(300000.0);
  invalid.vbr.bits_ema_weight = 1.5;
  CHECK(!rate_control_config_valid(invalid));

  error.clear();
  CHECK(!controller.end_frame(-5, &error));
  CHECK(!error.empty());
}

void test_segmentation() {
  std::string error;
  CHECK(segmentation_params_valid(disabled_segmentation()));

  SegmentationParams params = disabled_segmentation();
  params.update_map = true;
  CHECK(!segmentation_params_valid(params));

  params = disabled_segmentation();
  params.segments[0].enabled[kSegLvlAltQ] = true;
  CHECK(!segmentation_params_valid(params));

  params = disabled_segmentation();
  CHECK(set_segment_alt_q(params, 3, -12, &error));
  CHECK(params.enabled);
  CHECK(params.segments[3].enabled[kSegLvlAltQ]);
  CHECK(params.segments[3].data[kSegLvlAltQ] == -12);
  CHECK(segmentation_params_valid(params));

  CHECK(!set_segment_alt_q(params, -1, 0, &error));
  CHECK(!set_segment_alt_q(params, kMaxSegments, 0, &error));
  CHECK(!set_segment_alt_q(params, 0, kSegAltQLimit + 1, &error));

  SegmentationParams bad = disabled_segmentation();
  bad.enabled = true;
  bad.temporal_update = true;
  CHECK(!segmentation_params_valid(bad));

  bad = disabled_segmentation();
  bad.enabled = true;
  bad.segments[1].enabled[kSegLvlSkip] = true;
  bad.segments[1].data[kSegLvlSkip] = 1;
  CHECK(!segmentation_params_valid(bad));

  bad = disabled_segmentation();
  bad.enabled = true;
  bad.segments[1].enabled[kSegLvlAltQ] = true;
  bad.segments[1].data[kSegLvlAltQ] = kSegAltQLimit + 1;
  CHECK(!segmentation_params_valid(bad));

  QpContext context;
  context.base_q_idx = 90;
  context.current_q_index = 90;
  context.segmentation = disabled_segmentation();
  CHECK(set_segment_alt_q(context.segmentation, 0, -10, &error));
  CHECK(qp_context_valid(context));

  int qindex = -1;
  CHECK(effective_qindex(context, 0, false, qindex, &error));
  CHECK(qindex == 80);
  CHECK(effective_qindex(context, 1, false, qindex, &error));
  CHECK(qindex == 90);
  CHECK(effective_qindex(context, 0, true, qindex, &error));
  CHECK(qindex == 80);

  // Clipping at the bottom of the qindex domain.
  context.base_q_idx = 5;
  context.current_q_index = 5;
  context.segmentation.segments[0].data[kSegLvlAltQ] = -20;
  CHECK(effective_qindex(context, 0, false, qindex, &error));
  CHECK(qindex == 0);

  // Delta-Q uses CurrentQIndex, not base_q_idx, when ALT_Q is active.
  QpContext delta_context;
  delta_context.base_q_idx = 90;
  delta_context.current_q_index = 70;
  delta_context.delta_q.present = true;
  delta_context.delta_q.res = 1;
  delta_context.segmentation = disabled_segmentation();
  CHECK(set_segment_alt_q(delta_context.segmentation, 0, 5, &error));
  CHECK(qp_context_valid(delta_context));
  CHECK(effective_qindex(delta_context, 0, false, qindex, &error));
  CHECK(qindex == 75);
  CHECK(effective_qindex(delta_context, 0, true, qindex, &error));
  CHECK(qindex == 95);

  // Without ALT_Q, delta-Q uses CurrentQIndex for normal blocks and base for
  // ignoreDeltaQ paths.
  CHECK(effective_qindex(delta_context, 1, false, qindex, &error));
  CHECK(qindex == 70);
  CHECK(effective_qindex(delta_context, 1, true, qindex, &error));
  CHECK(qindex == 90);

  CHECK(!effective_qindex(delta_context, -1, false, qindex, &error));
  CHECK(!effective_qindex(delta_context, kMaxSegments, false, qindex, &error));

  QpContext invalid = delta_context;
  invalid.base_q_idx = 0;
  CHECK(!qp_context_valid(invalid));
  invalid = delta_context;
  invalid.current_q_index = 0;
  CHECK(!qp_context_valid(invalid));
  invalid = delta_context;
  invalid.delta_q.res = 4;
  CHECK(!qp_context_valid(invalid));

  // Segmentation map on the MI grid.
  SegmentationMap map;
  map.mi_cols = 2;
  map.mi_rows = 2;
  map.segment_id = {0, 1, 2, 3};
  CHECK(segmentation_map_valid(map));
  map.segment_id[3] = static_cast<std::uint8_t>(kMaxSegments);
  CHECK(!segmentation_map_valid(map));
  map.segment_id = {0, 1};
  CHECK(!segmentation_map_valid(map));
  map.mi_cols = 0;
  CHECK(!segmentation_map_valid(map));
}

void test_delta_q() {
  std::string error;
  CHECK(delta_q_params_valid(DeltaQParams{}));
  DeltaQParams absent_res = DeltaQParams{false, 1};
  CHECK(!delta_q_params_valid(absent_res));
  DeltaQParams present = DeltaQParams{true, 3};
  CHECK(delta_q_params_valid(present));
  present.res = 4;
  CHECK(!delta_q_params_valid(present));

  DeltaQState state;
  CHECK(!delta_q_reset(0, state, &error));
  CHECK(delta_q_reset(90, state, &error));
  CHECK(state.current_q_index == 90);

  DeltaQParams params{true, 1};
  CHECK(delta_q_apply(params, 2, state, &error));
  CHECK(state.current_q_index == 94);
  CHECK(delta_q_apply(params, -3, state, &error));
  CHECK(state.current_q_index == 88);
  CHECK(delta_q_apply(params, 100, state, &error));
  CHECK(state.current_q_index == 255);
  CHECK(delta_q_apply(params, -100, state, &error));
  CHECK(state.current_q_index == 55);
  CHECK(delta_q_apply(params, -1000, state, &error));
  CHECK(state.current_q_index == 1);

  DeltaQState uninitialized;
  CHECK(!delta_q_apply(params, 1, uninitialized, &error));
  CHECK(!delta_q_apply(DeltaQParams{}, 1, state, &error));

  DeltaQPlanConfig plan_config;
  plan_config.base_q_idx = 90;
  plan_config.delta_q_res = 0;
  plan_config.cols = 3;
  plan_config.rows = 2;
  CHECK(delta_q_plan_config_valid(plan_config));
  CHECK(!delta_q_plan_config_valid(DeltaQPlanConfig{}));

  DeltaQPlanConfig invalid = plan_config;
  invalid.base_q_idx = 0;
  CHECK(!delta_q_plan_config_valid(invalid));
  invalid = plan_config;
  invalid.delta_q_res = 4;
  CHECK(!delta_q_plan_config_valid(invalid));
  invalid = plan_config;
  invalid.cols = 0;
  CHECK(!delta_q_plan_config_valid(invalid));

  const std::vector<int> targets = {100, 100, 80, 80, 120, 90};
  DeltaQPlan plan;
  CHECK(plan_delta_q(plan_config, targets, plan, &error));
  CHECK(plan.steps.size() == targets.size());
  CHECK(plan.steps[0].qindex_before == 90);
  CHECK(plan.steps[0].qindex_after == 100);
  CHECK(plan.final_qindex == 90);
  CHECK(plan.max_abs_reduced_delta == 40);
  for (std::size_t i = 0; i < plan.steps.size(); ++i) {
    CHECK(plan.steps[i].qindex_after == targets[i]);
    CHECK(plan.steps[i].sb_x == static_cast<int>(i % 3));
    CHECK(plan.steps[i].sb_y == static_cast<int>(i / 3));
  }

  // With delta_q_res = 1 the achievable grid is every 2 qindex steps.
  DeltaQPlanConfig quantized = plan_config;
  quantized.delta_q_res = 1;
  const std::vector<int> quantized_targets = {101, 101, 80, 80, 120, 90};
  DeltaQPlan quantized_plan;
  CHECK(plan_delta_q(quantized, quantized_targets, quantized_plan, &error));
  CHECK(quantized_plan.steps[0].qindex_before == 90);
  for (std::size_t i = 0; i < quantized_plan.steps.size(); ++i) {
    CHECK(std::abs(quantized_plan.steps[i].qindex_after -
                   quantized_targets[i]) <= 1);
    CHECK(quantized_plan.steps[i].qindex_after ==
          std::clamp(quantized_plan.steps[i].qindex_before +
                         quantized_plan.steps[i].reduced_delta * 2,
                     1, 255));
  }

  DeltaQPlan repeat;
  CHECK(plan_delta_q(plan_config, targets, repeat, &error));
  CHECK(repeat.steps.size() == plan.steps.size());
  for (std::size_t i = 0; i < plan.steps.size(); ++i) {
    CHECK(repeat.steps[i].reduced_delta == plan.steps[i].reduced_delta);
    CHECK(repeat.steps[i].qindex_after == plan.steps[i].qindex_after);
  }

  // Each tile restarts from base_q_idx.
  DeltaQPlan tile_two;
  CHECK(plan_delta_q(plan_config, targets, tile_two, &error));
  CHECK(tile_two.steps[0].qindex_before == plan_config.base_q_idx);

  error.clear();
  CHECK(!plan_delta_q(plan_config, std::vector<int>{1, 2, 3}, plan, &error));
  CHECK(!error.empty());
  error.clear();
  CHECK(!plan_delta_q(plan_config, std::vector<int>{1, 2, 3, 4, 5, 6, 7}, plan,
                      &error));
  CHECK(!error.empty());
  error.clear();
  CHECK(!plan_delta_q(plan_config, std::vector<int>{100, 100, 100, 100, 100, 0},
                      plan, &error));
  CHECK(!error.empty());
  error.clear();
  CHECK(!plan_delta_q(
      plan_config, std::vector<int>{100, 100, 100, 100, 100, 300}, plan, &error));
  CHECK(!error.empty());

  // A failed plan is cleared, never partially populated.
  CHECK(plan.steps.empty());
  CHECK(plan.final_qindex == 0);
}

}  // namespace

int main() {
  test_quantizer_and_slices();
  test_complexity();
  test_bitrate_model();
  test_cqp();
  test_crf();
  test_vbr();
  test_segmentation();
  test_delta_q();

  if (g_failures != 0) {
    std::cerr << "rc_test: " << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "rc_test: PASS (" << g_checks << " checks)\n";
  return 0;
}
