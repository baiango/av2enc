#include "rdo.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check_impl(bool condition, const char* expression, int line) {
  if (!condition) {
    ++g_failures;
    std::cerr << "line " << line << ": CHECK failed: " << expression << '\n';
  }
}

#define CHECK(expression) check_impl((expression), #expression, __LINE__)

bool near(double a, double b, double epsilon = 1e-9) {
  return std::fabs(a - b) <= epsilon;
}

void test_lagrangian_cost() {
  using namespace av2enc::rdo;
  CHECK(near(lagrangian_cost(100.0, 10.0, 2.0), 120.0));
  CHECK(near(lagrangian_cost(100.0, 10.0, 0.0), 100.0));

  CandidateCost candidate = make_intra_candidate(7, 25.0, 5.0);
  CHECK(near(candidate_cost(candidate, 3.0), 40.0));
}

void test_distortion_metrics() {
  using namespace av2enc::rdo;

  const std::vector<std::uint8_t> zero(16, 0);
  const std::vector<std::uint8_t> ones(16, 1);

  CHECK(ssd_u8(zero, ones, 4, 4, 4) == 16);
  CHECK(sad_u8(zero, ones, 4, 4, 4) == 16);
  CHECK(satd_u8(zero, ones, 4, 4, 4) == 16);
  CHECK(near(compute_distortion(DistortionMetric::kSsd, zero, ones, 4, 4, 4),
             16.0));
  CHECK(near(compute_distortion(DistortionMetric::kSad, zero, ones, 4, 4, 4),
             16.0));
  CHECK(near(compute_distortion(DistortionMetric::kSatd, zero, ones, 4, 4, 4),
             16.0));

  std::vector<std::uint8_t> impulse(16, 0);
  impulse[5] = 3;
  CHECK(ssd_u8(zero, impulse, 4, 4, 4) == 9);
  CHECK(sad_u8(zero, impulse, 4, 4, 4) == 3);
  // A delta at (1,1) spreads to all 16 Hadamard coefficients: 16 * 3.
  CHECK(satd_u8(zero, impulse, 4, 4, 4) == 48);
  CHECK(ssd_u8(zero, impulse, 0, 4, 4) == 0);
  CHECK(ssd_u8(zero, impulse, 8, 8, 8) == 0);  // span too small

  // Stride test: a 3x2 visible region inside a 5-byte row stride. Padding must
  // not contribute to any metric.
  std::vector<std::uint8_t> a(10, 0);
  std::vector<std::uint8_t> b(10, 0);
  a[0] = 1;              // (0,0) difference 1
  a[5 + 2] = 5;          // (2,1) difference 3
  b[5 + 2] = 2;
  a[4] = 200;            // padding, must be ignored
  b[4] = 0;
  a[9] = 255;            // padding, must be ignored
  CHECK(ssd_u8(a, b, 3, 2, 5) == 10);
  CHECK(sad_u8(a, b, 3, 2, 5) == 4);
  CHECK(satd_u8(a, b, 3, 2, 5) ==
        static_cast<std::uint64_t>(compute_distortion(
            DistortionMetric::kSatd, a, b, 3, 2, 5)));

  std::vector<double> original = {1.0, 2.0};
  std::vector<double> reconstructed = {2.0, 4.0};
  CHECK(near(coefficient_sse(original, reconstructed), 5.0));
  std::vector<std::int32_t> original_i32 = {1, 2};
  std::vector<std::int32_t> reconstructed_i32 = {2, 4};
  CHECK(near(coefficient_sse_i32(original_i32, reconstructed_i32), 5.0));
}

void test_lambda_selection() {
  using namespace av2enc::rdo;

  LambdaParams params;
  CHECK(lambda_params_valid(params));
  CHECK(near(quant_step_from_qp(0, params), params.step_at_qp0));
  CHECK(near(quant_step_from_qp(-10, params), quant_step_from_qp(0, params)));
  CHECK(near(quant_step_from_qp(999, params), quant_step_from_qp(255, params)));

  double previous_step = 0.0;
  for (int qp = 0; qp <= 255; ++qp) {
    const double step = quant_step_from_qp(qp, params);
    CHECK(step > previous_step);
    previous_step = step;
  }

  const double step90 = quant_step_from_qp(90, params);
  const double lambda_sse = lambda_sse_from_qp(90, params);
  CHECK(near(lambda_sse, lambda_sse_from_quant_step(step90, params.sse_scale)));
  CHECK(near(lambda_sad_from_qp(90, params), std::sqrt(lambda_sse)));
  CHECK(near(lambda_satd_from_qp(90, params), std::sqrt(lambda_sse)));
  CHECK(near(lambda_from_qp(90, DistortionMetric::kSsd, params), lambda_sse));
  CHECK(near(lambda_from_qp(90, DistortionMetric::kSad, params),
             lambda_sad_from_qp(90, params)));
  CHECK(near(lambda_from_qp(90, DistortionMetric::kSatd, params),
             lambda_satd_from_qp(90, params)));

  LambdaParams bad = params;
  bad.qp_per_octave = 0.0;
  std::string error;
  CHECK(!lambda_params_valid(bad));
  CHECK(!validate_lambda_params(bad, &error));
  CHECK(!error.empty());
  CHECK(near(quant_step_from_qp(90, bad), 0.0));

  RdoConfig config = make_rdo_config(90, DistortionMetric::kSad);
  CHECK(rdo_config_valid(config));
  CHECK(near(config.lambda, lambda_sad_from_qp(90)));
  LambdaParams custom = params;
  custom.sse_scale = 0.25;
  const RdoConfig custom_config = make_rdo_config(
      90, DistortionMetric::kSsd, TieBreak::kRateThenDistortion, custom);
  CHECK(near(custom_config.lambda, lambda_sse_from_qp(90, custom)));
  CHECK(custom_config.tie_break == TieBreak::kRateThenDistortion);
  config.lambda = -1.0;
  error.clear();
  CHECK(!rdo_config_valid(config));
  CHECK(!validate_rdo_config(config, &error));
  CHECK(!error.empty());
}

void test_candidate_selection() {
  using namespace av2enc::rdo;

  RdoConfig config;
  config.lambda = 1.0;
  config.tie_break = TieBreak::kDistortionThenRate;

  std::vector<CandidateCost> candidates = {
      make_intra_candidate(1, 100.0, 10.0),
      make_intra_candidate(2, 90.0, 20.0),
      make_intra_candidate(3, 80.0, 41.0),
  };

  RdoDecision decision = select_best(candidates, config);
  CHECK(decision.found);
  CHECK(decision.candidate_count == 3);
  CHECK(decision.rejected_count == 0);
  CHECK(decision.best.candidate.id == 2);
  CHECK(near(decision.best.cost, 110.0));
  CHECK(decision.has_second);
  CHECK(decision.second.candidate.id == 1);
  CHECK(near(decision.margin(), 0.0));

  // Equal cost: prefer lower rate under the opposite tie-break policy.
  config.tie_break = TieBreak::kRateThenDistortion;
  decision = select_best(candidates, config);
  CHECK(decision.best.candidate.id == 1);
  CHECK(decision.second.candidate.id == 2);

  // High lambda: the cheapest-rate candidate wins.
  config.tie_break = TieBreak::kDistortionThenRate;
  config.lambda = 10.0;
  decision = select_best(candidates, config);
  CHECK(decision.best.candidate.id == 1);
  CHECK(near(decision.best.cost, 200.0));

  // Zero lambda: the lowest-distortion candidate wins.
  config.lambda = 0.0;
  decision = select_best(candidates, config);
  CHECK(decision.best.candidate.id == 3);
  CHECK(near(decision.best.cost, 80.0));

  // Input order must not change a non-index tie-break.
  config.lambda = 1.0;
  std::vector<CandidateCost> permuted = {candidates[2], candidates[0],
                                         candidates[1]};
  decision = select_best(permuted, config);
  CHECK(decision.best.candidate.id == 2);
  CHECK(decision.second.candidate.id == 1);

  // Kind is part of the deterministic tie-break order.
  std::vector<CandidateCost> kinds = {
      make_intra_candidate(0, 50.0, 50.0),
      make_partition_candidate(0, 50.0, 50.0),
  };
  decision = select_best(kinds, config);
  CHECK(decision.best.candidate.kind == CandidateKind::kPartition);

  // Invalid candidates are rejected without changing the best.
  CandidateSelector selector(config);
  std::string error;
  CHECK(!selector.add(make_intra_candidate(9, -1.0, 5.0), &error));
  CHECK(!error.empty());
  CHECK(selector.decision().rejected_count == 1);
  CHECK(!selector.decision().found);
  error.clear();
  CHECK(!selector.add(
      make_intra_candidate(9, 1.0,
                           std::numeric_limits<double>::quiet_NaN()),
      &error));
  CHECK(selector.decision().rejected_count == 2);
  error.clear();
  CHECK(selector.add(make_intra_candidate(4, 10.0, 2.0), &error));
  CHECK(selector.decision().found);
  CHECK(selector.decision().best.candidate.id == 4);

  std::vector<CandidateCost> empty;
  const RdoDecision none = select_best(empty, config);
  CHECK(!none.found);
  CHECK(none.candidate_count == 0);

  // Direct total-order checks.
  CandidateScore a;
  a.cost = 10.0;
  a.candidate.distortion = 5.0;
  a.candidate.rate_bits = 5.0;
  a.input_index = 1;
  CandidateScore b;
  b.cost = 10.0;
  b.candidate.distortion = 4.0;
  b.candidate.rate_bits = 6.0;
  b.input_index = 0;
  CHECK(candidate_better(b, a, TieBreak::kDistortionThenRate));
  CHECK(candidate_better(a, b, TieBreak::kRateThenDistortion));

  // Partition aggregation: child costs plus split signaling bits.
  std::vector<CandidateCost> children = {
      make_intra_candidate(0, 10.0, 2.0),
      make_transform_candidate(1, 20.0, 3.0),
  };
  const CandidateCost split =
      make_partition_split_candidate(1, children, 4.0);
  CHECK(split.kind == CandidateKind::kPartition);
  CHECK(split.id == 1);
  CHECK(near(split.distortion, 30.0));
  CHECK(near(split.rate_bits, 9.0));
}

struct TestRateContext {
  double zero = 0.0;
  double one = 0.0;
  double two = 0.0;
  double eob = 0.0;
};

double test_coefficient_rate(void* context,
                             std::int32_t /*scan_pos*/,
                             std::int32_t /*prev_abs_level*/,
                             std::int32_t abs_level,
                             std::int32_t /*level_context*/) {
  const auto* rate = static_cast<const TestRateContext*>(context);
  if (abs_level == 0) {
    return rate->zero;
  }
  if (abs_level == 1) {
    return rate->one;
  }
  return rate->two;
}

double test_eob_rate(void* context,
                     std::int32_t /*eob*/,
                     std::int32_t /*num_coefficients*/) {
  return static_cast<const TestRateContext*>(context)->eob;
}

double negative_rate(void* /*context*/,
                     std::int32_t /*scan_pos*/,
                     std::int32_t /*prev_abs_level*/,
                     std::int32_t /*abs_level*/,
                     std::int32_t /*level_context*/) {
  return -1.0;
}

double negative_eob_rate(void* /*context*/,
                         std::int32_t /*eob*/,
                         std::int32_t /*num_coefficients*/) {
  return -1.0;
}

av2enc::rdo::RdoqRateModel test_rate_model(TestRateContext& context) {
  av2enc::rdo::RdoqRateModel model;
  model.context = &context;
  model.coefficient_rate = &test_coefficient_rate;
  model.eob_rate = &test_eob_rate;
  return model;
}

void test_rdoq_bruteforce() {
  using namespace av2enc::rdo;

  // Four coefficients, two levels each: compare the trellis against exhaustive
  // enumeration of every 0/initial-level combination. This validates the DP,
  // the EOB termination path and the final cost recomputation together.
  std::vector<RdoqCoefficient> coefficients = {
      {5.0, 1, 0},
      {-7.0, -2, 0},
      {2.0, 1, 0},
      {9.0, 2, 0},
  };
  RdoqConfig config;
  config.quant_step = 3.0;
  config.lambda = 2.0;
  config.max_level_delta = 0;
  config.allow_level_decrease = false;
  config.allow_level_increase = false;

  TestRateContext context;
  context.zero = 0.3;
  context.one = 1.7;
  context.two = 3.1;
  context.eob = 0.9;
  const RdoqRateModel model = test_rate_model(context);

  RdoqResult result;
  std::string error;
  CHECK(rdoq_quantize(coefficients, config, model, result, &error));

  const std::size_t count = coefficients.size();
  double best_cost = std::numeric_limits<double>::infinity();
  const std::uint32_t combinations = 1u << static_cast<unsigned>(count);
  for (std::uint32_t mask = 0; mask < combinations; ++mask) {
    std::vector<std::int32_t> levels(count, 0);
    for (std::size_t i = 0; i < count; ++i) {
      if (((mask >> static_cast<unsigned>(i)) & 1u) != 0) {
        levels[i] = coefficients[i].initial_level;
      }
    }

    std::int32_t eob = 0;
    for (std::size_t i = 0; i < count; ++i) {
      if (levels[i] != 0) {
        eob = static_cast<std::int32_t>(i) + 1;
      }
    }

    double distortion = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      const double dequantized =
          static_cast<double>(levels[i]) * config.quant_step;
      const double difference = coefficients[i].original - dequantized;
      distortion += difference * difference;
    }

    double rate_bits = 0.0;
    std::int32_t previous_abs = 0;
    for (std::size_t i = 0; i < count; ++i) {
      if (static_cast<std::int32_t>(i) >= eob) {
        break;
      }
      const std::int32_t level = levels[i];
      const std::int32_t abs_level = level < 0 ? -level : level;
      rate_bits += test_coefficient_rate(&context,
                                         static_cast<std::int32_t>(i),
                                         previous_abs,
                                         abs_level,
                                         0);
      previous_abs = abs_level == 0 ? 0 : (abs_level == 1 ? 1 : 2);
    }
    rate_bits += test_eob_rate(&context, eob, static_cast<std::int32_t>(count));

    const double cost = distortion + config.lambda * rate_bits;
    if (cost < best_cost) {
      best_cost = cost;
    }
  }

  CHECK(near(result.cost, best_cost));

  // Independently recompute the returned path's cost.
  std::int32_t expected_eob = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (result.levels[i] != 0) {
      expected_eob = static_cast<std::int32_t>(i) + 1;
    }
  }
  CHECK(result.eob == expected_eob);
  CHECK(result.cost ==
        result.distortion + config.lambda * result.rate_bits);
}

void test_rdoq() {
  using namespace av2enc::rdo;

  std::vector<RdoqCoefficient> coefficients = {
      {8.0, 2, 0},
      {-4.0, -1, 0},
      {0.0, 0, 0},
      {12.0, 3, 0},
  };

  RdoqConfig config;
  config.quant_step = 4.0;
  config.lambda = 0.0;

  TestRateContext zero_context;
  RdoqRateModel zero_model = test_rate_model(zero_context);

  RdoqResult result;
  std::string error;
  CHECK(rdoq_quantize(coefficients, config, zero_model, result, &error));
  std::vector<std::int32_t> expected_levels = {2, -1, 0, 3};
  CHECK(result.levels == expected_levels);
  CHECK(result.eob == 4);
  CHECK(near(result.distortion, 0.0));
  CHECK(near(result.rate_bits, 0.0));
  CHECK(near(result.cost, 0.0));

  // Regression: the terminal EOB rate is a rate, so the trellis must multiply
  // it by lambda. With lambda < 1 an unscaled EOB cost would wrongly prefer
  // terminating the block.
  TestRateContext eob_context;
  eob_context.zero = 1.0;
  eob_context.one = 1.0;
  eob_context.two = 1.0;
  eob_context.eob = 10.0;
  const RdoqRateModel eob_model = test_rate_model(eob_context);
  std::vector<RdoqCoefficient> single_coefficient = {{1.0, 1, 0}};
  RdoqConfig eob_config;
  eob_config.quant_step = 1.0;
  eob_config.lambda = 0.1;
  CHECK(rdoq_quantize(single_coefficient, eob_config, eob_model, result,
                      &error));
  CHECK(result.levels[0] == 1);
  CHECK(result.eob == 1);
  CHECK(near(result.distortion, 0.0));
  CHECK(near(result.rate_bits, 11.0));
  CHECK(near(result.cost, 1.1));

  // A huge lambda drives every coefficient to zero.
  config.lambda = 1.0e9;
  const RdoqRateModel default_model = default_rdoq_rate_model();
  CHECK(rdoq_quantize(coefficients, config, default_model, result, &error));
  std::vector<std::int32_t> zero_levels = {0, 0, 0, 0};
  CHECK(result.levels == zero_levels);
  CHECK(result.eob == 0);
  CHECK(near(result.distortion, 224.0));
  CHECK(result.rate_bits > 0.0);

  // Equal distortion: the cheaper level wins deterministically.
  TestRateContext tie_context;
  tie_context.zero = 0.0;
  tie_context.one = 1.0;
  tie_context.two = 2.0;
  tie_context.eob = 0.0;
  const RdoqRateModel tie_model = test_rate_model(tie_context);

  std::vector<RdoqCoefficient> tie_coefficients = {{6.0, 2, 0}};
  config.lambda = 0.0;
  config.quant_step = 4.0;
  CHECK(rdoq_quantize(tie_coefficients, config, tie_model, result, &error));
  CHECK(result.levels.size() == 1);
  CHECK(result.levels[0] == 1);
  CHECK(result.eob == 1);
  CHECK(near(result.distortion, 4.0));
  CHECK(near(result.rate_bits, 1.0));
  CHECK(near(result.cost, 4.0));

  // allow_level_increase controls whether |q|+1 is searched.
  std::vector<RdoqCoefficient> increase_coefficients = {{7.0, 1, 0}};
  config.allow_level_increase = true;
  config.max_level_delta = 1;
  CHECK(rdoq_quantize(increase_coefficients, config, zero_model, result,
                      &error));
  CHECK(result.levels[0] == 2);
  CHECK(near(result.distortion, 1.0));

  config.allow_level_increase = false;
  CHECK(rdoq_quantize(increase_coefficients, config, zero_model, result,
                      &error));
  CHECK(result.levels[0] == 1);
  CHECK(near(result.distortion, 9.0));

  // allow_zero_to_one controls whether an initial zero can become +/-1.
  std::vector<RdoqCoefficient> zero_to_one = {{3.0, 0, 0}};
  config = RdoqConfig{};
  config.quant_step = 4.0;
  config.lambda = 0.0;
  config.allow_zero_to_one = true;
  CHECK(rdoq_quantize(zero_to_one, config, zero_model, result, &error));
  CHECK(result.levels[0] == 1);
  CHECK(near(result.distortion, 1.0));

  config.allow_zero_to_one = false;
  CHECK(rdoq_quantize(zero_to_one, config, zero_model, result, &error));
  CHECK(result.levels[0] == 0);
  CHECK(near(result.distortion, 9.0));

  // EOB terminates after the last non-zero coefficient.
  std::vector<RdoqCoefficient> trailing_zeros = {
      {8.0, 2, 0},
      {0.0, 0, 0},
      {0.0, 0, 0},
      {0.0, 0, 0},
  };
  config = RdoqConfig{};
  config.quant_step = 4.0;
  config.lambda = 0.0;
  CHECK(rdoq_quantize(trailing_zeros, config, zero_model, result, &error));
  std::vector<std::int32_t> trailing_levels = {2, 0, 0, 0};
  CHECK(result.levels == trailing_levels);
  CHECK(result.eob == 1);
  CHECK(near(result.distortion, 0.0));

  // Determinism: identical inputs give bit-identical results.
  config.lambda = 1.0;
  RdoqResult first;
  CHECK(rdoq_quantize(coefficients, config, default_model, first, &error));
  for (int repeat = 0; repeat < 8; ++repeat) {
    RdoqResult again;
    CHECK(rdoq_quantize(coefficients, config, default_model, again, &error));
    CHECK(again.levels == first.levels);
    CHECK(again.eob == first.eob);
    CHECK(again.distortion == first.distortion);
    CHECK(again.rate_bits == first.rate_bits);
    CHECK(again.cost == first.cost);
  }

  // Invalid configs and rate hooks fail with an error and reset the result.
  config.quant_step = 0.0;
  error.clear();
  CHECK(!rdoq_quantize(coefficients, config, default_model, result, &error));
  CHECK(!error.empty());
  CHECK(result.levels.empty());

  config.quant_step = 4.0;
  config.max_coefficients = 2;
  error.clear();
  CHECK(!rdoq_quantize(coefficients, config, default_model, result, &error));
  CHECK(!error.empty());

  config.max_coefficients = 4096;
  config.max_level_delta = 2048;
  error.clear();
  CHECK(!rdoq_config_valid(config));
  CHECK(!rdoq_quantize(coefficients, config, default_model, result, &error));
  CHECK(!error.empty());
  config.max_level_delta = 1;

  RdoqRateModel negative_model = default_model;
  negative_model.coefficient_rate = &negative_rate;
  error.clear();
  CHECK(!rdoq_quantize(coefficients, config, negative_model, result, &error));
  CHECK(!error.empty());

  RdoqRateModel negative_eob_model = default_model;
  negative_eob_model.eob_rate = &negative_eob_rate;
  error.clear();
  CHECK(!rdoq_quantize(coefficients, config, negative_eob_model, result,
                       &error));
  CHECK(!error.empty());

  // Empty block is a valid all-zero result.
  std::vector<RdoqCoefficient> empty;
  error.clear();
  CHECK(rdoq_quantize(empty, config, default_model, result, &error));
  CHECK(result.levels.empty());
  CHECK(result.eob == 0);
  CHECK(near(result.cost, 0.0));
}

}  // namespace

int main() {
  test_lagrangian_cost();
  test_distortion_metrics();
  test_lambda_selection();
  test_candidate_selection();
  test_rdoq_bruteforce();
  test_rdoq();

  if (g_failures != 0) {
    std::cerr << "rdo_test: " << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "rdo_test: PASS\n";
  return 0;
}
