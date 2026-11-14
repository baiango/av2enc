// Standalone tests for the full coefficient RDOQ trellis.
//
// The exhaustive checks enumerate every canonical path for small blocks and
// compare the DP optimum, the reported cost and the reconstructed levels.  The
// forward reference DP in this file is an independent implementation (forward
// over positions, explicit EOB enumeration) used on larger random blocks.

#include "rdoq.h"
#include "rdoq_av2_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace av2enc::rdoq;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

void check_close(double actual, double expected, double tolerance,
                 const char* what) {
  const double scale = std::max(1.0, std::max(std::fabs(actual),
                                              std::fabs(expected)));
  check(std::fabs(actual - expected) <= tolerance * scale, what);
}

struct TestRateContext {
  double position_weight = 0.01;
  double zero_prev_bits = 0.9;
};

double test_coefficient_rate(void* context,
                             std::int32_t scan_pos,
                             std::int32_t prev_abs_state,
                             std::int32_t level,
                             std::int32_t level_context,
                             std::int32_t is_eob) {
  const TestRateContext* rate = static_cast<const TestRateContext*>(context);
  const std::int32_t magnitude = std::abs(level);
  double bits = 0.0;
  if (magnitude == 0) {
    bits = prev_abs_state == 0 ? 0.4 : 1.1;
  } else {
    bits = 1.5 + 0.7 * std::log2(static_cast<double>(magnitude) + 1.0);
    if (prev_abs_state == 0) bits += rate->zero_prev_bits;
    if (is_eob != 0) bits += 0.3;
    // Deliberate non-monotone dip: the trellis must use the distortion stop
    // rule, not assume a monotone rate, to find the level-2/3 optimum.
    if (magnitude == 2) bits -= 0.7;
    if (magnitude == 3) bits -= 0.5;
  }
  bits += rate->position_weight * static_cast<double>(scan_pos);
  bits += 0.05 * static_cast<double>(level_context & 3);
  return bits;
}

double test_eob_rate(void* context,
                     std::int32_t eob,
                     std::int32_t num_coefficients) {
  (void)context;
  (void)num_coefficients;
  return 1.2 + 0.35 * std::log2(static_cast<double>(eob) + 1.0);
}

RateModel test_rate_model(TestRateContext* context) {
  RateModel model;
  model.context = context;
  model.coefficient_rate = &test_coefficient_rate;
  model.eob_rate = &test_eob_rate;
  return model;
}

std::vector<Coefficient> make_random_block(std::size_t count, double max_abs,
                                           std::uint32_t seed,
                                           double quant_step) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> value(-max_abs, max_abs);
  std::vector<Coefficient> block(count);
  for (std::size_t i = 0; i < count; ++i) {
    block[i].original = value(rng);
    block[i].level_context = static_cast<std::int32_t>(i % 4);
    const double ratio = std::fabs(block[i].original) / quant_step;
    block[i].scalar_level = static_cast<std::int32_t>(std::lround(ratio)) *
                            (block[i].original < 0.0 ? -1 : 1);
  }
  return block;
}

// Cost of one explicit canonical path.  Levels after `eob` must be zero.
double evaluate_path(std::span<const Coefficient> coefficients,
                     std::span<const std::int32_t> levels,
                     std::int32_t eob,
                     const Config& config,
                     const RateModel& rate_model) {
  const CoefficientRateFn coefficient_rate =
      rate_model.coefficient_rate != nullptr ? rate_model.coefficient_rate
                                             : &default_coefficient_rate;
  const EobRateFn eob_rate =
      rate_model.eob_rate != nullptr ? rate_model.eob_rate
                                     : &default_eob_rate;
  double distortion = 0.0;
  for (std::size_t i = 0; i < coefficients.size(); ++i) {
    const double reconstructed =
        static_cast<double>(levels[i]) * config.quant_step;
    const double difference = coefficients[i].original - reconstructed;
    distortion += difference * difference;
  }
  double rate = eob_rate(rate_model.context, eob,
                         static_cast<std::int32_t>(coefficients.size()));
  std::int32_t previous = 0;
  for (std::int32_t i = 0; i < eob; ++i) {
    const std::int32_t level = levels[static_cast<std::size_t>(i)];
    const std::int32_t is_eob = (i == eob - 1) ? 1 : 0;
    rate += coefficient_rate(
        rate_model.context, i, previous, level,
        coefficients[static_cast<std::size_t>(i)].level_context, is_eob);
    previous = std::min(std::abs(level), config.prev_level_cap);
  }
  return distortion + config.lambda * rate;
}

// Exhaustive enumeration of canonical paths with magnitude <= level_max for
// the first `active` coefficients (the rest are forced zero).
std::vector<std::int32_t> brute_force_best(
    std::span<const Coefficient> coefficients,
    const Config& config,
    const RateModel& rate_model,
    std::int32_t level_max,
    std::int32_t active,
    double* best_cost,
    std::string* error) {
  const std::int32_t count = static_cast<std::int32_t>(coefficients.size());
  if (active < 0) active = 0;
  if (active > count) active = count;
  std::vector<std::int32_t> current(coefficients.size(), 0);
  std::vector<std::int32_t> best(coefficients.size(), 0);
  double best_value = std::numeric_limits<double>::infinity();
  std::int32_t best_eob = 0;

  const auto consider = [&](std::int32_t eob) {
    const double cost =
        evaluate_path(coefficients, current, eob, config, rate_model);
    if (cost < best_value) {
      best_value = cost;
      best = current;
      best_eob = eob;
    }
  };

  // eob == 0
  current.assign(coefficients.size(), 0);
  consider(0);

  // For every eob, enumerate the magnitudes of positions [0, eob-2] over
  // 0..level_max, and the last position over 1..level_max.  The last position
  // is non-zero by definition of a canonical EOB.
  const std::int32_t radix = level_max + 1;
  for (std::int32_t eob = 1; eob <= active; ++eob) {
    std::int64_t total = 1;
    for (std::int32_t i = 0; i < eob - 1; ++i) {
      total *= radix;
    }
    for (std::int64_t code = 0; code < total; ++code) {
      std::fill(current.begin(), current.end(), 0);
      std::int64_t value = code;
      for (std::int32_t i = 0; i < eob - 1; ++i) {
        const std::int32_t magnitude =
            static_cast<std::int32_t>(value % radix);
        value /= radix;
        if (magnitude != 0) {
          const double original =
              coefficients[static_cast<std::size_t>(i)].original;
          const std::int32_t sign = original < 0.0 ? -1 : 1;
          current[static_cast<std::size_t>(i)] = sign * magnitude;
        }
      }
      for (std::int32_t last = 1; last <= level_max; ++last) {
        const std::size_t index = static_cast<std::size_t>(eob - 1);
        const double original = coefficients[index].original;
        const std::int32_t sign = original < 0.0 ? -1 : 1;
        current[index] = sign * last;
        consider(eob);
      }
    }
  }

  if (error != nullptr) {
    error->clear();
  }
  if (best_cost != nullptr) *best_cost = best_value;
  (void)best_eob;
  return best;
}

// Independent forward DP: f[i][state] is the best cost with coefficients
// [0, i) coded (not terminated).  At each position it can terminate with a
// non-zero level or continue.  No reverse formulation is reused.
double forward_dp_cost(std::span<const Coefficient> coefficients,
                       const Config& config,
                       const RateModel& rate_model,
                       std::span<const std::int32_t> bounds) {
  const std::int32_t count = static_cast<std::int32_t>(coefficients.size());
  if (count == 0) {
    return config.lambda * rate_model.eob_rate(rate_model.context, 0, 0);
  }
  const std::int32_t states = config.prev_level_cap + 1;
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> current(static_cast<std::size_t>(states), infinity);
  std::vector<double> next(static_cast<std::size_t>(states), infinity);
  current[0] = 0.0;

  std::vector<double> suffix_zero(static_cast<std::size_t>(count) + 1, 0.0);
  for (std::int32_t i = count; i-- > 0;) {
    const double original = coefficients[static_cast<std::size_t>(i)].original;
    suffix_zero[static_cast<std::size_t>(i)] =
        suffix_zero[static_cast<std::size_t>(i + 1)] + original * original;
  }

  // All-zero block (distortion included).
  double best = suffix_zero[0] +
                config.lambda *
                    rate_model.eob_rate(rate_model.context, 0, count);

  for (std::int32_t i = 0; i < count; ++i) {
    std::fill(next.begin(), next.end(), infinity);
    const Coefficient& coefficient = coefficients[static_cast<std::size_t>(i)];
    const double original = coefficient.original;
    const std::int32_t sign = original < 0.0 ? -1 : 1;
    const std::int32_t bound = bounds[static_cast<std::size_t>(i)];
    for (std::int32_t prev = 0; prev < states; ++prev) {
      if (!std::isfinite(current[static_cast<std::size_t>(prev)])) continue;
      const double base = current[static_cast<std::size_t>(prev)];
      for (std::int32_t magnitude = 0; magnitude <= bound; ++magnitude) {
        const std::int32_t level = magnitude == 0 ? 0 : sign * magnitude;
        const double reconstructed =
            static_cast<double>(level) * config.quant_step;
        const double difference = original - reconstructed;
        const double distortion = difference * difference;
        const std::int32_t next_state = std::min(magnitude, config.prev_level_cap);

        const double normal = rate_model.coefficient_rate(
            rate_model.context, i, prev, level, coefficient.level_context, 0);
        const double normal_cost =
            base + distortion + config.lambda * normal;
        if (normal_cost < next[static_cast<std::size_t>(next_state)]) {
          next[static_cast<std::size_t>(next_state)] = normal_cost;
        }

        if (magnitude != 0) {
          const double eob_coeff = rate_model.coefficient_rate(
              rate_model.context, i, prev, level, coefficient.level_context, 1);
          const double eob_payload =
              rate_model.eob_rate(rate_model.context, i + 1, count);
          const double terminal = base + distortion +
                                  config.lambda * (eob_coeff + eob_payload) +
                                  suffix_zero[static_cast<std::size_t>(i + 1)];
          if (terminal < best) best = terminal;
        }
      }
    }
    current.swap(next);
  }
  return best;
}

void test_config_validation() {
  Config config;
  check(rdoq_config_valid(config), "default config is valid");
  config.lambda = -1.0;
  check(!rdoq_config_valid(config), "negative lambda rejected");
  config.lambda = 1.0;
  config.quant_step = 0.0;
  check(!rdoq_config_valid(config), "zero step rejected");
  config.quant_step = 1.0;
  config.prev_level_cap = 64;
  check(!rdoq_config_valid(config), "state cap > 63 rejected");
  config.prev_level_cap = 2;
  config.level_cap = -1;
  check(!rdoq_config_valid(config), "negative level cap rejected");
}

void test_lambda_zero_recovers_scalar() {
  TestRateContext context;
  RateModel rate_model = test_rate_model(&context);
  const std::size_t count = 37;
  std::vector<Coefficient> block =
      make_random_block(count, 300.0, 1234, 8.0);
  Config config;
  config.lambda = 0.0;
  config.quant_step = 8.0;
  config.prev_level_cap = 2;
  RdoqResult result;
  std::string error;
  check(rdoq_quantize(block, config, rate_model, result, &error),
        "lambda=0 quantisation succeeds");
  std::vector<std::int32_t> scalar(count);
  std::int32_t scalar_eob = 0;
  check(estimate_scalar_levels(block, config.quant_step, scalar, &scalar_eob,
                               &error),
        "scalar estimate succeeds");
  check(result.eob == scalar_eob, "lambda=0 matches scalar EOB");
  bool levels_match = true;
  for (std::size_t i = 0; i < count; ++i) {
    if (result.levels[i] != scalar[i]) levels_match = false;
  }
  check(levels_match, "lambda=0 matches scalar levels");
  check_close(result.cost, result.distortion, 1.0e-12,
              "lambda=0 cost is distortion");
}

void test_determinism() {
  TestRateContext context;
  RateModel rate_model = test_rate_model(&context);
  std::vector<Coefficient> block =
      make_random_block(128, 900.0, 99, 7.5);
  Config config;
  config.lambda = 30.0;
  config.quant_step = 7.5;
  RdoqResult first;
  std::string error;
  check(rdoq_quantize(block, config, rate_model, first, &error),
        "determinism: first run succeeds");
  for (int trial = 0; trial < 5; ++trial) {
    RdoqResult again;
    check(rdoq_quantize(block, config, rate_model, again, &error),
          "determinism: repeated run succeeds");
    check(again.eob == first.eob, "determinism: EOB stable");
    check(again.levels == first.levels, "determinism: levels stable");
    check(again.cost == first.cost, "determinism: cost stable");
  }
}

void test_exhaustive(std::size_t count,
                     std::int32_t level_max,
                     std::int32_t active,
                     double max_abs,
                     std::uint32_t seed,
                     double lambda,
                     const char* label) {
  TestRateContext context;
  RateModel rate_model = test_rate_model(&context);
  std::vector<Coefficient> block =
      make_random_block(count, max_abs, seed, 8.0);
  for (std::size_t i = static_cast<std::size_t>(active); i < count; ++i) {
    block[i].original = 0.0;
    block[i].scalar_level = 0;
  }
  Config config;
  config.lambda = lambda;
  config.quant_step = 8.0;
  // Search exactly 0..level_max so the comparison is apples-to-apples.
  config.prev_level_cap = level_max;
  config.level_cap = level_max;
  // The brute force only enumerates the active prefix.  Do not let the core
  // extend a non-zero level into the zero-filled tail.
  config.allow_zero_to_one = (active == static_cast<std::int32_t>(count));

  RdoqResult result;
  std::string error;
  check(rdoq_quantize(block, config, rate_model, result, &error),
        label);
  double brute_cost = 0.0;
  std::vector<std::int32_t> brute =
      brute_force_best(block, config, rate_model, level_max, active,
                       &brute_cost, &error);
  check_close(result.cost, brute_cost, 1.0e-9, label);
  const double recomputed =
      evaluate_path(block, result.levels, result.eob, config, rate_model);
  check_close(result.cost, recomputed, 1.0e-9,
              "reported cost matches the chosen path");
  const double brute_path_cost =
      evaluate_path(block, brute, [&]() {
        std::int32_t e = 0;
        for (std::size_t i = 0; i < brute.size(); ++i) {
          if (brute[i] != 0) e = static_cast<std::int32_t>(i) + 1;
        }
        return e;
      }(), config, rate_model);
  check_close(brute_path_cost, brute_cost, 1.0e-9,
              "brute-force path cost is consistent");
}

void test_random_larger_blocks() {
  TestRateContext context;
  RateModel rate_model = test_rate_model(&context);
  std::mt19937 rng(2026);
  const std::size_t sizes[] = {32, 64, 128, 1024};
  for (std::size_t size : sizes) {
    for (int trial = 0; trial < 4; ++trial) {
      std::vector<Coefficient> block =
          make_random_block(size, 800.0, rng(), 6.0 + 0.5 * trial);
      Config config;
      config.lambda = 10.0 + 20.0 * trial;
      config.quant_step = 6.0 + 0.5 * trial;
      config.prev_level_cap = 2;
      config.allow_zero_to_one = true;
      RdoqResult result;
      std::string error;
      check(rdoq_quantize(block, config, rate_model, result, &error),
            "random larger block quantises");

      // Independent forward DP over exactly the same per-coefficient range
      // as the CPU trellis (the safe stop rule may search less, but must not
      // miss a better level).
      std::vector<std::int32_t> bounds(block.size(), 0);
      for (std::size_t i = 0; i < block.size(); ++i) {
        bounds[i] = av2enc::rdoq::rdoq_level_bound(block[i], config);
        check(bounds[i] >= 0, "random block level bound is valid");
      }
      const double forward =
          forward_dp_cost(block, config, rate_model, bounds);
      check_close(result.cost, forward, 1.0e-7,
                  "reverse DP matches forward DP");
      std::vector<std::int32_t> scalar(size);
      std::int32_t scalar_eob = 0;
      std::string scalar_error;
      check(estimate_scalar_levels(block, config.quant_step, scalar,
                                   &scalar_eob, &scalar_error),
            "random larger block scalar estimate");
      const double scalar_cost =
          evaluate_path(block, scalar, scalar_eob, config, rate_model);
      check(result.cost <= scalar_cost + 1.0e-7,
            "DP is no worse than scalar");
      double zero_distortion = 0.0;
      for (const Coefficient& coefficient : block) {
        zero_distortion += coefficient.original * coefficient.original;
      }
      const double zero_cost =
          zero_distortion +
          config.lambda * rate_model.eob_rate(
                              rate_model.context, 0,
                              static_cast<std::int32_t>(size));
      check(result.cost <= zero_cost + 1.0e-7,
            "DP is no worse than all-zero");
    }
  }
}

void test_batch() {
  TestRateContext context;
  RateModel rate_model = test_rate_model(&context);
  const std::int32_t sizes[] = {16, 37, 64};
  const double lambdas[] = {5.0, 25.0, 100.0};
  std::vector<double> original;
  std::vector<std::int32_t> level_context;
  std::vector<BatchBlock> blocks;
  std::int32_t offset = 0;
  for (int b = 0; b < 3; ++b) {
    std::vector<Coefficient> block =
        make_random_block(static_cast<std::size_t>(sizes[b]), 400.0,
                          static_cast<std::uint32_t>(700 + b), 8.0);
    BatchBlock entry;
    entry.coefficient_offset = offset;
    entry.num_coefficients = sizes[b];
    entry.level_offset = offset;
    entry.quant_step = 8.0;
    entry.lambda = lambdas[b];
    entry.prev_level_cap = 2;
    blocks.push_back(entry);
    for (const Coefficient& coefficient : block) {
      original.push_back(coefficient.original);
      level_context.push_back(coefficient.level_context);
    }
    offset += sizes[b];
  }
  std::vector<std::int32_t> levels(original.size(), 0);
  std::vector<std::int32_t> eob(blocks.size(), 0);
  std::vector<double> distortion(blocks.size(), 0.0);
  std::vector<double> rates(blocks.size(), 0.0);
  std::vector<double> costs(blocks.size(), 0.0);
  std::vector<std::int32_t> status(blocks.size(), -1);

  BatchInput input;
  input.original = original;
  input.level_context = level_context;
  input.blocks = blocks;
  BatchOutput output;
  output.levels = levels;
  output.eob = eob;
  output.distortion = distortion;
  output.rate_bits = rates;
  output.cost = costs;
  output.status = status;
  BatchConfig config;
  std::string error;
  check(rdoq_quantize_batch(input, config, rate_model, output, &error),
        "batch quantisation succeeds");
  for (int b = 0; b < 3; ++b) {
    const std::size_t block_offset = static_cast<std::size_t>(
        blocks[static_cast<std::size_t>(b)].coefficient_offset);
    std::vector<Coefficient> block;
    block.reserve(static_cast<std::size_t>(sizes[b]));
    for (std::int32_t i = 0; i < sizes[b]; ++i) {
      Coefficient coefficient;
      coefficient.original =
          original[block_offset + static_cast<std::size_t>(i)];
      coefficient.level_context =
          level_context[block_offset + static_cast<std::size_t>(i)];
      block.push_back(coefficient);
    }
    Config single_config;
    single_config.lambda = lambdas[b];
    single_config.quant_step = 8.0;
    single_config.prev_level_cap = 2;
    RdoqResult single;
    std::string single_error;
    check(rdoq_quantize(block, single_config, rate_model, single,
                        &single_error),
          "batch: scalar path succeeds");
    check(status[static_cast<std::size_t>(b)] == 0, "batch status is ok");
    check(eob[static_cast<std::size_t>(b)] == single.eob,
          "batch EOB matches scalar path");
    check(costs[static_cast<std::size_t>(b)] == single.cost,
          "batch cost matches scalar path");
    const std::size_t level_offset = static_cast<std::size_t>(
        blocks[static_cast<std::size_t>(b)].level_offset);
    for (std::int32_t i = 0; i < sizes[b]; ++i) {
      check(levels[level_offset + static_cast<std::size_t>(i)] ==
                single.levels[static_cast<std::size_t>(i)],
            "batch levels match scalar path");
    }
  }
}

void test_invalid_inputs() {
  TestRateContext context;
  RateModel rate_model = test_rate_model(&context);
  std::vector<Coefficient> block(4);
  Config config;
  config.quant_step = 4.0;
  RdoqResult result;
  std::string error;
  block[2].original = std::numeric_limits<double>::quiet_NaN();
  check(!rdoq_quantize(block, config, rate_model, result, &error),
        "non-finite coefficient rejected");
  check(result.levels.empty(), "failed call leaves the result empty");
  block[2].original = 0.0;

  config.max_coefficients = 2;
  check(!rdoq_quantize(block, config, rate_model, result, &error),
        "oversized block rejected");

  config.max_coefficients = 4096;
  RateModel bad = rate_model;
  bad.coefficient_rate = [](void*, std::int32_t, std::int32_t, std::int32_t,
                            std::int32_t, std::int32_t) -> double {
    return -1.0;
  };
  check(!rdoq_quantize(block, config, bad, result, &error),
        "negative rate rejected");
  bad.coefficient_rate = [](void*, std::int32_t, std::int32_t, std::int32_t,
                            std::int32_t, std::int32_t) -> double {
    return std::numeric_limits<double>::quiet_NaN();
  };
  check(!rdoq_quantize(block, config, bad, result, &error),
        "NaN rate rejected");
}

void test_av2_model() {
  std::vector<std::int32_t> scan(64, 0);
  av2::build_scan(8, 8, scan);
  check(scan[0] == 0, "scan starts at DC");
  check(scan[1] == 8, "scan step 1 is below DC");
  check(scan.size() == 64, "scan covers the block");
  bool bijective = true;
  std::vector<int> seen(64, 0);
  for (std::int32_t pos : scan) {
    if (pos < 0 || pos >= 64 || seen[static_cast<std::size_t>(pos)] != 0) {
      bijective = false;
    } else {
      seen[static_cast<std::size_t>(pos)] = 1;
    }
  }
  check(bijective, "scan is a permutation");

  av2::BlockInfo info;
  info.params = av2enc::txv::TxParams{};
  info.params.tx_size = 1;
  info.params.coded_w = 8;
  info.params.coded_h = 8;
  info.params.w = 8;
  info.params.h = 8;
  info.plane = 0;
  info.txb_skip_context = 0;
  info.dc_sign_context = 0;

  std::vector<std::int32_t> levels(64, 0);
  levels[0] = 12;
  levels[1] = -3;
  levels[9] = 2;
  av2::Model model;
  std::string error;
  check(model.init(info, levels, &error), "AV2 model initialises");
  check(model.scalar_eob() == 10, "AV2 model scalar EOB");
  const double scalar = model.scalar_bits();
  check(std::isfinite(scalar) && scalar > 0.0, "AV2 model bits are positive");

  // §8.3.2: an inter luma block selects txb_skip[1] and EOB context 1, so its
  // rate must not be the intra rate.  The intra default keeps every existing
  // caller unchanged.
  av2::BlockInfo inter_info = info;
  inter_info.is_inter = true;
  av2::Model inter_model;
  check(inter_model.init(inter_info, levels, &error),
        "AV2 inter model initialises");
  const double inter_scalar = inter_model.scalar_bits();
  check(std::isfinite(inter_scalar) && inter_scalar > 0.0,
        "AV2 inter bits are positive");
  check(inter_scalar != scalar || av2::Model::eob_rate(&inter_model, 0, 64) !=
                                      av2::Model::eob_rate(&model, 0, 64),
        "inter mode selects the inter coefficient CDFs");

  // Spot-check monotonicity at DC.  The trellis does not rely on it (it uses a
  // distortion-based stop rule), but a monotone cost here is a useful sanity
  // check of the CDF lookup.
  double previous = -1.0;
  bool monotone = true;
  for (std::int32_t magnitude = 1; magnitude <= 40; ++magnitude) {
    const double bits = av2::Model::coefficient_rate(
        &model, 0, 0, magnitude, 0, 0);
    if (bits < previous) monotone = false;
    previous = bits;
  }
  check(monotone, "AV2 DC coefficient cost is monotone in magnitude");

  // RDOQ with the AV2 model must not be worse than the scalar path under the
  // same model (the model is static, so this is a like-for-like comparison).
  std::vector<Coefficient> block(64);
  for (std::int32_t i = 0; i < 64; ++i) {
    block[static_cast<std::size_t>(i)].original =
        static_cast<double>(levels[static_cast<std::size_t>(i)]) * 8.0;
    block[static_cast<std::size_t>(i)].level_context = i;
  }
  Config config;
  config.lambda = 20.0;
  config.quant_step = 8.0;
  config.prev_level_cap = 2;
  RdoqResult result;
  check(rdoq_quantize(block, config, model.rate_model(), result, &error),
        "RDOQ with the AV2 model succeeds");
  const double scalar_cost = [&]() {
    double distortion = 0.0;
    for (std::size_t i = 0; i < block.size(); ++i) {
      const double difference = block[i].original -
                                static_cast<double>(levels[i]) * config.quant_step;
      distortion += difference * difference;
    }
    return distortion + config.lambda * scalar;
  }();
  check(result.cost <= scalar_cost + 1.0e-6,
        "RDOQ with the AV2 model is no worse than scalar");
}

}  // namespace

int main() {
  std::printf("av2enc RDOQ trellis test\n");
  test_config_validation();
  test_lambda_zero_recovers_scalar();
  test_determinism();
  // 4x4 with binary levels: 2^16 canonical paths.
  test_exhaustive(16, 1, 16, 30.0, 11, 3.0, "4x4 binary exhaustive");
  // 4x4 with a 8-coefficient active prefix and levels 0..3.
  test_exhaustive(16, 3, 8, 120.0, 12, 7.0, "4x4 four-level exhaustive");
  // 8x8 with a 10-coefficient active prefix and levels 0..2.
  test_exhaustive(64, 2, 10, 80.0, 13, 5.0, "8x8 active-prefix exhaustive");
  test_random_larger_blocks();
  test_batch();
  test_invalid_inputs();
  test_av2_model();
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  if (g_failures != 0) {
    std::printf("RESULT: FAIL\n");
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
