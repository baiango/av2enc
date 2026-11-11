#include "rdoq.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::rdoq {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// The AV2 CDF costs are not strictly monotone in magnitude (a higher base/BR
// symbol can be cheaper than a lower one by a fraction of a bit).  The
// distortion-based stop rule in the DP handles that safely, but the loop needs
// a generous cap before the rule can trigger.  The standard 2*|c|/s+1 bound is
// extended by this margin; real M1 blocks stop well before it.
constexpr std::int32_t kLevelMargin = 64;

std::int32_t abs_i32(std::int32_t value) noexcept {
  if (value == std::numeric_limits<std::int32_t>::min()) {
    return std::numeric_limits<std::int32_t>::max();
  }
  return value < 0 ? -value : value;
}

bool valid_rate(double rate) noexcept {
  return std::isfinite(rate) && rate >= 0.0;
}

std::int32_t scalar_level_for(const Coefficient& coefficient,
                              double quant_step) noexcept {
  if (coefficient.scalar_level != kAutoScalarLevel) {
    return coefficient.scalar_level;
  }
  const double value = coefficient.original / quant_step;
  if (!std::isfinite(value)) {
    return 0;
  }
  const double rounded = std::round(value);  // half away from zero
  if (rounded >= static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
    return std::numeric_limits<std::int32_t>::max();
  }
  if (rounded <= static_cast<double>(std::numeric_limits<std::int32_t>::min())) {
    return std::numeric_limits<std::int32_t>::min();
  }
  return static_cast<std::int32_t>(rounded);
}

// One transition for the DP.  `distortion` and `rate_bits` are path totals, so
// the tie-break is a total order over the suffix, not just the local choice.
struct Choice {
  double cost = kInfinity;
  double distortion = 0.0;
  double rate_bits = 0.0;
  std::int32_t signed_level = 0;
  std::int32_t abs_level = 0;
  std::int32_t mode = 0;  // 0 = terminate here, 1 = continue, 2 = eob here
  bool terminal = false;
};

bool choice_better(const Choice& a, const Choice& b) noexcept {
  if (a.cost != b.cost) {
    return a.cost < b.cost;
  }
  if (a.distortion != b.distortion) {
    return a.distortion < b.distortion;
  }
  if (a.rate_bits != b.rate_bits) {
    return a.rate_bits < b.rate_bits;
  }
  if (a.abs_level != b.abs_level) {
    return a.abs_level < b.abs_level;
  }
  if (a.terminal != b.terminal) {
    return a.terminal;  // prefer the earlier canonical EOB
  }
  return a.signed_level < b.signed_level;
}

bool level_bound_for(const Coefficient& coefficient,
                     const Config& config,
                     std::int32_t* bound,
                     bool* truncated) noexcept {
  const double ratio = std::fabs(coefficient.original) / config.quant_step;
  if (!std::isfinite(ratio)) {
    return false;
  }
  // Classic dominated-free bound.  The DP additionally uses a distortion stop
  // rule, so a non-monotone rate cannot make the search miss a higher level;
  // this value is the loop cap (plus kLevelMargin).  Levels below the state cap
  // still matter because they change the next state.
  double raw = std::floor(2.0 * ratio) + 1.0;
  const double max_level = static_cast<double>(std::numeric_limits<std::int32_t>::max() - 1);
  if (raw > max_level) {
    raw = max_level;
  }
  std::int64_t value = static_cast<std::int64_t>(raw);
  if (coefficient.original == 0.0 && !config.allow_zero_to_one) {
    value = 0;
  } else {
    value = std::max<std::int64_t>(value, config.prev_level_cap);
    const std::int64_t with_margin =
        value + static_cast<std::int64_t>(kLevelMargin);
    value = std::min<std::int64_t>(
        with_margin, static_cast<std::int64_t>(
                         std::numeric_limits<std::int32_t>::max() - 1));
  }
  value = std::max<std::int64_t>(value, 0);

  bool clipped = false;
  if (config.level_cap > 0 &&
      value > static_cast<std::int64_t>(config.level_cap)) {
    value = config.level_cap;
    clipped = true;
  }
  *bound = static_cast<std::int32_t>(value);
  *truncated = clipped;
  return true;
}

}  // namespace

bool rdoq_config_valid(const Config& config) noexcept {
  return std::isfinite(config.lambda) && config.lambda >= 0.0 &&
         std::isfinite(config.quant_step) && config.quant_step > 0.0 &&
         config.prev_level_cap >= 0 && config.prev_level_cap <= 63 &&
         config.level_cap >= 0 && config.max_coefficients >= 0;
}

bool validate_rdoq_config(const Config& config, std::string* error) {
  if (!std::isfinite(config.lambda) || config.lambda < 0.0) {
    if (error != nullptr) *error = "RdoqConfig.lambda must be finite and >= 0";
    return false;
  }
  if (!std::isfinite(config.quant_step) || config.quant_step <= 0.0) {
    if (error != nullptr) *error = "RdoqConfig.quant_step must be finite and > 0";
    return false;
  }
  if (config.prev_level_cap < 0 || config.prev_level_cap > 63) {
    if (error != nullptr) *error = "RdoqConfig.prev_level_cap must be in [0, 63]";
    return false;
  }
  if (config.level_cap < 0) {
    if (error != nullptr) *error = "RdoqConfig.level_cap must be >= 0";
    return false;
  }
  if (config.max_coefficients < 0) {
    if (error != nullptr) *error = "RdoqConfig.max_coefficients must be >= 0";
    return false;
  }
  return true;
}

double default_coefficient_rate(void* context,
                                std::int32_t scan_pos,
                                std::int32_t prev_abs_state,
                                std::int32_t level,
                                std::int32_t level_context,
                                std::int32_t is_eob) noexcept {
  (void)context;
  (void)scan_pos;
  (void)level_context;
  const std::int32_t magnitude = abs_i32(level);
  if (magnitude == 0) {
    return prev_abs_state == 0 ? 0.5 : 1.0;
  }
  double bits = 2.0 + std::floor(std::log2(static_cast<double>(magnitude) + 1.0));
  if (prev_abs_state == 0) {
    bits += 1.0;
  }
  if (is_eob != 0) {
    bits += 0.5;
  }
  return bits + 1.0;  // sign
}

double default_eob_rate(void* context,
                        std::int32_t eob,
                        std::int32_t num_coefficients) noexcept {
  (void)context;
  if (num_coefficients <= 0 || eob >= num_coefficients) {
    return 0.0;
  }
  return 1.0 + std::log2(static_cast<double>(eob) + 1.0);
}

RateModel default_rate_model() noexcept {
  RateModel model;
  model.context = nullptr;
  model.coefficient_rate = &default_coefficient_rate;
  model.eob_rate = &default_eob_rate;
  return model;
}

std::int32_t rdoq_level_bound(const Coefficient& coefficient,
                              const Config& config) noexcept {
  std::int32_t bound = 0;
  bool truncated = false;
  if (!level_bound_for(coefficient, config, &bound, &truncated)) {
    return -1;
  }
  return bound;
}

bool estimate_scalar_levels(std::span<const Coefficient> coefficients,
                            double quant_step,
                            std::span<std::int32_t> levels,
                            std::int32_t* eob,
                            std::string* error) {
  if (!std::isfinite(quant_step) || quant_step <= 0.0) {
    if (error != nullptr) *error = "quant_step must be finite and > 0";
    return false;
  }
  if (levels.size() != coefficients.size()) {
    if (error != nullptr) *error = "levels span size must match the block";
    return false;
  }
  std::int32_t last = 0;
  for (std::size_t i = 0; i < coefficients.size(); ++i) {
    if (!std::isfinite(coefficients[i].original)) {
      if (error != nullptr) *error = "Coefficient.original must be finite";
      return false;
    }
    const std::int32_t level = scalar_level_for(coefficients[i], quant_step);
    levels[i] = level;
    if (level != 0) {
      last = static_cast<std::int32_t>(i) + 1;
    }
  }
  if (eob != nullptr) *eob = last;
  return true;
}

bool rdoq_quantize(std::span<const Coefficient> coefficients,
                   const Config& config,
                   const RateModel& rate_model,
                   RdoqResult& result,
                   std::string* error) {
  result = RdoqResult{};

  if (!validate_rdoq_config(config, error)) {
    return false;
  }
  const std::size_t count = coefficients.size();
  if (count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    if (error != nullptr) *error = "coefficient count does not fit in int32";
    return false;
  }
  const std::int32_t num_coefficients = static_cast<std::int32_t>(count);
  if (num_coefficients > config.max_coefficients) {
    if (error != nullptr) {
      *error = "coefficient count exceeds RdoqConfig.max_coefficients";
    }
    return false;
  }

  CoefficientRateFn coefficient_rate = rate_model.coefficient_rate;
  if (coefficient_rate == nullptr) {
    coefficient_rate = &default_coefficient_rate;
  }
  EobRateFn eob_rate = rate_model.eob_rate;
  if (eob_rate == nullptr) {
    eob_rate = &default_eob_rate;
  }
  void* rate_context = rate_model.context;
  const double lambda = config.lambda;
  const double quant_step = config.quant_step;
  const std::int32_t cap = config.prev_level_cap;
  const std::int32_t states = cap + 1;

  if (count == 0) {
    const double bits = eob_rate(rate_context, 0, 0);
    if (!valid_rate(bits)) {
      if (error != nullptr) *error = "RdoqEobRateFn returned an invalid value";
      return false;
    }
    result.eob = 0;
    result.rate_bits = bits;
    result.cost = lambda * bits;
    if (!std::isfinite(result.cost)) {
      if (error != nullptr) *error = "RDOQ cost is not finite";
      result = RdoqResult{};
      return false;
    }
    return true;
  }

  // EOB costs, including the all-zero / not-all-zero flag.
  std::vector<double> eob_bits(static_cast<std::size_t>(num_coefficients) + 1, 0.0);
  for (std::int32_t i = 0; i <= num_coefficients; ++i) {
    const double bits = eob_rate(rate_context, i, num_coefficients);
    if (!valid_rate(bits)) {
      if (error != nullptr) *error = "RdoqEobRateFn returned an invalid value";
      return false;
    }
    eob_bits[static_cast<std::size_t>(i)] = bits;
  }

  // Distortion if every coefficient from i onward is forced to zero.
  std::vector<double> suffix_zero(static_cast<std::size_t>(count) + 1, 0.0);
  std::vector<std::int32_t> signs(count, 1);
  for (std::size_t i = count; i-- > 0;) {
    const double original = coefficients[i].original;
    if (!std::isfinite(original)) {
      if (error != nullptr) *error = "Coefficient.original must be finite";
      return false;
    }
    suffix_zero[i] = suffix_zero[i + 1] + original * original;
    if (original < 0.0) {
      signs[i] = -1;
    } else if (original > 0.0) {
      signs[i] = 1;
    } else {
      const std::int32_t seed = scalar_level_for(coefficients[i], quant_step);
      signs[i] = seed < 0 ? -1 : 1;
    }
  }

  const std::size_t path_cells =
      count * static_cast<std::size_t>(states);
  std::vector<std::int32_t> chosen_level(path_cells, 0);
  std::vector<std::uint8_t> chosen_mode(path_cells, 0);

  // State at position `count` is unreachable: the trellis must terminate with
  // an explicit EOB transition at a non-zero coefficient, so there is no valid
  // path that "runs off the end" with a trailing zero.
  std::vector<double> dp_next(static_cast<std::size_t>(states), kInfinity);
  std::vector<double> dd_next(static_cast<std::size_t>(states), 0.0);
  std::vector<double> dr_next(static_cast<std::size_t>(states), 0.0);

  std::vector<double> dp_cur(static_cast<std::size_t>(states), kInfinity);
  std::vector<double> dd_cur(static_cast<std::size_t>(states), 0.0);
  std::vector<double> dr_cur(static_cast<std::size_t>(states), 0.0);
  std::int32_t max_level_searched = 0;
  bool truncated = false;

  for (std::size_t ii = count; ii-- > 0;) {
    const std::int32_t scan_pos = static_cast<std::int32_t>(ii);
    const Coefficient& coefficient = coefficients[ii];
    const double original = coefficient.original;
    const std::int32_t sign = signs[ii];

    std::int32_t bound = 0;
    bool clipped = false;
    if (!level_bound_for(coefficient, config, &bound, &clipped)) {
      if (error != nullptr) *error = "Coefficient.original must be finite";
      return false;
    }
    truncated = truncated || clipped;

    const double ratio = std::fabs(original) / quant_step;
    std::int32_t closest = 0;
    if (std::isfinite(ratio)) {
      const double rounded = std::round(ratio);
      if (rounded <= 0.0) {
        closest = 0;
      } else if (rounded >= static_cast<double>(bound)) {
        closest = bound;
      } else {
        closest = static_cast<std::int32_t>(rounded);
      }
    }
    std::int32_t max_here = 0;
    bool capped = false;

    const double zero_after = suffix_zero[ii + 1];

    for (std::int32_t prev = 0; prev < states; ++prev) {
      Choice best;
      // The all-zero block is only meaningful at position 0; every other path
      // must code at least one non-zero coefficient as the EOB coefficient.
      if (ii == 0) {
        best.distortion = suffix_zero[0];
        best.rate_bits = eob_bits[0];
        best.cost = best.distortion + lambda * best.rate_bits;
        best.signed_level = 0;
        best.abs_level = 0;
        best.mode = 0;
        best.terminal = true;
      }

      for (std::int32_t magnitude = 0; magnitude <= bound; ++magnitude) {
        const std::int32_t signed_level =
            magnitude == 0 ? 0 : sign * magnitude;
        const double reconstructed =
            static_cast<double>(signed_level) * quant_step;
        const double difference = original - reconstructed;
        const double distortion = difference * difference;
        const std::int32_t next_state = std::min(magnitude, cap);

        // Continue coding: this coefficient is a normal (possibly zero) level
        // and a later coefficient is the EOB coefficient.
        const double normal_bits = coefficient_rate(
            rate_context, scan_pos, prev, signed_level,
            coefficient.level_context, 0);
        if (!valid_rate(normal_bits)) {
          if (error != nullptr) *error = "RdoqRateFn returned an invalid value";
          return false;
        }
        const std::size_t next_index = static_cast<std::size_t>(next_state);
        if (std::isfinite(dp_next[next_index])) {
          Choice candidate;
          candidate.cost =
              distortion + lambda * normal_bits + dp_next[next_index];
          candidate.distortion = distortion + dd_next[next_index];
          candidate.rate_bits = normal_bits + dr_next[next_index];
          candidate.signed_level = signed_level;
          candidate.abs_level = magnitude;
          candidate.mode = 1;
          candidate.terminal = false;
          if (choice_better(candidate, best)) {
            best = candidate;
          }
        }

        // This coefficient is the last non-zero coefficient (EOB).
        if (magnitude != 0) {
          const double eob_coeff_bits = coefficient_rate(
              rate_context, scan_pos, prev, signed_level,
              coefficient.level_context, 1);
          if (!valid_rate(eob_coeff_bits)) {
            if (error != nullptr) *error = "RdoqRateFn returned an invalid value";
            return false;
          }
          Choice candidate;
          candidate.distortion = distortion + zero_after;
          candidate.rate_bits =
              eob_coeff_bits + eob_bits[ii + 1];
          candidate.cost =
              candidate.distortion + lambda * candidate.rate_bits;
          candidate.signed_level = signed_level;
          candidate.abs_level = magnitude;
          candidate.mode = 2;
          candidate.terminal = true;
          if (choice_better(candidate, best)) {
            best = candidate;
          }
        }
        max_here = std::max(max_here, magnitude);
        // Safe stop rule: at or above the distortion minimum every higher
        // magnitude has at least this distortion, so once the local
        // distortion alone exceeds the best total cost found for this state,
        // no higher magnitude can win.  This does not assume a monotone rate.
        if (magnitude >= closest && std::isfinite(best.cost) &&
            distortion >= best.cost) {
          break;
        }
      }
      dp_cur[static_cast<std::size_t>(prev)] = best.cost;
      dd_cur[static_cast<std::size_t>(prev)] = best.distortion;
      dr_cur[static_cast<std::size_t>(prev)] = best.rate_bits;
      const std::size_t slot =
          ii * static_cast<std::size_t>(states) +
          static_cast<std::size_t>(prev);
      chosen_level[slot] = best.signed_level;
      chosen_mode[slot] = static_cast<std::uint8_t>(best.mode);
    }
    // A coefficient is "capped" if any state ran to `bound` without the safe
    // distortion stop.  `max_here` is shared across states, so compare it with
    // the bound after the loop.
    if (bound > 0 && max_here >= bound) {
      capped = true;
    }
    max_level_searched = std::max(max_level_searched, max_here);
    truncated = truncated || capped;

    dp_next.swap(dp_cur);
    dd_next.swap(dd_cur);
    dr_next.swap(dr_cur);
  }

  if (!std::isfinite(dp_next[0])) {
    if (error != nullptr) *error = "RDOQ trellis has no valid path";
    return false;
  }

  result.levels.assign(count, 0);
  std::int32_t state = 0;
  std::int32_t eob = num_coefficients;
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t slot =
        i * static_cast<std::size_t>(states) +
        static_cast<std::size_t>(state);
    const std::int32_t level = chosen_level[slot];
    const std::int32_t mode = chosen_mode[slot];
    if (mode == 0) {
      eob = static_cast<std::int32_t>(i);
      break;
    }
    result.levels[i] = level;
    state = std::min(abs_i32(level), cap);
    if (mode == 2) {
      eob = static_cast<std::int32_t>(i) + 1;
      break;
    }
  }

  // Recompute cost from the chosen path.  This is the reported value and also
  // catches a non-deterministic rate callback.
  double distortion = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double reconstructed =
        static_cast<double>(result.levels[i]) * quant_step;
    const double difference = coefficients[i].original - reconstructed;
    distortion += difference * difference;
  }
  double rate_bits = eob_bits[static_cast<std::size_t>(eob)];
  std::int32_t previous = 0;
  for (std::int32_t i = 0; i < eob; ++i) {
    const std::int32_t level = result.levels[static_cast<std::size_t>(i)];
    const std::int32_t is_eob =
        (i == eob - 1 && level != 0) ? 1 : 0;
    const double bits = coefficient_rate(
        rate_context, i, previous, level,
        coefficients[static_cast<std::size_t>(i)].level_context, is_eob);
    if (!valid_rate(bits)) {
      if (error != nullptr) *error = "RdoqRateFn returned an invalid value";
      result = RdoqResult{};
      return false;
    }
    rate_bits += bits;
    previous = std::min(abs_i32(level), cap);
  }
  const double cost = distortion + lambda * rate_bits;
  if (!std::isfinite(distortion) || !std::isfinite(rate_bits) ||
      !valid_rate(rate_bits) || !std::isfinite(cost)) {
    if (error != nullptr) *error = "RDOQ result is not finite";
    result = RdoqResult{};
    return false;
  }

  result.eob = eob;
  result.distortion = distortion;
  result.rate_bits = rate_bits;
  result.cost = cost;
  result.max_level_searched = max_level_searched;
  result.level_range_truncated = truncated;
  return true;
}

bool rdoq_quantize_batch(const BatchInput& input,
                         const BatchConfig& config,
                         const RateModel& rate_model,
                         BatchOutput& output,
                         std::string* error) {
  if (input.blocks.empty()) {
    return true;
  }
  if (output.eob.size() < input.blocks.size() ||
      output.distortion.size() < input.blocks.size() ||
      output.rate_bits.size() < input.blocks.size() ||
      output.cost.size() < input.blocks.size() ||
      output.status.size() < input.blocks.size()) {
    if (error != nullptr) *error = "batch output arrays are too small";
    return false;
  }
  const bool has_context = !input.level_context.empty();
  const bool has_scalar = !input.scalar_level.empty();
  if (has_context && input.level_context.size() != input.original.size()) {
    if (error != nullptr) *error = "level_context size must match original";
    return false;
  }
  if (has_scalar && input.scalar_level.size() != input.original.size()) {
    if (error != nullptr) *error = "scalar_level size must match original";
    return false;
  }

  std::vector<Coefficient> scratch;
  for (std::size_t block_index = 0; block_index < input.blocks.size();
       ++block_index) {
    const BatchBlock& block = input.blocks[block_index];
    output.status[block_index] = -1;
    if (block.coefficient_offset < 0 || block.num_coefficients < 0 ||
        block.level_offset < 0) {
      if (error != nullptr) *error = "batch block has a negative offset/size";
      return false;
    }
    const std::size_t begin =
        static_cast<std::size_t>(block.coefficient_offset);
    const std::size_t n = static_cast<std::size_t>(block.num_coefficients);
    const std::size_t level_begin =
        static_cast<std::size_t>(block.level_offset);
    if (begin + n > input.original.size() ||
        level_begin + n > output.levels.size()) {
      if (error != nullptr) *error = "batch block is outside its arrays";
      return false;
    }

    scratch.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      Coefficient& coefficient = scratch[i];
      coefficient.original = input.original[begin + i];
      coefficient.level_context =
          has_context ? input.level_context[begin + i] : 0;
      coefficient.scalar_level =
          has_scalar ? input.scalar_level[begin + i] : kAutoScalarLevel;
    }

    Config block_config;
    block_config.lambda = block.lambda;
    block_config.quant_step = block.quant_step;
    block_config.prev_level_cap = block.prev_level_cap;
    block_config.level_cap = block.level_cap;
    block_config.allow_zero_to_one =
        (block.flags & kBatchAllowZeroToOne) != 0;
    block_config.max_coefficients = config.max_coefficients;

    RdoqResult result;
    std::string block_error;
    if (!rdoq_quantize(scratch, block_config, rate_model, result,
                       &block_error)) {
      if (error != nullptr) {
        *error = "block " + std::to_string(block_index) + ": " + block_error;
      }
      return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
      output.levels[level_begin + i] = result.levels[i];
    }
    output.eob[block_index] = result.eob;
    output.distortion[block_index] = result.distortion;
    output.rate_bits[block_index] = result.rate_bits;
    output.cost[block_index] = result.cost;
    output.status[block_index] = 0;
  }
  return true;
}

}  // namespace av2enc::rdoq
