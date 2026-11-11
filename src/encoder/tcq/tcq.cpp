#include "tcq.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::tcq {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();
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
  const double rounded = std::round(value);
  if (rounded >= static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
    return std::numeric_limits<std::int32_t>::max();
  }
  if (rounded <= static_cast<double>(std::numeric_limits<std::int32_t>::min())) {
    return std::numeric_limits<std::int32_t>::min();
  }
  return static_cast<std::int32_t>(rounded);
}

// One DP choice.  `distortion` and `rate_bits` are suffix totals, so the
// comparison is a total order over the chosen suffix.
struct Choice {
  double cost = kInfinity;
  double distortion = 0.0;
  double rate_bits = 0.0;
  std::int32_t base_level = 0;   // signed syntax level
  std::int32_t quant_level = 0;  // signed final Quant magnitude
  std::int32_t next_state = 0;
  std::int32_t abs_level = 0;
};

bool choice_better(const Choice& a, const Choice& b) noexcept {
  if (a.cost != b.cost) return a.cost < b.cost;
  if (a.distortion != b.distortion) return a.distortion < b.distortion;
  if (a.rate_bits != b.rate_bits) return a.rate_bits < b.rate_bits;
  if (a.abs_level != b.abs_level) return a.abs_level < b.abs_level;
  return a.quant_level < b.quant_level;
}

bool bound_for(const Coefficient& coefficient, const Config& config,
               std::int32_t* bound, bool* truncated) noexcept {
  const double ratio = std::fabs(coefficient.original) / config.quant_step;
  if (!std::isfinite(ratio)) {
    return false;
  }
  // m = 2*q - q0, so q is about ratio.  The +2 covers q0 and rounding; the
  // margin lets the safe stop rule deal with non-monotone CDF costs.
  double raw = std::floor(2.0 * ratio) + 2.0;
  if (coefficient.original == 0.0 && !config.allow_zero_to_one) {
    raw = 0.0;
  }
  const double max_level =
      static_cast<double>(std::numeric_limits<std::int32_t>::max() - 1);
  if (raw > max_level) raw = max_level;
  std::int64_t value = static_cast<std::int64_t>(raw);
  if (!(coefficient.original == 0.0 && !config.allow_zero_to_one)) {
    value = std::max<std::int64_t>(value, 1);
    value = std::min<std::int64_t>(
        value + static_cast<std::int64_t>(kLevelMargin),
        static_cast<std::int64_t>(
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

bool config_valid(const Config& config) noexcept {
  return std::isfinite(config.lambda) && config.lambda >= 0.0 &&
         std::isfinite(config.quant_step) && config.quant_step > 0.0 &&
         config.level_cap >= 0 && config.max_coefficients >= 0;
}

bool validate_config(const Config& config, std::string* error) {
  if (!std::isfinite(config.lambda) || config.lambda < 0.0) {
    if (error != nullptr) *error = "tcq::Config.lambda must be finite and >= 0";
    return false;
  }
  if (!std::isfinite(config.quant_step) || config.quant_step <= 0.0) {
    if (error != nullptr) {
      *error = "tcq::Config.quant_step must be finite and > 0";
    }
    return false;
  }
  if (config.level_cap < 0) {
    if (error != nullptr) *error = "tcq::Config.level_cap must be >= 0";
    return false;
  }
  if (config.max_coefficients < 0) {
    if (error != nullptr) *error = "tcq::Config.max_coefficients must be >= 0";
    return false;
  }
  return true;
}

double default_coefficient_rate(void* context, std::int32_t scan_pos,
                                std::int32_t state,
                                std::int32_t base_level,
                                std::int32_t next_state,
                                std::int32_t signed_quant,
                                std::int32_t level_context,
                                std::int32_t is_eob) noexcept {
  (void)context;
  (void)scan_pos;
  (void)next_state;
  (void)signed_quant;
  (void)level_context;
  const std::int32_t magnitude = abs_i32(base_level);
  if (magnitude == 0) {
    return state == 0 ? 0.5 : 1.0;
  }
  double bits =
      2.0 + std::floor(std::log2(static_cast<double>(magnitude) + 1.0));
  bits += 1.0;  // sign
  if (is_eob != 0) bits += 0.5;
  return bits;
}

double default_eob_rate(void* context, std::int32_t eob,
                        std::int32_t num_coefficients) noexcept {
  (void)context;
  if (num_coefficients <= 0 || eob >= num_coefficients) return 0.0;
  return 1.0 + std::log2(static_cast<double>(eob) + 1.0);
}

RateModel default_rate_model() noexcept {
  RateModel model;
  model.context = nullptr;
  model.coefficient_rate = &default_coefficient_rate;
  model.eob_rate = &default_eob_rate;
  return model;
}

std::int32_t level_bound(const Coefficient& coefficient,
                         const Config& config) noexcept {
  std::int32_t bound = 0;
  bool truncated = false;
  if (!bound_for(coefficient, config, &bound, &truncated)) return -1;
  return bound;
}

bool estimate_scalar_levels(std::span<const Coefficient> coefficients,
                            double quant_step,
                            std::span<std::int32_t> levels,
                            std::int32_t* eob, std::string* error) {
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
    if (level != 0) last = static_cast<std::int32_t>(i) + 1;
  }
  if (eob != nullptr) *eob = last;
  return true;
}

bool quantize(std::span<const Coefficient> coefficients, const Config& config,
              const RateModel& rate_model, Result& result,
              std::string* error) {
  result = Result{};
  if (!validate_config(config, error)) return false;
  if (rate_model.coefficient_rate == nullptr || rate_model.eob_rate == nullptr) {
    if (error != nullptr) *error = "TCQ rate model callbacks must be set";
    return false;
  }
  const std::int32_t n = static_cast<std::int32_t>(coefficients.size());
  if (n < 0 || n > config.max_coefficients) {
    if (error != nullptr) *error = "block exceeds max_coefficients";
    return false;
  }
  result.base_levels.assign(static_cast<std::size_t>(n), 0);
  result.quant_levels.assign(static_cast<std::size_t>(n), 0);
  result.states.assign(static_cast<std::size_t>(n), 0);
  if (n == 0) return true;

  std::vector<std::int32_t> bounds(static_cast<std::size_t>(n), 0);
  bool truncated = false;
  for (std::int32_t c = 0; c < n; ++c) {
    const Coefficient& coefficient = coefficients[static_cast<std::size_t>(c)];
    if (!std::isfinite(coefficient.original)) {
      if (error != nullptr) *error = "Coefficient.original must be finite";
      return false;
    }
    bool clipped = false;
    if (!bound_for(coefficient, config, &bounds[static_cast<std::size_t>(c)],
                   &clipped)) {
      if (error != nullptr) *error = "coefficient level bound is not finite";
      return false;
    }
    truncated = truncated || clipped;
  }

  const double half_step = config.quant_step * 0.5;
  Choice empty_suffix;
  empty_suffix.cost = 0.0;
  // dp[c][s] = best cost for positions c..0 given state s entering c.
  std::vector<Choice> dp(static_cast<std::size_t>(n) * kStateCount);
  const auto dp_at = [&](std::int32_t c, std::int32_t s) -> const Choice& {
    return dp[static_cast<std::size_t>(c) * kStateCount +
              static_cast<std::size_t>(s)];
  };
  const auto dp_mut = [&](std::int32_t c, std::int32_t s) -> Choice& {
    return dp[static_cast<std::size_t>(c) * kStateCount +
              static_cast<std::size_t>(s)];
  };

  std::int32_t max_level_searched = 0;
  for (std::int32_t c = 0; c < n; ++c) {
    const Coefficient& coefficient = coefficients[static_cast<std::size_t>(c)];
    const double original = coefficient.original;
    const int sign = original < 0.0 ? -1 : 1;
    const std::int32_t bound = bounds[static_cast<std::size_t>(c)];
    const std::int32_t pre_max =
        coefficient.pre_max > 0 ? coefficient.pre_max : 6;
    max_level_searched = std::max(max_level_searched, bound);
    for (std::int32_t s = 0; s < kStateCount; ++s) {
      Choice best;
      for (std::int32_t q = 0; q <= bound; ++q) {
        const bool nonzero = q > 0;
        // `read_quant` adds x << 1 from pre_max - 1 upward.  The writer
        // chooses pre_max or pre_max - 1 by parity, so every q is codable
        // and the state parity stays q & 1.
        const std::int32_t syntax_level =
            q == 0 ? 0 : syntax_level_for(q, pre_max);
        const std::int32_t q0 = nonzero ? state_q0(s) : 0;
        const std::int32_t magnitude = nonzero ? 2 * q - q0 : 0;
        const std::int32_t base = nonzero ? sign * q : 0;
        const std::int32_t quant = nonzero ? sign * magnitude : 0;
        const std::int32_t next = next_state(s, syntax_level & 1);
        const double local_dist =
            (original - static_cast<double>(quant) * half_step) *
            (original - static_cast<double>(quant) * half_step);
        const double local_rate = rate_model.coefficient_rate(
            rate_model.context, c, s, base, next, quant,
            coefficient.level_context, /*is_eob=*/0);
        if (!valid_rate(local_rate)) {
          if (error != nullptr) *error = "coefficient rate must be finite and >= 0";
          return false;
        }
        const Choice& suffix = c > 0 ? dp_at(c - 1, next) : empty_suffix;
        Choice candidate;
        candidate.cost = local_dist + config.lambda * local_rate + suffix.cost;
        candidate.distortion = local_dist + suffix.distortion;
        candidate.rate_bits = local_rate + suffix.rate_bits;
        candidate.base_level = base;
        candidate.quant_level = quant;
        candidate.next_state = next;
        candidate.abs_level = q;
        if (choice_better(candidate, best)) best = candidate;
      }
      dp_mut(c, s) = best;
    }
  }

  const double all_zero_rate =
      rate_model.eob_rate(rate_model.context, 0, n);
  if (!valid_rate(all_zero_rate)) {
    if (error != nullptr) *error = "EOB rate must be finite and >= 0";
    return false;
  }
  // Positions above the start (EOB) are coded as zero, so their original
  // energy is reconstruction error and must be part of the path cost.
  std::vector<double> tail(static_cast<std::size_t>(n) + 1, 0.0);
  for (std::int32_t c = n - 1; c >= 0; --c) {
    const double original = coefficients[static_cast<std::size_t>(c)].original;
    tail[static_cast<std::size_t>(c)] =
        original * original + tail[static_cast<std::size_t>(c) + 1];
  }
  const double all_zero_distortion = tail[0];
  Choice best;
  best.cost = all_zero_distortion + all_zero_rate * config.lambda;
  best.distortion = all_zero_distortion;
  best.rate_bits = all_zero_rate;
  best.abs_level = 0;
  best.quant_level = 0;
  best.next_state = 0;
  std::int32_t start_position = -1;

  // Start at position c: positions above c are zero and not coded; the
  // coefficient at c is the unique EOB coefficient (coeff_base_eob).
  for (std::int32_t c = 0; c < n; ++c) {
    const Coefficient& coefficient = coefficients[static_cast<std::size_t>(c)];
    const double original = coefficient.original;
    const int sign = original < 0.0 ? -1 : 1;
    const std::int32_t bound = bounds[static_cast<std::size_t>(c)];
    const std::int32_t pre_max =
        coefficient.pre_max > 0 ? coefficient.pre_max : 6;
    for (std::int32_t q = 1; q <= bound; ++q) {
      const std::int32_t syntax_level = syntax_level_for(q, pre_max);
      const std::int32_t q0 = state_q0(0);  // entering state at the start is 0
      const std::int32_t magnitude = 2 * q - q0;
      const std::int32_t base = sign * q;
      const std::int32_t quant = sign * magnitude;
      const std::int32_t next = next_state(0, syntax_level & 1);
      const double local_dist =
          (original - static_cast<double>(quant) * half_step) *
          (original - static_cast<double>(quant) * half_step);
      const double local_rate = rate_model.coefficient_rate(
          rate_model.context, c, 0, base, next, quant,
          coefficient.level_context, /*is_eob=*/1);
      if (!valid_rate(local_rate)) {
        if (error != nullptr) *error = "coefficient rate must be finite and >= 0";
        return false;
      }
      const double eob_rate =
          rate_model.eob_rate(rate_model.context, c + 1, n);
      if (!valid_rate(eob_rate)) {
        if (error != nullptr) *error = "EOB rate must be finite and >= 0";
        return false;
      }
      const Choice& suffix = c > 0 ? dp_at(c - 1, next) : empty_suffix;
      const double tail_distortion = tail[static_cast<std::size_t>(c) + 1];
      Choice candidate;
      candidate.cost = local_dist + tail_distortion +
                       config.lambda * (local_rate + eob_rate) + suffix.cost;
      candidate.distortion = local_dist + tail_distortion + suffix.distortion;
      candidate.rate_bits = local_rate + eob_rate + suffix.rate_bits;
      candidate.base_level = base;
      candidate.quant_level = quant;
      candidate.next_state = next;
      candidate.abs_level = q;
      // Prefer the shorter EOB (smaller c) after all cost/rate ties.  The
      // all-zero path is already in `best`; a start candidate must beat it.
      const bool better =
          choice_better(candidate, best) ||
          (start_position >= 0 && !choice_better(best, candidate) &&
           c < start_position);
      if (better) {
        best = candidate;
        start_position = c;
      }
    }
  }

  result.distortion = best.distortion;
  result.rate_bits = best.rate_bits;
  result.cost = best.cost;
  result.max_level_searched = max_level_searched;
  result.level_range_truncated = truncated;
  if (start_position < 0) {
    result.eob = 0;
    return true;
  }

  result.eob = start_position + 1;
  result.base_levels[static_cast<std::size_t>(start_position)] =
      best.base_level;
  result.quant_levels[static_cast<std::size_t>(start_position)] =
      best.quant_level;
  result.states[static_cast<std::size_t>(start_position)] = 0;
  std::int32_t state = best.next_state;
  for (std::int32_t c = start_position - 1; c >= 0; --c) {
    const Choice& choice = dp_at(c, state);
    result.base_levels[static_cast<std::size_t>(c)] = choice.base_level;
    result.quant_levels[static_cast<std::size_t>(c)] = choice.quant_level;
    result.states[static_cast<std::size_t>(c)] = state;
    state = choice.next_state;
  }
  return true;
}

bool decode_levels(std::span<const std::int32_t> base_levels_scan,
                   bool allow_tcq, bool lossless, DecodeResult* result,
                   std::string* error,
                   std::span<const std::int32_t> pre_max_scan) {
  if (result == nullptr) {
    if (error != nullptr) *error = "decode result must not be null";
    return false;
  }
  *result = DecodeResult{};
  const std::int32_t n = static_cast<std::int32_t>(base_levels_scan.size());
  result->quant_levels.assign(static_cast<std::size_t>(n), 0);
  result->states.assign(static_cast<std::size_t>(n), 0);
  if (n == 0) return true;
  if (!pre_max_scan.empty() &&
      pre_max_scan.size() != static_cast<std::size_t>(n)) {
    if (error != nullptr) *error = "pre_max span size must match the block";
    return false;
  }
  std::int32_t eob = 0;
  for (std::int32_t c = 0; c < n; ++c) {
    if (base_levels_scan[static_cast<std::size_t>(c)] != 0) eob = c + 1;
  }
  result->eob = eob;
  const bool use_tcq = allow_tcq && !lossless;
  const auto pre_max_at = [&](std::int32_t c) -> std::int32_t {
    if (pre_max_scan.empty()) return 6;
    const std::int32_t value = pre_max_scan[static_cast<std::size_t>(c)];
    return value > 0 ? value : 6;
  };

  // First pass, exactly §5.20.7.27: the state is updated from the level the
  // entropy syntax reconstructs, syntax_level_for(full_level, pre_max).  The
  // state is reset before the second pass.
  if (use_tcq) {
    std::int32_t state = 0;
    for (std::int32_t c = eob - 1; c >= 0; --c) {
      const std::int32_t level =
          abs_i32(base_levels_scan[static_cast<std::size_t>(c)]);
      const std::int32_t syntax_level =
          level == 0 ? 0 : syntax_level_for(level, pre_max_at(c));
      state = next_state(state, syntax_level & 1);
    }
  }

  std::int32_t state = 0;
  for (std::int32_t c = eob - 1; c >= 0; --c) {
    const std::int32_t signed_level =
        base_levels_scan[static_cast<std::size_t>(c)];
    const std::int32_t level = abs_i32(signed_level);
    const std::int32_t pre_max = pre_max_at(c);
    const std::int32_t syntax_level =
        level == 0 ? 0 : syntax_level_for(level, pre_max);
    const int sign = signed_level < 0 ? -1 : 1;
    result->states[static_cast<std::size_t>(c)] = state;
    std::int32_t quant = level;
    if (use_tcq) {
      const std::int32_t q0 = state_q0(state);
      state = next_state(state, syntax_level & 1);
      if (quant > 0) quant = quant * 2 - q0;
    }
    result->quant_levels[static_cast<std::size_t>(c)] =
        quant == 0 ? 0 : sign * quant;
  }
  return true;
}

}  // namespace av2enc::tcq
