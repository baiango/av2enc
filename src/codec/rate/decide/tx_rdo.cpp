#include "tx_rdo.h"

#include "codec/transform/variants/tx_variants.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace av2enc::still::rdo_decide {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

std::int32_t abs_i32(std::int32_t value) noexcept {
  if (value == std::numeric_limits<std::int32_t>::min()) {
    return std::numeric_limits<std::int32_t>::max();
  }
  return value < 0 ? -value : value;
}

bool valid_rate(double rate) noexcept {
  return std::isfinite(rate) && rate >= 0.0;
}

bool type_cost_better(double cost_a, double sse_a, int type_a,
                      double cost_b, double sse_b, int type_b) noexcept {
  if (cost_a != cost_b) {
    return cost_a < cost_b;
  }
  if (sse_a != sse_b) {
    return sse_a < sse_b;
  }
  return type_a < type_b;
}

struct Choice {
  double cost = kInfinity;
  double distortion = 0.0;
  double rate_bits = 0.0;
  std::int32_t signed_level = 0;
  std::int32_t abs_level = 0;
  std::int32_t mode = 0; // 0 = terminate, 1 = continue, 2 = EOB here
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
    return a.terminal;
  }
  return a.signed_level < b.signed_level;
}

std::int32_t scalar_level(double original, double quant_step) noexcept {
  const double ratio = original / quant_step;
  if (!std::isfinite(ratio)) {
    return 0;
  }
  const double rounded = std::round(ratio);
  if (rounded >=
      static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
    return std::numeric_limits<std::int32_t>::max();
  }
  if (rounded <=
      static_cast<double>(std::numeric_limits<std::int32_t>::min())) {
    return std::numeric_limits<std::int32_t>::min();
  }
  return static_cast<std::int32_t>(rounded);
}

bool level_bound_for(double original, const RdoqConfig& config,
                     std::int32_t* bound, bool* truncated) noexcept {
  const double ratio = std::fabs(original) / config.quant_step;
  if (!std::isfinite(ratio)) {
    return false;
  }
  double raw = std::floor(2.0 * ratio) + 1.0;
  const double max_level = static_cast<double>(
      std::numeric_limits<std::int32_t>::max() - 1);
  if (raw > max_level) {
    raw = max_level;
  }
  std::int64_t value = static_cast<std::int64_t>(raw);
  if (original == 0.0 && !config.allow_zero_to_one) {
    value = 0;
  } else {
    value = std::max<std::int64_t>(value, config.prev_level_cap);
    value = std::min<std::int64_t>(
        value + static_cast<std::int64_t>(kRdoqLevelMargin),
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

bool decide_tx_type(const std::int16_t* residual_coded, int coded_w,
                    int coded_h, int mapped_mode, int reduced_tx_set,
                    bool dct_only, const TxTypeDecisionConfig& config,
                    std::span<std::int32_t> levels_scratch,
                    std::span<std::int16_t> recon_scratch,
                    std::span<std::int32_t> levels_out,
                    std::span<std::int16_t> recon_out,
                    TransformStreamRate& stream, TxTypeDecision& out,
                    std::string* error) {
  out = TxTypeDecision{};
  if (error != nullptr) error->clear();
  if (!std::isfinite(config.lambda) || config.lambda < 0.0) {
    if (error != nullptr) *error = "TxTypeDecisionConfig.lambda must be >= 0";
    return false;
  }
  if (residual_coded == nullptr || coded_w <= 0 || coded_h <= 0) {
    if (error != nullptr) *error = "invalid coded residual block";
    return false;
  }
  const auto params_opt = txv::make_tx_params(coded_w, coded_h,
                                              txv::TxType::kDctDct);
  if (!params_opt.has_value()) {
    if (error != nullptr) *error = "unsupported coded transform size";
    return false;
  }
  const int area = params_opt->w * params_opt->h;
  const int coded_area = coded_w * coded_h;
  if (levels_scratch.size() < static_cast<std::size_t>(area) ||
      levels_out.size() < static_cast<std::size_t>(area) ||
      recon_scratch.size() < static_cast<std::size_t>(coded_area) ||
      recon_out.size() < static_cast<std::size_t>(coded_area)) {
    if (error != nullptr) *error = "TX decision scratch/output spans too small";
    return false;
  }

  txtype::TxTypeQuery query;
  query.coded_w = coded_w;
  query.coded_h = coded_h;
  query.mapped_mode = mapped_mode;
  query.reduced_tx_set = reduced_tx_set;
  query.dct_only = dct_only;

  txtype::TxTypeSet set;
  if (!txtype::intra_tx_candidates(query, &set)) {
    if (error != nullptr) *error = "no legal intra TX-type set for this block";
    return false;
  }

  txtype::TxTypeCostConfig eval_config;
  eval_config.lambda = 0.0;
  eval_config.rate_fn = nullptr;
  eval_config.coefficient_bits = 0.0;
  eval_config.use_static_type_cost = false;

  bool found = false;
  for (int i = 0; i < set.count; ++i) {
    const txtype::TxTypeCandidate& candidate = set.candidates[i];
    if (!config.enable_idtx_intra) {
      txv::Tx1dType row_type = txv::Tx1dType::kDct;
      txv::Tx1dType col_type = txv::Tx1dType::kDct;
      txv::tx_1d_types(candidate.type, &row_type, &col_type);
      if (row_type == txv::Tx1dType::kIdt ||
          col_type == txv::Tx1dType::kIdt) {
        ++out.candidates_rejected;
        continue;
      }
    }
    txtype::TxTypeEval eval;
    if (!txtype::evaluate_intra_tx_type(residual_coded, query, candidate,
                                        eval_config, &eval,
                                        levels_scratch.data(),
                                        recon_scratch.data())) {
      ++out.candidates_rejected;
      continue;
    }
    if (!eval.codable) {
      ++out.candidates_rejected;
      continue;
    }
    ++out.candidates_considered;

    TransformStreamBlock block;
    block.plane = 0;
    block.tx_size = candidate.params.tx_size;
    block.w = candidate.params.w;
    block.h = candidate.params.h;
    block.txb_skip_context = 0;
    block.dc_sign_context = 0;
    block.levels_scan = levels_scratch.first(static_cast<std::size_t>(area));
    block.eob = eval.eob;
    std::string stream_error;
    if (!stream.init(block, &stream_error)) {
      if (error != nullptr && error->empty()) *error = stream_error;
      ++out.candidates_rejected;
      continue;
    }
    const double coefficient_bits = stream.block_bits();
    // §5.20.8.2: eob <= 1 signals no type.  A zero EOB is txb_skip only.
    const double type_bits =
        eval.eob <= 1 ? 0.0 : txtype::static_type_bits(query, candidate);
    const double rate_bits = type_bits + coefficient_bits;
    const double cost = config.rate_aware
                            ? eval.sse + config.lambda * rate_bits
                            : eval.sse;
    if (!std::isfinite(cost) || !valid_rate(rate_bits)) {
      if (error != nullptr) *error = "TX-type candidate cost is not finite";
      ++out.candidates_rejected;
      continue;
    }

    const bool better =
        !found ||
        type_cost_better(cost, eval.sse, static_cast<int>(candidate.type),
                         out.cost, out.sse, static_cast<int>(out.candidate.type));
    if (!better) {
      continue;
    }
    found = true;
    std::copy_n(levels_scratch.begin(), area, levels_out.begin());
    std::copy_n(recon_scratch.begin(), coded_area, recon_out.begin());
    out.valid = true;
    out.candidate = candidate;
    out.eob = eval.eob;
    out.nonzero = eval.nonzero;
    out.sum_abs = eval.sum_abs;
    out.sse = eval.sse;
    out.type_bits = type_bits;
    out.coefficient_bits = coefficient_bits;
    out.rate_bits = rate_bits;
    out.cost = cost;
    out.candidate_index = i;
  }

  if (!found) {
    if (error != nullptr && error->empty()) {
      *error = "no codable intra TX type";
    }
    out = TxTypeDecision{};
    return false;
  }
  return true;
}

bool rdoq_config_valid(const RdoqConfig& config) noexcept {
  return std::isfinite(config.lambda) && config.lambda >= 0.0 &&
         std::isfinite(config.quant_step) && config.quant_step > 0.0 &&
         config.prev_level_cap >= 0 && config.prev_level_cap <= 3 &&
         config.level_cap >= 0;
}

bool decide_rdoq(std::span<const double> original_scan,
                 std::span<const std::int32_t> scalar_levels_scan,
                 int plane, int tx_size, int w, int h,
                 int txb_skip_context, int dc_sign_context,
                 const RdoqConfig& config, RdoqWorkspace& workspace,
                 TransformStreamRate& stream,
                 std::span<std::int32_t> levels_out, RdoqDecision& out,
                 std::string* error) {
  out = RdoqDecision{};
  if (error != nullptr) error->clear();
  if (!rdoq_config_valid(config)) {
    if (error != nullptr) *error = "invalid RdoqConfig";
    return false;
  }
  if (w <= 0 || h <= 0 || w * h > kMaxTransformArea) {
    if (error != nullptr) *error = "RDOQ block must be <= 32x32";
    return false;
  }
  const std::size_t count = static_cast<std::size_t>(w) *
                            static_cast<std::size_t>(h);
  if (original_scan.size() != count ||
      scalar_levels_scan.size() != count ||
      levels_out.size() < count) {
    if (error != nullptr) *error = "RDOQ span size must equal w*h";
    return false;
  }
  const int states = config.prev_level_cap + 1;

  std::int32_t scalar_eob = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(original_scan[i])) {
      if (error != nullptr) *error = "RDOQ original coefficients must be finite";
      return false;
    }
    if (scalar_levels_scan[i] != 0) {
      scalar_eob = static_cast<std::int32_t>(i) + 1;
    }
  }

  TransformStreamBlock block;
  block.plane = plane;
  block.tx_size = tx_size;
  block.w = w;
  block.h = h;
  block.txb_skip_context = txb_skip_context;
  block.dc_sign_context = dc_sign_context;
  block.levels_scan = scalar_levels_scan;
  block.eob = scalar_eob;
  std::string stream_error;
  if (!stream.init(block, &stream_error)) {
    if (error != nullptr) *error = stream_error;
    return false;
  }

  for (int i = 0; i <= static_cast<int>(count); ++i) {
    const double bits = stream.eob_bits(i);
    if (!valid_rate(bits)) {
      if (error != nullptr) *error = "stream EOB rate is invalid";
      return false;
    }
    workspace.eob_bits[static_cast<std::size_t>(i)] = bits;
  }

  workspace.suffix_zero[count] = 0.0;
  for (std::size_t i = count; i-- > 0;) {
    const double original = original_scan[i];
    workspace.suffix_zero[i] =
        workspace.suffix_zero[i + 1] + original * original;
    if (original < 0.0) {
      workspace.signs[i] = -1;
    } else if (original > 0.0) {
      workspace.signs[i] = 1;
    } else {
      const std::int32_t seed = scalar_level(original, config.quant_step);
      workspace.signs[i] = seed < 0 ? -1 : 1;
    }
  }

  const std::size_t path_cells = count * static_cast<std::size_t>(states);
  std::fill_n(workspace.chosen_level.begin(), path_cells, 0);
  std::fill_n(workspace.chosen_mode.begin(), path_cells, 0);

  std::fill(workspace.dp_next.begin(), workspace.dp_next.end(), kInfinity);
  std::fill(workspace.dd_next.begin(), workspace.dd_next.end(), 0.0);
  std::fill(workspace.dr_next.begin(), workspace.dr_next.end(), 0.0);

  std::int32_t max_level_searched = 0;
  bool truncated = false;

  for (std::size_t ii = count; ii-- > 0;) {
    const std::int32_t scan_pos = static_cast<std::int32_t>(ii);
    const double original = original_scan[ii];
    const std::int32_t sign = workspace.signs[ii];

    std::int32_t bound = 0;
    bool clipped = false;
    if (!level_bound_for(original, config, &bound, &clipped)) {
      if (error != nullptr) *error = "invalid RDOQ coefficient";
      return false;
    }
    truncated = truncated || clipped;

    const double ratio = std::fabs(original) / config.quant_step;
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
    const double zero_after = workspace.suffix_zero[ii + 1];

    for (int prev = 0; prev < states; ++prev) {
      Choice best;
      if (ii == 0) {
        best.distortion = workspace.suffix_zero[0];
        best.rate_bits = workspace.eob_bits[0];
        best.cost = best.distortion + config.lambda * best.rate_bits;
        best.signed_level = 0;
        best.abs_level = 0;
        best.mode = 0;
        best.terminal = true;
      }

      for (std::int32_t magnitude = 0; magnitude <= bound; ++magnitude) {
        const std::int32_t signed_level = magnitude == 0 ? 0 : sign * magnitude;
        const double reconstructed =
            static_cast<double>(signed_level) * config.quant_step;
        const double difference = original - reconstructed;
        const double distortion = difference * difference;
        const std::int32_t next_state = std::min(magnitude, config.prev_level_cap);
        const std::size_t next_index = static_cast<std::size_t>(next_state);

        const double normal_bits =
            stream.coefficient_bits(scan_pos, signed_level, false);
        if (!valid_rate(normal_bits)) {
          if (error != nullptr) *error = "stream coefficient rate is invalid";
          return false;
        }
        if (std::isfinite(workspace.dp_next[next_index])) {
          Choice candidate;
          candidate.cost =
              distortion + config.lambda * normal_bits +
              workspace.dp_next[next_index];
          candidate.distortion =
              distortion + workspace.dd_next[next_index];
          candidate.rate_bits =
              normal_bits + workspace.dr_next[next_index];
          candidate.signed_level = signed_level;
          candidate.abs_level = magnitude;
          candidate.mode = 1;
          candidate.terminal = false;
          if (choice_better(candidate, best)) {
            best = candidate;
          }
        }

        if (magnitude != 0) {
          const double eob_coeff_bits =
              stream.coefficient_bits(scan_pos, signed_level, true);
          if (!valid_rate(eob_coeff_bits)) {
            if (error != nullptr) *error = "stream EOB coefficient rate is invalid";
            return false;
          }
          Choice candidate;
          candidate.distortion = distortion + zero_after;
          candidate.rate_bits =
              eob_coeff_bits + workspace.eob_bits[ii + 1];
          candidate.cost =
              candidate.distortion + config.lambda * candidate.rate_bits;
          candidate.signed_level = signed_level;
          candidate.abs_level = magnitude;
          candidate.mode = 2;
          candidate.terminal = true;
          if (choice_better(candidate, best)) {
            best = candidate;
          }
        }
        max_here = std::max(max_here, magnitude);
        if (magnitude >= closest && std::isfinite(best.cost) &&
            distortion >= best.cost) {
          break;
        }
      }

      workspace.dp_cur[static_cast<std::size_t>(prev)] = best.cost;
      workspace.dd_cur[static_cast<std::size_t>(prev)] = best.distortion;
      workspace.dr_cur[static_cast<std::size_t>(prev)] = best.rate_bits;
      const std::size_t slot =
          ii * static_cast<std::size_t>(states) +
          static_cast<std::size_t>(prev);
      workspace.chosen_level[slot] = best.signed_level;
      workspace.chosen_mode[slot] = static_cast<std::uint8_t>(best.mode);
    }
    if (bound > 0 && max_here >= bound) {
      truncated = true;
    }
    max_level_searched = std::max(max_level_searched, max_here);

    workspace.dp_next.swap(workspace.dp_cur);
    workspace.dd_next.swap(workspace.dd_cur);
    workspace.dr_next.swap(workspace.dr_cur);
  }

  if (!std::isfinite(workspace.dp_next[0])) {
    if (error != nullptr) *error = "RDOQ trellis has no valid path";
    return false;
  }

  std::fill_n(levels_out.begin(), count, 0);
  std::int32_t state = 0;
  std::int32_t eob = static_cast<std::int32_t>(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t slot =
        i * static_cast<std::size_t>(states) +
        static_cast<std::size_t>(state);
    const std::int32_t level = workspace.chosen_level[slot];
    const std::int32_t mode = workspace.chosen_mode[slot];
    if (mode == 0) {
      eob = static_cast<std::int32_t>(i);
      break;
    }
    levels_out[i] = level;
    state = std::min(abs_i32(level), config.prev_level_cap);
    if (mode == 2) {
      eob = static_cast<std::int32_t>(i) + 1;
      break;
    }
  }

  double distortion = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double reconstructed =
        static_cast<double>(levels_out[i]) * config.quant_step;
    const double difference = original_scan[i] - reconstructed;
    distortion += difference * difference;
  }
  double rate_bits = workspace.eob_bits[static_cast<std::size_t>(eob)];
  for (std::int32_t i = 0; i < eob; ++i) {
    const std::int32_t level = levels_out[static_cast<std::size_t>(i)];
    const bool is_eob = (i == eob - 1 && level != 0);
    const double bits =
        stream.coefficient_bits(i, level, is_eob);
    if (!valid_rate(bits)) {
      if (error != nullptr) *error = "RDOQ rate callback returned invalid bits";
      return false;
    }
    rate_bits += bits;
  }
  const double cost = distortion + config.lambda * rate_bits;
  if (!std::isfinite(distortion) || !valid_rate(rate_bits) ||
      !std::isfinite(cost)) {
    if (error != nullptr) *error = "RDOQ result is not finite";
    return false;
  }

  out.valid = true;
  out.eob = eob;
  out.distortion = distortion;
  out.rate_bits = rate_bits;
  out.cost = cost;
  out.max_level_searched = max_level_searched;
  out.level_range_truncated = truncated;
  return true;
}

double rdoq_lambda_from_pixel_lambda(double lambda_pixel) noexcept {
  if (!std::isfinite(lambda_pixel) || lambda_pixel < 0.0) {
    return 0.0;
  }
  return lambda_pixel / 64.0;
}

}  // namespace av2enc::still::rdo_decide
