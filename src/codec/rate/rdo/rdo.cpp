#include "rdo.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::rdo {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

bool plane_args_valid(std::span<const std::uint8_t> a,
                      std::span<const std::uint8_t> b,
                      int width,
                      int height,
                      int stride) noexcept {
  if (width <= 0 || height <= 0 || stride < width) {
    return false;
  }
  const std::size_t rows = static_cast<std::size_t>(height - 1);
  const std::size_t columns = static_cast<std::size_t>(width);
  const std::size_t row_stride = static_cast<std::size_t>(stride);
  const std::size_t needed = rows * row_stride + columns;
  return a.size() >= needed && b.size() >= needed;
}

void hadamard4(int* values) noexcept {
  // In-place 4-point Walsh-Hadamard transform, unnormalised.
  const int a0 = values[0] + values[1];
  const int a1 = values[0] - values[1];
  const int a2 = values[2] + values[3];
  const int a3 = values[2] - values[3];
  values[0] = a0 + a2;
  values[1] = a1 + a3;
  values[2] = a0 - a2;
  values[3] = a1 - a3;
}

std::int32_t abs_i32(std::int32_t value) noexcept {
  if (value == std::numeric_limits<std::int32_t>::min()) {
    return std::numeric_limits<std::int32_t>::max();
  }
  return value < 0 ? -value : value;
}

std::int32_t magnitude_state(std::int32_t abs_level) noexcept {
  if (abs_level == 0) {
    return 0;
  }
  return abs_level == 1 ? 1 : 2;
}

bool valid_rate(double rate) noexcept {
  return std::isfinite(rate) && rate >= 0.0;
}

// Signed level candidates for one coefficient, ordered by magnitude. The sign
// is taken from the original coefficient; the initial level only supplies the
// magnitude to search around.
std::vector<std::int32_t> make_level_candidates(
    const RdoqCoefficient& coefficient,
    const RdoqConfig& config) {
  const std::int64_t initial = static_cast<std::int64_t>(coefficient.initial_level);
  const std::int64_t raw_abs = initial < 0 ? -initial : initial;
  const std::int64_t abs_initial =
      std::min<std::int64_t>(raw_abs,
                             static_cast<std::int64_t>(
                                 std::numeric_limits<std::int32_t>::max()));
  const std::int64_t sign =
      coefficient.original < 0.0 ? -1
                                 : (coefficient.original > 0.0 ? 1
                                                               : (initial < 0 ? -1 : 1));

  std::vector<std::int64_t> magnitudes;
  magnitudes.reserve(
      static_cast<std::size_t>(2 + 2 * config.max_level_delta));
  magnitudes.push_back(0);
  magnitudes.push_back(abs_initial);

  const std::int64_t delta = static_cast<std::int64_t>(config.max_level_delta);
  if (config.allow_level_decrease) {
    for (std::int64_t d = 1; d <= delta && abs_initial - d >= 1; ++d) {
      magnitudes.push_back(abs_initial - d);
    }
  }
  if (config.allow_level_increase && abs_initial > 0) {
    const std::int64_t maximum =
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
    for (std::int64_t d = 1; d <= delta; ++d) {
      if (abs_initial + d > maximum) {
        break;
      }
      magnitudes.push_back(abs_initial + d);
    }
  }
  if (abs_initial == 0 && config.allow_zero_to_one) {
    magnitudes.push_back(1);
  }

  std::sort(magnitudes.begin(), magnitudes.end());
  magnitudes.erase(std::unique(magnitudes.begin(), magnitudes.end()),
                   magnitudes.end());

  std::vector<std::int32_t> levels;
  levels.reserve(magnitudes.size());
  for (std::int64_t magnitude : magnitudes) {
    if (magnitude == 0) {
      levels.push_back(0);
    } else {
      levels.push_back(static_cast<std::int32_t>(sign * magnitude));
    }
  }
  return levels;
}

struct LevelChoice {
  double cost = kInfinity;
  double distortion = 0.0;
  double rate_bits = 0.0;
  std::int32_t level = 0;
  std::int32_t abs_level = 0;
  bool terminate = false;
};

// Total order for trellis choices:
//   lower cost, lower distortion, lower rate, smaller |level|, terminate first,
//   then lower signed level. Candidate iteration is sorted, so ties are stable.
bool choice_better(const LevelChoice& a, const LevelChoice& b) noexcept {
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
  if (a.terminate != b.terminate) {
    return a.terminate;
  }
  return a.level < b.level;
}

}  // namespace

// ---------------------------------------------------------------------------
// Distortion
// ---------------------------------------------------------------------------

std::uint64_t ssd_u8(std::span<const std::uint8_t> a,
                     std::span<const std::uint8_t> b,
                     int width,
                     int height,
                     int stride) noexcept {
  if (!plane_args_valid(a, b, width, height, stride)) {
    return 0;
  }
  std::uint64_t ssd = 0;
  for (int y = 0; y < height; ++y) {
    const std::size_t offset =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(stride);
    const std::uint8_t* row_a = a.data() + offset;
    const std::uint8_t* row_b = b.data() + offset;
    for (int x = 0; x < width; ++x) {
      const int difference =
          static_cast<int>(row_a[x]) - static_cast<int>(row_b[x]);
      ssd += static_cast<std::uint64_t>(difference * difference);
    }
  }
  return ssd;
}

std::uint64_t sad_u8(std::span<const std::uint8_t> a,
                     std::span<const std::uint8_t> b,
                     int width,
                     int height,
                     int stride) noexcept {
  if (!plane_args_valid(a, b, width, height, stride)) {
    return 0;
  }
  std::uint64_t sad = 0;
  for (int y = 0; y < height; ++y) {
    const std::size_t offset =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(stride);
    const std::uint8_t* row_a = a.data() + offset;
    const std::uint8_t* row_b = b.data() + offset;
    for (int x = 0; x < width; ++x) {
      sad += static_cast<std::uint64_t>(
          std::abs(static_cast<int>(row_a[x]) - static_cast<int>(row_b[x])));
    }
  }
  return sad;
}

std::uint64_t satd_u8(std::span<const std::uint8_t> a,
                      std::span<const std::uint8_t> b,
                      int width,
                      int height,
                      int stride) noexcept {
  if (!plane_args_valid(a, b, width, height, stride)) {
    return 0;
  }
  std::uint64_t satd = 0;
  for (int block_y = 0; block_y < height; block_y += 4) {
    for (int block_x = 0; block_x < width; block_x += 4) {
      int block[16] = {};
      for (int y = 0; y < 4; ++y) {
        const int source_y = block_y + y;
        if (source_y >= height) {
          continue;
        }
        const std::size_t offset =
            static_cast<std::size_t>(source_y) * static_cast<std::size_t>(stride);
        const std::uint8_t* row_a = a.data() + offset;
        const std::uint8_t* row_b = b.data() + offset;
        for (int x = 0; x < 4; ++x) {
          const int source_x = block_x + x;
          if (source_x >= width) {
            continue;
          }
          block[y * 4 + x] = static_cast<int>(row_a[source_x]) -
                             static_cast<int>(row_b[source_x]);
        }
      }

      for (int y = 0; y < 4; ++y) {
        hadamard4(&block[y * 4]);
      }
      for (int x = 0; x < 4; ++x) {
        int column[4] = {block[x], block[4 + x], block[8 + x], block[12 + x]};
        hadamard4(column);
        block[x] = column[0];
        block[4 + x] = column[1];
        block[8 + x] = column[2];
        block[12 + x] = column[3];
      }
      for (int value : block) {
        satd += static_cast<std::uint64_t>(std::abs(value));
      }
    }
  }
  return satd;
}

double compute_distortion(DistortionMetric metric,
                          std::span<const std::uint8_t> a,
                          std::span<const std::uint8_t> b,
                          int width,
                          int height,
                          int stride) noexcept {
  switch (metric) {
    case DistortionMetric::kSsd:
      return static_cast<double>(ssd_u8(a, b, width, height, stride));
    case DistortionMetric::kSad:
      return static_cast<double>(sad_u8(a, b, width, height, stride));
    case DistortionMetric::kSatd:
      return static_cast<double>(satd_u8(a, b, width, height, stride));
  }
  return 0.0;
}

double coefficient_sse(std::span<const double> original,
                       std::span<const double> reconstructed) noexcept {
  const std::size_t count = std::min(original.size(), reconstructed.size());
  double sse = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double difference = original[i] - reconstructed[i];
    sse += difference * difference;
  }
  return sse;
}

double coefficient_sse_i32(std::span<const std::int32_t> original,
                           std::span<const std::int32_t> reconstructed) noexcept {
  const std::size_t count = std::min(original.size(), reconstructed.size());
  double sse = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double difference = static_cast<double>(original[i]) -
                              static_cast<double>(reconstructed[i]);
    sse += difference * difference;
  }
  return sse;
}

// ---------------------------------------------------------------------------
// Lambda
// ---------------------------------------------------------------------------

bool lambda_params_valid(const LambdaParams& params) noexcept {
  return std::isfinite(params.step_at_qp0) && params.step_at_qp0 > 0.0 &&
         std::isfinite(params.qp_per_octave) && params.qp_per_octave > 0.0 &&
         std::isfinite(params.sse_scale) && params.sse_scale >= 0.0;
}

bool validate_lambda_params(const LambdaParams& params, std::string* error) {
  if (std::isfinite(params.step_at_qp0) && params.step_at_qp0 > 0.0) {
    // ok
  } else {
    if (error != nullptr) {
      *error = "LambdaParams.step_at_qp0 must be finite and > 0";
    }
    return false;
  }
  if (std::isfinite(params.qp_per_octave) && params.qp_per_octave > 0.0) {
    // ok
  } else {
    if (error != nullptr) {
      *error = "LambdaParams.qp_per_octave must be finite and > 0";
    }
    return false;
  }
  if (std::isfinite(params.sse_scale) && params.sse_scale >= 0.0) {
    // ok
  } else {
    if (error != nullptr) {
      *error = "LambdaParams.sse_scale must be finite and >= 0";
    }
    return false;
  }
  return true;
}

double quant_step_from_qp(int base_q_idx, LambdaParams params) noexcept {
  if (!lambda_params_valid(params)) {
    return 0.0;
  }
  const int qp = std::clamp(base_q_idx, 0, 255);
  return params.step_at_qp0 *
         std::exp2(static_cast<double>(qp) / params.qp_per_octave);
}

double lambda_sse_from_quant_step(double quant_step,
                                  double sse_scale) noexcept {
  if (!std::isfinite(quant_step) || quant_step <= 0.0 ||
      !std::isfinite(sse_scale) || sse_scale < 0.0) {
    return 0.0;
  }
  return sse_scale * quant_step * quant_step;
}

double lambda_sse_from_qp(int base_q_idx, LambdaParams params) noexcept {
  return lambda_sse_from_quant_step(quant_step_from_qp(base_q_idx, params),
                                    params.sse_scale);
}

double lambda_sad_from_qp(int base_q_idx, LambdaParams params) noexcept {
  return std::sqrt(lambda_sse_from_qp(base_q_idx, params));
}

double lambda_satd_from_qp(int base_q_idx, LambdaParams params) noexcept {
  return std::sqrt(lambda_sse_from_qp(base_q_idx, params));
}

double lambda_from_qp(int base_q_idx,
                      DistortionMetric metric,
                      LambdaParams params) noexcept {
  switch (metric) {
    case DistortionMetric::kSsd:
      return lambda_sse_from_qp(base_q_idx, params);
    case DistortionMetric::kSad:
      return lambda_sad_from_qp(base_q_idx, params);
    case DistortionMetric::kSatd:
      return lambda_satd_from_qp(base_q_idx, params);
  }
  return 0.0;
}

// ---------------------------------------------------------------------------
// Candidate selection
// ---------------------------------------------------------------------------

double lagrangian_cost(double distortion,
                       double rate_bits,
                       double lambda) noexcept {
  return distortion + lambda * rate_bits;
}

double candidate_cost(const CandidateCost& candidate, double lambda) noexcept {
  return lagrangian_cost(candidate.distortion, candidate.rate_bits, lambda);
}

bool rdo_config_valid(const RdoConfig& config) noexcept {
  return std::isfinite(config.lambda) && config.lambda >= 0.0 &&
         static_cast<std::uint8_t>(config.metric) <=
             static_cast<std::uint8_t>(DistortionMetric::kSatd) &&
         static_cast<std::uint8_t>(config.tie_break) <=
             static_cast<std::uint8_t>(TieBreak::kRateThenDistortion);
}

bool validate_rdo_config(const RdoConfig& config, std::string* error) {
  if (!std::isfinite(config.lambda) || config.lambda < 0.0) {
    if (error != nullptr) {
      *error = "RdoConfig.lambda must be finite and >= 0";
    }
    return false;
  }
  if (static_cast<std::uint8_t>(config.metric) >
      static_cast<std::uint8_t>(DistortionMetric::kSatd)) {
    if (error != nullptr) {
      *error = "RdoConfig.metric is not a valid DistortionMetric";
    }
    return false;
  }
  if (static_cast<std::uint8_t>(config.tie_break) >
      static_cast<std::uint8_t>(TieBreak::kRateThenDistortion)) {
    if (error != nullptr) {
      *error = "RdoConfig.tie_break is not a valid TieBreak";
    }
    return false;
  }
  return true;
}

RdoConfig make_rdo_config(int base_q_idx,
                          DistortionMetric metric,
                          TieBreak tie_break,
                          LambdaParams params) {
  RdoConfig config;
  config.lambda = lambda_from_qp(base_q_idx, metric, params);
  config.metric = metric;
  config.tie_break = tie_break;
  return config;
}

bool candidate_better(const CandidateScore& a,
                      const CandidateScore& b,
                      TieBreak tie_break) noexcept {
  if (a.cost != b.cost) {
    return a.cost < b.cost;
  }
  if (tie_break == TieBreak::kDistortionThenRate) {
    if (a.candidate.distortion != b.candidate.distortion) {
      return a.candidate.distortion < b.candidate.distortion;
    }
    if (a.candidate.rate_bits != b.candidate.rate_bits) {
      return a.candidate.rate_bits < b.candidate.rate_bits;
    }
  } else {
    if (a.candidate.rate_bits != b.candidate.rate_bits) {
      return a.candidate.rate_bits < b.candidate.rate_bits;
    }
    if (a.candidate.distortion != b.candidate.distortion) {
      return a.candidate.distortion < b.candidate.distortion;
    }
  }
  if (a.candidate.kind != b.candidate.kind) {
    return static_cast<std::uint8_t>(a.candidate.kind) <
           static_cast<std::uint8_t>(b.candidate.kind);
  }
  if (a.candidate.id != b.candidate.id) {
    return a.candidate.id < b.candidate.id;
  }
  if (a.candidate.aux != b.candidate.aux) {
    return a.candidate.aux < b.candidate.aux;
  }
  return a.input_index < b.input_index;
}

CandidateSelector::CandidateSelector(const RdoConfig& config) noexcept
    : config_(config) {}

bool CandidateSelector::add(const CandidateCost& candidate, std::string* error) {
  if (!rdo_config_valid(config_)) {
    ++decision_.rejected_count;
    if (error != nullptr) {
      *error = "invalid RdoConfig";
    }
    return false;
  }
  if (!std::isfinite(candidate.distortion) || candidate.distortion < 0.0 ||
      !std::isfinite(candidate.rate_bits) || candidate.rate_bits < 0.0) {
    ++decision_.rejected_count;
    if (error != nullptr) {
      *error = "candidate distortion/rate must be finite and >= 0";
    }
    return false;
  }

  CandidateScore score;
  score.candidate = candidate;
  score.cost = candidate_cost(candidate, config_.lambda);
  if (!std::isfinite(score.cost)) {
    ++decision_.rejected_count;
    if (error != nullptr) {
      *error = "candidate cost is not finite";
    }
    return false;
  }
  score.input_index = decision_.candidate_count;
  ++decision_.candidate_count;

  if (!decision_.found) {
    decision_.best = score;
    decision_.found = true;
    return true;
  }

  if (candidate_better(score, decision_.best, config_.tie_break)) {
    decision_.second = decision_.best;
    decision_.has_second = true;
    decision_.best = score;
  } else if (!decision_.has_second ||
             candidate_better(score, decision_.second, config_.tie_break)) {
    decision_.second = score;
    decision_.has_second = true;
  }
  return true;
}

void CandidateSelector::reset() noexcept { decision_ = RdoDecision{}; }

RdoDecision select_best(std::span<const CandidateCost> candidates,
                        const RdoConfig& config,
                        std::string* error) {
  CandidateSelector selector(config);
  for (const CandidateCost& candidate : candidates) {
    std::string local_error;
    if (!selector.add(candidate, &local_error)) {
      if (error != nullptr && error->empty()) {
        *error = local_error;
      }
    }
  }
  return selector.decision();
}

CandidateCost make_partition_candidate(int partition_id,
                                       double distortion,
                                       double rate_bits) {
  CandidateCost candidate;
  candidate.kind = CandidateKind::kPartition;
  candidate.id = partition_id;
  candidate.distortion = distortion;
  candidate.rate_bits = rate_bits;
  return candidate;
}

CandidateCost make_intra_candidate(int mode_id,
                                   double distortion,
                                   double rate_bits) {
  CandidateCost candidate;
  candidate.kind = CandidateKind::kIntra;
  candidate.id = mode_id;
  candidate.distortion = distortion;
  candidate.rate_bits = rate_bits;
  return candidate;
}

CandidateCost make_inter_candidate(int mode_id,
                                   int reference_index,
                                   double distortion,
                                   double rate_bits) {
  CandidateCost candidate;
  candidate.kind = CandidateKind::kInter;
  candidate.id = mode_id;
  candidate.aux = reference_index;
  candidate.distortion = distortion;
  candidate.rate_bits = rate_bits;
  return candidate;
}

CandidateCost make_transform_candidate(int tx_type_id,
                                       double distortion,
                                       double rate_bits) {
  CandidateCost candidate;
  candidate.kind = CandidateKind::kTransform;
  candidate.id = tx_type_id;
  candidate.distortion = distortion;
  candidate.rate_bits = rate_bits;
  return candidate;
}

CandidateCost make_partition_split_candidate(
    int partition_id,
    std::span<const CandidateCost> children,
    double signaling_rate_bits) {
  CandidateCost candidate = make_partition_candidate(
      partition_id, 0.0, signaling_rate_bits);
  for (const CandidateCost& child : children) {
    candidate.distortion += child.distortion;
    candidate.rate_bits += child.rate_bits;
  }
  return candidate;
}

// ---------------------------------------------------------------------------
// RDOQ
// ---------------------------------------------------------------------------

bool rdoq_config_valid(const RdoqConfig& config) noexcept {
  return std::isfinite(config.lambda) && config.lambda >= 0.0 &&
         std::isfinite(config.quant_step) && config.quant_step > 0.0 &&
         config.max_level_delta >= 0 && config.max_level_delta <= 1024 &&
         config.max_coefficients >= 0;
}

bool validate_rdoq_config(const RdoqConfig& config, std::string* error) {
  if (!std::isfinite(config.lambda) || config.lambda < 0.0) {
    if (error != nullptr) {
      *error = "RdoqConfig.lambda must be finite and >= 0";
    }
    return false;
  }
  if (!std::isfinite(config.quant_step) || config.quant_step <= 0.0) {
    if (error != nullptr) {
      *error = "RdoqConfig.quant_step must be finite and > 0";
    }
    return false;
  }
  if (config.max_level_delta < 0 || config.max_level_delta > 1024) {
    if (error != nullptr) {
      *error = "RdoqConfig.max_level_delta must be in [0, 1024]";
    }
    return false;
  }
  if (config.max_coefficients < 0) {
    if (error != nullptr) {
      *error = "RdoqConfig.max_coefficients must be >= 0";
    }
    return false;
  }
  return true;
}

double default_rdoq_coefficient_rate(void* context,
                                     std::int32_t scan_pos,
                                     std::int32_t prev_abs_level,
                                     std::int32_t abs_level,
                                     std::int32_t level_context) noexcept {
  (void)context;
  (void)scan_pos;
  (void)level_context;
  if (abs_level == 0) {
    return prev_abs_level == 0 ? 0.5 : 1.0;
  }
  const double magnitude = static_cast<double>(abs_level);
  double bits = 2.0 + std::floor(std::log2(magnitude + 1.0));
  if (prev_abs_level == 0) {
    bits += 1.0;
  }
  return bits;
}

double default_rdoq_eob_rate(void* context,
                             std::int32_t eob,
                             std::int32_t num_coefficients) noexcept {
  (void)context;
  if (num_coefficients <= 0 || eob >= num_coefficients) {
    return 0.0;
  }
  return 1.0 + std::log2(static_cast<double>(eob) + 1.0);
}

RdoqRateModel default_rdoq_rate_model() noexcept {
  RdoqRateModel model;
  model.context = nullptr;
  model.coefficient_rate = &default_rdoq_coefficient_rate;
  model.eob_rate = &default_rdoq_eob_rate;
  return model;
}

bool rdoq_quantize(std::span<const RdoqCoefficient> coefficients,
                   const RdoqConfig& config,
                   const RdoqRateModel& rate_model,
                   RdoqResult& result,
                   std::string* error) {
  result = RdoqResult{};

  if (!rdoq_config_valid(config)) {
    if (error != nullptr) {
      *error = "invalid RdoqConfig";
    }
    return false;
  }

  const std::size_t count = coefficients.size();
  if (count >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    if (error != nullptr) {
      *error = "coefficient count does not fit in int32";
    }
    return false;
  }
  const std::int32_t num_coefficients = static_cast<std::int32_t>(count);
  if (num_coefficients > config.max_coefficients) {
    if (error != nullptr) {
      *error = "coefficient count exceeds RdoqConfig.max_coefficients";
    }
    return false;
  }

  if (count == 0) {
    return true;
  }

  RdoqRateFn coefficient_rate = rate_model.coefficient_rate;
  if (coefficient_rate == nullptr) {
    coefficient_rate = &default_rdoq_coefficient_rate;
  }
  RdoqEobRateFn eob_rate = rate_model.eob_rate;
  if (eob_rate == nullptr) {
    eob_rate = &default_rdoq_eob_rate;
  }
  void* rate_context = rate_model.context;
  const double lambda = config.lambda;
  const double quant_step = config.quant_step;

  // Distortion if every remaining coefficient is forced to zero.
  std::vector<double> suffix_zero_distortion(count + 1, 0.0);
  for (std::size_t i = count; i-- > 0;) {
    const double original = coefficients[i].original;
    if (!std::isfinite(original)) {
      if (error != nullptr) {
        *error = "RdoqCoefficient.original must be finite";
      }
      return false;
    }
    suffix_zero_distortion[i] =
        suffix_zero_distortion[i + 1] + original * original;
  }

  const double eob_at_end =
      eob_rate(rate_context, num_coefficients, num_coefficients);
  if (!valid_rate(eob_at_end)) {
    if (error != nullptr) {
      *error = "RdoqEobRateFn returned an invalid value";
    }
    return false;
  }
  const double eob_at_end_cost = lambda * eob_at_end;
  if (!std::isfinite(eob_at_end_cost)) {
    if (error != nullptr) {
      *error = "RDOQ terminal EOB cost is not finite";
    }
    return false;
  }

  // State at position count: the block may only end with a non-zero last
  // coefficient (state 1 or 2). State 0 is unreachable because trailing zeros
  // would have terminated at an earlier position. These entries are costs,
  // so the terminal EOB rate is already multiplied by lambda.
  double dp_next[3] = {kInfinity, eob_at_end_cost, eob_at_end_cost};

  std::vector<std::int32_t> chosen_level(count * 3, 0);
  std::vector<std::uint8_t> terminated(count * 3, 0);

  for (std::size_t ii = count; ii-- > 0;) {
    const std::int32_t scan_pos = static_cast<std::int32_t>(ii);
    const RdoqCoefficient& coefficient = coefficients[ii];
    const double original = coefficient.original;

    const double termination_distortion = suffix_zero_distortion[ii];
    const double termination_rate =
        eob_rate(rate_context, scan_pos, num_coefficients);
    if (!valid_rate(termination_rate)) {
      if (error != nullptr) {
        *error = "RdoqEobRateFn returned an invalid value";
      }
      return false;
    }
    const double termination_cost =
        termination_distortion + lambda * termination_rate;

    const std::vector<std::int32_t> level_candidates =
        make_level_candidates(coefficient, config);

    double dp_current[3] = {kInfinity, kInfinity, kInfinity};
    for (std::int32_t state = 0; state < 3; ++state) {
      LevelChoice best;
      // EOB is canonical only at position 0 or immediately after a non-zero
      // coefficient. Terminating after a coded zero would set EOB past the
      // last non-zero coefficient and charge a redundant zero.
      if (ii == 0 || state != 0) {
        best.cost = termination_cost;
        best.distortion = termination_distortion;
        best.rate_bits = termination_rate;
        best.level = 0;
        best.abs_level = 0;
        best.terminate = true;
      }

      for (std::int32_t level : level_candidates) {
        const std::int32_t abs_level = abs_i32(level);
        const double dequantized = static_cast<double>(level) * quant_step;
        const double difference = original - dequantized;
        const double distortion = difference * difference;
        const double rate = coefficient_rate(rate_context, scan_pos, state,
                                             abs_level, coefficient.context);
        if (!valid_rate(rate)) {
          if (error != nullptr) {
            *error = "RdoqRateFn returned an invalid value";
          }
          return false;
        }
        const std::int32_t next_state = magnitude_state(abs_level);
        if (!std::isfinite(dp_next[next_state])) {
          continue;
        }

        LevelChoice choice;
        choice.cost = distortion + lambda * rate + dp_next[next_state];
        choice.distortion = distortion;
        choice.rate_bits = rate;
        choice.level = level;
        choice.abs_level = abs_level;
        choice.terminate = false;
        if (choice_better(choice, best)) {
          best = choice;
        }
      }

      dp_current[state] = best.cost;
      const std::size_t slot =
          ii * 3 + static_cast<std::size_t>(state);
      chosen_level[slot] = best.level;
      terminated[slot] = best.terminate ? 1 : 0;
    }

    for (std::int32_t state = 0; state < 3; ++state) {
      dp_next[state] = dp_current[state];
    }
  }

  if (!std::isfinite(dp_next[0])) {
    if (error != nullptr) {
      *error = "RDOQ trellis has no valid path";
    }
    return false;
  }

  result.levels.assign(count, 0);
  std::int32_t state = 0;
  std::int32_t eob = num_coefficients;
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t slot = i * 3 + static_cast<std::size_t>(state);
    if (terminated[slot] != 0) {
      eob = static_cast<std::int32_t>(i);
      break;
    }
    const std::int32_t level = chosen_level[slot];
    result.levels[i] = level;
    state = magnitude_state(abs_i32(level));
  }

  double distortion = 0.0;
  double rate_bits = 0.0;
  std::int32_t previous_abs = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const std::int32_t level = result.levels[i];
    const double dequantized = static_cast<double>(level) * quant_step;
    const double difference = coefficients[i].original - dequantized;
    distortion += difference * difference;

    if (static_cast<std::int32_t>(i) < eob) {
      const std::int32_t abs_level = abs_i32(level);
      const double rate = coefficient_rate(rate_context,
                                           static_cast<std::int32_t>(i),
                                           previous_abs,
                                           abs_level,
                                           coefficients[i].context);
      if (!valid_rate(rate)) {
        if (error != nullptr) {
          *error = "RdoqRateFn returned an invalid value";
        }
        result = RdoqResult{};
        return false;
      }
      rate_bits += rate;
      previous_abs = magnitude_state(abs_level);
    }
  }
  const double final_eob_rate =
      eob_rate(rate_context, eob, num_coefficients);
  if (!valid_rate(final_eob_rate)) {
    if (error != nullptr) {
      *error = "RdoqEobRateFn returned an invalid value";
    }
    result = RdoqResult{};
    return false;
  }
  rate_bits += final_eob_rate;
  if (!std::isfinite(distortion) || !std::isfinite(rate_bits) ||
      !valid_rate(rate_bits)) {
    if (error != nullptr) {
      *error = "RDOQ result is not finite";
    }
    result = RdoqResult{};
    return false;
  }

  result.eob = eob;
  result.distortion = distortion;
  result.rate_bits = rate_bits;
  result.cost = distortion + lambda * rate_bits;
  if (!std::isfinite(result.cost)) {
    if (error != nullptr) {
      *error = "RDOQ cost is not finite";
    }
    result = RdoqResult{};
    return false;
  }
  return true;
}

}  // namespace av2enc::rdo
