#include "rc.h"

#include "rc_tables.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace av2enc::rc {
namespace {

constexpr double kMaxEstimateBits = 1.0e12;
constexpr double kMaxActualBits = 1.0e15;
constexpr double kQpOctave = 24.0;  // qlookup doubles about every 24 qindex steps

void set_error(std::string* error, const char* message) {
  if (error != nullptr) {
    *error = message;
  }
}

bool finite_nonnegative(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

int nearest_reduced_delta(int diff, int step) noexcept {
  if (diff >= 0) {
    return (diff + step / 2) / step;
  }
  return -((-diff + step / 2) / step);
}

bool select_vbr_qindex(const BitrateEstimator& estimator,
                       const FrameComplexity& complexity,
                       int min_qindex,
                       int max_qindex,
                       double target_bits,
                       int& qindex,
                       std::string* error) {
  double at_min = 0.0;
  if (!estimate_frame_bits(estimator, complexity, min_qindex, at_min, error)) {
    return false;
  }
  if (at_min <= target_bits) {
    qindex = min_qindex;
    return true;
  }

  double at_max = 0.0;
  if (!estimate_frame_bits(estimator, complexity, max_qindex, at_max, error)) {
    return false;
  }
  if (at_max > target_bits) {
    // Even the coarsest quantizer exceeds the budget.
    qindex = max_qindex;
    return true;
  }

  // Invariant: estimate(lo) > target and estimate(hi) <= target. Find the
  // smallest qindex that fits.
  int lo = min_qindex;
  int hi = max_qindex;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    double bits = 0.0;
    if (!estimate_frame_bits(estimator, complexity, mid, bits, error)) {
      return false;
    }
    if (bits <= target_bits) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  qindex = lo;
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Quantizer index domain, slices and coefficient-CDF contexts
// ---------------------------------------------------------------------------

int clamp_qindex(int qindex) noexcept {
  return std::clamp(qindex, 0, kMaxQIndex);
}

int clamp_base_qindex(int qindex) noexcept {
  return std::clamp(qindex, kMinQIndex, kMaxQIndex);
}

double quant_step_from_qindex(int qindex) noexcept {
  return static_cast<double>(kQlookupQ0To255[clamp_qindex(qindex)]);
}

int qindex_from_qp(double qp) noexcept {
  if (!std::isfinite(qp)) {
    return 0;
  }
  const double clamped = std::clamp(qp, 0.0, static_cast<double>(kMaxQIndex));
  return static_cast<int>(std::floor(clamped + 0.5));
}

double lambda_sse_from_qindex(int qindex, double sse_scale) noexcept {
  if (qindex <= kLosslessQIndex || !finite_nonnegative(sse_scale)) {
    return 0.0;
  }
  const double step = quant_step_from_qindex(qindex);
  return sse_scale * step * step;
}

int coeff_cdf_qctx(int base_q_idx) noexcept {
  const int q = clamp_qindex(base_q_idx);
  if (q <= 90) {
    return 0;
  }
  if (q <= 140) {
    return 1;
  }
  if (q <= 190) {
    return 2;
  }
  return 3;
}

const QpSlice& coeff_cdf_slice(int base_q_idx) noexcept {
  return kCoeffCdfQpSlices[static_cast<std::size_t>(coeff_cdf_qctx(base_q_idx))];
}

// ---------------------------------------------------------------------------
// Frame complexity and bitrate estimation
// ---------------------------------------------------------------------------

bool frame_complexity_valid(const FrameComplexity& complexity,
                            std::string* error) {
  if (complexity.width <= 0 || complexity.height <= 0) {
    set_error(error, "frame complexity: width and height must be positive");
    return false;
  }
  if (!finite_nonnegative(complexity.residual_mad) ||
      !finite_nonnegative(complexity.residual_mse) ||
      !std::isfinite(complexity.intra_fraction)) {
    set_error(error, "frame complexity: statistics must be finite and >= 0");
    return false;
  }
  if (complexity.intra_fraction < 0.0 || complexity.intra_fraction > 1.0) {
    set_error(error, "frame complexity: intra_fraction must be in [0, 1]");
    return false;
  }
  return true;
}

double frame_complexity_sigma(const FrameComplexity& complexity,
                              double mad_to_sigma) noexcept {
  if (!frame_complexity_valid(complexity) || !std::isfinite(mad_to_sigma) ||
      mad_to_sigma <= 0.0) {
    return 0.0;
  }
  if (complexity.residual_mse > 0.0) {
    return std::sqrt(complexity.residual_mse);
  }
  return complexity.residual_mad * mad_to_sigma;
}

FrameComplexity complexity_from_u8_planes(std::span<const std::uint8_t> original,
                                          std::span<const std::uint8_t> predicted,
                                          int width,
                                          int height,
                                          int stride) noexcept {
  FrameComplexity result;
  if (width <= 0 || height <= 0 || stride < width) {
    return result;
  }
  const std::size_t needed =
      static_cast<std::size_t>(height - 1) * static_cast<std::size_t>(stride) +
      static_cast<std::size_t>(width);
  if (original.size() < needed || predicted.size() < needed) {
    return result;
  }

  std::uint64_t sum_abs = 0;
  std::uint64_t sum_sq = 0;
  for (int y = 0; y < height; ++y) {
    const std::size_t row =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(stride);
    for (int x = 0; x < width; ++x) {
      const int a = static_cast<int>(original[row + static_cast<std::size_t>(x)]);
      const int b =
          static_cast<int>(predicted[row + static_cast<std::size_t>(x)]);
      const int diff = a - b;
      sum_abs += static_cast<std::uint64_t>(diff < 0 ? -diff : diff);
      sum_sq += static_cast<std::uint64_t>(diff * diff);
    }
  }

  const double count =
      static_cast<double>(width) * static_cast<double>(height);
  result.width = width;
  result.height = height;
  result.residual_mad = static_cast<double>(sum_abs) / count;
  result.residual_mse = static_cast<double>(sum_sq) / count;
  result.intra_fraction = 0.0;
  return result;
}

bool bitrate_model_params_valid(const BitrateModelParams& params,
                                std::string* error) {
  if (!finite_nonnegative(params.header_bits) ||
      !finite_nonnegative(params.coefficient_bits_per_sample) ||
      !finite_nonnegative(params.max_bits_per_pixel) ||
      !finite_nonnegative(params.mad_to_sigma)) {
    set_error(error, "bitrate model: parameters must be finite and >= 0");
    return false;
  }
  if (params.max_bits_per_pixel <= 0.0) {
    set_error(error, "bitrate model: max_bits_per_pixel must be > 0");
    return false;
  }
  if (params.mad_to_sigma <= 0.0) {
    set_error(error, "bitrate model: mad_to_sigma must be > 0");
    return false;
  }
  return true;
}

double estimate_frame_bits_model(const FrameComplexity& complexity,
                                 int base_q_idx,
                                 const BitrateModelParams& params) noexcept {
  if (!frame_complexity_valid(complexity) ||
      !bitrate_model_params_valid(params)) {
    return 0.0;
  }
  const int q = clamp_base_qindex(base_q_idx);
  const double step = quant_step_from_qindex(q);
  const double sigma = frame_complexity_sigma(complexity, params.mad_to_sigma);
  const double ratio = sigma / step;
  double bits_per_sample =
      params.coefficient_bits_per_sample * std::log2(1.0 + ratio);
  if (!std::isfinite(bits_per_sample) || bits_per_sample < 0.0) {
    bits_per_sample = 0.0;
  }
  bits_per_sample = std::min(bits_per_sample, params.max_bits_per_pixel);

  const double bits =
      params.header_bits +
      static_cast<double>(complexity.pixel_count()) * bits_per_sample;
  if (!std::isfinite(bits) || bits < 0.0) {
    return 0.0;
  }
  return bits;
}

bool bitrate_estimator_valid(const BitrateEstimator& estimator,
                             std::string* error) {
  return bitrate_model_params_valid(estimator.params, error);
}

bool estimate_frame_bits(const BitrateEstimator& estimator,
                         const FrameComplexity& complexity,
                         int base_q_idx,
                         double& bits,
                         std::string* error) {
  bits = 0.0;
  if (!frame_complexity_valid(complexity, error)) {
    return false;
  }
  if (!bitrate_estimator_valid(estimator, error)) {
    return false;
  }
  const int q = clamp_qindex(base_q_idx);
  double value = 0.0;
  if (estimator.estimate != nullptr) {
    value = estimator.estimate(estimator.context, complexity, q);
  } else {
    value = estimate_frame_bits_model(complexity, q, estimator.params);
  }
  if (!std::isfinite(value) || value < 0.0 || value > kMaxEstimateBits) {
    set_error(error, "bitrate estimator returned an invalid bit count");
    return false;
  }
  bits = value;
  return true;
}

// ---------------------------------------------------------------------------
// Segmentation
// ---------------------------------------------------------------------------

SegmentationParams disabled_segmentation() noexcept {
  return SegmentationParams{};
}

bool segmentation_params_valid(const SegmentationParams& params,
                               std::string* error) {
  if (!params.enabled) {
    if (params.update_map || params.temporal_update) {
      set_error(error, "segmentation: disabled params must not update the map");
      return false;
    }
    for (int i = 0; i < kMaxSegments; ++i) {
      for (int j = 0; j < kSegLvlMax; ++j) {
        if (params.segments[static_cast<std::size_t>(i)]
                .enabled[static_cast<std::size_t>(j)] ||
            params.segments[static_cast<std::size_t>(i)]
                    .data[static_cast<std::size_t>(j)] != 0) {
          set_error(error,
                    "segmentation: disabled params must not carry features");
          return false;
        }
      }
    }
    return true;
  }

  if (params.temporal_update && !params.update_map) {
    set_error(error, "segmentation: temporal_update requires update_map");
    return false;
  }

  for (int i = 0; i < kMaxSegments; ++i) {
    const SegmentFeatures& segment =
        params.segments[static_cast<std::size_t>(i)];
    for (int j = 0; j < kSegLvlMax; ++j) {
      const bool enabled = segment.enabled[static_cast<std::size_t>(j)];
      const std::int32_t data = segment.data[static_cast<std::size_t>(j)];
      if (!enabled) {
        if (data != 0) {
          set_error(error,
                    "segmentation: disabled feature must have zero data");
          return false;
        }
        continue;
      }
      if (j == kSegLvlAltQ) {
        if (data < -kSegAltQLimit || data > kSegAltQLimit) {
          set_error(error, "segmentation: ALT_Q data out of [-351, 351]");
          return false;
        }
      } else if (data != 0) {
        set_error(error, "segmentation: SKIP/GLOBALMV data must be zero");
        return false;
      }
    }
  }
  return true;
}

bool set_segment_alt_q(SegmentationParams& params,
                       int segment_id,
                       int delta_q,
                       std::string* error) {
  if (segment_id < 0 || segment_id >= kMaxSegments) {
    set_error(error, "segmentation: segment id out of range");
    return false;
  }
  if (delta_q < -kSegAltQLimit || delta_q > kSegAltQLimit) {
    set_error(error, "segmentation: ALT_Q data out of [-351, 351]");
    return false;
  }
  params.enabled = true;
  SegmentFeatures& segment = params.segments[static_cast<std::size_t>(segment_id)];
  segment.enabled[static_cast<std::size_t>(kSegLvlAltQ)] = true;
  segment.data[static_cast<std::size_t>(kSegLvlAltQ)] =
      static_cast<std::int32_t>(delta_q);
  return true;
}

// ---------------------------------------------------------------------------
// Delta-Q
// ---------------------------------------------------------------------------

bool delta_q_params_valid(const DeltaQParams& params, std::string* error) {
  if (params.res < 0 || params.res > 3) {
    set_error(error, "delta_q: res must be in [0, 3]");
    return false;
  }
  if (!params.present && params.res != 0) {
    set_error(error, "delta_q: absent params must have res == 0");
    return false;
  }
  return true;
}

bool qp_context_valid(const QpContext& context, std::string* error) {
  if (context.base_q_idx < 0 || context.base_q_idx > kMaxQIndex) {
    set_error(error, "qp context: base_q_idx must be in [0, 255]");
    return false;
  }
  if (!delta_q_params_valid(context.delta_q, error)) {
    return false;
  }
  if (!segmentation_params_valid(context.segmentation, error)) {
    return false;
  }
  if (context.current_q_index < 0 || context.current_q_index > kMaxQIndex) {
    set_error(error, "qp context: current_q_index must be in [0, 255]");
    return false;
  }
  if (context.delta_q.present && context.base_q_idx < kMinQIndex) {
    set_error(error, "qp context: delta_q requires base_q_idx >= 1");
    return false;
  }
  if (context.delta_q.present && context.current_q_index < kMinQIndex) {
    set_error(error, "qp context: delta_q requires current_q_index >= 1");
    return false;
  }
  return true;
}

bool effective_qindex(const QpContext& context,
                      int segment_id,
                      bool ignore_delta_q,
                      int& qindex,
                      std::string* error) {
  qindex = context.base_q_idx;
  if (!qp_context_valid(context, error)) {
    return false;
  }
  if (segment_id < 0 || segment_id >= kMaxSegments) {
    set_error(error, "qp context: segment id out of range");
    return false;
  }

  const SegmentFeatures& segment =
      context.segmentation.segments[static_cast<std::size_t>(segment_id)];
  const bool alt_q_active =
      context.segmentation.enabled &&
      segment.enabled[static_cast<std::size_t>(kSegLvlAltQ)];

  int value = context.base_q_idx;
  if (alt_q_active) {
    const int data =
        static_cast<int>(segment.data[static_cast<std::size_t>(kSegLvlAltQ)]);
    if (!ignore_delta_q && context.delta_q.present) {
      value = context.current_q_index + data;
    } else {
      value = context.base_q_idx + data;
    }
  } else if (!ignore_delta_q && context.delta_q.present) {
    value = context.current_q_index;
  }
  qindex = clamp_qindex(value);
  return true;
}

bool delta_q_reset(int base_q_idx,
                   DeltaQState& state,
                   std::string* error) {
  if (base_q_idx < kMinQIndex || base_q_idx > kMaxQIndex) {
    set_error(error, "delta_q: base_q_idx must be in [1, 255]");
    return false;
  }
  state.current_q_index = base_q_idx;
  return true;
}

bool delta_q_apply(const DeltaQParams& params,
                   int reduced_delta,
                   DeltaQState& state,
                   std::string* error) {
  if (!delta_q_params_valid(params, error)) {
    return false;
  }
  if (!params.present) {
    set_error(error, "delta_q: apply called with delta_q_present == 0");
    return false;
  }
  if (state.current_q_index < kMinQIndex ||
      state.current_q_index > kMaxQIndex) {
    set_error(error, "delta_q: state is not initialized");
    return false;
  }
  const std::int64_t offset =
      static_cast<std::int64_t>(reduced_delta) << params.res;
  const std::int64_t next =
      static_cast<std::int64_t>(state.current_q_index) + offset;
  state.current_q_index = std::clamp(
      static_cast<int>(std::clamp<std::int64_t>(
          next, static_cast<std::int64_t>(kMinQIndex),
          static_cast<std::int64_t>(kMaxQIndex))),
      kMinQIndex, kMaxQIndex);
  return true;
}

bool delta_q_plan_config_valid(const DeltaQPlanConfig& config,
                               std::string* error) {
  if (config.base_q_idx < kMinQIndex || config.base_q_idx > kMaxQIndex) {
    set_error(error, "delta_q plan: base_q_idx must be in [1, 255]");
    return false;
  }
  if (config.delta_q_res < 0 || config.delta_q_res > 3) {
    set_error(error, "delta_q plan: res must be in [0, 3]");
    return false;
  }
  if (config.cols <= 0 || config.rows <= 0) {
    set_error(error, "delta_q plan: cols and rows must be positive");
    return false;
  }
  const std::int64_t cells =
      static_cast<std::int64_t>(config.cols) * static_cast<std::int64_t>(config.rows);
  if (cells > 1000000) {
    set_error(error, "delta_q plan: grid is too large");
    return false;
  }
  return true;
}

void DeltaQPlan::clear() noexcept {
  steps.clear();
  max_abs_reduced_delta = 0;
  final_qindex = 0;
}

bool plan_delta_q(const DeltaQPlanConfig& config,
                  std::span<const int> target_qindex,
                  DeltaQPlan& plan,
                  std::string* error) {
  plan.clear();
  if (!delta_q_plan_config_valid(config, error)) {
    return false;
  }
  const std::size_t cells =
      static_cast<std::size_t>(config.cols) * static_cast<std::size_t>(config.rows);
  if (target_qindex.size() != cells) {
    set_error(error, "delta_q plan: target size does not match cols * rows");
    return false;
  }
  for (const int target : target_qindex) {
    if (target < kMinQIndex || target > kMaxQIndex) {
      set_error(error, "delta_q plan: targets must be in [1, 255]");
      return false;
    }
  }

  const int step = 1 << config.delta_q_res;
  int current = config.base_q_idx;
  plan.steps.reserve(cells);
  for (int y = 0; y < config.rows; ++y) {
    for (int x = 0; x < config.cols; ++x) {
      const std::size_t index =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(config.cols) +
          static_cast<std::size_t>(x);
      const int desired = target_qindex[index];
      const int diff = desired - current;
      const int reduced = nearest_reduced_delta(diff, step);
      const std::int64_t raw_next =
          static_cast<std::int64_t>(current) +
          static_cast<std::int64_t>(reduced) * static_cast<std::int64_t>(step);
      const int next = std::clamp(
          static_cast<int>(std::clamp<std::int64_t>(
              raw_next, static_cast<std::int64_t>(kMinQIndex),
              static_cast<std::int64_t>(kMaxQIndex))),
          kMinQIndex, kMaxQIndex);

      DeltaQPlanStep entry;
      entry.sb_x = x;
      entry.sb_y = y;
      entry.reduced_delta = reduced;
      entry.qindex_before = current;
      entry.qindex_after = next;
      plan.steps.push_back(entry);
      plan.max_abs_reduced_delta =
          std::max(plan.max_abs_reduced_delta, reduced < 0 ? -reduced : reduced);
      current = next;
    }
  }
  plan.final_qindex = current;
  return true;
}

bool segmentation_map_valid(const SegmentationMap& map, std::string* error) {
  if (map.mi_cols <= 0 || map.mi_rows <= 0) {
    set_error(error, "segmentation map: dimensions must be positive");
    return false;
  }
  const std::size_t cells =
      static_cast<std::size_t>(map.mi_cols) * static_cast<std::size_t>(map.mi_rows);
  if (map.segment_id.size() != cells) {
    set_error(error, "segmentation map: size does not match dimensions");
    return false;
  }
  for (const std::uint8_t id : map.segment_id) {
    if (static_cast<int>(id) >= kMaxSegments) {
      set_error(error, "segmentation map: segment id out of range");
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Per-frame rate control
// ---------------------------------------------------------------------------

bool rate_control_config_valid(const RateControlConfig& config,
                               std::string* error) {
  switch (config.mode) {
    case RateControlMode::kCqp:
    case RateControlMode::kCrf:
    case RateControlMode::kVbr:
      break;
    default:
      set_error(error, "rate control: unknown mode");
      return false;
  }

  if (!bitrate_estimator_valid(config.estimator, error)) {
    return false;
  }
  if (!finite_nonnegative(config.lambda_sse_scale)) {
    set_error(error, "rate control: lambda_sse_scale must be finite and >= 0");
    return false;
  }

  if (config.cqp.base_q_idx < kMinQIndex ||
      config.cqp.base_q_idx > kMaxQIndex) {
    set_error(error, "rate control: cqp base_q_idx must be in [1, 255]");
    return false;
  }

  const CrfParams& crf = config.crf;
  if (!std::isfinite(crf.crf) || crf.crf < 0.0 || crf.crf > 63.0) {
    set_error(error, "rate control: crf must be in [0, 63]");
    return false;
  }
  if (!std::isfinite(crf.qindex_at_crf0) ||
      !std::isfinite(crf.qindex_at_crf63) ||
      crf.qindex_at_crf0 < 0.0 || crf.qindex_at_crf0 > kMaxQIndex ||
      crf.qindex_at_crf63 < 0.0 || crf.qindex_at_crf63 > kMaxQIndex ||
      crf.qindex_at_crf0 > crf.qindex_at_crf63) {
    set_error(error,
              "rate control: crf qindex endpoints must satisfy 0 <= q0 <= q63 <= 255");
    return false;
  }
  if (!finite_nonnegative(crf.qcomp) ||
      !finite_nonnegative(crf.max_qindex_adjust)) {
    set_error(error, "rate control: crf qcomp/max_qindex_adjust must be >= 0");
    return false;
  }
  if (crf.qcomp > 0.0 &&
      (!std::isfinite(crf.complexity_reference) ||
       crf.complexity_reference <= 0.0)) {
    set_error(error,
              "rate control: crf complexity_reference must be > 0 when qcomp > 0");
    return false;
  }

  const VbrParams& vbr = config.vbr;
  if (!std::isfinite(vbr.target_bitrate_bps) || vbr.target_bitrate_bps <= 0.0) {
    set_error(error, "rate control: vbr target_bitrate_bps must be > 0");
    return false;
  }
  if (vbr.frame_rate_num <= 0 || vbr.frame_rate_den <= 0) {
    set_error(error, "rate control: vbr frame rate must be positive");
    return false;
  }
  if (vbr.min_qindex < kMinQIndex || vbr.max_qindex > kMaxQIndex ||
      vbr.min_qindex > vbr.max_qindex) {
    set_error(error, "rate control: vbr qindex range is invalid");
    return false;
  }
  if (vbr.max_qindex_step < 1) {
    set_error(error, "rate control: vbr max_qindex_step must be >= 1");
    return false;
  }
  if (!std::isfinite(vbr.buffer_frames) || vbr.buffer_frames <= 0.0 ||
      !finite_nonnegative(vbr.buffer_gain) ||
      !std::isfinite(vbr.bits_ema_weight) || vbr.bits_ema_weight < 0.0 ||
      vbr.bits_ema_weight > 1.0) {
    set_error(error, "rate control: vbr buffer/EMA parameters are invalid");
    return false;
  }
  return true;
}

bool RateController::configure(const RateControlConfig& config,
                               std::string* error) {
  if (!rate_control_config_valid(config, error)) {
    return false;
  }
  config_ = config;
  configured_ = true;
  reset();
  return true;
}

void RateController::reset() noexcept {
  frame_open_ = false;
  previous_qindex_ = 0;
  bits_ema_ = 0.0;
  buffer_bits_ = 0.0;
  frame_count_ = 0;
  last_decision_ = QpDecision{};
}

bool RateController::begin_frame(const FrameComplexity& complexity,
                                 QpDecision& decision,
                                 std::string* error) {
  if (!configured_) {
    set_error(error, "rate control: controller is not configured");
    return false;
  }
  if (frame_open_) {
    set_error(error, "rate control: end_frame() must precede the next begin_frame()");
    return false;
  }
  if (!frame_complexity_valid(complexity, error)) {
    return false;
  }

  QpDecision out;
  out.continuous_qp = static_cast<double>(config_.cqp.base_q_idx);

  switch (config_.mode) {
    case RateControlMode::kCqp: {
      out.base_q_idx = clamp_base_qindex(config_.cqp.base_q_idx);
      out.continuous_qp = static_cast<double>(out.base_q_idx);
      break;
    }
    case RateControlMode::kCrf: {
      const CrfParams& crf = config_.crf;
      const double span = crf.qindex_at_crf63 - crf.qindex_at_crf0;
      double raw = crf.qindex_at_crf0 + span * (crf.crf / 63.0);
      if (crf.qcomp > 0.0) {
        const double sigma =
            frame_complexity_sigma(complexity,
                                   config_.estimator.params.mad_to_sigma);
        const double ratio = std::max(sigma, 1.0e-9) / crf.complexity_reference;
        double adjust = crf.qcomp * kQpOctave * std::log2(ratio);
        adjust = std::clamp(adjust, -crf.max_qindex_adjust,
                            crf.max_qindex_adjust);
        raw += adjust;
      }
      if (!std::isfinite(raw)) {
        set_error(error, "rate control: crf produced a non-finite qindex");
        return false;
      }
      raw = std::clamp(raw, static_cast<double>(kMinQIndex),
                       static_cast<double>(kMaxQIndex));
      out.continuous_qp = raw;
      out.base_q_idx = clamp_base_qindex(qindex_from_qp(raw));
      break;
    }
    case RateControlMode::kVbr: {
      const VbrParams& vbr = config_.vbr;
      const double target_bits =
          vbr.target_bitrate_bps *
          static_cast<double>(vbr.frame_rate_den) /
          static_cast<double>(vbr.frame_rate_num);
      double adjusted_target = target_bits + vbr.buffer_gain * buffer_bits_;
      adjusted_target = std::clamp(adjusted_target, 0.25 * target_bits,
                                   2.0 * target_bits);

      int qindex = vbr.min_qindex;
      if (!select_vbr_qindex(config_.estimator, complexity, vbr.min_qindex,
                             vbr.max_qindex, adjusted_target, qindex, error)) {
        return false;
      }

      bool clamped = false;
      if (previous_qindex_ != 0 && vbr.max_qindex_step < kMaxQIndex) {
        const int lo = std::max(vbr.min_qindex,
                                previous_qindex_ - vbr.max_qindex_step);
        const int hi = std::min(vbr.max_qindex,
                                previous_qindex_ + vbr.max_qindex_step);
        const int limited = std::clamp(qindex, lo, hi);
        clamped = limited != qindex;
        qindex = limited;
      }
      out.base_q_idx = clamp_base_qindex(qindex);
      out.continuous_qp = static_cast<double>(out.base_q_idx);
      out.clamped_by_step = clamped;

      double final_bits = 0.0;
      if (!estimate_frame_bits(config_.estimator, complexity, out.base_q_idx,
                               final_bits, error)) {
        return false;
      }
      out.target_reachable = final_bits <= adjusted_target;
      out.estimated_bits = final_bits;
      break;
    }
    default:
      set_error(error, "rate control: unknown mode");
      return false;
  }

  if (config_.mode != RateControlMode::kVbr) {
    if (!estimate_frame_bits(config_.estimator, complexity, out.base_q_idx,
                             out.estimated_bits, error)) {
      return false;
    }
    out.target_reachable = true;
    out.clamped_by_step = false;
  }

  out.coeff_cdf_qctx = coeff_cdf_qctx(out.base_q_idx);
  out.qp_slice = out.coeff_cdf_qctx;
  out.lambda_sse = lambda_sse_from_qindex(out.base_q_idx,
                                          config_.lambda_sse_scale);

  last_decision_ = out;
  frame_open_ = true;
  decision = out;
  return true;
}

bool RateController::end_frame(std::int64_t actual_bits, std::string* error) {
  if (!configured_) {
    set_error(error, "rate control: controller is not configured");
    return false;
  }
  if (!frame_open_) {
    set_error(error, "rate control: end_frame() called without begin_frame()");
    return false;
  }
  if (actual_bits < 0 ||
      static_cast<double>(actual_bits) > kMaxActualBits) {
    set_error(error, "rate control: actual_bits is out of range");
    return false;
  }

  const double actual = static_cast<double>(actual_bits);
  if (frame_count_ == 0) {
    bits_ema_ = actual;
  } else {
    const double weight = config_.vbr.bits_ema_weight;
    bits_ema_ = weight * actual + (1.0 - weight) * bits_ema_;
  }

  if (config_.mode == RateControlMode::kVbr) {
    const VbrParams& vbr = config_.vbr;
    const double target_bits =
        vbr.target_bitrate_bps * static_cast<double>(vbr.frame_rate_den) /
        static_cast<double>(vbr.frame_rate_num);
    buffer_bits_ += target_bits - actual;
    const double cap = vbr.buffer_frames * target_bits;
    buffer_bits_ = std::clamp(buffer_bits_, -cap, cap);
  }

  ++frame_count_;
  previous_qindex_ = last_decision_.base_q_idx;
  frame_open_ = false;
  return true;
}

}  // namespace av2enc::rc
