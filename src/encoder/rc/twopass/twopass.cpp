#include "twopass.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

namespace av2enc::rc::twopass {
namespace {

constexpr double kSigmaFloor = 1.0e-6;
constexpr double kWeightFloor = 1.0e-9;
constexpr double kBitEpsilon = 1.0e-6;
constexpr double kMaxActualBits = 1.0e15;
constexpr int kMaxSpatialAttempts = 12;

void set_error(std::string* error, const char* message) {
  if (error != nullptr) {
    *error = message;
  }
}

bool finite_nonnegative(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

bool finite_positive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

int round_half_away(double value) noexcept {
  if (!std::isfinite(value)) {
    return 0;
  }
  const double rounded = value >= 0.0 ? std::floor(value + 0.5)
                                      : std::ceil(value - 0.5);
  if (rounded <= static_cast<double>(std::numeric_limits<int>::min())) {
    return std::numeric_limits<int>::min();
  }
  if (rounded >= static_cast<double>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(rounded);
}

int clamp_qindex_range(int value, const TwoPassConfig& config) noexcept {
  return std::clamp(value, config.min_qindex, config.max_qindex);
}

bool estimate_bits(const BitrateEstimator& estimator,
                   const FrameComplexity& complexity,
                   int qindex,
                   double& bits,
                   std::string* error) {
  return estimate_frame_bits(estimator, complexity, qindex, bits, error);
}

bool validate_vbv_config(const VbvConfig& config, std::string* error) {
  if (!finite_nonnegative(config.buffer_size_bits)) {
    set_error(error, "twopass: vbv buffer_size_bits must be finite and >= 0");
    return false;
  }
  if (!finite_positive(config.buffer_frames)) {
    set_error(error, "twopass: vbv buffer_frames must be > 0");
    return false;
  }
  if (!std::isfinite(config.initial_fill) || config.initial_fill < 0.0 ||
      config.initial_fill > 1.0) {
    set_error(error, "twopass: vbv initial_fill must be in [0, 1]");
    return false;
  }
  if (!finite_nonnegative(config.max_rate_bps)) {
    set_error(error, "twopass: vbv max_rate_bps must be finite and >= 0");
    return false;
  }
  if (!std::isfinite(config.underflow_margin) ||
      config.underflow_margin < 0.0 || config.underflow_margin >= 0.5 ||
      !std::isfinite(config.overflow_margin) ||
      config.overflow_margin < 0.0 || config.overflow_margin >= 0.5) {
    set_error(error, "twopass: vbv margins must be in [0, 0.5)");
    return false;
  }
  if (config.underflow_margin + config.overflow_margin >= 1.0) {
    set_error(error, "twopass: vbv margins leave no usable buffer");
    return false;
  }
  if (!config.enabled) {
    return true;
  }
  return true;
}

int max_segment_classes(bool enable_ext_seg) noexcept {
  return enable_ext_seg ? 15 : 7;
}

// ---------------------------------------------------------------------------
// First-pass normalization
// ---------------------------------------------------------------------------

bool normalize_frame_stats_impl(const FrameStats& input,
                                FrameStats& normalized,
                                std::string* error) {
  normalized = input;
  if (!frame_complexity_valid(normalized.complexity, error)) {
    return false;
  }
  if (normalized.sb_size != 64 && normalized.sb_size != 128 &&
      normalized.sb_size != 256) {
    set_error(error, "twopass: sb_size must be 64, 128 or 256");
    return false;
  }
  if (!finite_nonnegative(normalized.first_pass_bits)) {
    set_error(error, "twopass: first_pass_bits must be finite and >= 0");
    return false;
  }

  const int sbx =
      (normalized.complexity.width + normalized.sb_size - 1) /
      normalized.sb_size;
  const int sby =
      (normalized.complexity.height + normalized.sb_size - 1) /
      normalized.sb_size;

  if (normalized.tiles.empty()) {
    TileStats tile;
    tile.sb_cols = sbx;
    tile.sb_rows = sby;
    tile.sb_col_start = 0;
    tile.sb_row_start = 0;
    tile.complexity = normalized.complexity;
    normalized.tiles.push_back(std::move(tile));
    return true;
  }

  // Tiles must cover the frame's superblock grid exactly: no gaps, no
  // overlaps, no out-of-frame blocks. That is what a real tile layout is.
  std::vector<int> coverage(
      static_cast<std::size_t>(sbx) * static_cast<std::size_t>(sby), -1);
  for (std::size_t t = 0; t < normalized.tiles.size(); ++t) {
    TileStats& tile = normalized.tiles[t];
    if (tile.sb_cols <= 0 || tile.sb_rows <= 0) {
      set_error(error, "twopass: tile sb_cols/sb_rows must be positive");
      return false;
    }
    if (tile.sb_col_start < 0 || tile.sb_row_start < 0 ||
        tile.sb_col_start + tile.sb_cols > sbx ||
        tile.sb_row_start + tile.sb_rows > sby) {
      set_error(error, "twopass: tile superblock extent is outside the frame");
      return false;
    }
    std::string tile_error;
    if (!frame_complexity_valid(tile.complexity, &tile_error)) {
      // Allow an unset tile complexity and derive a uniform one from the
      // frame. A half-set record (geometry but non-finite stats) is rejected
      // with the original validation message.
      if (tile.complexity.width == 0 && tile.complexity.height == 0 &&
          tile.complexity.residual_mad == 0.0 &&
          tile.complexity.residual_mse == 0.0 &&
          tile.complexity.intra_fraction == 0.0) {
        const int px0 = tile.sb_col_start * normalized.sb_size;
        const int py0 = tile.sb_row_start * normalized.sb_size;
        const int px1 = std::min(normalized.complexity.width,
                                 (tile.sb_col_start + tile.sb_cols) *
                                     normalized.sb_size);
        const int py1 = std::min(normalized.complexity.height,
                                 (tile.sb_row_start + tile.sb_rows) *
                                     normalized.sb_size);
        tile.complexity.width = std::max(0, px1 - px0);
        tile.complexity.height = std::max(0, py1 - py0);
        if (tile.complexity.width <= 0 || tile.complexity.height <= 0) {
          set_error(error, "twopass: tile has no visible pixels");
          return false;
        }
        tile.complexity.residual_mad = normalized.complexity.residual_mad;
        tile.complexity.residual_mse = normalized.complexity.residual_mse;
        tile.complexity.intra_fraction =
            normalized.complexity.intra_fraction;
      } else {
        set_error(error, tile_error.c_str());
        return false;
      }
    }
    const std::size_t count =
        static_cast<std::size_t>(tile.sb_cols) *
        static_cast<std::size_t>(tile.sb_rows);
    if (!tile.sb_weights.empty() && tile.sb_weights.size() != count) {
      set_error(error, "twopass: tile sb_weights size does not match extent");
      return false;
    }
    for (const double weight : tile.sb_weights) {
      if (!finite_nonnegative(weight)) {
        set_error(error, "twopass: sb_weights must be finite and >= 0");
        return false;
      }
    }
    for (int y = 0; y < tile.sb_rows; ++y) {
      for (int x = 0; x < tile.sb_cols; ++x) {
        const std::size_t index =
            static_cast<std::size_t>(tile.sb_row_start + y) *
                static_cast<std::size_t>(sbx) +
            static_cast<std::size_t>(tile.sb_col_start + x);
        if (coverage[index] != -1) {
          set_error(error,
                    "twopass: tile superblock extents overlap");
          return false;
        }
        coverage[index] = static_cast<int>(t);
      }
    }
  }
  for (const int owner : coverage) {
    if (owner == -1) {
      set_error(error, "twopass: tiles do not cover the frame");
      return false;
    }
  }
  return true;
}

double frame_weight(const FrameStats& stats,
                    const TwoPassConfig& config) noexcept {
  double weight = 0.0;
  if (stats.first_pass_bits > 0.0) {
    weight = stats.first_pass_bits;
  } else {
    const double sigma =
        frame_complexity_sigma(stats.complexity,
                               config.rc.estimator.params.mad_to_sigma);
    const double pixels = static_cast<double>(stats.complexity.pixel_count());
    weight = pixels * std::pow(std::max(sigma, kSigmaFloor),
                               config.complexity_exponent);
  }
  if (stats.keyframe) {
    weight *= config.keyframe_weight;
  }
  if (!std::isfinite(weight) || weight < 0.0) {
    return 0.0;
  }
  return weight;
}

// Deterministic water-filling: proportional to weights, clamped per frame.
void waterfill(std::span<const double> weights,
               double total_bits,
               double min_bits,
               double max_bits,
               std::vector<double>& target) {
  const std::size_t count = weights.size();
  target.assign(count, 0.0);

  std::vector<char> at_min(count, 0);
  std::vector<char> at_max(count, 0);
  for (int iteration = 0; iteration <= static_cast<int>(count) + 1;
       ++iteration) {
    double free_budget = total_bits;
    double free_weight = 0.0;
    int free_count = 0;
    for (std::size_t i = 0; i < count; ++i) {
      if (at_min[i] != 0) {
        target[i] = min_bits;
        free_budget -= min_bits;
      } else if (at_max[i] != 0) {
        target[i] = max_bits;
        free_budget -= max_bits;
      } else {
        free_weight += weights[i];
        ++free_count;
      }
    }

    bool changed = false;
    for (std::size_t i = 0; i < count; ++i) {
      if (at_min[i] != 0 || at_max[i] != 0) {
        continue;
      }
      double candidate = 0.0;
      if (free_weight > 0.0) {
        candidate = free_budget * weights[i] / free_weight;
      } else if (free_count > 0) {
        candidate = free_budget / static_cast<double>(free_count);
      }
      if (candidate < min_bits - kBitEpsilon) {
        at_min[i] = 1;
        target[i] = min_bits;
        changed = true;
      } else if (candidate > max_bits + kBitEpsilon) {
        at_max[i] = 1;
        target[i] = max_bits;
        changed = true;
      } else {
        target[i] = std::max(0.0, candidate);
      }
    }
    if (!changed) {
      break;
    }
  }
}

}  // namespace

bool normalize_frame_stats(const FrameStats& input,
                           FrameStats& normalized,
                           std::string* error) {
  return normalize_frame_stats_impl(input, normalized, error);
}

// ---------------------------------------------------------------------------
// VBV
// ---------------------------------------------------------------------------

bool make_vbv_state(const VbvConfig& config,
                    double average_frame_bits,
                    double frame_duration,
                    VbvState& state,
                    std::string* error) {
  state = VbvState{};
  state.enabled = config.enabled;
  if (!validate_vbv_config(config, error)) {
    return false;
  }
  if (!config.enabled) {
    return true;
  }
  if (!finite_positive(average_frame_bits)) {
    set_error(error, "twopass: vbv needs a positive average frame budget");
    return false;
  }
  if (!finite_positive(frame_duration)) {
    set_error(error, "twopass: vbv needs a positive frame duration");
    return false;
  }

  const double size = config.buffer_size_bits > 0.0
                          ? config.buffer_size_bits
                          : config.buffer_frames * average_frame_bits;
  if (!finite_positive(size)) {
    set_error(error, "twopass: vbv buffer size must be > 0");
    return false;
  }
  const double drain = config.max_rate_bps > 0.0
                           ? config.max_rate_bps * frame_duration
                           : average_frame_bits;
  if (!finite_positive(drain)) {
    set_error(error, "twopass: vbv drain rate must be > 0");
    return false;
  }

  state.size_bits = size;
  state.drain_bits_per_frame = drain;
  state.underflow_margin_bits = config.underflow_margin * size;
  state.overflow_margin_bits = config.overflow_margin * size;
  const double low = state.underflow_margin_bits;
  const double high = size - state.overflow_margin_bits;
  state.fill_bits = std::clamp(config.initial_fill * size, low, high);
  return true;
}

double vbv_max_bits(const VbvState& state) noexcept {
  if (!state.enabled) {
    return std::numeric_limits<double>::infinity();
  }
  return state.fill_bits + state.drain_bits_per_frame -
         state.underflow_margin_bits;
}

double vbv_min_bits(const VbvState& state) noexcept {
  if (!state.enabled) {
    return 0.0;
  }
  return state.fill_bits + state.drain_bits_per_frame -
         (state.size_bits - state.overflow_margin_bits);
}

bool vbv_apply(VbvState& state,
               double bits,
               bool& underflow,
               bool& overflow) noexcept {
  underflow = false;
  overflow = false;
  if (!finite_nonnegative(bits)) {
    return false;
  }
  if (!state.enabled) {
    return true;
  }
  const double next = state.fill_bits + state.drain_bits_per_frame - bits;
  if (next < state.underflow_margin_bits - kBitEpsilon) {
    underflow = true;
    ++state.underflow_events;
  }
  if (next > state.size_bits - state.overflow_margin_bits + kBitEpsilon) {
    overflow = true;
    ++state.overflow_events;
  }
  state.fill_bits = std::clamp(next, state.underflow_margin_bits,
                               state.size_bits - state.overflow_margin_bits);
  return true;
}

double frame_duration_seconds(const VbrParams& params) noexcept {
  if (params.frame_rate_num <= 0 || params.frame_rate_den <= 0) {
    return 0.0;
  }
  return static_cast<double>(params.frame_rate_den) /
         static_cast<double>(params.frame_rate_num);
}

double nominal_frame_bits(const VbrParams& params) noexcept {
  const double duration = frame_duration_seconds(params);
  if (!finite_positive(duration) || !finite_positive(params.target_bitrate_bps)) {
    return 0.0;
  }
  return params.target_bitrate_bps * duration;
}

// ---------------------------------------------------------------------------
// Configuration validation
// ---------------------------------------------------------------------------

bool two_pass_config_valid(const TwoPassConfig& config,
                           std::string* error) {
  if (!rate_control_config_valid(config.rc, error)) {
    return false;
  }
  if (config.lookahead_frames < 0) {
    set_error(error, "twopass: lookahead_frames must be >= 0");
    return false;
  }
  if (!std::isfinite(config.complexity_exponent) ||
      config.complexity_exponent < 0.0 || config.complexity_exponent > 2.0) {
    set_error(error, "twopass: complexity_exponent must be in [0, 2]");
    return false;
  }
  if (!finite_nonnegative(config.keyframe_weight)) {
    set_error(error, "twopass: keyframe_weight must be finite and >= 0");
    return false;
  }
  if (!std::isfinite(config.min_frame_bits_ratio) ||
      config.min_frame_bits_ratio <= 0.0 ||
      config.min_frame_bits_ratio > 1.0) {
    set_error(error, "twopass: min_frame_bits_ratio must be in (0, 1]");
    return false;
  }
  if (!std::isfinite(config.max_frame_bits_ratio) ||
      config.max_frame_bits_ratio < 1.0 ||
      config.max_frame_bits_ratio > 100.0 ||
      config.max_frame_bits_ratio < config.min_frame_bits_ratio) {
    set_error(error, "twopass: max_frame_bits_ratio must be in [1, 100]");
    return false;
  }
  if (config.min_qindex < kMinQIndex || config.max_qindex > kMaxQIndex ||
      config.min_qindex > config.max_qindex) {
    set_error(error, "twopass: qindex range is invalid");
    return false;
  }
  if (config.max_qindex_step < 1) {
    set_error(error, "twopass: max_qindex_step must be >= 1");
    return false;
  }
  if (config.spatial_qp_strength < 0 ||
      config.spatial_qp_strength > kMaxQIndex) {
    set_error(error, "twopass: spatial_qp_strength must be in [0, 255]");
    return false;
  }
  if (config.delta_q_res < 0 || config.delta_q_res > 3) {
    set_error(error, "twopass: delta_q_res must be in [0, 3]");
    return false;
  }
  if (config.segment_levels < 1 ||
      config.segment_levels > max_segment_classes(config.enable_ext_seg)) {
    set_error(error, "twopass: segment_levels exceeds the segment budget");
    return false;
  }
  if (config.segment_qindex_step < 1 ||
      config.segment_qindex_step > kMaxQIndex) {
    set_error(error, "twopass: segment_qindex_step must be in [1, 255]");
    return false;
  }
  if (!std::isfinite(config.rate_feedback_weight) ||
      config.rate_feedback_weight < 0.0 || config.rate_feedback_weight > 1.0) {
    set_error(error, "twopass: rate_feedback_weight must be in [0, 1]");
    return false;
  }
  if (!std::isfinite(config.rate_feedback_clamp) ||
      config.rate_feedback_clamp < 1.0) {
    set_error(error, "twopass: rate_feedback_clamp must be >= 1");
    return false;
  }
  if (!validate_vbv_config(config.vbv, error)) {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

bool allocate_frames(std::span<const FrameStats> stats,
                     const TwoPassConfig& config,
                     double total_bits,
                     const VbvState& initial_buffer,
                     AllocationResult& result,
                     std::string* error) {
  result = AllocationResult{};
  if (stats.empty()) {
    set_error(error, "twopass: allocation needs at least one frame");
    return false;
  }
  if (!finite_nonnegative(total_bits)) {
    set_error(error, "twopass: allocation budget must be finite and >= 0");
    return false;
  }
  if (!two_pass_config_valid(config, error)) {
    return false;
  }
  if (initial_buffer.enabled && !finite_positive(initial_buffer.size_bits)) {
    set_error(error, "twopass: allocation got an invalid VBV state");
    return false;
  }

  std::vector<FrameStats> normalized;
  normalized.reserve(stats.size());
  for (const FrameStats& input : stats) {
    FrameStats frame;
    if (!normalize_frame_stats_impl(input, frame, error)) {
      return false;
    }
    normalized.push_back(std::move(frame));
  }

  const std::size_t count = normalized.size();
  std::vector<double> weights(count, 0.0);
  double weight_sum = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    weights[i] = frame_weight(normalized[i], config);
    weight_sum += weights[i];
  }
  if (!(weight_sum > 0.0)) {
    // All-flat clip: split the budget evenly.
    std::fill(weights.begin(), weights.end(), 1.0);
  }

  const double average_bits = total_bits / static_cast<double>(count);
  const double min_bits = config.min_frame_bits_ratio * average_bits;
  const double max_bits =
      std::max(min_bits, config.max_frame_bits_ratio * average_bits);

  std::vector<double> target;
  waterfill(weights, total_bits, min_bits, max_bits, target);

  result.frames.resize(count);
  result.total_bits = total_bits;
  for (std::size_t i = 0; i < count; ++i) {
    FrameAllocation& frame = result.frames[i];
    frame.weight = weights[i];
    frame.target_bits = target[i];
    frame.min_bits = min_bits;
    frame.max_bits = max_bits;
  }

  if (!initial_buffer.enabled) {
    double sum = 0.0;
    for (const FrameAllocation& frame : result.frames) {
      sum += frame.target_bits;
    }
    result.allocated_bits = sum;
    result.dropped_bits = std::max(0.0, total_bits - sum);
    result.underflow_prevented = false;
    result.underflow_feasible = true;
    result.overflow_feasible = true;
    return true;
  }

  const double low_margin = initial_buffer.underflow_margin_bits;
  const double high_margin =
      initial_buffer.size_bits - initial_buffer.overflow_margin_bits;
  std::vector<char> touched(count, 0);

  // Forward VBV feasibility pass. A frame that would spend more than the
  // bucket can supply is capped to the sustainable size. If even the
  // per-frame minimum is too large, the window is flagged infeasible; the
  // controller still clamps rather than dropping or padding frames.
  double fill = initial_buffer.fill_bits;
  for (std::size_t i = 0; i < count; ++i) {
    FrameAllocation& frame = result.frames[i];
    const double cap =
        fill + initial_buffer.drain_bits_per_frame - low_margin;
    if (frame.target_bits > cap + kBitEpsilon) {
      const double capped = std::max(frame.min_bits, cap);
      if (cap < frame.min_bits - kBitEpsilon) {
        result.underflow_feasible = false;
      }
      frame.target_bits = capped;
      frame.vbv_capped = true;
      result.underflow_prevented = true;
      touched[i] = 1;
    }
    fill += initial_buffer.drain_bits_per_frame - frame.target_bits;
    if (fill < low_margin - kBitEpsilon) {
      result.underflow_feasible = false;
    }
  }

  // Redistribute the budget freed by the caps to later frames that still have
  // headroom. Earlier frames are never increased: their bitstream may already
  // be on the wire in a streaming second pass.
  double allocated = 0.0;
  for (const FrameAllocation& frame : result.frames) {
    allocated += frame.target_bits;
  }
  double unused = std::max(0.0, total_bits - allocated);
  if (unused > kBitEpsilon) {
    fill = initial_buffer.fill_bits;
    for (std::size_t i = 0; i < count; ++i) {
      FrameAllocation& frame = result.frames[i];
      const double cap =
          fill + initial_buffer.drain_bits_per_frame - low_margin;
      const double room = std::min(frame.max_bits, cap) - frame.target_bits;
      if (room > kBitEpsilon && unused > kBitEpsilon) {
        const double add = std::min(room, unused);
        frame.target_bits += add;
        unused -= add;
        touched[i] = 1;
      }
      fill += initial_buffer.drain_bits_per_frame - frame.target_bits;
    }
  }

  // Final trajectory: recompute diagnostics from the actual allocation.
  fill = initial_buffer.fill_bits;
  result.underflow_feasible = true;
  result.overflow_feasible = true;
  double final_sum = 0.0;
  int reallocations = 0;
  for (std::size_t i = 0; i < count; ++i) {
    FrameAllocation& frame = result.frames[i];
    fill += initial_buffer.drain_bits_per_frame - frame.target_bits;
    frame.vbv_fill_after = fill;
    if (fill < low_margin - kBitEpsilon) {
      result.underflow_feasible = false;
    }
    if (fill > high_margin + kBitEpsilon) {
      frame.overflow_risk = true;
      result.overflow_feasible = false;
    }
    final_sum += frame.target_bits;
    if (touched[i] != 0) {
      ++reallocations;
    }
  }
  result.allocated_bits = final_sum;
  result.dropped_bits = std::max(0.0, total_bits - final_sum);
  if (result.dropped_bits < kBitEpsilon) {
    result.dropped_bits = 0.0;
  }
  result.reallocations = reallocations;
  return true;
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

struct TwoPassController::QualityState {
  RateController controller;
};

TwoPassController::TwoPassController() = default;
TwoPassController::~TwoPassController() = default;

bool TwoPassController::configure(const TwoPassConfig& config,
                                  std::string* error) {
  if (!two_pass_config_valid(config, error)) {
    return false;
  }
  config_ = config;
  configured_ = true;
  if (quality_ == nullptr) {
    quality_ = std::make_unique<QualityState>();
  }
  if (config_.rc.mode != RateControlMode::kVbr) {
    if (!quality_->controller.configure(config_.rc, error)) {
      configured_ = false;
      return false;
    }
  }
  return reset(error);
}

bool TwoPassController::reset(std::string* error) {
  if (!configured_) {
    set_error(error, "twopass: controller is not configured");
    return false;
  }
  const double duration = frame_duration_seconds(config_.rc.vbr);
  const double average = nominal_frame_bits(config_.rc.vbr);
  if (!(duration > 0.0) || !(average > 0.0)) {
    set_error(error, "twopass: frame rate and target bitrate must be positive");
    return false;
  }
  if (!make_vbv_state(config_.vbv, average, duration, vbv_, error)) {
    return false;
  }
  pending_.clear();
  allocation_ = AllocationResult{};
  rate_feedback_ = 1.0;
  total_target_bits_ = 0.0;
  total_actual_bits_ = 0.0;
  previous_qindex_ = 0;
  frame_count_ = 0;
  frame_open_ = false;
  offline_ = false;
  open_plan_ = FramePlan{};
  open_target_bits_ = 0.0;
  open_estimate_bits_ = 0.0;
  if (quality_ != nullptr && config_.rc.mode != RateControlMode::kVbr) {
    quality_->controller.reset();
  }
  return true;
}

bool TwoPassController::validate_and_plan(std::span<const FrameStats> stats,
                                          std::size_t horizon,
                                          double total_bits,
                                          std::string* error) {
  std::vector<FrameStats> normalized;
  normalized.reserve(stats.size());
  for (const FrameStats& input : stats) {
    FrameStats frame;
    if (!normalize_frame_stats_impl(input, frame, error)) {
      return false;
    }
    normalized.push_back(std::move(frame));
  }
  const std::size_t effective_horizon =
      std::min(horizon, normalized.size());
  if (effective_horizon == 0) {
    set_error(error, "twopass: empty lookahead window");
    return false;
  }
  AllocationResult allocation;
  if (!allocate_frames(
          std::span<const FrameStats>(normalized.data(), effective_horizon),
          config_, total_bits, vbv_, allocation, error)) {
    return false;
  }
  allocation_ = std::move(allocation);
  pending_ = std::move(normalized);
  return true;
}

bool TwoPassController::analyze(std::span<const FrameStats> stats,
                                std::string* error) {
  if (!configured_) {
    set_error(error, "twopass: controller is not configured");
    return false;
  }
  if (frame_open_) {
    set_error(error, "twopass: end_frame() must precede analyze()");
    return false;
  }
  if (stats.empty()) {
    set_error(error, "twopass: analyze needs at least one frame");
    return false;
  }
  if (!reset(error)) {
    return false;
  }
  offline_ = true;
  const double average = nominal_frame_bits(config_.rc.vbr);
  const double total = average * static_cast<double>(stats.size());
  return validate_and_plan(stats, stats.size(), total, error);
}

bool TwoPassController::set_lookahead(std::span<const FrameStats> window,
                                      std::string* error) {
  if (!configured_) {
    set_error(error, "twopass: controller is not configured");
    return false;
  }
  if (frame_open_) {
    set_error(error, "twopass: end_frame() must precede set_lookahead()");
    return false;
  }
  if (window.empty()) {
    set_error(error, "twopass: lookahead window must not be empty");
    return false;
  }
  offline_ = false;
  std::size_t horizon = window.size();
  if (config_.lookahead_frames > 0) {
    horizon = std::min<std::size_t>(
        horizon, static_cast<std::size_t>(config_.lookahead_frames));
  }
  const double average = nominal_frame_bits(config_.rc.vbr);
  double total = average * static_cast<double>(horizon);
  if (vbv_.enabled) {
    // A streaming window starts from the current buffer, not from the ideal
    // fill. Credit (or debit) the window budget with the buffer surplus so an
    // underspent window is made up in the next one instead of persisting.
    // The credit is bounded by one window's budget.
    const double target_fill = config_.vbv.initial_fill * vbv_.size_bits;
    const double credit =
        std::clamp(vbv_.fill_bits - target_fill, -total, total);
    total = std::max(0.0, total + credit);
  }
  return validate_and_plan(window, horizon, total, error);
}

namespace {

struct SegmentLadder {
  std::array<int, kMaxSegments> alt{};
  int count = 1;
  int step = 4;
};

SegmentLadder make_segment_ladder(const TwoPassConfig& config) noexcept {
  SegmentLadder ladder;
  ladder.count = config.enable_segmentation
                     ? 2 * config.segment_levels + 1
                     : 1;
  ladder.step = config.segment_qindex_step;
  for (int k = 1; k <= config.segment_levels; ++k) {
    ladder.alt[static_cast<std::size_t>(k)] = -ladder.step * k;
    ladder.alt[static_cast<std::size_t>(config.segment_levels + k)] =
        ladder.step * k;
  }
  return ladder;
}

int nearest_segment(int desired_offset, const SegmentLadder& ladder) noexcept {
  int best = 0;
  int best_error = desired_offset < 0 ? -desired_offset : desired_offset;
  for (int id = 1; id < ladder.count; ++id) {
    const int alt = ladder.alt[static_cast<std::size_t>(id)];
    const int error = desired_offset >= alt ? desired_offset - alt
                                            : alt - desired_offset;
    if (error < best_error) {
      best_error = error;
      best = id;
    }
  }
  return best;
}

// Builds per-tile and per-superblock targets, plus the delta-Q and
// segmentation syntax plans. Returns the raw model estimate for the spatial
// plan. `over_budget` tells the caller that reducing the spatial offsets was
// not enough to fit the allowed budget.
bool build_spatial_plan(const FrameStats& stats,
                        const TwoPassConfig& config,
                        int base_q,
                        double rate_feedback,
                        double frame_estimate,
                        double hard_cap,
                        FramePlan& plan,
                        double& spatial_estimate,
                        bool& over_budget,
                        bool& spatial_clamped,
                        std::string* error) {
  over_budget = false;
  spatial_estimate = frame_estimate;
  const std::size_t tile_count = stats.tiles.size();
  if (tile_count == 0) {
    set_error(error, "twopass: no tiles in normalized stats");
    return false;
  }

  const SegmentLadder ladder = make_segment_ladder(config);
  plan.delta_q.present = config.enable_delta_q;
  plan.delta_q.res = config.enable_delta_q ? config.delta_q_res : 0;
  plan.enable_ext_seg = config.enable_segmentation && config.enable_ext_seg;

  // Tile weights use the same complexity measure as the frame allocation.
  std::vector<double> tile_weight(tile_count, 0.0);
  double tile_weight_sum = 0.0;
  double tile_pixel_sum = 0.0;
  for (std::size_t t = 0; t < tile_count; ++t) {
    const TileStats& tile = stats.tiles[t];
    const double sigma =
        frame_complexity_sigma(tile.complexity,
                               config.rc.estimator.params.mad_to_sigma);
    const double pixels = static_cast<double>(tile.complexity.pixel_count());
    const double weight =
        pixels * std::pow(std::max(sigma, kSigmaFloor),
                          config.complexity_exponent);
    tile_weight[t] = std::isfinite(weight) && weight > 0.0 ? weight : 0.0;
    tile_weight_sum += tile_weight[t];
    tile_pixel_sum += pixels;
  }
  const double mean_tile_weight =
      tile_weight_sum > 0.0 ? tile_weight_sum / static_cast<double>(tile_count)
                            : 0.0;

  plan.segmentation = SegmentationParams{};
  if (config.enable_segmentation) {
    plan.segmentation.enabled = true;
    plan.segmentation.update_map = true;
    plan.segmentation.temporal_update = false;
    for (int id = 1; id < ladder.count; ++id) {
      const int alt = ladder.alt[static_cast<std::size_t>(id)];
      if (alt != 0 &&
          !set_segment_alt_q(plan.segmentation, id, alt, error)) {
        return false;
      }
    }
    if (!segmentation_params_valid(plan.segmentation, error)) {
      return false;
    }
  }

  // Offsets are computed once per tile, then reduced by a scalar when the
  // spatial plan would exceed the frame's budget. Scaling uses truncation so
  // a scale of zero is exactly the frame-uniform plan.
  std::vector<int> tile_offset(tile_count, 0);
  for (std::size_t t = 0; t < tile_count; ++t) {
    if (config.spatial_qp_strength <= 0 || mean_tile_weight <= 0.0 ||
        tile_weight[t] <= 0.0) {
      tile_offset[t] = 0;
      continue;
    }
    const double ratio = tile_weight[t] / mean_tile_weight;
    const double octaves = std::log2(std::max(ratio, kWeightFloor));
    const double normalized = std::clamp(octaves / 2.0, -1.0, 1.0);
    const double scaled =
        normalized * static_cast<double>(config.spatial_qp_strength);
    // Higher activity gets a finer quantiser (more bits).
    tile_offset[t] = round_half_away(-scaled);
  }
  // Remove the pixel-weighted mean offset so the frame's base_q_idx is not
  // shifted by the spatial assignment.
  if (tile_pixel_sum > 0.0 && config.spatial_qp_strength > 0) {
    double mean_offset = 0.0;
    for (std::size_t t = 0; t < tile_count; ++t) {
      mean_offset +=
          static_cast<double>(tile_offset[t]) *
          static_cast<double>(stats.tiles[t].complexity.pixel_count());
    }
    mean_offset /= tile_pixel_sum;
    const int correction = round_half_away(mean_offset);
    for (std::size_t t = 0; t < tile_count; ++t) {
      tile_offset[t] -= correction;
      tile_offset[t] = std::clamp(tile_offset[t],
                                  -config.spatial_qp_strength,
                                  config.spatial_qp_strength);
    }
  }

  const double allowed = std::min(hard_cap, frame_estimate * 1.02);

  double scale = 1.0;
  std::vector<int> desired;  // per-SB desired effective qindex, flattened
  std::vector<std::uint8_t> segment_ids;
  std::vector<int> sb_alt;
  std::vector<TilePlan> tiles;

  auto build_targets = [&](double used_scale) {
    desired.clear();
    segment_ids.clear();
    tiles.clear();
    spatial_estimate = 0.0;
    for (std::size_t t = 0; t < tile_count; ++t) {
      const TileStats& source = stats.tiles[t];
      TilePlan tile;
      tile.sb_cols = source.sb_cols;
      tile.sb_rows = source.sb_rows;
      tile.sb_col_start = source.sb_col_start;
      tile.sb_row_start = source.sb_row_start;
      const std::size_t count =
          static_cast<std::size_t>(source.sb_cols) *
          static_cast<std::size_t>(source.sb_rows);
      tile.sb_target_qindex.assign(count, 0);
      tile.sb_achieved_qindex.assign(count, 0);
      tile.sb_segment_id.assign(count, 0);

      double weight_mean = 0.0;
      if (!source.sb_weights.empty()) {
        for (const double weight : source.sb_weights) {
          weight_mean += weight;
        }
        weight_mean /= static_cast<double>(count);
      }
      const int tile_off = static_cast<int>(
          std::trunc(static_cast<double>(tile_offset[t]) * used_scale));
      int qsum = 0;
      for (int y = 0; y < source.sb_rows; ++y) {
        for (int x = 0; x < source.sb_cols; ++x) {
          const std::size_t local =
              static_cast<std::size_t>(y) *
                  static_cast<std::size_t>(source.sb_cols) +
              static_cast<std::size_t>(x);
          int sb_off = 0;
          if (config.spatial_qp_strength > 0 && weight_mean > 0.0 &&
              !source.sb_weights.empty()) {
            const double weight = source.sb_weights[local];
            const double ratio = std::max(weight, kWeightFloor) / weight_mean;
            const double normalized =
                std::clamp(std::log2(ratio), -1.0, 1.0);
            const double scaled =
                normalized * 0.5 * static_cast<double>(config.spatial_qp_strength);
            // Higher activity within a tile gets a finer quantiser too.
            sb_off = round_half_away(-scaled * used_scale);
          }
          const int q = std::clamp(base_q + tile_off + sb_off,
                                   config.min_qindex, config.max_qindex);
          tile.sb_target_qindex[local] = q;
          qsum += q;
          desired.push_back(q);
        }
      }
      tile.target_qindex =
          count > 0 ? std::clamp(round_half_away(
                                     static_cast<double>(qsum) /
                                     static_cast<double>(count)),
                                 config.min_qindex, config.max_qindex)
                    : base_q;
      tiles.push_back(std::move(tile));
    }

    // Model cost of the spatial assignment at the tile's representative
    // qindex. The estimator is per-pixel plus a per-call header, so this is
    // comparable to the frame estimate.
    for (std::size_t t = 0; t < tile_count; ++t) {
      double bits = 0.0;
      if (!estimate_bits(config.rc.estimator, stats.tiles[t].complexity,
                         tiles[t].target_qindex, bits, error)) {
        return false;
      }
      spatial_estimate += bits;
    }
    return true;
  };

  bool built = false;
  for (int attempt = 0; attempt < kMaxSpatialAttempts; ++attempt) {
    if (!build_targets(scale)) {
      return false;
    }
    built = true;
    if (spatial_estimate * rate_feedback <= allowed + kBitEpsilon) {
      break;
    }
    spatial_clamped = true;
    if (scale > 0.0) {
      scale = scale * 0.5 >= 0.0625 ? scale * 0.5 : 0.0;
      continue;
    }
    over_budget = true;
    break;
  }
  if (!built) {
    set_error(error, "twopass: failed to build a spatial plan");
    return false;
  }

  // Assign segments from the final desired offsets.
  if (config.enable_segmentation) {
    segment_ids.assign(desired.size(), 0);
    std::size_t cursor = 0;
    for (std::size_t t = 0; t < tile_count; ++t) {
      const std::size_t count =
          static_cast<std::size_t>(tiles[t].sb_cols) *
          static_cast<std::size_t>(tiles[t].sb_rows);
      for (std::size_t i = 0; i < count; ++i) {
        const int offset = desired[cursor + i] - base_q;
        tiles[t].sb_segment_id[i] = static_cast<std::uint8_t>(
            nearest_segment(offset, ladder));
      }
      cursor += count;
    }
  }

  // Build delta-Q targets after the segment offsets are known. With delta-Q
  // present, ALT_Q is relative to CurrentQIndex, so target CurrentQIndex is
  // desired - ALT_Q. Without delta-Q, the effective qindex is base + ALT_Q
  // and the segment ladder is the only spatial control.
  std::size_t cursor = 0;
  for (std::size_t t = 0; t < tile_count; ++t) {
    TilePlan& tile = tiles[t];
    const std::size_t count =
        static_cast<std::size_t>(tile.sb_cols) *
        static_cast<std::size_t>(tile.sb_rows);
    if (config.enable_delta_q) {
      std::vector<int> current_targets(count, base_q);
      for (std::size_t i = 0; i < count; ++i) {
        int alt = 0;
        if (config.enable_segmentation) {
          const int id = static_cast<int>(tile.sb_segment_id[i]);
          alt = ladder.alt[static_cast<std::size_t>(id)];
        }
        current_targets[i] =
            std::clamp(desired[cursor + i] - alt, kMinQIndex, kMaxQIndex);
      }
      DeltaQPlanConfig plan_config;
      plan_config.base_q_idx = base_q;
      plan_config.delta_q_res = config.delta_q_res;
      plan_config.cols = tile.sb_cols;
      plan_config.rows = tile.sb_rows;
      if (!plan_delta_q(plan_config, current_targets, tile.delta_q, error)) {
        return false;
      }
      for (std::size_t i = 0; i < count; ++i) {
        const int alt = config.enable_segmentation
                            ? ladder.alt[static_cast<std::size_t>(
                                  tile.sb_segment_id[i])]
                            : 0;
        tile.sb_achieved_qindex[i] =
            std::clamp(tile.delta_q.steps[i].qindex_after + alt,
                       kLosslessQIndex, kMaxQIndex);
      }
    } else {
      tile.delta_q.clear();
      for (std::size_t i = 0; i < count; ++i) {
        const int alt = config.enable_segmentation
                            ? ladder.alt[static_cast<std::size_t>(
                                  tile.sb_segment_id[i])]
                            : 0;
        tile.sb_achieved_qindex[i] =
            std::clamp(base_q + alt, kLosslessQIndex, kMaxQIndex);
      }
    }
    cursor += count;
  }
  plan.tiles = std::move(tiles);

  // Frame-wide segmentation map on the 4x4 MI grid. Tile coverage was
  // validated by normalize_frame_stats(), so every MI cell is assigned.
  plan.segmentation_map = SegmentationMap{};
  if (config.enable_segmentation) {
    const int mi_per_sb = stats.sb_size / 4;
    const int mi_cols = (stats.complexity.width + 3) / 4;
    const int mi_rows = (stats.complexity.height + 3) / 4;
    plan.segmentation_map.mi_cols = mi_cols;
    plan.segmentation_map.mi_rows = mi_rows;
    plan.segmentation_map.segment_id.assign(
        static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows),
        0);
    for (const TilePlan& tile : plan.tiles) {
      for (int sy = 0; sy < tile.sb_rows; ++sy) {
        const int mi_y0 = (tile.sb_row_start + sy) * mi_per_sb;
        const int mi_y1 = std::min(mi_rows, mi_y0 + mi_per_sb);
        for (int sx = 0; sx < tile.sb_cols; ++sx) {
          const int mi_x0 = (tile.sb_col_start + sx) * mi_per_sb;
          const int mi_x1 = std::min(mi_cols, mi_x0 + mi_per_sb);
          const std::size_t local =
              static_cast<std::size_t>(sy) *
                  static_cast<std::size_t>(tile.sb_cols) +
              static_cast<std::size_t>(sx);
          const std::uint8_t id = tile.sb_segment_id[local];
          for (int y = mi_y0; y < mi_y1; ++y) {
            for (int x = mi_x0; x < mi_x1; ++x) {
              plan.segmentation_map
                  .segment_id[static_cast<std::size_t>(y) *
                                  static_cast<std::size_t>(mi_cols) +
                              static_cast<std::size_t>(x)] = id;
            }
          }
        }
      }
    }
    if (!segmentation_map_valid(plan.segmentation_map, error)) {
      return false;
    }
  }
  return true;
}

// Model cost of the frame at `qindex` when each tile is coded at that same
// qindex. Summing per-tile estimates (rather than using one frame-average
// sigma) is what makes the base-q selection consistent with a spatially
// varying QP assignment: the sum is the actual predicted cost of the zero
// offset plan.
bool spatial_cost_at_q(const FrameStats& stats,
                       const TwoPassConfig& config,
                       int qindex,
                       double& bits,
                       std::string* error) {
  bits = 0.0;
  for (const TileStats& tile : stats.tiles) {
    double tile_bits = 0.0;
    if (!estimate_bits(config.rc.estimator, tile.complexity, qindex,
                       tile_bits, error)) {
      return false;
    }
    bits += tile_bits;
  }
  return true;
}

// Smallest qindex in [lo, hi] whose corrected spatial cost fits `budget`.
bool select_qindex_for_budget(const FrameStats& stats,
                              const TwoPassConfig& config,
                              int lo,
                              int hi,
                              double budget,
                              double rate_feedback,
                              int& qindex,
                              bool& reachable,
                              std::string* error) {
  double at_lo = 0.0;
  if (!spatial_cost_at_q(stats, config, lo, at_lo, error)) {
    return false;
  }
  if (at_lo * rate_feedback <= budget + kBitEpsilon) {
    qindex = lo;
    reachable = true;
    return true;
  }
  double at_hi = 0.0;
  if (!spatial_cost_at_q(stats, config, hi, at_hi, error)) {
    return false;
  }
  if (at_hi * rate_feedback > budget + kBitEpsilon) {
    qindex = hi;
    reachable = false;
    return true;
  }
  int low = lo;
  int high = hi;
  while (low < high) {
    const int mid = low + (high - low) / 2;
    double bits = 0.0;
    if (!spatial_cost_at_q(stats, config, mid, bits, error)) {
      return false;
    }
    if (bits * rate_feedback <= budget + kBitEpsilon) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  qindex = low;
  reachable = true;
  return true;
}

}  // namespace

bool TwoPassController::begin_frame(FramePlan& plan, std::string* error) {
  if (!configured_) {
    set_error(error, "twopass: controller is not configured");
    return false;
  }
  if (frame_open_) {
    set_error(error, "twopass: end_frame() must precede the next begin_frame()");
    return false;
  }
  if (pending_.empty()) {
    set_error(error,
              "twopass: no first-pass statistics (call analyze or set_lookahead)");
    return false;
  }
  if (allocation_.frames.empty()) {
    set_error(error, "twopass: lookahead window exhausted (call set_lookahead)");
    return false;
  }

  const FrameStats& stats = pending_.front();
  const FrameAllocation& allocation = allocation_.frames.front();
  const RateControlMode mode = config_.rc.mode;

  // If a later planning step fails, close the inner controller's frame so a
  // retry does not see a half-open frame. Error paths are exceptional; a zero
  // sample in its bits EMA is harmless.
  struct QualityFrameGuard {
    RateController* controller = nullptr;
    bool armed = false;
    ~QualityFrameGuard() {
      if (armed && controller != nullptr) {
        std::string ignored;
        controller->end_frame(0, &ignored);
      }
    }
  } quality_guard;

  QpDecision quality_decision{};
  if (quality_ != nullptr && mode != RateControlMode::kVbr) {
    if (!quality_->controller.begin_frame(stats.complexity, quality_decision,
                                          error)) {
      return false;
    }
    quality_guard.controller = &quality_->controller;
    quality_guard.armed = true;
  }

  const double hard_cap =
      vbv_.enabled ? vbv_max_bits(vbv_)
                   : std::numeric_limits<double>::infinity();

  bool underflow_prevented = false;
  bool target_reachable = true;
  double target_bits = allocation.target_bits;
  int qindex = config_.min_qindex;
  double continuous_qp = static_cast<double>(config_.min_qindex);
  bool crf_target = false;

  if (mode == RateControlMode::kCqp) {
    qindex = clamp_qindex_range(config_.rc.cqp.base_q_idx, config_);
    continuous_qp = static_cast<double>(qindex);
  } else if (mode == RateControlMode::kCrf) {
    qindex = clamp_qindex_range(quality_decision.base_q_idx, config_);
    continuous_qp = quality_decision.continuous_qp;
    crf_target = true;
  } else {
    double budget = std::min(allocation.target_bits, hard_cap);
    if (budget < allocation.target_bits - kBitEpsilon) {
      underflow_prevented = true;
      target_bits = std::max(0.0, budget);
    }
    if (budget <= 0.0) {
      qindex = config_.max_qindex;
      target_reachable = false;
    } else if (!select_qindex_for_budget(
                   stats, config_, config_.min_qindex, config_.max_qindex,
                   budget, rate_feedback_, qindex, target_reachable, error)) {
      return false;
    }
    continuous_qp = static_cast<double>(qindex);
  }

  // Slew limit. Keyframes are independent and reset the trajectory.
  bool clamped_by_slew = false;
  if (previous_qindex_ != 0 && !stats.keyframe &&
      config_.max_qindex_step < kMaxQIndex) {
    const int low = std::max(config_.min_qindex,
                             previous_qindex_ - config_.max_qindex_step);
    const int high = std::min(config_.max_qindex,
                              previous_qindex_ + config_.max_qindex_step);
    const int limited = std::clamp(qindex, low, high);
    if (limited != qindex) {
      clamped_by_slew = true;
      qindex = limited;
    }
  }

  // Hard underflow override for CQP/CRF (VBR already used the capped target).
  if (mode != RateControlMode::kVbr && vbv_.enabled) {
    double bits = 0.0;
    if (!spatial_cost_at_q(stats, config_, qindex, bits, error)) {
      return false;
    }
    if (bits * rate_feedback_ > hard_cap + kBitEpsilon) {
      int limited = qindex;
      bool reachable = true;
      if (!select_qindex_for_budget(stats, config_, qindex,
                                    config_.max_qindex, hard_cap,
                                    rate_feedback_, limited, reachable,
                                    error)) {
        return false;
      }
      if (limited != qindex) {
        underflow_prevented = true;
        qindex = limited;
        continuous_qp = static_cast<double>(qindex);
      }
      if (!reachable) {
        target_reachable = false;
      }
      target_bits = std::max(0.0, hard_cap);
    }
  }

  // VBR: after the slew clamp, a finer qindex could overspend the budget.
  if (mode == RateControlMode::kVbr && vbv_.enabled) {
    double bits = 0.0;
    if (!spatial_cost_at_q(stats, config_, qindex, bits, error)) {
      return false;
    }
    if (bits * rate_feedback_ > target_bits + kBitEpsilon) {
      int limited = qindex;
      bool reachable = true;
      if (!select_qindex_for_budget(stats, config_, qindex,
                                    config_.max_qindex, target_bits,
                                    rate_feedback_, limited, reachable,
                                    error)) {
        return false;
      }
      qindex = limited;
      continuous_qp = static_cast<double>(qindex);
      if (!reachable) {
        target_reachable = false;
      }
    }
  }

  bool clamped_by_range = false;
  if (qindex < config_.min_qindex || qindex > config_.max_qindex) {
    clamped_by_range = true;
    qindex = clamp_qindex_range(qindex, config_);
  }

  // Frame estimate at the chosen qindex, summed over tiles so it matches the
  // spatial plan's own cost model.
  double frame_estimate = 0.0;
  if (!spatial_cost_at_q(stats, config_, qindex, frame_estimate, error)) {
    return false;
  }

  FramePlan out;
  out.frame_index = static_cast<int>(frame_count_);
  out.keyframe = stats.keyframe;
  out.base_q_idx = qindex;
  out.coeff_cdf_qctx = coeff_cdf_qctx(qindex);
  out.allocation_bits = allocation.target_bits;
  out.target_bits = target_bits;
  out.rate_feedback = rate_feedback_;
  out.crf_quality_target = crf_target;
  out.vbv_fill_before = vbv_.enabled ? vbv_.fill_bits : 0.0;
  out.clamped_by_slew = clamped_by_slew;
  out.clamped_by_range = clamped_by_range;
  out.target_reachable = target_reachable;

  bool spatial_clamped = false;
  bool over_budget = false;
  double spatial_estimate = frame_estimate;
  // VBR treats the per-frame allocation as a hard budget. CRF/CQP are
  // quality-driven, so only the VBV cap is hard for them.
  const double spatial_budget_cap =
      mode == RateControlMode::kVbr
          ? target_bits
          : (vbv_.enabled ? vbv_max_bits(vbv_)
                          : std::numeric_limits<double>::infinity());
  out.tiles.clear();
  if (!build_spatial_plan(stats, config_, qindex, rate_feedback_,
                          frame_estimate, spatial_budget_cap, out,
                          spatial_estimate, over_budget, spatial_clamped,
                          error)) {
    return false;
  }
  if (over_budget) {
    // The qindex range was exhausted (or a hard cap is below the plan's
    // floor). The plan is still returned; target_reachable reports it.
    target_reachable = false;
    out.target_reachable = false;
  }

  out.continuous_qp = continuous_qp;
  out.estimated_bits = spatial_estimate;
  out.lambda_sse =
      lambda_sse_from_qindex(out.base_q_idx, config_.rc.lambda_sse_scale);
  out.spatial_clamped = spatial_clamped;

  if (vbv_.enabled) {
    out.vbv_fill_after_estimate =
        vbv_.fill_bits + vbv_.drain_bits_per_frame - out.estimated_bits;
    out.vbv_overflow_risk =
        out.vbv_fill_after_estimate >
        vbv_.size_bits - vbv_.overflow_margin_bits + kBitEpsilon;
    out.vbv_underflow_prevented = underflow_prevented;
  }

  quality_guard.armed = false;
  open_plan_ = out;
  open_target_bits_ = out.target_bits;
  open_estimate_bits_ = out.estimated_bits;
  frame_open_ = true;
  total_target_bits_ += out.target_bits;
  plan = std::move(out);
  return true;
}

bool TwoPassController::reallocate_remaining(double error_bits,
                                             std::string* error) {
  if (!offline_ || pending_.empty() || allocation_.frames.empty()) {
    return true;
  }
  std::size_t horizon = pending_.size();
  if (config_.lookahead_frames > 0) {
    horizon = std::min<std::size_t>(
        horizon, static_cast<std::size_t>(config_.lookahead_frames));
  }
  horizon = std::min(horizon, allocation_.frames.size());
  if (horizon == 0) {
    return true;
  }
  double window_target = 0.0;
  for (std::size_t i = 0; i < horizon; ++i) {
    window_target += allocation_.frames[i].target_bits;
  }
  const double remaining_budget =
      std::max(0.0, window_target - error_bits);
  AllocationResult replanned;
  if (!allocate_frames(
          std::span<const FrameStats>(pending_.data(), horizon), config_,
          remaining_budget, vbv_, replanned, error)) {
    allocation_.frames.clear();
    return false;
  }
  for (std::size_t i = 0; i < horizon; ++i) {
    allocation_.frames[i] = replanned.frames[i];
  }
  return true;
}

bool TwoPassController::end_frame(std::int64_t actual_bits,
                                  std::string* error) {
  if (!configured_) {
    set_error(error, "twopass: controller is not configured");
    return false;
  }
  if (!frame_open_) {
    set_error(error, "twopass: end_frame() called without begin_frame()");
    return false;
  }
  if (actual_bits < 0 ||
      static_cast<double>(actual_bits) > kMaxActualBits) {
    set_error(error, "twopass: actual_bits is out of range");
    return false;
  }

  if (quality_ != nullptr && config_.rc.mode != RateControlMode::kVbr) {
    if (!quality_->controller.end_frame(actual_bits, error)) {
      return false;
    }
  }

  const double actual = static_cast<double>(actual_bits);
  bool underflow = false;
  bool overflow = false;
  if (vbv_.enabled) {
    if (!vbv_apply(vbv_, actual, underflow, overflow)) {
      set_error(error, "twopass: vbv rejected the actual bit count");
      return false;
    }
  }
  (void)underflow;
  (void)overflow;

  const double weight = config_.rate_feedback_weight;
  if (weight > 0.0 && open_estimate_bits_ > 1.0) {
    double ratio = actual / open_estimate_bits_;
    ratio = std::clamp(ratio, 1.0 / config_.rate_feedback_clamp,
                       config_.rate_feedback_clamp);
    if (frame_count_ == 0) {
      rate_feedback_ = ratio;
    } else {
      rate_feedback_ = (1.0 - weight) * rate_feedback_ + weight * ratio;
    }
  } else if (weight <= 0.0) {
    rate_feedback_ = 1.0;
  }

  total_actual_bits_ += actual;
  previous_qindex_ = open_plan_.base_q_idx;
  const double error_bits = actual - open_target_bits_;
  ++frame_count_;
  pending_.erase(pending_.begin());
  allocation_.frames.erase(allocation_.frames.begin());
  frame_open_ = false;

  if (offline_ && config_.redistribute_bits && !pending_.empty() &&
      !allocation_.frames.empty()) {
    if (!reallocate_remaining(error_bits, error)) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// First pass over a luma plane
// ---------------------------------------------------------------------------

bool first_pass_luma(std::span<const std::uint8_t> luma,
                     int width,
                     int height,
                     int stride,
                     int tile_cols,
                     int tile_rows,
                     bool keyframe,
                     FrameStats& stats,
                     std::string* error) {
  stats = FrameStats{};
  if (width <= 0 || height <= 0 || stride < width) {
    set_error(error, "twopass: invalid luma geometry");
    return false;
  }
  const std::size_t needed =
      static_cast<std::size_t>(height - 1) * static_cast<std::size_t>(stride) +
      static_cast<std::size_t>(width);
  if (luma.size() < needed) {
    set_error(error, "twopass: luma span is too small");
    return false;
  }
  constexpr int kSb = 64;
  const int sbx = (width + kSb - 1) / kSb;
  const int sby = (height + kSb - 1) / kSb;
  if (tile_cols < 1 || tile_rows < 1 || tile_cols > sbx ||
      tile_rows > sby) {
    set_error(error, "twopass: tile grid exceeds the superblock grid");
    return false;
  }

  std::vector<int> xs(static_cast<std::size_t>(tile_cols) + 1, 0);
  std::vector<int> ys(static_cast<std::size_t>(tile_rows) + 1, 0);
  for (int t = 0; t <= tile_cols; ++t) {
    xs[static_cast<std::size_t>(t)] = t * sbx / tile_cols;
  }
  for (int t = 0; t <= tile_rows; ++t) {
    ys[static_cast<std::size_t>(t)] = t * sby / tile_rows;
  }
  std::vector<int> tile_of_sb(
      static_cast<std::size_t>(sbx) * static_cast<std::size_t>(sby), 0);
  for (int ty = 0; ty < tile_rows; ++ty) {
    for (int tx = 0; tx < tile_cols; ++tx) {
      for (int y = ys[static_cast<std::size_t>(ty)];
           y < ys[static_cast<std::size_t>(ty) + 1]; ++y) {
        for (int x = xs[static_cast<std::size_t>(tx)];
             x < xs[static_cast<std::size_t>(tx) + 1]; ++x) {
          tile_of_sb[static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(sbx) +
                     static_cast<std::size_t>(x)] = ty * tile_cols + tx;
        }
      }
    }
  }

  struct Accumulator {
    std::uint64_t sum_abs = 0;
    std::uint64_t sum_sq = 0;
    std::uint64_t pixels = 0;
  };
  std::vector<Accumulator> tile_acc(
      static_cast<std::size_t>(tile_cols) * static_cast<std::size_t>(tile_rows));
  std::vector<std::uint64_t> sb_sq(
      static_cast<std::size_t>(sbx) * static_cast<std::size_t>(sby), 0);
  std::uint64_t frame_abs = 0;
  std::uint64_t frame_sq = 0;
  std::uint64_t frame_pixels = 0;

  constexpr int kBlock = 16;
  for (int by = 0; by < height; by += kBlock) {
    const int bh = std::min(kBlock, height - by);
    for (int bx = 0; bx < width; bx += kBlock) {
      const int bw = std::min(kBlock, width - bx);
      const std::uint64_t block_pixels =
          static_cast<std::uint64_t>(bw) * static_cast<std::uint64_t>(bh);
      std::uint64_t sum = 0;
      for (int y = 0; y < bh; ++y) {
        const std::size_t row =
            static_cast<std::size_t>(by + y) * static_cast<std::size_t>(stride);
        for (int x = 0; x < bw; ++x) {
          sum += luma[row + static_cast<std::size_t>(bx + x)];
        }
      }
      const int mean = static_cast<int>(
          (sum + block_pixels / 2U) / block_pixels);
      const std::size_t sb_index =
          static_cast<std::size_t>(by / kSb) *
              static_cast<std::size_t>(sbx) +
          static_cast<std::size_t>(bx / kSb);
      const int tile_index = tile_of_sb[sb_index];
      Accumulator& tile =
          tile_acc[static_cast<std::size_t>(tile_index)];

      std::uint64_t local_abs = 0;
      std::uint64_t local_sq = 0;
      for (int y = 0; y < bh; ++y) {
        const std::size_t row =
            static_cast<std::size_t>(by + y) * static_cast<std::size_t>(stride);
        for (int x = 0; x < bw; ++x) {
          const int sample =
              static_cast<int>(luma[row + static_cast<std::size_t>(bx + x)]);
          const int diff = sample - mean;
          const std::uint64_t magnitude =
              static_cast<std::uint64_t>(diff < 0 ? -diff : diff);
          const std::uint64_t square =
              static_cast<std::uint64_t>(diff * diff);
          local_abs += magnitude;
          local_sq += square;
        }
      }
      tile.sum_abs += local_abs;
      tile.sum_sq += local_sq;
      tile.pixels += block_pixels;
      sb_sq[sb_index] += local_sq;
      frame_abs += local_abs;
      frame_sq += local_sq;
      frame_pixels += block_pixels;
    }
  }

  if (frame_pixels == 0) {
    set_error(error, "twopass: empty luma plane");
    return false;
  }
  stats.sb_size = kSb;
  stats.keyframe = keyframe;
  stats.complexity.width = width;
  stats.complexity.height = height;
  stats.complexity.residual_mad =
      static_cast<double>(frame_abs) / static_cast<double>(frame_pixels);
  stats.complexity.residual_mse =
      static_cast<double>(frame_sq) / static_cast<double>(frame_pixels);
  stats.complexity.intra_fraction = 1.0;

  stats.tiles.resize(static_cast<std::size_t>(tile_cols) *
                     static_cast<std::size_t>(tile_rows));
  for (int ty = 0; ty < tile_rows; ++ty) {
    for (int tx = 0; tx < tile_cols; ++tx) {
      const std::size_t index =
          static_cast<std::size_t>(ty) * static_cast<std::size_t>(tile_cols) +
          static_cast<std::size_t>(tx);
      TileStats& tile = stats.tiles[index];
      tile.sb_col_start = xs[static_cast<std::size_t>(tx)];
      tile.sb_row_start = ys[static_cast<std::size_t>(ty)];
      tile.sb_cols =
          xs[static_cast<std::size_t>(tx) + 1] - tile.sb_col_start;
      tile.sb_rows =
          ys[static_cast<std::size_t>(ty) + 1] - tile.sb_row_start;
      const int px0 = tile.sb_col_start * kSb;
      const int py0 = tile.sb_row_start * kSb;
      const int px1 =
          std::min(width, (tile.sb_col_start + tile.sb_cols) * kSb);
      const int py1 =
          std::min(height, (tile.sb_row_start + tile.sb_rows) * kSb);
      const Accumulator& acc = tile_acc[index];
      tile.complexity.width = std::max(0, px1 - px0);
      tile.complexity.height = std::max(0, py1 - py0);
      if (acc.pixels > 0) {
        tile.complexity.residual_mad =
            static_cast<double>(acc.sum_abs) /
            static_cast<double>(acc.pixels);
        tile.complexity.residual_mse =
            static_cast<double>(acc.sum_sq) /
            static_cast<double>(acc.pixels);
      }
      tile.complexity.intra_fraction = 1.0;
      const std::size_t sb_count =
          static_cast<std::size_t>(tile.sb_cols) *
          static_cast<std::size_t>(tile.sb_rows);
      tile.sb_weights.resize(sb_count, 0.0);
      for (int y = 0; y < tile.sb_rows; ++y) {
        for (int x = 0; x < tile.sb_cols; ++x) {
          const std::size_t global =
              static_cast<std::size_t>(tile.sb_row_start + y) *
                  static_cast<std::size_t>(sbx) +
              static_cast<std::size_t>(tile.sb_col_start + x);
          const std::size_t local =
              static_cast<std::size_t>(y) *
                  static_cast<std::size_t>(tile.sb_cols) +
              static_cast<std::size_t>(x);
          tile.sb_weights[local] = static_cast<double>(sb_sq[global]);
        }
      }
    }
  }
  return true;
}

}  // namespace av2enc::rc::twopass
