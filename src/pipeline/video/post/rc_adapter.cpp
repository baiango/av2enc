#include "pipeline/video/post/rc_adapter.h"

#include "codec/entropy/qctx/qctx.h"

#include <algorithm>
#include <cmath>

namespace av2enc::video::post {
namespace {

bool fail(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

}  // namespace

bool video_coeff_cdf_slice_available(int qctx) noexcept {
  return qctx >= 0 && qctx < av2enc::still::qctx::kSliceCount;
}

int video_base_q_idx_bits(fmt::BitDepth depth) noexcept {
  switch (depth) {
    case fmt::BitDepth::k8:
      return 8;
    case fmt::BitDepth::k10:
      return 9;
    case fmt::BitDepth::k12:
      return -1;
  }
  return -1;
}

int video_max_quantizer(fmt::BitDepth depth) noexcept {
  switch (depth) {
    case fmt::BitDepth::k8:
      return 255;
    case fmt::BitDepth::k10:
      return 303;
    case fmt::BitDepth::k12:
      return -1;
  }
  return -1;
}

int video_qp_slice(int base_q_idx) noexcept {
  return rc::coeff_cdf_qctx(base_q_idx);
}

bool video_rc_config_valid(const VideoRcConfig& config, std::string* error) {
  if (video_base_q_idx_bits(config.depth) < 0 ||
      video_max_quantizer(config.depth) < 0) {
    return fail(error, "m2 rate control: 12-bit is reserved in AV2 v1.0.0");
  }
  const int format_max = video_max_quantizer(config.depth);
  if (config.policy.min_base_q_idx < rc::kMinQIndex ||
      config.policy.max_base_q_idx > rc::kMaxQIndex ||
      config.policy.min_base_q_idx > config.policy.max_base_q_idx) {
    return fail(error,
                "m2 rate control: policy qindex range must satisfy "
                "1 <= min <= max <= 255");
  }
  if (config.policy.max_base_q_idx > format_max) {
    return fail(error,
                "m2 rate control: policy max exceeds the format MaxQ");
  }
  if (config.policy.require_video_coeff_cdf_slice) {
    for (int qctx = 0; qctx < av2enc::still::qctx::kSliceCount; ++qctx) {
      if (!video_coeff_cdf_slice_available(qctx)) {
        return fail(error,
                    "m2 rate control: a coefficient-CDF qctx slice is missing "
                    "from the shared encoder/m1/qctx tables");
      }
    }
  }
  // The underlying controller is still the 8-bit-calibrated CQP/CRF/VBR
  // component.  Its own validation handles mode-specific fields.
  if (!rc::rate_control_config_valid(config.rc, error)) return false;
  // A CQP value outside the M2 policy is not rejected here: it is the point of
  // begin_frame() to clamp it deterministically (or fail when clamping is off).
  return true;
}

bool VideoRateControl::configure(const VideoRcConfig& config, std::string* error) {
  if (!video_rc_config_valid(config, error)) {
    configured_ = false;
    return false;
  }
  const int format_max = video_max_quantizer(config.depth);
  int max_allowed = std::min(config.policy.max_base_q_idx, format_max);
  int min_allowed = std::max(config.policy.min_base_q_idx, rc::kMinQIndex);
  // Feed the boundary into the controller itself.  This matters for VBR: its
  // slew limit and virtual buffer should reason about the qindexes the M2
  // writer can actually signal, not an unreachable slice.  With clamping
  // disabled the controller keeps the caller's domain and begin_frame()
  // rejects an out-of-policy decision.
  rc::RateControlConfig effective = config.rc;
  bool config_clamped = false;
  if (config.policy.clamp_to_policy) {
    const auto clamp_into = [&](int value) {
      if (value < min_allowed) {
        config_clamped = true;
        return min_allowed;
      }
      if (value > max_allowed) {
        config_clamped = true;
        return max_allowed;
      }
      return value;
    };
    switch (config.rc.mode) {
      case rc::RateControlMode::kCqp:
        effective.cqp.base_q_idx = clamp_into(effective.cqp.base_q_idx);
        break;
      case rc::RateControlMode::kCrf:
        effective.crf.qindex_at_crf0 = static_cast<double>(
            clamp_into(static_cast<int>(effective.crf.qindex_at_crf0)));
        effective.crf.qindex_at_crf63 = static_cast<double>(
            clamp_into(static_cast<int>(effective.crf.qindex_at_crf63)));
        if (effective.crf.qindex_at_crf0 > effective.crf.qindex_at_crf63) {
          effective.crf.qindex_at_crf63 = effective.crf.qindex_at_crf0;
        }
        // The endpoints are a policy bound, not a request for a specific
        // frame qindex; only flag an explicit CQP clamp.
        config_clamped = false;
        break;
      case rc::RateControlMode::kVbr:
        effective.vbr.min_qindex =
            std::max(effective.vbr.min_qindex, min_allowed);
        effective.vbr.max_qindex =
            std::min(effective.vbr.max_qindex, max_allowed);
        if (effective.vbr.min_qindex > effective.vbr.max_qindex) {
          return fail(error,
                      "m2 rate control: VBR qindex range does not intersect "
                      "the M2 policy range");
        }
        config_clamped = false;
        break;
    }
  }
  if (!rc_.configure(effective, error)) {
    configured_ = false;
    return false;
  }
  config_ = config;
  config_clamped_ = config_clamped;
  configured_ = true;
  return true;
}

bool VideoRateControl::begin_frame(const rc::FrameComplexity& complexity,
                                VideoFrameQp& qp, std::string* error) {
  if (!configured_) return fail(error, "m2 rate control: not configured");
  rc::QpDecision decision;
  if (!rc_.begin_frame(complexity, decision, error)) return false;

  const int format_max = video_max_quantizer(config_.depth);
  const int bits = video_base_q_idx_bits(config_.depth);
  if (format_max < 0 || bits < 0) {
    return fail(error, "m2 rate control: unsupported bit depth");
  }
  int max_allowed = std::min(config_.policy.max_base_q_idx, format_max);
  int min_allowed = std::max(config_.policy.min_base_q_idx, rc::kMinQIndex);

  int base_q_idx = decision.base_q_idx;
  bool clamped = false;
  if (base_q_idx < min_allowed) {
    base_q_idx = min_allowed;
    clamped = true;
  } else if (base_q_idx > max_allowed) {
    base_q_idx = max_allowed;
    clamped = true;
  }
  if (base_q_idx < rc::kMinQIndex || base_q_idx > rc::kMaxQIndex) {
    if (!config_.policy.clamp_to_policy) {
      return fail(error,
                  "m2 rate control: decision is outside the M2 qindex policy");
    }
    base_q_idx = base_q_idx < rc::kMinQIndex ? rc::kMinQIndex : rc::kMaxQIndex;
    clamped = true;
  }

  const int qctx = rc::coeff_cdf_qctx(base_q_idx);
  const bool slice_ok = video_coeff_cdf_slice_available(qctx);
  if (!slice_ok && config_.policy.require_video_coeff_cdf_slice) {
    return fail(error,
                "m2 rate control: decision needs a coefficient-CDF slice that "
                "is not shipped");
  }

  if (clamped && !config_.policy.clamp_to_policy) {
    // The controller frame is already open; close the accounting frame with
    // the model estimate so the caller can retry or abort cleanly.
    (void)rc_.end_frame(
        static_cast<std::int64_t>(std::llround(decision.estimated_bits)));
    return fail(error,
                "m2 rate control: decision is outside the M2 qindex policy and "
                "clamping is disabled");
  }

  // Re-evaluate the model at the final qindex so estimated_bits, lambda and
  // the writer's base_q_idx describe the same quantizer.
  double estimated = 0.0;
  if (!rc::estimate_frame_bits(config_.rc.estimator, complexity, base_q_idx,
                               estimated, error)) {
    (void)rc_.end_frame(
        static_cast<std::int64_t>(std::llround(decision.estimated_bits)));
    return false;
  }

  qp = VideoFrameQp{};
  qp.base_q_idx = base_q_idx;
  qp.coeff_cdf_qctx = qctx;
  qp.qp_slice = qctx;
  qp.base_q_idx_bits = bits;
  qp.max_quantizer = format_max;
  qp.continuous_qp = static_cast<double>(base_q_idx);
  qp.estimated_bits = estimated;
  qp.lambda_sse = rc::lambda_sse_from_qindex(base_q_idx,
                                             config_.rc.lambda_sse_scale);
  qp.target_reachable = decision.target_reachable;
  qp.clamped_by_step = decision.clamped_by_step;
  qp.clamped_for_m2 =
      clamped || config_clamped_ || base_q_idx != decision.base_q_idx;
  qp.coeff_cdf_slice_available = slice_ok;
  return true;
}

bool VideoRateControl::end_frame(std::int64_t actual_bits, std::string* error) {
  if (!configured_) return fail(error, "m2 rate control: not configured");
  return rc_.end_frame(actual_bits, error);
}

void VideoRateControl::reset() noexcept {
  rc_.reset();
  configured_ = false;
  config_clamped_ = false;
}

}  // namespace av2enc::video::post
