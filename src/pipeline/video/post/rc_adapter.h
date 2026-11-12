// M2 per-frame rate-control adapter.
//
// This layer sits between the existing, already-tested frame-level controller
// in src/encoder/rc/ and the M2 frame-header writer in this directory.  It adds
// only the M2-specific constraints that the generic controller cannot know:
//
//   * the §8.3.1 coefficient-CDF quantizer-context slice is selected by the
//     raw frame base_q_idx.  The shared `src/encoder/m1/qctx/` tables cover all
//     four slices, so the policy may use the full qindex range;
//   * the frame header codes base_q_idx with f(8) at 8-bit and f(9) at 10-bit
//     (§5.18.6.1), and MaxQ is 255 at 8-bit and 303 at 10-bit;
//   * the coefficient-CDF slice is a function of the frame base_q_idx only,
//     never of a tile/delta-Q/segment qindex.
//
// The adapter is deterministic: no clock, RNG or hidden host state.  It wraps
// exactly one rc::RateController, so VBR's virtual-buffer history is explicit.
//
// Normative references (AV2 v1.0.0): §5.18.6.1 base_q_idx, §7.14.2 qlookup,
// §8.3.1 init_coeff_cdfs() slice selection, §6.4.1 Table 6.3 bit depths/MaxQ.
#pragma once

#include "formats/format.h"
#include "codec/rate/rc/rc.h"

#include <cstdint>
#include <string>

namespace av2enc::video::post {

// Historical limit used before the shared qctx tables landed.  Kept for
// callers that still reference the name; the policy default no longer uses it.
inline constexpr int kM2MaxCoeffCdfSlice0QIndex = 90;

// True when the M2 coefficient-CDF tables cover `qctx`.  All four §8.3.1
// slices are available from the shared `encoder/m1/qctx/` tables.
bool video_coeff_cdf_slice_available(int qctx) noexcept;

// §5.18.6.1: base_q_idx is f(8) for 8-bit and f(9) otherwise.
int video_base_q_idx_bits(fmt::BitDepth depth) noexcept;

// §6.4.1 Table 6.3: 255 (8-bit), 303 (10-bit).  12-bit is reserved and
// returns -1 so callers reject it.
int video_max_quantizer(fmt::BitDepth depth) noexcept;

// One frame's quantizer decision as it must be consumed by the M2 writer.
struct VideoFrameQp {
  int base_q_idx = 90;       // always in [1, 255], the rc component's domain
  int coeff_cdf_qctx = 0;    // derived from base_q_idx only
  int qp_slice = 0;          // same index as coeff_cdf_qctx
  int base_q_idx_bits = 8;   // f(8) at 8-bit, f(9) at 10-bit
  int max_quantizer = 255;   // format MaxQ
  double continuous_qp = 90.0;
  double estimated_bits = 0.0;
  double lambda_sse = 0.0;
  bool target_reachable = true;
  bool clamped_by_step = false;
  // True when a CQP request or the controller's frame decision was moved into
  // the available coefficient-CDF slice / format domain.  This is a
  // table/policy limit, not a rate-control decision.  CRF endpoint bounds and
  // the VBR qindex range are enforced inside the controller and are not
  // reported here.
  bool clamped_for_m2 = false;
  bool coeff_cdf_slice_available = true;
};

struct VideoRcPolicy {
  int min_base_q_idx = rc::kMinQIndex;
  // All four coefficient-CDF qctx slices ship, so the default policy may use
  // the whole 8-bit qindex range (the format MaxQ still bounds it).
  int max_base_q_idx = rc::kMaxQIndex;
  // When true, a decision outside [min, max] is clamped and flagged.  When
  // false, begin_frame() fails instead of silently changing the decision.
  bool clamp_to_policy = true;
  // When true, every qctx slice the policy can reach must exist.  All four
  // slices are shipped, so this only guards against a future table regression.
  bool require_video_coeff_cdf_slice = true;
};

struct VideoRcConfig {
  rc::RateControlConfig rc{};
  VideoRcPolicy policy{};
  fmt::BitDepth depth = fmt::BitDepth::k8;
};

bool video_rc_config_valid(const VideoRcConfig& config,
                        std::string* error = nullptr);

// Stateful per-frame controller: one instance per encode.  One begin_frame()
// must be followed by exactly one end_frame() carrying the actual payload
// bits (not bytes, not OBU overhead).
class VideoRateControl {
 public:
  VideoRateControl() = default;

  bool configure(const VideoRcConfig& config, std::string* error = nullptr);

  bool begin_frame(const rc::FrameComplexity& complexity, VideoFrameQp& qp,
                   std::string* error = nullptr);

  bool end_frame(std::int64_t actual_bits, std::string* error = nullptr);

  void reset() noexcept;

  bool configured() const noexcept { return configured_; }
  bool frame_open() const noexcept { return rc_.frame_open(); }
  const VideoRcConfig& config() const noexcept { return config_; }
  int previous_qindex() const noexcept { return rc_.previous_qindex(); }
  double buffer_bits() const noexcept { return rc_.buffer_bits(); }

 private:
  VideoRcConfig config_{};
  rc::RateController rc_{};
  bool configured_ = false;
  bool config_clamped_ = false;
};

// Convenience for tests and integration logging: the M2 coefficient-CDF slice
// for a raw qindex (same boundaries as rc::coeff_cdf_qctx).
int video_qp_slice(int base_q_idx) noexcept;

}  // namespace av2enc::video::post
