// M2 per-frame rate-control adapter.
//
// This layer sits between the existing, already-tested frame-level controller
// in src/encoder/rc/ and the M2 frame-header writer in this directory.  It adds
// only the M2-specific constraints that the generic controller cannot know:
//
//   * the M2 core coefficient CDF tables currently cover only the first
//     §8.3.1 quantizer-context slice (base_q_idx <= 90), so a decision above
//     that slice would desynchronise coefficients unless tables for qctx 1..3
//     are generated;
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

#include "encoder/format/format.h"
#include "encoder/rc/rc.h"

#include <cstdint>
#include <string>

namespace av2enc::m2::post {

// M2's shipped coefficient CDF tables are slice 0 only.
inline constexpr int kM2MaxCoeffCdfSlice0QIndex = 90;

// True when the current M2 coefficient-CDF tables can encode a frame whose
// base_q_idx selects `qctx`.  Only slice 0 (qctx == 0) exists today.
bool m2_coeff_cdf_slice_available(int qctx) noexcept;

// §5.18.6.1: base_q_idx is f(8) for 8-bit and f(9) otherwise.
int m2_base_q_idx_bits(fmt::BitDepth depth) noexcept;

// §6.4.1 Table 6.3: 255 (8-bit), 303 (10-bit).  12-bit is reserved and
// returns -1 so callers reject it.
int m2_max_quantizer(fmt::BitDepth depth) noexcept;

// One frame's quantizer decision as it must be consumed by the M2 writer.
struct M2FrameQp {
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

struct M2RcPolicy {
  int min_base_q_idx = rc::kMinQIndex;
  // Default stops at the only shipped M2 coefficient-CDF slice.
  int max_base_q_idx = kM2MaxCoeffCdfSlice0QIndex;
  // When true, a decision outside [min, max] is clamped and flagged.  When
  // false, begin_frame() fails instead of silently changing the decision.
  bool clamp_to_policy = true;
  // When true, the adapter additionally refuses to leave coefficient-CDF
  // slice 0.  Set false only after generating §9.3 tables for slices 1..3.
  bool require_m2_coeff_cdf_slice = true;
};

struct M2RcConfig {
  rc::RateControlConfig rc{};
  M2RcPolicy policy{};
  fmt::BitDepth depth = fmt::BitDepth::k8;
};

bool m2_rc_config_valid(const M2RcConfig& config,
                        std::string* error = nullptr);

// Stateful per-frame controller: one instance per encode.  One begin_frame()
// must be followed by exactly one end_frame() carrying the actual payload
// bits (not bytes, not OBU overhead).
class M2RateControl {
 public:
  M2RateControl() = default;

  bool configure(const M2RcConfig& config, std::string* error = nullptr);

  bool begin_frame(const rc::FrameComplexity& complexity, M2FrameQp& qp,
                   std::string* error = nullptr);

  bool end_frame(std::int64_t actual_bits, std::string* error = nullptr);

  void reset() noexcept;

  bool configured() const noexcept { return configured_; }
  bool frame_open() const noexcept { return rc_.frame_open(); }
  const M2RcConfig& config() const noexcept { return config_; }
  int previous_qindex() const noexcept { return rc_.previous_qindex(); }
  double buffer_bits() const noexcept { return rc_.buffer_bits(); }

 private:
  M2RcConfig config_{};
  rc::RateController rc_{};
  bool configured_ = false;
  bool config_clamped_ = false;
};

// Convenience for tests and integration logging: the M2 coefficient-CDF slice
// for a raw qindex (same boundaries as rc::coeff_cdf_qctx).
int m2_qp_slice(int base_q_idx) noexcept;

}  // namespace av2enc::m2::post
