// Least-squares Wiener-NS coefficient search.
//
// The decoder's §7.20.3 filter is
//
//   s = m << 7 + sum_i (src(tap_i) - m) * coeff[group_i]
//   out = Round2(s, 7)
//
// where several taps share one coefficient group (Wiener_Ns_Config_*[i][2]).
// This module fits those coefficients to minimise SSE against a target plane
// (the encoder's source picture) using the *same* stripe-aware source sampler
// as the decoder core (lr_get_source_sample), so the fitted filter is evaluated
// on exactly the samples the decoder will see.
//
// The fit is a host-side encoder decision, not bit-exact arithmetic: a double
// precision normal-equation solve is rounded to the normative integer tap
// ranges and then re-evaluated with the exact integer filter.  Every choice is
// deterministic (fixed traversal order, deterministic pivoting, round-half-
// away-from-zero) so two runs on the same inputs produce identical plans.

#ifndef AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_FIT_H_
#define AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_FIT_H_

#include <cstdint>
#include <vector>

#include "encoder/filters/restoration/wiener_core.h"

namespace av2enc::restoration::classes {

enum class WienerFitStatus {
  kOk = 0,
  kInvalidArgument,
  kNoSamples,
  kSingular,
  kUnsupportedPlane,
};

const char* wiener_fit_status_string(WienerFitStatus status);

struct WienerFitOptions {
  // -1 fits every subset the plane allows; otherwise only that subset.
  int forced_subset = -1;
  // Chroma only: also try wiener_ns_uv_sym == 1 for subsets > 0.
  bool try_uv_sym = true;
  // Relative Tikhonov term added to the normal equations for stability.
  double ridge = 1e-6;
};

// One quantised candidate filter.  `coeffs` uses the §7.20.3 layout: luma
// 0..15; chroma 0..5 (chroma taps) and 6..17 (luma-derived taps).
struct WienerFitCandidate {
  int subset = 0;
  bool uv_sym = false;
  std::int16_t coeffs[kWienerNsCoeffStride] = {};
  double sse = 0.0;           // exact integer-filter SSE over the region
  double identity_sse = 0.0;  // SSE with no filtering
  int samples = 0;
  int present_coeffs = 0;     // number of coded coefficient values
};

struct WienerFitResult {
  std::vector<WienerFitCandidate> candidates;
  double identity_sse = 0.0;
  int samples = 0;
};

// Fits filters for one plane region.
//
//   unit_row/unit_col : >= 0 restricts the fit to that restoration unit;
//                       negative means the whole plane (frame-level filter).
//   class_index       : >= 0 restricts to cells whose subclass_grid entry
//                       equals class_index; negative means all cells.
//   subclass_grid     : optional luma MI grid of coefficient classes, used
//                       only when class_index >= 0.
//   target            : original picture for this plane, plane coordinates.
//
// `core` must be configured for the restoration unit size and unit type grid
// of the candidate (frame_type[plane] == kRestoreWienerNs, unit_type all
// kRestoreWienerNs) because lr_block_context() supplies the stripe bounds.
WienerFitStatus wiener_fit_candidates(
    const LrCoreFrame& core, int plane, int unit_row, int unit_col,
    int class_index, const std::uint8_t* subclass_grid,
    const std::uint8_t* target, int target_stride,
    const WienerFitOptions& options, WienerFitResult* result);

// Exact SSE of an arbitrary coefficient set over a region.  `coeffs` uses the
// §7.20.3 layout.  When `unit_row`/`unit_col` are negative the whole plane is
// evaluated.
double wiener_filter_sse(const LrCoreFrame& core, int plane, int unit_row,
                         int unit_col, const std::uint8_t* target,
                         int target_stride,
                         const std::int16_t coeffs[kWienerNsCoeffStride]);

// Class-restricted variant used when fitting/merging frame-level filters.
double wiener_filter_sse_class(
    const LrCoreFrame& core, int plane, int class_index,
    const std::uint8_t* subclass_grid, const std::uint8_t* target,
    int target_stride, const std::int16_t coeffs[kWienerNsCoeffStride]);

// Exact SSE of a frame-level filter.  `frame_coeffs[class]` is selected with
// `subclass_grid` (luma MI grid); chroma always uses class 0.
double wiener_frame_filter_sse(
    const LrCoreFrame& core, int plane, int unit_row, int unit_col,
    int num_classes, const std::uint8_t* subclass_grid,
    const std::uint8_t* target, int target_stride,
    const std::int16_t frame_coeffs[kWienerNsClasses][kWienerNsCoeffStride]);

// Convenience: best candidate by exact SSE, deterministic tie-break:
// lower subset, then uv_sym == false, then lexicographic coefficients.
const WienerFitCandidate* wiener_best_fit_candidate(
    const WienerFitResult& result);

}  // namespace av2enc::restoration::classes

#endif  // AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_FIT_H_
