#ifndef AV2ENC_ENCODER_INTER_GME_GME_H_
#define AV2ENC_ENCODER_INTER_GME_GME_H_

// AV2 v1.0.0 global motion estimation — CPU reference (encoder side).
//
// Scope:
//   * §5.18.9  global_motion_params() syntax: GmType, per-reference params,
//              use_global_motion, base/their-ref differential prediction,
//              scale_warp_model, read/write_global_param.
//   * §7.12.2.1/.2  setup global MV and get warp motion vector projection.
//   * §7.13.3.21/.23  shear validation and model reduction.
//   * Encoder-side fitting from point correspondences, per-block warps, or
//     block MVs, including deterministic outlier rejection, degradation from
//     AFFINE to ROTZOOM/IDENTITY, and quantisation to the syntax grid.
//
// This component writes no bitstream itself. `gme_syntax.{h,cpp}` writes and
// reads the GM syntax into/from a BitWriter; the caller owns frame-header
// ordering. The README lists the exact call sites in the M2 core.
//
// No AVM/SVT/clave code. Constants and formulas are transcribed from the AV2
// v1.0.0 specification (spec/AV2_Spec_v1.0.0.pdf, commit 38f28e7).

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "gme_core.h"

namespace av2enc::inter::gme {

inline constexpr int kMaxRefsPerFrame = 7;   // REFS_PER_FRAME
inline constexpr int kNumRefFrames = 16;     // ref slots

// §5.18.9: IDENTITY = 0, ROTZOOM = 1, AFFINE = 2.
enum class GmType : std::int32_t {
  kIdentity = 0,
  kRotZoom = 1,
  kAffine = 2,
};

const char* gm_type_name(GmType type);

// Six Q16 parameters in §7.12.2.2 / §7.13.3.19 order:
//   p[0] column (x) translation, p[1] row (y) translation,
//   p[2] x/x coefficient, p[3] x/y coefficient,
//   p[4] y/x coefficient, p[5] y/y coefficient.
// The model maps current-frame coordinates to reference coordinates:
//   ref_x = p[2]*x + p[3]*y + p[0]
//   ref_y = p[4]*x + p[5]*y + p[1].
struct GmParams {
  GmType type = GmType::kIdentity;
  std::array<std::int32_t, 6> p{0, 0, kGmOne, 0, 0, kGmOne};

  bool operator==(const GmParams& other) const {
    return type == other.type && p == other.p;
  }
  bool operator!=(const GmParams& other) const { return !(*this == other); }
};

GmParams make_identity_gm();
GmParams make_gm(GmType type, const std::array<std::int32_t, 6>& p);
bool is_identity_model(const GmParams& params);

// ---------------------------------------------------------------------------
// Correspondences and block MVs.
// ---------------------------------------------------------------------------

// One current->reference correspondence in 1/8 luma samples. `weight` must be
// positive; the fitter uses it as an integer weight.
struct GmPoint {
  std::int32_t src_x = 0;  // column, q8
  std::int32_t src_y = 0;  // row, q8
  std::int32_t dst_x = 0;  // column, q8
  std::int32_t dst_y = 0;  // row, q8
  std::int32_t weight = 1;
};

// One block's MV as a correspondence at the block centre. `x`/`y` are the
// centre in q8; `mv_x`/`mv_y` are in q8 (so dst = centre + mv).
struct GmBlockMv {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t mv_x = 0;
  std::int32_t mv_y = 0;
  std::int32_t weight = 1;
};

// A per-block warp model (for example local warp or warp candidates). The
// fitter samples the model at the centre and at four offsets of `radius` q8
// units so that rotation/zoom is retained. `model` maps current -> reference.
struct GmWarpSample {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t radius = 32;  // 4 luma samples
  GmParams model{};
  std::int32_t weight = 1;
};

// ---------------------------------------------------------------------------
// Fit options, diagnostics and results.
// ---------------------------------------------------------------------------

enum class GmFitModel {
  kTranslation = 0,  // p2..p5 identity; only p0/p1 fitted
  kRotZoom = 1,      // similarity: p4 = -p3, p5 = p2
  kAffine = 2,       // general affine
};

const char* gm_fit_model_name(GmFitModel model);

struct GmOptions {
  // Fitting.
  GmFitModel model = GmFitModel::kAffine;
  int min_points = 3;
  int max_points = 8192;
  int outlier_passes = 2;                 // 0 disables rejection
  std::int64_t outlier_threshold_q8 = 32; // 4 luma samples
  // Degradation / validation.
  bool allow_rotzoom_degrade = true;
  bool require_shear_valid = false;  // require §7.13.3.21 shear validity
  std::int32_t rotzoom_mismatch_q16 = 1 << 12;  // 1/16 scale mismatch
  std::int32_t min_scale_q16 = kGmOne / 3;      // reject < 1/3 scale
  std::int32_t max_scale_q16 = 3 * kGmOne;      // reject > 3x scale
  std::int64_t max_corner_displacement_q8 = 2048LL * 8;  // 2048 luma samples
  bool force_integer_mv = false;            // quantise translations to 1 pel
  bool quantize = true;                      // quantise to the syntax grid
  // Acceptance: a non-identity model must reduce the weighted correspondence
  // squared error by at least this percentage versus identity.
  int min_gain_percent = 1;
  // Optional frame geometry for corner projection validation. Zero disables.
  int frame_width = 0;
  int frame_height = 0;
};

struct GmDiagnostics {
  bool ok = false;             // fit produced a model
  bool accepted = false;       // final model is useful (identity or gain)
  bool codable = false;        // representable by global_motion_params()
  bool shear_valid = true;     // §7.13.3.21
  bool degraded = false;       // AFFINE -> ROTZOOM or model -> IDENTITY
  bool identity = true;
  int points_in = 0;
  int points_used = 0;
  int outlier_passes_run = 0;
  std::int64_t identity_sse_q8 = 0;
  std::int64_t model_sse_q8 = 0;
  double gain_db = 0.0;        // 10 log10(identity_sse / model_sse)
  double model_rmse_q8 = 0.0;  // sqrt(model_sse / total_weight)
  double max_residual_q8 = 0.0;
  std::int32_t shear_mismatch_q16 = 0;
  // When a fitted model is rejected for insufficient gain, the candidate
  // measurement is kept here so the caller can report why.
  double candidate_gain_db = 0.0;
  std::int64_t candidate_model_sse_q8 = 0;
  std::string reason;
};

struct GmEstimate {
  GmParams params{};  // raw or final, depending on the entry point
  GmDiagnostics diag{};
};

// Weighted correspondence error of a model. Errors are q8 sample units.
struct GmQuality {
  std::int64_t identity_sse_q8 = 0;
  std::int64_t model_sse_q8 = 0;
  std::int64_t total_weight = 0;
  double identity_rmse_q8 = 0.0;
  double model_rmse_q8 = 0.0;
  double gain_db = 0.0;
  bool improves = false;
};

GmQuality evaluate_gm_quality(const GmParams& params, std::span<const GmPoint> points);

// Fit only: returns Q16 parameters with the requested model. `diag.ok` is
// false for degenerate input (too few points, collinear, zero weight).
GmEstimate fit_gm_points(std::span<const GmPoint> points, const GmOptions& options);
GmEstimate fit_gm_block_mvs(std::span<const GmBlockMv> blocks, const GmOptions& options);
GmEstimate fit_gm_warps(std::span<const GmWarpSample> warps, const GmOptions& options);

// Quantise, degrade and validate a raw model. This is the step that makes a
// fitted model codable by global_motion_params(). The returned `params` is
// either the final model or IDENTITY when the model is not useful.
GmEstimate finalize_gm_params(const GmParams& raw, const GmOptions& options);

// Fit + finalize in one call.
GmEstimate estimate_gm_points(std::span<const GmPoint> points, const GmOptions& options);
GmEstimate estimate_gm_block_mvs(std::span<const GmBlockMv> blocks, const GmOptions& options);
GmEstimate estimate_gm_warps(std::span<const GmWarpSample> warps, const GmOptions& options);

// ---------------------------------------------------------------------------
// Model validation and projection.
// ---------------------------------------------------------------------------

// True when the model is representable by §5.18.9.1 (type-ranges and the
// syntax quantisation grid). On failure `reason` (when non-null) explains why.
bool gm_params_codable(const GmParams& params, std::string* reason);
bool validate_gm_params(const GmParams& params, const GmOptions& options, GmDiagnostics* diag);

// §7.12.2.2 get_warp_motion_vector. `x`/`y` are Q16 luma samples of the
// projected point. `precision` follows §6.19.7.x: 7 = 1/8, 6 = 1/4,
// 5 = 1/2, 4 = 1 pel. Output is (mv_y, mv_x) in 1/8 sample units.
enum class MvPrecision : int {
  kOnePel = 4,
  kHalfPel = 5,
  kQuarterPel = 6,
  kEighthPel = 7,
};

void project_gm_point(const GmParams& params, std::int32_t x_q16, std::int32_t y_q16,
                      std::int32_t* ref_x_q16, std::int32_t* ref_y_q16);

// Projection at the centre of a block, including the MV_LOW/MV_UPP clip, the
// §5.20.9.4/.5 clamp and lower_mv_precision(). `mi_col`/`mi_row` are in 4x4
// units; `bw4`/`bh4` are the block size in 4x4 units.
void get_warp_motion_vector(const GmParams& params, int mi_col, int mi_row, int bw4, int bh4,
                            int mi_cols, int mi_rows, MvPrecision precision, int mv[2]);

// Encoder decision for GLOBALMV (§5.20.7.6 / §7.13.3.14):
//   * GLOBALMV is usable when GmType > IDENTITY.
//   * The decoder uses the affine warp predictor when min(block_w, block_h) >= 8;
//     otherwise it uses the projected centre MV with normal MC.
//   * A frame with force_integer_mv = 1 disables warp prediction entirely
//     (§7.13.3.14); the caller must override `use_warp` in that case.
struct GmGlobalMvUse {
  bool usable = false;
  bool use_warp = false;
  int ref = 0;
  int mv[2] = {0, 0};  // 1/8 units
};

GmGlobalMvUse global_mv_for_block(const GmParams& params, int ref, int mi_col, int mi_row,
                                  int bw4, int bh4, int mi_cols, int mi_rows);

// Frame-level plan for the sequence/frame header bits.
struct GmFramePlan {
  bool sequence_enable_global_motion = false;
  bool use_global_motion = false;
  int num_total_refs = 0;
  std::array<GmParams, kMaxRefsPerFrame> params{};
};

GmFramePlan make_gm_frame_plan(std::span<const GmEstimate> estimates,
                               bool enable_sequence_global_motion);

// ---------------------------------------------------------------------------
// Small deterministic image helpers (tests and quality reporting only).
// ---------------------------------------------------------------------------

// Bilinear warp of an 8-bit plane by `params` (current -> reference). This is
// an encoder-side quality proxy, NOT the normative §7.13.3.19 predictor.
bool warp_plane_bilinear(const GmParams& params, const std::uint8_t* src, int src_w, int src_h,
                         int src_stride, std::uint8_t* dst, int dst_w, int dst_h, int dst_stride,
                         int bit_depth);

// SSE of a warped prediction against `ref`; returns -1 on invalid input.
std::int64_t warp_plane_sse(const GmParams& params, const std::uint8_t* ref, int ref_w,
                            int ref_h, int ref_stride, const std::uint8_t* cur, int cur_w,
                            int cur_h, int cur_stride, int bit_depth);

}  // namespace av2enc::inter::gme

#endif  // AV2ENC_ENCODER_INTER_GME_GME_H_
