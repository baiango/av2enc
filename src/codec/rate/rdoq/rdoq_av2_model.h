// AV2 §7.16 / §8.3 coefficient rate model for the RDOQ core.
//
// This is an *example* rate model, not part of the normative entropy coder.
// It computes the ideal code length of the coefficient syntax used by the M1
// writer (`src/encoder/m1/still_syntax.cpp`) from the static §9.3 CDFs in
// `src/encoder/m1/qctx/` (all four COEFF_CDF_Q_CTXS slices).  `BlockInfo::qctx`
// selects the slice from the raw frame base_q_idx; the default 0 keeps the
// historical QP <= 90 model.
//
// The model is deliberately static: M1 disables CDF adaptation, so the
// per-symbol cost is `-log2(p)` from the initial CDF.  The range coder is
// within a fraction of a bit of this estimate per block.  The entropy owner
// can replace it with exact measured costs through the same callback.
//
// Approximation (documented, not hidden)
// --------------------------------------
// The AV2 base/BR contexts are derived from already-coded neighbouring levels.
// A trellis that only carries the previous level cannot recompute them for
// every candidate.  This model freezes the contexts from the *scalar*
// quantisation supplied by the caller:
//   * the base / BR context neighbour sums are computed from the scalar
//     levels at positions below/right of the coefficient;
//   * the escape `hr_level_avg` predictor is the scalar reverse-scan state;
//   * the EOB context and the txb_skip context are supplied by the caller
//     (the block writer knows the above/left cumulative levels).
// Coefficients that move relative to the scalar path therefore use the scalar
// context, which is the standard RDOQ approximation and keeps the DP additive.
//
// Everything here is spec-derived.  The only tables are the normative default
// CDFs; no AVM/SVT/clave code is used.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "codec/transform/variants/tx_variants.h"
#include "rdoq.h"

namespace av2enc::rdoq::av2 {

// Per-block inputs the writer already knows.
struct BlockInfo {
  // 0 = luma, 1 = U, 2 = V.
  int plane = 0;
  // Coded and adjusted transform geometry.  Only the fields listed are read:
  // coded_w, coded_h, w, h, tx_size.
  txv::TxParams params{};
  // Exact §5.20.7.27 txb_skip context for this block.  For luma it is
  // `(above_or + left_or + 3) >> 1` (or 0 when the leaf matches the transform
  // size); chroma adds the plane offsets documented in the M1 writer.
  int txb_skip_context = 0;
  // DC sign context 0..2 from the above/left DC categories (luma only).
  int dc_sign_context = 0;
  // §8.3.1 COEFF_CDF_Q_CTXS slice for the raw frame base_q_idx (0..3).
  // 0 reproduces the historical base_q_idx <= 90 model.
  int qctx = 0;
  // §8.3.2 luma CDF select: txb_skip uses txb_skip[is_inter], and the EOB
  // plane context is `plane > 0 ? 2 : (is_inter ? 1 : 0)`.  False keeps the
  // intra tables, so every still/single-picture caller is unchanged.
  bool is_inter = false;
};

// Static, per-block coefficient cost model.  Build it from the scalar levels,
// then hand `rate_model()` to rdoq_quantize().
class Model {
 public:
  Model() = default;

  // `scalar_levels_scan` is the scalar-quantised block in scan order, size
  // params.w * params.h.  Returns false for an unsupported size/context.
  bool init(const BlockInfo& info,
            std::span<const std::int32_t> scalar_levels_scan,
            std::string* error = nullptr);

  RateModel rate_model() noexcept;

  // Scalar EOB (last non-zero scan position + 1), as derived during init().
  std::int32_t scalar_eob() const noexcept { return scalar_eob_; }

  // Diagnostics: estimated bits for the scalar path (coefficients + EOB).
  double scalar_bits() const;

  // Direct callbacks (exposed for tests).
  static double coefficient_rate(void* context,
                                 std::int32_t scan_pos,
                                 std::int32_t prev_abs_state,
                                 std::int32_t level,
                                 std::int32_t level_context,
                                 std::int32_t is_eob);
  static double eob_rate(void* context,
                         std::int32_t eob,
                         std::int32_t num_coefficients);

 private:
  struct Position {
    std::int32_t pos = 0;      // raster index
    std::int32_t row = 0;
    std::int32_t col = 0;
    std::int32_t is_lf = 0;
    std::int32_t base_levels = 2;
    std::int32_t base_max = 3;
    std::int32_t br_max = 3;
    std::int32_t pre_max = 6;
    std::int32_t max_level = 6;
    std::int32_t base_ctx = 0;   // normal coeff_base context
    std::int32_t eob_ctx = 0;    // coeff_base_eob context
    std::int32_t br_ctx = 0;
    std::int32_t hr_pred = 0;    // escape predictor before this position
    std::int32_t is_dc = 0;
  };

  double coefficient_bits(std::int32_t scan_pos,
                          std::int32_t signed_level,
                          std::int32_t is_eob) const;
  double eob_bits(std::int32_t eob) const;
  const std::uint16_t* txb_skip_cdf() const;
  std::int32_t scalar_level(std::int32_t scan_pos) const;

  BlockInfo info_{};
  int qctx_ = 0;  // == info_.qctx; clamped 0..3 in init()
  std::vector<Position> positions_;
  std::vector<std::int32_t> levels_scan_;
  std::int32_t area_ = 0;
  std::int32_t scalar_eob_ = 0;
};

// Diagonal (up-right) scan from §5.20.7.30 for a 2D transform class.  Exposed
// so a caller can build `scalar_levels_scan` from a raster coefficient block.
void build_scan(std::int32_t w, std::int32_t h,
                std::span<std::int32_t> scan);

}  // namespace av2enc::rdoq::av2
