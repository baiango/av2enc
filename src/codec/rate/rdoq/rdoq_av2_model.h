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

#include "codec/entropy/qctx/qctx.h"
#include "codec/transform/variants/tx_variants.h"
#include "rdoq.h"

namespace av2enc::rdoq::av2 {

// Optional per-symbol sink.
//
// The default model charges every symbol with the ideal `-log2 p` of the
// static CDF.  An ablation (the R19 interval-width estimate) needs the *same
// symbol sequence* priced by a different coder: an interval accumulator or a
// real §8.2 encoder.  A sink receives one call per emitted symbol (with the
// CDF row the writer would use) and one call per raw literal bit group, and
// returns the bits it charged.  With no sink installed the model is bit-exact
// with its historical arithmetic.
class SymbolSink {
 public:
  virtual ~SymbolSink() = default;
  // `cdf` has n+1 entries (n-1 boundaries, base rate, use count).
  virtual double symbol(const std::uint16_t* cdf, int n, int symbol) = 0;
  // `bits` raw literal bits (no CDF), e.g. §5.20.7.28 escape bits.
  virtual double literal(double bits) = 0;
};

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
  // Optional coefficient CDF slice the model reads instead of the §9.3 static
  // slice `qctx`.  The R19 experiment points this at a copied §8.2 CDF image
  // (the live frame-start state).  Null keeps the static slice.
  const av2enc::still::qctx::CoeffCdfSlice* cdf_override = nullptr;
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
            std::string& error);

  RateModel rate_model() noexcept;

  // Scalar EOB (last non-zero scan position + 1), as derived during init().
  std::int32_t scalar_eob() const noexcept { return scalar_eob_; }

  // Diagnostics: estimated bits for the scalar path (coefficients + EOB).
  double scalar_bits() const;

  // Optional alternative coder.  Null (the default) restores the built-in
  // `-log2 p` accounting; see SymbolSink.  The sink is per-block: install it
  // after init() and call scalar_bits() once.
  void set_symbol_sink(SymbolSink* sink) noexcept { sink_ = sink; }

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

  // Testing hook: when disabled, init() always builds the per-block table,
  // even for an all-zero block, so a test can compare the shared zero table
  // against the generic build bit for bit.  Enabled by default; the encoder
  // never calls this.
  static void set_zero_table_cache_for_testing(bool enabled) noexcept;

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

  // Builds the per-scan-position context table from the coded "pre-level"
  // plane.  Pure: the result is a function of (w, h, plane, scan, level_pre)
  // only, which is what lets an all-zero block share one immutable table.
  // `error` is set (and false returned) if a context leaves its table range.
  static bool build_positions(std::int32_t w, std::int32_t h, int plane,
                              std::span<const std::int32_t> scan,
                              const std::vector<std::int32_t>& level_pre,
                              std::vector<Position>& out, std::string& error);

  // Context table for an all-zero block of this geometry, built once and
  // shared by every later all-zero block of the same (w, h, plane).  Such a
  // block is ~93% of Model::init() calls on the --nightly-b recipe and its
  // table is identical every time, so this is pure work elimination.  Returns
  // nullptr (caller falls back to the per-block build) for geometry outside
  // the cached range.  Lock-free on the hot path.
  static const std::vector<Position>* zero_positions(std::int32_t w,
                                                     std::int32_t h,
                                                     int plane) noexcept;

  BlockInfo info_{};
  int qctx_ = 0;  // == info_.qctx; clamped 0..3 in init()
  // The per-scan-position context table.  A block with any non-zero level owns
  // it in `owned_positions_`; an all-zero block points at a shared immutable
  // cache entry instead, because when every level is zero the whole table
  // (contexts, escape predictors) is a pure function of (w, h, plane) and
  // rebuilding it per candidate is pure wasted work.  Either way the table is
  // read-only once init() returns.  `shared_positions_` is null unless the
  // shared entry is in use, so a moved/copied model cannot leave it dangling
  // on its own buffer.
  std::vector<Position> owned_positions_;
  const std::vector<Position>* shared_positions_ = nullptr;
  const std::vector<Position>& positions() const noexcept {
    return shared_positions_ != nullptr ? *shared_positions_ : owned_positions_;
  }
  std::vector<std::int32_t> levels_scan_;
  std::int32_t area_ = 0;
  std::int32_t scalar_eob_ = 0;
  SymbolSink* sink_ = nullptr;

  // Coefficient CDF slice: the caller override when present, else the static
  // §9.3 qctx slice.
  const av2enc::still::qctx::CoeffCdfSlice& slice() const;
  // One symbol / one raw-bit group through the sink, or the static model.
  double cost_symbol(const std::uint16_t* cdf, int n, int symbol) const;
  double cost_literal(double bits) const;
};

// Diagonal (up-right) scan from §5.20.7.30 for a 2D transform class.  Exposed
// so a caller can build `scalar_levels_scan` from a raster coefficient block.
void build_scan(std::int32_t w, std::int32_t h,
                std::span<std::int32_t> scan);

}  // namespace av2enc::rdoq::av2
