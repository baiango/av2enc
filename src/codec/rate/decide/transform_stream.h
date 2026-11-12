// Allocation-free rate estimate for one quantised transform stream.
//
// The M1 syntax writer consumes a flat, scan-order level stream per plane.  The
// TX-type and RDOQ decisions need the cost of that stream *before* bytes are
// written, and they need it for many candidates.  This adapter is the single
// place that maps
//
//     (plane, TX geometry, scan-order levels, EOB)
//
// to an estimated number of bits.  It is the default backend for the TX layer;
// the integrator can replace it with exact §8.2 entropy costs through the same
// callback shape as the RDO framework.
//
// Model
// -----
// The default backend is the static-CDF coefficient model used by
// src/encoder/rdoq/rdoq_av2_model.cpp: §9.3 default CDFs, with the
// COEFF_CDF_Q_CTXS slice selected by `TransformStreamBlock::qctx` (raw frame
// base_q_idx), and M1's disable_cdf_update = 1.  Per §5.20.7.28 it charges
// txb_skip, eob_pt + extra bits, coeff_base / coeff_base_eob, coeff_br,
// quant escape and signs.  Contexts are frozen from the supplied scalar level
// stream, which is the standard RDOQ approximation: a coefficient that moves
// relative to the scalar path keeps the context derived from the scalar path.
//
// The model is not a promise of byte-exactness.  It is deterministic, derived
// from the normative tables, and close enough to rank candidates; the entropy
// owner must still replace it for a BD-rate claim.
//
// Allocation
// ----------
// The object owns fixed 1024-coefficient arrays and never allocates.  A caller
// creates one workspace and reuses it for every block in the frame.

#pragma once

#include "codec/rate/rdoq/rdoq.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace av2enc::still::rdo_decide {

inline constexpr int kMaxTransformArea = 32 * 32;

struct TransformStreamBlock {
  int plane = 0;   // 0 = luma, 1 = U, 2 = V
  int tx_size = 0; // AV2 TX_SIZE enum index 0..24
  int w = 0;       // adjusted transform width (4..32)
  int h = 0;       // adjusted transform height (4..32)
  // §5.20.7.27 txb_skip context.  Luma 0..9, U 0..9, V 0..11.  The stream
  // adapter accepts the caller's value; 0 is a safe neutral default.
  int txb_skip_context = 0;
  // §5.20.7.26 DC sign context 0..2 (luma only).
  int dc_sign_context = 0;
  // §8.3.1 COEFF_CDF_Q_CTXS slice for the raw frame base_q_idx (0..3).
  // 0 reproduces the historical base_q_idx <= 90 model.
  int qctx = 0;
  std::span<const std::int32_t> levels_scan; // exactly w*h, scan order
  int eob = 0;                               // 0..w*h
};

class TransformStreamRate {
 public:
  // Optional pluggable rate override.  When set, coefficient_bits()/eob_bits()
  // call these instead of the static-CDF model.  This is the hook for exact
  // §8.2 entropy costs or a measured table; the rest of the decision layer
  // (TX type, RDOQ) consumes the same interface either way.
  using CoefficientRateCallback = double (*)(void* context,
                                             std::int32_t scan_pos,
                                             std::int32_t signed_level,
                                             bool is_eob);
  using EobRateCallback = double (*)(void* context, std::int32_t eob);

  TransformStreamRate() = default;

  // Builds the per-position context tables from `block`.  Returns false on an
  // invalid geometry/context or an inconsistent level span.  No allocation.
  bool init(const TransformStreamBlock& block, std::string* error = nullptr);

  // Bits for the block described at init(): EOB syntax plus every coefficient
  // below the supplied EOB.  Includes txb_skip (the all-zero flag).
  double block_bits() const noexcept;

  // Bits for one coefficient of this block.  `scan_pos` is the scan index,
  // `signed_level` is the candidate level, `is_eob` marks the last non-zero
  // coefficient (coeff_base_eob syntax).
  double coefficient_bits(int scan_pos, std::int32_t signed_level,
                          bool is_eob) const noexcept;

  // Bits for the EOB syntax at `eob`, including txb_skip.
  double eob_bits(int eob) const noexcept;

  int area() const noexcept { return area_; }
  int eob() const noexcept { return eob_; }

  void set_rate_override(CoefficientRateCallback coefficient,
                         EobRateCallback eob, void* context) noexcept {
    coefficient_override_ = coefficient;
    eob_override_ = eob;
    override_context_ = context;
  }
  void clear_rate_override() noexcept {
    coefficient_override_ = nullptr;
    eob_override_ = nullptr;
    override_context_ = nullptr;
  }

  // rdoq::RateModel plumbing.  The returned model borrows `*this`; it is valid
  // until the next init()/destruction.  It never allocates.
  av2enc::rdoq::RateModel rdoq_rate_model() noexcept;

  static double coefficient_rate_callback(void* context, std::int32_t scan_pos,
                                           std::int32_t prev_abs_state,
                                           std::int32_t level,
                                           std::int32_t level_context,
                                           std::int32_t is_eob);
  static double eob_rate_callback(void* context, std::int32_t eob,
                                  std::int32_t num_coefficients);

 private:
  struct Position {
    std::int32_t pos = 0; // raster index inside the adjusted block
    double hr_pred = 0.0; // escape predictor before this position
    std::int32_t base_ctx = 0;
    std::int32_t br_ctx = 0;
    std::int32_t eob_ctx = 0;
    std::int32_t base_levels = 2;
    std::int32_t base_max = 3;
    std::int32_t pre_max = 6;
    std::int32_t max_level = 6;
    std::uint8_t is_lf = 0;
    std::uint8_t is_dc = 0;
  };

  const std::uint16_t* txb_skip_cdf() const noexcept;

  std::array<std::int32_t, kMaxTransformArea> levels_{};
  std::array<std::int32_t, kMaxTransformArea> pre_levels_{};
  std::array<std::int32_t, kMaxTransformArea> scan_{};
  std::array<Position, kMaxTransformArea> positions_{};
  int plane_ = 0;
  int tx_size_ = 0;
  int coded_w_ = 0;
  int coded_h_ = 0;
  int w_ = 0;
  int h_ = 0;
  int area_ = 0;
  int eob_ = 0;
  int txb_skip_context_ = 0;
  int dc_sign_context_ = 0;
  int qctx_ = 0;
  CoefficientRateCallback coefficient_override_ = nullptr;
  EobRateCallback eob_override_ = nullptr;
  void* override_context_ = nullptr;
  bool valid_ = false;
};

}  // namespace av2enc::still::rdo_decide
