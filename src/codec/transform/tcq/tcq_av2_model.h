// Static §9.3 coefficient rate model for the TCQ trellis.
//
// This is the TCQ counterpart of `src/encoder/rdoq/rdoq_av2_model.h`, and the
// only substantive difference is normative: `coeff_base` and
// `coeff_base_lf` select their CDF with `(tcqState >> 1) & 1`, exactly as
// §5.20.7.27 requires.  The rest of the coefficient syntax (EOB prefix and
// extra bits, `coeff_br`, the `read_quant` escape, signs) is unchanged.
//
// TCQ is luma-only and TX_CLASS_2D-only in the spec, so this model accepts
// plane 0 with `tx_class_2d = true` and rejects everything else.  CCTX never
// uses TCQ.
//
// Approximation, identical in spirit to the RDOQ model: contexts that depend
// on already-coded neighbouring levels are frozen from a caller-supplied
// reference base-level block (the scalar quantisation at half step, or a
// previous TCQ pass).  This keeps the DP additive.  The TCQ state itself is
// *not* frozen: the rate callback receives the exact state from the trellis,
// so the CDF selector is exact along the chosen path.
//
// All costs are `-log2(p)` from the static §8.3.1 COEFF_CDF_Q_CTXS slice.  The
// tables come from `src/encoder/m1/qctx/qctx_cdf_tables.h`, which already
// carries the normative `tcqState` axis.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "codec/transform/variants/tx_variants.h"
#include "tcq.h"

namespace av2enc::tcq::av2 {

struct BlockInfo {
  // TCQ is plane 0 only (§5.20.7.27 `useTcq`).
  int plane = 0;
  // Coded/adjusted transform geometry.  Only coded_w, coded_h, w, h and
  // tx_size are read.
  txv::TxParams params{};
  // §5.20.7.27 txb_skip context, luma range [0, 9].
  int txb_skip_context = 0;
  // 0..2 from AboveDcContext/LeftDcContext, luma only.
  int dc_sign_context = 0;
  // §8.3.1 COEFF_CDF_Q_CTXS slice for the raw frame base_q_idx (0..3).
  int qctx = 0;
  // The EOB context is `is_inter` for luma.
  bool is_inter = false;
  // Must be true: the caller must not invoke TCQ for a 1D transform class.
  bool tx_class_2d = true;
};

class Model {
 public:
  Model() = default;

  // `reference_base_levels_scan` is a base-level block in §5.20.7.30 scan
  // order, size params.w * params.h.  It is used only for neighbour contexts
  // and the escape predictor.
  bool init(const BlockInfo& info,
            std::span<const std::int32_t> reference_base_levels_scan,
            std::string* error = nullptr);

  RateModel rate_model() noexcept;

  std::int32_t reference_eob() const noexcept { return reference_eob_; }

  // Estimated bits for the reference base levels (TCQ state included).
  double reference_bits() const;

  static double coefficient_rate(void* context, std::int32_t scan_pos,
                                 std::int32_t state, std::int32_t base_level,
                                 std::int32_t next_state,
                                 std::int32_t signed_quant,
                                 std::int32_t level_context,
                                 std::int32_t is_eob);
  static double eob_rate(void* context, std::int32_t eob,
                         std::int32_t num_coefficients);

  // §5.20.7.30 diagonal scan.
  static void build_scan(std::int32_t w, std::int32_t h,
                         std::span<std::int32_t> scan);

 private:
  struct Position {
    std::int32_t pos = 0;
    std::int32_t row = 0;
    std::int32_t col = 0;
    std::int32_t is_lf = 0;
    std::int32_t base_levels = 2;
    std::int32_t base_max = 3;
    std::int32_t br_max = 3;
    std::int32_t pre_max = 6;
    std::int32_t max_level = 6;
    std::int32_t base_ctx = 0;
    std::int32_t eob_ctx = 0;
    std::int32_t br_ctx = 0;
    std::int32_t hr_pred = 0;
    std::int32_t is_dc = 0;
  };

  double coefficient_bits(std::int32_t scan_pos, std::int32_t base_level,
                          std::int32_t state, std::int32_t is_eob) const;
  double eob_bits(std::int32_t eob) const;
  const std::uint16_t* txb_skip_cdf() const;

  BlockInfo info_{};
  int qctx_ = 0;
  std::vector<Position> positions_;
  std::vector<std::int32_t> reference_scan_;
  std::int32_t area_ = 0;
  std::int32_t reference_eob_ = 0;
};

}  // namespace av2enc::tcq::av2
