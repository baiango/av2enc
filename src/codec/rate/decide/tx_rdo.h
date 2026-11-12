// TX-type selection and RDOQ for one intra transform block.
//
// Both decisions rank candidates with the same transform-stream rate adapter
// (transform_stream.h) and the same Lagrangian convention as the rest of this
// layer:
//
//     cost = SSE + lambda * rate_bits
//
// TX type
// -------
// Enumerates the spec-legal intra TX types for the block/mode
// (src/encoder/tx_type/), runs the real forward -> quantise -> inverse path for
// each (src/encoder/tx_variants/), and charges the type symbol plus the
// transform-stream coefficient bits.  This is stricter than the txtype default
// model, which only charges one bit per nonzero coefficient.
//
// RDOQ
// ----
// A coefficient-level trellis over scan order with a fixed previous-magnitude
// state and explicit EOB termination.  It follows src/encoder/rdoq/ but keeps
// all scratch in a caller-owned workspace, so the hot loop does not allocate.
// The rate callback is the same transform-stream adapter; contexts are frozen
// from the caller's scalar levels, which is the documented RDOQ approximation.
//
// Units
// -----
// RDOQ distortion is `(original - level * quant_step)^2` in the forward
// coefficient domain.  `quant_step` there is 1/TxParams::quant_factor (8.25 for
// M1 at QP 90).  The equivalent pixel-domain lambda is 64x larger:
//
//     lambda_coefficient = lambda_pixel / 64
//
// rdoq_lambda_from_pixel_lambda() performs that conversion.  TX-type SSE comes
// from txtype::evaluate_intra_tx_type and is already in the coded residual
// domain used by that component; pass the matching lambda.

#pragma once

#include "config.h"
#include "codec/transform/tx_type/tx_type.h"
#include "transform_stream.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace av2enc::still::rdo_decide {

inline constexpr int kMaxRdoqStates = 4; // prev_level_cap 0..3
inline constexpr std::int32_t kRdoqLevelMargin = 64;

struct TxTypeDecisionConfig {
  // Lambda in the distortion units of txtype::evaluate_intra_tx_type (the
  // coded residual domain).  Set rate_aware = false for the SSE-only baseline.
  double lambda = 0.0;
  bool rate_aware = true;
  // M1 signals enable_idtx_intra = 0 in the sequence header.  The long-side
  // TX sets still *list* IDTX / V_DCT / H_DCT, but the decoder rejects them in
  // that case, so the decision must filter them out.  Leave false for M1.
  bool enable_idtx_intra = false;
};

struct TxTypeDecision {
  bool valid = false;
  txtype::TxTypeCandidate candidate{};
  int eob = 0;
  int nonzero = 0;
  std::int64_t sum_abs = 0;
  double sse = 0.0;
  double type_bits = 0.0;
  double coefficient_bits = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  int candidate_index = 0;
  int candidates_considered = 0;
  int candidates_rejected = 0;
};

// `levels_scratch` and `recon_scratch` are used for candidate evaluation;
// `levels_out` and `recon_out` receive the winner.  All are caller-owned.  The
// stream workspace must be initialised internally from each candidate, so it
// is passed in to avoid a 28 KiB object per candidate.
bool decide_tx_type(const std::int16_t* residual_coded, int coded_w,
                    int coded_h, int mapped_mode, int reduced_tx_set,
                    bool dct_only, const TxTypeDecisionConfig& config,
                    std::span<std::int32_t> levels_scratch,
                    std::span<std::int16_t> recon_scratch,
                    std::span<std::int32_t> levels_out,
                    std::span<std::int16_t> recon_out,
                    TransformStreamRate& stream, TxTypeDecision& out,
                    std::string* error = nullptr);

struct RdoqConfig {
  double lambda = 0.0;    // coefficient-domain lambda
  double quant_step = 8.25; // forward coefficient step, > 0
  int prev_level_cap = 2; // state = min(|prev|, cap), 0..3
  int level_cap = 0;      // 0 = derived bound + safety margin
  bool allow_zero_to_one = false;
};

struct RdoqWorkspace {
  std::array<double, kMaxTransformArea + 1> eob_bits{};
  std::array<double, kMaxTransformArea + 1> suffix_zero{};
  std::array<std::int32_t, kMaxTransformArea> signs{};
  std::array<std::int32_t, kMaxTransformArea * kMaxRdoqStates> chosen_level{};
  std::array<std::uint8_t, kMaxTransformArea * kMaxRdoqStates> chosen_mode{};
  std::array<double, kMaxRdoqStates> dp_next{};
  std::array<double, kMaxRdoqStates> dd_next{};
  std::array<double, kMaxRdoqStates> dr_next{};
  std::array<double, kMaxRdoqStates> dp_cur{};
  std::array<double, kMaxRdoqStates> dd_cur{};
  std::array<double, kMaxRdoqStates> dr_cur{};
};

struct RdoqDecision {
  bool valid = false;
  std::int32_t eob = 0;
  double distortion = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  std::int32_t max_level_searched = 0;
  bool level_range_truncated = false;
};

bool rdoq_config_valid(const RdoqConfig& config) noexcept;

// Runs the trellis.  `scalar_levels_scan` seeds the rate contexts and is not
// modified.  `levels_out` has w*h entries and receives the winner in scan
// order; positions at/after eob are zero.  No allocation.
bool decide_rdoq(std::span<const double> original_scan,
                 std::span<const std::int32_t> scalar_levels_scan,
                 int plane, int tx_size, int w, int h,
                 int txb_skip_context, int dc_sign_context,
                 const RdoqConfig& config, RdoqWorkspace& workspace,
                 TransformStreamRate& stream, std::span<std::int32_t> levels_out,
                 RdoqDecision& out, std::string* error = nullptr);

// lambda_pixel / 64 for the M1 transform normalisation (coefficient = pixel/8).
double rdoq_lambda_from_pixel_lambda(double lambda_pixel) noexcept;

}  // namespace av2enc::still::rdo_decide
