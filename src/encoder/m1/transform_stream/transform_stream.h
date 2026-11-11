// M1 transform-stream adapter: TX-type selection + full-trellis RDOQ.
//
// This is the integration surface between the M1 TX stage and the two verified
// components:
//
//   * `src/encoder/tx_type/`  - legal intra TX sets and §5.20.8.2 symbols
//   * `src/encoder/rdoq/`     - coefficient-level trellis quantisation
//
// One call consumes a coded-size residual block plus geometry, the signalled
// intra mode, a Lagrangian lambda and a coefficient rate model, and returns the
// winner's quantised levels, EOB, exact TX-type syntax symbol and RD cost.
//
// Scope
// -----
//   * 8-bit lossy intra, every one of the 25 AV2 transform sizes.
//   * Luma (plane 0) signals a TX type; chroma never does.  A chroma caller
//     passes `dct_only = true` and the adapter returns `DCT_DCT` with no type
//     syntax, mirroring `compute_tx_type()`.
//   * `reduced_tx_set` 0..3, including INTRA_2 (`reduced_tx_set` 1/3).
//   * `eob <= 1` forces `DCT_DCT` (§5.20.8.2); non-DCT candidates whose RDOQ
//     result collapses to `eob <= 1` are dropped (they are not codable).
//
// Not in scope: lossless/FSC, inter TX sets, secondary transforms, bitstream
// writing.  The adapter never writes a byte and never changes a reference
// stream by itself.
//
// Numeric contract
// ----------------
// The CPU reference is compiled with `-ffp-contract=off`; the GPU path uses
// `-fmad=false`.  Candidate comparison is exact binary64 in this order:
//
//   1. lowest cost = coded-domain SSE + lambda * (coefficient/EOB/type bits)
//   2. lowest coded-domain SSE
//   3. lowest numeric TX_TYPE (DCT_DCT first)
//
// `levels` and `eob` follow the §5.20.7.30 diagonal scan that the M1 writer
// consumes: `levels[c]` is scan position `c`, trailing positions are zero.
//
// Rate model
// ----------
// The default is the AV2 §9.3 CDF model in `src/encoder/rdoq/rdoq_av2_model.h`
// (static qctx 0, base_q_idx <= 90), built from each candidate's scalar levels.
// `RateModelHooks::build` replaces it with an exact-entropy or measured model.
// The hook owns the model's context until `release` (or forever when `release`
// is null).  `RateModelHooks::type_bits` replaces the static §9.3 TX-type
// symbol cost.
//
// Lambda
// ------
// Pass an SSE-domain lambda in the *coded residual / forward coefficient*
// domain.  For the shipped QP 90 quantiser:
//
//   quant_step = 1.0 / params.quant_factor;
//   lambda     = rdo::lambda_sse_from_quant_step(quant_step, sse_scale);
//
// `sse_scale = 0.125` is the RDO-framework placeholder; recalibrate it at the
// operating point.  Do not pass `rdo::lambda_sse_from_qp()` unchanged for a
// different quantiser model.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "encoder/rdoq/rdoq.h"
#include "encoder/tx_type/tx_type.h"
#include "encoder/tx_variants/tx_variants.h"

namespace av2enc::m1::ts {

// Largest legal intra candidate set (WIDE_32/HIGH_32 with both long-side
// branches).  Mirrors txtype::kMaxTxTypeCandidates.
inline constexpr int kMaxTxTypeCandidates = txtype::kMaxTxTypeCandidates;

// §5.20.8.2 syntax classes, including INTRA_2.  `src/encoder/m1/m1_syntax.cpp`
// has the same enum without INTRA_2 because M1 writes reduced_tx_set = 0.
enum class TxTypeSyntaxKind : std::uint8_t {
  kNone = 0,       // no type syntax (DCTONLY / eob <= 1 / chroma)
  kIntraSet1 = 1,  // 7-symbol mode-dependent set
  kIntraSet2 = 2,  // 2-symbol set (DCT_DCT, ADST_ADST)
  kLong32 = 3,     // is_long_side_dct flag + 4-symbol set
  kLong64 = 4,     // 4-symbol set; is_long_side_dct is implied 1
};

// Exact §5.20.8.2 emission data for one selected TX_TYPE.
struct TxTypeSymbol {
  TxTypeSyntaxKind kind = TxTypeSyntaxKind::kNone;
  // Symbol for the `intra_tx_type` CDF, or -1 when no type is coded.
  std::int8_t symbol = -1;
  // True only for WIDE_32/HIGH_32: the `is_long_side_dct` flag is signalled.
  bool has_long_flag = false;
  // Value of `is_long_side_dct` (implied true for WIDE_64/HIGH_64).
  bool is_long_side_dct = false;
  // Number of symbols in the CDF used for `symbol` (7/2/4), 0 when none.
  std::int8_t cdf_symbols = 0;
  // Static §9.3 cost of the type syntax (flag included), in bits.
  double bits = 0.0;
  // False when the requested type cannot be signalled for this query/eob.
  bool valid = false;
};

// Classifies one TX_TYPE into its §5.20.8.2 syntax.  `eob` is the final EOB
// after quantisation.  `query.mapped_mode` must already be wide-angle mapped.
TxTypeSymbol classify_tx_type(const txtype::TxTypeQuery& query,
                              txtype::TxType type, int eob);

// Per-candidate rate-model build request.
struct RateModelRequest {
  txv::TxParams params{};
  int plane = 0;              // 0 = luma, 1 = U, 2 = V
  int txb_skip_context = 0;   // §5.20.7.27 context the writer will use
  int dc_sign_context = 0;    // 0..2, luma only
  // Scalar-quantised levels in §5.20.7.30 scan order, params.w * params.h.
  std::span<const std::int32_t> scalar_levels_scan;
  std::int32_t scalar_eob = 0;
  double quant_step = 1.0;
  double lambda = 0.0;
};

// Pluggable rate model.  All hooks are optional.
struct RateModelHooks {
  // Builds the per-candidate coefficient/EOB model.  When null the adapter
  // builds rdoq::av2::Model from the request.  On success `*model` must be
  // usable for the immediately following rdoq_quantize() call.
  bool (*build)(void* user, const RateModelRequest& request,
                rdoq::RateModel* model, std::string* error) = nullptr;
  // Releases a model returned by build().  Called after rdoq_quantize().
  void (*release)(void* user, rdoq::RateModel* model) noexcept = nullptr;
  void* user = nullptr;
  // Exact cost of the TX-type syntax.  When null the static §9.3 cost in
  // TxTypeSymbol::bits is used.
  double (*type_bits)(void* user, const txtype::TxTypeQuery& query,
                      const txtype::TxTypeCandidate& candidate) = nullptr;
};

// One block decision request.
struct Request {
  // Coded-size residual, row-major, tightly packed: coded_w * coded_h int16.
  const std::int16_t* residual_coded = nullptr;
  int coded_w = 0;  // 4..64
  int coded_h = 0;  // 4..64
  // Signalled intra mode (YMode/UVMode 0..12) and its angle delta.  The
  // adapter applies §5.20.7.29 wide_angle_mapping internally.
  int intra_mode = 0;
  int angle_delta = 0;
  // Frame header reduced_tx_set (§5.20.2), 0..3; ignored for chroma when
  // dct_only is set.
  int reduced_tx_set = 0;
  // Force DCT_DCT and code no TX type (compute_tx_type()'s chroma rule).
  bool dct_only = false;
  // §5.4.8 enable_idtx_intra.  M1 signals 0, which forbids any 1D identity
  // transform (IDTX, V_DCT, H_DCT, V/H_ADST, V/H_FLIPADST).  The verified
  // tx_type component also lists them for the long-32 sets, so an M1 caller
  // must pass false here.  Default true keeps the component's full set.
  bool enable_idtx_intra = true;
  // Distortion units per bit, in the coded residual / forward coefficient
  // domain.  Finite and >= 0.
  double lambda = 0.0;
  // Entropy contexts for the default AV2 model.  They are the contexts the
  // writer will use for this block; see the rate-model notes above.
  int plane = 0;
  int txb_skip_context = 0;
  int dc_sign_context = 0;
  // RDOQ knobs.  Defaults match the component; `level_cap = 0` means the
  // derived bound + 64-level margin.
  int prev_level_cap = 2;
  int level_cap = 0;
  bool allow_zero_to_one = false;
  // Optional rate model override; null uses the AV2 §9.3 CDF model.
  const RateModelHooks* hooks = nullptr;
  // Optional candidate subset (RDO pruning / parity tests).  Every type must
  // be legal for the query; an empty span means the full legal set.
  std::span<const txtype::TxType> candidate_types;
};

// One block decision result.
struct Decision {
  bool valid = false;
  bool codable = false;  // false for a non-codable per-candidate evaluation
  txtype::TxType type = txtype::TxType::kDctDct;
  txv::TxParams params{};  // includes tx_size, adjusted geometry and TX type
  TxTypeSymbol symbol;

  std::int32_t eob = 0;    // coded scan positions, 0 = all-zero block
  std::int32_t nonzero = 0;
  double sse = 0.0;        // coded-domain reconstruction SSE of the winner

  double coeff_rate_bits = 0.0;  // RDOQ coefficient + EOB bits
  double type_bits = 0.0;        // signalled TX-type bits (0 when kNone)
  double rate_bits = 0.0;        // coeff_rate_bits + type_bits
  double cost = 0.0;             // sse + lambda * rate_bits

  // RDOQ DP diagnostics (coefficient-domain cost, before type bits).
  double rdoq_distortion = 0.0;
  double rdoq_rate_bits = 0.0;
  double rdoq_cost = 0.0;
  std::int32_t max_level_searched = 0;
  bool level_range_truncated = false;

  int candidates_considered = 0;
  // Winner levels in §5.20.7.30 scan order, size params.w * params.h.
  std::vector<std::int32_t> levels;
  // Winner coded-size inverse residual, coded_w * coded_h.
  std::vector<std::int16_t> recon_coded;
};

// Runs the full TX-type + RDOQ decision for one block.
//
// On success `*decision` is fully populated and true is returned.  On failure
// `decision->valid` is false and `*error` (when non-null) explains why.
bool quantize_block(const Request& request, Decision* decision,
                    std::string* error = nullptr);

// Scalar-quantisation baseline under the same rate model and TX-type rule.
// Useful for measuring RDOQ gains without duplicating this logic.  The rate
// includes the signalled TX-type symbol.  `lambda` only affects `cost`.
bool quantize_block_scalar(const Request& request, Decision* decision,
                           std::string* error = nullptr);

// Evaluates every legal candidate (after `candidate_types` filtering) without
// selecting one.  `decisions` receives one entry per candidate in symbol order;
// `valid` means the evaluation succeeded and `codable` is the §5.20.8.2
// `eob <= 1` rule.  This is the CPU parity surface for the GPU batch path.
bool evaluate_candidates(const Request& request,
                         std::vector<Decision>* decisions,
                         std::string* error = nullptr);

// Convenience lambda for the shipped quantiser:
//   rdo::lambda_sse_from_quant_step(1 / params.quant_factor, sse_scale)
// Returns 0 for an invalid size or scale.
double default_lambda(const txv::TxParams& params, double sse_scale = 0.125);

}  // namespace av2enc::m1::ts
