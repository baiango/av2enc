// AV2 intra transform-type selection (CPU reference).
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//
//   §5.20.8.2  transform_type() syntax: which symbol is read, and the
//              `eob == 1` / `reduced_tx_set == 2` DCT_DCT overrides
//   §5.20.8.3  get_tx_set(): transform set from the TX size and frame flags
//   §5.20.7.29 compute_tx_type() + Size_Class / Md_Idx_To_Type
//   §6.19.8.2  transform set enum values
//   §9.2       Size_Class, Tx_Size_Sqr, Tx_Size_Sqr_Up, Md_Idx_To_Type
//   §9.3       default intra TX-type CDFs
//   §7.15      the transforms themselves, via `src/encoder/tx_variants/`
//
// What this component owns
// ------------------------
// `src/encoder/tx_variants/` implements *how* one transform block is
// transformed; this component implements *which* TX_TYPE to signal for an
// intra block.  It:
//
//   * derives the legal intra transform set for a TX size (§5.20.8.3),
//   * enumerates the codable TX_TYPEs in §5.20.8.2 symbol order, including the
//     mode-dependent `Md_Idx_To_Type` table and the long-side sets,
//   * runs the full forward -> quantize -> dequantize -> inverse path for every
//     candidate through `txv::*`, and costs it as `SSE + lambda * rate`,
//   * picks a winner with a deterministic tie-break.
//
// It does not write bits, choose partitions, or reconstruct a frame.  Chroma
// never signals a TX type (compute_tx_type() derives it), so this API is for
// luma; `TxTypeQuery::dct_only` mirrors `enable_chroma_dctonly` and forces
// DCT_DCT for callers that want the chroma rule.
//
// Spec facts that matter (and that the tests pin)
// -----------------------------------------------
//   * A length-32 1D transform is always Dct_Kernel32 (§7.15.2.1), so ADST and
//     FLIPADST alias DCT on a 32-sample dimension.  64-wide/tall blocks are
//     coded as adjusted 32-sample transforms, so this also applies to 64x16,
//     16x64 and every other 64-point size in that dimension.
//   * `eob == 1` forces DCT_DCT (`transform_type()`).  A non-DCT candidate is
//     therefore only codable when `eob >= 2`.
//   * `reduced_tx_set == 2` forces DCT_DCT for intra blocks.
//   * WIDE_32/HIGH_32 sets (16x32, 32x16, 8x32, ...) code an
//     `is_long_side_dct` flag and then one of four types; WIDE_64/HIGH_64
//     (16x64, 64x16, ...) imply `is_long_side_dct = 1` and code one of four.
//   * INTRA_1 (all sizes with sqr_up <= 16x16) codes one of 7 mode-dependent
//     types; INTRA_2 (reduced_tx_set == 1 or 3) codes DCT_DCT or ADST_ADST.
//
// Scope: 8-bit lossy intra.  Lossless IDTX/FSC and the inter-only DDT/FDDT
// 1D transforms are out of scope; the inter sets are represented by name only.
#pragma once

#include <cstdint>

#include "encoder/tx_variants/tx_variants.h"

namespace av2enc::txtype {

using txv::Tx1dType;
using txv::TxParams;
using txv::TxSize;
using txv::TxType;

// §3: INTRA_TX_TYPES = 7, INTRA_MODES = 13.
inline constexpr int kIntraTxTypes = 7;
inline constexpr int kNumIntraModes = 13;
// Largest legal intra set: WIDE_32/HIGH_32 with both is_long_side_dct values.
inline constexpr int kMaxTxTypeCandidates = 8;

// §6.19.8.2 transform set values.
enum class TxSet : std::int8_t {
  kDctOnly = 0,
  kWide64 = 1,
  kHigh64 = 2,
  kWide32 = 3,
  kHigh32 = 4,
  kIntra1 = 5,
  kIntra2 = 6,
  kInter1 = 7,
  kInter2 = 8,
  kDctIdtx = 9,
  kDctIdtxIddct = 10,
};

const char* tx_set_name(TxSet set);

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

struct TxTypeQuery {
  int coded_w = 0;   // coded TX width, 4..64
  int coded_h = 0;   // coded TX height, 4..64
  // Intra mode index after §5.20.7.29 wide_angle_mapping (0..12).  Use
  // map_intra_mode() to obtain it from a signalled YMode + AngleDeltaY.
  int mapped_mode = 0;
  // Frame header `reduced_tx_set` (§5.20.2), 0..3.
  int reduced_tx_set = 0;
  // Force DCT_DCT (chroma's enable_chroma_dctonly rule, or an RDO prefilter).
  bool dct_only = false;
};

// §5.20.7.29 wide_angle_mapping for `mode` (YMode 0..12) with angle_delta in
// [-3, 3] on a coded_w x coded_h block.  Returns the mapped mode index, or -1
// when the mode, delta, or size is invalid.  Non-directional modes ignore the
// delta and the size.
int map_intra_mode(int mode, int angle_delta, int coded_w, int coded_h);

// §5.20.8.3 get_tx_set(txSz, plane) for an intra block, i.e. is_inter = 0.
// Returns false for a size that is not one of the 25 AV2 TX sizes.
bool intra_tx_set(const TxTypeQuery& query, TxSet* set);

// One codable candidate and the symbols needed to signal it.
struct TxTypeCandidate {
  TxType type = TxType::kDctDct;
  // Symbol index for the intra_tx_type CDF of this set, or -1 when the set
  // codes no type at all (DCTONLY / reduced_tx_set == 2 / dct_only).
  std::int8_t symbol = -1;
  // Value of is_long_side_dct when the set codes it (WIDE_32/HIGH_32).
  bool is_long_side_dct = false;
  // True when this set signals is_long_side_dct (WIDE_32 / HIGH_32 only).
  bool codes_long_flag = false;
  TxParams params;  // fully derived transform parameters (row-major contract)
};

// The legal intra TX-type set for one size + mode, in §5.20.8.2 symbol order.
struct TxTypeSet {
  TxSet set = TxSet::kDctOnly;
  int count = 0;
  TxTypeCandidate candidates[kMaxTxTypeCandidates];
};

// Enumerates the codable candidates.  Returns false for an unsupported size or
// an out-of-range mode / reduced_tx_set.
//
// Candidate order: INTRA_1/INTRA_2 follow Md_Idx_To_Type symbol order; the long
// sets follow Tx_Type_Inv_Long[is_long_side_dct][orientation]; WIDE_32/HIGH_32
// list is_long_side_dct = 0 then 1.  Selection does not depend on this order
// (see the tie-break), but symbol coding does.
bool intra_tx_candidates(const TxTypeQuery& query, TxTypeSet* out);

// Every type that *some* intra mode can signal for this size, sorted by TxType
// value.  Use it for a mode/type joint search; a winner still has to be checked
// with tx_type_symbol() before it can be written.  Returns the count, or -1 for
// an unsupported size.
int intra_tx_types_for_size(int coded_w, int coded_h, int reduced_tx_set,
                            bool dct_only, TxType* out, int cap);

// Inverse of the candidate table: the symbols needed to signal `type`.
// Returns false when the type is not codable for this size/mode/set.
// `is_long_side_dct` is set to the signalled flag where one exists, to true for
// WIDE_64/HIGH_64 (the spec implies 1), and to false otherwise.
bool tx_type_symbol(const TxTypeQuery& query, TxType type, int* symbol,
                    bool* is_long_side_dct);

// §5.20.8.2: !is_inter && eob == 1 infers DCT_DCT.  A non-DCT type is only
// codable when eob >= 2 (eob == 0 means txb_skip; only DCT is reported).
inline bool candidate_codable(TxType type, int eob) {
  return eob >= 2 || type == TxType::kDctDct;
}

// ---------------------------------------------------------------------------
// Rate model
// ---------------------------------------------------------------------------

struct TxTypeRateInputs {
  TxTypeQuery query;
  TxTypeCandidate candidate;
  int eob = 0;             // 0..w*h, §5.20.7.30 TX_CLASS_2D scan order
  int nonzero = 0;         // quantized coefficients != 0
  std::int64_t sum_abs = 0;
  double sse = 0.0;        // reconstruction SSE in the coded domain
  double type_bits = 0.0;  // default static cost of the signalled type
};

// Pluggable rate estimate in bits.  The default model is an approximation, not
// the §8.2 entropy coder; see docs/negative-results.md.
using TxTypeRateFn = double (*)(const TxTypeRateInputs&, void* user);

struct TxTypeCostConfig {
  // cost = sse + lambda * rate_bits.  lambda is in SSE units per bit.
  double lambda = 0.0;
  // nullptr selects default_rate_bits().
  TxTypeRateFn rate_fn = nullptr;
  void* rate_user = nullptr;
  // Default model knobs: bits charged per nonzero coefficient, and whether the
  // static §9.3 type-symbol cost is included.
  double coefficient_bits = 1.0;
  bool use_static_type_cost = true;
};

// Static cost of the type syntax for this candidate: -log2(P(symbol)) from the
// §9.3 default CDF (plus the is_long_side_dct flag for WIDE_32/HIGH_32).
// 0 when the set codes no type.
double static_type_bits(const TxTypeQuery& query,
                        const TxTypeCandidate& candidate);
double static_type_bits(TxSet set, int tx_size_index,
                        const TxTypeCandidate& candidate);

// Default pluggable rate: static_type_bits() + coefficient_bits * nonzero.
double default_rate_bits(const TxTypeRateInputs& inputs, void* user);

// Applies cfg.rate_fn (or the default) to one candidate.
double tx_type_rate_bits(const TxTypeRateInputs& inputs,
                         const TxTypeCostConfig& config);

// ---------------------------------------------------------------------------
// Per-candidate evaluation and selection
// ---------------------------------------------------------------------------

struct TxTypeEval {
  TxTypeCandidate candidate;
  bool codable = false;   // candidate_codable(type, eob)
  int eob = 0;
  int nonzero = 0;
  std::int64_t sum_abs = 0;
  double sse = 0.0;
  double type_bits = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
};

// Runs the full forward -> quantize -> dequantize -> inverse path for one
// candidate on a coded-size residual (`coded_w * coded_h` int16, row-major,
// tightly packed).  `levels_out` (w*h int32) and `recon_out` (coded_w*coded_h
// int16) may be null.  Returns false on invalid arguments or when the
// candidate does not belong to the query's size.
bool evaluate_intra_tx_type(const std::int16_t* residual_coded,
                            const TxTypeQuery& query,
                            const TxTypeCandidate& candidate,
                            const TxTypeCostConfig& config, TxTypeEval* out,
                            std::int32_t* levels_out = nullptr,
                            std::int16_t* recon_out = nullptr);

// Evaluates every legal candidate (see intra_tx_candidates) into out[0..cap).
// Returns the number written, or -1 for invalid input.  Candidates with
// eob <= 1 are still evaluated and reported, with `codable == false` when they
// are not DCT_DCT.
int evaluate_intra_tx_candidates(const std::int16_t* residual_coded,
                                 const TxTypeQuery& query,
                                 const TxTypeCostConfig& config, TxTypeEval* out,
                                 int cap);

struct TxTypeChoice {
  bool valid = false;
  TxTypeCandidate candidate;
  TxTypeEval eval;
  int candidate_index = 0;       // position in the symbol-ordered set
  int candidates_considered = 0;
};

// Selects one TX type.
//
// Deterministic tie-break, applied in order:
//   1. lowest cost (sse + lambda * rate),
//   2. lowest reconstruction SSE,
//   3. lowest TxType numeric value (DCT_DCT first).
// Candidates that are not codable (eob <= 1 and type != DCT_DCT) are skipped.
// The legal set always contains DCT_DCT for every size, so a winner exists.
//
// `levels_out` (w*h int32) and `recon_out` (coded_w*coded_h int16) receive the
// winner's quantized coefficients and inverse-transform residual; both may be
// null.
bool choose_intra_tx_type(const std::int16_t* residual_coded,
                          const TxTypeQuery& query,
                          const TxTypeCostConfig& config, TxTypeChoice* out,
                          std::int32_t* levels_out = nullptr,
                          std::int16_t* recon_out = nullptr);

// Chooses among an explicit candidate list, e.g. an RDO-pruned subset or a
// parity-test pair.  The same cost, codability and tie-break rules apply.
// `candidates[i].params` must match the query size; `symbol` and
// `is_long_side_dct` must be filled as intra_tx_candidates() would.
bool choose_tx_type_from_candidates(
    const std::int16_t* residual_coded, const TxTypeQuery& query,
    const TxTypeCandidate* candidates, int count,
    const TxTypeCostConfig& config, TxTypeChoice* out,
    std::int32_t* levels_out = nullptr, std::int16_t* recon_out = nullptr);

// ---------------------------------------------------------------------------
// Scan / EOB helpers (§5.20.7.30, TX_CLASS_2D)
// ---------------------------------------------------------------------------

// Diagonal scan for an adjusted w x h transform (4/8/16/32 per axis).
// scan[c] is the raster position (y*w + x) of scan position c.
void diagonal_scan(int w, int h, std::uint16_t* scan);
// 0 when all levels are zero, otherwise (last nonzero scan position + 1).
int eob_from_levels(const std::int32_t* levels, int w, int h);

}  // namespace av2enc::txtype
