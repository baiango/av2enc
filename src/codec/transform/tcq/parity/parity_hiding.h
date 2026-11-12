// AV2 v1.0.0 parity hiding: encoder level choice + normative decoder inverse.
//
// Scope
// -----
// This component implements every normative piece of AV2 parity hiding that
// the specification defines, plus an encoder-side quantiser that produces
// levels the hidden-parity syntax can carry exactly:
//
//   * §5.4.8  `enable_parity_hiding` and its exclusion by
//             `enable_tcq && !choose_tcq_per_frame`;
//   * §5.18.2 `allow_parity_hiding` (forced 0 for CodedLossless or allow_tcq);
//   * §5.20.7.27 `parityHiding`, `isHidden`, `sumAbs1`, the `coeff_base_ph`
//             DC symbol, `TileDcSignCdf[ptype][isHidden][ctx]`, the
//             `maxLevel = NUM_BASE_LEVELS + 1` DC override, and the
//             `Quant = 2 * Quant + sumAbs1` reconstruction;
//   * §5.20.7.28 `read_quant` including the `lvlShift = 1` hidden-DC escape
//             predictor and the `hrLevelAvg` update;
//   * §7.14.4  dequantisation interaction (parity hiding adds no dequant
//             shift; the reconstructed `Quant[]` is the full level).
//
// Normative vs encoder design
// ---------------------------
// `derive_syntax()`, `decode_syntax()` and the entropy reader/writer are
// spec-shaped.  The only non-normative part is the level *choice* in
// `quantize()`: the spec says nothing about how an encoder should satisfy
// `|DC| & 1 == sumAbs1`.  The candidate set is deliberately small and
// deterministic (see `parity_hiding.cpp`).
//
// Geometry
// --------
// A block is described by its §9.2 `TX_SIZE` index plus the transform class.
// `w`/`h` are the *adjusted* transform dimensions (`min(Tx_Width, 32)` and
// `min(Tx_Height, 32)`), exactly the dimensions `get_tx_row_col()` and
// `get_scan()` use.  `coded_w`/`coded_h` are the unadjusted dimensions; they
// matter for `eobMultisize` and for the size-context `txSzCtx`.
//
// CDF axes
// --------
// `qctx` is the §8.3.1 `COEFF_CDF_Q_CTXS` slice selected from the raw frame
// `base_q_idx` (`av2enc::still::qctx::slice_for_base_q_idx`).  `derive_syntax()` records
// the exact CDF identifier and context for every symbol so a caller can price
// or write it without re-deriving contexts.
//
// All arithmetic is integer except the static rate model.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::tcq::parity {

// ---------------------------------------------------------------------------
// Normative constants (§3 / §9.2)
// ---------------------------------------------------------------------------

inline constexpr std::int32_t kPhThresh = 4;          // PHTHRESH
inline constexpr std::int32_t kNumBaseLevels = 2;     // NUM_BASE_LEVELS
inline constexpr std::int32_t kCoeffBaseRange = 3;    // COEFF_BASE_RANGE
inline constexpr std::int32_t kMaxBaseBrRange = 6;    // NUM_BASE_LEVELS + COEFF_BASE_RANGE + 1
inline constexpr std::int32_t kLfNumBaseLevels = 4;   // LF_NUM_BASE_LEVELS
inline constexpr std::int32_t kLfBaseSymbols = 6;     // LF_BASE_SYMBOLS
inline constexpr std::int32_t kPhContexts = 5;        // COEFF_BASE_PH_CONTEXTS
inline constexpr std::int32_t kTxSizesAll = 25;
inline constexpr std::int32_t kAdjustedMaxDim = 32;

// §5.20.7.27 Tx_Width / Tx_Height, indexed by TX_SIZE 0..24.
inline constexpr std::int32_t kTxWidth[kTxSizesAll] = {
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64,
    4, 16, 8, 32, 16, 64, 4, 32, 8, 64, 4, 64};
inline constexpr std::int32_t kTxHeight[kTxSizesAll] = {
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32,
    16, 4, 32, 8, 64, 16, 32, 4, 64, 8, 64, 4};

// §5.20.7.30 scan order class (`get_tx_class`).
enum class TxClass : std::uint8_t { k2D = 0, kHoriz = 1, kVert = 2 };

// §9.3 CDF identifiers used by the coefficient path and the hidden DC.
enum class CdfKind : std::uint8_t {
  kNone = 0,
  kTxbSkip,
  kEobPt16,
  kEobPt32,
  kEobPt64,
  kEobPt128,
  kEobPt256,
  kEobPt512,
  kEobPt1024,
  kEobExtra,
  kCoeffBaseEob,
  kCoeffBaseLfEob,
  kCoeffBaseUv,
  kCoeffBaseLfUv,
  kCoeffBaseEobUv,
  kCoeffBaseLfEobUv,
  kCoeffBase,
  kCoeffBaseLf,
  kCoeffBasePh,
  kCoeffBr,
  kCoeffBrLf,
  kCoeffBrUv,
  kDcSign,
};

const char* cdf_kind_name(CdfKind kind) noexcept;

// ---------------------------------------------------------------------------
// Geometry and eligibility
// ---------------------------------------------------------------------------

struct Geometry {
  std::int32_t tx_size = 0;
  std::int32_t coded_w = 0;  // Tx_Width[tx_size]
  std::int32_t coded_h = 0;  // Tx_Height[tx_size]
  std::int32_t w = 0;        // min(coded_w, 32)
  std::int32_t h = 0;        // min(coded_h, 32)
  TxClass tx_class = TxClass::k2D;
  std::int32_t plane = 0;    // 0 luma, 1 U, 2 V
};

// Builds the geometry for a legal TX size.  Rejects tx_size outside 0..24 and
// plane outside 0..2.
bool make_geometry(std::int32_t tx_size, TxClass tx_class, std::int32_t plane,
                   Geometry* geometry, std::string* error = nullptr);

inline std::int32_t tx_size_sqr(std::int32_t tx_size) noexcept {
  static constexpr std::int32_t kSqr[kTxSizesAll] = {
      0, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3,
      0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 0, 0};
  return (tx_size >= 0 && tx_size < kTxSizesAll) ? kSqr[tx_size] : 0;
}

inline std::int32_t tx_size_sqr_up(std::int32_t tx_size) noexcept {
  static constexpr std::int32_t kSqrUp[kTxSizesAll] = {
      0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4,
      2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 4, 4};
  return (tx_size >= 0 && tx_size < kTxSizesAll) ? kSqrUp[tx_size] : 0;
}

// §9.2 / §5.20.7.27 txSzCtx used by the coefficient CDFs.
inline std::int32_t tx_sz_ctx(std::int32_t tx_size) noexcept {
  return (tx_size_sqr(tx_size) + tx_size_sqr_up(tx_size) + 1) >> 1;
}

// §5.20.7.27 eobMultisize = Min(log2(coded_w),5)+Min(log2(coded_h),5)-4.
std::int32_t eob_multisize(const Geometry& geometry) noexcept;

inline constexpr std::int32_t kNoCtx = -1;

// Header-level `parityHiding` precondition (§5.20.7.27).  This is the
// `allow_parity_hiding && !Lossless && plane == 0 && PlaneTxType != IDTX`
// expression.  `allow_tcq` is included because §5.4.8/§5.18.2 force
// allow_parity_hiding = 0 when the frame allows TCQ; accepting it here would
// create a stream the sequence header can never signal.
struct Eligibility {
  bool allow_parity_hiding = false;  // frame-level flag
  bool lossless = false;             // CodedLossless / segment lossless
  bool allow_tcq = false;            // frame-level allow_tcq
  bool tx_type_is_idtx = false;      // PlaneTxType == IDTX
  bool use_fsc = false;              // forward skip coding
  bool is_inter = false;             // EOB / txb_skip CDF context
};

bool parity_hiding_eligible(const Geometry& geometry,
                            const Eligibility& eligibility) noexcept;

// `get_lf_limits(row, col, txClass, plane)` (§5.20.7.27).
bool is_low_frequency(std::int32_t row, std::int32_t col, TxClass tx_class,
                      std::int32_t plane) noexcept;

// Syntax parameters at one raster position.
struct LevelLimits {
  std::int32_t base_levels = 2;  // 2 or 4
  std::int32_t base_max = 3;     // 3 or 5
  std::int32_t br_max = 3;       // 0 for LF chroma
  std::int32_t pre_max = 6;      // base_max + br_max; non-hidden maxLevel
  bool is_lf = false;
};

LevelLimits level_limits(std::int32_t row, std::int32_t col, TxClass tx_class,
                         std::int32_t plane) noexcept;

// CDF/context decision for one symbol.  `ctx2` is the TCQ axis of
// `coeff_base`/`coeff_base_lf`; it is always 0 when parity hiding is active
// (parity hiding and TCQ are mutually exclusive).
struct CdfDecision {
  CdfKind kind = CdfKind::kNone;
  std::int32_t ctx = 0;
  std::int32_t ctx2 = 0;
  std::int32_t symbols = 0;
};

// §5.20.7.27 coeff_base CDF selection.  `level_pre` is the first-pass
// `Level[][]` array in raster order.  `c` is the scan index, `eob` the block
// EOB.  All reference positions have higher scan indices and must be filled.
CdfDecision coeff_base_cdf_decision(
    std::int32_t c, std::int32_t eob, std::int32_t pos, std::int32_t row,
    std::int32_t col, std::span<const std::int32_t> level_pre,
    const Geometry& geometry, bool is_hidden, bool is_lf);

// §5.20.7.27 coeff_br CDF selection.
CdfDecision coeff_br_cdf_decision(
    std::int32_t pos, std::int32_t row, std::int32_t col,
    std::span<const std::int32_t> level_pre, const Geometry& geometry,
    bool is_lf);

struct EobCdfDecision {
  CdfKind kind = CdfKind::kNone;
  std::int32_t ctx = 0;
  std::int32_t symbols = 0;
  std::int32_t escape_bits = 0;
};

// §5.20.7.27 sign class: true when the sign uses TileDcSignCdf (the DC, or
// the col/row-zero coefficient of a HORIZ/VERT transform).
bool sign_is_dc_style(std::int32_t plane, TxClass tx_class, std::int32_t pos,
                      std::int32_t row, std::int32_t col) noexcept;

// §5.20.7.27 EOB prefix CDF selection (`eobMultisize` + plane/inter context).
EobCdfDecision eob_cdf_decision(const Geometry& geometry, bool is_inter);

// Base EOB from `eobPt` (before the `eob_extra` / extra-bit additions) and
// the inverse mapping used by the writer.
std::int32_t eob_from_pt(std::int32_t eob_pt) noexcept;
std::int32_t eob_pt_for_eob(std::int32_t eob) noexcept;

// §5.20.7.30 get_scan(): 2D diagonal, HORIZ column-major, VERT row-major.
void build_scan(std::int32_t w, std::int32_t h, TxClass tx_class,
                std::span<std::int32_t> scan) noexcept;
bool build_scan(const Geometry& geometry, std::span<std::int32_t> scan,
                std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Exact coefficient syntax (first pass + second pass)
// ---------------------------------------------------------------------------

struct PositionSyntax {
  std::int32_t scan_index = 0;
  std::int32_t pos = 0;  // raster index in the adjusted block
  std::int32_t row = 0;
  std::int32_t col = 0;
  bool is_lf = false;
  bool is_dc = false;

  // First pass: the level the entropy syntax reconstructs (`Level[][]` in the
  // spec).  For a hidden DC this is the `coeff_base_ph` symbol (0..3).
  std::int32_t level = 0;
  std::int32_t max_level = 6;

  // Base symbol.  `base_offset` is +1 for `coeff_base_eob` / `coeff_base_lf_eob`.
  CdfKind base_kind = CdfKind::kNone;
  std::int32_t base_ctx = 0;
  std::int32_t base_ctx2 = 0;  // the second index, when the CDF has one
  std::int32_t base_symbols = 0;
  std::int32_t base_symbol = 0;
  std::int32_t base_offset = 0;

  // coeff_br symbol (present only when the spec reads it).
  bool br_present = false;
  CdfKind br_kind = CdfKind::kNone;
  std::int32_t br_ctx = 0;
  std::int32_t br_symbols = 0;
  std::int32_t br_symbol = 0;

  // `read_quant` escape.  `escape_bits` is the exact number of bypass bits
  // the writer emits for `escape_x` at this position's predictor.
  bool escape_present = false;
  std::int32_t escape_x = 0;
  std::int32_t hr_predictor = 0;
  std::int32_t escape_bits = 0;

  // Sign syntax.
  bool sign_present = false;
  std::int32_t sign = 0;  // 1 == negative
  bool dc_style_sign = false;  // uses TileDcSignCdf[ptype][isHidden][ctx]
};

struct BlockSyntax {
  std::int32_t eob = 0;
  bool parity_hiding_eligible = false;
  bool is_hidden = false;
  std::int32_t sum_abs1 = 0;  // §5.20.7.27 sumAbs1; 0 if parityHiding is false
  std::int32_t num_nz = 0;    // §5.20.7.27 numNz; 0 if parityHiding is false
  std::int32_t dc_ctx = 0;    // TileCoeffBasePhCdf context

  // EOB syntax (§5.20.7.27 eob_pt / eob_extra).
  CdfKind eob_kind = CdfKind::kNone;
  std::int32_t eob_ctx = 0;
  std::int32_t eob_symbols = 0;
  std::int32_t eob_symbol = 0;
  std::int32_t eob_pt = 0;
  std::int32_t eob_escape_bits = 0;
  std::int32_t eob_escape_value = 0;
  std::int32_t eob_extra_symbol = 0;
  std::int32_t eob_extra_lsb_bits = 0;
  std::int32_t eob_extra_lsb_value = 0;

  // Scan order, size == eob.
  std::vector<PositionSyntax> positions;
};

struct DecodedBlock {
  // Signed §5.20.7.27 `Quant[]` in raster order, size == w*h.  Positions at
  // or after `eob` are zero.  The encoder's `quantize()` input/output arrays
  // are in §5.20.7.30 scan order; use `decoded_levels()` to convert.
  std::vector<std::int32_t> quant_levels;
  std::int32_t eob = 0;
  bool is_hidden = false;
  std::int32_t sum_abs1 = 0;
  std::int32_t cul_level = 0;    // Min(4, sum |Quant|)
  std::int32_t dc_category = 0;  // 0 none, 1 negative, 2 positive
};

// Builds the exact syntax for signed full levels in scan order.
//
// The levels must already satisfy the hidden-parity constraint
// (`|DC| & 1 == sumAbs1`) when `is_hidden` would be selected.  Use
// `quantize()`/`apply_parity_hiding()` to produce such levels.  Returns false
// with a diagnostic rather than silently changing a level.
bool derive_syntax(std::span<const std::int32_t> levels_scan,
                   const Geometry& geometry, const Eligibility& eligibility,
                   BlockSyntax* syntax, std::string* error = nullptr);

// Pure §5.20.7.27 inverse over a syntax record: recomputes `isHidden` and
// `sumAbs1` from the coded base/br symbols, then applies
// `Quant = 2 * Quant + sumAbs1` at the DC.  This is the reader oracle; the
// entropy reader in `parity_entropy.h` produces the same record from bytes.
bool decode_syntax(const BlockSyntax& syntax, const Geometry& geometry,
                   DecodedBlock* decoded, std::string* error = nullptr);

// Full-level magnitude a syntax record reconstructs, scan order.  This is the
// `Quant[]` array the §7.14.4 dequant consumes.
bool decoded_levels(const BlockSyntax& syntax, const Geometry& geometry,
                    std::span<std::int32_t> levels_scan,
                    std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Static §9.3 rate model (no CDF adaptation)
// ---------------------------------------------------------------------------

struct RateConfig {
  std::int32_t qctx = 0;           // 0..3
  std::int32_t txb_skip_ctx = 0;   // 0..9 for luma, 0..11 for V
  std::int32_t dc_sign_ctx = 0;    // 0..2
  bool is_inter = false;
};

// Exact `-log2(p)` bits for the whole block (txb_skip, EOB, coefficients,
// signs, and `read_quant` bypass bits).  Returns a negative value on error.
double block_rate_bits(const BlockSyntax& syntax, const Geometry& geometry,
                       const RateConfig& config);

// `read_quant` escape bit count for a value x and predictor.
std::int32_t escape_bit_count(std::int32_t x, std::int32_t predictor) noexcept;

// ---------------------------------------------------------------------------
// Encoder level choice
// ---------------------------------------------------------------------------

enum class QuantizeMode : std::uint8_t {
  // Adjust one coefficient (DC or AC) by one level so |DC| parity matches
  // sumAbs1, choosing the smallest reconstruction error.  This is the
  // classic "trellis off" behaviour.
  kMinDistortion = 0,
  // Evaluate the same candidates plus a no-hiding fallback and choose the
  // smallest `distortion + lambda * rate` with the static §9.3 model.
  kRateDistortion = 1,
};

struct QuantizeConfig {
  Eligibility eligibility{};
  QuantizeMode mode = QuantizeMode::kRateDistortion;
  // Reconstruction step in the forward coefficient domain.  Distortion is
  // measured as `(original - level * quant_step)^2`, the same convention as
  // `src/encoder/rdoq/` and `src/encoder/tcq/`.
  double quant_step = 1.0;
  // Optional per-position step (scan order, size == w*h).  Empty means use
  // `quant_step` everywhere.  Use this for a quantiser matrix: the §7.14.4
  // dequant step varies per coefficient, and parity hiding itself adds no
  // shift.
  std::span<const double> dequant_step{};
  double lambda = 0.0;
  RateConfig rate{};
  // Exact search over every AC position for small blocks.  Larger blocks use
  // a deterministic pruned candidate set (DC + the best few AC positions).
  std::int32_t exhaustive_max_area = 64;
  std::int32_t pruned_ac_candidates = 8;
};

struct QuantizeResult {
  // Signed full levels in scan order, size == w*h.  These are the values the
  // §7.14.4 dequant consumes (`Quant[]`).
  std::vector<std::int32_t> levels;
  BlockSyntax syntax;
  std::int32_t eob = 0;
  bool hidden = false;
  bool hiding_considered = false;
  bool hidable = false;       // numNz >= PHTHRESH and header allowed
  bool hiding_turned_off = false;
  std::int32_t adjusted_scan_index = -1;
  std::int32_t adjusted_delta = 0;  // +1 or -1 applied to |level|
  double distortion = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
};

bool quantize_config_valid(const QuantizeConfig& config) noexcept;

// Chooses levels so the hidden parity bit is signalled correctly.
//
// `original` and `levels_seed` are in §5.20.7.30 scan order and must both have
// `w*h` entries.  `levels_seed` is normally the scalar or RDOQ quantiser
// output.  The returned `levels` are always representable; `syntax` is built
// with `derive_syntax()` and round-tripped through `decode_syntax()` before
// being returned.
bool quantize(std::span<const double> original,
              std::span<const std::int32_t> levels_seed,
              const Geometry& geometry, const QuantizeConfig& config,
              QuantizeResult* result, std::string* error = nullptr);

// Convenience: scalar-quantise `original` at `quant_step`, then `quantize()`.
bool quantize_scalar(std::span<const double> original,
                     const Geometry& geometry, const QuantizeConfig& config,
                     QuantizeResult* result, std::string* error = nullptr);

// Post-processing entry point: fix `levels_seed` in place so the hidden DC
// parity is legal.  Kept separate from `quantize()` for callers that already
// own an RDOQ decision and only need the parity constraint.
bool apply_parity_hiding(std::span<const double> original,
                         std::span<std::int32_t> levels,
                         const Geometry& geometry, const QuantizeConfig& config,
                         BlockSyntax* syntax, std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Small helpers shared by the tests and the RD harness
// ---------------------------------------------------------------------------

std::int32_t clip3(std::int32_t low, std::int32_t high,
                   std::int32_t value) noexcept;
std::int32_t get_msb(std::uint32_t value) noexcept;

// §7.14.4 dequant of one coefficient with the parity-hiding path (no extra
// shift).  `dq` is the dequantiser for that position.
std::int32_t dequantize_level(std::int32_t level, std::int32_t dq,
                              std::int32_t shift) noexcept;

// Recomputed `hrLevelAvg` state after coding one escape.  `lvl_shift` is 1
// only for the hidden DC.
std::int32_t update_hr_level_avg(std::int32_t x, std::int32_t hr_level_avg,
                                 std::int32_t lvl_shift) noexcept;

}  // namespace av2enc::tcq::parity
