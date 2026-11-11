// AV2 trellis-coded quantization (TCQ), encoder reference + normative
// decoder-side level mapping.
//
// Scope
// -----
// This component implements the parts of AV2 v1.0.0 that are *beyond*
// scalar/RDOQ coefficient quantisation:
//
//   * §5.20.7.27 `Tcq_Next_State` and the two-pass TCQ state sequence;
//   * the §5.20.7.27 level mapping  `Quant = Quant * 2 - ((state >> 1) & 1)`;
//   * the §7.14.4 dequant shift interaction  (`shift += 1` when
//     `allow_tcq && plane == 0 && !Lossless && txClass == TX_CLASS_2D &&
//     !useFsc`);
//   * an encoder trellis over the eight TCQ states, choosing the syntax base
//     level for each scan position;
//   * a decoder-side reconstruction routine so an encoder output can be
//     checked against the normative mapping without touching core files.
//
// CCTX (§5.4.8 `enable_cctx`, §5.20.7.27 `cctx_type`, §7.14.6) lives in
// `cctx.h`.  Header flag derivation lives in `tcq_flags.h`.
//
// Numeric contract
// ----------------
// The encoder input is the unquantised forward transform coefficient in the
// §5.20.7.30 scan order.  `Config::quant_step` is the *scalar* forward step S,
// i.e. scalar quantisation is `lround(original / S)`.  With TCQ enabled the
// decoder dequantises one extra shift, so the reconstructed step is S/2 and
// the final quantised magnitude is `m = 2*q - q0`, where `q` is the syntax
// base level (what `coeff_base`/`coeff_br` code) and `q0 = (state >> 1) & 1`.
// A coefficient's coded base level and final quant magnitude are both returned:
// the former is what the entropy writer consumes, the latter is what the
// inverse dequant consumes.
//
// Rate is supplied through the same style of callback as `src/encoder/rdoq/`,
// but the callback also receives the TCQ state and the next state.  That is
// required because the §5.20.7.27 `coeff_base` CDF is selected with
// `(tcqState >> 1) & 1`.  `tcq_av2_model.h` provides the static §9.3 model.
//
// All arithmetic is binary64 with `-ffp-contract=off`.  Tie-breaks are exact
// and documented in tcq.cpp.

#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace av2enc::tcq {

// ---------------------------------------------------------------------------
// Normative TCQ state machine (§5.20.7.27)
// ---------------------------------------------------------------------------

inline constexpr int kStateCount = 8;

// Tcq_Next_State[ 8 ][ 2 ], copied verbatim from §5.20.7.27.
inline constexpr std::int8_t kNextState[kStateCount][2] = {
    {0, 4}, {4, 0}, {1, 5}, {5, 1},
    {6, 2}, {2, 6}, {7, 3}, {3, 7},
};

// State transition for one coded coefficient parity (`level & 1`).  The
// decoder uses the parity of `read_quant`'s output; because read_quant only
// adds even escape increments, that parity equals the syntax base level parity.
constexpr int next_state(int state, int parity) noexcept {
  return kNextState[state & 7][parity & 1];
}

// q0 in §5.20.7.27: the parity bit subtracted from twice the magnitude.
// It is also the `coeff_base` CDF `tcqState` selector.
constexpr int state_q0(int state) noexcept { return (state >> 1) & 1; }
constexpr int state_cdf_index(int state) noexcept { return (state >> 1) & 1; }

// Syntax base level `l` the coefficient writer must emit for a full level
// `q` under TCQ.  `read_quant` adds `x << 1` when `l >= pre_max - 1`
// (`maxLevel - allowTcq`), so above that threshold both `pre_max` and
// `pre_max - 1` are usable; choosing by parity of `q` keeps every full level
// representable and the state parity equal to `q & 1`.
constexpr int syntax_level_for(int full_level, int pre_max) noexcept {
  if (full_level < pre_max - 1) return full_level;
  return ((full_level ^ pre_max) & 1) != 0 ? pre_max - 1 : pre_max;
}

// `read_quant` escape increment x for a full level.  Zero when the syntax
// level is below `pre_max - 1` (no escape syntax is present); otherwise
// `x = (full_level - syntax_level) / 2`.
constexpr int syntax_escape_for(int full_level, int pre_max) noexcept {
  const int level = syntax_level_for(full_level, pre_max);
  if (level < pre_max - 1) return 0;
  return (full_level - level) / 2;
}

// ---------------------------------------------------------------------------
// Dequant shift interaction (§7.14.4)
// ---------------------------------------------------------------------------

// Extra shift applied to the dequantiser when TCQ is active.  `tx_class_2d`
// is true for TX_CLASS_2D.  The condition is copied from §7.14.4.
inline constexpr int dequant_shift_extra(bool allow_tcq, int plane,
                                         bool lossless, bool tx_class_2d,
                                         bool use_fsc) noexcept {
  return (allow_tcq && plane == 0 && !lossless && tx_class_2d && !use_fsc)
             ? 1
             : 0;
}

// The full §7.14.4 shift: `(pels > 256) + (pels > 1024) + extra`.
inline constexpr int dequant_shift(int coded_pels, bool allow_tcq, int plane,
                                   bool lossless, bool tx_class_2d,
                                   bool use_fsc) noexcept {
  return (coded_pels > 256 ? 1 : 0) + (coded_pels > 1024 ? 1 : 0) +
         dequant_shift_extra(allow_tcq, plane, lossless, tx_class_2d, use_fsc);
}

// ---------------------------------------------------------------------------
// Encoder trellis
// ---------------------------------------------------------------------------

inline constexpr std::int32_t kAutoScalarLevel =
    std::numeric_limits<std::int32_t>::min();

// One unquantised forward coefficient in scan order.
struct Coefficient {
  double original = 0.0;
  // Opaque caller context forwarded to the rate callback.
  std::int32_t level_context = 0;
  // Optional scalar seed (at the scalar step S).  Used only to document the
  // baseline and to choose the sign of a zero-valued coefficient.  Set to
  // kAutoScalarLevel to derive it.
  std::int32_t scalar_level = kAutoScalarLevel;
  // `pre_max` from §5.20.7.27: 6 for a non-low-frequency 2D luma position,
  // 8 for `(row + col) < 4`.  It is the largest base level the
  // coeff_base/coeff_br syntax can carry directly.  TCQ escape increments are
  // `x << 1` and start at `pre_max - 1`, so `syntax_level_for()` picks
  // `pre_max` or `pre_max - 1` by parity.  The default is the HF value.
  std::int32_t pre_max = 6;
};

struct Config {
  double lambda = 0.0;      // SSE-domain lambda, finite and >= 0
  double quant_step = 1.0;  // scalar forward step S, finite and > 0
  // Upper bound on the syntax base level.  0 means the derived bound plus a
  // 64-level safety margin.
  std::int32_t level_cap = 0;
  std::int32_t max_coefficients = 4096;
  // Let a coefficient whose scalar level is zero try a non-zero level.
  bool allow_zero_to_one = false;
};

bool config_valid(const Config& config) noexcept;
bool validate_config(const Config& config, std::string* error = nullptr);

// Bits for coding one coefficient.
//
//   scan_pos       scan position in [0, num_coefficients)
//   state          TCQ state entering this coefficient (0..7)
//   base_level     signed full coefficient level (the writer's levels array);
//                  derive the syntax level with syntax_level_for()
//   next_state     Tcq_Next_State[state][syntax_level & 1]
//   signed_quant   signed final Quant magnitude after the TCQ mapping
//   is_eob         1 for the last non-zero coefficient (coeff_base_eob)
//
// A zero base level must return the cost of the coeff_base zero symbol.  The
// callback must return a finite, non-negative number of bits.
using CoefficientRateFn = double (*)(void* context,
                                     std::int32_t scan_pos,
                                     std::int32_t state,
                                     std::int32_t base_level,
                                     std::int32_t next_state,
                                     std::int32_t signed_quant,
                                     std::int32_t level_context,
                                     std::int32_t is_eob);

using EobRateFn = double (*)(void* context,
                             std::int32_t eob,
                             std::int32_t num_coefficients);

struct RateModel {
  void* context = nullptr;
  CoefficientRateFn coefficient_rate = nullptr;
  EobRateFn eob_rate = nullptr;
};

// Bring-up model; monotone, not an entropy estimate.
double default_coefficient_rate(void* context,
                                std::int32_t scan_pos,
                                std::int32_t state,
                                std::int32_t base_level,
                                std::int32_t next_state,
                                std::int32_t signed_quant,
                                std::int32_t level_context,
                                std::int32_t is_eob) noexcept;
double default_eob_rate(void* context,
                        std::int32_t eob,
                        std::int32_t num_coefficients) noexcept;
RateModel default_rate_model() noexcept;

struct Result {
  // Signed syntax base levels in scan order, size == input count.
  std::vector<std::int32_t> base_levels;
  // Signed final Quant values in scan order after the TCQ mapping, size ==
  // input count.  These are the values the §7.14.4 dequant consumes.
  std::vector<std::int32_t> quant_levels;
  // TCQ state entering each scan position; entries at/after eob are 0.
  std::vector<std::int32_t> states;
  std::int32_t eob = 0;
  double distortion = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  std::int32_t max_level_searched = 0;
  bool level_range_truncated = false;
};

bool quantize(std::span<const Coefficient> coefficients, const Config& config,
              const RateModel& rate_model, Result& result,
              std::string* error = nullptr);

std::int32_t level_bound(const Coefficient& coefficient,
                         const Config& config) noexcept;

bool estimate_scalar_levels(std::span<const Coefficient> coefficients,
                            double quant_step,
                            std::span<std::int32_t> levels, std::int32_t* eob,
                            std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Normative decoder-side mapping
// ---------------------------------------------------------------------------

struct DecodeResult {
  // Signed Quant values in scan order, size == area.
  std::vector<std::int32_t> quant_levels;
  // TCQ state entering each scan position.
  std::vector<std::int32_t> states;
  std::int32_t eob = 0;
};

// Applies the §5.20.7.27 two-pass level mapping to signed full coefficient
// levels in scan order.  `allow_tcq` is the frame-level flag.  When it is
// false (or `lossless` is true) the output is the input unchanged and no
// state runs.
//
// `pre_max_scan` is optional and supplies the §5.20.7.27 `pre_max` for each
// position (6 for HF, 8 for `(row + col) < 4`).  When empty, 6 is used.  The
// first-pass state parity is derived from `syntax_level_for(level, pre_max)`,
// which is the level the entropy syntax actually reconstructs.  Every full
// level is representable under TCQ because `pre_max - 1` and `pre_max` cover
// both parities.
bool decode_levels(std::span<const std::int32_t> base_levels_scan,
                   bool allow_tcq, bool lossless, DecodeResult* result,
                   std::string* error = nullptr,
                   std::span<const std::int32_t> pre_max_scan = {});

}  // namespace av2enc::tcq
