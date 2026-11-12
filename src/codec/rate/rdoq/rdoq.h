// AV2 encoder-side rate-distortion optimised quantisation (RDOQ).
//
// This is a *full* coefficient-level trellis, not the limited local search in
// `src/encoder/rdo/`.  It is spec-shaped but remains independent of the
// bitstream: the caller supplies the per-coefficient rate through a callback,
// exactly like the RDO framework does.  The entropy owner can therefore plug in
// the normative §8.3 coefficient costs (or a measured table) without changing
// this component.
//
// Contract
// --------
//   * Input is the unquantised forward transform block in scan order.
//   * Output is the signed quantised level for every scan position plus the
//     end-of-block (EOB) position, in the same scan order the syntax writer
//     consumes.
//   * Distortion is `(original - level * quant_step)^2`.  `quant_step` is the
//     reconstruction step in the *forward coefficient* domain: the scalar
//     quantiser satisfies `level = lround(original / quant_step)`.
//   * Rate is in bits.  The cost is `distortion + lambda * rate_bits`, with
//     lambda taken from the RDO framework (`rdo::lambda_sse_from_quant_step`).
//   * The trellis state is the previous coefficient magnitude clamped to
//     `[0, prev_level_cap]`.  The default cap 2 is the three-state model used by
//     the AV2 base-level contexts; callers with a richer entropy model may
//     raise it.
//   * Termination is modelled explicitly.  EOB is canonical: positions after
//     the last non-zero coefficient are not coded, and the block may only end
//     at position 0 or immediately after a non-zero coefficient.  The last
//     non-zero coefficient is offered to the rate callback with `is_eob = 1`
//     so it can use the `coeff_base_eob` syntax.
//   * The search is safe for any finite, non-negative rate.  The DP evaluates
//     every level up to the distortion minimum, then keeps going while the
//     local distortion alone is below the best total cost found for that
//     state; once it exceeds that cost, no higher level can win (distortion is
//     non-decreasing past the minimum and both rate and future cost are
//     non-negative).  The loop cap is the classic `floor(2*c/s)+1` bound plus a
//     64-level margin (the AV2 CDF costs are not perfectly monotone in
//     magnitude), or `Config::level_cap` when set.  Reaching the cap without
//     the stop rule sets `level_range_truncated`.
//
// Determinism
// -----------
// All arithmetic is binary64 with `-ffp-contract=off` in the component build.
// Equal-cost candidates are ordered by: total cost, total distortion, total
// rate, smaller magnitude, terminal before non-terminal, then smaller signed
// level.  The order is a total order on the DP path from each state, so the
// output is deterministic for a given build and callback.  Build CPU and GPU
// with the same flags before comparing decisions bit-for-bit.

#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace av2enc::rdoq {

// `Coefficient::scalar_level` sentinel: derive it from `original/quant_step`.
inline constexpr std::int32_t kAutoScalarLevel =
    std::numeric_limits<std::int32_t>::min();

// One unquantised forward coefficient in scan order.
struct Coefficient {
  double original = 0.0;
  // Opaque caller context, copied into the rate callback.  Examples: a band /
  // position class, an entropy-model row, or the index of a per-position
  // context record owned by `RateModel::context`.
  std::int32_t level_context = 0;
  // Scalar-quantised seed level.  The trellis does not depend on it; it is
  // reported by `estimate_scalar_levels()` and used to choose the coefficient
  // sign when `original == 0`.  Set to kAutoScalarLevel to derive it.
  std::int32_t scalar_level = kAutoScalarLevel;
};

struct Config {
  double lambda = 0.0;      // SSE-domain lambda, finite and >= 0
  double quant_step = 1.0;  // reconstruction step, finite and > 0
  // State is min(|previous_level|, prev_level_cap).  Must be in [0, 63]; the
  // AV2 base-level contexts saturate well below that.
  std::int32_t prev_level_cap = 2;
  // Additional upper bound on the searched magnitude.  0 means use the
  // derived bound plus the 64-level safety margin described above.  A positive
  // value clips the loop cap and sets RdoqResult::level_range_truncated when it
  // actually bites.  The distortion stop rule still applies inside the cap.
  std::int32_t level_cap = 0;
  // Let a coefficient whose scalar level is zero try a non-zero level.  Those
  // levels are normally dominated; the exhaustive validation tests enable this
  // to exercise the full state space.
  bool allow_zero_to_one = false;
  // Safety cap for the reference path.
  std::int32_t max_coefficients = 4096;
};

bool rdoq_config_valid(const Config& config) noexcept;
bool validate_rdoq_config(const Config& config, std::string* error = nullptr);

// Estimated bits for coding one signed coefficient level.
//
//   context        RateModel::context
//   scan_pos       scan position in [0, num_coefficients)
//   prev_abs_state min(|previous coefficient|, prev_level_cap)
//   level          signed candidate level (0, or sign(original) * magnitude)
//   level_context  Coefficient::level_context for this position
//   is_eob         1 when this is the last non-zero coefficient of the block,
//                  so the model should use the coeff_base_eob syntax
//
// The callback must return a finite, non-negative number of bits.  It does not
// need to be monotone in |level|; the trellis uses a distortion-based stop
// rule rather than a monotonicity assumption.
using CoefficientRateFn = double (*)(void* context,
                                     std::int32_t scan_pos,
                                     std::int32_t prev_abs_state,
                                     std::int32_t level,
                                     std::int32_t level_context,
                                     std::int32_t is_eob);

// Estimated bits for signalling the block end.
//
//   eob                0 for an all-zero block, otherwise the number of coded
//                      coefficients (the last one is non-zero)
//   num_coefficients   block area in scan positions
//
// The all-zero/not-all-zero flag belongs here, so eob == 0 is the total cost of
// the skipped block.  A null callback uses the documented bring-up model.
using EobRateFn = double (*)(void* context,
                             std::int32_t eob,
                             std::int32_t num_coefficients);

struct RateModel {
  void* context = nullptr;
  CoefficientRateFn coefficient_rate = nullptr;
  EobRateFn eob_rate = nullptr;
};

// Bring-up model.  It is deliberately simple and is *not* an entropy estimate;
// tests and harnesses use it, production decisions must replace it.  It is
// non-decreasing in |level|.
double default_coefficient_rate(void* context,
                                std::int32_t scan_pos,
                                std::int32_t prev_abs_state,
                                std::int32_t level,
                                std::int32_t level_context,
                                std::int32_t is_eob) noexcept;
double default_eob_rate(void* context,
                        std::int32_t eob,
                        std::int32_t num_coefficients) noexcept;
RateModel default_rate_model() noexcept;

struct RdoqResult {
  std::vector<std::int32_t> levels;  // signed, scan order, size == input count
  std::int32_t eob = 0;              // number of coded coefficients
  double distortion = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;  // distortion + lambda * rate_bits
  // Largest magnitude the DP actually searched (diagnostic).
  std::int32_t max_level_searched = 0;
  // True when the loop cap (derived+margin or Config::level_cap) was reached
  // without the distortion stop rule proving that higher levels cannot win.
  bool level_range_truncated = false;
};

// Runs the trellis.  Returns false and sets `error` on an invalid config, a
// non-finite coefficient, an invalid rate from a callback, or an oversized
// block.  `result` is reset before any work, so a failed call leaves it empty.
bool rdoq_quantize(std::span<const Coefficient> coefficients,
                   const Config& config,
                   const RateModel& rate_model,
                   RdoqResult& result,
                   std::string* error = nullptr);

// Derives the largest magnitude the trellis searches for one coefficient under
// `config` (derived bound + 64-level margin, or `Config::level_cap` when set).
// Returns -1 if the coefficient is not finite.  Exposed for GPU/table builders
// that must search exactly the same range as the CPU.
std::int32_t rdoq_level_bound(const Coefficient& coefficient,
                              const Config& config) noexcept;

// Computes the scalar baseline levels (`lround(original/quant_step)`) and EOB
// for the same input.  Provided so callers can report the baseline without
// duplicating the rounding rule.  `levels` receives signed levels in scan
// order; `eob` the last non-zero position + 1.
bool estimate_scalar_levels(std::span<const Coefficient> coefficients,
                            double quant_step,
                            std::span<std::int32_t> levels,
                            std::int32_t* eob,
                            std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Batch interface
// ---------------------------------------------------------------------------
//
// The batch form is pointer/offset based and allocation-free, so the same
// descriptor can be handed to a device kernel (raw pointers, no std::span in
// the POD).  Blocks may have different sizes, quant steps and lambdas; the
// rate model is shared and therefore host-only.  A GPU implementation can
// replace the callback with a device-side table or closed-form model while
// keeping this layout.

struct BatchBlock {
  std::int32_t coefficient_offset = 0;  // index into BatchInput::original
  std::int32_t num_coefficients = 0;
  std::int32_t level_offset = 0;  // index into BatchOutput::levels
  double quant_step = 1.0;
  double lambda = 0.0;
  std::int32_t prev_level_cap = 2;
  std::int32_t level_cap = 0;
  std::int32_t flags = 0;  // bit 0: allow_zero_to_one
};

enum BatchBlockFlags : std::int32_t {
  kBatchAllowZeroToOne = 1 << 0,
};

struct BatchConfig {
  std::int32_t max_coefficients = 4096;
};

struct BatchInput {
  std::span<const double> original;
  // Optional per-coefficient context; may be empty (all zero) or exactly the
  // same length as `original`.
  std::span<const std::int32_t> level_context;
  // Optional scalar seed; may be empty (derived) or the same length as
  // `original`.
  std::span<const std::int32_t> scalar_level;
  std::span<const BatchBlock> blocks;
};

struct BatchOutput {
  std::span<std::int32_t> levels;   // enough for every block's level_offset
  std::span<std::int32_t> eob;      // one per block
  std::span<double> distortion;     // one per block
  std::span<double> rate_bits;      // one per block
  std::span<double> cost;           // one per block
  std::span<std::int32_t> status;   // one per block: 0 = ok, -1 = failed
};

bool rdoq_quantize_batch(const BatchInput& input,
                         const BatchConfig& config,
                         const RateModel& rate_model,
                         BatchOutput& output,
                         std::string* error = nullptr);

}  // namespace av2enc::rdoq
