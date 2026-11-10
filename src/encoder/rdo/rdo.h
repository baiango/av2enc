#pragma once

// CPU reference RDO / RDOQ framework for av2enc.
//
// Scope:
//   - Lagrangian cost (distortion + lambda * rate).
//   - Distortion hooks: SSD, SAD, SATD (plus coefficient-domain SSE).
//   - A deterministic candidate/mode selector used by partition, intra, inter
//     and transform decisions.
//   - Lambda selection from a QP (base_q_idx) or a quantiser step.
//   - A limited RDOQ trellis for coefficient levels with a pluggable rate model.
//
// This component is deliberately independent of the bitstream, entropy and M0
// encoder sources. It performs no I/O and owns no global state. Rate values are
// supplied by the caller (typically an entropy-cost estimate or a calibrated
// model), so this code never writes bits and never changes a reference stream.
//
// Units:
//   - distortion: squared error (SSD) or sum of absolute values (SAD/SATD).
//   - rate: bits, not bytes. A caller with byte costs must multiply by 8.
//   - lambda: distortion units per bit. Mixing units gives meaningless
//     decisions, so use lambda_from_qp() for the same distortion metric that the
//     candidate costs use.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::rdo {

// ---------------------------------------------------------------------------
// Distortion metrics
// ---------------------------------------------------------------------------

enum class DistortionMetric : std::uint8_t {
  kSsd = 0,   // sum of squared differences
  kSad = 1,   // sum of absolute differences
  kSatd = 2,  // sum of absolute 4x4 Hadamard-transformed differences
};

// All plane metrics use 8-bit samples with a row stride in bytes. The visible
// region is width x height; samples outside it are ignored, including any
// padding inside the stride. Returns 0 for invalid dimensions or too-small
// spans. The functions are branch-free in the inner loop and safe on edges.
std::uint64_t ssd_u8(std::span<const std::uint8_t> a,
                     std::span<const std::uint8_t> b,
                     int width,
                     int height,
                     int stride) noexcept;

std::uint64_t sad_u8(std::span<const std::uint8_t> a,
                     std::span<const std::uint8_t> b,
                     int width,
                     int height,
                     int stride) noexcept;

// SATD is defined over 4x4 blocks. Partial 4x4 blocks at the right/bottom edge
// are zero-padded, so a 3x2 block computes one 4x4 transform. This is the
// usual encoder cost hook; it is not a normative transform.
std::uint64_t satd_u8(std::span<const std::uint8_t> a,
                      std::span<const std::uint8_t> b,
                      int width,
                      int height,
                      int stride) noexcept;

// Dispatch on the metric. Returns 0 for an out-of-range enum value.
double compute_distortion(DistortionMetric metric,
                          std::span<const std::uint8_t> a,
                          std::span<const std::uint8_t> b,
                          int width,
                          int height,
                          int stride) noexcept;

// Coefficient-domain squared error, used by RDOQ and transform-domain RDO.
// Iterates over min(original.size(), reconstructed.size()).
double coefficient_sse(std::span<const double> original,
                       std::span<const double> reconstructed) noexcept;

double coefficient_sse_i32(std::span<const std::int32_t> original,
                           std::span<const std::int32_t> reconstructed) noexcept;

// ---------------------------------------------------------------------------
// Lambda selection
// ---------------------------------------------------------------------------

// Heuristic quantiser model. base_q_idx 0..255 is mapped to a step that doubles
// every qp_per_octave index, and lambda_sse = sse_scale * step^2.
//
// These defaults are a calibration placeholder, not a normative AV2 table.
// They make lambda monotonic in QP and consistent with the quantiser step.
// The integration pass should calibrate sse_scale or, better, pass the real
// per-block quantiser step to lambda_sse_from_quant_step().
struct LambdaParams {
  double step_at_qp0 = 64.0;
  double qp_per_octave = 32.0;
  double sse_scale = 0.125;
};

bool lambda_params_valid(const LambdaParams& params) noexcept;
bool validate_lambda_params(const LambdaParams& params,
                            std::string* error = nullptr);

// Quantiser step for base_q_idx. base_q_idx is clamped to [0, 255].
// Returns 0 if `params` is invalid.
double quant_step_from_qp(int base_q_idx, LambdaParams params = {}) noexcept;

// lambda_sse = sse_scale * quant_step^2. Returns 0 for an invalid step/scale.
double lambda_sse_from_quant_step(double quant_step,
                                  double sse_scale = 0.125) noexcept;

double lambda_sse_from_qp(int base_q_idx, LambdaParams params = {}) noexcept;

// SAD/SATD costs are L1-like, so lambda is the square root of the SSE lambda.
// SATD has a larger dynamic range than SAD but the same dimensional scaling;
// the same value is used here and the caller may rescale if calibrated.
double lambda_sad_from_qp(int base_q_idx, LambdaParams params = {}) noexcept;
double lambda_satd_from_qp(int base_q_idx, LambdaParams params = {}) noexcept;

// Selects the lambda domain matching `metric`.
double lambda_from_qp(int base_q_idx,
                      DistortionMetric metric,
                      LambdaParams params = {}) noexcept;

// ---------------------------------------------------------------------------
// Candidate / mode selection
// ---------------------------------------------------------------------------

enum class CandidateKind : std::uint8_t {
  kPartition = 0,
  kIntra = 1,
  kInter = 2,
  kTransform = 3,
  kRdoq = 4,
};

// One pre-costed decision candidate. `distortion` and `rate_bits` come from
// the calling tool (partition tree, intra predictor, inter mode, transform).
// `id` and `aux` are caller-defined discriminators used only for deterministic
// tie-breaking; examples:
//   - partition: id = partition enum, aux = depth
//   - intra:     id = intra mode, aux = angle delta
//   - inter:     id = inter mode, aux = reference list index
//   - transform: id = transform type, aux = tx size
struct CandidateCost {
  CandidateKind kind = CandidateKind::kPartition;
  std::int32_t id = 0;
  std::int32_t aux = 0;
  double distortion = 0.0;
  double rate_bits = 0.0;
};

// distortion + lambda * rate_bits.
double lagrangian_cost(double distortion,
                       double rate_bits,
                       double lambda) noexcept;

double candidate_cost(const CandidateCost& candidate, double lambda) noexcept;

enum class TieBreak : std::uint8_t {
  // Equal total cost: prefer lower distortion, then lower rate.
  kDistortionThenRate = 0,
  // Equal total cost: prefer lower rate, then lower distortion.
  kRateThenDistortion = 1,
};

struct RdoConfig {
  double lambda = 0.0;
  DistortionMetric metric = DistortionMetric::kSsd;
  TieBreak tie_break = TieBreak::kDistortionThenRate;
};

bool rdo_config_valid(const RdoConfig& config) noexcept;
bool validate_rdo_config(const RdoConfig& config,
                         std::string* error = nullptr);

// Builds a config whose lambda matches `metric` for base_q_idx.
RdoConfig make_rdo_config(int base_q_idx,
                          DistortionMetric metric = DistortionMetric::kSsd,
                          TieBreak tie_break = TieBreak::kDistortionThenRate,
                          LambdaParams params = {});

struct CandidateScore {
  CandidateCost candidate{};
  double cost = 0.0;
  // Insertion index among valid candidates. Used only as the final tie-break,
  // so candidate generation must itself be deterministic.
  std::int32_t input_index = 0;
};

// True if `a` is strictly better than `b` under the total order:
//   1. lower total cost
//   2. lower distortion or lower rate (see TieBreak)
//   3. lower CandidateKind
//   4. lower id
//   5. lower aux
//   6. lower input_index
// Comparisons are exact (no epsilon), so the result is deterministic for a
// given build. Build with -ffp-contract=off when cross-checking CPU vs GPU.
bool candidate_better(const CandidateScore& a,
                      const CandidateScore& b,
                      TieBreak tie_break) noexcept;

struct RdoDecision {
  bool found = false;
  CandidateScore best{};
  CandidateScore second{};
  bool has_second = false;
  std::int32_t candidate_count = 0;  // valid candidates considered
  std::int32_t rejected_count = 0;   // invalid candidates ignored

  // best-to-second cost gap. 0 when fewer than two valid candidates.
  double margin() const noexcept {
    return has_second ? second.cost - best.cost : 0.0;
  }
};

// Streaming, allocation-free selector. Invalid candidates (negative or
// non-finite distortion/rate, or an invalid config) are rejected and counted.
class CandidateSelector {
 public:
  explicit CandidateSelector(const RdoConfig& config) noexcept;

  // Returns false and fills `error` if the candidate or config is invalid.
  bool add(const CandidateCost& candidate, std::string* error = nullptr);

  const RdoDecision& decision() const noexcept { return decision_; }
  void reset() noexcept;

 private:
  RdoConfig config_;
  RdoDecision decision_;
};

// Convenience wrapper around CandidateSelector.
RdoDecision select_best(std::span<const CandidateCost> candidates,
                        const RdoConfig& config,
                        std::string* error = nullptr);

// Factories. They only fill kind/id/aux; distortion and rate are caller data.
CandidateCost make_partition_candidate(int partition_id,
                                       double distortion,
                                       double rate_bits);
CandidateCost make_intra_candidate(int mode_id,
                                   double distortion,
                                   double rate_bits);
CandidateCost make_inter_candidate(int mode_id,
                                   int reference_index,
                                   double distortion,
                                   double rate_bits);
CandidateCost make_transform_candidate(int tx_type_id,
                                       double distortion,
                                       double rate_bits);

// Partition integration helper: a split candidate's cost is the sum of the
// best costs of its children plus the bits used to signal the split.
CandidateCost make_partition_split_candidate(
    int partition_id,
    std::span<const CandidateCost> children,
    double signaling_rate_bits);

// ---------------------------------------------------------------------------
// RDOQ: limited coefficient-level trellis
// ---------------------------------------------------------------------------

// The trellis optimises one transform block in scan order. State is the
// previous coded coefficient magnitude clamped to {0, 1, 2} (two or more).
// For each coefficient it tries zero, the scalar-quantised level, and levels
// within max_level_delta. It also considers terminating the block (EOB) at
// position 0 or after a non-zero coefficient, so trailing zeros are not coded.
//
// This is a documented limited path, not a full AV2 coefficient-context model:
//   - the real context derivation belongs in the caller's rate hook;
//   - level candidates are local (initial +/- max_level_delta), not an
//     unbounded trellis;
//   - the default rate model is a placeholder and must be replaced by an
//     entropy-cost model for production decisions.
//
// The DP is O(N * 3 * L) with L <= 2 + 2*max_level_delta, so it is cheap enough
// for the CPU reference path.

struct RdoqConfig {
  double lambda = 0.0;        // SSE-domain lambda; must be finite and >= 0
  double quant_step = 1.0;    // dequant step; must be finite and > 0
  std::int32_t max_level_delta = 1;  // in [0, 1024]: levels above/below the initial one
  bool allow_level_decrease = true;  // try levels 1 .. |q|-1
  bool allow_level_increase = true;  // try |q|+1 .. |q|+delta for |q| > 0
  bool allow_zero_to_one = false;    // let a zero initial level become +/-1
  std::int32_t max_coefficients = 4096;  // safety cap for the reference path
};

bool rdoq_config_valid(const RdoqConfig& config) noexcept;
bool validate_rdoq_config(const RdoqConfig& config,
                          std::string* error = nullptr);

struct RdoqCoefficient {
  double original = 0.0;           // forward transform coefficient
  std::int32_t initial_level = 0;  // signed scalar-quantised level
  std::int32_t context = 0;        // opaque caller context for the rate hook
};

// Estimated bits for coding one coefficient level. `prev_abs_level` is the
// previous magnitude clamped to {0, 1, 2}; `scan_pos` is the scan index;
// `level_context` is RdoqCoefficient::context. The callback must return a
// finite, non-negative number of bits. A null callback uses the default model.
using RdoqRateFn = double (*)(void* context,
                              std::int32_t scan_pos,
                              std::int32_t prev_abs_level,
                              std::int32_t abs_level,
                              std::int32_t level_context);

// Estimated bits to signal an end-of-block at `eob`. `eob == num_coefficients`
// means the last coefficient is non-zero (usually implicit and free).
// A null callback uses the default model.
using RdoqEobRateFn = double (*)(void* context,
                                 std::int32_t eob,
                                 std::int32_t num_coefficients);

struct RdoqRateModel {
  void* context = nullptr;
  RdoqRateFn coefficient_rate = nullptr;
  RdoqEobRateFn eob_rate = nullptr;
};

// Placeholder rate model for tests and bring-up. Replace it with an entropy
// cost estimate for real decisions.
double default_rdoq_coefficient_rate(void* context,
                                     std::int32_t scan_pos,
                                     std::int32_t prev_abs_level,
                                     std::int32_t abs_level,
                                     std::int32_t level_context) noexcept;
double default_rdoq_eob_rate(void* context,
                             std::int32_t eob,
                             std::int32_t num_coefficients) noexcept;

RdoqRateModel default_rdoq_rate_model() noexcept;

struct RdoqResult {
  // Signed levels in scan order. Positions >= eob are zero.
  std::vector<std::int32_t> levels;
  std::int32_t eob = 0;  // number of coded coefficients; 0 means all zero
  double distortion = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;  // distortion + lambda * rate_bits
};

// Runs the trellis. Returns false and fills `error` on an invalid config, an
// invalid rate from the hook, or a block larger than max_coefficients. The
// result is reset before any work, so a failed call leaves empty levels.
bool rdoq_quantize(std::span<const RdoqCoefficient> coefficients,
                   const RdoqConfig& config,
                   const RdoqRateModel& rate_model,
                   RdoqResult& result,
                   std::string* error = nullptr);

}  // namespace av2enc::rdo
