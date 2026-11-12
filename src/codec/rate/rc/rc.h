#pragma once

// Frame-level rate control for av2enc: CQP, CRF and VBR (CPU policy).
//
// This component owns only per-frame quantizer policy and the data structures
// that describe segmentation and delta-Q plans. It does not write bits, touch
// the entropy coder, or modify any reference stream. The integration pass owns
// wiring these decisions into the frame header and block syntax.
//
// Normative references (AV2 Bitstream & Decoding Process Specification v1.0.0):
//   - §5.18.6.1  base_q_idx
//   - §5.18.7.1  segmentation_params()
//   - §5.18.7.8  delta_q_params()
//   - §5.20.5.11 read_delta_qindex()
//   - §5.20.5.12 segmentation feature active functions
//   - §6.17.7.1  segmentation semantics
//   - §7.14.2    get_qindex() / qlookup()
//   - §6.17.2    coefficient CDF quantizer-context slice (see docs/open-questions B1)
//
// Units and conventions:
//   - base_q_idx / qindex: integer 0..255 for 8-bit streams. 0 is lossless and
//     is outside the rate-control domain; RC decisions clamp to 1..255.
//   - qp: a continuous real-valued quantizer level used internally by CRF/VBR.
//     It is rounded to base_q_idx only at the decision boundary.
//   - rate: bits per second for targets, bits per frame for estimates.
//   - distortion/lambda: squared-error units per bit; lambda uses the same
//     calibration placeholder as src/encoder/rdo/ but with the normative step.
//
// The bitrate model is a calibration placeholder, not an AV2 normative model.
// Do not cite its estimates as measured quality or as a BD-rate result.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::rc {

// ---------------------------------------------------------------------------
// Quantizer index domain, slices and coefficient-CDF contexts
// ---------------------------------------------------------------------------

inline constexpr int kMinQIndex = 1;       // lowest non-lossless qindex
inline constexpr int kMaxQIndex = 255;     // 8-bit max_q
inline constexpr int kLosslessQIndex = 0;  // lossless special case

// The 8-bit qindex is clamped into [0, 255]. Use clamp_base_qindex() for a
// frame-level base_q_idx, which must be >= 1.
int clamp_qindex(int qindex) noexcept;
int clamp_base_qindex(int qindex) noexcept;

// Normative AC quantizer step for qindex (AV2 §7.14.2, qlookup).
// qindex is clamped to [0, 255]. qindex 0 returns the lossless special-case
// value 64 and is not a usable rate-control quantizer.
double quant_step_from_qindex(int qindex) noexcept;

// qp -> qindex with deterministic round-half-up, clamped to [0, 255].
// NaN maps to 0. The result may be lossless; callers that need a coded frame
// must apply clamp_base_qindex().
int qindex_from_qp(double qp) noexcept;

// SSE-domain lambda = sse_scale * quant_step^2. qindex <= 0 (lossless) returns
// 0: there is no rate/distortion tradeoff to make. sse_scale defaults to the
// RDO component's placeholder (0.125); calibrate it before quality decisions.
double lambda_sse_from_qindex(int qindex, double sse_scale = 0.125) noexcept;

// One coefficient-CDF quantizer-context slice. The AV2 coefficient CDFs are
// initialized once per frame from base_q_idx, not from the per-block effective
// qindex, so delta-Q and segmentation change the quantizer but NOT this slice.
struct QpSlice {
  int index = 0;
  int min_qindex = 0;
  int max_qindex = 0;
};

// §6.17.2 init_coeff_cdfs() slice selection (docs/open-questions B1):
//   idx = base_q_idx <= 90  ? 0 :
//         base_q_idx <= 140 ? 1 :
//         base_q_idx <= 190 ? 2 : 3
inline constexpr std::array<QpSlice, 4> kCoeffCdfQpSlices{{
    {0, 0, 90},
    {1, 91, 140},
    {2, 141, 190},
    {3, 191, 255},
}};

// Slice index (coefficient-CDF qctx) for a frame base_q_idx. The input is
// clamped to [0, 255].
int coeff_cdf_qctx(int base_q_idx) noexcept;
const QpSlice& coeff_cdf_slice(int base_q_idx) noexcept;

// ---------------------------------------------------------------------------
// Frame complexity and bitrate estimation hooks
// ---------------------------------------------------------------------------

// Complexity input for one frame. `residual_mad` and `residual_mse` are in
// 8-bit sample units and describe the prediction residual (or any equivalent
// spatial-activity measure). `intra_fraction` is informational and is recorded
// for logging/model calibration; the default estimator does not use it.
struct FrameComplexity {
  int width = 0;
  int height = 0;
  double residual_mad = 0.0;    // mean absolute residual, 0..255
  double residual_mse = 0.0;    // mean squared residual, 0..65025
  double intra_fraction = 0.0;  // 0..1

  std::int64_t pixel_count() const noexcept {
    return static_cast<std::int64_t>(width) * static_cast<std::int64_t>(height);
  }
};

bool frame_complexity_valid(const FrameComplexity& complexity,
                            std::string* error = nullptr);

// Residual sigma used by the model and by CRF complexity adaptation:
// sqrt(residual_mse) when mse > 0, otherwise mad_to_sigma * residual_mad.
// Returns 0 for an invalid complexity or a non-positive mad_to_sigma.
double frame_complexity_sigma(const FrameComplexity& complexity,
                              double mad_to_sigma = 1.2533) noexcept;

// Computes MAD/MSE of (original - predicted) over the visible width x height
// region. Samples outside the visible region (stride padding) are ignored.
// Returns a default (invalid) FrameComplexity if the spans are too small or the
// dimensions are invalid. This is a deterministic integer accumulation, not a
// normative distortion metric.
FrameComplexity complexity_from_u8_planes(std::span<const std::uint8_t> original,
                                          std::span<const std::uint8_t> predicted,
                                          int width,
                                          int height,
                                          int stride) noexcept;

// Placeholder bitrate model parameters. The model is
//   sigma = sqrt(mse) (or mad_to_sigma * mad when mse == 0)
//   bits  = header_bits
//         + pixels * min(max_bits_per_pixel,
//                        coefficient_bits_per_sample * log2(1 + sigma / step))
// where step = quant_step_from_qindex(base_q_idx). It is monotone
// non-increasing in qindex by construction. Calibrate the three scaling terms
// against a real encode before using it for quality decisions.
struct BitrateModelParams {
  double header_bits = 160.0;
  double coefficient_bits_per_sample = 1.5;
  double max_bits_per_pixel = 16.0;
  double mad_to_sigma = 1.2533;
};

bool bitrate_model_params_valid(const BitrateModelParams& params,
                                std::string* error = nullptr);

// Direct model evaluation (caller must pass valid inputs). base_q_idx is
// clamped to [1, 255].
double estimate_frame_bits_model(const FrameComplexity& complexity,
                                 int base_q_idx,
                                 const BitrateModelParams& params = {}) noexcept;

// Optional external estimator. The callback must not throw and must return a
// finite, non-negative bit count. A null callback selects the built-in model.
// This is the integration hook for a calibrated or lookahead-driven estimator.
using BitrateEstimateFn = double (*)(void* context,
                                     const FrameComplexity& complexity,
                                     int base_q_idx);

struct BitrateEstimator {
  BitrateModelParams params{};
  void* context = nullptr;
  BitrateEstimateFn estimate = nullptr;
};

bool bitrate_estimator_valid(const BitrateEstimator& estimator,
                             std::string* error = nullptr);

// Validates the inputs, dispatches to the hook or built-in model, and writes
// the bit estimate. Returns false and fills `error` on invalid input or an
// invalid (NaN/Inf/negative) hook result.
bool estimate_frame_bits(const BitrateEstimator& estimator,
                         const FrameComplexity& complexity,
                         int base_q_idx,
                         double& bits,
                         std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Segmentation and delta-Q data structures (no bitstream syntax)
// ---------------------------------------------------------------------------

inline constexpr int kMaxSegments = 16;   // MAX_SEGMENTS
inline constexpr int kSegLvlMax = 3;      // SEG_LVL_MAX
inline constexpr int kSegLvlAltQ = 0;     // SEG_LVL_ALT_Q
inline constexpr int kSegLvlSkip = 1;     // SEG_LVL_SKIP
inline constexpr int kSegLvlGlobalMv = 2; // SEG_LVL_GLOBALMV

// Segmentation_Feature_Bits / Signed / Max: ALT_Q is signed 9-bit, clipped to
// +/-MAXQ_BITS (351 in the v1.0.0 constants). The other two are zero-bit flags.
inline constexpr int kSegAltQLimit = 351;

struct SegmentFeatures {
  std::array<bool, kSegLvlMax> enabled{};
  std::array<std::int32_t, kSegLvlMax> data{};
};

struct SegmentationParams {
  bool enabled = false;
  std::array<SegmentFeatures, kMaxSegments> segments{};
  bool update_map = false;
  bool temporal_update = false;  // requires update_map
};

SegmentationParams disabled_segmentation() noexcept;
bool segmentation_params_valid(const SegmentationParams& params,
                               std::string* error = nullptr);

// Enables segmentation and sets SEG_LVL_ALT_Q for `segment_id`. delta_q is a
// quantizer index delta (base + delta, or CurrentQIndex + delta when delta-Q is
// present), not a bitstream-scaled value.
bool set_segment_alt_q(SegmentationParams& params,
                       int segment_id,
                       int delta_q,
                       std::string* error = nullptr);

struct DeltaQParams {
  bool present = false;
  int res = 0;  // delta_q_res, [0, 3]; must be 0 when !present
};

bool delta_q_params_valid(const DeltaQParams& params,
                          std::string* error = nullptr);

// Mirrors get_qindex(ignoreDeltaQ, segmentId) (§7.14.2). `current_q_index` is
// the decoder's CurrentQIndex for delta-Q, reset to base_q_idx at each tile.
// `ignore_delta_q` is the decoder's ignoreDeltaQ argument (1 for some filter
// paths; pass false for the normal block quantizer).
//
// The result is clamped to [0, 255]. Returns false for an invalid context or
// segment id.
struct QpContext {
  int base_q_idx = 90;
  DeltaQParams delta_q{};
  int current_q_index = 90;
  SegmentationParams segmentation{};
};

bool qp_context_valid(const QpContext& context, std::string* error = nullptr);

bool effective_qindex(const QpContext& context,
                      int segment_id,
                      bool ignore_delta_q,
                      int& qindex,
                      std::string* error = nullptr);

// Incremental delta-Q state used by a planner or a real block encoder.
struct DeltaQState {
  int current_q_index = 0;  // 0 means uninitialized
};

// Sets CurrentQIndex = base_q_idx at the start of a tile (§5.20.2.1). Requires
// base_q_idx in [1, 255] because delta_q_present is only read when base > 0.
bool delta_q_reset(int base_q_idx,
                   DeltaQState& state,
                   std::string* error = nullptr);

// Applies one coded reduced delta: current = Clip3(1, 255,
// current + (reduced_delta << res)). Requires params.present and an initialized
// state. This is the state update performed by read_delta_qindex().
bool delta_q_apply(const DeltaQParams& params,
                   int reduced_delta,
                   DeltaQState& state,
                   std::string* error = nullptr);

// Raster-order delta-Q plan for one tile. The AV2 decoder resets CurrentQIndex
// to base_q_idx at every tile, so a plan is per tile; callers must not carry
// state across tile starts. The plan records the coded reduced deltas that
// drive the cumulative qindex toward `target_qindex`, one entry per superblock
// in raster order. It assumes the delta is coded at the superblock root (i.e.
// the superblock is not a fully skipped MiSize == SbSize block).
struct DeltaQPlanConfig {
  int base_q_idx = 90;
  int delta_q_res = 0;  // [0, 3]
  int cols = 0;         // superblock columns in this tile
  int rows = 0;         // superblock rows in this tile
};

struct DeltaQPlanStep {
  int sb_x = 0;
  int sb_y = 0;
  int reduced_delta = 0;  // coded value before << delta_q_res
  int qindex_before = 0;
  int qindex_after = 0;   // decoder state after clipping
};

struct DeltaQPlan {
  std::vector<DeltaQPlanStep> steps;
  int max_abs_reduced_delta = 0;
  int final_qindex = 0;

  bool empty() const noexcept { return steps.empty(); }
  void clear() noexcept;
};

bool delta_q_plan_config_valid(const DeltaQPlanConfig& config,
                               std::string* error = nullptr);

// Builds the plan. `target_qindex` is row-major, cols * rows entries, and must
// be in [1, 255]; target 0 (lossless) is not reachable through delta-Q and must
// be expressed with a segmentation ALT_Q value instead. Each step chooses the
// nearest reachable qindex given delta_q_res; achieved values may differ from
// the target by up to 2^res - 1 when clipping does not intervene.
bool plan_delta_q(const DeltaQPlanConfig& config,
                  std::span<const int> target_qindex,
                  DeltaQPlan& plan,
                  std::string* error = nullptr);

// Segmentation map on the 4x4 luma MI grid, matching the decoder's MI arrays.
// `segment_id` is row-major, mi_cols * mi_rows entries, each < kMaxSegments.
struct SegmentationMap {
  int mi_cols = 0;
  int mi_rows = 0;
  std::vector<std::uint8_t> segment_id;
};

bool segmentation_map_valid(const SegmentationMap& map,
                            std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Per-frame rate control
// ---------------------------------------------------------------------------

enum class RateControlMode : std::uint8_t {
  kCqp = 0,  // constant base_q_idx
  kCrf = 1,  // constant quality factor, variable rate
  kVbr = 2,  // target bitrate with a virtual-buffer correction
};

struct CqpParams {
  int base_q_idx = 90;
};

struct CrfParams {
  // crf in [0, 63]; lower is better quality. The mapping to qindex is a linear
  // calibration placeholder with explicit endpoints, not a standard CRF curve.
  double crf = 32.0;
  double qindex_at_crf0 = 1.0;
  double qindex_at_crf63 = 255.0;

  // Optional per-frame complexity adaptation (`qcomp`). 0 disables it, which
  // is the default and yields constant qindex across frame complexity. When
  // non-zero, complexity_reference must be > 0 and is in residual-sigma units.
  double qcomp = 0.0;
  double complexity_reference = 0.0;
  double max_qindex_adjust = 12.0;
};

struct VbrParams {
  double target_bitrate_bps = 500000.0;
  std::int32_t frame_rate_num = 30;
  std::int32_t frame_rate_den = 1;

  int min_qindex = kMinQIndex;
  int max_qindex = kMaxQIndex;
  int max_qindex_step = kMaxQIndex;  // per-frame slew limit; >= 1

  double buffer_frames = 30.0;  // virtual buffer capacity in frames of target
  double buffer_gain = 0.5;     // proportional rate-error correction
  double bits_ema_weight = 0.2; // 0..1, 1 = ignore history
};

struct RateControlConfig {
  RateControlMode mode = RateControlMode::kCqp;
  CqpParams cqp{};
  CrfParams crf{};
  VbrParams vbr{};
  BitrateEstimator estimator{};
  double lambda_sse_scale = 0.125;  // calibration placeholder, see rdo.h
};

bool rate_control_config_valid(const RateControlConfig& config,
                               std::string* error = nullptr);

struct QpDecision {
  int base_q_idx = 90;
  int coeff_cdf_qctx = 0;  // derived from base_q_idx, NOT from delta-Q qindex
  int qp_slice = 0;        // same index as coeff_cdf_qctx
  double continuous_qp = 90.0;
  double estimated_bits = 0.0;
  double lambda_sse = 0.0;
  bool target_reachable = true;  // VBR: estimate <= adjusted target
  bool clamped_by_step = false;  // VBR slew limit changed the choice
};

// Stateful per-frame controller. One instance per encode; do not call
// begin_frame() twice without end_frame().
//
// VBR virtual buffer: buffer_bits accumulates (target_bits - actual_bits) and
// is clamped to +/- buffer_frames * target_bits. The next frame targets
// target_bits + buffer_gain * buffer_bits. This is a deterministic proportional
// correction, NOT an HRD/VBV model; VBV is deferred.
class RateController {
 public:
  RateController() = default;

  bool configure(const RateControlConfig& config, std::string* error = nullptr);

  const RateControlConfig& config() const noexcept { return config_; }

  bool begin_frame(const FrameComplexity& complexity,
                   QpDecision& decision,
                   std::string* error = nullptr);

  // Reports the actual encoded payload bits (not bytes, not OBU overhead) for
  // the frame opened by begin_frame(). Updates the VBR virtual buffer.
  bool end_frame(std::int64_t actual_bits, std::string* error = nullptr);

  void reset() noexcept;

  bool frame_open() const noexcept { return frame_open_; }
  int previous_qindex() const noexcept { return previous_qindex_; }
  double bits_ema() const noexcept { return bits_ema_; }
  std::int64_t frame_count() const noexcept { return frame_count_; }
  double buffer_bits() const noexcept { return buffer_bits_; }

 private:
  RateControlConfig config_{};
  bool configured_ = false;
  bool frame_open_ = false;
  int previous_qindex_ = 0;
  double bits_ema_ = 0.0;
  double buffer_bits_ = 0.0;
  std::int64_t frame_count_ = 0;
  QpDecision last_decision_{};
};

}  // namespace av2enc::rc
