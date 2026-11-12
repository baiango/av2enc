#pragma once

// Two-pass, lookahead and VBV/HRD-aware rate-control policy for av2enc.
//
// This is a *policy layer* on top of src/encoder/rc/. It produces decisions
// only: no bits, no OBU/frame-header writing, no entropy coding. The second
// pass consumes the plan with the existing delta-Q / segmentation syntax
// module; the bitstream is unchanged.
//
// Normative references (AV2 Bitstream & Decoding Process Specification v1.0.0):
//   - §5.18.6.1  base_q_idx
//   - §5.18.7.1  segmentation_params()
//   - §5.18.7.8  delta_q_params()
//   - §5.20.1    CurrentQIndex = base_q_idx at each tile
//   - §5.20.2.1  ReadDeltas at each superblock
//   - §5.20.5.11 read_delta_qindex()
//   - §6.17.2    coefficient CDF slice from raw base_q_idx
//   - §7.14.2    get_qindex() / qlookup()
//
// What this layer adds over rc::RateController
// --------------------------------------------
//   * first-pass complexity analysis and a two-pass bit allocation;
//   * a bounded lookahead window for online re-planning;
//   * a leaky-bucket VBV model with underflow prevention and deterministic
//     reallocation of the remaining window;
//   * CRF and VBR target tracking with per-frame and per-tile QP assignment;
//   * per-superblock QP targets expressed through delta-Q and/or segmentation,
//     with correct per-tile CurrentQIndex resets.
//
// What it is not
// --------------
//   * It is not an HRD conformance implementation. The buffer model is an
//     internal encoder-side leaky bucket. It prevents modelled underflow; it
//     does not claim decoder HRD conformance.
//   * It does not add CPB/DPB timing syntax, padding frames, or dropped frames.
//     When a target cannot be met, it clamps, flags it, and reports it.
//   * The bitrate model is the rc:: placeholder unless the caller installs a
//     calibrated estimator. Tests must distinguish model results from measured
//     results.
//
// Determinism
// -----------
// Every decision is a pure function of the configuration, the first-pass
// statistics vector, and the sequence of actual-bit reports. There are no
// timers, RNGs, threads or hidden state. The bitrate estimator hook must be
// pure for a given (context, complexity, qindex).
//
// Units
// -----
//   * bits: bits per frame for estimates and allocations, bits per second for
//     targets. "Actual bits" means the coded frame payload, not container or
//     filesystem bytes.
//   * qindex/base_q_idx: integer 1..255. 0 is lossless and is outside this
//     controller's domain.
//   * qp: a continuous internal level; only qindex is ever emitted.

#include "../rc.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace av2enc::rc::twopass {

// ---------------------------------------------------------------------------
// VBV / leaky-bucket model
// ---------------------------------------------------------------------------

// The buffer starts at `initial_fill * size`. At each frame interval it is
// refilled by `drain` bits (the channel rate) and drained by the frame's coded
// bits. Underflow means the buffer would go below `underflow_margin`;
// overflow means it would rise above `size - overflow_margin`.
//
// Underflow is preventable by spending fewer bits (raising qindex). Overflow
// cannot be fixed without padding or dropping frames, neither of which changes
// the bitstream here, so it is reported but not "fixed".
struct VbvConfig {
  bool enabled = true;
  // Buffer size. <= 0 selects `buffer_frames * average_frame_bits`.
  double buffer_size_bits = 0.0;
  double buffer_frames = 2.0;
  // Initial occupancy as a fraction of the buffer size, [0, 1].
  double initial_fill = 0.7;
  // Channel rate in bits per second used to refill the bucket. <= 0 selects
  // the average target bitrate (rc::VbrParams::target_bitrate_bps).
  double max_rate_bps = 0.0;
  // Safety margins as fractions of the buffer size, [0, 0.5).
  double underflow_margin = 0.02;
  double overflow_margin = 0.02;
};

struct VbvState {
  bool enabled = false;
  double size_bits = 0.0;
  double drain_bits_per_frame = 0.0;
  double underflow_margin_bits = 0.0;
  double overflow_margin_bits = 0.0;
  double fill_bits = 0.0;
  // Diagnostics accumulated by vbv_apply()/the controller.
  std::int64_t underflow_events = 0;
  std::int64_t overflow_events = 0;
};

// Validates and initializes the bucket. `average_frame_bits` is the nominal
// per-frame budget (target_bitrate * frame_duration). `frame_duration` is in
// seconds and must be > 0. A disabled config produces a disabled state.
bool make_vbv_state(const VbvConfig& config,
                    double average_frame_bits,
                    double frame_duration,
                    VbvState& state,
                    std::string* error = nullptr);

// Largest frame size that does not underflow at the current fill, and the
// smallest size that does not overflow. Both are unclamped diagnostics; a
// negative max means the bucket is already below the underflow margin.
double vbv_max_bits(const VbvState& state) noexcept;
double vbv_min_bits(const VbvState& state) noexcept;

// Applies one coded frame of `bits` bits. Updates fill_bits and the event
// counters. `underflow`/`overflow` report this frame's result. The fill is
// clamped into [underflow_margin, size - overflow_margin] after an event so
// subsequent planning continues from a defined state. Returns false only for
// invalid inputs (negative/NaN bits); a disabled state is a no-op.
bool vbv_apply(VbvState& state,
               double bits,
               bool& underflow,
               bool& overflow) noexcept;

// Nominal frame duration in seconds from rc::VbrParams. Returns 0 for an
// invalid rate.
double frame_duration_seconds(const VbrParams& params) noexcept;

// Nominal bits per frame from rc::VbrParams. Returns 0 for an invalid rate.
double nominal_frame_bits(const VbrParams& params) noexcept;

// ---------------------------------------------------------------------------
// First-pass input
// ---------------------------------------------------------------------------

// Per-tile first-pass statistics. `sb_cols` x `sb_rows` is the tile's
// superblock extent and must be positive. `sb_weights` is optional,
// row-major and either empty (uniform) or exactly sb_cols * sb_rows entries.
// Weights are non-negative activity/energy values; only ratios matter.
//
// `sb_col_start`/`sb_row_start` place the tile in the frame's superblock grid
// so the controller can build a frame-wide SegmentationMap. A single-tile
// input may leave them at 0.
struct TileStats {
  int sb_cols = 1;
  int sb_rows = 1;
  int sb_col_start = 0;
  int sb_row_start = 0;
  FrameComplexity complexity{};
  std::vector<double> sb_weights;
};

// First-pass statistics for one frame. `tiles` empty means one tile covering
// the whole frame (the superblock extent is derived from width/height at
// `sb_size`). `first_pass_bits`, when > 0, is a measured first-pass payload
// size and takes precedence over `complexity` for relative bit allocation; it
// must not be used for the second-pass estimate.
//
// `sb_size` is the luma superblock size in pixels used by the tile geometry
// and the segmentation-map expansion (M1/M2 use 64). 64, 128 and 256 are
// accepted. first_pass_luma() always uses 64.
struct FrameStats {
  FrameComplexity complexity{};
  bool keyframe = false;
  double first_pass_bits = 0.0;
  int sb_size = 64;
  std::vector<TileStats> tiles;
};

// Validates a first-pass frame record after filling in defaults for an empty
// tile list. Width/height must be positive (superblock geometry is derived
// from them) and statistics must be finite. On success, `normalized` has the
// same content with the implicit single tile materialized.
bool normalize_frame_stats(const FrameStats& input,
                           FrameStats& normalized,
                           std::string* error = nullptr);

// Convenience first pass: block-DC intra prediction over one 8-bit luma plane.
//
// Each 16x16 block is predicted by its integer rounded mean; the residual
// moments are measured against that prediction. The plane is split into
// `tile_cols` x `tile_rows` superblock tiles (proportional split of the SB
// grid), and per-SB residual energy becomes `sb_weights`. This is a cheap,
// deterministic spatial-activity estimate, not a motion-compensated residual
// and not a normative distortion metric. `luma` may be padded (`stride >=
// width`); samples outside width x height are ignored.
bool first_pass_luma(std::span<const std::uint8_t> luma,
                     int width,
                     int height,
                     int stride,
                     int tile_cols,
                     int tile_rows,
                     bool keyframe,
                     FrameStats& stats,
                     std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct TwoPassConfig {
  // Frame-level mode, estimator and CRF/VBR parameters. kCqp, kCrf and kVbr
  // are accepted. `rc.vbr.frame_rate_*` is the frame clock for all modes and
  // `rc.vbr.target_bitrate_bps` is the nominal VBV rate/allocation budget
  // (in CRF and CQP modes it does not change the quality mapping except
  // through underflow prevention).
  RateControlConfig rc{};

  // Frames of future data the online correction may re-plan after a rate
  // error. 0 means the whole remaining window. In streaming mode the window
  // supplied to set_lookahead() is the horizon; this value caps it.
  int lookahead_frames = 0;

  // Bit-allocation weight = pixels * max(sigma, 1e-6)^complexity_exponent,
  // or first_pass_bits when a frame supplies it. [0, 2].
  double complexity_exponent = 0.5;
  // Extra allocation weight for keyframes, >= 0. 1 = no boost.
  double keyframe_weight = 1.0;

  // Per-frame allocation clamps relative to the window average. Min is
  // clamped to (0, 1]; max to [1, 100].
  double min_frame_bits_ratio = 0.1;
  double max_frame_bits_ratio = 8.0;

  // Per-frame qindex range and slew limit. A hard underflow override may
  // exceed the slew limit but never the qindex range.
  int min_qindex = kMinQIndex;
  int max_qindex = kMaxQIndex;
  int max_qindex_step = kMaxQIndex;

  // Maximum per-tile/per-SB qindex offset from base_q_idx, in qindex units.
  // 0 disables spatial QP variation. Requires delta-Q or segmentation to be
  // representable; otherwise only base_q_idx is effective.
  int spatial_qp_strength = 0;

  // Per-superblock delta-Q plan per tile. `delta_q_res` is the normative
  // 1 << res granularity, [0, 3]. Each tile is planned independently because
  // the decoder resets CurrentQIndex to base_q_idx at every tile.
  bool enable_delta_q = false;
  int delta_q_res = 0;

  // Per-superblock QP classes via SEG_LVL_ALT_Q. `segment_levels` is per
  // side of zero: offsets are -step*1..-step*L and +step*1..+step*L, so the
  // segment count is 2*L + 1. With enable_ext_seg false at most 7 classes are
  // available (8 segments); with it at most 15 (16 segments).
  bool enable_segmentation = false;
  int segment_levels = 1;
  int segment_qindex_step = 4;
  bool enable_ext_seg = false;

  // Internal leaky-bucket model. Underflow prevention is active whenever
  // vbv.enabled is true.
  VbvConfig vbv{};

  // Rate-feedback EMA: estimator scale = actual_bits / estimated_bits. 0
  // disables it. The ratio is clamped to [1/clamp, clamp].
  double rate_feedback_weight = 0.25;
  double rate_feedback_clamp = 4.0;

  // Redistribute the remaining window budget after an actual-bit error
  // (two-pass closed loop). When false only the QP feedback loop responds.
  bool redistribute_bits = true;
};

bool two_pass_config_valid(const TwoPassConfig& config,
                           std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Allocation (first pass result)
// ---------------------------------------------------------------------------

struct FrameAllocation {
  double weight = 0.0;
  double target_bits = 0.0;
  double min_bits = 0.0;
  double max_bits = 0.0;
  double vbv_fill_after = 0.0;
  bool vbv_capped = false;
  bool overflow_risk = false;
};

struct AllocationResult {
  std::vector<FrameAllocation> frames;
  double total_bits = 0.0;
  double allocated_bits = 0.0;
  // Budget that could not be placed without underflowing the bucket.
  double dropped_bits = 0.0;
  // Number of frames whose allocation changed in the VBV pass.
  int reallocations = 0;
  bool underflow_prevented = false;
  // False when even the per-frame minimum allocation underflows. The
  // controller then clamps and flags the frame; it never drops or pads.
  bool underflow_feasible = true;
  bool overflow_feasible = true;
};

// Deterministic water-filling allocation over `stats`, followed by a forward
// VBV pass that caps underflowing frames and redistributes the unused budget
// to later frames. `total_bits` is the window budget; `initial_buffer` is the
// bucket state before the first frame and is not modified.
bool allocate_frames(std::span<const FrameStats> stats,
                     const TwoPassConfig& config,
                     double total_bits,
                     const VbvState& initial_buffer,
                     AllocationResult& result,
                     std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Second-pass decisions
// ---------------------------------------------------------------------------

struct TilePlan {
  int sb_cols = 0;
  int sb_rows = 0;
  int sb_col_start = 0;
  int sb_row_start = 0;
  // Representative tile target (pixel-weighted mean of sb_target_qindex).
  int target_qindex = 0;
  // Desired effective qindex for each superblock, row-major,
  // sb_cols * sb_rows entries, in [1, 255].
  std::vector<int> sb_target_qindex;
  // Achieved effective qindex after delta-Q/segmentation clipping. This is
  // what the decoder's get_qindex() yields given the plan below.
  std::vector<int> sb_achieved_qindex;
  // SEG_LVL_ALT_Q class per superblock, row-major. All zero when
  // segmentation is disabled.
  std::vector<std::uint8_t> sb_segment_id;
  // Per-tile delta-Q plan. CurrentQIndex starts at base_q_idx for every tile.
  DeltaQPlan delta_q;
};

struct FramePlan {
  int frame_index = 0;
  bool keyframe = false;

  // Frame header decision.
  int base_q_idx = 90;
  int coeff_cdf_qctx = 0;  // always derived from base_q_idx
  double continuous_qp = 90.0;
  double lambda_sse = 0.0;

  // Rate decisions.
  double allocation_bits = 0.0;  // first-pass allocation
  double target_bits = 0.0;      // effective budget after the VBV cap
  double estimated_bits = 0.0;   // model estimate at the chosen plan
  double rate_feedback = 1.0;    // estimator scale applied to checks

  // Spatial syntax plans. delta_q.present and segmentation.enabled are false
  // when the corresponding feature is disabled in the config.
  DeltaQParams delta_q{};
  SegmentationParams segmentation{};
  bool enable_ext_seg = false;
  // Frame-wide segmentation map on the 4x4 MI grid. Empty when segmentation
  // is disabled or the tile geometry does not cover the frame.
  SegmentationMap segmentation_map{};

  std::vector<TilePlan> tiles;

  // Diagnostics.
  double vbv_fill_before = 0.0;
  double vbv_fill_after_estimate = 0.0;
  bool vbv_underflow_prevented = false;  // the budget was lowered
  bool vbv_overflow_risk = false;        // modelled fill exceeds the margin
  bool target_reachable = true;          // estimate <= budget at max qindex
  bool clamped_by_slew = false;
  bool clamped_by_range = false;
  bool spatial_clamped = false;  // spatial offsets reduced to fit the budget
  bool crf_quality_target = false;
};

// Stateful second-pass controller. Usage (offline two-pass):
//
//   TwoPassController rc;
//   rc.configure(config, &err);
//   rc.analyze(first_pass_stats, &err);
//   for (;;) {
//     FramePlan plan;
//     rc.begin_frame(plan, &err);
//     // ... encode with plan.base_q_idx / plan.tiles ...
//     rc.end_frame(actual_payload_bits, &err);
//   }
//
// Streaming replaces analyze() with set_lookahead(window) whenever the
// visible window changes (including before the first frame). The window must
// start at the next uncoded frame.
class TwoPassController {
 public:
  TwoPassController();
  ~TwoPassController();
  TwoPassController(const TwoPassController&) = delete;
  TwoPassController& operator=(const TwoPassController&) = delete;

  bool configure(const TwoPassConfig& config, std::string* error = nullptr);
  const TwoPassConfig& config() const noexcept { return config_; }
  bool configured() const noexcept { return configured_; }

  // Clears all history but keeps the configuration.
  bool reset(std::string* error = nullptr);

  // Offline first pass. Computes the global allocation for the supplied clip
  // and stores a private copy of the statistics for the second pass.
  bool analyze(std::span<const FrameStats> stats, std::string* error = nullptr);

  // Streaming first pass. Replaces the lookahead window; window[0] is the next
  // frame to encode (index frame_count()). The allocation is recomputed from
  // the current VBV state. Must not be called while a frame is open.
  bool set_lookahead(std::span<const FrameStats> window,
                     std::string* error = nullptr);

  // Produces the plan for frame_count(). Requires pending first-pass stats.
  bool begin_frame(FramePlan& plan, std::string* error = nullptr);

  // Reports the actual coded payload bits for the frame opened by
  // begin_frame(). Updates the VBV fill, the rate-feedback EMA, the totals,
  // and (offline) deterministically re-plans the remaining lookahead window.
  bool end_frame(std::int64_t actual_bits, std::string* error = nullptr);

  std::size_t frame_count() const noexcept { return frame_count_; }
  bool frame_open() const noexcept { return frame_open_; }
  int previous_qindex() const noexcept { return previous_qindex_; }
  double rate_feedback() const noexcept { return rate_feedback_; }
  const VbvState& vbv() const noexcept { return vbv_; }
  const AllocationResult& allocation() const noexcept { return allocation_; }
  double total_target_bits() const noexcept { return total_target_bits_; }
  double total_actual_bits() const noexcept { return total_actual_bits_; }
  std::size_t lookahead_size() const noexcept { return pending_.size(); }

 private:
  bool validate_and_plan(std::span<const FrameStats> stats,
                         std::size_t horizon,
                         double total_bits,
                         std::string* error);
  bool reallocate_remaining(double error_bits, std::string* error);

  TwoPassConfig config_{};
  bool configured_ = false;
  bool offline_ = false;
  bool frame_open_ = false;

  std::vector<FrameStats> pending_;   // window, front = next frame
  AllocationResult allocation_{};

  VbvState vbv_{};
  double rate_feedback_ = 1.0;
  double total_target_bits_ = 0.0;
  double total_actual_bits_ = 0.0;
  int previous_qindex_ = 0;
  std::size_t frame_count_ = 0;

  FramePlan open_plan_{};
  double open_target_bits_ = 0.0;
  double open_estimate_bits_ = 0.0;

  // Existing RC component used for the CQP/CRF quality decision. Unused for
  // VBR because the two-pass target replaces the constant-rate target.
  struct QualityState;
  std::unique_ptr<QualityState> quality_;
};

}  // namespace av2enc::rc::twopass
