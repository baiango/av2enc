// Frame pipeline planner: rate control + tile layout + per-tile QP + GPU
// assignment, in one deterministic decision.
//
// Status: planning/integration component. It produces data, not bits: no OBU is
// written, no entropy coder is called, no CUDA allocation is performed. The M1
// writer in src/encoder/m1/ still emits a single tile with
// `segmentation_enabled = 0`, `delta_q_present = 0` and a pinned
// `base_q_idx = 90`; wiring those decisions into a writer is the integration
// step described in README.md.
//
// This module is intentionally a composition layer. It does not reimplement
// any of the three owners it consumes:
//   * src/encoder/rc/        per-frame CQP/CRF/VBR, QP slices, segmentation and
//                            delta-Q planning data structures (§5.18.6.1,
//                            §5.18.7.1, §5.18.7.8, §7.14.2).
//   * src/encoder/tiles/     uniform-spacing tile geometry (§5.18.7.2-7,
//                            §9.2) and tile-group framing rules (§5.19,
//                            §5.20.1).
//   * src/multi_gpu/         deterministic tile-to-GPU assignment, reference
//                            duplication and per-GPU VRAM estimates.
//
// Determinism contract: a FramePipelinePlan is a pure function of
// (PipelineConfig, FrameStats, the controller's actual-bits history). There is
// no clock, RNG, device query or host state in the decision path. The
// independent-bits history is explicit: `end_frame(actual_bits)`.
//
// Conventions:
//   * `base_q_idx` is the frame-level qindex in [1, 255] (0 is lossless and
//     outside rate control).
//   * A "tile QP" is the effective qindex a decoder derives for blocks inside
//     that tile. It is signalled either by per-tile delta-Q (one reduced delta
//     at the tile's first superblock, then zeros) or by segmentation ALT_Q.
//     The coefficient-CDF slice is always derived from the frame
//     `base_q_idx`, never from the tile qindex (src/encoder/rc/README.md,
//     docs/negative-results.md).
//   * Tile geometry is always taken from av2enc::tiles::plan_tile_layout().
//     Never recompute tile starts elsewhere: a mismatch is an invalid stream.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "codec/rate/rc/rc.h"
#include "pipeline/tiles/tile_layout.h"
#include "pipeline/dispatch/planner.h"

namespace av2enc::pipeline {

// ---------------------------------------------------------------------------
// Tile layout selection
// ---------------------------------------------------------------------------

enum class LayoutPolicyMode : std::uint8_t {
  // Pick a verified layout that fits the GPU count and the frame geometry.
  // The chosen preset and the reason are reported in FramePipelinePlan.
  kAuto = 0,
  // Use `explicit_preset`; fail if it is invalid or not decoder-verified.
  kExplicit = 1,
};

struct LayoutPolicy {
  LayoutPolicyMode mode = LayoutPolicyMode::kAuto;
  tiles::TileLayoutPreset explicit_preset = tiles::TileLayoutPreset::k4x2;

  // Which presets have actually decoded with the pinned dav2d build. The
  // project baseline verifies 1x1, 2x1, 2x2, 4x2 and 8x2; 16x2 is left off
  // until its evidence is reviewed, even though the local 1920x1080 gate run
  // passes (see src/encoder/pipeline/README.md). Update this table only after
  // running src/encoder/pipeline/gate_dav2d_tiled.sh on the layout.
  std::array<bool, 6> decoder_verified{true, true, true, true, true, false};

  // Explicit opt-in for a layout that is not in `decoder_verified`. It is
  // never set implicitly: the bitstream gate is a hard requirement.
  bool allow_unverified = false;
};

const char* layout_policy_mode_name(LayoutPolicyMode mode) noexcept;

// True when the preset may be selected under `policy`.
bool layout_preset_allowed(tiles::TileLayoutPreset preset,
                           const LayoutPolicy& policy) noexcept;

// Default preset for a GPU count before geometry validation:
//   1 GPU  -> 4x2
//   2 GPUs -> 8x2
//   3+ GPUs-> 16x2 (only if allowed) else 8x2
// This mirrors src/multi_gpu/README.md with the 16x2 decoder gate made
// explicit. kAuto may still fall back to a smaller layout if the frame is too
// small for this preset; the reason is reported.
tiles::TileLayoutPreset default_layout_for_gpus(int gpu_count,
                                                const LayoutPolicy& policy) noexcept;

const char* layout_preset_name(tiles::TileLayoutPreset preset) noexcept;
int preset_index(tiles::TileLayoutPreset preset) noexcept;

// ---------------------------------------------------------------------------
// Per-tile QP policy
// ---------------------------------------------------------------------------

enum class TileQpMode : std::uint8_t {
  // Every tile uses the frame base_q_idx. No delta-Q and no segmentation.
  kUniform = 0,
  // One delta at each tile's first superblock, then zero deltas; the decoder
  // resets CurrentQIndex to base_q_idx at every tile, so this is a per-tile
  // QP with no cross-tile state. Requires the writer to emit
  // delta_q_present/delta_q_res and read_delta_qindex() syntax.
  kDeltaQ = 1,
  // A frame-global segmentation map whose ALT_Q values are constant inside a
  // tile. Requires <= 16 distinct tile qindexes (MAX_SEGMENTS) and a writer
  // that codes the segment-id map. Mutually exclusive with kDeltaQ.
  kSegmentation = 2,
};

const char* tile_qp_mode_name(TileQpMode mode) noexcept;

struct TileQpConfig {
  TileQpMode mode = TileQpMode::kUniform;

  // Complexity-adaptive redistribution around base_q_idx. The tile delta is
  //
  //   delta = round(strength * 24 * log2(sigma_tile / sigma_frame))
  //
  // clamped to +/-max_abs_delta. Higher-activity tiles get a coarser qindex
  // (a larger number) so a fixed frame budget is spent where it is cheapest.
  // This is a deterministic policy model, not a measured BD-rate result. With
  // strength 0 or no per-tile statistics every tile stays at base_q_idx.
  double complexity_strength = 1.0;
  int max_abs_delta = 12;

  // Only used by kDeltaQ. delta_q_res in [0, 3]; the reachable qindex step is
  // 2^res, so a requested tile qindex can be off by up to 2^res - 1. The
  // achieved value is reported per tile.
  int delta_q_res = 0;
};

bool tile_qp_config_valid(const TileQpConfig& config,
                          std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Device-sized arena budget (per GPU)
// ---------------------------------------------------------------------------

// The project rule (AGENTS.md, docs/vram.md) is a device-derived arena:
//
//   capacity = min(device_total, device_free) - headroom
//
// optionally bounded by an explicit --vram-cap. There is no fixed preset. The
// same capacity is used as the per-GPU cap passed to the multi-GPU planner, so
// a plan is rejected when a worker's estimate does not fit the arena the
// runtime can actually allocate.
//
// `total_bytes` and `free_bytes` are both 0 when the caller cannot query the
// device (for example a pure planning test). In that case an explicit
// `cap_bytes` is required.
struct DeviceBudget {
  std::uint64_t total_bytes = 0;
  std::uint64_t free_bytes = 0;
  std::uint64_t headroom_bytes = 512ull * 1024ull * 1024ull;
  std::uint64_t cap_bytes = 0;  // optional user cap; 0 = device-derived only
};

// ---------------------------------------------------------------------------
// Input: sequence configuration + per-frame statistics
// ---------------------------------------------------------------------------

struct PipelineConfig {
  int width = 0;
  int height = 0;
  int sb_size = 64;
  int seq_level_idx = 7;  // M1 default: level 5.1
  int seq_tier = 0;

  LayoutPolicy layout{};
  TileQpConfig tile_qp{};
  rc::RateControlConfig rate_control{};
  multigpu::PlannerConfig gpu{};  // gpu_count, assignment, refs, lookahead, ...
  DeviceBudget device{};

  // The current M1 writer only supports base_q_idx = 90 (qctx 0 coefficient
  // CDFs) and pins it in encode_still_picture(). Set min = max = 90 for an
  // M1 frame. The default admits the full RC domain for M2/other writers.
  int min_base_q_idx = rc::kMinQIndex;
  int max_base_q_idx = rc::kMaxQIndex;

  // Reject a plan whose per-GPU live estimate exceeds the device arena.
  bool require_vram_fit = true;

  // gpu.vram_cap_mb is overwritten from `device`; the multi-GPU planner's
  // legacy 7,500 MB field is not the product policy.
};

bool pipeline_config_valid(const PipelineConfig& config,
                           std::string* error = nullptr);

// Per-tile statistics for tile-QP redistribution. `tile_index` may be -1 for
// every entry, in which case entries are in raster tile order; mixed or
// permuted lists are accepted as long as every index appears exactly once.
// An entry with width == 0 inherits the frame complexity (delta 0).
struct TileStats {
  int tile_index = -1;
  rc::FrameComplexity complexity;
};

struct FrameStats {
  std::uint64_t frame_index = 0;
  rc::FrameComplexity complexity;
  // Empty means "no per-tile data": every tile inherits the frame sigma.
  std::vector<TileStats> tiles;
};

bool frame_stats_valid(const FrameStats& stats, int tile_count,
                       std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Output: one frame plan
// ---------------------------------------------------------------------------

struct TilePlan {
  int index = 0;
  int row = 0;
  int col = 0;
  int gpu_index = -1;

  tiles::TileRect rect{};

  // `target_qindex` is the policy request. `qindex` is what the chosen
  // signalling mechanism can actually represent (identical for segmentation;
  // equal to target +/- (2^res - 1) for delta-Q). Both are in [1, 255].
  int target_qindex = 0;
  int qindex = 0;

  // kSegmentation: base_q_idx + alt_q = qindex.
  int segment_id = -1;
  int alt_q = 0;

  // kDeltaQ: the reduced delta coded at the tile's first superblock. The full
  // per-superblock plan is below; the decoder resets CurrentQIndex at every
  // tile, so plans are never carried across tiles.
  int delta_q_reduced = 0;
  rc::DeltaQPlan delta_plan{};
};

struct FramePipelinePlan {
  bool valid = false;
  std::string error;

  std::uint64_t frame_index = 0;

  // Frame-level rate-control decision. `base_q_idx`, `coeff_cdf_qctx`,
  // `qp_slice`, `lambda_sse` and `estimated_bits` all come from
  // rc::RateController::begin_frame().
  rc::QpDecision qp{};
  int base_q_idx = 0;
  TileQpMode tile_qp_mode = TileQpMode::kUniform;

  // Signalling decisions. Both are disabled in kUniform mode. They are never
  // both enabled (segmentation ALT_Q would stack on delta-Q). A configured
  // non-uniform mode can still produce disabled params when every tile lands
  // on base_q_idx; `tile_qp_mode` records the policy that was asked for.
  rc::DeltaQParams delta_q{};
  rc::SegmentationParams segmentation{};
  rc::SegmentationMap segment_map{};
  bool has_segment_map = false;

  tiles::TileLayout layout{};
  std::string layout_reason;

  std::vector<TilePlan> tiles;  // raster tile order, always every tile

  // Same assignment/VRAM/transfer model as src/multi_gpu/, computed once per
  // (geometry, GPU count, arena) and copied here.
  multigpu::MultiGpuPlan gpu{};

  double estimated_frame_bits = 0.0;  // rate-control model, not a measurement
};

// ---------------------------------------------------------------------------
// Stateful planner (real encode loop)
// ---------------------------------------------------------------------------

class PipelinePlanner {
 public:
  PipelinePlanner() = default;

  // Validates the config, computes the per-GPU arena, selects and validates
  // the tile layout, runs the multi-GPU planner once, and configures the rate
  // controller. Call before plan_frame().
  bool configure(const PipelineConfig& config, std::string* error = nullptr);

  // One rate-control decision + tile QP plan + GPU assignment. Opens a frame;
  // exactly one end_frame() must follow before the next plan_frame().
  bool plan_frame(const FrameStats& stats, FramePipelinePlan& plan,
                  std::string* error = nullptr);

  // Report the actual encoded payload bits (not bytes, not OBU overhead) for
  // the frame opened by plan_frame(). This advances the VBR virtual buffer.
  bool end_frame(std::int64_t actual_bits, std::string* error = nullptr);

  void reset() noexcept;

  bool configured() const noexcept { return configured_; }
  bool frame_open() const noexcept { return frame_open_; }
  const PipelineConfig& config() const noexcept { return config_; }
  const tiles::TileLayout& layout() const noexcept { return layout_; }
  std::uint64_t arena_capacity_bytes() const noexcept {
    return arena_capacity_bytes_;
  }
  int layout_preset_index() const noexcept { return layout_preset_index_; }

 private:
  PipelineConfig config_{};
  bool configured_ = false;
  bool frame_open_ = false;
  rc::RateController rc_{};
  tiles::TileLayout layout_{};
  std::string layout_reason_;
  int layout_preset_index_ = -1;
  std::uint64_t arena_capacity_bytes_ = 0;
  multigpu::MultiGpuPlan gpu_plan_{};
};

// ---------------------------------------------------------------------------
// Batch preview (planning pass / tests)
// ---------------------------------------------------------------------------

struct SequencePlan {
  bool valid = false;
  std::string error;
  PipelineConfig config{};
  std::uint64_t arena_capacity_bytes = 0;
  tiles::TileLayout layout{};
  std::vector<FramePipelinePlan> frames;
};

// Plans every frame in order. `actual_bits` must be empty or have exactly one
// entry per frame. When empty, the rate-control model estimate is fed back as
// a deterministic preview; that is a planning aid, not a substitute for
// end_frame(actual payload bits) in a real encode.
bool plan_sequence(const PipelineConfig& config,
                   std::span<const FrameStats> stats,
                   std::span<const std::int64_t> actual_bits,
                   SequencePlan& plan, std::string* error = nullptr);

// The batch deliverable name: one decision per frame plus the shared layout and
// arena. `SequencePlan` is the same type and remains the primary name.
using PipelinePlan = SequencePlan;

// ---------------------------------------------------------------------------
// Small deterministic helpers shared by the tests and the integration
// ---------------------------------------------------------------------------

// qindex -> tile delta for a given tile/frame sigma ratio. Exposed so a caller
// can reproduce the policy or calibrate it without duplicating the formula.
int tile_qindex_delta(double tile_sigma, double frame_sigma,
                      const TileQpConfig& config) noexcept;

// Effective qindex for a segment id, exactly as rc::effective_qindex() does
// but without needing the caller to build a QpContext.
int effective_tile_qindex(int base_q_idx, const rc::DeltaQParams& delta_q,
                          const rc::SegmentationParams& segmentation,
                          int segment_id, int current_q_index) noexcept;

}  // namespace av2enc::pipeline
