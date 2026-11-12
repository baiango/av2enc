// AV2 v1.0.0 TIP — CUDA motion-field construction and sample prediction.
//
// Scope:
//   §7.10.2  temporal scale of the motion field
//   §7.10.3  TIP hole fill
//   §7.10.4  block-average filter (Weight_Div_Mult)
//   §7.10.5  ProjStep==2 sample-gap fill
//   §7.10.6  build TIP: per-8x8/16x16 block MVs, per-4x4 MV grid
//   §7.13.3.2/.17/.18  TIP prediction and COMPOUND_AVERAGE blend
//
// The arithmetic lives in `tip_device.cuh`; it is shared by the kernels in
// `tip.cu` and by the host-emulation parity test (`tip_emulation_test.cpp`).
//
// Memory model (mandatory):
//   * Every device buffer comes from the single persistent
//     `av2enc::cuda::Arena` (or from caller-allocated device buffers through
//     `tip_workspace_attach`).
//   * `required_device_bytes()` reports the exact arena need of the largest
//     launch (motion field, block-average scratch, optional plan arrays).
//   * `tip_workspace_init()` allocates once.  Per frame callers take
//     `tip_frame_mark()` before optional transient work and `tip_frame_rewind()`
//     afterwards.  The launchers themselves never allocate, synchronise, or
//     copy.
//   * An undersized arena returns `Status::kArenaExhausted` and writes the
//     required size to the out-parameter; the arena bump pointer is rolled
//     back, so the caller can release/re-init it with the larger size.

#ifndef AV2ENC_CUDA_TIP_TIP_H_
#define AV2ENC_CUDA_TIP_TIP_H_

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "../../../arena.h"
#include "../../../error.h"
#include "codec/inter/temporal/temporal_types.h"

namespace av2enc::cuda::tip {

using Mv = inter::temporal::Mv;
using InterpFilter = inter::temporal::InterpFilter;
using TipOffsets = inter::temporal::TipOffsets;

// ---------------------------------------------------------------------------
// Motion field (§7.10)
// ---------------------------------------------------------------------------

// `[y8][x8]` field with `mv[2]`, validity and MotionFieldOffset.
struct TipField {
  int w8 = 0;
  int h8 = 0;
  std::int32_t* mv = nullptr;      // [y8][x8][2]
  std::uint8_t* valid = nullptr;   // [y8][x8]
  std::int32_t* offset = nullptr;  // [y8][x8]
};

// §7.10.4 needs a separate destination for the filtered field.
struct TipFieldScratch {
  std::int32_t* avg_mv = nullptr;     // [y8][x8][2]
  std::uint8_t* avg_valid = nullptr;  // [y8][x8]
};

// Inputs that the CPU reference reads from MotionFieldConfig + TipConfig.
struct TipMotionFieldConfig {
  int mi_cols = 0;
  int mi_rows = 0;
  int proj_step = 1;   // ProjStep
  int sb_64x64 = 1;    // SbSize == BLOCK_64X64
  int enable_tip = 1;  // false: setup is a no-op
  int order_hint = 0;
  int num_past_refs = 0;
  int num_future_refs = 0;
  int closest_past_order_hint = -1;    // kRestrictedOrderHint
  int closest_future_order_hint = -1;  // kRestrictedOrderHint
};

// §7.22 get_tip_offsets() for `TipMotionFieldConfig`.
TipOffsets tip_get_offsets(const TipMotionFieldConfig& config);

// One thread per `proj_step` lattice cell.  §7.10.2.
cudaError_t tip_scale_motion_field(const TipField& field,
                                   const TipMotionFieldConfig& config,
                                   cudaStream_t stream = nullptr);

// One thread per TMVP superblock; each superblock walks its cells in the
// reference row-major order.  §7.10.3.
cudaError_t tip_hole_fill_motion_field(const TipField& field,
                                       const TipMotionFieldConfig& config,
                                       cudaStream_t stream = nullptr);

// §7.10.4 block-average filter into `scratch`, then copies the lattice cells
// back into `field` (two asynchronous launches, no allocation).
cudaError_t tip_block_average_motion_field(const TipField& field,
                                           const TipMotionFieldConfig& config,
                                           const TipFieldScratch& scratch,
                                           cudaStream_t stream = nullptr);

// §7.10.5 sample-gap fill.  No-op unless `proj_step == 2`.
cudaError_t tip_fill_motion_field_sample_gap(
    const TipField& field, const TipMotionFieldConfig& config,
    cudaStream_t stream = nullptr);

// Full §7.10.2–.5 sequence: scale, optional hole fill + block average, then
// the ProjStep==2 sample gap fill.
cudaError_t tip_setup_motion_field(const TipField& field,
                                   const TipMotionFieldConfig& config,
                                   const TipFieldScratch& scratch,
                                   int allow_tip_hole_fill,
                                   cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Build TIP plan (§7.10.6)
// ---------------------------------------------------------------------------

struct TipPlanBlock {
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 2;
  int bh4 = 2;
  Mv mv[2];
  int motion_field_valid = 0;
};

struct TipPlan {
  int num_blocks = 0;       // output block count
  int block_capacity = 0;   // capacity on input
  int weight = 8;           // Tip_Weighting_Factor
  TipOffsets offsets{};
  TipPlanBlock* blocks = nullptr;
  Mv* mv_grid = nullptr;   // [mi_rows][mi_cols], list 0
  Mv* mv_grid1 = nullptr;  // optional [mi_rows][mi_cols], list 1
};

struct TipPlanConfig {
  int bw4 = 2;   // 2 for BLOCK_8X8, 4 for BLOCK_16X16
  int bh4 = 2;
  int tip_global_wtd_index = 0;  // 0..7
  Mv tip_global_mv{};
};

// Fills `plan->blocks`, `plan->mv_grid`, optional `plan->mv_grid1`, the
// offsets and the weight.  `plan->block_capacity` is checked on the host.
cudaError_t tip_build_plan(const TipField& field,
                           const TipMotionFieldConfig& config,
                           const TipPlanConfig& plan_config, TipPlan* plan,
                           cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Prediction (§7.13.3.2 / §7.13.3.17 / §7.13.3.18)
// ---------------------------------------------------------------------------

// A read-only plane.  `data` points at 8-bit or 16-bit LSB-aligned samples,
// selected by `sample_bytes`; `width`/`height` are the MI-aligned plane extent.
struct TipPlane {
  const void* data = nullptr;
  int stride = 0;
  int width = 0;
  int height = 0;
};

// One independent TIP prediction block job.  All pointers are device pointers.
struct TipPredictJob {
  TipPlane ref[2];
  void* dst = nullptr;
  int dst_stride = 0;
  int x = 0;  // plane coordinates of the block origin
  int y = 0;
  int w = 0;  // visible width/height (edge blocks are clipped by the caller)
  int h = 0;
  int sub_x = 0;
  int sub_y = 0;
  int frame_w = 0;  // luma frame dimensions for §7.13.3.17 scaling
  int frame_h = 0;
  Mv mv[2];
  int num_refs = 2;      // 1 when Tip_Weighting_Factor == 16
  int cwp_weight = 8;    // Tip_Weighting_Factor, may be negative
  int filter = 2;        // InterpFilter::kEighttapSharp
  int bit_depth = 8;     // 8 or 10
  int sample_bytes = 1;  // 1 (uint8) or 2 (uint16)
  int use_ref_area = 0;
  int area_first_x[2] = {0, 0};
  int area_first_y[2] = {0, 0};
  int area_last_x[2] = {0, 0};
  int area_last_y[2] = {0, 0};
};

// Predicts one block (one launch, job passed by value).
cudaError_t tip_predict_block(const TipPredictJob& job,
                              cudaStream_t stream = nullptr);

// One CUDA block per job.  `device_jobs` is device memory; the caller owns job
// validity (device memory cannot be inspected here).  `sample_bytes` selects
// the sample type and must match every job in the batch.  This is the encoder
// path.
cudaError_t tip_predict_batch(const TipPredictJob* device_jobs, int count,
                              int sample_bytes,
                              cudaStream_t stream = nullptr);

// Whole-plan prediction parameters.  The plan itself is device memory.
struct TipPlanPredictConfig {
  TipPlane ref[3][2];
  void* dst[3] = {nullptr, nullptr, nullptr};
  int dst_stride[3] = {0, 0, 0};
  int dst_width[3] = {0, 0, 0};
  int dst_height[3] = {0, 0, 0};
  int sub_x[3] = {0, 1, 1};
  int sub_y[3] = {0, 1, 1};
  int frame_w = 0;
  int frame_h = 0;
  int bit_depth = 8;
  int sample_bytes = 1;
  int cwp_weight = 8;
  int filter = 2;
};

// One CUDA block per plan block; predicts all three planes.
cudaError_t tip_predict_plan(const TipPlan& plan,
                             const TipPlanPredictConfig& config,
                             cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Arena workspace
// ---------------------------------------------------------------------------

struct TipWorkspaceConfig {
  int mi_cols = 0;
  int mi_rows = 0;
  int bw4 = 4;
  int bh4 = 4;
  int include_plan = 1;
  int block_capacity = 0;  // 0: derive ceil(mi_rows/bh4)*ceil(mi_cols/bw4)
};

inline constexpr std::size_t kTipArenaAlignment = 256;

inline std::size_t tip_align_bytes(std::size_t bytes) {
  return (bytes + kTipArenaAlignment - 1) & ~(kTipArenaAlignment - 1);
}

// Exact arena bytes for the largest launch of this configuration:
//   field mv   8 * w8*h8
//   field valid     w8*h8
//   field offset 4 * w8*h8
//   avg_mv     8 * w8*h8
//   avg_valid       w8*h8
//   blocks     sizeof(TipPlanBlock) * count        (include_plan)
//   mv_grid    sizeof(Mv) * mi_cols*mi_rows        (include_plan)
//   mv_grid1   sizeof(Mv) * mi_cols*mi_rows        (include_plan)
// Each array is rounded up to the arena's 256-byte alignment.
inline std::size_t required_device_bytes(const TipWorkspaceConfig& config) {
  if (config.mi_cols <= 0 || config.mi_rows <= 0 || config.bw4 <= 0 ||
      config.bh4 <= 0) {
    return 0;
  }
  const std::size_t w8 = static_cast<std::size_t>(config.mi_cols >> 1);
  const std::size_t h8 = static_cast<std::size_t>(config.mi_rows >> 1);
  if (w8 == 0 || h8 == 0) {
    return 0;
  }
  const std::size_t cells = w8 * h8;
  std::size_t bytes = tip_align_bytes(cells * 2 * sizeof(std::int32_t)) +
                      tip_align_bytes(cells * sizeof(std::uint8_t)) +
                      tip_align_bytes(cells * sizeof(std::int32_t)) +
                      tip_align_bytes(cells * 2 * sizeof(std::int32_t)) +
                      tip_align_bytes(cells * sizeof(std::uint8_t));
  if (config.include_plan != 0) {
    const std::size_t blocks =
        config.block_capacity > 0
            ? static_cast<std::size_t>(config.block_capacity)
            : static_cast<std::size_t>((config.mi_rows + config.bh4 - 1) /
                                       config.bh4) *
                  static_cast<std::size_t>((config.mi_cols + config.bw4 - 1) /
                                           config.bw4);
    const std::size_t grid = static_cast<std::size_t>(config.mi_cols) *
                            static_cast<std::size_t>(config.mi_rows);
    bytes += tip_align_bytes(blocks * sizeof(TipPlanBlock));
    bytes += tip_align_bytes(grid * sizeof(Mv));
    bytes += tip_align_bytes(grid * sizeof(Mv));
  }
  return bytes;
}

struct TipWorkspace {
  TipField field;
  TipFieldScratch scratch;
  TipPlan plan;
  std::size_t device_bytes = 0;  // arena bytes owned by this workspace
};

// Caller-allocated device buffers alternative to `tip_workspace_init`.  Every
// pointer must be device memory large enough for `TipWorkspaceConfig`.
struct TipWorkspaceBuffers {
  std::int32_t* mv = nullptr;
  std::uint8_t* valid = nullptr;
  std::int32_t* offset = nullptr;
  std::int32_t* avg_mv = nullptr;
  std::uint8_t* avg_valid = nullptr;
  TipPlanBlock* blocks = nullptr;
  Mv* mv_grid = nullptr;
  Mv* mv_grid1 = nullptr;
  int block_capacity = 0;
};

// Allocates the persistent field, scratch and plan arrays from `arena` once.
// On `Status::kArenaExhausted` the bump pointer is restored and
// `*required_bytes` receives the exact `required_device_bytes()` value.
inline Status tip_workspace_init(Arena& arena,
                                 const TipWorkspaceConfig& config,
                                 TipWorkspace* workspace,
                                 std::size_t* required_bytes = nullptr) {
  if (workspace == nullptr) {
    return Status::kInvalidArgument;
  }
  *workspace = TipWorkspace{};
  const std::size_t required = required_device_bytes(config);
  if (required_bytes != nullptr) {
    *required_bytes = required;
  }
  if (required == 0) {
    return Status::kInvalidArgument;
  }
  if (!arena.initialized()) {
    return Status::kNotInitialized;
  }
  if (arena.remaining() < required) {
    return Status::kArenaExhausted;
  }

  const std::uint64_t mark = arena.mark();
  const std::size_t w8 = static_cast<std::size_t>(config.mi_cols >> 1);
  const std::size_t h8 = static_cast<std::size_t>(config.mi_rows >> 1);
  const std::size_t cells = w8 * h8;
  const std::size_t blocks =
      config.block_capacity > 0
          ? static_cast<std::size_t>(config.block_capacity)
          : static_cast<std::size_t>((config.mi_rows + config.bh4 - 1) /
                                     config.bh4) *
                static_cast<std::size_t>((config.mi_cols + config.bw4 - 1) /
                                         config.bw4);
  const std::size_t grid = static_cast<std::size_t>(config.mi_cols) *
                           static_cast<std::size_t>(config.mi_rows);

  void* p_mv = nullptr;
  void* p_valid = nullptr;
  void* p_offset = nullptr;
  void* p_avg_mv = nullptr;
  void* p_avg_valid = nullptr;
  void* p_blocks = nullptr;
  void* p_grid0 = nullptr;
  void* p_grid1 = nullptr;
  Status status = arena.allocate_checked(cells * 2 * sizeof(std::int32_t), &p_mv);
  if (is_ok(status)) {
    status = arena.allocate_checked(cells * sizeof(std::uint8_t), &p_valid);
  }
  if (is_ok(status)) {
    status = arena.allocate_checked(cells * sizeof(std::int32_t), &p_offset);
  }
  if (is_ok(status)) {
    status = arena.allocate_checked(cells * 2 * sizeof(std::int32_t), &p_avg_mv);
  }
  if (is_ok(status)) {
    status = arena.allocate_checked(cells * sizeof(std::uint8_t), &p_avg_valid);
  }
  if (is_ok(status) && config.include_plan != 0) {
    status = arena.allocate_checked(blocks * sizeof(TipPlanBlock), &p_blocks);
    if (is_ok(status)) {
      status = arena.allocate_checked(grid * sizeof(Mv), &p_grid0);
    }
    if (is_ok(status)) {
      status = arena.allocate_checked(grid * sizeof(Mv), &p_grid1);
    }
  }
  if (!is_ok(status)) {
    (void)arena.rewind(mark);
    return status == Status::kArenaExhausted ? Status::kArenaExhausted
                                             : status;
  }

  workspace->field.w8 = static_cast<int>(w8);
  workspace->field.h8 = static_cast<int>(h8);
  workspace->field.mv = static_cast<std::int32_t*>(p_mv);
  workspace->field.valid = static_cast<std::uint8_t*>(p_valid);
  workspace->field.offset = static_cast<std::int32_t*>(p_offset);
  workspace->scratch.avg_mv = static_cast<std::int32_t*>(p_avg_mv);
  workspace->scratch.avg_valid = static_cast<std::uint8_t*>(p_avg_valid);
  workspace->plan.block_capacity = static_cast<int>(blocks);
  workspace->plan.blocks = static_cast<TipPlanBlock*>(p_blocks);
  workspace->plan.mv_grid = static_cast<Mv*>(p_grid0);
  workspace->plan.mv_grid1 = static_cast<Mv*>(p_grid1);
  workspace->device_bytes = static_cast<std::size_t>(arena.used() - mark);
  return Status::kOk;
}

// Uses caller-owned device buffers instead of the arena.  No allocation is
// performed; all pointers are validated and `block_capacity` is checked.
inline Status tip_workspace_attach(const TipWorkspaceConfig& config,
                                   const TipWorkspaceBuffers& buffers,
                                   TipWorkspace* workspace,
                                   std::size_t* required_bytes = nullptr) {
  if (workspace == nullptr) {
    return Status::kInvalidArgument;
  }
  *workspace = TipWorkspace{};
  const std::size_t required = required_device_bytes(config);
  if (required_bytes != nullptr) {
    *required_bytes = required;
  }
  if (required == 0 || buffers.mv == nullptr || buffers.valid == nullptr ||
      buffers.offset == nullptr || buffers.avg_mv == nullptr ||
      buffers.avg_valid == nullptr) {
    return Status::kInvalidArgument;
  }
  const std::size_t blocks =
      config.block_capacity > 0
          ? static_cast<std::size_t>(config.block_capacity)
          : static_cast<std::size_t>((config.mi_rows + config.bh4 - 1) /
                                     config.bh4) *
                static_cast<std::size_t>((config.mi_cols + config.bw4 - 1) /
                                         config.bw4);
  if (config.include_plan != 0 &&
      (buffers.blocks == nullptr || buffers.mv_grid == nullptr ||
       buffers.block_capacity < static_cast<int>(blocks))) {
    return Status::kInvalidArgument;
  }
  workspace->field.w8 = config.mi_cols >> 1;
  workspace->field.h8 = config.mi_rows >> 1;
  workspace->field.mv = buffers.mv;
  workspace->field.valid = buffers.valid;
  workspace->field.offset = buffers.offset;
  workspace->scratch.avg_mv = buffers.avg_mv;
  workspace->scratch.avg_valid = buffers.avg_valid;
  workspace->plan.block_capacity = buffers.block_capacity;
  workspace->plan.blocks = buffers.blocks;
  workspace->plan.mv_grid = buffers.mv_grid;
  workspace->plan.mv_grid1 = buffers.mv_grid1;
  workspace->device_bytes = 0;
  return Status::kOk;
}

// Per-frame scratch watermark.  The TIP launchers allocate nothing, so a frame
// that only calls them is net-zero; the mark/rewind pair makes that explicit
// and lets future per-frame scratch grow without changing the call pattern.
inline std::uint64_t tip_frame_mark(const Arena& arena) { return arena.mark(); }
inline Status tip_frame_rewind(Arena& arena, std::uint64_t mark) {
  return arena.rewind(mark) ? Status::kOk : Status::kInvalidArgument;
}

}  // namespace av2enc::cuda::tip

#endif  // AV2ENC_CUDA_TIP_TIP_H_
