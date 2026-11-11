// M1 GPU still-picture pipeline — shared contract and orchestrator entry point.
//
// One tile, one keyframe, 8-bit 4:2:0, byte-identical to the CPU M1 reference
// (`m1::encode_m1_still_picture`).  The CPU still owns the §8.2 syntax writer;
// this layer owns the frame/block plan hand-off, the persistent device arena
// and stream, and the staged GPU kernels as they land.
//
// Contract: docs/m1-gpu.md.  Every stage exchanges plain PODs; device pointers
// are explicit and come from the orchestrator's arena.  No stage allocates,
// synchronizes, or touches CUDA error state directly.  The orchestrator does at
// most one synchronize() per frame.
//
// Staging policy: while a component directory under `src/cuda/m1/<stage>/` has
// not landed, `encode_m1_gpu` runs the existing CPU M1 encoder and reports that
// stage as `kCpuReference`.  Frame bytes stay exact because the CPU reference
// produces them; the data-flow (plan -> host buffers -> frame bytes) is still
// exercised end to end.  `GpuResult::frame_bytes_from_cpu` records which path
// produced the bitstream.
#ifndef AV2ENC_CUDA_M1_GPU_H_
#define AV2ENC_CUDA_M1_GPU_H_

#include "encoder/m1/m1_still.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::cuda::m1gpu {

// ---------------------------------------------------------------------------
// Shared PODs
// ---------------------------------------------------------------------------

// Stage identifiers in execution order.  `kSched` is the wavefront/CUDA-graph
// scheduler adapter; it is optional and does not exist yet.
enum class Stage : std::uint8_t {
  kPlan = 0,
  kSearch,
  kTx,
  kRecon,
  kChroma,
  kCfl,
  kDeblock,
  kSched,
  kCount,
};

// How a stage contributed to this frame.  `kCpuReference` means the stage's
// effect is inside the CPU M1 encode used for this frame, not that a GPU kernel
// ran; `kGpu` is only reported after the stage hook returned success.
enum class StageState : std::uint8_t {
  kStub = 0,
  kCpuReference = 1,
  kGpu = 2,
};

inline const char* stage_name(Stage stage) noexcept {
  switch (stage) {
    case Stage::kPlan:
      return "plan";
    case Stage::kSearch:
      return "search";
    case Stage::kTx:
      return "tx";
    case Stage::kRecon:
      return "recon";
    case Stage::kChroma:
      return "chroma";
    case Stage::kCfl:
      return "cfl";
    case Stage::kDeblock:
      return "deblock";
    case Stage::kSched:
      return "sched";
    case Stage::kCount:
      break;
  }
  return "?";
}

inline const char* stage_state_name(StageState state) noexcept {
  switch (state) {
    case StageState::kStub:
      return "stub";
    case StageState::kCpuReference:
      return "cpu-reference";
    case StageState::kGpu:
      return "gpu";
  }
  return "?";
}

// Neighbour availability bits for intra prediction (§5.20.9.1 is_inside).
enum AvailabilityBit : std::uint8_t {
  kAvailLeft = 1u << 0,
  kAvailAbove = 1u << 1,
  kAvailAboveLeft = 1u << 2,
  kAvailAboveRight = 1u << 3,
  kAvailBelowLeft = 1u << 4,
};

// One leaf coding block, in §5.20 syntax order (raster over the tile).
//
// Geometry is in luma samples for `luma_*`/`tx_*` and chroma samples for
// `chroma_*`.  `plane_offset_*` is the sample offset of the block's top-left
// corner inside the padded recon plane (`FramePlan::y_stride` /
// `u_stride`).  `coeff_offset_*` is the element offset inside the per-plane
// coefficient level arrays.
struct BlockPlan {
  std::uint16_t mi_row = 0;  // 4x4 units
  std::uint16_t mi_col = 0;  // 4x4 units
  std::uint8_t luma_w = 0;
  std::uint8_t luma_h = 0;
  std::uint8_t chroma_w = 0;
  std::uint8_t chroma_h = 0;
  std::uint8_t tx_w = 0;         // adjusted luma TX (§7.15.2)
  std::uint8_t tx_h = 0;
  std::uint8_t chroma_tx_w = 0;
  std::uint8_t chroma_tx_h = 0;
  std::uint8_t has_chroma = 0;
  std::uint8_t avail = 0;        // AvailabilityBit mask
  std::uint8_t tree_depth = 0;
  std::uint8_t tx_upscale_w = 1;  // 2 for the 64 -> 32 adjusted luma TX
  std::uint8_t tx_upscale_h = 1;
  std::uint8_t reserved = 0;
  std::uint32_t plane_offset_y = 0;
  std::uint32_t plane_offset_u = 0;
  std::uint32_t plane_offset_v = 0;
  std::uint32_t coeff_offset_y = 0;
  std::uint32_t coeff_offset_u = 0;
  std::uint32_t coeff_offset_v = 0;
  // Packed-block scratch offsets (samples).  Prediction and inverse residual
  // are coded-size w*h blocks concatenated per plane; the orchestrator fills
  // these prefix sums after the plan stage returns.
  std::uint32_t pred_offset_y = 0;
  std::uint32_t pred_offset_u = 0;
  std::uint32_t pred_offset_v = 0;
  std::uint32_t residual_offset_y = 0;
  std::uint32_t residual_offset_u = 0;
  std::uint32_t residual_offset_v = 0;
};

// Host-side mode decision for one block.  Filled by the plan/host logic today
// (stage 1) or by the GPU candidate search (stage 2).
struct BlockDecision {
  std::uint8_t luma_mode = 0;      // §6.19.5.5 YMode, 0 = DC
  std::int8_t luma_angle_delta = 0;
  std::uint8_t uv_mode = 0;        // §6.19.5.5 UVMode
  std::uint8_t is_cfl = 0;
  std::int16_t cfl_alpha_u = 0;    // Q8, ±2048 max (§7.13.5)
  std::int16_t cfl_alpha_v = 0;
  std::uint8_t tx_type_y = 0;      // transform type syntax, DCT_DCT = 0
  std::uint8_t tx_type_u = 0;
  std::uint8_t tx_type_v = 0;
  std::uint16_t eob_y = 0;
  std::uint16_t eob_u = 0;
  std::uint16_t eob_v = 0;
  std::uint32_t sse = 0;           // luma SSE of the chosen mode
};

// Frame geometry plus the block list in syntax order.
//
// Strides mirror the CPU M1 reconstruction planes: MiCols/MiRows from §5.18.4.4
// rounded up to whole 64x64 superblocks.  Chroma planes are half size.
struct FramePlan {
  std::uint32_t width = 0;          // visible luma
  std::uint32_t height = 0;
  std::uint32_t padded_width = 0;    // == y_stride
  std::uint32_t padded_height = 0;
  std::uint32_t y_stride = 0;        // padded luma stride
  std::uint32_t u_stride = 0;        // padded chroma stride
  std::uint32_t v_stride = 0;
  std::uint16_t mi_cols = 0;         // 2 * ((width + 7) >> 3)
  std::uint16_t mi_rows = 0;
  std::uint16_t sb_cols = 0;
  std::uint16_t sb_rows = 0;
  std::uint8_t sb_size = 64;
  std::uint8_t chroma_shift_x = 1;
  std::uint8_t chroma_shift_y = 1;
  // 1 when this plan is the orchestrator's fallback raster (one PARTITION_NONE
  // block per superblock), not the real M1 partition tree.
  std::uint8_t is_placeholder = 1;
  std::uint8_t max_chroma_w = 0;
  std::uint8_t max_chroma_h = 0;
  std::uint32_t block_count = 0;
  std::uint32_t leaf_count = 0;
  std::uint32_t coeff_count_y = 0;
  std::uint32_t coeff_count_u = 0;
  std::uint32_t coeff_count_v = 0;
  std::span<const BlockPlan> blocks;  // syntax order; see GpuResult lifetime
};

// Option set for the M1 GPU path.  The first block mirrors M1Options.  The
// trailing fields are orchestrator controls and do not change the bitstream.
struct GpuOptions {
  std::uint8_t base_q_idx = 90;   // M1 pins 90 (coefficient CDF qctx 0)
  bool enable_cfl = true;
  bool enable_deblock = true;
  bool enable_edge_filter = true;
  bool luma_dc_only = true;
  int min_block_size = 8;
  int max_partition_depth = 4;

  // Pass-through M1 mode switches, needed to reproduce the CPU path exactly
  // when a debug mode is active.
  bool adaptive_cdf = false;
  bool force_single_block = false;
  bool force_dc = false;

  int device = 0;                    // CUDA ordinal
  std::uint32_t vram_cap_mb = 0;     // 0 = device-derived (see README)
  bool allow_cpu_fallback = true;    // false => fail if the GPU path is absent
  bool verify_against_cpu = true;    // compare GPU recon with the CPU reference
  bool verbose = false;              // one stderr line per stage

  // Legacy bring-up switch.  The full GPU pipeline is now the default when
  // every stage is compiled in; this flag no longer selects a path.  Kept so
  // existing callers keep compiling.
  bool enable_stage1_gpu = false;
};

// Explicit device planes.  Populated only when the orchestrator has a real
// plan and at least one GPU stage; all pointers are inside the arena.
struct U8Planes {
  std::uint8_t* y = nullptr;
  std::uint8_t* u = nullptr;
  std::uint8_t* v = nullptr;
  std::size_t y_bytes = 0;
  std::size_t u_bytes = 0;
  std::size_t v_bytes = 0;
};

struct I32Planes {
  std::int32_t* y = nullptr;
  std::int32_t* u = nullptr;
  std::int32_t* v = nullptr;
  std::size_t y_bytes = 0;
  std::size_t u_bytes = 0;
  std::size_t v_bytes = 0;
};

// Host + device view passed to every stage.  Host recon/coeff/pred buffers are
// padded (`FramePlan` strides); the source spans are the tightly packed picture
// planes.  `stream` is the orchestrator's `cudaStream_t` as an opaque pointer.
struct GpuFrame {
  FramePlan plan;

  // Host input (visible, tightly packed).
  std::span<const std::uint8_t> src_y;
  std::span<const std::uint8_t> src_u;
  std::span<const std::uint8_t> src_v;

  // Host outputs, padded.
  std::span<std::uint8_t> recon_y;
  std::span<std::uint8_t> recon_u;
  std::span<std::uint8_t> recon_v;
  std::span<std::int32_t> coeff_y;
  std::span<std::int32_t> coeff_u;
  std::span<std::int32_t> coeff_v;
  // Legacy padded-plane prediction view.  The stage-1 executor uses the
  // packed-block layout instead: BlockPlan::pred_offset_* addresses
  // coded-size w*h predictors concatenated per plane, and
  // BlockPlan::residual_offset_* does the same for the inverse residual.
  std::span<std::uint8_t> pred_y;
  std::span<std::uint8_t> pred_u;
  std::span<std::uint8_t> pred_v;
  std::span<BlockDecision> decisions;

  // Device views (arena-owned, orchestrator lifetime).
  U8Planes dev_src;
  U8Planes dev_recon;
  I32Planes dev_coeff;
  U8Planes dev_pred;
  BlockPlan* dev_blocks = nullptr;
  BlockDecision* dev_decisions = nullptr;
  void* stream = nullptr;  // cudaStream_t

  std::uint32_t block_count = 0;
  std::uint32_t leaf_count = 0;
};

struct GpuStageStatus {
  StageState plan = StageState::kStub;
  StageState search = StageState::kStub;
  StageState tx = StageState::kStub;
  StageState recon = StageState::kStub;
  StageState chroma = StageState::kStub;
  StageState cfl = StageState::kStub;
  StageState deblock = StageState::kStub;
  StageState sched = StageState::kStub;

  StageState get(Stage stage) const noexcept;
};

struct GpuStats {
  int block_count = 0;
  int leaf_count = 0;
  std::uint64_t host_upload_bytes = 0;    // device transfers, staged per frame
  std::uint64_t host_download_bytes = 0;
  int kernel_launches = 0;                // reported by stage hooks
  int sync_count = 0;                     // <= 1 per frame
  std::uint64_t arena_capacity_bytes = 0;
  std::uint64_t arena_used_bytes = 0;
  std::uint64_t device_total_bytes = 0;
  std::uint64_t device_free_bytes = 0;
  bool gpu_runtime_ready = false;
  char device_name[128] = {};
  int tx_jobs = 0;                 // TX descriptors actually launched
  bool gpu_frame_verified = false;   // GPU frame/recon == CPU reference
  double total_ms = 0.0;
};

// Encoder result.  `frame` is the finished IVF/OBU frame payload, identical to
// `m1::M1Result::frame` when the CPU fallback produced it.  Recon planes are
// visible-size (cropped from the padded host buffers).  `blocks` owns the
// storage behind `plan.blocks`; do not copy the result (copy is deleted), and
// after a move use the destination's `plan`.
struct GpuResult {
  GpuResult() = default;
  GpuResult(const GpuResult&) = delete;
  GpuResult& operator=(const GpuResult&) = delete;
  GpuResult(GpuResult&&) noexcept = default;
  GpuResult& operator=(GpuResult&&) noexcept = default;

  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;
  std::vector<std::uint8_t> frame;
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;

  FramePlan plan;
  std::vector<BlockPlan> blocks;
  std::vector<BlockDecision> decisions;
  // Populated only when the TX/recon stages actually ran on the GPU.
  std::vector<std::int32_t> coeff_y;
  std::vector<std::int32_t> coeff_u;
  std::vector<std::int32_t> coeff_v;
  std::vector<std::uint8_t> pred_y;
  std::vector<std::uint8_t> pred_u;
  std::vector<std::uint8_t> pred_v;

  GpuStageStatus stages;
  GpuStats stats;
  // False when the GPU stages supplied the decisions/coefficients and the CPU
  // §8.2 writer assembled the frame from them.  True on the CPU fallback.
  bool frame_bytes_from_cpu = true;

  // Point `plan.blocks` at `blocks`.  The orchestrator calls this before
  // returning; tests may call it after moving a result.
  void rebind_plan() noexcept {
    plan.blocks = std::span<const BlockPlan>(blocks);
  }
};

// Public entry point.  Returns false and fills `error` on invalid options,
// CUDA failure, or a GPU/CPU parity failure when `verify_against_cpu` is set.
// On success `result.frame` matches the CPU M1 reference for the same picture
// and the equivalent M1Options.
bool encode_m1_gpu(const ::av2enc::m1::StillPicture& picture,
                   const GpuOptions& options,
                   GpuResult& result, std::string& error);

// ---------------------------------------------------------------------------
// Optional stage hooks
//
// Stage owners implement these entry points as they land.  They are declared
// weak, so the orchestrator links and runs (CPU fallback) before every stage
// exists.  An implementation must:
//   * use the arena pointers in `GpuFrame` and the explicit `stream`;
//   * never call cudaMalloc/cudaFree or cudaDeviceSynchronize;
//   * return false with a non-empty error on failure.
//
// `build_plan` fills `plan` and the block list (leaf count, possibly larger
// than one block per superblock); it may resize `blocks`, and `plan.blocks`
// must point at that vector.  The remaining hooks consume `frame.plan.blocks`
// and the arena planes in `frame`.
//
// Mode decisions: `frame.dev_decisions` (device) is authoritative for the
// block stages; a host-decided stage may write `frame.decisions` and upload it.
// The orchestrator downloads `dev_decisions` after the single per-frame sync,
// so a stage never synchronizes to read another stage's output.
//
// The hooks live in a nested namespace so a component header can coexist.
// ---------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
#define AV2ENC_M1GPU_WEAK __attribute__((weak))
#else
#define AV2ENC_M1GPU_WEAK
#endif

namespace stage {

bool build_plan(const ::av2enc::m1::StillPicture& picture,
                const GpuOptions& options,
                FramePlan& plan, std::vector<BlockPlan>& blocks,
                std::string& error) AV2ENC_M1GPU_WEAK;

bool search(const GpuFrame& frame, std::string& error) AV2ENC_M1GPU_WEAK;

bool transform(const GpuFrame& frame, std::string& error) AV2ENC_M1GPU_WEAK;

bool reconstruct(const GpuFrame& frame, std::string& error) AV2ENC_M1GPU_WEAK;

bool chroma(const GpuFrame& frame, std::string& error) AV2ENC_M1GPU_WEAK;

bool cfl(const GpuFrame& frame, std::string& error) AV2ENC_M1GPU_WEAK;

bool deblock(const GpuFrame& frame, std::string& error) AV2ENC_M1GPU_WEAK;

// Optional wavefront/CUDA-graph adapter.  When present and it returns true it
// owns the block-stage launches (search/tx/recon/chroma/cfl/deblock); returning
// false means "no graph", and the orchestrator calls the individual hooks.
bool schedule(const GpuFrame& frame, std::string& error) AV2ENC_M1GPU_WEAK;

}  // namespace stage

#undef AV2ENC_M1GPU_WEAK

// Number of stage hooks currently linked into this binary.  Used by tests and
// by the `--m1-gpu` readiness check; 0 means "CPU reference fallback only".
int linked_stage_count() noexcept;

}  // namespace av2enc::cuda::m1gpu

#endif  // AV2ENC_CUDA_M1_GPU_H_
