// M1 GPU still-picture pipeline — shared contract and orchestrator entry point.
//
// One tile, one keyframe, 8-bit 4:2:0, byte-identical to the CPU M1 reference
// (`av2enc::still::encode_still_picture`).  The CPU still owns the §8.2 syntax writer;
// this layer owns the frame/block plan hand-off, the persistent device arena
// and stream, and the staged GPU kernels as they land.
//
// Contract: docs/m1-gpu.md.  Every stage exchanges plain PODs; device pointers
// are explicit and come from the orchestrator's arena.  No stage allocates,
// synchronizes, or touches CUDA error state directly.  The orchestrator does at
// most one synchronize() per frame.
//
// Staging policy: while a component directory under `src/cuda/pipeline/still/<stage>/` has
// not landed, `encode_still_gpu` runs the existing CPU M1 encoder and reports that
// stage as `kCpuReference`.  Frame bytes stay exact because the CPU reference
// produces them; the data-flow (plan -> host buffers -> frame bytes) is still
// exercised end to end.  `GpuResult::frame_bytes_from_cpu` records which path
// produced the bitstream.
#ifndef AV2ENC_CUDA_STILL_GPU_H_
#define AV2ENC_CUDA_STILL_GPU_H_

#include "pipeline/still/still_encoder.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::cuda::still_gpu {

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
  kFilters,
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
    case Stage::kFilters:
      return "filters";
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
  // Owned §5.20.3.1 ChromaMi block.  A non-owning extended/sub-8 leaf has
  // has_chroma == 0 and codes no U/V payload; an owning leaf's chroma origin
  // can differ from its luma origin.
  std::uint8_t chroma_offset = 0;
  std::uint16_t chroma_mi_row = 0;
  std::uint16_t chroma_mi_col = 0;
  std::uint8_t chroma_mi_size = 0;  // partition::BlockSize (luma units)
  std::uint8_t chroma_mi_w4 = 0;    // ChromaMi extent, luma 4x4 units
  std::uint8_t chroma_mi_h4 = 0;
  // §6.19.3.1 TreeType: 0 SHARED, 1 LUMA_PART, 2 CHROMA_PART.
  std::uint8_t tree_type = 0;
  std::uint8_t avail = 0;        // AvailabilityBit mask
  std::uint8_t tree_depth = 0;
  // §7.13.2.1 above-right / below-left availability counts in 4x4 units,
  // exported by the CPU plan (superblock-local `clear_block_decoded_flags`
  // replay).  The orchestrator feeds them to the luma/chroma search so the
  // counts follow the plan's real SbSize instead of a hard-coded 64.
  std::uint8_t n4_above_right = 0;
  std::uint8_t n4_below_left = 0;
  std::uint8_t chroma_n4_above_right = 0;
  std::uint8_t chroma_n4_below_left = 0;
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
  // §5.20.5.5 MRL.  mrl_index == 0 is the ordinary prediction; the secondary
  // line needs mrl_index > 0 and is unavailable on a 4x4 block.
  std::uint8_t mrl_index = 0;
  std::uint8_t mrl_sec_index = 0;
  // §5.20.7.9 DIP alternative to DC.  `use_dip` must be 0 unless the block is
  // DC and the size predicate holds.
  std::uint8_t use_dip = 0;
  std::uint8_t dip_mode = 0;
  std::uint8_t dip_transpose = 0;
  // §5.20.7.32 CfL signalling: 1 = derived (default), 0 = CFL_EXPLICIT,
  // 2 = CFL_MULTI (MHCCP).  `cfl_alpha_signs`/`cfl_alpha_u`/`cfl_alpha_v` are
  // the explicit syntax values; `cfl_mh_dir` is the MHCCP direction.
  std::uint8_t cfl_index = 1;
  std::uint8_t cfl_alpha_signs = 0;
  std::uint8_t cfl_alpha_u_syntax = 0;  // magnitude minus one (explicit)
  std::uint8_t cfl_alpha_v_syntax = 0;
  std::uint8_t cfl_mh_dir = 0;
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

// Option set for the M1 GPU path.  The first block mirrors StillOptions.  The
// trailing fields are orchestrator controls and do not change the bitstream.
struct GpuOptions {
  // GPU pinch: the TX/batch path still bakes qlookup(90) = 528, so the GPU
  // rejects QP != 90 until its quantiser uses qlookup(base_q_idx).  The CPU M1
  // path supports 1..255 with the matching §8.3.1 qctx slice.
  std::uint8_t base_q_idx = 90;
  bool enable_cfl = true;
  int cfl_ds_filter = 0;           // CPU --still-cfl-ds (0..3)
  bool enable_deblock = true;
  bool enable_edge_filter = true;
  bool luma_dc_only = true;
  int min_block_size = 8;
  int max_partition_depth = 4;
  // §5.4.3/§5.18.1 sequence superblock selection: 64 (default), 128, or 256.
  // An intra still clamps 256 to 128x128 coding roots, so 256 only changes the
  // sequence-level bit.  The GPU plan/search/recon path uses the same effective
  // frame SbSize as the CPU reference.
  int sb_size = 64;

  // Pass-through M1 mode switches, needed to reproduce the CPU path exactly
  // when a debug mode is active.
  bool adaptive_cdf = false;
  bool force_single_block = false;
  bool force_dc = false;
  bool uv_full = false;
  // Intra tool decisions wired on the GPU decision path.
  bool enable_ibp = false;         // CPU --still-ibp (§5.4.5, no per-block bit)
  bool ext_partitions = false;     // CPU --still-ext-partitions
  bool sdp = false;                // CPU --still-sdp
  bool sub8 = false;               // CPU --still-sub8
  bool enable_mrl = false;         // CPU --still-mrl (§5.20.5.5)
  bool enable_dip = false;         // CPU --still-dip  (§5.20.7.9)
  bool cfl_explicit = false;       // CPU --still-cfl-explicit
  bool mhccp = false;              // CPU --still-mhccp
  // Tools integrated on the CPU path but not yet ported to the GPU pipeline.
  // Every one of them is present here so the CLI cannot silently drop a flag;
  // gpu_options_to_m1() maps them and gpu_m3_options_supported() rejects the
  // ones the GPU cannot code yet.  All of the decision knobs are carried too,
  // so the internal CPU parity encode sees exactly the same configuration.
  bool enable_cdef = false;
  int cdef_num_strengths = 4;      // 1..8 (CPU --still-cdef-strengths)
  int cdef_damping = 3;            // 3..6
  bool cdef_on_skip_txfm = false;
  bool enable_ccso = false;
  int ccso_bands_log2 = 6;         // 0..6 (CPU --still-ccso-bands)
  bool ccso_edge = false;          // CPU --still-ccso-edge
  bool enable_wiener = false;      // CPU --still-wiener
  bool enable_chroma_lr = false;   // CPU --still-chroma-lr
  bool enable_pc_wiener = false;   // CPU --still-pc-wiener
  bool enable_gdf = false;         // CPU --still-gdf
  bool enable_grain = false;       // CPU --still-grain
  // §5.18.6.1/§7.14.4 DC/AC delta-Q.  The CPU reference plans the frame deltas;
  // the GPU TX stage feeds the resolved per-plane deltas to the device
  // delta-Q quantizer.
  bool delta_q = false;            // CPU --still-dc-ac-delta-q
  // §5.18.6.2 quantizer matrices (spec-defined default levels).
  bool qm = false;                 // CPU --still-qm
  int qm_level = 3;                // CPU --still-qm-level
  bool rdoq = false;
  double rdoq_sse_scale = 0.125;
  bool tx_type = false;
  double tx_type_sse_scale = 0.125;
  bool rdo_intra = false;
  double rdo_sse_scale = 0.125;

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

// Exact GpuOptions -> StillOptions mapping.  This is the one place the GPU's
// internal CPU parity encode gets its configuration, so a flag that the CLI
// parsed can never be missing from the parity check.  Both the orchestrator and
// the host-side option test call this function.
inline ::av2enc::still::StillOptions gpu_options_to_m1(const GpuOptions& options) {
  ::av2enc::still::StillOptions cpu;
  cpu.base_q_idx = options.base_q_idx;
  cpu.enable_cfl = options.enable_cfl;
  cpu.cfl_ds_filter = options.cfl_ds_filter;
  cpu.enable_deblock = options.enable_deblock;
  cpu.enable_intra_edge_filter = options.enable_edge_filter;
  cpu.luma_dc_only = options.luma_dc_only;
  cpu.min_block_size = options.min_block_size;
  cpu.max_partition_depth = options.max_partition_depth;
  cpu.sb128 = options.sb_size == 128;
  cpu.sb256 = options.sb_size == 256;
  cpu.adaptive_cdf = options.adaptive_cdf;
  cpu.force_single_block = options.force_single_block;
  cpu.force_dc = options.force_dc;
  cpu.uv_full = options.uv_full;
  cpu.enable_ibp = options.enable_ibp;
  cpu.ext_partitions = options.ext_partitions;
  cpu.sdp = options.sdp;
  cpu.sub8 = options.sub8;
  cpu.enable_mrl = options.enable_mrl;
  cpu.enable_dip = options.enable_dip;
  cpu.cfl_explicit = options.cfl_explicit;
  cpu.mhccp = options.mhccp;
  cpu.rdo_intra = options.rdo_intra;
  cpu.rdo_sse_scale = options.rdo_sse_scale;
  cpu.rdoq = options.rdoq;
  cpu.rdoq_sse_scale = options.rdoq_sse_scale;
  cpu.tx_type = options.tx_type;
  cpu.tx_type_sse_scale = options.tx_type_sse_scale;
  cpu.enable_cdef = options.enable_cdef;
  cpu.cdef_num_strengths = options.cdef_num_strengths;
  cpu.cdef_damping = options.cdef_damping;
  cpu.cdef_on_skip_txfm = options.cdef_on_skip_txfm;
  cpu.enable_ccso = options.enable_ccso;
  cpu.ccso_bands_log2 = options.ccso_bands_log2;
  cpu.ccso_edge = options.ccso_edge;
  cpu.enable_wiener = options.enable_wiener;
  cpu.enable_chroma_lr = options.enable_chroma_lr;
  cpu.enable_pc_wiener = options.enable_pc_wiener;
  cpu.enable_gdf = options.enable_gdf;
  cpu.enable_grain = options.enable_grain;
  cpu.delta_q = options.delta_q;
  cpu.qm = options.qm;
  cpu.qm_level = options.qm_level;
  return cpu;
}

// M3 tool options the GPU pipeline can code today.  Filters that have a wired
// GPU stage (CDEF, CCSO band-offset-only and edge-classified) and metadata-only
// tools (film grain, whose synthesis is a decoder-side output process with no
// encoder pixel work) are accepted; every remaining tool must be
// rejected by name instead of being accepted and silently dropped.  `error`
// receives the first unsupported flag (the gate classifies a message containing
// "--still-" plus "not ported" as SKIP-REJECTED).
inline bool gpu_m3_options_supported(const GpuOptions& options,
                                     std::string* error) {
  if (options.rdoq) {
    if (error != nullptr) {
      *error =
          "M1 GPU: --still-rdoq is not ported to the GPU pipeline yet; use --still "
          "(CPU)";
    }
    return false;
  }
  if (options.tx_type) {
    if (error != nullptr) {
      *error =
          "M1 GPU: --still-tx-type is not ported to the GPU pipeline yet; use "
          "--still (CPU)";
    }
    return false;
  }
  if (options.rdo_intra) {
    if (error != nullptr) {
      *error =
          "M1 GPU: --still-rdo-intra is not ported to the GPU pipeline yet; use "
          "--still (CPU)";
    }
    return false;
  }
  if (options.uv_full) {
    if (error != nullptr) {
      *error =
          "M1 GPU: --still-uv-full is not ported to the GPU pipeline yet (the "
          "chroma job lacks the luma-mode/UVSmooth inputs); use --still (CPU)";
    }
    return false;
  }
  return true;
}

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

// Luma-only 16-bit working planes used by the PC-Wiener and GDF kernels, whose
// device cores take `uint16_t` samples at every bit depth.  The still GPU path
// converts from/to its 8-bit reconstruction at the stage boundary.
struct U16Planes {
  std::uint16_t* y = nullptr;
  std::size_t y_bytes = 0;
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
  StageState filters = StageState::kStub;
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
// `av2enc::still::StillResult::frame` when the CPU fallback produced it.  Recon planes are
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
// and the equivalent StillOptions.
bool encode_still_gpu(const ::av2enc::still::StillPicture& picture,
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
#define AV2ENC_STILL_GPU_WEAK __attribute__((weak))
#else
#define AV2ENC_STILL_GPU_WEAK
#endif

namespace stage {

bool build_plan(const ::av2enc::still::StillPicture& picture,
                const GpuOptions& options,
                FramePlan& plan, std::vector<BlockPlan>& blocks,
                std::string& error) AV2ENC_STILL_GPU_WEAK;

bool search(const GpuFrame& frame, std::string& error) AV2ENC_STILL_GPU_WEAK;

bool transform(const GpuFrame& frame, std::string& error) AV2ENC_STILL_GPU_WEAK;

bool reconstruct(const GpuFrame& frame, std::string& error) AV2ENC_STILL_GPU_WEAK;

bool chroma(const GpuFrame& frame, std::string& error) AV2ENC_STILL_GPU_WEAK;

bool cfl(const GpuFrame& frame, std::string& error) AV2ENC_STILL_GPU_WEAK;

bool deblock(const GpuFrame& frame, std::string& error) AV2ENC_STILL_GPU_WEAK;

// Optional wavefront/CUDA-graph adapter.  When present and it returns true it
// owns the block-stage launches (search/tx/recon/chroma/cfl/deblock); returning
// false means "no graph", and the orchestrator calls the individual hooks.
bool schedule(const GpuFrame& frame, std::string& error) AV2ENC_STILL_GPU_WEAK;

}  // namespace stage

#undef AV2ENC_STILL_GPU_WEAK

// Number of stage hooks currently linked into this binary.  Used by tests and
// by the `--still-gpu` readiness check; 0 means "CPU reference fallback only".
int linked_stage_count() noexcept;

}  // namespace av2enc::cuda::still_gpu

#endif  // AV2ENC_CUDA_STILL_GPU_H_
