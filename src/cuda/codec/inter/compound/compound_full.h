#ifndef AV2ENC_CUDA_COMPOUND_FULL_COMPOUND_FULL_H_
#define AV2ENC_CUDA_COMPOUND_FULL_COMPOUND_FULL_H_

// Full compound-prediction GPU path for the M2 multiref layer (AV2 v1.0.0).
//
// Scope:
//   * §7.13.3.14 compound prediction: AVERAGE, CWP weights, implicit mask,
//     WEDGE, DIFFWTD, inter-intra (§7.13.3.29/.30);
//   * skip-mode style derived prediction: both references predicted with the
//     same MV (no residual), batched over blocks;
//   * §7.13.3.8-.12 OPFL refinement (`OpflMvs` / `MvDeltas`);
//   * §7.13.3.3-.6 refine-MV pixel-step search and the final
//     EIGHTTAP_SHARP + compound-average prediction.
//
// 8-bit and 10-bit 4:2:0 samples. All block sizes 4x4..64x64, including edge
// blocks whose rectangles are clipped at the frame boundary.
//
// Memory contract
// ---------------
// Every device buffer this module needs is allocated once, from the caller's
// persistent `av2enc::cuda::Arena` (src/cuda/arena.h), by
// `workspace_create()`. Launchers never allocate, never synchronize and never
// copy implicitly. `required_device_bytes(config)` is the exact size of the
// persistent set; a short arena returns `Status::kArenaExhausted` together
// with that size so the caller can grow the arena and retry.
//
// Reference frames, prediction planes and destinations stay caller-owned
// (frame buffers come from the encoder's own arena regions). Only descriptors,
// the per-job luma-mask pool and the wedge-mask cache live in the workspace.
//
// Bit-exactness
// -------------
// `compound_full_device.cuh` is the single transcription of the arithmetic.
// It is compiled by nvcc for the kernels and by the host-emulation test, so the
// parity tests exercise the code the GPU runs. For 8-bit, the blend output is
// also compared against `inter::adv::run_compound_job()` (the CPU reference)
// and OPFL/refine-MV against `inter::temporal::{opfl_compute,search_refinemv,
// refine_predict_block}`. 10-bit blend arithmetic is compared against the
// independent literal reference in `compound_full_reference.h`; the CPU
// advanced module hard-clips to 8 bits, so it is not a 10-bit oracle.

#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "codec/inter/advanced/compound_common.h"
#include "codec/inter/mc/mc_types.h"

namespace av2enc::cuda::compound_full {

namespace adv = av2enc::inter::adv;
namespace mc = av2enc::inter::mc;

// Integer constants are enumerators so they are compile-time values with no
// storage in either the host or the device pass.
enum : int {
  kMaxBlockDim = 64,           // M2 largest coding block
  kMaxMaskSamples = 64 * 64,   // one luma mask per job
  kMaxJobsDefault = 64,
  kWedgeTypes = 68,
  kWedgeCacheEntries = 2 * 68,
  kCompoundWedge = 0,          // §6.19.7.15
  kCompoundDiffWtd = 1,
  kCompoundAverage = 2,
  kCompoundIntra = 3,
};

// Byte size of the persistent wedge-mask cache (see compound_full_device.cuh
// for the layout and offsets). Host-only size arithmetic.
inline constexpr std::uint64_t kWedgeCacheDeviceBytes = 1958400ull;

// One read-only reference plane. `data` points at uint8_t samples for 8-bit
// and at LSB-aligned uint16_t samples for 10-bit. `stride`, `width`, `height`
// are in samples. `width`/`height` are the visible (or padded) plane extent the
// prediction may read; reads are clamped to it, matching the CPU reference's
// `PlaneView`.
struct RefPlane {
  const void* data = nullptr;
  int stride = 0;
  int width = 0;
  int height = 0;
  int is_16bit = 0;
};

// ---------------------------------------------------------------------------
// Compound / inter-intra blend
// ---------------------------------------------------------------------------
//
// Mirrors `adv::CompoundJob` plus 10-bit destinations and an optional
// plane-independent luma prediction for chroma DIFFWTD.
//
// `dst`, `mask_out`, `luma_mask`, `p0`, `p1`, `intra` and `luma_p0`/`luma_p1`
// are device pointers.
struct BlendJob {
  // Output geometry, in samples of this plane.
  int plane = 0;
  int sub_x = 0;
  int sub_y = 0;
  int rect_x = 0;
  int rect_y = 0;
  int rect_w = 0;
  int rect_h = 0;
  int bsize = 0;       // luma block size; selects the wedge mask
  int bit_depth = 8;

  // Compound predictions at 1/16 sample precision (int16_t for both depths).
  const std::int16_t* p0 = nullptr;
  int p0_stride = 0;
  const std::int16_t* p1 = nullptr;
  int p1_stride = 0;

  // Inter-intra inputs: integer samples (`intra_16bit` selects uint16_t).
  const void* intra = nullptr;
  int intra_stride = 0;
  int intra_16bit = 0;

  // Chroma wedge/diffwtd mask source. `luma_mask` is the plane-0 mask written
  // by an earlier job. `luma_p0`/`luma_p1` (1/16, luma geometry) let a chroma
  // DIFFWTD job derive the mask by itself, so a batch needs no job ordering.
  const std::uint8_t* luma_mask = nullptr;
  int luma_mask_stride = 0;
  const std::int16_t* luma_p0 = nullptr;
  int luma_p0_stride = 0;
  const std::int16_t* luma_p1 = nullptr;
  int luma_p1_stride = 0;

  // Optional plane-0 mask output (luma geometry).
  std::uint8_t* mask_out = nullptr;
  int mask_out_stride = 0;

  // Destination (`dst_16bit` selects uint16_t; stride in samples).
  void* dst = nullptr;
  int dst_stride = 0;
  int dst_16bit = 0;

  // Mode. `cwp_weight` is the derived weight (0..16); `type` is one of the
  // kCompound* constants above.
  int type = kCompoundAverage;
  int cwp_weight = 8;  // CWP_EQUAL
  int wedge_index = 0;
  int wedge_sign = 0;
  int mask_type = 0;

  // Implicit-mask blend eligibility (§7.13.3.14): accepted as given.
  int imp_mask_eligible = 0;
  int ref_start_x[2] = {0, 0};
  int ref_start_y[2] = {0, 0};
  int mi_cols = 0;
  int mi_rows = 0;

  // Inter-intra: set `is_inter_intra`, pass the intra prediction, and have
  // `dst` pre-filled with the clipped inter prediction.
  int is_inter_intra = 0;
  int interintra_mode = 0;  // adv::InterIntraMode (DC/V/H/SMOOTH)

  // Optional precomputed wedge-mask cache (workspace.wedge_cache). When null
  // the wedge mask is evaluated on the fly; both paths are bit-identical.
  const std::uint8_t* wedge_cache = nullptr;
};

// ---------------------------------------------------------------------------
// Skip-mode derived prediction
// ---------------------------------------------------------------------------
//
// Both lists predict with the same MV; the result is the §7.13.3.1 compound
// average (CWP_EQUAL) written directly as reconstruction, with no residual.
// All three planes are produced by one job.
struct SkipJob {
  int bit_depth = 8;
  RefPlane ref[2][3];
  int frame_w = 0;  // luma frame dimensions, needed by §7.13.3.17
  int frame_h = 0;
  int mi_cols = 0;  // reference MI dimensions
  int mi_rows = 0;

  int x = 0;  // luma rectangle, even in 4:2:0
  int y = 0;
  int w = 0;
  int h = 0;
  int bsize = 0;
  int mv_row = 0;  // 1/8 luma samples, shared by both lists
  int mv_col = 0;
  int filter = static_cast<int>(mc::InterpFilter::kEighttap);
  int cwp_weight = 8;  // CWP_EQUAL
  // §5.20.7.16 / §7.13.3.27-.30 masked blend.  `type` is one of the
  // kCompound* constants above; AVERAGE keeps the historical CWP blend.
  // WEDGE/DIFFWTD build the luma mask from `bsize`/`wedge_*`/`mask_type` and
  // subsample it for chroma exactly like `adv::compound_mask_sample()`.
  int type = kCompoundAverage;
  int wedge_index = 0;
  int wedge_sign = 0;
  int mask_type = 0;

  void* dst[3] = {nullptr, nullptr, nullptr};
  int dst_stride[3] = {0, 0, 0};
  int dst_16bit[3] = {0, 0, 0};
  // When set, `dst` is a block-local buffer and the store index is
  // `r * dst_stride + c` instead of the frame-absolute `(y + r) * stride + x`.
  // The candidate-SSE batch uses this so many blends of one block do not race.
  int dst_local = 0;
};

// ---------------------------------------------------------------------------
// OPFL (§7.13.3.8-.12)
// ---------------------------------------------------------------------------
//
// Inputs are the two single-reference predictions as integer samples. Outputs
// are `opfl_mvs` (1/16 luma) and `mv_deltas` (1/8 luma) indexed by 4x4 cells:
//   index = ((row4 * stride4 + col4) * 2 + list) * 2 + comp
// with comp 0 = row, 1 = column, exactly like `temporal::OpflOutput`.
struct OpflJob {
  int w = 0;  // multiple of n, <= 64
  int h = 0;
  int bit_depth = 8;
  const void* p0 = nullptr;  // integer samples, w x h block-local
  int p0_stride = 0;
  int p0_16 = 0;
  const void* p1 = nullptr;
  int p1_stride = 0;
  int p1_16 = 0;
  int mv[2][2] = {{0, 0}, {0, 0}};  // [list][comp], 1/8 luma
  int order_hint = 0;
  int ref0_order_hint = 0;
  int ref1_order_hint = 0;
  int n = 8;       // optical-flow block size: 8 or 4
  int opt_x = 0;   // subblock offset inside the inter block (samples)
  int opt_y = 0;
  std::int16_t* opfl_mvs = nullptr;
  int opfl_stride4 = 0;
  std::int16_t* mv_deltas = nullptr;
  int delta_stride4 = 0;
};

// ---------------------------------------------------------------------------
// Refine-MV (§7.13.3.3-.6)
// ---------------------------------------------------------------------------

struct RefAreaInput {
  int w = 0;  // luma block dimensions
  int h = 0;
  int sub_x = 0;
  int sub_y = 0;
  int x = 0;  // plane sample coordinates
  int y = 0;
  int frame_w = 0;
  int frame_h = 0;
  int ref_mi_cols[2] = {0, 0};
  int ref_mi_rows[2] = {0, 0};
  int ref_luma_w[2] = {0, 0};
  int ref_luma_h[2] = {0, 0};
  int cand_mv[2][2] = {{0, 0}, {0, 0}};
  int num_refs = 2;
};

// §7.13.3.4/.5 get_ref_area(): out[list] = {first_x, first_y, last_x, last_y}.
AV2ENC_ADV_HD void ref_area_compute(const RefAreaInput& in, int out[2][4]);

struct RefineSearchJob {
  int bit_depth = 8;
  int x = 0;  // luma block origin (search expands it by 2)
  int y = 0;
  int w = 0;
  int h = 0;
  int frame_w = 0;
  int frame_h = 0;
  RefPlane ref[2];
  int cand_mv[2][2] = {{0, 0}, {0, 0}};
  int num_refs = 2;
  int tip_pred = 0;
  int switchable = 1;
  int use_ref_area = 1;
  int area[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
};

struct RefineResult {
  int dx = 0;
  int dy = 0;
  int offset_mv[2][2] = {{0, 0}, {0, 0}};
  std::int64_t best_sad = 0;
};

struct RefinePredictJob {
  int bit_depth = 8;
  int sub_x = 0;
  int sub_y = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int frame_w = 0;
  int frame_h = 0;
  RefPlane ref[2];
  int num_refs = 2;
  int offset_mv[2][2] = {{0, 0}, {0, 0}};
  int cwp_weight = 8;  // CWP_EQUAL
  int use_ref_area = 1;
  int area[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
  void* dst = nullptr;
  int dst_stride = 0;
  int dst_16bit = 0;
};

// ---------------------------------------------------------------------------
// Arena workspace
// ---------------------------------------------------------------------------

struct CompoundFullConfig {
  // Maximum jobs in one batch launch, shared by every family.
  int max_jobs = kMaxJobsDefault;
  // Include the 1.87 MiB wedge-mask cache. Mask evaluation still works without
  // it (on the fly), but the GPU path uses the cache when present.
  int wedge_cache = 1;
  // Luma-mask samples reserved per job for plane-0 mask output.
  int max_mask_samples = kMaxMaskSamples;
};

// Exact persistent device bytes for `config`, including arena 256-byte
// rounding of every allocation. Returns 0 for an invalid configuration.
inline std::uint64_t required_device_bytes(const CompoundFullConfig& config) {
  if (config.max_jobs <= 0 || config.max_mask_samples <= 0 ||
      config.max_mask_samples > kMaxMaskSamples) {
    return 0;
  }
  const auto align_up = [](std::uint64_t value) {
    return (value + 255ull) & ~255ull;
  };
  const std::uint64_t jobs = static_cast<std::uint64_t>(config.max_jobs);
  std::uint64_t total = 0;
  total += align_up(sizeof(BlendJob) * jobs);
  total += align_up(sizeof(SkipJob) * jobs);
  total += align_up(sizeof(OpflJob) * jobs);
  total += align_up(sizeof(RefineSearchJob) * jobs);
  total += align_up(sizeof(RefinePredictJob) * jobs);
  total += align_up(sizeof(RefineResult) * jobs);
  total += align_up(static_cast<std::uint64_t>(config.max_mask_samples) * jobs);
  if (config.wedge_cache) {
    total += align_up(kWedgeCacheDeviceBytes);
  }
  return total;
}

// Persistent buffers carved from one Arena. All pointers are device pointers.
struct CompoundFullWorkspace {
  BlendJob* jobs = nullptr;
  SkipJob* skip_jobs = nullptr;
  OpflJob* opfl_jobs = nullptr;
  RefineSearchJob* refine_search_jobs = nullptr;
  RefinePredictJob* refine_predict_jobs = nullptr;
  RefineResult* refine_results = nullptr;
  std::uint8_t* masks = nullptr;
  std::uint8_t* wedge_cache = nullptr;

  int max_jobs = 0;
  int max_mask_samples = 0;
  std::uint64_t required_bytes = 0;
  // Arena watermark after the persistent allocations; scratch users rewind to
  // this value. Launchers themselves allocate nothing.
  std::uint64_t persistent_mark = 0;
};

// Allocates the persistent set once. Never cudaMalloc()s directly.
//
// Returns:
//   kOk              workspace ready; *out_required_bytes = required_device_bytes
//   kInvalidArgument bad config / null output
//   kNotInitialized  arena not initialised
//   kArenaExhausted  arena too small; *out_required_bytes tells the caller how
//                    much to reserve (no allocation is leaked: the arena is
//                    rewound)
inline Status workspace_create(Arena& arena, const CompoundFullConfig& config,
                               CompoundFullWorkspace* out,
                               std::uint64_t* out_required_bytes = nullptr,
                               std::string* error = nullptr) {
  const std::uint64_t required = required_device_bytes(config);
  if (out_required_bytes != nullptr) {
    *out_required_bytes = required;
  }
  if (out == nullptr || required == 0) {
    if (error != nullptr) {
      *error = "compound_full: invalid workspace configuration";
    }
    return Status::kInvalidArgument;
  }
  *out = CompoundFullWorkspace{};
  if (!arena.initialized()) {
    if (error != nullptr) {
      *error = "compound_full: arena not initialized";
    }
    return Status::kNotInitialized;
  }
  if (arena.remaining() < required) {
    if (error != nullptr) {
      *error = "compound_full: arena exhausted: need " +
               std::to_string(required) + " bytes, have " +
               std::to_string(arena.remaining()) + " available";
    }
    return Status::kArenaExhausted;
  }

  const std::uint64_t start_mark = arena.mark();
  const std::uint64_t jobs = static_cast<std::uint64_t>(config.max_jobs);
  std::uint8_t* masks = nullptr;
  std::uint8_t* cache = nullptr;
  struct Request {
    std::size_t bytes;
    void** target;
  };
  const Request requests[] = {
      {static_cast<std::size_t>(sizeof(BlendJob) * jobs),
       reinterpret_cast<void**>(&out->jobs)},
      {static_cast<std::size_t>(sizeof(SkipJob) * jobs),
       reinterpret_cast<void**>(&out->skip_jobs)},
      {static_cast<std::size_t>(sizeof(OpflJob) * jobs),
       reinterpret_cast<void**>(&out->opfl_jobs)},
      {static_cast<std::size_t>(sizeof(RefineSearchJob) * jobs),
       reinterpret_cast<void**>(&out->refine_search_jobs)},
      {static_cast<std::size_t>(sizeof(RefinePredictJob) * jobs),
       reinterpret_cast<void**>(&out->refine_predict_jobs)},
      {static_cast<std::size_t>(sizeof(RefineResult) * jobs),
       reinterpret_cast<void**>(&out->refine_results)},
      {static_cast<std::size_t>(config.max_mask_samples) *
           static_cast<std::size_t>(config.max_jobs),
       reinterpret_cast<void**>(&masks)},
  };
  for (const Request& request : requests) {
    const Status status = arena.allocate_checked(request.bytes, request.target);
    if (status != Status::kOk) {
      arena.rewind(start_mark);
      if (error != nullptr) {
        *error = "compound_full: arena allocation failed: need " +
                 std::to_string(required) + " bytes";
      }
      return status;
    }
  }
  if (config.wedge_cache) {
    const Status status = arena.allocate_checked(
        static_cast<std::size_t>(kWedgeCacheDeviceBytes),
        reinterpret_cast<void**>(&cache));
    if (status != Status::kOk) {
      arena.rewind(start_mark);
      if (error != nullptr) {
        *error = "compound_full: wedge-cache allocation failed: need " +
                 std::to_string(required) + " bytes";
      }
      return status;
    }
  }

  out->masks = masks;
  out->wedge_cache = cache;
  out->max_jobs = config.max_jobs;
  out->max_mask_samples = config.max_mask_samples;
  out->required_bytes = required;
  out->persistent_mark = arena.mark();
  if (error != nullptr) {
    error->clear();
  }
  return Status::kOk;
}

// Fills `workspace.wedge_cache` on the device once. No-op when the workspace
// has no cache. Asynchronous.
cudaError_t wedge_cache_build(CompoundFullWorkspace& workspace,
                              cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Launchers (asynchronous, no allocation, no synchronization)
// ---------------------------------------------------------------------------
//
// `d_jobs` must be device memory (workspace.jobs and friends). The `_ws` forms
// copy `host_jobs` (count <= workspace.max_jobs) into the workspace descriptor
// array with cudaMemcpyAsync and then launch; they are conveniences, still
// allocation-free.

cudaError_t blend_batch(const BlendJob* d_jobs, int job_count,
                        cudaStream_t stream = nullptr);
cudaError_t blend_batch_ws(CompoundFullWorkspace& workspace,
                           const BlendJob* host_jobs, int job_count,
                           cudaStream_t stream = nullptr);

cudaError_t skip_batch(const SkipJob* d_jobs, int job_count,
                       cudaStream_t stream = nullptr);
cudaError_t skip_batch_ws(CompoundFullWorkspace& workspace,
                          const SkipJob* host_jobs, int job_count,
                          cudaStream_t stream = nullptr);

cudaError_t opfl_batch(OpflJob* d_jobs, int job_count,
                       cudaStream_t stream = nullptr);
cudaError_t opfl_batch_ws(CompoundFullWorkspace& workspace,
                          const OpflJob* host_jobs, int job_count,
                          cudaStream_t stream = nullptr);

cudaError_t refine_search_batch(const RefineSearchJob* d_jobs,
                                RefineResult* d_results, int job_count,
                                cudaStream_t stream = nullptr);
cudaError_t refine_search_batch_ws(CompoundFullWorkspace& workspace,
                                   const RefineSearchJob* host_jobs,
                                   int job_count,
                                   cudaStream_t stream = nullptr);

cudaError_t refine_predict_batch(const RefinePredictJob* d_jobs,
                                 int job_count, cudaStream_t stream = nullptr);
cudaError_t refine_predict_batch_ws(CompoundFullWorkspace& workspace,
                                    const RefinePredictJob* host_jobs,
                                    int job_count,
                                    cudaStream_t stream = nullptr);

// Named host-side validity checks. The batch forms trust device descriptors;
// these helpers let a caller or a test reject a malformed job before upload.
bool blend_job_valid(const BlendJob& job);
bool skip_job_valid(const SkipJob& job);
bool opfl_job_valid(const OpflJob& job);
bool refine_search_job_valid(const RefineSearchJob& job);
bool refine_predict_job_valid(const RefinePredictJob& job);

}  // namespace av2enc::cuda::compound_full

#endif  // AV2ENC_CUDA_COMPOUND_FULL_COMPOUND_FULL_H_
