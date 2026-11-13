// Batched device TX-selection kernel for the M2 video residual path.
//
// The M2 coder currently codes every leaf with one DCT_DCT transform sized to
// the whole leaf.  This module is the missing decision stage: for a batch of
// residual blocks it evaluates the TX sizes / TX types / TX partitions that
// AV2 permits, runs the real forward -> quantise -> dequantise -> inverse path
// for every candidate, scores each with distortion (coded-domain SSE) and an
// estimated rate, and returns the RD-best plan with scan-order levels.
//
// Numeric reference (the product CPU path, not a stand-in):
//   * residual -> adjusted residual: `txv::downsample_residual` (§7.15.4)
//   * forward 2D transform:          `txv::forward_block`  (binary64)
//   * quantise:                      `TileEncoder`'s per-coefficient delta-Q
//                                    expression (`lround(coeff * qf*528/q)`),
//                                    the same order `still_tx.cu` uses
//   * dequantise:                    `tile_encoder`'s §7.14.4 form
//                                    (`Round2((|q|*q2)&0xFFFFFF, 3)/dqDenom`)
//   * inverse 2D transform:          `txv::inverse_block`   (int64, §7.15.4)
//   * §7.14.2 quantisers:            `dcdeltaq::get_q`
//   * scan orders:                   `txpartition::get_scan` (§5.20.7.30,
//                                    TX_CLASS_2D/HORIZ/VERT)
//
// The device kernel evaluates the whole batch in ONE launch (one CUDA block per
// input block) and the host entry point synchronises once.  It does not
// allocate, and every device byte is allocated once from the caller's single
// `av2enc::cuda::Arena` by `VideoTxWorkspace::init()`.
//
// Ownership: only files under `src/cuda/pipeline/video/tx/` are part of this
// module.  It reuses the shared CPU tables/quantisers and the `tx_variants`
// device tables; it does not modify any other component and is not wired into
// the product path by this module.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/transform/variants/tx_variants.h"
#include "cuda/arena.h"
#include "cuda/error.h"

namespace av2enc::cuda::videotx {

using av2enc::txv::TxParams;

// ---------------------------------------------------------------------------
// Caps.  They mirror the AV2 syntax limits, not tuning knobs.
// ---------------------------------------------------------------------------

// Largest adjusted transform area (32x32).
inline constexpr int kMaxAdjDim = av2enc::txv::kMaxAdjTxDim;
inline constexpr int kMaxAdjElems = av2enc::txv::kMaxAdjTxElems;
// Largest coded transform dimension / area (64x64).
inline constexpr int kMaxCodedDim = av2enc::txv::kMaxCodedTxDim;
inline constexpr int kMaxCodedElems = kMaxCodedDim * kMaxCodedDim;
// §5.20.6.3 legal_splits() returns at most 8 options; every block <= 64x64 has
// at most 4 (NONE + SPLIT/HORZ/VERT or NONE + HORZ + HORZ4).
inline constexpr int kMaxPlansPerJob = 8;
inline constexpr int kMaxLeavesPerPlan = 4;
// Largest canonical transform-type set: TX_SET_INTER_1 has 16 types.
inline constexpr int kMaxCandidatesPerLeaf = 16;
// 25 TX sizes x 3 TX classes, each scan table at the 32x32 adjusted area.
inline constexpr int kScanTables = 25 * 3;
inline constexpr int kScanStride = kMaxAdjElems;

// ---------------------------------------------------------------------------
// Input description
// ---------------------------------------------------------------------------

enum class VideoTxMode : std::int32_t {
  kIntra = 0,  // texture luma: §5.20.8.3 INTRA_1/INTRA_2/long intra sets
  kInter = 1,  // inter luma: §5.20.8.3 inter sets
};

// Which candidate dimensions the search may explore.  The defaults enable the
// whole §5.20.6 / §5.20.8 search; a caller can pin a subset (e.g. the M2
// baseline is `search_tx_type = 0, search_partition = 0`).
struct VideoTxSearchOptions {
  // Evaluate the block's own TX size (TX_PARTITION_NONE).  Turning this off is
  // only useful when the caller supplies partition candidates explicitly.
  std::int32_t search_none = 1;
  // Evaluate §5.20.6.3 partitions (SPLIT/HORZ/VERT/HORZ4/VERT4).
  std::int32_t search_partition = 1;
  // Evaluate the legal TX_TYPE set of every leaf instead of DCT_DCT only.
  std::int32_t search_tx_type = 1;
  // Include the 1-D TX classes (V_*/H_*, TX_CLASS_VERT/HORIZ).  They are legal
  // AV2 candidates and this module produces their scan-order levels, but the
  // current M2 syntax writer only codes TX_CLASS_2D, so the product path should
  // leave this at 0 until that writer lands.
  std::int32_t include_1d_types = 0;
  // Bit mask of VideoTxPartition bits allowed in the search.  Default: all.
  std::int32_t partition_mask = 0x3f;
  // Frame-header §5.20.2 / §5.18.8.1 flags.
  std::int32_t reduced_tx_set = 0;
  std::int32_t reduced_tx_part_set = 0;
  // §7.15.4.1: with enable_inter_ddt the decoder swaps ADST/FDST 1-D kernels
  // for DDTX/FDDT.  This module has no DDTX/FDDT inverse, so those candidates
  // are withheld (same conservative guard as the CPU inter component).
  std::int32_t enable_inter_ddt = 0;
  // §5.20.7.29 enable_chroma_dctonly: chroma never signals a TX type.
  std::int32_t dct_only = 0;
  // §5.20.7.29 wide_angle_mapping() output for the intra luma mode.
  std::int32_t intra_mapped_mode = 0;
  // RD weights: cost = sse + lambda * rate_bits.
  double lambda = 0.0;
  double coefficient_bits = 1.0;
  std::int32_t use_type_bits = 1;
  // Rate charged for a candidate whose levels are all zero (txb_skip).
  double skip_bits = 1.2;
  // Upper bound on a candidate leaf's long side (64 keeps all §5.20.6 sizes).
  std::int32_t max_tx_size = kMaxCodedDim;
};

// Partition bit indices for VideoTxSearchOptions::partition_mask.
enum VideoTxPartitionBit : std::int32_t {
  kPartBitNone = 1 << 0,
  kPartBitSplit = 1 << 1,
  kPartBitHorz = 1 << 2,
  kPartBitVert = 1 << 3,
  kPartBitHorz4 = 1 << 4,
  kPartBitVert4 = 1 << 5,
};

// One residual block (one plane) and the quantiser state resolved by the
// caller.  `delta_dc` / `delta_ac` are the *resolved* §7.14.2 deltas for this
// plane (sequence base + frame): Y AC is 0, U/V AC use their own totals.
struct VideoTxJobRequest {
  std::int32_t plane = 0;      // 0 Y, 1 U, 2 V
  VideoTxMode mode = VideoTxMode::kInter;
  std::int32_t coded_w = 0;    // 4..64, one of the 25 AV2 TX sizes
  std::int32_t coded_h = 0;
  std::int32_t qindex = 90;
  std::int32_t delta_dc = 0;
  std::int32_t delta_ac = 0;

  // Residual source.  `residual_offset >= 0` selects a precomputed coded-size
  // int16 residual (`residual` buffer, stride = coded_w, `residual_offset`
  // points at this block).  Otherwise the kernel builds src-pred.
  std::int32_t residual_offset = -1;
  // 1: `src` holds a packed coded-size block at `src_offset` (stride
  // `src_stride`, 0 == coded_w).  0: `src` is a plane; coordinates are
  // `src_x`/`src_y` + the leaf origin, clamped to `src_visible_w`/h.
  std::int32_t src_packed = 1;
  std::int32_t src_offset = 0;
  std::int32_t src_stride = 0;
  std::int32_t src_x = 0;
  std::int32_t src_y = 0;
  std::int32_t src_visible_w = 0;
  std::int32_t src_visible_h = 0;
  // Prediction is always a packed coded-size block of this job (stride
  // `pred_stride`, 0 == coded_w).
  std::int32_t pred_offset = 0;
  std::int32_t pred_stride = 0;

  VideoTxSearchOptions search{};
};

// ---------------------------------------------------------------------------
// Plan PODs (host-built, uploaded verbatim)
// ---------------------------------------------------------------------------

struct VideoTxJobDesc {
  std::int32_t plane = 0;
  std::int32_t coded_w = 0;
  std::int32_t coded_h = 0;
  std::int32_t qindex = 90;
  std::int32_t delta_dc = 0;
  std::int32_t delta_ac = 0;

  std::int32_t use_residual = 0;
  std::int32_t src_packed = 1;
  std::int32_t src_offset = 0;
  std::int32_t src_stride = 0;
  std::int32_t src_x = 0;
  std::int32_t src_y = 0;
  std::int32_t src_visible_w = 0;
  std::int32_t src_visible_h = 0;
  std::int32_t pred_offset = 0;
  std::int32_t pred_stride = 0;
  std::int32_t residual_offset = 0;

  // Ranges into the plan arrays.
  std::int32_t plan_begin = 0;
  std::int32_t plan_count = 0;
  // Output slots: leaf results at job*max_leaves, levels at
  // (job*max_leaves + leaf_position)*kMaxAdjElems.
  std::int32_t level_base = 0;

  // Codability / cost inputs.
  std::int32_t is_inter = 1;
  std::int32_t use_type_bits = 1;
  double lambda = 0.0;
  double coefficient_bits = 1.0;
  double skip_bits = 1.2;
};

struct VideoTxPlanDesc {
  std::int32_t leaf_begin = 0;
  std::int32_t leaf_count = 0;
  std::int32_t partition = 0;  // av2enc::video::txpartition::TxPartition
  std::int32_t reserved = 0;
  // Estimated signalling cost of this partition decision, in bits (from the
  // §9.3 default tx_do_partition / tx_partition_type CDFs).  Added once to the
  // plan cost.
  double partition_bits = 0.0;
};

struct VideoTxLeafDesc {
  std::int32_t x = 0;  // coded-sample origin inside the job block
  std::int32_t y = 0;
  std::int32_t coded_w = 0;
  std::int32_t coded_h = 0;
  std::int32_t cand_begin = 0;
  std::int32_t cand_count = 0;
  // Absolute element offset of this leaf's scan-order level output.
  std::int32_t level_offset = 0;
  std::int32_t leaf_position = 0;  // 0..kMaxLeavesPerPlan-1
};

struct VideoTxCandDesc {
  TxParams params{};          // coded/adjusted size, type, host-computed scales
  std::int32_t tx_size = 0;   // TxSize index 0..24
  std::int32_t tx_type = 0;   // TX_TYPE 0..15
  std::int32_t scan_table = 0;  // tx_size*3 + TX class
  std::int32_t codes_type = 1;  // 0 for TX_SET_DCTONLY (no type bits)
  // Static type-symbol cost per §8.3.2 context (0..2; intra uses slot 0).
  double type_bits[3] = {0.0, 0.0, 0.0};
};

// ---------------------------------------------------------------------------
// Outputs (workspace-owned device arrays)
// ---------------------------------------------------------------------------

struct VideoTxCandMetrics {
  double sse = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  std::int32_t nonzero = 0;
  std::int32_t eob = 0;
  std::int32_t codable = 0;
  std::int32_t valid = 0;
};

struct VideoTxLeafResult {
  std::int32_t tx_size = -1;
  std::int32_t tx_type = -1;
  std::int32_t eob = 0;
  std::int32_t nonzero = 0;
  std::int32_t level_offset = 0;
  std::int32_t level_elems = 0;
  std::int32_t reserved = 0;
  std::int32_t valid = 0;
  double sse = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
};

struct VideoTxPlanResult {
  std::int32_t partition = 0;
  std::int32_t leaf_result_begin = 0;
  std::int32_t leaf_count = 0;
  std::int32_t valid = 0;
  double sse = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
};

struct VideoTxJobResult {
  std::int32_t plan_index = -1;  // absolute plan index, -1 when no plan won
  std::int32_t partition = 0;
  std::int32_t leaf_result_begin = 0;
  std::int32_t leaf_count = 0;
  std::int32_t status = 0;  // 0 ok, -1 invalid descriptor / no codable plan
  std::int32_t reserved = 0;
  double sse = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
};

// ---------------------------------------------------------------------------
// Host plan
// ---------------------------------------------------------------------------

struct VideoTxPlanData {
  std::vector<VideoTxJobDesc> jobs;
  std::vector<VideoTxPlanDesc> plans;
  std::vector<VideoTxLeafDesc> leaves;
  std::vector<VideoTxCandDesc> candidates;

  std::size_t candidate_count() const { return candidates.size(); }
  std::size_t plan_count() const { return plans.size(); }
  std::size_t leaf_count() const { return leaves.size(); }
  void clear() {
    jobs.clear();
    plans.clear();
    leaves.clear();
    candidates.clear();
  }
};

// Builds the candidate plan for `count` blocks.
//
// Stage-candidate rules:
//   * `search_none`:      one TX_PARTITION_NONE plan whose TX size is the block
//                         size (capped at 64 in each dimension).
//   * `search_partition`: every §5.20.6.3 legal partition of the block, in
//                         legal_splits() order, filtered by partition_mask.
//   * `search_tx_type`:   the §5.20.8.3 legal set of each leaf (intra via the
//                         intra sets, inter via the §5.20.8.2 inter inverse
//                         tables), with the §7.15.4.1 DDT guard.
//
// Returns false and fills `*error` on an invalid request, an unsupported size,
// or a plan that would exceed the module caps.
bool video_tx_build_plan(const VideoTxJobRequest* requests, int count,
                         VideoTxPlanData* plan, std::string* error);

// Candidate types for one size/mode (exposed for the orchestrator and tests).
// `out` must have room for kMaxCandidatesPerLeaf entries.  Returns the count,
// or -1 for an invalid size.  `type_bits[i][0..2]` receives the estimated
// symbol cost per §8.3.2 context.
int video_tx_tx_types(std::int32_t coded_w, std::int32_t coded_h,
                      VideoTxMode mode, const VideoTxSearchOptions& options,
                      std::int32_t* out, double type_bits[][3], int cap);

// ---------------------------------------------------------------------------
// Arena workspace
// ---------------------------------------------------------------------------

// Worst-case batch shape.  `required_device_bytes(config)` sizes every array
// for exactly this shape; the plan builder checks a concrete plan against it.
struct VideoTxConfig {
  int max_jobs = 1;
  int max_plans_per_job = kMaxPlansPerJob;
  int max_candidates_per_leaf = kMaxCandidatesPerLeaf;
};

// Exact arena requirement: persistent scan tables + all descriptor and output
// arrays, rounded to Arena::kAlignment (256 B) the same way the bump allocator
// rounds.
std::uint64_t required_device_bytes(const VideoTxConfig& config);
std::uint64_t persistent_device_bytes(const VideoTxConfig& config);
std::uint64_t scratch_device_bytes(const VideoTxConfig& config);

// One arena allocation per module.  All device buffers are allocated
// `allocate_checked()` once in init(); a short arena returns
// `Status::kArenaExhausted` with `*required_bytes == required_device_bytes()`
// and leaves the arena's bump pointer unchanged.
struct VideoTxWorkspace {
  VideoTxConfig config{};
  std::uint64_t bytes = 0;
  int last_error_count = 0;

  // Persistent device tables.
  std::uint16_t* d_scan = nullptr;  // [kScanTables][kScanStride]

  // Uploaded plan descriptors.
  VideoTxJobDesc* d_jobs = nullptr;
  VideoTxPlanDesc* d_plans = nullptr;
  VideoTxLeafDesc* d_leaves = nullptr;
  VideoTxCandDesc* d_candidates = nullptr;

  // Outputs.
  VideoTxCandMetrics* d_cand_metrics = nullptr;
  VideoTxPlanResult* d_plan_results = nullptr;
  VideoTxLeafResult* d_leaf_results = nullptr;  // max_jobs * max_leaves
  VideoTxJobResult* d_job_results = nullptr;
  std::int32_t* d_levels = nullptr;  // max_jobs * max_leaves * kMaxAdjElems
  int* d_error_count = nullptr;

  Status init(Arena& arena, const VideoTxConfig& config_in,
              std::uint64_t* required_bytes = nullptr,
              std::string* error = nullptr);
  void reset() noexcept;
  bool valid() const noexcept;
};

// Caller-owned input buffers.  Unused pointers stay null.
struct VideoTxBuffers {
  const std::uint8_t* src = nullptr;
  const std::uint8_t* pred = nullptr;
  const std::int16_t* residual = nullptr;
};

// Runs the whole plan in one kernel launch and then synchronises the stream
// exactly once.  Descriptors are copied to the workspace's device arrays; the
// caller reads the results from the workspace after the call.
//
// Returns kInvalidArgument when the plan does not fit the workspace config or
// is malformed, kArenaExhausted (with the workspace already reporting the
// required size) never occurs here, and kCudaError for a launch failure.
Status video_tx_select_batch(const VideoTxPlanData& plan,
                             const VideoTxBuffers& buffers,
                             VideoTxWorkspace& workspace,
                             cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::videotx
