// M2 video intra candidate generator (device, batched).
//
// The M2 video coder currently hard-codes intra leaves to DC_PRED
// (`entry.y_mode = 0` in `video_pipeline.cpp`).  The still path already has
// every intra mode on the device (`src/cuda/codec/intra/modes/`).  This module
// is the batched candidate generator the video orchestrator can consume: one
// device batch produces, for every leaf and every legal candidate, the
// predictor plane, its SSE against the source, and (optionally) the quantised
// TX plan produced by the still `still_tx` kernels.
//
// Normative sources
// -----------------
//   §5.20.9.1  is_inside()                       neighbour availability
//   §6.19.2.3  clear_block_decoded_flags()       above-right/below-left bounds
//   §7.13.2.1  AboveRow / LeftCol preparation    (via the shared predictor)
//   §7.13.2.2  basic (Paeth)                     (via the shared predictor)
//   §7.13.2.7  directional + angle deltas + WAIP (via the shared predictor)
//   §7.13.2.8  single directional (luma IDIF, chroma bilinear)
//   §7.13.2.10 DC                                (via the shared predictor)
//   §7.13.2.13 smooth / smooth-v / smooth-h
//   §7.13.2.14-.18 filter corner / edge filter
//
// No predictor arithmetic lives here.  Every candidate is described by the
// existing `intra_modes::IntraBlock` and predicted by the shared device body
// `intra_predict_device.cuh`; the module only expands candidates, batches
// them, and reduces SSE.  That is the same header `intra_modes.cu` compiles,
// so there is no second transcription that can drift.
//
// Scope (M2 wave 1)
// -----------------
//   * Luma: all 13 base modes plus the 8 directional modes x 6 non-zero angle
//     deltas = 61 candidates, for 4x4/8x8/16x16/32x32/64x64 (and rectangular
//     transform sizes) leaves.
//   * Chroma: DC_PRED only, one job per plane.  The shared predictor applies
//     the chroma derivation (no IDIF, bilinear interpolation, §7.13.2.1
//     chroma > 32 extension rule); the optional TX plan uses the shared
//     §7.13.5/§5.20.7.29 chroma TX-type derivation from
//     `intra_full::Workspace::derive_chroma_type` when the orchestrator needs a
//     non-DCT type (default DCT_DCT, the M2 baseline).
//   * Availability (§5.20.9.1) is derived from tile/frame geometry by
//     `derive_availability()`; the block origin may extend past the visible
//     plane (odd partial row/column) and the predictor clamps all reads.
//
// Execution model
// ---------------
//   `Workspace::evaluate()` stages the host job array into one persistent
//   arena staging region and launches ONE candidate kernel for the whole
//   batch.  The optional TX plan reuses `av2enc::cuda::still_tx_batch` as a
//   second launch on the same stream.  Host-visible results are read with
//   `download_results()` / `download_tx_results()` after ONE `synchronize()`.
//
//   Jobs must be mutually independent: no job may read reconstruction that
//   another job in the same call writes.  For a raster-coded tile that means
//   "batch blocks whose left/above reconstruction is final" (a wavefront level,
//   or the whole frame for a frozen reference plane).  The predictor itself
//   never writes the reconstruction plane.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/codec/intra/modes/intra_modes.h"
#include "cuda/error.h"
#include "cuda/pipeline/still/tx/still_tx.h"

namespace av2enc::cuda::video::intra {

// ---------------------------------------------------------------------------
// Candidate set
// ---------------------------------------------------------------------------

inline constexpr int kMaxBlockDim = intra_modes::kMaxBlockDim;  // 64
inline constexpr int kNumBaseModes = intra_modes::kIntraModes;  // 13
inline constexpr int kNumDirectionalModes = 8;
inline constexpr int kMaxAngleDelta = intra_modes::kMaxAngleDelta;  // 3
inline constexpr int kMaxCandidates = 61;  // 13 + 8 * 6
inline constexpr int kMaxThreads = 128;    // one CUDA block per candidate

// Every base mode enabled.  Chroma ignores the mask (DC is the only candidate).
inline constexpr std::uint32_t kAllModesMask =
    (1u << static_cast<unsigned>(kNumBaseModes)) - 1u;
inline constexpr std::uint32_t kDcOnlyMask = 1u << 0;
inline constexpr std::uint32_t kNoDirectionalMask =
    (1u << 0) | (1u << 9) | (1u << 10) | (1u << 11) | (1u << 12);

// One (mode, angle_delta) pair.  Values match `intra_modes::IntraMode` /
// AV2 YMode/UVMode.
struct Candidate {
  std::uint8_t mode = 0;
  std::int8_t angle_delta = 0;
  std::uint8_t reserved[2] = {0, 0};
};

// The 61-candidate luma list in CPU/still search order: the 13 base modes
// first (angle_delta 0), then V..D67 with deltas -3, -2, -1, +1, +2, +3.
const Candidate* luma_candidates() noexcept;
int luma_candidate_count() noexcept;

// Fills `out` with the candidate list for one plane and returns the number
// written, or -1 when `out` is too small.
//
//   plane == 0 : every mode selected by `mode_mask`, in base-then-delta order.
//                `include_angle_deltas == 0` drops the 48 delta candidates.
//   plane  > 0 : exactly one candidate (DC_PRED).  `mode_mask` and
//                `include_angle_deltas` are ignored: chroma directional search
//                is not wired in the M2 product (docs/negative-results.md).
int build_candidates(int plane, std::uint32_t mode_mask,
                     int include_angle_deltas, Candidate* out,
                     int capacity) noexcept;

// ---------------------------------------------------------------------------
// §5.20.9.1 availability
// ---------------------------------------------------------------------------

// Geometry for one plane.  Tile bounds are inclusive- origin / exclusive-end in
// plane samples.  For the M2 backend the reconstruction planes are tile-local,
// so `tile_x0 = tile_y0 = 0` and `tile_x1/tile_y1` equal the plane extent; the
// tile form is provided for a future multi-tile backend and for tests.
struct AvailRequest {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int tile_x0 = 0;
  int tile_y0 = 0;
  int tile_x1 = 0;
  int tile_y1 = 0;
  int plane_w = 0;
  int plane_h = 0;
  // Coded-flag counts in 4x4 units from the partition/BlockDecoded map
  // (§6.19.2.3).  -1 means "derive from geometry": the geometric bound is
  // used for above-right and 0 for below-left (raster coding order).
  int decoded_above_right4 = -1;
  int decoded_below_left4 = -1;
};

struct Availability {
  int max_x = 0;  // last visible column in the plane (inclusive)
  int max_y = 0;  // last visible row in the plane (inclusive)
  int have_left = 0;
  int have_above = 0;
  int num4_above_right = 0;
  int num4_below_left = 0;
};

// §5.20.9.1 is_inside() plus the §7.13.2.1 extension bounds.  `have_left` and
// `have_above` are relative to the tile origin, so a block at the tile start
// has no left/above neighbour even when the frame continues.  The block may
// extend past `plane_w`/`plane_h` (partial right/bottom leaves); `max_x` and
// `max_y` still clamp to the visible plane.
Availability derive_availability(const AvailRequest& request) noexcept;

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

// One block of one plane.  All pointers are device pointers.  The availability
// fields are normally filled from `derive_availability()`.
struct BlockJob {
  const std::uint8_t* recon_plane = nullptr;
  int recon_stride = 0;
  const std::uint8_t* source_plane = nullptr;
  int source_stride = 0;

  int x = 0;
  int y = 0;
  int w = 0;  // 4, 8, 16, 32 or 64
  int h = 0;
  int plane = 0;  // 0 = luma, 1 = U, 2 = V

  int max_x = 0;  // last visible reconstructed column, inclusive
  int max_y = 0;
  int source_max_x = 0;  // last visible source column, inclusive
  int source_max_y = 0;

  int have_left = 0;
  int have_above = 0;
  int num4_above_right = 0;
  int num4_below_left = 0;

  int filter_type_above = 0;  // §7.13.2.15: neighbour above is SMOOTH*
  int filter_type_left = 0;   // §7.13.2.16: neighbour left is SMOOTH*
  int enable_edge_filter = 1;

  std::uint32_t mode_mask = kAllModesMask;
  std::int32_t include_angle_deltas = 1;
  std::int32_t reserved0 = 0;
};

// Per-candidate output.  `pred_offset` is the element offset of the packed
// w*h predictor plane in the workspace's prediction buffer.
struct CandidateResult {
  std::int64_t sse = 0;
  std::int32_t pred_offset = 0;
  std::int32_t block = 0;
  std::uint8_t mode = 0;
  std::int8_t angle_delta = 0;
  std::uint8_t plane = 0;
  std::uint8_t reserved0 = 0;
  std::int32_t reserved1 = 0;
};

// Per-candidate TX metadata (written by the optional TX stage).
struct TxCandidateResult {
  std::int32_t eob = 0;
  std::int32_t levels_offset = -1;
};

// Host-side best candidate for one block, using the CPU tie-break (candidate
// order, replace only on a strictly smaller SSE).
struct BestResult {
  std::int32_t block = 0;
  std::uint8_t mode = 0;
  std::int8_t angle_delta = 0;
  std::uint8_t plane = 0;
  std::uint8_t reserved0 = 0;
  std::int32_t candidate_index = 0;
  std::int64_t sse = 0;
};

struct EvaluateOptions {
  // Run the shared still TX kernels for every candidate.  `BlockJob` has no
  // predictor/residual buffer, so the candidate kernel writes the coded-size
  // residual (`source - pred`) when this is set.
  bool run_tx = false;
  // TX_TYPE for luma / chroma.  The M2 baseline and the only element the
  // video syntax writer can currently signal is DCT_DCT.
  av2enc::txv::TxType luma_tx_type = av2enc::txv::TxType::kDctDct;
  av2enc::txv::TxType chroma_tx_type = av2enc::txv::TxType::kDctDct;
  // Largest adjusted transform area in the batch (0 = derive from jobs).
  int tx_smem_elems = 0;
};

// ---------------------------------------------------------------------------
// Arena workspace
// ---------------------------------------------------------------------------

// `max_blocks` block jobs per batch, each with at most `max_candidates`
// candidates.  `max_block_area` is the largest coded w*h per candidate (4096
// for 64x64).  TX buffers are only reserved when `enable_tx` is set.
struct Config {
  int max_blocks = 64;
  int max_candidates = kMaxCandidates;
  int max_block_area = 64 * 64;
  bool enable_tx = false;
  int max_tx_area = 32 * 32;  // largest adjusted area per TX job
};

std::uint64_t required_device_bytes(const Config& config) noexcept;

class Workspace {
 public:
  Workspace() = default;

  // Allocates every persistent buffer from `arena`.  On failure the arena bump
  // pointer is restored, `*required_out` receives required_device_bytes(), and
  // `Status::kArenaExhausted` is returned when the arena is merely too small.
  Status init(Arena& arena, const Config& config = {},
              std::string* error = nullptr,
              std::uint64_t* required_out = nullptr);

  bool initialized() const noexcept { return arena_ != nullptr; }
  const Config& config() const noexcept { return config_; }

  // Stages `jobs[0 .. count)` and launches the whole candidate batch on
  // `stream`.  Returns kInvalidArgument (with `error`) for a malformed job,
  // kArenaExhausted when the batch exceeds the configured capacity, and
  // kNotInitialized before init().  No hidden synchronisation.
  Status evaluate(const BlockJob* jobs, int count,
                  const EvaluateOptions& options = {},
                  cudaStream_t stream = nullptr, std::string* error = nullptr);

  // One synchronize for the whole batch.  Call before downloading.
  Status synchronize(cudaStream_t stream = nullptr);

  // Host copies; each call completes before returning (a blocking transfer,
  // not the batch synchronisation).  `out` must hold at least
  // `candidate_total()` / `prediction_elems()` elements.
  Status download_results(CandidateResult* out, int capacity,
                          cudaStream_t stream = nullptr) const;
  Status download_prediction(std::uint8_t* out, std::size_t capacity,
                             cudaStream_t stream = nullptr) const;
  Status download_tx(TxCandidateResult* out, int capacity,
                     cudaStream_t stream = nullptr);
  // Host copy of the whole coded-size residual buffer written by the
  // candidate kernel when TX ran (element offset = CandidateResult.pred_offset).
  Status download_residual(std::int16_t* out, std::size_t capacity,
                           cudaStream_t stream = nullptr) const;

  // Downloads the results and applies the CPU tie-break per block; this
  // convenience performs the batch's single synchronize itself.  `out` must
  // hold at least `block_count()` entries.
  Status download_best(BestResult* out, int capacity,
                       cudaStream_t stream = nullptr);

  // Batch shape of the last evaluate() call.
  int block_count() const noexcept { return block_count_; }
  int candidate_total() const noexcept { return candidate_total_; }
  int candidate_offset(int block_index) const noexcept;
  int candidate_count(int block_index) const noexcept;
  std::size_t prediction_elems() const noexcept { return pred_total_; }
  std::size_t tx_level_elems() const noexcept { return tx_level_total_; }

  // Device views (valid until the arena is re-initialised).
  const CandidateResult* results_device() const noexcept { return d_results_; }
  const std::uint8_t* prediction_device() const noexcept { return d_pred_; }
  const TxCandidateResult* tx_results_device() const noexcept {
    return d_tx_results_;
  }
  const std::int32_t* levels_device() const noexcept { return d_levels_; }
  const std::int16_t* residual_device() const noexcept { return d_residual_; }

 private:
  struct HostPlan {
    std::int32_t candidate_offset = 0;
    std::int32_t candidate_count = 0;
  };

  Status release_and_restore(std::uint64_t mark, std::string* error,
                             const char* what);

  Arena* arena_ = nullptr;
  Config config_{};

  void* staging_ = nullptr;
  std::uint64_t staging_bytes_ = 0;

  CandidateResult* d_results_ = nullptr;
  std::uint8_t* d_pred_ = nullptr;
  TxCandidateResult* d_tx_results_ = nullptr;
  std::int32_t* d_levels_ = nullptr;
  std::int16_t* d_residual_ = nullptr;

  int block_count_ = 0;
  int candidate_total_ = 0;
  std::size_t pred_total_ = 0;
  std::size_t tx_level_total_ = 0;

  std::vector<HostPlan> host_plan_;
  std::vector<CandidateResult> host_results_;
};

}  // namespace av2enc::cuda::video::intra
