#ifndef AV2ENC_CUDA_M1_SEARCH_LUMA_SEARCH_H_
#define AV2ENC_CUDA_M1_SEARCH_LUMA_SEARCH_H_

// M1 luma intra candidate search (GPU, stage 2).
//
// Evaluates the AV2 luma intra candidate list (13 base modes + 8 directional
// modes x 6 non-zero angle deltas = 61 candidates) against the source plane and
// returns the SSE-minimising candidate with **exactly** the CPU tie-break used
// by `m1_still.cpp::choose_luma_mode`: candidates are visited in the same order
// and the best is replaced only on a strictly smaller SSE.
//
// Prediction is delegated to `src/cuda/intra_modes` (every candidate is one
// `intra_modes::IntraJob`). This component adds no predictor arithmetic; it
// only expands candidates into jobs, batches the predictor, reduces SSE, and
// applies the ordered tie-break.
//
// Data flow:
//
//   d_blocks ──expand──> d_jobs (nblocks*candidates_per_chunk)
//                         │
//                         ├─ intra_modes::intra_predict_batch (existing)
//                         │
//                         └─ SSE reduce ──ordered combine──> d_results
//
// Chunking: the candidate list is split into consecutive chunks of
// `candidates_per_chunk` when the caller wants to bound scratch memory. The
// ordered combine carries the running best across chunks, so the final result
// is independent of the chunk size and identical to a single 61-candidate pass.
//
// Legality / wavefront batching: `luma_search()` only *reads* `recon_plane` and
// writes its own pred/SSE/result buffers. Any set of blocks whose
// reconstruction is already final may be batched in one call. Blocks that read
// reconstruction produced later in the same wavefront level must not be batched
// together; the caller groups by dependency level.
//
// Ownership: this file is the component API for `src/cuda/m1/search/`; the
// orchestrator owns `src/cuda/m1/m1_gpu.h` and the arena.
//
// Copyright-free implementation from the AV2 v1.0.0 specification.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "cuda/intra_modes/intra_modes.h"

#if defined(__CUDACC__)
#define AV2ENC_M1_SEARCH_HD __host__ __device__
#else
#define AV2ENC_M1_SEARCH_HD
#endif

namespace av2enc::cuda::m1::search {

// ---------------------------------------------------------------------------
// Candidate set
// ---------------------------------------------------------------------------

inline constexpr int kMaxLumaCandidates = 61;  // 13 + 8 * 6
inline constexpr int kLumaBaseModes = 13;

// Candidate order is normative for the tie-break. It matches
// `av2enc::intra::build_mode_candidates(candidates, true)`:
//   first all 13 base modes with angle_delta 0,
//   then for each directional mode (V..D67) deltas -3,-2,-1,1,2,3.
struct LumaCandidate {
  int mode;         // intra_modes::IntraMode value (0..12)
  int angle_delta;  // -3..3, 0 for non-directional modes
};

#define AV2ENC_M1_LUMA_CANDIDATE_LIST                                        \
  {0, 0},  {1, 0},  {2, 0},  {3, 0},  {4, 0},  {5, 0},  {6, 0},              \
  {7, 0},  {8, 0},  {9, 0},  {10, 0}, {11, 0}, {12, 0},                      \
  {1, -3}, {1, -2}, {1, -1}, {1, 1},  {1, 2},  {1, 3},                       \
  {2, -3}, {2, -2}, {2, -1}, {2, 1},  {2, 2},  {2, 3},                       \
  {3, -3}, {3, -2}, {3, -1}, {3, 1},  {3, 2},  {3, 3},                       \
  {4, -3}, {4, -2}, {4, -1}, {4, 1},  {4, 2},  {4, 3},                       \
  {5, -3}, {5, -2}, {5, -1}, {5, 1},  {5, 2},  {5, 3},                       \
  {6, -3}, {6, -2}, {6, -1}, {6, 1},  {6, 2},  {6, 3},                       \
  {7, -3}, {7, -2}, {7, -1}, {7, 1},  {7, 2},  {7, 3},                       \
  {8, -3}, {8, -2}, {8, -1}, {8, 1},  {8, 2},  {8, 3}

inline constexpr LumaCandidate kLumaCandidates[kMaxLumaCandidates] = {
    AV2ENC_M1_LUMA_CANDIDATE_LIST};

// Gates supported by the CPU path. `kGateDcOnly` mirrors
// `luma_dc_only`/`force_dc`/`AV2ENC_M1_LUMA_DC` and also reports `sse = 0`
// without running any prediction, exactly like `choose_luma_mode`.
// `kGateBaseOnly` mirrors `AV2ENC_M1_BASE_ONLY` (drop non-zero angle deltas).
// `kGateNoDirectional` mirrors `AV2ENC_M1_NO_DIR` (drop directional modes).
enum LumaSearchGate : unsigned {
  kGateNone = 0u,
  kGateDcOnly = 1u << 0,
  kGateBaseOnly = 1u << 1,
  kGateNoDirectional = 1u << 2,
};

struct LumaSearchGates {
  unsigned flags = kGateNone;
  // `enable_intra_edge_filter` from GpuOptions / M1Options. The CPU path also
  // honours AV2ENC_M1_NO_EDGE; the orchestrator resolves that before calling.
  int enable_edge_filter = 1;
};

AV2ENC_M1_SEARCH_HD inline constexpr bool luma_candidate_enabled(
    const LumaCandidate& candidate, unsigned flags) noexcept {
  if ((flags & kGateDcOnly) != 0u) {
    return candidate.mode == intra_modes::kDcPred;
  }
  if ((flags & kGateBaseOnly) != 0u && candidate.angle_delta != 0) {
    return false;
  }
  if ((flags & kGateNoDirectional) != 0u &&
      intra_modes::is_directional_mode(candidate.mode)) {
    return false;
  }
  return true;
}

// Number of candidates the gates leave active (>= 1; DC is always enabled).
int luma_search_candidate_count(const LumaSearchGates& gates) noexcept;

// Writes the active candidates, in CPU search order, to `out`. Returns the
// number written, or -1 if `capacity` is too small. Passing `out = nullptr`
// with `capacity = 0` is valid and just reports the count.
int luma_search_candidates(const LumaSearchGates& gates, LumaCandidate* out,
                           int capacity) noexcept;

// ---------------------------------------------------------------------------
// Per-block input / output
// ---------------------------------------------------------------------------

// Everything the search needs for one block. Geometry follows the CPU
// reference (`prepare_edges`): `max_x`/`max_y` are the last *visible* plane
// coordinates, inclusive; `have_left`/`have_above` are the spec availability
// flags; the `num4_*` counts are the valid 4x4 units above-right / below-left
// (§7.13.2.1), already bounded by the caller's SB/tile rules.
struct LumaSearchBlock {
  const std::uint8_t* recon_plane = nullptr;  // reconstructed plane (input)
  int recon_stride = 0;                       // samples per row

  int x0 = 0;  // block origin in plane samples
  int y0 = 0;
  int width = 0;   // 4, 8, 16, 32, 64
  int height = 0;  // 4, 8, 16, 32, 64

  int max_x = 0;  // last visible column, inclusive
  int max_y = 0;  // last visible row, inclusive

  int have_left = 0;
  int have_above = 0;
  int num4_above_right = 0;
  int num4_below_left = 0;

  // Neighbour-smooth flags (§7.13.2.15/16): 1 if the block above / left uses
  // SMOOTH, SMOOTH_V or SMOOTH_H. Only consulted when the neighbour exists.
  int filter_type_above = 0;
  int filter_type_left = 0;

  const std::uint8_t* source_plane = nullptr;  // source luma (SSE only)
  int source_stride = 0;
  int source_max_x = 0;
  int source_max_y = 0;
};

struct LumaSearchResult {
  int mode = 0;         // intra_modes::IntraMode
  int angle_delta = 0;  // -3..3
  int reserved = 0;
  std::int64_t sse = 0;
};

// Optional diagnostics: per-block SSE for every active candidate, in CPU search
// order (`stride` entries per block). Null disables the writes. When the
// `kGateDcOnly` shortcut runs, no candidate SSE is produced (the CPU path also
// returns before predicting).
struct LumaSearchDebug {
  std::int64_t* d_candidate_sse = nullptr;
  int stride = 0;
};

// ---------------------------------------------------------------------------
// Scratch / arena layout
// ---------------------------------------------------------------------------

// Host-side layout helper. `max_blocks` is the largest batch the caller will
// pass; `max_block_area` the largest width*height in that batch;
// `candidates_per_chunk` in [1, 61]. The orchestrator carves `total_bytes` from
// the arena and calls `luma_search_make_scratch`.
struct LumaSearchLayout {
  std::size_t jobs_offset = 0;
  std::size_t sse_offset = 0;
  std::size_t pred_offset = 0;
  std::size_t total_bytes = 0;

  int job_capacity = 0;        // max_blocks * candidates_per_chunk
  int candidates_per_chunk = 0;
  int max_block_area = 0;
};

LumaSearchLayout luma_search_layout(int max_blocks, int max_block_area,
                                    int candidates_per_chunk) noexcept;

struct LumaSearchScratch {
  intra_modes::IntraJob* d_jobs = nullptr;
  std::int64_t* d_sse = nullptr;
  std::uint8_t* d_pred = nullptr;
  int job_capacity = 0;
  std::size_t pred_capacity = 0;  // bytes
  int max_block_area = 0;
  int candidates_per_chunk = 0;
};

// Maps `base` (arena allocation) into device pointers using `layout`.
LumaSearchScratch luma_search_make_scratch(void* base,
                                           const LumaSearchLayout& layout) noexcept;

// Device bytes required for `max_blocks` blocks at `max_block_area` with the
// given candidate chunk size.
std::size_t luma_search_scratch_bytes(int max_blocks, int max_block_area,
                                      int candidates_per_chunk) noexcept;

// ---------------------------------------------------------------------------
// Search entry point
// ---------------------------------------------------------------------------

// Evaluates the gated candidate list for every block and writes one result per
// block. `d_blocks` and `d_results` are device arrays of `block_count`
// elements. `d_jobs`, `d_sse`, `d_pred` come from the scratch layout and must
// be large enough for `block_count` and the batch's largest block area.
//
// The blocks must be mutually independent: every block's `recon_plane` must be
// final before the call (no block may read reconstruction written by another
// block in the same call). Results are deterministic and independent of
// batching and of `candidates_per_chunk`.
//
// `debug` may be null. `stream` may be null (default stream). No allocation,
// no synchronisation, one launch sequence per candidate chunk.
cudaError_t luma_search(const LumaSearchBlock* d_blocks, int block_count,
                        const LumaSearchGates& gates,
                        const LumaSearchScratch& scratch,
                        LumaSearchResult* d_results,
                        const LumaSearchDebug* debug = nullptr,
                        cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::m1::search

#undef AV2ENC_M1_SEARCH_HD

#endif  // AV2ENC_CUDA_M1_SEARCH_LUMA_SEARCH_H_
