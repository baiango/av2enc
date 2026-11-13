// M1 luma intra candidate search kernels.
//
// Reuses `src/cuda/intra_modes` for every prediction; this file only expands
// candidates into IntraJobs, reduces SSE, and applies the CPU's ordered
// strict-less-than tie-break. See luma_search.h for the contract.

#include "luma_search.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace av2enc::cuda::still::search {
namespace {

using intra_modes::IntraBlock;
using intra_modes::IntraJob;

constexpr int kExpandThreads = 128;
constexpr int kSseThreads = 128;
constexpr int kCombineThreads = 128;

// Same candidate order as the host table; both expand the same macro.
__device__ __constant__ LumaCandidate kDevCandidates[kMaxLumaCandidates] = {
    AV2ENC_STILL_LUMA_CANDIDATE_LIST};

// Finds the `ordinal`-th gated-on candidate. Returns false if out of range.
__device__ __forceinline__ bool ls_active_candidate(unsigned flags, int ordinal,
                                                    LumaCandidate* out) {
  int seen = 0;
  for (int i = 0; i < kMaxLumaCandidates; ++i) {
    const LumaCandidate candidate = kDevCandidates[i];
    if (!luma_candidate_enabled(candidate, flags)) {
      continue;
    }
    if (seen == ordinal) {
      *out = candidate;
      return true;
    }
    ++seen;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Candidate expansion: block x active candidate -> IntraJob
// ---------------------------------------------------------------------------

__global__ void ls_expand_kernel(const LumaSearchBlock* __restrict__ blocks,
                                 int block_count, unsigned flags,
                                 int active_base, int active_count,
                                 int pred_area, int enable_edge_filter,
                                 IntraJob* __restrict__ jobs,
                                 std::uint8_t* __restrict__ pred) {
  const int job_index =
      static_cast<int>(blockIdx.x) * kExpandThreads +
      static_cast<int>(threadIdx.x);
  const int total = block_count * active_count;
  if (job_index >= total) {
    return;
  }
  const int block_index = job_index / active_count;
  const int local = job_index - block_index * active_count;

  LumaCandidate candidate{};
  if (!ls_active_candidate(flags, active_base + local, &candidate)) {
    // Unreachable when the host-side chunk math matches the device table; fall
    // back to DC so a malformed job never carries a garbage mode.
    candidate.mode = intra_modes::kDcPred;
    candidate.angle_delta = 0;
  }

  const LumaSearchBlock src = blocks[block_index];
  IntraBlock block{};
  block.recon_plane = src.recon_plane;
  block.recon_stride = src.recon_stride;
  block.x0 = src.x0;
  block.y0 = src.y0;
  block.width = src.width;
  block.height = src.height;
  block.max_x = src.max_x;
  block.max_y = src.max_y;
  block.have_left = src.have_left;
  block.have_above = src.have_above;
  block.num4_above_right = src.num4_above_right;
  block.num4_below_left = src.num4_below_left;
  block.plane = 0;  // luma
  block.mode = static_cast<intra_modes::IntraMode>(candidate.mode);
  block.angle_delta = candidate.angle_delta;
  block.filter_type_above = src.filter_type_above;
  block.filter_type_left = src.filter_type_left;
  block.enable_edge_filter = enable_edge_filter;
  block.source_plane = src.source_plane;
  block.source_stride = src.source_stride;
  block.source_max_x = src.source_max_x;
  block.source_max_y = src.source_max_y;

  IntraJob job{};
  job.block = block;
  job.pred = pred + static_cast<std::size_t>(job_index) *
                       static_cast<std::size_t>(pred_area);
  job.residual = nullptr;
  jobs[job_index] = job;
}

// ---------------------------------------------------------------------------
// SSE: source - prediction, one CUDA block per job
// ---------------------------------------------------------------------------

__device__ __forceinline__ int ls_clip(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

__global__ void ls_sse_kernel(const IntraJob* __restrict__ jobs,
                              std::int64_t* __restrict__ out) {
  const int job_index = static_cast<int>(blockIdx.x);
  const IntraJob job = jobs[job_index];
  const IntraBlock& block = job.block;

  const int width = block.width;
  const int height = block.height;
  const int n = width * height;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);

  // Each thread takes one contiguous span, so row/column tracking is
  // incremental (no per-pixel division).
  const int per_thread = (n + nthr - 1) / nthr;
  int index = tid * per_thread;
  const int end = min(n, index + per_thread);

  std::int64_t acc = 0;
  if (index < end) {
    int py = index / width;
    int px = index - py * width;
    const std::uint8_t* pred = job.pred;
    for (; index < end; ++index) {
      const int sy = ls_clip(0, block.source_max_y, block.y0 + py);
      const int sx = ls_clip(0, block.source_max_x, block.x0 + px);
      const int src = static_cast<int>(block.source_plane[
          static_cast<std::size_t>(sy) *
              static_cast<std::size_t>(block.source_stride) +
          static_cast<std::size_t>(sx)]);
      const int value = static_cast<int>(pred[index]);
      const int diff = src - value;
      acc += static_cast<std::int64_t>(diff) * diff;
      ++px;
      if (px == width) {
        px = 0;
        ++py;
      }
    }
  }

  // Warp reduction, then one shared slot per warp.
  constexpr int kWarps = kSseThreads / 32;
  __shared__ std::int64_t warp_sum[kWarps];
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  }
  if ((tid & 31) == 0) {
    warp_sum[tid >> 5] = acc;
  }
  __syncthreads();
  if (tid == 0) {
    std::int64_t total = 0;
    const int warps = nthr >> 5;
    for (int w = 0; w < warps; ++w) {
      total += warp_sum[w];
    }
    out[job_index] = total;
  }
}

// ---------------------------------------------------------------------------
// Ordered combine: candidates are visited in CPU order, replace only on <
// ---------------------------------------------------------------------------

__global__ void ls_combine_kernel(int block_count, unsigned flags,
                                  int active_base, int active_count,
                                  const std::int64_t* __restrict__ sse,
                                  LumaSearchResult* __restrict__ results,
                                  std::int64_t* __restrict__ debug_sse,
                                  int debug_stride) {
  const int block_index =
      static_cast<int>(blockIdx.x) * kCombineThreads +
      static_cast<int>(threadIdx.x);
  if (block_index >= block_count) {
    return;
  }

  // The first chunk starts a fresh search; later chunks carry the winner.
  LumaSearchResult best = results[block_index];
  if (active_base == 0) {
    best.mode = intra_modes::kDcPred;
    best.angle_delta = 0;
    best.reserved = 0;
    best.sse = INT64_MAX;
  }

  const std::int64_t* block_sse =
      sse + static_cast<std::size_t>(block_index) *
                static_cast<std::size_t>(active_count);
  for (int k = 0; k < active_count; ++k) {
    const int ordinal = active_base + k;
    const std::int64_t value = block_sse[k];
    if (debug_sse != nullptr) {
      debug_sse[static_cast<std::size_t>(block_index) *
                    static_cast<std::size_t>(debug_stride) +
                static_cast<std::size_t>(ordinal)] = value;
    }
    if (value < best.sse) {
      LumaCandidate candidate{};
      if (ls_active_candidate(flags, ordinal, &candidate)) {
        best.sse = value;
        best.mode = candidate.mode;
        best.angle_delta = candidate.angle_delta;
      }
    }
  }
  results[block_index] = best;
}

// DC-only shortcut: CPU `choose_luma_mode` returns this without predicting.
__global__ void ls_dc_only_kernel(int block_count,
                                  LumaSearchResult* __restrict__ results) {
  const int block_index =
      static_cast<int>(blockIdx.x) * kCombineThreads +
      static_cast<int>(threadIdx.x);
  if (block_index >= block_count) {
    return;
  }
  LumaSearchResult result{};
  result.mode = intra_modes::kDcPred;
  result.angle_delta = 0;
  result.sse = 0;  // matches the CPU shortcut
  results[block_index] = result;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

int clamp_int(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

}  // namespace

// ---------------------------------------------------------------------------
// Candidate list
// ---------------------------------------------------------------------------

int luma_search_candidate_count(const LumaSearchGates& gates) noexcept {
  int count = 0;
  for (int i = 0; i < kMaxLumaCandidates; ++i) {
    if (luma_candidate_enabled(kLumaCandidates[i], gates.flags)) {
      ++count;
    }
  }
  return count;
}

int luma_search_candidates(const LumaSearchGates& gates, LumaCandidate* out,
                           int capacity) noexcept {
  int count = 0;
  for (int i = 0; i < kMaxLumaCandidates; ++i) {
    if (!luma_candidate_enabled(kLumaCandidates[i], gates.flags)) {
      continue;
    }
    if (out != nullptr) {
      if (count >= capacity) {
        return -1;
      }
      out[count] = kLumaCandidates[i];
    }
    ++count;
  }
  if (out == nullptr && capacity != 0) {
    return -1;
  }
  return count;
}

// ---------------------------------------------------------------------------
// Scratch layout
// ---------------------------------------------------------------------------

LumaSearchLayout luma_search_layout(int max_blocks, int max_block_area,
                                    int candidates_per_chunk) noexcept {
  LumaSearchLayout layout{};
  if (max_blocks <= 0 || max_block_area <= 0) {
    return layout;
  }
  layout.candidates_per_chunk =
      clamp_int(candidates_per_chunk, 1, kMaxLumaCandidates);
  layout.max_block_area = max_block_area;
  if (max_blocks >
      std::numeric_limits<int>::max() / layout.candidates_per_chunk) {
    return LumaSearchLayout{};
  }
  layout.job_capacity = max_blocks * layout.candidates_per_chunk;

  const std::size_t job_bytes =
      static_cast<std::size_t>(layout.job_capacity) * sizeof(IntraJob);
  const std::size_t sse_bytes =
      static_cast<std::size_t>(layout.job_capacity) *
      sizeof(std::int64_t);
  const std::size_t pred_bytes =
      static_cast<std::size_t>(layout.job_capacity) *
      static_cast<std::size_t>(max_block_area);

  layout.jobs_offset = 0;
  layout.sse_offset = align_up(job_bytes, 16);
  layout.pred_offset = align_up(layout.sse_offset + sse_bytes, 16);
  layout.total_bytes = align_up(layout.pred_offset + pred_bytes, 256);
  return layout;
}

std::size_t luma_search_scratch_bytes(int max_blocks, int max_block_area,
                                      int candidates_per_chunk) noexcept {
  return luma_search_layout(max_blocks, max_block_area, candidates_per_chunk)
      .total_bytes;
}

LumaSearchScratch luma_search_make_scratch(
    void* base, const LumaSearchLayout& layout) noexcept {
  LumaSearchScratch scratch{};
  if (base == nullptr || layout.total_bytes == 0) {
    return scratch;
  }
  std::uint8_t* bytes = static_cast<std::uint8_t*>(base);
  scratch.d_jobs =
      reinterpret_cast<IntraJob*>(bytes + layout.jobs_offset);
  scratch.d_sse =
      reinterpret_cast<std::int64_t*>(bytes + layout.sse_offset);
  scratch.d_pred = bytes + layout.pred_offset;
  scratch.job_capacity = layout.job_capacity;
  scratch.pred_capacity =
      static_cast<std::size_t>(layout.job_capacity) *
      static_cast<std::size_t>(layout.max_block_area);
  scratch.max_block_area = layout.max_block_area;
  scratch.candidates_per_chunk = layout.candidates_per_chunk;
  return scratch;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

cudaError_t luma_search(const LumaSearchBlock* d_blocks, int block_count,
                        const LumaSearchGates& gates,
                        const LumaSearchScratch& scratch,
                        LumaSearchResult* d_results,
                        const LumaSearchDebug* debug,
                        cudaStream_t stream) {
  constexpr unsigned kKnownGates =
      kGateDcOnly | kGateBaseOnly | kGateNoDirectional | kGateDcOnlyPredict;
  if (block_count < 0 || (gates.flags & ~kKnownGates) != 0u) {
    return cudaErrorInvalidValue;
  }
  if (block_count == 0) {
    return cudaSuccess;
  }
  if (d_blocks == nullptr || d_results == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int active_total = luma_search_candidate_count(gates);
  if (active_total <= 0) {
    return cudaErrorInvalidValue;
  }

  const int chunk = clamp_int(scratch.candidates_per_chunk, 1,
                             kMaxLumaCandidates);
  const long long required_jobs =
      static_cast<long long>(block_count) * chunk;
  if (required_jobs > scratch.job_capacity) {
    return cudaErrorInvalidValue;
  }
  if (scratch.max_block_area <= 0 || scratch.d_jobs == nullptr ||
      scratch.d_sse == nullptr || scratch.d_pred == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::size_t required_pred =
      static_cast<std::size_t>(required_jobs) *
      static_cast<std::size_t>(scratch.max_block_area);
  if (scratch.pred_capacity < required_pred) {
    return cudaErrorInvalidValue;
  }
  std::int64_t* debug_sse = nullptr;
  int debug_stride = 0;
  if (debug != nullptr && debug->d_candidate_sse != nullptr) {
    if (debug->stride < active_total) {
      return cudaErrorInvalidValue;
    }
    debug_sse = debug->d_candidate_sse;
    debug_stride = debug->stride;
  }

  if ((gates.flags & kGateDcOnly) != 0u) {
    const int grid = (block_count + kCombineThreads - 1) / kCombineThreads;
    ls_dc_only_kernel<<<grid, kCombineThreads, 0, stream>>>(block_count,
                                                            d_results);
    return cudaGetLastError();
  }

  const int enable_edge_filter = gates.enable_edge_filter != 0 ? 1 : 0;
  for (int active_base = 0; active_base < active_total;
       active_base += chunk) {
    const int active_count = clamp_int(chunk, 1, active_total - active_base);
    const int jobs = block_count * active_count;

    const int expand_grid = (jobs + kExpandThreads - 1) / kExpandThreads;
    ls_expand_kernel<<<expand_grid, kExpandThreads, 0, stream>>>(
        d_blocks, block_count, gates.flags, active_base, active_count,
        scratch.max_block_area, enable_edge_filter, scratch.d_jobs,
        scratch.d_pred);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return err;
    }

    // Existing component: one CUDA block per job, jobs are independent.
    err = intra_modes::intra_predict_batch(scratch.d_jobs, jobs, stream);
    if (err != cudaSuccess) {
      return err;
    }

    ls_sse_kernel<<<jobs, kSseThreads, 0, stream>>>(scratch.d_jobs,
                                                    scratch.d_sse);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      return err;
    }

    const int combine_grid =
        (block_count + kCombineThreads - 1) / kCombineThreads;
    ls_combine_kernel<<<combine_grid, kCombineThreads, 0, stream>>>(
        block_count, gates.flags, active_base, active_count, scratch.d_sse,
        d_results, debug_sse, debug_stride);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      return err;
    }
  }
  return cudaSuccess;
}

}  // namespace av2enc::cuda::still::search
