// MRL/DIP extra intra decision stages.  See intra_extra.h.

#include "intra_extra.h"

#include <climits>
#include <cstddef>
#include <cstdint>

#include "cuda/codec/intra/dip/dip_device.cuh"
#include "cuda/codec/intra/ibp/ibp_predict_device.cuh"
#include "cuda/codec/intra/mrl/mrl_device.cuh"

namespace av2enc::cuda::still::intra_extra {
namespace {

namespace srch = ::av2enc::cuda::still::search;
namespace im = ::av2enc::cuda::intra_modes;

constexpr int kThreads = 128;
constexpr int kMaxArea = 64 * 64;

// Device-side view of one search block plus the plan's SbSize.
struct Spec {
  const std::uint8_t* recon = nullptr;
  int recon_stride = 0;
  const std::uint8_t* source = nullptr;
  int source_stride = 0;
  int source_max_x = 0;
  int source_max_y = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int max_x = 0;
  int max_y = 0;
  int have_left = 0;
  int have_above = 0;
  int num4_above_right = 0;
  int num4_below_left = 0;
  int filter_type_above = 0;
  int filter_type_left = 0;
  int enable_edge_filter = 1;
  int sb_height = 64;
};

__device__ Spec make_spec(const srch::LumaSearchBlock& b, int sb_height,
                          int enable_edge_filter) {
  Spec spec{};
  spec.recon = b.recon_plane;
  spec.recon_stride = b.recon_stride;
  spec.source = b.source_plane;
  spec.source_stride = b.source_stride;
  spec.source_max_x = b.source_max_x;
  spec.source_max_y = b.source_max_y;
  spec.x = b.x0;
  spec.y = b.y0;
  spec.w = b.width;
  spec.h = b.height;
  spec.max_x = b.max_x;
  spec.max_y = b.max_y;
  spec.have_left = b.have_left;
  spec.have_above = b.have_above;
  spec.num4_above_right = b.num4_above_right;
  spec.num4_below_left = b.num4_below_left;
  spec.filter_type_above = b.filter_type_above;
  spec.filter_type_left = b.filter_type_left;
  spec.enable_edge_filter = enable_edge_filter;
  spec.sb_height = sb_height;
  return spec;
}

__device__ __constant__ srch::LumaCandidate kDevCandidates
    [srch::kMaxLumaCandidates] = {AV2ENC_STILL_LUMA_CANDIDATE_LIST};

// Block-wide SSE of `pred` against the visible source.  Matches the CPU
// `sse_against_source()` (source samples clamp to the visible plane).
__device__ long long block_sse(const Spec& b, const std::uint8_t* pred,
                               long long* scratch) {
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  const int total = b.w * b.h;
  long long acc = 0;
  for (int index = tid; index < total; index += nthr) {
    const int py = index / b.w;
    const int px = index - py * b.w;
    const int sy = min(b.source_max_y, b.y + py);
    const int sx = min(b.source_max_x, b.x + px);
    const int src = static_cast<int>(b.source[
        static_cast<std::size_t>(sy) * static_cast<std::size_t>(b.source_stride) +
        static_cast<std::size_t>(sx)]);
    const int diff = src - static_cast<int>(pred[index]);
    acc += static_cast<long long>(diff) * diff;
  }
  scratch[tid] = acc;
  __syncthreads();
  for (int stride = nthr / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ mrl::MrlBlock make_mrl_block(const Spec& b) {
  mrl::MrlBlock block{};
  block.recon = b.recon;
  block.recon_stride = b.recon_stride;
  block.x = b.x;
  block.y = b.y;
  block.w = b.w;
  block.h = b.h;
  block.max_x = b.max_x;
  block.max_y = b.max_y;
  block.have_left = b.have_left;
  block.have_above = b.have_above;
  block.num4_above_right = b.num4_above_right;
  block.num4_below_left = b.num4_below_left;
  block.plane = 0;
  block.sb_height = b.sb_height;
  return block;
}

__device__ dip::DipBlock make_dip_block(const Spec& b) {
  dip::DipBlock block{};
  block.recon = b.recon;
  block.recon_stride = b.recon_stride;
  block.x = b.x;
  block.y = b.y;
  block.w = b.w;
  block.h = b.h;
  block.max_x = b.max_x;
  block.max_y = b.max_y;
  block.have_left = b.have_left;
  block.have_above = b.have_above;
  block.num4_above_right = b.num4_above_right;
  block.num4_below_left = b.num4_below_left;
  return block;
}

// ---------------------------------------------------------------------------
// IBP luma search (61 candidates, CPU order/tie-break)
// ---------------------------------------------------------------------------

__device__ __forceinline__ im::IntraBlock make_intra_block(
    const srch::LumaSearchBlock& b, const srch::LumaCandidate& candidate,
    int enable_edge_filter) {
  im::IntraBlock block{};
  block.recon_plane = b.recon_plane;
  block.recon_stride = b.recon_stride;
  block.x0 = b.x0;
  block.y0 = b.y0;
  block.width = b.width;
  block.height = b.height;
  block.max_x = b.max_x;
  block.max_y = b.max_y;
  block.have_left = b.have_left;
  block.have_above = b.have_above;
  block.num4_above_right = b.num4_above_right;
  block.num4_below_left = b.num4_below_left;
  block.plane = 0;
  block.mode = static_cast<im::IntraMode>(candidate.mode);
  block.angle_delta = candidate.angle_delta;
  block.filter_type_above = b.filter_type_above;
  block.filter_type_left = b.filter_type_left;
  block.enable_edge_filter = enable_edge_filter;
  block.source_plane = b.source_plane;
  block.source_stride = b.source_stride;
  block.source_max_x = b.source_max_x;
  block.source_max_y = b.source_max_y;
  return block;
}

__global__ void ibp_luma_search_kernel(const srch::LumaSearchBlock* d_block,
                                       int enable_edge_filter, unsigned flags,
                                       srch::LumaSearchResult* result,
                                       std::int64_t* debug_sse) {
  __shared__ ibp::SharedScratch ibp_s;
  __shared__ im::SharedScratch intra_s;
  __shared__ std::uint8_t pred[kMaxArea];
  __shared__ long long red[kThreads];
  __shared__ long long best_sse;
  __shared__ int best_mode;
  __shared__ int best_angle;

  const int tid = static_cast<int>(threadIdx.x);
  const srch::LumaSearchBlock block = *d_block;
  if (tid == 0) {
    best_sse = LLONG_MAX;
    best_mode = im::kDcPred;
    best_angle = 0;
  }
  __syncthreads();

  int ordinal = 0;
  for (int i = 0; i < srch::kMaxLumaCandidates; ++i) {
    const srch::LumaCandidate candidate = kDevCandidates[i];
    if (!srch::luma_candidate_enabled(candidate, flags)) {
      continue;
    }
    const im::IntraBlock intra =
        make_intra_block(block, candidate, enable_edge_filter);
    const int mode = candidate.mode;
    if (mode == im::kDcPred || im::is_directional_mode(mode)) {
      ibp::IbpBlock ibp_block{};
      ibp_block.intra = intra;
      ibp_block.enable_ibp = 1;
      ibp_block.use_dip = 0;
      ibp_block.uv_is_cfl = 0;
      ibp::ibp_run_block_device(ibp_block, pred, &ibp_s);
    } else {
      im::intra_run_block_device(intra, pred, nullptr,
                                                &intra_s);
    }
    __syncthreads();
    const long long value = block_sse(
        make_spec(block, 64, enable_edge_filter), pred, red);
    if (tid == 0) {
      if (debug_sse != nullptr) {
        debug_sse[ordinal] = value;
      }
      if (value < best_sse) {
        best_sse = value;
        best_mode = mode;
        best_angle = candidate.angle_delta;
      }
    }
    ++ordinal;
    __syncthreads();
  }

  if (tid == 0) {
    srch::LumaSearchResult value = *result;
    value.mode = best_mode;
    value.angle_delta = best_angle;
    value.sse = best_sse;
    *result = value;
  }
}

// ---------------------------------------------------------------------------
// MRL search
// ---------------------------------------------------------------------------

__global__ void mrl_search_kernel(const srch::LumaSearchBlock* d_block,
                                  int sb_height, int enable_edge_filter,
                                  const std::int64_t* base_sse,
                                  unsigned flags,
                                  srch::LumaSearchResult* result,
                                  Choice* choice) {
  const Spec b = make_spec(*d_block, sb_height, enable_edge_filter);
  __shared__ mrl::MrlSharedScratch mrl_s;
  __shared__ std::uint8_t pred[kMaxArea];
  __shared__ long long red[kThreads];
  __shared__ long long best_sse;
  __shared__ int best_mode;
  __shared__ int best_angle;
  __shared__ int best_mrl;
  __shared__ int best_sec;

  const int tid = static_cast<int>(threadIdx.x);
  if (tid == 0) {
    best_sse = LLONG_MAX;
    best_mode = im::kDcPred;
    best_angle = 0;
    best_mrl = 0;
    best_sec = 0;
  }
  __syncthreads();

  const mrl::MrlBlock block = make_mrl_block(b);
  const bool sec_allowed = !(b.w == 4 && b.h == 4);
  int ordinal = 0;
  for (int i = 0; i < srch::kMaxLumaCandidates; ++i) {
    const srch::LumaCandidate candidate = kDevCandidates[i];
    if (!srch::luma_candidate_enabled(candidate, flags)) {
      continue;
    }
    if (tid == 0) {
      const long long value = base_sse[ordinal];
      if (value < best_sse) {
        best_sse = value;
        best_mode = candidate.mode;
        best_angle = candidate.angle_delta;
        best_mrl = 0;
        best_sec = 0;
      }
    }
    ++ordinal;
    if (!im::is_directional_mode(candidate.mode)) {
      continue;
    }
    for (int mrl_index = 1; mrl_index <= mrl::kMaxMrlIndex; ++mrl_index) {
      const int sec_count = sec_allowed ? 2 : 1;
      for (int sec = 0; sec < sec_count; ++sec) {
        mrl::MrlParams params{};
        params.mode = candidate.mode;
        params.angle_delta = candidate.angle_delta;
        params.mrl_index = mrl_index;
        params.mrl_sec_index = sec;
        params.enable_edge_filter = b.enable_edge_filter;
        params.filter_type_above = b.filter_type_above;
        params.filter_type_left = b.filter_type_left;
        mrl::mrl_run_block_device(block, params, pred, &mrl_s);
        __syncthreads();
        const long long value = block_sse(b, pred, red);
        if (tid == 0 && value < best_sse) {
          best_sse = value;
          best_mode = candidate.mode;
          best_angle = candidate.angle_delta;
          best_mrl = mrl_index;
          best_sec = sec;
        }
        __syncthreads();
      }
    }
  }

  if (tid == 0) {
    srch::LumaSearchResult value = *result;
    value.mode = best_mode;
    value.angle_delta = best_angle;
    value.sse = best_sse;
    *result = value;
    choice->mrl_index = best_mrl;
    choice->mrl_sec_index = best_sec;
  }
}

// ---------------------------------------------------------------------------
// DIP search
// ---------------------------------------------------------------------------

__global__ void dip_search_kernel(const srch::LumaSearchBlock* d_block,
                                  int sb_height, int enable_edge_filter,
                                  int enable_dip, int base_only,
                                  srch::LumaSearchResult* result,
                                  Choice* choice) {
  const Spec b = make_spec(*d_block, sb_height, enable_edge_filter);
  __shared__ dip::DipSharedScratch dip_s;
  __shared__ std::uint8_t pred[kMaxArea];
  __shared__ long long red[kThreads];
  __shared__ long long best_sse;
  __shared__ int best_mode;
  __shared__ int best_transpose;

  const int tid = static_cast<int>(threadIdx.x);
  if (enable_dip == 0 || base_only != 0 || !dip::dip_block_available(b.w, b.h)) {
    return;
  }
  const dip::DipBlock block = make_dip_block(b);
  if (tid == 0) {
    best_sse = result->sse;
    best_mode = -1;
    best_transpose = 0;
  }
  __syncthreads();

  for (int mode = 0; mode < dip::kNumModes; ++mode) {
    for (int transpose = 0; transpose < 2; ++transpose) {
      dip::DipParams params{};
      params.mode = mode;
      params.transpose = transpose;
      dip::dip_run_block_device(block, params, pred, &dip_s);
      __syncthreads();
      const long long value = block_sse(b, pred, red);
      if (tid == 0 && value < best_sse) {
        best_sse = value;
        best_mode = mode;
        best_transpose = transpose;
      }
      __syncthreads();
    }
  }

  if (tid == 0 && best_mode >= 0) {
    srch::LumaSearchResult value = *result;
    value.mode = im::kDcPred;
    value.angle_delta = 0;
    value.sse = best_sse;
    *result = value;
    choice->mrl_index = 0;
    choice->mrl_sec_index = 0;
    choice->use_dip = 1;
    choice->dip_mode = best_mode;
    choice->dip_transpose = best_transpose;
  }
}

// ---------------------------------------------------------------------------
// Chosen-prediction descriptors + selection
// ---------------------------------------------------------------------------

__global__ void prepare_jobs_kernel(mrl::MrlJob* mrl_jobs,
                                    dip::DipJob* dip_jobs,
                                    ibp::IbpJob* ibp_jobs,
                                    const srch::LumaSearchBlock* search_blocks,
                                    const srch::LumaSearchResult* results,
                                    const Choice* choices, int index,
                                    int enable_edge_filter) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const srch::LumaSearchResult result = results[index];
  const srch::LumaSearchBlock search = search_blocks[index];
  const Choice choice = choices[index];
  if (mrl_jobs != nullptr) {
    mrl::MrlJob& job = mrl_jobs[index];
    job.params.mode = result.mode;
    job.params.angle_delta = result.angle_delta;
    job.params.mrl_index = choice.mrl_index;
    job.params.mrl_sec_index = choice.mrl_sec_index;
    job.params.enable_edge_filter = enable_edge_filter;
    job.params.filter_type_above = search.filter_type_above;
    job.params.filter_type_left = search.filter_type_left;
  }
  if (dip_jobs != nullptr) {
    dip::DipJob& job = dip_jobs[index];
    job.params.mode = choice.dip_mode;
    job.params.transpose = choice.dip_transpose;
  }
  if (ibp_jobs != nullptr) {
    ibp::IbpJob& job = ibp_jobs[index];
    job.block.intra.mode = static_cast<im::IntraMode>(result.mode);
    job.block.intra.angle_delta = result.angle_delta;
    job.block.intra.filter_type_above = search.filter_type_above;
    job.block.intra.filter_type_left = search.filter_type_left;
    job.block.intra.enable_edge_filter = enable_edge_filter;
  }
}

__global__ void select_prediction_kernel(const Choice* choice,
                                         const std::uint8_t* mrl_pred,
                                         const std::uint8_t* dip_pred,
                                         std::uint8_t* dst, int area) {
  const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
  if (index >= area) {
    return;
  }
  const Choice value = *choice;
  const std::uint8_t* src = nullptr;
  if (value.use_dip != 0) {
    src = dip_pred;
  } else if (value.mrl_index > 0) {
    src = mrl_pred;
  }
  if (src != nullptr) {
    dst[index] = src[index];
  }
}

}  // namespace

cudaError_t ibp_luma_search(const srch::LumaSearchBlock* block, int sb_height,
                            unsigned flags, int enable_edge_filter,
                            srch::LumaSearchResult* result,
                            std::int64_t* debug_sse, cudaStream_t stream) {
  (void)sb_height;
  if (block == nullptr || result == nullptr) {
    return cudaErrorInvalidValue;
  }
  ibp_luma_search_kernel<<<1, kThreads, 0, stream>>>(
      block, enable_edge_filter, flags, result, debug_sse);
  return cudaGetLastError();
}

cudaError_t mrl_search(const srch::LumaSearchBlock* block, int sb_height,
                       int enable_edge_filter,
                       const std::int64_t* base_sse, unsigned flags,
                       srch::LumaSearchResult* result, Choice* choice,
                       cudaStream_t stream) {
  if (block == nullptr || base_sse == nullptr || result == nullptr ||
      choice == nullptr) {
    return cudaErrorInvalidValue;
  }
  mrl_search_kernel<<<1, kThreads, 0, stream>>>(
      block, sb_height, enable_edge_filter, base_sse, flags, result, choice);
  return cudaGetLastError();
}

cudaError_t dip_search(const srch::LumaSearchBlock* block, int sb_height,
                       int enable_edge_filter, int enable_dip, int base_only,
                       srch::LumaSearchResult* result, Choice* choice,
                       cudaStream_t stream) {
  if (block == nullptr || result == nullptr || choice == nullptr) {
    return cudaErrorInvalidValue;
  }
  dip_search_kernel<<<1, kThreads, 0, stream>>>(
      block, sb_height, enable_edge_filter, enable_dip, base_only, result,
      choice);
  return cudaGetLastError();
}

cudaError_t prepare_jobs(mrl::MrlJob* mrl_jobs, dip::DipJob* dip_jobs,
                         ibp::IbpJob* ibp_jobs,
                         const srch::LumaSearchBlock* search_blocks,
                         const srch::LumaSearchResult* results,
                         const Choice* choices, int block_index,
                         int enable_edge_filter, cudaStream_t stream) {
  if (search_blocks == nullptr || results == nullptr || choices == nullptr ||
      block_index < 0) {
    return cudaErrorInvalidValue;
  }
  prepare_jobs_kernel<<<1, 1, 0, stream>>>(
      mrl_jobs, dip_jobs, ibp_jobs, search_blocks, results, choices,
      block_index, enable_edge_filter);
  return cudaGetLastError();
}

cudaError_t select_prediction(const Choice* choice,
                              const std::uint8_t* mrl_pred,
                              const std::uint8_t* dip_pred, std::uint8_t* dst,
                              int area, cudaStream_t stream) {
  if (choice == nullptr || dst == nullptr || area <= 0) {
    return cudaErrorInvalidValue;
  }
  const int grid = (area + kThreads - 1) / kThreads;
  select_prediction_kernel<<<grid, kThreads, 0, stream>>>(
      choice, mrl_pred, dip_pred, dst, area);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::still::intra_extra
