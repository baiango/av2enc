// AV2 inter motion estimation, extended — CUDA kernels.
//
// One warp per job.  The search state machine is me2_search_core() shared with
// the CPU reference; only the cost functor differs (warp-cooperative rows /
// 8x8 SATD tiles).  Integer sums are associative, so costs, MVs, precision
// decisions and evaluation counts match the CPU exactly.

#include "codec/inter/me_precision/me2_gpu.h"

#include <cstddef>
#include <cstdint>

#include "codec/inter/me_precision/me2_gpu_cost.cuh"

namespace av2enc::cuda {
namespace {

using me2::Me2CostJob;
using me2::Me2Options;
using me2::Me2Result;
using me2::MeCoreFrame;
using me2::MeCoreJob;
using me2::MeCoreMvField;
using me2::MeCorePlane;
using me2::MeCoreRef;

constexpr int kWarpsPerBlock = 1;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;

bool plane_valid(const MeCorePlane& p) {
  return p.data != nullptr && p.width > 0 && p.height > 0 &&
         p.stride >= p.width;
}

bool frame_valid(const MeCoreFrame& f) {
  return plane_valid(f.plane[0]) && f.width == f.plane[0].width &&
         f.height == f.plane[0].height;
}

bool options_valid(const Me2Options& o) {
  const me::MeCoreOptions& s = o.search;
  if (s.method < me::kMeFull || s.method > me::kMeTestZone) {
    return false;
  }
  if (s.metric != me::kMeSad && s.metric != me::kMeSatd) {
    return false;
  }
  if (s.range < 0 || s.range > 4096 || s.integer_step < 1 ||
      s.max_iterations < 1 || s.subpel_iterations < 0 ||
      s.max_candidates < 0 || s.diamond_step < 0 || s.tz_start_step < 0) {
    return false;
  }
  if ((s.subpel_stages & ~(me::kMeSubpelHalf | me::kMeSubpelQuarter |
                           me::kMeSubpelEighth |
                           me::kMeSubpelSixteenth)) != 0) {
    return false;
  }
  if (s.subpel_interp < 0 || s.subpel_interp > me::kMeMaxInterpFilter) {
    return false;
  }
  if (s.use_predictors < 0 || s.use_predictors > 1 ||
      s.use_merge_candidates < 0 || s.use_merge_candidates > 1 ||
      s.max_merge_candidates < 0 ||
      s.max_merge_candidates > me::kMeMaxMerge) {
    return false;
  }
  if (!me2::me2_precision_valid(o.frame_precision) ||
      o.enable_flex_mvres < 0 || o.enable_flex_mvres > 1 ||
      o.precision_mask < 0 ||
      o.precision_mask >= (1 << me2::kMvPrecisionCount)) {
    return false;
  }
  if (o.refine_radius < 0 || o.refine_radius > 64 || o.refine_step < 0 ||
      o.refine_step > 16 || o.mv_rate_lambda < 0 || o.mv_rate_shift < 0 ||
      o.mv_rate_shift > 24 || o.count_precision_bins < 0 ||
      o.count_precision_bins > 1) {
    return false;
  }
  return true;
}

int launch_grid(int num_jobs) {
  return (num_jobs + kWarpsPerBlock - 1) / kWarpsPerBlock;
}

__global__ void me2_search_kernel(MeCoreFrame cur, const MeCoreRef* refs,
                                  int num_jobs, Me2Options options,
                                  const MeCoreMvField* field,
                                  const MeCoreJob* jobs,
                                  Me2Result* results) {
  const int warp =
      static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  if (warp >= num_jobs) {
    return;
  }
  const MeCoreJob& job = jobs[warp];
  const Me2Result r =
      me2::me2_search_core(cur, refs, job, options, field,
                           me2::gpu::Me2WarpCost{});
  if ((static_cast<int>(threadIdx.x) & 31) == 0) {
    results[warp] = r;
  }
}

__global__ void me2_cost_kernel(MeCoreFrame cur, const MeCoreRef* refs,
                                const Me2CostJob* jobs, int num_jobs,
                                std::int32_t* out) {
  const int warp =
      static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  if (warp >= num_jobs) {
    return;
  }
  const Me2CostJob& job = jobs[warp];
  const MeCorePlane cur_p = cur.plane[job.base.plane];
  const MeCorePlane ref_p = refs[job.base.ref_index].plane[job.base.plane];
  const std::int32_t c =
      me2::gpu::Me2WarpCost{}(cur_p, ref_p, job.base.x, job.base.y, job.base.w,
                              job.base.h, job.base.mv_x, job.base.mv_y,
                              job.base.metric, job.base.interp);
  if ((static_cast<int>(threadIdx.x) & 31) == 0) {
    out[warp] = c;
  }
}

}  // namespace

cudaError_t me2_search_batch(const MeCoreFrame& cur,
                             const MeCoreRef* d_refs, int num_refs,
                             const MeCoreJob* d_jobs, int num_jobs,
                             const Me2Options& options,
                             const MeCoreMvField* d_field,
                             Me2Result* d_results, cudaStream_t stream) {
  if (!frame_valid(cur) || !options_valid(options)) {
    return cudaErrorInvalidValue;
  }
  if (num_refs <= 0 || num_refs > me::kMeMaxRefs || d_refs == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (num_jobs < 0 ||
      (num_jobs > 0 && (d_jobs == nullptr || d_results == nullptr))) {
    return cudaErrorInvalidValue;
  }
  if (num_jobs == 0) {
    return cudaSuccess;
  }
  me2_search_kernel<<<launch_grid(num_jobs), kThreadsPerBlock, 0, stream>>>(
      cur, d_refs, num_jobs, options, d_field, d_jobs, d_results);
  return cudaGetLastError();
}

cudaError_t me2_cost_batch(const MeCoreFrame& cur,
                           const MeCoreRef* d_refs, int num_refs,
                           const Me2CostJob* d_jobs, int num_jobs,
                           std::int32_t* d_out, cudaStream_t stream) {
  if (!frame_valid(cur)) {
    return cudaErrorInvalidValue;
  }
  if (num_refs <= 0 || num_refs > me::kMeMaxRefs || d_refs == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (num_jobs < 0 ||
      (num_jobs > 0 && (d_jobs == nullptr || d_out == nullptr))) {
    return cudaErrorInvalidValue;
  }
  if (num_jobs == 0) {
    return cudaSuccess;
  }
  me2_cost_kernel<<<launch_grid(num_jobs), kThreadsPerBlock, 0, stream>>>(
      cur, d_refs, d_jobs, num_jobs, d_out);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
