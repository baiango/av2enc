// AV2 inter motion estimation — CUDA kernels.
//
// One warp per job.  The search state machine is me_search_core() from the
// shared core header; only the cost functor differs (warp-cooperative rows /
// 8x8 SATD tiles instead of a scalar loop).  Integer sums are associative, so
// the reduced cost equals the CPU cost exactly.

#include "cuda/codec/inter/me/me.h"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda {
namespace {

using me::MeCoreFrame;
using me::MeCoreJob;
using me::MeCoreMvField;
using me::MeCoreOptions;
using me::MeCorePlane;
using me::MeCoreRef;
using me::MeCoreResult;
using me::MeCostJob;
using me::MeInterpJob;

constexpr int kWarpsPerBlock = 1;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;

__device__ __forceinline__ std::int32_t warp_sum(std::int32_t v) {
  v += __shfl_down_sync(0xffffffffu, v, 16);
  v += __shfl_down_sync(0xffffffffu, v, 8);
  v += __shfl_down_sync(0xffffffffu, v, 4);
  v += __shfl_down_sync(0xffffffffu, v, 2);
  v += __shfl_down_sync(0xffffffffu, v, 1);
  return __shfl_sync(0xffffffffu, v, 0);
}

struct WarpCost {
  __device__ __noinline__ std::int32_t operator()(const MeCorePlane& cur,
                                     const MeCorePlane& ref, int x, int y,
                                     int w, int h, std::int32_t mv_x,
                                     std::int32_t mv_y, int metric,
                                     int filter) const {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    std::int32_t local = 0;
    if (metric == me::kMeSad) {
      if (me::me_is_integer_mv(mv_x, mv_y)) {
        local = me::me_sad_rows(cur, ref, x, y, w, mv_x,
                                mv_y >> me::kSubpelBits, lane, h, 32);
      } else {
        local = me::me_sad_subpel_rows(cur, ref, x, y, w, mv_x, mv_y, filter,
                                       lane, h, 32);
      }
    } else {
      const int tiles_x = (w + 7) >> 3;
      const int tiles_y = (h + 7) >> 3;
      const int units = tiles_x * tiles_y;
      for (int u = lane; u < units; u += 32) {
        const int tx = (u % tiles_x) << 3;
        const int ty = (u / tiles_x) << 3;
        local += me::me_satd_unit(cur, ref, x, y, w, h, mv_x, mv_y, filter, tx,
                                  ty);
      }
    }
    return warp_sum(local);
  }
};

__device__ __forceinline__ MeCorePlane plane_for(const MeCoreRef& ref,
                                                 int plane) {
  return ref.plane[plane];
}

__global__ void me_search_kernel(MeCoreFrame cur, const MeCoreRef* refs,
                                 int num_jobs, MeCoreOptions options,
                                 const MeCoreMvField* field,
                                 const MeCoreJob* jobs, MeCoreResult* results) {
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >>
                                    5);
  if (warp >= num_jobs) {
    return;
  }
  const MeCoreJob& job = jobs[warp];
  const MeCoreResult r =
      me::me_search_core(cur, refs, job, options, field, WarpCost{});
  if ((static_cast<int>(threadIdx.x) & 31) == 0) {
    results[warp] = r;
  }
}

__global__ void me_cost_kernel(MeCoreFrame cur, const MeCoreRef* refs,
                               const MeCostJob* jobs, int num_jobs,
                               std::int32_t* out) {
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >>
                                    5);
  if (warp >= num_jobs) {
    return;
  }
  const MeCostJob& job = jobs[warp];
  const MeCorePlane cur_p = cur.plane[job.plane];
  const MeCorePlane ref_p = plane_for(refs[job.ref_index], job.plane);
  const std::int32_t c =
      WarpCost{}(cur_p, ref_p, job.x, job.y, job.w, job.h, job.mv_x, job.mv_y,
                job.metric, job.interp);
  if ((static_cast<int>(threadIdx.x) & 31) == 0) {
    out[warp] = c;
  }
}

__global__ void me_interp_kernel(const MeCoreRef* refs, int num_jobs,
                                 const MeInterpJob* jobs, std::uint8_t* out) {
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >>
                                    5);
  if (warp >= num_jobs) {
    return;
  }
  const MeInterpJob& job = jobs[warp];
  const MeCorePlane ref_p = plane_for(refs[job.ref_index], job.plane);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int count = job.w * job.h;
  for (int i = lane; i < count; i += 32) {
    const int r = i / job.w;
    const int c = i % job.w;
    const int v = me::me_sample_at(ref_p, job.x + c, job.y + r, job.mv_x,
                                   job.mv_y, job.interp);
    out[static_cast<std::size_t>(job.out_offset) +
        static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(v);
  }
}

bool host_plane_valid(const MeCorePlane& p) {
  return p.data != nullptr && p.width > 0 && p.height > 0 &&
         p.stride >= p.width;
}

bool host_frame_valid(const MeCoreFrame& f) {
  return host_plane_valid(f.plane[0]) && f.width == f.plane[0].width &&
         f.height == f.plane[0].height;
}

bool host_options_valid(const MeCoreOptions& o) {
  if (o.method < me::kMeFull || o.method > me::kMeTestZone) {
    return false;
  }
  if (o.metric != me::kMeSad && o.metric != me::kMeSatd) {
    return false;
  }
  if (o.range < 0 || o.range > 4096 || o.integer_step < 1 ||
      o.max_iterations < 1 || o.subpel_iterations < 0 ||
      o.max_candidates < 0 || o.diamond_step < 0 || o.tz_start_step < 0) {
    return false;
  }
  if ((o.subpel_stages & ~(me::kMeSubpelHalf | me::kMeSubpelQuarter |
                           me::kMeSubpelEighth | me::kMeSubpelSixteenth)) != 0) {
    return false;
  }
  if (o.subpel_interp < 0 || o.subpel_interp > me::kMeMaxInterpFilter) {
    return false;
  }
  if (o.use_predictors < 0 || o.use_predictors > 1 ||
      o.use_merge_candidates < 0 || o.use_merge_candidates > 1) {
    return false;
  }
  if (o.max_merge_candidates < 0 ||
      o.max_merge_candidates > me::kMeMaxMerge) {
    return false;
  }
  return true;
}

int launch_grid(int num_jobs) {
  return (num_jobs + kWarpsPerBlock - 1) / kWarpsPerBlock;
}

}  // namespace

cudaError_t me_search_batch(const MeCoreFrame& cur,
                            const MeCoreRef* d_refs, int num_refs,
                            const MeCoreJob* d_jobs, int num_jobs,
                            const MeCoreOptions& options,
                            const MeCoreMvField* d_field,
                            MeCoreResult* d_results, cudaStream_t stream) {
  if (!host_frame_valid(cur) || !host_options_valid(options)) {
    return cudaErrorInvalidValue;
  }
  if (num_refs <= 0 || num_refs > me::kMeMaxRefs || d_refs == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (num_jobs < 0 || (num_jobs > 0 && (d_jobs == nullptr || d_results == nullptr))) {
    return cudaErrorInvalidValue;
  }
  if (num_jobs == 0) {
    return cudaSuccess;
  }
  me_search_kernel<<<launch_grid(num_jobs), kThreadsPerBlock, 0, stream>>>(
      cur, d_refs, num_jobs, options, d_field, d_jobs, d_results);
  return cudaGetLastError();
}

cudaError_t me_cost_batch(const MeCoreFrame& cur, const MeCoreRef* d_refs,
                          int num_refs, const MeCostJob* d_jobs, int num_jobs,
                          std::int32_t* d_out, cudaStream_t stream) {
  if (!host_frame_valid(cur)) {
    return cudaErrorInvalidValue;
  }
  if (num_refs <= 0 || num_refs > me::kMeMaxRefs || d_refs == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (num_jobs < 0 || (num_jobs > 0 && (d_jobs == nullptr || d_out == nullptr))) {
    return cudaErrorInvalidValue;
  }
  if (num_jobs == 0) {
    return cudaSuccess;
  }
  me_cost_kernel<<<launch_grid(num_jobs), kThreadsPerBlock, 0, stream>>>(
      cur, d_refs, d_jobs, num_jobs, d_out);
  return cudaGetLastError();
}

cudaError_t me_interp_batch(const MeCoreRef* d_refs, int num_refs,
                            const MeInterpJob* d_jobs, int num_jobs,
                            std::uint8_t* d_out, cudaStream_t stream) {
  if (num_refs <= 0 || num_refs > me::kMeMaxRefs || d_refs == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (num_jobs < 0 || (num_jobs > 0 && (d_jobs == nullptr || d_out == nullptr))) {
    return cudaErrorInvalidValue;
  }
  if (num_jobs == 0) {
    return cudaSuccess;
  }
  me_interp_kernel<<<launch_grid(num_jobs), kThreadsPerBlock, 0, stream>>>(
      d_refs, num_jobs, d_jobs, d_out);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
