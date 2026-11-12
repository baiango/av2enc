// Host emulation of the shared warp-stack device header.
//
// Compiled by the host compiler, not nvcc: `__CUDACC__` is undefined, so
// `warp_stack_device.cuh` turns its device entry points into plain inline
// functions. The only CUDA runtime symbol the header needs is
// `__syncthreads()`, stubbed below; the host runner executes single-threaded.

#include "warp_stack_host_ref.h"

#include <cstdint>

namespace {
inline void __syncthreads() {}
}  // namespace

#include "warp_stack_device.cuh"

namespace av2enc::cuda::warpstack::hostref {

void predict(const WarpPredictJob& job, std::int16_t* dst, int dst_stride,
             WarpPredictResult* result) {
  WarpPredictJob local = job;
  local.dst = dst;
  local.dst_stride = dst_stride;
  std::int32_t intermediate[kWarpIntermediateInts] = {};
  if (local.geom.bit_depth > 8) {
    ws_predict_job_device<std::uint16_t>(local, 0, 1, intermediate, result);
  } else {
    ws_predict_job_device<std::uint8_t>(local, 0, 1, intermediate, result);
  }
}

bool corners(const WarpCornerJob& job, WarpCornerResult* result) {
  if (result == nullptr) {
    return false;
  }
  result->model_valid =
      ws_generate_points_from_corners(job.grid, job.sb_h4, job.mi_row, job.mi_col, job.bw4,
                                      job.bh4, job.ref_frame0, job.iter, &result->corner)
          ? 1
          : 0;
  return result->model_valid != 0;
}

bool samples(const inter::warp::WarpSampleParams& params,
             inter::warp::WarpSampleSet* result) {
  return ws_find_warp_samples(params, result);
}

ia::WarpParams estimate(const WarpEstimateJob& job) {
  return ws_estimate_warp_from_samples(job.samples, job.mid_x, job.mid_y, job.block_mv);
}

bool needs_extended(const ia::WarpParams& params, int w, int h, int ref_w, int ref_h,
                    int frame_w, int frame_h) {
  return ws_needs_extended(params, w, h, ref_w, ref_h, frame_w, frame_h);
}

bool shear_valid(const ia::WarpParams& params) { return ia::setup_shear(params).valid; }

}  // namespace av2enc::cuda::warpstack::hostref
