// M1 GPU chroma decision + prediction (AV2 v1.0.0 §7.13.5, §7.13.2.2).
//
// Every arithmetic step below is a transcription of the CPU path that M1
// actually uses:
//   * DC prediction: src/encoder/intra_modes/intra_modes.cpp predict_dc() +
//     prepare_edges(), including the x>0 / y>0 availability rule, the coded
//     plane clamp, and approx_divide().
//   * CfL: src/encoder/cfl/cfl.cpp compute_luma()/derive_implicit_alpha()/
//     apply_prediction(), with make_cfl_block()'s parameters from m1_still.cpp
//     (ds_filter = 0, 4:2:0, coded bounds, chroma-mi availability).
//   * Scoring/decision: m1_still.cpp choose_uv_mode(). The default policy
//     scores DC on U only (CPU quirk) and CfL on U+V; enable_uv_full runs the
//     CPU full-UV search: all 13 base modes with U+V SSE, derived
//     AngleDeltaUV, then CfL.
//
// The device body lives in m1_chroma_device.cuh so the kernel and the
// no-nvcc host-emulation tests run exactly the same integer code.

#include "m1_chroma.h"

#include "m1_chroma_device.cuh"

namespace av2enc::cuda::m1 {
namespace {

__device__ void run_block(const M1ChromaJob& job, const M1ChromaOptions& opt,
                          M1ChromaResult* result) {
  if (opt.enable_uv_full != 0 || opt.enable_uv_directional != 0) {
    __shared__ ChromaUvShared uv;
    chroma_uv_full_search(job, opt, &uv, result);
    return;
  }
  __shared__ ChromaDcCflShared dc_cfl;
  chroma_dc_cfl_search(job, opt, static_cast<int>(threadIdx.x),
                       static_cast<int>(blockDim.x), &dc_cfl, result);
}

__global__ void decide_kernel(M1ChromaJob job, M1ChromaOptions options,
                              M1ChromaResult* result) {
  run_block(job, options, result);
}

__global__ void decide_batch_kernel(const M1ChromaJob* jobs,
                                    M1ChromaOptions options,
                                    M1ChromaResult* results) {
  const int index = static_cast<int>(blockIdx.x);
  run_block(jobs[index], options, results + index);
}

bool full_uv_requested(const M1ChromaOptions& options) {
  return options.enable_uv_full != 0 || options.enable_uv_directional != 0;
}

cudaError_t validate_job(const M1ChromaJob& job,
                         const M1ChromaOptions& options) {
  if (options.ds_filter < 0 || options.ds_filter > 3) {
    return cudaErrorInvalidValue;
  }
  if (options.alpha_q8_min < -511 || options.alpha_q8_max > 511 ||
      options.alpha_q8_min > options.alpha_q8_max) {
    return cudaErrorInvalidValue;
  }
  if (job.luma == nullptr || job.chroma_u == nullptr ||
      job.chroma_v == nullptr || job.src_u == nullptr || job.src_v == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (job.luma_stride <= 0 || job.chroma_stride <= 0 ||
      job.src_stride_u <= 0 || job.src_stride_v <= 0) {
    return cudaErrorInvalidValue;
  }
  if (job.w <= 0 || job.h <= 0 || job.w > kMaxDim || job.h > kMaxDim) {
    return cudaErrorInvalidValue;
  }
  // CPU valid_block_size(): power-of-two dimensions in [4, 64].
  if ((job.w & (job.w - 1)) != 0 || (job.h & (job.h - 1)) != 0 ||
      job.w < 4 || job.h < 4) {
    return cudaErrorInvalidValue;
  }
  if (job.x < 0 || job.y < 0) {
    return cudaErrorInvalidValue;
  }
  if (job.coded_width <= 0 || job.coded_height <= 0 ||
      job.luma_max_x < 0 || job.luma_max_y < 0) {
    return cudaErrorInvalidValue;
  }
  // A bottom/right-edge block may extend past the coded plane; CfL truncates
  // its neighbour replication at coded_width/coded_height. Do not reject that.
  if (job.chroma_stride < job.x + job.w) {
    return cudaErrorInvalidValue;
  }
  if (job.src_max_x < 0 || job.src_max_y < 0) {
    return cudaErrorInvalidValue;
  }
  if (job.pred_stride < 0 ||
      (job.pred_stride > 0 && job.pred_stride < job.w)) {
    return cudaErrorInvalidValue;
  }
  if (full_uv_requested(options)) {
    if (job.y_mode < 0 || job.y_mode >= av2enc::cuda::intra_modes::kIntraModes ||
        job.y_angle_delta < -3 || job.y_angle_delta > 3) {
      return cudaErrorInvalidValue;
    }
    if ((job.filter_type_above & ~1) != 0 ||
        (job.filter_type_left & ~1) != 0 ||
        (job.enable_edge_filter & ~1) != 0) {
      return cudaErrorInvalidValue;
    }
    if (job.num4_above_right < 0 || job.num4_below_left < 0) {
      return cudaErrorInvalidValue;
    }
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t chroma_decide(const M1ChromaJob& job, const M1ChromaOptions& options,
                          M1ChromaResult* d_result, cudaStream_t stream) {
  if (d_result == nullptr) {
    return cudaErrorInvalidValue;
  }
  const cudaError_t status = validate_job(job, options);
  if (status != cudaSuccess) {
    return status;
  }
  decide_kernel<<<1, kThreads, 0, stream>>>(job, options, d_result);
  return cudaGetLastError();
}

cudaError_t chroma_decide_batch(const M1ChromaJob* d_jobs, int job_count,
                                const M1ChromaOptions& options,
                                M1ChromaResult* d_results,
                                cudaStream_t stream) {
  if (job_count <= 0) {
    return cudaSuccess;
  }
  if (d_jobs == nullptr || d_results == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (options.ds_filter < 0 || options.ds_filter > 3 ||
      options.alpha_q8_min < -511 || options.alpha_q8_max > 511 ||
      options.alpha_q8_min > options.alpha_q8_max) {
    return cudaErrorInvalidValue;
  }
  // The batch entry point cannot validate a device array of descriptors on
  // the host without a sync. The single-block path rejects malformed full-UV
  // descriptors; batch callers own batching correctness.
  decide_batch_kernel<<<job_count, kThreads, 0, stream>>>(d_jobs, options,
                                                         d_results);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::m1
