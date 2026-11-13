// CUDA kernels for the full compound path. See compound_full.h and
// compound_full_device.cuh.
//
// Design:
//   * one CUDA block per job; blocks are independent, so a batch is a plain
//     grid launch with no inter-block synchronisation;
//   * the blend and skip kernels are pure per-sample loops;
//   * OPFL stages the w*h difference and gradient source image in 32 KiB of
//     shared memory (two int arrays, the M2 64x64 maximum) and derives the
//     gradients inside the 8x8/4x4 solve;
//   * refine-MV reduces each candidate SAD in shared memory;
//   * no allocation, no implicit synchronization, no global mutable state.

#include "compound_full_device.cuh"

namespace av2enc::cuda::compound_full {
namespace {

enum : int { kThreads = 128 };

__global__ void blend_kernel(const BlendJob* jobs, const std::uint8_t* cache) {
  BlendJob job = jobs[blockIdx.x];
  if (cache != nullptr) {
    job.wedge_cache = cache;
  }
  detail::run_blend_job_device(job, static_cast<int>(threadIdx.x),
                               static_cast<int>(blockDim.x));
}

__global__ void skip_kernel(const SkipJob* jobs) {
  __shared__ int s_mask[kMaxMaskSamples];
  detail::run_skip_job_device(jobs[blockIdx.x], static_cast<int>(threadIdx.x),
                              static_cast<int>(blockDim.x), s_mask);
}

__global__ void opfl_kernel(OpflJob* jobs) {
  __shared__ int s_tmp[kMaxMaskSamples];
  __shared__ int s_pdiff[kMaxMaskSamples];
  detail::run_opfl_job_device(jobs[blockIdx.x], static_cast<int>(threadIdx.x),
                              static_cast<int>(blockDim.x), s_tmp, s_pdiff);
}

__global__ void refine_search_kernel(const RefineSearchJob* jobs,
                                     RefineResult* results) {
  __shared__ std::int64_t s_sad[kThreads];
  detail::run_refine_search_device(
      jobs[blockIdx.x], results + blockIdx.x, static_cast<int>(threadIdx.x),
      static_cast<int>(blockDim.x), s_sad);
}

__global__ void refine_predict_kernel(const RefinePredictJob* jobs) {
  detail::run_refine_predict_device(
      jobs[blockIdx.x], static_cast<int>(threadIdx.x),
      static_cast<int>(blockDim.x));
}

__global__ void wedge_cache_kernel(std::uint8_t* cache) {
  detail::wedge_cache_build_slot_device(cache, static_cast<int>(blockIdx.x),
                                        static_cast<int>(threadIdx.x),
                                        static_cast<int>(blockDim.x));
}

bool in_block_range(int v) { return v > 0 && v <= kMaxBlockDim; }

}  // namespace

cudaError_t wedge_cache_build(CompoundFullWorkspace& workspace,
                              cudaStream_t stream) {
  if (workspace.wedge_cache == nullptr) {
    return cudaSuccess;
  }
  wedge_cache_kernel<<<16, kThreads, 0, stream>>>(workspace.wedge_cache);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

cudaError_t blend_batch(const BlendJob* d_jobs, int job_count,
                        cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && d_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) {
    return cudaSuccess;
  }
  blend_kernel<<<job_count, kThreads, 0, stream>>>(d_jobs, nullptr);
  return cudaGetLastError();
}

cudaError_t blend_batch_ws(CompoundFullWorkspace& workspace,
                           const BlendJob* host_jobs, int job_count,
                           cudaStream_t stream) {
  if (host_jobs == nullptr || job_count <= 0 ||
      job_count > workspace.max_jobs || workspace.jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  const cudaError_t copy = cudaMemcpy(
      workspace.jobs, host_jobs,
      static_cast<std::size_t>(job_count) * sizeof(BlendJob),
      cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) {
    return copy;
  }
  blend_kernel<<<job_count, kThreads, 0, stream>>>(workspace.jobs,
                                                   workspace.wedge_cache);
  return cudaGetLastError();
}

cudaError_t skip_batch(const SkipJob* d_jobs, int job_count,
                       cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && d_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) {
    return cudaSuccess;
  }
  skip_kernel<<<job_count, kThreads, 0, stream>>>(d_jobs);
  return cudaGetLastError();
}

cudaError_t skip_batch_ws(CompoundFullWorkspace& workspace,
                          const SkipJob* host_jobs, int job_count,
                          cudaStream_t stream) {
  if (host_jobs == nullptr || job_count <= 0 ||
      job_count > workspace.max_jobs || workspace.skip_jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  const cudaError_t copy = cudaMemcpy(
      workspace.skip_jobs, host_jobs,
      static_cast<std::size_t>(job_count) * sizeof(SkipJob),
      cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) {
    return copy;
  }
  skip_kernel<<<job_count, kThreads, 0, stream>>>(workspace.skip_jobs);
  return cudaGetLastError();
}

cudaError_t opfl_batch(OpflJob* d_jobs, int job_count, cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && d_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) {
    return cudaSuccess;
  }
  opfl_kernel<<<job_count, kThreads, 0, stream>>>(d_jobs);
  return cudaGetLastError();
}

cudaError_t opfl_batch_ws(CompoundFullWorkspace& workspace,
                          const OpflJob* host_jobs, int job_count,
                          cudaStream_t stream) {
  if (host_jobs == nullptr || job_count <= 0 ||
      job_count > workspace.max_jobs || workspace.opfl_jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  const cudaError_t copy = cudaMemcpy(
      workspace.opfl_jobs, host_jobs,
      static_cast<std::size_t>(job_count) * sizeof(OpflJob),
      cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) {
    return copy;
  }
  opfl_kernel<<<job_count, kThreads, 0, stream>>>(workspace.opfl_jobs);
  return cudaGetLastError();
}

cudaError_t refine_search_batch(const RefineSearchJob* d_jobs,
                                RefineResult* d_results, int job_count,
                                cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && (d_jobs == nullptr ||
                                          d_results == nullptr))) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) {
    return cudaSuccess;
  }
  refine_search_kernel<<<job_count, kThreads, 0, stream>>>(d_jobs, d_results);
  return cudaGetLastError();
}

cudaError_t refine_search_batch_ws(CompoundFullWorkspace& workspace,
                                   const RefineSearchJob* host_jobs,
                                   int job_count, cudaStream_t stream) {
  if (host_jobs == nullptr || job_count <= 0 ||
      job_count > workspace.max_jobs || workspace.refine_search_jobs == nullptr ||
      workspace.refine_results == nullptr) {
    return cudaErrorInvalidValue;
  }
  const cudaError_t copy = cudaMemcpy(
      workspace.refine_search_jobs, host_jobs,
      static_cast<std::size_t>(job_count) * sizeof(RefineSearchJob),
      cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) {
    return copy;
  }
  refine_search_kernel<<<job_count, kThreads, 0, stream>>>(
      workspace.refine_search_jobs, workspace.refine_results);
  return cudaGetLastError();
}

cudaError_t refine_predict_batch(const RefinePredictJob* d_jobs, int job_count,
                                 cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && d_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) {
    return cudaSuccess;
  }
  refine_predict_kernel<<<job_count, kThreads, 0, stream>>>(d_jobs);
  return cudaGetLastError();
}

cudaError_t refine_predict_batch_ws(CompoundFullWorkspace& workspace,
                                    const RefinePredictJob* host_jobs,
                                    int job_count, cudaStream_t stream) {
  if (host_jobs == nullptr || job_count <= 0 ||
      job_count > workspace.max_jobs ||
      workspace.refine_predict_jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  const cudaError_t copy = cudaMemcpy(
      workspace.refine_predict_jobs, host_jobs,
      static_cast<std::size_t>(job_count) * sizeof(RefinePredictJob),
      cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) {
    return copy;
  }
  refine_predict_kernel<<<job_count, kThreads, 0, stream>>>(
      workspace.refine_predict_jobs);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Host-side validity checks
// ---------------------------------------------------------------------------

bool blend_job_valid(const BlendJob& job) {
  if (job.dst == nullptr || job.dst_stride <= 0 || !in_block_range(job.rect_w) ||
      !in_block_range(job.rect_h) || job.rect_x < 0 || job.rect_y < 0) {
    return false;
  }
  if (job.bit_depth != 8 && job.bit_depth != 10) {
    return false;
  }
  if (job.type < kCompoundWedge || job.type > kCompoundIntra) {
    return false;
  }
  if (job.bsize < 0 || job.bsize >= detail::kCfBlockSizes) {
    return false;
  }
  if (job.is_inter_intra) {
    if (job.intra == nullptr || job.intra_stride <= 0) {
      return false;
    }
    if (job.interintra_mode < 0 || job.interintra_mode >= detail::kCfInterIntraModes) {
      return false;
    }
  } else {
    if (job.p0 == nullptr || job.p1 == nullptr || job.p0_stride <= 0 ||
        job.p1_stride <= 0) {
      return false;
    }
    if (job.type == kCompoundAverage &&
        (job.cwp_weight < -4 || job.cwp_weight > 20)) {
      return false;
    }
    if (job.type == kCompoundWedge &&
        (!detail::cf_bsize_has_wedge(job.bsize) || job.wedge_index < 0 ||
         job.wedge_index >= detail::kCfWedgeTypes)) {
      return false;
    }
  }
  if (job.mask_out != nullptr && job.mask_out_stride < job.rect_w) {
    return false;
  }
  return true;
}

bool skip_job_valid(const SkipJob& job) {
  if (job.bit_depth != 8 && job.bit_depth != 10) {
    return false;
  }
  if (job.w <= 0 || job.h <= 0 || job.w > kMaxBlockDim ||
      job.h > kMaxBlockDim || (job.w & 1) != 0 || (job.h & 1) != 0 ||
      (job.x & 1) != 0 || (job.y & 1) != 0) {
    return false;
  }
  if (job.filter < 0 ||
      job.filter > static_cast<int>(mc::InterpFilter::kFourtapSmooth)) {
    return false;
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (job.dst[plane] == nullptr || job.dst_stride[plane] <= 0) {
      return false;
    }
    if (job.dst_stride[plane] < (job.w >> (plane > 0 ? 1 : 0))) {
      return false;
    }
    for (int list = 0; list < 2; ++list) {
      const RefPlane& ref = job.ref[list][plane];
      if (ref.data == nullptr || ref.stride <= 0 || ref.width <= 0 ||
          ref.height <= 0 || (ref.is_16bit != 0) != (job.bit_depth == 10)) {
        return false;
      }
    }
  }
  return true;
}

bool opfl_job_valid(const OpflJob& job) {
  if (job.bit_depth != 8 && job.bit_depth != 10) {
    return false;
  }
  if (job.w <= 0 || job.h <= 0 || job.w > kMaxBlockDim ||
      job.h > kMaxBlockDim || (job.n != 4 && job.n != 8) ||
      job.w % job.n != 0 || job.h % job.n != 0) {
    return false;
  }
  if (job.p0 == nullptr || job.p1 == nullptr || job.p0_stride < job.w ||
      job.p1_stride < job.w || (job.p0_16 != 0) != (job.bit_depth == 10) ||
      (job.p1_16 != 0) != (job.bit_depth == 10)) {
    return false;
  }
  if (job.opfl_mvs == nullptr || job.mv_deltas == nullptr ||
      job.opfl_stride4 <= 0 || job.delta_stride4 <= 0) {
    return false;
  }
  if (job.opt_x < 0 || job.opt_y < 0 || (job.opt_x & 3) != 0 ||
      (job.opt_y & 3) != 0) {
    return false;
  }
  return true;
}

bool refine_search_job_valid(const RefineSearchJob& job) {
  if (job.bit_depth != 8 && job.bit_depth != 10) {
    return false;
  }
  if (job.w <= 0 || job.h <= 0 || job.w > kMaxBlockDim ||
      job.h > kMaxBlockDim || job.num_refs < 2) {
    return false;
  }
  for (int list = 0; list < 2; ++list) {
    if (job.ref[list].data == nullptr || job.ref[list].stride <= 0 ||
        job.ref[list].width <= 0 || job.ref[list].height <= 0 ||
        (job.ref[list].is_16bit != 0) != (job.bit_depth == 10)) {
      return false;
    }
  }
  return true;
}

bool refine_predict_job_valid(const RefinePredictJob& job) {
  if (job.bit_depth != 8 && job.bit_depth != 10) {
    return false;
  }
  if (job.w <= 0 || job.h <= 0 || job.w > kMaxBlockDim ||
      job.h > kMaxBlockDim || job.dst == nullptr || job.dst_stride <= 0) {
    return false;
  }
  for (int list = 0; list < 2; ++list) {
    if (job.ref[list].data == nullptr || job.ref[list].stride <= 0 ||
        job.ref[list].width <= 0 || job.ref[list].height <= 0 ||
        (job.ref[list].is_16bit != 0) != (job.bit_depth == 10)) {
      return false;
    }
  }
  return true;
}

}  // namespace av2enc::cuda::compound_full
