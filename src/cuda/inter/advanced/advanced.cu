// CUDA kernels for the advanced inter predictors.
//
// Design notes:
//  * One CUDA block per job. Blocks are independent, so a batch is a plain
//    grid launch with no inter-block synchronisation.
//  * Warp kernels stage the 8x8 (or 4x4) intermediate rows in shared memory:
//    15*8 ints = 480 B for block warp, 64*4 ints = 1 KiB for extended warp.
//  * Every numeric step calls the shared `AV2ENC_ADV_HD` helpers, so the result
//    is bit-identical to the CPU reference by construction, not by testing.

#include "advanced.h"

namespace av2enc::cuda::advinter {
namespace {

using namespace av2enc::inter::adv;

constexpr int kThreads = 128;

__device__ inline void compound_job_device(const CompoundJob& job) {
  const int w = job.rect.w;
  const int h = job.rect.h;
  const int total = w * h;
  for (int index = static_cast<int>(threadIdx.x); index < total;
       index += static_cast<int>(blockDim.x)) {
    const int i = index / w;
    const int j = index - i * w;
    const std::uint8_t* intra_row =
        job.intra ? job.intra + static_cast<std::ptrdiff_t>(i) * job.intra_stride : nullptr;
    const std::int16_t* p0_row =
        job.p0 ? job.p0 + static_cast<std::ptrdiff_t>(i) * job.p0_stride : nullptr;
    const std::int16_t* p1_row =
        job.p1 ? job.p1 + static_cast<std::ptrdiff_t>(i) * job.p1_stride : nullptr;
    std::uint8_t* dst_row =
        job.dst + static_cast<std::ptrdiff_t>(job.rect.y + i) * job.dst_stride + job.rect.x;
    const int pred0 = p0_row ? p0_row[j] : 0;
    const int pred1 = p1_row ? p1_row[j] : 0;
    const int intra_pred = intra_row ? intra_row[j] : 0;
    dst_row[j] = static_cast<std::uint8_t>(
        compound_sample_at(job, i, j, pred0, pred1, intra_pred, dst_row[j]));
  }
  if (job.mask_out != nullptr && job.plane == 0) {
    for (int index = static_cast<int>(threadIdx.x); index < total;
         index += static_cast<int>(blockDim.x)) {
      const int i = index / w;
      const int j = index - i * w;
      job.mask_out[static_cast<std::ptrdiff_t>(i) * job.mask_out_stride + j] =
          static_cast<std::uint8_t>(compound_mask_out_at(job, i, j));
    }
  }
}

__global__ void compound_kernel(const CompoundJob* jobs) { compound_job_device(jobs[blockIdx.x]); }

__global__ void compound_one_kernel(CompoundJob job) { compound_job_device(job); }

__device__ inline void warp_job_device(const WarpJob& job) {
  __shared__ std::int32_t intermediate[kBlockWarpRows * kBlockWarpCols];
  const int h = job.geom.rect.h;
  const int w = job.geom.rect.w;
  const int inter_round1 = job.compound ? kInterRound1Compound : kInterRound1Single;
  for (int i8 = 0; i8 <= (h - 1) >> 3; ++i8) {
    for (int j8 = 0; j8 <= (w - 1) >> 3; ++j8) {
      BlockWarpState state{};
      if (!block_warp_prepare(job.params, job.geom.rect.x, job.geom.rect.y, i8, j8, job.geom.sub_x,
                              job.geom.sub_y, &state)) {
        continue;
      }
      __syncthreads();
      for (int t = static_cast<int>(threadIdx.x); t < kBlockWarpRows * kBlockWarpCols;
           t += static_cast<int>(blockDim.x)) {
        const int i1 = t / kBlockWarpCols - 7;
        const int i2 = t % kBlockWarpCols - 4;
        intermediate[t] = block_warp_intermediate_value(job.ref, state, i1, i2);
      }
      __syncthreads();
      for (int t = static_cast<int>(threadIdx.x); t < 64; t += static_cast<int>(blockDim.x)) {
        const int i1 = t / 8 - 4;
        const int i2 = t % 8 - 4;
        const int local_y = i8 * 8 + i1 + 4;
        const int local_x = j8 * 8 + i2 + 4;
        if (local_y < 0 || local_y >= h || local_x < 0 || local_x >= w) {
          continue;
        }
        job.dst[static_cast<std::ptrdiff_t>(job.geom.dst_y + local_y) * job.dst_stride +
                job.geom.dst_x + local_x] =
            block_warp_store_value(state, i1, i2, intermediate, inter_round1, job.compound == 0);
      }
    }
  }
}

__global__ void warp_kernel(const WarpJob* jobs) { warp_job_device(jobs[blockIdx.x]); }

__device__ inline void ext_warp_job_device(const WarpJob& job) {
  __shared__ std::int32_t intermediate[kExtWarpRows * kExtWarpCols];
  ExtWarpContext ctx{};
  ctx.plane = job.geom.plane;
  ctx.sub_x = job.geom.sub_x;
  ctx.sub_y = job.geom.sub_y;
  ctx.block_w = job.geom.rect.w;
  ctx.block_h = job.geom.rect.h;
  ctx.mi_col = job.geom.mi_col;
  ctx.mi_row = job.geom.mi_row;
  ctx.bw4 = job.geom.bw4;
  ctx.bh4 = job.geom.bh4;
  ctx.mi_cols = job.geom.mi_cols;
  ctx.mi_rows = job.geom.mi_rows;
  ctx.scaled = warp_ref_is_scaled(job.geom);
  ctx.ref_w = job.geom.ref_w;
  ctx.ref_h = job.geom.ref_h;
  ctx.frame_w = job.geom.frame_w;
  ctx.frame_h = job.geom.frame_h;
  ctx.inter_round1 = job.compound ? kInterRound1Compound : kInterRound1Single;
  ctx.dst_x = job.geom.dst_x;
  ctx.dst_y = job.geom.dst_y;
  ctx.clip_output = job.compound == 0;

  for (int i4 = 0; i4 < (job.geom.rect.h >> 2); ++i4) {
    for (int j4 = 0; j4 < (job.geom.rect.w >> 2); ++j4) {
      ExtWarpSection section{};
      if (!ext_warp_prepare(job.params, job.ref, ctx, job.geom.rect.x, job.geom.rect.y, i4, j4,
                            &section)) {
        continue;
      }
      __syncthreads();
      const int entries = section.rows * kExtWarpCols;
      for (int t = static_cast<int>(threadIdx.x); t < entries; t += static_cast<int>(blockDim.x)) {
        intermediate[t] = ext_warp_intermediate_value(job.ref, section, t);
      }
      __syncthreads();
      for (int t = static_cast<int>(threadIdx.x); t < 16; t += static_cast<int>(blockDim.x)) {
        const int k_local = t / 4;
        const int l_local = t % 4;
        const int k = section.scaled ? k_local : k_local - 2;
        const int l = section.scaled ? l_local : l_local - 2;
        const int local_y = section.scaled ? i4 * 4 + k : i4 * 4 + k + 2;
        const int local_x = section.scaled ? j4 * 4 + l : j4 * 4 + l + 2;
        if (local_y < 0 || local_y >= ctx.block_h || local_x < 0 || local_x >= ctx.block_w) {
          continue;
        }
        job.dst[static_cast<std::ptrdiff_t>(ctx.dst_y + local_y) * job.dst_stride + ctx.dst_x +
                local_x] =
            ext_warp_store_value(section, k, l, intermediate, ctx.inter_round1, ctx.clip_output);
      }
    }
  }
}

__global__ void ext_warp_kernel(const WarpJob* jobs) { ext_warp_job_device(jobs[blockIdx.x]); }

__device__ inline void bawp_job_device(const BawpGpuJob& gpu_job) {
  // Plain ints, not BawpFit: CUDA shared variables must be trivially
  // default-constructible, and BawpFit has default member initialisers.
  __shared__ int shared_alpha;
  __shared__ int shared_beta;
  const BawpJob& job = gpu_job.job;
  if (threadIdx.x == 0) {
    const int dy = to_fullmv(job.mv[0]);
    const int dx = to_fullmv(job.mv[1]);
    const int ref_y = (job.ctx.mi_row * kMiSize + dy) >> job.ctx.sub_y;
    const int ref_x = (job.ctx.mi_col * kMiSize + dx) >> job.ctx.sub_x;
    const PlaneView& template_ref = job.morph_pred ? job.cur : job.prev;
    const BawpFit fit =
        bawp_fit(job.ctx, job.cur, template_ref, job.rect.x, job.rect.y, job.rect.w, job.rect.h,
                 ref_x, ref_y, job.bawp_alpha, job.explicit_bawp && !job.morph_pred,
                 job.explicit_bawp_scale, job.list_index, job.first_ref_dist);
    shared_alpha = fit.alpha;
    shared_beta = fit.beta;
    if (gpu_job.fit_out != nullptr) {
      *gpu_job.fit_out = fit;
    }
  }
  __syncthreads();
  // Every sample is independent, so this is safe to interleave.
  for (int i = static_cast<int>(threadIdx.x); i < job.rect.h; i += static_cast<int>(blockDim.x)) {
    std::uint8_t* row = job.dst.data + static_cast<std::ptrdiff_t>(job.rect.y + i) * job.dst.stride;
    for (int j = 0; j < job.rect.w; ++j) {
      const int x = job.rect.x + j;
      row[x] = static_cast<std::uint8_t>(bawp_apply_sample(row[x], shared_alpha, shared_beta, 8));
    }
  }
}

__global__ void bawp_kernel(const BawpGpuJob* jobs) { bawp_job_device(jobs[blockIdx.x]); }

__device__ inline void obmc_job_device(const ObmcGpuJob& job) {
  ObmcConfig config = job.config;
  const int overlap = obmc_overlap_for(config);
  const int total = config.rect.w * config.rect.h;
  for (int index = static_cast<int>(threadIdx.x); index < total;
       index += static_cast<int>(blockDim.x)) {
    const int i = index / config.rect.w;
    const int j = index - i * config.rect.w;
    std::uint8_t* row = config.dst +
                        static_cast<std::ptrdiff_t>(config.rect.y + i) * config.dst_stride +
                        config.rect.x;
    const int own = row[j];
    const int above =
        job.above ? job.above[static_cast<std::ptrdiff_t>(i) * job.above_stride + j] : own;
    const int left =
        job.left ? job.left[static_cast<std::ptrdiff_t>(i) * job.left_stride + j] : own;
    row[j] = static_cast<std::uint8_t>(obmc_sample_at(own, above, left, i, j, overlap));
  }
}

__global__ void obmc_kernel(const ObmcGpuJob* jobs) { obmc_job_device(jobs[blockIdx.x]); }

}  // namespace

cudaError_t compound_batch(const CompoundJob* jobs, int job_count, cudaStream_t stream) {
  if (job_count <= 0) {
    return cudaSuccess;
  }
  if (jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  compound_kernel<<<job_count, kThreads, 0, stream>>>(jobs);
  return cudaGetLastError();
}

cudaError_t compound_one(const CompoundJob& job, cudaStream_t stream) {
  compound_one_kernel<<<1, kThreads, 0, stream>>>(job);
  return cudaGetLastError();
}

cudaError_t warp_batch(const WarpJob* jobs, int job_count, cudaStream_t stream) {
  if (job_count <= 0) {
    return cudaSuccess;
  }
  if (jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  warp_kernel<<<job_count, kThreads, 0, stream>>>(jobs);
  return cudaGetLastError();
}

cudaError_t ext_warp_batch(const WarpJob* jobs, int job_count, cudaStream_t stream) {
  if (job_count <= 0) {
    return cudaSuccess;
  }
  if (jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  ext_warp_kernel<<<job_count, kThreads, 0, stream>>>(jobs);
  return cudaGetLastError();
}

cudaError_t bawp_batch(const BawpGpuJob* jobs, int job_count, cudaStream_t stream) {
  if (job_count <= 0) {
    return cudaSuccess;
  }
  if (jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  bawp_kernel<<<job_count, kThreads, 0, stream>>>(jobs);
  return cudaGetLastError();
}

cudaError_t obmc_batch(const ObmcGpuJob* jobs, int job_count, cudaStream_t stream) {
  if (job_count <= 0) {
    return cudaSuccess;
  }
  if (jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  obmc_kernel<<<job_count, kThreads, 0, stream>>>(jobs);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::advinter
