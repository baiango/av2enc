// Device-side AV2 lossless inter residual transforms (§7.15.2.2 / §7.15.4).
//
// Shared by the CUDA kernels and the host-emulation parity test.  Each block
// routine is called by every thread of one CUDA block; the job descriptor is
// uniform across the block and the routines synchronise internally.  With
// `blockDim.x == 1` (the emulation harness) they degrade to the exact serial
// CPU algorithm, which is how the no-CUDA parity test exercises this header.
//
// Buffers:
//   * `scratch` has kMaxTransformSamples int32 entries (shared memory in the
//     kernels, a plain array in emulation);
//   * `residual` (reconstruction only) has kMaxTransformSamples int32 entries.
//
// The arithmetic is the CPU reference's integer arithmetic, including the
// arithmetic `>>` floor in the WHT lifting step, so odd (a - d) values match
// bit for bit.

#ifndef AV2ENC_CUDA_LOSSLESS_INTER_LOSSLESS_INTER_DEVICE_CUH_
#define AV2ENC_CUDA_LOSSLESS_INTER_LOSSLESS_INTER_DEVICE_CUH_

#include "lossless_inter.h"
#include "lossless_inter_math.h"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::lossless_inter {

// Encoder residual -> transform levels:
//   levels = WHT(difference(residual))   when dct_dct != 0
//   levels =      difference(residual)   when dct_dct == 0 (IDTX / FSC)
// Identical to `forward_lossless_block(dpcm_difference(...))` in the CPU
// reference.  In-place (`levels == residual`) is safe: the difference phase has
// already copied the source into `scratch` before `levels` is written.
__device__ inline void li_forward_block_device(const ForwardJob& job, int tid,
                                               int nthr, int* scratch) {
  const int w = job.w;
  const int h = job.h;
  if (w <= 0 || h <= 0 || nthr <= 0) return;
  if (w > kMaxTransformDim || h > kMaxTransformDim) return;
  const int n = w * h;

  // DPCM first difference (§7.15.4 encoder side).  The first row (V) or first
  // column (H) is unchanged, exactly like the CPU helper.
  for (int idx = tid; idx < n; idx += nthr) {
    int value = job.residual[idx];
    if (job.dpcm != 0) {
      if (job.vertical != 0) {
        if (idx >= w) value -= job.residual[idx - w];
      } else if ((idx % w) != 0) {
        value -= job.residual[idx - 1];
      }
    }
    scratch[idx] = value;
  }
  __syncthreads();

  if (job.dct_dct == 0) {
    for (int idx = tid; idx < n; idx += nthr) {
      job.levels[idx] = scratch[idx];
    }
    return;
  }

  // Column pass first: the decoder does inverse rows then inverse columns, so
  // the exact forward transform runs columns then rows.
  for (int c = tid; c < w; c += nthr) {
    for (int r = 0; r < h; r += 4) {
      int in[4];
      int out[4];
      for (int k = 0; k < 4; ++k) in[k] = scratch[(r + k) * w + c];
      li_wht4_forward(in, out);
      for (int k = 0; k < 4; ++k) job.levels[(r + k) * w + c] = out[k];
    }
  }
  __syncthreads();

  // Row pass.  Each row is owned by one thread, so the in-place update of
  // `levels` is race free.
  for (int r = tid; r < h; r += nthr) {
    for (int c = 0; c < w; c += 4) {
      int in[4];
      int out[4];
      for (int k = 0; k < 4; ++k) in[k] = job.levels[r * w + c + k];
      li_wht4_forward(in, out);
      for (int k = 0; k < 4; ++k) job.levels[r * w + c + k] = out[k];
    }
  }
}

// Decoder residual: inverse transform then optional DPCM cumulative sum.
//   residual = cumsum(inverse(levels))
// `levels`/`residual` may alias: the row pass copies `levels` into `scratch`
// before the column pass writes `residual`.
__device__ inline void li_inverse_residual_device(
    const int* levels, int* residual, int w, int h, int dct_dct, int dpcm,
    int vertical, int tid, int nthr, int* scratch) {
  if (w <= 0 || h <= 0 || nthr <= 0) return;
  if (w > kMaxTransformDim || h > kMaxTransformDim) return;
  const int n = w * h;

  if (dct_dct != 0) {
    // Row pass: Dequant = level * 8 (q = 0), inverse WHT with shift 3.
    for (int r = tid; r < h; r += nthr) {
      for (int c = 0; c < w; c += 4) {
        int in[4];
        int out[4];
        for (int k = 0; k < 4; ++k) in[k] = levels[r * w + c + k] * 8;
        li_wht4_inverse(in, 3, out);
        for (int k = 0; k < 4; ++k) scratch[r * w + c + k] = out[k];
      }
    }
    __syncthreads();
    // Column pass with shift 0.
    for (int c = tid; c < w; c += nthr) {
      for (int r = 0; r < h; r += 4) {
        int in[4];
        int out[4];
        for (int k = 0; k < 4; ++k) in[k] = scratch[(r + k) * w + c];
        li_wht4_inverse(in, 0, out);
        for (int k = 0; k < 4; ++k) residual[(r + k) * w + c] = out[k];
      }
    }
  } else {
    // IDTX / FSC: Dequant >> 3 == level at q = 0, so the levels are copied.
    for (int idx = tid; idx < n; idx += nthr) {
      residual[idx] = levels[idx];
    }
  }
  __syncthreads();

  if (dpcm != 0) {
    // Each line is independent and owned by one thread, so the in-place
    // cumulative sum matches the CPU helper.
    if (vertical != 0) {
      for (int c = tid; c < w; c += nthr) {
        for (int r = 1; r < h; ++r) {
          residual[r * w + c] += residual[(r - 1) * w + c];
        }
      }
    } else {
      for (int r = tid; r < h; r += nthr) {
        for (int c = 1; c < w; ++c) {
          residual[r * w + c] += residual[r * w + c - 1];
        }
      }
    }
  }
  __syncthreads();
}

// Full reconstruction: inverse + DPCM + Clip1(pred + residual).  `scratch` and
// `residual` are each kMaxTransformSamples int32 entries.
__device__ inline void li_reconstruct_block_device(const ReconJob& job, int tid,
                                                   int nthr, int* scratch,
                                                   int* residual) {
  li_inverse_residual_device(job.levels, residual, job.w, job.h, job.dct_dct,
                             job.dpcm, job.vertical, tid, nthr, scratch);
  const int w = job.w;
  const int h = job.h;
  if (w <= 0 || h <= 0) return;
  const int n = w * h;
  for (int idx = tid; idx < n; idx += nthr) {
    const int y = idx / w;
    const int x = idx - y * w;
    const std::size_t pred_index =
        static_cast<std::size_t>(y) *
            static_cast<std::size_t>(job.pred_stride) +
        static_cast<std::size_t>(x);
    const int p = static_cast<int>(job.pred[pred_index]);
    const int value = li_clip3(0, 255, p + residual[idx]);
    const std::size_t recon_index =
        static_cast<std::size_t>(y) *
            static_cast<std::size_t>(job.recon_stride) +
        static_cast<std::size_t>(x);
    job.recon[recon_index] = static_cast<std::uint8_t>(value);
  }
}

// Standalone DPCM first difference (out of place).
__device__ inline void li_dpcm_difference_device(const DpcmJob& job, int tid,
                                                 int nthr) {
  const int w = job.w;
  const int h = job.h;
  if (w <= 0 || h <= 0 || nthr <= 0) return;
  if (w > kMaxTransformDim || h > kMaxTransformDim) return;
  const int n = w * h;
  for (int idx = tid; idx < n; idx += nthr) {
    int value = job.in[idx];
    if (job.vertical != 0) {
      if (idx >= w) value -= job.in[idx - w];
    } else if ((idx % w) != 0) {
      value -= job.in[idx - 1];
    }
    job.out[idx] = value;
  }
}

// Standalone DPCM cumulative sum (in place or out of place).
__device__ inline void li_dpcm_cumsum_device(const DpcmJob& job, int tid,
                                             int nthr) {
  const int w = job.w;
  const int h = job.h;
  if (w <= 0 || h <= 0 || nthr <= 0) return;
  if (w > kMaxTransformDim || h > kMaxTransformDim) return;
  const int n = w * h;
  for (int idx = tid; idx < n; idx += nthr) job.out[idx] = job.in[idx];
  __syncthreads();
  if (job.vertical != 0) {
    for (int c = tid; c < w; c += nthr) {
      for (int r = 1; r < h; ++r) {
        job.out[r * w + c] += job.out[(r - 1) * w + c];
      }
    }
  } else {
    for (int r = tid; r < h; r += nthr) {
      for (int c = 1; c < w; ++c) {
        job.out[r * w + c] += job.out[r * w + c - 1];
      }
    }
  }
}

}  // namespace av2enc::cuda::lossless_inter

#endif  // AV2ENC_CUDA_LOSSLESS_INTER_LOSSLESS_INTER_DEVICE_CUH_
