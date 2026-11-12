// Batched M1 transform / quantize / inverse kernels.
// See still_tx.h for the contract; the numeric reference is
// `src/encoder/tx_variants/tx_variants.cpp` driven exactly like
// `TileEncoder::compute_tx` in `src/encoder/m1/still_encoder.cpp`.
#include "cuda/pipeline/still/tx/still_tx.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "codec/transform/variants/tx_tables_dev.cuh"

namespace av2enc::cuda {
namespace {

using namespace av2enc::txv::dev;

constexpr int kQuantStep = av2enc::txv::kQuantStep;
constexpr int kQuantTableBits = av2enc::txv::kQuantTableBits;
constexpr int kMaxElems = av2enc::txv::kMaxAdjTxElems;
constexpr int kThreads = 256;
// Static part of the dynamic shared allocation: two 32x32 int16 kernels.
constexpr int kKernelBytes = 2 * kMaxElems * static_cast<int>(sizeof(std::int16_t));

__device__ __forceinline__ int round2_dev(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

__device__ __forceinline__ int clip3_dev(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

// §7.15.2.1 kernel selection: IDT has no matrix, length 32 always uses the
// DCT kernel (so 32x32 and every adjusted 32 dimension alias DCT).
__device__ __forceinline__ const std::int16_t* kernel_dev(int type, int n) {
  switch (type) {
    case 1:  // IDT
      return nullptr;
    case 2:  // ADST
      if (n == 4) return &kAdst4[0][0];
      if (n == 8) return &kAdst8[0][0];
      if (n == 16) return &kAdst16[0][0];
      return &kDct32[0][0];
    case 3:  // FDST / FLIPADST
      if (n == 4) return &kFdst4[0][0];
      if (n == 8) return &kFdst8[0][0];
      if (n == 16) return &kFdst16[0][0];
      return &kDct32[0][0];
    default:  // DCT
      if (n == 4) return &kDct4[0][0];
      if (n == 8) return &kDct8[0][0];
      if (n == 16) return &kDct16[0][0];
      return &kDct32[0][0];
  }
}

__device__ __forceinline__ int dequant_dev(int qc, int denom) {
  const int abs_qc = qc < 0 ? -qc : qc;
  const int dq_high = (abs_qc * kQuantStep) & 0xFFFFFF;
  const int dq = round2_dev(dq_high, kQuantTableBits) / denom;
  // §7.14.4: Dequant = Clip3(-(1 << (7 + BitDepth)), ...).
  return clip3_dev(-(1 << 15), (1 << 15) - 1, qc < 0 ? -dq : dq);
}

__device__ __forceinline__ int prescale_dev(int value, bool odd_rect) {
  if (!odd_rect) {
    return value;
  }
  const int scaled = static_cast<int>(static_cast<long long>(value) * 2896LL);
  return round2_dev(scaled, 12);
}

__device__ __forceinline__ int log2_pos(int value) {
  int n = 0;
  while ((1 << n) < value) {
    ++n;
  }
  return n;
}

// Position `c` in the coefficient scan built by `build_scan()` in
// `still_encoder.cpp`.  That loop walks anti-diagonals in increasing order and
// descends the row within each diagonal; this is the closed form, so each
// thread resolves its own index without a serial pass over the block.
__device__ __forceinline__ int scan_pos(int c, int w, int h) {
  int diagonal = 0;
  int rem = c;
  for (;;) {
    const int lo = diagonal - (w - 1) > 0 ? diagonal - (w - 1) : 0;
    const int hi = diagonal < h - 1 ? diagonal : h - 1;
    const int len = hi - lo + 1;
    if (rem < len) {
      const int row = hi - rem;
      const int col = diagonal - row;
      return row * w + col;
    }
    rem -= len;
    ++diagonal;
  }
}

__device__ __forceinline__ bool valid_params_dev(const TxParams& p) {
  const int w = p.w;
  const int h = p.h;
  const bool dim_ok = (w == 4 || w == 8 || w == 16 || w == 32) &&
                      (h == 4 || h == 8 || h == 16 || h == 32);
  if (!dim_ok) return false;
  if (p.coded_w < w || p.coded_h < h || p.coded_w > 64 || p.coded_h > 64) {
    return false;
  }
  if (p.coded_w % w != 0 || p.coded_h % h != 0) return false;
  if (p.row_type < 0 || p.row_type > 3 || p.col_type < 0 || p.col_type > 3) {
    return false;
  }
  if (p.row_shift <= 0 || p.row_shift >= 31 || p.col_shift <= 0 ||
      p.col_shift >= 31) {
    return false;
  }
  if (p.dq_denom != 1 && p.dq_denom != 2 && p.dq_denom != 4) return false;
  if (p.log2_w < 2 || p.log2_w > 6 || p.log2_h < 2 || p.log2_h > 6) {
    return false;
  }
  return p.scale_w > 0.0 && p.scale_h > 0.0 && p.quant_factor > 0.0 &&
         p.rect_prescale > 0.0;
}

__device__ __forceinline__ void reject(const StillTxJob& job,
                                       std::int32_t* __restrict__ eob,
                                       int* __restrict__ error_count) {
  if (error_count != nullptr) {
    atomicAdd(error_count, 1);
  }
  if (eob != nullptr) {
    eob[job.eob_offset] = -1;
  }
}

// One block per job.  Shared layout (all sized from `smem_elems`):
//   int16  s_res / s_adj  [elems]  adjusted residual
//   double s_tmp          [elems]  forward intermediate (reused as s_inter)
//   int32  s_dq           [elems]  levels then dequantized coefficients
//   int16  s_kw, s_kh     [1024]   transform kernels
__global__ void still_tx_kernel(const StillTxJob* __restrict__ jobs,
                             const std::uint8_t* __restrict__ src,
                             const std::uint8_t* __restrict__ pred,
                             const std::int16_t* __restrict__ residual_in,
                             std::int16_t* __restrict__ residual_adj,
                             double* __restrict__ coeff,
                             std::int32_t* __restrict__ levels,
                             std::int32_t* __restrict__ dequant,
                             std::int16_t* __restrict__ residual_out,
                             std::int32_t* __restrict__ eob,
                             int* __restrict__ error_count, int smem_elems) {
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ int s_eob;

  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  const StillTxJob job = jobs[blockIdx.x];
  const TxParams p = job.params;
  const int w = p.w;
  const int h = p.h;
  const int elems = w * h;

  if (!valid_params_dev(p) || elems > smem_elems) {
    if (tid == 0) {
      reject(job, eob, error_count);
    }
    return;
  }
  if (job.input == static_cast<std::uint8_t>(StillTxInput::kResidual)) {
    if (residual_in == nullptr) {
      if (tid == 0) reject(job, eob, error_count);
      return;
    }
  } else if (job.input == static_cast<std::uint8_t>(StillTxInput::kSourcePred)) {
    if (src == nullptr || pred == nullptr) {
      if (tid == 0) reject(job, eob, error_count);
      return;
    }
  } else {
    if (tid == 0) reject(job, eob, error_count);
    return;
  }

  // The hint is the batch maximum; pad to a 16-byte boundary so the double
  // staging area stays naturally aligned for any accepted job.
  const int elems_pad = (smem_elems + 7) & ~7;
  auto* s_adj = reinterpret_cast<std::int16_t*>(smem);
  auto* s_tmp = reinterpret_cast<double*>(smem + 2 * elems_pad);
  auto* s_dq = reinterpret_cast<std::int32_t*>(smem + 2 * elems_pad +
                                               8 * elems_pad);
  auto* s_kw = reinterpret_cast<std::int16_t*>(smem + 14 * elems_pad);
  auto* s_kh = s_kw + kMaxElems;
  auto* s_inter = reinterpret_cast<std::int32_t*>(s_tmp);

  const std::int16_t* kw = kernel_dev(static_cast<int>(p.row_type), w);
  const std::int16_t* kh = kernel_dev(static_cast<int>(p.col_type), h);
  if (kw != nullptr) {
    for (int t = tid; t < w * w; t += nthreads) {
      s_kw[t] = kw[t];
    }
  }
  if (kh != nullptr) {
    for (int t = tid; t < h * h; t += nthreads) {
      s_kh[t] = kh[t];
    }
  }

  // ---------------------------------------------------------------------
  // Coded residual -> adjusted residual (§7.15.4 downsample, Round2).
  // ---------------------------------------------------------------------
  const int sx = p.coded_w / w;
  const int sy = p.coded_h / h;
  const int shift = log2_pos(sx * sy);
  const int coded_w = p.coded_w;
  std::int16_t* d_adj =
      residual_adj != nullptr ? residual_adj + job.residual_adj_offset : nullptr;

  if (job.input == static_cast<std::uint8_t>(StillTxInput::kResidual)) {
    const std::int16_t* base = residual_in + job.residual_in_offset;
    for (int t = tid; t < elems; t += nthreads) {
      const int j = t / w;
      const int i = t - j * w;
      int sum = 0;
      for (int dy = 0; dy < sy; ++dy) {
        const int row = (j * sy + dy) * coded_w;
        for (int dx = 0; dx < sx; ++dx) {
          sum += static_cast<int>(base[row + i * sx + dx]);
        }
      }
      const std::int16_t value = static_cast<std::int16_t>(round2_dev(sum, shift));
      s_adj[t] = value;
      if (d_adj != nullptr) d_adj[t] = value;
    }
  } else {
    const std::uint8_t* block_pred = pred + job.pred_offset;
    for (int t = tid; t < elems; t += nthreads) {
      const int j = t / w;
      const int i = t - j * w;
      int sum = 0;
      for (int dy = 0; dy < sy; ++dy) {
        const int ay = j * sy + dy;
        for (int dx = 0; dx < sx; ++dx) {
          const int ax = i * sx + dx;
          int s;
          if (job.src_packed != 0) {
            s = static_cast<int>(src[job.src_offset + ay * coded_w + ax]);
          } else {
            int px = static_cast<int>(job.src_x) + ax;
            int py = static_cast<int>(job.src_y) + ay;
            px = clip3_dev(0, static_cast<int>(job.src_width) - 1, px);
            py = clip3_dev(0, static_cast<int>(job.src_height) - 1, py);
            s = static_cast<int>(
                src[job.src_offset + py * job.src_stride + px]);
          }
          sum += s - static_cast<int>(block_pred[ay * coded_w + ax]);
        }
      }
      const std::int16_t value = static_cast<std::int16_t>(round2_dev(sum, shift));
      s_adj[t] = value;
      if (d_adj != nullptr) d_adj[t] = value;
    }
  }
  __syncthreads();

  // ---------------------------------------------------------------------
  // Forward: row pass, column pass, rect pre-scale, quantize.
  // ---------------------------------------------------------------------
  if (kw == nullptr) {
    for (int t = tid; t < elems; t += nthreads) {
      s_tmp[t] = static_cast<double>(s_adj[t]);
    }
  } else {
    for (int t = tid; t < elems; t += nthreads) {
      const int i = t / w;
      const int v = t - i * w;
      const int row = i * w;
      double sum = 0.0;
      for (int j = 0; j < w; ++j) {
        sum = __dadd_rn(
            sum, __dmul_rn(static_cast<double>(s_kw[v * w + j]),
                           static_cast<double>(s_adj[row + j])));
      }
      s_tmp[t] = __dmul_rn(sum, p.scale_w);
    }
  }
  __syncthreads();

  for (int t = tid; t < elems; t += nthreads) {
    const int u = t / w;
    const int v = t - u * w;
    double out;
    if (kh == nullptr) {
      out = s_tmp[t];
    } else {
      double sum = 0.0;
      for (int i = 0; i < h; ++i) {
        sum = __dadd_rn(
            sum, __dmul_rn(static_cast<double>(s_kh[u * h + i]),
                           s_tmp[i * w + v]));
      }
      out = __dmul_rn(sum, p.scale_h);
    }
    if (p.rect_prescale != 1.0) {
      out = __dmul_rn(out, p.rect_prescale);
    }
    const std::int32_t level =
        static_cast<std::int32_t>(lround(__dmul_rn(out, p.quant_factor)));
    s_dq[t] = level;
    levels[job.levels_offset + t] = level;
    if (coeff != nullptr) coeff[job.coeff_offset + t] = out;
  }
  __syncthreads();

  // ---------------------------------------------------------------------
  // End of block over the scan order (build_scan() equivalent).
  // ---------------------------------------------------------------------
  if (tid == 0) s_eob = 0;
  __syncthreads();
  for (int c = tid; c < elems; c += nthreads) {
    if (s_dq[scan_pos(c, w, h)] != 0) {
      atomicMax(&s_eob, c + 1);
    }
  }
  __syncthreads();
  if (tid == 0 && eob != nullptr) {
    eob[job.eob_offset] = s_eob;
  }

  // ---------------------------------------------------------------------
  // Dequantize in place (levels are dead after the eob pass).
  // ---------------------------------------------------------------------
  for (int t = tid; t < elems; t += nthreads) {
    const std::int32_t value = dequant_dev(s_dq[t], p.dq_denom);
    s_dq[t] = value;
    if (dequant != nullptr) dequant[job.dequant_offset + t] = value;
  }
  __syncthreads();

  if (residual_out == nullptr) {
    return;
  }

  // ---------------------------------------------------------------------
  // Inverse: row pass, column pass, clips, coded-size duplication.
  // ---------------------------------------------------------------------
  const bool odd_rect = ((p.log2_w ^ p.log2_h) & 1) != 0;
  const int identity_w = kIdentityScaleDev[static_cast<int>(p.log2_w)];
  const int identity_h = kIdentityScaleDev[static_cast<int>(p.log2_h)];

  if (kw == nullptr) {
    for (int t = tid; t < elems; t += nthreads) {
      const int i = t / w;
      const int oj = t - i * w;
      const int row = i * w;
      const long long s = static_cast<long long>(
                              prescale_dev(s_dq[row + oj], odd_rect)) *
                          identity_w;
      s_inter[t] = clip3_dev(-(1 << 15), (1 << 15) - 1,
                             round2_dev(static_cast<int>(s), p.row_shift));
    }
  } else {
    for (int t = tid; t < elems; t += nthreads) {
      const int i = t / w;
      const int oj = t - i * w;
      const int row = i * w;
      long long sum = 0;
      for (int j = 0; j < w; ++j) {
        sum += static_cast<long long>(s_kw[j * w + oj]) *
               prescale_dev(s_dq[row + j], odd_rect);
      }
      s_inter[t] = clip3_dev(-(1 << 15), (1 << 15) - 1,
                             round2_dev(static_cast<int>(sum), p.row_shift));
    }
  }
  __syncthreads();

  std::int16_t* out_base = residual_out + job.residual_out_offset;
  if (kh == nullptr) {
    for (int t = tid; t < elems; t += nthreads) {
      const int oi = t / w;
      const int j = t - oi * w;
      const long long s =
          static_cast<long long>(s_inter[t]) * identity_h;
      const int value = clip3_dev(
          -256, 255, round2_dev(static_cast<int>(s), p.col_shift));
      const std::int16_t v16 = static_cast<std::int16_t>(value);
      for (int dy = 0; dy < sy; ++dy) {
        const int row = (oi * sy + dy) * coded_w;
        for (int dx = 0; dx < sx; ++dx) {
          out_base[row + j * sx + dx] = v16;
        }
      }
    }
  } else {
    for (int t = tid; t < elems; t += nthreads) {
      const int oi = t / w;
      const int j = t - oi * w;
      long long sum = 0;
      for (int i = 0; i < h; ++i) {
        sum += static_cast<long long>(s_kh[i * h + oi]) * s_inter[i * w + j];
      }
      const int value = clip3_dev(
          -256, 255, round2_dev(static_cast<int>(sum), p.col_shift));
      const std::int16_t v16 = static_cast<std::int16_t>(value);
      for (int dy = 0; dy < sy; ++dy) {
        const int row = (oi * sy + dy) * coded_w;
        for (int dx = 0; dx < sx; ++dx) {
          out_base[row + j * sx + dx] = v16;
        }
      }
    }
  }
}

int padded_elems(int smem_elems) {
  const int hint = smem_elems <= 0 ? kMaxElems : smem_elems;
  const int clamped = hint > kMaxElems ? kMaxElems : hint;
  return (clamped + 7) & ~7;
}

}  // namespace

bool still_tx_init_job(StillTxJob* job, int coded_w, int coded_h, TxType type) {
  if (job == nullptr) {
    return false;
  }
  const auto params = av2enc::txv::make_tx_params(coded_w, coded_h, type);
  if (!params.has_value()) {
    return false;
  }
  *job = StillTxJob{};
  job->params = *params;
  return true;
}

cudaError_t still_tx_batch(const StillTxJob* d_jobs, int nblocks,
                        const StillTxBuffers& buffers, int smem_elems,
                        cudaStream_t stream) {
  if (nblocks < 0) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  if (d_jobs == nullptr || buffers.levels == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int elems = padded_elems(smem_elems);
  // int16 adjusted residual (2E) + double forward staging (8E) + int32
  // levels/dequant (4E) + the two int16 transform kernels (2 * 1024 * 2 B).
  const int smem_bytes = 14 * elems + kKernelBytes;
  still_tx_kernel<<<static_cast<unsigned>(nblocks), kThreads,
                 static_cast<std::size_t>(smem_bytes), stream>>>(
      d_jobs, buffers.src, buffers.pred, buffers.residual_in,
      buffers.residual_adj, buffers.coeff, buffers.levels, buffers.dequant,
      buffers.residual_out, buffers.eob, buffers.error_count, elems);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
