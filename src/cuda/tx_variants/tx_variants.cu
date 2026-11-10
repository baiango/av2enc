// CUDA transform-variant kernels.  Bit-exact port of
// src/encoder/tx_variants/tx_variants.cpp; see tx_variants.h for the contract.
#include "cuda/tx_variants/tx_variants.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "encoder/tx_variants/tx_tables_dev.cuh"

namespace av2enc::cuda {
namespace {

using namespace av2enc::txv::dev;

constexpr int kMaxTxElems = av2enc::txv::kMaxAdjTxElems;
constexpr int kQuantStep = av2enc::txv::kQuantStep;
constexpr int kQuantTableBits = av2enc::txv::kQuantTableBits;
constexpr int kMaxThreads = 1024;

__device__ __forceinline__ int round2_dev(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

__device__ __forceinline__ int clip3_dev(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

// §7.15.2.1 kernel selection.  IDT has no matrix (returns nullptr); length 32
// always uses Dct_Kernel32, matching the spec's `sz == 32` branch.
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
  return qc < 0 ? -dq : dq;
}

__device__ __forceinline__ int prescale_dev(int value, bool odd_rect) {
  if (!odd_rect) {
    return value;
  }
  const int scaled =
      static_cast<int>(static_cast<long long>(value) * 2896LL);
  return round2_dev(scaled, 12);
}

// Shared body for both the uniform and per-block-parameter kernels.  All
// threads in the block call it; guards keep the multi launch (1024 threads)
// safe when `w*h < 1024`.  __syncthreads() is never inside a divergent branch.
__device__ __forceinline__ void forward_body(const std::int16_t* residual,
                                             double* coeff,
                                             std::int32_t* levels,
                                             const TxParams& p,
                                             std::int16_t* s_kw,
                                             std::int16_t* s_kh, double* s_tmp,
                                             int base_in) {
  const int tid = static_cast<int>(threadIdx.x);
  const int w = p.w;
  const int h = p.h;
  const int elems = w * h;
  const std::int16_t* kw = kernel_dev(static_cast<int>(p.row_type), w);
  const std::int16_t* kh = kernel_dev(static_cast<int>(p.col_type), h);

  if (tid < elems) {
    if (kw != nullptr) {
      for (int t = tid; t < w * w; t += elems) {
        s_kw[t] = kw[t];
      }
    }
    if (kh != nullptr) {
      for (int t = tid; t < h * h; t += elems) {
        s_kh[t] = kh[t];
      }
    }
  }
  __syncthreads();

  if (tid < elems) {
    const int i = tid / w;
    const int v = tid - i * w;
    const int row = i * w;
    if (kw == nullptr) {
      s_tmp[tid] = static_cast<double>(residual[base_in + tid]);
    } else {
      double sum = 0.0;
      for (int j = 0; j < w; ++j) {
        sum = __dadd_rn(
            sum, __dmul_rn(static_cast<double>(s_kw[v * w + j]),
                           static_cast<double>(residual[base_in + row + j])));
      }
      s_tmp[tid] = __dmul_rn(sum, p.scale_w);
    }
  }
  __syncthreads();

  if (tid < elems) {
    const int u = tid / w;
    const int v = tid - u * w;
    double out = 0.0;
    if (kh == nullptr) {
      out = s_tmp[tid];
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
    coeff[base_in + tid] = out;
    if (levels != nullptr) {
      levels[base_in + tid] =
          static_cast<std::int32_t>(lround(out * p.quant_factor));
    }
  }
}

__device__ __forceinline__ void inverse_body(const std::int32_t* dequant,
                                             const std::int32_t* levels,
                                             std::int16_t* residual_coded,
                                             const TxParams& p,
                                             std::int16_t* s_kw,
                                             std::int16_t* s_kh,
                                             std::int32_t* s_dq,
                                             int* s_inter, int base_in,
                                             int base_out) {
  const int tid = static_cast<int>(threadIdx.x);
  const int w = p.w;
  const int h = p.h;
  const int elems = w * h;
  const std::int16_t* kw = kernel_dev(static_cast<int>(p.row_type), w);
  const std::int16_t* kh = kernel_dev(static_cast<int>(p.col_type), h);
  const bool odd_rect = ((p.log2_w ^ p.log2_h) & 1) != 0;
  const int identity_w = kIdentityScaleDev[p.log2_w];
  const int identity_h = kIdentityScaleDev[p.log2_h];

  // Exactly one of `dequant` / `levels` is non-null.  Dequantize once into
  // shared memory instead of recomputing it inside every row dot product.
  if (tid < elems) {
    if (kw != nullptr) {
      for (int t = tid; t < w * w; t += elems) {
        s_kw[t] = kw[t];
      }
    }
    if (kh != nullptr) {
      for (int t = tid; t < h * h; t += elems) {
        s_kh[t] = kh[t];
      }
    }
    s_dq[tid] = (levels != nullptr)
                    ? dequant_dev(levels[base_in + tid], p.dq_denom)
                    : dequant[base_in + tid];
  }
  __syncthreads();
  auto load = [&](int idx) -> int { return s_dq[idx]; };

  // Row pass.
  if (tid < elems) {
    const int i = tid / w;
    const int oj = tid - i * w;
    const int row = i * w;
    int value = 0;
    if (kw == nullptr) {
      const long long s = static_cast<long long>(
                              prescale_dev(load(row + oj), odd_rect)) *
                          identity_w;
      value = clip3_dev(-(1 << 15), (1 << 15) - 1,
                        round2_dev(static_cast<int>(s), p.row_shift));
    } else {
      long long sum = 0;
      for (int j = 0; j < w; ++j) {
        sum += static_cast<long long>(s_kw[j * w + oj]) *
               prescale_dev(load(row + j), odd_rect);
      }
      value = clip3_dev(-(1 << 15), (1 << 15) - 1,
                        round2_dev(static_cast<int>(sum), p.row_shift));
    }
    s_inter[tid] = value;
  }
  __syncthreads();

  // Column pass + §7.15.4 coded-size duplication.
  if (tid < elems) {
    const int oi = tid / w;
    const int j = tid - oi * w;
    int value = 0;
    if (kh == nullptr) {
      const long long s =
          static_cast<long long>(s_inter[oi * w + j]) * identity_h;
      value = round2_dev(static_cast<int>(s), p.col_shift);
    } else {
      long long sum = 0;
      for (int i = 0; i < h; ++i) {
        sum += static_cast<long long>(s_kh[i * h + oi]) *
               s_inter[i * w + j];
      }
      value = round2_dev(static_cast<int>(sum), p.col_shift);
    }
    value = clip3_dev(-256, 255, value);
    const std::int16_t out = static_cast<std::int16_t>(value);

    const int sx = p.coded_w / w;
    const int sy = p.coded_h / h;
    for (int dy = 0; dy < sy; ++dy) {
      const int y = (oi * sy + dy) * p.coded_w;
      for (int dx = 0; dx < sx; ++dx) {
        residual_coded[base_out + y + j * sx + dx] = out;
      }
    }
  }
}

__global__ void forward_kernel(const std::int16_t* __restrict__ residual,
                               double* __restrict__ coeff,
                               std::int32_t* __restrict__ levels, TxParams p) {
  __shared__ std::int16_t s_kw[kMaxTxElems];
  __shared__ std::int16_t s_kh[kMaxTxElems];
  extern __shared__ double s_tmp[];
  forward_body(residual, coeff, levels, p, s_kw, s_kh, s_tmp,
               static_cast<int>(blockIdx.x) * (p.w * p.h));
}

__global__ void forward_job_kernel(const std::int16_t* __restrict__ residual,
                                   double* __restrict__ coeff,
                                   std::int32_t* __restrict__ levels,
                                   const TxBlockJob* __restrict__ jobs) {
  __shared__ std::int16_t s_kw[kMaxTxElems];
  __shared__ std::int16_t s_kh[kMaxTxElems];
  extern __shared__ double s_tmp[];
  const TxBlockJob job = jobs[blockIdx.x];
  forward_body(residual, coeff, levels, job.params, s_kw, s_kh, s_tmp,
               job.in_offset);
}

__global__ void inverse_kernel(const std::int32_t* __restrict__ dequant,
                               const std::int32_t* __restrict__ levels,
                               std::int16_t* __restrict__ residual_coded,
                               TxParams p) {
  __shared__ std::int16_t s_kw[kMaxTxElems];
  __shared__ std::int16_t s_kh[kMaxTxElems];
  __shared__ std::int32_t s_dq[kMaxTxElems];
  extern __shared__ int s_inter[];
  inverse_body(dequant, levels, residual_coded, p, s_kw, s_kh, s_dq, s_inter,
               static_cast<int>(blockIdx.x) * (p.w * p.h),
               static_cast<int>(blockIdx.x) * (p.coded_w * p.coded_h));
}

__global__ void inverse_job_kernel(const std::int32_t* __restrict__ dequant,
                                   const std::int32_t* __restrict__ levels,
                                   std::int16_t* __restrict__ residual_coded,
                                   const TxBlockJob* __restrict__ jobs) {
  __shared__ std::int16_t s_kw[kMaxTxElems];
  __shared__ std::int16_t s_kh[kMaxTxElems];
  __shared__ std::int32_t s_dq[kMaxTxElems];
  extern __shared__ int s_inter[];
  const TxBlockJob job = jobs[blockIdx.x];
  inverse_body(dequant, levels, residual_coded, job.params, s_kw, s_kh, s_dq,
               s_inter, job.in_offset, job.out_offset);
}

__global__ void quantize_kernel(const double* __restrict__ coeff,
                                std::int32_t* __restrict__ levels, TxParams p) {
  const int elems = p.w * p.h;
  const int base = static_cast<int>(blockIdx.x) * elems;
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < elems) {
    levels[base + tid] =
        static_cast<std::int32_t>(lround(coeff[base + tid] * p.quant_factor));
  }
}

__global__ void dequantize_kernel(const std::int32_t* __restrict__ levels,
                                  std::int32_t* __restrict__ dequant,
                                  TxParams p) {
  const int elems = p.w * p.h;
  const int base = static_cast<int>(blockIdx.x) * elems;
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < elems) {
    dequant[base + tid] = dequant_dev(levels[base + tid], p.dq_denom);
  }
}

__global__ void downsample_kernel(const std::int16_t* __restrict__ coded,
                                  std::int16_t* __restrict__ adj, TxParams p) {
  const int elems = p.w * p.h;
  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= elems) {
    return;
  }
  const int sx = p.coded_w / p.w;
  const int sy = p.coded_h / p.h;
  const int shift = 31 - __clz(sx * sy);
  const int base_in =
      static_cast<int>(blockIdx.x) * (p.coded_w * p.coded_h);
  const int base_out = static_cast<int>(blockIdx.x) * elems;
  const int j = tid / p.w;
  const int i = tid - j * p.w;
  int sum = 0;
  for (int dy = 0; dy < sy; ++dy) {
    const int row = base_in + (j * sy + dy) * p.coded_w;
    for (int dx = 0; dx < sx; ++dx) {
      sum += static_cast<int>(coded[row + i * sx + dx]);
    }
  }
  adj[base_out + tid] = static_cast<std::int16_t>(round2_dev(sum, shift));
}

__global__ void upsample_kernel(const std::int16_t* __restrict__ adj,
                                std::int16_t* __restrict__ coded, TxParams p) {
  const int coded_n = p.coded_w * p.coded_h;
  const int base_in = static_cast<int>(blockIdx.x) * (p.w * p.h);
  const int base_out = static_cast<int>(blockIdx.x) * coded_n;
  const int sx = p.coded_w / p.w;
  const int sy = p.coded_h / p.h;
  for (int idx = static_cast<int>(threadIdx.x); idx < coded_n;
       idx += static_cast<int>(blockDim.x)) {
    const int y = idx / p.coded_w;
    const int x = idx - y * p.coded_w;
    coded[base_out + idx] = adj[base_in + (y / sy) * p.w + (x / sx)];
  }
}

bool is_tx_dim(int n) { return n == 4 || n == 8 || n == 16 || n == 32; }

bool valid_size(const TxParams& p) {
  return is_tx_dim(p.w) && is_tx_dim(p.h) && p.w * p.h <= kMaxTxElems &&
         p.coded_w >= p.w && p.coded_h >= p.h &&
         p.coded_w % p.w == 0 && p.coded_h % p.h == 0 &&
         av2enc::txv::tx_size_supported(p.coded_w, p.coded_h);
}

bool valid_params(const TxParams& p) {
  if (!valid_size(p)) {
    return false;
  }
  if (p.row_type < 0 || p.row_type > 3 || p.col_type < 0 || p.col_type > 3) {
    return false;
  }
  if (p.row_shift <= 0 || p.row_shift >= 31 || p.col_shift <= 0 ||
      p.col_shift >= 31) {
    return false;
  }
  if (p.dq_denom != 1 && p.dq_denom != 2 && p.dq_denom != 4) {
    return false;
  }
  if (!(p.scale_w > 0.0) || !(p.scale_h > 0.0) ||
      !(p.quant_factor > 0.0) || !(p.rect_prescale > 0.0)) {
    return false;
  }
  return true;
}

bool valid_batch(const TxParams& p, int nblocks) {
  return valid_params(p) && nblocks >= 0;
}

bool valid_multi(int nblocks) {
  return nblocks >= 0;
}

cudaError_t launch_forward(const std::int16_t* d_residual, double* d_coeff,
                           std::int32_t* d_levels, const TxParams& p,
                           int nblocks, cudaStream_t stream) {
  const int elems = p.w * p.h;
  const std::size_t smem = static_cast<std::size_t>(elems) * sizeof(double);
  forward_kernel<<<static_cast<unsigned>(nblocks),
                   static_cast<unsigned>(elems), smem, stream>>>(
      d_residual, d_coeff, d_levels, p);
  return cudaGetLastError();
}

}  // namespace

cudaError_t txv_forward_block(const std::int16_t* d_residual_adj,
                              double* d_coeff, const TxParams& p, int nblocks,
                              cudaStream_t stream) {
  if (!valid_batch(p, nblocks) || d_residual_adj == nullptr ||
      d_coeff == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  return launch_forward(d_residual_adj, d_coeff, nullptr, p, nblocks, stream);
}

cudaError_t txv_quantize(const double* d_coeff, std::int32_t* d_levels,
                         const TxParams& p, int nblocks, cudaStream_t stream) {
  if (!valid_batch(p, nblocks) || d_coeff == nullptr || d_levels == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  const int elems = p.w * p.h;
  quantize_kernel<<<static_cast<unsigned>(nblocks),
                    static_cast<unsigned>(elems), 0, stream>>>(
      d_coeff, d_levels, p);
  return cudaGetLastError();
}

cudaError_t txv_dequantize(const std::int32_t* d_levels,
                           std::int32_t* d_dequant, const TxParams& p,
                           int nblocks, cudaStream_t stream) {
  if (!valid_batch(p, nblocks) || d_levels == nullptr || d_dequant == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  const int elems = p.w * p.h;
  dequantize_kernel<<<static_cast<unsigned>(nblocks),
                      static_cast<unsigned>(elems), 0, stream>>>(
      d_levels, d_dequant, p);
  return cudaGetLastError();
}

cudaError_t txv_inverse_block(const std::int32_t* d_dequant,
                              std::int16_t* d_residual_coded,
                              const TxParams& p, int nblocks,
                              cudaStream_t stream) {
  if (!valid_batch(p, nblocks) || d_dequant == nullptr ||
      d_residual_coded == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  const int elems = p.w * p.h;
  const std::size_t smem = static_cast<std::size_t>(elems) * sizeof(int);
  inverse_kernel<<<static_cast<unsigned>(nblocks),
                   static_cast<unsigned>(elems), smem, stream>>>(
      d_dequant, nullptr, d_residual_coded, p);
  return cudaGetLastError();
}

cudaError_t txv_forward_transform(const std::int16_t* d_residual_adj,
                                  double* d_coeff_scratch,
                                  std::int32_t* d_levels, const TxParams& p,
                                  int nblocks, cudaStream_t stream) {
  if (!valid_batch(p, nblocks) || d_residual_adj == nullptr ||
      d_coeff_scratch == nullptr || d_levels == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  return launch_forward(d_residual_adj, d_coeff_scratch, d_levels, p, nblocks,
                        stream);
}

cudaError_t txv_inverse_transform(const std::int32_t* d_levels,
                                  std::int16_t* d_residual_coded,
                                  const TxParams& p, int nblocks,
                                  cudaStream_t stream) {
  if (!valid_batch(p, nblocks) || d_levels == nullptr ||
      d_residual_coded == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  const int elems = p.w * p.h;
  const std::size_t smem = static_cast<std::size_t>(elems) * sizeof(int);
  inverse_kernel<<<static_cast<unsigned>(nblocks),
                   static_cast<unsigned>(elems), smem, stream>>>(
      nullptr, d_levels, d_residual_coded, p);
  return cudaGetLastError();
}

cudaError_t txv_forward_block_multi(const std::int16_t* d_residual_adj,
                                    double* d_coeff,
                                    const TxBlockJob* d_jobs, int nblocks,
                                    cudaStream_t stream) {
  if (!valid_multi(nblocks) || d_residual_adj == nullptr || d_coeff == nullptr ||
      d_jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  const std::size_t smem =
      static_cast<std::size_t>(kMaxTxElems) * sizeof(double);
  forward_job_kernel<<<static_cast<unsigned>(nblocks), kMaxThreads, smem,
                       stream>>>(d_residual_adj, d_coeff, nullptr, d_jobs);
  return cudaGetLastError();
}

cudaError_t txv_forward_transform_multi(const std::int16_t* d_residual_adj,
                                        double* d_coeff,
                                        std::int32_t* d_levels,
                                        const TxBlockJob* d_jobs, int nblocks,
                                        cudaStream_t stream) {
  if (!valid_multi(nblocks) || d_residual_adj == nullptr || d_coeff == nullptr ||
      d_levels == nullptr || d_jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  const std::size_t smem =
      static_cast<std::size_t>(kMaxTxElems) * sizeof(double);
  forward_job_kernel<<<static_cast<unsigned>(nblocks), kMaxThreads, smem,
                       stream>>>(d_residual_adj, d_coeff, d_levels, d_jobs);
  return cudaGetLastError();
}

cudaError_t txv_inverse_block_multi(const std::int32_t* d_dequant,
                                    std::int16_t* d_residual_coded,
                                    const TxBlockJob* d_jobs, int nblocks,
                                    cudaStream_t stream) {
  if (!valid_multi(nblocks) || d_dequant == nullptr ||
      d_residual_coded == nullptr || d_jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  const std::size_t smem = static_cast<std::size_t>(kMaxTxElems) * sizeof(int);
  inverse_job_kernel<<<static_cast<unsigned>(nblocks), kMaxThreads, smem,
                       stream>>>(d_dequant, nullptr, d_residual_coded, d_jobs);
  return cudaGetLastError();
}

cudaError_t txv_inverse_transform_multi(const std::int32_t* d_levels,
                                        std::int16_t* d_residual_coded,
                                        const TxBlockJob* d_jobs, int nblocks,
                                        cudaStream_t stream) {
  if (!valid_multi(nblocks) || d_levels == nullptr ||
      d_residual_coded == nullptr || d_jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  const std::size_t smem = static_cast<std::size_t>(kMaxTxElems) * sizeof(int);
  inverse_job_kernel<<<static_cast<unsigned>(nblocks), kMaxThreads, smem,
                       stream>>>(nullptr, d_levels, d_residual_coded, d_jobs);
  return cudaGetLastError();
}

cudaError_t txv_downsample_residual(const std::int16_t* d_residual_coded,
                                    std::int16_t* d_residual_adj,
                                    const TxParams& p, int nblocks,
                                    cudaStream_t stream) {
  if (!valid_batch(p, nblocks) || d_residual_coded == nullptr ||
      d_residual_adj == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  const int elems = p.w * p.h;
  downsample_kernel<<<static_cast<unsigned>(nblocks),
                      static_cast<unsigned>(elems), 0, stream>>>(
      d_residual_coded, d_residual_adj, p);
  return cudaGetLastError();
}

cudaError_t txv_upsample_residual(const std::int16_t* d_residual_adj,
                                  std::int16_t* d_residual_coded,
                                  const TxParams& p, int nblocks,
                                  cudaStream_t stream) {
  if (!valid_batch(p, nblocks) || d_residual_adj == nullptr ||
      d_residual_coded == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  upsample_kernel<<<static_cast<unsigned>(nblocks), 256, 0, stream>>>(
      d_residual_adj, d_residual_coded, p);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
