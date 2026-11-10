#include "common.cuh"

#include <cstddef>

namespace {

std::int16_t* g_kernel32 = nullptr;
std::int16_t* g_kernel16 = nullptr;

// ---------------------------------------------------------------------------
// Fixed-point final quantization: round(|acc| * mul / 2^kQshift), half away
// from zero, via a 128-bit product (64x64 -> 128) so a full-precision 60-bit
// multiplier fits.  acc is at most ~3.1e12 and mul ~3.7e8; the product needs
// 71 bits.
// ---------------------------------------------------------------------------
__device__ __forceinline__ std::int64_t quantize_fixed(std::int64_t acc,
                                                       std::int64_t mul) {
  const bool negative = acc < 0;
  const unsigned long long a =
      negative ? static_cast<unsigned long long>(-acc)
               : static_cast<unsigned long long>(acc);
  const unsigned long long m = static_cast<unsigned long long>(mul);
  unsigned long long lo = a * m;
  unsigned long long hi = __umul64hi(a, m);
  const unsigned long long half =
      static_cast<unsigned long long>(1) << (kQshift - 1);
  lo += half;
  if (lo < half) {
    ++hi;
  }
  const unsigned long long q =
      (hi << (64 - kQshift)) | (lo >> kQshift);
  return negative ? -static_cast<std::int64_t>(q)
                  : static_cast<std::int64_t>(q);
}

// ---------------------------------------------------------------------------
// Double precision.  One warp owns N/32 output frequencies; lane i is input
// row i (first pass) or output column v (second pass).  The accumulation order
// is j = 0..N-1 (rows) and i = 0..N-1 (columns), exactly like the CPU loop.
// ---------------------------------------------------------------------------
template <int N, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void dct_double_kernel(
    const std::int16_t* __restrict__ in, int nblocks,
    const std::int16_t* __restrict__ kern, double scale_w, double scale_h,
    double qf, double* __restrict__ coeff_out, std::int32_t* __restrict__ q_out) {
  constexpr int NV = N / WARPS;
  __shared__ std::int16_t s_in[N * N];
  __shared__ double s_tmp[N * N];
  __shared__ std::int16_t s_k[N * N];

  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * (N * N);
  for (int i = threadIdx.x; i < N * N; i += WARPS * 32) {
    s_in[i] = in[base + i];
    s_k[i] = kern[i];
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane < N) {
    double acc[NV];
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      acc[q] = 0.0;
    }
    for (int j = 0; j < N; ++j) {
      const double xv = static_cast<double>(s_in[lane * N + j]);
#pragma unroll
      for (int q = 0; q < NV; ++q) {
        const int v = warp * NV + q;
        acc[q] = __dadd_rn(
            acc[q], __dmul_rn(static_cast<double>(s_k[v * N + j]), xv));
      }
    }
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      const int v = warp * NV + q;
      s_tmp[lane * N + v] = __dmul_rn(acc[q], scale_w);
    }
  }
  __syncthreads();

  if (lane < N) {
    double acc[NV];
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      acc[q] = 0.0;
    }
    for (int i = 0; i < N; ++i) {
      const double tv = s_tmp[i * N + lane];
#pragma unroll
      for (int q = 0; q < NV; ++q) {
        const int u = warp * NV + q;
        acc[q] = __dadd_rn(
            acc[q], __dmul_rn(static_cast<double>(s_k[u * N + i]), tv));
      }
    }
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      const int u = warp * NV + q;
      const int idx = u * N + lane;
      const double c = __dmul_rn(acc[q], scale_h);
      if (coeff_out != nullptr) {
        coeff_out[base + idx] = c;
      }
      q_out[base + idx] =
          static_cast<std::int32_t>(llround(__dmul_rn(c, qf)));
    }
  }
}

// ---------------------------------------------------------------------------
// Float precision, same structure and order.  Row sums are exact in float
// (all partial sums are integers below 2^24); the column pass is where the
// bits are lost.
// ---------------------------------------------------------------------------
template <int N, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void dct_float_kernel(
    const std::int16_t* __restrict__ in, int nblocks,
    const std::int16_t* __restrict__ kern, float scale_w, float scale_h,
    float qf, float* __restrict__ coeff_out, std::int32_t* __restrict__ q_out) {
  constexpr int NV = N / WARPS;
  __shared__ std::int16_t s_in[N * N];
  __shared__ float s_tmp[N * N];
  __shared__ std::int16_t s_k[N * N];

  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * (N * N);
  for (int i = threadIdx.x; i < N * N; i += WARPS * 32) {
    s_in[i] = in[base + i];
    s_k[i] = kern[i];
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane < N) {
    float acc[NV];
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      acc[q] = 0.0f;
    }
    for (int j = 0; j < N; ++j) {
      const float xv = static_cast<float>(s_in[lane * N + j]);
#pragma unroll
      for (int q = 0; q < NV; ++q) {
        const int v = warp * NV + q;
        acc[q] = __fadd_rn(
            acc[q], __fmul_rn(static_cast<float>(s_k[v * N + j]), xv));
      }
    }
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      const int v = warp * NV + q;
      s_tmp[lane * N + v] = __fmul_rn(acc[q], scale_w);
    }
  }
  __syncthreads();

  if (lane < N) {
    float acc[NV];
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      acc[q] = 0.0f;
    }
    for (int i = 0; i < N; ++i) {
      const float tv = s_tmp[i * N + lane];
#pragma unroll
      for (int q = 0; q < NV; ++q) {
        const int u = warp * NV + q;
        acc[q] = __fadd_rn(
            acc[q], __fmul_rn(static_cast<float>(s_k[u * N + i]), tv));
      }
    }
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      const int u = warp * NV + q;
      const int idx = u * N + lane;
      const float c = __fmul_rn(acc[q], scale_h);
      if (coeff_out != nullptr) {
        coeff_out[base + idx] = c;
      }
      q_out[base + idx] =
          static_cast<std::int32_t>(llroundf(__fmul_rn(c, qf)));
    }
  }
}

// ---------------------------------------------------------------------------
// Hybrid: integer row dot product (exact, same value as the CPU's double sum
// of integer products) followed by the CPU's double scale and double column
// pass.  This is bit-exact and removes half of the FP64 multiply-adds.
// ---------------------------------------------------------------------------
template <int N, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void dct_hybrid_kernel(
    const std::int16_t* __restrict__ in, int nblocks,
    const std::int16_t* __restrict__ kern, double scale_w, double scale_h,
    double qf, double* __restrict__ coeff_out, std::int32_t* __restrict__ q_out) {
  constexpr int NV = N / WARPS;
  __shared__ std::int16_t s_in[N * N];
  __shared__ double s_tmp[N * N];
  __shared__ std::int16_t s_k[N * N];

  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * (N * N);
  for (int i = threadIdx.x; i < N * N; i += WARPS * 32) {
    s_in[i] = in[base + i];
    s_k[i] = kern[i];
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane < N) {
    std::int32_t acc[NV];
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      acc[q] = 0;
    }
    for (int j = 0; j < N; ++j) {
      const std::int32_t xv = s_in[lane * N + j];
#pragma unroll
      for (int q = 0; q < NV; ++q) {
        const int v = warp * NV + q;
        acc[q] += static_cast<std::int32_t>(s_k[v * N + j]) * xv;
      }
    }
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      const int v = warp * NV + q;
      s_tmp[lane * N + v] = __dmul_rn(static_cast<double>(acc[q]), scale_w);
    }
  }
  __syncthreads();

  if (lane < N) {
    double acc[NV];
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      acc[q] = 0.0;
    }
    for (int i = 0; i < N; ++i) {
      const double tv = s_tmp[i * N + lane];
#pragma unroll
      for (int q = 0; q < NV; ++q) {
        const int u = warp * NV + q;
        acc[q] = __dadd_rn(
            acc[q], __dmul_rn(static_cast<double>(s_k[u * N + i]), tv));
      }
    }
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      const int u = warp * NV + q;
      const int idx = u * N + lane;
      const double c = __dmul_rn(acc[q], scale_h);
      if (coeff_out != nullptr) {
        coeff_out[base + idx] = c;
      }
      q_out[base + idx] =
          static_cast<std::int32_t>(llround(__dmul_rn(c, qf)));
    }
  }
}

// ---------------------------------------------------------------------------
// Integer / fixed point.  row_mul = round(scale_w * 2^kQrow); the row pass
// stores sum * row_mul, i.e. tmp * 2^kQrow * (1+delta).  quant_mul bakes in the
// reciprocal of (1+delta), so the systematic gain error cancels.  The column
// pass accumulates int64; only quantization needs a 64x64->128 multiply.
// ---------------------------------------------------------------------------
template <int N, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void dct_fixed_kernel(
    const std::int16_t* __restrict__ in, int nblocks,
    const std::int16_t* __restrict__ kern, std::int32_t row_mul,
    std::int64_t quant_mul, std::int32_t* __restrict__ q_out) {
  constexpr int NV = N / WARPS;
  __shared__ std::int16_t s_in[N * N];
  __shared__ std::int32_t s_t[N * N];
  __shared__ std::int16_t s_k[N * N];

  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * (N * N);
  for (int i = threadIdx.x; i < N * N; i += WARPS * 32) {
    s_in[i] = in[base + i];
    s_k[i] = kern[i];
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane < N) {
    std::int32_t acc[NV];
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      acc[q] = 0;
    }
    for (int j = 0; j < N; ++j) {
      const std::int32_t xv = s_in[lane * N + j];
#pragma unroll
      for (int q = 0; q < NV; ++q) {
        const int v = warp * NV + q;
        acc[q] += static_cast<std::int32_t>(s_k[v * N + j]) * xv;
      }
    }
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      const int v = warp * NV + q;
      s_t[lane * N + v] = static_cast<std::int32_t>(
          static_cast<std::int64_t>(acc[q]) * row_mul);
    }
  }
  __syncthreads();

  if (lane < N) {
    std::int64_t acc[NV];
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      acc[q] = 0;
    }
    for (int i = 0; i < N; ++i) {
      const std::int32_t tv = s_t[i * N + lane];
#pragma unroll
      for (int q = 0; q < NV; ++q) {
        const int u = warp * NV + q;
        acc[q] += static_cast<std::int64_t>(s_k[u * N + i]) * tv;
      }
    }
#pragma unroll
    for (int q = 0; q < NV; ++q) {
      const int u = warp * NV + q;
      const int idx = u * N + lane;
      q_out[base + idx] =
          static_cast<std::int32_t>(quantize_fixed(acc[q], quant_mul));
    }
  }
}

// ---------------------------------------------------------------------------
// Dispatch helpers.
// ---------------------------------------------------------------------------
template <int N, int WARPS>
void launch_double_cfg(const std::int16_t* d_in, int nblocks,
                       const std::int16_t* kern, const BlockParams& p,
                       void* d_coeff, std::int32_t* d_q, bool write_coeff,
                       cudaStream_t stream) {
  const int threads = WARPS * 32;
  double* coeff = write_coeff ? static_cast<double*>(d_coeff) : nullptr;
  dct_double_kernel<N, WARPS><<<nblocks, threads, 0, stream>>>(
      d_in, nblocks, kern, p.scale_w, p.scale_h, p.quant_factor, coeff, d_q);
}

template <int N, int WARPS>
void launch_float_cfg(const std::int16_t* d_in, int nblocks,
                      const std::int16_t* kern, const BlockParams& p,
                      void* d_coeff, std::int32_t* d_q, bool write_coeff,
                      cudaStream_t stream) {
  const int threads = WARPS * 32;
  float* coeff = write_coeff ? static_cast<float*>(d_coeff) : nullptr;
  dct_float_kernel<N, WARPS><<<nblocks, threads, 0, stream>>>(
      d_in, nblocks, kern, static_cast<float>(p.scale_w),
      static_cast<float>(p.scale_h), static_cast<float>(p.quant_factor), coeff,
      d_q);
}

template <int N, int WARPS>
void launch_hybrid_cfg(const std::int16_t* d_in, int nblocks,
                       const std::int16_t* kern, const BlockParams& p,
                       void* d_coeff, std::int32_t* d_q, bool write_coeff,
                       cudaStream_t stream) {
  const int threads = WARPS * 32;
  double* coeff = write_coeff ? static_cast<double*>(d_coeff) : nullptr;
  dct_hybrid_kernel<N, WARPS><<<nblocks, threads, 0, stream>>>(
      d_in, nblocks, kern, p.scale_w, p.scale_h, p.quant_factor, coeff, d_q);
}

template <int N, int WARPS>
void launch_fixed_cfg(const std::int16_t* d_in, int nblocks,
                      const std::int16_t* kern, const BlockParams& p,
                      std::int32_t* d_q, bool naive, cudaStream_t stream) {
  const int threads = WARPS * 32;
  const std::int64_t mul = naive ? p.quant_mul_naive : p.quant_mul;
  dct_fixed_kernel<N, WARPS><<<nblocks, threads, 0, stream>>>(
      d_in, nblocks, kern, p.row_mul, mul, d_q);
}

template <int N>
void launch_variant(Variant variant, int warps, bool write_coeff,
                    const std::int16_t* d_in, int nblocks,
                    const BlockParams& p, void* d_coeff, std::int32_t* d_q,
                    bool fixed_naive, cudaStream_t stream) {
  const std::int16_t* kern = (N == 32) ? g_kernel32 : g_kernel16;
  auto bad_warps = [] {
    std::fprintf(stderr, "invalid warps for this size\n");
    std::exit(EXIT_FAILURE);
  };

  if (variant == Variant::kDouble) {
    if (warps == 4) {
      launch_double_cfg<N, 4>(d_in, nblocks, kern, p, d_coeff, d_q, write_coeff,
                              stream);
    } else if (warps == 8) {
      launch_double_cfg<N, 8>(d_in, nblocks, kern, p, d_coeff, d_q, write_coeff,
                              stream);
    } else if constexpr (N == 32) {
      if (warps == 32) {
        launch_double_cfg<N, 32>(d_in, nblocks, kern, p, d_coeff, d_q,
                                 write_coeff, stream);
      } else {
        bad_warps();
      }
    } else {
      if (warps == 2) {
        launch_double_cfg<N, 2>(d_in, nblocks, kern, p, d_coeff, d_q,
                                write_coeff, stream);
      } else {
        bad_warps();
      }
    }
  } else if (variant == Variant::kFloat) {
    if (warps == 4) {
      launch_float_cfg<N, 4>(d_in, nblocks, kern, p, d_coeff, d_q, write_coeff,
                             stream);
    } else if (warps == 8) {
      launch_float_cfg<N, 8>(d_in, nblocks, kern, p, d_coeff, d_q, write_coeff,
                             stream);
    } else if constexpr (N == 32) {
      if (warps == 32) {
        launch_float_cfg<N, 32>(d_in, nblocks, kern, p, d_coeff, d_q,
                                write_coeff, stream);
      } else {
        bad_warps();
      }
    } else {
      if (warps == 2) {
        launch_float_cfg<N, 2>(d_in, nblocks, kern, p, d_coeff, d_q,
                               write_coeff, stream);
      } else {
        bad_warps();
      }
    }
  } else if (variant == Variant::kHybrid) {
    if (warps == 4) {
      launch_hybrid_cfg<N, 4>(d_in, nblocks, kern, p, d_coeff, d_q, write_coeff,
                              stream);
    } else if (warps == 8) {
      launch_hybrid_cfg<N, 8>(d_in, nblocks, kern, p, d_coeff, d_q, write_coeff,
                              stream);
    } else if constexpr (N == 32) {
      if (warps == 32) {
        launch_hybrid_cfg<N, 32>(d_in, nblocks, kern, p, d_coeff, d_q,
                                 write_coeff, stream);
      } else {
        bad_warps();
      }
    } else {
      if (warps == 2) {
        launch_hybrid_cfg<N, 2>(d_in, nblocks, kern, p, d_coeff, d_q,
                                write_coeff, stream);
      } else {
        bad_warps();
      }
    }
  } else {
    if (warps == 4) {
      launch_fixed_cfg<N, 4>(d_in, nblocks, kern, p, d_q, fixed_naive, stream);
    } else if (warps == 8) {
      launch_fixed_cfg<N, 8>(d_in, nblocks, kern, p, d_q, fixed_naive, stream);
    } else if constexpr (N == 32) {
      if (warps == 32) {
        launch_fixed_cfg<N, 32>(d_in, nblocks, kern, p, d_q, fixed_naive,
                                stream);
      } else {
        bad_warps();
      }
    } else {
      if (warps == 2) {
        launch_fixed_cfg<N, 2>(d_in, nblocks, kern, p, d_q, fixed_naive,
                               stream);
      } else {
        bad_warps();
      }
    }
  }
}

}  // namespace

void init_device_tables(const std::int16_t* kernel32,
                        const std::int16_t* kernel16) {
  CUDA_CHECK(cudaMalloc(&g_kernel32, 32 * 32 * sizeof(std::int16_t)));
  CUDA_CHECK(cudaMalloc(&g_kernel16, 16 * 16 * sizeof(std::int16_t)));
  CUDA_CHECK(cudaMemcpy(g_kernel32, kernel32,
                        32 * 32 * sizeof(std::int16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(g_kernel16, kernel16,
                        16 * 16 * sizeof(std::int16_t), cudaMemcpyHostToDevice));
}

void shutdown_device_tables() {
  if (g_kernel32 != nullptr) {
    CUDA_CHECK(cudaFree(g_kernel32));
    g_kernel32 = nullptr;
  }
  if (g_kernel16 != nullptr) {
    CUDA_CHECK(cudaFree(g_kernel16));
    g_kernel16 = nullptr;
  }
}

int loaded_binary_version() {
  cudaFuncAttributes attr{};
  const cudaError_t status = cudaFuncGetAttributes(
      &attr, reinterpret_cast<const void*>(&dct_double_kernel<32, 8>));
  if (status != cudaSuccess) {
    return -1;
  }
  return attr.binaryVersion;
}

void launch_transform(Variant variant, int n, int warps, bool write_coeff,
                      const std::int16_t* d_in, int nblocks,
                      const BlockParams& params, void* d_coeff,
                      std::int32_t* d_q, bool fixed_naive,
                      cudaStream_t stream) {
  if (n == 32) {
    launch_variant<32>(variant, warps, write_coeff, d_in, nblocks, params,
                       d_coeff, d_q, fixed_naive, stream);
  } else {
    launch_variant<16>(variant, warps, write_coeff, d_in, nblocks, params,
                       d_coeff, d_q, fixed_naive, stream);
  }
}
