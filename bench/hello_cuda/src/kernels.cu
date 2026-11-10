#include "common.cuh"

#include <cmath>

namespace {

__constant__ int32_t c_dct_basis[kDctNumSizes][kDctMaxSize * kDctMaxSize];

int32_t g_host_basis[kDctNumSizes][kDctMaxSize * kDctMaxSize];

__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

__global__ void copy_kernel(const uint4* __restrict__ in,
                            uint4* __restrict__ out, int n4) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    out[i] = in[i];
  }
}

// Naive direct 2D integer DCT: one thread per output coefficient, O(N^4)
// multiplies per block. Not an AV2 transform; a throughput baseline only.
template <int N, int SIZE_INDEX>
__global__ void dct_naive_kernel(const int32_t* __restrict__ in,
                                 int32_t* __restrict__ out) {
  __shared__ int32_t s_in[N * N];
  __shared__ int32_t s_bt[N * N];  // transposed basis: s_bt[x * N + u] = B[u][x]

  const int tid = threadIdx.x;
  const size_t base = static_cast<size_t>(blockIdx.x) * (N * N);

  s_in[tid] = in[base + tid];
  const int load_x = tid % N;
  const int load_u = tid / N;
  s_bt[load_x * N + load_u] = c_dct_basis[SIZE_INDEX][load_u * kDctMaxSize + load_x];
  __syncthreads();

  const int ou = tid / N;
  const int ov = tid % N;

  int32_t sum = 0;
  for (int y = 0; y < N; ++y) {
    const int32_t bv = s_bt[y * N + ov];
#pragma unroll 8
    for (int x = 0; x < N; ++x) {
      sum += s_in[y * N + x] * s_bt[x * N + ou] * bv;
    }
  }

  out[base + tid] = sum >> (2 * kDctQ);
}

constexpr int kSadWarps = 8;

// One warp per 16x16 block. Lane-strided byte loads + warp reduction.
__global__ void sad16_kernel(const uint8_t* __restrict__ a,
                             const uint8_t* __restrict__ b,
                             int32_t* __restrict__ out, int nblocks) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int blk = blockIdx.x * kSadWarps + warp;
  if (blk >= nblocks) {
    return;
  }

  const size_t base = static_cast<size_t>(blk) * 256;
  const uint8_t* pa = a + base;
  const uint8_t* pb = b + base;

  int32_t sum = 0;
#pragma unroll
  for (int i = lane; i < 256; i += 32) {
    const int diff = static_cast<int>(pa[i]) - static_cast<int>(pb[i]);
    sum += diff < 0 ? -diff : diff;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0) {
    out[blk] = sum;
  }
}

}  // namespace

void init_dct_basis() {
  constexpr double kPi = 3.14159265358979323846;
  for (int s = 0; s < kDctNumSizes; ++s) {
    const int n = kDctSizes[s];
    const double norm0 = std::sqrt(1.0 / static_cast<double>(n));
    const double norm = std::sqrt(2.0 / static_cast<double>(n));
    for (int u = 0; u < n; ++u) {
      for (int x = 0; x < n; ++x) {
        const double scale = (u == 0) ? norm0 : norm;
        const double angle =
            (2.0 * static_cast<double>(x) + 1.0) * static_cast<double>(u) * kPi /
            (2.0 * static_cast<double>(n));
        const long rounded =
            std::lround(scale * std::cos(angle) * static_cast<double>(1 << kDctQ));
        g_host_basis[s][u * kDctMaxSize + x] = static_cast<int32_t>(rounded);
      }
    }
  }
  CUDA_CHECK(cudaMemcpyToSymbol(c_dct_basis, g_host_basis, sizeof(g_host_basis)));
}

const int32_t* host_dct_basis(int size_index) {
  return g_host_basis[size_index];
}

void launch_vector_add(const float* a, const float* b, float* out, int n,
                       cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int blocks = (n + kThreads - 1) / kThreads;
  vector_add_kernel<<<blocks, kThreads, 0, stream>>>(a, b, out, n);
}

void launch_copy(const void* in, void* out, size_t bytes, cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int n4 = static_cast<int>(bytes / 16);
  const int blocks = (n4 + kThreads - 1) / kThreads;
  copy_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const uint4*>(in), static_cast<uint4*>(out), n4);
}

void launch_dct(const int32_t* in, int32_t* out, int nblocks, int n,
                cudaStream_t stream) {
  switch (n) {
    case 8:
      dct_naive_kernel<8, 0><<<nblocks, 8 * 8, 0, stream>>>(in, out);
      break;
    case 16:
      dct_naive_kernel<16, 1><<<nblocks, 16 * 16, 0, stream>>>(in, out);
      break;
    case 32:
      dct_naive_kernel<32, 2><<<nblocks, 32 * 32, 0, stream>>>(in, out);
      break;
    default:
      std::fprintf(stderr, "launch_dct: unsupported size %d\n", n);
      std::exit(EXIT_FAILURE);
  }
}

void launch_sad16(const uint8_t* a, const uint8_t* b, int32_t* out, int nblocks,
                  cudaStream_t stream) {
  const int blocks = (nblocks + kSadWarps - 1) / kSadWarps;
  sad16_kernel<<<blocks, kSadWarps * 32, 0, stream>>>(a, b, out, nblocks);
}
