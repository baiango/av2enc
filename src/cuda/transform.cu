// CUDA transform/quantization kernels for the av2enc M0 still-picture path.
//
// Bit-exact port of TileEncoder::forward_dct(), TileEncoder::inverse_dct() and
// the transform part of TileEncoder::encode_residual_plane() in
// src/encoder/m0_still.cpp.  See transform.h for the API contract.
//
// Floating point: the CPU reference accumulates in binary64 with a scalar
// multiply then add per term.  The kernels use the same order and force
// __dmul_rn()/__dadd_rn() so nvcc cannot contract them into FMA; the component
// build additionally passes -fmad=false.  No transcendental function is
// evaluated on the device: scale_w/scale_h/quant_factor are computed on the
// host by make_tx_params() with the same std::sqrt expression as the CPU.
#include "cuda/transform.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace av2enc::cuda {
namespace {

constexpr int kMaxTxDim = 32;
constexpr int kMaxTxElems = kMaxTxDim * kMaxTxDim;
constexpr int kQuantStep = 528;
constexpr int kQuantTableBits = 3;  // QUANT_TABLE_BITS

// Dct_Kernel{n}[j][i] = 64 * sqrt(n) * B[j][i], copied verbatim from
// src/encoder/m0_tables.h.  transform_test.cu checks every entry against the
// CPU reference: impulse residuals at each position exercise the whole table.
__device__ const std::int16_t kDct4Dev[4][4] = {
    {64, 64, 64, 64},
    {83, 35, -35, -83},
    {64, -64, -64, 64},
    {35, -83, 83, -35},
};

__device__ const std::int16_t kDct8Dev[8][8] = {
    {64, 64, 64, 64, 64, 64, 64, 64},
    {89, 75, 50, 18, -18, -50, -75, -89},
    {83, 35, -35, -83, -83, -35, 35, 83},
    {75, -18, -89, -50, 50, 89, 18, -75},
    {64, -64, -64, 64, 64, -64, -64, 64},
    {50, -89, 18, 75, -75, -18, 89, -50},
    {35, -83, 83, -35, -35, 83, -83, 35},
    {18, -50, 75, -89, 89, -75, 50, -18},
};

__device__ const std::int16_t kDct16Dev[16][16] = {
    {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64},
    {90, 87, 80, 70, 57, 43, 26, 9, -9, -26, -43, -57, -70, -80, -87, -90},
    {89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89},
    {87, 57, 9, -43, -80, -90, -70, -26, 26, 70, 90, 80, 43, -9, -57, -87},
    {83, 35, -35, -83, -83, -35, 35, 83, 83, 35, -35, -83, -83, -35, 35, 83},
    {80, 9, -70, -87, -26, 57, 90, 43, -43, -90, -57, 26, 87, 70, -9, -80},
    {75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75},
    {70, -43, -87, 9, 90, 26, -80, -57, 57, 80, -26, -90, -9, 87, 43, -70},
    {64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64},
    {57, -80, -26, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 26, 80, -57},
    {50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50},
    {43, -90, 57, 26, -87, 70, 9, -80, 80, -9, -70, 87, -26, -57, 90, -43},
    {35, -83, 83, -35, -35, 83, -83, 35, 35, -83, 83, -35, -35, 83, -83, 35},
    {26, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -26},
    {18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18},
    {9, -26, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 26, -9},
};

__device__ const std::int16_t kDct32Dev[32][32] = {
    {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64},
    {90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 47, 39, 30, 22, 13, 4, -4, -13, -22, -30, -39, -47, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90},
    {90, 87, 80, 70, 57, 43, 26, 9, -9, -26, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -26, -9, 9, 26, 43, 57, 70, 80, 87, 90},
    {90, 82, 67, 47, 22, -4, -30, -54, -73, -85, -90, -88, -78, -61, -39, -13, 13, 39, 61, 78, 88, 90, 85, 73, 54, 30, 4, -22, -47, -67, -82, -90},
    {89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89},
    {88, 67, 30, -13, -54, -82, -90, -78, -47, -4, 39, 73, 90, 85, 61, 22, -22, -61, -85, -90, -73, -39, 4, 47, 78, 90, 82, 54, 13, -30, -67, -88},
    {87, 57, 9, -43, -80, -90, -70, -26, 26, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80, 90, 70, 26, -26, -70, -90, -80, -43, 9, 57, 87},
    {85, 47, -13, -67, -90, -73, -22, 39, 82, 88, 54, -4, -61, -90, -78, -30, 30, 78, 90, 61, 4, -54, -88, -82, -39, 22, 73, 90, 67, 13, -47, -85},
    {83, 35, -35, -83, -83, -35, 35, 83, 83, 35, -35, -83, -83, -35, 35, 83, 83, 35, -35, -83, -83, -35, 35, 83, 83, 35, -35, -83, -83, -35, 35, 83},
    {82, 22, -54, -90, -61, 13, 78, 85, 30, -47, -90, -67, 4, 73, 88, 39, -39, -88, -73, -4, 67, 90, 47, -30, -85, -78, -13, 61, 90, 54, -22, -82},
    {80, 9, -70, -87, -26, 57, 90, 43, -43, -90, -57, 26, 87, 70, -9, -80, -80, -9, 70, 87, 26, -57, -90, -43, 43, 90, 57, -26, -87, -70, 9, 80},
    {78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 30, 90, 54, -39, -90, -47, 47, 90, 39, -54, -90, -30, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78},
    {75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75},
    {73, -30, -90, -22, 78, 67, -39, -90, -13, 82, 61, -47, -88, -4, 85, 54, -54, -85, 4, 88, 47, -61, -82, 13, 90, 39, -67, -78, 22, 90, 30, -73},
    {70, -43, -87, 9, 90, 26, -80, -57, 57, 80, -26, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90, -26, 80, 57, -57, -80, 26, 90, 9, -87, -43, 70},
    {67, -54, -78, 39, 85, -22, -90, 4, 90, 13, -88, -30, 82, 47, -73, -61, 61, 73, -47, -82, 30, 88, -13, -90, -4, 90, 22, -85, -39, 78, 54, -67},
    {64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64},
    {61, -73, -47, 82, 30, -88, -13, 90, -4, -90, 22, 85, -39, -78, 54, 67, -67, -54, 78, 39, -85, -22, 90, 4, -90, 13, 88, -30, -82, 47, 73, -61},
    {57, -80, -26, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 26, 80, -57, -57, 80, 26, -90, 9, 87, -43, -70, 70, 43, -87, -9, 90, -26, -80, 57},
    {54, -85, -4, 88, -47, -61, 82, 13, -90, 39, 67, -78, -22, 90, -30, -73, 73, 30, -90, 22, 78, -67, -39, 90, -13, -82, 61, 47, -88, 4, 85, -54},
    {50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50, 50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50},
    {47, -90, 39, 54, -90, 30, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, -78, -4, 82, -73, -13, 85, -67, -22, 88, -61, -30, 90, -54, -39, 90, -47},
    {43, -90, 57, 26, -87, 70, 9, -80, 80, -9, -70, 87, -26, -57, 90, -43, -43, 90, -57, -26, 87, -70, -9, 80, -80, 9, 70, -87, 26, 57, -90, 43},
    {39, -88, 73, -4, -67, 90, -47, -30, 85, -78, 13, 61, -90, 54, 22, -82, 82, -22, -54, 90, -61, -13, 78, -85, 30, 47, -90, 67, 4, -73, 88, -39},
    {35, -83, 83, -35, -35, 83, -83, 35, 35, -83, 83, -35, -35, 83, -83, 35, 35, -83, 83, -35, -35, 83, -83, 35, 35, -83, 83, -35, -35, 83, -83, 35},
    {30, -78, 90, -61, 4, 54, -88, 82, -39, -22, 73, -90, 67, -13, -47, 85, -85, 47, 13, -67, 90, -73, 22, 39, -82, 88, -54, -4, 61, -90, 78, -30},
    {26, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -26, -26, 70, -90, 80, -43, -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 26},
    {22, -61, 85, -90, 73, -39, -4, 47, -78, 90, -82, 54, -13, -30, 67, -88, 88, -67, 30, 13, -54, 82, -90, 78, -47, 4, 39, -73, 90, -85, 61, -22},
    {18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18},
    {13, -39, 61, -78, 88, -90, 85, -73, 54, -30, 4, 22, -47, 67, -82, 90, -90, 82, -67, 47, -22, -4, 30, -54, 73, -85, 90, -88, 78, -61, 39, -13},
    {9, -26, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 26, -9, -9, 26, -43, 57, -70, 80, -87, 90, -90, 87, -80, 70, -57, 43, -26, 9},
    {4, -13, 22, -30, 39, -47, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90, 90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 47, -39, 30, -22, 13, -4},
};

__device__ __forceinline__ const std::int16_t* dct_kernel_dev(int n) {
  switch (n) {
    case 4:
      return &kDct4Dev[0][0];
    case 8:
      return &kDct8Dev[0][0];
    case 16:
      return &kDct16Dev[0][0];
    default:
      return &kDct32Dev[0][0];
  }
}

// §4.8 Round2(x, n) = (x + (1 << (n - 1))) >> n, arithmetic shift.
__device__ __forceinline__ int round2_dev(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

__device__ __forceinline__ int clip3_dev(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ int clip_coord(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

// One CUDA block per transform block, one thread per output coefficient.
// Shared memory: kw and kh copies of the integer kernel plus the binary64
// intermediate plane (the largest case, 32x32, is 8 KiB + 4 KiB).
__global__ void forward_dct_kernel(const std::int16_t* __restrict__ residual,
                                   double* __restrict__ coeff, TxParams p) {
  extern __shared__ double s_tmp[];
  __shared__ std::int16_t s_kw[kMaxTxElems];
  __shared__ std::int16_t s_kh[kMaxTxElems];

  const int w = p.w;
  const int h = p.h;
  const int tid = static_cast<int>(threadIdx.x);
  const std::int16_t* kw = dct_kernel_dev(w);
  const std::int16_t* kh = dct_kernel_dev(h);
  for (int t = tid; t < w * w; t += static_cast<int>(blockDim.x)) {
    s_kw[t] = kw[t];
  }
  for (int t = tid; t < h * h; t += static_cast<int>(blockDim.x)) {
    s_kh[t] = kh[t];
  }
  __syncthreads();

  const long long base =
      static_cast<long long>(blockIdx.x) * static_cast<long long>(w * h);

  // Pass 1: tmp[i][v] = scale_w * sum_j kw[v][j] * residual[i][j].
  const int i = tid / w;
  const int v = tid - i * w;
  const std::int16_t* row = residual + base + static_cast<long long>(i) * w;
  double sum = 0.0;
  for (int j = 0; j < w; ++j) {
    sum = __dadd_rn(sum, __dmul_rn(static_cast<double>(s_kw[v * w + j]),
                                   static_cast<double>(row[j])));
  }
  s_tmp[tid] = __dmul_rn(sum, p.scale_w);
  __syncthreads();

  // Pass 2: out[u][v] = scale_h * sum_i kh[u][i] * tmp[i][v].
  const int u = tid / w;
  const int vv = tid - u * w;
  double sum2 = 0.0;
  for (int ii = 0; ii < h; ++ii) {
    sum2 = __dadd_rn(sum2, __dmul_rn(static_cast<double>(s_kh[u * h + ii]),
                                     s_tmp[ii * w + vv]));
  }
  coeff[base + tid] = __dmul_rn(sum2, p.scale_h);
}

__global__ void quantize_kernel(const double* __restrict__ coeff,
                                std::int32_t* __restrict__ levels, TxParams p) {
  const long long idx =
      static_cast<long long>(blockIdx.x) *
          static_cast<long long>(p.w * p.h) +
      static_cast<long long>(threadIdx.x);
  levels[idx] =
      static_cast<std::int32_t>(lround(coeff[idx] * p.quant_factor));
}

__global__ void dequantize_kernel(const std::int32_t* __restrict__ levels,
                                  std::int32_t* __restrict__ dequant,
                                  TxParams p) {
  const long long idx =
      static_cast<long long>(blockIdx.x) *
          static_cast<long long>(p.w * p.h) +
      static_cast<long long>(threadIdx.x);
  const int qc = levels[idx];
  const int abs_qc = qc < 0 ? -qc : qc;
  const int dq_high = (abs_qc * kQuantStep) & 0xFFFFFF;
  const int dq = round2_dev(dq_high, kQuantTableBits);
  const int value = dq / p.dq_denom;
  dequant[idx] = qc < 0 ? -value : value;
}

__global__ void inverse_dct_kernel(const std::int32_t* __restrict__ dequant,
                                   std::int16_t* __restrict__ residual,
                                   TxParams p) {
  extern __shared__ int s_inter[];
  __shared__ std::int16_t s_kw[kMaxTxElems];
  __shared__ std::int16_t s_kh[kMaxTxElems];

  const int w = p.w;
  const int h = p.h;
  const int tid = static_cast<int>(threadIdx.x);
  const std::int16_t* kw = dct_kernel_dev(w);
  const std::int16_t* kh = dct_kernel_dev(h);
  for (int t = tid; t < w * w; t += static_cast<int>(blockDim.x)) {
    s_kw[t] = kw[t];
  }
  for (int t = tid; t < h * h; t += static_cast<int>(blockDim.x)) {
    s_kh[t] = kh[t];
  }
  __syncthreads();

  const long long base =
      static_cast<long long>(blockIdx.x) * static_cast<long long>(w * h);

  // Pass 1 (transposed kernel): intermediate[i][oj].
  const int i = tid / w;
  const int oj = tid - i * w;
  const std::int32_t* dq_row = dequant + base + static_cast<long long>(i) * w;
  long long sum = 0;
  for (int j = 0; j < w; ++j) {
    sum += static_cast<long long>(s_kw[j * w + oj]) *
           static_cast<long long>(dq_row[j]);
  }
  const int rounded = round2_dev(static_cast<int>(sum), p.row_shift);
  s_inter[tid] = clip3_dev(-(1 << 15), (1 << 15) - 1, rounded);
  __syncthreads();

  // Pass 2: residual[oi][j].
  const int oi = tid / w;
  const int j = tid - oi * w;
  long long sum2 = 0;
  for (int ii = 0; ii < h; ++ii) {
    sum2 += static_cast<long long>(s_kh[ii * h + oi]) *
            static_cast<long long>(s_inter[ii * w + j]);
  }
  residual[base + tid] = static_cast<std::int16_t>(
      clip3_dev(-256, 255, round2_dev(static_cast<int>(sum2), p.col_shift)));
}

// 64x64 luma edge block: adjusted 32x32 residual from source samples.
__global__ void make_residual_64x64_kernel(const std::uint8_t* __restrict__ src,
                                           int src_stride, int src_width,
                                           int src_height, int x0, int y0,
                                           int pred,
                                           std::int16_t* __restrict__ out) {
  const int tid = static_cast<int>(threadIdx.x);
  const int j = tid / 32;
  const int i = tid - j * 32;
  int sum = 0;
  for (int dy = 0; dy < 2; ++dy) {
    const int sy = clip_coord(y0 + 2 * j + dy, 0, src_height - 1);
    const long long row_base = static_cast<long long>(sy) * src_stride;
    for (int dx = 0; dx < 2; ++dx) {
      const int sx = clip_coord(x0 + 2 * i + dx, 0, src_width - 1);
      sum += static_cast<int>(src[row_base + sx]) - pred;
    }
  }
  out[tid] = static_cast<std::int16_t>((sum + 2) >> 2);
}

__global__ void downsample_residual_64x64_kernel(
    const std::int16_t* __restrict__ residual64,
    std::int16_t* __restrict__ residual32) {
  const int tid = static_cast<int>(threadIdx.x);
  const int j = tid / 32;
  const int i = tid - j * 32;
  int sum = 0;
  for (int dy = 0; dy < 2; ++dy) {
    const long long row =
        static_cast<long long>(2 * j + dy) * 64 + static_cast<long long>(2 * i);
    sum += static_cast<int>(residual64[row]);
    sum += static_cast<int>(residual64[row + 1]);
  }
  residual32[tid] = static_cast<std::int16_t>((sum + 2) >> 2);
}

__global__ void reconstruct_64x64_kernel(const std::int16_t* __restrict__ residual,
                                         int pred, std::uint8_t* __restrict__ dst,
                                         int dst_stride, int dst_width,
                                         int dst_height, int x0, int y0) {
  const int tid = static_cast<int>(threadIdx.x);
  const int j = tid / 32;
  const int i = tid - j * 32;
  const int value = clip3_dev(0, 255, pred + static_cast<int>(residual[tid]));
  for (int dy = 0; dy < 2; ++dy) {
    const int yy = y0 + 2 * j + dy;
    if (yy < 0 || yy >= dst_height) {
      continue;
    }
    const long long row = static_cast<long long>(yy) * dst_stride;
    for (int dx = 0; dx < 2; ++dx) {
      const int xx = x0 + 2 * i + dx;
      if (xx >= 0 && xx < dst_width) {
        dst[row + xx] = static_cast<std::uint8_t>(value);
      }
    }
  }
}

// §7.15.4.1 transform_shifts(), copied from the CPU reference.
void transform_shifts(int w, int h, int* row_shift, int* col_shift) {
  if (w == 32 && h == 32) {
    *row_shift = 6;
    *col_shift = 13;
  } else if (w == 16 && h == 16) {
    *row_shift = 6;
    *col_shift = 13;
  } else if ((w == 32 && h == 16) || (w == 16 && h == 32)) {
    *row_shift = 6;
    *col_shift = 12;
  } else if ((w == 32 && h == 8) || (w == 8 && h == 32)) {
    *row_shift = 6;
    *col_shift = 13;
  } else if ((w == 16 && h == 4) || (w == 4 && h == 16)) {
    *row_shift = 6;
    *col_shift = 12;
  } else if (w <= 16 && h <= 16) {
    *row_shift = 7;
    *col_shift = 11;
  } else {
    *row_shift = 6;
    *col_shift = 13;
  }
}

bool valid_tx_params(const TxParams& p) {
  return tx_size_supported(p.w, p.h) &&
         (p.dq_denom == 1 || p.dq_denom == 2 || p.dq_denom == 4) &&
         p.row_shift > 0 && p.row_shift < 31 && p.col_shift > 0 &&
         p.col_shift < 31;
}

bool valid_batch(const TxParams& p, int nblocks) {
  return valid_tx_params(p) && nblocks >= 0;
}

}  // namespace

bool tx_size_supported(int w, int h) {
  const auto ok = [](int n) {
    return n == 4 || n == 8 || n == 16 || n == 32;
  };
  return ok(w) && ok(h);
}

std::optional<TxParams> make_tx_params(int coded_w, int coded_h, bool luma) {
  if (coded_w <= 0 || coded_h <= 0) {
    return std::nullopt;
  }
  int tw = coded_w;
  int th = coded_h;
  int upscale_x = 1;
  int upscale_y = 1;
  if (coded_w == 64 && coded_h == 64 && luma) {
    // TX_64X64 is coded through Adjusted_Tx_Size = TX_32X32; the 32x32
    // residual is duplicated to 64x64 by the normative transform process.
    tw = 32;
    th = 32;
    upscale_x = 2;
    upscale_y = 2;
  }
  if (!tx_size_supported(tw, th)) {
    return std::nullopt;
  }
  TxParams p;
  p.w = tw;
  p.h = th;
  const int tx_pels = (upscale_x * tw) * (upscale_y * th);
  p.dq_denom = 1 << ((tx_pels > 256) + (tx_pels > 1024));
  int row_shift = 0;
  int col_shift = 0;
  transform_shifts(tw, th, &row_shift, &col_shift);
  p.row_shift = row_shift;
  p.col_shift = col_shift;
  p.scale_w = 1.0 / (64.0 * std::sqrt(static_cast<double>(tw)));
  p.scale_h = 1.0 / (64.0 * std::sqrt(static_cast<double>(th)));
  const double inv_scale =
      (64.0 * std::sqrt(static_cast<double>(tw)) /
       static_cast<double>(1 << row_shift)) *
      (64.0 * std::sqrt(static_cast<double>(th)) /
       static_cast<double>(1 << col_shift));
  p.quant_factor = 8.0 * static_cast<double>(p.dq_denom) /
                   (inv_scale * static_cast<double>(kQuantStep));
  return p;
}

cudaError_t forward_dct(const std::int16_t* d_residual, double* d_coeff,
                        const TxParams& p, int nblocks, cudaStream_t stream) {
  if (!valid_batch(p, nblocks)) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  if (d_residual == nullptr || d_coeff == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int elems = p.w * p.h;
  const std::size_t smem = static_cast<std::size_t>(elems) * sizeof(double);
  forward_dct_kernel<<<static_cast<unsigned>(nblocks),
                       static_cast<unsigned>(elems), smem, stream>>>(
      d_residual, d_coeff, p);
  return cudaGetLastError();
}

cudaError_t quantize(const double* d_coeff, std::int32_t* d_levels,
                     const TxParams& p, int nblocks, cudaStream_t stream) {
  if (!valid_batch(p, nblocks)) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  if (d_coeff == nullptr || d_levels == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int elems = p.w * p.h;
  quantize_kernel<<<static_cast<unsigned>(nblocks),
                    static_cast<unsigned>(elems), 0, stream>>>(
      d_coeff, d_levels, p);
  return cudaGetLastError();
}

cudaError_t dequantize(const std::int32_t* d_levels, std::int32_t* d_dequant,
                       const TxParams& p, int nblocks, cudaStream_t stream) {
  if (!valid_batch(p, nblocks)) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  if (d_levels == nullptr || d_dequant == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int elems = p.w * p.h;
  dequantize_kernel<<<static_cast<unsigned>(nblocks),
                      static_cast<unsigned>(elems), 0, stream>>>(
      d_levels, d_dequant, p);
  return cudaGetLastError();
}

cudaError_t inverse_dct(const std::int32_t* d_dequant,
                        std::int16_t* d_residual, const TxParams& p,
                        int nblocks, cudaStream_t stream) {
  if (!valid_batch(p, nblocks)) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0) {
    return cudaSuccess;
  }
  if (d_dequant == nullptr || d_residual == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int elems = p.w * p.h;
  const std::size_t smem = static_cast<std::size_t>(elems) * sizeof(int);
  inverse_dct_kernel<<<static_cast<unsigned>(nblocks),
                       static_cast<unsigned>(elems), smem, stream>>>(
      d_dequant, d_residual, p);
  return cudaGetLastError();
}

cudaError_t forward_transform(const std::int16_t* d_residual,
                              double* d_coeff_scratch,
                              std::int32_t* d_levels, const TxParams& p,
                              int nblocks, cudaStream_t stream) {
  const cudaError_t err =
      forward_dct(d_residual, d_coeff_scratch, p, nblocks, stream);
  if (err != cudaSuccess) {
    return err;
  }
  return quantize(d_coeff_scratch, d_levels, p, nblocks, stream);
}

cudaError_t inverse_transform(const std::int32_t* d_levels,
                              std::int32_t* d_dequant_scratch,
                              std::int16_t* d_residual, const TxParams& p,
                              int nblocks, cudaStream_t stream) {
  const cudaError_t err =
      dequantize(d_levels, d_dequant_scratch, p, nblocks, stream);
  if (err != cudaSuccess) {
    return err;
  }
  return inverse_dct(d_dequant_scratch, d_residual, p, nblocks, stream);
}

cudaError_t make_residual_64x64(const std::uint8_t* d_src, int src_stride,
                                int src_width, int src_height, int x0, int y0,
                                int pred, std::int16_t* d_residual,
                                cudaStream_t stream) {
  if (d_src == nullptr || d_residual == nullptr || src_stride <= 0 ||
      src_width <= 0 || src_height <= 0 || pred < 0 || pred > 255) {
    return cudaErrorInvalidValue;
  }
  make_residual_64x64_kernel<<<1, 1024, 0, stream>>>(
      d_src, src_stride, src_width, src_height, x0, y0, pred, d_residual);
  return cudaGetLastError();
}

cudaError_t downsample_residual_64x64(const std::int16_t* d_residual64,
                                      std::int16_t* d_residual32,
                                      cudaStream_t stream) {
  if (d_residual64 == nullptr || d_residual32 == nullptr) {
    return cudaErrorInvalidValue;
  }
  downsample_residual_64x64_kernel<<<1, 1024, 0, stream>>>(d_residual64,
                                                           d_residual32);
  return cudaGetLastError();
}

cudaError_t reconstruct_64x64(const std::int16_t* d_residual, int pred,
                              std::uint8_t* d_dst, int dst_stride,
                              int dst_width, int dst_height, int x0, int y0,
                              cudaStream_t stream) {
  if (d_residual == nullptr || d_dst == nullptr || dst_stride <= 0 ||
      dst_width <= 0 || dst_height <= 0 || pred < 0 || pred > 255) {
    return cudaErrorInvalidValue;
  }
  reconstruct_64x64_kernel<<<1, 1024, 0, stream>>>(
      d_residual, pred, d_dst, dst_stride, dst_width, dst_height, x0, y0);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
