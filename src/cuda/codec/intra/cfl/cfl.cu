// CUDA CfL kernel — AV2 v1.0.0 §7.13.5.
//
// Integer only and shared with the CPU reference through cfl_common.h, so the
// only way this file can disagree with cfl.cpp is a bookkeeping mistake in the
// luma-average collection or the edge replication. cfl_test.cu compares the
// two paths byte for byte, including partial blocks and all four
// cfl_ds_filter_index values.

#include "cfl.h"

namespace av2enc::cuda {
namespace {

using av2enc::cfl::Block;

constexpr int kThreads = 128;
constexpr int kMaxDim = av2enc::cfl::kMaxDim;

__device__ __forceinline__ int clamp_int(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ int max_int(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ int min_int(int a, int b) { return a < b ? a : b; }

__device__ __forceinline__ int plane_at(const Block& b, int row, int col) {
  return static_cast<int>(b.luma[row * b.luma_stride + col]);
}

__device__ __forceinline__ int filtered_above(const Block& b, int filter,
                                              int i) {
  const int sub_x = b.sub_x;
  const int sub_y = b.sub_y;
  const int x = b.x << sub_x;
  const int y = b.y << sub_y;
  int t = 0;
  for (int dy = -sub_y; dy <= sub_y; ++dy) {
    const int row =
        clamp_int(0, b.luma_max_y,
                  max_int(b.sb_top_y - 1, y - 1 - sub_y + dy));
    for (int dx = -sub_x; dx <= sub_x; ++dx) {
      const int offset = max_int(0, (i << sub_x) + dx);
      const int col = clamp_int(0, b.luma_max_x, x + offset);
      const int weight =
          av2enc::cfl::ds_weight(filter, dy, dx, sub_x, sub_y);
      if (weight != 0) {
        t += weight * plane_at(b, row, col);
      }
    }
  }
  return t;
}

__device__ __forceinline__ int filtered_left(const Block& b, int filter,
                                             int j) {
  const int sub_x = b.sub_x;
  const int sub_y = b.sub_y;
  const int x = b.x << sub_x;
  const int y = b.y << sub_y;
  int t = 0;
  for (int dy = -sub_y; dy <= sub_y; ++dy) {
    const int offset = max_int(0, (j << sub_y) + dy);
    const int row = clamp_int(0, b.luma_max_y, y + offset);
    for (int dx = -sub_x; dx <= sub_x; ++dx) {
      const int col = clamp_int(0, b.luma_max_x, x - 1 - sub_x + dx);
      const int weight =
          av2enc::cfl::ds_weight(filter, dy, dx, sub_x, sub_y);
      if (weight != 0) {
        t += weight * plane_at(b, row, col);
      }
    }
  }
  return t;
}

// §7.13.5 lumaAvg plus the lumaAbove/lumaLeft lines used by derived alpha.
// Runs in thread 0; w + h <= 128 taps.
__device__ void compute_lines(const Block& b, int* above, int* left,
                              int* out_avg) {
  using namespace av2enc::cfl;
  const int filter = (b.ds_filter == 3) ? 0 : b.ds_filter;
  const int init = 1 << (b.bit_depth - 1);
  const int step_w = b.w > 32 ? 2 : 1;
  const int step_h = b.h > 32 ? 2 : 1;
  int luma_sum = 0;
  int luma_count = 0;

  if (b.have_above != 0) {
    const int limit = clamp_int(0, b.w, b.coded_width - b.x);
    for (int i = 0; i < b.w; ++i) {
      int t;
      if (i < limit) {
        t = filtered_above(b, filter, i);
      } else if (limit > 0) {
        t = filtered_above(b, filter, limit - 1);
      } else {
        t = init;
      }
      if (i % step_w == 0) {
        luma_sum += t;
      }
      above[i] = t >> 3;
    }
    luma_count += b.w / step_w;
  }
  if (b.have_left != 0) {
    const int limit = clamp_int(0, b.h, b.coded_height - b.y);
    for (int j = 0; j < b.h; ++j) {
      int t;
      if (j < limit) {
        t = filtered_left(b, filter, j);
      } else if (limit > 0) {
        t = filtered_left(b, filter, limit - 1);
      } else {
        t = init;
      }
      if (j % step_h == 0) {
        luma_sum += t;
      }
      left[j] = t >> 3;
    }
    luma_count += b.h / step_h;
  }

  int avg = 8 << (b.bit_depth - 1);
  if (luma_count > 0) {
    const int average = approx_divide(luma_sum, luma_count);
    const int maximum = (8 << b.bit_depth) - 1;
    avg = average < maximum ? average : maximum;
  }
  *out_avg = avg;
}

// §7.13.5 CFL_DERIVED_ALPHA, identical to cfl::derive_implicit_alpha().
__device__ int derive_alpha(const Block& b, const int* above,
                            const int* left) {
  using namespace av2enc::cfl;
  const int w = b.w;
  const int h = b.h;
  int num_above = 0;
  int num_left = 0;
  if (b.have_above != 0 && b.have_left != 0) {
    if (w > h * 2) {
      num_above = kNumRefSamCfl;
    } else if (h > w * 2) {
      num_left = kNumRefSamCfl;
    } else {
      num_above = kNumRefSamCfl >> 1;
      num_left = kNumRefSamCfl >> 1;
    }
  } else {
    num_above = b.have_above != 0 ? kNumRefSamCfl : 0;
    num_left = b.have_left != 0 ? kNumRefSamCfl : 0;
  }
  num_above = min_int(num_above, w);
  num_left = min_int(num_left, h);

  int count = 0;
  long long sum_x = 0;
  long long sum_y = 0;
  long long sum_xy = 0;
  long long sum_xx = 0;
  const int init = 1 << (b.bit_depth - 1);

  if (num_above > 0) {
    const int step = w / num_above;
    const int limit = clamp_int(0, w, b.coded_width - b.x);
    const int row = b.y - 1;
    for (int i = step >> 1; i < w; i += step) {
      int sample;
      if (i < limit) {
        sample = static_cast<int>(b.chroma[row * b.chroma_stride + b.x + i]);
      } else if (limit > 0) {
        sample = static_cast<int>(
            b.chroma[row * b.chroma_stride + b.x + limit - 1]);
      } else {
        sample = init;
      }
      const int lx = above[i];
      sum_x += lx;
      sum_y += sample;
      sum_xy += static_cast<long long>(lx) * sample;
      sum_xx += static_cast<long long>(lx) * lx;
      ++count;
    }
  }
  if (num_left > 0) {
    const int step = h / num_left;
    const int limit = clamp_int(0, h, b.coded_height - b.y);
    const int col = b.x - 1;
    for (int j = step >> 1; j < h; j += step) {
      int sample;
      if (j < limit) {
        sample = static_cast<int>(b.chroma[(b.y + j) * b.chroma_stride + col]);
      } else if (limit > 0) {
        sample = static_cast<int>(
            b.chroma[(b.y + limit - 1) * b.chroma_stride + col]);
      } else {
        sample = init;
      }
      const int lx = left[j];
      sum_x += lx;
      sum_y += sample;
      sum_xy += static_cast<long long>(lx) * sample;
      sum_xx += static_cast<long long>(lx) * lx;
      ++count;
    }
  }

  if (count == 0) {
    return 0;
  }
  const long long der =
      sum_xx - (sum_x * sum_x) / static_cast<long long>(count);
  const long long nor =
      sum_xy - (sum_x * sum_y) / static_cast<long long>(count);
  if (der == 0 || nor == 0) {
    return 0;
  }
  return resolve_division(static_cast<int>(nor), static_cast<int>(der), 8);
}

// §7.13.5 array L for one sample, evaluated inline (no L array is stored).
__device__ __forceinline__ int sample_luma(const Block& b, int filter, int i,
                                           int j) {
  const int sub_x = b.sub_x;
  const int sub_y = b.sub_y;
  const int luma_y = (b.y + i) << sub_y;
  const int luma_x = (b.x + j) << sub_x;
  const bool clamp_y = (i == 0) || (luma_y % 64 == 0);
  const bool clamp_x = (j == 0) || (luma_x % 64 == 0);
  int t = 0;
  for (int dy = -sub_y; dy <= sub_y; ++dy) {
    const int clamp_dy = clamp_y ? max_int(dy, 0) : dy;
    const int row = clamp_int(0, b.luma_max_y, luma_y + clamp_dy);
    for (int dx = -sub_x; dx <= sub_x; ++dx) {
      const int clamp_dx = clamp_x ? max_int(dx, 0) : dx;
      const int col = clamp_int(0, b.luma_max_x, luma_x + clamp_dx);
      const int weight =
          av2enc::cfl::ds_weight(filter, dy, dx, sub_x, sub_y);
      if (weight != 0) {
        t += weight * plane_at(b, row, col);
      }
    }
  }
  return t;
}

__device__ __forceinline__ void cfl_body(const Block& b, int alpha_in,
                                         bool derived, int* alpha_out) {
  __shared__ int s_above[kMaxDim];
  __shared__ int s_left[kMaxDim];
  __shared__ int s_avg;
  __shared__ int s_alpha;
  if (threadIdx.x == 0) {
    compute_lines(b, s_above, s_left, &s_avg);
    s_alpha = derived ? derive_alpha(b, s_above, s_left) : alpha_in;
    if (alpha_out != nullptr) {
      alpha_out[0] = s_alpha;
    }
  }
  __syncthreads();

  if (b.w <= 0 || b.h <= 0) {
    return;
  }
  const int filter = (b.ds_filter == 3) ? 0 : b.ds_filter;
  const int alpha = s_alpha;
  const int avg = s_avg;
  const int total = b.w * b.h;
  const int step = static_cast<int>(blockDim.x);
  for (int index = static_cast<int>(threadIdx.x); index < total;
       index += step) {
    const int i = index / b.w;
    const int j = index % b.w;
    const int luma_value = sample_luma(b, filter, i, j);
    const int scaled = av2enc::cfl::round2_signed(
        alpha * (luma_value - avg), av2enc::cfl::kScaleShift);
    const int pos = (b.y + i) * b.chroma_stride + b.x + j;
    const int dc = static_cast<int>(b.chroma[pos]);
    b.chroma[pos] = static_cast<std::uint8_t>(
        av2enc::cfl::clip1(dc + scaled, b.bit_depth));
  }
}

__global__ void predict_kernel(Block block, int alpha_q8) {
  cfl_body(block, alpha_q8, /*derived=*/false, nullptr);
}

__global__ void predict_derived_kernel(Block block, int* alpha_out) {
  cfl_body(block, 0, /*derived=*/true, alpha_out);
}

__global__ void predict_batch_kernel(const CflJob* jobs) {
  const CflJob& job = jobs[blockIdx.x];
  cfl_body(job.block, job.alpha_q8, /*derived=*/false, nullptr);
}

__global__ void predict_derived_batch_kernel(const Block* blocks,
                                             int* alpha_out) {
  const int index = static_cast<int>(blockIdx.x);
  cfl_body(blocks[index], 0, /*derived=*/true,
           alpha_out == nullptr ? nullptr : alpha_out + index);
}

}  // namespace

cudaError_t cfl_predict(const Block& block, int alpha_q8,
                        cudaStream_t stream) {
  if (block.luma == nullptr || block.chroma == nullptr) {
    return cudaErrorInvalidValue;
  }
  predict_kernel<<<1, kThreads, 0, stream>>>(block, alpha_q8);
  return cudaGetLastError();
}

cudaError_t cfl_predict_derived(const Block& block, int* d_alpha_out,
                                cudaStream_t stream) {
  if (block.luma == nullptr || block.chroma == nullptr) {
    return cudaErrorInvalidValue;
  }
  predict_derived_kernel<<<1, kThreads, 0, stream>>>(block, d_alpha_out);
  return cudaGetLastError();
}

cudaError_t cfl_predict_batch(const CflJob* d_jobs, int job_count,
                              cudaStream_t stream) {
  if (job_count <= 0) {
    return cudaSuccess;
  }
  if (d_jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  predict_batch_kernel<<<job_count, kThreads, 0, stream>>>(d_jobs);
  return cudaGetLastError();
}

cudaError_t cfl_predict_derived_batch(const Block* d_blocks, int* d_alpha_out,
                                      int job_count, cudaStream_t stream) {
  if (job_count <= 0) {
    return cudaSuccess;
  }
  if (d_blocks == nullptr) {
    return cudaErrorInvalidValue;
  }
  predict_derived_batch_kernel<<<job_count, kThreads, 0, stream>>>(
      d_blocks, d_alpha_out);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
