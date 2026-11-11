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
//   * Scoring/decision: m1_still.cpp choose_uv_mode() — DC is scored on U
//     only, CfL on U+V, and ties keep DC.
//
// All arithmetic is integer, so there is no FP contraction risk. The helpers
// in src/encoder/cfl/cfl_common.h (Round2/Round2Signed, Div_Lut,
// resolve_division, ds_weight) are shared with the CPU reference.

#include "m1_chroma.h"

#include "../../../encoder/cfl/cfl_common.h"

namespace av2enc::cuda::m1 {
namespace {

constexpr int kBitDepth = 8;   // M1 is 8-bit 4:2:0 only.
constexpr int kSubX = 1;
constexpr int kSubY = 1;
constexpr int kMid = 1 << (kBitDepth - 1);
constexpr int kMaxDim = av2enc::cfl::kMaxDim;  // 64
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ int clamp_int(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ int max_int(int a, int b) { return a > b ? a : b; }

__device__ __forceinline__ int min_int(int a, int b) { return a < b ? a : b; }

__device__ __forceinline__ int plane_at(const std::uint8_t* plane, int stride,
                                        int row, int col) {
  return static_cast<int>(plane[row * stride + col]);
}

// ---------------------------------------------------------------------------
// §7.13.2.2 / §7.13.2.10 DC prediction (CPU predict_dc + prepare_edges).
//
// The CPU sets have_above = (y > 0) and have_left = (x > 0) for the chroma
// predict_block() call; the above/left samples are the reconstructed coded
// plane clamped to max_x/max_y. The above-right / below-left counts only feed
// the directional modes and do not influence the DC sums.
// ---------------------------------------------------------------------------
__device__ int dc_predict(const std::uint8_t* plane, int stride, int x, int y,
                          int w, int h, int max_x, int max_y) {
  const bool have_above = y > 0;
  const bool have_left = x > 0;
  if (!have_above && !have_left) {
    return kMid;
  }

  int value = 0;
  if (have_left && have_above) {
    int sum = 0;
    for (int k = 0; k < h; ++k) {
      sum += plane_at(plane, stride, min_int(max_y, y + k), x - 1);
    }
    for (int k = 0; k < w; ++k) {
      sum += plane_at(plane, stride, y - 1, min_int(max_x, x + k));
    }
    value = av2enc::cfl::clip1(av2enc::cfl::approx_divide(sum, w + h),
                               kBitDepth);
  } else if (have_left) {
    int sum = 0;
    for (int k = 0; k < h; ++k) {
      sum += plane_at(plane, stride, min_int(max_y, y + k), x - 1);
    }
    value = av2enc::cfl::round2(sum, av2enc::cfl::floor_log2(h));
  } else {
    int sum = 0;
    for (int k = 0; k < w; ++k) {
      sum += plane_at(plane, stride, y - 1, min_int(max_x, x + k));
    }
    value = av2enc::cfl::round2(sum, av2enc::cfl::floor_log2(w));
  }
  return value;
}

// ---------------------------------------------------------------------------
// §7.13.5 luma down-sampling (CPU cfl.cpp filtered_above/filtered_left).
// ---------------------------------------------------------------------------
__device__ __forceinline__ int filtered_above(const M1ChromaJob& job,
                                              int filter, int i) {
  const int x = job.x << kSubX;
  const int y = job.y << kSubY;
  int t = 0;
  for (int dy = -kSubY; dy <= kSubY; ++dy) {
    const int row = clamp_int(
        0, job.luma_max_y,
        max_int(job.sb_top_y - 1, y - 1 - kSubY + dy));
    for (int dx = -kSubX; dx <= kSubX; ++dx) {
      const int offset = max_int(0, (i << kSubX) + dx);
      const int col = clamp_int(0, job.luma_max_x, x + offset);
      const int weight =
          av2enc::cfl::ds_weight(filter, dy, dx, kSubX, kSubY);
      if (weight != 0) {
        t += weight * plane_at(job.luma, job.luma_stride, row, col);
      }
    }
  }
  return t;
}

__device__ __forceinline__ int filtered_left(const M1ChromaJob& job,
                                             int filter, int j) {
  const int x = job.x << kSubX;
  const int y = job.y << kSubY;
  int t = 0;
  for (int dy = -kSubY; dy <= kSubY; ++dy) {
    const int offset = max_int(0, (j << kSubY) + dy);
    const int row = clamp_int(0, job.luma_max_y, y + offset);
    for (int dx = -kSubX; dx <= kSubX; ++dx) {
      const int col =
          clamp_int(0, job.luma_max_x, x - 1 - kSubX + dx);
      const int weight =
          av2enc::cfl::ds_weight(filter, dy, dx, kSubX, kSubY);
      if (weight != 0) {
        t += weight * plane_at(job.luma, job.luma_stride, row, col);
      }
    }
  }
  return t;
}

// §7.13.5 lumaAvg + the lumaAbove/lumaLeft lines used by derived alpha.
// Identical to cfl::compute_luma() for the M1 geometry.
__device__ void compute_luma_lines(const M1ChromaJob& job, int filter,
                                   int* above, int* left, int* out_avg) {
  const int w = job.w;
  const int h = job.h;
  const int step_w = w > 32 ? 2 : 1;
  const int step_h = h > 32 ? 2 : 1;
  const int limit_w = clamp_int(0, w, job.coded_width - job.x);
  const int limit_h = clamp_int(0, h, job.coded_height - job.y);
  const bool have_above = job.y >= 4;  // cfl_avail_above(y, 1)
  const bool have_left = job.x >= 4;   // cfl_avail_left(x, 1)

  int luma_sum = 0;
  int luma_count = 0;

  if (have_above) {
    for (int i = 0; i < w; ++i) {
      int t;
      if (i < limit_w) {
        t = filtered_above(job, filter, i);
      } else if (limit_w > 0) {
        t = filtered_above(job, filter, limit_w - 1);
      } else {
        t = kMid;
      }
      if (i % step_w == 0) {
        luma_sum += t;
      }
      above[i] = t >> 3;
    }
    luma_count += w / step_w;
  }
  if (have_left) {
    for (int j = 0; j < h; ++j) {
      int t;
      if (j < limit_h) {
        t = filtered_left(job, filter, j);
      } else if (limit_h > 0) {
        t = filtered_left(job, filter, limit_h - 1);
      } else {
        t = kMid;
      }
      if (j % step_h == 0) {
        luma_sum += t;
      }
      left[j] = t >> 3;
    }
    luma_count += h / step_h;
  }

  int avg = 8 << kBitDepth;
  if (luma_count > 0) {
    const int average = av2enc::cfl::approx_divide(luma_sum, luma_count);
    const int maximum = (8 << kBitDepth) - 1;
    avg = average < maximum ? average : maximum;
  }
  *out_avg = avg;
}

// §7.13.5 CFL_DERIVED_ALPHA, identical to cfl::derive_implicit_alpha().
__device__ int derive_alpha(const M1ChromaJob& job, const std::uint8_t* chroma,
                            const int* above, const int* left) {
  const int w = job.w;
  const int h = job.h;
  const bool have_above = job.y >= 4;
  const bool have_left = job.x >= 4;

  int num_above = 0;
  int num_left = 0;
  if (have_above && have_left) {
    if (w > h * 2) {
      num_above = av2enc::cfl::kNumRefSamCfl;
    } else if (h > w * 2) {
      num_left = av2enc::cfl::kNumRefSamCfl;
    } else {
      num_above = av2enc::cfl::kNumRefSamCfl >> 1;
      num_left = av2enc::cfl::kNumRefSamCfl >> 1;
    }
  } else {
    num_above = have_above ? av2enc::cfl::kNumRefSamCfl : 0;
    num_left = have_left ? av2enc::cfl::kNumRefSamCfl : 0;
  }
  num_above = min_int(num_above, w);
  num_left = min_int(num_left, h);

  int count = 0;
  long long sum_x = 0;
  long long sum_y = 0;
  long long sum_xy = 0;
  long long sum_xx = 0;

  if (num_above > 0) {
    const int step = w / num_above;
    const int limit = clamp_int(0, w, job.coded_width - job.x);
    const int row = job.y - 1;
    for (int i = step >> 1; i < w; i += step) {
      int sample;
      if (i < limit) {
        sample = plane_at(chroma, job.chroma_stride, row, job.x + i);
      } else if (limit > 0) {
        sample = plane_at(chroma, job.chroma_stride, row, job.x + limit - 1);
      } else {
        sample = kMid;
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
    const int limit = clamp_int(0, h, job.coded_height - job.y);
    const int col = job.x - 1;
    for (int j = step >> 1; j < h; j += step) {
      int sample;
      if (j < limit) {
        sample =
            plane_at(chroma, job.chroma_stride, job.y + j, col);
      } else if (limit > 0) {
        sample = plane_at(chroma, job.chroma_stride, job.y + limit - 1, col);
      } else {
        sample = kMid;
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
  return av2enc::cfl::resolve_division(static_cast<int>(nor),
                                       static_cast<int>(der), 8);
}

// §7.13.5 array L for one sample (CPU compute_luma() inner loop).
__device__ __forceinline__ int sample_luma(const M1ChromaJob& job, int filter,
                                           int i, int j) {
  const int luma_y = (job.y + i) << kSubY;
  const int luma_x = (job.x + j) << kSubX;
  const bool clamp_y = (i == 0) || (luma_y % 64 == 0);
  const bool clamp_x = (j == 0) || (luma_x % 64 == 0);
  int t = 0;
  for (int dy = -kSubY; dy <= kSubY; ++dy) {
    const int clamp_dy = clamp_y ? max_int(dy, 0) : dy;
    const int row = clamp_int(0, job.luma_max_y, luma_y + clamp_dy);
    for (int dx = -kSubX; dx <= kSubX; ++dx) {
      const int clamp_dx = clamp_x ? max_int(dx, 0) : dx;
      const int col = clamp_int(0, job.luma_max_x, luma_x + clamp_dx);
      const int weight =
          av2enc::cfl::ds_weight(filter, dy, dx, kSubX, kSubY);
      if (weight != 0) {
        t += weight * plane_at(job.luma, job.luma_stride, row, col);
      }
    }
  }
  return t;
}

__device__ __forceinline__ int cfl_sample(int dc, int alpha, int luma_value,
                                          int luma_avg) {
  const int scaled = av2enc::cfl::round2_signed(
      alpha * (luma_value - luma_avg), av2enc::cfl::kScaleShift);
  return av2enc::cfl::clip1(dc + scaled, kBitDepth);
}

__device__ __forceinline__ long long warp_reduce(long long value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// Shared per-block state. Kept out of the locals so the reduction and the
// prediction writeback can be driven by every thread.
struct BlockState {
  int above[kMaxDim];
  int left[kMaxDim];
  int luma_avg;
  int alpha_u;
  int alpha_v;
  int dc_u;
  int dc_v;
  int is_cfl;
};

__device__ void run_block(const M1ChromaJob& job, const M1ChromaOptions& opt,
                          M1ChromaResult* result) {
  __shared__ BlockState state;
  __shared__ long long warp_sums[kWarps][4];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int filter = opt.ds_filter == 3 ? 0 : opt.ds_filter;
  const int w = job.w;
  const int h = job.h;
  const int max_x = job.coded_width - 1;
  const int max_y = job.coded_height - 1;

  if (tid == 0) {
    state.dc_u = dc_predict(job.chroma_u, job.chroma_stride, job.x, job.y, w,
                            h, max_x, max_y);
    state.dc_v = dc_predict(job.chroma_v, job.chroma_stride, job.x, job.y, w,
                            h, max_x, max_y);
    compute_luma_lines(job, filter, state.above, state.left, &state.luma_avg);
    int alpha_u = derive_alpha(job, job.chroma_u, state.above, state.left);
    int alpha_v = derive_alpha(job, job.chroma_v, state.above, state.left);
    alpha_u = clamp_int(opt.alpha_q8_min, opt.alpha_q8_max, alpha_u);
    alpha_v = clamp_int(opt.alpha_q8_min, opt.alpha_q8_max, alpha_v);
    state.alpha_u = alpha_u;
    state.alpha_v = alpha_v;
    state.is_cfl = 0;
  }
  __syncthreads();

  // Score the DC candidate on U only and the CfL candidate on U+V, exactly as
  // choose_uv_mode() does.
  long long dc_u = 0;
  long long dc_v = 0;
  long long cfl_u = 0;
  long long cfl_v = 0;
  const int total = w * h;
  for (int index = tid; index < total; index += kThreads) {
    const int i = index / w;
    const int j = index - i * w;
    const int sy = min_int(job.src_max_y, job.y + i);
    const int sx = min_int(job.src_max_x, job.x + j);
    const int src_u = plane_at(job.src_u, job.src_stride_u, sy, sx);
    const int src_v = plane_at(job.src_v, job.src_stride_v, sy, sx);
    const int diff_dc_u = src_u - state.dc_u;
    const int diff_dc_v = src_v - state.dc_v;
    dc_u += static_cast<long long>(diff_dc_u) * diff_dc_u;
    dc_v += static_cast<long long>(diff_dc_v) * diff_dc_v;
    if (opt.enable_cfl != 0) {
      const int luma_value = sample_luma(job, filter, i, j);
      const int pred_u = cfl_sample(state.dc_u, state.alpha_u, luma_value,
                                    state.luma_avg);
      const int pred_v = cfl_sample(state.dc_v, state.alpha_v, luma_value,
                                    state.luma_avg);
      const int diff_cfl_u = src_u - pred_u;
      const int diff_cfl_v = src_v - pred_v;
      cfl_u += static_cast<long long>(diff_cfl_u) * diff_cfl_u;
      cfl_v += static_cast<long long>(diff_cfl_v) * diff_cfl_v;
    }
  }
  __syncthreads();

  dc_u = warp_reduce(dc_u);
  dc_v = warp_reduce(dc_v);
  cfl_u = warp_reduce(cfl_u);
  cfl_v = warp_reduce(cfl_v);
  if (lane == 0) {
    warp_sums[warp][0] = dc_u;
    warp_sums[warp][1] = dc_v;
    warp_sums[warp][2] = cfl_u;
    warp_sums[warp][3] = cfl_v;
  }
  __syncthreads();

  if (tid == 0) {
    long long total_dc_u = 0;
    long long total_dc_v = 0;
    long long total_cfl_u = 0;
    long long total_cfl_v = 0;
    for (int i = 0; i < kWarps; ++i) {
      total_dc_u += warp_sums[i][0];
      total_dc_v += warp_sums[i][1];
      total_cfl_u += warp_sums[i][2];
      total_cfl_v += warp_sums[i][3];
    }

    // choose_uv_mode(): strict `<`, DC evaluated first, so a tie keeps DC.
    const long long cfl_total = total_cfl_u + total_cfl_v;
    const bool is_cfl = opt.enable_cfl != 0 && cfl_total < total_dc_u;
    state.is_cfl = is_cfl ? 1 : 0;

    M1ChromaResult r{};
    r.winner = is_cfl ? kM1ChromaCfl : kM1ChromaDc;
    r.uv_mode = 0;  // YMode DC; CfL reuses the DC uv mode + is_cfl
    r.angle_delta = 0;
    r.is_cfl = is_cfl ? 1 : 0;
    r.alpha_u = state.alpha_u;
    r.alpha_v = state.alpha_v;
    r.dc_u = state.dc_u;
    r.dc_v = state.dc_v;
    r.sse_dc_u = total_dc_u;
    r.sse_dc_v = total_dc_v;
    r.sse_cfl_u = total_cfl_u;
    r.sse_cfl_v = total_cfl_v;
    if (opt.enable_cfl == 0) {
      r.sse = 0;  // CPU early-out path
    } else {
      r.sse = is_cfl ? cfl_total : total_dc_u;
    }
    result[0] = r;
  }
  __syncthreads();

  // Write the winning prediction (CPU predict_block / cfl apply_prediction).
  const bool is_cfl = state.is_cfl != 0;
  const int pred_stride = job.pred_stride > 0 ? job.pred_stride : w;
  for (int index = tid; index < total; index += kThreads) {
    const int i = index / w;
    const int j = index - i * w;
    int pred_u = state.dc_u;
    int pred_v = state.dc_v;
    if (is_cfl) {
      const int luma_value = sample_luma(job, filter, i, j);
      pred_u = cfl_sample(state.dc_u, state.alpha_u, luma_value,
                          state.luma_avg);
      pred_v = cfl_sample(state.dc_v, state.alpha_v, luma_value,
                          state.luma_avg);
    }
    const int offset = i * pred_stride + j;
    if (job.pred_u != nullptr) {
      job.pred_u[offset] = static_cast<std::uint8_t>(pred_u);
    }
    if (job.pred_v != nullptr) {
      job.pred_v[offset] = static_cast<std::uint8_t>(pred_v);
    }
  }
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

cudaError_t validate_job(const M1ChromaJob& job,
                         const M1ChromaOptions& options) {
  if (options.enable_uv_directional != 0 || options.enable_uv_full != 0) {
    // Directional/smooth UV is verified on the CPU but the GPU candidate loop
    // is not wired yet; reject instead of diverging.
    return cudaErrorNotSupported;
  }
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
  // Note: a bottom/right-edge block may extend past the coded plane
  // (MiCols*MI_SIZE, MiRows*MI_SIZE): the decoder parses and reconstructs the
  // whole transform block into the padded plane, and CfL truncates its coded
  // neighbour replication at coded_width/coded_height. Do not reject that.
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
  if (options.enable_uv_directional != 0 || options.enable_uv_full != 0) {
    return cudaErrorNotSupported;
  }
  if (options.ds_filter < 0 || options.ds_filter > 3 ||
      options.alpha_q8_min < -511 || options.alpha_q8_max > 511 ||
      options.alpha_q8_min > options.alpha_q8_max) {
    return cudaErrorInvalidValue;
  }
  decide_batch_kernel<<<job_count, kThreads, 0, stream>>>(d_jobs, options,
                                                         d_results);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::m1
