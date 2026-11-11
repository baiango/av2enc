// Device body shared by the M1 GPU chroma decision kernel and its host
// emulation test.
//
// This file contains:
//   * the DC prediction used by the DC candidate and the CfL base;
//   * the §7.13.5 CfL luma-downsample / derived-alpha / apply arithmetic;
//   * the full-UV candidate search (CPU M1Options::uv_full), which runs the
//     shared intra predictor from src/cuda/intra_modes/.
//
// It is a header so the CUDA kernel and the host-emulation parity test execute
// exactly the same integer code.  CUDA keyword stubs are supplied by the
// emulation TU; nothing here allocates or synchronises the stream.

#ifndef AV2ENC_CUDA_M1_CHROMA_DEVICE_CUH_
#define AV2ENC_CUDA_M1_CHROMA_DEVICE_CUH_

#include "m1_chroma.h"

#include "../../../encoder/cfl/cfl_common.h"
#include "../../intra_modes/intra_predict_device.cuh"

namespace av2enc::cuda::m1 {
namespace {

constexpr int kBitDepth = 8;   // M1 is 8-bit 4:2:0 only.
constexpr int kSubX = 1;
constexpr int kSubY = 1;
constexpr int kMid = 1 << (kBitDepth - 1);
constexpr int kMaxDim = av2enc::cfl::kMaxDim;  // 64
constexpr int kThreads = 128;
constexpr long long kSseMax = 0x7fffffffffffffffLL;

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

// §7.13.5 CFL_DERIVED_ALPHA.
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

// §7.13.5 array L for one sample.
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

__device__ __forceinline__ int intra_source_sample(const M1ChromaJob& job,
                                                   bool use_v, int x, int y) {
  const int sx = min_int(job.src_max_x, x);
  const int sy = min_int(job.src_max_y, y);
  return use_v ? plane_at(job.src_v, job.src_stride_v, sy, sx)
               : plane_at(job.src_u, job.src_stride_u, sy, sx);
}

// Generic block reduction used by the full-UV search and its host emulation.
// nthr must be a power of two and <= kThreads.
__device__ __forceinline__ long long block_reduce_sum(long long value,
                                                      long long* scratch,
                                                      int tid, int nthr) {
  scratch[tid] = value;
  __syncthreads();
  for (int stride = nthr >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ __forceinline__ int intra_derived_delta(const M1ChromaJob& job,
                                                   int mode) {
  return (av2enc::cuda::intra_modes::is_directional_mode(job.y_mode) &&
          mode == job.y_mode)
             ? job.y_angle_delta
             : 0;
}

__device__ __forceinline__ void fill_intra_block(const M1ChromaJob& job,
                                                 const std::uint8_t* plane,
                                                 int mode, int delta,
                                                 av2enc::cuda::intra_modes::IntraBlock* b) {
  b->recon_plane = plane;
  b->recon_stride = job.chroma_stride;
  b->x0 = job.x;
  b->y0 = job.y;
  b->width = job.w;
  b->height = job.h;
  b->max_x = job.coded_width - 1;
  b->max_y = job.coded_height - 1;
  b->have_left = job.x > 0 ? 1 : 0;
  b->have_above = job.y > 0 ? 1 : 0;
  b->num4_above_right = job.num4_above_right;
  b->num4_below_left = job.num4_below_left;
  b->plane = 1;
  b->mode = static_cast<av2enc::cuda::intra_modes::IntraMode>(mode);
  b->angle_delta = delta;
  b->filter_type_above = job.filter_type_above != 0 ? 1 : 0;
  b->filter_type_left = job.filter_type_left != 0 ? 1 : 0;
  b->enable_edge_filter = job.enable_edge_filter != 0 ? 1 : 0;
  b->source_plane = nullptr;
  b->source_stride = 0;
  b->source_max_x = 0;
  b->source_max_y = 0;
}

constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ long long warp_reduce_sum(long long value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// Shared state for the default DC/CfL policy.
struct ChromaDcCflShared {
  int above[kMaxDim];
  int left[kMaxDim];
  int luma_avg;
  int alpha_u;
  int alpha_v;
  int dc_u;
  int dc_v;
  int is_cfl;
  long long warp_sums[kWarps][4];
};

__device__ __forceinline__ void chroma_dc_cfl_search(
    const M1ChromaJob& job, const M1ChromaOptions& opt, int tid, int nthr,
    ChromaDcCflShared* s, M1ChromaResult* out) {
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int filter = opt.ds_filter == 3 ? 0 : opt.ds_filter;
  const int w = job.w;
  const int h = job.h;
  const int max_x = job.coded_width - 1;
  const int max_y = job.coded_height - 1;

  if (tid == 0) {
    s->dc_u = dc_predict(job.chroma_u, job.chroma_stride, job.x, job.y, w, h,
                         max_x, max_y);
    s->dc_v = dc_predict(job.chroma_v, job.chroma_stride, job.x, job.y, w, h,
                         max_x, max_y);
    compute_luma_lines(job, filter, s->above, s->left, &s->luma_avg);
    int alpha_u = derive_alpha(job, job.chroma_u, s->above, s->left);
    int alpha_v = derive_alpha(job, job.chroma_v, s->above, s->left);
    alpha_u = clamp_int(opt.alpha_q8_min, opt.alpha_q8_max, alpha_u);
    alpha_v = clamp_int(opt.alpha_q8_min, opt.alpha_q8_max, alpha_v);
    s->alpha_u = alpha_u;
    s->alpha_v = alpha_v;
    s->is_cfl = 0;
  }
  __syncthreads();

  long long dc_u = 0;
  long long dc_v = 0;
  long long cfl_u = 0;
  long long cfl_v = 0;
  const int total = w * h;
  for (int index = tid; index < total; index += nthr) {
    const int i = index / w;
    const int j = index - i * w;
    const int sy = min_int(job.src_max_y, job.y + i);
    const int sx = min_int(job.src_max_x, job.x + j);
    const int src_u = plane_at(job.src_u, job.src_stride_u, sy, sx);
    const int src_v = plane_at(job.src_v, job.src_stride_v, sy, sx);
    const int diff_dc_u = src_u - s->dc_u;
    const int diff_dc_v = src_v - s->dc_v;
    dc_u += static_cast<long long>(diff_dc_u) * diff_dc_u;
    dc_v += static_cast<long long>(diff_dc_v) * diff_dc_v;
    if (opt.enable_cfl != 0) {
      const int luma_value = sample_luma(job, filter, i, j);
      const int pred_u =
          cfl_sample(s->dc_u, s->alpha_u, luma_value, s->luma_avg);
      const int pred_v =
          cfl_sample(s->dc_v, s->alpha_v, luma_value, s->luma_avg);
      const int diff_cfl_u = src_u - pred_u;
      const int diff_cfl_v = src_v - pred_v;
      cfl_u += static_cast<long long>(diff_cfl_u) * diff_cfl_u;
      cfl_v += static_cast<long long>(diff_cfl_v) * diff_cfl_v;
    }
  }
  __syncthreads();

  dc_u = warp_reduce_sum(dc_u);
  dc_v = warp_reduce_sum(dc_v);
  cfl_u = warp_reduce_sum(cfl_u);
  cfl_v = warp_reduce_sum(cfl_v);
  if (lane == 0) {
    s->warp_sums[warp][0] = dc_u;
    s->warp_sums[warp][1] = dc_v;
    s->warp_sums[warp][2] = cfl_u;
    s->warp_sums[warp][3] = cfl_v;
  }
  __syncthreads();

  if (tid == 0) {
    long long total_dc_u = 0;
    long long total_dc_v = 0;
    long long total_cfl_u = 0;
    long long total_cfl_v = 0;
    const int active_warps = (nthr + 31) / 32;
    for (int i = 0; i < active_warps; ++i) {
      total_dc_u += s->warp_sums[i][0];
      total_dc_v += s->warp_sums[i][1];
      total_cfl_u += s->warp_sums[i][2];
      total_cfl_v += s->warp_sums[i][3];
    }

    const long long cfl_total = total_cfl_u + total_cfl_v;
    const bool is_cfl = opt.enable_cfl != 0 && cfl_total < total_dc_u;
    s->is_cfl = is_cfl ? 1 : 0;

    M1ChromaResult r{};
    r.winner = is_cfl ? kM1ChromaCfl : kM1ChromaDc;
    r.uv_mode = 0;
    r.angle_delta = 0;
    r.is_cfl = is_cfl ? 1 : 0;
    r.alpha_u = s->alpha_u;
    r.alpha_v = s->alpha_v;
    r.dc_u = s->dc_u;
    r.dc_v = s->dc_v;
    r.sse_dc_u = total_dc_u;
    r.sse_dc_v = total_dc_v;
    r.sse_cfl_u = total_cfl_u;
    r.sse_cfl_v = total_cfl_v;
    r.sse = opt.enable_cfl == 0 ? 0
                                : (is_cfl ? cfl_total : total_dc_u);
    out[0] = r;
  }
  __syncthreads();

  const bool is_cfl = s->is_cfl != 0;
  const int pred_stride = job.pred_stride > 0 ? job.pred_stride : w;
  for (int index = tid; index < total; index += nthr) {
    const int i = index / w;
    const int j = index - i * w;
    int pred_u = s->dc_u;
    int pred_v = s->dc_v;
    if (is_cfl) {
      const int luma_value = sample_luma(job, filter, i, j);
      pred_u = cfl_sample(s->dc_u, s->alpha_u, luma_value, s->luma_avg);
      pred_v = cfl_sample(s->dc_v, s->alpha_v, luma_value, s->luma_avg);
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

struct ChromaUvShared {
  av2enc::cuda::intra_modes::SharedScratch intra;
  long long reduce[kThreads];
  std::uint8_t pred[static_cast<std::size_t>(kMaxDim) * kMaxDim];
  long long best_sse;
  int best_mode;
  int best_delta;
  int best_is_cfl;
  long long sse_dc_u;
  long long sse_dc_v;
  long long sse_cfl_u;
  long long sse_cfl_v;
  int dc_u;
  int dc_v;
  int alpha_u;
  int alpha_v;
  int luma_avg;
};

// Full-UV candidate search, matching CPU m1_still.cpp choose_uv_mode() with
// options_.uv_full == true:
//   * all 13 base modes, U+V SSE, DC first, strict '<' tie-break;
//   * AngleDeltaUV derived from AngleDeltaY only when UVMode == YMode;
//   * CfL last, U+V SSE, strict '<'.
__device__ __forceinline__ void chroma_uv_full_search(
    const M1ChromaJob& job, const M1ChromaOptions& opt, ChromaUvShared* s,
    M1ChromaResult* out) {
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  const int filter = opt.ds_filter == 3 ? 0 : opt.ds_filter;

  if (tid == 0) {
    s->dc_u = dc_predict(job.chroma_u, job.chroma_stride, job.x, job.y, job.w,
                         job.h, job.coded_width - 1, job.coded_height - 1);
    s->dc_v = dc_predict(job.chroma_v, job.chroma_stride, job.x, job.y, job.w,
                         job.h, job.coded_width - 1, job.coded_height - 1);
    compute_luma_lines(job, filter, s->intra.above + 0, s->intra.left + 0,
                       &s->luma_avg);
    s->alpha_u =
        clamp_int(opt.alpha_q8_min, opt.alpha_q8_max,
                  derive_alpha(job, job.chroma_u, s->intra.above + 0,
                               s->intra.left + 0));
    s->alpha_v =
        clamp_int(opt.alpha_q8_min, opt.alpha_q8_max,
                  derive_alpha(job, job.chroma_v, s->intra.above + 0,
                               s->intra.left + 0));
    s->best_sse = kSseMax;
    s->best_mode = 0;
    s->best_delta = 0;
    s->best_is_cfl = 0;
    s->sse_dc_u = 0;
    s->sse_dc_v = 0;
    s->sse_cfl_u = 0;
    s->sse_cfl_v = 0;
  }
  __syncthreads();

  // The luma lines above were staged in s->intra.above/left; the intra
  // predictor overwrites those arrays, so save the alpha/luma state in the
  // dedicated fields first (done above) and never reread the staging area.
  const int total = job.w * job.h;
  av2enc::cuda::intra_modes::IntraBlock block_u{};
  av2enc::cuda::intra_modes::IntraBlock block_v{};

  for (int mode = 0; mode < av2enc::cuda::intra_modes::kIntraModes; ++mode) {
    const int delta = intra_derived_delta(job, mode);

    fill_intra_block(job, job.chroma_u, mode, delta, &block_u);
    av2enc::cuda::intra_modes::intra_run_block_device(
        block_u, s->pred, nullptr, &s->intra);
    __syncthreads();

    long long sum_u = 0;
    for (int index = tid; index < total; index += nthr) {
      const int i = index / job.w;
      const int j = index - i * job.w;
      const int src = intra_source_sample(job, false, job.x + j, job.y + i);
      const int p = static_cast<int>(s->pred[index]);
      const int d = src - p;
      sum_u += static_cast<long long>(d) * d;
    }
    sum_u = block_reduce_sum(sum_u, s->reduce, tid, nthr);

    fill_intra_block(job, job.chroma_v, mode, delta, &block_v);
    av2enc::cuda::intra_modes::intra_run_block_device(
        block_v, s->pred, nullptr, &s->intra);
    __syncthreads();

    long long sum_v = 0;
    for (int index = tid; index < total; index += nthr) {
      const int i = index / job.w;
      const int j = index - i * job.w;
      const int src = intra_source_sample(job, true, job.x + j, job.y + i);
      const int p = static_cast<int>(s->pred[index]);
      const int d = src - p;
      sum_v += static_cast<long long>(d) * d;
    }
    sum_v = block_reduce_sum(sum_v, s->reduce, tid, nthr);

    if (tid == 0) {
      const long long candidate = sum_u + sum_v;
      if (mode == av2enc::cuda::intra_modes::kDcPred) {
        s->sse_dc_u = sum_u;
        s->sse_dc_v = sum_v;
      }
      if (candidate < s->best_sse) {
        s->best_sse = candidate;
        s->best_mode = mode;
        s->best_delta = delta;
        s->best_is_cfl = 0;
      }
    }
    __syncthreads();
  }

  if (opt.enable_cfl != 0) {
    long long sum_u = 0;
    long long sum_v = 0;
    for (int index = tid; index < total; index += nthr) {
      const int i = index / job.w;
      const int j = index - i * job.w;
      const int src_u = intra_source_sample(job, false, job.x + j, job.y + i);
      const int src_v = intra_source_sample(job, true, job.x + j, job.y + i);
      const int luma_value = sample_luma(job, filter, i, j);
      const int pred_u =
          cfl_sample(s->dc_u, s->alpha_u, luma_value, s->luma_avg);
      const int pred_v =
          cfl_sample(s->dc_v, s->alpha_v, luma_value, s->luma_avg);
      const int du = src_u - pred_u;
      const int dv = src_v - pred_v;
      sum_u += static_cast<long long>(du) * du;
      sum_v += static_cast<long long>(dv) * dv;
    }
    sum_u = block_reduce_sum(sum_u, s->reduce, tid, nthr);
    sum_v = block_reduce_sum(sum_v, s->reduce, tid, nthr);

    if (tid == 0) {
      s->sse_cfl_u = sum_u;
      s->sse_cfl_v = sum_v;
      const long long candidate = sum_u + sum_v;
      if (candidate < s->best_sse) {
        s->best_sse = candidate;
        s->best_mode = av2enc::cuda::intra_modes::kDcPred;
        s->best_delta = 0;
        s->best_is_cfl = 1;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    M1ChromaResult r{};
    r.is_cfl = s->best_is_cfl != 0 ? 1 : 0;
    r.winner = r.is_cfl
                   ? kM1ChromaCfl
                   : (s->best_mode == av2enc::cuda::intra_modes::kDcPred
                          ? kM1ChromaDc
                          : kM1ChromaUvDirectional);
    r.uv_mode = s->best_mode;
    r.angle_delta = s->best_delta;
    r.alpha_u = s->alpha_u;
    r.alpha_v = s->alpha_v;
    r.dc_u = s->dc_u;
    r.dc_v = s->dc_v;
    r.sse_dc_u = s->sse_dc_u;
    r.sse_dc_v = s->sse_dc_v;
    r.sse_cfl_u = s->sse_cfl_u;
    r.sse_cfl_v = s->sse_cfl_v;
    r.sse = s->best_sse;
    out[0] = r;
  }
  __syncthreads();

  // Write the winning prediction. CfL is recomputed from the shared alpha; a
  // directional/smooth winner is predicted once more into the tight scratch
  // and copied to the caller's plane (which may have a wider stride).
  const int pred_stride = job.pred_stride > 0 ? job.pred_stride : job.w;
  if (out[0].is_cfl != 0) {
    for (int index = tid; index < total; index += nthr) {
      const int i = index / job.w;
      const int j = index - i * job.w;
      const int luma_value = sample_luma(job, filter, i, j);
      const int offset = i * pred_stride + j;
      if (job.pred_u != nullptr) {
        job.pred_u[offset] = static_cast<std::uint8_t>(
            cfl_sample(s->dc_u, s->alpha_u, luma_value, s->luma_avg));
      }
      if (job.pred_v != nullptr) {
        job.pred_v[offset] = static_cast<std::uint8_t>(
            cfl_sample(s->dc_v, s->alpha_v, luma_value, s->luma_avg));
      }
    }
    return;
  }

  const int mode = out[0].uv_mode;
  const int delta = out[0].angle_delta;
  fill_intra_block(job, job.chroma_u, mode, delta, &block_u);
  av2enc::cuda::intra_modes::intra_run_block_device(
      block_u, s->pred, nullptr, &s->intra);
  __syncthreads();
  for (int index = tid; index < total; index += nthr) {
    const int i = index / job.w;
    const int j = index - i * job.w;
    if (job.pred_u != nullptr) {
      job.pred_u[i * pred_stride + j] = s->pred[index];
    }
  }
  fill_intra_block(job, job.chroma_v, mode, delta, &block_v);
  av2enc::cuda::intra_modes::intra_run_block_device(
      block_v, s->pred, nullptr, &s->intra);
  __syncthreads();
  for (int index = tid; index < total; index += nthr) {
    const int i = index / job.w;
    const int j = index - i * job.w;
    if (job.pred_v != nullptr) {
      job.pred_v[i * pred_stride + j] = s->pred[index];
    }
  }
}

}  // namespace
}  // namespace av2enc::cuda::m1

#endif  // AV2ENC_CUDA_M1_CHROMA_DEVICE_CUH_
