// AV2 v1.0.0 full intra prediction kernels (M1).
//
// Integer only, mirroring the spec arithmetic exactly (Round2 / Clip1 /
// resolve_divisor). See intra_modes.h for the normative section map and the
// API contract.
//
// One CUDA block predicts one transform block. The above-row / left-column
// reference arrays live in shared memory with two slots of headroom so that
// the spec's AboveRow[-1], AboveRow[-2] and the directional extension indices
// are addressable without branches.

#include "intra_modes.h"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::intra_modes {
namespace {

constexpr int kEdgeHead = 2;                      // slots for indices -2 and -1
constexpr int kEdgeSlots = 2 * kMaxBlockDim + 8;  // covers -2 .. w+h+2
constexpr int kEdgeCopySlots = kEdgeMaxPx + 1;    // filter input copy
constexpr int kBatchThreads = 128;
constexpr int kSingleMaxThreads = 256;

// ---------------------------------------------------------------------------
// Device copies of the normative tables (§9).  Read-only device memory, same
// pattern as transform.cu's DCT kernels.  Declared in device code because
// namespace-scope constexpr arrays are not visible to nvcc without a
// per-thread local copy (which costs 512 B of stack per thread).
// ---------------------------------------------------------------------------

__device__ const int kModeToAngleDev[kModeToAngleCount] = {
    AV2ENC_INTRA_MODE_TO_ANGLE_TABLE};

__device__ const int kDrIntraDerivativeDev[kDrDerivativeCount] = {
    AV2ENC_INTRA_DR_DERIVATIVE_TABLE};

__device__ const int kDrInterpFilterDev[32][4] = {
    AV2ENC_INTRA_DR_INTERP_TABLE};

__device__ const int kIntraEdgeKernelDev[kEdgeKernels][kEdgeTaps] = {
    AV2ENC_INTRA_EDGE_KERNEL_TABLE};

__device__ const int kDivLutDev[kDivLutNum] = {AV2ENC_INTRA_DIV_LUT_TABLE};

// Scratch shared by all threads of one prediction block.
struct SharedScratch {
  int above[kEdgeSlots];
  int left[kEdgeSlots];
  int copy_above[kEdgeCopySlots];
  int copy_left[kEdgeCopySlots];
  int pangle;
  int dc;
  int strength_above;
  int strength_left;
  int numpx_above;
  int numpx_left;
  int apply_corner;
};

// ---------------------------------------------------------------------------
// Scalar arithmetic
// ---------------------------------------------------------------------------

__device__ __forceinline__ int im_abs(int v) { return v < 0 ? -v : v; }

__device__ __forceinline__ int im_clip3(int low, int high, int v) {
  return v < low ? low : (v > high ? high : v);
}

__device__ __forceinline__ int im_clip1(int v) { return im_clip3(0, 255, v); }

// Round2(value, n) = (value + 2^(n-1)) >> n. Negative values use arithmetic
// shift, matching the spec's two's-complement definition and the CPU reference.
__device__ __forceinline__ int im_round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

__device__ __forceinline__ int im_floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// §7.13.3.22 resolve_divisor.
__device__ __forceinline__ void im_resolve_divisor(int d, int* shift,
                                                   int* factor) {
  const int abs_d = d < 0 ? -d : d;
  const int n = im_floor_log2(abs_d);
  const int e = abs_d - (1 << n);
  const int f = (n > kDivLutBits) ? im_round2(e, n - kDivLutBits)
                                  : (e << (kDivLutBits - n));
  *shift = n + kDivLutPrecBits;
  *factor = (d < 0) ? -kDivLutDev[f] : kDivLutDev[f];
}

// §7.13.2.11 approx_divide.
__device__ __forceinline__ int im_approx_divide(int num, int den) {
  int shift = 0;
  int factor = 0;
  im_resolve_divisor(den, &shift, &factor);
  return im_round2(num * factor, shift);
}

// ---------------------------------------------------------------------------
// Reference sample construction (§7.13.2.1)
// ---------------------------------------------------------------------------

__device__ __forceinline__ int im_above_sample(const IntraBlock& b, int i) {
  const std::size_t stride = static_cast<std::size_t>(b.recon_stride);
  if (b.have_above == 0) {
    if (b.have_left != 0) {
      const std::size_t x = static_cast<std::size_t>(b.x0 - 1);
      return static_cast<int>(b.recon_plane[static_cast<std::size_t>(b.y0) *
                                                stride +
                                            x]);
    }
    return 127;  // (1 << (BitDepth - 1)) - 1
  }
  int limit = b.max_x;
  const int num4 =
      (b.plane > 0 && b.width > 32) ? 0 : b.num4_above_right;
  const int ext = b.x0 + b.width + 4 * num4 - 1;
  if (ext < limit) {
    limit = ext;
  }
  int x = b.x0 + i;
  if (x > limit) {
    x = limit;
  }
  return static_cast<int>(
      b.recon_plane[static_cast<std::size_t>(b.y0 - 1) * stride +
                    static_cast<std::size_t>(x)]);
}

__device__ __forceinline__ int im_left_sample(const IntraBlock& b, int i) {
  const std::size_t stride = static_cast<std::size_t>(b.recon_stride);
  if (b.have_left == 0) {
    if (b.have_above != 0) {
      const std::size_t x = static_cast<std::size_t>(b.x0);
      return static_cast<int>(
          b.recon_plane[static_cast<std::size_t>(b.y0 - 1) * stride + x]);
    }
    return 129;  // (1 << (BitDepth - 1)) + 1
  }
  int limit = b.max_y;
  const int num4 =
      (b.plane > 0 && b.height > 32) ? 0 : b.num4_below_left;
  const int ext = b.y0 + b.height + 4 * num4 - 1;
  if (ext < limit) {
    limit = ext;
  }
  int y = b.y0 + i;
  if (y > limit) {
    y = limit;
  }
  return static_cast<int>(
      b.recon_plane[static_cast<std::size_t>(y) * stride +
                    static_cast<std::size_t>(b.x0 - 1)]);
}

__device__ __forceinline__ void im_build_edges(const IntraBlock& b, int* above,
                                               int* left) {
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  const int need = b.width + b.height;
  for (int i = tid; i < need; i += nthr) {
    above[i] = im_above_sample(b, i);
    left[i] = im_left_sample(b, i);
  }
}

// AboveRow[-1] / LeftCol[-1] and the corner filter.
__device__ __forceinline__ int im_corner_value(const IntraBlock& b,
                                               const int* above,
                                               const int* left) {
  const std::size_t stride = static_cast<std::size_t>(b.recon_stride);
  if (b.have_above != 0 && b.have_left != 0) {
    return static_cast<int>(b.recon_plane[static_cast<std::size_t>(b.y0 - 1) *
                                              stride +
                                          static_cast<std::size_t>(b.x0 - 1)]);
  }
  if (b.have_above != 0) {
    return above[0];
  }
  if (b.have_left != 0) {
    return left[0];
  }
  return 128;  // 1 << (BitDepth - 1)
}

// ---------------------------------------------------------------------------
// Edge filter (§7.13.2.14-18)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void im_compute_filter_params(const IntraBlock& b,
                                                         SharedScratch* s) {
  const int mode = static_cast<int>(b.mode);
  const int base_angle = kModeToAngleDev[mode];
  const int pa = is_directional_mode(mode)
                     ? wide_angle_map(base_angle + b.angle_delta * kAngleStep,
                                      b.width, b.height)
                     : base_angle;
  s->pangle = pa;
  s->dc = 128;
  s->strength_above = 0;
  s->strength_left = 0;
  s->numpx_above = 0;
  s->numpx_left = 0;
  s->apply_corner = 0;
  if (!is_directional_mode(static_cast<int>(b.mode)) ||
      b.enable_edge_filter == 0) {
    return;
  }
  if (pa == 90 || pa == 180) {
    return;
  }
  // IBP is disabled, so both directions share the OR-ed filter type. A
  // neighbour flag only counts when that neighbour exists (§7.13.2.15/16).
  const int filter_type =
      ((b.have_above != 0 && b.filter_type_above != 0) ||
       (b.have_left != 0 && b.filter_type_left != 0))
          ? 1
          : 0;
  const bool need_right = pa < 90;
  const bool need_bottom = pa > 180;
  if (pa > 90 && pa < 180 && (b.width + b.height) >= 24) {
    s->apply_corner = 1;
  }
  if (b.have_above != 0) {
    s->strength_above = intra_edge_strength(b.width, b.height, filter_type,
                                            pa - 90);
    const int vis_w = im_clip3(1, b.width, b.max_x - b.x0 + 1);
    s->numpx_above = vis_w + (need_right ? b.height : 0) + 1;
  }
  if (b.have_left != 0) {
    s->strength_left = intra_edge_strength(b.width, b.height, filter_type,
                                           pa - 180);
    const int vis_h = im_clip3(1, b.height, b.max_y - b.y0 + 1);
    s->numpx_left = vis_h + (need_bottom ? b.width : 0) + 1;
  }
}

__device__ __forceinline__ void im_edge_filter_pass(int* edge, int sz,
                                                    int strength, int* copy) {
  if (strength <= 0 || sz <= 1) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  for (int i = 1 + tid; i < sz; i += nthr) {
    int acc = 0;
#pragma unroll
    for (int j = 0; j < kEdgeTaps; ++j) {
      const int k = im_clip3(0, sz - 1, i - 2 + j);
      acc += kIntraEdgeKernelDev[strength - 1][j] * copy[k];
    }
    edge[i - 1] = (acc + 8) >> 4;
  }
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

__device__ __forceinline__ int im_edge_interpolate(const int* edge, int base,
                                                   int shift, int enable_idif) {
  if (enable_idif != 0) {
    int s = 0;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      s += kDrInterpFilterDev[shift][t] * edge[base + t - 1];
    }
    return im_clip1(im_round2(s, 7));
  }
  return im_round2(edge[base] * (32 - shift) + edge[base + 1] * shift, 5);
}

// §7.13.2.8 single directional prediction, MrlIndex = 0.
__device__ __forceinline__ int im_directional_sample(int pAngle, int px,
                                                     int py, int w, int h,
                                                     const int* above,
                                                     const int* left,
                                                     int enable_idif) {
  if (pAngle < 90) {
    const int dx = kDrIntraDerivativeDev[pAngle];
    const int idx = (py + 1) * dx;
    const int base = (idx >> 6) + px;
    const int shift = (idx >> 1) & 0x1F;
    const int max_base_x = w + h - 1;
    if (base < max_base_x + enable_idif) {
      return im_edge_interpolate(above, base, shift, enable_idif);
    }
    return above[max_base_x];
  }
  if (pAngle > 90 && pAngle < 180) {
    const int dx = kDrIntraDerivativeDev[180 - pAngle];
    const int dy = kDrIntraDerivativeDev[pAngle - 90];
    const int idx = (px << 6) - (py + 1) * dx;
    const int base = idx >> 6;
    if (base >= -1) {
      return im_edge_interpolate(above, base, (idx >> 1) & 0x1F, enable_idif);
    }
    const int idx2 = (py << 6) - (px + 1) * dy;
    return im_edge_interpolate(left, idx2 >> 6, (idx2 >> 1) & 0x1F,
                               enable_idif);
  }
  if (pAngle > 180) {
    const int dy = kDrIntraDerivativeDev[270 - pAngle];
    const int idx = (px + 1) * dy;
    const int base = (idx >> 6) + py;
    const int shift = (idx >> 1) & 0x1F;
    const int max_base_y = w + h - 1;
    if (base < max_base_y + enable_idif) {
      return im_edge_interpolate(left, base, shift, enable_idif);
    }
    return left[max_base_y];
  }
  if (pAngle == 90) {
    return above[px];
  }
  return left[py];  // pAngle == 180
}

__device__ __forceinline__ int im_predict_sample(const IntraBlock& b, int px,
                                                 int py, int log2w, int log2h,
                                                 int dc, int pAngle,
                                                 const int* above,
                                                 const int* left) {
  const int mode = static_cast<int>(b.mode);
  const int enable_idif = (b.plane == 0) ? 1 : 0;

  if (mode == kDcPred) {
    return dc;
  }
  if (mode == kPaethPred) {
    const int base = above[px] + left[py] - above[-1];
    const int p_left = im_abs(base - left[py]);
    const int p_top = im_abs(base - above[px]);
    const int p_top_left = im_abs(base - above[-1]);
    if (p_left <= p_top && p_left <= p_top_left) {
      return left[py];
    }
    if (p_top <= p_top_left) {
      return above[px];
    }
    return above[-1];
  }
  if (mode == kSmoothPred || mode == kSmoothVPred || mode == kSmoothHPred) {
    const int bl = left[b.height];
    const int tr = above[b.width];
    const int scale = im_round2(log2w + log2h - 4, 2);
    int s_top = (py << 1) >> scale;
    int s_left = (px << 1) >> scale;
    if (s_top > 6) s_top = 6;
    if (s_left > 6) s_left = 6;
    const int top = above[px];
    const int lft = left[py];
    const int pred_h = tr + im_round2((lft - tr) * (b.width - 1 - px), log2w);
    const int pred_v = bl + im_round2((top - bl) * (b.height - 1 - py), log2h);
    const int pred_h2 =
        pred_h + im_round2((lft - pred_h) * (kBlendWeightMax >> s_left), 6);
    const int pred_v2 =
        pred_v + im_round2((top - pred_v) * (kBlendWeightMax >> s_top), 6);
    if (mode == kSmoothHPred) {
      return pred_h2;
    }
    if (mode == kSmoothVPred) {
      return pred_v2;
    }
    return im_round2(pred_v2 + pred_h2, 1);
  }
  return im_directional_sample(pAngle, px, py, b.width, b.height, above, left,
                               enable_idif);
}

__device__ __forceinline__ int im_dc_value(const IntraBlock& b,
                                           const int* above, const int* left) {
  if (b.have_left != 0 && b.have_above != 0) {
    int sum = 0;
    for (int k = 0; k < b.height; ++k) {
      sum += left[k];
    }
    for (int k = 0; k < b.width; ++k) {
      sum += above[k];
    }
    return im_clip1(im_approx_divide(sum, b.width + b.height));
  }
  if (b.have_left != 0) {
    int sum = 0;
    for (int k = 0; k < b.height; ++k) {
      sum += left[k];
    }
    return im_round2(sum, im_floor_log2(b.height));
  }
  if (b.have_above != 0) {
    int sum = 0;
    for (int k = 0; k < b.width; ++k) {
      sum += above[k];
    }
    return im_round2(sum, im_floor_log2(b.width));
  }
  return 128;
}

// ---------------------------------------------------------------------------
// Whole-block driver
// ---------------------------------------------------------------------------

__device__ __forceinline__ void im_run_block(const IntraBlock& b,
                                             std::uint8_t* __restrict__ pred,
                                             std::int16_t* __restrict__ residual,
                                             SharedScratch* s) {
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  int* above = s->above + kEdgeHead;
  int* left = s->left + kEdgeHead;

  im_build_edges(b, above, left);
  __syncthreads();

  if (tid == 0) {
    above[-1] = im_corner_value(b, above, left);
    left[-1] = above[-1];
    im_compute_filter_params(b, s);
    if (s->apply_corner != 0) {
      const int corner =
          im_round2(left[0] * 5 + above[-1] * 6 + above[0] * 5, 4);
      above[-1] = corner;
      left[-1] = corner;
    }
  }
  __syncthreads();

  // Copy the original edge samples for the in-place 5-tap filter.
  if (s->strength_above > 0) {
    for (int i = tid; i < s->numpx_above; i += nthr) {
      s->copy_above[i] = above[i - 1];
    }
  }
  if (s->strength_left > 0) {
    for (int i = tid; i < s->numpx_left; i += nthr) {
      s->copy_left[i] = left[i - 1];
    }
  }
  __syncthreads();

  im_edge_filter_pass(above, s->numpx_above, s->strength_above, s->copy_above);
  im_edge_filter_pass(left, s->numpx_left, s->strength_left, s->copy_left);
  __syncthreads();

  // §7.13.2.8 edge extension (MrlIndex = 0) and the DC average.
  if (tid == 0) {
    const bool directional = is_directional_mode(static_cast<int>(b.mode));
    if (directional) {
      const int p_angle = s->pangle;
      const int max_base = b.width + b.height - 1;
      if (p_angle > 90 && p_angle < 180) {
        left[b.height] = left[b.height - 1];
        above[b.width] = above[b.width - 1];
        left[b.height + 1] = left[b.height - 1];
        above[b.width + 1] = above[b.width - 1];
      } else {
        left[max_base + 1] = left[max_base];
        above[max_base + 1] = above[max_base];
        left[max_base + 2] = left[max_base];
        above[max_base + 2] = above[max_base];
      }
      left[-2] = left[-1];
      above[-2] = above[-1];
    }
    if (b.mode == kDcPred) {
      s->dc = im_dc_value(b, above, left);
    }
  }
  __syncthreads();

  const int log2w = im_floor_log2(b.width);
  const int log2h = im_floor_log2(b.height);
  const int dc = s->dc;
  const int p_angle = s->pangle;
  const std::size_t source_stride = static_cast<std::size_t>(b.source_stride);
  const int n = b.width * b.height;
  for (int i = tid; i < n; i += nthr) {
    const int px = i % b.width;
    const int py = i / b.width;
    const int value = im_predict_sample(b, px, py, log2w, log2h, dc, p_angle,
                                        above, left);
    pred[i] = static_cast<std::uint8_t>(value);
    if (residual != nullptr) {
      const int sx = im_clip3(0, b.source_max_x, b.x0 + px);
      const int sy = im_clip3(0, b.source_max_y, b.y0 + py);
      const int sample = static_cast<int>(
          b.source_plane[static_cast<std::size_t>(sy) * source_stride +
                         static_cast<std::size_t>(sx)]);
      residual[i] = static_cast<std::int16_t>(sample - value);
    }
  }
}

__global__ void im_predict_kernel(IntraBlock b,
                                  std::uint8_t* __restrict__ pred,
                                  std::int16_t* __restrict__ residual) {
  __shared__ SharedScratch s;
  im_run_block(b, pred, residual, &s);
}

__global__ void im_predict_batch_kernel(const IntraJob* __restrict__ jobs) {
  __shared__ SharedScratch s;
  const IntraJob job = jobs[blockIdx.x];
  im_run_block(job.block, job.pred, job.residual, &s);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

int im_threads(int work_items) {
  int threads = ((work_items + 31) / 32) * 32;
  if (threads < 32) {
    threads = 32;
  }
  if (threads > kSingleMaxThreads) {
    threads = kSingleMaxThreads;
  }
  return threads;
}

bool im_valid_dim(int v) {
  return v == 4 || v == 8 || v == 16 || v == 32 || v == 64;
}

bool im_valid_block(const IntraBlock& b, bool require_source) {
  if (!im_valid_dim(b.width) || !im_valid_dim(b.height)) {
    return false;
  }
  if (b.x0 < 0 || b.y0 < 0 || b.max_x < b.x0 || b.max_y < b.y0) {
    return false;
  }
  if (b.angle_delta < -kMaxAngleDelta || b.angle_delta > kMaxAngleDelta) {
    return false;
  }
  if (!is_directional_mode(static_cast<int>(b.mode)) && b.angle_delta != 0) {
    return false;
  }
  if (static_cast<int>(b.mode) < 0 || static_cast<int>(b.mode) >= kIntraModes) {
    return false;
  }
  if ((b.filter_type_above & ~1) != 0 || (b.filter_type_left & ~1) != 0) {
    return false;
  }
  if (b.num4_above_right < 0 || b.num4_below_left < 0) {
    return false;
  }
  if (b.plane < 0) {
    return false;
  }
  if (b.have_left != 0 || b.have_above != 0) {
    if (b.recon_plane == nullptr || b.recon_stride <= 0) {
      return false;
    }
  }
  if (b.have_left != 0 && b.x0 < 1) {
    return false;
  }
  if (b.have_above != 0 && b.y0 < 1) {
    return false;
  }
  if (require_source) {
    if (b.source_plane == nullptr || b.source_stride <= 0 ||
        b.source_max_x < b.x0 || b.source_max_y < b.y0) {
      return false;
    }
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

cudaError_t intra_predict(const IntraBlock& block, std::uint8_t* pred,
                          cudaStream_t stream) {
  if (!im_valid_block(block, false) || pred == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int work = block.width * block.height;
  im_predict_kernel<<<1, im_threads(work), 0, stream>>>(block, pred, nullptr);
  return cudaGetLastError();
}

cudaError_t intra_predict_residual(const IntraBlock& block, std::uint8_t* pred,
                                   std::int16_t* residual,
                                   cudaStream_t stream) {
  if (!im_valid_block(block, true) || pred == nullptr || residual == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int work = block.width * block.height;
  im_predict_kernel<<<1, im_threads(work), 0, stream>>>(block, pred, residual);
  return cudaGetLastError();
}

cudaError_t intra_predict_batch(const IntraJob* d_jobs, int job_count,
                                cudaStream_t stream) {
  if (job_count < 0 || (job_count > 0 && d_jobs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (job_count == 0) {
    return cudaSuccess;
  }
  im_predict_batch_kernel<<<job_count, kBatchThreads, 0, stream>>>(d_jobs);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::intra_modes
