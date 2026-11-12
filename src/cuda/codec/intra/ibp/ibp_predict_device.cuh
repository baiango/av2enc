// Device-side AV2 intra bi-prediction (IBP).
//
// This header is the exact integer body used by ibp.cu and by the host
// emulation parity test.  It follows AV2 v1.0.0 §7.13.2.7/.8/.9/.12/.14/.17/
// .18.  Every thread of one CUDA block cooperates on one prediction block.
//
// The edge arrays are stored with two slots of headroom (logical -2, -1), so
// the spec's AboveRow[-1], AboveRow[-2] and directional extension indices are
// directly addressable.

#ifndef AV2ENC_CUDA_IBP_IBP_PREDICT_DEVICE_CUH_
#define AV2ENC_CUDA_IBP_IBP_PREDICT_DEVICE_CUH_

#include "ibp.h"

// Reuse the intra component's bit-exact §7.13.2.1 edge construction and
// §7.13.2.8 single-directional predictor instead of maintaining a second
// transcription.  This is a read-only include; see docs/negative-results.md
// on duplicate intra implementations.
#include "../modes/intra_predict_device.cuh"

#include "../../../../codec/intra/ibp/ibp_tables.h"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::ibp {
namespace {

constexpr int kEdgeHead = 2;                // slots for -2 and -1
constexpr int kEdgeSlots = 2 * 64 + 8;      // covers -2 .. w+h+2
constexpr int kEdgeCopySlots = 2 * 64 + 1;  // sz <= 129
constexpr int kWeightSize = 16;
constexpr int kWeightMax = 128;
constexpr int kWeightShift = 7;

// ---------------------------------------------------------------------------
// Normative tables (single source: src/encoder/ibp/ibp_tables.h)
// ---------------------------------------------------------------------------

static __device__ const int kDrDev[90] = {
    AV2ENC_IBP_DR_INTRA_DERIVATIVE_TABLE};
static __device__ const int kDivLutDev[129] = {AV2ENC_IBP_DIV_LUT_TABLE};
static __device__ const int kIbpWeightsDev[5][kWeightSize] = {
    AV2ENC_IBP_WEIGHTS_TABLE};

struct SharedScratch {
  int above[kEdgeHead + kEdgeSlots];
  int left[kEdgeHead + kEdgeSlots];
  int weights[kWeightSize][kWeightSize];
  int dc;
};

// ---------------------------------------------------------------------------
// Scalar arithmetic
// ---------------------------------------------------------------------------

__device__ __forceinline__ int ibp_abs(int v) { return v < 0 ? -v : v; }

__device__ __forceinline__ int ibp_clip3(int low, int high, int v) {
  return v < low ? low : (v > high ? high : v);
}

__device__ __forceinline__ int ibp_clip1(int v) { return ibp_clip3(0, 255, v); }

__device__ __forceinline__ int ibp_round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

__device__ __forceinline__ int ibp_floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

__device__ __forceinline__ void ibp_resolve_divisor(int divisor, int* shift,
                                                    int* factor) {
  const int magnitude = ibp_abs(divisor);
  const int n = ibp_floor_log2(magnitude);
  const int e = magnitude - (1 << n);
  const int f = n > intra_modes::kDivLutBits
                    ? ibp_round2(e, n - intra_modes::kDivLutBits)
                    : (e << (intra_modes::kDivLutBits - n));
  *shift = n + intra_modes::kDivLutPrecBits;
  *factor = divisor < 0 ? -kDivLutDev[f] : kDivLutDev[f];
}

__device__ __forceinline__ int ibp_approx_divide(int num, int den) {
  int shift = 0;
  int factor = 0;
  ibp_resolve_divisor(den, &shift, &factor);
  return ibp_round2(num * factor, shift);
}

// ---------------------------------------------------------------------------
// §7.13.2.14 / §7.13.2.18
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ibp_filter_corner(int* above, int* left) {
  const int sum = left[0] * 5 + above[-1] * 6 + above[0] * 5;
  const int value = ibp_round2(sum, 4);
  left[-1] = value;
  above[-1] = value;
}

// `row` points at logical index 0.  The local array is the spec's edge[]
// snapshot: edge[i] = row[i-1] is read before row[i-1] is written.
__device__ __forceinline__ void ibp_edge_filter(int* row, int size,
                                                int strength) {
  if (size <= 1 || strength < 1 || strength > 3) {
    return;
  }
  int edge[kEdgeCopySlots];
  for (int i = 0; i < size; ++i) {
    edge[i] = row[i - 1];
  }
  for (int i = 1; i < size; ++i) {
    int sum = 0;
    for (int j = 0; j < 5; ++j) {
      const int k = ibp_clip3(0, size - 1, i - 2 + j);
      sum += intra_modes::intra_edge_kernel(strength - 1, j) * edge[k];
    }
    row[i - 1] = (sum + 8) >> 4;
  }
}

// ---------------------------------------------------------------------------
// Full block body
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ibp_run_block_device(const IbpBlock& job,
                                                     std::uint8_t* pred,
                                                     SharedScratch* s) {
  const IntraBlock& b = job.intra;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  const int w = b.width;
  const int h = b.height;

  int* above = s->above + kEdgeHead;
  int* left = s->left + kEdgeHead;

  intra_modes::im_build_edges(b, above, left);
  __syncthreads();

  // The device body owns DC + directional only.  Host ibp_predict()
  // delegates smooth/Paeth to intra_predict(); batch callers must not submit
  // those modes, so leave the output untouched instead of mispredicting.
  const int mode_int = static_cast<int>(b.mode);
  if (b.mode != intra_modes::kDcPred &&
      !intra_modes::is_directional_mode(mode_int)) {
    return;
  }
  const IbpDecision d = ibp_decide(job);
  if (d.valid == 0) {
    return;
  }

  // §7.13.2.1 always defines the -1 corner, with or without edge filtering.
  if (tid == 0) {
    const int corner = intra_modes::im_corner_value(b, above, left);
    above[-1] = corner;
    left[-1] = corner;
  }
  __syncthreads();

  if (b.mode == intra_modes::kDcPred) {
    if (tid == 0) {
      s->dc = intra_modes::im_dc_value(b, above, left);
    }
    __syncthreads();
    for (int p = tid; p < w * h; p += nthr) {
      pred[p] = static_cast<std::uint8_t>(s->dc);
    }

    const bool not4x4 = !(w == 4 && h == 4);
    const bool dc_ibp = (job.enable_ibp != 0) && (job.use_dip == 0) &&
                        not4x4 && (b.plane == 0 || job.uv_is_cfl == 0);
    if (!dc_ibp) {
      return;
    }
    __syncthreads();

    const int lw = ibp_floor_log2(w);
    const int lh = ibp_floor_log2(h);
    if (b.have_above != 0) {
      const int c0 = (w < h && b.have_left != 0) ? (w >> 2) : 0;
      const int cols = w - c0;
      const int rows = h >> 2;
      for (int n = tid; n < rows * cols; n += nthr) {
        const int r = n / cols;
        const int c = c0 + (n - r * cols);
        const int weight = kIbpWeightsDev[lh - 2][r];
        const int old = pred[r * w + c];
        pred[r * w + c] = static_cast<std::uint8_t>(
            ibp_round2(above[c] * (kWeightMax - weight) + old * weight,
                       kWeightShift));
      }
    }
    __syncthreads();
    if (b.have_left != 0) {
      const int r0 = (w >= h && b.have_above != 0) ? (h >> 2) : 0;
      const int rows = h - r0;
      const int cols = w >> 2;
      for (int n = tid; n < rows * cols; n += nthr) {
        const int r = r0 + n / cols;
        const int c = n - (n / cols) * cols;
        const int weight = kIbpWeightsDev[lw - 2][c];
        const int old = pred[r * w + c];
        pred[r * w + c] = static_cast<std::uint8_t>(
            ibp_round2(left[r] * (kWeightMax - weight) + old * weight,
                       kWeightShift));
      }
    }
    return;
  }

  // Directional path.
  const int p_angle = d.p_angle;
  const bool filter_enabled =
      (b.enable_edge_filter != 0) && p_angle != 90 && p_angle != 180;

  if (filter_enabled) {
    if (tid == 0 && (d.apply_ibp_edge_rules != 0 ||
                     (p_angle > 90 && p_angle < 180)) &&
        (w + h) >= 24) {
      ibp_filter_corner(above, left);
    }
    __syncthreads();

    int fta = (b.have_above != 0 && b.filter_type_above != 0) ? 1 : 0;
    int ftl = (b.have_left != 0 && b.filter_type_left != 0) ? 1 : 0;
    int angle_above = p_angle - 90;
    int angle_left = p_angle - 180;
    int need_right = p_angle < 90 ? 1 : 0;
    int need_bottom = p_angle > 180 ? 1 : 0;
    if (d.apply_ibp_edge_rules != 0) {
      if (p_angle > 180) {
        need_right = 1;
      }
      if (p_angle < 90) {
        need_bottom = 1;
      }
      if (angle_above > 90) {
        angle_above -= 180;
      }
      if (angle_left < -90) {
        angle_left += 180;
      }
    } else {
      const int ft = fta | ftl;
      fta = ft;
      ftl = ft;
    }

    if (tid == 0 && b.have_above != 0) {
      const int strength =
          intra_modes::intra_edge_strength(w, h, fta, angle_above);
      const int visible =
          b.max_x - b.x0 + 1 < w ? b.max_x - b.x0 + 1 : w;
      const int num_px = visible + (need_right != 0 ? h : 0) + 1;
      ibp_edge_filter(above, num_px, strength);
    }
    if (((nthr == 1 && tid == 0) || (nthr > 1 && tid == 1)) &&
        b.have_left != 0) {
      const int strength =
          intra_modes::intra_edge_strength(w, h, ftl, angle_left);
      const int visible =
          b.max_y - b.y0 + 1 < h ? b.max_y - b.y0 + 1 : h;
      const int num_px = visible + (need_bottom != 0 ? w : 0) + 1;
      ibp_edge_filter(left, num_px, strength);
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int max_base = w + h - 1;
    if (p_angle > 90 && p_angle < 180) {
      left[h] = left[h - 1];
      above[w] = above[w - 1];
      left[h + 1] = left[h - 1];
      above[w + 1] = above[w - 1];
    } else {
      left[max_base + 1] = left[max_base];
      above[max_base + 1] = above[max_base];
      left[max_base + 2] = left[max_base];
      above[max_base + 2] = above[max_base];
    }
    left[-2] = left[-1];
    above[-2] = above[-1];
  }

  if (d.use_ibp != 0) {
    const int angle = p_angle < 90 ? p_angle : 270 - p_angle;
    const int clamped = angle < 39 ? 39 : (angle > 89 ? 89 : angle);
    const int dy = kDrDev[90 - clamped];
    for (int r = tid; r < kWeightSize; r += nthr) {
      int y = dy;
      for (int c = 0; c < kWeightSize; ++c) {
        const int dist = ((r + 1) << 6) + y;
        int shift = 0;
        int factor = 0;
        ibp_resolve_divisor(dist, &shift, &factor);
        s->weights[r][c] = ibp_round2(y * factor, shift - 7);
        y += dy;
      }
    }
  }
  __syncthreads();

  const bool idif = b.plane == 0;
  for (int p = tid; p < w * h; p += nthr) {
    const int i = p / w;
    const int j = p - i * w;
    int value = intra_modes::im_directional_sample(
        p_angle, j, i, w, h, above, left, idif ? 1 : 0);
    if (d.use_ibp != 0) {
      const int second = intra_modes::im_directional_sample(
          d.second_angle, j, i, w, h, above, left, idif ? 1 : 0);
      const int cs = w >> 5;
      const int rs = h >> 5;
      const int weight = p_angle < 90 ? s->weights[i >> rs][j >> cs]
                                      : s->weights[j >> cs][i >> rs];
      value = ibp_round2(value * weight + second * (kWeightMax - weight),
                         kWeightShift);
    }
    pred[p] = static_cast<std::uint8_t>(value);
  }
}

}  // namespace
}  // namespace av2enc::cuda::ibp

#endif  // AV2ENC_CUDA_IBP_IBP_PREDICT_DEVICE_CUH_
