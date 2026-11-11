// Device-side AV2 DIP prediction (§7.13.2.1, .3-.6).
//
// Shared by the CUDA kernel and the host-emulation parity test.  It performs
// exactly the CPU reference's integer operations: 5-tap dip_avg features,
// the Q10 Dip_Weights transform with (c + 512) >> 10 and Clip1, then the
// two-pass horizontal/vertical resampler.

#ifndef AV2ENC_CUDA_DIP_DIP_DEVICE_CUH_
#define AV2ENC_CUDA_DIP_DIP_DEVICE_CUH_

#include "dip.h"

#include "encoder/dip/dip_weights.h"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::dip {
namespace {

// Device copy of Dip_Weights.  The initializer body is the same generated
// file used by the CPU reference; only the storage class differs.
__device__ const std::int16_t kDipWeightsDevice[6][64][11] = {
#include "encoder/dip/dip_weights_data.inc"
};

constexpr int kNumFeatures = 11;
constexpr int kTransformSize = 8;
constexpr int kTransformSamples = kTransformSize * kTransformSize;
constexpr int kMaxEdgeSamples = kMaxBlockDim + kMaxBlockDim / 4;  // 80

struct EdgeArray {
  std::int16_t samples[kMaxEdgeSamples + 2];  // index -1 .. 80

  __device__ __forceinline__ std::int16_t& operator[](int index) {
    return samples[index + 1];
  }
  __device__ __forceinline__ std::int16_t operator[](int index) const {
    return samples[index + 1];
  }
};

struct DipSharedScratch {
  EdgeArray above;
  EdgeArray left;
  int features[kNumFeatures];
  std::uint16_t dip_pred[kTransformSamples];
  std::int16_t horz[kTransformSize * kMaxBlockDim];
};

__device__ __forceinline__ int dip_clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ int dip_clip1_8(int value) {
  return dip_clip3(0, 255, value);
}

__device__ __forceinline__ int dip_round2_10(int value) {
  return (value + 512) >> 10;
}

__device__ __forceinline__ int dip_read(const DipBlock& block, int x, int y) {
  return static_cast<int>(
      block.recon[static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(block.recon_stride) +
                  static_cast<std::size_t>(x)]);
}

__device__ __forceinline__ int dip_valid_dim(int v) {
  return v == 4 || v == 8 || v == 16 || v == 32 || v == 64;
}

__device__ __forceinline__ int dip_block_available(int w, int h) {
  return dip_valid_dim(w) && dip_valid_dim(h) && w > 4 && h > 4 &&
         w * h >= 128;
}

// §7.13.2.4 dip_avg.  Returns false if n cannot be downsampled.
__device__ __forceinline__ bool dip_avg_device(const EdgeArray& row, int n,
                                               int* out) {
  const int down = n >> 2;
  if (down <= 0 || n > kMaxBlockDim) return false;
  if (5 * down - 1 > kMaxEdgeSamples - 1) return false;
  for (int i = 0; i < 5; ++i) {
    int t = 0;
    for (int j = 0; j < down; ++j) {
      t += row[i * down + j];
    }
    out[i] = (t + (down >> 1)) / down;
  }
  return true;
}

// §7.13.2.1 edge preparation with useDip = 1 and MrlIndex = 0.
__device__ __forceinline__ void dip_build_edges_device(
    const DipBlock& block, int tid, int nthr, DipSharedScratch* s) {
  const int num_above = block.w + (block.w >> 2);
  const int num_left = block.h + (block.h >> 2);
  const int mid = 128;

  for (int i = tid; i < num_left; i += nthr) {
    if (block.have_left == 0 && block.have_above != 0) {
      s->left[i] =
          static_cast<std::int16_t>(dip_read(block, block.x, block.y - 1));
    } else if (block.have_left == 0) {
      s->left[i] = static_cast<std::int16_t>(mid + 1);
    } else {
      const int raw_limit =
          block.y + block.h + 4 * block.num4_below_left - 1;
      const int limit = block.max_y < raw_limit ? block.max_y : raw_limit;
      const int row = limit < block.y + i ? limit : block.y + i;
      s->left[i] = static_cast<std::int16_t>(
          dip_read(block, block.x - 1, row));
    }
  }

  for (int i = tid; i < num_above; i += nthr) {
    if (block.have_above == 0 && block.have_left != 0) {
      s->above[i] =
          static_cast<std::int16_t>(dip_read(block, block.x - 1, block.y));
    } else if (block.have_above == 0) {
      s->above[i] = static_cast<std::int16_t>(mid - 1);
    } else {
      const int raw_limit =
          block.x + block.w + 4 * block.num4_above_right - 1;
      const int limit = block.max_x < raw_limit ? block.max_x : raw_limit;
      const int col = limit < block.x + i ? limit : block.x + i;
      s->above[i] =
          static_cast<std::int16_t>(dip_read(block, col, block.y - 1));
    }
  }

  if (tid == 0) {
    if (block.have_above != 0 && block.have_left != 0) {
      const int corner = dip_read(block, block.x - 1, block.y - 1);
      s->above[-1] = static_cast<std::int16_t>(corner);
      s->left[-1] = static_cast<std::int16_t>(corner);
    } else if (block.have_above != 0) {
      const int corner = dip_read(block, block.x, block.y - 1);
      s->above[-1] = static_cast<std::int16_t>(corner);
      s->left[-1] = static_cast<std::int16_t>(corner);
    } else if (block.have_left != 0) {
      const int corner = dip_read(block, block.x - 1, block.y);
      s->above[-1] = static_cast<std::int16_t>(corner);
      s->left[-1] = static_cast<std::int16_t>(corner);
    } else {
      s->above[-1] = static_cast<std::int16_t>(mid);
      s->left[-1] = static_cast<std::int16_t>(mid);
    }
  }
}

// §7.13.2.4 features.  Thread 0 writes s->features.
__device__ __forceinline__ void dip_features_device(int w, int h,
                                                    int transpose,
                                                    DipSharedScratch* s) {
  int above_avg[5];
  int left_avg[5];
  if (!dip_avg_device(s->above, w, above_avg)) return;
  if (!dip_avg_device(s->left, h, left_avg)) return;
  s->features[0] = s->above[-1];
  for (int i = 0; i < 4; ++i) {
    s->features[i + 1] = transpose ? left_avg[i] : above_avg[i];
    s->features[i + 5] = transpose ? above_avg[i] : left_avg[i];
  }
  s->features[9] = transpose ? left_avg[4] : above_avg[4];
  s->features[10] = transpose ? above_avg[4] : left_avg[4];
}

// §7.13.2.5 transform into the 8x8 dip_pred matrix.
__device__ __forceinline__ void dip_transform_device(int mode, int transpose,
                                                     int tid, int nthr,
                                                     DipSharedScratch* s) {
  for (int idx = tid; idx < kTransformSamples; idx += nthr) {
    const int i = idx / kTransformSize;
    const int j = idx % kTransformSize;
    std::int32_t c = 0;
    for (int k = 0; k < kNumFeatures; ++k) {
      c += static_cast<std::int32_t>(
               kDipWeightsDevice[mode][idx][k]) *
           s->features[k];
    }
    const int v = dip_clip1_8(dip_round2_10(c));
    const int dst_index =
        transpose ? j * kTransformSize + i : i * kTransformSize + j;
    s->dip_pred[dst_index] = static_cast<std::uint16_t>(v);
  }
}

// §7.13.2.6 resample to w x h.
__device__ __forceinline__ void dip_resample_device(const DipBlock& block,
                                                    int tid, int nthr,
                                                    DipSharedScratch* s,
                                                    std::uint8_t* dst) {
  const int w = block.w;
  const int h = block.h;
  const int upx = w / kTransformSize > 1 ? w / kTransformSize : 1;
  const int upy = h / kTransformSize > 1 ? h / kTransformSize : 1;
  const int downx = kTransformSize / w > 1 ? kTransformSize / w : 1;
  const int downy = kTransformSize / h > 1 ? kTransformSize / h : 1;
  const int rows = h < kTransformSize ? h : kTransformSize;
  const int cols = w < kTransformSize ? w : kTransformSize;

  const int pairs = rows * cols;
  for (int idx = tid; idx < pairs; idx += nthr) {
    const int i = idx / cols;
    const int j = idx % cols;
    const int y = (i + 1) * upy - 1;
    const int p0 =
        j == 0 ? s->left[y]
               : s->dip_pred[i * downy * kTransformSize + (j - 1) * downx];
    const int p1 = s->dip_pred[i * downy * kTransformSize + j * downx];
    for (int k = 0; k < upx; ++k) {
      const int x = j * upx + k;
      const int w1 = k + 1;
      s->horz[i * kMaxBlockDim + x] = static_cast<std::int16_t>(
          ((upx - w1) * p0 + w1 * p1) / upx);
    }
  }

  __syncthreads();

  const int vpairs = w * rows;
  for (int idx = tid; idx < vpairs; idx += nthr) {
    const int x = idx / rows;
    const int i = idx % rows;
    const int p0 = i == 0 ? s->above[x] : s->horz[(i - 1) * kMaxBlockDim + x];
    const int p1 = s->horz[i * kMaxBlockDim + x];
    for (int k = 0; k < upy; ++k) {
      const int y = i * upy + k;
      const int w1 = k + 1;
      dst[static_cast<std::size_t>(y) *
              static_cast<std::size_t>(w) +
          static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(
          dip_clip1_8(((upy - w1) * p0 + w1 * p1) / upy));
    }
  }
}

// Runs one complete DIP prediction.  All threads in the CUDA block must call
// this function with the same block/params.
__device__ __forceinline__ void dip_run_block_device(
    const DipBlock& block, const DipParams& params, std::uint8_t* pred,
    DipSharedScratch* s) {
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  if (nthr <= 0 || !dip_block_available(block.w, block.h)) return;

  dip_build_edges_device(block, tid, nthr, s);
  __syncthreads();
  if (tid == 0) {
    dip_features_device(block.w, block.h, params.transpose, s);
  }
  __syncthreads();
  dip_transform_device(params.mode, params.transpose, tid, nthr, s);
  __syncthreads();
  dip_resample_device(block, tid, nthr, s, pred);
}

}  // namespace
}  // namespace av2enc::cuda::dip

#endif  // AV2ENC_CUDA_DIP_DIP_DEVICE_CUH_
