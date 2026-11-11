// Device-side AV2 MRL prediction (§7.13.2.1/.7/.8).
//
// Shared by the CUDA kernel and the host-emulation parity test, so both run
// the same integer code. Callers provide the reconstructed luma plane; the
// module never writes it.

#ifndef AV2ENC_CUDA_MRL_MRL_DEVICE_CUH_
#define AV2ENC_CUDA_MRL_MRL_DEVICE_CUH_

#include "mrl.h"

#include "../intra_modes/intra_modes.h"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::mrl {
namespace {

constexpr int kEdgeMinIndex = -2 - kMaxMrlIndex;
constexpr int kEdgeMaxIndex = 2 * kMaxBlockDim + 2 * kMaxMrlIndex + 2;
constexpr int kEdgeStorage = kEdgeMaxIndex - kEdgeMinIndex + 1;

struct EdgeLine {
  std::int16_t samples[kEdgeStorage];

  __device__ __forceinline__ std::int16_t& operator[](int index) {
    return samples[index - kEdgeMinIndex];
  }
  __device__ __forceinline__ std::int16_t operator[](int index) const {
    return samples[index - kEdgeMinIndex];
  }
};

struct MrlSharedScratch {
  EdgeLine above;
  EdgeLine left;
  EdgeLine above_sec;
  EdgeLine left_sec;
  int pangle;
};

__device__ __forceinline__ int mrl_clip3(int low, int high, int v) {
  return v < low ? low : (v > high ? high : v);
}

__device__ __forceinline__ int mrl_clip1(int v) {
  return mrl_clip3(0, 255, v);
}

__device__ __forceinline__ int mrl_round2(int value, int n) {
  if (n <= 0) return value;
  return (value + (1 << (n - 1))) >> n;
}

__device__ __forceinline__ int mrl_angle_step(int mrl_index) {
  return mrl_index == 1 ? 1 : (mrl_index == 2 ? -1 : 0);
}

__device__ __forceinline__ int mrl_pangle(int mode, int angle_delta,
                                          int mrl_index, int w, int h) {
  const int raw = av2enc::cuda::intra_modes::mode_to_angle(mode) +
                  angle_delta * av2enc::cuda::intra_modes::kAngleStep +
                  mrl_angle_step(mrl_index);
  return av2enc::cuda::intra_modes::wide_angle_map(raw, w, h);
}

__device__ __forceinline__ int mrl_read(const MrlBlock& b, int x, int y) {
  return static_cast<int>(b.recon[static_cast<std::size_t>(y) *
                                      static_cast<std::size_t>(b.recon_stride) +
                                  static_cast<std::size_t>(x)]);
}

// §7.13.2.1 with MrlIndex > 0 and the secondary arrays.
__device__ __forceinline__ void mrl_build_edges(const MrlBlock& b, int mrl_index,
                                                int tid, int nthr,
                                                MrlSharedScratch* s) {
  const bool sb_boundary =
      b.sb_height > 0 && (b.y & (b.sb_height - 1)) == 0;
  const int above_mrl = sb_boundary ? 0 : mrl_index;
  const int needed = b.w + b.h + (mrl_index << 1);

  for (int i = tid; i < needed; i += nthr) {
    if (b.have_left == 0 && b.have_above != 0) {
      s->left[i] = static_cast<std::int16_t>(
          mrl_read(b, b.x, b.y - 1 - above_mrl));
      s->left_sec[i] = static_cast<std::int16_t>(mrl_read(b, b.x, b.y - 1));
    } else if (b.have_left == 0) {
      s->left[i] = 129;
      s->left_sec[i] = 129;
    } else {
      const int limit =
          b.max_y < b.y + b.h + 4 * b.num4_below_left - 1
              ? b.max_y
              : b.y + b.h + 4 * b.num4_below_left - 1;
      const int row = limit < b.y + i ? limit : b.y + i;
      s->left[i] = static_cast<std::int16_t>(
          mrl_read(b, b.x - 1 - mrl_index, row));
      s->left_sec[i] =
          static_cast<std::int16_t>(mrl_read(b, b.x - 1, row));
    }

    if (b.have_above == 0 && b.have_left != 0) {
      s->above[i] = static_cast<std::int16_t>(
          mrl_read(b, b.x - 1 - mrl_index, b.y));
      s->above_sec[i] =
          static_cast<std::int16_t>(mrl_read(b, b.x - 1, b.y));
    } else if (b.have_above == 0) {
      s->above[i] = 127;
      s->above_sec[i] = 127;
    } else {
      const int limit =
          b.max_x < b.x + b.w + 4 * b.num4_above_right - 1
              ? b.max_x
              : b.x + b.w + 4 * b.num4_above_right - 1;
      const int col = limit < b.x + i ? limit : b.x + i;
      s->above[i] = static_cast<std::int16_t>(
          mrl_read(b, col, b.y - 1 - above_mrl));
      s->above_sec[i] =
          static_cast<std::int16_t>(mrl_read(b, col, b.y - 1));
    }
  }
}

__device__ __forceinline__ void mrl_build_negative(const MrlBlock& b,
                                                   int mrl_index, int above_mrl,
                                                   MrlSharedScratch* s) {
  for (int i = 1; i <= 1 + mrl_index; ++i) {
    if (b.have_above != 0 && b.have_left != 0) {
      s->above[-i] = static_cast<std::int16_t>(
          mrl_read(b, b.x - i, b.y - 1 - above_mrl));
      const int row = b.y - (i < 1 + above_mrl ? i : 1 + above_mrl);
      s->left[-i] = static_cast<std::int16_t>(
          mrl_read(b, b.x - 1 - mrl_index, row));
      s->above_sec[-i] =
          static_cast<std::int16_t>(mrl_read(b, b.x - i, b.y - 1));
      s->left_sec[-i] = static_cast<std::int16_t>(
          mrl_read(b, b.x - 1, b.y - 1));
    } else if (b.have_above != 0) {
      s->above[-i] = static_cast<std::int16_t>(
          mrl_read(b, b.x, b.y - 1 - above_mrl));
      s->left[-i] = s->above[-i];
      s->above_sec[-i] =
          static_cast<std::int16_t>(mrl_read(b, b.x, b.y - 1));
      s->left_sec[-i] = s->above_sec[-i];
    } else if (b.have_left != 0) {
      s->above[-i] = static_cast<std::int16_t>(
          mrl_read(b, b.x - 1 - mrl_index, b.y));
      s->left[-i] = s->above[-i];
      s->above_sec[-i] =
          static_cast<std::int16_t>(mrl_read(b, b.x - 1, b.y));
      s->left_sec[-i] = s->above_sec[-i];
    } else {
      s->above[-i] = 128;
      s->left[-i] = 128;
      s->above_sec[-i] = 128;
      s->left_sec[-i] = 128;
    }
  }
}

// §7.13.2.18 on one edge line (thread 0, size <= 129).
__device__ __forceinline__ void mrl_edge_filter(EdgeLine& row, int size,
                                                int strength) {
  if (size <= 1 || strength < 1 || strength > 3) return;
  std::int16_t snapshot[kEdgeStorage];
  for (int i = 0; i < size; ++i) {
    snapshot[i - kEdgeMinIndex] = row[i - 1];
  }
  for (int i = 1; i < size; ++i) {
    int sum = 0;
    for (int j = 0; j < av2enc::cuda::intra_modes::kEdgeTaps; ++j) {
      const int k = mrl_clip3(0, size - 1, i - 2 + j);
      sum += av2enc::cuda::intra_modes::intra_edge_kernel(strength - 1, j) *
             snapshot[k - kEdgeMinIndex];
    }
    row[i - 1] = static_cast<std::int16_t>((sum + 8) >> 4);
  }
}

// §7.13.2.8 edge extension for one (pAngle, mrlIndex) pair.
__device__ __forceinline__ void mrl_extend(EdgeLine& above, EdgeLine& left,
                                           int p_angle, int w, int h,
                                           int mrl_index) {
  const int min_base = -(1 + mrl_index);
  const int max_base = w + h - 1 + (mrl_index << 1);
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
  left[min_base - 1] = left[min_base];
  above[min_base - 1] = above[min_base];
}

__device__ __forceinline__ int mrl_edge_interpolate(const EdgeLine& edge,
                                                    int base, int shift) {
  int sum = 0;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    sum += av2enc::cuda::intra_modes::dr_interp_filter(shift, t) *
           edge[base + t - 1];
  }
  return mrl_clip1(mrl_round2(sum, 7));
}

// §7.13.2.8 single directional prediction with MrlIndex.
__device__ __forceinline__ int mrl_sample(int p_angle, int px, int py, int w,
                                          int h, int mrl_index,
                                          const EdgeLine& above,
                                          const EdgeLine& left) {
  if (p_angle < 90) {
    const int dx = av2enc::cuda::intra_modes::dr_intra_derivative(p_angle);
    const int idx = (py + 1 + mrl_index) * dx;
    const int base = (idx >> 6) + px;
    const int shift = (idx >> 1) & 0x1F;
    const int max_base_x = w + h - 1 + (mrl_index << 1);
    if (base < max_base_x + 1) {
      return mrl_edge_interpolate(above, base, shift);
    }
    return above[max_base_x];
  }
  if (p_angle > 90 && p_angle < 180) {
    const int dx = av2enc::cuda::intra_modes::dr_intra_derivative(180 - p_angle);
    const int dy = av2enc::cuda::intra_modes::dr_intra_derivative(p_angle - 90);
    const int idx = (px << 6) - (py + 1 + mrl_index) * dx;
    const int base = idx >> 6;
    if (base >= -(1 + mrl_index)) {
      return mrl_edge_interpolate(above, base, (idx >> 1) & 0x1F);
    }
    const int idx2 = (py << 6) - (px + 1 + mrl_index) * dy;
    return mrl_edge_interpolate(left, idx2 >> 6, (idx2 >> 1) & 0x1F);
  }
  if (p_angle > 180) {
    const int dy = av2enc::cuda::intra_modes::dr_intra_derivative(270 - p_angle);
    const int idx = (px + 1 + mrl_index) * dy;
    const int base = (idx >> 6) + py;
    const int shift = (idx >> 1) & 0x1F;
    const int max_base_y = w + h - 1 + (mrl_index << 1);
    if (base < max_base_y + 1) {
      return mrl_edge_interpolate(left, base, shift);
    }
    return left[max_base_y];
  }
  if (p_angle == 90) {
    return above[px];
  }
  return left[py];
}

__device__ __forceinline__ void mrl_run_block_device(const MrlBlock& b,
                                                     const MrlParams& p,
                                                     std::uint8_t* pred,
                                                     MrlSharedScratch* s) {
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  const int mrl_index = p.mrl_index;
  const int above_mrl =
      (b.sb_height > 0 && (b.y & (b.sb_height - 1)) == 0) ? 0 : mrl_index;

  mrl_build_edges(b, mrl_index, tid, nthr, s);
  __syncthreads();
  if (tid == 0) {
    mrl_build_negative(b, mrl_index, above_mrl, s);
    s->pangle = mrl_pangle(p.mode, p.angle_delta, mrl_index, b.w, b.h);
    if (mrl_index == 0 && p.enable_edge_filter != 0 && s->pangle != 90 &&
        s->pangle != 180) {
      const bool filter_type =
          (b.have_above != 0 && p.filter_type_above != 0) ||
          (b.have_left != 0 && p.filter_type_left != 0);
      if (s->pangle > 90 && s->pangle < 180 && b.w + b.h >= 24) {
        const int corner =
            mrl_round2(s->left[0] * 5 + s->above[-1] * 6 + s->above[0] * 5, 4);
        s->above[-1] = static_cast<std::int16_t>(corner);
        s->left[-1] = static_cast<std::int16_t>(corner);
      }
      if (b.have_above != 0) {
        const int strength = av2enc::cuda::intra_modes::intra_edge_strength(
            b.w, b.h, filter_type ? 1 : 0, s->pangle - 90);
        const int num_px = (b.w < b.max_x - b.x + 1 ? b.w : b.max_x - b.x + 1) +
                           (s->pangle < 90 ? b.h : 0) + 1;
        mrl_edge_filter(s->above, num_px, strength);
      }
      if (b.have_left != 0) {
        const int strength = av2enc::cuda::intra_modes::intra_edge_strength(
            b.w, b.h, filter_type ? 1 : 0, s->pangle - 180);
        const int num_px = (b.h < b.max_y - b.y + 1 ? b.h : b.max_y - b.y + 1) +
                           (s->pangle > 180 ? b.w : 0) + 1;
        mrl_edge_filter(s->left, num_px, strength);
      }
    }
    mrl_extend(s->above, s->left, s->pangle, b.w, b.h, mrl_index);
    if (mrl_index > 0 && p.mrl_sec_index != 0 && !(b.w == 4 && b.h == 4)) {
      mrl_extend(s->above_sec, s->left_sec, s->pangle, b.w, b.h, 0);
    }
  }
  __syncthreads();

  const int p_angle = s->pangle;
  const bool secondary = mrl_index > 0 && p.mrl_sec_index != 0 &&
                         !(b.w == 4 && b.h == 4);
  const int total = b.w * b.h;
  for (int index = tid; index < total; index += nthr) {
    const int py = index / b.w;
    const int px = index - py * b.w;
    int value = mrl_sample(p_angle, px, py, b.w, b.h, mrl_index, s->above,
                           s->left);
    if (secondary) {
      const int second = mrl_sample(p_angle, px, py, b.w, b.h, 0, s->above_sec,
                                    s->left_sec);
      value = (value + second + 1) >> 1;
    }
    pred[index] = static_cast<std::uint8_t>(value);
  }
}

}  // namespace
}  // namespace av2enc::cuda::mrl

#endif  // AV2ENC_CUDA_MRL_MRL_DEVICE_CUH_
