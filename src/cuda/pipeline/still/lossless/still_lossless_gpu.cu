// CUDA DPCM/FSC block stage for the CPU lossless still writer.
//
// See still_lossless_gpu.h for the contract.  The kernel reproduces the exact
// integer arithmetic of `src/codec/entropy/lossless/lossless.cpp`:
//   * DC / V / H prediction, including the §7.13.2.10 availability rules and
//     the DC denominator (8 or 16, both powers of two, so the §7.13.3.22
//     Div_Lut form reduces to round2(sum, log2(w + h)));
//   * the encoder-side reversible WHT of §7.15.2.2 and the DPCM first
//     difference of §7.15.4.
//
// One thread codes one 8x8 luma block.  Blocks only read the source plane, so
// no ordering or synchronization is needed between them.

#include "cuda/pipeline/still/lossless/still_lossless_gpu.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/arena.h"

namespace av2enc::cuda::still_lossless {
namespace {

using ::av2enc::lossless::BlockCoding;
using ::av2enc::lossless::Options;

constexpr int kDcPred = 0;
constexpr int kV = 1;
constexpr int kH = 2;

// §5.20.7.30 get_scan(2D), 8x8 (the same table the CPU writer uses).
__device__ const int kScan8[64] = {
    0,  8,  1,  16, 9,  2,  24, 17, 10, 3,  32, 25, 18, 11, 4,  40,
    33, 26, 19, 12, 5,  48, 41, 34, 27, 20, 13, 6,  56, 49, 42, 35,
    28, 21, 14, 7,  57, 50, 43, 36, 29, 22, 15, 58, 51, 44, 37, 30,
    23, 59, 52, 45, 38, 31, 60, 53, 46, 39, 61, 54, 47, 62, 55, 63};

__device__ __forceinline__ int clip3_dev(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ int round2_dev(int value, int shift) {
  if (shift <= 0) return value;
  return (value + (1 << (shift - 1))) >> shift;
}

__device__ __forceinline__ int floor_log2_dev(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// Encoder-side forward WHT, exactly inverting the decoder's lifting steps.
__device__ __forceinline__ void wht4_forward(const int in[4], int out[4]) {
  const int A = in[0];
  const int B = in[1];
  const int C = in[2];
  const int D = in[3];
  const int a1 = A + B;
  const int d1 = D - C;
  const int e = (a1 - d1) >> 1;
  const int b0 = e - B;
  const int c0 = e - C;
  const int a0 = a1 - c0;
  const int d0 = d1 + b0;
  out[0] = a0;
  out[1] = c0;
  out[2] = d0;
  out[3] = b0;
}

// Column pass then row pass, matching forward_lossless_4x4().
__device__ __forceinline__ void lossless_forward_4x4(const int residual[16],
                                                     int levels[16]) {
  int tmp[16] = {};
  for (int c = 0; c < 4; ++c) {
    int col[4] = {residual[c], residual[4 + c], residual[8 + c],
                  residual[12 + c]};
    int out[4] = {};
    wht4_forward(col, out);
    for (int r = 0; r < 4; ++r) tmp[r * 4 + c] = out[r];
  }
  for (int r = 0; r < 4; ++r) {
    int row[4] = {tmp[r * 4 + 0], tmp[r * 4 + 1], tmp[r * 4 + 2],
                  tmp[r * 4 + 3]};
    int out[4] = {};
    wht4_forward(row, out);
    for (int c = 0; c < 4; ++c) levels[r * 4 + c] = out[c];
  }
}

__device__ __forceinline__ void dpcm_difference_dev(const int* residual, int w,
                                                    int h, int vertical,
                                                    int* out) {
  for (int i = 0; i < w * h; ++i) out[i] = residual[i];
  if (vertical != 0) {
    for (int j = 0; j < w; ++j) {
      for (int i = 1; i < h; ++i) {
        out[i * w + j] = residual[i * w + j] - residual[(i - 1) * w + j];
      }
    }
  } else {
    for (int i = 0; i < h; ++i) {
      for (int j = 1; j < w; ++j) {
        out[i * w + j] = residual[i * w + j] - residual[i * w + (j - 1)];
      }
    }
  }
}

// §7.13.2.10 DC / §7.13.2.14 V / H prediction from the source plane.  Uses the
// same availability substitutions as intra::prepare_edges(): no left -> 128
// (mid), no above -> 127 (mid - 1), no neighbours -> 128.
__device__ __forceinline__ void predict_block_dev(
    const std::uint8_t* plane, int stride, int x, int y, int w, int h, int mode,
    int have_left, int have_above, int* pred) {
  // §7.13.2.1 edge substitution: a missing above row is filled with the left
  // sample (x - 1, y), a missing left column with the above sample (x, y - 1).
  // Only when both are missing do the mid +/- 1 defaults apply.
  if (mode == kV) {
    if (have_above) {
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          pred[i * w + j] =
              static_cast<int>(plane[(y - 1) * stride + x + j]);
        }
      }
    } else {
      const int value = have_left
                            ? static_cast<int>(plane[y * stride + x - 1])
                            : 127;
      for (int i = 0; i < w * h; ++i) pred[i] = value;
    }
    return;
  }
  if (mode == kH) {
    for (int i = 0; i < h; ++i) {
      const int value =
          have_left ? static_cast<int>(plane[(y + i) * stride + x - 1])
                    : (have_above
                           ? static_cast<int>(plane[(y - 1) * stride + x])
                           : 128);
      for (int j = 0; j < w; ++j) pred[i * w + j] = value;
    }
    return;
  }
  int value = 128;
  if (have_left && have_above) {
    int sum = 0;
    for (int k = 0; k < h; ++k) {
      sum += static_cast<int>(plane[(y + k) * stride + x - 1]);
    }
    for (int k = 0; k < w; ++k) {
      sum += static_cast<int>(plane[(y - 1) * stride + x + k]);
    }
    value = clip3_dev(0, 255, round2_dev(sum, floor_log2_dev(w + h)));
  } else if (have_left) {
    int sum = 0;
    for (int k = 0; k < h; ++k) {
      sum += static_cast<int>(plane[(y + k) * stride + x - 1]);
    }
    value = round2_dev(sum, floor_log2_dev(h));
  } else if (have_above) {
    int sum = 0;
    for (int k = 0; k < w; ++k) {
      sum += static_cast<int>(plane[(y - 1) * stride + x + k]);
    }
    value = round2_dev(sum, floor_log2_dev(w));
  }
  for (int i = 0; i < w * h; ++i) pred[i] = value;
}

// One 4x4 candidate: residual -> (optional DPCM) -> reversible WHT.  Returns
// the sum of absolute syntax levels, the CPU cost function.
__device__ int luma_cost_dev(const std::uint8_t* src, int stride, int bx,
                             int by, int mode, int use_dpcm, int vertical) {
  int total = 0;
  for (int ty = 0; ty < 2; ++ty) {
    for (int tx = 0; tx < 2; ++tx) {
      const int x = bx * 8 + tx * 4;
      const int y = by * 8 + ty * 4;
      const int have_left = (bx > 0) || (tx > 0);
      const int have_above = (by > 0) || (ty > 0);
      int pred[16];
      int residual[16];
      int tmp[16];
      int levels[16];
      predict_block_dev(src, stride, x, y, 4, 4, mode, have_left, have_above,
                        pred);
      for (int i = 0; i < 16; ++i) {
        residual[i] =
            static_cast<int>(src[static_cast<std::size_t>(y + i / 4) * stride +
                                 (x + i % 4)]) -
            pred[i];
      }
      if (use_dpcm != 0) {
        dpcm_difference_dev(residual, 4, 4, vertical, tmp);
      } else {
        for (int i = 0; i < 16; ++i) tmp[i] = residual[i];
      }
      lossless_forward_4x4(tmp, levels);
      for (int i = 0; i < 16; ++i) total += abs(levels[i]);
    }
  }
  return total;
}

__device__ int chroma_cost_dev(const std::uint8_t* plane, int stride, int bx,
                               int by, int mode, int use_dpcm, int vertical) {
  const int x = bx * 4;
  const int y = by * 4;
  int pred[16];
  int residual[16];
  int tmp[16];
  int levels[16];
  predict_block_dev(plane, stride, x, y, 4, 4, mode, bx > 0, by > 0, pred);
  for (int i = 0; i < 16; ++i) {
    residual[i] =
        static_cast<int>(
            plane[static_cast<std::size_t>(y + i / 4) * stride + (x + i % 4)]) -
        pred[i];
  }
  if (use_dpcm != 0) {
    dpcm_difference_dev(residual, 4, 4, vertical, tmp);
  } else {
    for (int i = 0; i < 16; ++i) tmp[i] = residual[i];
  }
  lossless_forward_4x4(tmp, levels);
  int total = 0;
  for (int i = 0; i < 16; ++i) total += abs(levels[i]);
  return total;
}

__global__ void lossless_block_kernel(const std::uint8_t* src_y, int y_stride,
                                      const std::uint8_t* src_u,
                                      const std::uint8_t* src_v, int c_stride,
                                      int width, int height, int blocks_x,
                                      int blocks_y, int enable_dpcm,
                                      int fsc_mode, BlockCoding* out) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = blocks_x * blocks_y;
  if (index >= count) return;
  const int bx = index % blocks_x;
  const int by = index / blocks_x;

  BlockCoding coding{};
  coding.y_mode = kDcPred;
  coding.uv_mode = kDcPred;

  // ---- luma candidate search (CPU choose_plan() order) ------------------
  if (fsc_mode != 2) {  // kForce skips the DPCM search
    int best_cost = luma_cost_dev(src_y, y_stride, bx, by, kDcPred, 0, 0);
    if (enable_dpcm != 0) {
      const int vc = luma_cost_dev(src_y, y_stride, bx, by, kV, 1, 1);
      const int hc = luma_cost_dev(src_y, y_stride, bx, by, kH, 1, 0);
      if (vc < best_cost) {
        best_cost = vc;
        coding.y_mode = kV;
        coding.use_dpcm_y = 1;
        coding.dpcm_y_vertical = 1;
      }
      if (hc < best_cost) {
        coding.y_mode = kH;
        coding.use_dpcm_y = 1;
        coding.dpcm_y_vertical = 0;
      }
    }
  }

  // ---- chroma candidate search, keeping the luma decision -----------------
  //
  // Reference quirk (keep it): the CPU's V/H chroma candidates copy `best`
  // and only flip the DPCM flags, so their prediction mode is the incumbent
  // uv_mode, not the candidate's own mode.  `uv_mode` is assigned after the
  // cost comparison.  The stream must follow the same search.
  int best_chroma = chroma_cost_dev(src_u, c_stride, bx, by, kDcPred, 0, 0) +
                    chroma_cost_dev(src_v, c_stride, bx, by, kDcPred, 0, 0);
  if (enable_dpcm != 0) {
    const int vc = chroma_cost_dev(src_u, c_stride, bx, by, coding.uv_mode, 1,
                                   1) +
                   chroma_cost_dev(src_v, c_stride, bx, by, coding.uv_mode, 1,
                                   1);
    if (vc < best_chroma) {
      best_chroma = vc;
      coding.use_dpcm_uv = 1;
      coding.dpcm_uv_vertical = 1;
      coding.uv_mode = kV;
    }
    const int hc = chroma_cost_dev(src_u, c_stride, bx, by, coding.uv_mode, 1,
                                   0) +
                   chroma_cost_dev(src_v, c_stride, bx, by, coding.uv_mode, 1,
                                   0);
    if (hc < best_chroma) {
      coding.use_dpcm_uv = 1;
      coding.dpcm_uv_vertical = 0;
      coding.uv_mode = kH;
    }
  }

  // ---- FSC decision ------------------------------------------------------
  if (fsc_mode == 2) {
    coding.fsc = 1;
    coding.use_dpcm_y = 0;
    coding.y_mode = kDcPred;
  } else if (fsc_mode == 1) {
    int pred[64];
    int residual[64];
    predict_block_dev(src_y, y_stride, bx * 8, by * 8, 8, 8, kDcPred, bx > 0,
                      by > 0, pred);
    for (int i = 0; i < 64; ++i) {
      residual[i] =
          static_cast<int>(src_y[static_cast<std::size_t>(by * 8 + i / 8) *
                                     y_stride +
                                 (bx * 8 + i % 8)]) -
          pred[i];
    }
    int first = -1;
    for (int i = 0; i < 64; ++i) {
      if (residual[kScan8[i]] != 0) {
        first = i;
        break;
      }
    }
    if (first > 0) coding.fsc = 1;
  }

  // ---- luma levels for the decided plan ---------------------------------
  if (coding.fsc != 0) {
    int pred[64];
    int residual[64];
    predict_block_dev(src_y, y_stride, bx * 8, by * 8, 8, 8, coding.y_mode,
                      bx > 0, by > 0, pred);
    for (int i = 0; i < 64; ++i) {
      residual[i] =
          static_cast<int>(src_y[static_cast<std::size_t>(by * 8 + i / 8) *
                                     y_stride +
                                 (bx * 8 + i % 8)]) -
          pred[i];
    }
    if (coding.use_dpcm_y != 0) {
      dpcm_difference_dev(residual, 8, 8, coding.dpcm_y_vertical,
                          coding.luma_fsc);
    } else {
      for (int i = 0; i < 64; ++i) coding.luma_fsc[i] = residual[i];
    }
  } else {
    for (int ty = 0; ty < 2; ++ty) {
      for (int tx = 0; tx < 2; ++tx) {
        const int x = bx * 8 + tx * 4;
        const int y = by * 8 + ty * 4;
        const int have_left = (bx > 0) || (tx > 0);
        const int have_above = (by > 0) || (ty > 0);
        int pred[16];
        int residual[16];
        int tmp[16];
        predict_block_dev(src_y, y_stride, x, y, 4, 4, coding.y_mode,
                          have_left, have_above, pred);
        for (int i = 0; i < 16; ++i) {
          residual[i] =
              static_cast<int>(src_y[static_cast<std::size_t>(y + i / 4) *
                                         y_stride +
                                     (x + i % 4)]) -
              pred[i];
        }
        if (coding.use_dpcm_y != 0) {
          dpcm_difference_dev(residual, 4, 4, coding.dpcm_y_vertical, tmp);
        } else {
          for (int i = 0; i < 16; ++i) tmp[i] = residual[i];
        }
        lossless_forward_4x4(tmp, coding.luma[ty * 2 + tx]);
      }
    }
  }

  // ---- chroma levels -----------------------------------------------------
  const std::uint8_t* chroma[2] = {src_u, src_v};
  for (int plane = 0; plane < 2; ++plane) {
    const int x = bx * 4;
    const int y = by * 4;
    int pred[16];
    int residual[16];
    int tmp[16];
    predict_block_dev(chroma[plane], c_stride, x, y, 4, 4, coding.uv_mode,
                      bx > 0, by > 0, pred);
    for (int i = 0; i < 16; ++i) {
      residual[i] =
          static_cast<int>(chroma[plane][static_cast<std::size_t>(y + i / 4) *
                                             c_stride +
                                         (x + i % 4)]) -
          pred[i];
    }
    if (coding.use_dpcm_uv != 0) {
      dpcm_difference_dev(residual, 4, 4, coding.dpcm_uv_vertical, tmp);
    } else {
      for (int i = 0; i < 16; ++i) tmp[i] = residual[i];
    }
    if (coding.fsc != 0) {
      for (int i = 0; i < 16; ++i) coding.chroma[plane][i] = tmp[i];
    } else {
      lossless_forward_4x4(tmp, coding.chroma[plane]);
    }
  }

  out[index] = coding;
  (void)width;
  (void)height;
}

bool cuda_ok(cudaError_t status, const char* what, std::string& error) {
  if (status == cudaSuccess) return true;
  error = std::string("lossless GPU: ") + what + ": " + cudaGetErrorString(status);
  return false;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

bool available() noexcept { return true; }

std::uint64_t required_device_bytes(int width, int height) noexcept {
  if (width <= 0 || height <= 0) return 0;
  const std::uint64_t blocks = static_cast<std::uint64_t>((width + 7) / 8) *
                               static_cast<std::uint64_t>((height + 7) / 8);
  std::uint64_t bytes = 0;
  bytes += align_up(static_cast<std::uint64_t>(width) * height,
                    Arena::kAlignment);
  const std::uint64_t chroma = static_cast<std::uint64_t>((width + 1) / 2) *
                               static_cast<std::uint64_t>((height + 1) / 2);
  bytes += align_up(chroma, Arena::kAlignment) * 2;
  bytes += align_up(blocks * sizeof(BlockCoding), Arena::kAlignment);
  bytes += Arena::kAlignment;  // init() base alignment slack
  return bytes;
}

bool compute_block_coding(const ::av2enc::lossless::StillPicture& picture,
                          const ::av2enc::lossless::Options& options,
                          std::vector<::av2enc::lossless::BlockCoding>* blocks,
                          std::string& error) {
  if (blocks == nullptr) {
    error = "lossless GPU: null output";
    return false;
  }
  blocks->clear();
  error.clear();
  const int width = static_cast<int>(picture.width);
  const int height = static_cast<int>(picture.height);
  if (width <= 0 || height <= 0 || (width & 7) != 0 || (height & 7) != 0) {
    error = "lossless GPU requires width/height positive multiples of 8";
    return false;
  }
  const int chroma_w = width / 2;
  const int chroma_h = height / 2;
  const std::size_t y_size = static_cast<std::size_t>(width) * height;
  const std::size_t c_size = static_cast<std::size_t>(chroma_w) * chroma_h;
  if (picture.y.size() < y_size || picture.u.size() < c_size ||
      picture.v.size() < c_size) {
    error = "lossless GPU: source planes are too small";
    return false;
  }

  const int blocks_x = width / 8;
  const int blocks_y = height / 8;
  const std::size_t count = static_cast<std::size_t>(blocks_x) * blocks_y;
  if (count == 0) {
    error = "lossless GPU: no coding blocks";
    return false;
  }

  const std::uint64_t need = required_device_bytes(width, height);
  Arena arena;
  std::string arena_error;
  const Status init = arena.init(need, Arena::kAlignment, &arena_error);
  if (init != Status::kOk) {
    error = arena_error.empty() ? "lossless GPU: arena init failed" : arena_error;
    return false;
  }

  std::uint8_t* d_y = arena.allocate_array<std::uint8_t>(y_size);
  std::uint8_t* d_u = arena.allocate_array<std::uint8_t>(c_size);
  std::uint8_t* d_v = arena.allocate_array<std::uint8_t>(c_size);
  BlockCoding* d_out = arena.allocate_array<BlockCoding>(count);
  if (d_y == nullptr || d_u == nullptr || d_v == nullptr || d_out == nullptr) {
    error = "lossless GPU: arena allocation failed";
    return false;
  }

  if (!cuda_ok(cudaMemcpy(d_y, picture.y.data(), y_size * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "upload Y", error) ||
      !cuda_ok(cudaMemcpy(d_u, picture.u.data(), c_size * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "upload U", error) ||
      !cuda_ok(cudaMemcpy(d_v, picture.v.data(), c_size * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "upload V", error)) {
    return false;
  }

  const int fsc_mode = options.fsc == ::av2enc::lossless::FscMode::kForce
                           ? 2
                           : (options.fsc == ::av2enc::lossless::FscMode::kAuto
                                  ? 1
                                  : 0);
  const int threads = 128;
  const int grid = static_cast<int>((count + threads - 1) / threads);
  lossless_block_kernel<<<grid, threads, 0, nullptr>>>(
      d_y, width, d_u, d_v, chroma_w, width, height, blocks_x, blocks_y,
      options.enable_dpcm ? 1 : 0, fsc_mode, d_out);
  if (!cuda_ok(cudaGetLastError(), "block kernel launch", error)) return false;
  if (!cuda_ok(cudaDeviceSynchronize(), "block kernel sync", error)) {
    return false;
  }

  blocks->resize(count);
  if (!cuda_ok(cudaMemcpy(blocks->data(), d_out, count * sizeof(BlockCoding),
                          cudaMemcpyDeviceToHost),
               "download block coding", error)) {
    blocks->clear();
    return false;
  }
  return true;
}

}  // namespace av2enc::cuda::still_lossless
