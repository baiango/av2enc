// AV2 v1.0.0 MRL CPU reference. See mrl.h for the normative section map.

#include "encoder/mrl/mrl.h"

#include <algorithm>
#include <cstdlib>

namespace av2enc::mrl {
namespace {

constexpr int kAngleStep = intra::kAngleStep;
constexpr int kMaxAngleDelta = intra::kMaxAngleDelta;
constexpr int kWaipRatio2Thres = 61;
constexpr int kWaipRatio4Thres = 73;
constexpr int kWaipRatio8Thres = 82;
constexpr int kWaipRatio16Thres = 86;

// Normative tables (§9), transcribed verbatim from AV2 v1.0.0.
constexpr int kDrIntraDerivative[90] = {
    0, 4096, 2048, 1365, 1024, 819, 682, 585, 512, 455, 409, 409, \
    409, 372, 341, 292, 273, 256, 227, 215, 204, 186, 178, 170, \
    157, 151, 146, 136, 132, 128, 117, 110, 107, 99, 97, 97, \
    93, 87, 83, 81, 77, 74, 73, 69, 66, 64, 62, 59, \
    56, 55, 53, 50, 49, 47, 44, 42, 42, 41, 38, 37, \
    35, 32, 31, 30, 28, 27, 26, 24, 23, 22, 20, 19, \
    18, 16, 15, 14, 12, 11, 10, 10, 10, 9, 8, 7, \
    6, 5, 4, 3, 2, 1};

constexpr int kDrInterpFilter[32][4] = {
    {0, 128, 0, 0}, \
    {-2, 127, 4, -1}, \
    {-3, 125, 8, -2}, \
    {-5, 123, 13, -3}, \
    {-6, 121, 17, -4}, \
    {-7, 118, 22, -5}, \
    {-9, 116, 27, -6}, \
    {-9, 112, 32, -7}, \
    {-10, 109, 37, -8}, \
    {-11, 106, 41, -8}, \
    {-11, 102, 46, -9}, \
    {-12, 98, 52, -10}, \
    {-12, 94, 56, -10}, \
    {-12, 90, 61, -11}, \
    {-12, 85, 66, -11}, \
    {-12, 81, 71, -12}, \
    {-12, 76, 76, -12}, \
    {-12, 71, 81, -12}, \
    {-11, 66, 85, -12}, \
    {-11, 61, 90, -12}, \
    {-10, 56, 94, -12}, \
    {-10, 52, 98, -12}, \
    {-9, 46, 102, -11}, \
    {-8, 41, 106, -11}, \
    {-8, 37, 109, -10}, \
    {-7, 32, 112, -9}, \
    {-6, 27, 116, -9}, \
    {-5, 22, 118, -7}, \
    {-4, 17, 121, -6}, \
    {-3, 13, 123, -5}, \
    {-2, 8, 125, -3}, \
    {-1, 4, 127, -2}};

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

int clip1(int value, int bit_depth) {
  return clip3(0, (1 << bit_depth) - 1, value);
}

int round2(int value, int n) {
  if (n <= 0) return value;
  return (value + (1 << (n - 1))) >> n;
}

int wide_angle_map(int p_angle, int w, int h) {
  if ((h == 2 * w && p_angle < kWaipRatio2Thres) ||
      (h == 4 * w && p_angle < kWaipRatio4Thres) ||
      (h == 8 * w && p_angle < kWaipRatio8Thres) ||
      (h == 16 * w && p_angle < kWaipRatio16Thres)) {
    return 180 + p_angle;
  }
  if ((w == 2 * h && p_angle > 270 - kWaipRatio2Thres) ||
      (w == 4 * h && p_angle > 270 - kWaipRatio4Thres) ||
      (w == 8 * h && p_angle > 270 - kWaipRatio8Thres) ||
      (w == 16 * h && p_angle > 270 - kWaipRatio16Thres)) {
    return p_angle - 180;
  }
  return p_angle;
}

void edge_filter(EdgeLine& row, int size, int strength) {
  if (size <= 1 || strength < 1 || strength > 3) return;
  constexpr int kKernel[3][5] = {
    {0, 4, 8, 4, 0}, {0, 5, 6, 5, 0}, {2, 4, 4, 4, 2}};
  const int* taps = kKernel[strength - 1];
  std::array<int, kEdgeMaxIndex + 2> snapshot{};
  for (int i = 0; i < size; ++i) {
    snapshot[static_cast<std::size_t>(i)] = row[i - 1];
  }
  for (int i = 1; i < size; ++i) {
    int sum = 0;
    for (int j = 0; j < 5; ++j) {
      const int k = clip3(0, size - 1, i - 2 + j);
      sum += taps[j] * snapshot[static_cast<std::size_t>(k)];
    }
    row[i - 1] = static_cast<std::int16_t>((sum + 8) >> 4);
  }
}

std::uint8_t read(const PlaneView& plane, int x, int y) {
  return plane.data[static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(plane.stride) +
                     static_cast<std::size_t>(x)];
}

bool valid_dim(int v) {
  return v == 4 || v == 8 || v == 16 || v == 32 || v == 64;
}

// §7.13.2.8 single directional prediction. `enable_idif` is 1 for luma.
void single_directional(int p_angle, int w, int h, int mrl_index,
                        int enable_idif, EdgeLine& above, EdgeLine& left,
                        std::uint8_t* dst, std::size_t dst_stride,
                        int bit_depth) {
  const int min_base = -(1 + mrl_index);
  const int max_base = w + h - 1 + (mrl_index << 1);
  if (enable_idif != 0) {
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

  if (p_angle < 90) {
    const int dx = kDrIntraDerivative[p_angle];
    const int max_base_x = w + h - 1 + (mrl_index << 1);
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const int idx = (i + 1 + mrl_index) * dx;
        const int base = (idx >> 6) + j;
        const int shift = (idx >> 1) & 0x1F;
        int value = 0;
        if (base < max_base_x + enable_idif) {
          if (enable_idif != 0) {
            int sum = 0;
            for (int t = 0; t < 4; ++t) {
              sum += kDrInterpFilter[shift][t] * above[base + t - 1];
            }
            value = clip1(round2(sum, 7), bit_depth);
          } else {
            value = round2(above[base] * (32 - shift) +
                               above[base + 1] * shift, 5);
          }
        } else {
          value = above[max_base_x];
        }
        dst[static_cast<std::size_t>(i) * dst_stride +
            static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
      }
    }
    return;
  }

  if (p_angle > 90 && p_angle < 180) {
    const int dx = kDrIntraDerivative[180 - p_angle];
    const int dy = kDrIntraDerivative[p_angle - 90];
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const int idx = (j << 6) - (i + 1 + mrl_index) * dx;
        const int base = idx >> 6;
        int value = 0;
        if (base >= min_base) {
          const int shift = (idx >> 1) & 0x1F;
          if (enable_idif != 0) {
            int sum = 0;
            for (int t = 0; t < 4; ++t) {
              sum += kDrInterpFilter[shift][t] * above[base + t - 1];
            }
            value = clip1(round2(sum, 7), bit_depth);
          } else {
            value = round2(above[base] * (32 - shift) +
                               above[base + 1] * shift, 5);
          }
        } else {
          const int idx2 = (i << 6) - (j + 1 + mrl_index) * dy;
          const int base2 = idx2 >> 6;
          const int shift = (idx2 >> 1) & 0x1F;
          if (enable_idif != 0) {
            int sum = 0;
            for (int t = 0; t < 4; ++t) {
              sum += kDrInterpFilter[shift][t] * left[base2 + t - 1];
            }
            value = clip1(round2(sum, 7), bit_depth);
          } else {
            value = round2(left[base2] * (32 - shift) +
                               left[base2 + 1] * shift, 5);
          }
        }
        dst[static_cast<std::size_t>(i) * dst_stride +
            static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
      }
    }
    return;
  }

  if (p_angle > 180) {
    const int dy = kDrIntraDerivative[270 - p_angle];
    const int max_base_y = w + h - 1 + (mrl_index << 1);
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const int idx = (j + 1 + mrl_index) * dy;
        const int base = (idx >> 6) + i;
        const int shift = (idx >> 1) & 0x1F;
        int value = 0;
        if (base < max_base_y + enable_idif) {
          if (enable_idif != 0) {
            int sum = 0;
            for (int t = 0; t < 4; ++t) {
              sum += kDrInterpFilter[shift][t] * left[base + t - 1];
            }
            value = clip1(round2(sum, 7), bit_depth);
          } else {
            value = round2(left[base] * (32 - shift) +
                               left[base + 1] * shift, 5);
          }
        } else {
          value = left[max_base_y];
        }
        dst[static_cast<std::size_t>(i) * dst_stride +
            static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
      }
    }
    return;
  }

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int value = p_angle == 90 ? above[j] : left[i];
      dst[static_cast<std::size_t>(i) * dst_stride +
          static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
    }
  }
}

}  // namespace

bool mode_allows_mrl(Mode mode) noexcept { return intra::is_directional(mode); }

const char* status_string(MrlStatus status) noexcept {
  return status == MrlStatus::kOk ? "ok" : "invalid argument";
}

MrlStatus prepare_edges(const Block& block, int bit_depth, int mrl_index,
                        Edges& edges) noexcept {
  if (block.plane != 0) {
    return MrlStatus::kInvalidArgument;  // MRL is luma-only
  }
  if (bit_depth < 8 || bit_depth > 12) {
    return MrlStatus::kInvalidArgument;
  }
  if (!valid_dim(block.w) || !valid_dim(block.h)) {
    return MrlStatus::kInvalidArgument;
  }
  if (block.recon.data == nullptr || block.recon.stride <= 0 ||
      block.x < 0 || block.y < 0 || block.max_x < block.x ||
      block.max_y < block.y || block.recon.stride < block.max_x + 1) {
    return MrlStatus::kInvalidArgument;
  }
  if (mrl_index < 0 || mrl_index > kMaxMrlIndex) {
    return MrlStatus::kInvalidArgument;
  }
  if (block.have_left && block.x == 0) return MrlStatus::kInvalidArgument;
  if (block.have_above && block.y == 0) return MrlStatus::kInvalidArgument;
  if (block.num4_above_right < 0 || block.num4_below_left < 0) {
    return MrlStatus::kInvalidArgument;
  }
  // The selected lines must be inside the reconstructed plane. A block at the
  // left edge has no left line; one below the SB boundary with MRL > 0 must
  // have enough rows above. The decoder gets this for free from block
  // alignment; reject malformed descriptors instead of reading out of bounds.
  if (block.have_left && block.x - 1 - mrl_index < 0) {
    return MrlStatus::kInvalidArgument;
  }
  const bool sb_boundary =
      block.sb_height > 0 && (block.y & (block.sb_height - 1)) == 0;
  const int above_mrl = sb_boundary ? 0 : mrl_index;
  if (block.have_above && block.y - 1 - above_mrl < 0) {
    return MrlStatus::kInvalidArgument;
  }

  edges.above.clear();
  edges.left.clear();
  edges.above_sec.clear();
  edges.left_sec.clear();
  edges.w = block.w;
  edges.h = block.h;
  edges.bit_depth = bit_depth;
  edges.mrl_index = mrl_index;
  edges.above_mrl_index = above_mrl;
  edges.have_left = block.have_left;
  edges.have_above = block.have_above;
  edges.is_luma = true;
  edges.valid_above_px = std::min(block.w, block.max_x - block.x + 1);
  edges.valid_left_px = std::min(block.h, block.max_y - block.y + 1);

  const int mid = 1 << (bit_depth - 1);
  const int num_needed = block.w + block.h + (mrl_index << 1);

  for (int i = 0; i < num_needed; ++i) {
    if (!block.have_left && block.have_above) {
      edges.left[i] = read(block.recon, block.x, block.y - 1 - above_mrl);
      edges.left_sec[i] = read(block.recon, block.x, block.y - 1);
    } else if (!block.have_left) {
      edges.left[i] = static_cast<std::int16_t>(mid + 1);
      edges.left_sec[i] = static_cast<std::int16_t>(mid + 1);
    } else {
      const int limit =
          std::min(block.max_y, block.y + block.h + 4 * block.num4_below_left - 1);
      const int row = std::min(limit, block.y + i);
      edges.left[i] =
          read(block.recon, block.x - 1 - mrl_index, row);
      edges.left_sec[i] = read(block.recon, block.x - 1, row);
    }
  }

  int num4_above_right = block.num4_above_right;
  if (block.plane > 0 && block.w > 32) num4_above_right = 0;
  for (int i = 0; i < num_needed; ++i) {
    if (!block.have_above && block.have_left) {
      edges.above[i] = read(block.recon, block.x - 1 - mrl_index, block.y);
      edges.above_sec[i] = read(block.recon, block.x - 1, block.y);
    } else if (!block.have_above) {
      edges.above[i] = static_cast<std::int16_t>(mid - 1);
      edges.above_sec[i] = static_cast<std::int16_t>(mid - 1);
    } else {
      const int limit =
          std::min(block.max_x, block.x + block.w + 4 * num4_above_right - 1);
      const int col = std::min(limit, block.x + i);
      edges.above[i] = read(block.recon, col, block.y - 1 - above_mrl);
      edges.above_sec[i] = read(block.recon, col, block.y - 1);
    }
  }

  for (int i = 1; i <= 1 + mrl_index; ++i) {
    if (block.have_above && block.have_left) {
      edges.above[-i] = read(block.recon, block.x - i, block.y - 1 - above_mrl);
      edges.left[-i] =
          read(block.recon, block.x - 1 - mrl_index,
               block.y - std::min(i, 1 + above_mrl));
      edges.above_sec[-i] = read(block.recon, block.x - i, block.y - 1);
      edges.left_sec[-i] = read(block.recon, block.x - 1, block.y - 1);
    } else if (block.have_above) {
      edges.above[-i] = read(block.recon, block.x, block.y - 1 - above_mrl);
      edges.left[-i] = edges.above[-i];
      edges.above_sec[-i] = read(block.recon, block.x, block.y - 1);
      edges.left_sec[-i] = edges.above_sec[-i];
    } else if (block.have_left) {
      edges.above[-i] = read(block.recon, block.x - 1 - mrl_index, block.y);
      edges.left[-i] = edges.above[-i];
      edges.above_sec[-i] = read(block.recon, block.x - 1, block.y);
      edges.left_sec[-i] = edges.above_sec[-i];
    } else {
      edges.above[-i] = static_cast<std::int16_t>(mid);
      edges.left[-i] = static_cast<std::int16_t>(mid);
      edges.above_sec[-i] = static_cast<std::int16_t>(mid);
      edges.left_sec[-i] = static_cast<std::int16_t>(mid);
    }
  }
  return MrlStatus::kOk;
}

MrlStatus validate_params(const Block& block, const Params& params) noexcept {
  if (block.plane != 0) return MrlStatus::kInvalidArgument;
  if (!mode_allows_mrl(params.mode)) return MrlStatus::kInvalidArgument;
  if (params.angle_delta < -kMaxAngleDelta ||
      params.angle_delta > kMaxAngleDelta) {
    return MrlStatus::kInvalidArgument;
  }
  if (params.mrl_index < 0 || params.mrl_index > kMaxMrlIndex) {
    return MrlStatus::kInvalidArgument;
  }
  if (params.mrl_index == 0 && params.mrl_sec_index) {
    return MrlStatus::kInvalidArgument;
  }
  if (params.mrl_sec_index && block.w == 4 && block.h == 4) {
    return MrlStatus::kInvalidArgument;  // decoder ignores secondary at 4x4
  }
  return MrlStatus::kOk;
}

MrlStatus predict(const Edges& edges, const Params& params, std::uint8_t* dst,
                  std::size_t dst_stride) noexcept {
  if (dst == nullptr || dst_stride < static_cast<std::size_t>(edges.w)) {
    return MrlStatus::kInvalidArgument;
  }
  if (!mode_allows_mrl(params.mode)) return MrlStatus::kInvalidArgument;
  if (params.angle_delta < -kMaxAngleDelta ||
      params.angle_delta > kMaxAngleDelta) {
    return MrlStatus::kInvalidArgument;
  }
  if (params.mrl_index != edges.mrl_index) return MrlStatus::kInvalidArgument;
  if (params.mrl_sec_index &&
      (edges.mrl_index == 0 || (edges.w == 4 && edges.h == 4))) {
    return MrlStatus::kInvalidArgument;
  }

  const int p_angle = wide_angle_map(
      intra::mode_to_angle(params.mode) + params.angle_delta * kAngleStep +
          mrl_angle_delta(edges.mrl_index),
      edges.w, edges.h);
  if (p_angle <= 0 || p_angle >= 270) return MrlStatus::kInvalidArgument;

  EdgeLine above = edges.above;
  EdgeLine left = edges.left;

  if (edges.mrl_index == 0 && params.enable_edge_filter && p_angle != 90 &&
      p_angle != 180) {
    const bool filter_type =
        (edges.have_above && params.above_neighbor_is_smooth) ||
        (edges.have_left && params.left_neighbor_is_smooth);
    if (p_angle > 90 && p_angle < 180 && edges.w + edges.h >= 24) {
      const int corner =
          round2(left[0] * 5 + above[-1] * 6 + above[0] * 5, 4);
      above[-1] = static_cast<std::int16_t>(corner);
      left[-1] = static_cast<std::int16_t>(corner);
    }
    if (edges.have_above) {
      const int strength = intra::edge_filter_strength(
          edges.w, edges.h, filter_type, p_angle - 90);
      const int num_px = std::min(edges.w, edges.valid_above_px) +
                         (p_angle < 90 ? edges.h : 0) + 1;
      edge_filter(above, num_px, strength);
    }
    if (edges.have_left) {
      const int strength = intra::edge_filter_strength(
          edges.w, edges.h, filter_type, p_angle - 180);
      const int num_px = std::min(edges.h, edges.valid_left_px) +
                         (p_angle > 180 ? edges.w : 0) + 1;
      edge_filter(left, num_px, strength);
    }
  }

  single_directional(p_angle, edges.w, edges.h, edges.mrl_index, 1, above,
                     left, dst, dst_stride, edges.bit_depth);

  if (edges.mrl_index > 0 && params.mrl_sec_index &&
      !(edges.w == 4 && edges.h == 4)) {
    EdgeLine above2 = edges.above_sec;
    EdgeLine left2 = edges.left_sec;
    std::array<std::uint8_t,
               static_cast<std::size_t>(kMaxBlockDim) * kMaxBlockDim>
        scratch{};
    single_directional(p_angle, edges.w, edges.h, 0, 1, above2, left2,
                       scratch.data(), static_cast<std::size_t>(edges.w),
                       edges.bit_depth);
    for (int i = 0; i < edges.h; ++i) {
      for (int j = 0; j < edges.w; ++j) {
        const std::size_t index = static_cast<std::size_t>(i) * edges.w + j;
        const int primary =
            dst[static_cast<std::size_t>(i) * dst_stride + j];
        const int secondary = scratch[index];
        dst[static_cast<std::size_t>(i) * dst_stride + j] =
            static_cast<std::uint8_t>((primary + secondary + 1) >> 1);
      }
    }
  }
  return MrlStatus::kOk;
}

MrlStatus predict_block(const Block& block, int bit_depth,
                        const Params& params, std::uint8_t* dst,
                        std::size_t dst_stride) noexcept {
  const MrlStatus status = validate_params(block, params);
  if (status != MrlStatus::kOk) return status;
  Edges edges;
  const MrlStatus edge_status =
      prepare_edges(block, bit_depth, params.mrl_index, edges);
  if (edge_status != MrlStatus::kOk) return edge_status;
  return predict(edges, params, dst, dst_stride);
}

}  // namespace av2enc::mrl
