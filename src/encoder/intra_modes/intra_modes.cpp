// AV2 intra prediction — CPU reference implementation.
// See intra_modes.h and README.md for scope and integration notes.

#include "encoder/intra_modes/intra_modes.h"

#include <algorithm>
#include <cstdlib>

namespace av2enc::intra {
namespace {

// ---------------------------------------------------------------------------
// §3 constants and small integer helpers
// ---------------------------------------------------------------------------

constexpr int kBlendWeightMax = 32;  // BLEND_WEIGHT_MAX

constexpr int kDivLutBits = 7;       // DIV_LUT_BITS
constexpr int kDivLutPrecBits = 9;   // DIV_LUT_PREC_BITS

// Mode_To_Angle[INTRA_MODES] (§5.20.7.29 constants).
constexpr int kModeToAngle[kNumModes] = {0,  90, 180, 45, 135, 113, 157,
                                         203, 67, 0,  0,   0,   0};

// Dr_Intra_Derivative[90] (§7.13.2.8).
constexpr int kDrIntraDerivative[90] = {
    0,    4096, 2048, 1365, 1024, 819,  682,  585,  512,  455,  409,  409,
    409,  372,  341,  292,  273,  256,  227,  215,  204,  186,  178,  170,
    157,  151,  146,  136,  132,  128,  117,  110,  107,  99,   97,   97,
    93,   87,   83,   81,   77,   74,   73,   69,   66,   64,   62,   59,
    56,   55,   53,   50,   49,   47,   44,   42,   42,   41,   38,   37,
    35,   32,   31,   30,   28,   27,   26,   24,   23,   22,   20,   19,
    18,   16,   15,   14,   12,   11,   10,   10,   10,   9,    8,    7,
    6,    5,    4,    3,    2,    1};

// Dr_Interp_Filter[32][4] (§7.13.2.8, luma IDIF).
constexpr int kDrInterpFilter[32][4] = {
    {0, 128, 0, 0},     {-2, 127, 4, -1},   {-3, 125, 8, -2},
    {-5, 123, 13, -3},  {-6, 121, 17, -4},  {-7, 118, 22, -5},
    {-9, 116, 27, -6},  {-9, 112, 32, -7},  {-10, 109, 37, -8},
    {-11, 106, 41, -8}, {-11, 102, 46, -9}, {-12, 98, 52, -10},
    {-12, 94, 56, -10}, {-12, 90, 61, -11}, {-12, 85, 66, -11},
    {-12, 81, 71, -12}, {-12, 76, 76, -12}, {-12, 71, 81, -12},
    {-11, 66, 85, -12}, {-11, 61, 90, -12}, {-10, 56, 94, -12},
    {-10, 52, 98, -12}, {-9, 46, 102, -11}, {-8, 41, 106, -11},
    {-8, 37, 109, -10}, {-7, 32, 112, -9},  {-6, 27, 116, -9},
    {-5, 22, 118, -7},  {-4, 17, 121, -6},  {-3, 13, 123, -5},
    {-2, 8, 125, -3},   {-1, 4, 127, -2}};

// Div_Lut[DIV_LUT_NUM] (§7.13.3.22).
constexpr int kDivLut[129] = {
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465,
    462, 458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 426, 423,
    420, 417, 415, 412, 410, 407, 405, 402, 400, 397, 395, 392, 390, 388,
    386, 383, 381, 379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358,
    356, 354, 352, 350, 349, 347, 345, 343, 341, 340, 338, 336, 334, 333,
    331, 329, 328, 326, 324, 323, 321, 320, 318, 317, 315, 314, 312, 311,
    309, 308, 306, 305, 303, 302, 301, 299, 298, 297, 295, 294, 293, 291,
    290, 289, 287, 286, 285, 284, 282, 281, 280, 279, 278, 277, 275, 274,
    273, 272, 271, 270, 269, 267, 266, 265, 264, 263, 262, 261, 260, 259,
    258, 257, 256};

int clip3(int low, int high, int value) noexcept {
  return value < low ? low : (value > high ? high : value);
}

int round2(int value, int shift) noexcept {
  if (shift <= 0) {
    return value;
  }
  return (value + (1 << (shift - 1))) >> shift;
}

int floor_log2(int value) noexcept {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

bool is_power_of_two(int value) noexcept {
  return value > 0 && (value & (value - 1)) == 0;
}

int resolve_divisor(int divisor) noexcept {
  const int magnitude = std::abs(divisor);
  const int n = floor_log2(magnitude);
  const int e = magnitude - (1 << n);
  const int f =
      n > kDivLutBits ? round2(e, n - kDivLutBits) : (e << (kDivLutBits - n));
  return kDivLut[f] * (divisor < 0 ? -1 : 1);
}

int divisor_shift(int divisor) noexcept {
  return floor_log2(std::abs(divisor)) + kDivLutPrecBits;
}

// §7.13.2.10/.11 approx_divide().
int approx_divide(int numerator, int denominator) noexcept {
  return round2(numerator * resolve_divisor(denominator),
                divisor_shift(denominator));
}

int clip1(int value, int bit_depth) noexcept {
  return clip3(0, (1 << bit_depth) - 1, value);
}

std::uint8_t read_sample(const PlaneView& plane, int col, int row) noexcept {
  return plane.data[static_cast<std::size_t>(row) *
                         static_cast<std::size_t>(plane.stride) +
                     static_cast<std::size_t>(col)];
}

bool valid_mode(Mode mode) noexcept {
  return static_cast<unsigned>(mode) <
         static_cast<unsigned>(kNumModes);
}

bool valid_block_size(int w, int h) noexcept {
  return w >= kMinBlockDim && w <= kMaxBlockDim && h >= kMinBlockDim &&
         h <= kMaxBlockDim && is_power_of_two(w) && is_power_of_two(h);
}

// ---------------------------------------------------------------------------
// §7.13.2.17 / §7.13.2.18 edge filtering
// ---------------------------------------------------------------------------

void filter_corner(EdgeRow& above, EdgeRow& left) noexcept {
  const int sum = left[0] * 5 + above[-1] * 6 + above[0] * 5;
  const int value = round2(sum, 4);
  left[-1] = static_cast<std::int16_t>(value);
  above[-1] = static_cast<std::int16_t>(value);
}

void predict_directional(const IntraEdges& edges, const MappedAngle& mapped,
                         EdgeRow& above, EdgeRow& left, std::uint8_t* dst,
                         std::size_t dst_stride) noexcept {
  const int w = edges.w;
  const int h = edges.h;
  const int p_angle = mapped.p_angle;
  const int enable_idif = edges.is_luma ? 1 : 0;

  // §7.13.2.8 single directional prediction: extend the edges so the
  // interpolation taps stay in bounds.
  constexpr int kMinBase = -1;  // MrlIndex = 0
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
  left[kMinBase - 1] = left[kMinBase];
  above[kMinBase - 1] = above[kMinBase];

  if (p_angle < 90) {
    const int dx = kDrIntraDerivative[p_angle];
    const int max_base_x = w + h - 1;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const int idx = (i + 1) * dx;
        const int base = (idx >> 6) + j;
        const int shift = (idx >> 1) & 0x1F;
        int value = 0;
        if (base < max_base_x + enable_idif) {
          if (enable_idif != 0) {
            const int* taps = kDrInterpFilter[shift];
            int sum = 0;
            for (int t = 0; t < 4; ++t) {
              sum += taps[t] * above[base + t - 1];
            }
            value = clip1(round2(sum, 7), edges.bit_depth);
          } else {
            value = round2(above[base] * (32 - shift) +
                               above[base + 1] * shift,
                           5);
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
        int idx = (j << 6) - (i + 1) * dx;
        int base = idx >> 6;
        int value = 0;
        if (base >= kMinBase) {
          const int shift = (idx >> 1) & 0x1F;
          if (enable_idif != 0) {
            const int* taps = kDrInterpFilter[shift];
            int sum = 0;
            for (int t = 0; t < 4; ++t) {
              sum += taps[t] * above[base + t - 1];
            }
            value = clip1(round2(sum, 7), edges.bit_depth);
          } else {
            value = round2(above[base] * (32 - shift) +
                               above[base + 1] * shift,
                           5);
          }
        } else {
          idx = (i << 6) - (j + 1) * dy;
          base = idx >> 6;
          const int shift = (idx >> 1) & 0x1F;
          if (enable_idif != 0) {
            const int* taps = kDrInterpFilter[shift];
            int sum = 0;
            for (int t = 0; t < 4; ++t) {
              sum += taps[t] * left[base + t - 1];
            }
            value = clip1(round2(sum, 7), edges.bit_depth);
          } else {
            value = round2(left[base] * (32 - shift) +
                               left[base + 1] * shift,
                           5);
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
    const int max_base_y = w + h - 1;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const int idx = (j + 1) * dy;
        const int base = (idx >> 6) + i;
        const int shift = (idx >> 1) & 0x1F;
        int value = 0;
        if (base < max_base_y + enable_idif) {
          if (enable_idif != 0) {
            const int* taps = kDrInterpFilter[shift];
            int sum = 0;
            for (int t = 0; t < 4; ++t) {
              sum += taps[t] * left[base + t - 1];
            }
            value = clip1(round2(sum, 7), edges.bit_depth);
          } else {
            value = round2(left[base] * (32 - shift) +
                               left[base + 1] * shift,
                           5);
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

  // p_angle == 90 or 180.
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int value = p_angle == 90 ? above[j] : left[i];
      dst[static_cast<std::size_t>(i) * dst_stride +
          static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

const char* status_string(IntraStatus status) noexcept {
  switch (status) {
    case IntraStatus::kOk:
      return "ok";
    case IntraStatus::kInvalidArgument:
      return "invalid argument";
  }
  return "unknown";
}

bool is_directional(Mode mode) noexcept {
  return mode >= Mode::kV && mode <= Mode::kD67;
}

int mode_to_angle(Mode mode) noexcept {
  if (!valid_mode(mode)) {
    return 0;
  }
  return kModeToAngle[static_cast<unsigned>(mode)];
}

MappedAngle map_angle(Mode mode, int angle_delta, int w, int h) noexcept {
  if (!is_directional(mode)) {
    return {mode, mode_to_angle(mode)};
  }
  int p_angle = mode_to_angle(mode) + angle_delta * kAngleStep;

  // wide_angle_mapping(), §5.20.7.29 constants (WAIP_WH_RATIO_*_THRES).
  constexpr int kThresh2 = 61;
  constexpr int kThresh4 = 73;
  constexpr int kThresh8 = 82;
  constexpr int kThresh16 = 86;
  if ((h == 2 * w && p_angle < kThresh2) ||
      (h == 4 * w && p_angle < kThresh4) ||
      (h == 8 * w && p_angle < kThresh8) ||
      (h == 16 * w && p_angle < kThresh16)) {
    return {Mode::kD203, 180 + p_angle};
  }
  if ((w == 2 * h && p_angle > 270 - kThresh2) ||
      (w == 4 * h && p_angle > 270 - kThresh4) ||
      (w == 8 * h && p_angle > 270 - kThresh8) ||
      (w == 16 * h && p_angle > 270 - kThresh16)) {
    return {Mode::kD45, p_angle - 180};
  }
  return {mode, p_angle};
}

int edge_filter_strength(int w, int h, bool smooth_neighbors,
                         int delta) noexcept {
  const int d = std::abs(delta);
  const int blk_wh = w + h;
  int strength = 0;
  if (!smooth_neighbors) {
    if (blk_wh <= 8) {
      if (d >= 56) {
        strength = 1;
      }
    } else if (blk_wh <= 12) {
      if (d >= 40) {
        strength = 1;
      }
    } else if (blk_wh <= 16) {
      if (d >= 40) {
        strength = 1;
      }
    } else if (blk_wh <= 24) {
      if (d >= 8) {
        strength = 1;
      }
      if (d >= 16) {
        strength = 2;
      }
      if (d >= 32) {
        strength = 3;
      }
    } else if (blk_wh <= 32) {
      strength = 1;
      if (d >= 4) {
        strength = 2;
      }
      if (d >= 32) {
        strength = 3;
      }
    } else {
      strength = 3;
    }
  } else {
    if (blk_wh <= 8) {
      if (d >= 40) {
        strength = 1;
      }
      if (d >= 64) {
        strength = 2;
      }
    } else if (blk_wh <= 16) {
      if (d >= 20) {
        strength = 1;
      }
      if (d >= 48) {
        strength = 2;
      }
    } else if (blk_wh <= 24) {
      if (d >= 4) {
        strength = 3;
      }
    } else {
      strength = 3;
    }
  }
  return strength;
}

void intra_edge_filter(EdgeRow& row, int size, int strength) noexcept {
  if (size <= 1 || size > EdgeRow::kMaxIndex || strength < 1 ||
      strength > 3) {
    return;
  }
  constexpr int kKernel[3][5] = {{0, 4, 8, 4, 0},
                                 {0, 5, 6, 5, 0},
                                 {2, 4, 4, 4, 2}};
  const int* taps = kKernel[strength - 1];
  for (int i = 1; i < size; ++i) {
    int sum = 0;
    for (int j = 0; j < 5; ++j) {
      const int k = clip3(0, size - 1, i - 2 + j);
      sum += taps[j] * row[k - 1];
    }
    row[i - 1] = static_cast<std::int16_t>((sum + 8) >> 4);
  }
}

// ---------------------------------------------------------------------------
// §7.13.2.1 edge preparation
// ---------------------------------------------------------------------------

IntraStatus prepare_edges(const NeighborRequest& request, int bit_depth,
                          IntraEdges& edges) noexcept {
  if (request.recon.data == nullptr || request.recon.stride <= 0) {
    return IntraStatus::kInvalidArgument;
  }
  if (!valid_block_size(request.w, request.h)) {
    return IntraStatus::kInvalidArgument;
  }
  if (bit_depth < 8 || bit_depth > 12) {
    return IntraStatus::kInvalidArgument;
  }
  if (request.max_x < 0 || request.max_y < 0 || request.x < 0 ||
      request.y < 0 || request.x > request.max_x ||
      request.y > request.max_y) {
    return IntraStatus::kInvalidArgument;
  }
  if (request.recon.stride < request.max_x + 1) {
    return IntraStatus::kInvalidArgument;
  }
  if (request.have_left && request.x == 0) {
    return IntraStatus::kInvalidArgument;
  }
  if (request.have_above && request.y == 0) {
    return IntraStatus::kInvalidArgument;
  }
  if (request.num4_above_right < 0 || request.num4_below_left < 0 ||
      request.color_plane < 0) {
    return IntraStatus::kInvalidArgument;
  }

  int num4_above_right = request.num4_above_right;
  int num4_below_left = request.num4_below_left;
  if (request.color_plane > 0) {
    if (request.w > 32) {
      num4_above_right = 0;
    }
    if (request.h > 32) {
      num4_below_left = 0;
    }
  }

  edges.above.clear();
  edges.left.clear();
  edges.w = request.w;
  edges.h = request.h;
  edges.bit_depth = bit_depth;
  edges.have_left = request.have_left;
  edges.have_above = request.have_above;
  edges.is_luma = request.color_plane == 0;
  edges.valid_above_px = std::min(request.w, request.max_x - request.x + 1);
  edges.valid_left_px = std::min(request.h, request.max_y - request.y + 1);

  const int mid = 1 << (bit_depth - 1);
  const int num_needed = request.w + request.h;

  for (int i = 0; i < num_needed; ++i) {
    if (!request.have_left && request.have_above) {
      edges.left[i] = read_sample(request.recon, request.x, request.y - 1);
    } else if (!request.have_left) {
      edges.left[i] = static_cast<std::int16_t>(mid + 1);
    } else {
      const int limit =
          std::min(request.max_y,
                   request.y + request.h + 4 * num4_below_left - 1);
      edges.left[i] = read_sample(request.recon, request.x - 1,
                                  std::min(limit, request.y + i));
    }
  }

  for (int i = 0; i < num_needed; ++i) {
    if (!request.have_above && request.have_left) {
      edges.above[i] = read_sample(request.recon, request.x - 1, request.y);
    } else if (!request.have_above) {
      edges.above[i] = static_cast<std::int16_t>(mid - 1);
    } else {
      const int limit =
          std::min(request.max_x,
                   request.x + request.w + 4 * num4_above_right - 1);
      edges.above[i] = read_sample(request.recon,
                                   std::min(limit, request.x + i),
                                   request.y - 1);
    }
  }

  // §7.13.2.1, i = 1 (MrlIndex = 0).
  if (request.have_above && request.have_left) {
    const int corner = read_sample(request.recon, request.x - 1, request.y - 1);
    edges.above[-1] = static_cast<std::int16_t>(corner);
    edges.left[-1] = static_cast<std::int16_t>(corner);
  } else if (request.have_above) {
    const int value = edges.above[0];
    edges.above[-1] = static_cast<std::int16_t>(value);
    edges.left[-1] = static_cast<std::int16_t>(value);
  } else if (request.have_left) {
    const int value = edges.left[0];
    edges.above[-1] = static_cast<std::int16_t>(value);
    edges.left[-1] = static_cast<std::int16_t>(value);
  } else {
    edges.above[-1] = static_cast<std::int16_t>(mid);
    edges.left[-1] = static_cast<std::int16_t>(mid);
  }

  return IntraStatus::kOk;
}

// ---------------------------------------------------------------------------
// §7.13.2.2 / .10 / .13 non-directional predictors
// ---------------------------------------------------------------------------

namespace {

void predict_dc(const IntraEdges& edges, std::uint8_t* dst,
                std::size_t dst_stride) noexcept {
  const int w = edges.w;
  const int h = edges.h;
  int value = 0;
  if (edges.have_left && edges.have_above) {
    int sum = 0;
    for (int k = 0; k < h; ++k) {
      sum += edges.left[k];
    }
    for (int k = 0; k < w; ++k) {
      sum += edges.above[k];
    }
    value = clip1(approx_divide(sum, w + h), edges.bit_depth);
  } else if (edges.have_left) {
    int sum = 0;
    for (int k = 0; k < h; ++k) {
      sum += edges.left[k];
    }
    value = round2(sum, floor_log2(h));
  } else if (edges.have_above) {
    int sum = 0;
    for (int k = 0; k < w; ++k) {
      sum += edges.above[k];
    }
    value = round2(sum, floor_log2(w));
  } else {
    value = 1 << (edges.bit_depth - 1);
  }

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      dst[static_cast<std::size_t>(i) * dst_stride +
          static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
    }
  }
}

void predict_smooth(const IntraEdges& edges, Mode mode, std::uint8_t* dst,
                    std::size_t dst_stride) noexcept {
  const int w = edges.w;
  const int h = edges.h;
  const int log2_w = floor_log2(w);
  const int log2_h = floor_log2(h);
  const int scale = round2(log2_w + log2_h - 4, 2);
  const int bottom_left = edges.left[h];
  const int top_right = edges.above[w];

  for (int i = 0; i < h; ++i) {
    const int s_top =
        kBlendWeightMax >> std::min(6, (i << 1) >> scale);
    for (int j = 0; j < w; ++j) {
      const int s_left =
          kBlendWeightMax >> std::min(6, (j << 1) >> scale);
      const int top = edges.above[j];
      const int left = edges.left[i];

      const int pred_h =
          top_right + round2((left - top_right) * (w - 1 - j), log2_w);
      const int pred_v =
          bottom_left + round2((top - bottom_left) * (h - 1 - i), log2_h);
      const int pred_h2 = pred_h + round2((left - pred_h) * s_left, 6);
      const int pred_v2 = pred_v + round2((top - pred_v) * s_top, 6);

      int value = 0;
      if (mode == Mode::kSmoothH) {
        value = pred_h2;
      } else if (mode == Mode::kSmoothV) {
        value = pred_v2;
      } else {
        value = round2(pred_v2 + pred_h2, 1);
      }
      dst[static_cast<std::size_t>(i) * dst_stride +
          static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
    }
  }
}

void predict_paeth(const IntraEdges& edges, std::uint8_t* dst,
                   std::size_t dst_stride) noexcept {
  const int w = edges.w;
  const int h = edges.h;
  const int corner = edges.above[-1];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int base = edges.above[j] + edges.left[i] - corner;
      const int p_left = std::abs(base - edges.left[i]);
      const int p_top = std::abs(base - edges.above[j]);
      const int p_top_left = std::abs(base - corner);
      int value = 0;
      if (p_left <= p_top && p_left <= p_top_left) {
        value = edges.left[i];
      } else if (p_top <= p_top_left) {
        value = edges.above[j];
      } else {
        value = corner;
      }
      dst[static_cast<std::size_t>(i) * dst_stride +
          static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public prediction entry points
// ---------------------------------------------------------------------------

IntraStatus predict_intra(const IntraEdges& edges, const PredictParams& params,
                          std::uint8_t* dst,
                          std::size_t dst_stride) noexcept {
  if (dst == nullptr) {
    return IntraStatus::kInvalidArgument;
  }
  if (!valid_block_size(edges.w, edges.h)) {
    return IntraStatus::kInvalidArgument;
  }
  if (edges.bit_depth < 8 || edges.bit_depth > 12) {
    return IntraStatus::kInvalidArgument;
  }
  if (!valid_mode(params.mode)) {
    return IntraStatus::kInvalidArgument;
  }
  if (params.angle_delta < -kMaxAngleDelta ||
      params.angle_delta > kMaxAngleDelta) {
    return IntraStatus::kInvalidArgument;
  }
  if (!is_directional(params.mode) && params.angle_delta != 0) {
    return IntraStatus::kInvalidArgument;
  }
  if (edges.valid_above_px < 0 || edges.valid_above_px > edges.w ||
      edges.valid_left_px < 0 || edges.valid_left_px > edges.h) {
    return IntraStatus::kInvalidArgument;
  }
  if (dst_stride < static_cast<std::size_t>(edges.w)) {
    return IntraStatus::kInvalidArgument;
  }

  if (!is_directional(params.mode)) {
    switch (params.mode) {
      case Mode::kDc:
        predict_dc(edges, dst, dst_stride);
        return IntraStatus::kOk;
      case Mode::kSmooth:
      case Mode::kSmoothV:
      case Mode::kSmoothH:
        predict_smooth(edges, params.mode, dst, dst_stride);
        return IntraStatus::kOk;
      case Mode::kPaeth:
        predict_paeth(edges, dst, dst_stride);
        return IntraStatus::kOk;
      default:
        break;
    }
    return IntraStatus::kInvalidArgument;
  }

  const MappedAngle mapped =
      map_angle(params.mode, params.angle_delta, edges.w, edges.h);
  if (mapped.p_angle <= 0 || mapped.p_angle >= 270) {
    return IntraStatus::kInvalidArgument;
  }

  // The spec mutates AboveRow/LeftCol (corner + edge filtering + directional
  // padding). Work on copies so an IntraEdges object is reusable for RDO.
  EdgeRow above = edges.above;
  EdgeRow left = edges.left;

  const int p_angle = mapped.p_angle;
  if (params.enable_edge_filter && p_angle != 90 && p_angle != 180) {
    const bool filter_type =
        (edges.have_above && params.above_neighbor_is_smooth) ||
        (edges.have_left && params.left_neighbor_is_smooth);

    // §7.13.2.14 filter corner process.
    if (p_angle > 90 && p_angle < 180 && edges.w + edges.h >= 24) {
      filter_corner(above, left);
    }

    // §7.13.2.7 ordered steps: above first, then left.
    if (edges.have_above) {
      const int strength =
          edge_filter_strength(edges.w, edges.h, filter_type, p_angle - 90);
      const int num_px = std::min(edges.w, edges.valid_above_px) +
                         (p_angle < 90 ? edges.h : 0) + 1;
      intra_edge_filter(above, num_px, strength);
    }
    if (edges.have_left) {
      const int strength =
          edge_filter_strength(edges.w, edges.h, filter_type, p_angle - 180);
      const int num_px = std::min(edges.h, edges.valid_left_px) +
                         (p_angle > 180 ? edges.w : 0) + 1;
      intra_edge_filter(left, num_px, strength);
    }
  }

  predict_directional(edges, mapped, above, left, dst, dst_stride);
  return IntraStatus::kOk;
}

IntraStatus predict_block(const NeighborRequest& request, int bit_depth,
                          const PredictParams& params, std::uint8_t* dst,
                          std::size_t dst_stride) noexcept {
  IntraEdges edges;
  const IntraStatus status = prepare_edges(request, bit_depth, edges);
  if (status != IntraStatus::kOk) {
    return status;
  }
  return predict_intra(edges, params, dst, dst_stride);
}

// ---------------------------------------------------------------------------
// Cost hooks
// ---------------------------------------------------------------------------

PredictionCost prediction_cost(const std::uint8_t* src, std::size_t src_stride,
                               const std::uint8_t* pred,
                               std::size_t pred_stride, int w,
                               int h) noexcept {
  PredictionCost cost;
  if (src == nullptr || pred == nullptr || w <= 0 || h <= 0) {
    return cost;
  }
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int diff =
          static_cast<int>(src[static_cast<std::size_t>(i) * src_stride +
                               static_cast<std::size_t>(j)]) -
          static_cast<int>(pred[static_cast<std::size_t>(i) * pred_stride +
                                static_cast<std::size_t>(j)]);
      const int magnitude = std::abs(diff);
      cost.sad += static_cast<std::uint64_t>(magnitude);
      cost.sse += static_cast<std::uint64_t>(magnitude * magnitude);
    }
  }
  return cost;
}

IntraStatus predict_and_cost(const IntraEdges& edges,
                             const PredictParams& params,
                             std::span<std::uint8_t> pred_scratch,
                             const std::uint8_t* src, std::size_t src_stride,
                             PredictionCost& cost) noexcept {
  if (src == nullptr) {
    return IntraStatus::kInvalidArgument;
  }
  const std::size_t samples =
      static_cast<std::size_t>(edges.w) * static_cast<std::size_t>(edges.h);
  if (edges.w <= 0 || edges.h <= 0 || pred_scratch.size() < samples) {
    return IntraStatus::kInvalidArgument;
  }

  const IntraStatus status = predict_intra(
      edges, params, pred_scratch.data(), static_cast<std::size_t>(edges.w));
  if (status != IntraStatus::kOk) {
    return status;
  }
  cost = prediction_cost(src, src_stride, pred_scratch.data(),
                         static_cast<std::size_t>(edges.w), edges.w, edges.h);
  return IntraStatus::kOk;
}

int build_mode_candidates(std::span<ModeCandidate> out,
                          bool include_angle_deltas) noexcept {
  const int required =
      kNumModes + (include_angle_deltas
                       ? kNumDirectionalModes * 2 * kMaxAngleDelta
                       : 0);
  if (out.size() < static_cast<std::size_t>(required)) {
    return -1;
  }
  int count = 0;
  for (int m = 0; m < kNumModes; ++m) {
    out[static_cast<std::size_t>(count)].mode = static_cast<Mode>(m);
    out[static_cast<std::size_t>(count)].angle_delta = 0;
    ++count;
  }
  if (include_angle_deltas) {
    for (int m = static_cast<int>(Mode::kV);
         m <= static_cast<int>(Mode::kD67); ++m) {
      for (int delta = -kMaxAngleDelta; delta <= kMaxAngleDelta; ++delta) {
        if (delta == 0) {
          continue;
        }
        out[static_cast<std::size_t>(count)].mode =
            static_cast<Mode>(m);
        out[static_cast<std::size_t>(count)].angle_delta =
            static_cast<std::int8_t>(delta);
        ++count;
      }
    }
  }
  return count;
}

}  // namespace av2enc::intra
