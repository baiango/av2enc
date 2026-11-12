// Shared device bodies for the CUDA chroma-format component.
//
// This header is the single device implementation.  `chroma_formats.cu`
// compiles it into kernels; `chroma_formats_emulation_test.cpp` compiles the
// same bytes with the CUDA keywords stubbed, so the host and device paths
// cannot drift.
//
// Device-code audit rules (see the module README):
//   * every helper below is `__device__`/`__device__ __forceinline__`; none is
//     host-only and none calls a host-only symbol;
//   * the only namespace-scope object is `kDivLutDev`, declared
//     `static __device__ const` (the DCT kernels come from
//     `encoder/tx_variants/tx_tables_dev.cuh`, which uses the same form);
//   * no std:: containers or algorithms are used in device code;
//   * no mutable globals at all.
//
// Arithmetic contract
// -------------------
// The forward transform is binary64 with explicit `__dadd_rn`/`__dmul_rn`
// (no FMA contraction), the same accumulation order as
// `av2enc::chroma::forward_dct`, and host-computed scale/quantizer constants
// carried in `TxShape`.  The inverse is the §7.15.4.1 integer matrix with
// int64 accumulation and the §7.14.3/§7.15.2.1 BitDepth-dependent clips.
#ifndef AV2ENC_CUDA_CHROMA_FORMATS_CHROMA_FORMATS_DEVICE_CUH_
#define AV2ENC_CUDA_CHROMA_FORMATS_CHROMA_FORMATS_DEVICE_CUH_

#include <cmath>
#include <cstdint>

#include "cuda/formats/chroma/chroma_formats.h"
#include "codec/transform/variants/tx_tables_dev.cuh"

namespace av2enc::cuda::chroma {
namespace detail {

// ---------------------------------------------------------------------------
// Arithmetic helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ int clip3_dev(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ int clip1_dev(int value, int bit_depth) {
  return clip3_dev(0, (1 << bit_depth) - 1, value);
}

// §4.8 Round2: n <= 0 is a pass-through (matches chroma::round2).
__device__ __forceinline__ int round2_dev(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

__device__ __forceinline__ int round2_64_dev(std::int64_t value, int n) {
  if (n <= 0) {
    return static_cast<int>(value);
  }
  return static_cast<int>((value +
                           (static_cast<std::int64_t>(1) << (n - 1))) >>
                          n);
}

__device__ __forceinline__ int floor_log2_dev(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// Explicit binary64 rounding on device, plain (separately compiled)
// operations on the host-emulation path.  The emulation test is built with
// -ffp-contract=off so the two are the same arithmetic.
__device__ __forceinline__ double fp_add(double a, double b) {
#if defined(__CUDA_ARCH__)
  return __dadd_rn(a, b);
#else
  return a + b;
#endif
}

__device__ __forceinline__ double fp_mul(double a, double b) {
#if defined(__CUDA_ARCH__)
  return __dmul_rn(a, b);
#else
  return a * b;
#endif
}

__device__ __forceinline__ int lround_dev(double value) {
#if defined(__CUDA_ARCH__)
  return static_cast<int>(::lround(value));
#else
  return static_cast<int>(std::lround(value));
#endif
}

// ---------------------------------------------------------------------------
// §7.13.3.22 Div_Lut (DIV_LUT_NUM = 129), same values as the CPU reference.
// `static __device__ const` is the device-safe form (audit rule 2).
// ---------------------------------------------------------------------------

static __device__ const int kDivLutDev[129] = {
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465, 462,
    458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 426, 423, 420, 417,
    415, 412, 410, 407, 405, 402, 400, 397, 395, 392, 390, 388, 386, 383, 381,
    379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358, 356, 354, 352, 350,
    349, 347, 345, 343, 341, 340, 338, 336, 334, 333, 331, 329, 328, 326, 324,
    323, 321, 320, 318, 317, 315, 314, 312, 311, 309, 308, 306, 305, 303, 302,
    301, 299, 298, 297, 295, 294, 293, 291, 290, 289, 287, 286, 285, 284, 282,
    281, 280, 279, 278, 277, 275, 274, 273, 272, 271, 270, 269, 267, 266, 265,
    264, 263, 262, 261, 260, 259, 258, 257, 256};

// §7.13.2.10 / §7.13.3.22 approximate division.  The three constants are
// function-local so no namespace-scope host constexpr is referenced from
// device code (audit rule 2).
__device__ __forceinline__ int approx_divide_dev(int numerator,
                                                 int denominator) {
  constexpr int kBits = 7;      // kDivLutBits
  constexpr int kPrecBits = 9;  // kDivLutPrecBits
  constexpr int kNum = 129;     // kDivLutNum
  const int abs_denominator = denominator < 0 ? -denominator : denominator;
  if (abs_denominator == 0) {
    return 0;
  }
  const int n = floor_log2_dev(abs_denominator);
  const int e = abs_denominator - (1 << n);
  const int f = n > kBits ? round2_dev(e, n - kBits) : e << (kBits - n);
  int scale = kDivLutDev[f < kNum ? f : kNum - 1];
  if (denominator < 0) {
    scale = -scale;
  }
  return round2_dev(numerator * scale, n + kPrecBits);
}

// ---------------------------------------------------------------------------
// §7.15.2.1 / §7.14.3 BitDepth-dependent clips
// ---------------------------------------------------------------------------

__device__ __forceinline__ int clip_dequant_dev(int value, int bit_depth) {
  return clip3_dev(-(1 << (7 + bit_depth)), (1 << (7 + bit_depth)) - 1, value);
}

__device__ __forceinline__ int clip_inverse_row_dev(int value, int bit_depth) {
  return clip3_dev(-(1 << (bit_depth + 7)), (1 << (bit_depth + 7)) - 1, value);
}

__device__ __forceinline__ int clip_inverse_col_dev(int value, int bit_depth) {
  return clip3_dev(-(1 << bit_depth), (1 << bit_depth) - 1, value);
}

// ---------------------------------------------------------------------------
// DCT kernels (§9.6).  Length 32 (and therefore every 64-wide/tall adjusted
// size) uses Dct_Kernel32.
// ---------------------------------------------------------------------------

__device__ __forceinline__ const std::int16_t* dct_kernel_dev(int n) {
  if (n == 4) return &av2enc::txv::dev::kDct4[0][0];
  if (n == 8) return &av2enc::txv::dev::kDct8[0][0];
  if (n == 16) return &av2enc::txv::dev::kDct16[0][0];
  return &av2enc::txv::dev::kDct32[0][0];
}

// ---------------------------------------------------------------------------
// Forward DCT (binary64, same loop order as chroma::forward_dct)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void forward_dct_dev(
    const std::int16_t* residual, int width, int height, bool odd_rect,
    double scale_w, double scale_h, double rect_prescale, double* coeff,
    double* tmp) {
  const std::int16_t* kw = dct_kernel_dev(width);
  const std::int16_t* kh = dct_kernel_dev(height);

  for (int i = 0; i < height; ++i) {
    const int row = i * width;
    for (int v = 0; v < width; ++v) {
      double sum = 0.0;
      for (int j = 0; j < width; ++j) {
        sum = fp_add(sum, fp_mul(static_cast<double>(kw[v * width + j]),
                                 static_cast<double>(residual[row + j])));
      }
      tmp[row + v] = fp_mul(sum, scale_w);
    }
  }

  for (int u = 0; u < height; ++u) {
    for (int v = 0; v < width; ++v) {
      double sum = 0.0;
      for (int i = 0; i < height; ++i) {
        sum = fp_add(sum, fp_mul(static_cast<double>(kh[u * height + i]),
                                 tmp[i * width + v]));
      }
      double out = fp_mul(sum, scale_h);
      if (odd_rect) {
        out = fp_mul(out, rect_prescale);
      }
      coeff[u * width + v] = out;
    }
  }
}

// coeff -> levels, `(int)lround(coeff * quant_factor)` like the CPU reference.
__device__ __forceinline__ void quantize_dev(const double* coeff, int count,
                                             double quant_factor,
                                             std::int32_t* levels) {
  for (int i = 0; i < count; ++i) {
    levels[i] = lround_dev(fp_mul(coeff[i], quant_factor));
  }
}

// ---------------------------------------------------------------------------
// Dequant (§7.14.3, BitDepth clip) and inverse DCT (§7.15.2.1)
// ---------------------------------------------------------------------------

__device__ __forceinline__ int dequantize_level_dev(int level, int dq_denom,
                                                    int bit_depth) {
  if (level == 0) {
    return 0;
  }
  const int abs_level = level < 0 ? -level : level;
  const int dq_high = (abs_level * 528) & 0xFFFFFF;
  const int dq = round2_dev(dq_high, 3);
  const int value = dq / dq_denom;
  return clip_dequant_dev(level < 0 ? -value : value, bit_depth);
}

__device__ __forceinline__ void dequantize_dev(const std::int32_t* levels,
                                               int count, int dq_denom,
                                               int bit_depth,
                                               std::int32_t* dequant) {
  for (int i = 0; i < count; ++i) {
    dequant[i] = dequantize_level_dev(levels[i], dq_denom, bit_depth);
  }
}

__device__ __forceinline__ int prescale_dev(int value, bool odd_rect) {
  if (!odd_rect) {
    return value;
  }
  return round2_dev(
      static_cast<int>(static_cast<std::int64_t>(value) * 2896), 12);
}

// Full adjusted-size inverse.  `intermediate` is caller scratch of adj elems.
// The output is int16: the §7.15.2.1 column clip bounds it to
// [-2^BitDepth, 2^BitDepth - 1].  The row intermediate stays int32 because at
// 10-bit it can reach ±(1 << (BitDepth + 7)) = ±131072.
__device__ __forceinline__ void inverse_dct_dev(
    const std::int32_t* dequant, const TxShape& shape, int bit_depth,
    std::int16_t* residual, std::int32_t* intermediate) {
  const int width = shape.adj_w;
  const int height = shape.adj_h;
  const int row_shift = shape.row_shift;
  const int col_shift = shape.col_shift;
  const bool odd_rect = shape.odd_rect;
  const std::int16_t* kw = dct_kernel_dev(width);
  const std::int16_t* kh = dct_kernel_dev(height);

  for (int i = 0; i < height; ++i) {
    const int row = i * width;
    for (int oj = 0; oj < width; ++oj) {
      std::int64_t sum = 0;
      for (int j = 0; j < width; ++j) {
        sum += static_cast<std::int64_t>(kw[j * width + oj]) *
               prescale_dev(dequant[row + j], odd_rect);
      }
      intermediate[row + oj] =
          clip_inverse_row_dev(round2_64_dev(sum, row_shift), bit_depth);
    }
  }

  for (int oi = 0; oi < height; ++oi) {
    const int out_row = oi * width;
    for (int j = 0; j < width; ++j) {
      std::int64_t sum = 0;
      for (int i = 0; i < height; ++i) {
        sum += static_cast<std::int64_t>(kh[i * height + oi]) *
               intermediate[i * width + j];
      }
      residual[out_row + j] = static_cast<std::int16_t>(
          clip_inverse_col_dev(round2_64_dev(sum, col_shift), bit_depth));
    }
  }
}

// §7.15.4 residual duplication: adjusted -> coded (both int16).
__device__ __forceinline__ void duplicate_residual_dev(
    const std::int16_t* adjusted, int adj_w, int adj_h, int upscale_x,
    int upscale_y, std::int16_t* coded) {
  const int coded_w = adj_w * upscale_x;
  for (int j = 0; j < adj_h; ++j) {
    for (int i = 0; i < adj_w; ++i) {
      const std::int16_t value = adjusted[j * adj_w + i];
      for (int dy = 0; dy < upscale_y; ++dy) {
        for (int dx = 0; dx < upscale_x; ++dx) {
          coded[(j * upscale_y + dy) * coded_w + i * upscale_x + dx] = value;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// §7.13.2.1 / §7.13.2.10 DC prediction
// ---------------------------------------------------------------------------

__device__ __forceinline__ int dc_predict_dev(
    const std::uint16_t* recon, int stride, int max_x, int max_y, int x0,
    int y0, int nominal_w, int nominal_h, int have_left, int have_above,
    int bit_depth) {
  int sum_left = 0;
  int sum_above = 0;
  if (have_left != 0) {
    for (int i = 0; i < nominal_h; ++i) {
      const int y = clip3_dev(0, max_y, y0 + i);
      sum_left += static_cast<int>(recon[y * stride + (x0 - 1)]);
    }
  }
  if (have_above != 0) {
    for (int j = 0; j < nominal_w; ++j) {
      const int x = clip3_dev(0, max_x, x0 + j);
      sum_above += static_cast<int>(recon[(y0 - 1) * stride + x]);
    }
  }

  int pred = 1 << (bit_depth - 1);
  if (have_left != 0 && have_above != 0) {
    pred = clip1_dev(approx_divide_dev(sum_left + sum_above,
                                       nominal_w + nominal_h),
                     bit_depth);
  } else if (have_left != 0) {
    pred = round2_dev(sum_left, floor_log2_dev(nominal_h));
  } else if (have_above != 0) {
    pred = round2_dev(sum_above, floor_log2_dev(nominal_w));
  }
  return pred;
}

// ---------------------------------------------------------------------------
// Forward residual (source minus constant prediction, §7.15.4 averaging)
// ---------------------------------------------------------------------------

__device__ __forceinline__ std::int16_t residual_sample_dev(
    const std::uint16_t* src, int stride, int max_x, int max_y, int x0, int y0,
    int pred, const TxShape& shape, int i, int j) {
  int sum = 0;
  for (int dy = 0; dy < shape.upscale_y; ++dy) {
    const int sy = clip3_dev(0, max_y, y0 + j * shape.upscale_y + dy);
    for (int dx = 0; dx < shape.upscale_x; ++dx) {
      const int sx = clip3_dev(0, max_x, x0 + i * shape.upscale_x + dx);
      sum += static_cast<int>(src[sy * stride + sx]) - pred;
    }
  }
  const int shift = floor_log2_dev(shape.upscale_x * shape.upscale_y);
  return static_cast<std::int16_t>(round2_dev(sum, shift));
}

__device__ __forceinline__ void make_residual_dev(
    const std::uint16_t* src, int stride, int max_x, int max_y, int x0, int y0,
    int pred, const TxShape& shape, std::int16_t* residual_adj) {
  for (int j = 0; j < shape.adj_h; ++j) {
    for (int i = 0; i < shape.adj_w; ++i) {
      residual_adj[j * shape.adj_w + i] =
          residual_sample_dev(src, stride, max_x, max_y, x0, y0, pred, shape,
                              i, j);
    }
  }
}

// ---------------------------------------------------------------------------
// Reconstruction (§7.15.4 duplication + §4.8 Clip1)
// ---------------------------------------------------------------------------

__device__ __forceinline__ std::uint16_t reconstruct_value_dev(
    int pred, int residual, int bit_depth) {
  return static_cast<std::uint16_t>(
      clip1_dev(pred + residual, bit_depth));
}

__device__ __forceinline__ void reconstruct_block_dev(
    const std::int16_t* residual_adj, int pred, const TxShape& shape,
    int bit_depth, std::uint16_t* dst, int dst_stride, int max_x, int max_y,
    int x0, int y0) {
  for (int j = 0; j < shape.adj_h; ++j) {
    for (int i = 0; i < shape.adj_w; ++i) {
      const std::uint16_t value = reconstruct_value_dev(
          pred, residual_adj[j * shape.adj_w + i], bit_depth);
      for (int dy = 0; dy < shape.upscale_y; ++dy) {
        const int py = y0 + j * shape.upscale_y + dy;
        if (py < 0 || py > max_y) continue;
        for (int dx = 0; dx < shape.upscale_x; ++dx) {
          const int px = x0 + i * shape.upscale_x + dx;
          if (px < 0 || px > max_x) continue;
          dst[py * dst_stride + px] = value;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Depth-clipping rectangular copy (8/10-bit samples share uint16 storage)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void copy_rect_dev(
    const std::uint16_t* src, int src_stride, std::uint16_t* dst,
    int dst_stride, int width, int height, int bit_depth) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      dst[y * dst_stride + x] = static_cast<std::uint16_t>(
          clip1_dev(static_cast<int>(src[y * src_stride + x]), bit_depth));
    }
  }
}

}  // namespace detail
}  // namespace av2enc::cuda::chroma

#endif  // AV2ENC_CUDA_CHROMA_FORMATS_CHROMA_FORMATS_DEVICE_CUH_
