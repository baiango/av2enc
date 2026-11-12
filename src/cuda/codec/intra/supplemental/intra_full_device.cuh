// Shared device bodies for the CUDA intra-full component.
//
// This header is the single device implementation.  `intra_full.cu` compiles it
// into kernels; `intra_full_emulation_test.cpp` compiles the same bytes with the
// CUDA keywords stubbed and one emulated thread, so the host and device paths
// cannot drift.
//
// Threading contract
// ------------------
//   * intra_full_dc_subsampled_dev()      any block size, `dst` rows strided
//   * intra_full_filter_corner_*_dev()    scalar, callable from one thread
//   * intra_full_forward_block_dev()      blockDim.x threads, __syncthreads()
//   * intra_full_inverse_block_dev()      blockDim.x threads, __syncthreads()
//   * intra_full_derive_chroma_type_dev() scalar
//
// The transform bodies are a transcription of
// `src/encoder/tx_variants/tx_variants.cpp` (`forward_block` /
// `inverse_block`), with `__dadd_rn` / `__dmul_rn` instead of contracted
// expressions.  `tmp` / `inter` are caller scratch of at least w*h elements.
//
// Everything is integer except the binary64 forward transform, where the exact
// accumulation order and the pre-computed TxParams scales are the bit-exactness
// contract.
#ifndef AV2ENC_CUDA_INTRA_FULL_INTRA_FULL_DEVICE_CUH_
#define AV2ENC_CUDA_INTRA_FULL_INTRA_FULL_DEVICE_CUH_

#include "intra_full.h"

#include <cmath>
#include <cstdint>

#include "codec/transform/variants/tx_tables_dev.cuh"

namespace av2enc::cuda::intra_full {
namespace {

// ---------------------------------------------------------------------------
// §7.13.3.22 Div_Lut (DIV_LUT_NUM = 129), same table as the CPU reference.
// ---------------------------------------------------------------------------

static __device__ const int kDivLutDev[129] = {
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

// ---------------------------------------------------------------------------
// §9.2 / §5.20.7.29 tables needed by the chroma derivation.
// ---------------------------------------------------------------------------

static __device__ const std::int16_t kTxDimsDev[25][2] = {
    {4, 4},   {8, 8},   {16, 16}, {32, 32}, {64, 64}, {4, 8},   {8, 4},
    {8, 16},  {16, 8},  {16, 32}, {32, 16}, {32, 64}, {64, 32}, {4, 16},
    {16, 4},  {8, 32},  {32, 8},  {16, 64}, {64, 16}, {4, 32},  {32, 4},
    {8, 64},  {64, 8},  {4, 64},  {64, 4}};

static __device__ const int kTxSizeSqrDev[25] = {
    0, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3, 0,
    0, 1, 1, 2, 2, 0, 0, 1, 1, 0, 0};

static __device__ const int kTxSizeSqrUpDev[25] = {
    0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2,
    2, 3, 3, 4, 4, 3, 3, 4, 4, 4, 4};

// §5.20.7.29 Tx_Type_In_Set_Intra[7][16].
static __device__ const int kTxTypeInSetIntraDev[7][16] = {
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
    {1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0},
    {1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

// §5.20.7.29 Mode_To_Txfm[14].
static __device__ const int kModeToTxfmDev[14] = {
    0,  // DC_PRED      -> DCT_DCT
    1,  // V_PRED       -> ADST_DCT
    2,  // H_PRED       -> DCT_ADST
    0,  // D45_PRED     -> DCT_DCT
    3,  // D135_PRED    -> ADST_ADST
    1,  // D113_PRED    -> ADST_DCT
    2,  // D157_PRED    -> DCT_ADST
    2,  // D203_PRED    -> DCT_ADST
    1,  // D67_PRED     -> ADST_DCT
    3,  // SMOOTH_PRED  -> ADST_ADST
    1,  // SMOOTH_V     -> ADST_DCT
    2,  // SMOOTH_H     -> DCT_ADST
    3,  // PAETH_PRED   -> ADST_ADST
    0}; // UV_CFL_PRED  -> DCT_DCT

// §5.20.7.29 Mode_To_Angle[13].
static __device__ const int kModeToAngleDev[13] = {0,  90, 180, 45, 135, 113, 157,
                                                   203, 67, 0,  0,   0,   0};

// §7.15.4.1 get_identity_scale(log2Sz).
static __device__ const int kIdentityScaleDev[7] = {0, 0, 128, 181, 256, 362, 362};

// ---------------------------------------------------------------------------
// Scalar arithmetic
// ---------------------------------------------------------------------------

__device__ __forceinline__ int if_abs(int v) { return v < 0 ? -v : v; }

__device__ __forceinline__ int if_clip3(int low, int high, int v) {
  return v < low ? low : (v > high ? high : v);
}

__device__ __forceinline__ int if_round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

__device__ __forceinline__ int if_floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// §7.13.3.22 resolve_divisor().
__device__ __forceinline__ int if_resolve_divisor(int divisor) {
  constexpr int kDivLutBits = 7;      // DIV_LUT_BITS
  constexpr int kDivLutPrecBits = 9;  // DIV_LUT_PREC_BITS
  (void)kDivLutPrecBits;
  const int magnitude = if_abs(divisor);
  const int n = if_floor_log2(magnitude);
  const int e = magnitude - (1 << n);
  const int f =
      n > kDivLutBits ? if_round2(e, n - kDivLutBits) : (e << (kDivLutBits - n));
  return kDivLutDev[f] * (divisor < 0 ? -1 : 1);
}

__device__ __forceinline__ int if_divisor_shift(int divisor) {
  constexpr int kDivLutPrecBits = 9;
  return if_floor_log2(if_abs(divisor)) + kDivLutPrecBits;
}

// §7.13.2.11 approx_divide().
__device__ __forceinline__ int if_approx_divide(int numerator, int denominator) {
  if (denominator == 0) {
    return 0;
  }
  return if_round2(numerator * if_resolve_divisor(denominator),
                   if_divisor_shift(denominator));
}

// ---------------------------------------------------------------------------
// §7.13.2.11 DC intra, subsampled
// ---------------------------------------------------------------------------

__device__ __forceinline__ int if_dc_step(int extent) {
  return extent > 32 ? 2 : 1;
}

// Computes sum/count/avg exactly like av2enc::still::missing_bits::dc_intra_
// subsampled(), then fills the h x w block with `avg`.
__device__ __forceinline__ void intra_full_dc_subsampled_dev(
    const DcSubsampledDesc& in, std::uint16_t* dst, int dst_stride,
    DcResult* trace) {
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);

  DcResult local;
  int sum = 0;
  int count = 0;
  if (in.have_left != 0) {
    local.step_h = if_dc_step(in.h);
    for (int k = 0; k < in.h; k += local.step_h) {
      sum += static_cast<int>(in.left[k]);
      ++count;
    }
  }
  if (in.have_above != 0) {
    local.step_w = if_dc_step(in.w);
    for (int k = 0; k < in.w; k += local.step_w) {
      sum += static_cast<int>(in.above[k]);
      ++count;
    }
  }

  int avg = 0;
  if (count == 0) {
    avg = 1 << (in.bit_depth - 1);
  } else {
    const int high = (1 << in.bit_depth) - 1;
    const int divided = if_approx_divide(sum, count);
    avg = divided < 0 ? 0 : (divided > high ? high : divided);
  }
  local.sum = sum;
  local.count = count;
  local.avg = avg;

  const std::uint16_t value = static_cast<std::uint16_t>(avg);
  for (int i = 0; i < in.h; ++i) {
    std::uint16_t* row = dst + static_cast<std::ptrdiff_t>(i) * dst_stride;
    for (int j = tid; j < in.w; j += nthr) {
      row[j] = value;
    }
  }
  if (trace != nullptr && tid == 0) {
    *trace = local;
  }
}

// ---------------------------------------------------------------------------
// §7.13.2.14 filter corner
// ---------------------------------------------------------------------------

__device__ __forceinline__ int intra_full_filter_corner_value_dev(
    int left0, int above_m1, int above0) {
  return (left0 * 5 + above_m1 * 6 + above0 * 5 + 8) >> 4;
}

__device__ __forceinline__ int intra_full_filter_corner_applies_dev(
    int w, int h, int p_angle) {
  return (p_angle > 90 && p_angle < 180 && (w + h) >= 24) ? 1 : 0;
}

// `above` = &AboveRow[0], `left` = &LeftCol[0].  Returns 1 when the corner was
// rewritten like the CPU reference's filter_corner_if_applicable().
__device__ __forceinline__ int intra_full_filter_corner_apply_dev(
    int w, int h, int p_angle, std::int16_t* above, std::int16_t* left) {
  if (intra_full_filter_corner_applies_dev(w, h, p_angle) == 0) {
    return 0;
  }
  const int value =
      intra_full_filter_corner_value_dev(static_cast<int>(left[0]),
                                         static_cast<int>(above[-1]),
                                         static_cast<int>(above[0]));
  left[-1] = static_cast<std::int16_t>(value);
  above[-1] = static_cast<std::int16_t>(value);
  return 1;
}

// ---------------------------------------------------------------------------
// §7.15.2 1D kernels for the identity TX_TYPEs
// ---------------------------------------------------------------------------

// Same selection as txv::tx_kernel(): IDT has no matrix; length 32 uses
// Dct_Kernel32 for every type (§7.15.2.1).
__device__ __forceinline__ const std::int16_t* if_tx_kernel(int type, int n) {
  using namespace av2enc::txv::dev;
  switch (type) {
    case 1:  // IDT
      return nullptr;
    case 2:  // ADST
      if (n == 4) return &kAdst4[0][0];
      if (n == 8) return &kAdst8[0][0];
      if (n == 16) return &kAdst16[0][0];
      return &kDct32[0][0];
    case 3:  // FDST / FLIPADST
      if (n == 4) return &kFdst4[0][0];
      if (n == 8) return &kFdst8[0][0];
      if (n == 16) return &kFdst16[0][0];
      return &kDct32[0][0];
    default:  // DCT
      if (n == 4) return &kDct4[0][0];
      if (n == 8) return &kDct8[0][0];
      if (n == 16) return &kDct16[0][0];
      return &kDct32[0][0];
  }
}

// §7.15.4.1 Round2(Dequant * 2896, 12) for an odd log2 difference.
__device__ __forceinline__ int if_prescale(int value, bool odd_rect) {
  if (!odd_rect) {
    return value;
  }
  const int scaled = static_cast<int>(static_cast<long long>(value) * 2896LL);
  return if_round2(scaled, 12);
}

__device__ __forceinline__ int if_identity_scale(int log2_sz) {
  return (log2_sz >= 2 && log2_sz <= 6) ? kIdentityScaleDev[log2_sz] : 362;
}

// residual (int16, w*h) -> coeff (double, w*h).  Bit-exact with
// txv::forward_block(): same loop order, same scale order, same rect prescale.
__device__ __forceinline__ void intra_full_forward_block_dev(
    const std::int16_t* residual, double* coeff,
    const av2enc::txv::TxParams& p, double* tmp) {
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  const int w = p.w;
  const int h = p.h;
  const int elems = w * h;
  const std::int16_t* kw = if_tx_kernel(static_cast<int>(p.row_type), w);
  const std::int16_t* kh = if_tx_kernel(static_cast<int>(p.col_type), h);

  for (int idx = tid; idx < elems; idx += nthr) {
    const int i = idx / w;
    const int v = idx - i * w;
    const int row = i * w;
    if (kw == nullptr) {
      tmp[idx] = static_cast<double>(residual[row + v]);
    } else {
      double sum = 0.0;
      for (int j = 0; j < w; ++j) {
        sum = __dadd_rn(sum,
                        __dmul_rn(static_cast<double>(kw[v * w + j]),
                                  static_cast<double>(residual[row + j])));
      }
      tmp[idx] = __dmul_rn(sum, p.scale_w);
    }
  }
  __syncthreads();

  for (int idx = tid; idx < elems; idx += nthr) {
    const int u = idx / w;
    const int v = idx - u * w;
    double out = 0.0;
    if (kh == nullptr) {
      out = tmp[idx];
    } else {
      double sum = 0.0;
      for (int i = 0; i < h; ++i) {
        sum = __dadd_rn(
            sum, __dmul_rn(static_cast<double>(kh[u * h + i]), tmp[i * w + v]));
      }
      out = __dmul_rn(sum, p.scale_h);
    }
    if (p.rect_prescale != 1.0) {
      out = __dmul_rn(out, p.rect_prescale);
    }
    coeff[idx] = out;
  }
}

// dequant (int32, w*h) -> residual_coded (int16, coded_w*coded_h).  Bit-exact
// with txv::inverse_block(), including §7.15.4 codedsize duplication.
__device__ __forceinline__ void intra_full_inverse_block_dev(
    const std::int32_t* dequant, std::int16_t* residual_coded,
    const av2enc::txv::TxParams& p, std::int32_t* inter) {
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  const int w = p.w;
  const int h = p.h;
  const int elems = w * h;
  const std::int16_t* kw = if_tx_kernel(static_cast<int>(p.row_type), w);
  const std::int16_t* kh = if_tx_kernel(static_cast<int>(p.col_type), h);
  const bool odd_rect = ((p.log2_w ^ p.log2_h) & 1) != 0;
  const int identity_w = if_identity_scale(p.log2_w);
  const int identity_h = if_identity_scale(p.log2_h);

  // Row pass.
  for (int idx = tid; idx < elems; idx += nthr) {
    const int i = idx / w;
    const int oj = idx - i * w;
    const int row = i * w;
    int value = 0;
    if (kw == nullptr) {
      const long long s = static_cast<long long>(
                              if_prescale(dequant[row + oj], odd_rect)) *
                          identity_w;
      value = if_clip3(-(1 << 15), (1 << 15) - 1,
                       if_round2(static_cast<int>(s), p.row_shift));
    } else {
      long long sum = 0;
      for (int j = 0; j < w; ++j) {
        sum += static_cast<long long>(kw[j * w + oj]) *
               if_prescale(dequant[row + j], odd_rect);
      }
      value = if_clip3(-(1 << 15), (1 << 15) - 1,
                       if_round2(static_cast<int>(sum), p.row_shift));
    }
    inter[idx] = value;
  }
  __syncthreads();

  // Column pass + coded-size duplication.
  for (int idx = tid; idx < elems; idx += nthr) {
    const int oi = idx / w;
    const int j = idx - oi * w;
    int value = 0;
    if (kh == nullptr) {
      const long long s = static_cast<long long>(inter[oi * w + j]) * identity_h;
      value = if_round2(static_cast<int>(s), p.col_shift);
    } else {
      long long sum = 0;
      for (int i = 0; i < h; ++i) {
        sum += static_cast<long long>(kh[i * h + oi]) * inter[i * w + j];
      }
      value = if_round2(static_cast<int>(sum), p.col_shift);
    }
    value = if_clip3(-256, 255, value);
    const std::int16_t out = static_cast<std::int16_t>(value);

    const int sx = p.coded_w / w;
    const int sy = p.coded_h / h;
    for (int dy = 0; dy < sy; ++dy) {
      const int y = (oi * sy + dy) * p.coded_w;
      for (int dx = 0; dx < sx; ++dx) {
        residual_coded[y + j * sx + dx] = out;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// §5.20.7.29 chroma PlaneTxType derivation
// ---------------------------------------------------------------------------

__device__ __forceinline__ int if_tx_size_index(int coded_w, int coded_h) {
  for (int i = 0; i < 25; ++i) {
    if (kTxDimsDev[i][0] == coded_w && kTxDimsDev[i][1] == coded_h) {
      return i;
    }
  }
  return -1;
}

// §7.13.2.7 wide_angle_mapping(), non-IBP (same constants as intra::map_angle).
__device__ __forceinline__ int if_map_uv_mode(int mode, int angle_delta, int w,
                                              int h) {
  if (mode < 1 || mode > 8) {
    return mode;
  }
  constexpr int kThresh2 = 61;
  constexpr int kThresh4 = 73;
  constexpr int kThresh8 = 82;
  constexpr int kThresh16 = 86;
  const int p_angle = kModeToAngleDev[mode] + angle_delta * 3;
  if ((h == 2 * w && p_angle < kThresh2) ||
      (h == 4 * w && p_angle < kThresh4) ||
      (h == 8 * w && p_angle < kThresh8) ||
      (h == 16 * w && p_angle < kThresh16)) {
    return 7;  // D203_PRED
  }
  if ((w == 2 * h && p_angle > 270 - kThresh2) ||
      (w == 4 * h && p_angle > 270 - kThresh4) ||
      (w == 8 * h && p_angle > 270 - kThresh8) ||
      (w == 16 * h && p_angle > 270 - kThresh16)) {
    return 3;  // D45_PRED
  }
  return mode;
}

// §5.20.8.3 get_tx_set() for a chroma block (plane > 0).  Returns 0..6 or -1.
__device__ __forceinline__ int if_chroma_tx_set(int coded_w, int coded_h,
                                                int enable_chroma_dctonly) {
  const int idx = if_tx_size_index(coded_w, coded_h);
  if (idx < 0) {
    return -1;
  }
  const int sqr = kTxSizeSqrDev[idx];
  const int sqr_up = kTxSizeSqrUpDev[idx];
  if (sqr_up > 3) {
    if (sqr >= 3) return 0;  // DCTONLY
    return coded_w > coded_h ? 1 : 2;  // WIDE_64 : HIGH_64
  }
  if (sqr_up == 3 && sqr != 3) {
    return coded_w > coded_h ? 3 : 4;  // WIDE_32 : HIGH_32
  }
  if (sqr_up == 3) {
    return 0;  // DCTONLY (32x32)
  }
  // reducedTxSet = enable_chroma_dctonly ? 1 : 0 for plane > 0.
  return enable_chroma_dctonly != 0 ? 6 : 5;  // INTRA_2 : INTRA_1
}

__device__ __forceinline__ int if_set_allows(int set, int type) {
  if (set < 0 || set > 6 || type < 0 || type > 15) {
    return 0;
  }
  return kTxTypeInSetIntraDev[set][type];
}

// Independent of the CPU planner: derives the effective chroma PlaneTxType.
// Returns 1 on success and fills `out`; 0 for an unsupported size/mode/angle.
__device__ __forceinline__ int intra_full_derive_chroma_type_dev(
    const ChromaQuery& q, ChromaResult* out) {
  ChromaResult local;
  local.tx_type = 0;
  local.set = 0;
  local.valid = 0;
  if (out == nullptr) {
    return 0;
  }
  const int set = if_chroma_tx_set(q.coded_w, q.coded_h,
                                   q.enable_chroma_dctonly);
  if (set < 0 || q.uv_mode < 0 || q.uv_mode > 13 ||
      q.uv_angle_delta < -3 || q.uv_angle_delta > 3) {
    *out = local;
    return 0;
  }
  local.set = set;
  local.valid = 1;
  if (q.enable_chroma_dctonly != 0) {
    local.tx_type = 0;  // DCT_DCT
    *out = local;
    return 1;
  }
  int mapped = q.uv_mode;
  if (q.uv_mode >= 1 && q.uv_mode <= 8) {
    mapped = if_map_uv_mode(q.uv_mode, q.uv_angle_delta, q.coded_w, q.coded_h);
  }
  const int derived = kModeToTxfmDev[mapped];
  local.tx_type = if_set_allows(set, derived) != 0 ? derived : 0;
  *out = local;
  return 1;
}

}  // namespace
}  // namespace av2enc::cuda::intra_full

#endif  // AV2ENC_CUDA_INTRA_FULL_INTRA_FULL_DEVICE_CUH_
