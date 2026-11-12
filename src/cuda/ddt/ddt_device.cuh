// Bit-exact device body for AV2 inter DDT (Data Dependent Transform).
//
// This header is the single numerical implementation used by
//   * src/cuda/ddt/ddt_gpu.cu            (the real CUDA kernels), and
//   * src/cuda/ddt/ddt_device_emulation_test.cpp
//                                        (host emulation, no GPU required).
// It is also included by ddt_gpu_test.cu for direct 1D device parity checks.
//
// Normative sources (AV2 v1.0.0):
//   * §7.15.4.1 get_transform_1d_type(): a signalled ADST/FDST becomes DDTX/FDDT
//     when enable_inter_ddt && !use_intrabc && is_inter and the *adjusted* 1D
//     length is not 4.
//   * §7.15.2.1 1d inverse transform: the only lengths with a DDT branch are
//     sz == 8 (Ddtx_Kernel8) and sz == 16 (Ddtx_Kernel16).  The sz == 32 branch
//     uses Dct_Kernel32 for every type, so DDTX/FDDT alias DCT there.  sz == 4
//     is excluded by the get_transform_1d_type() predicate.
//   * §9.6 Ddtx_Kernel8 / Ddtx_Kernel16 (pinned in encoder/m2/ddt/ddt_tables.h).
//   * FDDT is Ddtx_KernelN with the sample (second) index mirrored.
//
// The integer arithmetic mirrors encoder/m2/ddt/ddt_transform.cpp exactly:
// int64 accumulation, §4.8 Round2 (arithmetic shift = floor), the row clip
// +-(1 << (BitDepth + 7)) and the column clip +-(1 << BitDepth) - 1, the
// §7.15.4.1 2896/4096 odd-log2 rectangular pre-scale, and the encoder forward
// Q8 hand-off between the row and column stages.
//
// Threading: the *_elem helpers are pure per-element functions.  A kernel runs
// one CUDA block per transform block with one thread per adjusted-size element;
// the emulation test calls the same helpers in a plain host loop.  There is no
// second transcription that can drift.

#ifndef AV2ENC_CUDA_DDT_DDT_DEVICE_CUH_
#define AV2ENC_CUDA_DDT_DDT_DEVICE_CUH_

#include <cstdint>

#include "encoder/tx_variants/tx_tables_dev.cuh"
#include "encoder/tx_variants/tx_variants.h"

namespace av2enc::cuda::ddt {
namespace dev {

using av2enc::txv::TxParams;

// Table 7.1 1D transform types.  DDTX/FDDT are decoder-derived and never
// signalled, but they are valid effective types here.
//
// These are unscoped enumerators, not namespace-scope `constexpr` variables:
// enumerators are pure compile-time constants with no host storage, so a device
// function can use them without nvcc resolving a host-only symbol.
enum : int {
  kTypeDct = 0,
  kTypeIdt = 1,
  kTypeAdst = 2,
  kTypeFdst = 3,
  kTypeDdtx = 4,
  kTypeFddt = 5,
};

enum : int {
  kMaxDim = 32,
  kMaxElems = kMaxDim * kMaxDim,
  kMaxCodedDim = 64,
  kMaxCodedElems = kMaxCodedDim * kMaxCodedDim,
  kForwardFracBits = 8,
};

// ---------------------------------------------------------------------------
// Ddtx_Kernel8 / Ddtx_Kernel16 (§9.6), [coefficient][sample] orientation.
//
// They are mirrored here (rather than read from ddt_tables.h) so the CUDA
// translation unit does not depend on host constexpr array materialisation.
// Both test executables compare all 320 entries against
// av2enc::m2::ddt::kDdtxKernel8 / kDdtxKernel16, so drift fails the build's
// tests rather than silently changing the transform.
// ---------------------------------------------------------------------------
static __device__ const std::int16_t kDdtxKernel8[8][8] = {
    {   4,    7,   15,   33,   65,   98,  106,   80},
    {   6,   14,   36,   77,  100,   45,  -57,  -98},
    {  22,   48,   85,   88,    0,  -86,  -23,   82},
    {  57,   94,   76,  -26,  -73,   34,   54,  -66},
    {  96,   73,  -43,  -69,   55,   20,  -71,   53},
    { 103,  -17,  -80,   56,   15,  -66,   75,  -41},
    {  78,  -79,    7,   56,  -82,   79,  -56,   26},
    {  56,  -96,   98,  -77,   54,  -33,   19,   -6},
};

static __device__ const std::int16_t kDdtxKernel16[16][16] = {
    {  12,   15,   19,   23,   30,   39,   51,   66,   78,   88,   94,   97,   93,   83,   68,   50},
    {  17,   23,   30,   38,   48,   61,   76,   87,   83,   59,   19,  -30,  -73,  -99,  -99,  -76},
    {  37,   49,   60,   69,   75,   75,   61,   29,  -18,  -67,  -96,  -83,  -28,   40,   84,   83},
    {  45,   60,   69,   73,   66,   40,   -8,  -65,  -91,  -57,   21,   86,   81,    8,  -69,  -90},
    {  47,   60,   61,   49,   19,  -29,  -77,  -83,  -16,   75,   93,    3,  -92,  -74,   32,   97},
    {  60,   74,   64,   28,  -31,  -87,  -82,    4,   88,   54,  -55,  -77,   29,   88,    3,  -86},
    {  64,   70,   40,  -19,  -79,  -78,   11,   92,   28,  -85,  -41,   82,   39,  -83,  -37,   83},
    {  82,   73,    3,  -80,  -91,   10,   94,   18,  -84,   -5,   80,  -17,  -70,   47,   55,  -68},
    {  89,   48,  -53,  -96,   -5,   89,   16,  -83,   12,   75,  -51,  -43,   81,  -14,  -75,   67},
    { 100,    9,  -99,  -45,   84,   36,  -81,    4,   73,  -60,  -17,   76,  -55,  -21,   81,  -56},
    {  92,  -35,  -91,   42,   71,  -69,  -22,   85,  -60,  -17,   77,  -70,   11,   56,  -83,   49},
    {  84,  -71,  -46,   88,  -16,  -67,   79,  -22,  -46,   84,  -68,   15,   46,  -83,   82,  -40},
    {  69,  -83,    2,   75,  -78,   18,   50,  -85,   81,  -43,   -6,   53,  -81,   88,  -69,   32},
    {  50,  -79,   47,   14,  -60,   67,  -37,   -6,   49,  -80,   98,  -99,   90,  -71,   48,  -19},
    {  51,  -89,   73,  -17,  -45,   89, -103,   97,  -83,   71,  -56,   44,  -31,   22,  -11,    5},
    {  44,  -95,  124, -126,  108,  -81,   54,  -30,   16,   -6,    1,    3,   -4,    5,   -3,    2},
};

// ---------------------------------------------------------------------------
// Scalar arithmetic (§4.8 Round2, clips).
// ---------------------------------------------------------------------------

__device__ __forceinline__ int log2_dim(int n) {
  switch (n) {
    case 4:
      return 2;
    case 8:
      return 3;
    case 16:
      return 4;
    case 32:
      return 5;
    default:
      return -1;
  }
}

// §7.15.4.1 get_identity_scale(log2Sz).
__device__ __forceinline__ int identity_scale(int log2_n) {
  switch (log2_n) {
    case 2:
      return 128;
    case 3:
      return 181;
    case 4:
      return 256;
    default:
      return 362;
  }
}

// §4.8 Round2(x, n): floor((x + 2^(n-1)) / 2^n), arithmetic shift.
__device__ __forceinline__ long long round2_ll(long long value, int n) {
  if (n <= 0) return value;
  return (value + (1LL << (n - 1))) >> n;
}

__device__ __forceinline__ int clip_round2(long long value, int n, int low,
                                           int high) {
  const long long rounded = round2_ll(value, n);
  if (rounded < low) return low;
  if (rounded > high) return high;
  return static_cast<int>(rounded);
}

// Round-to-nearest, symmetric around zero (encoder-only helper).
__device__ __forceinline__ long long round_div_ll(long long value,
                                                  long long divisor) {
  const long long half = divisor / 2;
  return value >= 0 ? (value + half) / divisor
                    : -((-value + half) / divisor);
}

__device__ __forceinline__ int clip3_int(int low, int high, int v) {
  return v < low ? low : (v > high ? high : v);
}

// §7.15.4.1 row-prescale for odd-log2 rectangles.
__device__ __forceinline__ int prescale_int(int value, bool odd_rect) {
  if (!odd_rect) return value;
  return static_cast<int>(
      round2_ll(static_cast<long long>(value) * 2896, 12));
}

// ---------------------------------------------------------------------------
// Kernel selection (§7.15.2.1 + §9.6).
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool kernel_is_flipped(int type, int n) {
  return type == kTypeFddt && (n == 8 || n == 16);
}

__device__ __forceinline__ const std::int16_t* kernel_for(int type, int n) {
  using namespace av2enc::txv::dev;
  switch (type) {
    case kTypeIdt:
      return nullptr;
    case kTypeDdtx:
    case kTypeFddt:
      // Only 8 and 16 have a Ddtx kernel.  §7.15.2.1's sz == 32 branch uses
      // Dct_Kernel32, and sz == 4 uses Dct_Kernel4, exactly like the CPU
      // effective_kernel() fallback.
      if (n == 8) return &kDdtxKernel8[0][0];
      if (n == 16) return &kDdtxKernel16[0][0];
      if (n == 4) return &kDct4[0][0];
      return &kDct32[0][0];
    case kTypeAdst:
      if (n == 4) return &kAdst4[0][0];
      if (n == 8) return &kAdst8[0][0];
      if (n == 16) return &kAdst16[0][0];
      return &kDct32[0][0];
    case kTypeFdst:
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

// ---------------------------------------------------------------------------
// §7.15.4.1 get_transform_1d_type().
// ---------------------------------------------------------------------------

__device__ __forceinline__ int effective_1d(int signaled, int sz,
                                            bool enable_inter_ddt,
                                            bool is_inter, bool use_intrabc) {
  if (enable_inter_ddt && !use_intrabc && is_inter &&
      (signaled == kTypeAdst || signaled == kTypeFdst) && sz != 4) {
    return signaled == kTypeAdst ? kTypeDdtx : kTypeFddt;
  }
  return signaled;
}

struct Plan {
  int row = kTypeDct;
  int col = kTypeDct;
  bool row_ddt = false;
  bool col_ddt = false;
};

__device__ __forceinline__ Plan make_plan(const TxParams& p,
                                          bool enable_inter_ddt,
                                          bool is_inter, bool use_intrabc) {
  Plan plan;
  plan.row = effective_1d(static_cast<int>(p.row_type), p.w,
                          enable_inter_ddt, is_inter, use_intrabc);
  plan.col = effective_1d(static_cast<int>(p.col_type), p.h,
                          enable_inter_ddt, is_inter, use_intrabc);
  plan.row_ddt = plan.row == kTypeDdtx || plan.row == kTypeFddt;
  plan.col_ddt = plan.col == kTypeDdtx || plan.col == kTypeFddt;
  return plan;
}

// Mirrors ddt::plan_has_ddt(): a DDTX/FDDT derived at adjusted length 32 is
// alias-ed to Dct_Kernel32, so only length 8/16 changes the reconstruction.
__device__ __forceinline__ bool plan_has_ddt(const TxParams& p,
                                             bool enable_inter_ddt,
                                             bool is_inter,
                                             bool use_intrabc) {
  const Plan plan = make_plan(p, enable_inter_ddt, is_inter, use_intrabc);
  const bool row_changes = plan.row_ddt && (p.w == 8 || p.w == 16);
  const bool col_changes = plan.col_ddt && (p.h == 8 || p.h == 16);
  return row_changes || col_changes;
}

// ---------------------------------------------------------------------------
// §7.15.2.1 1d inverse / encoder 1d forward.
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool inverse_1d(const std::int32_t* src, int n,
                                           int shift, bool col_tx, int type,
                                           int bit_depth, std::int32_t* out) {
  if (type == kTypeIdt) return false;  // identity is not a matrix
  if (log2_dim(n) < 0) return false;
  const std::int16_t* kernel = kernel_for(type, n);
  if (kernel == nullptr) return false;
  const bool flipped = kernel_is_flipped(type, n);
  const int clip_bits = bit_depth + (col_tx ? 0 : 7);
  const int low = -(1 << clip_bits);
  const int high = (1 << clip_bits) - 1;
  for (int i = 0; i < n; ++i) {
    const int sample = flipped ? (n - 1 - i) : i;
    long long sum = 0;
    for (int j = 0; j < n; ++j) {
      sum += static_cast<long long>(kernel[j * n + sample]) *
             static_cast<long long>(src[j]);
    }
    out[i] = clip_round2(sum, shift, low, high);
  }
  return true;
}

// Encoder forward with explicit Q-format bookkeeping; this is the per-stage
// body of ddt::forward_stage() (not its 1D wrapper).
__device__ __forceinline__ long long forward_stage_elem(
    const long long* src, int n, int inverse_shift, int type,
    int input_frac_bits, int output_frac_bits, int r) {
  const int log2_n = log2_dim(n);
  const std::int16_t* kernel = kernel_for(type, n);
  const bool flipped = kernel_is_flipped(type, n);
  const int shift =
      12 + log2_n - inverse_shift + input_frac_bits - output_frac_bits;
  long long sum = 0;
  for (int i = 0; i < n; ++i) {
    const int sample = flipped ? (n - 1 - i) : i;
    sum += static_cast<long long>(kernel[r * n + sample]) * src[i];
  }
  if (shift > 0) {
    return (sum + (1LL << (shift - 1))) >> shift;
  }
  if (shift == 0) return sum;
  return sum << (-shift);
}

__device__ __forceinline__ bool forward_1d(const std::int32_t* src, int n,
                                           int inverse_shift, int type,
                                           std::int32_t* out) {
  if (type == kTypeIdt) return false;
  const int log2_n = log2_dim(n);
  if (log2_n < 0) return false;
  const std::int16_t* kernel = kernel_for(type, n);
  if (kernel == nullptr) return false;
  const bool flipped = kernel_is_flipped(type, n);
  const int shift = 12 + log2_n - inverse_shift;
  if (shift < 0) return false;
  for (int r = 0; r < n; ++r) {
    long long sum = 0;
    for (int i = 0; i < n; ++i) {
      const int sample = flipped ? (n - 1 - i) : i;
      sum += static_cast<long long>(kernel[r * n + sample]) *
             static_cast<long long>(src[i]);
    }
    out[r] = clip_round2(sum, shift, -(1 << 30), (1 << 30) - 1);
  }
  return true;
}

// ---------------------------------------------------------------------------
// Encoder forward block: one row element and one column element.
// ---------------------------------------------------------------------------

// Row pass output for adjusted position (row, j); returns the Q8 value that the
// CPU keeps in its int64 `temp` array.
__device__ __forceinline__ long long forward_row_elem(
    const std::int16_t* row, const TxParams& p, int type, int j) {
  if (type == kTypeIdt) {
    return round_div_ll(
        static_cast<long long>(row[j]) << (p.row_shift + kForwardFracBits),
        identity_scale(p.log2_w));
  }
  long long src[kMaxDim];
  for (int k = 0; k < p.w; ++k) src[k] = static_cast<long long>(row[k]);
  return forward_stage_elem(src, p.w, p.row_shift, type, 0,
                            kForwardFracBits, j);
}

// Column pass output (still Q8) for adjusted position (i, j).
__device__ __forceinline__ long long forward_col_elem(
    const long long* temp, int w, const TxParams& p, int type, int i, int j) {
  if (type == kTypeIdt) {
    const long long column = temp[i * w + j];
    return round_div_ll(column << p.col_shift,
                        static_cast<long long>(identity_scale(p.log2_h))
                            << kForwardFracBits);
  }
  long long column[kMaxDim];
  for (int k = 0; k < p.h; ++k) column[k] = temp[k * w + j];
  return forward_stage_elem(column, p.h, p.col_shift, type,
                            kForwardFracBits, 0, i);
}

// Final encoder compensation: CPU narrows to int32 first, then applies
// 4096/2896 for odd-log2 rectangles and clamps to +-(1 << 30).
__device__ __forceinline__ std::int32_t forward_finish(long long value,
                                                       bool odd_rect) {
  long long narrowed = static_cast<std::int32_t>(value);
  if (odd_rect) narrowed = round_div_ll(narrowed * 4096, 2896);
  constexpr long long kLimit = 1LL << 30;
  if (narrowed < -kLimit) return static_cast<std::int32_t>(-kLimit);
  if (narrowed > kLimit) return static_cast<std::int32_t>(kLimit);
  return static_cast<std::int32_t>(narrowed);
}

// ---------------------------------------------------------------------------
// Decoder inverse block: one row element and one column element.
// ---------------------------------------------------------------------------

// Row pass output for adjusted row position (dq_row[0..w-1], output j).
__device__ __forceinline__ int inverse_row_elem(const std::int32_t* dq_row,
                                                const TxParams& p, int type,
                                                int bit_depth, bool odd_rect,
                                                int j) {
  const int low = -(1 << (bit_depth + 7));
  const int high = (1 << (bit_depth + 7)) - 1;
  if (type == kTypeIdt) {
    const long long sum =
        static_cast<long long>(prescale_int(dq_row[j], odd_rect)) *
        identity_scale(p.log2_w);
    return clip_round2(sum, p.row_shift, low, high);
  }
  const int w = p.w;
  const std::int16_t* kernel = kernel_for(type, w);
  const int sample = kernel_is_flipped(type, w) ? (w - 1 - j) : j;
  long long sum = 0;
  for (int k = 0; k < w; ++k) {
    sum += static_cast<long long>(kernel[k * w + sample]) *
           prescale_int(dq_row[k], odd_rect);
  }
  return clip_round2(sum, p.row_shift, low, high);
}

// Column pass output for adjusted position (i, j); the caller casts to int16.
__device__ __forceinline__ int inverse_col_elem(const std::int32_t* inter, int w,
                                                const TxParams& p, int type,
                                                int bit_depth, int i, int j) {
  const int low = -(1 << bit_depth);
  const int high = (1 << bit_depth) - 1;
  if (type == kTypeIdt) {
    const long long sum = static_cast<long long>(inter[i * w + j]) *
                          identity_scale(p.log2_h);
    return clip_round2(sum, p.col_shift, low, high);
  }
  const int h = p.h;
  const std::int16_t* kernel = kernel_for(type, h);
  const int sample = kernel_is_flipped(type, h) ? (h - 1 - i) : i;
  long long sum = 0;
  for (int k = 0; k < h; ++k) {
    sum += static_cast<long long>(kernel[k * h + sample]) * inter[k * w + j];
  }
  return clip_round2(sum, p.col_shift, low, high);
}

// §7.15.4 adjusted-size expansion by sample duplication:
//   coded[y][x] = adjusted[y / sy][x / sx].
__device__ __forceinline__ void store_coded(std::int16_t* coded, int coded_w,
                                            int coded_h, int w, int h, int i,
                                            int j, std::int16_t value) {
  const int sx = coded_w / w;
  const int sy = coded_h / h;
  for (int dy = 0; dy < sy; ++dy) {
    const int row = (i * sy + dy) * coded_w;
    for (int dx = 0; dx < sx; ++dx) {
      coded[row + j * sx + dx] = value;
    }
  }
}

// ---------------------------------------------------------------------------
// Quantiser helpers, identical to the CPU reference.
// ---------------------------------------------------------------------------

__device__ __forceinline__ std::int32_t level_from_coeff(std::int32_t coeff,
                                                         int dq_denom) {
  return static_cast<std::int32_t>(
      round_div_ll(static_cast<long long>(coeff) * dq_denom, 66));
}

// §7.14.4 dequant for the inverse-from-levels path (matches txv::dequantize).
__device__ __forceinline__ std::int32_t dequantize_level(int level,
                                                         int dq_denom) {
  const int abs_qc = level < 0 ? -level : level;
  const int dq_high = (abs_qc * 528) & 0xFFFFFF;
  const int dq = ((dq_high + (1 << 2)) >> 3) / dq_denom;
  return clip3_int(-(1 << 15), (1 << 15) - 1, level < 0 ? -dq : dq);
}

}  // namespace dev
}  // namespace av2enc::cuda::ddt

#endif  // AV2ENC_CUDA_DDT_DDT_DEVICE_CUH_
