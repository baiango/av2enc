// AV2 transform variants: DCT / ADST / IDTX / FLIPADST, square and
// rectangular, 4x4 through 64x64.
//
// This is the CPU reference half of the M1 "transform variants" component
// (`src/encoder/tx_variants/` + `src/cuda/tx_variants/`).  It is deliberately
// independent of `src/encoder/m0_still.cpp` and `src/cuda/transform.*`: those
// files are owned by other agents and are not modified.
//
// Numeric contract (same style as the existing transform layer):
//
//   * forward 1D transform: binary64 accumulation in the same loop order as
//     `TileEncoder::forward_dct()`, using the §9.6 integer kernel divided by
//     `64 * sqrt(n)` (orthonormal DCT for `n`, and the analogous ADST/FDST
//     normalization).  IDTX forwards the residual unchanged.
//   * quantization: `(int32)lround(coeff * quant_factor)`, with the same
//     `quant_factor` expression as the M0 layer.
//   * dequantization: `(abs(q) * 528) & 0xFFFFFF`, `Round2(., 3)`, `/ dq_denom`.
//   * inverse: the normative §7.15.2.1 / §7.15.2.3 / §7.15.4.1 integer
//     matrices, int64 accumulation, `Round2`, row clip to ±(1<<15), column
//     clip to [-256, 255], then §7.15.4 adjusted-size sample duplication.
//
// Differences from `src/cuda/transform.*`, all intentional and spec-driven:
//
//   * §7.15.4.1's `Round2(Dequant * 2896, 12)` pre-scale is applied when
//     `abs(log2W - log2H)` is odd.  The M0 reference omits it, so rectangular
//     DCT sizes with an odd log2 difference will not match `transform.*`
//     coefficient-for-coefficient; square sizes are identical.
//   * transform shifts come from the *coded* TX size (Transform_Shift[txSz]),
//     not from the adjusted size.  They coincide for every square size and
//     differ for 64-wide/tall adjusted sizes.
//   * adjusted sizes above 32x32 are handled generically (1x or 2x sample
//     duplication), not only the 64x64 2x2 case.
//
// §7.15.2.1 defines the length-32 1D transform with `Dct_Kernel32` only, so a
// 32-length ADST/FDST dimension canonically uses the DCT kernel.  64x64 is
// coded as an adjusted 32x32 transform, so 64x64 ADST/FLIPADST likewise alias
// 64x64 DCT.  This file preserves that normative behavior.
//
// Scope: lossy 8-bit transforms.  Lossless IDTX (§7.15.4) and DPCM cumulative
// sums are not part of this component.
#pragma once

#include <cstdint>
#include <optional>

namespace av2enc::txv {

inline constexpr int kMaxCodedTxDim = 64;
inline constexpr int kMaxAdjTxDim = 32;
inline constexpr int kMaxAdjTxElems = kMaxAdjTxDim * kMaxAdjTxDim;
inline constexpr int kQuantStep = 528;
inline constexpr int kQuantTableBits = 3;  // QUANT_TABLE_BITS

// AV2 TX_TYPE values (constants.json: DCT_DCT=0 .. H_FLIPADST=15).
enum class TxType : std::int8_t {
  kDctDct = 0,
  kAdstDct = 1,
  kDctAdst = 2,
  kAdstAdst = 3,
  kFlipadstDct = 4,
  kDctFlipadst = 5,
  kFlipadstFlipadst = 6,
  kFlipadstAdst = 7,
  kAdstFlipadst = 8,
  kIdtx = 9,
  kVdct = 10,
  kHdct = 11,
  kVadst = 12,
  kHadst = 13,
  kVflipadst = 14,
  kHflipadst = 15,
};

// Table 7.1 1D transform type values.
enum class Tx1dType : std::int8_t {
  kDct = 0,
  kIdt = 1,
  kAdst = 2,
  kFdst = 3,
};

// AV2 TX_SIZE enum indices 0..24 (enum_tx_size §6.19.6.1).
enum class TxSize : std::int8_t {
  k4x4 = 0,
  k8x8 = 1,
  k16x16 = 2,
  k32x32 = 3,
  k64x64 = 4,
  k4x8 = 5,
  k8x4 = 6,
  k8x16 = 7,
  k16x8 = 8,
  k16x32 = 9,
  k32x16 = 10,
  k32x64 = 11,
  k64x32 = 12,
  k4x16 = 13,
  k16x4 = 14,
  k8x32 = 15,
  k32x8 = 16,
  k16x64 = 17,
  k64x16 = 18,
  k4x32 = 19,
  k32x4 = 20,
  k8x64 = 21,
  k64x8 = 22,
  k4x64 = 23,
  k64x4 = 24,
};

// Fully derived parameters for one transform block.
struct TxParams {
  std::int8_t tx_size = 0;    // TxSize index 0..24
  std::int8_t tx_type = 0;    // TxType
  std::int8_t row_type = 0;   // Tx1dType applied across width (dir 0)
  std::int8_t col_type = 0;   // Tx1dType applied down height (dir 1)
  std::int16_t coded_w = 0;   // coded block width  (4..64)
  std::int16_t coded_h = 0;   // coded block height (4..64)
  std::int16_t w = 0;         // adjusted transform width  (4..32)
  std::int16_t h = 0;         // adjusted transform height (4..32)
  std::int8_t log2_w = 0;     // log2(coded_w)
  std::int8_t log2_h = 0;     // log2(coded_h)
  std::int8_t row_shift = 0;  // Transform_Shift[tx_size][0]
  std::int8_t col_shift = 0;  // Transform_Shift[tx_size][1]
  std::uint8_t dq_denom = 1;  // 1 << ((coded pels > 256) + (> 1024))
  double scale_w = 0.0;       // 1 / (64 * sqrt(w))
  double scale_h = 0.0;       // 1 / (64 * sqrt(h))
  // 4096/2896 when abs(log2_w - log2_h) is odd (compensates §7.15.4.1's
  // Round2(Dequant * 2896, 12) decoder pre-scale), otherwise 1.0.
  double rect_prescale = 1.0;
  double quant_factor = 0.0;  // 8*dq_denom / (inv_scale*528)
};

const char* tx_type_name(TxType type);
const char* tx_size_name(TxSize size);

// Returns the TxSize index for a coded (width, height), or -1 when the pair is
// not one of the 25 AV2 transform sizes.
int tx_size_index(int coded_w, int coded_h);

// (row, col) 1D types from a TX_TYPE via Transform_1d_Type.  dir 0 (row) is
// horizontal; dir 1 (col) is vertical.
void tx_1d_types(TxType type, Tx1dType* row, Tx1dType* col);

// Inverse of tx_1d_types; returns false when the pair is not a TX_TYPE.
bool make_tx_type(Tx1dType row, Tx1dType col, TxType* out);

// Kernel for one 1D transform dimension.  n is 4, 8, 16 or 32.  IDT returns
// nullptr (identity is not a matrix).  Length 32 always returns kDct32,
// matching §7.15.2.1's `sz == 32` branch.
const std::int16_t* tx_kernel(Tx1dType type, int n);

// Builds all derived parameters.  Returns nullopt for a size that is not one
// of the 25 AV2 transform sizes (e.g. 64x128 or 2x8).
std::optional<TxParams> make_tx_params(int coded_w, int coded_h,
                                       TxType type = TxType::kDctDct);
std::optional<TxParams> make_tx_params(TxSize size,
                                       TxType type = TxType::kDctDct);

// True when coded (width, height) map to a supported TxSize.
bool tx_size_supported(int coded_w, int coded_h);

// Dimensions of the adjusted transform for a coded size (== coded, capped 32).
void adjusted_size(const TxParams& p, int* w, int* h);

// §4.8 Round2: (value + (1 << (n-1))) >> n, arithmetic shift.  n <= 0 is a
// pass-through (the existing M0 reference does the same).
std::int32_t round2(std::int32_t value, int n);

// ---------------------------------------------------------------------------
// Core block transforms.  Residuals are int16, coefficients double, levels and
// dequantized coefficients int32.  Residuals are tightly packed row-major at
// the *adjusted* size for forward/inverse core calls, except
// inverse_block / inverse_transform which emit a coded-size residual.
// ---------------------------------------------------------------------------

// residual_adjusted[w*h] -> coeff[w*h] (binary64).
void forward_block(const std::int16_t* residual_adjusted, const TxParams& p,
                   double* coeff);

// coeff[w*h] -> levels[w*h].
void quantize(const double* coeff, const TxParams& p, std::int32_t* levels);

// levels[w*h] -> dequant[w*h].
void dequantize(const std::int32_t* levels, const TxParams& p,
                std::int32_t* dequant);

// dequant[w*h] -> residual_coded[coded_w*coded_h] (normative inverse).
void inverse_block(const std::int32_t* dequant, const TxParams& p,
                   std::int16_t* residual_coded);

// forward_block + quantize.  coeff_scratch has w*h elements.
void forward_transform(const std::int16_t* residual_adjusted, const TxParams& p,
                       double* coeff_scratch, std::int32_t* levels);

// dequantize + inverse_block.  dequant_scratch has w*h elements.
void inverse_transform(const std::int32_t* levels, const TxParams& p,
                       std::int32_t* dequant_scratch,
                       std::int16_t* residual_coded);

// ---------------------------------------------------------------------------
// Residual / reconstruction helpers for coded-size blocks whose adjusted size
// is smaller (64x64, 64x32, 32x64, 16x64, ...).
// ---------------------------------------------------------------------------

// Averages a coded-size residual down to the adjusted size:
//   residual_adj[j][i] = Round2(sum over sx*sy samples, log2(sx*sy))
// with sx = coded_w/w, sy = coded_h/h.  A 1x1 scale is a copy.
void downsample_residual(const std::int16_t* residual_coded, const TxParams& p,
                         std::int16_t* residual_adjusted);

// Builds the adjusted residual directly from a source plane and a constant
// prediction, clipping source coordinates to the visible plane.  This mirrors
// the 64x64 edge path in TileEncoder::encode_residual_plane for every
// adjusted size.
void make_residual(const std::uint8_t* src, int src_stride, int src_width,
                   int src_height, int x0, int y0, int pred,
                   const TxParams& p, std::int16_t* residual_adjusted);

// Expands an adjusted residual to the coded size by sample duplication.
void upsample_residual(const std::int16_t* residual_adjusted, const TxParams& p,
                       std::int16_t* residual_coded);

// reconstruct = clip1(pred + residual_adj), duplicated to the coded size and
// written to dst[x0 + x][y0 + y] for coordinates inside the visible plane.
void reconstruct_block(const std::int16_t* residual_adjusted, int pred,
                       const TxParams& p, std::uint8_t* dst, int dst_stride,
                       int dst_width, int dst_height, int x0, int y0);

// ---------------------------------------------------------------------------
// Partition / split membership (no bitstream syntax; these are encoder hooks).
// ---------------------------------------------------------------------------

// AV2 TX partition types used by M1 (§5.20.6.3).  Extended 3/4/5-way
// partitions are out of scope.
enum class TxPartition : std::int8_t {
  kNone = 0,
  kSplit = 1,
  kHorz = 2,
  kVert = 3,
};

const char* tx_partition_name(TxPartition part);

// A legal partition must produce coded sub-blocks whose dimensions are >= 4.
bool tx_partition_legal(int coded_w, int coded_h, TxPartition part);

// Enumerates legal partitions for a block into out[0..cap); returns the count.
int tx_partition_candidates(int coded_w, int coded_h, TxPartition* out,
                            int cap);

// Number of leaves a partition produces (1, 2 or 4), or 0 if illegal.
int tx_partition_parts(TxPartition part);

// Leaf geometry for one partition.  out must hold 4 entries.  Returns the
// count, or 0 if the partition is illegal for the block.
int tx_partition_geometry(int coded_w, int coded_h, TxPartition part,
                          int* xs, int* ys, int* ws, int* hs);

}  // namespace av2enc::txv
