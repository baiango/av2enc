// Residual / transform / dequant / reconstruction helpers for the extra chroma
// shapes (4:2:2, 4:4:4, 4:0:0).
//
// The arithmetic is the 8-bit path already validated by M0
// (`src/encoder/basic_encoder.cpp`): the integer DCT kernels in
// `src/encoder/basic_tables.h`, the §7.14 quantiser step 528 (qlookup(90)), the
// §7.15.4 inverse transform, and the 8-bit clip bounds.  What is new here is
// the *shape* layer:
//
//   * a nominal transform size (32x64 for 4:2:2 chroma, 64x64 for 4:4:4
//     chroma) maps through Adjusted_Tx_Size (§9.2) to a 32x32 coefficient grid;
//   * the inverse residual is expanded by sample duplication (§7.15.4), and
//     the encoder's forward residual averages the corresponding source
//     samples so the decoder's duplication is invertible;
//   * txSzCtx (§5.20.7.27) is computed from the nominal size, while EOB/scan
//     use the adjusted size.
//
// No entropy symbols are written here.  `chroma_still.cpp` owns the syntax.
#pragma once

#include <cstdint>
#include <vector>

namespace av2enc::chroma {

// 8-bit coefficient quantiser step: qlookup(base_q_idx = 90) = 528.
inline constexpr int kQuantStep = 528;
inline constexpr int kQuantTableBits = 3;  // QUANT_TABLE_BITS

// Shape of one transform block after Adjusted_Tx_Size and duplication.
struct TxShape {
  int nominal_w = 0;   // Tx_Width[txSz]
  int nominal_h = 0;   // Tx_Height[txSz]
  int adj_w = 0;       // Tx_Width[Adjusted_Tx_Size[txSz]]
  int adj_h = 0;       // Tx_Height[Adjusted_Tx_Size[txSz]]
  int upscale_x = 1;   // nominal_w / adj_w (1 or 2)
  int upscale_y = 1;   // nominal_h / adj_h (1 or 2)
  int tx_size_ctx = 0; // §5.20.7.27 txSzCtx
  int eob_multisize = 0;
  int seg_eob = 0;     // Min(32, nominal_w) * Min(32, nominal_h)
  int tx_pels = 0;     // nominal_w * nominal_h
  int dq_denom = 1;    // 1 << ((tx_pels > 256) + (tx_pels > 1024))
  // Transform_Shift[txSz] (§7.15.4.1) uses the *nominal* tx size, not the
  // adjusted one.  This matters for 32x64: nominal shifts are (6,12), while
  // the adjusted 32x32 square would give (6,13).
  int row_shift = 0;
  int col_shift = 0;
  // Rectangular pre-scale predicate: Abs(log2(nominal_w) - log2(nominal_h))
  // is odd.  Also a nominal-size property, not an adjusted-size one.
  bool odd_rect = false;
};

// Returns false for zero/unsupported nominal sizes.  The minimal still writer
// uses 16x4..64x64; all adjusted sizes are <= 32.
bool make_tx_shape(int nominal_width, int nominal_height, TxShape& shape);

// §8.3.2 coefficient context from an adjusted (<=32) shape.
int tx_size_context(int nominal_width, int nominal_height);

// §7.15.4.1 rectangular pre-scale predicate.
bool odd_rect_transform(int width, int height);

// §4.8 Round2(x, n) = (x + (1 << (n-1))) >> n.
int round2(int value, int n);

// §7.13.2.10 / §7.13.3.22 DC division using Div_Lut.
int approx_divide(int numerator, int denominator);

// Transform_Shift table lookup for a nominal tx size (§7.15.4.1, §9.2).
void transform_shifts(int nominal_width, int nominal_height, int* row_shift,
                      int* col_shift);

// Forward 2D DCT-II through the normative integer kernel.  `residual` and
// `coeff` are adjusted-size (<= 32x32); `odd_rect` selects the §7.15.4.1
// rectangular pre-scale from the nominal size.
void forward_dct(const std::int16_t* residual, int adj_width, int adj_height,
                 bool odd_rect, double* coeff);

// §7.15.4.1 inverse 2D matrix transform.  The matrix runs on the adjusted size
// (adj_width, adj_height <= 32), while the row/column shifts and the
// rectangular pre-scale use the nominal size carried by `shape`.
void inverse_dct(const int* dequant, const TxShape& shape, int* residual);

// §7.16-style diagonal scan over an adjusted transform size.
void build_diagonal_scan(int width, int height, std::vector<std::uint16_t>& scan);

// Quantise the forward coefficients: levels[i] = lround(coeff[i] * factor).
void quantize_coefficients(const double* coeff, int count, double quant_factor,
                           int* levels);

// Dequantise one block (§7.14.3, no TCQ):
//   dq_high = (abs(level) * 528) & 0xFFFFFF
//   dq      = Round2(dq_high, 3) / dq_denom
int dequantize_level(int level, int dq_denom);

// Full dequant of a coefficient array.
void dequantize_block(const int* levels, int count, int dq_denom, int* dequant);

// Forward quantisation factor for a shape, matching M0's arithmetic order.
double quant_factor_for(const TxShape& shape);

}  // namespace av2enc::chroma
