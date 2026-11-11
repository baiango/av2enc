// CPU reference for the AV2 10-bit (high-bit-depth) M1 transform path.
//
// This is the oracle for `hbd_tx.cu` (GPU).  It mirrors
// `TileEncoder::compute_tx()` in `src/encoder/m1/m1_still.cpp` and the
// bit-exact 8-bit reference in `src/encoder/tx_variants/`, with the normative
// bit-depth-dependent bounds from AV2 v1.0.0:
//
//   §4.8       Clip1(x) = Clip3(0, (1 << BitDepth) - 1, x)
//   §7.14.3    Dequant = Clip3(-(1 << (7 + BitDepth)), (1 << (7 + BitDepth)) - 1, dq)
//   §7.15.2.1  inverse row clip    ±(1 << (BitDepth + 7))
//              inverse column clip ±(1 << BitDepth)
//   §7.13.2.10 DC predictor neutral value = 1 << (BitDepth - 1)
//
// The forward transform / quantization are bit-depth independent: they reuse
// `txv::downsample_residual()` and `txv::forward_transform()` so the 10-bit
// path cannot drift from the 8-bit reference.  Only dequantization, the
// inverse clips, and reconstruction are depth-parameterized.
//
// The scan order is the §7.15 coefficient scan used by `m1_syntax::build_scan()`
// (anti-diagonals in increasing order, descending rows).
#pragma once

#include <cstdint>
#include <vector>

#include "encoder/tx_variants/tx_variants.h"

namespace av2enc::hbd {

using av2enc::txv::TxParams;
using av2enc::txv::TxType;

// ---------------------------------------------------------------------------
// Bit-depth bounds (mirrors fmt::bit_depth_limits() for CUDA/host asymmetry)
// ---------------------------------------------------------------------------

int dc_pred_bias(int bit_depth);
int clip1(int value, int bit_depth);
int clip_dequant(std::int64_t value, int bit_depth);
int clip_inverse_row(std::int64_t value, int bit_depth);
int clip_inverse_col(std::int64_t value, int bit_depth);

// ---------------------------------------------------------------------------
// Scan / EOB
// ---------------------------------------------------------------------------

// Coefficient scan position `c` -> raster index.  Same order as
// `m1_syntax::build_scan()` and the closed form in `m1_tx.cu`.
int scan_pos(int c, int w, int h);

// Fills scan[0 .. w*h-1] with raster indices in scan order.
void build_scan(int w, int h, std::uint16_t* scan);

// Number of coefficients up to and including the last non-zero level, in scan
// order.  levels[] is indexed by raster position at the adjusted TX size.
int eob_from_levels(const std::int32_t* levels, int w, int h);

// ---------------------------------------------------------------------------
// Core block stages
// ---------------------------------------------------------------------------

// Coded-size residual -> adjusted-size residual (§7.15.4 downsample, Round2).
// Delegates to txv::downsample_residual().
void downsample(const std::int16_t* residual_coded, const TxParams& p,
                std::int16_t* residual_adjusted);

// Forward transform + quantization (bit-depth independent).
// Delegates to txv::forward_transform().
void forward(const std::int16_t* residual_adjusted, const TxParams& p,
             double* coeff, std::int32_t* levels);

// §7.14.3 dequantization with the BitDepth-dependent clip.
void dequantize(const std::int32_t* levels, const TxParams& p,
                std::int32_t* dequant, int bit_depth);

// §7.15.4.1 inverse transform with the BitDepth-dependent row/column clips
// and the adjusted-size sample duplication.  `residual_coded` must hold
// p.coded_w * p.coded_h entries.
void inverse_block(const std::int32_t* dequant, const TxParams& p,
                   std::int16_t* residual_coded, int bit_depth);

// ---------------------------------------------------------------------------
// Whole-block reference (mirrors TileEncoder::compute_tx)
// ---------------------------------------------------------------------------

struct BlockResult {
  int eob = 0;
  std::vector<std::int32_t> levels;    // p.w * p.h, adjusted size
  std::vector<std::int32_t> dequant;   // p.w * p.h, adjusted size
  std::vector<std::int16_t> residual;  // p.coded_w * p.coded_h
  std::vector<std::uint16_t> recon;    // p.coded_w * p.coded_h, clipped samples
};

// Computes one block from a visible source plane and a packed coded-size
// prediction.  Source reads are clamped to [0, src_width) x [0, src_height)
// exactly like `TileEncoder::source_sample()`.  `src` points at element
// (0, 0) of the plane; `pred` is packed at coded_w.
bool encode_block(const std::uint16_t* src, int src_stride, int src_width,
                  int src_height, int x0, int y0, const std::uint16_t* pred,
                  const TxParams& p, int bit_depth, BlockResult* out);

// Same, but the coded-size residual is supplied directly (the RDO / stage-1
// shape).  `pred` still supplies the reconstruction prediction.
bool encode_block_residual(const std::int16_t* residual_coded,
                           const std::uint16_t* pred, const TxParams& p,
                           int bit_depth, BlockResult* out);

// recon[i] = clip1(pred[i] + residual[i], bit_depth), coded size.
void reconstruct(const std::int16_t* residual, const std::uint16_t* pred,
                 int count, int bit_depth, std::uint16_t* recon);

}  // namespace av2enc::hbd
