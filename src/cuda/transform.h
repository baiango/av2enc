// CUDA forward/inverse transform and quantization for the av2enc M0 path.
//
// The arithmetic is a bit-exact port of the CPU reference in
// src/encoder/basic_encoder.cpp (TileEncoder::forward_dct, TileEncoder::inverse_dct
// and the transform/quantization block of encode_residual_plane):
//
//   * forward DCT: binary64 accumulation, same loop order, no FMA contraction
//   * quantization: (int)std::lround(coeff * quant_factor), with the same
//     double expression for quant_factor and dq_denom
//   * dequantization: (abs(q) * 528) & 0xFFFFFF, Round2(., 3), / dq_denom
//   * inverse DCT: normative integer transform, int64 accumulation, Round2
//   * 64x64 luma: Adjusted_Tx_Size = 32x32, residual built by 2x2 averaging
//     and reconstruction by 2x2 duplication
//
// Every function is pure: device pointers in, cudaStream_t out. Nothing is
// allocated, freed, or synchronized, and there is no mutable global state.
// Errors are reported through the cudaError_t return value (parameter
// validation or launch failure).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

namespace av2enc::cuda {

// Parameters of one transform block type, mirroring the host arithmetic in
// encode_residual_plane() lines 1251-1277 of src/encoder/basic_encoder.cpp.
struct TxParams {
  int w = 0;              // adjusted transform width  (32 for 64x64 luma)
  int h = 0;              // adjusted transform height (32 for 64x64 luma)
  int dq_denom = 1;       // 1 << ((tx_pels > 256) + (tx_pels > 1024))
  int row_shift = 0;      // from transform_shifts()
  int col_shift = 0;      // from transform_shifts()
  double scale_w = 0.0;   // 1.0 / (64.0 * sqrt(w))
  double scale_h = 0.0;   // 1.0 / (64.0 * sqrt(h))
  double quant_factor = 0.0;  // 8.0 * dq_denom / (inv_scale * 528)
};

// Builds the parameters for a coded block of `coded_w` x `coded_h` pixels.
// `luma` selects the M0 64x64 -> adjusted 32x32 rule; chroma blocks never
// upscale. Returns std::nullopt for a size the CPU reference cannot transform
// (only 64x64 is adjusted; 64x32/32x64 have no reference implementation).
std::optional<TxParams> make_tx_params(int coded_w, int coded_h, bool luma);

// True for an adjusted transform size handled by the kernels: any combination
// of {4, 8, 16, 32} x {4, 8, 16, 32}.
bool tx_size_supported(int w, int h);

// ---------------------------------------------------------------------------
// Stage kernels. `nblocks` independent blocks are processed per call; block b
// starts at element b * w * h. All pointers are device pointers.
// ---------------------------------------------------------------------------

// residual (int16, [-255, 255]) -> coefficients (double, w*h).
cudaError_t forward_dct(const std::int16_t* d_residual, double* d_coeff,
                        const TxParams& p, int nblocks, cudaStream_t stream);

// coefficients (double) -> quantized levels (int32).
cudaError_t quantize(const double* d_coeff, std::int32_t* d_levels,
                     const TxParams& p, int nblocks, cudaStream_t stream);

// levels (int32) -> dequantized coefficients (int32).
cudaError_t dequantize(const std::int32_t* d_levels, std::int32_t* d_dequant,
                       const TxParams& p, int nblocks, cudaStream_t stream);

// dequantized coefficients (int32) -> reconstructed residual (int16).
cudaError_t inverse_dct(const std::int32_t* d_dequant,
                        std::int16_t* d_residual, const TxParams& p,
                        int nblocks, cudaStream_t stream);

// Convenience compositions. The caller owns the intermediate scratch buffers:
//   forward_transform: d_coeff_scratch (double, nblocks*w*h)
//   inverse_transform: d_dequant_scratch (int32, nblocks*w*h)
cudaError_t forward_transform(const std::int16_t* d_residual,
                              double* d_coeff_scratch,
                              std::int32_t* d_levels, const TxParams& p,
                              int nblocks, cudaStream_t stream);
cudaError_t inverse_transform(const std::int32_t* d_levels,
                              std::int32_t* d_dequant_scratch,
                              std::int16_t* d_residual, const TxParams& p,
                              int nblocks, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 64x64 luma edge block helpers (Adjusted_Tx_Size = TX_32X32).
// ---------------------------------------------------------------------------

// Builds the adjusted 32x32 residual for a 64x64 luma block at (x0, y0):
//   residual[j][i] = Round2(sum_{dy,dx} (source(x0+2i+dx, y0+2j+dy) - pred), 2)
// Source coordinates are clipped to [0, src_width) x [0, src_height), as in
// TileEncoder::source_sample(). d_residual is 32*32 int16.
cudaError_t make_residual_64x64(const std::uint8_t* d_src, int src_stride,
                                int src_width, int src_height, int x0, int y0,
                                int pred, std::int16_t* d_residual,
                                cudaStream_t stream);

// Down-samples an already computed 64x64 residual (int16, 64*64) to the
// adjusted 32x32 residual by the same 2x2 averaging, for callers that build
// the full residual first. Equivalent to make_residual_64x64 when
// residual64 = source - pred with clipped source coordinates.
cudaError_t downsample_residual_64x64(const std::int16_t* d_residual64,
                                      std::int16_t* d_residual32,
                                      cudaStream_t stream);

// Reconstructs a 64x64 luma block from the adjusted 32x32 residual:
//   value = clip1(pred + residual[j][i])
//   dst[x0 + 2i + dx][y0 + 2j + dy] = value
// Writes outside [0, dst_width) x [0, dst_height) are skipped.
cudaError_t reconstruct_64x64(const std::int16_t* d_residual, int pred,
                              std::uint8_t* d_dst, int dst_stride,
                              int dst_width, int dst_height, int x0, int y0,
                              cudaStream_t stream);

}  // namespace av2enc::cuda
