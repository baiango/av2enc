// CUDA transform-variant kernels (DCT/ADST/IDTX/FLIPADST, square and
// rectangular, coded 4x4 through 64x64).
//
// The CPU reference is `src/encoder/tx_variants/`; this layer is a bit-exact
// port of it.  Arithmetic fidelity rules are the same as `src/cuda/transform.*`:
//
//   * binary64 forward accumulation with `__dadd_rn`/`__dmul_rn`, same loop
//     order, and `-fmad=false` in the component build;
//   * host-computed `scale_w`, `scale_h`, `rect_prescale` and `quant_factor`
//     (no device transcendental, no re-derived expressions);
//   * integer §7.15.2.1 / §7.15.2.3 / §7.15.4.1 inverse with int64
//     accumulation and the normative clips;
//   * `lround` quantization and the 528/3-bit dequantization from the CPU
//     reference.
//
// Every entry point is pure: device pointers + `cudaStream_t` in, a
// `cudaError_t` out.  Nothing is allocated, freed or synchronized.  `nblocks`
// blocks are processed per launch; block b starts at element
// `b * coded_w * coded_h` (outputs) / `b * w * h` (adjusted inputs).
//
// The `_multi` entry points read one TxParams per block from a device array, so
// an RDO candidate batch can mix sizes and transform types in one launch.  The
// caller owns and uploads that array.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "codec/transform/variants/tx_variants.h"

namespace av2enc::cuda {

using av2enc::txv::TxParams;

// One mixed-batch job: parameters plus explicit element offsets.  Offsets are
// required because AV2 transform blocks have different element counts; a
// uniform `blockIdx.x * elems` stride cannot represent a mixed-size batch.
struct TxBlockJob {
  TxParams params;
  // Element offset into the adjusted-size buffers (residual in, coeff/levels
  // out, or levels in for the inverse).
  std::int32_t in_offset = 0;
  // Element offset into the coded-size residual buffer (inverse output only;
  // forward jobs may leave this equal to in_offset).
  std::int32_t out_offset = 0;
};

// residual_adjusted (int16, w*h) -> coeff (double, w*h).
cudaError_t txv_forward_block(const std::int16_t* d_residual_adj,
                              double* d_coeff, const TxParams& p, int nblocks,
                              cudaStream_t stream);

// coeff (double) -> levels (int32).
cudaError_t txv_quantize(const double* d_coeff, std::int32_t* d_levels,
                         const TxParams& p, int nblocks, cudaStream_t stream);

// levels (int32) -> dequant (int32).
cudaError_t txv_dequantize(const std::int32_t* d_levels,
                           std::int32_t* d_dequant, const TxParams& p,
                           int nblocks, cudaStream_t stream);

// dequant (int32, w*h) -> residual_coded (int16, coded_w*coded_h).
cudaError_t txv_inverse_block(const std::int32_t* d_dequant,
                              std::int16_t* d_residual_coded,
                              const TxParams& p, int nblocks,
                              cudaStream_t stream);

// Fused forward (coeff + levels) and inverse (levels -> residual_coded).
cudaError_t txv_forward_transform(const std::int16_t* d_residual_adj,
                                  double* d_coeff_scratch,
                                  std::int32_t* d_levels, const TxParams& p,
                                  int nblocks, cudaStream_t stream);
cudaError_t txv_inverse_transform(const std::int32_t* d_levels,
                                  std::int16_t* d_residual_coded,
                                  const TxParams& p, int nblocks,
                                  cudaStream_t stream);

// Per-job parameters and offsets (device array of nblocks TxBlockJob).
cudaError_t txv_forward_block_multi(const std::int16_t* d_residual_adj,
                                    double* d_coeff,
                                    const TxBlockJob* d_jobs, int nblocks,
                                    cudaStream_t stream);
cudaError_t txv_forward_transform_multi(const std::int16_t* d_residual_adj,
                                        double* d_coeff,
                                        std::int32_t* d_levels,
                                        const TxBlockJob* d_jobs, int nblocks,
                                        cudaStream_t stream);
cudaError_t txv_inverse_block_multi(const std::int32_t* d_dequant,
                                    std::int16_t* d_residual_coded,
                                    const TxBlockJob* d_jobs, int nblocks,
                                    cudaStream_t stream);
cudaError_t txv_inverse_transform_multi(const std::int32_t* d_levels,
                                        std::int16_t* d_residual_coded,
                                        const TxBlockJob* d_jobs, int nblocks,
                                        cudaStream_t stream);

// Coded residual (coded_w*coded_h) -> adjusted residual (w*h) by the same
// Round2 averaging as the CPU reference.
cudaError_t txv_downsample_residual(const std::int16_t* d_residual_coded,
                                    std::int16_t* d_residual_adj,
                                    const TxParams& p, int nblocks,
                                    cudaStream_t stream);

// Adjusted residual -> coded residual by §7.15.4 sample duplication.
cudaError_t txv_upsample_residual(const std::int16_t* d_residual_adj,
                                  std::int16_t* d_residual_coded,
                                  const TxParams& p, int nblocks,
                                  cudaStream_t stream);

}  // namespace av2enc::cuda
