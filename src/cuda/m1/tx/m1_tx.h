// Batched M1 transform / quantize / inverse stage (GPU, stage 1).
//
// One CUDA block per transform block, all blocks of a batch in one launch.
// The stage implements the exact arithmetic of `TileEncoder::compute_tx`
// (`src/encoder/m1/m1_still.cpp`) on top of the bit-exact CPU reference in
// `src/encoder/tx_variants/`:
//
//   coded residual (src - pred, or supplied)
//     -> §7.15.4 downsample to the adjusted transform size      (Round2)
//     -> forward 2D transform (binary64, __dadd_rn/__dmul_rn, -fmad=false)
//     -> quantize with lround(coeff * quant_factor)
//     -> eob over the §7.15 zigzag scan (same order as build_scan())
//     -> dequantize (abs(q)*528 & 0xFFFFFF, Round2(.,3), /dq_denom, clip)
//     -> inverse 2D transform (int64, normative clips, odd-log2 2896 prescale)
//     -> coded-size residual by §7.15.4 sample duplication
//
// Contract:
//   * every buffer is caller-owned and device-resident; the entry points
//     allocate nothing, free nothing and never synchronize;
//   * per-job parameters and element offsets are read from a device array, so
//     one launch can mix all 25 coded sizes (4x4..64x64, incl. 16x64/64x16/
//     32x8/8x32 and the 64->32 adjusted sizes) and all 16 TX_TYPE values;
//   * `TxParams` is the CPU POD from `tx_variants.h`; the quantizer state is
//     fully encoded in it (`dq_denom`, `quant_factor`).  M1 pins base_q_idx to
//     90, so there is no per-block qindex here.
//
// The output pointers in `M1TxBuffers` select the work: `levels` is required;
// `residual_adj`, `coeff`, `dequant`, `residual_out` and `eob` are optional
// and cost nothing when null.  A forward-only call (no `residual_out`) is the
// RDO / candidate-search shape; a full call is the stage-1 shape.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "encoder/tx_variants/tx_variants.h"

namespace av2enc::cuda {

using av2enc::txv::TxParams;
using av2enc::txv::TxType;

// Largest adjusted transform area (32x32).
inline constexpr int kM1TxMaxAdjElems = av2enc::txv::kMaxAdjTxElems;

// Where the coded-size residual comes from.
enum class M1TxInput : std::uint8_t {
  kResidual = 0,    // d_residual_in already holds int16 src-pred samples
  kSourcePred = 1,  // residual = source - prediction, computed in-kernel
};

// One transform block.  All offsets are element offsets into the matching
// device buffer in `M1TxBuffers`; the caller builds them (prefix sums are the
// natural choice for a mixed-size batch).
struct M1TxJob {
  TxParams params;  // coded/adjusted size, TX_TYPE, quantizer state

  std::uint8_t input = static_cast<std::uint8_t>(M1TxInput::kSourcePred);
  // 1: `src` is a packed coded-size block (src_offset only, stride = coded_w).
  // 0: `src` is a plane; src_offset/src_stride/src_x/src_y/src_width/
  //    src_height address the block and clamp reads exactly like
  //    TileEncoder::source_sample.
  std::uint8_t src_packed = 1;
  std::uint16_t reserved = 0;

  std::int16_t src_x = 0;       // plane mode: block origin in the source plane
  std::int16_t src_y = 0;
  std::int16_t src_width = 0;   // plane mode: visible width / height
  std::int16_t src_height = 0;

  std::int32_t src_offset = 0;       // element offset of the block / plane
  std::int32_t src_stride = 0;       // plane mode: elements per row
  std::int32_t pred_offset = 0;      // packed coded-size prediction
  std::int32_t residual_in_offset = 0;   // coded-size int16 residual
  std::int32_t residual_adj_offset = 0;  // adjusted-size int16 scratch/output
  std::int32_t coeff_offset = 0;         // adjusted-size double coefficients
  std::int32_t levels_offset = 0;        // adjusted-size int32 levels
  std::int32_t dequant_offset = 0;       // adjusted-size int32 dequantized
  std::int32_t residual_out_offset = 0;  // coded-size int16 inverse residual
  std::int32_t eob_offset = 0;           // int32 end-of-block position
};

// Caller-owned device buffers.  Unused pointers stay null.
struct M1TxBuffers {
  const std::uint8_t* src = nullptr;
  const std::uint8_t* pred = nullptr;
  const std::int16_t* residual_in = nullptr;
  std::int16_t* residual_adj = nullptr;
  double* coeff = nullptr;
  std::int32_t* levels = nullptr;
  std::int32_t* dequant = nullptr;
  std::int16_t* residual_out = nullptr;
  std::int32_t* eob = nullptr;
  // Optional: counts jobs rejected by the device-side parameter/smem guards.
  // A rejected job also writes eob = -1 when `eob` is non-null.
  int* error_count = nullptr;
};

// Fills `job->params` from the coded size and TX_TYPE and resets the offsets.
// Returns false when the coded size is not one of the 25 AV2 transform sizes.
// `src_packed` defaults to 1 and `input` to kSourcePred.
bool m1_tx_init_job(M1TxJob* job, int coded_w, int coded_h,
                    TxType type = TxType::kDctDct);

// Runs `nblocks` jobs in one launch on `stream`.
//
// `smem_elems` is the largest adjusted area (w*h) any job in the batch will
// use; pass 0 to size for the 32x32 maximum.  Jobs that exceed it are rejected
// through `error_count` / eob = -1 instead of touching out-of-range shared
// memory.  No allocation, no synchronization, no host callback: safe to record
// into a CUDA graph.
cudaError_t m1_tx_batch(const M1TxJob* d_jobs, int nblocks,
                        const M1TxBuffers& buffers, int smem_elems,
                        cudaStream_t stream);

}  // namespace av2enc::cuda
