// Batched 10-bit M1 transform / quantize / inverse stage (GPU).
//
// This is the high-bit-depth sibling of `src/cuda/m1/tx/still_tx.*`.  The stage
// layout, batching contract and shared-memory plan are identical; the three
// differences are the ones AV2 makes bit-depth dependent:
//
//   * source samples are `uint16_t` (LSB-aligned 10-bit; the same code works
//     for 8/12-bit tests);
//   * dequantization clips to ±(1 << (7 + BitDepth)) (§7.14.3);
//   * the inverse transform row pass clips to ±(1 << (BitDepth + 7)) and the
//     column pass to ±(1 << BitDepth) (§7.15.2.1).
//
// The forward transform, quantizer, scan/EOB and §7.15.4 downsample are
// bit-depth independent.  `hbd_cpu.cpp` is the bit-exact oracle; the CUDA
// kernels use binary64 `__dadd_rn`/`__dmul_rn` with `-fmad=false`, matching
// `still_tx` and `tx_variants`.
//
// Contract (unchanged from still_tx): every buffer is caller-owned and
// device-resident; the entry points allocate nothing, free nothing and never
// synchronize; per-job element offsets are explicit so one launch can mix all
// 25 coded sizes; `levels` is the only required output.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "codec/transform/variants/tx_variants.h"

namespace av2enc::cuda::hbd {

using av2enc::txv::TxParams;
using av2enc::txv::TxType;

inline constexpr int kHbdTxMaxAdjElems = av2enc::txv::kMaxAdjTxElems;

enum class HbdTxInput : std::uint8_t {
  kResidual = 0,    // d_residual_in already holds int16 src-pred samples
  kSourcePred = 1,  // residual = source - prediction, computed in-kernel
};

// One transform block.  Offsets are element offsets into the matching buffer
// in `HbdTxBuffers`; the caller builds them (prefix sums for mixed batches).
struct HbdTxJob {
  TxParams params;  // coded/adjusted size, TX_TYPE, quantizer state

  std::uint8_t input = static_cast<std::uint8_t>(HbdTxInput::kSourcePred);
  // 1: `src` is a packed coded-size block (src_offset only, stride = coded_w).
  // 0: `src` is a plane; src_offset/src_stride/src_x/src_y/src_width/
  //    src_height address the block and clamp reads to the visible plane.
  std::uint8_t src_packed = 1;
  std::uint16_t reserved = 0;

  std::int16_t src_x = 0;
  std::int16_t src_y = 0;
  std::int16_t src_width = 0;
  std::int16_t src_height = 0;

  std::int32_t src_offset = 0;
  std::int32_t src_stride = 0;
  std::int32_t pred_offset = 0;
  std::int32_t residual_in_offset = 0;
  std::int32_t residual_adj_offset = 0;
  std::int32_t coeff_offset = 0;
  std::int32_t levels_offset = 0;
  std::int32_t dequant_offset = 0;
  std::int32_t residual_out_offset = 0;
  std::int32_t eob_offset = 0;
};

struct HbdTxBuffers {
  const std::uint16_t* src = nullptr;      // source plane / packed block
  const std::uint16_t* pred = nullptr;     // packed coded-size prediction
  const std::int16_t* residual_in = nullptr;
  std::int16_t* residual_adj = nullptr;
  double* coeff = nullptr;
  std::int32_t* levels = nullptr;
  std::int32_t* dequant = nullptr;
  std::int16_t* residual_out = nullptr;
  std::int32_t* eob = nullptr;
  // Optional: jobs rejected by the device-side parameter/smem guards.  A
  // rejected job also writes eob = -1 when `eob` is non-null.
  int* error_count = nullptr;
};

// Fills `job->params` from the coded size and TX_TYPE and resets the offsets.
// Returns false when the coded size is not one of the 25 AV2 transform sizes.
bool hbd_tx_init_job(HbdTxJob* job, int coded_w, int coded_h,
                     TxType type = TxType::kDctDct);

// Runs `nblocks` jobs in one launch on `stream`.
//
// `bit_depth` is 8, 10 or 12; it only changes the clip bounds.  `smem_elems`
// is the largest adjusted area (w*h) in the batch; pass 0 for the 32x32
// maximum.  Jobs that exceed it are rejected through `error_count` / eob = -1
// instead of touching out-of-range shared memory.
cudaError_t hbd_tx_batch(const HbdTxJob* d_jobs, int nblocks,
                         const HbdTxBuffers& buffers, int bit_depth,
                         int smem_elems, cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::hbd
