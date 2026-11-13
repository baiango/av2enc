// CUDA §7.14.4 DC/AC delta-Q forward/inverse quantizer.
//
// Device half of `src/codec/transform/delta_q/`.  Per-plane DC and AC
// quantizers are derived on the device from `base_q_idx` and the resolved
// (sequence base + frame) deltas (`get_q()`), so the delta-Q decision itself
// does not need a host round trip.  A prebuilt q2 matrix may instead be
// supplied per job for quantizer-matrix interplay.
//
// Every entry point is pure: no allocation, no free, no synchronization.  All
// device memory is caller-owned; long-lived buffers belong to the single
// `av2enc::cuda::Arena`.
#pragma once

#include <cuda_runtime.h>

#include "cuda/codec/transform/delta_q/deltaq_types.h"

namespace av2enc::cuda::deltaq {

// coeff (double) -> levels (int32).  `Batch::coeff`, `Batch::out`,
// `Batch::jobs` are required; `Batch::q2` is required only for jobs with
// `q2_offset >= 0`.
cudaError_t quantize_batch(const Batch& batch, cudaStream_t stream = nullptr);

// levels (int32) -> dequant (int32).  `Batch::levels` and `Batch::out` may
// alias for an in-place dequantization.
cudaError_t dequantize_batch(const Batch& batch, cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::deltaq
