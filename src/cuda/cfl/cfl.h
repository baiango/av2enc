#ifndef AV2ENC_CUDA_CFL_H_
#define AV2ENC_CUDA_CFL_H_

// CUDA implementation of AV2 v1.0.0 §7.13.5 chroma-from-luma.
//
// Bit-exactness contract: the kernel uses the same integer helpers and the
// same POD block descriptor as the CPU reference (src/encoder/cfl/), from
// cfl_common.h. The arithmetic is integer only, so there is no FP contraction
// or rounding difference to guard against. cfl_test.cu asserts byte equality
// for explicit and derived alpha over randomised and edge geometry.
//
// One CUDA block covers one chroma transform block. Thread 0 performs the
// §7.13.5 luma-average / derived-alpha correlation (O(w + h) work); all
// threads then evaluate the prediction for their samples. No global state, no
// allocation, no synchronization inside the caller's stream.

#include <cstdint>

#include <cuda_runtime.h>

#include "../../encoder/cfl/cfl_common.h"

namespace av2enc::cuda {

// Applies CfL to one chroma block. `block.chroma` holds the DC prediction on
// input and receives the CfL prediction. alpha_q8 is CflAlpha * 32 for
// explicit signalling, or an arbitrary Q8 value (0..511) for derived alpha.
cudaError_t cfl_predict(const cfl::Block& block, int alpha_q8,
                        cudaStream_t stream = nullptr);

// §7.13.5 CFL_DERIVED_ALPHA: derives implicitAlpha from the reconstructed
// chroma above/left neighbours and applies it. When d_alpha_out is non-null
// the computed Q8 alpha is written there (device memory, one int).
cudaError_t cfl_predict_derived(const cfl::Block& block,
                                int* d_alpha_out,
                                cudaStream_t stream = nullptr);

struct CflJob {
  cfl::Block block;
  int alpha_q8 = 0;
};

// One CUDA block per job. `d_jobs` is device memory; job_count == 0 is a
// no-op. Designed for the batched RDO/intra path: descriptors must not alias.
cudaError_t cfl_predict_batch(const CflJob* d_jobs, int job_count,
                              cudaStream_t stream = nullptr);

// `d_blocks` and optional `d_alpha_out` are device memory. The derived alpha
// chosen for each block is written to d_alpha_out when non-null.
cudaError_t cfl_predict_derived_batch(const cfl::Block* d_blocks,
                                      int* d_alpha_out, int job_count,
                                      cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_CFL_H_
