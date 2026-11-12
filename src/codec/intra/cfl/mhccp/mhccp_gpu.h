#ifndef AV2ENC_ENCODER_CFL_MHCCP_GPU_H_
#define AV2ENC_ENCODER_CFL_MHCCP_GPU_H_

// CUDA entry points for MHCCP. The kernel uses the same POD descriptors and
// integer arithmetic as the CPU reference (mhccp_common.h); the test compares
// the two byte for byte.
//
// One CUDA block per chroma transform block. Thread 0 captures CflRef and
// derives the Q16 weights; all threads then evaluate their samples. No
// allocation, no global state, asynchronous on `stream`.

#include "mhccp_common.h"

#include <cuda_runtime.h>

namespace av2enc::cuda::mhccp {

using av2enc::cfl::mhccp::Block;
using av2enc::cfl::mhccp::MultiParams;

struct Job {
  Block block;
  MultiParams params;
  int cfl_mh_dir = 0;
  int derived = 0;  // 1: derive params, 0: use `params`
};

// Single block. `derived` selects CFL_MULTI parameter derivation; otherwise
// `params` is applied. `d_params_out` is optional device memory (one value)
// and receives the parameters actually used.
cudaError_t mhccp_predict(const Block& block, const MultiParams& params,
                          int cfl_mh_dir, bool derived,
                          MultiParams* d_params_out = nullptr,
                          cudaStream_t stream = nullptr);

// Batched form. `d_jobs` is device memory. `d_params_out`, when non-null, is a
// device array of `job_count` MultiParams receiving the used parameters.
cudaError_t mhccp_predict_batch(const Job* d_jobs, int job_count,
                                MultiParams* d_params_out = nullptr,
                                cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::mhccp

#endif  // AV2ENC_ENCODER_CFL_MHCCP_GPU_H_
