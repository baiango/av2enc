// AV2 inter motion estimation, extended — CUDA batch API.
//
// One warp per job, exactly like src/cuda/inter/me/.  The kernel calls the
// shared me2_search_core() with the warp-cooperative cost functor in
// me2_gpu_cost.cuh, so device MVs, costs, precision choices and evaluation
// counts are bit-exact with the CPU reference (me2.cpp) for the same inputs.
//
// Every pointer below is a device pointer.  Nothing is allocated, freed or
// synchronised here; the caller owns the buffers and the stream.

#ifndef AV2ENC_ENCODER_INTER_ME2_ME2_GPU_H_
#define AV2ENC_ENCODER_INTER_ME2_ME2_GPU_H_

#include <cstdint>

#include <cuda_runtime.h>

#include "encoder/inter/me2/me2.h"

namespace av2enc::cuda {

namespace me2 = ::av2enc::me2;

// Searches `num_jobs` blocks on device-resident references.  `d_field` may be
// null (no spatial merge candidates).
cudaError_t me2_search_batch(const me2::MeCoreFrame& cur,
                             const me2::MeCoreRef* d_refs, int num_refs,
                             const me2::MeCoreJob* d_jobs, int num_jobs,
                             const me2::Me2Options& options,
                             const me2::MeCoreMvField* d_field,
                             me2::Me2Result* d_results,
                             cudaStream_t stream = nullptr);

// Cost of explicit candidates at their signalled precision.  `d_out` receives
// num_jobs int32 costs.
cudaError_t me2_cost_batch(const me2::MeCoreFrame& cur,
                           const me2::MeCoreRef* d_refs, int num_refs,
                           const me2::Me2CostJob* d_jobs, int num_jobs,
                           std::int32_t* d_out, cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda

#endif  // AV2ENC_ENCODER_INTER_ME2_ME2_GPU_H_
