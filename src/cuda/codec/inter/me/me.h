// AV2 inter motion estimation — CUDA API.
//
// The kernel calls the exact host/device core in
// src/encoder/inter/me/me_core.h, so device costs and selected MVs are
// bit-exact with the CPU reference (src/encoder/inter/me/me.cpp) for the same
// inputs.
//
// Scheduling: one warp per ME job (one warp per CUDA block by default).  SAD
// rows (or 8x8 SATD tiles) are split across the 32 lanes and reduced with warp
// shuffles; all search decisions are uniform across the warp.
//
// Every pointer below (frame planes, refs, jobs, MV field, outputs) must be a
// device pointer.  Nothing is allocated, freed, or synchronized here; the
// caller owns the buffers and the stream.  The caller must upload the ref/job
// arrays once per frame/batch, not per block.

#ifndef AV2ENC_CUDA_INTER_ME_ME_H_
#define AV2ENC_CUDA_INTER_ME_ME_H_

#include <cstdint>

#include <cuda_runtime.h>

#include "codec/inter/me/me_core.h"

namespace av2enc::cuda {

namespace me = ::av2enc::me;

// Searches `num_jobs` blocks against `num_refs` device-resident references.
// `d_field` may be null (no spatial merge candidates).
cudaError_t me_search_batch(const me::MeCoreFrame& cur,
                            const me::MeCoreRef* d_refs, int num_refs,
                            const me::MeCoreJob* d_jobs, int num_jobs,
                            const me::MeCoreOptions& options,
                            const me::MeCoreMvField* d_field,
                            me::MeCoreResult* d_results,
                            cudaStream_t stream = nullptr);

// Cost of one candidate per job.  `d_out` receives num_jobs int32 costs.
cudaError_t me_cost_batch(const me::MeCoreFrame& cur,
                          const me::MeCoreRef* d_refs, int num_refs,
                          const me::MeCostJob* d_jobs, int num_jobs,
                          std::int32_t* d_out, cudaStream_t stream = nullptr);

// Normative subpel prediction per job.  Each job writes w*h samples at
// `d_out + out_offset`.  Used by the CPU-vs-GPU interpolation test and by MC
// as a device oracle.
cudaError_t me_interp_batch(const me::MeCoreRef* d_refs, int num_refs,
                            const me::MeInterpJob* d_jobs, int num_jobs,
                            std::uint8_t* d_out,
                            cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_INTER_ME_ME_H_
