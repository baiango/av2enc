#ifndef AV2ENC_CUDA_INTER_ADVANCED_H_
#define AV2ENC_CUDA_INTER_ADVANCED_H_

// CUDA implementations of the advanced inter predictors (AV2 v1.0.0):
// compound / inter-intra blending (§7.13.3.14, .27–.30), block and extended
// warp (§7.13.3.19/.20), BAWP (§7.13.3.25) and the encoder-only OBMC experiment.
//
// Bit-exactness contract: every kernel calls the same `AV2ENC_ADV_HD` inline
// helpers as the CPU reference in src/encoder/inter/advanced/, so there is no
// second transcription of the arithmetic. The tests assert byte equality over
// randomised jobs, edge sizes and both precision modes.
//
// All pointers inside the job structs are *device* pointers. Jobs are small POD
// values; the batches copy them by value into kernel parameters, so there is no
// per-job allocation. Use `av2enc::cuda::arena` in the encoder; the component
// test uses cudaMalloc directly.
//
// Reference-list selection (§7.7/§7.8) and GOP scheduling stay on the host: they
// run once per frame, not per block.

#include <cuda_runtime.h>

#include <cstdint>

#include "../../../encoder/inter/advanced/bawp.h"
#include "../../../encoder/inter/advanced/compound.h"
#include "../../../encoder/inter/advanced/obmc.h"
#include "../../../encoder/inter/advanced/warp.h"

namespace av2enc::cuda::advinter {

// ---------------------------------------------------------------------------
// Compound / inter-intra
// ---------------------------------------------------------------------------

// One CUDA block per job. `jobs` lives in device memory; `job_count == 0` is a
// no-op. Each block strides `blockDim.x` threads over the job's samples.
cudaError_t compound_batch(const inter::adv::CompoundJob* jobs, int job_count,
                           cudaStream_t stream = nullptr);

// Single-job convenience (one host-to-device copy of the descriptor).
cudaError_t compound_one(const inter::adv::CompoundJob& job, cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Warp
// ---------------------------------------------------------------------------

struct WarpJob {
  inter::adv::WarpParams params{};
  inter::adv::PlaneView ref{};
  inter::adv::WarpBlockGeometry geom{};
  int compound = 0;
  std::int16_t* dst = nullptr;
  int dst_stride = 0;
};

// §7.13.3.19: 8x8 sections. One CUDA block per job, one shared intermediate.
cudaError_t warp_batch(const WarpJob* jobs, int job_count, cudaStream_t stream = nullptr);

// §7.13.3.20: 4x4 sections of one translation each.
cudaError_t ext_warp_batch(const WarpJob* jobs, int job_count, cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// BAWP
// ---------------------------------------------------------------------------

struct BawpGpuJob {
  inter::adv::BawpJob job{};
  inter::adv::BawpFit* fit_out = nullptr;  // optional, device memory
};

// One CUDA block per job: thread 0 fits alpha/beta from the template, then all
// threads apply the adjustment. The returned `BawpFit` is bit-identical to
// `bawp_derive_block()`.
cudaError_t bawp_batch(const BawpGpuJob* jobs, int job_count, cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// OBMC (encoder-only experiment; not part of AV2 v1.0.0)
// ---------------------------------------------------------------------------

struct ObmcGpuJob {
  inter::adv::ObmcConfig config{};
  const std::uint8_t* above = nullptr;
  int above_stride = 0;
  const std::uint8_t* left = nullptr;
  int left_stride = 0;
};

cudaError_t obmc_batch(const ObmcGpuJob* jobs, int job_count, cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::advinter

#endif  // AV2ENC_CUDA_INTER_ADVANCED_H_
