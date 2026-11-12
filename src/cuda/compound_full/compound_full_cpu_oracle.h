#ifndef AV2ENC_CUDA_COMPOUND_FULL_CPU_ORACLE_H_
#define AV2ENC_CUDA_COMPOUND_FULL_CPU_ORACLE_H_

// Host-only bridge to the CPU compound reference.
//
// `adv::run_compound_job()` is `__host__ __device__` and its body reaches
// `adv::tables::kWedgeBits` / `kWedgeCodebook` through `bsize_has_wedge()` and
// `wedge_mask_value()`, which nvcc cannot compile for device code when the
// call appears in a .cu translation unit. The GPU test therefore calls this
// plain host function (compiled by the host compiler) instead. This keeps the
// CPU oracle exactly `adv::run_compound_job()` while nvcc never sees it.

#include "encoder/inter/advanced/compound.h"

namespace av2enc::cuda::compound_full::test {

void run_compound_job_cpu(const av2enc::inter::adv::CompoundJob& job);

}  // namespace av2enc::cuda::compound_full::test

#endif  // AV2ENC_CUDA_COMPOUND_FULL_CPU_ORACLE_H_
