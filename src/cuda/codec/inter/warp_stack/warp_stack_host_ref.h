#ifndef AV2ENC_CUDA_WARP_STACK_WARP_STACK_HOST_REF_H_
#define AV2ENC_CUDA_WARP_STACK_WARP_STACK_HOST_REF_H_

// Host emulation of the shared warp-stack device header.
//
// `warp_stack_host_ref.cpp` includes `warp_stack_device.cuh` with the CUDA
// execution-space keywords disabled and exposes plain host functions. The GPU
// parity test links this translation unit and compares the kernels against it,
// so the "CPU" side of the parity check is literally the same source the
// kernels are compiled from.
//
// This header is CUDA-free; the GPU test compiles it with the host compiler.

#include <cstdint>

#include "warp_stack_types.h"

namespace av2enc::cuda::warpstack::hostref {

// Runs the prediction body with one host thread. `dst`/`dst_stride` override
// the job's device destination so the same job value can be reused.
void predict(const WarpPredictJob& job, std::int16_t* dst, int dst_stride,
             WarpPredictResult* result);

// §7.12.2.3/.4 corner generation against host memory.
bool corners(const WarpCornerJob& job, WarpCornerResult* result);

// §7.12.3 sample collection against host memory.
bool samples(const inter::warp::WarpSampleParams& params,
             inter::warp::WarpSampleSet* result);

// §7.13.3.23 least-squares derivation.
ia::WarpParams estimate(const WarpEstimateJob& job);

// §7.13.3.14 fallback decision and §7.13.3.21 validity, for parity checks.
bool needs_extended(const ia::WarpParams& params, int w, int h, int ref_w, int ref_h,
                    int frame_w, int frame_h);
bool shear_valid(const ia::WarpParams& params);

}  // namespace av2enc::cuda::warpstack::hostref

#endif  // AV2ENC_CUDA_WARP_STACK_WARP_STACK_HOST_REF_H_
