// Device-side MHCCP preparation for the CUDA kernel and host-emulation test.
//
// Both entry points call the shared arithmetic in mhccp_common.h, so the only
// difference from the CPU reference is the thread mapping in mhccp_gpu.cu.

#ifndef AV2ENC_ENCODER_CFL_MHCCP_DEVICE_CUH_
#define AV2ENC_ENCODER_CFL_MHCCP_DEVICE_CUH_

#include "mhccp_common.h"

namespace av2enc::cfl::mhccp::device {

// Thread 0 entry point: capture CflRef and either derive the Q16 weights or
// use the caller-supplied explicit weights.
__device__ __forceinline__ bool prepare(const Block& block, int cfl_mh_dir,
                                        bool derived,
                                        const MultiParams& explicit_params,
                                        Refs* refs, MultiParams* params) {
  if (refs == nullptr || params == nullptr) {
    return false;
  }
  if (!capture_refs(block, *refs)) {
    return false;
  }
  if (derived) {
    *params = derive_multi_params(block, *refs, cfl_mh_dir, nullptr);
  } else {
    *params = explicit_params;
  }
  return true;
}

// One predicted chroma sample.
__device__ __forceinline__ int sample(const Block& block, const Refs& refs,
                                      const MultiParams& params,
                                      int cfl_mh_dir, int i, int j) {
  return predict_sample(block, refs, params, cfl_mh_dir, i, j);
}

}  // namespace av2enc::cfl::mhccp::device

#endif  // AV2ENC_ENCODER_CFL_MHCCP_DEVICE_CUH_
