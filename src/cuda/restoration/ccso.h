// AV2 §7.19 CCSO — CUDA launcher.
//
// The kernel calls the same host/device core as the CPU reference
// (src/encoder/filters/restoration/ccso_core.h), so the device output is
// bit-exact with the CPU for identical planes and metadata.
//
// Device residency: every pointer in `frame` must be a device pointer,
// including `plane[p].offsets` (a device copy of the parsed
// CcsoFilterOffset table) and the CcsoBlks / BruModes / segment / tile grids.
// `restoration::ccso_to_core()` only copies pointers, so a CUDA caller builds
// the core directly instead of going through the host `CcsoFrameParams`, whose
// offset table is inline host memory.
//
// No allocations, no hidden synchronisation, no global state.

#ifndef AV2ENC_CUDA_RESTORATION_CCSO_H_
#define AV2ENC_CUDA_RESTORATION_CCSO_H_

#include <cuda_runtime.h>

#include "encoder/filters/restoration/ccso_core.h"

namespace av2enc::cuda {

// One thread visits one sample of one plane and applies §7.19.1 to it.
// `frame.plane[p].enabled` selects the planes; disabled planes are skipped.
cudaError_t ccso_frame(const restoration::CcsoCoreFrame& frame,
                       cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_RESTORATION_CCSO_H_
