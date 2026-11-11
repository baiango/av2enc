// AV2 §7.20 loop restoration / §7.20.3 non-separable Wiener — CUDA launcher.
//
// Shares the normative arithmetic with the CPU reference through
// src/encoder/filters/restoration/wiener_core.h, so device output is bit-exact
// with the CPU for identical planes, unit grids and coefficients.
//
// Device residency: every pointer in `frame` and `params` must be a device
// pointer (curr/cdef/lr planes, unit_type, unit_coeffs, frame_coeffs,
// subclass_grid, segment ids, lossless, tile grids, BruModes).
//
// Two limitations, both inherited from the unimplemented §7.20.4:
//   * frame_type == RESTORE_PC_WIENER is rejected with cudaErrorNotSupported.
//   * For RESTORE_SWITCHABLE the per-unit LrType grid is device memory, so the
//     launcher cannot verify that no unit selects RESTORE_PC_WIENER.  Such a
//     unit is left unfiltered (a copy of CdefFrame); callers must not emit it.
// GDF is rejected (see README) so `gdf_frame_enable` must be 0.
//
// The launcher performs the §7.20 copy step (LrFrame := CdefFrame) with three
// cudaMemcpy2DAsync calls, then one kernel launch.  No allocations and no
// hidden synchronisation.

#ifndef AV2ENC_CUDA_RESTORATION_WIENER_H_
#define AV2ENC_CUDA_RESTORATION_WIENER_H_

#include <cuda_runtime.h>

#include "encoder/filters/restoration/restoration.h"

namespace av2enc::cuda {

// One thread per 4x4 luma MI block per plane, matching the §7.20 loop.
cudaError_t loop_restoration_frame(
    const restoration::RestorationFrame& frame,
    const restoration::LoopRestorationParams& params,
    cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_RESTORATION_WIENER_H_
