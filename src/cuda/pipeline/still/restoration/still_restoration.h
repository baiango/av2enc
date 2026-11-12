// M1 GPU in-loop restoration stage — CCSO (§7.19) + loop restoration (§7.20).
//
// Placement: this stage runs **after CDEF** and before the reference/output is
// finalised.  The normative order is
//
//     reconstruct -> deblock -> CDEF -> CCSO -> loop restoration
//
// so `curr` is the deblocked (pre-CDEF) frame, `cdef` is the post-CDEF frame
// (CCSO modifies it in place), and `lr` receives the loop-restoration output,
// which is the final reference/output plane.  When CDEF is disabled, `curr` and
// `cdef` may hold two copies of the same deblocked frame; when CCSO is enabled
// they must not alias (CCSO reads CurrFrame luma while writing CdefFrame).
//
// Scope: 8-bit 4:2:0, one tile, no GDF, no PC-Wiener.  Every pointer is a
// device pointer, including the optional metadata grids and all coefficient
// arrays.  The stage allocates nothing, never calls cudaDeviceSynchronize(),
// and launches at most one CCSO kernel plus the restoration copy/filter.
//
// Ownership: `src/cuda/m1/restoration/` only.  The orchestrator owns the arena,
// the stream and the surrounding pipeline.  This header deliberately does not
// include `src/cuda/m1/still_gpu.h` so the stage can be tested before the
// orchestrator hook is wired.

#ifndef AV2ENC_CUDA_STILL_RESTORATION_STILL_RESTORATION_H_
#define AV2ENC_CUDA_STILL_RESTORATION_STILL_RESTORATION_H_

#include <cstdint>

#include <cuda_runtime.h>

#include "codec/filters/loop_restoration/restoration.h"

namespace av2enc::cuda {
namespace still {

// ---------------------------------------------------------------------------
// CCSO stage parameters
// ---------------------------------------------------------------------------

// One §5.18.7.12 state.  `plane[p].offsets` in the host POD is ignored by this
// stage: the kernel reads `offsets_dev[p]`, a device copy of the 576-entry
// CcsoFilterOffset table produced by restoration::ccso_set_offsets().  Upload
// it once per parameter change; do not put it in the per-frame hot path.
struct RestorationCcsoStage {
  // Parsed per-plane values (host copies).  Only the scalar fields are read by
  // the stage; `offsets` is not dereferenced.
  restoration::CcsoParams plane[3];

  // Device-resident per-plane tables/grids.
  const std::int16_t* offsets_dev[3] = {nullptr, nullptr, nullptr};
  const std::uint8_t* blks[3] = {nullptr, nullptr, nullptr};
  int blk_cols[3] = {0, 0, 0};
  int blk_rows[3] = {0, 0, 0};

  // §5.18.7.12 CcsoLumaSizeLog2: 6/7/8 (64x64 for M1's 64x64 SB).
  int ccso_luma_size_log2 = 6;

  bool any_enabled() const {
    return plane[0].enabled || plane[1].enabled || plane[2].enabled;
  }
};

// ---------------------------------------------------------------------------
// Whole stage
// ---------------------------------------------------------------------------

// `lr` carries every §5.18.7.11 field; all of its pointers and the pointers in
// `RestorationFrame` must be device pointers.  `lr.unit_type`, `unit_coeffs`,
// `frame_coeffs`, `filter_class_grid`/`subclass_lookup`/`subclass_grid`,
// `segment_ids`, `lossless`, tile grids and BruModes are all read by kernels.
struct RestorationStageParams {
  RestorationCcsoStage ccso;
  restoration::LoopRestorationParams lr;
};

// Validates a configuration without launching or dereferencing device grids.
// The class/tile grids are checked for presence and shape only; their contents
// are a bitstream-conformance responsibility of the caller.
//
//   cudaSuccess            ready to run
//   cudaErrorInvalidValue  bad geometry/pointers/unit sizes/aliasing
//   cudaErrorNotSupported  GDF, PC-Wiener, or an unsupported bit depth
cudaError_t restoration_stage_validate(const restoration::RestorationFrame& frame,
                                       const RestorationStageParams& params);

// Runs CCSO (if any plane is enabled), then loop restoration.  The stage
// always performs the §7.20 `LrFrame := CdefFrame` copy, even when every
// FrameRestorationType is RESTORE_NONE, so `lr` is always the final plane.
//
// No allocation, no synchronization, no global state.  The caller must keep
// `frame`, `params` and every referenced device buffer alive until the stream
// work completes.
cudaError_t restoration_stage_apply(const restoration::RestorationFrame& frame,
                                    const RestorationStageParams& params,
                                    cudaStream_t stream = nullptr);

// Builds the device-side §7.19 core from the stage parameters.  Exposed for the
// parity tests and for orchestrators that want to issue CCSO separately.
restoration::CcsoCoreFrame restoration_stage_ccso_core(
    const restoration::RestorationFrame& frame,
    const RestorationCcsoStage& params);

}  // namespace still
}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_STILL_RESTORATION_STILL_RESTORATION_H_
