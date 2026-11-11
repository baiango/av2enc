// M1 GPU reconstruction + deblock-metadata stage.
//
// Contract (docs/m1-gpu.md, workstream 5):
//   * write Clip1(pred + residual) into the padded reconstruction planes with
//     the same edge clamping as the CPU path (`Encoder::write_recon` in
//     src/encoder/m1/m1_still.cpp);
//   * maintain the block/edge state prediction needs (per-4x4 decoded flags,
//     IntraJointModes, the UV smooth grid) and the state the deblock needs
//     (per-plane `DeblockMi` grids on the luma MI grid, qindex, tile/frame
//     edges);
//   * invoke the existing `src/cuda/deblock` component once per frame, after
//     all reconstruction.  Deblocking is *never* applied during prediction:
//     normative order is reconstruct -> deblock -> CDEF/CCSO/LR.
//
// Ownership: this directory only.  No allocations, no synchronisation, no
// global state.  The caller owns the (arena) device buffers and the stream.
//
// The CPU reference arithmetic is scalar; the GPU path reproduces it exactly:
//
//   recon[x,y] = clip1(pred[x,y] + residual[x,y])
//   write only when 0 <= x0+x < plane.stride and 0 <= y0+y < plane.height
//
// which mirrors the CPU: the encoder writes the *padded* plane, clamping to
// the padded extent, not to the visible width/height.  Prediction later clamps
// reads to the coded MiCols*4 / MiRows*4 extent.

#ifndef AV2ENC_CUDA_M1_RECON_RECON_H_
#define AV2ENC_CUDA_M1_RECON_RECON_H_

#include <cstdint>

#include <cuda_runtime.h>

#include "../../deblock/deblock.h"

namespace av2enc::cuda {
namespace m1 {

// One padded reconstruction plane (device).  `stride` is the padded row
// stride; `height` the padded row count.  Both match the CPU `Plane` fields
// (luma stride/height are 64-aligned, chroma are half of those).
struct ReconPlane {
  std::uint8_t* data = nullptr;
  int stride = 0;
  int height = 0;
};

// Frame-level reconstruction state.  Device pointers only.
struct ReconFrame {
  ReconPlane plane[3];
  int mi_cols = 0;
  int mi_rows = 0;
  // Per-plane metadata grids, mi_cols*mi_rows entries each, row-major.  Chroma
  // grids are indexed on the full luma MI grid, exactly like the CPU path.
  deblock::DeblockMi* deblock_mi[3] = {nullptr, nullptr, nullptr};
  // Effective deblock qindex (M1: base_q_idx; no segmentation/delta-Q).
  std::uint8_t base_q_idx = 90;
  // Tile starts in luma MI units.  M1 is a single tile: leave num_* at 0.
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

// Prediction-side state, i.e. the CPU predictor's availability inputs:
//   * per-plane per-4x4 decoded flags (`DecodedMap`, reset per superblock);
//   * IntraJointModes (mi_rows*mi_cols) for the luma smooth-neighbour flags;
//   * UV smooth flags on the chroma 4x4 grid ((mi_cols/2)*(mi_rows/2)).
// Tile/frame partition contexts are re-derived by the syntax stage, so this
// stage does not need AboveMi/LeftMi.
struct ReconPredState {
  int mi_cols = 0;
  int mi_rows = 0;
  // Per-plane decoded flags, plane_cols*plane_rows bytes:
  //   plane 0: mi_cols x mi_rows; planes 1/2: mi_cols/2 x mi_rows/2
  std::uint8_t* decoded[3] = {nullptr, nullptr, nullptr};
  int decoded_cols[3] = {0, 0, 0};
  int decoded_rows[3] = {0, 0, 0};
  int* intra_joint = nullptr;  // mi_rows * mi_cols ints
  std::uint8_t* uv_smooth = nullptr;  // (mi_cols/2)*(mi_rows/2)
  // Current superblock, luma MI units.  Set by recon_begin_sb().
  int sb_mi_row = 0;
  int sb_mi_col = 0;
  int sb_cols[3] = {0, 0, 0};  // plane extent inside the current SB
  int sb_rows[3] = {0, 0, 0};
};

// One batched reconstruction write.  `pred` is w*h uint8, `residual` is w*h
// int16, both device-resident and in coded (possibly rectangular) block order
// with row stride w.  Regions of different jobs must not overlap.
struct ReconWriteJob {
  const std::uint8_t* pred = nullptr;
  const std::int16_t* residual = nullptr;
  std::uint8_t* plane = nullptr;
  int stride = 0;
  int height = 0;
  int x0 = 0;
  int y0 = 0;
  int w = 0;
  int h = 0;
};

// Per-block metadata for state maintenance.  Everything is in luma MI units
// unless noted.
struct ReconBlockInfo {
  int mi_row = 0;
  int mi_col = 0;
  int w4 = 0;  // luma width in 4x4 units
  int h4 = 0;  // luma height in 4x4 units
  int chroma_w4 = 0;  // chroma residual block extent in chroma 4x4 units
  int chroma_h4 = 0;
  std::uint8_t joint = 0;      // IntraJointMode written over the luma MI cells
  std::uint8_t uv_smooth = 0;  // §7.13.5.5 smooth-neighbour flag
  std::uint8_t qindex = 90;
  std::uint8_t tx_size_luma = 0;    // deblock::TxSize
  std::uint8_t tx_size_chroma = 0;  // deblock::TxSize
  std::uint8_t skip = 0;
  std::uint8_t region_intra = 1;
  std::uint8_t lossless = 0;
};

// Availability query for one plane block, in that plane's 4x4 units.
struct ReconAvailRequest {
  int plane = 0;
  int x4 = 0;
  int y4 = 0;
  int w4 = 0;
  int h4 = 0;
};

struct ReconAvail {
  int have_left = 0;
  int have_above = 0;
  int num4_above_right = 0;
  int num4_below_left = 0;
};

// Zeroes all three padded planes and resets the deblock metadata grids to the
// neutral `DeblockMi` (block_w4/h4 = 1, qindex = frame.base_q_idx), matching
// the CPU Encoder constructor.
cudaError_t recon_init_frame(const ReconFrame& frame,
                             cudaStream_t stream = nullptr);

// Initialises the prediction state: decoded flags, IntraJointModes and the UV
// smooth grid all zero.
cudaError_t recon_init_state(const ReconPredState& state,
                             cudaStream_t stream = nullptr);

// Starts a superblock: sets the host-side sb_* fields and resets all three
// decoded maps (the CPU resets `clear_block_decoded_flags` per SB).
cudaError_t recon_begin_sb(ReconPredState& state, int sb_mi_row, int sb_mi_col,
                           cudaStream_t stream = nullptr);

// Clip1(pred + residual) into the padded plane, same clamping as CPU
// `write_recon`.  One launch for the whole batch; jobs are independent.
cudaError_t recon_write_blocks(const ReconWriteJob* d_jobs, int job_count,
                               cudaStream_t stream = nullptr);
cudaError_t recon_write_block(const ReconWriteJob& job,
                              cudaStream_t stream = nullptr);

// Applies one block's state updates: decoded flags, IntraJointModes over its
// luma MI cells, the UV smooth flag over its chroma cells, and the three
// `DeblockMi` grids.  Batches are safe only when blocks do not share MI cells
// (partition leaves never do).
cudaError_t recon_commit_blocks(const ReconBlockInfo* d_infos, int count,
                                const ReconPredState& state,
                                const ReconFrame& frame,
                                cudaStream_t stream = nullptr);
cudaError_t recon_commit_block(const ReconBlockInfo& info,
                               const ReconPredState& state,
                               const ReconFrame& frame,
                               cudaStream_t stream = nullptr);

// Computes the CPU's count_top_right / count_bottom_left from the current
// state, plus have_left/have_above.  Reads the state only; does not commit.
cudaError_t recon_block_availability(const ReconPredState& state,
                                     const ReconAvailRequest* d_requests,
                                     ReconAvail* d_out, int count,
                                     cudaStream_t stream = nullptr);

// Single deblock invocation for the frame, after all reconstruction.  Wraps
// `cuda::deblock_frame`; the component itself issues the normative two passes
// (vertical then horizontal).  Returns cudaErrorInvalidValue for bad geometry.
cudaError_t recon_deblock_frame(const ReconFrame& frame,
                                const deblock::DeblockOptions& options,
                                cudaStream_t stream = nullptr);

}  // namespace m1
}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_M1_RECON_RECON_H_
