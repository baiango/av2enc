#ifndef AV2ENC_CUDA_INTRA_H_
#define AV2ENC_CUDA_INTRA_H_

// M0 DC intra prediction / residual / reconstruction kernels.
//
// Bit-exactness contract: the arithmetic mirrors the CPU reference in
// src/encoder/m0_still.cpp (encode_residual_plane):
//
//   pred = 128;
//   if (have_left && have_above)
//     pred = clip1(round2((sum_left + sum_above) * 512,
//                         9 + floor_log2(width + height)));
//   else if (have_left)
//     pred = round2(sum_left, floor_log2(height));
//   else if (have_above)
//     pred = round2(sum_above, floor_log2(width));
//
//   residual[i] = source(clamped) - pred;
//   recon[i]    = clip1(pred + residual[i]);
//
// Prediction samples are read from the *reconstructed* plane at x0-1 / y0-1
// and clamped to the visible plane bounds (plane_max_x / plane_max_y), exactly
// like the CPU. Source samples are clamped to source_max_x / source_max_y.
// Reconstruction writes into a strided plane are dropped outside the visible
// bounds; the CPU writes the padded plane instead, but padded samples are
// never referenced by prediction (all reads clamp to the visible bounds), so
// the visible output is identical.
//
// All functions are pure: device pointers in, cudaStream_t, no global state.

#include <cstdint>

#include <cuda_runtime.h>

namespace av2enc::cuda {

// Largest supported block extent per axis (64x64 luma / 32x32 chroma in M0).
constexpr int kIntraMaxBlockDim = 64;

// One DC-intra block description.
struct DcBlock {
  // Reconstructed plane used for prediction (left column / above row).
  const std::uint8_t* recon_plane = nullptr;
  int recon_stride = 0;  // samples per row
  int x0 = 0;            // block origin, plane samples
  int y0 = 0;
  int block_w = 0;  // 32x32 / 32x16 / 32x8 luma, 16x16 / 16x8 / 16x4 chroma
  int block_h = 0;
  int plane_max_x = 0;  // last visible column, inclusive (width - 1)
  int plane_max_y = 0;  // last visible row, inclusive (height - 1)
  int have_left = 0;    // left neighbour exists (block not in column 0)
  int have_above = 0;   // above neighbour exists (block not in row 0)

  // Source plane (only used by the residual / fused entry points).
  const std::uint8_t* source_plane = nullptr;
  int source_stride = 0;
  int source_max_x = 0;  // last visible source column, inclusive
  int source_max_y = 0;  // last visible source row, inclusive
};

// Device-resident job for the batched prediction + residual path. The whole
// descriptor lives in device memory, so one launch can cover many independent
// blocks. Jobs must not depend on one another's output planes.
struct DcBlockJob {
  DcBlock block{};
  std::uint8_t* pred_block = nullptr;      // block_w * block_h, required
  std::int16_t* residual_block = nullptr;  // block_w * block_h, required
};

// Normative DC predictor value for one block; writes one int32 to *d_pred.
cudaError_t dc_pred_value(const DcBlock& block, int* d_pred,
                          cudaStream_t stream = nullptr);

// Fills pred_block[0 .. block_w*block_h) with the DC predictor (uint8).
cudaError_t dc_predict(const DcBlock& block, std::uint8_t* pred_block,
                       cudaStream_t stream = nullptr);

// residual = source - prediction. If pred_block is null the predictor is
// recomputed on the fly; otherwise the supplied per-sample prediction is used.
cudaError_t dc_residual(const DcBlock& block, const std::uint8_t* pred_block,
                        std::int16_t* residual_block,
                        cudaStream_t stream = nullptr);

// Fused prediction + residual for one block (single kernel launch).
cudaError_t dc_predict_residual(const DcBlock& block, std::uint8_t* pred_block,
                                std::int16_t* residual_block,
                                cudaStream_t stream = nullptr);

// Batched fused prediction + residual: one CUDA block per job. d_jobs must be
// device memory; job_count == 0 is a no-op. Intended for dependency-free
// throughput measurement and for a later wavefront pass.
cudaError_t dc_predict_residual_batch(const DcBlockJob* d_jobs, int job_count,
                                      cudaStream_t stream = nullptr);

// recon = clip1(pred + residual) for block_w * block_h samples.
cudaError_t add_residual_clip(const std::uint8_t* pred_block,
                              const std::int16_t* residual_block,
                              std::uint8_t* recon_block, int block_w,
                              int block_h, cudaStream_t stream = nullptr);

// Same, but writes into a strided plane at (x0, y0). Samples outside
// [0, plane_max_x] x [0, plane_max_y] are not written. This is the
// reconstruction step after inverse transform.
cudaError_t add_residual_clip_to_plane(
    const std::uint8_t* pred_block, const std::int16_t* residual_block,
    std::uint8_t* plane, int stride, int x0, int y0, int block_w, int block_h,
    int plane_max_x, int plane_max_y, cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_INTRA_H_
