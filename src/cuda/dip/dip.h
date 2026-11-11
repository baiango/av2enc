#ifndef AV2ENC_CUDA_DIP_DIP_H_
#define AV2ENC_CUDA_DIP_DIP_H_

// AV2 v1.0.0 data-driven intra prediction (DIP) — CUDA predictor.
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//   §7.13.2.1  AboveRow/LeftCol with useDip
//   §7.13.2.3  data driven intra prediction process
//   §7.13.2.4  DIP features process
//   §7.13.2.5  DIP transform process
//   §7.13.2.6  DIP resample process
//
// Scope mirrors the CPU reference in src/encoder/dip/: luma, 8-bit samples,
// DIP-capable transform extents only (w > 4, h > 4, w*h >= 128).  The kernel
// runs the same integer operations as the CPU reference; entry points never
// allocate, synchronise, or write the reconstructed plane.

#include <cstdint>

#include <cuda_runtime.h>

namespace av2enc::cuda::dip {

inline constexpr int kNumModes = 6;
inline constexpr int kMaxBlockDim = 64;

// One luma block.  `recon` is the reconstructed plane containing row y-1 and
// column x-1 (subject to have_above/have_left).
struct DipBlock {
  const std::uint8_t* recon = nullptr;
  int recon_stride = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int max_x = 0;  // inclusive visible bounds
  int max_y = 0;
  int have_left = 0;
  int have_above = 0;
  int num4_above_right = 0;
  int num4_below_left = 0;
};

struct DipParams {
  int mode = 0;       // dip_mode 0..5
  int transpose = 0;  // dip_transpose
};

struct DipJob {
  DipBlock block{};
  DipParams params{};
  std::uint8_t* pred = nullptr;
};

// One DIP block.  `pred` is a device pointer with at least w*h bytes.
cudaError_t dip_predict(const DipBlock& block, const DipParams& params,
                        std::uint8_t* pred, cudaStream_t stream = nullptr);

// One CUDA block per job.  `d_jobs` is device memory; jobs must be independent.
cudaError_t dip_predict_batch(const DipJob* d_jobs, int job_count,
                              cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::dip

#endif  // AV2ENC_CUDA_DIP_DIP_H_
