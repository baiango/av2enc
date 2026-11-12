#ifndef AV2ENC_CUDA_MRL_MRL_H_
#define AV2ENC_CUDA_MRL_MRL_H_

// AV2 v1.0.0 multiple reference line selection (MRL) — CUDA predictor.
//
// §7.13.2.1 AboveRow/LeftCol with MrlIndex > 0, §7.13.2.7 Mrl_Index_To_Delta
// and the secondary-line blend, §7.13.2.8 single directional prediction with
// mrlIndex.
//
// Scope mirrors the CPU reference in src/encoder/mrl/: luma directional modes
// only, mrl_index 0..3, optional mrl_sec_index. MRL is luma-only; chroma forces
// MrlIndex = 0 and is rejected by this module (use src/cuda/intra_modes/).
//
// All arithmetic is integer and identical to the CPU reference. Entry points
// never allocate or synchronise; the caller owns the stream and buffers.

#include <cstdint>

#include <cuda_runtime.h>

namespace av2enc::cuda::mrl {

inline constexpr int kMaxMrlIndex = 3;
inline constexpr int kMaxBlockDim = 64;
inline constexpr int kMaxAngleDelta = 3;

// One luma block. `recon` is the reconstructed plane and must contain the
// selected reference lines (row y-1-AboveMrlIndex, column x-1-MrlIndex).
struct MrlBlock {
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
  int plane = 0;       // must be 0 (luma)
  int sb_height = 64;  // Block_Height[SbSize]
};

struct MrlParams {
  int mode = 3;          // AV2 directional YMode (1..8)
  int angle_delta = 0;   // -3..3
  int mrl_index = 0;     // 0..3
  int mrl_sec_index = 0; // only with mrl_index > 0 and block != 4x4
  int enable_edge_filter = 1;
  int filter_type_above = 0;  // 0/1 smooth-neighbour flags
  int filter_type_left = 0;
};

// Device-resident batch job.
struct MrlJob {
  MrlBlock block{};
  MrlParams params{};
  std::uint8_t* pred = nullptr;
};

cudaError_t mrl_predict(const MrlBlock& block, const MrlParams& params,
                        std::uint8_t* pred, cudaStream_t stream = nullptr);

// One CUDA block per job. d_jobs is device memory; jobs must be independent.
cudaError_t mrl_predict_batch(const MrlJob* d_jobs, int job_count,
                              cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::mrl

#endif  // AV2ENC_CUDA_MRL_MRL_H_
