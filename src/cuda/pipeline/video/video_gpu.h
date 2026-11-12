// CUDA M2 P-frame baseline — public integration API.
//
// `encode_video_sequence_gpu()` is a drop-in replacement for
// `av2enc::video::encode_video_sequence()`: same inputs, same options, same result
// structure.  It mirrors the CPU decisions exactly (partition tree, MV stack,
// GLOBALMV/NEARMV/NEWMV, DC-intra fallback) and produces byte-identical
// temporal units for the same input.
//
// What runs on the device:
//   * partition block-variance costs        (m2 variance kernel)
//   * integer/subpel motion estimation      (cuda/inter/me)
//   * motion compensation, batched          (cuda/inter/mc)
//   * DC intra prediction + residual        (cuda/intra_modes)
//   * residual downsample / forward / quant / inverse transform
//                                           (cuda/tx_variants)
//   * reconstruction (clip(pred + residual)) and recon-plane writes
//                                           (m2 recon kernel)
//
// What stays on the host for this milestone: partition-tree recursion, MV-stack
// derivation and candidate comparison (the CPU decision helpers), EOB scan,
// and the §8.2 range coder (src/encoder/m2/video_syntax.cpp).  Decisions are
// therefore the CPU decisions bit-for-bit; the GPU owns the pixels.
//
// The backend is selected at link time:
//   * AV2ENC_VIDEO_BACKEND=cuda            -> video_backend_cuda.cu
//   * AV2ENC_VIDEO_BACKEND=cpu-emulation   -> video_backend_cpu.cpp (no GPU needed)
//
// `video_backend_name()` reports which one is linked so a harness can label its
// numbers honestly.
#ifndef AV2ENC_CUDA_VIDEO_M2_GPU_H_
#define AV2ENC_CUDA_VIDEO_M2_GPU_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pipeline/video/video_frame.h"

namespace av2enc::cuda {
class Arena;
}

namespace av2enc::cuda::video {

// Device bytes the CUDA M2 backend needs for one frame configuration.
// `max_batch_jobs` is the largest variance/ME/SSE batch the caller will pass;
// 0 selects the backend default.  The persistent frame planes and per-block
// scratch are always included.
struct VideoDeviceConfig {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t mi_cols = 0;
  std::uint32_t mi_rows = 0;
  std::size_t max_batch_jobs = 0;
};

// Defined by the CUDA backend (video_backend_cuda.cu); includes arena alignment
// padding so `Arena::init(required_device_bytes(c))` is sufficient.
std::uint64_t required_device_bytes(const VideoDeviceConfig& config);

// Kernel/transfer accounting for one encode call.  The CUDA backend fills all
// fields from the real device; the CPU-emulation backend fills the launch
// counters with the number of equivalent component calls (it performs no GPU
// work) and leaves `device_ms` at 0.
struct VideoGpuStats {
  std::uint64_t variance_launches = 0;
  std::uint64_t me_launches = 0;
  std::uint64_t mc_launches = 0;
  std::uint64_t sse_launches = 0;
  std::uint64_t intra_launches = 0;
  std::uint64_t tx_launches = 0;
  std::uint64_t residual_launches = 0;
  std::uint64_t recon_launches = 0;
  std::uint64_t syncs = 0;
  std::uint64_t bytes_uploaded = 0;
  std::uint64_t bytes_downloaded = 0;
  std::uint64_t me_jobs = 0;
  std::uint64_t inter_candidates = 0;
  std::uint64_t intra_blocks = 0;
  std::uint64_t inter_blocks = 0;
  // Sum of the per-frame GPU intervals (CUDA events) or of the per-frame wall
  // time for the emulation backend, in milliseconds.
  double device_ms = 0.0;
  // One entry per encoded frame: CUDA event interval on the GPU backend,
  // wall time of the emulated backend calls on the CPU backend.
  std::vector<double> frame_ms;

  std::uint64_t total_launches() const {
    return variance_launches + me_launches + mc_launches + sse_launches +
           intra_launches + tx_launches + residual_launches + recon_launches;
  }
};

// Encodes every frame.  `frames` must be 8-bit 4:2:0 with dimensions divisible
// by 8, exactly like the CPU M2 entry point.  Returns false and sets `error`
// on failure.  On success `result` is byte-identical to the CPU reference for
// the same inputs and options.  `stats` may be null.
//
// `arena` may point at an already-initialised device Arena.  When null the
// CUDA backend owns a private arena sized from the device (see
// required_device_bytes()); the CPU-emulation backend ignores it.  An external
// arena must be at least required_device_bytes() for the frame size and must
// not be released while the call is running.
bool encode_video_sequence_gpu(
    std::span<const av2enc::video::VideoInputFrame> frames,
    const av2enc::video::VideoOptions& options,
    av2enc::video::VideoSequenceResult& result, std::string& error,
    VideoGpuStats* stats = nullptr, av2enc::cuda::Arena* arena = nullptr);

// "cuda" or "cpu-emulation".
const char* video_backend_name();

// Device residual TX/quant/inverse + reconstruction chain used by the inter
// residual path.  The M2 baseline codes every inter leaf with skip_flag = 1,
// so this chain is not selected by encode_video_sequence_gpu(); this entry point
// verifies it against the CPU transform reference on the linked backend.
bool video_inter_residual_selftest(int trials, std::string& error);

}  // namespace av2enc::cuda::video

#endif  // AV2ENC_CUDA_VIDEO_M2_GPU_H_
