// AV2 v1.0.0 global motion estimation — CUDA API.
//
// The GPU pipeline is the device-side counterpart of
// `src/encoder/inter/gme/`:
//
//   * correspondence extraction (block MVs and per-block warps -> points,
//     plus direct point correspondences),
//   * deterministic integer least squares for TRANSLATION / ROTZOOM / AFFINE,
//   * median outlier rejection, §5.18.9 quantisation,
//   * AFFINE -> ROTZOOM -> IDENTITY degradation and validation,
//   * §7.12.2.2 projection / GLOBALMV, full MV range and all block sizes,
//   * the bilinear warp quality proxy.
//
// The integer core lives in `gme_device.cuh`, which is also compiled by the
// host for the no-nvcc emulation parity test.  The kernels in `gme.cu` are
// correctness-first: the fit runs one job per block with one active thread so
// the accumulation order is exactly the shared core's order; extraction and
// the image proxy are data-parallel.
//
// Memory contract (mandatory, project-wide)
// -----------------------------------------
// Every device byte comes from the single persistent `av2enc::cuda::Arena`.
// This module never calls cudaMalloc / cudaFree / cudaMallocHost, and neither
// do its tests.
//
//   * `required_device_bytes(config)` is the exact arena footprint of the
//     persistent buffers, including Arena::kAlignment padding.
//   * `allocate_checked(arena, config, &buffers, &required, &error)` bump
//     allocates once.  A short arena returns `GmeStatus::kArenaTooSmall` with
//     the exact requirement and leaves the bump pointer unchanged.
//   * `gme_mark()` / `gme_rewind()` expose the arena watermark; transient
//     scratch can be bracketed without growth.
//   * Launchers allocate nothing and never grow the arena.  The only copies
//     are the explicit `stage_*` / `download_*` setup calls.
//
// `GmeConfig::max_points` bounds both the number of point correspondences and
// the expanded warps (5 points per valid warp).  A warp batch whose expansion
// exceeds `max_points` reports the CPU's "too many correspondences" outcome
// rather than growing the arena.
#ifndef AV2ENC_CUDA_GME_GME_H_
#define AV2ENC_CUDA_GME_GME_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "cuda/codec/inter/gme/gme_device.cuh"

namespace av2enc::cuda::gme {

// Largest accepted count-like configuration field.  2^26 is far above an 8K
// 4x4 grid and keeps all byte arithmetic inside 64 bits.
inline constexpr int kGmeMaxCount = 1 << 26;

// Sentinel for "these buffers were never allocated".
inline constexpr std::uint64_t kGmeNoArenaMark = ~static_cast<std::uint64_t>(0);

struct GmeConfig {
  int max_points = 8192;  // point correspondences / expanded warp points
  int max_blocks = 8192;  // block-MV descriptors per job
  int max_warps = 2048;   // warp descriptors per job (5 points each)
  int max_jobs = 1;       // independent estimates (for example references)
};

// Exact persistent arena bytes for one buffer set, including alignment
// padding.  Returns 0 for an invalid configuration.
std::uint64_t required_device_bytes(const GmeConfig& config) noexcept;

enum class GmeStatus : int {
  kOk = 0,
  kInvalidArgument,
  kNotInitialized,
  kArenaTooSmall,
  kCudaError,
};

const char* gme_status_name(GmeStatus status) noexcept;

// Per-job extraction outcome.  `point_count` is the number of points the
// extraction produced (possibly above `max_points`, in which case
// `too_many_mask` is set).  `overflow_mask` records the CPU's int32-overflow
// identity path; `reason_mask` carries kGmeReasonBlockMvOverflow or
// kGmeReasonWarpOverflow.
struct GmeJobControl {
  int point_count = 0;
  int overflow_mask = 0;
  int too_many_mask = 0;
  int reason_mask = kGmeReasonNone;
};

// Device-resident buffers for one configuration.  Every pointer is a device
// pointer.  Arrays are per job, with job `j` at `job * max_*` entries.
struct GmeBuffers {
  GmePoint* d_points = nullptr;       // [jobs][max_points]
  GmePoint* d_active = nullptr;       // [jobs][max_points] fit scratch
  GmeI64* d_residuals = nullptr;      // [jobs][max_points] outlier scratch
  GmeU8* d_keep = nullptr;            // [jobs][max_points] kept flags
  GmeBlockMv* d_blocks = nullptr;     // [jobs][max_blocks]
  GmeWarpSample* d_warps = nullptr;   // [jobs][max_warps]
  GmeEstimate* d_results = nullptr;   // [jobs]
  GmeJobControl* d_control = nullptr; // [jobs]
  int max_points = 0;
  int max_blocks = 0;
  int max_warps = 0;
  int max_jobs = 0;
  std::uint64_t persistent_mark = kGmeNoArenaMark;
};

// Allocates every persistent buffer once.  On kArenaTooSmall `*required_bytes`
// receives the exact required_device_bytes(config) and the bump pointer is
// restored.
GmeStatus allocate_checked(Arena& arena, const GmeConfig& config,
                           GmeBuffers* out,
                           std::uint64_t* required_bytes = nullptr,
                           std::string* error = nullptr);

// Rewinds to the allocation mark and clears the buffer set.  Safe on a
// default-constructed/unallocated set.
void deallocate(Arena& arena, GmeBuffers& buffers) noexcept;

// Arena watermark wrappers.
std::uint64_t gme_mark(const Arena& arena) noexcept;
bool gme_rewind(Arena& arena, std::uint64_t mark) noexcept;

// Setup copies into the persistent staging arrays.  These are the only copies
// in the module and are never called from a launcher.
GmeStatus stage_points(const GmeBuffers& buffers, int job,
                       const GmePoint* host_points, int count,
                       cudaStream_t stream = nullptr,
                       std::string* error = nullptr);
GmeStatus stage_blocks(const GmeBuffers& buffers, int job,
                       const GmeBlockMv* host_blocks, int count,
                       cudaStream_t stream = nullptr,
                       std::string* error = nullptr);
GmeStatus stage_warps(const GmeBuffers& buffers, int job,
                      const GmeWarpSample* host_warps, int count,
                      cudaStream_t stream = nullptr,
                      std::string* error = nullptr);

// Downloads one result (or `count` results).  Both forms synchronise `stream`
// so the host output is valid on return.
GmeStatus download_estimate(const GmeBuffers& buffers, int job,
                            GmeEstimate* out, cudaStream_t stream = nullptr,
                            std::string* error = nullptr);
GmeStatus download_estimates(const GmeBuffers& buffers, GmeEstimate* out,
                             int count, cudaStream_t stream = nullptr,
                             std::string* error = nullptr);

// One estimate per call.  These launch kernels and do not synchronise; the
// caller downloads with download_estimate() (or synchronises the stream).
// `count` is points / blocks / warps respectively and must fit the staging
// capacity.
GmeStatus estimate_points(const GmeBuffers& buffers, int job, int count,
                          const GmeOptions& options,
                          cudaStream_t stream = nullptr,
                          std::string* error = nullptr);
GmeStatus estimate_blocks(const GmeBuffers& buffers, int job, int count,
                          const GmeOptions& options,
                          cudaStream_t stream = nullptr,
                          std::string* error = nullptr);
GmeStatus estimate_warps(const GmeBuffers& buffers, int job, int count,
                         const GmeOptions& options,
                         cudaStream_t stream = nullptr,
                         std::string* error = nullptr);

// Bilinear warp quality proxy over device planes.  `bit_depth` must be 8,
// matching the CPU helper.  `warp_plane_sse` writes one int64 SSE value to
// `d_sse`; synchronise before reading it on the host.
GmeStatus warp_plane_bilinear(const GmeBuffers& buffers,
                              const GmeParams& params, const GmeU8* d_src,
                              int src_w, int src_h, int src_stride,
                              GmeU8* d_dst, int dst_w, int dst_h,
                              int dst_stride, int bit_depth,
                              cudaStream_t stream = nullptr,
                              std::string* error = nullptr);
GmeStatus warp_plane_sse(const GmeBuffers& buffers, const GmeParams& params,
                         const GmeU8* d_ref, int ref_w, int ref_h,
                         int ref_stride, const GmeU8* d_cur, int cur_w,
                         int cur_h, int cur_stride, int bit_depth,
                         GmeI64* d_sse, cudaStream_t stream = nullptr,
                         std::string* error = nullptr);

}  // namespace av2enc::cuda::gme

#endif  // AV2ENC_CUDA_GME_GME_H_
