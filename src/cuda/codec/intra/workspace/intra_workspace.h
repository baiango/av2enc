// Arena-backed workspace for the AV2 intra tools.
//
// The component kernels (modes, MRL, IBP, DIP, CfL, palette, IntraBC and the
// supplemental intra-full rules) take device descriptors and caller-owned
// planes.  The still orchestrator would otherwise have to stage every
// descriptor array itself, which is exactly the failure documented in
// `docs/negative-results.md` ("pass a host std::vector<Job> to a *_batch
// device API").  This workspace is the single entry point that:
//
//   * owns one persistent descriptor-staging buffer plus the supplemental
//     module's persistent result arrays, all from the caller's single Arena;
//   * uploads host descriptors into arena staging before each launch;
//   * returns a Status per call and never allocates outside the arena;
//   * exposes `required_device_bytes(config)` for arena sizing.
//
// Execution model
// ---------------
// The tool methods are asynchronous: they stage descriptors and launch on the
// caller's stream, then return.  Calls are serialized by the stream, so the one
// staging buffer is safe: a later upload is ordered after every earlier kernel
// that reads it.  Use one stream per workspace (do not interleave streams) and
// call `synchronize()` once per frame before reusing the result buffers.
//
// Host descriptor arrays may be pageable memory: `cudaMemcpyAsync` from pageable
// host memory is staged by the runtime before it returns, so the caller may
// free or modify the array after the method returns.  Device descriptors inside
// the jobs (plane pointers, output pointers) must stay valid until
// `synchronize()`.
//
// The supplemental tools (DC-subsampled, filter corner, chroma type, dc_sign,
// identity transforms) keep their own synchronous batch semantics; they are
// reached through `supplemental()` with the same Arena passed at init.
#ifndef AV2ENC_CUDA_INTRA_WORKSPACE_INTRA_WORKSPACE_H_
#define AV2ENC_CUDA_INTRA_WORKSPACE_INTRA_WORKSPACE_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "cuda/arena.h"
#include "cuda/codec/intra/cfl/cfl.h"
#include "cuda/codec/intra/dip/dip.h"
#include "cuda/codec/intra/ibc/ibc.h"
#include "cuda/codec/intra/ibp/ibp.h"
#include "cuda/codec/intra/modes/intra_modes.h"
#include "cuda/codec/intra/mrl/mrl.h"
#include "cuda/codec/intra/palette/palette.h"
#include "cuda/codec/intra/supplemental/intra_full.h"
#include "cuda/error.h"

namespace av2enc::cuda::intra_workspace {

// `max_jobs` bounds every descriptor batch (per tool).  `supplemental` is the
// intra-full configuration forwarded by `init()`.
struct Config {
  int max_jobs = 256;
  intra_full::Config supplemental{};
};

namespace detail {

constexpr int kMaxBatch = 1 << 20;

constexpr std::uint64_t align_up_bytes(std::uint64_t value,
                                       std::uint64_t alignment) noexcept {
  return alignment == 0 ? value
                        : ((value + alignment - 1) / alignment) * alignment;
}

// Largest descriptor the staging buffer must hold.
constexpr std::size_t max_job_bytes() noexcept {
  std::size_t largest = sizeof(intra_modes::IntraJob);
  if (sizeof(mrl::MrlJob) > largest) largest = sizeof(mrl::MrlJob);
  if (sizeof(ibp::IbpJob) > largest) largest = sizeof(ibp::IbpJob);
  if (sizeof(dip::DipJob) > largest) largest = sizeof(dip::DipJob);
  if (sizeof(CflJob) > largest) largest = sizeof(CflJob);
  if (sizeof(PaletteDeviceBlock) > largest) largest = sizeof(PaletteDeviceBlock);
  if (sizeof(PaletteContextDevice) > largest) {
    largest = sizeof(PaletteContextDevice);
  }
  if (sizeof(av2enc::ibc::BlockVector) > largest) {
    largest = sizeof(av2enc::ibc::BlockVector);
  }
  return largest;
}

constexpr std::uint64_t staging_bytes(const Config& c) noexcept {
  if (c.max_jobs < 1 || c.max_jobs > kMaxBatch) {
    return 0;
  }
  return align_up_bytes(
      static_cast<std::uint64_t>(c.max_jobs) * max_job_bytes(),
      Arena::kAlignment);
}

constexpr bool config_valid(const Config& c) noexcept {
  return c.max_jobs >= 1 && c.max_jobs <= kMaxBatch &&
         staging_bytes(c) != 0;
}

}  // namespace detail

// One descriptor-staging region, shared by every tool method.
inline std::uint64_t staging_device_bytes(const Config& config) noexcept {
  return detail::staging_bytes(config);
}

// Persistent allocations: staging + the supplemental module's result arrays.
inline std::uint64_t persistent_device_bytes(const Config& config) noexcept {
  if (!detail::config_valid(config)) {
    return 0;
  }
  return detail::staging_bytes(config) +
         intra_full::persistent_device_bytes(config.supplemental);
}

// Largest per-launch scratch of the forwarded supplemental module.
inline std::uint64_t scratch_device_bytes(const Config& config) noexcept {
  if (!detail::config_valid(config)) {
    return 0;
  }
  return intra_full::scratch_device_bytes(config.supplemental);
}

inline std::uint64_t required_device_bytes(const Config& config) noexcept {
  return persistent_device_bytes(config) + scratch_device_bytes(config);
}

class Workspace {
 public:
  Workspace() = default;

  // Allocates the staging region and initializes the supplemental workspace.
  // On failure the arena bump pointer is restored and `*required_out` (when
  // non-null) receives `required_device_bytes(config)`.
  Status init(Arena& arena, const Config& config = {},
              std::string* error = nullptr,
              std::uint64_t* required_out = nullptr);

  bool initialized() const noexcept { return arena_ != nullptr; }
  const Config& config() const noexcept { return config_; }
  const intra_full::Workspace& supplemental() const noexcept {
    return supplemental_;
  }
  intra_full::Workspace& supplemental() noexcept { return supplemental_; }

  void* staging() noexcept { return staging_; }
  const void* staging() const noexcept { return staging_; }

  // Stream synchronization for the async tool launches below.
  Status synchronize(cudaStream_t stream = nullptr);

  // --- Prediction tools ----------------------------------------------------

  // §7.13.2 full intra mode set (DC/V/H/directional/smooth/Paeth), luma and
  // chroma.  `jobs` is a host array of `count` descriptors.
  Status modes_predict(const intra_modes::IntraJob* jobs, int count,
                       cudaStream_t stream = nullptr);

  // §7.13.2.1/.7/.8 multiple reference lines (luma directional, mrl_index 0..3
  // plus the optional secondary-line blend).
  Status mrl_predict(const mrl::MrlJob* jobs, int count,
                     cudaStream_t stream = nullptr);

  // §7.13.2.7/.9/.12 intra bi-prediction.  Jobs must be DC or directional.
  Status ibp_predict(const ibp::IbpJob* jobs, int count,
                     cudaStream_t stream = nullptr);

  // §7.13.2.3-.6 data-driven intra prediction (luma, 8-bit).
  Status dip_predict(const dip::DipJob* jobs, int count,
                     cudaStream_t stream = nullptr);

  // §7.13.5 chroma-from-luma, explicit alpha (`CflJob.alpha_q8`).
  Status cfl_predict(const CflJob* jobs, int count,
                     cudaStream_t stream = nullptr);

  // §7.13.5 CFL_DERIVED_ALPHA.  `d_alpha_out` may be null; otherwise it is a
  // device array of `count` ints receiving the applied Q8 alpha.
  Status cfl_predict_derived(const cfl::Block* blocks, int* d_alpha_out,
                             int count, cudaStream_t stream = nullptr);

  // §5.20.8/§7.13.4 palette: nearest-colour map (with edge replication),
  // prediction and optional §5.20.8.5 colour contexts for `count` blocks.
  // `contexts` may be null.  Palette blocks are capped at 64x64.
  Status palette_apply(const PaletteDeviceBlock* blocks,
                       const PaletteContextDevice* contexts, int count,
                       cudaStream_t stream = nullptr);

  // §7.13.4/§6.19.7.12 IntraBC.  `ibc_copy` applies a validated block vector;
  // `ibc_sad` / `ibc_validate` stage a host array of candidate BVs in the
  // arena and write device results (`d_sads` / `d_valid`).
  Status ibc_copy(const IbcDeviceFrame& frame, int mi_row, int mi_col,
                  int bsize, av2enc::ibc::BlockVector bv,
                  std::uint8_t* dst_y, int dst_stride_y,
                  std::uint8_t* dst_u, int dst_stride_u,
                  std::uint8_t* dst_v, int dst_stride_v, int visible_w,
                  int visible_h, cudaStream_t stream = nullptr);
  Status ibc_sad(const IbcDeviceFrame& frame, int mi_row, int mi_col,
                 int bsize, const av2enc::ibc::BlockVector* host_bvs, int n,
                 std::int64_t* d_sads, cudaStream_t stream = nullptr);
  Status ibc_validate(const IbcDeviceGeometry& geometry, int mi_row,
                      int mi_col, int bsize,
                      const av2enc::ibc::BlockVector* host_bvs, int n,
                      std::uint8_t* d_valid, cudaStream_t stream = nullptr);

 private:
  Status stage_bytes(const void* host, std::size_t bytes,
                     cudaStream_t stream);

  Arena* arena_ = nullptr;
  Config config_{};
  void* staging_ = nullptr;
  intra_full::Workspace supplemental_{};
};

inline Status Workspace::stage_bytes(const void* host, std::size_t bytes,
                                     cudaStream_t stream) {
  if (bytes > detail::staging_bytes(config_)) {
    return Status::kInvalidArgument;
  }
  return status_from_cuda(
      cudaMemcpyAsync(staging_, host, bytes, cudaMemcpyHostToDevice, stream));
}

inline Status Workspace::synchronize(cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  return status_from_cuda(cudaStreamSynchronize(stream));
}

inline Status Workspace::init(Arena& arena, const Config& config,
                              std::string* error,
                              std::uint64_t* required_out) {
  arena_ = nullptr;
  config_ = Config{};
  staging_ = nullptr;
  supplemental_ = intra_full::Workspace{};

  if (!detail::config_valid(config)) {
    if (required_out != nullptr) *required_out = 0;
    if (error != nullptr) *error = "intra_workspace: invalid configuration";
    return Status::kInvalidArgument;
  }
  const std::uint64_t need = required_device_bytes(config);
  if (required_out != nullptr) *required_out = need;
  if (!arena.initialized()) {
    if (error != nullptr) *error = "intra_workspace: arena is not initialized";
    return Status::kNotInitialized;
  }
  if (arena.remaining() < need) {
    if (error != nullptr) {
      *error = "intra_workspace: arena needs " + std::to_string(need) +
               " bytes, " + std::to_string(arena.remaining()) + " available";
    }
    return Status::kArenaExhausted;
  }

  const std::uint64_t mark = arena.mark();
  staging_ = arena.allocate_array<std::uint8_t>(
      static_cast<std::size_t>(detail::staging_bytes(config)));
  if (staging_ == nullptr) {
    arena.rewind(mark);
    if (error != nullptr) {
      *error = "intra_workspace: arena exhausted staging " +
               std::to_string(detail::staging_bytes(config)) + " bytes";
    }
    return Status::kArenaExhausted;
  }
  std::uint64_t supplemental_need = 0;
  const Status status =
      supplemental_.init(arena, config.supplemental, error, &supplemental_need);
  if (!is_ok(status)) {
    arena.rewind(mark);
    staging_ = nullptr;
    return status;
  }
  config_ = config;
  arena_ = &arena;
  return Status::kOk;
}

}  // namespace av2enc::cuda::intra_workspace

#endif  // AV2ENC_CUDA_INTRA_WORKSPACE_INTRA_WORKSPACE_H_
