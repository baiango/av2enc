#ifndef AV2ENC_CUDA_WARP_STACK_WARP_STACK_H_
#define AV2ENC_CUDA_WARP_STACK_WARP_STACK_H_

// CUDA warp stack: batched prediction, corner generation and sample collection.
//
// Product target: Linux + CUDA, sm_86 primary. This is component code; it does
// not edit or replace the CPU stack builder in
// `src/encoder/inter/warp_stack/`.
//
// Design contract
// ---------------
//  * All device memory comes from one persistent `av2enc::cuda::Arena`.
//    `required_device_bytes(config)` reports the exact workspace footprint;
//    `Workspace::create()` bump-allocates it once. There is no cudaMalloc,
//    cudaFree, cudaMallocHost or per-launch allocation in this module.
//  * Launchers (`run_predict` / `*_launch`) only enqueue kernels. They do not
//    allocate, copy, synchronise or call cudaDeviceSynchronize.
//  * Upload helpers use cudaMemcpyAsync into arena-backed descriptor tables.
//    They are separate from the launchers.
//  * An undersized arena is reported as `Status::kArenaExhausted` together with
//    `required_bytes()`; the caller grows the arena and retries.
//
// Stack ordering stays on the CPU. The GPU produces `WarpCornerResult` and
// `WarpSampleSet` tables; the existing builder consumes them in the same order
// it would have called the CPU routines and still performs
// insert_warp_candidate()/sorting/clamping.

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "../arena.h"
#include "../error.h"
#include "warp_stack_types.h"

namespace av2enc::cuda::warpstack {

static_assert(Arena::kAlignment == kWarpStackAlignment,
              "required_device_bytes() mirrors Arena::allocate_array() rounding");

// ---------------------------------------------------------------------------
// Raw launchers (device job arrays; no allocation, no copy, no sync)
// ---------------------------------------------------------------------------

// One CUDA block per prediction job. `results` may be null.
Status predict_launch(const WarpPredictJob* device_jobs, WarpPredictResult* results, int count,
                      cudaStream_t stream = nullptr);

// One thread per corner job. `results[i]` receives job i.
Status corner_launch(const WarpCornerJob* device_jobs, WarpCornerResult* results, int count,
                     cudaStream_t stream = nullptr);

// One thread per sample job.
Status sample_launch(const WarpSampleJob* device_jobs, inter::warp::WarpSampleSet* results,
                     int count, cudaStream_t stream = nullptr);

// One thread per least-squares estimate job.
Status estimate_launch(const WarpEstimateJob* device_jobs, ia::WarpParams* results, int count,
                       cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Arena-backed batch workspace
// ---------------------------------------------------------------------------

class Workspace {
 public:
  Workspace() = default;
  ~Workspace() = default;
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  // Allocates the persistent descriptor/result tables from `arena`. The arena
  // must contain at least `required_device_bytes(config)` bytes. On
  // kArenaExhausted the caller should read `required_bytes()`.
  Status create(Arena& arena, const WarpStackConfig& config, std::string* error = nullptr) {
    if (created_) {
      return Status::kAlreadyInitialized;
    }
    if (!arena.initialized()) {
      return Status::kNotInitialized;
    }
    if (config.max_predict_jobs < 0 || config.max_corner_jobs < 0 ||
        config.max_sample_jobs < 0 || config.max_estimate_jobs < 0 ||
        config.max_mi_cells < 0) {
      return Status::kInvalidArgument;
    }
    config_ = config;
    required_bytes_ = required_device_bytes(config);
    if (arena.remaining() < required_bytes_) {
      if (error != nullptr) {
        *error = "warp_stack workspace needs " + std::to_string(required_bytes_) +
                 " bytes, arena has " + std::to_string(arena.remaining());
      }
      return Status::kArenaExhausted;
    }

    arena_ = &arena;
    predict_jobs_ = arena.allocate_array<WarpPredictJob>(static_cast<std::size_t>(config.max_predict_jobs));
    predict_results_ = arena.allocate_array<WarpPredictResult>(static_cast<std::size_t>(config.max_predict_jobs));
    corner_jobs_ = arena.allocate_array<WarpCornerJob>(static_cast<std::size_t>(config.max_corner_jobs));
    corner_results_ = arena.allocate_array<WarpCornerResult>(static_cast<std::size_t>(config.max_corner_jobs));
    sample_jobs_ = arena.allocate_array<WarpSampleJob>(static_cast<std::size_t>(config.max_sample_jobs));
    sample_sets_ = arena.allocate_array<inter::warp::WarpSampleSet>(static_cast<std::size_t>(config.max_sample_jobs));
    estimate_jobs_ = arena.allocate_array<WarpEstimateJob>(static_cast<std::size_t>(config.max_estimate_jobs));
    estimate_params_ = arena.allocate_array<ia::WarpParams>(static_cast<std::size_t>(config.max_estimate_jobs));
    mi_grid_ = arena.allocate_array<inter::warp::WarpMi>(static_cast<std::size_t>(config.max_mi_cells));

    if (config.max_predict_jobs > 0 && (predict_jobs_ == nullptr || predict_results_ == nullptr)) {
      reset();
      return Status::kArenaExhausted;
    }
    if (config.max_corner_jobs > 0 && (corner_jobs_ == nullptr || corner_results_ == nullptr)) {
      reset();
      return Status::kArenaExhausted;
    }
    if (config.max_sample_jobs > 0 && (sample_jobs_ == nullptr || sample_sets_ == nullptr)) {
      reset();
      return Status::kArenaExhausted;
    }
    if (config.max_estimate_jobs > 0 &&
        (estimate_jobs_ == nullptr || estimate_params_ == nullptr)) {
      reset();
      return Status::kArenaExhausted;
    }
    if (config.max_mi_cells > 0 && mi_grid_ == nullptr) {
      reset();
      return Status::kArenaExhausted;
    }

    allocated_bytes_ = required_bytes_;
    created_ = true;
    return Status::kOk;
  }

  // Forget the pointers. Does not free arena memory (use Arena::rewind(mark)).
  void reset() noexcept {
    arena_ = nullptr;
    config_ = WarpStackConfig{};
    required_bytes_ = 0;
    allocated_bytes_ = 0;
    created_ = false;
    predict_jobs_ = nullptr;
    predict_results_ = nullptr;
    corner_jobs_ = nullptr;
    corner_results_ = nullptr;
    sample_jobs_ = nullptr;
    sample_sets_ = nullptr;
    estimate_jobs_ = nullptr;
    estimate_params_ = nullptr;
    mi_grid_ = nullptr;
    mi_cols_ = 0;
    mi_rows_ = 0;
  }

  bool created() const noexcept { return created_; }
  const WarpStackConfig& config() const noexcept { return config_; }
  std::uint64_t required_bytes() const noexcept { return required_bytes_; }
  std::uint64_t allocated_bytes() const noexcept { return allocated_bytes_; }

  // Persistent tables. Sizes are `config().max_*_jobs`; the MI grid table is
  // `config().max_mi_cells`.
  WarpPredictJob* predict_jobs() const noexcept { return predict_jobs_; }
  WarpPredictResult* predict_results() const noexcept { return predict_results_; }
  WarpCornerJob* corner_jobs() const noexcept { return corner_jobs_; }
  WarpCornerResult* corner_results() const noexcept { return corner_results_; }
  WarpSampleJob* sample_jobs() const noexcept { return sample_jobs_; }
  inter::warp::WarpSampleSet* sample_sets() const noexcept { return sample_sets_; }
  WarpEstimateJob* estimate_jobs() const noexcept { return estimate_jobs_; }
  ia::WarpParams* estimate_params() const noexcept { return estimate_params_; }
  inter::warp::WarpMi* mi_grid() const noexcept { return mi_grid_; }

  // Upload host descriptor tables into the arena-backed device tables. These
  // are the only entry points that copy; they do no allocation.
  Status upload_predict(const WarpPredictJob* host_jobs, int count,
                        cudaStream_t stream = nullptr);
  Status upload_corner(const WarpCornerJob* host_jobs, int count,
                       cudaStream_t stream = nullptr);
  Status upload_sample(const WarpSampleJob* host_jobs, int count,
                       cudaStream_t stream = nullptr);
  Status upload_estimate(const WarpEstimateJob* host_jobs, int count,
                         cudaStream_t stream = nullptr);

  // Upload an MI grid into the optional arena table and remember its shape.
  // Inline so the host-emulation test can exercise the exact API path with the
  // CUDA shim (the kernel path is identical).
  Status upload_mi_grid(const inter::warp::WarpMi* host_mi, int mi_cols, int mi_rows,
                        cudaStream_t stream = nullptr) {
    if (!created_) {
      return Status::kNotInitialized;
    }
    if (host_mi == nullptr || mi_cols <= 0 || mi_rows <= 0 || mi_grid_ == nullptr) {
      return Status::kInvalidArgument;
    }
    const std::int64_t cells = std::int64_t{mi_cols} * mi_rows;
    if (cells > config_.max_mi_cells) {
      return Status::kInvalidArgument;
    }
    const Status status = status_from_cuda(
        cudaMemcpyAsync(mi_grid_, host_mi,
                        static_cast<std::size_t>(cells) * sizeof(inter::warp::WarpMi),
                        cudaMemcpyHostToDevice, stream));
    if (status == Status::kOk) {
      mi_cols_ = mi_cols;
      mi_rows_ = mi_rows;
    }
    return status;
  }

  inter::warp::WarpGrid grid() const noexcept {
    inter::warp::WarpGrid out;
    out.mi_cols = mi_cols_;
    out.mi_rows = mi_rows_;
    out.mi = mi_grid_;
    return out;
  }

  // Download result tables for the CPU-side stack builder. These are copies
  // out of the arena; they do not allocate and do not synchronise.
  Status download_predict(WarpPredictResult* host_results, int count,
                          cudaStream_t stream = nullptr) const;
  Status download_corner(WarpCornerResult* host_results, int count,
                         cudaStream_t stream = nullptr) const;
  Status download_sample(inter::warp::WarpSampleSet* host_results, int count,
                         cudaStream_t stream = nullptr) const;
  Status download_estimate(ia::WarpParams* host_results, int count,
                           cudaStream_t stream = nullptr) const;

  // Upload + launch convenience. Still no allocation/sync; the copy is a plain
  // cudaMemcpyAsync into the persistent descriptor table.
  Status submit_predict(const WarpPredictJob* host_jobs, int count,
                        cudaStream_t stream = nullptr);
  Status submit_corner(const WarpCornerJob* host_jobs, int count,
                       cudaStream_t stream = nullptr);
  Status submit_sample(const WarpSampleJob* host_jobs, int count,
                       cudaStream_t stream = nullptr);
  Status submit_estimate(const WarpEstimateJob* host_jobs, int count,
                         cudaStream_t stream = nullptr);

  // Launchers over the persistent tables. No allocation, no copy, no sync.
  Status run_predict(int count, cudaStream_t stream = nullptr);
  Status run_corner(int count, cudaStream_t stream = nullptr);
  Status run_sample(int count, cudaStream_t stream = nullptr);
  Status run_estimate(int count, cudaStream_t stream = nullptr);

 private:
  template <typename T>
  Status copy_in(T* device_dst, const T* host_src, int count, int capacity,
                 cudaStream_t stream);

  Arena* arena_ = nullptr;
  WarpStackConfig config_{};
  std::uint64_t required_bytes_ = 0;
  std::uint64_t allocated_bytes_ = 0;
  bool created_ = false;

  WarpPredictJob* predict_jobs_ = nullptr;
  WarpPredictResult* predict_results_ = nullptr;
  WarpCornerJob* corner_jobs_ = nullptr;
  WarpCornerResult* corner_results_ = nullptr;
  WarpSampleJob* sample_jobs_ = nullptr;
  inter::warp::WarpSampleSet* sample_sets_ = nullptr;
  WarpEstimateJob* estimate_jobs_ = nullptr;
  ia::WarpParams* estimate_params_ = nullptr;
  inter::warp::WarpMi* mi_grid_ = nullptr;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
};

}  // namespace av2enc::cuda::warpstack

#endif  // AV2ENC_CUDA_WARP_STACK_WARP_STACK_H_
