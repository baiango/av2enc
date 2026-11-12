#ifndef AV2ENC_CUDA_MORPH_MORPH_H_
#define AV2ENC_CUDA_MORPH_MORPH_H_

// AV2 v1.0.0 morphological prediction (§7.13.3.26, which invokes the
// §7.13.3.25 block adaptive weighted prediction process with morphPred = 1)
// on CUDA.
//
// The CPU reference is src/encoder/inter/morph/morph_common.h. This component
// does not reinterpret the process: it derives the block's full-pel reference
// origin from the decoded 1/8-pel block vector, fits alpha/beta from the
// reconstructed above/left template, then adapts every w*h sample with
//
//     clip1((orig * alpha + beta) >> 8, 8)
//
// The fit depends on the neighbouring samples (AvailU/AvailL) and on the
// reference block selected by the motion vector; there is no dense motion
// field in the CPU reference. The optional `field_out` below materialises the
// per-sample full-pel displacement that the predictor uses, which is the
// block-constant part of that field, for callers that need it. It never
// affects the prediction.
//
// Arithmetic lives once in morph_device.cuh, annotated for both host and
// device. The CPU emulation test compiles that header as host code and
// compares it byte-for-byte with morph_common.h; the .cu test does the same on
// the device.
//
// Memory contract
// ---------------
// Every device buffer comes from the caller's single persistent
// `av2enc::cuda::Arena`. Nothing here calls cudaMalloc/cudaFree/cudaMallocHost.
// Use MorphWorkspace for the production path: it allocates the persistent
// descriptor/fit/status/field arrays once, then upload+launch repeatedly with
// no allocation. required_device_bytes() reports the exact persistent size; a
// short arena returns Status::kArenaExhausted and the caller grows the arena.
// morph_stage_and_run() is the mark()/rewind() scratch path; it allocates its
// staging descriptors from the arena inside the call and rewinds before
// returning.

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "cuda/error.h"
#include "codec/inter/morph/morph_common.h"

namespace av2enc::cuda::morph {

using inter::morph::MorphContext;
using inter::morph::MorphFit;
using inter::morph::MorphRect;

// One predicted block. All pointers are device pointers. `frame` is the luma
// plane that already contains the IBC prediction; it is adapted in place, as
// in the CPU reference.
struct MorphJob {
  MorphContext ctx{};
  MorphRect rect{};
  int mv_row = 0;  // 1/8-pel, as decoded
  int mv_col = 0;
  std::uint8_t* frame = nullptr;
  int stride = 0;
  // Optional outputs, written only on success. May be null.
  MorphFit* fit_out = nullptr;
  int* status_out = nullptr;  // 1 = predicted, 0 = §7.13.3.25 box rejected
  // Optional per-sample full-pel field: 2 * rect.w * rect.h int16 entries
  // (dx, dy). Leave null to skip the writes.
  std::int16_t* field_out = nullptr;
};

static_assert(std::is_trivially_copyable<MorphJob>::value,
              "MorphJob is passed by value as a kernel parameter");
static_assert(std::is_trivially_copyable<MorphFit>::value,
              "MorphFit is copied between host and device");

// ---------------------------------------------------------------------------
// Persistent arena workspace
// ---------------------------------------------------------------------------

// `max_jobs` bounds one batch (batch descriptors = the block-level motion
// field: one decoded vector + rect + template context per predicted block).
// `max_field_samples` sizes the optional per-sample displacement field; each
// sample consumes two int16. Zero disables the field buffer.
struct MorphArenaConfig {
  int max_jobs = 1;
  int max_field_samples = 0;
};

// Exact number of arena bytes MorphWorkspace::init() consumes for `config`.
// Every array is 256-byte aligned, matching Arena's bump allocation.
std::size_t required_device_bytes(const MorphArenaConfig& config);

// Exact scratch size used by morph_stage_and_run(): job descriptors, fit
// outputs and status outputs, each 256-byte aligned.
std::size_t required_scratch_bytes(int job_count);

class MorphWorkspace {
 public:
  MorphWorkspace() = default;
  MorphWorkspace(const MorphWorkspace&) = delete;
  MorphWorkspace& operator=(const MorphWorkspace&) = delete;

  // Allocates the persistent arrays from `arena` once. On any failure the
  // arena bump pointer is restored and `required_bytes()` still reports the
  // size the caller must make available (Status::kArenaExhausted).
  Status init(Arena& arena, const MorphArenaConfig& config,
              std::string* error = nullptr);

  bool initialized() const noexcept { return jobs_ != nullptr; }
  int capacity() const noexcept { return capacity_; }
  std::size_t required_bytes() const noexcept { return required_bytes_; }
  std::size_t field_samples() const noexcept { return field_samples_; }
  const char* last_error() const noexcept { return error_.c_str(); }

  MorphJob* jobs() const noexcept { return jobs_; }
  MorphFit* fits() const noexcept { return fits_; }
  int* statuses() const noexcept { return statuses_; }
  std::int16_t* field() const noexcept { return field_; }

 private:
  MorphJob* jobs_ = nullptr;
  MorphFit* fits_ = nullptr;
  int* statuses_ = nullptr;
  std::int16_t* field_ = nullptr;
  int capacity_ = 0;
  std::size_t field_samples_ = 0;
  std::size_t required_bytes_ = 0;
  std::string error_;
};

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

// One block. No allocation, no host synchronization; returns the launch error.
cudaError_t morph_predict(const MorphJob& job, cudaStream_t stream = nullptr);

// `d_jobs` is a device array of `job_count` descriptors, e.g. from a
// MorphWorkspace or another arena allocation. Each job's own fit_out /
// status_out pointers are honoured. `job_count == 0` is a no-op.
cudaError_t morph_predict_batch(const MorphJob* d_jobs, int job_count,
                                cudaStream_t stream = nullptr);

// Batch with persistent output arrays: the kernel writes fits[i]/status[i]
// instead of the per-job pointers. Pass null to disable either output.
cudaError_t morph_predict_batch_persist(const MorphJob* d_jobs,
                                        MorphFit* d_fits, int* d_statuses,
                                        int job_count,
                                        cudaStream_t stream = nullptr);

// Uploads `count` host descriptors into the workspace's persistent job array.
// Asynchronous on `stream`; the caller must keep `host_jobs` alive until it
// completes (the stream order is enough).
Status morph_upload_jobs(MorphWorkspace& workspace, const MorphJob* host_jobs,
                         int count, cudaStream_t stream = nullptr);

// Launches a workspace batch. The kernel writes workspace.fits()/statuses(),
// so the per-job fit_out/status_out pointers in the uploaded descriptors are
// ignored. `count <= workspace.capacity()`.
Status morph_run_batch(const MorphWorkspace& workspace, int count,
                       cudaStream_t stream = nullptr);

// Host-side result sinks for morph_stage_and_run().
struct MorphStagedResult {
  int* host_status = nullptr;  // count ints, optional
  MorphFit* host_fits = nullptr;  // count entries, optional
};

// Scratch path: allocates count descriptors plus fit/status arrays at
// arena.mark(), uploads the host descriptors, launches, synchronizes `stream`
// so the scratch can be rewound, copies the optional results back, then
// rewinds. On an undersized arena returns Status::kArenaExhausted with
// `scratch_bytes_out` set to required_scratch_bytes(job_count) and leaves the
// arena untouched. Production callers should prefer MorphWorkspace.
Status morph_stage_and_run(Arena& arena, const MorphJob* host_jobs,
                           int job_count, MorphStagedResult* result,
                           cudaStream_t stream = nullptr,
                           std::size_t* scratch_bytes_out = nullptr);

}  // namespace av2enc::cuda::morph

#endif  // AV2ENC_CUDA_MORPH_MORPH_H_
