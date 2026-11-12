// Host-side arena plumbing for the AV2 morphological prediction component.
//
// This file performs no arithmetic and no device allocation: it asks the
// caller's persistent av2enc::cuda::Arena for the arrays and uses only
// cudaMemcpy/cudaStreamSynchronize plus the kernel entry points. It is
// compiled by nvcc in the real build and by the host compiler against a CUDA
// stub in the emulation build.

#include "cuda/codec/inter/morph/morph.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace av2enc::cuda::morph {
namespace {

constexpr std::size_t kAlign = Arena::kAlignment;

std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// Aligned byte size of `count` elements, 0 on overflow/zero count.
std::size_t array_bytes(std::size_t element_size, int count) {
  if (count <= 0) {
    return 0;
  }
  const std::size_t n = static_cast<std::size_t>(count);
  if (element_size > std::numeric_limits<std::size_t>::max() / n) {
    return 0;
  }
  return align_up(element_size * n, kAlign);
}

std::size_t field_bytes(int samples) {
  if (samples <= 0) {
    return 0;
  }
  const std::size_t n = static_cast<std::size_t>(samples);
  if (n > std::numeric_limits<std::size_t>::max() / (2 * sizeof(std::int16_t))) {
    return 0;
  }
  return align_up(n * 2 * sizeof(std::int16_t), kAlign);
}

void set_error(std::string* error, const std::string& text) {
  if (error != nullptr) {
    *error = text;
  }
}

}  // namespace

std::size_t required_device_bytes(const MorphArenaConfig& config) {
  if (config.max_jobs <= 0 || config.max_field_samples < 0) {
    return 0;
  }
  const std::size_t jobs = array_bytes(sizeof(MorphJob), config.max_jobs);
  const std::size_t fits = array_bytes(sizeof(MorphFit), config.max_jobs);
  const std::size_t statuses = array_bytes(sizeof(int), config.max_jobs);
  const std::size_t field = field_bytes(config.max_field_samples);
  if (jobs == 0 || fits == 0 || statuses == 0 ||
      (config.max_field_samples > 0 && field == 0)) {
    return 0;
  }
  if (jobs > std::numeric_limits<std::size_t>::max() - fits ||
      jobs + fits > std::numeric_limits<std::size_t>::max() - statuses ||
      jobs + fits + statuses >
          std::numeric_limits<std::size_t>::max() - field) {
    return 0;
  }
  return jobs + fits + statuses + field;
}

std::size_t required_scratch_bytes(int job_count) {
  if (job_count < 0) {
    return 0;
  }
  const std::size_t jobs = array_bytes(sizeof(MorphJob), job_count);
  const std::size_t fits = array_bytes(sizeof(MorphFit), job_count);
  const std::size_t statuses = array_bytes(sizeof(int), job_count);
  return jobs + fits + statuses;
}

Status MorphWorkspace::init(Arena& arena, const MorphArenaConfig& config,
                            std::string* error) {
  jobs_ = nullptr;
  fits_ = nullptr;
  statuses_ = nullptr;
  field_ = nullptr;
  capacity_ = 0;
  field_samples_ = 0;
  required_bytes_ = 0;
  error_.clear();

  if (config.max_jobs <= 0 || config.max_field_samples < 0) {
    error_ = "morph workspace: max_jobs must be > 0 and max_field_samples >= 0";
    set_error(error, error_);
    return Status::kInvalidArgument;
  }
  const std::size_t required = required_device_bytes(config);
  required_bytes_ = required;
  if (required == 0) {
    error_ = "morph workspace: size overflow";
    set_error(error, error_);
    return Status::kInvalidArgument;
  }
  if (!arena.initialized()) {
    error_ = "morph workspace: arena is not initialized";
    set_error(error, error_);
    return Status::kNotInitialized;
  }
  if (arena.remaining() < required) {
    error_ = "morph workspace: arena exhausted; required " +
             std::to_string(required) + " bytes, " +
             std::to_string(arena.remaining()) + " available";
    set_error(error, error_);
    return Status::kArenaExhausted;
  }

  const std::uint64_t mark = arena.mark();
  MorphJob* jobs = arena.allocate_array<MorphJob>(
      static_cast<std::size_t>(config.max_jobs));
  MorphFit* fits = arena.allocate_array<MorphFit>(
      static_cast<std::size_t>(config.max_jobs));
  int* statuses = arena.allocate_array<int>(
      static_cast<std::size_t>(config.max_jobs));
  std::int16_t* field =
      config.max_field_samples > 0
          ? arena.allocate_array<std::int16_t>(
                static_cast<std::size_t>(config.max_field_samples) * 2)
          : nullptr;
  const std::uint64_t after = arena.mark();
  const std::uint64_t consumed = after - mark;
  const bool field_ok = config.max_field_samples == 0 || field != nullptr;
  if (jobs == nullptr || fits == nullptr || statuses == nullptr || !field_ok ||
      consumed != static_cast<std::uint64_t>(required)) {
    arena.rewind(mark);
    error_ = "morph workspace: arena allocation failed; required " +
             std::to_string(required) + " bytes";
    set_error(error, error_);
    return Status::kArenaExhausted;
  }

  jobs_ = jobs;
  fits_ = fits;
  statuses_ = statuses;
  field_ = field;
  capacity_ = config.max_jobs;
  field_samples_ = static_cast<std::size_t>(config.max_field_samples);
  return Status::kOk;
}

Status morph_upload_jobs(MorphWorkspace& workspace, const MorphJob* host_jobs,
                         int count, cudaStream_t stream) {
  if (!workspace.initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > workspace.capacity()) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  if (host_jobs == nullptr) {
    return Status::kInvalidArgument;
  }
  const cudaError_t error = cudaMemcpyAsync(
      workspace.jobs(), host_jobs,
      static_cast<std::size_t>(count) * sizeof(MorphJob),
      cudaMemcpyHostToDevice, stream);
  return status_from_cuda(error);
}

Status morph_run_batch(const MorphWorkspace& workspace, int count,
                       cudaStream_t stream) {
  if (!workspace.initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > workspace.capacity()) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  return status_from_cuda(morph_predict_batch_persist(
      workspace.jobs(), workspace.fits(), workspace.statuses(), count, stream));
}

Status morph_stage_and_run(Arena& arena, const MorphJob* host_jobs,
                           int job_count, MorphStagedResult* result,
                           cudaStream_t stream,
                           std::size_t* scratch_bytes_out) {
  if (scratch_bytes_out != nullptr) {
    *scratch_bytes_out = 0;
  }
  if (job_count < 0 || (job_count > 0 && host_jobs == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (job_count == 0) {
    return Status::kOk;
  }
  const std::size_t required = required_scratch_bytes(job_count);
  if (scratch_bytes_out != nullptr) {
    *scratch_bytes_out = required;
  }
  if (required == 0) {
    return Status::kInvalidArgument;
  }
  if (!arena.initialized()) {
    return Status::kNotInitialized;
  }
  if (arena.remaining() < required) {
    return Status::kArenaExhausted;
  }

  const std::uint64_t mark = arena.mark();
  MorphJob* jobs = arena.allocate_array<MorphJob>(
      static_cast<std::size_t>(job_count));
  MorphFit* fits = arena.allocate_array<MorphFit>(
      static_cast<std::size_t>(job_count));
  int* statuses =
      arena.allocate_array<int>(static_cast<std::size_t>(job_count));
  if (jobs == nullptr || fits == nullptr || statuses == nullptr ||
      arena.mark() - mark != static_cast<std::uint64_t>(required)) {
    arena.rewind(mark);
    return Status::kArenaExhausted;
  }

  cudaError_t cuda_error = cudaMemcpyAsync(
      jobs, host_jobs, static_cast<std::size_t>(job_count) * sizeof(MorphJob),
      cudaMemcpyHostToDevice, stream);
  if (cuda_error != cudaSuccess) {
    arena.rewind(mark);
    return status_from_cuda(cuda_error);
  }
  cuda_error = morph_predict_batch_persist(jobs, fits, statuses, job_count,
                                           stream);
  if (cuda_error != cudaSuccess) {
    (void)cudaStreamSynchronize(stream);
    arena.rewind(mark);
    return status_from_cuda(cuda_error);
  }
  // The scratch lives in the arena, so it cannot be rewound until every
  // asynchronous read/write has completed.
  cuda_error = cudaStreamSynchronize(stream);
  if (cuda_error != cudaSuccess) {
    arena.rewind(mark);
    return status_from_cuda(cuda_error);
  }
  if (result != nullptr) {
    if (result->host_status != nullptr) {
      cuda_error = cudaMemcpy(
          result->host_status, statuses,
          static_cast<std::size_t>(job_count) * sizeof(int),
          cudaMemcpyDeviceToHost);
      if (cuda_error != cudaSuccess) {
        arena.rewind(mark);
        return status_from_cuda(cuda_error);
      }
    }
    if (result->host_fits != nullptr) {
      cuda_error = cudaMemcpy(
          result->host_fits, fits,
          static_cast<std::size_t>(job_count) * sizeof(MorphFit),
          cudaMemcpyDeviceToHost);
      if (cuda_error != cudaSuccess) {
        arena.rewind(mark);
        return status_from_cuda(cuda_error);
      }
    }
  }
  arena.rewind(mark);
  return Status::kOk;
}

}  // namespace av2enc::cuda::morph
