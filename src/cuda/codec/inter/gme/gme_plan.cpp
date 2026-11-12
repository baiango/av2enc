// Host-side arena, sizing, staging and download layer for the CUDA GME
// module.  No kernels and no direct device allocation live here, so the same
// translation unit is compiled by the no-nvcc emulation parity test.
#include "cuda/codec/inter/gme/gme.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace av2enc::cuda::gme {
namespace {

std::uint64_t align_up(std::uint64_t bytes) noexcept {
  const std::uint64_t alignment =
      static_cast<std::uint64_t>(Arena::kAlignment);
  return (bytes + alignment - 1) & ~(alignment - 1);
}

void set_error(std::string* error, const char* context,
               const std::string& detail) {
  if (error != nullptr) {
    *error = std::string(context) + ": " + detail;
  }
}

bool buffers_ready(const GmeBuffers& buffers) noexcept {
  return buffers.d_points != nullptr && buffers.d_active != nullptr &&
         buffers.d_residuals != nullptr && buffers.d_keep != nullptr &&
         buffers.d_blocks != nullptr && buffers.d_warps != nullptr &&
         buffers.d_results != nullptr && buffers.d_control != nullptr &&
         buffers.max_points > 0 && buffers.max_blocks > 0 &&
         buffers.max_warps > 0 && buffers.max_jobs > 0;
}

bool valid_config(const GmeConfig& config) noexcept {
  return config.max_points >= 1 && config.max_points <= kGmeMaxCount &&
         config.max_blocks >= 1 && config.max_blocks <= kGmeMaxCount &&
         config.max_warps >= 1 && config.max_warps <= kGmeMaxCount &&
         config.max_jobs >= 1 && config.max_jobs <= kGmeMaxCount;
}

bool valid_job(const GmeBuffers& buffers, int job) noexcept {
  return job >= 0 && job < buffers.max_jobs;
}

std::uint64_t count_times(std::uint64_t count,
                          std::uint64_t element) noexcept {
  return align_up(count * element);
}

}  // namespace

std::uint64_t required_device_bytes(const GmeConfig& config) noexcept {
  if (!valid_config(config)) return 0;
  const std::uint64_t jobs = static_cast<std::uint64_t>(config.max_jobs);
  const std::uint64_t points = jobs * static_cast<std::uint64_t>(config.max_points);
  const std::uint64_t blocks = jobs * static_cast<std::uint64_t>(config.max_blocks);
  const std::uint64_t warps = jobs * static_cast<std::uint64_t>(config.max_warps);

  return count_times(points, sizeof(GmePoint)) +
         count_times(points, sizeof(GmePoint)) +
         count_times(points, sizeof(GmeI64)) +
         count_times(points, sizeof(GmeU8)) +
         count_times(blocks, sizeof(GmeBlockMv)) +
         count_times(warps, sizeof(GmeWarpSample)) +
         count_times(jobs, sizeof(GmeEstimate)) +
         count_times(jobs, sizeof(GmeJobControl));
}

const char* gme_status_name(GmeStatus status) noexcept {
  switch (status) {
    case GmeStatus::kOk:
      return "ok";
    case GmeStatus::kInvalidArgument:
      return "invalid argument";
    case GmeStatus::kNotInitialized:
      return "not initialized";
    case GmeStatus::kArenaTooSmall:
      return "arena too small";
    case GmeStatus::kCudaError:
      return "CUDA error";
  }
  return "unknown status";
}

GmeStatus allocate_checked(Arena& arena, const GmeConfig& config,
                           GmeBuffers* out,
                           std::uint64_t* required_bytes,
                           std::string* error) {
  if (required_bytes != nullptr) *required_bytes = 0;
  if (out == nullptr) {
    set_error(error, "allocate_checked", "null output");
    return GmeStatus::kInvalidArgument;
  }
  *out = GmeBuffers{};

  const std::uint64_t need = required_device_bytes(config);
  if (need == 0) {
    set_error(error, "allocate_checked", "configuration out of range");
    return GmeStatus::kInvalidArgument;
  }
  if (required_bytes != nullptr) *required_bytes = need;
  if (!arena.initialized()) {
    set_error(error, "allocate_checked", "arena is not initialized");
    return GmeStatus::kNotInitialized;
  }
  if (arena.remaining() < need) {
    set_error(error, "allocate_checked",
              "arena too small: required " + std::to_string(need) +
                  " bytes, remaining " + std::to_string(arena.remaining()));
    return GmeStatus::kArenaTooSmall;
  }

  const std::uint64_t arena_mark = arena.mark();
  const std::uint64_t jobs = static_cast<std::uint64_t>(config.max_jobs);
  const std::uint64_t points =
      jobs * static_cast<std::uint64_t>(config.max_points);
  const std::uint64_t blocks =
      jobs * static_cast<std::uint64_t>(config.max_blocks);
  const std::uint64_t warps =
      jobs * static_cast<std::uint64_t>(config.max_warps);

  bool ok = true;
  void* raw = nullptr;
  auto allocate = [&](std::uint64_t bytes) -> void* {
    raw = nullptr;
    if (arena.allocate_checked(static_cast<std::size_t>(bytes), &raw) !=
        Status::kOk) {
      ok = false;
      return nullptr;
    }
    return raw;
  };

  out->d_points = static_cast<GmePoint*>(
      allocate(points * sizeof(GmePoint)));
  if (ok) {
    out->d_active = static_cast<GmePoint*>(
        allocate(points * sizeof(GmePoint)));
  }
  if (ok) {
    out->d_residuals = static_cast<GmeI64*>(
        allocate(points * sizeof(GmeI64)));
  }
  if (ok) {
    out->d_keep = static_cast<GmeU8*>(allocate(points * sizeof(GmeU8)));
  }
  if (ok) {
    out->d_blocks = static_cast<GmeBlockMv*>(
        allocate(blocks * sizeof(GmeBlockMv)));
  }
  if (ok) {
    out->d_warps = static_cast<GmeWarpSample*>(
        allocate(warps * sizeof(GmeWarpSample)));
  }
  if (ok) {
    out->d_results = static_cast<GmeEstimate*>(
        allocate(jobs * sizeof(GmeEstimate)));
  }
  if (ok) {
    out->d_control = static_cast<GmeJobControl*>(
        allocate(jobs * sizeof(GmeJobControl)));
  }

  if (!ok) {
    (void)arena.rewind(arena_mark);
    *out = GmeBuffers{};
    set_error(error, "allocate_checked",
              "arena too small: required " + std::to_string(need) + " bytes");
    return GmeStatus::kArenaTooSmall;
  }

  out->max_points = config.max_points;
  out->max_blocks = config.max_blocks;
  out->max_warps = config.max_warps;
  out->max_jobs = config.max_jobs;
  out->persistent_mark = arena_mark;
  return GmeStatus::kOk;
}

void deallocate(Arena& arena, GmeBuffers& buffers) noexcept {
  if (buffers.persistent_mark != kGmeNoArenaMark) {
    (void)arena.rewind(buffers.persistent_mark);
  }
  buffers = GmeBuffers{};
}

std::uint64_t gme_mark(const Arena& arena) noexcept { return arena.mark(); }

bool gme_rewind(Arena& arena, std::uint64_t mark) noexcept {
  return arena.rewind(mark);
}

GmeStatus stage_points(const GmeBuffers& buffers, int job,
                       const GmePoint* host_points, int count,
                       cudaStream_t stream, std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "stage_points", "buffers are not initialized");
    return GmeStatus::kNotInitialized;
  }
  if (!valid_job(buffers, job)) {
    set_error(error, "stage_points", "job index out of range");
    return GmeStatus::kInvalidArgument;
  }
  if (count < 0 || count > buffers.max_points) {
    set_error(error, "stage_points",
              "count must be in [0, max_points=" +
                  std::to_string(buffers.max_points) + "]");
    return GmeStatus::kInvalidArgument;
  }
  if (count == 0) return GmeStatus::kOk;
  if (host_points == nullptr) {
    set_error(error, "stage_points", "null host points");
    return GmeStatus::kInvalidArgument;
  }
  const cudaError_t result = cudaMemcpyAsync(
      buffers.d_points + static_cast<std::size_t>(job) *
                             static_cast<std::size_t>(buffers.max_points),
      host_points, static_cast<std::size_t>(count) * sizeof(GmePoint),
      cudaMemcpyHostToDevice, stream);
  if (result != cudaSuccess) {
    set_error(error, "stage_points", "host-to-device copy failed");
    return GmeStatus::kCudaError;
  }
  return GmeStatus::kOk;
}

GmeStatus stage_blocks(const GmeBuffers& buffers, int job,
                       const GmeBlockMv* host_blocks, int count,
                       cudaStream_t stream, std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "stage_blocks", "buffers are not initialized");
    return GmeStatus::kNotInitialized;
  }
  if (!valid_job(buffers, job)) {
    set_error(error, "stage_blocks", "job index out of range");
    return GmeStatus::kInvalidArgument;
  }
  if (count < 0 || count > buffers.max_blocks) {
    set_error(error, "stage_blocks",
              "count must be in [0, max_blocks=" +
                  std::to_string(buffers.max_blocks) + "]");
    return GmeStatus::kInvalidArgument;
  }
  if (count == 0) return GmeStatus::kOk;
  if (host_blocks == nullptr) {
    set_error(error, "stage_blocks", "null host blocks");
    return GmeStatus::kInvalidArgument;
  }
  const cudaError_t result = cudaMemcpyAsync(
      buffers.d_blocks + static_cast<std::size_t>(job) *
                             static_cast<std::size_t>(buffers.max_blocks),
      host_blocks, static_cast<std::size_t>(count) * sizeof(GmeBlockMv),
      cudaMemcpyHostToDevice, stream);
  if (result != cudaSuccess) {
    set_error(error, "stage_blocks", "host-to-device copy failed");
    return GmeStatus::kCudaError;
  }
  return GmeStatus::kOk;
}

GmeStatus stage_warps(const GmeBuffers& buffers, int job,
                      const GmeWarpSample* host_warps, int count,
                      cudaStream_t stream, std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "stage_warps", "buffers are not initialized");
    return GmeStatus::kNotInitialized;
  }
  if (!valid_job(buffers, job)) {
    set_error(error, "stage_warps", "job index out of range");
    return GmeStatus::kInvalidArgument;
  }
  if (count < 0 || count > buffers.max_warps) {
    set_error(error, "stage_warps",
              "count must be in [0, max_warps=" +
                  std::to_string(buffers.max_warps) + "]");
    return GmeStatus::kInvalidArgument;
  }
  if (count == 0) return GmeStatus::kOk;
  if (host_warps == nullptr) {
    set_error(error, "stage_warps", "null host warps");
    return GmeStatus::kInvalidArgument;
  }
  const cudaError_t result = cudaMemcpyAsync(
      buffers.d_warps + static_cast<std::size_t>(job) *
                            static_cast<std::size_t>(buffers.max_warps),
      host_warps, static_cast<std::size_t>(count) * sizeof(GmeWarpSample),
      cudaMemcpyHostToDevice, stream);
  if (result != cudaSuccess) {
    set_error(error, "stage_warps", "host-to-device copy failed");
    return GmeStatus::kCudaError;
  }
  return GmeStatus::kOk;
}

GmeStatus download_estimate(const GmeBuffers& buffers, int job,
                            GmeEstimate* out, cudaStream_t stream,
                            std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "download_estimate", "buffers are not initialized");
    return GmeStatus::kNotInitialized;
  }
  if (!valid_job(buffers, job)) {
    set_error(error, "download_estimate", "job index out of range");
    return GmeStatus::kInvalidArgument;
  }
  if (out == nullptr) {
    set_error(error, "download_estimate", "null output");
    return GmeStatus::kInvalidArgument;
  }
  const cudaError_t copy = cudaMemcpyAsync(
      out, buffers.d_results + job, sizeof(GmeEstimate),
      cudaMemcpyDeviceToHost, stream);
  if (copy != cudaSuccess) {
    set_error(error, "download_estimate", "device-to-host copy failed");
    return GmeStatus::kCudaError;
  }
  const cudaError_t sync = cudaStreamSynchronize(stream);
  if (sync != cudaSuccess) {
    set_error(error, "download_estimate", "stream synchronisation failed");
    return GmeStatus::kCudaError;
  }
  return GmeStatus::kOk;
}

GmeStatus download_estimates(const GmeBuffers& buffers, GmeEstimate* out,
                             int count, cudaStream_t stream,
                             std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "download_estimates", "buffers are not initialized");
    return GmeStatus::kNotInitialized;
  }
  if (count < 0 || count > buffers.max_jobs) {
    set_error(error, "download_estimates",
              "count must be in [0, max_jobs=" +
                  std::to_string(buffers.max_jobs) + "]");
    return GmeStatus::kInvalidArgument;
  }
  if (count == 0) return GmeStatus::kOk;
  if (out == nullptr) {
    set_error(error, "download_estimates", "null output");
    return GmeStatus::kInvalidArgument;
  }
  const cudaError_t copy = cudaMemcpyAsync(
      out, buffers.d_results,
      static_cast<std::size_t>(count) * sizeof(GmeEstimate),
      cudaMemcpyDeviceToHost, stream);
  if (copy != cudaSuccess) {
    set_error(error, "download_estimates", "device-to-host copy failed");
    return GmeStatus::kCudaError;
  }
  const cudaError_t sync = cudaStreamSynchronize(stream);
  if (sync != cudaSuccess) {
    set_error(error, "download_estimates",
              "stream synchronisation failed");
    return GmeStatus::kCudaError;
  }
  return GmeStatus::kOk;
}

}  // namespace av2enc::cuda::gme
