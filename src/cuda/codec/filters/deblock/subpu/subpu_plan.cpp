// Host-side arena, sizing and download layer for the CUDA §7.13.3.15 sub-PU
// grid.  No kernels and no direct device allocation live here, so the same
// translation unit is compiled by the no-nvcc emulation test.
#include "cuda/codec/filters/deblock/subpu/subpu.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace av2enc::cuda::subpu {
namespace {

std::uint64_t align_up(std::uint64_t bytes) noexcept {
  const std::uint64_t alignment = static_cast<std::uint64_t>(Arena::kAlignment);
  return (bytes + alignment - 1) & ~(alignment - 1);
}

void set_error(std::string* error, const char* context,
               const std::string& detail) {
  if (error != nullptr) {
    *error = std::string(context) + ": " + detail;
  }
}

bool buffers_ready(const SubPuBuffers& buffers) noexcept {
  if (buffers.grid.mi_rows <= 0 || buffers.grid.mi_cols <= 0 ||
      buffers.max_jobs <= 0 || buffers.d_jobs == nullptr ||
      buffers.d_codes == nullptr || buffers.d_tx_table == nullptr) {
    return false;
  }
  for (int plane = 0; plane < kSubPuPlaneCount; ++plane) {
    if (buffers.grid.size[plane] == nullptr ||
        buffers.grid.col_base[plane] == nullptr ||
        buffers.grid.row_base[plane] == nullptr) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::uint64_t required_device_bytes(const SubPuConfig& config) noexcept {
  if (config.max_jobs <= 0 || config.max_jobs > kSubPuMaxJobs ||
      config.mi_rows <= 0 || config.mi_cols <= 0) {
    return 0;
  }
  const std::uint64_t rows = static_cast<std::uint64_t>(config.mi_rows);
  const std::uint64_t cols = static_cast<std::uint64_t>(config.mi_cols);
  if (rows > (~static_cast<std::uint64_t>(0)) / cols) {
    return 0;
  }
  const std::uint64_t cells = rows * cols;
  const std::uint64_t jobs = static_cast<std::uint64_t>(config.max_jobs);
  if (cells > (~static_cast<std::uint64_t>(0)) / sizeof(std::int32_t)) {
    return 0;
  }

  const std::uint64_t grid_per_plane =
      align_up(cells * sizeof(std::uint8_t)) +
      align_up(cells * sizeof(std::int32_t)) +
      align_up(cells * sizeof(std::int32_t));
  const std::uint64_t grid =
      static_cast<std::uint64_t>(kSubPuPlaneCount) * grid_per_plane;
  const std::uint64_t staging = align_up(jobs * sizeof(SubPuJob));
  const std::uint64_t codes = align_up(jobs * sizeof(SubPuResult));
  const std::uint64_t tx_table =
      align_up(static_cast<std::uint64_t>(kTxSizeCount) *
               sizeof(SubPuTxEntry));
  return grid + staging + codes + tx_table;
}

std::uint64_t mark(const Arena& arena) noexcept { return arena.mark(); }

bool rewind(Arena& arena, std::uint64_t mark_value) noexcept {
  return arena.rewind(mark_value);
}

const char* subpu_status_name(SubPuStatus status) noexcept {
  switch (status) {
    case SubPuStatus::kOk:
      return "ok";
    case SubPuStatus::kInvalidArgument:
      return "invalid argument";
    case SubPuStatus::kNotInitialized:
      return "not initialized";
    case SubPuStatus::kArenaTooSmall:
      return "arena too small";
    case SubPuStatus::kCudaError:
      return "CUDA error";
  }
  return "unknown status";
}

SubPuStatus allocate_checked(Arena& arena, const SubPuConfig& config,
                             SubPuBuffers* out,
                             std::uint64_t* required_bytes,
                             std::string* error) {
  if (required_bytes != nullptr) {
    *required_bytes = 0;
  }
  if (out == nullptr) {
    set_error(error, "allocate_checked", "null output");
    return SubPuStatus::kInvalidArgument;
  }
  *out = SubPuBuffers{};

  const std::uint64_t need = required_device_bytes(config);
  if (need == 0) {
    set_error(error, "allocate_checked",
              "max_jobs/mi_rows/mi_cols out of range");
    return SubPuStatus::kInvalidArgument;
  }
  if (required_bytes != nullptr) {
    *required_bytes = need;
  }
  if (!arena.initialized()) {
    set_error(error, "allocate_checked", "arena is not initialized");
    return SubPuStatus::kNotInitialized;
  }
  if (arena.remaining() < need) {
    set_error(error, "allocate_checked",
              "arena too small: required " + std::to_string(need) +
                  " bytes, remaining " + std::to_string(arena.remaining()));
    return SubPuStatus::kArenaTooSmall;
  }

  const std::uint64_t arena_mark = arena.mark();
  const std::uint64_t cells =
      static_cast<std::uint64_t>(config.mi_rows) *
      static_cast<std::uint64_t>(config.mi_cols);
  const std::size_t cell_count = static_cast<std::size_t>(cells);

  bool ok = true;
  for (int plane = 0; plane < kSubPuPlaneCount && ok; ++plane) {
    void* raw = nullptr;
    ok = arena.allocate_checked(cell_count * sizeof(std::uint8_t), &raw) ==
         Status::kOk;
    out->grid.size[plane] = static_cast<std::uint8_t*>(raw);
    if (!ok) break;

    raw = nullptr;
    ok = arena.allocate_checked(cell_count * sizeof(std::int32_t), &raw) ==
         Status::kOk;
    out->grid.col_base[plane] = static_cast<std::int32_t*>(raw);
    if (!ok) break;

    raw = nullptr;
    ok = arena.allocate_checked(cell_count * sizeof(std::int32_t), &raw) ==
         Status::kOk;
    out->grid.row_base[plane] = static_cast<std::int32_t*>(raw);
  }

  void* raw = nullptr;
  if (ok) {
    ok = arena.allocate_checked(
             static_cast<std::size_t>(config.max_jobs) * sizeof(SubPuJob),
             &raw) == Status::kOk;
    out->d_jobs = static_cast<SubPuJob*>(raw);
  }
  if (ok) {
    raw = nullptr;
    ok = arena.allocate_checked(
             static_cast<std::size_t>(config.max_jobs) * sizeof(SubPuResult),
             &raw) == Status::kOk;
    out->d_codes = static_cast<SubPuResult*>(raw);
  }
  if (ok) {
    raw = nullptr;
    ok = arena.allocate_checked(
             static_cast<std::size_t>(kTxSizeCount) * sizeof(SubPuTxEntry),
             &raw) == Status::kOk;
    out->d_tx_table = static_cast<SubPuTxEntry*>(raw);
  }

  if (!ok) {
    (void)arena.rewind(arena_mark);
    *out = SubPuBuffers{};
    set_error(error, "allocate_checked",
              "arena too small: required " + std::to_string(need) + " bytes");
    return SubPuStatus::kArenaTooSmall;
  }

  out->grid.mi_rows = config.mi_rows;
  out->grid.mi_cols = config.mi_cols;
  out->max_jobs = config.max_jobs;
  out->persistent_mark = arena_mark;
  return SubPuStatus::kOk;
}

void deallocate(Arena& arena, SubPuBuffers& buffers) noexcept {
  if (buffers.persistent_mark != kNoArenaMark) {
    (void)arena.rewind(buffers.persistent_mark);
  }
  buffers = SubPuBuffers{};
}

SubPuStatus download_tx_table(const SubPuBuffers& buffers, SubPuTxEntry* out,
                              int capacity, cudaStream_t stream,
                              std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "download_tx_table", "buffers are not initialized");
    return SubPuStatus::kNotInitialized;
  }
  if (out == nullptr || capacity < kTxSizeCount) {
    set_error(error, "download_tx_table",
              "output capacity must be at least 25 entries");
    return SubPuStatus::kInvalidArgument;
  }
  const cudaError_t copy = cudaMemcpyAsync(
      out, buffers.d_tx_table,
      static_cast<std::size_t>(kTxSizeCount) * sizeof(SubPuTxEntry),
      cudaMemcpyDeviceToHost, stream);
  if (copy != cudaSuccess) {
    set_error(error, "download_tx_table", "device-to-host copy failed");
    return SubPuStatus::kCudaError;
  }
  const cudaError_t sync = cudaStreamSynchronize(stream);
  if (sync != cudaSuccess) {
    set_error(error, "download_tx_table", "stream synchronisation failed");
    return SubPuStatus::kCudaError;
  }
  return SubPuStatus::kOk;
}

SubPuStatus download_grid(const SubPuBuffers& buffers, SubPuHostGrid* out,
                          cudaStream_t stream, std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "download_grid", "buffers are not initialized");
    return SubPuStatus::kNotInitialized;
  }
  if (out == nullptr) {
    set_error(error, "download_grid", "null output");
    return SubPuStatus::kInvalidArgument;
  }
  const std::size_t cells =
      static_cast<std::size_t>(buffers.grid.mi_rows) *
      static_cast<std::size_t>(buffers.grid.mi_cols);
  out->reset(buffers.grid.mi_rows, buffers.grid.mi_cols);

  for (int plane = 0; plane < kSubPuPlaneCount; ++plane) {
    cudaError_t copy = cudaMemcpyAsync(
        out->size[plane].data(), buffers.grid.size[plane],
        cells * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream);
    if (copy == cudaSuccess) {
      copy = cudaMemcpyAsync(out->col_base[plane].data(),
                             buffers.grid.col_base[plane],
                             cells * sizeof(std::int32_t),
                             cudaMemcpyDeviceToHost, stream);
    }
    if (copy == cudaSuccess) {
      copy = cudaMemcpyAsync(out->row_base[plane].data(),
                             buffers.grid.row_base[plane],
                             cells * sizeof(std::int32_t),
                             cudaMemcpyDeviceToHost, stream);
    }
    if (copy != cudaSuccess) {
      set_error(error, "download_grid", "device-to-host copy failed");
      return SubPuStatus::kCudaError;
    }
  }
  const cudaError_t sync = cudaStreamSynchronize(stream);
  if (sync != cudaSuccess) {
    set_error(error, "download_grid", "stream synchronisation failed");
    return SubPuStatus::kCudaError;
  }
  return SubPuStatus::kOk;
}

}  // namespace av2enc::cuda::subpu
