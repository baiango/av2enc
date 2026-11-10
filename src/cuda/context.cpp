#include "context.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace av2enc::cuda {
namespace {

void set_error(std::string* error, const char* context, Status status,
               cudaError_t cuda_error = cudaSuccess) {
  if (error == nullptr) {
    return;
  }
  char detail[256];
  format_status(status, cuda_error, detail, sizeof(detail));
  *error = std::string(context) + ": " + detail;
}

}  // namespace

double DeviceInfo::memory_bandwidth_gbps() const noexcept {
  if (memory_clock_khz <= 0 || memory_bus_width_bits <= 0) {
    return 0.0;
  }
  const double hz = static_cast<double>(memory_clock_khz) * 1000.0;
  const double bytes_per_transfer = static_cast<double>(memory_bus_width_bits) / 8.0;
  // DDR: two transfers per memory clock.
  return hz * 2.0 * bytes_per_transfer / 1.0e9;
}

Status device_count(int* count) noexcept {
  if (count == nullptr) {
    return Status::kInvalidArgument;
  }
  *count = 0;
  const cudaError_t error = cudaGetDeviceCount(count);
  return status_from_cuda(error);
}

Status query_device(int index, DeviceInfo* info) noexcept {
  if (info == nullptr || index < 0) {
    return Status::kInvalidArgument;
  }

  int count = 0;
  cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  if (index >= count) {
    return Status::kDeviceUnavailable;
  }

  cudaDeviceProp prop{};
  error = cudaGetDeviceProperties(&prop, index);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }

  DeviceInfo result{};
  result.index = index;
  // prop.name is NUL-terminated; result.name is zero-initialized, so copying
  // sizeof - 1 bytes preserves the terminator.
  std::memcpy(result.name, prop.name, sizeof(result.name) - 1);
  result.cc_major = prop.major;
  result.cc_minor = prop.minor;
  result.sm_count = prop.multiProcessorCount;
  result.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  result.max_threads_per_block = prop.maxThreadsPerBlock;
  result.shared_mem_per_block = prop.sharedMemPerBlock;
  result.shared_mem_per_block_optin = prop.sharedMemPerBlockOptin;
  result.shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
  result.regs_per_sm = prop.regsPerMultiprocessor;
  result.memory_clock_khz = prop.memoryClockRate;
  result.memory_bus_width_bits = prop.memoryBusWidth;

  // cudaMemGetInfo() reports the current device, so switch and restore. This
  // may create the primary context on `index`, which is intentional: callers
  // query before sizing the arena.
  int previous = 0;
  const bool have_previous = (cudaGetDevice(&previous) == cudaSuccess);
  cudaError_t op_error = cudaSetDevice(index);
  if (op_error == cudaSuccess) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    op_error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (op_error == cudaSuccess) {
      result.free_vram_bytes = static_cast<std::uint64_t>(free_bytes);
      result.total_vram_bytes = static_cast<std::uint64_t>(total_bytes);
    }
  }
  if (have_previous) {
    const cudaError_t restore_error = cudaSetDevice(previous);
    if (op_error == cudaSuccess) {
      op_error = restore_error;
    }
  }
  if (op_error != cudaSuccess) {
    return status_from_cuda(op_error);
  }

  *info = result;
  return Status::kOk;
}

Context::~Context() {
  shutdown();
}

Status Context::init(int device_index, std::string* error) {
  if (initialized_) {
    set_error(error, "Context::init", Status::kAlreadyInitialized);
    return Status::kAlreadyInitialized;
  }
  if (device_index < 0) {
    set_error(error, "Context::init", Status::kInvalidArgument);
    return Status::kInvalidArgument;
  }

  int count = 0;
  Status status = device_count(&count);
  if (!is_ok(status)) {
    set_error(error, "Context::init: device_count", status);
    return status;
  }
  if (device_index >= count) {
    set_error(error, "Context::init: device index out of range", Status::kDeviceUnavailable);
    return Status::kDeviceUnavailable;
  }

  cudaError_t cuda_error = cudaSetDevice(device_index);
  if (cuda_error != cudaSuccess) {
    const Status mapped_status = status_from_cuda(cuda_error);
    set_error(error, "Context::init: cudaSetDevice", mapped_status, cuda_error);
    return mapped_status;
  }

  // Force primary-context creation before sampling VRAM, so the context's own
  // memory is counted in the baseline rather than attributed to the arena.
  cuda_error = cudaFree(nullptr);
  if (cuda_error != cudaSuccess) {
    const Status mapped_status = status_from_cuda(cuda_error);
    set_error(error, "Context::init: cudaFree(nullptr)", mapped_status, cuda_error);
    return mapped_status;
  }

  DeviceInfo info{};
  status = query_device(device_index, &info);
  if (!is_ok(status)) {
    set_error(error, "Context::init: query_device", status);
    return status;
  }

  cudaStream_t stream = nullptr;
  cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cuda_error != cudaSuccess) {
    const Status mapped_status = status_from_cuda(cuda_error);
    set_error(error, "Context::init: cudaStreamCreateWithFlags", mapped_status, cuda_error);
    return mapped_status;
  }

  device_ = info;
  device_index_ = device_index;
  stream_ = stream;
  total_vram_bytes_ = info.total_vram_bytes;
  free_vram_bytes_ = info.free_vram_bytes;
  initialized_ = true;
  return Status::kOk;
}

void Context::shutdown() noexcept {
  if (stream_ != nullptr) {
    if (initialized_) {
      (void)cudaSetDevice(device_index_);
    }
    // cudaStreamDestroy may wait for outstanding work. This is teardown only;
    // it is never called from a per-frame path.
    (void)cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  initialized_ = false;
  device_index_ = -1;
  total_vram_bytes_ = 0;
  free_vram_bytes_ = 0;
  device_ = DeviceInfo{};
  (void)cudaGetLastError();
}

Status Context::refresh_vram() noexcept {
  if (!initialized_) {
    return Status::kNotInitialized;
  }
  const cudaError_t set_error_code = cudaSetDevice(device_index_);
  if (set_error_code != cudaSuccess) {
    return status_from_cuda(set_error_code);
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  total_vram_bytes_ = static_cast<std::uint64_t>(total_bytes);
  free_vram_bytes_ = static_cast<std::uint64_t>(free_bytes);
  return Status::kOk;
}

Status Context::synchronize() noexcept {
  if (!initialized_ || stream_ == nullptr) {
    return Status::kNotInitialized;
  }
  const cudaError_t set_error_code = cudaSetDevice(device_index_);
  if (set_error_code != cudaSuccess) {
    return status_from_cuda(set_error_code);
  }
  const cudaError_t error = cudaStreamSynchronize(stream_);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  // Pick up a launch/kernel error that the stream sync did not surface.
  return last_error();
}

Status Context::query_stream() noexcept {
  if (!initialized_ || stream_ == nullptr) {
    return Status::kNotInitialized;
  }
  const cudaError_t set_error_code = cudaSetDevice(device_index_);
  if (set_error_code != cudaSuccess) {
    return status_from_cuda(set_error_code);
  }
  const cudaError_t error = cudaStreamQuery(stream_);
  if (error == cudaErrorNotReady) {
    (void)cudaGetLastError();
    return Status::kBusy;
  }
  return status_from_cuda(error);
}

StagingBuffer::~StagingBuffer() {
  release();
}

Status StagingBuffer::init(std::size_t capacity_bytes, std::string* error) {
  if (data_ != nullptr) {
    set_error(error, "StagingBuffer::init", Status::kAlreadyInitialized);
    return Status::kAlreadyInitialized;
  }
  if (capacity_bytes == 0) {
    set_error(error, "StagingBuffer::init: zero capacity", Status::kInvalidArgument);
    return Status::kInvalidArgument;
  }

  void* data = nullptr;
  const cudaError_t cuda_error = cudaMallocHost(&data, capacity_bytes);
  if (cuda_error != cudaSuccess) {
    const Status status = status_from_cuda(cuda_error);
    set_error(error, "StagingBuffer::init: cudaMallocHost", status, cuda_error);
    return status;
  }

  data_ = data;
  capacity_ = capacity_bytes;
  downloaded_bytes_ = 0;
  return Status::kOk;
}

void StagingBuffer::release() noexcept {
  if (data_ != nullptr) {
    (void)cudaFreeHost(data_);
    data_ = nullptr;
  }
  capacity_ = 0;
  downloaded_bytes_ = 0;
}

Status StagingBuffer::upload(cudaStream_t stream, void* device_dst, const void* host_src,
                             std::size_t bytes) noexcept {
  if (data_ == nullptr) {
    return Status::kNotInitialized;
  }
  if (device_dst == nullptr || host_src == nullptr || bytes == 0 || bytes > capacity_) {
    return Status::kInvalidArgument;
  }
  if (host_src != data_) {
    std::memmove(data_, host_src, bytes);
  }
  return upload_async(stream, device_dst, data_, bytes);
}

Status StagingBuffer::download(cudaStream_t stream, const void* device_src,
                               std::size_t bytes) noexcept {
  if (data_ == nullptr) {
    return Status::kNotInitialized;
  }
  if (device_src == nullptr || bytes == 0 || bytes > capacity_) {
    return Status::kInvalidArgument;
  }
  const Status status = download_async(stream, data_, device_src, bytes);
  if (is_ok(status)) {
    downloaded_bytes_ = bytes;
  }
  return status;
}

Status upload_async(cudaStream_t stream, void* device_dst, const void* host_src,
                    std::size_t bytes) noexcept {
  if (device_dst == nullptr || host_src == nullptr || bytes == 0) {
    return Status::kInvalidArgument;
  }
  const cudaError_t error =
      cudaMemcpyAsync(device_dst, host_src, bytes, cudaMemcpyHostToDevice, stream);
  return status_from_cuda(error);
}

Status download_async(cudaStream_t stream, void* host_dst, const void* device_src,
                      std::size_t bytes) noexcept {
  if (host_dst == nullptr || device_src == nullptr || bytes == 0) {
    return Status::kInvalidArgument;
  }
  const cudaError_t error =
      cudaMemcpyAsync(host_dst, device_src, bytes, cudaMemcpyDeviceToHost, stream);
  return status_from_cuda(error);
}

}  // namespace av2enc::cuda
