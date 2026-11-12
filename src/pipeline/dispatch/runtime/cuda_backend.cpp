#include "cuda_backend.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "../../../cuda/arena.h"

namespace av2enc::multigpu::runtime {
namespace {

Status from_cuda(cudaError_t error) {
  if (error == cudaSuccess) {
    return Status::kOk;
  }
  (void)cudaGetLastError();  // clear the sticky per-thread error
  switch (error) {
    case cudaErrorMemoryAllocation:
      return Status::kOutOfMemory;
    case cudaErrorInvalidDevice:
    case cudaErrorNoDevice:
    case cudaErrorInsufficientDriver:
    case cudaErrorInitializationError:
      return Status::kDeviceUnavailable;
    case cudaErrorInvalidValue:
    case cudaErrorInvalidConfiguration:
    case cudaErrorInvalidResourceHandle:
    case cudaErrorLaunchOutOfResources:
      return Status::kInvalidArgument;
    case cudaErrorNotReady:
      return Status::kBusy;
    default:
      return Status::kBackendError;
  }
}

Status from_arena(av2enc::cuda::Status status) {
  switch (status) {
    case av2enc::cuda::Status::kOk:
      return Status::kOk;
    case av2enc::cuda::Status::kInvalidArgument:
      return Status::kInvalidArgument;
    case av2enc::cuda::Status::kNotInitialized:
      return Status::kNotInitialized;
    case av2enc::cuda::Status::kAlreadyInitialized:
      return Status::kAlreadyInitialized;
    case av2enc::cuda::Status::kDeviceUnavailable:
      return Status::kDeviceUnavailable;
    case av2enc::cuda::Status::kOutOfMemory:
      return Status::kOutOfMemory;
    case av2enc::cuda::Status::kCapExceeded:
      return Status::kCapExceeded;
    case av2enc::cuda::Status::kArenaExhausted:
      return Status::kArenaExhausted;
    case av2enc::cuda::Status::kBusy:
      return Status::kBusy;
    case av2enc::cuda::Status::kCudaError:
      return Status::kBackendError;
  }
  return Status::kBackendError;
}

Status select_device(int ordinal) {
  return from_cuda(cudaSetDevice(ordinal));
}

void set_error(std::string* error, const std::string& text) {
  if (error != nullptr) {
    *error = text;
  }
}

struct StreamState {
  int ordinal = -1;
  cudaStream_t stream = nullptr;
};

struct EventState {
  int ordinal = -1;
  cudaEvent_t event = nullptr;
  bool timing = false;
};

struct ArenaState {
  int ordinal = -1;
  std::unique_ptr<av2enc::cuda::Arena> arena;
};

template <typename T>
T* from_handle(std::uintptr_t handle) {
  return reinterpret_cast<T*>(handle);
}

template <typename T>
std::uintptr_t to_handle(T* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer);
}

class CudaBackend final : public DeviceBackend {
 public:
  const char* backend_name() const noexcept override { return "cuda"; }

  ~CudaBackend() override {
    for (auto& entry : events_) {
      if (entry.second->event != nullptr) {
        (void)select_device(entry.second->ordinal);
        (void)cudaEventDestroy(entry.second->event);
      }
    }
    for (auto& entry : streams_) {
      if (entry.second->stream != nullptr) {
        (void)select_device(entry.second->ordinal);
        (void)cudaStreamDestroy(entry.second->stream);
      }
    }
    for (auto& entry : arenas_) {
      if (entry.second->arena != nullptr) {
        (void)select_device(entry.second->ordinal);
        entry.second->arena->release();
      }
    }
  }

  Status device_count(int* out) override {
    if (out == nullptr) {
      return Status::kInvalidArgument;
    }
    int count = 0;
    const Status status = from_cuda(cudaGetDeviceCount(&count));
    if (!is_ok(status)) {
      return status;
    }
    *out = count;
    return Status::kOk;
  }

  Status query_device(int ordinal, DeviceProperties* out) override {
    if (out == nullptr || ordinal < 0) {
      return Status::kInvalidArgument;
    }
    int previous = 0;
    const cudaError_t previous_error = cudaGetDevice(&previous);
    Status status = select_device(ordinal);
    if (!is_ok(status)) {
      return status;
    }
    cudaDeviceProp properties{};
    status = from_cuda(cudaGetDeviceProperties(&properties, ordinal));
    if (is_ok(status)) {
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      const cudaError_t mem_error = cudaMemGetInfo(&free_bytes, &total_bytes);
      if (mem_error != cudaSuccess) {
        status = from_cuda(mem_error);
      } else {
        out->ordinal = ordinal;
        out->name = properties.name;
        out->cc_major = properties.major;
        out->cc_minor = properties.minor;
        out->sm_count = properties.multiProcessorCount;
        out->total_vram_bytes = static_cast<std::uint64_t>(total_bytes);
        out->free_vram_bytes = static_cast<std::uint64_t>(free_bytes);
      }
    }
    if (previous_error == cudaSuccess) {
      (void)select_device(previous);
    }
    return status;
  }

  Status context_init(int ordinal, std::string* error) override {
    const Status status = select_device(ordinal);
    if (!is_ok(status)) {
      set_error(error, "cudaSetDevice failed");
      return status;
    }
    // Force primary-context creation without resetting or synchronizing.
    const Status init_status = from_cuda(cudaFree(nullptr));
    if (!is_ok(init_status)) {
      set_error(error, "CUDA primary context initialization failed");
    }
    return init_status;
  }

  void context_shutdown(int ordinal) override {
    // The primary context is process-wide; do not call cudaDeviceReset().
    (void)select_device(ordinal);
  }

  Status stream_create(int ordinal, StreamHandle* out,
                       std::string* error) override {
    if (out == nullptr) {
      return Status::kInvalidArgument;
    }
    Status status = select_device(ordinal);
    if (!is_ok(status)) {
      set_error(error, "cudaSetDevice failed");
      return status;
    }
    auto state = std::make_unique<StreamState>();
    state->ordinal = ordinal;
    const cudaError_t result = cudaStreamCreateWithFlags(
        &state->stream, cudaStreamNonBlocking);
    status = from_cuda(result);
    if (!is_ok(status)) {
      set_error(error, "cudaStreamCreateWithFlags failed");
      return status;
    }
    const StreamHandle handle = to_handle(state.get());
    streams_[handle] = std::move(state);
    *out = handle;
    return Status::kOk;
  }

  void stream_destroy(StreamHandle stream) override {
    const auto it = streams_.find(stream);
    if (it == streams_.end()) {
      return;
    }
    if (it->second->stream != nullptr) {
      (void)select_device(it->second->ordinal);
      (void)cudaStreamDestroy(it->second->stream);
    }
    streams_.erase(it);
  }

  Status stream_synchronize(StreamHandle stream) override {
    const auto it = streams_.find(stream);
    if (it == streams_.end()) {
      return Status::kInvalidArgument;
    }
    (void)select_device(it->second->ordinal);
    return from_cuda(cudaStreamSynchronize(it->second->stream));
  }

  Status event_create(int ordinal, bool timing, EventHandle* out,
                      std::string* error) override {
    if (out == nullptr) {
      return Status::kInvalidArgument;
    }
    Status status = select_device(ordinal);
    if (!is_ok(status)) {
      set_error(error, "cudaSetDevice failed");
      return status;
    }
    auto state = std::make_unique<EventState>();
    state->ordinal = ordinal;
    state->timing = timing;
    const unsigned int flags =
        timing ? static_cast<unsigned int>(cudaEventDefault)
               : static_cast<unsigned int>(cudaEventDisableTiming);
    status = from_cuda(cudaEventCreateWithFlags(&state->event, flags));
    if (!is_ok(status)) {
      set_error(error, "cudaEventCreateWithFlags failed");
      return status;
    }
    const EventHandle handle = to_handle(state.get());
    events_[handle] = std::move(state);
    *out = handle;
    return Status::kOk;
  }

  void event_destroy(EventHandle event) override {
    const auto it = events_.find(event);
    if (it == events_.end()) {
      return;
    }
    if (it->second->event != nullptr) {
      (void)select_device(it->second->ordinal);
      (void)cudaEventDestroy(it->second->event);
    }
    events_.erase(it);
  }

  Status event_record(EventHandle event, StreamHandle stream) override {
    const auto event_it = events_.find(event);
    const auto stream_it = streams_.find(stream);
    if (event_it == events_.end() || stream_it == streams_.end()) {
      return Status::kInvalidArgument;
    }
    (void)select_device(event_it->second->ordinal);
    return from_cuda(
        cudaEventRecord(event_it->second->event, stream_it->second->stream));
  }

  Status event_wait(EventHandle event, StreamHandle stream) override {
    const auto event_it = events_.find(event);
    const auto stream_it = streams_.find(stream);
    if (event_it == events_.end() || stream_it == streams_.end()) {
      return Status::kInvalidArgument;
    }
    if (event_it->second->ordinal != stream_it->second->ordinal) {
      return Status::kDeviceMismatch;
    }
    (void)select_device(event_it->second->ordinal);
    return from_cuda(cudaStreamWaitEvent(stream_it->second->stream,
                                         event_it->second->event, 0));
  }

  Status event_synchronize(EventHandle event) override {
    const auto it = events_.find(event);
    if (it == events_.end()) {
      return Status::kInvalidArgument;
    }
    (void)select_device(it->second->ordinal);
    return from_cuda(cudaEventSynchronize(it->second->event));
  }

  Status event_elapsed_ms(EventHandle start, EventHandle stop,
                          double* out) override {
    if (out == nullptr) {
      return Status::kInvalidArgument;
    }
    const auto start_it = events_.find(start);
    const auto stop_it = events_.find(stop);
    if (start_it == events_.end() || stop_it == events_.end() ||
        !start_it->second->timing || !stop_it->second->timing) {
      return Status::kInvalidArgument;
    }
    if (start_it->second->ordinal != stop_it->second->ordinal) {
      return Status::kDeviceMismatch;
    }
    (void)select_device(start_it->second->ordinal);
    float milliseconds = 0.0f;
    const Status status = from_cuda(cudaEventElapsedTime(
        &milliseconds, start_it->second->event, stop_it->second->event));
    if (is_ok(status)) {
      *out = static_cast<double>(milliseconds);
    }
    return status;
  }

  Status arena_init(int ordinal, std::uint64_t capacity_bytes,
                    ArenaHandle* out, std::string* error) override {
    if (out == nullptr) {
      return Status::kInvalidArgument;
    }
    Status status = select_device(ordinal);
    if (!is_ok(status)) {
      set_error(error, "cudaSetDevice failed");
      return status;
    }
    auto state = std::make_unique<ArenaState>();
    state->ordinal = ordinal;
    state->arena = std::make_unique<av2enc::cuda::Arena>();
    const av2enc::cuda::Status arena_status = state->arena->init(
        capacity_bytes, av2enc::cuda::Arena::kAlignment, error);
    status = from_arena(arena_status);
    if (!is_ok(status)) {
      return status;
    }
    const ArenaHandle handle = to_handle(state.get());
    arenas_[handle] = std::move(state);
    *out = handle;
    return Status::kOk;
  }

  void arena_release(ArenaHandle arena) override {
    const auto it = arenas_.find(arena);
    if (it == arenas_.end()) {
      return;
    }
    (void)select_device(it->second->ordinal);
    if (it->second->arena != nullptr) {
      it->second->arena->release();
    }
    arenas_.erase(it);
  }

  Status arena_allocate(ArenaHandle arena, std::size_t bytes,
                        std::size_t alignment, void** out,
                        std::string* error) override {
    (void)error;
    if (out == nullptr) {
      return Status::kInvalidArgument;
    }
    *out = nullptr;
    const auto it = arenas_.find(arena);
    if (it == arenas_.end()) {
      return Status::kNotInitialized;
    }
    (void)select_device(it->second->ordinal);
    return from_arena(
        it->second->arena->allocate_checked(bytes, out, alignment));
  }

  std::uint64_t arena_capacity(ArenaHandle arena) const override {
    const auto it = arenas_.find(arena);
    return it == arenas_.end() ? 0 : it->second->arena->capacity();
  }

  std::uint64_t arena_used(ArenaHandle arena) const override {
    const auto it = arenas_.find(arena);
    return it == arenas_.end() ? 0 : it->second->arena->used();
  }

  Status pinned_alloc(std::size_t bytes, void** out,
                      std::string* error) override {
    if (out == nullptr || bytes == 0) {
      return Status::kInvalidArgument;
    }
    *out = nullptr;
    const Status status = from_cuda(cudaMallocHost(out, bytes));
    if (!is_ok(status)) {
      set_error(error, "cudaMallocHost failed");
    }
    return status;
  }

  void pinned_free(void* ptr) override {
    if (ptr != nullptr) {
      (void)cudaFreeHost(ptr);
    }
  }

  Status copy_h2d_async(StreamHandle stream, void* device_dst,
                        const void* host_src, std::size_t bytes) override {
    return copy(stream, device_dst, host_src, bytes, cudaMemcpyHostToDevice);
  }

  Status copy_d2h_async(StreamHandle stream, void* host_dst,
                        const void* device_src, std::size_t bytes) override {
    return copy(stream, host_dst, device_src, bytes, cudaMemcpyDeviceToHost);
  }

  Status copy_d2d_async(StreamHandle stream, void* device_dst,
                        const void* device_src, std::size_t bytes) override {
    return copy(stream, device_dst, device_src, bytes, cudaMemcpyDeviceToDevice);
  }

  bool peer_access(int from_ordinal, int to_ordinal) override {
    int can_access = 0;
    if (!is_ok(select_device(from_ordinal))) {
      return false;
    }
    const cudaError_t error =
        cudaDeviceCanAccessPeer(&can_access, from_ordinal, to_ordinal);
    if (error != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    return can_access != 0;
  }

  Status device_synchronize(int ordinal) override {
    const Status status = select_device(ordinal);
    if (!is_ok(status)) {
      return status;
    }
    return from_cuda(cudaDeviceSynchronize());
  }

 private:
  Status copy(StreamHandle stream, void* dst, const void* src,
              std::size_t bytes, cudaMemcpyKind kind) {
    const auto it = streams_.find(stream);
    if (it == streams_.end() || dst == nullptr || src == nullptr || bytes == 0) {
      return Status::kInvalidArgument;
    }
    (void)select_device(it->second->ordinal);
    return from_cuda(
        cudaMemcpyAsync(dst, src, bytes, kind, it->second->stream));
  }

  std::unordered_map<StreamHandle, std::unique_ptr<StreamState>> streams_;
  std::unordered_map<EventHandle, std::unique_ptr<EventState>> events_;
  std::unordered_map<ArenaHandle, std::unique_ptr<ArenaState>> arenas_;
};

}  // namespace

std::unique_ptr<DeviceBackend> make_cuda_backend(std::string* error) {
  int count = 0;
  const cudaError_t result = cudaGetDeviceCount(&count);
  if (result != cudaSuccess) {
    set_error(error, std::string("cudaGetDeviceCount: ") +
                         cudaGetErrorString(result));
    (void)cudaGetLastError();
    return nullptr;
  }
  if (count < 1) {
    set_error(error, "no CUDA devices visible");
    return nullptr;
  }
  return std::make_unique<CudaBackend>();
}

}  // namespace av2enc::multigpu::runtime
