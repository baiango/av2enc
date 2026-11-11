// Device abstraction for the multi-GPU runtime.
//
// The runtime owns scheduling, deterministic merge, and reference exchange. It
// reaches hardware only through this interface, so:
//   * the CPU-only test build links a MockBackend (tests/mock_backend.h),
//   * the CUDA build links CudaBackend (cuda_backend.cpp),
//   * no public header includes cuda_runtime.h.
//
// Handles are opaque integers owned by the backend. Zero is invalid. The
// runtime drives every stream from one coordinator thread; the FrameWorker
// callback enqueues compute/copies on that same thread. Backends do not need to
// be thread-safe. Method calls are synchronous on the host except the *_async
// copies, which only enqueue.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "status.h"

namespace av2enc::multigpu::runtime {

struct DeviceProperties {
  int ordinal = -1;
  std::string name;
  int cc_major = 0;
  int cc_minor = 0;
  int sm_count = 0;
  std::uint64_t total_vram_bytes = 0;
  std::uint64_t free_vram_bytes = 0;
};

using StreamHandle = std::uintptr_t;
using EventHandle = std::uintptr_t;
using ArenaHandle = std::uintptr_t;

inline constexpr StreamHandle kInvalidStream = 0;
inline constexpr EventHandle kInvalidEvent = 0;
inline constexpr ArenaHandle kInvalidArena = 0;

// Transfer direction for region copies.
enum class CopyDirection : std::uint8_t {
  kHostToDevice = 0,
  kDeviceToHost = 1,
  kDeviceToDevice = 2,
};

class DeviceBackend {
 public:
  virtual ~DeviceBackend() = default;

  virtual const char* backend_name() const noexcept = 0;

  // Device enumeration. device_count() must not create a context if the
  // backend can avoid it; query_device() may.
  virtual Status device_count(int* out) = 0;
  virtual Status query_device(int ordinal, DeviceProperties* out) = 0;

  // Initialize the primary context for one device. Idempotent backends should
  // return kOk on a second call. context_shutdown() must not call
  // cudaDeviceReset(). Both are called from the coordinator thread.
  virtual Status context_init(int ordinal, std::string* error) = 0;
  virtual void context_shutdown(int ordinal) = 0;

  virtual Status stream_create(int ordinal, StreamHandle* out,
                               std::string* error) = 0;
  virtual void stream_destroy(StreamHandle stream) = 0;
  virtual Status stream_synchronize(StreamHandle stream) = 0;

  // Timing events are only required for reporting; scheduling events are
  // created with timing=false.
  virtual Status event_create(int ordinal, bool timing, EventHandle* out,
                              std::string* error) = 0;
  virtual void event_destroy(EventHandle event) = 0;
  virtual Status event_record(EventHandle event, StreamHandle stream) = 0;
  // Enqueue a wait on `stream` for a previously recorded event.
  virtual Status event_wait(EventHandle event, StreamHandle stream) = 0;
  virtual Status event_synchronize(EventHandle event) = 0;
  virtual Status event_elapsed_ms(EventHandle start, EventHandle stop,
                                  double* out) = 0;

  // Capacity is a request, not a preset: callers derive it from device VRAM
  // (or an explicit cap) before calling arena_init(). The backend must not
  // silently apply a different default.
  virtual Status arena_init(int ordinal, std::uint64_t capacity_bytes,
                            ArenaHandle* out, std::string* error) = 0;
  virtual void arena_release(ArenaHandle arena) = 0;
  virtual Status arena_allocate(ArenaHandle arena, std::size_t bytes,
                                std::size_t alignment, void** out,
                                std::string* error) = 0;
  virtual std::uint64_t arena_capacity(ArenaHandle arena) const = 0;
  virtual std::uint64_t arena_used(ArenaHandle arena) const = 0;

  // Pinned host memory is host RAM, not device VRAM. It is used for source
  // staging, payload/recon downloads, and the full-frame reference assembly.
  virtual Status pinned_alloc(std::size_t bytes, void** out,
                              std::string* error) = 0;
  virtual void pinned_free(void* ptr) = 0;

  virtual Status copy_h2d_async(StreamHandle stream, void* device_dst,
                                const void* host_src, std::size_t bytes) = 0;
  virtual Status copy_d2h_async(StreamHandle stream, void* host_dst,
                                const void* device_src, std::size_t bytes) = 0;
  virtual Status copy_d2d_async(StreamHandle stream, void* device_dst,
                                const void* device_src, std::size_t bytes) = 0;

  // Probe only; the runtime always uses host staging. A true result is
  // informational and never changes the merge order.
  virtual bool peer_access(int from_ordinal, int to_ordinal) = 0;

  // Teardown/reporting only. Never call in a per-frame hot path.
  virtual Status device_synchronize(int ordinal) = 0;
};

}  // namespace av2enc::multigpu::runtime
