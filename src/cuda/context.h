// CUDA device selection, one stream, and pinned host staging for av2enc.
//
// The runtime layer owns no per-frame allocations and performs no implicit
// synchronization. The only blocking call is Context::synchronize(), intended
// to be used once per frame at the frame boundary. No function here calls
// cudaDeviceSynchronize().
//
// Typical startup order (the arena and staging buffers must be created after
// Context::init() has selected the device):
//
//   av2enc::cuda::Context ctx;
//   if (!av2enc::cuda::is_ok(ctx.init(0))) { ... }
//
//   av2enc::cuda::Arena arena;          // see arena.h
//   if (!av2enc::cuda::is_ok(arena.init())) { ... }
//
//   av2enc::cuda::StagingBuffer upload_staging;
//   if (!av2enc::cuda::is_ok(upload_staging.init(max_frame_bytes))) { ... }
//
// Per frame:
//
//   upload_staging.upload(ctx.stream(), device_frame, host_frame, frame_bytes);
//   ... enqueue kernels on ctx.stream() ...
//   download_staging.download(ctx.stream(), device_payload, payload_bytes);
//   ctx.synchronize();                  // exactly one sync per frame
//   // payload is now readable from download_staging.data()
//
// The layer assumes one device and one control thread. The stream and all
// allocations are not safe to use concurrently from multiple threads without
// external synchronization.
#ifndef AV2ENC_CUDA_CONTEXT_H_
#define AV2ENC_CUDA_CONTEXT_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "error.h"

namespace av2enc::cuda {

struct DeviceInfo {
  int index = -1;
  char name[256] = {};
  int cc_major = 0;
  int cc_minor = 0;
  int sm_count = 0;
  int max_threads_per_sm = 0;
  int max_threads_per_block = 0;
  std::size_t shared_mem_per_block = 0;
  std::size_t shared_mem_per_block_optin = 0;
  std::size_t shared_mem_per_sm = 0;
  int regs_per_sm = 0;
  std::uint64_t total_vram_bytes = 0;
  std::uint64_t free_vram_bytes = 0;
  int memory_clock_khz = 0;
  int memory_bus_width_bits = 0;

  // Theoretical DRAM bandwidth from the reported memory clock and bus width.
  // Uses the DDR transfer rate (2 transfers per clock). Returns 0 when the
  // driver does not report the clocks.
  double memory_bandwidth_gbps() const noexcept;
};

// Number of CUDA devices visible to this process.
Status device_count(int* count) noexcept;

// Query one device. This may initialize the CUDA runtime on that device (and
// therefore create a primary context); call it before allocating an arena.
// `info` must be non-null; index must be in [0, device_count).
Status query_device(int index, DeviceInfo* info) noexcept;

class Context {
 public:
  Context() = default;
  ~Context();

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  // Select `device_index`, create the primary context, query the device, and
  // create one non-blocking stream. On failure nothing is left initialized.
  Status init(int device_index = 0, std::string* error = nullptr);

  // Destroy the stream. Does not call cudaDeviceReset() and does not free the
  // arena or staging buffers; destroy those first. Safe to call twice.
  void shutdown() noexcept;

  bool initialized() const noexcept { return initialized_; }
  int device_index() const noexcept { return device_index_; }
  const DeviceInfo& device() const noexcept { return device_; }
  cudaStream_t stream() const noexcept { return stream_; }

  // Refresh total/free VRAM from cudaMemGetInfo() on this device. Cheap enough
  // for frame boundaries (~10 us) but not for the block loop.
  Status refresh_vram() noexcept;

  std::uint64_t total_vram_bytes() const noexcept { return total_vram_bytes_; }
  std::uint64_t free_vram_bytes() const noexcept { return free_vram_bytes_; }
  std::uint64_t used_vram_bytes() const noexcept {
    return (total_vram_bytes_ > free_vram_bytes_) ? total_vram_bytes_ - free_vram_bytes_ : 0;
  }

  // The only blocking call in this layer. Waits for the one stream and then
  // picks up any sticky launch error. Use at most once per frame.
  Status synchronize() noexcept;

  // Non-blocking. kOk when the stream is idle, kBusy when work is pending.
  Status query_stream() noexcept;

 private:
  DeviceInfo device_;
  cudaStream_t stream_ = nullptr;
  int device_index_ = -1;
  std::uint64_t total_vram_bytes_ = 0;
  std::uint64_t free_vram_bytes_ = 0;
  bool initialized_ = false;
};

// Pinned host staging buffer. init() uses cudaMallocHost(), which is host-side
// memory: it does not consume device VRAM and may be called after the arena is
// initialized.
//
// upload() copies host bytes into the pinned block, then enqueues an H2D copy.
// download() enqueues a D2H copy into the pinned block. Both are asynchronous
// and never synchronize. Do not read downloaded bytes or reuse the buffer
// until Context::synchronize() (or a stream event wait) has completed.
class StagingBuffer {
 public:
  StagingBuffer() = default;
  ~StagingBuffer();

  StagingBuffer(const StagingBuffer&) = delete;
  StagingBuffer& operator=(const StagingBuffer&) = delete;

  Status init(std::size_t capacity_bytes, std::string* error = nullptr);
  void release() noexcept;

  bool initialized() const noexcept { return data_ != nullptr; }
  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t downloaded_bytes() const noexcept { return downloaded_bytes_; }
  void* data() noexcept { return data_; }
  const void* data() const noexcept { return data_; }

  // Host -> pinned staging -> device. `host_src` may alias data(); it is then
  // used in place. Returns kInvalidArgument for zero bytes, null pointers, or
  // bytes > capacity(). Doesn't wait for completion.
  Status upload(cudaStream_t stream, void* device_dst, const void* host_src,
                std::size_t bytes) noexcept;

  // Device -> pinned staging. The bytes are in data() after the stream
  // completes. Returns kInvalidArgument for zero bytes, null pointers, or
  // bytes > capacity().
  Status download(cudaStream_t stream, const void* device_src, std::size_t bytes) noexcept;

 private:
  void* data_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t downloaded_bytes_ = 0;
};

// Raw asynchronous copy helpers for callers that already own pinned host
// memory. `stream` may be nullptr for the legacy default stream, but production
// code should pass Context::stream(). Neither helper synchronizes.
Status upload_async(cudaStream_t stream, void* device_dst, const void* host_src,
                    std::size_t bytes) noexcept;
Status download_async(cudaStream_t stream, void* host_dst, const void* device_src,
                      std::size_t bytes) noexcept;

}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_CONTEXT_H_
