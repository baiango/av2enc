// Persistent device arena for the av2enc CUDA runtime.
//
// One cudaMalloc at init(), then bump allocation only. There is no per-object
// cudaMalloc and no per-allocation free. All allocations are at least
// 256-byte aligned; allocation sizes are rounded up to the same alignment so
// every bump offset (and therefore every mark) stays aligned.
//
// Cap policy (8 GB Ampere target):
//   * kHardCapBytes = 7500 MiB is the process-wide ceiling. The sum of all
//     live Arena reservations may never exceed it, and a single request above
//     it is rejected with Status::kCapExceeded.
//   * kHeadroomBytes = 256 MiB covers the primary context, driver allocations
//     and fragmentation on an 8 GB card. The default arena is
//     kHardCapBytes - kHeadroomBytes = 7244 MiB, matching bench/vram_budget.
//     A larger arena may be requested explicitly, up to the hard cap.
//
// Methods are not thread-safe: use the arena from the single launch/control
// thread. Device VRAM statistics remain readable after release() for reporting.
#ifndef AV2ENC_CUDA_ARENA_H_
#define AV2ENC_CUDA_ARENA_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "error.h"

namespace av2enc::cuda {

class Arena {
 public:
  static constexpr std::uint64_t kMiB = 1024ull * 1024ull;
  static constexpr std::uint64_t kHardCapBytes = 7500ull * kMiB;
  static constexpr std::uint64_t kHeadroomBytes = 256ull * kMiB;
  static constexpr std::uint64_t kDefaultCapacityBytes = kHardCapBytes - kHeadroomBytes;
  static constexpr std::size_t kAlignment = 256;

  // Process-wide cap override, for explicit experiments on a larger GPU.
  // set_cap_bytes(0) restores kHardCapBytes; disable_cap() removes the cap.
  // Intended to be set once, before any Arena::init(), from the CLI flag.
  static void set_cap_bytes(std::uint64_t bytes) noexcept;
  static void disable_cap() noexcept;
  static std::uint64_t cap_bytes() noexcept;

  Arena() = default;
  ~Arena();

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  // Reserve one persistent device block. `capacity_bytes` must be in
  // (0, kHardCapBytes]. The default leaves kHeadroomBytes for the context and
  // driver. `alignment` must be a power of two >= kAlignment.
  Status init(std::uint64_t capacity_bytes = kDefaultCapacityBytes,
              std::size_t alignment = kAlignment, std::string* error = nullptr);

  Status init_mb(std::uint32_t capacity_mb, std::size_t alignment = kAlignment,
                 std::string* error = nullptr);

  // cudaFree() the block and release its cap reservation. Device VRAM
  // statistics stay readable. Safe to call twice. Any pointers previously
  // returned by allocate() are invalid after this call.
  void release() noexcept;

  bool initialized() const noexcept { return base_ != nullptr; }

  // Bump-allocate `bytes`, aligned to at least `alignment`. Returns nullptr on
  // any failure; use allocate_checked() to get the reason.
  void* allocate(std::size_t bytes, std::size_t alignment = kAlignment) noexcept;

  // Status-returning form. On success writes the pointer to *out; on failure
  // sets *out to nullptr and leaves the bump pointer unchanged.
  Status allocate_checked(std::size_t bytes, void** out,
                          std::size_t alignment = kAlignment) noexcept;

  template <typename T>
  T* allocate_array(std::size_t count, std::size_t alignment = kAlignment) noexcept {
    if (count == 0 || count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return nullptr;
    }
    const std::size_t effective_alignment =
        (alignof(T) > alignment) ? alignof(T) : alignment;
    return static_cast<T*>(allocate(count * sizeof(T), effective_alignment));
  }

  std::uint64_t reserved_bytes() const noexcept { return reserved_bytes_; }
  std::uint64_t capacity() const noexcept { return capacity_; }
  std::uint64_t usable_capacity() const noexcept { return capacity_ - base_offset_; }
  std::uint64_t used() const noexcept { return used_; }
  std::uint64_t peak_used() const noexcept { return peak_used_; }
  std::uint64_t remaining() const noexcept;
  std::size_t alignment() const noexcept { return alignment_; }

  // Aligned base pointer (nullptr before init / after release).
  void* base() noexcept { return base_; }
  const void* base() const noexcept { return base_; }

  // Bump watermark support for a reusable scratch region: alloc persistent
  // data first, record mark(), rewind(mark) each frame, and allocate scratch.
  // peak_used() is a high-water mark and is not reduced by rewind/reset.
  std::uint64_t mark() const noexcept { return used_; }
  bool rewind(std::uint64_t mark) noexcept;
  void reset() noexcept { used_ = 0; }

  // cudaMemGetInfo() reporting for the current device. init() samples once
  // after the cudaMalloc. Call sample_vram() after large allocations or at
  // frame boundaries; it is not called per allocate().
  Status sample_vram() noexcept;

  std::uint64_t device_total_bytes() const noexcept { return device_total_; }
  std::uint64_t device_free_bytes() const noexcept { return device_free_; }
  std::uint64_t device_used_bytes() const noexcept;
  std::uint64_t device_peak_used_bytes() const noexcept { return device_peak_used_; }
  std::uint64_t device_free_at_init_bytes() const noexcept { return device_free_at_init_; }
  bool vram_sampled() const noexcept { return vram_sampled_; }

  // Sum of capacities reserved by live Arena instances in this process.
  static std::uint64_t process_reserved_bytes() noexcept;

 private:
  void unreserve(std::uint64_t bytes) noexcept;

  void* raw_ = nullptr;
  void* base_ = nullptr;
  std::uint64_t reserved_bytes_ = 0;
  std::uint64_t capacity_ = 0;
  std::uint64_t base_offset_ = 0;
  std::uint64_t used_ = 0;
  std::uint64_t peak_used_ = 0;
  std::size_t alignment_ = kAlignment;

  std::uint64_t device_total_ = 0;
  std::uint64_t device_free_ = 0;
  std::uint64_t device_free_at_init_ = 0;
  std::uint64_t device_peak_used_ = 0;
  int device_index_ = -1;
  bool vram_sampled_ = false;
};

}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_ARENA_H_
