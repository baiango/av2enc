#include "arena.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>

namespace av2enc::cuda {
namespace {

std::atomic<std::uint64_t> g_reserved_bytes{0};
std::atomic<std::uint64_t> g_cap_bytes{Arena::kHardCapBytes};

bool is_valid_alignment(std::size_t alignment) noexcept {
  return alignment >= Arena::kAlignment && (alignment & (alignment - 1)) == 0;
}

// Returns 0 on overflow. Callers treat that as "does not fit".
std::uint64_t align_up_u64(std::uint64_t value, std::uint64_t alignment) noexcept {
  const std::uint64_t mask = alignment - 1;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    return 0;
  }
  return (value + mask) & ~mask;
}

std::uintptr_t align_up_ptr(std::uintptr_t value, std::size_t alignment) noexcept {
  const std::uintptr_t mask = static_cast<std::uintptr_t>(alignment) - 1;
  return (value + mask) & ~mask;
}

void set_error(std::string* error, const std::string& context, Status status,
               cudaError_t cuda_error = cudaSuccess) {
  if (error == nullptr) {
    return;
  }
  char detail[256];
  format_status(status, cuda_error, detail, sizeof(detail));
  *error = context + ": " + detail;
}

std::string mib_text(std::uint64_t bytes) {
  const std::uint64_t mib = bytes / Arena::kMiB + ((bytes % Arena::kMiB) != 0 ? 1 : 0);
  return std::to_string(mib) + " MiB";
}

}  // namespace

Arena::~Arena() {
  release();
}

std::uint64_t Arena::process_reserved_bytes() noexcept {
  return g_reserved_bytes.load(std::memory_order_relaxed);
}

std::uint64_t Arena::cap_bytes() noexcept {
  return g_cap_bytes.load(std::memory_order_relaxed);
}

void Arena::set_cap_bytes(std::uint64_t bytes) noexcept {
  g_cap_bytes.store(bytes == 0 ? kHardCapBytes : bytes, std::memory_order_relaxed);
}

void Arena::disable_cap() noexcept {
  g_cap_bytes.store(std::numeric_limits<std::uint64_t>::max(),
                    std::memory_order_relaxed);
}

Status Arena::init(std::uint64_t capacity_bytes, std::size_t alignment, std::string* error) {
  if (raw_ != nullptr) {
    set_error(error, "Arena::init", Status::kAlreadyInitialized);
    return Status::kAlreadyInitialized;
  }
  if (capacity_bytes == 0) {
    set_error(error, "Arena::init: zero capacity", Status::kInvalidArgument);
    return Status::kInvalidArgument;
  }
  const std::uint64_t cap = cap_bytes();
  if (capacity_bytes > cap) {
    set_error(error,
              "Arena::init: " + mib_text(capacity_bytes) + " request exceeds the " +
                  mib_text(cap) + " hard cap",
              Status::kCapExceeded);
    return Status::kCapExceeded;
  }
  if (!is_valid_alignment(alignment)) {
    set_error(error, "Arena::init: alignment must be a power of two >= 256 bytes",
              Status::kInvalidArgument);
    return Status::kInvalidArgument;
  }

  // Process-wide cap: the sum of live arena reservations may not exceed the
  // hard cap, even if each individual arena is small.
  std::uint64_t reserved = g_reserved_bytes.load(std::memory_order_relaxed);
  for (;;) {
    if (reserved > cap || capacity_bytes > cap - reserved) {
      set_error(error,
                "Arena::init: " + mib_text(capacity_bytes) + " request would exceed the " +
                    mib_text(cap) + " process cap (" + mib_text(reserved) +
                    " already reserved)",
                Status::kCapExceeded);
      return Status::kCapExceeded;
    }
    if (g_reserved_bytes.compare_exchange_weak(reserved, reserved + capacity_bytes,
                                               std::memory_order_acq_rel,
                                               std::memory_order_relaxed)) {
      break;
    }
  }

  void* raw = nullptr;
  const cudaError_t alloc_error = cudaMalloc(&raw, static_cast<std::size_t>(capacity_bytes));
  if (alloc_error != cudaSuccess) {
    unreserve(capacity_bytes);
    const Status status = status_from_cuda(alloc_error);
    set_error(error, "Arena::init: cudaMalloc " + mib_text(capacity_bytes), status, alloc_error);
    return status;
  }

  const std::uintptr_t raw_address = reinterpret_cast<std::uintptr_t>(raw);
  const std::uintptr_t aligned_address = align_up_ptr(raw_address, alignment);
  const std::uint64_t base_offset =
      static_cast<std::uint64_t>(aligned_address - raw_address);
  if (base_offset >= capacity_bytes) {
    (void)cudaFree(raw);
    unreserve(capacity_bytes);
    set_error(error, "Arena::init: alignment leaves no usable arena space",
              Status::kInvalidArgument);
    return Status::kInvalidArgument;
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t vram_error = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (vram_error != cudaSuccess) {
    (void)cudaFree(raw);
    unreserve(capacity_bytes);
    const Status status = status_from_cuda(vram_error);
    set_error(error, "Arena::init: cudaMemGetInfo", status, vram_error);
    return status;
  }

  int device_index = -1;
  (void)cudaGetDevice(&device_index);

  raw_ = raw;
  base_ = reinterpret_cast<void*>(aligned_address);
  reserved_bytes_ = capacity_bytes;
  capacity_ = capacity_bytes;
  base_offset_ = base_offset;
  used_ = 0;
  peak_used_ = 0;
  alignment_ = alignment;
  device_index_ = device_index;
  device_total_ = static_cast<std::uint64_t>(total_bytes);
  device_free_ = static_cast<std::uint64_t>(free_bytes);
  device_free_at_init_ = device_free_;
  device_peak_used_ = device_used_bytes();
  vram_sampled_ = true;
  return Status::kOk;
}

Status Arena::init_mb(std::uint32_t capacity_mb, std::size_t alignment, std::string* error) {
  if (capacity_mb == 0) {
    set_error(error, "Arena::init_mb: zero capacity", Status::kInvalidArgument);
    return Status::kInvalidArgument;
  }
  return init(static_cast<std::uint64_t>(capacity_mb) * kMiB, alignment, error);
}

void Arena::release() noexcept {
  if (raw_ != nullptr) {
    // cudaFree may wait for outstanding work. Teardown only, never per-frame.
    (void)cudaFree(raw_);
    unreserve(reserved_bytes_);
  }
  raw_ = nullptr;
  base_ = nullptr;
  reserved_bytes_ = 0;
  capacity_ = 0;
  base_offset_ = 0;
  used_ = 0;
  peak_used_ = 0;
  alignment_ = kAlignment;
  device_index_ = -1;
  // device_total_/free_at_init_/peak_used_ are retained for reporting.
}

void Arena::unreserve(std::uint64_t bytes) noexcept {
  if (bytes != 0) {
    (void)g_reserved_bytes.fetch_sub(bytes, std::memory_order_acq_rel);
  }
}

Status Arena::allocate_checked(std::size_t bytes, void** out,
                               std::size_t alignment) noexcept {
  if (out == nullptr) {
    return Status::kInvalidArgument;
  }
  *out = nullptr;
  if (base_ == nullptr) {
    return Status::kNotInitialized;
  }
  if (bytes == 0) {
    return Status::kInvalidArgument;
  }
  if (!is_valid_alignment(alignment)) {
    return Status::kInvalidArgument;
  }

  const std::uint64_t usable = usable_capacity();
  const std::uint64_t request = static_cast<std::uint64_t>(bytes);
  const std::uintptr_t base_address = reinterpret_cast<std::uintptr_t>(base_);
  const std::uintptr_t next_address = base_address + used_;
  const std::uintptr_t aligned_address = align_up_ptr(next_address, alignment);
  const std::uint64_t offset = used_ + static_cast<std::uint64_t>(aligned_address - next_address);
  if (offset < used_ || offset > usable) {
    return Status::kArenaExhausted;
  }
  const std::uint64_t rounded = align_up_u64(request, static_cast<std::uint64_t>(alignment));
  if (rounded < request || rounded > usable - offset) {
    return Status::kArenaExhausted;
  }

  *out = reinterpret_cast<void*>(aligned_address);
  used_ = offset + rounded;
  if (used_ > peak_used_) {
    peak_used_ = used_;
  }
  return Status::kOk;
}

void* Arena::allocate(std::size_t bytes, std::size_t alignment) noexcept {
  void* out = nullptr;
  (void)allocate_checked(bytes, &out, alignment);
  return out;
}

std::uint64_t Arena::remaining() const noexcept {
  if (base_ == nullptr) {
    return 0;
  }
  const std::uint64_t usable = usable_capacity();
  return (used_ < usable) ? usable - used_ : 0;
}

bool Arena::rewind(std::uint64_t mark) noexcept {
  if (base_ == nullptr || mark > used_) {
    return false;
  }
  used_ = mark;
  return true;
}

Status Arena::sample_vram() noexcept {
  if (base_ == nullptr) {
    return Status::kNotInitialized;
  }
  if (device_index_ >= 0) {
    const cudaError_t set_device_error = cudaSetDevice(device_index_);
    if (set_device_error != cudaSuccess) {
      return status_from_cuda(set_device_error);
    }
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  device_total_ = static_cast<std::uint64_t>(total_bytes);
  device_free_ = static_cast<std::uint64_t>(free_bytes);
  const std::uint64_t used = device_used_bytes();
  if (used > device_peak_used_) {
    device_peak_used_ = used;
  }
  vram_sampled_ = true;
  return Status::kOk;
}

std::uint64_t Arena::device_used_bytes() const noexcept {
  return (device_total_ > device_free_) ? device_total_ - device_free_ : 0;
}

}  // namespace av2enc::cuda
