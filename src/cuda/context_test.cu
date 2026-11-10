// Runtime-layer tests for src/cuda: device query, one stream, arena cap and
// alignment, bump allocation, peak tracking, pinned staging round-trips, and
// cudaMemGetInfo reporting.
//
// Build and run on the 3060 box:
//   ./build_runtime_test.sh
//   ./build-runtime/context_test --full-cap   # allocates the 7244 MiB default
//
// The full-cap run is opt-in so the normal test suite stays small and cannot
// collide with another process on a shared GPU.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "arena.h"
#include "context.h"

namespace {

using av2enc::cuda::Arena;
using av2enc::cuda::Context;
using av2enc::cuda::DeviceInfo;
using av2enc::cuda::StagingBuffer;
using av2enc::cuda::Status;
using av2enc::cuda::is_ok;
using av2enc::cuda::status_name;

constexpr std::uint64_t kMiB = 1024ull * 1024ull;

int g_checks = 0;
int g_failures = 0;
int g_test_checks = 0;
int g_test_failures = 0;

bool check(bool condition, const char* expression, const char* file, int line) {
  ++g_checks;
  ++g_test_checks;
  if (!condition) {
    ++g_failures;
    ++g_test_failures;
    std::printf("  FAIL %s:%d: %s\n", file, line, expression);
  }
  return condition;
}

bool check_status(Status status, const char* expression, const char* file, int line) {
  if (!is_ok(status)) {
    std::printf("  status '%s' from %s\n", status_name(status), expression);
  }
  return check(is_ok(status), expression, file, line);
}

#define CHECK(condition) check((condition), #condition, __FILE__, __LINE__)
#define CHECK_OK(expression) check_status((expression), #expression, __FILE__, __LINE__)

void begin_test(const char* name) {
  g_test_checks = 0;
  g_test_failures = 0;
  std::printf("[ RUN      ] %s\n", name);
}

void end_test(const char* name) {
  if (g_test_failures == 0) {
    std::printf("[       OK ] %s (%d checks)\n", name, g_test_checks);
  } else {
    std::printf("[  FAILED  ] %s (%d/%d checks failed)\n", name, g_test_failures,
                g_test_checks);
  }
}

std::uint64_t to_mib(std::uint64_t bytes) {
  return bytes / kMiB;
}

unsigned long long ull(std::uint64_t value) {
  return static_cast<unsigned long long>(value);
}

bool is_aligned(const void* pointer, std::size_t alignment) {
  return (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void fill_host(std::vector<unsigned char>& data, unsigned int seed) {
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<unsigned char>((i * 37u + seed) & 0xFFu);
  }
}

__global__ void fill_pattern_kernel(unsigned char* data, std::size_t bytes,
                                    unsigned int seed) {
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
           static_cast<std::size_t>(threadIdx.x);
       i < bytes; i += stride) {
    data[i] = static_cast<unsigned char>((i * 131u + seed) & 0xFFu);
  }
}

void test_device_query() {
  begin_test("device_query");

  int count = 0;
  CHECK_OK(av2enc::cuda::device_count(&count));
  CHECK(count >= 1);

  DeviceInfo info{};
  CHECK_OK(av2enc::cuda::query_device(0, &info));
  CHECK(info.index == 0);
  CHECK(info.name[0] != '\0');
  CHECK(info.cc_major >= 1);
  CHECK(info.sm_count > 0);
  CHECK(info.max_threads_per_sm > 0);
  CHECK(info.max_threads_per_block > 0);
  CHECK(info.shared_mem_per_block > 0);
  CHECK(info.shared_mem_per_block_optin >= info.shared_mem_per_block);
  CHECK(info.shared_mem_per_sm >= info.shared_mem_per_block);
  CHECK(info.total_vram_bytes > 0);
  CHECK(info.free_vram_bytes > 0);
  CHECK(info.free_vram_bytes <= info.total_vram_bytes);

  std::printf("  device %d: %s\n", info.index, info.name);
  std::printf("  compute capability %d.%d, %d SMs, %d threads/SM, %d threads/block\n",
              info.cc_major, info.cc_minor, info.sm_count, info.max_threads_per_sm,
              info.max_threads_per_block);
  std::printf("  shared memory: %zu KiB/block, %zu KiB/block opt-in, %zu KiB/SM\n",
              info.shared_mem_per_block / 1024, info.shared_mem_per_block_optin / 1024,
              info.shared_mem_per_sm / 1024);
  std::printf("  VRAM: total %llu MiB, free %llu MiB\n", ull(to_mib(info.total_vram_bytes)),
              ull(to_mib(info.free_vram_bytes)));

  const double bandwidth = info.memory_bandwidth_gbps();
  CHECK(bandwidth > 0.0);
  std::printf("  memory clock %d kHz, bus %d bits, theoretical %.1f GB/s\n",
              info.memory_clock_khz, info.memory_bus_width_bits, bandwidth);

  CHECK(av2enc::cuda::query_device(-1, &info) == Status::kInvalidArgument);
  CHECK(av2enc::cuda::query_device(0, nullptr) == Status::kInvalidArgument);
  CHECK(av2enc::cuda::query_device(count, &info) == Status::kDeviceUnavailable);

  end_test("device_query");
}

void test_context() {
  begin_test("context_stream");

  Context invalid;
  CHECK(invalid.init(-1) == Status::kInvalidArgument);
  CHECK(invalid.init(999) == Status::kDeviceUnavailable);
  CHECK(!invalid.initialized());
  CHECK(invalid.stream() == nullptr);

  Context context;
  std::string error;
  CHECK_OK(context.init(0, &error));
  CHECK(context.initialized());
  CHECK(context.device_index() == 0);
  CHECK(context.stream() != nullptr);
  CHECK(context.device().sm_count > 0);
  CHECK(context.device().name[0] != '\0');
  CHECK(context.total_vram_bytes() > 0);
  CHECK(context.free_vram_bytes() > 0);
  CHECK(context.used_vram_bytes() ==
        context.total_vram_bytes() - context.free_vram_bytes());

  CHECK(context.init(0, &error) == Status::kAlreadyInitialized);

  const std::uint64_t total_before = context.total_vram_bytes();
  CHECK_OK(context.refresh_vram());
  CHECK(context.total_vram_bytes() == total_before);
  CHECK(context.free_vram_bytes() <= context.total_vram_bytes());

  CHECK_OK(context.query_stream());
  CHECK_OK(context.synchronize());

  context.shutdown();
  CHECK(!context.initialized());
  CHECK(context.stream() == nullptr);
  context.shutdown();  // idempotent

  CHECK_OK(context.init(0, &error));
  CHECK(context.initialized());
  context.shutdown();
  CHECK(!context.initialized());

  end_test("context_stream");
}

void test_arena_cap() {
  begin_test("arena_cap");

  CHECK(Arena::kHardCapBytes == 7500ull * kMiB);
  CHECK(Arena::kHeadroomBytes == 256ull * kMiB);
  CHECK(Arena::kDefaultCapacityBytes == Arena::kHardCapBytes - Arena::kHeadroomBytes);
  CHECK(Arena::kDefaultCapacityBytes == 7244ull * kMiB);
  CHECK(Arena::kAlignment == 256);

  const std::uint64_t baseline = Arena::process_reserved_bytes();
  CHECK(baseline == 0);

  {
    Arena over;
    std::string error;
    CHECK(over.init(Arena::kHardCapBytes + 1, Arena::kAlignment, &error) ==
          Status::kCapExceeded);
    CHECK(!over.initialized());
    CHECK(!error.empty());
    if (!error.empty()) {
      std::printf("  over-cap error: %s\n", error.c_str());
    }
  }
  CHECK(Arena::process_reserved_bytes() == baseline);

  {
    Arena zero;
    std::string error;
    CHECK(zero.init(0, Arena::kAlignment, &error) == Status::kInvalidArgument);
    CHECK(!error.empty());
  }

  {
    Arena arena;
    CHECK_OK(arena.init_mb(64));
    CHECK(arena.initialized());
    CHECK(arena.reserved_bytes() == 64ull * kMiB);
    CHECK(arena.capacity() == 64ull * kMiB);
    CHECK(arena.usable_capacity() <= arena.capacity());
    CHECK(arena.used() == 0);
    CHECK(arena.peak_used() == 0);
    CHECK(arena.remaining() == arena.usable_capacity());
    CHECK(arena.alignment() == Arena::kAlignment);
    CHECK(arena.init_mb(64) == Status::kAlreadyInitialized);
    CHECK(Arena::process_reserved_bytes() == baseline + 64ull * kMiB);

    {
      Arena second;
      CHECK_OK(second.init_mb(32));
      CHECK(second.reserved_bytes() == 32ull * kMiB);
      CHECK(Arena::process_reserved_bytes() == baseline + 96ull * kMiB);
    }
    CHECK(Arena::process_reserved_bytes() == baseline + 64ull * kMiB);

    arena.release();
    CHECK(!arena.initialized());
    CHECK(arena.capacity() == 0);
    CHECK(arena.remaining() == 0);
    CHECK(Arena::process_reserved_bytes() == baseline);

    CHECK_OK(arena.init_mb(8));
    CHECK(arena.reserved_bytes() == 8ull * kMiB);
    CHECK(Arena::process_reserved_bytes() == baseline + 8ull * kMiB);
  }
  CHECK(Arena::process_reserved_bytes() == baseline);

  end_test("arena_cap");
}

void test_arena_alignment() {
  begin_test("arena_alignment");

  Arena arena;
  CHECK_OK(arena.init(16ull * kMiB, 4096));
  CHECK(is_aligned(arena.base(), 4096));
  CHECK(arena.capacity() == 16ull * kMiB);
  CHECK(arena.usable_capacity() >= 16ull * kMiB - 4096);
  CHECK(arena.remaining() == arena.usable_capacity());

  void* p1 = arena.allocate(1);
  CHECK(p1 != nullptr);
  CHECK(is_aligned(p1, 256));
  CHECK(arena.used() == 256);

  void* p255 = arena.allocate(255);
  CHECK(p255 != nullptr);
  CHECK(is_aligned(p255, 256));
  CHECK(arena.used() == 512);
  CHECK(static_cast<std::uint8_t*>(p255) == static_cast<std::uint8_t*>(p1) + 256);

  void* p256 = arena.allocate(256);
  CHECK(p256 != nullptr);
  CHECK(is_aligned(p256, 256));
  CHECK(arena.used() == 768);

  void* p257 = arena.allocate(257);
  CHECK(p257 != nullptr);
  CHECK(is_aligned(p257, 256));
  CHECK(arena.used() == 1280);
  CHECK(static_cast<std::uint8_t*>(p257) == static_cast<std::uint8_t*>(p256) + 256);

  void* p4k = arena.allocate(100, 4096);
  CHECK(p4k != nullptr);
  CHECK(is_aligned(p4k, 4096));
  CHECK(arena.used() == 8192);
  CHECK(static_cast<std::uint8_t*>(p4k) == static_cast<std::uint8_t*>(arena.base()) + 4096);

  CHECK(arena.allocate(16, 3) == nullptr);    // not a power of two
  CHECK(arena.allocate(16, 128) == nullptr);  // below the 256-byte contract
  CHECK(arena.allocate(0) == nullptr);

  void* out = reinterpret_cast<void*>(0x1);
  CHECK(arena.allocate_checked(0, &out) == Status::kInvalidArgument);
  CHECK(out == nullptr);
  CHECK(arena.allocate_checked(16, nullptr) == Status::kInvalidArgument);

  const std::uint64_t mark = arena.mark();
  CHECK(mark == 8192);
  void* scratch = arena.allocate(1ull * kMiB);
  CHECK(scratch != nullptr);
  CHECK(arena.used() == mark + 1ull * kMiB);
  CHECK(arena.peak_used() == mark + 1ull * kMiB);
  CHECK(arena.rewind(mark));
  CHECK(arena.used() == mark);
  CHECK(arena.peak_used() == mark + 1ull * kMiB);
  CHECK(!arena.rewind(mark + 1));
  CHECK(arena.rewind(0));
  CHECK(arena.used() == 0);
  CHECK(arena.peak_used() == mark + 1ull * kMiB);
  arena.reset();
  CHECK(arena.used() == 0);

  end_test("arena_alignment");
}

void test_arena_over_cap() {
  begin_test("arena_over_cap");

  Arena arena;
  CHECK_OK(arena.init_mb(8));
  const std::uint64_t usable = arena.usable_capacity();

  void* out = nullptr;
  CHECK_OK(arena.allocate_checked(1ull * kMiB, &out));
  CHECK(out != nullptr);
  CHECK(arena.used() == 1ull * kMiB);
  CHECK(arena.remaining() == usable - 1ull * kMiB);

  CHECK(arena.allocate(8ull * kMiB) == nullptr);
  CHECK(arena.used() == 1ull * kMiB);

  out = reinterpret_cast<void*>(0x1);
  CHECK(arena.allocate_checked(static_cast<std::size_t>(arena.remaining()) + 1, &out) ==
        Status::kArenaExhausted);
  CHECK(out == nullptr);
  CHECK(arena.used() == 1ull * kMiB);

  CHECK(arena.allocate_checked(static_cast<std::size_t>(SIZE_MAX), &out) ==
        Status::kArenaExhausted);
  CHECK(out == nullptr);
  CHECK(arena.used() == 1ull * kMiB);

  CHECK_OK(arena.allocate_checked(static_cast<std::size_t>(arena.remaining()), &out));
  CHECK(out != nullptr);
  CHECK(arena.used() == usable);
  CHECK(arena.remaining() == 0);
  CHECK(arena.allocate(1) == nullptr);

  CHECK(arena.rewind(0));
  CHECK_OK(arena.allocate_checked(static_cast<std::size_t>(usable), &out));
  CHECK(out != nullptr);
  CHECK(arena.used() == usable);
  CHECK(arena.allocate(1) == nullptr);

  end_test("arena_over_cap");
}

void test_arena_peak() {
  begin_test("arena_peak");

  Arena arena;
  CHECK_OK(arena.init_mb(16));
  CHECK(arena.peak_used() == 0);

  CHECK(arena.allocate(1ull * kMiB) != nullptr);
  CHECK(arena.used() == 1ull * kMiB);
  CHECK(arena.peak_used() == 1ull * kMiB);

  CHECK(arena.allocate(2ull * kMiB) != nullptr);
  CHECK(arena.used() == 3ull * kMiB);
  CHECK(arena.peak_used() == 3ull * kMiB);

  CHECK(arena.allocate(512ull * 1024ull) != nullptr);
  CHECK(arena.used() == 3ull * kMiB + 512ull * 1024ull);
  CHECK(arena.peak_used() == 3ull * kMiB + 512ull * 1024ull);

  arena.reset();
  CHECK(arena.used() == 0);
  CHECK(arena.peak_used() == 3ull * kMiB + 512ull * 1024ull);

  CHECK(arena.allocate(4ull * kMiB) != nullptr);
  CHECK(arena.used() == 4ull * kMiB);
  CHECK(arena.peak_used() == 4ull * kMiB);

  CHECK(arena.allocate(13ull * kMiB) == nullptr);
  CHECK(arena.peak_used() == 4ull * kMiB);

  CHECK(arena.rewind(0));
  CHECK(arena.used() == 0);
  CHECK(arena.peak_used() == 4ull * kMiB);

  CHECK(arena.allocate(1ull * kMiB) != nullptr);
  CHECK(arena.used() == 1ull * kMiB);
  CHECK(arena.peak_used() == 4ull * kMiB);
  CHECK(arena.peak_used() >= arena.used());

  end_test("arena_peak");
}

void test_staging_roundtrip(Context& context) {
  begin_test("staging_roundtrip");

  StagingBuffer staging;
  CHECK(!staging.initialized());
  CHECK_OK(staging.init(1ull * kMiB));
  CHECK(staging.initialized());
  CHECK(staging.capacity() == 1ull * kMiB);
  CHECK(staging.data() != nullptr);
  CHECK(staging.downloaded_bytes() == 0);

  cudaPointerAttributes attributes{};
  const cudaError_t attribute_error =
      cudaPointerGetAttributes(&attributes, staging.data());
  CHECK(attribute_error == cudaSuccess);
  if (attribute_error == cudaSuccess) {
    CHECK(attributes.type == cudaMemoryTypeHost);
  }
  (void)cudaGetLastError();

  CHECK(staging.init(1ull * kMiB) == Status::kAlreadyInitialized);

  Arena arena;
  CHECK_OK(arena.init_mb(8));
  const std::size_t bytes = 1ull * kMiB;
  void* device = arena.allocate(bytes);
  CHECK(device != nullptr);

  std::vector<unsigned char> host(bytes);
  fill_host(host, 1);

  CHECK_OK(staging.upload(context.stream(), device, host.data(), host.size()));
  CHECK_OK(staging.download(context.stream(), device, host.size()));
  CHECK_OK(context.synchronize());
  CHECK(staging.downloaded_bytes() == bytes);
  CHECK(std::memcmp(staging.data(), host.data(), bytes) == 0);

  fill_host(host, 2);
  CHECK_OK(staging.upload(context.stream(), device, host.data(), host.size()));
  CHECK_OK(staging.download(context.stream(), device, host.size()));
  CHECK_OK(context.synchronize());
  CHECK(std::memcmp(staging.data(), host.data(), bytes) == 0);

  CHECK(staging.upload(context.stream(), device, host.data(), 0) ==
        Status::kInvalidArgument);
  CHECK(staging.upload(context.stream(), device, host.data(), bytes + 1) ==
        Status::kInvalidArgument);
  CHECK(staging.upload(context.stream(), nullptr, host.data(), bytes) ==
        Status::kInvalidArgument);
  CHECK(staging.upload(context.stream(), device, nullptr, bytes) ==
        Status::kInvalidArgument);
  CHECK(staging.download(context.stream(), nullptr, bytes) == Status::kInvalidArgument);
  CHECK(staging.download(context.stream(), device, 0) == Status::kInvalidArgument);
  CHECK(staging.download(context.stream(), device, bytes + 1) ==
        Status::kInvalidArgument);

  staging.release();
  CHECK(!staging.initialized());
  CHECK(staging.capacity() == 0);
  CHECK(staging.upload(context.stream(), device, host.data(), bytes) ==
        Status::kNotInitialized);

  CHECK_OK(staging.init(4096));
  CHECK(staging.initialized());
  CHECK(staging.capacity() == 4096);

  end_test("staging_roundtrip");
}

void test_kernel_roundtrip(Context& context) {
  begin_test("kernel_roundtrip");

  Arena arena;
  CHECK_OK(arena.init_mb(4));
  const std::size_t bytes = 1ull * kMiB;
  void* device = arena.allocate(bytes);
  CHECK(device != nullptr);

  const unsigned int threads = 256;
  const std::size_t block_count = (bytes + threads - 1) / threads;
  const unsigned int blocks =
      static_cast<unsigned int>(std::min<std::size_t>(block_count, 4096));
  fill_pattern_kernel<<<blocks, threads, 0, context.stream()>>>(
      static_cast<unsigned char*>(device), bytes, 7u);
  CHECK_OK(av2enc::cuda::last_error());

  StagingBuffer staging;
  CHECK_OK(staging.init(bytes));
  CHECK_OK(staging.download(context.stream(), device, bytes));
  CHECK_OK(context.synchronize());
  CHECK_OK(context.query_stream());

  const unsigned char* result = static_cast<const unsigned char*>(staging.data());
  bool match = true;
  for (std::size_t i = 0; i < bytes; ++i) {
    const unsigned char expected = static_cast<unsigned char>((i * 131u + 7u) & 0xFFu);
    if (result[i] != expected) {
      match = false;
      break;
    }
  }
  CHECK(match);

  end_test("kernel_roundtrip");
}

void test_no_device_alloc() {
  begin_test("no_device_alloc");

  Arena arena;
  CHECK_OK(arena.init_mb(32));
  CHECK_OK(arena.sample_vram());
  const std::uint64_t free_before = arena.device_free_bytes();
  CHECK(free_before > 0);

  constexpr std::size_t kBlockBytes = 4096;
  constexpr std::size_t kCount = 4096;  // 16 MiB
  const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(arena.base());
  const std::uintptr_t end = base + static_cast<std::uintptr_t>(arena.capacity());
  bool blocks_ok = true;
  std::size_t allocated = 0;
  for (std::size_t i = 0; i < kCount; ++i) {
    void* block = arena.allocate(kBlockBytes);
    if (block == nullptr) {
      break;
    }
    ++allocated;
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(block);
    if (address < base || address >= end || !is_aligned(block, 256)) {
      blocks_ok = false;
      break;
    }
  }

  CHECK(allocated == kCount);
  CHECK(blocks_ok);
  CHECK(arena.used() == kCount * kBlockBytes);
  CHECK(arena.peak_used() == kCount * kBlockBytes);

  CHECK_OK(arena.sample_vram());
  const std::uint64_t free_after = arena.device_free_bytes();
  const std::uint64_t delta =
      (free_before > free_after) ? free_before - free_after : free_after - free_before;
  CHECK(delta <= kMiB);  // no cudaMalloc; tolerate another process's noise
  CHECK(arena.device_peak_used_bytes() >= arena.device_used_bytes());

  std::printf("  device free before=%llu MiB after=%llu MiB (delta %llu MiB)\n",
              ull(to_mib(free_before)), ull(to_mib(free_after)), ull(to_mib(delta)));
  std::printf("  arena: %llu MiB reserved, %llu MiB used, peak %llu MiB\n",
              ull(to_mib(arena.reserved_bytes())), ull(to_mib(arena.used())),
              ull(to_mib(arena.peak_used())));

  end_test("no_device_alloc");
}

void test_vram_reporting(Context& context) {
  begin_test("vram_reporting");

  Arena arena;
  CHECK_OK(arena.init_mb(32));
  CHECK(arena.vram_sampled());
  CHECK(arena.device_total_bytes() > 0);
  CHECK(arena.device_free_bytes() > 0);
  CHECK(arena.device_free_bytes() <= arena.device_total_bytes());
  CHECK(arena.device_free_at_init_bytes() > 0);
  CHECK(arena.device_used_bytes() ==
        arena.device_total_bytes() - arena.device_free_bytes());
  CHECK(arena.device_peak_used_bytes() >= arena.device_used_bytes());

  CHECK_OK(context.refresh_vram());
  CHECK(context.total_vram_bytes() == arena.device_total_bytes());
  const std::uint64_t free_delta =
      (context.free_vram_bytes() > arena.device_free_bytes())
          ? context.free_vram_bytes() - arena.device_free_bytes()
          : arena.device_free_bytes() - context.free_vram_bytes();
  CHECK(free_delta <= kMiB);

  CHECK_OK(arena.sample_vram());
  CHECK(arena.device_peak_used_bytes() >= arena.device_used_bytes());

  std::printf("  cudaMemGetInfo: total=%llu MiB free=%llu MiB used=%llu MiB peak=%llu MiB\n",
              ull(to_mib(arena.device_total_bytes())), ull(to_mib(arena.device_free_bytes())),
              ull(to_mib(arena.device_used_bytes())),
              ull(to_mib(arena.device_peak_used_bytes())));
  std::printf("  arena: reserved=%llu MiB used=%llu MiB peak=%llu MiB remaining=%llu MiB\n",
              ull(to_mib(arena.reserved_bytes())), ull(to_mib(arena.used())),
              ull(to_mib(arena.peak_used())), ull(to_mib(arena.remaining())));

  end_test("vram_reporting");
}

void test_full_cap(Context& context) {
  begin_test("full_cap_vram");

  CHECK(Arena::process_reserved_bytes() == 0);
  CHECK_OK(context.refresh_vram());
  const std::uint64_t total = context.total_vram_bytes();
  const std::uint64_t free_before = context.free_vram_bytes();

  Arena arena;
  std::string error;
  CHECK_OK(arena.init(Arena::kDefaultCapacityBytes, Arena::kAlignment, &error));
  CHECK(arena.reserved_bytes() == Arena::kDefaultCapacityBytes);
  CHECK(arena.capacity() == Arena::kDefaultCapacityBytes);
  CHECK(arena.used() == 0);
  CHECK(arena.remaining() == arena.usable_capacity());
  CHECK(arena.device_total_bytes() == total);

  std::printf("  arena: reserved=%llu MiB (hard cap %llu MiB, headroom %llu MiB)\n",
              ull(to_mib(arena.reserved_bytes())), ull(to_mib(Arena::kHardCapBytes)),
              ull(to_mib(Arena::kHeadroomBytes)));
  std::printf("  VRAM before alloc: total=%llu MiB free=%llu MiB used=%llu MiB\n",
              ull(to_mib(total)), ull(to_mib(free_before)),
              ull(to_mib(total - free_before)));
  std::printf("  VRAM after alloc:  free=%llu MiB used=%llu MiB\n",
              ull(to_mib(arena.device_free_bytes())),
              ull(to_mib(arena.device_used_bytes())));

  const Status memset_status = av2enc::cuda::status_from_cuda(
      cudaMemsetAsync(arena.base(), 0x5a, static_cast<std::size_t>(arena.capacity()),
                      context.stream()));
  CHECK_OK(memset_status);

  StagingBuffer head;
  StagingBuffer tail;
  CHECK_OK(head.init(4096));
  CHECK_OK(tail.init(4096));
  CHECK_OK(head.download(context.stream(), arena.base(), 4096));
  const void* tail_source = static_cast<const std::uint8_t*>(arena.base()) +
                            (arena.capacity() - 4096);
  CHECK_OK(tail.download(context.stream(), tail_source, 4096));
  CHECK_OK(context.synchronize());

  const unsigned char* head_bytes = static_cast<const unsigned char*>(head.data());
  const unsigned char* tail_bytes = static_cast<const unsigned char*>(tail.data());
  bool touched = true;
  for (std::size_t i = 0; i < 4096; ++i) {
    if (head_bytes[i] != 0x5a || tail_bytes[i] != 0x5a) {
      touched = false;
      break;
    }
  }
  CHECK(touched);

  void* out = nullptr;
  CHECK(arena.allocate_checked(static_cast<std::size_t>(arena.remaining()) + 1, &out) ==
        Status::kArenaExhausted);
  CHECK(out == nullptr);

  CHECK_OK(arena.sample_vram());
  const std::uint64_t peak = arena.device_peak_used_bytes();
  const std::uint64_t used_before = total - free_before;
  const std::uint64_t peak_increment =
      (peak > used_before) ? peak - used_before : 0;
  constexpr std::uint64_t kNoiseTolerance = 256ull * kMiB;
  const std::uint64_t device_spare =
      (peak < Arena::kHardCapBytes) ? Arena::kHardCapBytes - peak : 0;
  CHECK(peak_increment >= arena.capacity() - kNoiseTolerance);
  CHECK(peak_increment <= arena.capacity() + kNoiseTolerance);
  CHECK(arena.device_peak_used_bytes() >= arena.device_used_bytes());

  std::printf("  VRAM peak: device used=%llu MiB (baseline %llu MiB, arena delta %llu MiB)\n",
              ull(to_mib(peak)), ull(to_mib(used_before)), ull(to_mib(peak_increment)));
  std::printf("  budget: arena %llu MiB + headroom %llu MiB = hard cap %llu MiB; "
              "device-wide spare=%llu MiB\n",
              ull(to_mib(arena.capacity())), ull(to_mib(Arena::kHeadroomBytes)),
              ull(to_mib(Arena::kHardCapBytes)), ull(to_mib(device_spare)));

  arena.release();
  CHECK(!arena.initialized());
  CHECK(Arena::process_reserved_bytes() == 0);

  CHECK_OK(context.refresh_vram());
  const std::uint64_t free_after = context.free_vram_bytes();
  const std::uint64_t recovery =
      (free_before > free_after) ? free_before - free_after : free_after - free_before;
  CHECK(recovery <= 64ull * kMiB);
  std::printf("  VRAM after release: free=%llu MiB (recovery delta %llu MiB)\n",
              ull(to_mib(free_after)), ull(to_mib(recovery)));

  end_test("full_cap_vram");
}

}  // namespace

int main(int argc, char** argv) {
  bool full_cap = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--full-cap") == 0) {
      full_cap = true;
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::printf("usage: %s [--full-cap]\n", argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }

  std::printf("=== av2enc CUDA runtime layer tests ===\n");
  int runtime_version = 0;
  const cudaError_t version_error = cudaRuntimeGetVersion(&runtime_version);
  if (version_error != cudaSuccess) {
    std::printf("cudaRuntimeGetVersion failed: %s\n", cudaGetErrorString(version_error));
    return 1;
  }
  std::printf("CUDA runtime %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);

  Context context;
  std::string error;
  const Status init_status = context.init(0, &error);
  if (!is_ok(init_status)) {
    std::printf("context init failed: %s (%s)\n", status_name(init_status), error.c_str());
    return 1;
  }
  std::printf("device 0: %s (sm_%d%d), %d SMs\n", context.device().name,
              context.device().cc_major, context.device().cc_minor,
              context.device().sm_count);
  std::printf("VRAM: total=%llu MiB free=%llu MiB used=%llu MiB\n",
              ull(to_mib(context.total_vram_bytes())),
              ull(to_mib(context.free_vram_bytes())),
              ull(to_mib(context.used_vram_bytes())));
  std::printf("\n");

  test_device_query();
  test_context();
  test_arena_cap();
  test_arena_alignment();
  test_arena_over_cap();
  test_arena_peak();
  test_staging_roundtrip(context);
  test_kernel_roundtrip(context);
  test_no_device_alloc();
  test_vram_reporting(context);
  if (full_cap) {
    test_full_cap(context);
  }

  std::printf("\n=== summary ===\n");
  std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
  std::printf("RESULT: %s\n", (g_failures == 0) ? "PASS" : "FAIL");

  context.shutdown();
  return (g_failures == 0) ? 0 : 1;
}
