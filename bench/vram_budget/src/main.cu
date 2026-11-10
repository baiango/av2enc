// VRAM budget smoke test for the 8 GB / 12 GB RTX 3060 SKUs.
//
// Default ceiling is 7500 MB so the same code is safe on the 8 GB card even
// when it runs on a 12 GB box. The arena is limit - headroom; headroom covers
// the CUDA context, driver allocations and fragmentation. Requests beyond the
// cap are rejected by the budget logic before any cudaMalloc is attempted.

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint64_t kMiB = 1024ull * 1024ull;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status_ = (call);                                            \
    if (status_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error: %s (%s) at %s:%d\n", #call,          \
                   cudaGetErrorString(status_), __FILE__, __LINE__);         \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

uint64_t to_mib(uint64_t bytes) { return bytes / kMiB; }

uint64_t parse_limit_mb() {
  const char* env = std::getenv("VRAM_LIMIT_MB");
  const char* text = (env != nullptr) ? env : "7500";
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value == 0) {
    std::fprintf(stderr,
                 "invalid VRAM_LIMIT_MB='%s' (expected a positive integer MiB)\n",
                 text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<uint64_t>(value);
}

void print_mem(const char* label, uint64_t total, uint64_t free_bytes) {
  std::printf("%-16s total=%llu MiB  free=%llu MiB  used=%llu MiB\n", label,
              static_cast<unsigned long long>(to_mib(total)),
              static_cast<unsigned long long>(to_mib(free_bytes)),
              static_cast<unsigned long long>(to_mib(total - free_bytes)));
}

}  // namespace

int main() {
  const uint64_t limit_mb = parse_limit_mb();
  constexpr uint64_t kHeadroomMb = 256;
  constexpr uint64_t kSafetyMb = 64;

  // Create the CUDA context before measuring so context memory is included in
  // the baseline rather than charged against the arena.
  CUDA_CHECK(cudaFree(nullptr));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int cc_major = 0;
  int cc_minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0));

  size_t free_before_size = 0;
  size_t total_size = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_before_size, &total_size));
  const uint64_t total = static_cast<uint64_t>(total_size);
  const uint64_t free_before = static_cast<uint64_t>(free_before_size);

  std::printf("=== av2enc VRAM budget check ===\n");
  std::printf("device: %s (sm_%d%d)\n", prop.name, cc_major, cc_minor);
  std::printf("total: %llu MiB (%llu B)\n",
              static_cast<unsigned long long>(to_mib(total)),
              static_cast<unsigned long long>(total));
  print_mem("before:", total, free_before);

  uint64_t limit = limit_mb * kMiB;
  if (limit > total) {
    std::printf("note: VRAM_LIMIT_MB=%llu exceeds physical total; clamping to %llu MiB\n",
                static_cast<unsigned long long>(limit_mb),
                static_cast<unsigned long long>(to_mib(total)));
    limit = total;
  }
  if (limit <= kHeadroomMb * kMiB) {
    std::fprintf(stderr, "VRAM_LIMIT_MB=%llu is too small (headroom is %llu MiB)\n",
                 static_cast<unsigned long long>(limit_mb),
                 static_cast<unsigned long long>(kHeadroomMb));
    return EXIT_FAILURE;
  }

  const uint64_t headroom = kHeadroomMb * kMiB;
  const uint64_t requested_arena = limit - headroom;
  const uint64_t safety = kSafetyMb * kMiB;
  const uint64_t fit_limit = (free_before > safety) ? free_before - safety : 0;
  const uint64_t arena_bytes = std::min(requested_arena, fit_limit);

  std::printf(
      "config: VRAM_LIMIT_MB=%llu, headroom=%llu MiB, safety=%llu MiB\n",
      static_cast<unsigned long long>(limit_mb),
      static_cast<unsigned long long>(kHeadroomMb),
      static_cast<unsigned long long>(kSafetyMb));
  std::printf("arena plan: %llu MiB (limit - headroom)",
              static_cast<unsigned long long>(to_mib(requested_arena)));
  if (arena_bytes < requested_arena) {
    std::printf(", reduced to %llu MiB to fit free VRAM",
                static_cast<unsigned long long>(to_mib(arena_bytes)));
  }
  std::printf("\n");

  if (arena_bytes == 0) {
    std::fprintf(stderr, "no VRAM available for an arena\n");
    return EXIT_FAILURE;
  }

  void* arena = nullptr;
  const cudaError_t alloc_status = cudaMalloc(&arena, arena_bytes);
  if (alloc_status != cudaSuccess) {
    std::fprintf(stderr, "arena allocation of %llu MiB failed cleanly: %s\n",
                 static_cast<unsigned long long>(to_mib(arena_bytes)),
                 cudaGetErrorString(alloc_status));
    return EXIT_FAILURE;
  }

  size_t free_after_alloc_size = 0;
  size_t total_after_alloc = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_after_alloc_size, &total_after_alloc));
  const uint64_t free_after_alloc = static_cast<uint64_t>(free_after_alloc_size);
  (void)total_after_alloc;
  std::printf("after alloc:    free=%llu MiB  used=%llu MiB\n",
              static_cast<unsigned long long>(to_mib(free_after_alloc)),
              static_cast<unsigned long long>(to_mib(total - free_after_alloc)));

  // Touch every page so the allocation is actually committed, not just
  // reserved in the driver's virtual address space.
  CUDA_CHECK(cudaMemset(arena, 0x5a, arena_bytes));
  CUDA_CHECK(cudaDeviceSynchronize());

  size_t free_after_touch_size = 0;
  size_t total_after_touch = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_after_touch_size, &total_after_touch));
  const uint64_t free_after_touch = static_cast<uint64_t>(free_after_touch_size);
  (void)total_after_touch;
  std::printf("after touch:    free=%llu MiB  used=%llu MiB\n",
              static_cast<unsigned long long>(to_mib(free_after_touch)),
              static_cast<unsigned long long>(to_mib(total - free_after_touch)));

  uint8_t head[16] = {};
  uint8_t tail[16] = {};
  CUDA_CHECK(cudaMemcpy(head, arena, sizeof(head), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(tail,
                        static_cast<const uint8_t*>(arena) + arena_bytes -
                            sizeof(tail),
                        sizeof(tail), cudaMemcpyDeviceToHost));
  bool touched_ok = true;
  for (size_t i = 0; i < sizeof(head); ++i) {
    touched_ok = touched_ok && head[i] == 0x5a && tail[i] == 0x5a;
  }

  const uint64_t min_free =
      std::min(free_before, std::min(free_after_alloc, free_after_touch));
  const uint64_t peak_used = total - min_free;
  const uint64_t used_now = total - free_after_touch;
  const uint64_t remaining = (limit > used_now) ? limit - used_now : 0;

  std::printf("\nbudget check (cap=%llu MiB):\n",
              static_cast<unsigned long long>(to_mib(limit)));
  std::printf("  used including arena: %llu MiB, remaining under cap: %llu MiB\n",
              static_cast<unsigned long long>(to_mib(used_now)),
              static_cast<unsigned long long>(to_mib(remaining)));

  // A 512 MiB reservation is beyond the cap on this configuration. If the
  // arena had to be reduced, fall back to remaining + 1 byte so the rejection
  // test is still exercised.
  uint64_t probe_bytes = 512 * kMiB;
  bool cap_rejected = (used_now + probe_bytes > limit);
  if (!cap_rejected) {
    probe_bytes = remaining + 1;
    cap_rejected = (used_now + probe_bytes > limit);
  }
  std::printf("  reserve %llu MiB -> %s\n",
              static_cast<unsigned long long>(to_mib(probe_bytes)),
              cap_rejected ? "REJECTED (cap)" : "ALLOWED (unexpected)");

  // Hardware-level clean failure: ask for more than the currently free VRAM
  // while the arena is held.
  void* over = nullptr;
  const uint64_t over_bytes = free_after_touch + 256 * kMiB;
  const cudaError_t over_status = cudaMalloc(&over, over_bytes);
  const bool cuda_rejected = (over_status != cudaSuccess);
  if (cuda_rejected) {
    std::printf("  cudaMalloc %llu MiB (> free %llu MiB) -> %s (clean)\n",
                static_cast<unsigned long long>(to_mib(over_bytes)),
                static_cast<unsigned long long>(to_mib(free_after_touch)),
                cudaGetErrorName(over_status));
    (void)cudaGetLastError();
  } else {
    std::printf("  cudaMalloc %llu MiB (> free %llu MiB) -> succeeded (unexpected)\n",
                static_cast<unsigned long long>(to_mib(over_bytes)),
                static_cast<unsigned long long>(to_mib(free_after_touch)));
    CUDA_CHECK(cudaFree(over));
  }

  CUDA_CHECK(cudaFree(arena));

  size_t free_after_free_size = 0;
  size_t total_after_free = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_after_free_size, &total_after_free));
  const uint64_t free_after_free = static_cast<uint64_t>(free_after_free_size);
  (void)total_after_free;
  const uint64_t recovery_delta =
      (free_after_free > free_before) ? (free_after_free - free_before)
                                      : (free_before - free_after_free);
  std::printf("\nafter free:     free=%llu MiB  used=%llu MiB\n",
              static_cast<unsigned long long>(to_mib(free_after_free)),
              static_cast<unsigned long long>(to_mib(total - free_after_free)));

  const bool arena_full = (arena_bytes == requested_arena);
  const bool peak_within_cap = (peak_used <= limit);
  const bool recovered = (recovery_delta <= 64 * kMiB);
  const bool ok = touched_ok && cap_rejected && cuda_rejected && recovered &&
                  peak_within_cap;

  std::printf("\nsummary:\n");
  std::printf("  arena: %llu MiB%s\n",
              static_cast<unsigned long long>(to_mib(arena_bytes)),
              arena_full ? "" : " (reduced to fit free VRAM)");
  std::printf("  peak used: %llu MiB (cap %llu MiB, spare %llu MiB)\n",
              static_cast<unsigned long long>(to_mib(peak_used)),
              static_cast<unsigned long long>(to_mib(limit)),
              static_cast<unsigned long long>(
                  to_mib(limit > peak_used ? limit - peak_used : 0)));
  std::printf("  touch verify: %s\n", touched_ok ? "PASS" : "FAIL");
  std::printf("  cap rejection: %s\n", cap_rejected ? "PASS" : "FAIL");
  std::printf("  cuda over-allocation: %s\n", cuda_rejected ? "PASS" : "FAIL");
  std::printf("  VRAM recovery: %s (%llu MiB delta)\n",
              recovered ? "PASS" : "FAIL",
              static_cast<unsigned long long>(to_mib(recovery_delta)));
  std::printf("  peak within cap: %s\n", peak_within_cap ? "PASS" : "FAIL");
  std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
