// Minimal, deterministic CPU parallel-for helper.
//
// Scope: host-side encoder work with a clear ownership boundary.  It exists
// because macOS clang has no libomp in the default toolchain, so the CPU
// reference path cannot depend on OpenMP; std::thread is always available.
//
// Determinism rules:
//   * The iteration space is partitioned into contiguous chunks up front, so
//     the assignment of indices to workers never depends on thread timing.
//   * Every index is visited exactly once, in its own chunk, in increasing
//     order within that chunk.
//   * No reduction happens here.  The caller writes each result into its own
//     slot and merges the slots in index order afterwards, which makes the
//     merged result identical to a serial loop.
//
// `workers <= 1` (the encoder default) runs the serial loop with no thread
// creation at all, so single-threaded behaviour and output are unchanged.
#pragma once

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace av2enc::parallel {

// Clamp a requested worker count: 0 means "auto" (hardware concurrency), and
// the result is always >= 1.
inline unsigned resolve_workers(unsigned requested) {
  if (requested == 0) {
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1U : hw;
  }
  return requested;
}

namespace detail {

// True inside a parallel_for worker.  Nested parallel_for calls fall back to
// the serial loop so a parallel region never spawns threads recursively.
inline bool& in_parallel_region() {
  static thread_local bool active = false;
  return active;
}

}  // namespace detail

// Runs `body(i)` for every i in [0, count).  `body` must be safe to call from
// several threads; each index is invoked exactly once.
template <class Body>
void parallel_for(std::size_t count, unsigned workers, Body&& body) {
  if (count == 0) return;
  const unsigned resolved = resolve_workers(workers);
  if (resolved <= 1 || count == 1 || detail::in_parallel_region()) {
    for (std::size_t i = 0; i < count; ++i) body(i);
    return;
  }
  const std::size_t chunks =
      std::min<std::size_t>(static_cast<std::size_t>(resolved), count);
  const std::size_t chunk_size = (count + chunks - 1) / chunks;
  std::vector<std::thread> threads;
  threads.reserve(chunks - 1);
  for (std::size_t c = 1; c < chunks; ++c) {
    threads.emplace_back([&body, count, c, chunk_size]() {
      detail::in_parallel_region() = true;
      const std::size_t begin = c * chunk_size;
      const std::size_t end = std::min(count, begin + chunk_size);
      for (std::size_t i = begin; i < end; ++i) body(i);
      detail::in_parallel_region() = false;
    });
  }
  detail::in_parallel_region() = true;
  const std::size_t first_end = std::min(count, chunk_size);
  for (std::size_t i = 0; i < first_end; ++i) body(i);
  detail::in_parallel_region() = false;
  for (std::thread& thread : threads) thread.join();
}

}  // namespace av2enc::parallel
