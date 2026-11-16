// Minimal, deterministic CPU parallel-for helper.
//
// Scope: host-side encoder work with a clear ownership boundary.  It exists
// because macOS clang has no libomp in the default toolchain, so the CPU
// reference path cannot depend on OpenMP; std::thread is always available.
//
// Determinism rules:
//   * The iteration space is partitioned into chunks up front, so the
//     assignment of indices to workers never depends on thread timing.
//   * Every index is visited exactly once, in its own chunk, in increasing
//     order within that chunk.
//   * No reduction happens here.  The caller writes each result into its own
//     slot and merges the slots in index order afterwards, which makes the
//     merged result identical to a serial loop.
//
// `workers <= 1` (the encoder default) runs the serial loop with no thread
// creation at all, so single-threaded behaviour and output are unchanged.
//
// ---------------------------------------------------------------------------
// Persistent pool (phase B)
// ---------------------------------------------------------------------------
//
// `parallel_for` above keeps its signature and its per-call std::thread path
// when no pool is installed.  That path is right for whole-frame batches, but
// a per-block RDO loop calls it thousands of times per frame, where thread
// creation (tens of microseconds each) costs more than the work.
//
// `Pool` is a long-lived fork-join pool with the same contract: `run` blocks
// until every index has been visited exactly once.  Install it once per encode
// at the composition root with `PoolScope`, sized from `--cpu-threads`:
//
//   parallel::Pool pool(parallel::resolve_workers(options.cpu_threads));
//   parallel::PoolScope scope(pool);   // active for the rest of the scope
//
// While a pool is active, `parallel_for` routes through it.  Nested calls (a
// pool worker or a participating caller invoking another parallel loop) still
// run serially, via `in_parallel_region()`, so a parallel region never
// recurses.  Per-block call sites use `parallel_for_pooled()`, which never
// falls back to the per-call thread path: without a pool it stays serial.
//
// Pool determinism: worker wake-up and chunk assignment follow timing, not a
// fixed schedule.  Output is unaffected because every index still runs exactly
// once and the caller writes disjoint per-index slots, merging them in index
// order.  A pool iteration body must not depend on the order of indices.
//
// Idle workers sleep on their own condition variable; there is no spinning.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
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

class Pool;

namespace detail {

// True inside a parallel_for worker, and on the submitting thread while it
// drains chunks of its own job.  Nested parallel_for calls fall back to the
// serial loop so a parallel region never spawns threads recursively.
inline bool& in_parallel_region() {
  static thread_local bool active = false;
  return active;
}

// The pool installed by the innermost PoolScope on this thread, or nullptr.
// Thread-local: an encode that owns a pool never shares it with an unrelated
// thread that happens to call parallel_for at the same time.
inline Pool*& active_pool();

// One unit of work handed to a pool.  Type-erased with a plain function
// pointer so `Pool::run` allocates nothing per call (no std::function, no
// heap block), which matters when the loop is entered once per coded block.
struct Job {
  void* ctx = nullptr;
  void (*visit)(void*, std::size_t) = nullptr;
  std::size_t count = 0;
  std::size_t chunk = 1;
  std::atomic<std::size_t> next{0};
};

template <class Body>
void visit_thunk(void* ctx, std::size_t index) {
  (*static_cast<Body*>(ctx))(index);
}

// Drains `job` chunk by chunk.  Called by the submitting thread and by every
// woken worker.  Each index runs exactly once; which thread runs it is not
// fixed (see the determinism note above).
inline void drain(Job& job) {
  for (;;) {
    const std::size_t begin =
        job.next.fetch_add(job.chunk, std::memory_order_relaxed);
    if (begin >= job.count) return;
    const std::size_t end = std::min(job.count, begin + job.chunk);
    for (std::size_t i = begin; i < end; ++i) job.visit(job.ctx, i);
  }
}

}  // namespace detail

// A fork-join pool: `workers` counts the submitting thread, so the pool owns
// `workers - 1` long-lived threads.  `workers <= 1` owns none.
//
// One submitting thread at a time: `run` publishes the job in shared state, so
// two threads must not fork different jobs into the same Pool concurrently.
// The encoder installs one pool on the encode thread (PoolScope), and nested
// calls from workers fall back to serial, so this holds by construction.  A
// body that throws in a worker terminates the process, exactly as it would in
// the per-call std::thread path.
class Pool {
 public:
  explicit Pool(unsigned workers)
      : workers_(workers == 0 ? 1U : workers) {
    const unsigned live = workers_ > 1 ? workers_ - 1 : 0;
    if (live == 0) return;
    wake_ = std::make_unique<std::condition_variable[]>(live);
    want_.assign(live, 0);
    threads_.reserve(live);
    for (unsigned i = 0; i < live; ++i) {
      threads_.emplace_back([this, i] { worker_main(i); });
    }
  }

  ~Pool() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    for (std::size_t i = 0; i < want_.size(); ++i) wake_[i].notify_one();
    for (std::thread& thread : threads_) thread.join();
  }

  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  unsigned workers() const { return workers_; }

  // Runs `body(i)` for every i in [0, count), blocking until all are done.
  template <class Body>
  void run(std::size_t count, Body&& body) {
    run(count, workers_, std::forward<Body>(body));
  }

  // As above, but wakes at most `hint` ways of parallelism (a call site that
  // only has `count` independent items should not wake idle workers).
  template <class Body>
  void run(std::size_t count, unsigned hint, Body&& body) {
    if (count == 0) return;
    const std::size_t wanted = std::min<std::size_t>(
        count, hint == 0 ? workers_ : static_cast<std::size_t>(hint));
    const unsigned participants =
        static_cast<unsigned>(std::min<std::size_t>(workers_, wanted));
    if (participants <= 1 || detail::in_parallel_region()) {
      for (std::size_t i = 0; i < count; ++i) body(i);
      return;
    }
    // Four chunks per participant keeps the tail short without paying an
    // atomic per index.
    const std::size_t chunk =
        std::max<std::size_t>(1, count / (static_cast<std::size_t>(participants) * 4));
    using Fn = std::remove_reference_t<Body>;
    detail::Job job;
    job.ctx = static_cast<void*>(
        const_cast<std::remove_const_t<Fn>*>(&body));
    job.visit = &detail::visit_thunk<Fn>;
    job.count = count;
    job.chunk = chunk;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      job_ = &job;
      pending_ = participants - 1;
      for (unsigned i = 0; i + 1 < participants; ++i) want_[i] = 1;
    }
    for (unsigned i = 0; i + 1 < participants; ++i) wake_[i].notify_one();
    detail::in_parallel_region() = true;
    detail::drain(job);
    detail::in_parallel_region() = false;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_.wait(lock, [this] { return pending_ == 0; });
      job_ = nullptr;
    }
  }

 private:
  void worker_main(unsigned index) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      wake_[index].wait(
          lock, [this, index] { return shutdown_ || want_[index] != 0; });
      if (shutdown_) return;
      want_[index] = 0;
      detail::Job* job = job_;
      lock.unlock();
      detail::in_parallel_region() = true;
      detail::drain(*job);
      detail::in_parallel_region() = false;
      lock.lock();
      if (--pending_ == 0) done_.notify_one();
    }
  }

  unsigned workers_ = 1;
  std::mutex mutex_;
  std::unique_ptr<std::condition_variable[]> wake_;  // one per worker thread
  std::vector<char> want_;  // 1 = this worker is wanted
  std::condition_variable done_;
  detail::Job* job_ = nullptr;
  std::size_t pending_ = 0;  // workers still inside the current job
  bool shutdown_ = false;
  std::vector<std::thread> threads_;
};

namespace detail {

inline Pool*& active_pool() {
  static thread_local Pool* pool = nullptr;
  return pool;
}

}  // namespace detail

// Installs `pool` as this thread's default for parallel_for.  A pool with a
// single worker is not installed, so the `--cpu-threads 1` path keeps the
// historical serial/per-call behaviour exactly.  Nested scopes keep the
// outermost pool.
class PoolScope {
 public:
  explicit PoolScope(Pool& pool) : previous_(detail::active_pool()) {
    if (previous_ == nullptr && pool.workers() > 1) {
      detail::active_pool() = &pool;
    }
  }
  ~PoolScope() { detail::active_pool() = previous_; }
  PoolScope(const PoolScope&) = delete;
  PoolScope& operator=(const PoolScope&) = delete;

 private:
  Pool* previous_ = nullptr;
};

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
  if (Pool* pool = detail::active_pool()) {
    pool->run(count, resolved, std::forward<Body>(body));
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

// Per-block helper: an independent-candidate loop that is only worth
// parallelising above `min_count` items and only when a pool is active.
// Without a pool it runs the serial loop, never the per-call thread path.
template <class Body>
void parallel_for_pooled(std::size_t count, std::size_t min_count,
                         Body&& body) {
  Pool* pool = detail::active_pool();
  if (count == 0) return;
  if (pool == nullptr || count < min_count || detail::in_parallel_region()) {
    for (std::size_t i = 0; i < count; ++i) body(i);
    return;
  }
  pool->run(count, std::forward<Body>(body));
}

// True when this thread may hand a per-block loop to an installed pool.
inline bool pool_active() {
  return detail::active_pool() != nullptr && !detail::in_parallel_region();
}

}  // namespace av2enc::parallel
