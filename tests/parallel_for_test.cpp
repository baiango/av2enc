// Unit test for the deterministic parallel-for helper.
#include <cstdio>
#include <vector>

#include "core/parallel/parallel_for.h"

namespace {

int failures = 0;

void check(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

}  // namespace

int main() {
  // Every index runs exactly once and results keep index order for every
  // worker count, including the serial fallbacks.
  for (unsigned workers : {1U, 2U, 3U, 4U, 8U, 0U}) {
    constexpr std::size_t kCount = 1013;
    std::vector<int> hits(kCount, 0);
    std::vector<int> value(kCount, -1);
    av2enc::parallel::parallel_for(kCount, workers, [&](std::size_t i) {
      ++hits[i];
      value[i] = static_cast<int>(i * 7);
    });
    bool ok = true;
    for (std::size_t i = 0; i < kCount; ++i) {
      ok = ok && hits[i] == 1 && value[i] == static_cast<int>(i * 7);
    }
    check(ok, "coverage/order is independent of the worker count");
  }

  // Zero work is a no-op.
  bool called = false;
  av2enc::parallel::parallel_for(0, 8, [&](std::size_t) { called = true; });
  check(!called, "empty range does not call the body");

  // A nested parallel_for from a worker must not spawn another generation of
  // threads.  Each outer index owns its own inner segment, so this is also a
  // race check under ThreadSanitizer.
  constexpr std::size_t kOuter = 8;
  constexpr std::size_t kInner = 64;
  std::vector<int> nested(kOuter * kInner, 0);
  av2enc::parallel::parallel_for(kOuter, 4, [&](std::size_t i) {
    av2enc::parallel::parallel_for(
        kInner, 4, [&](std::size_t j) { nested[i * kInner + j] += 1; });
  });
  bool nested_ok = true;
  for (int v : nested) nested_ok = nested_ok && v == 1;
  check(nested_ok, "nested parallel_for runs every index exactly once");

  if (failures == 0) std::printf("parallel_for_test: ok\n");
  return failures == 0 ? 0 : 1;
}
