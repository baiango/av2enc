// Coder-only microbenchmark for the R19 interval sink vs the R05 exact sink.
// Each batch is one leaf-sized symbol run (a fresh sink, adaptation on), which
// is how the estimator is called.  Not part of the encoder; build with:
//   c++ -std=c++20 -O2 -ffp-contract=off -I src bench_rate_interval.cpp \
//       src/codec/transform/tx_partition/tx_partition.cpp -o /tmp/ribench
#include "pipeline/video/video_rate_interval.h"

#include <chrono>
#include <cstdio>

using namespace av2enc::video::rate_interval;

int main() {
  CdfStore store{};
  load_defaults(store, 106);
  const std::uint16_t* cdf = &store.coeff_base_uv[0][0];
  constexpr int kSymbols = 4;
  constexpr int kBatch = 512;      // one leaf's coefficient symbols
  constexpr int kBatches = 40000;  // ~20M symbols per sink
  static std::uint8_t symbols[kBatch];
  std::uint32_t state = 12345;
  for (int i = 0; i < kBatch; ++i) {
    state = state * 1664525u + 1013904223u;
    symbols[i] = static_cast<std::uint8_t>((state >> 28) == 0 ? 0 : 1);
  }

  double interval_bits = 0.0;
  double exact_prefix = 0.0;
  double interval_ns = 0.0;
  double exact_ns = 0.0;
  {
    const auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < kBatches; ++b) {
      IntervalSink sink(/*adapt=*/true);
      for (int i = 0; i < kBatch; ++i) sink.symbol(cdf, kSymbols, symbols[i]);
      interval_bits += sink.bits();
    }
    const auto t1 = std::chrono::steady_clock::now();
    interval_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() /
                  (static_cast<double>(kBatches) * kBatch);
    std::fprintf(stderr, "interval: %.3f ns/symbol %.1f bits/leaf\n",
                 interval_ns, interval_bits / kBatches);
  }
  {
    const auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < kBatches; ++b) {
      ExactSink sink(/*adapt=*/true);
      for (int i = 0; i < kBatch; ++i) sink.symbol(cdf, kSymbols, symbols[i]);
      exact_prefix += sink.prefix_bits();
    }
    const auto t1 = std::chrono::steady_clock::now();
    exact_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() /
               (static_cast<double>(kBatches) * kBatch);
    std::fprintf(stderr, "exact:    %.3f ns/symbol %.1f bits/leaf\n",
                 exact_ns, exact_prefix / kBatches);
  }
  std::fprintf(stderr, "ratio exact/interval = %.2fx\n",
               interval_ns > 0.0 ? exact_ns / interval_ns : 0.0);
  return 0;
}
