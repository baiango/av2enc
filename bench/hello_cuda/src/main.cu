#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kTimingRuns = 10;

struct Timing {
  double median_ms = 0.0;
  double min_ms = 0.0;
};

double median_of(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  if (n % 2 == 1) {
    return values[n / 2];
  }
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

// Runs `once` (which itself launches `iters` kernels) `runs` times, timing
// each run with CUDA events. The caller is responsible for warmup.
template <typename F>
Timing time_runs(F&& once, int runs = kTimingRuns) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(runs));
  for (int r = 0; r < runs; ++r) {
    CUDA_CHECK(cudaEventRecord(start));
    once();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    samples.push_back(static_cast<double>(elapsed_ms));
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  Timing out;
  out.median_ms = median_of(samples);
  out.min_ms = *std::min_element(samples.begin(), samples.end());
  return out;
}

void fill_random_u8(std::vector<uint8_t>& v, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  for (uint8_t& x : v) {
    x = static_cast<uint8_t>(dist(rng));
  }
}

void fill_random_i32(std::vector<int32_t>& v, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  for (int32_t& x : v) {
    x = dist(rng);
  }
}

// Xorshift fill for the large benchmark buffers (fast, deterministic).
void fill_fast_i32(std::vector<int32_t>& v, uint32_t seed) {
  uint32_t s = seed ? seed : 1u;
  for (int32_t& x : v) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    x = static_cast<int32_t>(s & 0xffu);
  }
}

bool check_vector_add(bool verbose) {
  constexpr int kN = 1 << 22;
  std::vector<float> ha(kN);
  std::vector<float> hb(kN);
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  for (int i = 0; i < kN; ++i) {
    ha[i] = dist(rng);
    hb[i] = dist(rng);
  }

  float* da = nullptr;
  float* db = nullptr;
  float* dout = nullptr;
  CUDA_CHECK(cudaMalloc(&da, kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, kN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), kN * sizeof(float), cudaMemcpyHostToDevice));

  launch_vector_add(da, db, dout, kN, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hout(kN);
  CUDA_CHECK(cudaMemcpy(hout.data(), dout, kN * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int mismatches = 0;
  float max_abs_diff = 0.0f;
  for (int i = 0; i < kN; ++i) {
    const float ref = ha[i] + hb[i];
    const float diff = std::fabs(hout[i] - ref);
    if (hout[i] != ref) {
      ++mismatches;
    }
    max_abs_diff = std::max(max_abs_diff, diff);
  }

  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dout));

  if (verbose) {
    std::printf("[check] %-22s %s  (mismatches=%d/%d, max_abs_diff=%g)\n",
                "vector_add", mismatches == 0 ? "PASS" : "FAIL", mismatches, kN,
                max_abs_diff);
  }
  return mismatches == 0;
}

void cpu_dct_block(const int32_t* in, int32_t* out, int n, const int32_t* basis) {
  for (int u = 0; u < n; ++u) {
    for (int v = 0; v < n; ++v) {
      int32_t sum = 0;
      for (int y = 0; y < n; ++y) {
        const int32_t bv = basis[v * kDctMaxSize + y];
        for (int x = 0; x < n; ++x) {
          sum += in[y * n + x] * basis[u * kDctMaxSize + x] * bv;
        }
      }
      out[u * n + v] = sum >> (2 * kDctQ);
    }
  }
}

bool check_dct(bool verbose) {
  constexpr int kBlocks = 256;
  constexpr int kMaxElems = kBlocks * 32 * 32;

  int32_t* din = nullptr;
  int32_t* dout = nullptr;
  CUDA_CHECK(cudaMalloc(&din, kMaxElems * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&dout, kMaxElems * sizeof(int32_t)));

  bool ok = true;
  for (int s = 0; s < kDctNumSizes; ++s) {
    const int n = kDctSizes[s];
    const int elems = kBlocks * n * n;
    const int32_t* basis = host_dct_basis(s);

    std::vector<int32_t> hin(static_cast<size_t>(elems));
    fill_random_i32(hin, 9000u + static_cast<uint32_t>(s));

    // Structured edge cases in the first blocks.
    std::fill(hin.begin(), hin.begin() + n * n, 0);
    std::fill(hin.begin() + n * n, hin.begin() + 2 * n * n, 128);
    std::fill(hin.begin() + 2 * n * n, hin.begin() + 3 * n * n, 0);
    hin[2 * n * n] = 255;
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        hin[3 * n * n + y * n + x] = ((x + y) & 1) ? 255 : 0;
      }
    }

    CUDA_CHECK(cudaMemcpy(din, hin.data(), static_cast<size_t>(elems) * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    launch_dct(din, dout, kBlocks, n, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> hout(static_cast<size_t>(elems));
    CUDA_CHECK(cudaMemcpy(hout.data(), dout,
                          static_cast<size_t>(elems) * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    int mismatches = 0;
    int64_t max_abs_diff = 0;
    std::vector<int32_t> ref(static_cast<size_t>(n * n));
    for (int b = 0; b < kBlocks; ++b) {
      cpu_dct_block(hin.data() + static_cast<size_t>(b) * n * n, ref.data(), n,
                    basis);
      for (int i = 0; i < n * n; ++i) {
        const int32_t got = hout[static_cast<size_t>(b) * n * n + i];
        if (got != ref[i]) {
          ++mismatches;
          max_abs_diff =
              std::max<int64_t>(max_abs_diff, std::llabs(static_cast<long long>(got) - ref[i]));
        }
      }
    }

    ok = ok && (mismatches == 0);
    if (verbose) {
      std::printf("[check] %-22s %s  (mismatches=%d/%d, max_abs_diff=%lld)\n",
                  ("dct" + std::to_string(n) + "x" + std::to_string(n)).c_str(),
                  mismatches == 0 ? "PASS" : "FAIL", mismatches, elems,
                  static_cast<long long>(max_abs_diff));
    }
  }

  CUDA_CHECK(cudaFree(din));
  CUDA_CHECK(cudaFree(dout));
  return ok;
}

bool check_sad16(bool verbose) {
  constexpr int kBlocks = 1 << 16;
  constexpr int kBytes = kBlocks * 256;
  std::vector<uint8_t> ha(static_cast<size_t>(kBytes));
  std::vector<uint8_t> hb(static_cast<size_t>(kBytes));
  fill_random_u8(ha, 4242);
  fill_random_u8(hb, 9999);

  uint8_t* da = nullptr;
  uint8_t* db = nullptr;
  int32_t* dout = nullptr;
  CUDA_CHECK(cudaMalloc(&da, kBytes));
  CUDA_CHECK(cudaMalloc(&db, kBytes));
  CUDA_CHECK(cudaMalloc(&dout, kBlocks * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), kBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), kBytes, cudaMemcpyHostToDevice));

  launch_sad16(da, db, dout, kBlocks, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int32_t> hout(static_cast<size_t>(kBlocks));
  CUDA_CHECK(cudaMemcpy(hout.data(), dout, kBlocks * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));

  int mismatches = 0;
  int32_t max_abs_diff = 0;
  for (int b = 0; b < kBlocks; ++b) {
    int32_t ref = 0;
    for (int i = 0; i < 256; ++i) {
      ref += std::abs(static_cast<int>(ha[static_cast<size_t>(b) * 256 + i]) -
                      static_cast<int>(hb[static_cast<size_t>(b) * 256 + i]));
    }
    if (hout[static_cast<size_t>(b)] != ref) {
      ++mismatches;
      max_abs_diff = std::max(max_abs_diff, std::abs(hout[static_cast<size_t>(b)] - ref));
    }
  }

  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dout));

  if (verbose) {
    std::printf("[check] %-22s %s  (mismatches=%d/%d, max_abs_diff=%d)\n",
                "sad16x16", mismatches == 0 ? "PASS" : "FAIL", mismatches,
                kBlocks, max_abs_diff);
  }
  return mismatches == 0;
}

bool check_bandwidth(size_t bytes, bool verbose) {
  uint8_t* dsrc = nullptr;
  uint8_t* ddst = nullptr;
  uint8_t* pinned_src = nullptr;
  uint8_t* pinned_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&dsrc, bytes));
  CUDA_CHECK(cudaMalloc(&ddst, bytes));
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinned_src), bytes,
                           cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinned_dst), bytes,
                           cudaHostAllocDefault));

  std::vector<uint8_t> pageable_src(bytes);
  std::vector<uint8_t> pageable_dst(bytes);
  std::mt19937 rng(7);
  for (size_t i = 0; i < bytes; ++i) {
    pinned_src[i] = static_cast<uint8_t>(rng() & 0xffu);
    pageable_src[i] = static_cast<uint8_t>(rng() & 0xffu);
  }

  bool ok = true;

  // Pinned H2D -> D2H round trip.
  std::memset(pinned_dst, 0, bytes);
  CUDA_CHECK(cudaMemcpy(dsrc, pinned_src, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pinned_dst, dsrc, bytes, cudaMemcpyDeviceToHost));
  ok = ok && (std::memcmp(pinned_src, pinned_dst, bytes) == 0);

  // Device-to-device, verified by reading back.
  CUDA_CHECK(cudaMemset(ddst, 0, bytes));
  CUDA_CHECK(cudaMemcpy(ddst, dsrc, bytes, cudaMemcpyDeviceToDevice));
  std::memset(pinned_dst, 0, bytes);
  CUDA_CHECK(cudaMemcpy(pinned_dst, ddst, bytes, cudaMemcpyDeviceToHost));
  ok = ok && (std::memcmp(pinned_src, pinned_dst, bytes) == 0);

  // Pageable H2D, verified by reading back.
  CUDA_CHECK(cudaMemset(ddst, 0, bytes));
  CUDA_CHECK(cudaMemcpy(ddst, pageable_src.data(), bytes, cudaMemcpyHostToDevice));
  std::memset(pageable_dst.data(), 0, bytes);
  CUDA_CHECK(cudaMemcpy(pageable_dst.data(), ddst, bytes, cudaMemcpyDeviceToHost));
  ok = ok && (std::memcmp(pageable_src.data(), pageable_dst.data(), bytes) == 0);

  // SM copy kernel, verified by reading back.
  CUDA_CHECK(cudaMemset(ddst, 0, bytes));
  launch_copy(dsrc, ddst, bytes, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::memset(pinned_dst, 0, bytes);
  CUDA_CHECK(cudaMemcpy(pinned_dst, ddst, bytes, cudaMemcpyDeviceToHost));
  ok = ok && (std::memcmp(pinned_src, pinned_dst, bytes) == 0);

  CUDA_CHECK(cudaFreeHost(pinned_src));
  CUDA_CHECK(cudaFreeHost(pinned_dst));
  CUDA_CHECK(cudaFree(dsrc));
  CUDA_CHECK(cudaFree(ddst));

  if (verbose) {
    std::printf("[check] %-22s %s  (pinned H2D, D2D, pageable H2D, SM copy)\n",
                "memcpy_bw", ok ? "PASS" : "FAIL");
  }
  return ok;
}

Timing bench_memcpy(void* dst, const void* src, size_t bytes,
                    cudaMemcpyKind kind, int iters) {
  auto once = [&]() {
    for (int i = 0; i < iters; ++i) {
      CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind));
    }
  };
  once();  // warmup
  return time_runs(once);
}

void run_vector_add_bench() {
  constexpr int kN = 1 << 22;
  constexpr int kIters = 100;
  std::vector<float> ha(kN);
  std::vector<float> hb(kN);
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  for (int i = 0; i < kN; ++i) {
    ha[i] = dist(rng);
    hb[i] = dist(rng);
  }

  float* da = nullptr;
  float* db = nullptr;
  float* dout = nullptr;
  CUDA_CHECK(cudaMalloc(&da, kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, kN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), kN * sizeof(float), cudaMemcpyHostToDevice));

  auto once = [&]() {
    for (int i = 0; i < kIters; ++i) {
      launch_vector_add(da, db, dout, kN, nullptr);
    }
  };
  once();  // warmup
  const Timing t = time_runs(once);

  const double bytes = 3.0 * static_cast<double>(kN) * sizeof(float) * kIters;
  const double gbps = bytes / (t.median_ms * 1e6);
  std::printf(
      "vector_add          n=%d iters=%d  median_ms=%.3f  min_ms=%.3f  "
      "bandwidth=%.2f GB/s\n",
      kN, kIters, t.median_ms, t.min_ms, gbps);

  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dout));
}

void run_dct_bench() {
  constexpr size_t kElems = 1ull << 24;  // 16.7M int32 = 64 MiB per buffer
  int32_t* din = nullptr;
  int32_t* dout = nullptr;
  CUDA_CHECK(cudaMalloc(&din, kElems * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&dout, kElems * sizeof(int32_t)));

  std::vector<int32_t> hin(static_cast<size_t>(kElems));
  fill_fast_i32(hin, 2024);
  CUDA_CHECK(cudaMemcpy(din, hin.data(), kElems * sizeof(int32_t),
                        cudaMemcpyHostToDevice));

  const int iters[kDctNumSizes] = {20, 10, 5};
  for (int s = 0; s < kDctNumSizes; ++s) {
    const int n = kDctSizes[s];
    const int nblocks = static_cast<int>(kElems / static_cast<size_t>(n * n));
    auto once = [&]() {
      for (int i = 0; i < iters[s]; ++i) {
        launch_dct(din, dout, nblocks, n, nullptr);
      }
    };
    once();  // warmup
    const Timing t = time_runs(once);

    const double terms = static_cast<double>(nblocks) * iters[s] * n * n * n * n;
    const double gterms = terms / (t.median_ms * 1e6);
    const double ns_per_block =
        (t.median_ms * 1e6) / (static_cast<double>(nblocks) * iters[s]);
    std::printf(
        "dct%-2dx%-2d          blocks=%d iters=%d  median_ms=%.3f  "
        "min_ms=%.3f  ns_per_block=%.1f  throughput=%.2f Gterm/s\n",
        n, n, nblocks, iters[s], t.median_ms, t.min_ms, ns_per_block, gterms);
  }

  CUDA_CHECK(cudaFree(din));
  CUDA_CHECK(cudaFree(dout));
}

void run_sad_bench() {
  constexpr int kBlocks = 1 << 18;
  constexpr int kIters = 50;
  constexpr size_t kBytes = static_cast<size_t>(kBlocks) * 256;

  std::vector<uint8_t> ha(kBytes);
  std::vector<uint8_t> hb(kBytes);
  fill_random_u8(ha, 31);
  fill_random_u8(hb, 32);

  uint8_t* da = nullptr;
  uint8_t* db = nullptr;
  int32_t* dout = nullptr;
  CUDA_CHECK(cudaMalloc(&da, kBytes));
  CUDA_CHECK(cudaMalloc(&db, kBytes));
  CUDA_CHECK(cudaMalloc(&dout, kBlocks * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), kBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), kBytes, cudaMemcpyHostToDevice));

  auto once = [&]() {
    for (int i = 0; i < kIters; ++i) {
      launch_sad16(da, db, dout, kBlocks, nullptr);
    }
  };
  once();  // warmup
  const Timing t = time_runs(once);

  const double pixels = static_cast<double>(kBlocks) * kIters * 256.0;
  const double gpix = pixels / (t.median_ms * 1e6);
  const double gblocks = static_cast<double>(kBlocks) * kIters / (t.median_ms * 1e6);
  std::printf(
      "sad16x16            blocks=%d iters=%d  median_ms=%.3f  min_ms=%.3f  "
      "throughput=%.2f Gpix/s (%.2f Gblock/s)\n",
      kBlocks, kIters, t.median_ms, t.min_ms, gpix, gblocks);

  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dout));
}

void run_bandwidth_bench() {
  constexpr size_t kBytes = 64ull << 20;  // 64 MiB
  uint8_t* dsrc = nullptr;
  uint8_t* ddst = nullptr;
  uint8_t* pinned_src = nullptr;
  uint8_t* pinned_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&dsrc, kBytes));
  CUDA_CHECK(cudaMalloc(&ddst, kBytes));
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinned_src), kBytes,
                           cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinned_dst), kBytes,
                           cudaHostAllocDefault));
  std::vector<uint8_t> pageable_src(kBytes);
  std::memset(pinned_src, 0x5a, kBytes);
  std::memset(pageable_src.data(), 0xa5, kBytes);

  struct Case {
    const char* name;
    void* dst;
    const void* src;
    cudaMemcpyKind kind;
    int iters;
  };
  const Case cases[] = {
      {"memcpy_d2d", ddst, dsrc, cudaMemcpyDeviceToDevice, 30},
      {"memcpy_h2d_pinned", dsrc, pinned_src, cudaMemcpyHostToDevice, 10},
      {"memcpy_h2d_pageable", dsrc, pageable_src.data(), cudaMemcpyHostToDevice, 10},
      {"memcpy_d2h_pinned", pinned_dst, dsrc, cudaMemcpyDeviceToHost, 10},
  };

  for (const Case& c : cases) {
    const Timing t = bench_memcpy(c.dst, c.src, kBytes, c.kind, c.iters);
    const double gbps = static_cast<double>(kBytes) * c.iters / (t.median_ms * 1e6);
    std::printf(
        "%-19s bytes=%zu iters=%d  median_ms=%.3f  min_ms=%.3f  "
        "bandwidth=%.2f GB/s\n",
        c.name, kBytes, c.iters, t.median_ms, t.min_ms, gbps);
  }

  // SM copy path, for comparison with the copy-engine cudaMemcpyAsync path.
  constexpr int kCopyIters = 30;
  auto copy_once = [&]() {
    for (int i = 0; i < kCopyIters; ++i) {
      launch_copy(dsrc, ddst, kBytes, nullptr);
    }
  };
  copy_once();  // warmup
  const Timing ct = time_runs(copy_once);
  const double copy_gbps =
      static_cast<double>(kBytes) * kCopyIters / (ct.median_ms * 1e6);
  std::printf(
      "%-19s bytes=%zu iters=%d  median_ms=%.3f  min_ms=%.3f  "
      "bandwidth=%.2f GB/s\n",
      "copy_kernel_d2d", kBytes, kCopyIters, ct.median_ms, ct.min_ms, copy_gbps);

  CUDA_CHECK(cudaFreeHost(pinned_src));
  CUDA_CHECK(cudaFreeHost(pinned_dst));
  CUDA_CHECK(cudaFree(dsrc));
  CUDA_CHECK(cudaFree(ddst));
}

void print_device_info() {
  int runtime_version = 0;
  int driver_version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  std::printf("=== av2enc CUDA baseline ===\n");
  std::printf("device: %s (sm_%d%d), %d MiB, %d SMs, clock %.2f GHz\n", prop.name,
              prop.major, prop.minor,
              static_cast<int>(prop.totalGlobalMem >> 20), prop.multiProcessorCount,
              static_cast<double>(prop.clockRate) / 1e6);
  std::printf("runtime: CUDA %d.%d, driver: CUDA %d.%d, nvcc: %d.%d\n",
              runtime_version / 1000, (runtime_version % 1000) / 10,
              driver_version / 1000, (driver_version % 1000) / 10,
              __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  bool tiny = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--tiny") == 0) {
      tiny = true;
    } else {
      std::fprintf(stderr, "usage: %s [--tiny]\n", argv[0]);
      return EXIT_FAILURE;
    }
  }

  print_device_info();
  init_dct_basis();

  std::printf("--- correctness ---\n");
  int failures = 0;
  failures += check_vector_add(true) ? 0 : 1;
  failures += check_dct(true) ? 0 : 1;
  failures += check_sad16(true) ? 0 : 1;
  failures += check_bandwidth(32ull << 20, true) ? 0 : 1;

  if (tiny) {
    std::printf("\n(tiny mode: benchmark suite skipped)\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  std::printf("\n--- benchmarks (median of %d runs) ---\n", kTimingRuns);
  run_vector_add_bench();
  run_dct_bench();
  run_sad_bench();
  run_bandwidth_bench();

  std::printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
