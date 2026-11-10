// CUDA transform-variants test: CPU vs GPU bit-exact comparison plus timing.
//
// The CPU reference is the real component implementation in
// src/encoder/tx_variants/tx_variants.cpp (linked into this binary), not a
// second copy.  Every size (25) x transform type (16) is exercised with random,
// structured and impulse residuals; the fused and multi-parameter paths are
// checked separately.
//
// Build/run on the CUDA box:
//   ./build_tx_variants_test.sh
//   ./tx_variants_cuda_test
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "cuda/tx_variants/tx_variants.h"
#include "encoder/tx_variants/tx_variants.h"

namespace {

using namespace av2enc::txv;

constexpr int kMaxBlocks = 256;
constexpr int kMaxAdj = kMaxAdjTxElems;
constexpr int kMaxCoded = 64 * 64;

int g_failures = 0;
long long g_checks = 0;

struct DeviceBuf {
  std::int16_t* res = nullptr;
  std::int16_t* coded = nullptr;
  std::int16_t* adj_out = nullptr;
  std::int16_t* coded_out = nullptr;
  double* coeff = nullptr;
  std::int32_t* levels = nullptr;
  std::int32_t* dequant = nullptr;
  av2enc::cuda::TxBlockJob* jobs = nullptr;
};

DeviceBuf g_dev;

#define CUDA_TRY(expr, what)                                             \
  do {                                                                   \
    const cudaError_t err_ = (expr);                                     \
    if (err_ != cudaSuccess) {                                           \
      std::printf("CUDA error %s at %s: %s\n", cudaGetErrorString(err_), \
                  what, #expr);                                          \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

template <typename T>
void dev_alloc(T** ptr, std::size_t count, const char* what) {
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(ptr), sizeof(T) * count), what);
}

void allocate() {
  dev_alloc(&g_dev.res, static_cast<std::size_t>(kMaxBlocks) * kMaxAdj, "res");
  dev_alloc(&g_dev.coded, static_cast<std::size_t>(kMaxBlocks) * kMaxCoded,
            "coded");
  dev_alloc(&g_dev.adj_out, static_cast<std::size_t>(kMaxBlocks) * kMaxAdj,
            "adj_out");
  dev_alloc(&g_dev.coded_out, static_cast<std::size_t>(kMaxBlocks) * kMaxCoded,
            "coded_out");
  dev_alloc(&g_dev.coeff, static_cast<std::size_t>(kMaxBlocks) * kMaxAdj,
            "coeff");
  dev_alloc(&g_dev.levels, static_cast<std::size_t>(kMaxBlocks) * kMaxAdj,
            "levels");
  dev_alloc(&g_dev.dequant, static_cast<std::size_t>(kMaxBlocks) * kMaxAdj,
            "dequant");
  dev_alloc(&g_dev.jobs, static_cast<std::size_t>(kMaxBlocks), "jobs");
}

template <typename T>
bool compare_exact(const std::vector<T>& cpu, const T* dev, int n,
                   const char* name) {
  std::vector<T> gpu(static_cast<std::size_t>(n));
  CUDA_TRY(cudaMemcpy(gpu.data(), dev, sizeof(T) * static_cast<std::size_t>(n),
                      cudaMemcpyDeviceToHost),
           "compare memcpy");
  int mismatches = 0;
  int first = -1;
  double max_abs = 0.0;
  for (int i = 0; i < n; ++i) {
    const double a = static_cast<double>(cpu[static_cast<std::size_t>(i)]);
    const double b = static_cast<double>(gpu[static_cast<std::size_t>(i)]);
    if (a != b) {
      ++mismatches;
      if (first < 0) {
        first = i;
      }
      max_abs = std::max(max_abs, std::fabs(a - b));
    }
  }
  ++g_checks;
  if (mismatches != 0) {
    ++g_failures;
    std::printf("  FAIL %-10s n=%d mismatch=%d max_abs=%.17g first=%d\n", name,
                n, mismatches, max_abs, first);
  }
  return mismatches == 0;
}

void fill_random(std::int16_t* data, int n, std::mt19937* rng) {
  std::uniform_int_distribution<int> dist(-255, 255);
  for (int i = 0; i < n; ++i) {
    data[i] = static_cast<std::int16_t>(dist(*rng));
  }
}

void fill_structured(std::int16_t* data, int n, int kind) {
  for (int i = 0; i < n; ++i) {
    switch (kind) {
      case 0:
        data[i] = 0;
        break;
      case 1:
        data[i] = 255;
        break;
      case 2:
        data[i] = -255;
        break;
      case 3:
        data[i] = static_cast<std::int16_t>(i % 32);
        break;
      case 4:
        data[i] = static_cast<std::int16_t>((i / 32) % 32);
        break;
      case 5:
        data[i] = static_cast<std::int16_t>((i % 2) ? 127 : -127);
        break;
      default:
        data[i] = static_cast<std::int16_t>((i * 37) % 511 - 255);
        break;
    }
  }
}

// Runs every transform stage for one (size, type) and compares CPU vs GPU.
void verify_case(const TxParams& p, const std::vector<std::int16_t>& residual,
                 int nblocks) {
  const int adj_n = p.w * p.h;
  const int coded_n = p.coded_w * p.coded_h;
  const int total_adj = adj_n * nblocks;
  const int total_coded = coded_n * nblocks;

  std::vector<double> coeff_cpu(static_cast<std::size_t>(total_adj));
  std::vector<std::int32_t> levels_cpu(static_cast<std::size_t>(total_adj));
  std::vector<std::int32_t> dequant_cpu(static_cast<std::size_t>(total_adj));
  std::vector<std::int16_t> recon_cpu(static_cast<std::size_t>(total_coded));
  std::vector<std::int16_t> recon_fused(static_cast<std::size_t>(total_coded));
  std::vector<std::int16_t> adj_cpu(static_cast<std::size_t>(total_adj));
  std::vector<std::int16_t> coded_cpu(static_cast<std::size_t>(total_coded));

  CUDA_TRY(cudaMemcpy(g_dev.res, residual.data(),
                      sizeof(std::int16_t) * static_cast<std::size_t>(total_adj),
                      cudaMemcpyHostToDevice),
           "upload residual");

  // CPU reference.
  for (int b = 0; b < nblocks; ++b) {
    const std::int16_t* r = residual.data() + static_cast<std::size_t>(b) * adj_n;
    forward_transform(r, p,
                      coeff_cpu.data() + static_cast<std::size_t>(b) * adj_n,
                      levels_cpu.data() + static_cast<std::size_t>(b) * adj_n);
    dequantize(levels_cpu.data() + static_cast<std::size_t>(b) * adj_n, p,
               dequant_cpu.data() + static_cast<std::size_t>(b) * adj_n);
    inverse_block(dequant_cpu.data() + static_cast<std::size_t>(b) * adj_n, p,
                  recon_cpu.data() + static_cast<std::size_t>(b) * coded_n);
    upsample_residual(r, p,
                      coded_cpu.data() + static_cast<std::size_t>(b) * coded_n);
    downsample_residual(
        coded_cpu.data() + static_cast<std::size_t>(b) * coded_n, p,
        adj_cpu.data() + static_cast<std::size_t>(b) * adj_n);
  }

  const char* size_name = tx_size_name(static_cast<TxSize>(p.tx_size));
  const char* type_name = tx_type_name(static_cast<TxType>(p.tx_type));
  char label[64];
  std::snprintf(label, sizeof(label), "%s/%s", size_name, type_name);

  // GPU forward_block.
  CUDA_TRY(av2enc::cuda::txv_forward_block(g_dev.res, g_dev.coeff, p, nblocks,
                                           nullptr),
           "forward_block");
  char what[96];
  std::snprintf(what, sizeof(what), "fwd %s", label);
  compare_exact(coeff_cpu, g_dev.coeff, total_adj, what);

  // GPU quantize.
  CUDA_TRY(av2enc::cuda::txv_quantize(g_dev.coeff, g_dev.levels, p, nblocks,
                                      nullptr),
           "quantize");
  std::snprintf(what, sizeof(what), "quant %s", label);
  compare_exact(levels_cpu, g_dev.levels, total_adj, what);

  // GPU fused forward.
  CUDA_TRY(av2enc::cuda::txv_forward_transform(g_dev.res, g_dev.coeff,
                                               g_dev.levels, p, nblocks,
                                               nullptr),
           "forward_transform");
  std::snprintf(what, sizeof(what), "fused %s", label);
  compare_exact(coeff_cpu, g_dev.coeff, total_adj, what);
  std::snprintf(what, sizeof(what), "fusedlv %s", label);
  compare_exact(levels_cpu, g_dev.levels, total_adj, what);

  // GPU dequantize + inverse_block.
  CUDA_TRY(av2enc::cuda::txv_dequantize(g_dev.levels, g_dev.dequant, p, nblocks,
                                        nullptr),
           "dequantize");
  std::snprintf(what, sizeof(what), "dequant %s", label);
  compare_exact(dequant_cpu, g_dev.dequant, total_adj, what);

  CUDA_TRY(av2enc::cuda::txv_inverse_block(g_dev.dequant, g_dev.coded_out, p,
                                           nblocks, nullptr),
           "inverse_block");
  std::snprintf(what, sizeof(what), "inv %s", label);
  compare_exact(recon_cpu, g_dev.coded_out, total_coded, what);

  // GPU fused inverse.
  CUDA_TRY(av2enc::cuda::txv_inverse_transform(g_dev.levels, g_dev.coded_out, p,
                                               nblocks, nullptr),
           "inverse_transform");
  std::snprintf(what, sizeof(what), "ifused %s", label);
  compare_exact(recon_cpu, g_dev.coded_out, total_coded, what);

  // Upsample / downsample.  `coded_cpu` is the upsampled residual.
  CUDA_TRY(cudaMemcpy(g_dev.coded, coded_cpu.data(),
                      sizeof(std::int16_t) *
                          static_cast<std::size_t>(total_coded),
                      cudaMemcpyHostToDevice),
           "upload coded");
  CUDA_TRY(av2enc::cuda::txv_downsample_residual(g_dev.coded, g_dev.adj_out, p,
                                                 nblocks, nullptr),
           "downsample");
  std::snprintf(what, sizeof(what), "down %s", label);
  compare_exact(adj_cpu, g_dev.adj_out, total_adj, what);

  CUDA_TRY(av2enc::cuda::txv_upsample_residual(g_dev.res, g_dev.coded_out, p,
                                               nblocks, nullptr),
           "upsample");
  std::snprintf(what, sizeof(what), "up %s", label);
  compare_exact(coded_cpu, g_dev.coded_out, total_coded, what);
}

void test_all_sizes_types() {
  std::mt19937 rng(1234);
  for (int s = 0; s < 25; ++s) {
    for (int t = 0; t < 16; ++t) {
      const auto p = make_tx_params(static_cast<TxSize>(s),
                                    static_cast<TxType>(t));
      if (!p.has_value()) {
        continue;
      }
      const int adj_n = p->w * p->h;

      // 8 random + 7 structured blocks.
      const int nblocks = 15;
      std::vector<std::int16_t> residual(
          static_cast<std::size_t>(nblocks) * adj_n);
      for (int b = 0; b < 8; ++b) {
        fill_random(residual.data() + static_cast<std::size_t>(b) * adj_n,
                    adj_n, &rng);
      }
      for (int b = 8; b < nblocks; ++b) {
        fill_structured(residual.data() + static_cast<std::size_t>(b) * adj_n,
                        adj_n, b - 8);
      }
      verify_case(*p, residual, nblocks);
    }
  }
}

void test_impulses() {
  std::mt19937 rng(99);
  (void)rng;
  // Dense impulse coverage for 4x4/8x8; sampled for 16x16/32x32.
  for (int s = 0; s < 25; ++s) {
    const auto p0 = make_tx_params(static_cast<TxSize>(s));
    if (!p0.has_value()) {
      continue;
    }
    const int adj_n = p0->w * p0->h;
    int stride = 1;
    if (adj_n > 256) {
      stride = 7;
    } else if (adj_n > 64) {
      stride = 3;
    }
    std::vector<int> positions;
    for (int i = 0; i < adj_n; i += stride) {
      positions.push_back(i);
      if (static_cast<int>(positions.size()) >= kMaxBlocks) {
        break;
      }
    }
    for (int t = 0; t < 16; ++t) {
      const auto p = make_tx_params(static_cast<TxSize>(s),
                                    static_cast<TxType>(t));
      if (!p.has_value()) {
        continue;
      }
      const int nblocks = static_cast<int>(positions.size());
      std::vector<std::int16_t> residual(
          static_cast<std::size_t>(nblocks) * adj_n, 0);
      for (int b = 0; b < nblocks; ++b) {
        residual[static_cast<std::size_t>(b) * adj_n +
                 static_cast<std::size_t>(positions[b])] =
            static_cast<std::int16_t>(1 + (b % 255));
      }
      verify_case(*p, residual, nblocks);
    }
    (void)rng;
  }
}

void test_multi_params() {
  std::mt19937 rng(7);
  const int nblocks = 64;
  std::vector<av2enc::cuda::TxBlockJob> jobs(
      static_cast<std::size_t>(nblocks));
  std::vector<std::int16_t> residual;
  std::vector<double> coeff_cpu;
  std::vector<std::int32_t> levels_cpu;
  std::vector<std::int16_t> recon_cpu;

  std::int32_t in_offset = 0;
  std::int32_t out_offset = 0;
  for (int b = 0; b < nblocks; ++b) {
    // A deterministic mix of sizes and types.
    const int s = (b * 7 + 3) % 25;
    const int t = (b * 5 + 1) % 16;
    const auto p = make_tx_params(static_cast<TxSize>(s),
                                  static_cast<TxType>(t));
    av2enc::cuda::TxBlockJob& job = jobs[static_cast<std::size_t>(b)];
    job.params = *p;
    job.in_offset = in_offset;
    job.out_offset = out_offset;

    const int adj_n = p->w * p->h;
    std::vector<std::int16_t> block(static_cast<std::size_t>(adj_n));
    fill_random(block.data(), adj_n, &rng);
    residual.insert(residual.end(), block.begin(), block.end());
    std::vector<double> coeff(static_cast<std::size_t>(adj_n));
    std::vector<std::int32_t> levels(static_cast<std::size_t>(adj_n));
    forward_transform(block.data(), *p, coeff.data(), levels.data());
    coeff_cpu.insert(coeff_cpu.end(), coeff.begin(), coeff.end());
    levels_cpu.insert(levels_cpu.end(), levels.begin(), levels.end());

    std::vector<std::int32_t> dequant(static_cast<std::size_t>(adj_n));
    std::vector<std::int16_t> recon(
        static_cast<std::size_t>(p->coded_w) * p->coded_h);
    dequantize(levels.data(), *p, dequant.data());
    inverse_block(dequant.data(), *p, recon.data());
    recon_cpu.insert(recon_cpu.end(), recon.begin(), recon.end());

    in_offset += static_cast<std::int32_t>(adj_n);
    out_offset += static_cast<std::int32_t>(p->coded_w * p->coded_h);
  }

  const int total_adj = static_cast<int>(residual.size());
  CUDA_TRY(cudaMemcpy(g_dev.res, residual.data(),
                      sizeof(std::int16_t) * static_cast<std::size_t>(total_adj),
                      cudaMemcpyHostToDevice),
           "multi res");
  CUDA_TRY(cudaMemcpy(g_dev.jobs, jobs.data(),
                      sizeof(av2enc::cuda::TxBlockJob) *
                          static_cast<std::size_t>(nblocks),
                      cudaMemcpyHostToDevice),
           "multi jobs");

  CUDA_TRY(av2enc::cuda::txv_forward_transform_multi(
               g_dev.res, g_dev.coeff, g_dev.levels, g_dev.jobs, nblocks,
               nullptr),
           "multi forward");
  compare_exact(coeff_cpu, g_dev.coeff, total_adj, "multi coeff");
  compare_exact(levels_cpu, g_dev.levels, total_adj, "multi levels");

  CUDA_TRY(av2enc::cuda::txv_inverse_transform_multi(
               g_dev.levels, g_dev.coded_out, g_dev.jobs, nblocks, nullptr),
           "multi inverse");
  compare_exact(recon_cpu, g_dev.coded_out,
                static_cast<int>(recon_cpu.size()), "multi recon");
}

void benchmark() {
  struct Case {
    int size;
    TxType type;
    int blocks;
  };
  const Case cases[] = {
      {3, TxType::kDctDct, 8192},      // 32x32 DCT
      {3, TxType::kAdstAdst, 8192},    // 32x32 ADST
      {3, TxType::kIdtx, 8192},        // 32x32 IDTX
      {2, TxType::kDctDct, 32768},     // 16x16 DCT
      {1, TxType::kDctDct, 32768},     // 8x8 DCT
      {4, TxType::kFlipadstFlipadst, 4096},  // 64x64 adjusted 32x32
  };
  std::printf("\n[timing] GPU only; the CPU path is a correctness oracle\n");
  for (const Case& c : cases) {
    const auto p = make_tx_params(static_cast<TxSize>(c.size), c.type);
    if (!p.has_value()) {
      continue;
    }
    const int adj_n = p->w * p->h;
    const int coded_n = p->coded_w * p->coded_h;
    const std::size_t total_adj =
        static_cast<std::size_t>(c.blocks) * adj_n;
    const std::size_t total_coded =
        static_cast<std::size_t>(c.blocks) * coded_n;
    std::vector<std::int16_t> residual(total_adj, 13);
    std::int16_t* d_res = nullptr;
    double* d_coeff = nullptr;
    std::int32_t* d_levels = nullptr;
    std::int16_t* d_recon = nullptr;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_res),
                        sizeof(std::int16_t) * total_adj),
             "bench res");
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_coeff),
                        sizeof(double) * total_adj),
             "bench coeff");
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_levels),
                        sizeof(std::int32_t) * total_adj),
             "bench levels");
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_recon),
                        sizeof(std::int16_t) * total_coded),
             "bench recon");
    CUDA_TRY(cudaMemcpy(d_res, residual.data(),
                        sizeof(std::int16_t) * total_adj,
                        cudaMemcpyHostToDevice),
             "bench res");

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_TRY(cudaEventCreate(&start), "event");
    CUDA_TRY(cudaEventCreate(&stop), "event");

    const int iters = 5;
    float forward_ms = 0.0f;
    float inverse_ms = 0.0f;
    CUDA_TRY(cudaEventRecord(start), "record");
    for (int i = 0; i < iters; ++i) {
      CUDA_TRY(av2enc::cuda::txv_forward_transform(
                   d_res, d_coeff, d_levels, *p, c.blocks, nullptr),
               "bench fwd");
    }
    CUDA_TRY(cudaEventRecord(stop), "record");
    CUDA_TRY(cudaEventSynchronize(stop), "sync");
    CUDA_TRY(cudaEventElapsedTime(&forward_ms, start, stop), "elapsed");

    CUDA_TRY(cudaEventRecord(start), "record");
    for (int i = 0; i < iters; ++i) {
      CUDA_TRY(av2enc::cuda::txv_inverse_transform(d_levels, d_recon, *p,
                                                   c.blocks, nullptr),
               "bench inv");
    }
    CUDA_TRY(cudaEventRecord(stop), "record");
    CUDA_TRY(cudaEventSynchronize(stop), "sync");
    CUDA_TRY(cudaEventElapsedTime(&inverse_ms, start, stop), "elapsed");

    const double fwd_us = forward_ms * 1000.0 / (iters * c.blocks);
    const double inv_us = inverse_ms * 1000.0 / (iters * c.blocks);
    std::printf(
        "  %2dx%-2d %-16s %6d blocks: fwd(tx+quant)=%8.3f ms "
        "(%6.3f us/block) | inv=%8.3f ms (%6.3f us/block)\n",
        p->w, p->h, tx_type_name(c.type), c.blocks,
        forward_ms / iters, fwd_us, inverse_ms / iters, inv_us);

    CUDA_TRY(cudaEventDestroy(start), "event");
    CUDA_TRY(cudaEventDestroy(stop), "event");
    CUDA_TRY(cudaFree(d_res), "free");
    CUDA_TRY(cudaFree(d_coeff), "free");
    CUDA_TRY(cudaFree(d_levels), "free");
    CUDA_TRY(cudaFree(d_recon), "free");
  }
}

}  // namespace

int main() {
  cudaDeviceProp prop{};
  CUDA_TRY(cudaGetDeviceProperties(&prop, 0), "device props");
  std::printf("av2enc CUDA transform-variants test\n");
  std::printf("device: %s (sm_%d%d, %d SMs, %.1f GiB)\n", prop.name,
              prop.major, prop.minor, prop.multiProcessorCount,
              static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 *
                                                          1024.0));

  allocate();
  test_all_sizes_types();
  test_impulses();
  test_multi_params();
  benchmark();

  CUDA_TRY(cudaFree(g_dev.res), "free");
  CUDA_TRY(cudaFree(g_dev.coded), "free");
  CUDA_TRY(cudaFree(g_dev.adj_out), "free");
  CUDA_TRY(cudaFree(g_dev.coded_out), "free");
  CUDA_TRY(cudaFree(g_dev.coeff), "free");
  CUDA_TRY(cudaFree(g_dev.levels), "free");
  CUDA_TRY(cudaFree(g_dev.dequant), "free");
  CUDA_TRY(cudaFree(g_dev.jobs), "free");

  std::printf("\n%lld checks, %d failures\n", g_checks, g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
