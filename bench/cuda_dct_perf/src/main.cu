// Standalone precision and performance benchmark for the M0 forward DCT.
//
// This target lives entirely under bench/cuda_dct_perf/ and is not wired into
// the root CMake project.  It compares three GPU arithmetic paths against a
// line-for-line CPU transcription of src/pipeline/still/basic_encoder.cpp:
//
//   double : full FP64 separable DCT, -fmad=false + explicit __fadd_rn/__fmul_rn
//   float  : full FP32 separable DCT
//   hybrid : exact int32 row dot product + the CPU's double scale/column pass
//   fixed  : Q20 integer row/column pass with a corrected Q60 quant multiplier
//
// Usage:
//   ./cuda_dct_perf [--y4m PATH] [--quick]

#include "common.cuh"
#include "cpu_reference.h"

#include "src/pipeline/still/basic_tables.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kLuma32Nominal = 2040;    // 60 * 34: flat 32x32 grid over 1920x1080
constexpr int kChroma16Nominal = 4080;  // 60 * 34 * 2 planes over 960x540
// M0 k32x32 mode actual transform counts (see README.md).
constexpr int kLuma32Actual = 1950;     // 480 SBs * 4 + 30 edge 64x64->32x32
constexpr int kChroma16Actual = 3840;   // 480 SBs * 4 luma blocks * 2 planes
constexpr int kChroma32Actual = 60;     // 30 edge SBs * 2 planes

struct Frame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool load_y4m_first_frame(const char* path, Frame* frame, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + std::string(path);
    return false;
  }
  std::string line;
  if (!std::getline(in, line) || line.rfind("YUV4MPEG2", 0) != 0) {
    *error = "not a YUV4MPEG2 file";
    return false;
  }
  int width = 0;
  int height = 0;
  std::istringstream header(line);
  std::string token;
  while (header >> token) {
    if (token.size() > 1 && token[0] == 'W') {
      width = std::atoi(token.c_str() + 1);
    } else if (token.size() > 1 && token[0] == 'H') {
      height = std::atoi(token.c_str() + 1);
    }
  }
  if (width <= 0 || height <= 0) {
    *error = "missing W/H in y4m header";
    return false;
  }
  if (!std::getline(in, line) || line.rfind("FRAME", 0) != 0) {
    *error = "missing FRAME header";
    return false;
  }
  frame->width = width;
  frame->height = height;
  const int cw = (width + 1) / 2;
  const int ch = (height + 1) / 2;
  frame->y.resize(static_cast<std::size_t>(width) * height);
  frame->u.resize(static_cast<std::size_t>(cw) * ch);
  frame->v.resize(static_cast<std::size_t>(cw) * ch);
  in.read(reinterpret_cast<char*>(frame->y.data()),
          static_cast<std::streamsize>(frame->y.size()));
  in.read(reinterpret_cast<char*>(frame->u.data()),
          static_cast<std::streamsize>(frame->u.size()));
  in.read(reinterpret_cast<char*>(frame->v.data()),
          static_cast<std::streamsize>(frame->v.size()));
  if (!in) {
    *error = "truncated y4m frame";
    return false;
  }
  return true;
}

std::uint8_t sample_clamped(const std::uint8_t* plane, int w, int h, int x,
                            int y) {
  x = std::min(std::max(x, 0), w - 1);
  y = std::min(std::max(y, 0), h - 1);
  return plane[static_cast<std::size_t>(y) * w + x];
}

// 32x32 / 16x16 blocks with a DC (block-mean) predictor.  M0 uses reconstructed
// neighbours for the predictor; the block mean gives the same residual
// statistics for a precision benchmark and keeps the reference standalone.
void extract_blocks(const std::uint8_t* plane, int w, int h, int n,
                    std::vector<std::int16_t>* blocks,
                    std::vector<int>* preds) {
  const int bx_count = (w + n - 1) / n;
  const int by_count = (h + n - 1) / n;
  blocks->resize(static_cast<std::size_t>(bx_count) * by_count * n * n);
  preds->resize(static_cast<std::size_t>(bx_count) * by_count);
  std::size_t b = 0;
  for (int by = 0; by < by_count; ++by) {
    for (int bx = 0; bx < bx_count; ++bx) {
      const int x0 = bx * n;
      const int y0 = by * n;
      long long sum = 0;
      for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
          sum += sample_clamped(plane, w, h, x0 + x, y0 + y);
        }
      }
      const int pred =
          static_cast<int>(std::lround(static_cast<double>(sum) / (n * n)));
      (*preds)[b] = pred;
      for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
          const int v = static_cast<int>(
                            sample_clamped(plane, w, h, x0 + x, y0 + y)) -
                        pred;
          (*blocks)[b * n * n + static_cast<std::size_t>(y) * n + x] =
              static_cast<std::int16_t>(v);
        }
      }
      ++b;
    }
  }
}

void make_random_blocks(int n, int nblocks, std::uint32_t seed,
                        std::vector<std::int16_t>* blocks,
                        std::vector<int>* preds) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-255, 255);
  blocks->resize(static_cast<std::size_t>(nblocks) * n * n);
  preds->assign(nblocks, 0);
  for (std::int16_t& v : *blocks) {
    v = static_cast<std::int16_t>(dist(rng));
  }
}

struct Batch {
  int n = 0;
  std::vector<std::int16_t> blocks;
  std::vector<int> preds;
  std::vector<double> ref_coeff;
  std::vector<std::int32_t> ref_q;
};

void build_reference(Batch* batch, const BlockParams& p) {
  const int n = batch->n;
  const int n2 = n * n;
  const std::size_t nblocks = batch->preds.size();
  batch->ref_coeff.resize(nblocks * n2);
  batch->ref_q.resize(nblocks * n2);
  std::vector<double> tmp(n2);
  for (std::size_t b = 0; b < nblocks; ++b) {
    ref::forward_dct_quant(batch->blocks.data() + b * n2, n, p, tmp.data(),
                           batch->ref_coeff.data() + b * n2,
                           batch->ref_q.data() + b * n2);
  }
}

struct Metrics {
  long long blocks = 0;
  long long coeffs = 0;
  long long coeff_exact = 0;
  bool coeff_exact_available = true;
  double max_abs_coeff = 0.0;
  double rms_coeff = 0.0;
  double max_rel_coeff = 0.0;
  long long q_mismatch = 0;
  long long q_mismatch_dc = 0;
  long long q_mismatch_lf = 0;
  long long q_mismatch_hf = 0;
  long long blocks_with_q_mismatch = 0;
  long long max_abs_dq = 0;
  long long near_1e3 = 0;
  long long near_1e4 = 0;
  long long near_1e5 = 0;
  bool near_available = false;
  double recon_psnr = std::numeric_limits<double>::infinity();
  long long recon_diff_pixels = 0;
  int max_recon_diff = 0;
};

template <typename F>
double time_launches(F&& once, int iters, int runs) {
  for (int i = 0; i < 20; ++i) {
    once();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  double best_ms = std::numeric_limits<double>::infinity();
  for (int r = 0; r < runs; ++r) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
      once();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    const double per_launch = static_cast<double>(ms) / iters;
    best_ms = std::min(best_ms, per_launch);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  // This is a shared box.  The minimum rejects transient contention from
  // another tenant.
  return best_ms;
}

Metrics run_accuracy(const Batch& batch, const BlockParams& p, Variant variant,
                     int warps, bool fixed_naive, bool* failed) {
  const int n = batch.n;
  const int n2 = n * n;
  const std::size_t nblocks = batch.preds.size();
  const bool has_coeff = variant == Variant::kDouble ||
                         variant == Variant::kFloat ||
                         variant == Variant::kHybrid;
  std::int16_t* d_in = nullptr;
  std::int32_t* d_q = nullptr;
  void* d_coeff = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, nblocks * n2 * sizeof(std::int16_t)));
  CUDA_CHECK(cudaMalloc(&d_q, nblocks * n2 * sizeof(std::int32_t)));
  if (has_coeff) {
    const std::size_t bytes = nblocks * n2 *
                              (variant == Variant::kFloat ? sizeof(float)
                                                          : sizeof(double));
    CUDA_CHECK(cudaMalloc(&d_coeff, bytes));
  }
  CUDA_CHECK(cudaMemcpy(d_in, batch.blocks.data(),
                        nblocks * n2 * sizeof(std::int16_t),
                        cudaMemcpyHostToDevice));
  launch_transform(variant, n, warps, true, d_in,
                   static_cast<int>(nblocks), p, d_coeff, d_q, fixed_naive,
                   nullptr);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::int32_t> gpu_q(nblocks * n2);
  CUDA_CHECK(cudaMemcpy(gpu_q.data(), d_q,
                        nblocks * n2 * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost));
  std::vector<double> gpu_coeff_d;
  std::vector<float> gpu_coeff_f;
  if (variant == Variant::kDouble || variant == Variant::kHybrid) {
    gpu_coeff_d.resize(nblocks * n2);
    CUDA_CHECK(cudaMemcpy(gpu_coeff_d.data(), d_coeff,
                          nblocks * n2 * sizeof(double),
                          cudaMemcpyDeviceToHost));
  } else if (variant == Variant::kFloat) {
    gpu_coeff_f.resize(nblocks * n2);
    CUDA_CHECK(cudaMemcpy(gpu_coeff_f.data(), d_coeff,
                          nblocks * n2 * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  Metrics m;
  m.blocks = static_cast<long long>(nblocks);
  m.coeffs = static_cast<long long>(nblocks) * n2;
  m.coeff_exact_available = has_coeff;
  const bool has_numeric_coeff = has_coeff;
  double sumsq = 0.0;
  const float qf_f = static_cast<float>(p.quant_factor);
  for (std::size_t b = 0; b < nblocks; ++b) {
    bool block_mismatch = false;
    for (int i = 0; i < n2; ++i) {
      const std::size_t pos = b * n2 + i;
      if (has_numeric_coeff) {
        double ref_c = batch.ref_coeff[pos];
        double got_c = 0.0;
        if (variant == Variant::kFloat) {
          got_c = static_cast<double>(gpu_coeff_f[pos]);
        } else {
          got_c = gpu_coeff_d[pos];
        }
        if (variant == Variant::kFloat) {
          if (got_c == ref_c) {
            ++m.coeff_exact;
          }
        } else if (std::memcmp(&gpu_coeff_d[pos], &batch.ref_coeff[pos],
                               sizeof(double)) == 0) {
          ++m.coeff_exact;
        }
        const double diff = std::fabs(got_c - ref_c);
        m.max_abs_coeff = std::max(m.max_abs_coeff, diff);
        sumsq += diff * diff;
        m.max_rel_coeff =
            std::max(m.max_rel_coeff,
                     diff / std::max(std::fabs(ref_c), 1.0));
        if (variant == Variant::kFloat) {
          const float yf = gpu_coeff_f[pos] * qf_f;
          // Distance to the nearest rounding boundary k+0.5.  A zero
          // coefficient sits 0.5 away from its boundary, so it is not counted.
          const double y = static_cast<double>(yf);
          const double dist = std::fabs(y - (std::floor(y) + 0.5));
          if (dist <= 1e-3) {
            ++m.near_1e3;
          }
          if (dist <= 1e-4) {
            ++m.near_1e4;
          }
          if (dist <= 1e-5) {
            ++m.near_1e5;
          }
        }
      }
      const std::int32_t rq = batch.ref_q[pos];
      const std::int32_t gq = gpu_q[pos];
      if (rq != gq) {
        ++m.q_mismatch;
        const int row = i / n;
        const int col = i % n;
        if (i == 0) {
          ++m.q_mismatch_dc;
        } else if (row + col < 4) {
          ++m.q_mismatch_lf;
        } else {
          ++m.q_mismatch_hf;
        }
        block_mismatch = true;
        m.max_abs_dq =
            std::max<long long>(m.max_abs_dq, std::llabs(
                                                  static_cast<long long>(gq) -
                                                  static_cast<long long>(rq)));
      }
    }
    if (block_mismatch) {
      ++m.blocks_with_q_mismatch;
    }
  }
  if (has_numeric_coeff) {
    m.rms_coeff = std::sqrt(sumsq / static_cast<double>(m.coeffs));
  }
  m.near_available = variant == Variant::kFloat;

  // Reconstruction impact of every quantized-coefficient difference.
  const std::size_t pixels = nblocks * n2;
  double sse = 0.0;
  std::vector<int> ref_res(n2);
  std::vector<int> gpu_res(n2);
  for (std::size_t b = 0; b < nblocks; ++b) {
    bool block_mismatch = false;
    for (int i = 0; i < n2; ++i) {
      if (batch.ref_q[b * n2 + i] != gpu_q[b * n2 + i]) {
        block_mismatch = true;
        break;
      }
    }
    if (!block_mismatch) {
      continue;
    }
    ref::inverse_dct_from_q(batch.ref_q.data() + b * n2, n, p, ref_res.data());
    ref::inverse_dct_from_q(gpu_q.data() + b * n2, n, p, gpu_res.data());
    const int pred = batch.preds[b];
    for (int i = 0; i < n2; ++i) {
      const int a = ref::clip1(pred + ref_res[i]);
      const int c = ref::clip1(pred + gpu_res[i]);
      const int d = a - c;
      sse += static_cast<double>(d) * d;
      if (d != 0) {
        ++m.recon_diff_pixels;
        m.max_recon_diff = std::max(m.max_recon_diff, std::abs(d));
      }
    }
  }
  if (sse > 0.0) {
    m.recon_psnr =
        10.0 * std::log10(255.0 * 255.0 * static_cast<double>(pixels) / sse);
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_q));
  if (d_coeff != nullptr) {
    CUDA_CHECK(cudaFree(d_coeff));
  }

  if ((variant == Variant::kDouble || variant == Variant::kHybrid) &&
      (m.coeff_exact != m.coeffs || m.q_mismatch != 0)) {
    *failed = true;
  }
  return m;
}

std::string fmt_psnr(double v) {
  if (!std::isfinite(v)) {
    return "inf";
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2f", v);
  return buf;
}

void print_accuracy_header(const char* size_label, const char* data_label,
                           std::size_t blocks, int n) {
  std::printf("\n#### Accuracy: %s (%s, %zu blocks, %d x %d)\n\n", size_label,
              data_label, blocks, n, n);
  std::printf(
      "| variant | coeff bit-exact | max \\|Δcoeff\\| | RMS Δcoeff | max rel "
      "| q mismatch (DC/LF/HF) | blocks w/ q mismatch | max \\|Δq\\| | near "
      "boundary ≤1e-3/≤1e-4/≤1e-5 | recon PSNR | max recon Δ |\n");
  std::printf(
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
}

void print_accuracy_row(const char* label, const Metrics& m) {
  char exact[64];
  if (m.coeff_exact_available) {
    std::snprintf(exact, sizeof(exact), "%lld/%lld", m.coeff_exact, m.coeffs);
  } else {
    std::snprintf(exact, sizeof(exact), "n/a");
  }
  char coeff[64];
  if (m.coeff_exact_available || m.max_abs_coeff > 0.0) {
    std::snprintf(coeff, sizeof(coeff), "%.3g", m.max_abs_coeff);
  } else {
    std::snprintf(coeff, sizeof(coeff), "n/a");
  }
  char rms[64];
  if (m.coeff_exact_available || m.rms_coeff > 0.0) {
    std::snprintf(rms, sizeof(rms), "%.3g", m.rms_coeff);
  } else {
    std::snprintf(rms, sizeof(rms), "n/a");
  }
  char rel[64];
  if (m.coeff_exact_available || m.max_rel_coeff > 0.0) {
    std::snprintf(rel, sizeof(rel), "%.3g", m.max_rel_coeff);
  } else {
    std::snprintf(rel, sizeof(rel), "n/a");
  }
  char near[64];
  if (m.near_available) {
    std::snprintf(near, sizeof(near), "%lld/%lld/%lld", m.near_1e3, m.near_1e4,
                  m.near_1e5);
  } else {
    std::snprintf(near, sizeof(near), "n/a");
  }
  char qmis[64];
  std::snprintf(qmis, sizeof(qmis), "%lld (%lld/%lld/%lld)", m.q_mismatch,
                m.q_mismatch_dc, m.q_mismatch_lf, m.q_mismatch_hf);
  std::printf("| %s | %s | %s | %s | %s | %s | %lld | %lld | %s | %s | %d |\n",
              label, exact, coeff, rms, rel, qmis,
              m.blocks_with_q_mismatch, m.max_abs_dq, near,
              fmt_psnr(m.recon_psnr).c_str(), m.max_recon_diff);
}

struct TimingRow {
  int n = 0;
  Variant variant = Variant::kDouble;
  int warps = 0;
  bool write_coeff = false;
  double ms = 0.0;
  double ns_per_block = 0.0;
  double mblocks_per_s = 0.0;
};

double gflops_for(int n, double blocks_per_s) {
  // 2 passes * n^2 outputs * n mul-add * 2 FLOP.
  return 4.0 * n * n * n * blocks_per_s / 1e9;
}

void print_timing_table(const std::vector<TimingRow>& rows,
                        const char* title) {
  std::printf("\n#### %s\n\n", title);
  std::printf(
      "| size | variant | output | threads | ms/launch | ns/block | "
      "Mblocks/s | GFLOP/s* |\n");
  std::printf("|---|---|---|---:|---:|---:|---:|---:|\n");
  for (const TimingRow& r : rows) {
    std::printf("| %dx%d | %s | %s | %d | %.4f | %.1f | %.2f | %.1f |\n", r.n,
                r.n, variant_name(r.variant),
                r.write_coeff ? "coeff+q" : "q", r.warps * 32, r.ms,
                r.ns_per_block, r.mblocks_per_s, gflops_for(r.n, r.mblocks_per_s * 1e6));
  }
  std::printf(
      "\n\\* GFLOP/s counts 2 passes × n² outputs × n mul-add × 2 FLOP; the "
      "fixed variant uses integer ALU and its value is nominal.\n");
}

int run(int argc, char** argv) {
  const char* y4m_path = nullptr;
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--y4m") == 0 && i + 1 < argc) {
      y4m_path = argv[++i];
    } else if (std::strcmp(argv[i], "--quick") == 0) {
      quick = true;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int driver_version = 0;
  int runtime_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  std::printf("# CUDA DCT precision/performance benchmark\n\n");
  std::printf("- GPU: %s\n", prop.name);
  std::printf("- compute capability: %d.%d, SMs: %d\n", prop.major, prop.minor,
              prop.multiProcessorCount);
  std::printf("- global memory: %.1f MiB, max clock: %.0f MHz\n",
              static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0),
              static_cast<double>(prop.clockRate) / 1000.0);
  std::printf("- CUDA driver %d.%d, runtime %d.%d, nvcc %d.%d.%d\n",
              driver_version / 1000, (driver_version % 1000) / 10,
              runtime_version / 1000, (runtime_version % 1000) / 10,
              __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__,
              __CUDACC_VER_BUILD__);
  std::printf("- loaded kernel SASS binaryVersion: %d\n",
              loaded_binary_version());

  init_device_tables(&av2enc::basic::kDct32[0][0], &av2enc::basic::kDct16[0][0]);
  const BlockParams p32 = ref::make_block_params(32);
  const BlockParams p16 = ref::make_block_params(16);
  std::printf(
      "- params 32x32: scale_w=%.17g quant_factor=%.17g dq_denom=%d "
      "row_mul=%d quant_mul=%lld\n",
      p32.scale_w, p32.quant_factor, p32.dq_denom, p32.row_mul,
      static_cast<long long>(p32.quant_mul));
  std::printf(
      "- params 16x16: scale_w=%.17g quant_factor=%.17g dq_denom=%d "
      "row_mul=%d quant_mul=%lld\n",
      p16.scale_w, p16.quant_factor, p16.dq_denom, p16.row_mul,
      static_cast<long long>(p16.quant_mul));

  bool failed = false;
  const int random_blocks_32 = quick ? 256 : 2048;
  const int random_blocks_16 = quick ? 512 : 4096;

  {
    Batch batch;
    batch.n = 32;
    make_random_blocks(32, random_blocks_32, 0x5eed1234u, &batch.blocks,
                       &batch.preds);
    build_reference(&batch, p32);
    print_accuracy_header("32x32", "uniform random residual", batch.preds.size(),
                          32);
    const int warps = 8;
    print_accuracy_row("double",
                       run_accuracy(batch, p32, Variant::kDouble, warps, false,
                                    &failed));
    print_accuracy_row("float",
                       run_accuracy(batch, p32, Variant::kFloat, warps, false,
                                    &failed));
    print_accuracy_row("hybrid",
                       run_accuracy(batch, p32, Variant::kHybrid, warps, false,
                                    &failed));
    print_accuracy_row("fixed",
                       run_accuracy(batch, p32, Variant::kFixed, warps, false,
                                    &failed));
    print_accuracy_row("fixed (no δ fix)",
                       run_accuracy(batch, p32, Variant::kFixed, warps, true,
                                    &failed));
  }
  {
    Batch batch;
    batch.n = 16;
    make_random_blocks(16, random_blocks_16, 0x5eed5678u, &batch.blocks,
                       &batch.preds);
    build_reference(&batch, p16);
    print_accuracy_header("16x16", "uniform random residual", batch.preds.size(),
                          16);
    const int warps = 4;
    print_accuracy_row("double",
                       run_accuracy(batch, p16, Variant::kDouble, warps, false,
                                    &failed));
    print_accuracy_row("float",
                       run_accuracy(batch, p16, Variant::kFloat, warps, false,
                                    &failed));
    print_accuracy_row("hybrid",
                       run_accuracy(batch, p16, Variant::kHybrid, warps, false,
                                    &failed));
    print_accuracy_row("fixed",
                       run_accuracy(batch, p16, Variant::kFixed, warps, false,
                                    &failed));
    print_accuracy_row("fixed (no δ fix)",
                       run_accuracy(batch, p16, Variant::kFixed, warps, true,
                                    &failed));
  }

  Frame frame;
  if (y4m_path != nullptr) {
    std::string error;
    if (!load_y4m_first_frame(y4m_path, &frame, &error)) {
      std::fprintf(stderr, "y4m: %s\n", error.c_str());
      return 2;
    }
    std::printf("\n- real frame: %s (%dx%d 4:2:0)\n", y4m_path, frame.width,
                frame.height);
    {
      Batch batch;
      batch.n = 32;
      extract_blocks(frame.y.data(), frame.width, frame.height, 32,
                     &batch.blocks, &batch.preds);
      build_reference(&batch, p32);
      print_accuracy_header("32x32", "shine 1080p, block-mean predictor",
                            batch.preds.size(), 32);
      const int warps = 8;
      print_accuracy_row("double",
                         run_accuracy(batch, p32, Variant::kDouble, warps,
                                      false, &failed));
      print_accuracy_row("float",
                         run_accuracy(batch, p32, Variant::kFloat, warps, false,
                                      &failed));
      print_accuracy_row("hybrid",
                         run_accuracy(batch, p32, Variant::kHybrid, warps,
                                      false, &failed));
      print_accuracy_row("fixed",
                         run_accuracy(batch, p32, Variant::kFixed, warps, false,
                                      &failed));
      print_accuracy_row("fixed (no δ fix)",
                         run_accuracy(batch, p32, Variant::kFixed, warps, true,
                                      &failed));
    }
    {
      Batch batch;
      batch.n = 16;
      std::vector<std::int16_t> u_blocks;
      std::vector<std::int16_t> v_blocks;
      std::vector<int> u_preds;
      std::vector<int> v_preds;
      extract_blocks(frame.u.data(), (frame.width + 1) / 2,
                     (frame.height + 1) / 2, 16, &u_blocks, &u_preds);
      extract_blocks(frame.v.data(), (frame.width + 1) / 2,
                     (frame.height + 1) / 2, 16, &v_blocks, &v_preds);
      batch.blocks = u_blocks;
      batch.preds = u_preds;
      batch.blocks.insert(batch.blocks.end(), v_blocks.begin(), v_blocks.end());
      batch.preds.insert(batch.preds.end(), v_preds.begin(), v_preds.end());
      build_reference(&batch, p16);
      print_accuracy_header("16x16", "shine 1080p chroma, block-mean predictor",
                            batch.preds.size(), 16);
      const int warps = 4;
      print_accuracy_row("double",
                         run_accuracy(batch, p16, Variant::kDouble, warps,
                                      false, &failed));
      print_accuracy_row("float",
                         run_accuracy(batch, p16, Variant::kFloat, warps, false,
                                      &failed));
      print_accuracy_row("hybrid",
                         run_accuracy(batch, p16, Variant::kHybrid, warps,
                                      false, &failed));
      print_accuracy_row("fixed",
                         run_accuracy(batch, p16, Variant::kFixed, warps, false,
                                      &failed));
      print_accuracy_row("fixed (no δ fix)",
                         run_accuracy(batch, p16, Variant::kFixed, warps, true,
                                      &failed));
    }
  }

  // -------------------------------------------------------------------------
  // CPU reference timing (single thread, L1-hot block, preallocated buffers).
  // -------------------------------------------------------------------------
  std::printf("\n#### CPU reference (single thread, -O3 -ffp-contract=off)\n\n");
  std::printf("| size | ns/block | Mblocks/s | projected 1080p frame (ms) |\n");
  std::printf("|---|---:|---:|---:|\n");
  for (int n : {32, 16}) {
    const BlockParams p = (n == 32) ? p32 : p16;
    const int count = (n == 32) ? 1000 : 4000;
    const int reps = 3;
    std::vector<std::int16_t> block(static_cast<std::size_t>(n) * n);
    std::mt19937 rng(0xabc00000u + static_cast<unsigned>(n));
    std::uniform_int_distribution<int> dist(-255, 255);
    for (std::int16_t& v : block) {
      v = static_cast<std::int16_t>(dist(rng));
    }
    std::vector<double> tmp(static_cast<std::size_t>(n) * n);
    std::vector<double> coeff(static_cast<std::size_t>(n) * n);
    std::vector<std::int32_t> q(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < 50; ++i) {
      ref::forward_dct_quant(block.data(), n, p, tmp.data(), coeff.data(),
                             q.data());
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
      for (int b = 0; b < count; ++b) {
        ref::forward_dct_quant(block.data(), n, p, tmp.data(), coeff.data(),
                               q.data());
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns =
        std::chrono::duration<double, std::nano>(t1 - t0).count() /
        static_cast<double>(reps * count);
    const double frame_ms =
        ((n == 32) ? kLuma32Nominal : kChroma16Nominal) * ns / 1e6;
    std::printf("| %dx%d | %.1f | %.2f | %.2f |\n", n, n, ns, 1e3 / ns,
                frame_ms);
  }

  // -------------------------------------------------------------------------
  // GPU throughput, q-only and coeff+q, all warp configurations.
  // -------------------------------------------------------------------------
  CUDA_CHECK(cudaDeviceSynchronize());
  // Clock warm-up: enough FP64 work to leave the idle P8 state.
  {
    const int nblocks = 8192;
    constexpr int n = 32;
    std::vector<std::int16_t> h(static_cast<std::size_t>(nblocks) * n * n, 1);
    std::int16_t* d_in = nullptr;
    std::int32_t* d_q = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, h.size() * sizeof(std::int16_t)));
    CUDA_CHECK(cudaMalloc(&d_q, h.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), h.size() * sizeof(std::int16_t),
                          cudaMemcpyHostToDevice));
    for (int i = 0; i < 100; ++i) {
      launch_transform(Variant::kDouble, n, 8, false, d_in, nblocks, p32,
                       nullptr, d_q, false, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_q));
  }

  std::vector<TimingRow> sweep_rows;
  std::vector<TimingRow> best_rows;
  std::vector<TimingRow> coeff_rows;
  for (int n : {32, 16}) {
    const BlockParams& p = (n == 32) ? p32 : p16;
    const int nblocks = (n == 32) ? 16384 : 65536;
    const int iters = (n == 32) ? 10 : 20;
    const int runs = quick ? 3 : 7;
    const std::size_t elems = static_cast<std::size_t>(nblocks) * n * n;
    std::vector<std::int16_t> h_in(elems);
    std::mt19937 rng(0x13572468u + static_cast<unsigned>(n));
    std::uniform_int_distribution<int> dist(-255, 255);
    for (std::int16_t& v : h_in) {
      v = static_cast<std::int16_t>(dist(rng));
    }
    std::int16_t* d_in = nullptr;
    std::int32_t* d_q = nullptr;
    double* d_coeff_d = nullptr;
    float* d_coeff_f = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, elems * sizeof(std::int16_t)));
    CUDA_CHECK(cudaMalloc(&d_q, elems * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_coeff_d, elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_coeff_f, elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), elems * sizeof(std::int16_t),
                          cudaMemcpyHostToDevice));

    const int warp_options[3] = {n == 32 ? 4 : 2, n == 32 ? 8 : 4,
                                 n == 32 ? 32 : 8};
    for (Variant v : {Variant::kDouble, Variant::kFloat, Variant::kHybrid,
                      Variant::kFixed}) {
      int best_warps = warp_options[0];
      double best_ms = 1e30;
      for (int warps : warp_options) {
        auto launch = [&] {
          launch_transform(v, n, warps, false, d_in, nblocks, p, nullptr, d_q,
                           false, nullptr);
        };
        const double ms = time_launches(launch, iters, runs);
        TimingRow row;
        row.n = n;
        row.variant = v;
        row.warps = warps;
        row.write_coeff = false;
        row.ms = ms;
        row.ns_per_block = ms * 1e6 / nblocks;
        row.mblocks_per_s = nblocks / (ms * 1e3);
        sweep_rows.push_back(row);
        if (ms < best_ms) {
          best_ms = ms;
          best_warps = warps;
        }
      }
      // coeff+q timing at the best q-only configuration.
      if (v != Variant::kFixed) {
        void* d_coeff = (v == Variant::kFloat)
                            ? reinterpret_cast<void*>(d_coeff_f)
                            : reinterpret_cast<void*>(d_coeff_d);
        auto launch = [&] {
          launch_transform(v, n, best_warps, true, d_in, nblocks, p, d_coeff,
                           d_q, false, nullptr);
        };
        const double ms = time_launches(launch, iters, runs);
        TimingRow row;
        row.n = n;
        row.variant = v;
        row.warps = best_warps;
        row.write_coeff = true;
        row.ms = ms;
        row.ns_per_block = ms * 1e6 / nblocks;
        row.mblocks_per_s = nblocks / (ms * 1e3);
        coeff_rows.push_back(row);
      }
      TimingRow best;
      best.n = n;
      best.variant = v;
      best.warps = best_warps;
      best.write_coeff = false;
      best.ms = best_ms;
      best.ns_per_block = best_ms * 1e6 / nblocks;
      best.mblocks_per_s = nblocks / (best_ms * 1e3);
      best_rows.push_back(best);
    }
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_coeff_d));
    CUDA_CHECK(cudaFree(d_coeff_f));
  }

  print_timing_table(sweep_rows,
                     "GPU kernel sweep (q output only, one CUDA block per "
                     "transform; minimum of 7 runs)");
  std::vector<TimingRow> compact;
  for (const TimingRow& r : best_rows) {
    compact.push_back(r);
  }
  for (const TimingRow& r : coeff_rows) {
    compact.push_back(r);
  }
  print_timing_table(compact,
                     "Best warp configuration and coeff+q cost (q-only rows "
                     "repeat the sweep winner; minimum of 7 runs)");

  // -------------------------------------------------------------------------
  // 1080p frame projection.
  // -------------------------------------------------------------------------
  std::printf(
      "\n#### 1080p M0 transform projection (kernel time only, q output)\n\n");
  std::printf(
      "Nominal flat grid: %d luma 32x32 + %d chroma 16x16. "
      "M0 k32x32 actual: %d luma 32x32 + %d chroma 16x16 + %d chroma 32x32 "
      "(edge superblocks).\n\n",
      kLuma32Nominal, kChroma16Nominal, kLuma32Actual, kChroma16Actual,
      kChroma32Actual);
  std::printf(
      "| variant | nominal frame (ms) | max fps | %% of 1 fps budget | M0 "
      "actual frame (ms) | max fps | %% of 1 fps budget |\n");
  std::printf("|---|---:|---:|---:|---:|---:|---:|\n");
  for (Variant v : {Variant::kDouble, Variant::kFloat, Variant::kHybrid,
                    Variant::kFixed}) {
    int warps32 = 8;
    int warps16 = 4;
    for (const TimingRow& r : best_rows) {
      if (r.variant == v && r.n == 32) {
        warps32 = r.warps;
      }
      if (r.variant == v && r.n == 16) {
        warps16 = r.warps;
      }
    }
    const int blocks32 = kLuma32Nominal;
    const int blocks16 = kChroma16Nominal;
    std::vector<std::int16_t> h32(static_cast<std::size_t>(blocks32) * 32 * 32);
    std::vector<std::int16_t> h16(static_cast<std::size_t>(blocks16) * 16 * 16);
    std::int16_t* d32 = nullptr;
    std::int16_t* d16 = nullptr;
    std::int32_t* q32 = nullptr;
    std::int32_t* q16 = nullptr;
    CUDA_CHECK(cudaMalloc(&d32, h32.size() * sizeof(std::int16_t)));
    CUDA_CHECK(cudaMalloc(&d16, h16.size() * sizeof(std::int16_t)));
    CUDA_CHECK(cudaMalloc(&q32, h32.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&q16, h16.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMemcpy(d32, h32.data(), h32.size() * sizeof(std::int16_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d16, h16.data(), h16.size() * sizeof(std::int16_t),
                          cudaMemcpyHostToDevice));

    auto nominal = [&] {
      launch_transform(v, 32, warps32, false, d32, blocks32, p32, nullptr, q32,
                       false, nullptr);
      launch_transform(v, 16, warps16, false, d16, blocks16, p16, nullptr, q16,
                       false, nullptr);
    };
    auto actual = [&] {
      launch_transform(v, 32, warps32, false, d32, kLuma32Actual, p32, nullptr,
                       q32, false, nullptr);
      launch_transform(v, 16, warps16, false, d16, kChroma16Actual, p16,
                       nullptr, q16, false, nullptr);
      launch_transform(v, 32, warps32, false,
                       d32 + static_cast<std::size_t>(kLuma32Actual) * 32 * 32,
                       kChroma32Actual, p32, nullptr,
                       q32 + static_cast<std::size_t>(kLuma32Actual) * 32 * 32,
                       false, nullptr);
    };
    // One frame per sample: a contention spike then ruins at most one sample,
    // and the minimum over 20 samples still finds a clean window.
    const int iters = 1;
    const int runs = quick ? 5 : 20;
    const double nominal_ms = time_launches(nominal, iters, runs);
    const double actual_ms = time_launches(actual, iters, runs);
    std::printf("| %s | %.4f | %.0f | %.4f%% | %.4f | %.0f | %.4f%% |\n",
                variant_name(v), nominal_ms, 1000.0 / nominal_ms,
                100.0 * nominal_ms / 1000.0, actual_ms, 1000.0 / actual_ms,
                100.0 * actual_ms / 1000.0);
    CUDA_CHECK(cudaFree(d32));
    CUDA_CHECK(cudaFree(d16));
    CUDA_CHECK(cudaFree(q32));
    CUDA_CHECK(cudaFree(q16));
  }

  // Single-block latency (includes launch overhead).
  {
    const int n = 32;
    std::int16_t* d_in = nullptr;
    std::int32_t* d_q = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, 32 * 32 * sizeof(std::int16_t)));
    CUDA_CHECK(cudaMalloc(&d_q, 32 * 32 * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMemset(d_in, 0, 32 * 32 * sizeof(std::int16_t)));
    auto one = [&] {
      launch_transform(Variant::kDouble, n, 8, false, d_in, 1, p32, nullptr,
                       d_q, false, nullptr);
    };
    const double one_ms = time_launches(one, 1, quick ? 3 : 7);
    std::printf(
        "\n- single 32x32 double block, one kernel launch: %.2f us\n",
        one_ms * 1e3);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_q));
  }

  shutdown_device_tables();
  if (failed) {
    std::printf("\nFAIL: double or hybrid path is not bit-exact vs the CPU "
                "reference.\n");
    return 1;
  }
  std::printf("\nPASS: double and hybrid paths are bit-exact vs CPU.\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) { return run(argc, argv); }
