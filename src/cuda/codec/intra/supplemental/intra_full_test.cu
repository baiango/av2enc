// Real-GPU test for the intra-full component.
//
// Compares the kernels against the CPU reference components, over:
//   * DC subsampled: every w,h in 1..64, all 4 availability combinations,
//     8/10/12-bit edges, plus the process trace;
//   * filter corner: guard over every w,h in 1..64 and pAngle 0..270, the
//     5/6/5 Round2(.,4) value over every 8-bit triple, and the apply path
//     (both corner slots) over all extents and a set of guard-active angles;
//   * all seven intra identity TX_TYPEs (IDTX, V/H_DCT, V/H_ADST,
//     V/H_FLIPADST) x all 25 TX sizes, forward coefficients (binary64) and
//     coded-size inverse residual, bit-exact;
//   * chroma type derivation over 2 planes x 2 chroma_dctonly values x
//     14 UV modes x 7 angle deltas x 25 sizes.
//
// Memory: every device buffer comes from one av2enc::cuda::Arena.  There is no
// cudaMalloc / cudaMallocHost in this file.  The test also verifies:
//   * required_device_bytes() == actual init growth + largest scratch,
//   * back-to-back launches with mark()/rewind() do not grow the arena,
//   * a one-byte-short arena fails with Status::kArenaExhausted and reports
//     the required size.
//
// Build/run: ./build_gpu_test.sh   (sm_86 default, needs an NVIDIA GPU)
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/error.h"
#include "cuda/codec/intra/supplemental/intra_full.h"
#include "cuda/codec/intra/supplemental/intra_full_device.cuh"
#include "codec/intra/modes/intra_modes.h"
#include "codec/intra/supplemental/dc_intra_subsampled.h"
#include "codec/intra/supplemental/dc_sign_horz_vert.h"
#include "codec/intra/supplemental/intra_filter_corner.h"
#include "codec/transform/idtx/idtx_intra.h"
#include "codec/transform/tx_type/tx_type.h"
#include "codec/transform/tx_type/tx_type_tables.h"
#include "codec/transform/variants/tx_variants.h"

namespace {

using av2enc::cuda::Arena;
using av2enc::cuda::Status;
using av2enc::cuda::intra_full::ChromaQuery;
using av2enc::cuda::intra_full::ChromaResult;
using av2enc::cuda::intra_full::Config;
using av2enc::cuda::intra_full::CornerJob;
using av2enc::cuda::intra_full::CornerResult;
using av2enc::cuda::intra_full::DcJob;
using av2enc::cuda::intra_full::DcResult;
using av2enc::cuda::intra_full::DcSubsampledDesc;
using av2enc::cuda::intra_full::SignContextJob;
using av2enc::cuda::intra_full::SignElement;
using av2enc::cuda::intra_full::SignQuery;
using av2enc::cuda::intra_full::SignResult;
using av2enc::cuda::intra_full::SignTxClass;
using av2enc::cuda::intra_full::TxJob;
using av2enc::cuda::intra_full::Workspace;

long long g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, const char* detail, long long a,
            long long b) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 25) {
      std::printf("FAIL: %s (%s got=%lld want=%lld)\n", what, detail, a, b);
    }
  }
}

#define CHECK(cond) expect((cond), #cond, "", 0, 0)
#define CHECK_EQ(a, b, what)                                          \
  do {                                                                \
    const long long va_ = static_cast<long long>(a);                  \
    const long long vb_ = static_cast<long long>(b);                  \
    expect(va_ == vb_, what, #a " == " #b, va_, vb_);                 \
  } while (0)

std::uint32_t g_rng = 0x243f6a88u;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
int random_range(int low, int high) {
  return low + static_cast<int>(next_random() %
                                static_cast<std::uint32_t>(high - low + 1));
}

bool cuda_ok(cudaError_t error) {
  return av2enc::cuda::status_from_cuda(error) == Status::kOk;
}

Config make_config() {
  Config config;
  config.max_dc_jobs = 4;
  config.max_corner_jobs = 256;
  config.max_chroma_jobs = 256;
  config.max_tx_blocks = 3;
  config.max_sign_jobs = 256;
  return config;
}

constexpr int kMaxDcJobs = 4;
constexpr int kMaxSignJobs = 256;
constexpr int kMaxSamples = 64;
constexpr int kMaxTxElems = 32 * 32;
constexpr int kMaxCodedElems = 64 * 64;
constexpr int kValueCases = 256 * 256 * 256;
constexpr int kGuardCases = 64 * 64 * 271;

struct DeviceData {
  std::int16_t* dc_above = nullptr;
  std::int16_t* dc_left = nullptr;
  std::uint16_t* dc_dst = nullptr;
  std::int16_t* corner_above = nullptr;
  std::int16_t* corner_left = nullptr;
  std::uint8_t* corner_values = nullptr;
  std::uint8_t* corner_guard = nullptr;
  std::int16_t* tx_residual = nullptr;
  double* tx_coeff = nullptr;
  std::int32_t* tx_dequant = nullptr;
  std::int16_t* tx_coded = nullptr;
  std::int8_t* sign_above = nullptr;
  std::int8_t* sign_left = nullptr;
};

// Test-only kernels that call the shared device functions directly.
__global__ void corner_value_kernel(std::uint8_t* __restrict__ out) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= kValueCases) {
    return;
  }
  out[i] = static_cast<std::uint8_t>(
      av2enc::cuda::intra_full::intra_full_filter_corner_value_dev(
          i & 255, (i >> 8) & 255, (i >> 16) & 255));
}

__global__ void corner_guard_kernel(std::uint8_t* __restrict__ out) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= kGuardCases) {
    return;
  }
  const int w = (i % 64) + 1;
  const int h = ((i / 64) % 64) + 1;
  const int angle = i / (64 * 64);
  out[i] = static_cast<std::uint8_t>(
      av2enc::cuda::intra_full::intra_full_filter_corner_applies_dev(w, h,
                                                                     angle));
}

bool alloc_all(Arena& arena, DeviceData* d) {
  d->dc_above = arena.allocate_array<std::int16_t>(kMaxDcJobs * kMaxSamples);
  d->dc_left = arena.allocate_array<std::int16_t>(kMaxDcJobs * kMaxSamples);
  d->dc_dst = arena.allocate_array<std::uint16_t>(kMaxDcJobs * kMaxSamples *
                                                  kMaxSamples);
  d->corner_above = arena.allocate_array<std::int16_t>(256 * 4);
  d->corner_left = arena.allocate_array<std::int16_t>(256 * 4);
  d->corner_values = arena.allocate_array<std::uint8_t>(kValueCases);
  d->corner_guard = arena.allocate_array<std::uint8_t>(kGuardCases);
  d->tx_residual = arena.allocate_array<std::int16_t>(3 * kMaxTxElems);
  d->tx_coeff = arena.allocate_array<double>(3 * kMaxTxElems);
  d->tx_dequant = arena.allocate_array<std::int32_t>(3 * kMaxTxElems);
  d->tx_coded = arena.allocate_array<std::int16_t>(3 * kMaxCodedElems);
  d->sign_above = arena.allocate_array<std::int8_t>(kMaxSignJobs * 64);
  d->sign_left = arena.allocate_array<std::int8_t>(kMaxSignJobs * 64);
  return d->dc_above != nullptr && d->dc_left != nullptr &&
         d->dc_dst != nullptr && d->corner_above != nullptr &&
         d->corner_left != nullptr && d->corner_values != nullptr &&
         d->corner_guard != nullptr && d->tx_residual != nullptr &&
         d->tx_coeff != nullptr && d->tx_dequant != nullptr &&
         d->tx_coded != nullptr && d->sign_above != nullptr &&
         d->sign_left != nullptr;
}

// ---------------------------------------------------------------------------
// §7.13.2.11 DC subsampled
// ---------------------------------------------------------------------------

void test_dc(Workspace& workspace, Arena& arena, DeviceData& d) {
  const std::uint64_t before = arena.used();
  std::vector<std::int16_t> h_above(kMaxDcJobs * kMaxSamples);
  std::vector<std::int16_t> h_left(kMaxDcJobs * kMaxSamples);
  std::vector<std::uint16_t> h_dst(kMaxDcJobs * kMaxSamples * kMaxSamples);
  std::vector<std::uint16_t> want(h_dst.size());
  std::vector<DcResult> traces(kMaxDcJobs);
  std::vector<DcResult> cpu_traces(kMaxDcJobs);

  long long sample_checks = 0;
  int trace_checks = 0;
  for (int bit_depth = 8; bit_depth <= 12; bit_depth += 2) {
    const int high = (1 << bit_depth) - 1;
    for (int w = 1; w <= 64; ++w) {
      for (int h = 1; h <= 64; ++h) {
        DcJob jobs[kMaxDcJobs];
        for (int job = 0; job < kMaxDcJobs; ++job) {
          const int have_left = job >> 1;
          const int have_above = job & 1;
          for (int i = 0; i < w; ++i) {
            h_above[static_cast<std::size_t>(job * kMaxSamples + i)] =
                static_cast<std::int16_t>(random_range(0, high));
          }
          for (int i = 0; i < h; ++i) {
            h_left[static_cast<std::size_t>(job * kMaxSamples + i)] =
                static_cast<std::int16_t>(random_range(0, high));
          }

          av2enc::still::missing_bits::DcSubsampledInput cpu_in;
          cpu_in.above =
              h_above.data() + static_cast<std::size_t>(job * kMaxSamples);
          cpu_in.left =
              h_left.data() + static_cast<std::size_t>(job * kMaxSamples);
          cpu_in.w = w;
          cpu_in.h = h;
          cpu_in.bit_depth = bit_depth;
          cpu_in.have_above = have_above != 0;
          cpu_in.have_left = have_left != 0;
          av2enc::still::missing_bits::DcSubsampledTrace cpu_trace;
          CHECK(av2enc::still::missing_bits::dc_intra_subsampled(
              cpu_in, want.data() + static_cast<std::size_t>(job) * kMaxSamples *
                                         kMaxSamples,
              static_cast<std::size_t>(w), &cpu_trace));
          cpu_traces[static_cast<std::size_t>(job)].step_w = cpu_trace.step_w;
          cpu_traces[static_cast<std::size_t>(job)].step_h = cpu_trace.step_h;
          cpu_traces[static_cast<std::size_t>(job)].sum = cpu_trace.sum;
          cpu_traces[static_cast<std::size_t>(job)].count = cpu_trace.count;
          cpu_traces[static_cast<std::size_t>(job)].avg = cpu_trace.avg;

          DcSubsampledDesc in;
          in.above = d.dc_above + job * kMaxSamples;
          in.left = d.dc_left + job * kMaxSamples;
          in.w = w;
          in.h = h;
          in.bit_depth = bit_depth;
          in.have_above = have_above;
          in.have_left = have_left;
          jobs[job].in = in;
          jobs[job].dst = d.dc_dst + job * kMaxSamples * kMaxSamples;
          jobs[job].dst_stride = w;
        }

        CHECK(cuda_ok(cudaMemcpy(
            d.dc_above, h_above.data(), sizeof(std::int16_t) * h_above.size(),
            cudaMemcpyHostToDevice)));
        CHECK(cuda_ok(cudaMemcpy(
            d.dc_left, h_left.data(), sizeof(std::int16_t) * h_left.size(),
            cudaMemcpyHostToDevice)));
        CHECK_EQ(static_cast<int>(workspace.dc_subsampled_predict(
                     jobs, kMaxDcJobs, traces.data())),
                 static_cast<int>(Status::kOk), "dc launch");
        CHECK(cuda_ok(cudaMemcpy(
            h_dst.data(), d.dc_dst, sizeof(std::uint16_t) * h_dst.size(),
            cudaMemcpyDeviceToHost)));

        for (int job = 0; job < kMaxDcJobs; ++job) {
          const DcResult& got = traces[static_cast<std::size_t>(job)];
          const DcResult& exp = cpu_traces[static_cast<std::size_t>(job)];
          CHECK_EQ(got.step_w, exp.step_w, "dc step_w");
          CHECK_EQ(got.step_h, exp.step_h, "dc step_h");
          CHECK_EQ(got.sum, exp.sum, "dc sum");
          CHECK_EQ(got.count, exp.count, "dc count");
          CHECK_EQ(got.avg, exp.avg, "dc avg");
          trace_checks += 5;

          const std::uint16_t* want_job =
              want.data() + static_cast<std::size_t>(job) * kMaxSamples *
                                kMaxSamples;
          const std::uint16_t* got_job =
              h_dst.data() + static_cast<std::size_t>(job) * kMaxSamples *
                                 kMaxSamples;
          for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
              const std::size_t k = static_cast<std::size_t>(i) *
                                        static_cast<std::size_t>(w) +
                                    static_cast<std::size_t>(j);
              if (got_job[k] != want_job[k]) {
                expect(false, "dc sample", "", got_job[k], want_job[k]);
                return;
              }
              ++g_checks;
              ++sample_checks;
            }
          }
        }
      }
    }
  }
  CHECK_EQ(arena.used(), before, "dc: arena flat after");
  std::printf("  dc subsampled: samples=%lld trace_fields=%d\n", sample_checks,
              trace_checks);
}

// ---------------------------------------------------------------------------
// §7.13.2.14 filter corner
// ---------------------------------------------------------------------------

void test_corner(Workspace& workspace, Arena& arena, DeviceData& d) {
  const std::uint64_t before = arena.used();
  corner_value_kernel<<<(kValueCases + 255) / 256, 256>>>(d.corner_values);
  CHECK(cuda_ok(cudaGetLastError()));
  corner_guard_kernel<<<(kGuardCases + 255) / 256, 256>>>(d.corner_guard);
  CHECK(cuda_ok(cudaGetLastError()));
  CHECK(cuda_ok(cudaDeviceSynchronize()));

  std::vector<std::uint8_t> values(kValueCases);
  std::vector<std::uint8_t> guard(kGuardCases);
  CHECK(cuda_ok(cudaMemcpy(values.data(), d.corner_values,
                                       values.size(), cudaMemcpyDeviceToHost)));
  CHECK(cuda_ok(cudaMemcpy(guard.data(), d.corner_guard,
                                       guard.size(), cudaMemcpyDeviceToHost)));
  for (int i = 0; i < kValueCases; ++i) {
    const int exp = av2enc::still::missing_bits::filter_corner_value(
        i & 255, (i >> 8) & 255, (i >> 16) & 255);
    if (values[static_cast<std::size_t>(i)] !=
        static_cast<std::uint8_t>(exp)) {
      expect(false, "corner value", "", values[static_cast<std::size_t>(i)],
             exp);
      return;
    }
    ++g_checks;
  }
  long long guard_checks = 0;
  for (int i = 0; i < kGuardCases; ++i) {
    const int w = (i % 64) + 1;
    const int h = ((i / 64) % 64) + 1;
    const int angle = i / (64 * 64);
    const bool exp =
        av2enc::still::missing_bits::filter_corner_applies(w, h, angle);
    if ((guard[static_cast<std::size_t>(i)] != 0) != exp) {
      expect(false, "corner guard", "", guard[static_cast<std::size_t>(i)],
             exp ? 1 : 0);
      return;
    }
    ++g_checks;
    ++guard_checks;
  }

  // Apply path: all extents x guard-relevant angles, chunked by the workspace.
  static const int kAngles[] = {0, 45, 89, 90, 91, 120, 135, 179, 180, 181, 269};
  const int angle_count = static_cast<int>(sizeof(kAngles) / sizeof(kAngles[0]));
  const int max_jobs = workspace.config().max_corner_jobs;
  std::vector<CornerJob> jobs(static_cast<std::size_t>(max_jobs));
  std::vector<CornerResult> results(static_cast<std::size_t>(max_jobs));
  std::vector<std::int16_t> original_above(static_cast<std::size_t>(max_jobs) * 4);
  std::vector<std::int16_t> original_left(static_cast<std::size_t>(max_jobs) * 4);
  std::vector<std::int16_t> got_above(original_above.size());
  std::vector<std::int16_t> got_left(original_left.size());
  long long apply_checks = 0;

  auto run_chunk = [&](int count) {
    CHECK(cuda_ok(cudaMemcpy(
        d.corner_above, original_above.data(),
        sizeof(std::int16_t) * static_cast<std::size_t>(count) * 4,
        cudaMemcpyHostToDevice)));
    CHECK(cuda_ok(cudaMemcpy(
        d.corner_left, original_left.data(),
        sizeof(std::int16_t) * static_cast<std::size_t>(count) * 4,
        cudaMemcpyHostToDevice)));
    CHECK_EQ(static_cast<int>(workspace.filter_corner(jobs.data(), count,
                                                      results.data())),
             static_cast<int>(Status::kOk), "corner launch");
    CHECK(cuda_ok(cudaMemcpy(
        got_above.data(), d.corner_above,
        sizeof(std::int16_t) * static_cast<std::size_t>(count) * 4,
        cudaMemcpyDeviceToHost)));
    CHECK(cuda_ok(cudaMemcpy(
        got_left.data(), d.corner_left,
        sizeof(std::int16_t) * static_cast<std::size_t>(count) * 4,
        cudaMemcpyDeviceToHost)));
    for (int i = 0; i < count; ++i) {
      av2enc::intra::EdgeRow above_cpu;
      av2enc::intra::EdgeRow left_cpu;
      above_cpu[-1] = original_above[static_cast<std::size_t>(i) * 4 + 0];
      above_cpu[0] = original_above[static_cast<std::size_t>(i) * 4 + 1];
      left_cpu[-1] = original_left[static_cast<std::size_t>(i) * 4 + 0];
      left_cpu[0] = original_left[static_cast<std::size_t>(i) * 4 + 1];
      const bool applied =
          av2enc::still::missing_bits::filter_corner_if_applicable(
              jobs[static_cast<std::size_t>(i)].w,
              jobs[static_cast<std::size_t>(i)].h,
              jobs[static_cast<std::size_t>(i)].p_angle, above_cpu, left_cpu);
      CHECK_EQ(results[static_cast<std::size_t>(i)].applied, applied ? 1 : 0,
               "corner applied");
      CHECK_EQ(got_above[static_cast<std::size_t>(i) * 4 + 0], above_cpu[-1],
               "corner above[-1]");
      CHECK_EQ(got_above[static_cast<std::size_t>(i) * 4 + 1], above_cpu[0],
               "corner above[0]");
      CHECK_EQ(got_left[static_cast<std::size_t>(i) * 4 + 0], left_cpu[-1],
               "corner left[-1]");
      CHECK_EQ(got_left[static_cast<std::size_t>(i) * 4 + 1], left_cpu[0],
               "corner left[0]");
      apply_checks += 5;
    }
  };

  int pending = 0;
  for (int w = 1; w <= 64; ++w) {
    for (int h = 1; h <= 64; ++h) {
      for (int a = 0; a < angle_count; ++a) {
        const std::size_t base = static_cast<std::size_t>(pending) * 4;
        original_above[base + 0] =
            static_cast<std::int16_t>(random_range(0, 255));
        original_above[base + 1] =
            static_cast<std::int16_t>(random_range(0, 255));
        original_left[base + 0] =
            static_cast<std::int16_t>(random_range(0, 255));
        original_left[base + 1] =
            static_cast<std::int16_t>(random_range(0, 255));
        CornerJob& job = jobs[static_cast<std::size_t>(pending)];
        job.above = d.corner_above + static_cast<std::size_t>(pending) * 4 + 1;
        job.left = d.corner_left + static_cast<std::size_t>(pending) * 4 + 1;
        job.w = w;
        job.h = h;
        job.p_angle = kAngles[a];
        ++pending;
        if (pending == max_jobs) {
          run_chunk(pending);
          pending = 0;
        }
      }
    }
  }
  if (pending != 0) {
    run_chunk(pending);
  }
  CHECK_EQ(arena.used(), before, "corner: arena flat after");
  std::printf("  corner: values=%d guard=%lld apply_checks=%lld\n", kValueCases,
              guard_checks, apply_checks);
}

// ---------------------------------------------------------------------------
// Identity transforms
// ---------------------------------------------------------------------------

void test_transforms(Workspace& workspace, Arena& arena, DeviceData& d) {
  const std::uint64_t before = arena.used();
  using av2enc::txv::TxSize;
  using av2enc::txv::TxType;
  long long coeff_checks = 0;
  long long coded_checks = 0;
  int cases = 0;

  for (int t = 9; t <= 15; ++t) {
    for (int s = 0; s < 25; ++s) {
      const auto params = av2enc::txv::make_tx_params(
          static_cast<TxSize>(s), static_cast<TxType>(t));
      if (!params.has_value()) {
        continue;
      }
      ++cases;
      const int adj_n = params->w * params->h;
      const int coded_n = params->coded_w * params->coded_h;
      const int blocks = 3;
      const std::size_t adj_sz = static_cast<std::size_t>(adj_n);
      const std::size_t coded_sz = static_cast<std::size_t>(coded_n);
      const std::size_t blocks_sz = static_cast<std::size_t>(blocks);
      std::vector<std::int16_t> residual(blocks_sz * adj_sz);
      std::vector<double> want_coeff(blocks_sz * adj_sz);
      std::vector<std::int32_t> levels(blocks_sz * adj_sz);
      std::vector<std::int32_t> dequant(blocks_sz * adj_sz);
      std::vector<std::int16_t> want_coded(blocks_sz * coded_sz);
      std::vector<TxJob> jobs(blocks_sz);

      for (int b = 0; b < blocks; ++b) {
        std::int16_t* res =
            residual.data() + static_cast<std::size_t>(b) * adj_sz;
        for (int i = 0; i < adj_n; ++i) {
          if (b == 0) {
            res[i] = static_cast<std::int16_t>(random_range(-255, 255));
          } else if (b == 1) {
            res[i] = static_cast<std::int16_t>(i == adj_n / 2 ? 255 : 0);
          } else {
            res[i] = static_cast<std::int16_t>((i % 2) ? -255 : 255);
          }
        }
        double* coeff_b =
            want_coeff.data() + static_cast<std::size_t>(b) * adj_sz;
        std::int32_t* levels_b =
            levels.data() + static_cast<std::size_t>(b) * adj_sz;
        std::int32_t* dequant_b =
            dequant.data() + static_cast<std::size_t>(b) * adj_sz;
        std::int16_t* coded_b =
            want_coded.data() + static_cast<std::size_t>(b) * coded_sz;
        av2enc::txv::forward_block(res, *params, coeff_b);
        av2enc::txv::quantize(coeff_b, *params, levels_b);
        av2enc::txv::dequantize(levels_b, *params, dequant_b);
        av2enc::txv::inverse_block(dequant_b, *params, coded_b);
        jobs[static_cast<std::size_t>(b)].params = *params;
        jobs[static_cast<std::size_t>(b)].in_offset = b * adj_n;
        jobs[static_cast<std::size_t>(b)].out_offset = b * coded_n;
      }

      CHECK(cuda_ok(cudaMemcpy(
          d.tx_residual, residual.data(),
          sizeof(std::int16_t) * residual.size(), cudaMemcpyHostToDevice)));
      CHECK_EQ(static_cast<int>(workspace.identity_forward(
                   jobs.data(), blocks, d.tx_residual, d.tx_coeff)),
               static_cast<int>(Status::kOk), "tx forward launch");
      CHECK(cuda_ok(cudaMemcpy(
          d.tx_dequant, dequant.data(),
          sizeof(std::int32_t) * dequant.size(), cudaMemcpyHostToDevice)));
      CHECK_EQ(static_cast<int>(workspace.identity_inverse(
                   jobs.data(), blocks, d.tx_dequant, d.tx_coded)),
               static_cast<int>(Status::kOk), "tx inverse launch");

      std::vector<double> got_coeff(want_coeff.size());
      std::vector<std::int16_t> got_coded(want_coded.size());
      CHECK(cuda_ok(cudaMemcpy(
          got_coeff.data(), d.tx_coeff,
          sizeof(double) * got_coeff.size(), cudaMemcpyDeviceToHost)));
      CHECK(cuda_ok(cudaMemcpy(
          got_coded.data(), d.tx_coded,
          sizeof(std::int16_t) * got_coded.size(), cudaMemcpyDeviceToHost)));
      for (std::size_t i = 0; i < got_coeff.size(); ++i) {
        if (std::memcmp(&got_coeff[i], &want_coeff[i], sizeof(double)) != 0) {
          expect(false, "tx forward bit-exact", "", 0, 0);
          return;
        }
        ++g_checks;
        ++coeff_checks;
      }
      for (std::size_t i = 0; i < got_coded.size(); ++i) {
        if (got_coded[i] != want_coded[i]) {
          expect(false, "tx inverse bit-exact", "", got_coded[i],
                 want_coded[i]);
          return;
        }
        ++g_checks;
        ++coded_checks;
      }
    }
  }
  CHECK_EQ(arena.used(), before, "tx: arena flat after");
  std::printf("  identity tx: cases=%d coeffs=%lld coded=%lld\n", cases,
              coeff_checks, coded_checks);
}

// ---------------------------------------------------------------------------
// Chroma type derivation
// ---------------------------------------------------------------------------

void test_chroma(Workspace& workspace, Arena& arena) {
  const std::uint64_t before = arena.used();
  long long derived_checks = 0;
  int cases = 0;
  const int max_jobs = workspace.config().max_chroma_jobs;
  std::vector<ChromaQuery> queries(static_cast<std::size_t>(max_jobs));
  std::vector<ChromaResult> results(static_cast<std::size_t>(max_jobs));

  int pending = 0;
  auto flush = [&]() {
    if (pending == 0) {
      return;
    }
    CHECK_EQ(static_cast<int>(workspace.derive_chroma_type(
                 queries.data(), pending, results.data())),
             static_cast<int>(Status::kOk), "chroma launch");
    for (int i = 0; i < pending; ++i) {
      av2enc::idtx::SeqFlags flags;
      flags.enable_chroma_dctonly =
          queries[static_cast<std::size_t>(i)].enable_chroma_dctonly != 0;
      av2enc::txv::TxType want_type = av2enc::txv::TxType::kDctDct;
      av2enc::idtx::Set want_set = av2enc::idtx::Set::kDctOnly;
      CHECK(av2enc::idtx::derive_chroma_type(
          queries[static_cast<std::size_t>(i)].uv_mode,
          queries[static_cast<std::size_t>(i)].uv_angle_delta,
          queries[static_cast<std::size_t>(i)].coded_w,
          queries[static_cast<std::size_t>(i)].coded_h, flags, &want_type,
          &want_set));
      CHECK_EQ(results[static_cast<std::size_t>(i)].valid, 1, "chroma valid");
      CHECK_EQ(results[static_cast<std::size_t>(i)].tx_type,
               static_cast<int>(want_type), "chroma tx_type");
      CHECK_EQ(results[static_cast<std::size_t>(i)].set,
               static_cast<int>(want_set), "chroma set");
      ++cases;
      derived_checks += 3;
    }
    pending = 0;
  };

  for (int plane = 1; plane <= 2; ++plane) {
    for (int dctonly = 0; dctonly <= 1; ++dctonly) {
      for (int uv_mode = 0; uv_mode <= 13; ++uv_mode) {
        for (int delta = -3; delta <= 3; ++delta) {
          for (int s = 0; s < 25; ++s) {
            const auto size = av2enc::txv::make_tx_params(
                static_cast<av2enc::txv::TxSize>(s));
            ChromaQuery& q = queries[static_cast<std::size_t>(pending)];
            q.uv_mode = uv_mode;
            q.uv_angle_delta = delta;
            q.coded_w = size->coded_w;
            q.coded_h = size->coded_h;
            q.plane = plane;
            q.enable_chroma_dctonly = dctonly;
            ++pending;
            if (pending == max_jobs) {
              flush();
            }
          }
        }
      }
    }
  }
  flush();
  CHECK_EQ(arena.used(), before, "chroma: arena flat after");
  std::printf("  chroma derivation: cases=%d checks=%lld\n", cases,
              derived_checks);
}

// ---------------------------------------------------------------------------
// §5.20.7.27 / §8.3 dc_sign and dc_sign_horz_vert
// ---------------------------------------------------------------------------

void test_dc_sign(Workspace& workspace, Arena& arena, DeviceData& d) {
  namespace mb = av2enc::still::missing_bits;
  const std::uint64_t before = arena.used();
  long long element_checks = 0;
  long long context_checks = 0;

  // 1. Sign-element sweep: every plane, TX class and a 8x8 position grid.
  {
    std::vector<SignQuery> queries;
    std::vector<std::int8_t> above(kMaxSignJobs * 64);
    std::vector<std::int8_t> left(kMaxSignJobs * 64);
    int cases = 0;
    auto flush = [&]() {
      if (queries.empty()) {
        return;
      }
      std::vector<SignResult> got(queries.size());
      CHECK_EQ(static_cast<int>(workspace.select_sign_elements(
                   queries.data(), static_cast<int>(queries.size()),
                   got.data())),
               static_cast<int>(Status::kOk), "dc_sign select launch");
      for (std::size_t i = 0; i < queries.size(); ++i) {
        const SignQuery& q = queries[i];
        const mb::TxClass cpu_class = static_cast<mb::TxClass>(
            static_cast<int>(q.tx_class));
        const mb::SignElement want =
            mb::select_sign_element(q.plane, q.row, q.col, cpu_class);
        CHECK_EQ(got[i].element, static_cast<int>(want), "sign element");
        ++element_checks;
        if (want == mb::SignElement::kDcSign) {
          const int want_ctx = mb::dc_sign_context(
              above.data() + i * 64, left.data() + i * 64, q.x4, q.y4, q.w4,
              q.h4, q.mi_cols, q.mi_rows);
          CHECK_EQ(got[i].ctx, want_ctx, "sign ctx");
        } else {
          CHECK_EQ(got[i].ctx, 0, "non-dc ctx is zero");
        }
        ++element_checks;
      }
      cases += static_cast<int>(queries.size());
      queries.clear();
    };

    std::uint32_t seed = 0x9e3779b9u;
    auto next = [&seed]() {
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;
      return seed;
    };
    for (int plane = 0; plane <= 2; ++plane) {
      for (int tx = 0; tx <= 2; ++tx) {
        for (int row = 0; row < 8; ++row) {
          for (int col = 0; col < 8; ++col) {
            const std::size_t slot = queries.size();
            for (int k = 0; k < 64; ++k) {
              above[slot * 64 + static_cast<std::size_t>(k)] =
                  static_cast<std::int8_t>(next() % 3u);
              left[slot * 64 + static_cast<std::size_t>(k)] =
                  static_cast<std::int8_t>(next() % 3u);
            }
            SignQuery q;
            q.plane = plane;
            q.row = row;
            q.col = col;
            q.tx_class = static_cast<SignTxClass>(tx);
            q.above_dc = d.sign_above + slot * 64;
            q.left_dc = d.sign_left + slot * 64;
            q.x4 = 1;
            q.y4 = 1;
            q.w4 = 4;
            q.h4 = 4;
            q.mi_cols = 64;
            q.mi_rows = 64;
            queries.push_back(q);
            if (queries.size() == kMaxSignJobs) {
              CHECK(cuda_ok(cudaMemcpy(
                  d.sign_above, above.data(),
                  sizeof(std::int8_t) * above.size(), cudaMemcpyHostToDevice)));
              CHECK(cuda_ok(cudaMemcpy(
                  d.sign_left, left.data(), sizeof(std::int8_t) * left.size(),
                  cudaMemcpyHostToDevice)));
              flush();
            }
          }
        }
      }
    }
    flush();
    std::printf("  dc_sign elements: cases=%d checks=%lld\n", cases,
                element_checks);
  }

  // 2. Context sweep with random geometry, including frame-edge limits.
  {
    constexpr int kBatches = 16;
    std::vector<SignContextJob> jobs(kMaxSignJobs);
    std::vector<std::int8_t> above(kMaxSignJobs * 64);
    std::vector<std::int8_t> left(kMaxSignJobs * 64);
    std::vector<int> got(kMaxSignJobs);
    int cases = 0;
    for (int batch = 0; batch < kBatches; ++batch) {
      for (int i = 0; i < kMaxSignJobs; ++i) {
        for (int k = 0; k < 64; ++k) {
          above[static_cast<std::size_t>(i * 64 + k)] =
              static_cast<std::int8_t>(random_range(0, 2));
          left[static_cast<std::size_t>(i * 64 + k)] =
              static_cast<std::int8_t>(random_range(0, 2));
        }
        SignContextJob job;
        job.above_dc = d.sign_above + i * 64;
        job.left_dc = d.sign_left + i * 64;
        job.w4 = random_range(1, 8);
        job.h4 = random_range(1, 8);
        job.x4 = random_range(0, 32);
        job.y4 = random_range(0, 32);
        job.mi_cols = random_range(1, job.x4 + job.w4 + 2);
        job.mi_rows = random_range(1, job.y4 + job.h4 + 2);
        jobs[static_cast<std::size_t>(i)] = job;
      }
      CHECK(cuda_ok(cudaMemcpy(d.sign_above, above.data(),
                               sizeof(std::int8_t) * above.size(),
                               cudaMemcpyHostToDevice)));
      CHECK(cuda_ok(cudaMemcpy(d.sign_left, left.data(),
                               sizeof(std::int8_t) * left.size(),
                               cudaMemcpyHostToDevice)));
      CHECK_EQ(static_cast<int>(workspace.dc_sign_contexts(
                   jobs.data(), kMaxSignJobs, got.data())),
               static_cast<int>(Status::kOk), "dc_sign context launch");
      for (int i = 0; i < kMaxSignJobs; ++i) {
        const SignContextJob& job = jobs[static_cast<std::size_t>(i)];
        const int want = mb::dc_sign_context(
            above.data() + static_cast<std::size_t>(i * 64),
            left.data() + static_cast<std::size_t>(i * 64), job.x4, job.y4,
            job.w4, job.h4, job.mi_cols, job.mi_rows);
        CHECK_EQ(got[static_cast<std::size_t>(i)], want, "dc_sign ctx sweep");
      }
      cases += kMaxSignJobs;
      context_checks += kMaxSignJobs;
    }

    // Null neighbour arrays are legal and contribute nothing.
    SignContextJob null_job;
    null_job.x4 = 0;
    null_job.y4 = 0;
    null_job.w4 = 4;
    null_job.h4 = 4;
    null_job.mi_cols = 64;
    null_job.mi_rows = 64;
    int ctx = -1;
    CHECK_EQ(static_cast<int>(workspace.dc_sign_contexts(&null_job, 1, &ctx)),
             static_cast<int>(Status::kOk), "dc_sign null ctx launch");
    CHECK_EQ(ctx, 0, "dc_sign null ctx");
    ++context_checks;

    std::printf("  dc_sign contexts: cases=%d checks=%lld\n", cases,
                context_checks);
  }

  CHECK_EQ(arena.used(), before, "dc_sign: arena flat after");
}

// ---------------------------------------------------------------------------
// Argument validation, arena negative test, mark/rewind growth
// ---------------------------------------------------------------------------

void test_arguments_and_arena(Workspace& workspace, Arena& arena,
                              DeviceData& d, const Config& config,
                              std::uint64_t required) {
  CHECK_EQ(static_cast<int>(workspace.dc_subsampled_predict(nullptr, 1)),
           static_cast<int>(Status::kInvalidArgument), "null dc jobs");
  CHECK_EQ(static_cast<int>(workspace.dc_subsampled_predict(
                 nullptr, workspace.config().max_dc_jobs + 1)),
           static_cast<int>(Status::kInvalidArgument), "dc count over max");
  DcJob bad_dc{};
  bad_dc.in.w = 65;
  bad_dc.in.h = 4;
  bad_dc.dst = d.dc_dst;
  bad_dc.dst_stride = 65;
  CHECK_EQ(static_cast<int>(workspace.dc_subsampled_predict(&bad_dc, 1)),
           static_cast<int>(Status::kInvalidArgument), "bad dc extent");

  CornerJob bad_corner{};
  bad_corner.above = d.corner_above;
  bad_corner.left = nullptr;
  bad_corner.w = 8;
  bad_corner.h = 8;
  CHECK_EQ(static_cast<int>(workspace.filter_corner(&bad_corner, 1)),
           static_cast<int>(Status::kInvalidArgument), "bad corner ptr");

  ChromaQuery bad_chroma{};
  bad_chroma.plane = 0;
  bad_chroma.coded_w = 8;
  bad_chroma.coded_h = 8;
  ChromaResult chroma_result{};
  CHECK_EQ(static_cast<int>(
               workspace.derive_chroma_type(&bad_chroma, 1, &chroma_result)),
           static_cast<int>(Status::kInvalidArgument), "bad chroma plane");

  const auto dct =
      av2enc::txv::make_tx_params(8, 8, av2enc::txv::TxType::kDctDct);
  CHECK(dct.has_value());
  TxJob bad_tx;
  bad_tx.params = *dct;
  CHECK_EQ(static_cast<int>(
               workspace.identity_forward(&bad_tx, 1, d.tx_residual, d.tx_coeff)),
           static_cast<int>(Status::kInvalidArgument),
           "non-identity type rejected");

  // Two back-to-back launches: flat used(), stable peak.
  std::vector<std::int16_t> small_above(64, 0);
  std::vector<std::int16_t> small_left(64, 0);
  CHECK(cuda_ok(cudaMemcpy(d.dc_above, small_above.data(),
                                       sizeof(std::int16_t) * small_above.size(),
                                       cudaMemcpyHostToDevice)));
  CHECK(cuda_ok(cudaMemcpy(d.dc_left, small_left.data(),
                                       sizeof(std::int16_t) * small_left.size(),
                                       cudaMemcpyHostToDevice)));
  DcJob job{};
  job.in.above = d.dc_above;
  job.in.left = d.dc_left;
  job.in.w = 8;
  job.in.h = 8;
  job.in.bit_depth = 8;
  job.in.have_above = 1;
  job.in.have_left = 1;
  job.dst = d.dc_dst;
  job.dst_stride = 8;
  const std::uint64_t mark = arena.mark();
  CHECK_EQ(static_cast<int>(workspace.dc_subsampled_predict(&job, 1)),
           static_cast<int>(Status::kOk), "growth launch 1");
  const std::uint64_t peak1 = arena.peak_used();
  CHECK_EQ(arena.used(), mark, "launch 1 leaves used() flat");
  CHECK_EQ(static_cast<int>(workspace.dc_subsampled_predict(&job, 1)),
           static_cast<int>(Status::kOk), "growth launch 2");
  CHECK_EQ(arena.used(), mark, "launch 2 leaves used() flat");
  CHECK_EQ(arena.peak_used(), peak1, "peak does not grow on launch 2");

  // One-byte-short arena: distinct status + required size, no residue.
  Arena small;
  CHECK(av2enc::cuda::is_ok(small.init(required - 1)));
  Workspace short_ws;
  std::string error;
  std::uint64_t needed = 0;
  CHECK_EQ(static_cast<int>(short_ws.init(small, config, &error, &needed)),
           static_cast<int>(Status::kArenaExhausted), "short arena status");
  CHECK_EQ(needed, required, "short arena required size");
  CHECK_EQ(small.used(), 0, "short arena untouched");
  CHECK(!short_ws.initialized());

  Config invalid = config;
  invalid.max_dc_jobs = 0;
  CHECK_EQ(av2enc::cuda::intra_full::required_device_bytes(invalid), 0,
           "invalid config bytes");
  Workspace invalid_ws;
  CHECK_EQ(static_cast<int>(invalid_ws.init(arena, invalid)),
           static_cast<int>(Status::kInvalidArgument), "invalid config status");

  std::printf(
      "  arena: required=%llu persistent=%llu scratch=%llu capacity=%llu "
      "peak=%llu\n",
      static_cast<unsigned long long>(required),
      static_cast<unsigned long long>(
          av2enc::cuda::intra_full::persistent_device_bytes(config)),
      static_cast<unsigned long long>(
          av2enc::cuda::intra_full::scratch_device_bytes(config)),
      static_cast<unsigned long long>(arena.capacity()),
      static_cast<unsigned long long>(arena.peak_used()));
}

}  // namespace

int main() {
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    std::printf("no CUDA device\n");
    return 1;
  }
  std::printf("av2enc CUDA intra-full test\n");
  std::printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);

  const Config config = make_config();
  const std::uint64_t required =
      av2enc::cuda::intra_full::required_device_bytes(config);
  const std::uint64_t persistent =
      av2enc::cuda::intra_full::persistent_device_bytes(config);
  const std::uint64_t scratch =
      av2enc::cuda::intra_full::scratch_device_bytes(config);
  CHECK_EQ(required, persistent + scratch, "required = persistent + scratch");
  CHECK_EQ(required % Arena::kAlignment, 0, "required aligned");

  // One arena for the workspace plus this test's device data.
  const std::uint64_t data_bytes =
      static_cast<std::uint64_t>(2) * 1024 * 1024 +
      static_cast<std::uint64_t>(kValueCases) + kGuardCases;
  Arena arena;
  CHECK(av2enc::cuda::is_ok(arena.init(required + data_bytes + 4096)));

  Workspace workspace;
  std::string error;
  std::uint64_t need = 0;
  CHECK_EQ(static_cast<int>(workspace.init(arena, config, &error, &need)),
           static_cast<int>(Status::kOk), "workspace init");
  CHECK_EQ(need, required, "workspace required");
  CHECK_EQ(arena.used(), persistent, "init growth == persistent bytes");

  DeviceData data;
  CHECK(alloc_all(arena, &data));

  test_dc(workspace, arena, data);
  test_corner(workspace, arena, data);
  test_transforms(workspace, arena, data);
  test_chroma(workspace, arena);
  test_dc_sign(workspace, arena, data);
  test_arguments_and_arena(workspace, arena, data, config, required);

  std::printf(
      "  direct device allocation: none (Arena only); reserved=%llu bytes\n",
      static_cast<unsigned long long>(arena.reserved_bytes()));
  std::printf("intra_full_test: %s (%lld checks, %d failures)\n",
              g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
