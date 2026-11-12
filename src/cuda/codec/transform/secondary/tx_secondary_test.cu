// Real-GPU parity test for the AV2 secondary transform (IST).
//
// Launches the one-thread-per-block kernel over every 4x4/8x8 kernel,
// sec_tx_type 1..3, transpose 0/1 and all 25 TX sizes, then compares the
// device result bit-for-bit with the CPU reference.  Also checks the forward
// analysis projection and the in-place path.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "cuda/codec/transform/secondary/tx_secondary_gpu.h"
#include "codec/transform/secondary/tx_secondary.h"
#include "codec/transform/variants/tx_variants.h"

namespace {

using namespace av2enc;
using namespace av2enc::txsec;
using av2enc::cuda::TxSecondaryJob;
using av2enc::txv::TxParams;
using av2enc::txv::TxType;

int g_failures = 0;
long long g_checks = 0;

void check(bool cond, const char* what, int line) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("FAIL [%d] %s\n", line, what);
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define CUDA_CHECK(expr)                                                  \
  do {                                                                    \
    const cudaError_t err = (expr);                                       \
    check(err == cudaSuccess, #expr " == cudaSuccess", __LINE__);         \
    if (err != cudaSuccess) {                                             \
      std::printf("  CUDA error: %s\n", cudaGetErrorString(err));         \
    }                                                                     \
  } while (0)

std::mt19937 g_rng(0x9e3779b9u);

int rand_range(int lo, int hi) {
  std::uniform_int_distribution<int> dist(lo, hi);
  return dist(g_rng);
}

std::vector<TxParams> all_sizes() {
  static const int kSizes[25][2] = {
      {4, 4},   {8, 8},   {16, 16}, {32, 32}, {64, 64}, {4, 8},
      {8, 4},   {8, 16},  {16, 8},  {16, 32}, {32, 16}, {32, 64},
      {64, 32}, {4, 16},  {16, 4},  {8, 32},  {32, 8},  {16, 64},
      {64, 16}, {4, 32},  {32, 4},  {8, 64},  {64, 8},  {4, 64},
      {64, 4},
  };
  std::vector<TxParams> out;
  for (const auto& s : kSizes) {
    auto p = txv::make_tx_params(s[0], s[1], TxType::kDctDct);
    if (p) out.push_back(*p);
  }
  return out;
}

struct Case {
  TxParams params;
  TxType type;
  int kernel;
  int sec;
  int transpose;
  int elems;
  int offset;
};

void build_jobs(std::vector<Case>* cases, std::vector<TxSecondaryJob>* jobs,
                std::vector<std::int32_t>* input,
                std::vector<std::int32_t>* expected_inverse,
                std::vector<std::int32_t>* forward_input,
                std::vector<std::int32_t>* expected_forward) {
  for (const TxParams& base : all_sizes()) {
    for (TxType type : {TxType::kDctDct, TxType::kAdstAdst}) {
      auto pv = txv::make_tx_params(base.coded_w, base.coded_h, type);
      if (!pv) continue;
      const TxParams& p = *pv;
      SecondarySizeInfo info;
      if (!secondary_size_info(p, type, &info)) continue;
      const int elems = info.w * info.h;
      for (int kernel = 0; kernel < info.kernel_count; ++kernel) {
        for (int sec = 1; sec < 4; ++sec) {
          for (int transpose = 0; transpose <= 1; ++transpose) {
            Case c;
            c.params = p;
            c.type = type;
            c.kernel = kernel;
            c.sec = sec;
            c.transpose = transpose;
            c.elems = elems;
            c.offset = static_cast<int>(input->size());
            cases->push_back(c);

            TxSecondaryJob job;
            job.tx_size = static_cast<int>(p.tx_size);
            job.plane_tx_type = static_cast<int>(type);
            job.kernel = kernel;
            job.sec_tx_type = sec;
            job.transpose = transpose;
            job.bit_depth = 8;
            job.in_offset = c.offset;
            job.out_offset = c.offset;
            jobs->push_back(job);

            // Legal first-n input, sentinel elsewhere.
            std::vector<std::int32_t> block(static_cast<std::size_t>(elems), 0);
            std::uint16_t scan[1024];
            int x = 0;
            int y = 0;
            for (int i = 0; i < elems; ++i) {
              scan[i] = static_cast<std::uint16_t>(y * info.w + x);
              ++x;
              --y;
              if (y < 0 || x >= info.w) {
                ++x;
                const int s = std::min(x, info.h - 1 - y);
                x -= s;
                y += s;
              }
            }
            for (int j = 0; j < info.n; ++j) {
              block[static_cast<std::size_t>(scan[j])] =
                  rand_range(-32768, 32767);
            }
            std::vector<std::int32_t> inv = block;
            CHECK(inverse_secondary_dequant_kernel(
                p, type, kernel, sec, transpose != 0, 8, inv.data(),
                inv.size()));
            std::vector<std::int32_t> fwd_in(
                static_cast<std::size_t>(elems));
            for (int i = 0; i < elems; ++i) {
              fwd_in[static_cast<std::size_t>(i)] = rand_range(-3000, 3000);
            }
            std::vector<std::int32_t> fwd = fwd_in;
            CHECK(forward_secondary_coeffs_kernel(
                p, type, kernel, sec, transpose != 0, 8, fwd.data(),
                fwd.size()));

            input->insert(input->end(), block.begin(), block.end());
            expected_inverse->insert(expected_inverse->end(), inv.begin(),
                                     inv.end());
            forward_input->insert(forward_input->end(), fwd_in.begin(),
                                  fwd_in.end());
            expected_forward->insert(expected_forward->end(), fwd.begin(),
                                     fwd.end());
          }
        }
      }
    }
  }
}

void run_inverse_test(const std::vector<Case>& cases,
                      const std::vector<TxSecondaryJob>& jobs,
                      const std::vector<std::int32_t>& input,
                      const std::vector<std::int32_t>& expected) {
  std::int32_t* d_in = nullptr;
  std::int32_t* d_out = nullptr;
  TxSecondaryJob* d_jobs = nullptr;
  const std::size_t bytes = input.size() * sizeof(std::int32_t);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_jobs), jobs.size() * sizeof(TxSecondaryJob)));
  CUDA_CHECK(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_jobs, jobs.data(),
                        jobs.size() * sizeof(TxSecondaryJob),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(av2enc::cuda::txsec_inverse_dequant(d_in, d_out, d_jobs,
                                                 static_cast<int>(jobs.size())));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::int32_t> got(input.size());
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  for (std::size_t c = 0; c < cases.size(); ++c) {
    const Case& cs = cases[c];
    for (int i = 0; i < cs.elems; ++i) {
      CHECK(got[static_cast<std::size_t>(cs.offset + i)] ==
            expected[static_cast<std::size_t>(cs.offset + i)]);
    }
  }
  CUDA_CHECK(cudaFree(d_jobs));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_in));
}

void run_forward_test(const std::vector<Case>& cases,
                      const std::vector<TxSecondaryJob>& jobs,
                      const std::vector<std::int32_t>& input,
                      const std::vector<std::int32_t>& expected) {
  std::int32_t* d_in = nullptr;
  std::int32_t* d_out = nullptr;
  TxSecondaryJob* d_jobs = nullptr;
  const std::size_t bytes = input.size() * sizeof(std::int32_t);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_jobs), jobs.size() * sizeof(TxSecondaryJob)));
  CUDA_CHECK(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_jobs, jobs.data(),
                        jobs.size() * sizeof(TxSecondaryJob),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(av2enc::cuda::txsec_forward_coeffs(d_in, d_out, d_jobs,
                                                static_cast<int>(jobs.size())));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::int32_t> got(input.size());
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  for (std::size_t c = 0; c < cases.size(); ++c) {
    const Case& cs = cases[c];
    for (int i = 0; i < cs.elems; ++i) {
      CHECK(got[static_cast<std::size_t>(cs.offset + i)] ==
            expected[static_cast<std::size_t>(cs.offset + i)]);
    }
  }
  CUDA_CHECK(cudaFree(d_jobs));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_in));
}

void run_inplace_test() {
  TxParams p = *txv::make_tx_params(16, 16, TxType::kDctDct);
  SecondarySizeInfo info;
  CHECK(secondary_size_info(p, TxType::kDctDct, &info));
  const int elems = info.w * info.h;
  std::vector<std::int32_t> host(static_cast<std::size_t>(elems));
  for (int i = 0; i < elems; ++i) {
    host[static_cast<std::size_t>(i)] = rand_range(-2000, 2000);
  }
  std::vector<std::int32_t> expected = host;
  CHECK(inverse_secondary_dequant_kernel(p, TxType::kDctDct, 2, 2, true, 8,
                                         expected.data(), expected.size()));
  TxSecondaryJob job;
  job.tx_size = static_cast<int>(p.tx_size);
  job.plane_tx_type = static_cast<int>(TxType::kDctDct);
  job.kernel = 2;
  job.sec_tx_type = 2;
  job.transpose = 1;
  job.bit_depth = 8;
  std::int32_t* d_buf = nullptr;
  TxSecondaryJob* d_job = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buf), host.size() * sizeof(std::int32_t)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_job), sizeof(TxSecondaryJob)));
  CUDA_CHECK(cudaMemcpy(d_buf, host.data(), host.size() * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_job, &job, sizeof(TxSecondaryJob),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      av2enc::cuda::txsec_inverse_dequant(d_buf, d_buf, d_job, 1));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::int32_t> got(host.size());
  CUDA_CHECK(cudaMemcpy(got.data(), d_buf, host.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < got.size(); ++i) CHECK(got[i] == expected[i]);
  CUDA_CHECK(cudaFree(d_job));
  CUDA_CHECK(cudaFree(d_buf));
}

}  // namespace

int main() {
  std::printf("av2enc AV2 secondary transform (IST) CUDA test\n");
  std::vector<Case> cases;
  std::vector<TxSecondaryJob> jobs;
  std::vector<std::int32_t> input;
  std::vector<std::int32_t> expected_inverse;
  std::vector<std::int32_t> forward_input;
  std::vector<std::int32_t> expected_forward;
  build_jobs(&cases, &jobs, &input, &expected_inverse, &forward_input,
             &expected_forward);
  std::printf("  cases: %zu, coefficients: %zu\n", cases.size(),
              input.size());
  run_inverse_test(cases, jobs, input, expected_inverse);
  run_forward_test(cases, jobs, forward_input, expected_forward);
  run_inplace_test();

  // Argument validation.
  CHECK(av2enc::cuda::txsec_inverse_dequant(nullptr, nullptr, nullptr, 0) ==
        cudaErrorInvalidValue);
  CHECK(av2enc::cuda::txsec_inverse_dequant(nullptr, nullptr, nullptr, -1) ==
        cudaErrorInvalidValue);

  std::printf("%lld checks, %d failures\n", g_checks, g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
