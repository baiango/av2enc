// Real-GPU bit-exact parity test for the §7.14.4 DC/AC delta-Q quantizer.
//
//   ./build_gpu_test.sh          # sm_86
//
// Every case's expected output comes from the CPU reference
// (`dcdeltaq::quantize_block` / `dequantize_block`).  All device buffers come
// from one `av2enc::cuda::Arena`; the test contains no cudaMalloc.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/codec/transform/delta_q/deltaq_gpu.h"
#include "cuda/codec/transform/delta_q/deltaq_test_common.h"

namespace dq = av2enc::cuda::deltaq;
namespace test = av2enc::cuda::deltaq::test;

namespace {

int g_checks = 0;
int g_failures = 0;

void check_eq(long long actual, long long expected, const char* what,
              int index) {
  ++g_checks;
  if (actual != expected) {
    ++g_failures;
    if (g_failures <= 20) {
      std::fprintf(stderr, "FAIL: %s (case %d): got %lld expected %lld\n",
                   what, index, actual, expected);
    }
  }
}

#define CUDA_CHECK(expr)                                                   \
  do {                                                                     \
    const cudaError_t err__ = (expr);                                      \
    if (err__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                   cudaGetErrorString(err__), __FILE__, __LINE__);         \
      std::exit(2);                                                        \
    }                                                                      \
  } while (0)

struct Packed {
  std::vector<dq::Job> jobs;
  std::vector<double> coeff;
  std::vector<std::int32_t> out_template;  // sentinel values
  std::vector<std::int32_t> q2;
  std::vector<std::int32_t> expected_levels;
  std::vector<std::int32_t> expected_dequant;
};

Packed pack(const std::vector<test::Case>& cases) {
  Packed p;
  p.jobs.reserve(cases.size());
  for (const test::Case& c : cases) {
    dq::Job job = c.job;
    job.in_offset = static_cast<std::int32_t>(p.coeff.size());
    job.out_offset = static_cast<std::int32_t>(p.out_template.size());
    if (!c.q2.empty()) {
      job.q2_offset = static_cast<std::int32_t>(p.q2.size());
      p.q2.insert(p.q2.end(), c.q2.begin(), c.q2.end());
    } else {
      job.q2_offset = -1;
    }
    p.jobs.push_back(job);
    p.coeff.insert(p.coeff.end(), c.coeff.begin(), c.coeff.end());
    if (c.expect_reject) {
      p.out_template.insert(p.out_template.end(), c.expected_levels.size(),
                            -12345);
    } else {
      p.out_template.insert(p.out_template.end(), c.expected_levels.size(), 0);
    }
    p.expected_levels.insert(p.expected_levels.end(),
                             c.expected_levels.begin(),
                             c.expected_levels.end());
    p.expected_dequant.insert(p.expected_dequant.end(),
                              c.expected_dequant.begin(),
                              c.expected_dequant.end());
  }
  return p;
}

template <typename T>
T* arena_array(av2enc::cuda::Arena& arena, std::size_t count,
               const char* what) {
  void* raw = nullptr;
  const av2enc::cuda::Status status =
      arena.allocate_checked(count * sizeof(T), &raw);
  if (status != av2enc::cuda::Status::kOk) {
    std::fprintf(stderr, "arena allocation failed for %s\n", what);
    std::exit(2);
  }
  return static_cast<T*>(raw);
}

}  // namespace

int main() {
  std::vector<test::Case> cases = test::make_cases();
  Packed packed = pack(cases);
  const int njobs = static_cast<int>(packed.jobs.size());

  av2enc::cuda::Arena arena;
  std::string error;
  const av2enc::cuda::Status init =
      arena.init(16ull * 1024ull * 1024ull, av2enc::cuda::Arena::kAlignment,
                 &error);
  if (init != av2enc::cuda::Status::kOk) {
    std::fprintf(stderr, "arena init failed: %s\n", error.c_str());
    return 2;
  }

  dq::Job* d_jobs = arena_array<dq::Job>(arena, packed.jobs.size(), "jobs");
  double* d_coeff = arena_array<double>(arena, packed.coeff.size(), "coeff");
  std::int32_t* d_levels =
      arena_array<std::int32_t>(arena, packed.out_template.size(), "levels");
  std::int32_t* d_dequant =
      arena_array<std::int32_t>(arena, packed.out_template.size(), "dequant");
  std::int32_t* d_q2 =
      arena_array<std::int32_t>(arena, packed.q2.size(), "q2");
  std::int32_t* d_status = arena_array<std::int32_t>(arena, njobs, "status");

  CUDA_CHECK(cudaMemcpy(d_jobs, packed.jobs.data(),
                        packed.jobs.size() * sizeof(dq::Job),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_coeff, packed.coeff.data(),
                        packed.coeff.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  if (!packed.q2.empty()) {
    CUDA_CHECK(cudaMemcpy(d_q2, packed.q2.data(),
                          packed.q2.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d_levels, packed.out_template.data(),
                        packed.out_template.size() * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice));

  dq::Batch batch;
  batch.coeff = d_coeff;
  batch.levels = d_levels;
  batch.out = d_levels;
  batch.jobs = d_jobs;
  batch.num_jobs = njobs;
  batch.q2 = packed.q2.empty() ? nullptr : d_q2;
  batch.status = d_status;

  // ---- forward quantization ------------------------------------------------
  CUDA_CHECK(dq::quantize_batch(batch, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::int32_t> levels(packed.out_template.size(), 0);
  CUDA_CHECK(cudaMemcpy(levels.data(), d_levels,
                        levels.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost));
  std::vector<std::int32_t> status(static_cast<std::size_t>(njobs), 0);
  CUDA_CHECK(cudaMemcpy(status.data(), d_status,
                        status.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost));

  int compared = 0;
  int rejected = 0;
  int lossless = 0;
  int qm = 0;
  int idtx = 0;
  int tcq_shift = 0;
  for (int i = 0; i < njobs; ++i) {
    const test::Case& c = cases[static_cast<std::size_t>(i)];
    if (c.expect_reject) {
      check_eq(status[static_cast<std::size_t>(i)], -1, "reject status", i);
      ++rejected;
      for (std::size_t k = 0; k < c.expected_levels.size(); ++k) {
        check_eq(levels[static_cast<std::size_t>(packed.jobs[i].out_offset) + k],
                 -12345, "rejected output untouched", i);
      }
      continue;
    }
    check_eq(status[static_cast<std::size_t>(i)], 0, "ok status", i);
    if (c.job.lossless != 0) ++lossless;
    if (c.job.q2_offset >= 0) ++qm;
    if (c.job.tx_type == 9) ++idtx;
    if (c.job.allow_tcq != 0 && c.job.plane == 0 && c.job.lossless == 0 &&
        c.job.use_fsc == 0 && c.job.tx_type < 10) {
      ++tcq_shift;
    }
    for (std::size_t k = 0; k < c.expected_levels.size(); ++k) {
      check_eq(levels[static_cast<std::size_t>(packed.jobs[i].out_offset) + k],
               c.expected_levels[k], "level", i);
    }
    ++compared;
  }

  // ---- inverse quantization ------------------------------------------------
  batch.out = d_dequant;
  CUDA_CHECK(dq::dequantize_batch(batch, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::int32_t> dequant(packed.expected_dequant.size(), 0);
  CUDA_CHECK(cudaMemcpy(dequant.data(), d_dequant,
                        dequant.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < njobs; ++i) {
    const test::Case& c = cases[static_cast<std::size_t>(i)];
    if (c.expect_reject) continue;
    for (std::size_t k = 0; k < c.expected_dequant.size(); ++k) {
      check_eq(dequant[static_cast<std::size_t>(packed.jobs[i].out_offset) + k],
               c.expected_dequant[k], "dequant", i);
    }
  }

  // ---- in-place dequantize (levels alias output) ---------------------------
  batch.out = d_levels;
  CUDA_CHECK(dq::dequantize_batch(batch, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::int32_t> in_place(packed.expected_dequant.size(), 0);
  CUDA_CHECK(cudaMemcpy(in_place.data(), d_levels,
                        in_place.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < njobs; ++i) {
    const test::Case& c = cases[static_cast<std::size_t>(i)];
    if (c.expect_reject) continue;
    for (std::size_t k = 0; k < c.expected_dequant.size(); ++k) {
      check_eq(in_place[static_cast<std::size_t>(packed.jobs[i].out_offset) + k],
               c.expected_dequant[k], "in-place dequant", i);
    }
  }

  // ---- determinism: a second forward launch --------------------------------
  CUDA_CHECK(cudaMemcpy(d_levels, packed.out_template.data(),
                        packed.out_template.size() * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice));
  batch.out = d_levels;
  CUDA_CHECK(dq::quantize_batch(batch, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::int32_t> levels2(packed.out_template.size(), 0);
  CUDA_CHECK(cudaMemcpy(levels2.data(), d_levels,
                        levels2.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost));
  for (std::size_t k = 0; k < levels2.size(); ++k) {
    check_eq(levels2[k], levels[k], "deterministic level", static_cast<int>(k));
  }

  std::printf(
      "delta-q: %d blocks compared (%d rejected, %d lossless, %d QM, %d IDTX, "
      "%d TCQ-shift)\n",
      compared, rejected, lossless, qm, idtx, tcq_shift);
  std::printf("delta-q: %d checks, %d failures\n", g_checks, g_failures);
  if (g_failures != 0) {
    std::printf("RESULT: FAIL\n");
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
