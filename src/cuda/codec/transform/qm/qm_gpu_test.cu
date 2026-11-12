// CPU vs CUDA bit-exact test for AV2 quantizer-matrix dequantization.
//
//   ./build_and_test.sh
//   CUDA_ARCH=sm_86 ./build_and_test.sh
//
// The CPU reference is linked from src/encoder/qm/qm.cpp.  Every job is run
// through both paths and every dequantized coefficient is compared.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "cuda/codec/transform/qm/qm_gpu.h"
#include "codec/transform/qm/qm.h"
#include "codec/transform/qm/qm_tables.h"

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(condition)) {                                                      \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,           \
                   #condition);                                              \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    const cudaError_t status = (expr);                                       \
    ++g_checks;                                                              \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s: %s\n", __FILE__, __LINE__,       \
                   #expr, cudaGetErrorString(status));                       \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

using av2enc::cuda::qm::Batch;
using av2enc::cuda::qm::Job;
using av2enc::qm::QmDequantParams;
using av2enc::qm::QmFrameState;

int adjusted_size(int tx_size) {
  return av2enc::qm::adjusted_width(tx_size) *
         av2enc::qm::adjusted_height(tx_size);
}

void fill_user_qms(QmFrameState* state, std::mt19937* rng) {
  std::uniform_int_distribution<int> dist(1, 255);
  for (int level = 0; level < av2enc::qm::kNumQmLevels; ++level) {
    av2enc::qm::fill_user_qm_constant(&state->user_qm[level], 32);
    for (int t = 0; t < 3; ++t) {
      for (int plane = 0; plane < 3; ++plane) {
        for (int i = 0; i < 8; ++i) {
          for (int j = 0; j < 8; ++j) {
            state->user_qm[level].m[t][plane][i][j] =
                static_cast<std::uint8_t>(dist(*rng));
          }
        }
      }
    }
    state->qm_data_present[level] = true;
  }
}

struct Case {
  QmFrameState state;
  std::vector<Job> jobs;
  std::vector<std::int32_t> levels;
  std::vector<std::int32_t> dequant_cpu;
  std::vector<std::int32_t> dequant_gpu;
  std::vector<std::int32_t> status;
};

void add_job(Case* test, int tx_size, int plane, int tx_type, int seg_level,
             int qm_level, bool using_qmatrix, bool data_present,
             bool allow_tcq, bool lossless, int dc_quant, int ac_quant,
             int bit_depth, std::mt19937* rng) {
  Job job;
  job.level_offset = static_cast<std::int32_t>(test->levels.size());
  job.dequant_offset = static_cast<std::int32_t>(test->dequant_cpu.size());
  job.tx_size = tx_size;
  job.plane = plane;
  job.tx_type = tx_type;
  job.seg_qm_level = seg_level;
  job.qm_level = qm_level;
  job.dc_quant = dc_quant;
  job.ac_quant = ac_quant;
  job.bit_depth = bit_depth;
  job.using_qmatrix = using_qmatrix ? 1 : 0;
  job.qm_data_present = data_present ? 1 : 0;
  job.allow_tcq = allow_tcq ? 1 : 0;
  job.lossless = lossless ? 1 : 0;
  job.use_fsc = 0;
  test->jobs.push_back(job);

  const int n = adjusted_size(tx_size);
  std::uniform_int_distribution<int> dist(-2000, 2000);
  for (int i = 0; i < n; ++i) {
    int value = dist(*rng);
    if ((i % 11) == 0) value = 0;
    test->levels.push_back(value);
  }
  test->dequant_cpu.resize(test->dequant_cpu.size() +
                           static_cast<std::size_t>(n));
  test->dequant_gpu.resize(test->dequant_gpu.size() +
                           static_cast<std::size_t>(n));
  test->status.push_back(0);
}

void run_cpu_reference(Case* test) {
  for (std::size_t i = 0; i < test->jobs.size(); ++i) {
    const Job& job = test->jobs[i];
    QmFrameState& state = test->state;
    state.using_qmatrix = job.using_qmatrix != 0;
    state.qm_y[0] = static_cast<std::uint8_t>(job.qm_level);
    state.qm_u[0] = static_cast<std::uint8_t>(job.qm_level);
    state.qm_v[0] = static_cast<std::uint8_t>(job.qm_level);
    state.seg_qm_level[job.plane][0] =
        static_cast<std::uint8_t>(job.seg_qm_level);
    if (job.seg_qm_level >= 0 && job.seg_qm_level < av2enc::qm::kNumQmLevels) {
      state.qm_data_present[job.seg_qm_level] = job.qm_data_present != 0;
    }

    QmDequantParams params;
    params.state = &state;
    params.tx_size = job.tx_size;
    params.plane = job.plane;
    params.tx_type = job.tx_type;
    params.segment_id = 0;
    params.dc_quant = job.dc_quant;
    params.ac_quant = job.ac_quant;
    params.bit_depth = job.bit_depth;
    params.allow_tcq = job.allow_tcq != 0;
    params.lossless = job.lossless != 0;
    params.use_fsc = job.use_fsc != 0;
    const bool ok = av2enc::qm::dequantize_block(
        test->levels.data() + job.level_offset, params,
        test->dequant_cpu.data() + job.dequant_offset);
    test->status[i] = ok ? 0 : -1;
  }
}

void run_gpu(Case* test) {
  std::int32_t* d_levels = nullptr;
  std::int32_t* d_dequant = nullptr;
  Job* d_jobs = nullptr;
  std::int32_t* d_status = nullptr;
  std::uint8_t* d_qtable = nullptr;
  std::uint8_t* d_user_qm = nullptr;

  CUDA_CHECK(cudaMalloc(&d_levels,
                        test->levels.size() * sizeof(std::int32_t)));
  CUDA_CHECK(cudaMalloc(&d_dequant,
                        test->dequant_gpu.size() * sizeof(std::int32_t)));
  CUDA_CHECK(cudaMalloc(&d_jobs, test->jobs.size() * sizeof(Job)));
  CUDA_CHECK(cudaMalloc(&d_status,
                        test->status.size() * sizeof(std::int32_t)));
  CUDA_CHECK(cudaMalloc(&d_qtable, av2enc::cuda::qm::kQuantizerMatrixBytes));
  CUDA_CHECK(cudaMalloc(&d_user_qm, av2enc::cuda::qm::kUserQmBytes));

  CUDA_CHECK(cudaMemcpy(d_levels, test->levels.data(),
                        test->levels.size() * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_jobs, test->jobs.data(),
                        test->jobs.size() * sizeof(Job),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_qtable, av2enc::qm::kQuantizerMatrix,
                        av2enc::cuda::qm::kQuantizerMatrixBytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_user_qm, &test->state.user_qm[0].m[0][0][0][0],
                        av2enc::cuda::qm::kUserQmBytes,
                        cudaMemcpyHostToDevice));

  Batch batch;
  batch.levels = d_levels;
  batch.dequant = d_dequant;
  batch.jobs = d_jobs;
  batch.num_jobs = static_cast<std::int32_t>(test->jobs.size());
  batch.qtable = d_qtable;
  batch.user_qm = d_user_qm;
  batch.status = d_status;
  CUDA_CHECK(av2enc::cuda::qm::dequantize_batch(batch));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(test->dequant_gpu.data(), d_dequant,
                        test->dequant_gpu.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(test->status.data(), d_status,
                        test->status.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_levels));
  CUDA_CHECK(cudaFree(d_dequant));
  CUDA_CHECK(cudaFree(d_jobs));
  CUDA_CHECK(cudaFree(d_status));
  CUDA_CHECK(cudaFree(d_qtable));
  CUDA_CHECK(cudaFree(d_user_qm));
}

void test_parity() {
  static_assert(sizeof(av2enc::qm::UserQmLevel) == 576,
                "flattened user QM stride must be 576");
  std::mt19937 rng(0x5eed1234);
  Case test;
  fill_user_qms(&test.state, &rng);

  // Coverage: every TX size, every level 0..15, all three planes, 2-D and
  // non-2-D TX types, QM on/off, user data on/off, TCQ, lossless.
  const int quant_values[4] = {64, 132, 256, 528};
  for (int tx = 0; tx < 25; ++tx) {
    for (int level = 0; level <= 15; ++level) {
      const int plane = (tx + level) % 3;
      const int tx_type = (tx + level) % 16;
      const int q = quant_values[(tx + level) % 4];
      add_job(&test, tx, plane, tx_type, level, level % 16,
              (level % 3) != 0, (level % 2) == 0, (level % 5) == 0,
              level == 15, q, q + 1, (tx + level) % 2 == 0 ? 8 : 10, &rng);
    }
  }
  // Extra randomized jobs, including 8x8-and-smaller user-QM paths.
  for (int i = 0; i < 400; ++i) {
    const int tx = static_cast<int>(rng() % 25);
    const int level = static_cast<int>(rng() % 16);
    const int plane = static_cast<int>(rng() % 3);
    const int tx_type = static_cast<int>(rng() % 16);
    const int q = quant_values[rng() % 4];
    add_job(&test, tx, plane, tx_type, level, level,
            (i % 4) != 0, (i % 3) != 0, (i % 2) == 0, false, q, q + 7,
            (i % 2) == 0 ? 8 : 10, &rng);
  }

  run_cpu_reference(&test);
  run_gpu(&test);

  CHECK(test.jobs.size() == test.status.size());
  for (std::size_t i = 0; i < test.jobs.size(); ++i) {
    CHECK(test.status[i] == 0);
  }
  for (std::size_t i = 0; i < test.dequant_cpu.size(); ++i) {
    if (test.dequant_cpu[i] != test.dequant_gpu[i]) {
      std::fprintf(stderr,
                   "mismatch at coefficient %zu: cpu=%d gpu=%d\n", i,
                   test.dequant_cpu[i], test.dequant_gpu[i]);
      ++g_failures;
      break;
    }
    ++g_checks;
  }
  std::printf("qm gpu parity: %zu jobs, %zu coefficients\n",
              test.jobs.size(), test.dequant_cpu.size());
}

void test_invalid_jobs() {
  std::mt19937 rng(7);
  Case test;
  fill_user_qms(&test.state, &rng);
  add_job(&test, 25, 0, 0, 0, 0, true, true, false, false, 64, 132, 8,
          &rng);
  add_job(&test, 0, 3, 0, 0, 0, true, true, false, false, 64, 132, 8,
          &rng);
  add_job(&test, 0, 0, 0, 0, 0, true, true, false, false, 0, 132, 8,
          &rng);
  add_job(&test, 0, 0, 0, 0, 0, true, true, false, false, 64, 132, 13,
          &rng);
  run_cpu_reference(&test);
  run_gpu(&test);
  for (std::size_t i = 0; i < test.status.size(); ++i) {
    CHECK(test.status[i] == -1);
  }
}

}  // namespace

int main() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status != cudaSuccess || devices == 0) {
    std::fprintf(stderr, "qm_gpu_test: no CUDA device (%s)\n",
                 cudaGetErrorString(status));
    return 77;
  }
  test_parity();
  test_invalid_jobs();
  std::printf("qm_gpu_test: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
