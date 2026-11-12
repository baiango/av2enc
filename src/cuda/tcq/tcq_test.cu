// Real-GPU parity test for src/cuda/tcq.
//
// Build/run (needs nvcc and a visible CUDA device):
//   ./build_gpu_test.sh
//   ./build_gpu_test.sh --arch=sm_120
//
// What it proves:
//   * the kernels' levels, quant levels, states, EOB, distortion, rate and cost
//     are bit-identical to tcq::quantize() for randomized 8/10-bit blocks of
//     every TX size and scan class, with two rate models;
//   * all seven CCTX candidates match cctx_enumerate_scalar() at 8/10-bit;
//   * parity-hiding minimum-distortion levels match parity::quantize() for
//     every eligibility exclusion, exhaustive and pruned search;
//   * the arena contract: required_device_bytes() is exact, a one-byte-short
//     arena returns kArenaExhausted with that size and leaks nothing, and two
//     launches do not grow the arena or change its peak.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/tcq/tcq.h"
#include "cuda/tcq/tcq_host.h"
#include "encoder/tcq/cctx.h"
#include "encoder/tcq/parity/parity_hiding.h"
#include "encoder/tcq/tcq.h"

namespace ctcq = av2enc::cuda::tcq;
namespace tcq = av2enc::tcq;
namespace parity = av2enc::tcq::parity;

namespace {

int g_failures = 0;
long long g_checks = 0;

void expect(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL [%d] %s\n", line, what);
  }
}

#define EXPECT(condition) expect((condition), #condition, __LINE__)

#define CUDA_OK(expr)                                                      \
  do {                                                                     \
    const cudaError_t error_ = (expr);                                     \
    if (error_ != cudaSuccess) {                                           \
      std::printf("CUDA error %s at %s:%d (%s)\n",                         \
                  av2enc::cuda::cuda_error_name(error_), __FILE__,         \
                  __LINE__, #expr);                                        \
      return;                                                              \
    }                                                                      \
  } while (0)

std::mt19937 g_rng(0x51c7a9u);

double uniform(double lo, double hi) {
  std::uniform_real_distribution<double> dist(lo, hi);
  return dist(g_rng);
}

double nonmonotone_coefficient_rate(void*, std::int32_t scan_pos,
                                    std::int32_t state,
                                    std::int32_t base_level, std::int32_t,
                                    std::int32_t, std::int32_t,
                                    std::int32_t is_eob) {
  const std::int32_t magnitude = base_level < 0 ? -base_level : base_level;
  double bits = magnitude == 0 ? 0.65 : 1.1 + 0.31 * magnitude;
  bits += 0.07 * ((state * 7 + scan_pos * 3) % 5);
  if (is_eob != 0) bits += 0.37;
  return bits;
}

double nonmonotone_eob_rate(void*, std::int32_t eob, std::int32_t) {
  return eob == 0 ? 1.9 : 0.8 + 0.17 * eob;
}

tcq::RateModel test_rate_model(int which) {
  if (which == 1) {
    tcq::RateModel model;
    model.context = nullptr;
    model.coefficient_rate = &nonmonotone_coefficient_rate;
    model.eob_rate = &nonmonotone_eob_rate;
    return model;
  }
  return tcq::default_rate_model();
}

struct SizeCase {
  int tx_size;
  int tx_class;
};

std::vector<SizeCase> size_cases() {
  std::vector<SizeCase> out;
  for (int tx_size = 0; tx_size < 25; ++tx_size) {
    for (int tx_class = 0; tx_class < 3; ++tx_class) {
      out.push_back({tx_size, tx_class});
    }
  }
  return out;
}

// Host-side staging shared by the TCQ batches.
struct Staging {
  explicit Staging(const ctcq::TcqConfig& config) {
    const std::size_t jobs = static_cast<std::size_t>(config.max_jobs);
    const std::size_t coeff =
        static_cast<std::size_t>(config.max_coefficients);
    const std::size_t levels = static_cast<std::size_t>(config.max_levels);
    normal.assign(jobs * coeff * 8 * levels, 0.0);
    eob_coeff.assign(jobs * coeff * 8 * levels, 0.0);
    block_eob.assign(jobs * (coeff + 1), 0.0);
    original.assign(jobs * coeff, 0.0);
    pre_max.assign(jobs * coeff, 0);
    level_context.assign(jobs * coeff, 0);
    u_coeff.assign(jobs * coeff, 0.0);
    v_coeff.assign(jobs * coeff, 0.0);
    scan.assign(jobs * coeff, 0);
    seed.assign(jobs * coeff, 0);
    base_levels.assign(jobs * coeff, 0);
    quant_levels.assign(jobs * coeff, 0);
    states.assign(jobs * coeff, 0);
    parity_levels.assign(jobs * coeff, 0);
    cctx_u_levels.assign(jobs * 7 * coeff, 0);
    cctx_v_levels.assign(jobs * 7 * coeff, 0);
    eob.assign(jobs, -1);
    max_level.assign(jobs, 0);
    truncated.assign(jobs, 0);
    distortion.assign(jobs, 0.0);
    rate_bits.assign(jobs, 0.0);
    cost.assign(jobs, 0.0);
    status.assign(jobs, -1);
    cctx_type.assign(jobs * 7, -1);
    cctx_distortion.assign(jobs * 7, 0.0);
    cctx_eob_u.assign(jobs * 7, 0);
    cctx_eob_v.assign(jobs * 7, 0);
    cctx_best_type.assign(jobs, -1);
    cctx_best_distortion.assign(jobs, 0.0);
    par_eob.assign(jobs, 0);
    par_hidden.assign(jobs, 0);
    par_hidable.assign(jobs, 0);
    par_considered.assign(jobs, 0);
    par_turned_off.assign(jobs, 0);
    par_adjusted_index.assign(jobs, -1);
    par_adjusted_delta.assign(jobs, 0);
    par_distortion.assign(jobs, 0.0);
    par_status.assign(jobs, -1);
  }

  std::vector<double> normal;
  std::vector<double> eob_coeff;
  std::vector<double> block_eob;
  std::vector<double> original;
  std::vector<std::int32_t> pre_max;
  std::vector<std::int32_t> level_context;
  std::vector<double> u_coeff;
  std::vector<double> v_coeff;
  std::vector<std::int32_t> scan;
  std::vector<std::int32_t> seed;
  std::vector<std::int32_t> base_levels;
  std::vector<std::int32_t> quant_levels;
  std::vector<std::int32_t> states;
  std::vector<std::int32_t> parity_levels;
  std::vector<std::int32_t> cctx_u_levels;
  std::vector<std::int32_t> cctx_v_levels;
  std::vector<std::int32_t> eob;
  std::vector<std::int32_t> max_level;
  std::vector<std::uint8_t> truncated;
  std::vector<double> distortion;
  std::vector<double> rate_bits;
  std::vector<double> cost;
  std::vector<std::int32_t> status;
  std::vector<std::int32_t> cctx_type;
  std::vector<double> cctx_distortion;
  std::vector<std::int32_t> cctx_eob_u;
  std::vector<std::int32_t> cctx_eob_v;
  std::vector<std::int32_t> cctx_best_type;
  std::vector<double> cctx_best_distortion;
  std::vector<std::int32_t> par_eob;
  std::vector<std::int32_t> par_hidden;
  std::vector<std::int32_t> par_hidable;
  std::vector<std::int32_t> par_considered;
  std::vector<std::int32_t> par_turned_off;
  std::vector<std::int32_t> par_adjusted_index;
  std::vector<std::int32_t> par_adjusted_delta;
  std::vector<double> par_distortion;
  std::vector<std::int32_t> par_status;
};

av2enc::cuda::Arena g_arena;
ctcq::TcqWorkspace g_workspace;
ctcq::TcqConfig g_config;
Staging* g_staging = nullptr;

// ---------------------------------------------------------------------------
// TCQ
// ---------------------------------------------------------------------------

struct TcqCase {
  ctcq::TcqJob job;
  int count = 0;
  int slot = 0;
  int table_levels = 0;
  tcq::Result cpu;
  std::vector<std::int32_t> pre_max;
  std::vector<double> original;
};

bool build_tcq_case(const SizeCase& size, int plane, int bitdepth, int trial,
                    int slot, TcqCase* out) {
  out->slot = slot;
  out->count = 0;
  const int capacity = g_config.max_coefficients;
  std::vector<std::int32_t> scan(static_cast<std::size_t>(capacity), 0);
  std::vector<std::int32_t> pre_max(static_cast<std::size_t>(capacity), 0);
  int count = 0;
  int w = 0;
  int h = 0;
  if (!ctcq::tcq_scan_and_pre_max(size.tx_size, size.tx_class, plane,
                                  capacity, &count, &w, &h, scan.data(),
                                  pre_max.data(), nullptr)) {
    return false;
  }
  out->count = count;
  const double scale = bitdepth == 8 ? 32.0 : 128.0;
  const double step = (bitdepth == 8 ? 1.0 : 4.0) * uniform(0.75, 4.0);
  static const std::int32_t kCaps[4] = {5, 11, 23, 40};
  const std::int32_t cap = kCaps[trial & 3];
  const double lambda = (trial % 5 == 0) ? 0.0 : uniform(0.0, 4.0);
  const bool allow_zero = (trial & 1) != 0;
  std::vector<double> original(static_cast<std::size_t>(count), 0.0);
  std::vector<tcq::Coefficient> coefficients(static_cast<std::size_t>(count));
  std::vector<std::int32_t> context(static_cast<std::size_t>(count), 0);
  tcq::Config cpu_config;
  cpu_config.lambda = lambda;
  cpu_config.quant_step = step;
  cpu_config.level_cap = cap;
  cpu_config.max_coefficients = count;
  cpu_config.allow_zero_to_one = allow_zero;
  std::int32_t max_bound = 0;
  for (int c = 0; c < count; ++c) {
    const double value = uniform(-scale, scale);
    original[static_cast<std::size_t>(c)] = value;
    coefficients[static_cast<std::size_t>(c)].original = value;
    coefficients[static_cast<std::size_t>(c)].pre_max =
        pre_max[static_cast<std::size_t>(c)];
    coefficients[static_cast<std::size_t>(c)].level_context = c & 3;
    context[static_cast<std::size_t>(c)] = c & 3;
    const std::int32_t bound =
        tcq::level_bound(coefficients[static_cast<std::size_t>(c)],
                         cpu_config);
    if (bound < 0) return false;
    max_bound = std::max(max_bound, bound);
  }
  const std::int32_t table_levels = max_bound + 1;
  if (table_levels > g_config.max_levels) return false;
  out->table_levels = table_levels;
  const tcq::RateModel model = test_rate_model((trial ^ size.tx_size) & 1);
  out->job = ctcq::tcq_make_job(g_config, slot, count, table_levels, step,
                                lambda, cap, allow_zero);
  std::string error;
  if (!ctcq::tcq_fill_rate_tables(out->job, original.data(), pre_max.data(),
                                  context.data(), model,
                                  g_staging->normal.data() +
                                      out->job.rate_offset,
                                  g_staging->eob_coeff.data() +
                                      out->job.rate_offset,
                                  g_staging->block_eob.data() +
                                      out->job.eob_table_offset,
                                  &error)) {
    std::printf("fill_rate_tables: %s\n", error.c_str());
    return false;
  }
  if (!tcq::quantize(coefficients, cpu_config, model, out->cpu, &error)) {
    return false;
  }
  std::memcpy(g_staging->original.data() + out->job.coefficient_offset,
              original.data(), sizeof(double) * static_cast<std::size_t>(count));
  std::memcpy(g_staging->pre_max.data() + out->job.coefficient_offset,
              pre_max.data(),
              sizeof(std::int32_t) * static_cast<std::size_t>(count));
  std::memcpy(g_staging->level_context.data() + out->job.coefficient_offset,
              context.data(),
              sizeof(std::int32_t) * static_cast<std::size_t>(count));
  out->pre_max.assign(pre_max.begin(), pre_max.begin() + count);
  out->original = original;
  return true;
}

void launch_tcq_batch(std::vector<TcqCase>& cases) {
  const int count = static_cast<int>(cases.size());
  std::vector<ctcq::TcqJob> jobs(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    jobs[static_cast<std::size_t>(i)] = cases[static_cast<std::size_t>(i)].job;
    const ctcq::TcqJob& job = jobs[static_cast<std::size_t>(i)];
    const int n = cases[static_cast<std::size_t>(i)].count;
    const std::size_t table_bytes =
        sizeof(double) * static_cast<std::size_t>(n) * 8 *
        static_cast<std::size_t>(job.table_levels);
    CUDA_OK(cudaMemcpy(g_workspace.original + job.coefficient_offset,
                       g_staging->original.data() + job.coefficient_offset,
                       sizeof(double) * static_cast<std::size_t>(n),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_workspace.pre_max + job.coefficient_offset,
                       g_staging->pre_max.data() + job.coefficient_offset,
                       sizeof(std::int32_t) * static_cast<std::size_t>(n),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_workspace.level_context + job.coefficient_offset,
                       g_staging->level_context.data() +
                           job.coefficient_offset,
                       sizeof(std::int32_t) * static_cast<std::size_t>(n),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_workspace.normal_rate + job.rate_offset,
                       g_staging->normal.data() + job.rate_offset,
                       table_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_workspace.eob_coeff_rate + job.rate_offset,
                       g_staging->eob_coeff.data() + job.rate_offset,
                       table_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(
        g_workspace.block_eob_rate + job.eob_table_offset,
        g_staging->block_eob.data() + job.eob_table_offset,
        sizeof(double) * static_cast<std::size_t>(n + 1),
        cudaMemcpyHostToDevice));
  }
  CUDA_OK(ctcq::tcq_quantize_batch(g_workspace, jobs.data(), count));
  CUDA_OK(cudaDeviceSynchronize());
}

void compare_tcq_batch(const std::vector<TcqCase>& cases) {
  const int count = static_cast<int>(cases.size());
  CUDA_OK(cudaMemcpy(g_staging->eob.data(), g_workspace.eob,
                     sizeof(std::int32_t) * static_cast<std::size_t>(count),
                     cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(g_staging->max_level.data(),
                     g_workspace.max_level_searched,
                     sizeof(std::int32_t) * static_cast<std::size_t>(count),
                     cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(g_staging->truncated.data(),
                     g_workspace.level_range_truncated,
                     sizeof(std::uint8_t) * static_cast<std::size_t>(count),
                     cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(g_staging->distortion.data(), g_workspace.distortion,
                     sizeof(double) * static_cast<std::size_t>(count),
                     cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(g_staging->rate_bits.data(), g_workspace.rate_bits,
                     sizeof(double) * static_cast<std::size_t>(count),
                     cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(g_staging->cost.data(), g_workspace.cost,
                     sizeof(double) * static_cast<std::size_t>(count),
                     cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(g_staging->status.data(), g_workspace.status,
                     sizeof(std::int32_t) * static_cast<std::size_t>(count),
                     cudaMemcpyDeviceToHost));
  for (int i = 0; i < count; ++i) {
    const TcqCase& test_case = cases[static_cast<std::size_t>(i)];
    const ctcq::TcqJob& job = test_case.job;
    const int n = test_case.count;
    std::vector<std::int32_t> base(static_cast<std::size_t>(n), 0);
    std::vector<std::int32_t> quant(static_cast<std::size_t>(n), 0);
    std::vector<std::int32_t> states(static_cast<std::size_t>(n), 0);
    CUDA_OK(cudaMemcpy(base.data(),
                       g_workspace.base_levels + job.output_offset,
                       sizeof(std::int32_t) * static_cast<std::size_t>(n),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(quant.data(),
                       g_workspace.quant_levels + job.output_offset,
                       sizeof(std::int32_t) * static_cast<std::size_t>(n),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(states.data(), g_workspace.states + job.output_offset,
                       sizeof(std::int32_t) * static_cast<std::size_t>(n),
                       cudaMemcpyDeviceToHost));
    EXPECT(g_staging->status[static_cast<std::size_t>(i)] == 0);
    EXPECT(g_staging->eob[static_cast<std::size_t>(i)] == test_case.cpu.eob);
    EXPECT(g_staging->max_level[static_cast<std::size_t>(i)] ==
           test_case.cpu.max_level_searched);
    EXPECT((g_staging->truncated[static_cast<std::size_t>(i)] != 0) ==
           test_case.cpu.level_range_truncated);
    EXPECT(g_staging->distortion[static_cast<std::size_t>(i)] ==
           test_case.cpu.distortion);
    EXPECT(g_staging->rate_bits[static_cast<std::size_t>(i)] ==
           test_case.cpu.rate_bits);
    EXPECT(g_staging->cost[static_cast<std::size_t>(i)] ==
           test_case.cpu.cost);
    for (int c = 0; c < n; ++c) {
      EXPECT(base[static_cast<std::size_t>(c)] ==
             test_case.cpu.base_levels[static_cast<std::size_t>(c)]);
      EXPECT(quant[static_cast<std::size_t>(c)] ==
             test_case.cpu.quant_levels[static_cast<std::size_t>(c)]);
      EXPECT(states[static_cast<std::size_t>(c)] ==
             test_case.cpu.states[static_cast<std::size_t>(c)]);
    }
  }
}

void test_tcq_gpu() {
  long long cases = 0;
  std::vector<TcqCase> batch;
  batch.reserve(static_cast<std::size_t>(g_config.max_jobs));
  auto flush = [&]() {
    if (batch.empty()) return;
    launch_tcq_batch(batch);
    compare_tcq_batch(batch);
    cases += static_cast<long long>(batch.size());
    batch.clear();
  };
  for (const SizeCase& size : size_cases()) {
    for (int bitdepth : {8, 10}) {
      for (int trial = 0; trial < 2; ++trial) {
        TcqCase test_case;
        const int slot = static_cast<int>(batch.size());
        if (!build_tcq_case(size, /*plane=*/0, bitdepth, trial, slot,
                            &test_case)) {
          continue;
        }
        batch.push_back(std::move(test_case));
        if (static_cast<int>(batch.size()) == g_config.max_jobs) flush();
      }
    }
  }
  flush();
  // Plane 1/2 spot checks.
  for (int plane : {1, 2}) {
    for (int tx_size : {0, 3, 5, 12, 24}) {
      for (int tx_class = 0; tx_class < 3; ++tx_class) {
        TcqCase test_case;
        const int slot = static_cast<int>(batch.size());
        if (!build_tcq_case({tx_size, tx_class}, plane, 10, 1, slot,
                            &test_case)) {
          continue;
        }
        batch.push_back(std::move(test_case));
        if (static_cast<int>(batch.size()) == g_config.max_jobs) flush();
      }
    }
  }
  flush();
  std::printf("tcq: %lld blocks launched\n", cases);
  EXPECT(cases >= 170);
}

// Two identical launches must not grow the arena and must produce identical
// output.
void test_two_launches_no_growth() {
  TcqCase test_case;
  if (!build_tcq_case({4, 0}, 0, 10, 1, 0, &test_case)) {
    EXPECT(false);
    return;
  }
  std::vector<TcqCase> batch;
  batch.push_back(test_case);
  const std::uint64_t used_before = g_arena.used();
  const std::uint64_t peak_before = g_arena.peak_used();
  const std::uint64_t mark = g_arena.mark();

  launch_tcq_batch(batch);
  std::vector<std::int32_t> first(
      static_cast<std::size_t>(test_case.count), 0);
  CUDA_OK(cudaMemcpy(first.data(),
                     g_workspace.base_levels + test_case.job.output_offset,
                     sizeof(std::int32_t) *
                         static_cast<std::size_t>(test_case.count),
                     cudaMemcpyDeviceToHost));
  launch_tcq_batch(batch);
  std::vector<std::int32_t> second(
      static_cast<std::size_t>(test_case.count), 0);
  CUDA_OK(cudaMemcpy(second.data(),
                     g_workspace.base_levels + test_case.job.output_offset,
                     sizeof(std::int32_t) *
                         static_cast<std::size_t>(test_case.count),
                     cudaMemcpyDeviceToHost));
  EXPECT(g_arena.used() == used_before);
  EXPECT(g_arena.peak_used() == peak_before);
  EXPECT(g_arena.mark() == mark);
  EXPECT(std::memcmp(first.data(), second.data(),
                     sizeof(std::int32_t) *
                         static_cast<std::size_t>(test_case.count)) == 0);
  std::printf("two-launch: used=%llu peak=%llu unchanged\n",
              static_cast<unsigned long long>(used_before),
              static_cast<unsigned long long>(peak_before));
}

// ---------------------------------------------------------------------------
// CCTX
// ---------------------------------------------------------------------------

void test_cctx_gpu() {
  long long blocks = 0;
  std::vector<ctcq::CctxJob> jobs;
  std::vector<std::vector<tcq::CctxResult>> expected;
  auto flush = [&]() {
    if (jobs.empty()) return;
    const int count = static_cast<int>(jobs.size());
    CUDA_OK(ctcq::cctx_search_batch(g_workspace, jobs.data(), count));
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<std::int32_t> types(static_cast<std::size_t>(count) * 7, -1);
    std::vector<double> distortion(static_cast<std::size_t>(count) * 7, 0.0);
    std::vector<std::int32_t> eob_u(static_cast<std::size_t>(count) * 7, 0);
    std::vector<std::int32_t> eob_v(static_cast<std::size_t>(count) * 7, 0);
    std::vector<std::int32_t> chosen(static_cast<std::size_t>(count), -1);
    std::vector<std::int32_t> chosen_eob_u(static_cast<std::size_t>(count), 0);
    std::vector<std::int32_t> chosen_eob_v(static_cast<std::size_t>(count), 0);
    std::vector<double> chosen_dist(static_cast<std::size_t>(count), 0.0);
    CUDA_OK(cudaMemcpy(types.data(), g_workspace.cctx_candidate_type,
                       sizeof(std::int32_t) *
                           static_cast<std::size_t>(count) * 7,
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(distortion.data(),
                       g_workspace.cctx_candidate_distortion,
                       sizeof(double) * static_cast<std::size_t>(count) * 7,
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(eob_u.data(), g_workspace.cctx_candidate_eob_u,
                       sizeof(std::int32_t) *
                           static_cast<std::size_t>(count) * 7,
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(eob_v.data(), g_workspace.cctx_candidate_eob_v,
                       sizeof(std::int32_t) *
                           static_cast<std::size_t>(count) * 7,
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(chosen.data(), g_workspace.cctx_best_type,
                       sizeof(std::int32_t) * static_cast<std::size_t>(count),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(chosen_dist.data(),
                       g_workspace.cctx_best_distortion,
                       sizeof(double) * static_cast<std::size_t>(count),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(chosen_eob_u.data(), g_workspace.cctx_best_eob_u,
                       sizeof(std::int32_t) * static_cast<std::size_t>(count),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(chosen_eob_v.data(), g_workspace.cctx_best_eob_v,
                       sizeof(std::int32_t) * static_cast<std::size_t>(count),
                       cudaMemcpyDeviceToHost));
    std::vector<std::int32_t> u_levels;
    std::vector<std::int32_t> v_levels;
    for (int j = 0; j < count; ++j) {
      const ctcq::CctxJob& job = jobs[static_cast<std::size_t>(j)];
      const std::vector<tcq::CctxResult>& candidates =
          expected[static_cast<std::size_t>(j)];
      const int n = job.num_coefficients;
      u_levels.assign(static_cast<std::size_t>(7) *
                          static_cast<std::size_t>(n),
                      0);
      v_levels.assign(u_levels.size(), 0);
      CUDA_OK(cudaMemcpy(u_levels.data(),
                         g_workspace.cctx_u_levels + job.candidate_offset,
                         sizeof(std::int32_t) * u_levels.size(),
                         cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(v_levels.data(),
                         g_workspace.cctx_v_levels + job.candidate_offset,
                         sizeof(std::int32_t) * v_levels.size(),
                         cudaMemcpyDeviceToHost));
      for (int t = 0; t < 7; ++t) {
        const int slot = j * 7 + t;
        if (t < static_cast<int>(candidates.size())) {
          EXPECT(types[static_cast<std::size_t>(slot)] == t);
          EXPECT(distortion[static_cast<std::size_t>(slot)] ==
                 candidates[static_cast<std::size_t>(t)].distortion);
          EXPECT(eob_u[static_cast<std::size_t>(slot)] ==
                 candidates[static_cast<std::size_t>(t)].eob_u);
          EXPECT(eob_v[static_cast<std::size_t>(slot)] ==
                 candidates[static_cast<std::size_t>(t)].eob_v);
          for (int i = 0; i < n; ++i) {
            EXPECT(u_levels[static_cast<std::size_t>(t * n + i)] ==
                   candidates[static_cast<std::size_t>(t)]
                       .u_levels[static_cast<std::size_t>(i)]);
            EXPECT(v_levels[static_cast<std::size_t>(t * n + i)] ==
                   candidates[static_cast<std::size_t>(t)]
                       .v_levels[static_cast<std::size_t>(i)]);
          }
        } else {
          EXPECT(types[static_cast<std::size_t>(slot)] == -1);
        }
      }
      std::size_t best_index = 0;
      for (std::size_t t = 1; t < candidates.size(); ++t) {
        if (candidates[t].distortion < candidates[best_index].distortion) {
          best_index = t;
        }
      }
      EXPECT(chosen[static_cast<std::size_t>(j)] ==
             static_cast<std::int32_t>(candidates[best_index].type));
      EXPECT(chosen_dist[static_cast<std::size_t>(j)] ==
             candidates[best_index].distortion);
      EXPECT(chosen_eob_u[static_cast<std::size_t>(j)] ==
             candidates[best_index].eob_u);
      EXPECT(chosen_eob_v[static_cast<std::size_t>(j)] ==
             candidates[best_index].eob_v);
    }
    blocks += count;
    jobs.clear();
    expected.clear();
  };

  for (const SizeCase& size : size_cases()) {
    if (size.tx_class != 0) continue;
    parity::Geometry geometry;
    if (!parity::make_geometry(size.tx_size, parity::TxClass::k2D, 1,
                               &geometry, nullptr)) {
      continue;
    }
    const int n = geometry.w * geometry.h;
    if (n > g_config.max_coefficients) continue;
    for (int bitdepth : {8, 10}) {
      const double scale = bitdepth == 8 ? 127.0 : 511.0;
      std::vector<double> u(static_cast<std::size_t>(n), 0.0);
      std::vector<double> v(static_cast<std::size_t>(n), 0.0);
      for (int i = 0; i < n; ++i) {
        u[static_cast<std::size_t>(i)] = uniform(-scale, scale);
        v[static_cast<std::size_t>(i)] = uniform(-scale, scale);
      }
      const double step_u = uniform(0.5, 3.0);
      const double step_v = uniform(0.5, 3.0);
      for (int enabled = 0; enabled <= 1; ++enabled) {
        const int slot = static_cast<int>(jobs.size());
        ctcq::CctxJob job = ctcq::tcq_make_cctx_job(
            g_config, slot, n, step_u, step_v, bitdepth, enabled != 0);
        CUDA_OK(cudaMemcpy(g_workspace.u_coeff + job.coefficient_offset,
                           u.data(),
                           sizeof(double) * static_cast<std::size_t>(n),
                           cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(g_workspace.v_coeff + job.coefficient_offset,
                           v.data(),
                           sizeof(double) * static_cast<std::size_t>(n),
                           cudaMemcpyHostToDevice));
        tcq::CctxConfig cpu_config;
        cpu_config.enabled = enabled != 0;
        cpu_config.quant_step_u = step_u;
        cpu_config.quant_step_v = step_v;
        cpu_config.bitdepth = bitdepth;
        std::vector<tcq::CctxResult> candidates;
        std::string error;
        if (!tcq::cctx_enumerate_scalar(
                std::span<const double>(u), std::span<const double>(v), n, 1,
                cpu_config, &candidates, &error)) {
          continue;
        }
        jobs.push_back(job);
        expected.push_back(std::move(candidates));
        if (static_cast<int>(jobs.size()) == g_config.max_jobs) flush();
      }
    }
  }
  flush();
  std::printf("cctx: %lld blocks (7 candidates each when enabled)\n", blocks);
  EXPECT(blocks > 40);
}

// ---------------------------------------------------------------------------
// Parity hiding
// ---------------------------------------------------------------------------

void test_parity_gpu() {
  long long blocks = 0;
  long long hidden_blocks = 0;
  std::vector<ctcq::ParityJob> jobs;
  std::vector<parity::QuantizeResult> results;
  auto flush = [&]() {
    if (jobs.empty()) return;
    CUDA_OK(ctcq::parity_hide_batch(g_workspace, jobs.data(),
                                    static_cast<int>(jobs.size())));
    CUDA_OK(cudaDeviceSynchronize());
    const int count = static_cast<int>(jobs.size());
    std::vector<std::int32_t> eob(static_cast<std::size_t>(count), 0);
    std::vector<std::int32_t> hidden(static_cast<std::size_t>(count), 0);
    std::vector<std::int32_t> hidable(static_cast<std::size_t>(count), 0);
    std::vector<std::int32_t> considered(static_cast<std::size_t>(count), 0);
    std::vector<std::int32_t> turned_off(static_cast<std::size_t>(count), 0);
    std::vector<std::int32_t> adjusted_index(static_cast<std::size_t>(count),
                                             -1);
    std::vector<std::int32_t> adjusted_delta(static_cast<std::size_t>(count),
                                             0);
    std::vector<double> distortion(static_cast<std::size_t>(count), 0.0);
    std::vector<std::int32_t> status(static_cast<std::size_t>(count), -1);
    CUDA_OK(cudaMemcpy(eob.data(), g_workspace.parity_eob,
                       sizeof(std::int32_t) * static_cast<std::size_t>(count),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(hidden.data(), g_workspace.parity_hidden,
                       sizeof(std::int32_t) * static_cast<std::size_t>(count),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(hidable.data(), g_workspace.parity_hidable,
                       sizeof(std::int32_t) * static_cast<std::size_t>(count),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(considered.data(),
                       g_workspace.parity_hiding_considered,
                       sizeof(std::int32_t) * static_cast<std::size_t>(count),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(turned_off.data(),
                       g_workspace.parity_hiding_turned_off,
                       sizeof(std::int32_t) * static_cast<std::size_t>(count),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(adjusted_index.data(),
                       g_workspace.parity_adjusted_scan_index,
                       sizeof(std::int32_t) * static_cast<std::size_t>(count),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(adjusted_delta.data(),
                       g_workspace.parity_adjusted_delta,
                       sizeof(std::int32_t) * static_cast<std::size_t>(count),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(distortion.data(), g_workspace.parity_distortion,
                       sizeof(double) * static_cast<std::size_t>(count),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(status.data(), g_workspace.parity_status,
                       sizeof(std::int32_t) * static_cast<std::size_t>(count),
                       cudaMemcpyDeviceToHost));
    for (int j = 0; j < count; ++j) {
      const ctcq::ParityJob& job = jobs[static_cast<std::size_t>(j)];
      const parity::QuantizeResult& cpu =
          results[static_cast<std::size_t>(j)];
      const int n = job.num_coefficients;
      std::vector<std::int32_t> levels(static_cast<std::size_t>(n), 0);
      CUDA_OK(cudaMemcpy(levels.data(),
                         g_workspace.parity_levels + job.output_offset,
                         sizeof(std::int32_t) *
                             static_cast<std::size_t>(n),
                         cudaMemcpyDeviceToHost));
      EXPECT(status[static_cast<std::size_t>(j)] == 0);
      EXPECT(eob[static_cast<std::size_t>(j)] == cpu.eob);
      EXPECT((hidden[static_cast<std::size_t>(j)] != 0) == cpu.hidden);
      EXPECT((hidable[static_cast<std::size_t>(j)] != 0) == cpu.hidable);
      EXPECT((considered[static_cast<std::size_t>(j)] != 0) ==
             cpu.hiding_considered);
      EXPECT((turned_off[static_cast<std::size_t>(j)] != 0) ==
             cpu.hiding_turned_off);
      EXPECT(adjusted_index[static_cast<std::size_t>(j)] ==
             cpu.adjusted_scan_index);
      EXPECT(adjusted_delta[static_cast<std::size_t>(j)] ==
             cpu.adjusted_delta);
      EXPECT(distortion[static_cast<std::size_t>(j)] == cpu.distortion);
      for (int c = 0; c < n; ++c) {
        EXPECT(levels[static_cast<std::size_t>(c)] ==
               cpu.levels[static_cast<std::size_t>(c)]);
      }
      if (cpu.hidden) ++hidden_blocks;
    }
    blocks += count;
    jobs.clear();
    results.clear();
  };

  for (const SizeCase& size : size_cases()) {
    for (int plane = 0; plane <= 2; ++plane) {
      parity::Geometry geometry;
      if (!parity::make_geometry(size.tx_size,
                                 static_cast<parity::TxClass>(size.tx_class),
                                 plane, &geometry, nullptr)) {
        continue;
      }
      const int n = geometry.w * geometry.h;
      if (n > g_config.max_coefficients) continue;
      std::vector<std::int32_t> scan(static_cast<std::size_t>(n), 0);
      std::vector<std::int32_t> pre_max(static_cast<std::size_t>(n), 0);
      int got_count = 0;
      int got_w = 0;
      int got_h = 0;
      if (!ctcq::tcq_scan_and_pre_max(size.tx_size, size.tx_class, plane,
                                      g_config.max_coefficients, &got_count,
                                      &got_w, &got_h, scan.data(),
                                      pre_max.data(), nullptr)) {
        continue;
      }
      const double step = uniform(1.0, 4.0);
      std::vector<double> original(static_cast<std::size_t>(n), 0.0);
      std::vector<std::int32_t> seed(static_cast<std::size_t>(n), 0);
      for (int i = 0; i < n; ++i) {
        original[static_cast<std::size_t>(i)] = uniform(-24.0, 24.0);
        std::int32_t level = static_cast<std::int32_t>(std::llround(
            original[static_cast<std::size_t>(i)] / step));
        if ((i % 6) == 0) level = 0;
        seed[static_cast<std::size_t>(i)] = level;
      }
      for (int allow = 0; allow <= 1; ++allow) {
        for (int mode = 0; mode < 2; ++mode) {
          const int slot = static_cast<int>(jobs.size());
          ctcq::ParityJob job = ctcq::tcq_make_parity_job(
              g_config, slot, n, geometry.w, size.tx_class, plane, step,
              allow != 0);
          job.exhaustive_max_area = mode == 0 ? 64 : 0;
          job.pruned_ac_candidates = 8;
          parity::Eligibility eligibility;
          eligibility.allow_parity_hiding = allow != 0;
          eligibility.lossless = false;
          eligibility.allow_tcq = false;
          eligibility.tx_type_is_idtx = false;
          eligibility.use_fsc = false;
          eligibility.is_inter = false;
          parity::QuantizeConfig cpu_config;
          cpu_config.eligibility = eligibility;
          cpu_config.mode = parity::QuantizeMode::kMinDistortion;
          cpu_config.quant_step = step;
          cpu_config.lambda = 0.0;
          cpu_config.rate.qctx = 0;
          cpu_config.rate.txb_skip_ctx = 0;
          cpu_config.rate.dc_sign_ctx = 0;
          cpu_config.rate.is_inter = false;
          cpu_config.exhaustive_max_area = job.exhaustive_max_area;
          cpu_config.pruned_ac_candidates = job.pruned_ac_candidates;
          parity::QuantizeResult cpu;
          std::string error;
          if (!parity::quantize(std::span<const double>(original),
                                std::span<const std::int32_t>(seed), geometry,
                                cpu_config, &cpu, &error)) {
            continue;
          }
          CUDA_OK(cudaMemcpy(g_workspace.original + job.coefficient_offset,
                             original.data(),
                             sizeof(double) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice));
          CUDA_OK(cudaMemcpy(g_workspace.scan + job.scan_offset,
                             scan.data(),
                             sizeof(std::int32_t) *
                                 static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice));
          CUDA_OK(cudaMemcpy(g_workspace.parity_levels +
                                 job.coefficient_offset,
                             seed.data(),
                             sizeof(std::int32_t) *
                                 static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice));
          jobs.push_back(job);
          results.push_back(cpu);
          if (static_cast<int>(jobs.size()) == g_config.max_jobs) flush();
        }
      }
    }
  }
  flush();
  std::printf("parity: %lld blocks (%lld hidden)\n", blocks, hidden_blocks);
  EXPECT(blocks > 800);
  EXPECT(hidden_blocks > 0);
}

// ---------------------------------------------------------------------------
// Arena contract
// ---------------------------------------------------------------------------

void test_arena_contract() {
  const std::uint64_t required = ctcq::required_device_bytes(g_config);
  EXPECT(required > 0);
  EXPECT((required % ctcq::kTcqAlignment) == 0);
  const ctcq::TcqBytes split = ctcq::tcq_bytes(g_config);
  EXPECT(split.total == required);

  // One byte short: distinct status, exact required size, arena untouched.
  av2enc::cuda::Arena small;
  std::string error;
  EXPECT(av2enc::cuda::is_ok(small.init(required - 1, 256, &error)));
  ctcq::TcqWorkspace rejected;
  std::uint64_t reported = 0;
  EXPECT(ctcq::workspace_create(small, g_config, &rejected, &reported,
                                &error) ==
        av2enc::cuda::Status::kArenaExhausted);
  EXPECT(reported == required);
  EXPECT(small.used() == 0);
  small.release();

  EXPECT(g_workspace.required_bytes == required);
  EXPECT(g_arena.used() == required);
  EXPECT(g_workspace.persistent_mark == required);
  std::printf(
      "arena: required=%llu (tcq=%llu cctx=%llu parity=%llu), used=%llu\n",
      static_cast<unsigned long long>(required),
      static_cast<unsigned long long>(split.tcq),
      static_cast<unsigned long long>(split.cctx),
      static_cast<unsigned long long>(split.parity),
      static_cast<unsigned long long>(g_arena.used()));
}

}  // namespace

int main() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    std::printf("tcq_test: no CUDA device available\n");
    return 77;
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
    std::printf("device: %s (cc %d.%d, %d SMs)\n", properties.name,
                properties.major, properties.minor,
                properties.multiProcessorCount);
  }

  g_config.max_jobs = 8;
  g_config.max_coefficients = 1024;
  g_config.max_levels = 48;
  const std::uint64_t required = ctcq::required_device_bytes(g_config);
  std::string error;
  if (!av2enc::cuda::is_ok(
          g_arena.init(required + 64ull * 1024 * 1024, 256, &error))) {
    std::printf("tcq_test: arena init failed: %s\n", error.c_str());
    return 2;
  }
  if (ctcq::workspace_create(g_arena, g_config, &g_workspace, nullptr,
                             &error) != av2enc::cuda::Status::kOk) {
    std::printf("tcq_test: workspace_create failed: %s\n", error.c_str());
    return 2;
  }
  Staging staging(g_config);
  g_staging = &staging;

  test_arena_contract();
  long long before = g_checks;
  test_tcq_gpu();
  std::printf("tcq checks: %lld\n", g_checks - before);
  before = g_checks;
  test_two_launches_no_growth();
  std::printf("no-growth checks: %lld\n", g_checks - before);
  before = g_checks;
  test_cctx_gpu();
  std::printf("cctx checks: %lld\n", g_checks - before);
  before = g_checks;
  test_parity_gpu();
  std::printf("parity checks: %lld\n", g_checks - before);

  g_arena.release();
  g_staging = nullptr;
  if (g_failures == 0) {
    std::printf("tcq_test: %lld checks passed\n", g_checks);
    return 0;
  }
  std::printf("tcq_test: %d/%lld checks FAILED\n", g_failures, g_checks);
  return 1;
}
