// CPU-vs-GPU parity test for the full-trellis RDOQ batch kernel.
//
// The rate model is a host-built table, so the GPU never calls a
// transcendental and the comparison exercises only the DP/arithmetic parity.
// Build on the sm_86 box:
//
//   ./build_rdoq_batch_test.sh
//
// The CPU reference is compiled by the host compiler with -ffp-contract=off;
// nvcc uses -fmad=false.

#include "rdoq_batch.h"

#include "encoder/rdoq/rdoq.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using av2enc::rdoq::Coefficient;
using av2enc::rdoq::Config;
using av2enc::rdoq::RdoqResult;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

void check_cuda(cudaError_t status, const char* what) {
  ++g_checks;
  if (status != cudaSuccess) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(status));
  }
}

double host_rate(std::int32_t scan_pos, std::int32_t prev_abs_state,
                 std::int32_t magnitude, std::int32_t is_eob) {
  double bits = 0.0;
  if (magnitude == 0) {
    bits = prev_abs_state == 0 ? 0.4 : 1.1;
  } else {
    bits = 1.3 + 0.6 * std::log2(static_cast<double>(magnitude) + 1.0);
    if (prev_abs_state == 0) bits += 0.8;
    if (is_eob != 0) bits += 0.2;
  }
  return bits + 0.01 * static_cast<double>(scan_pos);
}

double host_eob_rate(std::int32_t eob, std::int32_t num_coefficients) {
  (void)num_coefficients;
  return 1.0 + 0.3 * std::log2(static_cast<double>(eob) + 1.0);
}

struct TableContext {
  const double* normal = nullptr;
  const double* eob = nullptr;
  const double* block_eob = nullptr;
  std::int32_t states = 0;
  std::int32_t levels = 0;
};

double table_coefficient_rate(void* context, std::int32_t scan_pos,
                              std::int32_t prev_abs_state,
                              std::int32_t level,
                              std::int32_t level_context,
                              std::int32_t is_eob) {
  (void)level_context;
  const TableContext* table = static_cast<const TableContext*>(context);
  const std::int32_t magnitude = std::abs(level);
  const std::int32_t index =
      (scan_pos * table->states + prev_abs_state) * table->levels + magnitude;
  return is_eob != 0 ? table->eob[index] : table->normal[index];
}

double table_eob_rate(void* context, std::int32_t eob,
                      std::int32_t num_coefficients) {
  (void)num_coefficients;
  const TableContext* table = static_cast<const TableContext*>(context);
  return table->block_eob[eob];
}

struct BlockFixture {
  std::vector<Coefficient> coefficients;
  Config config;
  std::vector<std::int32_t> sign;
  std::vector<std::int32_t> bound;
  std::vector<double> normal;
  std::vector<double> eob;
  std::vector<double> block_eob;
  std::int32_t states = 0;
  std::int32_t table_levels = 0;
  RdoqResult cpu;
};

BlockFixture make_block(std::size_t count, std::int32_t cap, double lambda,
                        double quant_step, std::uint32_t seed,
                        bool all_zero = false) {
  BlockFixture fixture;
  fixture.config.lambda = lambda;
  fixture.config.quant_step = quant_step;
  fixture.config.prev_level_cap = cap;
  fixture.config.level_cap = 0;
  fixture.config.allow_zero_to_one = false;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> value(-500.0, 500.0);
  fixture.coefficients.resize(count);
  fixture.sign.resize(count);
  fixture.bound.resize(count);
  std::int32_t max_bound = 0;
  for (std::size_t i = 0; i < count; ++i) {
    Coefficient& coefficient = fixture.coefficients[i];
    coefficient.original = all_zero ? 0.0 : value(rng);
    coefficient.level_context = static_cast<std::int32_t>(i % 4);
    coefficient.scalar_level =
        static_cast<std::int32_t>(std::lround(coefficient.original / quant_step));
    const std::int32_t bound =
        av2enc::rdoq::rdoq_level_bound(coefficient, fixture.config);
    if (bound < 0) {
      std::fprintf(stderr, "invalid level bound\n");
      std::exit(1);
    }
    fixture.bound[i] = bound;
    max_bound = std::max(max_bound, bound);
    if (coefficient.original < 0.0) {
      fixture.sign[i] = -1;
    } else if (coefficient.original > 0.0) {
      fixture.sign[i] = 1;
    } else {
      fixture.sign[i] = coefficient.scalar_level < 0 ? -1 : 1;
    }
  }

  fixture.states = cap + 1;
  fixture.table_levels = max_bound + 1;
  const std::size_t table_size = count *
                                 static_cast<std::size_t>(fixture.states) *
                                 static_cast<std::size_t>(fixture.table_levels);
  fixture.normal.resize(table_size);
  fixture.eob.resize(table_size);
  for (std::size_t i = 0; i < count; ++i) {
    for (std::int32_t state = 0; state < fixture.states; ++state) {
      for (std::int32_t magnitude = 0; magnitude < fixture.table_levels;
           ++magnitude) {
        const std::size_t index =
            (i * static_cast<std::size_t>(fixture.states) +
             static_cast<std::size_t>(state)) *
                static_cast<std::size_t>(fixture.table_levels) +
            static_cast<std::size_t>(magnitude);
        fixture.normal[index] = host_rate(
            static_cast<std::int32_t>(i), state, magnitude, 0);
        fixture.eob[index] = host_rate(
            static_cast<std::int32_t>(i), state, magnitude, 1);
      }
    }
  }
  fixture.block_eob.resize(count + 1);
  for (std::size_t eob = 0; eob <= count; ++eob) {
    fixture.block_eob[eob] =
        host_eob_rate(static_cast<std::int32_t>(eob),
                      static_cast<std::int32_t>(count));
  }

  TableContext context;
  context.normal = fixture.normal.data();
  context.eob = fixture.eob.data();
  context.block_eob = fixture.block_eob.data();
  context.states = fixture.states;
  context.levels = fixture.table_levels;
  av2enc::rdoq::RateModel rate_model;
  rate_model.context = &context;
  rate_model.coefficient_rate = &table_coefficient_rate;
  rate_model.eob_rate = &table_eob_rate;
  std::string error;
  if (!av2enc::rdoq::rdoq_quantize(fixture.coefficients, fixture.config,
                                   rate_model, fixture.cpu, &error)) {
    std::fprintf(stderr, "CPU reference failed: %s\n", error.c_str());
    std::exit(1);
  }
  return fixture;
}

template <typename T>
void device_alloc(T** pointer, std::size_t count) {
  check_cuda(cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(T)),
             "cudaMalloc");
}

template <typename T>
void upload(T* device, const std::vector<T>& host) {
  check_cuda(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy host-to-device");
}

template <typename T>
void download(std::vector<T>* host, const T* device) {
  check_cuda(cudaMemcpy(host->data(), device, host->size() * sizeof(T),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy device-to-host");
}

}  // namespace

int main() {
  std::printf("av2enc CUDA RDOQ batch parity test\n");
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  std::printf("device: %s (sm_%d%d)\n", properties.name, properties.major,
              properties.minor);

  struct Case {
    std::size_t count;
    std::int32_t cap;
    double lambda;
    double step;
    std::uint32_t seed;
    bool all_zero;
  };
  const Case cases[] = {
      {16, 2, 5.0, 8.0, 1, false},
      {37, 2, 20.0, 6.0, 2, false},
      {64, 3, 40.0, 8.0, 3, false},
      {128, 2, 15.0, 4.0, 4, false},
      {256, 2, 60.0, 6.0, 5, false},
      {8, 2, 10.0, 8.0, 6, true},
  };

  std::vector<BlockFixture> fixtures;
  for (const Case& item : cases) {
    fixtures.push_back(make_block(item.count, item.cap, item.lambda, item.step,
                                  item.seed, item.all_zero));
  }

  std::size_t total_coefficients = 0;
  std::size_t total_table = 0;
  std::size_t total_path = 0;
  for (const BlockFixture& fixture : fixtures) {
    total_coefficients += fixture.coefficients.size();
    total_table += fixture.normal.size();
    total_path += fixture.coefficients.size() *
                  static_cast<std::size_t>(fixture.states);
  }

  std::vector<double> original(total_coefficients);
  std::vector<std::int32_t> sign(total_coefficients);
  std::vector<std::int32_t> bound(total_coefficients);
  std::vector<double> normal(total_table);
  std::vector<double> eob_rate(total_table);
  std::vector<std::int32_t> levels(total_coefficients);
  std::vector<std::int32_t> eob(fixtures.size());
  std::vector<double> distortion(fixtures.size());
  std::vector<double> rate_bits(fixtures.size());
  std::vector<double> cost(fixtures.size());
  std::vector<std::int32_t> status(fixtures.size());
  std::vector<std::int32_t> path_levels(total_path);
  std::vector<std::uint8_t> path_modes(total_path);
  std::vector<double> block_eob;
  std::vector<av2enc::cuda::rdoq::BlockJob> jobs;

  std::size_t coefficient_offset = 0;
  std::size_t table_offset = 0;
  std::size_t path_offset = 0;
  for (const BlockFixture& fixture : fixtures) {
    const std::size_t count = fixture.coefficients.size();
    for (std::size_t i = 0; i < count; ++i) {
      original[coefficient_offset + i] = fixture.coefficients[i].original;
      sign[coefficient_offset + i] = fixture.sign[i];
      bound[coefficient_offset + i] = fixture.bound[i];
    }
    std::copy(fixture.normal.begin(), fixture.normal.end(),
              normal.begin() + static_cast<std::ptrdiff_t>(table_offset));
    std::copy(fixture.eob.begin(), fixture.eob.end(),
              eob_rate.begin() + static_cast<std::ptrdiff_t>(table_offset));
    av2enc::cuda::rdoq::BlockJob job;
    job.coefficient_offset = static_cast<std::int32_t>(coefficient_offset);
    job.num_coefficients = static_cast<std::int32_t>(count);
    job.level_offset = static_cast<std::int32_t>(coefficient_offset);
    job.path_offset = static_cast<std::int32_t>(path_offset);
    job.table_offset = static_cast<std::int32_t>(table_offset);
    job.eob_table_offset = static_cast<std::int32_t>(block_eob.size());
    job.prev_level_cap = fixture.config.prev_level_cap;
    job.table_levels = fixture.table_levels;
    job.quant_step = fixture.config.quant_step;
    job.lambda = fixture.config.lambda;
    jobs.push_back(job);
    block_eob.insert(block_eob.end(), fixture.block_eob.begin(),
                     fixture.block_eob.end());
    coefficient_offset += count;
    table_offset += fixture.normal.size();
    path_offset += count * static_cast<std::size_t>(fixture.states);
  }

  double* d_original = nullptr;
  std::int32_t* d_sign = nullptr;
  std::int32_t* d_bound = nullptr;
  double* d_normal = nullptr;
  double* d_eob_rate = nullptr;
  double* d_block_eob = nullptr;
  av2enc::cuda::rdoq::BlockJob* d_jobs = nullptr;
  std::int32_t* d_levels = nullptr;
  std::int32_t* d_eob = nullptr;
  double* d_distortion = nullptr;
  double* d_rate = nullptr;
  double* d_cost = nullptr;
  std::int32_t* d_status = nullptr;
  std::int32_t* d_path_levels = nullptr;
  std::uint8_t* d_path_modes = nullptr;

  const std::size_t bytes_double = sizeof(double);
  device_alloc(&d_original, original.size());
  device_alloc(&d_sign, sign.size());
  device_alloc(&d_bound, bound.size());
  device_alloc(&d_normal, normal.size());
  device_alloc(&d_eob_rate, eob_rate.size());
  device_alloc(&d_block_eob, block_eob.size());
  device_alloc(&d_jobs, jobs.size());
  device_alloc(&d_levels, levels.size());
  device_alloc(&d_eob, eob.size());
  device_alloc(&d_distortion, distortion.size());
  device_alloc(&d_rate, rate_bits.size());
  device_alloc(&d_cost, cost.size());
  device_alloc(&d_status, status.size());
  device_alloc(&d_path_levels, path_levels.size());
  device_alloc(&d_path_modes, path_modes.size());

  upload(d_original, original);
  upload(d_sign, sign);
  upload(d_bound, bound);
  upload(d_normal, normal);
  upload(d_eob_rate, eob_rate);
  upload(d_block_eob, block_eob);
  upload(d_jobs, jobs);

  av2enc::cuda::rdoq::Batch batch;
  batch.original = d_original;
  batch.sign = d_sign;
  batch.bound = d_bound;
  batch.normal_rate = d_normal;
  batch.eob_rate = d_eob_rate;
  batch.block_eob_rate = d_block_eob;
  batch.jobs = d_jobs;
  batch.num_jobs = static_cast<std::int32_t>(jobs.size());
  batch.levels = d_levels;
  batch.eob = d_eob;
  batch.distortion = d_distortion;
  batch.rate_bits = d_rate;
  batch.cost = d_cost;
  batch.status = d_status;
  batch.path_levels = d_path_levels;
  batch.path_modes = d_path_modes;

  check_cuda(av2enc::cuda::rdoq::quantize_batch(batch, nullptr), "quantize_batch");
  check_cuda(cudaDeviceSynchronize(), "sync");
  download(&levels, d_levels);
  download(&eob, d_eob);
  download(&distortion, d_distortion);
  download(&rate_bits, d_rate);
  download(&cost, d_cost);
  download(&status, d_status);

  std::size_t global_offset = 0;
  for (std::size_t b = 0; b < fixtures.size(); ++b) {
    const BlockFixture& fixture = fixtures[b];
    const std::size_t count = fixture.coefficients.size();
    check(status[b] == 0, "job status is ok");
    check(eob[b] == fixture.cpu.eob, "GPU EOB matches CPU");
    bool levels_match = true;
    for (std::size_t i = 0; i < count; ++i) {
      if (levels[global_offset + i] != fixture.cpu.levels[i]) {
        levels_match = false;
        std::fprintf(stderr,
                     "  job %zu level %zu: gpu=%d cpu=%d\n", b, i,
                     levels[global_offset + i], fixture.cpu.levels[i]);
        break;
      }
    }
    check(levels_match, "GPU levels match CPU");
    const double scale = std::max(1.0, std::fabs(fixture.cpu.cost));
    check(std::fabs(cost[b] - fixture.cpu.cost) <= 1.0e-9 * scale,
          "GPU cost matches CPU");
    check(std::fabs(rate_bits[b] - fixture.cpu.rate_bits) <= 1.0e-9 * scale,
          "GPU rate matches CPU");
    global_offset += count;
  }

  cudaFree(d_original);
  cudaFree(d_sign);
  cudaFree(d_bound);
  cudaFree(d_normal);
  cudaFree(d_eob_rate);
  cudaFree(d_block_eob);
  cudaFree(d_jobs);
  cudaFree(d_levels);
  cudaFree(d_eob);
  cudaFree(d_distortion);
  cudaFree(d_rate);
  cudaFree(d_cost);
  cudaFree(d_status);
  cudaFree(d_path_levels);
  cudaFree(d_path_modes);

  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  if (g_failures != 0) {
    std::printf("RESULT: FAIL\n");
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
