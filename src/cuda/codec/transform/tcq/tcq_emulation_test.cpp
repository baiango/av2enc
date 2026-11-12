// Host-emulation parity test for the CUDA TCQ / CCTX / parity-hiding core.
//
// Needs no GPU and no nvcc: build_and_test.sh provides a cuda_runtime.h stub so
// Arena and the workspace API compile, then this test compiles the exact device
// core (tcq_device.cuh) with the host compiler and compares it bit for bit
// against the CPU reference in src/encoder/tcq/.
//
//   ./build_and_test.sh
//
// Coverage:
//   * state map, syntax-level/escape mapping, §7.14.4 shift, level_limits and
//     the §5.20.7.30 scan for all 25 TX sizes x 3 scan classes x 3 planes;
//   * the derived level bound for randomized coefficients and caps;
//   * the 8-state DP against tcq::quantize() for randomized 8/10-bit blocks of
//     every TX size and scan class, two rate models (monotone and deliberately
//     non-monotone), all-zero blocks, level caps and the uncapped margin;
//   * CCTX: seven-candidate levels/eob/distortion, the joint dequant pair and
//     the forward rotation for 8/10-bit;
//   * parity hiding: minimum-distortion level choice against
//     parity::quantize(QuantizeMode::kMinDistortion), every eligibility
//     exclusion, exhaustive and pruned AC search;
//   * the arena contract: exact required bytes, one-byte-short rejection, and
//     no growth over repeated launches.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/codec/transform/tcq/tcq.h"
#include "cuda/codec/transform/tcq/tcq_device.cuh"
#include "cuda/codec/transform/tcq/tcq_host.h"
#include "codec/transform/tcq/cctx.h"
#include "codec/transform/tcq/parity/parity_hiding.h"
#include "codec/transform/tcq/tcq.h"

namespace ctcq = av2enc::cuda::tcq;
namespace ctcq_detail = av2enc::cuda::tcq::detail;
namespace tcq = av2enc::tcq;
namespace parity = av2enc::tcq::parity;

namespace {

int g_failures = 0;
long long g_checks = 0;

void check(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL [%d] %s\n", line, what);
  }
}

#define CHECK(condition) check((condition), #condition, __LINE__)

std::mt19937 g_rng(0x7c9a51u);

double uniform(double lo, double hi) {
  std::uniform_real_distribution<double> dist(lo, hi);
  return dist(g_rng);
}

int rand_int(int lo, int hi) {
  std::uniform_int_distribution<int> dist(lo, hi);
  return dist(g_rng);
}

// The CPU suite's deliberately non-monotone rate model (tcq_test.cpp), used to
// prove the DP does not assume monotonicity.
double nonmonotone_coefficient_rate(void* context, std::int32_t scan_pos,
                                    std::int32_t state,
                                    std::int32_t base_level,
                                    std::int32_t next_state,
                                    std::int32_t signed_quant,
                                    std::int32_t level_context,
                                    std::int32_t is_eob) {
  (void)context;
  (void)next_state;
  (void)signed_quant;
  (void)level_context;
  const std::int32_t magnitude = base_level < 0 ? -base_level : base_level;
  double bits = magnitude == 0 ? 0.65 : 1.1 + 0.31 * magnitude;
  bits += 0.07 * ((state * 7 + scan_pos * 3) % 5);
  if (is_eob != 0) bits += 0.37;
  return bits;
}

double nonmonotone_eob_rate(void* context, std::int32_t eob,
                            std::int32_t num_coefficients) {
  (void)context;
  (void)num_coefficients;
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

// 25 TX sizes x {2D, HORIZ, VERT}.
std::vector<SizeCase> size_cases() {
  std::vector<SizeCase> out;
  for (int tx_size = 0; tx_size < 25; ++tx_size) {
    for (int tx_class = 0; tx_class < 3; ++tx_class) {
      out.push_back({tx_size, tx_class});
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Constants and mapping parity
// ---------------------------------------------------------------------------

void test_state_and_syntax() {
  for (int state = 0; state < 8; ++state) {
    for (int parity = 0; parity < 2; ++parity) {
      CHECK(ctcq_detail::tcq_next_state(state, parity) ==
            tcq::next_state(state, parity));
    }
    CHECK(ctcq_detail::tcq_state_q0(state) == tcq::state_q0(state));
    CHECK(ctcq_detail::tcq_state_cdf_index(state) ==
          tcq::state_cdf_index(state));
  }
  for (std::int32_t pre_max = 5; pre_max <= 8; ++pre_max) {
    for (std::int32_t q = 0; q <= 200; ++q) {
      CHECK(ctcq_detail::tcq_syntax_level_for(q, pre_max) ==
            tcq::syntax_level_for(q, pre_max));
      CHECK(ctcq_detail::tcq_syntax_escape_for(q, pre_max) ==
            tcq::syntax_escape_for(q, pre_max));
    }
  }
  for (int allow = 0; allow <= 1; ++allow) {
    for (int plane = 0; plane <= 2; ++plane) {
      for (int lossless = 0; lossless <= 1; ++lossless) {
        for (int class_2d = 0; class_2d <= 1; ++class_2d) {
          for (int fsc = 0; fsc <= 1; ++fsc) {
            CHECK(ctcq_detail::tcq_dequant_shift_extra(
                      allow != 0, plane, lossless != 0, class_2d != 0,
                      fsc != 0) ==
                  tcq::dequant_shift_extra(allow != 0, plane, lossless != 0,
                                           class_2d != 0, fsc != 0));
            for (int pels : {16, 256, 257, 1024, 1025}) {
              CHECK(ctcq_detail::tcq_dequant_shift(
                        pels, allow != 0, plane, lossless != 0,
                        class_2d != 0, fsc != 0) ==
                    tcq::dequant_shift(pels, allow != 0, plane, lossless != 0,
                                       class_2d != 0, fsc != 0));
            }
          }
        }
      }
    }
  }
}

void test_level_limits_and_scan() {
  for (const SizeCase& size : size_cases()) {
    for (int plane = 0; plane <= 2; ++plane) {
      parity::Geometry geometry;
      CHECK(parity::make_geometry(size.tx_size,
                                  static_cast<parity::TxClass>(size.tx_class),
                                  plane, &geometry, nullptr));
      std::vector<std::int32_t> reference(
          static_cast<std::size_t>(geometry.w * geometry.h), 0);
      CHECK(parity::build_scan(geometry, std::span<std::int32_t>(reference),
                               nullptr));
      std::vector<std::int32_t> scan(
          static_cast<std::size_t>(geometry.w * geometry.h), 0);
      std::vector<std::int32_t> pre_max(scan.size(), 0);
      int count = 0;
      int w = 0;
      int h = 0;
      CHECK(ctcq::tcq_scan_and_pre_max(
          size.tx_size, size.tx_class, plane,
          static_cast<int>(scan.size()), &count, &w, &h, scan.data(),
          pre_max.data(), nullptr));
      CHECK(count == geometry.w * geometry.h);
      CHECK(w == geometry.w && h == geometry.h);
      for (int c = 0; c < count; ++c) {
        const std::int32_t pos = scan[static_cast<std::size_t>(c)];
        CHECK(pos == reference[static_cast<std::size_t>(c)]);
        const int row = pos / geometry.w;
        const int col = pos % geometry.w;
        const parity::LevelLimits limits = parity::level_limits(
            row, col, geometry.tx_class, plane);
        const ctcq_detail::TcQLevelLimits device_limits =
            ctcq_detail::tcq_level_limits(row, col, size.tx_class, plane);
        CHECK(device_limits.pre_max == limits.pre_max);
        CHECK(device_limits.base_max == limits.base_max);
        CHECK(device_limits.br_max == limits.br_max);
        CHECK(device_limits.base_levels == limits.base_levels);
        CHECK(device_limits.is_lf == limits.is_lf);
        CHECK(pre_max[static_cast<std::size_t>(c)] == limits.pre_max);
      }
    }
  }
}

void test_level_bound() {
  for (int trial = 0; trial < 4000; ++trial) {
    tcq::Coefficient coefficient;
    coefficient.original = (trial % 17 == 0) ? 0.0 : uniform(-200.0, 200.0);
    const double step = uniform(0.25, 16.0);
    const std::int32_t cap = (trial % 4 == 0) ? 0 : rand_int(1, 40);
    const bool allow = (trial & 1) != 0;
    tcq::Config config;
    config.quant_step = step;
    config.level_cap = cap;
    config.allow_zero_to_one = allow;
    std::int32_t bound = 0;
    bool truncated = false;
    CHECK(ctcq_detail::tcq_level_bound(coefficient.original, step, cap, allow,
                                       &bound, &truncated));
    CHECK(bound == tcq::level_bound(coefficient, config));
    tcq::Config uncapped = config;
    uncapped.level_cap = 0;
    const std::int32_t raw = tcq::level_bound(coefficient, uncapped);
    CHECK(truncated == (cap > 0 && raw > cap));
    CHECK(bound >= 0);
  }
}

// ---------------------------------------------------------------------------
// TCQ DP parity
// ---------------------------------------------------------------------------

struct TcqCaseResult {
  bool ok = false;
  bool matched = false;
};

TcqCaseResult run_tcq_case(ctcq::TcqWorkspace& workspace,
                           const ctcq::TcqConfig& config, int tx_size,
                           int tx_class, int plane, int bitdepth, int trial) {
  TcqCaseResult result;
  const int capacity = config.max_coefficients;
  std::vector<std::int32_t> scan(static_cast<std::size_t>(capacity), 0);
  std::vector<std::int32_t> pre_max(static_cast<std::size_t>(capacity), 0);
  int count = 0;
  int w = 0;
  int h = 0;
  if (!ctcq::tcq_scan_and_pre_max(tx_size, tx_class, plane, capacity, &count,
                                  &w, &h, scan.data(), pre_max.data(),
                                  nullptr)) {
    return result;
  }
  const double scale = bitdepth == 8 ? 32.0 : 128.0;
  const double step = (bitdepth == 8 ? 1.0 : 4.0) * uniform(0.75, 4.0);
  static const std::int32_t kCaps[4] = {5, 11, 23, 40};
  const std::int32_t cap = kCaps[trial & 3];
  const double lambda = (trial % 5 == 0) ? 0.0 : uniform(0.0, 4.0);
  const bool allow_zero = (trial & 1) != 0;

  std::vector<double> original(static_cast<std::size_t>(count), 0.0);
  for (int i = 0; i < count; ++i) {
    original[static_cast<std::size_t>(i)] =
        (trial % 7 == 3 && i % 5 != 0) ? 0.0 : uniform(-scale, scale);
  }

  std::vector<tcq::Coefficient> coefficients(static_cast<std::size_t>(count));
  tcq::Config cpu_config;
  cpu_config.lambda = lambda;
  cpu_config.quant_step = step;
  cpu_config.level_cap = cap;
  cpu_config.max_coefficients = count;
  cpu_config.allow_zero_to_one = allow_zero;
  std::int32_t max_bound = 0;
  for (int c = 0; c < count; ++c) {
    coefficients[static_cast<std::size_t>(c)].original =
        original[static_cast<std::size_t>(c)];
    coefficients[static_cast<std::size_t>(c)].pre_max =
        pre_max[static_cast<std::size_t>(c)];
    coefficients[static_cast<std::size_t>(c)].level_context = c & 3;
    const std::int32_t bound =
        tcq::level_bound(coefficients[static_cast<std::size_t>(c)],
                         cpu_config);
    if (bound < 0) return result;
    max_bound = std::max(max_bound, bound);
  }
  const std::int32_t table_levels = max_bound + 1;
  if (table_levels > config.max_levels) return result;

  const tcq::RateModel model = test_rate_model((trial ^ tx_size) & 1);
  ctcq::TcqJob job = ctcq::tcq_make_job(config, 0, count, table_levels, step,
                                        lambda, cap, allow_zero);
  std::vector<std::int32_t> level_context(static_cast<std::size_t>(count), 0);
  for (int c = 0; c < count; ++c) {
    level_context[static_cast<std::size_t>(c)] = c & 3;
  }
  std::string error;
  if (!ctcq::tcq_fill_rate_tables(job, original.data(), pre_max.data(),
                                  level_context.data(), model,
                                  workspace.normal_rate + job.rate_offset,
                                  workspace.eob_coeff_rate + job.rate_offset,
                                  workspace.block_eob_rate +
                                      job.eob_table_offset,
                                  &error)) {
    std::printf("  fill_rate_tables: %s\n", error.c_str());
    return result;
  }
  std::memcpy(workspace.original + job.coefficient_offset, original.data(),
              sizeof(double) * static_cast<std::size_t>(count));
  std::memcpy(workspace.pre_max + job.coefficient_offset, pre_max.data(),
              sizeof(std::int32_t) * static_cast<std::size_t>(count));
  std::memcpy(workspace.level_context + job.coefficient_offset,
              level_context.data(),
              sizeof(std::int32_t) * static_cast<std::size_t>(count));
  workspace.jobs[0] = job;

  const ctcq::TcqDeviceBatch batch = ctcq::tcq_batch(workspace, 1);
  ctcq_detail::tcq_run_job(batch, 0);
  result.ok = workspace.status[0] == 0;

  tcq::Result cpu;
  const bool cpu_ok =
      tcq::quantize(coefficients, cpu_config, model, cpu, &error);
  CHECK(cpu_ok == result.ok);
  if (!cpu_ok || !result.ok) return result;

  CHECK(workspace.eob[0] == cpu.eob);
  CHECK(workspace.max_level_searched[0] == cpu.max_level_searched);
  CHECK((workspace.level_range_truncated[0] != 0) ==
        cpu.level_range_truncated);
  CHECK(workspace.distortion[0] == cpu.distortion);
  CHECK(workspace.rate_bits[0] == cpu.rate_bits);
  CHECK(workspace.cost[0] == cpu.cost);
  for (int c = 0; c < count; ++c) {
    CHECK(workspace.base_levels[job.output_offset + c] ==
          cpu.base_levels[static_cast<std::size_t>(c)]);
    CHECK(workspace.quant_levels[job.output_offset + c] ==
          cpu.quant_levels[static_cast<std::size_t>(c)]);
    CHECK(workspace.states[job.output_offset + c] ==
          cpu.states[static_cast<std::size_t>(c)]);
  }
  // The encoder output must also decode through the normative mapping.
  tcq::DecodeResult decoded;
  CHECK(tcq::decode_levels(
      std::span<const std::int32_t>(cpu.base_levels), true, false, &decoded,
      nullptr, std::span<const std::int32_t>(pre_max.data(), static_cast<std::size_t>(count))));
  CHECK(decoded.eob == cpu.eob);
  result.matched = true;
  return result;
}

void test_tcq(ctcq::TcqWorkspace& workspace,
              const ctcq::TcqConfig& config) {
  long long cases = 0;
  long long comparisons = 0;
  for (const SizeCase& size : size_cases()) {
    for (int bitdepth : {8, 10}) {
      for (int trial = 0; trial < 2; ++trial) {
        const TcqCaseResult result =
            run_tcq_case(workspace, config, size.tx_size, size.tx_class,
                         /*plane=*/0, bitdepth, trial);
        if (result.ok) {
          ++cases;
          if (result.matched) ++comparisons;
        }
      }
    }
  }
  // Plane 1/2 spot checks (the DP is plane-agnostic; pre_max is not).
  for (int plane : {1, 2}) {
    for (int tx_size : {0, 3, 5, 12, 24}) {
      for (int tx_class = 0; tx_class < 3; ++tx_class) {
        const TcqCaseResult result = run_tcq_case(
            workspace, config, tx_size, tx_class, plane, 10, 1);
        if (result.ok) {
          ++cases;
          if (result.matched) ++comparisons;
        }
      }
    }
  }
  std::printf("  tcq: %lld accepted cases, %lld exact comparisons\n", cases,
              comparisons);
  CHECK(cases > 140);
  CHECK(comparisons == cases);
}

void test_tcq_uncapped(ctcq::TcqWorkspace& workspace,
                       const ctcq::TcqConfig& config) {
  // 4x4 blocks with level_cap = 0 exercise the +64 derived margin and
  // level_range_truncated == false.
  long long cases = 0;
  for (int bitdepth : {8, 10}) {
    for (int tx_class = 0; tx_class < 3; ++tx_class) {
      for (int trial = 0; trial < 4; ++trial) {
        const TcqCaseResult result =
            run_tcq_case(workspace, config, 0, tx_class, 0, bitdepth, trial);
        if (result.ok) ++cases;
      }
    }
  }
  std::printf("  tcq uncapped 4x4: %lld accepted cases\n", cases);
  CHECK(cases == 24);
}

// ---------------------------------------------------------------------------
// CCTX parity
// ---------------------------------------------------------------------------

void test_cctx_devices() {
  for (int type = 0; type < 7; ++type) {
    for (int bitdepth : {8, 10, 12}) {
      for (int v = -40; v <= 40; v += 13) {
        for (int u = -40; u <= 40; u += 17) {
          const tcq::CctxPair cpu =
              tcq::cctx_dequant_pair(u, v, static_cast<tcq::CctxType>(type),
                                     bitdepth);
          const ctcq_detail::TcQCctxPair device =
              ctcq_detail::tcq_cctx_dequant_pair(u, v, type, bitdepth);
          CHECK(device.u == cpu.u);
          CHECK(device.v == cpu.v);
        }
      }
      for (double u : {-200.0, -1.0, 0.0, 3.5, 180.0}) {
        for (double v : {-121.0, 0.0, 7.25, 199.0}) {
          double cpu_u = 0.0;
          double cpu_v = 0.0;
          tcq::cctx_forward_rotate(u, v, static_cast<tcq::CctxType>(type),
                                   &cpu_u, &cpu_v);
          double device_u = 0.0;
          double device_v = 0.0;
          ctcq_detail::tcq_cctx_forward_rotate(u, v, type, &device_u,
                                               &device_v);
          CHECK(device_u == cpu_u);
          CHECK(device_v == cpu_v);
        }
      }
    }
  }
}

void test_cctx(ctcq::TcqWorkspace& workspace,
               const ctcq::TcqConfig& config) {
  long long cases = 0;
  for (const SizeCase& size : size_cases()) {
    if (size.tx_class != 0) continue;
    parity::Geometry geometry;
    CHECK(parity::make_geometry(size.tx_size, parity::TxClass::k2D, 1,
                                &geometry, nullptr));
    const int count = geometry.w * geometry.h;
    if (count > config.max_coefficients) continue;
    for (int bitdepth : {8, 10}) {
      const double scale = bitdepth == 8 ? 127.0 : 511.0;
      std::vector<double> u(static_cast<std::size_t>(count), 0.0);
      std::vector<double> v(static_cast<std::size_t>(count), 0.0);
      for (int i = 0; i < count; ++i) {
        u[static_cast<std::size_t>(i)] = uniform(-scale, scale);
        v[static_cast<std::size_t>(i)] = uniform(-scale, scale);
      }
      const double step_u = uniform(0.5, 3.0);
      const double step_v = uniform(0.5, 3.0);
      for (int enabled = 0; enabled <= 1; ++enabled) {
        ctcq::CctxJob job = ctcq::tcq_make_cctx_job(
            config, 0, count, step_u, step_v, bitdepth, enabled != 0);
        std::memcpy(workspace.u_coeff + job.coefficient_offset, u.data(),
                    sizeof(double) * static_cast<std::size_t>(count));
        std::memcpy(workspace.v_coeff + job.coefficient_offset, v.data(),
                    sizeof(double) * static_cast<std::size_t>(count));
        workspace.cctx_jobs[0] = job;
        const ctcq::CctxDeviceBatch batch = ctcq::cctx_batch(workspace, 1);
        ctcq_detail::tcq_run_cctx_job(batch, 0);
        CHECK(workspace.cctx_status[0] == 0);

        tcq::CctxConfig cpu_config;
        cpu_config.enabled = enabled != 0;
        cpu_config.quant_step_u = step_u;
        cpu_config.quant_step_v = step_v;
        cpu_config.bitdepth = bitdepth;
        std::vector<tcq::CctxResult> candidates;
        std::string error;
        CHECK(tcq::cctx_enumerate_scalar(
            std::span<const double>(u), std::span<const double>(v), count, 1,
            cpu_config, &candidates, &error));
        CHECK(candidates.size() ==
              static_cast<std::size_t>(enabled != 0 ? 7 : 1));
        for (std::size_t t = 0; t < candidates.size(); ++t) {
          const int slot = static_cast<int>(t);
          CHECK(workspace.cctx_candidate_type[slot] == slot);
          CHECK(workspace.cctx_candidate_distortion[slot] ==
                candidates[t].distortion);
          CHECK(workspace.cctx_candidate_eob_u[slot] == candidates[t].eob_u);
          CHECK(workspace.cctx_candidate_eob_v[slot] == candidates[t].eob_v);
          for (int i = 0; i < count; ++i) {
            CHECK(workspace.cctx_u_levels[job.candidate_offset + slot * count +
                                          i] ==
                  candidates[t].u_levels[static_cast<std::size_t>(i)]);
            CHECK(workspace.cctx_v_levels[job.candidate_offset + slot * count +
                                          i] ==
                  candidates[t].v_levels[static_cast<std::size_t>(i)]);
          }
        }
        for (std::size_t t = candidates.size(); t < 7; ++t) {
          CHECK(workspace.cctx_candidate_type[static_cast<int>(t)] == -1);
        }
        tcq::CctxResult chosen;
        CHECK(tcq::cctx_quantize_scalar(std::span<const double>(u),
                                        std::span<const double>(v), count, 1,
                                        cpu_config, &chosen, &error));
        CHECK(workspace.cctx_best_type[0] ==
              static_cast<std::int32_t>(chosen.type));
        CHECK(workspace.cctx_best_distortion[0] == chosen.distortion);
        CHECK(workspace.cctx_best_eob_u[0] == chosen.eob_u);
        CHECK(workspace.cctx_best_eob_v[0] == chosen.eob_v);
        ++cases;
      }
    }
  }
  std::printf("  cctx: %lld blocks (7 candidates each when enabled)\n", cases);
  CHECK(cases > 40);
}

// ---------------------------------------------------------------------------
// Parity-hiding minimum-distortion parity
// ---------------------------------------------------------------------------

void test_parity(ctcq::TcqWorkspace& workspace,
                 const ctcq::TcqConfig& config) {
  long long cases = 0;
  long long hidden_cases = 0;
  long long repaired = 0;
  long long pruned = 0;
  for (const SizeCase& size : size_cases()) {
    for (int plane = 0; plane <= 2; ++plane) {
      parity::Geometry geometry;
      CHECK(parity::make_geometry(size.tx_size,
                                  static_cast<parity::TxClass>(size.tx_class),
                                  plane, &geometry, nullptr));
      const int count = geometry.w * geometry.h;
      if (count > config.max_coefficients) continue;
      std::vector<std::int32_t> scan(static_cast<std::size_t>(count), 0);
      std::vector<std::int32_t> pre_max(static_cast<std::size_t>(count), 0);
      int got_count = 0;
      int got_w = 0;
      int got_h = 0;
      CHECK(ctcq::tcq_scan_and_pre_max(
          size.tx_size, size.tx_class, plane, config.max_coefficients,
          &got_count, &got_w, &got_h, scan.data(), pre_max.data(), nullptr));
      CHECK(got_count == count && got_w == geometry.w);
      const double scale = 24.0;
      const double step = uniform(1.0, 4.0);
      std::vector<double> original(static_cast<std::size_t>(count), 0.0);
      std::vector<std::int32_t> seed(static_cast<std::size_t>(count), 0);
      for (int i = 0; i < count; ++i) {
        original[static_cast<std::size_t>(i)] = uniform(-scale, scale);
        const double scaled = original[static_cast<std::size_t>(i)] / step;
        std::int32_t level = static_cast<std::int32_t>(std::llround(scaled));
        if ((i % 6) == 0) level = 0;
        seed[static_cast<std::size_t>(i)] = level;
      }
      for (int allow = 0; allow <= 1; ++allow) {
        for (int mode = 0; mode < 2; ++mode) {
          ctcq::ParityJob job = ctcq::tcq_make_parity_job(
              config, 0, count, geometry.w, size.tx_class, plane, step,
              allow != 0);
          job.exhaustive_max_area = 64;
          job.pruned_ac_candidates = 8;
          if (mode == 1) {
            // Force the pruned path even on small blocks.
            job.exhaustive_max_area = 0;
          }
          std::memcpy(workspace.original + job.coefficient_offset,
                      original.data(),
                      sizeof(double) * static_cast<std::size_t>(count));
          std::memcpy(workspace.scan + job.scan_offset, scan.data(),
                      sizeof(std::int32_t) * static_cast<std::size_t>(count));
          std::memcpy(workspace.parity_levels + job.coefficient_offset,
                      seed.data(),
                      sizeof(std::int32_t) * static_cast<std::size_t>(count));
          workspace.parity_jobs[0] = job;
          const ctcq::ParityDeviceBatch batch =
              ctcq::parity_batch(workspace, 1);
          ctcq_detail::tcq_run_parity_job(batch, 0);

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
          const bool cpu_ok = parity::quantize(
              std::span<const double>(original),
              std::span<const std::int32_t>(seed), geometry, cpu_config, &cpu,
              &error);
          const bool device_ok = workspace.parity_status[0] == 0;
          CHECK(cpu_ok == device_ok);
          if (!cpu_ok) continue;
          ++cases;
          CHECK(workspace.parity_eob[0] == cpu.eob);
          CHECK((workspace.parity_hidden[0] != 0) == cpu.hidden);
          CHECK((workspace.parity_hidable[0] != 0) == cpu.hidable);
          CHECK((workspace.parity_hiding_considered[0] != 0) ==
                cpu.hiding_considered);
          CHECK((workspace.parity_hiding_turned_off[0] != 0) ==
                cpu.hiding_turned_off);
          CHECK(workspace.parity_adjusted_scan_index[0] ==
                cpu.adjusted_scan_index);
          CHECK(workspace.parity_adjusted_delta[0] == cpu.adjusted_delta);
          CHECK(workspace.parity_distortion[0] == cpu.distortion);
          for (int i = 0; i < count; ++i) {
            CHECK(workspace.parity_levels[job.output_offset + i] ==
                  cpu.levels[static_cast<std::size_t>(i)]);
          }
          if (cpu.hidden) ++hidden_cases;
          if (cpu.adjusted_scan_index >= 0) ++repaired;
          if (job.exhaustive_max_area == 0) ++pruned;
          // The device path does not compute §9.3 CDF bits.
          CHECK(workspace.parity_status[0] == 0);
        }
      }
    }
  }
  std::printf(
      "  parity: %lld blocks (%lld hidden, %lld repaired, %lld pruned)\n",
      cases, hidden_cases, repaired, pruned);
  CHECK(cases > 200);
  CHECK(hidden_cases > 0);
  CHECK(repaired > 0);
  CHECK(pruned > 0);
}

// ---------------------------------------------------------------------------
// Arena contract
// ---------------------------------------------------------------------------

void test_arena() {
  ctcq::TcqConfig config;
  config.max_jobs = 8;
  config.max_coefficients = 1024;
  config.max_levels = 32;
  const std::uint64_t required = ctcq::required_device_bytes(config);
  CHECK(required > 0);
  CHECK((required % ctcq::kTcqAlignment) == 0);
  const ctcq::TcqBytes split = ctcq::tcq_bytes(config);
  CHECK(split.total == required);
  CHECK(split.tcq > 0 && split.cctx > 0 && split.parity > 0);

  {
    av2enc::cuda::Arena small;
    std::string error;
    CHECK(av2enc::cuda::is_ok(small.init(required - 1, 256, &error)));
    ctcq::TcqWorkspace rejected;
    std::uint64_t reported = 0;
    CHECK(ctcq::workspace_create(small, config, &rejected, &reported,
                                 &error) ==
          av2enc::cuda::Status::kArenaExhausted);
    CHECK(reported == required);
    CHECK(small.used() == 0);
    small.release();
  }

  av2enc::cuda::Arena arena;
  std::string error;
  CHECK(av2enc::cuda::is_ok(arena.init(required + 512, 256, &error)));
  ctcq::TcqWorkspace workspace;
  std::uint64_t reported = 0;
  CHECK(ctcq::workspace_create(arena, config, &workspace, &reported,
                               &error) == av2enc::cuda::Status::kOk);
  CHECK(reported == required);
  CHECK(workspace.required_bytes == required);
  CHECK(arena.used() == required);
  CHECK(workspace.persistent_mark == required);
  CHECK(arena.peak_used() == required);

  // Repeated launches must not move the watermark.
  const std::uint64_t mark = arena.mark();
  for (int pass = 0; pass < 2; ++pass) {
    CHECK(arena.mark() == mark);
    CHECK(arena.rewind(mark));
    CHECK(arena.used() == required);
  }
  CHECK(arena.peak_used() == required);

  ctcq::TcqConfig invalid;
  invalid.max_jobs = 0;
  CHECK(ctcq::required_device_bytes(invalid) == 0);
  ctcq::TcqWorkspace rejected;
  CHECK(ctcq::workspace_create(arena, invalid, &rejected, nullptr, &error) ==
        av2enc::cuda::Status::kInvalidArgument);
  std::printf("  arena: required=%llu bytes (tcq=%llu cctx=%llu parity=%llu)\n",
              static_cast<unsigned long long>(required),
              static_cast<unsigned long long>(split.tcq),
              static_cast<unsigned long long>(split.cctx),
              static_cast<unsigned long long>(split.parity));
  arena.release();
}

}  // namespace

int main() {
  std::printf("av2enc CUDA TCQ / CCTX / parity-hiding emulation parity\n");

  ctcq::TcqConfig config;
  config.max_jobs = 4;
  config.max_coefficients = 1024;
  config.max_levels = 48;
  const std::uint64_t required = ctcq::required_device_bytes(config);
  av2enc::cuda::Arena arena;
  std::string error;
  if (!av2enc::cuda::is_ok(arena.init(required + 4096, 256, &error))) {
    std::printf("arena init failed: %s\n", error.c_str());
    return 2;
  }
  ctcq::TcqWorkspace workspace;
  if (ctcq::workspace_create(arena, config, &workspace, nullptr, &error) !=
      av2enc::cuda::Status::kOk) {
    std::printf("workspace_create failed: %s\n", error.c_str());
    return 2;
  }

  test_state_and_syntax();
  test_level_limits_and_scan();
  test_level_bound();
  long long before = g_checks;
  test_tcq(workspace, config);
  std::printf("  tcq checks: %lld\n", g_checks - before);

  ctcq::TcqConfig uncapped_config;
  uncapped_config.max_jobs = 1;
  uncapped_config.max_coefficients = 16;
  uncapped_config.max_levels = 200;
  const std::uint64_t uncapped_bytes =
      ctcq::required_device_bytes(uncapped_config);
  av2enc::cuda::Arena uncapped_arena;
  CHECK(av2enc::cuda::is_ok(
      uncapped_arena.init(uncapped_bytes + 512, 256, &error)));
  ctcq::TcqWorkspace uncapped_workspace;
  CHECK(ctcq::workspace_create(uncapped_arena, uncapped_config,
                               &uncapped_workspace, nullptr,
                               &error) == av2enc::cuda::Status::kOk);
  before = g_checks;
  test_tcq_uncapped(uncapped_workspace, uncapped_config);
  std::printf("  tcq uncapped checks: %lld\n", g_checks - before);
  uncapped_arena.release();

  before = g_checks;
  test_cctx_devices();
  std::printf("  cctx device checks: %lld\n", g_checks - before);
  before = g_checks;
  test_cctx(workspace, config);
  std::printf("  cctx checks: %lld\n", g_checks - before);
  before = g_checks;
  test_parity(workspace, config);
  std::printf("  parity checks: %lld\n", g_checks - before);
  before = g_checks;
  test_arena();
  std::printf("  arena checks: %lld\n", g_checks - before);

  arena.release();
  std::printf("%lld checks, %d failures\n", g_checks, g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
