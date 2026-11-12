// Real-GPU bit-exact test for the CUDA inter DDT component.
//
// Build/run with ./build_gpu_test.sh on the sm_86 box.
//
// Coverage:
//   * every one of the 25 sizes x 16 types, 8- and 10-bit, against the CPU
//     reference (src/encoder/m2/ddt/), including the 100 legal DDT pairs,
//   * the 32 alias-to-DCT behaviour (DDT-on == DDT-off at adjusted length 32),
//   * the inverse-from-levels path (§7.14.4 dequant + DDT inverse),
//   * batch offsets for mixed sizes,
//   * the arena contract: exact required_device_bytes(), a second launch that
//     does not grow peak usage, and a one-byte-short arena that returns
//     kArenaTooSmall with the same required size.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/ddt/ddt_gpu.h"
#include "cuda/ddt/ddt_test_pairs.h"
#include "cuda/error.h"
#include "encoder/m2/ddt/ddt.h"
#include "encoder/tx_variants/tx_variants.h"

namespace {

using av2enc::cuda::Arena;
using av2enc::cuda::ddt::BatchConfig;
using av2enc::cuda::ddt::BatchStatus;
using av2enc::cuda::ddt::DdtJob;
using av2enc::cuda::ddt::Direction;
using av2enc::m2::ddt::Context;
using av2enc::txv::TxParams;
using av2enc::txv::TxSize;
using av2enc::txv::TxType;
namespace ddt = av2enc::m2::ddt;
namespace txv = av2enc::txv;

int g_checks = 0;
int g_failures = 0;
std::uint64_t g_elements = 0;

void check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 30) {
      std::printf("FAIL: %s\n", what.c_str());
    }
  }
}

void expect_elements(const void* got, const void* expected, int count,
                     int element_size, const std::string& what) {
  ++g_checks;
  g_elements += static_cast<std::uint64_t>(count);
  if (std::memcmp(got, expected,
                  static_cast<std::size_t>(count) *
                      static_cast<std::size_t>(element_size)) != 0) {
    ++g_failures;
    if (g_failures <= 30) {
      std::printf("FAIL: %s (%d elements differ)\n", what.c_str(), count);
    }
  }
}

Context make_ctx(int depth) {
  Context ctx;
  ctx.enable_inter_ddt = true;
  ctx.is_inter = true;
  ctx.use_intrabc = false;
  ctx.bit_depth = depth;
  return ctx;
}

DdtJob make_job(const TxParams& p, int depth, int in_offset, int out_offset) {
  DdtJob job;
  job.params = p;
  job.in_offset = in_offset;
  job.out_offset = out_offset;
  job.enable_inter_ddt = 1;
  job.is_inter = 1;
  job.use_intrabc = 0;
  job.bit_depth = static_cast<std::uint8_t>(depth);
  return job;
}

int random_range(int low, int high) {
  static std::uint32_t rng = 0x12345678U;
  rng ^= rng << 13;
  rng ^= rng >> 17;
  rng ^= rng << 5;
  return low + static_cast<int>(rng % static_cast<std::uint32_t>(high - low + 1));
}

struct PairInfo {
  TxSize size;
  int type;
  TxParams params;
  bool legal_ddt;
  bool component;
};

std::vector<PairInfo> all_pairs() {
  std::vector<PairInfo> pairs;
  for (const auto& row : av2enc::cuda::ddt::test::kSizeRows) {
    for (int type = 0; type < 16; ++type) {
      const auto params =
          txv::make_tx_params(row.size, static_cast<TxType>(type));
      if (!params.has_value()) continue;
      PairInfo info;
      info.size = row.size;
      info.type = type;
      info.params = *params;
      info.legal_ddt = (row.ddt_mask & (1U << type)) != 0;
      info.component = (row.component_mask & (1U << type)) != 0;
      pairs.push_back(info);
    }
  }
  return pairs;
}

std::int16_t residual_value(int index, int w, int h, int shape) {
  const int x = index % w;
  const int y = index / w;
  switch (shape) {
    case 0:
      return 40;
    case 1:
      return static_cast<std::int16_t>(x < w / 2 ? -80 : 70);
    case 2:
      return static_cast<std::int16_t>(y < h / 2 ? -60 : 55);
    case 3:
      return static_cast<std::int16_t>(((x ^ y) & 1) ? 90 : -90);
    case 4:
      return static_cast<std::int16_t>((x * 7 + y * 3) % 121 - 60);
    default:
      return static_cast<std::int16_t>(random_range(-120, 120));
  }
}

std::int32_t dequant_value(int index, int shape) {
  if (shape == 1) {
    return (index & 1) ? 32767 : -32768;  // extreme clipping
  }
  if (shape == 2) {
    return (index % 7 == 0) ? 3000 : ((index * 37) % 401) - 200;
  }
  return random_range(-3000, 3000);
}

// Runs one mixed-size forward + inverse batch at `depth` and compares every
// block with the CPU reference.
void run_block_parity(const std::vector<PairInfo>& pairs, int depth,
                      int residual_shape, int dequant_shape,
                      const char* label) {
  const std::size_t count = pairs.size();
  std::vector<DdtJob> fwd_jobs(count);
  std::vector<DdtJob> inv_jobs(count);
  int adj_total = 0;
  int coded_total = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const TxParams& p = pairs[i].params;
    const int adj = p.w * p.h;
    fwd_jobs[i] = make_job(p, depth, adj_total, adj_total);
    inv_jobs[i] = make_job(p, depth, adj_total, coded_total);
    adj_total += adj;
    coded_total += p.coded_w * p.coded_h;
  }

  std::vector<std::int16_t> residual(static_cast<std::size_t>(adj_total), 0);
  std::vector<std::int32_t> dequant(static_cast<std::size_t>(adj_total), 0);
  for (std::size_t i = 0; i < count; ++i) {
    const TxParams& p = pairs[i].params;
    const int adj = p.w * p.h;
    const int base = fwd_jobs[i].in_offset;
    for (int k = 0; k < adj; ++k) {
      residual[static_cast<std::size_t>(base + k)] =
          residual_value(k, p.w, p.h, residual_shape);
      dequant[static_cast<std::size_t>(base + k)] =
          dequant_value(k + base, dequant_shape);
    }
  }

  BatchConfig fcfg;
  fcfg.direction = Direction::kForward;
  fcfg.max_blocks = static_cast<int>(count);
  fcfg.with_levels = true;
  BatchConfig icfg;
  icfg.direction = Direction::kInverse;
  icfg.max_blocks = static_cast<int>(count);

  const std::uint64_t need =
      std::max(av2enc::cuda::ddt::required_device_bytes(fcfg),
               av2enc::cuda::ddt::required_device_bytes(icfg));
  Arena arena;
  std::string arena_error;
  check(av2enc::cuda::is_ok(arena.init(need, Arena::kAlignment, &arena_error)),
        "arena init: " + arena_error);

  std::vector<std::int32_t> coeff(static_cast<std::size_t>(adj_total), 0);
  std::vector<std::int32_t> levels(static_cast<std::size_t>(adj_total), 0);
  std::uint64_t reported = 0;
  const BatchStatus fstatus = av2enc::cuda::ddt::forward_batch(
      arena, fwd_jobs.data(), static_cast<int>(count), residual.data(),
      coeff.data(), levels.data(), nullptr, &reported, &arena_error);
  check(fstatus == BatchStatus::kOk,
        std::string(label) + " forward: " + arena_error);
  check(reported == av2enc::cuda::ddt::required_device_bytes(fcfg),
        std::string(label) + " forward required bytes");

  std::vector<std::int16_t> coded(static_cast<std::size_t>(coded_total), 0);
  reported = 0;
  const BatchStatus istatus = av2enc::cuda::ddt::inverse_batch(
      arena, inv_jobs.data(), static_cast<int>(count), dequant.data(),
      coded.data(), false, nullptr, &reported, &arena_error);
  check(istatus == BatchStatus::kOk,
        std::string(label) + " inverse: " + arena_error);
  check(reported == av2enc::cuda::ddt::required_device_bytes(icfg),
        std::string(label) + " inverse required bytes");

  const Context ctx = make_ctx(depth);
  for (std::size_t i = 0; i < count; ++i) {
    const TxParams& p = pairs[i].params;
    const int adj = p.w * p.h;
    const int coded_n = p.coded_w * p.coded_h;

    std::vector<std::int32_t> cpu_coeff(static_cast<std::size_t>(adj), 0);
    check(ddt::forward_block(
              residual.data() + fwd_jobs[i].in_offset, p, ctx,
              cpu_coeff.data()),
          "CPU forward_block");
    expect_elements(coeff.data() + fwd_jobs[i].out_offset, cpu_coeff.data(),
                    adj, static_cast<int>(sizeof(std::int32_t)),
                    std::string(label) + " coeff " +
                        txv::tx_size_name(pairs[i].size) + " type " +
                        std::to_string(pairs[i].type));
    for (int k = 0; k < adj; ++k) {
      const std::int32_t expected_level =
          ddt::level_from_coeff(cpu_coeff[static_cast<std::size_t>(k)], p);
      check(levels[static_cast<std::size_t>(fwd_jobs[i].out_offset + k)] ==
                expected_level,
            std::string(label) + " level " +
                txv::tx_size_name(pairs[i].size) + " type " +
                std::to_string(pairs[i].type));
    }

    std::vector<std::int16_t> cpu_coded(static_cast<std::size_t>(coded_n), 0);
    check(ddt::inverse_block(dequant.data() + inv_jobs[i].in_offset, p, ctx,
                             cpu_coded.data()),
          "CPU inverse_block");
    expect_elements(coded.data() + inv_jobs[i].out_offset, cpu_coded.data(),
                    coded_n, static_cast<int>(sizeof(std::int16_t)),
                    std::string(label) + " inverse " +
                        txv::tx_size_name(pairs[i].size) + " type " +
                        std::to_string(pairs[i].type));
  }
}

// 32 alias-to-DCT: when the adjusted length is 32, §7.15.4.1 derives DDTX/FDDT
// but §7.15.2.1 uses Dct_Kernel32, so DDT-on and DDT-off reconstructions must
// be identical.  Checked through the public batch API.
void test_32_alias(const std::vector<PairInfo>& pairs) {
  const Context ctx = make_ctx(8);
  for (TxSize size : {TxSize::k32x32, TxSize::k64x64}) {
    const auto params_opt = txv::make_tx_params(size, TxType::kAdstAdst);
    check(params_opt.has_value(), "32 alias params");
    if (!params_opt.has_value()) continue;
    const TxParams& p = *params_opt;
    check(!ddt::plan_has_ddt(p, ctx),
          std::string("plan_has_ddt false at adjusted 32 ") +
              txv::tx_size_name(size));
    const int adj = p.w * p.h;
    const int coded_n = p.coded_w * p.coded_h;
    std::vector<std::int32_t> dequant(static_cast<std::size_t>(adj));
    for (int i = 0; i < adj; ++i) dequant[static_cast<std::size_t>(i)] =
        random_range(-3000, 3000);

    BatchConfig cfg;
    cfg.direction = Direction::kInverse;
    cfg.max_blocks = 1;
    Arena arena;
    std::string error;
    check(av2enc::cuda::is_ok(arena.init(
              av2enc::cuda::ddt::required_device_bytes(cfg),
              Arena::kAlignment, &error)),
          "32 alias arena: " + error);

    std::vector<std::int16_t> with_ddt(static_cast<std::size_t>(coded_n), 0);
    std::vector<std::int16_t> without_ddt(static_cast<std::size_t>(coded_n), 0);
    DdtJob job = make_job(p, 8, 0, 0);
    job.enable_inter_ddt = 1;
    check(av2enc::cuda::ddt::inverse_batch(arena, &job, 1, dequant.data(),
                                           with_ddt.data(), false, nullptr,
                                           nullptr, &error) ==
              BatchStatus::kOk,
          "32 alias DDT-on: " + error);
    job.enable_inter_ddt = 0;
    check(av2enc::cuda::ddt::inverse_batch(arena, &job, 1, dequant.data(),
                                           without_ddt.data(), false, nullptr,
                                           nullptr, &error) ==
              BatchStatus::kOk,
          "32 alias DDT-off: " + error);
    expect_elements(with_ddt.data(), without_ddt.data(), coded_n,
                    static_cast<int>(sizeof(std::int16_t)),
                    std::string("32 alias reconstruction ") +
                        txv::tx_size_name(size));

    std::vector<std::int16_t> cpu_coded(static_cast<std::size_t>(coded_n), 0);
    check(ddt::inverse_block(dequant.data(), p, ctx, cpu_coded.data()),
          "32 alias CPU");
    expect_elements(with_ddt.data(), cpu_coded.data(), coded_n,
                    static_cast<int>(sizeof(std::int16_t)),
                    std::string("32 alias vs CPU ") +
                        txv::tx_size_name(size));
  }
  (void)pairs;
}

// Inverse from quantised levels: §7.14.4 dequant first, then the DDT inverse.
void test_inverse_from_levels(const std::vector<PairInfo>& pairs, int depth) {
  std::vector<DdtJob> jobs;
  int adj_total = 0;
  int coded_total = 0;
  for (const PairInfo& info : pairs) {
    if (!info.legal_ddt) continue;
    const TxParams& p = info.params;
    jobs.push_back(make_job(p, depth, adj_total, coded_total));
    adj_total += p.w * p.h;
    coded_total += p.coded_w * p.coded_h;
  }
  check(static_cast<int>(jobs.size()) ==
            av2enc::cuda::ddt::test::kLegalDdtPairCount,
        "inverse-from-levels uses the 100 legal DDT pairs");

  std::vector<std::int32_t> levels(static_cast<std::size_t>(adj_total), 0);
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    const TxParams& p = jobs[i].params;
    const int adj = p.w * p.h;
    for (int k = 0; k < adj; ++k) {
      const int value = ((k * 13 + static_cast<int>(i) * 7) % 41) - 20;
      levels[static_cast<std::size_t>(jobs[i].in_offset + k)] = value;
    }
  }

  BatchConfig cfg;
  cfg.direction = Direction::kInverse;
  cfg.max_blocks = static_cast<int>(jobs.size());
  cfg.inverse_from_levels = true;
  Arena arena;
  std::string error;
  check(av2enc::cuda::is_ok(arena.init(
            av2enc::cuda::ddt::required_device_bytes(cfg), Arena::kAlignment,
            &error)),
        "levels arena: " + error);
  std::vector<std::int16_t> coded(static_cast<std::size_t>(coded_total), 0);
  const BatchStatus status = av2enc::cuda::ddt::inverse_batch(
      arena, jobs.data(), static_cast<int>(jobs.size()), levels.data(),
      coded.data(), true, nullptr, nullptr, &error);
  check(status == BatchStatus::kOk, "levels inverse: " + error);

  const Context ctx = make_ctx(depth);
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    const TxParams& p = jobs[i].params;
    const int adj = p.w * p.h;
    const int coded_n = p.coded_w * p.coded_h;
    std::vector<std::int32_t> dequant(static_cast<std::size_t>(adj), 0);
    txv::dequantize(levels.data() + jobs[i].in_offset, p, dequant.data());
    std::vector<std::int16_t> expected(static_cast<std::size_t>(coded_n), 0);
    check(ddt::inverse_block(dequant.data(), p, ctx, expected.data()),
          "levels CPU inverse");
    expect_elements(coded.data() + jobs[i].out_offset, expected.data(), coded_n,
                    static_cast<int>(sizeof(std::int16_t)),
                    "levels inverse " + std::string(txv::tx_size_name(
                                              static_cast<TxSize>(p.tx_size))) +
                        " type " + std::to_string(p.tx_type) + " depth " +
                        std::to_string(depth));
  }
}

// Arena contract: exact size, no growth across launches, one-byte-short failure.
void test_arena_contract(const std::vector<PairInfo>& pairs) {
  std::vector<DdtJob> fwd_jobs;
  std::vector<DdtJob> inv_jobs;
  int adj_total = 0;
  int coded_total = 0;
  for (const PairInfo& info : pairs) {
    if (!info.legal_ddt) continue;
    const TxParams& p = info.params;
    fwd_jobs.push_back(make_job(p, 8, adj_total, adj_total));
    inv_jobs.push_back(make_job(p, 8, adj_total, coded_total));
    adj_total += p.w * p.h;
    coded_total += p.coded_w * p.coded_h;
  }
  std::vector<std::int16_t> residual(static_cast<std::size_t>(adj_total), 7);
  std::vector<std::int32_t> dequant(static_cast<std::size_t>(adj_total), 1);
  std::vector<std::int32_t> coeff(static_cast<std::size_t>(adj_total), 0);
  std::vector<std::int32_t> levels(static_cast<std::size_t>(adj_total), 0);
  std::vector<std::int16_t> coded(static_cast<std::size_t>(coded_total), 0);

  BatchConfig fcfg;
  fcfg.direction = Direction::kForward;
  fcfg.max_blocks = static_cast<int>(fwd_jobs.size());
  fcfg.with_levels = true;
  BatchConfig icfg;
  icfg.direction = Direction::kInverse;
  icfg.max_blocks = static_cast<int>(inv_jobs.size());

  std::uint64_t jobs_bytes = 0;
  std::uint64_t input_bytes = 0;
  std::uint64_t output_bytes = 0;
  const std::uint64_t fneed =
      av2enc::cuda::ddt::required_device_bytes_split(fcfg, &jobs_bytes,
                                                     &input_bytes,
                                                     &output_bytes);
  const std::uint64_t ineed =
      av2enc::cuda::ddt::required_device_bytes(icfg);
  std::printf(
      "required_device_bytes: forward n=%zu with_levels = %llu "
      "(jobs %llu + input %llu + output %llu); inverse n=%zu = %llu\n",
      fwd_jobs.size(), static_cast<unsigned long long>(fneed),
      static_cast<unsigned long long>(jobs_bytes),
      static_cast<unsigned long long>(input_bytes),
      static_cast<unsigned long long>(output_bytes), inv_jobs.size(),
      static_cast<unsigned long long>(ineed));

  // Two launches on one exact-size arena: peak usage must not grow, and used()
  // must return to the mark taken inside each call.
  Arena arena;
  std::string error;
  check(av2enc::cuda::is_ok(arena.init(fneed, Arena::kAlignment, &error)),
        "arena exact init: " + error);
  std::uint64_t reported = 0;
  check(av2enc::cuda::ddt::forward_batch(
            arena, fwd_jobs.data(), static_cast<int>(fwd_jobs.size()),
            residual.data(), coeff.data(), levels.data(), nullptr, &reported,
            &error) == BatchStatus::kOk,
        "arena first launch: " + error);
  const std::uint64_t used_after_first = arena.used();
  const std::uint64_t peak_after_first = arena.peak_used();
  check(reported == fneed, "forward reported required bytes");
  check(used_after_first == 0, "arena used() rewound after launch");
  check(peak_after_first == fneed, "arena peak equals required bytes");

  check(av2enc::cuda::ddt::forward_batch(
            arena, fwd_jobs.data(), static_cast<int>(fwd_jobs.size()),
            residual.data(), coeff.data(), levels.data(), nullptr, &reported,
            &error) == BatchStatus::kOk,
        "arena second launch: " + error);
  check(arena.used() == used_after_first, "two launches: used() does not grow");
  check(arena.peak_used() == peak_after_first,
        "two launches: peak_used() does not grow");

  // One byte short: the allocator must fail, the module must report the exact
  // requirement, and the failed call must leave the bump pointer untouched.
  Arena small;
  check(av2enc::cuda::is_ok(small.init(fneed - 1, Arena::kAlignment, &error)),
        "arena short init: " + error);
  reported = 0;
  const BatchStatus short_status = av2enc::cuda::ddt::forward_batch(
      small, fwd_jobs.data(), static_cast<int>(fwd_jobs.size()),
      residual.data(), coeff.data(), levels.data(), nullptr, &reported,
      &error);
  check(short_status == BatchStatus::kArenaTooSmall,
        "one-byte-short arena returns kArenaTooSmall, got " +
            std::string(av2enc::cuda::ddt::batch_status_name(short_status)));
  check(reported == fneed, "one-byte-short reports the exact requirement");
  check(small.used() == 0, "one-byte-short leaves the bump pointer unchanged");

  // Same for the inverse direction.
  Arena inv_arena;
  check(av2enc::cuda::is_ok(inv_arena.init(ineed, Arena::kAlignment, &error)),
        "inverse arena exact init: " + error);
  reported = 0;
  check(av2enc::cuda::ddt::inverse_batch(
            inv_arena, inv_jobs.data(), static_cast<int>(inv_jobs.size()),
            dequant.data(), coded.data(), false, nullptr, &reported, &error) ==
            BatchStatus::kOk,
        "inverse exact launch: " + error);
  check(reported == ineed, "inverse reported required bytes");
  check(inv_arena.used() == 0, "inverse used() rewound");
  Arena inv_small;
  check(av2enc::cuda::is_ok(
            inv_small.init(ineed - 1, Arena::kAlignment, &error)),
        "inverse short init: " + error);
  reported = 0;
  const BatchStatus inv_short = av2enc::cuda::ddt::inverse_batch(
      inv_small, inv_jobs.data(), static_cast<int>(inv_jobs.size()),
      dequant.data(), coded.data(), false, nullptr, &reported, &error);
  check(inv_short == BatchStatus::kArenaTooSmall,
        "one-byte-short inverse returns kArenaTooSmall");
  check(reported == ineed, "one-byte-short inverse required bytes");

  // Invalid parameters must be rejected before any device work.
  DdtJob bad = fwd_jobs[0];
  bad.bit_depth = 12;
  check(av2enc::cuda::ddt::forward_batch(
            arena, &bad, 1, residual.data(), coeff.data(), levels.data(),
            nullptr, nullptr, &error) == BatchStatus::kInvalidArgument,
        "invalid bit_depth rejected");
  check(av2enc::cuda::ddt::inverse_batch(arena, nullptr, 1, dequant.data(),
                                         coded.data(), false, nullptr, nullptr,
                                         &error) ==
            BatchStatus::kInvalidArgument,
        "null jobs rejected");
  check(av2enc::cuda::ddt::forward_batch(arena, nullptr, 0, nullptr, nullptr,
                                         nullptr, nullptr, nullptr, &error) ==
            BatchStatus::kOk,
        "empty batch is a no-op");
}

}  // namespace

int main() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    std::printf("no CUDA device\n");
    return 1;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    std::printf("cudaGetDeviceProperties failed\n");
    return 1;
  }
  std::printf("av2enc CUDA DDT test\ndevice: %s (sm_%d%d, %d SMs)\n\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount);

  const std::vector<PairInfo> pairs = all_pairs();
  int legal = 0;
  int component = 0;
  for (const PairInfo& info : pairs) {
    if (info.legal_ddt) ++legal;
    if (info.component) ++component;
  }
  check(legal == av2enc::cuda::ddt::test::kLegalDdtPairCount,
        "100 legal (tx_size, tx_type) DDT pairs, got " + std::to_string(legal));
  std::printf("pairs: %zu total, %d legal DDT, %d with a DDT component\n",
              pairs.size(), legal, component);

  run_block_parity(pairs, 8, 0, 0, "8-bit shape0");
  run_block_parity(pairs, 8, 1, 1, "8-bit shape1");
  run_block_parity(pairs, 8, 5, 2, "8-bit random");
  run_block_parity(pairs, 10, 2, 0, "10-bit shape2");
  run_block_parity(pairs, 10, 4, 1, "10-bit shape4");

  test_32_alias(pairs);
  test_inverse_from_levels(pairs, 8);
  test_inverse_from_levels(pairs, 10);
  test_arena_contract(pairs);

  std::printf(
      "\n%d checks, %llu element comparisons, %d failures\n", g_checks,
      static_cast<unsigned long long>(g_elements), g_failures);
  std::printf("%s\n", g_failures == 0 ? "RESULT: PASS" : "RESULT: FAIL");
  return g_failures == 0 ? 0 : 1;
}
