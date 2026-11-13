// Host-compiler parity test for the device core of the §7.14.4 delta-Q
// quantizer.  Needs no GPU and no nvcc: it compiles the exact shared device
// header and compares it with the CPU reference.
//
//   ./build_and_test.sh

#include "cuda/codec/transform/delta_q/deltaq_device.cuh"
#include "cuda/codec/transform/delta_q/deltaq_test_common.h"

#include <cstdio>
#include <cstdlib>

namespace dq = av2enc::cuda::deltaq;
namespace core = av2enc::cuda::deltaq::core;
namespace test = av2enc::cuda::deltaq::test;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what, int index) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 20) {
      std::fprintf(stderr, "FAIL: %s (case %d)\n", what, index);
    }
  }
}

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

int run_cases() {
  const std::vector<test::Case> cases = test::make_cases();
  std::printf("delta-q emulation: %zu cases\n", cases.size());
  int compared = 0;
  int rejected = 0;
  int lossless = 0;
  int qm = 0;
  int idtx = 0;
  int tcq_shift = 0;

  for (std::size_t i = 0; i < cases.size(); ++i) {
    const test::Case& c = cases[i];
    const int index = static_cast<int>(i);
    const bool valid = core::valid_job(c.job);
    int dc_quant = 0;
    int ac_quant = 0;
    core::plane_quant(c.job, &dc_quant, &ac_quant);
    const std::int32_t* q2 =
        c.q2.empty() ? nullptr : c.q2.data();
    const bool ready =
        valid && core::q2_ready(c.job, q2, dc_quant, ac_quant);

    if (c.expect_reject) {
      check(!ready, "invalid job rejected", index);
      ++rejected;
      continue;
    }
    check(ready, "valid job accepted", index);
    if (!ready) continue;

    if (c.job.lossless != 0) ++lossless;
    if (c.job.q2_offset >= 0) ++qm;
    if (c.job.tx_type == 9) ++idtx;
    if (core::extra_tcq_shift(c.job) != 0) ++tcq_shift;

    const std::size_t count = c.expected_levels.size();
    std::vector<std::int32_t> levels(count, 0);
    std::vector<std::int32_t> dequant(count, 0);
    core::quantize_job(c.job, c.coeff.data(), levels.data(), q2);
    core::dequantize_job(c.job, levels.data(), dequant.data(), q2);

    for (std::size_t k = 0; k < count; ++k) {
      check_eq(levels[k], c.expected_levels[k], "level", index);
      check_eq(dequant[k], c.expected_dequant[k], "dequant", index);
    }
    ++compared;
  }

  std::printf(
      "  compared %d blocks (%d rejected, %d lossless, %d QM, %d IDTX, "
      "%d TCQ-shift)\n",
      compared, rejected, lossless, qm, idtx, tcq_shift);
  return compared;
}

void run_quantizer_table() {
  // §7.14.2 get_q() over the full legal qindex/delta grid.
  for (int qindex = 0; qindex <= 255; ++qindex) {
    for (int delta = -87; delta <= 71; ++delta) {
      check_eq(core::get_q(qindex, delta),
               av2enc::dcdeltaq::get_q(qindex, delta), "get_q", qindex);
    }
  }
  for (int q = -1; q < 400; ++q) {
    check_eq(core::qlookup(q), av2enc::dcdeltaq::qlookup(q), "qlookup", q);
  }
}

void run_shift_rules() {
  dq::Job job;
  job.plane = 0;
  job.coded_w = 32;
  job.coded_h = 32;
  job.tx_type = 9;  // IDTX is TX_CLASS_2D
  for (int tcq = 0; tcq < 2; ++tcq) {
    for (int fsc = 0; fsc < 2; ++fsc) {
      for (int lossless = 0; lossless < 2; ++lossless) {
        for (int plane = 0; plane < 3; ++plane) {
          for (int type = 0; type < 16; ++type) {
            job.allow_tcq = tcq;
            job.use_fsc = fsc;
            job.lossless = lossless;
            job.plane = plane;
            job.tx_type = type;
            av2enc::dcdeltaq::QuantBlock block;
            block.plane = plane;
            block.coded_w = 32;
            block.coded_h = 32;
            block.tx_type = type;
            block.allow_tcq = tcq != 0;
            block.use_fsc = fsc != 0;
            block.lossless = lossless != 0;
            check_eq(core::dequant_shift(job),
                     av2enc::dcdeltaq::dequant_shift(block), "dequant_shift",
                     type);
          }
        }
      }
    }
  }
}

}  // namespace

int main() {
  run_cases();
  run_quantizer_table();
  run_shift_rules();
  std::printf("delta-q emulation: %d checks, %d failures\n", g_checks,
              g_failures);
  if (g_failures != 0) {
    std::printf("RESULT: FAIL\n");
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
