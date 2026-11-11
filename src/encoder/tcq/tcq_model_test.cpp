// Tests for the TCQ-aware §9.3 coefficient rate model.

#include "tcq_av2_model.h"

#include "encoder/tx_variants/tx_variants.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

void expect(bool condition, const char* what) {
  ++checks;
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

av2enc::tcq::av2::BlockInfo make_info() {
  av2enc::tcq::av2::BlockInfo info;
  const auto params = av2enc::txv::make_tx_params(
      8, 8, av2enc::txv::TxType::kDctDct);
  expect(params.has_value(), "8x8 params exist");
  if (params.has_value()) info.params = *params;
  info.plane = 0;
  info.qctx = 0;
  info.txb_skip_context = 0;
  info.dc_sign_context = 0;
  info.is_inter = false;
  info.tx_class_2d = true;
  return info;
}

void test_tcq_cdf_selector() {
  const av2enc::tcq::av2::BlockInfo info = make_info();
  std::vector<std::int32_t> reference(64, 0);
  // A few reference levels so the contexts are not all zero.
  reference[0] = 3;
  reference[1] = 2;
  reference[8] = 1;
  av2enc::tcq::av2::Model model;
  std::string error;
  expect(model.init(info, reference, &error), "model init");
  av2enc::tcq::RateModel rate = model.rate_model();
  expect(rate.coefficient_rate != nullptr && rate.eob_rate != nullptr,
         "rate model callbacks");

  bool saw_state_difference = false;
  for (int pos = 0; pos < 64; ++pos) {
    const double bits0 = rate.coefficient_rate(
        rate.context, pos, /*state=*/0, /*base_level=*/1,
        /*next_state=*/4, /*signed_quant=*/2, 0, 0);
    const double bits2 = rate.coefficient_rate(
        rate.context, pos, /*state=*/2, /*base_level=*/1,
        /*next_state=*/1, /*signed_quant=*/2, 0, 0);
    expect(std::isfinite(bits0) && std::isfinite(bits2),
           "coefficient costs are finite");
    expect(bits0 >= 0.0 && bits2 >= 0.0, "coefficient costs are non-negative");
    if (bits0 != bits2) saw_state_difference = true;
  }
  expect(saw_state_difference,
         "TCQ state selects a different coeff_base CDF");

  const double skip0 = rate.eob_rate(rate.context, 0, 64);
  const double skip1 = rate.eob_rate(rate.context, 1, 64);
  const double skip8 = rate.eob_rate(rate.context, 8, 64);
  expect(std::isfinite(skip0) && std::isfinite(skip1) &&
             std::isfinite(skip8),
         "EOB costs are finite");
  expect(skip1 != skip8, "EOB prefix changes the cost");

  expect(std::isfinite(model.reference_bits()), "reference bits are finite");
  expect(model.reference_bits() > 0.0, "reference bits are positive");
  expect(model.reference_eob() == 9, "reference EOB from scan position 8");
}

void test_rejections() {
  std::vector<std::int32_t> reference(64, 0);
  std::string error;
  av2enc::tcq::av2::BlockInfo info = make_info();
  info.plane = 1;
  av2enc::tcq::av2::Model model;
  expect(!model.init(info, reference, &error), "rejects chroma");

  info = make_info();
  info.tx_class_2d = false;
  expect(!model.init(info, reference, &error), "rejects non-2D class");

  info = make_info();
  std::vector<std::int32_t> wrong(63, 0);
  expect(!model.init(info, wrong, &error), "rejects wrong block size");

  info = make_info();
  info.txb_skip_context = 10;
  expect(!model.init(info, reference, &error), "rejects bad txb_skip context");

  info = make_info();
  info.dc_sign_context = 3;
  expect(!model.init(info, reference, &error), "rejects bad DC context");
}

void test_large_level_cost() {
  const av2enc::tcq::av2::BlockInfo info = make_info();
  std::vector<std::int32_t> reference(64, 0);
  reference[0] = 10;
  av2enc::tcq::av2::Model model;
  std::string error;
  expect(model.init(info, reference, &error), "large-level model init");
  av2enc::tcq::RateModel rate = model.rate_model();
  // HF position 1 has pre_max 6.  Level 9 uses syntax 5 + x*2, level 10 uses
  // syntax 6 + x*2; both must have finite escape costs.
  const double bits9 = rate.coefficient_rate(
      rate.context, 1, 0, 9, 0, 18, 0, 0);
  const double bits10 = rate.coefficient_rate(
      rate.context, 1, 0, 10, 0, 20, 0, 0);
  expect(std::isfinite(bits9) && bits9 > 0.0,
         "odd large TCQ escape cost is finite");
  expect(std::isfinite(bits10) && bits10 > 0.0,
         "even large TCQ escape cost is finite");
}

}  // namespace

int main() {
  test_tcq_cdf_selector();
  test_rejections();
  test_large_level_cost();

  std::printf("av2enc TCQ rate-model test\n%d checks, %d failures\n", checks,
              failures);
  if (failures != 0) {
    std::printf("RESULT: FAIL\n");
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
