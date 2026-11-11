// Self-tests for the M1 RDO decision layer.
//
// Build and run with ./build_and_test.sh.  No external test framework and no
// bitstream dependency: this checks the selectors, the stream adapter and the
// RDOQ trellis in isolation.

#include "rdo_decide.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL: %s\n", what);
  }
}

void check_near(double a, double b, double tolerance, const char* what) {
  ++g_checks;
  if (!(std::fabs(a - b) <= tolerance)) {
    ++g_failures;
    std::printf("FAIL: %s (%.6f vs %.6f)\n", what, a, b);
  }
}

using av2enc::m1::rdo_decide::ChromaCandidateCost;
using av2enc::m1::rdo_decide::ChromaRdoConfig;
using av2enc::m1::rdo_decide::ChromaChoice;
using av2enc::m1::rdo_decide::LambdaConfig;
using av2enc::m1::rdo_decide::PartitionCostInputs;
using av2enc::m1::rdo_decide::PartitionDecideConfig;
using av2enc::m1::rdo_decide::RdoqConfig;
using av2enc::m1::rdo_decide::RdoqDecision;
using av2enc::m1::rdo_decide::RdoqWorkspace;
using av2enc::m1::rdo_decide::TransformStreamBlock;
using av2enc::m1::rdo_decide::TransformStreamRate;
using av2enc::m1::rdo_decide::TxTypeDecision;
using av2enc::m1::rdo_decide::TxTypeDecisionConfig;

double override_coefficient(void*, std::int32_t, std::int32_t, bool) {
  return 2.0;
}
double override_eob(void*, std::int32_t) { return 1.0; }

bool intra_rate_hook(void* context, const av2enc::rdo::IntraRateRequest& request,
                     av2enc::rdo::IntraRateEstimate& out) {
  const auto* seen = static_cast<const int*>(context);
  if (seen != nullptr && (request.block_w != *seen || request.block_h != 7)) {
    return false;
  }
  out.syntax_rate_bits = 1.0;
  out.coefficient_rate_bits = 100.0;
  return true;
}

void test_lambda() {
  LambdaConfig config;
  config.quant_step = 66.0;
  config.sse_scale = 0.125;
  check_near(av2enc::m1::rdo_decide::lambda_sse(config), 544.5, 1e-9,
             "lambda(66, 0.125)");

  LambdaConfig forward = config;
  forward.quant_step = 528.0;
  check_near(av2enc::m1::rdo_decide::lambda_sse(forward), 34848.0, 1e-6,
             "lambda(528, 0.125) is 64x the dequant-step lambda");

  check_near(av2enc::m1::rdo_decide::rdoq_lambda_from_pixel_lambda(544.5),
             8.5078125, 1e-9, "RDOQ lambda conversion");

  std::string error;
  LambdaConfig bad;
  bad.quant_step = 0.0;
  check(!av2enc::m1::rdo_decide::validate_lambda_config(bad, &error),
        "zero quant step rejected");
}

void test_transform_stream() {
  TransformStreamRate rate;
  std::vector<std::int32_t> levels(16, 0);
  TransformStreamBlock block;
  block.plane = 0;
  block.tx_size = 0;
  block.w = 4;
  block.h = 4;
  block.levels_scan = levels;
  block.eob = 0;
  std::string error;
  check(rate.init(block, &error), "empty 4x4 stream init");
  const double zero_bits = rate.block_bits();
  check(std::isfinite(zero_bits) && zero_bits > 0.0,
        "empty stream has txb_skip bits");

  levels[0] = 5;
  block.eob = 1;
  check(rate.init(block, &error), "DC-only 4x4 stream init");
  const double dc_bits = rate.block_bits();
  check(std::isfinite(dc_bits) && dc_bits > zero_bits,
        "DC-only stream costs more than skip");
  check(rate.coefficient_bits(0, 5, true) > 0.0, "EOB coefficient bits");

  // Determinism: the same input must produce the same bits after re-init.
  check(rate.init(block, &error), "stream re-init");
  check_near(rate.block_bits(), dc_bits, 0.0, "stream determinism");

  // Pluggable override: one EOB symbol plus two bits per coefficient.
  rate.set_rate_override(&override_coefficient, &override_eob, nullptr);
  check_near(rate.block_bits(), 3.0, 1e-12, "stream rate override");
  rate.clear_rate_override();
  check_near(rate.block_bits(), dc_bits, 0.0, "stream override cleared");

  // Out-of-range inputs are rejected.
  block.eob = 17;
  check(!rate.init(block, &error), "eob beyond area rejected");
  block.eob = 1;
  block.txb_skip_context = 99;
  check(!rate.init(block, &error), "bad txb_skip context rejected");
}

void test_intra() {
  using av2enc::rdo::IntraCandidateCost;
  using av2enc::m1::rdo_decide::decide_luma_intra;

  IntraCandidateCost candidates[2];
  candidates[0].mode = 0;
  candidates[0].angle_delta = 0;
  candidates[0].sse = 100.0;
  candidates[0].syntax_rate_bits = 1.0;
  candidates[0].coefficient_rate_bits = 1.0;
  candidates[1].mode = 1;
  candidates[1].angle_delta = 0;
  candidates[1].sse = 90.0;
  candidates[1].syntax_rate_bits = 4.0;
  candidates[1].coefficient_rate_bits = 6.0;

  LambdaConfig sse_only;
  sse_only.rate_aware = false;
  LambdaConfig rate_aware;
  rate_aware.quant_step = 66.0;
  rate_aware.sse_scale = 10.0; // large enough that 8 bits outweigh 10 SSE

  std::string error;
  const auto sse_choice = decide_luma_intra(candidates, {}, sse_only, nullptr,
                                            nullptr, &error);
  check(sse_choice.found && sse_choice.mode == 1,
        "SSE-only intra picks lower SSE");

  const auto rate_choice = decide_luma_intra(candidates, {}, rate_aware, nullptr,
                                             nullptr, &error);
  check(rate_choice.found && rate_choice.mode == 0,
        "rate-aware intra picks lower rate");

  // Equal total cost: lower SSE wins first.  lambda = 43560, so a 50-SSE
  // advantage exactly balances a 0.0011475-bit rate disadvantage.
  IntraCandidateCost tie[2] = {candidates[0], candidates[1]};
  tie[0].sse = 50.0;
  tie[0].syntax_rate_bits = 2.0011475;
  tie[0].coefficient_rate_bits = 0.0;
  tie[1].sse = 100.0;
  tie[1].syntax_rate_bits = 2.0;
  tie[1].coefficient_rate_bits = 0.0;
  LambdaConfig unit;
  unit.quant_step = 66.0;
  unit.sse_scale = 10.0; // lambda 43560: both costs 435600 + sse
  const auto tie_choice = decide_luma_intra(tie, {}, unit, nullptr, nullptr,
                                            &error);
  check(tie_choice.found && tie_choice.input_index == 0,
        "intra tie-break prefers lower SSE");

  // Pluggable estimator path with block geometry.
  IntraCandidateCost est[2] = {candidates[0], candidates[1]};
  IntraCandidateCost scratch[2];
  av2enc::m1::rdo_decide::IntraBlockGeometry geometry;
  geometry.block_w = 8;
  geometry.block_h = 7;
  int expected_w = 8;
  const auto est_choice = decide_luma_intra(
      est, scratch, unit, &intra_rate_hook, &expected_w, &error, geometry);
  check(est_choice.found, "intra estimator path returns a choice");
  check(est_choice.rate_bits == 101.0, "intra estimator rates are applied");
}

void test_chroma() {
  ChromaCandidateCost candidates[2];
  candidates[0].is_cfl = false;
  candidates[0].mode = 0;
  candidates[0].block_w = 8;
  candidates[0].block_h = 8;
  candidates[0].sse_u = 100.0;
  candidates[0].sse_v = 100.0;
  candidates[0].syntax_rate_bits = 1.0;
  candidates[0].coefficient_rate_bits = 1.0;
  candidates[1].is_cfl = true;
  candidates[1].block_w = 8;
  candidates[1].block_h = 8;
  candidates[1].sse_u = 80.0;
  candidates[1].sse_v = 80.0;
  candidates[1].syntax_rate_bits = 5.0;
  candidates[1].coefficient_rate_bits = 5.0;

  LambdaConfig sse_only;
  sse_only.rate_aware = false;
  LambdaConfig rate_aware;
  rate_aware.quant_step = 66.0;
  rate_aware.sse_scale = 20.0;

  std::string error;
  const auto sse_choice = av2enc::m1::rdo_decide::decide_chroma(
      candidates, {}, sse_only, nullptr, nullptr, &error);
  check(sse_choice.found && sse_choice.is_cfl,
        "SSE-only chroma picks lower U+V SSE");

  const auto rate_choice = av2enc::m1::rdo_decide::decide_chroma(
      candidates, {}, rate_aware, nullptr, nullptr, &error);
  check(rate_choice.found && !rate_choice.is_cfl,
        "rate-aware chroma picks DC");

  // U+V score: a U-only winner that is worse for V must lose to DC.
  ChromaCandidateCost planes[2] = {candidates[0], candidates[0]};
  planes[0].is_cfl = false;
  planes[0].sse_u = 100.0;
  planes[0].sse_v = 100.0;
  planes[1].is_cfl = true;
  planes[1].sse_u = 50.0;
  planes[1].sse_v = 190.0;
  const auto plane_choice = av2enc::m1::rdo_decide::decide_chroma(
      planes, {}, sse_only, nullptr, nullptr, &error);
  check(plane_choice.found && !plane_choice.is_cfl,
        "U+V score rejects a U-only win");

  // Non-CfL with a non-zero angle on a non-directional mode is invalid.
  ChromaCandidateCost invalid = candidates[0];
  invalid.angle_delta = 1;
  const auto invalid_choice = av2enc::m1::rdo_decide::decide_chroma(
      std::span<const ChromaCandidateCost>(&invalid, 1), {}, sse_only, nullptr,
      nullptr, &error);
  check(!invalid_choice.found, "invalid chroma angle rejected");
}

void test_partition() {
  PartitionCostInputs inputs;
  inputs.leaf_distortion = 100.0;
  inputs.leaf_rate_bits = 10.0;
  inputs.split_signal_bits = 2.0;

  PartitionDecideConfig config;
  config.lambda = 5.0;
  config.rate_aware = true;
  const auto rate_costs =
      av2enc::m1::rdo_decide::lagrangian_node_costs(inputs, config);
  check_near(rate_costs.leaf, 150.0, 1e-9, "partition leaf cost");
  check_near(rate_costs.split_signal, 10.0, 1e-9, "partition split cost");

  config.rate_aware = false;
  config.sse_only_split_cost = 7.0;
  const auto sse_costs =
      av2enc::m1::rdo_decide::lagrangian_node_costs(inputs, config);
  check_near(sse_costs.leaf, 100.0, 1e-9, "partition SSE-only leaf");
  check_near(sse_costs.split_signal, 7.0, 1e-9,
             "partition SSE-only signal");

  av2enc::partition::FrameGeometry geometry;
  geometry.width = 64;
  geometry.height = 64;
  geometry.sb_size = 64;
  av2enc::partition::DecisionOptions options;
  options.min_block_size_px = 8;
  options.max_depth = 4;
  const auto raw = [](const av2enc::partition::PartitionNodeInfo& info) {
    PartitionCostInputs out;
    const int w = av2enc::partition::block_width(info.size);
    const int h = av2enc::partition::block_height(info.size);
    out.leaf_distortion = static_cast<double>(w * h) * 2.0;
    out.leaf_rate_bits = 1.0;
    out.split_signal_bits = 0.5;
    out.horz_signal_bits = 0.5;
    out.vert_signal_bits = 0.5;
    return out;
  };
  PartitionDecideConfig frame_config;
  frame_config.lambda = 1.0;
  frame_config.rate_aware = true;
  std::string error;
  const auto decision = av2enc::m1::rdo_decide::decide_frame_rdo(
      geometry, raw, options, frame_config, &error);
  check(decision.status == av2enc::partition::DecisionStatus::kOk,
        "partition frame decision succeeds");
  check(decision.leaves >= 1, "partition decision has leaves");
  check(av2enc::partition::validate_frame(geometry, decision.frame, error),
        "partition tree validates");
}

void test_tx_type() {
  constexpr int kCodedW = 8;
  constexpr int kCodedH = 8;
  std::vector<std::int16_t> residual(kCodedW * kCodedH);
  for (int y = 0; y < kCodedH; ++y) {
    for (int x = 0; x < kCodedW; ++x) {
      residual[static_cast<std::size_t>(y * kCodedW + x)] =
          static_cast<std::int16_t>(((x * 3 + y * 5) % 11) - 5);
    }
  }
  std::vector<std::int32_t> levels_scratch(64);
  std::vector<std::int16_t> recon_scratch(kCodedW * kCodedH);
  std::vector<std::int32_t> levels(64);
  std::vector<std::int16_t> recon(kCodedW * kCodedH);
  TransformStreamRate stream;
  TxTypeDecisionConfig config;
  config.lambda = 544.5;
  config.rate_aware = true;

  TxTypeDecision first;
  std::string error;
  check(av2enc::m1::rdo_decide::decide_tx_type(
            residual.data(), kCodedW, kCodedH, 0, 0, false, config,
            levels_scratch, recon_scratch, levels, recon, stream, first,
            &error),
        "TX-type decision succeeds");
  check(first.valid && first.candidates_considered > 0,
        "TX-type decision considered candidates");
  check(first.eob >= 0 && first.eob <= 64, "TX-type EOB in range");
  {
    av2enc::txv::Tx1dType row_type = av2enc::txv::Tx1dType::kDct;
    av2enc::txv::Tx1dType col_type = av2enc::txv::Tx1dType::kDct;
    av2enc::txv::tx_1d_types(first.candidate.type, &row_type, &col_type);
    check(row_type != av2enc::txv::Tx1dType::kIdt &&
              col_type != av2enc::txv::Tx1dType::kIdt,
          "M1 TX-type decision excludes identity transforms");
  }

  TxTypeDecision second;
  check(av2enc::m1::rdo_decide::decide_tx_type(
            residual.data(), kCodedW, kCodedH, 0, 0, false, config,
            levels_scratch, recon_scratch, levels, recon, stream, second,
            &error),
        "TX-type decision repeats");
  check(second.candidate.type == first.candidate.type &&
            second.eob == first.eob && second.cost == first.cost,
        "TX-type decision is deterministic");

  TxTypeDecision dct_only;
  check(av2enc::m1::rdo_decide::decide_tx_type(
            residual.data(), kCodedW, kCodedH, 0, 0, true, config,
            levels_scratch, recon_scratch, levels, recon, stream, dct_only,
            &error),
        "DCT-only TX decision succeeds");
  check(dct_only.candidate.type == av2enc::txtype::TxType::kDctDct,
        "DCT-only returns DCT_DCT");

  // 32x32 is a DCT-only set.
  std::vector<std::int16_t> big(32 * 32, 7);
  std::vector<std::int32_t> big_levels(32 * 32);
  std::vector<std::int16_t> big_recon(32 * 32);
  std::vector<std::int32_t> big_out(32 * 32);
  std::vector<std::int16_t> big_recon_out(32 * 32);
  TxTypeDecision big_choice;
  check(av2enc::m1::rdo_decide::decide_tx_type(
            big.data(), 32, 32, 0, 0, false, config, big_levels, big_recon,
            big_out, big_recon_out, stream, big_choice, &error),
        "32x32 TX decision succeeds");
  check(big_choice.candidate.type == av2enc::txtype::TxType::kDctDct,
        "32x32 set is DCT-only");
}

void test_rdoq() {
  constexpr int kW = 4;
  constexpr int kH = 4;
  std::vector<double> original(kW * kH, 0.0);
  original[0] = 40.0;
  original[1] = -17.0;
  std::vector<std::int32_t> scalar(kW * kH, 0);
  for (int i = 0; i < kW * kH; ++i) {
    scalar[static_cast<std::size_t>(i)] =
        static_cast<std::int32_t>(std::lround(original[static_cast<std::size_t>(i)] /
                                              8.25));
  }
  std::vector<std::int32_t> levels(kW * kH, 0);
  TransformStreamRate stream;
  RdoqWorkspace workspace;
  std::string error;

  RdoqConfig zero_lambda;
  zero_lambda.lambda = 0.0;
  zero_lambda.quant_step = 8.25;
  RdoqDecision low;
  check(av2enc::m1::rdo_decide::decide_rdoq(
            original, scalar, 0, 0, kW, kH, 0, 0, zero_lambda, workspace,
            stream, levels, low, &error),
        "RDOQ zero-lambda runs");
  check(low.valid && low.eob == 2, "RDOQ zero-lambda keeps two coefficients");
  check(levels[0] == 5 && levels[1] == -2,
        "RDOQ zero-lambda recovers scalar levels");

  RdoqConfig huge_lambda;
  huge_lambda.lambda = 1.0e9;
  huge_lambda.quant_step = 8.25;
  RdoqDecision high;
  check(av2enc::m1::rdo_decide::decide_rdoq(
            original, scalar, 0, 0, kW, kH, 0, 0, huge_lambda, workspace,
            stream, levels, high, &error),
        "RDOQ huge-lambda runs");
  check(high.valid && high.eob == 0, "RDOQ huge lambda yields all-zero block");
  for (int i = 0; i < kW * kH; ++i) {
    check(levels[static_cast<std::size_t>(i)] == 0,
          "RDOQ huge lambda zeroes all levels");
  }

  // Determinism for a non-degenerate lambda.
  RdoqConfig mid;
  mid.lambda = 8.5078125;
  mid.quant_step = 8.25;
  RdoqDecision a;
  RdoqDecision b;
  std::vector<std::int32_t> levels_b(kW * kH, 0);
  check(av2enc::m1::rdo_decide::decide_rdoq(
            original, scalar, 0, 0, kW, kH, 0, 0, mid, workspace, stream,
            levels, a, &error),
        "RDOQ mid-lambda runs");
  check(av2enc::m1::rdo_decide::decide_rdoq(
            original, scalar, 0, 0, kW, kH, 0, 0, mid, workspace, stream,
            levels_b, b, &error),
        "RDOQ mid-lambda repeats");
  check(a.eob == b.eob && a.cost == b.cost && levels == levels_b,
        "RDOQ is deterministic");

  RdoqConfig invalid;
  invalid.lambda = -1.0;
  RdoqDecision rejected;
  check(!av2enc::m1::rdo_decide::decide_rdoq(
            original, scalar, 0, 0, kW, kH, 0, 0, invalid, workspace, stream,
            levels, rejected, &error),
        "negative RDOQ lambda rejected");
}

}  // namespace

int main() {
  test_lambda();
  test_transform_stream();
  test_intra();
  test_chroma();
  test_partition();
  test_tx_type();
  test_rdoq();
  if (g_failures != 0) {
    std::printf("%d checks, %d failures\nRESULT: FAIL\n", g_checks,
                g_failures);
    return 1;
  }
  std::printf("%d checks, 0 failures\nRESULT: PASS\n", g_checks);
  return 0;
}
