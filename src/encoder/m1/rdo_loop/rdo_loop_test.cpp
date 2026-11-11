// Standalone self-tests for the unified M1 RDO loop policy layer.
//
// No CMake, no CUDA.  The tests pin:
//   * lambda calibration and the CTU activity clamp,
//   * cost breakdown arithmetic and deterministic node ordering,
//   * spec-derived partition sets (SPLIT is illegal at 64, legal at 128),
//   * policy depth/min-block limits,
//   * exact branch-and-bound behaviour,
//   * candidate pruning order and forced keeps,
//   * the GPU-batch reduction and invalid-record handling,
//   * the fixed-capacity cost trace,
//   * integer activity measurement.

#include "rdo_loop.h"

#include "encoder/partition/partition.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

namespace {

namespace partition = av2enc::partition;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

void check_near(double actual, double expected, double tolerance,
                const char* message) {
  if (!(std::fabs(actual - expected) <= tolerance)) {
    std::fprintf(stderr, "FAIL: %s (got %.9f expected %.9f)\n", message,
                 actual, expected);
    ++failures;
  }
}

using av2enc::m1::rdo_loop::CandidateBatch;
using av2enc::m1::rdo_loop::CandidateDomain;
using av2enc::m1::rdo_loop::CandidateRecord;
using av2enc::m1::rdo_loop::CandidateScore;
using av2enc::m1::rdo_loop::CostTerms;
using av2enc::m1::rdo_loop::CostTrace;
using av2enc::m1::rdo_loop::LambdaConfig;
using av2enc::m1::rdo_loop::NodeDecisionLoop;
using av2enc::m1::rdo_loop::NodeInfo;
using av2enc::m1::rdo_loop::NodeCost;
using av2enc::m1::rdo_loop::PruneConfig;
using av2enc::m1::rdo_loop::SearchOptions;
using partition::BlockSize;
using partition::PartitionType;

partition::FrameGeometry geometry_64() {
  partition::FrameGeometry geometry;
  geometry.width = 640;
  geometry.height = 360;
  geometry.sb_size = 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;
  return geometry;
}

partition::FrameGeometry geometry_128() {
  partition::FrameGeometry geometry = geometry_64();
  geometry.sb_size = 128;
  return geometry;
}

void test_lambda() {
  LambdaConfig config;
  config.sse_scale = 0.5;
  check(av2enc::m1::rdo_loop::lambda_config_valid(config),
        "default lambda config valid");
  // 0.5 * 66^2 = 2178.
  check_near(av2enc::m1::rdo_loop::lambda_sse(config), 2178.0, 1e-9,
             "lambda_sse QP90");
  config.sse_scale = 1.0;
  check_near(av2enc::m1::rdo_loop::lambda_sse(config), 4356.0, 1e-9,
             "lambda_sse doubles with sse_scale");

  // The joint-loop calibration is 0.0005, i.e. lambda = 2.178.
  const LambdaConfig calibrated = av2enc::m1::rdo_loop::calibrated_m1_config();
  check_near(calibrated.sse_scale, 0.0005, 1e-15,
             "calibrated joint-loop sse_scale");
  check_near(av2enc::m1::rdo_loop::lambda_sse(calibrated), 2.178, 1e-9,
             "calibrated joint-loop lambda");
  check(calibrated.rate_aware, "calibrated loop is rate-aware");
  check_near(calibrated.ctu_activity_strength, 0.0, 1e-15,
             "calibrated CTU activity is neutral");

  config.sse_scale = 0.5;
  config.ctu_activity_strength = 0.25;
  const auto low = av2enc::m1::rdo_loop::ctu_lambda(config, 100, 200);
  const auto mid = av2enc::m1::rdo_loop::ctu_lambda(config, 200, 200);
  const auto high = av2enc::m1::rdo_loop::ctu_lambda(config, 400, 200);
  check(low.scale < 1.0 && mid.scale == 1.0 && high.scale > 1.0,
        "ctu lambda activity ordering");
  check_near(low.scale, 0.875, 1e-12, "ctu low scale");
  check_near(high.scale, 1.25, 1e-12, "ctu high scale");
  const auto clipped = av2enc::m1::rdo_loop::ctu_lambda(config, 10000, 200);
  check_near(clipped.scale, 1.5, 1e-12, "ctu scale max clamp");
  check(av2enc::m1::rdo_loop::ctu_lambda(config, 1, 0).scale == 1.0,
        "ctu zero mean disables adaptation");
}

void test_cost() {
  CostTerms terms;
  terms.distortion_sse = 100.0;
  terms.mode_bits = 3.0;
  terms.coefficient_bits = 10.0;
  terms.tx_type_bits = 2.0;
  terms.signal_bits = 1.0;
  check_near(av2enc::m1::rdo_loop::total_bits(terms), 16.0, 1e-12,
             "total_bits");
  const NodeCost cost = av2enc::m1::rdo_loop::make_node_cost(terms, 2.0, true);
  check_near(cost.cost, 132.0, 1e-12, "node cost");
  const NodeCost sse_only =
      av2enc::m1::rdo_loop::make_node_cost(terms, 2.0, false);
  check_near(sse_only.cost, 100.0, 1e-12, "sse-only cost");

  NodeCost aggregate = av2enc::m1::rdo_loop::make_node_cost(
      CostTerms{}, 2.0, true, 0);
  av2enc::m1::rdo_loop::accumulate_node_cost(aggregate, cost);
  av2enc::m1::rdo_loop::add_signal_bits(aggregate, 2.0);
  check_near(aggregate.cost, 136.0, 1e-12, "aggregate signal bits");

  CostTerms bad;
  bad.mode_bits = -1.0;
  check(!av2enc::m1::rdo_loop::cost_terms_valid(bad),
        "negative bits invalid");
  bad = CostTerms{};
  bad.distortion_sse = std::numeric_limits<double>::quiet_NaN();
  check(!av2enc::m1::rdo_loop::cost_terms_valid(bad), "NaN SSE invalid");
}

void test_order() {
  CostTerms a_terms;
  a_terms.distortion_sse = 100.0;
  a_terms.mode_bits = 1.0;
  CostTerms b_terms = a_terms;
  const NodeCost a =
      av2enc::m1::rdo_loop::make_node_cost(a_terms, 10.0, true);
  const NodeCost b =
      av2enc::m1::rdo_loop::make_node_cost(b_terms, 10.0, true);
  check(!av2enc::m1::rdo_loop::node_cost_better(
            a, PartitionType::kHorz, 0, b, PartitionType::kNone, 0),
        "NONE wins an exact tie");
  check(av2enc::m1::rdo_loop::node_cost_better(
            a, PartitionType::kNone, 0, b, PartitionType::kHorz, 0),
        "NONE wins the reverse tie");
  b_terms.distortion_sse = 101.0;
  const NodeCost worse =
      av2enc::m1::rdo_loop::make_node_cost(b_terms, 10.0, true);
  check(av2enc::m1::rdo_loop::node_cost_better(
            a, PartitionType::kVert, 3, worse, PartitionType::kNone, 0),
        "lower cost wins");
}

void test_node_info() {
  const partition::FrameGeometry g64 = geometry_64();
  NodeInfo info;
  std::string error;
  check(av2enc::m1::rdo_loop::describe_node(g64, 0, 0, BlockSize::k64x64, 0,
                                            true, false, &info, &error),
        "describe 64x64 SB");
  check(!info.forced, "64x64 SB not forced");
  check(info.allowed.is_allowed(PartitionType::kNone),
        "NONE allowed at 64x64");
  check(info.allowed.is_allowed(PartitionType::kHorz), "HORZ allowed");
  check(info.allowed.is_allowed(PartitionType::kVert), "VERT allowed");
  check(!info.allowed.is_allowed(PartitionType::kSplit),
        "SPLIT illegal at 64x64");

  partition::FrameGeometry g128 = geometry_128();
  NodeInfo info128;
  check(av2enc::m1::rdo_loop::describe_node(g128, 0, 0, BlockSize::k128x128, 0,
                                            true, false, &info128, &error),
        "describe 128x128 SB");
  check(info128.allowed.is_allowed(PartitionType::kSplit),
        "SPLIT legal at 128x128");

  SearchOptions options;
  check(av2enc::m1::rdo_loop::policy_allows(options, 0, BlockSize::k64x64,
                                            PartitionType::kHorz),
        "HORZ at depth 0 allowed");
  check(!av2enc::m1::rdo_loop::policy_allows(options, 0, BlockSize::k64x64,
                                             PartitionType::kSplit),
        "SPLIT policy off");
  options.allow_split = true;
  check(av2enc::m1::rdo_loop::policy_allows(options, 0, BlockSize::k128x128,
                                            PartitionType::kSplit),
        "SPLIT at 128 allowed by policy");
  check(!av2enc::m1::rdo_loop::policy_allows(options, 4, BlockSize::k16x16,
                                             PartitionType::kHorz),
        "depth cap stops the split");

  av2enc::m1::rdo_loop::ChildRef children[4];
  const int count = av2enc::m1::rdo_loop::enumerate_node_children(
      g64, info, PartitionType::kHorz, children, &error);
  check(count == 2, "HORZ at 64x64 has two children");
  check(children[0].size == BlockSize::k64x32 &&
            children[1].size == BlockSize::k64x32,
        "HORZ child size 64x32");
  check(children[0].mi_row == 0 && children[1].mi_row == 8,
        "HORZ child rows");
  const int vert_count = av2enc::m1::rdo_loop::enumerate_node_children(
      g64, info, PartitionType::kVert, children, &error);
  check(vert_count == 2 && children[0].size == BlockSize::k32x64,
        "VERT child size 32x64");
}

void test_decision_loop() {
  const partition::FrameGeometry g64 = geometry_64();
  NodeInfo info;
  std::string error;
  check(av2enc::m1::rdo_loop::describe_node(g64, 0, 0, BlockSize::k64x64, 0,
                                            true, false, &info, &error),
        "describe node for decision loop");
  SearchOptions options;
  NodeDecisionLoop loop;
  loop.reset(info, options);
  loop.set_lambda(100.0);

  CostTerms leaf_terms;
  leaf_terms.distortion_sse = 1000.0;
  leaf_terms.coefficient_bits = 10.0;
  check(loop.add_terms(PartitionType::kNone, leaf_terms, 1, &error),
        "add NONE candidate");

  std::array<NodeCost, 2> children{};
  CostTerms child_terms;
  child_terms.distortion_sse = 200.0;
  child_terms.coefficient_bits = 5.0;
  children[0] = av2enc::m1::rdo_loop::make_node_cost(child_terms, 100.0, true,
                                                     1);
  child_terms.mode_bits = 1.0;
  children[1] = av2enc::m1::rdo_loop::make_node_cost(child_terms, 100.0, true,
                                                     1);
  check(loop.add_partition(PartitionType::kHorz, children, &error),
        "add HORZ candidate");
  // HORZ: SSE 400, bits 5+6+2 = 13 -> 400 + 1300 = 1700.
  // NONE: 1000 + 1000 = 2000.  HORZ wins.
  const auto& decision = loop.decision();
  check(decision.found, "decision found");
  check(decision.partition == PartitionType::kHorz, "HORZ wins");
  check_near(decision.cost.cost, 1700.0, 1e-9, "HORZ cost");
  check(decision.cost.leaves == 2, "HORZ leaves");

  // Branch-and-bound: a partial cost above the best cannot win.
  NodeCost partial = av2enc::m1::rdo_loop::make_node_cost(
      CostTerms{5000.0, 0.0, 0.0, 0.0, 0.0}, 100.0, true, 1);
  check(loop.cannot_beat_best(partial), "branch-and-bound prunes");
  NodeCost partial_low = av2enc::m1::rdo_loop::make_node_cost(
      CostTerms{10.0, 0.0, 0.0, 0.0, 0.0}, 100.0, true, 1);
  check(!loop.cannot_beat_best(partial_low), "branch-and-bound keeps a bound");

  CostTrace trace;
  trace.clear();
  check(av2enc::m1::rdo_loop::record_node_trace(info, decision, trace),
        "record trace");
  check(trace.size() == 1, "trace size");
  check(trace.at(0).partition == static_cast<std::uint8_t>(PartitionType::kHorz),
        "trace partition");

  // An invalid candidate must be rejected, not silently selected.
  CostTerms bad;
  bad.coefficient_bits = std::numeric_limits<double>::infinity();
  const int before = loop.decision().rejected_count;
  check(!loop.add_terms(PartitionType::kVert, bad, 1, &error),
        "reject invalid candidate");
  check(loop.decision().rejected_count == before + 1, "reject counted");
}

void test_pruning() {
  std::array<CandidateScore, 6> candidates{};
  candidates[0] = {0, 100.0};
  candidates[1] = {1, 120.0};
  candidates[2] = {2, 400.0};
  candidates[3] = {3, 90.0};
  candidates[4] = {4, 95.0};
  candidates[5] = {5, 1000.0};
  std::array<int, 6> keep{};
  std::array<int, 6> scratch{};
  PruneConfig config;
  config.max_keep = 3;
  config.sse_margin = 1.12;
  config.keep_index = 5;
  const int count = av2enc::m1::rdo_loop::prune_candidates(
      std::span<const CandidateScore>(candidates), std::span<int>(keep),
      std::span<int>(scratch), config);
  // Selected by SSE: 3 (90), 4 (95), 0 (100).  Margin 1.1 keeps 0..4 except
  // 1 (120 > 99) is out; 5 is force-kept.  Output in original order: 0,3,4,5.
  check(count == 4, "prune count");
  check(keep[0] == 0 && keep[1] == 3 && keep[2] == 4 && keep[3] == 5,
        "prune order/force keep");
}

void test_batch() {
  CandidateBatch batch;
  batch.batch_id = 7;
  batch.domain = static_cast<std::int32_t>(CandidateDomain::kLumaMode);
  batch.lambda = 10.0;
  std::string error;
  CandidateRecord a;
  a.group = 1;
  a.index = 0;
  a.domain = static_cast<std::uint8_t>(CandidateDomain::kLumaMode);
  a.distortion_sse = 200.0;
  a.coefficient_bits = 5.0;
  CandidateRecord b = a;
  b.index = 1;
  b.distortion_sse = 100.0;
  b.coefficient_bits = 20.0;  // 100 + 200 = 300 vs a = 250
  check(av2enc::m1::rdo_loop::append_candidate(batch, a, &error),
        "batch add a");
  check(av2enc::m1::rdo_loop::append_candidate(batch, b, &error),
        "batch add b");
  check(av2enc::m1::rdo_loop::finalize_batch(batch, &error),
        "batch finalize");
  check(batch.best_index == 0, "batch best is a");
  check_near(batch.best_cost, 250.0, 1e-9, "batch best cost");

  CandidateRecord bad = a;
  bad.index = 2;
  bad.coefficient_bits = -1.0;
  check(av2enc::m1::rdo_loop::append_candidate(batch, bad, &error),
        "batch add invalid record");
  check(av2enc::m1::rdo_loop::finalize_batch(batch, &error),
        "batch finalize with invalid record");
  check(batch.evaluated == 2 && batch.rejected == 1,
        "batch rejects invalid record");
  check(batch.best_index == 0, "invalid record cannot win");

  CandidateBatch empty;
  empty.domain = static_cast<std::int32_t>(CandidateDomain::kLumaMode);
  empty.lambda = 1.0;
  check(!av2enc::m1::rdo_loop::finalize_batch(empty, &error),
        "empty batch rejected");
}

void test_batch_parity() {
  const partition::FrameGeometry g64 = geometry_64();
  NodeInfo info;
  std::string error;
  check(av2enc::m1::rdo_loop::describe_node(g64, 0, 0, BlockSize::k64x64, 0,
                                            true, false, &info, &error),
        "describe node for batch parity");
  SearchOptions options;
  NodeDecisionLoop loop;
  loop.reset(info, options);
  loop.set_lambda(10.0);

  CandidateBatch batch;
  batch.batch_id = 1;
  batch.domain = static_cast<std::int32_t>(CandidateDomain::kPartition);
  batch.lambda = 10.0;

  // Four partition candidates with identical terms; the search order index is
  // the tie-break key in both selectors.
  const PartitionType partitions[4] = {
      PartitionType::kNone, PartitionType::kHorz, PartitionType::kVert,
      PartitionType::kSplit};
  for (int i = 0; i < 4; ++i) {
    CostTerms terms;
    terms.distortion_sse = 500.0;
    terms.coefficient_bits = 10.0;
    terms.signal_bits = 1.0;
    check(loop.add_terms(partitions[i], terms, 1, &error),
          "batch parity: add node candidate");
    CandidateRecord record = av2enc::m1::rdo_loop::make_candidate_record(
        1, static_cast<std::uint16_t>(i), CandidateDomain::kPartition, terms);
    record.partition = static_cast<std::uint8_t>(partitions[i]);
    check(av2enc::m1::rdo_loop::append_candidate(batch, record, &error),
          "batch parity: add batch candidate");
  }
  check(av2enc::m1::rdo_loop::finalize_batch(batch, &error),
        "batch parity: finalize");
  check(loop.decision().found, "batch parity: node decision found");
  check(batch.best_index == loop.decision().candidate_index,
        "batch and node selectors choose the same slot");
  check_near(batch.best_cost, loop.decision().cost.cost, 1e-12,
             "batch and node selectors agree on cost");
  check(loop.decision().partition == PartitionType::kNone,
        "batch parity: NONE wins the exact tie");
}

void test_activity() {
  std::array<std::uint8_t, 16> plane{};
  for (int i = 0; i < 16; ++i) {
    plane[static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>((i & 1) != 0 ? 200 : 0);
  }
  const std::uint32_t activity =
      av2enc::m1::rdo_loop::luma_activity_q8(plane.data(), 4, 0, 0, 4, 4);
  // mean 100, mean absolute deviation 100 -> Q8 25600.
  check(activity == 25600, "activity Q8");
  std::array<std::uint32_t, 3> values{100, 200, 300};
  check(av2enc::m1::rdo_loop::mean_activity_q8(values.data(), 3) == 200,
        "mean activity");
  check(av2enc::m1::rdo_loop::luma_activity_q8(nullptr, 4, 0, 0, 4, 4) == 0,
        "null activity");
}

}  // namespace

int main() {
  test_lambda();
  test_cost();
  test_order();
  test_node_info();
  test_decision_loop();
  test_pruning();
  test_batch();
  test_batch_parity();
  test_activity();
  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("rdo_loop_test: all tests passed\n");
  return EXIT_SUCCESS;
}
