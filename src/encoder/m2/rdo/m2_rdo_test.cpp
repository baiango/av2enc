// Self-tests for the M2 P-frame rate-aware decision layer.
//
// These tests exercise the cost conventions, the default §9.3 rate model,
// deterministic tie-breaks, the residual skip wrapper, the intra coefficient
// adapter and the partition solver wrapper.  They do not encode a stream; the
// decoder gate lives in experiment/run_m2_rdo_experiment.sh.

#include "encoder/m2/rdo/rdo.h"

#include "encoder/partition/partition.h"
#include "encoder/tx_variants/tx_variants.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace {

using av2enc::m2::rdo::InterCandidate;
using av2enc::m2::rdo::InterCandidateKind;
using av2enc::m2::rdo::InterCandidateSpec;
using av2enc::m2::rdo::InterRateModel;
using av2enc::m2::rdo::RateCallbacks;
using av2enc::m2::rdo::RateContext;
using av2enc::m2::rdo::RdoConfig;
using av2enc::m2::rdo::ScoredInterCandidate;

int failures = 0;

void check(bool condition, const std::string& what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

bool close(double a, double b, double tolerance = 1e-9) {
  return std::fabs(a - b) <= tolerance;
}

av2enc::m2::residual::PlaneView view_of(
    const std::vector<std::uint8_t>& data, int stride) {
  return av2enc::m2::residual::PlaneView{
      std::span<const std::uint8_t>(data.data(), data.size()), stride};
}

struct TestBlock {
  TestBlock() = default;
  TestBlock(const TestBlock&) = delete;
  TestBlock& operator=(const TestBlock&) = delete;
  TestBlock(TestBlock&&) = default;
  TestBlock& operator=(TestBlock&&) = delete;

  std::vector<std::uint8_t> source[3];
  std::vector<std::uint8_t> prediction[3];
  av2enc::m2::residual::BlockInput input;
};

TestBlock make_block(int w, int h, int src_value, int pred_value) {
  const int cw = w / 2;
  const int ch = h / 2;
  TestBlock block;
  block.source[0].assign(
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h),
      static_cast<std::uint8_t>(src_value));
  block.source[1].assign(
      static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch),
      static_cast<std::uint8_t>(src_value));
  block.source[2] = block.source[1];
  block.prediction[0].assign(
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h),
      static_cast<std::uint8_t>(pred_value));
  block.prediction[1].assign(
      static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch),
      static_cast<std::uint8_t>(pred_value));
  block.prediction[2] = block.prediction[1];

  block.input.geometry =
      av2enc::m2::residual::BlockGeometry{0, 0, w, h, cw, ch};
  block.input.source[0] = view_of(block.source[0], w);
  block.input.source[1] = view_of(block.source[1], cw);
  block.input.source[2] = view_of(block.source[2], cw);
  block.input.prediction[0] = view_of(block.prediction[0], w);
  block.input.prediction[1] = view_of(block.prediction[1], cw);
  block.input.prediction[2] = view_of(block.prediction[2], cw);
  return block;
}

double fake_mode_bits(void* context, const av2enc::m2::rdo::ModeRateInput&) {
  const double value = *static_cast<const double*>(context);
  return value;
}

double fake_residual_bits(
    void*, const av2enc::m2::residual::BlockDecision&) {
  return 0.0;
}

double fake_intra_syntax_bits(void*, int, int, int, int) { return 3.0; }

double fake_ref_bits(void*, int) { return 2.0; }

void test_config() {
  RdoConfig config = av2enc::m2::rdo::calibrated_m2_config();
  check(config.rate_aware, "calibrated config is rate aware");
  check(config.metric == av2enc::rdo::DistortionMetric::kSsd,
        "calibrated config uses SSE");
  check(close(config.lambda.quant_step, 66.0), "calibrated step is 66");
  check(close(config.lambda.sse_scale, 0.5), "calibrated sse_scale is 0.5");
  check(close(av2enc::m2::rdo::lambda_sse(config), 2178.0),
        "lambda_sse is 2178");
  check(close(av2enc::m2::rdo::lambda_for_metric(config), 2178.0),
        "SSE lambda is 2178");
  config.metric = av2enc::rdo::DistortionMetric::kSad;
  check(close(av2enc::m2::rdo::lambda_for_metric(config), std::sqrt(2178.0)),
        "SAD lambda is sqrt(lambda_sse)");
  config.metric = av2enc::rdo::DistortionMetric::kSatd;
  check(close(av2enc::m2::rdo::lambda_for_metric(config), std::sqrt(2178.0)),
        "SATD lambda is sqrt(lambda_sse)");
  config.metric = av2enc::rdo::DistortionMetric::kSsd;
  config.rate_aware = false;
  check(close(av2enc::m2::rdo::selector_config(config).lambda, 0.0),
        "rate_aware=false gives selector lambda 0");
  check(av2enc::m2::rdo::config_valid(config), "config is valid");
}

void test_rate_model() {
  const InterRateModel model;
  check(close(model.default_mv_bits(0, 0), model.default_mv_bits(0, 0)),
        "zero MVD is deterministic");
  check(model.default_mv_bits(8, 0) > model.default_mv_bits(0, 0),
        "8/8 MVD costs more than zero MVD");
  check(model.default_mv_bits(64, 0) > model.default_mv_bits(8, 0),
        "larger MVD costs more");
  check(close(model.default_mv_bits(-8, 0), model.default_mv_bits(8, 0)),
        "MVD shell cost is sign-symmetric");
  check(model.default_drl_bits(0, 1, 5) > model.default_drl_bits(0, 0, 5),
        "DRL index 1 costs more than index 0");
  check(model.default_single_mode_bits(0, 0) > 0.0,
        "single_mode NEARMV has a positive cost");
  check(model.default_single_mode_bits(0, 1) > 0.0,
        "single_mode GLOBALMV has a positive cost");
  check(model.default_single_mode_bits(0, 2) > 0.0,
        "single_mode NEWMV has a positive cost");
  check(model.default_skip_flag_bits(0, true) > 0.0,
        "skip_flag has a positive cost");

  RateContext context;
  av2enc::m2::rdo::ModeRateInput near;
  near.inter_mode = 1;
  near.drl_index = 0;
  near.skip_flag = true;
  near.context = context;
  av2enc::m2::rdo::ModeRateInput new_mv = near;
  new_mv.inter_mode = 2;
  new_mv.mvd = av2enc::inter::mc::Mv{32, -16};
  check(model.mode_bits(new_mv) > model.mode_bits(near),
        "NEWMV with MVD costs more than NEARMV");
  av2enc::m2::rdo::ModeRateInput residual = near;
  residual.skip_flag = false;
  check(!close(model.mode_bits(residual), model.mode_bits(near)),
        "skip_flag changes the mode rate");

  double forced = 7.5;
  RateCallbacks callbacks;
  callbacks.context = &forced;
  callbacks.mode_bits = fake_mode_bits;
  const InterRateModel overridden(callbacks);
  check(close(overridden.mode_bits(near), 7.5),
        "mode_bits callback overrides the default model");
  check(model.intra_syntax_bits(0, 0, 0, 0) > 0.0,
        "intra syntax has a positive cost");

  RateCallbacks ref_callbacks;
  ref_callbacks.ref_bits = fake_ref_bits;
  const InterRateModel with_ref(ref_callbacks);
  check(close(with_ref.mode_bits(near) - model.mode_bits(near), 2.0),
        "ref_bits callback adds to the default mode rate");

  RateCallbacks intra_callbacks;
  intra_callbacks.intra_syntax_bits = fake_intra_syntax_bits;
  const InterRateModel with_intra(intra_callbacks);
  check(close(with_intra.intra_syntax_bits(0, 0, 0, 0), 3.0),
        "intra_syntax_bits callback overrides the default");

  RateCallbacks residual_callbacks;
  residual_callbacks.residual_bits = fake_residual_bits;
  const InterRateModel with_residual(residual_callbacks);
  av2enc::m2::residual::BlockDecision decision;
  decision.skip = false;
  decision.rate_bits = 42.0;
  check(close(with_residual.residual_bits(decision), 0.0),
        "residual_bits callback overrides the default");
}

void test_inter_selection() {
  RdoConfig config = av2enc::m2::rdo::calibrated_m2_config();
  InterCandidate cheap;
  cheap.valid = true;
  cheap.kind = InterCandidateKind::kSkipGlobal;
  cheap.distortion = 100.0;
  cheap.rate_bits = 0.0;
  cheap.skip_flag = true;

  InterCandidate quality;
  quality.valid = true;
  quality.kind = InterCandidateKind::kNewMv;
  quality.distortion = 0.0;
  quality.rate_bits = 1.0;
  quality.skip_flag = true;

  std::vector<InterCandidate> candidates = {cheap, quality};
  const auto decision =
      av2enc::m2::rdo::decide_inter_candidates(candidates, config);
  check(decision.found, "selection finds a candidate");
  check(decision.best_index == 0,
        "rate-aware selection prefers the cheaper total cost");

  config.rate_aware = false;
  const auto distortion_only =
      av2enc::m2::rdo::decide_inter_candidates(candidates, config);
  check(distortion_only.found && distortion_only.best_index == 1,
        "SSE-only control prefers the lower distortion");

  RdoConfig sad_config = av2enc::m2::rdo::calibrated_m2_config();
  sad_config.metric = av2enc::rdo::DistortionMetric::kSad;
  InterCandidate sad_cheap;
  sad_cheap.valid = true;
  sad_cheap.kind = InterCandidateKind::kSkipGlobal;
  sad_cheap.distortion = 100.0;  // SAD units
  sad_cheap.rate_bits = 0.0;
  InterCandidate sad_quality;
  sad_quality.valid = true;
  sad_quality.kind = InterCandidateKind::kNewMv;
  sad_quality.distortion = 0.0;
  sad_quality.rate_bits = 2.0;
  std::vector<InterCandidate> sad_candidates = {sad_cheap, sad_quality};
  const auto sad_decision =
      av2enc::m2::rdo::decide_inter_candidates(sad_candidates, sad_config);
  check(sad_decision.found && sad_decision.best_index == 1,
        "SAD metric uses the sqrt(lambda_sse) scaling");
  av2enc::m2::rdo::InterCandidateSpec sad_spec;
  sad_spec.kind = InterCandidateKind::kNewMv;
  sad_spec.inter_mode = 2;
  sad_spec.has_distortion_override = true;
  sad_spec.distortion_override = 17.0;
  const InterRateModel model;
  RateContext context;
  ScoredInterCandidate sad_scored;
  const auto sad_block = make_block(16, 16, 128, 128);
  std::string sad_error;
  check(av2enc::m2::rdo::score_inter_candidate(
            sad_spec, sad_block.input, true, context, sad_config, model,
            sad_scored, &sad_error),
        "SAD candidate scoring succeeds: " + sad_error);
  check(close(sad_scored.candidate.distortion, 17.0),
        "distortion override is used for SAD/SATD pre-selection");

  RdoConfig tie_config = av2enc::m2::rdo::calibrated_m2_config();
  InterCandidate intra;
  intra.valid = true;
  intra.kind = InterCandidateKind::kIntra;
  intra.distortion = 100.0;
  intra.rate_bits = 10.0;
  InterCandidate merge;
  merge.valid = true;
  merge.kind = InterCandidateKind::kMerge;
  merge.distortion = 100.0;
  merge.rate_bits = 10.0;
  InterCandidate skip;
  skip.valid = true;
  skip.kind = InterCandidateKind::kSkipGlobal;
  skip.distortion = 100.0;
  skip.rate_bits = 10.0;
  std::vector<InterCandidate> tie = {intra, merge, skip};
  const auto tie_decision =
      av2enc::m2::rdo::decide_inter_candidates(tie, tie_config);
  check(tie_decision.found && tie_decision.best_index == 2,
        "equal costs prefer the lower candidate kind");

  InterCandidate invalid;
  invalid.valid = true;
  invalid.kind = InterCandidateKind::kNewMv;
  invalid.distortion = -1.0;
  invalid.rate_bits = 0.0;
  std::vector<InterCandidate> mixed = {invalid, cheap};
  const auto mixed_decision =
      av2enc::m2::rdo::decide_inter_candidates(mixed, tie_config);
  check(mixed_decision.found && mixed_decision.best_index == 1,
        "invalid candidates are ignored");
}

void test_residual_and_scoring() {
  RdoConfig config = av2enc::m2::rdo::calibrated_m2_config();
  const InterRateModel model;
  RateContext context;

  const auto flat = make_block(16, 16, 128, 128);
  av2enc::m2::residual::BlockDecision skip;
  std::string error;
  check(av2enc::m2::rdo::decide_residual(flat.input, config, true, false, skip,
                                          &error),
        "residual decision succeeds on a flat block: " + error);
  check(skip.skip, "flat prediction is skipped");
  check(close(skip.skip_distortion, 0.0), "flat skip distortion is zero");

  av2enc::m2::residual::BlockDecision no_error_decision;
  check(av2enc::m2::rdo::decide_residual(
            flat.input, config, true, false, no_error_decision, nullptr),
        "residual decision accepts a null error pointer");

  const auto edge = make_block(16, 16, 128, 0);
  av2enc::m2::residual::BlockDecision residual;
  check(av2enc::m2::rdo::decide_residual(edge.input, config, true, false,
                                          residual, &error),
        "residual decision succeeds on a constant offset: " + error);
  check(residual.skip_distortion > 0.0, "offset skip distortion is positive");
  check(std::isfinite(residual.cost), "residual cost is finite");

  av2enc::m2::residual::BlockDecision forced;
  check(av2enc::m2::rdo::decide_residual(edge.input, config, true, true,
                                          forced, &error),
        "forced residual decision succeeds: " + error);
  check(!forced.skip, "force_residual keeps the residual");

  InterCandidateSpec spec;
  spec.kind = InterCandidateKind::kNewMv;
  spec.inter_mode = 2;
  spec.mvd = av2enc::inter::mc::Mv{0, 0};
  ScoredInterCandidate scored;
  check(av2enc::m2::rdo::score_inter_candidate(
            spec, flat.input, true, context, config, model, scored, &error),
        "score_inter_candidate succeeds: " + error);
  check(scored.valid, "scored candidate is valid");
  check(scored.candidate.rate_bits >= 0.0, "candidate rate is non-negative");
  check(scored.candidate.mode_bits > 0.0, "NEWMV pays mode/MV bits");

  ScoredInterCandidate scored_far;
  spec.mvd = av2enc::inter::mc::Mv{64, 0};
  check(av2enc::m2::rdo::score_inter_candidate(
            spec, flat.input, true, context, config, model, scored_far,
            &error),
        "score_inter_candidate with a large MVD succeeds: " + error);
  check(scored_far.candidate.mode_bits > scored.candidate.mode_bits,
        "a large MVD costs more mode bits");

  const auto intra = av2enc::m2::rdo::make_intra_candidate(
      100.0, 20.0, 0, 0, 0, context, config, model);
  check(intra.valid, "intra candidate is valid");
  check(intra.rate_bits >= 20.0, "intra rate includes coefficient bits");
}

void test_intra_rate() {
  std::vector<std::int32_t> levels(64, 0);
  std::array<av2enc::m2::rdo::IntraPlaneRateInput, 1> zero;
  zero[0].plane = 0;
  zero[0].tx_size = 3;  // 8x8
  zero[0].adj_w = 8;
  zero[0].adj_h = 8;
  zero[0].levels_raster = std::span<const std::int32_t>(levels);
  zero[0].eob = 0;
  double zero_bits = 0.0;
  std::string error;
  check(av2enc::m2::rdo::intra_coefficient_bits(
            std::span<const av2enc::m2::rdo::IntraPlaneRateInput>(zero),
            &zero_bits, &error),
        "zero-level intra rate succeeds: " + error);
  check(zero_bits > 0.0, "zero-level intra rate is positive");

  levels[0] = 8;
  auto one = zero;
  one[0].eob = 8;
  double one_bits = 0.0;
  check(av2enc::m2::rdo::intra_coefficient_bits(
            std::span<const av2enc::m2::rdo::IntraPlaneRateInput>(one),
            &one_bits, &error),
        "non-zero intra rate succeeds: " + error);
  check(one_bits > zero_bits, "a coded coefficient costs more");
}

void test_context_adapters() {
  av2enc::m2::merge::NeighborModes neighbours;
  neighbours.above.available = true;
  neighbours.above.is_inter = true;
  neighbours.above.skip_flag = true;
  neighbours.left.available = true;
  neighbours.left.is_inter = true;
  neighbours.left.skip_flag = true;
  check(av2enc::m2::rdo::skip_context_from_neighbours(neighbours) == 2,
        "skip context sums both decoded neighbours");

  av2enc::m2::merge::NewMvContextParams params;
  std::vector<av2enc::inter::mc::MiInfo> mi(16);
  params.mi = mi.data();
  params.mi_cols = 4;
  params.mi_rows = 4;
  params.bw4 = 2;
  params.bh4 = 2;
  params.sb_h4 = 16;
  int new_mv_ctx = -1;
  const bool new_mv_ok =
      av2enc::m2::rdo::new_mv_context_from_neighbours(params, &new_mv_ctx);
  check(new_mv_ok, "new-MV context adapter succeeds without a neighbour grid");
  check(new_mv_ctx >= 0 && new_mv_ctx <= 4,
        "new-MV context is in the §7.11.2 range");

  const av2enc::m2::rdo::RateContext context =
      av2enc::m2::rdo::make_rate_context(1, neighbours, params, 4);
  check(context.is_inter_ctx == 1, "rate context carries is_inter context");
  check(context.skip_ctx == 2, "rate context carries skip context");
  check(context.max_drl_bits_minus_1 == 4,
        "rate context carries the DRL cap");
}

void test_partition() {
  av2enc::partition::FrameGeometry geometry;
  geometry.width = 64;
  geometry.height = 64;
  geometry.sb_size = 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;

  av2enc::partition::DecisionOptions options;
  options.min_block_size_px = 16;
  options.max_depth = 4;
  options.allow_rect = true;
  options.allow_split = false;

  av2enc::m2::rdo::PartitionRdoConfig config;
  config.rdo = av2enc::m2::rdo::calibrated_m2_config();
  config.split_signal_bits = 2.0;
  config.horz_signal_bits = 2.0;
  config.vert_signal_bits = 2.0;

  std::string error;
  const auto flat = av2enc::m2::rdo::decide_partition_rdo(
      geometry, options,
      [](const av2enc::partition::PartitionNodeInfo&) {
        av2enc::m2::rdo::LeafCost cost;
        cost.distortion = 0.0;
        cost.rate_bits = 0.0;
        return cost;
      },
      config, &error);
  check(flat.status == av2enc::partition::DecisionStatus::kOk,
        "flat partition decision succeeds: " + error);
  check(av2enc::partition::validate_frame(geometry, flat, error),
        "flat partition tree validates: " + error);
  check(flat.leaves == 1, "a zero-cost leaf is not split");

  const auto split = av2enc::m2::rdo::decide_partition_rdo(
      geometry, options,
      [](const av2enc::partition::PartitionNodeInfo& info) {
        av2enc::m2::rdo::LeafCost cost;
        const int w = av2enc::partition::block_width(info.size);
        const int h = av2enc::partition::block_height(info.size);
        cost.distortion = (w > 32 || h > 32) ? 1.0e9 : 0.0;
        cost.rate_bits = 0.0;
        return cost;
      },
      config, &error);
  check(split.status == av2enc::partition::DecisionStatus::kOk,
        "split partition decision succeeds: " + error);
  check(av2enc::partition::validate_frame(geometry, split, error),
        "split partition tree validates: " + error);
  check(split.leaves > 1, "a high 64x64 distortion is split");

  check(av2enc::m2::rdo::variance_leaf_rate_bits(1000.0, 16, 16, 66.0) > 0.0,
        "variance leaf rate is positive");
  check(av2enc::m2::rdo::variance_leaf_rate_bits(4000.0, 16, 16, 66.0) >
            av2enc::m2::rdo::variance_leaf_rate_bits(1000.0, 16, 16, 66.0),
        "variance leaf rate grows with distortion");
}

}  // namespace

int main() {
  test_config();
  test_rate_model();
  test_inter_selection();
  test_residual_and_scoring();
  test_intra_rate();
  test_context_adapters();
  test_partition();
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("m2 rdo: all checks passed\n");
  return 0;
}
