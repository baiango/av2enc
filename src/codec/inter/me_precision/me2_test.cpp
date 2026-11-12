// AV2 inter motion estimation, extended — CPU reference tests.
//
// Coverage:
//   * MV precision enum, frame-header bits (use_qtr_precision_mv /
//     allow_high_precision_mv), enable_flex_mvres per-block derivation and
//     inverse signalling;
//   * normative lower_mv_precision() rounding, 1/16 -> 1/8 conversion and
//     quantization to every signalable precision;
//   * MVD shell-code rate proxy and precision side-information bins;
//   * eighth/sixteenth-pel planted synthetic shifts, normative filters;
//   * exhaustive refinement (deterministic order, exact eval counts) and
//     hill-climb escape cases;
//   * deterministic tie-breaks (reference order, flat-cost precision choice,
//     rate tie-break);
//   * partial blocks, guard canaries, multi-reference, API validation;
//   * quality/rate proxy report on real 720p frames.
//
// Usage: ./me2_test [repo_root]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "codec/inter/me/me_test_common.h"
#include "codec/inter/me_precision/me2.h"
#include "codec/inter/mc/mc.h"

namespace {

using namespace av2enc::me;
using namespace av2enc::me::test;
using namespace av2enc::me2;
using av2enc::me2::Me2Options;
using av2enc::me2::Me2Result;
using av2enc::me2::Me2Status;
namespace prec = av2enc::me2;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      if (g_failures <= 30) {                                              \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
      }                                                                    \
    }                                                                      \
  } while (0)

#define CHECK_EQ(a, b, msg)                                                \
  do {                                                                     \
    ++g_checks;                                                            \
    const long long va = static_cast<long long>(a);                        \
    const long long vb = static_cast<long long>(b);                        \
    if (va != vb) {                                                        \
      ++g_failures;                                                        \
      if (g_failures <= 30) {                                              \
        std::fprintf(stderr, "FAIL %s:%d: %s (%lld != %lld)\n", __FILE__,  \
                     __LINE__, msg, va, vb);                               \
      }                                                                    \
    }                                                                      \
  } while (0)

// ---------------------------------------------------------------------------
// Precision / signalling
// ---------------------------------------------------------------------------
void test_frame_precision_bits() {
  using namespace av2enc::me2;
  {
    const FramePrecisionSyntax s =
        me2_frame_precision_from_bits(0, 0, 0, 0);
    CHECK(s.valid, "half valid");
    CHECK_EQ(s.frame_precision, kMvPrecisionHalfPel, "half precision");
    CHECK_EQ(s.coded_bits, 2, "half uses two header bits");
    CHECK_EQ(s.use_per_block_mv_precision, 0, "flex off");
  }
  {
    const FramePrecisionSyntax s =
        me2_frame_precision_from_bits(0, 1, 1, 0);
    CHECK_EQ(s.frame_precision, kMvPrecisionQuarterPel, "quarter precision");
    CHECK_EQ(s.coded_bits, 1, "quarter codes only use_qtr_precision_mv");
    CHECK_EQ(s.allow_high_precision_mv, 0,
             "allow_high_precision_mv not coded for quarter");
  }
  {
    const FramePrecisionSyntax s =
        me2_frame_precision_from_bits(0, 0, 1, 1);
    CHECK_EQ(s.frame_precision, kMvPrecisionEighthPel, "eighth precision");
    CHECK_EQ(s.coded_bits, 2, "eighth uses two header bits");
    CHECK_EQ(s.use_per_block_mv_precision, 1, "flex on");
  }
  {
    const FramePrecisionSyntax s =
        me2_frame_precision_from_bits(1, 1, 1, 1);
    CHECK_EQ(s.frame_precision, kMvPrecisionOnePel, "force integer");
    CHECK_EQ(s.coded_bits, 0, "force integer codes neither precision flag");
    CHECK_EQ(s.use_per_block_mv_precision, 0,
             "force integer disables per-block precision");
  }

  const int targets[3] = {kMvPrecisionHalfPel, kMvPrecisionQuarterPel,
                          kMvPrecisionEighthPel};
  for (int flex = 0; flex <= 1; ++flex) {
    for (int t : targets) {
      FramePrecisionSyntax s;
      CHECK(me2_frame_precision_to_bits(t, flex, 0, &s),
            "target frame precision signallable");
      const FramePrecisionSyntax d = me2_frame_precision_from_bits(
          0, s.use_qtr_precision_mv, s.allow_high_precision_mv, flex);
      CHECK_EQ(d.frame_precision, t, "frame precision round-trip");
      CHECK_EQ(d.coded_bits, s.coded_bits, "header bit count round-trip");
      CHECK_EQ(d.use_per_block_mv_precision, flex,
               "per-block flag round-trip");
    }
  }
  FramePrecisionSyntax s;
  CHECK(!me2_frame_precision_to_bits(kMvPrecisionOnePel, 0, 0, &s),
        "one-pel needs force_integer_mv");
  CHECK(me2_frame_precision_to_bits(kMvPrecisionOnePel, 1, 1, &s),
        "one-pel with force_integer_mv");
  CHECK_EQ(s.coded_bits, 0, "one-pel header bits");
  CHECK(!me2_frame_precision_to_bits(kMvPrecisionTwoPel, 0, 0, &s),
        "two-pel is not signallable by the frame header");
  CHECK(!me2_frame_precision_to_bits(kMvPrecisionEighthPel, 0, 1, &s),
        "force_integer_mv pins frame precision");
}

void test_block_precision_syntax() {
  using namespace av2enc::me2;
  struct Case {
    int frame;
    int target;
    int use_most;
    int pb;
  };
  const Case cases[] = {
      {kMvPrecisionEighthPel, kMvPrecisionEighthPel, 1, -1},
      {kMvPrecisionEighthPel, kMvPrecisionHalfPel, 0, 0},
      {kMvPrecisionEighthPel, kMvPrecisionOnePel, 0, 1},
      {kMvPrecisionEighthPel, kMvPrecisionFourPel, 0, 2},
      {kMvPrecisionEighthPel, kMvPrecisionEightPel, 0, 3},
      {kMvPrecisionQuarterPel, kMvPrecisionQuarterPel, 1, -1},
      {kMvPrecisionQuarterPel, kMvPrecisionOnePel, 0, 0},
      {kMvPrecisionQuarterPel, kMvPrecisionFourPel, 0, 1},
      {kMvPrecisionQuarterPel, kMvPrecisionEightPel, 0, 2},
      {kMvPrecisionHalfPel, kMvPrecisionHalfPel, 1, -1},
      {kMvPrecisionHalfPel, kMvPrecisionOnePel, 0, 0},
      {kMvPrecisionHalfPel, kMvPrecisionFourPel, 0, 1},
      {kMvPrecisionHalfPel, kMvPrecisionEightPel, 0, 2},
  };
  for (const Case& c : cases) {
    int um = -1;
    int pb = -1;
    CHECK(me2_block_precision_to_syntax(c.frame, c.target, &um, &pb),
          "reachable block precision");
    CHECK_EQ(um, c.use_most, "use_most_probable_precision");
    CHECK_EQ(pb, c.pb, "pb_mv_precision symbol");
    if (pb >= 0) {
      CHECK_EQ(me2_pb_precision_derive(c.frame, pb), c.target,
               "pb derivation round-trip");
    }
  }

  // Unreachable / invalid targets.
  int um = 0;
  int pb = 0;
  CHECK(!me2_block_precision_to_syntax(kMvPrecisionEighthPel,
                                       kMvPrecisionQuarterPel, &um, &pb),
        "quarter is not reachable when the frame is eighth");
  CHECK(!me2_block_precision_to_syntax(kMvPrecisionEighthPel,
                                       kMvPrecisionTwoPel, &um, &pb),
        "two-pel gap is not reachable");
  CHECK(!me2_block_precision_to_syntax(kMvPrecisionQuarterPel,
                                       kMvPrecisionHalfPel, &um, &pb),
        "half is not reachable when the frame is quarter");
  CHECK(!me2_block_precision_to_syntax(kMvPrecisionHalfPel,
                                       kMvPrecisionQuarterPel, &um, &pb),
        "per-block precision cannot get finer");

  CHECK(me2_precision_reachable(kMvPrecisionEighthPel, kMvPrecisionHalfPel,
                                1),
        "eighth frame can lower to half");
  CHECK(!me2_precision_reachable(kMvPrecisionEighthPel, kMvPrecisionHalfPel,
                                 0),
        "flex off pins the frame precision");
  CHECK(!me2_precision_reachable(kMvPrecisionOnePel, kMvPrecisionFourPel,
                                 1),
        "one-pel frame disables per-block precision (force_integer)");

  // Candidate sets, coarse-to-fine order.
  int cand[prec::kMvPrecisionCount];
  int n = me2_precision_candidates(kMvPrecisionEighthPel, 1, 0, cand,
                                   prec::kMvPrecisionCount);
  CHECK_EQ(n, 5, "eighth frame reachable set size");
  const int expect[5] = {kMvPrecisionEightPel, kMvPrecisionFourPel,
                         kMvPrecisionOnePel, kMvPrecisionHalfPel,
                         kMvPrecisionEighthPel};
  for (int i = 0; i < 5; ++i) {
    CHECK_EQ(cand[i], expect[i], "candidate ordering");
  }
  n = me2_precision_candidates(kMvPrecisionEighthPel, 0, 0, cand,
                               prec::kMvPrecisionCount);
  CHECK_EQ(n, 1, "flex off one candidate");
  CHECK_EQ(cand[0], kMvPrecisionEighthPel, "flex off keeps frame precision");

  n = me2_precision_candidates(kMvPrecisionEighthPel, 1,
                               (1 << kMvPrecisionHalfPel), cand,
                               prec::kMvPrecisionCount);
  CHECK_EQ(n, 2, "mask keeps only the frame and the allowed coarser set");
  CHECK_EQ(cand[0], kMvPrecisionHalfPel, "masked coarser candidate");
  CHECK_EQ(cand[1], kMvPrecisionEighthPel, "mask keeps the frame precision");

  // Context helpers.
  const int most[2] = {1, 0};
  CHECK_EQ(me2_ctx_use_most_probable(most, 2), 1, "use_most ctx");
  const int np[3] = {kMvPrecisionEighthPel, kMvPrecisionOnePel,
                     kMvPrecisionEighthPel};
  CHECK_EQ(me2_ctx_pb_precision(np, 3, kMvPrecisionEighthPel), 1,
           "pb ctx");
  CHECK_EQ(me2_pb_precision_cdf_row(kMvPrecisionEighthPel), 2, "cdf row");
  CHECK_EQ(me2_pb_precision_cdf_row(kMvPrecisionHalfPel), 0, "cdf row half");
}

void test_lower_precision() {
  using namespace av2enc::me2;
  CHECK_EQ(me2_lower_mv_precision_component(7, kMvPrecisionEighthPel), 7,
           "eighth is identity");
  // Quarter: ties down, sign preserved.
  CHECK_EQ(me2_lower_mv_precision_component(1, kMvPrecisionQuarterPel), 0,
           "quarter 0.125 -> 0");
  CHECK_EQ(me2_lower_mv_precision_component(3, kMvPrecisionQuarterPel), 2,
           "quarter 0.375 -> 0.25");
  CHECK_EQ(me2_lower_mv_precision_component(5, kMvPrecisionQuarterPel), 4,
           "quarter 0.625 -> 0.5");
  CHECK_EQ(me2_lower_mv_precision_component(-3, kMvPrecisionQuarterPel), -2,
           "quarter negative tie down");
  CHECK_EQ(me2_lower_mv_precision_component(-1, kMvPrecisionQuarterPel), 0,
           "quarter -0.125 -> 0");
  // Half: ties down in 4-unit steps.
  CHECK_EQ(me2_lower_mv_precision_component(6, kMvPrecisionHalfPel), 4,
           "half 0.75 -> 0.5 (tie down)");
  CHECK_EQ(me2_lower_mv_precision_component(7, kMvPrecisionHalfPel), 8,
           "half 0.875 -> 1.0");
  CHECK_EQ(me2_lower_mv_precision_component(2, kMvPrecisionHalfPel), 0,
           "half 0.25 -> 0");
  // One pel.
  CHECK_EQ(me2_lower_mv_precision_component(7, kMvPrecisionOnePel), 8,
           "one 0.875 -> 1");
  CHECK_EQ(me2_lower_mv_precision_component(12, kMvPrecisionOnePel), 8,
           "one 1.5 -> 1 (tie down)");
  // Clipping at the MV bounds.
  CHECK_EQ(me2_lower_mv_precision_component(kMvLow + 1,
                                            kMvPrecisionQuarterPel),
           kMvLow + 2, "lower clip to MV_LOW + radix");
  CHECK_EQ(me2_lower_mv_precision_component(kMvUpp - 1,
                                            kMvPrecisionQuarterPel),
           kMvUpp - 2, "upper clip to MV_UPP - radix");

  // 1/16 -> 1/8 conversion rounds ties away from zero.
  CHECK_EQ(me2_1_16_to_eighth(1), 1, "1/16 -> 1/8 tie up");
  CHECK_EQ(me2_1_16_to_eighth(3), 2, "1/16 -> 1/8 3 -> 2");
  CHECK_EQ(me2_1_16_to_eighth(-1), -1, "1/16 -> 1/8 negative tie away");
  CHECK_EQ(me2_1_16_to_eighth(-3), -2, "1/16 -> 1/8 negative");
  // Quantized 1/16 MVs sit on the precision grid.
  const int probes[9] = {-9, -5, -3, -1, 0, 1, 3, 5, 9};
  for (int p = kMvPrecisionOnePel; p <= kMvPrecisionEighthPel; ++p) {
    for (int v : probes) {
      const int q = me2_quantize_mv_component_1_16(v, p);
      CHECK(me2_mv_is_at_precision(q, p), "quantized MV is on the grid");
      CHECK_EQ(me2_quantize_mv_component_1_16(q, p), q,
               "quantization is idempotent");
    }
  }
  CHECK_EQ(me2_quantize_mv_component_1_16(1, kMvPrecisionEighthPel), 2,
         "eighth quantum is 2/16");
  CHECK_EQ(me2_quantize_mv_component_1_16(3, kMvPrecisionQuarterPel), 4,
         "quarter quantum is 4/16");
  CHECK_EQ(me2_quantize_mv_component_1_16(5, kMvPrecisionHalfPel), 8,
         "half quantum is 8/16");
}

void test_rate_proxy() {
  CHECK_EQ(me2_mvd_shell_bins(0, 0, kMvPrecisionEighthPel), 3,
           "zero MVD still codes shell_set/class/offset");
  CHECK_EQ(me2_mvd_shell_bins(2, 0, kMvPrecisionEighthPel), 6,
           "eighth MVD 0.25 pel");
  CHECK_EQ(me2_mvd_shell_bins(2, 0, kMvPrecisionQuarterPel), 5,
           "same physical MVD at quarter precision is cheaper");
  CHECK_EQ(me2_mvd_shell_bins(1, 0, kMvPrecisionQuarterPel), -1,
           "non-representable MVD rejected");
  CHECK_EQ(me2_mvd_shell_bins(-2, 0, kMvPrecisionQuarterPel), 5,
           "sign bit counted once");
  CHECK_EQ(me2_precision_side_bins(kMvPrecisionEighthPel,
                                   kMvPrecisionEighthPel, 1, true),
           1, "use_most_probable side bit");
  CHECK_EQ(me2_precision_side_bins(kMvPrecisionEighthPel,
                                   kMvPrecisionHalfPel, 1, true),
           2, "pb_mv_precision side bits");
  CHECK_EQ(me2_precision_side_bins(kMvPrecisionEighthPel,
                                   kMvPrecisionHalfPel, 0, true),
           0, "no side bits when flex is off");
  CHECK_EQ(me2_precision_side_bins(kMvPrecisionEighthPel,
                                   kMvPrecisionHalfPel, 1, false),
           0, "no side bits for non-NEWMV");
  CHECK_EQ(me2_precision_side_bins(kMvPrecisionOnePel, kMvPrecisionOnePel,
                                   1, true),
           0, "force-integer frame disables precision side bits");
}

void test_mc_contract() {
  // The MC component's 1/16 -> 1/8 quantizer must agree with me2's.
  for (int v = -33; v <= 33; ++v) {
    const av2enc::inter::mc::Mv m =
        av2enc::inter::mc::mv_from_me_1_16(v, v * 2 + 1);
    CHECK_EQ(m.y, me2_1_16_to_eighth(v), "MC/me2 1/16 -> 1/8 y contract");
    CHECK_EQ(m.x, me2_1_16_to_eighth(v * 2 + 1),
             "MC/me2 1/16 -> 1/8 x contract");
  }
}

// ---------------------------------------------------------------------------
// Deterministic refinement and precision selection (cost-function level)
// ---------------------------------------------------------------------------
struct QuadraticCost {
  std::int32_t tx = 0;
  std::int32_t ty = 0;
  std::int32_t operator()(const MeCorePlane&, const MeCorePlane&, int, int,
                          int, int, std::int32_t mv_x, std::int32_t mv_y,
                          int, int) const {
    const std::int32_t dx = mv_x - tx;
    const std::int32_t dy = mv_y - ty;
    return dx * dx + dy * dy;
  }
};

void test_refinement_deterministic() {
  TestFrame frame;
  frame.init(32, 32);
  fill_flat(&frame, 128, 128);
  const MeCoreFrame cur = frame.view();
  const MeCoreRef refv = frame.ref_view();

  Me2Options opt;
  opt.search.method = kMeFull;
  opt.search.metric = kMeSad;
  opt.search.range = 0;  // predictor stage only; refinement does the work
  opt.search.subpel_stages = 0;
  opt.search.subpel_iterations = 0;
  opt.frame_precision = prec::kMvPrecisionEighthPel;
  opt.enable_flex_mvres = 0;
  opt.refine_radius = 3;
  opt.refine_step = 1;
  opt.mv_rate_lambda = 0;

  MeCoreJob job;
  job.x = 0;
  job.y = 0;
  job.w = 16;
  job.h = 16;
  job.num_refs = 1;
  job.ref_indices[0] = 0;

  // Target (2,1) is found exactly by the 1/16 refinement.
  {
    QuadraticCost q;
    q.tx = 2;
    q.ty = 1;
    const Me2Result r =
        prec::me2_search_core(cur, &refv, job, opt, nullptr, q);
    CHECK_EQ(r.search_mv_x, 2, "refinement finds exact x");
    CHECK_EQ(r.search_mv_y, 1, "refinement finds exact y");
    CHECK_EQ(r.search_cost, 0, "refinement reaches zero cost");
    CHECK_EQ(r.stage, kMeStageSixteenth, "refinement stage");
    CHECK_EQ(r.precision, prec::kMvPrecisionEighthPel, "frame precision kept");
    // Signalled eighth-pel MV: (2,1) -> (2,2), cost 1.
    CHECK_EQ(r.mv_x, 2, "signalled x quantized");
    CHECK_EQ(r.mv_y, 2, "signalled y quantized");
    CHECK_EQ(r.cost, 1, "signalled cost reflects quantization");
  }

  // Two equal-cost minima: fixed dy-then-dx order picks (1,-1) first.
  {
    QuadraticCost q;
    q.tx = 1;
    q.ty = 0;
    // Add a second minimum by making the cost independent of the sign of y.
    struct TwoMinCost {
      std::int32_t operator()(const MeCorePlane&, const MeCorePlane&, int,
                              int, int, int, std::int32_t mv_x,
                              std::int32_t mv_y, int, int) const {
        const std::int32_t dx = mv_x - 1;
        const std::int32_t dy = mv_y < 0 ? mv_y + 1 : mv_y - 1;
        return dx * dx + dy * dy;
      }
    } two;
    const Me2Result r =
        prec::me2_search_core(cur, &refv, job, opt, nullptr, two);
    // Both (1,-1) and (1,1) cost 0; dy=-1 is visited first.
    CHECK_EQ(r.search_mv_y, -1, "deterministic first-minimum tie-break");
    CHECK_EQ(r.search_mv_x, 1, "deterministic x");
    (void)q;
  }

  // Eval count is deterministic and covers (2r+1)^2 - 1 offsets.
  {
    QuadraticCost q;
    q.tx = 0;
    q.ty = 0;
    const Me2Result a =
        prec::me2_search_core(cur, &refv, job, opt, nullptr, q);
    const Me2Result b =
        prec::me2_search_core(cur, &refv, job, opt, nullptr, q);
    CHECK_EQ(a.evals, b.evals, "deterministic eval count");
    // predictor (1) + refinement (7*7 - 1 = 48) + precision (1) = 50.
    CHECK_EQ(a.evals, 50, "exact eval count");
    CHECK_EQ(a.search_mv_x, 0, "zero-cost center kept");
    CHECK_EQ(a.precision, prec::kMvPrecisionEighthPel,
             "flat cost prefers fine precision");
  }
}

void test_precision_rate_tradeoff() {
  TestFrame frame;
  frame.init(32, 32);
  fill_flat(&frame, 128, 128);
  const MeCoreFrame cur = frame.view();
  const MeCoreRef refv = frame.ref_view();

  Me2Options opt;
  opt.search.range = 0;
  opt.search.subpel_stages = 0;
  opt.search.subpel_iterations = 0;
  opt.frame_precision = prec::kMvPrecisionEighthPel;
  opt.enable_flex_mvres = 1;
  opt.refine_radius = 24;
  opt.refine_step = 1;
  opt.mv_rate_lambda = 10;

  MeCoreJob job;
  job.w = 16;
  job.h = 16;
  job.num_refs = 1;
  job.ref_indices[0] = 0;

  // Target 23/16 luma sample: eighth and half both quantize to x=24; the
  // eighth shell is one bin larger, so the rate term selects HALF_PEL.
  QuadraticCost q;
  q.tx = 23;
  q.ty = 0;
  const Me2Result r = prec::me2_search_core(cur, &refv, job, opt, nullptr, q);
  CHECK_EQ(r.mv_x, 24, "coarse quantization MV");
  CHECK_EQ(r.precision, prec::kMvPrecisionHalfPel,
           "rate term selects half precision");
  CHECK_EQ(r.use_most_probable_precision, 0,
           "non-frame precision uses pb_mv_precision");
  CHECK_EQ(r.pb_mv_precision, 0, "pb_mv_precision value");
  CHECK_EQ(r.pb_precision_coarser, 1, "block uses a coarser precision");
  CHECK(me2_mv_is_at_precision(r.mv_x, r.precision),
        "signalled MV matches precision");

  // With lambda 0 the distortion ties (both quantize to x=24); the fixed
  // tie-break prefers the lower-rate precision, which is HALF_PEL.
  opt.mv_rate_lambda = 0;
  const Me2Result f = prec::me2_search_core(cur, &refv, job, opt, nullptr, q);
  CHECK_EQ(f.precision, prec::kMvPrecisionHalfPel,
           "equal distortion tie-break prefers lower rate");
  CHECK_EQ(f.use_most_probable_precision, 0, "pb path for coarser precision");
  CHECK_EQ(f.pb_mv_precision, 0, "pb symbol for half precision");
}

// ---------------------------------------------------------------------------
// Image-domain search
// ---------------------------------------------------------------------------
TestFrame make_predicted_current(const TestFrame& ref, int mv_x, int mv_y,
                                 int filter) {
  TestFrame cur;
  cur.init(ref.width, ref.height);
  const TestPlane* src[3] = {&ref.y, &ref.u, &ref.v};
  TestPlane* dst[3] = {&cur.y, &cur.u, &cur.v};
  for (int p = 0; p < 3; ++p) {
    const MeCorePlane view = src[p]->view();
    const int mvc_x = p == 0 ? mv_x : mv_x / 2;
    const int mvc_y = p == 0 ? mv_y : mv_y / 2;
    for (int y = 0; y < dst[p]->height; ++y) {
      for (int x = 0; x < dst[p]->width; ++x) {
        dst[p]->row(y)[x] = static_cast<std::uint8_t>(
            me_sample_at(view, x, y, mvc_x, mvc_y, filter));
      }
    }
  }
  return cur;
}

Me2Options base_options(int stages, int radius) {
  Me2Options o;
  o.search.method = kMeFull;
  o.search.metric = kMeSad;
  o.search.range = 2;
  o.search.integer_step = 1;
  o.search.subpel_stages = stages;
  o.search.subpel_iterations = 3;
  o.search.subpel_interp = kMeInterpRegular;
  o.search.use_predictors = 1;
  o.search.use_merge_candidates = 0;
  o.frame_precision = prec::kMvPrecisionEighthPel;
  o.enable_flex_mvres = 0;
  o.refine_radius = radius;
  o.refine_step = 1;
  o.mv_rate_lambda = 0;
  return o;
}

void test_planted_subpel_shifts() {
  TestFrame ref;
  ref.init(128, 128);
  fill_smooth_texture(&ref, 0x1234u, 6);

  struct Shift {
    int x;
    int y;
    const char* name;
  };
  const Shift shifts[] = {
      {8, 0, "half-pel x"},      {4, 4, "quarter-pel diag"},
      {-2, 2, "eighth-pel diag"}, {1, 1, "sixteenth-pel diag"},
      {0, -1, "sixteenth-pel y"},
  };
  int eighth_improved = 0;
  int sixteenth_improved = 0;

  for (const Shift& s : shifts) {
    TestFrame cur = make_predicted_current(ref, s.x, s.y, kMeInterpRegular);
    const MeCoreFrame curv = cur.view();
    const MeCoreRef refv = ref.ref_view();

    MeCoreJob job;
    job.x = 32;
    job.y = 32;
    job.w = 64;
    job.h = 64;
    job.num_refs = 1;
    job.ref_indices[0] = 0;

    const std::int32_t planted =
        me_cost_plane(curv.plane[0], refv.plane[0], job.x, job.y, job.w,
                      job.h, s.x, s.y, kMeSad, kMeInterpRegular);

    const Me2Options quarter =
        base_options(kMeSubpelHalf | kMeSubpelQuarter, 0);
    const Me2Options eighth =
        base_options(kMeSubpelHalf | kMeSubpelQuarter | kMeSubpelEighth, 0);
    const Me2Options all = base_options(
        kMeSubpelHalf | kMeSubpelQuarter | kMeSubpelEighth |
            kMeSubpelSixteenth,
        2);

    const Me2Result rq =
        prec::me2_search_core(curv, &refv, job, quarter, nullptr,
                              MeScalarCost{});
    const Me2Result re =
        prec::me2_search_core(curv, &refv, job, eighth, nullptr,
                              MeScalarCost{});
    const Me2Result ra =
        prec::me2_search_core(curv, &refv, job, all, nullptr,
                              MeScalarCost{});

    CHECK(rq.ref_index == 0 && re.ref_index == 0 && ra.ref_index == 0,
          "planted shift found a reference");
    // The planted 1/16 MV may not be signalable: compare against the planted
    // MV quantized to the selected precision.
    const std::int32_t px =
        me2_quantize_mv_component_1_16(s.x, ra.precision);
    const std::int32_t py =
        me2_quantize_mv_component_1_16(s.y, ra.precision);
    const std::int32_t planted_signal =
        me_cost_plane(curv.plane[0], refv.plane[0], job.x, job.y, job.w,
                      job.h, px, py, kMeSad, kMeInterpRegular);
    CHECK(ra.cost <= planted_signal,
          "all-stage signalled cost no worse than quantized planted MV");
    CHECK(re.search_cost <= rq.search_cost,
          "eighth stage no worse than quarter (search cost)");
    CHECK(ra.search_cost <= re.search_cost,
          "sixteenth refinement no worse than eighth (search cost)");
    CHECK(ra.evals >= re.evals && re.evals >= rq.evals,
          "more stages evaluate more candidates");
    CHECK(me2_mv_is_at_precision(ra.mv_x, ra.precision) &&
              me2_mv_is_at_precision(ra.mv_y, ra.precision),
          "signalled MV on the precision grid");
    Me2CostJob cj;
    cj.precision = ra.precision;
    cj.base.x = job.x;
    cj.base.y = job.y;
    cj.base.w = job.w;
    cj.base.h = job.h;
    cj.base.mv_x = ra.mv_x;
    cj.base.mv_y = ra.mv_y;
    cj.base.metric = kMeSad;
    cj.base.interp = kMeInterpRegular;
    std::int32_t cost_check = 0;
    CHECK(me2_cost_block(curv, refv, cj, &cost_check) == Me2Status::kOk,
          "reported MV is a valid cost candidate");
    CHECK_EQ(cost_check, ra.cost,
             "reported cost equals me2_cost_block at the signalled MV");
    if (re.search_cost < rq.search_cost) {
      ++eighth_improved;
    }
    if (ra.search_cost < re.search_cost) {
      ++sixteenth_improved;
    }
    std::printf("  shift %-20s planted=%4d q=%4d e=%4d a=%4d  mv=(%d,%d)\n",
                s.name, planted, rq.cost, re.cost, ra.cost, ra.mv_x, ra.mv_y);
  }
  CHECK(eighth_improved > 0, "eighth stage improves at least one shift");
  CHECK(sixteenth_improved > 0,
        "sixteenth refinement improves at least one shift");
}

void test_hill_climb_escape() {
  // A 2-D quadratic minimum two 1/16 steps diagonally away from the predictor.
  // With subpel_iterations = 1 the 8-neighbour ring moves one step per stage
  // and can stall; the exhaustive pass must reach the true minimum.
  TestFrame frame;
  frame.init(64, 64);
  fill_flat(&frame, 100, 100);
  const MeCoreFrame cur = frame.view();
  const MeCoreRef refv = frame.ref_view();

  MeCoreJob job;
  job.w = 32;
  job.h = 32;
  job.num_refs = 1;
  job.ref_indices[0] = 0;

  QuadraticCost q;
  q.tx = 2;
  q.ty = 2;

  Me2Options no_refine = base_options(kMeSubpelHalf | kMeSubpelQuarter, 0);
  no_refine.search.range = 0;
  no_refine.search.subpel_iterations = 1;

  Me2Options refine = no_refine;
  refine.refine_radius = 2;

  const Me2Result a =
      prec::me2_search_core(cur, &refv, job, no_refine, nullptr, q);
  const Me2Result b =
      prec::me2_search_core(cur, &refv, job, refine, nullptr, q);
  CHECK(a.search_cost > 0, "hill climb stalls without refinement");
  CHECK_EQ(b.search_cost, 0, "exhaustive refinement escapes the stall");
  CHECK_EQ(b.search_mv_x, 2, "escaped x");
  CHECK_EQ(b.search_mv_y, 2, "escaped y");
}

void test_partial_blocks_and_guards() {
  TestFrame cur;
  cur.init(67, 53);
  fill_smooth_texture(&cur, 0xBEEFu, 5);
  TestFrame ref = make_shifted_ref(cur, 1, 1);
  const MeCoreFrame cv = cur.view();
  const MeCoreRef rv = ref.ref_view();

  const auto jobs = make_block_grid(67, 53, 16, 16);
  for (const MeCoreJob& job : jobs) {
    Me2Options opt = base_options(
        kMeSubpelHalf | kMeSubpelQuarter | kMeSubpelEighth |
            kMeSubpelSixteenth,
        2);
    opt.search.range = 4;
    opt.enable_flex_mvres = 1;
    opt.mv_rate_lambda = 8;
    const Me2Result r =
        prec::me2_search_core(cv, &rv, job, opt, nullptr, MeScalarCost{});
    CHECK(r.visible_w > 0 && r.visible_h > 0, "partial block visible");
    CHECK(r.visible_w <= 16 && r.visible_h <= 16, "partial block clipped");
    CHECK(me2_mv_is_at_precision(r.mv_x, r.precision) &&
              me2_mv_is_at_precision(r.mv_y, r.precision),
          "partial block MV precision");
  }
  CHECK(cur.guard_intact(), "current frame guard intact");
  CHECK(ref.guard_intact(), "reference frame guard intact");
}

void test_reference_tie_and_determinism() {
  TestFrame cur;
  cur.init(64, 64);
  fill_smooth_texture(&cur, 0xCAFEu, 7);
  TestFrame ref = make_shifted_ref(cur, 2, -1);
  const MeCoreFrame cv = cur.view();
  MeCoreRef refs[2] = {ref.ref_view(), ref.ref_view()};  // identical refs

  MeCoreJob job;
  job.w = 32;
  job.h = 32;
  job.num_refs = 2;
  job.ref_indices[0] = 0;
  job.ref_indices[1] = 1;

  const Me2Options opt = base_options(
      kMeSubpelHalf | kMeSubpelQuarter | kMeSubpelEighth |
          kMeSubpelSixteenth,
      1);
  const Me2Result a =
      prec::me2_search_core(cv, refs, job, opt, nullptr, MeScalarCost{});
  const Me2Result b =
      prec::me2_search_core(cv, refs, job, opt, nullptr, MeScalarCost{});
  CHECK_EQ(a.ref_index, 0, "first reference wins a strict tie");
  CHECK(a.mv_x == b.mv_x && a.mv_y == b.mv_y && a.cost == b.cost &&
            a.precision == b.precision && a.evals == b.evals,
        "batch result deterministic");
}

void test_merge_field_passthrough() {
  TestFrame cur;
  cur.init(64, 64);
  fill_smooth_texture(&cur, 0xD00Du, 7);
  TestFrame ref = make_shifted_ref(cur, 2, 1);
  const MeCoreFrame cv = cur.view();
  const MeCoreRef rv = ref.ref_view();

  // 4x4 MI field populated with the planting MV (2,1) samples.
  const int cols = 16;
  const int rows = 16;
  std::vector<std::int32_t> mvx(static_cast<std::size_t>(cols) * rows, 32);
  std::vector<std::int32_t> mvy(static_cast<std::size_t>(cols) * rows, 16);
  std::vector<std::uint8_t> valid(static_cast<std::size_t>(cols) * rows, 1);
  MeCoreMvField field;
  field.mv_x = mvx.data();
  field.mv_y = mvy.data();
  field.valid = valid.data();
  field.cols = cols;
  field.rows = rows;

  MeCoreJob job;
  job.x = 16;
  job.y = 16;
  job.w = 16;
  job.h = 16;
  job.num_refs = 1;
  job.ref_indices[0] = 0;

  Me2Options opt = base_options(0, 0);
  opt.search.range = 0;
  opt.search.use_merge_candidates = 1;
  opt.enable_flex_mvres = 0;

  const Me2Result with_field =
      prec::me2_search_core(cv, &rv, job, opt, &field, MeScalarCost{});
  const Me2Result without_field =
      prec::me2_search_core(cv, &rv, job, opt, nullptr, MeScalarCost{});
  CHECK_EQ(with_field.mv_x, 32, "merge field MV used (x)");
  CHECK_EQ(with_field.mv_y, 16, "merge field MV used (y)");
  CHECK(with_field.cost < without_field.cost,
        "merge field MV beats the zero predictor");
}

// ---------------------------------------------------------------------------
// API validation
// ---------------------------------------------------------------------------
void test_api_validation() {
  TestFrame cur;
  cur.init(32, 32);
  fill_flat(&cur, 64, 64);
  TestFrame ref = make_shifted_ref(cur, 1, 0);
  const MeCoreFrame cv = cur.view();
  const MeCoreRef rv = ref.ref_view();

  MeCoreJob job;
  job.w = 16;
  job.h = 16;
  job.num_refs = 1;
  job.ref_indices[0] = 0;

  Me2Options opt = base_options(
      kMeSubpelHalf | kMeSubpelQuarter | kMeSubpelEighth |
          kMeSubpelSixteenth,
      2);
  Me2Result out[2];
  out[0].cost = 12345;
  out[1].cost = 67890;

  CHECK(prec::me2_search_batch(cv, &rv, 1, &job, 1, opt, nullptr, out) ==
            Me2Status::kOk,
        "valid batch ok");
  CHECK(out[0].ref_index == 0, "valid batch wrote a result");

  // Invalid options leave results untouched.
  out[0].cost = 12345;
  opt.frame_precision = 7;
  CHECK(prec::me2_search_batch(cv, &rv, 1, &job, 1, opt, nullptr, out) ==
            Me2Status::kInvalidArgument,
        "bad frame precision rejected");
  CHECK_EQ(out[0].cost, 12345, "results untouched on invalid options");

  opt = base_options(kMeSubpelHalf, 2);
  opt.enable_flex_mvres = 2;
  CHECK(prec::me2_search_batch(cv, &rv, 1, &job, 1, opt, nullptr, out) ==
            Me2Status::kInvalidArgument,
        "bad flex flag rejected");

  opt = base_options(kMeSubpelHalf, 2);
  opt.refine_radius = 65;
  CHECK(prec::me2_search_batch(cv, &rv, 1, &job, 1, opt, nullptr, out) ==
            Me2Status::kInvalidArgument,
        "bad refine radius rejected");

  opt = base_options(kMeSubpelHalf, 2);
  opt.mv_rate_shift = 25;
  CHECK(prec::me2_search_batch(cv, &rv, 1, &job, 1, opt, nullptr, out) ==
            Me2Status::kInvalidArgument,
        "bad rate shift rejected");

  opt = base_options(kMeSubpelHalf, 2);
  MeCoreJob bad = job;
  bad.num_refs = 0;
  CHECK(prec::me2_search_batch(cv, &rv, 1, &bad, 1, opt, nullptr, out) ==
            Me2Status::kInvalidArgument,
        "no reference rejected");
  bad = job;
  bad.ref_indices[0] = 3;
  CHECK(prec::me2_search_batch(cv, &rv, 1, &bad, 1, opt, nullptr, out) ==
            Me2Status::kInvalidArgument,
        "out-of-range reference rejected");
  bad = job;
  bad.x = 32;
  CHECK(prec::me2_search_batch(cv, &rv, 1, &bad, 1, opt, nullptr, out) ==
            Me2Status::kInvalidArgument,
        "origin outside frame rejected");

  prec::Me2CostJob cj;
  cj.base.x = 0;
  cj.base.y = 0;
  cj.base.w = 16;
  cj.base.h = 16;
  cj.base.mv_x = 3;  // not on the eighth grid
  cj.base.mv_y = 0;
  cj.precision = prec::kMvPrecisionEighthPel;
  std::int32_t cost = 0;
  CHECK(prec::me2_cost_block(cv, rv, cj, &cost) ==
            Me2Status::kInvalidArgument,
        "off-grid cost MV rejected");
  cj.base.mv_x = 4;
  cj.precision = prec::kMvPrecisionQuarterPel;
  CHECK(prec::me2_cost_block(cv, rv, cj, &cost) == Me2Status::kOk,
        "on-grid quarter cost ok");
  CHECK(prec::me2_status_string(Me2Status::kOk) != nullptr, "status string");
}

// ---------------------------------------------------------------------------
// Quality / rate proxy on real 720p
// ---------------------------------------------------------------------------
struct QualityTotals {
  std::int64_t sad = 0;
  std::int64_t cost = 0;
  std::int64_t search_cost = 0;
  std::int64_t sse = 0;
  std::int64_t samples = 0;
  std::int64_t mv_bins = 0;
  std::int64_t side_bins = 0;
  std::int64_t sel = 0;
  std::int64_t evals = 0;
  int blocks = 0;
  int precision_hist[prec::kMvPrecisionCount] = {};
};

QualityTotals measure_config(const std::vector<TestFrame>& frames,
                             const Me2Options& opt, int pair_stride) {
  QualityTotals t;
  for (int pair = 0; pair + 1 < static_cast<int>(frames.size());
       pair += pair_stride) {
    const TestFrame& cur_frame = frames[static_cast<std::size_t>(pair)];
    const TestFrame& ref_frame = frames[static_cast<std::size_t>(pair + 1)];
    const MeCoreFrame cur = cur_frame.view();
    const MeCoreRef ref = ref_frame.ref_view();
    const int bw = 64;
    const int bh = 64;
    const auto all = make_block_grid(cur.width, cur.height, bw, bh);
    // Deterministic ~1/4 subset of blocks keeps the CPU reference fast enough
    // for a gate while still covering the whole frame.
    int index = 0;
    for (const MeCoreJob& job : all) {
      ++index;
      if ((index % 2) != 0) {
        continue;
      }
      const Me2Result r =
          prec::me2_search_core(cur, &ref, job, opt, nullptr,
                                MeScalarCost{});
      if (r.ref_index < 0) {
        continue;
      }
      t.blocks += 1;
      t.cost += r.cost;
      t.search_cost += r.search_cost;
      t.mv_bins += r.mv_rate_bins;
      t.side_bins += r.precision_bins;
      t.evals += r.evals;
      t.sel += r.cost + ((static_cast<std::int64_t>(opt.mv_rate_lambda) *
                          r.total_rate_bins) >>
                         (opt.mv_rate_shift < 0 ? 0 : opt.mv_rate_shift));
      t.precision_hist[r.precision] += 1;
      const int w = r.visible_w;
      const int h = r.visible_h;
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const int cv =
              cur.plane[0]
                  .data[static_cast<std::size_t>(job.y + y) *
                            static_cast<std::size_t>(cur.plane[0].stride) +
                        static_cast<std::size_t>(job.x + x)];
          const int pv = me_sample_at(ref.plane[0], job.x + x, job.y + y,
                                      r.mv_x, r.mv_y, opt.search.subpel_interp);
          const int d = cv - pv;
          t.sad += d < 0 ? -d : d;
          t.sse += static_cast<std::int64_t>(d) * d;
        }
      }
      t.samples += static_cast<std::int64_t>(w) * h;
    }
  }
  return t;
}

void print_quality_row(const char* name, const QualityTotals& t,
                       int header_bits, int seq_bits) {
  const double mse =
      t.samples > 0 ? static_cast<double>(t.sse) / static_cast<double>(t.samples)
                    : 0.0;
  const double psnr =
      mse > 0.0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
  const double rate = static_cast<double>(t.mv_bins + t.side_bins);
  std::printf("%-18s SAD=%9lld  PSNR=%6.3f  mv+side=%8.0f  "
              "mv/side=%lld/%lld  searchSAD=%9lld  evals=%8lld  "
              "(hdr=%d+seq%d)\n",
              name, static_cast<long long>(t.sad), psnr, rate,
              static_cast<long long>(t.mv_bins),
              static_cast<long long>(t.side_bins),
              static_cast<long long>(t.search_cost),
              static_cast<long long>(t.evals), header_bits, seq_bits);
}

void test_quality_rate_720p(const std::string& root) {
  const char* clip_env = std::getenv("ME2_Y4M");
  const std::string clip =
      (clip_env != nullptr && clip_env[0] != '\0')
          ? std::string(clip_env)
          : root + "/testdata/shine_1280x720_60f.y4m";
  std::vector<TestFrame> frames;
  if (!load_y4m_frames(clip, 24, &frames) || frames.size() < 4) {
    std::printf("  (%s absent or short; skipping quality/rate report)\n",
                clip.c_str());
    return;
  }
  std::printf("\n== 720p quality/rate proxy (%s, 64x64 grid, every 2nd "
              "block) ==\n", clip.c_str());

  Me2Options quarter = base_options(kMeSubpelHalf | kMeSubpelQuarter, 0);
  quarter.search.method = kMeDiamond;
  quarter.search.range = 16;
  quarter.search.max_iterations = 4;
  quarter.frame_precision = prec::kMvPrecisionEighthPel;  // M2 header

  Me2Options eighth = quarter;
  eighth.search.subpel_stages =
      kMeSubpelHalf | kMeSubpelQuarter | kMeSubpelEighth;

  Me2Options sixteenth = eighth;
  sixteenth.search.subpel_stages |= kMeSubpelSixteenth;
  sixteenth.refine_radius = 2;

  Me2Options adapt_q = sixteenth;
  adapt_q.frame_precision = prec::kMvPrecisionQuarterPel;  // use_qtr = 1
  adapt_q.enable_flex_mvres = 1;
  adapt_q.mv_rate_lambda = 16;

  Me2Options adapt_e = sixteenth;
  adapt_e.frame_precision = prec::kMvPrecisionEighthPel;  // allow_high = 1
  adapt_e.enable_flex_mvres = 1;
  adapt_e.mv_rate_lambda = 16;

  Me2Options adapt_e64 = adapt_e;
  adapt_e64.mv_rate_lambda = 64;

  const int stride = 8;  // frames 0,8,16 -> three pairs
  const QualityTotals tq = measure_config(frames, quarter, stride);
  const QualityTotals te = measure_config(frames, eighth, stride);
  const QualityTotals ts = measure_config(frames, sixteenth, stride);
  const QualityTotals taq = measure_config(frames, adapt_q, stride);
  const QualityTotals tae = measure_config(frames, adapt_e, stride);
  const QualityTotals tae64 = measure_config(frames, adapt_e64, stride);

  print_quality_row("quarter_search", tq, 2, 0);
  print_quality_row("eighth_search", te, 2, 0);
  print_quality_row("sixteenth", ts, 2, 0);
  print_quality_row("flex@quarter", taq, 1, 1);
  print_quality_row("flex@eighth", tae, 2, 1);
  print_quality_row("flex@eighth lam64", tae64, 2, 1);

  std::printf("  precision histogram flex@eighth (8..1/8):");
  for (int p = 0; p < prec::kMvPrecisionCount; ++p) {
    std::printf(" %d", tae.precision_hist[p]);
  }
  std::printf("\n  precision histogram flex@eighth lam64 (8..1/8):");
  for (int p = 0; p < prec::kMvPrecisionCount; ++p) {
    std::printf(" %d", tae64.precision_hist[p]);
  }
  std::printf("\n");
  std::printf("  header bits per P frame: quarter=1, half/eighth=2; "
              "sequence enable_flex_mvres=1 bit\n");

  CHECK(tq.blocks > 0, "720p blocks measured");
  CHECK(ts.sad <= te.sad, "sixteenth SAD <= eighth SAD");
  CHECK(te.sad <= tq.sad, "eighth SAD <= quarter SAD");
  CHECK(ts.search_cost <= te.search_cost,
        "sixteenth search cost <= eighth search cost");
  // Adaptive selection optimises cost + lambda*rate.  Flex adds one
  // use_most_probable_precision bin per NEWMV block versus the fixed path,
  // so allow that unavoidable side information in the comparison.
  const std::int64_t fixed_e_sel16 =
      ts.cost + (static_cast<std::int64_t>(adapt_e.mv_rate_lambda) *
                 (ts.mv_bins + ts.side_bins));
  const std::int64_t fixed_e_sel64 =
      ts.cost + (static_cast<std::int64_t>(adapt_e64.mv_rate_lambda) *
                 (ts.mv_bins + ts.side_bins));
  CHECK(ts.cost == ts.sad, "ME cost equals measured SAD");
  CHECK(tae.sel <= fixed_e_sel16 +
                       static_cast<std::int64_t>(adapt_e.mv_rate_lambda) *
                           ts.blocks,
        "flex@eighth selection cost <= fixed fine path + side info");
  CHECK(tae64.sel <= fixed_e_sel64 +
                         static_cast<std::int64_t>(adapt_e64.mv_rate_lambda) *
                             ts.blocks,
        "flex@eighth lam64 selection cost <= fixed fine path + side info");
  CHECK(tae.blocks == ts.blocks, "same block coverage");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string root =
      argc > 1 ? argv[1] : std::string("../../../../..");
  std::printf("== me2 CPU reference tests ==\n");
  test_frame_precision_bits();
  test_block_precision_syntax();
  test_lower_precision();
  test_rate_proxy();
  test_mc_contract();
  test_refinement_deterministic();
  test_precision_rate_tradeoff();
  test_planted_subpel_shifts();
  test_hill_climb_escape();
  test_partial_blocks_and_guards();
  test_reference_tie_and_determinism();
  test_merge_field_passthrough();
  test_api_validation();
  const char* skip720 = std::getenv("ME2_SKIP_720P");
  if (skip720 == nullptr || skip720[0] == '\0' || skip720[0] == '0') {
    test_quality_rate_720p(root);
  } else {
    std::printf("  (ME2_SKIP_720P set; skipping quality/rate report)\n");
  }

  std::printf("\nchecks=%d failures=%d\n", g_checks, g_failures);
  if (g_failures != 0) {
    std::printf("FAIL\n");
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
