// CPU reference tests for the AV2 inter motion-estimation component.
//
// Coverage:
//   * SAD and SATD arithmetic (flat, random, integer and subpel positions)
//   * normative 8-tap subpel interpolation (integer, half, quarter phases)
//   * full search, diamond search, TZ-style search on known synthetic shifts
//   * half/quarter-pel refinement to exact fractional matches
//   * partial blocks at the right/bottom frame edge
//   * multiple references and reference selection
//   * explicit MV predictors and spatial merge candidates
//   * batch determinism and guard-canary checks (no out-of-bounds reads)
//   * real 256x256 and 1280x720 frames from testdata/
//   * API validation paths
//
// Usage: ./me_test [repo_root]
// The repo root is used to locate testdata/.  If omitted, it is derived from
// the source file path.  Missing real media only skips those cases.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "encoder/inter/me/me.h"
#include "encoder/inter/me/me_test_common.h"

namespace {

using namespace av2enc::me;
using namespace av2enc::me::test;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
    }                                                                      \
  } while (0)

MeCoreOptions base_options() {
  MeCoreOptions o;
  o.method = kMeFull;
  o.metric = kMeSad;
  o.range = 8;
  o.subpel_stages = kMeSubpelNone;
  o.use_merge_candidates = 0;
  return o;
}

std::vector<MeCoreResult> run_batch(const TestFrame& cur,
                                    const std::vector<TestFrame>& ref_frames,
                                    const std::vector<MeCoreJob>& jobs,
                                    const MeCoreOptions& opts,
                                    const MeCoreMvField* field = nullptr) {
  std::vector<MeCoreRef> refs;
  refs.reserve(ref_frames.size());
  for (const TestFrame& f : ref_frames) {
    refs.push_back(f.ref_view());
  }
  std::vector<MeCoreResult> results(jobs.size());
  const MeCoreFrame cv = cur.view();
  const MeStatus s =
      me_search_batch(cv, refs.data(), static_cast<int>(refs.size()),
                      jobs.data(), static_cast<int>(jobs.size()), opts, field,
                      results.data());
  CHECK(s == MeStatus::kOk, "me_search_batch failed");
  return results;
}

std::int32_t cost_of(const TestFrame& cur, const TestFrame& ref, int plane,
                     int x, int y, int w, int h, std::int32_t mv_x,
                     std::int32_t mv_y, int metric, int interp) {
  MeCostJob job;
  job.plane = plane;
  job.x = x;
  job.y = y;
  job.w = w;
  job.h = h;
  job.mv_x = mv_x;
  job.mv_y = mv_y;
  job.metric = metric;
  job.interp = interp;
  std::int32_t c = -1;
  const MeCoreFrame cv = cur.view();
  const MeCoreRef rv = ref.ref_view();
  const MeStatus s = me_cost_block(cv, rv, job, &c);
  CHECK(s == MeStatus::kOk, "me_cost_block failed");
  return c;
}

void test_sad_basic() {
  TestFrame cur;
  cur.init(64, 48);
  fill_flat(&cur, 100, 128);
  TestFrame ref = cur;
  CHECK(cost_of(cur, ref, 0, 8, 8, 16, 16, 0, 0, kMeSad,
                kMeInterpRegular) == 0,
        "identical blocks must have zero SAD");
  fill_flat(&ref, 110, 128);
  const std::int32_t sad =
      cost_of(cur, ref, 0, 8, 8, 16, 16, 0, 0, kMeSad, kMeInterpRegular);
  CHECK(sad == 10 * 16 * 16, "flat SAD must equal |delta| * area");

  // Known integer shift: ref(x,y) = cur(x-3,y+2); MV (48,-32) is exact.
  TestFrame cur2;
  cur2.init(64, 48);
  fill_random(&cur2, 12345u);
  TestFrame ref2 = make_shifted_ref(cur2, 3, -2);
  const std::int32_t shifted =
      cost_of(cur2, ref2, 0, 12, 12, 16, 16, 48, -32, kMeSad, kMeInterpRegular);
  CHECK(shifted == 0, "integer shifted block must have zero SAD");
  const std::int32_t other =
      cost_of(cur2, ref2, 0, 12, 12, 16, 16, 0, 0, kMeSad, kMeInterpRegular);
  CHECK(other > 0, "wrong MV must have non-zero SAD");
  std::cout << "  SAD: flat and shifted cases\n";
}

void test_satd_basic() {
  TestFrame cur;
  cur.init(64, 48);
  fill_flat(&cur, 100, 128);
  TestFrame ref;
  ref.init(64, 48);
  fill_flat(&ref, 116, 128);
  const std::int32_t satd =
      cost_of(cur, ref, 0, 0, 0, 16, 16, 0, 0, kMeSatd, kMeInterpRegular);
  CHECK(satd == 16 * 16 * 16 / 16, "flat 8x8 SATD normalisation");
  const std::int32_t sad =
      cost_of(cur, ref, 0, 0, 0, 16, 16, 0, 0, kMeSad, kMeInterpRegular);
  CHECK(satd == (sad + 8) >> 4, "flat SATD must be SAD/16 for 8x8 tiles");

  // Random residual: SATD is nonzero and integer/subpel costs agree at integer
  // positions.
  TestFrame rc;
  rc.init(64, 48);
  fill_random(&rc, 777u);
  TestFrame rr = make_shifted_ref(rc, 1, 1);
  const std::int32_t a =
      cost_of(rc, rr, 0, 8, 8, 32, 32, 0, 0, kMeSatd, kMeInterpRegular);
  const std::int32_t b =
      cost_of(rc, rr, 0, 8, 8, 32, 32, 0, 0, kMeSatd, kMeInterpBilinear);
  CHECK(a > 0, "random SATD must be nonzero");
  CHECK(a == b, "integer MV cost must ignore the subpel filter");

  // Partial 8x8 tile padding: a 12x12 block still works.
  const std::int32_t c =
      cost_of(rc, rr, 0, 8, 8, 12, 12, 0, 0, kMeSatd, kMeInterpRegular);
  CHECK(c > 0, "partial-block SATD must be nonzero");
  std::cout << "  SATD: flat normalisation, integer parity, partial tiles\n";
}

void test_interp_integer_and_ramp() {
  TestFrame ref;
  ref.init(64, 48);
  fill_ramp(&ref, 2, 1);

  // Integer MV copies samples exactly.
  std::vector<std::uint8_t> out(16 * 16);
  MeInterpJob ij;
  ij.plane = 0;
  ij.x = 8;
  ij.y = 8;
  ij.w = 16;
  ij.h = 16;
  ij.mv_x = 0;
  ij.mv_y = 0;
  ij.interp = kMeInterpRegular;
  CHECK(me_interp_block(ref.ref_view(), ij, out.data()) == MeStatus::kOk,
        "integer interp block failed");
  bool exact = true;
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      exact = exact && out[static_cast<std::size_t>(y) * 16 + static_cast<std::size_t>(x)] ==
                           ref.y.row(8 + y)[8 + x];
    }
  }
  CHECK(exact, "integer MV must copy reference samples");

  // Half-pel on a linear ramp: symmetric 8-tap yields the exact midpoint.
  ij.mv_x = 8;
  ij.mv_y = 0;
  CHECK(me_interp_block(ref.ref_view(), ij, out.data()) == MeStatus::kOk,
        "half-pel interp block failed");
  bool ramp_ok = true;
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      const int expected = (ref.y.row(8 + y)[8 + x] + 1) & 0xFF;
      ramp_ok = ramp_ok &&
                out[static_cast<std::size_t>(y) * 16 + static_cast<std::size_t>(x)] ==
                    static_cast<std::uint8_t>(expected);
    }
  }
  CHECK(ramp_ok, "half-pel on a 2-sample ramp must be the midpoint");

  // Quarter-pel: +0.25 sample on a 2/sample ramp rounds to +1 LSB.
  ij.mv_x = 4;
  CHECK(me_interp_block(ref.ref_view(), ij, out.data()) == MeStatus::kOk,
        "quarter-pel interp block failed");
  bool quarter_ok = true;
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      const int expected = (ref.y.row(8 + y)[8 + x] + 1) & 0xFF;
      quarter_ok = quarter_ok &&
                   out[static_cast<std::size_t>(y) * 16 + static_cast<std::size_t>(x)] ==
                       static_cast<std::uint8_t>(expected);
    }
  }
  CHECK(quarter_ok, "quarter-pel ramp rounding");
  std::cout << "  interp: integer copy, half/quarter ramp phases\n";
}

void test_mv_candidates() {
  std::vector<std::int32_t> mvx(16, 0);
  std::vector<std::int32_t> mvy(16, 0);
  std::vector<std::int16_t> refs(16, 0);
  std::vector<std::uint8_t> valid(16, 0);
  // 4x4 grid of 4x4 MI cells.
  // A1 cell (1,3), B1 cell (3,1), B0 (4,1) off-grid, A0 (1,4) off-grid,
  // B2 (1,1).
  valid[3 * 4 + 1] = 1;
  mvx[3 * 4 + 1] = 16;
  mvy[3 * 4 + 1] = 0;
  valid[1 * 4 + 3] = 1;
  mvx[1 * 4 + 3] = 16;  // duplicate of A1
  mvy[1 * 4 + 3] = 0;
  valid[1 * 4 + 1] = 1;
  mvx[1 * 4 + 1] = 0;
  mvy[1 * 4 + 1] = 8;
  refs[1 * 4 + 1] = 1;

  MeCoreMvField f;
  f.mv_x = mvx.data();
  f.mv_y = mvy.data();
  f.ref_index = refs.data();
  f.valid = valid.data();
  f.cols = 4;
  f.rows = 4;

  MeMvCandidate out[8];
  const int n = me_collect_mv_candidates(f, 8, 8, 8, 8, 0, out, 8);
  CHECK(n == 3, "merge list must dedup and append zero");
  CHECK(out[0].mv_x == 16 && out[0].mv_y == 0 && out[0].ref_index == 0,
        "A1 first");
  CHECK(out[1].mv_x == 0 && out[1].mv_y == 8 && out[1].ref_index == 1,
        "B2 second, different ref kept");
  CHECK(out[2].mv_x == 0 && out[2].mv_y == 0 && out[2].ref_index == 0,
        "zero appended last");

  MeCoreMvField empty;
  const int n0 = me_collect_mv_candidates(empty, 8, 8, 8, 8, 2, out, 8);
  CHECK(n0 == 1 && out[0].mv_x == 0 && out[0].mv_y == 0 &&
            out[0].ref_index == 2,
        "empty field yields the zero candidate");
  std::cout << "  MVP: AV1 scan order, dedup, zero fallback\n";
}

void test_full_search_shift() {
  TestFrame cur;
  cur.init(96, 96);
  fill_random(&cur, 0xC0FFEEu);
  TestFrame ref = make_shifted_ref(cur, 3, -4);

  MeCoreOptions o = base_options();
  o.range = 8;
  std::vector<MeCoreJob> jobs(1);
  jobs[0].x = 16;
  jobs[0].y = 24;
  jobs[0].w = 32;
  jobs[0].h = 32;
  jobs[0].num_refs = 1;
  jobs[0].ref_indices[0] = 0;
  const std::vector<MeCoreResult> r = run_batch(cur, {ref}, jobs, o);
  CHECK(r[0].mv_x == 3 * 16 && r[0].mv_y == -4 * 16,
        "full search must find the exact shift");
  CHECK(r[0].cost == 0, "exact shift must cost zero");
  CHECK(r[0].evals > 0, "search must count evaluations");
  CHECK(r[0].stage == kMeStageInteger, "full search stage");
  CHECK(r[0].visible_w == 32 && r[0].visible_h == 32,
        "full block visibility");
  CHECK(cur.guard_intact() && ref.guard_intact(), "no guard writes");

  // Degenerate candidate cap: must coarsen safely (no division by zero) and
  // still return a valid result.
  o.max_candidates = 1;
  o.range = 2;
  const std::vector<MeCoreResult> capped = run_batch(cur, {ref}, jobs, o);
  CHECK(capped[0].visible_w == 32 && capped[0].visible_h == 32,
        "capped full search returns a valid block");
  CHECK(capped[0].evals >= 1, "capped full search counts evaluations");
  std::cout << "  full search: exact 3,-4 shift; candidate cap\n";
}

void test_diamond_and_tz() {
  TestFrame cur;
  cur.init(96, 96);
  fill_smooth_texture(&cur, 0xBEEF01u, 5);
  TestFrame ref = make_shifted_ref(cur, -5, 6);

  MeCoreOptions o = base_options();
  o.range = 8;
  o.max_iterations = 16;
  MeCoreJob job;
  job.x = 24;
  job.y = 16;
  job.w = 32;
  job.h = 32;
  job.num_refs = 1;
  job.ref_indices[0] = 0;

  o.method = kMeDiamond;
  auto r = run_batch(cur, {ref}, {job}, o);
  CHECK(r[0].mv_x == -5 * 16 && r[0].mv_y == 6 * 16,
        "diamond search must find the exact shift");
  CHECK(r[0].cost == 0, "diamond exact cost");

  o.method = kMeTestZone;
  r = run_batch(cur, {ref}, {job}, o);
  CHECK(r[0].mv_x == -5 * 16 && r[0].mv_y == 6 * 16,
        "TZ search must find the exact shift");
  CHECK(r[0].cost == 0, "TZ exact cost");

  // TZ with SATD.
  o.metric = kMeSatd;
  r = run_batch(cur, {ref}, {job}, o);
  CHECK(r[0].mv_x == -5 * 16 && r[0].mv_y == 6 * 16,
        "TZ+SATD must find the exact shift");
  std::cout << "  diamond/TZ: exact shifts, SAD and SATD\n";
}

// cur(x,y) = interp(ref, x + fx/16, y + fy/16) using the normative filter.
TestFrame make_fractional_cur(const TestFrame& ref, int fx, int fy,
                              int interp) {
  TestFrame cur;
  cur.init(ref.width, ref.height);
  for (int p = 0; p < 3; ++p) {
    const TestPlane& src = p == 0 ? ref.y : (p == 1 ? ref.u : ref.v);
    TestPlane& dst = p == 0 ? cur.y : (p == 1 ? cur.u : cur.v);
    for (int y = 0; y < dst.height; ++y) {
      for (int x = 0; x < dst.width; ++x) {
        const int xq = (x << kSubpelBits) + fx;
        const int yq = (y << kSubpelBits) + fy;
        dst.row(y)[x] =
            static_cast<std::uint8_t>(me_interp_sample(src.view(), xq, yq, interp));
      }
    }
  }
  return cur;
}

void test_subpel_refinement() {
  TestFrame ref;
  ref.init(96, 96);
  fill_random(&ref, 0x51B2E1u);

  MeCoreOptions o = base_options();
  o.method = kMeFull;
  o.range = 4;
  MeCoreJob job;
  job.x = 24;
  job.y = 24;
  job.w = 32;
  job.h = 32;
  job.num_refs = 1;
  job.ref_indices[0] = 0;

  {
    const TestFrame cur = make_fractional_cur(ref, 8, 0, kMeInterpRegular);
    o.subpel_stages = kMeSubpelHalf;
    auto r = run_batch(cur, {ref}, {job}, o);
    CHECK(r[0].mv_x == 8 && r[0].mv_y == 0,
          "half-pel refinement must find +8/16");
    CHECK(r[0].cost == 0, "half-pel exact cost");
    CHECK(r[0].stage == kMeStageHalf, "half-pel stage reported");
    CHECK(cur.guard_intact(), "subpel no guard writes");
  }
  {
    const TestFrame cur = make_fractional_cur(ref, 4, 0, kMeInterpRegular);
    o.subpel_stages = kMeSubpelQuarter;
    auto r = run_batch(cur, {ref}, {job}, o);
    CHECK(r[0].mv_x == 4 && r[0].mv_y == 0,
          "quarter-pel refinement must find +4/16");
    CHECK(r[0].cost == 0, "quarter-pel exact cost");
  }
  {
    const TestFrame cur = make_fractional_cur(ref, 8, -8, kMeInterpRegular);
    o.subpel_stages = kMeSubpelHalf;
    auto r = run_batch(cur, {ref}, {job}, o);
    CHECK(r[0].mv_x == 8 && r[0].mv_y == -8,
          "diagonal half-pel refinement");
    CHECK(r[0].cost == 0, "diagonal half-pel exact cost");
  }
  {
    // Quarter-only stage should still reach a half-pel match by two steps.
    const TestFrame cur = make_fractional_cur(ref, 8, 0, kMeInterpRegular);
    o.subpel_stages = kMeSubpelQuarter;
    o.subpel_iterations = 4;
    auto r = run_batch(cur, {ref}, {job}, o);
    CHECK(r[0].mv_x == 8 && r[0].mv_y == 0,
          "quarter stage can accumulate to half-pel");
    CHECK(r[0].cost == 0, "accumulated quarter exact cost");
  }
  std::cout << "  subpel: half/quarter/diagonal exact matches\n";
}

void test_partial_blocks() {
  TestFrame cur;
  cur.init(40, 24);
  fill_random(&cur, 0x3344u);
  TestFrame ref = make_shifted_ref(cur, -4, -2);

  MeCoreOptions o = base_options();
  o.range = 4;
  std::vector<MeCoreJob> jobs(2);
  jobs[0].x = 32;
  jobs[0].y = 16;
  jobs[0].w = 32;
  jobs[0].h = 32;
  jobs[0].num_refs = 1;
  jobs[1] = jobs[0];
  jobs[1].x = 36;
  jobs[1].y = 20;
  jobs[1].w = 8;
  jobs[1].h = 8;
  const std::vector<MeCoreResult> r = run_batch(cur, {ref}, jobs, o);
  CHECK(r[0].visible_w == 8 && r[0].visible_h == 8,
        "right/bottom partial block clipping");
  CHECK(r[0].mv_x == -4 * 16 && r[0].mv_y == -2 * 16,
        "partial block exact shift");
  CHECK(r[0].cost == 0, "partial block exact cost");
  CHECK(r[1].visible_w == 4 && r[1].visible_h == 4,
        "small partial block clipping");
  CHECK(r[1].cost == 0, "tiny partial block exact cost");
  CHECK(cur.guard_intact() && ref.guard_intact(),
        "partial blocks must not read guard samples");
  std::cout << "  partial blocks: 8x8 and 4x4 visible regions\n";
}

void test_multiple_refs() {
  TestFrame cur;
  cur.init(64, 64);
  fill_random(&cur, 0x99AAu);
  TestFrame bad;
  bad.init(64, 64);
  fill_flat(&bad, 128, 128);
  TestFrame good = make_shifted_ref(cur, 2, 3);

  MeCoreOptions o = base_options();
  o.range = 6;
  MeCoreJob job;
  job.x = 16;
  job.y = 16;
  job.w = 16;
  job.h = 16;
  job.num_refs = 2;
  job.ref_indices[0] = 0;
  job.ref_indices[1] = 1;
  const std::vector<MeCoreResult> r = run_batch(cur, {bad, good}, {job}, o);
  CHECK(r[0].ref_index == 1, "must select the matching reference");
  CHECK(r[0].mv_x == 2 * 16 && r[0].mv_y == 3 * 16,
        "MV for the selected reference");
  CHECK(r[0].cost == 0, "selected reference exact cost");

  // Reference order controls ties: make both refs identical, first wins.
  const std::vector<MeCoreResult> r2 = run_batch(cur, {good, good}, {job}, o);
  CHECK(r2[0].ref_index == 0, "ties resolve to the first listed reference");
  std::cout << "  refs: selection and tie order\n";
}

void test_predictors_and_merge() {
  TestFrame cur;
  cur.init(64, 64);
  fill_random(&cur, 0x123456u);
  TestFrame ref = make_shifted_ref(cur, 4, -2);

  MeCoreOptions o = base_options();
  o.range = 0;  // predictor only
  MeCoreJob job;
  job.x = 16;
  job.y = 16;
  job.w = 16;
  job.h = 16;
  job.num_refs = 1;
  job.ref_indices[0] = 0;
  job.num_pred = 2;
  job.pred_mv[0][0] = 0;
  job.pred_mv[0][1] = 0;
  job.pred_mv[1][0] = 4 * 16;
  job.pred_mv[1][1] = -2 * 16;
  auto r = run_batch(cur, {ref}, {job}, o);
  CHECK(r[0].mv_x == 4 * 16 && r[0].mv_y == -2 * 16,
        "explicit predictor selection");
  CHECK(r[0].cost == 0, "explicit predictor exact cost");
  CHECK(r[0].stage == kMeStagePredictor, "predictor stage");

  // Merge candidate from the MI field supplies the same vector.
  std::vector<std::int32_t> mvx(16 * 16, 0);
  std::vector<std::int32_t> mvy(16 * 16, 0);
  std::vector<std::uint8_t> valid(16 * 16, 1);
  const int cell = 3 * 16 + 3;  // (12,12), left of the 16x16 block at 16,16
  mvx[static_cast<std::size_t>(cell)] = 4 * 16;
  mvy[static_cast<std::size_t>(cell)] = -2 * 16;
  MeCoreMvField field;
  field.mv_x = mvx.data();
  field.mv_y = mvy.data();
  field.valid = valid.data();
  field.cols = 16;
  field.rows = 16;

  job.num_pred = 0;
  o.use_merge_candidates = 1;
  o.max_merge_candidates = 4;
  r = run_batch(cur, {ref}, {job}, o, &field);
  CHECK(r[0].mv_x == 4 * 16 && r[0].mv_y == -2 * 16,
        "merge candidate selection");
  CHECK(r[0].cost == 0, "merge candidate exact cost");
  std::cout << "  predictors: explicit MVP and spatial merge\n";
}

void test_batch_determinism() {
  TestFrame cur;
  cur.init(80, 64);
  fill_random(&cur, 0xABCDEFu);
  TestFrame ref = make_shifted_ref(cur, -2, 2);
  MeCoreOptions o = base_options();
  o.method = kMeTestZone;
  o.range = 8;
  o.metric = kMeSatd;
  o.subpel_stages = kMeSubpelHalf | kMeSubpelQuarter;
  const std::vector<MeCoreJob> jobs = make_block_grid(80, 64, 16, 16);
  const auto a = run_batch(cur, {ref}, jobs, o);
  const auto b = run_batch(cur, {ref}, jobs, o);
  bool same = a.size() == b.size();
  for (std::size_t i = 0; same && i < a.size(); ++i) {
    same = a[i].mv_x == b[i].mv_x && a[i].mv_y == b[i].mv_y &&
           a[i].ref_index == b[i].ref_index && a[i].cost == b[i].cost &&
           a[i].evals == b[i].evals;
  }
  CHECK(same, "batch results must be deterministic");
  std::cout << "  batch: " << jobs.size() << " jobs deterministic\n";
}

void test_real_frames(const std::string& repo_root) {
  std::vector<TestFrame> frames;
  const std::string still =
      repo_root + "/testdata/shine_256x256_1f.y4m";
  if (!load_y4m_frames(still, 1, &frames)) {
    std::cout << "  real 256x256: skipped (no testdata)\n";
  } else {
    TestFrame cur = std::move(frames[0]);
    TestFrame ref = make_shifted_ref(cur, 5, -3);
    MeCoreOptions o = base_options();
    o.method = kMeTestZone;
    o.range = 8;
    std::vector<MeCoreJob> jobs;
    for (int y = 32; y + 32 <= cur.height; y += 32) {
      for (int x = 0; x + 32 <= cur.width - 5; x += 32) {
        MeCoreJob j;
        j.x = x;
        j.y = y;
        j.w = 32;
        j.h = 32;
        j.num_refs = 1;
        j.ref_indices[0] = 0;
        jobs.push_back(j);
      }
    }
    const auto r = run_batch(cur, {ref}, jobs, o);
    int exact_mv = 0;
    int zero_cost = 0;
    bool no_worse = true;
    for (std::size_t i = 0; i < r.size(); ++i) {
      const std::int32_t zero = cost_of(cur, ref, 0, jobs[i].x, jobs[i].y,
                                        jobs[i].w, jobs[i].h, 0, 0, kMeSad,
                                        kMeInterpRegular);
      if (r[i].mv_x == 5 * 16 && r[i].mv_y == -3 * 16) {
        ++exact_mv;
      }
      if (r[i].cost == 0) {
        ++zero_cost;
      }
      no_worse = no_worse && r[i].cost <= zero;
    }
    CHECK(no_worse, "search result must never be worse than the zero predictor");
    CHECK(zero_cost * 10 >= static_cast<int>(r.size()) * 9,
          "real 256x256: >=90% of shifted blocks reach zero cost");
    CHECK(cur.guard_intact() && ref.guard_intact(),
          "real 256x256: no guard writes");
    std::cout << "  real 256x256: " << zero_cost << "/" << r.size()
              << " blocks zero-cost (" << exact_mv << " exact MV)\n";
  }

  frames.clear();
  const std::string video = repo_root + "/testdata/shine_1280x720_60f.y4m";
  if (!load_y4m_frames(video, 2, &frames) || frames.size() < 2) {
    std::cout << "  real 720p: skipped (no testdata)\n";
  } else {
    TestFrame& cur = frames[0];
    TestFrame& ref = frames[1];
    MeCoreOptions o = base_options();
    o.method = kMeTestZone;
    o.range = 16;
    o.metric = kMeSad;
    o.subpel_stages = kMeSubpelHalf | kMeSubpelQuarter;
    const std::vector<MeCoreJob> jobs = make_block_grid(1280, 720, 16, 16);
    const auto r = run_batch(cur, {ref}, jobs, o);
    int improved = 0;
    bool no_worse = true;
    std::int64_t sum_found = 0;
    std::int64_t sum_zero = 0;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      const std::int32_t zero = cost_of(cur, ref, 0, jobs[i].x, jobs[i].y,
                                        jobs[i].w, jobs[i].h, 0, 0, kMeSad,
                                        kMeInterpRegular);
      sum_found += r[i].cost;
      sum_zero += zero;
      if (r[i].cost < zero) {
        ++improved;
      }
      no_worse = no_worse && r[i].cost <= zero;
    }
    CHECK(no_worse, "real 720p: search must not lose to the zero predictor");
    CHECK(sum_found <= sum_zero, "real 720p aggregate cost improves or ties");
    CHECK(cur.guard_intact() && ref.guard_intact(),
          "real 720p: no guard writes");
    std::cout << "  real 720p: " << improved << "/" << jobs.size()
              << " blocks improve on zero MV (aggregate " << sum_found
              << " vs " << sum_zero << ")\n";
  }
}

void test_validation() {
  TestFrame cur;
  cur.init(32, 32);
  fill_flat(&cur, 100, 100);
  TestFrame ref = cur;
  const MeCoreFrame cv = cur.view();
  const MeCoreRef rv = ref.ref_view();
  const MeCoreRef refs[1] = {rv};
  MeCoreJob job;
  job.x = 0;
  job.y = 0;
  job.w = 16;
  job.h = 16;
  job.num_refs = 1;
  job.ref_indices[0] = 0;
  MeCoreResult res;
  MeCoreOptions o = base_options();

  o.range = -1;
  CHECK(me_search_batch(cv, refs, 1, &job, 1, o, nullptr, &res) ==
            MeStatus::kInvalidArgument,
        "negative range rejected");
  o = base_options();
  job.ref_indices[0] = 3;
  CHECK(me_search_batch(cv, refs, 1, &job, 1, o, nullptr, &res) ==
            MeStatus::kInvalidArgument,
        "bad ref index rejected");
  job.ref_indices[0] = 0;
  job.w = 0;
  CHECK(me_search_batch(cv, refs, 1, &job, 1, o, nullptr, &res) ==
            MeStatus::kInvalidArgument,
        "zero block width rejected");
  job.w = 16;

  MeCostJob cj;
  cj.plane = 0;
  cj.x = 0;
  cj.y = 0;
  cj.w = 16;
  cj.h = 16;
  cj.mv_x = 16 * 32;
  cj.mv_y = 0;
  cj.metric = kMeSad;
  cj.interp = kMeInterpRegular;
  std::int32_t c = 0;
  CHECK(me_cost_block(cv, rv, cj, &c) == MeStatus::kInvalidArgument,
        "out-of-range MV rejected");
  cj.mv_x = 0;
  cj.w = 0;
  CHECK(me_cost_block(cv, rv, cj, &c) == MeStatus::kInvalidArgument,
        "zero cost width rejected");
  std::cout << "  validation: bad options/refs/MVs rejected\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string repo_root;
  if (argc > 1) {
    repo_root = argv[1];
  } else {
    repo_root = ".";
  }
  std::cout << "AV2 inter ME CPU reference tests\n";
  test_sad_basic();
  test_satd_basic();
  test_interp_integer_and_ramp();
  test_mv_candidates();
  test_full_search_shift();
  test_diamond_and_tz();
  test_subpel_refinement();
  test_partial_blocks();
  test_multiple_refs();
  test_predictors_and_merge();
  test_batch_determinism();
  test_real_frames(repo_root);
  test_validation();

  std::cout << "\nchecks: " << g_checks << ", failures: " << g_failures
            << "\n";
  return g_failures == 0 ? 0 : 1;
}
