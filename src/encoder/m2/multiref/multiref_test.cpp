// Unit and integration tests for src/encoder/m2/multiref/.
//
//   * reference map / NumTotalRefs / RefFrame pairs
//   * CWP and wedge syntax conversions
//   * compound prediction bit-exactness against the shared MC/blend code
//   * mode decision determinism and quality on synthetic pairs
//   * multi-reference MV stack / compound merge selection
//   * GPU-ready batch equals the per-job path
//   * real 720p frames when testdata/shine_1280x720_60f.y4m is present
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "encoder/inter/mc/mc.h"
#include "encoder/m2/m2_frame.h"
#include "encoder/m2/multiref/compound_predict.h"
#include "encoder/m2/multiref/decide.h"
#include "encoder/m2/multiref/gpu_batch.h"
#include "encoder/m2/multiref/mvstack_multi.h"
#include "encoder/m2/multiref/ref_map.h"
#include "io/y4m.h"

namespace {

using namespace av2enc::m2::multiref;

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

#define CHECK_EQ(a, b)                                                       \
  do {                                                                       \
    ++g_checks;                                                              \
    const auto va = (a);                                                     \
    const auto vb = (b);                                                     \
    if (!(va == vb)) {                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s (%lld vs %lld)\n", __FILE__, \
                   __LINE__, #a, #b, static_cast<long long>(va),              \
                   static_cast<long long>(vb));                              \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// Test reference frames
// ---------------------------------------------------------------------------

struct TestRef {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;

  void resize(int w, int h) {
    width = w;
    height = h;
    y.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
    const int cw = w / 2;
    const int ch = h / 2;
    u.assign(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch), 0);
    v.assign(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch), 0);
  }
  mc::RefFrame view() const {
    mc::RefFrame ref;
    ref.plane[0] = mc::PlaneView{y.data(), width};
    ref.plane[1] = mc::PlaneView{u.data(), width / 2};
    ref.plane[2] = mc::PlaneView{v.data(), width / 2};
    ref.mi_cols = width / 4;
    ref.mi_rows = height / 4;
    return ref;
  }
};

TestRef make_pattern_ref(int width, int height, int seed) {
  TestRef ref;
  ref.resize(width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int v = 32 + ((x * (3 + seed) + y * (5 + 2 * seed) + seed * 17) &
                          0x7f);
      ref.y[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(v);
    }
  }
  const int cw = width / 2;
  const int ch = height / 2;
  for (int p = 1; p < 3; ++p) {
    std::vector<std::uint8_t>& plane = p == 1 ? ref.u : ref.v;
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        plane[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) + static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(
                64 + ((x * (7 + p) + y * (11 - p) + seed * 29) & 0x3f));
      }
    }
  }
  return ref;
}

// ---------------------------------------------------------------------------
// Reference map
// ---------------------------------------------------------------------------

void test_ref_map() {
  RefSlotBank slots{};
  slots[0] = RefSlot{true, true, 0};
  slots[1] = RefSlot{true, false, 1};
  RefFrameMap map;
  std::string error;
  const std::uint8_t idx2[2] = {0, 1};
  CHECK(build_ref_frame_map(slots, idx2, 2, true, 0, map, error));
  CHECK_EQ(map.num_total_refs, 2);
  CHECK_EQ(map.frame_distance[0], 2);
  CHECK_EQ(map.frame_distance[1], 1);
  // Both references are in the past, so is_same_side() is true and the CWP
  // weight row is the "same side" row.
  CHECK(map.is_same_side(0, 1));
  CHECK(map.is_forward(0));
  CHECK(map.is_forward(1));
  CHECK(map.comp_allowed_at(64, 64));
  CHECK(!map.comp_allowed_at(4, 64));

  const auto pairs = enumerate_compound_pairs(map);
  CHECK_EQ(pairs.size(), static_cast<std::size_t>(1));
  CHECK_EQ(pairs[0].ref0, 0);
  CHECK_EQ(pairs[0].ref1, 1);
  CHECK(pairs[0].forced);
  CHECK_EQ(pairs[0].slot0(), 0);
  CHECK_EQ(pairs[0].slot1(), 1);

  // A newer-first encoder policy reverses the map for two slots.
  RefFrameMap recent;
  CHECK(make_recent_ref_map(slots, 2, 2, true, recent, error));
  CHECK_EQ(recent.ref_frame_idx[0], 1);
  CHECK_EQ(recent.ref_frame_idx[1], 0);
  CHECK(recent.is_same_side(0, 1));

  // Three references exercise the coded comp_ref paths.
  const std::uint8_t idx3[3] = {0, 1, 0};
  RefFrameMap map3;
  CHECK(build_ref_frame_map(slots, idx3, 2, true, 0, map3, error));
  const auto pairs3 = enumerate_compound_pairs(map3);
  CHECK_EQ(pairs3.size(), static_cast<std::size_t>(3));
  bool have01 = false;
  bool have02 = false;
  bool have12 = false;
  for (const CompoundRefPair& pair : pairs3) {
    have01 = have01 || (pair.ref0 == 0 && pair.ref1 == 1);
    have02 = have02 || (pair.ref0 == 0 && pair.ref1 == 2);
    have12 = have12 || (pair.ref0 == 1 && pair.ref1 == 2);
    CHECK(!pair.forced);
  }
  CHECK(have01 && have02 && have12);

  // Same-reference compound needs NumSameRefCompound > 0.  With one same-ref
  // slot the reachable set becomes {(1,2), (0,2), (0,0)}: the repeat of ref 0
  // replaces (0,1).
  RefFrameMap same;
  CHECK(build_ref_frame_map(slots, idx3, 2, true, 1, same, error));
  const auto same_pairs = enumerate_compound_pairs(same);
  CHECK_EQ(same_pairs.size(), static_cast<std::size_t>(4));
  bool have00 = false;
  for (const CompoundRefPair& pair : same_pairs) {
    have00 = have00 || (pair.ref0 == 0 && pair.ref1 == 0);
  }
  CHECK(have00);

  // Invalid: empty slot and out-of-range slot.
  RefSlotBank empty{};
  RefFrameMap unused;
  CHECK(!build_ref_frame_map(empty, idx2, 1, true, 0, unused, error));
  const std::uint8_t bad[1] = {2};
  CHECK(!build_ref_frame_map(slots, bad, 1, true, 0, unused, error));

  const ReferenceSelectSyntax syntax = reference_select_syntax(map);
  CHECK(syntax.reference_select);
  CHECK_EQ(syntax.num_total_refs, 2);
  CHECK_EQ(syntax.reference_map_bits(1), 1 + 3 + 2 * 1 + 1);
}

// ---------------------------------------------------------------------------
// CWP / wedge syntax helpers
// ---------------------------------------------------------------------------

void test_mode_helpers() {
  for (int row = 0; row <= 1; ++row) {
    for (int i = 0; i < kMaxCwpNum; ++i) {
      const int w = cwp_weight_from_index(i, row != 0);
      CHECK_EQ(cwp_index_from_weight(w, row != 0), i);
    }
  }
  CHECK_EQ(cwp_weight_from_index(1, false), 12);
  CHECK_EQ(cwp_weight_from_index(1, true), 12);
  CHECK_EQ(cwp_weight_from_index(3, false), 10);
  CHECK_EQ(cwp_weight_from_index(3, true), 20);
  CHECK_EQ(cwp_weight_from_index(4, true), -4);
  CHECK_EQ(cwp_index_from_weight(3, false), -1);

  int signalable = 0;
  for (int index = 0; index < adv::kWedgeTypes; ++index) {
    WedgeSyntax syntax;
    const bool ok = wedge_index_to_syntax(index, syntax);
    CHECK(ok);
    if (!ok) continue;
    CHECK_EQ(wedge_syntax_to_index(syntax.quad, syntax.angle, syntax.dist),
             index);
    CHECK(wedge_syntax_bits(index) > 0);
    if (adv::wedge_index_signalable(index)) ++signalable;
  }
  CHECK_EQ(signalable, adv::kWedgeTypes);
  CHECK_EQ(wedge_syntax_to_index(0, 0, 0), -1);  // angle 0/dist 0 is invalid

  CompoundModeParams average;
  average.type = adv::CompoundType::kAverage;
  average.cwp_weight = 12;
  CHECK(compound_mode_bits(average, false) > compound_mode_bits(average, true));
  CompoundModeParams wedge;
  wedge.type = adv::CompoundType::kWedge;
  wedge.wedge_index = 22;
  CHECK(compound_mode_bits(wedge, true) > 0);
  CHECK(same_mode(wedge, wedge));
  CompoundModeParams other = wedge;
  other.wedge_sign = 1;
  CHECK(!same_mode(wedge, other));
}

// ---------------------------------------------------------------------------
// Compound prediction and blending
// ---------------------------------------------------------------------------

void test_compound_prediction() {
  TestRef ref0 = make_pattern_ref(64, 64, 0);
  TestRef ref1 = make_pattern_ref(64, 64, 1);
  const mc::BlockRect rect{0, 0, 64, 64};
  CompoundPrediction prediction;
  std::string error;
  CHECK(build_compound_prediction(ref0.view(), ref1.view(), rect,
                                  static_cast<int>(12), mc::Mv{0, 0},
                                  mc::Mv{0, 0}, mc::InterpFilter::kEighttap,
                                  16, 16, prediction, error));
  CHECK_EQ(prediction.pred[0][0].samples.size(),
           static_cast<std::size_t>(64 * 64));
  CHECK_EQ(prediction.pred[0][1].samples.size(),
           static_cast<std::size_t>(32 * 32));

  // Integer MV -> 1/16 prediction is exactly sample << 4 for both planes.
  for (int i = 0; i < 64 * 64; ++i) {
    CHECK_EQ(static_cast<int>(prediction.pred[0][0].samples[static_cast<std::size_t>(i)]),
             static_cast<int>(ref0.y[static_cast<std::size_t>(i)]) << 4);
    CHECK_EQ(static_cast<int>(prediction.pred[1][0].samples[static_cast<std::size_t>(i)]),
             static_cast<int>(ref1.y[static_cast<std::size_t>(i)]) << 4);
  }
  for (int i = 0; i < 32 * 32; ++i) {
    CHECK_EQ(static_cast<int>(prediction.pred[0][1].samples[static_cast<std::size_t>(i)]),
             static_cast<int>(ref0.u[static_cast<std::size_t>(i)]) << 4);
    CHECK_EQ(static_cast<int>(prediction.pred[1][1].samples[static_cast<std::size_t>(i)]),
             static_cast<int>(ref1.u[static_cast<std::size_t>(i)]) << 4);
  }

  std::vector<std::uint8_t> dst_y(64 * 64), dst_u(32 * 32), dst_v(32 * 32);
  std::uint8_t* dst[3] = {dst_y.data(), dst_u.data(), dst_v.data()};
  int strides[3] = {64, 32, 32};

  // AVERAGE, CWP and the implicit mask.
  for (int cw = 0; cw <= 16; cw += 4) {
    CompoundModeParams params;
    params.type = adv::CompoundType::kAverage;
    params.cwp_weight = cw;
    CHECK(apply_compound_prediction(prediction, params, dst, strides, error));
    for (int i = 0; i < 64 * 64; ++i) {
      const int p0 = static_cast<int>(ref0.y[static_cast<std::size_t>(i)]) << 4;
      const int p1 = static_cast<int>(ref1.y[static_cast<std::size_t>(i)]) << 4;
      const int expect = (cw * p0 + (16 - cw) * p1 + 128) >> 8;
      CHECK_EQ(static_cast<int>(dst_y[static_cast<std::size_t>(i)]), expect);
    }
  }
  {
    CompoundModeParams params;
    params.type = adv::CompoundType::kAverage;
    params.cwp_weight = kCwpEqual;
    params.use_imp_mask = true;
    CHECK(apply_compound_prediction(prediction, params, dst, strides, error));
    for (int i = 0; i < 64 * 64; ++i) {
      const int p0 = static_cast<int>(ref0.y[static_cast<std::size_t>(i)]) << 4;
      const int p1 = static_cast<int>(ref1.y[static_cast<std::size_t>(i)]) << 4;
      // Both references are fully on-screen -> get_mask() == 1.
      const int expect = (p0 + p1 + 16) >> 5;
      CHECK_EQ(static_cast<int>(dst_y[static_cast<std::size_t>(i)]), expect);
    }
  }

  // WEDGE and DIFFWTD against the shared mask/blend transcription.
  {
    std::vector<std::uint8_t> mask(64 * 64);
    CompoundModeParams params;
    params.type = adv::CompoundType::kWedge;
    params.wedge_index = 22;
    params.wedge_sign = 1;
    adv::build_wedge_mask(static_cast<int>(12), params.wedge_sign,
                          params.wedge_index, mask.data(), 64);
    CHECK(apply_compound_prediction(prediction, params, dst, strides, error));
    for (int i = 0; i < 64 * 64; ++i) {
      const int p0 = static_cast<int>(ref0.y[static_cast<std::size_t>(i)]) << 4;
      const int p1 = static_cast<int>(ref1.y[static_cast<std::size_t>(i)]) << 4;
      const int expect =
          (mask[static_cast<std::size_t>(i)] * p0 +
           (64 - mask[static_cast<std::size_t>(i)]) * p1 + 512) >>
          10;
      CHECK_EQ(static_cast<int>(dst_y[static_cast<std::size_t>(i)]), expect);
    }
    // Chroma uses the subsampled luma mask.
    for (int y = 0; y < 32; ++y) {
      for (int x = 0; x < 32; ++x) {
        const std::size_t m00 =
            static_cast<std::size_t>(2 * y) * 64 + static_cast<std::size_t>(2 * x);
        const int m = (mask[m00] + mask[m00 + 1] + mask[m00 + 64] +
                       mask[m00 + 65] + 2) >> 2;
        const std::size_t i =
            static_cast<std::size_t>(y) * 32 + static_cast<std::size_t>(x);
        const int p0 = static_cast<int>(ref0.u[i]) << 4;
        const int p1 = static_cast<int>(ref1.u[i]) << 4;
        const int expect = (m * p0 + (64 - m) * p1 + 512) >> 10;
        CHECK_EQ(static_cast<int>(dst_u[i]), expect);
      }
    }
  }
  {
    CompoundModeParams params;
    params.type = adv::CompoundType::kDiffWtd;
    params.mask_type = 1;
    CHECK(apply_compound_prediction(prediction, params, dst, strides, error));
    for (int i = 0; i < 64 * 64; ++i) {
      const int p0 = static_cast<int>(ref0.y[static_cast<std::size_t>(i)]) << 4;
      const int p1 = static_cast<int>(ref1.y[static_cast<std::size_t>(i)]) << 4;
      const int m = adv::diffwtd_mask_value(p0, p1, params.mask_type, 4, 8);
      const int expect = (m * p0 + (64 - m) * p1 + 512) >> 10;
      CHECK_EQ(static_cast<int>(dst_y[static_cast<std::size_t>(i)]), expect);
    }
  }
}

// ---------------------------------------------------------------------------
// Decision API
// ---------------------------------------------------------------------------

DecisionInput make_decision_input(const TestRef& ref0, const TestRef& ref1,
                                  const TestRef& source, const TestRef& other) {
  (void)other;
  DecisionInput input;
  input.refs[0] = nullptr;  // set below; the struct stores pointers
  input.bsize = 12;
  input.luma_rect = mc::BlockRect{0, 0, 64, 64};
  input.mi_cols = 16;
  input.mi_rows = 16;
  input.source = source.y.data();
  input.source_stride = 64;
  input.options.lambda = 0.0;
  input.options.seq_enable_imp_msk_bld = true;
  CompoundCandidateInput candidate;
  candidate.ref0 = 0;
  candidate.ref1 = 1;
  candidate.mv[0] = mc::Mv{0, 0};
  candidate.mv[1] = mc::Mv{0, 0};
  candidate.global_global_mv = true;
  candidate.has_prediction = true;
  std::string error;
  CHECK(build_compound_prediction(ref0.view(), ref1.view(), input.luma_rect,
                                  input.bsize, mc::Mv{0, 0}, mc::Mv{0, 0},
                                  mc::InterpFilter::kEighttap, 16, 16,
                                  candidate.prediction, error));
  input.compound_candidates.push_back(candidate);
  return input;
}

void test_decision() {
  TestRef ref0 = make_pattern_ref(64, 64, 0);
  TestRef ref1 = make_pattern_ref(64, 64, 1);

  // 3/4 * ref0 + 1/4 * ref1 -> CWP weight 12.
  TestRef source;
  source.resize(64, 64);
  for (int i = 0; i < 64 * 64; ++i) {
    const int s = (3 * static_cast<int>(ref0.y[static_cast<std::size_t>(i)]) +
                   static_cast<int>(ref1.y[static_cast<std::size_t>(i)]) + 2) /
                  4;
    source.y[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(s);
  }
  DecisionInput input = make_decision_input(ref0, ref1, source, ref1);
  DecisionResult result;
  std::string error;
  CHECK(decide_compound(input, result, error));
  CHECK_EQ(static_cast<int>(result.kind),
           static_cast<int>(DecisionKind::kCompound));
  CHECK_EQ(result.ref0, 0);
  CHECK_EQ(result.ref1, 1);
  CHECK(result.compound.type == adv::CompoundType::kAverage);
  CHECK_EQ(result.compound.cwp_weight, 12);

  // Determinism.
  DecisionResult again;
  CHECK(decide_compound(input, again, error));
  CHECK(same_mode(result.compound, again.compound));
  CHECK_EQ(result.sse, again.sse);
  CHECK_EQ(result.rate_bits, again.rate_bits);

  // Implicit-mask path with CWP disabled: the decoder only has the get_mask()
  // average, so the decision must return use_imp_mask = true (not plain equal).
  {
    DecisionInput imp_input = make_decision_input(ref0, ref1, source, ref1);
    imp_input.options.enable_cwp = false;
    imp_input.options.enable_wedge = false;
    imp_input.options.enable_diffwtd = false;
    imp_input.compound_candidates[0].global_global_mv = false;
    DecisionResult imp_result;
    CHECK(decide_compound(imp_input, imp_result, error));
    CHECK(imp_result.kind == DecisionKind::kCompound);
    CHECK(imp_result.compound.type == adv::CompoundType::kAverage);
    CHECK_EQ(imp_result.compound.cwp_weight, kCwpEqual);
    CHECK(imp_result.compound.use_imp_mask);
  }

  // Source produced by a wedge blend -> WEDGE must be selected with SSE 0.
  std::vector<std::uint8_t> wedge_y(64 * 64), wedge_u(32 * 32), wedge_v(32 * 32);
  std::uint8_t* wedge_dst[3] = {wedge_y.data(), wedge_u.data(), wedge_v.data()};
  int wedge_strides[3] = {64, 32, 32};
  CompoundPrediction wedge_prediction;
  CHECK(build_compound_prediction(ref0.view(), ref1.view(),
                                  mc::BlockRect{0, 0, 64, 64}, 12,
                                  mc::Mv{0, 0}, mc::Mv{0, 0},
                                  mc::InterpFilter::kEighttap, 16, 16,
                                  wedge_prediction, error));
  CompoundModeParams wedge_source;
  wedge_source.type = adv::CompoundType::kWedge;
  wedge_source.wedge_index = 22;
  wedge_source.wedge_sign = 1;
  CHECK(apply_compound_prediction(wedge_prediction, wedge_source, wedge_dst,
                                  wedge_strides, error));
  TestRef wedge_ref;
  wedge_ref.resize(64, 64);
  wedge_ref.y = wedge_y;
  DecisionInput wedge_input = make_decision_input(ref0, ref1, wedge_ref, ref1);
  DecisionResult wedge_result;
  CHECK(decide_compound(wedge_input, wedge_result, error));
  CHECK(wedge_result.compound.type == adv::CompoundType::kWedge);
  CHECK(adv::wedge_index_signalable(wedge_result.compound.wedge_index));
  CHECK_EQ(wedge_result.sse, 0);

  // Source produced by the diff-weighted blend -> DIFFWTD must win with SSE 0.
  CompoundModeParams diff;
  diff.type = adv::CompoundType::kDiffWtd;
  diff.mask_type = 0;
  std::vector<std::uint8_t> dst_y(64 * 64), dst_u(32 * 32), dst_v(32 * 32);
  std::uint8_t* dst[3] = {dst_y.data(), dst_u.data(), dst_v.data()};
  int strides[3] = {64, 32, 32};
  CompoundPrediction prediction;
  CHECK(build_compound_prediction(ref0.view(), ref1.view(),
                                  mc::BlockRect{0, 0, 64, 64}, 12,
                                  mc::Mv{0, 0}, mc::Mv{0, 0},
                                  mc::InterpFilter::kEighttap, 16, 16,
                                  prediction, error));
  CHECK(apply_compound_prediction(prediction, diff, dst, strides, error));
  TestRef diff_source;
  diff_source.resize(64, 64);
  diff_source.y = dst_y;
  DecisionInput diff_input = make_decision_input(ref0, ref1, diff_source, ref1);
  DecisionResult diff_result;
  CHECK(decide_compound(diff_input, diff_result, error));
  CHECK(diff_result.compound.type == adv::CompoundType::kDiffWtd);
  CHECK_EQ(diff_result.sse, 0);

  // Single-reference candidate can win when it is better than any compound.
  TestRef same_as_ref1 = ref1;
  DecisionInput single_input = make_decision_input(ref0, ref1, same_as_ref1, ref1);
  SingleCandidateInput single;
  single.ref = 1;
  single.mode = 1;
  single.drl_index = 0;
  single.sse = 0;
  single_input.single_candidates.push_back(single);
  DecisionResult single_result;
  CHECK(decide_compound(single_input, single_result, error));
  CHECK_EQ(static_cast<int>(single_result.kind),
           static_cast<int>(DecisionKind::kSingle));
  CHECK_EQ(single_result.ref0, 1);

  // evaluate_compound_modes covers every enabled family and reports SSE.
  std::vector<EvaluatedMode> modes;
  evaluate_compound_modes(prediction, nullptr, 0, 1, wedge_ref.y.data(), 64,
                          input.options, modes);
  bool saw_average = false;
  bool saw_cwp = false;
  bool saw_wedge = false;
  bool saw_diff = false;
  int64_t best_wedge = -1;
  for (const EvaluatedMode& mode : modes) {
    CHECK(mode.sse >= 0);
    if (mode.params.type == adv::CompoundType::kAverage) {
      saw_average = true;
      if (mode.params.cwp_weight != kCwpEqual) saw_cwp = true;
    } else if (mode.params.type == adv::CompoundType::kWedge) {
      saw_wedge = true;
      best_wedge = best_wedge < 0 ? mode.sse : std::min(best_wedge, mode.sse);
    } else if (mode.params.type == adv::CompoundType::kDiffWtd) {
      saw_diff = true;
    }
  }
  CHECK(saw_average && saw_cwp && saw_wedge && saw_diff);
  CHECK(best_wedge >= 0);
}

// ---------------------------------------------------------------------------
// MV stack / merge
// ---------------------------------------------------------------------------

void test_mv_stack() {
  constexpr int kMiCols = 16;
  constexpr int kMiRows = 16;
  std::vector<mc::MiInfo> grid(
      static_cast<std::size_t>(kMiCols) * kMiRows, mc::MiInfo{});
  // Neighbour block: 16x16 at (0,0) with a non-zero MV.
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      mc::MiInfo& mi = grid[static_cast<std::size_t>(r) * kMiCols + static_cast<std::size_t>(c)];
      mi.ref[0] = 0;
      mi.ref[1] = 1;
      mi.mv[0][0] = 8;
      mi.mv[0][1] = -8;
      mi.mv[1][0] = -8;
      mi.mv[1][1] = 8;
      mi.is_inter = 1;
      mi.w4 = 4;
      mi.h4 = 4;
      mi.col_base = 0;
      mi.row_base = 0;
    }
  }
  merge::StackParams params;
  params.mi_row = 0;
  params.mi_col = 8;
  params.bw4 = 4;
  params.bh4 = 4;
  params.sb_h4 = 16;
  params.mi_cols = kMiCols;
  params.mi_rows = kMiRows;
  params.mi = grid.data();
  params.reorder = merge::DrlReorder::kDisabled;
  RefStackPair stacks;
  std::string error;
  CHECK(build_pair_stacks(params, 0, 1, stacks, error));
  CHECK(stacks.ok[0] && stacks.ok[1] && stacks.compound_ok);
  CHECK(stacks.list[0].count > 0);
  CHECK(stacks.list[1].count > 0);
  CHECK(stacks.compound.count > 0);

  const CompoundMergeSet set = build_compound_merge_candidates(stacks);
  CHECK(!set.candidates.empty());
  const CompoundMergeSet set2 = build_compound_merge_candidates(stacks);
  CHECK_EQ(set.candidates.size(), set2.candidates.size());
  for (std::size_t i = 0; i < set.candidates.size(); ++i) {
    CHECK_EQ(set.candidates[i].mv[0].y, set2.candidates[i].mv[0].y);
    CHECK_EQ(set.candidates[i].mv[0].x, set2.candidates[i].mv[0].x);
    CHECK_EQ(set.candidates[i].drl_index[0], set2.candidates[i].drl_index[0]);
    CHECK_EQ(set.candidates[i].drl_index[1], set2.candidates[i].drl_index[1]);
  }

  TestRef ref0 = make_pattern_ref(64, 64, 0);
  TestRef ref1 = make_pattern_ref(64, 64, 1);
  const mc::RefFrame refs[2] = {ref0.view(), ref1.view()};
  std::vector<std::uint8_t> scratch(16 * 16);
  MergeScore score;
  CHECK(select_compound_merge_candidate(set, refs, mc::BlockRect{32, 0, 16, 16},
                                        6, kMiCols, kMiRows, ref0.y.data() + 32,
                                        64, scratch.data(), 16, score, error));
  CHECK(score.candidate_index >= 0);
  CHECK(score.sse >= 0);
  MergeScore score2;
  CHECK(select_compound_merge_candidate(set, refs, mc::BlockRect{32, 0, 16, 16},
                                        6, kMiCols, kMiRows, ref0.y.data() + 32,
                                        64, scratch.data(), 16, score2, error));
  CHECK_EQ(score.candidate_index, score2.candidate_index);

  RefMergeScore single;
  CHECK(select_best_ref_merge(stacks, refs, mc::BlockRect{32, 0, 16, 16},
                              mc::InterpFilter::kEighttap,
                              ref0.y.data() + 32, 64, scratch.data(), 16,
                              single, error));
  CHECK(single.ref >= 0);
  // §7.12.2.20 MVP-combination gate lives in the shared merge component.
  CHECK(!merge::mvp_combinations_allowed(16, 16));
  CHECK(merge::mvp_combinations_allowed(64, 64));
}

// ---------------------------------------------------------------------------
// GPU-ready batch
// ---------------------------------------------------------------------------

void test_gpu_batch() {
  TestRef ref0 = make_pattern_ref(64, 64, 0);
  TestRef ref1 = make_pattern_ref(64, 64, 1);
  CompoundPrediction prediction;
  std::string error;
  CHECK(build_compound_prediction(ref0.view(), ref1.view(),
                                  mc::BlockRect{0, 0, 64, 64}, 12,
                                  mc::Mv{0, 0}, mc::Mv{0, 0},
                                  mc::InterpFilter::kEighttap, 16, 16,
                                  prediction, error));

  struct Job {
    CompoundModeParams params;
    mc::BlockRect rect;
  };
  std::vector<Job> jobs(3);
  jobs[0].params.type = adv::CompoundType::kAverage;
  jobs[0].params.cwp_weight = 12;
  jobs[0].rect = mc::BlockRect{0, 0, 32, 32};
  jobs[1].params.type = adv::CompoundType::kWedge;
  jobs[1].params.wedge_index = 22;
  jobs[1].params.wedge_sign = 1;
  jobs[1].rect = mc::BlockRect{32, 0, 32, 32};
  jobs[2].params.type = adv::CompoundType::kDiffWtd;
  jobs[2].params.mask_type = 1;
  jobs[2].rect = mc::BlockRect{0, 32, 64, 32};

  // Reference result through the per-job path.
  std::vector<std::uint8_t> ref_y(64 * 64), ref_u(32 * 32), ref_v(32 * 32);
  for (std::size_t j = 0; j < jobs.size(); ++j) {
    CompoundPrediction sub;
    CHECK(build_compound_prediction(ref0.view(), ref1.view(), jobs[j].rect, 12,
                                    mc::Mv{0, 0}, mc::Mv{0, 0},
                                    mc::InterpFilter::kEighttap, 16, 16, sub,
                                    error));
    std::uint8_t* dst[3] = {ref_y.data(), ref_u.data(), ref_v.data()};
    int strides[3] = {64, 32, 32};
    CHECK(apply_compound_prediction(sub, jobs[j].params, dst, strides, error));
  }

  CompoundBatch batch;
  batch.reset(64, 64);
  for (std::size_t j = 0; j < jobs.size(); ++j) {
    CompoundPrediction sub;
    CHECK(build_compound_prediction(ref0.view(), ref1.view(), jobs[j].rect, 12,
                                    mc::Mv{0, 0}, mc::Mv{0, 0},
                                    mc::InterpFilter::kEighttap, 16, 16, sub,
                                    error));
    int job_mask_offset = 0;
    int job_mask_stride = 0;
    for (int plane = 0; plane < 3; ++plane) {
      const int sub_x = plane > 0 ? 1 : 0;
      const int sub_y = plane > 0 ? 1 : 0;
      const mc::BlockRect rect{jobs[j].rect.x >> sub_x,
                               jobs[j].rect.y >> sub_y,
                               jobs[j].rect.w >> sub_x,
                               jobs[j].rect.h >> sub_y};
      const std::size_t samples =
          static_cast<std::size_t>(rect.w) * static_cast<std::size_t>(rect.h);
      const int offset = batch.alloc_pred(samples);
      const PredPlane16& a = sub.pred[0][plane];
      const PredPlane16& b = sub.pred[1][plane];
      std::memcpy(batch.pred0.data() + offset, a.samples.data(),
                  a.samples.size() * sizeof(std::int16_t));
      std::memcpy(batch.pred1.data() + offset, b.samples.data(),
                  b.samples.size() * sizeof(std::int16_t));
      CompoundBatchItem item;
      item.plane = plane;
      item.rect_x = rect.x;
      item.rect_y = rect.y;
      item.rect_w = rect.w;
      item.rect_h = rect.h;
      item.bsize = 12;
      item.sub_x = sub_x;
      item.sub_y = sub_y;
      item.p0_offset = offset;
      item.p0_stride = a.stride;
      item.p1_offset = offset;
      item.p1_stride = b.stride;
      item.dst_offset = batch.plane_dst_offset[plane];
      item.dst_stride = batch.plane_stride[plane];
      item.type = static_cast<int>(jobs[j].params.type);
      item.cwp_weight = jobs[j].params.cwp_weight;
      item.wedge_index = jobs[j].params.wedge_index;
      item.wedge_sign = jobs[j].params.wedge_sign;
      item.mask_type = jobs[j].params.mask_type;
      item.mi_cols = 16;
      item.mi_rows = 16;
      if (plane == 0) {
        job_mask_offset = batch.alloc_mask(samples);
        job_mask_stride = rect.w;
        item.mask_offset = job_mask_offset;
        item.mask_stride = job_mask_stride;
      } else if (jobs[j].params.type == adv::CompoundType::kWedge ||
                 jobs[j].params.type == adv::CompoundType::kDiffWtd) {
        item.luma_mask_offset = job_mask_offset;
        item.luma_mask_stride = job_mask_stride;
      }
      batch.items.push_back(item);
    }
  }

  if (!validate_compound_batch(batch, error)) {
    std::fprintf(stderr, "  batch validate: %s\n", error.c_str());
  }
  CHECK(validate_compound_batch(batch, error));
  if (!run_compound_batch_cpu(batch, error)) {
    std::fprintf(stderr, "  batch run: %s\n", error.c_str());
  }
  CHECK(run_compound_batch_cpu(batch, error));

  std::uint64_t mismatch = 0;
  for (std::size_t i = 0; i < ref_y.size(); ++i) {
    if (batch.dst[i] != ref_y[i]) ++mismatch;
  }
  CHECK_EQ(mismatch, 0);

  // Invalid batch is rejected.
  CompoundBatch broken = batch;
  broken.items[0].p0_offset = static_cast<int>(broken.pred0.size());
  CHECK(!validate_compound_batch(broken, error));
}

// ---------------------------------------------------------------------------
// Real 720p frames
// ---------------------------------------------------------------------------

void test_real_720p() {
  const char* path = "testdata/shine_1280x720_60f.y4m";
  av2enc::Y4mReader reader(path);
  std::string error;
  if (!reader.open(error)) {
    std::printf("  (720p media absent; skipped)\n");
    return;
  }
  av2enc::Y4mFrame y4m;
  std::vector<std::vector<std::uint8_t>> y;
  std::vector<std::vector<std::uint8_t>> u;
  std::vector<std::vector<std::uint8_t>> v;
  int width = 0;
  int height = 0;
  for (int i = 0; i < 2; ++i) {
    if (reader.read_frame(y4m, error) != av2enc::Y4mReadResult::kFrame) {
      CHECK(false);
      return;
    }
    width = static_cast<int>(y4m.width());
    height = static_cast<int>(y4m.height());
    y.emplace_back(y4m.y().begin(), y4m.y().end());
    u.emplace_back(y4m.u().begin(), y4m.u().end());
    v.emplace_back(y4m.v().begin(), y4m.v().end());
  }
  std::vector<av2enc::m2::M2InputFrame> inputs(2);
  for (int i = 0; i < 2; ++i) {
    inputs[static_cast<std::size_t>(i)].width = static_cast<std::uint32_t>(width);
    inputs[static_cast<std::size_t>(i)].height =
        static_cast<std::uint32_t>(height);
    inputs[static_cast<std::size_t>(i)].y = y[static_cast<std::size_t>(i)];
    inputs[static_cast<std::size_t>(i)].u = u[static_cast<std::size_t>(i)];
    inputs[static_cast<std::size_t>(i)].v = v[static_cast<std::size_t>(i)];
  }
  av2enc::m2::M2Options options;
  options.force_single_block = true;
  options.enable_intra_fallback = false;
  av2enc::m2::M2SequenceResult sequence;
  CHECK(av2enc::m2::encode_m2_sequence(inputs, options, sequence, error));
  if (sequence.recon.size() != 2) return;

  const av2enc::m2::M2ReconFrame& ref_a = sequence.recon[0];
  const av2enc::m2::M2ReconFrame& ref_b = sequence.recon[1];
  mc::RefFrame refs[2];
  refs[0].plane[0] = mc::PlaneView{ref_a.y.data(), width};
  refs[0].plane[1] = mc::PlaneView{ref_a.u.data(), width / 2};
  refs[0].plane[2] = mc::PlaneView{ref_a.v.data(), width / 2};
  refs[0].mi_cols = width / 4;
  refs[0].mi_rows = height / 4;
  refs[1].plane[0] = mc::PlaneView{ref_b.y.data(), width};
  refs[1].plane[1] = mc::PlaneView{ref_b.u.data(), width / 2};
  refs[1].plane[2] = mc::PlaneView{ref_b.v.data(), width / 2};
  refs[1].mi_cols = width / 4;
  refs[1].mi_rows = height / 4;

  // Sample a grid of interior 64x64 blocks.
  int blocks = 0;
  int compound_wins = 0;
  double sse_compound = 0.0;
  double sse_single = 0.0;
  std::vector<std::uint8_t> scratch(64 * 64);
  for (int by = 0; by + 64 <= height; by += 128) {
    for (int bx = 0; bx + 64 <= width; bx += 192) {
      CompoundCandidateInput candidate;
      candidate.ref0 = 0;
      candidate.ref1 = 1;
      candidate.global_global_mv = true;
      candidate.has_prediction = false;
      DecisionInput input;
      input.refs[0] = &refs[0];
      input.refs[1] = &refs[1];
      input.bsize = 12;
      input.luma_rect = mc::BlockRect{bx, by, 64, 64};
      input.mi_cols = width / 4;
      input.mi_rows = height / 4;
      input.options.lambda = 0.0;
      input.options.seq_enable_imp_msk_bld = true;
      const std::size_t src_offset =
          static_cast<std::size_t>(by) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(bx);
      input.source = y[1].data() + src_offset;
      input.source_stride = width;
      input.compound_candidates.push_back(candidate);
      SingleCandidateInput single;
      single.ref = 1;
      single.mode = 0;
      const mc::BlockRect rect{bx, by, 64, 64};
      CHECK(mc::predict_plane(refs[1], 0, rect, mc::Mv{0, 0},
                              mc::InterpFilter::kEighttap, scratch.data(),
                              64) == mc::McStatus::kOk);
      single.sse = adv::block_sse(scratch.data(), 64, 0, 0, y[1].data() + src_offset,
                                  width, 0, 0, 64, 64);
      input.single_candidates.push_back(single);

      DecisionResult result;
      if (!decide_compound(input, result, error)) {
        CHECK(false);
        continue;
      }
      ++blocks;
      if (result.kind == DecisionKind::kCompound && result.sse < single.sse) {
        ++compound_wins;
      }
      sse_compound += static_cast<double>(
          std::min(result.sse, single.sse));
      sse_single += static_cast<double>(single.sse);
    }
  }
  CHECK(blocks > 0);
  std::printf(
      "  720p blocks=%d compound_beats_single=%d best_sse=%.0f single_sse=%.0f "
      "(%.2f dB proxy gain)\n",
      blocks, compound_wins, sse_compound, sse_single,
      10.0 * std::log10(sse_single / std::max(1.0, sse_compound)));
}

}  // namespace

int main() {
  std::printf("== multiref tests ==\n");
  std::printf(" ref map...\n");
  test_ref_map();
  std::printf(" mode helpers...\n");
  test_mode_helpers();
  std::printf(" compound prediction...\n");
  test_compound_prediction();
  std::printf(" decision...\n");
  test_decision();
  std::printf(" mv stack...\n");
  test_mv_stack();
  std::printf(" gpu batch...\n");
  test_gpu_batch();
  std::printf(" real 720p...\n");
  test_real_720p();
  std::printf("== %d checks, %d failure(s) ==\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
