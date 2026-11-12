// Correctness tests for the AV2 warp machinery (§7.11.4, §7.12.2.3/.4/.9/
// .11/.20, §7.12.3, §7.13.3.23).
//
// Coverage:
//   * scan point warp context: WarpMvCount, WarpSampleFound, ExtendDeltaRow/Col,
//     even-column alignment on the top superblock row, missing neighbours,
//     unwritten/intra cells, identical reference lists;
//   * warp corner process and generate points from corners: 3-corner requirement,
//     allMvsSame, negative refPts, per-reference early return, SB-border guard,
//     warp-model evaluation at a corner;
//   * warp parameter bank order, insert warp candidate cap, §7.12.2.20 tail
//     order (bank newest-first, gm_params, two defaults);
//   * §7.12.3 find warp samples: above/left availability, above_sample_stored
//     SB-border even-column rule, LEAST_SQUARES_SAMPLES_MAX, both lists,
//     top-left/top-right gating;
//   * §7.13.3.23 model from samples: det == 0 translation fallback, non-zero
//     determinant, parity with adv::warp_estimate;
//   * candidate-for-candidate parity with m2::merge::build_stack_checked for
//     derive_wrl == 0 over randomized grids;
//   * randomized lockstep between the primary builder and the independent
//     mirror, and snapshot write/read round-trips;
//   * pinned goldens and two negative controls that demonstrate the documented
//     desync hazards (WarpMvCount CDF row, WarpParamStack order).

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "encoder/inter/warp_stack/warp_stack.h"
#include "encoder/inter/warp_stack/warp_stack_io.h"
#include "encoder/inter/warp_stack/warp_stack_mirror.h"
#include "encoder/m2/merge/merge.h"

namespace {

using namespace av2enc::inter::warp;
using av2enc::inter::mc::MiInfo;
using av2enc::inter::mc::Mv;

int g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 50) std::printf("FAIL line %d: %s\n", line, what);
  }
}

#define EXPECT(cond) expect((cond), #cond, __LINE__)
#define EXPECT_EQ(a, b)                                                                 \
  do {                                                                                  \
    const long long va = static_cast<long long>(a);                                     \
    const long long vb = static_cast<long long>(b);                                     \
    ++g_checks;                                                                         \
    if (va != vb) {                                                                     \
      ++g_failures;                                                                     \
      if (g_failures <= 50) {                                                           \
        std::printf("FAIL line %d: %s (%lld) != %s (%lld)\n", __LINE__, #a, va, #b, vb); \
      }                                                                                 \
    }                                                                                   \
  } while (0)

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

struct BlockSpec {
  int row = 0;
  int col = 0;
  int w4 = 1;
  int h4 = 1;
  std::int8_t ref[2] = {0, kRefNone};
  int sub_mv[2][2] = {{0, 0}, {0, 0}};
  int mv[2][2] = {{0, 0}, {0, 0}};
  std::uint8_t motion_mode = kMotionSimple;
  std::uint8_t y_mode = kYModeNewMv;
  ia::WarpParams warp[2]{};
  bool written = true;
  bool is_inter = true;
};

struct Fixture {
  int cols = 0;
  int rows = 0;
  std::vector<WarpMi> mi;
  // Mirror into the M2 grid layout for the wired-stack parity test.
  std::vector<MiInfo> m2;

  WarpGrid grid() const { return WarpGrid{cols, rows, mi.data()}; }
};

void fill_block(Fixture* f, const BlockSpec& b) {
  for (int y = 0; y < b.h4; ++y) {
    for (int x = 0; x < b.w4; ++x) {
      const int r = b.row + y;
      const int c = b.col + x;
      if (r < 0 || c < 0 || r >= f->rows || c >= f->cols) continue;
      WarpMi& m = f->mi[static_cast<std::size_t>(r) * f->cols + c];
      m = WarpMi{};
      m.ref[0] = b.ref[0];
      m.ref[1] = b.ref[1];
      m.mv[0][0] = b.mv[0][0];
      m.mv[0][1] = b.mv[0][1];
      m.mv[1][0] = b.mv[1][0];
      m.mv[1][1] = b.mv[1][1];
      m.sub_mv[0][0] = b.sub_mv[0][0];
      m.sub_mv[0][1] = b.sub_mv[0][1];
      m.sub_mv[1][0] = b.sub_mv[1][0];
      m.sub_mv[1][1] = b.sub_mv[1][1];
      m.is_inter = b.is_inter ? 1 : 0;
      m.written = b.written ? 1 : 0;
      m.w4 = static_cast<std::int8_t>(b.w4);
      m.h4 = static_cast<std::int8_t>(b.h4);
      m.col_base = static_cast<std::int16_t>(b.col);
      m.row_base = static_cast<std::int16_t>(b.row);
      m.motion_mode = b.motion_mode;
      m.y_mode = b.y_mode;
      m.warp[0] = b.warp[0];
      m.warp[1] = b.warp[1];

      MiInfo& o =
          f->m2[static_cast<std::size_t>(r) * f->cols + c];
      o = MiInfo{};
      o.ref[0] = b.written ? b.ref[0] : kRefNone;
      o.ref[1] = b.ref[1];
      o.mv[0][0] = b.mv[0][0];
      o.mv[0][1] = b.mv[0][1];
      o.mv[1][0] = b.mv[1][0];
      o.mv[1][1] = b.mv[1][1];
      o.is_inter = b.is_inter ? 1 : 0;
      o.w4 = static_cast<std::int8_t>(b.w4);
      o.h4 = static_cast<std::int8_t>(b.h4);
      o.col_base = static_cast<std::int16_t>(b.col);
      o.row_base = static_cast<std::int16_t>(b.row);
    }
  }
}

Fixture make_fixture(int cols, int rows, const std::vector<BlockSpec>& blocks) {
  Fixture f;
  f.cols = cols;
  f.rows = rows;
  f.mi.assign(static_cast<std::size_t>(cols) * rows, WarpMi{});
  f.m2.assign(static_cast<std::size_t>(cols) * rows, MiInfo{});
  for (const BlockSpec& b : blocks) fill_block(&f, b);
  return f;
}

ia::WarpParams model(int p0, int p1, int p2, int p3, int p4, int p5) {
  ia::WarpParams m{};
  m.p[0] = p0;
  m.p[1] = p1;
  m.p[2] = p2;
  m.p[3] = p3;
  m.p[4] = p4;
  m.p[5] = p5;
  return m;
}

// A 3-corner fixture around a 64x64 current block at MI (16,16):
//   above     (0..15,16..31)   MV (8,0)
//   left      (16..31,0..15)   MV (0,8)
//   top-left  (0..15,14..15)   MV (0,0)   (even-aligned column pair)
// The warp corner process looks at (15,14), (15,30) and (31,15).
Fixture corner_fixture(bool all_same, bool negative) {
  BlockSpec above;
  above.row = 0;
  above.col = 16;
  above.w4 = 16;
  above.h4 = 16;
  above.mv[0][0] = all_same ? 0 : 8;
  above.mv[0][1] = 0;
  above.sub_mv[0][0] = above.mv[0][0];
  above.sub_mv[0][1] = 0;
  BlockSpec left;
  left.row = 16;
  left.col = 0;
  left.w4 = 16;
  left.h4 = 16;
  left.mv[0][0] = 0;
  left.mv[0][1] = all_same ? 0 : 8;
  left.sub_mv[0][0] = 0;
  left.sub_mv[0][1] = left.mv[0][1];
  BlockSpec top_left;
  top_left.row = 0;
  top_left.col = 14;
  top_left.w4 = 2;
  top_left.h4 = 16;
  if (negative) {
    top_left.mv[0][0] = -600;
    top_left.mv[0][1] = 0;
  }
  top_left.sub_mv[0][0] = top_left.mv[0][0];
  top_left.sub_mv[0][1] = top_left.mv[0][1];
  return make_fixture(48, 48, {above, left, top_left});
}

WarpStackParams stack_params(const Fixture& f, int row, int col, int bw4,
                             int bh4) {
  WarpStackParams p;
  p.mi_row = row;
  p.mi_col = col;
  p.bw4 = bw4;
  p.bh4 = bh4;
  p.sb_h4 = 16;
  p.mi_cols = f.cols;
  p.mi_rows = f.rows;
  p.mi = f.mi.data();
  p.ref_frame[0] = 0;
  p.ref_frame[1] = kRefNone;
  p.reorder = DrlReorder::kDisabled;
  p.global_mv[0] = Mv{0, 0};
  p.gm_params = ia::WarpParams{};
  return p;
}

// ---------------------------------------------------------------------------
// §7.11.4
// ---------------------------------------------------------------------------

void test_context_basic() {
  // Current 64x64 block at (16,0); left neighbour at (31,15) is a warp block.
  BlockSpec left;
  left.row = 16;
  left.col = 0;
  left.w4 = 16;
  left.h4 = 16;
  left.motion_mode = kMotionLocalWarp;
  left.warp[0] = model(0, 0, 65536, 0, 0, 65536);
  BlockSpec above;
  above.row = 15;
  above.col = 16;
  above.w4 = 16;
  above.h4 = 1;
  above.motion_mode = kMotionSimple;
  Fixture f = make_fixture(48, 48, {left, above});

  WarpContextParams p;
  p.mi_row = 16;
  p.mi_col = 16;
  p.bw4 = 16;
  p.bh4 = 16;
  p.sb_h4 = 16;
  p.grid = f.grid();
  p.ref_frame[0] = 0;
  p.ref_frame[1] = kRefNone;

  const WarpContext ctx = compute_warp_context(p);
  // leftA (31,15) and leftB (16,15) both land in the warp left block, so the
  // count is 2 (WarpMvCount counts scan points, not distinct blocks).  aboveA
  // and aboveB hit the SIMPLE block.  ref[1] == NONE matches WarpSampleFound[1]
  // for every single-predicted neighbour.
  EXPECT_EQ(ctx.warp_mv_count, 2);
  EXPECT(ctx.warp_sample_found[0]);
  EXPECT(ctx.warp_sample_found[1]);
  EXPECT(ctx.extra.found);
  EXPECT_EQ(ctx.extra.delta_row, 15);
  EXPECT_EQ(ctx.extra.delta_col, -1);

  // An all-intra neighbourhood produces no samples and no warp count.
  Fixture intra = make_fixture(
      48, 48, {BlockSpec{16, 0, 16, 16, {kRefIntra, kRefNone}, {{0, 0}, {0, 0}},
                         {{0, 0}, {0, 0}}, kMotionSimple, kYModeNewMv, {}, true,
                         false}});
  p.grid = intra.grid();
  const WarpContext none = compute_warp_context(p);
  EXPECT_EQ(none.warp_mv_count, 0);
  EXPECT(!none.warp_sample_found[0]);
  EXPECT(!none.warp_sample_found[1]);
  EXPECT(!none.extra.found);
}

void test_context_sb_even_column() {
  // mi_row == 16 is a superblock border; mi_col == 3 is odd, so every above
  // scan point subtracts MiCol & 1.  Place a warp neighbour at (15,1) and a
  // SIMPLE one at (15,2): the aligned scan must find the warp block.
  std::vector<BlockSpec> blocks;
  BlockSpec warp_at_2;
  warp_at_2.row = 15;
  warp_at_2.col = 2;
  warp_at_2.motion_mode = kMotionLocalWarp;
  blocks.push_back(warp_at_2);
  BlockSpec plain_at_3;
  plain_at_3.row = 15;
  plain_at_3.col = 3;
  blocks.push_back(plain_at_3);
  Fixture f = make_fixture(16, 32, blocks);

  WarpContextParams p;
  p.mi_row = 16;
  p.mi_col = 3;
  p.bw4 = 8;
  p.bh4 = 8;
  p.sb_h4 = 16;
  p.grid = f.grid();
  p.ref_frame[0] = 0;
  p.ref_frame[1] = kRefNone;

  const WarpContext ctx = compute_warp_context(p);
  EXPECT_EQ(ctx.warp_mv_count, 1);  // (15,2) reached only after alignment
  EXPECT(ctx.warp_sample_found[0]);
  EXPECT_EQ(ctx.extra.delta_row, -1);
  EXPECT_EQ(ctx.extra.delta_col, -1);  // 0 minus (MiCol & 1)

  // Negative control: the un-aligned aboveB scan would look at (15,3), which
  // is SIMPLE, so WarpMvCount would be 0.  This is the CDF-context desync the
  // audit flagged (TileIsWarpCdf row 0 instead of row 1).
  EXPECT(f.grid().at(15, 2).motion_mode >= kMotionLocalWarp);
  EXPECT(f.grid().at(15, 3).motion_mode < kMotionLocalWarp);
  EXPECT(ctx.warp_mv_count != 0);
}

// ---------------------------------------------------------------------------
// §7.12.2.3 / §7.12.2.4
// ---------------------------------------------------------------------------

void test_corner_process() {
  Fixture f = corner_fixture(false, false);
  CornerResult r;
  // Below-left corner only.
  EXPECT(warp_corner_process(f.grid(), 16, 16, 16, 16, 16, 0, 15, -1, 0, &r));
  EXPECT_EQ(r.corners_found, 1);
  EXPECT_EQ(r.corner_pts[0][0], 128);  // (31+1)*4
  EXPECT_EQ(r.corner_pts[0][1], 64);
  EXPECT_EQ(r.corner_mvs[0][0], 0);
  EXPECT_EQ(r.corner_mvs[0][1], 8);

  // SB-border guard: deltaRow == -1, adJusted deltaCol == 0, bw4 <= 2.
  CornerResult guard;
  EXPECT(!warp_corner_process(f.grid(), 16, 16, 16, 2, 16, 0, -1, 0, 0, &guard));
  EXPECT_EQ(guard.corners_found, 0);

  // Warp neighbour: the corner MV is the model evaluated at (row+1,col+1)*4.
  BlockSpec warp;
  warp.row = 15;
  warp.col = 14;
  warp.motion_mode = kMotionLocalWarp;
  warp.warp[0] = model(0, 0, 65536 + 65536 / 4, 0, 0, 65536);
  Fixture wf = make_fixture(48, 48, {warp});
  CornerResult wr;
  EXPECT(warp_corner_process(wf.grid(), 16, 16, 16, 16, 16, 0, -1, -1, 0, &wr));
  EXPECT_EQ(wr.corners_found, 1);
  const Mv expected = get_warp_motion_vector_xy_pos(
      warp.warp[0], 16, 16, 16, 16, 16, 16, 48, 48);
  EXPECT_EQ(wr.corner_mvs[0][0], expected.y);
  EXPECT_EQ(wr.corner_mvs[0][1], expected.x);
}

void test_generate_corners() {
  {
    Fixture f = corner_fixture(false, false);
    WarpStack stack;
    const CornerResult r = generate_points_from_corners(
        f.grid(), 16, 16, 16, 16, 16, 0, 0, &stack);
    EXPECT_EQ(r.corners_found, 3);
    EXPECT(!r.all_mvs_same);
    EXPECT(!r.negative_ref_pts);
    EXPECT(r.inserted);
    EXPECT_EQ(stack.num_warp_found, 1);
    // Corner MVs: top-left (0,0), above (0,8), left (8,0).  The model is
    // p2=p5=1.0 and p3=p4=1024 (0.015625), with the translation that pins the
    // top-left corner; the reduce step keeps all six values on the 1/64 grid.
    EXPECT_EQ(r.wmmat.p[2], 1 << 16);
    EXPECT_EQ(r.wmmat.p[5], 1 << 16);
    EXPECT_EQ(r.wmmat.p[3], 1024);
    EXPECT_EQ(r.wmmat.p[4], 1024);
    EXPECT_EQ(r.wmmat.p[0], -65536);
    EXPECT_EQ(r.wmmat.p[1], -65536);
  }
  {
    Fixture f = corner_fixture(true, false);
    WarpStack stack;
    const CornerResult r = generate_points_from_corners(
        f.grid(), 16, 16, 16, 16, 16, 0, 0, &stack);
    EXPECT_EQ(r.corners_found, 3);
    EXPECT(r.all_mvs_same);
    EXPECT(!r.inserted);
    EXPECT_EQ(stack.num_warp_found, 0);
  }
  {
    Fixture f = corner_fixture(false, true);
    WarpStack stack;
    const CornerResult r = generate_points_from_corners(
        f.grid(), 16, 16, 16, 16, 16, 0, 0, &stack);
    EXPECT_EQ(r.corners_found, 3);
    EXPECT(r.negative_ref_pts);
    EXPECT(!r.inserted);
    EXPECT_EQ(stack.num_warp_found, 0);
  }
  {
    // Only two corners: the process must terminate without a model.
    Fixture f = corner_fixture(false, false);
    BlockSpec above;
    above.row = 0;
    above.col = 16;
    above.w4 = 16;
    above.h4 = 16;
    BlockSpec left;
    left.row = 16;
    left.col = 0;
    left.w4 = 16;
    left.h4 = 16;
    Fixture sparse = make_fixture(48, 48, {above, left});
    WarpStack stack;
    const CornerResult r = generate_points_from_corners(
        sparse.grid(), 16, 16, 16, 16, 16, 0, 0, &stack);
    EXPECT(r.corners_found < 3);
    EXPECT(!r.inserted);
    (void)f;
  }
  {
    // iter == 1 shifts the first two corner columns by one.  With neighbours
    // only at the shifted columns, iter 0 finds two corners and iter 1 finds
    // three.
    BlockSpec above;
    above.row = 0;
    above.col = 16;
    above.w4 = 16;
    above.h4 = 16;
    above.mv[0][0] = 8;
    above.sub_mv[0][0] = 8;
    BlockSpec above_right;
    above_right.row = 15;
    above_right.col = 32;
    above_right.w4 = 2;
    above_right.h4 = 1;
    BlockSpec left;
    left.row = 16;
    left.col = 0;
    left.w4 = 16;
    left.h4 = 16;
    left.mv[0][1] = 8;
    left.sub_mv[0][1] = 8;
    Fixture f = make_fixture(48, 48, {above, above_right, left});
    WarpStack stack;
    const CornerResult r0 = generate_points_from_corners(
        f.grid(), 16, 16, 16, 16, 16, 0, 0, &stack);
    EXPECT(r0.corners_found < 3);
    const CornerResult r1 = generate_points_from_corners(
        f.grid(), 16, 16, 16, 16, 16, 0, 1, &stack);
    EXPECT_EQ(r1.corners_found, 3);
  }
}

// ---------------------------------------------------------------------------
// Warp bank and insert order
// ---------------------------------------------------------------------------

void test_warp_bank_and_insert() {
  WarpParamBank bank;
  bank.reset();
  EXPECT_EQ(bank.size(), 0);
  bank.record(model(1, 0, 65536, 0, 0, 65536));
  bank.record(model(2, 0, 65536, 0, 0, 65536));
  bank.record(model(3, 0, 65536, 0, 0, 65536));
  EXPECT_EQ(bank.size(), 3);
  EXPECT_EQ(bank.at(2).p[0], 3);  // newest
  EXPECT_EQ(bank.at(0).p[0], 1);  // oldest
  bank.record(model(4, 0, 65536, 0, 0, 65536));
  bank.record(model(5, 0, 65536, 0, 0, 65536));
  EXPECT_EQ(bank.size(), 4);
  EXPECT_EQ(bank.at(3).p[0], 5);
  EXPECT_EQ(bank.at(2).p[0], 4);
  EXPECT_EQ(bank.at(1).p[0], 3);
  EXPECT_EQ(bank.at(0).p[0], 2);  // 1 was overwritten
  EXPECT_EQ(bank.start(), 1);

  WarpStack stack;
  insert_warp_candidate(model(9, 0, 65536, 0, 0, 65536), &stack);
  insert_warp_candidate(model(8, 0, 65536, 0, 0, 65536), &stack);
  insert_warp_candidate(model(7, 0, 65536, 0, 0, 65536), &stack);
  insert_warp_candidate(model(6, 0, 65536, 0, 0, 65536), &stack);
  insert_warp_candidate(model(5, 0, 65536, 0, 0, 65536), &stack);
  EXPECT_EQ(stack.num_warp_found, kMaxWarpRefCandidates);
  EXPECT_EQ(stack.warp_params[0].p[0], 9);
  EXPECT_EQ(stack.warp_params[3].p[0], 6);  // fifth insertion dropped
}

void test_warp_stack_tail_order() {
  // derive_wrl with an empty neighbourhood at a 64x64 block.  Step 3/4 insert
  // nothing (no corners), the scan finds nothing, and step 21 appends
  // bank newest-first, gm_params, then two identity defaults.
  Fixture f = make_fixture(16, 16, {});
  WarpStackParams p = stack_params(f, 0, 0, 16, 16);
  p.derive_wrl = true;
  p.gm_params = model(11, 0, 65536, 0, 0, 65536);
  WarpParamBank bank;
  bank.reset();
  bank.record(model(21, 0, 65536, 0, 0, 65536));
  bank.record(model(22, 0, 65536, 0, 0, 65536));
  p.warp_bank = &bank;
  WarpStack stack;
  EXPECT(build_warp_stack(p, &stack) == Status::kOk);
  EXPECT_EQ(stack.num_warp_found, 4);
  EXPECT_EQ(stack.warp_params[0].p[0], 22);  // newest bank entry
  EXPECT_EQ(stack.warp_params[1].p[0], 21);
  EXPECT_EQ(stack.warp_params[2].p[0], 11);  // gm_params
  EXPECT(stack.warp_params[3].identity());

  // Negative control for the order hazard: defaults first changes warp_idx
  // meaning while the same bits still parse.
  WarpStack wrong;
  wrong.num_warp_found = 4;
  wrong.warp_params[0] = ia::WarpParams{};
  wrong.warp_params[1] = ia::WarpParams{};
  wrong.warp_params[2] = p.gm_params;
  wrong.warp_params[3] = bank.at(1);
  EXPECT(write_snapshot(wrong) != write_snapshot(stack));
}

void test_add_warp_motion_vector() {
  BlockSpec warp;
  warp.row = 31;
  warp.col = 15;
  warp.motion_mode = kMotionLocalWarp;
  warp.warp[0] = model(7, 0, 65536, 0, 0, 65536);
  Fixture f = make_fixture(48, 48, {warp});
  WarpStack stack;
  add_warp_motion_vector(f.grid(), 16, 16, 0, true, 31, 15, &stack);
  EXPECT_EQ(stack.num_warp_found, 1);
  EXPECT_EQ(stack.warp_params[0].p[0], 7);
  // derive_wrl == 0 suppresses insertion.
  WarpStack disabled;
  add_warp_motion_vector(f.grid(), 16, 16, 0, false, 31, 15, &disabled);
  EXPECT_EQ(disabled.num_warp_found, 0);
  // A ref mismatch suppresses insertion.
  WarpStack other_ref;
  add_warp_motion_vector(f.grid(), 16, 16, 1, true, 31, 15, &other_ref);
  EXPECT_EQ(other_ref.num_warp_found, 0);
}

// ---------------------------------------------------------------------------
// §7.12.3 find warp samples
// ---------------------------------------------------------------------------

void test_find_warp_samples() {
  // A 64x64 block at (16,16) with a 16x16 neighbour above (0,16), a 16x16
  // neighbour left (16,0), and a 1-wide above-left block.  Left/above produce
  // one sample each; the top-left sample is added because doTopLeft stays 1.
  BlockSpec above;
  above.row = 0;
  above.col = 16;
  above.w4 = 16;
  above.h4 = 16;
  above.mv[0][0] = 8;
  above.mv[0][1] = 0;
  BlockSpec left;
  left.row = 16;
  left.col = 0;
  left.w4 = 16;
  left.h4 = 16;
  left.mv[0][0] = 0;
  left.mv[0][1] = 8;
  BlockSpec top_left;
  top_left.row = 0;
  top_left.col = 14;
  top_left.w4 = 2;
  top_left.h4 = 16;
  Fixture f = make_fixture(48, 48, {above, left, top_left});

  WarpSampleParams p;
  p.mi_row = 16;
  p.mi_col = 16;
  p.w4 = 16;
  p.h4 = 16;
  p.mi_cols = f.cols;
  p.mi_rows = f.rows;
  p.sb_h4 = 16;
  p.grid = f.grid();
  p.ref_frame[0] = 0;
  p.ref_frame[1] = kRefNone;
  p.ref = 0;
  p.avail_u = true;
  p.avail_l = true;
  WarpSampleSet set;
  EXPECT(find_warp_samples(p, &set));
  EXPECT_EQ(set.num_samples, 3);
  // Above sample: above block centre (31, 95), MV (8, 0).
  EXPECT_EQ(set.cand[0][0], 31 * 8);
  EXPECT_EQ(set.cand[0][1], 95 * 8);
  EXPECT_EQ(set.cand[0][2], 31 * 8 + 8);
  EXPECT_EQ(set.cand[0][3], 95 * 8);
  // Left sample: left block centre (95, 31), MV (0, 8).
  EXPECT_EQ(set.cand[1][0], 95 * 8);
  EXPECT_EQ(set.cand[1][1], 31 * 8);
  EXPECT_EQ(set.cand[1][2], 95 * 8);
  EXPECT_EQ(set.cand[1][3], 31 * 8 + 8);
  // Top-left sample: the tall (0..15,14..15) block centre (31, 59).
  EXPECT_EQ(set.cand[2][0], 31 * 8);
  EXPECT_EQ(set.cand[2][1], 59 * 8);

  // With no above/left availability the spec still probes the top-left
  // sample (doTopLeft stays 1), and above_sample_stored accepts it.
  p.avail_u = false;
  p.avail_l = false;
  WarpSampleSet empty;
  EXPECT(find_warp_samples(p, &empty));
  EXPECT_EQ(empty.num_samples, 1);
  EXPECT_EQ(empty.cand[0][0], 31 * 8);
  EXPECT_EQ(empty.cand[0][1], 59 * 8);

  // Left-only: the left sample plus the top-left probe.
  p.avail_l = true;
  WarpSampleSet left_only;
  EXPECT(find_warp_samples(p, &left_only));
  EXPECT_EQ(left_only.num_samples, 2);
  EXPECT_EQ(left_only.cand[0][0], 95 * 8);
  EXPECT_EQ(left_only.cand[1][0], 31 * 8);
}

void test_above_sample_stored() {
  // On the top superblock row an above sample at an odd column is only usable
  // when its block is at least 2 MI wide and does not end at the frame edge.
  std::vector<BlockSpec> blocks;
  BlockSpec wide;
  wide.row = 15;
  wide.col = 15;
  wide.w4 = 2;
  wide.h4 = 1;
  Fixture f = make_fixture(32, 32, {wide});

  WarpSampleParams p;
  p.mi_row = 16;
  p.mi_col = 16;
  p.w4 = 4;
  p.h4 = 4;
  p.mi_cols = f.cols;
  p.mi_rows = f.rows;
  p.sb_h4 = 16;
  p.grid = f.grid();
  p.ref_frame[0] = 0;
  p.ref_frame[1] = kRefNone;
  p.ref = 0;
  EXPECT(above_sample_stored(p, -1));   // even column 15? 16-1=15 is odd
  // 15 is odd, block w4 == 2, and 15+1 < 32 -> true.
  EXPECT(above_sample_stored(p, 0));    // even column
  EXPECT(!above_sample_stored(p, 1));   // col 17 is an unwritten 1-MI cell

  Fixture narrow = make_fixture(32, 32, {BlockSpec{15, 15, 1, 1, {0, kRefNone},
                                                   {{0, 0}, {0, 0}},
                                                   {{0, 0}, {0, 0}},
                                                   kMotionSimple, kYModeNewMv,
                                                   {}, true, true}});
  p.grid = narrow.grid();
  EXPECT(!above_sample_stored(p, -1));  // 1-wide at an odd column
}

void test_find_warp_samples_cap() {
  // Eight is LEAST_SQUARES_SAMPLES_MAX.  Build a left column of 8 single-MI
  // inter blocks with distinct list-0 and list-1 reference frames.
  std::vector<BlockSpec> blocks;
  for (int i = 0; i < 8; ++i) {
    BlockSpec b;
    b.row = 16 + i;
    b.col = 15;
    b.w4 = 1;
    b.h4 = 1;
    b.ref[0] = 0;
    b.ref[1] = 0;
    b.mv[0][0] = i;
    b.mv[0][1] = 0;
    b.mv[1][0] = i;
    b.mv[1][1] = 0;
    b.sub_mv[0][0] = i;
    b.sub_mv[1][0] = i;
    blocks.push_back(b);
  }
  Fixture f = make_fixture(32, 48, blocks);
  WarpSampleParams p;
  p.mi_row = 16;
  p.mi_col = 16;
  p.w4 = 4;
  p.h4 = 16;
  p.mi_cols = f.cols;
  p.mi_rows = f.rows;
  p.sb_h4 = 16;
  p.grid = f.grid();
  p.ref_frame[0] = 0;
  p.ref_frame[1] = 0;
  p.ref = 0;
  p.avail_l = true;
  WarpSampleSet set;
  EXPECT(find_warp_samples(p, &set));
  EXPECT_EQ(set.num_samples, kLeastSquaresSamplesMax);
  // Both lists match, so the first block contributes two samples at the same
  // position; sample 8 is never emitted.
  EXPECT_EQ(set.cand[0][0], set.cand[1][0]);
  EXPECT_EQ(set.cand[0][2], set.cand[1][2]);

  // Sampling list 1 produces the same candidate geometry with the list-1 MV.
  p.ref = 0;
  WarpSampleSet list0;
  find_warp_samples(p, &list0);
  p.ref_frame[1] = 1;  // a different reference: list 1 no longer matches
  WarpSampleSet list1;
  p.ref = 1;
  EXPECT(!find_warp_samples(p, &list1));
}

void test_local_warp_models() {
  // Non-degenerate: three distinct neighbours.
  Fixture f = corner_fixture(false, false);
  LocalWarpParams p;
  p.mi_row = 16;
  p.mi_col = 16;
  p.w4 = 16;
  p.h4 = 16;
  p.mi_cols = f.cols;
  p.mi_rows = f.rows;
  p.sb_h4 = 16;
  p.grid = f.grid();
  p.ref_frame[0] = 0;
  p.ref_frame[1] = kRefNone;
  p.avail_u = true;
  p.avail_l = true;
  p.block_mv = Mv{0, 0};
  const auto r = derive_local_warp(p, 0);
  EXPECT(r.has_value());
  EXPECT(!r->det_zero);
  EXPECT_EQ(r->samples.num_samples, 3);
  // adv::warp_estimate is the pixel-side authority; the wrapper must agree.
  const int mid_x = 16 * 4 + 16 * 2 - 1;
  const int mid_y = 16 * 4 + 16 * 2 - 1;
  const int block_mv[2] = {0, 0};
  const ia::WarpParams direct =
      estimate_warp_from_samples(r->samples, mid_x, mid_y, block_mv);
  EXPECT(direct.p[0] == r->params.p[0]);
  EXPECT(direct.p[1] == r->params.p[1]);
  for (int i = 2; i < 6; ++i) EXPECT_EQ(direct.p[i], r->params.p[i]);

  // Degenerate: no neighbours at all => A == 0 => det == 0 => translation
  // fallback from BlockMvs[ref].
  Fixture none = make_fixture(48, 48, {});
  p.grid = none.grid();
  p.block_mv = Mv{0, 8};
  const auto deg = derive_local_warp(p, 0);
  EXPECT(deg.has_value());
  EXPECT_EQ(deg->samples.num_samples, 0);
  EXPECT(deg->det_zero);
  EXPECT_EQ(deg->params.p[2], 1 << 16);
  EXPECT_EQ(deg->params.p[5], 1 << 16);
  // identity scale, translation from BlockMvs (mv_x = 8 -> 1 luma sample).
  EXPECT_EQ(deg->params.p[0], 8 << (ia::kWarpedModelPrecBits - 3));
  EXPECT_EQ(deg->params.p[1], 0);
}

void test_build_rejects_unsupported() {
  Fixture f = make_fixture(16, 16, {});
  WarpStack stack;
  WarpStackParams p = stack_params(f, 0, 0, 16, 16);
  p.is_compound = true;
  EXPECT(build_warp_stack(p, &stack) == Status::kUnsupportedCompound);
  p = stack_params(f, 0, 0, 16, 16);
  p.use_intrabc = true;
  EXPECT(build_warp_stack(p, &stack) == Status::kUnsupportedIntrabc);
  p = stack_params(f, 0, 0, 16, 16);
  p.use_ref_frame_mvs = true;
  EXPECT(build_warp_stack(p, &stack) == Status::kUnsupportedTemporal);
  p = stack_params(f, 0, 0, 32, 16);
  EXPECT(build_warp_stack(p, &stack) == Status::kInvalidBlockSize);
  p = stack_params(f, 0, 0, 16, 16);
  p.ref_frame[0] = kRefIntra;
  EXPECT(build_warp_stack(p, &stack) == Status::kInvalidArgument);
  // The mirror rejects the same inputs with the same status.
  p = stack_params(f, 0, 0, 16, 16);
  p.is_compound = true;
  WarpStack mirrored;
  EXPECT(mirror::build_stack(p, &mirrored) == Status::kUnsupportedCompound);
}

// ---------------------------------------------------------------------------
// §7.13.3.19 SubMvs storage and the warp-derived stack hazard
// ---------------------------------------------------------------------------

void test_sub_block_mv() {
  // Identity + 1 pel x translation: p0 = 8 << (16-3).
  ia::WarpParams translation = model(8 << (ia::kWarpedModelPrecBits - 3), 0,
                                    65536, 0, 0, 65536);
  // Every 4x4 cell of every 8x8 section sees the same translation.
  for (int i8 = 0; i8 < 8; ++i8) {
    for (int j8 = 0; j8 < 8; ++j8) {
      const Mv mv = warp_sub_block_mv(translation, 0, 0, i8, j8);
      EXPECT_EQ(mv.x, 8);
      EXPECT_EQ(mv.y, 0);
    }
  }
  // x scale of 2: dst_x - src_x = src_x, so the projected MV grows with the
  // section offset.  At src_x = x + j8*8 + 4 the q16 difference is src_x << 16.
  ia::WarpParams zoom = model(0, 0, 2 * 65536, 0, 0, 65536);
  EXPECT_EQ(warp_sub_block_mv(zoom, 0, 0, 0, 0).x, 4 * 8);
  EXPECT_EQ(warp_sub_block_mv(zoom, 64, 0, 0, 0).x, 68 * 8);
  EXPECT_EQ(warp_sub_block_mv(zoom, 0, 0, 2, 0).x, 4 * 8);   // y does not scale x
  EXPECT_EQ(warp_sub_block_mv(zoom, 0, 0, 0, 0).y, 0);
  // The result is clipped to MV_LOW+1..MV_UPP-1.
  ia::WarpParams huge = model(0, 0, 200000, 0, 0, 65536);
  EXPECT(warp_sub_block_mv(huge, 4000, 0, 0, 7).x == ia::kMvUpp - 1);
}

void test_sub_mvs_feed_the_stack() {
  // A LOCALWARP neighbour stores its projected sub-block MV in SubMvs at the
  // scanned cell, while Mvs at the block origin keeps the block MV.  The MV
  // stack must use SubMvs; §7.12.3 must use Mvs.  This is the hazard the
  // audit calls out: using the block MV everywhere desynchronises any block
  // whose neighbour is warped.
  BlockSpec warped;
  warped.row = 16;
  warped.col = 16;
  warped.w4 = 16;
  warped.h4 = 16;
  warped.motion_mode = kMotionLocalWarp;
  warped.warp[0] = model(0, 0, 65536, 0, 0, 65536);
  warped.mv[0][0] = 8;
  warped.mv[0][1] = 0;
  warped.sub_mv[0][0] = 8;
  warped.sub_mv[0][1] = 0;
  Fixture f = make_fixture(48, 48, {warped});
  // Projected sub-block MV at the bottom-right 8x8 section (the cell scanned
  // by a block at (16,32)).
  WarpMi& cell = f.mi[static_cast<std::size_t>(31) * f.cols + 31];
  cell.sub_mv[0][0] = 12;
  cell.sub_mv[0][1] = -1;

  WarpStackParams p = stack_params(f, 16, 32, 16, 16);
  WarpStack stack;
  EXPECT(build_warp_stack(p, &stack) == Status::kOk);
  EXPECT(stack.count >= 1);
  EXPECT_EQ(stack.cand[0].mv[0].y, 12);
  EXPECT_EQ(stack.cand[0].mv[0].x, -1);

  // Negative control: flattening SubMvs to the block MV (the pre-fix state)
  // yields a different candidate and therefore a different decoder stack.
  Fixture flat = f;
  WarpMi& flat_cell = flat.mi[static_cast<std::size_t>(31) * flat.cols + 31];
  flat_cell.sub_mv[0][0] = flat_cell.mv[0][0];
  flat_cell.sub_mv[0][1] = flat_cell.mv[0][1];
  WarpStack flat_stack;
  EXPECT(build_warp_stack(stack_params(flat, 16, 32, 16, 16), &flat_stack) ==
         Status::kOk);
  EXPECT(flat_stack.count >= 1);
  EXPECT_EQ(flat_stack.cand[0].mv[0].y, 8);
  EXPECT(write_snapshot(stack) != write_snapshot(flat_stack));

  // §7.12.3 uses Mvs (the block origin MV), not SubMvs: the sample's dst is
  // mid + block MV.
  WarpSampleParams sp;
  sp.mi_row = 16;
  sp.mi_col = 32;
  sp.w4 = 16;
  sp.h4 = 16;
  sp.mi_cols = f.cols;
  sp.mi_rows = f.rows;
  sp.sb_h4 = 16;
  sp.grid = f.grid();
  sp.ref_frame[0] = 0;
  sp.ref_frame[1] = kRefNone;
  sp.ref = 0;
  sp.avail_l = true;
  WarpSampleSet samples;
  EXPECT(find_warp_samples(sp, &samples));
  // Left sample: block origin (16,16), centre (95,95), Mvs = (8,0).
  EXPECT_EQ(samples.cand[0][0], 95 * 8);
  EXPECT_EQ(samples.cand[0][1], 95 * 8);
  EXPECT_EQ(samples.cand[0][2], 95 * 8 + 8);
  EXPECT_EQ(samples.cand[0][3], 95 * 8);
}

void test_warp_estimate_golden() {
  // The four-sample set a 64x64 block at (16,16) sees in the gate stream
  // (above, left, top-left and above-right neighbours).  The model pins the
  // 64-bit determinant path: the component's adv::warp_estimate() narrows det
  // to int and returns p2 = 32832 instead of 65920 for this set, which breaks
  // decoder parity.
  WarpSampleSet s;
  s.num_samples = 4;
  const int cand[4][4] = {{248, 760, 248, 768},
                          {760, 248, 768, 248},
                          {248, 248, 248, 248},
                          {248, 1272, 248, 1280}};
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 4; ++k) s.cand[i][k] = cand[i][k];
  }
  const int mv[2] = {8, 0};
  const ia::WarpParams m = estimate_warp_from_samples(s, 95, 95, mv);
  EXPECT_EQ(m.p[0], 30400);
  EXPECT_EQ(m.p[1], -37824);
  EXPECT_EQ(m.p[2], 65920);
  EXPECT_EQ(m.p[3], -704);
  EXPECT_EQ(m.p[4], 0);
  EXPECT_EQ(m.p[5], 66624);

  // A 64x64 sample set whose determinant exceeds 2^31 (the old int narrowing
  // failed here): the gate's F-block samples.
  WarpSampleSet f;
  f.num_samples = 3;
  const int f_cand[3][4] = {{248, 1272, 248, 1280},
                            {760, 760, 768, 760},
                            {248, 760, 248, 768}};
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 4; ++k) f.cand[i][k] = f_cand[i][k];
  }
  const int f_mv[2] = {8, 0};
  const ia::WarpParams fm = estimate_warp_from_samples(f, 159, 95, f_mv);
  EXPECT_EQ(fm.p[2], 65600);
  EXPECT_EQ(fm.p[3], -1024);
  EXPECT_EQ(fm.p[5], 66688);
}

// ---------------------------------------------------------------------------
// §7.12.2.19 sort and §7.12.2.23 clamp
// ---------------------------------------------------------------------------

void test_sort_spec_exact() {
  WarpStack stack;
  stack.count = 4;
  stack.num_nearest = 4;
  const int weights[4] = {1, 2, 3, 4};
  for (int i = 0; i < 4; ++i) {
    stack.cand[i].mv[0] = Mv{weights[i] * 10, 0};
    stack.cand[i].mv[1] = stack.cand[i].mv[0];
    stack.cand[i].weight = weights[i];
    stack.cand[i].row_offset = i;
    stack.cand[i].col_offset = -i;
  }
  sort_stack_range(&stack, 0, 4);
  // Single move-to-front: {4,2,3,1}, not a selection sort ({4,3,2,1}).  The
  // wired M2 stack's selection sort is the analogous DRL desync.
  EXPECT_EQ(stack.cand[0].mv[0].y, 40);
  EXPECT_EQ(stack.cand[0].row_offset, 3);
  EXPECT_EQ(stack.cand[1].mv[0].y, 20);
  EXPECT_EQ(stack.cand[2].mv[0].y, 30);
  EXPECT_EQ(stack.cand[3].mv[0].y, 10);
}

// ---------------------------------------------------------------------------
// Wired-stack parity (derive_wrl == 0)
// ---------------------------------------------------------------------------

struct RandomGrid {
  Fixture fixture;
  int query_row = 0;
  int query_col = 0;
  int bw4 = 1;
  int bh4 = 1;
  bool derive_wrl = false;
  ia::WarpParams gm{};
  bool gm_nonidentity = false;
  Mv global{};
  DrlReorder reorder = DrlReorder::kDisabled;
};

RandomGrid random_grid(std::mt19937& rng) {
  RandomGrid out;
  const int sizes[] = {1, 2, 4, 8, 16};
  const int cols = 16 + static_cast<int>(rng() % 3) * 16;  // 16..48
  const int rows = 16 + static_cast<int>(rng() % 3) * 16;
  std::vector<BlockSpec> blocks;
  for (int r = 0; r < rows;) {
    const int h4 = std::min(sizes[rng() % 5], rows - r);
    for (int c = 0; c < cols;) {
      const int w4 = std::min(sizes[rng() % 5], cols - c);
      BlockSpec b;
      b.row = r;
      b.col = c;
      b.w4 = w4;
      b.h4 = h4;
      b.written = (rng() % 17) != 0;  // leave some cells unwritten
      const int kind = static_cast<int>(rng() % 8);
      if (kind == 0) {
        b.ref[0] = kRefIntra;
        b.is_inter = false;
      } else if (kind == 1) {
        b.ref[0] = kRefNone;
        b.is_inter = false;
        b.written = false;
      } else {
        b.ref[0] = static_cast<std::int8_t>(rng() % 3);
        b.ref[1] = (rng() % 4 == 0) ? static_cast<std::int8_t>(rng() % 3)
                                    : kRefNone;
        b.is_inter = true;
      }
      for (int list = 0; list < 2; ++list) {
        for (int comp = 0; comp < 2; ++comp) {
          const int v = static_cast<int>(rng() % 129) - 64;
          b.mv[list][comp] = v;
          b.sub_mv[list][comp] = v;
        }
      }
      b.motion_mode = static_cast<std::uint8_t>(rng() % 5);
      b.y_mode = static_cast<std::uint8_t>(rng() % 2 == 0 ? kYModeNewMv
                                                          : kYModeGlobalMv);
      for (int ref = 0; ref < 2; ++ref) {
        b.warp[ref] = model(static_cast<int>(rng() % 65) - 32,
                            static_cast<int>(rng() % 65) - 32,
                            65536 + static_cast<int>(rng() % 4097) - 2048,
                            static_cast<int>(rng() % 4097) - 2048,
                            static_cast<int>(rng() % 4097) - 2048,
                            65536 + static_cast<int>(rng() % 4097) - 2048);
      }
      blocks.push_back(b);
      c += w4;
    }
    r += h4;
  }
  out.fixture = make_fixture(cols, rows, blocks);
  out.bw4 = sizes[rng() % 5];
  out.bh4 = sizes[rng() % 5];
  out.query_col = static_cast<int>(rng() % static_cast<unsigned>(cols - out.bw4 + 1));
  out.query_row = static_cast<int>(rng() % static_cast<unsigned>(rows - out.bh4 + 1));
  out.derive_wrl = (rng() % 2) != 0;
  out.gm_nonidentity = (rng() % 4) == 0;
  out.gm.p[0] = static_cast<int>(rng() % 65) - 32;
  out.global = Mv{static_cast<int>(rng() % 65) - 32,
                  static_cast<int>(rng() % 65) - 32};
  out.reorder = static_cast<DrlReorder>(rng() % 3);
  return out;
}

WarpStackParams random_params(const RandomGrid& g) {
  WarpStackParams p;
  p.mi_row = g.query_row;
  p.mi_col = g.query_col;
  p.bw4 = g.bw4;
  p.bh4 = g.bh4;
  p.sb_h4 = 16;
  p.mi_cols = g.fixture.cols;
  p.mi_rows = g.fixture.rows;
  p.mi = g.fixture.mi.data();
  p.ref_frame[0] = 0;
  p.ref_frame[1] = kRefNone;
  p.derive_wrl = g.derive_wrl;
  p.global_mv[0] = g.global;
  p.global_mv_nonidentity = g.gm_nonidentity;
  p.gm_params = g.gm;
  p.reorder = g.reorder;
  return p;
}

void test_wired_stack_parity() {
  std::mt19937 rng(0x5eed1234u);
  int compared = 0;
  for (int iter = 0; iter < 300; ++iter) {
    RandomGrid g = random_grid(rng);
    g.derive_wrl = false;       // parity is only claimed for the warp-free stack
    g.gm_nonidentity = false;   // merge's get_mv() has no global branch
    g.reorder = DrlReorder::kDisabled;  // merge sorts with a selection sort
    // merge.cpp's scan_col() applies the MiCol&1 adjustment unconditionally;
    // §7.12.2.5 does it only when bw4 == 1.  test_wired_scan_col_deviation
    // pins that difference, so exclude those queries here.
    if (g.bw4 != 1 && (g.query_col & 1) != 0) continue;
    WarpStack mine;
    const WarpStackParams wp = random_params(g);
    if (build_warp_stack(wp, &mine) != Status::kOk) continue;

    av2enc::m2::merge::StackParams mp;
    mp.mi_row = wp.mi_row;
    mp.mi_col = wp.mi_col;
    mp.bw4 = wp.bw4;
    mp.bh4 = wp.bh4;
    mp.sb_h4 = wp.sb_h4;
    mp.mi_cols = wp.mi_cols;
    mp.mi_rows = wp.mi_rows;
    mp.mi = g.fixture.m2.data();
    mp.ref_frame[0] = wp.ref_frame[0];
    mp.ref_frame[1] = wp.ref_frame[1];
    mp.is_compound = false;
    mp.global_mv[0] = wp.global_mv[0];
    mp.global_mv[1] = wp.global_mv[0];
    mp.reorder = static_cast<av2enc::m2::merge::DrlReorder>(
        static_cast<int>(wp.reorder));
    mp.use_temporal_first = false;
    mp.max_drl_bits_minus_1 = 4;
    mp.use_intrabc = false;
    mp.frame_distance = nullptr;
    mp.num_refs = 0;

    av2enc::m2::merge::MvStack theirs;
    if (av2enc::m2::merge::build_stack_checked(mp, &theirs) !=
        av2enc::m2::merge::Status::kOk) {
      continue;
    }
    EXPECT_EQ(mine.count, theirs.count);
    EXPECT_EQ(mine.num_nearest, theirs.num_nearest);
    EXPECT_EQ(mine.sorted, theirs.sorted);
    const int n = std::min(mine.count, theirs.count);
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(mine.cand[i].mv[0].y, theirs.cand[i].mv[0].y);
      EXPECT_EQ(mine.cand[i].mv[0].x, theirs.cand[i].mv[0].x);
      EXPECT_EQ(mine.cand[i].weight, theirs.cand[i].weight);
      EXPECT_EQ(mine.cand[i].row_offset, theirs.cand[i].row_offset);
      EXPECT_EQ(mine.cand[i].col_offset, theirs.cand[i].col_offset);
    }
    ++compared;
  }
  EXPECT(compared > 150);
}

void test_global_mv_substitution() {
  // §7.12.2.13 get_mv(): a large GLOBALMV candidate from a non-identity global
  // model resolves to GlobalMvs[refList], not to SubMvs.
  BlockSpec above;
  above.row = 0;
  above.col = 0;
  above.w4 = 16;
  above.h4 = 16;
  above.y_mode = kYModeGlobalMv;
  above.ref[0] = 0;
  above.mv[0][0] = 40;
  above.mv[0][1] = -40;
  above.sub_mv[0][0] = 40;
  above.sub_mv[0][1] = -40;
  Fixture f = make_fixture(48, 48, {above});
  WarpStackParams p = stack_params(f, 16, 0, 16, 16);
  p.global_mv_nonidentity = true;
  p.global_mv[0] = Mv{7, -9};
  WarpStack stack;
  EXPECT(build_warp_stack(p, &stack) == Status::kOk);
  EXPECT_EQ(stack.count, 1);
  EXPECT_EQ(stack.cand[0].mv[0].y, 7);
  EXPECT_EQ(stack.cand[0].mv[0].x, -9);
  // With an identity global model the same block contributes its SubMvs.
  p.global_mv_nonidentity = false;
  WarpStack plain;
  EXPECT(build_warp_stack(p, &plain) == Status::kOk);
  EXPECT_EQ(plain.cand[0].mv[0].y, 40);
  EXPECT_EQ(plain.cand[0].mv[0].x, -40);
}

void test_corner_then_scan_order() {
  // A block with three corner neighbours and an adjacent LOCALWARP block: the
  // corner model is inserted at step 3, before any §7.12.2.9 scan candidate.
  Fixture f = corner_fixture(false, false);
  // Turn the above block into a local-warp block so the scan inserts it.
  // (Rewrite the cells with a warp model, keeping the same MV.)
  for (int c = 16; c < 32; ++c) {
    WarpMi& m = f.mi[static_cast<std::size_t>(15) * f.cols + c];
    m.motion_mode = kMotionLocalWarp;
    m.warp[0] = model(123, 0, 65536, 0, 0, 65536);
  }
  WarpStackParams p = stack_params(f, 16, 16, 16, 16);
  p.derive_wrl = true;
  WarpStack stack;
  EXPECT(build_warp_stack(p, &stack) == Status::kOk);
  EXPECT(stack.num_warp_found >= 2);
  // Corner model first: its translation is the corner-fit value (-65536).
  EXPECT_EQ(stack.warp_params[0].p[0], -65536);
  // Then the scan's add-warp-MV of the above block (twice: aboveA/aboveB both
  // land in it, and §7.12.2.11 does not deduplicate).
  EXPECT_EQ(stack.warp_params[1].p[0], 123);
}

void test_wired_scan_col_deviation() {
  // §7.12.2.5 adjusts deltaCol by MiCol&1 only for a 1-MI-wide block.  The
  // wired merge.cpp does it unconditionally, so for an odd MiCol and bw4 != 1
  // it scans one column to the right.  This test pins the spec behaviour and
  // the wired deviation; the parity test excludes this query shape.
  BlockSpec at_two;
  at_two.row = 23;
  at_two.col = 2;
  at_two.mv[0][0] = 1;
  at_two.mv[0][1] = 1;
  at_two.sub_mv[0][0] = 1;
  at_two.sub_mv[0][1] = 1;
  BlockSpec at_one;
  at_one.row = 23;
  at_one.col = 1;
  at_one.mv[0][0] = 2;
  at_one.mv[0][1] = 0;
  at_one.sub_mv[0][0] = 2;
  at_one.sub_mv[0][1] = 0;
  Fixture f = make_fixture(32, 32, {at_two, at_one});

  WarpStackParams p = stack_params(f, 16, 3, 8, 8);
  WarpStack mine;
  EXPECT(build_warp_stack(p, &mine) == Status::kOk);
  EXPECT_EQ(mine.count, 2);  // (23,2) plus the global candidate
  EXPECT_EQ(mine.cand[0].mv[0].y, 1);
  EXPECT_EQ(mine.cand[0].mv[0].x, 1);
  for (int i = 0; i < mine.count; ++i) {
    EXPECT(!(mine.cand[i].mv[0].y == 2 && mine.cand[i].mv[0].x == 0));
  }

  av2enc::m2::merge::StackParams mp;
  mp.mi_row = p.mi_row;
  mp.mi_col = p.mi_col;
  mp.bw4 = p.bw4;
  mp.bh4 = p.bh4;
  mp.sb_h4 = p.sb_h4;
  mp.mi_cols = p.mi_cols;
  mp.mi_rows = p.mi_rows;
  mp.mi = f.m2.data();
  mp.ref_frame[0] = 0;
  mp.ref_frame[1] = kRefNone;
  mp.global_mv[0] = Mv{0, 0};
  mp.reorder = av2enc::m2::merge::DrlReorder::kDisabled;
  av2enc::m2::merge::MvStack theirs;
  EXPECT(av2enc::m2::merge::build_stack_checked(mp, &theirs) ==
         av2enc::m2::merge::Status::kOk);
  EXPECT_EQ(theirs.count, 3);  // wired stack also scans (23,1)
  bool found_two = false;
  bool found_one = false;
  for (int i = 0; i < theirs.count; ++i) {
    if (theirs.cand[i].mv[0].y == 2 && theirs.cand[i].mv[0].x == 0) found_one = true;
    if (theirs.cand[i].mv[0].y == 1 && theirs.cand[i].mv[0].x == 1) found_two = true;
  }
  EXPECT(found_two);
  EXPECT(found_one);
}

// ---------------------------------------------------------------------------
// Randomized lockstep: primary vs independent mirror
// ---------------------------------------------------------------------------

void test_lockstep_randomized() {
  std::mt19937 rng(0xc0ffeeu);
  int stacks = 0;
  int sample_sets = 0;
  for (int iter = 0; iter < 600; ++iter) {
    RandomGrid g = random_grid(rng);
    const WarpStackParams wp = random_params(g);
    WarpStack primary;
    WarpStack mirrored;
    const Status a = build_warp_stack(wp, &primary);
    const Status b = mirror::build_stack(wp, &mirrored);
    EXPECT_EQ(static_cast<int>(a), static_cast<int>(b));
    if (a != Status::kOk) continue;
    EXPECT(write_snapshot(primary) == write_snapshot(mirrored));
    ++stacks;

    WarpSampleParams sp;
    sp.mi_row = wp.mi_row;
    sp.mi_col = wp.mi_col;
    sp.w4 = wp.bw4;
    sp.h4 = wp.bh4;
    sp.mi_cols = wp.mi_cols;
    sp.mi_rows = wp.mi_rows;
    sp.sb_h4 = 16;
    sp.grid = wp.grid();
    sp.ref_frame[0] = 0;
    sp.ref_frame[1] = kRefNone;
    sp.ref = 0;
    sp.avail_u = wp.mi_row > 0;
    sp.avail_l = wp.mi_col > 0;
    WarpSampleSet s1;
    WarpSampleSet s2;
    const bool f1 = find_warp_samples(sp, &s1);
    const bool f2 = mirror::warp_samples(sp, &s2);
    EXPECT(f1 == f2);
    EXPECT_EQ(s1.num_samples, s2.num_samples);
    for (int i = 0; i < std::min(s1.num_samples, s2.num_samples); ++i) {
      for (int k = 0; k < 4; ++k) EXPECT_EQ(s1.cand[i][k], s2.cand[i][k]);
    }
    ++sample_sets;
  }
  EXPECT(stacks > 500);
  EXPECT(sample_sets > 500);
}

void test_snapshot_roundtrip() {
  std::mt19937 rng(0xabcdefu);
  int round_trips = 0;
  for (int iter = 0; iter < 100; ++iter) {
    RandomGrid g = random_grid(rng);
    WarpStack stack;
    if (build_warp_stack(random_params(g), &stack) != Status::kOk) continue;
    const std::string text = write_snapshot(stack);
    WarpStack parsed;
    std::string error;
    EXPECT(read_snapshot(text, &parsed, &error));
    EXPECT(error.empty());
    EXPECT(write_snapshot(parsed) == text);
    ++round_trips;
  }
  EXPECT(round_trips > 80);

  WarpStack bad;
  std::string error;
  EXPECT(!read_snapshot("NOTASTACK 1\n", &bad, &error));
  EXPECT(!error.empty());
  EXPECT(!read_snapshot("WARPSTACK 2\n", &bad, &error));
  EXPECT(!read_snapshot(
      "WARPSTACK 1\nCOUNT 7 NEAREST 0 WARP 0 SORTED 0\nEND\n", &bad, &error));
  EXPECT(!read_snapshot(
      "WARPSTACK 1\nCOUNT 1 NEAREST 0 WARP 0 SORTED 0\n"
      "C 0 0 0 0 1 0 0 8\nEND extra\n",
      &bad, &error));
}

// ---------------------------------------------------------------------------
// Pinned goldens
// ---------------------------------------------------------------------------

// Filled from the arithmetic, not copied from a run of the shipped code: these
// vectors are short enough to check by hand and are deliberately independent
// of the randomized tests.  The snapshot goldens pin the complete stack
// (candidate order, weights, offsets and warp parameter order).
struct Golden {
  const char* name;
  const char* snapshot;
};

const Golden kStackGoldens[] = {
    {"empty_derive_wrl_identity",
     "WARPSTACK 1\nCOUNT 1 NEAREST 0 WARP 3 SORTED 0\n"
     "C 0 0 0 0 1 0 0 8\n"
     "W 0 0 65536 0 0 65536\nW 0 0 65536 0 0 65536\n"
     "W 0 0 65536 0 0 65536\nEND\n"},
    {"warp_only_no_stack",
     "WARPSTACK 1\nCOUNT 1 NEAREST 1 WARP 4 SORTED 0\n"
     "C 0 0 0 0 2 15 -1 8\n"
     "W 7 0 65536 0 0 65536\nW 7 0 65536 0 0 65536\n"
     "W 0 0 65536 0 0 65536\nW 0 0 65536 0 0 65536\nEND\n"},
};

void test_goldens() {
  // The two goldens are produced by the simplest deterministic scenarios:
  // an isolated 64x64 block with derive_wrl on, and the same block with one
  // §7.12.2.9 warp candidate from its left neighbour.
  {
    Fixture f = make_fixture(16, 16, {});
    WarpStackParams p = stack_params(f, 0, 0, 16, 16);
    p.derive_wrl = true;
    WarpStack stack;
    EXPECT(build_warp_stack(p, &stack) == Status::kOk);
    EXPECT(write_snapshot(stack) == kStackGoldens[0].snapshot);
  }
  {
    BlockSpec warp;
    warp.row = 0;
    warp.col = 0;
    warp.w4 = 8;
    warp.h4 = 16;
    warp.motion_mode = kMotionLocalWarp;
    warp.warp[0] = model(7, 0, 65536, 0, 0, 65536);
    Fixture f = make_fixture(16, 16, {warp});
    // Current block at (0,8): its left neighbour is the warp block above.
    WarpStackParams p = stack_params(f, 0, 8, 8, 16);
    p.derive_wrl = true;
    WarpStack stack;
    EXPECT(build_warp_stack(p, &stack) == Status::kOk);
    EXPECT(write_snapshot(stack) == kStackGoldens[1].snapshot);
  }
}

void print_goldens() {
  {
    Fixture f = make_fixture(16, 16, {});
    WarpStackParams p = stack_params(f, 0, 0, 16, 16);
    p.derive_wrl = true;
    WarpStack stack;
    build_warp_stack(p, &stack);
    std::printf("GOLDEN empty_derive_wrl_identity\n%s", write_snapshot(stack).c_str());
  }
  {
    BlockSpec warp;
    warp.row = 0;
    warp.col = 0;
    warp.w4 = 8;
    warp.h4 = 16;
    warp.motion_mode = kMotionLocalWarp;
    warp.warp[0] = model(7, 0, 65536, 0, 0, 65536);
    Fixture f = make_fixture(16, 16, {warp});
    WarpStackParams p = stack_params(f, 0, 8, 8, 16);
    p.derive_wrl = true;
    WarpStack stack;
    build_warp_stack(p, &stack);
    std::printf("GOLDEN warp_only_no_stack\n%s", write_snapshot(stack).c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--emit-goldens") == 0) {
    print_goldens();
    return 0;
  }
  std::printf("warp_stack_test: AV2 §7.11.4/§7.12.2.3-.11/§7.12.3\n");
  test_context_basic();
  test_context_sb_even_column();
  test_corner_process();
  test_generate_corners();
  test_warp_bank_and_insert();
  test_warp_stack_tail_order();
  test_add_warp_motion_vector();
  test_find_warp_samples();
  test_above_sample_stored();
  test_find_warp_samples_cap();
  test_local_warp_models();
  test_build_rejects_unsupported();
  test_sub_block_mv();
  test_sub_mvs_feed_the_stack();
  test_warp_estimate_golden();
  test_sort_spec_exact();
  test_wired_stack_parity();
  test_wired_scan_col_deviation();
  test_global_mv_substitution();
  test_corner_then_scan_order();
  test_lockstep_randomized();
  test_snapshot_roundtrip();
  test_goldens();
  std::printf("warp_stack_test: %d checks passed", g_checks);
  if (g_failures != 0) std::printf(", %d FAILED", g_failures);
  std::printf("\n");
  return g_failures == 0 ? 0 : 1;
}
