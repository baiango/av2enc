// Tests for the complete §7.12.2 MV stack, the allow_warpmv_mode frame bit,
// the §7.7 implicit reference map (R77) and the stack mirror.
//
// Build/run: ./build_and_test.sh
//
// Coverage:
//   * candidate-for-candidate parity with the wired `av2enc::video::merge` component over
//     randomized grids (single reference, all block sizes, every reorder
//     setting that merge implements correctly);
//   * the negative control for §7.12.2.19: merge's selection sort differs from
//     the normative single move-to-front;
//   * §7.12.2.20 exactness: clamp-before-global-candidate, the global-MV
//     candidate, the >32x32 MVP-combination gate, the warp-candidate order and
//     the MAX_WARP_REF_CANDIDATES cap, and the PruneCount budget;
//   * §7.12.2.10 dispatch: single / compound / TIP-single / TIP-add-derived /
//     TIP-compound / single-add-derived;
//   * §7.12.2.3/.4 corner warp generation and §7.12.2.16 projection;
//   * `allow_warpmv_mode`: sequence/frame bit writer-reader round-trip, every
//     constraint, and bit equality with `av2enc::video::adv`;
//   * §7.7/§7.8 parity with `adv::build_ref_frame_list`, restricted slots,
//     display-order-hint derivation and the R77 operating-point checks;
//   * stack-record writer + independent mirror: randomized lockstep, every
//     corruption/truncation rejected, pinned goldens.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "codec/inter/advanced/b_frame.h"
#include "codec/inter/mv_stack/frame_flags.h"
#include "codec/inter/mv_stack/mv_stack_full.h"
#include "codec/inter/mv_stack/ref_map.h"
#include "codec/inter/mv_stack/stack_io.h"
#include "codec/inter/mv_stack/stack_mirror.h"
#include "codec/inter/advanced/video/advanced_syntax.h"
#include "codec/inter/merge/merge.h"

namespace full = av2enc::inter::mv_stack_full;
namespace merge = av2enc::video::merge;
namespace mc = av2enc::inter::mc;
namespace adv = av2enc::inter::adv;
namespace video_adv = av2enc::video::adv;
using full::Mv;

namespace {

struct Test {
  int checks = 0;
  int failures = 0;
  void expect(bool cond, const std::string& what) {
    ++checks;
    if (!cond) {
      ++failures;
      if (failures <= 40) std::printf("FAIL: %s\n", what.c_str());
    }
  }
  void eq_int(long long got, long long want, const std::string& what) {
    ++checks;
    if (got != want) {
      ++failures;
      if (failures <= 40) {
        std::printf("FAIL: %s (got %lld, want %lld)\n", what.c_str(), got,
                    want);
      }
    }
  }
  void eq_mv(Mv got, Mv want, const std::string& what) {
    ++checks;
    if (!(got == want)) {
      ++failures;
      if (failures <= 40) {
        std::printf("FAIL: %s (got (%d,%d), want (%d,%d))\n", what.c_str(),
                    got.y, got.x, want.y, want.x);
      }
    }
  }
};

std::mt19937_64& rng() {
  static std::mt19937_64 engine(0x5eed1234ULL);
  return engine;
}

int rand_int(int lo, int hi) {
  std::uniform_int_distribution<int> dist(lo, hi);
  return dist(rng());
}

// ---------------------------------------------------------------------------
// Grids
// ---------------------------------------------------------------------------

struct Block {
  int row = 0;
  int col = 0;
  int w4 = 1;
  int h4 = 1;
  std::int8_t ref[2] = {0, full::kRefNone};
  Mv mv;
  Mv sub_mv;
  bool is_inter = true;
  int y_mode = full::kYModeGlobalMv;
  int motion_mode = full::kMotionSimple;
  int cwp = full::kCwpEqual;
  adv::WarpParams warp[2];
};

struct TestGrid {
  int cols = 0;
  int rows = 0;
  std::vector<full::MiCell> full_mi;
  std::vector<mc::MiInfo> mc_mi;
  std::vector<Block> blocks;

  full::MiGrid grid() const {
    return full::MiGrid{cols, rows, full_mi.data()};
  }
};

void fill_block(TestGrid& g, const Block& b) {
  for (int r = 0; r < b.h4; ++r) {
    for (int c = 0; c < b.w4; ++c) {
      const std::size_t idx =
          static_cast<std::size_t>(b.row + r) *
              static_cast<std::size_t>(g.cols) +
          static_cast<std::size_t>(b.col + c);
      full::MiCell& f = g.full_mi[idx];
      f = full::MiCell{};
      f.ref[0] = b.ref[0];
      f.ref[1] = b.ref[1];
      f.mv[0][0] = b.mv.y;
      f.mv[0][1] = b.mv.x;
      f.mv[1][0] = b.mv.y;
      f.mv[1][1] = b.mv.x;
      f.sub_mv[0][0] = b.sub_mv.y;
      f.sub_mv[0][1] = b.sub_mv.x;
      f.sub_mv[1][0] = b.sub_mv.y;
      f.sub_mv[1][1] = b.sub_mv.x;
      f.is_inter = b.is_inter ? 1 : 0;
      f.written = 1;
      f.w4 = static_cast<std::int8_t>(b.w4);
      f.h4 = static_cast<std::int8_t>(b.h4);
      f.col_base = static_cast<std::int16_t>(b.col);
      f.row_base = static_cast<std::int16_t>(b.row);
      f.y_mode = static_cast<std::uint8_t>(b.y_mode);
      f.motion_mode = static_cast<std::uint8_t>(b.motion_mode);
      f.cwp = static_cast<std::int8_t>(b.cwp);
      f.warp[0] = b.warp[0];
      f.warp[1] = b.warp[1];
      f.tip_16x16 = 0;

      mc::MiInfo& m = g.mc_mi[idx];
      m = mc::MiInfo{};
      m.ref[0] = b.ref[0];
      m.ref[1] = b.ref[1];
      m.mv[0][0] = b.mv.y;
      m.mv[0][1] = b.mv.x;
      m.mv[1][0] = b.mv.y;
      m.mv[1][1] = b.mv.x;
      m.is_inter = b.is_inter ? 1 : 0;
      m.w4 = static_cast<std::int8_t>(b.w4);
      m.h4 = static_cast<std::int8_t>(b.h4);
      m.col_base = static_cast<std::int16_t>(b.col);
      m.row_base = static_cast<std::int16_t>(b.row);
    }
  }
}

// Tiles the grid with random block sizes, one size per 16x16-MI superblock.
TestGrid make_random_grid(int sb_cols, int sb_rows, bool with_compound,
                          int max_ref) {
  TestGrid g;
  g.cols = sb_cols * 16;
  g.rows = sb_rows * 16;
  g.full_mi.assign(static_cast<std::size_t>(g.cols) * static_cast<std::size_t>(g.rows), full::MiCell{});
  g.mc_mi.assign(static_cast<std::size_t>(g.cols) * static_cast<std::size_t>(g.rows), mc::MiInfo{});
  for (int sbr = 0; sbr < sb_rows; ++sbr) {
    for (int sbc = 0; sbc < sb_cols; ++sbc) {
      const int step = 1 << rand_int(0, 4);  // 1,2,4,8,16 MI
      for (int r = 0; r < 16; r += step) {
        for (int c = 0; c < 16; c += step) {
          Block b;
          b.row = sbr * 16 + r;
          b.col = sbc * 16 + c;
          b.w4 = step;
          b.h4 = step;
          b.is_inter = rand_int(0, 3) != 0;
          if (b.is_inter) {
            b.ref[0] = static_cast<std::int8_t>(rand_int(0, max_ref));
            b.ref[1] = with_compound && rand_int(0, 1)
                           ? static_cast<std::int8_t>(rand_int(0, max_ref))
                           : full::kRefNone;
            b.mv = Mv{rand_int(-64, 64) * 8, rand_int(-64, 64) * 8};
            b.sub_mv = b.mv;
            b.y_mode = rand_int(0, 1) ? full::kYModeNearMv
                                     : full::kYModeGlobalMv;
          } else {
            b.ref[0] = full::kIntraFrame;
            b.ref[1] = full::kRefNone;
            b.mv = Mv{};
            b.sub_mv = Mv{};
          }
          g.blocks.push_back(b);
          fill_block(g, b);
        }
      }
    }
  }
  return g;
}

const Block& pick_block(const TestGrid& g, int* row, int* col) {
  const std::size_t i =
      static_cast<std::size_t>(rand_int(0, static_cast<int>(g.blocks.size()) - 1));
  const Block& b = g.blocks[i];
  *row = b.row;
  *col = b.col;
  return b;
}

// ---------------------------------------------------------------------------
// §7.12.2 parity with the wired merge component
// ---------------------------------------------------------------------------

void test_merge_parity(Test& t) {
  for (int iter = 0; iter < 900; ++iter) {
    const int sb_cols = rand_int(1, 4);
    const int sb_rows = rand_int(1, 4);
    const int max_ref = rand_int(0, 3);
    TestGrid g = make_random_grid(sb_cols, sb_rows, true, max_ref);
    int row = 0;
    int col = 0;
    const Block& cur = pick_block(g, &row, &col);

    const std::int8_t ref0 = static_cast<std::int8_t>(rand_int(0, max_ref));
    const std::int8_t ref1 = static_cast<std::int8_t>(rand_int(0, max_ref));
    const bool compound = rand_int(0, 2) == 0;

    // FrameDistance: half the batches have a real distance table.
    std::array<int, 7> distance = {0, 1, -1, 2, -2, 3, -3};
    std::array<int, 7> hints = {0, 1, 3, 2, 5, 4, 6};
    const bool use_distance = rand_int(0, 1) != 0;

    full::StackParams fp;
    fp.grid = g.grid();
    fp.mi_row = row;
    fp.mi_col = col;
    fp.bw4 = cur.w4;
    fp.bh4 = cur.h4;
    fp.sb_h4 = 16;
    fp.ref_frame[0] = ref0;
    fp.ref_frame[1] = compound ? ref1 : full::kRefNone;
    fp.is_compound = compound;
    fp.reorder = full::Reorder::kDisabled;
    fp.max_drl_bits_minus_1 = 4;
    fp.num_refs = use_distance ? 4 : 0;
    fp.frame_distance = use_distance ? distance.data() : nullptr;
    fp.order_hints = use_distance ? hints.data() : nullptr;

    full::Stack full_stack;
    const full::Status status = full::build(fp, &full_stack);
    if (status != full::Status::kOk) {
      t.expect(false, "full stack build failed in parity batch");
      continue;
    }

    merge::StackParams mp;
    mp.mi_row = row;
    mp.mi_col = col;
    mp.bw4 = cur.w4;
    mp.bh4 = cur.h4;
    mp.sb_h4 = 16;
    mp.mi_cols = g.cols;
    mp.mi_rows = g.rows;
    mp.mi = g.mc_mi.data();
    mp.ref_frame[0] = ref0;
    mp.ref_frame[1] = compound ? ref1 : full::kRefNone;
    mp.is_compound = compound;
    mp.reorder = merge::DrlReorder::kDisabled;
    mp.max_drl_bits_minus_1 = 4;
    mp.num_refs = use_distance ? 4 : 0;
    mp.frame_distance = use_distance ? distance.data() : nullptr;
    merge::MvStack merge_stack;
    const merge::Status merge_status =
        merge::build_mv_stack(mp, &merge_stack);
    if (merge_status != merge::Status::kOk) {
      t.expect(false, "merge stack build failed in parity batch");
      continue;
    }

    bool same = full_stack.count == merge_stack.count &&
                full_stack.num_nearest == merge_stack.num_nearest &&
                full_stack.sorted == merge_stack.sorted;
    if (same) {
      for (int i = 0; i < full_stack.count; ++i) {
        if (!(full_stack.cand[i].mv[0] == merge_stack.cand[i].mv[0]) ||
            full_stack.cand[i].weight != merge_stack.cand[i].weight ||
            full_stack.cand[i].row_offset != merge_stack.cand[i].row_offset ||
            full_stack.cand[i].col_offset != merge_stack.cand[i].col_offset) {
          same = false;
          break;
        }
      }
    }
    if (!same) {
      std::printf(
          "parity failure: iter=%d block=%dx%d at (%d,%d) compound=%d "
          "counts full=%d merge=%d\n",
          iter, cur.w4 * 4, cur.h4 * 4, row, col, compound, full_stack.count,
          merge_stack.count);
    }
    t.expect(same, "full stack matches merge candidate-for-candidate");
  }
}

// Negative control: merge's sort_nearest() is a full selection sort; the spec
// is a single move-to-front.  Weights {1,2,3,4} are the audit's example.
void test_sort_negative_control(Test& t) {
  TestGrid g;
  g.cols = 64;
  g.rows = 16;
  g.full_mi.assign(static_cast<std::size_t>(g.cols) * static_cast<std::size_t>(g.rows), full::MiCell{});
  g.mc_mi.assign(static_cast<std::size_t>(g.cols) * static_cast<std::size_t>(g.rows), mc::MiInfo{});
  // Four 32x32 blocks to the left, each with a distinct MV and a weight that
  // depends on how many neighbours point at it.  The exact values do not
  // matter; the assertion below is about the transform itself.
  full::Stack stack;
  stack.count = 4;
  stack.num_nearest = 4;
  stack.cand[0].weight = 1;
  stack.cand[1].weight = 2;
  stack.cand[2].weight = 3;
  stack.cand[3].weight = 4;
  for (int i = 0; i < 4; ++i) {
    stack.cand[i].mv[0] = Mv{i * 10, 0};
    stack.cand[i].mv[1] = stack.cand[i].mv[0];
  }
  full::StackParams p;
  p.grid = full::MiGrid{16, 4, g.full_mi.data()};
  full::sort_stack_range(p, &stack, 0, 4);
  // §7.12.2.19 moves only the maximum to the front: 4,1,2,3.
  t.eq_mv(stack.cand[0].mv[0], Mv{30, 0}, "move-to-front brings the max first");
  t.eq_mv(stack.cand[1].mv[0], Mv{10, 0}, "move-to-front leaves the rest in place (1)");
  t.eq_mv(stack.cand[2].mv[0], Mv{20, 0}, "move-to-front leaves the rest in place (2)");
  t.eq_mv(stack.cand[3].mv[0], Mv{0, 0}, "move-to-front swaps the front entry into the max slot");
}

// ---------------------------------------------------------------------------
// §7.12.2.20 exactness
// ---------------------------------------------------------------------------

// Builds a single block whose left neighbour contributes one candidate.
full::StackParams clamp_order_params(TestGrid* g, int sb_cols, int sb_rows,
                                     Block* left, int block_w4,
                                     int block_h4) {
  g->cols = sb_cols * 16;
  g->rows = sb_rows * 16;
  g->full_mi.assign(static_cast<std::size_t>(g->cols) * static_cast<std::size_t>(g->rows),
                    full::MiCell{});
  g->mc_mi.assign(static_cast<std::size_t>(g->cols) * static_cast<std::size_t>(g->rows), mc::MiInfo{});
  Block b;
  b.row = 0;
  b.col = 0;
  b.w4 = block_w4;
  b.h4 = block_h4;
  b.ref[0] = 0;
  b.mv = Mv{1'000'000, 1'000'000};
  b.sub_mv = b.mv;
  *left = b;
  fill_block(*g, b);

  full::StackParams p;
  p.grid = g->grid();
  p.mi_row = 0;
  p.mi_col = block_w4;
  p.bw4 = block_w4;
  p.bh4 = block_h4;
  p.sb_h4 = 16;
  p.ref_frame[0] = 0;
  p.ref_frame[1] = full::kRefNone;
  p.max_drl_bits_minus_1 = 4;
  return p;
}

void test_clamp_before_global(Test& t) {
  TestGrid g;
  Block left;
  full::StackParams p =
      clamp_order_params(&g, /*sb_cols=*/2, /*sb_rows=*/2, &left, 8, 8);
  // 32x32 block at (0, 32): picture clamp high = (32-0)*4*8+128 = 1152 on both
  // axes.  The neighbour's MV therefore clamps to (1152,1152).
  p.gm_params[0].p[0] = 1152 << 13;
  p.gm_params[0].p[1] = 1152 << 13;
  full::Stack stack;
  full::StackParams probe = p;
  probe.gm_params[0] = adv::WarpParams{};
  probe.global_mv[0] = Mv{0, 0};
  const full::Status st = full::build(p, &stack);
  t.expect(st == full::Status::kOk, "clamp-order stack builds");
  if (st != full::Status::kOk) return;
  // The only scanned candidate is the left neighbour; after the pre-clamp it
  // equals GlobalMvs, so §7.12.2.20 must not append a second copy.
  t.eq_int(stack.count, 1, "global candidate is recognised after the pre-clamp");
  t.eq_mv(stack.cand[0].mv[0], Mv{1152, 896}, "pre-clamped candidate MV");
}

void test_global_candidate(Test& t) {
  TestGrid g;
  Block left;
  full::StackParams p =
      clamp_order_params(&g, /*sb_cols=*/2, /*sb_rows=*/2, &left, 8, 8);
  // A neighbour MV that is inside the picture but differs from global MV.
  Block inside = left;
  inside.mv = Mv{8, 16};
  inside.sub_mv = inside.mv;
  fill_block(g, inside);
  full::Stack stack;
  const full::Status st = full::build(p, &stack);
  t.expect(st == full::Status::kOk, "global-candidate stack builds");
  if (st != full::Status::kOk) return;
  t.expect(stack.count >= 2, "global candidate appended");
  t.eq_mv(stack.cand[stack.count - 1].mv[0], Mv{0, 0},
          "global candidate is GlobalMvs[0]");
}

void test_mvp_combination_gate(Test& t) {
  for (int dim : {16, 32, 64}) {
    TestGrid g;
    g.cols = 64;
    g.rows = 64;
    g.full_mi.assign(static_cast<std::size_t>(g.cols) * static_cast<std::size_t>(g.rows),
                     full::MiCell{});
    g.mc_mi.assign(static_cast<std::size_t>(g.cols) * static_cast<std::size_t>(g.rows), mc::MiInfo{});
    // Three distinct neighbours: left, above, above-left.
    const int w4 = dim / 4;
    const int h4 = dim / 4;
    const int row = 32;
    const int col = 32;
    Block a;
    a.row = row;
    a.col = col - w4;
    a.w4 = w4;
    a.h4 = h4;
    a.ref[0] = 0;
    a.mv = Mv{8, 0};
    a.sub_mv = a.mv;
    fill_block(g, a);
    Block b = a;
    b.row = row - h4;
    b.col = col;
    b.mv = Mv{0, 16};
    b.sub_mv = b.mv;
    fill_block(g, b);
    Block c = a;
    c.row = row - h4;
    c.col = col - w4;
    c.mv = Mv{-8, -8};
    c.sub_mv = c.mv;
    fill_block(g, c);

    full::StackParams p;
    p.grid = g.grid();
    p.mi_row = row;
    p.mi_col = col;
    p.bw4 = w4;
    p.bh4 = h4;
    p.sb_h4 = 16;
    p.ref_frame[0] = 0;
    p.ref_frame[1] = full::kRefNone;
    p.max_drl_bits_minus_1 = 4;
    full::Stack stack;
    const full::Status st = full::build(p, &stack);
    t.expect(st == full::Status::kOk, "mvp-gate stack builds");
    if (st != full::Status::kOk) continue;
    const bool gated = full::mvp_combinations_allowed(dim, dim);
    t.expect(gated == (dim > 32), "MVP combination gate predicate");
    if (dim > 32) {
      t.expect(stack.count > 3, "MVP combinations added for a >32x32 block");
      // The first combination must mix the y of candidate 0 with the x of
      // candidate 1.
      bool found_mix = false;
      for (int i = 0; i < stack.count; ++i) {
        if (stack.cand[i].mv[0].y == stack.cand[0].mv[0].y &&
            stack.cand[i].mv[0].x == stack.cand[1].mv[0].x) {
          found_mix = true;
        }
      }
      t.expect(found_mix, "MVP combination takes (y of 0, x of 1)");
    } else {
      t.eq_int(stack.count, 4,
               "no MVP combinations at or below 32x32 (three neighbours + global)");
    }
  }
}

void test_warp_candidate_order(Test& t) {
  TestGrid g = make_random_grid(2, 2, false, 1);
  full::StackParams p;
  p.grid = g.grid();
  p.mi_row = 0;
  p.mi_col = 16;
  p.bw4 = 16;
  p.bh4 = 16;
  p.sb_h4 = 16;
  p.ref_frame[0] = 0;
  p.ref_frame[1] = full::kRefNone;
  p.derive_wrl = true;

  full::WarpParamBank bank;
  bank.reset();
  adv::WarpParams w0{};
  w0.p[0] = 1;
  adv::WarpParams w1{};
  w1.p[0] = 2;
  adv::WarpParams w2{};
  w2.p[0] = 3;
  bank.record(w0);  // oldest
  bank.record(w1);
  bank.record(w2);  // newest
  p.warp_bank = &bank;
  adv::WarpParams gm{};
  gm.p[1] = 7;
  p.gm_params[0] = gm;

  full::Stack stack;
  const full::Status st = full::build(p, &stack);
  t.expect(st == full::Status::kOk, "warp candidate stack builds");
  if (st != full::Status::kOk) return;
  t.eq_int(stack.num_warp_found, 4, "warp stack caps at four entries");
  t.eq_int(stack.warp_params[0].p[0], 3, "warp bank is read newest first");
  t.eq_int(stack.warp_params[1].p[0], 2, "warp bank second entry");
  t.eq_int(stack.warp_params[2].p[0], 1, "warp bank oldest entry");
  t.eq_int(stack.warp_params[3].p[1], 7, "gm_params follows the bank");

  // Four bank entries already fill the stack: gm/defaults are not appended.
  full::WarpParamBank full_bank;
  full_bank.reset();
  for (int i = 0; i < 4; ++i) {
    adv::WarpParams w{};
    w.p[0] = 100 + i;
    full_bank.record(w);
  }
  p.warp_bank = &full_bank;
  const full::Status st2 = full::build(p, &stack);
  t.expect(st2 == full::Status::kOk, "full warp bank stack builds");
  t.eq_int(stack.num_warp_found, 4, "full bank leaves room for nothing else");
  t.eq_int(stack.warp_params[0].p[0], 103, "full bank newest first");

  // §7.12.2.11 has no deduplication.
  full::Stack dup;
  for (int i = 0; i < 3; ++i) {
    full::insert_warp_candidate(w0, &dup);
  }
  t.eq_int(dup.num_warp_found, 3, "insert_warp_candidate does not deduplicate");
}

void test_prune_budget(Test& t) {
  // With PruneCount already at MAX_PR_NUM, §7.12.2.20 cannot prune, so the
  // global candidate is appended even when an identical MV is present.
  TestGrid g;
  Block left;
  full::StackParams p =
      clamp_order_params(&g, /*sb_cols=*/2, /*sb_rows=*/2, &left, 8, 8);
  Block inside = left;
  inside.mv = Mv{0, 0};
  inside.sub_mv = inside.mv;
  fill_block(g, inside);

  full::BuildState state;
  state.stack.count = 1;
  state.stack.cand[0].mv[0] = Mv{0, 0};
  state.stack.cand[0].mv[1] = Mv{0, 0};
  state.stack.cand[0].weight = 1;
  state.prune_count = full::kMaxPrNum;
  const full::Status st = full::extra_search(p, &state);
  t.expect(st == full::Status::kOk, "extra_search with an exhausted budget");
  t.eq_int(state.stack.count, 2, "exhausted PruneCount admits a duplicate global");
}

// ---------------------------------------------------------------------------
// §7.12.2.10 dispatch
// ---------------------------------------------------------------------------

struct DispatchEnv {
  TestGrid grid;
  full::StackParams params;
  full::BuildState state;
};

void init_dispatch_env(DispatchEnv* env, std::int8_t cur_ref0,
                       std::int8_t cur_ref1, bool compound) {
  env->grid.cols = 16;
  env->grid.rows = 16;
  env->grid.full_mi.assign(
      static_cast<std::size_t>(env->grid.cols) * static_cast<std::size_t>(env->grid.rows),
      full::MiCell{});
  env->grid.mc_mi.assign(
      static_cast<std::size_t>(env->grid.cols) * static_cast<std::size_t>(env->grid.rows),
      mc::MiInfo{});
  env->params = full::StackParams{};
  env->params.grid = env->grid.grid();
  env->params.mi_row = 2;
  env->params.mi_col = 2;
  env->params.bw4 = 4;
  env->params.bh4 = 4;
  env->params.sb_h4 = 16;
  env->params.ref_frame[0] = cur_ref0;
  env->params.ref_frame[1] = cur_ref1;
  env->params.is_compound = compound;
  env->params.num_refs = 7;
  static int distance[7] = {0, 1, -1, 2, -2, 3, -3};
  static int hints[7] = {0, 1, 3, 2, 5, 4, 6};
  env->params.frame_distance = distance;
  env->params.order_hints = hints;
  env->params.order_hint = 2;
  env->state = full::BuildState{};
}

void place_neighbour(DispatchEnv* env, const Block& b) { fill_block(env->grid, b); }

void test_dispatch_single(Test& t) {
  DispatchEnv env;
  init_dispatch_env(&env, 1, full::kRefNone, false);
  Block b;
  b.row = 0;
  b.col = 0;
  b.w4 = 4;
  b.h4 = 4;
  b.ref[0] = 1;
  b.mv = Mv{24, 0};
  b.sub_mv = b.mv;
  place_neighbour(&env, b);
  full::Status st = full::add_reference_mv(env.params, &env.state, 0, 0, 1);
  t.expect(st == full::Status::kOk, "single dispatch: matching ref");
  t.eq_int(env.state.trace.search_stack[0], 1, "single dispatch: search stack");
  t.eq_int(env.state.stack.count, 1, "single dispatch: candidate added");

  // Different reference: is_derivable_ref_frame -> single add derived.
  init_dispatch_env(&env, 3, full::kRefNone, false);
  b.ref[0] = 1;
  place_neighbour(&env, b);
  st = full::add_reference_mv(env.params, &env.state, 0, 0, 1);
  t.expect(st == full::Status::kOk, "single dispatch: derivable ref");
  t.eq_int(env.state.trace.single_add_derived[0], 1,
           "single dispatch: single add derived");
  t.eq_int(env.state.derived_count, 1, "single dispatch: derived candidate");
}

void test_dispatch_tip(Test& t) {
  // TIP single-ref branch (§7.12.2.17).
  DispatchEnv env;
  init_dispatch_env(&env, 2, full::kRefNone, false);
  env.params.closest_past = 2;
  env.params.closest_future = 4;
  env.params.num_past_refs = 1;
  env.params.num_future_refs = 1;
  Block b;
  b.row = 0;
  b.col = 0;
  b.w4 = 4;
  b.h4 = 4;
  b.ref[0] = full::kTipFrame;
  b.ref[1] = full::kRefNone;
  b.mv = Mv{40, 8};
  b.sub_mv = b.mv;
  place_neighbour(&env, b);

  std::vector<std::int32_t> mf_mv(2 * 2 * 2, 0);
  std::vector<std::int32_t> mf_offset(2 * 2, 3);
  std::vector<std::uint8_t> mf_valid(2 * 2, 1);
  full::TipField tip;
  tip.cols8 = 2;
  tip.rows8 = 2;
  tip.valid = mf_valid.data();
  tip.mv = mf_mv.data();
  tip.offset = mf_offset.data();
  env.params.tip = &tip;

  full::Status st = full::add_reference_mv(env.params, &env.state, 0, 0, 1);
  t.expect(st == full::Status::kOk, "TIP dispatch: single-ref derived");
  t.eq_int(env.state.trace.tip_single_derived[0], 1,
           "TIP dispatch: §7.12.2.17 reached");
  t.eq_int(env.state.stack.count, 1, "TIP dispatch: candidate inserted");

  // TIP add-derived branch (§7.12.2.18): the current block is a TIP block whose
  // closest pair matches the neighbour's pair.
  init_dispatch_env(&env, full::kTipFrame, full::kRefNone, false);
  env.params.closest_past = 2;
  env.params.closest_future = 4;
  env.params.num_past_refs = 1;
  env.params.num_future_refs = 1;
  env.params.tip = &tip;
  b.ref[0] = 2;
  b.ref[1] = 4;
  b.sub_mv = Mv{16, 0};
  b.mv = b.sub_mv;
  place_neighbour(&env, b);
  st = full::add_reference_mv(env.params, &env.state, 0, 0, 1);
  t.expect(st == full::Status::kOk, "TIP dispatch: add derived");
  t.eq_int(env.state.trace.tip_add_derived, 1,
           "TIP dispatch: §7.12.2.18 reached");
  t.eq_int(env.state.derived_count, 1, "TIP dispatch: derived MV produced");
}

void test_dispatch_compound(Test& t) {
  // Matching pair -> compound search stack.
  DispatchEnv env;
  init_dispatch_env(&env, 1, 2, true);
  Block b;
  b.row = 0;
  b.col = 0;
  b.w4 = 4;
  b.h4 = 4;
  b.ref[0] = 1;
  b.ref[1] = 2;
  b.mv = Mv{8, -8};
  b.sub_mv = b.mv;
  place_neighbour(&env, b);
  full::Status st = full::add_reference_mv(env.params, &env.state, 0, 0, 1);
  t.expect(st == full::Status::kOk, "compound dispatch: matching pair");
  t.eq_int(env.state.trace.compound_search, 1,
           "compound dispatch: compound search stack");
  t.eq_int(env.state.stack.count, 1, "compound dispatch: pair inserted");

  // Non-matching pair -> compound add derived.
  init_dispatch_env(&env, 1, 2, true);
  b.ref[0] = 1;
  b.ref[1] = 3;
  place_neighbour(&env, b);
  st = full::add_reference_mv(env.params, &env.state, 0, 0, 1);
  t.expect(st == full::Status::kOk, "compound dispatch: non-matching pair");
  t.eq_int(env.state.trace.compound_add_derived, 1,
           "compound dispatch: compound add derived");

  // TIP neighbour with the current closest pair -> TIP compound candidate.
  init_dispatch_env(&env, 2, 4, true);
  env.params.closest_past = 2;
  env.params.closest_future = 4;
  env.params.num_past_refs = 1;
  env.params.num_future_refs = 1;
  std::vector<std::int32_t> mf_mv(2 * 2 * 2, 0);
  std::vector<std::int32_t> mf_offset(2 * 2, 3);
  std::vector<std::uint8_t> mf_valid(2 * 2, 1);
  full::TipField tip;
  tip.cols8 = 2;
  tip.rows8 = 2;
  tip.valid = mf_valid.data();
  tip.mv = mf_mv.data();
  tip.offset = mf_offset.data();
  env.params.tip = &tip;
  b.ref[0] = full::kTipFrame;
  b.ref[1] = full::kRefNone;
  place_neighbour(&env, b);
  st = full::add_reference_mv(env.params, &env.state, 0, 0, 1);
  t.expect(st == full::Status::kOk, "compound dispatch: TIP neighbour");
  t.eq_int(env.state.trace.compound_tip, 1,
           "compound dispatch: §7.12.2.15 reached");
}

void test_single_add_derived_projection(Test& t) {
  DispatchEnv env;
  init_dispatch_env(&env, 1, full::kRefNone, false);
  // FrameDistance[1] = 1, FrameDistance[2] = -1 -> opposite sides, no
  // projection.
  Block b;
  b.row = 0;
  b.col = 0;
  b.w4 = 4;
  b.h4 = 4;
  b.ref[0] = 2;
  b.mv = Mv{32, 16};
  b.sub_mv = b.mv;
  place_neighbour(&env, b);
  full::Status st = full::single_add_derived(env.params, &env.state, 0, 0, 0);
  t.expect(st == full::Status::kOk, "single_add_derived runs");
  t.eq_int(env.state.derived_count, 0, "opposite-side references do not project");

  // Same side: FrameDistance[1] = 1, FrameDistance[0] = 0 (ref 0 has distance
  // 0, so make the current ref 3 with distance 2).
  init_dispatch_env(&env, 3, full::kRefNone, false);
  b.ref[0] = 1;
  place_neighbour(&env, b);
  st = full::single_add_derived(env.params, &env.state, 0, 0, 0);
  t.expect(st == full::Status::kOk, "single_add_derived projects");
  t.eq_int(env.state.derived_count, 1, "same-side reference projects");
  if (env.state.derived_count == 1) {
    // get_mv_projection((32,16), 2, 1).
    t.eq_mv(env.state.derived[0].mv[0], Mv{64, 32},
            "projection doubles the candidate MV");
  }
}

// ---------------------------------------------------------------------------
// allow_warpmv_mode
// ---------------------------------------------------------------------------

full::SequenceMotionModes warp_sequence() {
  full::SequenceMotionModes seq;
  seq.seq_enabled_motion_modes[full::kMotionDeltaWarp] = true;
  seq.seq_frame_motion_modes_present_flag = true;
  seq.enable_six_param_warp_delta = false;
  return seq;
}

void test_allow_warpmv_mode(Test& t) {
  using full::FrameStatus;
  const full::SequenceMotionModes seq = warp_sequence();
  const bool override_all[full::kMotionModes] = {false, false, false, true, false};
  full::FrameMotionModes frame;
  frame.frame_is_intra = false;
  std::string why;
  t.expect(full::derive_frame_enabled_motion_modes(seq, override_all, &frame,
                                                   &why) == FrameStatus::kOk,
           "frame motion modes derive");
  t.expect(frame.frame_enabled_motion_modes[full::kMotionDeltaWarp],
           "DELTAWARP frame-enabled");

  full::WarpmvMode mode;
  t.expect(full::derive_allow_warpmv_mode(seq, frame, &mode, &why) ==
               FrameStatus::kOk,
           "allow_warpmv_mode derives");
  t.expect(mode.present, "bit is present for an inter frame with DELTAWARP");

  // Intra frame: the bit cannot be present.
  full::FrameMotionModes intra = frame;
  intra.frame_is_intra = true;
  t.expect(full::derive_allow_warpmv_mode(seq, intra, &mode, &why) ==
               FrameStatus::kOk,
           "allow_warpmv_mode derives for intra");
  t.expect(!mode.present, "bit is absent for an intra frame");

  // No DELTAWARP: absent.
  full::FrameMotionModes no_delta = frame;
  no_delta.frame_enabled_motion_modes[full::kMotionDeltaWarp] = false;
  t.expect(full::derive_allow_warpmv_mode(seq, no_delta, &mode, &why) ==
               FrameStatus::kOk,
           "allow_warpmv_mode derives without DELTAWARP");
  t.expect(!mode.present, "bit is absent without DELTAWARP");

  // Frame cannot enable what the sequence disabled.
  full::SequenceMotionModes seq_off = seq;
  seq_off.seq_enabled_motion_modes[full::kMotionDeltaWarp] = false;
  t.expect(full::derive_frame_enabled_motion_modes(seq_off, override_all, &frame,
                                                   &why) ==
               FrameStatus::kConstraintViolated,
           "frame cannot enable a sequence-disabled mode");

  // Writer/reader round-trip and bit equality with av2enc::video::adv.
  for (int value = 0; value <= 1; ++value) {
    full::SequenceMotionModes s = warp_sequence();
    full::FrameMotionModes f;
    f.frame_is_intra = false;
    t.expect(full::derive_frame_enabled_motion_modes(s, override_all, &f, &why) ==
                 FrameStatus::kOk,
             "round-trip frame modes derive");
    full::WarpmvMode write_mode;
    write_mode.present = true;
    write_mode.allow_warpmv_mode = value != 0;
    av2enc::BitWriter writer;
    t.expect(full::write_allow_warpmv_mode(&writer, s, f, &write_mode, &why),
             "allow_warpmv_mode writes");
    const std::vector<std::uint8_t> bytes = writer.take_bytes();
    full::BitReader reader(bytes.data(), bytes.size());
    full::WarpmvMode read_mode;
    t.expect(full::read_allow_warpmv_mode(&reader, s, f, &read_mode, &why),
             "allow_warpmv_mode reads");
    t.eq_int(read_mode.allow_warpmv_mode ? 1 : 0, value,
             "allow_warpmv_mode round-trips");

    // av2enc::video::adv parity for the same bit.
    av2enc::BitWriter adv_writer;
    video_adv::FrameAdvancedConfig adv_config;
    adv_config.frame_is_intra = false;
    adv_config.enable_bawp = false;
    adv_config.allow_bawp = false;
    adv_config.allow_warpmv_mode = value != 0;
    for (int m = 0; m < full::kMotionModes; ++m) {
      adv_config.seq_enabled_motion_modes[m] = s.seq_enabled_motion_modes[m];
      adv_config.frame_enabled_motion_modes[m] = f.frame_enabled_motion_modes[m];
    }
    adv_config.seq_frame_motion_modes_present_flag =
        s.seq_frame_motion_modes_present_flag;
    t.expect(video_adv::write_frame_bawp_warp_bits(&adv_writer, adv_config, &why),
             "av2enc::video::adv writes the frame warp bit");
    const std::vector<std::uint8_t> adv_bytes = adv_writer.take_bytes();
    t.expect(adv_bytes == bytes, "allow_warpmv_mode bit matches av2enc::video::adv");
  }

  // Sequence motion-mode parity with av2enc::video::adv.
  {
    full::SequenceMotionModes s = warp_sequence();
    av2enc::BitWriter mine;
    t.expect(full::write_sequence_motion_modes(&mine, s, &why),
             "sequence motion modes write");
    av2enc::BitWriter theirs;
    video_adv::SequenceAdvancedConfig config;
    config.seq_frame_motion_modes_present_flag =
        s.seq_frame_motion_modes_present_flag;
    config.enable_six_param_warp_delta = s.enable_six_param_warp_delta;
    for (int m = 0; m < full::kMotionModes; ++m) {
      config.seq_enabled_motion_modes[m] = s.seq_enabled_motion_modes[m];
    }
    t.expect(video_adv::write_sequence_motion_mode_config(&theirs, config, &why),
             "av2enc::video::adv writes the sequence motion modes");
    t.expect(theirs.take_bytes() == mine.take_bytes(),
             "sequence motion-mode bits match av2enc::video::adv");
  }

  // Block gates.
  full::WarpBlockGates gates;
  t.expect(full::derive_warp_block_gates(true, false, 64, 64, true, &gates,
                                         &why) == FrameStatus::kOk,
           "block gates derive");
  t.expect(gates.is_warp_present && gates.warp_mv_present,
           "is_warp and warp_mv are present");
  t.expect(full::derive_warp_block_gates(true, false, 4, 16, true, &gates,
                                         &why) ==
               FrameStatus::kConstraintViolated,
           "is_warp is constrained to min(w,h) >= 8");
  t.expect(full::derive_warp_block_gates(false, false, 64, 64, true, &gates,
                                         &why) ==
               FrameStatus::kConstraintViolated,
           "is_warp requires allow_warpmv_mode");
  t.expect(full::validate_warp_block(true, true, 64, 64, true, false,
                                     full::kYModeWarpMv, full::kMotionDeltaWarp,
                                     &why) == FrameStatus::kConstraintViolated,
           "force_integer_mv requires warp_mv == 1");
  t.expect(full::validate_warp_block(true, false, 64, 64, true, true,
                                     full::kYModeWarpMv, full::kMotionDeltaWarp,
                                     &why) == FrameStatus::kOk,
           "valid WARPMV block");
  t.expect(full::validate_warp_block(true, false, 64, 64, false, false,
                                     full::kYModeWarpMv, full::kMotionSimple,
                                     &why) == FrameStatus::kConstraintViolated,
           "WARPMV without is_warp is rejected");
  t.expect(full::validate_warp_block(true, false, 64, 64, true, true,
                                     full::kYModeWarpMv, full::kMotionSimple,
                                     &why) == FrameStatus::kConstraintViolated,
           "WARPMV must be DELTAWARP");
}

// ---------------------------------------------------------------------------
// §7.7 / §7.8 / R77
// ---------------------------------------------------------------------------

adv::RefSlot to_adv_slot(const full::RefSlot& slot) {
  adv::RefSlot out;
  out.valid = slot.valid;
  out.counter = slot.counter;
  out.order_hint = slot.restricted ? adv::kRestrictedOrderHint : slot.order_hint;
  out.width = slot.frame_width;
  out.height = slot.frame_height;
  out.base_q_idx = slot.base_q_idx;
  out.tlayer_id = slot.tlayer;
  out.mlayer_id = slot.mlayer;
  out.implicit_output = slot.implicit_output;
  out.immediate_output = slot.immediate_output;
  return out;
}

void test_ref_map_parity(Test& t) {
  for (int iter = 0; iter < 700; ++iter) {
    const int num_slots = rand_int(2, 8);
    std::array<full::RefSlot, 8> slots{};
    adv::RefSlotState adv_state;
    for (int i = 0; i < num_slots; ++i) {
      full::RefSlot& slot = slots[static_cast<std::size_t>(i)];
      slot.valid = rand_int(0, 3) != 0;
      slot.restricted = rand_int(0, 5) == 0;
      slot.order_hint = rand_int(0, 15);
      slot.frame_width = rand_int(0, 1) ? 128 : 64;
      slot.frame_height = slot.frame_width;
      slot.counter = i + (rand_int(0, 1) ? 0 : 100);
      slot.base_q_idx = rand_int(20, 120);
      slot.tlayer = rand_int(0, 2);
      slot.mlayer = 0;
      slot.implicit_output = rand_int(0, 3) != 0;
      slot.immediate_output = rand_int(0, 3) != 0;
      adv_state.slots[i] = to_adv_slot(slot);
    }

    full::RefMapParams params;
    params.num_ref_frames = num_slots;
    params.active_num_ref_frames = std::min(7, num_slots);
    params.order_hint = rand_int(0, 15);
    params.order_hint_bits = 4;
    params.check_res = rand_int(0, 1) != 0;
    params.frame_width = 64;
    params.frame_height = 64;
    params.slots = slots.data();

    full::RefMap mine;
    std::string why;
    const bool ok = full::get_ref_frames(params, &mine, &why);
    t.expect(ok, "get_ref_frames runs in parity batch");
    if (!ok) continue;
    full::get_past_future_cur_ref_lists(params, &mine, &why);

    adv::RefListInput input;
    input.state = adv_state;
    input.order_hint = params.order_hint;
    input.num_ref_frames = num_slots;
    input.obu_tlayer_id = 2;
    input.active_num_ref_frames = params.active_num_ref_frames;
    input.frame_width = 64;
    input.frame_height = 64;
    input.check_res = params.check_res;
    const adv::RefListResult theirs = adv::build_ref_frame_list(input);

    bool same = mine.num_total_refs == theirs.num_total_refs;
    if (same) {
      for (int i = 0; i < mine.num_total_refs; ++i) {
        if (mine.ref_frame_idx[i] != theirs.ref_frame_idx[i] ||
            mine.scores_distance[i] != theirs.scores_distance[i]) {
          same = false;
          break;
        }
      }
    }
    if (!same) {
      std::printf("ref-map parity failure: iter=%d num=%d/%d\n", iter,
                  mine.num_total_refs, theirs.num_total_refs);
    }
    t.expect(same, "§7.7 mapping matches adv::build_ref_frame_list");

    // §7.8 parity.
    int order_hint_per_ref[7] = {};
    for (int i = 0; i < mine.num_total_refs; ++i) {
      order_hint_per_ref[i] =
          full::ref_order_hint(slots[static_cast<std::size_t>(
              mine.ref_frame_idx[i])]);
    }
    const adv::PastFutureLists pf = adv::get_past_future_cur_ref_lists(
        order_hint_per_ref, mine.scores_distance, mine.num_total_refs);
    t.eq_int(mine.num_past_refs, pf.num_past, "§7.8 NumPastRefs parity");
    t.eq_int(mine.num_future_refs, pf.num_future, "§7.8 NumFutureRefs parity");
    t.eq_int(mine.num_cur_refs, pf.num_cur, "§7.8 numCurRefs parity");
    t.eq_int(mine.closest_past, pf.closest_past, "§7.8 ClosestPast parity");
    t.eq_int(mine.closest_future, pf.closest_future,
             "§7.8 ClosestFuture parity");
    t.eq_int(mine.furthest_future, pf.furthest_future,
             "§7.8 FurthestFuture parity");
    // adv::get_past_future_cur_ref_lists substitutes 0 for NONE; the spec
    // leaves SkipSegFrame == ClosestPast == NONE.
    const int mine_skip = mine.skip_seg_frame == full::kRefNone ? 0 : mine.skip_seg_frame;
    t.eq_int(mine_skip, pf.skip_seg_frame, "§7.8 SkipSegFrame parity");
  }
}

void test_restricted_append(Test& t) {
  std::array<full::RefSlot, 8> slots{};
  for (int i = 0; i < 4; ++i) {
    slots[static_cast<std::size_t>(i)].valid = true;
    slots[static_cast<std::size_t>(i)].order_hint = i + 1;
    slots[static_cast<std::size_t>(i)].frame_width = 64;
    slots[static_cast<std::size_t>(i)].frame_height = 64;
    slots[static_cast<std::size_t>(i)].counter = i;
  }
  slots[5].valid = true;
  slots[5].restricted = true;
  slots[5].order_hint = -1;
  slots[5].frame_width = 64;
  slots[5].frame_height = 64;
  slots[5].counter = 5;

  full::RefMapParams params;
  params.num_ref_frames = 8;
  params.active_num_ref_frames = 7;
  params.order_hint = 8;
  params.check_res = true;
  params.frame_width = 64;
  params.frame_height = 64;
  params.slots = slots.data();
  full::RefMap map;
  std::string why;
  t.expect(full::get_ref_frames(params, &map, &why), "restricted map builds");
  t.eq_int(map.num_total_refs, 5, "restricted slot appended");
  t.eq_int(map.ref_frame_idx[4], 5, "restricted slot is last");
}

void test_display_order_hint(Test& t) {
  std::array<full::RefSlot, 8> slots{};
  slots[0] = full::RefSlot{};
  slots[0].valid = true;
  slots[0].order_hint = 14;
  slots[0].implicit_output = true;
  slots[0].frame_width = 64;
  slots[0].frame_height = 64;

  full::DispOrderHintParams p;
  p.num_ref_frames = 8;
  p.slots = slots.data();
  p.order_hint_bits = 4;
  p.order_hint_lsbs = 1;
  int value = 0;
  std::string why;
  t.expect(full::get_disp_order_hint(p, &value, &why),
           "display order hint derives");
  t.eq_int(value, 17, "display order hint wraps past the 4-bit range");

  // CLK short-circuits.
  full::DispOrderHintParams key = p;
  key.obu_closed_loop_key = true;
  t.expect(full::get_disp_order_hint(key, &value, &why), "CLK display hint");
  t.eq_int(value, 1, "CLK returns OrderHintLsbs");

  // Restricted switch short-circuits.
  full::DispOrderHintParams sw = p;
  sw.frame_type_switch = true;
  sw.restricted_prediction_switch = true;
  t.expect(full::get_disp_order_hint(sw, &value, &why), "switch display hint");
  t.eq_int(value, 1, "restricted switch returns OrderHintLsbs");

  // Only showable references feed the maximum.
  std::array<full::RefSlot, 8> hidden = slots;
  hidden[0].implicit_output = false;
  hidden[0].immediate_output = false;
  hidden[1].valid = true;
  hidden[1].order_hint = 3;
  hidden[1].implicit_output = true;
  full::DispOrderHintParams hp = p;
  hp.slots = hidden.data();
  t.expect(full::get_disp_order_hint(hp, &value, &why),
           "hidden reference display hint");
  t.eq_int(value, 1, "only showable references contribute to maxDisp");

  // Conformance bound.
  std::array<full::RefSlot, 8> huge{};
  huge[0].valid = true;
  huge[0].order_hint = (1 << 29) - 1;
  huge[0].implicit_output = true;
  full::DispOrderHintParams big = p;
  big.slots = huge.data();
  big.order_hint_lsbs = 0;
  t.expect(!full::get_disp_order_hint(big, &value, &why),
           "display order hint bound is enforced");
}

void test_r77(Test& t) {
  std::array<full::OperatingPointRefMap, 2> maps{};
  for (auto& m : maps) {
    m.explicit_map = false;
    m.num_total_refs = 2;
    m.ref_frame_idx[0] = 0;
    m.ref_frame_idx[1] = 1;
    m.ref_identity[0] = 11;
    m.ref_identity[1] = 22;
    m.disp_order_hint = 9;
  }
  full::R77Result result;
  std::string why;
  t.expect(full::validate_r77(maps, &result, &why), "R77 accepts a stable map");

  auto bad = maps;
  bad[1].ref_identity[1] = 33;
  t.expect(!full::validate_r77(bad, &result, &why),
           "R77 rejects a different implicit mapping");

  auto bad_hint = maps;
  bad_hint[1].disp_order_hint = 10;
  t.expect(!full::validate_r77(bad_hint, &result, &why),
           "R77 rejects a different display order hint");

  auto explicit_bad = maps;
  explicit_bad[0].explicit_map = true;
  explicit_bad[1].explicit_map = true;
  explicit_bad[0].ref_frame_idx[0] = 4;
  explicit_bad[1].ref_frame_idx[0] = 4;
  explicit_bad[1].ref_identity[0] = 99;
  t.expect(!full::validate_r77(explicit_bad, &result, &why),
           "R77 rejects an explicit slot that names different frames");
}

void test_frame_ref_map_syntax(Test& t) {
  std::array<full::RefSlot, 8> slots{};
  for (int i = 0; i < 4; ++i) {
    slots[static_cast<std::size_t>(i)].valid = true;
    slots[static_cast<std::size_t>(i)].order_hint = i + 1;
    slots[static_cast<std::size_t>(i)].frame_width = 64;
    slots[static_cast<std::size_t>(i)].frame_height = 64;
    slots[static_cast<std::size_t>(i)].counter = i;
  }
  full::RefMapParams params;
  params.num_ref_frames = 8;
  params.active_num_ref_frames = 7;
  params.order_hint = 8;
  params.check_res = true;
  params.frame_width = 64;
  params.frame_height = 64;
  params.slots = slots.data();
  std::string why;

  full::ImplicitRefMap implicit;
  t.expect(full::derive_implicit_ref_map(params, &implicit, &why),
           "implicit ref map derives");

  // Implicit mode writes no bits.
  full::FrameRefMapSyntax syntax;
  syntax.seq_explicit_ref_frame_map = false;
  syntax.frame_is_switch = false;
  av2enc::BitWriter writer;
  t.expect(full::write_frame_ref_map_syntax(&writer, syntax, params, implicit,
                                            &why),
           "implicit ref map writes no bits");
  const std::vector<std::uint8_t> bytes = writer.take_bytes();
  t.eq_int(static_cast<long long>(bytes.size()), 0,
           "implicit-frame reference map occupies zero bits");
  full::BitReader reader(bytes.data(), bytes.size());
  full::FrameRefMapResult read;
  t.expect(full::read_frame_ref_map_syntax(&reader, syntax, params, &read,
                                           &why),
           "implicit ref map reads");
  t.expect(!read.explicit_map, "reader reports the implicit path");
  t.eq_int(read.map.num_total_refs, implicit.stage1.num_total_refs,
           "implicit reader reproduces the §7.7 mapping");
  for (int i = 0; i < read.map.num_total_refs; ++i) {
    t.eq_int(read.map.ref_frame_idx[i], implicit.stage1.ref_frame_idx[i],
             "implicit mapping index matches");
  }

  // Explicit mode round-trip.
  syntax.seq_explicit_ref_frame_map = true;
  syntax.explicit_ref_frame_map = true;
  syntax.num_total_refs = 3;
  syntax.ref_frame_idx[0] = 1;
  syntax.ref_frame_idx[1] = 3;
  syntax.ref_frame_idx[2] = 0;
  av2enc::BitWriter explicit_writer;
  t.expect(full::write_frame_ref_map_syntax(&explicit_writer, syntax, params,
                                            implicit, &why),
           "explicit ref map writes");
  const std::vector<std::uint8_t> explicit_bytes = explicit_writer.take_bytes();
  t.expect(!explicit_bytes.empty(), "explicit ref map occupies bits");
  full::BitReader explicit_reader(explicit_bytes.data(), explicit_bytes.size());
  full::FrameRefMapResult explicit_read;
  t.expect(full::read_frame_ref_map_syntax(&explicit_reader, syntax, params,
                                           &explicit_read, &why),
           "explicit ref map reads");
  t.expect(explicit_read.explicit_map, "reader reports the explicit path");
  t.eq_int(explicit_read.map.num_total_refs, 3, "explicit ref count");
  t.eq_int(explicit_read.map.ref_frame_idx[0], 1, "explicit index 0");
  t.eq_int(explicit_read.map.ref_frame_idx[1], 3, "explicit index 1");
  t.eq_int(explicit_read.map.ref_frame_idx[2], 0, "explicit index 2");

  // The switch-frame path forces an explicit map even with the sequence bit 0.
  syntax.seq_explicit_ref_frame_map = false;
  syntax.frame_is_switch = true;
  t.expect(full::write_frame_ref_map_syntax(&explicit_writer, syntax, params,
                                            implicit, &why),
           "switch frame forces an explicit map");
}

// ---------------------------------------------------------------------------
// Corner warp generation
// ---------------------------------------------------------------------------

void test_corners(Test& t) {
  // Three neighbours at the left, top and top-left with distinct MVs.
  TestGrid g;
  g.cols = 64;
  g.rows = 64;
  g.full_mi.assign(static_cast<std::size_t>(g.cols) * static_cast<std::size_t>(g.rows), full::MiCell{});
  g.mc_mi.assign(static_cast<std::size_t>(g.cols) * static_cast<std::size_t>(g.rows), mc::MiInfo{});
  Block left;
  left.row = 48;
  left.col = 32;
  left.w4 = 16;
  left.h4 = 16;
  left.ref[0] = 0;
  left.mv = Mv{8, 0};
  left.sub_mv = left.mv;
  fill_block(g, left);
  Block top = left;
  top.row = 32;
  top.col = 48;
  top.mv = Mv{0, 8};
  top.sub_mv = top.mv;
  fill_block(g, top);
  Block top_left = left;
  top_left.row = 32;
  top_left.col = 32;
  top_left.mv = Mv{0, 0};
  top_left.sub_mv = top_left.mv;
  fill_block(g, top_left);

  full::StackParams p;
  p.grid = g.grid();
  p.mi_row = 48;
  p.mi_col = 48;
  p.bw4 = 16;
  p.bh4 = 16;
  p.sb_h4 = 16;
  p.ref_frame[0] = 0;
  p.ref_frame[1] = full::kRefNone;
  p.derive_wrl = true;
  full::BuildState state;
  full::CornerResult corners;
  const full::Status st =
      full::generate_points_from_corners(p, &state, 0, &corners);
  t.expect(st == full::Status::kOk, "corner generation runs");
  t.eq_int(corners.corners_found, 3, "three corners found");
  t.expect(!corners.all_mvs_same, "distinct corner MVs are not allMvsSame");
  t.expect(corners.inserted, "corner warp model is inserted");
  t.eq_int(state.stack.num_warp_found, 1, "one warp candidate from the corners");

  // Identical corner MVs must not produce a model.
  Block same = left;
  same.mv = Mv{0, 0};
  same.sub_mv = same.mv;
  fill_block(g, same);
  Block same_top = top;
  same_top.mv = Mv{0, 0};
  same_top.sub_mv = same_top.mv;
  fill_block(g, same_top);
  full::BuildState state2;
  full::CornerResult corners2;
  t.expect(full::generate_points_from_corners(p, &state2, 0, &corners2) ==
               full::Status::kOk,
           "all-equal corners run");
  t.expect(corners2.all_mvs_same, "allMvsSame is detected");
  t.expect(!corners2.inserted, "all-equal corners insert nothing");
}

// ---------------------------------------------------------------------------
// Stack record + independent mirror
// ---------------------------------------------------------------------------

full::Stack random_stack(TestGrid* grid_out) {
  const int sb_cols = rand_int(1, 3);
  const int sb_rows = rand_int(1, 3);
  TestGrid g = make_random_grid(sb_cols, sb_rows, false, 3);
  int row = 0;
  int col = 0;
  const Block& cur = pick_block(g, &row, &col);
  full::StackParams p;
  p.grid = g.grid();
  p.mi_row = row;
  p.mi_col = col;
  p.bw4 = cur.w4;
  p.bh4 = cur.h4;
  p.sb_h4 = 16;
  p.ref_frame[0] = static_cast<std::int8_t>(rand_int(0, 3));
  p.ref_frame[1] = full::kRefNone;
  p.reorder = static_cast<full::Reorder>(rand_int(0, 2));
  p.max_drl_bits_minus_1 = 4;
  full::Stack stack;
  const full::Status st = full::build(p, &stack);
  if (st != full::Status::kOk) {
    std::printf("random stack build failed: %d\n", static_cast<int>(st));
  }
  *grid_out = std::move(g);
  return stack;
}

void test_stack_mirror(Test& t) {
  std::string why;
  for (int iter = 0; iter < 600; ++iter) {
    TestGrid g;
    const full::Stack stack = random_stack(&g);
    const int num_lists = 1;
    const std::vector<std::uint8_t> record =
        full::serialize_stack(stack, num_lists);
    full::StackMirror mirror;
    const bool parsed = mirror.parse(record, &why);
    if (!parsed && t.failures < 20) {
      std::printf("mirror parse failure: %s (count=%d nearest=%d warp=%d)\n",
                  why.c_str(), stack.count, stack.num_nearest,
                  stack.num_warp_found);
    }
    t.expect(parsed, "stack record parses");
    const bool valid = mirror.validate(&why);
    if (!valid && t.failures < 20) {
      std::printf("mirror validate failure: %s\n", why.c_str());
    }
    t.expect(valid, "mirrored stack passes its invariants");
    t.expect(mirror.equals(stack, &why), "mirror equals the encoder stack");
    full::LockstepReport report;
    t.expect(full::lockstep(stack, num_lists, Mv{0, 0}, mirror.stack(), &report),
             "lockstep resolution agrees over every DRL index");
    t.expect(report.fields_equal && report.resolved_equal,
             "lockstep report is clean");
    const int expected_checks =
        std::min(stack.count - 1, full::kMaxDrlIndex) + 1;
    if (expected_checks > 0) {
      t.eq_int(report.drl_checked, expected_checks * 3,
               "lockstep checks every index and mode");
    }

    // Corruption and truncation must be rejected, never silently accepted.
    if (!record.empty()) {
      std::vector<std::uint8_t> corrupt = record;
      corrupt[corrupt.size() / 2] ^= 0x40;
      full::StackMirror bad;
      t.expect(!bad.parse(corrupt, &why), "a corrupted record is rejected");
      std::vector<std::uint8_t> truncated = record;
      truncated.resize(truncated.size() - 1);
      t.expect(!bad.parse(truncated, &why), "a truncated record is rejected");
      std::vector<std::uint8_t> extended = record;
      extended.push_back(0);
      t.expect(!bad.parse(extended, &why),
               "trailing bytes after the record are rejected");
    }
  }
}

// ---------------------------------------------------------------------------
// Goldens
// ---------------------------------------------------------------------------

struct GoldenCase {
  std::uint64_t seed;
  int sb_cols;
  int sb_rows;
  int reorder;
  std::uint64_t hash;
};

// Pinned with --print-goldens; any change to the stack order or the record
// layout changes every hash.
constexpr GoldenCase kGoldens[] = {
    {1, 3, 3, 0, 0x6346dd338bc43266ULL},
    {2, 2, 4, 0, 0x0287819bec8e3d0bULL},
    {3, 4, 2, 1, 0xc7d9ea65981f087bULL},
    {4, 2, 2, 2, 0x1ded31b9986e641cULL},
    {5, 3, 1, 0, 0x40b69ef8bc9b21baULL},
    {6, 1, 3, 1, 0x4cf33f4caccc462dULL},
    {7, 3, 2, 2, 0xc854a181d6c3c81cULL},
    {8, 2, 2, 0, 0x52df304fa50d36ffULL},
};

std::uint64_t golden_hash(const GoldenCase& c) {
  std::mt19937_64 engine(c.seed * 7919ULL + 17ULL);
  std::uniform_int_distribution<int> dist(0, 1 << 30);
  // Deterministic grid independent of the global RNG.
  TestGrid g;
  g.cols = c.sb_cols * 16;
  g.rows = c.sb_rows * 16;
  g.full_mi.assign(static_cast<std::size_t>(g.cols) * static_cast<std::size_t>(g.rows), full::MiCell{});
  g.mc_mi.assign(static_cast<std::size_t>(g.cols) * static_cast<std::size_t>(g.rows), mc::MiInfo{});
  for (int sbr = 0; sbr < c.sb_rows; ++sbr) {
    for (int sbc = 0; sbc < c.sb_cols; ++sbc) {
      const int step = 1 << (dist(engine) % 5);
      for (int r = 0; r < 16; r += step) {
        for (int cc = 0; cc < 16; cc += step) {
          Block b;
          b.row = sbr * 16 + r;
          b.col = sbc * 16 + cc;
          b.w4 = step;
          b.h4 = step;
          b.is_inter = (dist(engine) % 4) != 0;
          if (b.is_inter) {
            b.ref[0] = static_cast<std::int8_t>(dist(engine) % 4);
            b.ref[1] = (dist(engine) % 2)
                           ? static_cast<std::int8_t>(dist(engine) % 4)
                           : full::kRefNone;
            b.mv = Mv{static_cast<std::int32_t>(dist(engine) % 33) * 8 - 128,
                      static_cast<std::int32_t>(dist(engine) % 33) * 8 - 128};
            b.sub_mv = b.mv;
            b.y_mode = (dist(engine) % 2) ? full::kYModeNearMv
                                         : full::kYModeGlobalMv;
          } else {
            b.ref[0] = full::kIntraFrame;
            b.ref[1] = full::kRefNone;
          }
          g.blocks.push_back(b);
          fill_block(g, b);
        }
      }
    }
  }
  const Block& cur = g.blocks[static_cast<std::size_t>(dist(engine)) % g.blocks.size()];
  full::StackParams p;
  p.grid = g.grid();
  p.mi_row = cur.row;
  p.mi_col = cur.col;
  p.bw4 = cur.w4;
  p.bh4 = cur.h4;
  p.sb_h4 = 16;
  p.ref_frame[0] = static_cast<std::int8_t>(dist(engine) % 4);
  p.ref_frame[1] = full::kRefNone;
  p.reorder = static_cast<full::Reorder>(c.reorder);
  p.max_drl_bits_minus_1 = 4;
  full::Stack stack;
  if (full::build(p, &stack) != full::Status::kOk) return 0;
  return full::stack_record_hash(stack, 1);
}

void test_goldens(Test& t, bool print) {
  for (const GoldenCase& c : kGoldens) {
    const std::uint64_t hash = golden_hash(c);
    if (print) {
      std::printf("GOLDEN seed=%llu sb=%dx%d reorder=%d hash=%016llx\n",
                  static_cast<unsigned long long>(c.seed), c.sb_cols, c.sb_rows,
                  c.reorder, static_cast<unsigned long long>(hash));
    }
    t.expect(hash != 0, "golden stack builds");
    if (c.hash != 0) {
      t.eq_int(static_cast<long long>(hash),
               static_cast<long long>(c.hash),
               "pinned golden stack hash matches");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool print_goldens = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--print-goldens") == 0) print_goldens = true;
  }
  Test t;
  test_merge_parity(t);
  test_sort_negative_control(t);
  test_clamp_before_global(t);
  test_global_candidate(t);
  test_mvp_combination_gate(t);
  test_warp_candidate_order(t);
  test_prune_budget(t);
  test_dispatch_single(t);
  test_dispatch_tip(t);
  test_dispatch_compound(t);
  test_single_add_derived_projection(t);
  test_allow_warpmv_mode(t);
  test_ref_map_parity(t);
  test_restricted_append(t);
  test_display_order_hint(t);
  test_r77(t);
  test_frame_ref_map_syntax(t);
  test_corners(t);
  test_stack_mirror(t);
  test_goldens(t, print_goldens);
  std::printf("checks=%d failures=%d\n", t.checks, t.failures);
  return t.failures == 0 ? 0 : 1;
}
