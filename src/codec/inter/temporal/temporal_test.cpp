// CPU reference tests for the AV2 temporal inter tools.
//
// Coverage:
//   * §7.9.4/.5 MV projection, compression and clamping
//   * §7.9.6–.8 sampled positions, block-position checks and TMVP units
//   * §7.9/.7.10 motion-field estimation, TIP scaling, hole fill, averaging
//   * §7.22 motion-field storage ordering and wedge allow-list
//   * §7.12.2.7/.8 temporal scan and get_tip_cand
//   * §7.12.2.21 reference-MV bank ring, reset, hit budget and MVP fill
//   * §7.10.6 TIP block structures and Tip_Weighting_Factor blends
//   * §7.13.3.8–.12 OPFL distances, gradient statistics and solved MVs
//   * §7.13.3.3–.6 refine-MV gating, reference area, search and prediction
//   * §5.4.6/§5.18.2/§5.18.3.2/§5.20.7.17 syntax round trips
//
// Usage: ./temporal_test
// Self-contained; no real media or decoder is needed.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "codec/inter/temporal/motion_field.h"
#include "codec/inter/temporal/opfl.h"
#include "codec/inter/temporal/ref_mv_bank.h"
#include "codec/inter/temporal/refinemv.h"
#include "codec/inter/temporal/temporal_math.h"
#include "codec/inter/temporal/temporal_predict.h"
#include "codec/inter/temporal/temporal_syntax.h"
#include "codec/inter/temporal/tip.h"

namespace {

using namespace av2enc::inter::temporal;

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

#define CHECK_EQ(a, b, msg) CHECK((a) == (b), msg)

// ---------------------------------------------------------------------------
// Projection / range rules.
// ---------------------------------------------------------------------------

void test_projection_math() {
  const int kDivMultRef[32] = {
      0,    16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638,
      1489, 1365,  1260, 1170, 1092, 1024, 963,  910,  862,  819,  780,
      744,  712,   682,  655,  630,  606,  585,  564,  546,  528};
  for (int i = 0; i < 32; ++i) {
    CHECK_EQ(kDivMult[i], kDivMultRef[i], "Div_Mult table");
  }

  // Projection is Round2Signed(mv * num * Div_Mult[den], 14).
  CHECK_EQ(get_mv_projection(Mv{8, -8}, 1, 1).row, 8, "projection identity");
  CHECK_EQ(get_mv_projection(Mv{8, -8}, 1, 1).col, -8, "projection identity");
  CHECK_EQ(get_mv_projection(Mv{8, -8}, 2, 1).row, 16, "projection 2x");
  CHECK_EQ(get_mv_projection(Mv{8, -8}, 1, 2).row, 4, "projection half");
  CHECK_EQ(get_mv_projection(Mv{8, -8}, 1, 2).col, -4, "projection half");
  CHECK_EQ(get_mv_projection(Mv{9, -9}, 1, 2).row, 5, "projection rounding");
  CHECK_EQ(get_mv_projection(Mv{9, -9}, 1, 2).col, -5, "projection rounding");
  CHECK_EQ(get_mv_projection(Mv{8, 0}, 0, 5).row, 0, "projection zero num");
  CHECK_EQ(get_mv_projection(Mv{8, 0}, 40, 5).row, 50, "projection num clip");
  CHECK_EQ(get_mv_projection(Mv{0, 0}, 1, 0).row, 0, "projection zero den");

  // Large values are clamped to the 17-bit MV range, not wrapped.
  const Mv large = get_mv_projection(Mv{65535, -65535}, 31, 1);
  CHECK(large.row <= kMvUpp - 1 && large.row >= kMvLow + 1, "projection clamp");

  const Mv clamped = get_mv_projection_clamp(Mv{4000, -4000}, 1, 1);
  CHECK_EQ(clamped.row, kRefMvsLimit, "projection clamp limit");
  CHECK_EQ(clamped.col, -kRefMvsLimit, "projection clamp limit");

  // compression_mv / uncompression_mv round trip over the REFMVS range.
  for (int v = -2047; v <= 2047; ++v) {
    const std::int16_t c = compression_mv(v);
    const int back = uncompression_mv(c);
    // The compression is lossy for |v| > 255; only require no overflow and the
    // documented bound.  Exact values are checked separately.
    CHECK(c >= -4095 && c <= 4095, "compression range");
    CHECK(back >= -2048 && back <= 2048, "uncompression range");
  }
  CHECK_EQ(uncompression_mv(compression_mv(0)), 0, "compress 0");
  CHECK_EQ(uncompression_mv(compression_mv(1)), 1, "compress 1");
  CHECK_EQ(uncompression_mv(compression_mv(-1)), -1, "compress -1");
  CHECK_EQ(uncompression_mv(compression_mv(15)), 15, "compress 15");
  CHECK_EQ(uncompression_mv(compression_mv(-15)), -15, "compress -15");
  CHECK_EQ(uncompression_mv(compression_mv(16)), 16, "compress 16");
  CHECK_EQ(uncompression_mv(compression_mv(255)), 248, "compress 255 lossy");
  CHECK_EQ(uncompression_mv(compression_mv(-255)), -248, "compress -255 lossy");
}

void test_sampled_position() {
  // 32x32 field (mi 64x64), 64x64 SB -> tmvp shift 3.
  bool valid = true;
  const int y = project_no_constraint(4, 64, 1, 8, &valid);
  CHECK_EQ(y, 5, "project_no_constraint positive");
  CHECK(valid, "project_no_constraint valid");
  valid = true;
  const int y2 = project_no_constraint(4, -64, 1, 8, &valid);
  CHECK_EQ(y2, 3, "project_no_constraint negative");
  CHECK(valid, "project_no_constraint negative valid");
  valid = true;
  (void)project_no_constraint(0, -64, 1, 8, &valid);
  CHECK(!valid, "project_no_constraint out of range");

  const SampledPosition p =
      get_sampled_position(4, 4, 1, Mv{0, 64}, 64, 64, 1);
  CHECK(p.valid, "sampled position valid");
  CHECK_EQ(p.x8, 5, "sampled position x");
  CHECK_EQ(p.y8, 4, "sampled position y");
  const SampledPosition even =
      get_sampled_position(4, 4, 1, Mv{0, 64}, 64, 64, 2);
  CHECK_EQ(even.x8, 4, "sampled position ProjStep=2 even");

  // check_block_position: ProjStep=1 allows the left half-unit, not more.
  CHECK(check_block_position(true, 8, 8, 8, 8, true, 1), "block position same");
  CHECK(check_block_position(true, 8, 8, 8, 8, true, 1), "block position same");
  CHECK(check_block_position(true, 8, 8, 4, 8, true, 1),
        "block position left half-unit");
  CHECK(!check_block_position(true, 8, 8, 0, 8, true, 1),
        "block position too far left");
  CHECK(check_block_position(true, 8, 8, 8, 15, true, 1),
        "block position inside unit");
  CHECK(!check_block_position(true, 8, 8, 8, 16, true, 1),
        "block position next unit");
  CHECK(!check_block_position(false, 8, 8, 8, 8, true, 1),
        "block position invalid input");
}

// ---------------------------------------------------------------------------
// Motion field / TIP setup.
// ---------------------------------------------------------------------------

struct FieldFixture {
  int mi_cols = 16;
  int mi_rows = 16;
  int w8 = 8;
  int h8 = 8;
  std::vector<std::int32_t> mv;
  std::vector<std::uint8_t> valid;
  std::vector<std::int32_t> offset;
  MotionField field{};

  FieldFixture() {
    mv.assign(static_cast<std::size_t>(w8 * h8 * 2), 0);
    valid.assign(static_cast<std::size_t>(w8 * h8), 0);
    offset.assign(static_cast<std::size_t>(w8 * h8), 0);
    field.w8 = w8;
    field.h8 = h8;
    field.mv = mv.data();
    field.valid = valid.data();
    field.offset = offset.data();
  }
};

MotionFieldConfig make_two_ref_config(FieldFixture* f,
                                      std::vector<std::int16_t>* saved_mv,
                                      std::vector<std::int8_t>* saved_ref) {
  MotionFieldConfig cfg;
  cfg.mi_cols = f->mi_cols;
  cfg.mi_rows = f->mi_rows;
  cfg.proj_step = 1;
  cfg.sb_64x64 = true;
  cfg.enable_tip = true;
  cfg.order_hint = 5;
  cfg.num_total_refs = 2;
  cfg.num_past_refs = 1;
  cfg.num_future_refs = 1;
  cfg.frame_width = 64;
  cfg.frame_height = 64;
  cfg.ref_order_hint[0] = 4;
  cfg.ref_order_hint[1] = 6;
  cfg.refs[0].slot = 0;
  cfg.refs[0].type = FrameType::kInter;
  cfg.refs[0].width = 64;
  cfg.refs[0].height = 64;
  cfg.refs[0].order_hint = 4;
  cfg.refs[0].num_saved_refs = 1;
  cfg.refs[0].saved_order_hint[0] = 0;
  cfg.refs[1].slot = 1;
  cfg.refs[1].type = FrameType::kInter;
  cfg.refs[1].width = 64;
  cfg.refs[1].height = 64;
  cfg.refs[1].order_hint = 6;
  cfg.refs[1].num_saved_refs = 1;
  cfg.refs[1].saved_order_hint[0] = 4;

  saved_mv->assign(static_cast<std::size_t>(f->w8 * f->h8 * 2 * 2), 0);
  saved_ref->assign(static_cast<std::size_t>(f->w8 * f->h8 * 2), kRefNone);
  for (int i = 0; i < f->w8 * f->h8; ++i) {
    (*saved_mv)[static_cast<std::size_t>(i * 4 + 0)] = compression_mv(8);
    (*saved_mv)[static_cast<std::size_t>(i * 4 + 1)] = compression_mv(0);
    (*saved_mv)[static_cast<std::size_t>(i * 4 + 2)] = compression_mv(8);
    (*saved_mv)[static_cast<std::size_t>(i * 4 + 3)] = compression_mv(0);
    (*saved_ref)[static_cast<std::size_t>(i * 2 + 0)] = 0;
    (*saved_ref)[static_cast<std::size_t>(i * 2 + 1)] = 0;
  }
  cfg.saved[0].w8 = f->w8;
  cfg.saved[0].h8 = f->h8;
  cfg.saved[0].mv = saved_mv->data();
  cfg.saved[0].ref = saved_ref->data();
  cfg.saved[1].w8 = f->w8;
  cfg.saved[1].h8 = f->h8;
  cfg.saved[1].mv = nullptr;
  cfg.saved[1].ref = nullptr;
  return cfg;
}

void test_motion_field_estimation() {
  FieldFixture f;
  std::vector<std::int16_t> saved_mv;
  std::vector<std::int8_t> saved_ref;
  MotionFieldConfig cfg = make_two_ref_config(&f, &saved_mv, &saved_ref);
  MotionFieldResult result;
  const TemporalStatus s = motion_field_estimation(cfg, &f.field, nullptr,
                                                   &result);
  CHECK_EQ(static_cast<int>(s), 0, "motion field estimation ok");
  CHECK_EQ(result.closest_past, 0, "closest past");
  CHECK_EQ(result.closest_future, 1, "closest future");
  CHECK(result.projection_count > 0, "projection count");

  for (int i = 0; i < f.w8 * f.h8; ++i) {
    CHECK_EQ(f.valid[static_cast<std::size_t>(i)], 1, "field valid");
    CHECK_EQ(f.mv[static_cast<std::size_t>(i * 2 + 0)], 8, "field mv row");
    CHECK_EQ(f.mv[static_cast<std::size_t>(i * 2 + 1)], 0, "field mv col");
    CHECK_EQ(f.offset[static_cast<std::size_t>(i)], 4, "field offset");
  }

  // TIP temporal scaling: refOffset = future-past = 2, startOffset = 4.
  TipConfig tip;
  tip.mode = TipFrameMode::kAsReference;
  tip.num_past_refs = 1;
  tip.num_future_refs = 1;
  tip.order_hint = 5;
  tip.closest_past_order_hint = 4;
  tip.closest_future_order_hint = 6;
  const TemporalStatus ts = setup_tip_motion_field(cfg, tip, &f.field, nullptr,
                                                   nullptr, nullptr);
  CHECK_EQ(static_cast<int>(ts), 0, "tip setup ok");
  const TipOffsets offsets = get_tip_offsets(tip);
  CHECK_EQ(offsets.ref_offset, 2, "tip ref offset");
  CHECK_EQ(offsets.past_offset, 1, "tip past offset");
  CHECK_EQ(offsets.future_offset, -1, "tip future offset");
  for (int i = 0; i < f.w8 * f.h8; ++i) {
    CHECK_EQ(f.mv[static_cast<std::size_t>(i * 2 + 0)], 4, "scaled mv row");
    CHECK_EQ(f.offset[static_cast<std::size_t>(i)], 2, "scaled offset");
  }

  // get_tip_cand: base MV adds to both projected lists.
  TipCandInput cand;
  cand.cand_row = 0;
  cand.cand_col = 0;
  cand.base_row = 0;
  cand.base_col = 0;
  cand.is_16x16 = false;
  cand.base_mv = Mv{16, 0};
  cand.field = &f.field;
  const MvPair pair = get_tip_cand(cand, tip);
  CHECK_EQ(pair.mv[0].row, 18, "tip cand list0 row");
  CHECK_EQ(pair.mv[1].row, 14, "tip cand list1 row");

  // Build TIP: default 16x16 blocks (4x4 MI), then 8x8 with refine-MV.
  std::vector<TipBlock> blocks(256);
  std::vector<Mv> grid(static_cast<std::size_t>(f.mi_cols * f.mi_rows));
  std::vector<Mv> grid1(static_cast<std::size_t>(f.mi_cols * f.mi_rows));
  TipPlan plan;
  plan.num_blocks = static_cast<int>(blocks.size());
  plan.blocks = blocks.data();
  plan.mv_grid = grid.data();
  plan.mv_grid1 = grid1.data();
  const TemporalStatus bp = build_tip_plan(cfg, tip, f.field, &plan);
  CHECK_EQ(static_cast<int>(bp), 0, "build tip plan ok");
  CHECK_EQ(plan.num_blocks, 16, "tip 16x16 block count");
  CHECK_EQ(plan.blocks[0].mv[0].row, 2, "tip block list0 row");
  CHECK_EQ(plan.blocks[0].mv[1].row, -2, "tip block list1 row");
  CHECK_EQ(grid[0].row, 2, "tip grid row");
  CHECK_EQ(grid1[0].row, -2, "tip grid1 row");

  tip.enable_tip_refinemv = true;
  plan.num_blocks = static_cast<int>(blocks.size());
  const TemporalStatus bp2 = build_tip_plan(cfg, tip, f.field, &plan);
  CHECK_EQ(static_cast<int>(bp2), 0, "build tip plan 8x8 ok");
  CHECK_EQ(plan.num_blocks, 64, "tip 8x8 block count");
  CHECK_EQ(plan.blocks[0].bw4, 2, "tip 8x8 bw4");
}

void test_tip_hole_fill_and_average() {
  // 4x4 motion field (mi 8x8); one valid point propagates through holes.
  FieldFixture f;
  f.mi_cols = 8;
  f.mi_rows = 8;
  f.w8 = 4;
  f.h8 = 4;
  f.mv.assign(32, 0);
  f.valid.assign(16, 0);
  f.offset.assign(16, 0);
  f.field.w8 = 4;
  f.field.h8 = 4;
  f.field.mv = f.mv.data();
  f.field.valid = f.valid.data();
  f.field.offset = f.offset.data();
  f.valid[0] = 1;
  f.mv[0] = 10;
  f.offset[0] = 4;

  MotionFieldConfig cfg;
  cfg.mi_cols = 8;
  cfg.mi_rows = 8;
  cfg.proj_step = 1;
  cfg.sb_64x64 = true;
  cfg.enable_tip = true;

  TipConfig tip;
  tip.mode = TipFrameMode::kAsReference;
  tip.allow_tip_hole_fill = true;
  tip.num_past_refs = 1;
  tip.num_future_refs = 1;
  tip.order_hint = 5;
  tip.closest_past_order_hint = 4;
  tip.closest_future_order_hint = 6;

  std::vector<std::int32_t> avg_mv(32, 0);
  std::vector<std::uint8_t> avg_valid(16, 0);
  MotionFieldScratch scratch;
  scratch.avg_mv = avg_mv.data();
  scratch.avg_valid = avg_valid.data();
  const TemporalStatus s = setup_tip_motion_field(cfg, tip, &f.field, nullptr,
                                                   &scratch, nullptr);
  CHECK_EQ(static_cast<int>(s), 0, "hole fill setup ok");
  for (int i = 0; i < 4; ++i) {
    CHECK_EQ(f.valid[static_cast<std::size_t>(i)], 1, "hole filled");
    CHECK_EQ(f.mv[static_cast<std::size_t>(i * 2 + 0)], 5, "avg mv");
    CHECK_EQ(f.offset[static_cast<std::size_t>(i)], 2, "avg offset");
  }
}

// ---------------------------------------------------------------------------
// Storage / temporal scan.
// ---------------------------------------------------------------------------

void test_motion_field_storage() {
  RefMvField out;
  std::vector<std::int8_t> refs(2 * 2 * 2, kRefNone);
  std::vector<std::int16_t> mvs(2 * 2 * 2 * 2, 0);
  out.w8 = 2;
  out.h8 = 2;
  out.ref = refs.data();
  out.mv = mvs.data();

  MotionFieldStoreEntry e;
  e.refs[0] = 0;
  e.refs[1] = kRefNone;
  e.mv[0] = Mv{16, -8};
  e.order_hint = 5;
  e.ref_order_hint[0] = 4;
  CHECK_EQ(static_cast<int>(store_motion_field_entry(e, &out, 0, 0)), 0,
           "store ok");
  CHECK_EQ(out.ref[0], 0, "stored ref0");
  CHECK_EQ(out.ref[1], 0, "stored ref1 mirrors list0");
  CHECK_EQ(uncompression_mv(out.mv[0]), 16, "stored mv row");
  CHECK_EQ(uncompression_mv(out.mv[1]), -8, "stored mv col");
  CHECK_EQ(uncompression_mv(out.mv[2]), 16, "mirrored mv row");
  CHECK_EQ(uncompression_mv(out.mv[3]), -8, "mirrored mv col");

  // Both lists present, future first: the past frame is moved to list 0.
  MotionFieldStoreEntry e2;
  e2.refs[0] = 2;
  e2.refs[1] = 1;
  e2.mv[0] = Mv{8, 0};
  e2.mv[1] = Mv{-8, 0};
  e2.order_hint = 5;
  e2.ref_order_hint[0] = 6;
  e2.ref_order_hint[1] = 4;
  CHECK_EQ(static_cast<int>(store_motion_field_entry(e2, &out, 0, 1)), 0,
           "store order ok");
  const std::size_t pos = 1;
  CHECK_EQ(out.ref[pos * 2 + 0], 1, "past moved to list0");
  CHECK_EQ(out.ref[pos * 2 + 1], 2, "future moved to list1");
  CHECK_EQ(uncompression_mv(out.mv[(pos * 2 + 0) * 2 + 0]), -8,
           "past mv moved");
  CHECK_EQ(uncompression_mv(out.mv[(pos * 2 + 1) * 2 + 0]), 8,
           "future mv moved");

  // Uniform wedge mask: samples >60 everywhere, so list 1 is dropped.
  std::vector<std::uint8_t> mask(64 * 64, 64);
  MotionFieldStoreEntry e3;
  e3.refs[0] = 0;
  e3.refs[1] = 1;
  e3.mv[0] = Mv{4, 4};
  e3.mv[1] = Mv{8, 8};
  e3.is_wedge = true;
  e3.mask = mask.data();
  e3.mask_stride = 64;
  e3.order_hint = 5;
  e3.ref_order_hint[0] = 4;
  e3.ref_order_hint[1] = 6;
  CHECK_EQ(static_cast<int>(store_motion_field_entry(e3, &out, 1, 0)), 0,
           "store wedge ok");
  const std::size_t wpos = 2;
  CHECK_EQ(out.ref[wpos * 2 + 1], 0, "wedge list1 replaced by list0");
  CHECK_EQ(uncompression_mv(out.mv[(wpos * 2 + 1) * 2 + 0]), 4,
           "wedge list1 mv from list0");
  CHECK_EQ(out.ref[wpos * 2 + 0], 0, "wedge keeps list0");
}

void test_temporal_scan() {
  FieldFixture f;
  for (int i = 0; i < f.w8 * f.h8; ++i) {
    f.valid[static_cast<std::size_t>(i)] = 1;
    f.mv[static_cast<std::size_t>(i * 2 + 0)] = 4;
    f.mv[static_cast<std::size_t>(i * 2 + 1)] = 0;
    f.offset[static_cast<std::size_t>(i)] = 2;
  }
  TemporalScanInput in;
  in.mi_row = 0;
  in.mi_col = 0;
  in.mi_cols = f.mi_cols;
  in.mi_rows = f.mi_rows;
  in.bw4 = 4;
  in.bh4 = 4;
  in.ref0 = 0;
  in.ref0_order_hint = 4;
  in.order_hint = 5;
  in.is_compound = false;
  in.field = &f.field;
  StackMv stack[kMaxRefMvStackSize]{};
  int num_found = 0;
  int prune = 0;
  const TemporalStatus s =
      temporal_scan(in, stack, &num_found, &prune);
  CHECK_EQ(static_cast<int>(s), 0, "temporal scan ok");
  CHECK_EQ(num_found, 1, "temporal scan candidates");
  CHECK_EQ(stack[0].mv[0].row, 2, "temporal scan mv");
  CHECK_EQ(stack[0].weight, 2, "temporal scan weight");

  // No candidate at an invalid field position.
  f.valid[9] = 0;
  num_found = 0;
  prune = 0;
  CHECK_EQ(static_cast<int>(temporal_scan(in, stack, &num_found, &prune)), 0,
           "temporal scan invalid ok");
  CHECK_EQ(num_found, 0, "no candidate for invalid field");

  // Tile-relative is_inside(): a 1x1 MI tile excludes the delta=2 candidate.
  f.valid[9] = 1;
  in.tile_mi_row_end = 1;
  in.tile_mi_col_end = 1;
  num_found = 0;
  prune = 0;
  CHECK_EQ(static_cast<int>(temporal_scan(in, stack, &num_found, &prune)), 0,
           "temporal scan tile ok");
  CHECK_EQ(num_found, 0, "tile bounds exclude candidate");
}

// ---------------------------------------------------------------------------
// Reference-MV bank.
// ---------------------------------------------------------------------------

void test_ref_mv_bank() {
  auto put_bank = [](RefMvBank* bank, const RefMvBankBlockContext& ctx,
                     const int refs[2], Mv m0, Mv m1, int cwp, bool within) {
    const Mv mvs[2] = {m0, m1};
    return update_ref_mv_bank(bank, ctx, refs, mvs, cwp, within);
  };
  const int r0[2] = {0, kRefNone};
  const int r5[2] = {5, kRefNone};
  const int r6[2] = {6, kRefNone};
  const int rr00[2] = {0, 0};
  const int rr01[2] = {0, 1};
  const int rr12[2] = {1, 2};
  CHECK_EQ(get_rmb_list_index(r0), 0, "rmb list 0");
  CHECK_EQ(get_rmb_list_index(r5), 5, "rmb list 5");
  CHECK_EQ(get_rmb_list_index(r6), 8, "rmb list ref 6");
  CHECK_EQ(get_rmb_list_index(rr00), 6, "rmb list compound 0,0");
  CHECK_EQ(get_rmb_list_index(rr01), 7, "rmb list compound 0,1");
  CHECK_EQ(get_rmb_list_index(rr12), 8, "rmb list compound other");

  RefMvBank bank;
  RefMvBankBlockContext ctx;
  ctx.mi_row = 0;
  ctx.mi_col = 0;
  ctx.bw4 = 2;
  ctx.bh4 = 2;
  ctx.sb_size4 = 16;
  const int refs[2] = {0, kRefNone};
  CHECK(put_bank(&bank, ctx, refs, Mv{4, 8}, Mv{0, 0}, 8, false),
        "bank insert");
  CHECK_EQ(bank.size[0], 1, "bank size 1");
  CHECK(put_bank(&bank, ctx, refs, Mv{4, 8}, Mv{0, 0}, 8, false),
        "bank duplicate");
  CHECK_EQ(bank.size[0], 1, "bank duplicate stays 1");
  CHECK(put_bank(&bank, ctx, refs, Mv{12, 16}, Mv{0, 0}, 8, false),
        "bank insert 2");
  CHECK_EQ(bank.size[0], 2, "bank size 2");
  CHECK(put_bank(&bank, ctx, refs, Mv{4, 8}, Mv{0, 0}, 12, false),
        "bank refresh duplicate");
  CHECK_EQ(bank.size[0], 2, "bank refresh keeps size");
  CHECK_EQ(bank.bank[0][(bank.start[0] + 1) % kRefMvBankSize].p[0], 8,
           "bank refresh keeps the stored cwp (cwp not in key)");

  // Hit budget: four calls at the superblock start, then blocked.
  RefMvBank limited;
  update_ref_mv_count(&limited, ctx);
  CHECK_EQ(limited.remain_hits, 4, "bank remain hits");
  for (int i = 0; i < 4; ++i) {
    RefMvBankBlockContext block_ctx = ctx;
    block_ctx.mi_row = 1;
    block_ctx.mi_col = 1;
    CHECK(put_bank(&limited, block_ctx, refs, Mv{i, i}, Mv{0, 0}, 8, true),
          "bank budgeted insert");
  }
  RefMvBankBlockContext final_ctx = ctx;
  final_ctx.mi_row = 1;
  final_ctx.mi_col = 1;
  CHECK(!put_bank(&limited, final_ctx, refs, Mv{9, 9}, Mv{0, 0}, 8, true),
        "bank budget exhausted");

  // reset_ref_mv_bank seeds from the row above (fromWithinSb = 0).
  std::vector<RefMvBankRowSource> row(16);
  for (int c = 0; c < 16; c += 2) {
    row[static_cast<std::size_t>(c)].is_inter = 1;
    row[static_cast<std::size_t>(c)].ref[0] = 0;
    row[static_cast<std::size_t>(c)].mv[0][0] = c;
    row[static_cast<std::size_t>(c)].bw4 = 2;
  }
  RefMvBankRow row_view{row.data(), 16};
  RefMvBank seeded;
  reset_ref_mv_bank(&seeded, 4, 0, 16, false, false, row_view);
  CHECK_EQ(seeded.bank_hits, 4, "reset seeds four entries");
  CHECK_EQ(seeded.size[0], 4, "reset bank size");

  // fill_mvp_from_ref_mv_bank appends matching keys and prunes duplicates.
  RefMvBank fbank;
  RefMvBankBlockContext fctx;
  fctx.sb_size4 = 16;
  put_bank(&fbank, fctx, refs, Mv{16, 24}, Mv{0, 0}, 8, false);
  StackMv stack[kMaxRefMvStackSize]{};
  int num_found = 0;
  int prune = 0;
  fill_mvp_from_ref_mv_bank(fbank, refs, false, false, 4, 0, 0, 16, 16, 8, 8,
                            kMaxRefMvStackSize, stack, &num_found, &prune);
  CHECK_EQ(num_found, 1, "bank mvp appended");
  CHECK_EQ(stack[0].mv[0].row, 16, "bank mvp row");
  CHECK_EQ(stack[0].mv[0].col, 24, "bank mvp col");
  fill_mvp_from_ref_mv_bank(fbank, refs, false, false, 4, 0, 0, 16, 16, 8, 8,
                            kMaxRefMvStackSize, stack, &num_found, &prune);
  CHECK_EQ(num_found, 1, "bank mvp pruned");

  // Out-of-frame candidate is rejected.
  StackMv stack2[kMaxRefMvStackSize]{};
  num_found = 0;
  prune = 0;
  RefMvBank obank;
  put_bank(&obank, fctx, refs, Mv{0, 512}, Mv{0, 0}, 8, false);
  fill_mvp_from_ref_mv_bank(obank, refs, false, false, 4, 0, 0, 16, 16, 8, 8,
                            kMaxRefMvStackSize, stack2, &num_found, &prune);
  CHECK_EQ(num_found, 0, "bank mvp out of frame rejected");
}

// ---------------------------------------------------------------------------
// TIP prediction / weights.
// ---------------------------------------------------------------------------

void test_tip_weights_and_prediction() {
  CHECK_EQ(kTipWeightingFactor[0], 8, "tip weight 0");
  CHECK_EQ(kTipWeightingFactor[2], 16, "tip weight 2");
  CHECK_EQ(kTipWeightingFactor[7], -4, "tip weight 7");
  int w = 0;
  CHECK(tip_weight(0, &w) && w == 8, "tip_weight 0");
  CHECK(!tip_weight(8, &w), "tip_weight out of range");

  TipConfig tip;
  tip.mode = TipFrameMode::kAsReference;
  tip.num_past_refs = 1;
  tip.num_future_refs = 1;
  tip.order_hint = 4;
  tip.closest_past_order_hint = 0;
  tip.closest_future_order_hint = 8;
  const TipOffsets off = get_tip_offsets(tip);
  CHECK_EQ(off.ref_offset, 8, "tip offsets ref");
  CHECK_EQ(off.past_offset, 4, "tip offsets past");
  CHECK_EQ(off.future_offset, -4, "tip offsets future");

  // Constant planes make the blend exactly hand-checkable.
  constexpr int kSize = 16;
  std::vector<std::uint8_t> p0(kSize * kSize, 100);
  std::vector<std::uint8_t> p1(kSize * kSize, 200);
  PlaneView ref0{p0.data(), kSize, kSize, kSize};
  PlaneView ref1{p1.data(), kSize, kSize, kSize};
  std::vector<std::int16_t> s0(kSize * kSize);
  std::vector<std::int16_t> s1(kSize * kSize);
  std::vector<std::uint8_t> dst(kSize * kSize);
  TipPredictInput in;
  in.ref[0] = ref0;
  in.ref[1] = ref1;
  in.num_refs = 2;
  in.w = kSize;
  in.h = kSize;
  in.frame_w = kSize;
  in.frame_h = kSize;
  in.mv[0] = Mv{0, 0};
  in.mv[1] = Mv{0, 0};
  in.dst = dst.data();
  in.dst_stride = kSize;
  in.scratch0 = s0.data();
  in.scratch1 = s1.data();
  in.scratch_stride = kSize;

  in.cwp_weight = 8;
  CHECK_EQ(static_cast<int>(tip_predict_block(in)), 0, "tip average ok");
  CHECK_EQ(dst[0], 150, "tip average 8");
  in.cwp_weight = 12;
  CHECK_EQ(static_cast<int>(tip_predict_block(in)), 0, "tip weight 12 ok");
  CHECK_EQ(dst[0], 125, "tip weight 12");
  in.cwp_weight = 4;
  CHECK_EQ(static_cast<int>(tip_predict_block(in)), 0, "tip weight 4 ok");
  CHECK_EQ(dst[0], 175, "tip weight 4");
  in.cwp_weight = 16;
  in.num_refs = 1;
  CHECK_EQ(static_cast<int>(tip_predict_block(in)), 0, "tip single ok");
  CHECK_EQ(dst[0], 100, "tip weight 16 uses list 0");

  // 4:2:0 chroma uses luma frame dimensions for §7.13.3.17 scaling.
  constexpr int kChroma = 32;
  std::vector<std::uint8_t> c0(kChroma * kChroma);
  std::vector<std::uint8_t> c1(kChroma * kChroma);
  for (int y = 0; y < kChroma; ++y) {
    for (int x = 0; x < kChroma; ++x) {
      c0[static_cast<std::size_t>(y * kChroma + x)] =
          static_cast<std::uint8_t>((y * 5 + x * 3) & 255);
      c1[static_cast<std::size_t>(y * kChroma + x)] =
          static_cast<std::uint8_t>((y * 5 + x * 3 + 7) & 255);
    }
  }
  PlaneView chroma0{c0.data(), kChroma, kChroma, kChroma};
  PlaneView chroma1{c1.data(), kChroma, kChroma, kChroma};
  std::vector<std::uint8_t> cdst(8 * 8);
  std::vector<std::int16_t> cs0(8 * 8);
  std::vector<std::int16_t> cs1(8 * 8);
  TipPredictInput cin;
  cin.ref[0] = chroma0;
  cin.ref[1] = chroma1;
  cin.num_refs = 2;
  cin.x = 4;
  cin.y = 4;
  cin.w = 8;
  cin.h = 8;
  cin.sub_x = 1;
  cin.sub_y = 1;
  cin.frame_w = 64;
  cin.frame_h = 64;
  cin.mv[0] = Mv{16, 0};  // two luma samples = one chroma sample
  cin.mv[1] = Mv{16, 0};
  cin.cwp_weight = 8;
  cin.dst = cdst.data();
  cin.dst_stride = 8;
  cin.scratch0 = cs0.data();
  cin.scratch1 = cs1.data();
  cin.scratch_stride = 8;
  CHECK_EQ(static_cast<int>(tip_predict_block(cin)), 0, "tip chroma ok");
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      const int ca =
          static_cast<int>(c0[static_cast<std::size_t>((y + 5) * kChroma +
                                                       (x + 4))]);
      const int cb =
          static_cast<int>(c1[static_cast<std::size_t>((y + 5) * kChroma +
                                                       (x + 4))]);
      // Round2(cwp*a + (16-cwp)*b, 8) with cwp = 8 is (a+b+1)/2.
      const int expected = (ca + cb + 1) / 2;
      CHECK_EQ(static_cast<int>(cdst[static_cast<std::size_t>(y * 8 + x)]),
               expected, "tip chroma shift");
    }
  }
}

// ---------------------------------------------------------------------------
// OPFL.
// ---------------------------------------------------------------------------

void test_opfl() {
  int dist[2] = {0, 0};
  CHECK(!opfl_scale_distances(0, 1, dist), "opfl distance zero");
  CHECK(opfl_scale_distances(1, 1, dist) && dist[0] == 1 && dist[1] == 1,
        "opfl distance equal");
  CHECK(opfl_scale_distances(3, 3, dist) && dist[0] == 1 && dist[1] == 1,
        "opfl distance equal 3");
  CHECK(opfl_scale_distances(2, 1, dist) && dist[0] == 2 && dist[1] == 1,
        "opfl distance 2:1");
  CHECK(opfl_scale_distances(1, 2, dist) && dist[0] == 1 && dist[1] == 2,
        "opfl distance 1:2");
  CHECK(opfl_scale_distances(-2, 1, dist) && dist[0] == -2 && dist[1] == 1,
        "opfl distance signed");

  int result[2] = {0, 0};
  const int sol1[2] = {5, -5};
  divide_and_round_array(sol1, 1, 0, result);
  CHECK_EQ(result[0], 5, "divide exact");
  CHECK_EQ(result[1], -5, "divide exact negative");
  const int sol2[2] = {3, -3};
  divide_and_round_array(sol2, 2, 0, result);
  CHECK_EQ(result[0], 2, "divide round up");
  CHECK_EQ(result[1], -2, "divide round up negative");

  CHECK(opfl_allowed_for_refs(-1, 1, false, false, false), "opfl allowed");
  CHECK(!opfl_allowed_for_refs(1, 1, false, false, false),
        "opfl same side rejected");
  CHECK(!opfl_allowed_for_refs(-1, 1, true, false, false),
        "opfl switch frame rejected");
  CHECK(!opfl_allowed_for_refs(-1, 1, false, true, false),
        "opfl scaled rejected");

  constexpr int kSize = 16;
  std::vector<std::uint8_t> p0(kSize * kSize, 100);
  std::vector<std::uint8_t> p1(kSize * kSize, 100);
  std::vector<std::int16_t> opfl(kSize * kSize * 4, 0);
  std::vector<std::int16_t> delta(kSize * kSize * 4, 0);
  OpflJob job;
  job.w = kSize;
  job.h = kSize;
  job.p0 = p0.data();
  job.p0_stride = kSize;
  job.p1 = p1.data();
  job.p1_stride = kSize;
  job.mvs[0] = Mv{8, -8};
  job.mvs[1] = Mv{-8, 8};
  job.order_hint = 4;
  job.ref0_order_hint = 0;
  job.ref1_order_hint = 8;
  job.n = 8;
  OpflOutput out;
  out.opfl_mvs = opfl.data();
  out.opfl_stride4 = kSize / 4;
  out.mv_deltas = delta.data();
  out.delta_stride4 = kSize / 4;
  CHECK_EQ(static_cast<int>(opfl_compute(job, out)), 0, "opfl compute ok");
  for (int i = 0; i < kSize * kSize / 16; ++i) {
    for (int c = 0; c < 4; ++c) {
      CHECK_EQ(delta[static_cast<std::size_t>(i * 4 + c)], 0,
               "opfl zero delta for zero difference");
    }
  }
  // Prescaled 1/16 output is 2 * the 1/8 starting MV.
  CHECK_EQ(opfl[0], 16, "opfl list0 row prescaled");
  CHECK_EQ(opfl[1], -16, "opfl list0 col prescaled");
  CHECK_EQ(opfl[2], -16, "opfl list1 row prescaled");
  CHECK_EQ(opfl[3], 16, "opfl list1 col prescaled");
}

// ---------------------------------------------------------------------------
// Refine-MV.
// ---------------------------------------------------------------------------

void test_refinemv() {
  RefineMvConfig ctx;
  ctx.enable_refinemv = true;
  ctx.is_compound = true;
  ctx.block_width = 16;
  ctx.block_height = 16;
  ctx.order_hint = 4;
  ctx.ref0_order_hint = 0;
  ctx.ref1_order_hint = 8;
  ctx.use_refinemv_bit = true;
  RefineMvSyntax syn;
  CHECK_EQ(static_cast<int>(refine_mv_syntax(ctx, &syn)), 0, "refinemv syn ok");
  CHECK(syn.use_refinemv, "refinemv switchable use");
  CHECK(syn.switchable, "refinemv switchable");
  CHECK(refine_mv_allowed_reference(ctx), "refinemv allowed refs");

  RefineMvConfig bad = ctx;
  bad.ref0_order_hint = 4;
  CHECK(!refine_mv_allowed_reference(bad), "refinemv same-distance refs");
  bad = ctx;
  bad.block_width = 8;
  bad.block_height = 8;
  CHECK_EQ(static_cast<int>(refine_mv_syntax(bad, &syn)), 0,
           "refinemv size rejected ok");
  CHECK(!syn.use_refinemv, "refinemv size gate");
  bad = ctx;
  bad.skip_mode = true;
  CHECK(!refine_mv_allowed_mode(bad), "refinemv skip mode rejected");

  const MvPair offset =
      offset_refinemv(MvPair{Mv{16, 16}, Mv{16, 16}}, 1, -2);
  CHECK_EQ(offset.mv[0].row, 0, "offset list0 row");
  CHECK_EQ(offset.mv[0].col, 24, "offset list0 col");
  CHECK_EQ(offset.mv[1].row, 32, "offset list1 row");
  CHECK_EQ(offset.mv[1].col, 8, "offset list1 col");

  RefAreaInput area_in;
  area_in.w = 16;
  area_in.h = 16;
  area_in.x = 0;
  area_in.y = 0;
  area_in.frame_w = 64;
  area_in.frame_h = 64;
  area_in.ref_mi_cols[0] = 16;
  area_in.ref_mi_rows[0] = 16;
  area_in.ref_mi_cols[1] = 16;
  area_in.ref_mi_rows[1] = 16;
  area_in.ref_luma_w[0] = 64;
  area_in.ref_luma_h[0] = 64;
  area_in.ref_luma_w[1] = 64;
  area_in.ref_luma_h[1] = 64;
  area_in.cand_mvs[0] = Mv{0, 0};
  area_in.cand_mvs[1] = Mv{0, 0};
  RefArea areas[2];
  CHECK_EQ(static_cast<int>(compute_ref_area(area_in, areas)), 0,
           "ref area ok");
  CHECK_EQ(areas[0].first_x, 0, "ref area first x");
  CHECK_EQ(areas[0].last_x, 19, "ref area last x");
  CHECK_EQ(areas[0].first_y, 0, "ref area first y");
  CHECK_EQ(areas[0].last_y, 19, "ref area last y");

  // Search: ref1 is ref0 shifted by +2 samples; the symmetric offset dx=1
  // aligns the two predictions.
  constexpr int kSize = 64;
  std::vector<std::uint8_t> ref0_data(kSize * kSize);
  std::vector<std::uint8_t> ref1_data(kSize * kSize);
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      const int v = (x * 7 + y * 11 + ((x * y) % 13)) & 255;
      ref0_data[static_cast<std::size_t>(y * kSize + x)] =
          static_cast<std::uint8_t>(v);
    }
  }
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      const int sx = std::min(x + 2, kSize - 1);
      ref1_data[static_cast<std::size_t>(y * kSize + x)] =
          ref0_data[static_cast<std::size_t>(y * kSize + sx)];
    }
  }
  const int w = 16;
  const int h = 16;
  const int stride = w + 4;
  std::vector<std::uint8_t> scratch0(static_cast<std::size_t>(stride * (h + 4)));
  std::vector<std::uint8_t> scratch1(scratch0.size());
  RefineSearchInput search;
  search.x = 8;
  search.y = 8;
  search.w = w;
  search.h = h;
  search.frame_w = kSize;
  search.frame_h = kSize;
  search.ref_mi_cols[0] = search.ref_mi_cols[1] = kSize / 4;
  search.ref_mi_rows[0] = search.ref_mi_rows[1] = kSize / 4;
  search.ref_luma_w[0] = search.ref_luma_w[1] = kSize;
  search.ref_luma_h[0] = search.ref_luma_h[1] = kSize;
  search.ref[0] = PlaneView{ref0_data.data(), kSize, kSize, kSize};
  search.ref[1] = PlaneView{ref1_data.data(), kSize, kSize, kSize};
  search.cand_mvs[0] = Mv{0, 0};
  search.cand_mvs[1] = Mv{0, 0};
  search.num_refs = 2;
  search.tip_pred = true;
  search.switchable = false;
  search.use_ref_area = false;
  search.scratch0 = scratch0.data();
  search.scratch1 = scratch1.data();
  search.scratch_stride = stride;
  RefineSearchResult result;
  CHECK_EQ(static_cast<int>(search_refinemv(search, &result)), 0,
           "refinemv search ok");
  CHECK_EQ(result.dx, 1, "refinemv search dx");
  CHECK_EQ(result.dy, 0, "refinemv search dy");
  CHECK_EQ(result.offset_mvs.mv[0].col, 8, "refinemv offset list0");
  CHECK_EQ(result.offset_mvs.mv[1].col, -8, "refinemv offset list1");

  // Centre early-exit: identical references at the starting MVs.
  std::vector<std::uint8_t> flat(kSize * kSize, 128);
  search.ref[1] = PlaneView{flat.data(), kSize, kSize, kSize};
  search.ref[0] = PlaneView{flat.data(), kSize, kSize, kSize};
  CHECK_EQ(static_cast<int>(search_refinemv(search, &result)), 0,
           "refinemv centre ok");
  CHECK_EQ(result.dx, 0, "refinemv centre dx");
  CHECK_EQ(result.dy, 0, "refinemv centre dy");

  // Overflow guard returns zero.
  search.cand_mvs[0] = Mv{kMvLow + 1, 0};
  CHECK_EQ(static_cast<int>(search_refinemv(search, &result)), 0,
           "refinemv overflow ok");
  CHECK_EQ(result.dx, 0, "refinemv overflow dx");
}

// ---------------------------------------------------------------------------
// Syntax round trips.
// ---------------------------------------------------------------------------

struct Bits {
  av2enc::BitWriter writer;
  std::vector<std::uint8_t> bytes;
  std::size_t bit_count = 0;
};

void finish(Bits* b) {
  b->bit_count = b->writer.bit_size();
  b->writer.align_to_byte();
  b->bytes.assign(b->writer.bytes().begin(), b->writer.bytes().end());
}

bool same_sequence(const TemporalSequenceConfig& a,
                   const TemporalSequenceConfig& b) {
  return a.enable_ref_frame_mvs == b.enable_ref_frame_mvs &&
         a.reduced_ref_frame_mvs_mode == b.reduced_ref_frame_mvs_mode &&
         a.enable_refmvbank == b.enable_refmvbank &&
         a.enable_bawp == b.enable_bawp && a.enable_cwp == b.enable_cwp &&
         a.enable_imp_msk_bld == b.enable_imp_msk_bld &&
         a.enable_df_sub_pu == b.enable_df_sub_pu &&
         a.enable_tip == b.enable_tip &&
         a.disable_tip_output == b.disable_tip_output &&
         a.enable_tip_hole_fill == b.enable_tip_hole_fill &&
         a.enable_mv_traj == b.enable_mv_traj &&
         a.enable_opfl_refine == b.enable_opfl_refine &&
         a.enable_refinemv == b.enable_refinemv &&
         a.enable_tip_refinemv == b.enable_tip_refinemv &&
         a.enable_tip_explicit_qp == b.enable_tip_explicit_qp;
}

void test_sequence_syntax() {
  TemporalSequenceConfig cfg;
  cfg.enable_ref_frame_mvs = true;
  cfg.reduced_ref_frame_mvs_mode = true;
  cfg.enable_refmvbank = true;
  cfg.enable_bawp = true;
  cfg.enable_cwp = true;
  cfg.enable_imp_msk_bld = true;
  cfg.enable_df_sub_pu = true;
  cfg.enable_tip = true;
  cfg.disable_tip_output = false;
  cfg.enable_tip_hole_fill = true;
  cfg.enable_mv_traj = true;
  cfg.enable_opfl_refine = RefineType::kSwitchable;
  cfg.enable_refinemv = true;
  cfg.enable_tip_refinemv = true;
  cfg.enable_tip_explicit_qp = true;

  Bits bits;
  CHECK_EQ(static_cast<int>(write_sequence_temporal_syntax(bits.writer, cfg)),
           0, "seq write ok");
  finish(&bits);
  // 1+1 +1+1+1+1+1 +1+1+1 +1 +1 +2+1+1 = 16 bits.
  CHECK_EQ(bits.bit_count, 16, "seq bit count");
  TemporalSequenceConfig decoded;
  BitReader reader(bits.bytes.data(), bits.bytes.size());
  CHECK_EQ(static_cast<int>(read_sequence_temporal_syntax(reader, &decoded)),
           0, "seq read ok");
  CHECK(same_sequence(cfg, decoded), "seq round trip");

  // Minimal: no reference MVs, no TIP, no OPFL/refineMV.
  TemporalSequenceConfig min;
  Bits min_bits;
  CHECK_EQ(static_cast<int>(write_sequence_temporal_syntax(min_bits.writer, min)),
           0, "seq min write ok");
  finish(&min_bits);
  CHECK_EQ(min_bits.bit_count, 11, "seq min bit count");
  TemporalSequenceConfig min_decoded;
  BitReader min_reader(min_bits.bytes.data(), min_bits.bytes.size());
  CHECK_EQ(static_cast<int>(read_sequence_temporal_syntax(min_reader,
                                                          &min_decoded)),
           0, "seq min read ok");
  CHECK(same_sequence(min, min_decoded), "seq min round trip");

  // REFINE_AUTO (3) is a valid sequence-header value.
  TemporalSequenceConfig auto_cfg = min;
  auto_cfg.enable_tip = true;
  auto_cfg.enable_opfl_refine = RefineType::kAuto;
  auto_cfg.enable_refinemv = true;
  auto_cfg.enable_tip_refinemv = true;
  Bits auto_bits;
  CHECK_EQ(static_cast<int>(write_sequence_temporal_syntax(auto_bits.writer,
                                                          auto_cfg)),
           0, "seq auto write ok");
  finish(&auto_bits);
  TemporalSequenceConfig auto_decoded;
  BitReader auto_reader(auto_bits.bytes.data(), auto_bits.bytes.size());
  CHECK_EQ(static_cast<int>(read_sequence_temporal_syntax(auto_reader,
                                                          &auto_decoded)),
           0, "seq auto read ok");
  CHECK(auto_decoded.enable_opfl_refine == RefineType::kAuto,
        "seq auto value");

  // Truncated input must fail rather than read past the buffer.
  BitReader truncated(bits.bytes.data(), 1);
  TemporalSequenceConfig bad;
  CHECK(read_sequence_temporal_syntax(truncated, &bad) ==
            TemporalStatus::kInvalidArgument,
        "seq truncated rejected");
}

void test_frame_syntax() {
  TemporalFrameConfig base;
  base.use_ref_frame_mvs = true;
  base.num_total_refs = 2;
  base.sb_64x64 = false;
  base.enable_tip = true;
  base.enable_tip_hole_fill = true;
  base.enable_opfl_refine = RefineType::kSwitchable;
  base.opfl_refine_type = RefineType::kSwitchable;
  base.enable_refinemv = true;
  base.num_past_refs = 1;
  base.num_future_refs = 1;
  base.tmvp_sample_step_minus_1 = 1;
  base.tip.mode = TipFrameMode::kAsReference;
  base.tip.tip_global_wtd_index = 3;
  base.tip.allow_tip_hole_fill = true;
  base.tip_sharp = true;

  Bits bits;
  CHECK_EQ(static_cast<int>(write_frame_temporal_syntax(bits.writer, base)), 0,
           "frame write as-ref ok");
  finish(&bits);
  // tmvp(1) + tip_mode(1) + hole(1) + weight(3) + sharp(1) = 7, no opfl bits.
  CHECK_EQ(bits.bit_count, 7, "frame as-ref bit count");
  TemporalFrameConfig decoded = base;
  decoded.tip = TipConfig{};
  BitReader reader(bits.bytes.data(), bits.bytes.size());
  CHECK_EQ(static_cast<int>(read_frame_temporal_syntax(reader, &decoded)), 0,
           "frame read as-ref ok");
  CHECK_EQ(decoded.tmvp_sample_step_minus_1, 1, "frame tmvp");
  CHECK(decoded.tip.mode == TipFrameMode::kAsReference, "frame mode");
  CHECK_EQ(decoded.tip.tip_global_wtd_index, 3, "frame weight");
  CHECK(decoded.tip.allow_tip_hole_fill, "frame hole fill");
  CHECK_EQ(static_cast<int>(decoded.tip.interp_filter),
           static_cast<int>(InterpFilter::kEighttapSharp), "frame filter");

  // TIP as output with a global MV and the regular filter branch.
  TemporalFrameConfig as_out = base;
  as_out.is_tip_frame = true;
  as_out.enable_tip_output = true;
  as_out.tip.mode = TipFrameMode::kAsOutput;
  as_out.tip.tip_global_wtd_index = 5;
  as_out.tip.tip_global_mv = Mv{-3, 5};
  as_out.tip_sharp = false;
  as_out.tip_regular = true;
  Bits out_bits;
  CHECK_EQ(static_cast<int>(write_frame_temporal_syntax(out_bits.writer, as_out)),
           0, "frame write as-out ok");
  finish(&out_bits);
  // tmvp(1) + opfl(0) + hole(1) + weight(3) + zero(1) + row(4)+col(4)
  // + row sign(1) + col sign(1) + sharp(1) + regular(1) = 18.
  CHECK_EQ(out_bits.bit_count, 18, "frame as-out bit count");
  TemporalFrameConfig out_decoded = as_out;
  out_decoded.tip = TipConfig{};
  BitReader out_reader(out_bits.bytes.data(), out_bits.bytes.size());
  CHECK_EQ(static_cast<int>(read_frame_temporal_syntax(out_reader,
                                                       &out_decoded)),
           0, "frame read as-out ok");
  CHECK(out_decoded.tip.mode == TipFrameMode::kAsOutput, "frame as-out mode");
  CHECK_EQ(out_decoded.tip.tip_global_mv.row, -3, "frame global mv row");
  CHECK_EQ(out_decoded.tip.tip_global_mv.col, 5, "frame global mv col");
  CHECK_EQ(static_cast<int>(out_decoded.tip.interp_filter),
           static_cast<int>(InterpFilter::kEighttap), "frame regular filter");

  // TIP disabled: only frame_opfl_refine_type may be present.
  TemporalFrameConfig disabled;
  disabled.enable_tip = false;
  disabled.enable_opfl_refine = RefineType::kAuto;
  disabled.opfl_refine_type = RefineType::kAll;
  Bits disabled_bits;
  CHECK_EQ(static_cast<int>(write_frame_temporal_syntax(disabled_bits.writer,
                                                        disabled)),
           0, "frame write disabled ok");
  finish(&disabled_bits);
  CHECK_EQ(disabled_bits.bit_count, 2, "frame disabled bit count");
  TemporalFrameConfig disabled_decoded = disabled;
  BitReader disabled_reader(disabled_bits.bytes.data(),
                            disabled_bits.bytes.size());
  CHECK_EQ(static_cast<int>(read_frame_temporal_syntax(disabled_reader,
                                                       &disabled_decoded)),
           0, "frame read disabled ok");
  CHECK(disabled_decoded.opfl_refine_type == RefineType::kAll,
        "frame auto all");

  // enable_tip with TipFrameMode == DISABLED still codes tip_sharp.
  TemporalFrameConfig tip_off = base;
  tip_off.tip.mode = TipFrameMode::kDisabled;
  tip_off.tip_sharp = true;
  Bits off_bits;
  CHECK_EQ(static_cast<int>(write_frame_temporal_syntax(off_bits.writer,
                                                        tip_off)),
           0, "frame write tip-off ok");
  finish(&off_bits);
  CHECK_EQ(off_bits.bit_count, 3, "frame tip-off bit count");
  TemporalFrameConfig off_decoded = tip_off;
  off_decoded.tip = TipConfig{};
  BitReader off_reader(off_bits.bytes.data(), off_bits.bytes.size());
  CHECK_EQ(static_cast<int>(read_frame_temporal_syntax(off_reader,
                                                       &off_decoded)),
           0, "frame read tip-off ok");
  CHECK(off_decoded.tip.mode == TipFrameMode::kDisabled,
        "frame tip-off mode");

  // TIP_SMOOTH branch: tip_sharp = 0, tip_regular = 0.
  TemporalFrameConfig smooth = base;
  smooth.tip_sharp = false;
  smooth.tip_regular = false;
  Bits smooth_bits;
  CHECK_EQ(static_cast<int>(write_frame_temporal_syntax(smooth_bits.writer,
                                                        smooth)),
           0, "frame write smooth ok");
  finish(&smooth_bits);
  TemporalFrameConfig smooth_decoded = smooth;
  smooth_decoded.tip = TipConfig{};
  BitReader smooth_reader(smooth_bits.bytes.data(), smooth_bits.bytes.size());
  CHECK_EQ(static_cast<int>(read_frame_temporal_syntax(smooth_reader,
                                                       &smooth_decoded)),
           0, "frame read smooth ok");
  CHECK_EQ(static_cast<int>(smooth_decoded.tip.interp_filter),
           static_cast<int>(InterpFilter::kEighttapSmooth),
           "frame smooth filter");

  // Auto derive: opfl_refine_type switchable -> one bit.
  TemporalFrameConfig auto_switch = disabled;
  auto_switch.opfl_refine_type = RefineType::kSwitchable;
  Bits auto_bits;
  CHECK_EQ(static_cast<int>(write_frame_temporal_syntax(auto_bits.writer,
                                                       auto_switch)),
           0, "frame write auto switch ok");
  finish(&auto_bits);
  CHECK_EQ(auto_bits.bit_count, 1, "frame auto switch bit count");
}

void test_block_syntax() {
  TipBlockSyntaxConfig cfg;
  cfg.tip_frame_mode = TipFrameMode::kAsReference;
  cfg.bw4 = 4;
  cfg.bh4 = 4;
  cfg.mi_size_matches_chroma = true;
  cfg.tip_mode = true;
  cfg.tip_pred_mode = 1;
  Bits bits;
  CHECK_EQ(static_cast<int>(write_tip_block_syntax(bits.writer, cfg)), 0,
           "tip block write ok");
  finish(&bits);
  CHECK_EQ(bits.bit_count, 2, "tip block bit count");
  TipBlockSyntaxConfig decoded;
  BitReader reader(bits.bytes.data(), bits.bytes.size());
  CHECK_EQ(static_cast<int>(read_tip_block_syntax(reader, cfg, &decoded)), 0,
           "tip block read ok");
  CHECK(decoded.tip_mode, "tip block mode");
  CHECK_EQ(decoded.tip_pred_mode, 1, "tip block pred mode");

  // Skip mode and tiny blocks do not code tip_mode.
  TipBlockSyntaxConfig skip;
  skip.tip_frame_mode = TipFrameMode::kAsReference;
  skip.skip_mode = true;
  skip.bw4 = 4;
  skip.bh4 = 4;
  Bits skip_bits;
  CHECK_EQ(static_cast<int>(write_tip_block_syntax(skip_bits.writer, skip)), 0,
           "tip skip write ok");
  finish(&skip_bits);
  CHECK_EQ(skip_bits.bit_count, 0, "tip skip bit count");

  TipBlockSyntaxConfig tiny;
  tiny.tip_frame_mode = TipFrameMode::kAsReference;
  tiny.bw4 = 1;
  tiny.bh4 = 1;
  Bits tiny_bits;
  CHECK_EQ(static_cast<int>(write_tip_block_syntax(tiny_bits.writer, tiny)), 0,
           "tip tiny write ok");
  finish(&tiny_bits);
  CHECK_EQ(tiny_bits.bit_count, 0, "tip tiny bit count");

  // read_refinemv: switchable writes one bit; fixed writes none.
  RefineMvBlockSyntaxConfig rcfg;
  rcfg.ctx.enable_refinemv = true;
  rcfg.ctx.is_compound = true;
  rcfg.ctx.block_width = 16;
  rcfg.ctx.block_height = 16;
  rcfg.ctx.order_hint = 4;
  rcfg.ctx.ref0_order_hint = 0;
  rcfg.ctx.ref1_order_hint = 8;
  rcfg.ctx.use_refinemv_bit = true;
  Bits rbits;
  CHECK_EQ(static_cast<int>(write_refinemv_block_syntax(rbits.writer, rcfg)), 0,
           "refinemv syntax write ok");
  finish(&rbits);
  CHECK_EQ(rbits.bit_count, 1, "refinemv syntax bit count");
  RefineMvBlockSyntaxConfig rdecoded;
  BitReader rreader(rbits.bytes.data(), rbits.bytes.size());
  CHECK_EQ(static_cast<int>(read_refinemv_block_syntax(rreader, rcfg.ctx,
                                                        &rdecoded)),
           0, "refinemv syntax read ok");
  CHECK(rdecoded.syntax.use_refinemv, "refinemv syntax value");

  RefineMvBlockSyntaxConfig rfixed = rcfg;
  rfixed.ctx.y_mode_near_near_mv = true;
  Bits rfix_bits;
  CHECK_EQ(static_cast<int>(write_refinemv_block_syntax(rfix_bits.writer,
                                                        rfixed)),
           0, "refinemv fixed write ok");
  finish(&rfix_bits);
  CHECK_EQ(rfix_bits.bit_count, 0, "refinemv fixed bit count");
  RefineMvBlockSyntaxConfig rfix_decoded;
  BitReader rfix_reader(rfix_bits.bytes.data(), rfix_bits.bytes.size());
  CHECK_EQ(static_cast<int>(read_refinemv_block_syntax(rfix_reader,
                                                       rfixed.ctx,
                                                       &rfix_decoded)),
           0, "refinemv fixed read ok");
  CHECK(rfix_decoded.syntax.use_refinemv, "refinemv fixed value");
}

// ---------------------------------------------------------------------------
// Determinism.
// ---------------------------------------------------------------------------

void test_determinism() {
  FieldFixture f1;
  FieldFixture f2;
  std::vector<std::int16_t> saved_mv;
  std::vector<std::int8_t> saved_ref;
  MotionFieldConfig cfg = make_two_ref_config(&f1, &saved_mv, &saved_ref);
  MotionFieldResult r1;
  MotionFieldResult r2;
  motion_field_estimation(cfg, &f1.field, nullptr, &r1);
  cfg.mi_cols = f2.mi_cols;
  cfg.mi_rows = f2.mi_rows;
  cfg.saved[0].mv = saved_mv.data();
  cfg.saved[0].ref = saved_ref.data();
  motion_field_estimation(cfg, &f2.field, nullptr, &r2);
  CHECK_EQ(r1.closest_past, r2.closest_past, "determinism past");
  CHECK_EQ(r1.closest_future, r2.closest_future, "determinism future");
  CHECK(f1.valid == f2.valid, "determinism valid");
  CHECK(f1.mv == f2.mv, "determinism mv");
  CHECK(f1.offset == f2.offset, "determinism offset");
}

}  // namespace

int main() {
  std::cout << "AV2 temporal inter CPU reference tests\n";
  test_projection_math();
  test_sampled_position();
  test_motion_field_estimation();
  test_tip_hole_fill_and_average();
  test_motion_field_storage();
  test_temporal_scan();
  test_ref_mv_bank();
  test_tip_weights_and_prediction();
  test_opfl();
  test_refinemv();
  test_sequence_syntax();
  test_frame_syntax();
  test_block_syntax();
  test_determinism();

  std::cout << "\nchecks: " << g_checks << ", failures: " << g_failures
            << "\n";
  return g_failures == 0 ? 0 : 1;
}
