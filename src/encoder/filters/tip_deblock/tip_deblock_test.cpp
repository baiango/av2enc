// CPU reference tests for the AV2 §7.16 TIP deblocking filter.
//
// Coverage:
//   * tipSize = 8x8 / 16x16 selection (all enable_tip_refinemv x filter cases)
//   * per-plane baseFilterLevel / qThr / side derivation, including chroma
//     DeltaQUAc / DeltaQVAc / BaseUVAcDeltaQ
//   * filter_maximum_width classes for every tipSize/plane/boundary kind
//   * the TIP edge lattice: visited vs. unvisited boundaries, x==0 / y==0
//   * exact filtered sample values on steps (hand-derived from §7.17.7)
//   * vertical and horizontal tile edges with both
//     disable_loopfilters_across_tiles values
//   * the horizontal 64x64-grid sbEdge clamp
//   * enable_df_sub_pu / allow_df_sub_pu / apply_deblocking_filter_tip
//     interaction and "do not run on non-TIP frames" behaviour
//   * flat / checker rejection, guard columns, randomized fuzz with a
//     conservative write-envelope invariant, determinism
//   * a real reconstructed frame when decoded.y4m is present
//
// Usage: ./tip_deblock_test [repo_root]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "encoder/filters/tip_deblock/tip_deblock.h"
#include "encoder/filters/tip_deblock/tip_deblock_test_common.h"

namespace {

using av2enc::tip_deblock::InterpFilter;
using av2enc::tip_deblock::TipDeblockEdgeInfo;
using av2enc::tip_deblock::TipDeblockFrame;
using av2enc::tip_deblock::TipDeblockOptions;
using av2enc::tip_deblock::TipDeblockPlaneInfo;
using av2enc::tip_deblock::TipDeblockStatus;
using av2enc::tip_deblock::TipFrameMode;
using av2enc::tip_deblock::TipSize;
using namespace av2enc::tip_deblock::test;

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

TipDeblockOptions base_options() {
  TipDeblockOptions options;
  options.tip_frame_mode = TipFrameMode::kAsOutput;
  options.apply_deblocking_filter_tip = true;
  options.enable_df_sub_pu = true;
  options.allow_df_sub_pu = true;
  options.disable_loopfilters_across_tiles = false;
  options.base_q_idx = 255;
  options.delta_q_u_ac = 0;
  options.delta_q_v_ac = 0;
  options.base_uv_ac_delta_q = 0;
  options.bit_depth = 8;
  options.enable_tip_refinemv = false;
  options.tip_interp_filter = InterpFilter::kEighttapSharp;
  return options;
}

TipDeblockOptions tip8_options() {
  TipDeblockOptions options = base_options();
  options.enable_tip_refinemv = true;
  options.tip_interp_filter = InterpFilter::kEighttapSharp;
  return options;
}

TipDeblockStatus run(TipTestFrame& frame, const TipDeblockOptions& options) {
  TipDeblockFrame view = frame.view();
  return av2enc::tip_deblock::tip_deblock_frame(view, options);
}

int pixel(const TipTestFrame& frame, int plane, int x, int y) {
  return static_cast<int>(
      frame.plane(plane)[pix_index(y, frame.stride[plane], x)]);
}

// ---------------------------------------------------------------------------
// tipSize selection
// ---------------------------------------------------------------------------
void test_tip_size_selection() {
  const InterpFilter filters[] = {InterpFilter::kEighttap,
                                  InterpFilter::kEighttapSmooth,
                                  InterpFilter::kEighttapSharp};
  for (InterpFilter filter : filters) {
    TipDeblockOptions options = base_options();
    options.tip_interp_filter = filter;
    options.enable_tip_refinemv = false;
    CHECK(av2enc::tip_deblock::tip_deblock_tip_size(options) ==
              TipSize::kBlock16x16,
          "refineMV off must select 16x16");
    options.enable_tip_refinemv = true;
    const bool sharp = filter == InterpFilter::kEighttapSharp;
    CHECK(av2enc::tip_deblock::tip_deblock_tip_size(options) ==
              (sharp ? TipSize::kBlock8x8 : TipSize::kBlock16x16),
          "refineMV on selects 8x8 only for EIGHTTAP_SHARP");
  }
  CHECK(av2enc::tip_deblock::tip_width(TipSize::kBlock8x8) == 8,
        "BLOCK_8X8 width");
  CHECK(av2enc::tip_deblock::tip_height(TipSize::kBlock8x8) == 8,
        "BLOCK_8X8 height");
  CHECK(av2enc::tip_deblock::tip_width(TipSize::kBlock16x16) == 16,
        "BLOCK_16X16 width");
  CHECK(av2enc::tip_deblock::tip_height(TipSize::kBlock16x16) == 16,
        "BLOCK_16X16 height");
  std::cout << "  tipSize selection\n";
}

// ---------------------------------------------------------------------------
// Strength derivation.  Literals are independently computed from the §9.2
// tables with the §7.16 formula (not by calling the component).
// ---------------------------------------------------------------------------
void test_strength_derivation() {
  struct Case {
    int base_q_idx;
    int q_thr;
    int side;
  };
  const Case cases[] = {{0, 0, 0},   {1, 0, 0},   {8, 0, 0},
                        {24, 0, 0},  {40, 0, 0},  {60, 0, 0},
                        {80, 0, 1},  {96, 1, 1},  {100, 1, 1},
                        {128, 3, 3}, {160, 7, 11}, {200, 24, 23},
                        {255, 120, 40}};
  for (const Case& c : cases) {
    TipDeblockOptions options = base_options();
    options.base_q_idx = c.base_q_idx;
    for (int plane = 0; plane < 3; ++plane) {
      const TipDeblockPlaneInfo info =
          av2enc::tip_deblock::tip_deblock_plane_info(options, plane);
      CHECK(info.base_filter_level == c.base_q_idx,
            "plane 0/U/V with zero deltas shares baseFilterLevel");
      CHECK(info.q_thr == c.q_thr, "qThr from QUANT_TABLE_BITS formula");
      CHECK(info.side == c.side, "side from Side_Thresholds formula");
    }
  }

  // Chroma deltas are added before the qThr/side derivation, and
  // BaseUVAcDeltaQ applies to both U and V.
  {
    TipDeblockOptions options = base_options();
    options.base_q_idx = 100;
    options.delta_q_u_ac = -100;
    options.delta_q_v_ac = 100;
    options.base_uv_ac_delta_q = 0;
    const TipDeblockPlaneInfo u =
        av2enc::tip_deblock::tip_deblock_plane_info(options, 1);
    const TipDeblockPlaneInfo v =
        av2enc::tip_deblock::tip_deblock_plane_info(options, 2);
    const TipDeblockPlaneInfo y =
        av2enc::tip_deblock::tip_deblock_plane_info(options, 0);
    CHECK(u.base_filter_level == 0 && u.q_thr == 0 && u.side == 0,
          "DeltaQUAc makes U level zero -> no filtering");
    CHECK(v.base_filter_level == 200 && v.q_thr == 24 && v.side == 23,
          "DeltaQVAc raises V level");
    CHECK(y.base_filter_level == 100 && y.q_thr == 1 && y.side == 1,
          "luma ignores chroma deltas");
  }
  {
    TipDeblockOptions options = base_options();
    options.base_q_idx = 100;
    options.delta_q_u_ac = 10;
    options.delta_q_v_ac = 10;
    options.base_uv_ac_delta_q = 10;
    const TipDeblockPlaneInfo u =
        av2enc::tip_deblock::tip_deblock_plane_info(options, 1);
    CHECK(u.base_filter_level == 120, "BaseUVAcDeltaQ is added for U");
    CHECK(u.q_thr == 2 && u.side == 2,
          "U strength at baseFilterLevel 120");
  }
  {
    TipDeblockOptions options = base_options();
    options.base_q_idx = 10;
    options.delta_q_u_ac = -10;
    options.base_uv_ac_delta_q = -10;
    const TipDeblockPlaneInfo u =
        av2enc::tip_deblock::tip_deblock_plane_info(options, 1);
    CHECK(u.base_filter_level == -10, "negative baseFilterLevel is preserved");
    CHECK(u.q_thr == 0 && u.side == 0,
          "negative baseFilterLevel yields qThr=0/side=0");
  }
  std::cout << "  strength derivation (base q, planes, chroma deltas)\n";
}

// Independent transcription of the §7.16 strength formula, used to cover all
// 256 base_q_idx values without calling the component's derivation helpers.
TipDeblockPlaneInfo reference_plane_info(const TipDeblockOptions& options,
                                         int plane) {
  auto ref_clip3 = [](int lo, int hi, int value) {
    return value < lo ? lo : (value > hi ? hi : value);
  };
  auto ref_round2 = [](int value, int shift) {
    return shift == 0 ? value : ((value + (1 << (shift - 1))) >> shift);
  };
  auto ref_qlookup = [](int q) -> int {
    if (q < 25) {
      return av2enc::deblock::kAcQlookup[q];
    }
    return av2enc::deblock::kAcQlookup[((q - 1) % 24) + 1]
           << ((q - 1) / 24);
  };
  auto ref_get_q = [&](int qindex, int delta) -> int {
    if (qindex == 0 && delta <= 0) {
      return av2enc::deblock::kAcQlookup[0];
    }
    return ref_qlookup(ref_clip3(1, 255, qindex + delta));
  };
  auto ref_max = [](int a, int b) { return a > b ? a : b; };

  int base = options.base_q_idx;
  if (plane == 1) {
    base += options.delta_q_u_ac + options.base_uv_ac_delta_q;
  } else if (plane == 2) {
    base += options.delta_q_v_ac + options.base_uv_ac_delta_q;
  }
  TipDeblockPlaneInfo info;
  info.base_filter_level = base;
  info.q_thr = ref_round2(ref_get_q(base, 0), 3) >> 6;
  const int q_ind =
      ref_clip3(0, av2enc::deblock::kMaxSideTable - 1,
                base - 24 * (options.bit_depth - 8));
  info.side =
      ref_max(av2enc::deblock::kSideThresholds[q_ind] +
                  (1 << (12 - options.bit_depth)),
              0) >>
      (13 - options.bit_depth);
  return info;
}

void test_strength_all_q() {
  struct DeltaConfig {
    int u;
    int v;
    int uv;
  };
  const DeltaConfig configs[] = {{0, 0, 0}, {-37, 23, -11}};
  TipDeblockOptions options = base_options();
  for (const DeltaConfig& config : configs) {
    options.delta_q_u_ac = config.u;
    options.delta_q_v_ac = config.v;
    options.base_uv_ac_delta_q = config.uv;
    for (int q = 0; q <= 255; ++q) {
      options.base_q_idx = q;
      for (int plane = 0; plane < 3; ++plane) {
        const TipDeblockPlaneInfo actual =
            av2enc::tip_deblock::tip_deblock_plane_info(options, plane);
        const TipDeblockPlaneInfo expected =
            reference_plane_info(options, plane);
        CHECK(actual.base_filter_level == expected.base_filter_level,
              "all-q baseFilterLevel");
        CHECK(actual.q_thr == expected.q_thr, "all-q qThr");
        CHECK(actual.side == expected.side, "all-q side");
      }
    }
  }
  std::cout << "  strength derivation (all 256 base_q_idx values)\n";
}

// ---------------------------------------------------------------------------
// filter_maximum_width for every tipSize/plane/boundary combination.
// ---------------------------------------------------------------------------
void test_max_width_classes() {
  TipTestFrame frame = make_frame(40, 40);  // 160x160 luma, 80x80 chroma
  frame.tile_cols = {0, 8};
  frame.tile_rows = {0, 2};
  TipDeblockFrame view = frame.view();

  struct Case {
    int plane;
    int pass;
    int x;
    int y;
    bool tip8;
    int filter_size;
    int pos;
    int neg;
    int sb_edge;
    int tile_vert;
    int tile_horz;
  };
  // Luma: 8x8 tip -> filterSize 8 -> 3/3; 16x16 tip -> 16 -> 6/6.
  // Chroma 4:2:0: 8x8 tip -> 4 -> 1/1; 16x16 tip -> 8 -> 3/3.
  // sbEdge clamps neg to 6 luma / 2 chroma.
  const Case cases[] = {
      {0, 0, 16, 0, true, 8, 3, 3, 0, 0, 0},
      {0, 0, 16, 0, false, 16, 6, 6, 0, 0, 0},
      {0, 0, 32, 0, false, 16, 6, 6, 1, 1, 0},   // vertical tile edge
      {0, 1, 0, 8, true, 8, 3, 3, 0, 0, 1},      // horizontal tile, no 64
      {0, 1, 0, 64, true, 8, 3, 3, 1, 0, 0},     // 64-grid
      {1, 0, 8, 0, true, 4, 1, 1, 0, 0, 0},
      {1, 0, 8, 0, false, 8, 3, 3, 0, 0, 0},
      {1, 0, 16, 0, false, 8, 3, 2, 1, 1, 0},     // chroma tile clamp
      {2, 0, 16, 0, false, 8, 3, 2, 1, 1, 0},
      {1, 1, 0, 48, false, 8, 3, 3, 0, 0, 0},
      {1, 1, 0, 64, false, 8, 3, 2, 1, 0, 0},    // chroma 64-grid clamp
      {2, 1, 0, 64, false, 8, 3, 2, 1, 0, 0},
  };

  for (const Case& c : cases) {
    TipDeblockOptions options = c.tip8 ? tip8_options() : base_options();
    const TipDeblockEdgeInfo info = av2enc::tip_deblock::tip_deblock_inspect_edge(
        view, options, c.plane, c.pass, c.x, c.y);
    CHECK(info.on_screen == 1, "edge should be on the TIP lattice");
    CHECK(info.filter_size == c.filter_size, "filterSize = sw/sh");
    CHECK(info.max_width_pos == c.pos, "maxWidthPos class");
    CHECK(info.max_width_neg == c.neg, "maxWidthNeg class / sbEdge clamp");
    CHECK(info.sb_edge == c.sb_edge, "sbEdge selection per pass");
    CHECK(info.tile_vert_edge == c.tile_vert, "vertical tile flag");
    CHECK(info.tile_horz_edge == c.tile_horz, "horizontal tile flag");
  }

  // Unvisited positions are off the §7.16 lattice.
  {
    const TipDeblockOptions tip16 = base_options();
    const TipDeblockOptions tip8 = tip8_options();
    CHECK(av2enc::tip_deblock::tip_deblock_inspect_edge(view, tip16, 0, 0, 8, 0)
                  .on_screen == 0,
          "16x16 tip does not visit luma x=8");
    CHECK(av2enc::tip_deblock::tip_deblock_inspect_edge(view, tip8, 0, 0, 4, 0)
                  .on_screen == 0,
          "8x8 tip does not visit luma x=4");
    CHECK(av2enc::tip_deblock::tip_deblock_inspect_edge(view, tip16, 0, 0, 16, 1)
                  .on_screen == 0,
          "vertical pass visits y multiples of 4 only");
    CHECK(av2enc::tip_deblock::tip_deblock_inspect_edge(view, tip16, 0, 1, 1, 16)
                  .on_screen == 0,
          "horizontal pass visits x multiples of 4 only");
    CHECK(av2enc::tip_deblock::tip_deblock_inspect_edge(view, tip16, 0, 0, 0, 0)
                  .on_screen == 0,
          "x == 0 is not a filtered vertical edge");
    CHECK(av2enc::tip_deblock::tip_deblock_inspect_edge(view, tip16, 0, 1, 0, 0)
                  .on_screen == 0,
          "y == 0 is not a filtered horizontal edge");
  }
  std::cout << "  filter_maximum_width / edge lattice\n";
}

// ---------------------------------------------------------------------------
// Exact filtered samples, hand-derived from §7.17.7 for a 20-code step.
// ---------------------------------------------------------------------------
void test_exact_luma8() {
  TipTestFrame frame = make_frame(4, 4);  // 16x16 luma
  fill_vertical_step(frame, 8, 100, 120);
  TipTestFrame before = frame;
  const TipDeblockStatus status = run(frame, tip8_options());
  CHECK(status == TipDeblockStatus::kOk, "8x8 tip luma run");

  const int expected[6] = {103, 106, 109, 111, 114, 117};
  for (int i = 0; i < 6; ++i) {
    CHECK(pixel(frame, 0, 5 + i, 6) == expected[i],
          "8x8 tip luma exact filtered samples");
  }
  CHECK(pixel(frame, 0, 4, 6) == 100, "sample beyond maxWidthNeg untouched");
  CHECK(pixel(frame, 0, 11, 6) == 120, "sample beyond maxWidthPos untouched");
  CHECK(pixel(frame, 0, 0, 6) == 100, "x == 0 never modified");
  CHECK(guard_intact(frame), "guard samples intact");
  CHECK(count_changed_in_plane(before, frame, 0) == 16 * 6,
        "8x8 tip writes exactly 6 rows x 16 visits");
  std::cout << "  exact luma 8x8\n";
}

void test_exact_luma16() {
  TipTestFrame frame = make_frame(8, 8);  // 32x32 luma
  fill_vertical_step(frame, 16, 100, 120);
  TipTestFrame before = frame;
  CHECK(run(frame, base_options()) == TipDeblockStatus::kOk,
        "16x16 tip luma run");

  const int expected[12] = {102, 103, 105, 106, 108, 109,
                            111, 112, 114, 115, 117, 118};
  for (int i = 0; i < 12; ++i) {
    CHECK(pixel(frame, 0, 10 + i, 5) == expected[i],
          "16x16 tip luma exact filtered samples");
  }
  CHECK(pixel(frame, 0, 9, 5) == 100, "beyond maxWidthNeg untouched");
  CHECK(pixel(frame, 0, 22, 5) == 120, "beyond maxWidthPos untouched");
  CHECK(count_changed_in_plane(before, frame, 0) == 32 * 12,
        "16x16 tip writes 12 samples across 32 rows");
  CHECK(guard_intact(frame), "guard samples intact");
  std::cout << "  exact luma 16x16\n";
}

void test_exact_chroma() {
  // 16x16 tip: chroma filterSize 8 -> maxWidthPos/Neg 3 (same as luma 8x8).
  {
    TipTestFrame frame = make_frame(8, 8);  // chroma 16x16
    fill_vertical_step(frame, 16, 100, 120);
    TipTestFrame before = frame;
    CHECK(run(frame, base_options()) == TipDeblockStatus::kOk,
          "16x16 tip chroma run");
    const int expected[6] = {103, 106, 109, 111, 114, 117};
    for (int i = 0; i < 6; ++i) {
      CHECK(pixel(frame, 1, 5 + i, 6) == expected[i],
            "16x16 tip U exact filtered samples");
      CHECK(pixel(frame, 2, 5 + i, 6) == expected[i],
            "16x16 tip V exact filtered samples");
    }
    CHECK(count_changed_in_plane(before, frame, 1) == 16 * 6 &&
              count_changed_in_plane(before, frame, 2) == 16 * 6,
          "16x16 tip chroma writes both planes");
  }
  // 8x8 tip: chroma filterSize 4 -> maxWidthPos/Neg 1.
  {
    TipTestFrame frame = make_frame(4, 4);  // chroma 8x8
    fill_vertical_step(frame, 8, 100, 120);
    CHECK(run(frame, tip8_options()) == TipDeblockStatus::kOk,
          "8x8 tip chroma run");
    CHECK(pixel(frame, 1, 3, 2) == 107, "8x8 tip U maxWidthNeg sample");
    CHECK(pixel(frame, 1, 4, 2) == 113, "8x8 tip U maxWidthPos sample");
    CHECK(pixel(frame, 1, 2, 2) == 100, "8x8 tip U farther sample untouched");
    CHECK(pixel(frame, 1, 5, 2) == 120, "8x8 tip U farther sample untouched");
  }
  std::cout << "  exact chroma (both tip sizes)\n";
}

void test_unvisited_edges() {
  // A step in the middle of a 16x16 TIP block must not be filtered.
  {
    TipTestFrame frame = make_frame(8, 8);
    fill_vertical_step(frame, 8, 100, 120);
    TipTestFrame before = frame;
    CHECK(run(frame, base_options()) == TipDeblockStatus::kOk, "run");
    CHECK(count_changed(before, frame) == 0,
          "luma step at x=8 is inside a 16x16 TIP block");
  }
  {
    TipTestFrame frame = make_frame(4, 4);
    fill_horizontal_step(frame, 4, 100, 120);
    TipTestFrame before = frame;
    CHECK(run(frame, tip8_options()) == TipDeblockStatus::kOk, "run");
    CHECK(count_changed(before, frame) == 0,
          "luma step at y=4 is inside an 8x8 TIP block");
  }
  {
    // 16x16 tip chroma: chroma x=4 is inside the 8-wide chroma TIP block.
    TipTestFrame frame = make_frame(8, 8);
    fill_vertical_step(frame, 8, 100, 120);
    TipTestFrame before = frame;
    CHECK(run(frame, base_options()) == TipDeblockStatus::kOk, "run");
    CHECK(pixel(frame, 1, 3, 2) == 100 && pixel(frame, 1, 4, 2) == 120,
          "chroma step at x=4 is not a 16x16-tip chroma edge");
    CHECK(pixel(frame, 2, 3, 2) == 100 && pixel(frame, 2, 4, 2) == 120,
          "V also ignores the chroma x=4 position");
    CHECK(count_changed(before, frame) == 0,
          "no 16x16 TIP edge intersects this step");
  }
  std::cout << "  unvisited boundaries\n";
}

// ---------------------------------------------------------------------------
// Plane-specific strength: U and V can be disabled independently.
// ---------------------------------------------------------------------------
void test_plane_deltas() {
  TipTestFrame frame = make_frame(4, 4);
  fill_vertical_step(frame, 8, 100, 120);
  TipTestFrame before = frame;
  TipDeblockOptions options = tip8_options();
  options.base_q_idx = 100;
  options.delta_q_u_ac = -100;  // U level 0 -> side 0 -> no filtering
  options.delta_q_v_ac = 100;   // V level 200 -> filtering
  CHECK(run(frame, options) == TipDeblockStatus::kOk, "run");
  CHECK(count_changed(before, frame) > 0, "luma and V are filtered");
  for (int y = 0; y < frame.plane_height(1); ++y) {
    for (int x = 0; x < frame.plane_width(1); ++x) {
      if (pixel(frame, 1, x, y) != pixel(before, 1, x, y)) {
        CHECK(false, "U must not change when its baseFilterLevel is 0");
        y = frame.plane_height(1);
        break;
      }
    }
  }
  CHECK(pixel(frame, 2, 4, 6) != pixel(before, 2, 4, 6),
        "V is filtered at baseFilterLevel 200");
  CHECK(pixel(frame, 0, 8, 6) != pixel(before, 0, 8, 6),
        "Y is filtered at baseFilterLevel 100");
  std::cout << "  per-plane chroma deltas\n";
}

// ---------------------------------------------------------------------------
// Tile edges.
// ---------------------------------------------------------------------------
void test_tile_edges() {
  // 64x64 luma, 16x16 tip; vertical tile starts at luma x=32 (MI col 8).
  const auto make_vertical = []() {
    TipTestFrame frame = make_frame(16, 16);
    fill_vertical_step(frame, 32, 100, 120);
    frame.tile_cols = {0, 8};
    return frame;
  };
  {
    TipTestFrame frame = make_vertical();
    TipTestFrame before = frame;
    TipDeblockOptions options = base_options();
    CHECK(run(frame, options) == TipDeblockStatus::kOk, "tile run");
    CHECK(count_changed(before, frame) != 0,
          "vertical tile edge is filtered when disabling is off");
    CHECK(pixel(frame, 0, 31, 5) > 100 && pixel(frame, 0, 32, 5) < 120,
          "samples on both sides of the tile edge move");
  }
  {
    TipTestFrame frame = make_vertical();
    TipTestFrame before = frame;
    TipDeblockOptions options = base_options();
    options.disable_loopfilters_across_tiles = true;
    CHECK(run(frame, options) == TipDeblockStatus::kOk, "tile run");
    CHECK(count_changed(before, frame) == 0,
          "vertical tile edge is skipped when disabling is on");
    const TipDeblockEdgeInfo info =
        av2enc::tip_deblock::tip_deblock_inspect_edge(frame.view(), options, 0,
                                                      0, 32, 0);
    CHECK(info.tile_vert_edge == 1 && info.tile_skipped == 1 &&
              info.apply_filter == 0,
          "inspect reports the skipped vertical tile edge");
  }

  // Horizontal tile at luma y=32 (MI row 8), 16x16 tip.
  const auto make_horizontal = []() {
    TipTestFrame frame = make_frame(16, 16);
    fill_horizontal_step(frame, 32, 100, 120);
    frame.tile_rows = {0, 8};
    return frame;
  };
  {
    TipTestFrame frame = make_horizontal();
    TipTestFrame before = frame;
    CHECK(run(frame, base_options()) == TipDeblockStatus::kOk, "tile run");
    CHECK(count_changed(before, frame) != 0,
          "horizontal tile edge is filtered when disabling is off");
  }
  {
    TipTestFrame frame = make_horizontal();
    TipTestFrame before = frame;
    TipDeblockOptions options = base_options();
    options.disable_loopfilters_across_tiles = true;
    CHECK(run(frame, options) == TipDeblockStatus::kOk, "tile run");
    CHECK(count_changed(before, frame) == 0,
          "horizontal tile edge is skipped when disabling is on");
  }

  // Chroma vertical tile edge at chroma x=16 (tile MI col 8).
  {
    TipTestFrame frame = make_frame(16, 16);
    fill_vertical_step(frame, 32, 100, 120);
    frame.tile_cols = {0, 8};
    TipDeblockOptions options = base_options();
    const TipDeblockEdgeInfo info =
        av2enc::tip_deblock::tip_deblock_inspect_edge(frame.view(), options, 1,
                                                      0, 16, 0);
    CHECK(info.tile_vert_edge == 1 && info.sb_edge == 1 &&
              info.max_width_neg == 2,
          "chroma 16x16 tile edge gets the 2-sample neg clamp");
    TipTestFrame before = frame;
    options.disable_loopfilters_across_tiles = true;
    CHECK(run(frame, options) == TipDeblockStatus::kOk, "run");
    CHECK(count_changed(before, frame) == 0,
          "chroma tile edge skipped as well");
  }
  std::cout << "  tile edges (vertical/horizontal, luma/chroma)\n";
}

void test_64_grid_clamp() {
  TipTestFrame frame = make_frame(40, 40);  // 160x160 luma, 80x80 chroma
  TipDeblockOptions tip16 = base_options();
  TipDeblockFrame view = frame.view();

  const TipDeblockEdgeInfo y64 =
      av2enc::tip_deblock::tip_deblock_inspect_edge(view, tip16, 0, 1, 0, 64);
  const TipDeblockEdgeInfo y48 =
      av2enc::tip_deblock::tip_deblock_inspect_edge(view, tip16, 0, 1, 0, 48);
  CHECK(y64.sb_edge == 1 && y64.max_width_neg == 6, "luma 64-grid sbEdge");
  CHECK(y48.sb_edge == 0 && y48.max_width_neg == 6,
        "luma non-64 edge also reaches neg 6 at 16x16 tip");
  CHECK(y64.max_width_pos == 6, "luma pos width at 64-grid");

  const TipDeblockEdgeInfo u64 =
      av2enc::tip_deblock::tip_deblock_inspect_edge(view, tip16, 1, 1, 0, 64);
  const TipDeblockEdgeInfo u48 =
      av2enc::tip_deblock::tip_deblock_inspect_edge(view, tip16, 1, 1, 0, 48);
  CHECK(u64.sb_edge == 1 && u64.max_width_neg == 2,
        "chroma 64-grid neg clamp to 2");
  CHECK(u48.sb_edge == 0 && u48.max_width_neg == 3,
        "chroma non-64 edge neg 3");

  // A horizontal tile edge is not an sbEdge for the width clamp.
  frame.tile_rows = {0, 4};  // luma y=16 -> chroma y=8
  const TipDeblockEdgeInfo tile8 =
      av2enc::tip_deblock::tip_deblock_inspect_edge(frame.view(), tip16, 1, 1,
                                                    0, 8);
  CHECK(tile8.tile_horz_edge == 1 && tile8.sb_edge == 0 &&
            tile8.max_width_neg == 3,
        "horizontal tile edge does not clamp (only 64-grid does)");

  // Filtering across the 64 grid uses the wider negative reach.
  TipTestFrame filtered = make_frame(40, 40);
  fill_horizontal_step(filtered, 64, 100, 120);
  TipTestFrame before = filtered;
  CHECK(run(filtered, tip16) == TipDeblockStatus::kOk, "64-grid run");
  CHECK(pixel(filtered, 0, 6, 64) != pixel(before, 0, 6, 64),
        "luma row 58 (64-6) can be modified by the 64-grid edge");
  CHECK(pixel(filtered, 0, 6, 57) == pixel(before, 0, 6, 57),
        "row 57 is outside the 6-sample reach");
  std::cout << "  horizontal 64-grid sbEdge clamp\n";
}

// ---------------------------------------------------------------------------
// Flag interaction and invocation.
// ---------------------------------------------------------------------------
void test_flag_interaction() {
  const auto make_step = []() {
    TipTestFrame frame = make_frame(4, 4);
    fill_vertical_step(frame, 8, 100, 120);
    return frame;
  };

  // Full matrix of (mode, apply, enable, allow).
  const TipFrameMode modes[] = {TipFrameMode::kDisabled,
                                TipFrameMode::kAsReference,
                                TipFrameMode::kAsOutput};
  for (TipFrameMode mode : modes) {
    for (int apply = 0; apply <= 1; ++apply) {
      for (int enable = 0; enable <= 1; ++enable) {
        for (int allow = 0; allow <= 1; ++allow) {
          TipDeblockOptions options = tip8_options();
          options.tip_frame_mode = mode;
          options.apply_deblocking_filter_tip = apply != 0;
          options.enable_df_sub_pu = enable != 0;
          options.allow_df_sub_pu = allow != 0;

          const bool signallable =
              mode == TipFrameMode::kAsOutput && enable != 0;
          const bool codable = signallable && allow != 0;
          const bool active = codable && apply != 0;
          CHECK(av2enc::tip_deblock::tip_deblock_flag_signallable(options) ==
                    signallable,
                "flag signallability");
          CHECK(av2enc::tip_deblock::tip_deblock_is_active(options) == active,
                "flag activity");

          TipTestFrame frame = make_step();
          TipTestFrame before = frame;
          const TipDeblockStatus status = run(frame, options);
          if (apply != 0 && !codable) {
            CHECK(status == TipDeblockStatus::kUnsupportedFlagCombination,
                  "apply=1 without the df_sub_pu syntax is rejected");
            CHECK(count_changed(before, frame) == 0,
                  "rejected invocation changes no pixels");
          } else if (!active) {
            CHECK(status == TipDeblockStatus::kNotApplicable,
                  "inactive invocation is not applicable");
            CHECK(count_changed(before, frame) == 0,
                  "inactive invocation changes no pixels");
          } else {
            CHECK(status == TipDeblockStatus::kOk, "active invocation runs");
            CHECK(count_changed(before, frame) != 0,
                  "active invocation filters the step");
          }
        }
      }
    }
  }

  // plane_pass follows the same gate.
  {
    TipTestFrame frame = make_step();
    TipDeblockOptions options = tip8_options();
    options.apply_deblocking_filter_tip = false;
    TipDeblockFrame view = frame.view();
    CHECK(av2enc::tip_deblock::tip_deblock_plane_pass(view, options, 0, 0) ==
              TipDeblockStatus::kNotApplicable,
          "plane_pass respects the frame flag");
    options.apply_deblocking_filter_tip = true;
    options.tip_frame_mode = TipFrameMode::kAsReference;
    CHECK(av2enc::tip_deblock::tip_deblock_plane_pass(view, options, 0, 0) ==
              TipDeblockStatus::kUnsupportedFlagCombination,
          "plane_pass rejects non-TIP frames with apply=1");
  }

  // Unsupported / invalid geometry.
  {
    TipTestFrame frame = make_step();
    TipDeblockOptions options = tip8_options();
    options.bit_depth = 10;
    TipDeblockFrame view = frame.view();
    CHECK(av2enc::tip_deblock::tip_deblock_frame(view, options) ==
              TipDeblockStatus::kUnsupportedBitDepth,
          "10-bit rejected");
    options = tip8_options();
    frame.mi_cols = 3;  // odd
    TipDeblockFrame odd = frame.view();
    CHECK(av2enc::tip_deblock::tip_deblock_frame(odd, options) ==
              TipDeblockStatus::kInvalidArgument,
          "odd MI grid rejected");
  }
  std::cout << "  enable_df_sub_pu / allow_df_sub_pu / apply flag matrix\n";
}

// ---------------------------------------------------------------------------
// Content that must not change.
// ---------------------------------------------------------------------------
void test_flat_and_checker() {
  for (int tip8 = 0; tip8 <= 1; ++tip8) {
    TipTestFrame flat = make_frame(8, 8);
    fill_flat(flat, 128);
    TipTestFrame before = flat;
    CHECK(run(flat, tip8 ? tip8_options() : base_options()) ==
              TipDeblockStatus::kOk,
          "flat run");
    CHECK(count_changed(before, flat) == 0,
          "flat content is unchanged by TIP deblocking");

    TipTestFrame checker = make_frame(8, 8);
    fill_checker(checker);
    before = checker;
    CHECK(run(checker, tip8 ? tip8_options() : base_options()) ==
              TipDeblockStatus::kOk,
          "checker run");
    CHECK(count_changed(before, checker) == 0,
          "alternating texture is rejected by the side mask");
  }

  // q=0: qThr = 0 and side = 0, so no edge is filtered.
  TipTestFrame frame = make_frame(4, 4);
  fill_vertical_step(frame, 8, 100, 120);
  TipTestFrame before = frame;
  TipDeblockOptions options = tip8_options();
  options.base_q_idx = 0;
  CHECK(run(frame, options) == TipDeblockStatus::kOk, "q=0 run");
  CHECK(count_changed(before, frame) == 0, "base_q_idx=0 filters nothing");
  std::cout << "  flat / checker / q=0\n";
}

// ---------------------------------------------------------------------------
// Randomized structural fuzz.
// ---------------------------------------------------------------------------
void test_randomized() {
  std::mt19937 rng(0x7F16u);
  for (int iteration = 0; iteration < 120; ++iteration) {
    const int mi_cols = 8 + 2 * static_cast<int>(rng() % 9);  // 8..24, even
    const int mi_rows = 8 + 2 * static_cast<int>(rng() % 9);
    TipTestFrame frame = make_frame(mi_cols, mi_rows);

    // Random content with a deliberate step on a visited lattice position.
    std::uint32_t state = 0x9E3779B9u ^ static_cast<std::uint32_t>(iteration);
    for (int plane = 0; plane < 3; ++plane) {
      fill_plane(frame, plane, [&state](int x, int y) {
        state = state * 1664525u + 1013904223u;
        const std::uint32_t px = static_cast<std::uint32_t>(x);
        const std::uint32_t py = static_cast<std::uint32_t>(y);
        return static_cast<int>(((state >> 24) ^ (px * 3u) ^ (py * 5u)) &
                                0xFFu);
      });
    }
    const bool tip8 = (rng() % 2) != 0;
    const int block = tip8 ? 8 : 16;
    const int luma_w = frame.luma_width();
    const int luma_h = frame.luma_height();
    if (luma_w > block) {
      fill_vertical_step(frame, block * (1 + static_cast<int>(rng() % 2)),
                         64, 192);
    } else if (luma_h > block) {
      fill_horizontal_step(frame, block * (1 + static_cast<int>(rng() % 2)),
                           64, 192);
    }

    // Random tile starts.
    frame.tile_cols.clear();
    frame.tile_rows.clear();
    frame.tile_cols.push_back(0);
    frame.tile_rows.push_back(0);
    if ((rng() % 3) == 0) {
      frame.tile_cols.push_back(static_cast<std::uint16_t>(
          1 + rng() % static_cast<unsigned>(mi_cols - 1)));
    }
    if ((rng() % 3) == 0) {
      frame.tile_rows.push_back(static_cast<std::uint16_t>(
          1 + rng() % static_cast<unsigned>(mi_rows - 1)));
    }
    std::sort(frame.tile_cols.begin(), frame.tile_cols.end());
    std::sort(frame.tile_rows.begin(), frame.tile_rows.end());
    frame.tile_cols.erase(
        std::unique(frame.tile_cols.begin(), frame.tile_cols.end()),
        frame.tile_cols.end());
    frame.tile_rows.erase(
        std::unique(frame.tile_rows.begin(), frame.tile_rows.end()),
        frame.tile_rows.end());

    TipDeblockOptions options = tip8 ? tip8_options() : base_options();
    options.base_q_idx = static_cast<int>(rng() % 256);
    options.delta_q_u_ac = static_cast<int>(rng() % 65) - 32;
    options.delta_q_v_ac = static_cast<int>(rng() % 65) - 32;
    options.base_uv_ac_delta_q = static_cast<int>(rng() % 33) - 16;
    options.disable_loopfilters_across_tiles = (rng() % 2) != 0;

    TipTestFrame before = frame;
    WriteMask mask;
    build_write_mask(frame, options, &mask);

    CHECK(run(frame, options) == TipDeblockStatus::kOk, "fuzz run");
    CHECK(guard_intact(frame), "fuzz guard samples intact");

    int plane = 0;
    int px = 0;
    int py = 0;
    CHECK(!changed_pixels_outside_mask(before, frame, mask, &plane, &px, &py),
          "fuzz changes stay inside the declared filter envelope");

    // Determinism: a second identical run produces the same samples.
    TipTestFrame second = before;
    CHECK(run(second, options) == TipDeblockStatus::kOk, "fuzz rerun");
    CHECK(checksum(second) == checksum(frame), "fuzz run is deterministic");
  }
  std::cout << "  randomized fuzz (120 frames)\n";
}

// ---------------------------------------------------------------------------
// Real reconstructed frame.
// ---------------------------------------------------------------------------
bool load_y4m_first_frame(const std::string& path, TipTestFrame* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::string line;
  if (!std::getline(in, line) || line.rfind("YUV4MPEG2", 0) != 0) {
    return false;
  }
  std::istringstream header(line);
  std::string token;
  int width = 0;
  int height = 0;
  bool c420 = false;
  while (header >> token) {
    if (token.size() > 1) {
      const char tag = token[0];
      const std::string value = token.substr(1);
      if (tag == 'W') width = std::stoi(value);
      if (tag == 'H') height = std::stoi(value);
      if (tag == 'C' && value == "420") c420 = true;
    }
  }
  if (width <= 0 || height <= 0 || !c420 || (width % 8) != 0 ||
      (height % 8) != 0) {
    return false;
  }
  if (!std::getline(in, line) || line.rfind("FRAME", 0) != 0) {
    return false;
  }
  const int mi_cols = width / 4;
  const int mi_rows = height / 4;
  TipTestFrame frame = make_frame(mi_cols, mi_rows);
  const int chroma_w = width / 2;
  const int chroma_h = height / 2;
  std::vector<std::uint8_t> row(static_cast<std::size_t>(width));
  for (int y = 0; y < height; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), width);
    if (!in) return false;
    std::memcpy(frame.data[0].data() +
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(frame.stride[0]),
                row.data(), static_cast<std::size_t>(width));
  }
  row.assign(static_cast<std::size_t>(chroma_w), 0);
  for (int y = 0; y < chroma_h; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), chroma_w);
    if (!in) return false;
    std::memcpy(frame.data[1].data() +
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(frame.stride[1]),
                row.data(), static_cast<std::size_t>(chroma_w));
  }
  for (int y = 0; y < chroma_h; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), chroma_w);
    if (!in) return false;
    std::memcpy(frame.data[2].data() +
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(frame.stride[2]),
                row.data(), static_cast<std::size_t>(chroma_w));
  }
  *out = std::move(frame);
  return true;
}

void test_real_frame(const std::string& repo_root) {
  const std::string path = repo_root + "/decoded.y4m";
  TipTestFrame frame;
  if (!load_y4m_first_frame(path, &frame)) {
    std::cout << "  real frame: skipped (decoded.y4m not usable)\n";
    return;
  }
  if ((frame.mi_cols & 1) != 0 || (frame.mi_rows & 1) != 0) {
    std::cout << "  real frame: skipped (odd MI grid)\n";
    return;
  }
  const std::size_t baseline = checksum(frame);
  for (int tip8 = 0; tip8 <= 1; ++tip8) {
    TipTestFrame run_frame = frame;
    TipTestFrame before = run_frame;
    TipDeblockOptions options = tip8 ? tip8_options() : base_options();
    options.base_q_idx = 128;
    CHECK(run(run_frame, options) == TipDeblockStatus::kOk,
          "real frame run");
    CHECK(guard_intact(run_frame), "real frame guards intact");
    CHECK(checksum(run_frame) != baseline, "real frame is modified");
    CHECK(count_changed(before, run_frame) > 0, "real frame has filtered samples");
    TipTestFrame rerun = before;
    CHECK(run(rerun, options) == TipDeblockStatus::kOk, "real frame rerun");
    CHECK(checksum(rerun) == checksum(run_frame),
          "real frame run is deterministic");
    std::cout << "  real frame " << (tip8 ? "8x8" : "16x16") << ": "
              << count_changed(before, run_frame) << " samples changed, max |d| "
              << max_abs_delta(before, run_frame) << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string repo_root;
  if (argc > 1) {
    repo_root = argv[1];
  } else {
    repo_root = ".";
  }

  std::cout << "AV2 §7.16 TIP deblocking tests\n";
  test_tip_size_selection();
  test_strength_derivation();
  test_strength_all_q();
  test_max_width_classes();
  test_exact_luma8();
  test_exact_luma16();
  test_exact_chroma();
  test_unvisited_edges();
  test_plane_deltas();
  test_tile_edges();
  test_64_grid_clamp();
  test_flag_interaction();
  test_flat_and_checker();
  test_randomized();
  test_real_frame(repo_root);

  std::cout << "\nchecks: " << g_checks << "  failures: " << g_failures
            << "\n";
  if (g_failures != 0) {
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
