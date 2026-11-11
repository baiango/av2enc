// CPU reference tests for the AV2 inter motion-compensation component.
//
// Build/run: ./build_and_test.sh
//
// Coverage:
//   * generated subpel table sanity (sums, delta phase, even coefficients)
//   * integer-MV copy equivalence for every filter and pattern
//   * CPU two-pass reference vs the per-sample core used by the CUDA kernel
//   * guard bands: no reads/writes outside the block or MI-aligned plane
//   * partial (edge) blocks and large out-of-picture MVs
//   * 4:2:0 chroma (2x2 minimum) and scaled references
//   * spatial MV stack scan order, weights, dedup, sorting, clamping
//   * skip/merge, NEARMV, NEWMV and ME-absolute MV resolution
//   * real 720p frames with a known integer shift + CPU oracle timing

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "encoder/inter/mc/mc.h"
#include "encoder/inter/mc/mc_test_common.h"

namespace mc = av2enc::inter::mc;
using namespace av2enc::inter::mc::test;

namespace {

struct Test {
  int checks = 0;
  int failures = 0;
  void expect(bool cond, const std::string& what) {
    ++checks;
    if (!cond) {
      ++failures;
      if (failures <= 20) {
        std::printf("FAIL: %s\n", what.c_str());
      }
    }
  }
  void eq_int(long long got, long long want, const std::string& what) {
    ++checks;
    if (got != want) {
      ++failures;
      if (failures <= 20) {
        std::printf("FAIL: %s (got %lld, want %lld)\n", what.c_str(), got,
                    want);
      }
    }
  }
};

const char* filter_name(mc::InterpFilter f) {
  switch (f) {
    case mc::InterpFilter::kEighttap:
      return "eighttap";
    case mc::InterpFilter::kEighttapSmooth:
      return "smooth";
    case mc::InterpFilter::kEighttapSharp:
      return "sharp";
    case mc::InterpFilter::kBilinear:
      return "bilinear";
    case mc::InterpFilter::kFourtap:
      return "fourtap";
    case mc::InterpFilter::kFourtapSmooth:
      return "fourtap_smooth";
  }
  return "?";
}

std::string label(const std::string& base, int a, int b) {
  return base + " [" + std::to_string(a) + "," + std::to_string(b) + "]";
}

// Per-output-sample evaluation of the shared core (the form used by CUDA).
mc::McStatus predict_with_core(const mc::RefFrame& ref, int plane,
                               const mc::BlockRect& rect, mc::Mv mv,
                               mc::InterpFilter filter, std::uint8_t* dst,
                               int dst_stride) {
  const int sub_x = plane > 0 ? 1 : 0;
  const int sub_y = plane > 0 ? 1 : 0;
  const mc::McSampleLocation loc = mc::mc_derive_sample_location(
      rect.x, rect.y, mv, sub_x, sub_y);
  const int last_x = ref.plane_width(plane) - 1;
  const int last_y = ref.plane_height(plane) - 1;
  const mc::InterpFilter fh = mc::mc_pass_filter(filter, rect.w <= 4);
  const mc::InterpFilter fv = mc::mc_pass_filter(filter, rect.h <= 4);
  const std::int16_t* taps_h = &mc::kSubpelFilters[static_cast<int>(fh)][0][0];
  const std::int16_t* taps_v = &mc::kSubpelFilters[static_cast<int>(fv)][0][0];
  for (int r = 0; r < rect.h; ++r) {
    for (int c = 0; c < rect.w; ++c) {
      dst[static_cast<std::size_t>(r) * static_cast<std::size_t>(dst_stride) +
          static_cast<std::size_t>(c)] =
          mc::mc_predict_sample(ref.plane[plane].data, ref.plane[plane].stride,
                                0, 0, last_x, last_y, loc.start_x, loc.start_y,
                                loc.step_x, loc.step_y, taps_h, taps_v, r, c);
    }
  }
  return mc::McStatus::kOk;
}

void test_tables(Test& t) {
  for (int f = 0; f < mc::kSubpelFilterTypes; ++f) {
    for (int p = 0; p < mc::kSubpelPhases; ++p) {
      int sum = 0;
      for (int k = 0; k < mc::kSubpelTaps; ++k) {
        const int v = mc::kSubpelFilters[f][p][k];
        sum += v;
        t.expect((v & 1) == 0, label("filter coefficient is even", f, p));
        t.expect(v >= -128 && v <= 128, label("filter coefficient range", f, p));
      }
      t.eq_int(sum, 128, label("filter phase sum", f, p));
    }
    // Phase 0 is a pure sample copy for every filter.
    t.eq_int(mc::kSubpelFilters[f][0][3], 128, label("phase0 centre", f, 0));
    for (int k = 0; k < 8; ++k) {
      if (k != 3) {
        t.eq_int(mc::kSubpelFilters[f][0][k], 0, label("phase0 tap", f, k));
      }
    }
  }

  // Spot-check rows from the spec PDF to catch a reversed or shifted
  // extraction.  These are the phase 1 / 8 / 15 rows of each filter.
  const std::int16_t expected[6][3][8] = {
      {{0, 2, -6, 126, 8, -2, 0, 0},
       {0, 2, -14, 76, 76, -14, 2, 0},
       {0, 0, -2, 8, 126, -6, 2, 0}},
      {{0, 2, 28, 62, 34, 2, 0, 0},
       {0, -2, 14, 52, 52, 14, -2, 0},
       {0, 0, 2, 34, 62, 28, 2, 0}},
      {{-2, 2, -6, 126, 8, -2, 2, 0},
       {-4, 12, -24, 80, 80, -24, 12, -4},
       {0, 2, -2, 8, 126, -6, 2, -2}},
      {{0, 0, 0, 120, 8, 0, 0, 0},
       {0, 0, 0, 64, 64, 0, 0, 0},
       {0, 0, 0, 8, 120, 0, 0, 0}},
      {{0, 0, -4, 126, 8, -2, 0, 0},
       {0, 0, -12, 76, 76, -12, 0, 0},
       {0, 0, -2, 8, 126, -4, 0, 0}},
      {{0, 0, 30, 62, 34, 2, 0, 0},
       {0, 0, 12, 52, 52, 12, 0, 0},
       {0, 0, 2, 34, 62, 30, 0, 0}},
  };
  const int phases[3] = {1, 8, 15};
  for (int f = 0; f < 6; ++f) {
    for (int i = 0; i < 3; ++i) {
      for (int k = 0; k < 8; ++k) {
        t.eq_int(mc::kSubpelFilters[f][phases[i]][k], expected[f][i][k],
                 "spec table row");
      }
    }
  }
}

void test_integer_copy(Test& t) {
  TestFrame frame = TestFrame::make(12, 10);
  const Pattern patterns[] = {Pattern::kFlat, Pattern::kRampX, Pattern::kRampY,
                              Pattern::kChecker, Pattern::kNoise,
                              Pattern::kSoftGradient, Pattern::kVerticalEdge};
  const mc::Mv mvs[] = {{0, 0}, {24, -16}, {-40, 32}};
  const mc::InterpFilter filters[] = {
      mc::InterpFilter::kEighttap,      mc::InterpFilter::kEighttapSmooth,
      mc::InterpFilter::kEighttapSharp, mc::InterpFilter::kBilinear};

  for (Pattern pattern : patterns) {
    frame.fill(pattern);
    for (mc::InterpFilter filter : filters) {
      for (mc::Mv mv : mvs) {
        const mc::BlockRect rect{8, 8, 16, 16};
        DestBuf dst;
        dst.init(rect.w, rect.h);
        const mc::McStatus st =
            mc::predict_plane(frame.ref(), 0, rect, mv, filter, dst.ptr(),
                              dst.stride);
        t.expect(st == mc::McStatus::kOk, "integer copy status");
        bool same = true;
        const int oy = mv.y >> 3;
        const int ox = mv.x >> 3;
        for (int y = 0; y < rect.h; ++y) {
          for (int x = 0; x < rect.w; ++x) {
            if (dst.at(y, x) != frame.plane[0].at(rect.y + oy + y,
                                                 rect.x + ox + x)) {
              same = false;
            }
          }
        }
        t.expect(same, std::string("integer MV copy ") +
                           filter_name(filter) + " mv=(" +
                           std::to_string(mv.y) + "," + std::to_string(mv.x) +
                           ")");
        t.expect(dst.guards_intact(), "integer copy dst guards");
      }
    }
  }
}

void test_core_equivalence(Test& t) {
  TestFrame frame = TestFrame::make(24, 20);
  frame.fill(Pattern::kNoise);
  const mc::BlockRect rects[] = {
      {0, 0, 4, 4},   {4, 0, 4, 8},    {12, 8, 8, 4},  {8, 4, 8, 8},
      {0, 0, 16, 16}, {16, 8, 16, 32}, {8, 16, 32, 16}, {0, 0, 32, 32},
      {8, 0, 64, 16}, {0, 0, 16, 64},  {0, 0, 64, 64}};
  const mc::Mv mvs[] = {{0, 0},    {1, 0},   {0, 1},  {3, -5}, {7, 7},
                        {-8, 13}, {-1, -1}, {15, 9}, {-15, -9}};
  const mc::InterpFilter filters[] = {
      mc::InterpFilter::kEighttap,       mc::InterpFilter::kEighttapSmooth,
      mc::InterpFilter::kEighttapSharp,  mc::InterpFilter::kBilinear};

  for (const mc::BlockRect& rect : rects) {
    // Keep the largest rectangles inside the 96x80 luma frame.
    if (rect.x + rect.w > 96 || rect.y + rect.h > 80) {
      continue;
    }
    for (mc::InterpFilter filter : filters) {
      for (mc::Mv mv : mvs) {
        DestBuf a;
        DestBuf b;
        a.init(rect.w, rect.h);
        b.init(rect.w, rect.h);
        const mc::McStatus s1 =
            mc::predict_plane(frame.ref(), 0, rect, mv, filter, a.ptr(),
                              a.stride);
        predict_with_core(frame.ref(), 0, rect, mv, filter, b.ptr(), b.stride);
        t.expect(s1 == mc::McStatus::kOk, "core equivalence status");
        bool same = true;
        for (int y = 0; y < rect.h && same; ++y) {
          for (int x = 0; x < rect.w; ++x) {
            if (a.at(y, x) != b.at(y, x)) {
              same = false;
              break;
            }
          }
        }
        t.expect(same, "CPU reference == per-sample core " +
                           std::string(filter_name(filter)) + " rect " +
                           std::to_string(rect.w) + "x" +
                           std::to_string(rect.h) + " mv(" +
                           std::to_string(mv.y) + "," + std::to_string(mv.x) +
                           ")");
        t.expect(a.guards_intact() && b.guards_intact(),
                 "core equivalence guards");
      }
    }
  }
}

void test_bilinear_math(Test& t) {
  // Phase 8 with the bilinear filter: 64/64 average, then vertical phase 0.
  TestFrame frame = TestFrame::make(8, 8);
  for (int y = 0; y < frame.plane[0].height; ++y) {
    for (int x = 0; x < frame.plane[0].width; ++x) {
      frame.plane[0].at(y, x) = static_cast<std::uint8_t>((x * 37 + y * 11) & 0xFF);
    }
  }
  const mc::BlockRect rect{0, 0, 8, 8};
  DestBuf dst;
  dst.init(rect.w, rect.h);
  const mc::McStatus st = mc::predict_plane(
      frame.ref(), 0, rect, mc::Mv{0, 4}, mc::InterpFilter::kBilinear,
      dst.ptr(), dst.stride);
  t.expect(st == mc::McStatus::kOk, "bilinear status");
  for (int y = 0; y < rect.h; ++y) {
    for (int x = 0; x < rect.w; ++x) {
      const int a = frame.plane[0].at(y, x);
      const int b = frame.plane[0].at(y, x + 1);
      const int intermediate = (64 * (a + b) + 4) >> 3;
      const int want = (intermediate + 8) >> 4;
      t.eq_int(dst.at(y, x), want, "bilinear 64/64 rounding");
    }
  }
}

void test_partial_blocks(Test& t) {
  TestFrame frame = TestFrame::make(8, 6);  // 32x24 luma, 16x12 chroma
  frame.fill(Pattern::kSoftGradient);
  const mc::BlockRect rects[] = {
      {28, 0, 4, 8},   {0, 20, 8, 4},   {28, 20, 4, 4},
      {24, 16, 8, 8},  {16, 16, 16, 8}, {0, 0, 32, 24}};
  const mc::Mv mvs[] = {{0, 0}, {40, 40}, {-40, -40}, {33, -37}};
  for (const mc::BlockRect& rect : rects) {
    for (mc::Mv mv : mvs) {
      DestBuf dst;
      dst.init(rect.w, rect.h);
      const mc::McStatus st = mc::predict_plane(
          frame.ref(), 0, rect, mv, mc::InterpFilter::kEighttap, dst.ptr(),
          dst.stride);
      t.expect(st == mc::McStatus::kOk, "partial block status");
      bool in_range = true;
      for (int y = 0; y < rect.h; ++y) {
        for (int x = 0; x < rect.w; ++x) {
          const int v = dst.at(y, x);
          if (v < 0 || v > 255) {
            in_range = false;
          }
        }
      }
      t.expect(in_range, "partial block sample range");
      t.expect(dst.guards_intact(), "partial block guards");
      DestBuf core;
      core.init(rect.w, rect.h);
      predict_with_core(frame.ref(), 0, rect, mv,
                        mc::InterpFilter::kEighttap, core.ptr(), core.stride);
      bool same = true;
      for (int y = 0; y < rect.h && same; ++y) {
        for (int x = 0; x < rect.w; ++x) {
          if (dst.at(y, x) != core.at(y, x)) {
            same = false;
            break;
          }
        }
      }
      t.expect(same, "partial block == core");
    }
  }
}

void test_chroma(Test& t) {
  TestFrame frame = TestFrame::make(16, 12);  // 64x48
  frame.fill(Pattern::kRampX);
  const mc::BlockRect luma_rects[] = {
      {0, 0, 8, 8}, {8, 8, 4, 4}, {48, 32, 16, 16}, {56, 40, 8, 8}};
  const mc::Mv mvs[] = {{0, 0}, {3, 5}, {-7, 11}};
  for (const mc::BlockRect& rect : luma_rects) {
    for (mc::Mv mv : mvs) {
      DestBuf y;
      DestBuf u;
      DestBuf v;
      y.init(rect.w, rect.h);
      u.init(rect.w / 2, rect.h / 2);
      v.init(rect.w / 2, rect.h / 2);
      mc::MutablePlane dst[3] = {{y.ptr(), y.stride},
                                 {u.ptr(), u.stride},
                                 {v.ptr(), v.stride}};
      const mc::McStatus st = mc::predict_block_420(
          frame.ref(), rect, mv, mc::InterpFilter::kEighttapSmooth, dst);
      t.expect(st == mc::McStatus::kOk, "420 status");
      const mc::BlockRect cr{rect.x >> 1, rect.y >> 1, rect.w >> 1,
                             rect.h >> 1};
      DestBuf u2;
      DestBuf v2;
      u2.init(cr.w, cr.h);
      v2.init(cr.w, cr.h);
      mc::predict_plane(frame.ref(), 1, cr, mv,
                        mc::InterpFilter::kEighttapSmooth, u2.ptr(), u2.stride);
      mc::predict_plane(frame.ref(), 2, cr, mv,
                        mc::InterpFilter::kEighttapSmooth, v2.ptr(), v2.stride);
      bool same = true;
      for (int yy = 0; yy < cr.h && same; ++yy) {
        for (int xx = 0; xx < cr.w; ++xx) {
          if (u.at(yy, xx) != u2.at(yy, xx) ||
              v.at(yy, xx) != v2.at(yy, xx)) {
            same = false;
            break;
          }
        }
      }
      t.expect(same, "420 wrapper == per-plane calls");
      t.expect(u.guards_intact() && v.guards_intact(), "420 chroma guards");
    }
  }
  // 4x4 luma -> 2x2 chroma exercises the four-tap downgrade path.
  const mc::BlockRect rect{12, 8, 4, 4};
  DestBuf y;
  DestBuf u;
  DestBuf v;
  y.init(4, 4);
  u.init(2, 2);
  v.init(2, 2);
  mc::MutablePlane dst[3] = {{y.ptr(), y.stride},
                             {u.ptr(), u.stride},
                             {v.ptr(), v.stride}};
  t.expect(mc::predict_block_420(frame.ref(), rect, mc::Mv{1, 2},
                                 mc::InterpFilter::kEighttap, dst) ==
               mc::McStatus::kOk,
           "2x2 chroma status");
  // Odd luma geometry is rejected by the 4:2:0 wrapper.
  const mc::BlockRect odd{1, 0, 4, 4};
  t.expect(mc::predict_block_420(frame.ref(), odd, mc::Mv{}, 
                                 mc::InterpFilter::kEighttap, dst) ==
               mc::McStatus::kUnsupportedSubsampling,
           "odd 420 geometry rejected");
}

// ---------------------------------------------------------------------------
// MV stack
// ---------------------------------------------------------------------------

void set_block(std::vector<mc::MiInfo>& grid, int mi_cols, int row, int col,
               int w4, int h4, mc::Mv mv, std::int8_t ref = mc::kRefLast) {
  for (int r = row; r < row + h4; ++r) {
    for (int c = col; c < col + w4; ++c) {
      mc::MiInfo m;
      m.ref[0] = ref;
      m.ref[1] = mc::kRefNone;
      m.mv[0][0] = mv.y;
      m.mv[0][1] = mv.x;
      m.is_inter = 1;
      m.w4 = static_cast<std::int8_t>(w4);
      m.h4 = static_cast<std::int8_t>(h4);
      m.col_base = static_cast<std::int16_t>(col);
      m.row_base = static_cast<std::int16_t>(row);
      grid[static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols) +
           static_cast<std::size_t>(c)] = m;
    }
  }
}

struct StackFixture {
  std::vector<mc::MiInfo> grid;
  mc::MvStackParams params;
  mc::MvStack stack;
};

StackFixture make_stack_fixture() {
  StackFixture f;
  f.grid.assign(16 * 16, mc::MiInfo{});
  // Left, above-left, above, above-right, below-left and far-left blocks.
  set_block(f.grid, 16, 4, 2, 2, 4, mc::Mv{8, 0});    // A, left
  set_block(f.grid, 16, 0, 4, 4, 4, mc::Mv{0, 8});    // B, above
  set_block(f.grid, 16, 0, 2, 2, 4, mc::Mv{16, 0});   // C, above-left
  set_block(f.grid, 16, 8, 2, 2, 4, mc::Mv{-8, 0});   // D, below-left
  set_block(f.grid, 16, 0, 8, 4, 4, mc::Mv{0, -8});   // E, above-right
  set_block(f.grid, 16, 4, 0, 2, 8, mc::Mv{0, 16});   // F, far left
  f.params.mi_row = 4;
  f.params.mi_col = 4;
  f.params.bw4 = 4;
  f.params.bh4 = 4;
  f.params.sb_h4 = 16;
  f.params.mi_cols = 16;
  f.params.mi_rows = 16;
  f.params.mi = f.grid.data();
  f.params.ref_frame = mc::kRefLast;
  f.params.global_mv = mc::Mv{0, 0};
  return f;
}

void test_mv_stack(Test& t) {
  StackFixture f = make_stack_fixture();
  t.expect(mc::build_mv_stack(f.params, &f.stack) == mc::McStatus::kOk,
           "stack build status");
  t.eq_int(f.stack.count, 6, "stack count");
  t.eq_int(f.stack.num_nearest, 5, "stack num_nearest");

  // Order after weight sorting: A(2), B(2), D(1), E(1), C(0), then F.
  const mc::Mv expect[6] = {{8, 0},  {0, 8},  {-8, 0},
                            {0, -8}, {16, 0}, {0, 16}};
  const int expect_weight[6] = {2, 2, 1, 1, 0, 0};
  for (int i = 0; i < 6; ++i) {
    t.expect(f.stack.cand[i].mv[0] == expect[i],
             label("stack candidate", i, f.stack.cand[i].mv[0].y));
    t.eq_int(f.stack.cand[i].weight, expect_weight[i],
             label("stack weight", i, 0));
  }
  t.eq_int(f.stack.cand[0].row_offset, 3, "A row offset");
  t.eq_int(f.stack.cand[0].col_offset, -1, "A col offset");
  t.eq_int(f.stack.cand[5].row_offset, 3, "F row offset");
  t.eq_int(f.stack.cand[5].col_offset, -3, "F col offset");

  // Top-left corner: no spatial neighbours, so only the global MV remains.
  StackFixture corner;
  corner.grid.assign(16 * 16, mc::MiInfo{});
  corner.params = f.params;
  corner.params.mi_row = 0;
  corner.params.mi_col = 0;
  corner.params.mi = corner.grid.data();
  corner.params.global_mv = mc::Mv{4, -4};
  t.expect(mc::build_mv_stack(corner.params, &corner.stack) ==
               mc::McStatus::kOk,
           "corner stack status");
  t.eq_int(corner.stack.count, 1, "corner stack uses global MV");
  t.expect(corner.stack.cand[0].mv[0] == mc::Mv{4, -4},
           "corner global MV value");

  // A neighbour equal to the global MV must not be duplicated.
  StackFixture dup;
  dup.grid.assign(16 * 16, mc::MiInfo{});
  set_block(dup.grid, 16, 4, 2, 2, 4, mc::Mv{4, -4});
  dup.params = f.params;
  dup.params.mi = dup.grid.data();
  dup.params.global_mv = mc::Mv{4, -4};
  t.expect(mc::build_mv_stack(dup.params, &dup.stack) == mc::McStatus::kOk,
           "dup global status");
  t.eq_int(dup.stack.count, 1, "global MV deduplicated");

  // Picture-bound clamping (§5.20.9.4/5).
  StackFixture clamped;
  clamped.grid.assign(16 * 16, mc::MiInfo{});
  set_block(clamped.grid, 16, 4, 2, 2, 4, mc::Mv{30000, -30000});
  clamped.params = f.params;
  clamped.params.mi = clamped.grid.data();
  t.expect(mc::build_mv_stack(clamped.params, &clamped.stack) ==
               mc::McStatus::kOk,
           "clamped stack status");
  t.expect(clamped.stack.cand[0].mv[0] == mc::Mv{512, -384},
           "stack picture-bound clamp");

  // Validation.
  mc::MvStack bad;
  t.expect(mc::build_mv_stack(mc::MvStackParams{}, &bad) ==
               mc::McStatus::kInvalidArgument,
           "stack rejects null grid");
  mc::MvStackParams intra = f.params;
  intra.ref_frame = mc::kRefIntra;
  t.expect(mc::build_mv_stack(intra, &bad) == mc::McStatus::kInvalidArgument,
           "stack rejects intra ref");
  mc::MvStackParams intrabc = f.params;
  intrabc.use_intrabc = true;
  t.expect(mc::build_mv_stack(intrabc, &bad) ==
               mc::McStatus::kInvalidArgument,
           "stack rejects intrabc");
  mc::MvStackParams outside = f.params;
  outside.mi_col = 99;
  t.expect(mc::build_mv_stack(outside, &bad) ==
               mc::McStatus::kInvalidArgument,
           "stack rejects out-of-grid block");

  // Candidates using a different reference are not placed in this stack.
  StackFixture other;
  other.grid.assign(16 * 16, mc::MiInfo{});
  set_block(other.grid, 16, 4, 2, 2, 4, mc::Mv{8, 0}, mc::kRefGolden);
  other.params = f.params;
  other.params.mi = other.grid.data();
  other.params.global_mv = mc::Mv{};
  t.expect(mc::build_mv_stack(other.params, &other.stack) ==
               mc::McStatus::kOk,
           "other-ref stack status");
  t.eq_int(other.stack.count, 1, "other-ref candidate ignored");
}

void test_modes(Test& t) {
  StackFixture f = make_stack_fixture();
  mc::build_mv_stack(f.params, &f.stack);

  mc::ModeDecision d;
  mc::ResolvedMode r;

  d.mode = mc::InterMode::kSkipMerge;
  d.drl_index = 0;
  t.expect(mc::resolve_mode(f.stack, f.params, d, &r) == mc::McStatus::kOk,
           "skip resolve status");
  t.expect(r.mv == f.stack.cand[0].mv[0], "skip uses stack[0]");
  t.eq_int(r.mvd.y, 0, "skip mvd y");
  t.eq_int(r.mvd.x, 0, "skip mvd x");

  d.mode = mc::InterMode::kNearmv;
  d.drl_index = 2;
  t.expect(mc::resolve_mode(f.stack, f.params, d, &r) == mc::McStatus::kOk,
           "nearmv resolve status");
  t.expect(r.mv == f.stack.cand[2].mv[0], "nearmv uses stack[drl]");
  t.expect(r.pred_mv == f.stack.cand[2].mv[0], "nearmv predictor");

  d.mode = mc::InterMode::kNewmv;
  d.drl_index = 0;
  d.mvd = mc::Mv{3, -4};
  t.expect(mc::resolve_mode(f.stack, f.params, d, &r) == mc::McStatus::kOk,
           "newmv resolve status");
  t.expect(r.pred_mv == f.stack.cand[0].mv[0], "newmv predictor");
  t.eq_int(r.mv.y, f.stack.cand[0].mv[0].y + 3, "newmv mv y");
  t.eq_int(r.mv.x, f.stack.cand[0].mv[0].x - 4, "newmv mv x");
  t.eq_int(r.mvd.y, 3, "newmv mvd y");

  // ME-absolute hook.
  d.me_absolute = true;
  d.me_mv = mc::Mv{f.stack.cand[0].mv[0].y + 5,
                   f.stack.cand[0].mv[0].x + 6};
  t.expect(mc::resolve_mode(f.stack, f.params, d, &r) == mc::McStatus::kOk,
           "me-absolute status");
  t.eq_int(r.mvd.y, 5, "me-absolute mvd y");
  t.eq_int(r.mvd.x, 6, "me-absolute mvd x");
  t.expect(r.mv == d.me_mv, "me-absolute final mv");

  // Rejections.
  d.me_absolute = false;
  d.drl_index = 99;
  t.expect(mc::resolve_mode(f.stack, f.params, d, &r) ==
               mc::McStatus::kInvalidArgument,
           "drl out of range rejected");
  d.drl_index = 0;
  d.ref_frame = mc::kRefGolden;
  t.expect(mc::resolve_mode(f.stack, f.params, d, &r) ==
               mc::McStatus::kInvalidArgument,
           "ref mismatch rejected");

  // ME output is 1/16 luma; MC consumes the signaled 1/8 precision.
  t.expect(mc::mv_from_me_1_16(17, -17) == mc::Mv{9, -9},
           "1/16 -> 1/8 MV rounding");
  t.expect(mc::mv_from_me_1_16(16, -16) == mc::Mv{8, -8},
           "1/16 -> 1/8 exact");
  t.expect(mc::mv_from_me_1_16(0, 0) == mc::Mv{0, 0},
           "1/16 -> 1/8 zero");
}

void test_merge_selection(Test& t) {
  TestFrame frame = TestFrame::make(16, 12);
  frame.fill(Pattern::kRampY);
  const mc::BlockRect rect{8, 8, 16, 16};

  // Build a target frame shifted by (dy=3, dx=5) with edge replication.
  DestBuf target;
  target.init(rect.w, rect.h);
  for (int y = 0; y < rect.h; ++y) {
    for (int x = 0; x < rect.w; ++x) {
      const int sy = std::min(rect.y + y + 3, frame.plane[0].height - 1);
      const int sx = std::min(rect.x + x + 5, frame.plane[0].width - 1);
      target.data[static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(target.stride) +
                   static_cast<std::size_t>(x)] = frame.plane[0].at(sy, sx);
    }
  }

  mc::MvStack stack{};
  stack.count = 2;
  stack.num_nearest = 2;
  stack.cand[0].mv[0] = mc::Mv{0, 0};
  stack.cand[0].mv[1] = mc::Mv{0, 0};
  stack.cand[1].mv[0] = mc::Mv{24, 40};
  stack.cand[1].mv[1] = mc::Mv{24, 40};

  DestBuf scratch;
  scratch.init(rect.w, rect.h);
  mc::MergeScore score;
  t.expect(mc::select_merge_candidate(stack, frame.ref(), mc::kRefLast, rect,
                                      mc::InterpFilter::kEighttap,
                                      target.ptr(), target.stride,
                                      scratch.ptr(), scratch.stride,
                                      &score) == mc::McStatus::kOk,
           "merge selection status");
  t.eq_int(score.drl_index, 1, "merge picks the shifted candidate");
  t.eq_int(score.sad, 0, "merge candidate SAD");

  mc::MergeScore bad;
  t.expect(mc::select_merge_candidate(stack, frame.ref(), mc::kRefIntra, rect,
                                      mc::InterpFilter::kEighttap,
                                      target.ptr(), target.stride,
                                      scratch.ptr(), scratch.stride,
                                      &bad) == mc::McStatus::kInvalidArgument,
           "merge rejects intra ref");
}

void test_validation(Test& t) {
  TestFrame frame = TestFrame::make(4, 4);
  frame.fill(Pattern::kFlat);
  const mc::BlockRect rect{0, 0, 8, 8};
  DestBuf dst;
  dst.init(8, 8);
  t.expect(mc::predict_plane(frame.ref(), 3, rect, mc::Mv{},
                             mc::InterpFilter::kEighttap, dst.ptr(),
                             dst.stride) == mc::McStatus::kInvalidArgument,
           "invalid plane rejected");
  t.expect(mc::predict_plane(frame.ref(), 0, rect, mc::Mv{},
                             mc::InterpFilter::kEighttap, dst.ptr(),
                             dst.stride) == mc::McStatus::kOk,
           "valid prediction status");
  t.expect(mc::predict_plane(frame.ref(), 0, mc::BlockRect{12, 0, 8, 8},
                             mc::Mv{}, mc::InterpFilter::kEighttap, dst.ptr(),
                             dst.stride) == mc::McStatus::kInvalidArgument,
           "out-of-plane rect rejected");
  t.expect(mc::predict_plane(frame.ref(), 0, rect, mc::Mv{},
                             mc::InterpFilter::kEighttap, nullptr,
                             dst.stride) == mc::McStatus::kInvalidArgument,
           "null dst rejected");
  t.expect(mc::predict_plane(frame.ref(), 0, rect, mc::Mv{},
                             mc::InterpFilter::kEighttap, dst.ptr(), 2) ==
               mc::McStatus::kInvalidArgument,
           "short dst stride rejected");
  // 128x128 exceeds the M2 block budget and the intermediate buffer.
  TestFrame big = TestFrame::make(32, 32);
  big.fill(Pattern::kFlat);
  DestBuf big_dst;
  big_dst.init(128, 128);
  t.expect(mc::predict_plane(big.ref(), 0, mc::BlockRect{0, 0, 128, 128},
                             mc::Mv{}, mc::InterpFilter::kEighttap,
                             big_dst.ptr(), big_dst.stride) ==
               mc::McStatus::kBlockTooLarge,
           "128x128 block rejected");

  // MVs use the full AV2 17-bit range, not int16.
  DestBuf large;
  large.init(8, 8);
  t.expect(mc::predict_plane(frame.ref(), 0, mc::BlockRect{0, 0, 8, 8},
                             mc::Mv{65535, -65535},
                             mc::InterpFilter::kEighttap, large.ptr(),
                             large.stride) == mc::McStatus::kOk,
           "large 17-bit MV accepted");

  // §7.13.3.17 must use 64-bit intermediates: 8K position + full MV range.
  const mc::McSampleLocation loc = mc::mc_derive_sample_location(
      7679, 4319, mc::Mv{65535, -65535}, 0, 0);
  const std::int64_t orig_x = (7679LL << 4) + (2LL * -65535) + 8;
  const std::int64_t orig_y = (4319LL << 4) + (2LL * 65535) + 8;
  t.eq_int(loc.start_x, orig_x * 64 - 480, "8K startX no overflow");
  t.eq_int(loc.start_y, orig_y * 64 - 480, "8K startY no overflow");
  t.eq_int(loc.step_x, 1024, "unscaled stepX");
  t.eq_int(loc.step_y, 1024, "unscaled stepY");

  t.expect(mc::predict_plane(frame.ref(), 0, mc::BlockRect{0, 0, 8, 8},
                             mc::Mv{},
                             static_cast<mc::InterpFilter>(9), dst.ptr(),
                             dst.stride) == mc::McStatus::kInvalidArgument,
           "invalid filter rejected");
}

void test_real_frames(Test& t, const std::string& repo_root) {
  const std::string path = repo_root + "/tests/data/m0_640x360_1f.y4m";
  std::vector<Y4mFrame> frames;
  if (!load_y4m(path, 1, &frames) || frames.empty()) {
    std::printf("note: %s unavailable, skipping real-frame test\n",
                path.c_str());
    return;
  }
  TestFrame ref = frame_from_y4m(frames[0]);
  const int dy = 2;
  const int dx = -4;
  const int cdy = dy >> 1;
  const int cdx = dx >> 1;

  // Current frame = reference shifted by a known integer displacement, with
  // edge replication.  An integer-MV prediction must reproduce it exactly.
  TestFrame cur = TestFrame::make(ref.mi_cols, ref.mi_rows);
  for (int y = 0; y < cur.plane[0].height; ++y) {
    const int sy = std::min(std::max(y + dy, 0), ref.plane[0].height - 1);
    for (int x = 0; x < cur.plane[0].width; ++x) {
      const int sx = std::min(std::max(x + dx, 0), ref.plane[0].width - 1);
      cur.plane[0].at(y, x) = ref.plane[0].at(sy, sx);
    }
  }
  for (int p = 1; p <= 2; ++p) {
    for (int y = 0; y < cur.plane[p].height; ++y) {
      const int sy =
          std::min(std::max(y + cdy, 0), ref.plane[p].height - 1);
      for (int x = 0; x < cur.plane[p].width; ++x) {
        const int sx =
            std::min(std::max(x + cdx, 0), ref.plane[p].width - 1);
        cur.plane[p].at(y, x) = ref.plane[p].at(sy, sx);
      }
    }
  }

  const mc::Mv mv{dy * 8, dx * 8};
  int mismatched_blocks = 0;
  int blocks = 0;
  auto start = std::chrono::steady_clock::now();
  for (int by = 0; by < ref.plane[0].height; by += 32) {
    for (int bx = 0; bx < ref.plane[0].width; bx += 32) {
      const mc::BlockRect rect{bx, by, std::min(32, ref.plane[0].width - bx),
                               std::min(32, ref.plane[0].height - by)};
      DestBuf dst;
      dst.init(rect.w, rect.h);
      if (mc::predict_plane(ref.ref(), 0, rect, mv,
                            mc::InterpFilter::kEighttap, dst.ptr(),
                            dst.stride) != mc::McStatus::kOk) {
        ++mismatched_blocks;
        continue;
      }
      ++blocks;
      bool same = true;
      for (int y = 0; y < rect.h && same; ++y) {
        for (int x = 0; x < rect.w; ++x) {
          if (dst.at(y, x) != cur.plane[0].at(rect.y + y, rect.x + x)) {
            same = false;
            break;
          }
        }
      }
      if (!same) {
        ++mismatched_blocks;
      }
    }
  }
  // Chroma at half the displacement.
  int chroma_blocks = 0;
  int chroma_mismatch = 0;
  for (int by = 0; by < ref.plane[1].height; by += 16) {
    for (int bx = 0; bx < ref.plane[1].width; bx += 16) {
      const mc::BlockRect rect{bx, by, std::min(16, ref.plane[1].width - bx),
                               std::min(16, ref.plane[1].height - by)};
      DestBuf dst;
      dst.init(rect.w, rect.h);
      if (mc::predict_plane(ref.ref(), 1, rect, mv,
                            mc::InterpFilter::kEighttap, dst.ptr(),
                            dst.stride) != mc::McStatus::kOk) {
        ++chroma_mismatch;
        continue;
      }
      ++chroma_blocks;
      bool same = true;
      for (int y = 0; y < rect.h && same; ++y) {
        for (int x = 0; x < rect.w; ++x) {
          if (dst.at(y, x) != cur.plane[1].at(rect.y + y, rect.x + x)) {
            same = false;
            break;
          }
        }
      }
      if (!same) {
        ++chroma_mismatch;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  t.eq_int(mismatched_blocks, 0, "real frame integer-MV luma mismatches");
  t.eq_int(chroma_mismatch, 0, "real frame integer-MV chroma mismatches");
  std::printf(
      "real frame: %d luma blocks + %d chroma blocks, 0 mismatches, "
      "%.2f ms (%.1f MP/s) [CPU oracle, not a target]\n",
      blocks, chroma_blocks, ms,
      (static_cast<double>(ref.plane[0].width) * ref.plane[0].height * 1.5 /
       1e6) /
          (ms / 1000.0));

  // Fractional MV on real content: reference == per-sample core, and the
  // known integer displacement beats a zero MV in SAD.
  const mc::BlockRect rect{64, 64, 32, 32};
  DestBuf a;
  DestBuf b;
  a.init(rect.w, rect.h);
  b.init(rect.w, rect.h);
  mc::predict_plane(ref.ref(), 0, rect, mc::Mv{dy * 8 + 1, dx * 8 - 1},
                    mc::InterpFilter::kEighttapSharp, a.ptr(), a.stride);
  predict_with_core(ref.ref(), 0, rect, mc::Mv{dy * 8 + 1, dx * 8 - 1},
                    mc::InterpFilter::kEighttapSharp, b.ptr(), b.stride);
  bool same = true;
  for (int y = 0; y < rect.h && same; ++y) {
    for (int x = 0; x < rect.w; ++x) {
      if (a.at(y, x) != b.at(y, x)) {
        same = false;
        break;
      }
    }
  }
  t.expect(same, "real frame fractional MV == core");

  DestBuf dst;
  dst.init(rect.w, rect.h);
  mc::predict_plane(ref.ref(), 0, rect, mv, mc::InterpFilter::kEighttap,
                    dst.ptr(), dst.stride);
  std::int64_t sad_true = 0;
  std::int64_t sad_zero = 0;
  DestBuf zero;
  zero.init(rect.w, rect.h);
  mc::predict_plane(ref.ref(), 0, rect, mc::Mv{0, 0},
                    mc::InterpFilter::kEighttap, zero.ptr(), zero.stride);
  for (int y = 0; y < rect.h; ++y) {
    for (int x = 0; x < rect.w; ++x) {
      const int src = cur.plane[0].at(rect.y + y, rect.x + x);
      sad_true += std::abs(dst.at(y, x) - src);
      sad_zero += std::abs(zero.at(y, x) - src);
    }
  }
  t.eq_int(sad_true, 0, "real frame true-MV SAD");
  bool block_flat = true;
  const int first = cur.plane[0].at(rect.y, rect.x);
  for (int y = 0; y < rect.h && block_flat; ++y) {
    for (int x = 0; x < rect.w; ++x) {
      if (cur.plane[0].at(rect.y + y, rect.x + x) != first) {
        block_flat = false;
        break;
      }
    }
  }
  if (!block_flat) {
    t.expect(sad_zero > 0, "real frame zero-MV SAD is nonzero");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string repo_root =
      (argc > 1) ? std::string(argv[1]) : std::string("../../../../..");
  Test t;

  test_tables(t);
  test_integer_copy(t);
  test_core_equivalence(t);
  test_bilinear_math(t);
  test_partial_blocks(t);
  test_chroma(t);
  test_mv_stack(t);
  test_modes(t);
  test_merge_selection(t);
  test_validation(t);
  test_real_frames(t, repo_root);

  std::printf("mc CPU: %d checks, %d failures\n", t.checks, t.failures);
  return t.failures == 0 ? 0 : 1;
}
