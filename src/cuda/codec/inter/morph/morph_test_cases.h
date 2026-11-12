// Shared deterministic case list for the morph parity tests.
//
// The CUDA test (morph_test.cu) and the host emulation test
// (morph_emulation_test.cpp) both iterate this list, so the two parity counts
// cover the same inputs. Nothing here is CUDA-specific.
//
// Coverage:
//   * all 22 IBC-signallable block sizes (4x4 .. 64x32, including 4x32/32x4);
//   * all four (AvailU, AvailL) combinations;
//   * interior, top-left, right/bottom-clipped and overhanging rectangles;
//   * zero, integer-pel, sub-pel, negative and very large motion vectors,
//     including every to_fullmv rounding residue and out-of-box rejections;
//   * the 12 -> 8 template rule, count == 0, 64x64 (process-valid), and
//     bit_depth = 10 (the CPU reference rejects everything but 8-bit);
//   * 0/255 extreme sample content.

#ifndef AV2ENC_CUDA_MORPH_MORPH_TEST_CASES_H_
#define AV2ENC_CUDA_MORPH_MORPH_TEST_CASES_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "codec/inter/morph/morph_common.h"

namespace av2enc::cuda::morph::testcases {

struct Case {
  const char* name = "";
  int stride = 0;
  int plane_w = 0;
  int plane_h = 0;
  int rows = 0;
  inter::morph::MorphContext ctx{};
  inter::morph::MorphRect rect{};
  int mv_row = 0;
  int mv_col = 0;
  int fill_mode = 0;  // 0 random, 1 alternating 0/255, 2 all 255, 3 all 0
  std::uint32_t seed = 1;
};

inline std::uint32_t next_random(std::uint32_t* state) {
  std::uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

inline std::vector<std::uint8_t> make_plane(const Case& c) {
  std::vector<std::uint8_t> plane(static_cast<std::size_t>(c.stride) *
                                  static_cast<std::size_t>(c.rows));
  std::uint32_t state = c.seed == 0 ? 1u : c.seed;
  for (std::size_t i = 0; i < plane.size(); ++i) {
    const std::uint32_t random = next_random(&state);
    switch (c.fill_mode) {
      case 1:
        plane[i] = ((i & 1u) != 0u) ? 255 : 0;
        break;
      case 2:
        plane[i] = 255;
        break;
      case 3:
        plane[i] = 0;
        break;
      default:
        plane[i] = static_cast<std::uint8_t>((random >> 24) & 0xFFu);
        break;
    }
  }

  // IBC prediction: the current rectangle contains a copy of the reference
  // block, as it does when morph_predict() runs in the stream writer. Overlap
  // (zero BV) is a copy onto itself. Bounds use the allocation, not the plane,
  // because overhanging rectangles are part of the coverage.
  int ref_x = 0;
  int ref_y = 0;
  int bw = 0;
  int bh = 0;
  if (inter::morph::morph_reference_geometry(c.ctx, c.rect, c.mv_row, c.mv_col,
                                             &ref_x, &ref_y, &bw, &bh) &&
      inter::morph::morph_reference_in_bounds(ref_x, ref_y, bw, bh, c.plane_w,
                                               c.plane_h)) {
    for (int i = 0; i < c.rect.h; ++i) {
      for (int j = 0; j < c.rect.w; ++j) {
        const long long dst_y = static_cast<long long>(c.rect.y) + i;
        const long long dst_x = static_cast<long long>(c.rect.x) + j;
        const long long src_y = static_cast<long long>(ref_y) + i;
        const long long src_x = static_cast<long long>(ref_x) + j;
        if (dst_y < 0 || dst_y >= c.rows || dst_x < 0 || dst_x >= c.stride ||
            src_y < 0 || src_y >= c.rows || src_x < 0 || src_x >= c.stride) {
          continue;
        }
        plane[static_cast<std::size_t>(dst_y) *
                  static_cast<std::size_t>(c.stride) +
              static_cast<std::size_t>(dst_x)] =
            plane[static_cast<std::size_t>(src_y) *
                      static_cast<std::size_t>(c.stride) +
                  static_cast<std::size_t>(src_x)];
      }
    }
  }
  return plane;
}

inline std::vector<Case> all_cases() {
  std::vector<Case> cases;
  std::uint32_t seed = 0x9e3779b9u;

  const auto add = [&](const char* name, int plane_w, int plane_h, int rows,
                       inter::morph::MorphContext ctx,
                       inter::morph::MorphRect rect, int mv_row, int mv_col,
                       int fill_mode = 0) {
    Case c;
    c.name = name;
    c.plane_w = plane_w;
    c.plane_h = plane_h;
    c.stride = plane_w + 64;
    c.rows = std::max(plane_h, rows);
    c.ctx = ctx;
    c.ctx.mi_col = rect.x / 4;
    c.ctx.mi_row = rect.y / 4;
    c.rect = rect;
    c.mv_row = mv_row;
    c.mv_col = mv_col;
    c.fill_mode = fill_mode;
    c.seed = seed++;
    cases.push_back(c);
  };

  const int kSizes[][2] = {
      {4, 4},   {4, 8},   {8, 4},   {8, 8},   {8, 16},  {16, 8},
      {16, 16}, {16, 32}, {32, 16}, {32, 32}, {32, 64}, {64, 32},
      {4, 16},  {16, 4},  {8, 32},  {32, 8},  {16, 64}, {64, 16},
      {4, 32},  {32, 4},  {8, 64},  {64, 8},
  };
  const int kPositions[][2] = {{32, 32}, {8, 8}, {192, 96}, {224, 224}};

  for (const auto& size : kSizes) {
    const int w = size[0];
    const int h = size[1];
    for (int mask = 0; mask < 4; ++mask) {
      const bool avail_u = (mask & 1) != 0;
      const bool avail_l = (mask & 2) != 0;
      for (const auto& position : kPositions) {
        const int x = position[0];
        const int y = position[1];
        inter::morph::MorphContext ctx;
        ctx.mi_cols = 64;  // 256 luma samples
        ctx.mi_rows = 64;
        ctx.avail_u = avail_u;
        ctx.avail_l = avail_l;
        const inter::morph::MorphRect rect{x, y, w, h};
        const int bw = std::max(1, std::min(w, 256 - x));
        const int bh = std::max(1, std::min(h, 256 - y));

        struct Pattern {
          int ref_x;
          int ref_y;
          int d_row;
          int d_col;
        };
        const Pattern patterns[6] = {
            {x, y, 0, 0},
            {8, 8, 0, 0},
            {1, 1, 0, 0},
            {256 - bw - 1, 256 - bh - 1, 0, 0},
            {8, 8, -3, 5},
            {1, 1, 7, -7},
        };
        for (const Pattern& pattern : patterns) {
          const int mv_row = (pattern.ref_y - y) * 8 + pattern.d_row;
          const int mv_col = (pattern.ref_x - x) * 8 + pattern.d_col;
          add("size/avail/position/pattern", 256, 256, y + h,
              ctx, rect, mv_row, mv_col);
        }
      }
    }
  }

  // to_fullmv residue sweep at a negative and a positive origin. d spans two
  // full 8-sample periods around each base, so every residue in both signs is
  // exercised.
  {
    inter::morph::MorphContext ctx;
    ctx.mi_cols = 64;
    ctx.mi_rows = 64;
    ctx.avail_u = true;
    ctx.avail_l = true;
    const inter::morph::MorphRect rect{32, 32, 16, 16};
    for (int d = -24; d <= 24; ++d) {
      add("to_fullmv negative", 256, 256, 256, ctx, rect, -192 + d, -192 + d);
      add("to_fullmv positive", 256, 256, 256, ctx, rect, 64 + d, 64 + d);
    }
  }

  // Large motion magnitudes: a wide plane and a tall plane let the vector move
  // the reference origin by ~32k 1/8-pel units while staying in the box.
  {
    inter::morph::MorphContext wide;
    wide.mi_cols = 1024;  // 4096
    wide.mi_rows = 16;    // 64
    wide.avail_u = true;
    wide.avail_l = true;
    const inter::morph::MorphRect left_block{64, 8, 64, 16};
    const inter::morph::MorphRect right_block{4032, 8, 64, 16};
    add("wide +31744", 4096, 64, 24, wide, left_block, 0, 31744);
    add("wide -32248", 4096, 64, 24, wide, right_block, 0, -32248);
    add("wide +31744 subpel", 4096, 64, 24, wide, left_block, 7, 31744 + 5);
    add("wide -32248 subpel", 4096, 64, 24, wide, right_block, -7, -32248 - 3);

    inter::morph::MorphContext tall;
    tall.mi_cols = 16;
    tall.mi_rows = 1024;
    tall.avail_u = true;
    tall.avail_l = true;
    const inter::morph::MorphRect top_block{8, 64, 16, 64};
    const inter::morph::MorphRect bottom_block{8, 4032, 16, 64};
    add("tall +31744", 64, 4096, 4096, tall, top_block, 31744, 0);
    add("tall -32248", 64, 4096, 4096, tall, bottom_block, -32248, 0);
    add("tall +31744 subpel", 64, 4096, 4096, tall, top_block, 31744 + 7, 3);
    add("tall -32248 subpel", 64, 4096, 4096, tall, bottom_block,
        -32248 - 7, -3);

    // The legal 1/8-pel range includes values that cannot stay in the box.
    const int extremes[] = {65536, 65528, 32768, -32768, -65528, -65536};
    for (const int extreme : extremes) {
      add("wide extreme column reject", 4096, 64, 24, wide, left_block, 0,
          extreme);
      add("tall extreme row reject", 64, 4096, 4096, tall, top_block, extreme,
          0);
    }
  }

  // Conformance-box rejections and consistency checks.
  {
    inter::morph::MorphContext ctx;
    ctx.mi_cols = 64;
    ctx.mi_rows = 64;
    ctx.avail_u = true;
    ctx.avail_l = true;
    const inter::morph::MorphRect rect{32, 32, 16, 16};
    add("refX==0", 256, 256, 256, ctx, rect, 0, (0 - 32) * 8);
    add("refY==0", 256, 256, 256, ctx, rect, (0 - 32) * 8, 0);
    add("refX+bw>plane", 256, 256, 256, ctx, rect, 0, (241 - 32) * 8);
    add("refY+bh>plane", 256, 256, 256, ctx, rect, (241 - 32) * 8, 0);
    {
      inter::morph::MorphContext top = ctx;
      top.avail_u = true;
      add("availU at y=0", 256, 256, 256, top,
          inter::morph::MorphRect{32, 0, 16, 16}, 0, 0);
    }
    {
      inter::morph::MorphContext left = ctx;
      left.avail_l = true;
      add("availL at x=0", 256, 256, 256, left,
          inter::morph::MorphRect{0, 32, 16, 16}, 0, 0);
    }
    add("negative x", 256, 256, 256, ctx,
        inter::morph::MorphRect{-4, 32, 16, 16}, 0, 0);
    add("negative y", 256, 256, 256, ctx,
        inter::morph::MorphRect{32, -4, 16, 16}, 0, 0);
    add("zero width", 256, 256, 256, ctx,
        inter::morph::MorphRect{32, 32, 0, 16}, 0, 0);
    add("zero height", 256, 256, 256, ctx,
        inter::morph::MorphRect{32, 32, 16, 0}, 0, 0);
    {
      inter::morph::MorphContext none = ctx;
      none.mi_cols = 0;
      add("mi_cols==0", 256, 256, 256, none, rect, 0, 0);
      none = ctx;
      none.mi_rows = 0;
      add("mi_rows==0", 256, 256, 256, none, rect, 0, 0);
    }
  }

  // bit_depth = 10: the CPU reference rejects anything but 8-bit before it
  // touches the plane. The GPU must match that rejection exactly.
  {
    const int ten_bit_sizes[][2] = {{4, 4}, {8, 8}, {16, 16}, {32, 32},
                                    {64, 32}, {16, 64}};
    for (const auto& size : ten_bit_sizes) {
      for (int mask = 0; mask < 4; ++mask) {
        inter::morph::MorphContext ctx;
        ctx.mi_cols = 64;
        ctx.mi_rows = 64;
        ctx.bit_depth = 10;
        ctx.avail_u = (mask & 1) != 0;
        ctx.avail_l = (mask & 2) != 0;
        add("bit_depth 10", 256, 256, 256, ctx,
            inter::morph::MorphRect{32, 32, size[0], size[1]}, -192, -192);
      }
    }
  }

  // bw == 12 or bh == 12: the template extent is clamped to 8.
  {
    inter::morph::MorphContext ctx;
    ctx.mi_cols = 64;
    ctx.mi_rows = 64;
    ctx.avail_u = true;
    ctx.avail_l = true;
    for (int mask = 0; mask < 4; ++mask) {
      ctx.avail_u = (mask & 1) != 0;
      ctx.avail_l = (mask & 2) != 0;
      add("bw=12,bh=12", 256, 256, 288, ctx,
          inter::morph::MorphRect{244, 244, 32, 32}, 0, 0);
      add("bw=12", 256, 256, 256, ctx,
          inter::morph::MorphRect{244, 32, 32, 16}, 0, 0);
      add("bh=12", 256, 256, 288, ctx,
          inter::morph::MorphRect{32, 244, 16, 32}, 0, 0);
    }
  }

  // 64x64 is not IBC-signallable but the §7.13.3.26 process is defined for it;
  // keep it covered so a caller cannot get a silently different answer.
  {
    inter::morph::MorphContext ctx;
    ctx.mi_cols = 64;
    ctx.mi_rows = 64;
    for (int mask = 0; mask < 4; ++mask) {
      ctx.avail_u = (mask & 1) != 0;
      ctx.avail_l = (mask & 2) != 0;
      add("64x64", 256, 256, 256, ctx,
          inter::morph::MorphRect{32, 32, 64, 64}, 0, 0);
      add("64x64 shifted", 256, 256, 256, ctx,
          inter::morph::MorphRect{32, 32, 64, 64}, (8 - 32) * 8,
          (8 - 32) * 8);
    }
  }

  // Extreme sample content, including the Clip1 boundaries.
  {
    inter::morph::MorphContext ctx;
    ctx.mi_cols = 64;
    ctx.mi_rows = 64;
    ctx.avail_u = true;
    ctx.avail_l = true;
    const inter::morph::MorphRect rect{32, 32, 32, 32};
    for (int fill = 1; fill <= 3; ++fill) {
      add("fill extremes", 256, 256, 256, ctx, rect, -192, -192, fill);
      add("fill extremes subpel", 256, 256, 256, ctx, rect, -189, -187, fill);
    }
  }

  return cases;
}

}  // namespace av2enc::cuda::morph::testcases

#endif  // AV2ENC_CUDA_MORPH_MORPH_TEST_CASES_H_
