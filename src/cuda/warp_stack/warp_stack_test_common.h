#ifndef AV2ENC_CUDA_WARP_STACK_WARP_STACK_TEST_COMMON_H_
#define AV2ENC_CUDA_WARP_STACK_WARP_STACK_TEST_COMMON_H_

// Shared fixtures for the host-emulation and GPU parity tests.
//
// Keeping the model/case generators in one place means the emulation test and
// the GPU test exercise the same corner/rotation/zoom/shear matrix, block
// sizes, edge blocks and bit depths.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "encoder/inter/advanced/warp.h"
#include "encoder/inter/warp_stack/warp_stack.h"

namespace av2enc::cuda::warpstack::test {

namespace ia = inter::adv;
namespace iw = inter::warp;

inline constexpr int kTestPlaneSize = 96;
inline constexpr int kTestDstPitch = 64;

inline std::uint32_t g_rng = 0x51ed270bu;

inline std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

inline int random_range(int low, int high) {
  return low + static_cast<int>(next_random() % static_cast<std::uint32_t>(high - low + 1));
}

struct ModelCase {
  std::string name;
  ia::WarpParams params;
};

inline ia::WarpParams identity_model() {
  ia::WarpParams p{};
  p.p[2] = 1 << 16;
  p.p[5] = 1 << 16;
  return p;
}

inline std::vector<ModelCase> make_models() {
  std::vector<ModelCase> out;
  out.push_back({"identity", identity_model()});

  ia::WarpParams translation = identity_model();
  translation.p[0] = 3 << 16;
  translation.p[1] = -2 << 16;
  out.push_back({"translation", translation});

  {
    ia::WarpParams p = identity_model();
    const int c = 65536 - 700;
    const int s = 700;
    p.p[2] = c;
    p.p[3] = -s;
    p.p[4] = s;
    p.p[5] = c;
    p.p[0] = 1 << 16;
    p.p[1] = -1 << 16;
    out.push_back({"rotation", p});
  }
  {
    ia::WarpParams p = identity_model();
    p.p[2] = 65536 + 1500;
    p.p[5] = 65536 - 1100;
    out.push_back({"zoom", p});
  }
  {
    ia::WarpParams p = identity_model();
    p.p[3] = 900;
    p.p[4] = -600;
    out.push_back({"shear", p});
  }
  {
    ia::WarpParams p = identity_model();
    const int c = 65536 - 500;
    const int s = 500;
    p.p[2] = c;
    p.p[3] = -s + 300;
    p.p[4] = s - 200;
    p.p[5] = c - 400;
    p.p[0] = -2 << 16;
    p.p[1] = 3 << 16;
    out.push_back({"rotation+zoom+shear", p});
  }
  {
    // adv::setup_shear() rejects this: 4*|alpha| + 7*|beta| >= 3 << 16.
    ia::WarpParams p = identity_model();
    p.p[3] = 2 << 16;
    out.push_back({"invalid shear", p});
  }
  for (int i = 0; i < 4; ++i) {
    ia::WarpParams p = identity_model();
    p.p[2] = 65536 + random_range(-2500, 2500);
    p.p[5] = 65536 + random_range(-2500, 2500);
    p.p[3] = random_range(-1200, 1200);
    p.p[4] = random_range(-1200, 1200);
    p.p[0] = random_range(-3, 3) << 16;
    p.p[1] = random_range(-3, 3) << 16;
    out.push_back({"random affine", p});
  }
  return out;
}

inline ia::WarpBlockGeometry make_geometry(int plane, int sub_x, int sub_y, int x, int y, int w,
                                           int h, int ref_w, int ref_h, int bit_depth) {
  ia::WarpBlockGeometry g{};
  g.plane = plane;
  g.sub_x = sub_x;
  g.sub_y = sub_y;
  g.rect = {x, y, w, h};
  g.dst_x = 0;
  g.dst_y = 0;
  g.mi_col = x >> 2;
  g.mi_row = y >> 2;
  g.bw4 = std::max(1, w >> 2);
  g.bh4 = std::max(1, h >> 2);
  g.mi_cols = std::max(g.mi_col + g.bw4, 1);
  g.mi_rows = std::max(g.mi_row + g.bh4, 1);
  g.ref_w = ref_w;
  g.ref_h = ref_h;
  g.frame_w = ref_w;
  g.frame_h = ref_h;
  g.bit_depth = bit_depth;
  return g;
}

struct Plane8 {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> data;
  ia::PlaneView view{};
};

inline Plane8 make_plane8(int w, int h) {
  Plane8 p;
  p.w = w;
  p.h = h;
  p.data.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  for (auto& v : p.data) {
    v = static_cast<std::uint8_t>(random_range(0, 255));
  }
  p.view.data = p.data.data();
  p.view.stride = w;
  p.view.max_x = w - 1;
  p.view.max_y = h - 1;
  return p;
}

struct Plane16 {
  int w = 0;
  int h = 0;
  std::vector<std::uint16_t> data;
};

inline Plane16 make_plane16(int w, int h, int max_value) {
  Plane16 p;
  p.w = w;
  p.h = h;
  p.data.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  for (auto& v : p.data) {
    v = static_cast<std::uint16_t>(random_range(0, max_value));
  }
  return p;
}

// One prediction case. `w`/`h`/`x`/`y` are luma coordinates; for `sub != 0`
// the test halves them before building the geometry, matching a 4:2:0 plane.
struct PredictionCase {
  ModelCase model;
  int w = 0;
  int h = 0;
  int x = 0;
  int y = 0;
  int plane = 0;
  int sub = 0;
  int compound = 0;
  int bit_depth = 8;
};

inline std::string case_name(const PredictionCase& c) {
  return c.model.name + " " + std::to_string(c.w) + "x" + std::to_string(c.h) +
         (c.x != 0 || c.y != 0 ? " edge" : " origin") +
         (c.sub ? " chroma" : "") + (c.compound ? " compound" : " single") +
         (c.bit_depth == 10 ? " 10-bit" : "");
}

// Full matrix: all 8/10-bit block sizes, origin and edge positions, both
// precision modes, plus a 4:2:0 subset. "All block sizes" in AV2 are the
// 4/8/16/32/64 extents in both dimensions (including rectangular).
inline std::vector<PredictionCase> prediction_cases() {
  const int dims[] = {4, 8, 16, 32, 64};
  const std::vector<ModelCase> models = make_models();
  std::vector<PredictionCase> out;
  for (const ModelCase& model : models) {
    for (int w : dims) {
      for (int h : dims) {
        for (int edge = 0; edge < 2; ++edge) {
          for (int compound = 0; compound < 2; ++compound) {
            for (int depth : {8, 10}) {
              PredictionCase c;
              c.model = model;
              c.w = w;
              c.h = h;
              c.x = edge ? kTestPlaneSize - w : 0;
              c.y = edge ? kTestPlaneSize - h : 0;
              c.plane = 0;
              c.sub = 0;
              c.compound = compound;
              c.bit_depth = depth;
              out.push_back(c);
            }
          }
        }
      }
    }
    for (int w : {4, 8, 16, 32}) {
      PredictionCase c;
      c.model = model;
      c.w = w * 2;
      c.h = w * 2;
      c.x = 0;
      c.y = 0;
      c.plane = 1;
      c.sub = 1;
      c.compound = 0;
      c.bit_depth = 8;
      out.push_back(c);
      c.x = (kTestPlaneSize - w) * 2;
      c.y = (kTestPlaneSize - w) * 2;
      out.push_back(c);
    }
  }
  return out;
}

// Fully-populated MI grid with random blocks, modes, references and warp
// models. Every cell of a block carries the block base/extent so §7.12.2.4 and
// §7.12.3 see a consistent layout.
inline void fill_grid(std::vector<iw::WarpMi>* mi, int cols, int rows) {
  mi->assign(static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows), iw::WarpMi{});
  const int sizes[] = {1, 2, 4, 8, 16};
  for (int r = 0; r < rows;) {
    const int h4 = std::min(sizes[random_range(0, 4)], rows - r);
    for (int c = 0; c < cols;) {
      const int w4 = std::min(sizes[random_range(0, 4)], cols - c);
      const bool inter = random_range(0, 5) != 0;
      const std::int8_t ref0 =
          inter ? static_cast<std::int8_t>(random_range(1, 3)) : iw::kRefIntra;
      const std::int8_t ref1 = inter && random_range(0, 2) == 0
                                   ? static_cast<std::int8_t>(random_range(1, 3))
                                   : iw::kRefNone;
      const int motion = random_range(0, 4);
      ia::WarpParams warp{};
      warp.p[2] = 65536 + random_range(-900, 900);
      warp.p[5] = 65536 + random_range(-900, 900);
      warp.p[3] = random_range(-400, 400);
      warp.p[4] = random_range(-400, 400);
      warp.p[0] = random_range(-2, 2) << 16;
      warp.p[1] = random_range(-2, 2) << 16;
      for (int y = r; y < r + h4; ++y) {
        for (int x = c; x < c + w4; ++x) {
          iw::WarpMi& cell = (*mi)[static_cast<std::size_t>(y) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(x)];
          cell.written = 1;
          cell.is_inter = inter ? 1 : 0;
          cell.ref[0] = ref0;
          cell.ref[1] = ref1;
          cell.mv[0][0] = random_range(-60, 60);
          cell.mv[0][1] = random_range(-60, 60);
          cell.mv[1][0] = random_range(-60, 60);
          cell.mv[1][1] = random_range(-60, 60);
          cell.sub_mv[0][0] = random_range(-60, 60);
          cell.sub_mv[0][1] = random_range(-60, 60);
          cell.sub_mv[1][0] = random_range(-60, 60);
          cell.sub_mv[1][1] = random_range(-60, 60);
          cell.w4 = static_cast<std::int8_t>(w4);
          cell.h4 = static_cast<std::int8_t>(h4);
          cell.col_base = static_cast<std::int16_t>(c);
          cell.row_base = static_cast<std::int16_t>(r);
          cell.y_mode = static_cast<std::uint8_t>(random_range(0, 23));
          cell.motion_mode = static_cast<std::uint8_t>(motion);
          cell.warp[0] = warp;
          cell.warp[1] = warp;
        }
      }
      c += w4;
    }
    r += h4;
  }
}

inline bool corner_fields_equal(const iw::CornerResult& a, const iw::CornerResult& b) {
  if (a.corners_found != b.corners_found || a.all_mvs_same != b.all_mvs_same ||
      a.negative_ref_pts != b.negative_ref_pts) {
    return false;
  }
  for (int i = 0; i < 3; ++i) {
    for (int c = 0; c < 2; ++c) {
      if (a.corner_pts[i][c] != b.corner_pts[i][c] ||
          a.corner_mvs[i][c] != b.corner_mvs[i][c] ||
          a.ref_pts[i][c] != b.ref_pts[i][c]) {
        return false;
      }
    }
  }
  for (int i = 0; i < 6; ++i) {
    if (a.wmmat.p[i] != b.wmmat.p[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace av2enc::cuda::warpstack::test

#endif  // AV2ENC_CUDA_WARP_STACK_WARP_STACK_TEST_COMMON_H_
