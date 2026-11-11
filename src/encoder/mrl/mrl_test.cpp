// AV2 MRL CPU reference tests.
//
// Two independent checks:
//   * mrl_index = 0 must reproduce the already-verified CPU intra module
//     (src/encoder/intra_modes) byte for byte, including edge filtering and
//     smooth-neighbour flags;
//   * mrl_index = 1..3 (and the secondary-line blend) is compared against a
//     literal nested-loop transcription of §7.13.2.1/.7/.8 in this file.
//
// The naive model deliberately shares no code with src/encoder/mrl/mrl.cpp.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "encoder/intra_modes/intra_modes.h"
#include "encoder/mrl/mrl.h"

namespace {

using av2enc::intra::IntraStatus;
using av2enc::intra::Mode;
using av2enc::mrl::Block;
using av2enc::mrl::Edges;
using av2enc::mrl::MrlStatus;
using av2enc::mrl::Params;

int g_checks = 0;
int g_failures = 0;

void fail(const char* what, int detail = 0) {
  ++g_checks;
  ++g_failures;
  if (g_failures <= 25) {
    std::printf("FAIL: %s (%d)\n", what, detail);
  }
}

void expect(bool condition, const char* what) {
  ++g_checks;
  if (!condition) fail(what);
}

std::uint32_t g_rng = 0x51ed270bu;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
int random_range(int low, int high) {
  return low + static_cast<int>(next_random() %
                                static_cast<std::uint32_t>(high - low + 1));
}

constexpr int kModes[8] = {1, 2, 3, 4, 5, 6, 7, 8};
constexpr int kDeltas[7] = {-3, -2, -1, 0, 1, 2, 3};

// ---------------------------------------------------------------------------
// Literal spec model (mrl_index > 0 only; no edge filtering by §7.13.2.7).
// ---------------------------------------------------------------------------

constexpr int kAngleStep = 3;
constexpr int kDrIntraDerivative[90] = {
    0, 4096, 2048, 1365, 1024, 819, 682, 585, 512, 455, 409, 409, 409, 372,
    341, 292, 273, 256, 227, 215, 204, 186, 178, 170, 157, 151, 146, 136, 132,
    128, 117, 110, 107, 99, 97, 97, 93, 87, 83, 81, 77, 74, 73, 69, 66, 64, 62,
    59, 56, 55, 53, 50, 49, 47, 44, 42, 42, 41, 38, 37, 35, 32, 31, 30, 28, 27,
    26, 24, 23, 22, 20, 19, 18, 16, 15, 14, 12, 11, 10, 10, 10, 9, 8, 7, 6, 5,
    4, 3, 2, 1};

constexpr int kDrInterpFilter[32][4] = {
    {0, 128, 0, 0},   {-2, 127, 4, -1},  {-3, 125, 8, -2},
    {-5, 123, 13, -3}, {-6, 121, 17, -4}, {-7, 118, 22, -5},
    {-9, 116, 27, -6}, {-9, 112, 32, -7}, {-10, 109, 37, -8},
    {-11, 106, 41, -8}, {-11, 102, 46, -9}, {-12, 98, 52, -10},
    {-12, 94, 56, -10}, {-12, 90, 61, -11}, {-12, 85, 66, -11},
    {-12, 81, 71, -12}, {-12, 76, 76, -12}, {-12, 71, 81, -12},
    {-11, 66, 85, -12}, {-11, 61, 90, -12}, {-10, 56, 94, -12},
    {-10, 52, 98, -12}, {-9, 46, 102, -11}, {-8, 41, 106, -11},
    {-8, 37, 109, -10}, {-7, 32, 112, -9}, {-6, 27, 116, -9},
    {-5, 22, 118, -7}, {-4, 17, 121, -6}, {-3, 13, 123, -5},
    {-2, 8, 125, -3}, {-1, 4, 127, -2}};

int naive_mode_angle(int mode) {
  static constexpr int kTable[13] = {0, 90, 180, 45, 135, 113, 157,
                                     203, 67, 0, 0, 0, 0};
  return kTable[mode];
}

int naive_wide(int p, int w, int h) {
  if ((h == 2 * w && p < 61) || (h == 4 * w && p < 73) ||
      (h == 8 * w && p < 82) || (h == 16 * w && p < 86)) {
    return 180 + p;
  }
  if ((w == 2 * h && p > 270 - 61) || (w == 4 * h && p > 270 - 73) ||
      (w == 8 * h && p > 270 - 82) || (w == 16 * h && p > 270 - 86)) {
    return p - 180;
  }
  return p;
}

int naive_clip1(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }
int naive_round2(int v, int n) {
  return n <= 0 ? v : (v + (1 << (n - 1))) >> n;
}

struct NaiveEdges {
  std::array<int, 160> above{};
  std::array<int, 160> left{};
  std::array<int, 160> above_sec{};
  std::array<int, 160> left_sec{};
  int offset = 0;
  int& a(int i) { return above[static_cast<std::size_t>(i + offset)]; }
  int& l(int i) { return left[static_cast<std::size_t>(i + offset)]; }
  int& as(int i) { return above_sec[static_cast<std::size_t>(i + offset)]; }
  int& ls(int i) { return left_sec[static_cast<std::size_t>(i + offset)]; }
};

std::uint8_t plane_read(const std::vector<std::uint8_t>& p, int stride, int x,
                        int y) {
  return p[static_cast<std::size_t>(y) * stride + x];
}

NaiveEdges naive_edges(const std::vector<std::uint8_t>& plane, int stride,
                       int x, int y, int w, int h, int max_x, int max_y,
                       bool have_left, bool have_above, int num4_ar,
                       int num4_bl, int mrl) {
  NaiveEdges e;
  e.offset = 8;
  const int sb_h = 64;
  const bool sb_boundary = (y & (sb_h - 1)) == 0;
  const int above_mrl = sb_boundary ? 0 : mrl;
  const int needed = w + h + (mrl << 1);
  for (int i = 0; i < needed; ++i) {
    if (!have_left && have_above) {
      e.l(i) = plane_read(plane, stride, x, y - 1 - above_mrl);
      e.ls(i) = plane_read(plane, stride, x, y - 1);
    } else if (!have_left) {
      e.l(i) = 129;
      e.ls(i) = 129;
    } else {
      const int limit = std::min(max_y, y + h + 4 * num4_bl - 1);
      const int row = std::min(limit, y + i);
      e.l(i) = plane_read(plane, stride, x - 1 - mrl, row);
      e.ls(i) = plane_read(plane, stride, x - 1, row);
    }
    if (!have_above && have_left) {
      e.a(i) = plane_read(plane, stride, x - 1 - mrl, y);
      e.as(i) = plane_read(plane, stride, x - 1, y);
    } else if (!have_above) {
      e.a(i) = 127;
      e.as(i) = 127;
    } else {
      const int limit = std::min(max_x, x + w + 4 * num4_ar - 1);
      const int col = std::min(limit, x + i);
      e.a(i) = plane_read(plane, stride, col, y - 1 - above_mrl);
      e.as(i) = plane_read(plane, stride, col, y - 1);
    }
  }
  for (int i = 1; i <= 1 + mrl; ++i) {
    if (have_above && have_left) {
      e.a(-i) = plane_read(plane, stride, x - i, y - 1 - above_mrl);
      e.l(-i) = plane_read(plane, stride, x - 1 - mrl,
                           y - std::min(i, 1 + above_mrl));
      e.as(-i) = plane_read(plane, stride, x - i, y - 1);
      e.ls(-i) = plane_read(plane, stride, x - 1, y - 1);
    } else if (have_above) {
      e.a(-i) = plane_read(plane, stride, x, y - 1 - above_mrl);
      e.l(-i) = e.a(-i);
      e.as(-i) = plane_read(plane, stride, x, y - 1);
      e.ls(-i) = e.as(-i);
    } else if (have_left) {
      e.a(-i) = plane_read(plane, stride, x - 1 - mrl, y);
      e.l(-i) = e.a(-i);
      e.as(-i) = plane_read(plane, stride, x - 1, y);
      e.ls(-i) = e.as(-i);
    } else {
      e.a(-i) = 128;
      e.l(-i) = 128;
      e.as(-i) = 128;
      e.ls(-i) = 128;
    }
  }
  return e;
}

void naive_single(int p_angle, int w, int h, int mrl, NaiveEdges& e,
                  std::uint8_t* dst, int dst_stride) {
  const int min_base = -(1 + mrl);
  const int max_base = w + h - 1 + (mrl << 1);
  if (p_angle > 90 && p_angle < 180) {
    e.l(h) = e.l(h - 1);
    e.a(w) = e.a(w - 1);
    e.l(h + 1) = e.l(h - 1);
    e.a(w + 1) = e.a(w - 1);
  } else {
    e.l(max_base + 1) = e.l(max_base);
    e.a(max_base + 1) = e.a(max_base);
    e.l(max_base + 2) = e.l(max_base);
    e.a(max_base + 2) = e.a(max_base);
  }
  e.l(min_base - 1) = e.l(min_base);
  e.a(min_base - 1) = e.a(min_base);

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int v = 0;
      if (p_angle < 90) {
        const int dx = kDrIntraDerivative[p_angle];
        const int idx = (i + 1 + mrl) * dx;
        const int base = (idx >> 6) + j;
        const int shift = (idx >> 1) & 0x1F;
        const int max_base_x = w + h - 1 + (mrl << 1);
        if (base < max_base_x + 1) {
          int sum = 0;
          for (int t = 0; t < 4; ++t) {
            sum += kDrInterpFilter[shift][t] * e.a(base + t - 1);
          }
          v = naive_clip1(naive_round2(sum, 7));
        } else {
          v = e.a(max_base_x);
        }
      } else if (p_angle > 90 && p_angle < 180) {
        const int dx = kDrIntraDerivative[180 - p_angle];
        const int dy = kDrIntraDerivative[p_angle - 90];
        const int idx = (j << 6) - (i + 1 + mrl) * dx;
        const int base = idx >> 6;
        if (base >= min_base) {
          const int shift = (idx >> 1) & 0x1F;
          int sum = 0;
          for (int t = 0; t < 4; ++t) {
            sum += kDrInterpFilter[shift][t] * e.a(base + t - 1);
          }
          v = naive_clip1(naive_round2(sum, 7));
        } else {
          const int idx2 = (i << 6) - (j + 1 + mrl) * dy;
          const int base2 = idx2 >> 6;
          const int shift = (idx2 >> 1) & 0x1F;
          int sum = 0;
          for (int t = 0; t < 4; ++t) {
            sum += kDrInterpFilter[shift][t] * e.l(base2 + t - 1);
          }
          v = naive_clip1(naive_round2(sum, 7));
        }
      } else if (p_angle > 180) {
        const int dy = kDrIntraDerivative[270 - p_angle];
        const int idx = (j + 1 + mrl) * dy;
        const int base = (idx >> 6) + i;
        const int shift = (idx >> 1) & 0x1F;
        const int max_base_y = w + h - 1 + (mrl << 1);
        if (base < max_base_y + 1) {
          int sum = 0;
          for (int t = 0; t < 4; ++t) {
            sum += kDrInterpFilter[shift][t] * e.l(base + t - 1);
          }
          v = naive_clip1(naive_round2(sum, 7));
        } else {
          v = e.l(max_base_y);
        }
      } else {
        v = p_angle == 90 ? e.a(j) : e.l(i);
      }
      dst[static_cast<std::size_t>(i) * dst_stride + j] =
          static_cast<std::uint8_t>(v);
    }
  }
}

void naive_model(const std::vector<std::uint8_t>& plane, int stride, int x,
                 int y, int w, int h, int max_x, int max_y, bool have_left,
                 bool have_above, int num4_ar, int num4_bl, int mode,
                 int angle_delta, int mrl, bool sec, std::uint8_t* dst,
                 int dst_stride) {
  NaiveEdges e = naive_edges(plane, stride, x, y, w, h, max_x, max_y, have_left,
                             have_above, num4_ar, num4_bl, mrl);
  const int mrl_delta = mrl == 1 ? 1 : (mrl == 2 ? -1 : 0);
  const int p_angle = naive_wide(
      naive_mode_angle(mode) + angle_delta * kAngleStep + mrl_delta, w, h);
  naive_single(p_angle, w, h, mrl, e, dst, dst_stride);
  if (mrl > 0 && sec && !(w == 4 && h == 4)) {
    NaiveEdges sec_edges = naive_edges(plane, stride, x, y, w, h, max_x, max_y,
                                       have_left, have_above, num4_ar, num4_bl,
                                       0);
    std::vector<std::uint8_t> second(static_cast<std::size_t>(w) * h);
    naive_single(p_angle, w, h, 0, sec_edges, second.data(), w);
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const std::size_t index = static_cast<std::size_t>(i) * dst_stride + j;
        dst[index] = static_cast<std::uint8_t>(
            (dst[index] + second[static_cast<std::size_t>(i) * w + j] + 1) >> 1);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

struct Geometry {
  int x;
  int y;
  int w;
  int h;
};

std::vector<Geometry> geometries() {
  std::vector<Geometry> out;
  const int dims[] = {4, 8, 16, 32, 64};
  const int positions[][2] = {{0, 0},   {4, 4},   {8, 4},   {4, 8},
                              {12, 20}, {32, 32}, {64, 4},  {4, 64},
                              {60, 60}};
  for (int w : dims) {
    for (int h : dims) {
      for (const auto& p : positions) {
        out.push_back(Geometry{p[0], p[1], w, h});
      }
    }
  }
  return out;
}

void run_one(const Geometry& g, int mode, int delta, int mrl, bool sec,
             bool smooth_above, bool smooth_left) {
  const int stride = 512;
  const int rows = 512;
  std::vector<std::uint8_t> plane(static_cast<std::size_t>(stride) * rows);
  for (std::uint8_t& v : plane) {
    v = static_cast<std::uint8_t>(next_random() & 0xFFu);
  }
  // Make the current block distinct from the references.
  for (int i = 0; i < g.h; ++i) {
    for (int j = 0; j < g.w; ++j) {
      plane[static_cast<std::size_t>(g.y + i) * stride + g.x + j] =
          static_cast<std::uint8_t>((i * 3 + j * 5 + 91) & 0xFF);
    }
  }
  const int max_x = stride - 40;
  const int max_y = rows - 40;
  const bool have_left = g.x > 0;
  const bool have_above = g.y > 0;
  const int num4_ar = random_range(0, g.w / 4 + 2);
  const int num4_bl = random_range(0, g.h / 4 + 2);

  Block block;
  block.recon = av2enc::intra::PlaneView{plane.data(), stride};
  block.x = g.x;
  block.y = g.y;
  block.w = g.w;
  block.h = g.h;
  block.max_x = max_x;
  block.max_y = max_y;
  block.have_left = have_left;
  block.have_above = have_above;
  block.num4_above_right = num4_ar;
  block.num4_below_left = num4_bl;
  block.plane = 0;
  block.sb_height = 64;

  const bool effective_sec = sec && !(g.w == 4 && g.h == 4);
  Params params;
  params.mode = static_cast<Mode>(mode);
  params.angle_delta = delta;
  params.mrl_index = mrl;
  params.mrl_sec_index = effective_sec;
  params.enable_edge_filter = true;
  params.above_neighbor_is_smooth = smooth_above;
  params.left_neighbor_is_smooth = smooth_left;

  std::vector<std::uint8_t> got(static_cast<std::size_t>(g.w) * g.h);
  const MrlStatus status = av2enc::mrl::predict_block(
      block, 8, params, got.data(), static_cast<std::size_t>(g.w));
  if (status != MrlStatus::kOk) {
    if (g_failures < 25) {
      std::printf("REJECT mrl=%d sec=%d mode=%d delta=%d %dx%d x=%d y=%d\n",
                  mrl, sec ? 1 : 0, mode, delta, g.w, g.h, g.x, g.y);
    }
    fail("predict_block rejected a conforming case");
    return;
  }

  std::vector<std::uint8_t> want(static_cast<std::size_t>(g.w) * g.h);
  if (mrl == 0) {
    // The verified CPU intra module is the oracle for mrl_index 0.
    av2enc::intra::NeighborRequest request;
    request.recon = block.recon;
    request.x = g.x;
    request.y = g.y;
    request.w = g.w;
    request.h = g.h;
    request.max_x = max_x;
    request.max_y = max_y;
    request.have_left = have_left;
    request.have_above = have_above;
    request.num4_above_right = num4_ar;
    request.num4_below_left = num4_bl;
    request.color_plane = 0;
    av2enc::intra::IntraEdges edges;
    if (av2enc::intra::prepare_edges(request, 8, edges) != IntraStatus::kOk) {
      fail("intra prepare_edges failed");
      return;
    }
    av2enc::intra::PredictParams p;
    p.mode = static_cast<Mode>(mode);
    p.angle_delta = delta;
    p.enable_edge_filter = true;
    p.above_neighbor_is_smooth = smooth_above;
    p.left_neighbor_is_smooth = smooth_left;
    if (av2enc::intra::predict_intra(edges, p, want.data(), g.w) !=
        IntraStatus::kOk) {
      fail("intra predict_intra failed");
      return;
    }
  } else {
    naive_model(plane, stride, g.x, g.y, g.w, g.h, max_x, max_y, have_left,
                have_above, num4_ar, num4_bl, mode, delta, mrl, effective_sec,
                want.data(), g.w);
  }

  for (std::size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      ++g_checks;
      ++g_failures;
      if (g_failures <= 25) {
        std::printf(
            "FAIL mrl=%d sec=%d mode=%d delta=%d %dx%d x=%d y=%d sample=%zu "
            "got=%u want=%u\n",
            mrl, sec ? 1 : 0, mode, delta, g.w, g.h, g.x, g.y, i,
            static_cast<unsigned>(got[i]), static_cast<unsigned>(want[i]));
      }
      return;
    }
  }
  ++g_checks;
}

void availability_tests() {
  std::vector<std::uint8_t> plane(512 * 512, 100);
  Block block;
  block.recon = av2enc::intra::PlaneView{plane.data(), 512};
  block.x = 4;
  block.y = 4;
  block.w = 8;
  block.h = 8;
  block.max_x = 500;
  block.max_y = 500;
  block.have_left = true;
  block.have_above = true;
  block.plane = 0;
  Params params;
  params.mode = Mode::kD45;
  params.mrl_index = 1;

  expect(av2enc::mrl::mode_allows_mrl(Mode::kD45), "directional allows MRL");
  expect(!av2enc::mrl::mode_allows_mrl(Mode::kDc), "DC rejects MRL");
  expect(!av2enc::mrl::mode_allows_mrl(Mode::kSmooth), "smooth rejects MRL");

  std::uint8_t out[64] = {};
  Params dc = params;
  dc.mode = Mode::kDc;
  expect(av2enc::mrl::predict_block(block, 8, dc, out, 8) ==
             MrlStatus::kInvalidArgument,
         "predict rejects non-directional MRL");

  Params bad_index = params;
  bad_index.mrl_index = 4;
  expect(av2enc::mrl::predict_block(block, 8, bad_index, out, 8) ==
             MrlStatus::kInvalidArgument,
         "predict rejects mrl_index 4");

  Params bad_delta = params;
  bad_delta.angle_delta = 4;
  expect(av2enc::mrl::predict_block(block, 8, bad_delta, out, 8) ==
             MrlStatus::kInvalidArgument,
         "predict rejects angle_delta 4");

  Params sec_zero = params;
  sec_zero.mrl_index = 0;
  sec_zero.mrl_sec_index = true;
  expect(av2enc::mrl::predict_block(block, 8, sec_zero, out, 8) ==
             MrlStatus::kInvalidArgument,
         "secondary line requires mrl_index > 0");

  Params tiny = params;
  tiny.mrl_sec_index = true;
  Block tiny_block = block;
  tiny_block.w = 4;
  tiny_block.h = 4;
  expect(av2enc::mrl::predict_block(tiny_block, 8, tiny, out, 4) ==
             MrlStatus::kInvalidArgument,
         "secondary line unavailable at 4x4");

  Block chroma = block;
  chroma.plane = 1;
  expect(av2enc::mrl::predict_block(chroma, 8, params, out, 8) ==
             MrlStatus::kInvalidArgument,
         "MRL is luma-only");
}

}  // namespace

int main() {
  availability_tests();

  const std::vector<Geometry> geo = geometries();
  // Targeted full sweep of the angle/index space on representative geometry.
  const Geometry representative[] = {{0, 0, 4, 4},   {4, 4, 8, 8},
                                     {8, 4, 16, 8},  {4, 8, 8, 16},
                                     {12, 20, 32, 32}, {32, 32, 64, 4},
                                     {4, 64, 4, 64}, {60, 60, 8, 8}};
  for (const Geometry& g : representative) {
    for (int mode : kModes) {
      for (int delta : kDeltas) {
        for (int mrl = 0; mrl <= 3; ++mrl) {
          for (int sec = 0; sec <= 1; ++sec) {
            if (mrl == 0 && sec != 0) continue;
            run_one(g, mode, delta, mrl, sec != 0, false, false);
            run_one(g, mode, delta, mrl, sec != 0, true, false);
            run_one(g, mode, delta, mrl, sec != 0, false, true);
          }
        }
      }
    }
  }

  // Random coverage over all sizes/edges.
  for (int trial = 0; trial < 6000; ++trial) {
    const Geometry& g = geo[static_cast<std::size_t>(
        random_range(0, static_cast<int>(geo.size()) - 1))];
    const int mode = kModes[random_range(0, 7)];
    const int delta = kDeltas[random_range(0, 6)];
    const int mrl = random_range(0, 3);
    const bool sec = mrl > 0 && random_range(0, 1) != 0;
    run_one(g, mode, delta, mrl, sec, random_range(0, 1) != 0,
            random_range(0, 1) != 0);
  }

  std::printf("mrl_test: %s (%d checks)\n",
              g_failures == 0 ? "PASS" : "FAIL", g_checks);
  return g_failures == 0 ? 0 : 1;
}
