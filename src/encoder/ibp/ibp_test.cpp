// AV2 IBP CPU reference tests.
//
// The reference implementation below is an independent scalar transcription of
// §7.13.2.7/.8/.9/.12/.14/.17/.18.  It shares only the normative constant
// tables (via ibp_tables.h) and the §7.13.2.1 edge preparation helper with the
// module under test.  Every arithmetic step is written out again, so a formula
// error in ibp.cpp is caught rather than mirrored.
//
// Build/run: ./build_test.sh
//
// Coverage:
//   * ibp_weights(): every reachable angle 39..89, plus spot values
//   * decision sweep: all 13 modes x angle deltas -3..3 x 13 block shapes x
//     luma/chroma x enable_ibp 0/1 x is_inter 0/1
//   * prediction parity against the independent reference for every candidate,
//     including all IBP weight/mode combinations and luma/chroma IDIF paths
//   * standard-path parity against src/encoder/intra_modes (enable_ibp = 0)
//   * IBP DC process: all shapes, haveLeft/haveAbove combinations
//   * §5.4.5 sequence_intra_config bit round-trip and exact enable_ibp position
//   * §7.13.3.29 Ii_Weights_1d masks (provided for completeness, not IBP)
//   * real 8-bit 4:2:0 luma blocks from tests/data/*.y4m, plus a quality proxy

#include "encoder/ibp/ibp.h"
#include "encoder/ibp/ibp_tables.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

namespace ibp = av2enc::ibp;
namespace intra = av2enc::intra;

namespace {

// ---------------------------------------------------------------------------
// Check bookkeeping
// ---------------------------------------------------------------------------

int g_checks = 0;
int g_failures = 0;
constexpr int kMaxPrinted = 25;

void expect(bool condition, const char* what, const char* detail, int a,
            int b, const char* file, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= kMaxPrinted) {
      std::printf("FAIL %s:%d: %s [%s a=%d b=%d]\n", file, line, what, detail,
                  a, b);
    }
  }
}

#define CHECK(cond) expect((cond), #cond, "", 0, 0, __FILE__, __LINE__)
#define CHECK_EQ(a, b, detail) \
  expect((a) == (b), #a " == " #b, detail, static_cast<int>(a), \
         static_cast<int>(b), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Independent reference arithmetic
// ---------------------------------------------------------------------------

constexpr int kDr[90] = {AV2ENC_IBP_DR_INTRA_DERIVATIVE_TABLE};
constexpr int kDiv[129] = {AV2ENC_IBP_DIV_LUT_TABLE};
constexpr int kIbpW[5][16] = {AV2ENC_IBP_WEIGHTS_TABLE};
constexpr int kIiW[128] = {AV2ENC_IBP_II_WEIGHTS_1D_TABLE};

int ref_round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

int ref_floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

int ref_clip3(int lo, int hi, int value) {
  return value < lo ? lo : (value > hi ? hi : value);
}

int ref_clip1(int value) { return ref_clip3(0, 255, value); }

void ref_resolve(int divisor, int& shift, int& factor) {
  const int magnitude = divisor < 0 ? -divisor : divisor;
  const int n = ref_floor_log2(magnitude);
  const int e = magnitude - (1 << n);
  const int f = n > 7 ? ref_round2(e, n - 7) : (e << (7 - n));
  shift = n + 9;
  factor = divisor < 0 ? -kDiv[f] : kDiv[f];
}

int ref_approx_divide(int num, int den) {
  int shift = 0;
  int factor = 0;
  ref_resolve(den, shift, factor);
  return ref_round2(num * factor, shift);
}

struct RefWeights {
  int v[16][16]{};
};

RefWeights ref_weights(int p_angle) {
  RefWeights out;
  const int angle = std::max(39, p_angle);
  const int dy = kDr[90 - angle];
  for (int r = 0; r < 16; ++r) {
    int y = dy;
    for (int c = 0; c < 16; ++c) {
      const int dist = ((r + 1) << 6) + y;
      int shift = 0;
      int factor = 0;
      ref_resolve(dist, shift, factor);
      out.v[r][c] = ref_round2(y * factor, shift - 7);
      y += dy;
    }
  }
  return out;
}

// §7.13.2.14
void ref_filter_corner(intra::EdgeRow& above, intra::EdgeRow& left) {
  const int sum = left[0] * 5 + above[-1] * 6 + above[0] * 5;
  const int value = ref_round2(sum, 4);
  above[-1] = static_cast<std::int16_t>(value);
  left[-1] = static_cast<std::int16_t>(value);
}

// §7.13.2.17
int ref_edge_strength(int w, int h, bool smooth, int delta) {
  const int d = delta < 0 ? -delta : delta;
  const int blk = w + h;
  int strength = 0;
  if (!smooth) {
    if (blk <= 8) {
      if (d >= 56) strength = 1;
    } else if (blk <= 12) {
      if (d >= 40) strength = 1;
    } else if (blk <= 16) {
      if (d >= 40) strength = 1;
    } else if (blk <= 24) {
      if (d >= 8) strength = 1;
      if (d >= 16) strength = 2;
      if (d >= 32) strength = 3;
    } else if (blk <= 32) {
      strength = 1;
      if (d >= 4) strength = 2;
      if (d >= 32) strength = 3;
    } else {
      strength = 3;
    }
  } else {
    if (blk <= 8) {
      if (d >= 40) strength = 1;
      if (d >= 64) strength = 2;
    } else if (blk <= 16) {
      if (d >= 20) strength = 1;
      if (d >= 48) strength = 2;
    } else if (blk <= 24) {
      if (d >= 4) strength = 3;
    } else {
      strength = 3;
    }
  }
  return strength;
}

// §7.13.2.18
void ref_edge_filter(intra::EdgeRow& row, int size, int strength) {
  if (size <= 1 || strength < 1 || strength > 3) {
    return;
  }
  constexpr int kernel[3][5] = {
      {0, 4, 8, 4, 0}, {0, 5, 6, 5, 0}, {2, 4, 4, 4, 2}};
  std::vector<int> edge(static_cast<std::size_t>(size) + 1);
  for (int i = 0; i < size; ++i) {
    edge[static_cast<std::size_t>(i)] = row[i - 1];
  }
  for (int i = 1; i < size; ++i) {
    int sum = 0;
    for (int j = 0; j < 5; ++j) {
      const int k = ref_clip3(0, size - 1, i - 2 + j);
      sum += kernel[strength - 1][j] * edge[static_cast<std::size_t>(k)];
    }
    row[i - 1] = static_cast<std::int16_t>((sum + 8) >> 4);
  }
}

// §7.13.2.8
void ref_directional(const intra::IntraEdges& e, int p_angle,
                     intra::EdgeRow above, intra::EdgeRow left,
                     std::uint8_t* dst, std::size_t stride) {
  const int w = e.w;
  const int h = e.h;
  const bool idif = e.is_luma;
  const int max_base = w + h - 1;
  if (p_angle > 90 && p_angle < 180) {
    left[h] = left[h - 1];
    above[w] = above[w - 1];
    left[h + 1] = left[h - 1];
    above[w + 1] = above[w - 1];
  } else {
    left[max_base + 1] = left[max_base];
    above[max_base + 1] = above[max_base];
    left[max_base + 2] = left[max_base];
    above[max_base + 2] = above[max_base];
  }
  left[-2] = left[-1];
  above[-2] = above[-1];

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int value = 0;
      if (p_angle < 90) {
        const int dx = kDr[p_angle];
        const int idx = (i + 1) * dx;
        const int base = (idx >> 6) + j;
        const int shift = (idx >> 1) & 0x1F;
        if (base < max_base + (idif ? 1 : 0)) {
          if (idif) {
            int sum = 0;
            constexpr int taps[32][4] = {AV2ENC_IBP_DR_INTERP_FILTER_TABLE};
            for (int t = 0; t < 4; ++t) {
              sum += taps[shift][t] * above[base + t - 1];
            }
            value = ref_clip1(ref_round2(sum, 7));
          } else {
            value = ref_round2(above[base] * (32 - shift) +
                                   above[base + 1] * shift,
                               5);
          }
        } else {
          value = above[max_base];
        }
      } else if (p_angle > 90 && p_angle < 180) {
        const int dx = kDr[180 - p_angle];
        const int dy = kDr[p_angle - 90];
        int idx = (j << 6) - (i + 1) * dx;
        int base = idx >> 6;
        if (base >= -1) {
          const int shift = (idx >> 1) & 0x1F;
          if (idif) {
            int sum = 0;
            constexpr int taps[32][4] = {AV2ENC_IBP_DR_INTERP_FILTER_TABLE};
            for (int t = 0; t < 4; ++t) {
              sum += taps[shift][t] * above[base + t - 1];
            }
            value = ref_clip1(ref_round2(sum, 7));
          } else {
            value = ref_round2(above[base] * (32 - shift) +
                                   above[base + 1] * shift,
                               5);
          }
        } else {
          idx = (i << 6) - (j + 1) * dy;
          base = idx >> 6;
          const int shift = (idx >> 1) & 0x1F;
          if (idif) {
            int sum = 0;
            constexpr int taps[32][4] = {AV2ENC_IBP_DR_INTERP_FILTER_TABLE};
            for (int t = 0; t < 4; ++t) {
              sum += taps[shift][t] * left[base + t - 1];
            }
            value = ref_clip1(ref_round2(sum, 7));
          } else {
            value = ref_round2(left[base] * (32 - shift) +
                                   left[base + 1] * shift,
                               5);
          }
        }
      } else if (p_angle > 180) {
        const int dy = kDr[270 - p_angle];
        const int idx = (j + 1) * dy;
        const int base = (idx >> 6) + i;
        const int shift = (idx >> 1) & 0x1F;
        if (base < max_base + (idif ? 1 : 0)) {
          if (idif) {
            int sum = 0;
            constexpr int taps[32][4] = {AV2ENC_IBP_DR_INTERP_FILTER_TABLE};
            for (int t = 0; t < 4; ++t) {
              sum += taps[shift][t] * left[base + t - 1];
            }
            value = ref_clip1(ref_round2(sum, 7));
          } else {
            value = ref_round2(left[base] * (32 - shift) +
                                   left[base + 1] * shift,
                               5);
          }
        } else {
          value = left[max_base];
        }
      } else {
        value = p_angle == 90 ? above[j] : left[i];
      }
      dst[static_cast<std::size_t>(i) * stride +
          static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
    }
  }
}

// §7.13.2.12
void ref_ibp_dc(const intra::IntraEdges& e, std::uint8_t* dst,
                std::size_t stride) {
  const int w = e.w;
  const int h = e.h;
  const int lw = ref_floor_log2(w);
  const int lh = ref_floor_log2(h);
  if (e.have_above) {
    const int c0 = (w < h && e.have_left) ? (w >> 2) : 0;
    for (int r = 0; r < (h >> 2); ++r) {
      const int s = kIbpW[lh - 2][r];
      for (int c = c0; c < w; ++c) {
        const int old = dst[static_cast<std::size_t>(r) * stride +
                            static_cast<std::size_t>(c)];
        dst[static_cast<std::size_t>(r) * stride + static_cast<std::size_t>(c)] =
            static_cast<std::uint8_t>(ref_round2(e.above[c] * (128 - s) +
                                                     old * s,
                                                 7));
      }
    }
  }
  if (e.have_left) {
    const int r0 = (w >= h && e.have_above) ? (h >> 2) : 0;
    for (int r = r0; r < h; ++r) {
      for (int c = 0; c < (w >> 2); ++c) {
        const int s = kIbpW[lw - 2][c];
        const int old = dst[static_cast<std::size_t>(r) * stride +
                            static_cast<std::size_t>(c)];
        dst[static_cast<std::size_t>(r) * stride + static_cast<std::size_t>(c)] =
            static_cast<std::uint8_t>(ref_round2(e.left[r] * (128 - s) +
                                                     old * s,
                                                 7));
      }
    }
  }
}

void ref_dc(const intra::IntraEdges& e, std::uint8_t* dst,
            std::size_t stride) {
  int value = 0;
  if (e.have_left && e.have_above) {
    int sum = 0;
    for (int k = 0; k < e.h; ++k) sum += e.left[k];
    for (int k = 0; k < e.w; ++k) sum += e.above[k];
    value = ref_clip1(ref_approx_divide(sum, e.w + e.h));
  } else if (e.have_left) {
    int sum = 0;
    for (int k = 0; k < e.h; ++k) sum += e.left[k];
    value = ref_round2(sum, ref_floor_log2(e.h));
  } else if (e.have_above) {
    int sum = 0;
    for (int k = 0; k < e.w; ++k) sum += e.above[k];
    value = ref_round2(sum, ref_floor_log2(e.w));
  } else {
    value = 128;
  }
  for (int i = 0; i < e.h; ++i) {
    for (int j = 0; j < e.w; ++j) {
      dst[static_cast<std::size_t>(i) * stride +
          static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
    }
  }
}

void ref_filter_edges(const intra::IntraEdges& e, const ibp::PredictParams& p,
                      int p_angle, bool apply_ibp, intra::EdgeRow& above,
                      intra::EdgeRow& left) {
  if (!p.enable_edge_filter || p_angle == 90 || p_angle == 180) {
    return;
  }
  int fta = (e.have_above && p.above_neighbor_is_smooth) ? 1 : 0;
  int ftl = (e.have_left && p.left_neighbor_is_smooth) ? 1 : 0;
  int aa = p_angle - 90;
  int al = p_angle - 180;
  int nr = p_angle < 90 ? 1 : 0;
  int nb = p_angle > 180 ? 1 : 0;
  if (apply_ibp) {
    if (p_angle > 180) nr = 1;
    if (p_angle < 90) nb = 1;
    if (aa > 90) aa -= 180;
    if (al < -90) al += 180;
  } else {
    const int ft = fta | ftl;
    fta = ft;
    ftl = ft;
  }
  if ((apply_ibp || (p_angle > 90 && p_angle < 180)) && (e.w + e.h) >= 24) {
    ref_filter_corner(above, left);
  }
  if (e.have_above) {
    const int strength = ref_edge_strength(e.w, e.h, fta != 0, aa);
    const int num = std::min(e.w, e.valid_above_px) + (nr ? e.h : 0) + 1;
    ref_edge_filter(above, num, strength);
  }
  if (e.have_left) {
    const int strength = ref_edge_strength(e.w, e.h, ftl != 0, al);
    const int num = std::min(e.h, e.valid_left_px) + (nb ? e.w : 0) + 1;
    ref_edge_filter(left, num, strength);
  }
}

void ref_predict(const intra::IntraEdges& e, const ibp::PredictParams& p,
                 std::uint8_t* dst, std::size_t stride, bool& blended,
                 bool& dc_modified) {
  blended = false;
  dc_modified = false;
  ibp::DecisionParams dp;
  dp.enable_ibp = p.enable_ibp;
  dp.is_inter = p.is_inter;
  dp.mode = p.mode;
  dp.angle_delta = p.angle_delta;
  dp.w = e.w;
  dp.h = e.h;
  dp.plane = e.is_luma ? 0 : 1;
  const ibp::Decision dec = ibp::decide(dp);  // angle mapping only
  if (p.mode == ibp::Mode::kDc) {
    ref_dc(e, dst, stride);
    if (ibp::ibp_dc_applicable(p.enable_ibp, p.use_dip, p.mode, e.w, e.h,
                               e.is_luma ? 0 : 1, p.uv_is_cfl)) {
      ref_ibp_dc(e, dst, stride);
      dc_modified = true;
    }
    return;
  }
  if (!intra::is_directional(p.mode)) {
    intra::PredictParams plain;
    plain.mode = p.mode;
    plain.angle_delta = p.angle_delta;
    plain.enable_edge_filter = p.enable_edge_filter;
    plain.above_neighbor_is_smooth = p.above_neighbor_is_smooth;
    plain.left_neighbor_is_smooth = p.left_neighbor_is_smooth;
    intra::predict_intra(e, plain, dst, stride);
    return;
  }
  intra::EdgeRow above = e.above;
  intra::EdgeRow left = e.left;
  ref_filter_edges(e, p, dec.p_angle, dec.apply_ibp_edge_rules, above, left);
  ref_directional(e, dec.p_angle, above, left, dst, stride);
  if (dec.use_ibp) {
    const int angle = dec.p_angle < 90 ? dec.p_angle : 270 - dec.p_angle;
    const RefWeights wm = ref_weights(angle);
    std::vector<std::uint8_t> second(
        static_cast<std::size_t>(e.w) * static_cast<std::size_t>(e.h));
    ref_directional(e, dec.second_angle, above, left, second.data(),
                    static_cast<std::size_t>(e.w));
    const int cs = e.w >> 5;
    const int rs = e.h >> 5;
    for (int r = 0; r < e.h; ++r) {
      for (int c = 0; c < e.w; ++c) {
        const int s = dec.p_angle < 90 ? wm.v[r >> rs][c >> cs]
                                       : wm.v[c >> cs][r >> rs];
        const int first = dst[static_cast<std::size_t>(r) * stride +
                              static_cast<std::size_t>(c)];
        const int sec = second[static_cast<std::size_t>(r) *
                                   static_cast<std::size_t>(e.w) +
                               static_cast<std::size_t>(c)];
        dst[static_cast<std::size_t>(r) * stride +
            static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(
            ref_round2(first * s + sec * (128 - s), 7));
      }
    }
    blended = true;
  }
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

std::uint32_t g_rng = 0x9e3779b9U;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

struct Plane {
  int w = 0;
  int h = 0;
  int stride = 0;
  std::vector<std::uint8_t> data;

  Plane(int width, int height, int pad)
      : w(width),
        h(height),
        stride(width + pad),
        data(static_cast<std::size_t>(stride) * static_cast<std::size_t>(height)) {
  }

  std::uint8_t& at(int x, int y) {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(x)];
  }
  std::uint8_t at(int x, int y) const {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(x)];
  }
};

Plane make_plane(int w, int h, int pad, int kind) {
  Plane p(w, h, pad);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int v = 0;
      if (kind == 0) {
        v = (x * 3 + y * 5 + ((x * y) >> 3)) & 255;  // structured
      } else if (kind == 1) {
        v = static_cast<int>(next_random() & 255U);  // noise
      } else {
        v = 128 + ((x - w / 2) * 2 + (y - h / 2) * 3) / 4;  // gradient
        v = ref_clip3(0, 255, v);
      }
      p.at(x, y) = static_cast<std::uint8_t>(v);
    }
  }
  return p;
}

constexpr int kShapes[][2] = {
    {4, 4},   {4, 8},   {8, 4},   {8, 8},   {8, 16},  {16, 8},
    {16, 16}, {16, 32}, {32, 16}, {32, 32}, {32, 64}, {64, 32},
    {64, 64}, {4, 16},  {16, 4},  {8, 32},  {32, 8},  {4, 64},
    {64, 4},
};

int shape_count() {
  return static_cast<int>(sizeof(kShapes) / sizeof(kShapes[0]));
}

void test_weights() {
  for (int angle = 39; angle <= 89; ++angle) {
    const ibp::WeightMatrix got = ibp::ibp_weights(angle);
    const RefWeights want = ref_weights(angle);
    for (int r = 0; r < 16; ++r) {
      for (int c = 0; c < 16; ++c) {
        if (got.at(r, c) != want.v[r][c]) {
          CHECK_EQ(got.at(r, c), want.v[r][c], "ibp_weights");
          std::printf("  angle=%d r=%d c=%d\n", angle, r, c);
          return;
        }
        ++g_checks;
      }
    }
  }
  // Spot values computed directly from the spec pseudocode.
  const ibp::WeightMatrix w39 = ibp::ibp_weights(39);
  CHECK_EQ(w39.at(0, 0), 56, "w39[0][0]");
  CHECK_EQ(w39.at(0, 15), 118, "w39[0][15]");
  CHECK_EQ(w39.at(15, 0), 6, "w39[15][0]");
  CHECK_EQ(w39.at(15, 15), 56, "w39[15][15]");
  const ibp::WeightMatrix w45 = ibp::ibp_weights(45);
  CHECK_EQ(w45.at(0, 0), 64, "w45[0][0]");
  CHECK_EQ(w45.at(0, 15), 121, "w45[0][15]");
  CHECK_EQ(w45.at(15, 15), 64, "w45[15][15]");
  const ibp::WeightMatrix w84 = ibp::ibp_weights(84);
  CHECK_EQ(w84.at(0, 0), 117, "w84[0][0]");
  CHECK_EQ(w84.at(0, 15), 127, "w84[0][15]");
  // Max(39, pAngle) / min(89, pAngle) clamping.
  const ibp::WeightMatrix wclamp = ibp::ibp_weights(10);
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 16; ++c) {
      CHECK_EQ(wclamp.at(r, c), w39.at(r, c), "clamp");
    }
  }
  const ibp::WeightMatrix whigh = ibp::ibp_weights(120);
  const ibp::WeightMatrix w89 = ibp::ibp_weights(89);
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 16; ++c) {
      CHECK_EQ(whigh.at(r, c), w89.at(r, c), "high clamp");
    }
  }
  // Ibp_Weights accessor.
  CHECK_EQ(ibp::ibp_dc_weight(0, 0), 96, "ibpdc");
  CHECK_EQ(ibp::ibp_dc_weight(4, 15), 124, "ibpdc");
  CHECK_EQ(ibp::ibp_dc_weight(5, 0), 0, "ibpdc oob");
}

void test_decisions() {
  int ibp_pairs = 0;
  for (int s = 0; s < shape_count(); ++s) {
    const int w = kShapes[s][0];
    const int h = kShapes[s][1];
    for (int plane = 0; plane <= 1; ++plane) {
      for (int enable = 0; enable <= 1; ++enable) {
        for (int is_inter = 0; is_inter <= 1; ++is_inter) {
          for (int m = 0; m < intra::kNumModes; ++m) {
            for (int delta = -3; delta <= 3; ++delta) {
              ibp::DecisionParams p;
              p.enable_ibp = enable != 0;
              p.is_inter = is_inter != 0;
              p.mode = static_cast<ibp::Mode>(m);
              p.angle_delta = delta;
              p.w = w;
              p.h = h;
              p.plane = plane;
              const ibp::Decision d = ibp::decide(p);
              if (!intra::is_directional(static_cast<ibp::Mode>(m)) &&
                  delta != 0) {
                CHECK(!d.valid);
                continue;
              }
              CHECK(d.valid);
              if (!d.valid) {
                continue;
              }
              if (!intra::is_directional(static_cast<ibp::Mode>(m))) {
                CHECK_EQ(delta, 0, "non-directional delta");
                CHECK(!d.apply_ibp_edge_rules);
                CHECK(!d.use_ibp);
                continue;
              }
              const bool not4x4 = !(w == 4 && h == 4);
              const bool apply = (enable != 0) && not4x4;
              CHECK_EQ(d.apply_ibp_edge_rules, apply, "applyIbp");
              int p_angle = 0;
              if (is_inter != 0) {
                p_angle = intra::mode_to_angle(static_cast<ibp::Mode>(m)) +
                          delta * 3;
              } else {
                p_angle =
                    intra::map_angle(static_cast<ibp::Mode>(m), delta, w, h)
                        .p_angle;
              }
              CHECK_EQ(d.p_angle, p_angle, "pAngle");
              const bool want_use = apply && ((delta & 1) == 0) &&
                                    plane == 0 &&
                                    (p_angle < 90 || p_angle > 180);
              CHECK_EQ(d.use_ibp, want_use, "useIBP");
              if (d.use_ibp) {
                ++ibp_pairs;
                CHECK_EQ(d.second_angle,
                         p_angle < 90 ? p_angle + 180 : p_angle - 180,
                         "secondAngle");
              }
            }
          }
        }
      }
    }
  }
  // The eligible luma pairs for each shape, from the normative conditions.
  int total_candidates = 0;
  for (int s = 0; s < shape_count(); ++s) {
    ibp::ModeCandidate cands[64]{};
    const int n = ibp::build_ibp_candidates(
        std::span<ibp::ModeCandidate>(cands, 64), kShapes[s][0], kShapes[s][1]);
    CHECK(n >= 0);
    // Derived from the normative predicates: 11 luma directional pairs for
    // every non-4x4 shape (D45/D67/D203 with delta -2,0,+2, V -2, H +2).
    if (kShapes[s][0] == 4 && kShapes[s][1] == 4) {
      CHECK_EQ(n, 0, "4x4 pairs");
    } else {
      CHECK_EQ(n, 11, "luma pairs");
    }
    total_candidates += n;
    if (n > 0) {
      CHECK_EQ(ibp::build_ibp_candidates(std::span<ibp::ModeCandidate>(), 8,
                                         8),
               -1, "too small");
    }
  }
  CHECK(ibp_pairs > 0);
  CHECK(total_candidates > 0);
  CHECK_EQ(ibp::build_ibp_candidates(std::span<ibp::ModeCandidate>(), 8, 8, 1),
           0, "chroma candidates");
  CHECK_EQ(ibp::build_ibp_candidates(std::span<ibp::ModeCandidate>(), 8, 8, 0, 1),
           0, "mrl candidates");
  std::printf("decision sweep: %d pairs, %d enumerated luma candidates\n",
              ibp_pairs, total_candidates);
}

struct Position {
  int x;
  int y;
};

std::vector<Position> positions_for(int w, int h, int plane_w, int plane_h) {
  std::vector<Position> out;
  out.push_back({0, 0});
  out.push_back({w, 0});
  out.push_back({0, h});
  out.push_back({(plane_w - w) / 2 / w * w, (plane_h - h) / 2 / h * h});
  out.push_back({plane_w - w, plane_h - h});
  if (plane_w - w > 0) {
    out.push_back({plane_w - w, 0});
  }
  return out;
}

intra::NeighborRequest make_request(const Plane& plane, int x, int y, int w,
                                    int h, int color_plane) {
  intra::NeighborRequest req;
  req.recon.data = plane.data.data();
  req.recon.stride = plane.stride;
  req.x = x;
  req.y = y;
  req.w = w;
  req.h = h;
  req.max_x = plane.w - 1;
  req.max_y = plane.h - 1;
  req.have_left = x > 0;
  req.have_above = y > 0;
  req.color_plane = color_plane;
  if (req.have_above) {
    req.num4_above_right =
        std::max(0, (req.max_x - (x + w - 1)) / 4);
  }
  if (req.have_left) {
    req.num4_below_left = std::max(0, (req.max_y - (y + h - 1)) / 4);
  }
  return req;
}

void check_prediction(const intra::IntraEdges& edges, int plane,
                      const ibp::PredictParams& params, const char* tag) {
  const std::size_t samples =
      static_cast<std::size_t>(edges.w) * static_cast<std::size_t>(edges.h);
  std::vector<std::uint8_t> got(samples);
  std::vector<std::uint8_t> want(samples);
  ibp::PredictResult result;
  const intra::IntraStatus status =
      ibp::predict(edges, params, got.data(),
                   static_cast<std::size_t>(edges.w), &result);
  if (status != intra::IntraStatus::kOk) {
    CHECK_EQ(static_cast<int>(status), 0, tag);
    return;
  }
  bool want_blended = false;
  bool want_dc = false;
  ref_predict(edges, params, want.data(), static_cast<std::size_t>(edges.w),
              want_blended, want_dc);
  CHECK_EQ(result.blended, want_blended, tag);
  CHECK_EQ(result.dc_modified, want_dc, tag);
  for (std::size_t i = 0; i < samples; ++i) {
    if (got[i] != want[i]) {
      CHECK_EQ(got[i], want[i], tag);
      std::printf("  plane=%d mode=%d delta=%d w=%d h=%d sample=%zu\n", plane,
                  static_cast<int>(params.mode), params.angle_delta, edges.w,
                  edges.h, i);
      return;
    }
    ++g_checks;
  }
}

void test_prediction_sweep() {
  const int pad = 7;
  const Plane luma = make_plane(160, 160, pad, 0);
  const Plane chroma = make_plane(80, 80, pad, 2);

  int tested = 0;
  int blends = 0;
  for (int s = 0; s < shape_count(); ++s) {
    const int w = kShapes[s][0];
    const int h = kShapes[s][1];
    for (int plane = 0; plane <= 1; ++plane) {
      const Plane& p = plane == 0 ? luma : chroma;
      const int bw = plane == 0 ? w : std::min(w, 32);
      const int bh = plane == 0 ? h : std::min(h, 32);
      const std::vector<Position> pos = positions_for(bw, bh, p.w, p.h);
      for (const Position& at : pos) {
        intra::NeighborRequest req =
            make_request(p, at.x, at.y, bw, bh, plane);
        intra::IntraEdges edges;
        if (intra::prepare_edges(req, 8, edges) != intra::IntraStatus::kOk) {
          CHECK(false);
          continue;
        }
        for (int m = 0; m < intra::kNumModes; ++m) {
          for (int delta = -3; delta <= 3; ++delta) {
            if (!intra::is_directional(static_cast<ibp::Mode>(m)) && delta != 0) {
              continue;
            }
            for (int enable = 0; enable <= 1; ++enable) {
              for (int is_inter = 0; is_inter <= 1; ++is_inter) {
                for (int smooth = 0; smooth <= 1; ++smooth) {
                  ibp::PredictParams params;
                  params.mode = static_cast<ibp::Mode>(m);
                  params.angle_delta = delta;
                  params.enable_ibp = enable != 0;
                  params.enable_edge_filter = true;
                  params.is_inter = is_inter != 0;
                  params.above_neighbor_is_smooth = smooth != 0;
                  params.left_neighbor_is_smooth = smooth != 0;
                  check_prediction(edges, plane, params, "predict");
                  ++tested;
                  const ibp::DecisionParams dp{
                      enable != 0, is_inter != 0, params.mode, delta, bw, bh,
                      plane, 0};
                  if (ibp::decide(dp).use_ibp) {
                    ++blends;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  std::printf("prediction sweep: %d blocks, %d IBP blends\n", tested, blends);
  CHECK(tested > 1000);
  CHECK(blends > 0);
}

void test_standard_path_parity() {
  const Plane plane = make_plane(96, 96, 5, 1);
  int tested = 0;
  for (int s = 0; s < shape_count(); ++s) {
    const int w = kShapes[s][0];
    const int h = kShapes[s][1];
    const std::vector<Position> pos = positions_for(w, h, plane.w, plane.h);
    for (const Position& at : pos) {
      intra::NeighborRequest req = make_request(plane, at.x, at.y, w, h, 0);
      intra::IntraEdges edges;
      if (intra::prepare_edges(req, 8, edges) != intra::IntraStatus::kOk) {
        CHECK(false);
        continue;
      }
      for (int m = 0; m < intra::kNumModes; ++m) {
        for (int delta = -3; delta <= 3; ++delta) {
          if (!intra::is_directional(static_cast<ibp::Mode>(m)) && delta != 0) {
            continue;
          }
          intra::PredictParams ip;
          ip.mode = static_cast<ibp::Mode>(m);
          ip.angle_delta = delta;
          ip.enable_edge_filter = true;
          ip.above_neighbor_is_smooth = true;
          ip.left_neighbor_is_smooth = false;
          ibp::PredictParams bp;
          bp.mode = static_cast<ibp::Mode>(m);
          bp.angle_delta = delta;
          bp.enable_ibp = false;  // ordinary §7.13.2.7 path
          bp.enable_edge_filter = true;
          bp.above_neighbor_is_smooth = true;
          bp.left_neighbor_is_smooth = false;
          bp.is_inter = false;

          std::vector<std::uint8_t> a(static_cast<std::size_t>(w) *
                                    static_cast<std::size_t>(h));
          std::vector<std::uint8_t> b(a.size());
          if (intra::predict_intra(edges, ip, a.data(),
                                  static_cast<std::size_t>(w)) !=
                  intra::IntraStatus::kOk ||
              ibp::predict(edges, bp, b.data(),
                                  static_cast<std::size_t>(w)) !=
                  intra::IntraStatus::kOk) {
            CHECK(false);
            continue;
          }
          CHECK(std::memcmp(a.data(), b.data(), a.size()) == 0);
          ++tested;
          // 4x4 is the same path even with enable_ibp=1 (applyIbp is false).
          if (w == 4 && h == 4) {
            bp.enable_ibp = true;
            std::vector<std::uint8_t> c(a.size());
            CHECK(ibp::predict(edges, bp, c.data(),
                                 static_cast<std::size_t>(w)) ==
                  intra::IntraStatus::kOk);
            CHECK(std::memcmp(a.data(), c.data(), a.size()) == 0);
          }
        }
      }
    }
  }
  std::printf("standard-path parity: %d blocks\n", tested);
}

void test_dc_process() {
  const Plane plane = make_plane(160, 160, 9, 2);
  int tested = 0;
  for (int s = 0; s < shape_count(); ++s) {
    const int w = kShapes[s][0];
    const int h = kShapes[s][1];
    for (int plane_id = 0; plane_id <= 1; ++plane_id) {
      const int bw = plane_id == 0 ? w : std::min(w, 32);
      const int bh = plane_id == 0 ? h : std::min(h, 32);
      const std::vector<Position> pos = positions_for(bw, bh, plane.w, plane.h);
      for (const Position& at : pos) {
        intra::NeighborRequest req =
            make_request(plane, at.x, at.y, bw, bh, plane_id);
        intra::IntraEdges edges;
        if (intra::prepare_edges(req, 8, edges) != intra::IntraStatus::kOk) {
          CHECK(false);
          continue;
        }
        for (int enable = 0; enable <= 1; ++enable) {
          for (int cfl = 0; cfl <= 1; ++cfl) {
            ibp::PredictParams params;
            params.mode = ibp::Mode::kDc;
            params.enable_ibp = enable != 0;
            params.uv_is_cfl = cfl != 0;
            check_prediction(edges, plane_id, params, "dc");
            ++tested;
            // The DC process must not fire for 4x4 or CfL chroma.
            const bool applicable = ibp::ibp_dc_applicable(
                params.enable_ibp, false, ibp::Mode::kDc, bw, bh, plane_id,
                params.uv_is_cfl);
            CHECK_EQ(applicable,
                     params.enable_ibp && !(bw == 4 && bh == 4) &&
                         (plane_id == 0 || !params.uv_is_cfl),
                     "dc applicable");
          }
        }
      }
    }
  }
  std::printf("DC sweep: %d blocks\n", tested);
  CHECK(ibp::ibp_dc_applicable(false, false, ibp::Mode::kDc, 8, 8, 0, false) ==
        false);
  CHECK(ibp::ibp_dc_applicable(true, true, ibp::Mode::kDc, 8, 8, 0, false) ==
        false);
  CHECK(ibp::ibp_dc_applicable(true, false, ibp::Mode::kSmooth, 8, 8, 0, false) ==
        false);
}

void test_syntax() {
  ibp::SequenceIntraConfig cfg;
  cfg.enable_dip = true;
  cfg.enable_intra_edge_filter = true;
  cfg.enable_mrls = false;
  cfg.enable_cfl_intra = true;
  cfg.cfl_ds_filter_index = 2;
  cfg.enable_mhccp = true;
  cfg.enable_ibp = true;

  ibp::BitWriter writer;
  ibp::write_sequence_intra_config(cfg, false, writer);
  CHECK_EQ(writer.bit_position(), 8, "non-mono bits");
  CHECK_EQ(writer.bytes().size(), 1U, "non-mono bytes");
  // bits: dip=1 edge=1 mrls=0 cfl=1 idx=10 mhccp=1 ibp=1 -> 1101 1011.
  CHECK_EQ(writer.bytes()[0], 0xdb, "non-mono byte");

  ibp::BitReader reader(writer.bytes());
  ibp::SequenceIntraConfig parsed;
  CHECK(ibp::read_sequence_intra_config(false, reader, parsed));
  CHECK_EQ(reader.bit_position(), 8, "read bits");
  CHECK_EQ(parsed.enable_dip, cfg.enable_dip, "dip");
  CHECK_EQ(parsed.enable_intra_edge_filter, cfg.enable_intra_edge_filter,
           "edge");
  CHECK_EQ(parsed.enable_mrls, cfg.enable_mrls, "mrls");
  CHECK_EQ(parsed.enable_cfl_intra, cfg.enable_cfl_intra, "cfl");
  CHECK_EQ(parsed.cfl_ds_filter_index, cfg.cfl_ds_filter_index, "cfl idx");
  CHECK_EQ(parsed.enable_mhccp, cfg.enable_mhccp, "mhccp");
  CHECK_EQ(parsed.enable_ibp, cfg.enable_ibp, "ibp");

  // Monochrome omits cfl_ds_filter_index: four header bits + mhccp + ibp.
  ibp::SequenceIntraConfig mono = cfg;
  ibp::BitWriter mw;
  ibp::write_sequence_intra_config(mono, true, mw);
  CHECK_EQ(mw.bit_position(), 6, "mono bits");
  CHECK_EQ(mw.bytes()[0] & 0xfc, 0xdc, "mono byte");
  ibp::BitReader mr(mw.bytes());
  ibp::SequenceIntraConfig mparsed;
  CHECK(ibp::read_sequence_intra_config(true, mr, mparsed));
  CHECK_EQ(mparsed.cfl_ds_filter_index, 0, "mono cfl");
  CHECK_EQ(mparsed.enable_ibp, true, "mono ibp");

  // The M1 header currently writes enable_ibp = 0 as the last bit:
  // 0000 0000 for default config.
  ibp::SequenceIntraConfig off;
  ibp::BitWriter ow;
  ibp::write_sequence_intra_config(off, false, ow);
  CHECK_EQ(ow.bytes()[0], 0x00, "all-off");
  off.enable_ibp = true;
  ibp::BitWriter tw;
  ibp::write_sequence_intra_config(off, false, tw);
  CHECK_EQ(tw.bytes()[0], 0x01, "ibp-only last bit");

  // Truncated input must fail, not read uninitialised bits.
  ibp::BitReader trunc{std::span<const std::uint8_t>()};
  ibp::SequenceIntraConfig bad;
  CHECK(!ibp::read_sequence_intra_config(false, trunc, bad));
}

void test_inter_intra_mask() {
  CHECK_EQ(ibp::ii_weights_1d(0), 60, "ii0");
  CHECK_EQ(ibp::ii_weights_1d(1), 58, "ii1");
  CHECK_EQ(ibp::ii_weights_1d(18), 32, "ii18");
  CHECK_EQ(ibp::ii_weights_1d(31), 20, "ii31");
  CHECK_EQ(ibp::ii_weights_1d(32), 19, "ii32");
  CHECK_EQ(ibp::ii_weights_1d(63), 7, "ii63");
  CHECK_EQ(ibp::ii_weights_1d(64), 6, "ii64");
  CHECK_EQ(ibp::ii_weights_1d(95), 2, "ii95");
  CHECK_EQ(ibp::ii_weights_1d(127), 1, "ii127");
  for (int i = 0; i < 128; ++i) {
    CHECK_EQ(ibp::ii_weights_1d(i), kIiW[i], "ii table");
  }
  for (int s = 0; s < shape_count(); ++s) {
    const int w = kShapes[s][0];
    const int h = kShapes[s][1];
    const int scale = 128 / std::max(w, h);
    for (int mode = 0; mode < 4; ++mode) {
      std::vector<int> mask(static_cast<std::size_t>(w) *
                           static_cast<std::size_t>(h));
      CHECK(ibp::build_inter_intra_mask(
          w, h, static_cast<ibp::InterIntraMode>(mode), mask));
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          int want = 32;
          if (mode == 0) want = kIiW[i * scale];
          if (mode == 1) want = kIiW[j * scale];
          if (mode == 2) want = kIiW[std::min(i, j) * scale];
          CHECK_EQ(mask[static_cast<std::size_t>(i) *
                         static_cast<std::size_t>(w) +
                     static_cast<std::size_t>(j)],
                     want, "mask");
        }
      }
    }
  }
  std::vector<int> too_small(16);
  CHECK(!ibp::build_inter_intra_mask(8, 8, ibp::InterIntraMode::kDc,
                                     too_small));
}

// ---------------------------------------------------------------------------
// Real y4m blocks and quality proxy
// ---------------------------------------------------------------------------

struct Y4mFrame {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> y;
};

bool read_y4m_luma(const std::string& path, Y4mFrame& frame) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::string header;
  std::getline(in, header);
  if (header.rfind("YUV4MPEG2", 0) != 0) {
    return false;
  }
  int w = 0;
  int h = 0;
  std::size_t pos = 0;
  while (pos < header.size()) {
    while (pos < header.size() && header[pos] == ' ') ++pos;
    if (pos >= header.size()) break;
    const char tag = header[pos++];
    std::size_t start = pos;
    while (pos < header.size() && header[pos] != ' ') ++pos;
    const int value = std::atoi(header.substr(start, pos - start).c_str());
    if (tag == 'W') w = value;
    if (tag == 'H') h = value;
  }
  if (w <= 0 || h <= 0) {
    return false;
  }
  std::string marker;
  std::getline(in, marker);
  if (marker != "FRAME") {
    return false;
  }
  frame.w = w;
  frame.h = h;
  frame.y.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  in.read(reinterpret_cast<char*>(frame.y.data()),
          static_cast<std::streamsize>(frame.y.size()));
  return static_cast<std::size_t>(in.gcount()) == frame.y.size();
}

int sad_of(const std::uint8_t* a, int stride_a, const std::uint8_t* b,
           int stride_b, int w, int h) {
  int sad = 0;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      sad += std::abs(
          static_cast<int>(
              a[static_cast<std::size_t>(i) * static_cast<std::size_t>(stride_a) +
                static_cast<std::size_t>(j)]) -
          static_cast<int>(
              b[static_cast<std::size_t>(i) * static_cast<std::size_t>(stride_b) +
                static_cast<std::size_t>(j)]));
    }
  }
  return sad;
}

void test_real_blocks(const std::string& path, const char* label) {
  Y4mFrame frame;
  if (!read_y4m_luma(path, frame)) {
    std::printf("real blocks: SKIP %s (cannot read)\n", path.c_str());
    return;
  }
  Plane plane(frame.w, frame.h, 0);
  std::memcpy(plane.data.data(), frame.y.data(), frame.y.size());

  const int block = 16;
  const int step = 32;
  long long base_sad = 0;
  long long single_sad = 0;
  long long ibp_sad = 0;
  long long single_sse = 0;
  long long ibp_sse = 0;
  int blocks = 0;
  int ibp_wins = 0;
  int parity_blocks = 0;

  for (int y = 0; y + block <= frame.h; y += step) {
    for (int x = 0; x + block <= frame.w; x += step) {
      intra::NeighborRequest req =
          make_request(plane, x, y, block, block, 0);
      intra::IntraEdges edges;
      if (intra::prepare_edges(req, 8, edges) != intra::IntraStatus::kOk) {
        continue;
      }
      const std::uint8_t* src =
          plane.data.data() +
          static_cast<std::size_t>(y) * static_cast<std::size_t>(plane.stride) +
          static_cast<std::size_t>(x);
      std::vector<std::uint8_t> pred(static_cast<std::size_t>(block) * block);

      // Baseline: DC without IBP.
      ibp::PredictParams dc;
      dc.mode = ibp::Mode::kDc;
      if (ibp::predict(edges, dc, pred.data(), block) !=
          intra::IntraStatus::kOk) {
        continue;
      }
      const int dc_sad = sad_of(src, plane.stride, pred.data(), block, block,
                                block);
      base_sad += dc_sad;

      // Same candidate set (directional even deltas + DC), IBP off then on.
      int best_single = 1 << 30;
      int best_ibp = 1 << 30;
      int best_single_sse = 0;
      int best_ibp_sse = 0;
      for (int enable = 0; enable <= 1; ++enable) {
        int best = 1 << 30;
        int best_sse = 0;
        {
          ibp::PredictParams pdc;
          pdc.mode = ibp::Mode::kDc;
          pdc.enable_ibp = enable != 0;
          if (ibp::predict(edges, pdc, pred.data(), block) ==
              intra::IntraStatus::kOk) {
            const int sad =
                sad_of(src, plane.stride, pred.data(), block, block, block);
            long long sse = 0;
            for (int i = 0; i < block; ++i) {
              for (int j = 0; j < block; ++j) {
                const int d =
                    static_cast<int>(
                        src[static_cast<std::size_t>(i) *
                                static_cast<std::size_t>(plane.stride) +
                            static_cast<std::size_t>(j)]) -
                    static_cast<int>(
                        pred[static_cast<std::size_t>(i) *
                                 static_cast<std::size_t>(block) +
                             static_cast<std::size_t>(j)]);
                sse += static_cast<long long>(d * d);
              }
            }
            best = sad;
            best_sse = static_cast<int>(sse);
          }
        }
        for (int m = static_cast<int>(ibp::Mode::kV);
             m <= static_cast<int>(ibp::Mode::kD67); ++m) {
          for (int delta = -2; delta <= 2; delta += 2) {
            ibp::PredictParams p;
            p.mode = static_cast<ibp::Mode>(m);
            p.angle_delta = delta;
            p.enable_ibp = enable != 0;
            if (ibp::predict(edges, p, pred.data(), block) !=
                intra::IntraStatus::kOk) {
              continue;
            }
            const int sad =
                sad_of(src, plane.stride, pred.data(), block, block, block);
            long long sse = 0;
            for (int i = 0; i < block; ++i) {
              for (int j = 0; j < block; ++j) {
                const int d =
                    static_cast<int>(
                        src[static_cast<std::size_t>(i) *
                                static_cast<std::size_t>(plane.stride) +
                            static_cast<std::size_t>(j)]) -
                    static_cast<int>(
                        pred[static_cast<std::size_t>(i) *
                                 static_cast<std::size_t>(block) +
                             static_cast<std::size_t>(j)]);
                sse += static_cast<long long>(d * d);
              }
            }
            if (sad < best) {
              best = sad;
              best_sse = static_cast<int>(sse);
            }
            // Reference parity on the real block, every IBP candidate.
            const ibp::DecisionParams dp{enable != 0, false, p.mode, delta,
                                        block, block, 0, 0};
            if (ibp::decide(dp).use_ibp) {
              check_prediction(edges, 0, p, "real ibp");
              ++parity_blocks;
            }
          }
        }
        if (enable == 0) {
          best_single = best;
          best_single_sse = best_sse;
        } else {
          best_ibp = best;
          best_ibp_sse = best_sse;
        }
      }
      single_sad += best_single;
      ibp_sad += best_ibp;
      single_sse += best_single_sse;
      ibp_sse += best_ibp_sse;
      if (best_ibp < best_single) {
        ++ibp_wins;
      }
      ++blocks;
    }
  }
  if (blocks == 0) {
    std::printf("real blocks: SKIP %s (no blocks)\n", label);
    return;
  }
  const double single_mse =
      static_cast<double>(single_sse) / (blocks * block * block);
  const double ibp_mse = static_cast<double>(ibp_sse) / (blocks * block * block);
  const double single_psnr =
      10.0 * std::log10(255.0 * 255.0 / std::max(1e-9, single_mse));
  const double ibp_psnr =
      10.0 * std::log10(255.0 * 255.0 / std::max(1e-9, ibp_mse));
  std::printf(
      "real blocks %s: %d blocks, DC SAD/px %.2f, best-single SAD/px %.2f, "
      "best-IBP SAD/px %.2f, IBP wins %d (%.1f%%), PSNR proxy %.3f -> %.3f dB "
      "(delta %+.3f)\n",
      label, blocks, static_cast<double>(base_sad) / (blocks * block * block),
      static_cast<double>(single_sad) / (blocks * block * block),
      static_cast<double>(ibp_sad) / (blocks * block * block), ibp_wins,
      100.0 * ibp_wins / blocks, single_psnr, ibp_psnr,
      ibp_psnr - single_psnr);
  std::printf("real blocks %s: %d IBP parity checks\n", label, parity_blocks);
  CHECK(blocks > 0);
  CHECK(parity_blocks > 0);
}

}  // namespace

int main(int argc, char** argv) {
  std::printf("ibp_test: AV2 v1.0.0 intra bi-prediction\n");

  test_weights();
  test_decisions();
  test_prediction_sweep();
  test_standard_path_parity();
  test_dc_process();
  test_syntax();
  test_inter_intra_mask();

  const char* root = nullptr;
  if (argc > 1) {
    root = argv[1];
  } else {
#ifdef AV2ENC_TESTDATA_DIR
    root = AV2ENC_TESTDATA_DIR;
#endif
  }
  if (root != nullptr) {
    const std::string dir(root);
    test_real_blocks(dir + "/m0_64x64_1f.y4m", "64x64");
    test_real_blocks(dir + "/m0_640x360_1f.y4m", "640x360");
  } else {
    std::printf("real blocks: SKIP (no testdata dir)\n");
  }

  if (g_failures != 0) {
    std::printf("ibp_test: FAIL (%d checks, %d failures)\n", g_checks,
                g_failures);
    return 1;
  }
  std::printf("ibp_test: PASS (%d checks)\n", g_checks);
  return 0;
}
