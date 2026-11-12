// CPU reference tests for AV2 v1.0.0 §7.13.5 chroma-from-luma.
//
// The strongest check here is `test_random_vs_literal_reference`: a second,
// deliberately naive transcription of the spec pseudocode (literal filter
// tables, sequential prevSample chains, no shared helpers) is compared against
// the optimised library over randomised block geometry. Everything else pins
// the published constants, the sign/magnitude signalling ranges, and the
// edge/partial-block rules.
//
// Build:
//   ./src/encoder/cfl/build_and_test.sh
//   SANITIZE=1 ./src/encoder/cfl/build_and_test.sh

#include "cfl.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using av2enc::cfl::Alpha;
using av2enc::cfl::AlphaPair;
using av2enc::cfl::Block;
using av2enc::cfl::Scratch;
using av2enc::cfl::Signalling;
using av2enc::cfl::SourcePlane;

int g_checks = 0;
int g_failures = 0;

// Bounds-checked plane index (the test deliberately keeps all strides int).
std::size_t ix(int row, int col, int stride) {
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(stride) +
         static_cast<std::size_t>(col);
}

#define CHECK(cond)                                                       \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(cond)) {                                                        \
      ++g_failures;                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                     \
  } while (false)

#define CHECK_EQ(actual, expected)                                          \
  do {                                                                      \
    ++g_checks;                                                             \
    const long long actual_value = static_cast<long long>(actual);          \
    const long long expected_value = static_cast<long long>(expected);      \
    if (actual_value != expected_value) {                                   \
      ++g_failures;                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s = %lld, expected %lld\n",        \
                   __FILE__, __LINE__, #actual, actual_value,               \
                   expected_value);                                         \
    }                                                                       \
  } while (false)

// ---------------------------------------------------------------------------
// Normative tables, transcribed literally from §7.13.5 and §7.13.3.22.
// ---------------------------------------------------------------------------

constexpr int kRef420[3][3][3] = {
    {{0, 0, 0}, {0, 2, 2}, {0, 2, 2}},
    {{0, 0, 0}, {1, 2, 1}, {1, 2, 1}},
    {{0, 1, 0}, {1, 4, 1}, {0, 1, 0}},
};

constexpr int kRef422[3][3] = {
    {0, 4, 4},
    {2, 4, 2},
    {0, 8, 0},
};

constexpr int kRefDivLut[129] = {
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465, 462,
    458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 426, 423, 420, 417,
    415, 412, 410, 407, 405, 402, 400, 397, 395, 392, 390, 388, 386, 383, 381,
    379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358, 356, 354, 352, 350,
    349, 347, 345, 343, 341, 340, 338, 336, 334, 333, 331, 329, 328, 326, 324,
    323, 321, 320, 318, 317, 315, 314, 312, 311, 309, 308, 306, 305, 303, 302,
    301, 299, 298, 297, 295, 294, 293, 291, 290, 289, 287, 286, 285, 284, 282,
    281, 280, 279, 278, 277, 275, 274, 273, 272, 271, 270, 269, 267, 266, 265,
    264, 263, 262, 261, 260, 259, 258, 257, 256,
};

// ---------------------------------------------------------------------------
// Synthetic planes / blocks
// ---------------------------------------------------------------------------

struct Fixture {
  int luma_stride = 192;
  int chroma_stride = 128;
  std::vector<std::uint8_t> luma;
  std::vector<std::uint8_t> chroma_u;
  std::vector<std::uint8_t> chroma_v;
  Block u{};
  Block v{};

  Fixture(int chroma_grid_w, int chroma_grid_h, int block_w, int block_h,
          int x, int y)
      : luma(static_cast<std::size_t>(luma_stride) * 192, 128),
        chroma_u(static_cast<std::size_t>(chroma_stride) * 128, 128),
        chroma_v(static_cast<std::size_t>(chroma_stride) * 128, 128) {
    u.luma = luma.data();
    u.luma_stride = luma_stride;
    u.luma_max_x = 191;
    u.luma_max_y = 191;
    u.sb_top_y = ((y * 2) / 64) * 64;
    u.chroma = chroma_u.data();
    u.chroma_stride = chroma_stride;
    u.x = x;
    u.y = y;
    u.w = block_w;
    u.h = block_h;
    u.coded_width = chroma_grid_w;
    u.coded_height = chroma_grid_h;
    // AvailUChroma/AvailLChroma are block-level mi availability: for 4:2:0
    // the first chroma mi row/column (4 chroma samples) has no neighbour.
    u.have_above = y >= 4 ? 1 : 0;
    u.have_left = x >= 4 ? 1 : 0;
    u.bit_depth = 8;
    u.ds_filter = 0;
    u.sub_x = 1;
    u.sub_y = 1;
    v = u;
    v.chroma = chroma_v.data();
  }

  std::uint8_t& luma_at(int row, int col) {
    return luma[ix(row, col, luma_stride)];
  }
  std::uint8_t& u_at(int row, int col) {
    return chroma_u[ix(row, col, chroma_stride)];
  }
  std::uint8_t& v_at(int row, int col) {
    return chroma_v[ix(row, col, chroma_stride)];
  }

  SourcePlane source_u() const {
    return SourcePlane{chroma_u.data(), chroma_stride, 127, 127};
  }
  SourcePlane source_v() const {
    return SourcePlane{chroma_v.data(), chroma_stride, 127, 127};
  }
};

// Fill the luma plane with a deterministic smooth pattern around mid-grey so
// that L - lumaAvg stays small.
void fill_luma_pattern(Fixture& f, int seed) {
  for (int row = 0; row < 192; ++row) {
    for (int col = 0; col < 192; ++col) {
      const int value = 128 + ((row * 3 + col * 5 + seed * 7) % 61) - 30;
      f.luma_at(row, col) =
          static_cast<std::uint8_t>(std::clamp(value, 0, 255));
    }
  }
}

// ---------------------------------------------------------------------------
// Literal §7.13.5 transcription used only by the randomised cross-check.
// ---------------------------------------------------------------------------

struct LiteralOutput {
  int luma_avg = 0;
  std::vector<int> luma;  // L[i*w+j]
  std::vector<std::uint8_t> pred;
  int implicit_alpha = 0;
};

LiteralOutput literal_predict(const Block& b, int alpha_q8, bool want_pred) {
  const int sub_x = b.sub_x;
  const int sub_y = b.sub_y;
  const int w = b.w;
  const int h = b.h;
  int filter = b.ds_filter;
  if (filter == 3) {
    filter = 0;
  }
  const int x = b.x << sub_x;
  const int y = b.y << sub_y;
  const int step_w = w > 32 ? 2 : 1;
  const int step_h = h > 32 ? 2 : 1;
  const int init = 1 << (b.bit_depth - 1);

  auto tap = [&](int dy, int dx) {
    if (sub_x != 0 && sub_y != 0) {
      return kRef420[filter][dy + sub_y][dx + sub_x];
    }
    if (sub_x != 0) {
      return dy == 0 ? kRef422[filter][dx + sub_x] : 0;
    }
    return (dy == 0 && dx == 0) ? 8 : 0;
  };

  std::vector<int> luma_above(static_cast<std::size_t>(w), 0);
  std::vector<int> luma_left(static_cast<std::size_t>(h), 0);
  int luma_sum = 0;
  int luma_count = 0;

  if (b.have_above != 0) {
    int prev = init;
    for (int i = 0; i < w; ++i) {
      int t = 0;
      if (i >= (b.coded_width - b.x)) {
        t = prev;
      } else {
        for (int dy = -sub_y; dy <= sub_y; ++dy) {
          const int row = std::max(b.sb_top_y - 1, y - 1 - sub_y + dy);
          for (int dx = -sub_x; dx <= sub_x; ++dx) {
            const int col = x + std::max(0, (i << sub_x) + dx);
            const int v = b.luma[ix(row, col, b.luma_stride)];
            t += tap(dy, dx) * v;
          }
        }
      }
      if (i % step_w == 0) {
        luma_sum += t;
      }
      luma_above[static_cast<std::size_t>(i)] = t >> 3;
      prev = t;
    }
    luma_count += w / step_w;
  }

  if (b.have_left != 0) {
    int prev = init;
    for (int j = 0; j < h; ++j) {
      int t = 0;
      if (j >= (b.coded_height - b.y)) {
        t = prev;
      } else {
        for (int dy = -sub_y; dy <= sub_y; ++dy) {
          const int row = y + std::max(0, (j << sub_y) + dy);
          for (int dx = -sub_x; dx <= sub_x; ++dx) {
            const int col = x - 1 - sub_x + dx;
            const int v = b.luma[ix(row, col, b.luma_stride)];
            t += tap(dy, dx) * v;
          }
        }
      }
      if (j % step_h == 0) {
        luma_sum += t;
      }
      luma_left[static_cast<std::size_t>(j)] = t >> 3;
      prev = t;
    }
    luma_count += h / step_h;
  }

  LiteralOutput out;
  out.luma_avg = 8 << (b.bit_depth - 1);
  if (luma_count > 0) {
    // Literal resolve_divisor(den) -> Round2(num * scale, shift).
    const int n = av2enc::cfl::floor_log2(luma_count);
    const int e = luma_count - (1 << n);
    const int f = (n > 7) ? av2enc::cfl::round2(e, n - 7) : (e << (7 - n));
    const int scale = kRefDivLut[f];
    const int shift = n + 9;
    const int average = av2enc::cfl::round2(luma_sum * scale, shift);
    out.luma_avg = std::min((8 << b.bit_depth) - 1, average);
  }

  out.luma.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  for (int i = 0; i < h; ++i) {
    const int luma_y = (b.y + i) << sub_y;
    const bool clamp_y = (i == 0) || (luma_y % 64 == 0);
    for (int j = 0; j < w; ++j) {
      const int luma_x = (b.x + j) << sub_x;
      const bool clamp_x = (j == 0) || (luma_x % 64 == 0);
      int t = 0;
      for (int dy = -sub_y; dy <= sub_y; ++dy) {
        const int row = luma_y + (clamp_y ? std::max(dy, 0) : dy);
        for (int dx = -sub_x; dx <= sub_x; ++dx) {
          const int col = luma_x + (clamp_x ? std::max(dx, 0) : dx);
          const int v = b.luma[ix(row, col, b.luma_stride)];
          t += tap(dy, dx) * v;
        }
      }
      out.luma[ix(i, j, w)] = t;
    }
  }

  // Derived alpha, literal transcription.
  int num_above = 0;
  int num_left = 0;
  if (b.have_above != 0 && b.have_left != 0) {
    if (w > h * 2) {
      num_above = av2enc::cfl::kNumRefSamCfl;
    } else if (h > w * 2) {
      num_left = av2enc::cfl::kNumRefSamCfl;
    } else {
      num_above = av2enc::cfl::kNumRefSamCfl >> 1;
      num_left = av2enc::cfl::kNumRefSamCfl >> 1;
    }
  } else {
    num_above = b.have_above != 0 ? av2enc::cfl::kNumRefSamCfl : 0;
    num_left = b.have_left != 0 ? av2enc::cfl::kNumRefSamCfl : 0;
  }
  num_above = std::min(num_above, w);
  num_left = std::min(num_left, h);

  int count = 0;
  std::int64_t sum_x = 0;
  std::int64_t sum_y = 0;
  std::int64_t sum_xy = 0;
  std::int64_t sum_xx = 0;
  std::vector<int> samples(static_cast<std::size_t>(std::max(w, h)), 0);
  if (num_above > 0) {
    const int step = w / num_above;
    int prev = init;
    for (int i = 0; i < w; ++i) {
      int sample = prev;
      if (i < b.coded_width - b.x) {
        sample = b.chroma[ix(b.y - 1, b.x + i, b.chroma_stride)];
      }
      samples[static_cast<std::size_t>(i)] = sample;
      prev = sample;
    }
    for (int i = step >> 1; i < w; i += step) {
      const int lx = luma_above[static_cast<std::size_t>(i)];
      const int sample = samples[static_cast<std::size_t>(i)];
      sum_x += lx;
      sum_y += sample;
      sum_xy += static_cast<std::int64_t>(lx) * sample;
      sum_xx += static_cast<std::int64_t>(lx) * lx;
      ++count;
    }
  }
  if (num_left > 0) {
    const int step = h / num_left;
    int prev = init;
    for (int j = 0; j < h; ++j) {
      int sample = prev;
      if (j < b.coded_height - b.y) {
        sample = b.chroma[ix(b.y + j, b.x - 1, b.chroma_stride)];
      }
      samples[static_cast<std::size_t>(j)] = sample;
      prev = sample;
    }
    for (int j = step >> 1; j < h; j += step) {
      const int lx = luma_left[static_cast<std::size_t>(j)];
      const int sample = samples[static_cast<std::size_t>(j)];
      sum_x += lx;
      sum_y += sample;
      sum_xy += static_cast<std::int64_t>(lx) * sample;
      sum_xx += static_cast<std::int64_t>(lx) * lx;
      ++count;
    }
  }
  if (count > 0) {
    const std::int64_t der =
        sum_xx - (sum_x * sum_x) / static_cast<std::int64_t>(count);
    const std::int64_t nor =
        sum_xy - (sum_x * sum_y) / static_cast<std::int64_t>(count);
    if (der != 0 && nor != 0) {
      out.implicit_alpha = av2enc::cfl::resolve_division(
          static_cast<int>(nor), static_cast<int>(der), 8);
    }
  }

  if (want_pred) {
    out.pred.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const int luma_value = out.luma[ix(i, j, w)];
        const int scaled = av2enc::cfl::round2_signed(
            alpha_q8 * (luma_value - out.luma_avg), av2enc::cfl::kScaleShift);
        const std::size_t index = ix(b.y + i, b.x + j, b.chroma_stride);
        const int dc = b.chroma[index];
        out.pred[ix(i, j, w)] =
            static_cast<std::uint8_t>(
                std::clamp(dc + scaled, 0, (1 << b.bit_depth) - 1));
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_math() {
  using namespace av2enc::cfl;
  CHECK_EQ(div_lut_entry(0), 512);
  for (int f = 0; f < kDivLutNum; ++f) {
    CHECK_EQ(div_lut_entry(f), kRefDivLut[f]);
  }

  CHECK_EQ(round2(3, 0), 3);
  CHECK_EQ(round2(0, 3), 0);
  CHECK_EQ(round2(1, 1), 1);
  CHECK_EQ(round2(2, 1), 1);
  CHECK_EQ(round2(3, 1), 2);
  CHECK_EQ(round2_signed(-3, 1), -2);
  CHECK_EQ(round2_signed(-1, 1), -1);
  CHECK_EQ(round2_signed(-2, 2), -1);
  CHECK_EQ(round2_signed(0, 11), 0);

  // Power-of-two divisors are exact through approx_divide.
  for (int den = 1; den <= 128; den <<= 1) {
    for (int num = 0; num <= 2040; num += 37) {
      CHECK_EQ(approx_divide(num * den, den), num);
    }
  }

  // resolve_division approximates (N << 8) / D. Bound the error over the
  // ranges the CfL correlation can produce.
  int worst = 0;
  for (int d = 1; d <= 4096; d += 7) {
    for (int n = 1; n <= 200000; n = n * 3 + 11) {
      const int got = resolve_division(n, d, 8);
      long long exact = (static_cast<long long>(n) << 8) / d;
      exact = std::min<long long>(exact, (2 << 8) - 1);
      const long long diff = std::llabs(got - exact);
      worst = std::max(worst, static_cast<int>(diff));
      CHECK(diff <= 3);
    }
  }
  CHECK(worst <= 3);
  CHECK_EQ(resolve_division(-1024, 512, 8), -resolve_division(1024, 512, 8));
  CHECK_EQ(resolve_division(1, 1 << 20, 8), 0);
  CHECK_EQ(resolve_division(1 << 20, 1, 8), (2 << 8) - 1);
}

void test_filters() {
  using namespace av2enc::cfl;
  for (int filter = 0; filter < 3; ++filter) {
    int sum420 = 0;
    int sum422 = 0;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        CHECK_EQ(ds_weight_420(filter, dy, dx),
                 kRef420[filter][dy + 1][dx + 1]);
        sum420 += ds_weight_420(filter, dy, dx);
      }
    }
    for (int dx = -1; dx <= 1; ++dx) {
      CHECK_EQ(ds_weight_422(filter, dx), kRef422[filter][dx + 1]);
      sum422 += kRef422[filter][dx + 1];
    }
    // Every row of both tables integrates to 8, so a constant luma plane maps
    // to itself (t == 8 * value).
    CHECK_EQ(sum420, 8);
    CHECK_EQ(sum422, 8);
    for (int dx = -1; dx <= 1; ++dx) {
      CHECK_EQ(ds_weight(filter, 0, dx, 1, 1),
               kRef420[filter][1][dx + 1]);
      CHECK_EQ(ds_weight(filter, 0, dx, 1, 0),
               kRef422[filter][dx + 1]);
      CHECK_EQ(ds_weight(filter, 0, dx, 0, 0), dx == 0 ? 8 : 0);
    }
  }
}

void test_signalling() {
  using namespace av2enc::cfl;

  // §6.19.7.25 Table 6.27.
  const int expected_sign_u[8] = {0, 0, 1, 1, 1, 2, 2, 2};
  const int expected_sign_v[8] = {1, 2, 0, 1, 2, 0, 1, 2};
  for (int signs = 0; signs < kJointSigns; ++signs) {
    CHECK_EQ((signs + 1) / 3, expected_sign_u[signs]);
    CHECK_EQ((signs + 1) % 3, expected_sign_v[signs]);
    Signalling s{};
    s.cfl_index = kIndexExplicit;
    s.alpha_signs = signs;
    s.alpha_u = 4;
    s.alpha_v = 6;
    const AlphaPair p = pair_from_signalling(s);
    CHECK_EQ(p.u.sign, expected_sign_u[signs]);
    CHECK_EQ(p.v.sign, expected_sign_v[signs]);
    CHECK_EQ(joint_signs(p), signs);
    CHECK(signalable(p));
    const Signalling back = explicit_signalling(p);
    CHECK_EQ(back.cfl_index, kIndexExplicit);
    CHECK_EQ(back.alpha_signs, signs);
    CHECK_EQ(back.alpha_u, p.u.sign == kSignZero ? 0 : s.alpha_u);
    CHECK_EQ(back.alpha_v, p.v.sign == kSignZero ? 0 : s.alpha_v);
  }

  // Explicit magnitude range: alphabet 0..7 maps to |alpha| 1..8.
  CHECK_EQ(kAlphabetSize, 8);
  for (int magnitude = 1; magnitude <= kAlphabetSize; ++magnitude) {
    CHECK_EQ(alpha_q8(Alpha{kSignPos, static_cast<std::int8_t>(magnitude)}),
             magnitude * 32);
    CHECK_EQ(alpha_q8(Alpha{kSignNeg, static_cast<std::int8_t>(magnitude)}),
             -magnitude * 32);
    Signalling s{};
    s.alpha_signs = 7;  // (pos, pos)
    s.alpha_u = magnitude - 1;
    s.alpha_v = magnitude - 1;
    CHECK_EQ(pair_from_signalling(s).u.magnitude, magnitude);
  }
  CHECK_EQ(alpha_q8(Alpha{kSignZero, 0}), 0);

  // Snapping: ties round away from zero, magnitude clamped to 8.
  CHECK_EQ(snap_alpha_q8(0).sign, kSignZero);
  CHECK_EQ(snap_alpha_q8(15).magnitude, 1);
  CHECK_EQ(snap_alpha_q8(16).magnitude, 1);
  CHECK_EQ(snap_alpha_q8(48).magnitude, 2);
  CHECK_EQ(snap_alpha_q8(-48).sign, kSignNeg);
  CHECK_EQ(snap_alpha_q8(-48).magnitude, 2);
  CHECK_EQ(snap_alpha_q8(511).magnitude, 8);
  CHECK_EQ(snap_alpha_q8(-511).magnitude, 8);

  // (zero, zero) is explicitly prohibited.
  CHECK(!signalable(AlphaPair{Alpha{kSignZero, 0}, Alpha{kSignZero, 0}}));
  CHECK_EQ(joint_signs(AlphaPair{Alpha{kSignZero, 0}, Alpha{kSignZero, 0}}),
           -1);
  CHECK(!signalable(AlphaPair{Alpha{kSignZero, 1}, Alpha{kSignZero, 0}}));
  CHECK(!signalable(AlphaPair{Alpha{kSignPos, 9}, Alpha{kSignZero, 0}}));
  CHECK(!signalable(AlphaPair{Alpha{7, 1}, Alpha{kSignZero, 0}}));

  const Signalling derived = derived_signalling();
  CHECK_EQ(derived.cfl_index, kIndexDerivedAlpha);
}

void test_explicit_prediction() {
  using namespace av2enc::cfl;
  // 8x8 chroma block at (0,0): luma 16x16 constant 100 -> L = 800.
  Fixture f(64, 64, 8, 8, 0, 0);
  for (int row = 0; row < 32; ++row) {
    for (int col = 0; col < 32; ++col) {
      f.luma_at(row, col) = 100;
    }
  }
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      f.u_at(i, j) = 100;
    }
  }
  Scratch scratch;
  // No neighbours: lumaAvg is the midpoint (1024) and L == 800, so a
  // positive alpha predicts below DC.
  compute_luma(f.u, scratch);
  CHECK_EQ(scratch.luma_avg, 1024);
  CHECK_EQ(scratch.luma[0], 800);
  const int alpha = alpha_q8(Alpha{kSignPos, 8});  // 256
  apply_prediction(f.u, alpha, scratch);
  const int expected = std::clamp(
      100 + round2_signed(256 * (800 - 1024), kScaleShift), 0, 255);
  CHECK_EQ(f.u_at(0, 0), expected);
  CHECK_EQ(f.u_at(7, 7), expected);
}

void test_constant_luma_no_change() {
  using namespace av2enc::cfl;
  // A constant reconstructed luma plane gives L == lumaAvg (up to the derived
  // average's approximation), so CfL must not move the DC prediction.
  Fixture f(64, 64, 16, 16, 16, 16);
  for (int row = 0; row < 96; ++row) {
    for (int col = 0; col < 96; ++col) {
      f.luma_at(row, col) = 77;
    }
  }
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      f.u_at(16 + i, 16 + j) = static_cast<std::uint8_t>(40 + (i + j) % 7);
    }
  }
  const std::vector<std::uint8_t> before = f.chroma_u;
  Scratch scratch;
  for (int magnitude = -8; magnitude <= 8; ++magnitude) {
    f.chroma_u = before;  // restore DC prediction
    const int alpha = magnitude * 32;
    compute_luma(f.u, scratch);
    CHECK(scratch.luma_avg >= 615 && scratch.luma_avg <= 617);
    apply_prediction(f.u, alpha, scratch);
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        CHECK_EQ(f.u_at(16 + i, 16 + j),
                 before[ix(16 + i, 16 + j, f.chroma_stride)]);
      }
    }
  }
}

void test_subsampling_variants() {
  using namespace av2enc::cfl;
  // Constant luma maps to itself for every chroma format because each filter
  // row integrates to 8. 4:2:2 and 4:4:4 are reference-only, but the shared
  // code path must still be arithmetically correct.
  for (int sub_x = 0; sub_x <= 1; ++sub_x) {
    for (int sub_y = 0; sub_y <= 1; ++sub_y) {
      if (sub_x == 1 && sub_y == 0) {
        continue;  // 4:2:2 uses the horizontal-only branch; covered below
      }
      Fixture f(64, 64, 8, 8, 0, 0);
      f.u.sub_x = sub_x;
      f.u.sub_y = sub_y;
      f.u.have_above = 0;
      f.u.have_left = 0;
      for (auto& value : f.luma) {
        value = 50;
      }
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
          f.u_at(i, j) = 80;
        }
      }
      Scratch scratch;
      compute_luma(f.u, scratch);
      CHECK_EQ(scratch.luma[0], 400);
      CHECK_EQ(scratch.luma_avg, 1024);
      apply_prediction(f.u, alpha_q8(Alpha{kSignPos, 1}), scratch);
      CHECK_EQ(f.u_at(0, 0), 80 + round2_signed(32 * (400 - 1024),
                                                kScaleShift));
    }
  }

  // 4:2:2 horizontal-only branch.
  Fixture f(64, 64, 8, 8, 0, 0);
  f.u.sub_x = 1;
  f.u.sub_y = 0;
  f.u.have_above = 0;
  f.u.have_left = 0;
  for (auto& value : f.luma) {
    value = 50;
  }
  Scratch scratch;
  compute_luma(f.u, scratch);
  CHECK_EQ(scratch.luma[0], 400);

  // cfl_ds_filter_index == 3 is specified to behave as 0.
  Fixture a(64, 64, 16, 16, 8, 8);
  fill_luma_pattern(a, 2);
  Fixture b = a;
  b.u.luma = b.luma.data();
  b.u.chroma = b.chroma_u.data();
  b.u.ds_filter = 3;
  Scratch sa;
  Scratch sb;
  compute_luma(a.u, sa);
  compute_luma(b.u, sb);
  CHECK_EQ(sa.luma_avg, sb.luma_avg);
  for (int i = 0; i < 16 * 16; ++i) {
    CHECK_EQ(sa.luma[i], sb.luma[i]);
  }
  for (int i = 0; i < 16; ++i) {
    CHECK_EQ(sa.luma_above[i], sb.luma_above[i]);
    CHECK_EQ(sa.luma_left[i], sb.luma_left[i]);
  }
}

void test_extreme_clipping() {
  using namespace av2enc::cfl;
  Fixture f(64, 64, 4, 4, 0, 0);
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      f.luma_at(row, col) = 255;
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      f.u_at(i, j) = 250;
      f.v_at(i, j) = 5;
    }
  }
  Scratch scratch;
  compute_luma(f.u, scratch);
  // L = 2040, lumaAvg = 1024 for a top-left block: a large positive alpha
  // clips at 255 and a large negative alpha clips at 0.
  apply_prediction(f.u, alpha_q8(Alpha{kSignPos, 8}), scratch);
  CHECK_EQ(f.u_at(0, 0), 255);
  apply_prediction(f.v, -alpha_q8(Alpha{kSignPos, 8}), scratch);
  CHECK_EQ(f.v_at(0, 0), 0);
}

void test_edge_and_partial_blocks() {
  using namespace av2enc::cfl;

  // ----- Right-edge replication in the above-line average ------------------
  Fixture f(8, 8, 8, 8, 0, 4);  // coded chroma grid 8 wide, 8-tall block
  for (int row = 0; row < 32; ++row) {
    for (int col = 0; col < 32; ++col) {
      f.luma_at(row, col) = static_cast<std::uint8_t>(row * 2 + (col % 3));
    }
  }
  Scratch scratch;
  compute_luma(f.u, scratch);
  // i >= coded_width - x == 8 -> never for w == 8. Shrink the grid instead:
  f.u.coded_width = 5;  // only i = 0..4 are valid
  compute_luma(f.u, scratch);
  for (int i = 5; i < 8; ++i) {
    CHECK_EQ(scratch.luma_above[i], scratch.luma_above[4]);
  }

  // ----- Bottom-edge replication in the left-column average ----------------
  Fixture g(8, 8, 8, 8, 4, 0);
  for (int row = 0; row < 32; ++row) {
    for (int col = 0; col < 32; ++col) {
      g.luma_at(row, col) = static_cast<std::uint8_t>(col * 2 + (row % 3));
    }
  }
  g.u.coded_height = 5;
  compute_luma(g.u, scratch);
  for (int j = 5; j < 8; ++j) {
    CHECK_EQ(scratch.luma_left[j], scratch.luma_left[4]);
  }

  // ----- Spec superblock-top clamp on the above window ---------------------
  // Block at chroma y = 32 -> luma y = 64 == sb_top_y. The above window would
  // reach luma row 61, but §7.13.5 clamps it to sbTop - 1 = 63.
  Fixture s(64, 64, 8, 8, 0, 32);
  for (int row = 0; row < 128; ++row) {
    for (int col = 0; col < 64; ++col) {
      s.luma_at(row, col) = 0;
    }
  }
  for (int col = 0; col < 64; ++col) {
    s.luma_at(63, col) = 200;
  }
  compute_luma(s.u, scratch);
  CHECK_EQ(s.u.sb_top_y, 64);
  for (int i = 0; i < 8; ++i) {
    CHECK_EQ(scratch.luma_above[i], 200);  // 8 * 200 >> 3
  }

  // ----- Partial right/bottom block: reads clamp, all samples written ------
  Fixture p(6, 6, 8, 8, 2, 2);  // block extends past the coded grid
  for (int row = 0; row < 64; ++row) {
    for (int col = 0; col < 64; ++col) {
      p.luma_at(row, col) = static_cast<std::uint8_t>((row * 7 + col * 11) & 0xFF);
    }
  }
  p.u.coded_width = 6;
  p.u.coded_height = 6;
  p.u.have_above = 0;  // y == 2 is still inside the first chroma mi row
  p.u.have_left = 0;
  compute_luma(p.u, scratch);
  const int partial_alpha = alpha_q8(Alpha{kSignPos, 3});
  const LiteralOutput partial_ref = literal_predict(p.u, partial_alpha, true);
  // L covers the whole coded block, including samples past the coded grid.
  for (int i = 0; i < 8 * 8; ++i) {
    CHECK_EQ(scratch.luma[i], partial_ref.luma[static_cast<std::size_t>(i)]);
  }
  apply_prediction(p.u, partial_alpha, scratch);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      CHECK_EQ(p.u_at(2 + i, 2 + j),
               partial_ref.pred[ix(i, j, 8)]);
    }
  }
}

void test_derived_alpha() {
  using namespace av2enc::cfl;
  // 8x8 block with above/left neighbours: build a linear relation between the
  // reconstructed luma above/left samples and the reconstructed chroma
  // neighbours, and check the recovered implicit alpha.
  for (int magnitude = 1; magnitude <= 8; ++magnitude) {
    Fixture f(64, 64, 8, 8, 8, 8);
    for (int row = 0; row < 64; ++row) {
      for (int col = 0; col < 64; ++col) {
        f.luma_at(row, col) =
            static_cast<std::uint8_t>(40 + (row * 5 + col * 7) % 180);
      }
    }
    Scratch scratch;
    compute_luma(f.u, scratch);
    // lumaAbove/lumaLeft are the §7.13.5 subsampled neighbour lines in pixel
    // units (t >> 3). A chroma neighbour with slope magnitude/8 relative to
    // luma must yield implicitAlpha == magnitude * 32 (Q8).
    const auto chroma_of = [&](int luma_value) {
      const int scaled = magnitude * (luma_value - 128);
      const int rounded =
          scaled >= 0 ? (scaled + 4) >> 3 : -((-scaled + 4) >> 3);
      return std::clamp(128 + rounded, 0, 255);
    };
    for (int i = 0; i < f.u.w; ++i) {
      f.u_at(f.u.y - 1, f.u.x + i) =
          static_cast<std::uint8_t>(chroma_of(scratch.luma_above[i]));
    }
    for (int j = 0; j < f.u.h; ++j) {
      f.u_at(f.u.y + j, f.u.x - 1) =
          static_cast<std::uint8_t>(chroma_of(scratch.luma_left[j]));
    }
    const int alpha = derive_implicit_alpha(f.u, scratch);
    CHECK(alpha >= magnitude * 32 - 4 && alpha <= magnitude * 32 + 4);
  }

  // Constant neighbours: zero variance -> implicitAlpha == 0.
  Fixture flat(64, 64, 8, 8, 8, 8);
  for (int row = 0; row < 64; ++row) {
    for (int col = 0; col < 64; ++col) {
      flat.luma_at(row, col) = 120;
    }
  }
  for (int i = 0; i < 64; ++i) {
    flat.u_at(7, i) = 90;
    flat.u_at(i, 7) = 90;
  }
  Scratch scratch;
  compute_luma(flat.u, scratch);
  CHECK_EQ(derive_implicit_alpha(flat.u, scratch), 0);

  // No neighbours at all: lumaAvg = 1024, implicitAlpha = 0.
  Fixture none(64, 64, 8, 8, 0, 0);
  compute_luma(none.u, scratch);
  CHECK_EQ(none.u.have_above, 0);
  CHECK_EQ(none.u.have_left, 0);
  CHECK_EQ(scratch.luma_avg, 1024);
  CHECK_EQ(derive_implicit_alpha(none.u, scratch), 0);
}

void test_alpha_search() {
  using namespace av2enc::cfl;
  for (int target = -8; target <= 8; ++target) {
    if (target == 0) {
      continue;
    }
    Fixture f(64, 64, 8, 8, 16, 16);
    fill_luma_pattern(f, target + 9);
    Scratch scratch;
    compute_luma(f.u, scratch);
    // Build both source planes from the same known alpha, then copy them into
    // the DC planes so the search can read the DC input without mutating it.
    for (int i = 0; i < 30; ++i) {
      for (int j = 0; j < 30; ++j) {
        f.u_at(16 + i, 16 + j) = 128;
        f.v_at(16 + i, 16 + j) = 128;
      }
    }
    const int alpha = target * 32;
    std::vector<std::uint8_t> src_u(f.chroma_u);
    std::vector<std::uint8_t> src_v(f.chroma_v);
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        const int luma_value = scratch.luma[i * 8 + j];
        const int delta =
            round2_signed(alpha * (luma_value - scratch.luma_avg), kScaleShift);
        const std::size_t index = ix(16 + i, 16 + j, f.chroma_stride);
        src_u[index] =
            static_cast<std::uint8_t>(std::clamp(128 + delta, 8, 247));
        src_v[index] =
            static_cast<std::uint8_t>(std::clamp(128 - delta, 8, 247));
      }
    }
    const SourcePlane search_source_u{src_u.data(), f.chroma_stride, 127, 127};
    const SourcePlane search_source_v{src_v.data(), f.chroma_stride, 127, 127};
    Scratch search_scratch;
    const auto found =
        search_alpha(f.u, f.v, search_source_u, search_source_v,
                     SearchMode::kExhaustiveSse, search_scratch);
    CHECK(found.has_value());
    if (!found.has_value()) {
      continue;
    }
    const SearchResult result = *found;
    CHECK_EQ(result.alpha.u.sign, target > 0 ? kSignPos : kSignNeg);
    CHECK_EQ(result.alpha.u.magnitude, std::abs(target));
    CHECK_EQ(result.alpha.v.sign, target > 0 ? kSignNeg : kSignPos);
    CHECK_EQ(result.alpha.v.magnitude, std::abs(target));
    CHECK_EQ(result.total_sse(), 0);
    CHECK_EQ(result.syntax.cfl_index, kIndexExplicit);
    CHECK_EQ(result.syntax.alpha_signs, joint_signs(result.alpha));
  }

  // Both planes prefer DC: the pair must still be signalable.
  Fixture flat(64, 64, 8, 8, 16, 16);
  fill_luma_pattern(flat, 1);
  for (int i = 0; i < 30; ++i) {
    for (int j = 0; j < 30; ++j) {
      flat.u_at(16 + i, 16 + j) = 100;
      flat.v_at(16 + i, 16 + j) = 100;
    }
  }
  Scratch scratch;
  const auto flat_result =
      search_alpha(flat.u, flat.v, flat.source_u(), flat.source_v(),
                   SearchMode::kExhaustiveSse, scratch);
  CHECK(flat_result.has_value());
  CHECK(signalable(flat_result->alpha));
  CHECK(flat_result->alpha.u.sign != kSignZero ||
        flat_result->alpha.v.sign != kSignZero);

  // Neighbour-least-squares mode does not need source planes.
  Fixture ls(64, 64, 8, 8, 8, 8);
  fill_luma_pattern(ls, 3);
  for (int i = 0; i < 64; ++i) {
    ls.u_at(7, i) = static_cast<std::uint8_t>(100 + i % 9);
    ls.u_at(i, 7) = static_cast<std::uint8_t>(100 + i % 9);
  }
  Scratch ls_scratch;
  const auto ls_result =
      search_alpha(ls.u, ls.v, SourcePlane{}, SourcePlane{},
                   SearchMode::kNeighbourLeastSquares, ls_scratch);
  CHECK(ls_result.has_value());
  if (ls_result.has_value()) {
    CHECK(signalable(ls_result->alpha));
  }

  // Least-squares mode snaps the derived slope to explicit units: build
  // above/left chroma with slope +2/8 relative to subsampled luma.
  Fixture ls2(64, 64, 8, 8, 8, 8);
  for (int row = 0; row < 64; ++row) {
    for (int col = 0; col < 64; ++col) {
      ls2.luma_at(row, col) =
          static_cast<std::uint8_t>(40 + (row * 5 + col * 7) % 180);
    }
  }
  Scratch ls2_scratch;
  compute_luma(ls2.u, ls2_scratch);
  const auto rounded_chroma = [](int luma_value, int magnitude) {
    const int scaled = magnitude * (luma_value - 128);
    const int rounded =
        scaled >= 0 ? (scaled + 4) >> 3 : -((-scaled + 4) >> 3);
    return std::clamp(128 + rounded, 0, 255);
  };
  for (int i = 0; i < 8; ++i) {
    const int value = rounded_chroma(ls2_scratch.luma_above[i], 2);
    ls2.u_at(ls2.u.y - 1, ls2.u.x + i) = static_cast<std::uint8_t>(value);
  }
  for (int j = 0; j < 8; ++j) {
    const int value = rounded_chroma(ls2_scratch.luma_left[j], 2);
    ls2.u_at(ls2.u.y + j, ls2.u.x - 1) = static_cast<std::uint8_t>(value);
  }
  const auto ls2_result =
      search_alpha(ls2.u, ls2.v, SourcePlane{}, SourcePlane{},
                   SearchMode::kNeighbourLeastSquares, ls2_scratch);
  CHECK(ls2_result.has_value());
  if (ls2_result.has_value()) {
    CHECK_EQ(ls2_result->alpha.u.sign, kSignPos);
    CHECK_EQ(ls2_result->alpha.u.magnitude, 2);
  }

  // Invalid input is rejected.
  CHECK(!search_alpha(ls.u, ls.v, SourcePlane{}, SourcePlane{},
                      SearchMode::kExhaustiveSse, ls_scratch)
             .has_value());
  Block bad = ls.u;
  bad.w = 65;
  CHECK(!search_alpha(bad, ls.v, ls.source_u(), ls.source_v(),
                      SearchMode::kExhaustiveSse, ls_scratch)
             .has_value());
}

void test_geometry_helpers() {
  using namespace av2enc::cfl;
  for (int y = 0; y < 40; ++y) {
    CHECK_EQ(cfl_avail_above(y, 1), y >= 4);
    CHECK_EQ(cfl_avail_above(y, 0), y >= 4);
  }
  for (int x = 0; x < 40; ++x) {
    CHECK_EQ(cfl_avail_left(x, 1), x >= 4);
    CHECK_EQ(cfl_avail_left(x, 0), x >= 4);
  }
  CHECK_EQ(cfl_sb_top_y(0, 1), 0);
  CHECK_EQ(cfl_sb_top_y(31, 1), 0);
  CHECK_EQ(cfl_sb_top_y(32, 1), 64);
  CHECK_EQ(cfl_sb_top_y(160, 1), 320);
  CHECK_EQ(cfl_sb_top_y(33, 0), 0);   // 4:4:4: luma y = 33 is in SB 0
  CHECK_EQ(cfl_sb_top_y(64, 0), 64);
}

void test_validation_and_allowed() {
  using namespace av2enc::cfl;
  Fixture f(64, 64, 8, 8, 0, 0);
  CHECK(valid(f.u));
  Block bad = f.u;
  bad.luma = nullptr;
  CHECK(!valid(bad));
  bad = f.u;
  bad.chroma = nullptr;
  CHECK(!valid(bad));
  bad = f.u;
  bad.bit_depth = 7;
  CHECK(!valid(bad));
  bad = f.u;
  bad.ds_filter = 4;
  CHECK(!valid(bad));
  bad = f.u;
  bad.have_above = 1;  // y == 0
  CHECK(!valid(bad));
  bad = f.u;
  bad.chroma_stride = f.u.x + f.u.w - 1;
  CHECK(!valid(bad));

  CHECK(cfl_allowed(32, 32, true, false, true, false, false));
  CHECK(cfl_allowed(64, 64, true, false, true, false, false));
  CHECK(!cfl_allowed(65, 64, true, false, true, false, false));
  CHECK(!cfl_allowed(32, 32, false, false, true, false, false));
  CHECK(!cfl_allowed(32, 32, true, true, true, false, false));
  CHECK(cfl_allowed(32, 32, true, true, true, true, false));
  CHECK(cfl_allowed(4, 4, true, false, true, false, true));
  CHECK(!cfl_allowed(8, 8, true, false, true, false, true));
}

void test_random_vs_literal_reference() {
  using namespace av2enc::cfl;
  std::mt19937 rng(0xCf1u);
  int cases = 0;
  int alpha_cases = 0;

  for (int iteration = 0; iteration < 400; ++iteration) {
    const int coded_w = 2 + static_cast<int>(rng() % 62);
    const int coded_h = 2 + static_cast<int>(rng() % 62);
    const int w = 1 + static_cast<int>(rng() % 16);
    const int h = 1 + static_cast<int>(rng() % 16);
    const int x = static_cast<int>(rng() % static_cast<unsigned>(coded_w));
    const int y = static_cast<int>(rng() % static_cast<unsigned>(coded_h));
    if (x + w > 128 || y + h > 128) {
      continue;
    }
    Fixture f(coded_w, coded_h, w, h, x, y);
    for (auto& value : f.luma) {
      value = static_cast<std::uint8_t>(rng() & 0xFF);
    }
    for (auto& value : f.chroma_u) {
      value = static_cast<std::uint8_t>(rng() & 0xFF);
    }
    for (auto& value : f.chroma_v) {
      value = static_cast<std::uint8_t>(rng() & 0xFF);
    }
    f.u.have_above = y >= 4 ? 1 : 0;
    f.u.have_left = x >= 4 ? 1 : 0;
    f.v.have_above = f.u.have_above;
    f.v.have_left = f.u.have_left;
    f.u.ds_filter = static_cast<int>(rng() % 4);
    f.v.ds_filter = f.u.ds_filter;
    f.u.sb_top_y = ((y * 2) / 64) * 64;
    f.v.sb_top_y = f.u.sb_top_y;

    const int alpha = (static_cast<int>(rng() % 17) - 8) * 32;
    Scratch scratch;
    compute_luma(f.u, scratch);

    const LiteralOutput reference = literal_predict(f.u, alpha, true);
    CHECK_EQ(scratch.luma_avg, reference.luma_avg);
    bool luma_ok = true;
    for (int i = 0; i < h * w; ++i) {
      if (scratch.luma[i] != reference.luma[static_cast<std::size_t>(i)]) {
        luma_ok = false;
      }
    }
    CHECK(luma_ok);
    CHECK_EQ(derive_implicit_alpha(f.u, scratch), reference.implicit_alpha);

    std::vector<std::uint8_t> predicted(f.chroma_u);
    apply_prediction(f.u, alpha, scratch);
    bool pred_ok = true;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        if (f.u_at(y + i, x + j) !=
            reference.pred[ix(i, j, w)]) {
          pred_ok = false;
        }
      }
    }
    CHECK(pred_ok);
    ++cases;

    if (alpha != 0) {
      ++alpha_cases;
    }
  }
  CHECK(cases > 300);
  CHECK(alpha_cases > 200);
}

void test_scratch_layout_and_pair_prediction() {
  using namespace av2enc::cfl;
  Fixture f(64, 64, 4, 4, 0, 0);
  fill_luma_pattern(f, 5);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      f.u_at(i, j) = 60;
      f.v_at(i, j) = 200;
    }
  }
  Scratch scratch;
  compute_luma(f.u, scratch);
  apply_prediction_pair(f.u, f.v, AlphaPair{Alpha{kSignPos, 2}, Alpha{kSignNeg, 3}},
                        scratch);
  // Independent per-sample recomputation.
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int luma_value = scratch.luma[i * 4 + j];
      const int du = round2_signed(2 * 32 * (luma_value - scratch.luma_avg),
                                   kScaleShift);
      const int dv = round2_signed(-3 * 32 * (luma_value - scratch.luma_avg),
                                   kScaleShift);
      CHECK_EQ(f.u_at(i, j), std::clamp(60 + du, 0, 255));
      CHECK_EQ(f.v_at(i, j), std::clamp(200 + dv, 0, 255));
    }
  }
}

}  // namespace

int main() {
  test_math();
  test_filters();
  test_signalling();
  test_explicit_prediction();
  test_constant_luma_no_change();
  test_subsampling_variants();
  test_extreme_clipping();
  test_edge_and_partial_blocks();
  test_derived_alpha();
  test_alpha_search();
  test_geometry_helpers();
  test_validation_and_allowed();
  test_random_vs_literal_reference();
  test_scratch_layout_and_pair_prediction();

  if (g_failures == 0) {
    std::printf("cfl_test: %d checks passed\n", g_checks);
    return 0;
  }
  std::fprintf(stderr, "cfl_test: %d/%d checks FAILED\n", g_failures, g_checks);
  return 1;
}
