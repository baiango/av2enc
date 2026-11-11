// Unit tests for the AV2 intra prediction CPU reference.
//
// The tests are standalone (no gtest): build with build_test.sh and run.
// Coverage: normative tables and helpers, edge sample preparation (fillers,
// corner, clamping, chroma extension rule, padded strides), every predictor on
// synthetic flat/gradient/step blocks, boundary blocks, angle deltas and
// wide-angle mapping, edge filtering, and the RDO cost/candidate hooks.

#include "encoder/intra_modes/intra_modes.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using av2enc::intra::EdgeRow;
using av2enc::intra::IntraEdges;
using av2enc::intra::IntraStatus;
using av2enc::intra::Mode;
using av2enc::intra::ModeCandidate;
using av2enc::intra::MappedAngle;
using av2enc::intra::NeighborRequest;
using av2enc::intra::PlaneView;
using av2enc::intra::PredictionCost;
using av2enc::intra::PredictParams;

int g_checks = 0;
int g_failures = 0;

void fail(const char* file, int line, const char* expression) {
  ++g_failures;
  std::cerr << file << ':' << line << ": CHECK failed: " << expression << '\n';
}

#define CHECK(expression)                                        \
  do {                                                           \
    ++g_checks;                                                  \
    if (!(expression)) {                                         \
      ::fail(__FILE__, __LINE__, #expression);                   \
    }                                                            \
  } while (0)

#define CHECK_EQ(actual, expected)                                        \
  do {                                                                    \
    ++g_checks;                                                           \
    const long long actual_value = static_cast<long long>(actual);        \
    const long long expected_value = static_cast<long long>(expected);    \
    if (actual_value != expected_value) {                                 \
      ++g_failures;                                                       \
      std::cerr << __FILE__ << ':' << __LINE__                            \
                << ": CHECK_EQ failed: " #actual " = " << actual_value    \
                << ", " #expected " = " << expected_value << '\n';        \
    }                                                                     \
  } while (0)

long long status_code(IntraStatus status) {
  return static_cast<long long>(status);
}

bool ok(IntraStatus status) { return status == IntraStatus::kOk; }

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

struct Plane {
  int width = 0;
  int height = 0;
  int stride = 0;
  std::vector<std::uint8_t> data;

  Plane() = default;
  Plane(int w, int h, int padding = 0)
      : width(w),
        height(h),
        stride(w + padding),
        data(static_cast<std::size_t>(w + padding) *
                 static_cast<std::size_t>(h),
             0) {}

  std::uint8_t& at(int x, int y) {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(x)];
  }
  std::uint8_t at(int x, int y) const {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(x)];
  }
  PlaneView view() const { return PlaneView{data.data(), stride}; }
};

void fill(Plane& plane, int value) {
  for (int y = 0; y < plane.height; ++y) {
    for (int x = 0; x < plane.width; ++x) {
      plane.at(x, y) = static_cast<std::uint8_t>(value);
    }
  }
}

void fill_gradient(Plane& plane, int x_mul, int y_mul) {
  for (int y = 0; y < plane.height; ++y) {
    for (int x = 0; x < plane.width; ++x) {
      plane.at(x, y) =
          static_cast<std::uint8_t>((x * x_mul + y * y_mul + 17) & 0xFF);
    }
  }
}

NeighborRequest make_request(const Plane& plane, int x, int y, int w, int h,
                             bool have_left, bool have_above,
                             int num4_above_right = 0,
                             int num4_below_left = 0, int color_plane = 0) {
  NeighborRequest request;
  request.recon = plane.view();
  request.x = x;
  request.y = y;
  request.w = w;
  request.h = h;
  request.max_x = plane.width - 1;
  request.max_y = plane.height - 1;
  request.have_left = have_left;
  request.have_above = have_above;
  request.num4_above_right = num4_above_right;
  request.num4_below_left = num4_below_left;
  request.color_plane = color_plane;
  return request;
}

IntraEdges prepare_ok(const NeighborRequest& request, int bit_depth = 8) {
  IntraEdges edges;
  const IntraStatus status = av2enc::intra::prepare_edges(request, bit_depth,
                                                          edges);
  CHECK_EQ(status_code(status), status_code(IntraStatus::kOk));
  return edges;
}

void predict_ok(const IntraEdges& edges, const PredictParams& params,
                std::vector<std::uint8_t>& out) {
  out.assign(static_cast<std::size_t>(edges.w) *
                 static_cast<std::size_t>(edges.h),
             0);
  const IntraStatus status = av2enc::intra::predict_intra(
      edges, params, out.data(), static_cast<std::size_t>(edges.w));
  CHECK_EQ(status_code(status), status_code(IntraStatus::kOk));
}

std::vector<std::uint8_t> predict_vec(const IntraEdges& edges, Mode mode,
                                      int angle_delta = 0,
                                      bool enable_edge_filter = true,
                                      bool above_smooth = false,
                                      bool left_smooth = false) {
  PredictParams params;
  params.mode = mode;
  params.angle_delta = angle_delta;
  params.enable_edge_filter = enable_edge_filter;
  params.above_neighbor_is_smooth = above_smooth;
  params.left_neighbor_is_smooth = left_smooth;
  std::vector<std::uint8_t> out;
  predict_ok(edges, params, out);
  return out;
}

bool all_equal(const std::vector<std::uint8_t>& values, int expected) {
  return std::all_of(values.begin(), values.end(),
                     [expected](std::uint8_t value) {
                       return static_cast<int>(value) == expected;
                     });
}

bool same_edges(const IntraEdges& a, const IntraEdges& b) {
  for (int i = EdgeRow::kMinIndex; i <= EdgeRow::kMaxIndex; ++i) {
    if (a.above[i] != b.above[i] || a.left[i] != b.left[i]) {
      return false;
    }
  }
  return true;
}

IntraEdges manual_edges(int w, int h, int above_value_offset = 0) {
  IntraEdges edges;
  edges.w = w;
  edges.h = h;
  edges.bit_depth = 8;
  edges.have_left = true;
  edges.have_above = true;
  edges.is_luma = true;
  edges.valid_above_px = w;
  edges.valid_left_px = h;
  for (int i = 0; i < w + h; ++i) {
    edges.above[i] = static_cast<std::int16_t>(10 + i + above_value_offset);
    edges.left[i] = static_cast<std::int16_t>(200 + i + above_value_offset);
  }
  edges.above[-1] = 5;
  edges.left[-1] = 5;
  return edges;
}

// ---------------------------------------------------------------------------
// §5.20.7.29 tables, mode mapping, wide-angle mapping
// ---------------------------------------------------------------------------

void test_mode_tables() {
  const std::array<int, av2enc::intra::kNumModes> expected = {
      0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0};
  for (int m = 0; m < av2enc::intra::kNumModes; ++m) {
    const Mode mode = static_cast<Mode>(m);
    CHECK_EQ(av2enc::intra::mode_to_angle(mode), expected[static_cast<std::size_t>(m)]);
    const bool directional = m >= 1 && m <= 8;
    CHECK_EQ(av2enc::intra::is_directional(mode), directional);
  }

  // No angle delta: pAngle is Mode_To_Angle.
  CHECK_EQ(av2enc::intra::map_angle(Mode::kV, 0, 8, 8).p_angle, 90);
  CHECK_EQ(av2enc::intra::map_angle(Mode::kH, 0, 8, 8).p_angle, 180);
  CHECK_EQ(av2enc::intra::map_angle(Mode::kD45, 0, 8, 8).p_angle, 45);

  // ANGLE_STEP = 3 degrees; +3 -> +9.
  CHECK_EQ(av2enc::intra::map_angle(Mode::kD45, 1, 8, 8).p_angle, 48);
  CHECK_EQ(av2enc::intra::map_angle(Mode::kD45, 3, 8, 8).p_angle, 54);
  CHECK_EQ(av2enc::intra::map_angle(Mode::kD45, -3, 8, 8).p_angle, 36);
  CHECK_EQ(av2enc::intra::map_angle(Mode::kV, 3, 8, 8).p_angle, 99);

  // Wide-angle mapping: h = 4w, pAngle < 73 -> D203 at 180 + pAngle.
  MappedAngle mapped = av2enc::intra::map_angle(Mode::kD45, 0, 4, 16);
  CHECK_EQ(static_cast<int>(mapped.mode), static_cast<int>(Mode::kD203));
  CHECK_EQ(mapped.p_angle, 225);
  mapped = av2enc::intra::map_angle(Mode::kD67, 0, 4, 16);
  CHECK_EQ(static_cast<int>(mapped.mode), static_cast<int>(Mode::kD203));
  CHECK_EQ(mapped.p_angle, 247);

  // w = 4h, pAngle > 197 -> D45 at pAngle - 180.
  mapped = av2enc::intra::map_angle(Mode::kD203, 0, 16, 4);
  CHECK_EQ(static_cast<int>(mapped.mode), static_cast<int>(Mode::kD45));
  CHECK_EQ(mapped.p_angle, 23);

  // No mapping for square blocks.
  mapped = av2enc::intra::map_angle(Mode::kD203, 0, 8, 8);
  CHECK_EQ(static_cast<int>(mapped.mode), static_cast<int>(Mode::kD203));
  CHECK_EQ(mapped.p_angle, 203);

  // Rectangular cases requested by the M1 audit. The same mapped pAngle is
  // used for both edge filtering and prediction (predict_intra maps once).
  mapped = av2enc::intra::map_angle(Mode::kD45, 0, 16, 64);
  CHECK_EQ(static_cast<int>(mapped.mode), static_cast<int>(Mode::kD203));
  CHECK_EQ(mapped.p_angle, 225);
  mapped = av2enc::intra::map_angle(Mode::kD203, 0, 64, 16);
  CHECK_EQ(static_cast<int>(mapped.mode), static_cast<int>(Mode::kD45));
  CHECK_EQ(mapped.p_angle, 23);
  mapped = av2enc::intra::map_angle(Mode::kD203, 0, 32, 8);
  CHECK_EQ(static_cast<int>(mapped.mode), static_cast<int>(Mode::kD45));
  CHECK_EQ(mapped.p_angle, 23);
  mapped = av2enc::intra::map_angle(Mode::kD45, 0, 8, 32);
  CHECK_EQ(static_cast<int>(mapped.mode), static_cast<int>(Mode::kD203));
  CHECK_EQ(mapped.p_angle, 225);
  mapped = av2enc::intra::map_angle(Mode::kD45, 0, 32, 64);
  CHECK_EQ(static_cast<int>(mapped.mode), static_cast<int>(Mode::kD203));
  CHECK_EQ(mapped.p_angle, 225);
  mapped = av2enc::intra::map_angle(Mode::kD67, 0, 32, 64);
  CHECK_EQ(static_cast<int>(mapped.mode), static_cast<int>(Mode::kD67));
  CHECK_EQ(mapped.p_angle, 67);
  mapped = av2enc::intra::map_angle(Mode::kD203, 0, 64, 32);
  CHECK_EQ(static_cast<int>(mapped.mode), static_cast<int>(Mode::kD203));
  CHECK_EQ(mapped.p_angle, 203);
  mapped = av2enc::intra::map_angle(Mode::kD203, 3, 64, 32);
  CHECK_EQ(static_cast<int>(mapped.mode), static_cast<int>(Mode::kD45));
  CHECK_EQ(mapped.p_angle, 32);
}

// ---------------------------------------------------------------------------
// §7.13.2.17 edge filter strength
// ---------------------------------------------------------------------------

void test_edge_filter_strength() {
  using av2enc::intra::edge_filter_strength;

  // filterType == 0.
  CHECK_EQ(edge_filter_strength(4, 4, false, 55), 0);
  CHECK_EQ(edge_filter_strength(4, 4, false, 56), 1);
  CHECK_EQ(edge_filter_strength(4, 8, false, 39), 0);
  CHECK_EQ(edge_filter_strength(4, 8, false, 40), 1);
  CHECK_EQ(edge_filter_strength(8, 8, false, 40), 1);
  CHECK_EQ(edge_filter_strength(8, 16, false, 7), 0);
  CHECK_EQ(edge_filter_strength(8, 16, false, 8), 1);
  CHECK_EQ(edge_filter_strength(8, 16, false, 16), 2);
  CHECK_EQ(edge_filter_strength(8, 16, false, 32), 3);
  CHECK_EQ(edge_filter_strength(16, 16, false, 0), 1);
  CHECK_EQ(edge_filter_strength(16, 16, false, 4), 2);
  CHECK_EQ(edge_filter_strength(16, 16, false, 31), 2);
  CHECK_EQ(edge_filter_strength(16, 16, false, 32), 3);
  CHECK_EQ(edge_filter_strength(8, 32, false, 0), 3);

  // Negative deltas use Abs(delta).
  CHECK_EQ(edge_filter_strength(8, 16, false, -32), 3);

  // filterType == 1 (smooth neighbour).
  CHECK_EQ(edge_filter_strength(4, 4, true, 39), 0);
  CHECK_EQ(edge_filter_strength(4, 4, true, 40), 1);
  CHECK_EQ(edge_filter_strength(4, 4, true, 64), 2);
  CHECK_EQ(edge_filter_strength(8, 8, true, 20), 1);
  CHECK_EQ(edge_filter_strength(8, 8, true, 48), 2);
  CHECK_EQ(edge_filter_strength(8, 16, true, 4), 3);
  CHECK_EQ(edge_filter_strength(16, 16, true, 0), 3);
}

// ---------------------------------------------------------------------------
// §7.13.2.18 edge filter: five taps on a snapshot of the original edge
// ---------------------------------------------------------------------------

void test_edge_filter_golden() {
  EdgeRow row;
  row.clear();
  row[-1] = 0;
  row[0] = 0;
  row[1] = 255;
  row[2] = 0;
  row[3] = 0;
  av2enc::intra::intra_edge_filter(row, 5, 1);
  // Computed by hand from the spec kernel {0,4,8,4,0} and the derived
  // snapshot edge[] (§7.13.2.18); later taps must not see filtered samples.
  CHECK_EQ(row[-1], 0);
  CHECK_EQ(row[0], 64);
  CHECK_EQ(row[1], 128);
  CHECK_EQ(row[2], 64);
  CHECK_EQ(row[3], 0);

  // Explicit snapshot reference for all three strengths on a longer row.
  constexpr int kKernels[3][5] = {{0, 4, 8, 4, 0},
                                  {0, 5, 6, 5, 0},
                                  {2, 4, 4, 4, 2}};
  constexpr int kSize = 8;
  EdgeRow input;
  input.clear();
  for (int i = -1; i < kSize - 1; ++i) {
    input[i] = static_cast<std::int16_t>(17 + 23 * (i + 1));
  }
  for (int strength = 1; strength <= 3; ++strength) {
    EdgeRow filtered = input;
    av2enc::intra::intra_edge_filter(filtered, kSize, strength);
    for (int i = 1; i < kSize; ++i) {
      int sum = 0;
      for (int j = 0; j < 5; ++j) {
        const int k = std::max(0, std::min(kSize - 1, i - 2 + j));
        sum += kKernels[strength - 1][j] * input[k - 1];
      }
      CHECK_EQ(filtered[i - 1], (sum + 8) >> 4);
    }
  }

  // Strength 0 and size 1 are no-ops.
  EdgeRow untouched = row;
  av2enc::intra::intra_edge_filter(untouched, 5, 0);
  for (int i = 0; i < 4; ++i) {
    CHECK_EQ(untouched[i], row[i]);
  }
  av2enc::intra::intra_edge_filter(untouched, 1, 3);
  for (int i = 0; i < 4; ++i) {
    CHECK_EQ(untouched[i], row[i]);
  }
}

// ---------------------------------------------------------------------------
// §7.13.2.1 edge preparation
// ---------------------------------------------------------------------------

void test_prepare_edges_fillers() {
  Plane plane(16, 16);
  fill_gradient(plane, 3, 5);

  // No neighbours: AboveRow = 127, LeftCol = 129, corner = 128 (8-bit).
  {
    const IntraEdges edges =
        prepare_ok(make_request(plane, 0, 0, 8, 8, false, false));
    for (int i = 0; i < 16; ++i) {
      CHECK_EQ(edges.above[i], 127);
      CHECK_EQ(edges.left[i], 129);
    }
    CHECK_EQ(edges.above[-1], 128);
    CHECK_EQ(edges.left[-1], 128);
    CHECK_EQ(edges.valid_above_px, 8);
    CHECK_EQ(edges.valid_left_px, 8);
    CHECK(edges.is_luma);
  }

  // Above only (top row, not left edge): left col repeats the above sample.
  {
    const IntraEdges edges =
        prepare_ok(make_request(plane, 0, 8, 8, 8, false, true));
    for (int i = 0; i < 16; ++i) {
      CHECK_EQ(edges.left[i], plane.at(0, 7));
      CHECK_EQ(edges.above[i], plane.at(std::min(7, i), 7));
    }
    CHECK_EQ(edges.above[-1], plane.at(0, 7));
    CHECK_EQ(edges.left[-1], plane.at(0, 7));
  }

  // Left only (left column, not top edge): above row repeats the left sample.
  {
    const IntraEdges edges =
        prepare_ok(make_request(plane, 8, 0, 8, 8, true, false));
    for (int i = 0; i < 16; ++i) {
      CHECK_EQ(edges.above[i], plane.at(7, 0));
      CHECK_EQ(edges.left[i], plane.at(7, std::min(7, i)));
    }
    CHECK_EQ(edges.above[-1], plane.at(7, 0));
    CHECK_EQ(edges.left[-1], plane.at(7, 0));
  }

  // Both: real top-left corner and unclamped samples.
  {
    const IntraEdges edges =
        prepare_ok(make_request(plane, 8, 8, 8, 8, true, true));
    for (int i = 0; i < 16; ++i) {
      CHECK_EQ(edges.above[i], plane.at(std::min(15, 8 + i), 7));
      CHECK_EQ(edges.left[i], plane.at(7, std::min(15, 8 + i)));
    }
    CHECK_EQ(edges.above[-1], plane.at(7, 7));
    CHECK_EQ(edges.left[-1], plane.at(7, 7));
  }
}

void test_prepare_edges_clamping() {
  Plane plane(32, 32);
  fill_gradient(plane, 7, 11);

  // Block hangs off the right and bottom edges: samples clamp to max_x/max_y.
  {
    const IntraEdges edges =
        prepare_ok(make_request(plane, 28, 28, 8, 8, true, true));
    for (int i = 0; i < 16; ++i) {
      CHECK_EQ(edges.above[i], plane.at(std::min(31, 28 + i), 27));
      CHECK_EQ(edges.left[i], plane.at(27, std::min(31, 28 + i)));
    }
    CHECK_EQ(edges.valid_above_px, 4);
    CHECK_EQ(edges.valid_left_px, 4);
  }

  // num4_above_right / num4_below_left extend the edge beyond the block.
  {
    const IntraEdges plain =
        prepare_ok(make_request(plane, 12, 12, 8, 8, true, true));
    const IntraEdges extended = prepare_ok(
        make_request(plane, 12, 12, 8, 8, true, true, 1, 1));
    CHECK_EQ(plain.above[8], plane.at(19, 11));
    CHECK_EQ(extended.above[8], plane.at(20, 11));
    CHECK_EQ(extended.above[11], plane.at(23, 11));
    CHECK_EQ(extended.above[12], plane.at(23, 11));
    CHECK_EQ(plain.left[8], plane.at(11, 19));
    CHECK_EQ(extended.left[8], plane.at(11, 20));
    CHECK_EQ(extended.left[11], plane.at(11, 23));
  }

  // §7.13.2.1: chroma with w > 32 / h > 32 forces the extension counts to 0.
  {
    Plane wide(160, 64);
    fill_gradient(wide, 1, 2);
    const IntraEdges chroma = prepare_ok(
        make_request(wide, 40, 40, 64, 4, true, true, 2, 0, 1));
    CHECK_EQ(chroma.above[64], wide.at(103, 39));
    CHECK_EQ(chroma.above[67], wide.at(103, 39));
    const IntraEdges luma = prepare_ok(
        make_request(wide, 40, 40, 64, 4, true, true, 2, 0, 0));
    CHECK_EQ(luma.above[64], wide.at(104, 39));
    CHECK_EQ(luma.above[67], wide.at(107, 39));
    CHECK(!chroma.is_luma);
  }

  // Padded stride: identical edges, and padding is never touched.
  {
    Plane tight(40, 40);
    fill_gradient(tight, 3, 9);
    Plane padded(40, 40, 24);
    std::fill(padded.data.begin(), padded.data.end(),
              static_cast<std::uint8_t>(0xEE));
    for (int y = 0; y < 40; ++y) {
      for (int x = 0; x < 40; ++x) {
        padded.at(x, y) = tight.at(x, y);
      }
    }
    const IntraEdges a =
        prepare_ok(make_request(tight, 20, 20, 8, 8, true, true));
    const IntraEdges b =
        prepare_ok(make_request(padded, 20, 20, 8, 8, true, true));
    CHECK(same_edges(a, b));
    for (int y = 0; y < padded.height; ++y) {
      for (int x = padded.width; x < padded.stride; ++x) {
        CHECK_EQ(padded.at(x, y), 0xEE);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Predictors
// ---------------------------------------------------------------------------

void test_extension_and_edge_filter_golden() {
  // 8x8 D45 at (8,8) with above-right extension (num4AboveRight = 2).
  // The above row is a step pattern; edge filtering is strength 1 for
  // w+h=16 and |pAngle-90| = 45. Expected values below are the spec's
  // sample-level outputs: filtered AboveRow first, then D45 projection with
  // Dr_Intra_Derivative[45] = 64 (shift = 0, IDIF row 0).
  Plane plane(32, 32);
  fill_gradient(plane, 1, 1);
  const int above_original[17] = {0, 0, 0, 0, 0, 255, 255, 255, 255,
                                  16, 16, 16, 16, 128, 128, 128, 128};
  // above_original[0] is AboveRow[-1] (the corner), [1..16] are AboveRow[0..15].
  plane.at(7, 7) = static_cast<std::uint8_t>(above_original[0]);
  for (int i = 1; i <= 16; ++i) {
    plane.at(7 + i, 7) = static_cast<std::uint8_t>(above_original[i]);
  }

  const IntraEdges edges =
      prepare_ok(make_request(plane, 8, 8, 8, 8, true, true, 2, 0));
  for (int i = 1; i <= 16; ++i) {
    CHECK_EQ(edges.above[i - 1], above_original[i]);
  }

  const int filtered_above[16] = {0,  0,  0,  64, 191, 255, 255, 195,
                                  76, 16, 16, 44, 100, 128, 128, 128};
  const std::vector<std::uint8_t> out =
      predict_vec(edges, Mode::kD45, 0, true);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      const int expected = filtered_above[i + 1 + j];
      CHECK_EQ(out[static_cast<std::size_t>(i) * 8 +
                   static_cast<std::size_t>(j)],
               expected);
    }
  }

  // Without edge filtering the projection uses the raw extension samples.
  const std::vector<std::uint8_t> raw =
      predict_vec(edges, Mode::kD45, 0, false);
  CHECK_EQ(raw[0 * 8 + 3], edges.above[4]);
  CHECK_EQ(raw[0 * 8 + 3], 255);
  CHECK_EQ(out[0 * 8 + 3], 191);
}

void test_below_left_extension_golden() {
  // 4x16 D45 at (8,8): wide-angle mapping turns 45 deg into 225 deg, so the
  // predictor reads LeftCol. num4BelowLeft = 3 supplies real below-left
  // samples at y = 24..27 with value 200; the first 16 left samples are 0.
  // Edge filtering is strength 3 for w+h=20 and |pAngle-180| = 45.
  Plane plane(64, 64);
  fill_gradient(plane, 1, 1);
  plane.at(7, 7) = 0;  // corner
  for (int i = 0; i < 16; ++i) {
    plane.at(7, 8 + i) = 0;
  }
  for (int i = 0; i < 4; ++i) {
    plane.at(7, 24 + i) = 200;
  }

  const IntraEdges edges =
      prepare_ok(make_request(plane, 8, 8, 4, 16, true, true, 0, 3));
  for (int i = 0; i < 16; ++i) {
    CHECK_EQ(edges.left[i], 0);
  }
  for (int i = 16; i < 20; ++i) {
    CHECK_EQ(edges.left[i], 200);
  }

  // Snapshot-filtered LeftCol[0..19].
  const int filtered_left[20] = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                 0,  0,  0,  0,  25, 75, 125, 175, 200, 200};
  const std::vector<std::uint8_t> out =
      predict_vec(edges, Mode::kD45, 0, true);
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 4; ++j) {
      // pAngle = 225, dy = Dr_Intra_Derivative[45] = 64, base = i + j + 1,
      // shift = 0 and IDIF tap row 0: pred = filtered LeftCol[base].
      const int expected = filtered_left[i + j + 1];
      CHECK_EQ(out[static_cast<std::size_t>(i) * 4 +
                   static_cast<std::size_t>(j)],
               expected);
    }
  }

  // Without the below-left extension, LeftCol[16..19] clamp to LeftCol[15]=0
  // and the same samples become 0 instead of 125/175/200.
  const IntraEdges clamped =
      prepare_ok(make_request(plane, 8, 8, 4, 16, true, true, 0, 0));
  const std::vector<std::uint8_t> clamped_out =
      predict_vec(clamped, Mode::kD45, 0, true);
  CHECK_EQ(clamped_out[15 * 4 + 0], 0);
  CHECK_EQ(out[15 * 4 + 0], filtered_left[16]);
  CHECK_EQ(out[15 * 4 + 0], 125);
}

void test_superblock_two_sb_border() {
  // 64x8 D45 at the top-left of an SB (x=0, y=64) with above-right extension
  // crossing into the next SB (num4AboveRight = 16). The above row is 0 for
  // x=0..63 and 200 for x=64..127. Edge filtering is strength 3.
  Plane plane(128, 128);
  fill_gradient(plane, 3, 2);
  for (int x = 0; x < 64; ++x) {
    plane.at(x, 63) = 0;
  }
  for (int x = 64; x < 128; ++x) {
    plane.at(x, 63) = 200;
  }

  const IntraEdges edges =
      prepare_ok(make_request(plane, 0, 64, 64, 8, false, true, 16, 0));
  // AboveRow[64..71] must come from the next superblock's above row.
  CHECK_EQ(edges.above[63], 0);
  CHECK_EQ(edges.above[64], 200);
  CHECK_EQ(edges.above[71], 200);
  CHECK_EQ(edges.above[-1], 0);  // above-only corner: AboveRow[0]

  const std::vector<std::uint8_t> out =
      predict_vec(edges, Mode::kD45, 0, true);
  // filtered AboveRow[63] = 75, AboveRow[64] = 125 (hand-computed from the
  // strength-3 kernel {2,4,4,4,2} with the snapshotted step).
  CHECK_EQ(out[0 * 64 + 62], 75);   // base = 63
  CHECK_EQ(out[0 * 64 + 63], 125);  // base = 64, next-SB sample

  // Restricting the extension to the current SB (num4AboveRight = 0 for a
  // 64-wide block) clamps AboveRow[64] to x=63 (value 0).
  const IntraEdges sb_local =
      prepare_ok(make_request(plane, 0, 64, 64, 8, false, true, 0, 0));
  CHECK_EQ(sb_local.above[64], 0);
  const std::vector<std::uint8_t> sb_local_out =
      predict_vec(sb_local, Mode::kD45, 0, true);
  CHECK_EQ(sb_local_out[0 * 64 + 63], 0);
}

void test_sb_32x8_above_right_golden() {
  // 32x8 D45 at the top-left of an SB (x=192, y=64) with above-right
  // extension into the other half of the same SB (num4AboveRight = 8).
  // Above row: x=192..223 is 16, x=224..255 is 200. Edge filtering is
  // strength 3 (w+h=40); hand-computed filtered values at the step.
  Plane plane(256, 128);
  fill(plane, 16);
  for (int x = 192; x < 224; ++x) {
    plane.at(x, 63) = 16;
  }
  for (int x = 224; x < 256; ++x) {
    plane.at(x, 63) = 200;
  }
  const IntraEdges edges =
      prepare_ok(make_request(plane, 192, 64, 32, 8, true, true, 8, 0));
  CHECK_EQ(edges.above[31], 16);
  CHECK_EQ(edges.above[32], 200);
  CHECK_EQ(edges.above[39], 200);

  const std::vector<std::uint8_t> out =
      predict_vec(edges, Mode::kD45, 0, true);
  CHECK_EQ(out[0 * 32 + 30], 85);   // base = 31
  CHECK_EQ(out[0 * 32 + 31], 131);  // base = 32, above-right sample

  const IntraEdges sb_local =
      prepare_ok(make_request(plane, 192, 64, 32, 8, true, true, 0, 0));
  CHECK_EQ(sb_local.above[32], 16);
  const std::vector<std::uint8_t> sb_local_out =
      predict_vec(sb_local, Mode::kD45, 0, true);
  CHECK_EQ(sb_local_out[0 * 32 + 31], 16);
}

void test_sb_border_below_left_clamp() {
  // A 64x64 block at the top of an SB cannot extend below-left into the next
  // SB row: the spec's count_bottom_left_avail stops at BlockDecoded[16][-1]
  // = 0, so the correct num4BelowLeft is 0. LeftCol[64..127] must clamp to
  // the last available left sample (y + h - 1).
  Plane plane(192, 256);
  fill_gradient(plane, 5, 3);
  const IntraEdges edges =
      prepare_ok(make_request(plane, 64, 64, 64, 64, true, true, 0, 0));
  for (int i = 64; i < 128; ++i) {
    CHECK_EQ(edges.left[i], plane.at(63, 127));
  }
}

void test_dc_prediction() {
  Plane flat(32, 32);
  fill(flat, 100);

  // Both neighbours, left only, above only, none.
  CHECK(all_equal(predict_vec(prepare_ok(make_request(flat, 8, 8, 8, 8, true,
                                                      true)),
                              Mode::kDc),
                  100));
  CHECK(all_equal(predict_vec(prepare_ok(make_request(flat, 8, 0, 8, 8, true,
                                                      false)),
                              Mode::kDc),
                  100));
  CHECK(all_equal(predict_vec(prepare_ok(make_request(flat, 0, 8, 8, 8, false,
                                                      true)),
                              Mode::kDc),
                  100));
  CHECK(all_equal(predict_vec(prepare_ok(make_request(flat, 0, 0, 8, 8, false,
                                                      false)),
                              Mode::kDc),
                  128));

  // Hand-computed approx_divide: left {10,20,30,40}, above {50,60,70,80},
  // sum = 360, denominator 8 -> 45.
  {
    IntraEdges edges = manual_edges(4, 4);
    for (int i = 0; i < 4; ++i) {
      edges.left[i] = static_cast<std::int16_t>(10 * (i + 1));
      edges.above[i] = static_cast<std::int16_t>(50 + 10 * i);
    }
    CHECK(all_equal(predict_vec(edges, Mode::kDc), 45));
  }

  // 64x64 both-neighbour DC is a power-of-two denominator; verify the sum.
  {
    Plane gradient(96, 96);
    fill_gradient(gradient, 2, 3);
    const IntraEdges edges =
        prepare_ok(make_request(gradient, 16, 16, 64, 64, true, true));
    int sum = 0;
    for (int i = 0; i < 64; ++i) {
      sum += static_cast<int>(gradient.at(std::min(95, 15 + i), 15));
      sum += static_cast<int>(gradient.at(15, std::min(95, 15 + i)));
    }
    const int expected = (sum + 64) >> 7;
    CHECK(all_equal(predict_vec(edges, Mode::kDc), expected));
  }
}

void test_flat_all_modes_and_sizes() {
  Plane flat(96, 96);
  fill(flat, 100);

  const std::array<std::pair<int, int>, 9> sizes = {
      std::pair{4, 4},   std::pair{4, 8},   std::pair{8, 4},
      std::pair{8, 8},   std::pair{16, 16}, std::pair{32, 8},
      std::pair{8, 32},  std::pair{32, 32}, std::pair{64, 64}};

  std::array<ModeCandidate, av2enc::intra::kMaxModeCandidates> candidates{};
  const int count = av2enc::intra::build_mode_candidates(candidates, true);
  CHECK_EQ(count, av2enc::intra::kMaxModeCandidates);

  for (const auto& size : sizes) {
    const int w = size.first;
    const int h = size.second;
    const IntraEdges edges =
        prepare_ok(make_request(flat, 16, 16, w, h, true, true));
    for (int c = 0; c < count; ++c) {
      PredictParams params;
      params.mode = candidates[static_cast<std::size_t>(c)].mode;
      params.angle_delta = candidates[static_cast<std::size_t>(c)].angle_delta;
      std::vector<std::uint8_t> out;
      predict_ok(edges, params, out);
      if (!all_equal(out, 100)) {
        ++g_failures;
        std::cerr << "flat prediction failed: " << w << 'x' << h << " mode "
                  << static_cast<int>(params.mode) << " delta "
                  << params.angle_delta << '\n';
      }
      ++g_checks;
    }
  }
}

void test_v_and_h_exact() {
  Plane plane(48, 48);
  fill_gradient(plane, 9, 4);
  const IntraEdges edges =
      prepare_ok(make_request(plane, 16, 16, 16, 8, true, true));

  const std::vector<std::uint8_t> v = predict_vec(edges, Mode::kV);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 16; ++j) {
      CHECK_EQ(v[static_cast<std::size_t>(i) * 16 +
                 static_cast<std::size_t>(j)],
               edges.above[j]);
    }
  }

  const std::vector<std::uint8_t> h = predict_vec(edges, Mode::kH);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 16; ++j) {
      CHECK_EQ(h[static_cast<std::size_t>(i) * 16 +
                 static_cast<std::size_t>(j)],
               edges.left[i]);
    }
  }

  // Edge filtering does not apply at exactly 90/180 degrees.
  const std::vector<std::uint8_t> v_filtered =
      predict_vec(edges, Mode::kV, 0, true, true, true);
  const std::vector<std::uint8_t> v_unfiltered =
      predict_vec(edges, Mode::kV, 0, false, true, true);
  CHECK(v_filtered == v_unfiltered);
}

void test_directional_golden_45_135() {
  Plane plane(32, 32);
  fill_gradient(plane, 1, 1);
  // Distinct above row at y = 7, x = 7..15 (corner + 8 edge samples).
  const int above_values[9] = {5, 10, 20, 30, 40, 50, 60, 70, 80};
  for (int i = 0; i < 9; ++i) {
    plane.at(7 + i, 7) = static_cast<std::uint8_t>(above_values[i]);
  }
  // Distinct left column at x = 7, y = 8..11.
  for (int i = 0; i < 4; ++i) {
    plane.at(7, 8 + i) = static_cast<std::uint8_t>(200 + i);
  }

  const IntraEdges edges =
      prepare_ok(make_request(plane, 8, 8, 4, 4, true, true, 1, 0));
  for (int i = 0; i < 8; ++i) {
    CHECK_EQ(edges.above[i], above_values[i + 1]);
  }

  // D45 / pAngle 45: Dr_Intra_Derivative[45] = 64, so base = i + 1 + j and
  // shift = 0 (the IDIF tap row is {0,128,0,0}). Prediction is exactly the
  // above sample at the projected position.
  const std::vector<std::uint8_t> d45 =
      predict_vec(edges, Mode::kD45, 0, false);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      CHECK_EQ(d45[static_cast<std::size_t>(i) * 4 +
                   static_cast<std::size_t>(j)],
               edges.above[i + 1 + j]);
    }
  }

  // D135 / pAngle 135: dx = dy = Dr_Intra_Derivative[45] = 64, base is
  // j - i - 1 for the above branch, i - j - 1 for the left branch.
  const std::vector<std::uint8_t> d135 =
      predict_vec(edges, Mode::kD135, 0, false);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int base = j - i - 1;
      const int expected = base >= -1 ? edges.above[base] : edges.left[i - j - 1];
      CHECK_EQ(d135[static_cast<std::size_t>(i) * 4 +
                    static_cast<std::size_t>(j)],
               expected);
    }
  }
}

void test_wide_angle_mapping_prediction() {
  // Tall 4x16 block: D45 (45 deg) is mapped to D203 at 225 deg. At 225 deg
  // dy = Dr_Intra_Derivative[270-225] = Dr_Intra_Derivative[45] = 64, so
  // base = i + j + 1 and shift = 0; with IDIF tap row 0 the prediction is
  // exactly LeftCol[i + j + 1]. This exercises the mapping and the p>180
  // branch, including the left-edge padding to w+h+1.
  Plane plane(32, 32);
  fill_gradient(plane, 4, 7);
  const IntraEdges edges =
      prepare_ok(make_request(plane, 8, 8, 4, 16, true, true, 0, 1));
  const MappedAngle mapped = av2enc::intra::map_angle(Mode::kD45, 0, 4, 16);
  CHECK_EQ(static_cast<int>(mapped.mode), static_cast<int>(Mode::kD203));
  CHECK_EQ(mapped.p_angle, 225);

  const std::vector<std::uint8_t> out =
      predict_vec(edges, Mode::kD45, 0, false);
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 4; ++j) {
      CHECK_EQ(out[static_cast<std::size_t>(i) * 4 +
                   static_cast<std::size_t>(j)],
               edges.left[i + j + 1]);
    }
  }
}

void test_idif_luma_vs_chroma_linear() {
  Plane plane(32, 32);
  fill_gradient(plane, 1, 1);
  // Non-linear step in the above row so the 4-tap IDIF differs from linear.
  const int above_values[8] = {10, 20, 200, 30, 40, 50, 60, 70};
  for (int i = 0; i < 8; ++i) {
    plane.at(8 + i, 7) = static_cast<std::uint8_t>(above_values[i]);
  }

  const IntraEdges luma =
      prepare_ok(make_request(plane, 8, 8, 4, 4, true, true, 0, 0, 0));
  const IntraEdges chroma =
      prepare_ok(make_request(plane, 8, 8, 4, 4, true, true, 0, 0, 1));

  // Chroma uses the linear 2-tap form (§7.13.2.8 enableIdif == 0).
  const std::vector<std::uint8_t> chroma_out =
      predict_vec(chroma, Mode::kD67, 0, false);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int idx = (i + 1) * 24;  // Dr_Intra_Derivative[67] = 24
      const int base = (idx >> 6) + j;
      const int shift = (idx >> 1) & 0x1F;
      const int expected =
          (chroma.above[base] * (32 - shift) + chroma.above[base + 1] * shift +
           16) >>
          5;
      CHECK_EQ(chroma_out[static_cast<std::size_t>(i) * 4 +
                          static_cast<std::size_t>(j)],
               expected);
    }
  }

  // Luma IDIF at (i=3, j=0): shift 16, base 1, tap row {-12,76,76,-12}.
  const std::vector<std::uint8_t> luma_out =
      predict_vec(luma, Mode::kD67, 0, false);
  const int sum = -12 * luma.above[0] + 76 * luma.above[1] +
                  76 * luma.above[2] - 12 * luma.above[3];
  const int expected_idif = std::max(0, std::min(255, (sum + 64) >> 7));
  CHECK_EQ(luma_out[3 * 4 + 0], expected_idif);

  // And the two planes must not produce identical output on this content.
  bool differs = false;
  for (std::size_t i = 0; i < chroma_out.size(); ++i) {
    if (chroma_out[i] != luma_out[i]) {
      differs = true;
    }
  }
  CHECK(differs);
}

void test_smooth_prediction() {
  // Hand-computed 4x4 SMOOTH/SMOOTH_V/SMOOTH_H at (0,0).
  IntraEdges edges = manual_edges(4, 4);
  const int above4[5] = {100, 104, 108, 112, 116};
  const int left4[5] = {120, 124, 128, 132, 136};
  for (int i = 0; i < 5; ++i) {
    edges.above[i] = static_cast<std::int16_t>(above4[i]);
    edges.left[i] = static_cast<std::int16_t>(left4[i]);
  }
  edges.above[-1] = 50;
  edges.left[-1] = 50;

  CHECK_EQ(predict_vec(edges, Mode::kSmooth)[0], 113);
  CHECK_EQ(predict_vec(edges, Mode::kSmoothV)[0], 105);
  CHECK_EQ(predict_vec(edges, Mode::kSmoothH)[0], 120);

  // Full-block comparison against a direct transcription of §7.13.2.13.
  IntraEdges square = manual_edges(8, 8);
  square.above[8] = 170;  // tr
  square.left[8] = 180;   // bl
  const int log2_w = 3;
  const int log2_h = 3;
  const int scale = (log2_w + log2_h - 4 + 2) >> 2;
  auto expected_smooth = [&](int i, int j, Mode mode) {
    const int s_top = 32 >> std::min(6, (i << 1) >> scale);
    const int s_left = 32 >> std::min(6, (j << 1) >> scale);
    const int top = square.above[j];
    const int left = square.left[i];
    const int tr = square.above[8];
    const int bl = square.left[8];
    const int pred_h =
        tr + (((left - tr) * (8 - 1 - j) + (1 << (log2_w - 1))) >> log2_w);
    const int pred_v =
        bl + (((top - bl) * (8 - 1 - i) + (1 << (log2_h - 1))) >> log2_h);
    const int pred_h2 = pred_h + (((left - pred_h) * s_left + 32) >> 6);
    const int pred_v2 = pred_v + (((top - pred_v) * s_top + 32) >> 6);
    if (mode == Mode::kSmoothH) {
      return pred_h2;
    }
    if (mode == Mode::kSmoothV) {
      return pred_v2;
    }
    return (pred_v2 + pred_h2 + 1) >> 1;
  };

  for (const Mode mode : {Mode::kSmooth, Mode::kSmoothV, Mode::kSmoothH}) {
    const std::vector<std::uint8_t> out = predict_vec(square, mode);
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        CHECK_EQ(out[static_cast<std::size_t>(i) * 8 +
                     static_cast<std::size_t>(j)],
                 expected_smooth(i, j, mode));
      }
    }
  }
}

void test_paeth_prediction() {
  IntraEdges edges = manual_edges(4, 4);
  for (int i = 0; i < 4; ++i) {
    edges.above[i] = 100;
    edges.left[i] = 50;
  }
  edges.above[-1] = 200;
  // base = 100 + 50 - 200 = -50; pLeft=100, pTop=150, pTopLeft=250 -> left.
  CHECK(all_equal(predict_vec(edges, Mode::kPaeth), 50));

  for (int i = 0; i < 4; ++i) {
    edges.above[i] = 200;
    edges.left[i] = 10;
  }
  edges.above[-1] = 10;
  // base=200; pTop=0 wins.
  CHECK(all_equal(predict_vec(edges, Mode::kPaeth), 200));

  for (int i = 0; i < 4; ++i) {
    edges.above[i] = 10;
    edges.left[i] = 10;
  }
  edges.above[-1] = 10;
  // All distances zero: pLeft wins (tie-break order in §7.13.2.2).
  CHECK(all_equal(predict_vec(edges, Mode::kPaeth), 10));
}

void test_boundary_blocks() {
  Plane plane(64, 64);
  fill_gradient(plane, 2, 3);

  // Top-left block: no neighbours. DC = 128, V = 127, H = 129, Paeth = 128.
  {
    const IntraEdges edges =
        prepare_ok(make_request(plane, 0, 0, 8, 8, false, false));
    CHECK(all_equal(predict_vec(edges, Mode::kDc), 128));
    CHECK(all_equal(predict_vec(edges, Mode::kV), 127));
    CHECK(all_equal(predict_vec(edges, Mode::kH), 129));
    CHECK(all_equal(predict_vec(edges, Mode::kPaeth), 128));
  }

  // Top-edge block (left only): V repeats the sample above the top-left
  // neighbour, H copies the left column, DC averages the left column.
  {
    const IntraEdges edges =
        prepare_ok(make_request(plane, 8, 0, 8, 8, true, false));
    const std::vector<std::uint8_t> v = predict_vec(edges, Mode::kV);
    CHECK(all_equal(v, plane.at(7, 0)));

    int sum = 0;
    const std::vector<std::uint8_t> h = predict_vec(edges, Mode::kH);
    for (int i = 0; i < 8; ++i) {
      sum += plane.at(7, std::min(7, i));
      for (int j = 0; j < 8; ++j) {
        CHECK_EQ(h[static_cast<std::size_t>(i) * 8 +
                   static_cast<std::size_t>(j)],
                 plane.at(7, std::min(7, i)));
      }
    }
    CHECK(all_equal(predict_vec(edges, Mode::kDc), (sum + 4) >> 3));
  }

  // Left-edge block (above only).
  {
    const IntraEdges edges =
        prepare_ok(make_request(plane, 0, 8, 8, 8, false, true));
    const std::vector<std::uint8_t> h = predict_vec(edges, Mode::kH);
    CHECK(all_equal(h, plane.at(0, 7)));

    const std::vector<std::uint8_t> v = predict_vec(edges, Mode::kV);
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        CHECK_EQ(v[static_cast<std::size_t>(i) * 8 +
                   static_cast<std::size_t>(j)],
                 plane.at(std::min(7, j), 7));
      }
    }
  }

  // Block crossing the right and bottom frame edges: V/H clamp at maxX/maxY.
  {
    const IntraEdges edges =
        prepare_ok(make_request(plane, 56, 56, 16, 16, true, true));
    CHECK_EQ(edges.valid_above_px, 8);
    CHECK_EQ(edges.valid_left_px, 8);
    const std::vector<std::uint8_t> v = predict_vec(edges, Mode::kV);
    const std::vector<std::uint8_t> h = predict_vec(edges, Mode::kH);
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        CHECK_EQ(v[static_cast<std::size_t>(i) * 16 +
                   static_cast<std::size_t>(j)],
                 plane.at(std::min(63, 56 + j), 55));
        CHECK_EQ(h[static_cast<std::size_t>(i) * 16 +
                   static_cast<std::size_t>(j)],
                 plane.at(55, std::min(63, 56 + i)));
      }
    }
  }

  // Block crossing only the right edge on a large block uses 64 samples; the
  // 4x4 below-left extension must not read outside the plane.
  {
    Plane wide(96, 64);
    fill_gradient(wide, 5, 1);
    const IntraEdges edges = prepare_ok(
        make_request(wide, 90, 20, 8, 8, true, true, 2, 2));
    CHECK_EQ(edges.valid_above_px, 6);
    CHECK_EQ(edges.valid_left_px, 8);
    const std::vector<std::uint8_t> v = predict_vec(edges, Mode::kV);
    for (int j = 0; j < 8; ++j) {
      CHECK_EQ(v[static_cast<std::size_t>(j)], wide.at(std::min(95, 90 + j), 19));
    }
  }
}

void test_edge_filter_changes_prediction() {
  // A sharp step plus a 32x32 block: 135-degree prediction with edge
  // filtering enabled must differ from the unfiltered prediction.
  Plane plane(64, 64);
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 64; ++x) {
      plane.at(x, y) = static_cast<std::uint8_t>(x < 24 ? 0 : 255);
    }
  }
  const IntraEdges edges =
      prepare_ok(make_request(plane, 16, 16, 32, 32, true, true));
  const std::vector<std::uint8_t> filtered =
      predict_vec(edges, Mode::kD135, 0, true);
  const std::vector<std::uint8_t> plain =
      predict_vec(edges, Mode::kD135, 0, false);
  CHECK(filtered != plain);

  // On a 8x8 block (w+h = 16) the smooth-neighbour filter type selects a
  // different strength for the 141-degree angle: strength 1 vs 2 above and
  // strength 0 vs 1 on the left.
  const IntraEdges small =
      prepare_ok(make_request(plane, 20, 16, 8, 8, true, true));
  const std::vector<std::uint8_t> small_plain =
      predict_vec(small, Mode::kD135, 2, true, false, false);
  const std::vector<std::uint8_t> small_smooth =
      predict_vec(small, Mode::kD135, 2, true, true, true);
  CHECK(small_plain != small_smooth);
}

// ---------------------------------------------------------------------------
// RDO candidates and cost hooks
// ---------------------------------------------------------------------------

void test_mode_candidates() {
  std::array<ModeCandidate, av2enc::intra::kMaxModeCandidates> candidates{};

  const int base = av2enc::intra::build_mode_candidates(candidates, false);
  CHECK_EQ(base, 13);
  for (int m = 0; m < 13; ++m) {
    CHECK_EQ(static_cast<int>(candidates[static_cast<std::size_t>(m)].mode), m);
    CHECK_EQ(candidates[static_cast<std::size_t>(m)].angle_delta, 0);
  }

  const int full = av2enc::intra::build_mode_candidates(candidates, true);
  CHECK_EQ(full, av2enc::intra::kMaxModeCandidates);

  std::array<std::array<bool, 7>, 13> seen{};
  for (int c = 0; c < full; ++c) {
    const ModeCandidate& candidate = candidates[static_cast<std::size_t>(c)];
    const int mode = static_cast<int>(candidate.mode);
    const int delta = candidate.angle_delta;
    CHECK(mode >= 0 && mode < 13);
    CHECK(delta >= -3 && delta <= 3);
    CHECK(!seen[static_cast<std::size_t>(mode)]
             [static_cast<std::size_t>(delta + 3)]);
    seen[static_cast<std::size_t>(mode)][static_cast<std::size_t>(delta + 3)] =
        true;
    if (!av2enc::intra::is_directional(candidate.mode)) {
      CHECK_EQ(delta, 0);
    }
  }
  for (int m = 0; m < 13; ++m) {
    for (int d = -3; d <= 3; ++d) {
      const bool directional = m >= 1 && m <= 8;
      const bool expected = (d == 0) || directional;
      CHECK_EQ(seen[static_cast<std::size_t>(m)]
                   [static_cast<std::size_t>(d + 3)],
               expected);
    }
  }

  // Too-small outputs are rejected rather than truncated.
  std::array<ModeCandidate, av2enc::intra::kMaxModeCandidates - 1> small{};
  CHECK_EQ(av2enc::intra::build_mode_candidates(small, true), -1);
  CHECK_EQ(av2enc::intra::build_mode_candidates(
               std::span<ModeCandidate>(small.data(), 13), false),
           13);
}

void test_cost_hooks() {
  Plane src(64, 64);
  Plane recon(64, 64);
  fill_gradient(src, 2, 5);
  fill_gradient(recon, 3, 1);

  const IntraEdges edges =
      prepare_ok(make_request(recon, 16, 16, 8, 8, true, true));
  const std::vector<std::uint8_t> pred = predict_vec(edges, Mode::kD45);

  PredictionCost manual;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      const int diff = static_cast<int>(src.at(16 + j, 16 + i)) -
                       static_cast<int>(pred[static_cast<std::size_t>(i) * 8 +
                                             static_cast<std::size_t>(j)]);
      const int magnitude = std::abs(diff);
      manual.sad += static_cast<std::uint64_t>(magnitude);
      manual.sse += static_cast<std::uint64_t>(magnitude * magnitude);
    }
  }

  const PredictionCost direct = av2enc::intra::prediction_cost(
      src.data.data() + 16 * static_cast<std::size_t>(src.stride) + 16,
      static_cast<std::size_t>(src.stride), pred.data(), 8, 8, 8);
  CHECK_EQ(direct.sad, manual.sad);
  CHECK_EQ(direct.sse, manual.sse);

  // predict_and_cost writes the same prediction into the scratch buffer.
  std::vector<std::uint8_t> scratch(64, 0);
  PredictionCost cost;
  PredictParams params;
  params.mode = Mode::kD45;
  const IntraStatus status = av2enc::intra::predict_and_cost(
      edges, params, scratch,
      src.data.data() + 16 * static_cast<std::size_t>(src.stride) + 16,
      static_cast<std::size_t>(src.stride), cost);
  CHECK_EQ(status_code(status), status_code(IntraStatus::kOk));
  PredictionCost expected_cost;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      const int diff = static_cast<int>(
                           src.at(16 + j, 16 + i)) -
                       static_cast<int>(
                           scratch[static_cast<std::size_t>(i) * 8 +
                                   static_cast<std::size_t>(j)]);
      const int magnitude = std::abs(diff);
      expected_cost.sad += static_cast<std::uint64_t>(magnitude);
      expected_cost.sse +=
          static_cast<std::uint64_t>(magnitude * magnitude);
    }
  }
  CHECK_EQ(cost.sad, expected_cost.sad);
  CHECK_EQ(cost.sse, expected_cost.sse);
  CHECK(pred == scratch);

  // Scratch buffer too small is rejected.
  std::vector<std::uint8_t> tiny(63, 0);
  CHECK_EQ(status_code(av2enc::intra::predict_and_cost(
               edges, params, tiny, src.data.data(),
               static_cast<std::size_t>(src.stride), cost)),
           status_code(IntraStatus::kInvalidArgument));

  // predict_intra must not mutate the prepared edges (RDO reuse).
  IntraEdges before = edges;
  for (const Mode mode :
       {Mode::kDc, Mode::kV, Mode::kH, Mode::kD45, Mode::kD135, Mode::kSmooth,
        Mode::kSmoothV, Mode::kSmoothH, Mode::kPaeth}) {
    (void)predict_vec(edges, mode, 0, true, true, true);
    CHECK(same_edges(edges, before));
  }
}

// ---------------------------------------------------------------------------
// Invalid arguments
// ---------------------------------------------------------------------------

void test_invalid_arguments() {
  Plane plane(32, 32);
  fill_gradient(plane, 1, 2);
  IntraEdges edges;
  NeighborRequest request = make_request(plane, 8, 8, 8, 8, true, true);

  CHECK_EQ(status_code(av2enc::intra::prepare_edges(request, 8, edges)),
           status_code(IntraStatus::kOk));

  request.recon.data = nullptr;
  CHECK_EQ(status_code(av2enc::intra::prepare_edges(request, 8, edges)),
           status_code(IntraStatus::kInvalidArgument));
  request = make_request(plane, 8, 8, 8, 8, true, true);

  request.recon.stride = 0;
  CHECK_EQ(status_code(av2enc::intra::prepare_edges(request, 8, edges)),
           status_code(IntraStatus::kInvalidArgument));
  request = make_request(plane, 8, 8, 8, 8, true, true);

  request.w = 0;
  CHECK_EQ(status_code(av2enc::intra::prepare_edges(request, 8, edges)),
           status_code(IntraStatus::kInvalidArgument));
  request = make_request(plane, 8, 8, 128, 8, true, true);
  CHECK_EQ(status_code(av2enc::intra::prepare_edges(request, 8, edges)),
           status_code(IntraStatus::kInvalidArgument));
  request = make_request(plane, 8, 8, 6, 8, true, true);
  CHECK_EQ(status_code(av2enc::intra::prepare_edges(request, 8, edges)),
           status_code(IntraStatus::kInvalidArgument));
  request = make_request(plane, 8, 8, 8, 8, true, true);
  CHECK_EQ(status_code(av2enc::intra::prepare_edges(request, 16, edges)),
           status_code(IntraStatus::kInvalidArgument));

  request = make_request(plane, 8, 8, 8, 8, true, true);
  request.x = 40;
  CHECK_EQ(status_code(av2enc::intra::prepare_edges(request, 8, edges)),
           status_code(IntraStatus::kInvalidArgument));
  request = make_request(plane, 0, 8, 8, 8, true, true);
  CHECK_EQ(status_code(av2enc::intra::prepare_edges(request, 8, edges)),
           status_code(IntraStatus::kInvalidArgument));
  request = make_request(plane, 8, 0, 8, 8, true, true);
  CHECK_EQ(status_code(av2enc::intra::prepare_edges(request, 8, edges)),
           status_code(IntraStatus::kInvalidArgument));

  const IntraEdges good = prepare_ok(make_request(plane, 8, 8, 8, 8, true, true));
  std::vector<std::uint8_t> out(64, 0);
  PredictParams params;

  CHECK_EQ(status_code(av2enc::intra::predict_intra(good, params, nullptr, 8)),
           status_code(IntraStatus::kInvalidArgument));
  CHECK_EQ(status_code(av2enc::intra::predict_intra(good, params, out.data(), 7)),
           status_code(IntraStatus::kInvalidArgument));

  params.mode = Mode::kDc;
  params.angle_delta = 1;
  CHECK_EQ(status_code(av2enc::intra::predict_intra(good, params, out.data(), 8)),
           status_code(IntraStatus::kInvalidArgument));

  params.angle_delta = 4;
  params.mode = Mode::kV;
  CHECK_EQ(status_code(av2enc::intra::predict_intra(good, params, out.data(), 8)),
           status_code(IntraStatus::kInvalidArgument));

  params.angle_delta = 0;
  params.mode = static_cast<Mode>(13);
  CHECK_EQ(status_code(av2enc::intra::predict_intra(good, params, out.data(), 8)),
           status_code(IntraStatus::kInvalidArgument));

  IntraEdges broken = good;
  broken.bit_depth = 7;
  params.mode = Mode::kDc;
  CHECK_EQ(status_code(av2enc::intra::predict_intra(broken, params, out.data(), 8)),
           status_code(IntraStatus::kInvalidArgument));
}

// ---------------------------------------------------------------------------
// Randomized sweep (deterministic LCG), including blocks past frame edges
// ---------------------------------------------------------------------------

void test_random_sweep() {
  constexpr std::uint32_t kSeed = 0x12345678U;
  std::uint32_t rng = kSeed;
  auto next = [&rng]() {
    rng = rng * 1664525U + 1013904223U;
    return rng;
  };

  Plane plane(80, 80);
  for (int y = 0; y < plane.height; ++y) {
    for (int x = 0; x < plane.width; ++x) {
      plane.at(x, y) = static_cast<std::uint8_t>(next() >> 24);
    }
  }

  const int sizes[5] = {4, 8, 16, 32, 64};
  std::array<ModeCandidate, av2enc::intra::kMaxModeCandidates> candidates{};
  const int count = av2enc::intra::build_mode_candidates(candidates, true);

  for (int trial = 0; trial < 120; ++trial) {
    const int w = sizes[next() % 5U];
    const int h = sizes[next() % 5U];
    const int x = static_cast<int>(next() % 80U);
    const int y = static_cast<int>(next() % 80U);
    const bool have_left = x > 0;
    const bool have_above = y > 0;
    const int n4ar = static_cast<int>(next() % 3U);
    const int n4bl = static_cast<int>(next() % 3U);
    const int color_plane = static_cast<int>(next() % 2U);

    const NeighborRequest request = make_request(
        plane, x, y, w, h, have_left, have_above, n4ar, n4bl, color_plane);
    IntraEdges edges;
    const IntraStatus status =
        av2enc::intra::prepare_edges(request, 8, edges);
    CHECK_EQ(status_code(status), status_code(IntraStatus::kOk));
    if (!ok(status)) {
      continue;
    }
    const IntraEdges snapshot = edges;

    for (int c = 0; c < count; ++c) {
      const ModeCandidate candidate = candidates[static_cast<std::size_t>(c)];
      PredictParams params;
      params.mode = candidate.mode;
      params.angle_delta = candidate.angle_delta;
      params.enable_edge_filter = (next() & 1U) != 0;
      params.above_neighbor_is_smooth = (next() & 1U) != 0;
      params.left_neighbor_is_smooth = (next() & 1U) != 0;

      std::vector<std::uint8_t> a;
      predict_ok(edges, params, a);
      std::vector<std::uint8_t> b;
      predict_ok(edges, params, b);
      CHECK(a == b);

      // predict_block() must agree with prepare+predict.
      std::vector<std::uint8_t> direct(
          static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
      const IntraStatus direct_status = av2enc::intra::predict_block(
          request, 8, params, direct.data(), static_cast<std::size_t>(w));
      CHECK_EQ(status_code(direct_status), status_code(IntraStatus::kOk));
      CHECK(direct == a);

      // Edges stay untouched across candidates.
      CHECK(same_edges(edges, snapshot));
    }
  }
}

}  // namespace

int main() {
  test_mode_tables();
  test_edge_filter_strength();
  test_edge_filter_golden();
  test_prepare_edges_fillers();
  test_prepare_edges_clamping();
  test_extension_and_edge_filter_golden();
  test_below_left_extension_golden();
  test_superblock_two_sb_border();
  test_sb_32x8_above_right_golden();
  test_sb_border_below_left_clamp();
  test_dc_prediction();
  test_flat_all_modes_and_sizes();
  test_v_and_h_exact();
  test_directional_golden_45_135();
  test_wide_angle_mapping_prediction();
  test_idif_luma_vs_chroma_linear();
  test_smooth_prediction();
  test_paeth_prediction();
  test_boundary_blocks();
  test_edge_filter_changes_prediction();
  test_mode_candidates();
  test_cost_hooks();
  test_invalid_arguments();
  test_random_sweep();

  if (g_failures != 0) {
    std::cerr << "intra_modes_test: " << g_failures << " failure(s) in "
              << g_checks << " checks\n";
    return 1;
  }
  std::cout << "intra_modes_test: PASS (" << g_checks << " checks)\n";
  return 0;
}
