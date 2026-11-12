// CPU reference tests for AV2 v1.0.0 DIP (§5.20.7.9, §6.19.7.8,
// §7.13.2.3-.6).
//
// The expected values are produced by an independent, literal transcription of
// the specification in this file.  The production implementation is never used
// to generate an expectation.  Coverage:
//   * the normative Dip_Weights table checksum / spot values;
//   * every DIP-capable and non-capable block size;
//   * edge preparation for all have_left/have_above combinations and partial
//     blocks at the right/bottom frame edges;
//   * all 6 dip_modes x 2 dip_transpose values;
//   * 8-, 10-, and 12-bit samples;
//   * real luma blocks from a Y4M fixture;
//   * the decision API and syntax predicates/CDFs.

#include "codec/intra/dip/dip.h"
#include "codec/intra/dip/dip_syntax.h"

#include "core/io/y4m.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using av2enc::dip::Block;
using av2enc::dip::Decision;
using av2enc::dip::DecisionParams;
using av2enc::dip::EdgeArray;
using av2enc::dip::Edges;
using av2enc::dip::Params;
using av2enc::dip::PlaneView;
using av2enc::dip::Status;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL: %s\n", what);
  }
}

std::uint32_t g_rng = 0x1234567u;
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

constexpr int kDims[] = {4, 8, 16, 32, 64};
constexpr int kValidDims[][2] = {
    {8, 16},  {8, 32},  {8, 64},  {16, 8},  {16, 16},
    {16, 32}, {16, 64}, {32, 8},  {32, 16}, {32, 32},
    {32, 64}, {64, 8},  {64, 16}, {64, 32}, {64, 64},
};

int read_ref(const PlaneView& plane, int x, int y) {
  if (plane.bytes_per_sample == 1) {
    const auto* data = static_cast<const std::uint8_t*>(plane.data);
    return data[static_cast<std::size_t>(y) *
                static_cast<std::size_t>(plane.stride) +
            static_cast<std::size_t>(x)];
  }
  const auto* data = static_cast<const std::uint16_t*>(plane.data);
  return data[static_cast<std::size_t>(y) *
                static_cast<std::size_t>(plane.stride) +
            static_cast<std::size_t>(x)];
}

struct RefEdges {
  int above[80]{};
  int left[80]{};
  int corner = 0;
};

void ref_prepare_edges(const PlaneView& recon, int bit_depth, int x, int y,
                       int w, int h, int max_x, int max_y, bool have_left,
                       bool have_above, int num4_ar, int num4_bl,
                       RefEdges& e) {
  const int num_above = w + (w >> 2);
  const int num_left = h + (h >> 2);
  const int mid = 1 << (bit_depth - 1);
  for (int i = 0; i < num_left; ++i) {
    if (!have_left && have_above) {
      e.left[i] = read_ref(recon, x, y - 1);
    } else if (!have_left) {
      e.left[i] = mid + 1;
    } else {
      const int limit = std::min(max_y, y + h + 4 * num4_bl - 1);
      e.left[i] = read_ref(recon, x - 1, std::min(limit, y + i));
    }
  }
  for (int i = 0; i < num_above; ++i) {
    if (!have_above && have_left) {
      e.above[i] = read_ref(recon, x - 1, y);
    } else if (!have_above) {
      e.above[i] = mid - 1;
    } else {
      const int limit = std::min(max_x, x + w + 4 * num4_ar - 1);
      e.above[i] = read_ref(recon, std::min(limit, x + i), y - 1);
    }
  }
  if (have_above && have_left) {
    e.corner = read_ref(recon, x - 1, y - 1);
  } else if (have_above) {
    e.corner = read_ref(recon, x, y - 1);
  } else if (have_left) {
    e.corner = read_ref(recon, x - 1, y);
  } else {
    e.corner = mid;
  }
}

void ref_avg(const int* row, int n, int* out) {
  const int down = n >> 2;
  for (int i = 0; i < 5; ++i) {
    int t = 0;
    for (int j = 0; j < down; ++j) t += row[i * down + j];
    out[i] = (t + (down >> 1)) / down;
  }
}

void ref_features(const RefEdges& e, int w, int h, bool transpose, int* f) {
  int above_avg[5];
  int left_avg[5];
  ref_avg(e.above, w, above_avg);
  ref_avg(e.left, h, left_avg);
  f[0] = e.corner;
  for (int i = 0; i < 4; ++i) {
    f[i + 1] = transpose ? left_avg[i] : above_avg[i];
    f[i + 5] = transpose ? above_avg[i] : left_avg[i];
  }
  f[9] = transpose ? left_avg[4] : above_avg[4];
  f[10] = transpose ? above_avg[4] : left_avg[4];
}

void ref_predict(const PlaneView& recon, int bit_depth, int x, int y, int w,
                 int h, int max_x, int max_y, bool have_left, bool have_above,
                 int num4_ar, int num4_bl, int mode, bool transpose,
                 std::uint16_t* out) {
  RefEdges e;
  ref_prepare_edges(recon, bit_depth, x, y, w, h, max_x, max_y, have_left,
                    have_above, num4_ar, num4_bl, e);
  int f[11];
  ref_features(e, w, h, transpose, f);

  std::uint16_t dip_pred[64]{};
  const int max_value = (1 << bit_depth) - 1;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      int c = 0;
      for (int k = 0; k < 11; ++k) {
        c += av2enc::dip::dip_weight(mode, i * 8 + j, k) * f[k];
      }
      int v = (c + 512) >> 10;
      v = std::max(0, std::min(max_value, v));
      const int index = transpose ? j * 8 + i : i * 8 + j;
      dip_pred[index] = static_cast<std::uint16_t>(v);
    }
  }

  const int upx = std::max(1, w / 8);
  const int upy = std::max(1, h / 8);
  const int downx = std::max(1, 8 / w);
  const int downy = std::max(1, 8 / h);
  int horz[8][64]{};
  const int rows = std::min(h, 8);
  const int cols = std::min(w, 8);
  for (int i = 0; i < rows; ++i) {
    const int yy = (i + 1) * upy - 1;
    for (int j = 0; j < cols; ++j) {
      const int p0 =
          j == 0 ? e.left[yy] : dip_pred[i * downy * 8 + (j - 1) * downx];
      const int p1 = dip_pred[i * downy * 8 + j * downx];
      for (int k = 0; k < upx; ++k) {
        const int xx = j * upx + k;
        const int w1 = k + 1;
        horz[i][xx] = ((upx - w1) * p0 + w1 * p1) / upx;
      }
    }
  }
  for (int xx = 0; xx < w; ++xx) {
    for (int i = 0; i < rows; ++i) {
      const int p0 = i == 0 ? e.above[xx] : horz[i - 1][xx];
      const int p1 = horz[i][xx];
      for (int k = 0; k < upy; ++k) {
        const int yy = i * upy + k;
        const int w1 = k + 1;
        out[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) + static_cast<std::size_t>(xx)] =
            static_cast<std::uint16_t>(((upy - w1) * p0 + w1 * p1) / upy);
      }
    }
  }
}

Block make_block(const PlaneView& recon, int x, int y, int w, int h, int max_x,
                 int max_y, bool have_left, bool have_above, int num4_ar,
                 int num4_bl) {
  Block block;
  block.recon = recon;
  block.x = x;
  block.y = y;
  block.w = w;
  block.h = h;
  block.max_x = max_x;
  block.max_y = max_y;
  block.have_left = have_left;
  block.have_above = have_above;
  block.num4_above_right = num4_ar;
  block.num4_below_left = num4_bl;
  return block;
}

void compare_edges(const Edges& got, const RefEdges& want, int w, int h) {
  check(got.above[-1] == want.corner, "edge corner");
  for (int i = 0; i < w + w / 4; ++i) {
    check(got.above[i] == want.above[i], "AboveRow sample");
  }
  for (int i = 0; i < h + h / 4; ++i) {
    check(got.left[i] == want.left[i], "LeftCol sample");
  }
}

void compare_prediction(const std::uint16_t* got, const std::uint16_t* want,
                        int w, int h, const char* what) {
  for (int i = 0; i < w * h; ++i) {
    if (got[i] != want[i]) {
      ++g_checks;
      ++g_failures;
      std::printf("FAIL: %s at %d: got %u want %u\n", what, i, got[i],
                  want[i]);
      return;
    }
  }
  ++g_checks;
}

void test_weight_table() {
  check(av2enc::dip::kNumModes == 6, "Dip_Weights mode count");
  check(av2enc::dip::kNumFeatures == 11, "Dip_Weights feature count");
  // Spot values read directly from the PDF.  These also pin the table's
  // mode-major ordering.
  const int expected0[11] = {-248, 690, 53, -26, 19, 413, 133, 3, 3, -13, -1};
  const int expected5[11] = {-199, 662, 105, 76, 15, 269, 68, 9, 31, -4, -4};
  for (int k = 0; k < 11; ++k) {
    check(av2enc::dip::dip_weight(0, 0, k) == expected0[k],
          "Dip_Weights[0][0]");
    check(av2enc::dip::dip_weight(5, 0, k) == expected5[k],
          "Dip_Weights[5][0]");
  }
  std::int64_t sum = 0;
  std::int64_t sum_sq = 0;
  for (int m = 0; m < 6; ++m) {
    for (int p = 0; p < 64; ++p) {
      int row_sum = 0;
      for (int k = 0; k < 11; ++k) {
        const int v = av2enc::dip::dip_weight(m, p, k);
        sum += v;
        sum_sq += static_cast<std::int64_t>(v) * v;
        row_sum += v;
      }
      // Each output position is a Q10 linear predictor; a shifted/misaligned
      // table row shows up immediately as a sum far from 1024.
      check(row_sum >= 1000 && row_sum <= 1048,
            "Dip_Weights row sum near Q10 unity");
    }
  }
  check(sum == 393140, "Dip_Weights checksum sum");
  check(sum_sq == 311734342, "Dip_Weights checksum sum of squares");
  check(av2enc::dip::dip_weight(6, 0, 0) == 0, "dip_weight range guard");
  check(av2enc::dip::dip_weight(0, 64, 0) == 0, "dip_weight position guard");
}

void test_availability() {
  for (int w : kDims) {
    for (int h : kDims) {
      const bool want = w > 4 && h > 4 && w * h >= 128;
      check(av2enc::dip::dip_block_available(w, h) == want,
            "dip_block_available");
      check(av2enc::dip::valid_block_dim(w), "valid_block_dim");
    }
  }
  check(!av2enc::dip::dip_block_available(3, 64), "non-power-of-two dim");
  check(!av2enc::dip::dip_block_available(0, 0), "zero dim");
  check(av2enc::dip::dip_block_available(8, 16), "minimum DIP block");
  check(!av2enc::dip::dip_block_available(8, 8), "8x8 is too small");
  check(!av2enc::dip::dip_block_available(4, 64), "4-wide is too narrow");
}

void test_edges_and_prediction_8bit() {
  constexpr int kStride = 160;
  constexpr int kRows = 160;
  std::vector<std::uint8_t> plane(static_cast<std::size_t>(kStride) * kRows);
  for (std::uint8_t& v : plane) {
    v = static_cast<std::uint8_t>(next_random() & 0xFFu);
  }
  const PlaneView recon = PlaneView::from_u8(plane.data(), kStride);
  const int positions[][2] = {{0, 0},   {8, 8},   {16, 24},
                              {32, 16}, {64, 48}, {80, 80}};

  for (const auto& dim : kValidDims) {
    const int w = dim[0];
    const int h = dim[1];
    for (const auto& pos : positions) {
      const int x = pos[0];
      const int y = pos[1];
      if (x + w > 120 || y + h > 120) continue;
      for (int avail = 0; avail < 4; ++avail) {
        const bool have_left = (avail & 1) != 0;
        const bool have_above = (avail & 2) != 0;
        if (have_left && x == 0) continue;
        if (have_above && y == 0) continue;
        // Right/bottom partial blocks: max_x/max_y may cut into the block.
        const int max_x = std::min(120, x + w - 1) - ((avail == 3) ? 2 : 0);
        const int max_y = std::min(120, y + h - 1) - ((avail == 3) ? 3 : 0);
        if (max_x < x || max_y < y) continue;
        const int num4_ar = random_range(0, w / 4 + 3);
        const int num4_bl = random_range(0, h / 4 + 3);
        const Block block = make_block(recon, x, y, w, h, max_x, max_y,
                                       have_left, have_above, num4_ar,
                                       num4_bl);
        Edges edges;
        const Status status = av2enc::dip::prepare_edges(block, 8, edges);
        check(status == Status::kOk, "prepare_edges 8-bit");
        if (status != Status::kOk) continue;
        RefEdges ref;
        ref_prepare_edges(recon, 8, x, y, w, h, max_x, max_y, have_left,
                          have_above, num4_ar, num4_bl, ref);
        compare_edges(edges, ref, w, h);

        for (int mode = 0; mode < 6; ++mode) {
          for (int transpose = 0; transpose < 2; ++transpose) {
            std::vector<std::uint16_t> got(
                static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
            std::vector<std::uint16_t> want(
                static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
            Params params;
            params.mode = mode;
            params.transpose = transpose != 0;
            check(av2enc::dip::predict(edges, params, got.data(),
                                     static_cast<std::size_t>(w)) ==
                      Status::kOk,
                  "predict 8-bit");
            ref_predict(recon, 8, x, y, w, h, max_x, max_y, have_left,
                        have_above, num4_ar, num4_bl, mode, transpose != 0,
                        want.data());
            compare_prediction(got.data(), want.data(), w, h,
                               "8-bit DIP prediction");
            for (int i = 0; i < w * h; ++i) {
              check(got[static_cast<std::size_t>(i)] <= 255,
                    "8-bit prediction range");
            }
          }
        }

        std::vector<std::uint8_t> got8(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
        Params params;
        params.mode = 3;
        params.transpose = true;
        check(av2enc::dip::predict_block8(block, 8, params, got8.data(),
                                       static_cast<std::size_t>(w)) ==
                  Status::kOk,
              "predict_block8");
      }
    }
  }
}

void test_high_bit_depth() {
  constexpr int kStride = 160;
  constexpr int kRows = 160;
  for (int bit_depth : {10, 12}) {
    std::vector<std::uint16_t> plane(
        static_cast<std::size_t>(kStride) * kRows);
    const int max_value = (1 << bit_depth) - 1;
    for (std::uint16_t& v : plane) {
      v = static_cast<std::uint16_t>(next_random() & static_cast<std::uint32_t>(max_value));
    }
    const PlaneView recon = PlaneView::from_u16(plane.data(), kStride);
    for (const auto& dim : kValidDims) {
      const int w = dim[0];
      const int h = dim[1];
      const int x = 16;
      const int y = 16;
      const int max_x = 150;
      const int max_y = 150;
      const Block block = make_block(recon, x, y, w, h, max_x, max_y, true,
                                     true, 4, 4);
      Edges edges;
      check(av2enc::dip::prepare_edges(block, bit_depth, edges) ==
                Status::kOk,
            "prepare_edges high bit depth");
      for (int mode = 0; mode < 6; ++mode) {
        for (int transpose = 0; transpose < 2; ++transpose) {
          std::vector<std::uint16_t> got(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
          std::vector<std::uint16_t> want(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
          Params params;
          params.mode = mode;
          params.transpose = transpose != 0;
          check(av2enc::dip::predict(edges, params, got.data(),
                                     static_cast<std::size_t>(w)) ==
                    Status::kOk,
                "predict high bit depth");
          ref_predict(recon, bit_depth, x, y, w, h, max_x, max_y, true, true,
                      4, 4, mode, transpose != 0, want.data());
          compare_prediction(got.data(), want.data(), w, h,
                             "high-bit-depth DIP prediction");
          for (int i = 0; i < w * h; ++i) {
            check(got[static_cast<std::size_t>(i)] <=
                      static_cast<std::uint16_t>(max_value),
                  "high-bit-depth range");
          }
        }
      }
    }
  }
}

void test_decision() {
  constexpr int kStride = 128;
  constexpr int kRows = 128;
  std::vector<std::uint8_t> plane(static_cast<std::size_t>(kStride) * kRows);
  for (std::uint8_t& v : plane) {
    v = static_cast<std::uint8_t>(next_random() & 0xFFu);
  }
  const PlaneView recon = PlaneView::from_u8(plane.data(), kStride);
  const int w = 16;
  const int h = 16;
  const Block block =
      make_block(recon, 32, 32, w, h, 127, 127, true, true, 4, 4);
  Edges edges;
  check(av2enc::dip::prepare_edges(block, 8, edges) == Status::kOk,
        "decision edges");

  DecisionParams params;
  params.enable_dip = true;
  params.y_mode = 0;
  params.palette_size_y = 0;
  params.metric = av2enc::dip::CostMetric::kSad;

  Decision decision;
  check(av2enc::dip::decide(edges, plane.data() + 32 * kStride + 32, kStride,
                            params, decision) == Status::kOk,
        "decision ok");
  check(decision.available, "decision available");
  check(decision.mode >= 0 && decision.mode < 6, "decision mode range");

  // The selected candidate must not be worse than every other candidate.
  std::uint64_t best = UINT64_MAX;
  for (int mode = 0; mode < 6; ++mode) {
    for (int transpose = 0; transpose < 2; ++transpose) {
      Params p;
      p.mode = mode;
      p.transpose = transpose != 0;
      std::vector<std::uint8_t> pred(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
      check(av2enc::dip::predict8(edges, p, pred.data(), w) == Status::kOk,
            "decision candidate");
      const auto cost = av2enc::dip::prediction_cost8(
          plane.data() + 32 * kStride + 32, kStride, pred.data(), w, w, h);
      best = std::min(best, cost.sad);
    }
  }
  check(decision.cost.sad == best, "decision picks minimum SAD");

  // An unavailable syntax context must not evaluate candidates.
  DecisionParams off;
  off.enable_dip = false;
  Decision unavailable;
  check(av2enc::dip::decide(edges, plane.data(), kStride, off, unavailable) ==
            Status::kUnavailable,
        "decision unavailable without enable_dip");
  check(!unavailable.available, "unavailable decision flag");
}

void test_syntax() {
  using av2enc::dip::DipModeInfo;
  using av2enc::dip::SyntaxContext;
  SyntaxContext c;
  c.enable_dip = true;
  c.y_mode = 0;
  c.palette_size_y = 0;
  c.w = 16;
  c.h = 16;
  check(av2enc::dip::dip_mode_info_present(c), "syntax present");
  c.y_mode = 1;
  check(!av2enc::dip::dip_mode_info_present(c), "YMode must be DC_PRED");
  c.y_mode = 0;
  c.palette_size_y = 1;
  check(!av2enc::dip::dip_mode_info_present(c), "palette must be empty");
  c.palette_size_y = 0;
  c.tree_is_chroma_part = true;
  check(!av2enc::dip::dip_mode_info_present(c), "chroma part has no DIP");
  c.tree_is_chroma_part = false;
  c.w = 8;
  c.h = 8;
  check(!av2enc::dip::dip_mode_info_present(c), "8x8 has no DIP");
  c.w = 16;
  c.enable_dip = false;
  check(!av2enc::dip::dip_mode_info_present(c), "enable_dip gate");

  check(av2enc::dip::use_dip_context(false, false) == 0, "use_dip ctx 0");
  check(av2enc::dip::use_dip_context(true, false) == 1, "use_dip ctx 1");
  check(av2enc::dip::use_dip_context(true, true) == 2, "use_dip ctx 2");

  DipModeInfo info;
  info.use_dip = true;
  info.dip_mode = 5;
  info.dip_transpose = true;
  c.enable_dip = true;
  check(av2enc::dip::validate_dip_mode_info(c, info), "validate use_dip");
  info.dip_mode = 6;
  check(!av2enc::dip::validate_dip_mode_info(c, info), "validate mode range");
  info.dip_mode = 0;
  info.use_dip = false;
  check(!av2enc::dip::validate_dip_mode_info(c, info),
        "transpose without use_dip");
  info.dip_transpose = false;
  check(av2enc::dip::validate_dip_mode_info(c, info),
        "use_dip false is valid");

  check(av2enc::dip::kDefaultUseDipCdf[0][0] == 11914, "use_dip CDF 0");
  check(av2enc::dip::kDefaultUseDipCdf[2][0] == 3555, "use_dip CDF 2");
  check(av2enc::dip::kDefaultDipModeCdf[4] == 27578, "dip_mode CDF");
}

void test_real_blocks(const char* path) {
  av2enc::Y4mReader reader{path};
  std::string error;
  if (!reader.open(error)) {
    std::printf("SKIP: real-block fixture %s: %s\n", path, error.c_str());
    return;
  }
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult result = reader.read_frame(frame, error);
  if (result != av2enc::Y4mReadResult::kFrame) {
    std::printf("SKIP: real-block fixture %s: %s\n", path, error.c_str());
    return;
  }
  const auto y = frame.y();
  const int width = static_cast<int>(frame.width());
  const int height = static_cast<int>(frame.height());
  const PlaneView recon = PlaneView::from_u8(y.data(), width);
  const int max_x = width - 1;
  const int max_y = height - 1;

  int cases = 0;
  for (const auto& dim : kValidDims) {
    const int w = dim[0];
    const int h = dim[1];
    if (w > width || h > height) continue;
    const int xs[] = {0, std::max(0, width - w), std::max(0, width - 8),
                      std::max(0, (width - w) / 2)};
    const int ys[] = {0, std::max(0, height - h), std::max(0, height - 8),
                      std::max(0, (height - h) / 2)};
    for (int x : xs) {
      for (int ypos : ys) {
        if (x + 1 >= width || ypos + 1 >= height) continue;
        const bool have_left = x > 0;
        const bool have_above = ypos > 0;
        for (int mode = 0; mode < 6; ++mode) {
          for (int transpose = 0; transpose < 2; ++transpose) {
            const Block block = make_block(recon, x, ypos, w, h, max_x, max_y,
                                           have_left, have_above, 4, 4);
            std::vector<std::uint16_t> got(
                static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
            std::vector<std::uint16_t> want(
                static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
            Params params;
            params.mode = mode;
            params.transpose = transpose != 0;
            check(av2enc::dip::predict_block(block, 8, params, got.data(),
                                     static_cast<std::size_t>(w)) ==
                      Status::kOk,
                  "real-block predict");
            ref_predict(recon, 8, x, ypos, w, h, max_x, max_y, have_left,
                        have_above, 4, 4, mode, transpose != 0, want.data());
            compare_prediction(got.data(), want.data(), w, h,
                               "real-block DIP prediction");
            ++cases;
          }
        }
      }
    }
  }
  check(cases > 0, "real blocks exercised");
  std::printf("real blocks: %d predictions from %s\n", cases, path);
}


void test_syntax_codec() {
  using av2enc::dip::DipCdfState;
  using av2enc::dip::DipModeInfo;
  using av2enc::dip::SyntaxContext;
  std::array<std::uint8_t, 512> buffer{};

  for (int ctx = 0; ctx < 3; ++ctx) {
    for (int use_dip = 0; use_dip < 2; ++use_dip) {
      for (int mode = 0; mode < 6; ++mode) {
        for (int transpose = 0; transpose < 2; ++transpose) {
          if (!use_dip && (mode != 0 || transpose != 0)) continue;
          SyntaxContext context;
          context.enable_dip = true;
          context.y_mode = 0;
          context.palette_size_y = 0;
          context.w = 16;
          context.h = 16;
          context.use_dip_ctx = ctx;
          DipModeInfo info;
          info.use_dip = use_dip != 0;
          info.dip_mode = mode;
          info.dip_transpose = transpose != 0;

          av2enc::av2::Encoder encoder;
          check(encoder.init_symbol(buffer.data(), buffer.size()) ==
                    av2enc::av2::Status::kOk,
                "syntax encoder init");
          DipCdfState write_cdfs;
          check(av2enc::dip::write_dip_mode_info(context, info, encoder,
                                                 write_cdfs),
                "write_dip_mode_info");
          std::size_t bytes = 0;
          check(encoder.exit_symbol(&bytes) == av2enc::av2::Status::kOk,
                "syntax encoder exit");

          av2enc::av2::Decoder decoder;
          check(decoder.init_symbol(buffer.data(), bytes, bytes) ==
                    av2enc::av2::Status::kOk,
                "syntax decoder init");
          DipCdfState read_cdfs;
          DipModeInfo got;
          check(av2enc::dip::read_dip_mode_info(context, decoder, read_cdfs,
                                                got),
                "read_dip_mode_info");
          check(got.use_dip == info.use_dip, "syntax use_dip roundtrip");
          check(got.dip_transpose == info.dip_transpose,
                "syntax transpose roundtrip");
          check(got.dip_mode == info.dip_mode, "syntax mode roundtrip");
          check(decoder.exit_symbol() == av2enc::av2::Status::kOk,
                "syntax decoder exit");
        }
      }
    }
  }

  // Several blocks share one per-tile CDF state; encoder and decoder must
  // adapt identically across the sequence.
  {
    SyntaxContext context;
    context.enable_dip = true;
    context.y_mode = 0;
    context.w = 16;
    context.h = 16;
    std::array<DipModeInfo, 32> written{};
    av2enc::av2::Encoder encoder;
    check(encoder.init_symbol(buffer.data(), buffer.size()) ==
              av2enc::av2::Status::kOk,
          "sequence encoder init");
    DipCdfState write_cdfs;
    for (int i = 0; i < 32; ++i) {
      context.use_dip_ctx = i % 3;
      const bool use_dip = (i % 2) != 0;
      written[static_cast<std::size_t>(i)].use_dip = use_dip;
      written[static_cast<std::size_t>(i)].dip_mode = use_dip ? i % 6 : 0;
      written[static_cast<std::size_t>(i)].dip_transpose =
          use_dip && (i % 3) == 0;
      check(av2enc::dip::write_dip_mode_info(
                context, written[static_cast<std::size_t>(i)], encoder,
                write_cdfs),
            "sequence write");
    }
    std::size_t bytes = 0;
    check(encoder.exit_symbol(&bytes) == av2enc::av2::Status::kOk,
          "sequence encoder exit");
    av2enc::av2::Decoder decoder;
    check(decoder.init_symbol(buffer.data(), bytes, bytes) ==
              av2enc::av2::Status::kOk,
          "sequence decoder init");
    DipCdfState read_cdfs;
    for (int i = 0; i < 32; ++i) {
      context.use_dip_ctx = i % 3;
      DipModeInfo got;
      check(av2enc::dip::read_dip_mode_info(context, decoder, read_cdfs, got),
            "sequence read");
      check(got.use_dip == written[static_cast<std::size_t>(i)].use_dip,
            "sequence use_dip");
      check(got.dip_mode == written[static_cast<std::size_t>(i)].dip_mode,
            "sequence mode");
      check(got.dip_transpose ==
                written[static_cast<std::size_t>(i)].dip_transpose,
            "sequence transpose");
    }
    check(decoder.exit_symbol() == av2enc::av2::Status::kOk,
          "sequence decoder exit");
  }

  // Absent syntax writes no bits and infers use_dip = 0.
  SyntaxContext absent;
  absent.enable_dip = false;
  absent.w = 16;
  absent.h = 16;
  av2enc::av2::Encoder encoder;
  check(encoder.init_symbol(buffer.data(), buffer.size()) ==
            av2enc::av2::Status::kOk,
        "absent encoder init");
  DipCdfState cdfs;
  DipModeInfo info;
  check(av2enc::dip::write_dip_mode_info(absent, info, encoder, cdfs),
        "absent write");
  std::size_t bytes = 0;
  check(encoder.exit_symbol(&bytes) == av2enc::av2::Status::kOk,
        "absent encoder exit");
  av2enc::av2::Decoder decoder;
  check(decoder.init_symbol(buffer.data(), bytes, bytes) ==
            av2enc::av2::Status::kOk,
        "absent decoder init");
  DipModeInfo got;
  check(av2enc::dip::read_dip_mode_info(absent, decoder, cdfs, got),
        "absent read");
  check(!got.use_dip && !got.dip_transpose && got.dip_mode == 0,
        "absent inferred defaults");
}

void test_rejections() {
  std::vector<std::uint8_t> plane(128 * 128, 77);
  const PlaneView recon = PlaneView::from_u8(plane.data(), 128);
  Edges edges;
  Block block = make_block(recon, 8, 8, 8, 8, 127, 127, true, true, 2, 2);
  check(av2enc::dip::prepare_edges(block, 8, edges) == Status::kUnavailable,
        "8x8 edge prep rejected");
  block.w = 16;
  block.h = 16;
  block.have_left = true;
  block.x = 0;
  check(av2enc::dip::prepare_edges(block, 8, edges) ==
            Status::kInvalidArgument,
        "have_left at x=0 rejected");
  block.x = 8;
  block.recon.stride = 4;
  check(av2enc::dip::prepare_edges(block, 8, edges) ==
            Status::kInvalidArgument,
        "short stride rejected");
  block.recon.stride = 128;
  block.recon.data = nullptr;
  check(av2enc::dip::prepare_edges(block, 8, edges) ==
            Status::kInvalidArgument,
        "null recon rejected");
  block.recon.data = plane.data();
  block.have_left = false;
  block.have_above = false;
  block.x = 0;
  block.y = 0;
  check(av2enc::dip::prepare_edges(block, 8, edges) == Status::kOk,
        "unavailable neighbours use fillers");
  check(edges.above[-1] == 128, "both-unavailable corner filler");
  check(edges.above[0] == 127, "above unavailable filler");
  check(edges.left[0] == 129, "left unavailable filler");
}

}  // namespace

int main(int argc, char** argv) {
  const char* fixture =
      argc > 1 ? argv[1] : "testdata/shine_256x256_1f.y4m";
  test_weight_table();
  test_availability();
  test_edges_and_prediction_8bit();
  test_high_bit_depth();
  test_decision();
  test_syntax();
  test_syntax_codec();
  test_rejections();
  test_real_blocks(fixture);
  std::printf("dip_test: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
