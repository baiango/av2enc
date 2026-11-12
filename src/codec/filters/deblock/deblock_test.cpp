// CPU reference tests for the AV2 §7.17 deblocking filter.
//
// Coverage:
//   * synthetic flat / ramp / step / texture frames
//   * filter-width classes (4/8/14/wide) for luma and chroma
//   * edge masks, skip / boundary strength, lossless sides
//   * partial blocks, MI padding, off-screen edges
//   * tile edges with and without disable_loopfilters_across_tiles
//   * randomized metadata fuzzing
//   * a real reconstructed frame (decoded.y4m / testdata/*.y4m)
//
// Usage: ./deblock_test [repo_root]
// The repo root is used to locate decoded.y4m and testdata/.  If omitted, it
// is derived from the source file path.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "codec/filters/deblock/deblock.h"
#include "codec/filters/deblock/deblock_test_common.h"

namespace {

using av2enc::deblock::DeblockEdgeInfo;
using av2enc::deblock::DeblockFrame;
using av2enc::deblock::DeblockOptions;
using av2enc::deblock::DeblockStatus;
using av2enc::deblock::TxSize;
using namespace av2enc::deblock::test;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                     \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_failures;                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg));   \
    }                                                                        \
  } while (0)

TestFrame copy_frame(const TestFrame& src) { return src; }

DeblockOptions base_options() {
  DeblockOptions o;
  o.apply_deblocking_filter[0] = true;
  o.apply_deblocking_filter[1] = true;
  o.apply_deblocking_filter[2] = true;
  o.apply_deblocking_filter[3] = true;
  o.bit_depth = 8;
  return o;
}

void run(TestFrame& f, const DeblockOptions& o) {
  DeblockFrame view = f.view();
  const DeblockStatus s = av2enc::deblock::deblock_frame(view, o);
  CHECK(s == DeblockStatus::kOk, "deblock_frame returned an error");
}

void test_flat_and_ramp() {
  const Pattern patterns[] = {Pattern::kFlat, Pattern::kRampX, Pattern::kRampY};
  for (Pattern p : patterns) {
    TestFrame f = make_frame(16, 12);
    set_uniform_blocks(f, 4, TxSize::k16x16, 120);
    fill_pattern(f, p);
    TestFrame before = copy_frame(f);
    const DeblockOptions o = base_options();
    run(f, o);
    CHECK(count_changed(before, f) == 0,
          "flat/linear-ramp content must be unchanged by deblocking");
    CHECK(guard_intact(f), "guard samples must not be touched");
  }
  std::cout << "  flat/ramp: identical output\n";
}

void test_step_filtering() {
  {
    TestFrame f = make_frame(16, 12);
    set_uniform_blocks(f, 4, TxSize::k16x16, 120);
    fill_pattern(f, Pattern::kVerticalStep);
    const int y = 17;
    const int center = f.luma_width() / 2;
    const int before_diff = std::abs(static_cast<int>(
        f.y[pix_index(y, f.y_stride, center) - 1]) -
        static_cast<int>(f.y[pix_index(y, f.y_stride, center)]));
    TestFrame before = copy_frame(f);
    const DeblockOptions o = base_options();
    run(f, o);
    const std::size_t changed = count_changed(before, f);
    const int after_diff = std::abs(static_cast<int>(
        f.y[pix_index(y, f.y_stride, center) - 1]) -
        static_cast<int>(f.y[pix_index(y, f.y_stride, center)]));
    CHECK(changed > 0, "a vertical step must be filtered");
    CHECK(after_diff < before_diff, "the step amplitude must be reduced");
    CHECK(guard_intact(f), "guard samples must not be touched");
    std::cout << "  vertical step: changed=" << changed
              << " edge|diff| " << before_diff << " -> " << after_diff << "\n";
  }
  {
    TestFrame f = make_frame(16, 16);
    set_uniform_blocks(f, 4, TxSize::k16x16, 120);
    fill_pattern(f, Pattern::kHorizontalStep);
    const int x = 17;
    const int center = f.luma_height() / 2;
    const int before_diff = std::abs(static_cast<int>(
        f.y[pix_index(center - 1, f.y_stride, x)]) -
        static_cast<int>(f.y[pix_index(center, f.y_stride, x)]));
    TestFrame before = copy_frame(f);
    const DeblockOptions o = base_options();
    run(f, o);
    const std::size_t changed = count_changed(before, f);
    const int after_diff = std::abs(static_cast<int>(
        f.y[pix_index(center - 1, f.y_stride, x)]) -
        static_cast<int>(f.y[pix_index(center, f.y_stride, x)]));
    CHECK(changed > 0, "a horizontal step must be filtered");
    CHECK(after_diff < before_diff, "the horizontal step must be reduced");
    std::cout << "  horizontal step: changed=" << changed
              << " edge|diff| " << before_diff << " -> " << after_diff << "\n";
  }
}

void test_filter_widths() {
  struct Case {
    int block4;
    TxSize tx;
    int col;
    int luma_pos;
    int chroma_pos;
  };
  const Case cases[] = {
      {4, TxSize::k4x4, 4, 1, 1},
      {2, TxSize::k8x8, 2, 3, 3},
      {4, TxSize::k16x16, 4, 6, 4},
      {8, TxSize::k32x32, 8, 8, 4},
  };
  for (const Case& c : cases) {
    TestFrame f = make_frame(16, 12);
    set_uniform_blocks(f, c.block4, c.tx, 200);
    fill_pattern(f, Pattern::kSoftEdge);
    const DeblockOptions o = base_options();
    const DeblockFrame view = f.view();
    const DeblockEdgeInfo y =
        av2enc::deblock::deblock_inspect_edge(view, o, 0, 0, 4, c.col);
    CHECK(y.on_screen == 1, "luma edge must be on screen");
    CHECK(y.apply_filter == 1, "luma edge must be filtered");
    CHECK(y.max_width_pos == c.luma_pos, "luma maxWidthPos class");
    CHECK(y.width >= 1 && y.width <= c.luma_pos,
          "luma chosen width within the max");
    const DeblockEdgeInfo u =
        av2enc::deblock::deblock_inspect_edge(view, o, 1, 0, 4, c.col);
    CHECK(u.on_screen == 1, "chroma edge must be on screen");
    CHECK(u.max_width_pos == c.chroma_pos, "chroma maxWidthPos class");
    std::cout << "  width class tx=" << av2enc::deblock::tx_width(c.tx)
              << " luma maxPos=" << y.max_width_pos
              << " width=" << y.width << ", chroma maxPos="
              << u.max_width_pos << " width=" << u.width << "\n";
  }

  // sbEdge clamps maxWidthNeg on horizontal 64-grid edges and vertical tile
  // edges.  At y=0 the edge is off-screen, so use the 64-grid at y=64.
  {
    TestFrame f = make_frame(32, 32);
    set_uniform_blocks(f, 8, TxSize::k32x32, 120);
    fill_pattern(f, Pattern::kSoftEdge);
    const DeblockOptions o = base_options();
    const DeblockFrame view = f.view();
    const DeblockEdgeInfo e =
        av2enc::deblock::deblock_inspect_edge(view, o, 0, 1, 16, 4);
    CHECK(e.on_screen == 1, "y=64 edge must be on screen");
    CHECK(e.sb_edge == 1, "y=64 is an sbEdge");
    CHECK(e.max_width_pos == 8 && e.max_width_neg == 6,
          "sbEdge clamps luma maxWidthNeg to 6");
    std::cout << "  sbEdge y=64: maxPos=" << e.max_width_pos
              << " maxNeg=" << e.max_width_neg << "\n";
  }
}

void test_texture_mask() {
  TestFrame f = make_frame(16, 12);
  set_uniform_blocks(f, 4, TxSize::k16x16, 220);
  fill_pattern(f, Pattern::kAlternating);
  const DeblockOptions o = base_options();
  const DeblockFrame view = f.view();
  const DeblockEdgeInfo e =
      av2enc::deblock::deblock_inspect_edge(view, o, 0, 0, 4, 8);
  CHECK(e.on_screen == 1, "checker edge on screen");
  CHECK(e.width == 0 && e.apply_filter == 1,
        "high-frequency content must be rejected by the side mask");
  TestFrame before = copy_frame(f);
  run(f, o);
  CHECK(count_changed(before, f) == 0,
        "high-frequency content must not be modified at aligned edges");
  std::cout << "  texture mask: width=" << e.width
            << " side=" << e.side << " qThr=" << e.q_thr << "\n";
}

void test_skip_and_boundary_strength() {
  DeblockOptions o = base_options();

  for (int skip = 0; skip <= 1; ++skip) {
    TestFrame f = make_frame(8, 6);
    set_tx_grid(f, 4, 2, 120, skip != 0);
    for (auto& m : f.mi_y) m.region_intra = 0;  // keep the skip flag effective
    for (auto& m : f.mi_u) m.region_intra = 0;
    for (auto& m : f.mi_v) m.region_intra = 0;
    fill_pattern(f, Pattern::kSoftEdge);
    const DeblockFrame view = f.view();
    const DeblockEdgeInfo e =
        av2enc::deblock::deblock_inspect_edge(view, o, 0, 0, 2, 2);
    CHECK(e.is_tx_edge == 1, "col=2 must be a transform edge");
    CHECK(e.is_block_edge == 0, "col=2 must not be a block edge");
    if (skip == 0) {
      CHECK(e.apply_filter == 1, "non-skip transform edge filters");
    } else {
      CHECK(e.apply_filter == 0, "skip transform edge does not filter");
    }
  }
  std::cout << "  skip / boundary strength: OK\n";
}

void test_lossless_sides() {
  TestFrame f = make_frame(16, 12);
  set_uniform_blocks(f, 4, TxSize::k16x16, 140);
  for (std::size_t i = 0; i < f.mi_y.size(); ++i) {
    const int r = static_cast<int>(i) / f.mi_cols;
    const int c = static_cast<int>(i) % f.mi_cols;
    (void)r;
    f.mi_y[i].lossless = (c < 4) ? 1 : 0;
    f.mi_u[i].lossless = (c < 4) ? 1 : 0;
    f.mi_v[i].lossless = (c < 4) ? 1 : 0;
  }
  fill_pattern(f, Pattern::kVerticalStep);
  const int lossless_edge = 16;  // MI col 4 is the first non-lossless block
  TestFrame before = copy_frame(f);
  const DeblockOptions o = base_options();
  const DeblockFrame view = f.view();
  const DeblockEdgeInfo e =
      av2enc::deblock::deblock_inspect_edge(view, o, 0, 0, 4, 4);
  CHECK(e.prev_lossless == 1 && e.curr_lossless == 0,
        "lossless flags must reach the edge decision");
  run(f, o);
  // The left block is lossless on both sides of its internal edges, so it must
  // remain untouched; the lossless side of the centre edge must also stay.
  bool left_unchanged = true;
  for (int y = 0; y < f.luma_height(); ++y) {
    for (int x = 0; x < lossless_edge; ++x) {
      if (before.y[pix_index(y, before.y_stride, x)] !=
          f.y[pix_index(y, f.y_stride, x)]) {
        left_unchanged = false;
      }
    }
  }
  CHECK(left_unchanged, "lossless side must not be modified");
  CHECK(count_changed(before, f) > 0, "the non-lossless side still filters");
  std::cout << "  lossless sides: left unchanged, right filtered\n";
}

void test_tile_edges() {
  // Vertical tile boundary at x=32 (MI col 8).
  for (int disable = 0; disable <= 1; ++disable) {
    TestFrame f = make_frame(16, 12);
    set_uniform_blocks(f, 4, TxSize::k16x16, 140);
    fill_pattern(f, Pattern::kVerticalStep);
    f.tile_cols = {0, 8};
    DeblockOptions o = base_options();
    o.disable_loopfilters_across_tiles = (disable != 0);
    const int center = f.luma_width() / 2;
    const int before_diff = std::abs(static_cast<int>(
        f.y[pix_index(17, f.y_stride, center) - 1]) -
        static_cast<int>(f.y[pix_index(17, f.y_stride, center)]));
    run(f, o);
    const int after_diff = std::abs(static_cast<int>(
        f.y[pix_index(17, f.y_stride, center) - 1]) -
        static_cast<int>(f.y[pix_index(17, f.y_stride, center)]));
    if (disable != 0) {
      CHECK(after_diff == before_diff,
            "disabled tile edge must not be filtered");
    } else {
      CHECK(after_diff < before_diff, "enabled tile edge must be filtered");
    }
  }

  // Horizontal tile boundary at y=32 (MI row 8).
  for (int disable = 0; disable <= 1; ++disable) {
    TestFrame f = make_frame(16, 16);
    set_uniform_blocks(f, 4, TxSize::k16x16, 140);
    fill_pattern(f, Pattern::kHorizontalStep);
    f.tile_rows = {0, 8};
    DeblockOptions o = base_options();
    o.disable_loopfilters_across_tiles = (disable != 0);
    const int before_diff = std::abs(static_cast<int>(
        f.y[pix_index(31, f.y_stride, 17)]) -
        static_cast<int>(f.y[pix_index(32, f.y_stride, 17)]));
    run(f, o);
    const int after_diff = std::abs(static_cast<int>(
        f.y[pix_index(31, f.y_stride, 17)]) -
        static_cast<int>(f.y[pix_index(32, f.y_stride, 17)]));
    if (disable != 0) {
      CHECK(after_diff == before_diff,
            "disabled horizontal tile edge must not be filtered");
    } else {
      CHECK(after_diff < before_diff, "enabled horizontal tile edge filters");
    }
  }
  std::cout << "  tile edges: disable/enable both exact\n";
}

void test_partial_and_offscreen() {
  TestFrame f = make_frame(18, 14);  // 72x56 luma, 36x28 chroma
  set_uniform_blocks(f, 16, TxSize::k32x32, 150);
  fill_pattern(f, Pattern::kVerticalStep);
  const DeblockOptions o = base_options();
  const DeblockFrame view = f.view();
  const DeblockEdgeInfo top =
      av2enc::deblock::deblock_inspect_edge(view, o, 0, 1, 0, 8);
  const DeblockEdgeInfo left =
      av2enc::deblock::deblock_inspect_edge(view, o, 0, 0, 8, 0);
  CHECK(top.on_screen == 0 && top.apply_filter == 0,
        "top frame edge is off-screen");
  CHECK(left.on_screen == 0 && left.apply_filter == 0,
        "left frame edge is off-screen");
  run(f, o);
  CHECK(guard_intact(f), "partial blocks must stay inside the MI-aligned plane");
  std::cout << "  partial blocks: guard intact, frame edges skipped\n";
}

void test_api_validation() {
  TestFrame f = make_frame(8, 6);
  set_uniform_blocks(f, 4, TxSize::k16x16, 120);
  fill_pattern(f, Pattern::kSoftEdge);
  DeblockOptions o = base_options();

  {
    DeblockFrame v = f.view();
    v.planes[0].data = nullptr;
    DeblockStatus s = DeblockStatus::kOk;
    CHECK(!av2enc::deblock::deblock_supported(v, o, &s) &&
              s == DeblockStatus::kInvalidArgument,
          "null plane must be rejected");
  }
  {
    DeblockFrame v = f.view();
    v.planes[1].stride = 1;
    DeblockStatus s = DeblockStatus::kOk;
    CHECK(!av2enc::deblock::deblock_supported(v, o, &s),
          "short stride must be rejected");
  }
  {
    DeblockFrame v = f.view();
    DeblockOptions bad = o;
    bad.bit_depth = 10;
    DeblockStatus s = DeblockStatus::kOk;
    CHECK(!av2enc::deblock::deblock_supported(v, bad, &s) &&
              s == DeblockStatus::kUnsupportedBitDepth,
          "10-bit must be rejected");
  }
  {
    DeblockFrame v = f.view();
    v.subsampling_x = 0;
    v.subsampling_y = 0;
    DeblockStatus s = DeblockStatus::kOk;
    CHECK(!av2enc::deblock::deblock_supported(v, o, &s) &&
              s == DeblockStatus::kUnsupportedSubsampling,
          "4:4:4 must be rejected");
  }
  {
    DeblockFrame v = f.view();
    const std::uint16_t bad = 99;
    v.tile_col_starts = &bad;
    v.num_tile_cols = 1;
    DeblockStatus s = DeblockStatus::kOk;
    CHECK(!av2enc::deblock::deblock_supported(v, o, &s),
          "out-of-range tile start must be rejected");
  }
  {
    DeblockFrame v = f.view();
    v.num_tile_cols = 1;
    v.tile_col_starts = nullptr;
    DeblockStatus s = DeblockStatus::kOk;
    CHECK(!av2enc::deblock::deblock_supported(v, o, &s),
          "missing tile starts must be rejected");
  }
  std::cout << "  API validation: OK\n";
}

void test_random_fuzz(std::mt19937& rng) {
  const int sizes[][2] = {{6, 4}, {8, 6}, {16, 12}, {24, 18}, {32, 24}};
  std::size_t total_changed = 0;
  for (int iter = 0; iter < 160; ++iter) {
    const auto& sz = sizes[iter % 5];
    TestFrame f = make_frame(sz[0], sz[1]);
    const Pattern patterns[] = {Pattern::kFlat,        Pattern::kVerticalStep,
                                Pattern::kHorizontalStep, Pattern::kRampX,
                                Pattern::kChecker,     Pattern::kNoise,
                                Pattern::kSoftEdge};
    fill_pattern(f, patterns[iter % 7]);
    set_random_all_planes(f, rng);
    set_random_tiles(f, rng);

    DeblockOptions o = base_options();
    o.apply_deblocking_filter[0] = (rng() & 1) != 0;
    o.apply_deblocking_filter[1] = (rng() & 1) != 0;
    o.apply_deblocking_filter[2] = (rng() & 1) != 0;
    o.apply_deblocking_filter[3] = (rng() & 1) != 0;
    o.disable_loopfilters_across_tiles = (rng() & 1) != 0;
    o.allow_df_sub_pu = (rng() & 1) != 0;
    o.frame_is_intra = (rng() & 3) != 0;
    o.enable_sdp = (rng() & 1) != 0;
    o.df_delta_q[0] = static_cast<int>(rng() % 5) - 2;
    o.df_delta_q[1] = static_cast<int>(rng() % 5) - 2;
    o.df_delta_q[2] = static_cast<int>(rng() % 5) - 2;
    o.df_delta_q[3] = static_cast<int>(rng() % 5) - 2;
    o.delta_q_u_ac = static_cast<int>(rng() % 9) - 4;
    o.delta_q_v_ac = static_cast<int>(rng() % 9) - 4;
    o.base_uv_ac_delta_q = static_cast<int>(rng() % 5) - 2;

    TestFrame before = copy_frame(f);
    DeblockFrame view = f.view();
    const DeblockStatus s = av2enc::deblock::deblock_frame(view, o);
    CHECK(s == DeblockStatus::kOk, "fuzz deblock_frame failed");
    CHECK(guard_intact(f), "fuzz guard samples must not be touched");
    total_changed += count_changed(before, f);

    // Determinism.
    TestFrame again = copy_frame(before);
    DeblockFrame view2 = again.view();
    (void)av2enc::deblock::deblock_frame(view2, o);
    CHECK(memcmp(f.y.data(), again.y.data(), f.y.size()) == 0 &&
              memcmp(f.u.data(), again.u.data(), f.u.size()) == 0 &&
              memcmp(f.v.data(), again.v.data(), f.v.size()) == 0,
          "deblocking must be deterministic");
  }
  CHECK(total_changed > 0, "fuzz corpus must exercise filtering");
  std::cout << "  random fuzz: 160 frames, changed samples=" << total_changed
            << "\n";
}

std::string repo_root_from_source() {
  std::string path = __FILE__;
  for (int i = 0; i < 5; ++i) {
    const std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) break;
    path.resize(slash);
  }
  return path;
}

std::string find_recon_frame(const std::string& root) {
  const std::string candidates[] = {
      root + "/decoded.y4m",
      root + "/testdata/shine_256x256_1f.y4m",
      root + "/testdata/shine_1280x720_60f.y4m",
  };
  for (const std::string& p : candidates) {
    std::ifstream in(p, std::ios::binary);
    if (in) return p;
  }
  return {};
}

void test_real_recon(const std::string& root) {
  const std::string path = find_recon_frame(root);
  if (path.empty()) {
    std::cout << "  real recon: skipped (no Y4M found)\n";
    return;
  }
  TestFrame f;
  if (!load_y4m_first_frame(path, &f)) {
    std::cout << "  real recon: skipped (could not parse " << path << ")\n";
    return;
  }
  set_uniform_blocks(f, 8, TxSize::k32x32, 120);
  TestFrame before = copy_frame(f);
  const DeblockOptions o = base_options();
  Timer t;
  run(f, o);
  const double ms = t.elapsed_ms();
  const std::size_t changed = count_changed(before, f);
  const double mad = mean_abs_delta(before, f);
  const int maxd = max_abs_delta(before, f);
  CHECK(guard_intact(f), "real recon guard samples must not be touched");
  CHECK(changed > 0, "real recon must be filtered");
  CHECK(maxd <= 255, "output stays in range");

  TestFrame again = copy_frame(before);
  run(again, o);
  CHECK(checksum(f) == checksum(again), "real recon must be deterministic");

  std::cout << "  real recon: " << path << " " << f.luma_width() << "x"
            << f.luma_height() << ", changed=" << changed << "/"
            << (static_cast<std::size_t>(f.mi_cols) *
                    static_cast<std::size_t>(f.mi_rows) +
                2 * (static_cast<std::size_t>(f.mi_cols) / 2) *
                    (static_cast<std::size_t>(f.mi_rows) / 2))
            << " samples, mean|d|=" << mad << ", max|d|=" << maxd
            << ", checksum=0x" << std::hex << checksum(f) << std::dec
            << ", cpu=" << ms << " ms\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string root = (argc > 1) ? argv[1] : repo_root_from_source();
  std::mt19937 rng(0xC0DEC0DEu);

  std::cout << "AV2 deblocking filter CPU reference tests\n";
  std::cout << "repo root: " << root << "\n";
  std::cout << "tables: MAX_SIDE_TABLE=296, MAX_DBL_FLT_LEN=8, DF_SHIFT="
            << av2enc::deblock::kDfShift << "\n";

  test_api_validation();
  test_flat_and_ramp();
  test_step_filtering();
  test_filter_widths();
  test_texture_mask();
  test_skip_and_boundary_strength();
  test_lossless_sides();
  test_tile_edges();
  test_partial_and_offscreen();
  test_random_fuzz(rng);
  test_real_recon(root);

  std::cout << "checks: " << g_checks << ", failures: " << g_failures << "\n";
  return g_failures == 0 ? 0 : 1;
}
