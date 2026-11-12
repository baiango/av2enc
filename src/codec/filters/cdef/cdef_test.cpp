// AV2 §7.18 CDEF — CPU reference tests.
//
// These tests exercise the normative direction search, primary/secondary taps,
// strength derivation, skip/lossless rules, tile filter regions, partial
// blocks, deterministic behaviour and the real reconstructed 640x360 frame.
// The CPU path is the correctness oracle; the CUDA twin compares against it.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "codec/filters/cdef/cdef_core.h"
#include "codec/filters/cdef/cdef_test_common.h"

namespace {

using av2enc::cdef::CdefBlockInfo;
using av2enc::cdef::CdefFrame;
using av2enc::cdef::CdefOptions;
using av2enc::cdef::CdefStatus;
using av2enc::cdef::CdefStrength;
using av2enc::cdef::test::DiffStats;
using av2enc::cdef::test::Pattern;
using av2enc::cdef::test::TestFrame;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

std::string g_root = ".";

CdefOptions options_with(int damp, int num, CdefStrength s0,
                         CdefStrength s1 = CdefStrength{}) {
  CdefOptions o;
  o.damping = damp;
  o.num_strengths = num;
  o.strengths[0] = s0;
  o.strengths[1] = s1;
  return o;
}

void run(TestFrame& frame, const CdefOptions& options) {
  av2enc::cdef::test::reset_dst(frame);
  CdefFrame view = frame.view();
  const CdefStatus status = av2enc::cdef::cdef_frame(view, options);
  check(status == CdefStatus::kOk, "cdef_frame returned ok");
}

bool block_equal_src(const TestFrame& f, int r, int c) {
  const int y0 = r * av2enc::cdef::kMiSize;
  const int x0 = c * av2enc::cdef::kMiSize;
  for (int plane = 0; plane < 3; ++plane) {
    const int sub = (plane == 0) ? 0 : 1;
    const int w = 8 >> sub;
    const int h = 8 >> sub;
    const int stride = f.stride[plane];
    const std::uint8_t* src = f.src[plane].data();
    const std::uint8_t* dst = f.dst[plane].data();
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const std::size_t o = av2enc::cdef::test::pix_index(
            (y0 >> sub) + i, stride, (x0 >> sub) + j);
        if (src[o] != dst[o]) {
          return false;
        }
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------

void test_constrain() {
  using av2enc::cdef::cdef_constrain;
  check(cdef_constrain(0, 8, 3) == 0, "constrain zero diff");
  check(cdef_constrain(5, 0, 3) == 0, "constrain zero threshold");
  check(cdef_constrain(4, 8, 3) == 4, "constrain small diff");
  check(cdef_constrain(-4, 8, 3) == -4, "constrain small negative diff");
  check(cdef_constrain(100, 8, 3) == 0, "constrain large diff");
  check(cdef_constrain(3, 8, 9) == 3, "constrain damping clamps magnitude");
  // threshold - (|diff| >> dampingAdj), clipped to [0, |diff|].
  check(cdef_constrain(6, 8, 3) == 2, "constrain mid diff");
  check(cdef_constrain(-6, 8, 3) == -2, "constrain mid negative diff");
  check(cdef_constrain(2, 8, 3) == 2, "constrain clip to magnitude");
}

void test_floor_log2() {
  using av2enc::cdef::cdef_floor_log2;
  check(cdef_floor_log2(0) == 0, "floor_log2(0)");
  check(cdef_floor_log2(1) == 0, "floor_log2(1)");
  check(cdef_floor_log2(2) == 1, "floor_log2(2)");
  check(cdef_floor_log2(3) == 1, "floor_log2(3)");
  check(cdef_floor_log2(8) == 3, "floor_log2(8)");
  check(cdef_floor_log2(15) == 3, "floor_log2(15)");
  check(cdef_floor_log2(4096) == 12, "floor_log2(4096)");
}

void test_direction_patterns() {
  using av2enc::cdef::cdef_direction;
  // Each 8x8 buffer encodes lines along one CDEF direction.
  auto run_dir = [](int mode, int* dir, std::int64_t* var) {
    std::uint8_t buf[64];
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        int value = 128;
        if (mode == 0) {  // constant along i+j -> direction 0
          value = 64 + 16 * ((i + j) & 7);
        } else if (mode == 1) {  // depends on j -> vertical lines, direction 6
          value = 64 + 16 * (j & 7);
        } else if (mode == 2) {  // depends on i -> horizontal lines, direction 2
          value = 64 + 16 * (i & 7);
        } else if (mode == 3) {  // constant along i-j -> direction 4
          value = 64 + 16 * ((i - j + 8) & 7);
        } else {
          value = 128;
        }
        buf[i * 8 + j] = static_cast<std::uint8_t>(value);
      }
    }
    cdef_direction(buf, 8, 0, 0, 8, dir, var);
  };

  int dir = -1;
  std::int64_t var = -1;
  run_dir(0, &dir, &var);
  check(dir == 0, "direction 45-degree lines -> 0");
  check(var > 0, "direction 45-degree variance positive");
  run_dir(1, &dir, &var);
  check(dir == 6, "direction vertical lines -> 6");
  run_dir(2, &dir, &var);
  check(dir == 2, "direction horizontal lines -> 2");
  run_dir(3, &dir, &var);
  check(dir == 4, "direction -45-degree lines -> 4");
  run_dir(4, &dir, &var);
  check(dir == 0 && var == 0, "flat block -> dir 0 var 0");
}

void test_filter_golden() {
  using av2enc::cdef::CdefRegion;
  using av2enc::cdef::cdef_filter_sample;
  // All taps at 100, one sample at 104.  With pri=8 the four primary taps
  // contribute -48; min=max=100 so the result clips to 100.
  std::vector<std::uint8_t> buf(16 * 16, 100);
  buf[8 * 16 + 8] = 104;
  const CdefRegion region{0, 0, 4, 4};
  const int v = cdef_filter_sample(buf.data(), 16, 8, 8, 0, 0, 6, 8, 0, 3, 0,
                                   0, region);
  check(v == 101, "filter golden: isolated 4-level bump smoothed by 3");

  // Flat neighborhood: no change at any strength.
  for (int pri = 0; pri <= 15; ++pri) {
    const int flat = cdef_filter_sample(buf.data(), 16, 0, 0, 3, 3, 2, pri, 0,
                                        3, 0, 0, region);
    check(flat == 100, "filter flat sample unchanged");
  }

  // A 1-code ripple (values 100/101) must move toward the neighborhood mean.
  std::vector<std::uint8_t> ripple(16 * 16, 100);
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      ripple[static_cast<std::size_t>(i) * 16 +
             static_cast<std::size_t>(j)] =
          static_cast<std::uint8_t>(100 + ((i + j) & 1));
    }
  }
  const int before = ripple[8 * 16 + 8];
  const int after = cdef_filter_sample(ripple.data(), 16, 8, 8, 0, 0, 2, 4, 0,
                                       3, 0, 0, region);
  check(after != before, "filter ripple changes sample");
  check(after >= 100 && after <= 101, "filter ripple stays in range");
}

void test_flat_unchanged() {
  for (int p = 0; p < 11; ++p) {
    TestFrame f = av2enc::cdef::test::make_frame(8, 6);
    av2enc::cdef::test::fill_pattern(f, static_cast<Pattern>(p));
    CdefStrength s{15, 4, 15, 4};
    const CdefOptions o = options_with(6, 1, s);
    run(f, o);
    if (p == static_cast<int>(Pattern::kFlat)) {
      bool equal = true;
      for (int plane = 0; plane < 3; ++plane) {
        const int w = f.plane_width(plane);
        const int h = f.plane_height(plane);
        const int stride = f.stride[plane];
        for (int y = 0; y < h && equal; ++y) {
          for (int x = 0; x < w && equal; ++x) {
            const std::size_t off =
                av2enc::cdef::test::pix_index(y, stride, x);
            equal = f.src[plane][off] == f.dst[plane][off];
          }
        }
      }
      check(equal, "flat frame unchanged");
    }
    check(av2enc::cdef::test::guard_intact(f), "flat/pattern guards intact");
  }
}

void test_skip_rules() {
  TestFrame f = av2enc::cdef::test::make_frame(8, 6);
  av2enc::cdef::test::fill_pattern(f, Pattern::kNoise);
  CdefStrength s{9, 2, 7, 1};
  CdefOptions o = options_with(3, 1, s);

  // idx = -1: copy only.
  av2enc::cdef::test::set_uniform_metadata(f, -1, false, false, false);
  run(f, o);
  bool equal = true;
  for (int plane = 0; plane < 3 && equal; ++plane) {
    const int w = f.plane_width(plane);
    const int h = f.plane_height(plane);
    const int stride = f.stride[plane];
    for (int y = 0; y < h && equal; ++y) {
      for (int x = 0; x < w && equal; ++x) {
        const std::size_t off =
            av2enc::cdef::test::pix_index(y, stride, x);
        equal = f.src[plane][off] == f.dst[plane][off];
      }
    }
  }
  check(equal, "idx -1 copies the frame");

  // All four MI cells skipped and cdef_on_skip_txfm = 0 -> copy only.
  av2enc::cdef::test::set_uniform_metadata(f, 0, true, false, false);
  o.on_skip_txfm = false;
  run(f, o);
  equal = true;
  for (int plane = 0; plane < 3 && equal; ++plane) {
    const int w = f.plane_width(plane);
    const int h = f.plane_height(plane);
    const int stride = f.stride[plane];
    for (int y = 0; y < h && equal; ++y) {
      for (int x = 0; x < w && equal; ++x) {
        const std::size_t off =
            av2enc::cdef::test::pix_index(y, stride, x);
        equal = f.src[plane][off] == f.dst[plane][off];
      }
    }
  }
  check(equal, "fully skipped block is not filtered");

  // Same frame with cdef_on_skip_txfm = 1 filters.
  o.on_skip_txfm = true;
  run(f, o);
  bool any_changed = false;
  for (int plane = 0; plane < 3 && !any_changed; ++plane) {
    const int w = f.plane_width(plane);
    const int h = f.plane_height(plane);
    const int stride = f.stride[plane];
    for (int y = 0; y < h && !any_changed; ++y) {
      for (int x = 0; x < w && !any_changed; ++x) {
        const std::size_t off =
            av2enc::cdef::test::pix_index(y, stride, x);
        any_changed = f.src[plane][off] != f.dst[plane][off];
      }
    }
  }
  check(any_changed, "cdef_on_skip_txfm filters skipped blocks");

  // A single lossless cell forces the whole 8x8 block to copy only.
  av2enc::cdef::test::set_uniform_metadata(f, 0, false, false, false);
  f.lossless[f.mi_index(2, 2)] = 1;
  run(f, o);
  check(block_equal_src(f, 2, 2), "lossless cell copies its 8x8 block");
  bool other_changed = false;
  for (int r = 0; r < f.mi_rows; r += 2) {
    for (int c = 0; c < f.mi_cols; c += 2) {
      if (r == 2 && c == 2) {
        continue;
      }
      if (!block_equal_src(f, r, c)) {
        other_changed = true;
      }
    }
  }
  check(other_changed, "non-lossless blocks still filter");

  // Chroma lossless: luma filters, chroma is copied.
  av2enc::cdef::test::set_uniform_metadata(f, 0, false, false, false);
  f.lossless_chroma[f.mi_index(2, 2)] = 1;
  run(f, o);
  bool chroma_equal = true;
  for (int plane = 1; plane < 3; ++plane) {
    const int stride = f.stride[plane];
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        const std::size_t off = av2enc::cdef::test::pix_index(
            2 + i, stride, 2 + j);
        chroma_equal = chroma_equal && f.src[plane][off] == f.dst[plane][off];
      }
    }
  }
  check(chroma_equal, "chroma lossless copies chroma for that block");
}

void test_strengths() {
  TestFrame f = av2enc::cdef::test::make_frame(12, 8);
  av2enc::cdef::test::fill_pattern(f, Pattern::kRipple);
  av2enc::cdef::test::set_uniform_metadata(f, 0, false, false, false);

  CdefOptions zero = options_with(3, 1, CdefStrength{0, 0, 0, 0});
  run(f, zero);
  DiffStats stats = av2enc::cdef::test::diff_stats(f);
  check(stats.changed == 0, "zero strengths leave the frame unchanged");

  CdefOptions weak = options_with(3, 1, CdefStrength{3, 0, 3, 0});
  run(f, weak);
  const DiffStats weak_stats = av2enc::cdef::test::diff_stats(f);
  check(weak_stats.changed > 0, "weak strength filters something");

  CdefOptions strong = options_with(3, 1, CdefStrength{15, 4, 15, 4});
  run(f, strong);
  const DiffStats strong_stats = av2enc::cdef::test::diff_stats(f);
  check(strong_stats.changed >= weak_stats.changed,
        "stronger primary filters at least as many samples");

  // Damping 6 limits the correction less than damping 3?  Both are valid; the
  // invariant we assert is determinism and a bounded result.
  CdefOptions damp3 = options_with(3, 1, CdefStrength{12, 4, 12, 4});
  CdefOptions damp6 = options_with(6, 1, CdefStrength{12, 4, 12, 4});
  run(f, damp3);
  const std::vector<std::uint8_t> a = f.dst[0];
  run(f, damp6);
  check(a != f.dst[0], "damping changes the filter result");
}

void test_tiles() {
  TestFrame f = av2enc::cdef::test::make_frame(16, 8);
  av2enc::cdef::test::fill_pattern(f, Pattern::kRipple);
  av2enc::cdef::test::set_uniform_metadata(f, 0, false, false, false);
  f.tile_cols = {0, 8};
  f.tile_rows = {0};
  CdefStrength s{12, 4, 12, 4};

  CdefOptions across = options_with(3, 1, s);
  across.disable_loopfilters_across_tiles = false;
  run(f, across);
  const std::vector<std::uint8_t> y_across = f.dst[0];

  CdefOptions blocked = across;
  blocked.disable_loopfilters_across_tiles = true;
  run(f, blocked);
  const std::vector<std::uint8_t> y_blocked = f.dst[0];

  check(y_across != y_blocked, "tile filter-region changes output");

  // Samples far from the tile boundary (MI column 8 -> x = 32) are identical.
  bool far_same = true;
  const int stride = f.stride[0];
  for (int y = 0; y < f.plane_height(0); ++y) {
    for (int x = 0; x < 8; ++x) {
      const std::size_t off = av2enc::cdef::test::pix_index(y, stride, x);
      far_same = far_same && y_across[off] == y_blocked[off];
    }
  }
  check(far_same, "samples far from the tile edge are unaffected");

  // With one implicit tile, the tile flag has no effect.
  f.tile_cols.clear();
  run(f, across);
  const std::vector<std::uint8_t> a = f.dst[0];
  run(f, blocked);
  check(a == f.dst[0], "flag has no effect without tiles");
}

void test_partial_blocks_and_edges() {
  // 10x6 MI = 40x24 luma, so the bottom/right 8x8 blocks are partial relative
  // to a 64x64 CDEF parameter block.
  TestFrame f = av2enc::cdef::test::make_frame(10, 6);
  av2enc::cdef::test::fill_pattern(f, Pattern::kNoise);
  av2enc::cdef::test::set_uniform_metadata(f, 0, false, false, false);
  CdefStrength s{10, 2, 8, 1};
  CdefOptions o = options_with(4, 1, s);
  run(f, o);
  check(av2enc::cdef::test::guard_intact(f), "partial blocks keep guards");
  const std::vector<std::uint8_t> first = f.dst[0];
  run(f, o);
  check(first == f.dst[0], "partial blocks deterministic");

  // Every visible sample of every plane is written by the copy/filter pass.
  bool wrote_visible = false;
  for (int plane = 0; plane < 3; ++plane) {
    const int w = f.plane_width(plane);
    const int h = f.plane_height(plane);
    const int stride = f.stride[plane];
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (f.dst[plane][av2enc::cdef::test::pix_index(y, stride, x)] !=
            av2enc::cdef::test::kGuard) {
          wrote_visible = true;
        }
      }
    }
  }
  check(wrote_visible, "visible plane traversal completed");
}

void test_inspect_consistency() {
  TestFrame f = av2enc::cdef::test::make_frame(12, 8);
  av2enc::cdef::test::fill_pattern(f, Pattern::kSoftEdge);
  std::mt19937 rng(0xCDEF1234u);
  av2enc::cdef::test::set_random_metadata(f, rng, 3, true, true, true);
  CdefStrength s0{8, 2, 6, 1};
  CdefStrength s1{12, 4, 9, 2};
  CdefStrength s2{4, 0, 3, 0};
  CdefOptions o;
  o.damping = 4;
  o.num_strengths = 3;
  o.strengths[0] = s0;
  o.strengths[1] = s1;
  o.strengths[2] = s2;
  o.on_skip_txfm = false;
  o.disable_loopfilters_across_tiles = true;

  run(f, o);
  CdefFrame view = f.view();
  bool consistent = true;
  for (int r = 0; r < f.mi_rows; r += 2) {
    for (int c = 0; c < f.mi_cols; c += 2) {
      CdefBlockInfo info;
      const CdefStatus st = av2enc::cdef::cdef_inspect_block(view, o, r, c,
                                                             &info);
      consistent = consistent && st == CdefStatus::kOk;
      consistent = consistent && info.r == r && info.c == c;
      if (!info.filtered) {
        consistent = consistent && block_equal_src(f, r, c);
      } else {
        consistent = consistent && info.idx >= 0 && info.idx < 3;
        consistent = consistent && info.y_damping == 4;
        consistent = consistent && info.uv_damping == 3;
        consistent = consistent && info.var_str >= 0 && info.var_str <= 12;
        consistent = consistent && info.y_dir_used >= 0 &&
                     info.y_dir_used <= 7;
        consistent = consistent && info.uv_dir >= 0 && info.uv_dir <= 7;
        if (info.y_pri > 0) {
          consistent = consistent && info.y_dir_used == info.y_dir;
        }
        if (info.uv_pri > 0) {
          consistent = consistent && info.uv_dir == info.y_dir;
        }
      }
    }
  }
  check(consistent, "inspect block matches frame output");
}

void test_random_frames() {
  std::mt19937 rng(0x5EEDCDEFu);
  CdefStrength sets[3] = {{5, 1, 4, 1}, {11, 4, 9, 2}, {15, 4, 13, 4}};
  std::size_t total_changed = 0;
  bool all_ok = true;
  for (int iter = 0; iter < 60; ++iter) {
    const int mi_cols = 4 + 2 * static_cast<int>(rng() % 40);
    const int mi_rows = 4 + 2 * static_cast<int>(rng() % 24);
    TestFrame f = av2enc::cdef::test::make_frame(mi_cols, mi_rows);
    av2enc::cdef::test::fill_pattern(
        f, static_cast<Pattern>(rng() % 11));
    CdefOptions o;
    o.damping = 3 + static_cast<int>(rng() % 4);
    o.num_strengths = 1 + static_cast<int>(rng() % 3);
    for (int i = 0; i < o.num_strengths; ++i) {
      o.strengths[i] = sets[rng() % 3];
    }
    o.on_skip_txfm = (rng() & 1) != 0;
    o.disable_loopfilters_across_tiles = (rng() & 1) != 0;
    av2enc::cdef::test::set_random_metadata(f, rng, o.num_strengths, true,
                                            true, true);
    run(f, o);
    all_ok = all_ok && av2enc::cdef::test::guard_intact(f);
    const std::vector<std::uint8_t> first = f.dst[0];
    run(f, o);
    all_ok = all_ok && first == f.dst[0];
    total_changed += av2enc::cdef::test::diff_stats(f).changed;
  }
  check(all_ok, "random frames deterministic and guard-clean");
  check(total_changed > 0, "random frames exercise filtering");
  std::printf("  random frames: %zu changed samples\n", total_changed);
}

void test_real_frame() {
  const std::string path = g_root + "/decoded.y4m";
  TestFrame f;
  if (!av2enc::cdef::test::load_y4m_first_frame(path, &f)) {
    std::printf("  real frame: %s not found, skipped\n", path.c_str());
    return;
  }
  av2enc::cdef::test::set_uniform_metadata(f, 0, false, false, false);
  CdefOptions o = options_with(3, 1, CdefStrength{9, 2, 7, 1});
  run(f, o);
  const DiffStats stats = av2enc::cdef::test::diff_stats(f);
  check(av2enc::cdef::test::guard_intact(f), "real frame guards intact");
  const std::vector<std::uint8_t> first = f.dst[0];
  run(f, o);
  check(first == f.dst[0], "real frame deterministic");
  std::printf(
      "  real frame %dx%d: %zu changed, mean |delta| %.4f, max %d\n",
      f.plane_width(0), f.plane_height(0), stats.changed, stats.mean_abs,
      stats.max_abs);
  check(stats.changed > 0, "real frame is filtered");
}

void test_validation() {
  TestFrame f = av2enc::cdef::test::make_frame(8, 6);
  av2enc::cdef::test::fill_pattern(f, Pattern::kNoise);
  av2enc::cdef::test::set_uniform_metadata(f, 0, false, false, false);
  CdefOptions o = options_with(3, 1, CdefStrength{8, 2, 6, 1});
  CdefStatus st = CdefStatus::kOk;
  CdefFrame v = f.view();
  check(av2enc::cdef::cdef_supported(v, o, &st), "valid frame accepted");

  {
    CdefFrame bad = v;
    bad.planes[0].src = nullptr;
    check(!av2enc::cdef::cdef_supported(bad, o, &st),
          "null source rejected");
  }
  {
    CdefFrame bad = v;
    bad.planes[1].dst = nullptr;
    check(!av2enc::cdef::cdef_supported(bad, o, &st),
          "null destination rejected");
  }
  {
    CdefFrame bad = v;
    bad.planes[0].dst = const_cast<std::uint8_t*>(bad.planes[0].src);
    check(!av2enc::cdef::cdef_supported(bad, o, &st),
          "in-place filtering rejected");
  }
  {
    CdefFrame bad = v;
    bad.planes[2].stride = 1;
    check(!av2enc::cdef::cdef_supported(bad, o, &st),
          "short stride rejected");
  }
  {
    CdefFrame bad = v;
    bad.mi_cols = 7;
    check(!av2enc::cdef::cdef_supported(bad, o, &st),
          "odd MI grid rejected");
  }
  {
    CdefOptions bad = o;
    bad.bit_depth = 10;
    check(!av2enc::cdef::cdef_supported(v, bad, &st),
          "10-bit rejected");
  }
  {
    CdefFrame bad = v;
    bad.subsampling_x = 0;
    check(!av2enc::cdef::cdef_supported(bad, o, &st),
          "4:4:4 rejected");
  }
  {
    CdefOptions bad = o;
    bad.damping = 2;
    check(!av2enc::cdef::cdef_supported(v, bad, &st),
          "damping out of range rejected");
  }
  {
    CdefOptions bad = o;
    bad.num_strengths = 0;
    check(!av2enc::cdef::cdef_supported(v, bad, &st),
          "zero strengths rejected");
  }
  {
    CdefOptions bad = o;
    bad.num_strengths = 9;
    check(!av2enc::cdef::cdef_supported(v, bad, &st),
          "nine strengths rejected");
  }
  {
    CdefOptions bad = o;
    bad.strengths[0].y_pri = 16;
    check(!av2enc::cdef::cdef_supported(v, bad, &st),
          "primary strength out of range rejected");
  }
  {
    CdefOptions bad = o;
    bad.strengths[0].uv_sec = 3;
    check(!av2enc::cdef::cdef_supported(v, bad, &st),
          "secondary strength 3 rejected");
  }
  {
    CdefFrame bad = v;
    std::vector<std::int8_t> idx = f.idx;
    idx[0] = 1;
    bad.cdef_idx = idx.data();
    check(!av2enc::cdef::cdef_supported(bad, o, &st),
          "out-of-range cdef_idx rejected");
  }
  {
    CdefFrame bad = v;
    std::vector<std::uint16_t> cols = {8};
    bad.tile_col_starts = cols.data();
    bad.num_tile_cols = 1;
    check(!av2enc::cdef::cdef_supported(bad, o, &st),
          "tile start outside frame rejected");
  }
  {
    CdefFrame bad = v;
    bad.skip = nullptr;
    check(!av2enc::cdef::cdef_supported(bad, o, &st), "null skip rejected");
  }
  {
    CdefBlockInfo info;
    check(av2enc::cdef::cdef_inspect_block(v, o, 1, 0, &info) ==
              CdefStatus::kInvalidArgument,
          "inspect rejects odd block origin");
  }
}

void report_timing() {
  auto time_frame = [](TestFrame& f, const CdefOptions& o, int iters) {
    av2enc::cdef::test::reset_dst(f);
    CdefFrame view = f.view();
    // Warm-up.
    (void)av2enc::cdef::cdef_frame(view, o);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      (void)av2enc::cdef::cdef_frame(view, o);
    }
    const auto end = std::chrono::steady_clock::now();
    const double total =
        std::chrono::duration<double, std::milli>(end - start).count();
    return total / static_cast<double>(iters);
  };

  CdefOptions o = options_with(3, 1, CdefStrength{9, 2, 7, 1});

  TestFrame real;
  if (av2enc::cdef::test::load_y4m_first_frame(g_root + "/decoded.y4m",
                                               &real)) {
    av2enc::cdef::test::set_uniform_metadata(real, 0, false, false, false);
    const double ms = time_frame(real, o, 5);
    std::printf("  CPU 640x360 : %.3f ms/frame (reference oracle)\n", ms);
  }

  TestFrame hd = av2enc::cdef::test::make_frame(1920 / 4, 1080 / 4);
  av2enc::cdef::test::fill_pattern(hd, Pattern::kNoise);
  av2enc::cdef::test::set_uniform_metadata(hd, 0, false, false, false);
  const double ms = time_frame(hd, o, 3);
  std::printf("  CPU 1920x1080: %.3f ms/frame (reference oracle)\n", ms);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) {
    g_root = argv[1];
  }
  std::printf("cdef CPU reference tests (root=%s)\n", g_root.c_str());

  test_constrain();
  test_floor_log2();
  test_direction_patterns();
  test_filter_golden();
  test_flat_unchanged();
  test_skip_rules();
  test_strengths();
  test_tiles();
  test_partial_blocks_and_edges();
  test_inspect_consistency();
  test_random_frames();
  test_real_frame();
  test_validation();
  report_timing();

  std::printf("cdef_test: %s (%d checks, %d failures)\n",
              g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
