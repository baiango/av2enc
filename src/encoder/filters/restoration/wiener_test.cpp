// CPU reference tests for AV2 §7.20 loop restoration / §7.20.3 non-separable
// Wiener filter.
//
// Coverage:
//   * RESTORE_NONE planes pass through (LrFrame := CdefFrame)
//   * per-unit Wiener coefficients, luma (32 taps) and chroma (12 + 6 taps)
//   * chroma-from-luma taps and all four cfl_ds_filter_index modes
//   * stripe sample sourcing: inside the 64-line stripe from CdefFrame,
//     outside from CurrFrame, clamped to two lines
//   * frame-level filters with a per-class subclass grid
//   * per-unit LrType mix, including RESTORE_NONE units
//   * partial right/bottom units, odd MI dimensions
//   * tile clipping with disable_loopfilters_across_tiles
//   * lossless segments and BRU-inactive regions
//   * validation errors (GDF, PC-Wiener, missing class grid, aliasing)
//   * randomized CPU-vs-naive-oracle comparison across all planes
//
// Usage: ./wiener_test

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "encoder/filters/restoration/restoration.h"
#include "encoder/filters/restoration/restoration_test_common.h"

namespace {

using namespace av2enc::restoration;
using namespace av2enc::restoration::test;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                    \
  do {                                                                      \
    ++g_checks;                                                             \
    if (!(cond)) {                                                          \
      ++g_failures;                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg));  \
    }                                                                       \
  } while (0)

TestFrame make_frame(int mi_cols, int mi_rows) {
  TestFrame f;
  f.mi_cols = mi_cols;
  f.mi_rows = mi_rows;
  f.allocate();
  return f;
}

struct LrTest {
  std::vector<std::uint8_t> type[3];
  std::vector<std::int16_t> coeffs[3];
  std::vector<std::int16_t> frame_coeffs[3];
  std::vector<std::uint8_t> subclass;
};

int units_for(int unit_size, int frame_size) {
  return restoration_count_units(unit_size, frame_size);
}

void setup_params(const TestFrame& f, int unit_size, RestorationType t,
                  LoopRestorationParams* p, LrTest* tst,
                  int chroma_unit_size = -1) {
  for (int pl = 0; pl < 3; ++pl) {
    const int us = (pl == 0 || chroma_unit_size <= 0) ? unit_size
                                                      : chroma_unit_size;
    p->frame_type[pl] = t;
    p->unit_size[pl] = us;
    const int sub = (pl == 0) ? 0 : 1;
    const int rows = units_for(us, (f.mi_rows * 4) >> sub);
    const int cols = units_for(us, (f.mi_cols * 4) >> sub);
    p->unit_rows[pl] = rows;
    p->unit_cols[pl] = cols;
    tst->type[pl].assign(static_cast<std::size_t>(rows * cols),
                         static_cast<std::uint8_t>(kRestoreWienerNs));
    tst->coeffs[pl].assign(
        static_cast<std::size_t>(rows * cols) * kWienerNsCoeffStride, 0);
    p->unit_type[pl] = tst->type[pl].data();
    p->unit_coeffs[pl] = tst->coeffs[pl].data();
  }
}

bool compare_all(const TestFrame& got, const TestFrame& want,
                 const char* msg) {
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < got.height[p]; ++y) {
      for (int x = 0; x < got.width[p]; ++x) {
        if (got.at(got.lr[p], p, y, x) != want.at(want.lr[p], p, y, x)) {
          std::fprintf(stderr,
                       "  mismatch plane=%d (%d,%d): got %d want %d\n", p, x,
                       y, got.at(got.lr[p], p, y, x),
                       want.at(want.lr[p], p, y, x));
          CHECK(false, msg);
          return false;
        }
      }
    }
  }
  CHECK(true, msg);
  return true;
}

// Builds the naive oracle output into `out` (same layout as `f.lr`).
void run_naive(const TestFrame& f, const LoopRestorationParams& params,
               const NaiveLrFrame& n, TestFrame* out) {
  *out = f;
  for (int p = 0; p < 3; ++p) {
    std::fill(out->lr[p].begin(), out->lr[p].end(), kGuard);
  }
  NaiveLrFrame nn = n;
  nn.src = &f;
  nn.params = &params;
  nn.filter_class_grid = params.filter_class_grid;
  nn.subclass_lookup = params.subclass_lookup;
  nn.subclass_grid = params.subclass_grid;
  naive_loop_restoration(nn, out->lr);
}

// --- tests -----------------------------------------------------------------

void test_identity_copy() {
  TestFrame f = make_frame(20, 12);
  Rng rng(1);
  fill_random(f, rng);
  LoopRestorationParams params;
  LrTest tst;
  setup_params(f, 64, kRestoreWienerNs, &params, &tst);
  RestorationFrame view = f.view();
  CHECK(loop_restoration_frame(view, params) == RestorationStatus::kOk,
        "identity restoration ok");
  bool same = true;
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p]; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        if (f.at(f.lr[p], p, y, x) != f.at(f.cdef[p], p, y, x)) {
          same = false;
        }
      }
    }
  }
  CHECK(same, "zero coefficients reproduce CdefFrame exactly");
  CHECK(f.lr_guard_intact(), "LrFrame guard samples untouched");

  // RESTORE_NONE also copies the frame through.
  TestFrame g = make_frame(20, 12);
  Rng rng2(2);
  fill_random(g, rng2);
  LoopRestorationParams none;
  LrTest tst2;
  setup_params(g, 64, kRestoreNone, &none, &tst2);
  CHECK(loop_restoration_frame(g.view(), none) == RestorationStatus::kOk,
        "RESTORE_NONE ok");
  bool copied = true;
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < g.height[p]; ++y) {
      for (int x = 0; x < g.width[p]; ++x) {
        if (g.at(g.lr[p], p, y, x) != g.at(g.cdef[p], p, y, x)) {
          copied = false;
        }
      }
    }
  }
  CHECK(copied, "RESTORE_NONE copies CdefFrame to LrFrame");
  std::cout << "  identity copy: zero coeffs and RESTORE_NONE\n";
}

void test_per_unit_filters() {
  TestFrame f = make_frame(24, 18);  // 96x72
  Rng rng(3);
  fill_random(f, rng);
  LoopRestorationParams params;
  LrTest tst;
  setup_params(f, 64, kRestoreWienerNs, &params, &tst);
  // Give every actual restoration unit a distinct symmetric filter.
  const int luma_units =
      static_cast<int>(tst.coeffs[0].size() / kWienerNsCoeffStride);
  for (int unit = 0; unit < luma_units; ++unit) {
    std::int16_t* c = tst.coeffs[0].data() +
                      static_cast<std::size_t>(unit) * kWienerNsCoeffStride;
    c[0] = static_cast<std::int16_t>(8 * (unit + 1));
    c[1] = static_cast<std::int16_t>(4 * unit);
    c[2] = static_cast<std::int16_t>(6);
    c[3] = static_cast<std::int16_t>(-3 * unit);
    c[15] = 2;  // (3,3)/(-3,-3)
  }
  for (int pl = 1; pl < 3; ++pl) {
    const int chroma_units =
        static_cast<int>(tst.coeffs[pl].size() / kWienerNsCoeffStride);
    for (int unit = 0; unit < chroma_units; ++unit) {
      std::int16_t* cc = tst.coeffs[pl].data() +
                         static_cast<std::size_t>(unit) * kWienerNsCoeffStride;
      cc[4] = static_cast<std::int16_t>(10 + unit);
      cc[5] = static_cast<std::int16_t>(-4);
      cc[10] = 3;
    }
  }
  TestFrame want;
  NaiveLrFrame n;
  run_naive(f, params, n, &want);
  CHECK(loop_restoration_frame(f.view(), params) == RestorationStatus::kOk,
        "per-unit restoration ok");
  compare_all(f, want, "per-unit Wiener matches the naive oracle");
  CHECK(f.lr_guard_intact(), "LrFrame guard samples untouched");
  std::cout << "  per-unit filters: 4 luma + 4 chroma units\n";
}

void test_chroma_from_luma_and_cfl() {
  for (int cfl = 0; cfl < 4; ++cfl) {
    TestFrame f = make_frame(16, 10);  // 64x40 luma
    Rng rng(static_cast<std::uint32_t>(10 + cfl));
    fill_random(f, rng);
    LoopRestorationParams params;
    LrTest tst;
    setup_params(f, 64, kRestoreWienerNs, &params, &tst);
    for (int pl = 0; pl < 3; ++pl) {
      std::int16_t* c = tst.coeffs[pl].data();
      if (pl == 0) {
        c[0] = 5;
      } else {
        // Only the chroma-from-luma taps (indices 6..17) are nonzero.
        for (int i = 0; i < 12; ++i) {
          c[6 + i] = static_cast<std::int16_t>((i % 5) - 2);
        }
      }
    }
    RestorationFrame view = f.view();
    view.cfl_ds_filter_index = cfl;
    TestFrame want;
    NaiveLrFrame n;
    n.cfl_ds_filter_index = cfl;
    run_naive(f, params, n, &want);
    CHECK(loop_restoration_frame(view, params) == RestorationStatus::kOk,
          "chroma-from-luma ok");
    compare_all(f, want, "chroma-from-luma matches the naive oracle");
  }
  std::cout << "  chroma from luma: cfl_ds_filter_index 0..3\n";
}

void test_stripe_sourcing() {
  // 64x128 luma: two 64-line stripes (stripe 0 covers rows 0..55).
  TestFrame f = make_frame(16, 32);
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p]; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        f.at(f.cdef[p], p, y, x) = 100;
        f.at(f.curr[p], p, y, x) = 200;
      }
    }
  }
  LoopRestorationParams params;
  LrTest tst;
  setup_params(f, 64, kRestoreWienerNs, &params, &tst);
  // idx 0 covers both (1,0) and (-1,0), so it reads one row across.
  tst.coeffs[0][0] = 128;
  tst.coeffs[1][0] = 128;
  tst.coeffs[2][0] = 128;

  CHECK(loop_restoration_frame(f.view(), params) == RestorationStatus::kOk,
        "stripe restoration ok");
  // Interior sample: all taps inside the stripe -> 100.
  CHECK(f.at(f.lr[0], 0, 10, 10) == 100, "interior sample stays on CdefFrame");
  // Row 55 is the last stripe-0 row; the +1 tap at row 56 is outside.
  CHECK(f.at(f.lr[0], 0, 55, 10) == 200,
        "tap below the stripe uses CurrFrame");
  // Row 56 is outside stripe 0 but inside stripe 1 (rows 56..119).
  CHECK(f.at(f.lr[0], 0, 56, 10) == 100,
        "row 56 belongs to the next stripe and reads CdefFrame");
  // The clamp to two lines: row 0 reads row 0 (not -1/-2) from CurrFrame.
  CHECK(f.at(f.lr[0], 0, 0, 10) == 100, "clamped tap inside stripe");

  // Naive oracle agreement on the same content (covers many stripes).
  TestFrame want;
  NaiveLrFrame n;
  run_naive(f, params, n, &want);
  // The oracle must be run before f.lr is compared; both used the same inputs.
  compare_all(f, want, "stripe sourcing matches the naive oracle");
  std::cout << "  stripe sourcing: CurrFrame outside, CdefFrame inside\n";
}

void test_frame_filters_and_subclass() {
  TestFrame f = make_frame(16, 12);
  Rng rng(44);
  fill_random(f, rng);
  LoopRestorationParams params;
  LrTest tst;
  setup_params(f, 64, kRestoreWienerNs, &params, &tst);
  params.frame_filters_on[0] = 1;
  params.num_filter_classes = 2;
  tst.subclass.assign(static_cast<std::size_t>(f.mi_cols * f.mi_rows), 0);
  for (int y = 0; y < f.mi_rows; ++y) {
    for (int x = 0; x < f.mi_cols; ++x) {
      tst.subclass[static_cast<std::size_t>(y * f.mi_cols + x)] =
          static_cast<std::uint8_t>((x + y) & 1);
    }
  }
  params.subclass_grid = tst.subclass.data();
  for (int pl = 0; pl < 3; ++pl) {
    tst.frame_coeffs[pl].assign(2 * kWienerNsCoeffStride, 0);
    params.frame_coeffs[pl] = tst.frame_coeffs[pl].data();
    if (pl == 0) {
      tst.frame_coeffs[0][0] = 20;
      tst.frame_coeffs[0][1] = -10;
      tst.frame_coeffs[0][18 + 0] = -30;
      tst.frame_coeffs[0][18 + 2] = 15;
    } else {
      tst.frame_coeffs[pl][0] = 9;
      tst.frame_coeffs[pl][18 + 4] = -6;
    }
  }
  RestorationFrame view = f.view();
  TestFrame want;
  NaiveLrFrame n;
  run_naive(f, params, n, &want);
  CHECK(loop_restoration_frame(view, params) == RestorationStatus::kOk,
        "frame filters ok");
  compare_all(f, want, "frame-level filters + subclass grid match the oracle");
  std::cout << "  frame filters: 2 pixel classes\n";
}

void test_class_grid_and_lookup() {
  TestFrame f = make_frame(16, 12);
  Rng rng(45);
  fill_random(f, rng);
  LoopRestorationParams params;
  LrTest tst;
  setup_params(f, 64, kRestoreWienerNs, &params, &tst);
  params.frame_filters_on[0] = 1;
  params.num_filter_classes = 2;

  // FilterClass lives in 0..15; SubclassLookup maps it to one of the two
  // coefficient classes.  This is the §7.20.3 shape, distinct from the legacy
  // pre-looked-up subclass_grid.
  std::vector<std::uint8_t> cls(static_cast<std::size_t>(f.mi_cols * f.mi_rows),
                                0);
  for (std::size_t i = 0; i < cls.size(); ++i) {
    cls[i] = static_cast<std::uint8_t>(rng.range(0, 15));
  }
  std::vector<std::uint8_t> lookup(16, 0);
  for (int i = 0; i < 16; ++i) {
    lookup[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i & 1);
  }
  params.filter_class_grid = cls.data();
  params.subclass_lookup = lookup.data();
  for (int pl = 0; pl < 3; ++pl) {
    tst.frame_coeffs[pl].assign(2 * kWienerNsCoeffStride, 0);
    params.frame_coeffs[pl] = tst.frame_coeffs[pl].data();
    if (pl == 0) {
      tst.frame_coeffs[0][0] = 17;
      tst.frame_coeffs[0][18 + 1] = -9;
    } else {
      tst.frame_coeffs[pl][2] = 7;
      tst.frame_coeffs[pl][18 + 5] = -5;
    }
  }
  RestorationFrame view = f.view();
  TestFrame want;
  NaiveLrFrame n;
  run_naive(f, params, n, &want);
  CHECK(loop_restoration_frame(view, params) == RestorationStatus::kOk,
        "filter-class grid + lookup ok");
  compare_all(f, want,
              "FilterClass + SubclassLookup matches the oracle");
  std::cout << "  frame filters: FilterClass grid + SubclassLookup\n";
}


void test_per_unit_types_and_partial() {
  // 100x36 luma, unit 64 -> 2x1 luma units, 1 unit for chroma 50x18.
  TestFrame f = make_frame(25, 9);
  Rng rng(55);
  fill_random(f, rng);
  LoopRestorationParams params;
  LrTest tst;
  setup_params(f, 64, kRestoreSwitchable, &params, &tst);
  CHECK(tst.type[0].size() >= 2, "expected two luma restoration units");
  // Luma: unit 0 filtered, unit 1 disabled.
  tst.type[0][0] = static_cast<std::uint8_t>(kRestoreWienerNs);
  if (tst.type[0].size() > 1) {
    tst.type[0][1] = static_cast<std::uint8_t>(kRestoreNone);
  }
  for (int pl = 1; pl < 3; ++pl) {
    tst.type[pl][0] = static_cast<std::uint8_t>(kRestoreWienerNs);
  }
  for (int pl = 0; pl < 3; ++pl) {
    for (std::size_t i = 0; i < tst.coeffs[pl].size() / 18; ++i) {
      std::int16_t* c = tst.coeffs[pl].data() + i * 18;
      c[0] = static_cast<std::int16_t>(pl == 0 ? 12 : 7);
    }
  }
  TestFrame want;
  NaiveLrFrame n;
  run_naive(f, params, n, &want);
  CHECK(loop_restoration_frame(f.view(), params) == RestorationStatus::kOk,
        "switchable ok");
  compare_all(f, want, "per-unit LrType mix matches the oracle");
  // Disabled luma unit is a straight copy.
  bool copy = true;
  for (int y = 0; y < f.height[0]; ++y) {
    for (int x = 64; x < f.width[0]; ++x) {
      if (f.at(f.lr[0], 0, y, x) != f.at(f.cdef[0], 0, y, x)) {
        copy = false;
      }
    }
  }
  CHECK(copy, "RESTORE_NONE unit is a straight copy");
  CHECK(f.lr_guard_intact(), "LrFrame guard samples untouched");
  std::cout << "  per-unit types: 100x36 frame, mixed NONE/WIENER\n";
}

void test_tile_clipping() {
  TestFrame f = make_frame(32, 32);
  Rng rng(66);
  fill_random(f, rng);
  LoopRestorationParams params;
  LrTest tst;
  // §6.17.7.7 check_ru_size: maxPlaneRuSize = max(64, 32 << 1) = 64, so the
  // 64x64 synthetic tiles below are conforming.  This also exercises distinct
  // luma/chroma unit sizes.
  setup_params(f, 64, kRestoreWienerNs, &params, &tst, /*chroma=*/32);
  for (int pl = 0; pl < 3; ++pl) {
    tst.coeffs[pl][0] = 30;
    tst.coeffs[pl][1] = -20;
    tst.coeffs[pl][12] = 10;
  }
  const int n = f.mi_cols * f.mi_rows;
  std::vector<std::uint16_t> mcs(static_cast<std::size_t>(n));
  std::vector<std::uint16_t> mce(static_cast<std::size_t>(n));
  std::vector<std::uint16_t> mrs(static_cast<std::size_t>(n));
  std::vector<std::uint16_t> mre(static_cast<std::size_t>(n));
  for (int y = 0; y < f.mi_rows; ++y) {
    for (int x = 0; x < f.mi_cols; ++x) {
      const std::size_t i = static_cast<std::size_t>(y * f.mi_cols + x);
      mcs[i] = static_cast<std::uint16_t>(x < 16 ? 0 : 16);
      mce[i] = static_cast<std::uint16_t>(x < 16 ? 16 : 32);
      mrs[i] = static_cast<std::uint16_t>(y < 16 ? 0 : 16);
      mre[i] = static_cast<std::uint16_t>(y < 16 ? 16 : 32);
    }
  }
  RestorationFrame view = f.view();
  view.mi_col_start = mcs.data();
  view.mi_col_end = mce.data();
  view.mi_row_start = mrs.data();
  view.mi_row_end = mre.data();
  view.disable_loopfilters_across_tiles = 1;

  NaiveLrFrame nv;
  nv.mcs = mcs.data();
  nv.mce = mce.data();
  nv.mrs = mrs.data();
  nv.mre = mre.data();
  nv.disable_across_tiles = 1;
  TestFrame want;
  run_naive(f, params, nv, &want);
  CHECK(loop_restoration_frame(view, params) == RestorationStatus::kOk,
        "tile-clipped restoration ok");
  compare_all(f, want, "tile clipping matches the naive oracle");
  std::cout << "  tile clipping: 2x2 tiles, disable_loopfilters_across_tiles\n";
}

void test_lossless_and_bru() {
  TestFrame f = make_frame(24, 16);
  Rng rng(77);
  fill_random(f, rng);
  LoopRestorationParams params;
  LrTest tst;
  setup_params(f, 64, kRestoreWienerNs, &params, &tst);
  for (int pl = 0; pl < 3; ++pl) {
    tst.coeffs[pl][0] = 16;
  }
  std::vector<std::uint8_t> seg(static_cast<std::size_t>(f.mi_cols * f.mi_rows),
                                0);
  for (int y = 0; y < f.mi_rows; ++y) {
    for (int x = 0; x < f.mi_cols; ++x) {
      seg[static_cast<std::size_t>(y * f.mi_cols + x)] =
          static_cast<std::uint8_t>(y >= f.mi_rows / 2 ? 1 : 0);
    }
  }
  std::uint8_t lossless[8] = {0, 1, 0, 0, 0, 0, 0, 0};
  RestorationFrame view = f.view();
  view.segment_ids = seg.data();
  view.chroma_segment_ids = seg.data();
  view.lossless = lossless;
  NaiveLrFrame nv;
  nv.segment_ids = seg.data();
  nv.chroma_segment_ids = seg.data();
  nv.lossless = lossless;
  TestFrame want;
  run_naive(f, params, nv, &want);
  CHECK(loop_restoration_frame(view, params) == RestorationStatus::kOk,
        "lossless restoration ok");
  compare_all(f, want, "lossless segments match the naive oracle");
  bool untouched = true;
  for (int p = 0; p < 3; ++p) {
    for (int y = f.height[p] / 2; y < f.height[p]; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        if (f.at(f.lr[p], p, y, x) != f.at(f.cdef[p], p, y, x)) {
          untouched = false;
        }
      }
    }
  }
  CHECK(untouched, "lossless segment rows are copied unfiltered");

  // BRU-inactive superblock region: bottom half skipped.
  TestFrame g = make_frame(24, 16);
  Rng rng2(78);
  fill_random(g, rng2);
  std::vector<std::uint8_t> bru(static_cast<std::size_t>(g.mi_cols * g.mi_rows),
                                kBruActive);
  for (int y = 0; y < g.mi_rows / 2; ++y) {
    for (int x = 0; x < g.mi_cols; ++x) {
      bru[static_cast<std::size_t>(y * g.mi_cols + x)] = kBruInactive;
    }
  }
  RestorationFrame view2 = g.view();
  view2.use_bru = 1;
  view2.bru_modes = bru.data();
  CHECK(loop_restoration_frame(view2, params) == RestorationStatus::kOk,
        "BRU restoration ok");
  bool skipped = true;
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < g.height[p] / 2; ++y) {
      for (int x = 0; x < g.width[p]; ++x) {
        if (g.at(g.lr[p], p, y, x) != g.at(g.cdef[p], p, y, x)) {
          skipped = false;
        }
      }
    }
  }
  CHECK(skipped, "BRU-inactive superblock rows are copied unfiltered");
  std::cout << "  lossless segments and BRU-inactive regions: skipped\n";
}

void test_validation() {
  TestFrame f = make_frame(16, 8);
  LoopRestorationParams params;
  LrTest tst;
  setup_params(f, 64, kRestoreWienerNs, &params, &tst);
  RestorationFrame view = f.view();
  CHECK(loop_restoration_frame(view, params) == RestorationStatus::kOk,
        "baseline ok");

  LoopRestorationParams gdf = params;
  gdf.gdf_frame_enable = 1;
  CHECK(loop_restoration_frame(view, gdf) ==
            RestorationStatus::kUnsupportedGdf,
        "GDF rejected because its tables are not in the spec text");

  LoopRestorationParams pc = params;
  pc.frame_type[0] = kRestorePcWiener;
  CHECK(loop_restoration_frame(view, pc) ==
            RestorationStatus::kUnsupportedPcWiener,
        "PC-Wiener rejected");

  LoopRestorationParams sw = params;
  sw.frame_type[0] = kRestoreSwitchable;
  tst.type[0][0] = static_cast<std::uint8_t>(kRestorePcWiener);
  CHECK(loop_restoration_frame(view, sw) ==
            RestorationStatus::kUnsupportedPcWiener,
        "switchable unit selecting PC-Wiener rejected");
  tst.type[0][0] = static_cast<std::uint8_t>(kRestoreWienerNs);

  std::vector<std::int16_t> frame_coeffs(4 * kWienerNsCoeffStride, 0);
  LoopRestorationParams cls = params;
  cls.frame_filters_on[0] = 1;
  cls.num_filter_classes = 4;
  for (int pl = 0; pl < 3; ++pl) {
    cls.frame_coeffs[pl] = frame_coeffs.data();
  }
  CHECK(loop_restoration_frame(view, cls) ==
            RestorationStatus::kMissingSubclassGrid,
        "frame filters without a subclass grid rejected");

  LoopRestorationParams cls_no_lookup = params;
  cls_no_lookup.frame_filters_on[0] = 1;
  cls_no_lookup.num_filter_classes = 2;
  for (int pl = 0; pl < 3; ++pl) {
    cls_no_lookup.frame_coeffs[pl] = frame_coeffs.data();
  }
  cls_no_lookup.filter_class_grid = tst.type[0].data();
  CHECK(loop_restoration_frame(view, cls_no_lookup) ==
            RestorationStatus::kMissingSubclassGrid,
        "FilterClass grid without SubclassLookup rejected");

  LoopRestorationParams bad_classes = params;
  bad_classes.frame_filters_on[0] = 1;
  bad_classes.num_filter_classes = 5;  // not in Decode_Num_Filter_Classes
  bad_classes.frame_coeffs[0] = frame_coeffs.data();
  bad_classes.subclass_grid = tst.type[0].data();
  CHECK(loop_restoration_frame(view, bad_classes) ==
            RestorationStatus::kInvalidArgument,
        "num_filter_classes outside Decode_Num_Filter_Classes rejected");

  LoopRestorationParams small_luma = params;
  small_luma.unit_size[0] = 32;
  CHECK(loop_restoration_frame(view, small_luma) ==
            RestorationStatus::kNonConformingUnitSize,
        "luma restoration unit below one stripe rejected");

  LoopRestorationParams bad_chroma = params;
  bad_chroma.unit_size[1] = 32;
  bad_chroma.unit_size[2] = 64;
  CHECK(loop_restoration_frame(view, bad_chroma) ==
            RestorationStatus::kNonConformingUnitSize,
        "LoopRestorationSize[2] != [1] rejected");

  // A 64-sample tile boundary with unit sizes that imply maxPlaneRu = 128
  // violates §6.17.7.7 check_ru_size().
  const int n = f.mi_cols * f.mi_rows;
  std::vector<std::uint16_t> mcs(static_cast<std::size_t>(n), 0);
  std::vector<std::uint16_t> mce(static_cast<std::size_t>(n), 0);
  std::vector<std::uint16_t> mrs(static_cast<std::size_t>(n), 0);
  std::vector<std::uint16_t> mre(static_cast<std::size_t>(n), 0);
  for (int y = 0; y < f.mi_rows; ++y) {
    for (int x = 0; x < f.mi_cols; ++x) {
      const std::size_t i = static_cast<std::size_t>(y * f.mi_cols + x);
      mcs[i] = static_cast<std::uint16_t>(x < 8 ? 0 : 8);
      mce[i] = static_cast<std::uint16_t>(x < 8 ? 8 : 16);
      mrs[i] = 0;
      mre[i] = static_cast<std::uint16_t>(f.mi_rows);
    }
  }
  RestorationFrame tiles = view;
  tiles.mi_col_start = mcs.data();
  tiles.mi_col_end = mce.data();
  tiles.mi_row_start = mrs.data();
  tiles.mi_row_end = mre.data();
  CHECK(loop_restoration_frame(tiles, params) ==
            RestorationStatus::kNonConformingUnitSize,
        "tile boundary not divisible by maxPlaneRuSize rejected");

  RestorationFrame alias = view;
  alias.lr[0].data = alias.cdef[0].data;
  CHECK(loop_restoration_frame(alias, params) ==
            RestorationStatus::kInvalidArgument,
        "aliasing LrFrame/CdefFrame rejected");

  RestorationFrame nolr = view;
  nolr.lr[1].data = nullptr;
  CHECK(loop_restoration_frame(nolr, params) ==
            RestorationStatus::kInvalidArgument,
        "missing LrFrame rejected");

  LoopRestorationParams bad_coeff = params;
  bad_coeff.unit_coeffs[0] = nullptr;
  CHECK(loop_restoration_frame(view, bad_coeff) ==
            RestorationStatus::kInvalidArgument,
        "missing per-unit coefficients rejected");

  LoopRestorationParams bad_type = params;
  bad_type.frame_type[2] = static_cast<RestorationType>(9);
  CHECK(loop_restoration_frame(view, bad_type) ==
            RestorationStatus::kInvalidArgument,
        "out-of-range frame type rejected");

  RestorationFrame wide = view;
  wide.bit_depth = 10;
  CHECK(loop_restoration_frame(wide, params) ==
            RestorationStatus::kUnsupportedBitDepth,
        "10-bit rejected");
  std::cout << "  validation: GDF, PC-Wiener, class grid, aliasing\n";
}

void test_randomized_oracle() {
  int mismatches = 0;
  long samples = 0;
  for (int seed = 1; seed <= 30; ++seed) {
    Rng rng(static_cast<std::uint32_t>(seed * 7919));
    const int mi_cols = rng.range(6, 20);
    const int mi_rows = rng.range(4, 14);
    TestFrame f = make_frame(mi_cols, mi_rows);
    fill_random(f, rng, rng.range(0, 40), rng.range(200, 255));
    LoopRestorationParams params;
    LrTest tst;
    const int unit_size = rng.range(0, 1) ? 64 : 128;
    setup_params(f, unit_size, kRestoreSwitchable, &params, &tst);
    const bool frame_filters = rng.range(0, 3) == 0;
    for (int pl = 0; pl < 3; ++pl) {
      for (std::size_t i = 0; i < tst.type[pl].size(); ++i) {
        tst.type[pl][i] = static_cast<std::uint8_t>(
            rng.range(0, 3) == 0 ? kRestoreNone : kRestoreWienerNs);
      }
      for (std::size_t i = 0; i < tst.coeffs[pl].size(); ++i) {
        tst.coeffs[pl][i] = static_cast<std::int16_t>(rng.range(-24, 24));
      }
    }
    if (frame_filters) {
      params.frame_filters_on[0] = 1;
      params.num_filter_classes = 2;
      tst.subclass.assign(static_cast<std::size_t>(mi_cols * mi_rows), 0);
      for (std::size_t i = 0; i < tst.subclass.size(); ++i) {
        tst.subclass[i] = static_cast<std::uint8_t>(rng.range(0, 1));
      }
      params.subclass_grid = tst.subclass.data();
      for (int pl = 0; pl < 3; ++pl) {
        tst.frame_coeffs[pl].assign(2 * kWienerNsCoeffStride, 0);
        for (std::size_t i = 0; i < tst.frame_coeffs[pl].size(); ++i) {
          tst.frame_coeffs[pl][i] = static_cast<std::int16_t>(rng.range(-16, 16));
        }
        params.frame_coeffs[pl] = tst.frame_coeffs[pl].data();
      }
    }
    RestorationFrame view = f.view();
    view.cfl_ds_filter_index = rng.range(0, 3);
    NaiveLrFrame nv;
    nv.cfl_ds_filter_index = view.cfl_ds_filter_index;
    TestFrame want;
    run_naive(f, params, nv, &want);
    CHECK(loop_restoration_frame(view, params) == RestorationStatus::kOk,
          "random restoration ok");
    for (int p = 0; p < 3; ++p) {
      for (int y = 0; y < f.height[p]; ++y) {
        for (int x = 0; x < f.width[p]; ++x) {
          ++samples;
          if (f.at(f.lr[p], p, y, x) != want.at(want.lr[p], p, y, x)) {
            ++mismatches;
          }
        }
      }
    }
    CHECK(f.lr_guard_intact(), "random LrFrame guard samples untouched");
  }
  CHECK(mismatches == 0, "randomized loop restoration matches the oracle");
  std::cout << "  randomized: " << samples << " samples, " << mismatches
            << " mismatches\n";
}

}  // namespace

int main() {
  std::cout << "AV2 §7.20 loop restoration (Wiener NS) — CPU reference tests\n";
  test_identity_copy();
  test_per_unit_filters();
  test_chroma_from_luma_and_cfl();
  test_stripe_sourcing();
  test_frame_filters_and_subclass();
  test_class_grid_and_lookup();
  test_per_unit_types_and_partial();
  test_tile_clipping();
  test_lossless_and_bru();
  test_validation();
  test_randomized_oracle();
  std::cout << (g_failures == 0 ? "PASS" : "FAIL") << ": " << g_checks
            << " checks, " << g_failures << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
