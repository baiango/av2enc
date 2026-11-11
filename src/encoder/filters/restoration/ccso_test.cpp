// CPU reference tests for AV2 §7.19 CCSO.
//
// Coverage:
//   * band-offset-only and edge-classified modes, edge_clf 0/1/2
//   * all 7 extension filter positions
//   * every scale_idx/quant_idx pair, including quantStep == 0
//   * band shifts 1..64 bands, offsets scaled by scale_idx+1
//   * per-plane enable flags, disabled planes untouched
//   * CcsoBlks unit grid (enabled/disabled units)
//   * lossless segments
//   * BRU-inactive superblock regions
//   * tile clipping with disable_loopfilters_across_tiles
//   * partial units at the right/bottom plane edge, odd MI dimensions
//   * output clipping at 0 and 255
//   * randomized CPU-vs-naive-oracle comparison
//
// Usage: ./ccso_test

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

// One enabled plane with every class/band usable.
CcsoFrameParams base_params(int bands_log2 = 6, bool bo_only = false,
                            int edge_clf = 0, int ext = 0) {
  CcsoFrameParams params;
  params.ccso_luma_size_log2 = 6;
  CcsoParams& p = params.plane[0];
  p.enabled = true;
  p.bo_only = bo_only;
  p.scale_idx = 1;
  p.quant_idx = 2;
  p.ext_filter = ext;
  p.edge_clf = edge_clf;
  p.max_band_log2 = bands_log2;
  const int count = ccso_offset_count(p);
  std::vector<std::uint8_t> idx(static_cast<std::size_t>(count), 0);
  for (int i = 0; i < count; ++i) {
    idx[static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>((i * 5 + 3) & 7);
  }
  const bool ok = ccso_set_offsets(p, idx.data(), idx.size());
  CHECK(ok, "ccso_set_offsets must accept a valid table");
  return params;
}

// --- tests -----------------------------------------------------------------

void test_offset_helper() {
  CcsoParams p;
  p.enabled = true;
  p.scale_idx = 2;
  p.quant_idx = 1;
  p.ext_filter = 3;
  p.edge_clf = 1;
  p.max_band_log2 = 3;
  CHECK(ccso_offset_count(p) == 8 * 2 * 2, "edge_clf=1 halves each edge axis");
  std::vector<std::uint8_t> idx(
      static_cast<std::size_t>(ccso_offset_count(p)), 0);
  idx[0] = 1;                              // band 0, d0 0, d1 0
  idx[8 * 2 * 2 - 1] = 7;                  // last entry
  CHECK(ccso_set_offsets(p, idx.data(), idx.size()), "offsets accepted");
  CHECK(p.offsets[ccso_offset_index(0, 0, 0)] ==
            static_cast<std::int16_t>(1 * (2 + 1)),
        "offset = Ccso_Offset[idx] * (scale_idx + 1)");
  CHECK(p.offsets[ccso_offset_index(7, 1, 1)] ==
            static_cast<std::int16_t>(-10 * 3),
        "last entry maps to band 7, classes (1,1) with -10 offset");

  // bo_only: one class pair, interval forced to 1.
  p.bo_only = true;
  CHECK(ccso_offset_count(p) == 8, "bo_only keeps one (cls0,cls1) pair");
  std::vector<std::uint8_t> bo(8, 2);
  CHECK(ccso_set_offsets(p, bo.data(), bo.size()), "bo offsets accepted");
  CHECK(p.offsets[ccso_offset_index(3, 0, 0)] ==
            static_cast<std::int16_t>(-1 * 3),
        "bo_only writes only class (0,0)");

  // Rejecting bad counts / indices.
  CHECK(!ccso_set_offsets(p, bo.data(), 7), "short offset table rejected");
  bo[0] = 8;
  CHECK(!ccso_set_offsets(p, bo.data(), bo.size()),
        "offset index >= 8 rejected");
  std::cout << "  offset helper: order, scaling, bo_only, rejection\n";
}

void test_bo_only_band_offsets() {
  TestFrame f = make_frame(16, 8);  // 64x32 luma
  Rng rng(11);
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p]; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        // Bands by top bits of the luma value.
        f.at(f.curr[p], p, y, x) = static_cast<std::uint8_t>(rng.range(0, 255));
        f.at(f.cdef[p], p, y, x) = 200;
      }
    }
  }
  CcsoFrameParams params = base_params(2, /*bo_only=*/true, 0, 0);
  params.plane[0].scale_idx = 0;
  std::vector<std::uint8_t> blks(static_cast<std::size_t>(1), 1);
  params.blks[0] = blks.data();
  params.blk_cols[0] = 1;
  params.blk_rows[0] = 1;

  TestFrame naive = f;
  naive_ccso(naive, params);
  CHECK(ccso_frame(f.view(), params) == RestorationStatus::kOk,
        "ccso_frame ok");
  int changed = 0;
  for (int y = 0; y < f.height[0]; ++y) {
    for (int x = 0; x < f.width[0]; ++x) {
      CHECK(f.at(f.cdef[0], 0, y, x) == naive.at(naive.cdef[0], 0, y, x),
            "bo_only luma matches the naive oracle");
      if (f.at(f.cdef[0], 0, y, x) != 200) {
        ++changed;
      }
    }
  }
  CHECK(changed > 0, "band offsets must modify at least one sample");
  CHECK(f.guard_intact(), "guard samples untouched");
  std::cout << "  bo_only: band offsets match the oracle (" << changed
            << " samples)\n";
}

void test_edge_classification() {
  for (int edge_clf = 0; edge_clf <= 2; ++edge_clf) {
    for (int ext = 0; ext < 7; ++ext) {
      TestFrame f = make_frame(12, 7);
      fill_smooth(f);
      CcsoFrameParams params = base_params(5, false, edge_clf, ext);
      std::vector<std::uint8_t> blks(static_cast<std::size_t>(1), 1);
      params.blks[0] = blks.data();
      params.blk_cols[0] = 1;
      params.blk_rows[0] = 1;

      TestFrame naive = f;
      TestFrame before = f;
      naive_ccso(naive, params);
      CHECK(ccso_frame(f.view(), params) == RestorationStatus::kOk,
            "ccso_frame ok");
      bool same = true;
      int changed = 0;
      for (int y = 0; y < f.height[0] && same; ++y) {
        for (int x = 0; x < f.width[0]; ++x) {
          if (f.at(f.cdef[0], 0, y, x) != naive.at(naive.cdef[0], 0, y, x)) {
            same = false;
            break;
          }
          if (f.at(f.cdef[0], 0, y, x) != before.at(before.cdef[0], 0, y, x)) {
            ++changed;
          }
        }
      }
      CHECK(same, "edge-classified CCSO matches the naive oracle");
      if (edge_clf < 2) {
        CHECK(changed > 0, "edge-classified CCSO must change samples");
      }
    }
  }
  std::cout << "  edge classifier: classes 0/1/2 x 7 extension filters\n";
}

void test_all_scale_quant_pairs() {
  for (int scale = 0; scale < 4; ++scale) {
    for (int quant = 0; quant < 4; ++quant) {
      TestFrame f = make_frame(9, 5);
      Rng rng(static_cast<std::uint32_t>(scale * 4 + quant + 1));
      fill_random(f, rng, 0, 255);
      CcsoFrameParams params = base_params(4, false, 0, 4);
      params.plane[0].scale_idx = scale;
      params.plane[0].quant_idx = quant;
      std::vector<std::uint8_t> blks(static_cast<std::size_t>(1), 1);
      params.blks[0] = blks.data();
      params.blk_cols[0] = 1;
      params.blk_rows[0] = 1;
      TestFrame naive = f;
      naive_ccso(naive, params);
      CHECK(ccso_frame(f.view(), params) == RestorationStatus::kOk, "ok");
      bool same = true;
      for (int y = 0; y < f.height[0] && same; ++y) {
        for (int x = 0; x < f.width[0]; ++x) {
          if (f.at(f.cdef[0], 0, y, x) != naive.at(naive.cdef[0], 0, y, x)) {
            same = false;
            break;
          }
        }
      }
      CHECK(same, "every (scale_idx, quant_idx) pair matches the oracle");
    }
  }
  std::cout << "  quant table: 16 (scale, quant) pairs\n";
}

void test_planes_and_units() {
  TestFrame f = make_frame(20, 12);  // 80x48 luma; 64-unit -> 2x1 units
  Rng rng(99);
  fill_random(f, rng, 0, 255);
  CcsoFrameParams params;
  params.ccso_luma_size_log2 = 6;
  for (int p = 0; p < 3; ++p) {
    CcsoParams& cp = params.plane[p];
    cp.enabled = true;
    cp.scale_idx = 2;
    cp.quant_idx = 1;
    cp.ext_filter = 2;
    cp.edge_clf = 0;
    cp.max_band_log2 = 6;
    std::vector<std::uint8_t> idx(
        static_cast<std::size_t>(ccso_offset_count(cp)));
    for (std::size_t i = 0; i < idx.size(); ++i) {
      idx[i] = static_cast<std::uint8_t>((i * 3) & 7);
    }
    CHECK(ccso_set_offsets(cp, idx.data(), idx.size()), "offsets ok");
  }
  // 2 unit columns x 1 unit row for luma; chroma uses the same luma-unit grid.
  std::vector<std::uint8_t> blk0 = {1, 0};
  std::vector<std::uint8_t> blk1 = {0, 1};
  std::vector<std::uint8_t> blk2 = {1, 1};
  params.blks[0] = blk0.data();
  params.blks[1] = blk1.data();
  params.blks[2] = blk2.data();
  for (int p = 0; p < 3; ++p) {
    params.blk_cols[p] = 2;
    params.blk_rows[p] = 1;
  }

  TestFrame naive = f;
  TestFrame before = f;
  naive_ccso(naive, params);
  CHECK(ccso_frame(f.view(), params) == RestorationStatus::kOk, "ok");
  bool same = true;
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p] && same; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        if (f.at(f.cdef[p], p, y, x) != naive.at(naive.cdef[p], p, y, x)) {
          same = false;
          break;
        }
      }
    }
  }
  CHECK(same, "per-plane unit grids match the oracle");
  // Plane 0 unit 1 is disabled: the right 16 samples must be untouched.
  bool unit0_unchanged = true;
  for (int y = 0; y < f.height[0]; ++y) {
    for (int x = 64; x < f.width[0]; ++x) {
      if (f.at(f.cdef[0], 0, y, x) != before.at(before.cdef[0], 0, y, x)) {
        unit0_unchanged = false;
      }
    }
  }
  CHECK(unit0_unchanged, "disabled CCSO unit must not change samples");
  CHECK(f.guard_intact(), "guard samples untouched");
  std::cout << "  per-plane CcsoBlks: 3 planes, 2 units each\n";
}

void test_lossless_and_bru() {
  TestFrame f = make_frame(16, 8);
  Rng rng(5);
  fill_random(f, rng, 0, 255);
  std::vector<std::uint8_t> seg(static_cast<std::size_t>(f.mi_cols * f.mi_rows), 0);
  // Right half of the frame: segment 1 (lossless).
  for (int y = 0; y < f.mi_rows; ++y) {
    for (int x = 0; x < f.mi_cols; ++x) {
      seg[static_cast<std::size_t>(y * f.mi_cols + x)] =
          static_cast<std::uint8_t>(x >= f.mi_cols / 2 ? 1 : 0);
    }
  }
  std::uint8_t lossless[8] = {0, 1, 0, 0, 0, 0, 0, 0};
  CcsoFrameParams params = base_params(6, false, 0, 1);
  std::vector<std::uint8_t> blks(static_cast<std::size_t>(1), 1);
  params.blks[0] = blks.data();
  params.blk_cols[0] = 1;
  params.blk_rows[0] = 1;

  RestorationFrame view = f.view();
  view.segment_ids = seg.data();
  view.chroma_segment_ids = seg.data();
  view.lossless = lossless;

  TestFrame before = f;
  TestFrame naive = f;
  naive_ccso(naive, params, seg.data(), seg.data(), lossless);
  CHECK(ccso_frame(view, params) == RestorationStatus::kOk, "ok");
  bool same = true;
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p] && same; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        if (f.at(f.cdef[p], p, y, x) != naive.at(naive.cdef[p], p, y, x)) {
          same = false;
          break;
        }
      }
    }
  }
  CHECK(same, "lossless segments skip filtering exactly like the oracle");
  bool lossless_unchanged = true;
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p]; ++y) {
      for (int x = f.width[p] / 2; x < f.width[p]; ++x) {
        if (f.at(f.cdef[p], p, y, x) != before.at(before.cdef[p], p, y, x)) {
          lossless_unchanged = false;
        }
      }
    }
  }
  CHECK(lossless_unchanged, "lossless segment samples untouched");

  // BRU: left half inactive.
  std::vector<std::uint8_t> bru(static_cast<std::size_t>(f.mi_cols * f.mi_rows),
                                kBruActive);
  for (int y = 0; y < f.mi_rows; ++y) {
    for (int x = 0; x < f.mi_cols / 2; ++x) {
      bru[static_cast<std::size_t>(y * f.mi_cols + x)] = kBruInactive;
    }
  }
  TestFrame f2 = make_frame(16, 8);
  Rng rng2(6);
  fill_random(f2, rng2, 0, 255);
  TestFrame before2 = f2;
  TestFrame naive2 = f2;
  naive_ccso(naive2, params, nullptr, nullptr, nullptr, bru.data());
  RestorationFrame view2 = f2.view();
  view2.bru_modes = bru.data();
  CHECK(ccso_frame(view2, params) == RestorationStatus::kOk, "ok");
  bool same2 = true;
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f2.height[p] && same2; ++y) {
      for (int x = 0; x < f2.width[p]; ++x) {
        if (f2.at(f2.cdef[p], p, y, x) !=
            naive2.at(naive2.cdef[p], p, y, x)) {
          same2 = false;
          break;
        }
      }
    }
  }
  CHECK(same2, "BRU-inactive regions skip filtering");
  bool bru_unchanged = true;
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f2.height[p]; ++y) {
      for (int x = 0; x < f2.width[p] / 2; ++x) {
        if (f2.at(f2.cdef[p], p, y, x) !=
            before2.at(before2.cdef[p], p, y, x)) {
          bru_unchanged = false;
        }
      }
    }
  }
  CHECK(bru_unchanged, "BRU-inactive region samples untouched");
  std::cout << "  lossless segments and BRU-inactive regions: skipped\n";
}

void test_tile_clipping() {
  // 128x64 luma, two tiles 16 MI wide.  The left tile's CCSO reads must clamp
  // to columns 0..63 of CurrFrame even when the tap reaches right.
  TestFrame f = make_frame(32, 16);
  Rng rng(17);
  fill_random(f, rng, 0, 255);
  CcsoFrameParams params = base_params(6, false, 0, 6);  // ext 6 => dx=+2
  std::vector<std::uint8_t> blks(static_cast<std::size_t>(2), 1);
  params.blks[0] = blks.data();
  params.blk_cols[0] = 2;
  params.blk_rows[0] = 1;

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
      mrs[i] = 0;
      mre[i] = 16;
    }
  }
  RestorationFrame view = f.view();
  view.mi_col_start = mcs.data();
  view.mi_col_end = mce.data();
  view.mi_row_start = mrs.data();
  view.mi_row_end = mre.data();
  view.disable_loopfilters_across_tiles = 1;

  TestFrame naive = f;
  naive_ccso(naive, params, nullptr, nullptr, nullptr, nullptr, 1, mcs.data(),
             mce.data(), mrs.data(), mre.data());
  CHECK(ccso_frame(view, params) == RestorationStatus::kOk, "ok");
  bool same = true;
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p] && same; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        if (f.at(f.cdef[p], p, y, x) != naive.at(naive.cdef[p], p, y, x)) {
          same = false;
          break;
        }
      }
    }
  }
  CHECK(same, "tile-clipped luma taps match the oracle");
  std::cout << "  tile clipping: disable_loopfilters_across_tiles\n";
}

void test_partial_units_and_clipping() {
  // 68x20 luma: CCSO unit 64 -> second unit column is a 4-sample sliver.
  TestFrame f = make_frame(17, 5);
  Rng rng(21);
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p]; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        f.at(f.curr[p], p, y, x) = 255;  // force positive clipping
        f.at(f.cdef[p], p, y, x) = 255;
      }
    }
  }
  CcsoFrameParams params;
  params.ccso_luma_size_log2 = 6;
  CcsoParams& cp = params.plane[0];
  cp.enabled = true;
  cp.scale_idx = 3;
  cp.quant_idx = 3;
  cp.ext_filter = 0;
  cp.max_band_log2 = 6;
  std::vector<std::uint8_t> idx(
      static_cast<std::size_t>(ccso_offset_count(cp)), 5);  // +7 each
  CHECK(ccso_set_offsets(cp, idx.data(), idx.size()), "offsets ok");
  std::vector<std::uint8_t> blks(2, 1);
  params.blks[0] = blks.data();
  params.blk_cols[0] = 2;
  params.blk_rows[0] = 1;

  CHECK(ccso_frame(f.view(), params) == RestorationStatus::kOk, "ok");
  bool clipped = true;
  for (int y = 0; y < f.height[0]; ++y) {
    for (int x = 0; x < f.width[0]; ++x) {
      if (f.at(f.cdef[0], 0, y, x) != 255) {
        clipped = false;
      }
    }
  }
  CHECK(clipped, "positive offsets clip at 255");

  // Negative clipping: all-zero samples with a negative offset.
  TestFrame g = make_frame(17, 5);
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < g.height[p]; ++y) {
      for (int x = 0; x < g.width[p]; ++x) {
        g.at(g.curr[p], p, y, x) = 0;
        g.at(g.cdef[p], p, y, x) = 0;
      }
    }
  }
  CcsoFrameParams gp;
  gp.ccso_luma_size_log2 = 6;
  gp.plane[0].enabled = true;
  gp.plane[0].scale_idx = 3;
  gp.plane[0].quant_idx = 0;
  gp.plane[0].edge_clf = 0;
  gp.plane[0].max_band_log2 = 6;
  std::vector<std::uint8_t> nidx(
      static_cast<std::size_t>(ccso_offset_count(gp.plane[0])), 7);
  CHECK(ccso_set_offsets(gp.plane[0], nidx.data(), nidx.size()), "neg ok");
  std::vector<std::uint8_t> gblk = {1, 1};
  gp.blks[0] = gblk.data();
  gp.blk_cols[0] = 2;
  gp.blk_rows[0] = 1;
  CHECK(ccso_frame(g.view(), gp) == RestorationStatus::kOk, "ok");
  bool neg_clipped = true;
  for (int y = 0; y < g.height[0]; ++y) {
    for (int x = 0; x < g.width[0]; ++x) {
      if (g.at(g.cdef[0], 0, y, x) != 0) {
        neg_clipped = false;
      }
    }
  }
  CHECK(neg_clipped, "negative offsets clip at 0");
  CHECK(g.guard_intact(), "guard samples untouched");
  std::cout << "  partial units (68x20) and 0/255 clipping\n";
}

void test_unit_selection_and_small_unit() {
  // §5.18.7.12 CcsoLumaSizeLog2 derivation.
  CHECK(ccso_select_luma_size_log2(64, 0, true) == 6,
        "64 SB with unit==SB selects 64x64 CCSO units");
  CHECK(ccso_select_luma_size_log2(128, 0, true) == 7,
        "128 SB with unit==SB selects 128x128 CCSO units");
  CHECK(ccso_select_luma_size_log2(256, 0, true) == 8,
        "256 SB with unit==SB selects 256x256 CCSO units");
  CHECK(ccso_select_luma_size_log2(64, 0, false) == 8,
        "64-aligned tile starts permit 256x256 CCSO units");
  CHECK(ccso_select_luma_size_log2(64, 32, false) == 7,
        "32-aligned tile starts select 128x128 CCSO units");
  CHECK(ccso_select_luma_size_log2(64, 16, false) == 6,
        "otherwise CCSO units are 64x64");
  int cols = 0;
  int rows = 0;
  ccso_blk_grid_size(640, 360, 6, 0, 0, &cols, &rows);
  CHECK(cols == 10 && rows == 6, "640x360 luma needs a 10x6 CcsoBlks grid");
  ccso_blk_grid_size(320, 180, 6, 1, 1, &cols, &rows);
  CHECK(cols == 10 && rows == 6, "4:2:0 chroma keeps the same grid");

  // §5.18.7.12 f(2 + bo_only) and f(1) limits.
  {
    CcsoParams normal = base_params(3, false, 0, 0).plane[0];
    CHECK(ccso_header_conforms(normal), "3-band non-bo_only header-conforming");
    normal.max_band_log2 = 4;
    CHECK(!ccso_header_conforms(normal),
          "4-band non-bo_only is not signalable by f(2)");
    CcsoParams bo = base_params(6, true, 0, 0).plane[0];
    CHECK(ccso_header_conforms(bo), "6-band bo_only header-conforming");
    CcsoParams edge = base_params(3, false, 2, 0).plane[0];
    CHECK(!ccso_header_conforms(edge), "edge_clf=2 is not signalable by f(1)");
    CcsoParams zero = base_params(3, false, 1, 0).plane[0];
    zero.scale_idx = 0;
    zero.quant_idx = 3;  // CCSO_Quant_Sz[0][3] == 0
    CHECK(!ccso_header_conforms(zero),
          "quantStep 0 forces edge_clf 0 in the header");
  }

  // A 64x64 unit inside a 128x128 superblock: the §7.19.1 inner loops step by
  // sbBlkH/sbBlkW, so every sample is still visited exactly once.
  TestFrame f = make_frame(16, 8);
  Rng rng(12);
  fill_random(f, rng);
  CcsoFrameParams params = base_params(4, /*bo_only=*/false, 1, 3);
  params.ccso_luma_size_log2 = 6;
  for (int p = 1; p < 3; ++p) {
    params.plane[p] = params.plane[0];
    params.plane[p].enabled = true;
  }
  std::vector<std::uint8_t> blks(1, 1);
  for (int p = 0; p < 3; ++p) {
    params.blks[p] = blks.data();
    params.blk_cols[p] = 1;
    params.blk_rows[p] = 1;
  }
  TestFrame naive = f;
  naive_ccso(naive, params, nullptr, nullptr, nullptr, nullptr, 0, nullptr,
             nullptr, nullptr, nullptr, /*sb_width=*/128, /*sb_height=*/128);
  RestorationFrame view = f.view();
  view.sb_width = 128;
  view.sb_height = 128;
  CHECK(ccso_frame(view, params) == RestorationStatus::kOk,
        "CCSO with unit < superblock accepted");
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p]; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        CHECK(f.at(f.cdef[p], p, y, x) == naive.at(naive.cdef[p], p, y, x),
              "unit < SB matches the naive oracle");
      }
    }
  }
  CHECK(f.guard_intact(), "unit < SB leaves guard samples untouched");
  std::cout << "  unit selection: header derivation, unit < superblock\n";
}


void test_validation() {
  TestFrame f = make_frame(16, 8);
  CcsoFrameParams params = base_params();
  std::vector<std::uint8_t> blks(1, 1);
  params.blks[0] = blks.data();
  params.blk_cols[0] = 1;
  params.blk_rows[0] = 1;
  RestorationFrame view = f.view();
  CHECK(ccso_frame(view, params) == RestorationStatus::kOk, "baseline ok");

  RestorationFrame bad = view;
  bad.bit_depth = 10;
  CHECK(ccso_frame(bad, params) == RestorationStatus::kUnsupportedBitDepth,
        "10-bit rejected");
  bad = view;
  bad.subsampling_y = 0;
  CHECK(ccso_frame(bad, params) == RestorationStatus::kUnsupportedSubsampling,
        "4:2:2 rejected");
  bad = view;
  bad.curr[0].data = nullptr;
  CHECK(ccso_frame(bad, params) == RestorationStatus::kInvalidArgument,
        "missing CurrFrame rejected");
  bad = view;
  bad.mi_cols = 0;
  CHECK(ccso_frame(bad, params) == RestorationStatus::kInvalidArgument,
        "empty frame rejected");

  CcsoFrameParams p2 = params;
  p2.plane[0].max_band_log2 = 7;
  CHECK(ccso_frame(view, p2) == RestorationStatus::kInvalidArgument,
        "7-band log2 rejected");
  p2 = params;
  p2.plane[0].ext_filter = 7;
  CHECK(ccso_frame(view, p2) == RestorationStatus::kInvalidArgument,
        "ccso_ext_filter == 7 is non-conforming");
  p2 = params;
  p2.blks[0] = nullptr;
  CHECK(ccso_frame(view, p2) == RestorationStatus::kInvalidArgument,
        "missing CcsoBlks rejected");
  p2 = params;
  p2.blk_cols[0] = 0;
  CHECK(ccso_frame(view, p2) == RestorationStatus::kInvalidArgument,
        "undersized CcsoBlks grid rejected (would skip samples)");
  p2 = params;
  p2.blk_rows[0] = 0;
  CHECK(ccso_frame(view, p2) == RestorationStatus::kInvalidArgument,
        "undersized CcsoBlks rows rejected");
  p2 = params;
  p2.ccso_luma_size_log2 = 3;
  CHECK(ccso_frame(view, p2) == RestorationStatus::kInvalidArgument,
        "bad CCSO unit log2 rejected");
  p2 = params;
  p2.ccso_luma_size_log2 = 5;  // not a legal CcsoLumaSizeLog2 value
  CHECK(ccso_frame(view, p2) == RestorationStatus::kInvalidArgument,
        "log2=5 rejected as non-conforming");
  // A 64x64 CCSO unit inside a 128x128 superblock is conforming: §7.19.1's
  // inner loops step by sbBlkH/sbBlkW and run once per unit.  It is exercised
  // in test_ccso_unit_smaller_than_sb.
  p2 = params;
  p2.ccso_luma_size_log2 = 6;
  CHECK(ccso_frame(view, p2) == RestorationStatus::kOk,
        "64x64 CCSO unit accepted");
  CHECK(std::strcmp(restoration_status_string(RestorationStatus::kOk), "ok") ==
            0,
        "status string ok");
  std::cout << "  validation: bit depth, subsampling, geometry, ranges\n";
}

void test_randomized_oracle() {
  int cases = 0;
  int mismatches = 0;
  for (int seed = 1; seed <= 40; ++seed) {
    Rng rng(static_cast<std::uint32_t>(seed * 977));
    const int mi_cols = rng.range(5, 24);
    const int mi_rows = rng.range(3, 12);
    TestFrame f = make_frame(mi_cols, mi_rows);
    fill_random(f, rng, 0, 255);
    CcsoFrameParams params;
    params.ccso_luma_size_log2 = rng.range(6, 8);
    const int shift_y_luma = params.ccso_luma_size_log2;
    const int shift_x_luma = params.ccso_luma_size_log2;
    std::vector<std::uint8_t> grids[3];
    for (int p = 0; p < 3; ++p) {
      CcsoParams& cp = params.plane[p];
      cp.enabled = rng.range(0, 1) != 0;
      cp.bo_only = rng.range(0, 3) == 0;
      cp.scale_idx = rng.range(0, 3);
      cp.quant_idx = rng.range(0, 3);
      cp.ext_filter = rng.range(0, 6);
      cp.edge_clf = rng.range(0, 2);
      cp.max_band_log2 = rng.range(1, 6);
      const int count = ccso_offset_count(cp);
      std::vector<std::uint8_t> idx(static_cast<std::size_t>(count));
      for (int i = 0; i < count; ++i) {
        idx[static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>(rng.range(0, 7));
      }
      CHECK(ccso_set_offsets(cp, idx.data(), idx.size()), "random offsets ok");
      const int sub = (p == 0) ? 0 : 1;
      const int shift_x = shift_x_luma - sub;
      const int shift_y = shift_y_luma - sub;
      const int cols = (f.width[p] + (1 << shift_x) - 1) >> shift_x;
      const int rows = (f.height[p] + (1 << shift_y) - 1) >> shift_y;
      grids[p].assign(static_cast<std::size_t>(cols * rows), 0);
      for (std::size_t i = 0; i < grids[p].size(); ++i) {
        grids[p][i] = static_cast<std::uint8_t>(rng.range(0, 3) != 0 ? 1 : 0);
      }
      params.blks[p] = grids[p].data();
      params.blk_cols[p] = cols;
      params.blk_rows[p] = rows;
    }
    TestFrame naive = f;
    naive_ccso(naive, params);
    CHECK(ccso_frame(f.view(), params) == RestorationStatus::kOk,
          "random ccso ok");
    for (int p = 0; p < 3; ++p) {
      for (int y = 0; y < f.height[p]; ++y) {
        for (int x = 0; x < f.width[p]; ++x) {
          ++cases;
          if (f.at(f.cdef[p], p, y, x) != naive.at(naive.cdef[p], p, y, x)) {
            ++mismatches;
          }
        }
      }
    }
    CHECK(f.guard_intact(), "random guard samples untouched");
  }
  CHECK(mismatches == 0, "randomized CCSO matches the naive oracle");
  std::cout << "  randomized: " << cases << " samples, " << mismatches
            << " mismatches\n";
}

}  // namespace

int main() {
  std::cout << "AV2 §7.19 CCSO — CPU reference tests\n";
  test_offset_helper();
  test_bo_only_band_offsets();
  test_edge_classification();
  test_all_scale_quant_pairs();
  test_planes_and_units();
  test_lossless_and_bru();
  test_tile_clipping();
  test_partial_units_and_clipping();
  test_unit_selection_and_small_unit();
  test_validation();
  test_randomized_oracle();
  std::cout << (g_failures == 0 ? "PASS" : "FAIL") << ": " << g_checks
            << " checks, " << g_failures << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
