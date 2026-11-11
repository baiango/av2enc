// CPU reference tests for §7.20.5 Apply GDF filter process.
//
// The oracle below is a second, independent transcription of the pseudocode.
// It shares only the normative generated tables with gdf.cpp.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "encoder/filters/spec_tables/gdf.h"
#include "encoder/filters/spec_tables/gdf_tables.h"
#include "encoder/filters/spec_tables/spec_test_common.h"

namespace {

using namespace av2enc::filters::spec;
using namespace av2enc::filters::spec::test;
using namespace av2enc::filters::spec_tables;

struct Extents {
  int start_x, end_x, start_y, end_y, stripe_start_y, stripe_end_y;
};

int OracleAbs(int v) { return v < 0 ? -v : v; }
int OracleClip3(int lo, int hi, int v) { return v < lo ? lo : (v > hi ? hi : v); }
std::int64_t OracleRound2(std::int64_t x, int n) {
  return n == 0 ? x : (x + (std::int64_t{1} << (n - 1))) >> n;
}
std::int64_t OracleRound2Signed(std::int64_t x, int n) {
  return x >= 0 ? OracleRound2(x, n) : -OracleRound2(-x, n);
}

int OracleGradSum(const std::vector<int>& g, int width, int i, int j, int down,
                  int across) {
  int t = 0;
  for (int i2 = 0; i2 < down; ++i2) {
    for (int j2 = 0; j2 < across; ++j2) t += g[(i + i2) * width + (j + j2)];
  }
  return t;
}

// Independent §7.20.5 apply_gdf_filter.
void OracleGdf(const Image& curr, const Image& cdef, const Extents& e,
               std::uint16_t* lr, int lr_stride, int x, int y, int qp_idx,
               int ref_dst_idx, int w, int h, int gdf_pix_scale, int bd) {
  std::vector<int> grad[4];
  for (int d = 0; d < 4; ++d) grad[d].resize((w + 2) * (h + 2));

  for (int i = 0; i < h + 2; ++i) {
    for (int j = 0; j < w + 2; ++j) {
      for (int d = 0; d < 4; ++d) {
        int dx = 0, dy = 0;
        if (d == 0) {
          dy = 1;
        } else if (d == 1) {
          dx = 1;
        } else if (d == 2) {
          dx = 1;
          dy = 1;
        } else {
          dx = 1;
          dy = -1;
        }
        const int a = OracleSample(curr, cdef, e.start_x, e.end_x, e.start_y,
                                   e.end_y, e.stripe_start_y, e.stripe_end_y,
                                   x - 1 + j - dx, y - 1 + i - dy);
        const int b = OracleSample(curr, cdef, e.start_x, e.end_x, e.start_y,
                                   e.end_y, e.stripe_start_y, e.stripe_end_y,
                                   x - 1 + j, y - 1 + i);
        const int c = OracleSample(curr, cdef, e.start_x, e.end_x, e.start_y,
                                   e.end_y, e.stripe_start_y, e.stripe_end_y,
                                   x - 1 + j + dx, y - 1 + i + dy);
        grad[d][i * (w + 2) + j] = OracleAbs(b * 2 - a - c);
      }
    }
  }

  std::vector<int> cls_map((w / 2) * (h / 2));
  for (int i = (h / 2) - 1; i >= 0; --i) {
    for (int j = 0; j < w / 2; ++j) {
      int str[4];
      for (int d = 0; d < 4; ++d) {
        str[d] = OracleGradSum(grad[d], w + 2, i * 2, j * 2, 4, 4);
      }
      int cls = str[0] > str[1] ? 0 : 1;
      cls |= str[2] > str[3] ? 0 : 2;
      cls_map[i * (w / 2) + j] = cls;
    }
  }

  const int scale = ref_dst_idx == 0 ? 8 : 5;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int y2 = i + y;
      const int x2 = x + j;
      const int cls = cls_map[(i / 2) * (w / 2) + (j / 2)];
      std::int64_t gdf_idx[3] = {0, 0, 0};
      for (int k = 0; k < 22; ++k) {
        const int alpha = kGdfAlpha[ref_dst_idx][qp_idx][k][cls];
        int comb;
        if (k < 18) {
          const int dy = kGdfCoords[k].dy;
          const int dx = kGdfCoords[k].dx;
          const int sample2 = OracleSample(
              curr, cdef, e.start_x, e.end_x, e.start_y, e.end_y,
              e.stripe_start_y, e.stripe_end_y, x2, y2);
          const int sample3 = OracleSample(
              curr, cdef, e.start_x, e.end_x, e.start_y, e.end_y,
              e.stripe_start_y, e.stripe_end_y, x2 - dx, y2 - dy);
          const int sample4 = OracleSample(
              curr, cdef, e.start_x, e.end_x, e.start_y, e.end_y,
              e.stripe_start_y, e.stripe_end_y, x2 + dx, y2 + dy);
          const int shift = 10 - (bd < 10 ? bd : 10);
          const int above = OracleClip3(-alpha, alpha, (sample3 - sample2) << shift);
          const int below = OracleClip3(-alpha, alpha, (sample4 - sample2) << shift);
          comb = OracleClip3(-512, 511, above + below);
        } else {
          const int d = k - 18;
          int v = OracleGradSum(grad[d], w + 2, (i / 2) * 2, (j / 2) * 2, 4, 4);
          v = bd == 8 ? v >> 2 : v >> 4;
          comb = v < alpha ? v : alpha;
        }
        for (int idx = 0; idx < 3; ++idx) {
          gdf_idx[idx] +=
              static_cast<std::int64_t>(comb) * kGdfWeight[ref_dst_idx][qp_idx][idx][k][cls];
        }
      }
      int pos = 0;
      for (int idx = 0; idx < 3; ++idx) {
        const std::int64_t v = OracleRound2Signed(
            (gdf_idx[idx] + kGdfBias[ref_dst_idx][qp_idx][idx]) * scale, 15);
        const int clipped = OracleClip3(-scale, scale - 1, static_cast<int>(v));
        pos = pos * scale * 2 + clipped + scale;
      }
      const int err = ref_dst_idx == 0 ? kGdfIntraError[qp_idx][pos]
                                       : kGdfInterError[ref_dst_idx - 1][qp_idx][pos];
      const int delta =
          static_cast<int>(OracleRound2Signed(static_cast<std::int64_t>(err) * gdf_pix_scale,
                                              12 - bd));
      const int base = lr[y2 * lr_stride + x2];
      const int maxv = (1 << bd) - 1;
      const int result = base + delta;
      lr[y2 * lr_stride + x2] =
          static_cast<std::uint16_t>(result < 0 ? 0 : (result > maxv ? maxv : result));
    }
  }
}

LumaSource MakeSource(const Image& curr, const Image& cdef, const Extents& e) {
  LumaSource src;
  src.curr = curr.px.data();
  src.cdef = cdef.px.data();
  src.curr_stride = curr.stride;
  src.cdef_stride = cdef.stride;
  src.luma_start_x = e.start_x;
  src.luma_end_x = e.end_x;
  src.luma_start_y = e.start_y;
  src.luma_end_y = e.end_y;
  src.luma_stripe_start_y = e.stripe_start_y;
  src.luma_stripe_end_y = e.stripe_end_y;
  return src;
}

void TestTableIndices() {
  GdfParams p;
  p.frame_is_intra = true;
  p.base_q_idx = 0;
  p.bit_depth = 8;
  p.gdf_pic_qc_idx = 0;
  auto idx = GdfBlockTableIndices(p, nullptr, nullptr, 0, 0);
  CHECK(idx.ref_dst_idx == 0);
  // qpDiff = 0 - 85 = -85; (-85 - 37) / 25 = -122 / 25 = -4 (toward zero).
  // Clip3(0, 2, -4) = 0.
  CHECK(idx.qp_idx == 0);

  p.base_q_idx = 255;
  idx = GdfBlockTableIndices(p, nullptr, nullptr, 0, 0);
  // 255 - 85 = 170; (170 - 37) / 25 = 5 -> Clip3 -> 2.
  CHECK(idx.qp_idx == 2);

  p.gdf_pic_qc_idx = 3;
  idx = GdfBlockTableIndices(p, nullptr, nullptr, 0, 0);
  CHECK(idx.qp_idx == 5);

  p.frame_is_intra = false;
  p.base_q_idx = 200;
  p.gdf_pic_qc_idx = 0;
  // 200 - 110 = 90; (90 - 37) / 25 = 2.
  idx = GdfBlockTableIndices(p, nullptr, nullptr, 0, 0);
  CHECK(idx.qp_idx == 2);

  const int hints[2] = {7, 9};
  const int dist0[2] = {0, 0};
  const int restricted = 42;
  idx = GdfBlockTableIndices(p, hints, dist0, 2, restricted);
  CHECK(idx.ref_dst_idx == 5);  // maxDist == 0

  const int dist1[2] = {1, 0};
  idx = GdfBlockTableIndices(p, hints, dist1, 2, restricted);
  CHECK(idx.ref_dst_idx == 1);

  const int dist2[2] = {2, 0};
  idx = GdfBlockTableIndices(p, hints, dist2, 2, restricted);
  CHECK(idx.ref_dst_idx == 2);

  const int dist3[2] = {5, 0};
  idx = GdfBlockTableIndices(p, hints, dist3, 2, restricted);
  CHECK(idx.ref_dst_idx == 3);

  const int dist4[2] = {10, 0};
  idx = GdfBlockTableIndices(p, hints, dist4, 2, restricted);
  CHECK(idx.ref_dst_idx == 4);

  const int dist5[2] = {11, 0};
  idx = GdfBlockTableIndices(p, hints, dist5, 2, restricted);
  CHECK(idx.ref_dst_idx == 5);

  // A restricted reference is ignored when picking the maximum distance.
  const int hints_restricted[2] = {restricted, 3};
  const int dist_restricted[2] = {100, 2};
  idx = GdfBlockTableIndices(p, hints_restricted, dist_restricted, 2, restricted);
  CHECK(idx.ref_dst_idx == 2);
}

void TestSampleClamping() {
  Image curr(16, 16);
  Image cdef(16, 16);
  curr.Fill(200);
  cdef.Fill(100);
  Extents e{0, 15, 0, 15, 4, 11};

  CHECK(luma_source_sample(MakeSource(curr, cdef, e), 5, 7) == 100);
  CHECK(luma_source_sample(MakeSource(curr, cdef, e), 5, 3) == 200);
  CHECK(luma_source_sample(MakeSource(curr, cdef, e), 5, 1) == 200);
  CHECK(luma_source_sample(MakeSource(curr, cdef, e), 5, 0) == 200);
  CHECK(luma_source_sample(MakeSource(curr, cdef, e), 5, 12) == 200);
  CHECK(luma_source_sample(MakeSource(curr, cdef, e), 5, 15) == 200);

  // The x/y clamps keep reads inside the allowed rectangle.
  CHECK(luma_source_sample(MakeSource(curr, cdef, e), -4, 7) == 100);
  CHECK(luma_source_sample(MakeSource(curr, cdef, e), 99, 7) == 100);
}

void TestAgainstOracle() {
  TestRng rng(0x5eed1234);
  int comparisons = 0;

  for (int bd : {8, 10, 12}) {
    const int maxv = (1 << bd) - 1;
    for (int ref = 0; ref < 6; ++ref) {
      for (int qp = 0; qp < 6; ++qp) {
        for (int pix_scale = 1; pix_scale <= 4; ++pix_scale) {
          Image curr(24, 24);
          Image cdef(24, 24);
          Image lr(24, 24);
          for (int y = 0; y < 24; ++y) {
            for (int x = 0; x < 24; ++x) {
              curr.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
              cdef.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
              lr.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
            }
          }
          const int x = 8;
          const int w = 4;
          const int h = 4;
          const Extents e{0, 23, 0, 23, 4, 19};
          for (int y : {2, 8, 18}) {
            Image impl = lr;
            Image oracle = lr;
            ApplyGdfFilter(MakeSource(curr, cdef, e), impl.px.data(), impl.stride,
                           x, y, qp, ref, w, h, e.stripe_end_y + 1, pix_scale, bd);
            OracleGdf(curr, cdef, e, oracle.px.data(), oracle.stride, x, y, qp,
                      ref, w, h, pix_scale, bd);

            for (int yy = 0; yy < 24; ++yy) {
              for (int xx = 0; xx < 24; ++xx) {
                if (impl.At(xx, yy) != oracle.At(xx, yy)) {
                  std::fprintf(
                      stderr,
                      "gdf mismatch bd=%d ref=%d qp=%d scale=%d y0=%d at %d,%d: %d != %d\n",
                      bd, ref, qp, pix_scale, y, xx, yy, impl.At(xx, yy),
                      oracle.At(xx, yy));
                  CHECK(false);
                  return;
                }
                ++comparisons;
              }
            }
          }
        }
      }
    }
  }
  CHECK(comparisons > 100000);
}

void TestSplitEquivalence() {
  TestRng rng(0xabcdef01);
  for (int bd : {8, 10}) {
    const int maxv = (1 << bd) - 1;
    Image curr(32, 32);
    Image cdef(32, 32);
    Image base(32, 32);
    for (int y = 0; y < 32; ++y) {
      for (int x = 0; x < 32; ++x) {
        curr.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
        cdef.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
        base.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
      }
    }
    Extents e{0, 31, 0, 31, 0, 31};
    const auto src = MakeSource(curr, cdef, e);

    for (int ref : {0, 2, 5}) {
      for (int qp : {0, 4}) {
        Image whole = base;
        Image split = base;
        ApplyGdfFilter(src, whole.px.data(), whole.stride, 8, 8, qp, ref, 8, 8,
                       32, 2, bd);
        for (int dy = 0; dy < 8; dy += 4) {
          for (int dx = 0; dx < 8; dx += 4) {
            ApplyGdfFilter(src, split.px.data(), split.stride, 8 + dx, 8 + dy,
                           qp, ref, 4, 4, 32, 2, bd);
          }
        }
        CHECK(whole.px == split.px);
      }
    }
  }
}

void TestOutputRange() {
  TestRng rng(0x13579bdf);
  for (int bd : {8, 10}) {
    const int maxv = (1 << bd) - 1;
    Image curr(16, 16);
    Image cdef(16, 16);
    Image lr(16, 16);
    for (int i = 0; i < 16 * 16; ++i) {
      curr.px[i] = static_cast<std::uint16_t>(rng.Range(0, maxv));
      cdef.px[i] = static_cast<std::uint16_t>(rng.Range(0, maxv));
      lr.px[i] = static_cast<std::uint16_t>(rng.Range(0, maxv));
    }
    Extents e{0, 15, 0, 15, 0, 15};
    const auto src = MakeSource(curr, cdef, e);
    for (int ref = 0; ref < 6; ++ref) {
      for (int qp = 0; qp < 6; ++qp) {
        for (int scale = 1; scale <= 4; ++scale) {
          Image out = lr;
          ApplyGdfFilter(src, out.px.data(), out.stride, 4, 4, qp, ref, 8, 8, 16,
                         scale, bd);
          bool ok = true;
          for (int v : out.px) ok = ok && v <= maxv;
          CHECK(ok);
        }
      }
    }
  }
}

}  // namespace

int main() {
  TestTableIndices();
  TestSampleClamping();
  TestAgainstOracle();
  TestSplitEquivalence();
  TestOutputRange();
  return Finish("gdf_test");
}
