// Host-emulation parity test for the shared AV2 GDF device core.
//
// The same gdf_device.cuh that the CUDA kernel includes is compiled here by
// the host compiler and compared, sample for sample, with the CPU reference in
// src/encoder/filters/spec_tables/gdf.cpp.  No GPU, no nvcc and no CUDA
// runtime are needed.
//
// Coverage: §7.20.1 derivation (exhaustive parameter grid plus random refs),
// §7.20.2 sample sourcing, all 6x6 refDstIdx x qpIdx combinations, all four
// direction classes, bit depths 8/10/12, GdfPixScale 1..4, unit sizes 2..128
// (unit-sized and SB-sized), stripe/plane edge positions and random blocks.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/gdf/gdf_device.cuh"
#include "encoder/filters/spec_tables/gdf.h"
#include "encoder/filters/spec_tables/gdf_tables.h"
#include "encoder/filters/spec_tables/luma_source.h"

namespace {

using av2enc::cuda::gdf::GdfFrameSource;
using av2enc::cuda::gdf::GdfIndexParams;
using av2enc::cuda::gdf::GdfTableIndices;
using av2enc::cuda::gdf::GdfTables;
namespace spec = av2enc::filters::spec;
namespace tab = av2enc::filters::spec_tables;

int g_checks = 0;
int g_failures = 0;

void Check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

struct Rng {
  std::uint64_t state = 1;
  explicit Rng(std::uint64_t seed) : state(seed == 0 ? 1 : seed) {}
  std::uint32_t next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<std::uint32_t>(state >> 32);
  }
  int range(int low, int high) {  // inclusive
    return low + static_cast<int>(next() %
                                  static_cast<std::uint32_t>(high - low + 1));
  }
  bool coin() { return (next() & 1u) != 0; }
};

struct Plane {
  int width = 0;
  int height = 0;
  int stride = 0;
  std::vector<std::uint16_t> px;

  Plane() = default;
  Plane(int w, int h)
      : width(w), height(h), stride(w),
        px(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0) {}

  std::uint16_t& At(int x, int y) {
    return px[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
              static_cast<std::size_t>(x)];
  }
  std::uint16_t At(int x, int y) const {
    return px[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
              static_cast<std::size_t>(x)];
  }
};

struct Extents {
  int start_x = 0;
  int end_x = 0;
  int start_y = 0;
  int end_y = 0;
  int stripe_start_y = 0;
  int stripe_end_y = 0;
};

GdfFrameSource MakeSource(const Plane& curr, const Plane& cdef,
                          const Extents& e) {
  GdfFrameSource src;
  src.curr = curr.px.data();
  src.curr_stride = curr.stride;
  src.cdef = cdef.px.data();
  src.cdef_stride = cdef.stride;
  src.start_x = e.start_x;
  src.end_x = e.end_x;
  src.start_y = e.start_y;
  src.end_y = e.end_y;
  src.stripe_start_y = e.stripe_start_y;
  src.stripe_end_y = e.stripe_end_y;
  return src;
}

spec::LumaSource MakeCpuSource(const Plane& curr, const Plane& cdef,
                               const Extents& e) {
  spec::LumaSource src;
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

struct CaseStats {
  long long samples = 0;
  long long changed = 0;
  long long classes[4] = {0, 0, 0, 0};
};

void FillRandom(Plane* plane, Rng* rng, int bit_depth) {
  const int maxv = (1 << bit_depth) - 1;
  for (auto& v : plane->px) {
    v = static_cast<std::uint16_t>(rng->range(0, maxv));
  }
}

// Compares the shared device core against ApplyGdfFilter for one unit.
void RunCase(Rng* rng, const GdfTables& tables, int width, int height,
             int x, int y, int w, int h, int qp_idx, int ref_dst_idx,
             int pix_scale, int bit_depth, const Extents& e, CaseStats* stats,
             const char* label) {
  Plane curr(width, height);
  Plane cdef(width, height);
  Plane lr_cpu(width, height);
  Plane lr_impl(width, height);
  FillRandom(&curr, rng, bit_depth);
  FillRandom(&cdef, rng, bit_depth);
  FillRandom(&lr_cpu, rng, bit_depth);
  lr_impl.px = lr_cpu.px;

  const GdfFrameSource src = MakeSource(curr, cdef, e);
  av2enc::cuda::gdf::gdf_filter_unit(
      src, tables, lr_impl.px.data(), lr_impl.stride, x, y, w, h, qp_idx,
      ref_dst_idx, pix_scale, bit_depth);

  const spec::LumaSource cpu_src = MakeCpuSource(curr, cdef, e);
  spec::ApplyGdfFilter(cpu_src, lr_cpu.px.data(), lr_cpu.stride, x, y, qp_idx,
                       ref_dst_idx, w, h, e.stripe_end_y + 1, pix_scale,
                       bit_depth);

  for (int yy = 0; yy < height; ++yy) {
    for (int xx = 0; xx < width; ++xx) {
      ++stats->samples;
      ++g_checks;
      if (lr_impl.At(xx, yy) != lr_cpu.At(xx, yy)) {
        ++g_failures;
        if (g_failures < 8) {
          std::fprintf(stderr,
                       "FAIL %s bd=%d ref=%d qp=%d pix=%d unit=%d,%d %dx%d at "
                       "%d,%d: impl=%d cpu=%d\n",
                       label, bit_depth, ref_dst_idx, qp_idx, pix_scale, x, y,
                       w, h, xx, yy, lr_impl.At(xx, yy), lr_cpu.At(xx, yy));
        }
      }
    }
  }
  for (int yy = y; yy < y + h; ++yy) {
    for (int xx = x; xx < x + w; ++xx) {
      if (lr_impl.At(xx, yy) != lr_cpu.At(xx, yy)) {
        ++stats->changed;
      }
    }
  }

  // Class coverage uses the same cell sums as the filter.
  for (int ci = 0; ci < (h >> 1); ++ci) {
    for (int cj = 0; cj < (w >> 1); ++cj) {
      int sums[4];
      av2enc::cuda::gdf::gdf_cell_sums(src, x, y, ci, cj, sums);
      ++stats->classes[av2enc::cuda::gdf::gdf_class_from_sums(sums)];
    }
  }
}

void TestSourceSample() {
  Rng rng(0x1234abcd);
  Plane curr(37, 29);
  Plane cdef(37, 29);
  FillRandom(&curr, &rng, 12);
  FillRandom(&cdef, &rng, 12);
  Extents e;
  e.start_x = 3;
  e.end_x = 33;
  e.start_y = 2;
  e.end_y = 26;
  e.stripe_start_y = 9;
  e.stripe_end_y = 18;
  const GdfFrameSource src = MakeSource(curr, cdef, e);
  const spec::LumaSource cpu_src = MakeCpuSource(curr, cdef, e);
  for (int i = 0; i < 4000; ++i) {
    const int x = rng.range(-12, 48);
    const int y = rng.range(-12, 40);
    Check(av2enc::cuda::gdf::gdf_source_sample(src, x, y) ==
              static_cast<int>(spec::luma_source_sample(cpu_src, x, y)),
          "§7.20.2 get_gdf_sample parity");
  }
  // Explicit stripe clamps.
  Check(av2enc::cuda::gdf::gdf_source_sample(src, 10, 0) ==
            static_cast<int>(spec::luma_source_sample(cpu_src, 10, 0)),
        "§7.20.2 above-stripe clamp");
  Check(av2enc::cuda::gdf::gdf_source_sample(src, 10, 28) ==
            static_cast<int>(spec::luma_source_sample(cpu_src, 10, 28)),
        "§7.20.2 below-stripe clamp");
}

void TestDerivation() {
  int seen_ref[6] = {0, 0, 0, 0, 0, 0};
  int seen_qp[6] = {0, 0, 0, 0, 0, 0};

  auto compare = [&](const GdfIndexParams& params, const int* hints,
                     const int* dists) {
    const GdfTableIndices got =
        av2enc::cuda::gdf::gdf_derive_table_indices(params);
    spec::GdfParams cpu;
    cpu.frame_is_intra = params.frame_is_intra != 0;
    cpu.base_q_idx = params.base_q_idx;
    cpu.bit_depth = params.bit_depth;
    cpu.gdf_pic_qc_idx = params.gdf_pic_qc_idx;
    const spec::GdfBlockIndices want = spec::GdfBlockTableIndices(
        cpu, hints, dists, params.num_total_refs, params.restricted_oh);
    ++g_checks;
    if (got.qp_idx != want.qp_idx || got.ref_dst_idx != want.ref_dst_idx) {
      ++g_failures;
      if (g_failures < 8) {
        std::fprintf(stderr,
                     "FAIL §7.20.1: intra=%d bq=%d bd=%d qc=%d n=%d oh=(%d,%d) "
                     "d=(%d,%d) restricted=%d -> impl(%d,%d) cpu(%d,%d)\n",
                     params.frame_is_intra, params.base_q_idx,
                     params.bit_depth, params.gdf_pic_qc_idx,
                     params.num_total_refs, hints ? hints[0] : 0,
                     hints ? hints[1] : 0, dists ? dists[0] : 0,
                     dists ? dists[1] : 0, params.restricted_oh, got.qp_idx,
                     got.ref_dst_idx, want.qp_idx, want.ref_dst_idx);
      }
    }
    seen_qp[got.qp_idx]++;
    seen_ref[got.ref_dst_idx]++;
  };

  // Exhaustive frame-level grid; refs vary deterministically over the whole
  // range so every refDstIdx outcome appears.
  for (int intra = 0; intra <= 1; ++intra) {
    for (int bd : {8, 10, 12}) {
      for (int qc = 0; qc < 4; ++qc) {
        for (int bq = 0; bq <= 255; ++bq) {
          GdfIndexParams p;
          p.frame_is_intra = intra;
          p.base_q_idx = bq;
          p.bit_depth = bd;
          p.gdf_pic_qc_idx = qc;
          int hints[2] = {0, 1};
          int dists[2] = {0, 0};
          if (intra == 0) {
            const int max_dist = (bq + qc) % 16;
            hints[0] = 0;
            hints[1] = (bq & 1) ? 1 : 0;
            dists[0] = ((bq + qc) & 1) ? max_dist : -max_dist;
            dists[1] = (bq & 1) ? 0 : max_dist;
            p.num_total_refs = 2;
            p.order_hints = hints;
            p.frame_distance = dists;
            p.restricted_oh = 3;
          }
          compare(p, hints, dists);
        }
      }
    }
  }

  // Random reference arrays, including the RESTRICTED_OH exclusion and the
  // num_total_refs = 0..3 truncation.
  Rng rng(0x51ee7);
  for (int i = 0; i < 4000; ++i) {
    GdfIndexParams p;
    p.frame_is_intra = rng.coin() ? 1 : 0;
    p.base_q_idx = rng.range(0, 255);
    p.bit_depth = (rng.coin() ? 8 : (rng.coin() ? 10 : 12));
    p.gdf_pic_qc_idx = rng.range(0, 3);
    int hints[2] = {rng.range(-5, 9), rng.range(-5, 9)};
    int dists[2] = {rng.range(-20, 20), rng.range(-20, 20)};
    p.num_total_refs = rng.range(0, 3);
    p.restricted_oh = rng.range(-1, 9);
    if (rng.coin()) {
      hints[0] = p.restricted_oh;
    }
    if (rng.coin()) {
      dists[0] = 0;
      dists[1] = 0;
    }
    p.order_hints = p.frame_is_intra ? nullptr : hints;
    p.frame_distance = p.frame_is_intra ? nullptr : dists;
    compare(p, hints, dists);
  }

  for (int i = 0; i < 6; ++i) {
    Check(seen_ref[i] > 0, "all refDstIdx outcomes exercised");
    Check(seen_qp[i] > 0, "all qpIdx outcomes exercised");
  }
  std::printf(
      "  §7.20.1: refDstIdx counts 0..5 = %d,%d,%d,%d,%d,%d; qpIdx 0..5 = "
      "%d,%d,%d,%d,%d,%d\n",
      seen_ref[0], seen_ref[1], seen_ref[2], seen_ref[3], seen_ref[4],
      seen_ref[5], seen_qp[0], seen_qp[1], seen_qp[2], seen_qp[3], seen_qp[4],
      seen_qp[5]);
}

// Independent transcription of the CPU decision's GdfUnitRegions
// (src/encoder/filters/gdf_stream/gdf_decision.cpp), restricted to a
// single-tile frame.  Used only as an oracle for gdf_unit_regions.
int CpuUnitRegions(int unit_size, int row, int col, int cols, int rows,
                   int mi_rows, int width, int height,
                   av2enc::cuda::gdf::GdfRegion out[8]) {
  if (unit_size <= 0 || row < 0 || row >= rows || col < 0 || col >= cols) {
    return 0;
  }
  const int unit_mi = unit_size / 4;
  int start_mi = unit_mi * row - 2;
  int end_mi = start_mi + unit_mi - 1;
  if (row == rows - 1) end_mi = mi_rows - 1;
  if (end_mi >= mi_rows) end_mi = mi_rows - 1;
  if (start_mi > end_mi) return 0;

  int ys = std::max(0, start_mi * 4);
  int ye = std::min(height - 1, end_mi * 4 + 3);
  const int xs = col * unit_size;
  if (xs >= width) return 0;
  const int xe = std::min(xs + unit_size - 1, width - 1);

  int count = 0;
  int stripe = (ys + 8) / 64;
  while (stripe * 64 - 8 <= ye && count < 8) {
    const int unclipped_start = stripe * 64 - 8;
    const int stripe_start = std::max(0, unclipped_start);
    const int stripe_end = std::min(height - 1, unclipped_start + 63);
    const int region_y = std::max(ys, stripe_start);
    const int region_end = std::min(ye, stripe_end);
    if (region_y <= region_end) {
      av2enc::cuda::gdf::GdfRegion& region = out[count++];
      region.x = xs;
      region.y = region_y;
      region.w = xe - xs + 1;
      region.h = region_end - region_y + 1;
      region.stripe_start_y = stripe_start;
      region.stripe_end_y = stripe_end;
    }
    ++stripe;
  }
  return count;
}

void TestRegions() {
  Rng rng(0x1a2b3c4d);
  int max_regions = 0;
  for (int unit_size : {32, 64, 128}) {
    for (int height : {128, 192, 360, 1088}) {
      for (int width : {320, 640, 1920}) {
        const int mi_rows = height / 4;
        const int cols = (width + unit_size - 1) / unit_size;
        const int rows = (height + unit_size - 1) / unit_size;
        for (int row = 0; row < rows; ++row) {
          for (int col = 0; col < cols; ++col) {
            av2enc::cuda::gdf::GdfRegion got[8];
            av2enc::cuda::gdf::GdfRegion want[8];
            const int got_count = av2enc::cuda::gdf::gdf_unit_regions(
                unit_size, row, col, cols, rows, mi_rows, width, height, got);
            const int want_count =
                CpuUnitRegions(unit_size, row, col, cols, rows, mi_rows, width,
                               height, want);
            ++g_checks;
            if (got_count != want_count) {
              ++g_failures;
              if (g_failures < 8) {
                std::fprintf(stderr,
                             "FAIL regions: unit=%d %dx%d row=%d col=%d counts "
                             "%d != %d\n",
                             unit_size, width, height, row, col, got_count,
                             want_count);
              }
              continue;
            }
            max_regions = std::max(max_regions, want_count);
            for (int i = 0; i < want_count; ++i) {
              ++g_checks;
              if (got[i].x != want[i].x || got[i].y != want[i].y ||
                  got[i].w != want[i].w || got[i].h != want[i].h ||
                  got[i].stripe_start_y != want[i].stripe_start_y ||
                  got[i].stripe_end_y != want[i].stripe_end_y) {
                ++g_failures;
                if (g_failures < 8) {
                  std::fprintf(stderr,
                               "FAIL regions: unit=%d %dx%d row=%d col=%d "
                               "region=%d mismatch\n",
                               unit_size, width, height, row, col, i);
                }
              }
            }
          }
        }
      }
    }
  }
  // Random layouts, including odd last-row clamps.
  for (int i = 0; i < 3000; ++i) {
    const int unit_size = rng.coin() ? (rng.coin() ? 64 : 128) : 32;
    const int height = 2 * rng.range(1, 640);
    const int width = 2 * rng.range(1, 960);
    const int mi_rows = std::max(1, (height + 3) / 4);
    const int cols = (width + unit_size - 1) / unit_size;
    const int rows = (height + unit_size - 1) / unit_size;
    const int row = rng.range(0, rows - 1);
    const int col = rng.range(0, cols - 1);
    av2enc::cuda::gdf::GdfRegion got[8];
    av2enc::cuda::gdf::GdfRegion want[8];
    const int got_count = av2enc::cuda::gdf::gdf_unit_regions(
        unit_size, row, col, cols, rows, mi_rows, width, height, got);
    const int want_count = CpuUnitRegions(unit_size, row, col, cols, rows,
                                          mi_rows, width, height, want);
    Check(got_count == want_count, "region count matches CPU decision");
    for (int r = 0; r < std::min(got_count, want_count); ++r) {
      Check(got[r].x == want[r].x && got[r].y == want[r].y &&
                got[r].w == want[r].w && got[r].h == want[r].h &&
                got[r].stripe_start_y == want[r].stripe_start_y &&
                got[r].stripe_end_y == want[r].stripe_end_y,
            "region bounds match CPU decision");
    }
  }
  std::printf("  §7.20.1 stripe regions: max %d per unit, exact vs CPU\n",
              max_regions);
  Check(max_regions <= av2enc::cuda::gdf::kGdfMaxRegionsPerUnit,
        "region cap covers every layout");
}

Extents RandomExtents(Rng* rng, int width, int height) {
  Extents e;
  if (rng->coin()) {
    e.start_x = 0;
    e.end_x = width - 1;
    e.start_y = 0;
    e.end_y = height - 1;
    e.stripe_start_y = 0;
    e.stripe_end_y = height - 1;
    return e;
  }
  e.start_x = rng->range(0, width - 1);
  e.end_x = rng->range(e.start_x, width - 1);
  e.start_y = rng->range(0, height - 1);
  e.end_y = rng->range(e.start_y, height - 1);
  e.stripe_start_y = rng->range(0, height - 1);
  e.stripe_end_y = rng->range(e.stripe_start_y, height - 1);
  return e;
}

void TestAllCombinations(const GdfTables& tables, CaseStats* stats) {
  Rng rng(0x9e3779b9);
  const int width = 288;
  const int height = 288;
  for (int bit_depth : {8, 10, 12}) {
    for (int ref = 0; ref < 6; ++ref) {
      for (int qp = 0; qp < 6; ++qp) {
        for (int pix = 1; pix <= 4; ++pix) {
          const int w = 2 * rng.range(1, 16);  // 2..32
          const int h = 2 * rng.range(1, 16);
          const int x = rng.range(0, width - w);
          const int y = rng.range(0, height - h);
          RunCase(&rng, tables, width, height, x, y, w, h, qp, ref, pix,
                  bit_depth, RandomExtents(&rng, width, height), stats,
                  "all-combinations");
        }
      }
    }
  }
}

void TestUnitSizes(const GdfTables& tables, CaseStats* stats) {
  Rng rng(0x1234567);
  const int sizes[] = {2, 4, 8, 16, 32, 64, 128};
  const int width = 288;
  const int height = 288;
  int bd_index = 0;
  for (int ref = 0; ref < 6; ++ref) {
    for (int qp = 0; qp < 6; ++qp) {
      for (int size : sizes) {
        const int bit_depth = (bd_index % 3 == 0) ? 8
                               : (bd_index % 3 == 1) ? 10
                                                     : 12;
        ++bd_index;
        const int w = size;
        const int h = (size == 128 && ((ref + qp) & 1)) ? 64 : size;
        const int x = rng.range(0, width - w);
        const int y = rng.range(0, height - h);
        const int pix = 1 + ((ref + qp + size) % 4);
        RunCase(&rng, tables, width, height, x, y, w, h, qp, ref, pix,
                bit_depth, RandomExtents(&rng, width, height), stats,
                "unit-size");
      }
    }
  }
}

void TestEdgePositions(const GdfTables& tables, CaseStats* stats) {
  Rng rng(0xfeedface);
  const int width = 160;
  const int height = 128;
  const int sizes[] = {2, 8, 32, 64, 128};
  for (int bit_depth : {8, 10, 12}) {
    for (int size : sizes) {
      if (size > height) {
        continue;
      }
      const int xs[] = {0, width - size};
      const int ys[] = {0, height - size};
      for (int x : xs) {
        for (int y : ys) {
          for (int ref : {0, 1, 3, 5}) {
            Extents e;
            e.start_x = 0;
            e.end_x = width - 1;
            e.start_y = 0;
            e.end_y = height - 1;
            // Put the stripe boundary on and around the unit's top/bottom.
            e.stripe_start_y = std::max(0, y - 1);
            e.stripe_end_y = std::min(height - 1, y + size);
            RunCase(&rng, tables, width, height, x, y, size, size, 2, ref,
                    3, bit_depth, e, stats, "edge-position");
          }
        }
      }
    }
  }
}

void TestRandomBlocks(const GdfTables& tables, CaseStats* stats) {
  Rng rng(0xc001d00d);
  for (int i = 0; i < 260; ++i) {
    const int width = 320;
    const int height = 256;
    int w = 2 * rng.range(1, 64);  // 2..128
    int h = 2 * rng.range(1, 64);
    if (w > width) {
      w = width;
    }
    if (h > height) {
      h = height;
    }
    const int x = rng.range(0, width - w);
    const int y = rng.range(0, height - h);
    const int bit_depth = rng.coin() ? 8 : (rng.coin() ? 10 : 12);
    const int ref = rng.range(0, 5);
    const int qp = rng.range(0, 5);
    const int pix = rng.range(1, 4);
    RunCase(&rng, tables, width, height, x, y, w, h, qp, ref, pix,
            bit_depth, RandomExtents(&rng, width, height), stats,
            "random-block");
  }
}

}  // namespace

int main() {
  std::printf("gdf host-emulation parity (shared device core vs CPU reference)\n");

  TestSourceSample();
  TestDerivation();
  TestRegions();

  av2enc::cuda::gdf::GdfCoord coords[18];
  GdfTables tables;
  for (int i = 0; i < 18; ++i) {
    coords[i].dy = tab::kGdfCoords[i].dy;
    coords[i].dx = tab::kGdfCoords[i].dx;
  }
  tables.alpha = &tab::kGdfAlpha[0][0][0][0];
  tables.weight = &tab::kGdfWeight[0][0][0][0][0];
  tables.bias = &tab::kGdfBias[0][0][0];
  tables.intra_error = &tab::kGdfIntraError[0][0];
  tables.inter_error = &tab::kGdfInterError[0][0][0];
  tables.coords = coords;

  CaseStats stats;
  TestAllCombinations(tables, &stats);
  TestUnitSizes(tables, &stats);
  TestEdgePositions(tables, &stats);
  TestRandomBlocks(tables, &stats);

  std::printf("  filter: %lld samples compared, %lld differed inside units\n",
              stats.samples, stats.changed);
  std::printf("  classes: c0=%lld c1=%lld c2=%lld c3=%lld\n", stats.classes[0],
              stats.classes[1], stats.classes[2], stats.classes[3]);
  for (int c = 0; c < 4; ++c) {
    Check(stats.classes[c] > 0, "all four direction classes exercised");
  }

  if (g_failures == 0) {
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
  }
  std::printf("FAIL: %d checks, %d failures\n", g_checks, g_failures);
  return 1;
}
