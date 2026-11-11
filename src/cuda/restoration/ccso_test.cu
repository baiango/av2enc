// CPU-vs-CUDA bit-exactness and timing test for AV2 §7.19 CCSO.
//
// The kernel and the CPU reference share ccso_core.h, so this test checks
// host/device codegen, thread mapping, stride/guard handling and metadata
// indexing.  The expected output comes from the CPU reference, which in turn
// is checked against an independent spec transcription by ccso_test.cpp.
//
// Coverage:
//   * bo_only band offsets, edge classes 0/1/2, all 7 extension filters
//   * all 16 (scale_idx, quant_idx) pairs
//   * per-plane CcsoBlks grids, disabled units
//   * lossless segments, BRU-inactive regions
//   * tile clipping (disable_loopfilters_across_tiles)
//   * odd/partial frames and guard columns
//   * randomized frames
//   * 1280x720 timing
//
// Usage: ./ccso_test

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "cuda/restoration/ccso.h"
#include "cuda/restoration/device_buffer.h"
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

#define CUDA_CHECK(expr, msg)                                               \
  do {                                                                      \
    const cudaError_t err__ = (expr);                                       \
    ++g_checks;                                                             \
    if (err__ != cudaSuccess) {                                             \
      ++g_failures;                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__,     \
                   (msg), cudaGetErrorString(err__));                       \
    }                                                                       \
  } while (0)

struct CcsoMeta {
  const std::uint8_t* segment_ids = nullptr;
  const std::uint8_t* chroma_segment_ids = nullptr;
  const std::uint8_t* lossless = nullptr;
  const std::uint8_t* bru_modes = nullptr;
  int disable_across_tiles = 0;
  const std::uint16_t* mcs = nullptr;
  const std::uint16_t* mce = nullptr;
  const std::uint16_t* mrs = nullptr;
  const std::uint16_t* mre = nullptr;
};

struct DeviceFrame {
  CcsoCoreFrame core{};
  av2enc::cuda::test::DevBuf<std::uint8_t> curr;
  av2enc::cuda::test::DevBuf<std::uint8_t> cdef[3];
  av2enc::cuda::test::DevBuf<std::uint8_t> blks[3];
  av2enc::cuda::test::DevBuf<std::int16_t> offsets[3];
  av2enc::cuda::test::DevBuf<std::uint8_t> seg;
  av2enc::cuda::test::DevBuf<std::uint8_t> chroma_seg;
  av2enc::cuda::test::DevBuf<std::uint8_t> lossless;
  av2enc::cuda::test::DevBuf<std::uint8_t> bru;
  av2enc::cuda::test::DevBuf<std::uint16_t> mcs, mce, mrs, mre;

  bool upload(const TestFrame& f, const CcsoFrameParams& params,
              const CcsoMeta& meta) {
    core = CcsoCoreFrame{};
    const std::size_t luma_n =
        static_cast<std::size_t>(f.stride[0]) * (f.height[0] + kGuardRows);
    if (!curr.upload(f.curr[0].data(), luma_n)) {
      return false;
    }
    for (int p = 0; p < 3; ++p) {
      const std::size_t n =
          static_cast<std::size_t>(f.stride[p]) * (f.height[p] + kGuardRows);
      if (!cdef[p].upload(f.cdef[p].data(), n)) {
        return false;
      }
      CcsoCorePlane& c = core.plane[p];
      c.enabled = params.plane[p].enabled;
      c.curr_luma = curr.ptr;
      c.curr_luma_stride = f.stride[0];
      c.dst = cdef[p].ptr;
      c.dst_stride = f.stride[p];
      c.scale_idx = params.plane[p].scale_idx;
      c.quant_idx = params.plane[p].quant_idx;
      c.ext_filter = params.plane[p].ext_filter;
      c.edge_clf = params.plane[p].edge_clf;
      c.max_band_log2 = params.plane[p].max_band_log2;
      c.bo_only = params.plane[p].bo_only;
      if (!offsets[p].upload(params.plane[p].offsets, kCcsoOffsetEntries)) {
        return false;
      }
      c.offsets = offsets[p].ptr;
      const int cols = params.blk_cols[p];
      const int rows = params.blk_rows[p];
      if (params.blks[p] != nullptr && cols > 0 && rows > 0) {
        if (!blks[p].upload(params.blks[p],
                            static_cast<std::size_t>(cols) * rows)) {
          return false;
        }
        c.blks = blks[p].ptr;
        c.blk_cols = cols;
        c.blk_rows = rows;
      }
    }
    if (meta.segment_ids != nullptr &&
        !seg.upload(meta.segment_ids,
                    static_cast<std::size_t>(f.mi_cols * f.mi_rows))) {
      return false;
    }
    if (meta.chroma_segment_ids != nullptr &&
        !chroma_seg.upload(meta.chroma_segment_ids,
                           static_cast<std::size_t>(f.mi_cols * f.mi_rows))) {
      return false;
    }
    if (meta.lossless != nullptr && !lossless.upload(meta.lossless, 8)) {
      return false;
    }
    if (meta.bru_modes != nullptr &&
        !bru.upload(meta.bru_modes,
                    static_cast<std::size_t>(f.mi_cols * f.mi_rows))) {
      return false;
    }
    const std::size_t mi_n = static_cast<std::size_t>(f.mi_cols * f.mi_rows);
    if (meta.disable_across_tiles != 0) {
      if (meta.mcs == nullptr || !mcs.upload(meta.mcs, mi_n) ||
          !mce.upload(meta.mce, mi_n) || !mrs.upload(meta.mrs, mi_n) ||
          !mre.upload(meta.mre, mi_n)) {
        return false;
      }
    }
    core.mi_cols = f.mi_cols;
    core.mi_rows = f.mi_rows;
    core.sub_x = f.sub_x;
    core.sub_y = f.sub_y;
    core.bit_depth = 8;
    core.ccso_luma_size_log2 = params.ccso_luma_size_log2;
    core.sb_width = 64;
    core.sb_height = 64;
    core.bru_modes = bru.ptr;
    core.segment_ids = seg.ptr;
    core.chroma_segment_ids = chroma_seg.ptr;
    core.lossless = lossless.ptr;
    core.mi_col_start = meta.disable_across_tiles != 0 ? mcs.ptr : nullptr;
    core.mi_col_end = meta.disable_across_tiles != 0 ? mce.ptr : nullptr;
    core.mi_row_start = meta.disable_across_tiles != 0 ? mrs.ptr : nullptr;
    core.mi_row_end = meta.disable_across_tiles != 0 ? mre.ptr : nullptr;
    core.disable_loopfilters_across_tiles = meta.disable_across_tiles;
    return true;
  }

  bool download(TestFrame& f) {
    for (int p = 0; p < 3; ++p) {
      const std::size_t n =
          static_cast<std::size_t>(f.stride[p]) * (f.height[p] + kGuardRows);
      if (!cdef[p].download(f.cdef[p].data(), n)) {
        return false;
      }
    }
    return true;
  }
};

// Runs CPU reference and GPU and requires a bit-exact match.
bool run_case(TestFrame& f, const CcsoFrameParams& params, const CcsoMeta& meta,
              const char* tag, bool check_guard = true) {
  TestFrame cpu = f;
  RestorationFrame cpu_view = cpu.view();
  cpu_view.segment_ids = meta.segment_ids;
  cpu_view.chroma_segment_ids = meta.chroma_segment_ids;
  cpu_view.lossless = meta.lossless;
  cpu_view.bru_modes = meta.bru_modes;
  cpu_view.disable_loopfilters_across_tiles = meta.disable_across_tiles;
  cpu_view.mi_col_start = meta.mcs;
  cpu_view.mi_col_end = meta.mce;
  cpu_view.mi_row_start = meta.mrs;
  cpu_view.mi_row_end = meta.mre;
  const RestorationStatus st = ccso_frame(cpu_view, params);
  CHECK(st == RestorationStatus::kOk, "CPU reference failed");

  DeviceFrame dev;
  if (!dev.upload(f, params, meta)) {
    CHECK(false, "device upload failed");
    return false;
  }
  CUDA_CHECK(av2enc::cuda::ccso_frame(dev.core), "ccso_frame launch");
  if (!dev.download(f)) {
    CHECK(false, "device download failed");
    return false;
  }
  bool same = true;
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p]; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        if (f.at(f.cdef[p], p, y, x) != cpu.at(cpu.cdef[p], p, y, x)) {
          if (same) {
            std::fprintf(stderr, "  first mismatch %s plane=%d (%d,%d): %d vs %d\n",
                         tag, p, x, y, f.at(f.cdef[p], p, y, x),
                         cpu.at(cpu.cdef[p], p, y, x));
          }
          same = false;
        }
      }
    }
  }
  CHECK(same, tag);
  if (check_guard) {
    CHECK(f.guard_intact(), "guard columns and rows untouched");
  }
  return same;
}

TestFrame make_frame(int mi_cols, int mi_rows) {
  TestFrame f;
  f.mi_cols = mi_cols;
  f.mi_rows = mi_rows;
  f.allocate();
  return f;
}

CcsoFrameParams base_params(int bands_log2, bool bo_only, int edge_clf,
                            int ext) {
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
    idx[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>((i * 5 + 3) & 7);
  }
  ccso_set_offsets(p, idx.data(), idx.size());
  return params;
}

void attach_unit_grid(CcsoFrameParams& params, std::vector<std::uint8_t>* grids,
                      const TestFrame& f) {
  for (int p = 0; p < 3; ++p) {
    const int sub = (p == 0) ? 0 : 1;
    const int shift = params.ccso_luma_size_log2 - sub;
    const int cols = (f.width[p] + (1 << shift) - 1) >> shift;
    const int rows = (f.height[p] + (1 << shift) - 1) >> shift;
    grids[p].assign(static_cast<std::size_t>(cols * rows), 1);
    params.blks[p] = grids[p].data();
    params.blk_cols[p] = cols;
    params.blk_rows[p] = rows;
  }
}

// --- tests -----------------------------------------------------------------

void test_modes_and_filters() {
  for (int edge_clf = 0; edge_clf <= 2; ++edge_clf) {
    for (int ext = 0; ext < 7; ++ext) {
      TestFrame f = make_frame(12, 7);
      Rng rng(static_cast<std::uint32_t>(edge_clf * 31 + ext + 1));
      fill_random(f, rng);
      CcsoFrameParams params = base_params(5, false, edge_clf, ext);
      std::vector<std::uint8_t> grids[3];
      attach_unit_grid(params, grids, f);
      run_case(f, params, CcsoMeta{}, "edge filter modes");
    }
  }
  for (int bands = 1; bands <= 6; ++bands) {
    TestFrame f = make_frame(10, 6);
    Rng rng(static_cast<std::uint32_t>(100 + bands));
    fill_random(f, rng);
    CcsoFrameParams params = base_params(bands, true, 0, 0);
    std::vector<std::uint8_t> grids[3];
    attach_unit_grid(params, grids, f);
    run_case(f, params, CcsoMeta{}, "bo_only bands");
  }
  std::cout << "  modes: 21 edge/filter combos, 6 bo_only band counts\n";
}

void test_quant_pairs() {
  for (int scale = 0; scale < 4; ++scale) {
    for (int quant = 0; quant < 4; ++quant) {
      TestFrame f = make_frame(9, 5);
      Rng rng(static_cast<std::uint32_t>(scale * 4 + quant + 1));
      fill_random(f, rng);
      CcsoFrameParams params = base_params(4, false, 0, 4);
      params.plane[0].scale_idx = scale;
      params.plane[0].quant_idx = quant;
      std::vector<std::uint8_t> grids[3];
      attach_unit_grid(params, grids, f);
      run_case(f, params, CcsoMeta{}, "quant pairs");
    }
  }
  std::cout << "  quant table: all 16 (scale, quant) pairs\n";
}

void test_planes_units_and_metadata() {
  // 96x48 luma: 2 unit columns at 64 luma.
  TestFrame f = make_frame(24, 12);
  Rng rng(7);
  fill_random(f, rng);
  CcsoFrameParams params = base_params(6, false, 1, 3);
  for (int p = 0; p < 3; ++p) {
    CcsoParams& cp = params.plane[p];
    cp.enabled = true;
    cp.scale_idx = p;
    cp.quant_idx = (p + 1) & 3;
    cp.ext_filter = p * 2;
    cp.edge_clf = p % 3;
  }
  std::vector<std::uint8_t> g0 = {1, 0};
  std::vector<std::uint8_t> g1 = {0, 1};
  std::vector<std::uint8_t> g2 = {1, 1};
  params.blks[0] = g0.data();
  params.blks[1] = g1.data();
  params.blks[2] = g2.data();
  for (int p = 0; p < 3; ++p) {
    params.blk_cols[p] = 2;
    params.blk_rows[p] = 1;
  }
  run_case(f, params, CcsoMeta{}, "per-plane grids");

  // Lossless + BRU.
  TestFrame g = make_frame(20, 10);
  Rng rng2(8);
  fill_random(g, rng2);
  std::vector<std::uint8_t> seg(static_cast<std::size_t>(g.mi_cols * g.mi_rows),
                                0);
  for (int y = 0; y < g.mi_rows; ++y) {
    for (int x = 0; x < g.mi_cols; ++x) {
      seg[static_cast<std::size_t>(y * g.mi_cols + x)] =
          static_cast<std::uint8_t>(x >= g.mi_cols / 2 ? 1 : 0);
    }
  }
  std::uint8_t lossless[8] = {0, 1, 0, 0, 0, 0, 0, 0};
  std::vector<std::uint8_t> bru(
      static_cast<std::size_t>(g.mi_cols * g.mi_rows), kBruActive);
  for (int y = 0; y < g.mi_rows / 2; ++y) {
    for (int x = 0; x < g.mi_cols; ++x) {
      bru[static_cast<std::size_t>(y * g.mi_cols + x)] = kBruInactive;
    }
  }
  CcsoMeta meta;
  meta.segment_ids = seg.data();
  meta.chroma_segment_ids = seg.data();
  meta.lossless = lossless;
  meta.bru_modes = bru.data();
  std::vector<std::uint8_t> grids[3];
  attach_unit_grid(params, grids, g);
  run_case(g, params, meta, "lossless + BRU");

  // Tile clipping.
  TestFrame t = make_frame(32, 16);
  Rng rng3(9);
  fill_random(t, rng3);
  const int n = t.mi_cols * t.mi_rows;
  std::vector<std::uint16_t> mcs(static_cast<std::size_t>(n));
  std::vector<std::uint16_t> mce(static_cast<std::size_t>(n));
  std::vector<std::uint16_t> mrs(static_cast<std::size_t>(n));
  std::vector<std::uint16_t> mre(static_cast<std::size_t>(n));
  for (int y = 0; y < t.mi_rows; ++y) {
    for (int x = 0; x < t.mi_cols; ++x) {
      const std::size_t i = static_cast<std::size_t>(y * t.mi_cols + x);
      mcs[i] = static_cast<std::uint16_t>(x < 16 ? 0 : 16);
      mce[i] = static_cast<std::uint16_t>(x < 16 ? 16 : 32);
      mrs[i] = 0;
      mre[i] = 16;
    }
  }
  CcsoMeta tmeta;
  tmeta.disable_across_tiles = 1;
  tmeta.mcs = mcs.data();
  tmeta.mce = mce.data();
  tmeta.mrs = mrs.data();
  tmeta.mre = mre.data();
  std::vector<std::uint8_t> tgrids[3];
  attach_unit_grid(params, tgrids, t);
  run_case(t, params, tmeta, "tile clipping");
  std::cout << "  planes/units/metadata: grids, lossless, BRU, tiles\n";
}

void test_odd_geometry() {
  const int dims[][2] = {{5, 3}, {17, 5}, {21, 11}, {7, 13}};
  for (const auto& d : dims) {
    TestFrame f = make_frame(d[0], d[1]);
    Rng rng(static_cast<std::uint32_t>(d[0] * 131 + d[1]));
    fill_random(f, rng);
    CcsoFrameParams params = base_params(6, false, 0, 6);
    std::vector<std::uint8_t> grids[3];
    attach_unit_grid(params, grids, f);
    run_case(f, params, CcsoMeta{}, "odd geometry");
  }
  std::cout << "  odd geometry: 5x3, 17x5, 21x11, 7x13 MI\n";
}

void test_randomized() {
  for (int seed = 1; seed <= 24; ++seed) {
    Rng rng(static_cast<std::uint32_t>(seed * 3571));
    TestFrame f = make_frame(rng.range(6, 28), rng.range(4, 16));
    fill_random(f, rng);
    CcsoFrameParams params;
    params.ccso_luma_size_log2 = rng.range(6, 8);
    std::vector<std::uint8_t> grids[3];
    for (int p = 0; p < 3; ++p) {
      CcsoParams& cp = params.plane[p];
      cp.enabled = rng.range(0, 3) != 0;
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
      ccso_set_offsets(cp, idx.data(), idx.size());
      const int sub = (p == 0) ? 0 : 1;
      const int shift = params.ccso_luma_size_log2 - sub;
      const int cols = (f.width[p] + (1 << shift) - 1) >> shift;
      const int rows = (f.height[p] + (1 << shift) - 1) >> shift;
      grids[p].assign(static_cast<std::size_t>(cols * rows), 0);
      for (std::size_t i = 0; i < grids[p].size(); ++i) {
        grids[p][i] = static_cast<std::uint8_t>(rng.range(0, 3) != 0 ? 1 : 0);
      }
      params.blks[p] = grids[p].data();
      params.blk_cols[p] = cols;
      params.blk_rows[p] = rows;
    }
    run_case(f, params, CcsoMeta{}, "randomized");
  }
  std::cout << "  randomized: 24 frames\n";
}

void test_validation() {
  CcsoCoreFrame core{};
  core.mi_cols = 16;
  core.mi_rows = 8;
  core.sub_x = 1;
  core.sub_y = 1;
  core.bit_depth = 8;
  core.ccso_luma_size_log2 = 6;
  core.sb_width = 64;
  core.sb_height = 64;
  ++g_checks;
  if (av2enc::cuda::ccso_frame(core) != cudaErrorInvalidValue) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: incomplete core not rejected\n");
  }
  core.bit_depth = 10;
  ++g_checks;
  if (av2enc::cuda::ccso_frame(core) != cudaErrorInvalidValue) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: 10-bit core not rejected\n");
  }
  std::cout << "  validation: incomplete core, 10-bit rejected\n";
}

void test_timing() {
  // 1280x720: mi 320x180, one CCSO unit 64, all three planes enabled.
  TestFrame f = make_frame(320, 180);
  Rng rng(123);
  fill_random(f, rng);
  CcsoFrameParams params = base_params(6, false, 0, 2);
  std::vector<std::uint8_t> grids[3];
  for (int p = 0; p < 3; ++p) {
    params.plane[p] = params.plane[0];
    params.plane[p].ext_filter = p;
  }
  attach_unit_grid(params, grids, f);

  DeviceFrame dev;
  if (!dev.upload(f, params, CcsoMeta{})) {
    CHECK(false, "timing upload failed");
    return;
  }
  // Warm-up and correctness once.
  CUDA_CHECK(av2enc::cuda::ccso_frame(dev.core), "timing launch");
  CUDA_CHECK(cudaDeviceSynchronize(), "timing sync");

  constexpr int kIters = 100;
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start), "event create");
  CUDA_CHECK(cudaEventCreate(&stop), "event create");
  CUDA_CHECK(cudaEventRecord(start), "event record");
  for (int i = 0; i < kIters; ++i) {
    CUDA_CHECK(av2enc::cuda::ccso_frame(dev.core), "timing launch");
  }
  CUDA_CHECK(cudaEventRecord(stop), "event record");
  CUDA_CHECK(cudaEventSynchronize(stop), "event sync");
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  const double per_frame = static_cast<double>(ms) / kIters;
  const double mpx = 1280.0 * 720.0 * 1.5 / 1e6;
  std::printf("  timing: 1280x720 CCSO %.3f ms/frame (%.1f MPix/s, %d iters)\n",
              per_frame, mpx / (per_frame * 1e-3), kIters);
}

}  // namespace

int main() {
  std::cout << "AV2 §7.19 CCSO — CPU vs CUDA bit-exactness\n";
  test_modes_and_filters();
  test_quant_pairs();
  test_planes_units_and_metadata();
  test_odd_geometry();
  test_randomized();
  test_validation();
  test_timing();
  std::cout << (g_failures == 0 ? "PASS" : "FAIL") << ": " << g_checks
            << " checks, " << g_failures << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
