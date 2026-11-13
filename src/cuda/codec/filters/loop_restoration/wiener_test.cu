// CPU-vs-CUDA bit-exactness and timing test for AV2 §7.20 loop restoration
// (non-separable Wiener filter).
//
// The kernel and the CPU reference share wiener_core.h; the CPU reference is
// checked against an independent spec transcription by wiener_test.cpp.
//
// Coverage:
//   * per-unit Wiener coefficients on luma and chroma
//   * chroma-from-luma taps, cfl_ds_filter_index 0..3
//   * stripe sample sourcing (CurrFrame outside / CdefFrame inside)
//   * frame-level filters with a subclass grid
//   * mixed LrType units, odd/partial geometry, guard columns
//   * tile clipping, lossless segments, BRU-inactive regions
//   * validation (GDF / PC-Wiener are refused)
//   * randomized frames
//   * 1280x720 timing
//
// Usage: ./wiener_test

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "cuda/codec/filters/loop_restoration/device_buffer.h"
#include "cuda/codec/filters/loop_restoration/wiener.h"
#include "codec/filters/loop_restoration/restoration.h"
#include "codec/filters/loop_restoration/restoration_test_common.h"

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

struct LrMeta {
  const std::uint8_t* segment_ids = nullptr;
  const std::uint8_t* chroma_segment_ids = nullptr;
  const std::uint8_t* lossless = nullptr;
  const std::uint8_t* bru_modes = nullptr;
  int use_bru = 0;
  int disable_across_tiles = 0;
  const std::uint16_t* mcs = nullptr;
  const std::uint16_t* mce = nullptr;
  const std::uint16_t* mrs = nullptr;
  const std::uint16_t* mre = nullptr;
  int cfl_ds_filter_index = 0;
};

struct DeviceFrame {
  RestorationFrame frame{};
  LoopRestorationParams params{};
  av2enc::cuda::test::DevBuf<std::uint8_t> curr[3];
  av2enc::cuda::test::DevBuf<std::uint8_t> cdef[3];
  av2enc::cuda::test::DevBuf<std::uint8_t> lr[3];
  av2enc::cuda::test::DevBuf<std::uint8_t> utype[3];
  av2enc::cuda::test::DevBuf<std::int16_t> ucoeff[3];
  av2enc::cuda::test::DevBuf<std::int16_t> fcoeff[3];
  av2enc::cuda::test::DevBuf<std::uint8_t> subclass;
  av2enc::cuda::test::DevBuf<std::uint8_t> seg;
  av2enc::cuda::test::DevBuf<std::uint8_t> chroma_seg;
  av2enc::cuda::test::DevBuf<std::uint8_t> lossless;
  av2enc::cuda::test::DevBuf<std::uint8_t> bru;
  av2enc::cuda::test::DevBuf<std::uint16_t> mcs, mce, mrs, mre;

  bool upload(const TestFrame& f, const LoopRestorationParams& p,
              const LrMeta& meta) {
    frame = RestorationFrame{};
    params = p;
    for (int pl = 0; pl < 3; ++pl) {
      const std::size_t n =
          static_cast<std::size_t>(f.stride[pl]) * (f.height[pl] + kGuardRows);
      if (!curr[pl].upload(f.curr[pl].data(), n) ||
          !cdef[pl].upload(f.cdef[pl].data(), n) ||
          !lr[pl].upload(f.lr[pl].data(), n)) {
        return false;
      }
      frame.curr[pl].data = curr[pl].ptr;
      frame.curr[pl].stride = f.stride[pl];
      frame.cdef[pl].data = cdef[pl].ptr;
      frame.cdef[pl].stride = f.stride[pl];
      frame.lr[pl].data = lr[pl].ptr;
      frame.lr[pl].stride = f.stride[pl];
      if (p.unit_type[pl] != nullptr) {
        const std::size_t units =
            static_cast<std::size_t>(p.unit_rows[pl]) * p.unit_cols[pl];
        if (!utype[pl].upload(p.unit_type[pl], units)) {
          return false;
        }
        params.unit_type[pl] = utype[pl].ptr;
      }
      if (p.unit_coeffs[pl] != nullptr) {
        const std::size_t coeffs =
            static_cast<std::size_t>(p.unit_rows[pl]) * p.unit_cols[pl] *
            kWienerNsCoeffStride;
        if (!ucoeff[pl].upload(p.unit_coeffs[pl], coeffs)) {
          return false;
        }
        params.unit_coeffs[pl] = ucoeff[pl].ptr;
      }
      if (p.frame_coeffs[pl] != nullptr) {
        const std::size_t coeffs =
            static_cast<std::size_t>(p.num_filter_classes) *
            kWienerNsCoeffStride;
        if (!fcoeff[pl].upload(p.frame_coeffs[pl], coeffs)) {
          return false;
        }
        params.frame_coeffs[pl] = fcoeff[pl].ptr;
      }
    }
    const std::size_t mi_n = static_cast<std::size_t>(f.mi_cols * f.mi_rows);
    if (p.subclass_grid != nullptr) {
      if (!subclass.upload(p.subclass_grid, mi_n)) {
        return false;
      }
      params.subclass_grid = subclass.ptr;
    }
    if (meta.segment_ids != nullptr && !seg.upload(meta.segment_ids, mi_n)) {
      return false;
    }
    if (meta.chroma_segment_ids != nullptr &&
        !chroma_seg.upload(meta.chroma_segment_ids, mi_n)) {
      return false;
    }
    if (meta.lossless != nullptr && !lossless.upload(meta.lossless, 8)) {
      return false;
    }
    if (meta.bru_modes != nullptr && !bru.upload(meta.bru_modes, mi_n)) {
      return false;
    }
    if (meta.disable_across_tiles != 0) {
      if (meta.mcs == nullptr || !mcs.upload(meta.mcs, mi_n) ||
          !mce.upload(meta.mce, mi_n) || !mrs.upload(meta.mrs, mi_n) ||
          !mre.upload(meta.mre, mi_n)) {
        return false;
      }
    }
    frame.mi_cols = f.mi_cols;
    frame.mi_rows = f.mi_rows;
    frame.subsampling_x = f.sub_x;
    frame.subsampling_y = f.sub_y;
    frame.bit_depth = 8;
    frame.segment_ids = seg.ptr;
    frame.chroma_segment_ids = chroma_seg.ptr;
    frame.lossless = lossless.ptr;
    frame.mi_col_start = meta.disable_across_tiles != 0 ? mcs.ptr : nullptr;
    frame.mi_col_end = meta.disable_across_tiles != 0 ? mce.ptr : nullptr;
    frame.mi_row_start = meta.disable_across_tiles != 0 ? mrs.ptr : nullptr;
    frame.mi_row_end = meta.disable_across_tiles != 0 ? mre.ptr : nullptr;
    frame.disable_loopfilters_across_tiles = meta.disable_across_tiles;
    frame.use_bru = meta.use_bru;
    frame.bru_modes = bru.ptr;
    frame.sb_shift = 4;
    frame.cfl_ds_filter_index = meta.cfl_ds_filter_index;
    return true;
  }

  bool download(TestFrame& f) {
    for (int pl = 0; pl < 3; ++pl) {
      const std::size_t n =
          static_cast<std::size_t>(f.stride[pl]) * (f.height[pl] + kGuardRows);
      if (!lr[pl].download(f.lr[pl].data(), n)) {
        return false;
      }
    }
    return true;
  }
};

bool run_case(TestFrame& f, const LoopRestorationParams& params,
              const LrMeta& meta, const char* tag) {
  TestFrame cpu = f;
  RestorationFrame cpu_view = cpu.view();
  cpu_view.segment_ids = meta.segment_ids;
  cpu_view.chroma_segment_ids = meta.chroma_segment_ids;
  cpu_view.lossless = meta.lossless;
  cpu_view.use_bru = meta.use_bru;
  cpu_view.bru_modes = meta.bru_modes;
  cpu_view.disable_loopfilters_across_tiles = meta.disable_across_tiles;
  cpu_view.mi_col_start = meta.mcs;
  cpu_view.mi_col_end = meta.mce;
  cpu_view.mi_row_start = meta.mrs;
  cpu_view.mi_row_end = meta.mre;
  cpu_view.cfl_ds_filter_index = meta.cfl_ds_filter_index;
  const RestorationStatus st = loop_restoration_frame(cpu_view, params);
  ++g_checks;
  if (st != RestorationStatus::kOk) {
    // The CPU oracle is the reference the device output is compared against.
    // When it refuses the fixture there is nothing to compare, so report the
    // fixture problem once instead of a launch error plus a bogus mismatch.
    ++g_failures;
    std::fprintf(stderr, "FAIL %s: CPU oracle rejected the fixture: %s\n", tag,
                 restoration_status_string(st));
    return false;
  }

  DeviceFrame dev;
  if (!dev.upload(f, params, meta)) {
    CHECK(false, "device upload failed");
    return false;
  }
  CUDA_CHECK(av2enc::cuda::loop_restoration_frame(dev.frame, dev.params),
             "loop_restoration_frame launch");
  if (!dev.download(f)) {
    CHECK(false, "device download failed");
    return false;
  }
  bool same = true;
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p]; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        if (f.at(f.lr[p], p, y, x) != cpu.at(cpu.lr[p], p, y, x)) {
          if (same) {
            std::fprintf(stderr,
                         "  first mismatch %s plane=%d (%d,%d): %d vs %d\n",
                         tag, p, x, y, f.at(f.lr[p], p, y, x),
                         cpu.at(cpu.lr[p], p, y, x));
          }
          same = false;
        }
      }
    }
  }
  CHECK(same, tag);
  CHECK(f.lr_guard_intact(), "LrFrame guard samples untouched");
  return same;
}

TestFrame make_frame(int mi_cols, int mi_rows) {
  TestFrame f;
  f.mi_cols = mi_cols;
  f.mi_rows = mi_rows;
  f.allocate();
  return f;
}

struct LrBuffers {
  std::vector<std::uint8_t> type[3];
  std::vector<std::int16_t> coeffs[3];
  std::vector<std::int16_t> frame_coeffs[3];
  std::vector<std::uint8_t> subclass;
};

void setup(const TestFrame& f, int unit_size, RestorationType t,
           LoopRestorationParams* p, LrBuffers* buf, int chroma_unit_size = -1) {
  for (int pl = 0; pl < 3; ++pl) {
    // §6.17.7.7: the luma unit must be at least one 64-line stripe; the chroma
    // unit may be half that (one chroma stripe).  Applying the luma size to
    // every plane makes the fixture non-conforming and the CPU oracle refuses
    // it before the device is ever launched (BUG-GPU-001).
    const int us = (pl == 0 || chroma_unit_size <= 0) ? unit_size
                                                      : chroma_unit_size;
    p->frame_type[pl] = t;
    p->unit_size[pl] = us;
    const int sub = (pl == 0) ? 0 : 1;
    const int rows = restoration_count_units(us, (f.mi_rows * 4) >> sub);
    const int cols = restoration_count_units(us, (f.mi_cols * 4) >> sub);
    p->unit_rows[pl] = rows;
    p->unit_cols[pl] = cols;
    buf->type[pl].assign(static_cast<std::size_t>(rows * cols),
                         static_cast<std::uint8_t>(kRestoreWienerNs));
    buf->coeffs[pl].assign(
        static_cast<std::size_t>(rows * cols) * kWienerNsCoeffStride, 0);
    p->unit_type[pl] = buf->type[pl].data();
    p->unit_coeffs[pl] = buf->coeffs[pl].data();
  }
}

void fill_coeffs(LrBuffers* buf, Rng* rng, int spread) {
  for (int pl = 0; pl < 3; ++pl) {
    for (std::size_t i = 0; i < buf->coeffs[pl].size(); ++i) {
      buf->coeffs[pl][i] =
          static_cast<std::int16_t>(rng->range(-spread, spread));
    }
  }
}

// --- tests -----------------------------------------------------------------

void test_per_unit_filters() {
  TestFrame f = make_frame(24, 18);
  Rng rng(3);
  fill_random(f, rng);
  LoopRestorationParams params;
  LrBuffers buf;
  setup(f, 64, kRestoreWienerNs, &params, &buf);
  fill_coeffs(&buf, &rng, 20);
  run_case(f, params, LrMeta{}, "per-unit Wiener");

  // Identity coefficients must also match (and equal CdefFrame).
  TestFrame g = make_frame(20, 12);
  Rng rng2(4);
  fill_random(g, rng2);
  LoopRestorationParams id;
  LrBuffers idbuf;
  setup(g, 64, kRestoreWienerNs, &id, &idbuf);
  run_case(g, id, LrMeta{}, "zero coefficients");
  std::cout << "  per-unit filters: random and zero coefficients\n";
}

void test_chroma_from_luma() {
  for (int cfl = 0; cfl < 4; ++cfl) {
    TestFrame f = make_frame(16, 10);
    Rng rng(static_cast<std::uint32_t>(20 + cfl));
    fill_random(f, rng);
    LoopRestorationParams params;
    LrBuffers buf;
    setup(f, 64, kRestoreWienerNs, &params, &buf);
    for (int pl = 0; pl < 3; ++pl) {
      std::int16_t* c = buf.coeffs[pl].data();
      if (pl == 0) {
        c[0] = 7;
        c[3] = -5;
      } else {
        for (int i = 0; i < 12; ++i) {
          c[6 + i] = static_cast<std::int16_t>((i % 5) - 2);
        }
      }
    }
    LrMeta meta;
    meta.cfl_ds_filter_index = cfl;
    run_case(f, params, meta, "chroma-from-luma");
  }
  std::cout << "  chroma from luma: cfl_ds_filter_index 0..3\n";
}

void test_stripe_sourcing() {
  TestFrame f = make_frame(16, 32);  // 64x128
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p]; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        f.at(f.cdef[p], p, y, x) = 100;
        f.at(f.curr[p], p, y, x) = 200;
      }
    }
  }
  LoopRestorationParams params;
  LrBuffers buf;
  setup(f, 64, kRestoreWienerNs, &params, &buf);
  for (int pl = 0; pl < 3; ++pl) {
    buf.coeffs[pl][0] = 128;  // (+1,0) and (-1,0)
  }
  run_case(f, params, LrMeta{}, "stripe sourcing");
  std::cout << "  stripe sourcing: CurrFrame outside the stripe\n";
}

void test_frame_filters() {
  TestFrame f = make_frame(16, 12);
  Rng rng(44);
  fill_random(f, rng);
  LoopRestorationParams params;
  LrBuffers buf;
  setup(f, 64, kRestoreWienerNs, &params, &buf);
  params.frame_filters_on[0] = 1;
  params.num_filter_classes = 2;
  buf.subclass.assign(static_cast<std::size_t>(f.mi_cols * f.mi_rows), 0);
  for (std::size_t i = 0; i < buf.subclass.size(); ++i) {
    buf.subclass[i] = static_cast<std::uint8_t>(rng.range(0, 1));
  }
  params.subclass_grid = buf.subclass.data();
  for (int pl = 0; pl < 3; ++pl) {
    buf.frame_coeffs[pl].assign(2 * kWienerNsCoeffStride, 0);
    for (std::size_t i = 0; i < buf.frame_coeffs[pl].size(); ++i) {
      buf.frame_coeffs[pl][i] =
          static_cast<std::int16_t>(rng.range(-18, 18));
    }
    params.frame_coeffs[pl] = buf.frame_coeffs[pl].data();
  }
  run_case(f, params, LrMeta{}, "frame filters");
  std::cout << "  frame filters: 2 pixel classes\n";
}

void test_types_odd_geometry() {
  const int dims[][2] = {{5, 3}, {17, 5}, {25, 9}, {7, 13}};
  for (const auto& d : dims) {
    Rng rng(static_cast<std::uint32_t>(d[0] * 977 + d[1]));
    TestFrame f = make_frame(d[0], d[1]);
    fill_random(f, rng);
    LoopRestorationParams params;
    LrBuffers buf;
    setup(f, 64, kRestoreSwitchable, &params, &buf, /*chroma=*/32);
    for (int pl = 0; pl < 3; ++pl) {
      for (std::size_t i = 0; i < buf.type[pl].size(); ++i) {
        buf.type[pl][i] = static_cast<std::uint8_t>(
            rng.range(0, 3) == 0 ? kRestoreNone : kRestoreWienerNs);
      }
    }
    fill_coeffs(&buf, &rng, 12);
    run_case(f, params, LrMeta{}, "mixed unit types");
  }
  std::cout << "  mixed unit types: 4 odd geometries\n";
}

void test_unit_sizes() {
  // §5.18.7.11 / §6.17.7.7: LoopRestorationSize is a power of two in 64..512
  // for luma and 32..256 for 4:2:0 chroma.  Exercise every legal size with a
  // frame large enough to hold one unit, plus a frame smaller than the unit
  // (the unit index must clamp to the last unit).
  struct Case {
    int mi_cols;
    int mi_rows;
    int luma;
    int chroma;
  };
  const Case cases[] = {
      {20, 12, 64, 32},
      {40, 24, 128, 64},
      {80, 48, 256, 128},
      {128, 128, 512, 256},
      {12, 8, 512, 256},  // unit larger than the frame
  };
  for (const Case& c : cases) {
    Rng rng(static_cast<std::uint32_t>(c.luma * 31 + c.mi_cols));
    TestFrame f = make_frame(c.mi_cols, c.mi_rows);
    fill_random(f, rng);
    LoopRestorationParams params;
    LrBuffers buf;
    setup(f, c.luma, kRestoreWienerNs, &params, &buf, c.chroma);
    fill_coeffs(&buf, &rng, 16);
    char tag[64];
    std::snprintf(tag, sizeof(tag), "unit %d/%d", c.luma, c.chroma);
    run_case(f, params, LrMeta{}, tag);
  }
  std::cout << "  unit sizes: 64/128/256/512 luma, 32..256 chroma\n";
}

void test_tiles_lossless_bru() {
  // §6.17.7.7 check_ru_size(): the LR unit must not cross an internal tile
  // boundary and the boundary spacing must be a multiple of maxPlaneRuSize.
  // With luma 64 / chroma 32 that maximum is 64 samples, so 64x64 synthetic
  // tiles (16 MI) are conforming and the CPU oracle accepts the fixture.
  TestFrame f = make_frame(32, 32);
  Rng rng(66);
  fill_random(f, rng);
  LoopRestorationParams params;
  LrBuffers buf;
  setup(f, 64, kRestoreWienerNs, &params, &buf, /*chroma=*/32);
  fill_coeffs(&buf, &rng, 14);
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
  LrMeta meta;
  meta.disable_across_tiles = 1;
  meta.mcs = mcs.data();
  meta.mce = mce.data();
  meta.mrs = mrs.data();
  meta.mre = mre.data();
  run_case(f, params, meta, "tile clipping");

  TestFrame g = make_frame(24, 16);
  Rng rng2(77);
  fill_random(g, rng2);
  std::vector<std::uint8_t> seg(static_cast<std::size_t>(g.mi_cols * g.mi_rows),
                                0);
  for (int y = 0; y < g.mi_rows; ++y) {
    for (int x = 0; x < g.mi_cols; ++x) {
      seg[static_cast<std::size_t>(y * g.mi_cols + x)] =
          static_cast<std::uint8_t>(y >= g.mi_rows / 2 ? 1 : 0);
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
  LrMeta meta2;
  meta2.segment_ids = seg.data();
  meta2.chroma_segment_ids = seg.data();
  meta2.lossless = lossless;
  meta2.bru_modes = bru.data();
  meta2.use_bru = 1;
  LoopRestorationParams params2;
  LrBuffers buf2;
  setup(g, 64, kRestoreWienerNs, &params2, &buf2);
  fill_coeffs(&buf2, &rng2, 10);
  run_case(g, params2, meta2, "lossless + BRU");
  std::cout << "  tiles, lossless segments, BRU-inactive regions\n";
}

void test_randomized() {
  for (int seed = 1; seed <= 16; ++seed) {
    Rng rng(static_cast<std::uint32_t>(seed * 6151));
    TestFrame f = make_frame(rng.range(6, 22), rng.range(4, 14));
    fill_random(f, rng, rng.range(0, 40), rng.range(200, 255));
    LoopRestorationParams params;
    LrBuffers buf;
    const int unit_size = rng.range(0, 1) ? 128 : 64;
    setup(f, unit_size, kRestoreWienerNs, &params, &buf, unit_size / 2);
    fill_coeffs(&buf, &rng, 22);
    if (rng.range(0, 3) == 0) {
      params.frame_filters_on[0] = 1;
      params.num_filter_classes = 2;
      buf.subclass.assign(static_cast<std::size_t>(f.mi_cols * f.mi_rows), 0);
      for (std::size_t i = 0; i < buf.subclass.size(); ++i) {
        buf.subclass[i] = static_cast<std::uint8_t>(rng.range(0, 1));
      }
      params.subclass_grid = buf.subclass.data();
      for (int pl = 0; pl < 3; ++pl) {
        buf.frame_coeffs[pl].assign(2 * kWienerNsCoeffStride, 0);
        for (std::size_t i = 0; i < buf.frame_coeffs[pl].size(); ++i) {
          buf.frame_coeffs[pl][i] =
              static_cast<std::int16_t>(rng.range(-16, 16));
        }
        params.frame_coeffs[pl] = buf.frame_coeffs[pl].data();
      }
    }
    LrMeta meta;
    meta.cfl_ds_filter_index = rng.range(0, 3);
    run_case(f, params, meta, "randomized");
  }
  std::cout << "  randomized: 16 frames\n";
}

void test_validation() {
  TestFrame f = make_frame(16, 8);
  Rng rng(5);
  fill_random(f, rng);
  LoopRestorationParams params;
  LrBuffers buf;
  setup(f, 64, kRestoreWienerNs, &params, &buf);
  DeviceFrame dev;
  if (!dev.upload(f, params, LrMeta{})) {
    CHECK(false, "validation upload failed");
    return;
  }
  LoopRestorationParams gdf = dev.params;
  gdf.gdf_frame_enable = 1;
  ++g_checks;
  if (av2enc::cuda::loop_restoration_frame(dev.frame, gdf) !=
      cudaErrorNotSupported) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: GDF not rejected\n");
  }
  LoopRestorationParams pc = dev.params;
  pc.frame_type[1] = kRestorePcWiener;
  ++g_checks;
  if (av2enc::cuda::loop_restoration_frame(dev.frame, pc) !=
      cudaErrorNotSupported) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: PC-Wiener not rejected\n");
  }
  RestorationFrame alias = dev.frame;
  alias.lr[0].data = alias.cdef[0].data;
  ++g_checks;
  if (av2enc::cuda::loop_restoration_frame(alias, dev.params) !=
      cudaErrorInvalidValue) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: aliasing not rejected\n");
  }
  // §6.17.7.7: LoopRestorationSize[0] < 64 is non-conforming.  The launcher
  // must refuse it instead of launching a kernel (BUG-GPU-001 regression pin:
  // the old fixtures used luma 32 and consumed the refusal as a launch error).
  LoopRestorationParams small = dev.params;
  small.unit_size[0] = 32;
  ++g_checks;
  if (av2enc::cuda::loop_restoration_frame(dev.frame, small) !=
      cudaErrorInvalidValue) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: non-conforming luma unit size not rejected\n");
  }
  std::cout << "  validation: GDF/PC-Wiener not supported, aliasing and "
               "luma unit < 64 invalid\n";
}

void test_timing() {
  TestFrame f = make_frame(320, 180);  // 1280x720
  Rng rng(321);
  fill_random(f, rng);
  LoopRestorationParams params;
  LrBuffers buf;
  setup(f, 64, kRestoreWienerNs, &params, &buf);
  fill_coeffs(&buf, &rng, 12);

  DeviceFrame dev;
  if (!dev.upload(f, params, LrMeta{})) {
    CHECK(false, "timing upload failed");
    return;
  }
  CUDA_CHECK(av2enc::cuda::loop_restoration_frame(dev.frame, dev.params),
             "timing launch");
  CUDA_CHECK(cudaDeviceSynchronize(), "timing sync");

  constexpr int kIters = 50;
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start), "event create");
  CUDA_CHECK(cudaEventCreate(&stop), "event create");
  CUDA_CHECK(cudaEventRecord(start), "event record");
  for (int i = 0; i < kIters; ++i) {
    CUDA_CHECK(av2enc::cuda::loop_restoration_frame(dev.frame, dev.params),
               "timing launch");
  }
  CUDA_CHECK(cudaEventRecord(stop), "event record");
  CUDA_CHECK(cudaEventSynchronize(stop), "event sync");
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  const double per_frame = static_cast<double>(ms) / kIters;
  const double mpx = 1280.0 * 720.0 * 1.5 / 1e6;
  std::printf(
      "  timing: 1280x720 Wiener NS %.3f ms/frame (%.1f MPix/s, %d iters)\n",
      per_frame, mpx / (per_frame * 1e-3), kIters);
}

}  // namespace

int main() {
  std::cout << "AV2 §7.20 loop restoration — CPU vs CUDA bit-exactness\n";
  test_per_unit_filters();
  test_chroma_from_luma();
  test_stripe_sourcing();
  test_frame_filters();
  test_types_odd_geometry();
  test_unit_sizes();
  test_tiles_lossless_bru();
  test_randomized();
  test_validation();
  test_timing();
  std::cout << (g_failures == 0 ? "PASS" : "FAIL") << ": " << g_checks
            << " checks, " << g_failures << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
