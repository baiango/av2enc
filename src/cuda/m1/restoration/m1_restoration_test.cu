// CPU-vs-GPU parity for the M1 post-CDEF restoration stage.
//
// The GPU path calls the same host/device cores as the CPU reference, so the
// expectation is bit-exact.  This test checks whole-frame output (CCSO-modified
// CdefFrame and the final LrFrame), including guard columns/rows, on synthetic
// randomized frames and on real decoder reconstructions.
//
// Real recon: pass one or more `--ivf FILE` options.  The test decodes each
// with dav2d (default tools/bin/dav2d) to a temporary Y4M and uses the decoded
// first frame as the deblocked input (`curr = cdef`, CDEF off).  That is the
// real 64x64 / 640x360 reconstruction used by the M1 verification harness.
//
// Build/run: ./build_and_test.sh  (see README.md).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <unistd.h>

#include "encoder/filters/restoration/restoration.h"
#include "encoder/filters/restoration/restoration_test_common.h"
#include "m1_restoration.h"

#include "../../restoration/device_buffer.h"

namespace {

using av2enc::restoration::CcsoFrameParams;
using av2enc::restoration::CcsoParams;
using av2enc::restoration::LoopRestorationParams;
using av2enc::restoration::RestorationFrame;
using av2enc::restoration::RestorationStatus;
using av2enc::restoration::test::TestFrame;
using av2enc::restoration::test::Rng;

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

const char* cuda_err(cudaError_t e) { return cudaGetErrorString(e); }

// ---------------------------------------------------------------------------
// Parameter containers
// ---------------------------------------------------------------------------

struct LrHost {
  std::vector<std::uint8_t> type[3];
  std::vector<std::int16_t> coeffs[3];
  std::vector<std::int16_t> frame_coeffs[3];
  std::vector<std::uint8_t> filter_class;
  std::vector<std::uint8_t> subclass_lookup;
  std::vector<std::uint8_t> subclass;
  std::vector<std::uint8_t> seg;
  std::vector<std::uint8_t> chroma_seg;
  std::uint8_t lossless[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<std::uint8_t> bru;
  std::vector<std::uint16_t> mcs, mce, mrs, mre;
};

struct CcsoHost {
  std::vector<std::uint8_t> blks[3];
};

int units(int unit_size, int frame_size) {
  return av2enc::restoration::restoration_count_units(unit_size, frame_size);
}

void make_per_unit_lr(const TestFrame& f, int luma_unit, int chroma_unit,
                      LoopRestorationParams* p, LrHost* h) {
  for (int pl = 0; pl < 3; ++pl) {
    const int us = (pl == 0) ? luma_unit : chroma_unit;
    const int sub = (pl == 0) ? 0 : 1;
    const int rows = units(us, (f.mi_rows * 4) >> sub);
    const int cols = units(us, (f.mi_cols * 4) >> sub);
    p->frame_type[pl] = av2enc::restoration::kRestoreWienerNs;
    p->unit_size[pl] = us;
    p->unit_rows[pl] = rows;
    p->unit_cols[pl] = cols;
    h->type[pl].assign(static_cast<std::size_t>(rows * cols),
                       av2enc::restoration::kRestoreWienerNs);
    h->coeffs[pl].assign(
        static_cast<std::size_t>(rows * cols) *
            av2enc::restoration::kWienerNsCoeffStride,
        0);
    p->unit_type[pl] = h->type[pl].data();
    p->unit_coeffs[pl] = h->coeffs[pl].data();
  }
  // A mild smoothing filter, inside the §6.17.7 coefficient ranges.
  static const std::int16_t kY[16] = {16, -6, 8, -4, 3, -2, 2,  -1,
                                      -1, 1,  -1, 0, 0, 0,  0, 0};
  static const std::int16_t kUv[18] = {14, -5, 6, -3, 2, -1, 1, -1, 0,
                                       0,  0,  0, 4,  -2, 1, 0, 0, 0};
  for (int pl = 0; pl < 3; ++pl) {
    const int n = (pl == 0) ? 16 : 18;
    for (std::size_t u = 0; u < h->type[pl].size(); ++u) {
      for (int i = 0; i < n; ++i) {
        h->coeffs[pl][u * av2enc::restoration::kWienerNsCoeffStride + i] =
            (pl == 0) ? kY[i] : kUv[i];
      }
    }
  }
}

void make_frame_filters(const TestFrame& f, int unit_size,
                        LoopRestorationParams* p, LrHost* h) {
  make_per_unit_lr(f, unit_size, unit_size, p, h);
  p->frame_filters_on[0] = 1;
  p->frame_filters_on[1] = 1;
  p->frame_filters_on[2] = 1;
  p->num_filter_classes = 2;
  h->filter_class.assign(static_cast<std::size_t>(f.mi_cols * f.mi_rows), 0);
  for (int y = 0; y < f.mi_rows; ++y) {
    for (int x = 0; x < f.mi_cols; ++x) {
      h->filter_class[static_cast<std::size_t>(y * f.mi_cols + x)] =
          static_cast<std::uint8_t>((x * 3 + y * 5) & 15);
    }
  }
  h->subclass_lookup.assign(16, 0);
  for (int i = 0; i < 16; ++i) {
    h->subclass_lookup[static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>(i & 1);
  }
  p->filter_class_grid = h->filter_class.data();
  p->subclass_lookup = h->subclass_lookup.data();
  for (int pl = 0; pl < 3; ++pl) {
    h->frame_coeffs[pl].assign(2 * av2enc::restoration::kWienerNsCoeffStride,
                               0);
    p->frame_coeffs[pl] = h->frame_coeffs[pl].data();
    for (int c = 0; c < 2; ++c) {
      for (int i = 0; i < 16; ++i) {
        h->frame_coeffs[pl][c * 18 + i] =
            static_cast<std::int16_t>((pl == 0 ? 10 : 7) - i - c);
      }
      for (int i = 0; i < 6; ++i) {
        h->frame_coeffs[pl][c * 18 + i + 6] =
            static_cast<std::int16_t>((pl == 0) ? 0 : (i == 0 ? 3 : 0));
      }
    }
  }
}

void make_ccso(const TestFrame& f, int luma_size_log2, bool bo_only,
               bool all_planes, CcsoFrameParams* p, CcsoHost* h) {
  p->ccso_luma_size_log2 = luma_size_log2;
  for (int pl = 0; pl < 3; ++pl) {
    p->plane[pl].enabled = all_planes || pl == 0;
    p->plane[pl].bo_only = bo_only;
    p->plane[pl].scale_idx = 1;
    p->plane[pl].quant_idx = 2;
    p->plane[pl].ext_filter = 3;
    p->plane[pl].edge_clf = 1;
    p->plane[pl].max_band_log2 = 3;
    const int count = av2enc::restoration::ccso_offset_count(p->plane[pl]);
    std::vector<std::uint8_t> idx(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
      idx[static_cast<std::size_t>(i)] =
          static_cast<std::uint8_t>((i * 7 + pl) & 7);
    }
    CHECK(av2enc::restoration::ccso_set_offsets(p->plane[pl], idx.data(),
                                                idx.size()),
          "ccso offsets");
    const int sub = (pl == 0) ? 0 : 1;
    int cols = 0;
    int rows = 0;
    av2enc::restoration::ccso_blk_grid_size(
        f.width[pl], f.height[pl], luma_size_log2, sub, sub, &cols, &rows);
    h->blks[pl].assign(static_cast<std::size_t>(cols * rows), 1);
    p->blks[pl] = h->blks[pl].data();
    p->blk_cols[pl] = cols;
    p->blk_rows[pl] = rows;
  }
}

// ---------------------------------------------------------------------------
// Device mirror
// ---------------------------------------------------------------------------

template <typename T>
void upload(av2enc::cuda::test::DevBuf<T>& dst, const std::vector<T>& src) {
  CHECK(dst.upload(src.data(), src.size()), "device upload");
}

template <typename T>
const T* maybe(const av2enc::cuda::test::DevBuf<T>& buf,
               const std::vector<T>& src) {
  return src.empty() ? nullptr : buf.ptr;
}

struct DeviceCase {
  av2enc::cuda::test::DevBuf<std::uint8_t> curr[3], cdef[3], lr[3];
  av2enc::cuda::test::DevBuf<std::uint8_t> type[3], blks[3], filter_class;
  av2enc::cuda::test::DevBuf<std::uint8_t> subclass_lookup, subclass;
  av2enc::cuda::test::DevBuf<std::uint8_t> seg, chroma_seg, lossless, bru;
  av2enc::cuda::test::DevBuf<std::uint16_t> mcs, mce, mrs, mre;
  av2enc::cuda::test::DevBuf<std::int16_t> coeffs[3], frame_coeffs[3];
  av2enc::cuda::test::DevBuf<std::int16_t> offsets[3];
};

void build_device_case(const TestFrame& f, const CcsoFrameParams& ccso,
                       const LoopRestorationParams& lr, const LrHost& lh,
                       const CcsoHost& ch, bool use_metadata,
                       DeviceCase* d, RestorationFrame* view,
                       av2enc::cuda::m1::RestorationStageParams* stage) {
  for (int p = 0; p < 3; ++p) {
    const std::size_t n = static_cast<std::size_t>(f.stride[p]) *
                          static_cast<std::size_t>(f.height[p] +
                                                   av2enc::restoration::test::
                                                       kGuardRows);
    CHECK(d->curr[p].upload(f.curr[p].data(), n), "upload curr");
    CHECK(d->cdef[p].upload(f.cdef[p].data(), n), "upload cdef");
    CHECK(d->lr[p].upload(f.lr[p].data(), n), "upload lr");
    view->curr[p].data = d->curr[p].ptr;
    view->curr[p].stride = f.stride[p];
    view->cdef[p].data = d->cdef[p].ptr;
    view->cdef[p].stride = f.stride[p];
    view->lr[p].data = d->lr[p].ptr;
    view->lr[p].stride = f.stride[p];
    upload(d->type[p], lh.type[p]);
    if (!lh.coeffs[p].empty()) {
      upload(d->coeffs[p], lh.coeffs[p]);
    }
    if (!lh.frame_coeffs[p].empty()) {
      upload(d->frame_coeffs[p], lh.frame_coeffs[p]);
    }
    upload(d->offsets[p], std::vector<std::int16_t>(
                              ccso.plane[p].offsets,
                              ccso.plane[p].offsets +
                                  av2enc::restoration::kCcsoOffsetEntries));
    upload(d->blks[p], ch.blks[p]);
  }
  view->mi_cols = f.mi_cols;
  view->mi_rows = f.mi_rows;
  view->subsampling_x = f.sub_x;
  view->subsampling_y = f.sub_y;
  view->bit_depth = 8;

  *stage = av2enc::cuda::m1::RestorationStageParams{};
  stage->ccso.ccso_luma_size_log2 = ccso.ccso_luma_size_log2;
  for (int p = 0; p < 3; ++p) {
    stage->ccso.plane[p] = ccso.plane[p];
    stage->ccso.offsets_dev[p] = d->offsets[p].ptr;
    stage->ccso.blks[p] = d->blks[p].ptr;
    stage->ccso.blk_cols[p] = ccso.blk_cols[p];
    stage->ccso.blk_rows[p] = ccso.blk_rows[p];
    stage->lr.frame_type[p] = lr.frame_type[p];
    stage->lr.unit_size[p] = lr.unit_size[p];
    stage->lr.unit_rows[p] = lr.unit_rows[p];
    stage->lr.unit_cols[p] = lr.unit_cols[p];
    stage->lr.unit_type[p] = d->type[p].ptr;
    stage->lr.unit_coeffs[p] = maybe(d->coeffs[p], lh.coeffs[p]);
    stage->lr.frame_coeffs[p] = maybe(d->frame_coeffs[p], lh.frame_coeffs[p]);
  }
  stage->lr.frame_filters_on[0] = lr.frame_filters_on[0];
  stage->lr.frame_filters_on[1] = lr.frame_filters_on[1];
  stage->lr.frame_filters_on[2] = lr.frame_filters_on[2];
  stage->lr.num_filter_classes = lr.num_filter_classes;

  if (use_metadata) {
    upload(d->filter_class, lh.filter_class);
    upload(d->subclass_lookup, lh.subclass_lookup);
    upload(d->subclass, lh.subclass);
    upload(d->seg, lh.seg);
    upload(d->chroma_seg, lh.chroma_seg);
    upload(d->lossless, std::vector<std::uint8_t>(
                            lh.lossless, lh.lossless + 8));
    upload(d->bru, lh.bru);
    upload(d->mcs, lh.mcs);
    upload(d->mce, lh.mce);
    upload(d->mrs, lh.mrs);
    upload(d->mre, lh.mre);
    view->segment_ids = maybe(d->seg, lh.seg);
    view->chroma_segment_ids = maybe(d->chroma_seg, lh.chroma_seg);
    view->lossless = d->lossless.ptr;
    view->use_bru = lh.bru.empty() ? 0 : 1;
    view->bru_modes = maybe(d->bru, lh.bru);
    view->disable_loopfilters_across_tiles =
        lh.mcs.empty() ? 0 : 1;
    view->mi_col_start = maybe(d->mcs, lh.mcs);
    view->mi_col_end = maybe(d->mce, lh.mce);
    view->mi_row_start = maybe(d->mrs, lh.mrs);
    view->mi_row_end = maybe(d->mre, lh.mre);
    stage->lr.filter_class_grid = maybe(d->filter_class, lh.filter_class);
    stage->lr.subclass_lookup =
        maybe(d->subclass_lookup, lh.subclass_lookup);
    stage->lr.subclass_grid = maybe(d->subclass, lh.subclass);
  }
  stage->lr.gdf_frame_enable = 0;
}

bool compare_planes(const TestFrame& cpu, const TestFrame& gpu,
                    const char* label) {
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < gpu.height[p]; ++y) {
      for (int x = 0; x < gpu.width[p]; ++x) {
        const int a = cpu.at(cpu.lr[p], p, y, x);
        const int b = gpu.at(gpu.lr[p], p, y, x);
        if (a != b) {
          std::fprintf(stderr, "  %s lr plane %d (%d,%d): cpu %d gpu %d\n",
                       label, p, x, y, a, b);
          CHECK(false, label);
          return false;
        }
      }
    }
    for (int y = 0; y < gpu.height[p]; ++y) {
      for (int x = 0; x < gpu.width[p]; ++x) {
        const int a = cpu.at(cpu.cdef[p], p, y, x);
        const int b = gpu.at(gpu.cdef[p], p, y, x);
        if (a != b) {
          std::fprintf(stderr, "  %s cdef plane %d (%d,%d): cpu %d gpu %d\n",
                       label, p, x, y, a, b);
          CHECK(false, label);
          return false;
        }
      }
    }
  }
  CHECK(true, label);
  return true;
}

void run_case(const TestFrame& input, const CcsoFrameParams& ccso,
              const LoopRestorationParams& lr, const LrHost& lh,
              const CcsoHost& ch, bool use_metadata, const char* label) {
  TestFrame cpu = input;
  RestorationFrame cpu_view = cpu.view();
  cpu_view.segment_ids = lh.seg.empty() ? nullptr : lh.seg.data();
  cpu_view.chroma_segment_ids =
      lh.chroma_seg.empty() ? nullptr : lh.chroma_seg.data();
  cpu_view.lossless = lh.lossless;
  cpu_view.use_bru = lh.bru.empty() ? 0 : 1;
  cpu_view.bru_modes = lh.bru.empty() ? nullptr : lh.bru.data();
  cpu_view.disable_loopfilters_across_tiles = lh.mcs.empty() ? 0 : 1;
  cpu_view.mi_col_start = lh.mcs.empty() ? nullptr : lh.mcs.data();
  cpu_view.mi_col_end = lh.mce.empty() ? nullptr : lh.mce.data();
  cpu_view.mi_row_start = lh.mrs.empty() ? nullptr : lh.mrs.data();
  cpu_view.mi_row_end = lh.mre.empty() ? nullptr : lh.mre.data();
  CHECK(av2enc::restoration::ccso_frame(cpu_view, ccso) ==
            RestorationStatus::kOk,
        "cpu ccso");
  CHECK(av2enc::restoration::loop_restoration_frame(cpu_view, lr) ==
            RestorationStatus::kOk,
        "cpu lr");

  TestFrame gpu = input;
  DeviceCase d;
  RestorationFrame gpu_view{};
  av2enc::cuda::m1::RestorationStageParams stage{};
  build_device_case(gpu, ccso, lr, lh, ch, use_metadata, &d, &gpu_view,
                    &stage);
  const cudaError_t err =
      av2enc::cuda::m1::restoration_stage_apply(gpu_view, stage, nullptr);
  CHECK(err == cudaSuccess, cuda_err(err));
  if (err != cudaSuccess) {
    return;
  }
  CHECK(cudaDeviceSynchronize() == cudaSuccess, "sync");
  for (int p = 0; p < 3; ++p) {
    const std::size_t n = static_cast<std::size_t>(gpu.stride[p]) *
                          static_cast<std::size_t>(
                              gpu.height[p] + av2enc::restoration::test::
                                                  kGuardRows);
    CHECK(d.lr[p].download(gpu.lr[p].data(), n), "download lr");
    CHECK(d.cdef[p].download(gpu.cdef[p].data(), n), "download cdef");
  }
  compare_planes(cpu, gpu, label);
  CHECK(gpu.lr_guard_intact(), "GPU leaves LrFrame guard columns untouched");
  CHECK(gpu.guard_intact(), "GPU leaves CdefFrame guard columns untouched");
}

// ---------------------------------------------------------------------------
// Synthetic coverage
// ---------------------------------------------------------------------------

void test_synthetic() {
  for (int seed = 1; seed <= 6; ++seed) {
    Rng rng(static_cast<std::uint32_t>(1000 + seed));
    const int mi_cols = 18 + seed % 5;
    const int mi_rows = 10 + seed % 4;
    TestFrame f;
    f.mi_cols = mi_cols;
    f.mi_rows = mi_rows;
    f.allocate();
    av2enc::restoration::test::fill_random(f, rng);

    CcsoFrameParams ccso;
    CcsoHost ch;
    make_ccso(f, 6, seed % 2 == 0, seed % 3 != 0, &ccso, &ch);
    LoopRestorationParams lr;
    LrHost lh;
    make_per_unit_lr(f, 64, 32, &lr, &lh);
    if (seed == 2) {
      for (int p = 0; p < 3; ++p) {
        lr.frame_type[p] = av2enc::restoration::kRestoreSwitchable;
      }
    }
    if (seed == 3) {
      // Lossless/BRU metadata and per-plane segment ids.
      lh.seg.assign(static_cast<std::size_t>(mi_cols * mi_rows), 0);
      for (std::size_t i = 0; i < lh.seg.size(); ++i) {
        lh.seg[i] = static_cast<std::uint8_t>((i / 7) & 1);
      }
      lh.chroma_seg = lh.seg;
      lh.lossless[1] = 1;
      lh.bru.assign(static_cast<std::size_t>(mi_cols * mi_rows),
                    av2enc::restoration::kBruActive);
      for (int y = 0; y < mi_rows / 2; ++y) {
        for (int x = 0; x < mi_cols; ++x) {
          lh.bru[static_cast<std::size_t>(y * mi_cols + x)] =
              av2enc::restoration::kBruInactive;
        }
      }
    }
    if (seed == 4) {
      make_frame_filters(f, 64, &lr, &lh);
    }
    run_case(f, ccso, lr, lh, ch, seed == 3 || seed == 4, "synthetic");
  }
  std::cout << "  synthetic randomized frames: CCSO + LR parity\n";
}

void test_validation() {
  TestFrame f;
  f.mi_cols = 16;
  f.mi_rows = 8;
  f.allocate();
  Rng rng(9);
  av2enc::restoration::test::fill_random(f, rng);
  CcsoFrameParams ccso;
  CcsoHost ch;
  make_ccso(f, 6, false, true, &ccso, &ch);
  LoopRestorationParams lr;
  LrHost lh;
  make_per_unit_lr(f, 64, 32, &lr, &lh);

  DeviceCase d;
  RestorationFrame v{};
  av2enc::cuda::m1::RestorationStageParams st{};
  build_device_case(f, ccso, lr, lh, ch, false, &d, &v, &st);
  CHECK(av2enc::cuda::m1::restoration_stage_validate(v, st) == cudaSuccess,
        "valid stage accepted");

  av2enc::cuda::m1::RestorationStageParams bad = st;
  bad.ccso.offsets_dev[0] = nullptr;
  CHECK(av2enc::cuda::m1::restoration_stage_validate(v, bad) ==
            cudaErrorInvalidValue,
        "missing CCSO offset table rejected");

  bad = st;
  bad.ccso.plane[0].enabled = true;
  bad.ccso.plane[1].enabled = false;
  bad.ccso.plane[2].enabled = false;
  RestorationFrame alias = v;
  alias.curr[0].data = alias.cdef[0].data;
  CHECK(av2enc::cuda::m1::restoration_stage_validate(alias, bad) ==
            cudaErrorInvalidValue,
        "CurrFrame/CdefFrame aliasing with CCSO rejected");

  bad = st;
  bad.lr.frame_type[0] = av2enc::restoration::kRestorePcWiener;
  CHECK(av2enc::cuda::m1::restoration_stage_validate(v, bad) ==
            cudaErrorNotSupported,
        "PC-Wiener rejected");

  bad = st;
  bad.lr.unit_size[0] = 32;
  CHECK(av2enc::cuda::m1::restoration_stage_validate(v, bad) ==
            cudaErrorInvalidValue,
        "non-conforming luma unit size rejected");

  bad = st;
  bad.lr.frame_filters_on[0] = 1;
  bad.lr.num_filter_classes = 2;
  CHECK(av2enc::cuda::m1::restoration_stage_validate(v, bad) ==
            cudaErrorInvalidValue,
        "frame filters without a class map rejected");

  bad = st;
  bad.lr.gdf_frame_enable = 1;
  CHECK(av2enc::cuda::m1::restoration_stage_validate(v, bad) ==
            cudaErrorNotSupported,
        "GDF rejected");
  std::cout << "  validation: offsets, aliasing, PC-Wiener, GDF\n";
}

// ---------------------------------------------------------------------------
// Real reconstruction input
// ---------------------------------------------------------------------------

struct Y4mFrame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y, u, v;
};

bool read_first_y4m(const std::string& path, Y4mFrame* out, std::string* err) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *err = "cannot open " + path;
    return false;
  }
  std::string header;
  std::getline(in, header);
  int w = 0;
  int h = 0;
  const char* p = header.c_str();
  while (*p != '\0') {
    while (*p == ' ') ++p;
    if (*p == 'W') w = std::atoi(p + 1);
    if (*p == 'H') h = std::atoi(p + 1);
    while (*p != '\0' && *p != ' ') ++p;
  }
  if (w <= 0 || h <= 0) {
    *err = "bad Y4M header: " + header;
    return false;
  }
  std::string frame_tag;
  std::getline(in, frame_tag);
  if (frame_tag.rfind("FRAME", 0) != 0) {
    *err = "missing FRAME tag";
    return false;
  }
  const std::size_t cw = static_cast<std::size_t>((w + 1) / 2);
  const std::size_t ch = static_cast<std::size_t>((h + 1) / 2);
  out->width = w;
  out->height = h;
  out->y.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  out->u.resize(cw * ch);
  out->v.resize(cw * ch);
  in.read(reinterpret_cast<char*>(out->y.data()),
          static_cast<std::streamsize>(out->y.size()));
  in.read(reinterpret_cast<char*>(out->u.data()),
          static_cast<std::streamsize>(out->u.size()));
  in.read(reinterpret_cast<char*>(out->v.data()),
          static_cast<std::streamsize>(out->v.size()));
  if (!in) {
    *err = "truncated Y4M frame";
    return false;
  }
  return true;
}

bool decode_ivf(const std::string& ivf, const std::string& dav2d,
                const std::string& out_y4m, std::string* err) {
  const std::string cmd = "\"" + dav2d + "\" -i \"" + ivf + "\" -o \"" +
                          out_y4m + "\" -q --threads 1 >/dev/null 2>&1";
  if (std::system(cmd.c_str()) != 0) {
    *err = "dav2d failed: " + cmd;
    return false;
  }
  return true;
}

void fill_real_frame(const Y4mFrame& src, TestFrame* f) {
  f->mi_cols = src.width / 4;
  f->mi_rows = src.height / 4;
  f->allocate();
  for (int p = 0; p < 3; ++p) {
    const std::vector<std::uint8_t>& s = (p == 0) ? src.y : (p == 1 ? src.u : src.v);
    const int w = f->width[p];
    const int h = f->height[p];
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const std::uint8_t v = s[static_cast<std::size_t>(y * w + x)];
        f->at(f->curr[p], p, y, x) = v;
        f->at(f->cdef[p], p, y, x) = v;
      }
    }
  }
}

// Real deblock -> CDEF -> CCSO -> Wiener chain.  `pre` is the decoder output
// of a no-CDEF M1 stream (post-deblock); `post` is the decoder output of the
// same M1 encode with `--m1-cdef` enabled (post-CDEF).  CCSO reads the
// pre-CDEF luma for its band/edge classifier and writes the post-CDEF plane;
// loop restoration reads curr=post-deblock outside the stripe and cdef=post-
// CCSO inside it.
void fill_real_chain(const Y4mFrame& pre, const Y4mFrame& post, TestFrame* f,
                     std::string* err) {
  if (pre.width != post.width || pre.height != post.height) {
    *err = "pre/post recon dimensions differ";
    return;
  }
  if (pre.width % 4 != 0 || pre.height % 4 != 0) {
    *err = "recon is not MI-aligned";
    return;
  }
  f->mi_cols = pre.width / 4;
  f->mi_rows = pre.height / 4;
  f->allocate();
  for (int p = 0; p < 3; ++p) {
    const std::vector<std::uint8_t>& a =
        (p == 0) ? pre.y : (p == 1 ? pre.u : pre.v);
    const std::vector<std::uint8_t>& b =
        (p == 0) ? post.y : (p == 1 ? post.u : post.v);
    const int w = f->width[p];
    const int h = f->height[p];
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        f->at(f->curr[p], p, y, x) =
            a[static_cast<std::size_t>(y * w + x)];
        f->at(f->cdef[p], p, y, x) =
            b[static_cast<std::size_t>(y * w + x)];
      }
    }
  }
}

void test_real_chain(const Y4mFrame& pre, const Y4mFrame& post,
                     const char* label) {
  TestFrame f;
  std::string err;
  fill_real_chain(pre, post, &f, &err);
  if (!err.empty()) {
    std::fprintf(stderr, "  skip %s: %s\n", label, err.c_str());
    return;
  }

  CcsoFrameParams ccso;
  CcsoHost ch;
  make_ccso(f, 6, false, true, &ccso, &ch);

  LoopRestorationParams lr;
  LrHost lh;
  make_per_unit_lr(f, 64, 32, &lr, &lh);
  char text[256];
  std::snprintf(text, sizeof(text), "%s per-unit", label);
  run_case(f, ccso, lr, lh, ch, false, text);

  LoopRestorationParams lr2;
  LrHost lh2;
  make_frame_filters(f, 64, &lr2, &lh2);
  std::snprintf(text, sizeof(text), "%s frame-filters", label);
  run_case(f, ccso, lr2, lh2, ch, true, text);
}

void test_real_recon(const std::string& ivf, const std::string& dav2d) {
  namespace fs = std::filesystem;
  const fs::path tmp =
      fs::temp_directory_path() /
      ("av2enc_lr_recon_" + std::to_string(::getpid()) + ".y4m");
  std::string err;
  if (!decode_ivf(ivf, dav2d, tmp.string(), &err)) {
    CHECK(false, err.c_str());
    return;
  }
  Y4mFrame src;
  if (!read_first_y4m(tmp.string(), &src, &err)) {
    CHECK(false, err.c_str());
    return;
  }
  fs::remove(tmp);
  if (src.width % 4 != 0 || src.height % 4 != 0) {
    std::fprintf(stderr, "  skip %s: %dx%d is not MI-aligned\n", ivf.c_str(),
                 src.width, src.height);
    return;
  }
  TestFrame f;
  fill_real_frame(src, &f);
  const int luma_unit = 64;
  const int chroma_unit = (src.width >= 128) ? 32 : 32;
  CcsoFrameParams ccso;
  CcsoHost ch;
  make_ccso(f, 6, false, true, &ccso, &ch);
  LoopRestorationParams lr;
  LrHost lh;
  make_per_unit_lr(f, luma_unit, chroma_unit, &lr, &lh);

  char label[256];
  std::snprintf(label, sizeof(label), "%dx%d per-unit", src.width, src.height);
  run_case(f, ccso, lr, lh, ch, false, label);

  LoopRestorationParams lr2;
  LrHost lh2;
  make_frame_filters(f, 64, &lr2, &lh2);
  std::snprintf(label, sizeof(label), "%dx%d frame-filters", src.width,
                src.height);
  run_case(f, ccso, lr2, lh2, ch, true, label);
}

void test_real_chain_ivf(const std::string& pre_ivf,
                          const std::string& post_ivf,
                          const std::string& dav2d) {
  namespace fs = std::filesystem;
  const fs::path pre_y4m =
      fs::temp_directory_path() /
      ("av2enc_lr_pre_" + std::to_string(::getpid()) + ".y4m");
  const fs::path post_y4m =
      fs::temp_directory_path() /
      ("av2enc_lr_post_" + std::to_string(::getpid()) + ".y4m");
  std::string err;
  if (!decode_ivf(pre_ivf, dav2d, pre_y4m.string(), &err) ||
      !decode_ivf(post_ivf, dav2d, post_y4m.string(), &err)) {
    CHECK(false, err.c_str());
    return;
  }
  Y4mFrame pre;
  Y4mFrame post;
  if (!read_first_y4m(pre_y4m.string(), &pre, &err) ||
      !read_first_y4m(post_y4m.string(), &post, &err)) {
    CHECK(false, err.c_str());
    fs::remove(pre_y4m);
    fs::remove(post_y4m);
    return;
  }
  fs::remove(pre_y4m);
  fs::remove(post_y4m);
  char label[256];
  std::snprintf(label, sizeof(label), "%dx%d deblock+CDEF", pre.width,
                pre.height);
  test_real_chain(pre, post, label);
}

void usage() {
  std::cout << "usage: m1_restoration_test [--ivf FILE]... "
            "[--ivf-pair PRE_IVF POST_IVF]... [--dav2d PATH]\n"
            "  --ivf      decoded M1 recon (curr = cdef; CDEF off)\n"
            "  --ivf-pair pre.ivf post.ivf  no-CDEF + CDEF recon of the same\n"
            "             M1 encode; exercises deblock->CDEF->CCSO->Wiener\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> ivfs;
  std::vector<std::pair<std::string, std::string>> ivf_pairs;
  std::string dav2d = "tools/bin/dav2d";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--ivf" && i + 1 < argc) {
      ivfs.push_back(argv[++i]);
    } else if (a == "--ivf-pair" && i + 2 < argc) {
      std::string first = argv[++i];
      std::string second = argv[++i];
      ivf_pairs.emplace_back(std::move(first), std::move(second));
    } else if (a == "--dav2d" && i + 1 < argc) {
      dav2d = argv[++i];
    } else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      usage();
      return 2;
    }
  }

  std::cout << "AV2 M1 post-CDEF restoration — CPU-vs-GPU parity\n";
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    std::fprintf(stderr, "no CUDA device\n");
    return 2;
  }
  std::cout << "  device: " << prop.name << " (sm_" << prop.major << prop.minor
            << ")\n";
  test_validation();
  test_synthetic();
  if (ivfs.empty() && ivf_pairs.empty()) {
    std::cout << "  real recon: SKIP (pass --ivf / --ivf-pair)\n";
  } else {
    for (const std::string& ivf : ivfs) {
      test_real_recon(ivf, dav2d);
    }
    for (const auto& pair : ivf_pairs) {
      test_real_chain_ivf(pair.first, pair.second, dav2d);
    }
  }
  std::cout << (g_failures == 0 ? "PASS" : "FAIL") << ": " << g_checks
            << " checks, " << g_failures << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
