// CPU reference vs independent spec-order oracle on real decoder recon.
//
// This is the non-CUDA half of the parity story: the GPU kernels call the same
// core as the CPU reference, and this test checks the CPU reference on real
// 64x64/640x360 planes against `restoration_test_common.h`'s second, naive
// transcription of §7.19.1/§7.20.  Run it with decoded QP90 IVF frames.
//
// Build:
//   c++ -std=c++20 -O2 -I src \
//       src/cuda/m1/restoration/real_recon_oracle_test.cpp \
//       src/encoder/filters/restoration/restoration.cpp \
//       -o /tmp/real_recon_oracle_test
//   /tmp/real_recon_oracle_test /tmp/recon64.y4m /tmp/recon360.y4m

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "encoder/filters/restoration/restoration.h"
#include "encoder/filters/restoration/restoration_test_common.h"

namespace {

using av2enc::restoration::CcsoFrameParams;
using av2enc::restoration::LoopRestorationParams;
using av2enc::restoration::RestorationFrame;
using av2enc::restoration::RestorationStatus;
using av2enc::restoration::test::TestFrame;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
    }                                                                      \
  } while (0)

struct Y4mFrame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y, u, v;
};

bool read_y4m(const std::string& path, Y4mFrame* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
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
  std::string tag;
  std::getline(in, tag);
  if (w <= 0 || h <= 0 || tag.rfind("FRAME", 0) != 0) return false;
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
  return static_cast<bool>(in);
}

void fill(const Y4mFrame& src, TestFrame* f) {
  f->mi_cols = src.width / 4;
  f->mi_rows = src.height / 4;
  f->allocate();
  for (int p = 0; p < 3; ++p) {
    const std::vector<std::uint8_t>& s =
        (p == 0) ? src.y : (p == 1 ? src.u : src.v);
    for (int y = 0; y < f->height[p]; ++y) {
      for (int x = 0; x < f->width[p]; ++x) {
        const std::uint8_t v = s[static_cast<std::size_t>(y * f->width[p] + x)];
        f->at(f->curr[p], p, y, x) = v;
        f->at(f->cdef[p], p, y, x) = v;
      }
    }
  }
}

void params(const TestFrame& f, int luma_unit, int chroma_unit,
            bool frame_filters, CcsoFrameParams* ccso,
            LoopRestorationParams* lr,
            std::vector<std::uint8_t> type[3],
            std::vector<std::int16_t> coeffs[3],
            std::vector<std::int16_t> frame_coeffs[3],
            std::vector<std::uint8_t>* filter_class,
            std::vector<std::uint8_t>* lookup,
            std::vector<std::uint8_t>* blks) {
  *ccso = CcsoFrameParams{};
  ccso->ccso_luma_size_log2 = 6;
  for (int p = 0; p < 3; ++p) {
    ccso->plane[p].enabled = true;
    ccso->plane[p].scale_idx = 1;
    ccso->plane[p].quant_idx = 2;
    ccso->plane[p].ext_filter = 3;
    ccso->plane[p].edge_clf = 1;
    ccso->plane[p].max_band_log2 = 3;
    const int count = av2enc::restoration::ccso_offset_count(ccso->plane[p]);
    std::vector<std::uint8_t> idx(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
      idx[static_cast<std::size_t>(i)] =
          static_cast<std::uint8_t>((i * 5 + p) & 7);
    }
    CHECK(av2enc::restoration::ccso_set_offsets(ccso->plane[p], idx.data(),
                                                idx.size()),
          "ccso offsets");
    const int sub = (p == 0) ? 0 : 1;
    int cols = 0;
    int rows = 0;
    av2enc::restoration::ccso_blk_grid_size(f.width[p], f.height[p], 6, sub,
                                            sub, &cols, &rows);
    blks[p].assign(static_cast<std::size_t>(cols * rows), 1);
    ccso->blks[p] = blks[p].data();
    ccso->blk_cols[p] = cols;
    ccso->blk_rows[p] = rows;
  }

  *lr = LoopRestorationParams{};
  for (int p = 0; p < 3; ++p) {
    const int us = (p == 0) ? luma_unit : chroma_unit;
    const int sub = (p == 0) ? 0 : 1;
    lr->frame_type[p] = av2enc::restoration::kRestoreWienerNs;
    lr->unit_size[p] = us;
    lr->unit_rows[p] = av2enc::restoration::restoration_count_units(
        us, (f.mi_rows * 4) >> sub);
    lr->unit_cols[p] = av2enc::restoration::restoration_count_units(
        us, (f.mi_cols * 4) >> sub);
    type[p].assign(static_cast<std::size_t>(lr->unit_rows[p] * lr->unit_cols[p]),
                   av2enc::restoration::kRestoreWienerNs);
    lr->unit_type[p] = type[p].data();
    if (frame_filters) {
      lr->frame_filters_on[p] = 1;
      lr->num_filter_classes = 2;
      frame_coeffs[p].assign(2 * av2enc::restoration::kWienerNsCoeffStride, 0);
      for (int c = 0; c < 2; ++c) {
        for (int i = 0; i < 16; ++i) {
          frame_coeffs[p][static_cast<std::size_t>(c * 18 + i)] = static_cast<std::int16_t>(
              (p == 0 ? 9 : 6) - i - c);
        }
      }
      lr->frame_coeffs[p] = frame_coeffs[p].data();
    } else {
      coeffs[p].assign(
          static_cast<std::size_t>(lr->unit_rows[p] * lr->unit_cols[p]) *
              av2enc::restoration::kWienerNsCoeffStride,
          0);
      for (std::size_t u = 0; u < type[p].size(); ++u) {
        for (int i = 0; i < 16; ++i) {
          coeffs[p][u * 18 + static_cast<std::size_t>(i)] = static_cast<std::int16_t>((p == 0 ? 8 : 5) - i);
        }
      }
      lr->unit_coeffs[p] = coeffs[p].data();
    }
  }
  if (frame_filters) {
    filter_class->assign(static_cast<std::size_t>(f.mi_cols * f.mi_rows), 0);
    for (std::size_t i = 0; i < filter_class->size(); ++i) {
      (*filter_class)[i] = static_cast<std::uint8_t>(i & 15);
    }
    lookup->assign(16, 0);
    for (int i = 0; i < 16; ++i) {
      (*lookup)[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i & 1);
    }
    lr->filter_class_grid = filter_class->data();
    lr->subclass_lookup = lookup->data();
  }
}

bool compare(const TestFrame& got, const TestFrame& want, const char* label) {
  long diff = 0;
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < got.height[p]; ++y) {
      for (int x = 0; x < got.width[p]; ++x) {
        if (got.at(got.lr[p], p, y, x) != want.at(want.lr[p], p, y, x)) {
          ++diff;
        }
      }
    }
  }
  CHECK(diff == 0, label);
  return diff == 0;
}

void run_frame(TestFrame& rec, bool frame_filters, const char* label) {
  CcsoFrameParams ccso;
  LoopRestorationParams lr;
  std::vector<std::uint8_t> type[3], blks[3], filter_class, lookup;
  std::vector<std::int16_t> coeffs[3], frame_coeffs[3];
  params(rec, 64, 32, frame_filters, &ccso, &lr, type, coeffs, frame_coeffs,
         &filter_class, &lookup, blks);

  TestFrame got = rec;
  RestorationFrame view = got.view();
  CHECK(av2enc::restoration::ccso_frame(view, ccso) == RestorationStatus::kOk,
        "ccso");
  CHECK(av2enc::restoration::loop_restoration_frame(view, lr) ==
            RestorationStatus::kOk,
        "lr");

  TestFrame want = rec;
  av2enc::restoration::test::naive_ccso(want, ccso);
  av2enc::restoration::test::NaiveLrFrame nv;
  nv.filter_class_grid = lr.filter_class_grid;
  nv.subclass_lookup = lr.subclass_lookup;
  nv.subclass_grid = lr.subclass_grid;
  nv.src = &want;
  nv.params = &lr;
  std::vector<std::uint8_t> out[3];
  for (int p = 0; p < 3; ++p) {
    out[p] = want.lr[p];
  }
  av2enc::restoration::test::naive_loop_restoration(nv, out);
  for (int p = 0; p < 3; ++p) {
    want.lr[p] = out[p];
  }
  compare(got, want, label);
}

void run(const Y4mFrame& src, bool frame_filters) {
  TestFrame rec;
  fill(src, &rec);
  run_frame(rec, frame_filters, frame_filters ? "real recon, frame filters"
                                              : "real recon, per-unit");
}

// Real deblock -> CDEF chain: `pre` is the no-CDEF decode (post-deblock) and
// `post` the CDEF decode of the same M1 encode.  curr = pre, cdef = post.
void run_pair(const Y4mFrame& pre, const Y4mFrame& post) {
  if (pre.width != post.width || pre.height != post.height) {
    std::fprintf(stderr, "pair dimensions differ\n");
    return;
  }
  TestFrame rec;
  rec.mi_cols = pre.width / 4;
  rec.mi_rows = pre.height / 4;
  rec.allocate();
  for (int p = 0; p < 3; ++p) {
    const std::vector<std::uint8_t>& a =
        (p == 0) ? pre.y : (p == 1 ? pre.u : pre.v);
    const std::vector<std::uint8_t>& b =
        (p == 0) ? post.y : (p == 1 ? post.u : post.v);
    for (int y = 0; y < rec.height[p]; ++y) {
      for (int x = 0; x < rec.width[p]; ++x) {
        rec.at(rec.curr[p], p, y, x) =
            a[static_cast<std::size_t>(y * rec.width[p] + x)];
        rec.at(rec.cdef[p], p, y, x) =
            b[static_cast<std::size_t>(y * rec.width[p] + x)];
      }
    }
  }
  run_frame(rec, false, "real deblock+CDEF chain, per-unit");
  run_frame(rec, true, "real deblock+CDEF chain, frame filters");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s recon.y4m [more.y4m ...]  (decoded QP90 frames)\n",
                 argv[0]);
    return 2;
  }
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--pair") {
      if (i + 2 >= argc) {
        std::fprintf(stderr, "--pair needs PRE.y4m POST.y4m\n");
        return 2;
      }
      Y4mFrame pre;
      Y4mFrame post;
      if (!read_y4m(argv[i + 1], &pre) || !read_y4m(argv[i + 2], &post)) {
        std::fprintf(stderr, "cannot read pair %s %s\n", argv[i + 1],
                     argv[i + 2]);
        return 2;
      }
      std::cout << "== pair " << argv[i + 1] << " + " << argv[i + 2] << " "
                << pre.width << "x" << pre.height << "\n";
      run_pair(pre, post);
      i += 2;
      continue;
    }
    Y4mFrame src;
    if (!read_y4m(argv[i], &src)) {
      std::fprintf(stderr, "cannot read %s\n", argv[i]);
      return 2;
    }
    if (src.width % 4 != 0 || src.height % 4 != 0) {
      std::fprintf(stderr, "skip %s: not MI aligned\n", argv[i]);
      continue;
    }
    std::cout << "== " << argv[i] << " " << src.width << "x" << src.height
              << "\n";
    run(src, false);
    run(src, true);
  }
  std::cout << (g_failures == 0 ? "PASS" : "FAIL") << ": " << g_checks
            << " checks, " << g_failures << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
