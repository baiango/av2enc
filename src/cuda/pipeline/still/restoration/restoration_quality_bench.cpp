// Offline quality probe for §7.19 CCSO + §7.20 Wiener NS on real M1 recon.
//
// This is an *encoder-side experiment*, not a bitstream gate.  It reads a
// reference Y4M and the decoder's reconstruction of the same frame (for example
// `dav2d -i s360.ivf -o recon.y4m -l 1`), trains CCSO offsets and frame-level
// Wiener coefficients on that pair, and writes four reconstructions:
//
//   <prefix>.none.y4m    recon unchanged
//   <prefix>.ccso.y4m    CCSO only
//   <prefix>.wiener.y4m  Wiener NS only
//   <prefix>.both.y4m    CCSO then Wiener NS (normative filter order)
//
// It also prints integer PSNR for each.  The coefficients are selected with
// knowledge of the source (an oracle), so treat the delta as a filter-potential
// measurement: syntax/rate cost is not included and no stream is produced.
//
// Build:
//   c++ -std=c++20 -O2 -I src \
//       src/cuda/m1/restoration/restoration_quality_bench.cpp \
//       src/encoder/filters/restoration/restoration.cpp \
//       -o /tmp/restoration_quality

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "codec/filters/loop_restoration/restoration.h"
#include "codec/filters/loop_restoration/restoration_test_common.h"
#include "codec/filters/loop_restoration/video/video_restoration.h"

namespace {

using av2enc::restoration::CcsoFrameParams;
using av2enc::restoration::CcsoParams;
using av2enc::restoration::LoopRestorationParams;
using av2enc::restoration::RestorationFrame;
using av2enc::restoration::RestorationStatus;
using av2enc::restoration::test::TestFrame;

struct Y4mFrame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y, u, v;
};

bool read_first_y4m(const std::string& path, Y4mFrame* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
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
    std::fprintf(stderr, "bad Y4M header: %s\n", header.c_str());
    return false;
  }
  std::string tag;
  std::getline(in, tag);
  if (tag.rfind("FRAME", 0) != 0) {
    std::fprintf(stderr, "missing FRAME tag in %s\n", path.c_str());
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
  return static_cast<bool>(in);
}

bool write_y4m(const std::string& path, const TestFrame& f, int w, int h) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  out << "YUV4MPEG2 W" << w << " H" << h << " F1:1 Ip A1:1 C420\nFRAME\n";
  for (int y = 0; y < h; ++y) {
    out.write(reinterpret_cast<const char*>(f.lr[0].data() +
                                            static_cast<std::size_t>(y) *
                                                static_cast<std::size_t>(f.stride[0])),
              w);
  }
  const int cw = (w + 1) / 2;
  const int ch = (h + 1) / 2;
  for (int y = 0; y < ch; ++y) {
    out.write(reinterpret_cast<const char*>(f.lr[1].data() +
                                            static_cast<std::size_t>(y) *
                                                static_cast<std::size_t>(f.stride[1])),
              cw);
  }
  for (int y = 0; y < ch; ++y) {
    out.write(reinterpret_cast<const char*>(f.lr[2].data() +
                                            static_cast<std::size_t>(y) *
                                                static_cast<std::size_t>(f.stride[2])),
              cw);
  }
  return static_cast<bool>(out);
}

void fill_test_frame(const Y4mFrame& src, TestFrame* f) {  f->mi_cols = src.width / 4;
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

double psnr_plane(const TestFrame& ref, const TestFrame& out, int p) {  double sse = 0;
  for (int y = 0; y < ref.height[p]; ++y) {
    for (int x = 0; x < ref.width[p]; ++x) {
      const double d = static_cast<double>(ref.at(ref.curr[p], p, y, x)) -
                       static_cast<double>(out.at(out.lr[p], p, y, x));
      sse += d * d;
    }
  }
  const double n = static_cast<double>(ref.width[p] * ref.height[p]);
  if (sse <= 0) {
    return 99.0;
  }
  return 10.0 * std::log10(255.0 * 255.0 * n / sse);
}

struct Psnr {
  double y = 0;
  double u = 0;
  double v = 0;
  double avg = 0;
};

Psnr measure(const TestFrame& ref, const TestFrame& out) {
  Psnr r;
  r.y = psnr_plane(ref, out, 0);
  r.u = psnr_plane(ref, out, 1);
  r.v = psnr_plane(ref, out, 2);
  const double ny = static_cast<double>(ref.width[0] * ref.height[0]);
  const double nc = static_cast<double>(ref.width[1] * ref.height[1]);
  r.avg = (r.y * ny + r.u * nc + r.v * nc) / (ny + 2 * nc);
  return r;
}

void sync_lr_from_cdef(TestFrame* f) {
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f->height[p]; ++y) {
      for (int x = 0; x < f->width[p]; ++x) {
        f->at(f->lr[p], p, y, x) = f->at(f->cdef[p], p, y, x);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// CCSO training
// ---------------------------------------------------------------------------

struct CcsoFit {
  CcsoParams params;
  std::vector<std::uint8_t> idx;
  double sse = 0;
  int signaled_offsets = 0;
};

int luma_at(const TestFrame& f, int x, int y) {
  const int cx = std::max(0, std::min(f.width[0] - 1, x));
  const int cy = std::max(0, std::min(f.height[0] - 1, y));
  return f.at(f.curr[0], 0, cy, cx);
}

CcsoFit fit_ccso_plane(const TestFrame& ref, const TestFrame& rec, int plane,
                       bool bo_only, int scale_idx, int quant_idx,
                       int ext_filter, int edge_clf, int max_band_log2) {
  CcsoFit fit;
  CcsoParams& p = fit.params;
  p.enabled = true;
  p.bo_only = bo_only;
  p.scale_idx = scale_idx;
  p.quant_idx = bo_only ? 0 : quant_idx;
  p.ext_filter = bo_only ? 0 : ext_filter;
  p.edge_clf = bo_only ? 0 : edge_clf;
  p.max_band_log2 = max_band_log2;

  const int sub = (plane == 0) ? 0 : 1;
  const int interval =
      bo_only ? 1 : (av2enc::restoration::kCcsoInputInterval - edge_clf);
  const int bands = 1 << max_band_log2;
  const int quant_step =
      av2enc::restoration::kCcsoQuantSz[scale_idx][p.quant_idx];
  const int scale = scale_idx + 1;
  const int ext = bo_only ? 0 : ext_filter;
  const int dy = av2enc::restoration::kCcsoPos[ext][0];
  const int dx = av2enc::restoration::kCcsoPos[ext][1];
  const int shift = 8 - max_band_log2;

  const std::size_t ncell =
      static_cast<std::size_t>(bands * interval * interval);
  std::vector<double> sum_d(ncell, 0.0);
  std::vector<double> sum_d2(ncell, 0.0);
  std::vector<double> count(ncell, 0.0);

  for (int y = 0; y < rec.height[plane]; ++y) {
    for (int x = 0; x < rec.width[plane]; ++x) {
      const int lx = x << sub;
      const int ly = y << sub;
      const int c = luma_at(rec, lx, ly);
      const int band = c >> shift;
      int cls0 = 0;
      int cls1 = 0;
      if (!bo_only) {
        cls0 = av2enc::restoration::ccso_score(
            luma_at(rec, lx + dx, ly + dy) - c, quant_step, edge_clf);
        cls1 = av2enc::restoration::ccso_score(
            luma_at(rec, lx - dx, ly - dy) - c, quant_step, edge_clf);
        if (cls0 >= interval) cls0 = 0;
        if (cls1 >= interval) cls1 = 0;
      }
      const std::size_t cell =
          static_cast<std::size_t>((band * interval + cls0) * interval + cls1);
      const double d = static_cast<double>(ref.at(ref.curr[plane], plane, y, x)) -
                       static_cast<double>(rec.at(rec.cdef[plane], plane, y, x));
      sum_d[cell] += d;
      sum_d2[cell] += d * d;
      count[cell] += 1.0;
    }
  }

  fit.idx.assign(static_cast<std::size_t>(bands * interval * interval), 0);
  double total = 0;
  for (int d0 = 0; d0 < interval; ++d0) {
    for (int d1 = 0; d1 < interval; ++d1) {
      for (int band = 0; band < bands; ++band) {
        const std::size_t cell =
            static_cast<std::size_t>((band * interval + d0) * interval + d1);
        double best = sum_d2[cell];
        int best_i = 0;
        for (int i = 0; i < av2enc::restoration::kCcsoOffsetRange; ++i) {
          const double off =
              static_cast<double>(av2enc::restoration::kCcsoOffset[i] * scale);
          const double sse = sum_d2[cell] - 2.0 * off * sum_d[cell] +
                             off * off * count[cell];
          if (sse < best - 1e-9) {
            best = sse;
            best_i = i;
          }
        }
        fit.idx[static_cast<std::size_t>((d0 * interval + d1) * bands + band)] =
            static_cast<std::uint8_t>(best_i);
        total += best;
      }
    }
  }
  fit.sse = total;
  fit.signaled_offsets = static_cast<int>(fit.idx.size());
  if (!av2enc::restoration::ccso_set_offsets(p, fit.idx.data(),
                                             fit.idx.size())) {
    std::fprintf(stderr, "ccso_set_offsets failed\n");
  }
  return fit;
}

CcsoFit fit_ccso_plane_best(const TestFrame& ref, const TestFrame& rec,
                            int plane, int max_band_log2) {
  CcsoFit best;
  best.sse = 1e300;
  for (int bo = 0; bo <= 1; ++bo) {
    for (int scale = 0; scale < 4; ++scale) {
      if (bo) {
        CcsoFit f = fit_ccso_plane(ref, rec, plane, true, scale, 0, 0, 0,
                                   max_band_log2);
        if (f.sse < best.sse) best = f;
      } else {
        for (int quant = 0; quant < 4; ++quant) {
          const int quant_step =
              av2enc::restoration::kCcsoQuantSz[scale][quant];
          for (int ext = 0; ext < 7; ++ext) {
            for (int edge = 0; edge <= 1; ++edge) {
              if (quant_step == 0 && edge != 0) {
                continue;  // §5.18.7.12 forces edge_clf = 0
              }
              CcsoFit f = fit_ccso_plane(ref, rec, plane, false, scale, quant,
                                         ext, edge, max_band_log2);
              if (f.sse < best.sse) best = f;
            }
          }
        }
      }
    }
  }
  return best;
}

void apply_ccso(const TestFrame& rec, const CcsoFit fit[3],
                CcsoFrameParams* params) {
  *params = CcsoFrameParams{};
  params->ccso_luma_size_log2 = 6;
  for (int p = 0; p < 3; ++p) {
    params->plane[p] = fit[p].params;
    params->plane[p].enabled = true;
  }
  // Build per-plane block grids, all enabled.
  static std::vector<std::uint8_t> blks[3];
  for (int p = 0; p < 3; ++p) {
    const int sub = (p == 0) ? 0 : 1;
    int cols = 0;
    int rows = 0;
    av2enc::restoration::ccso_blk_grid_size(rec.width[p], rec.height[p], 6,
                                            sub, sub, &cols, &rows);
    blks[p].assign(static_cast<std::size_t>(cols * rows), 1);
    params->blks[p] = blks[p].data();
    params->blk_cols[p] = cols;
    params->blk_rows[p] = rows;
  }
}

// ---------------------------------------------------------------------------
// Wiener NS training
// ---------------------------------------------------------------------------

struct LrFit {
  std::vector<std::int16_t> coeffs[3];
};

int source_sample(const TestFrame& f, const av2enc::restoration::LrBlock& block,
                  int plane, int x, int y) {
  const int sub_x = (plane == 0) ? 0 : f.sub_x;
  const int sub_y = (plane == 0) ? 0 : f.sub_y;
  x = std::max(block.luma_start_x >> sub_x, std::min(block.luma_end_x >> sub_x,
                                                     x));
  y = std::max(block.luma_start_y >> sub_y, std::min(block.luma_end_y >> sub_y,
                                                     y));
  const int stripe_start = block.stripe_start_y >> sub_y;
  const int stripe_end = block.stripe_end_y >> sub_y;
  if (y < stripe_start) {
    const int yy = std::max(stripe_start - 2, y);
    return f.at(f.curr[plane], plane, yy, x);
  }
  if (y > stripe_end) {
    const int yy = std::min(stripe_end + 2, y);
    return f.at(f.curr[plane], plane, yy, x);
  }
  return f.at(f.cdef[plane], plane, y, x);
}

int luma_source(const TestFrame& f, const av2enc::restoration::LrBlock& block,
                int x, int y) {
  const int last_y = f.mi_rows * 4 - 1 - f.sub_y;
  const int last_x = block.luma_end_x - f.sub_x;
  x <<= f.sub_x;
  y <<= f.sub_y;
  y = std::max(0, std::min(last_y, y));
  x = std::max(block.luma_start_x, std::min(last_x, x));
  int filter_idx = 0;  // cfl_ds_filter_index; §7.20.3 default in this probe
  if (filter_idx == 3) filter_idx = 0;
  if (f.sub_x && f.sub_y && filter_idx <= 1) {
    int t = 0;
    for (int dy = 0; dy < 2; ++dy) {
      for (int dx = 0; dx < 2; ++dx) {
        t += static_cast<int>(
                 av2enc::restoration::kWienerFilters420[filter_idx][dy][dx]) *
             source_sample(f, block, 0, x + dx, y + dy);
      }
    }
    return t >> 2;
  }
  return source_sample(f, block, 0, x, y);
}

void solve_linear(std::vector<double> a, std::vector<double> b, int n,
                  std::vector<double>* out) {
  for (int i = 0; i < n; ++i) {
    a[static_cast<std::size_t>(i * n + i)] += 1e-3;
  }
  for (int col = 0; col < n; ++col) {
    int piv = col;
    for (int r = col + 1; r < n; ++r) {
      if (std::fabs(a[static_cast<std::size_t>(r * n + col)]) >
          std::fabs(a[static_cast<std::size_t>(piv * n + col)])) {
        piv = r;
      }
    }
    if (std::fabs(a[static_cast<std::size_t>(piv * n + col)]) < 1e-12) {
      continue;
    }
    if (piv != col) {
      for (int c = 0; c < n; ++c) {
        std::swap(a[static_cast<std::size_t>(col * n + c)],
                  a[static_cast<std::size_t>(piv * n + c)]);
      }
      std::swap(b[static_cast<std::size_t>(col)],
                b[static_cast<std::size_t>(piv)]);
    }
    const double diag = a[static_cast<std::size_t>(col * n + col)];
    for (int c = col; c < n; ++c) {
      a[static_cast<std::size_t>(col * n + c)] /= diag;
    }
    b[static_cast<std::size_t>(col)] /= diag;
    for (int r = 0; r < n; ++r) {
      if (r == col) continue;
      const double factor = a[static_cast<std::size_t>(r * n + col)];
      if (factor == 0.0) continue;
      for (int c = col; c < n; ++c) {
        a[static_cast<std::size_t>(r * n + c)] -=
            factor * a[static_cast<std::size_t>(col * n + c)];
      }
      b[static_cast<std::size_t>(r)] -=
          factor * b[static_cast<std::size_t>(col)];
    }
  }
  *out = b;
}

LrFit fit_wiener(const TestFrame& ref, const TestFrame& lr_in) {
  LrFit fit;
  TestFrame f = lr_in;  // mutable copy for view()/at()
  f.sub_x = 1;
  f.sub_y = 1;
  LoopRestorationParams params{};
  for (int p = 0; p < 3; ++p) {
    params.frame_type[p] = av2enc::restoration::kRestoreWienerNs;
    params.unit_size[p] = (p == 0) ? 64 : 32;
    const int sub = (p == 0) ? 0 : 1;
    params.unit_rows[p] =
        av2enc::restoration::restoration_count_units(params.unit_size[p],
                                                     (f.mi_rows * 4) >> sub);
    params.unit_cols[p] =
        av2enc::restoration::restoration_count_units(params.unit_size[p],
                                                     (f.mi_cols * 4) >> sub);
    static std::vector<std::uint8_t> type[3];
    type[p].assign(static_cast<std::size_t>(params.unit_rows[p] *
                                            params.unit_cols[p]),
                   av2enc::restoration::kRestoreWienerNs);
    params.unit_type[p] = type[p].data();
    params.frame_filters_on[p] = 1;
    params.num_filter_classes = 1;
  }
  for (int plane = 0; plane < 3; ++plane) {
    const int n = (plane == 0) ? 16 : 18;
    std::vector<double> ata(static_cast<std::size_t>(n * n), 0.0);
    std::vector<double> atb(static_cast<std::size_t>(n), 0.0);
    const int sub_x = (plane == 0) ? 0 : 1;
    const int sub_y = (plane == 0) ? 0 : 1;
    for (int row = 0; row < f.mi_rows; ++row) {
      for (int col = 0; col < f.mi_cols; ++col) {
        av2enc::restoration::LrBlock block{};
        RestorationFrame view = f.view();
        if (!av2enc::restoration::lr_block_context(
                av2enc::restoration::loop_restoration_to_core(view, params),
                plane, row, col, &block)) {
          continue;
        }
        const int x0 = (col * 4) >> sub_x;
        const int y0 = (row * 4) >> sub_y;
        const int w = 4 >> sub_x;
        const int h = 4 >> sub_y;
        const int8_t(*cfg)[3] =
            (plane == 0) ? av2enc::restoration::kWienerNsConfigY
                         : av2enc::restoration::kWienerNsConfigUv;
        const int taps = (plane == 0) ? 32 : 12;
        for (int r = 0; r < h; ++r) {
          for (int c = 0; c < w; ++c) {
            const int px = x0 + c;
            const int py = y0 + r;
            const int m = source_sample(f, block, plane, px, py);
            const double target = 128.0 *
                (static_cast<double>(ref.at(ref.curr[plane], plane, py, px)) -
                 static_cast<double>(m));
            std::vector<double> feat(static_cast<std::size_t>(n), 0.0);
            for (int i = 0; i < taps; ++i) {
              const int idx = cfg[i][2];
              feat[static_cast<std::size_t>(idx)] += static_cast<double>(
                  source_sample(f, block, plane, px + cfg[i][1],
                                py + cfg[i][0]) -
                  m);
            }
            if (plane > 0) {
              const int ml = luma_source(f, block, px, py);
              for (int i = 0; i < 12; ++i) {
                feat[static_cast<std::size_t>(i + 6)] = static_cast<double>(
                    luma_source(f, block, px + cfg[i][1], py + cfg[i][0]) -
                    ml);
              }
            }
            for (int i = 0; i < n; ++i) {
              for (int j = 0; j < n; ++j) {
                ata[static_cast<std::size_t>(i * n + j)] +=
                    feat[static_cast<std::size_t>(i)] *
                    feat[static_cast<std::size_t>(j)];
              }
              atb[static_cast<std::size_t>(i)] +=
                  feat[static_cast<std::size_t>(i)] * target;
            }
          }
        }
      }
    }
    std::vector<double> sol;
    solve_linear(ata, atb, n, &sol);
    fit.coeffs[plane].assign(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
      int v = static_cast<int>(std::lround(sol[static_cast<std::size_t>(i)]));
      v = av2enc::restoration::wiener_ns_clip_coeff(plane > 0, i, v);
      fit.coeffs[plane][static_cast<std::size_t>(i)] =
          static_cast<std::int16_t>(v);
    }
  }
  return fit;
}

void make_lr_params(const TestFrame& f, const LrFit& fit,
                    LoopRestorationParams* p,
                    std::vector<std::uint8_t>* type[3]) {
  *p = LoopRestorationParams{};
  for (int plane = 0; plane < 3; ++plane) {
    p->frame_type[plane] = av2enc::restoration::kRestoreWienerNs;
    p->unit_size[plane] = (plane == 0) ? 64 : 32;
    const int sub = (plane == 0) ? 0 : 1;
    p->unit_rows[plane] = av2enc::restoration::restoration_count_units(
        p->unit_size[plane], (f.mi_rows * 4) >> sub);
    p->unit_cols[plane] = av2enc::restoration::restoration_count_units(
        p->unit_size[plane], (f.mi_cols * 4) >> sub);
    type[plane]->assign(
        static_cast<std::size_t>(p->unit_rows[plane] * p->unit_cols[plane]),
        av2enc::restoration::kRestoreWienerNs);
    p->unit_type[plane] = type[plane]->data();
    p->frame_filters_on[plane] = 1;
    p->num_filter_classes = 1;
    p->frame_coeffs[plane] = fit.coeffs[plane].data();
  }
}

void apply_wiener(const TestFrame& base, const LrFit& fit, TestFrame* out) {
  *out = base;
  for (int p = 0; p < 3; ++p) {
    std::fill(out->lr[p].begin(), out->lr[p].end(),
              av2enc::restoration::test::kGuard);
  }
  std::vector<std::uint8_t> type[3];
  LoopRestorationParams params{};
  std::vector<std::uint8_t>* type_ptr[3] = {&type[0], &type[1], &type[2]};
  make_lr_params(base, fit, &params, type_ptr);
  RestorationFrame view = out->view();
  const RestorationStatus st =
      av2enc::restoration::loop_restoration_frame(view, params);
  if (st != RestorationStatus::kOk) {
    std::fprintf(stderr, "wiener apply failed: %s\n",
                 av2enc::restoration::restoration_status_string(st));
  }
}

// ---------------------------------------------------------------------------
// Exact syntax cost of the fitted tools
// ---------------------------------------------------------------------------

void report_cost(const TestFrame& rec, const CcsoFit ccso_fit[3],
                 const LrFit& lr_fit) {
  namespace video_restoration = av2enc::video::restoration;
  namespace fs = av2enc::still::filters_stream;
  const int mi_cols = rec.mi_cols;
  const int mi_rows = rec.mi_rows;
  std::string error;

  fs::CcsoParams ccso;
  ccso.frame_flag = true;
  ccso.ccso_luma_size_log2 = 6;
  for (int p = 0; p < 3; ++p) {
    fs::CcsoPlaneParams& q = ccso.plane[p];
    q.enabled = ccso_fit[p].params.enabled;
    q.bo_only = ccso_fit[p].params.bo_only;
    q.scale_idx = ccso_fit[p].params.scale_idx;
    q.quant_idx = ccso_fit[p].params.quant_idx;
    q.ext_filter = ccso_fit[p].params.ext_filter;
    q.edge_clf = ccso_fit[p].params.edge_clf;
    q.max_band_log2 = ccso_fit[p].params.max_band_log2;
    q.offset_idx = ccso_fit[p].idx;
  }
  fs::CcsoTileState ccso_state;
  if (!video_restoration::init_ccso_tile(&ccso_state, ccso, mi_cols, mi_rows, 1, 1,
                           &error)) {
    std::printf("  cost: CCSO state init failed: %s\n", error.c_str());
    return;
  }

  fs::LrParams lr;
  for (int p = 0; p < 3; ++p) {
    lr.frame_type[p] = av2enc::restoration::kRestoreWienerNs;
    lr.frame_filters_on[p] = true;
    lr.frame_filter[p][0].match_index = 0;
    lr.frame_filter[p][0].merged = false;
    lr.frame_filter[p][0].subset = (p == 0) ? 3 : 2;
    const std::size_t n = lr_fit.coeffs[p].size();
    for (std::size_t j = 0; j < n; ++j) {
      lr.frame_filter[p][0].coeffs[j] = lr_fit.coeffs[p][j];
    }
    for (std::size_t j = n; j < av2enc::restoration::kWienerNsCoeffStride;
         ++j) {
      lr.frame_filter[p][0].coeffs[j] = 0;
    }
    if (p > 0 && lr.frame_filter[p][0].subset > 0) {
      // §5.20.10.6 uv_sym pairs: make the chroma-from-luma pairs symmetric so
      // the writer can code the shortest subset.  The fitted values are
      // averaged, which is what an encoder-side fitter should do anyway.
      for (std::size_t j = av2enc::restoration::kWienerNsShortCoeffs;
           j + 1 < n; j += 2) {
        const int v = (lr.frame_filter[p][0].coeffs[j] +
                       lr.frame_filter[p][0].coeffs[j + 1]) /
                      2;
        lr.frame_filter[p][0].coeffs[j] = static_cast<std::int16_t>(v);
        lr.frame_filter[p][0].coeffs[j + 1] = static_cast<std::int16_t>(v);
      }
      lr.frame_filter[p][0].uv_sym = true;
    }
  }
  lr.num_filter_classes = 1;
  lr.luma_shift = 3;
  lr.chroma_shift = 3;
  fs::LrState lr_state;
  if (!video_restoration::init_lr_tile(&lr_state, lr, mi_cols, mi_rows, 1, 1, 64,
                         &error)) {
    std::printf("  cost: LR state init failed: %s\n", error.c_str());
    return;
  }

  video_restoration::VideoRestorationTools ccso_tools;
  ccso_tools.ccso = true;
  video_restoration::VideoRestorationTools lr_tools;
  lr_tools.wiener_ns = true;
  lr_tools.frame_filters = true;
  lr_tools.num_filter_classes = 1;
  lr_tools.chroma_lr = true;

  video_restoration::VideoRestorationCost ccso_cost;
  if (!video_restoration::measure_restoration_cost(
          ccso_tools, mi_cols, mi_rows, 64, 90, true, 0, &ccso, &ccso_state,
          nullptr, &ccso_cost, &error)) {
    std::printf("  cost: CCSO measure failed: %s\n", error.c_str());
    return;
  }
  video_restoration::VideoRestorationCost lr_cost;
  if (!video_restoration::measure_restoration_cost(lr_tools, mi_cols, mi_rows, 64, 90, true,
                                     0, nullptr, nullptr, &lr_state, &lr_cost,
                                     &error)) {
    std::printf("  cost: LR measure failed: %s\n", error.c_str());
    return;
  }
  const std::size_t ccso_b = (ccso_cost.total_bits() + 7) / 8;
  const std::size_t lr_b = (lr_cost.total_bits() + 7) / 8;
  std::printf(
      "  cost CCSO: seq %zu b hdr %zu b sym %zu B total %zu B\n",
      ccso_cost.sequence_bits, ccso_cost.ccso_header_bits,
      ccso_cost.ccso_symbol_bytes, ccso_b);
  std::printf(
      "  cost LR  : seq %zu b hdr %zu b sym %zu B total %zu B\n",
      lr_cost.sequence_bits, lr_cost.lr_header_bits,
      lr_cost.lr_symbol_bytes, lr_b);
  std::printf("  cost both: %zu B (sum of the two independent rows)\n",
              ccso_b + lr_b);
}

}  // namespace

int main(int argc, char** argv) {
  std::string ref_path;
  std::string recon_path;
  std::string recon_pre_path;
  std::string prefix = "/tmp/lrq";
  int max_band_log2 = 6;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--ref" && i + 1 < argc) {
      ref_path = argv[++i];
    } else if (a == "--recon" && i + 1 < argc) {
      recon_path = argv[++i];
    } else if (a == "--recon-pre" && i + 1 < argc) {
      recon_pre_path = argv[++i];
    } else if (a == "--out-prefix" && i + 1 < argc) {
      prefix = argv[++i];
    } else if (a == "--max-band-log2" && i + 1 < argc) {
      max_band_log2 = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr,
                   "usage: %s --ref src.y4m --recon recon.y4m "
                   "[--recon-pre pre_cdef.y4m] [--out-prefix P] "
                   "[--max-band-log2 N]\n",
                   argv[0]);
      return 2;
    }
  }
  if (ref_path.empty() || recon_path.empty()) {
    std::fprintf(stderr, "need --ref and --recon\n");
    return 2;
  }
  Y4mFrame ref_y4m;
  Y4mFrame rec_y4m;
  if (!read_first_y4m(ref_path, &ref_y4m) ||
      !read_first_y4m(recon_path, &rec_y4m)) {
    return 2;
  }
  if (ref_y4m.width != rec_y4m.width || ref_y4m.height != rec_y4m.height) {
    std::fprintf(stderr, "dimension mismatch\n");
    return 2;
  }
  if (ref_y4m.width % 4 != 0 || ref_y4m.height % 4 != 0) {
    std::fprintf(stderr, "input must be 4-aligned\n");
    return 2;
  }
  TestFrame ref;
  TestFrame rec;
  fill_test_frame(ref_y4m, &ref);
  fill_test_frame(rec_y4m, &rec);
  rec.sub_x = 1;
  rec.sub_y = 1;
  if (!recon_pre_path.empty()) {
    Y4mFrame pre_y4m;
    if (!read_first_y4m(recon_pre_path, &pre_y4m) ||
        pre_y4m.width != rec_y4m.width ||
        pre_y4m.height != rec_y4m.height) {
      std::fprintf(stderr, "cannot read matching --recon-pre\n");
      return 2;
    }
    TestFrame pre;
    fill_test_frame(pre_y4m, &pre);
    // CCSO and §7.20 source pre-CDEF samples from CurrFrame.  CDEF already
    // ran, so curr = the no-CDEF decode and cdef = the CDEF decode.
    for (int p = 0; p < 3; ++p) {
      rec.curr[p] = pre.cdef[p];
    }
  }
  sync_lr_from_cdef(&rec);

  const Psnr base_psnr = measure(ref, rec);
  std::cout << "size " << ref_y4m.width << "x" << ref_y4m.height << "\n";
  std::cout << "none   Y " << base_psnr.y << "  U " << base_psnr.u << "  V "
            << base_psnr.v << "  avg " << base_psnr.avg << "\n";
  write_y4m(prefix + ".none.y4m", rec, ref_y4m.width, ref_y4m.height);

  // CCSO: train each plane independently.
  CcsoFit ccso_fit[3];
  for (int p = 0; p < 3; ++p) {
    ccso_fit[p] = fit_ccso_plane_best(ref, rec, p, max_band_log2);
  }
  CcsoFrameParams ccso_params;
  apply_ccso(rec, ccso_fit, &ccso_params);
  TestFrame rec_ccso = rec;
  RestorationFrame rec_ccso_view = rec_ccso.view();
  // curr must stay pre-CCSO for the edge classifier.
  const RestorationStatus ccso_st =
      av2enc::restoration::ccso_frame(rec_ccso_view, ccso_params);
  if (ccso_st != RestorationStatus::kOk) {
    std::fprintf(stderr, "ccso apply failed: %s\n",
                 av2enc::restoration::restoration_status_string(ccso_st));
    return 1;
  }
  sync_lr_from_cdef(&rec_ccso);
  const Psnr ccso_psnr = measure(ref, rec_ccso);
  std::cout << "ccso   Y " << ccso_psnr.y << "  U " << ccso_psnr.u << "  V "
            << ccso_psnr.v << "  avg " << ccso_psnr.avg << "\n";
  for (int p = 0; p < 3; ++p) {
    std::cout << "  plane " << p << ": bo_only "
              << (ccso_fit[p].params.bo_only ? 1 : 0) << " scale "
              << ccso_fit[p].params.scale_idx << " quant "
              << ccso_fit[p].params.quant_idx << " ext "
              << ccso_fit[p].params.ext_filter << " edge "
              << ccso_fit[p].params.edge_clf << " offsets "
              << ccso_fit[p].signaled_offsets;
    int nonzero = 0;
    for (std::uint8_t v : ccso_fit[p].idx) {
      if (v != 0) ++nonzero;
    }
    std::cout << " nonzero " << nonzero << "\n";
  }
  write_y4m(prefix + ".ccso.y4m", rec_ccso, ref_y4m.width, ref_y4m.height);

  // Wiener only.
  const LrFit wiener_fit = fit_wiener(ref, rec);
  TestFrame rec_wiener;
  apply_wiener(rec, wiener_fit, &rec_wiener);
  const Psnr wiener_psnr = measure(ref, rec_wiener);
  std::cout << "wiener Y " << wiener_psnr.y << "  U " << wiener_psnr.u << "  V "
            << wiener_psnr.v << "  avg " << wiener_psnr.avg << "\n";
  write_y4m(prefix + ".wiener.y4m", rec_wiener, ref_y4m.width, ref_y4m.height);

  // CCSO then Wiener (normative order).  Fit on the CCSO output, but source
  // outside the stripe from the pre-CCSO recon.
  const LrFit both_fit = fit_wiener(ref, rec_ccso);
  TestFrame both;
  apply_wiener(rec_ccso, both_fit, &both);
  const Psnr both_psnr = measure(ref, both);
  std::cout << "both   Y " << both_psnr.y << "  U " << both_psnr.u << "  V "
            << both_psnr.v << "  avg " << both_psnr.avg << "\n";
  write_y4m(prefix + ".both.y4m", both, ref_y4m.width, ref_y4m.height);

  for (int p = 0; p < 3; ++p) {
    std::cout << "  wiener plane " << p << " coeffs:";
    for (std::size_t i = 0; i < wiener_fit.coeffs[p].size(); ++i) {
      std::cout << " " << wiener_fit.coeffs[p][i];
    }
    std::cout << "\n";
  }
  report_cost(rec, ccso_fit, both_fit);
  return 0;
}
