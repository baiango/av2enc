// CPU reference tests for §7.21.7 film grain synthesis.
//
// The oracle is a second, independent transcription of the pseudocode and
// shares only the normative Gaussian_Sequence table with film_grain.cpp.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "encoder/filters/spec_tables/film_grain.h"
#include "encoder/filters/spec_tables/film_grain_tables.h"
#include "encoder/filters/spec_tables/spec_test_common.h"

namespace {

using namespace av2enc::filters::spec;
using namespace av2enc::filters::spec::test;
using namespace av2enc::filters::spec_tables;

int OClip3(int lo, int hi, int v) { return v < lo ? lo : (v > hi ? hi : v); }
std::int64_t ORound2(std::int64_t x, int n) {
  return n == 0 ? x : (x + (std::int64_t{1} << (n - 1))) >> n;
}

class ORandom {
 public:
  explicit ORandom(std::uint16_t seed) : r_(seed) {}
  void Set(std::uint16_t v) { r_ = v; }
  void Xor(std::uint16_t v) { r_ = static_cast<std::uint16_t>(r_ ^ v); }
  int Get(int bits) {
    const unsigned int r = r_;
    const unsigned int bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1u;
    const unsigned int next = (r >> 1) | (bit << 15);
    const int out = static_cast<int>((next >> (16 - bits)) & ((1u << bits) - 1u));
    r_ = static_cast<std::uint16_t>(next);
    return out;
  }

 private:
  std::uint16_t r_;
};

using Grid = std::vector<std::vector<int>>;

Grid MakeGrid(int w, int h) { return Grid(h, std::vector<int>(w, 0)); }

struct OGrain {
  Grid y;
  Grid u;
  Grid v;
  OGrain() : y(MakeGrid(82, 73)), u(MakeGrid(82, 73)), v(MakeGrid(82, 73)) {}
};

void OracleGenerate(const FilmGrainModel& m, std::uint16_t seed, int sub_x,
                    int sub_y, int bd, OGrain* g) {
  const int gmin = -(1 << (bd - 1));
  const int gmax = (1 << (bd - 1)) - 1;
  const int nshift = 12 - bd + m.grain_scale_shift;
  const int ashift = m.ar_coeff_shift_minus_6 + 6;

  ORandom rng(seed);
  for (int y = 0; y < 73; ++y) {
    for (int x = 0; x < 82; ++x) {
      g->y[y][x] = static_cast<int>(
          ORound2(m.num_y_points > 0 ? kGaussianSequence[rng.Get(11)] : 0, nshift));
    }
  }
  for (int y = 3; y < 73; ++y) {
    for (int x = 3; x < 79; ++x) {
      std::int64_t s = 0;
      int pos = 0;
      for (int dr = -m.ar_coeff_lag; dr <= 0; ++dr) {
        for (int dc = -m.ar_coeff_lag; dc <= m.ar_coeff_lag; ++dc) {
          if (dr == 0 && dc == 0) break;
          s += static_cast<std::int64_t>(g->y[y + dr][x + dc]) * m.ar_coeffs_y[pos++];
        }
      }
      g->y[y][x] = OClip3(gmin, gmax, g->y[y][x] + static_cast<int>(ORound2(s, ashift)));
    }
  }

  const int cw = sub_x ? 44 : 82;
  const int ch = sub_y ? 38 : 73;
  const int cshift = 12 - bd + m.grain_scale_shift;

  rng.Set(seed);
  rng.Xor(0xb524);
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      g->u[y][x] = static_cast<int>(
          ORound2((m.num_cb_points > 0 || m.chroma_scaling_from_luma)
                      ? kGaussianSequence[rng.Get(11)]
                      : 0,
                  cshift));
    }
  }
  rng.Set(seed);
  rng.Xor(0x49d8);
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      g->v[y][x] = static_cast<int>(
          ORound2((m.num_cr_points > 0 || m.chroma_scaling_from_luma)
                      ? kGaussianSequence[rng.Get(11)]
                      : 0,
                  cshift));
    }
  }

  for (int y = 3; y < ch; ++y) {
    for (int x = 3; x < cw - 3; ++x) {
      std::int64_t s0 = 0;
      std::int64_t s1 = 0;
      int pos = 0;
      for (int dr = -m.ar_coeff_lag; dr <= 0; ++dr) {
        for (int dc = -m.ar_coeff_lag; dc <= m.ar_coeff_lag; ++dc) {
          const int c0 = m.ar_coeffs_cb[pos];
          const int c1 = m.ar_coeffs_cr[pos];
          if (dr == 0 && dc == 0) {
            if (m.num_y_points > 0) {
              int luma = 0;
              const int lx = ((x - 3) << sub_x) + 3;
              const int ly = ((y - 3) << sub_y) + 3;
              for (int i = 0; i <= sub_y; ++i) {
                for (int j = 0; j <= sub_x; ++j) luma += g->y[ly + i][lx + j];
              }
              luma = static_cast<int>(ORound2(luma, sub_x + sub_y));
              s0 += static_cast<std::int64_t>(luma) * c0;
              s1 += static_cast<std::int64_t>(luma) * c1;
            }
            break;
          }
          s0 += static_cast<std::int64_t>(g->u[y + dr][x + dc]) * c0;
          s1 += static_cast<std::int64_t>(g->v[y + dr][x + dc]) * c1;
          ++pos;
        }
      }
      g->u[y][x] = OClip3(gmin, gmax, g->u[y][x] + static_cast<int>(ORound2(s0, ashift)));
      g->v[y][x] = OClip3(gmin, gmax, g->v[y][x] + static_cast<int>(ORound2(s1, ashift)));
    }
  }
}

void OracleScalingLut(const FilmGrainModel& m, int num_planes, int lut[3][256]) {
  for (int plane = 0; plane < num_planes; ++plane) {
    const int n = (plane == 0 || m.chroma_scaling_from_luma)
                      ? m.num_y_points
                      : (plane == 1 ? m.num_cb_points : m.num_cr_points);
    const int* xv = (plane == 0 || m.chroma_scaling_from_luma)
                        ? m.point_y_value
                        : (plane == 1 ? m.point_cb_value : m.point_cr_value);
    const int* yv = (plane == 0 || m.chroma_scaling_from_luma)
                        ? m.point_y_scaling
                        : (plane == 1 ? m.point_cb_scaling : m.point_cr_scaling);
    if (n == 0) {
      for (int x = 0; x < 256; ++x) lut[plane][x] = 0;
      continue;
    }
    for (int x = 0; x < xv[0]; ++x) lut[plane][x] = yv[0];
    for (int i = 0; i + 1 < n; ++i) {
      const int dy = yv[i + 1] - yv[i];
      const int dx = xv[i + 1] - xv[i];
      const int delta = dy * ((65536 + (dx >> 1)) / dx);
      for (int x = 0; x < dx; ++x) {
        lut[plane][xv[i] + x] = yv[i] + static_cast<int>((x * delta + 32768) >> 16);
      }
    }
    for (int x = xv[n - 1]; x < 256; ++x) lut[plane][x] = yv[n - 1];
  }
}

int OracleScaleLut(const int lut[3][256], int plane, int index, int bd) {
  const int shift = bd - 8;
  const int x = index >> shift;
  const int rem = index - (x << shift);
  if (x == 255) return lut[plane][255];
  return lut[plane][x] +
         static_cast<int>(ORound2(static_cast<std::int64_t>(lut[plane][x + 1] -
                                                           lut[plane][x]) *
                                      rem,
                                  shift));
}

struct OFrame {
  Grid y;
  Grid u;
  Grid v;
  int w = 0;
  int h = 0;
  int cw = 0;
  int ch = 0;
  int bd = 8;
  int num_planes = 3;
};

void OracleAddNoise(const FilmGrainModel& m, const OGrain& g, OFrame* f,
                    const int lut[3][256], std::uint16_t seed) {
  const int w = f->w;
  const int h = f->h;
  const int bd = f->bd;
  const int sub_x = f->cw == w ? 0 : 1;
  const int sub_y = f->ch == h ? 0 : 1;
  const int gmin = -(1 << (bd - 1));
  const int gmax = (1 << (bd - 1)) - 1;
  const int luma_size = m.film_grain_block_size ? 32 : 16;
  const int step = luma_size / 2;
  const int half_h = (h + 1) / 2;
  const int nstripes = (half_h + step - 1) / step;

  const int sh[3] = {34, 34 >> sub_y, 34 >> sub_y};
  const int csw = ((w + 1) >> 1) + luma_size + 4;
  const int sw[3] = {w + 2 * luma_size + 4, sub_x ? csw : w + 2 * luma_size + 4,
                     sub_x ? csw : w + 2 * luma_size + 4};
  std::vector<std::vector<Grid>> stripe(3);
  for (int p = 0; p < 3; ++p) {
    stripe[p].resize(nstripes);
    for (int n = 0; n < nstripes; ++n) stripe[p][n] = MakeGrid(sw[p], sh[p]);
  }

  ORandom rng(seed);
  int luma_num = 0;
  for (int y = 0; y < half_h; y += step) {
    rng.Set(seed);
    const int lr = y >> 3;
    rng.Xor(static_cast<std::uint16_t>(((lr * 37 + 178) & 255) << 8));
    rng.Xor(static_cast<std::uint16_t>((lr * 173 + 105) & 255));
    for (int x = 0; x < (w + 1) / 2; x += step) {
      const int oy = rng.Get(9) * (3 - static_cast<int>(m.film_grain_block_size)) >> 6;
      rng.Get(1);
      rng.Get(1);
      rng.Get(1);
      const int ox = rng.Get(9) * (3 - static_cast<int>(m.film_grain_block_size)) >> 6;
      rng.Get(1);
      rng.Get(1);
      rng.Get(1);
      for (int p = 0; p < f->num_planes; ++p) {
        const int psx = p > 0 ? sub_x : 0;
        const int psy = p > 0 ? sub_y : 0;
        const int pox = psx ? 6 + ox : 9 + ox * 2;
        const int poy = psy ? 6 + oy : 9 + oy * 2;
        for (int i = 0; i < (luma_size + 2) >> psy; ++i) {
          for (int j = 0; j < (luma_size + 2) >> psx; ++j) {
            int val = p == 0 ? g.y[poy + i][pox + j]
                             : (p == 1 ? g.u[poy + i][pox + j] : g.v[poy + i][pox + j]);
            if (psx == 0) {
              if (j < 2 && m.overlap_flag && x > 0) {
                const int old = stripe[p][luma_num][i][x * 2 + j];
                val = j == 0 ? old * 27 + val * 17 : old * 17 + val * 27;
                val = OClip3(gmin, gmax, static_cast<int>(ORound2(val, 5)));
              }
              stripe[p][luma_num][i][x * 2 + j] = val;
            } else {
              if (j == 0 && m.overlap_flag && x > 0) {
                const int old = stripe[p][luma_num][i][x + j];
                val = old * 23 + val * 22;
                val = OClip3(gmin, gmax, static_cast<int>(ORound2(val, 5)));
              }
              stripe[p][luma_num][i][x + j] = val;
            }
          }
        }
      }
    }
    ++luma_num;
  }

  std::vector<Grid> noise(3);
  noise[0] = MakeGrid(w, h);
  noise[1] = MakeGrid((w + sub_x) >> sub_x, (h + sub_y) >> sub_y);
  noise[2] = MakeGrid((w + sub_x) >> sub_x, (h + sub_y) >> sub_y);

  for (int p = 0; p < f->num_planes; ++p) {
    const int psx = p > 0 ? sub_x : 0;
    const int psy = p > 0 ? sub_y : 0;
    const int iw = (w + psx) >> psx;
    const int ih = (h + psy) >> psy;
    const int shift = 4 + static_cast<int>(m.film_grain_block_size) - psy;
    for (int y = 0; y < ih; ++y) {
      const int n = y >> shift;
      const int i = y - (n << shift);
      for (int x = 0; x < iw; ++x) {
        int val = stripe[p][n][i][x];
        if (psy == 0) {
          if (i < 2 && n > 0 && m.overlap_flag) {
            const int old = stripe[p][n - 1][i + luma_size][x];
            val = i == 0 ? old * 27 + val * 17 : old * 17 + val * 27;
            val = OClip3(gmin, gmax, static_cast<int>(ORound2(val, 5)));
          }
        } else {
          if (i < 1 && n > 0 && m.overlap_flag) {
            const int old = stripe[p][n - 1][i + (luma_size >> 1)][x];
            val = old * 23 + val * 22;
            val = OClip3(gmin, gmax, static_cast<int>(ORound2(val, 5)));
          }
        }
        noise[p][y][x] = val;
      }
    }
  }

  int min_value;
  int max_luma;
  int max_chroma;
  if (m.clip_to_restricted_range) {
    min_value = 16 << (bd - 8);
    max_luma = 235 << (bd - 8);
    max_chroma = m.fg_mc_identity ? max_luma : 240 << (bd - 8);
  } else {
    min_value = 0;
    max_luma = (256 << (bd - 8)) - 1;
    max_chroma = max_luma;
  }
  const int sshift = m.grain_scaling_minus_8 + 8;

  if (f->num_planes > 1) {
    for (int y = 0; y < f->ch; ++y) {
      for (int x = 0; x < f->cw; ++x) {
        const int lx = x << sub_x;
        const int ly = y << sub_y;
        const int lnext = std::min(lx + 1, w - 1);
        const int avg = sub_x ? static_cast<int>(ORound2(f->y[ly][lx] + f->y[ly][lnext], 1))
                              : f->y[ly][lx];
        if (m.num_cb_points > 0 || m.chroma_scaling_from_luma) {
          const int orig = f->u[y][x];
          int merged = avg;
          if (!m.chroma_scaling_from_luma) {
            const int comb = avg * (m.cb_luma_mult - 128) + orig * (m.cb_mult - 128);
            merged = OClip3(0, (1 << bd) - 1,
                            (comb >> 6) + ((m.cb_offset - 256) << (bd - 8)));
          }
          const int n = static_cast<int>(
              ORound2(static_cast<std::int64_t>(OracleScaleLut(lut, 1, merged, bd)) *
                          noise[1][y][x],
                      sshift));
          f->u[y][x] = OClip3(min_value, max_chroma, orig + n);
        }
        if (f->num_planes > 2 && (m.num_cr_points > 0 || m.chroma_scaling_from_luma)) {
          const int orig = f->v[y][x];
          int merged = avg;
          if (!m.chroma_scaling_from_luma) {
            const int comb = avg * (m.cr_luma_mult - 128) + orig * (m.cr_mult - 128);
            merged = OClip3(0, (1 << bd) - 1,
                            (comb >> 6) + ((m.cr_offset - 256) << (bd - 8)));
          }
          const int n = static_cast<int>(
              ORound2(static_cast<std::int64_t>(OracleScaleLut(lut, 2, merged, bd)) *
                          noise[2][y][x],
                      sshift));
          f->v[y][x] = OClip3(min_value, max_chroma, orig + n);
        }
      }
    }
  }

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (m.num_y_points == 0) continue;
      const int orig = f->y[y][x];
      const int n = static_cast<int>(
          ORound2(static_cast<std::int64_t>(OracleScaleLut(lut, 0, orig, bd)) *
                      noise[0][y][x],
                  sshift));
      f->y[y][x] = OClip3(min_value, max_luma, orig + n);
    }
  }
}

void OracleFilmGrain(const FilmGrainModel& m, std::uint16_t seed, OFrame* f) {
  OGrain g;
  const int sub_x = f->cw == f->w ? 0 : 1;
  const int sub_y = f->ch == f->h ? 0 : 1;
  OracleGenerate(m, seed, sub_x, sub_y, f->bd, &g);
  int lut[3][256] = {};
  OracleScalingLut(m, f->num_planes, lut);
  OracleAddNoise(m, g, f, lut, seed);
}

FilmGrainModel MakeModel(int lag, int num_y, int num_cb, int num_cr, bool csl,
                         bool overlap, bool block32, bool restricted,
                         bool mc_identity) {
  FilmGrainModel m;
  m.chroma_scaling_from_luma = csl;
  m.num_y_points = num_y;
  const int yv[4] = {0, 40, 128, 220};
  const int ys[4] = {6, 30, 80, 130};
  for (int i = 0; i < num_y; ++i) {
    m.point_y_value[i] = yv[i];
    m.point_y_scaling[i] = ys[i];
  }
  m.num_cb_points = num_cb;
  const int cbv[2] = {0, 160};
  const int cbs[2] = {3, 25};
  for (int i = 0; i < num_cb; ++i) {
    m.point_cb_value[i] = cbv[i];
    m.point_cb_scaling[i] = cbs[i];
  }
  m.num_cr_points = num_cr;
  const int crv[2] = {0, 140};
  const int crs[2] = {4, 28};
  for (int i = 0; i < num_cr; ++i) {
    m.point_cr_value[i] = crv[i];
    m.point_cr_scaling[i] = crs[i];
  }
  m.grain_scaling_minus_8 = 1;
  m.ar_coeff_lag = lag;
  const int n = 2 * lag * (lag + 1);
  for (int i = 0; i < n; ++i) {
    m.ar_coeffs_y[i] = (i % 5) - 2;
    m.ar_coeffs_cb[i] = (i % 3) - 1;
    m.ar_coeffs_cr[i] = 1 - (i % 3);
  }
  if (num_y > 0) {
    m.ar_coeffs_cb[n] = 1;
    m.ar_coeffs_cr[n] = -1;
  }
  m.ar_coeff_shift_minus_6 = 1;
  m.grain_scale_shift = 0;
  m.cb_mult = 128;
  m.cb_luma_mult = 140;
  m.cb_offset = 256;
  m.cr_mult = 120;
  m.cr_luma_mult = 150;
  m.cr_offset = 250;
  m.overlap_flag = overlap;
  m.clip_to_restricted_range = restricted;
  m.fg_mc_identity = mc_identity;
  m.film_grain_block_size = block32;
  return m;
}

struct HostFrame {
  Image y;
  Image u;
  Image v;
  int w = 0;
  int h = 0;
  int bd = 8;
  int num_planes = 3;
};

void RunFg(const FilmGrainModel& m, HostFrame& host, std::uint16_t seed);

FilmGrainFrame MakeView(HostFrame& f) {
  FilmGrainFrame view;
  view.plane[0].data = f.y.px.data();
  view.plane[0].stride = f.y.stride;
  view.plane[1].data = f.u.px.data();
  view.plane[1].stride = f.u.stride;
  view.plane[2].data = f.v.px.data();
  view.plane[2].stride = f.v.stride;
  view.w = f.w;
  view.h = f.h;
  view.sub_x = f.num_planes == 1 ? 1 : (f.u.width == f.w ? 0 : 1);
  view.sub_y = f.num_planes == 1 ? 1 : (f.u.height == f.h ? 0 : 1);
  view.bit_depth = f.bd;
  view.num_planes = f.num_planes;
  return view;
}

void RunFg(const FilmGrainModel& m, HostFrame& host, std::uint16_t seed) {
  FilmGrainFrame view = MakeView(host);
  ApplyFilmGrain(m, view, seed);
}

bool CompareFrame(const HostFrame& got, const OFrame& want) {
  if (got.w != want.w || got.h != want.h) return false;
  for (int y = 0; y < got.h; ++y) {
    for (int x = 0; x < got.w; ++x) {
      if (got.y.At(x, y) != want.y[y][x]) {
        std::fprintf(stderr, "fg Y mismatch at %d,%d: %d != %d\n", x, y,
                     got.y.At(x, y), want.y[y][x]);
        return false;
      }
    }
  }
  for (int p = 1; p < got.num_planes; ++p) {
    const Image& img = p == 1 ? got.u : got.v;
    const Grid& ref = p == 1 ? want.u : want.v;
    for (int y = 0; y < img.height; ++y) {
      for (int x = 0; x < img.width; ++x) {
        if (img.At(x, y) != ref[y][x]) {
          std::fprintf(stderr, "fg P%d mismatch at %d,%d: %d != %d\n", p, x, y,
                       img.At(x, y), ref[y][x]);
          return false;
        }
      }
    }
  }
  return true;
}

void TestAgainstOracle() {
  TestRng rng(0xfeedface);
  int cases = 0;
  const int sizes[3][2] = {{40, 33}, {24, 24}, {17, 9}};
  for (int bd : {8, 10}) {
    const int maxv = (1 << bd) - 1;
    for (int sub_mode = 0; sub_mode < 3; ++sub_mode) {  // 0=420, 1=444, 2=400
      const int planes = sub_mode == 2 ? 1 : 3;
      const int sub_x = sub_mode == 1 ? 0 : 1;
      const int sub_y = sub_mode == 1 ? 0 : 1;
      for (int lag : {0, 1, 2, 3}) {
        for (int block32 : {0, 1}) {
          for (int overlap : {0, 1}) {
            for (int restricted : {0, 1}) {
              const int ny = 4;
              const int ncb = (planes == 1 || sub_mode == 1) ? (sub_mode == 1 ? 2 : 0) : 2;
              const int ncr = ncb;
              FilmGrainModel m = MakeModel(lag, ny, ncb, ncr, false, overlap != 0,
                                           block32 != 0, restricted != 0,
                                           restricted != 0);
              for (const auto& size : sizes) {
                HostFrame host;
                host.w = size[0];
                host.h = size[1];
                host.bd = bd;
                host.num_planes = planes;
                host.y.Reset(host.w, host.h);
                if (planes > 1) {
                  host.u.Reset((host.w + sub_x) >> sub_x, (host.h + sub_y) >> sub_y);
                  host.v.Reset((host.w + sub_x) >> sub_x, (host.h + sub_y) >> sub_y);
                } else {
                  host.u.Reset(1, 1);
                  host.v.Reset(1, 1);
                }
                for (auto& val : host.y.px) val = static_cast<std::uint16_t>(rng.Range(0, maxv));
                for (auto& val : host.u.px) val = static_cast<std::uint16_t>(rng.Range(0, maxv));
                for (auto& val : host.v.px) val = static_cast<std::uint16_t>(rng.Range(0, maxv));

                OFrame want;
                want.w = host.w;
                want.h = host.h;
                want.bd = bd;
                want.num_planes = planes;
                want.cw = (host.w + sub_x) >> sub_x;
                want.ch = (host.h + sub_y) >> sub_y;
                want.y = MakeGrid(want.w, want.h);
                want.u = MakeGrid(want.cw, want.ch);
                want.v = MakeGrid(want.cw, want.ch);
                for (int y = 0; y < want.h; ++y)
                  for (int x = 0; x < want.w; ++x) want.y[y][x] = host.y.At(x, y);
                if (planes > 1) {
                  for (int y = 0; y < want.ch; ++y) {
                    for (int x = 0; x < want.cw; ++x) {
                      want.u[y][x] = host.u.At(x, y);
                      want.v[y][x] = host.v.At(x, y);
                    }
                  }
                }

                OracleFilmGrain(m, 0x1234, &want);
                RunFg(m, host, 0x1234);
                ++cases;
                CHECK(CompareFrame(host, want));
              }
            }
          }
        }
      }
    }
  }
  CHECK(cases >= 200);
}

void TestSeedsAndScaleShift() {
  const int sub_x = 1;
  const int sub_y = 1;
  for (int seed_value : {0, 1, 0xffff}) {
    const auto seed = static_cast<std::uint16_t>(seed_value);
    for (int gss = 0; gss <= 3; ++gss) {
      FilmGrainModel m = MakeModel(2, 4, 2, 2, false, true, false, false, false);
      m.grain_scale_shift = gss;
      HostFrame host;
      host.w = 23;
      host.h = 15;
      host.bd = 8;
      host.num_planes = 3;
      host.y.Reset(host.w, host.h);
      host.u.Reset((host.w + sub_x) >> sub_x, (host.h + sub_y) >> sub_y);
      host.v.Reset((host.w + sub_x) >> sub_x, (host.h + sub_y) >> sub_y);
      TestRng rng(0x22220000u + seed + static_cast<std::uint32_t>(gss));
      for (auto& v : host.y.px) v = static_cast<std::uint16_t>(rng.Range(0, 255));
      for (auto& v : host.u.px) v = static_cast<std::uint16_t>(rng.Range(0, 255));
      for (auto& v : host.v.px) v = static_cast<std::uint16_t>(rng.Range(0, 255));

      OFrame want;
      want.w = host.w;
      want.h = host.h;
      want.bd = 8;
      want.num_planes = 3;
      want.cw = host.u.width;
      want.ch = host.u.height;
      want.y = MakeGrid(want.w, want.h);
      want.u = MakeGrid(want.cw, want.ch);
      want.v = MakeGrid(want.cw, want.ch);
      for (int y = 0; y < want.h; ++y)
        for (int x = 0; x < want.w; ++x) want.y[y][x] = host.y.At(x, y);
      for (int y = 0; y < want.ch; ++y) {
        for (int x = 0; x < want.cw; ++x) {
          want.u[y][x] = host.u.At(x, y);
          want.v[y][x] = host.v.At(x, y);
        }
      }
      OracleFilmGrain(m, seed, &want);
      RunFg(m, host, seed);
      CHECK(CompareFrame(host, want));
    }
  }
}

void TestZeroLumaPoints() {
  const int bd = 8;
  for (int block32 : {0, 1}) {
    FilmGrainModel m = MakeModel(2, 0, 2, 2, false, true, block32 != 0, false, false);
    HostFrame host;
    host.w = 21;
    host.h = 13;
    host.bd = bd;
    host.num_planes = 3;
    host.y.Reset(host.w, host.h);
    host.u.Reset(11, 7);
    host.v.Reset(11, 7);
    for (auto& v : host.y.px) v = static_cast<std::uint16_t>(100 + (v & 31));
    for (auto& v : host.u.px) v = static_cast<std::uint16_t>(90 + (v & 31));
    for (auto& v : host.v.px) v = static_cast<std::uint16_t>(80 + (v & 31));
    HostFrame before = host;
    RunFg(m, host, 7);
    CHECK(host.y.px == before.y.px);  // num_y_points == 0 -> luma untouched
    CHECK(host.u.px != before.u.px);  // chroma still gets grain
  }
}

void TestNoChromaPoints() {
  FilmGrainModel m = MakeModel(2, 3, 0, 0, false, true, false, false, false);
  HostFrame host;
  host.w = 16;
  host.h = 16;
  host.bd = 8;
  host.num_planes = 3;
  host.y.Reset(host.w, host.h);
  host.u.Reset(8, 8);
  host.v.Reset(8, 8);
  for (auto& v : host.y.px) v = static_cast<std::uint16_t>(50 + (v & 63));
  host.u.Fill(77);
  host.v.Fill(88);
  HostFrame before = host;
  RunFg(m, host, 3);
  CHECK(host.y.px != before.y.px);
  CHECK(host.u.px == before.u.px);
  CHECK(host.v.px == before.v.px);
}

void TestChromaFromLuma() {
  FilmGrainModel m = MakeModel(2, 3, 0, 0, true, true, false, false, false);
  HostFrame host;
  host.w = 16;
  host.h = 16;
  host.bd = 8;
  host.num_planes = 3;
  host.y.Reset(host.w, host.h);
  host.u.Reset(8, 8);
  host.v.Reset(8, 8);
  for (auto& v : host.y.px) v = static_cast<std::uint16_t>(40 + (v & 63));
  host.u.Fill(77);
  host.v.Fill(88);
  HostFrame before = host;
  RunFg(m, host, 5);
  CHECK(host.u.px != before.u.px);
  CHECK(host.v.px != before.v.px);
}

void TestClipRange() {
  FilmGrainModel m = MakeModel(2, 4, 2, 2, false, true, false, true, true);
  HostFrame host;
  host.w = 24;
  host.h = 24;
  host.bd = 8;
  host.num_planes = 3;
  host.y.Reset(host.w, host.h);
  host.u.Reset(12, 12);
  host.v.Reset(12, 12);
  host.y.Fill(0);
  host.u.Fill(0);
  host.v.Fill(255);
  RunFg(m, host, 9);
  bool y_ok = true;
  for (auto v : host.y.px) y_ok = y_ok && v >= 16 && v <= 235;
  bool uv_ok = true;
  for (auto v : host.u.px) uv_ok = uv_ok && v >= 16 && v <= 235;
  for (auto v : host.v.px) uv_ok = uv_ok && v >= 16 && v <= 235;
  CHECK(y_ok);
  CHECK(uv_ok);
}

void TestDeterminism() {
  FilmGrainModel m = MakeModel(3, 4, 2, 2, false, true, true, false, false);
  auto run = [&](std::uint16_t seed, HostFrame* out) {
    out->w = 19;
    out->h = 11;
    out->bd = 8;
    out->num_planes = 3;
    out->y.Reset(out->w, out->h);
    out->u.Reset(10, 6);
    out->v.Reset(10, 6);
    for (auto& v : out->y.px) v = 60;
    for (auto& v : out->u.px) v = 70;
    for (auto& v : out->v.px) v = 80;
    RunFg(m, *out, seed);
  };
  HostFrame a;
  HostFrame b;
  HostFrame c;
  run(0xabcd, &a);
  run(0xabcd, &b);
  run(0xabce, &c);
  CHECK(a.y.px == b.y.px && a.u.px == b.u.px && a.v.px == b.v.px);
  CHECK(a.y.px != c.y.px);
}

void TestValidation() {
  std::string why;
  FilmGrainModel good = MakeModel(2, 3, 2, 2, false, true, false, false, false);
  CHECK(FilmGrainModelValid(good, 1, 1, 8, 3, &why));

  FilmGrainModel bad = good;
  bad.num_y_points = 15;
  CHECK(!FilmGrainModelValid(bad, 1, 1, 8, 3, &why));

  bad = good;
  bad.point_y_value[2] = bad.point_y_value[1];
  CHECK(!FilmGrainModelValid(bad, 1, 1, 8, 3, &why));

  bad = good;
  bad.num_cb_points = 2;
  bad.num_cr_points = 0;
  CHECK(!FilmGrainModelValid(bad, 1, 1, 8, 3, &why));

  bad = good;
  bad.chroma_scaling_from_luma = true;
  CHECK(!FilmGrainModelValid(bad, 1, 1, 8, 3, &why));

  bad = good;
  bad.ar_coeff_lag = 4;
  CHECK(!FilmGrainModelValid(bad, 1, 1, 8, 3, &why));

  bad = good;
  bad.grain_scale_shift = 4;
  CHECK(!FilmGrainModelValid(bad, 1, 1, 8, 3, &why));

  CHECK(!FilmGrainModelValid(good, 1, 1, 14, 3, &why));

  // Invalid models make ApplyFilmGrain a no-op.
  HostFrame host;
  host.w = 8;
  host.h = 8;
  host.bd = 8;
  host.num_planes = 1;
  host.y.Reset(8, 8);
  host.y.Fill(100);
  host.u.Reset(1, 1);
  host.v.Reset(1, 1);
  HostFrame before = host;
  bad = good;
  bad.ar_coeff_lag = 4;
  RunFg(bad, host, 1);
  CHECK(host.y.px == before.y.px);
}

}  // namespace

int main() {
  TestAgainstOracle();
  TestSeedsAndScaleShift();
  TestZeroLumaPoints();
  TestNoChromaPoints();
  TestChromaFromLuma();
  TestClipRange();
  TestDeterminism();
  TestValidation();
  return Finish("film_grain_test");
}
