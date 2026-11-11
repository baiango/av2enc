// CPU correctness tests for MHCCP (§7.13.6-§7.13.8).
//
// Coverage:
//   * an independent literal transcription of the spec (capture, least-squares
//     solve, Gaussian elimination, prediction) against the reference API on
//     synthetic and real 4:2:0 chroma;
//   * edge/partial blocks, superblock tops, above-right/below-left extents;
//   * conditioning limits (count == 0, constant luma, near-singular matrices);
//   * direction search and explicit/derived/multi hypothesis selection;
//   * alpha quantisation and read_cfl_alphas()/cfl_mh_dir round-trips through
//     the normative §8.2 entropy coder.

#include "mhccp.h"
#include "mhccp_syntax.h"

#include "../cfl.h"
#include "entropy/av2/av2_entropy.h"
#include "io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace mh = av2enc::cfl::mhccp;
namespace cfl = av2enc::cfl;
namespace av2 = av2enc::av2;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 20) {
      std::printf("FAIL: %s\n", what);
    }
  }
}

template <typename T>
void check_eq(T got, T want, const char* what) {
  ++g_checks;
  if (got != want) {
    ++g_failures;
    if (g_failures <= 20) {
      std::printf("FAIL: %s got %lld want %lld\n", what,
                  static_cast<long long>(got),
                  static_cast<long long>(want));
    }
  }
}

std::uint32_t g_rng = 0x9e3779b9u;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
int random_range(int low, int high) {
  return low + static_cast<int>(next_random() %
                                static_cast<std::uint32_t>(high - low + 1));
}

// ---------------------------------------------------------------------------
// Independent literal transcription of §7.13.2.1, §7.13.6-§7.13.8.
// ---------------------------------------------------------------------------

namespace literal {

constexpr int kDim = 66;

int abs_i(int v) { return v < 0 ? -v : v; }
int max_i(int a, int b) { return a > b ? a : b; }
int min_i(int a, int b) { return a < b ? a : b; }
int floor_log2_i(int v) {
  int s = 0;
  while (v > 1) {
    v >>= 1;
    ++s;
  }
  return s;
}
int get_msb_i(int v) { return v == 0 ? 0 : floor_log2_i(v); }
int ceil_log2_i(int v) {
  if (v < 2) return 0;
  int bits = 1;
  int p = 2;
  while (p < v) {
    ++bits;
    p <<= 1;
  }
  return bits;
}
int round2_i(int v, int n) {
  if (n <= 0) return v;
  return (v + (1 << (n - 1))) >> n;
}
std::int64_t asr_i(std::int64_t v, int s) {
  if (s <= 0) return v;
  if (v >= 0) return v >> s;
  std::int64_t d = static_cast<std::int64_t>(1) << s;
  return -(((-v) + d - 1) >> s);
}
std::int64_t round2_64_i(std::int64_t v, int n) {
  if (n <= 0) return v;
  std::int64_t h = static_cast<std::int64_t>(1) << (n - 1);
  return (v + h) >> n;
}
int round2_signed_i(int v, int n) {
  if (v >= 0) return round2_i(v, n);
  return -round2_i(-v, n);
}
int mul_adapt(int a, int b, int shift) {
  int bits_a = get_msb_i(abs_i(a)) + 1;
  int bits_b = get_msb_i(abs_i(b)) + 1;
  int need = max_i(0, bits_a + bits_b - 29);
  int s1 = need >> 1;
  int s2 = need - s1;
  int adj = shift - (s1 + s2);
  int aa = static_cast<int>(asr_i(a, s1));
  int bb = static_cast<int>(asr_i(b, s2));
  std::int64_t prod = static_cast<std::int64_t>(aa) * bb;
  if (adj <= 0) return static_cast<int>(prod);
  if (adj > 29) return 0;
  return round2_signed_i(static_cast<int>(prod), adj);
}

constexpr int kW[8] = {214, 153, 113, 86, 67, 53, 43, 35};
constexpr int kO[8] = {4822, 5952, 6624, 6792, 6408, 5424, 3792, 1466};
constexpr int kB[8] = {12784, 12054, 11670, 11583, 11764, 12195, 12870, 13782};

struct ScaleShift {
  int scale;
  int shift;
};
ScaleShift div_scale_shift(int denom) {
  int shift = floor_log2_i(denom);
  std::int64_t shifted = static_cast<std::int64_t>(denom) << 14;
  int rounded = static_cast<int>(round2_64_i(shifted, shift));
  int clipped = rounded < 1 ? 1 : (rounded > 32767 ? 32767 : rounded);
  int norm = clipped & 16383;
  int index = norm >> 11;
  int diff = norm - kO[index];
  std::int64_t term = (static_cast<std::int64_t>(kW[index]) *
                       ((static_cast<std::int64_t>(diff) * diff) >> 14)) >>
                      8;
  term -= static_cast<std::int64_t>(diff) >> 1;
  term += kB[index];
  return ScaleShift{static_cast<int>(term) << 2, shift};
}

struct Block {
  const std::uint8_t* luma = nullptr;
  int luma_stride = 0;
  int luma_max_x = 0;
  int luma_max_y = 0;
  std::uint8_t* chroma = nullptr;
  int chroma_stride = 0;
  int chroma_max_x = 0;
  int chroma_max_y = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int bit_depth = 8;
  int ds_filter = 0;
  int sub_x = 1;
  int sub_y = 1;
  int have_above = 0;
  int have_left = 0;
  int num4_above_right = 0;
  int num4_below_left = 0;
  int mi_col_end = 0;
  int mi_row_end = 0;
  int sb_luma_y = 0;
};

struct Refs {
  int above = 0;
  int left = 0;
  int rows = 0;
  int cols = 0;
  int target_h = 0;
  int target_w = 0;
  int ref_h = 0;
  int ref_w = 0;
  int luma[kDim][kDim] = {};
  int chroma[kDim][kDim] = {};
};

int ds_weight(int filter, int dy, int dx) {
  return cfl::ds_weight(filter, dy, dx, 1, 1);
}

int luma_sample(const Block& b, int chroma_x, int chroma_y, bool clamp_x,
                bool clamp_y) {
  int luma_x = chroma_x << b.sub_x;
  int luma_y = chroma_y << b.sub_y;
  int filter = b.ds_filter == 3 ? 0 : b.ds_filter;
  int limit = b.sb_luma_y - 1;
  int t = 0;
  for (int dy = -b.sub_y; dy <= b.sub_y; ++dy) {
    int sy = luma_y + (clamp_y ? max_i(0, dy) : dy);
    int row = max_i(limit, sy);
    row = row < 0 ? 0 : (row > b.luma_max_y ? b.luma_max_y : row);
    for (int dx = -b.sub_x; dx <= b.sub_x; ++dx) {
      int sx = luma_x + (clamp_x ? max_i(0, dx) : dx);
      int col = sx < 0 ? 0 : (sx > b.luma_max_x ? b.luma_max_x : sx);
      int wgt = ds_weight(filter, dy, dx);
      if (wgt != 0) {
        t += wgt * static_cast<int>(
                       b.luma[static_cast<std::size_t>(row) *
                                  static_cast<std::size_t>(b.luma_stride) +
                              static_cast<std::size_t>(col)]);
      }
    }
  }
  return t >> 3;
}

bool capture(const Block& b, Refs& r) {
  int above = b.have_above ? 2 : 0;
  int left = b.have_left ? 2 : 0;
  int px_tr = 4 * b.num4_above_right;
  int px_bl = 4 * b.num4_below_left;
  if ((b.w << b.sub_x) <= 4 || !b.have_above) px_tr = 0;
  if ((b.h << b.sub_y) <= 4 || !b.have_left) px_bl = 0;
  int right_x = min_i(b.mi_col_end, (b.x + b.w + px_tr) << b.sub_x);
  int bottom_y = min_i(b.mi_row_end, (b.y + b.h + px_bl) << b.sub_y);
  int width = min_i((left << b.sub_x) + (right_x - (b.x << b.sub_x)), 128);
  int height =
      min_i((above << b.sub_y) + (bottom_y - (b.y << b.sub_y)), 128);
  if (width < 0) width = 0;
  if (height < 0) height = 0;

  r.above = above;
  r.left = left;
  r.rows = b.h + above;
  r.cols = b.w + left;
  r.target_w = round2_i(width, b.sub_x);
  r.target_h = round2_i(height, b.sub_y);
  r.ref_w = width >> b.sub_x;
  r.ref_h = height >> b.sub_y;

  for (int i = 0; i < kDim; ++i) {
    for (int j = 0; j < kDim; ++j) {
      r.luma[i][j] = 0;
      r.chroma[i][j] = 0;
    }
  }

  int sb_chroma_y = b.sb_luma_y >> b.sub_y;
  for (int i = 0; i < r.target_h && i < kDim; ++i) {
    for (int j = 0; j < r.target_w && j < kDim; ++j) {
      int cx = b.x + j - left;
      int cy = b.y + i - above;
      if (i < above || j < left) {
        int row = max_i(cy, sb_chroma_y - 1);
        row = row < 0 ? 0 : (row > b.chroma_max_y ? b.chroma_max_y : row);
        int col = cx < 0 ? 0 : (cx > b.chroma_max_x ? b.chroma_max_x : cx);
        r.chroma[i][j] = static_cast<int>(
            b.chroma[static_cast<std::size_t>(row) *
                         static_cast<std::size_t>(b.chroma_stride) +
                     static_cast<std::size_t>(col)]);
      }
      const bool avail = (i < above || j < left + b.w) &&
                         (i < above + b.h || j < left);
      if (avail) {
        r.luma[i][j] = luma_sample(b, cx, cy, j == 0, i == 0);
      }
    }
  }
  return true;
}

struct Params {
  int p[3] = {0, 0, 65536};
};

struct DeriveResult {
  Params params;
  int count = 0;
};

DeriveResult derive(const Block& b, const Refs& r, int dir) {
  DeriveResult out{};
  std::int64_t ata[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  std::int64_t bv[3] = {0, 0, 0};
  int v[3] = {0, 0, 1 << (b.bit_depth - 1)};
  int ref_h = min_i(r.ref_h, kDim);
  int ref_w = min_i(r.ref_w, kDim);
  for (int i = 1; i < ref_h - 1; ++i) {
    for (int j = 1; j < ref_w - 1; ++j) {
      if (i < r.above || j < r.left) {
        if (dir == 0) {
          v[0] = r.luma[i][j];
        } else if (dir == 1) {
          v[0] = r.luma[max_i(0, i - 1)][j];
        } else {
          v[0] = r.luma[i][max_i(0, j - 1)];
        }
        v[1] = round2_i(r.luma[i][j] * r.luma[i][j], b.bit_depth);
        int target = r.chroma[i][j];
        for (int i0 = 0; i0 < 3; ++i0) {
          for (int i1 = i0; i1 < 3; ++i1) {
            ata[i0][i1] += static_cast<std::int64_t>(v[i0]) * v[i1];
          }
          bv[i0] += static_cast<std::int64_t>(v[i0]) * target;
        }
        ++out.count;
      }
    }
  }
  if (out.count == 0) {
    return out;
  }

  int matrix_shift =
      16 + 6 - 2 * b.bit_depth - ceil_log2_i(out.count);
  int left_shift = matrix_shift > 0 ? matrix_shift : 0;
  int right_shift = matrix_shift > 0 ? 0 : -matrix_shift;
  int na[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  int nb[3] = {0, 0, 0};
  for (int i0 = 0; i0 < 3; ++i0) {
    for (int i1 = i0; i1 < 3; ++i1) {
      na[i0][i1] = static_cast<int>(
          asr_i(ata[i0][i1] * std::int64_t{1} * (std::int64_t{1} << left_shift),
                right_shift));
    }
    nb[i0] = static_cast<int>(
        asr_i(bv[i0] * std::int64_t{1} * (std::int64_t{1} << left_shift),
              right_shift));
  }

  int c[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = j >= i ? na[i][j] : na[j][i];
    }
    c[i][i] += 2 << (b.bit_depth - 8);
    c[i][3] = nb[i];
  }

  for (int i = 0; i < 3; ++i) {
    int diag = max_i(1, abs_i(c[i][i]));
    ScaleShift ss = div_scale_shift(diag);
    for (int j = i + 1; j < 4; ++j) {
      c[i][j] = mul_adapt(c[i][j], ss.scale, ss.shift);
    }
    for (int j = i + 1; j < 3; ++j) {
      int sf = c[j][i];
      for (int k = i + 1; k < 4; ++k) {
        c[j][k] -= mul_adapt(sf, c[i][k], 16);
      }
    }
  }

  out.params.p[2] = c[2][3];
  for (int i = 2; i >= 0; --i) {
    out.params.p[i] = c[i][3];
    for (int j = i + 1; j < 3; ++j) {
      out.params.p[i] -= mul_adapt(c[i][j], out.params.p[j], 16);
    }
  }
  return out;
}

int predict(const Block& b, const Refs& r, const Params& p, int dir, int i,
            int j) {
  int row = r.above + i;
  int col = r.left + j;
  int a = r.luma[row][col];
  int v[3] = {0, 0, 1 << (b.bit_depth - 1)};
  if (dir == 0) {
    v[0] = a;
  } else if (dir == 1) {
    v[0] = r.luma[max_i(0, row - 1)][col];
  } else {
    v[0] = r.luma[row][max_i(0, col - 1)];
  }
  v[1] = round2_i(a * a, b.bit_depth);
  int t = 0;
  for (int k = 0; k < 3; ++k) {
    t += mul_adapt(p.p[k], v[k], 16);
  }
  return t < 0 ? 0 : (t > 255 ? 255 : t);
}

}  // namespace literal

// ---------------------------------------------------------------------------
// Case construction
// ---------------------------------------------------------------------------

constexpr int kLumaStride = 256;
constexpr int kLumaSize = kLumaStride * 256;
constexpr int kChromaStride = 128;
constexpr int kChromaSize = kChromaStride * 128;

struct Case {
  std::vector<std::uint8_t> luma;
  std::vector<std::uint8_t> chroma;
  mh::Block block;
  literal::Block literal_block;
};

Case make_case(int coded_w, int coded_h, int w, int h, int x, int y, int filter,
               int num4_ar, int num4_bl, bool random_values) {
  Case c;
  c.luma.assign(kLumaSize, 128);
  c.chroma.assign(kChromaSize, 128);
  if (random_values) {
    for (std::uint8_t& v : c.luma) v = static_cast<std::uint8_t>(next_random());
    for (std::uint8_t& v : c.chroma) v = static_cast<std::uint8_t>(next_random());
  } else {
    for (int row = 0; row < 256; ++row) {
      for (int col = 0; col < 256; ++col) {
        c.luma[static_cast<std::size_t>(row) * kLumaStride + static_cast<std::size_t>(col)] =
            static_cast<std::uint8_t>((row * 7 + col * 13 + (row * col) / 7) &
                                      0xFF);
      }
    }
    for (int row = 0; row < 128; ++row) {
      for (int col = 0; col < 128; ++col) {
        c.chroma[static_cast<std::size_t>(row) * kChromaStride + static_cast<std::size_t>(col)] =
            static_cast<std::uint8_t>((row * 3 + col * 5 + (row * row) / 9) &
                                      0xFF);
      }
    }
  }

  mh::Block& b = c.block;
  b.luma = c.luma.data();
  b.luma_stride = kLumaStride;
  b.luma_max_x = 255;
  b.luma_max_y = 255;
  b.chroma = c.chroma.data();
  b.chroma_stride = kChromaStride;
  b.chroma_max_x = 127;
  b.chroma_max_y = 127;
  b.x = x;
  b.y = y;
  b.w = w;
  b.h = h;
  b.bit_depth = 8;
  b.ds_filter = filter;
  b.sub_x = 1;
  b.sub_y = 1;
  b.have_above = y >= 4 ? 1 : 0;
  b.have_left = x >= 4 ? 1 : 0;
  b.num4_above_right = num4_ar;
  b.num4_below_left = num4_bl;
  b.mi_col_end = coded_w << 1;
  b.mi_row_end = coded_h << 1;
  b.sb_luma_y = ((y << 1) / 64) * 64;

  literal::Block& lb = c.literal_block;
  lb.luma = b.luma;
  lb.luma_stride = b.luma_stride;
  lb.luma_max_x = b.luma_max_x;
  lb.luma_max_y = b.luma_max_y;
  lb.chroma = b.chroma;
  lb.chroma_stride = b.chroma_stride;
  lb.chroma_max_x = b.chroma_max_x;
  lb.chroma_max_y = b.chroma_max_y;
  lb.x = b.x;
  lb.y = b.y;
  lb.w = b.w;
  lb.h = b.h;
  lb.bit_depth = b.bit_depth;
  lb.ds_filter = b.ds_filter;
  lb.sub_x = b.sub_x;
  lb.sub_y = b.sub_y;
  lb.have_above = b.have_above;
  lb.have_left = b.have_left;
  lb.num4_above_right = b.num4_above_right;
  lb.num4_below_left = b.num4_below_left;
  lb.mi_col_end = b.mi_col_end;
  lb.mi_row_end = b.mi_row_end;
  lb.sb_luma_y = b.sb_luma_y;
  return c;
}

void compare_one(Case& c, int dir, const char* tag) {
  mh::Refs refs;
  check(mh::capture_refs(c.block, refs), "reference capture_refs");

  literal::Refs literal_refs;
  literal::capture(c.literal_block, literal_refs);

  check_eq(refs.above, literal_refs.above, "capture above");
  check_eq(refs.left, literal_refs.left, "capture left");
  check_eq(refs.rows, literal_refs.rows, "capture rows");
  check_eq(refs.cols, literal_refs.cols, "capture cols");
  check_eq(refs.ref_h, literal_refs.ref_h, "capture ref_h");
  check_eq(refs.ref_w, literal_refs.ref_w, "capture ref_w");
  check_eq(refs.target_h, literal_refs.target_h, "capture target_h");
  check_eq(refs.target_w, literal_refs.target_w, "capture target_w");

  int ref_mismatch = 0;
  for (int i = 0; i < mh::kMaxRefDim; ++i) {
    for (int j = 0; j < mh::kMaxRefDim; ++j) {
      if (refs.luma[i][j] != literal_refs.luma[i][j] ||
          refs.chroma[i][j] != literal_refs.chroma[i][j]) {
        ++ref_mismatch;
      }
    }
  }
  check_eq(ref_mismatch, 0, "capture samples");

  int count_ref = 0;
  const mh::MultiParams params = mh::derive_multi_params(c.block, refs, dir,
                                                         &count_ref);
  const literal::DeriveResult want =
      literal::derive(c.literal_block, literal_refs, dir);
  check_eq(count_ref, want.count, "derive count");
  for (int k = 0; k < 3; ++k) {
    check_eq(params.p[k], want.params.p[k], "derive parameter");
  }

  std::vector<std::uint8_t> got = c.chroma;
  std::vector<std::uint8_t> expected = c.chroma;
  const int chroma_stride = c.block.chroma_stride;
  mh::Block got_block = c.block;
  got_block.chroma = got.data();
  mh::apply_mhccp(got_block, params, dir, refs);

  for (int i = 0; i < c.block.h; ++i) {
    for (int j = 0; j < c.block.w; ++j) {
      const std::size_t index =
          static_cast<std::size_t>(c.block.y + i) *
              static_cast<std::size_t>(chroma_stride) +
          static_cast<std::size_t>(c.block.x + j);
      expected[index] = static_cast<std::uint8_t>(
          literal::predict(c.literal_block, literal_refs, want.params, dir, i,
                           j));
    }
  }
  int pred_mismatch = 0;
  for (int i = 0; i < c.block.h; ++i) {
    for (int j = 0; j < c.block.w; ++j) {
      const std::size_t index =
          static_cast<std::size_t>(c.block.y + i) *
              static_cast<std::size_t>(chroma_stride) +
          static_cast<std::size_t>(c.block.x + j);
      if (got[index] != expected[index]) {
        if (pred_mismatch == 0 && g_failures < 2) {
          std::printf(
              "  mismatch %s %dx%d at %d,%d: got %d want %d direct %d "
              "params %d %d %d dir %d\n",
              tag, c.block.w, c.block.h, c.block.x + j, c.block.y + i,
              got[index], expected[index],
              mh::predict_sample(c.block, refs, params, dir, i, j),
              params.p[0], params.p[1], params.p[2], dir);
        }
        ++pred_mismatch;
      }
    }
  }
  check_eq(pred_mismatch, 0, tag);
}

void test_synthetic() {
  const int sizes[] = {4, 8, 16, 32};
  int cases = 0;
  for (int w : sizes) {
    for (int h : sizes) {
      for (int y = 0; y <= 64; y += (y < 4 ? 4 : 16)) {
        for (int x = 0; x <= 64; x += (x < 4 ? 4 : 16)) {
          const int coded_w = std::max(64, x + w);
          const int coded_h = std::max(64, y + h);
          const int num4_ar = random_range(0, w / 2 + 2);
          const int num4_bl = random_range(0, h / 2 + 2);
          const int filter = random_range(0, 3);
          Case c = make_case(coded_w, coded_h, w, h, x, y, filter, num4_ar,
                             num4_bl, (cases % 3) == 0);
          for (int dir = 0; dir < 3; ++dir) {
            compare_one(c, dir, "prediction");
          }
          ++cases;
        }
      }
    }
  }
  check(cases > 100, "synthetic case count");
}

void test_conditioning() {
  // count == 0: no above/left -> default {0, 0, 1<<16}.
  {
    Case c = make_case(64, 64, 8, 8, 0, 0, 0, 0, 0, false);
    mh::Refs refs;
    check(mh::capture_refs(c.block, refs), "no-neighbour capture");
    int count = -1;
    const mh::MultiParams p =
        mh::derive_multi_params(c.block, refs, 0, &count);
    check_eq(count, 0, "empty conditioning count");
    check_eq(p.p[0], 0, "empty p0");
    check_eq(p.p[1], 0, "empty p1");
    check_eq(p.p[2], 1 << mh::kBits, "empty p2");
    std::vector<std::uint8_t> plane = c.chroma;
    mh::Block b = c.block;
    b.chroma = plane.data();
    mh::apply_mhccp(b, p, 0, refs);
    for (int i = 0; i < b.h; ++i) {
      for (int j = 0; j < b.w; ++j) {
        check_eq(static_cast<int>(plane[static_cast<std::size_t>(b.y + i) *
                                               kChromaStride +
                                           static_cast<std::size_t>(b.x + j)]),
                 128, "empty prediction is bias");
      }
    }
  }

  // Constant luma / chroma: singular normal equations, Max(1, diag) keeps the
  // elimination finite.
  {
    Case c = make_case(64, 64, 16, 16, 16, 16, 0, 3, 3, false);
    std::fill(c.luma.begin(), c.luma.end(), 128);
    std::fill(c.chroma.begin(), c.chroma.end(), 128);
    for (int dir = 0; dir < 3; ++dir) {
      mh::Refs refs;
      check(mh::capture_refs(c.block, refs), "constant capture");
      int count = 0;
      const mh::MultiParams p =
          mh::derive_multi_params(c.block, refs, dir, &count);
      check(count > 0, "constant conditioning count");
      std::vector<std::uint8_t> plane = c.chroma;
      mh::Block b = c.block;
      b.chroma = plane.data();
      mh::apply_mhccp(b, p, dir, refs);
      int bad = 0;
      for (int i = 0; i < b.h; ++i) {
        for (int j = 0; j < b.w; ++j) {
          const int v = plane[static_cast<std::size_t>(b.y + i) * kChromaStride +
                              static_cast<std::size_t>(b.x + j)];
          // The singular normal equations have a null space; the §7.13.8
          // fixed-point elimination picks one valid solution. Its rounding
          // may be one LSB away from the exact constant.
          if (std::abs(v - 128) > 1) ++bad;
        }
      }
      check_eq(bad, 0, "constant-luma prediction");
    }
  }

  // mul_fixed32_adapt and get_division_scale_shift conditioning. For a
  // positive denom, denom * scale >> shift should be ~65536.
  for (int denom = 1; denom <= 4096; denom = denom < 64 ? denom + 1 : denom * 2) {
    const mh::DivisionScaleShift ds = mh::get_division_scale_shift(denom);
    const int approx = mh::mul_fixed32_adapt(denom, ds.scale, ds.shift);
    check(std::abs(approx - 65536) <= 65536 / 64,
          "division scale approximates 65536");
  }

  // Random mul_fixed32_adapt vs a 64-bit reference.
  for (int n = 0; n < 20000; ++n) {
    const int a = random_range(-70000, 70000);
    const int b = random_range(-70000, 70000);
    const int shift = random_range(0, 20);
    const int bits_a = literal::get_msb_i(literal::abs_i(a)) + 1;
    const int bits_b = literal::get_msb_i(literal::abs_i(b)) + 1;
    const int need = std::max(0, bits_a + bits_b - 29);
    const int s1 = need >> 1;
    const int s2 = need - s1;
    const int adj = shift - (s1 + s2);
    const std::int64_t prod =
        static_cast<std::int64_t>(a >> s1) * (b >> s2);
    std::int64_t want = 0;
    if (adj <= 0) {
      want = prod;
    } else if (adj <= 29) {
      want = prod >= 0 ? (prod + (std::int64_t{1} << (adj - 1))) >> adj
                       : -((-prod + (std::int64_t{1} << (adj - 1))) >> adj);
    }
    const int got = mh::mul_fixed32_adapt(a, b, shift);
    check_eq(got, static_cast<int>(want), "mul_fixed32_adapt");
  }
}

void test_validation_and_allowed() {
  Case c = make_case(64, 64, 8, 8, 8, 8, 0, 0, 0, false);
  check(mh::valid(c.block), "valid block");
  mh::Block bad = c.block;
  bad.sub_x = 0;
  check(!mh::valid(bad), "reject 4:4:4");
  bad = c.block;
  bad.w = 64;
  check(!mh::valid(bad), "reject >32 width");
  bad = c.block;
  bad.luma = nullptr;
  check(!mh::valid(bad), "reject null luma");

  check(!mh::allowed(4, 4, true, false, true, false, false),
        "allowed rejects 4x4 non-lossless");
  check(mh::allowed(4, 4, true, false, true, false, true),
        "allowed accepts lossless 4x4");
  check(mh::allowed(8, 4, true, false, true, false, false),
        "allowed accepts 8x4");
  check(!mh::allowed(8, 4, false, false, true, false, false),
        "allowed honours enable_mhccp");
  check(!mh::allowed(64, 64, true, false, true, false, false),
        "allowed rejects 64x64");
}

void test_search_and_decide() {
  Case c = make_case(128, 128, 16, 16, 32, 32, 1, 2, 2, false);
  std::vector<std::uint8_t> chroma_v = c.chroma;
  mh::Block u = c.block;
  mh::Block v = c.block;
  v.chroma = chroma_v.data();
  // Make the current chroma block the DC prediction for the CfL branch.
  int sum = 0;
  int count = 0;
  for (int i = 0; i < u.w; ++i) {
    sum += chroma_v[static_cast<std::size_t>(u.y - 1) * kChromaStride +
                    static_cast<std::size_t>(u.x + i)];
    ++count;
  }
  for (int j = 0; j < u.h; ++j) {
    sum += chroma_v[static_cast<std::size_t>(u.y + j) * kChromaStride +
                    static_cast<std::size_t>(u.x - 1)];
    ++count;
  }
  const std::uint8_t dc = static_cast<std::uint8_t>((sum + count / 2) / count);
  for (int i = 0; i < u.h; ++i) {
    for (int j = 0; j < u.w; ++j) {
      chroma_v[static_cast<std::size_t>(u.y + i) * kChromaStride +
               static_cast<std::size_t>(u.x + j)] = dc;
    }
  }

  mh::SourcePlane su{};
  su.plane = c.chroma.data();
  su.stride = kChromaStride;
  su.max_x = 127;
  su.max_y = 127;
  mh::SourcePlane sv = su;

  const auto best = mh::search_mhccp(u, v, su, sv);
  check(best.has_value(), "search_mhccp result");
  if (best) {
    check(best->mh_dir >= 0 && best->mh_dir < 3, "search direction");
    for (int dir = 0; dir < 3; ++dir) {
      const auto candidate = mh::evaluate_mhccp(u, v, su, sv, dir);
      check(candidate.has_value(), "evaluate_mhccp result");
      if (candidate) {
        check(best->total_sse() <= candidate->total_sse(),
              "search picks lowest SSE");
      }
    }
    Case literal_case = make_case(128, 128, 16, 16, 32, 32, 1, 2, 2, false);
    // Use the same planes as `c` for the literal comparison.
    literal_case.luma = c.luma;
    literal_case.chroma = c.chroma;
    literal_case.block = u;
    literal_case.block.chroma = c.chroma.data();
    literal_case.literal_block.chroma = c.chroma.data();
    compare_one(literal_case, best->mh_dir, "search prediction");
  }

  cfl::Block cu = cfl::Block{};
  cu.luma = c.block.luma;
  cu.luma_stride = c.block.luma_stride;
  cu.luma_max_x = c.block.luma_max_x;
  cu.luma_max_y = c.block.luma_max_y;
  cu.chroma = c.chroma.data();
  cu.chroma_stride = kChromaStride;
  cu.x = c.block.x;
  cu.y = c.block.y;
  cu.w = c.block.w;
  cu.h = c.block.h;
  cu.coded_width = 128;
  cu.coded_height = 128;
  cu.have_above = c.block.have_above;
  cu.have_left = c.block.have_left;
  cu.bit_depth = 8;
  cu.ds_filter = c.block.ds_filter;
  cu.sub_x = 1;
  cu.sub_y = 1;
  cu.sb_top_y = c.block.sb_luma_y;
  cfl::Block cv = cu;
  std::vector<std::uint8_t> v_plane = chroma_v;
  cv.chroma = v_plane.data();

  cfl::SourcePlane csu{};
  csu.plane = c.chroma.data();
  csu.stride = kChromaStride;
  csu.max_x = 127;
  csu.max_y = 127;
  cfl::SourcePlane csv = csu;

  cfl::Scratch scratch;
  mh::DecisionOptions options;
  const auto decision =
      mh::decide_cfl(cu, cv, csu, csv, 2, 2, 0, 0, options, scratch);
  check(decision.has_value(), "decide_cfl result");
  if (decision) {
    check(decision->cfl_index >= 0 && decision->cfl_index <= 2,
          "decision cfl_index");
    std::vector<std::uint8_t> applied_u = c.chroma;
    std::vector<std::uint8_t> applied_v = chroma_v;
    cfl::Block au = cu;
    cfl::Block av = cv;
    au.chroma = applied_u.data();
    av.chroma = applied_v.data();
    mh::Decision copy = *decision;
    copy.multi_u.chroma = applied_u.data();
    copy.multi_v.chroma = applied_v.data();
    mh::apply_decision(au, av, copy, scratch);
    // Applying the decision must not modify samples outside the block.
    int outside = 0;
    for (int row = 0; row < 128; ++row) {
      for (int col = 0; col < 128; ++col) {
        if (row >= cu.y && row < cu.y + cu.h && col >= cu.x &&
            col < cu.x + cu.w) {
          continue;
        }
        if (applied_u[static_cast<std::size_t>(row) * kChromaStride + static_cast<std::size_t>(col)] !=
            c.chroma[static_cast<std::size_t>(row) * kChromaStride + static_cast<std::size_t>(col)]) {
          ++outside;
        }
      }
    }
    check_eq(outside, 0, "decision does not write outside block");
  }
}

void test_alpha_quantisation() {
  for (int q8 = -600; q8 <= 600; ++q8) {
    const cfl::Alpha alpha = cfl::snap_alpha_q8(q8);
    const int quantised = cfl::alpha_q8(alpha);
    check(quantised >= -256 && quantised <= 256, "quantised alpha range");
    check(quantised % 32 == 0, "quantised alpha step");
    if (q8 != 0) {
      check(alpha.sign != cfl::kSignZero, "nonzero alpha has sign");
    }
  }
  // All signallable pairs survive explicit syntax.
  for (int su = 0; su < 3; ++su) {
    for (int sv = 0; sv < 3; ++sv) {
      for (int mu = 1; mu <= 8; ++mu) {
        for (int mv = 1; mv <= 8; ++mv) {
          cfl::AlphaPair p{};
          p.u.sign = static_cast<std::int8_t>(su);
          p.u.magnitude = static_cast<std::int8_t>(su == 0 ? 0 : mu);
          p.v.sign = static_cast<std::int8_t>(sv);
          p.v.magnitude = static_cast<std::int8_t>(sv == 0 ? 0 : mv);
          if (!cfl::signalable(p)) continue;
          const mh::CflSyntax syntax = mh::make_explicit_syntax(p);
          check(mh::valid_syntax(syntax, true, true), "explicit syntax valid");
          const cfl::Signalling round = cfl::explicit_signalling(
              cfl::pair_from_signalling(cfl::Signalling{syntax.cfl_index,
                                                        syntax.cfl_alpha_signs,
                                                        syntax.cfl_alpha_u,
                                                        syntax.cfl_alpha_v}));
          check_eq(round.alpha_signs, syntax.cfl_alpha_signs,
                   "explicit signs round-trip");
        }
      }
    }
  }
}

void test_syntax_contexts() {
  // Table 8.4 / 8.5: per-cfl_alpha_signs contexts for cfl_alpha_u/v.
  const int expected_u[8] = {-1, -1, 0, 1, 2, 3, 4, 5};
  const int expected_v[8] = {0, 3, -1, 1, 4, -1, 2, 5};
  for (int signs = 0; signs < 8; ++signs) {
    const int sign_u = (signs + 1) / 3;
    const int sign_v = (signs + 1) % 3;
    const int ctx_u = sign_u == 0 ? -1 : (sign_u - 1) * 3 + sign_v;
    const int ctx_v = sign_v == 0 ? -1 : (sign_v - 1) * 3 + sign_u;
    check_eq(ctx_u, expected_u[signs], "cfl_alpha_u context");
    check_eq(ctx_v, expected_v[signs], "cfl_alpha_v context");
  }

  // Size_Group context for 4:2:0 chroma blocks.
  check_eq(mh::size_group_for_chroma_block(4, 4), 1, "size group 4x4");
  check_eq(mh::size_group_for_chroma_block(4, 8), 1, "size group 4x8");
  check_eq(mh::size_group_for_chroma_block(8, 8), 2, "size group 8x8");
  check_eq(mh::size_group_for_chroma_block(16, 16), 3, "size group 16x16");
  check_eq(mh::size_group_for_chroma_block(32, 32), 3, "size group 32x32");
  check_eq(mh::size_group_for_chroma_block(32, 8), 2, "size group 32x8");
  check_eq(mh::size_group_for_chroma_block(32, 4), 2, "size group 32x4");

  // The two CDF arrays without a generated header come from §9.3.
  check_eq(static_cast<int>(mh::kCflMhccpCdf[0]), 15499, "mhccp cdf 0");
  check_eq(static_cast<int>(mh::kCflMhccpCdf[1]), 30, "mhccp cdf rate");
  check_eq(static_cast<int>(mh::kCflMhDirCdf[0][0]), 10923, "mh dir cdf 0");
  check_eq(static_cast<int>(mh::kCflMhDirCdf[3][1]), 21689, "mh dir cdf 3");
}

void test_syntax_round_trip() {
  const int size_groups[4] = {0, 1, 2, 3};
  std::vector<mh::CflSyntax> candidates;
  candidates.push_back(mh::make_derived_syntax());
  for (int dir = 0; dir < 3; ++dir) {
    candidates.push_back(mh::make_multi_syntax(dir));
  }
  for (int signs = 0; signs < 8; ++signs) {
    const int sign_u = (signs + 1) / 3;
    const int sign_v = (signs + 1) % 3;
    for (int au = 0; au < 8; ++au) {
      for (int av = 0; av < 8; ++av) {
        if (sign_u == 0 && au != 0) continue;
        if (sign_v == 0 && av != 0) continue;
        mh::CflSyntax s{};
        s.cfl_index = mh::kIndexExplicit;
        s.cfl_alpha_signs = signs;
        s.cfl_alpha_u = au;
        s.cfl_alpha_v = av;
        candidates.push_back(s);
      }
    }
  }

  for (bool enable_cfl : {false, true}) {
    for (bool allowed : {false, true}) {
      for (int size_group : size_groups) {
        for (const mh::CflSyntax& syntax : candidates) {
          if (!mh::valid_syntax(syntax, enable_cfl, allowed)) {
            continue;
          }
          for (bool disable_update : {false, true}) {
            mh::CflCdfs enc_cdfs;
            mh::CflCdfs dec_cdfs;
            enc_cdfs.init();
            dec_cdfs.init();

            std::vector<std::uint8_t> buffer(4096);
            av2::Encoder encoder;
            check(encoder.init_symbol(buffer.data(), buffer.size()) ==
                      av2::Status::kOk,
                  "encoder init");
            encoder.disable_cdf_update = disable_update ? 1 : 0;
            std::string error;
            check(mh::write_cfl_alphas(encoder, enc_cdfs, syntax, enable_cfl,
                                       allowed, size_group, &error),
                  "write_cfl_alphas");
            std::size_t bytes = 0;
            check(encoder.exit_symbol(&bytes) == av2::Status::kOk,
                  "encoder exit");

            av2::Decoder decoder;
            check(decoder.init_symbol(buffer.data(), buffer.size(), bytes) ==
                      av2::Status::kOk,
                  "decoder init");
            decoder.disable_cdf_update = disable_update ? 1 : 0;
            mh::CflSyntax got{};
            check(mh::read_cfl_alphas(decoder, dec_cdfs, enable_cfl, allowed,
                                      size_group, &got, &error),
                  "read_cfl_alphas");
            check_eq(got.cfl_index, syntax.cfl_index, "syntax index");
            check_eq(got.cfl_mh_dir, syntax.cfl_mh_dir, "syntax direction");
            check_eq(got.cfl_alpha_signs, syntax.cfl_alpha_signs,
                     "syntax signs");
            check_eq(got.cfl_alpha_u, syntax.cfl_alpha_u, "syntax alpha u");
            check_eq(got.cfl_alpha_v, syntax.cfl_alpha_v, "syntax alpha v");
            check(decoder.exit_symbol() == av2::Status::kOk,
                  "decoder exit");
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Real 4:2:0 chroma
// ---------------------------------------------------------------------------

void test_real_y4m(const char* path, bool use_v) {
  if (path == nullptr || !std::filesystem::exists(path)) {
    std::printf("real 4:2:0 test: skipped (no %s)\n",
                path != nullptr ? path : "(null)");
    return;
  }
  av2enc::Y4mReader reader(path);
  std::string error;
  if (!reader.open(error)) {
    std::printf("real 4:2:0 test: cannot open %s: %s\n", path, error.c_str());
    ++g_failures;
    ++g_checks;
    return;
  }
  av2enc::Y4mFrame frame;
  if (reader.read_frame(frame, error) != av2enc::Y4mReadResult::kFrame) {
    std::printf("real 4:2:0 test: cannot read frame: %s\n", error.c_str());
    ++g_failures;
    ++g_checks;
    return;
  }
  const int cw = static_cast<int>(frame.chroma_width());
  const int ch = static_cast<int>(frame.chroma_height());
  check(cw >= 32 && ch >= 32, "real frame chroma size");

  std::vector<std::uint8_t> luma(frame.y().begin(), frame.y().end());
  std::vector<std::uint8_t> chroma = use_v
                                         ? std::vector<std::uint8_t>(
                                               frame.v().begin(), frame.v().end())
                                         : std::vector<std::uint8_t>(
                                               frame.u().begin(), frame.u().end());
  const int luma_stride = static_cast<int>(frame.width());
  const int chroma_stride = cw;

  int blocks = 0;
  const int sizes[] = {4, 8, 16, 32};
  for (int w : sizes) {
    if (w > cw) continue;
    for (int h : sizes) {
      if (h > ch) continue;
      const int step = w <= 8 ? w : 16;
      for (int y = 0; y + h <= ch; y += step) {
        for (int x = 0; x + w <= cw; x += step) {
          Case c;
          c.luma = luma;
          c.chroma = chroma;
          mh::Block& b = c.block;
          b.luma = c.luma.data();
          b.luma_stride = luma_stride;
          b.luma_max_x = static_cast<int>(frame.width()) - 1;
          b.luma_max_y = static_cast<int>(frame.height()) - 1;
          b.chroma = c.chroma.data();
          b.chroma_stride = chroma_stride;
          b.chroma_max_x = cw - 1;
          b.chroma_max_y = ch - 1;
          b.x = x;
          b.y = y;
          b.w = w;
          b.h = h;
          b.bit_depth = 8;
          b.ds_filter = (x + y) & 3;
          b.sub_x = 1;
          b.sub_y = 1;
          b.have_above = y >= 4 ? 1 : 0;
          b.have_left = x >= 4 ? 1 : 0;
          b.num4_above_right = 0;
          b.num4_below_left = 0;
          b.mi_col_end = cw << 1;
          b.mi_row_end = ch << 1;
          b.sb_luma_y = ((y << 1) / 64) * 64;

          literal::Block& lb = c.literal_block;
          lb.luma = b.luma;
          lb.luma_stride = b.luma_stride;
          lb.luma_max_x = b.luma_max_x;
          lb.luma_max_y = b.luma_max_y;
          lb.chroma = b.chroma;
          lb.chroma_stride = b.chroma_stride;
          lb.chroma_max_x = b.chroma_max_x;
          lb.chroma_max_y = b.chroma_max_y;
          lb.x = b.x;
          lb.y = b.y;
          lb.w = b.w;
          lb.h = b.h;
          lb.bit_depth = 8;
          lb.ds_filter = b.ds_filter;
          lb.sub_x = 1;
          lb.sub_y = 1;
          lb.have_above = b.have_above;
          lb.have_left = b.have_left;
          lb.num4_above_right = 0;
          lb.num4_below_left = 0;
          lb.mi_col_end = b.mi_col_end;
          lb.mi_row_end = b.mi_row_end;
          lb.sb_luma_y = b.sb_luma_y;

          compare_one(c, (x + y) % 3, "real prediction");
          ++blocks;
        }
      }
    }
  }
  check(blocks > 50, "real block count");
  std::printf("real 4:2:0 %c: %d blocks checked (%dx%d chroma)\n",
              use_v ? 'V' : 'U', blocks, cw, ch);
}

}  // namespace

int main(int argc, char** argv) {
  test_validation_and_allowed();
  test_conditioning();
  test_alpha_quantisation();
  test_synthetic();
  test_search_and_decide();
  test_syntax_contexts();
  test_syntax_round_trip();

  const char* data64 = argc > 1 ? argv[1] : "tests/data/m0_64x64_1f.y4m";
  const char* data360 = argc > 2 ? argv[2] : "tests/data/m0_640x360_1f.y4m";
  test_real_y4m(data64, false);
  test_real_y4m(data64, true);
  test_real_y4m(data360, false);
  test_real_y4m(data360, true);

  std::printf("mhccp_test: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
