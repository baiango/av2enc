// Test utilities shared by the CPU and CUDA deblocking tests.
//
// Host-only: std::vector / std::string are used to build synthetic frames,
// metadata grids, and to load a real reconstructed Y4M frame.  The CUDA test
// includes this header and adds device upload/compare around it.

#ifndef AV2ENC_ENCODER_FILTERS_DEBLOCK_DEBLOCK_TEST_COMMON_H_
#define AV2ENC_ENCODER_FILTERS_DEBLOCK_DEBLOCK_TEST_COMMON_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "codec/filters/deblock/deblock.h"

namespace av2enc::deblock::test {

constexpr std::uint8_t kGuard = 0xEE;

inline std::size_t pix_index(int y, int stride, int x) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
         static_cast<std::size_t>(x);
}

enum class Pattern {
  kFlat,
  kVerticalStep,
  kHorizontalStep,
  kRampX,
  kRampY,
  kChecker,
  kAlternating,
  kNoise,
  kSoftEdge,
};

struct TestFrame {
  int mi_cols = 0;
  int mi_rows = 0;
  int guard_cols = 8;
  int guard_rows = 2;
  int y_stride = 0;
  int u_stride = 0;
  int v_stride = 0;
  int y_height = 0;
  int u_height = 0;
  int v_height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  std::vector<DeblockMi> mi_y;
  std::vector<DeblockMi> mi_u;
  std::vector<DeblockMi> mi_v;
  std::vector<std::uint16_t> tile_cols;
  std::vector<std::uint16_t> tile_rows;

  int luma_width() const { return mi_cols * kMiSize; }
  int luma_height() const { return mi_rows * kMiSize; }
  int chroma_width() const { return (mi_cols * kMiSize) >> 1; }
  int chroma_height() const { return (mi_rows * kMiSize) >> 1; }

  std::uint8_t* plane(int p) {
    return (p == 0) ? y.data() : (p == 1 ? u.data() : v.data());
  }
  const std::uint8_t* plane(int p) const {
    return (p == 0) ? y.data() : (p == 1 ? u.data() : v.data());
  }
  int stride(int p) const {
    return (p == 0) ? y_stride : (p == 1 ? u_stride : v_stride);
  }
  int plane_width(int p) const {
    return (p == 0) ? luma_width() : chroma_width();
  }
  int plane_height(int p) const {
    return (p == 0) ? luma_height() : chroma_height();
  }
  std::vector<DeblockMi>& mi(int p) {
    return (p == 0) ? mi_y : (p == 1 ? mi_u : mi_v);
  }
  const std::vector<DeblockMi>& mi(int p) const {
    return (p == 0) ? mi_y : (p == 1 ? mi_u : mi_v);
  }

  DeblockFrame view() {
    DeblockFrame f{};
    for (int p = 0; p < 3; ++p) {
      f.planes[p].data = plane(p);
      f.planes[p].stride = stride(p);
      f.mi[p] = mi(p).data();
    }
    f.mi_cols = mi_cols;
    f.mi_rows = mi_rows;
    f.subsampling_x = 1;
    f.subsampling_y = 1;
    f.tile_col_starts = tile_cols.empty() ? nullptr : tile_cols.data();
    f.num_tile_cols = static_cast<int>(tile_cols.size());
    f.tile_row_starts = tile_rows.empty() ? nullptr : tile_rows.data();
    f.num_tile_rows = static_cast<int>(tile_rows.size());
    return f;
  }

  const DeblockFrame view() const {
    DeblockFrame f{};
    for (int p = 0; p < 3; ++p) {
      f.planes[p].data = const_cast<std::uint8_t*>(plane(p));
      f.planes[p].stride = stride(p);
      f.mi[p] = mi(p).data();
    }
    f.mi_cols = mi_cols;
    f.mi_rows = mi_rows;
    f.subsampling_x = 1;
    f.subsampling_y = 1;
    f.tile_col_starts = tile_cols.empty() ? nullptr : tile_cols.data();
    f.num_tile_cols = static_cast<int>(tile_cols.size());
    f.tile_row_starts = tile_rows.empty() ? nullptr : tile_rows.data();
    f.num_tile_rows = static_cast<int>(tile_rows.size());
    return f;
  }
};

inline TestFrame make_frame(int mi_cols, int mi_rows, int guard_cols = 8,
                            int guard_rows = 2) {
  TestFrame f;
  f.mi_cols = mi_cols;
  f.mi_rows = mi_rows;
  f.guard_cols = guard_cols;
  f.guard_rows = guard_rows;
  f.y_stride = mi_cols * kMiSize + guard_cols;
  f.u_stride = (mi_cols * kMiSize) / 2 + guard_cols;
  f.v_stride = f.u_stride;
  f.y_height = mi_rows * kMiSize + guard_rows;
  f.u_height = (mi_rows * kMiSize) / 2 + guard_rows;
  f.v_height = f.u_height;
  f.y.assign(static_cast<std::size_t>(f.y_stride) *
                       static_cast<std::size_t>(f.y_height), kGuard);
  f.u.assign(static_cast<std::size_t>(f.u_stride) *
                       static_cast<std::size_t>(f.u_height), kGuard);
  f.v.assign(static_cast<std::size_t>(f.v_stride) *
                       static_cast<std::size_t>(f.v_height), kGuard);
  f.mi_y.assign(static_cast<std::size_t>(mi_cols) *
                           static_cast<std::size_t>(mi_rows), DeblockMi{});
  f.mi_u.assign(static_cast<std::size_t>(mi_cols) *
                           static_cast<std::size_t>(mi_rows), DeblockMi{});
  f.mi_v.assign(static_cast<std::size_t>(mi_cols) *
                           static_cast<std::size_t>(mi_rows), DeblockMi{});
  return f;
}

inline int pattern_value(Pattern p, int x, int y, int w, int h) {
  const int cx = w / 2;
  const int cy = h / 2;
  switch (p) {
    case Pattern::kFlat:
      return 128;
    case Pattern::kVerticalStep:
      return (x < cx) ? 90 : 170;
    case Pattern::kHorizontalStep:
      return (y < cy) ? 170 : 90;
    case Pattern::kRampX:
      return (x * 160) / (w > 0 ? w : 1) + 48;
    case Pattern::kRampY:
      return (y * 160) / (h > 0 ? h : 1) + 48;
    case Pattern::kChecker:
      return (((x >> 2) + (y >> 2)) & 1) ? 48 : 208;
    case Pattern::kAlternating:
      return ((x + y) & 1) ? 16 : 239;
    case Pattern::kNoise:
      return 16 + ((x * 37 + y * 91 + ((x * y) >> 3)) & 223);
    case Pattern::kSoftEdge: {
      // A gentle step (4-code step) that the wide filters can smooth.
      return (x < cx) ? 126 : 130;
    }
  }
  return 128;
}

inline void fill_pattern(TestFrame& f, Pattern p) {
  for (int plane = 0; plane < 3; ++plane) {
    const int w = f.plane_width(plane);
    const int h = f.plane_height(plane);
    const int stride = f.stride(plane);
    std::uint8_t* data = f.plane(plane);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        data[pix_index(y, stride, x)] =
            static_cast<std::uint8_t>(clip3(0, 255, pattern_value(p, x, y, w, h)));
      }
    }
  }
}

// Uniform coding blocks of `block4` luma MI units with a single transform size.
inline void set_uniform_blocks(TestFrame& f, int block4, TxSize tx,
                               int qindex) {
  if (block4 < 1) {
    block4 = 1;
  }
  for (int plane = 0; plane < 3; ++plane) {
    std::vector<DeblockMi>& grid = f.mi(plane);
    for (int r = 0; r < f.mi_rows; ++r) {
      for (int c = 0; c < f.mi_cols; ++c) {
        const int base_r = (r / block4) * block4;
        const int base_c = (c / block4) * block4;
        const int h4 = std::min(block4, f.mi_rows - base_r);
        const int w4 = std::min(block4, f.mi_cols - base_c);
        DeblockMi& m = grid[pix_index(r, f.mi_cols, c)];
        m = DeblockMi{};
        m.block_row_base = static_cast<std::uint16_t>(base_r);
        m.block_col_base = static_cast<std::uint16_t>(base_c);
        m.block_w4 = static_cast<std::uint8_t>(w4);
        m.block_h4 = static_cast<std::uint8_t>(h4);
        m.sub_pu_size = tx;
        m.tx_size = tx;
        m.tx_col_base = static_cast<std::uint16_t>(base_c);
        m.tx_row_base = static_cast<std::uint16_t>(base_r);
        m.sub_pu_col_base = static_cast<std::uint16_t>(base_c);
        m.sub_pu_row_base = static_cast<std::uint16_t>(base_r);
        m.skip = 0;
        m.region_intra = 1;
        m.lossless = 0;
        m.qindex = static_cast<std::uint8_t>(clip3(0, 255, qindex));
      }
    }
  }
}

// Uniform blocks where the transform grid is smaller than the coding block:
// the caller controls block extent, TX extent, and whether cells are skipped.
inline void set_tx_grid(TestFrame& f, int block4, int tx4, int qindex,
                        bool skip) {
  if (block4 < 1) block4 = 1;
  if (tx4 < 1) tx4 = 1;
  for (int plane = 0; plane < 3; ++plane) {
    std::vector<DeblockMi>& grid = f.mi(plane);
    for (int r = 0; r < f.mi_rows; ++r) {
      for (int c = 0; c < f.mi_cols; ++c) {
        const int base_r = (r / block4) * block4;
        const int base_c = (c / block4) * block4;
        const int tx_r = base_r + ((r - base_r) / tx4) * tx4;
        const int tx_c = base_c + ((c - base_c) / tx4) * tx4;
        const int h4 = std::min(block4, f.mi_rows - base_r);
        const int w4 = std::min(block4, f.mi_cols - base_c);
        DeblockMi& m = grid[pix_index(r, f.mi_cols, c)];
        m = DeblockMi{};
        m.block_row_base = static_cast<std::uint16_t>(base_r);
        m.block_col_base = static_cast<std::uint16_t>(base_c);
        m.block_w4 = static_cast<std::uint8_t>(w4);
        m.block_h4 = static_cast<std::uint8_t>(h4);
        m.sub_pu_size = TxSize::k32x32;
        m.tx_size = TxSize::k8x8;
        m.tx_col_base = static_cast<std::uint16_t>(tx_c);
        m.tx_row_base = static_cast<std::uint16_t>(tx_r);
        m.sub_pu_col_base = static_cast<std::uint16_t>(base_c);
        m.sub_pu_row_base = static_cast<std::uint16_t>(base_r);
        m.skip = skip ? 1 : 0;
        m.region_intra = 1;
        m.lossless = 0;
        m.qindex = static_cast<std::uint8_t>(clip3(0, 255, qindex));
      }
    }
  }
}

inline TxSize tx_from_dims(int w, int h) {
  for (int i = 0; i < kTxSizesAll; ++i) {
    if (kTxWidth[i] == w && kTxHeight[i] == h) {
      return static_cast<TxSize>(i);
    }
  }
  return TxSize::k4x4;
}

// Random, structurally valid metadata: coding blocks tile the grid, transform
// blocks tile each coding block, and every cell points at a base inside the
// cell's own block.  Used by both CPU fuzz tests and CPU-vs-GPU comparisons.
inline int pick_tx_step(int block_dim, bool chroma, std::mt19937& rng) {
  const int options[5] = {1, 2, 4, 8, 16};
  int valid[5];
  int n = 0;
  for (int d : options) {
    if (block_dim % d == 0 && (!chroma || (d % 2) == 0)) {
      valid[n++] = d;
    }
  }
  return valid[rng() % static_cast<unsigned>(n)];
}

inline void set_random_blocks(TestFrame& f, std::mt19937& rng, int plane,
                              bool allow_lossless) {
  std::vector<DeblockMi>& grid = f.mi(plane);
  const int mi_cols = f.mi_cols;
  const int mi_rows = f.mi_rows;
  const bool chroma = (plane > 0);
  // Chroma blocks/transforms are expressed on the luma MI grid; their luma
  // extent must be even (a 4x4 chroma block is 8x8 luma samples = 2 MI).
  const int sizes_luma[5] = {1, 2, 4, 8, 16};
  const int sizes_chroma[5] = {2, 4, 8, 16, 16};
  const int* sizes = chroma ? sizes_chroma : sizes_luma;
  int r = 0;
  while (r < mi_rows) {
    const int bh = std::min(sizes[rng() % 5], mi_rows - r);
    int c = 0;
    while (c < mi_cols) {
      const int bw = std::min(sizes[rng() % 5], mi_cols - c);
      const int tw = pick_tx_step(bw, chroma, rng);
      const int th = pick_tx_step(bh, chroma, rng);
      const int tx_scale = chroma ? 2 : kMiSize;
      const TxSize tx =
          tx_from_dims(tw * tx_scale, th * tx_scale);
      const TxSize sub = tx_from_dims(std::max(tw, th) * tx_scale,
                                      std::max(tw, th) * tx_scale);
      const bool skip = (rng() % 5) == 0;
      const bool region_intra = (rng() % 9) != 0;
      const bool lossless = allow_lossless && ((rng() % 23) == 0);
      const int qindex = static_cast<int>(rng() % 256);
      for (int yy = 0; yy < bh; ++yy) {
        for (int xx = 0; xx < bw; ++xx) {
          DeblockMi& m =
              grid[pix_index(r + yy, mi_cols, c) + static_cast<std::size_t>(xx)];
          m = DeblockMi{};
          m.block_row_base = static_cast<std::uint16_t>(r);
          m.block_col_base = static_cast<std::uint16_t>(c);
          m.block_w4 = static_cast<std::uint8_t>(bw);
          m.block_h4 = static_cast<std::uint8_t>(bh);
          m.sub_pu_size = sub;
          m.tx_size = tx;
          m.tx_col_base = static_cast<std::uint16_t>(c + (xx / tw) * tw);
          m.tx_row_base = static_cast<std::uint16_t>(r + (yy / th) * th);
          m.sub_pu_col_base = static_cast<std::uint16_t>(c);
          m.sub_pu_row_base = static_cast<std::uint16_t>(r);
          m.skip = skip ? 1 : 0;
          m.region_intra = region_intra ? 1 : 0;
          m.lossless = lossless ? 1 : 0;
          m.qindex = static_cast<std::uint8_t>(qindex);
        }
      }
      c += bw;
    }
    r += bh;
  }
}

inline void set_random_all_planes(TestFrame& f, std::mt19937& rng) {
  for (int p = 0; p < 3; ++p) {
    set_random_blocks(f, rng, p, /*allow_lossless=*/true);
  }
}

inline void set_random_tiles(TestFrame& f, std::mt19937& rng) {
  f.tile_cols.clear();
  f.tile_rows.clear();
  f.tile_cols.push_back(0);
  f.tile_rows.push_back(0);
  const int col_parts = 1 + static_cast<int>(rng() % 4);
  const int row_parts = 1 + static_cast<int>(rng() % 4);
  if (f.mi_cols > 1) {
    for (int i = 1; i < col_parts; ++i) {
      f.tile_cols.push_back(static_cast<std::uint16_t>(
          1 + (rng() % static_cast<unsigned>(f.mi_cols - 1))));
    }
  }
  if (f.mi_rows > 1) {
    for (int i = 1; i < row_parts; ++i) {
      f.tile_rows.push_back(static_cast<std::uint16_t>(
          1 + (rng() % static_cast<unsigned>(f.mi_rows - 1))));
    }
  }
  std::sort(f.tile_cols.begin(), f.tile_cols.end());
  std::sort(f.tile_rows.begin(), f.tile_rows.end());
  f.tile_cols.erase(std::unique(f.tile_cols.begin(), f.tile_cols.end()),
                    f.tile_cols.end());
  f.tile_rows.erase(std::unique(f.tile_rows.begin(), f.tile_rows.end()),
                    f.tile_rows.end());
}

inline bool guard_intact(const TestFrame& f) {
  for (int plane = 0; plane < 3; ++plane) {
    const int w = f.plane_width(plane);
    const int h = f.plane_height(plane);
    const int stride = f.stride(plane);
    const std::uint8_t* data = f.plane(plane);
    for (int yy = 0; yy < h; ++yy) {
      for (int xx = w; xx < stride; ++xx) {
        if (data[pix_index(yy, stride, xx)] != kGuard) {
          return false;
        }
      }
    }
    for (int yy = h; yy < f.plane_height(plane) + f.guard_rows; ++yy) {
      for (int xx = 0; xx < stride; ++xx) {
        if (data[pix_index(yy, stride, xx)] != kGuard) {
          return false;
        }
      }
    }
  }
  return true;
}

inline std::size_t count_changed(const TestFrame& a, const TestFrame& b) {
  std::size_t changed = 0;
  for (int plane = 0; plane < 3; ++plane) {
    const int w = a.plane_width(plane);
    const int h = a.plane_height(plane);
    const int sa = a.stride(plane);
    const int sb = b.stride(plane);
    const std::uint8_t* pa = a.plane(plane);
    const std::uint8_t* pb = b.plane(plane);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (pa[pix_index(y, sa, x)] !=
            pb[pix_index(y, sb, x)]) {
          ++changed;
        }
      }
    }
  }
  return changed;
}

inline std::size_t count_mismatch(const TestFrame& a, const TestFrame& b) {
  return count_changed(a, b);
}

inline std::uint64_t checksum(const TestFrame& f) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (int plane = 0; plane < 3; ++plane) {
    const int w = f.plane_width(plane);
    const int h = f.plane_height(plane);
    const int stride = f.stride(plane);
    const std::uint8_t* data = f.plane(plane);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        hash ^= data[pix_index(y, stride, x)];
        hash *= 1099511628211ULL;
      }
    }
  }
  return hash;
}

inline double mean_abs_delta(const TestFrame& a, const TestFrame& b) {
  std::uint64_t sum = 0;
  std::uint64_t n = 0;
  for (int plane = 0; plane < 3; ++plane) {
    const int w = a.plane_width(plane);
    const int h = a.plane_height(plane);
    const int sa = a.stride(plane);
    const int sb = b.stride(plane);
    const std::uint8_t* pa = a.plane(plane);
    const std::uint8_t* pb = b.plane(plane);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int va = pa[pix_index(y, sa, x)];
        const int vb = pb[pix_index(y, sb, x)];
        sum += static_cast<std::uint64_t>(va > vb ? va - vb : vb - va);
        ++n;
      }
    }
  }
  return n ? static_cast<double>(sum) / static_cast<double>(n) : 0.0;
}

inline bool first_mismatch(const TestFrame& a, const TestFrame& b, int* plane,
                           int* px, int* py, int* va, int* vb) {
  for (int p = 0; p < 3; ++p) {
    const int w = a.plane_width(p);
    const int h = a.plane_height(p);
    const int sa = a.stride(p);
    const int sb = b.stride(p);
    const std::uint8_t* pa = a.plane(p);
    const std::uint8_t* pb = b.plane(p);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int xa = pa[pix_index(y, sa, x)];
        const int xb = pb[pix_index(y, sb, x)];
        if (xa != xb) {
          *plane = p;
          *px = x;
          *py = y;
          *va = xa;
          *vb = xb;
          return true;
        }
      }
    }
  }
  return false;
}

inline int max_abs_delta(const TestFrame& a, const TestFrame& b) {  int worst = 0;
  for (int plane = 0; plane < 3; ++plane) {
    const int w = a.plane_width(plane);
    const int h = a.plane_height(plane);
    const int sa = a.stride(plane);
    const int sb = b.stride(plane);
    const std::uint8_t* pa = a.plane(plane);
    const std::uint8_t* pb = b.plane(plane);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int va = pa[pix_index(y, sa, x)];
        const int vb = pb[pix_index(y, sb, x)];
        worst = std::max(worst, va > vb ? va - vb : vb - va);
      }
    }
  }
  return worst;
}

// Minimal Y4M reader: first frame, 8-bit 4:2:0 only.
inline bool load_y4m_first_frame(const std::string& path, TestFrame* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::string line;
  if (!std::getline(in, line) || line.rfind("YUV4MPEG2", 0) != 0) {
    return false;
  }
  std::istringstream header(line);
  std::string token;
  int width = 0;
  int height = 0;
  bool c420 = false;
  while (header >> token) {
    if (token.size() > 1) {
      const char tag = token[0];
      const std::string value = token.substr(1);
      if (tag == 'W') width = std::stoi(value);
      if (tag == 'H') height = std::stoi(value);
      if (tag == 'C' && value == "420") c420 = true;
    }
  }
  if (width <= 0 || height <= 0 || !c420 || (width % 2) != 0 ||
      (height % 2) != 0) {
    return false;
  }
  if (!std::getline(in, line) || line.rfind("FRAME", 0) != 0) {
    return false;
  }
  const int mi_cols = (width + kMiSize - 1) / kMiSize;
  const int mi_rows = (height + kMiSize - 1) / kMiSize;
  TestFrame f = make_frame(mi_cols, mi_rows);
  const int cw = width / 2;
  const int ch = height / 2;
  std::vector<std::uint8_t> row(static_cast<std::size_t>(width));
  for (int y = 0; y < height; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), width);
    if (!in) return false;
    std::memcpy(f.y.data() + static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(f.y_stride),
                row.data(), static_cast<std::size_t>(width));
  }
  row.assign(static_cast<std::size_t>(cw), 0);
  for (int y = 0; y < ch; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), cw);
    if (!in) return false;
    std::memcpy(f.u.data() + static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(f.u_stride),
                row.data(), static_cast<std::size_t>(cw));
  }
  for (int y = 0; y < ch; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), cw);
    if (!in) return false;
    std::memcpy(f.v.data() + static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(f.v_stride),
                row.data(), static_cast<std::size_t>(cw));
  }
  *out = std::move(f);
  return true;
}

struct Timer {
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  double elapsed_ms() const {
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }
};

}  // namespace av2enc::deblock::test

#endif  // AV2ENC_ENCODER_FILTERS_DEBLOCK_DEBLOCK_TEST_COMMON_H_
