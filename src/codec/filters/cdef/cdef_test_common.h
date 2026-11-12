// Test utilities shared by the CPU and CUDA CDEF tests.
//
// Host-only: std::vector / std::string are used to build synthetic frames,
// metadata grids and to load a real reconstructed Y4M frame.  The CUDA test
// includes this header and adds device upload/compare around it.

#ifndef AV2ENC_ENCODER_FILTERS_CDEF_CDEF_TEST_COMMON_H_
#define AV2ENC_ENCODER_FILTERS_CDEF_CDEF_TEST_COMMON_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "codec/filters/cdef/cdef.h"

namespace av2enc::cdef::test {

constexpr std::uint8_t kGuard = 0xEE;

inline std::size_t pix_index(int y, int stride, int x) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
         static_cast<std::size_t>(x);
}

enum class Pattern {
  kFlat,
  kVerticalStep,
  kHorizontalStep,
  kDiagonalStep,
  kRampX,
  kRampY,
  kChecker,
  kAlternating,
  kNoise,
  kSoftEdge,
  kRipple,
};

struct TestFrame {
  int mi_cols = 0;
  int mi_rows = 0;
  int guard_cols = 8;
  int guard_rows = 2;
  int n64_cols = 0;
  int n64_rows = 0;
  int stride[3] = {0, 0, 0};
  int height[3] = {0, 0, 0};
  std::vector<std::uint8_t> src[3];
  std::vector<std::uint8_t> dst[3];
  std::vector<std::uint8_t> skip;
  std::vector<std::uint8_t> lossless;
  std::vector<std::uint8_t> lossless_chroma;
  std::vector<std::int8_t> idx;
  std::vector<std::uint16_t> tile_cols;
  std::vector<std::uint16_t> tile_rows;

  int plane_width(int plane) const {
    return (mi_cols * kMiSize) >> ((plane == 0) ? 0 : 1);
  }
  int plane_height(int plane) const {
    return (mi_rows * kMiSize) >> ((plane == 0) ? 0 : 1);
  }
  std::size_t mi_index(int r, int c) const {
    return static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols) +
           static_cast<std::size_t>(c);
  }

  CdefFrame view() {
    CdefFrame f{};
    for (int plane = 0; plane < 3; ++plane) {
      f.planes[plane].src = src[plane].data();
      f.planes[plane].dst = dst[plane].data();
      f.planes[plane].stride = stride[plane];
    }
    f.mi_cols = mi_cols;
    f.mi_rows = mi_rows;
    f.subsampling_x = 1;
    f.subsampling_y = 1;
    f.skip = skip.data();
    f.lossless = lossless.data();
    f.lossless_chroma = lossless_chroma.data();
    f.cdef_idx = idx.data();
    f.n64_cols = n64_cols;
    f.n64_rows = n64_rows;
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
  f.n64_cols = (mi_cols + kCdefSize4 - 1) / kCdefSize4;
  f.n64_rows = (mi_rows + kCdefSize4 - 1) / kCdefSize4;
  for (int plane = 0; plane < 3; ++plane) {
    const int width = f.plane_width(plane);
    const int height = f.plane_height(plane);
    f.stride[plane] = width + guard_cols;
    f.height[plane] = height + guard_rows;
    const std::size_t size =
        static_cast<std::size_t>(f.stride[plane]) *
        static_cast<std::size_t>(f.height[plane]);
    f.src[plane].assign(size, kGuard);
    f.dst[plane].assign(size, kGuard);
  }
  const std::size_t mi_size =
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows);
  f.skip.assign(mi_size, 0);
  f.lossless.assign(mi_size, 0);
  f.lossless_chroma.assign(mi_size, 0);
  f.idx.assign(static_cast<std::size_t>(f.n64_cols) *
                   static_cast<std::size_t>(f.n64_rows),
               static_cast<std::int8_t>(0));
  return f;
}

inline int pattern_value(Pattern p, int x, int y, int w, int h) {
  const int cx = w / 2;
  const int cy = h / 2;
  switch (p) {
    case Pattern::kFlat:
      return 128;
    case Pattern::kVerticalStep:
      return (x < cx) ? 96 : 160;
    case Pattern::kHorizontalStep:
      return (y < cy) ? 160 : 96;
    case Pattern::kDiagonalStep:
      return (x + y < cx + cy) ? 96 : 160;
    case Pattern::kRampX:
      return (x * 160) / (w > 0 ? w : 1) + 48;
    case Pattern::kRampY:
      return (y * 160) / (h > 0 ? h : 1) + 48;
    case Pattern::kChecker:
      return (((x >> 3) + (y >> 3)) & 1) ? 64 : 192;
    case Pattern::kAlternating:
      return ((x + y) & 1) ? 16 : 239;
    case Pattern::kNoise:
      return 16 + ((x * 37 + y * 91 + ((x * y) >> 3)) & 223);
    case Pattern::kSoftEdge:
      // A gentle 45-degree step that directional filtering can smooth.
      return (((x + y) < cx + cy) ? 124 : 132);
    case Pattern::kRipple:
      // Low-amplitude 2D texture: every strength has something to constrain.
      return 126 + ((x * 5 + y * 3 + ((x * y) >> 2)) % 7);
  }
  return 128;
}

inline void fill_pattern(TestFrame& f, Pattern p) {
  for (int plane = 0; plane < 3; ++plane) {
    const int w = f.plane_width(plane);
    const int h = f.plane_height(plane);
    const int stride = f.stride[plane];
    std::uint8_t* data = f.src[plane].data();
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int value = pattern_value(p, x, y, w, h);
        data[pix_index(y, stride, x)] =
            static_cast<std::uint8_t>(value < 0 ? 0 : (value > 255 ? 255
                                                                  : value));
      }
    }
  }
}

// Fill a rectangular region of a source plane with a deterministic texture.
inline void fill_rect(TestFrame& f, int plane, int x0, int y0, int w, int h,
                      int base, int delta) {
  const int stride = f.stride[plane];
  std::uint8_t* data = f.src[plane].data();
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int value = base + ((x * 13 + y * 29 + ((x * y) >> 2)) % delta);
      data[pix_index(y0 + y, stride, x0 + x)] =
          static_cast<std::uint8_t>(value < 0 ? 0 : (value > 255 ? 255
                                                                 : value));
    }
  }
}

inline void set_uniform_metadata(TestFrame& f, std::int8_t index, bool skip,
                                 bool lossless, bool lossless_chroma) {
  std::fill(f.idx.begin(), f.idx.end(), index);
  std::fill(f.skip.begin(), f.skip.end(), skip ? 1 : 0);
  std::fill(f.lossless.begin(), f.lossless.end(), lossless ? 1 : 0);
  std::fill(f.lossless_chroma.begin(), f.lossless_chroma.end(),
            lossless_chroma ? 1 : 0);
}

inline void set_random_metadata(TestFrame& f, std::mt19937& rng,
                                int num_strengths, bool allow_skip,
                                bool allow_lossless, bool allow_tiles) {
  for (std::size_t i = 0; i < f.idx.size(); ++i) {
    const unsigned pick = static_cast<unsigned>(rng() % 20);
    if (pick == 0) {
      f.idx[i] = static_cast<std::int8_t>(-1);
    } else {
      f.idx[i] =
          static_cast<std::int8_t>(rng() % static_cast<unsigned>(
                                              num_strengths));
    }
  }
  for (std::size_t i = 0; i < f.skip.size(); ++i) {
    f.skip[i] = (allow_skip && (rng() % 7) == 0) ? 1 : 0;
    f.lossless[i] = (allow_lossless && (rng() % 41) == 0) ? 1 : 0;
    f.lossless_chroma[i] = (allow_lossless && (rng() % 53) == 0) ? 1 : 0;
  }
  f.tile_cols.clear();
  f.tile_rows.clear();
  if (allow_tiles) {
    f.tile_cols.push_back(0);
    f.tile_rows.push_back(0);
    const int col_parts = 1 + static_cast<int>(rng() % 4);
    const int row_parts = 1 + static_cast<int>(rng() % 4);
    for (int i = 1; i < col_parts && f.mi_cols > 1; ++i) {
      f.tile_cols.push_back(static_cast<std::uint16_t>(
          1 + (rng() % static_cast<unsigned>(f.mi_cols - 1))));
    }
    for (int i = 1; i < row_parts && f.mi_rows > 1; ++i) {
      f.tile_rows.push_back(static_cast<std::uint16_t>(
          1 + (rng() % static_cast<unsigned>(f.mi_rows - 1))));
    }
    std::sort(f.tile_cols.begin(), f.tile_cols.end());
    std::sort(f.tile_rows.begin(), f.tile_rows.end());
    f.tile_cols.erase(std::unique(f.tile_cols.begin(), f.tile_cols.end()),
                      f.tile_cols.end());
    f.tile_rows.erase(std::unique(f.tile_rows.begin(), f.tile_rows.end()),
                      f.tile_rows.end());
  }
}

inline void reset_dst(TestFrame& f) {
  for (int plane = 0; plane < 3; ++plane) {
    std::fill(f.dst[plane].begin(), f.dst[plane].end(), kGuard);
  }
}

// True when every destination byte outside the visible plane is still kGuard.
inline bool guard_intact(const TestFrame& f) {
  for (int plane = 0; plane < 3; ++plane) {
    const int w = f.plane_width(plane);
    const int h = f.plane_height(plane);
    const int stride = f.stride[plane];
    const std::uint8_t* data = f.dst[plane].data();
    for (int y = 0; y < h; ++y) {
      for (int x = w; x < stride; ++x) {
        if (data[pix_index(y, stride, x)] != kGuard) {
          return false;
        }
      }
    }
    for (int y = h; y < f.height[plane]; ++y) {
      for (int x = 0; x < stride; ++x) {
        if (data[pix_index(y, stride, x)] != kGuard) {
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
    const int sa = a.stride[plane];
    const int sb = b.stride[plane];
    const std::uint8_t* pa = a.dst[plane].data();
    const std::uint8_t* pb = b.dst[plane].data();
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (pa[pix_index(y, sa, x)] != pb[pix_index(y, sb, x)]) {
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

inline int max_abs_diff(const TestFrame& a, const TestFrame& b) {
  int worst = 0;
  for (int plane = 0; plane < 3; ++plane) {
    const int w = a.plane_width(plane);
    const int h = a.plane_height(plane);
    const int sa = a.stride[plane];
    const int sb = b.stride[plane];
    const std::uint8_t* pa = a.dst[plane].data();
    const std::uint8_t* pb = b.dst[plane].data();
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int va = pa[pix_index(y, sa, x)];
        const int vb = pb[pix_index(y, sb, x)];
        const int diff = va > vb ? va - vb : vb - va;
        worst = diff > worst ? diff : worst;
      }
    }
  }
  return worst;
}

// Statistics of dst relative to src over the visible plane.
struct DiffStats {
  std::size_t changed = 0;
  double mean_abs = 0.0;  // mean |delta| over changed samples
  int max_abs = 0;
};

inline DiffStats diff_stats(const TestFrame& f) {
  DiffStats stats;
  std::uint64_t sum = 0;
  for (int plane = 0; plane < 3; ++plane) {
    const int w = f.plane_width(plane);
    const int h = f.plane_height(plane);
    const int stride = f.stride[plane];
    const std::uint8_t* src = f.src[plane].data();
    const std::uint8_t* dst = f.dst[plane].data();
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int a = src[pix_index(y, stride, x)];
        const int b = dst[pix_index(y, stride, x)];
        const int diff = a > b ? a - b : b - a;
        if (diff != 0) {
          ++stats.changed;
          sum += static_cast<std::uint64_t>(diff);
          stats.max_abs = diff > stats.max_abs ? diff : stats.max_abs;
        }
      }
    }
  }
  if (stats.changed != 0) {
    stats.mean_abs =
        static_cast<double>(sum) / static_cast<double>(stats.changed);
  }
  return stats;
}

// Minimal Y4M reader: first frame, 8-bit 4:2:0 only.  The returned MI grid uses
// the spec formula MiCols = 2 * ((width + 7) >> 3), so non-multiple-of-8 sizes
// are padded to the MI-aligned plane.
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
  const int mi_cols = 2 * ((width + 7) >> 3);
  const int mi_rows = 2 * ((height + 7) >> 3);
  TestFrame f = make_frame(mi_cols, mi_rows);
  const int cw = width / 2;
  const int ch = height / 2;
  std::vector<std::uint8_t> row(static_cast<std::size_t>(width));
  for (int y = 0; y < height; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), width);
    if (!in) {
      return false;
    }
    std::memcpy(f.src[0].data() +
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(f.stride[0]),
                row.data(), static_cast<std::size_t>(width));
  }
  row.assign(static_cast<std::size_t>(cw), 0);
  for (int y = 0; y < ch; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), cw);
    if (!in) {
      return false;
    }
    std::memcpy(f.src[1].data() +
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(f.stride[1]),
                row.data(), static_cast<std::size_t>(cw));
  }
  for (int y = 0; y < ch; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), cw);
    if (!in) {
      return false;
    }
    std::memcpy(f.src[2].data() +
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(f.stride[2]),
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

}  // namespace av2enc::cdef::test

#endif  // AV2ENC_ENCODER_FILTERS_CDEF_CDEF_TEST_COMMON_H_
