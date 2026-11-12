// Test utilities for the §7.16 TIP deblocking reference.
//
// Host-only: builds 8-bit 4:2:0 synthetic frames with guard columns/rows,
// applies simple patterns, and provides a conservative write-envelope mask
// derived from the public inspection API.

#ifndef AV2ENC_ENCODER_FILTERS_TIP_DEBLOCK_TIP_DEBLOCK_TEST_COMMON_H_
#define AV2ENC_ENCODER_FILTERS_TIP_DEBLOCK_TIP_DEBLOCK_TEST_COMMON_H_

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "codec/filters/tip_deblock/tip_deblock.h"

namespace av2enc::tip_deblock::test {

constexpr std::uint8_t kGuard = 0xEE;

inline std::size_t pix_index(int y, int stride, int x) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
         static_cast<std::size_t>(x);
}

struct TipTestFrame {
  int mi_cols = 0;
  int mi_rows = 0;
  int guard_cols = 8;
  int guard_rows = 2;
  int stride[3] = {0, 0, 0};
  int height[3] = {0, 0, 0};
  std::vector<std::uint8_t> data[3];
  std::vector<std::uint16_t> tile_cols;
  std::vector<std::uint16_t> tile_rows;

  int luma_width() const { return mi_cols * 4; }
  int luma_height() const { return mi_rows * 4; }
  int plane_width(int plane) const {
    return (plane == 0) ? luma_width() : luma_width() / 2;
  }
  int plane_height(int plane) const {
    return (plane == 0) ? luma_height() : luma_height() / 2;
  }
  std::uint8_t* plane(int plane) { return data[plane].data(); }
  const std::uint8_t* plane(int plane) const { return data[plane].data(); }

  TipDeblockFrame view() {
    TipDeblockFrame frame{};
    for (int plane = 0; plane < 3; ++plane) {
      frame.planes[plane].data = data[plane].data();
      frame.planes[plane].stride = stride[plane];
    }
    frame.mi_cols = mi_cols;
    frame.mi_rows = mi_rows;
    frame.subsampling_x = 1;
    frame.subsampling_y = 1;
    frame.tile_col_starts = tile_cols.empty() ? nullptr : tile_cols.data();
    frame.num_tile_cols = static_cast<int>(tile_cols.size());
    frame.tile_row_starts = tile_rows.empty() ? nullptr : tile_rows.data();
    frame.num_tile_rows = static_cast<int>(tile_rows.size());
    return frame;
  }

  TipDeblockFrame view() const {
    TipDeblockFrame frame{};
    for (int plane = 0; plane < 3; ++plane) {
      frame.planes[plane].data = const_cast<std::uint8_t*>(data[plane].data());
      frame.planes[plane].stride = stride[plane];
    }
    frame.mi_cols = mi_cols;
    frame.mi_rows = mi_rows;
    frame.subsampling_x = 1;
    frame.subsampling_y = 1;
    frame.tile_col_starts = tile_cols.empty() ? nullptr : tile_cols.data();
    frame.num_tile_cols = static_cast<int>(tile_cols.size());
    frame.tile_row_starts = tile_rows.empty() ? nullptr : tile_rows.data();
    frame.num_tile_rows = static_cast<int>(tile_rows.size());
    return frame;
  }
};

inline TipTestFrame make_frame(int mi_cols, int mi_rows) {
  TipTestFrame frame;
  frame.mi_cols = mi_cols;
  frame.mi_rows = mi_rows;
  frame.stride[0] = mi_cols * 4 + frame.guard_cols;
  frame.stride[1] = (mi_cols * 4) / 2 + frame.guard_cols;
  frame.stride[2] = frame.stride[1];
  frame.height[0] = mi_rows * 4 + frame.guard_rows;
  frame.height[1] = (mi_rows * 4) / 2 + frame.guard_rows;
  frame.height[2] = frame.height[1];
  for (int plane = 0; plane < 3; ++plane) {
    frame.data[plane].assign(
        static_cast<std::size_t>(frame.stride[plane]) *
            static_cast<std::size_t>(frame.height[plane]),
        kGuard);
  }
  return frame;
}

template <typename Fn>
inline void fill_plane(TipTestFrame& frame, int plane, Fn fn) {
  const int width = frame.plane_width(plane);
  const int height = frame.plane_height(plane);
  const int stride = frame.stride[plane];
  std::uint8_t* data = frame.plane(plane);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      data[pix_index(y, stride, x)] = static_cast<std::uint8_t>(fn(x, y));
    }
  }
}

inline void fill_flat(TipTestFrame& frame, int value) {
  for (int plane = 0; plane < 3; ++plane) {
    fill_plane(frame, plane, [value](int, int) { return value; });
  }
}

// Step at a *luma* x coordinate: left of the step is `left`, from the step on
// is `right`.  Chroma planes step at chroma_x = luma_x >> 1.
inline void fill_vertical_step(TipTestFrame& frame, int luma_x, int left,
                               int right) {
  for (int plane = 0; plane < 3; ++plane) {
    const int split = (plane == 0) ? luma_x : (luma_x >> 1);
    fill_plane(frame, plane, [split, left, right](int x, int) {
      return x < split ? left : right;
    });
  }
}

inline void fill_horizontal_step(TipTestFrame& frame, int luma_y, int top,
                                 int bottom) {
  for (int plane = 0; plane < 3; ++plane) {
    const int split = (plane == 0) ? luma_y : (luma_y >> 1);
    fill_plane(frame, plane, [split, top, bottom](int, int y) {
      return y < split ? top : bottom;
    });
  }
}

inline void fill_ramp(TipTestFrame& frame) {
  for (int plane = 0; plane < 3; ++plane) {
    const int width = frame.plane_width(plane);
    const int height = frame.plane_height(plane);
    fill_plane(frame, plane, [width, height](int x, int y) {
      return (x * 160) / (width > 0 ? width : 1) + (y * 32) /
                                                     (height > 0 ? height : 1);
    });
  }
}

// High-frequency checker: the side mask must reject these edges.
inline void fill_checker(TipTestFrame& frame) {
  for (int plane = 0; plane < 3; ++plane) {
    fill_plane(frame, plane, [](int x, int y) {
      return (((x + y) & 1) != 0) ? 16 : 239;
    });
  }
}

inline bool guard_intact(const TipTestFrame& frame) {
  for (int plane = 0; plane < 3; ++plane) {
    const int width = frame.plane_width(plane);
    const int height = frame.plane_height(plane);
    const int stride = frame.stride[plane];
    const std::uint8_t* data = frame.plane(plane);
    for (int y = 0; y < height + frame.guard_rows; ++y) {
      const int start = (y < height) ? width : 0;
      for (int x = start; x < stride; ++x) {
        if (data[pix_index(y, stride, x)] != kGuard) {
          return false;
        }
      }
    }
  }
  return true;
}

inline std::size_t count_changed(const TipTestFrame& a,
                                 const TipTestFrame& b) {
  std::size_t changed = 0;
  for (int plane = 0; plane < 3; ++plane) {
    const int width = a.plane_width(plane);
    const int height = a.plane_height(plane);
    const int stride = a.stride[plane];
    const std::uint8_t* pa = a.plane(plane);
    const std::uint8_t* pb = b.plane(plane);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (pa[pix_index(y, stride, x)] != pb[pix_index(y, stride, x)]) {
          ++changed;
        }
      }
    }
  }
  return changed;
}

inline std::size_t count_changed_in_plane(const TipTestFrame& a,
                                          const TipTestFrame& b, int plane) {
  const int width = a.plane_width(plane);
  const int height = a.plane_height(plane);
  const int stride = a.stride[plane];
  const std::uint8_t* pa = a.plane(plane);
  const std::uint8_t* pb = b.plane(plane);
  std::size_t changed = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (pa[pix_index(y, stride, x)] != pb[pix_index(y, stride, x)]) {
        ++changed;
      }
    }
  }
  return changed;
}

inline bool first_diff(const TipTestFrame& a, const TipTestFrame& b, int* plane,
                       int* px, int* py, int* va, int* vb) {
  for (int p = 0; p < 3; ++p) {
    const int width = a.plane_width(p);
    const int height = a.plane_height(p);
    const int stride = a.stride[p];
    const std::uint8_t* pa = a.plane(p);
    const std::uint8_t* pb = b.plane(p);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int xa = pa[pix_index(y, stride, x)];
        const int xb = pb[pix_index(y, stride, x)];
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

inline int max_abs_delta(const TipTestFrame& a, const TipTestFrame& b) {
  int worst = 0;
  for (int plane = 0; plane < 3; ++plane) {
    const int width = a.plane_width(plane);
    const int height = a.plane_height(plane);
    const int stride = a.stride[plane];
    const std::uint8_t* pa = a.plane(plane);
    const std::uint8_t* pb = b.plane(plane);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int va = pa[pix_index(y, stride, x)];
        const int vb = pb[pix_index(y, stride, x)];
        worst = std::max(worst, va > vb ? va - vb : vb - va);
      }
    }
  }
  return worst;
}

inline std::uint64_t checksum(const TipTestFrame& frame) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (int plane = 0; plane < 3; ++plane) {
    const int width = frame.plane_width(plane);
    const int height = frame.plane_height(plane);
    const int stride = frame.stride[plane];
    const std::uint8_t* data = frame.plane(plane);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        hash ^= data[pix_index(y, stride, x)];
        hash *= 1099511628211ULL;
      }
    }
  }
  return hash;
}

// Conservative envelope of samples the traversal is allowed to modify,
// computed from the public inspection API.  A pixel outside this mask that
// changes indicates a traversal or sample-offset bug.
struct WriteMask {
  int width[3] = {0, 0, 0};
  int height[3] = {0, 0, 0};
  std::vector<std::uint8_t> mask[3];

  void reset(const TipTestFrame& frame) {
    for (int plane = 0; plane < 3; ++plane) {
      width[plane] = frame.plane_width(plane);
      height[plane] = frame.plane_height(plane);
      mask[plane].assign(
          static_cast<std::size_t>(width[plane]) *
              static_cast<std::size_t>(height[plane]),
          0);
    }
  }

  void mark(int plane, int x, int y) {
    if (x < 0 || y < 0 || x >= width[plane] || y >= height[plane]) {
      return;
    }
    mask[plane][static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(width[plane]) +
                static_cast<std::size_t>(x)] = 1;
  }

  bool marked(int plane, int x, int y) const {
    if (x < 0 || y < 0 || x >= width[plane] || y >= height[plane]) {
      return false;
    }
    return mask[plane][static_cast<std::size_t>(y) *
                           static_cast<std::size_t>(width[plane]) +
                       static_cast<std::size_t>(x)] != 0;
  }
};

inline void build_write_mask(const TipTestFrame& frame,
                             const TipDeblockOptions& options,
                             WriteMask* out) {
  out->reset(frame);
  const TipSize size = tip_deblock_tip_size(options);
  for (int plane = 0; plane < 3; ++plane) {
    const int sub_x = (plane == 0) ? 0 : 1;
    const int sub_y = (plane == 0) ? 0 : 1;
    const int sw = tip_width(size) >> sub_x;
    const int sh = tip_height(size) >> sub_y;
    const int w = frame.plane_width(plane);
    const int h = frame.plane_height(plane);
    for (int y = 0; y < h; y += 4) {
      for (int x = 0; x < w; x += sw) {
        if (x <= 0) {
          continue;
        }
        const TipDeblockEdgeInfo info =
            tip_deblock_inspect_edge(frame.view(), options, plane, 0, x, y);
        if (!info.apply_filter) {
          continue;
        }
        for (int dy = 0; dy < 4; ++dy) {
          for (int px = x - info.max_width_neg; px < x + info.max_width_pos;
               ++px) {
            out->mark(plane, px, y + dy);
          }
        }
      }
    }
    for (int x = 0; x < w; x += 4) {
      for (int y = 0; y < h; y += sh) {
        if (y <= 0) {
          continue;
        }
        const TipDeblockEdgeInfo info =
            tip_deblock_inspect_edge(frame.view(), options, plane, 1, x, y);
        if (!info.apply_filter) {
          continue;
        }
        for (int dx = 0; dx < 4; ++dx) {
          for (int py = y - info.max_width_neg; py < y + info.max_width_pos;
               ++py) {
            out->mark(plane, x + dx, py);
          }
        }
      }
    }
  }
}

inline bool changed_pixels_outside_mask(const TipTestFrame& before,
                                        const TipTestFrame& after,
                                        const WriteMask& mask, int* plane,
                                        int* px, int* py) {
  for (int p = 0; p < 3; ++p) {
    const int width = before.plane_width(p);
    const int height = before.plane_height(p);
    const int stride = before.stride[p];
    const std::uint8_t* pa = before.plane(p);
    const std::uint8_t* pb = after.plane(p);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (pa[pix_index(y, stride, x)] != pb[pix_index(y, stride, x)] &&
            !mask.marked(p, x, y)) {
          *plane = p;
          *px = x;
          *py = y;
          return true;
        }
      }
    }
  }
  return false;
}

struct Timer {
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  double elapsed_ms() const {
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }
};

}  // namespace av2enc::tip_deblock::test

#endif  // AV2ENC_ENCODER_FILTERS_TIP_DEBLOCK_TIP_DEBLOCK_TEST_COMMON_H_
