// Test utilities shared by the CPU and CUDA motion-estimation tests.
//
// Host-only: std::vector / std::string build synthetic frames with guard
// canaries (to catch out-of-bounds reads) and load small 8-bit 4:2:0 Y4M
// frames.  The CUDA test includes this header for the same generated data.

#ifndef AV2ENC_ENCODER_INTER_ME_ME_TEST_COMMON_H_
#define AV2ENC_ENCODER_INTER_ME_ME_TEST_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "codec/inter/me/me.h"

namespace av2enc::me::test {

constexpr std::uint8_t kGuard = 0xA5;

struct TestPlane {
  std::vector<std::uint8_t> buf;
  int stride = 0;
  int width = 0;
  int height = 0;
  int guard_cols = 8;
  int guard_rows = 2;

  void init(int w, int h, int gc = 8, int gr = 2) {
    guard_cols = gc;
    guard_rows = gr;
    width = w;
    height = h;
    stride = w + 2 * gc;
    buf.assign(static_cast<std::size_t>(stride) *
                   static_cast<std::size_t>(h + 2 * gr),
               kGuard);
  }
  std::uint8_t* data() {
    return buf.data() + static_cast<std::size_t>(guard_rows) *
                            static_cast<std::size_t>(stride) +
           static_cast<std::size_t>(guard_cols);
  }
  const std::uint8_t* data() const {
    return buf.data() + static_cast<std::size_t>(guard_rows) *
                            static_cast<std::size_t>(stride) +
           static_cast<std::size_t>(guard_cols);
  }
  std::uint8_t* row(int y) {
    return data() + static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(stride);
  }
  const std::uint8_t* row(int y) const {
    return data() + static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(stride);
  }
  bool guard_intact() const {
    for (int y = 0; y < height + 2 * guard_rows; ++y) {
      for (int x = 0; x < stride; ++x) {
        const bool inside =
            y >= guard_rows && y < guard_rows + height && x >= guard_cols &&
            x < guard_cols + width;
        if (!inside &&
            buf[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(x)] != kGuard) {
          return false;
        }
      }
    }
    return true;
  }
  MeCorePlane view() const {
    MeCorePlane p;
    p.data = data();
    p.stride = stride;
    p.width = width;
    p.height = height;
    return p;
  }
};

struct TestFrame {
  int width = 0;
  int height = 0;
  TestPlane y;
  TestPlane u;
  TestPlane v;

  void init(int w, int h) {
    width = w;
    height = h;
    y.init(w, h);
    const int cw = (w + 1) / 2;
    const int ch = (h + 1) / 2;
    u.init(cw, ch);
    v.init(cw, ch);
  }
  int chroma_width() const { return (width + 1) / 2; }
  int chroma_height() const { return (height + 1) / 2; }
  MeCoreFrame view() const {
    MeCoreFrame f;
    f.plane[0] = y.view();
    f.plane[1] = u.view();
    f.plane[2] = v.view();
    f.width = width;
    f.height = height;
    return f;
  }
  MeCoreRef ref_view() const {
    MeCoreRef r;
    r.plane[0] = y.view();
    r.plane[1] = u.view();
    r.plane[2] = v.view();
    return r;
  }
  bool guard_intact() const {
    return y.guard_intact() && u.guard_intact() && v.guard_intact();
  }
};

inline std::uint32_t xorshift32(std::uint32_t* state) {
  std::uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

inline void fill_random(TestFrame* f, std::uint32_t seed) {
  std::uint32_t s = seed == 0 ? 1u : seed;
  for (int p = 0; p < 3; ++p) {
    TestPlane& pl = p == 0 ? f->y : (p == 1 ? f->u : f->v);
    for (int y = 0; y < pl.height; ++y) {
      for (int x = 0; x < pl.width; ++x) {
        pl.row(y)[x] = static_cast<std::uint8_t>(xorshift32(&s) >> 24);
      }
    }
  }
}

inline void fill_ramp(TestFrame* f, int step_x = 2, int step_y = 1) {
  for (int p = 0; p < 3; ++p) {
    TestPlane& pl = p == 0 ? f->y : (p == 1 ? f->u : f->v);
    for (int y = 0; y < pl.height; ++y) {
      for (int x = 0; x < pl.width; ++x) {
        pl.row(y)[x] = static_cast<std::uint8_t>(
            (x * step_x + y * step_y + (p == 0 ? 16 : 96)) & 0xFF);
      }
    }
  }
}

inline void fill_flat(TestFrame* f, std::uint8_t yv, std::uint8_t uv) {
  for (int p = 0; p < 3; ++p) {
    TestPlane& pl = p == 0 ? f->y : (p == 1 ? f->u : f->v);
    const std::uint8_t v = p == 0 ? yv : uv;
    for (int yy = 0; yy < pl.height; ++yy) {
      for (int xx = 0; xx < pl.width; ++xx) {
        pl.row(yy)[xx] = v;
      }
    }
  }
}

// Low-frequency texture: random coarse grid, bilinearly upsampled.  Unlike
// fill_random(), the matching cost has a basin around the true motion, which
// is what a greedy diamond/TZ search is designed to follow.
inline void fill_smooth_texture(TestFrame* f, std::uint32_t seed, int cell = 8) {
  std::uint32_t s = seed == 0 ? 1u : seed;
  for (int p = 0; p < 3; ++p) {
    TestPlane& pl = p == 0 ? f->y : (p == 1 ? f->u : f->v);
    const int gw = pl.width / cell + 2;
    const int gh = pl.height / cell + 2;
    std::vector<std::uint8_t> grid(static_cast<std::size_t>(gw) *
                                   static_cast<std::size_t>(gh));
    for (std::size_t i = 0; i < grid.size(); ++i) {
      grid[i] = static_cast<std::uint8_t>(xorshift32(&s) >> 24);
    }
    for (int y = 0; y < pl.height; ++y) {
      const int gy = y / cell;
      const int fy = y % cell;
      for (int x = 0; x < pl.width; ++x) {
        const int gx = x / cell;
        const int fx = x % cell;
        const int a = grid[static_cast<std::size_t>(gy) * static_cast<std::size_t>(gw) +
             static_cast<std::size_t>(gx)];
        const int b = grid[static_cast<std::size_t>(gy) * static_cast<std::size_t>(gw) +
             static_cast<std::size_t>(gx + 1)];
        const int c = grid[static_cast<std::size_t>(gy + 1) * static_cast<std::size_t>(gw) +
             static_cast<std::size_t>(gx)];
        const int d = grid[static_cast<std::size_t>(gy + 1) * static_cast<std::size_t>(gw) +
             static_cast<std::size_t>(gx + 1)];
        const int top = (a * (cell - fx) + b * fx + cell / 2) / cell;
        const int bot = (c * (cell - fx) + d * fx + cell / 2) / cell;
        pl.row(y)[x] = static_cast<std::uint8_t>(
            (top * (cell - fy) + bot * fy + cell / 2) / cell);
      }
    }
  }
}

inline int clamp_coord(int lo, int v, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ref(x,y) = cur(x - dx, y - dy) with border replication.  For a block fully
// inside the valid region, the exact-match MV is (dx*16, dy*16).
inline TestFrame make_shifted_ref(const TestFrame& cur, int dx, int dy) {
  TestFrame ref;
  ref.init(cur.width, cur.height);
  for (int p = 0; p < 3; ++p) {
    const TestPlane& src = p == 0 ? cur.y : (p == 1 ? cur.u : cur.v);
    TestPlane& dst = p == 0 ? ref.y : (p == 1 ? ref.u : ref.v);
    const int sx = p == 0 ? dx : dx / 2;
    const int sy = p == 0 ? dy : dy / 2;
    for (int y = 0; y < dst.height; ++y) {
      for (int x = 0; x < dst.width; ++x) {
        const int xx = clamp_coord(0, x - sx, src.width - 1);
        const int yy = clamp_coord(0, y - sy, src.height - 1);
        dst.row(y)[x] = src.row(yy)[xx];
      }
    }
  }
  return ref;
}

// Minimal Y4M reader: up to max_frames, 8-bit 4:2:0 only.
inline bool load_y4m_frames(const std::string& path, int max_frames,
                            std::vector<TestFrame>* out) {
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
      if (tag == 'W') {
        width = std::stoi(value);
      } else if (tag == 'H') {
        height = std::stoi(value);
      } else if (tag == 'C') {
        c420 = value == "420" || value == "420jpeg" || value == "420mpeg2" ||
               value == "420paldv";
      }
    }
  }
  if (width <= 0 || height <= 0 || !c420) {
    return false;
  }
  const int cw = (width + 1) / 2;
  const int ch = (height + 1) / 2;
  const std::size_t y_size =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t c_size =
      static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch);
  std::vector<std::uint8_t> ybuf(y_size);
  std::vector<std::uint8_t> ubuf(c_size);
  std::vector<std::uint8_t> vbuf(c_size);
  for (int frame = 0; frame < max_frames; ++frame) {
    if (!std::getline(in, line)) {
      break;
    }
    if (line.rfind("FRAME", 0) != 0) {
      return false;
    }
    if (!in.read(reinterpret_cast<char*>(ybuf.data()),
                 static_cast<std::streamsize>(y_size)) ||
        !in.read(reinterpret_cast<char*>(ubuf.data()),
                 static_cast<std::streamsize>(c_size)) ||
        !in.read(reinterpret_cast<char*>(vbuf.data()),
                 static_cast<std::streamsize>(c_size))) {
      break;
    }
    TestFrame f;
    f.init(width, height);
    for (int yy = 0; yy < height; ++yy) {
      std::memcpy(f.y.row(yy), ybuf.data() + static_cast<std::size_t>(yy) *
                        static_cast<std::size_t>(width),
                  static_cast<std::size_t>(width));
    }
    for (int yy = 0; yy < ch; ++yy) {
      std::memcpy(f.u.row(yy), ubuf.data() + static_cast<std::size_t>(yy) *
                        static_cast<std::size_t>(cw),
                  static_cast<std::size_t>(cw));
      std::memcpy(f.v.row(yy), vbuf.data() + static_cast<std::size_t>(yy) *
                        static_cast<std::size_t>(cw),
                  static_cast<std::size_t>(cw));
    }
    out->push_back(std::move(f));
  }
  return !out->empty();
}

// Build a raster grid of jobs with the given nominal block size.  The last row
// and column are partial blocks (w/h still nominal, clipped by the search).
inline std::vector<MeCoreJob> make_block_grid(int width, int height, int bw,
                                              int bh) {
  std::vector<MeCoreJob> jobs;
  for (int y = 0; y < height; y += bh) {
    for (int x = 0; x < width; x += bw) {
      MeCoreJob j;
      j.x = x;
      j.y = y;
      j.w = bw;
      j.h = bh;
      j.num_refs = 1;
      j.ref_indices[0] = 0;
      jobs.push_back(j);
    }
  }
  return jobs;
}

}  // namespace av2enc::me::test

#endif  // AV2ENC_ENCODER_INTER_ME_ME_TEST_COMMON_H_
