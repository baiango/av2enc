// Test utilities shared by the CPU and CUDA MC tests.
//
// Host-only: builds guard-banded synthetic frames, loads real Y4M frames, and
// provides deterministic patterns.  The CUDA test includes this header and
// adds device upload/compare around the same data.

#ifndef AV2ENC_ENCODER_INTER_MC_MC_TEST_COMMON_H_
#define AV2ENC_ENCODER_INTER_MC_MC_TEST_COMMON_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "codec/inter/mc/mc.h"

namespace av2enc::inter::mc::test {

constexpr std::uint8_t kGuard = 0xEE;
constexpr int kGuardCols = 16;
constexpr int kGuardRows = 4;

struct Rng {
  std::uint32_t state;
  explicit Rng(std::uint32_t seed) : state(seed ? seed : 1u) {}
  std::uint32_t next() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  }
  int uniform(int lo, int hi) {  // inclusive
    return lo + static_cast<int>(next() % static_cast<std::uint32_t>(hi - lo + 1));
  }
};

enum class Pattern {
  kFlat,
  kRampX,
  kRampY,
  kChecker,
  kNoise,
  kSoftGradient,
  kVerticalEdge,
};

inline std::uint8_t pattern_value(Pattern pattern, int y, int x, int salt) {
  switch (pattern) {
    case Pattern::kFlat:
      return static_cast<std::uint8_t>(96 + salt * 7);
    case Pattern::kRampX:
      return static_cast<std::uint8_t>((x * 3 + y + salt) & 0xFF);
    case Pattern::kRampY:
      return static_cast<std::uint8_t>((y * 5 + 2 * x + salt) & 0xFF);
    case Pattern::kChecker:
      return static_cast<std::uint8_t>(
          (((x >> 2) + (y >> 2) + salt) & 1) ? 220 : 32);
    case Pattern::kNoise: {
      std::uint32_t v = static_cast<std::uint32_t>(x) * 73856093u ^
                        static_cast<std::uint32_t>(y) * 19349663u ^
                        static_cast<std::uint32_t>(salt) * 83492791u;
      v ^= v >> 13;
      v *= 0x9E3779B1u;
      v ^= v >> 16;
      return static_cast<std::uint8_t>(v & 0xFF);
    }
    case Pattern::kSoftGradient:
      return static_cast<std::uint8_t>(
          (128 + (x * 40) / 64 - (y * 24) / 64 + salt * 3) & 0xFF);
    case Pattern::kVerticalEdge:
      return static_cast<std::uint8_t>((x / 4 + salt) & 1 ? 240 : 16);
  }
  return 0;
}

// One guard-banded plane: `stride` includes kGuardCols extra columns; the
// allocation includes kGuardRows extra rows.  All guard bytes stay kGuard.
struct PlaneBuf {
  int width = 0;
  int height = 0;
  int stride = 0;
  std::vector<std::uint8_t> data;

  void init(int w, int h) {
    width = w;
    height = h;
    stride = w + kGuardCols;
    data.assign(static_cast<std::size_t>(stride) *
                    static_cast<std::size_t>(h + kGuardRows),
                kGuard);
  }
  std::uint8_t* ptr() { return data.data(); }
  const std::uint8_t* ptr() const { return data.data(); }
  std::uint8_t& at(int y, int x) {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(x)];
  }
  std::uint8_t at(int y, int x) const {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(x)];
  }
  bool guards_intact() const {
    for (int y = 0; y < height; ++y) {
      for (int x = width; x < stride; ++x) {
        if (at(y, x) != kGuard) {
          return false;
        }
      }
    }
    for (int y = height; y < height + kGuardRows; ++y) {
      for (int x = 0; x < stride; ++x) {
        if (at(y, x) != kGuard) {
          return false;
        }
      }
    }
    return true;
  }
};

struct TestFrame {
  int mi_cols = 0;
  int mi_rows = 0;
  PlaneBuf plane[3];

  static TestFrame make(int mi_cols, int mi_rows) {
    TestFrame f;
    f.mi_cols = mi_cols;
    f.mi_rows = mi_rows;
    f.plane[0].init(mi_cols * kMiSize, mi_rows * kMiSize);
    f.plane[1].init((mi_cols * kMiSize) >> 1, (mi_rows * kMiSize) >> 1);
    f.plane[2].init((mi_cols * kMiSize) >> 1, (mi_rows * kMiSize) >> 1);
    return f;
  }

  void fill(Pattern pattern) {
    for (int p = 0; p < 3; ++p) {
      for (int y = 0; y < plane[p].height; ++y) {
        for (int x = 0; x < plane[p].width; ++x) {
          plane[p].at(y, x) =
              pattern_value(pattern, y, x, p * 11 + mi_cols + mi_rows);
        }
      }
    }
  }

  RefFrame ref() const {
    RefFrame r;
    for (int p = 0; p < 3; ++p) {
      r.plane[p].data = plane[p].ptr();
      r.plane[p].stride = plane[p].stride;
    }
    r.mi_cols = mi_cols;
    r.mi_rows = mi_rows;
    return r;
  }

  bool guards_intact() const {
    return plane[0].guards_intact() && plane[1].guards_intact() &&
           plane[2].guards_intact();
  }
};

// A destination block with guard columns/rows.
struct DestBuf {
  int w = 0;
  int h = 0;
  int stride = 0;
  std::vector<std::uint8_t> data;

  void init(int width, int height) {
    w = width;
    h = height;
    stride = w + kGuardCols;
    data.assign(static_cast<std::size_t>(stride) *
                    static_cast<std::size_t>(h + kGuardRows),
                kGuard);
  }
  std::uint8_t* ptr() { return data.data(); }
  const std::uint8_t* ptr() const { return data.data(); }
  std::uint8_t at(int y, int x) const {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(x)];
  }
  bool guards_intact() const {
    for (int y = 0; y < h; ++y) {
      for (int x = w; x < stride; ++x) {
        if (at(y, x) != kGuard) {
          return false;
        }
      }
    }
    for (int y = h; y < h + kGuardRows; ++y) {
      for (int x = 0; x < stride; ++x) {
        if (at(y, x) != kGuard) {
          return false;
        }
      }
    }
    return true;
  }
};

struct Y4mFrame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

inline bool load_y4m(const std::string& path, int max_frames,
                     std::vector<Y4mFrame>* frames) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::string header;
  std::getline(in, header);
  if (header.rfind("YUV4MPEG2", 0) != 0) {
    return false;
  }
  std::istringstream hs(header);
  std::string tok;
  int w = 0;
  int h = 0;
  bool chroma_420 = false;
  while (hs >> tok) {
    if (tok.size() < 2) {
      continue;
    }
    if (tok[0] == 'W') {
      w = std::stoi(tok.substr(1));
    } else if (tok[0] == 'H') {
      h = std::stoi(tok.substr(1));
    } else if (tok[0] == 'C') {
      chroma_420 = (tok == "C420" || tok == "C420jpeg" || tok == "C420mpeg2" ||
                    tok == "C420paldv");
    }
  }
  if (w <= 0 || h <= 0 || !chroma_420) {
    return false;
  }
  const std::size_t y_sz =
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  const std::size_t c_sz = y_sz / 4;
  frames->clear();
  for (int f = 0; f < max_frames; ++f) {
    std::string fh;
    if (!std::getline(in, fh)) {
      break;
    }
    if (fh.rfind("FRAME", 0) != 0) {
      break;
    }
    Y4mFrame fr;
    fr.width = w;
    fr.height = h;
    fr.y.resize(y_sz);
    fr.u.resize(c_sz);
    fr.v.resize(c_sz);
    in.read(reinterpret_cast<char*>(fr.y.data()),
            static_cast<std::streamsize>(y_sz));
    in.read(reinterpret_cast<char*>(fr.u.data()),
            static_cast<std::streamsize>(c_sz));
    in.read(reinterpret_cast<char*>(fr.v.data()),
            static_cast<std::streamsize>(c_sz));
    if (!in) {
      break;
    }
    frames->push_back(std::move(fr));
  }
  return !frames->empty();
}

// Copies a Y4M frame into an MI-aligned, guard-banded test frame.  The MI
// padding replicates the last visible row/column (decoder reference planes
// are MI-aligned).
inline TestFrame frame_from_y4m(const Y4mFrame& src) {
  const int mi_cols = (src.width + kMiSize - 1) / kMiSize;
  const int mi_rows = (src.height + kMiSize - 1) / kMiSize;
  TestFrame f = TestFrame::make(mi_cols, mi_rows);
  const int cw = src.width >> 1;
  const int ch = src.height >> 1;
  for (int y = 0; y < f.plane[0].height; ++y) {
    const int sy = std::min(y, src.height - 1);
    for (int x = 0; x < f.plane[0].width; ++x) {
      const int sx = std::min(x, src.width - 1);
      f.plane[0].at(y, x) =
          src.y[static_cast<std::size_t>(sy) *
                    static_cast<std::size_t>(src.width) +
                static_cast<std::size_t>(sx)];
    }
  }
  for (int p = 1; p <= 2; ++p) {
    const std::vector<std::uint8_t>& sp = (p == 1) ? src.u : src.v;
    for (int y = 0; y < f.plane[p].height; ++y) {
      const int sy = std::min(y, ch - 1);
      for (int x = 0; x < f.plane[p].width; ++x) {
        const int sx = std::min(x, cw - 1);
        f.plane[p].at(y, x) =
            sp[static_cast<std::size_t>(sy) *
                   static_cast<std::size_t>(cw) +
               static_cast<std::size_t>(sx)];
      }
    }
  }
  return f;
}

}  // namespace av2enc::inter::mc::test

#endif  // AV2ENC_ENCODER_INTER_MC_MC_TEST_COMMON_H_
