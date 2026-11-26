// Adaptive Reduced Noise Refresh (ARNR) — temporal prefilter for the source.
//
// The whole picture sequence is already resident before encoding
// (`run_video_encode` reads every frame into `owned`), so a temporal filter has
// the lookahead it needs without any change to the frame loop: frame i may be
// filtered against i-R..i+R, including *future* frames.
//
// This is a **source prefilter**, not a coding tool: the encoder codes the
// filtered pictures, so the decoder reconstructs them and no syntax changes.
// The metric is still scored against the ORIGINAL source, so any blur is
// charged in full — the filter only wins if removing noise costs less than the
// detail it removes.
//
// Design (deliberately the conservative one):
//   * Integer motion compensation per 16x16 luma block, 3-step search.  Without
//     MC a temporal filter averages unrelated pixels on moving content (blur) or
//     excludes everything (no-op); MC is what makes it adaptive rather than a
//     blur.
//   * Per-pixel temporal inclusion test: a motion-compensated neighbour sample
//     joins the average only when |cur - nb| <= threshold.  Edges and true
//     detail therefore keep the current sample, while small-amplitude temporal
//     noise averages away — this is the "adaptive" in ARNR.
//   * Blend fraction `strength/16` bounds how much of the neighbour mean can
//     replace the current sample, so strength 16 is a full average and 0 is off.
//
// Chroma is filtered with the luma block's motion vector (4:2:0 collocation),
// so the three planes stay registered with each other.
//
// Gate (env only, default off — no CLI flag, no Options field, so the pinned
// streams cannot change):
//   AV2ENC_ARNR=<radius>         temporal reach in frames, 1..3 (unset = off)
//   AV2ENC_ARNR_STRENGTH=<0..16> blend fraction *16 (default 8 = half)
//   AV2ENC_ARNR_THRESH=<0..255>  per-pixel inclusion threshold (default 6)
//   AV2ENC_ARNR_SEARCH=<0..16>   integer ME radius (default 8; 0 = collocated)
//   AV2ENC_ARNR_STATS=1          one stderr line per filtered frame
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "pipeline/video/video_frame.h"

namespace av2enc::video::arnr {

inline bool parse_off(const char* v) {
  if (v == nullptr || *v == '\0') return true;
  if (std::strcmp(v, "0") == 0) return true;
  auto eq = [v](const char* w) {
    for (std::size_t i = 0; w[i] != '\0'; ++i) {
      if (std::tolower(static_cast<unsigned char>(v[i])) != w[i]) return false;
    }
    return v[std::strlen(w)] == '\0';
  };
  return eq("false") || eq("off") || eq("no");
}

// Temporal radius.  Unset / 0 / false turns the whole stage off.
inline int radius() {
  static const int value = [] {
    const char* v = std::getenv("AV2ENC_ARNR");
    if (parse_off(v)) return 0;
    const int r = std::atoi(v);
    return std::clamp(r, 0, 3);
  }();
  return value;
}

inline bool enabled() { return radius() > 0; }

inline int env_int(const char* name, int fallback, int lo, int hi) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(v, &end, 10);
  if (end == v || *end != '\0') return fallback;
  return static_cast<int>(std::clamp<long>(parsed, lo, hi));
}

inline int strength() { return env_int("AV2ENC_ARNR_STRENGTH", 8, 0, 16); }
inline int threshold() { return env_int("AV2ENC_ARNR_THRESH", 6, 0, 255); }
inline int search() { return env_int("AV2ENC_ARNR_SEARCH", 8, 0, 16); }
inline bool stats_on() {
  const char* v = std::getenv("AV2ENC_ARNR_STATS");
  return !parse_off(v);
}

struct Frame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

// Storage for the filtered pictures.  The input spans are non-owning, so the
// filtered planes must outlive the encode call.
struct Storage {
  std::vector<Frame> frames;
  std::size_t filtered = 0;      // frames whose pixels changed
  std::size_t samples_blended = 0;
  std::size_t samples_total = 0;
};

namespace detail {

inline std::int64_t block_sad(const std::uint8_t* a, int a_stride,
                              const std::uint8_t* b, int b_stride, int w,
                              int h) {
  std::int64_t s = 0;
  for (int r = 0; r < h; ++r) {
    const std::uint8_t* ar = a + static_cast<std::ptrdiff_t>(r) * a_stride;
    const std::uint8_t* br = b + static_cast<std::ptrdiff_t>(r) * b_stride;
    for (int c = 0; c < w; ++c) {
      const int d = static_cast<int>(ar[c]) - static_cast<int>(br[c]);
      s += d < 0 ? -d : d;
    }
  }
  return s;
}

// 3-step integer search (radius r, steps r/2, r/4, ... 1).  Returns the best
// (dx, dy) for block (bx, by) in `cur` against `ref`, clamped to the frame.
struct Mv {
  int dx = 0;
  int dy = 0;
};

inline Mv estimate_mv(const std::uint8_t* cur, const std::uint8_t* ref,
                      int stride, int width, int height, int bx, int by,
                      int bw, int bh, int range) {
  Mv best;
  if (range <= 0) return best;
  const std::uint8_t* c = cur + static_cast<std::ptrdiff_t>(by) * stride + bx;
  std::int64_t best_sad = block_sad(c, stride, ref + static_cast<std::ptrdiff_t>(by) * stride + bx, stride, bw, bh);
  int step = range >= 8 ? 4 : (range >= 4 ? 2 : 1);
  while (step >= 1) {
    bool improved = false;
    for (int sy = -1; sy <= 1; ++sy) {
      for (int sx = -1; sx <= 1; ++sx) {
        if (sx == 0 && sy == 0) continue;
        const int cx = best.dx + sx * step;
        const int cy = best.dy + sy * step;
        if (bx + cx < 0 || by + cy < 0 || bx + cx + bw > width ||
            by + cy + bh > height) {
          continue;
        }
        const std::int64_t sad =
            block_sad(c, stride,
                      ref + static_cast<std::ptrdiff_t>(by + cy) * stride +
                          (bx + cx),
                      stride, bw, bh);
        if (sad < best_sad) {
          best_sad = sad;
          best.dx = cx;
          best.dy = cy;
          improved = true;
        }
      }
    }
    if (!improved && step == 1) break;
    step = std::max(1, step / 2);
    if (step == 1 && !improved) break;
  }
  return best;
}

}  // namespace detail

// Filters `frames` in place (into `storage`).  Returns false only on a format
// the stage does not support, so the caller can report it precisely.
inline bool apply(std::vector<VideoInputFrame>* inputs, Storage* storage,
                  std::string* error) {
  const int R = radius();
  if (R <= 0) return true;
  if (inputs == nullptr || inputs->empty()) return true;
  const int blend = strength();
  if (blend <= 0) return true;
  const int thr = threshold();
  const int range = search();

  const std::uint32_t w32 = (*inputs)[0].width;
  const std::uint32_t h32 = (*inputs)[0].height;
  if (w32 % 16U != 0U || h32 % 16U != 0U) {
    *error = "AV2ENC_ARNR needs 16x16-aligned dimensions";
    return false;
  }
  const int width = static_cast<int>(w32);
  const int height = static_cast<int>(h32);
  const std::size_t ysz =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if ((*inputs)[0].y.size() < ysz) {
    *error = "AV2ENC_ARNR is wired for the 8-bit 4:2:0 --video path only";
    return false;
  }
  const std::size_t csz = static_cast<std::size_t>(width / 2) *
                          static_cast<std::size_t>(height / 2);

  const std::size_t n = inputs->size();
  storage->frames.assign(n, Frame{});
  for (std::size_t i = 0; i < n; ++i) {
    Frame& f = storage->frames[i];
    f.width = width;
    f.height = height;
    f.y.assign((*inputs)[i].y.begin(), (*inputs)[i].y.end());
    f.u.assign((*inputs)[i].u.begin(), (*inputs)[i].u.end());
    f.v.assign((*inputs)[i].v.begin(), (*inputs)[i].v.end());
  }

  for (std::size_t i = 0; i < n; ++i) {
    Frame& cur = storage->frames[i];
    std::size_t blended_this = 0;
    std::size_t seen_this = 0;
    // Count how many neighbours actually contribute per pixel, so a pixel seen
    // by two neighbours is averaged over both rather than one at a time.
    std::vector<std::uint16_t> ysum(ysz, 0);
    std::vector<std::uint8_t> ycnt(ysz, 0);
    std::vector<std::uint16_t> usum(csz, 0);
    std::vector<std::uint8_t> ucnt(csz, 0);
    std::vector<std::uint16_t> vsum(csz, 0);
    std::vector<std::uint8_t> vcnt(csz, 0);

    for (int d = -R; d <= R; ++d) {
      if (d == 0) continue;
      const long j = static_cast<long>(i) + d;
      if (j < 0 || j >= static_cast<long>(n)) continue;
      const Frame& nb = storage->frames[static_cast<std::size_t>(j)];
      for (int by = 0; by < height; by += 16) {
        for (int bx = 0; bx < width; bx += 16) {
          const detail::Mv mv = detail::estimate_mv(
              cur.y.data(), nb.y.data(), width, width, height, bx, by, 16, 16,
              range);
          // Luma.
          for (int r = 0; r < 16; ++r) {
            const std::uint8_t* crow =
                cur.y.data() + static_cast<std::ptrdiff_t>(by + r) * width;
            const std::uint8_t* nrow =
                nb.y.data() + static_cast<std::ptrdiff_t>(by + mv.dy + r) * width;
            for (int c = 0; c < 16; ++c) {
              const int dd = static_cast<int>(crow[bx + c]) -
                             static_cast<int>(nrow[bx + mv.dx + c]);
              ++seen_this;
              if (dd <= thr && dd >= -thr) {
                const std::size_t p =
                    static_cast<std::size_t>(by + r) * static_cast<std::size_t>(width) +
                    static_cast<std::size_t>(bx + c);
                ysum[p] = static_cast<std::uint16_t>(
                    ysum[p] + nrow[bx + mv.dx + c]);
                ++ycnt[p];
                ++blended_this;
              }
            }
          }
          // Chroma: same MV, 4:2:0.
          const int cx0 = bx / 2;
          const int cy0 = by / 2;
          const int mvx = mv.dx >> 1;
          const int mvy = mv.dy >> 1;
          for (int r = 0; r < 8; ++r) {
            const std::uint8_t* crow =
                cur.u.data() + static_cast<std::ptrdiff_t>(cy0 + r) * (width / 2);
            const std::uint8_t* nrow =
                nb.u.data() + static_cast<std::ptrdiff_t>(cy0 + mvy + r) * (width / 2);
            for (int c = 0; c < 8; ++c) {
              const int dd = static_cast<int>(crow[cx0 + c]) -
                             static_cast<int>(nrow[cx0 + mvx + c]);
              if (dd <= thr && dd >= -thr) {
                const std::size_t p =
                    static_cast<std::size_t>(cy0 + r) *
                        static_cast<std::size_t>(width / 2) +
                    static_cast<std::size_t>(cx0 + c);
                usum[p] = static_cast<std::uint16_t>(
                    usum[p] + nrow[cx0 + mvx + c]);
                ++ucnt[p];
              }
            }
          }
          for (int r = 0; r < 8; ++r) {
            const std::uint8_t* crow =
                cur.v.data() + static_cast<std::ptrdiff_t>(cy0 + r) * (width / 2);
            const std::uint8_t* nrow =
                nb.v.data() + static_cast<std::ptrdiff_t>(cy0 + mvy + r) * (width / 2);
            for (int c = 0; c < 8; ++c) {
              const int dd = static_cast<int>(crow[cx0 + c]) -
                             static_cast<int>(nrow[cx0 + mvx + c]);
              if (dd <= thr && dd >= -thr) {
                const std::size_t p =
                    static_cast<std::size_t>(cy0 + r) *
                        static_cast<std::size_t>(width / 2) +
                    static_cast<std::size_t>(cx0 + c);
                vsum[p] = static_cast<std::uint16_t>(
                    vsum[p] + nrow[cx0 + mvx + c]);
                ++vcnt[p];
              }
            }
          }
        }
      }
    }

    // Blend the accumulated neighbour mean into the current picture.
    for (std::size_t p = 0; p < ysz; ++p) {
      if (ycnt[p] == 0) continue;
      const int mean = (ysum[p] + ycnt[p] / 2) / ycnt[p];
      const int cval = cur.y[p];
      cur.y[p] = static_cast<std::uint8_t>(
          (cval * (16 - blend) + mean * blend + 8) >> 4);
    }
    for (std::size_t p = 0; p < csz; ++p) {
      if (ucnt[p] != 0) {
        const int mean = (usum[p] + ucnt[p] / 2) / ucnt[p];
        cur.u[p] = static_cast<std::uint8_t>(
            (cur.u[p] * (16 - blend) + mean * blend + 8) >> 4);
      }
      if (vcnt[p] != 0) {
        const int mean = (vsum[p] + vcnt[p] / 2) / vcnt[p];
        cur.v[p] = static_cast<std::uint8_t>(
            (cur.v[p] * (16 - blend) + mean * blend + 8) >> 4);
      }
    }
    storage->samples_blended += blended_this;
    storage->samples_total += seen_this;
    if (blended_this != 0) ++storage->filtered;
  }

  // Re-point the non-owning spans at the storage that now outlives them.
  for (std::size_t i = 0; i < n; ++i) {
    Frame& f = storage->frames[i];
    (*inputs)[i].y = std::span<const std::uint8_t>(f.y.data(), f.y.size());
    (*inputs)[i].u = std::span<const std::uint8_t>(f.u.data(), f.u.size());
    (*inputs)[i].v = std::span<const std::uint8_t>(f.v.data(), f.v.size());
  }

  if (stats_on()) {
    const double frac = storage->samples_total == 0
                            ? 0.0
                            : 100.0 * static_cast<double>(storage->samples_blended) /
                                  static_cast<double>(storage->samples_total);
    std::fprintf(stderr,
                 "arnr: radius=%d strength=%d/%d thresh=%d search=%d "
                 "frames=%zu changed=%zu blended=%.2f%%\n",
                 R, blend, 16, thr, range, n, storage->filtered, frac);
  }
  return true;
}

}  // namespace av2enc::video::arnr
