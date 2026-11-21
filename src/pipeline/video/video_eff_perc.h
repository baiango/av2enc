// Extended perceptual RD core (AV2ENC_EFF_PERC).
//
// Builds on the merged pipe-perc path (`AV2ENC_PIPE_PERC`,
// docs/pipe-perc.md): exact inter syntax from the live-CDF probe, top-N
// real-path re-scoring, one lambda for the leaf and the TX candidate.  This
// gate replaces the two pieces pipe-perc left open:
//
//   D  a local masking / CSF distortion instead of the 8x8 source-gradient
//      cell weight.  The map is per-pixel and multi-scale (fine / mid /
//      low-mid band energies from box-blur differences), with optional
//      coherent edge emphasis.  Detail and edges stay up-weighted; there is
//      no smooth-cell down-weighting.  Chroma distortion is coupled into the
//      same D with a plane weight, so the leaf, skip/residual and intra/inter
//      decisions price U/V against luma on one scale.
//   R  the same exact inter syntax probe, plus a selectable coefficient term
//      (the static §8.3 context model from pipe-perc or the R19 range-coder
//      interval width against the live frame-start CDF image).
//   lambda  one refit weight for both the leaf J and the TX candidate J;
//      `AV2ENC_EFF_PERC_SCALE` is the sweep knob.
//
// Gate (env only, no CLI flag), default off:
//   AV2ENC_EFF_PERC=1              master gate
//   AV2ENC_EFF_PERC_SCALE=<x>      lambda multiplier (default 0.18)
//   AV2ENC_EFF_PERC_DSHAPE=<n>     0 neutral source weights (default; the
//                                  error-domain CSF carries D), 1 pipe-perc
//                                  8x8 gradient, 2 multi-scale CSF,
//                                  3 CSF + coherent edge, 4 8x8 variance +
//                                  edge, 5 coherent edge only
//   AV2ENC_EFF_PERC_DETAIL=<x>     detail/edge gain multiplier (default 1)
//   AV2ENC_EFF_PERC_CSF=<x>        error low/mid-band CSF gain (default 1.5)
//   AV2ENC_EFF_PERC_CHROMA=<x>     chroma distortion weight (default 1)
//   AV2ENC_EFF_PERC_TOP=<n>        cheap inter candidates re-scored (default 2)
//   AV2ENC_EFF_PERC_RATE=<n>       bit 0: R19 interval width for the leaf
//                                  plan (default 1); bit 1: live frame-start
//                                  coefficient CDF rows in the TX search and
//                                  the static leaf model.  0 = static §9.3,
//                                  1 = interval, 2 = live slice, 3 = both.
//
// Unset (or 0/false/off) does no work: no weight map is built, no CDF store is
// copied and every pipe-perc / historical branch runs unchanged, so the pins
// stay byte-identical.
#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pipeline/video/video_pipe_ba.h"

namespace av2enc::video::eff_perc {

inline bool parse_off(const char* value) {
  if (value == nullptr || *value == '\0') return true;
  if (std::strcmp(value, "0") == 0) return true;
  auto equal = [value](const char* word) {
    for (std::size_t i = 0; word[i] != '\0'; ++i) {
      if (std::tolower(static_cast<unsigned char>(value[i])) != word[i]) {
        return false;
      }
    }
    return value[std::strlen(word)] == '\0';
  };
  return equal("false") || equal("off") || equal("no");
}

// Process-wide CLI flag set by the InterFrameEncoder constructor from
// VideoOptions::eff_perc (--video-eff-perc / --nightly-b).
inline std::atomic<bool>& cli_flag() {
  static std::atomic<bool> value{false};
  return value;
}

inline bool enabled() {
  static const bool env = [] {
    // AV2ENC_PIPE_BA is a superset pipeline: it reuses this exact-rate front
    // end (the live-CDF probe, the R19 interval coefficient term, top-2
    // re-scoring) and the keyframe intra TX surface, and replaces the CSF with
    // its own Butteraugli-shaped model.  `video_frame_decision.cpp` branches on
    // PIPE_BA first, so this alias never makes the eff-perc *distortion* run.
    return !parse_off(std::getenv("AV2ENC_EFF_PERC")) ||
           pipe_ba::enabled();
  }();
  return env || cli_flag().load();
}

inline double env_double(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || !std::isfinite(parsed) || parsed <= 0.0) return fallback;
  return parsed;
}

inline double lambda_scale() {
  static const double value = env_double("AV2ENC_EFF_PERC_SCALE", 0.18);
  return value;
}

inline double detail_scale() {
  static const double value = [] {
    const char* env = std::getenv("AV2ENC_EFF_PERC_DETAIL");
    if (env == nullptr || *env == '\0') return 1.0;
    char* end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || !std::isfinite(parsed) || parsed < 0.0) return 1.0;
    return parsed;
  }();
  return value;
}

inline double chroma_weight() {
  static const double value = env_double("AV2ENC_EFF_PERC_CHROMA", 1.0);
  return value;
}

// Error-domain CSF gain: D = sum(w * (e^2 + g * e_low^2)) / 256, where e_low
// is the 3x3 box mean of the error.  Low/mid-frequency error is the band the
// measured Butteraugli gap lives in; g > 0 prices it above the high-frequency
// tail.  0 keeps the plain weighted SSE.
inline double csf_gain() {
  // PIPE_BA replaces the CSF: the low-band gain is its Butteraugli-shaped
  // gain, scaled by the tool weight (0 at the tool-off base arm).
  if (pipe_ba::enabled()) return pipe_ba::config().low_band_gain;
  static const double value = [] {
    const char* env = std::getenv("AV2ENC_EFF_PERC_CSF");
    if (env == nullptr || *env == '\0') return 1.5;
    char* end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || !std::isfinite(parsed) || parsed < 0.0) return 1.5;
    return parsed;
  }();
  return value;
}

// Keyframe CSF gain.  The key is intra and carries ~73% of the bytes at the
// 240p point, so its TX search has its own gain.  Default follows the inter
// gain; 0 is the plain-SSE key ablation.
inline double key_csf_gain() {
  if (pipe_ba::enabled()) return pipe_ba::config().low_band_gain;
  static const double value = [] {
    const char* env = std::getenv("AV2ENC_EFF_PERC_KEY_CSF");
    if (env == nullptr || *env == '\0') return csf_gain();
    char* end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || !std::isfinite(parsed) || parsed < 0.0) return csf_gain();
    return parsed;
  }();
  return value;
}

// Second, lower CSF band: the square of the 7x7 box mean error.  The 3x3 term
// is high-mid; the 7x7 term is the 2-4 cycles/degree band Butteraugli's low/mid
// gap lives in.  0 keeps the single-band model.
inline double csf2_gain() {
  // PIPE_BA has no second band at this stage: the Butteraugli shape is the
  // masked 3x3 low/mid term, measured separately from the 7x7 band that
  // eff-perc found negative.
  if (pipe_ba::enabled()) return 0.0;
  static const double value = [] {
    const char* env = std::getenv("AV2ENC_EFF_PERC_CSF2");
    if (env == nullptr || *env == '\0') return 0.0;
    char* end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || !std::isfinite(parsed) || parsed < 0.0) return 0.0;
    return parsed;
  }();
  return value;
}

inline int shape() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_EFF_PERC_DSHAPE");
    if (env == nullptr || *env == '\0') return 0;
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env) return 0;
    if (parsed < 0) return 0;
    if (parsed > 5) return 5;
    return static_cast<int>(parsed);
  }();
  return value;
}

inline int top_n() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_EFF_PERC_TOP");
    if (env == nullptr || *env == '\0') return 2;
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env) return 2;
    if (parsed < 1) return 1;
    if (parsed > 8) return 8;
    return static_cast<int>(parsed);
  }();
  return value;
}

// Bit 0 = R19 interval width, bit 1 = live frame-start CDF rows.
inline int rate_mode() {
  // PIPE_BA reuses exactly this front end: interval width on, live slice off.
  if (pipe_ba::enabled()) return 1;
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_EFF_PERC_RATE");
    if (env == nullptr || *env == '\0') return 1;
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed < 0) return 1;
    return parsed > 3 ? 3 : static_cast<int>(parsed);
  }();
  return value;
}

// Weight range, Q8: 256 is neutral.  The ceiling keeps the map inside the
// pipeline's integer contract and bounds the rate a detail cell can claim.
constexpr int kMinWeightQ8 = 256;
constexpr int kMaxWeightQ8 = 1280;
constexpr int kDetailGain = 384;
constexpr int kDetailRef = 8;

// Per-pixel luma weights in Q8 raster order (256 neutral).
struct FrameWeights {
  int width = 0;
  int height = 0;
  std::vector<std::uint16_t> pix;

  bool valid() const {
    return width > 0 && height > 0 &&
           pix.size() == static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  }
};

// Edge-clamped pixel fetch.
inline int at(const std::uint8_t* src, int stride, int width, int height, int x, int y) {
  x = x < 0 ? 0 : (x >= width ? width - 1 : x);
  y = y < 0 ? 0 : (y >= height ? height - 1 : y);
  return src[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
             static_cast<std::size_t>(x)];
}

// Separable box sum of radius `r`, edges clamped, output at every pixel is
// the (2r+1)^2 window sum.  Deterministic integer arithmetic.
inline void box_sum(const std::uint8_t* src, int stride, int width, int height, int r,
                    std::vector<std::int32_t>* out) {
  out->assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
  if (r <= 0 || width <= 0 || height <= 0) return;
  std::vector<std::int32_t> rows(static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                                 0);
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* row = src + static_cast<std::size_t>(y) * static_cast<std::size_t>(stride);
    std::int32_t running = 0;
    for (int x = -r; x <= r; ++x) {
      const int px = x < 0 ? 0 : (x >= width ? width - 1 : x);
      running += row[px];
    }
    for (int x = 0; x < width; ++x) {
      rows[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x)] = running;
      const int add = x + r + 1 >= width ? width - 1 : x + r + 1;
      const int sub = x - r < 0 ? 0 : x - r;
      running += row[add] - row[sub];
    }
  }
  for (int x = 0; x < width; ++x) {
    std::int32_t running = 0;
    for (int y = -r; y <= r; ++y) {
      const int py = y < 0 ? 0 : (y >= height ? height - 1 : y);
      running += rows[static_cast<std::size_t>(py) * static_cast<std::size_t>(width) +
                      static_cast<std::size_t>(x)];
    }
    for (int y = 0; y < height; ++y) {
      (*out)[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(x)] = running;
      const int add = y + r + 1 >= height ? height - 1 : y + r + 1;
      const int sub = y - r < 0 ? 0 : y - r;
      running += rows[static_cast<std::size_t>(add) * static_cast<std::size_t>(width) +
                      static_cast<std::size_t>(x)] -
                 rows[static_cast<std::size_t>(sub) * static_cast<std::size_t>(width) +
                      static_cast<std::size_t>(x)];
    }
  }
}

// 3x3 mean of an int activity map, rounded.
inline void smooth3(const std::vector<std::int32_t>& in, int width, int height,
                    std::vector<std::int32_t>* out) {
  out->assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
  std::vector<std::int32_t> rows(static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                                 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int xm = x > 0 ? x - 1 : 0;
      const int xp = x + 1 < width ? x + 1 : width - 1;
      rows[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x)] =
          in[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(xm)] +
          in[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(x)] +
          in[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(xp)];
    }
  }
  for (int y = 0; y < height; ++y) {
    const int ym = y > 0 ? y - 1 : 0;
    const int yp = y + 1 < height ? y + 1 : height - 1;
    for (int x = 0; x < width; ++x) {
      const std::int32_t sum = rows[static_cast<std::size_t>(ym) * static_cast<std::size_t>(width) +
                                    static_cast<std::size_t>(x)] +
                               rows[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                    static_cast<std::size_t>(x)] +
                               rows[static_cast<std::size_t>(yp) * static_cast<std::size_t>(width) +
                                    static_cast<std::size_t>(x)];
      (*out)[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(x)] = (sum + 4) / 9;
    }
  }
}

inline int clamp_weight(int value) {
  if (value < kMinWeightQ8) return kMinWeightQ8;
  if (value > kMaxWeightQ8) return kMaxWeightQ8;
  return value;
}

// Shape 1: the merged pipe-perc 8x8 mean-gradient cell weight, expanded to
// pixels.  Kept as the ablation anchor, not the default.
inline void build_cell_gradient(const std::uint8_t* src, int stride, int width, int height,
                                int gain, std::vector<std::uint16_t>* out) {
  const int cols = (width + 7) / 8;
  const int rows = (height + 7) / 8;
  for (int cy = 0; cy < rows; ++cy) {
    for (int cx = 0; cx < cols; ++cx) {
      const int sx = cx * 8;
      const int sy = cy * 8;
      int activity = 0;
      for (int y = 0; y < 8; ++y) {
        const int py = std::min(sy + y, height - 1);
        const int py_next = std::min(py + 1, height - 1);
        for (int x = 0; x < 8; ++x) {
          const int px = std::min(sx + x, width - 1);
          const int px_next = std::min(px + 1, width - 1);
          const int v = at(src, stride, width, height, px, py);
          activity += std::abs(v - at(src, stride, width, height, px_next, py));
          activity += std::abs(v - at(src, stride, width, height, px, py_next));
        }
      }
      const int detail = activity / 64;
      const int weight = clamp_weight(256 + (gain * detail) / (detail + 24));
      for (int y = 0; y < 8; ++y) {
        const int py = std::min(sy + y, height - 1);
        for (int x = 0; x < 8; ++x) {
          const int px = std::min(sx + x, width - 1);
          (*out)[static_cast<std::size_t>(py) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(px)] = static_cast<std::uint16_t>(weight);
        }
      }
    }
  }
}

// Shapes 2/3: per-pixel multi-scale contrast.  The three bands are the
// differences of 3x3 / 7x7 / 15x15 box means; the mid and low-mid bands carry
// the extra weight because that is where the measured Butteraugli sensitivity
// peaks on this clip.  Shape 3 adds the coherent (oriented) part of the local
// gradient so edges claim more than isotropic texture.
inline void build_multiscale(const std::uint8_t* src, int stride, int width, int height, int gain,
                             bool edge_boost, std::vector<std::uint16_t>* out,
                             std::vector<std::int32_t>* scratch_a,
                             std::vector<std::int32_t>* scratch_b,
                             std::vector<std::int32_t>* scratch_c) {
  std::vector<std::int32_t>& s1 = *scratch_a;
  std::vector<std::int32_t>& s3 = *scratch_b;
  std::vector<std::int32_t>& s7 = *scratch_c;
  box_sum(src, stride, width, height, 1, &s1);
  box_sum(src, stride, width, height, 3, &s3);
  box_sum(src, stride, width, height, 7, &s7);
  std::vector<std::int32_t> act(static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                                0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                            static_cast<std::size_t>(x);
      const int p = at(src, stride, width, height, x, y);
      const int m1 = (s1[i] + 4) / 9;
      const int m3 = (s3[i] + 24) / 49;
      const int m7 = (s7[i] + 112) / 225;
      const int high = std::abs(p - m1);
      const int mid = std::abs(m1 - m3);
      const int low = std::abs(m3 - m7);
      // Contrast-sensitivity band mix: mid and low-mid dominate.
      int value = (high + 2 * mid + 2 * low + 2) / 5;
      value = value > 2 ? value - 2 : 0;
      act[i] = value;
    }
  }
  std::vector<std::int32_t> smoothed;
  smooth3(act, width, height, &smoothed);
  if (edge_boost) {
    // Coherent gradient energy over 3x3: sums of signed dx/dy reinforce on an
    // edge and cancel in texture.  `edge` is the mean absolute coherent
    // gradient, in 8-bit levels.
    std::vector<std::int32_t> gx(static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                                 0);
    std::vector<std::int32_t> gy(static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                                 0);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const std::size_t i = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                              static_cast<std::size_t>(x);
        gx[i] = at(src, stride, width, height, x + 1, y) - at(src, stride, width, height, x - 1, y);
        gy[i] = at(src, stride, width, height, x, y + 1) - at(src, stride, width, height, x, y - 1);
      }
    }
    std::vector<std::int32_t> ex(static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                                 0);
    std::vector<std::int32_t> ey(ex.size(), 0);
    std::vector<std::int32_t> tmp(ex.size(), 0);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int sx = 0;
        int sy = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          const int py = std::min(std::max(y + dy, 0), height - 1);
          for (int dx = -1; dx <= 1; ++dx) {
            const int px = std::min(std::max(x + dx, 0), width - 1);
            const std::size_t j = static_cast<std::size_t>(py) * static_cast<std::size_t>(width) +
                                  static_cast<std::size_t>(px);
            sx += gx[j];
            sy += gy[j];
          }
        }
        const std::size_t i = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                              static_cast<std::size_t>(x);
        ex[i] = std::abs(sx);
        ey[i] = std::abs(sy);
        tmp[i] = (std::abs(sx) + std::abs(sy) + 4) / 9;
      }
    }
    smooth3(tmp, width, height, &ex);
    for (std::size_t i = 0; i < act.size(); ++i) {
      // Edge term: saturating, half the detail gain.  A coherent 12-level
      // gradient reaches ~40% of the extra gain.
      const int extra = (gain / 2) * ex[i] / (ex[i] + 6);
      act[i] = smoothed[i] + extra;
    }
  } else {
    act.swap(smoothed);
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                            static_cast<std::size_t>(x);
      const int a = act[i];
      (*out)[i] = static_cast<std::uint16_t>(clamp_weight(256 + (gain * a) / (a + kDetailRef)));
    }
  }
}

// Shape 5: per-pixel coherent edge weight only.  The signed 3x3 gradient sum
// reinforces along an oriented edge and cancels in isotropic texture, so the
// weight is "edges up" without the multi-scale texture term of shape 3.
inline void build_edges(const std::uint8_t* src, int stride, int width, int height, int gain,
                        std::vector<std::uint16_t>* out) {
  std::vector<std::int32_t> edge(static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                                 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int sx = 0;
      int sy = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          sx += at(src, stride, width, height, x + dx + 1, y + dy) -
                at(src, stride, width, height, x + dx - 1, y + dy);
          sy += at(src, stride, width, height, x + dx, y + dy + 1) -
                at(src, stride, width, height, x + dx, y + dy - 1);
        }
      }
      edge[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x)] = (std::abs(sx) + std::abs(sy) + 4) / 9;
    }
  }
  std::vector<std::int32_t> smoothed;
  smooth3(edge, width, height, &smoothed);
  for (std::size_t i = 0; i < smoothed.size(); ++i) {
    const int a = smoothed[i];
    (*out)[i] = static_cast<std::uint16_t>(clamp_weight(256 + (gain * a) / (a + 6)));
  }
}

// Shape 4: 8x8 source variance plus a coherent edge term over the same cell.
inline void build_variance(const std::uint8_t* src, int stride, int width, int height, int gain,
                           bool edge_boost, std::vector<std::uint16_t>* out) {
  const int cols = (width + 7) / 8;
  const int rows = (height + 7) / 8;
  for (int cy = 0; cy < rows; ++cy) {
    for (int cx = 0; cx < cols; ++cx) {
      const int sx = cx * 8;
      const int sy = cy * 8;
      std::int64_t sum = 0;
      std::int64_t sum2 = 0;
      for (int y = 0; y < 8; ++y) {
        const int py = std::min(sy + y, height - 1);
        for (int x = 0; x < 8; ++x) {
          const int px = std::min(sx + x, width - 1);
          const int v = at(src, stride, width, height, px, py);
          sum += v;
          sum2 += static_cast<std::int64_t>(v) * v;
        }
      }
      const std::int64_t mean = (sum + 32) / 64;
      std::int64_t var = (sum2 + 32) / 64 - mean * mean;
      if (var < 0) var = 0;
      int weight = 256 + static_cast<int>((gain * var) / (var + 48));
      if (edge_boost) {
        // Coherent cell gradient: |sum dx| + |sum dy| over the cell.
        int gx = 0;
        int gy = 0;
        for (int y = 0; y < 8; ++y) {
          const int py = std::min(sy + y, height - 1);
          for (int x = 0; x < 8; ++x) {
            const int px = std::min(sx + x, width - 1);
            gx += at(src, stride, width, height, px + 1, py) -
                  at(src, stride, width, height, px - 1, py);
            gy += at(src, stride, width, height, px, py + 1) -
                  at(src, stride, width, height, px, py - 1);
          }
        }
        const int edge = (std::abs(gx) + std::abs(gy)) / 64;
        weight += (gain / 2) * edge / (edge + 6);
      }
      weight = clamp_weight(weight);
      for (int y = 0; y < 8; ++y) {
        const int py = std::min(sy + y, height - 1);
        for (int x = 0; x < 8; ++x) {
          const int px = std::min(sx + x, width - 1);
          (*out)[static_cast<std::size_t>(py) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(px)] = static_cast<std::uint16_t>(weight);
        }
      }
    }
  }
}

inline void build_frame_weights(const std::uint8_t* src, int stride, int width, int height,
                                FrameWeights* out) {
  if (out == nullptr) return;
  *out = FrameWeights{};
  if (src == nullptr || width <= 0 || height <= 0 || stride < width) return;
  out->width = width;
  out->height = height;
  out->pix.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 256);
  const int detail_gain =
      static_cast<int>(std::lround(static_cast<double>(kDetailGain) * detail_scale()));
  const int gain = std::max(0, detail_gain);
  switch (shape()) {
    case 1:
      build_cell_gradient(src, stride, width, height, gain, &out->pix);
      break;
    case 2: {
      std::vector<std::int32_t> a, b, c;
      build_multiscale(src, stride, width, height, gain, /*edge_boost=*/false, &out->pix, &a, &b,
                       &c);
      break;
    }
    case 3: {
      std::vector<std::int32_t> a, b, c;
      build_multiscale(src, stride, width, height, gain, /*edge_boost=*/true, &out->pix, &a, &b,
                       &c);
      break;
    }
    case 4:
      build_variance(src, stride, width, height, gain, /*edge_boost=*/true, &out->pix);
      break;
    case 5:
      build_edges(src, stride, width, height, gain, &out->pix);
      break;
    default:
      break;  // shape 0: neutral map
  }
}

// Per-pixel weights for one leaf extent, Q8 raster order.  `x`/`y` are the
// absolute luma coordinates the frame map is indexed by.
inline void fill_leaf_weights(const FrameWeights& weights, int x, int y, int w, int h,
                              std::vector<std::uint16_t>* out) {
  if (out == nullptr) return;
  out->resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 256);
  if (!weights.valid() || w <= 0 || h <= 0) return;
  for (int yy = 0; yy < h; ++yy) {
    const int sy = std::min(std::max(y + yy, 0), weights.height - 1);
    for (int xx = 0; xx < w; ++xx) {
      const int sx = std::min(std::max(x + xx, 0), weights.width - 1);
      (*out)[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) +
             static_cast<std::size_t>(xx)] =
          weights.pix[static_cast<std::size_t>(sy) * static_cast<std::size_t>(weights.width) +
                      static_cast<std::size_t>(sx)];
    }
  }
}

// Weighted SSE in the reconstruction domain.  `weights` is Q8 raster order;
// a null pointer keeps the historical unweighted SSE.  Hidden bottom rows are
// skipped exactly like sse_against_source().  `lowpass_gain` adds the CSF
// term: the square of the 3x3 box mean error, so low/mid-frequency error
// costs `1 + gain` times a high-frequency error of the same amplitude.
inline double weighted_sse(const std::uint16_t* weights, int w, int h, const std::uint8_t* src,
                           int src_stride, int sx, int sy, const std::uint8_t* rec, int rec_stride,
                           int rx, int ry, int rows = -1, double lowpass_gain = 0.0,
                           double lowpass2_gain = 0.0) {
  if (w <= 0 || h <= 0 || src == nullptr || rec == nullptr) return 0.0;
  const int count = rows >= 0 && rows < h ? rows : h;
  double sum = 0.0;
  const bool csf = lowpass_gain > 0.0 || lowpass2_gain > 0.0;
  for (int y = 0; y < count; ++y) {
    const std::uint8_t* srow =
        src + static_cast<std::size_t>(sy + y) * static_cast<std::size_t>(src_stride);
    const std::uint8_t* rrow =
        rec + static_cast<std::size_t>(ry + y) * static_cast<std::size_t>(rec_stride);
    for (int x = 0; x < w; ++x) {
      const int d = static_cast<int>(srow[sx + x]) - static_cast<int>(rrow[rx + x]);
      const int weight =
          weights == nullptr
              ? 256
              : static_cast<int>(weights[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                                         static_cast<std::size_t>(x)]);
      double value = static_cast<double>(d * d);
      if (csf) {
        int low = 0;
        int used = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          const int py = y + dy;
          if (py < 0 || py >= count) continue;
          const std::uint8_t* sr =
              src + static_cast<std::size_t>(sy + py) * static_cast<std::size_t>(src_stride);
          const std::uint8_t* rr =
              rec + static_cast<std::size_t>(ry + py) * static_cast<std::size_t>(rec_stride);
          for (int dx = -1; dx <= 1; ++dx) {
            const int px = x + dx;
            if (px < 0 || px >= w) continue;
            low += static_cast<int>(sr[sx + px]) - static_cast<int>(rr[rx + px]);
            ++used;
          }
        }
        const double mean_low = static_cast<double>(low) / used;
        value += lowpass_gain * mean_low * mean_low;
        if (lowpass2_gain > 0.0) {
          int low2 = 0;
          int used2 = 0;
          for (int dy = -3; dy <= 3; ++dy) {
            const int py = y + dy;
            if (py < 0 || py >= count) continue;
            const std::uint8_t* sr =
                src + static_cast<std::size_t>(sy + py) * static_cast<std::size_t>(src_stride);
            const std::uint8_t* rr =
                rec + static_cast<std::size_t>(ry + py) * static_cast<std::size_t>(rec_stride);
            for (int dx = -3; dx <= 3; ++dx) {
              const int px = x + dx;
              if (px < 0 || px >= w) continue;
              low2 += static_cast<int>(sr[sx + px]) - static_cast<int>(rr[rx + px]);
              ++used2;
            }
          }
          const double mean_low2 = static_cast<double>(low2) / used2;
          value += lowpass2_gain * mean_low2 * mean_low2;
        }
      }
      sum += static_cast<double>(weight) * value;
    }
  }
  return sum / 256.0;
}

struct Counters {
  std::atomic<unsigned long long> leaves{0};
  std::atomic<unsigned long long> rescored{0};
  std::atomic<unsigned long long> inter_winner_changed{0};
  std::atomic<unsigned long long> skip_flips{0};
  std::atomic<unsigned long long> intra_flips{0};
  std::atomic<unsigned long long> residual_leaves{0};
  std::atomic<unsigned long long> fallbacks{0};
  std::atomic<unsigned long long> rate_interval{0};
};

inline Counters& counters() {
  static Counters value;
  return value;
}

inline void print_stats() {
  if (!enabled()) return;
  const Counters& c = counters();
  std::fprintf(stderr,
               "eff-perc: gate=1 scale=%.3f shape=%d detail=%.3f chroma=%.3f "
               "top=%d rate=%d csf=%.2f csf2=%.2f leaves=%llu rescored=%llu "
               "inter_winner_changed=%llu skip_flips=%llu intra_flips=%llu "
               "residual_leaves=%llu fallbacks=%llu rate_interval=%llu\n",
               lambda_scale(), shape(), detail_scale(), chroma_weight(), top_n(), rate_mode(),
               csf_gain(), csf2_gain(), static_cast<unsigned long long>(c.leaves.load()),
               static_cast<unsigned long long>(c.rescored.load()),
               static_cast<unsigned long long>(c.inter_winner_changed.load()),
               static_cast<unsigned long long>(c.skip_flips.load()),
               static_cast<unsigned long long>(c.intra_flips.load()),
               static_cast<unsigned long long>(c.residual_leaves.load()),
               static_cast<unsigned long long>(c.fallbacks.load()),
               static_cast<unsigned long long>(c.rate_interval.load()));
}

inline void init() {
  static const bool once = [] {
    std::atexit(&print_stats);
    return true;
  }();
  (void)once;
}

}  // namespace av2enc::video::eff_perc
