// AV2ENC_PIPE_BA: Butteraugli-shaped perceptual decision pipeline.
//
// The pipeline is the measured exact-rate decision path (live §7.4.4/§7.5 CDF
// probe, top-2 real-path re-scoring, R19 interval coefficient term, one lambda
// for the leaf and the TX candidate) with its own distortion model instead of
// the eff-perc 3x3/7x7 CSF:
//
//   Butteraugli is a masked, low-frequency-weighted error model.  It does not
//   up-weight detail; it down-weights error where the source already carries
//   local contrast, and it prices the low-frequency residue (the banding and
//   slow drift a decoded stream shows in smooth gradients) above the
//   high-frequency tail.  Chroma is part of the same model, masked by luma
//   activity because a busy luma neighbourhood hides chroma error too.
//
// Per coded sample of a plane, with e = src - rec and m the local mask:
//
//   D = sum( m(x,y) * (e^2 + g * mean3x3(e)^2) ) / 256
//   m = 256 * act_ref / (act_ref + act)                    Q8, floored
//   act = 3x3 mean of the source's 1/3/7/15-px band contrast, -2 deadband
//   g = weight * kLowBandGain                             (3x3 = low/mid)
//
// `weight` is the tool knob: 0 is the tool off (plain SSE, the pipeline's own
// base arm), 1 is the model as measured (g = 1.5, the eff-perc peak), and the
// sweep 0.5/1.0/2.0 walks the strength.  `weight` scales the mask's departure
// from neutral as well as g, so every arm is one monotone family.
//
// Default off.  Unset does no work: no mask is built, no weight map is filled
// and every historical / eff-perc branch runs unchanged, so all pins stay
// byte-identical.
//
//   AV2ENC_PIPE_BA            master gate (any non-zero value enables)
//   AV2ENC_PIPE_BA_WEIGHT     tool weight, default 0 (off); alias
//                             AV2ENC_PIPE_BA_W
//   AV2ENC_PIPE_BA_LAMBDA     lambda multiplier, default 0.18 (the exact-rate
//                             pipeline's calibrated point)
//
// Header-only by design (`video_eff_perc.h` is the same shape): a lane
// worktree must build this without touching the shared source list in
// CMakeLists.txt.  `video_pipe_ba.cpp` carries the translation-unit contract.
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

namespace av2enc::video::pipe_ba {

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

inline bool enabled() {
  static const bool value = [] { return !parse_off(std::getenv("AV2ENC_PIPE_BA")); }();
  return value;
}

inline double env_double(const char* name, double fallback, double min_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || !std::isfinite(parsed) || parsed < min_value) return fallback;
  return parsed;
}

// Tool weight.  0 keeps the plain-SSE pipeline base; 1.0 is the full model.
inline double weight() {
  static const double value = [] {
    const char* alias = std::getenv("AV2ENC_PIPE_BA_W");
    if (alias != nullptr && *alias != '\0') {
      return env_double("AV2ENC_PIPE_BA_W", 0.0, 0.0);
    }
    return env_double("AV2ENC_PIPE_BA_WEIGHT", 0.0, 0.0);
  }();
  return value;
}

inline bool tool_on() { return enabled() && weight() > 0.0; }

inline double lambda_scale() { return env_double("AV2ENC_PIPE_BA_LAMBDA", 0.18, 1e-6); }

// Low/mid-band error gain at weight 1.  1.5 is the measured eff-perc peak and
// the band the Butteraugli gap lives in; the tool weight scales it.
constexpr double kLowBandGain = 1.5;
// Butteraugli local-contrast masking: a saturating activity normaliser,
// measured on the 240p clip (`tools/lane-4/ba_mask_stats.py`: act p50 = 0,
// p75 = 6, p90 = 11, p99 = 26 levels, 59% of pixels flat).  act_ref 24 makes a
// typical textured pixel 0.6-0.8 and a fully flat one exactly neutral, so the
// model never discounts a sample entirely.
constexpr double kActRef = 24.0;
constexpr int kMaskFloorQ8 = 160;
constexpr int kMaskNeutralQ8 = 256;

struct Config {
  double weight = 0.0;
  double lambda = 0.18;
  double chroma_scale = 1.0;
  double low_band_gain = 0.0;
  double chroma_low_band_gain = 0.0;
};

inline const Config& config() {
  static const Config cfg = [] {
    Config out;
    out.weight = weight();
    out.lambda = lambda_scale();
    out.low_band_gain = out.weight * kLowBandGain;
    out.chroma_low_band_gain = out.weight * kLowBandGain;
    return out;
  }();
  return cfg;
}

// Per-pixel luma mask, Q8 raster order.  Already blended with the tool weight:
// weight 0 fills 256 everywhere (neutral), so the base arm is plain SSE.
struct FrameMask {
  int width = 0;
  int height = 0;
  std::vector<std::uint16_t> pix;

  bool valid() const {
    return width > 0 && height > 0 &&
           pix.size() == static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  }
};

struct Counters {
  std::atomic<unsigned long long> masks{0};
  std::atomic<unsigned long long> luma_leaves{0};
  std::atomic<unsigned long long> chroma_leaves{0};
  std::atomic<unsigned long long> tx_calls{0};
  std::atomic<unsigned long long> mask_q8_sum{0};
  std::atomic<unsigned long long> mask_pixels{0};
  std::atomic<int> mask_min_q8{kMaskNeutralQ8};
  std::atomic<int> mask_max_q8{kMaskNeutralQ8};
};

inline Counters& counters() {
  static Counters value;
  return value;
}

// Separable box sum of radius `r`, edges clamped; the same deterministic
// integer arithmetic as `video_eff_perc.h` so the source statistic matches the
// D-shape ablations recorded there.
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

inline int clamp_mask(std::int64_t value) {
  if (value < kMaskFloorQ8) return kMaskFloorQ8;
  if (value > kMaskNeutralQ8) return kMaskNeutralQ8;
  return static_cast<int>(value);
}

// Builds the frame mask from the source luma.  The activity map is the
// three-band box-difference contrast (1/3/7/15 px means, mid and low-mid
// weighted twice) smoothed 3x3 -- the same source statistic the eff-perc
// D-shape ablations used, so the mask is comparable with that work.
inline void build_frame_mask(const std::uint8_t* src, int stride, int width, int height,
                             FrameMask* out) {
  if (out == nullptr) return;
  *out = FrameMask{};
  if (src == nullptr || width <= 0 || height <= 0 || stride < width) return;
  out->width = width;
  out->height = height;
  out->pix.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                  kMaskNeutralQ8);
  const double blend = config().weight;
  if (blend <= 0.0) return;  // tool off: neutral map
  std::vector<std::int32_t> s1;
  std::vector<std::int32_t> s3;
  std::vector<std::int32_t> s7;
  box_sum(src, stride, width, height, 1, &s1);
  box_sum(src, stride, width, height, 3, &s3);
  box_sum(src, stride, width, height, 7, &s7);
  std::vector<std::int32_t> act(static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                                0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                            static_cast<std::size_t>(x);
      const int p = src[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
                        static_cast<std::size_t>(x)];
      const int m1 = (s1[i] + 4) / 9;
      const int m3 = (s3[i] + 24) / 49;
      const int m7 = (s7[i] + 112) / 225;
      const int high = std::abs(p - m1);
      const int mid = std::abs(m1 - m3);
      const int low = std::abs(m3 - m7);
      const int value = (high + 2 * mid + 2 * low + 2) / 5;
      act[i] = value > 2 ? value - 2 : 0;
    }
  }
  std::vector<std::int32_t> smoothed;
  smooth3(act, width, height, &smoothed);
  unsigned long long sum = 0;
  for (std::size_t i = 0; i < smoothed.size(); ++i) {
    const double mask = 256.0 * kActRef / (kActRef + static_cast<double>(smoothed[i]));
    const std::int64_t blended =
        static_cast<std::int64_t>(std::llround(256.0 + blend * (mask - 256.0)));
    const int value = clamp_mask(blended);
    out->pix[i] = static_cast<std::uint16_t>(value);
    sum += static_cast<unsigned long long>(value);
  }
  // Re-normalise to a frame mean of 256 Q8.  The mask is a *shape*: keeping
  // its mean neutral leaves lambda calibrated at every tool weight, so the
  // sweep measures the shape and nothing else (a uniform scale would act as a
  // lambda retune, which this pipeline holds fixed).
  double mean = static_cast<double>(sum) / static_cast<double>(smoothed.size());
  if (mean > 0.0) {
    const double scale = 256.0 / mean;
    unsigned long long q8_sum = 0;
    int lowest = kMaskNeutralQ8;
    int highest = kMaskNeutralQ8;
    for (std::uint16_t& value : out->pix) {
      const long long scaled =
          std::llround(static_cast<double>(value) * scale);
      const int clamped = static_cast<int>(std::min<long long>(std::max<long long>(scaled, 64), 512));
      value = static_cast<std::uint16_t>(clamped);
      q8_sum += static_cast<unsigned long long>(clamped);
      lowest = std::min(lowest, clamped);
      highest = std::max(highest, clamped);
    }
    Counters& c = counters();
    ++c.masks;
    c.mask_q8_sum += q8_sum;
    c.mask_pixels += smoothed.size();
    if (lowest < c.mask_min_q8.load()) c.mask_min_q8.store(lowest);
    if (highest > c.mask_max_q8.load()) c.mask_max_q8.store(highest);
  }
}

// Masks one leaf extent; `x`/`y` are absolute luma coordinates.
inline void fill_leaf_weights(const FrameMask& mask, int x, int y, int w, int h,
                              std::vector<std::uint16_t>* out) {
  if (out == nullptr) return;
  out->resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), kMaskNeutralQ8);
  if (!mask.valid() || w <= 0 || h <= 0) return;
  for (int yy = 0; yy < h; ++yy) {
    const int sy = std::min(std::max(y + yy, 0), mask.height - 1);
    for (int xx = 0; xx < w; ++xx) {
      const int sx = std::min(std::max(x + xx, 0), mask.width - 1);
      (*out)[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) +
             static_cast<std::size_t>(xx)] =
          mask.pix[static_cast<std::size_t>(sy) * static_cast<std::size_t>(mask.width) +
                   static_cast<std::size_t>(sx)];
    }
  }
}

// Chroma leaf variant: the mask is sampled at 2x the chroma coordinates so a
// chroma sample sees the luma neighbourhood it sits in.
inline void fill_chroma_leaf_weights(const FrameMask& mask, int cx, int cy, int w, int h,
                                     std::vector<std::uint16_t>* out) {
  if (out == nullptr) return;
  out->resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), kMaskNeutralQ8);
  if (!mask.valid() || w <= 0 || h <= 0) return;
  for (int yy = 0; yy < h; ++yy) {
    const int sy = std::min(std::max(2 * (cy + yy), 0), mask.height - 1);
    for (int xx = 0; xx < w; ++xx) {
      const int sx = std::min(std::max(2 * (cx + xx), 0), mask.width - 1);
      (*out)[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) +
             static_cast<std::size_t>(xx)] =
          mask.pix[static_cast<std::size_t>(sy) * static_cast<std::size_t>(mask.width) +
                   static_cast<std::size_t>(sx)];
    }
  }
}

// Masked low-frequency distortion of one coded plane of one leaf:
//   sum( w8 * (e^2 + gain * mean3x3(e)^2) ) / 256
// `rows` < h scores only the decoder-visible rows (bottom-edge mask).  A null
// `weights` pointer is neutral 256.  The 3x3 window is clamped inside the
// block, exactly like the TX candidate: the leaf J and the TX plan price the
// same error.
inline double plane_distortion(const std::uint16_t* weights, int w, int h,
                               const std::uint8_t* src, int src_stride, int sx, int sy,
                               const std::uint8_t* rec, int rec_stride, int rx, int ry,
                               int rows, double low_band_gain) {
  if (w <= 0 || h <= 0 || src == nullptr || rec == nullptr) return 0.0;
  const int count = rows >= 0 && rows < h ? rows : h;
  const bool band = low_band_gain > 0.0;
  double sum = 0.0;
  for (int y = 0; y < count; ++y) {
    const std::uint8_t* srow =
        src + static_cast<std::size_t>(sy + y) * static_cast<std::size_t>(src_stride);
    const std::uint8_t* rrow =
        rec + static_cast<std::size_t>(ry + y) * static_cast<std::size_t>(rec_stride);
    for (int x = 0; x < w; ++x) {
      const int d = static_cast<int>(srow[sx + x]) - static_cast<int>(rrow[rx + x]);
      double value = static_cast<double>(d * d);
      if (band) {
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
        value += low_band_gain * mean_low * mean_low;
      }
      const int weight =
          weights == nullptr
              ? kMaskNeutralQ8
              : static_cast<int>(weights[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                                         static_cast<std::size_t>(x)]);
      sum += static_cast<double>(weight) * value;
    }
  }
  return sum / 256.0;
}

inline void print_stats() {
  if (!enabled()) return;
  const Counters& c = counters();
  const double mean_q8 =
      c.mask_pixels.load() == 0
          ? 0.0
          : static_cast<double>(c.mask_q8_sum.load()) /
                static_cast<double>(c.mask_pixels.load());
  std::fprintf(stderr,
               "pipe-ba: gate=1 weight=%.3f lambda=%.3f low_band=%.3f "
               "masks=%llu mask_mean_q8=%.1f mask_range=%d..%d luma_leaves=%llu "
               "chroma_leaves=%llu tx_calls=%llu\n",
               config().weight, config().lambda, config().low_band_gain,
               static_cast<unsigned long long>(c.masks.load()), mean_q8,
               c.mask_min_q8.load(), c.mask_max_q8.load(),
               static_cast<unsigned long long>(c.luma_leaves.load()),
               static_cast<unsigned long long>(c.chroma_leaves.load()),
               static_cast<unsigned long long>(c.tx_calls.load()));
}

inline void init() {
  static const bool once = [] {
    std::atexit(&print_stats);
    return true;
  }();
  (void)once;
}

}  // namespace av2enc::video::pipe_ba
