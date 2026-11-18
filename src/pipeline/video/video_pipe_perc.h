// Perceptual RD pipeline gate (AV2ENC_PIPE_PERC).
//
// The earlier perceptual attempts changed one term inside an SSE/lambda
// pipeline calibrated around it.  This gate moves D, R and lambda together as
// one decision path:
//
//   D  local, source-driven distortion.  Every 8x8 luma cell carries an
//      integer weight derived from the source gradient activity; flat cells
//      stay at 1.0, textured/edge cells are up-weighted (never down).  The
//      weight is applied to the distortion of the block the decision would
//      emit, and the same per-pixel map is passed into the transform search,
//      so TX size/type/partition is selected against the same objective.
//   R  exact inter syntax from the R09 live-CDF probe plus the static §8.3.2
//      coefficient-context model (txb_skip, EOB, coeff_base/br, escape and
//      sign) for the residual plan.  The historical iid coefficient estimate
//      is not used on this path.
//   lambda  one QP-derived weight for both the leaf J and the TX candidate
//      J.  `AV2ENC_PIPE_PERC_SCALE` multiplies it for the K-vs-scale sweep;
//      the shipped default is the measured value, not the sweep value.
//
// Search: the top-N cheap inter candidates are re-scored through the real
// predict -> TX -> quantise -> reconstruct -> rate path, each with both the
// skip and the residual form priced by the new objective.  Intra enters the
// same final comparison.  The frame-level drift refresh is unchanged and
// remains part of the preset that carries this gate.
//
// Gate (env only, no CLI flag):
//   AV2ENC_PIPE_PERC=1             master gate, default off
//   AV2ENC_PIPE_PERC_SCALE=<x>     lambda multiplier (sweep knob)
//   AV2ENC_PIPE_PERC_TOP=<n>       cheap candidates re-scored (default 4)
//   AV2ENC_PIPE_PERC_DETAIL=<x>    detail gain multiplier (ablation)
//
// Unset (or 0/false/off) does no work and leaves every pinned stream
// byte-identical: no weight map is built, no probe is reset, and every
// historical cost branch runs unchanged.
#pragma once

#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace av2enc::video::pipe_perc {

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
  static const bool value = [] {
    return !parse_off(std::getenv("AV2ENC_PIPE_PERC"));
  }();
  return value;
}

inline double env_double(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || !std::isfinite(parsed) || parsed <= 0.0) return fallback;
  return parsed;
}

// The lambda multiplier for the K-vs-scale sweep.  1.0 is the historical
// leaf lambda; the pipeline default is the measured scale-0.2 point (the K
// curve peaks there, see docs/pipe-perc.md).  The environment knob remains for
// reproducing the sweep.
inline double lambda_scale() {
  static const double value = env_double("AV2ENC_PIPE_PERC_SCALE", 0.2);
  return value;
}

// Detail gain relative to the measured default.  1.0 is the shipped shape;
// 0.0 is the documented plain-SSE ablation.  The sweep is an ablation, not a
// second tuned knob.
inline double detail_scale() {
  static const double value = [] {
    const char* env = std::getenv("AV2ENC_PIPE_PERC_DETAIL");
    if (env == nullptr || *env == '\0') return 1.0;
    char* end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || !std::isfinite(parsed) || parsed < 0.0) return 1.0;
    return parsed;
  }();
  return value;
}

inline int top_n() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_PIPE_PERC_TOP");
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

// Weight gain and softness in 8-bit luma gradient units.  `detail` is the
// mean of |dx| + |dy| over the 8x8 cell (0..510).  The curve is monotone and
// bounded: a flat cell stays at 1.0 and a strong edge reaches ~2.2x.
constexpr int kDetailGain = 384;
constexpr int kDetailRef = 24;
constexpr int kMaxWeightQ8 = 1024;

// Frame-level 8x8 cell weights in Q8 (256 = neutral).
struct FrameWeights {
  int width = 0;
  int height = 0;
  int cols = 0;
  int rows = 0;
  std::vector<std::uint16_t> cell;

  bool valid() const {
    return width > 0 && height > 0 && cols > 0 && rows > 0 &&
           cell.size() == static_cast<std::size_t>(cols) *
                              static_cast<std::size_t>(rows);
  }
};

inline int cell_weight(const FrameWeights& weights, int x, int y) {
  if (!weights.valid()) return 256;
  const int cx = x / 8 < 0 ? 0
                           : (x / 8 >= weights.cols ? weights.cols - 1
                                                    : x / 8);
  const int cy = y / 8 < 0 ? 0
                           : (y / 8 >= weights.rows ? weights.rows - 1
                                                    : y / 8);
  return weights.cell[static_cast<std::size_t>(cy) *
                          static_cast<std::size_t>(weights.cols) +
                      static_cast<std::size_t>(cx)];
}

// Builds the per-frame cell map.  The gradient is clamped at the frame edge,
// the sum order is row-major and the arithmetic is integer, so the map is
// deterministic.
inline void build_frame_weights(const std::uint8_t* src, int stride, int width,
                                int height, FrameWeights* out) {
  if (out == nullptr) return;
  *out = FrameWeights{};
  if (src == nullptr || width <= 0 || height <= 0 || stride < width) return;
  out->width = width;
  out->height = height;
  out->cols = (width + 7) / 8;
  out->rows = (height + 7) / 8;
  out->cell.assign(static_cast<std::size_t>(out->cols) *
                       static_cast<std::size_t>(out->rows),
                   256);
  const int gain =
      static_cast<int>(std::lround(static_cast<double>(kDetailGain) *
                                   detail_scale()));
  for (int cy = 0; cy < out->rows; ++cy) {
    for (int cx = 0; cx < out->cols; ++cx) {
      const int sx = cx * 8;
      const int sy = cy * 8;
      int activity = 0;
      for (int y = 0; y < 8; ++y) {
        const int py = sy + y < height ? sy + y : height - 1;
        const int py_next = py + 1 < height ? py + 1 : py;
        const std::uint8_t* row =
            src + static_cast<std::size_t>(py) * static_cast<std::size_t>(stride);
        const std::uint8_t* row_next =
            src + static_cast<std::size_t>(py_next) *
                      static_cast<std::size_t>(stride);
        for (int x = 0; x < 8; ++x) {
          const int px = sx + x < width ? sx + x : width - 1;
          const int px_next = px + 1 < width ? px + 1 : px;
          const int v = row[px];
          activity += std::abs(v - row[px_next]);
          activity += std::abs(v - row_next[px]);
        }
      }
      const int detail = activity / 64;  // mean |dx| + |dy| per sample
      int weight = 256 + (gain * detail) / (detail + kDetailRef);
      if (weight > kMaxWeightQ8) weight = kMaxWeightQ8;
      if (weight < 256) weight = 256;
      out->cell[static_cast<std::size_t>(cy) *
                    static_cast<std::size_t>(out->cols) +
                static_cast<std::size_t>(cx)] =
          static_cast<std::uint16_t>(weight);
    }
  }
}

// Per-pixel weights for one leaf extent, Q8 raster order.  `x`/`y` are the
// absolute luma coordinates the source cell map is indexed by.
inline void fill_leaf_weights(const FrameWeights& weights, int x, int y, int w,
                              int h, std::vector<std::uint16_t>* out) {
  if (out == nullptr) return;
  out->resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 256);
  if (!weights.valid() || w <= 0 || h <= 0) return;
  for (int yy = 0; yy < h; ++yy) {
    for (int xx = 0; xx < w; ++xx) {
      (*out)[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) +
             static_cast<std::size_t>(xx)] = static_cast<std::uint16_t>(
          cell_weight(weights, x + xx, y + yy));
    }
  }
}

// Weighted SSE in the reconstruction domain.  `weights` is Q8 raster order;
// a null pointer keeps the historical unweighted SSE.  Hidden bottom rows are
// skipped exactly like sse_against_source().
inline double weighted_sse(const std::uint16_t* weights, int w, int h,
                           const std::uint8_t* src, int src_stride, int sx,
                           int sy, const std::uint8_t* rec, int rec_stride,
                           int rx, int ry, int rows = -1) {
  if (w <= 0 || h <= 0 || src == nullptr || rec == nullptr) return 0.0;
  const int count = rows >= 0 && rows < h ? rows : h;
  double sum = 0.0;
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
              : static_cast<int>(
                    weights[static_cast<std::size_t>(y) *
                                static_cast<std::size_t>(w) +
                            static_cast<std::size_t>(x)]);
      sum += static_cast<double>(weight) * static_cast<double>(d * d);
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
};

inline Counters& counters() {
  static Counters value;
  return value;
}

inline void print_stats() {
  if (!enabled()) return;
  const Counters& c = counters();
  std::fprintf(stderr,
               "pipe-perc: gate=1 scale=%.3f detail=%.3f top=%d leaves=%llu "
               "rescored=%llu inter_winner_changed=%llu skip_flips=%llu "
               "intra_flips=%llu residual_leaves=%llu fallbacks=%llu\n",
               lambda_scale(), detail_scale(), top_n(),
               static_cast<unsigned long long>(c.leaves.load()),
               static_cast<unsigned long long>(c.rescored.load()),
               static_cast<unsigned long long>(c.inter_winner_changed.load()),
               static_cast<unsigned long long>(c.skip_flips.load()),
               static_cast<unsigned long long>(c.intra_flips.load()),
               static_cast<unsigned long long>(c.residual_leaves.load()),
               static_cast<unsigned long long>(c.fallbacks.load()));
}

inline void init() {
  static const bool once = [] {
    std::atexit(&print_stats);
    return true;
  }();
  (void)once;
}

}  // namespace av2enc::video::pipe_perc
