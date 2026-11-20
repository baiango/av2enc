// AV2ENC_P_DRIFT (default off): head-only drift-aware skip/reference cost.
//
// The measured defect (docs/rd-compose.md, docs/video-inter-residual-rd.md,
// docs/port-lane2.md): the head f1-32 of the 240p clip spends only 45-65 B per
// picture against AVM's ~197.  The frames are almost all skip, and a skipped
// leaf does not just take this picture's prediction error: it installs that
// error in the reconstructed reference, where the following pictures copy it
// through fractional-pel MC.  Every later repair therefore pays for error a
// residual would have removed locally, for a fraction of the rate.
//
// The leaf decision prices the skip arm with this picture's error only
// (`pred_total`), so a stream that never repairs a frozen error keeps skipping
// it.  This gate adds the error each arm *installs* in the next K real source
// pictures:
//
//   J_skip = D_perc(prediction) + alpha * sum_k w_k * D_perc_future(skip)
//            + lambda * R_skip
//   J_res  = D_perc(recon)      + alpha * sum_k w_k * D_perc_future(res)
//            + lambda * R_res
//   w_k = 0.5^k, k = 1..K
//
// Two things separate this from the merged `AV2ENC_PORT_HORIZON`
// (docs/port-lane2.md), which lost 0.26-0.50 K on this exact recipe:
//
//   1. **Units.**  The port added a *plain* future SSE to a *perceptual*
//      current D.  On eff-perc the current term is
//      `sum(w * (e^2 + csf * mean3x3(e)^2)) / 256`, and the port's unweighted
//      future term under-prices low/mid-band error by the CSF factor, which is
//      what inverted the decision on high-activity leaves (skip share rose and
//      every segment lost PSNR).  `future_distortion_perc()` uses the *same*
//      eff-perc form and the same CSF gains as the current D, so the two arms
//      compare one quantity.
//   2. **Scope.**  Head pictures only.  f33-59 already carries the
//      drift-refresh schedule (32 refresh frames, `--video-drift-refresh-*`);
//      the term must not fight it, and the head is where the under-spend is.
//      The schedule itself is untouched.
//
// The future frames are the real source pictures the flat P chain already
// holds (`InterFrameEncoder::horizon_future_`, `frames.subspan(i + 1)`), so
// the term is evidence, not a tuned constant.
//
// Gate (env only, no CLI flag), default off:
//   AV2ENC_P_DRIFT=1            master gate
//   AV2ENC_P_DRIFT_K=<n>        horizon depth (4; 1..8)
//   AV2ENC_P_DRIFT_ALPHA=<x>    weight on the summed future term (0.5)
//   AV2ENC_P_DRIFT_FIRST=<n>    first armed picture ordinal (1)
//   AV2ENC_P_DRIFT_LAST=<n>     last armed picture ordinal (32)
//   AV2ENC_P_DRIFT_STATS=1      print the counters
//
// Unset (or 0/false/off) does no work at all: `armed()` is false and the leaf
// decision adds exactly zero to both arms, so every pinned stream stays
// byte-identical.
#pragma once

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>

#include "pipeline/video/video_frame.h"

namespace av2enc::video::p_drift {

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

inline double env_double(const char* name, double fallback, bool allow_zero) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || !std::isfinite(parsed)) return fallback;
  if (allow_zero ? parsed < 0.0 : parsed <= 0.0) return fallback;
  return parsed;
}

inline int env_int(const char* name, int fallback, int lo, int hi) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value) return fallback;
  if (parsed < lo) return lo;
  if (parsed > hi) return hi;
  return static_cast<int>(parsed);
}

struct Config {
  bool on = false;
  int k = 4;
  double alpha = 0.5;
  int first = 1;
  int last = 32;
  bool stats = false;
};

inline const Config& config() {
  static const Config cfg = [] {
    Config out;
    out.on = !parse_off(std::getenv("AV2ENC_P_DRIFT"));
    out.k = env_int("AV2ENC_P_DRIFT_K", out.k, 1, 8);
    out.alpha = env_double("AV2ENC_P_DRIFT_ALPHA", out.alpha, true);
    out.first = env_int("AV2ENC_P_DRIFT_FIRST", out.first, 0, 9999);
    out.last = env_int("AV2ENC_P_DRIFT_LAST", out.last, 0, 9999);
    const char* s = std::getenv("AV2ENC_P_DRIFT_STATS");
    out.stats = s != nullptr && !parse_off(s);
    return out;
  }();
  return cfg;
}

// The lane's stage-3 aggregation constant (0.5^k), same as the horizon port.
inline constexpr double kDecay = 0.5;

// One armed picture: an inter (P) picture whose ordinal is inside the head
// window.  `ordinal` is the shared one-per-picture counter the tail TX stage
// uses (key = 0, then one per P picture).
inline bool armed(bool keyframe, int ordinal) {
  const Config& cfg = config();
  if (!cfg.on || keyframe) return false;
  if (cfg.alpha <= 0.0) return false;
  return ordinal >= cfg.first && ordinal <= cfg.last;
}

// eff-perc weighted SSE over an explicit error block, in the same form the
// leaf D uses: sum(w * (e^2 + g1 * mean3x3(e)^2 + g2 * mean7x7(e)^2)) / 256,
// with the 3x3 / 7x7 windows clipped to the block exactly like
// eff_perc::weighted_sse().  `err` is w*h, raster order.  `weights` is the
// caller's Q8 map (null = the neutral 256).
inline double error_distortion(const double* err, int w, int h, double gain1,
                               double gain2,
                               const std::uint16_t* weights = nullptr) {
  if (w <= 0 || h <= 0) return 0.0;
  const bool csf = gain1 > 0.0 || gain2 > 0.0;
  double sum = 0.0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) *
                                static_cast<std::size_t>(w) +
                            static_cast<std::size_t>(x);
      const double d = err[i];
      double value = d * d;
      if (csf) {
        double low = 0.0;
        int used = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          const int py = y + dy;
          if (py < 0 || py >= h) continue;
          for (int dx = -1; dx <= 1; ++dx) {
            const int px = x + dx;
            if (px < 0 || px >= w) continue;
            low += err[static_cast<std::size_t>(py) *
                           static_cast<std::size_t>(w) +
                       static_cast<std::size_t>(px)];
            ++used;
          }
        }
        const double mean_low = low / static_cast<double>(used);
        value += gain1 * mean_low * mean_low;
        if (gain2 > 0.0) {
          double low2 = 0.0;
          int used2 = 0;
          for (int dy = -3; dy <= 3; ++dy) {
            const int py = y + dy;
            if (py < 0 || py >= h) continue;
            for (int dx = -3; dx <= 3; ++dx) {
              const int px = x + dx;
              if (px < 0 || px >= w) continue;
              low2 += err[static_cast<std::size_t>(py) *
                              static_cast<std::size_t>(w) +
                          static_cast<std::size_t>(px)];
              ++used2;
            }
          }
          const double mean_low2 = low2 / static_cast<double>(used2);
          value += gain2 * mean_low2 * mean_low2;
        }
      }
      sum += static_cast<double>(
                 weights == nullptr
                     ? 256
                     : static_cast<int>(weights[static_cast<std::size_t>(y) *
                                                    static_cast<std::size_t>(w) +
                                                static_cast<std::size_t>(x)])) *
             value;
    }
  }
  return sum / 256.0;
}

// Source-extent clamp, the same rule `source_sample()` applies.
inline int clamped_sample(const std::uint8_t* plane, int frame_w, int frame_h,
                          int sx, int sy) {
  const int x = sx < 0 ? 0 : (sx >= frame_w ? frame_w - 1 : sx);
  const int y = sy < 0 ? 0 : (sy >= frame_h ? frame_h - 1 : sy);
  return static_cast<int>(
      plane[static_cast<std::size_t>(y) * static_cast<std::size_t>(frame_w) +
            static_cast<std::size_t>(x)]);
}

// sum_{k=1..K} 0.5^k * D_perc(source[i+k] block, coded block), all three
// planes, in the eff-perc inter D units.  `future` is `frames.subspan(i+1)`
// (the real source pictures after this one).  `y/u/v` are tightly packed
// (w x h and (w/2) x (h/2)).  `rows_*` clip the decoder-invisible bottom rows
// exactly as the current-frame D does.
inline double future_distortion_perc(
    std::span<const VideoInputFrame> future, int horizon, int x0, int y0, int w,
    int h, const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v,
    int cw, int ch, int rows_luma, int rows_chroma, double csf1, double csf2,
    double chroma_weight, const std::uint16_t* luma_weights,
    std::vector<double>* scratch) {
  if (horizon <= 0 || future.empty() || y == nullptr) return 0.0;
  const Config& cfg = config();
  const int count = horizon < cfg.k ? horizon : cfg.k;
  std::vector<double>& err = *scratch;
  double value = 0.0;
  double weight = kDecay;
  for (int k = 0; k < count; ++k) {
    const std::size_t index = static_cast<std::size_t>(k);
    if (index >= future.size()) break;
    const VideoInputFrame& frame = future[index];
    const int lw = static_cast<int>(frame.width);
    const int lh = static_cast<int>(frame.height);
    const int cwid = (lw + 1) / 2;
    const int chgt = (lh + 1) / 2;
    if (lw <= 0 || lh <= 0) break;
    double plane_sum = 0.0;
    // Luma.
    {
      const int nrows = rows_luma >= 0 && rows_luma < h ? rows_luma : h;
      if (!frame.y.empty() && nrows > 0) {
        err.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h),
                   0.0);
        for (int yy = 0; yy < nrows; ++yy) {
          for (int xx = 0; xx < w; ++xx) {
            const double d =
                static_cast<double>(clamped_sample(frame.y.data(), lw, lh,
                                                   x0 + xx, y0 + yy)) -
                static_cast<double>(y[static_cast<std::size_t>(yy) *
                                          static_cast<std::size_t>(w) +
                                      static_cast<std::size_t>(xx)]);
            err[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(xx)] = d;
          }
        }
        plane_sum += error_distortion(err.data(), w, nrows, csf1, csf2,
                                      luma_weights);
      }
    }
    // Chroma, at the leaf's chroma geometry (cw x ch), origin (x0/2, y0/2).
    const int cx0 = x0 / 2;
    const int cy0 = y0 / 2;
    const int cnrows =
        rows_chroma >= 0 && rows_chroma < ch ? rows_chroma : ch;
    const std::uint8_t* chroma[2] = {u, v};
    const std::span<const std::uint8_t> planes[2] = {frame.u, frame.v};
    if (cw > 0 && ch > 0 && cnrows > 0) {
      for (int p = 0; p < 2; ++p) {
        if (chroma[p] == nullptr || planes[p].empty()) continue;
        err.assign(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch),
                   0.0);
        for (int yy = 0; yy < cnrows; ++yy) {
          for (int xx = 0; xx < cw; ++xx) {
            const double d =
                static_cast<double>(clamped_sample(planes[p].data(), cwid,
                                                   chgt, cx0 + xx,
                                                   cy0 + yy)) -
                static_cast<double>(chroma[p][static_cast<std::size_t>(yy) *
                                                  static_cast<std::size_t>(cw) +
                                              static_cast<std::size_t>(xx)]);
            err[static_cast<std::size_t>(yy) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(xx)] = d;
          }
        }
        plane_sum += error_distortion(err.data(), cw, cnrows, csf1, csf2) *
                     chroma_weight;
      }
    }
    value += weight * plane_sum;
    weight *= kDecay;
  }
  return value;
}

struct Counters {
  long long leaves = 0;
  long long skip_kept = 0;
  long long skip_to_residual = 0;
  long long residual_to_skip = 0;
  double future_skip = 0.0;
  double future_res = 0.0;
  double now_skip = 0.0;
  double now_res = 0.0;

  ~Counters() {
    const Config& cfg = config();
    if (!cfg.on && !cfg.stats) return;
    std::fprintf(stderr,
                 "p-drift: gate=%d k=%d alpha=%.3f window=%d..%d leaves=%lld "
                 "skip_kept=%lld skip->res=%lld res->skip=%lld "
                 "fut[skip=%.0f res=%.0f] now[skip=%.0f res=%.0f]\n",
                 cfg.on ? 1 : 0, cfg.k, cfg.alpha, cfg.first, cfg.last, leaves,
                 skip_kept, skip_to_residual, residual_to_skip, future_skip,
                 future_res, now_skip, now_res);
  }
};

inline Counters& counters() {
  static Counters value;
  return value;
}

}  // namespace av2enc::video::p_drift
