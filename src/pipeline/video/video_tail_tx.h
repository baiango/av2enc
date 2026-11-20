// AV2ENC_RD_TAIL_TX (default off): exact-rate, BA-shaped TX plan and
// coefficient-level RD on the P-frame tail (pictures 33-59).
//
// The measured tail defect (docs/luma-audit.md, docs/video-inter-residual-rd.md):
// frames 33-59 hold the stream's worst Butteraugli (2.269 matched) and its
// coded residual is where the error lives.  The inter luma TX search prices its
// candidate plans with the RDOQ calibration (`0.4 x` the static §8.3 model) and
// a distortion summed per transform leaf with the 3x3/7x7 windows clipped to
// the leaf.  The leaf decision *after* the search already charges the live
// §8.2.6 interval rate (`plan_rate_bits`), so the plan and the level choice are
// made on a different cost than the one the leaf is scored with.
//
// This gate applies the L3 exact-rate mechanism (docs/rd-lane3.md) to that
// residual:
//
//   D = seq_rd::leaf_distortion() on the assembled block reconstruction
//       (eff-perc inter CSF + the driver's high-pass / block-edge terms)
//   R = plan_rate_bits() -- the §8.3 symbol sequence through the §8.2.6
//       interval arithmetic against the carried live CDF image, scan order
//   J = D + lambda * R,  lambda = the inter leaf's pipe lambda
//
// Candidate plans: the screen winner (the historical search) and one exact
// re-search that prices its own candidates with the uncalibrated §8.3 model
// (`model_rate_scale = 1.0`) and the coefficient trellis at the L3 level
// scale.  The better J wins, tie to the screen plan.  Nothing runs on the key
// picture, on intra leaves, on chroma, on the TIP second pass or on tiled
// encodes, and the gate is unset everywhere by default.
#pragma once

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace av2enc::video::tail_tx {

struct Config {
  bool on = false;
  bool level = true;
  bool live_rate = true;
  bool ba_d = true;
  bool stats = false;
  int first = 33;
  int last = 59;
  // Coefficient trellis SSD-domain lambda as a fraction of the measured RDOQ
  // scale.  L3 calibrated 0.25 on the key (docs/rd-lane3.md §"Coefficient
  // level trellis"); the same auto value is the default here.
  double level_scale = 0.25;
  // Plan-J lambda as a fraction of the inter leaf's pipe lambda.
  double lambda = 1.0;
  // High-pass / block-edge weights of the driver's BA-shaped D
  // (`seq_rd::leaf_distortion`).  The driver's measured default is 0 for both.
  double hf = 0.0;
  double edge = 0.0;
  // Source high-pass activity gate for sub-16 coding blocks (the driver's
  // AV2ENC_RD_DRIVER_ACT counterpart).  0 = off: the tail keeps the historical
  // 16x16 floor.  Above 0 a 16x16 node whose mean (d - box3x3(d))^2 clears the
  // threshold must split, so the residual can reach 4x4/8x4 transforms.
  double act = 0.0;
};

// Per-picture arming for the exact-rate TX stage.  The historical gate is a
// fixed window (`first`..`last`, the tail 33..59).  AV2ENC_RD_HEAD_TX re-uses
// the same objective on the head pictures 1..32, where docs/rd-compose.md
// measures the other half of the Butteraugli gap.  Both arms keep every
// default off.
inline int head_first() { return 1; }
inline int head_last() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_RD_HEAD_LAST");
    if (env == nullptr || *env == '\0') return 32;
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed < 0) return 32;
    return static_cast<int>(parsed);
  }();
  return value;
}

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

// AV2ENC_RD_HEAD_TX (default off): arm the same exact-rate / BA-shaped TX and
// level stage on the head pictures 1..32 only.  The drift-refresh schedule code
// and thresholds are unchanged (its refresh count still moves: it is decided
// from the reconstruction the head has just changed).
inline bool head_enabled() {
  static const bool value = !parse_off(std::getenv("AV2ENC_RD_HEAD_TX"));
  return value;
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

inline const Config& config() {
  static const Config cfg = [] {
    Config out;
    out.on = !parse_off(std::getenv("AV2ENC_RD_TAIL_TX"));
    if (const char* v = std::getenv("AV2ENC_RD_TAIL_LEVEL")) {
      out.level = !parse_off(v);
    }
    if (const char* v = std::getenv("AV2ENC_RD_TAIL_RATE")) {
      out.live_rate = !parse_off(v);
    }
    if (const char* v = std::getenv("AV2ENC_RD_TAIL_TX_D")) {
      out.ba_d = !parse_off(v);
    }
    if (const char* v = std::getenv("AV2ENC_RD_TAIL_STATS")) {
      out.stats = !parse_off(v);
    }
    out.level_scale =
        env_double("AV2ENC_RD_TAIL_LEVEL_SCALE", out.level_scale, false);
    out.lambda = env_double("AV2ENC_RD_TAIL_TX_LAMBDA", out.lambda, false);
    out.hf = env_double("AV2ENC_RD_TAIL_HF", out.hf, true);
    out.edge = env_double("AV2ENC_RD_TAIL_EDGE", out.edge, true);
    out.act = env_double("AV2ENC_RD_TAIL_ACT", out.act, true);
    return out;
  }();
  return cfg;
}

// Mean squared high-pass energy of one source luma block: the same
// (d - box3x3(d))^2 term the BA-shaped leaf D uses, on the source instead of
// the error.  This is the driver's activity screen (rd_driver's
// AV2ENC_RD_DRIVER_ACT) for sub-16 coding blocks.  Clamped reads, so block
// edges and the picture border are both defined.
inline double source_activity(const std::uint8_t* src, int width, int height,
                              int stride, int sx, int sy, int bw, int bh) {
  if (src == nullptr || bw <= 0 || bh <= 0 || width <= 0 || height <= 0) {
    return 0.0;
  }
  const auto sample = [&](int x, int y) -> int {
    const int px = sx + x < 0 ? 0 : (sx + x >= width ? width - 1 : sx + x);
    const int py = sy + y < 0 ? 0 : (sy + y >= height ? height - 1 : sy + y);
    return static_cast<int>(
        src[static_cast<std::size_t>(py) * static_cast<std::size_t>(stride) +
            static_cast<std::size_t>(px)]);
  };
  double sum = 0.0;
  for (int y = 0; y < bh; ++y) {
    for (int x = 0; x < bw; ++x) {
      int low = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) low += sample(x + dx, y + dy);
      }
      const double d =
          static_cast<double>(sample(x, y)) - static_cast<double>(low) / 9.0;
      sum += d * d;
    }
  }
  return sum / static_cast<double>(bw * bh);
}

// One input picture ordinal per frame, maintained by the shared tx_detail
// counter in `video_frame_internal.h` (advanced once per whole picture and not
// by the TIP second pass).  Passed in by the caller so this header stays
// independent of the frame encoder's declaration order.  -1 means the ordinal
// is unavailable.
// Process-wide CLI modes set by the InterFrameEncoder constructor from
// VideoOptions::tail_tx/head_tx (--video-tail-tx / --video-head-tx /
// --nightly-b).  Bit 0 = tail window, bit 1 = head window.
inline int& cli_mode() {
  static int value = 0;
  return value;
}

inline bool tail_armed(int ordinal) {
  const Config& cfg = config();
  const bool tail =
      (cfg.on || (cli_mode() & 1) != 0) && ordinal >= cfg.first &&
      ordinal <= cfg.last;
  if (tail) return true;
  if (head_enabled() || (cli_mode() & 2) != 0) {
    return ordinal >= head_first() && ordinal <= head_last();
  }
  return false;
}

struct Counters {
  long long leaves = 0;
  long long exact_ok = 0;
  long long exact_failed = 0;
  long long kept_exact = 0;
  long long plan_changed = 0;
  long long act_nodes = 0;
  long long act_fired = 0;
  long long act_nodes16 = 0;
  double screen_d = 0.0;
  double screen_r = 0.0;
  double exact_d = 0.0;
  double exact_r = 0.0;

  ~Counters() {
    const Config& cfg = config();
    if (!cfg.on && !cfg.stats && cli_mode() == 0) return;
    std::fprintf(
        stderr,
        "rd-tail-tx: leaves=%lld exact_ok=%lld failed=%lld kept=%lld "
        "plan_changed=%lld act_nodes=%lld act_fired=%lld act16=%lld "
        "screen[D=%.0f R=%.0f] exact[D=%.0f R=%.0f] "
        "lambda=%.2f level_scale=%.2f hf=%.2f edge=%.2f\n",
        leaves, exact_ok, exact_failed, kept_exact, plan_changed, act_nodes,
        act_fired, act_nodes16, screen_d, screen_r, exact_d, exact_r, cfg.lambda,
        cfg.level_scale, cfg.hf, cfg.edge);
  }
};

inline Counters& counters() {
  static Counters value;
  return value;
}

}  // namespace av2enc::video::tail_tx
