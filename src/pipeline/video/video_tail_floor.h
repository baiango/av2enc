// AV2ENC_RD_TAIL_FLOOR (default off): measured P-frame coding-block floor on
// the tail pictures (33-59).
//
// The tail exact-rate lane (docs/rd-tail.md) measured the P-frame coding-block
// floor at 16x16 on this clip: the reference tree the partition probe solves is
// bounded at partition depth 4, so the 16x16 node is a leaf and no 8x8-class
// coding block exists.  The lane's forced sub-16 descent (AV2ENC_RD_TAIL_ACT)
// was negative because it replaced a measured decision with a source-activity
// screen: the tree retreated to coarser ancestors instead of producing 8x8
// leaves.
//
// This gate lowers the floor only where a *measured* decision says the smaller
// coding block is cheaper.  At every 16x16 tail leaf:
//
//   screen  the existing per-block partition probe (one NEWMV + one quantised
//           residual per candidate node, `partition_probe_leaf_cost`) prices
//           NONE against the legal HORZ / VERT layouts and their 8x8-class
//           descent.  A node whose screen prefers NONE keeps the 16x16 block
//           and costs nothing extra.
//   exact   both candidates are encoded with the coder's own leaf path into
//           the live reconstruction -- sequentially, in raster order, after
//           every earlier superblock was committed -- and scored with one
//           objective:
//
//             D = seq_rd::leaf_distortion_gains() on the leaf's own
//                 reconstruction (eff-perc inter low-band CSF + the driver's
//                 optional high-pass / block-edge terms) + weighted chroma SSE
//             R = rate_interval::estimate() on the emitted §8.3 symbol
//                 sequence (live frame-start CDF image, TX partition/type) +
//                 the live inter-syntax probe (is_inter/skip/mode/ref/DRL/MVD)
//                 + 2 bits per partition symbol
//             J = D + lambda * R,  lambda = the leaf's own pipe lambda
//
// The winner's encode *is* the committed encode (the trial state is kept), so
// a node whose leaf wins costs one extra split trial and no re-encode.
//
// Nothing runs on the key picture, on non-tail pictures, on tiled encodes, on
// SDP frames or under delta-Q/segmentation/palette/IBC.  Unset, the hook is a
// single false branch and every pinned stream is byte-identical.
#pragma once

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace av2enc::video::tail_floor {

struct Config {
  bool on = false;
  bool screen = true;
  bool stats = false;
  int first = 33;
  int last = 59;
  // Deepest coding block the tie-breaking descent may create.  8 is the
  // §5.20.3.1 inter floor (a P-frame inter leaf owns its chroma block); 16
  // reduces the gate to the two 16x8/8x16 splits.
  int min_px = 8;
  // Partition levels below the 16x16 root: 1 = HORZ/VERT only, 2 = also the
  // 8x8-level descent.  Measured on the 240p tail (docs/rd-tail-floor.md):
  // depth 2 is flat/negative (the 8x8 quadtree's four mode/skip sets cost more
  // than they save) and depth 1 is the measured-best composition (f0-32
  // ceiling fit +0.033, K fit +0.023, tail payload -0.5 % at the same QP).
  // The default stays 2 because that is the floor this lane was asked to
  // measure; enable 1 for the positive point.
  int depth = 2;
  // Screen margin in estimated bits: the probe split must beat the probe leaf
  // by this much to spend an exact trial.  0 = the probe's own decision.
  double screen_margin = 0.0;
  // Plan-J lambda as a fraction of the inter leaf's pipe lambda.
  double lambda = 1.0;
  // High-pass / block-edge weights of the measured D (driver defaults: 0).
  double hf = 0.0;
  double edge = 0.0;
};

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

inline const Config& config() {
  static const Config cfg = [] {
    Config out;
    out.on = !parse_off(std::getenv("AV2ENC_RD_TAIL_FLOOR"));
    if (const char* v = std::getenv("AV2ENC_RD_TAIL_FLOOR_SCREEN")) {
      out.screen = !parse_off(v);
    }
    if (const char* v = std::getenv("AV2ENC_RD_TAIL_FLOOR_STATS")) {
      out.stats = !parse_off(v);
    }
    out.min_px = env_int("AV2ENC_RD_TAIL_FLOOR_MIN", out.min_px, 8, 16);
    out.depth = env_int("AV2ENC_RD_TAIL_FLOOR_DEPTH", out.depth, 1, 2);
    out.first = env_int("AV2ENC_RD_TAIL_FLOOR_FIRST", out.first, 0, 4096);
    out.last = env_int("AV2ENC_RD_TAIL_FLOOR_LAST", out.last, 0, 4096);
    if (out.last < out.first) out.last = out.first;
    out.screen_margin = env_double("AV2ENC_RD_TAIL_FLOOR_MARGIN",
                                   out.screen_margin, true);
    out.lambda =
        env_double("AV2ENC_RD_TAIL_FLOOR_LAMBDA", out.lambda, false);
    out.hf = env_double("AV2ENC_RD_TAIL_FLOOR_HF", out.hf, true);
    out.edge = env_double("AV2ENC_RD_TAIL_FLOOR_EDGE", out.edge, true);
    return out;
  }();
  return cfg;
}

// One input-picture ordinal per frame, the same counter the tail exact-rate
// gate uses (`tx_detail::picture_ordinal()`), passed in by the caller.
inline bool armed(int ordinal) {
  const Config& cfg = config();
  if (!cfg.on) return false;
  return ordinal >= cfg.first && ordinal <= cfg.last;
}

struct Counters {
  long long nodes = 0;        // 16x16 tail leaves considered
  long long screened = 0;     // screen evaluated
  long long flagged = 0;      // screen spent an exact trial
  long long trials = 0;       // exact split layouts encoded
  long long trial_failed = 0;
  long long fallback = 0;     // candidate could not be measured
  long long none_kept = 0;
  long long split_kept = 0;
  long long split_leaves = 0;   // leaves the kept splits emitted
  long long leaves_16 = 0;      // kept 16x16 coding blocks
  long long leaves_8 = 0;       // kept 8x8/8x16/16x8 coding blocks
  long long leaves_4 = 0;       // never legal for inter; kept for the report
  double screen_bits = 0.0;     // sum of (screen leaf - screen split) in bits
  double d_none = 0.0;
  double r_none = 0.0;
  double d_split = 0.0;
  double r_split = 0.0;
  double j_delta = 0.0;         // sum of J_split - J_none on decided nodes

  ~Counters() {
    const Config& cfg = config();
    if (!cfg.on && !cfg.stats) return;
    std::fprintf(
        stderr,
        "rd-tail-floor: nodes=%lld screened=%lld flagged=%lld trials=%lld "
        "failed=%lld fallback=%lld none=%lld split=%lld split_leaves=%lld "
        "leaves16/8/4=%lld/%lld/%lld screen_bits=%.1f "
        "none[D=%.0f R=%.0f] split[D=%.0f R=%.0f] dJ=%.1f "
        "min=%d depth=%d screen=%d margin=%.2f lambda=%.2f hf=%.2f edge=%.2f\n",
        nodes, screened, flagged, trials, trial_failed, fallback, none_kept,
        split_kept, split_leaves, leaves_16, leaves_8, leaves_4, screen_bits,
        d_none, r_none, d_split, r_split, j_delta, cfg.min_px, cfg.depth,
        cfg.screen ? 1 : 0, cfg.screen_margin, cfg.lambda, cfg.hf, cfg.edge);
  }
};

inline Counters& counters() {
  static Counters value;
  return value;
}

}  // namespace av2enc::video::tail_floor
