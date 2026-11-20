// AV2ENC_PART_DIAG (default off): measure both sides of every P-frame leaf.
//
// The measured partition probe is bounded to `partition_probe_depth` = 1, so
// the P tree's leaves are priced one level down and every deeper node is
// priced by the NEWMV-only probe.  This gate does not change the stream: at
// every emitted leaf it encodes the leaf and each legal one-level split with
// the coder's own leaf path, scores both with the same objective the tail
// floor uses (BA-shaped D on the leaf's own reconstruction + live §8.2.6
// interval rate + 2 bits per partition symbol) and commits the reference
// leaf.  It answers one question before a partition driver is built:
//
//   for what share of leaves is J_split < J_leaf, and by how much?
//
// If the answer is "few leaves, tiny dJ", the tree is priced correctly and a
// P-frame sequential driver has nothing to win.  If a third of the leaves
// want to split, the driver is worth building.
//
// The gate runs the same whole-picture sequential walk as the tail floor so
// each trial is priced in the CDF state the real encode will have.  Inter
// leaves do not depend on the current frame's neighbours (prediction is from
// the reference picture), so a trial is exact for them; intra leaves inside P
// are the approximation (they read the current recon), and they are counted
// separately through the emitted block's is_inter flag only in the histogram.
#pragma once

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace av2enc::video::part_diag {

struct Config {
  bool on = false;
  // Pictures to measure.  Defaults to the head, where the probe clamp leaves
  // 70 % of the leaves at 64x64 / 64x32.
  int first = 1;
  int last = 32;
  // Leaves with a smaller side are skipped (the tail floor already measures
  // the 16x16 -> 8x8-class descent).
  int min_px = 16;
  // Plan-J lambda as a fraction of the leaf's own pipe lambda.
  double lambda = 1.0;
  // Optional high-pass / block-edge weights of the measured D.
  double hf = 0.0;
  double edge = 0.0;
  // 0 = measure every candidate (the diagnostic); >0 stops after this many.
  long long max_nodes = 0;
  bool stats = false;
  // Commit the measured winner instead of the reference leaf.  Only for the
  // validation run: it turns the diagnostic into the decision it is sizing.
  bool commit = false;
};

inline bool off_value(const char* value) {
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
    out.on = !off_value(std::getenv("AV2ENC_PART_DIAG"));
    out.first = env_int("AV2ENC_PART_DIAG_FIRST", out.first, 0, 4096);
    out.last = env_int("AV2ENC_PART_DIAG_LAST", out.last, 0, 4096);
    if (out.last < out.first) out.last = out.first;
    out.min_px = env_int("AV2ENC_PART_DIAG_MIN", out.min_px, 8, 64);
    out.lambda =
        env_double("AV2ENC_PART_DIAG_LAMBDA", out.lambda, false);
    out.hf = env_double("AV2ENC_PART_DIAG_HF", out.hf, true);
    out.edge = env_double("AV2ENC_PART_DIAG_EDGE", out.edge, true);
    if (const char* v = std::getenv("AV2ENC_PART_DIAG_MAX")) {
      char* end = nullptr;
      const long long n = std::strtoll(v, &end, 10);
      if (end != v && n > 0) out.max_nodes = n;
    }
    out.stats = !off_value(std::getenv("AV2ENC_PART_DIAG_STATS"));
    out.commit = !off_value(std::getenv("AV2ENC_PART_DIAG_COMMIT"));
    return out;
  }();
  return cfg;
}

inline bool armed(int ordinal) {
  const Config& cfg = config();
  if (!cfg.on) return false;
  return ordinal >= cfg.first && ordinal <= cfg.last;
}

enum ShapeIndex : int {
  k64x64 = 0,
  k64x32 = 1,
  k32x32 = 2,
  k32x16 = 3,
  k16x16 = 4,
  kOther = 5,
  kShapeCount = 6,
};

inline int shape_of(int w, int h) {
  const int mn = w < h ? w : h;
  const int mx = w > h ? w : h;
  if (mn == 64 && mx == 64) return k64x64;
  if (mn == 32 && mx == 64) return k64x32;
  if (mn == 32 && mx == 32) return k32x32;
  if (mn == 16 && mx == 32) return k32x16;
  if (mn == 16 && mx == 16) return k16x16;
  return kOther;
}

inline const char* shape_name(int index) {
  switch (index) {
    case k64x64: return "64x64";
    case k64x32: return "64x32";
    case k32x32: return "32x32";
    case k32x16: return "32x16";
    case k16x16: return "16x16";
    default: return "other";
  }
}

// One node's measurement, in bits / J units.  `r_*` includes the live inter
// syntax gold plus the residual interval; `signal` is 2 bits on the split.
struct Sample {
  int shape = kOther;
  double d_leaf = 0.0;
  double r_leaf = 0.0;
  double d_split = 0.0;
  double r_split = 0.0;
  int part = 0;  // 0 none, 1 horz, 2 vert, 3 split
  bool ok = false;
};

struct ShapeStats {
  long long nodes = 0;
  long long split_wins = 0;
  double d_leaf = 0.0;
  double r_leaf = 0.0;
  double d_split = 0.0;
  double r_split = 0.0;
  double dj = 0.0;
  double dj_neg = 0.0;
  double r_leaf_win = 0.0;
  double r_split_win = 0.0;
};

struct Counters {
  ShapeStats by_shape[kShapeCount];
  ShapeStats total;
  long long trials = 0;
  long long fails = 0;
  long long capped = 0;
  long long partitions[4] = {0, 0, 0, 0};

  void add(const Sample& s) {
    if (!s.ok) {
      ++fails;
      return;
    }
    ShapeStats& a = by_shape[s.shape];
    const double dj = (s.d_split + lambda_ * s.r_split) -
                      (s.d_leaf + lambda_ * s.r_leaf);
    for (ShapeStats* p : {&a, &total}) {
      ++p->nodes;
      p->d_leaf += s.d_leaf;
      p->r_leaf += s.r_leaf;
      p->d_split += s.d_split;
      p->r_split += s.r_split;
      p->dj += dj;
      if (dj < 0.0) {
        ++p->split_wins;
        p->dj_neg += dj;
        p->r_leaf_win += s.r_leaf;
        p->r_split_win += s.r_split;
      }
    }
    if (dj < 0.0) ++partitions[s.part];
  }

  // The lambda the caller measured with, set once per picture before add().
  double lambda_ = 1.0;

  ~Counters() {
    const Config& cfg = config();
    if (!cfg.on && !cfg.stats) return;
    const double leaf_bytes = total.r_leaf / 8.0;
    const double win_bytes = (total.r_split_win - total.r_leaf_win) / 8.0;
    std::fprintf(
        stderr,
        "rd-part-diag: nodes=%lld split_wins=%lld trials=%lld fails=%lld "
        "capped=%lld parts[h/v/s]=%lld/%lld/%lld dJ=%.1f dJ_neg=%.1f "
        "Dleaf=%.0f Dsplit=%.0f "
        "Rleaf=%.0fb(%.1fB) dR_win=%+.0fb(%+.1fB) "
        "min=%d lambda=%.2f hf=%.2f edge=%.2f\n",
        total.nodes, total.split_wins, trials, fails, capped, partitions[1],
        partitions[2], partitions[3], total.dj, total.dj_neg, total.d_leaf,
        total.d_split, total.r_leaf, leaf_bytes,
        total.r_split_win - total.r_leaf_win, win_bytes, cfg.min_px,
        cfg.lambda, cfg.hf, cfg.edge);
    for (int i = 0; i < kShapeCount; ++i) {
      const ShapeStats& s = by_shape[i];
      if (s.nodes == 0) continue;
      std::fprintf(
          stderr,
          "rd-part-diag  %-6s n=%lld split=%lld dJ=%.1f dJ_neg=%.1f "
          "dR_win=%+.1fB\n",
          shape_name(i), s.nodes, s.split_wins, s.dj, s.dj_neg,
          (s.r_split_win - s.r_leaf_win) / 8.0);
    }
  }
};

inline Counters& counters() {
  static Counters value;
  return value;
}

}  // namespace av2enc::video::part_diag
