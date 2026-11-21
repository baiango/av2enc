// AV2ENC_RD_JOINT_P (default off): measured P-frame joint plan search.
//
// The P chain currently solves partition, TX/levels and mode in three
// separate stages: `solve_partition()` prices the tree with the per-block
// NEWMV probe clamped to `partition_probe_depth`, `encode_leaf()` then picks
// mode + TX/levels for the leaf that probe chose, and the head/tail exact-rate
// stages refine the TX/levels afterwards without revisiting the tree
// (docs/rd-part-diag.md).
//
// This module is the P-frame port of the key driver
// (`InterFrameEncoder::seq_driver_decide_sb`, docs/rd-engine0.md §5).  One
// objective decides all three at once:
//
//   leaf     J = D_P(recon the leaf emitted) + lambda_P * R_exact(leaf)
//   split    J = sum(children J) + lambda_P * 2 bits
//
//   D_P      `seq_rd::leaf_distortion_gains()` on the leaf's own
//            reconstruction: the eff-perc **inter** low-band CSF (not the key
//            CSF) + weighted chroma SSE, with the optional high-pass /
//            block-edge terms.
//   R_exact  the emitted §8.3 symbols through the §8.2.6 interval model
//            (`rate_interval::estimate`, live frame-start CDF image) plus the
//            live inter-syntax group (is_inter / skip / mode / ref / DRL / MVD)
//            the pipe probe stashed while the leaf was written.
//   lambda_P `leaf_rd_lambda(qindex) * eff_perc::lambda_scale()`, the lambda
//            the P leaf and its TX/level search minimise with -- never the key
//            driver's `key_tx_rd_lambda`.
//
// Every candidate layout is encoded with the coder's own leaf path
// (`collect_node()` -> `encode_leaf()`), in the live reconstruction, in raster
// order after every earlier superblock was committed.  Inter prediction comes
// from the previous filtered frame, so an inter trial does not depend on the
// current recon; intra leaves inside P do, and the sequential walk is what
// makes them exact.  Trials snapshot and restore the reconstruction,
// §6.19.2.3 BlockDecoded, the §5.20.1/§5.20.5.7 state, the inter syntax probe
// and the per-leaf rate stash, so the committed encode is the only one that
// advances the frame.
//
// The reference tree is always a candidate: a node the search cannot price
// keeps the reference layout, so the gate can never make a node unsolvable.
// Unset, the hook is one false branch and every pinned stream is byte-
// identical.
#pragma once

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace av2enc::video::joint_plan {

struct Config {
  bool on = false;
  bool stats = true;
  // Display-order picture window.  The flat P chain is 1..59; the key picture
  // and the chained-B carriers are outside this arm.
  int first = 1;
  int last = 4096;
  // Deepest uniform leaf (px) the expanded tree reaches.  8 is the
  // §5.20.3.1 inter coding-block floor; unlike the key picture there is no
  // legal 4x4 inter leaf (chroma ownership needs an 8x8 group).
  int target = 8;
  // Policy depth cap on the expansion, plus the encoder's own cap.
  int max_depth = 8;
  // Plan-J lambda as a fraction of the P leaf's own pipe lambda.
  double lambda = 1.0;
  // High-pass / block-edge weights of the measured D (driver defaults: 0).
  double hf = 0.0;
  double edge = 0.0;
};

inline bool parse_off(const char* value) {
  if (value == nullptr || *value == '\0') return true;
  if (std::strcmp(value, "0") == 0) return true;
  const auto equal = [value](const char* word) {
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
    out.on = !parse_off(std::getenv("AV2ENC_RD_JOINT_P"));
    if (const char* v = std::getenv("AV2ENC_RD_JOINT_P_STATS")) {
      out.stats = !parse_off(v);
    }
    out.first = env_int("AV2ENC_RD_JOINT_P_FIRST", out.first, 0, 8192);
    out.last = env_int("AV2ENC_RD_JOINT_P_LAST", out.last, 0, 8192);
    if (out.last < out.first) out.last = out.first;
    out.target = env_int("AV2ENC_RD_JOINT_P_TARGET", out.target, 8, 64);
    out.max_depth =
        env_int("AV2ENC_RD_JOINT_P_DEPTH", out.max_depth, 1, 8);
    out.lambda =
        env_double("AV2ENC_RD_JOINT_P_LAMBDA", out.lambda, false);
    out.hf = env_double("AV2ENC_RD_JOINT_P_HF", out.hf, true);
    out.edge = env_double("AV2ENC_RD_JOINT_P_EDGE", out.edge, true);
    return out;
  }();
  return cfg;
}

inline bool armed(int ordinal) {
  const Config& cfg = config();
  if (!cfg.on) return false;
  return ordinal >= cfg.first && ordinal <= cfg.last;
}

// One bucket per short side: 4, 8, 16, 32, 64.
inline int bucket(int width, int height) {
  const int side = width < height ? width : height;
  int index = 0;
  while (index < 4 && side > (4 << index)) ++index;
  return index;
}

struct Counters {
  long long superblocks = 0;
  long long nodes = 0;             // prune nodes visited
  long long layouts_measured = 0;  // candidate layouts encoded
  long long leaves_measured = 0;   // leaf trials encoded
  long long split_trials = 0;      // split candidates measured
  long long split_kept = 0;        // nodes whose measured split won
  long long leaf_kept = 0;         // nodes whose measured leaf won
  long long reference_kept = 0;    // node whose trial failed -> reference
  long long reference_wins = 0;    // superblocks where the reference tree won
  long long failed = 0;            // unpriceable candidate
  long long fail_collect = 0;      // collect_node() failed in a trial
  long long fail_blocks = 0;       // trial did not write exactly one entry
  long long fail_cost = 0;         // exact leaf cost failed
  long long fail_entry = 0;        // leaf entry not covered by the probe
  long long fail_build = 0;        // residual input build failed
  long long fail_estimate = 0;     // interval estimate failed
  long long layouts_changed = 0;   // superblocks whose tree != reference
  long long ref_leaves[5] = {0, 0, 0, 0, 0};
  long long chosen_leaves[5] = {0, 0, 0, 0, 0};
  // Display-order split: f1-32 (head) / f33-59 (tail).
  long long head_ref[5] = {0, 0, 0, 0, 0};
  long long head_chosen[5] = {0, 0, 0, 0, 0};
  long long tail_ref[5] = {0, 0, 0, 0, 0};
  long long tail_chosen[5] = {0, 0, 0, 0, 0};
  double chosen_j = 0.0;   // committed trees (reference where it won)
  double ref_j = 0.0;      // reference trees (all measured superblocks)
  double d_measured = 0.0;
  double r_measured = 0.0;

  ~Counters() {
    const Config& cfg = config();
    if (!cfg.on || !cfg.stats) return;
    std::fprintf(
        stderr,
        "rd-joint-p: sbs=%lld nodes=%lld layouts=%lld leaves=%lld "
        "split_trials=%lld kept[split=%lld leaf=%lld ref=%lld/%lld]"
        " failed=%lld[collect=%lld blocks=%lld cost=%lld entry=%lld"
        " build=%lld estimate=%lld] "
        "changed=%lld ref_leaf=%lld/%lld/%lld/%lld/%lld "
        "leaf=%lld/%lld/%lld/%lld/%lld J=%.0f refJ=%.0f dJ=%.0f "
        "D=%.0f R=%.0f "
        "head ref=%lld/%lld/%lld/%lld/%lld leaf=%lld/%lld/%lld/%lld/%lld "
        "tail ref=%lld/%lld/%lld/%lld/%lld leaf=%lld/%lld/%lld/%lld/%lld "
        "target=%d depth=%d lambda=%.2f hf=%.2f edge=%.2f\n",
        superblocks, nodes, layouts_measured, leaves_measured, split_trials,
        split_kept, leaf_kept, reference_kept, reference_wins, failed,
        fail_collect, fail_blocks, fail_cost, fail_entry, fail_build,
        fail_estimate, layouts_changed,
        ref_leaves[0], ref_leaves[1], ref_leaves[2], ref_leaves[3],
        ref_leaves[4], chosen_leaves[0], chosen_leaves[1], chosen_leaves[2],
        chosen_leaves[3], chosen_leaves[4], chosen_j, ref_j,
        ref_j - chosen_j, d_measured, r_measured,
        head_ref[0], head_ref[1], head_ref[2], head_ref[3], head_ref[4],
        head_chosen[0], head_chosen[1], head_chosen[2], head_chosen[3],
        head_chosen[4], tail_ref[0], tail_ref[1], tail_ref[2], tail_ref[3],
        tail_ref[4], tail_chosen[0], tail_chosen[1], tail_chosen[2],
        tail_chosen[3], tail_chosen[4],
        cfg.target, cfg.max_depth, cfg.lambda, cfg.hf, cfg.edge);
  }
};

inline Counters& counters() {
  static Counters value;
  return value;
}

}  // namespace av2enc::video::joint_plan
