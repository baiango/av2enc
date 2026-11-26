// AV2ENC_PIPE_BUDGET (default off): a budget-closed per-superblock allocation
// pipeline for a still picture, written from scratch.
//
// Why it exists.  At 0.21 bpp (src720.png at a 25 KB budget) every single
// superblock of the best measured stream is coded at ONE global qindex, so the
// RD tradeoff is uniform across the picture while the perceptual metrics are
// not: SSIMULACRA2 charges the 1-px band (docs/band-d.md, r = -0.82) and
// Butteraugli charges local contrast, both of which are concentrated in
// textured regions.  The existing per-superblock knob
// (AV2ENC_RD_DRIVER_SBALLOC) re-prices the *tree* only, which measured 0 to
// -1.5 K because it cannot move the quantiser.  This pipeline moves the
// quantiser: each superblock gets its own qindex, chosen by a greedy
// marginal-gain pass under one total-byte budget.
//
// Its own frame loop, its own leaf loop, its own partition/RD order:
//
//   for each superblock (raster):
//     for q in [lo, hi]:                       own leaf loop
//       price every leaf of the SUPERBLOCK'S OWN expanded tree at q
//       R(q), D(q)  (exact rates: seq_driver_measure_leaf's model, plus the
//                    real §8.2 size of the emitted evaluation stream)
//     keep the (q, layout, recon) with the best D(q) + lambda*R(q)
//   greedy allocation of the frame byte budget over superblocks by marginal
//   gain per byte (sorted, exact D deltas from the table above)
//   commit: re-encode each superblock at its allocated q in the committed pass
//
// Only the *decision order* and the *allocation* are new here; the leaf body
// is the encoder's own leaf coder (`encode_leaf`), because a new leaf body
// would have to re-implement the whole §5.20 syntax tree and would then be
// measuring a different decoder than the one the gate uses.
//
// The per-superblock qindex is carried by the §5.18.7.8 delta_q mechanism
// (delta_q_present = 1 with an explicit per-superblock delta), which the
// single-picture key frame can signal once the frame-schedule guard for a lone
// picture is lifted.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace av2enc::video::pipe_budget {

inline bool parse_off(const char* value) {
  if (value == nullptr || *value == '\0') return true;
  if (std::strcmp(value, "0") == 0) return true;
  return false;
}

// Thread-local off-switch: the pipeline's own per-superblock byte probe is a
// normal one-superblock key encode, and it must not re-enter the pipeline.
inline bool& suspend_flag() {
  static thread_local bool value = false;
  return value;
}

inline bool enabled() {
  static const bool value = [] {
    return !parse_off(std::getenv("AV2ENC_PIPE_BUDGET"));
  }();
  return value && !suspend_flag();
}

inline double env_number(const char* name, double fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(v, &end);
  if (end == v || !std::isfinite(parsed)) return fallback;
  return parsed;
}

inline int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(v, &end, 10);
  if (end == v) return fallback;
  return static_cast<int>(parsed);
}

struct Config {
  bool on = false;
  // Qindex grid the per-superblock search walks.
  int q_lo = 150;
  int q_hi = 230;
  int q_step = 2;
  // Frame byte budget the greedy pass closes on.  <= 0 means "the size the
  // uniform-qindex reference build produced", so the arm is always read at the
  // same bytes.
  long long budget_bytes = 0;
  // Per-metric weights of the leaf distortion (eff-perc luma gains are
  // separate, set by the caller through the preset).
  double luma_weight = 1.0;
  double chroma_weight = 1.0;
  // Marginal-gain exponent: gain / bytes^gamma.  gamma 1 is plain RD,
  // gamma < 1 spends earlier on big superblocks, gamma > 1 favours cheap wins.
  double gamma = 1.0;
  bool stats = false;
};

inline const Config& config() {
  static const Config cfg = [] {
    Config out;
    out.on = enabled();
    if (!out.on) return out;
    out.q_lo = env_int("AV2ENC_PIPE_BUDGET_Q_LO", out.q_lo);
    out.q_hi = env_int("AV2ENC_PIPE_BUDGET_Q_HI", out.q_hi);
    out.q_step = std::max(1, env_int("AV2ENC_PIPE_BUDGET_Q_STEP", out.q_step));
    out.budget_bytes = env_int("AV2ENC_PIPE_BUDGET_BYTES", 0);
    out.luma_weight = env_number("AV2ENC_PIPE_BUDGET_LUMA", out.luma_weight);
    out.chroma_weight =
        env_number("AV2ENC_PIPE_BUDGET_CHROMA", out.chroma_weight);
    out.gamma = env_number("AV2ENC_PIPE_BUDGET_GAMMA", out.gamma);
    out.stats = !parse_off(std::getenv("AV2ENC_PIPE_BUDGET_STATS"));
    return out;
  }();
  return cfg;
}

// One superblock's candidate: the qindex, its measured distortion and rate,
// and the emitted syntax + reconstruction the trial built.
struct CurvePoint {
  int qindex = 0;
  double distortion = 0.0;
  double rate_bits = 0.0;
  long long bytes = 0;
  double cost = 0.0;
};

// Greedy marginal-gain allocation over per-superblock (bytes, distortion)
// curves.  Returns one index into each curve (0 = finest/first point).
std::vector<int> allocate(const std::vector<std::vector<CurvePoint>>& curves,
                          long long budget_bytes, double gamma);

std::vector<int> allocate_marginal(
    std::vector<std::vector<CurvePoint>> curves,
    const std::vector<long long>& ref_bytes, long long budget_bytes,
    double gamma);

void print_summary(const std::vector<std::vector<CurvePoint>>& curves,
                   const std::vector<int>& pick, long long budget);

}  // namespace av2enc::video::pipe_budget
