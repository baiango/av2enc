// Exact-RD experiment gate (AV2ENC_EXACT_RD).
//
// The historical leaf decision ranks inter candidates by prediction SSE plus a
// static mode/MV syntax estimate, then runs the real predict -> TX search ->
// quantise -> reconstruct -> rate path only for the winner.  The partition
// probe prices every node with a single NEWMV + quantised-residual probe.
// Both are proxies: a candidate with a slightly worse prediction can code a
// much cheaper residual, and a partition node's cost is not the cost of the
// mode the leaf would actually choose.
//
// This gate switches the encoder to an exact search:
//
//   AV2ENC_EXACT_RD=1          enable the exact path (default off)
//   AV2ENC_EXACT_RD_TOP=<n>    number of cheap candidates to re-score (2)
//   AV2ENC_EXACT_RD_PARTITION=0
//                              keep the NEWMV partition probe only (ablation)
//
// The gate is read once per process: it is configuration, not per-frame state.
// Unset (or 0 / false / off) leaves every code path untouched, so every pinned
// stream keeps its bytes.  There is deliberately no CLI flag.
#pragma once

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace av2enc::video::exact_rd {

// One leaf's measured cost, recorded by encode_leaf() for the partition
// refinement.  Distortion is the SSE of the reconstruction the leaf actually
// emitted; bits is the same estimated syntax+coefficient rate the leaf
// decision used, so D + lambda * R is a real objective and not a proxy.  The
// caller supplies lambda (the partition price) because the probe and the leaf
// decision use different calibrated scales.
struct LeafCost {
  double distortion = 0.0;
  double bits = 0.0;
  bool valid = false;
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

inline void print_stats();

inline bool enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_EXACT_RD");
    return !parse_off(env);
  }();
  // Register the one-line summary exactly once, from whichever thread first
  // asks.  print_stats() reads the already-initialised static above, so the
  // atexit callback cannot re-enter the env parse.
  static const bool registered = [] {
    if (value) std::atexit(&print_stats);
    return true;
  }();
  (void)registered;
  return value;
}

// Top-N candidate rescoring.  Clamped to 1..8 so a typo cannot turn a gate run
// into an accidental full-candidate search.  N=1 reproduces the historical
// "winner only" evaluation while still recording the exact cost.
inline int top_n() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_EXACT_RD_TOP");
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

// The partition refinement is part of the gate; set the value to 0 to measure
// the leaf rescoring alone.
inline bool partition_refine() {
  static const bool disabled = [] {
    const char* env = std::getenv("AV2ENC_EXACT_RD_PARTITION");
    // Unlike the master gate, this switch is on by default: only an explicit
    // 0/false/off disables it, so `AV2ENC_EXACT_RD=1` alone gets both parts.
    return env != nullptr && *env != '\0' && parse_off(env);
  }();
  return enabled() && !disabled;
}

// Counters are process-global because one encoder object exists per tile/frame
// and the wavefront may run several superblocks concurrently.  They are only
// touched when the gate is on; gate-off streams never read or write them.
struct Counters {
  std::atomic<unsigned long long> leaves{0};
  std::atomic<unsigned long long> rescored{0};
  std::atomic<unsigned long long> inter_winner_changed{0};
  std::atomic<unsigned long long> intra_flipped{0};
  std::atomic<unsigned long long> proxy_residual{0};
  std::atomic<unsigned long long> exact_residual{0};
  std::atomic<unsigned long long> partition_overrides{0};
  std::atomic<unsigned long long> partition_tree_changed{0};
  std::atomic<unsigned long long> partition_refine_runs{0};
};

inline Counters& counters() {
  static Counters value;
  return value;
}

inline void print_stats() {
  if (!enabled()) return;
  const Counters& c = counters();
  std::fprintf(stderr,
               "exact-rd: top_n=%d partition=%d leaves=%llu rescored=%llu "
               "inter_winner_changed=%llu intra_flipped=%llu "
               "proxy_residual=%llu exact_residual=%llu "
               "partition_overrides=%llu partition_tree_changed=%llu "
               "partition_refine_runs=%llu\n",
               top_n(), partition_refine() ? 1 : 0,
               static_cast<unsigned long long>(c.leaves.load()),
               static_cast<unsigned long long>(c.rescored.load()),
               static_cast<unsigned long long>(c.inter_winner_changed.load()),
               static_cast<unsigned long long>(c.intra_flipped.load()),
               static_cast<unsigned long long>(c.proxy_residual.load()),
               static_cast<unsigned long long>(c.exact_residual.load()),
               static_cast<unsigned long long>(c.partition_overrides.load()),
               static_cast<unsigned long long>(c.partition_tree_changed.load()),
               static_cast<unsigned long long>(c.partition_refine_runs.load()));
}

}  // namespace av2enc::video::exact_rd
