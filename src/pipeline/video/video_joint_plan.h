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
#include <chrono>
#include <mutex>
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
  // AV2ENC_RD_JOINT_BATCH: two-pass CPU batch for the same solve.  0 keeps the
  // serial raster walk (the reference); N > 0 runs pass 1 (candidate search)
  // on N workers against a frozen frame-start state and pass 2 (exact re-price
  // + raster commit) serially.  The committed stream can only differ where the
  // batch picked a different layout than the serial search would have.
  int batch = 0;
  // AV2ENC_RD_JOINT_BATCH_SEED=1 seeds every pass-1 superblock from the state
  // the *reference* layout leaves at that superblock's start (one extra serial
  // layout per superblock), instead of the frame-start state.  Closer to the
  // serial walk's raster context; 0 keeps the frame-start seed.
  bool batch_seed_ref = false;
  // AV2ENC_RD_JOINT_BATCH_ROUNDS: pass-1/pass-2 iterations.  1 is the plain
  // two-pass run.  N > 1 re-seeds pass 1 with the state pass 2 actually
  // committed at each superblock start, which is a Jacobi iteration on the
  // raster state: if the committed trees stop moving the result is a fixed
  // point of the serial walk's own recurrence.
  int batch_rounds = 1;
  // AV2ENC_RD_JOINT_BATCH_CHAIN: pass 1 hands each worker a *contiguous*
  // superblock range and chains the committed state through it, instead of
  // re-seeding every superblock from the pass-0 snapshot.  Only the first
  // superblock of each range starts from a stale state, so a changed layout
  // no longer poisons every later superblock.
  bool batch_chain = false;
};

// Set on a pass-1 worker thread.  Pass 1 measures the same leaf objective, but
// it runs against a frozen frame-start state, so its counters describe trials
// the serial walk never performs; they are not accumulated to keep the
// process-global counters race-free.  Its decisions are unaffected: nothing in
// the leaf coder reads a counter.
inline bool& trial_worker() {
  static thread_local bool value = false;
  return value;
}
inline bool counting() { return !trial_worker(); }

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
    out.batch = env_int("AV2ENC_RD_JOINT_BATCH", out.batch, 0, 256);
    out.batch_seed_ref =
        env_int("AV2ENC_RD_JOINT_BATCH_SEED", 0, 0, 1) != 0;
    out.batch_rounds = env_int("AV2ENC_RD_JOINT_BATCH_ROUNDS", 1, 1, 8);
    out.batch_chain =
        env_int("AV2ENC_RD_JOINT_BATCH_CHAIN", 0, 0, 1) != 0;
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

// AV2ENC_RD_JOINT_P_PROFILE: wall-clock split of the serial walk.  Pure
// instrumentation; every field is a monotonic-clock accumulation around an
// existing call, so an armed build codes the same stream.  The point is to
// size the batchable fraction before building the batch.
struct Profile {
  bool on = false;
  bool silent = false;
  // Frame-level stages.
  double me = 0.0;        // precompute_leaf_me()
  double pass1 = 0.0;     // batch pass 1 wall time (the parallel region)
  double pass2 = 0.0;     // batch pass 2 wall time (exact re-price + commit)
  double solve = 0.0;     // joint_p_decide_sb() (both passes of the walk)
  double commit = 0.0;    // the committed collect_node() in collect_frame()
  // Inside joint_p_decide_sb().
  double ref_walk = 0.0;  // measure_ref (reference tree, one full layout)
  double prune = 0.0;     // prune (candidate layouts)
  // Inside joint_p_measure_leaf().
  double leaf = 0.0;         // whole trial
  double leaf_encode = 0.0;  // collect_node: predict + mode + TX + recon
  double leaf_cost = 0.0;    // tail_floor_leaf_cost: D + exact rate
  double cost_d = 0.0;       // the distortion measure
  double cost_build = 0.0;   // rate_interval::build_from_syntax
  double cost_estimate = 0.0;  // rate_interval::estimate
  // Leaf-encode interior.
  double tx_search = 0.0;   // search_luma_tx()
  double tx_plain = 0.0;    // compute_tx() / chroma
  double me_search = 0.0;   // search_me() inside a trial
  long long me_calls = 0;
  long long me_hits = 0;
  // Leaf trial mix (counted at cost time).
  long long trials = 0;
  long long trials_inter = 0;
  long long trials_intra = 0;
  long long trials_skip = 0;
  long long committed_leaves = 0;
  long long committed_intra = 0;

  void dump() const {
    if (!on || silent) return;
    const double total = me + solve + commit;
    const auto pct = [total](double v) {
      return total > 0.0 ? 100.0 * v / total : 0.0;
    };
    std::fprintf(
        stderr,
        "rd-joint-p/profile: total=%.2fs me=%.2fs(%.1f%%) solve=%.2fs(%.1f%%) "
        "commit=%.2fs(%.1f%%) batch[pass1=%.2fs pass2=%.2fs] | solve: "
        "ref=%.2fs prune=%.2fs other=%.2fs | "
        "leaf=%.2fs encode=%.2fs cost=%.2fs [D=%.2fs build=%.2fs est=%.2fs] "
        "| leaf interior: tx=%.2fs tx_plain=%.2fs me=%.2fs "
        "[me calls=%lld cache=%lld] "
        "| trials=%lld inter=%lld intra=%lld skip=%lld committed=%lld "
        "intra=%lld\n",
        total, me, pct(me), solve, pct(solve), commit, pct(commit), pass1,
        pass2, ref_walk, prune, solve - ref_walk - prune, leaf, leaf_encode,
        leaf_cost,
        cost_d, cost_build,
        cost_estimate, tx_search, tx_plain, me_search, me_calls, me_hits,
        trials, trials_inter, trials_intra, trials_skip, committed_leaves,
        committed_intra);
  }
};

inline Profile& profile() {
  static Profile value = [] {
    Profile out;
    if (const char* v = std::getenv("AV2ENC_RD_JOINT_P_PROFILE")) {
      out.on = !parse_off(v);
    }
    return out;
  }();
  // Pass-1 worker threads accumulate into a throwaway profile: the serial
  // walk's numbers must not move just because the search ran in parallel.
  static thread_local Profile worker;
  if (!counting()) {
    worker.silent = true;
    return worker;
  }
  return value;
}

// The profile is a process-global static; the partition probe and the batch
// pass 1 call compute_tx/search_me from parallel workers.  Every write goes
// through this mutex so TSan (and the numbers) stay clean.
inline std::mutex& profile_mutex() {
  static std::mutex value;
  return value;
}

inline void profile_bump(long long* field) {
  if (!profile().on || field == nullptr) return;
  std::lock_guard<std::mutex> lock(profile_mutex());
  ++*field;
}

// A scoped monotonic-clock accumulator: `ScopedTimer t(&profile().me);`
class ScopedTimer {
 public:
  explicit ScopedTimer(double* sink) : sink_(sink), start_(Now()) {}
  ~ScopedTimer() {
    // Off is the common case: do not touch the shared static at all.
    if (sink_ == nullptr || !profile().on) return;
    const double elapsed = Now() - start_;
    std::lock_guard<std::mutex> lock(profile_mutex());
    *sink_ += elapsed;
  }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  static double Now() {
    if (!profile().on) return 0.0;
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch())
        .count();
  }
  double* sink_ = nullptr;
  double start_ = 0.0;
};

struct Counters {
  bool silent = false;
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
  // AV2ENC_RD_JOINT_BATCH pass 2.
  long long batch_reference_priced = 0;  // SBs whose reference was re-priced
  long long batch_reference_wins = 0;    // ... and beat the pass-1 candidate
  long long batch_candidate_kept = 0;    // pass-1 candidate committed
  long long batch_changed = 0;           // last round: layouts != reference
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
    if (silent) return;
    const Config& cfg = config();
    if (!cfg.on) return;
    profile().dump();
    if (!cfg.stats) return;
    std::fprintf(
        stderr,
        "rd-joint-p: sbs=%lld nodes=%lld layouts=%lld leaves=%lld "
        "split_trials=%lld kept[split=%lld leaf=%lld ref=%lld/%lld]"
        " failed=%lld[collect=%lld blocks=%lld cost=%lld entry=%lld"
        " build=%lld estimate=%lld] "
        "changed=%lld batch[ref_priced=%lld ref_wins=%lld cand=%lld] "
        "ref_leaf=%lld/%lld/%lld/%lld/%lld "
        "leaf=%lld/%lld/%lld/%lld/%lld J=%.0f refJ=%.0f dJ=%.0f "
        "D=%.0f R=%.0f "
        "head ref=%lld/%lld/%lld/%lld/%lld leaf=%lld/%lld/%lld/%lld/%lld "
        "tail ref=%lld/%lld/%lld/%lld/%lld leaf=%lld/%lld/%lld/%lld/%lld "
        "target=%d depth=%d lambda=%.2f hf=%.2f edge=%.2f\n",
        superblocks, nodes, layouts_measured, leaves_measured, split_trials,
        split_kept, leaf_kept, reference_kept, reference_wins, failed,
        fail_collect, fail_blocks, fail_cost, fail_entry, fail_build,
        fail_estimate, layouts_changed, batch_reference_priced,
        batch_reference_wins, batch_candidate_kept,
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
  // The worker is always silent: it is constructed on the first `counters()`
  // call of any thread, including a main thread that is counting, and its
  // destructor must not print an empty duplicate line at thread exit.
  static thread_local Counters worker = [] {
    Counters out;
    out.silent = true;
    return out;
  }();
  if (!counting()) {
    worker.silent = true;
    return worker;
  }
  return value;
}

}  // namespace av2enc::video::joint_plan
