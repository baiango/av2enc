// Sequential-reconstruction key partition tree (AV2ENC_RD_SEQTREE).
//
// rd-engine0 note: this copy is used only for `leaf_distortion()` by the
// AV2ENC_RD_DRIVER driver.  The SEQTREE gate/knobs/counters below are inert
// here (they are the rd-seq lane's own surface); the driver reads
// AV2ENC_RD_DRIVER_HF / AV2ENC_RD_DRIVER_EDGE.
//
// RD-engine lane "seq": every key-picture partition decision is measured in the
// reconstruction the candidate layout itself builds.  The earlier lane-2 tree
// priced a candidate geometry against the pass-1 picture, which is a fully
// reconstructed neighbourhood at the *reference* tree's quality; a small leaf
// that will be surrounded by its own tree's quantisation noise looks almost
// free there.  704 of 712 accepted leaves were under-priced for that reason.
//
// This driver removes the stale context by construction: superblocks are
// decided in raster order and, inside a superblock, leaves are encoded in
// raster order; each candidate layout is encoded with the coder's own leaf
// path (`encode_leaf`) into the live reconstruction, so D and R are the values
// the layout actually produces.  A candidate that cannot be encoded this way
// is not scored.
//
// Gate (env only, default off; no CLI flag, no CMake/main/options changes):
//   AV2ENC_RD_SEQTREE=1             master gate
//   AV2ENC_RD_SEQTREE_ROUNDS=<n>    refinement rounds per superblock (2, 0..4)
//   AV2ENC_RD_SEQTREE_TOPK=<n>      screened variants measured exactly (2, 1..6)
//   AV2ENC_RD_SEQTREE_SCREEN=<n>    prediction modes the cheap screen tries (7)
//   AV2ENC_RD_SEQTREE_SCALE=<x>     lambda multiplier on the key TX lambda (1.0)
//   AV2ENC_RD_SEQTREE_DEPTH=<n>     maximum partition depth (8)
//   AV2ENC_RD_SEQTREE_STATS=0       silence the one-line summary
//
// Unset (or 0/false/off) does no work at all: no leaf record is written and the
// pass-1 tree is the tree, so every pinned stream stays byte-identical.
//
// ## Cost
//
//     J = D_perc(luma recon the leaf emitted) + lambda * R(luma)
//
//   D  eff-perc key distortion of the reconstructed luma block (visible rows,
//      the frame's key CSF gains), computed from the live recon after the leaf
//      coder wrote it.
//   R  the luma rate the leaf coder priced for that leaf: the searched plan's
//      §8.3 model bits (`TxPlan::rate_bits`, TX-partition symbols included),
//      the luma mode/MRL/DIP symbols and the 2-bit prediction-mode signal.
//   lambda  key_tx_rd_lambda(options, qindex) * key TX scale * SEQTREE_SCALE.
//      The key TX plan is chosen at that lambda, so the layout comparison and
//      the plan comparison optimise one objective instead of two.
//
// Internal nodes are charged `partition_bits()` per split (the same 2-bit
// constant the historical key tree uses; the symbol is modelled, not coded).
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pipeline/video/video_eff_perc.h"

namespace av2enc::video::seq_rd {

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

inline bool enabled() {
  static const bool value = !parse_off(std::getenv("AV2ENC_RD_SEQTREE"));
  return value;
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

inline double env_double(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value) return fallback;
  return parsed;
}

inline int rounds() { return env_int("AV2ENC_RD_SEQTREE_ROUNDS", 2, 0, 4); }
inline int top_k() { return env_int("AV2ENC_RD_SEQTREE_TOPK", 2, 1, 6); }
inline int screen_modes() {
  return env_int("AV2ENC_RD_SEQTREE_SCREEN", 7, 1, 16);
}
inline int max_depth() { return env_int("AV2ENC_RD_SEQTREE_DEPTH", 8, 1, 8); }
inline double lambda_scale() {
  return env_double("AV2ENC_RD_SEQTREE_SCALE", 1.0);
}

// Block-edge / fine-detail terms (see leaf_distortion below).  The eff-perc key
// D is a *low-band* objective: it prices the 3x3 box mean error and is blind to
// the error a partition boundary leaves behind.  An 8x8-heavy key therefore
// looks cheap to the tree while Butteraugli sees the block edges.  These two
// weights put that error back into the measured D:
//
//   HF    sum (d - box3x3(d))^2, the high-pass part of the error (fine detail
//         the reconstruction lost, including the ramp at a block edge);
//   EDGE  sum ((src_in - src_out) - (rec_in - rec_out))^2 over the leaf's top
//         and left boundary, i.e. the gradient mismatch across the block edge
//         the decoder will actually show (causal neighbours only: the recon
//         already holds them).
//
// Both are measured on the leaf's own reconstruction, in the same call, so the
// record and the post-pass verification compute one number.
// Driver default 0.0: HF 20 puts the high-pass term at ~14x the low-band term
// and improves the key picture's clip-BA proxy, but on the 240p clip it loses
// matched K (docs/rd-engine0.md §7), so the gate-on default keeps the term
// available but off.  EDGE defaults off for the same reason.
inline double hf_gain() {
  // Default off: the term is implemented and measured (HF 5/20 improve the
  // key picture's BA but lose the clip's, docs/rd-engine0.md §7), so the
  // gate-on default keeps the best measured K point.  Set >0 to enable.
  return env_double("AV2ENC_RD_DRIVER_HF", 0.0);
}
inline double edge_gain() {
  return env_double("AV2ENC_RD_DRIVER_EDGE", 0.0);
}

// AV2ENC_TX_LAMBDA_TEXTURED (default 0 = off): scale the KEY leaf TX lambda by
// leaf texture content.  Unlike every arm in docs/k52-budget-pipe.md §8, this
// SPENDS precision instead of trading it for rate: a textured leaf gets a
// SMALLER lambda, so its RD cost prefers finer coefficient levels.  The
// candidate set and the syntax are unchanged; only the price changes.
//
// docs/k52-budget-pipe.md §4 measured 75.7% of our extra luma error against the
// reference encoder in high-activity blocks, where our 1-px band error is 1.70x
// the reference's against 1.43x in flat blocks -- the textured leaves are where
// the accuracy deficit lives, so they are where a lower lambda can pay.
//
// The value is the fractional lambda REDUCTION applied at full activity ratio
// (0.25 = a leaf at 2x the picture's mean activity searches at 0.75x lambda);
// the reduction is clamped to [0, 0.75] and the ratio is capped at 4x.  0 is
// byte-identical to the historical lambda at every leaf.
inline double& g_tx_lambda_textured_mut() {
  static double value = [] {
    const char* v = std::getenv("AV2ENC_TX_LAMBDA_TEXTURED");
    if (v == nullptr || *v == '\0') return 0.0;
    char* end = nullptr;
    const double parsed = std::strtod(v, &end);
    if (end == v || !std::isfinite(parsed) || parsed <= 0.0) return 0.0;
    return parsed > 0.75 ? 0.75 : parsed;
  }();
  return value;
}
// The per-leaf multiplier the encoder publishes before each key leaf search.
// 1.0 is the historical value.
inline double& g_tx_lambda_leaf_scale_mut() {
  static double value = 1.0;
  return value;
}
// The picture's mean 1-px activity, published once per frame by the encoder
// (video_frame_decision.cpp `publish_hf_mask_reference`).  0 when unset.
inline double& hf_mask_reference_mut() {
  static double value = 0.0;
  return value;
}
inline void set_hf_mask_reference(double value) {
  hf_mask_reference_mut() = value > 0.0 ? value : 0.0;
}
inline double hf_mask_reference() { return hf_mask_reference_mut(); }

// The lambda the key leaf's own TX plan search minimises with: the key TX RD
// gate (AV2ENC_RD_KEY_TX, docs/rd-lane3.md) prices the plan at
// `key_tx_rd_lambda * tx_scale` with tx_scale = 0.25 by default.  That constant
// lives in video_frame_intra.cpp's KeyTxRdConfig; read the same env knob here so
// the layout comparison and the plan comparison use one objective.  With the TX
// gate off the leaf searches at 1.0, so the tree does too.
inline double key_tx_scale() {
  const char* on = std::getenv("AV2ENC_RD_KEY_TX");
  if (on == nullptr || *on == '\0' || std::strcmp(on, "0") == 0) return 1.0;
  const char* value = std::getenv("AV2ENC_RD_KEY_TX_SCALE");
  if (value != nullptr && *value != '\0') {
    const double parsed = std::strtod(value, nullptr);
    if (parsed > 0.0) return parsed;
  }
  return 0.25;
}
inline bool stats() {
  const char* value = std::getenv("AV2ENC_RD_SEQTREE_STATS");
  if (value == nullptr || *value == '\0') return true;
  return !parse_off(value);
}

// No 16x16 floor: 4x4 is legal for the intra keyframe (§5.20.3.1).
inline int min_block_px() { return 4; }

// PARTITION_SPLIT/NONE symbol, modelled.  Charged once per internal node.
inline double partition_bits() { return 2.0; }

// One leaf's measured luma distortion for the sequential tree.  `plane` is the
// live reconstruction (plane_stride, plane_h) and (rx, ry) is the leaf origin in
// it; the causal neighbours the EDGE term reads are already in that plane, so
// the same call is exact for a trial encode and for the post-pass verification.
// `rows` is the visible-row clip (-1 = all).
//
// The explicit-weight form below is the same measured D with the caller's own
// high-pass / block-edge weights.  AV2ENC_RD_TAIL_TX uses it so the tail
// residual plan comparison and the key driver do not share one env knob; the
// historical wrapper reads the driver's AV2ENC_RD_DRIVER_HF/EDGE knobs.
inline double leaf_distortion_gains(
    const std::uint8_t* src, int src_w, int src_h, int src_stride, int sx,
    int sy, int w, int h, int rows, const std::uint8_t* plane,
    int plane_stride, int plane_h, int rx, int ry, double lowpass_gain,
    double lowpass2_gain, double hf, double edge) {
  if (src == nullptr || plane == nullptr || w <= 0 || h <= 0) return 0.0;
  if (src_w <= 0 || src_h <= 0 || plane_stride <= 0) return 0.0;
  const int count = rows >= 0 && rows < h ? rows : h;
  if (count <= 0 || rx < 0 || ry < 0 || rx + w > plane_stride ||
      ry + count > plane_h) {
    return 0.0;
  }
  const std::uint8_t* rec =
      plane + static_cast<std::size_t>(ry) * static_cast<std::size_t>(plane_stride) +
      static_cast<std::size_t>(rx);
  double sum = eff_perc::weighted_sse(/*weights=*/nullptr, w, h, src, src_stride,
                                      sx, sy, rec, plane_stride, 0, 0, count,
                                      lowpass_gain, lowpass2_gain);
  if (hf <= 0.0 && edge <= 0.0) return sum;
  const auto source = [&](int x, int y) -> int {
    const int px = std::clamp(sx + x, 0, src_w - 1);
    const int py = std::clamp(sy + y, 0, src_h - 1);
    return static_cast<int>(
        src[static_cast<std::size_t>(py) * static_cast<std::size_t>(src_stride) +
            static_cast<std::size_t>(px)]);
  };
  const auto recon = [&](int x, int y) -> int {
    return static_cast<int>(
        rec[static_cast<std::size_t>(y) * static_cast<std::size_t>(plane_stride) +
            static_cast<std::size_t>(x)]);
  };
  if (hf > 0.0) {
    double band = 0.0;
    for (int y = 0; y < count; ++y) {
      for (int x = 0; x < w; ++x) {
        const double d = static_cast<double>(source(x, y) - recon(x, y));
        int low = 0;
        int used = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          const int py = y + dy;
          if (py < 0 || py >= count) continue;
          for (int dx = -1; dx <= 1; ++dx) {
            const int px = x + dx;
            if (px < 0 || px >= w) continue;
            low += source(px, py) - recon(px, py);
            ++used;
          }
        }
        const double mean = static_cast<double>(low) / static_cast<double>(used);
        band += (d - mean) * (d - mean);
      }
    }
    sum += hf * band;
  }
  if (edge > 0.0) {
    double step = 0.0;
    if (ry > 0) {
      for (int x = 0; x < w; ++x) {
        const double src_step =
            static_cast<double>(source(x, 0) - source(x, -1));
        const double rec_step = static_cast<double>(recon(x, 0) - recon(x, -1));
        const double e = src_step - rec_step;
        step += e * e;
      }
    }
    if (rx > 0) {
      for (int y = 0; y < count; ++y) {
        const double src_step =
            static_cast<double>(source(0, y) - source(-1, y));
        const double rec_step = static_cast<double>(recon(0, y) - recon(-1, y));
        const double e = src_step - rec_step;
        step += e * e;
      }
    }
    sum += edge * step;
  }
  return sum;
}

inline double leaf_distortion(const std::uint8_t* src, int src_w, int src_h,
                              int src_stride, int sx, int sy, int w, int h,
                              int rows, const std::uint8_t* plane,
                              int plane_stride, int plane_h, int rx, int ry,
                              double lowpass_gain, double lowpass2_gain) {
  return leaf_distortion_gains(src, src_w, src_h, src_stride, sx, sy, w, h,
                               rows, plane, plane_stride, plane_h, rx, ry,
                               lowpass_gain, lowpass2_gain, hf_gain(),
                               edge_gain());
}

struct Counters {
  unsigned long long superblocks = 0;
  unsigned long long nodes = 0;              // solve nodes visited
  unsigned long long splits_taken = 0;       // nodes whose measured split won
  unsigned long long layouts_measured = 0;   // candidate layouts encoded
  unsigned long long layouts_ref = 0;        // incumbent (pass-1) layouts
  unsigned long long variants_screened = 0;  // HORZ/VERT screen evaluations
  unsigned long long sbs_changed = 0;        // superblock layout != pass 1
  unsigned long long leaves = 0;
  unsigned long long plans[5] = {0, 0, 0, 0, 0};  // 4,8,16,32,64 leaves
  unsigned long long plans_rect[5] = {0, 0, 0, 0, 0};  // rectangular leaves
  unsigned long long leaves_verified = 0;
  unsigned long long leaves_mismatched = 0;
  unsigned long long rejected = 0;  // candidate layouts that could not be scored
  double key_d = 0.0;
  double key_r = 0.0;
  double key_j = 0.0;
  double d_ratio_mean = 0.0;
  double d_measured_total = 0.0;  // sum of accepted leaves' measured D
  double d_coded_total = 0.0;     // sum of the same leaves' coded D

  double worst = 0.0;
  double decision_ms = 0.0;
};

inline Counters& counters() {
  static Counters value;
  return value;
}

// Bucket a leaf by its short side: 4, 8, 16, 32, 64.
inline int plan_bucket(int width, int height) {
  const int side = width < height ? width : height;
  int index = 0;
  while (index < 4 && side > (4 << index)) ++index;
  return index;
}

inline void plan_leaf(int width, int height) {
  ++counters().leaves;
  ++counters().plans[plan_bucket(width, height)];
  if (width != height) ++counters().plans_rect[plan_bucket(width, height)];
}

inline void print_stats() {
  if (!enabled() || !stats()) return;
  const Counters& c = counters();
  std::fprintf(
      stderr,
      "rd-seqtree: gate=1 rounds=%d topk=%d screen=%d depth=%d scale=%.3f "
      "hf=%.2f edge=%.2f "
      "sb=%llu nodes=%llu splits=%llu layouts=%llu ref=%llu "
      "screen_evals=%llu changed=%llu "
      "leaves=%llu plans=%llu/%llu/%llu/%llu/%llu "
      "rect=%llu/%llu/%llu/%llu/%llu D=%.0f R=%.0f J=%.0f "
      "verified=%llu mismatched=%llu D_ratio=%.3f Dsum=%.3f worst=%.0f "
      "rejected=%llu ms=%.0f\n",
      rounds(), top_k(), screen_modes(), max_depth(), lambda_scale(), hf_gain(),
      edge_gain(), c.superblocks, c.nodes, c.splits_taken, c.layouts_measured, c.layouts_ref,
      c.variants_screened, c.sbs_changed, c.leaves, c.plans[0], c.plans[1],
      c.plans[2], c.plans[3], c.plans[4], c.plans_rect[0], c.plans_rect[1],
      c.plans_rect[2], c.plans_rect[3], c.plans_rect[4], c.key_d, c.key_r,
      c.key_j, c.leaves_verified, c.leaves_mismatched, c.d_ratio_mean,
      c.d_measured_total > 0.0 ? c.d_coded_total / c.d_measured_total : 0.0,
      c.worst, c.rejected, c.decision_ms);
}

inline void init() {
  static const bool once = [] {
    std::atexit(&print_stats);
    return true;
  }();
  (void)once;
}

}  // namespace av2enc::video::seq_rd
