// R09 rate model: exact inter syntax + analytic Laplacian coefficients.
//
// Gate (env only, no CLI flag):
//
//   AV2ENC_RATE_HYBRID_S=1          use the model in the leaf decisions
//   AV2ENC_RATE_HYBRID_S_MEASURE=1  keep the historical decisions (stream
//                                   byte-identical) but record the model's
//                                   prediction error.  The master gate records
//                                   the same report for its own decisions.
//
// The historical leaf decision charges two proxies:
//
//   * `inter_syntax_bits()`: fixed constants for YMode / DRL / MVD shell,
//     nothing for the reference index, and a fixed 2.0-bit skip baseline;
//   * `estimate_tx_bits()`: 8 EOB bits + 2.5 per nonzero + 0.5 per EOB.
//
// This model replaces the first with exact -log2(p) from the frame's live
// adaptive CDFs (a probe that mirrors the writer's inter symbol sequence and
// advances in coded order), and the second with the analytic two-sided
// Laplacian rate of the block's quantised levels:
//
//   b = mean(|level|) over the nonzeros   (ML scale of a two-sided geometric)
//   p = exp(-1/b)
//   H = nonzero * log2(e * (1+p)/(1-p))
//   bits = 8 (+4 inter luma) + 0.5*eob + H
//
// H is the discrete-Laplacian entropy of the nonzero magnitudes.  The existing
// EOB/TX-type overhead (8 bits, +4 for inter luma) and the 0.5-bit per coded
// position term are kept: only the coefficient-magnitude constant (2.5 per
// nonzero) is replaced, so the branch isolates the syntax and coefficient
// models from the EOB/position syntax.
//
// The probe prices only what it can reproduce exactly.  It covers the nightly
// surface: single prediction, no TIP, eighth-pel raw-shell MVD, BAWP.  A leaf
// that uses compound / TIP / AMVD / flex-MV resolution is priced with the
// historical constants and counted as a fallback; the CDF probe then stops
// adapting (its state no longer matches the writer) and later leaves of that
// tile also fall back.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

#include "codec/inter/merge/merge.h"
#include "pipeline/video/video_syntax.h"

namespace av2enc::video::rate_hybrid {

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
    return !parse_off(std::getenv("AV2ENC_RATE_HYBRID_S"));
  }();
  return value;
}

// Measurement runs the historical decisions but records both models' error.
// `AV2ENC_RATE_HYBRID_S=1` also records (the two arms then share the report).
inline bool measure_enabled() {
  static const bool value = [] {
    return !parse_off(std::getenv("AV2ENC_RATE_HYBRID_S_MEASURE"));
  }();
  return value;
}

inline bool reporting() { return enabled() || measure_enabled(); }

// Ablation switches for the 2x2 (syntax x coefficient) decomposition.  Both
// default ON; only an explicit 0/false/off disables one half while the master
// gate stays on.  Measurement mode always reports both models.
inline bool syntax_enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_RATE_HYBRID_S_SYNTAX");
    return env == nullptr || *env == '\0' || !parse_off(env);
  }();
  return value;
}

inline bool coeff_enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_RATE_HYBRID_S_COEFF");
    return env == nullptr || *env == '\0' || !parse_off(env);
  }();
  return value;
}

// ---------------------------------------------------------------------------
// Analytic Laplacian coefficient rate
// ---------------------------------------------------------------------------

// `levels` is the coded scan prefix (exactly `count` entries, zeros included).
// `count` is the EOB, so the prefix ends at the last nonzero and at least one
// level is nonzero.  The nonzero magnitudes are modelled as a two-sided
// geometric (discrete Laplacian) with the ML scale b = mean(|level|); the
// caller adds the existing EOB/TX-type overhead and 0.5 bits per coded
// position.  b comes from the block itself -- no tuning constant.
inline double laplacian_bits(std::span<const std::int32_t> levels, int count) {
  if (count <= 0) return 0.0;
  const int n = std::min<int>(count, static_cast<int>(levels.size()));
  int nonzero = 0;
  std::int64_t sum_abs = 0;
  for (int i = 0; i < n; ++i) {
    const std::int64_t v = levels[static_cast<std::size_t>(i)];
    if (v != 0) {
      ++nonzero;
      sum_abs += v < 0 ? -v : v;
    }
  }
  if (nonzero == 0) return 0.0;
  // Coefficient magnitudes: ML scale b = mean(|level|) over the nonzeros, the
  // two-sided geometric entropy per coded nonzero.
  const double b = static_cast<double>(sum_abs) / nonzero;
  const double p = std::exp(-1.0 / b);
  const double mag =
      static_cast<double>(nonzero) *
      std::log2(std::exp(1.0) * (1.0 + p) / (1.0 - p));
  // Position/EOB term keeps the historical 0.5 bits per coded coefficient:
  // the Laplacian prices the magnitudes, not the significance map.
  return mag + 0.5 * static_cast<double>(count);
}

// ---------------------------------------------------------------------------
// Exact inter syntax probe
// ---------------------------------------------------------------------------

// Script parameters the probe needs.  Everything here is fixed per tile/frame.
struct ProbeConfig {
  int base_q_idx = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  int num_refs = 1;
  int max_drl_bits_minus_1 = 4;
  int frame_distance[7] = {0, 0, 0, 0, 0, 0, 0};
  bool reference_select = false;  // §5.20.7.10 comp_mode is coded
  bool allow_bawp = false;
  bool enable_interintra = false;
  bool enable_adaptive_mvd = false;
  bool enable_flex_mvres = false;
  bool enable_mvd_sign_derive = false;
  // §5.20.7.6 compound block layer.  `compound_exact` turns the probe's
  // compound pricing on (AV2ENC_COMPOUND_EXACT_RATE): comp_mode, is_joint,
  // compound_mode_non_joint, jmvd_scale_mode, the DRL indices, the MVD(s),
  // comp_group_idx, cwp_idx and the compound interp_filter symbol.
  bool compound_exact = false;
  bool enable_masked_compound = false;
  bool enable_cwp = false;
  int num_same_ref_compound = 0;
  int mv_precision = 0;  // 0 = eighth pel (raw shell), else coarser
  bool disable_cdf_update = false;
};

// One leaf's exact inter-syntax price.  `total()` is what replaces the
// historical `inter_syntax_bits()` (+ the mode-info extras the caller keeps).
struct Terms {
  double is_inter = 0.0;
  double skip = 0.0;
  double comp_mode = 0.0;
  double ref = 0.0;
  double mode = 0.0;
  double drl = 0.0;
  double mvd = 0.0;
  double bawp = 0.0;

  double total() const {
    return is_inter + skip + comp_mode + ref + mode + drl + mvd + bawp;
  }
  double mode_info() const { return comp_mode + ref + mode + drl + mvd + bawp; }
};

// Process-global prediction-error report (AV2ENC_RATE_HYBRID_S[_MEASURE]).
// The writer records the actual range-coder bits per leaf and the write-time
// -log2(p) decomposition; the decision stores its own prediction on the
// BlockSyntax.  `print()` runs from atexit and never touches the bitstream.
namespace report {

enum Tag : int {
  kTagIsInter = 0,
  kTagSkip,
  kTagCompMode,
  kTagSingleRef,
  kTagSingleMode,
  kTagDrl,
  kTagMvd,
  kTagBawp,
  kTagResid,
  kTagCount,
};

inline const char* tag_name(int tag) {
  switch (tag) {
    case kTagIsInter: return "is_inter";
    case kTagSkip: return "skip";
    case kTagCompMode: return "comp_mode";
    case kTagSingleRef: return "ref";
    case kTagSingleMode: return "mode";
    case kTagDrl: return "drl";
    case kTagMvd: return "mvd";
    case kTagBawp: return "bawp";
    default: return "resid";
  }
}

struct Sample {
  std::uint8_t is_inter = 0;
  std::uint8_t inter_mode = 0;
  std::uint8_t bsize = 0;
  std::uint8_t skip = 0;
  float pred_syntax_old = 0.0f;  // 2.0 + inter_syntax_bits()
  float pred_syntax_new = 0.0f;  // probe is_inter + skip + mode info
  float pred_coeff_old = 0.0f;   // luma_plan_bits + chroma estimate_tx_bits
  float pred_coeff_new = 0.0f;   // same structure, Laplacian magnitude term
  float pred_term[6] = {0, 0, 0, 0, 0, 0};  // is_inter, skip, mode, ref, drl,
                                            // mvd
  float pred_bawp = 0.0f;
  float eob = 0.0f;
  float nonzero = 0.0f;
  float sum_abs = 0.0f;
  float act_is_inter_skip = 0.0f;  // range-coder bits
  float act_mode_info = 0.0f;
  float act_residual = 0.0f;
  float exact[kTagCount] = {0};  // write-time -log2(p) per syntax group
};

void add(const Sample& sample);
void count_fallback();
void count_decision(const char* name);
void print();
// Constructs the collector before registering print(), so the atexit handler
// runs before the collector's static destructor (reverse registration order).
void init();
inline bool active() {
  if (!reporting()) return false;
  init();
  return true;
}

}  // namespace report

// ---------------------------------------------------------------------------
// SyntaxProbe
// ---------------------------------------------------------------------------

class SyntaxProbe {
 public:
  void reset(const ProbeConfig& config,
             std::span<const std::uint16_t> cdf_image);

  bool usable() const { return !degraded_; }
  void disable() { degraded_ = true; }

  // Mode-info terms only (comp_mode / ref / mode / drl / mvd).  Used by the
  // candidate ranking, where the skip decision has not been made yet.
  bool price_mode(const syntax::BlockSyntax& node, Terms* out);

  // Full price for a final leaf: is_inter + skip + mode info (+ BAWP).
  bool price_full(const syntax::BlockSyntax& node, Terms* out);

  // is_inter and skip symbols alone (context is independent of the candidate).
  bool price_leaf_flags(const syntax::BlockSyntax& node, double* is_inter,
                        double* skip);

  // Advance the CDF rows and the neighbour grids with the emitted leaf.
  void commit(const syntax::BlockSyntax& node);

  // `skip=1` minus `skip=0` cost: substituting the historical 2.0-bit skip
  // baseline, the residual decision compares
  //   D_res + lambda*C  <  D_pred + lambda*(skip1 - skip0).
  bool skip_delta(const syntax::BlockSyntax& node, double* delta) const;

  int mi_cols() const { return config_.mi_cols; }
  int mi_rows() const { return config_.mi_rows; }

 private:
  double cost(const std::uint16_t* cdf, int n, int symbol) const;
  void adapt(std::uint16_t* cdf, int n, int symbol);
  int neighbor_positions(int pos[2][2], int r, int c, int w4, int h4) const;
  int is_inter_context(int r, int c, int w4, int h4) const;
  int skip_context(int r, int c, int w4, int h4) const;
  int count_refs(int r, int c, int w4, int h4, int ref) const;
  bool ref_backward(int ref) const;
  int comp_mode_context(int r, int c, int w4, int h4) const;
  int new_mv_context(int r, int c, int w4, int h4) const;
  std::size_t cell_index(int r, int c) const {
    return static_cast<std::size_t>(r) *
               static_cast<std::size_t>(config_.mi_cols) +
           static_cast<std::size_t>(c);
  }
  bool walk_flags(const syntax::BlockSyntax& node, Terms* out, bool commit);
  bool walk_mode_info(const syntax::BlockSyntax& node, Terms* out,
                      bool include_bawp, bool commit);
  bool walk_compound_mode_info(const syntax::BlockSyntax& node, Terms* out,
                               bool commit);
  bool walk_shell(int dy, int dx, Terms* out, bool commit);
  int is_joint_context(int ref0, int ref1) const;
  int comp_group_context(int r, int c, int w4, int h4, int ref0,
                         int ref1) const;
  int interp_filter_context(int r, int c, int w4, int h4, int ref0,
                            bool is_compound) const;
  void update_grids(const syntax::BlockSyntax& node);

  ProbeConfig config_{};
  syntax::InterRateCdfRows cdf_{};
  std::vector<inter::mc::MiInfo> mi_;
  std::vector<std::uint8_t> skip_grid_;
  std::vector<std::uint8_t> ymode_grid_;
  std::vector<std::uint8_t> comp_group_grid_;
  std::vector<std::uint8_t> filter_grid_;
  bool degraded_ = false;
};

}  // namespace av2enc::video::rate_hybrid
