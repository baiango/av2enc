// Inter syntax rate model for the M2 P-frame decision layer.
//
// The default backend charges the same §9.3 default CDFs the M2 tile writer
// uses.  M2 sets `disable_cdf_update = 1` and
// `disable_cross_frame_cdf_init = 1`, so those CDFs are static for a frame;
// only the *context selection* (neighbour-dependent `is_inter`, `skip`, and
// `new_mv` contexts) is approximated here by neutral contexts.  When the
// integrator has the exact contexts, it supplies a `mode_bits` callback and
// this default is bypassed.
//
// What is charged:
//   * is_inter                         §5.20.7.3
//   * skip_flag                        §5.20.5.10
//   * single_mode                      §5.20.7.6
//   * drl_idx (NEARMV/NEWMV)           §5.20.7.8
//   * shell-coded MVD (NEWMV)          §5.20.7.20
//   * reference syntax                 (0 bits with NumTotalRefs = 1)
//
// The MVD shell model is transcribed from the writer's `write_mv()` and is
// exact for a given MVD because the shell CDFs do not depend on neighbours and
// are never adapted.  `single_mode`, `skip_flag` and `is_inter` use neutral
// contexts by default; they are 1-2 bits each and are fully replaceable.
//
// Rate units are bits.  Residual coefficient bits are *not* reimplemented
// here: `residual::decide_inter_block()` already measures them with the
// writer's own symbol sequence, and its result is passed to
// `residual_bits()`.  This keeps one authority for coefficient rate.

#pragma once

#include "codec/inter/mc/mc.h"
#include "codec/transform/inter_residual/inter_residual.h"

#include <cstdint>

namespace av2enc::video::rdo {

// Contexts the default backend needs.  The integration may derive them from
// the neighbour grids; 0 is a neutral, deterministic default.
struct RateContext {
  int is_inter_ctx = 0;  // TileIsInterCdf[ctx], ctx 0..3
  int skip_ctx = 0;      // TileSkipCdf[ctx], ctx 0..5
  int new_mv_ctx = 0;    // TileSingleModeCdf / TileDrlModeCdf, ctx 0..4
  int max_drl_bits_minus_1 = 4;  // M2 sequence header
};

struct ModeRateInput {
  int inter_mode = 1;  // 0 GLOBALMV, 1 NEARMV, 2 NEWMV
  int drl_index = 0;
  inter::mc::Mv mvd{};  // 1/8-pel, signalled for NEWMV only
  int ref = 0;
  bool skip_flag = true;
  bool is_inter = true;
  RateContext context;
};

// Optional callbacks.  A null callback selects the default backend.
//   mode_bits:  whole is_inter + skip + mode + DRL + MVD + ref group.
//   residual_bits: coefficient syntax (including skip_flag) for a decision.
//   intra_syntax_bits: intra fallback syntax group.
//   ref_bits: optional reference syntax term added by the default mode model
//             (0 for the M2 single-reference profile).
struct RateCallbacks {
  void* context = nullptr;
  double (*mode_bits)(void* context, const ModeRateInput& input) = nullptr;
  double (*residual_bits)(void* context,
                          const residual::BlockDecision& decision) = nullptr;
  double (*intra_syntax_bits)(void* context, int y_mode, int y_angle_delta,
                              int uv_mode, int is_inter_ctx) = nullptr;
  double (*ref_bits)(void* context, int ref) = nullptr;
};

class InterRateModel {
 public:
  InterRateModel() = default;
  explicit InterRateModel(RateCallbacks callbacks) : callbacks_(callbacks) {}

  const RateCallbacks& callbacks() const noexcept { return callbacks_; }
  RateCallbacks& callbacks() noexcept { return callbacks_; }

  double mode_bits(const ModeRateInput& input) const;
  double residual_bits(const residual::BlockDecision& decision) const;
  double intra_syntax_bits(int y_mode, int y_angle_delta, int uv_mode,
                           int is_inter_ctx) const;

  // Default backends, exposed for tests and for a future GPU transcription.
  static double default_mode_bits(void* context, const ModeRateInput& input);
  static double default_residual_bits(
      void* context, const residual::BlockDecision& decision);
  static double default_intra_syntax_bits(void* context, int y_mode,
                                          int y_angle_delta, int uv_mode,
                                          int is_inter_ctx);
  static double default_mv_bits(int mv_y, int mv_x);
  static double default_single_mode_bits(int ctx, int single_mode);
  static double default_drl_bits(int ctx, int drl_index, int max_drl_bits);
  static double default_skip_flag_bits(int ctx, bool skip_flag);
  static double default_is_inter_bits(int ctx, bool is_inter);
  // -log2 P(symbol) for a §9.3 CDF row.  `n` is the alphabet size; the array
  // has n + 1 entries (n-1 boundaries, an adaptation-rate index, a count).
  static double cdf_bits(const std::uint16_t* cdf, int n, int symbol);

 private:
  RateCallbacks callbacks_;
};

}  // namespace av2enc::video::rdo
