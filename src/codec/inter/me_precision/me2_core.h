// AV2 inter motion estimation, extended — shared host/device core.
//
// This is an *additive* layer over src/encoder/inter/me/me_core.h.  It does
// not modify that component; it instantiates the same search state machine and
// adds:
//
//   1. eighth- and sixteenth-pel search stages enabled by default, on top of
//      the existing integer / half / quarter path; all fractional costs use
//      the normative Subpel_Filters two-pass interpolation in me_core.h;
//   2. a deterministic exhaustive refinement pass around the best
//      integer/subpel candidate (the existing 8-neighbour hill climb can stop
//      one step early);
//   3. adaptive per-block MV precision (enable_flex_mvres): the 1/16 search
//      result is lowered to the frame precision or a coarser reachable
//      precision, and the *signaled* MV is re-costed at that precision;
//   4. a deterministic rate proxy (MVD shell-code decisions + precision side
//      information) so precision selection can trade distortion against MV
//      rate without owning an entropy coder.
//
// The AV2 bitstream cannot signal finer than 1/8 luma sample
// (MV_PRECISION_EIGHTH_PEL).  The 1/16 stage is an encoder-side search
// granularity only; every reported `mv_x`/`mv_y` is quantized and re-costed.
//
// CPU reference (me2.cpp) and CUDA kernel (me2_gpu.cu) instantiate the same
// inline functions with different cost functors, so MVs, costs, precision
// choices and evaluation counts are bit-exact.

#ifndef AV2ENC_ENCODER_INTER_ME2_ME2_CORE_H_
#define AV2ENC_ENCODER_INTER_ME2_ME2_CORE_H_

#include <cstdint>

#include "codec/inter/me/me_core.h"
#include "codec/inter/me_precision/me2_precision.h"

namespace av2enc::me2 {

// One ME2 job uses the ME job data model unchanged (luma block, explicit
// predictors, up to kMeMaxRefs references).
using MeCoreFrame = me::MeCoreFrame;
using MeCorePlane = me::MeCorePlane;
using MeCoreRef = me::MeCoreRef;
using MeCoreJob = me::MeCoreJob;
using MeCoreMvField = me::MeCoreMvField;

AV2ENC_ME_HD me::MeCoreOptions me2_default_search_options() {
  me::MeCoreOptions o;
  // Half + quarter are the M2 baseline; eighth + sixteenth are the extension.
  o.subpel_stages = me::kMeSubpelHalf | me::kMeSubpelQuarter |
                    me::kMeSubpelEighth | me::kMeSubpelSixteenth;
  o.subpel_iterations = 3;
  return o;
}

struct Me2Options {
  me::MeCoreOptions search = me2_default_search_options();

  // Frame-level precision from the frame header (§5.18.2).  ONE_PEL implies
  // force_integer_mv and therefore disables per-block precision.
  int frame_precision = kMvPrecisionEighthPel;

  // enable_flex_mvres: allow NEWMV blocks to signal a coarser precision.
  int enable_flex_mvres = 0;

  // Optional restriction on per-block precisions (bit i = MvPrecision i).
  // 0 means every reachable precision.  The frame precision is always kept.
  int precision_mask = 0;

  // Exhaustive refinement after the ME subpel stages.  Radius and step are in
  // 1/16-sample units.  radius 0 disables the pass.  step 0 is treated as 1.
  int refine_radius = 2;
  int refine_step = 1;

  // Precision selection cost = distortion + ((lambda * rate_bins) >> shift).
  // lambda 0 keeps the historical pure-distortion contract.  A positive lambda
  // enables the adaptive-precision rate/distortion trade-off.
  int mv_rate_lambda = 0;
  int mv_rate_shift = 0;

  // Include the precision side-information bins in the rate proxy.
  int count_precision_bins = 1;
};

struct Me2Result {
  // Signalled MV, 1/16 luma samples.  Always a multiple of the selected
  // precision's quantum (eighth = 2, quarter = 4, half = 8, ...).
  std::int32_t mv_x = 0;
  std::int32_t mv_y = 0;

  // Best refined search MV before precision lowering, 1/16 luma samples.
  std::int32_t search_mv_x = 0;
  std::int32_t search_mv_y = 0;

  int ref_index = -1;
  std::int32_t cost = 0;         // distortion at the signalled MV
  std::int32_t search_cost = 0;  // distortion at search_mv
  std::int32_t evals = 0;        // cost evaluations (CPU/GPU parity contract)
  std::int32_t mv_rate_bins = 0; // MVD shell bins at the signalled precision
  std::int32_t precision_bins = 0;
  std::int32_t total_rate_bins = 0;

  int precision = kMvPrecisionEighthPel;
  int frame_precision = kMvPrecisionEighthPel;
  int use_most_probable_precision = 1;
  int pb_mv_precision = -1;
  // Local part of the TilePbMvPrecisionCdf context: 1 when this block
  // uses a coarser precision than FrameMvPrecision.  M2 must OR this with
  // the neighbour state to form the actual ctx.
  int pb_precision_coarser = 0;
  int stage = me::kMeStageNone;

  int visible_w = 0;
  int visible_h = 0;
};

// --- helpers ---------------------------------------------------------------
AV2ENC_ME_HD std::int32_t me2_floor_div(std::int32_t a, std::int32_t b) {
  const std::int32_t q = a / b;
  return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

AV2ENC_ME_HD std::int32_t me2_ceil_div(std::int32_t a, std::int32_t b) {
  return -me2_floor_div(-a, b);
}

// Quantize + clamp a search MV to a signalled precision and the reference
// plane.  Returns false when the block is smaller than the precision grid and
// no representable MV fits inside the plane.
AV2ENC_ME_HD bool me2_clamp_signaled_mv(std::int32_t* mv_y, std::int32_t* mv_x,
                                        int x, int y, int w, int h,
                                        int ref_w, int ref_h, int precision) {
  if (mv_y == nullptr || mv_x == nullptr ||
      !me2_precision_valid(precision)) {
    return false;
  }
  const std::int32_t q = me2_precision_quantum_1_16(precision);
  // Fractional positions may reach up to one sample beyond the integer edge;
  // integer and coarser positions use integer bounds.
  const std::int32_t over =
      precision >= kMvPrecisionOnePel ? me::kSubpelMask : 0;
  const std::int32_t lo_x =
      -static_cast<std::int32_t>(x) * me::kSubpelOne - over;
  const std::int32_t hi_x =
      static_cast<std::int32_t>(ref_w - x - w) * me::kSubpelOne + over;
  const std::int32_t lo_y =
      -static_cast<std::int32_t>(y) * me::kSubpelOne - over;
  const std::int32_t hi_y =
      static_cast<std::int32_t>(ref_h - y - h) * me::kSubpelOne + over;

  const std::int32_t lo_qx = me2_ceil_div(lo_x, q) * q;
  const std::int32_t hi_qx = me2_floor_div(hi_x, q) * q;
  const std::int32_t lo_qy = me2_ceil_div(lo_y, q) * q;
  const std::int32_t hi_qy = me2_floor_div(hi_y, q) * q;
  if (lo_qx > hi_qx || lo_qy > hi_qy) {
    return false;
  }
  *mv_x = me::me_iclamp(lo_qx, *mv_x, hi_qx);
  *mv_y = me::me_iclamp(lo_qy, *mv_y, hi_qy);
  return true;
}

// Deterministic total order for precision candidates and references.
// Primary: selection cost (distortion + rate term).
// Then: fewer rate bins, finer precision, smaller |MV|, then y then x.
AV2ENC_ME_HD bool me2_choice_better(std::int64_t sel_a, std::int32_t rate_a,
                                    int precision_a, std::int32_t mv_y_a,
                                    std::int32_t mv_x_a, std::int64_t sel_b,
                                    std::int32_t rate_b, int precision_b,
                                    std::int32_t mv_y_b, std::int32_t mv_x_b) {
  if (sel_a != sel_b) {
    return sel_a < sel_b;
  }
  if (rate_a != rate_b) {
    return rate_a < rate_b;
  }
  if (precision_a != precision_b) {
    return precision_a > precision_b;  // finer wins a full tie
  }
  const std::int32_t ma = (mv_y_a < 0 ? -mv_y_a : mv_y_a) +
                          (mv_x_a < 0 ? -mv_x_a : mv_x_a);
  const std::int32_t mb = (mv_y_b < 0 ? -mv_y_b : mv_y_b) +
                          (mv_x_b < 0 ? -mv_x_b : mv_x_b);
  if (ma != mb) {
    return ma < mb;
  }
  if (mv_y_a != mv_y_b) {
    return mv_y_a < mv_y_b;
  }
  if (mv_x_a != mv_x_b) {
    return mv_x_a < mv_x_b;
  }
  return false;
}

// Search stage label for a refinement step in 1/16 units.
AV2ENC_ME_HD int me2_stage_for_step(int step) {
  switch (step) {
    case 1:
      return me::kMeStageSixteenth;
    case 2:
      return me::kMeStageEighth;
    case 4:
      return me::kMeStageQuarter;
    case 8:
      return me::kMeStageHalf;
    default:
      return me::kMeStageInteger;
  }
}

// Exhaustive refinement of the best subpel candidate.  Offsets are relative
// to `base_mv` (the ME result before this pass), evaluated in a fixed
// dy-then-dx order with the same cost functor.  Updates *best on strict
// improvement only, so the CPU and GPU instantiations agree.
template <class CostFn>
AV2ENC_ME_HD std::int32_t me2_refine_exhaustive(
    const MeCorePlane& cur, const MeCorePlane& ref, int x, int y, int w, int h,
    int metric, int filter, const Me2Options& opt, std::int32_t base_mv_x,
    std::int32_t base_mv_y, std::int32_t* best_cost, std::int32_t* best_mv_x,
    std::int32_t* best_mv_y, int* best_stage, const CostFn& cost) {
  std::int32_t evals = 0;
  int radius = opt.refine_radius;
  int step = opt.refine_step > 0 ? opt.refine_step : 1;
  if (radius <= 0) {
    return 0;
  }
  if (radius > 64) {
    radius = 64;
  }
  const int ref_w = ref.width;
  const int ref_h = ref.height;
  for (int dy = -radius; dy <= radius; dy += step) {
    for (int dx = -radius; dx <= radius; dx += step) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      std::int32_t mx =
          me::me_clamp_mv_x(base_mv_x + dx, x, w, ref_w, true);
      std::int32_t my =
          me::me_clamp_mv_y(base_mv_y + dy, y, h, ref_h, true);
      if (mx == *best_mv_x && my == *best_mv_y) {
        continue;
      }
      const std::int32_t c =
          cost(cur, ref, x, y, w, h, mx, my, metric, filter);
      ++evals;
      if (c < *best_cost) {
        *best_cost = c;
        *best_mv_x = mx;
        *best_mv_y = my;
        *best_stage = me2_stage_for_step(step);
      }
    }
  }
  return evals;
}

// The predictor against which the MVD rate proxy is measured.  The full DRL
// stack is owned by the MC component; ME2 uses the first explicit predictor
// (or zero) as a stable proxy.  Coarser-than-half precisions lower the
// predictor too, mirroring assign_mv() (§5.20.7.13).
AV2ENC_ME_HD void me2_predictor_for_precision(const MeCoreJob& job,
                                              int precision,
                                              std::int32_t* pred_y_eighth,
                                              std::int32_t* pred_x_eighth) {
  std::int32_t py = 0;
  std::int32_t px = 0;
  if (job.num_pred > 0) {
    py = me2_1_16_to_eighth(job.pred_mv[0][0]);
    px = me2_1_16_to_eighth(job.pred_mv[0][1]);
  }
  if (precision < kMvPrecisionHalfPel) {
    me2_lower_mv_precision(&py, &px, precision);
  }
  *pred_y_eighth = py;
  *pred_x_eighth = px;
}

// Search all references for one job.  `cost` is the scalar or warp
// cooperative cost functor.  Reference order is job order; a strict tie keeps
// the earlier reference.  Per reference: ME integer + subpel, exhaustive
// refinement, then precision selection with the deterministic order above.
template <class CostFn>
AV2ENC_ME_HD Me2Result me2_search_core(const MeCoreFrame& cur,
                                       const MeCoreRef* refs,
                                       const MeCoreJob& job,
                                       const Me2Options& opt,
                                       const MeCoreMvField* field,
                                       const CostFn& cost) {
  Me2Result res;
  res.frame_precision = opt.frame_precision;
  res.precision = opt.frame_precision;
  const int w = me::me_imin(job.w, cur.width - job.x);
  const int h = me::me_imin(job.h, cur.height - job.y);
  res.visible_w = w;
  res.visible_h = h;
  if (w <= 0 || h <= 0 || job.num_refs <= 0 ||
      !me2_precision_valid(opt.frame_precision)) {
    return res;
  }

  int candidates[kMvPrecisionCount];
  const int num_candidates = me2_precision_candidates(
      opt.frame_precision, opt.enable_flex_mvres, opt.precision_mask,
      candidates, kMvPrecisionCount);

  // The 1/16 stage is implemented by the exhaustive refinement pass below.
  // The ME core's 8-neighbour sixteenth ring is intentionally not used: it
  // can move to a 1/16 position whose 1/8 quantization is worse than the
  // eighth-grid result, which would make the extended path regress against
  // the eighth-only path.  Masking it off keeps the pre-refinement candidate
  // on the signalable grid; the refinement candidate is then scored after
  // quantization, so it can only improve the signalled result.
  me::MeCoreOptions search_opt = opt.search;
  search_opt.subpel_stages &= ~me::kMeSubpelSixteenth;
  Me2Options refine_opt = opt;
  if ((opt.search.subpel_stages & me::kMeSubpelSixteenth) != 0 &&
      refine_opt.refine_radius == 0) {
    refine_opt.refine_radius = 1;  // 3x3 one-step 1/16 pass
  }

  bool have_best = false;
  std::int64_t best_sel = 0;
  Me2Result best;
  best.frame_precision = opt.frame_precision;
  std::int32_t evals_total = 0;

  for (int ri = 0; ri < job.num_refs; ++ri) {
    const int ref_index = job.ref_indices[ri];
    if (ref_index < 0) {
      continue;
    }
    const MeCorePlane& ref = refs[ref_index].plane[0];
    if (ref.data == nullptr || ref.width <= 0 || ref.height <= 0 ||
        ref.stride < ref.width) {
      continue;
    }

    // Reuse the existing ME state machine per reference so refinement and
    // precision lowering can be applied to that reference's own best MV.
    MeCoreJob single = job;
    single.num_refs = 1;
    single.ref_indices[0] = static_cast<std::int8_t>(ref_index);
    const me::MeCoreResult sr =
        me::me_search_core(cur, refs, single, search_opt, field, cost);
    evals_total += sr.evals;
    if (sr.ref_index < 0) {
      continue;
    }

    std::int32_t search_cost = sr.cost;
    std::int32_t search_x = sr.mv_x;
    std::int32_t search_y = sr.mv_y;
    int stage = sr.stage;
    evals_total += me2_refine_exhaustive(
        cur.plane[0], ref, job.x, job.y, w, h, opt.search.metric,
        opt.search.subpel_interp, refine_opt, sr.mv_x, sr.mv_y, &search_cost,
        &search_x, &search_y, &stage, cost);

    // Precision candidates: quantize the pre-refinement and refined MVs,
    // clamp to the plane, re-cost at the signalled position, and score
    // distortion + rate proxy.  Keeping the pre-refinement candidate ensures
    // the 1/16 refinement can never make the *signalled* result worse.
    bool have_choice = false;
    std::int64_t choice_sel = 0;
    Me2Result choice;
    choice.frame_precision = opt.frame_precision;
    choice.search_mv_x = search_x;
    choice.search_mv_y = search_y;
    choice.search_cost = search_cost;
    choice.ref_index = ref_index;
    choice.stage = stage;

    const std::int32_t base_mv[2][2] = {
        {sr.mv_x, sr.mv_y}, {search_x, search_y}};
    const int num_bases =
        (sr.mv_x == search_x && sr.mv_y == search_y) ? 1 : 2;
    for (int bi = 0; bi < num_bases; ++bi) {
      for (int ci = 0; ci < num_candidates; ++ci) {
        const int p = candidates[ci];
        std::int32_t qy =
            me2_quantize_mv_component_1_16(base_mv[bi][1], p);
        std::int32_t qx =
            me2_quantize_mv_component_1_16(base_mv[bi][0], p);
        if (!me2_clamp_signaled_mv(&qy, &qx, job.x, job.y, w, h, ref.width,
                                   ref.height, p)) {
          continue;
        }
        // me2_clamp_signaled_mv keeps the quantum because the bounds are
        // aligned inward; assert the invariant defensively.
        if (!me2_mv_is_at_precision(qy, p) || !me2_mv_is_at_precision(qx, p)) {
          continue;
        }
        const std::int32_t c = cost(cur.plane[0], ref, job.x, job.y, w, h, qx,
                                    qy, opt.search.metric,
                                    opt.search.subpel_interp);
        ++evals_total;

        std::int32_t py_e = 0;
        std::int32_t px_e = 0;
        me2_predictor_for_precision(job, p, &py_e, &px_e);
        const std::int32_t dy_e = me2_1_16_to_eighth(qy) - py_e;
        const std::int32_t dx_e = me2_1_16_to_eighth(qx) - px_e;
        std::int32_t mv_bins = me2_mvd_shell_bins(dy_e, dx_e, p);
        if (mv_bins < 0) {
          mv_bins = 0;
        }
        std::int32_t side_bins = 0;
        if (opt.count_precision_bins != 0) {
          side_bins = me2_precision_side_bins(
              opt.frame_precision, p, opt.enable_flex_mvres, true);
        }
        const std::int32_t rate = mv_bins + side_bins;
        const std::int64_t rate_term =
            (static_cast<std::int64_t>(opt.mv_rate_lambda) * rate) >>
            (opt.mv_rate_shift < 0 ? 0 : opt.mv_rate_shift);
        const std::int64_t sel = static_cast<std::int64_t>(c) + rate_term;

        int um = 1;
        int pb = -1;
        if (!me2_block_precision_to_syntax(opt.frame_precision, p, &um,
                                            &pb)) {
          continue;  // candidate set/precision map disagree; skip
        }

        if (!have_choice ||
            me2_choice_better(sel, rate, p, qy, qx, choice_sel,
                              choice.total_rate_bins, choice.precision,
                              choice.mv_y, choice.mv_x)) {
          have_choice = true;
          choice_sel = sel;
          choice.mv_x = qx;
          choice.mv_y = qy;
          choice.cost = c;
          choice.mv_rate_bins = mv_bins;
          choice.precision_bins = side_bins;
          choice.total_rate_bins = rate;
          choice.precision = p;
          choice.use_most_probable_precision = um;
          choice.pb_mv_precision = pb;
          choice.pb_precision_coarser = p < opt.frame_precision ? 1 : 0;
        }
      }
    }
    if (!have_choice) {
      continue;
    }

    // Reference comparison: selection cost first, then the same deterministic
    // order.  A strict tie keeps the earlier reference.
    bool replace = !have_best;
    if (have_best) {
      replace = me2_choice_better(
          choice_sel, choice.total_rate_bins, choice.precision, choice.mv_y,
          choice.mv_x, best_sel, best.total_rate_bins, best.precision,
          best.mv_y, best.mv_x);
    }
    if (replace) {
      best = choice;
      best_sel = choice_sel;
      have_best = true;
    }
  }

  if (!have_best) {
    return res;
  }
  best.evals = evals_total;
  best.frame_precision = opt.frame_precision;
  best.visible_w = w;
  best.visible_h = h;
  return best;
}

}  // namespace av2enc::me2

#endif  // AV2ENC_ENCODER_INTER_ME2_ME2_CORE_H_
