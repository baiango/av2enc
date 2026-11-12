#include "codec/inter/multiref/decide.h"

#include <algorithm>
#include <limits>

namespace av2enc::video::multiref {
namespace {

CompoundModeParams params_from_decision(const adv::CompoundDecision& d) {
  CompoundModeParams out;
  out.type = d.type;
  out.cwp_weight = d.cwp_weight;
  out.cwp_index = 0;
  out.wedge_index = d.wedge_index;
  out.wedge_sign = d.wedge_sign;
  out.mask_type = d.mask_type;
  out.use_imp_mask = d.use_imp_mask;
  return out;
}

int single_mode_bits(int mode) {
  switch (mode) {
    case 0:
      return 2;  // single_mode = 1 (GLOBALMV): two binary symbols
    case 1:
      return 1;  // single_mode = 0 (NEARMV)
    default:
      return 2;  // single_mode = 2 (NEWMV)
  }
}

}  // namespace

void evaluate_compound_modes(const CompoundPrediction& prediction,
                             const RefFrameMap* map, int ref0, int ref1,
                             const std::uint8_t* source, int source_stride,
                             const DecisionOptions& options,
                             std::vector<EvaluatedMode>& out) {
  out.clear();
  if (source == nullptr || source_stride <= 0) return;
  const int w = prediction.luma_rect.w;
  const int h = prediction.luma_rect.h;
  std::vector<std::uint8_t> scratch(static_cast<std::size_t>(w) *
                                    static_cast<std::size_t>(h));
  const bool same_side = map != nullptr ? map->is_same_side(ref0, ref1) : true;
  const bool imp_ok = options.enable_imp_mask && options.seq_enable_imp_msk_bld;

  const auto measure = [&](const CompoundModeParams& params) {
    EvaluatedMode entry;
    entry.params = params;
    entry.sse = compound_luma_sse(prediction, params, source, source_stride,
                                  scratch.data(), w);
    out.push_back(entry);
  };

  if (options.enable_average) {
    if (imp_ok) {
      CompoundModeParams params;
      params.type = adv::CompoundType::kAverage;
      params.cwp_weight = kCwpEqual;
      params.use_imp_mask = true;
      measure(params);
    }
    CompoundModeParams equal;
    equal.type = adv::CompoundType::kAverage;
    equal.cwp_weight = kCwpEqual;
    if (!imp_ok) measure(equal);
    if (options.enable_cwp) {
      for (int i = 1; i < kMaxCwpNum; ++i) {
        CompoundModeParams params;
        params.type = adv::CompoundType::kAverage;
        params.cwp_index = i;
        params.cwp_weight = cwp_weight_from_index(i, same_side);
        measure(params);
      }
    }
  }
  if (options.enable_wedge && adv::bsize_has_wedge(prediction.bsize)) {
    for (int sign = 0; sign <= 1; ++sign) {
      for (int index = 0; index < adv::kWedgeTypes; ++index) {
        if (!adv::wedge_index_signalable(index)) continue;
        CompoundModeParams params;
        params.type = adv::CompoundType::kWedge;
        params.wedge_index = index;
        params.wedge_sign = sign;
        measure(params);
      }
    }
  }
  if (options.enable_diffwtd) {
    for (int mask_type = 0; mask_type <= 1; ++mask_type) {
      CompoundModeParams params;
      params.type = adv::CompoundType::kDiffWtd;
      params.mask_type = mask_type;
      measure(params);
    }
  }
}

bool decide_compound(const DecisionInput& input, DecisionResult& result,
                     std::string& error) {
  error.clear();
  result = DecisionResult{};
  if (input.source == nullptr || input.source_stride <= 0) {
    error = "source block missing";
    return false;
  }
  if (input.luma_rect.w <= 0 || input.luma_rect.h <= 0) {
    error = "empty block";
    return false;
  }
  const RefFrameMap* map = input.map;
  const bool compound_allowed =
      input.options.allow_compound &&
      (map == nullptr ||
       map->comp_allowed_at(input.luma_rect.w, input.luma_rect.h));

  std::vector<std::uint8_t> scratch(
      static_cast<std::size_t>(input.luma_rect.w) *
      static_cast<std::size_t>(input.luma_rect.h));

  if (compound_allowed) {
    for (const CompoundCandidateInput& candidate : input.compound_candidates) {
      CompoundPrediction owned;
      const CompoundPrediction* prediction = nullptr;
      if (candidate.has_prediction) {
        prediction = &candidate.prediction;
      } else {
        if (input.refs[0] == nullptr || input.refs[1] == nullptr) {
          error = "candidate needs predictions but no RefFrames were given";
          return false;
        }
        if (!build_compound_prediction(*input.refs[0], *input.refs[1],
                                       input.luma_rect, input.bsize,
                                       candidate.mv[0], candidate.mv[1],
                                       mc::InterpFilter::kEighttap,
                                       input.mi_cols, input.mi_rows, owned,
                                       error)) {
          return false;
        }
        prediction = &owned;
      }
      if (prediction->luma_rect.w != input.luma_rect.w ||
          prediction->luma_rect.h != input.luma_rect.h) {
        error = "candidate prediction geometry mismatch";
        return false;
      }
      const bool same_side =
          map != nullptr ? map->is_same_side(candidate.ref0, candidate.ref1)
                         : true;
      const bool imp_eligible = implicit_mask_eligible(
          input.options.seq_enable_imp_msk_bld && input.options.enable_imp_mask,
          CompoundModeParams{}, candidate.global_global_mv, false, false);

      adv::CompoundSearchInput search{};
      search.bsize = input.bsize;
      search.bit_depth = 8;
      search.rect = adv::BlockRect{0, 0, input.luma_rect.w,
                                   input.luma_rect.h};
      search.p0 = prediction->pred[0][0].samples.data();
      search.p0_stride = prediction->pred[0][0].stride;
      search.p1 = prediction->pred[1][0].samples.data();
      search.p1_stride = prediction->pred[1][0].stride;
      search.source = input.source;
      search.source_stride = input.source_stride;
      search.scratch = scratch.data();
      search.scratch_stride = input.luma_rect.w;
      // If the implicit mask applies, §7.13.3.14 replaces the CWP_EQUAL
      // average.  With CWP disabled, search_compound() would additionally
      // evaluate a plain CWP_EQUAL candidate that the decoder cannot produce,
      // so the average family is handled explicitly below and skipped here.
      const bool imp_only =
          imp_eligible && !input.options.enable_cwp;
      search.enable_average = input.options.enable_average && !imp_only;
      search.enable_cwp = input.options.enable_cwp;
      search.enable_wedge = input.options.enable_wedge;
      search.enable_diffwtd = input.options.enable_diffwtd;
      search.enable_imp_mask = input.options.enable_imp_mask;
      search.imp_mask_eligible = imp_eligible;
      search.mi_cols = input.mi_cols;
      search.mi_rows = input.mi_rows;
      search.ref_start_x[0] = prediction->ref_start_x[0][0];
      search.ref_start_x[1] = prediction->ref_start_x[1][0];
      search.ref_start_y[0] = prediction->ref_start_y[0][0];
      search.ref_start_y[1] = prediction->ref_start_y[1][0];
      for (int i = 0; i < kMaxCwpNum; ++i) {
        search.cwp_candidates[i] = cwp_weight_from_index(i, same_side);
      }
      search.cwp_count = input.options.enable_cwp ? kMaxCwpNum : 0;
      const auto decision = adv::search_compound(search);
      if (!decision.has_value() && !imp_only) {
        error = "search_compound rejected the candidate";
        return false;
      }
      const auto consider = [&](const CompoundModeParams& params,
                                std::int64_t sse) {
        const int bits =
            compound_mode_bits(params, same_side) + candidate.mv_bits;
        const double cost =
            static_cast<double>(sse) + input.options.lambda * bits;
        if (result.kind == DecisionKind::kNone || cost < result.cost) {
          result.kind = DecisionKind::kCompound;
          result.ref0 = candidate.ref0;
          result.ref1 = candidate.ref1;
          result.mv[0] = candidate.mv[0];
          result.mv[1] = candidate.mv[1];
          result.compound = params;
          result.sse = sse;
          result.rate_bits = bits;
          result.cost = cost;
        }
      };
      if (decision.has_value()) {
        consider(params_from_decision(*decision), decision->sse);
      }
      if (imp_only && input.options.enable_average) {
        CompoundModeParams params;
        params.type = adv::CompoundType::kAverage;
        params.cwp_weight = kCwpEqual;
        params.use_imp_mask = true;
        const std::int64_t sse = compound_luma_sse(
            *prediction, params, input.source, input.source_stride,
            scratch.data(), input.luma_rect.w);
        if (sse != std::numeric_limits<std::int64_t>::max()) {
          consider(params, sse);
        }
      }
    }
  }

  if (input.options.allow_single) {
    for (const SingleCandidateInput& candidate : input.single_candidates) {
      const int bits = single_mode_bits(candidate.mode) +
                       (candidate.mode == 0 ? 0 : candidate.drl_index + 1) +
                       candidate.mv_bits;
      const double cost =
          static_cast<double>(candidate.sse) + input.options.lambda * bits;
      // Strict `<` keeps the earlier candidate; compound candidates are
      // evaluated first, so a tie favours compound.  This is the documented
      // deterministic tie-break.
      if (result.kind == DecisionKind::kNone || cost < result.cost) {
        result.kind = DecisionKind::kSingle;
        result.ref0 = candidate.ref;
        result.ref1 = -1;
        result.mv[0] = candidate.mv;
        result.mv[1] = mc::Mv{};
        result.single_mode = candidate.mode;
        result.drl_index = candidate.drl_index;
        result.sse = candidate.sse;
        result.rate_bits = bits;
        result.cost = cost;
      }
    }
  }

  if (result.kind == DecisionKind::kNone) {
    error = "no candidate selected";
    return false;
  }
  return true;
}

}  // namespace av2enc::video::multiref
