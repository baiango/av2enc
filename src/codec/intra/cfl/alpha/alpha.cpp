#include "alpha.h"

#include <cstdlib>

namespace av2enc::cfl::alpha {
namespace {

bool valid_source(const SourcePlane& source) {
  return source.plane != nullptr && source.stride > 0 && source.max_x >= 0 &&
         source.max_y >= 0;
}

// Keep lambda in a range where `sse * 65536 + lambda_q8 * rate_q8` cannot
// overflow int64 and cannot be negative. `1 << 40` is far above any calibrated
// encoder lambda (~1e3) while leaving ~23 bits for the rate term.
bool valid_config(const Config& config) {
  if (!config.cost.use_rate) {
    return true;
  }
  return config.cost.lambda_q8 >= 0 &&
         config.cost.lambda_q8 <= (std::int64_t{1} << 40);
}

// Same descriptor fields as cfl.cpp::same_geometry(); u and v must share one
// luma plane and one block geometry.
bool same_geometry(const Block& a, const Block& b) {
  return a.luma == b.luma && a.luma_stride == b.luma_stride &&
         a.luma_max_x == b.luma_max_x && a.luma_max_y == b.luma_max_y &&
         a.sb_top_y == b.sb_top_y && a.x == b.x && a.y == b.y &&
         a.w == b.w && a.h == b.h && a.coded_width == b.coded_width &&
         a.coded_height == b.coded_height && a.bit_depth == b.bit_depth &&
         a.ds_filter == b.ds_filter && a.sub_x == b.sub_x &&
         a.sub_y == b.sub_y;
}

Candidate evaluate_with_luma(const Signalling& syntax, const Block& u,
                             const Block& v, const SourcePlane& source_u,
                             const SourcePlane& source_v, const CflCdfs& cdfs,
                             const Scratch& scratch) {
  Candidate candidate{};
  candidate.syntax = syntax;
  candidate.rate_q8 = alpha_syntax_rate_q8(syntax, cdfs);
  if (syntax.cfl_index == kIndexDerivedAlpha) {
    candidate.explicit_alpha = false;
    candidate.q8_u = derive_implicit_alpha(u, scratch);
    candidate.q8_v = derive_implicit_alpha(v, scratch);
  } else if (syntax.cfl_index == kIndexExplicit) {
    const std::optional<AlphaPair> pair =
        pair_from_signalling_checked(syntax);
    if (!pair.has_value()) {
      candidate.syntax.cfl_index = -1;
      return candidate;
    }
    candidate.explicit_alpha = true;
    candidate.explicit_pair = *pair;
    candidate.q8_u = alpha_q8(pair->u);
    candidate.q8_v = alpha_q8(pair->v);
  } else {
    candidate.syntax.cfl_index = -1;
    return candidate;
  }
  candidate.sse_u = prediction_sse(u, candidate.q8_u, scratch, source_u);
  candidate.sse_v = prediction_sse(v, candidate.q8_v, scratch, source_v);
  return candidate;
}

// True when candidate `a` should replace `b` (strictly better, or equal cost
// and lower SSE). Exact ties keep the incumbent, so callers control the
// deterministic order by iteration order.
bool better(std::int64_t a_cost, std::int64_t a_sse, std::int64_t b_cost,
            std::int64_t b_sse) {
  if (a_cost != b_cost) {
    return a_cost < b_cost;
  }
  return a_sse < b_sse;
}

std::optional<Candidate> search_explicit_with_luma(
    const Block& u, const Block& v, const SourcePlane& source_u,
    const SourcePlane& source_v, const Config& config, const CflCdfs& cdfs,
    const Scratch& scratch) {
  std::int64_t sse_u[kPerPlaneAlphas]{};
  std::int64_t sse_v[kPerPlaneAlphas]{};
  for (int i = 0; i < kPerPlaneAlphas; ++i) {
    sse_u[i] = prediction_sse(u, alpha_q8(per_plane_alpha(i)), scratch,
                              source_u);
    sse_v[i] = prediction_sse(v, alpha_q8(per_plane_alpha(i)), scratch,
                              source_v);
  }

  const int index_rate =
      symbol_rate_q8(cdfs.cfl_index, kCflIndexSymbols, 0);
  bool have_best = false;
  std::int64_t best_cost = 0;
  std::int64_t best_sse = 0;
  Candidate best{};

  for (int signs = 0; signs < kJointSigns; ++signs) {
    const int sign_u = joint_sign_u(signs);
    const int sign_v = joint_sign_v(signs);
    const int sign_rate =
        symbol_rate_q8(cdfs.cfl_sign, kCflSignSymbols, signs);
    const int ctx_u = alpha_u_context(signs);
    const int ctx_v = alpha_v_context(signs);
    const int mag_begin_u = sign_u == kSignZero ? 0 : 1;
    const int mag_end_u = sign_u == kSignZero ? 0 : kAlphabetSize;
    const int mag_begin_v = sign_v == kSignZero ? 0 : 1;
    const int mag_end_v = sign_v == kSignZero ? 0 : kAlphabetSize;

    for (int mag_u = mag_begin_u; mag_u <= mag_end_u; ++mag_u) {
      const Alpha alpha_u =
          sign_u == kSignZero
              ? Alpha{kSignZero, 0}
              : Alpha{static_cast<std::int8_t>(sign_u),
                      static_cast<std::int8_t>(mag_u)};
      const int index_u = per_plane_index(alpha_u);
      const int rate_u =
          sign_u == kSignZero
              ? 0
              : symbol_rate_q8(cdfs.cfl_alpha[ctx_u], kCflAlphaSymbols,
                               mag_u - 1);
      for (int mag_v = mag_begin_v; mag_v <= mag_end_v; ++mag_v) {
        const Alpha alpha_v =
            sign_v == kSignZero
                ? Alpha{kSignZero, 0}
                : Alpha{static_cast<std::int8_t>(sign_v),
                        static_cast<std::int8_t>(mag_v)};
        const int index_v = per_plane_index(alpha_v);
        const int rate_v =
            sign_v == kSignZero
                ? 0
                : symbol_rate_q8(cdfs.cfl_alpha[ctx_v], kCflAlphaSymbols,
                                 mag_v - 1);
        const std::int64_t total_sse = sse_u[index_u] + sse_v[index_v];
        const int total_rate = index_rate + sign_rate + rate_u + rate_v;
        const std::int64_t cost = config.cost.cost_q16(total_sse, total_rate);
        if (!have_best ||
            better(cost, total_sse, best_cost, best_sse)) {
          have_best = true;
          best_cost = cost;
          best_sse = total_sse;
          const AlphaPair pair{alpha_u, alpha_v};
          best = Candidate{};
          best.explicit_alpha = true;
          best.explicit_pair = pair;
          best.syntax = explicit_signalling(pair);
          best.q8_u = alpha_q8(alpha_u);
          best.q8_v = alpha_q8(alpha_v);
          best.sse_u = sse_u[index_u];
          best.sse_v = sse_v[index_v];
          best.rate_q8 = total_rate;
        }
      }
    }
  }
  if (!have_best) {
    return std::nullopt;
  }
  return best;
}

}  // namespace

AlphaPair quantise_pair(int q8_u, int q8_v) {
  Alpha u = quantise_q8(q8_u);
  Alpha v = quantise_q8(q8_v);
  if (signalable(AlphaPair{u, v})) {
    return AlphaPair{u, v};
  }
  // Only reachable when both slopes quantise to zero. Keep the larger slope's
  // sign at magnitude 1 (deterministic: U wins an exact tie, and U positive
  // when both slopes are zero), mirroring the core's (zero,zero) fallback.
  const int magnitude_u = std::abs(q8_u);
  const int magnitude_v = std::abs(q8_v);
  if (magnitude_u == 0 && magnitude_v == 0) {
    return AlphaPair{Alpha{kSignPos, 1}, Alpha{kSignZero, 0}};
  }
  if (magnitude_u >= magnitude_v) {
    u = Alpha{static_cast<std::int8_t>(q8_u < 0 ? kSignNeg : kSignPos), 1};
  } else {
    v = Alpha{static_cast<std::int8_t>(q8_v < 0 ? kSignNeg : kSignPos), 1};
  }
  return AlphaPair{u, v};
}

AlphaPair quantise_derived_pair(const Block& u, const Block& v,
                                const Scratch& scratch) {
  return quantise_pair(derive_implicit_alpha(u, scratch),
                       derive_implicit_alpha(v, scratch));
}

Signalling signalling_for(AlphaPair p) {
  if (!signalable(p)) {
    Signalling invalid{};
    invalid.cfl_index = -1;
    return invalid;
  }
  return explicit_signalling(p);
}

std::optional<AlphaPair> pair_from_signalling_checked(const Signalling& s) {
  if (s.cfl_index != kIndexExplicit) {
    return std::nullopt;
  }
  if (s.alpha_signs < 0 || s.alpha_signs >= kJointSigns) {
    return std::nullopt;
  }
  const int sign_u = joint_sign_u(s.alpha_signs);
  const int sign_v = joint_sign_v(s.alpha_signs);
  if (sign_u == kSignZero) {
    if (s.alpha_u != 0) {
      return std::nullopt;
    }
  } else if (s.alpha_u < 0 || s.alpha_u >= kAlphabetSize) {
    return std::nullopt;
  }
  if (sign_v == kSignZero) {
    if (s.alpha_v != 0) {
      return std::nullopt;
    }
  } else if (s.alpha_v < 0 || s.alpha_v >= kAlphabetSize) {
    return std::nullopt;
  }
  const AlphaPair pair = pair_from_signalling(s);
  if (!signalable(pair)) {
    return std::nullopt;
  }
  return pair;
}

std::optional<Candidate> evaluate_signalling(
    const Signalling& syntax, const Block& u, const Block& v,
    const SourcePlane& source_u, const SourcePlane& source_v,
    const CflCdfs& cdfs, Scratch& scratch) {
  if (!valid(u) || !valid(v) || !same_geometry(u, v) ||
      !valid_source(source_u) || !valid_source(source_v)) {
    return std::nullopt;
  }
  compute_luma(u, scratch);
  if (syntax.cfl_index == kIndexExplicit) {
    if (!pair_from_signalling_checked(syntax).has_value()) {
      return std::nullopt;
    }
  } else if (syntax.cfl_index != kIndexDerivedAlpha) {
    return std::nullopt;
  }
  return evaluate_with_luma(syntax, u, v, source_u, source_v, cdfs, scratch);
}

std::optional<Candidate> search_explicit(const Block& u, const Block& v,
                                         const SourcePlane& source_u,
                                         const SourcePlane& source_v,
                                         const Config& config,
                                         const CflCdfs& cdfs,
                                         Scratch& scratch) {
  if (!valid(u) || !valid(v) || !same_geometry(u, v) ||
      !valid_source(source_u) || !valid_source(source_v) ||
      !valid_config(config)) {
    return std::nullopt;
  }
  compute_luma(u, scratch);
  return search_explicit_with_luma(u, v, source_u, source_v, config, cdfs,
                                   scratch);
}

std::optional<Decision> decide_cfl_alpha(const Block& u, const Block& v,
                                         const SourcePlane& source_u,
                                         const SourcePlane& source_v,
                                         const Config& config,
                                         const CflCdfs& cdfs,
                                         Scratch& scratch) {
  if (!valid(u) || !valid(v) || !same_geometry(u, v) ||
      !valid_source(source_u) || !valid_source(source_v) ||
      !valid_config(config)) {
    return std::nullopt;
  }
  if (!config.allow_derived && !config.allow_explicit) {
    return std::nullopt;
  }
  compute_luma(u, scratch);

  Decision decision{};
  if (config.allow_derived) {
    const Signalling syntax = derived_signalling();
    decision.derived = evaluate_with_luma(syntax, u, v, source_u, source_v,
                                          cdfs, scratch);
    decision.has_derived = decision.derived.syntax.cfl_index >= 0;
  }
  if (config.allow_explicit) {
    const std::optional<Candidate> explicit_candidate =
        search_explicit_with_luma(u, v, source_u, source_v, config, cdfs,
                                  scratch);
    if (explicit_candidate.has_value()) {
      decision.explicit_alpha = *explicit_candidate;
      decision.has_explicit = true;
    }
  }

  if (!decision.has_derived && !decision.has_explicit) {
    return std::nullopt;
  }
  if (!decision.has_explicit) {
    decision.chosen = decision.derived;
    return decision;
  }
  if (!decision.has_derived) {
    decision.chosen = decision.explicit_alpha;
    return decision;
  }

  const std::int64_t derived_cost =
      decision.derived.cost_q16(config.cost);
  const std::int64_t explicit_cost =
      decision.explicit_alpha.cost_q16(config.cost);
  const std::int64_t derived_sse = decision.derived.total_sse();
  const std::int64_t explicit_sse = decision.explicit_alpha.total_sse();
  bool choose_explicit = false;
  if (derived_cost != explicit_cost) {
    choose_explicit = explicit_cost < derived_cost;
  } else if (derived_sse != explicit_sse) {
    choose_explicit = explicit_sse < derived_sse;
  } else {
    choose_explicit = config.tie_break == TieBreak::kPreferExplicit;
  }
  decision.chosen =
      choose_explicit ? decision.explicit_alpha : decision.derived;
  return decision;
}

void apply_candidate(const Block& u, const Block& v, const Candidate& candidate,
                     const Scratch& scratch) {
  apply_prediction(u, candidate.q8_u, scratch);
  apply_prediction(v, candidate.q8_v, scratch);
}

}  // namespace av2enc::cfl::alpha
