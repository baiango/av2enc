// CPU reference for AV2 v1.0.0 MHCCP (§7.13.6-§7.13.8).
//
// The integer arithmetic is shared with the CUDA kernel through
// mhccp_common.h. This translation unit owns validation, the source-SSE
// evaluation, the direction search and the explicit/derived/multi decision.

#include "mhccp.h"

#include <algorithm>

namespace av2enc::cfl::mhccp {
namespace {

int clamp_int(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

SourcePlane to_mhccp(const cfl::SourcePlane& source) {
  SourcePlane out{};
  out.plane = source.plane;
  out.stride = source.stride;
  out.max_x = source.max_x;
  out.max_y = source.max_y;
  return out;
}

bool same_geometry(const Block& a, const Block& b) {
  return a.luma == b.luma && a.luma_stride == b.luma_stride &&
         a.luma_max_x == b.luma_max_x && a.luma_max_y == b.luma_max_y &&
         a.chroma_stride == b.chroma_stride &&
         a.chroma_max_x == b.chroma_max_x &&
         a.chroma_max_y == b.chroma_max_y && a.x == b.x && a.y == b.y &&
         a.w == b.w && a.h == b.h && a.bit_depth == b.bit_depth &&
         a.ds_filter == b.ds_filter && a.sub_x == b.sub_x &&
         a.sub_y == b.sub_y && a.have_above == b.have_above &&
         a.have_left == b.have_left &&
         a.num4_above_right == b.num4_above_right &&
         a.num4_below_left == b.num4_below_left &&
         a.mi_col_end == b.mi_col_end && a.mi_row_end == b.mi_row_end &&
         a.sb_luma_y == b.sb_luma_y;
}

std::int64_t source_sse(const Block& block, const MultiParams& params,
                        int cfl_mh_dir, const Refs& refs,
                        const SourcePlane& source) {
  if (source.plane == nullptr || source.stride <= 0) {
    return 0;
  }
  std::int64_t sse = 0;
  for (int i = 0; i < block.h; ++i) {
    const int source_y = clamp_int(0, source.max_y, block.y + i);
    for (int j = 0; j < block.w; ++j) {
      const int source_x = clamp_int(0, source.max_x, block.x + j);
      const int predicted = predict_sample(block, refs, params, cfl_mh_dir, i, j);
      const int sample = static_cast<int>(
          source.plane[static_cast<std::size_t>(source_y) *
                           static_cast<std::size_t>(source.stride) +
                       static_cast<std::size_t>(source_x)]);
      const int diff = predicted - sample;
      sse += static_cast<std::int64_t>(diff) * diff;
    }
  }
  return sse;
}

}  // namespace

bool valid_pair(const Block& u, const Block& v) {
  return valid(u) && valid(v) && same_geometry(u, v);
}

std::int64_t prediction_sse(const Block& block, const MultiParams& params,
                            int cfl_mh_dir, const Refs& refs,
                            const SourcePlane& source) {
  if (cfl_mh_dir < 0 || cfl_mh_dir >= kDirections) {
    return 0;
  }
  return source_sse(block, params, cfl_mh_dir, refs, source);
}

void apply_mhccp(const Block& block, const MultiParams& params,
                 int cfl_mh_dir, const Refs& refs) {
  if (cfl_mh_dir < 0 || cfl_mh_dir >= kDirections) {
    return;
  }
  for (int i = 0; i < block.h; ++i) {
    for (int j = 0; j < block.w; ++j) {
      const int value = predict_sample(block, refs, params, cfl_mh_dir, i, j);
      const std::size_t index =
          static_cast<std::size_t>(block.y + i) *
              static_cast<std::size_t>(block.chroma_stride) +
          static_cast<std::size_t>(block.x + j);
      block.chroma[index] = static_cast<std::uint8_t>(value);
    }
  }
}

void apply_mhccp_pair(const Block& u, const Block& v, int cfl_mh_dir,
                      const MultiParams& params_u,
                      const MultiParams& params_v) {
  Refs refs_u;
  Refs refs_v;
  if (!capture_refs(u, refs_u) || !capture_refs(v, refs_v)) {
    return;
  }
  apply_mhccp(u, params_u, cfl_mh_dir, refs_u);
  apply_mhccp(v, params_v, cfl_mh_dir, refs_v);
}

std::optional<Candidate> evaluate_mhccp(const Block& u, const Block& v,
                                        const SourcePlane& source_u,
                                        const SourcePlane& source_v,
                                        int cfl_mh_dir) {
  if (!valid_pair(u, v) || cfl_mh_dir < 0 || cfl_mh_dir >= kDirections ||
      source_u.plane == nullptr || source_v.plane == nullptr) {
    return std::nullopt;
  }
  Refs refs_u;
  Refs refs_v;
  if (!capture_refs(u, refs_u) || !capture_refs(v, refs_v)) {
    return std::nullopt;
  }
  Candidate candidate{};
  candidate.mh_dir = cfl_mh_dir;
  candidate.params_u =
      derive_multi_params(u, refs_u, cfl_mh_dir, &candidate.count_u);
  candidate.params_v =
      derive_multi_params(v, refs_v, cfl_mh_dir, &candidate.count_v);
  candidate.sse_u =
      source_sse(u, candidate.params_u, cfl_mh_dir, refs_u, source_u);
  candidate.sse_v =
      source_sse(v, candidate.params_v, cfl_mh_dir, refs_v, source_v);
  return candidate;
}

std::optional<Candidate> search_mhccp(const Block& u, const Block& v,
                                      const SourcePlane& source_u,
                                      const SourcePlane& source_v) {
  if (!valid_pair(u, v) || source_u.plane == nullptr ||
      source_v.plane == nullptr) {
    return std::nullopt;
  }
  Refs refs_u;
  Refs refs_v;
  if (!capture_refs(u, refs_u) || !capture_refs(v, refs_v)) {
    return std::nullopt;
  }

  std::optional<Candidate> best;
  for (int dir = 0; dir < kDirections; ++dir) {
    Candidate candidate{};
    candidate.mh_dir = dir;
    candidate.params_u =
        derive_multi_params(u, refs_u, dir, &candidate.count_u);
    candidate.params_v =
        derive_multi_params(v, refs_v, dir, &candidate.count_v);
    candidate.sse_u = source_sse(u, candidate.params_u, dir, refs_u, source_u);
    candidate.sse_v = source_sse(v, candidate.params_v, dir, refs_v, source_v);
    if (!best.has_value() || candidate.total_sse() < best->total_sse()) {
      best = candidate;
    }
  }
  return best;
}

std::optional<Decision> decide_cfl(const cfl::Block& u, const cfl::Block& v,
                                   const cfl::SourcePlane& source_u,
                                   const cfl::SourcePlane& source_v,
                                   int num4_above_right,
                                   int num4_below_left,
                                   int tile_col_end, int tile_row_end,
                                   const DecisionOptions& options,
                                   cfl::Scratch& scratch) {
  if (!cfl::valid(u) || !cfl::valid(v) || source_u.plane == nullptr ||
      source_v.plane == nullptr) {
    return std::nullopt;
  }

  const bool cfl_ok = cfl::cfl_allowed(
      u.w, u.h, options.enable_cfl_intra, options.is_chroma_part,
      options.frame_is_intra, options.cfl_allowed_in_sdp, options.lossless);

  cfl::compute_luma(u, scratch);

  std::optional<Decision> best;

  if (options.allow_explicit && cfl_ok) {
    const auto explicit_result =
        cfl::search_alpha(u, v, source_u, source_v,
                          cfl::SearchMode::kExhaustiveSse, scratch);
    if (explicit_result) {
      Decision decision{};
      decision.kind = HypothesisKind::kExplicit;
      decision.cfl_index = kIndexExplicit;
      decision.alpha = explicit_result->alpha;
      decision.syntax = explicit_result->syntax;
      decision.alpha_q8_u = cfl::alpha_q8(explicit_result->alpha.u);
      decision.alpha_q8_v = cfl::alpha_q8(explicit_result->alpha.v);
      decision.sse_u = explicit_result->sse_u;
      decision.sse_v = explicit_result->sse_v;
      best = decision;
    }
  }

  if (options.allow_derived && cfl_ok) {
    Decision decision{};
    decision.kind = HypothesisKind::kDerivedAlpha;
    decision.cfl_index = kIndexDerivedAlpha;
    decision.syntax = cfl::derived_signalling();
    decision.alpha_q8_u = cfl::derive_implicit_alpha(u, scratch);
    decision.alpha_q8_v = cfl::derive_implicit_alpha(v, scratch);
    decision.sse_u =
        cfl::prediction_sse(u, decision.alpha_q8_u, scratch, source_u);
    decision.sse_v =
        cfl::prediction_sse(v, decision.alpha_q8_v, scratch, source_v);
    if (!best.has_value() || decision.total_sse() < best->total_sse()) {
      best = decision;
    }
  }

  if (options.allow_multi &&
      allowed(u.w, u.h, options.enable_mhccp, options.is_chroma_part,
              options.frame_is_intra, options.cfl_allowed_in_sdp,
              options.lossless)) {
    const Block multi_u =
        make_block(u, num4_above_right, num4_below_left, tile_col_end,
                   tile_row_end);
    const Block multi_v =
        make_block(v, num4_above_right, num4_below_left, tile_col_end,
                   tile_row_end);
    const auto multi_result =
        search_mhccp(multi_u, multi_v, to_mhccp(source_u), to_mhccp(source_v));
    if (multi_result) {
      Decision decision{};
      decision.kind = HypothesisKind::kMulti;
      decision.cfl_index = kIndexMulti;
      decision.cfl_mh_dir = multi_result->mh_dir;
      decision.syntax.cfl_index = kIndexMulti;
      decision.params_u = multi_result->params_u;
      decision.params_v = multi_result->params_v;
      decision.multi_u = multi_u;
      decision.multi_v = multi_v;
      decision.sse_u = multi_result->sse_u;
      decision.sse_v = multi_result->sse_v;
      if (!best.has_value() || decision.total_sse() < best->total_sse()) {
        best = decision;
      }
    }
  }

  return best;
}

void apply_decision(const cfl::Block& u, const cfl::Block& v,
                    const Decision& decision, cfl::Scratch& scratch) {
  switch (decision.kind) {
    case HypothesisKind::kExplicit:
    case HypothesisKind::kDerivedAlpha:
      cfl::apply_prediction(u, decision.alpha_q8_u, scratch);
      cfl::apply_prediction(v, decision.alpha_q8_v, scratch);
      break;
    case HypothesisKind::kMulti:
      apply_mhccp_pair(decision.multi_u, decision.multi_v,
                       decision.cfl_mh_dir, decision.params_u,
                       decision.params_v);
      break;
  }
}

Block make_block(const cfl::Block& base, int num4_above_right,
                 int num4_below_left, int tile_col_end, int tile_row_end) {
  Block block{};
  block.luma = base.luma;
  block.luma_stride = base.luma_stride;
  block.luma_max_x = base.luma_max_x;
  block.luma_max_y = base.luma_max_y;
  block.chroma = base.chroma;
  block.chroma_stride = base.chroma_stride;
  block.chroma_max_x =
      std::max(base.coded_width, base.x + base.w) - 1;
  block.chroma_max_y =
      std::max(base.coded_height, base.y + base.h) - 1;
  block.x = base.x;
  block.y = base.y;
  block.w = base.w;
  block.h = base.h;
  block.bit_depth = base.bit_depth;
  block.ds_filter = base.ds_filter;
  block.sub_x = base.sub_x;
  block.sub_y = base.sub_y;
  block.have_above = base.have_above;
  block.have_left = base.have_left;
  block.num4_above_right = num4_above_right;
  block.num4_below_left = num4_below_left;
  block.mi_col_end = tile_col_end > 0
                         ? tile_col_end
                         : base.coded_width << base.sub_x;
  block.mi_row_end = tile_row_end > 0
                         ? tile_row_end
                         : base.coded_height << base.sub_y;
  block.sb_luma_y = base.sb_top_y;
  return block;
}

}  // namespace av2enc::cfl::mhccp
