#!/usr/bin/env python3
"""Generate an experiment-only patched copy of src/encoder/m2/m2_frame.cpp.

The repository file is never modified.  The generated copy adds env-gated
hooks for the M2 P-frame rate-aware decision layer:

    AV2ENC_M2_RDO=1            enable the layer
    AV2ENC_M2_RDO_SCALE=<x>    sse_scale (default 0.5)
    AV2ENC_M2_RDO_SSE_ONLY=1   lambda = 0 (distortion-only control)
    AV2ENC_M2_RDO_METRIC=0|1|2 0 SSE (default), 1 SAD, 2 SATD
    AV2ENC_M2_RDO_RATE_TIE=1   rate-first tie-break (default distortion-first)
    AV2ENC_M2_RDO_ALL_RESIDUAL=1  evaluate residual for every candidate
    AV2ENC_M2_RDO_NO_PART=1    keep the legacy variance partition
    AV2ENC_M2_RDO_NO_RESIDUAL=1 keep skip_flag = 1 only
    AV2ENC_M2_RDO_TRACE=1      per-leaf cost trace on stderr

With AV2ENC_M2_RDO unset the patched binary must be byte-identical to the
unpatched M2 gate; experiment/run_m2_rdo_experiment.sh checks that.

The new code is a separate `encode_leaf_rdo()` member, so the baseline
`encode_leaf()` is not modified.  The only edits to existing control flow are
the two `if (m2_rdo_enabled())` dispatch sites in `collect_node()` and
`encode()`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

INCLUDE_ANCHOR = '#include "encoder/m2/merge/merge.h"\n'

RDO_INCLUDES = r'''
// Experiment-only M2 RDO hook.  The production integration should call the
// same API from encode_leaf() / encode() instead of patching a copy.
#include "encoder/m2/rdo/rdo.h"
#include <cstdlib>
'''

HELPERS = r'''
// ---------------------------------------------------------------------------
// Experiment-only M2 RDO helpers.  Kept in this generated copy; the production
// call sites are documented in src/encoder/m2/rdo/README.md.
// ---------------------------------------------------------------------------
bool m2_rdo_enabled() {
  return std::getenv("AV2ENC_M2_RDO") != nullptr;
}

bool m2_rdo_partition_enabled() {
  return m2_rdo_enabled() &&
         std::getenv("AV2ENC_M2_RDO_NO_PART") == nullptr;
}

bool m2_rdo_residual_enabled() {
  return m2_rdo_enabled() &&
         std::getenv("AV2ENC_M2_RDO_NO_RESIDUAL") == nullptr;
}

av2enc::m2::rdo::RdoConfig m2_rdo_config_from_env() {
  std::string config_error;
  av2enc::m2::rdo::RdoConfig config =
      av2enc::m2::rdo::calibrated_m2_config(&config_error);
  if (!config_error.empty()) return config;
  if (const char* scale = std::getenv("AV2ENC_M2_RDO_SCALE")) {
    const double parsed = std::atof(scale);
    if (parsed > 0.0) config.lambda.sse_scale = parsed;
  }
  if (const char* metric = std::getenv("AV2ENC_M2_RDO_METRIC")) {
    const int parsed = std::atoi(metric);
    if (parsed >= 0 && parsed <= 2) {
      config.metric = static_cast<av2enc::rdo::DistortionMetric>(parsed);
    }
  }
  if (std::getenv("AV2ENC_M2_RDO_SSE_ONLY") != nullptr) {
    config.rate_aware = false;
  }
  if (std::getenv("AV2ENC_M2_RDO_RATE_TIE") != nullptr) {
    config.tie_break = av2enc::rdo::TieBreak::kRateThenDistortion;
  }
  return config;
}
'''

RDO_FUNCTIONS = r'''
  // -------------------------------------------------------------------------
  // Rate-aware leaf decision (experiment hook).
  //
  // Candidate cost: distortion + lambda * (mode + MV + ref + residual) bits.
  // Distortion comes from the shared residual decision (total Y+U+V SSE); the
  // mode/MV term comes from the §9.3 CDF model; residual bits come from
  // m2::residual::decide_inter_block(), which measures the writer's own
  // symbols.  The intra fallback uses the M1 TransformStreamRate adapter.
  // -------------------------------------------------------------------------
  bool encode_leaf_rdo(const PartitionNode& node, std::size_t index,
                       std::string& error) {
    const int r = node.mi_row;
    const int c = node.mi_col;
    const int w = partition::block_width(node.size);
    const int h = partition::block_height(node.size);
    const int w4 = w >> 2;
    const int h4 = h >> 2;
    const int x0 = c * 4;
    const int y0 = r * 4;
    const int cw = w / 2;
    const int ch = h / 2;
    const int cx0 = x0 / 2;
    const int cy0 = y0 / 2;
    syntax::BlockSyntax& entry = blocks_[index];

    const av2enc::m2::rdo::RdoConfig rdo_config = m2_rdo_config_from_env();
    const av2enc::m2::rdo::InterRateModel rate_model;
    av2enc::m2::rdo::RateContext rate_ctx;
    const bool trace_rdo =
        std::getenv("AV2ENC_M2_RDO_TRACE") != nullptr;
    const bool residual_enabled = m2_rdo_residual_enabled();

    // Block-local source copies with the same edge clipping as source_sample().
    std::vector<std::uint8_t> source_y(
        static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    std::vector<std::uint8_t> source_u(
        static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    std::vector<std::uint8_t> source_v(
        static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        source_y[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                 static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(
            source_sample(source_, 0, x0 + x, y0 + y));
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const std::size_t pos =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
            static_cast<std::size_t>(x);
        source_u[pos] = static_cast<std::uint8_t>(
            source_sample(source_, 1, cx0 + x, cy0 + y));
        source_v[pos] = static_cast<std::uint8_t>(
            source_sample(source_, 2, cx0 + x, cy0 + y));
      }
    }

    struct CandidateStore {
      av2enc::m2::rdo::InterCandidateSpec spec;
      std::vector<std::uint8_t> y;
      std::vector<std::uint8_t> u;
      std::vector<std::uint8_t> v;
      av2enc::m2::rdo::ScoredInterCandidate scored;
    };
    std::vector<CandidateStore> stores;
    stores.reserve(4);

    const auto add_candidate =
        [&](av2enc::m2::rdo::InterCandidateKind kind, int inter_mode,
            int drl_index, inter::mc::Mv mv,
            inter::mc::Mv mvd) -> bool {
      CandidateStore store;
      store.spec.kind = kind;
      store.spec.inter_mode = inter_mode;
      store.spec.drl_index = drl_index;
      store.spec.mv = mv;
      store.spec.mvd = mvd;
      store.spec.ref = 0;
      store.spec.force_residual = options_.force_inter_residual;
      if (!predict_all(mv, x0, y0, w, h, store.y, store.u, store.v)) {
        return true;  // candidate unavailable, not a fatal error
      }
      if (rdo_config.metric != av2enc::rdo::DistortionMetric::kSsd) {
        // SAD/SATD pre-selection: use the configured L1-like metric on luma
        // for the candidate cost.  The residual decision still reconstructs
        // and scores SSE internally.
        store.spec.has_distortion_override = true;
        store.spec.distortion_override = static_cast<double>(
            av2enc::rdo::compute_distortion(
                rdo_config.metric,
                std::span<const std::uint8_t>(source_y.data(),
                                              source_y.size()),
                std::span<const std::uint8_t>(store.y.data(), store.y.size()),
                w, h, w));
      }
      av2enc::m2::residual::BlockInput block;
      block.geometry =
          av2enc::m2::residual::BlockGeometry{r, c, w, h, cw, ch};
      block.source[0] = av2enc::m2::residual::PlaneView{
          std::span<const std::uint8_t>(source_y.data(), source_y.size()), w};
      block.source[1] = av2enc::m2::residual::PlaneView{
          std::span<const std::uint8_t>(source_u.data(), source_u.size()), cw};
      block.source[2] = av2enc::m2::residual::PlaneView{
          std::span<const std::uint8_t>(source_v.data(), source_v.size()), cw};
      block.prediction[0] = av2enc::m2::residual::PlaneView{
          std::span<const std::uint8_t>(store.y.data(), store.y.size()), w};
      block.prediction[1] = av2enc::m2::residual::PlaneView{
          std::span<const std::uint8_t>(store.u.data(), store.u.size()), cw};
      block.prediction[2] = av2enc::m2::residual::PlaneView{
          std::span<const std::uint8_t>(store.v.data(), store.v.size()), cw};
      const bool evaluate = residual_enabled;
      if (!av2enc::m2::rdo::score_inter_candidate(
              store.spec, block, evaluate, rate_ctx, rdo_config, rate_model,
              store.scored, &error)) {
        return false;
      }
      if (store.scored.valid) stores.push_back(std::move(store));
      return true;
    };

    // GLOBALMV (zero MV) is always available and needs no stack.
    if (!add_candidate(av2enc::m2::rdo::InterCandidateKind::kSkipGlobal,
                       syntax::kInterGlobalMv, 0, inter::mc::Mv{0, 0},
                       inter::mc::Mv{0, 0})) {
      return false;
    }

    merge::StackParams stack_params;
    merge::MvStack stack;
    bool have_stack = false;
    if (!keyframe_) {
      have_stack = load_stack(r, c, w4, h4, stack_params, stack);
    }

    if (have_stack && !options_.force_global_mv) {
      if (options_.enable_nearmv) {
        const int near_limit =
            (w > 32 && h > 32) ? stack.count : std::min(stack.count, 2);
        for (int i = 0; i < near_limit; ++i) {
          merge::ResolvedMode resolved;
          if (merge::resolve_single_mode(stack, stack_params,
                                         merge::SingleMode::kNearMv, i,
                                         inter::mc::Mv{0, 0},
                                         &resolved) != merge::Status::kOk) {
            continue;
          }
          if (!add_candidate(
                  av2enc::m2::rdo::InterCandidateKind::kMerge,
                  syntax::kInterNearMv, i, resolved.mv[0],
                  inter::mc::Mv{0, 0})) {
            return false;
          }
        }
      }

      if (options_.enable_newmv) {
        inter::mc::Mv me_mv{};
        if (run_me(x0, y0, w, h, me_mv, error)) {
          const inter::mc::Mv pred = stack.cand[0].mv[0];
          const inter::mc::Mv mvd{me_mv.y - pred.y, me_mv.x - pred.x};
          merge::ResolvedMode resolved;
          if (merge::resolve_single_mode(
                  stack, stack_params, merge::SingleMode::kNewMv, 0, mvd,
                  &resolved) == merge::Status::kOk) {
            if (!add_candidate(
                    av2enc::m2::rdo::InterCandidateKind::kNewMv,
                    syntax::kInterNewMv, 0, resolved.mv[0],
                    resolved.mvd[0])) {
              return false;
            }
          }
        }
      }
    }

    // Intra fallback: DC prediction + DCT residual, exactly the core path.
    bool have_intra = false;
    double intra_distortion = 0.0;
    double intra_coeff_bits = 0.0;
    TxPlan intra_y;
    TxPlan intra_u;
    TxPlan intra_v;
    std::vector<std::uint8_t> intra_y_pred;
    std::vector<std::uint8_t> intra_u_pred;
    std::vector<std::uint8_t> intra_v_pred;
    if (options_.enable_intra_fallback || options_.force_intra) {
      if (!intra_dc_predict(recon_[0], source_, 0, x0, y0, w, h,
                            intra_y_pred) ||
          !intra_dc_predict(recon_[1], source_, 1, cx0, cy0, cw, ch,
                            intra_u_pred) ||
          !intra_dc_predict(recon_[2], source_, 2, cx0, cy0, cw, ch,
                            intra_v_pred)) {
        error = "intra DC prediction failed";
        return false;
      }
      if (!compute_tx(source_, 0, x0, y0, w, h, intra_y_pred, quant_step_,
                      intra_y, error) ||
          !compute_tx(source_, 1, cx0, cy0, cw, ch, intra_u_pred, quant_step_,
                      intra_u, error) ||
          !compute_tx(source_, 2, cx0, cy0, cw, ch, intra_v_pred, quant_step_,
                      intra_v, error)) {
        return false;
      }
      intra_distortion =
          static_cast<double>(sse_against_source(source_, 0, x0, y0, w, h,
                                                 intra_y.recon)) +
          static_cast<double>(sse_against_source(source_, 1, cx0, cy0, cw, ch,
                                                 intra_u.recon)) +
          static_cast<double>(sse_against_source(source_, 2, cx0, cy0, cw, ch,
                                                 intra_v.recon));
      std::array<av2enc::m2::rdo::IntraPlaneRateInput, 3> planes;
      planes[0] = av2enc::m2::rdo::IntraPlaneRateInput{
          0, intra_y.tx_size, intra_y.adj_w, intra_y.adj_h,
          std::span<const std::int32_t>(intra_y.levels.data(),
                                        intra_y.levels.size()),
          intra_y.eob};
      planes[1] = av2enc::m2::rdo::IntraPlaneRateInput{
          1, intra_u.tx_size, intra_u.adj_w, intra_u.adj_h,
          std::span<const std::int32_t>(intra_u.levels.data(),
                                        intra_u.levels.size()),
          intra_u.eob};
      planes[2] = av2enc::m2::rdo::IntraPlaneRateInput{
          2, intra_v.tx_size, intra_v.adj_w, intra_v.adj_h,
          std::span<const std::int32_t>(intra_v.levels.data(),
                                        intra_v.levels.size()),
          intra_v.eob};
      if (!av2enc::m2::rdo::intra_coefficient_bits(
              std::span<const av2enc::m2::rdo::IntraPlaneRateInput>(planes),
              &intra_coeff_bits, &error)) {
        return false;
      }
      have_intra = true;
    }

    std::vector<av2enc::m2::rdo::InterCandidate> candidates;
    candidates.reserve(stores.size() + 1);
    for (const CandidateStore& store : stores) {
      candidates.push_back(store.scored.candidate);
    }
    int intra_index = -1;
    if (have_intra) {
      intra_index = static_cast<int>(candidates.size());
      double intra_selector_distortion = intra_distortion;
      if (rdo_config.metric != av2enc::rdo::DistortionMetric::kSsd) {
        intra_selector_distortion = static_cast<double>(
            av2enc::rdo::compute_distortion(
                rdo_config.metric,
                std::span<const std::uint8_t>(source_y.data(),
                                              source_y.size()),
                std::span<const std::uint8_t>(intra_y.recon.data(),
                                              intra_y.recon.size()),
                w, h, w));
      }
      candidates.push_back(av2enc::m2::rdo::make_intra_candidate(
          intra_selector_distortion, intra_coeff_bits, 0, 0, 0, rate_ctx,
          rdo_config, rate_model));
    }
    if (candidates.empty()) {
      error = "no inter or intra candidate";
      return false;
    }
    const av2enc::m2::rdo::InterDecision decision =
        av2enc::m2::rdo::decide_inter_candidates(
            std::span<const av2enc::m2::rdo::InterCandidate>(
                candidates.data(), candidates.size()),
            rdo_config, &error);
    if (!decision.found) return false;
    int best_index = decision.best_index;
    if (options_.force_intra && have_intra) best_index = intra_index;

    if (best_index == intra_index) {
      entry.is_inter = 0;
      entry.skip_flag = 0;
      entry.y_mode = 0;   // DC_PRED
      entry.y_angle_delta = 0;
      entry.uv_mode = 0;  // DC_PRED
      write_plane_block(recon_[0], x0, y0, w, h, intra_y.recon);
      write_plane_block(recon_[1], cx0, cy0, cw, ch, intra_u.recon);
      write_plane_block(recon_[2], cx0, cy0, cw, ch, intra_v.recon);
      entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
      const TxPlan* plans[3] = {&intra_y, &intra_u, &intra_v};
      for (int plane = 0; plane < 3; ++plane) {
        entry.tx_size[plane] =
            static_cast<std::uint8_t>(plans[plane]->tx_size);
        entry.eob[plane] =
            static_cast<std::uint32_t>(plans[plane]->eob);
        const int area = plans[plane]->adj_w * plans[plane]->adj_h;
        std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
        syntax::build_scan(plans[plane]->adj_w, plans[plane]->adj_h, scan);
        for (int c2 = 0; c2 < plans[plane]->eob; ++c2) {
          levels_.push_back(plans[plane]->levels[scan[
              static_cast<std::size_t>(c2)]]);
        }
      }
      update_mv_grid(r, c, w4, h4, 0, inter::mc::Mv{0, 0});
      ++intra_blocks_;
      if (trace_rdo) {
        std::cerr << "m2 rdo intra r=" << r << " c=" << c << " " << w << "x"
                  << h << " dist=" << intra_distortion
                  << " bits=" << candidates[static_cast<std::size_t>(
                                      intra_index)].rate_bits
                  << " cost=" << decision.cost << "\n";
      }
      return true;
    }

    CandidateStore& chosen_store =
        stores[static_cast<std::size_t>(best_index)];
    const av2enc::m2::rdo::InterCandidate& chosen =
        chosen_store.scored.candidate;
    entry.is_inter = 1;
    entry.skip_flag = static_cast<std::uint8_t>(chosen.skip_flag ? 1 : 0);
    entry.inter_mode = static_cast<std::uint8_t>(chosen.inter_mode);
    entry.drl_index = static_cast<std::uint8_t>(chosen.drl_index);
    entry.mv_y = chosen.mv.y;
    entry.mv_x = chosen.mv.x;
    entry.mvd_y = chosen.mvd.y;
    entry.mvd_x = chosen.mvd.x;

    if (entry.skip_flag) {
      write_plane_block(recon_[0], x0, y0, w, h, chosen_store.y);
      write_plane_block(recon_[1], cx0, cy0, cw, ch, chosen_store.u);
      write_plane_block(recon_[2], cx0, cy0, cw, ch, chosen_store.v);
    } else {
      const av2enc::m2::residual::BlockDecision& residual =
          chosen_store.scored.residual;
      write_plane_block(recon_[0], x0, y0, w, h, residual.plane[0].recon);
      write_plane_block(recon_[1], cx0, cy0, cw, ch,
                        residual.plane[1].recon);
      write_plane_block(recon_[2], cx0, cy0, cw, ch,
                        residual.plane[2].recon);
      entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
      for (int plane = 0; plane < 3; ++plane) {
        entry.tx_size[plane] = static_cast<std::uint8_t>(
            residual.plane[plane].tx_size);
        entry.eob[plane] =
            static_cast<std::uint32_t>(residual.plane[plane].eob);
        av2enc::m2::residual::append_scan_levels(residual.plane[plane],
                                                 &levels_);
      }
    }
    update_mv_grid(r, c, w4, h4, 1, chosen.mv);
    ++inter_blocks_;
    if (trace_rdo) {
      std::cerr << "m2 rdo inter r=" << r << " c=" << c << " " << w << "x"
                << h << " kind=" << static_cast<int>(chosen.kind)
                << " mode=" << chosen.inter_mode
                << " drl=" << chosen.drl_index << " mv=(" << chosen.mv.y
                << "," << chosen.mv.x << ") skip="
                << static_cast<int>(entry.skip_flag)
                << " dist=" << chosen.distortion
                << " bits=" << chosen.rate_bits
                << " cost=" << decision.cost << "\n";
    }
    return true;
  }

  // Rate-aware partition decision.  The leaf distortion is the same source
  // variance the baseline uses; the rate term is the M1 leaf-rate proxy.  A
  // production integration with a real per-node inter cost should return that
  // cost from the callback instead.
  partition::FrameDecision rdo_partition_frame(
      const partition::DecisionOptions& poptions, std::string& error) {
    const av2enc::m2::rdo::RdoConfig config = m2_rdo_config_from_env();
    av2enc::m2::rdo::PartitionRdoConfig pconfig;
    pconfig.rdo = config;
    const auto leaf_costs =
        [&](const partition::PartitionNodeInfo& info)
        -> av2enc::m2::rdo::LeafCost {
      av2enc::m2::rdo::LeafCost cost;
      cost.distortion = static_cast<double>(block_variance_cost_at(info));
      const int bw = partition::block_width(info.size);
      const int bh = partition::block_height(info.size);
      cost.rate_bits = av2enc::m2::rdo::variance_leaf_rate_bits(
          cost.distortion, bw, bh, config.lambda.quant_step);
      return cost;
    };
    return av2enc::m2::rdo::decide_partition_rdo(
        geometry_, poptions, leaf_costs, pconfig, &error);
  }
'''

COLLECT_OLD = '''    if (node.partition == PartitionType::kNone) {
      if (!encode_leaf(node, index, error)) return false;
    } else {
'''

COLLECT_NEW = '''    if (node.partition == PartitionType::kNone) {
      if (m2_rdo_enabled() && !keyframe_) {
        if (!encode_leaf_rdo(node, index, error)) return false;
      } else if (!encode_leaf(node, index, error)) {
        return false;
      }
    } else {
'''

PARTITION_OLD = '''    const partition::FrameDecision frame =
        partition::decide_frame(geometry_, cost_fn, poptions);
    if (frame.status != partition::DecisionStatus::kOk) {
'''

PARTITION_NEW = '''    partition::FrameDecision frame;
    if (m2_rdo_partition_enabled() && !keyframe_) {
      frame = rdo_partition_frame(poptions, error);
    } else {
      frame = partition::decide_frame(geometry_, cost_fn, poptions);
    }
    if (frame.status != partition::DecisionStatus::kOk) {
'''


def replace_once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"patch: expected one {what} anchor, found {count}")
    return text.replace(old, new, 1)


def patch(source: str) -> str:
    if "encode_leaf_rdo" in source:
        raise SystemExit("patch: source already contains the M2 RDO hook")
    source = replace_once(source, INCLUDE_ANCHOR,
                          INCLUDE_ANCHOR + RDO_INCLUDES, "include")
    source = replace_once(source, "using partition::PartitionType;\n",
                          "using partition::PartitionType;\n" + HELPERS,
                          "helper insertion")
    source = replace_once(source, "  void update_mv_grid(",
                          RDO_FUNCTIONS + "\n  void update_mv_grid(",
                          "RDO function insertion")
    source = replace_once(source, COLLECT_OLD, COLLECT_NEW,
                          "collect_node dispatch")
    source = replace_once(source, PARTITION_OLD, PARTITION_NEW,
                          "partition dispatch")
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[5]
    source_path = args.input or (root / "src/encoder/m2/m2_frame.cpp")
    source = source_path.read_text()
    patched = patch(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(patched)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
