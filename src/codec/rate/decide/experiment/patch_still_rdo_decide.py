#!/usr/bin/env python3
"""Generate an experiment-only patched copy of src/encoder/m1/still_encoder.cpp.

The repository file is never modified.  The generated copy adds env-gated hooks
for the M1 RDO decision layer:

    AV2ENC_RDO_DECIDE=1         enable the layer
    AV2ENC_RDO_SCALE=<x>        sse_scale (default 1.0)
    AV2ENC_RDO_SSE_ONLY=1       rate_aware = false (control)
    AV2ENC_RDO_NO_INTRA=1       keep the legacy luma mode search
    AV2ENC_RDO_NO_PART=1        keep the legacy partition costs
    AV2ENC_RDO_NO_CHROMA=1      keep the legacy chroma decision

With AV2ENC_RDO_DECIDE unset the patched binary must be byte-identical to the
unpatched CLI; run_still_rdo_decide.sh checks that.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

INCLUDE_ANCHOR = '#include "codec/transform/variants/tx_variants.h"\n'

HELPERS = r'''
// ---------------------------------------------------------------------------
// Experiment-only RDO decision hooks.  Kept in this generated copy; the
// production integration should call the same API from choose_luma_mode().
// ---------------------------------------------------------------------------
bool rdo_decide_enabled() {
  return std::getenv("AV2ENC_RDO_DECIDE") != nullptr;
}

av2enc::still::rdo_decide::LambdaConfig rdo_lambda_from_env() {
  av2enc::still::rdo_decide::LambdaConfig config;
  if (const char* scale = std::getenv("AV2ENC_RDO_SCALE")) {
    const double parsed = std::atof(scale);
    if (parsed > 0.0) config.sse_scale = parsed;
  }
  if (std::getenv("AV2ENC_RDO_SSE_ONLY") != nullptr) {
    config.rate_aware = false;
  }
  return config;
}

'''

LUMA_FUNCTION = r'''  bool choose_luma_mode(int x0, int y0, int w, int h, ModeChoice& best) {
    const char* forced_mode = std::getenv("AV2ENC_STILL_FORCE_Y");
    if (x0 > 0) {
      if (const char* second = std::getenv("AV2ENC_STILL_FORCE_Y2")) {
        forced_mode = second;
      }
    }
    if (forced_mode != nullptr) {
      int mode = 0;
      int delta = 0;
      if (std::sscanf(forced_mode, "%d:%d", &mode, &delta) == 2) {
        best.mode = static_cast<intra::Mode>(mode);
        best.angle_delta = delta;
        best.av2_mode = mode;
        best.sse = 0;
        return true;
      }
    }
    if (options_.luma_dc_only || options_.force_dc ||
        std::getenv("AV2ENC_STILL_LUMA_DC") != nullptr) {
      best.mode = intra::Mode::kDc;
      best.angle_delta = 0;
      best.av2_mode = kDcPred;
      best.sse = 0;
      return true;
    }
    intra::IntraEdges edges;
    if (!prepare_edges(0, x0, y0, w, h, edges)) {
      error_ = "luma prepare_edges failed";
      return false;
    }
    bool above_smooth = false;
    bool left_smooth = false;
    neighborhood_smooth(0, y0 >> 2, x0 >> 2, w >> 2, h >> 2, above_smooth,
                        left_smooth);
    std::array<intra::ModeCandidate, 61> candidates;
    const int count = intra::build_mode_candidates(candidates, true);
    if (count <= 0) {
      error_ = "build_mode_candidates failed";
      return false;
    }
    std::vector<std::uint8_t> pred;
    pred.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    const bool base_only = std::getenv("AV2ENC_STILL_BASE_ONLY") != nullptr;
    const bool no_directional = std::getenv("AV2ENC_STILL_NO_DIR") != nullptr;
    const bool rdo_intra =
        rdo_decide_enabled() && std::getenv("AV2ENC_RDO_NO_INTRA") == nullptr;
    int rdo_intra_count = 0;
    best.sse = std::numeric_limits<std::int64_t>::max();
    for (int i = 0; i < count; ++i) {
      intra::PredictParams params;
      params.mode = candidates[static_cast<std::size_t>(i)].mode;
      params.angle_delta = candidates[static_cast<std::size_t>(i)].angle_delta;
      if (base_only && params.angle_delta != 0) continue;
      if (no_directional && intra::is_directional(params.mode)) continue;
      params.enable_edge_filter = edge_filter_enabled();
      params.above_neighbor_is_smooth = above_smooth;
      params.left_neighbor_is_smooth = left_smooth;
      if (intra::predict_intra(edges, params, pred.data(),
                               static_cast<std::size_t>(w)) !=
          intra::IntraStatus::kOk) {
        error_ = "luma predict_intra failed";
        return false;
      }
      const std::int64_t sse = sse_against_source(0, x0, y0, w, h, pred);
      if (sse < best.sse) {
        best.sse = sse;
        best.mode = params.mode;
        best.angle_delta = params.angle_delta;
        best.av2_mode = av2_y_mode(params.mode);
      }
      if (rdo_intra &&
          rdo_intra_count <
              static_cast<int>(rdo_intra_candidates_.size())) {
        TxPlan probe_plan;
        if (!compute_tx(0, x0, y0, w, h, pred, probe_plan)) {
          error_ = "RDO intra probe compute_tx failed";
          return false;
        }
        const int area = probe_plan.adj_w * probe_plan.adj_h;
        if (area <= static_cast<int>(rdo_levels_.size())) {
          syntax::build_scan(
              probe_plan.adj_w, probe_plan.adj_h,
              std::span<std::uint16_t>(
                  rdo_scan_.data(), static_cast<std::size_t>(area)));
          for (int c = 0; c < area; ++c) {
            rdo_levels_[static_cast<std::size_t>(c)] =
                probe_plan.levels[rdo_scan_[static_cast<std::size_t>(c)]];
          }
          av2enc::still::rdo_decide::TransformStreamBlock block;
          block.plane = 0;
          block.tx_size = probe_plan.tx_size;
          block.w = probe_plan.adj_w;
          block.h = probe_plan.adj_h;
          block.levels_scan = std::span<const std::int32_t>(
              rdo_levels_.data(), static_cast<std::size_t>(area));
          block.eob = probe_plan.eob;
          if (!rdo_stream_.init(block)) {
            error_ = "RDO intra stream init failed";
            return false;
          }
          av2enc::rdo::IntraCandidateCost& candidate =
              rdo_intra_candidates_[static_cast<std::size_t>(rdo_intra_count)];
          candidate.mode = av2_y_mode(params.mode);
          candidate.angle_delta = params.angle_delta;
          candidate.sse = static_cast<double>(sse);
          candidate.syntax_rate_bits =
              av2enc::still::rdo_decide::intra_syntax_bits(candidate.mode,
                                                        candidate.angle_delta);
          candidate.coefficient_rate_bits =
              av2enc::still::rdo_decide::intra_coefficient_bits(rdo_stream_);
          ++rdo_intra_count;
        }
      }
    }
    if (rdo_intra && rdo_intra_count > 0) {
      const auto rdo_config = rdo_lambda_from_env();
      const auto choice = av2enc::still::rdo_decide::decide_luma_intra(
          std::span<const av2enc::rdo::IntraCandidateCost>(
              rdo_intra_candidates_.data(),
              static_cast<std::size_t>(rdo_intra_count)),
          {}, rdo_config);
      if (choice.found) {
        best.mode = static_cast<intra::Mode>(choice.mode);
        best.angle_delta = choice.angle_delta;
        best.av2_mode = choice.mode;
        best.sse = static_cast<std::int64_t>(choice.sse);
      }
    }
    return true;
  }
'''

CHROMA_FUNCTION = r'''  bool choose_uv_mode(int cx0, int cy0, int cw, int ch, int y_mode,
                      int y_angle_delta, UvChoice& best) {
    best.is_cfl = false;
    best.sse = std::numeric_limits<std::int64_t>::max();
    const bool uv_full =
        options_.uv_full || std::getenv("AV2ENC_STILL_UV_FULL") != nullptr;
    if (std::getenv("AV2ENC_STILL_UV_DC") != nullptr) {
      best.mode = intra::Mode::kDc;
      best.angle_delta = 0;
      best.av2_mode = kDcPred;
      best.sse = 0;
      return true;
    }
    if (const char* forced = std::getenv("AV2ENC_STILL_FORCE_UV")) {
      int mode = 0;
      int delta = 0;
      if (std::sscanf(forced, "%d:%d", &mode, &delta) == 2 &&
          mode >= 0 && mode <= 12 && delta >= -3 && delta <= 3) {
        const int derived =
            (y_mode_is_directional(y_mode) && mode == y_mode)
                ? y_angle_delta
                : 0;
        if (delta != 0 && delta != derived) {
          error_ = "AV2ENC_STILL_FORCE_UV non-zero delta requires UVMode==YMode";
          return false;
        }
        best.mode = static_cast<intra::Mode>(mode);
        best.angle_delta = derived;
        best.av2_mode = av2_y_mode(best.mode);
        best.sse = 0;
        return true;
      }
      error_ = "invalid AV2ENC_STILL_FORCE_UV (expected mode:delta)";
      return false;
    }
    if (options_.force_dc || (!options_.enable_cfl && !uv_full)) {
      best.mode = intra::Mode::kDc;
      best.angle_delta = 0;
      best.av2_mode = kDcPred;
      best.sse = 0;
      return true;
    }
    std::vector<std::uint8_t> pred;
    const bool rdo_chroma = rdo_decide_enabled() &&
                            std::getenv("AV2ENC_RDO_NO_CHROMA") == nullptr;
    int rdo_chroma_count = 0;

    const auto add_chroma_candidate =
        [&](bool is_cfl, intra::Mode mode, int delta, std::int64_t sse_u,
            std::int64_t sse_v, const std::vector<std::uint8_t>& u_pred,
            const std::vector<std::uint8_t>& v_pred) -> bool {
      TxPlan u_plan;
      TxPlan v_plan;
      if (!compute_tx(1, cx0, cy0, cw, ch, u_pred, u_plan)) return false;
      if (!compute_tx(2, cx0, cy0, cw, ch, v_pred, v_plan)) return false;
      double coefficient_bits = 0.0;
      const TxPlan* plans[2] = {&u_plan, &v_plan};
      for (int plane = 1; plane <= 2; ++plane) {
        const TxPlan& plan = *plans[plane - 1];
        const int area = plan.adj_w * plan.adj_h;
        if (area > static_cast<int>(rdo_levels_.size())) return false;
        syntax::build_scan(
            plan.adj_w, plan.adj_h,
            std::span<std::uint16_t>(
                rdo_scan_.data(), static_cast<std::size_t>(area)));
        for (int c = 0; c < area; ++c) {
          rdo_levels_[static_cast<std::size_t>(c)] =
              plan.levels[rdo_scan_[static_cast<std::size_t>(c)]];
        }
        av2enc::still::rdo_decide::TransformStreamBlock block;
        block.plane = plane;
        block.tx_size = plan.tx_size;
        block.w = plan.adj_w;
        block.h = plan.adj_h;
        block.levels_scan = std::span<const std::int32_t>(
            rdo_levels_.data(), static_cast<std::size_t>(area));
        block.eob = plan.eob;
        if (!rdo_stream_.init(block)) return false;
        coefficient_bits += rdo_stream_.block_bits();
      }
      av2enc::still::rdo_decide::ChromaCandidateCost& candidate =
          rdo_chroma_candidates_[static_cast<std::size_t>(rdo_chroma_count)];
      candidate.is_cfl = is_cfl;
      candidate.mode = av2_y_mode(mode);
      candidate.angle_delta = delta;
      candidate.block_w = cw;
      candidate.block_h = ch;
      candidate.sse_u = static_cast<double>(sse_u);
      candidate.sse_v = static_cast<double>(sse_v);
      candidate.syntax_rate_bits =
          av2enc::still::rdo_decide::default_chroma_syntax_bits(
              is_cfl, candidate.mode, delta);
      candidate.coefficient_rate_bits = coefficient_bits;
      ++rdo_chroma_count;
      return true;
    };

    static const intra::Mode kBase[13] = {
        intra::Mode::kDc,      intra::Mode::kV,       intra::Mode::kH,
        intra::Mode::kD45,     intra::Mode::kD135,    intra::Mode::kD113,
        intra::Mode::kD157,    intra::Mode::kD203,    intra::Mode::kD67,
        intra::Mode::kSmooth,  intra::Mode::kSmoothV, intra::Mode::kSmoothH,
        intra::Mode::kPaeth};

    for (int i = 0; i < 13; ++i) {
      const intra::Mode mode = kBase[i];
      const int delta =
          (y_mode_is_directional(y_mode) &&
           static_cast<int>(mode) == y_mode)
              ? y_angle_delta
              : 0;
      if (!uv_full && mode != intra::Mode::kDc) continue;
      if (!predict_block(1, cx0, cy0, cw, ch, mode, delta, pred)) return false;
      const std::int64_t sse_u = sse_against_source(1, cx0, cy0, cw, ch, pred);
      std::int64_t sse_v = 0;
      if (uv_full) {
        if (!predict_block(2, cx0, cy0, cw, ch, mode, delta, pred)) return false;
        sse_v = sse_against_source(2, cx0, cy0, cw, ch, pred);
      }
      const std::int64_t sse = sse_u + sse_v;
      if (sse < best.sse) {
        best.sse = sse;
        best.mode = mode;
        best.angle_delta = delta;
        best.av2_mode = av2_y_mode(mode);
      }
      if (rdo_chroma &&
          rdo_chroma_count <
              static_cast<int>(rdo_chroma_candidates_.size())) {
        std::vector<std::uint8_t> u_pred;
        std::vector<std::uint8_t> v_pred;
        if (!predict_block(1, cx0, cy0, cw, ch, mode, delta, u_pred)) {
          return false;
        }
        if (!predict_block(2, cx0, cy0, cw, ch, mode, delta, v_pred)) {
          return false;
        }
        const std::int64_t rdo_sse_u =
            sse_against_source(1, cx0, cy0, cw, ch, u_pred);
        const std::int64_t rdo_sse_v =
            sse_against_source(2, cx0, cy0, cw, ch, v_pred);
        if (!add_chroma_candidate(false, mode, delta, rdo_sse_u, rdo_sse_v,
                                  u_pred, v_pred)) {
          error_ = "RDO chroma candidate failed";
          return false;
        }
      }
    }

    if (options_.enable_cfl) {
      std::vector<std::uint8_t> u_dc;
      std::vector<std::uint8_t> v_dc;
      if (!predict_block(1, cx0, cy0, cw, ch, intra::Mode::kDc, 0, u_dc)) {
        return false;
      }
      if (!predict_block(2, cx0, cy0, cw, ch, intra::Mode::kDc, 0, v_dc)) {
        return false;
      }
      for (int yy = 0; yy < ch; ++yy) {
        for (int xx = 0; xx < cw; ++xx) {
          recon_[1].at(cx0 + xx, cy0 + yy) =
              u_dc[static_cast<std::size_t>(yy) * cw + xx];
          recon_[2].at(cx0 + xx, cy0 + yy) =
              v_dc[static_cast<std::size_t>(yy) * cw + xx];
        }
      }
      cfl::Scratch scratch;
      cfl::Block ub;
      cfl::Block vb;
      make_cfl_block(1, cx0, cy0, cw, ch, ub);
      make_cfl_block(2, cx0, cy0, cw, ch, vb);
      cfl::compute_luma(ub, scratch);
      const int alpha_u = cfl::derive_implicit_alpha(ub, scratch);
      cfl::compute_luma(vb, scratch);
      const int alpha_v = cfl::derive_implicit_alpha(vb, scratch);
      cfl::apply_prediction(ub, alpha_u, scratch);
      cfl::apply_prediction(vb, alpha_v, scratch);
      std::int64_t sse_u = 0;
      std::int64_t sse_v = 0;
      for (int yy = 0; yy < ch; ++yy) {
        for (int xx = 0; xx < cw; ++xx) {
          const int su = source_sample(1, cx0 + xx, cy0 + yy);
          const int pu = recon_[1].at(cx0 + xx, cy0 + yy);
          const int du = su - pu;
          sse_u += static_cast<std::int64_t>(du) * du;
          const int sv = source_sample(2, cx0 + xx, cy0 + yy);
          const int pv = recon_[2].at(cx0 + xx, cy0 + yy);
          const int dv = sv - pv;
          sse_v += static_cast<std::int64_t>(dv) * dv;
        }
      }
      const std::int64_t sse = sse_u + sse_v;
      if (sse < best.sse) {
        best.sse = sse;
        best.is_cfl = true;
        best.mode = intra::Mode::kDc;
        best.angle_delta = 0;
        best.av2_mode = kDcPred;
      }
      if (rdo_chroma &&
          rdo_chroma_count <
              static_cast<int>(rdo_chroma_candidates_.size())) {
        std::vector<std::uint8_t> u_pred;
        std::vector<std::uint8_t> v_pred;
        u_pred.resize(static_cast<std::size_t>(cw) * ch);
        v_pred.resize(static_cast<std::size_t>(cw) * ch);
        for (int yy = 0; yy < ch; ++yy) {
          for (int xx = 0; xx < cw; ++xx) {
            u_pred[static_cast<std::size_t>(yy) * cw + xx] =
                recon_[1].at(cx0 + xx, cy0 + yy);
            v_pred[static_cast<std::size_t>(yy) * cw + xx] =
                recon_[2].at(cx0 + xx, cy0 + yy);
          }
        }
        if (!add_chroma_candidate(true, intra::Mode::kDc, 0, sse_u, sse_v,
                                  u_pred, v_pred)) {
          error_ = "RDO CfL candidate failed";
          return false;
        }
      }
    }

    if (rdo_chroma && rdo_chroma_count > 0) {
      const auto rdo_config = rdo_lambda_from_env();
      const auto choice = av2enc::still::rdo_decide::decide_chroma(
          std::span<const av2enc::still::rdo_decide::ChromaCandidateCost>(
              rdo_chroma_candidates_.data(),
              static_cast<std::size_t>(rdo_chroma_count)),
          {}, rdo_config);
      if (choice.found) {
        best.is_cfl = choice.is_cfl;
        best.mode = static_cast<intra::Mode>(choice.mode);
        best.angle_delta = choice.angle_delta;
        best.av2_mode = choice.mode;
        best.sse = static_cast<std::int64_t>(choice.sse_total);
      }
    }
    return true;
  }
'''

PARTITION_OLD = r'''    const double lambda = 0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
    const double signal_cost = lambda * 2.0;

    const auto cost_fn = [&](const partition::PartitionNodeInfo& info) {
      partition::NodeCosts costs;
      costs.leaf = static_cast<double>(
          block_variance_cost_at(info.mi_row, info.mi_col, info.size));
      if (std::getenv("AV2ENC_STILL_SQUARE_ONLY") != nullptr &&
          partition::block_width(info.size) !=
              partition::block_height(info.size)) {
        costs.leaf += 1.0e18;
      }
      if (std::getenv("AV2ENC_STILL_LEAF32") != nullptr &&
          (partition::block_width(info.size) > 32 ||
           partition::block_height(info.size) > 32)) {
        costs.leaf += 1.0e18;
      }
      costs.none_signal = 0.0;
      costs.split_signal = signal_cost;
      costs.horz_signal = signal_cost;
      costs.vert_signal = signal_cost;
      return costs;
    };

    const partition::FrameDecision frame =
        partition::decide_frame(geometry, cost_fn, poptions);
'''

PARTITION_NEW = r'''    const double lambda = 0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
    const double signal_cost = lambda * 2.0;

    const auto legacy_cost_fn = [&](const partition::PartitionNodeInfo& info) {
      partition::NodeCosts costs;
      costs.leaf = static_cast<double>(
          block_variance_cost_at(info.mi_row, info.mi_col, info.size));
      if (std::getenv("AV2ENC_STILL_SQUARE_ONLY") != nullptr &&
          partition::block_width(info.size) !=
              partition::block_height(info.size)) {
        costs.leaf += 1.0e18;
      }
      if (std::getenv("AV2ENC_STILL_LEAF32") != nullptr &&
          (partition::block_width(info.size) > 32 ||
           partition::block_height(info.size) > 32)) {
        costs.leaf += 1.0e18;
      }
      costs.none_signal = 0.0;
      costs.split_signal = signal_cost;
      costs.horz_signal = signal_cost;
      costs.vert_signal = signal_cost;
      return costs;
    };

    const bool rdo_partition =
        rdo_decide_enabled() && std::getenv("AV2ENC_RDO_NO_PART") == nullptr;
    partition::FrameDecision frame;
    if (rdo_partition) {
      const auto raw_cost_fn = [&](const partition::PartitionNodeInfo& info) {
        av2enc::still::rdo_decide::PartitionCostInputs raw;
        raw.leaf_distortion = static_cast<double>(
            block_variance_cost_at(info.mi_row, info.mi_col, info.size));
        if (std::getenv("AV2ENC_STILL_SQUARE_ONLY") != nullptr &&
            partition::block_width(info.size) !=
                partition::block_height(info.size)) {
          raw.leaf_distortion += 1.0e18;
        }
        if (std::getenv("AV2ENC_STILL_LEAF32") != nullptr &&
            (partition::block_width(info.size) > 32 ||
             partition::block_height(info.size) > 32)) {
          raw.leaf_distortion += 1.0e18;
        }
        raw.leaf_rate_bits = av2enc::still::rdo_decide::estimate_leaf_rate_bits(
            raw.leaf_distortion, partition::block_width(info.size),
            partition::block_height(info.size),
            av2enc::still::rdo_decide::kM1Qp90QuantStep);
        raw.none_signal_bits = 0.0;
        raw.split_signal_bits = 2.0;
        raw.horz_signal_bits = 2.0;
        raw.vert_signal_bits = 2.0;
        return raw;
      };
      av2enc::still::rdo_decide::PartitionDecideConfig pconfig;
      const auto rdo_lambda = rdo_lambda_from_env();
      pconfig.lambda = av2enc::still::rdo_decide::lambda_sse(rdo_lambda);
      pconfig.rate_aware = rdo_lambda.rate_aware;
      // Control: same lambda, no leaf-rate term.  Charging two split bits
      // keeps the signal cost comparable to the rate-aware path.
      pconfig.sse_only_split_cost = 2.0 * pconfig.lambda;
      std::string partition_error;
      auto rdo_frame = av2enc::still::rdo_decide::decide_frame_rdo(
          geometry, raw_cost_fn, poptions, pconfig, &partition_error);
      if (rdo_frame.status != partition::DecisionStatus::kOk) {
        error = "RDO partition decision failed: " + partition_error;
        return false;
      }
      frame = std::move(rdo_frame.frame);
    } else {
      frame = partition::decide_frame(geometry, legacy_cost_fn, poptions);
    }
'''

TX_TYPE_ANCHOR = "    if (choose_tx_type) {\n"

TX_TYPE_BRANCH = r'''    if (choose_tx_type && rdo_decide_enabled() &&
        std::getenv("AV2ENC_RDO_TXTYPE") != nullptr) {
      int mapped_mode = txtype::map_intra_mode(y_mode, y_angle_delta, cw, ch);
      if (mapped_mode < 0) mapped_mode = 0;
      const int tx_area = params_opt->w * params_opt->h;
      std::vector<std::int32_t> levels_scratch(
          static_cast<std::size_t>(tx_area));
      std::vector<std::int16_t> recon_scratch(
          static_cast<std::size_t>(coded_pels));
      std::vector<std::int32_t> levels_out(static_cast<std::size_t>(tx_area));
      std::vector<std::int16_t> recon_out(
          static_cast<std::size_t>(coded_pels));
      av2enc::still::rdo_decide::TransformStreamRate stream;
      av2enc::still::rdo_decide::TxTypeDecisionConfig tx_config;
      const auto rdo_lambda = rdo_lambda_from_env();
      tx_config.lambda = av2enc::still::rdo_decide::lambda_sse(rdo_lambda);
      tx_config.rate_aware = rdo_lambda.rate_aware;
      av2enc::still::rdo_decide::TxTypeDecision decision;
      std::string tx_error;
      if (!av2enc::still::rdo_decide::decide_tx_type(
              residual_coded.data(), cw, ch, mapped_mode, 0, false, tx_config,
              levels_scratch, recon_scratch, levels_out, recon_out, stream,
              decision, &tx_error)) {
        error_ = "rdo_decide tx_type failed: " + tx_error;
        return false;
      }
      tx_type = decision.candidate.type;
      if (const char* forced_type = std::getenv("AV2ENC_RDO_TXTYPE_FORCE")) {
        const int parsed_type = std::atoi(forced_type);
        if (parsed_type >= 0 && parsed_type <= 15) {
          tx_type = static_cast<txv::TxType>(parsed_type);
        }
      }
      if (std::getenv("AV2ENC_RDO_TXTYPE_DCT") != nullptr) {
        tx_type = txv::TxType::kDctDct;
      }
      if (std::getenv("AV2ENC_STILL_TXTYPE_TRACE") != nullptr) {
        std::fprintf(stderr,
                     "m1 txtype blk=%dx%d mode=%d delta=%d mapped=%d type=%d\n",
                     cw, ch, y_mode, y_angle_delta, mapped_mode,
                     static_cast<int>(tx_type));
      }
      const auto chosen = txv::make_tx_params(cw, ch, tx_type);
      if (!chosen.has_value()) {
        error_ = "unsupported transform size for the chosen tx_type";
        return false;
      }
      params_opt = chosen;
    } else if (choose_tx_type) {
'''

RDOQ_ANCHOR = "      if (options_.rdoq) {\n"

RDOQ_BRANCH = r'''      if (options_.rdoq && rdo_decide_enabled() &&
          std::getenv("AV2ENC_RDO_RDOQ") != nullptr) {
        std::vector<std::int32_t> scalar_scan(static_cast<std::size_t>(area));
        std::vector<double> original_scan(static_cast<std::size_t>(area));
        for (int c = 0; c < area; ++c) {
          const int pos = scan[static_cast<std::size_t>(c)];
          scalar_scan[static_cast<std::size_t>(c)] = plan.levels[pos];
          original_scan[static_cast<std::size_t>(c)] = coeff[pos];
        }
        av2enc::still::rdo_decide::RdoqConfig rdoq_config;
        rdoq_config.quant_step = 1.0 / p.quant_factor;
        // RDOQ lambda lives in the forward-coefficient domain.  The repository
        // RDOQ bring-up uses sse_scale 0.125 here; keep that separate from the
        // pixel-domain decision scale.
        double rdoq_sse_scale = 0.125;
        if (const char* rdoq_scale = std::getenv("AV2ENC_RDO_RDOQ_SCALE")) {
          const double parsed = std::atof(rdoq_scale);
          if (parsed > 0.0) rdoq_sse_scale = parsed;
        }
        rdoq_config.lambda = av2enc::rdo::lambda_sse_from_quant_step(
            rdoq_config.quant_step, rdoq_sse_scale);
        av2enc::still::rdo_decide::RdoqWorkspace rdoq_workspace;
        av2enc::still::rdo_decide::TransformStreamRate rdoq_stream;
        std::vector<std::int32_t> rdoq_levels(static_cast<std::size_t>(area));
        av2enc::still::rdo_decide::RdoqDecision rdoq_decision;
        std::string rdoq_error;
        if (!av2enc::still::rdo_decide::decide_rdoq(
                original_scan, scalar_scan, plane, p.tx_size, p.w, p.h, 0, 0,
                rdoq_config, rdoq_workspace, rdoq_stream, rdoq_levels,
                rdoq_decision, &rdoq_error)) {
          error_ = "rdo_decide rdoq failed: " + rdoq_error;
          return false;
        }
        std::fill(plan.levels.begin(), plan.levels.end(), 0);
        for (int c = 0; c < rdoq_decision.eob; ++c) {
          plan.levels[scan[static_cast<std::size_t>(c)]] =
              rdoq_levels[static_cast<std::size_t>(c)];
        }
      } else if (options_.rdoq) {
'''

MEMBERS_ANCHOR = "  std::string error_;\n"
MEMBERS = r'''  std::string error_;
  // Experiment-only RDO decision scratch.  One object per encoder, reused for
  // every block; no per-block allocation in the decision layer itself.
  av2enc::still::rdo_decide::TransformStreamRate rdo_stream_;
  std::array<std::int32_t, 1024> rdo_levels_{};
  std::array<std::uint16_t, 1024> rdo_scan_{};
  std::array<av2enc::rdo::IntraCandidateCost, 61> rdo_intra_candidates_{};
  std::array<av2enc::still::rdo_decide::ChromaCandidateCost, 14>
      rdo_chroma_candidates_{};
'''


def replace_once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"patch: expected one {what} anchor, found {count}")
    return text.replace(old, new, 1)


def patch(source: str) -> str:
    if "rdo_decide" in source:
        raise SystemExit("patch: source already contains the RDO hook")
    source = replace_once(
        source,
        INCLUDE_ANCHOR,
        INCLUDE_ANCHOR
        + '\n// Experiment-only RDO decision hook; the production M1 owner should\n'
        + "// call the same API from choose_luma_mode() instead of patching a copy.\n"
        + '#include "codec/rate/decide/rdo_decide.h"\n',
        "include",
    )
    source = replace_once(
        source,
        "using partition::PartitionType;\n",
        "using partition::PartitionType;\n" + HELPERS,
        "helper insertion",
    )
    source = replace_once(
        source, MEMBERS_ANCHOR, MEMBERS, "member insertion"
    )
    source = replace_once(source, TX_TYPE_ANCHOR, TX_TYPE_BRANCH,
                          "TX-type branch")
    source = replace_once(source, RDOQ_ANCHOR, RDOQ_BRANCH, "RDOQ branch")

    start = source.index("  bool choose_luma_mode(")
    end = source.index("  std::int64_t block_variance_cost_at(")
    source = source[:start] + LUMA_FUNCTION + "\n" + source[end:]

    start = source.index("  bool choose_uv_mode(")
    end = source.index("  void fill_deblock_mi(")
    source = source[:start] + CHROMA_FUNCTION + "\n" + source[end:]

    source = replace_once(source, PARTITION_OLD, PARTITION_NEW, "partition block")
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[5]
    source_path = args.input or (root / "src/pipeline/still/still_encoder.cpp")
    source = source_path.read_text()
    patched = patch(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(patched)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
