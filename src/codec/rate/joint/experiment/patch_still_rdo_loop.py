#!/usr/bin/env python3
"""Generate an experiment-only M1 copy that drives the unified RDO loop.

The repository file src/pipeline/still/still_encoder.cpp is never modified.  This script
writes a patched copy under build-rdo-loop/ with three small, env-gated branches:

    AV2ENC_RDO_LOOP=1            enable the unified loop
    AV2ENC_RDO_LOOP_SCALE=<x>    pixel-domain sse_scale (default 0.5)
    AV2ENC_RDO_LOOP_CTU=<x>      CTU activity strength (default 0)
    AV2ENC_RDO_LOOP_MAX_LUMA=<n> luma candidate probes (default 6)
    AV2ENC_RDO_LOOP_MAX_CHROMA=<n> chroma candidate probes (default 6)
    AV2ENC_RDO_LOOP_MARGIN=<x>   SSE prune margin (default 1.35)
    AV2ENC_RDO_LOOP_PROBES=<n>   per-SB partition probe cap (default 0)
    AV2ENC_RDO_LOOP_TRACE=<path> write the per-node cost CSV
    AV2ENC_RDO_LOOP_STATS=1      print aggregate loop stats to stderr

With AV2ENC_RDO_LOOP unset the patched binary must be byte-identical to the
unpatched CLI for the legacy `dc` and `sse` cases.  run_still_rdo_loop.sh checks
that.

The patched code calls the same public APIs the production integration should
call: rdo_loop for policy/cost, rdo_decide for the luma/chroma/TX/RDOQ
decisions, and partition/partition_ext for geometry.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

INCLUDE_ANCHOR = '#include "codec/transform/variants/tx_variants.h"\n'
INCLUDE_ADD = r'''
#include "codec/rate/decide/rdo_decide.h"
#include "codec/rate/joint/rdo_loop.h"
#include <fstream>
#include <span>
'''

HELPERS = r'''
// ---------------------------------------------------------------------------
// Experiment-only unified RDO loop hooks.  Kept in this generated copy; the
// production integration should call the same rdo_loop APIs from the encoder.
// ---------------------------------------------------------------------------
bool rdo_loop_enabled() {
  return std::getenv("AV2ENC_RDO_LOOP") != nullptr;
}

double rdo_loop_env_double(const char* name, double fallback) {
  if (const char* value = std::getenv(name)) {
    const double parsed = std::atof(value);
    if (std::isfinite(parsed) && parsed >= 0.0) return parsed;
  }
  return fallback;
}

int rdo_loop_env_int(const char* name, int fallback) {
  if (const char* value = std::getenv(name)) {
    const int parsed = std::atoi(value);
    if (parsed >= 0) return parsed;
  }
  return fallback;
}

av2enc::still::rdo_loop::LambdaConfig rdo_loop_lambda_from_env() {
  av2enc::still::rdo_loop::LambdaConfig config;
  config.sse_scale =
      rdo_loop_env_double("AV2ENC_RDO_LOOP_SCALE", config.sse_scale);
  config.ctu_activity_strength = rdo_loop_env_double(
      "AV2ENC_RDO_LOOP_CTU", config.ctu_activity_strength);
  return config;
}

// One source rect.  Coordinates are samples in the plane's padded layout.
struct RdoRect {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

// One saved encoder state.  The search recurses, so snapshots form a stack:
// an inner trial must not overwrite the outer trial's saved state.
RdoRect rdo_clamp_rect(RdoRect rect, int max_w, int max_h) {
  if (rect.x < 0) {
    rect.w += rect.x;
    rect.x = 0;
  }
  if (rect.y < 0) {
    rect.h += rect.y;
    rect.y = 0;
  }
  if (rect.x + rect.w > max_w) rect.w = max_w - rect.x;
  if (rect.y + rect.h > max_h) rect.h = max_h - rect.y;
  if (rect.w < 0) rect.w = 0;
  if (rect.h < 0) rect.h = 0;
  return rect;
}

struct RdoSnapshot {
  RdoRect rect_y;
  RdoRect rect_chroma;
  RdoRect rect_joint;
  RdoRect rect_uv;
  RdoRect rect_decoded[3];
  RdoRect rect_deblock;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  std::vector<int> joint;
  std::vector<std::uint8_t> uv;
  std::array<std::vector<std::uint8_t>, 3> decoded;
  std::array<std::vector<deblock::DeblockMi>, 3> deblock;
  std::size_t blocks = 0;
  std::size_t levels = 0;
};

inline constexpr int kRdoSnapshotDepth = 16;
'''

# Inserted at the top of choose_luma_mode / choose_uv_mode / encode and in
# compute_tx.  The original bodies stay untouched, so the loop-off path is
# byte-identical.
LUMA_BRANCH = r'''    if (rdo_loop_enabled()) {
      return choose_luma_mode_loop(x0, y0, w, h, best);
    }
'''
CHROMA_BRANCH = r'''    if (rdo_loop_enabled()) {
      return choose_uv_mode_loop(cx0, cy0, cw, ch, y_mode, y_angle_delta, best);
    }
'''
ENCODE_BRANCH = r'''    if (rdo_loop_enabled()) {
      return encode_with_rdo_loop(stream, poptions, error);
    }
'''

TX_BRANCH = r'''    if (rdo_loop_enabled() && choose_tx_type) {
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
      tx_config.lambda = rdo_current_lambda_;
      tx_config.rate_aware = true;
      av2enc::still::rdo_decide::TxTypeDecision decision;
      std::string tx_error;
      if (!av2enc::still::rdo_decide::decide_tx_type(
              residual_coded.data(), cw, ch, mapped_mode, 0, false, tx_config,
              levels_scratch, recon_scratch, levels_out, recon_out, stream,
              decision, &tx_error)) {
        error_ = "rdo_loop tx_type failed: " + tx_error;
        return false;
      }
      tx_type = decision.candidate.type;
      const auto chosen = txv::make_tx_params(cw, ch, tx_type);
      if (!chosen.has_value()) {
        error_ = "rdo_loop: unsupported chosen tx_type";
        return false;
      }
      params_opt = chosen;
    } else if (choose_tx_type) {
'''

RDOQ_BRANCH = r'''      if (rdo_loop_enabled() && options_.rdoq) {
        std::vector<std::int32_t> scalar_scan(static_cast<std::size_t>(area));
        std::vector<double> original_scan(static_cast<std::size_t>(area));
        for (int c = 0; c < area; ++c) {
          const int pos = scan[static_cast<std::size_t>(c)];
          scalar_scan[static_cast<std::size_t>(c)] = plan.levels[pos];
          original_scan[static_cast<std::size_t>(c)] = coeff[pos];
        }
        av2enc::still::rdo_decide::RdoqConfig rdoq_config;
        rdoq_config.quant_step = 1.0 / p.quant_factor;
        const double rdoq_mult =
            rdo_loop_env_double("AV2ENC_RDO_LOOP_RDOQ_MULT", 1.0);
        rdoq_config.lambda =
            av2enc::still::rdo_decide::rdoq_lambda_from_pixel_lambda(
                rdo_current_lambda_) * rdoq_mult;
        av2enc::still::rdo_decide::RdoqWorkspace rdoq_workspace;
        av2enc::still::rdo_decide::TransformStreamRate rdoq_stream;
        std::vector<std::int32_t> rdoq_levels(static_cast<std::size_t>(area));
        av2enc::still::rdo_decide::RdoqDecision rdoq_decision;
        std::string rdoq_error;
        if (!av2enc::still::rdo_decide::decide_rdoq(
                original_scan, scalar_scan, plane, p.tx_size, p.w, p.h, 0, 0,
                rdoq_config, rdoq_workspace, rdoq_stream, rdoq_levels,
                rdoq_decision, &rdoq_error)) {
          error_ = "rdo_loop rdoq failed: " + rdoq_error;
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
  // Experiment-only unified-loop scratch.  One object per encoder.
  av2enc::still::rdo_loop::CostTrace rdo_final_trace_;
  av2enc::still::rdo_decide::TransformStreamRate rdo_stream_;
  std::array<std::int32_t, 1024> rdo_levels_{};
  std::array<std::uint16_t, 1024> rdo_scan_{};
  std::array<av2enc::rdo::IntraCandidateCost, 61> rdo_luma_candidates_{};
  std::array<av2enc::still::rdo_decide::ChromaCandidateCost, 14>
      rdo_chroma_candidates_{};
  std::array<av2enc::still::rdo_loop::CandidateScore, 61> rdo_luma_scores_{};
  std::array<av2enc::still::rdo_loop::CandidateScore, 14> rdo_chroma_scores_{};
  std::array<int, 61> rdo_prune_keep_{};
  std::array<int, 61> rdo_prune_scratch_{};
  std::vector<std::uint32_t> rdo_sb_activity_;
  std::uint32_t rdo_frame_activity_ = 0;
  double rdo_current_lambda_ = 0.0;
  av2enc::still::rdo_decide::LambdaConfig rdo_current_decide_{};
  std::array<RdoSnapshot, kRdoSnapshotDepth> rdo_snap_{};
  int rdo_snap_depth_ = 0;
  std::size_t rdo_stat_nodes_ = 0;
  std::size_t rdo_stat_leaves_ = 0;
  std::size_t rdo_stat_probes_ = 0;
  av2enc::still::rdo_loop::CostTerms rdo_stat_terms_{};
'''

# All new member functions.  Inserted before block_variance_cost_at().
LOOP_BODY = r'''
  // -------------------------------------------------------------------
  // Unified RDO loop: measurement + recursive search
  // -------------------------------------------------------------------
  bool rdo_copy_rect(const Plane& plane, const RdoRect& rect,
                     std::vector<std::uint8_t>& out) const {
    if (rect.w <= 0 || rect.h <= 0 || rect.x < 0 || rect.y < 0 ||
        rect.x + rect.w > plane.width || rect.y + rect.h > plane.height) {
      return false;
    }
    out.resize(static_cast<std::size_t>(rect.w) * rect.h);
    for (int row = 0; row < rect.h; ++row) {
      std::memcpy(out.data() + static_cast<std::size_t>(row) * rect.w,
                  plane.samples.data() +
                      static_cast<std::size_t>(rect.y + row) * plane.width +
                      rect.x,
                  static_cast<std::size_t>(rect.w));
    }
    return true;
  }

  void rdo_restore_rect(Plane& plane, const RdoRect& rect,
                        const std::vector<std::uint8_t>& in) const {
    for (int row = 0; row < rect.h; ++row) {
      std::memcpy(plane.samples.data() +
                      static_cast<std::size_t>(rect.y + row) * plane.width +
                      rect.x,
                  in.data() + static_cast<std::size_t>(row) * rect.w,
                  static_cast<std::size_t>(rect.w));
    }
  }

  void rdo_snapshot_begin(const av2enc::still::rdo_loop::NodeInfo& info) {
    RdoSnapshot& snap = rdo_snap_[static_cast<std::size_t>(rdo_snap_depth_)];
    ++rdo_snap_depth_;
    const int w = partition::block_width(info.size);
    const int h = partition::block_height(info.size);
    const int w4 = w >> 2;
    const int h4 = h >> 2;
    snap.rect_y = rdo_clamp_rect(
        RdoRect{info.mi_col * 4, info.mi_row * 4, w, h}, recon_[0].width,
        recon_[0].height);
    snap.rect_joint = rdo_clamp_rect(
        RdoRect{info.mi_col, info.mi_row, w4, h4}, mi_cols_, mi_rows_);
    const BlockSize chroma_size =
        partition::chroma_plane_residual_size(geometry_, info.size);
    const int cw = partition::is_valid_block_size(chroma_size)
                       ? partition::block_width(chroma_size)
                       : w / 2;
    const int ch = partition::is_valid_block_size(chroma_size)
                       ? partition::block_height(chroma_size)
                       : h / 2;
    snap.rect_chroma = rdo_clamp_rect(
        RdoRect{info.mi_col * 2, info.mi_row * 2, cw, ch}, recon_[1].width,
        recon_[1].height);
    snap.rect_uv = rdo_clamp_rect(
        RdoRect{info.mi_col / 2, info.mi_row / 2, cw / 4, ch / 4},
        mi_cols_ / 2, mi_rows_ / 2);
    snap.rect_decoded[0] = rdo_clamp_rect(
        RdoRect{info.mi_col, info.mi_row, w4, h4}, mi_cols_, mi_rows_);
    snap.rect_decoded[1] = rdo_clamp_rect(
        RdoRect{info.mi_col / 2, info.mi_row / 2, cw / 4, ch / 4},
        mi_cols_ / 2, mi_rows_ / 2);
    snap.rect_decoded[2] = snap.rect_decoded[1];
    snap.rect_deblock = rdo_clamp_rect(
        RdoRect{info.mi_col, info.mi_row, w4, h4}, deblock_cols_,
        deblock_rows_);

    rdo_copy_rect(recon_[0], snap.rect_y, snap.y);
    rdo_copy_rect(recon_[1], snap.rect_chroma, snap.u);
    rdo_copy_rect(recon_[2], snap.rect_chroma, snap.v);

    const RdoRect& jr = snap.rect_joint;
    snap.joint.resize(static_cast<std::size_t>(jr.w) * jr.h);
    for (int row = 0; row < jr.h; ++row) {
      for (int col = 0; col < jr.w; ++col) {
        snap.joint[static_cast<std::size_t>(row) * jr.w + col] =
            intra_joint_[static_cast<std::size_t>(jr.y + row) * mi_cols_ +
                         static_cast<std::size_t>(jr.x + col)];
      }
    }
    const RdoRect& ur = snap.rect_uv;
    snap.uv.resize(static_cast<std::size_t>(ur.w) * ur.h);
    const int cmi_cols = mi_cols_ / 2;
    for (int row = 0; row < ur.h; ++row) {
      for (int col = 0; col < ur.w; ++col) {
        snap.uv[static_cast<std::size_t>(row) * ur.w + col] =
            uv_smooth_[static_cast<std::size_t>(ur.y + row) * cmi_cols +
                       static_cast<std::size_t>(ur.x + col)];
      }
    }
    for (int plane = 0; plane < 3; ++plane) {
      const RdoRect& dr = snap.rect_decoded[plane];
      std::vector<std::uint8_t>& out = snap.decoded[plane];
      out.resize(static_cast<std::size_t>(dr.w) * dr.h);
      for (int row = 0; row < dr.h; ++row) {
        for (int col = 0; col < dr.w; ++col) {
          out[static_cast<std::size_t>(row) * dr.w + col] =
              decoded_[plane].cells[static_cast<std::size_t>(dr.y + row) *
                                        decoded_[plane].cols +
                                    static_cast<std::size_t>(dr.x + col)];
        }
      }
      const RdoRect& br = snap.rect_deblock;
      std::vector<deblock::DeblockMi>& dout = snap.deblock[plane];
      dout.resize(static_cast<std::size_t>(br.w) * br.h);
      for (int row = 0; row < br.h; ++row) {
        for (int col = 0; col < br.w; ++col) {
          dout[static_cast<std::size_t>(row) * br.w + col] =
              deblock_mi_[plane][static_cast<std::size_t>(br.y + row) *
                                      deblock_cols_ +
                                  static_cast<std::size_t>(br.x + col)];
        }
      }
    }
    snap.blocks = blocks_.size();
    snap.levels = levels_.size();
  }

  void rdo_snapshot_restore() {
    --rdo_snap_depth_;
    const RdoSnapshot& snap =
        rdo_snap_[static_cast<std::size_t>(rdo_snap_depth_)];
    rdo_restore_rect(recon_[0], snap.rect_y, snap.y);
    rdo_restore_rect(recon_[1], snap.rect_chroma, snap.u);
    rdo_restore_rect(recon_[2], snap.rect_chroma, snap.v);

    const RdoRect& jr = snap.rect_joint;
    for (int row = 0; row < jr.h; ++row) {
      for (int col = 0; col < jr.w; ++col) {
        intra_joint_[static_cast<std::size_t>(jr.y + row) * mi_cols_ +
                     static_cast<std::size_t>(jr.x + col)] =
            snap.joint[static_cast<std::size_t>(row) * jr.w + col];
      }
    }
    const RdoRect& ur = snap.rect_uv;
    const int cmi_cols = mi_cols_ / 2;
    for (int row = 0; row < ur.h; ++row) {
      for (int col = 0; col < ur.w; ++col) {
        uv_smooth_[static_cast<std::size_t>(ur.y + row) * cmi_cols +
                   static_cast<std::size_t>(ur.x + col)] =
            snap.uv[static_cast<std::size_t>(row) * ur.w + col];
      }
    }
    for (int plane = 0; plane < 3; ++plane) {
      const RdoRect& dr = snap.rect_decoded[plane];
      const std::vector<std::uint8_t>& in = snap.decoded[plane];
      for (int row = 0; row < dr.h; ++row) {
        for (int col = 0; col < dr.w; ++col) {
          decoded_[plane].cells[static_cast<std::size_t>(dr.y + row) *
                                    decoded_[plane].cols +
                                static_cast<std::size_t>(dr.x + col)] =
              in[static_cast<std::size_t>(row) * dr.w + col];
        }
      }
      const RdoRect& br = snap.rect_deblock;
      const std::vector<deblock::DeblockMi>& din = snap.deblock[plane];
      for (int row = 0; row < br.h; ++row) {
        for (int col = 0; col < br.w; ++col) {
          deblock_mi_[plane][static_cast<std::size_t>(br.y + row) *
                                  deblock_cols_ +
                              static_cast<std::size_t>(br.x + col)] =
              din[static_cast<std::size_t>(row) * br.w + col];
        }
      }
    }
    blocks_.resize(snap.blocks);
    levels_.resize(snap.levels);
  }

  // Sum the newly appended leaf entries [block_begin, end).  The leaves are in
  // decode order and their level slices are in `levels_` scan order.
  bool rdo_measure_collected(std::size_t block_begin,
                             av2enc::still::rdo_loop::CostTerms& terms,
                             int& leaves, std::string& error) {
    terms = av2enc::still::rdo_loop::CostTerms{};
    leaves = 0;
    for (std::size_t index = block_begin; index < blocks_.size(); ++index) {
      const syntax::BlockSyntax& block = blocks_[index];
      if (block.partition !=
          static_cast<std::uint8_t>(PartitionType::kNone)) {
        continue;
      }
      if (!rdo_measure_leaf(block, terms, error)) return false;
      ++leaves;
    }
    return true;
  }

  bool rdo_measure_leaf(const syntax::BlockSyntax& block,
                        av2enc::still::rdo_loop::CostTerms& terms,
                        std::string& error) {
    const BlockSize size = static_cast<BlockSize>(block.bsize);
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    const int x0 = block.mi_col * 4;
    const int y0 = block.mi_row * 4;
    double sse = 0.0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int src = source_sample(0, x0 + x, y0 + y);
        const int rec = recon_[0].at(x0 + x, y0 + y);
        const int d = src - rec;
        sse += static_cast<double>(d) * d;
      }
    }
    const BlockSize chroma_size =
        partition::chroma_plane_residual_size(geometry_, size);
    if (!partition::is_valid_block_size(chroma_size)) {
      error = "rdo_loop: invalid chroma residual size";
      return false;
    }
    const int cw = partition::block_width(chroma_size);
    const int ch = partition::block_height(chroma_size);
    const int cx0 = x0 / 2;
    const int cy0 = y0 / 2;
    for (int plane = 1; plane <= 2; ++plane) {
      for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
          const int src = source_sample(plane, cx0 + x, cy0 + y);
          const int rec = recon_[plane].at(cx0 + x, cy0 + y);
          const int d = src - rec;
          sse += static_cast<double>(d) * d;
        }
      }
    }
    terms.distortion_sse += sse;

    const bool cfl = (block.flags & syntax::kFlagIsCfl) != 0;
    terms.mode_bits +=
        av2enc::still::rdo_decide::intra_syntax_bits(block.y_mode,
                                                  block.y_angle_delta);
    const int chroma_delta =
        (!cfl && y_mode_is_directional(block.y_mode) &&
         block.uv_mode == block.y_mode)
            ? block.y_angle_delta
            : 0;
    terms.mode_bits += av2enc::still::rdo_decide::default_chroma_syntax_bits(
        cfl, block.uv_mode, chroma_delta);

    if (block.eob[0] > 1 &&
        block.tx_type[0] !=
            static_cast<std::uint8_t>(txv::TxType::kDctDct)) {
      txtype::TxTypeQuery query;
      query.coded_w = w;
      query.coded_h = h;
      query.mapped_mode =
          txtype::map_intra_mode(block.y_mode, block.y_angle_delta, w, h);
      query.reduced_tx_set = 0;
      query.dct_only = false;
      txtype::TxTypeSet set;
      if (txtype::intra_tx_candidates(query, &set)) {
        for (int i = 0; i < set.count; ++i) {
          if (static_cast<std::uint8_t>(set.candidates[i].type) ==
              block.tx_type[0]) {
            terms.tx_type_bits +=
                txtype::static_type_bits(query, set.candidates[i]);
            break;
          }
        }
      }
    }

    std::size_t cursor = block.levels_offset;
    for (int plane = 0; plane < 3; ++plane) {
      const int eob = static_cast<int>(block.eob[plane]);
      if (eob <= 0) continue;
      int coded_w = w;
      int coded_h = h;
      if (plane > 0) {
        coded_w = cw;
        coded_h = ch;
      }
      const auto params =
          txv::make_tx_params(coded_w, coded_h, txv::TxType::kDctDct);
      if (!params.has_value()) {
        error = "rdo_loop: unsupported tx size in measurement";
        return false;
      }
      const int area = params->w * params->h;
      if (area > static_cast<int>(rdo_levels_.size()) ||
          cursor + static_cast<std::size_t>(eob) > levels_.size()) {
        error = "rdo_loop: level slice out of bounds";
        return false;
      }
      for (int c = 0; c < eob; ++c) {
        rdo_levels_[static_cast<std::size_t>(c)] =
            levels_[cursor + static_cast<std::size_t>(c)];
      }
      for (int c = eob; c < area; ++c) {
        rdo_levels_[static_cast<std::size_t>(c)] = 0;
      }
      av2enc::still::rdo_decide::TransformStreamBlock stream_block;
      stream_block.plane = plane;
      stream_block.tx_size = block.tx_size[plane];
      stream_block.w = params->w;
      stream_block.h = params->h;
      stream_block.levels_scan = std::span<const std::int32_t>(
          rdo_levels_.data(), static_cast<std::size_t>(area));
      stream_block.eob = eob;
      if (!rdo_stream_.init(stream_block)) {
        error = "rdo_loop: transform stream init failed";
        return false;
      }
      terms.coefficient_bits += rdo_stream_.block_bits();
      cursor += static_cast<std::size_t>(eob);
    }
    return true;
  }

  partition::PartitionNode rdo_make_plan(
      const av2enc::still::rdo_loop::NodeInfo& info,
      PartitionType partition) const {
    partition::PartitionNode node;
    node.mi_row = info.mi_row;
    node.mi_col = info.mi_col;
    node.size = info.size;
    node.partition = partition;
    node.depth = info.depth;
    node.has_chroma = info.has_chroma;
    node.chroma_offset = info.chroma_offset;
    node.boundary = info.boundary;
    node.allowed = info.allowed;
    node.forced = info.forced;
    node.cost = 0.0;
    return node;
  }

  bool rdo_commit_plan(partition::PartitionNode& plan) {
    return collect_node(plan);
  }

  bool rdo_search_node(
      const av2enc::still::rdo_loop::NodeInfo& info,
      const av2enc::still::rdo_loop::SearchOptions& options,
      av2enc::still::rdo_loop::ProbeBudget& budget,
      partition::PartitionNode& out_plan,
      av2enc::still::rdo_loop::NodeCost& out_cost, std::string& error) {
    av2enc::still::rdo_loop::NodeDecisionLoop loop;
    loop.reset(info, options);
    loop.set_lambda(rdo_current_lambda_);
    const bool relax =
        !info.boundary.has_rows || !info.boundary.has_cols;
    partition::PartitionNode best_plan;
    bool have_best = false;

    for (int order = 0; order < av2enc::still::rdo_loop::kSearchOrderCount;
         ++order) {
      const PartitionType partition =
          av2enc::still::rdo_loop::kSearchOrder[order];
      if (info.forced && partition != info.forced_partition) continue;
      if (!info.allowed.is_allowed(partition)) continue;
      if (!info.forced && !relax &&
          !av2enc::still::rdo_loop::policy_allows(options, info.depth, info.size,
                                               partition)) {
        continue;
      }
      if (partition != PartitionType::kNone && !budget.take()) {
        break;
      }

      if (partition == PartitionType::kNone) {
        partition::PartitionNode plan = rdo_make_plan(info, partition);
        rdo_snapshot_begin(info);
        const std::size_t block_begin = blocks_.size();
        ++rdo_stat_probes_;
        if (!collect_node(plan)) {
          rdo_snapshot_restore();
          error = "rdo_loop: NONE trial failed: " + error_;
          return false;
        }
        av2enc::still::rdo_loop::CostTerms terms;
        int leaves = 0;
        if (!rdo_measure_collected(block_begin, terms, leaves, error)) {
          rdo_snapshot_restore();
          return false;
        }
        terms.signal_bits +=
            av2enc::still::rdo_loop::default_partition_signal_bits(
                info, PartitionType::kNone);
        rdo_snapshot_restore();
        const int slot = loop.decision().candidate_count;
        if (!loop.add_terms(partition, terms, leaves, &error)) return false;
        if (loop.decision().candidate_index == slot) {
          best_plan = std::move(plan);
          have_best = true;
        }
        continue;
      }

      av2enc::still::rdo_loop::ChildRef children[4];
      const int child_count = av2enc::still::rdo_loop::enumerate_node_children(
          geometry_, info, partition, children, &error);
      if (child_count <= 0) continue;

      rdo_snapshot_begin(info);
      partition::PartitionNode plan = rdo_make_plan(info, partition);
      std::array<av2enc::still::rdo_loop::NodeCost, 4> child_costs{};
      bool ok = true;
      for (int ci = 0; ci < child_count; ++ci) {
        av2enc::still::rdo_loop::NodeInfo child_info;
        if (!av2enc::still::rdo_loop::describe_node(
                geometry_, children[ci].mi_row, children[ci].mi_col,
                children[ci].size, info.depth + 1, children[ci].has_chroma,
                children[ci].chroma_offset, &child_info, &error)) {
          ok = false;
          break;
        }
        partition::PartitionNode child_plan;
        av2enc::still::rdo_loop::NodeCost child_cost;
        if (!rdo_search_node(child_info, options, budget, child_plan,
                             child_cost, error)) {
          ok = false;
          break;
        }
        ++rdo_stat_probes_;
        if (!rdo_commit_plan(child_plan)) {
          ok = false;
          error = "rdo_loop: child commit failed: " + error_;
          break;
        }
        child_costs[static_cast<std::size_t>(ci)] = child_cost;
        plan.children.push_back(std::move(child_plan));

        av2enc::still::rdo_loop::NodeCost partial =
            av2enc::still::rdo_loop::make_node_cost(
                av2enc::still::rdo_loop::CostTerms{}, rdo_current_lambda_, true,
                0);
        for (int cj = 0; cj <= ci; ++cj) {
          av2enc::still::rdo_loop::accumulate_node_cost(
              partial, child_costs[static_cast<std::size_t>(cj)]);
        }
        if (loop.cannot_beat_best(partial)) {
          ok = false;
          break;
        }
      }
      if (ok) {
        std::span<const av2enc::still::rdo_loop::NodeCost> span(
            child_costs.data(), static_cast<std::size_t>(child_count));
        const int slot = loop.decision().candidate_count;
        if (!loop.add_partition(partition, span, &error)) {
          rdo_snapshot_restore();
          return false;
        }
        if (loop.decision().candidate_index == slot) {
          best_plan = std::move(plan);
          have_best = true;
        }
      }
      rdo_snapshot_restore();
    }

    if (!have_best || !loop.decision().found) {
      error = "rdo_loop: no valid partition candidate";
      return false;
    }
    out_plan = std::move(best_plan);
    out_cost = loop.decision().cost;
    return true;
  }

  bool rdo_record_final_tree(const partition::PartitionNode& node,
                             std::size_t& block_index,
                             av2enc::still::rdo_loop::CostTerms& terms,
                             int& leaves) {
    if (block_index >= blocks_.size()) {
      error_ = "rdo_loop: final trace overran the block stream";
      return false;
    }
    const syntax::BlockSyntax& entry = blocks_[block_index];
    const std::size_t my_index = block_index++;
    av2enc::still::rdo_loop::NodeInfo info;
    if (!av2enc::still::rdo_loop::describe_node(
            geometry_, node.mi_row, node.mi_col, node.size, node.depth,
            node.has_chroma, node.chroma_offset, &info, &error_)) {
      return false;
    }
    terms = av2enc::still::rdo_loop::CostTerms{};
    leaves = 0;
    int child_count = 0;
    if (entry.partition == static_cast<std::uint8_t>(PartitionType::kNone)) {
      if (!rdo_measure_leaf(entry, terms, error_)) return false;
      leaves = 1;
    } else {
      for (const partition::PartitionNode& child : node.children) {
        av2enc::still::rdo_loop::CostTerms child_terms;
        int child_leaves = 0;
        if (!rdo_record_final_tree(child, block_index, child_terms,
                                   child_leaves)) {
          return false;
        }
        terms.distortion_sse += child_terms.distortion_sse;
        terms.mode_bits += child_terms.mode_bits;
        terms.coefficient_bits += child_terms.coefficient_bits;
        terms.tx_type_bits += child_terms.tx_type_bits;
        terms.signal_bits += child_terms.signal_bits;
        leaves += child_leaves;
        ++child_count;
      }
    }
    terms.signal_bits +=
        av2enc::still::rdo_loop::default_partition_signal_bits(info,
                                                            node.partition);
    const av2enc::still::rdo_loop::NodeCost cost =
        av2enc::still::rdo_loop::make_node_cost(terms, rdo_current_lambda_, true,
                                             leaves);
    av2enc::still::rdo_loop::NodeTraceEntry trace_entry;
    trace_entry.mi_row = node.mi_row;
    trace_entry.mi_col = node.mi_col;
    trace_entry.size = static_cast<std::uint16_t>(node.size);
    trace_entry.depth = static_cast<std::int8_t>(node.depth);
    trace_entry.partition = static_cast<std::uint8_t>(node.partition);
    trace_entry.forced = node.forced ? 1 : 0;
    trace_entry.rate_aware = 1;
    trace_entry.child_count = static_cast<std::uint8_t>(child_count);
    trace_entry.lambda = rdo_current_lambda_;
    trace_entry.terms = terms;
    trace_entry.cost = cost.cost;
    trace_entry.leaves = leaves;
    (void)my_index;
    // A full trace only drops diagnostic rows; it must not fail the encode.
    (void)rdo_final_trace_.record(trace_entry);
    return true;
  }

  bool encode_with_rdo_loop(SyntaxStream* stream,
                            const partition::DecisionOptions& poptions,
                            std::string& error) {
    av2enc::still::rdo_loop::LambdaConfig lambda_config =
        rdo_loop_lambda_from_env();
    std::string lambda_error;
    if (!av2enc::still::rdo_loop::validate_lambda_config(lambda_config,
                                                      &lambda_error)) {
      error = "rdo_loop lambda: " + lambda_error;
      return false;
    }
    av2enc::still::rdo_loop::SearchOptions options;
    options.min_block_size_px = poptions.min_block_size_px;
    options.max_depth = poptions.max_depth;
    options.allow_rect = poptions.allow_rect;
    options.allow_split = poptions.allow_split;
    options.rate_aware = true;
    options.max_luma_candidates =
        rdo_loop_env_int("AV2ENC_RDO_LOOP_MAX_LUMA",
                         options.max_luma_candidates);
    options.max_chroma_candidates =
        rdo_loop_env_int("AV2ENC_RDO_LOOP_MAX_CHROMA",
                         options.max_chroma_candidates);
    options.candidate_sse_margin = rdo_loop_env_double(
        "AV2ENC_RDO_LOOP_MARGIN", options.candidate_sse_margin);
    options.max_leaf_probes_per_superblock =
        rdo_loop_env_int("AV2ENC_RDO_LOOP_PROBES", 0);
    if (!av2enc::still::rdo_loop::validate_search_options(options, &lambda_error)) {
      error = "rdo_loop options: " + lambda_error;
      return false;
    }

    const int sb_cols = partition::sb_cols(geometry_);
    const int sb_rows = partition::sb_rows(geometry_);
    rdo_sb_activity_.assign(static_cast<std::size_t>(sb_cols) * sb_rows, 0);
    for (int sb_r = 0; sb_r < sb_rows; ++sb_r) {
      for (int sb_c = 0; sb_c < sb_cols; ++sb_c) {
        const int x = sb_c * 64;
        const int y = sb_r * 64;
        const int w = std::min(64, width_ - x);
        const int h = std::min(64, height_ - y);
        rdo_sb_activity_[static_cast<std::size_t>(sb_r) * sb_cols + sb_c] =
            av2enc::still::rdo_loop::luma_activity_q8(picture_.y.data(), width_,
                                                    x, y, w, h);
      }
    }
    rdo_frame_activity_ = av2enc::still::rdo_loop::mean_activity_q8(
        rdo_sb_activity_.data(), sb_cols * sb_rows);

    blocks_.clear();
    levels_.clear();
    leaves_ = 0;
    rdo_final_trace_.clear();
    rdo_stat_nodes_ = 0;
    rdo_stat_leaves_ = 0;
    rdo_stat_probes_ = 0;
    rdo_stat_terms_ = av2enc::still::rdo_loop::CostTerms{};

    partition::FrameDecision frame;
    frame.status = partition::DecisionStatus::kOk;
    frame.cost = 0.0;
    for (int sbi = 0; sbi < sb_cols * sb_rows; ++sbi) {
      const int sb_r = sbi / sb_cols;
      const int sb_c = sbi % sb_cols;
      const int sb_mi_row = sb_r * partition::sb_size_mi(geometry_);
      const int sb_mi_col = sb_c * partition::sb_size_mi(geometry_);
      for (int plane = 0; plane < 3; ++plane) {
        decoded_[plane].reset(plane == 0 ? mi_cols_ : mi_cols_ / 2,
                              plane == 0 ? mi_rows_ : mi_rows_ / 2);
      }
      sb_mi_row_ = sb_mi_row;
      sb_mi_col_ = sb_mi_col;
      sb_cols_plane_[0] = std::min(16, mi_cols_ - sb_mi_col);
      sb_rows_plane_[0] = std::min(16, mi_rows_ - sb_mi_row);
      sb_cols_plane_[1] = std::min(8, mi_cols_ / 2 - sb_mi_col / 2);
      sb_rows_plane_[1] = std::min(8, mi_rows_ / 2 - sb_mi_row / 2);
      sb_cols_plane_[2] = sb_cols_plane_[1];
      sb_rows_plane_[2] = sb_rows_plane_[1];

      av2enc::still::rdo_loop::NodeInfo info;
      if (!av2enc::still::rdo_loop::describe_node(
              geometry_, sb_mi_row, sb_mi_col, BlockSize::k64x64, 0, true,
              false, &info, &error)) {
        return false;
      }
      const av2enc::still::rdo_loop::CtuLambda ctu =
          av2enc::still::rdo_loop::ctu_lambda(
              lambda_config,
              rdo_sb_activity_[static_cast<std::size_t>(sbi)],
              rdo_frame_activity_);
      rdo_current_lambda_ = ctu.lambda;
      rdo_current_decide_.quant_step = lambda_config.quant_step;
      rdo_current_decide_.sse_scale =
          ctu.lambda / (lambda_config.quant_step * lambda_config.quant_step);
      rdo_current_decide_.rate_aware = true;

      av2enc::still::rdo_loop::ProbeBudget budget;
      budget.reset(options.max_leaf_probes_per_superblock);
      partition::PartitionNode plan;
      av2enc::still::rdo_loop::NodeCost cost;
      if (!rdo_search_node(info, options, budget, plan, cost, error)) {
        return false;
      }
      const std::size_t root_block_begin = blocks_.size();
      if (!rdo_commit_plan(plan)) {
        error = "rdo_loop: root commit failed: " + error_;
        return false;
      }
      {
        std::size_t trace_index = root_block_begin;
        av2enc::still::rdo_loop::CostTerms trace_terms;
        int trace_leaves = 0;
        if (!rdo_record_final_tree(plan, trace_index, trace_terms,
                                   trace_leaves)) {
          error = "rdo_loop: final trace failed: " + error_;
          return false;
        }
      }
      if (std::getenv("AV2ENC_RDO_LOOP_DEBUG") != nullptr) {
        std::fprintf(stderr,
                     "rdo_loop sb=%d root_depth=%d root_part=%d blocks=%zu "
                     "first_flags=%u\n",
                     sbi, plan.depth, static_cast<int>(plan.partition),
                     blocks_.size(),
                     blocks_.empty() ? 0U
                                     : static_cast<unsigned>(blocks_[0].flags));
      }
      if (!partition::validate_tree(geometry_, plan, error)) {
        error = "rdo_loop: invalid tree: " + error;
        return false;
      }
      frame.cost += cost.cost;
      frame.nodes += partition::count_nodes(plan);
      frame.leaves += partition::count_leaves(plan);
      rdo_stat_nodes_ += static_cast<std::size_t>(
          partition::count_nodes(plan));
      rdo_stat_leaves_ += static_cast<std::size_t>(
          partition::count_leaves(plan));
      rdo_stat_terms_.distortion_sse += cost.terms.distortion_sse;
      rdo_stat_terms_.mode_bits += cost.terms.mode_bits;
      rdo_stat_terms_.coefficient_bits += cost.terms.coefficient_bits;
      rdo_stat_terms_.tx_type_bits += cost.terms.tx_type_bits;
      rdo_stat_terms_.signal_bits += cost.terms.signal_bits;
      frame.superblocks.push_back(std::move(plan));
    }
    if (!partition::validate_frame(geometry_, frame, error)) {
      error = "rdo_loop: invalid frame tree: " + error;
      return false;
    }
    leaves_ = frame.leaves;
    stream->blocks = std::move(blocks_);
    stream->levels = std::move(levels_);

    if (const char* trace_path = std::getenv("AV2ENC_RDO_LOOP_TRACE")) {
      std::ofstream out(trace_path);
      if (!out) {
        error = "rdo_loop: cannot open trace path";
        return false;
      }
      out << "mi_row,mi_col,size,depth,partition,forced,lambda,sse,"
             "mode_bits,coeff_bits,tx_bits,signal_bits,cost,leaves\n";
      for (std::size_t i = 0; i < rdo_final_trace_.size(); ++i) {
        const auto& entry = rdo_final_trace_.at(i);
        out << entry.mi_row << ',' << entry.mi_col << ','
            << static_cast<int>(entry.size) << ','
            << static_cast<int>(entry.depth) << ','
            << static_cast<int>(entry.partition) << ','
            << static_cast<int>(entry.forced) << ',' << entry.lambda << ','
            << entry.terms.distortion_sse << ',' << entry.terms.mode_bits
            << ',' << entry.terms.coefficient_bits << ','
            << entry.terms.tx_type_bits << ',' << entry.terms.signal_bits
            << ',' << entry.cost << ',' << entry.leaves << '\n';
      }
    }
    if (std::getenv("AV2ENC_RDO_LOOP_STATS") != nullptr) {
      std::fprintf(
          stderr,
          "rdo_loop stats: nodes=%zu leaves=%zu probes=%zu sse=%.3f "
          "mode_bits=%.3f coeff_bits=%.3f tx_bits=%.3f signal_bits=%.3f\n",
          rdo_stat_nodes_, rdo_stat_leaves_, rdo_stat_probes_,
          rdo_stat_terms_.distortion_sse, rdo_stat_terms_.mode_bits,
          rdo_stat_terms_.coefficient_bits, rdo_stat_terms_.tx_type_bits,
          rdo_stat_terms_.signal_bits);
    }
    return true;
  }

  // Luma mode/angle search with SSE pruning and the shared transform-stream
  // rate.  The legacy path is untouched and remains the patch-neutral control.
  bool choose_luma_mode_loop(int x0, int y0, int w, int h, ModeChoice& best) {
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
    intra::IntraEdges edges;
    if (!prepare_edges(0, x0, y0, w, h, edges)) {
      error_ = "rdo_loop luma prepare_edges failed";
      return false;
    }
    bool above_smooth = false;
    bool left_smooth = false;
    neighborhood_smooth(0, y0 >> 2, x0 >> 2, w >> 2, h >> 2, above_smooth,
                        left_smooth);
    std::array<intra::ModeCandidate, 61> candidates;
    const int count = intra::build_mode_candidates(candidates, true);
    if (count <= 0) {
      error_ = "rdo_loop build_mode_candidates failed";
      return false;
    }
    const bool base_only = std::getenv("AV2ENC_STILL_BASE_ONLY") != nullptr;
    const bool no_directional = std::getenv("AV2ENC_STILL_NO_DIR") != nullptr;
    std::array<intra::Mode, 61> modes{};
    std::array<int, 61> deltas{};
    std::vector<std::uint8_t> pred;
    pred.resize(static_cast<std::size_t>(w) * h);
    int score_count = 0;
    int best_score_index = -1;
    int dc_index = -1;
    std::int64_t best_sse = std::numeric_limits<std::int64_t>::max();
    for (int i = 0; i < count; ++i) {
      const intra::ModeCandidate& candidate =
          candidates[static_cast<std::size_t>(i)];
      if (base_only && candidate.angle_delta != 0) continue;
      if (no_directional && intra::is_directional(candidate.mode)) continue;
      intra::PredictParams params;
      params.mode = candidate.mode;
      params.angle_delta = candidate.angle_delta;
      params.enable_edge_filter = edge_filter_enabled();
      params.above_neighbor_is_smooth = above_smooth;
      params.left_neighbor_is_smooth = left_smooth;
      if (intra::predict_intra(edges, params, pred.data(),
                               static_cast<std::size_t>(w)) !=
          intra::IntraStatus::kOk) {
        error_ = "rdo_loop luma predict_intra failed";
        return false;
      }
      const std::int64_t sse = sse_against_source(0, x0, y0, w, h, pred);
      modes[static_cast<std::size_t>(score_count)] = candidate.mode;
      deltas[static_cast<std::size_t>(score_count)] = candidate.angle_delta;
      rdo_luma_scores_[static_cast<std::size_t>(score_count)] = {
          score_count, static_cast<double>(sse)};
      if (candidate.mode == intra::Mode::kDc && candidate.angle_delta == 0) {
        dc_index = score_count;
      }
      ++score_count;
      if (sse < best_sse) {
        best_sse = sse;
        best_score_index = score_count - 1;
      }
    }
    const auto lambda_config = rdo_current_decide_;
    const av2enc::still::rdo_loop::SearchOptions options =
        rdo_loop_search_options(current_options());
    int keep_count = av2enc::still::rdo_loop::prune_candidates(
        std::span<const av2enc::still::rdo_loop::CandidateScore>(
            rdo_luma_scores_.data(), static_cast<std::size_t>(score_count)),
        std::span<int>(rdo_prune_keep_.data(),
                       rdo_prune_keep_.size()),
        std::span<int>(rdo_prune_scratch_.data(),
                       rdo_prune_scratch_.size()),
        av2enc::still::rdo_loop::PruneConfig{
            options.max_luma_candidates, options.candidate_sse_margin,
            dc_index});
    if (keep_count <= 0) {
      keep_count = 0;
      rdo_prune_keep_[0] = 0;
      keep_count = 1;
    }
    int rdo_count = 0;
    for (int k = 0; k < keep_count; ++k) {
      const int index = rdo_prune_keep_[static_cast<std::size_t>(k)];
      intra::PredictParams params;
      params.mode = modes[static_cast<std::size_t>(index)];
      params.angle_delta = deltas[static_cast<std::size_t>(index)];
      params.enable_edge_filter = edge_filter_enabled();
      params.above_neighbor_is_smooth = above_smooth;
      params.left_neighbor_is_smooth = left_smooth;
      if (intra::predict_intra(edges, params, pred.data(),
                               static_cast<std::size_t>(w)) !=
          intra::IntraStatus::kOk) {
        error_ = "rdo_loop luma predict_intra failed";
        return false;
      }
      TxPlan probe;
      if (!compute_tx(0, x0, y0, w, h, pred, probe, options_.tx_type,
                      av2_y_mode(params.mode), params.angle_delta)) {
        error_ = "rdo_loop luma probe compute_tx failed: " + error_;
        return false;
      }
      if (probe.adj_w * probe.adj_h > 1024) {
        error_ = "rdo_loop luma probe too large";
        return false;
      }
      syntax::build_scan(probe.adj_w, probe.adj_h,
                         std::span<std::uint16_t>(
                             rdo_scan_.data(),
                             static_cast<std::size_t>(probe.adj_w *
                                                      probe.adj_h)));
      const int area = probe.adj_w * probe.adj_h;
      for (int c = 0; c < area; ++c) {
        rdo_levels_[static_cast<std::size_t>(c)] =
            probe.levels[rdo_scan_[static_cast<std::size_t>(c)]];
      }
      av2enc::still::rdo_decide::TransformStreamBlock block;
      block.plane = 0;
      block.tx_size = probe.tx_size;
      block.w = probe.adj_w;
      block.h = probe.adj_h;
      block.levels_scan = std::span<const std::int32_t>(
          rdo_levels_.data(), static_cast<std::size_t>(area));
      block.eob = probe.eob;
      std::string luma_stream_error;
      if (!rdo_stream_.init(block, &luma_stream_error)) {
        error_ = "rdo_loop luma stream init failed: " + luma_stream_error +
                 " area=" + std::to_string(area) +
                 " eob=" + std::to_string(probe.eob) +
                 " tx=" + std::to_string(probe.tx_size) +
                 " wh=" + std::to_string(probe.adj_w) + "x" +
                 std::to_string(probe.adj_h) +
                 " span=" + std::to_string(block.levels_scan.size()) +
                 " bw=" + std::to_string(block.w) +
                 " bh=" + std::to_string(block.h);
        return false;
      }
      av2enc::rdo::IntraCandidateCost& candidate =
          rdo_luma_candidates_[static_cast<std::size_t>(rdo_count)];
      candidate.mode = av2_y_mode(params.mode);
      candidate.angle_delta = params.angle_delta;
      candidate.sse = static_cast<double>(sse_against_source(0, x0, y0, w, h,
                                                             pred));
      candidate.syntax_rate_bits =
          av2enc::still::rdo_decide::intra_syntax_bits(candidate.mode,
                                                    candidate.angle_delta);
      candidate.coefficient_rate_bits =
          av2enc::still::rdo_decide::intra_coefficient_bits(rdo_stream_);
      ++rdo_count;
    }
    if (rdo_count > 0) {
      const auto choice = av2enc::still::rdo_decide::decide_luma_intra(
          std::span<const av2enc::rdo::IntraCandidateCost>(
              rdo_luma_candidates_.data(),
              static_cast<std::size_t>(rdo_count)),
          {}, lambda_config);
      if (choice.found) {
        best.mode = static_cast<intra::Mode>(choice.mode);
        best.angle_delta = choice.angle_delta;
        best.av2_mode = choice.mode;
        best.sse = static_cast<std::int64_t>(choice.sse);
        return true;
      }
    }
    // Deterministic fallback: SSE winner.
    if (best_score_index >= 0) {
      best.mode = modes[static_cast<std::size_t>(best_score_index)];
      best.angle_delta = deltas[static_cast<std::size_t>(best_score_index)];
      best.av2_mode = av2_y_mode(best.mode);
      best.sse = best_sse;
    }
    return true;
  }

  av2enc::still::rdo_loop::SearchOptions rdo_loop_search_options(
      const StillOptions& still_options) const {
    av2enc::still::rdo_loop::SearchOptions options;
    options.min_block_size_px = still_options.min_block_size;
    options.max_depth = still_options.max_partition_depth;
    options.allow_rect = true;
    options.allow_split = false;
    options.rate_aware = true;
    options.max_luma_candidates =
        rdo_loop_env_int("AV2ENC_RDO_LOOP_MAX_LUMA",
                         options.max_luma_candidates);
    options.max_chroma_candidates =
        rdo_loop_env_int("AV2ENC_RDO_LOOP_MAX_CHROMA",
                         options.max_chroma_candidates);
    options.candidate_sse_margin = rdo_loop_env_double(
        "AV2ENC_RDO_LOOP_MARGIN", options.candidate_sse_margin);
    options.max_leaf_probes_per_superblock =
        rdo_loop_env_int("AV2ENC_RDO_LOOP_PROBES", 0);
    return options;
  }

  const StillOptions& current_options() const { return options_; }

  // Chroma DC / CfL / directional search on u_sse + v_sse with pruning.
  bool choose_uv_mode_loop(int cx0, int cy0, int cw, int ch, int y_mode,
                           int y_angle_delta, UvChoice& best) {
    best.is_cfl = false;
    best.sse = std::numeric_limits<std::int64_t>::max();
    const bool uv_full =
        options_.uv_full || std::getenv("AV2ENC_STILL_UV_FULL") != nullptr;
    if (const char* forced = std::getenv("AV2ENC_STILL_FORCE_UV")) {
      int mode = 0;
      int delta = 0;
      if (std::sscanf(forced, "%d:%d", &mode, &delta) == 2) {
        const int derived =
            (y_mode_is_directional(y_mode) && mode == y_mode)
                ? y_angle_delta
                : 0;
        best.mode = static_cast<intra::Mode>(mode);
        best.angle_delta = derived;
        best.av2_mode = av2_y_mode(best.mode);
        best.sse = 0;
        return true;
      }
      error_ = "invalid AV2ENC_STILL_FORCE_UV (expected mode:delta)";
      return false;
    }

    static const intra::Mode kBase[13] = {
        intra::Mode::kDc,      intra::Mode::kV,       intra::Mode::kH,
        intra::Mode::kD45,     intra::Mode::kD135,    intra::Mode::kD113,
        intra::Mode::kD157,    intra::Mode::kD203,    intra::Mode::kD67,
        intra::Mode::kSmooth,  intra::Mode::kSmoothV, intra::Mode::kSmoothH,
        intra::Mode::kPaeth};

    std::vector<std::uint8_t> pred;
    std::array<intra::Mode, 13> modes{};
    std::array<int, 13> deltas{};
    int score_count = 0;
    int best_score_index = -1;
    std::int64_t best_sse = std::numeric_limits<std::int64_t>::max();
    for (int i = 0; i < 13; ++i) {
      const intra::Mode mode = kBase[i];
      const int delta =
          (y_mode_is_directional(y_mode) &&
           static_cast<int>(mode) == y_mode)
              ? y_angle_delta
              : 0;
      if (!uv_full && mode != intra::Mode::kDc) continue;
      if (!predict_block(1, cx0, cy0, cw, ch, mode, delta, pred)) return false;
      std::int64_t sse = sse_against_source(1, cx0, cy0, cw, ch, pred);
      if (uv_full) {
        if (!predict_block(2, cx0, cy0, cw, ch, mode, delta, pred)) return false;
        sse += sse_against_source(2, cx0, cy0, cw, ch, pred);
      }
      modes[static_cast<std::size_t>(score_count)] = mode;
      deltas[static_cast<std::size_t>(score_count)] = delta;
      rdo_chroma_scores_[static_cast<std::size_t>(score_count)] = {
          score_count, static_cast<double>(sse)};
      ++score_count;
      if (sse < best_sse) {
        best_sse = sse;
        best_score_index = score_count - 1;
      }
    }
    const auto lambda_config = rdo_current_decide_;
    const auto options = rdo_loop_search_options(options_);
    int keep_count = av2enc::still::rdo_loop::prune_candidates(
        std::span<const av2enc::still::rdo_loop::CandidateScore>(
            rdo_chroma_scores_.data(), static_cast<std::size_t>(score_count)),
        std::span<int>(rdo_prune_keep_.data(), rdo_prune_keep_.size()),
        std::span<int>(rdo_prune_scratch_.data(),
                       rdo_prune_scratch_.size()),
        av2enc::still::rdo_loop::PruneConfig{
            options.max_chroma_candidates, options.candidate_sse_margin, 0});
    if (keep_count <= 0) {
      rdo_prune_keep_[0] = 0;
      keep_count = 1;
    }
    int rdo_count = 0;
    for (int k = 0; k < keep_count; ++k) {
      const int index = rdo_prune_keep_[static_cast<std::size_t>(k)];
      const intra::Mode mode = modes[static_cast<std::size_t>(index)];
      const int delta = deltas[static_cast<std::size_t>(index)];
      std::vector<std::uint8_t> u_pred;
      std::vector<std::uint8_t> v_pred;
      if (!predict_block(1, cx0, cy0, cw, ch, mode, delta, u_pred)) {
        return false;
      }
      if (!predict_block(2, cx0, cy0, cw, ch, mode, delta, v_pred)) {
        return false;
      }
      TxPlan u_plan;
      TxPlan v_plan;
      if (!compute_tx(1, cx0, cy0, cw, ch, u_pred, u_plan)) {
        error_ = "rdo_loop chroma compute_tx U failed: " + error_;
        return false;
      }
      if (!compute_tx(2, cx0, cy0, cw, ch, v_pred, v_plan)) {
        error_ = "rdo_loop chroma compute_tx V failed: " + error_;
        return false;
      }
      double coefficient_bits = 0.0;
      const TxPlan* plans[2] = {&u_plan, &v_plan};
      for (int plane = 1; plane <= 2; ++plane) {
        const TxPlan& plan = *plans[plane - 1];
        const int area = plan.adj_w * plan.adj_h;
        syntax::build_scan(plan.adj_w, plan.adj_h,
                           std::span<std::uint16_t>(
                               rdo_scan_.data(),
                               static_cast<std::size_t>(area)));
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
        if (!rdo_stream_.init(block)) {
          error_ = "rdo_loop chroma stream init failed";
          return false;
        }
        coefficient_bits += rdo_stream_.block_bits();
      }
      av2enc::still::rdo_decide::ChromaCandidateCost& candidate =
          rdo_chroma_candidates_[static_cast<std::size_t>(rdo_count)];
      candidate.is_cfl = false;
      candidate.mode = av2_y_mode(mode);
      candidate.angle_delta = delta;
      candidate.block_w = cw;
      candidate.block_h = ch;
      candidate.sse_u = static_cast<double>(
          sse_against_source(1, cx0, cy0, cw, ch, u_pred));
      candidate.sse_v = static_cast<double>(
          sse_against_source(2, cx0, cy0, cw, ch, v_pred));
      candidate.syntax_rate_bits =
          av2enc::still::rdo_decide::default_chroma_syntax_bits(
              false, candidate.mode, delta);
      candidate.coefficient_rate_bits = coefficient_bits;
      ++rdo_count;
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
      TxPlan u_plan;
      TxPlan v_plan;
      if (!compute_tx(1, cx0, cy0, cw, ch, u_pred, u_plan) ||
          !compute_tx(2, cx0, cy0, cw, ch, v_pred, v_plan)) {
        error_ = "rdo_loop CfL compute_tx failed: " + error_;
        return false;
      }
      double coefficient_bits = 0.0;
      const TxPlan* plans[2] = {&u_plan, &v_plan};
      for (int plane = 1; plane <= 2; ++plane) {
        const TxPlan& plan = *plans[plane - 1];
        const int area = plan.adj_w * plan.adj_h;
        syntax::build_scan(plan.adj_w, plan.adj_h,
                           std::span<std::uint16_t>(
                               rdo_scan_.data(),
                               static_cast<std::size_t>(area)));
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
        if (!rdo_stream_.init(block)) {
          error_ = "rdo_loop CfL stream init failed";
          return false;
        }
        coefficient_bits += rdo_stream_.block_bits();
      }
      av2enc::still::rdo_decide::ChromaCandidateCost& candidate =
          rdo_chroma_candidates_[static_cast<std::size_t>(rdo_count)];
      candidate.is_cfl = true;
      candidate.mode = kDcPred;
      candidate.angle_delta = 0;
      candidate.block_w = cw;
      candidate.block_h = ch;
      candidate.sse_u = static_cast<double>(
          sse_against_source(1, cx0, cy0, cw, ch, u_pred));
      candidate.sse_v = static_cast<double>(
          sse_against_source(2, cx0, cy0, cw, ch, v_pred));
      candidate.syntax_rate_bits =
          av2enc::still::rdo_decide::default_chroma_syntax_bits(true, kDcPred, 0);
      candidate.coefficient_rate_bits = coefficient_bits;
      ++rdo_count;
    }

    if (rdo_count > 0) {
      const auto choice = av2enc::still::rdo_decide::decide_chroma(
          std::span<const av2enc::still::rdo_decide::ChromaCandidateCost>(
              rdo_chroma_candidates_.data(),
              static_cast<std::size_t>(rdo_count)),
          {}, lambda_config);
      if (choice.found) {
        best.is_cfl = choice.is_cfl;
        best.mode = static_cast<intra::Mode>(choice.mode);
        best.angle_delta = choice.angle_delta;
        best.av2_mode = choice.mode;
        best.sse = static_cast<std::int64_t>(choice.sse_total);
        return true;
      }
    }
    if (best_score_index >= 0) {
      best.mode = modes[static_cast<std::size_t>(best_score_index)];
      best.angle_delta = deltas[static_cast<std::size_t>(best_score_index)];
      best.av2_mode = av2_y_mode(best.mode);
      best.sse = best_sse;
    }
    return true;
  }
'''

TX_TYPE_ANCHOR = "    if (choose_tx_type) {\n"
RDOQ_ANCHOR = "      if (options_.rdoq) {\n"


def replace_once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"patch: expected one {what} anchor, found {count}")
    return text.replace(old, new, 1)


def insert_after(text: str, anchor: str, insertion: str, what: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise SystemExit(
            f"patch: expected one {what} anchor, found {count}")
    return text.replace(anchor, anchor + insertion, 1)


def patch(source: str) -> str:
    if "rdo_loop" in source:
        raise SystemExit("patch: source already contains the loop hook")
    source = insert_after(
        source, INCLUDE_ANCHOR,
        '\n// Experiment-only unified-loop hook; the production M1 owner should\n'
        "// call the same rdo_loop APIs instead of patching a copy.\n"
        + INCLUDE_ADD,
        "include")
    source = insert_after(source, "using partition::PartitionType;\n",
                          HELPERS, "helper insertion")
    source = replace_once(source, MEMBERS_ANCHOR, MEMBERS, "member insertion")

    source = insert_after(source, "  bool choose_luma_mode(int x0, int y0, int w, int h, ModeChoice& best) {\n",
                          LUMA_BRANCH, "luma branch")
    source = replace_once(
        source,
        "  bool choose_uv_mode(int cx0, int cy0, int cw, int ch, int y_mode,\n"
        "                      int y_angle_delta, UvChoice& best) {\n",
        "  bool choose_uv_mode(int cx0, int cy0, int cw, int ch, int y_mode,\n"
        "                      int y_angle_delta, UvChoice& best) {\n"
        + CHROMA_BRANCH,
        "chroma branch")

    encode_anchor = (
        "    if (options_.force_single_block) {\n"
        "      poptions.max_depth = 0;\n"
        "      poptions.min_block_size_px = 64;\n"
        "    }\n")
    source = insert_after(source, encode_anchor, "\n" + ENCODE_BRANCH,
                          "encode branch")

    source = replace_once(
        source, "  std::int64_t block_variance_cost_at(", LOOP_BODY +
        "\n  std::int64_t block_variance_cost_at(", "loop body")

    source = replace_once(source, TX_TYPE_ANCHOR, TX_BRANCH, "TX branch")
    source = replace_once(source, RDOQ_ANCHOR, RDOQ_BRANCH, "RDOQ branch")
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
