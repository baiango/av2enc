#include "pipeline/video/video_frame_internal.h"

namespace av2enc::video {

bool InterFrameEncoder::write(std::vector<std::uint8_t>& tile_payload,
           std::vector<syntax::BlockSyntax>& blocks,
           std::vector<std::int32_t>& levels,
           const post::VideoFilterPlan* filters, std::string& error) {
  syntax::FramePlan plan;
  plan.width = width_;
  plan.height = height_;
  plan.sb_size = sb_size_;
  plan.max_pb_aspect_ratio = geometry_.max_pb_aspect_ratio;
  plan.frame_mi_cols = frame_mi_cols_;
  plan.frame_mi_rows = frame_mi_rows_;
  plan.tile_mi_row = tile_mi_row_;
  plan.tile_mi_col = tile_mi_col_;
  plan.census_frame = census_frame_;
  plan.inter_frame = !keyframe_;
  plan.enable_sdp = options_.enable_sdp;
  plan.frame_sdp = keyframe_ && options_.enable_sdp;
  plan.tx_mode_select = options_.enable_tx_search;
  plan.reduced_tx_part_set = false;
  plan.reduced_tx_set = 0;
  plan.allow_tcq = options_.enable_tcq;
  plan.enable_cctx = options_.enable_cctx;
  plan.num_total_refs = num_total_refs_;
  plan.reference_select = options_.compound && num_total_refs_ > 1;
  plan.enable_cwp = options_.enable_cwp_weighted;
  plan.allow_bawp = (options_.enable_bawp || options_.enable_morph) &&
                    !keyframe_;
  // §5.4.7/§5.18.3.3/.4 screen-content tools.  The palette above/left cache
  // and the IBC BV ring are tile-local, so only the single-tile writer.
  // --video-scc-force extends the tools to the inter frames: the forced
  // sequence form infers allow_screen_content_tools = 1 there, and every
  // eligible intra leaf then codes palette_mode_info() (has_palette_y = 0).
  const bool scc_carrier =
      options_.tile_cols == 0 && options_.tile_rows == 0;
  plan.allow_screen_content_tools =
      scc_carrier &&
      (options_.seq_force_screen_content_tools ||
       (keyframe_ && (options_.enable_morph || options_.enable_palette ||
                      options_.enable_ibc)));
  plan.allow_intrabc =
      keyframe_ && scc_carrier &&
      (options_.enable_morph || options_.enable_ibc);
  plan.force_integer_mv = true;
  plan.frame_force_integer_mv = options_.seq_force_integer_mv;
  // §5.20.5.4 reads morph_pred inside read_intrabc_info() whenever
  // allow_screen_content_tools && enable_bawp, independent of --video-morph.
  // So any IBC carrier on a BAWP sequence codes the symbol; without the
  // morph fit it is always 0.
  plan.morph_pred_present =
      plan.allow_intrabc && (options_.enable_bawp || options_.enable_morph);
  plan.palettes = std::move(palettes_);
  plan.enable_interintra = !keyframe_ && options_.enable_interintra;
  plan.enable_masked_compound =
      plan.reference_select &&
      (options_.compound_wedge || options_.compound_diffwtd);
  plan.opfl_refine_type = plan.reference_select ? options_.opfl_refine_type : 0;
  plan.enable_refinemv = plan.reference_select && options_.enable_refinemv;
  plan.mv_precision =
      options_.seq_force_integer_mv ? 3 : options_.mv_precision;
  plan.enable_adaptive_mvd = options_.enable_adaptive_mvd;
  plan.enable_mvd_sign_derive = options_.enable_mvd_sign_derive;
  plan.enable_flex_mvres = options_.enable_flex_mvres;
  plan.drl_reorder = options_.drl_reorder
                         ? (options_.drl_reorder_always ? 2 : 1)
                         : 0;
  plan.frame_distance = frame_distance_;
#if defined(AV2ENC_VIDEO_DELTAQ)
  plan.segmentation = seg_params_;
  plan.delta_q = dq_params_;
  plan.primary_ref_is_none = primary_ref_is_none_;
  plan.delta_steps = sb_delta_steps_;
  plan.enable_ext_seg = options_.enable_ext_seg;
  plan.prev_segment_ids = prev_segment_ids_;
#endif
  plan.temporal.use_ref_frame_mvs = temporal_ctx_.use_ref_frame_mvs;
  plan.temporal.order_hint = temporal_ctx_.order_hint;
  plan.temporal.num_past_refs = temporal_ctx_.num_past_refs;
  plan.temporal.num_future_refs = temporal_ctx_.num_future_refs;
  plan.temporal.closest_past = temporal_ctx_.closest_past;
  plan.temporal.closest_future = temporal_ctx_.closest_future;
  plan.temporal.ref_order_hints = ref_order_hints_;
  plan.temporal.field = temporal_ctx_.field;
  plan.temporal.mv_traj = temporal_ctx_.mv_traj;
  plan.temporal.scan = temporal_ctx_.use_ref_frame_mvs
                           ? &video_temporal_scan
                           : nullptr;
  plan.tip_as_ref = temporal_ctx_.tip_as_ref;
  plan.tip_weight_index = temporal_ctx_.tip_weight_index;
  plan.tip_allow_hole_fill = temporal_ctx_.tip_allow_hole_fill;
#if defined(AV2ENC_VIDEO_GME)
  if (gm_plan_ != nullptr) {
    plan.gm_num_refs = gm_plan_->num_refs;
    plan.gm_params = gm_plan_->params;
  }
#endif
  VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width_);
  config.max_frame_height = static_cast<std::uint16_t>(height_);
  config.use_128x128_superblock = options_.sb128;
  config.use_256x256_superblock = options_.sb256;
  config.enable_ext_partitions = options_.ext_partitions;
  config.enable_uneven_4way_partitions = options_.ext_partitions;
  config.reduce_pb_aspect_ratio = options_.reduce_pb_aspect_ratio;
  config.max_pb_aspect_ratio_log2_minus_1 =
      options_.max_pb_aspect_ratio_log2_minus_1;
  config.base_q_idx = options_.base_q_idx;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.enable_refmvbank = options_.refmvbank && num_total_refs_ > 0;
  config.num_same_ref_compound =
      (options_.same_ref_compound && num_total_refs_ > 1) ? 1U : 0U;
  config.disable_cdf_update = !options_.adaptive_cdf;
  config.minimal_tile_trailer = options_.minimal_tile_trailer;
  config.enable_avg_cdf = options_.avg_cdf;
  config.avg_cdf_type = true;
  config.enable_cfl_intra = true;
  config.enable_mhccp = options_.enable_mhccp;
  config.enable_ibp = options_.enable_ibp;
  config.enable_idtx_intra = options_.enable_idtx_intra;
  config.enable_mrls = options_.enable_mrl;
  config.enable_dip = options_.enable_dip;
  config.enable_intra_ist = options_.enable_ist;
  config.enable_inter_ist = false;
  config.enable_tcq = options_.enable_tcq;
  config.choose_tcq_per_frame = options_.enable_tcq;
  config.enable_cctx = options_.enable_cctx;
  config.enable_sdp = options_.enable_sdp;
  config.enable_extended_sdp = false;
  config.enable_inter_ddt = options_.enable_inter_ddt;
  config.using_qmatrix = options_.enable_qm;
  config.qm_user_data = options_.user_qm;
  config.enable_deblock = options_.enable_deblock;
  config.tx_mode_select = options_.enable_tx_search;
  config.enable_bawp = options_.enable_bawp || options_.enable_morph;
  config.enable_adaptive_mvd = options_.enable_adaptive_mvd;
  config.enable_mvd_sign_derive = options_.enable_mvd_sign_derive;
  config.enable_flex_mvres = options_.enable_flex_mvres;
  config.seq_screen_content_tools_present =
      !options_.seq_force_screen_content_tools && keyframe_ &&
      (options_.enable_morph || options_.enable_palette ||
       options_.enable_ibc);
  config.seq_force_screen_content_tools =
      options_.seq_force_screen_content_tools;
  config.seq_force_integer_mv = options_.seq_force_integer_mv;
  config.enable_interintra = options_.enable_interintra;

  std::size_t tile_bytes = 0;
  if (!syntax::assemble_video_frame(plan, blocks_, levels_, config,
                                 tile_payload, error, &tile_bytes,
                                 cdf_seed_, cdf_final_, filters)) {
    return false;
  }
  blocks = std::move(blocks_);
  levels = std::move(levels_);
  return true;
}

bool InterFrameEncoder::encode(std::vector<std::uint8_t>& tile_payload,
            std::vector<syntax::BlockSyntax>& blocks,
            std::vector<std::int32_t>& levels, std::string& error) {
  if (!decide(error)) return false;
  if (!apply_tip_block_ref(error)) return false;
  if (!deblock(error)) return false;
  if (!apply_loop_filters(error)) return false;
  return write(tile_payload, blocks, levels,
               filter_plan_.any() ? &filter_plan_ : nullptr, error);
}

const std::vector<syntax::BlockSyntax>& InterFrameEncoder::blocks() const {
  return blocks_;
}

const std::vector<std::uint8_t>& InterFrameEncoder::segment_map() const {
  return enc_seg_grid_.current;
}

const Plane& InterFrameEncoder::recon(int plane) const { return recon_[plane]; }

const std::vector<inter::mc::MiInfo>& InterFrameEncoder::mv_grid() const { return mv_grid_; }

const GmFramePlan* InterFrameEncoder::gm_plan() const { return gm_plan_; }

int InterFrameEncoder::mv_mi_cols() const { return mi_cols_; }

int InterFrameEncoder::mv_mi_rows() const { return mi_rows_; }

int InterFrameEncoder::recon_mi_cols() const { return mi_cols_; }

int InterFrameEncoder::recon_mi_rows() const { return mi_rows_; }

std::uint32_t InterFrameEncoder::inter_blocks() const { return inter_blocks_; }

std::uint32_t InterFrameEncoder::intra_blocks() const { return intra_blocks_; }

int InterFrameEncoder::temporal_added() const { return temporal_added_; }

int InterFrameEncoder::temporal_selected() const { return temporal_selected_; }

int InterFrameEncoder::same_ref_blocks() const { return same_ref_blocks_; }

int InterFrameEncoder::refmvbank_added() const { return refmvbank_added_; }

int InterFrameEncoder::refmvbank_selected() const { return refmvbank_selected_; }

int InterFrameEncoder::mv_traj_derived() const { return mv_traj_derived_; }

int InterFrameEncoder::amvd_selected() const { return amvd_selected_; }

int InterFrameEncoder::flex_precision_selected() const { return flex_precision_selected_; }

int InterFrameEncoder::derived_sign_selected() const { return derived_sign_selected_; }

int InterFrameEncoder::tip_blocks() const { return tip_blocks_; }

const post::VideoFilterPlan& InterFrameEncoder::filter_plan() const { return filter_plan_; }

ChromaInfo InterFrameEncoder::chroma_step(const PartitionNode& node,
                              const ChromaInfo& inherited) {
  if (node.has_chroma && !node.chroma_offset) {
    return ChromaInfo{true, node.mi_row, node.mi_col, node.size};
  }
  return inherited;
}

}  // namespace av2enc::video
