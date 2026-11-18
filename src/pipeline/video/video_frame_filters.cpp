#include "pipeline/video/video_frame_internal.h"

namespace av2enc::video {

bool InterFrameEncoder::apply_loop_filters(std::string& error) {
  // Opt-in M2 loop filters (§7.18–§7.20).  They run on the completed
  // reconstruction, before it becomes the output picture and the reference
  // for later frames.  The plan is passed to the tile writer so the
  // per-superblock / per-block symbols match the encoder decisions.
  filter_plan_ = post::VideoFilterPlan{};
  if (options_.loop_filters.any()) {
    if (width_ != static_cast<int>(source_.width) ||
        height_ != static_cast<int>(source_.height)) {
      error = "video loop filters need the single-tile writer";
      return false;
    }
    post::VideoLoopFilterFrame lf;
    lf.width = width_;
    lf.height = height_;
    lf.mi_cols = mi_cols_;
    lf.mi_rows = mi_rows_;
    lf.frame_is_intra = keyframe_;
    lf.base_q_idx = options_.base_q_idx;
    lf.num_total_refs = num_total_refs_;
    lf.restricted_oh = -1;
    for (int r = 0; r < 2; ++r) {
      lf.ref_order_hints[r] = ref_order_hints_[static_cast<std::size_t>(r)];
      lf.frame_distance[r] = frame_distance_[static_cast<std::size_t>(r)];
    }
    // §7.18.1 Skips[]: derived from every coded inter leaf's skip_flag.  The
    // decoder derives the same array, and with CdefOnSkipTxfm DISABLED it
    // leaves a fully-skipped 8x8 block unfiltered.
    std::vector<post::VideoLoopFilterBlock> filter_blocks;
    {
      filter_blocks.reserve(blocks_.size());
      for (const syntax::BlockSyntax& leaf : blocks_) {
        if (leaf.partition !=
            static_cast<std::uint8_t>(PartitionType::kNone)) {
          continue;
        }
        post::VideoLoopFilterBlock block;
        block.mi_row = leaf.mi_row;
        block.mi_col = leaf.mi_col;
        block.w4 =
            partition::block_width(
                static_cast<partition::BlockSize>(leaf.bsize)) >>
            2;
        block.h4 =
            partition::block_height(
                static_cast<partition::BlockSize>(leaf.bsize)) >>
            2;
        // §7.18.1 Skips[]: an inter skip leaf and an IBC skip leaf both code
        // skip_flag = 1; with CdefOnSkipTxfm DISABLED the decoder leaves a
        // fully skipped 8x8 block unfiltered.
        block.skip = (leaf.is_inter && leaf.skip_flag != 0) ||
                     (leaf.flags & syntax::kFlagIsIbc) != 0;
        vf_detail::fill_filter_block_tx(leaf, &block);
        filter_blocks.push_back(block);
      }
    }
    lf.blocks = filter_blocks;
    PaddedSourcePlanes padded;
    pad_source_420(source_, mi_cols_, mi_rows_, &padded);
    for (int plane = 0; plane < 3; ++plane) {
      lf.plane[plane] = recon_[plane].samples.data();
      lf.stride[plane] = recon_[plane].width;
      lf.source[plane] = padded.data[plane];
      lf.source_stride[plane] = padded.stride[plane];
    }
    if (!post::run_video_loop_filters(lf, options_.loop_filters,
                                      &filter_plan_, nullptr, &error)) {
      return false;
    }
  }

  return true;
}

bool InterFrameEncoder::deblock(std::string& error) {
  if (!options_.enable_deblock) return true;
  // reference.  The metadata comes from the real per-leaf decisions above.
  // Multi-tile deblocking is rejected at the sequence level (the pinned
  // decoder filters across interior tile edges; see docs/negative-results.md).
  if (options_.enable_deblock) {
    const bool tile_local =
        (tile_w_ != 0 && tile_w_ != static_cast<int>(source_.width)) ||
        (tile_h_ != 0 && tile_h_ != static_cast<int>(source_.height));
    if (tile_local) {
      error = "video deblock: per-tile deblocking is not supported";
      return false;
    }
    post::VideoDeblockGrid grid;
    if (!post::build_video_deblock_grid(blocks_, mi_cols_, mi_rows_,
                                        options_.base_q_idx, keyframe_, &grid,
                                        &error, sub_pus_.data(),
                                        static_cast<int>(sub_pus_.size()))) {
      return false;
    }
    // §5.18.2 / §7.17.2: when the frame codes allow_df_sub_pu = 1 the
    // deblocker filters the §7.13.3.15 sub-PU edges.  The grid already
    // carries SubPuSize/SubPuColBase/SubPuRowBase for every leaf.
    grid.options.allow_df_sub_pu = options_.enable_df_sub_pu;
    // §7.17.6: chroma filter strength uses the effective UV AC delta
    // (DeltaQU/VAc + BaseUVAcDeltaQ).  Omitting it desynchronises the
    // decoder's chroma reconstruction even though the tile bytes are
    // unchanged (docs/negative-results.md).
    if (uv_delta_enabled_) {
      grid.options.delta_q_u_ac = resolved_delta_q_.u_ac;
      grid.options.delta_q_v_ac = resolved_delta_q_.v_ac;
      grid.options.base_uv_ac_delta_q =
          options_.seq_delta_q.base_uv_ac_delta_q;
    }
    std::uint8_t* data[3] = {recon_[0].samples.data(),
                             recon_[1].samples.data(),
                             recon_[2].samples.data()};
    const int stride[3] = {recon_[0].stride, recon_[1].stride,
                           recon_[2].stride};
    if (!post::apply_video_deblock_host(grid, data, stride, &error)) {
      return false;
    }
  }
  return true;
}

}  // namespace av2enc::video
