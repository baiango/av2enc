#include "codec/entropy/deltaq/video/video_deltaq.h"

#include "codec/partition/tree/partition.h"

namespace av2enc::video::deltaq {
namespace {

void set_error(std::string* error, const char* message) {
  if (error != nullptr) *error = message;
}

av2enc::deltaq::SegmentBlock to_segment_block(
    const syntax::BlockSyntax& node) noexcept {
  av2enc::deltaq::SegmentBlock block;
  block.mi_row = node.mi_row;
  block.mi_col = node.mi_col;
  const auto size = static_cast<partition::BlockSize>(node.bsize);
  block.bw4 = partition::num_4x4_blocks_wide(size);
  block.bh4 = partition::num_4x4_blocks_high(size);
  return block;
}

bool valid_node(const syntax::BlockSyntax& node, std::string* error) {
  const auto size = static_cast<partition::BlockSize>(node.bsize);
  if (!partition::is_valid_block_size(size)) {
    set_error(error, "deltaq: block has an invalid size");
    return false;
  }
  return true;
}

}  // namespace

av2enc::deltaq::FrameQpHeader frame_qp_header(const FrameQpPlan& plan) noexcept {
  av2enc::deltaq::FrameQpHeader out;
  out.primary_ref_is_none = plan.primary_ref_is_none;
  out.enable_ext_seg = plan.enable_ext_seg;
  out.frame_type_switch = false;  // M2 never emits SWITCH_FRAME
  out.using_qmatrix = false;
  out.seq_seg_info_present = plan.seq_seg_info_present;
  out.seq_allow_seg_info_change = plan.seq_allow_seg_info_change;
  out.reuse_seg_info = plan.reuse_seg_info;
  return out;
}

bool begin_frame(FrameQpContext& context, const FrameQpPlan& plan,
                 const VideoFrameHeader& header, int mi_cols, int mi_rows,
                 std::span<const std::uint8_t> prev_segment_ids,
                 std::string* error) {
  context = FrameQpContext{};
  context.plan = plan;
  context.header = frame_qp_header(plan);
  if (header.keyframe && !plan.primary_ref_is_none) {
    set_error(error,
              "deltaq: a key frame cannot have a primary reference");
    return false;
  }

  if (plan.base_q_idx < 1 || plan.base_q_idx > 255) {
    set_error(error, "deltaq: M2 base_q_idx must be in [1, 255]");
    return false;
  }
  if (!av2enc::deltaq::derive_frame_segmentation(
          plan.segmentation, plan.delta_q, plan.base_q_idx, context.header,
          context.segmentation, error)) {
    return false;
  }
  if (!av2enc::deltaq::begin_tile(context.tile_qp, plan.base_q_idx,
                                  plan.delta_q, error)) {
    return false;
  }
  if (!av2enc::deltaq::init_segment_grid(
          context.grid, mi_cols, mi_rows, /*tile_col_start=*/0,
          /*tile_col_end=*/mi_cols, /*tile_row_start=*/0,
          /*tile_row_end=*/mi_rows, plan.enable_ext_seg,
          context.segmentation.num_segments, prev_segment_ids, error)) {
    return false;
  }
  // A segment id used by the plan must be representable.
  for (int i = 0; i < av2enc::deltaq::kMaxSegmentsSpec; ++i) {
    const rc::SegmentFeatures& f =
        plan.segmentation.segments[static_cast<std::size_t>(i)];
    for (int j = 0; j < av2enc::deltaq::kSegLvlMax; ++j) {
      if (f.enabled[static_cast<std::size_t>(j)] &&
          i >= context.segmentation.num_segments) {
        set_error(error, "deltaq: an active segment id is >= MaxSegments");
        return false;
      }
    }
  }
  context.cdfs.init();
  av2enc::deltaq::clear_above_context(context.grid);
  return true;
}

std::size_t write_frame_qp_syntax(BitWriter& writer, const FrameQpPlan& plan,
                                  const VideoFrameHeader& header,
                                  std::string* error) {
  if (header.keyframe && !plan.primary_ref_is_none) {
    if (error != nullptr) {
      *error = "m2 deltaq: a key frame cannot have a primary reference";
    }
    return 0;
  }
  const av2enc::deltaq::FrameQpHeader qp_header = frame_qp_header(plan);
  return av2enc::deltaq::write_segmentation_and_delta_q(
      writer, plan.segmentation, plan.delta_q, plan.base_q_idx, qp_header,
      error);
}

void begin_superblock(FrameQpContext& context, int mi_col) noexcept {
  av2enc::deltaq::begin_superblock(context.tile_qp);
  if (mi_col == 0) {
    av2enc::deltaq::clear_left_context(context.grid);
  }
}

bool mi_size_is_sb(const syntax::BlockSyntax& node, int sb_size) noexcept {
  const auto size = static_cast<partition::BlockSize>(node.bsize);
  if (!partition::is_valid_block_size(size)) return false;
  return partition::num_4x4_blocks_wide(size) == (sb_size >> 2) &&
         partition::num_4x4_blocks_high(size) == (sb_size >> 2);
}

bool write_inter_segment_id_pre(FrameQpContext& context,
                                av2::Encoder& encoder,
                                const syntax::BlockSyntax& node,
                                int segment_id, bool skip_flag,
                                std::string* error) {
  if (!valid_node(node, error)) return false;
  bool coded = false;
  return av2enc::deltaq::write_inter_segment_id(
      encoder, context.cdfs, context.segmentation, context.grid,
      to_segment_block(node), /*pre_skip=*/true, segment_id, skip_flag, coded,
      error);
}

bool write_inter_segment_id_post(FrameQpContext& context,
                                 av2::Encoder& encoder,
                                 const syntax::BlockSyntax& node,
                                 int segment_id, bool skip_flag,
                                 std::string* error) {
  if (!valid_node(node, error)) return false;
  bool coded = false;
  return av2enc::deltaq::write_inter_segment_id(
      encoder, context.cdfs, context.segmentation, context.grid,
      to_segment_block(node), /*pre_skip=*/false, segment_id, skip_flag, coded,
      error);
}

bool write_intra_segment_id(FrameQpContext& context, av2::Encoder& encoder,
                            const syntax::BlockSyntax& node, int segment_id,
                            bool skip_flag, std::string* error) {
  if (!valid_node(node, error)) return false;
  return av2enc::deltaq::write_intra_segment_id(
      encoder, context.cdfs, context.segmentation, context.grid,
      to_segment_block(node), segment_id, skip_flag, error);
}

bool write_delta_qindex(FrameQpContext& context, av2::Encoder& encoder,
                        const syntax::BlockSyntax& node, bool skip_flag,
                        int reduced_delta, std::string* error) {
  if (!valid_node(node, error)) return false;
  // M2 uses 64x64 superblocks (VideoConfig has no superblock-size knob).
  constexpr int kSbSize = 64;
  return av2enc::deltaq::write_delta_qindex(
      encoder, context.cdfs.delta_q, context.tile_qp,
      mi_size_is_sb(node, kSbSize), skip_flag, reduced_delta, error);
}

bool write_superblock_delta_q(FrameQpContext& context, av2::Encoder& encoder,
                              const rc::DeltaQPlanStep& step,
                              const syntax::BlockSyntax& first_block,
                              bool skip_flag, std::string* error) {
  if (!av2enc::deltaq::check_plan_step(context.tile_qp, step, error)) {
    return false;
  }
  if (!write_delta_qindex(context, encoder, first_block, skip_flag,
                          step.reduced_delta, error)) {
    return false;
  }
  if (context.tile_qp.current_q_index != step.qindex_after) {
    set_error(error, "deltaq: plan qindex_after does not match tile state");
    return false;
  }
  return true;
}

bool commit_segment_block(FrameQpContext& context,
                          const syntax::BlockSyntax& node, int segment_id,
                          std::string* error) {
  if (!valid_node(node, error)) return false;
  return av2enc::deltaq::commit_segment_block(context.grid,
                                              to_segment_block(node),
                                              segment_id, error);
}

}  // namespace av2enc::video::deltaq
