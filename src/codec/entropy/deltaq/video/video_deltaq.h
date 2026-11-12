// M2-specific integration of the delta-Q / segmentation bitstream module.
//
// The normative syntax lives in src/encoder/m1/deltaq/ (namespace
// av2enc::deltaq).  This adapter only supplies the M2 frame/block glue:
//
//   * VideoFrameHeader.keyframe -> DerivedPrimaryRefFrame == PRIMARY_REF_NONE
//     (key frame: update_map inferred 1, temporal_update inferred 0, no bits);
//   * §5.20.7.1 inter_frame_mode_info() call order:
//         inter_segment_id(1)      <-- before read_is_inter()
//         read_is_inter()
//         read_skip()
//         inter_segment_id(0)      <-- only when SegIdPreSkip == 0
//         read_delta_qindex()
//     (M2 disables GDF/CDEF/CCSO, so read_delta_qindex() follows directly.)
//   * §5.20.5.2 intra_frame_mode_info() call order for key-frame blocks:
//         intra_segment_id()
//         read_delta_qindex()      <-- after the skipped cdef/ccso hooks
//   * BlockSyntax -> SegmentBlock (mi_row/mi_col and the 4x4 extent), and
//     MiSize == SbSize from the block size and the frame's superblock size.
//
// The caller keeps the ordering; this adapter only refuses calls that would
// desynchronise the stream (for example a post-skip segment id when
// SegIdPreSkip == 1, or a delta step whose qindex_before does not match).
#pragma once

#include "codec/entropy/deltaq/still/deltaq.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_syntax.h"
#include "codec/rate/rc/rc.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace av2enc::video::deltaq {

// Per-frame quantizer/segmentation decision, normally produced by
// src/encoder/rc/ and the pipeline planner.
struct FrameQpPlan {
  int base_q_idx = 90;
  rc::DeltaQParams delta_q{};
  rc::SegmentationParams segmentation{};
  bool enable_ext_seg = false;
  // DerivedPrimaryRefFrame == PRIMARY_REF_NONE.  This is NOT simply
  // `VideoFrameHeader.keyframe`: choose_primary_secondary_ref_frame() only
  // considers references whose frame type is INTER_FRAME.  M2's first P frame
  // references a key frame, so its derived primary reference is NONE and
  // segmentation_update_map / segmentation_temporal_update are inferred
  // (update_map = 1, temporal_update = 0, no bits).  A P frame that references
  // an earlier P frame must set this to false.
  bool primary_ref_is_none = true;
};

// One frame's derived syntax state plus the tile state that must live for the
// whole tile.  M2 emits one tile per frame, so one object per frame.
struct FrameQpContext {
  FrameQpPlan plan{};
  av2enc::deltaq::FrameQpHeader header{};
  av2enc::deltaq::FrameSegmentation segmentation{};
  av2enc::deltaq::TileQp tile_qp{};
  av2enc::deltaq::SegmentGrid grid{};
  av2enc::deltaq::BlockCdfs cdfs{};
};

// choose_primary_secondary_ref_frame() returns PRIMARY_REF_NONE iff no
// reference slot has RefFrameType == INTER_FRAME.  Callers that know the
// reference frame types should use this instead of `VideoFrameHeader.keyframe`.
constexpr bool primary_ref_is_none_for(bool has_inter_reference) noexcept {
  return !has_inter_reference;
}

// FrameQpHeader from the plan.  See FrameQpPlan::primary_ref_is_none for why
// this is not simply the M2 keyframe flag.
av2enc::deltaq::FrameQpHeader frame_qp_header(const FrameQpPlan& plan) noexcept;

// Builds the frame state, validates the plan, sets CurrentQIndex to
// base_q_idx (tile start), installs PrevSegmentIds and clears
// AboveSegPredContext.  `prev_segment_ids` may be empty (all zero); otherwise
// it must be mi_cols * mi_rows entries.
bool begin_frame(FrameQpContext& context, const FrameQpPlan& plan,
                 const VideoFrameHeader& header, int mi_cols, int mi_rows,
                 std::span<const std::uint8_t> prev_segment_ids,
                 std::string* error = nullptr);

// Writes the three frame-header pieces in normative order.  The caller has
// already written the M2 frame-header prefix up to and including base_q_idx
// and continues with deblocking_filter_params() afterwards.
std::size_t write_frame_qp_syntax(BitWriter& writer,
                                  const FrameQpPlan& plan,
                                  const VideoFrameHeader& header,
                                  std::string* error = nullptr);

// §5.20.2.1: ReadDeltas = delta_q_present and clear_left_context() for the
// segment-prediction arrays.  Call once at each superblock, before the first
// block of that superblock; `mi_col == 0` is the superblock-row start.
void begin_superblock(FrameQpContext& context, int mi_col) noexcept;

// §5.20.7.2 inter_segment_id(preSkip = 1) for one block.  This is the no-op
// call when SegIdPreSkip == 0.
bool write_inter_segment_id_pre(FrameQpContext& context,
                                av2::Encoder& encoder,
                                const syntax::BlockSyntax& node,
                                int segment_id, bool skip_flag,
                                std::string* error = nullptr);

// §5.20.7.2 inter_segment_id(preSkip = 0).  Must not be called when
// SegIdPreSkip == 1.
bool write_inter_segment_id_post(FrameQpContext& context,
                                 av2::Encoder& encoder,
                                 const syntax::BlockSyntax& node,
                                 int segment_id, bool skip_flag,
                                 std::string* error = nullptr);

// §5.20.5.7 intra_segment_id() for a key-frame block.
bool write_intra_segment_id(FrameQpContext& context, av2::Encoder& encoder,
                            const syntax::BlockSyntax& node, int segment_id,
                            bool skip_flag, std::string* error = nullptr);

// §5.20.5.11 read_delta_qindex() at the first non-full-skip block of a
// superblock.  `reduced_delta` is the planner's coded value; it is ignored
// when this block does not code a delta.
bool write_delta_qindex(FrameQpContext& context, av2::Encoder& encoder,
                        const syntax::BlockSyntax& node, bool skip_flag,
                        int reduced_delta, std::string* error = nullptr);

// Writes the delta for the superblock at (sb_x, sb_y) using an RC plan step.
// Verifies that the tile state matches the plan's qindex_before and that the
// achieved qindex matches qindex_after.
bool write_superblock_delta_q(FrameQpContext& context,
                              av2::Encoder& encoder,
                              const rc::DeltaQPlanStep& step,
                              const syntax::BlockSyntax& first_block,
                              bool skip_flag, std::string* error = nullptr);

// Fills SegmentIds for this block after its mode info.  Call after both
// segment-id calls (or after the single intra call).
bool commit_segment_block(FrameQpContext& context,
                          const syntax::BlockSyntax& node, int segment_id,
                          std::string* error = nullptr);

// MiSize == SbSize for M2's 64x64 superblocks.
bool mi_size_is_sb(const syntax::BlockSyntax& node, int sb_size) noexcept;

}  // namespace av2enc::video::deltaq
