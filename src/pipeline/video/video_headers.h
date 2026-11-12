#pragma once

#include "core/bitstream/bitwriter.h"

#include <cstdint>
#include <span>
#include <vector>

// AV2 v1.0.0 OBU/header writer for the M2 P-frame path.
//
// This is deliberately a *new* writer rather than an edit to the M1 header
// writer: M1 is a single still picture (single_picture_header_flag = 1), while
// a P-frame stream needs the multi-frame sequence header, order hints,
// reference slots and the inter-frame header fields.  The keyframe tile
// payload is still produced by the M1 still encoder and wrapped here.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §5.4.1, §5.4.6, §5.18.2, §5.19.
namespace av2enc::video {

struct VideoConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // level 5.1
  std::uint8_t base_q_idx = 90;    // §8.3.1 qctx slice from raw base_q_idx
  // §5.4.6 order_hint_bits_minus_1 + 1. Four bits covers a 16-frame GOP.
  std::uint8_t order_hint_bits = 4;
  // §5.4.6 explicit_num_ref_frames / num_ref_frames_minus_1.
  std::uint8_t num_ref_frames = 2;
  // §5.18.2 frame_explicit_ref_frame_map / num_total_refs.
  std::uint8_t num_total_refs = 1;
  // §5.4.6 seq_max_drl_bits_minus_1. m = value + 1 DRL indices are codable;
  // 4 -> m = 5, enough for the six-entry spatial stack (index 5 is the
  // read_drl_idx() exhaustion value).
  std::uint8_t max_drl_bits_minus_1 = 4;
  bool disable_cdf_update = true;
  // §5.4.6 DRL reordering.  Default is DRL_REORDER_DISABLED.
  bool disable_drl_reorder = true;
  bool constrain_drl_reorder = true;
  // Sequence-level intra tool used by the M1 keyframe tile payload and by the
  // M2 intra fallback blocks.
  bool enable_cfl_intra = true;
};

struct VideoFrameHeader {
  bool keyframe = false;
  // Coded order_hint. The key frame is 0; P frames increment by one.
  std::uint32_t order_hint = 0;
  // Reference slot used by every block in this frame (single reference).
  std::uint8_t ref_slot = 0;
  // §5.18.2 explicit reference map.  For num_total_refs > 1 the slots are
  // listed newest first; every block then selects one with single_ref syntax.
  std::uint8_t num_total_refs = 1;
  std::uint8_t ref_frame_idx[7] = {0, 0, 0, 0, 0, 0, 0};
  // §5.18.2 frame MV precision.  Defaults are the M2 1/8-pel pair.
  bool use_qtr_precision_mv = false;
  bool allow_high_precision_mv = true;
  // §5.18.2 refresh_frame_flags. The key frame refreshes all slots.
  std::uint32_t refresh_frame_flags = 1;
  // §5.18.2 immediate_output_frame.  Clearing it codes a hidden picture, which
  // a later show-existing-frame OBU can re-show (derive_sef_order_hint = 1).
  // monotonic_output_order_flag = 1 means implicit_output_frame stays absent.
  bool immediate_output_frame = true;
};

// sequence_header_obu() for single_picture_header_flag = 0, wrapped as an
// Annex B length-delimited OBU. The sequence enables only the tools the M1
// still tile payload and the M2 inter tile payload use.
std::vector<std::uint8_t> make_sequence_header_obu(const VideoConfig& config);

// frame_header_info() syntax bits for one frame, without the leading
// is_first_tile_group bit and without byte_alignment(). Returns the number of
// bits written.
std::size_t write_frame_header(BitWriter& writer, const VideoConfig& config,
                               const VideoFrameHeader& header);

// §5.19 tile_group_obu() header for a single tile:
//   is_first_tile_group f(1)
//   frame_header(isFirst = 1)
//   byte_alignment()
// The caller appends the tile payload.
std::size_t write_tile_group_header(BitWriter& writer, const VideoConfig& config,
                                    const VideoFrameHeader& header);

// Complete OBU_CLOSED_LOOP_KEY / OBU_REGULAR_TILE_GROUP payload, including
// the tile payload. The tile payload must already contain the §8.2.4
// exit_symbol() trailing bits.
std::vector<std::uint8_t> make_frame_obu(const VideoConfig& config,
                                         const VideoFrameHeader& header,
                                         std::span<const std::uint8_t> tile_payload);

// Empty temporal delimiter OBU.
std::vector<std::uint8_t> make_temporal_delimiter_obu();

}  // namespace av2enc::video
