#pragma once

#include "core/bitstream/bitwriter.h"
#include "pipeline/video/post/video_filter_plan.h"
#include "pipeline/video/video_gm_types.h"

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
  // §5.4.1 temporal layers.  The hierarchical-P mode (--video-tlayers) raises
  // max_tlayer_id and turns monotonic_output_order_flag off so frame OBUs can
  // carry obu_tlayer_id and code implicit_output_frame.  Defaults keep the
  // single-layer byte-identical stream.
  std::uint8_t max_tlayer_id = 0;
  bool monotonic_output_order_flag = true;
  bool tlayer_dependency_present_flag = false;
  // tlayer_dependency[curr][ref] for curr = 1..max_tlayer_id.
  bool tlayer_dependency[4][4] = {{false, false, false, false},
                                  {false, false, false, false},
                                  {false, false, false, false},
                                  {false, false, false, false}};
  // §5.4.6 compound tools.  --video-compound enables reference_select per
  // frame; the encoder only signals the AVERAGE (comp_group_idx == 0) blend, so
  // masked compound and CWP stay off in the sequence header.
  bool enable_masked_compound = false;
  bool enable_cwp = false;
  bool enable_imp_msk_bld = false;
  std::uint8_t num_same_ref_compound = 0;
  bool disable_cdf_update = true;
  // §5.18.7.9 deblocking_filter_params(): when enabled the frame header codes
  // all four apply_deblocking_filter flags with DfDeltaQ = 0 (df_par_bits =
  // 2 from the sequence header).  Off keeps the historical all-zero row.
  bool enable_deblock = false;
  // §5.18.8.1 read_tx_mode(): opt-in TX_MODE_SELECT (--video-tx-search).  Off
  // keeps the historical tx_mode_select = 0 / TX_MODE_LARGEST bitstream.
  bool tx_mode_select = false;
  // §5.4.8 average-CDF config.  Opt-in (--avg-cdf); both bits are written only
  // in the non-single-picture header, so the pinned --video stream is
  // unchanged while the flags stay off.
  bool enable_avg_cdf = false;
  bool avg_cdf_type = true;
  // §5.4.1 timing block, non-single-picture only (--video-timing).  This is
  // the pinned-decoder-compatible form: decoder_model_info_present_flag = 1
  // with num_units_in_decoding_tick, and seq_decoder_model_info_present_flag
  // = 0 (pinned dav2d f9dafbc does not parse the nested §5.4.13 block).
  bool initial_display_delay_present = false;
  std::uint8_t initial_display_delay_minus_1 = 0;
  bool decoder_model_info_present = false;
  std::uint32_t num_units_in_decoding_tick = 0;
  // §5.7 multi-frame header: 0 = no MFH (seq_header_id_in_frame_header is
  // coded); >0 makes every frame header reference that MFH and omit the
  // sequence-header id.  Opt-in via --video-mfh.
  std::uint32_t cur_mfh_id = 0;
  // §5.4.6 DRL reordering.  Default is DRL_REORDER_DISABLED.
  bool disable_drl_reorder = true;
  bool constrain_drl_reorder = true;
  // Sequence-level intra tool used by the M1 keyframe tile payload and by the
  // M2 intra fallback blocks.
  bool enable_cfl_intra = true;
  // §5.4.10 sequence_filter_config() flags for the opt-in M2 loop filters.
  // All default to the historical values, so a default stream keeps its bytes.
  bool enable_cdef = false;
  bool cdef_on_skip_txfm_always_on = false;
  bool cdef_on_skip_txfm_disabled = true;
  // CdefOnSkipTxfm mode the frame header derives: 0 DISABLED, 1 ALWAYS_ON,
  // 2 ADAPTIVE.  Mirrors the pair above.
  int cdef_on_skip_txfm_mode = 0;
  // §5.4.10 CCSO flags.  Off by default; `ccso_unit_matches_sb_size` is only
  // coded when `enable_ccso` is set.
  bool enable_ccso = false;
  bool ccso_unit_matches_sb_size = true;
  // §5.4.10 loop-restoration flags (non-separable Wiener / chroma LR).
  // §5.4.10 GDF flags.  Header-only mode (gdf_per_block = 0).
  bool enable_gdf = false;
  bool gdf_unit_matches_sb_size = true;
  bool enable_restoration = false;
  bool lr_disable_luma_pc_wiener = true;
  bool lr_disable_luma_wiener_ns = false;
  bool lr_uv_present = false;
  bool lr_disable_chroma_wiener_ns = false;
  // §5.4.1 layer limits for the SVC path.  The defaults reproduce the
  // single-layer header byte-for-byte (seq_lcr_id = 0, M = T = 1).
  std::uint8_t seq_lcr_id = 0;          // f(3), 1..7 with a global LCR
  std::uint8_t max_mlayer_id = 0;       // f(3)
  std::uint8_t seq_max_mlayer_cnt = 1;  // seq_max_mlayer_cnt_minus_1 + 1
  // Dependency-present flags.  The maps written are the spec defaults
  // (ref <= curr), so the bits are explicit but semantically inert.
  bool mlayer_dependency_present = true;
  // §5.4.6 enable_global_motion (opt-in --video-gme).  Off keeps the
  // historical sequence header and makes global_motion_params() a no-op.
  bool enable_global_motion = false;
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
  bool implicit_output_frame = false;
  // §5.18.2 reference_select: 1 lets a block code comp_mode and two refs.
  bool reference_select = false;
  // §5.18.2 cross-frame CDF initialisation.  The defaults reproduce the
  // historical inter header (signal off, reset on).  The continuity path
  // signals primary_ref_frame = 0 (ref_frame_idx[0]) and clears the reset bit.
  bool signal_primary_ref_frame = false;
  bool disable_cross_frame_cdf_init = true;
  std::uint8_t primary_ref_frame = 0;
  // §5.18.9 global motion for this frame.  Defaults to "no model", which
  // writes `use_global_motion = 0` when the sequence bit is enabled and writes
  // nothing when it is disabled.
  GmFramePlan gm{};
  // §5.18.7.10 cdef_params() for this frame.  `enabled` is the
  // cdef_frame_enable bit; the strengths are only coded when it is set.
  post::VideoCdefPlan cdef;
  // §5.18.7.12 ccso_params() for this frame.
  post::VideoCcsoPlan ccso;
  // §5.18.7.11 lr_params() for this frame.
  post::VideoLrPlan lr;
  // §5.18.7.9 gdf_params() for this frame.
  post::VideoGdfPlan gdf;
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
// exit_symbol() trailing bits.  `obu_tlayer_id` is the §5.2.2 layer id of the
// frame OBU (0 for a single-layer stream and always 0 for key/switch/RAS).
std::vector<std::uint8_t> make_frame_obu(const VideoConfig& config,
                                         const VideoFrameHeader& header,
                                         std::span<const std::uint8_t> tile_payload,
                                         std::uint8_t obu_tlayer_id = 0);

// Empty temporal delimiter OBU.
std::vector<std::uint8_t> make_temporal_delimiter_obu();

}  // namespace av2enc::video
