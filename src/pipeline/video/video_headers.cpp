#include "pipeline/video/video_headers.h"

#include "core/bitstream/basic_headers.h"

#include <cassert>

namespace av2enc::video {
namespace {

using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
    assert(leading_zeros < 32U);
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
  assert(n >= 2U);
  assert(value < n);
  unsigned w = 0;
  while ((1U << w) < n) {
    ++w;
  }
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

std::uint32_t ceil_log2(std::uint32_t value) {
  if (value <= 1U) {
    return 0U;
  }
  std::uint32_t bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

std::uint32_t mi_cols(const VideoConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
}
std::uint32_t mi_rows(const VideoConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
}

void write_tile_info(BitWriter& writer, const VideoConfig& c) {
  const std::uint32_t sb_cols = (mi_cols(c) + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows(c) + 15U) >> 4U;
  assert(sb_cols >= 1U && sb_rows >= 1U);
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) {
    writer.put_bit(false);  // increment_tile_cols_log2 = 0
  }
  if (sb_rows > 1U) {
    writer.put_bit(false);  // increment_tile_rows_log2 = 0
  }
}

std::size_t write_sequence_header(BitWriter& writer, const VideoConfig& c) {
  const std::size_t start = writer.bit_size();
  assert(c.order_hint_bits >= 1U && c.order_hint_bits <= 16U);
  assert(c.num_ref_frames >= 1U && c.num_ref_frames <= 16U);
  assert(c.max_drl_bits_minus_1 < 5U);  // MAX_REF_MV_STACK_SIZE - 1 == 5

  put_uvlc(writer, 0);                 // seq_header_id
  writer.put_bits(0U, 5U);             // seq_profile_idc
  writer.put_bit(false);               // single_picture_header_flag = 0
  writer.put_bits(c.seq_level_idx, 5U);  // seq_level_idx
  writer.put_bit(false);               // seq_tier (level > 3)
  put_uvlc(writer, 0);                 // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                 // bit_depth_idc = 1 => 8-bit

  // §5.4.1 non-single-picture header.
  writer.put_bits(0U, 3U);  // seq_lcr_id
  writer.put_bit(false);    // still_picture
  writer.put_bits(0U, 2U);  // max_tlayer_id
  writer.put_bits(0U, 3U);  // max_mlayer_id
  // SeqMaxMlayerCnt = 1 (max_mlayer_id == 0).
  writer.put_bit(true);     // monotonic_output_order_flag

  unsigned width_bits =
      bit_width(static_cast<std::uint32_t>(c.max_frame_width) - 1U);
  if (width_bits == 0) width_bits = 1;
  unsigned height_bits =
      bit_width(static_cast<std::uint32_t>(c.max_frame_height) - 1U);
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_width) - 1U,
                  width_bits);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_height) - 1U,
                  height_bits);

  writer.put_bit(false);  // seq_cropping_window_present_flag
  writer.put_bit(false);  // seq_initial_display_delay_present_flag
  writer.put_bit(false);  // decoder_model_info_present_flag

  // §5.4.3 sequence_partition_config(): 64x64 superblocks, no SDP.
  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(false);  // use_128x128_superblock
  writer.put_bit(false);  // enable_sdp
  writer.put_bit(false);  // enable_ext_partitions
  writer.put_bit(false);  // reduce_pb_aspect_ratio

  // §5.4.4 sequence_segment_config().
  writer.put_bit(false);  // enable_ext_seg
  writer.put_bit(false);  // seq_seg_info_present_flag

  // §5.4.5 sequence_intra_config().
  writer.put_bit(false);  // enable_dip
  writer.put_bit(true);   // enable_intra_edge_filter
  writer.put_bit(false);  // enable_mrls
  writer.put_bit(true);   // enable_cfl_intra
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);  // enable_mhccp
  writer.put_bit(false);  // enable_ibp

  // §5.4.6 sequence_inter_config() for single_picture_header_flag = 0.
  for (int mode = 1; mode < 5; ++mode) {  // INTERINTRA..EXTENDWARP
    writer.put_bit(false);                // seq_enabled_motion_modes[mode]
  }
  // motionModeEnabled == 0 -> seq_frame_motion_modes_present_flag = 0.
  writer.put_bit(false);  // enable_masked_compound
  writer.put_bit(false);  // enable_ref_frame_mvs
  // reduced_ref_frame_mvs_mode = 0.
  writer.put_bits(static_cast<std::uint64_t>(c.order_hint_bits) - 1U, 4U);
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder -> DRL_REORDER_DISABLED
  writer.put_bit(true);   // explicit_ref_frame_map
  writer.put_bit(true);   // explicit_num_ref_frames
  writer.put_bits(static_cast<std::uint64_t>(c.num_ref_frames) - 1U, 4U);
  writer.put_bits(0U, 3U);  // long_term_frame_id_bits
  put_ns(writer, 5U, c.max_drl_bits_minus_1);  // MAX_REF_MV_STACK_SIZE - 1
  writer.put_bit(false);   // allow_frame_max_drl_bits
  put_ns(writer, 3U, 0U);  // seq_max_bvp_drl_bits_minus_1 (MAX_REF_BV_STACK_SIZE - 1)
  writer.put_bit(false);   // allow_frame_max_bvp_drl_bits
  writer.put_bits(0U, 2U); // num_same_ref_compound
  writer.put_bit(false);   // enable_tip
  writer.put_bit(false);   // enable_mv_traj
  writer.put_bit(false);   // enable_bawp
  writer.put_bit(false);   // enable_cwp
  writer.put_bit(false);   // enable_imp_msk_bld
  writer.put_bit(false);   // enable_df_sub_pu
  writer.put_bits(0U, 2U); // enable_opfl_refine = REFINE_NONE
  writer.put_bit(false);   // enable_refinemv
  writer.put_bit(false);   // enable_bru
  writer.put_bit(false);   // enable_adaptive_mvd
  writer.put_bit(false);   // enable_mvd_sign_derive
  writer.put_bit(false);   // enable_flex_mvres
  writer.put_bit(false);   // enable_global_motion
  writer.put_bit(false);   // enable_short_refresh_frame_flags

  // §5.4.7 sequence_scc_config(): no screen content tools, so fractional MVs
  // are allowed (seq_force_integer_mv = SELECT_INTEGER_MV).
  writer.put_bit(false);  // seq_choose_screen_content_tools
  writer.put_bit(false);  // seq_force_screen_content_tools

  // §5.4.8 sequence_transform_quant_entropy_config().
  writer.put_bit(false);  // enable_fsc
  writer.put_bit(false);  // enable_idtx_intra
  writer.put_bit(false);  // enable_intra_ist
  writer.put_bit(false);  // enable_inter_ist
  writer.put_bit(true);   // enable_chroma_dctonly
  writer.put_bit(false);  // enable_inter_ddt
  writer.put_bit(false);  // reduced_tx_part_set
  writer.put_bit(false);  // enable_cctx
  writer.put_bit(false);  // enable_tcq
  writer.put_bit(false);  // enable_parity_hiding
  writer.put_bit(false);  // enable_avg_cdf (no cross-frame blending)
  writer.put_bit(false);  // separate_uv_delta_q
  writer.put_bit(true);   // equal_ac_dc_q
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q => BaseUVAcDeltaQ = 0
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  // §5.4.10 sequence_filter_config().
  writer.put_bit(false);  // disable_loopfilters_across_tiles
  writer.put_bit(false);  // enable_cdef
  writer.put_bit(false);  // enable_gdf
  writer.put_bit(false);  // enable_restoration
  writer.put_bit(false);  // enable_ccso
  writer.put_bit(false);  // cdef_on_skip_txfm_always_on
  writer.put_bit(true);   // cdef_on_skip_txfm_disabled
  writer.put_bits(0U, 2U); // df_par_bits_minus_2

  // §5.4.2 sequence_tile_config().
  writer.put_bit(false);  // seq_tile_info_present_flag

  writer.put_bit(false);  // film_grain_params_present

  return writer.bit_size() - start;
}

std::size_t write_frame_header_syntax(BitWriter& writer, const VideoConfig& c,
                                      const VideoFrameHeader& h) {
  const std::size_t start = writer.bit_size();
  assert(c.base_q_idx >= 1U);
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header

  const bool key = h.keyframe;
  if (!key) {
    writer.put_bit(true);  // frame_is_inter = 1 -> INTER_FRAME
  }
  // long_term_id_plus_1 is absent (long_term_frame_id_bits == 0).
  writer.put_bit(true);  // immediate_output_frame
  // implicit_output_frame is absent because immediate_output_frame == 1.

  if (key) {
    // OBU_CLOSED_LOOP_KEY + max_mlayer_id == 0 -> refresh_frame_flags =
    // allFrames, no bits coded.
  }

  writer.put_bit(false);  // frame_size_override_flag = 0
  writer.put_bits(h.order_hint, c.order_hint_bits);  // order_hint

  if (key) {
    // FrameIsIntra -> primary_ref_frame = PRIMARY_REF_NONE, no bits.
    // frame_size() uses mfh_frame_* and codes no bits.
    // screen_content_params(): seq_force_screen_content_tools == 0, so
    // allow_screen_content_tools is inferred 0 and no bit is coded.
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
    // refresh_frame_flags is coded after the primary-ref block
    // (§5.18.2). enable_short_refresh_frame_flags == 0.
    writer.put_bits(h.refresh_frame_flags, c.num_ref_frames);
    // explicit_ref_frame_map == 1 (sequence header).
    writer.put_bit(true);   // frame_explicit_ref_frame_map
    writer.put_bits(c.num_total_refs, 3U);  // num_total_refs
    writer.put_bits(h.ref_slot,
                    static_cast<unsigned>(ceil_log2(c.num_ref_frames)));
    // frame_size_override_flag == 0 -> frame_size() codes no bits.
    // get_ref_frames(1) is skipped because the map is explicit.
    // use_ref_frame_mvs = 0, TipFrameMode = TIP_FRAME_DISABLED.
    // frame_opfl_refine_type(): enable_opfl_refine == REFINE_NONE, no bits.
    // screen_content_params(): seq_force_screen_content_tools == 0, no bit.
    writer.put_bit(false);  // allow_intrabc
    // max_drl_bits_minus_1 = seq_max_drl_bits_minus_1, no bits.
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv -> 1/8 pel
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTTAP
    // frame_enabled_motion_modes[] all zero -> no bits.
  }

  writer.put_bit(c.disable_cdf_update);  // disable_cdf_update

  write_tile_info(writer, c);

  writer.put_bits(c.base_q_idx, 8U);  // base_q_idx
  writer.put_bit(false);              // segmentation_enabled
  writer.put_bit(false);              // using_qmatrix
  writer.put_bit(false);              // delta_q_present (base_q_idx > 0)
  writer.put_bit(false);              // apply_deblocking_filter[0]
  writer.put_bit(false);              // apply_deblocking_filter[1]
  // apply_deblocking_filter[2..3] and every df_delta_q_present are absent.
  writer.put_bit(false);   // tx_mode_select = 0 -> TX_MODE_LARGEST
  if (!key) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
  }
  // allow_bawp / allow_warpmv_mode are disabled at sequence level.
  writer.put_bits(0U, 2U);  // reduced_tx_set
  // global_motion_params() returns early (enable_global_motion == 0).
  // film_grain_config() returns immediately (film_grain_params_present == 0).

  return writer.bit_size() - start;
}

}  // namespace

std::vector<std::uint8_t> make_sequence_header_obu(const VideoConfig& config) {
  BitWriter writer;
  write_sequence_header(writer, config);
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

std::size_t write_frame_header(BitWriter& writer, const VideoConfig& config,
                               const VideoFrameHeader& header) {
  return write_frame_header_syntax(writer, config, header);
}

std::size_t write_tile_group_header(BitWriter& writer, const VideoConfig& config,
                                    const VideoFrameHeader& header) {
  const std::size_t start = writer.bit_size();
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_syntax(writer, config, header);
  // tile_start_and_end_present_flag is absent (NumTiles == 1).
  writer.align_to_byte();
  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_frame_obu(const VideoConfig& config,
                                         const VideoFrameHeader& header,
                                         std::span<const std::uint8_t> tile_payload) {
  BitWriter writer;
  write_tile_group_header(writer, config, header);
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  const ObuType type =
      header.keyframe ? ObuType::kClosedLoopKey : ObuType::kRegularTileGroup;
  return av2enc::basic::make_obu(ObuHeader{type}, payload);
}

std::vector<std::uint8_t> make_temporal_delimiter_obu() {
  return av2enc::basic::make_obu(ObuHeader{ObuType::kTemporalDelimiter},
                              std::span<const std::uint8_t>{});
}

}  // namespace av2enc::video
