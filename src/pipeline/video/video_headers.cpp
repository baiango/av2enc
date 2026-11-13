#include "pipeline/video/video_headers.h"

#include "core/bitstream/basic_headers.h"
#if defined(AV2ENC_VIDEO_GME)
#include "codec/inter/gme/gme_syntax.h"
#endif
#include "codec/filters/stream/filters_stream.h"
#include "codec/filters/pc_wiener/pc_wiener_stream.h"
#include "codec/transform/qm/qm.h"
#include "codec/filters/gdf/gdf_syntax.h"
#include "codec/inter/tip/tip_signalling.h"
#include "pipeline/video/post/loop_filters.h"

#include <cassert>
#include <string>

namespace av2enc::video {
namespace {

namespace filters_stream = av2enc::still::filters_stream;
namespace gdf = ::av2enc::filters::gdf;
namespace tip = ::av2enc::inter::tip;

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
  assert(c.seq_lcr_id < 8U);
  assert(c.max_tlayer_id < 4U);
  assert(c.max_mlayer_id < 8U);
  assert(c.seq_max_mlayer_cnt >= 1U &&
         c.seq_max_mlayer_cnt <= c.max_mlayer_id + 1U);

  const int chroma_idc = fmt::chroma_format_idc(c.chroma);
  const int depth_idc = fmt::bit_depth_idc(c.depth);
  const fmt::Format header_format{c.max_frame_width, c.max_frame_height,
                                  c.chroma, c.depth};
  const int profile = fmt::profile_idc(header_format);
  assert(chroma_idc >= 0);
  assert(depth_idc >= 0);  // 12-bit is not signalable
  assert(profile >= 0);
  const bool monochrome = !fmt::has_chroma(c.chroma);

  put_uvlc(writer, 0);                 // seq_header_id
  writer.put_bits(static_cast<std::uint64_t>(profile), 5U);  // seq_profile_idc
  writer.put_bit(false);               // single_picture_header_flag = 0
  writer.put_bits(c.seq_level_idx, 5U);  // seq_level_idx
  writer.put_bit(false);               // seq_tier (level > 3)
  // §6.4.1 Table 6.2: 0 = 4:2:0, 1 = 4:0:0, 2 = 4:4:4, 3 = 4:2:2.
  put_uvlc(writer, static_cast<std::uint32_t>(chroma_idc));
  // §6.4.1 Table 6.3: bit_depth_idc is one bit, 0 = 10-bit, 1 = 8-bit.
  put_uvlc(writer, static_cast<std::uint32_t>(depth_idc));

  // §5.4.1 non-single-picture header.
  writer.put_bits(c.seq_lcr_id, 3U);  // seq_lcr_id
  writer.put_bit(false);              // still_picture
  writer.put_bits(c.max_tlayer_id, 2U);
  writer.put_bits(c.max_mlayer_id, 3U);
  if (c.max_mlayer_id > 0U) {
    const unsigned n = ceil_log2(static_cast<std::uint32_t>(c.max_mlayer_id) +
                                 1U);
    writer.put_bits(
        static_cast<std::uint64_t>(c.seq_max_mlayer_cnt) - 1U, n);
  }
  writer.put_bit(c.monotonic_output_order_flag);

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
  // §5.4.1 timing block (non-single-picture path).
  writer.put_bit(c.initial_display_delay_present);
  if (c.initial_display_delay_present) {
    writer.put_bits(c.initial_display_delay_minus_1, 4U);
  }
  writer.put_bit(c.decoder_model_info_present);
  if (c.decoder_model_info_present) {
    writer.put_bits(c.num_units_in_decoding_tick, 32U);
    // §5.4.13 nested block: pinned dav2d f9dafbc rejects it; keep it off.
    writer.put_bit(false);  // seq_decoder_model_info_present_flag
  }

  // §5.4.1 dependency maps.  MLayerDependencyMap defaults to ref <= curr and
  // is only coded for a multi-layer plan.  TLayerDependencyMap is coded when
  // max_tlayer_id > 0; the multi-layer form carries an extra presence bit.
  if (c.max_mlayer_id > 0U) {
    writer.put_bit(c.mlayer_dependency_present);
    if (c.mlayer_dependency_present) {
      for (unsigned curr = 1U; curr <= c.max_mlayer_id; ++curr) {
        for (int ref = static_cast<int>(curr); ref >= 0; --ref) {
          writer.put_bit(true);  // MLayerDependencyMap[curr][ref] = ref<=curr
        }
      }
    }
  }
  if (c.max_tlayer_id > 0U) {
    writer.put_bit(c.tlayer_dependency_present_flag);
    if (c.tlayer_dependency_present_flag) {
      if (c.max_mlayer_id > 0U) {
        writer.put_bit(false);  // multi_tlayer_dependency_map_present_flag
      }
      for (unsigned mlayer = 0U; mlayer <= c.max_mlayer_id; ++mlayer) {
        for (int curr = 1; curr <= c.max_tlayer_id; ++curr) {
          for (int ref = curr; ref >= 0; --ref) {
            writer.put_bit(c.tlayer_dependency[curr][ref]);
          }
        }
      }
    }
  }

  // §5.4.3 sequence_partition_config(): 64x64 superblocks, no SDP.
  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(false);  // use_128x128_superblock
  if (!monochrome) {
    writer.put_bit(false);  // enable_sdp
  }
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
  if (!monochrome) {
    writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  }
  writer.put_bit(c.enable_mhccp);  // enable_mhccp
  writer.put_bit(false);  // enable_ibp

  // §5.4.6 sequence_inter_config() for single_picture_header_flag = 0.
  for (int mode = 1; mode < 5; ++mode) {  // INTERINTRA..EXTENDWARP
    writer.put_bit(false);                // seq_enabled_motion_modes[mode]
  }
  // motionModeEnabled == 0 -> seq_frame_motion_modes_present_flag = 0.
  writer.put_bit(c.enable_masked_compound);
  writer.put_bit(c.enable_ref_frame_mvs);
  if (c.enable_ref_frame_mvs) {
    writer.put_bit(false);  // reduced_ref_frame_mvs_mode
  }
  // reduced_ref_frame_mvs_mode = 0.
  writer.put_bits(static_cast<std::uint64_t>(c.order_hint_bits) - 1U, 4U);
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(c.disable_drl_reorder);  // disable_drl_reorder
  if (!c.disable_drl_reorder) {
    writer.put_bit(c.constrain_drl_reorder);  // constrain_drl_reorder
  }
  writer.put_bit(true);   // explicit_ref_frame_map
  writer.put_bit(true);   // explicit_num_ref_frames
  writer.put_bits(static_cast<std::uint64_t>(c.num_ref_frames) - 1U, 4U);
  writer.put_bits(0U, 3U);  // long_term_frame_id_bits
  put_ns(writer, 5U, c.max_drl_bits_minus_1);  // MAX_REF_MV_STACK_SIZE - 1
  writer.put_bit(false);   // allow_frame_max_drl_bits
  put_ns(writer, 3U, 0U);  // seq_max_bvp_drl_bits_minus_1 (MAX_REF_BV_STACK_SIZE - 1)
  writer.put_bit(false);   // allow_frame_max_bvp_drl_bits
  writer.put_bits(c.num_same_ref_compound, 2U);  // num_same_ref_compound
  writer.put_bit(c.enable_tip);  // enable_tip
  if (c.enable_tip) {
    writer.put_bit(c.disable_tip_output);     // disable_tip_output
    writer.put_bit(c.enable_tip_hole_fill);   // enable_tip_hole_fill
  }
  writer.put_bit(false);   // enable_mv_traj
  writer.put_bit(c.enable_bawp);  // enable_bawp (§5.4.6)
  writer.put_bit(c.enable_cwp);
  writer.put_bit(c.enable_imp_msk_bld);
  writer.put_bit(false);   // enable_df_sub_pu
  writer.put_bits(0U, 2U); // enable_opfl_refine = REFINE_NONE
  writer.put_bit(false);   // enable_refinemv
  writer.put_bit(false);   // enable_bru
  writer.put_bit(false);   // enable_adaptive_mvd
  writer.put_bit(false);   // enable_mvd_sign_derive
  writer.put_bit(false);   // enable_flex_mvres
  writer.put_bit(c.enable_global_motion);  // enable_global_motion (§5.4.6)
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
  if (!monochrome) {
    writer.put_bit(true);   // enable_chroma_dctonly
  }
  writer.put_bit(false);  // enable_inter_ddt
  writer.put_bit(false);  // reduced_tx_part_set
  if (!monochrome) {
    writer.put_bit(false);  // enable_cctx
  }
  writer.put_bit(false);  // enable_tcq
  writer.put_bit(false);  // enable_parity_hiding
  writer.put_bit(c.enable_avg_cdf);  // enable_avg_cdf
  if (c.enable_avg_cdf) {
    writer.put_bit(c.avg_cdf_type);  // avg_cdf_type: 1 = average across tiles
  }
  if (!monochrome) {
    writer.put_bit(false);  // separate_uv_delta_q
  }
  writer.put_bit(true);   // equal_ac_dc_q
  if (!monochrome) {
    writer.put_bits(23U, 5U);  // base_uv_ac_delta_q => BaseUVAcDeltaQ = 0
    writer.put_bit(false);     // uv_ac_delta_q_enabled
  }

  // §5.4.10 sequence_filter_config().
  writer.put_bit(false);  // disable_loopfilters_across_tiles
  writer.put_bit(c.enable_cdef);  // enable_cdef
  writer.put_bit(c.enable_gdf);  // enable_gdf
  if (c.enable_gdf) {
    // §5.4.10 codes gdf_unit_matches_sb_size only for a 64x64 sequence SB.
    writer.put_bit(c.gdf_unit_matches_sb_size);
  }
  writer.put_bit(c.enable_restoration);  // enable_restoration
  if (c.enable_restoration) {
    writer.put_bit(c.lr_disable_luma_pc_wiener);
    writer.put_bit(c.lr_disable_luma_wiener_ns);
    // lr_tools_disable[1][RESTORE_PC_WIENER] is inferred to 1.
    writer.put_bit(c.lr_uv_present);
    if (c.lr_uv_present) {
      writer.put_bit(c.lr_disable_chroma_wiener_ns);
    }
  }
  writer.put_bit(c.enable_ccso);  // enable_ccso
  if (c.enable_ccso) {
    writer.put_bit(c.ccso_unit_matches_sb_size);
  }
  writer.put_bit(c.cdef_on_skip_txfm_always_on);
  if (!c.cdef_on_skip_txfm_always_on) {
    writer.put_bit(c.cdef_on_skip_txfm_disabled);
  }
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
  assert(c.base_q_idx <=
         static_cast<unsigned>(fmt::max_quantizer(c.depth)));
  put_uvlc(writer, c.cur_mfh_id);  // cur_mfh_id
  if (c.cur_mfh_id == 0U) {
    put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  }

  const bool key = h.keyframe;
  if (!key && !h.tip_frame) {
    // §5.18.2: frame_is_inter is absent for a TIP OBU (FrameType comes from
    // the OBU type).
    writer.put_bit(true);  // frame_is_inter = 1 -> INTER_FRAME
  }
  // long_term_id_plus_1 is absent (long_term_frame_id_bits == 0).
  writer.put_bit(h.immediate_output_frame);  // immediate_output_frame
  if (!h.immediate_output_frame && !c.monotonic_output_order_flag) {
    writer.put_bit(h.implicit_output_frame);  // implicit_output_frame
  }

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
    writer.put_bit(h.signal_primary_ref_frame);
    // §5.18.2: disable_cross_frame_cdf_init is absent for a TIP frame.
    if (!h.tip_frame) {
      writer.put_bit(h.disable_cross_frame_cdf_init);
    }
    if (h.signal_primary_ref_frame) {
      writer.put_bits(h.primary_ref_frame, 3U);  // primary_ref_frame
    }
    // refresh_frame_flags is coded after the primary-ref block
    // (§5.18.2). enable_short_refresh_frame_flags == 0.
    writer.put_bits(h.refresh_frame_flags, c.num_ref_frames);
    // explicit_ref_frame_map == 1 (sequence header).
    writer.put_bit(true);   // frame_explicit_ref_frame_map
    const unsigned num_refs =
        h.num_total_refs == 0 ? c.num_total_refs : h.num_total_refs;
    writer.put_bits(num_refs, 3U);  // num_total_refs
    if (num_refs <= 1U) {
      writer.put_bits(h.ref_slot,
                      static_cast<unsigned>(ceil_log2(c.num_ref_frames)));
    } else {
      for (unsigned i = 0; i < num_refs; ++i) {
        writer.put_bits(h.ref_frame_idx[i],
                        static_cast<unsigned>(ceil_log2(c.num_ref_frames)));
      }
    }
    // frame_size_override_flag == 0 -> frame_size() codes no bits.
    // get_ref_frames(1) is skipped because the map is explicit.
    if (c.enable_ref_frame_mvs) {
      // §5.18.2: present only when the sequence enables the tool.  SbSize is
      // BLOCK_64X64 on the M2 path, so tmvp_sample_step_minus_1 is absent.
      writer.put_bit(h.use_ref_frame_mvs);
    }
    if (c.enable_tip) {
      tip::TipFrameSyntax ts;
      ts.use_ref_frame_mvs = h.use_ref_frame_mvs;
      ts.num_total_refs = num_refs;
      ts.sb_64x64 = true;
      ts.enable_tip = true;
      ts.enable_tip_output = !c.disable_tip_output;
      ts.enable_tip_refinemv = false;
      ts.enable_opfl_refine = tip::RefineType::kNone;
      ts.enable_tip_hole_fill = c.enable_tip_hole_fill;
      ts.is_tip_frame = h.tip_frame;
      ts.requested_mode = h.tip_frame ? tip::TipFrameMode::kAsOutput
                                      : tip::TipFrameMode::kDisabled;
      ts.num_past_refs = h.tip_frame ? 1 : 0;
      ts.num_future_refs = h.tip_frame ? 1 : 0;
      ts.tip_global_wtd_index = h.tip_global_wtd_index;
      ts.tip_global_mv =
          tip::Mv{h.tip_global_mv_row, h.tip_global_mv_col};
      ts.tip_sharp = h.tip_sharp;
      ts.tip_regular = false;
      if (tip::write_tip_frame_syntax(writer, ts) !=
          tip::TemporalStatus::kOk) {
        // The header writer has no error channel; every value above is
        // validated by the frame planner, so a failure is a programming error.
        assert(false && "invalid TIP frame syntax");
      }
    }
    if (h.tip_frame) {
      // §5.18.2 runs decode_frame_wrapup() right after the TIP block: no
      // disable_cdf_update / tile_info / quantization / tail syntax.
      return writer.bit_size() - start;
    }
    // use_ref_frame_mvs = 0, TipFrameMode = TIP_FRAME_DISABLED.
    // screen_content_params(): seq_force_screen_content_tools == 0, no bit.
    writer.put_bit(false);  // allow_intrabc
    // max_drl_bits_minus_1 = seq_max_drl_bits_minus_1, no bits.
    writer.put_bit(h.use_qtr_precision_mv);  // use_qtr_precision_mv
    if (!h.use_qtr_precision_mv) {
      writer.put_bit(h.allow_high_precision_mv);  // allow_high_precision_mv
    }    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTTAP
    // frame_enabled_motion_modes[] all zero -> no bits.
  }

  writer.put_bit(c.disable_cdf_update);  // disable_cdf_update

  write_tile_info(writer, c);

  // §5.18.6.1: f(8) at 8-bit, f(9) at 10-bit.
  writer.put_bits(c.base_q_idx,
                  static_cast<unsigned>(fmt::base_q_idx_bits(c.depth)));
  writer.put_bit(false);              // segmentation_enabled
  // §5.18.6.2 setup_qm_params(): default-matrix QM only (level 0).  No
  // quantizer-matrix OBU is emitted, so the decoder uses the §9.4 defaults.
  {
    qm::QmFrameState qm_state;
    if (c.using_qmatrix) {
      qm_state.using_qmatrix = true;
      qm_state.pic_qm_num_minus_1 = 0;
      qm_state.qm_y[0] = 0;
      qm_state.qm_u[0] = 0;
      qm_state.qm_v[0] = 0;
    }
    std::string qm_error;
    qm::write_setup_qm_params(writer, qm_state, /*num_planes=*/3,
                              /*segmentation_enabled=*/false, &qm_error);
    assert(qm_error.empty());
  }
  writer.put_bit(false);              // delta_q_present (base_q_idx > 0)
  if (c.enable_deblock) {
    // §5.18.7.9 deblocking_filter_params(): all four planes enabled, no
    // df_delta_q.  df_par_bits_minus_2 is 0 in the sequence header, so the
    // dfParBits = 2 domain only matters if a delta is ever signalled.
    const bool apply[4] = {true, true, true, true};
    const bool present[4] = {false, false, false, false};
    std::string deblock_error;
    const std::size_t deblock_bits = post::write_video_deblock_params(
        writer, apply, present, &deblock_error);
    assert(deblock_error.empty());
    assert(deblock_bits == 8U);  // 4 apply flags + 4 df_delta_q_present
    (void)deblock_bits;
  } else {
    writer.put_bit(false);            // apply_deblocking_filter[0]
    writer.put_bit(false);            // apply_deblocking_filter[1]
    // apply_deblocking_filter[2..3] and every df_delta_q_present are absent.
  }
  // §5.18.7.9 gdf_params(): after deblocking_filter_params(), before
  // cdef_params().  GDF is luma-only and wired in the header-only mode
  // (gdf_per_block = 0), so no §5.20.10.3 symbol is coded.
  if (c.enable_gdf) {
    gdf::FrameGdfContext gdf_context;
    gdf_context.enable_gdf = true;
    gdf_context.gdf_unit_matches_sb_size = c.gdf_unit_matches_sb_size;
    gdf_context.coded_lossless = false;
    gdf_context.single_picture_header_flag = false;
    gdf_context.sb_size = 64;
    gdf_context.width = static_cast<int>(c.max_frame_width);
    gdf_context.height = static_cast<int>(c.max_frame_height);
    gdf_context.mi_cols = static_cast<int>(mi_cols(c));
    gdf_context.mi_rows = static_cast<int>(mi_rows(c));
    gdf_context.disable_loopfilters_across_tiles = false;
    gdf_context.tile_cols = 1;
    gdf_context.tile_rows = 1;
    gdf_context.mi_col_starts.assign(1, 0);
    gdf_context.mi_row_starts.assign(1, 0);
    gdf::FrameGdfParams gdf_params;
    gdf_params.frame_enable = h.gdf.enabled;
    gdf_params.per_block = false;
    gdf_params.pic_qc_idx = h.gdf.qc_idx;
    gdf_params.pic_scale_idx = h.gdf.scale_idx;
    std::string gdf_error;
    gdf::WriteFrameGdfParams(writer, gdf_context, &gdf_params, &gdf_error);
    assert(gdf_error.empty());
  }
  // §5.18.7.10 cdef_params(): after deblocking_filter_params(), before
  // tx_mode_select.  Only present when the sequence enables CDEF.
  if (c.enable_cdef) {
    filters_stream::CdefParams params;
    params.frame_enable = h.cdef.enabled;
    params.damping_minus_3 = h.cdef.damping - 3;
    params.strengths_minus_1 = h.cdef.num_strengths - 1;
    params.num_strengths = h.cdef.num_strengths;
    params.on_skip_txfm_frame_enable = h.cdef.on_skip_txfm;
    for (int i = 0; i < h.cdef.num_strengths; ++i) {
      params.set[i].y_pri = h.cdef.strength[i].y_pri;
      params.set[i].y_sec = h.cdef.strength[i].y_sec;
      params.set[i].uv_pri = h.cdef.strength[i].uv_pri;
      params.set[i].uv_sec = h.cdef.strength[i].uv_sec;
    }
    filters_stream::CdefHeaderContext context;
    context.coded_lossless = false;
    context.enable_cdef = true;
    context.single_picture_header_flag = false;
    context.cdef_on_skip_txfm = c.cdef_on_skip_txfm_mode;
    context.num_planes = 3;
    std::string cdef_error;
    filters_stream::write_cdef_params(writer, params, context, &cdef_error);
    assert(cdef_error.empty());
  }
  // §5.18.7.11 lr_params(): between cdef_params() and ccso_params().  Only the
  // per-unit Wiener NS form is wired (no frame filters, no PC-Wiener yet).
  if (c.enable_restoration && h.lr.pc_wiener) {
    // §5.18.7.11 with FrameRestorationType[0] = RESTORE_PC_WIENER.  The
    // PC-Wiener tool list codes no frame_filters_on bit and no coefficients;
    // chroma Wiener NS is signalled independently.
    namespace pcw = ::av2enc::filters::pc_wiener_stream;
    pcw::LrHeaderContext context;
    context.coded_lossless = false;
    context.enable_restoration = true;
    context.single_picture_header_flag = false;
    context.frame_is_intra = key;
    context.sb_size = 64;
    context.num_planes = 3;
    context.sub_x = 1;
    context.sub_y = 1;
    context.base_q_idx = c.base_q_idx;
    context.lr_tools_disable[0][pcw::kRestorePcWiener] = false;
    context.lr_tools_disable[0][pcw::kRestoreWienerNs] = true;
    context.lr_tools_disable[1][pcw::kRestorePcWiener] = true;
    const bool chroma_lr =
        h.lr.plane[1].enabled || h.lr.plane[2].enabled;
    context.lr_tools_disable[1][pcw::kRestoreWienerNs] = !chroma_lr;
    pcw::LrParams params;
    params.frame_type[0] = pcw::kRestorePcWiener;
    params.frame_type[1] =
        h.lr.plane[1].enabled ? pcw::kRestoreWienerNs : pcw::kRestoreNone;
    params.frame_type[2] =
        h.lr.plane[2].enabled ? pcw::kRestoreWienerNs : pcw::kRestoreNone;
    params.luma_shift = h.lr.plane[0].shift;
    params.chroma_shift = h.lr.plane[1].enabled ? h.lr.plane[1].shift
                                                : h.lr.plane[2].shift;
    std::string lr_error;
    const std::size_t bits =
        pcw::WriteLrParams(writer, &params, context, &lr_error);
    assert(lr_error.empty());
    (void)bits;
  } else if (c.enable_restoration) {
    filters_stream::LrParams params;
    for (int plane = 0; plane < 3; ++plane) {
      params.frame_type[plane] =
          h.lr.plane[plane].enabled ? filters_stream::kRestoreWienerNs
                                    : filters_stream::kRestoreNone;
      params.frame_filters_on[plane] = false;
    }
    params.num_filter_classes = 1;
    params.num_filter_classes_idx = 0;
    params.luma_shift = h.lr.luma_shift;
    params.chroma_shift = h.lr.chroma_shift;
    filters_stream::LrHeaderContext context;
    context.coded_lossless = false;
    context.enable_restoration = true;
    context.single_picture_header_flag = false;
    context.frame_is_intra = key;
    context.sb_size = 64;
    context.num_planes = 3;
    context.sub_x = 1;
    context.sub_y = 1;
    context.base_q_idx = c.base_q_idx;
    context.lr_tools_disable[0][0] = false;
    context.lr_tools_disable[0][1] = true;   // PC-Wiener disabled
    context.lr_tools_disable[0][2] = c.lr_disable_luma_wiener_ns;
    context.lr_tools_disable[0][3] = false;
    context.lr_tools_disable[1][0] = false;
    context.lr_tools_disable[1][1] = true;
    context.lr_tools_disable[1][2] = c.lr_uv_present
                                         ? c.lr_disable_chroma_wiener_ns
                                         : c.lr_disable_luma_wiener_ns;
    context.lr_tools_disable[1][3] = false;
    std::string lr_error;
    filters_stream::write_lr_params(writer, params, context, &lr_error);
    assert(lr_error.empty());
  }
  // §5.18.7.12 ccso_params(): after lr_params(), before tx_mode_select.
  if (c.enable_ccso) {
    filters_stream::CcsoParams params;
    params.frame_flag = h.ccso.enabled;
    params.ccso_luma_size_log2 = h.ccso.ccso_luma_size_log2;
    for (int plane = 0; plane < 3; ++plane) {
      filters_stream::CcsoPlaneParams& dst = params.plane[plane];
      const post::VideoCcsoPlane& src = h.ccso.plane[plane];
      dst.enabled = src.enabled;
      dst.bo_only = src.bo_only;
      dst.scale_idx = src.scale_idx;
      dst.quant_idx = src.quant_idx;
      dst.ext_filter = src.ext_filter;
      dst.edge_clf = src.edge_clf;
      dst.max_band_log2 = src.max_band_log2;
      dst.offset_idx = src.offset_idx;
    }
    filters_stream::CcsoHeaderContext context;
    context.coded_lossless = false;
    context.enable_ccso = true;
    context.single_picture_header_flag = false;
    context.frame_is_intra = key;
    context.num_planes = 3;
    context.sb_size = 64;
    context.ccso_unit_matches_sb_size = c.ccso_unit_matches_sb_size;
    context.tile_start_or = 0;
    std::string ccso_error;
    filters_stream::write_ccso_params(writer, params, context, &ccso_error);
    assert(ccso_error.empty());  }
  writer.put_bit(c.tx_mode_select);  // 0 = TX_MODE_LARGEST, 1 = TX_MODE_SELECT
  if (!key) {
    writer.put_bit(h.reference_select);
    writer.put_bit(false);  // skip_mode_present
    // §5.18.2 allow_bawp == enable_bawp && !FrameIsIntra.  allow_warpmv_mode
    // stays absent because every seq_enabled_motion_modes bit is 0.
    if (c.enable_bawp) writer.put_bit(true);  // allow_bawp
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
#if defined(AV2ENC_VIDEO_GME)
  // §5.18.9.1 global_motion_params(): absent for an intra frame or when the
  // sequence bit is 0; otherwise one use_global_motion bit followed by the
  // per-reference models.  The default plan (num_refs == 0, identity models)
  // writes exactly the single zero bit every existing gate already produces.
  if (c.enable_global_motion && !key) {
    inter::gme::GmSyntaxContext gm;
    gm.frame_is_intra = false;
    gm.enable_global_motion = true;
    gm.switch_frame = false;
    gm.num_total_refs = static_cast<int>(h.num_total_refs);
    gm.order_hint = static_cast<int>(h.order_hint);
    gm.our_ref = -1;  // no base model; every parameter is coded directly
    for (int r = 0; r < gm.num_total_refs && r < kGmMaxRefs; ++r) {
      gm.ref_order_hint[static_cast<std::size_t>(r)] =
          h.gm.ref_order_hint[static_cast<std::size_t>(r)];
      gm.ref_frame_idx[static_cast<std::size_t>(r)] =
          h.ref_frame_idx[static_cast<std::size_t>(r)];
      gm.params[static_cast<std::size_t>(r)] =
          h.gm.params[static_cast<std::size_t>(r)];
    }
    gm.use_global_motion = h.gm.use_global_motion;
    std::string gm_error;
    inter::gme::write_global_motion_params(writer, gm, &gm_error);
  }
#else
  // Standalone component gates compile this writer without the GME syntax
  // module; the product build always defines AV2ENC_VIDEO_GME.
  assert(!c.enable_global_motion);
#endif
  // film_grain_config() returns immediately (film_grain_params_present == 0).

  return writer.bit_size() - start;
}

}  // namespace

void apply_video_loop_filter_sequence_config(
    VideoConfig& config, const post::VideoLoopFilterOptions& loop_filters) {
  if (loop_filters.cdef) {
    config.enable_cdef = true;
    config.cdef_on_skip_txfm_mode = loop_filters.cdef_on_skip_txfm_mode;
    config.cdef_on_skip_txfm_always_on =
        loop_filters.cdef_on_skip_txfm_mode == 1;
    config.cdef_on_skip_txfm_disabled =
        loop_filters.cdef_on_skip_txfm_mode == 0;
  }
  if (loop_filters.ccso) {
    config.enable_ccso = true;
    config.ccso_unit_matches_sb_size = true;
  }
  if (loop_filters.gdf) {
    config.enable_gdf = true;
    config.gdf_unit_matches_sb_size = true;
  }
  if (loop_filters.wiener || loop_filters.chroma_lr) {
    config.enable_restoration = true;
    config.lr_disable_luma_pc_wiener = true;
    config.lr_disable_luma_wiener_ns = !loop_filters.wiener;
    config.lr_uv_present = loop_filters.chroma_lr;
    config.lr_disable_chroma_wiener_ns = !loop_filters.chroma_lr;
  }
}

void apply_video_loop_filter_frame_header(const post::VideoFilterPlan& plan,
                                          VideoFrameHeader& header) {
  header.cdef = plan.cdef;
  header.ccso = plan.ccso;
  header.lr = plan.lr;
  header.gdf = plan.gdf;
}

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
                                         std::span<const std::uint8_t> tile_payload,
                                         std::uint8_t obu_tlayer_id) {
  BitWriter writer;
  write_tile_group_header(writer, config, header);
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  const ObuType type =
      header.keyframe ? ObuType::kClosedLoopKey : ObuType::kRegularTileGroup;
  ObuHeader obu{type};
  obu.tlayer_id = obu_tlayer_id;
  return av2enc::basic::make_obu(obu, payload);
}

std::vector<std::uint8_t> make_tip_frame_obu(const VideoConfig& config,
                                             const VideoFrameHeader& header,
                                             std::string& error) {
  error.clear();
  if (!header.tip_frame) {
    error = "TIP OBU requested for a non-TIP frame header";
    return {};
  }
  if (!config.enable_tip || !config.enable_ref_frame_mvs) {
    error = "TIP OBU requested without sequence enable_tip/"
            "enable_ref_frame_mvs";
    return {};
  }
  if (!header.use_ref_frame_mvs) {
    error = "TIP frame requires use_ref_frame_mvs = 1 (§7.10.1)";
    return {};
  }
  BitWriter writer;
  // No is_first_tile_group bit for a TIP OBU: the frame header is the whole
  // payload before the §5.2.3 trailing bits.
  write_frame_header_syntax(writer, config, header);
  std::vector<std::uint8_t> obu = tip::make_tip_frame_obu(
      tip::TipObuType::kRegularTip, &writer);
  if (obu.empty()) {
    error = "TIP OBU payload assembly failed";
    return {};
  }
  return obu;
}

std::vector<std::uint8_t> make_temporal_delimiter_obu() {
  return av2enc::basic::make_obu(ObuHeader{ObuType::kTemporalDelimiter},
                              std::span<const std::uint8_t>{});
}

}  // namespace av2enc::video
