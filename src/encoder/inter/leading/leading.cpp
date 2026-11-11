#include "encoder/inter/leading/leading.h"

#include "bitstream/m0_headers.h"

#include <cassert>
#include <limits>

namespace av2enc::inter::leading {
namespace {

// uvlc() (§4.11.3).
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

// ns(n) (§4.11.8).
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

unsigned ceil_log2(std::uint32_t value) {
  if (value <= 1U) return 0U;
  unsigned bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

std::uint32_t mi_cols(const SequenceConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
}
std::uint32_t mi_rows(const SequenceConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
}

// §5.18.7.3 tile_info() for the single-tile configuration M1/M2 also use.
std::size_t write_tile_info(BitWriter& writer, const SequenceConfig& c) {
  const std::size_t start = writer.bit_size();
  const std::uint32_t sb_cols = (mi_cols(c) + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows(c) + 15U) >> 4U;
  assert(sb_cols >= 1U && sb_rows >= 1U);
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) writer.put_bit(false);  // increment_tile_cols_log2 = 0
  if (sb_rows > 1U) writer.put_bit(false);  // increment_tile_rows_log2 = 0
  return writer.bit_size() - start;
}

std::size_t write_sequence_header(BitWriter& writer,
                                  const SequenceConfig& c) {
  const std::size_t start = writer.bit_size();
  assert(c.order_hint_bits >= 1U && c.order_hint_bits <= 16U);
  assert(c.num_ref_frames >= 1U && c.num_ref_frames <= 16U);
  assert(c.max_drl_bits_minus_1 < 5U);

  put_uvlc(writer, 0);                    // seq_header_id
  writer.put_bits(0U, 5U);                // seq_profile_idc
  writer.put_bit(false);                  // single_picture_header_flag = 0
  writer.put_bits(c.seq_level_idx, 5U);   // seq_level_idx
  writer.put_bit(false);                  // seq_tier (level > 3)
  put_uvlc(writer, 0);                    // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                    // bit_depth_idc = 1 => 8-bit

  // §5.4.1 non-single-picture header.
  writer.put_bits(0U, 3U);                // seq_lcr_id
  writer.put_bit(false);                  // still_picture
  writer.put_bits(0U, 2U);                // max_tlayer_id
  writer.put_bits(0U, 3U);                // max_mlayer_id
  writer.put_bit(c.monotonic_output_order_flag);

  unsigned width_bits = bit_width(static_cast<std::uint32_t>(c.max_frame_width) - 1U);
  if (width_bits == 0) width_bits = 1;
  unsigned height_bits = bit_width(static_cast<std::uint32_t>(c.max_frame_height) - 1U);
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_width) - 1U, width_bits);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_height) - 1U, height_bits);

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

  // §5.4.6 sequence_inter_config().
  for (int mode = 1; mode < 5; ++mode) {
    writer.put_bit(false);  // seq_enabled_motion_modes[mode]
  }
  writer.put_bit(false);  // enable_masked_compound
  writer.put_bit(false);  // enable_ref_frame_mvs
  writer.put_bits(static_cast<std::uint64_t>(c.order_hint_bits) - 1U, 4U);
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder
  writer.put_bit(true);   // explicit_ref_frame_map
  writer.put_bit(true);   // explicit_num_ref_frames
  writer.put_bits(static_cast<std::uint64_t>(c.num_ref_frames) - 1U, 4U);
  writer.put_bits(0U, 3U);  // long_term_frame_id_bits
  put_ns(writer, 5U, c.max_drl_bits_minus_1);
  writer.put_bit(false);    // allow_frame_max_drl_bits
  put_ns(writer, 3U, 0U);   // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);    // allow_frame_max_bvp_drl_bits
  writer.put_bits(0U, 2U);  // num_same_ref_compound
  writer.put_bit(false);    // enable_tip
  writer.put_bit(false);    // enable_mv_traj
  writer.put_bit(false);    // enable_bawp
  writer.put_bit(false);    // enable_cwp
  writer.put_bit(false);    // enable_imp_msk_bld
  writer.put_bit(false);    // enable_df_sub_pu
  writer.put_bits(0U, 2U);  // enable_opfl_refine = REFINE_NONE
  writer.put_bit(false);    // enable_refinemv
  writer.put_bit(false);    // enable_bru
  writer.put_bit(false);    // enable_adaptive_mvd
  writer.put_bit(false);    // enable_mvd_sign_derive
  writer.put_bit(false);    // enable_flex_mvres
  writer.put_bit(false);    // enable_global_motion
  writer.put_bit(false);    // enable_short_refresh_frame_flags

  // §5.4.7 sequence_scc_config().
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
  writer.put_bit(false);  // enable_avg_cdf
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
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2

  // §5.4.2 sequence_tile_config().
  writer.put_bit(false);  // seq_tile_info_present_flag

  writer.put_bit(false);  // film_grain_params_present

  return writer.bit_size() - start;
}

// Exact OBU header (§5.2.2), no extension data.
std::size_t write_obu_header(BitWriter& writer, ObuType type,
                             std::uint8_t tlayer_id) {
  writer.put_bit(false);  // obu_header_extension_flag
  writer.put_bits(static_cast<std::uint8_t>(type), 5U);
  writer.put_bits(tlayer_id, 2U);
  return 8U;
}

std::vector<std::uint8_t> make_obu(ObuType type,
                                   std::span<const std::uint8_t> payload) {
  constexpr std::size_t kHeaderBytes = 1;
  const std::size_t obu_size = kHeaderBytes + payload.size();
  assert(obu_size <= std::numeric_limits<std::uint32_t>::max());

  std::vector<std::uint8_t> out;
  out.reserve(m0::leb128_size(static_cast<std::uint32_t>(obu_size)) + obu_size);
  m0::append_leb128(out, static_cast<std::uint32_t>(obu_size));

  BitWriter header_writer;
  write_obu_header(header_writer, type, 0);
  const std::vector<std::uint8_t> header = header_writer.take_bytes();
  assert(header.size() == kHeaderBytes);
  out.insert(out.end(), header.begin(), header.end());
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

std::size_t write_frame_header_syntax(BitWriter& writer,
                                      const SequenceConfig& c,
                                      const FrameHeader& h) {
  const std::size_t start = writer.bit_size();
  const bool key = is_key_obu(obu_type_for(h.kind));

  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header

  if (!key) {
    writer.put_bit(true);  // frame_is_inter = 1 -> INTER_FRAME
  }

  // §5.18.2 immediate_output_frame / implicit_output_frame.
  // OLK: immediate_output_frame is inferred 0, so only implicit is coded.
  // CLK/regular: immediate_output_frame = 1, implicit is inferred 0.
  // Leading: explicit immediate bit, then implicit only when immediate == 0
  // and monotonic_output_order_flag == 0.
  switch (h.kind) {
    case FrameKind::kOpenLoopKey:
      if (!c.monotonic_output_order_flag) {
        writer.put_bit(true);  // implicit_output_frame = 1
      }
      break;
    case FrameKind::kClosedLoopKey:
    case FrameKind::kRegularTileGroup:
      writer.put_bit(true);  // immediate_output_frame = 1
      break;
    case FrameKind::kLeadingTileGroup:
      writer.put_bit(h.leading_output_frame);  // immediate_output_frame
      if (!h.leading_output_frame && !c.monotonic_output_order_flag) {
        writer.put_bit(false);  // implicit_output_frame = 0 (output suppressed)
      }
      break;
  }

  // frame_size_override_flag + order_hint.
  writer.put_bit(false);  // frame_size_override_flag = 0
  writer.put_bits(h.order_hint, c.order_hint_bits);

  if (key) {
    if (h.kind == FrameKind::kOpenLoopKey) {
      // CLK with max_mlayer_id == 0 infers allFrames; an OLK codes
      // refresh_frame_flags f(NumRefFrames).
      writer.put_bits(h.refresh_frame_flags,
                      static_cast<unsigned>(c.num_ref_frames));
    }
    // FrameIsIntra -> primary_ref_frame = PRIMARY_REF_NONE (no bits),
    // frame_size()/screen_content_params() code no bits for this tool set.
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
    writer.put_bits(h.refresh_frame_flags,
                    static_cast<unsigned>(c.num_ref_frames));
    writer.put_bit(true);   // frame_explicit_ref_frame_map
    writer.put_bits(h.num_total_refs, 3U);
    const unsigned ref_bits = ceil_log2(c.num_ref_frames);
    for (std::uint8_t i = 0; i < h.num_total_refs; ++i) {
      writer.put_bits(h.ref_frame_idx[i], ref_bits);
    }
    // frame_size_override_flag == 0 -> frame_size() codes no bits.
    writer.put_bit(false);  // allow_intrabc
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv -> 1/8 pel
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTTAP
  }

  writer.put_bit(c.disable_cdf_update);

  write_tile_info(writer, c);

  writer.put_bits(c.base_q_idx, 8U);
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present
  writer.put_bit(false);  // apply_deblocking_filter[0]
  writer.put_bit(false);  // apply_deblocking_filter[1]
  writer.put_bit(false);  // tx_mode_select = TX_MODE_LARGEST
  if (!key) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set

  return writer.bit_size() - start;
}

}  // namespace

ObuType obu_type_for(FrameKind kind) noexcept {
  switch (kind) {
    case FrameKind::kClosedLoopKey:
      return ObuType::kClosedLoopKey;
    case FrameKind::kOpenLoopKey:
      return ObuType::kOpenLoopKey;
    case FrameKind::kLeadingTileGroup:
      return ObuType::kLeadingTileGroup;
    case FrameKind::kRegularTileGroup:
      return ObuType::kRegularTileGroup;
  }
  return ObuType::kRegularTileGroup;
}

bool is_leading_obu(ObuType type) noexcept {
  return type == ObuType::kLeadingTileGroup || type == ObuType::kLeadingSef ||
         type == ObuType::kLeadingTip;
}

bool is_regular_obu(ObuType type) noexcept {
  return type == ObuType::kOpenLoopKey || type == ObuType::kRegularTileGroup ||
         type == ObuType::kRegularSef || type == ObuType::kRegularTip;
}

bool is_key_obu(ObuType type) noexcept {
  return type == ObuType::kClosedLoopKey || type == ObuType::kOpenLoopKey;
}

bool is_tile_group_obu(ObuType type) noexcept {
  return type == ObuType::kLeadingTileGroup ||
         type == ObuType::kRegularTileGroup || is_key_obu(type);
}

OutputFlags output_flags(const SequenceConfig& config,
                         const FrameHeader& header) noexcept {
  OutputFlags flags;
  switch (header.kind) {
    case FrameKind::kClosedLoopKey:
    case FrameKind::kRegularTileGroup:
      flags.immediate = true;
      break;
    case FrameKind::kOpenLoopKey:
      // §5.18.2: immediate_output_frame is inferred 0 for an OLK.
      if (!config.monotonic_output_order_flag) flags.implicit = true;
      break;
    case FrameKind::kLeadingTileGroup:
      flags.immediate = header.leading_output_frame;
      if (!flags.immediate && !config.monotonic_output_order_flag) {
        flags.implicit = false;  // explicit suppression bit
      }
      break;
  }
  return flags;
}

bool validate_plan(const SequenceConfig& config, const Plan& plan,
                   std::string& error) {
  error.clear();
  if (plan.frames.empty()) {
    error = "plan has no frames";
    return false;
  }
  if (config.num_ref_frames < 1U || config.num_ref_frames > 16U) {
    error = "num_ref_frames must be in [1, 16]";
    return false;
  }
  const std::uint32_t all_slots =
      (config.num_ref_frames == 16U)
          ? 0xFFFFU
          : ((1U << config.num_ref_frames) - 1U);

  std::uint32_t olk_refresh = 0;
  bool olk_seen = false;
  bool regular_seen = false;
  std::uint32_t leading_slots = 0;

  for (std::size_t i = 0; i < plan.frames.size(); ++i) {
    const FrameHeader& h = plan.frames[i];
    const std::string where = "frame " + std::to_string(i);

    if ((h.refresh_frame_flags & ~all_slots) != 0U) {
      error = where + ": refresh_frame_flags exceeds NumRefFrames";
      return false;
    }
    if (config.order_hint_bits < 16U &&
        h.order_hint >= (1U << config.order_hint_bits)) {
      error = where + ": order_hint does not fit order_hint_bits";
      return false;
    }
    if (h.num_total_refs > 8U || h.num_total_refs > config.num_ref_frames) {
      error = where + ": num_total_refs is out of range";
      return false;
    }
    for (std::uint8_t r = 0; r < h.num_total_refs; ++r) {
      if (h.ref_frame_idx[r] >= config.num_ref_frames) {
        error = where + ": ref_frame_idx exceeds NumRefFrames";
        return false;
      }
    }

    const OutputFlags flags = output_flags(config, h);
    if (!flags.immediate && !flags.implicit &&
        h.refresh_frame_flags == 0U) {
      // §6.17.2: immediate_output_frame == 0 requires a non-zero refresh.
      error = where + ": non-output frame must refresh at least one slot";
      return false;
    }

    switch (h.kind) {
      case FrameKind::kClosedLoopKey:
        if (i != 0) {
          error = where + ": CLK must be the first frame";
          return false;
        }
        break;
      case FrameKind::kOpenLoopKey:
        if (olk_seen) {
          error = where + ": more than one OLK";
          return false;
        }
        if (config.monotonic_output_order_flag) {
          error = where +
                  ": OLK requires monotonic_output_order_flag = 0 "
                  "(immediate_output_frame is inferred 0)";
          return false;
        }
        olk_seen = true;
        olk_refresh = h.refresh_frame_flags;
        break;
      case FrameKind::kLeadingTileGroup:
        if (!olk_seen) {
          error = where + ": leading frame before the OLK";
          return false;
        }
        if (regular_seen) {
          error = where + ": leading frame after the first regular frame";
          return false;
        }
        if ((h.refresh_frame_flags & olk_refresh) != 0U) {
          error = where +
                  ": leading refresh_frame_flags overlaps OlkRefresh "
                  "(§6.17.2 leading-frame restriction)";
          return false;
        }
        leading_slots |= h.refresh_frame_flags;
        break;
      case FrameKind::kRegularTileGroup:
        regular_seen = true;
        if (olk_seen) {
          // §7.4.4: regular frames after the OLK must only reference slots
          // refreshed by the OLK (no pre-OLK frame, no leading frame).
          for (std::uint8_t r = 0; r < h.num_total_refs; ++r) {
            const std::uint32_t bit = 1U << h.ref_frame_idx[r];
            if ((bit & olk_refresh) == 0U) {
              error = where +
                      ": regular frame references a slot not refreshed by "
                      "the OLK";
              return false;
            }
            if ((bit & leading_slots) != 0U) {
              error = where + ": regular frame references a leading frame";
              return false;
            }
          }
        }
        break;
    }

  }

  if (!olk_seen) {
    error = "plan contains no OLK random access point";
    return false;
  }
  return true;
}

std::size_t write_frame_header(BitWriter& writer, const SequenceConfig& config,
                               const FrameHeader& header) {
  return write_frame_header_syntax(writer, config, header);
}

std::size_t write_tile_group_header(BitWriter& writer,
                                    const SequenceConfig& config,
                                    const FrameHeader& header) {
  const std::size_t start = writer.bit_size();
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_syntax(writer, config, header);
  // tile_start_and_end_present_flag is absent (NumTiles == 1).
  writer.align_to_byte();
  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_frame_obu(
    const SequenceConfig& config, const FrameHeader& header,
    std::span<const std::uint8_t> tile_payload) {
  BitWriter writer;
  write_tile_group_header(writer, config, header);
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  return make_obu(obu_type_for(header.kind), payload);
}

std::vector<std::uint8_t> make_sequence_header_obu(
    const SequenceConfig& config) {
  BitWriter writer;
  write_sequence_header(writer, config);
  writer.put_bit(false);  // obu_extension_flag
  writer.put_bit(true);   // trailing_one_bit
  writer.align_to_byte();
  return make_obu(ObuType::kSequenceHeader, writer.take_bytes());
}

std::vector<std::uint8_t> make_temporal_delimiter_obu() {
  return make_obu(ObuType::kTemporalDelimiter,
                  std::span<const std::uint8_t>{});
}

std::vector<std::vector<std::uint8_t>> assemble_temporal_units(
    const SequenceConfig& config, const Plan& plan,
    std::span<const std::vector<std::uint8_t>> tile_payloads) {
  std::vector<std::vector<std::uint8_t>> units;
  if (plan.frames.size() != tile_payloads.size()) return units;
  units.reserve(plan.frames.size());
  const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
  for (std::size_t i = 0; i < plan.frames.size(); ++i) {
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    if (plan.frames[i].kind == FrameKind::kClosedLoopKey ||
        plan.frames[i].kind == FrameKind::kOpenLoopKey) {
      const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);
      unit.insert(unit.end(), seq.begin(), seq.end());
    }
    const std::vector<std::uint8_t> obu =
        make_frame_obu(config, plan.frames[i], tile_payloads[i]);
    unit.insert(unit.end(), obu.begin(), obu.end());
    units.push_back(std::move(unit));
  }
  return units;
}

}  // namespace av2enc::inter::leading
