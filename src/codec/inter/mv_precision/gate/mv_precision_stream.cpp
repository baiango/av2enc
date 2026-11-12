#include "codec/inter/mv_precision/gate/mv_precision_stream.h"

#include <cassert>

#include "core/bitstream/basic_headers.h"

namespace av2enc::inter::mvprec::gate {
namespace {

using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;

constexpr int kOrderHintBits = 4;
constexpr int kBaseQIdx = 90;

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
  assert(n >= 2U && value < n);
  unsigned width = 0;
  while ((1U << width) < n) ++width;
  const std::uint32_t cutoff = (1U << width) - n;
  if (value < cutoff) {
    writer.put_bits(value, width - 1U);
  } else {
    writer.put_bits(value + cutoff, width);
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
  if (value <= 1U) return 0U;
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

std::uint32_t mi_cols_of(int width) {
  return 2U * ((static_cast<std::uint32_t>(width) + 7U) >> 3U);
}
std::uint32_t mi_rows_of(int height) {
  return 2U * ((static_cast<std::uint32_t>(height) + 7U) >> 3U);
}

void write_tile_info(BitWriter& writer, int width, int height) {
  const std::uint32_t sb_cols = (mi_cols_of(width) + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows_of(height) + 15U) >> 4U;
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) writer.put_bit(false);
  if (sb_rows > 1U) writer.put_bit(false);
}

// §5.4 sequence header.  The M2 layout, with our motion-mode and
// MV-precision slices spliced in at their normative positions.
std::vector<std::uint8_t> make_sequence_header(const StreamConfig& config) {
  BitWriter writer;
  put_uvlc(writer, 0);              // seq_header_id
  writer.put_bits(0U, 5U);          // seq_profile_idc
  writer.put_bit(false);            // single_picture_header_flag
  writer.put_bits(7U, 5U);          // seq_level_idx = 5.1
  writer.put_bit(false);            // seq_tier
  put_uvlc(writer, 0);              // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);              // bit_depth_idc = 8-bit
  writer.put_bits(0U, 3U);          // seq_lcr_id
  writer.put_bit(false);            // still_picture
  writer.put_bits(0U, 2U);          // max_tlayer_id
  writer.put_bits(0U, 3U);          // max_mlayer_id
  writer.put_bit(true);             // monotonic_output_order_flag
  unsigned width_bits = bit_width(static_cast<std::uint32_t>(config.width) - 1U);
  unsigned height_bits =
      bit_width(static_cast<std::uint32_t>(config.height) - 1U);
  if (width_bits == 0) width_bits = 1;
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(config.width) - 1U, width_bits);
  writer.put_bits(static_cast<std::uint64_t>(config.height) - 1U, height_bits);
  writer.put_bit(false);  // seq_cropping_window_present_flag
  writer.put_bit(false);  // seq_initial_display_delay_present_flag
  writer.put_bit(false);  // decoder_model_info_present_flag
  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(false);  // use_128x128_superblock
  writer.put_bit(false);  // enable_sdp
  writer.put_bit(false);  // enable_ext_partitions
  writer.put_bit(false);  // reduce_pb_aspect_ratio
  writer.put_bit(false);  // enable_ext_seg
  writer.put_bit(false);  // seq_seg_info_present_flag
  writer.put_bit(false);  // enable_dip
  writer.put_bit(true);   // enable_intra_edge_filter
  writer.put_bit(false);  // enable_mrls
  writer.put_bit(true);   // enable_cfl_intra
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);  // enable_mhccp
  writer.put_bit(false);  // enable_ibp

  // §5.4.6 sequence_inter_config(), motion-mode slice.
  std::string error;
  if (!mvprec::write_sequence_motion_modes(writer, config.motion_modes,
                                           &error)) {
    return {};
  }
  writer.put_bit(false);  // enable_masked_compound
  writer.put_bit(false);  // enable_ref_frame_mvs
  writer.put_bits(static_cast<std::uint64_t>(kOrderHintBits) - 1U, 4U);
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder
  writer.put_bit(true);   // explicit_ref_frame_map
  writer.put_bit(true);   // explicit_num_ref_frames
  writer.put_bits(static_cast<std::uint64_t>(config.num_ref_frames) - 1U, 4U);
  writer.put_bits(0U, 3U);  // long_term_frame_id_bits
  put_ns(writer, 5U, static_cast<std::uint32_t>(config.max_drl_bits_minus_1));
  writer.put_bit(false);  // allow_frame_max_drl_bits
  put_ns(writer, 3U, 0U);  // seq_max_bvp_drl_bits_minus_1
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
  // §5.4.6 MV-precision slice.
  if (!mvprec::write_sequence_mv_res(writer, config.mv_res, &error)) {
    return {};
  }
  writer.put_bit(false);  // enable_global_motion
  writer.put_bit(false);  // enable_short_refresh_frame_flags
  writer.put_bit(false);  // seq_choose_screen_content_tools
  writer.put_bit(false);  // seq_force_screen_content_tools
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
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q
  writer.put_bit(false);     // uv_ac_delta_q_enabled
  writer.put_bit(false);     // disable_loopfilters_across_tiles
  writer.put_bit(false);     // enable_cdef
  writer.put_bit(false);     // enable_gdf
  writer.put_bit(false);     // enable_restoration
  writer.put_bit(false);     // enable_ccso
  writer.put_bit(false);     // cdef_on_skip_txfm_always_on
  writer.put_bit(true);      // cdef_on_skip_txfm_disabled
  writer.put_bits(0U, 2U);   // df_par_bits_minus_2
  writer.put_bit(false);     // seq_tile_info_present_flag
  writer.put_bit(false);     // film_grain_params_present
  writer.put_bit(false);     // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

struct FrameHeader {
  bool keyframe = false;
  std::uint32_t order_hint = 0;
  std::uint32_t refresh_frame_flags = 0;
  int num_total_refs = 1;
  std::uint8_t ref_frame_idx[2] = {0, 0};
  int frame_precision = me2::kMvPrecisionEighthPel;
  bool is_filter_switchable = false;
  int interpolation_filter = 0;
  mvprec::MotionModeSet frame_motion_modes;
};

bool write_frame_header_syntax(BitWriter& writer, const StreamConfig& config,
                               const FrameHeader& header, std::string* error) {
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  if (!header.keyframe) writer.put_bit(true);  // frame_is_inter
  writer.put_bit(true);  // immediate_output_frame
  writer.put_bit(false);  // frame_size_override_flag
  writer.put_bits(header.order_hint, kOrderHintBits);
  if (header.keyframe) {
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
    writer.put_bits(header.refresh_frame_flags,
                    static_cast<unsigned>(config.num_ref_frames));
    writer.put_bit(true);  // frame_explicit_ref_frame_map
    writer.put_bits(static_cast<std::uint64_t>(header.num_total_refs), 3U);
    for (int i = 0; i < header.num_total_refs; ++i) {
      writer.put_bits(header.ref_frame_idx[i],
                      ceil_log2(static_cast<std::uint32_t>(
                          config.num_ref_frames)));
    }
    writer.put_bit(false);  // allow_intrabc

    me2::FramePrecisionSyntax frame;
    if (!me2::me2_frame_precision_to_bits(header.frame_precision,
                                          config.mv_res.enable_flex_mvres ? 1 : 0,
                                          0, &frame)) {
      *error = "frame precision is not signallable";
      return false;
    }
    if (!mvprec::write_frame_mv_precision(writer, frame, 0, error)) {
      return false;
    }
    if (!mvprec::write_frame_interp_filter(
            writer, header.is_filter_switchable,
            header.is_filter_switchable ? mvprec::kInterpSwitchable
                                        : header.interpolation_filter,
            error)) {
      return false;
    }
    if (!mvprec::write_frame_motion_modes(writer, config.motion_modes,
                                         header.frame_motion_modes, error)) {
      return false;
    }
  }
  writer.put_bit(true);  // disable_cdf_update
  write_tile_info(writer, config.width, config.height);
  writer.put_bits(kBaseQIdx, 8U);
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present
  writer.put_bit(false);  // apply_deblocking_filter[0]
  writer.put_bit(false);  // apply_deblocking_filter[1]
  writer.put_bit(false);  // tx_mode_select = 0
  if (!header.keyframe) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
  return true;
}

std::vector<std::uint8_t> make_frame_obu(const StreamConfig& config,
                                         const FrameHeader& header,
                                         std::span<const std::uint8_t> tile,
                                         std::string* error) {
  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  if (!write_frame_header_syntax(writer, config, header, error)) return {};
  writer.align_to_byte();
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile.begin(), tile.end());
  const ObuType type = header.keyframe ? ObuType::kClosedLoopKey
                                       : ObuType::kRegularTileGroup;
  return av2enc::basic::make_obu(ObuHeader{type}, payload);
}

std::vector<std::uint8_t> make_td() {
  return av2enc::basic::make_obu(ObuHeader{ObuType::kTemporalDelimiter},
                              std::span<const std::uint8_t>{});
}

}  // namespace

bool build_stream(const StreamInput& input, StreamOutput* output,
                  std::string* error) {
  if (output == nullptr || error == nullptr) {
    *error = "null output";
    return false;
  }
  *output = StreamOutput{};
  error->clear();
  if (input.config.width <= 0 || input.config.height <= 0 ||
      input.config.width % 64 != 0 || input.config.height % 64 != 0) {
    *error = "stream requires 64-multiple dimensions";
    return false;
  }
  if (input.key_tile.empty() || input.p1_tile.empty() || input.p2_tile.empty()) {
    *error = "missing tile payload";
    return false;
  }

  const std::vector<std::uint8_t> sequence =
      make_sequence_header(input.config);
  if (sequence.empty()) {
    *error = "sequence header writer failed";
    return false;
  }
  const std::vector<std::uint8_t> td = make_td();

  FrameHeader key;
  key.keyframe = true;
  key.order_hint = 0;
  key.refresh_frame_flags =
      (1U << static_cast<unsigned>(input.config.num_ref_frames)) - 1U;

  FrameHeader p1;
  p1.order_hint = 1;
  p1.refresh_frame_flags = 2U;  // slot 1
  p1.num_total_refs = 1;
  p1.ref_frame_idx[0] = 0;
  p1.frame_precision = me2::kMvPrecisionEighthPel;
  p1.is_filter_switchable = false;
  p1.interpolation_filter = 0;  // EIGHTTAP
  // The M2 key/P1 tiles were written with the M2 baseline; keep their frame
  // interpolation filter fixed and disable the per-frame motion modes.
  p1.frame_motion_modes = mvprec::MotionModeSet{};

  FrameHeader p2;
  p2.order_hint = 2;
  p2.refresh_frame_flags = 1U;  // slot 0
  p2.num_total_refs = 1;
  p2.ref_frame_idx[0] = 0;
  p2.frame_precision = me2::kMvPrecisionQuarterPel;
  p2.is_filter_switchable = true;
  p2.frame_motion_modes = mvprec::MotionModeSet{};  // frame override: all 0

  const std::vector<std::uint8_t> key_obu =
      make_frame_obu(input.config, key, input.key_tile, error);
  const std::vector<std::uint8_t> p1_obu =
      make_frame_obu(input.config, p1, input.p1_tile, error);
  const std::vector<std::uint8_t> p2_obu =
      make_frame_obu(input.config, p2, input.p2_tile, error);
  if (key_obu.empty() || p1_obu.empty() || p2_obu.empty()) {
    if (error->empty()) *error = "frame OBU writer failed";
    return false;
  }

  output->units.resize(3);
  output->units[0].insert(output->units[0].end(), td.begin(), td.end());
  output->units[0].insert(output->units[0].end(), sequence.begin(),
                          sequence.end());
  output->units[0].insert(output->units[0].end(), key_obu.begin(),
                          key_obu.end());
  output->units[1].insert(output->units[1].end(), td.begin(), td.end());
  output->units[1].insert(output->units[1].end(), p1_obu.begin(), p1_obu.end());
  output->units[2].insert(output->units[2].end(), td.begin(), td.end());
  output->units[2].insert(output->units[2].end(), p2_obu.begin(), p2_obu.end());
  output->sequence_bytes = sequence.size();
  return true;
}

}  // namespace av2enc::inter::mvprec::gate
