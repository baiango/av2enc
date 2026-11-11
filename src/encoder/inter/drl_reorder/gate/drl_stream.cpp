#include "encoder/inter/drl_reorder/gate/drl_stream.h"

#include <cassert>
#include <string>

#include "bitstream/bitwriter.h"
#include "bitstream/m0_headers.h"

namespace av2enc::inter::drl::gate {
namespace {

using av2enc::m0::ObuHeader;
using av2enc::m0::ObuType;

constexpr int kOrderHintBits = 4;
constexpr int kNumRefFrames = 2;
constexpr int kMaxDrlBitsMinus1 = 4;
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
  unsigned w = 0;
  while ((1U << w) < n) ++w;
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

// §5.4 sequence header, matching src/encoder/m2/m2_headers.cpp except for the
// §5.4.6 DRL reorder fields.
std::vector<std::uint8_t> make_sequence_header(int width, int height,
                                               bool disable, bool constrain) {
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
  unsigned width_bits = bit_width(static_cast<std::uint32_t>(width) - 1U);
  unsigned height_bits = bit_width(static_cast<std::uint32_t>(height) - 1U);
  if (width_bits == 0) width_bits = 1;
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(width) - 1U, width_bits);
  writer.put_bits(static_cast<std::uint64_t>(height) - 1U, height_bits);
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
  for (int mode = 1; mode < 5; ++mode) {
    writer.put_bit(false);  // seq_enabled_motion_modes[mode]
  }
  writer.put_bit(false);  // enable_masked_compound
  writer.put_bit(false);  // enable_ref_frame_mvs
  writer.put_bits(static_cast<std::uint64_t>(kOrderHintBits) - 1U, 4U);
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(disable);  // disable_drl_reorder
  if (!disable) {
    writer.put_bit(constrain);  // constrain_drl_reorder
  }
  writer.put_bit(true);   // explicit_ref_frame_map
  writer.put_bit(true);   // explicit_num_ref_frames
  writer.put_bits(static_cast<std::uint64_t>(kNumRefFrames) - 1U, 4U);
  writer.put_bits(0U, 3U);  // long_term_frame_id_bits
  put_ns(writer, 5U, kMaxDrlBitsMinus1);
  writer.put_bit(false);  // allow_frame_max_drl_bits
  put_ns(writer, 3U, 0U);
  writer.put_bit(false);  // allow_frame_max_bvp_drl_bits
  writer.put_bits(0U, 2U);  // num_same_ref_compound
  writer.put_bit(false);  // enable_tip
  writer.put_bit(false);  // enable_mv_traj
  writer.put_bit(false);  // enable_bawp
  writer.put_bit(false);  // enable_cwp
  writer.put_bit(false);  // enable_imp_msk_bld
  writer.put_bit(false);  // enable_df_sub_pu
  writer.put_bits(0U, 2U);  // enable_opfl_refine = REFINE_NONE
  writer.put_bit(false);  // enable_refinemv
  writer.put_bit(false);  // enable_bru
  writer.put_bit(false);  // enable_adaptive_mvd
  writer.put_bit(false);  // enable_mvd_sign_derive
  writer.put_bit(false);  // enable_flex_mvres
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
  return av2enc::m0::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

struct FrameHeader {
  bool keyframe = false;
  std::uint32_t order_hint = 0;
  std::uint32_t refresh_frame_flags = 1;
  int num_total_refs = 1;
  std::uint8_t ref_frame_idx[2] = {0, 0};
  bool reference_select = false;
};

void write_frame_header_syntax(BitWriter& writer, const FrameHeader& h,
                               int width, int height) {
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  if (!h.keyframe) writer.put_bit(true);  // frame_is_inter
  writer.put_bit(true);  // immediate_output_frame
  writer.put_bit(false);  // frame_size_override_flag
  writer.put_bits(h.order_hint, kOrderHintBits);
  if (h.keyframe) {
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
    writer.put_bits(h.refresh_frame_flags, kNumRefFrames);
    writer.put_bit(true);   // frame_explicit_ref_frame_map
    writer.put_bits(static_cast<std::uint64_t>(h.num_total_refs), 3U);
    for (int i = 0; i < h.num_total_refs; ++i) {
      writer.put_bits(h.ref_frame_idx[i],
                      ceil_log2(static_cast<std::uint32_t>(kNumRefFrames)));
    }
    writer.put_bit(false);  // allow_intrabc
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTAP
  }
  writer.put_bit(true);  // disable_cdf_update
  write_tile_info(writer, width, height);
  writer.put_bits(kBaseQIdx, 8U);
  writer.put_bit(false);     // segmentation_enabled
  writer.put_bit(false);     // using_qmatrix
  writer.put_bit(false);     // delta_q_present
  writer.put_bit(false);     // apply_deblocking_filter[0]
  writer.put_bit(false);     // apply_deblocking_filter[1]
  writer.put_bit(false);     // tx_mode_select = 0
  if (!h.keyframe) {
    writer.put_bit(h.reference_select);
    writer.put_bit(false);  // skip_mode_present
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
}

std::vector<std::uint8_t> make_frame_obu(const FrameHeader& header, int width,
                                         int height,
                                         std::span<const std::uint8_t> tile) {
  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_syntax(writer, header, width, height);
  writer.align_to_byte();
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile.begin(), tile.end());
  const ObuType type = header.keyframe ? ObuType::kClosedLoopKey
                                       : ObuType::kRegularTileGroup;
  return av2enc::m0::make_obu(ObuHeader{type}, payload);
}

std::vector<std::uint8_t> make_td() {
  return av2enc::m0::make_obu(ObuHeader{ObuType::kTemporalDelimiter},
                              std::span<const std::uint8_t>{});
}

}  // namespace

bool build_drl_stream(const StreamInput& input, StreamOutput* output,
                      std::string* error) {
  if (output == nullptr || error == nullptr) {
    return false;
  }
  error->clear();
  *output = StreamOutput{};
  if (input.width <= 0 || input.height <= 0 || input.width % 64 != 0 ||
      input.height % 64 != 0) {
    *error = "gate stream requires 64-multiple dimensions";
    return false;
  }
  if (input.key_tile.empty() || input.p1_tile.empty() || input.p2_tile.empty()) {
    *error = "gate stream is missing a tile payload";
    return false;
  }

  const std::vector<std::uint8_t> seq =
      make_sequence_header(input.width, input.height,
                           input.disable_drl_reorder,
                           input.constrain_drl_reorder);
  const std::vector<std::uint8_t> td = make_td();

  FrameHeader key;
  key.keyframe = true;
  key.order_hint = 0;
  key.refresh_frame_flags = (1U << kNumRefFrames) - 1U;
  FrameHeader p1;
  p1.order_hint = 1;
  p1.refresh_frame_flags = 2U;  // slot 1
  p1.num_total_refs = 1;
  p1.ref_frame_idx[0] = 0;
  p1.reference_select = false;
  FrameHeader p2;
  p2.order_hint = 2;
  p2.refresh_frame_flags = 1U;  // slot 0
  p2.num_total_refs = input.num_total_refs;
  p2.ref_frame_idx[0] = 0;  // slot 0 = key
  p2.ref_frame_idx[1] = 1;  // slot 1 = P1 (present when NumTotalRefs == 2)
  p2.reference_select = false;

  const std::vector<std::uint8_t> key_obu =
      make_frame_obu(key, input.width, input.height, input.key_tile);
  const std::vector<std::uint8_t> p1_obu =
      make_frame_obu(p1, input.width, input.height, input.p1_tile);
  const std::vector<std::uint8_t> p2_obu =
      make_frame_obu(p2, input.width, input.height, input.p2_tile);

  output->units.resize(3);
  output->units[0].insert(output->units[0].end(), td.begin(), td.end());
  output->units[0].insert(output->units[0].end(), seq.begin(), seq.end());
  output->units[0].insert(output->units[0].end(), key_obu.begin(),
                          key_obu.end());
  output->units[1].insert(output->units[1].end(), td.begin(), td.end());
  output->units[1].insert(output->units[1].end(), p1_obu.begin(), p1_obu.end());
  output->units[2].insert(output->units[2].end(), td.begin(), td.end());
  output->units[2].insert(output->units[2].end(), p2_obu.begin(), p2_obu.end());
  return true;
}

}  // namespace av2enc::inter::drl::gate
