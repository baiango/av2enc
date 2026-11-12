#include "codec/inter/tlayers/tlayers_syntax.h"

#include "core/bitstream/bitwriter.h"

#include <cassert>
#include <cstddef>
#include <string>

namespace av2enc::inter::tlayers {
namespace {

// OBU types (§6.2.2, Table 6.1) used by this layer.
constexpr int kObuSequenceHeader = 1;
constexpr int kObuTemporalDelimiter = 2;
constexpr int kObuClosedLoopKey = 4;
constexpr int kObuRegularTileGroup = 7;

// §4.11.3 uvlc().
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

// §4.11.8 ns(n).
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

std::uint32_t mi_cols(const SequenceConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
}
std::uint32_t mi_rows(const SequenceConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
}

void write_tile_info(BitWriter& writer, const SequenceConfig& c) {
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

// §5.4.1 sequence_header_obu(). Mirrors av2enc::video::write_sequence_header() and adds
// the temporal-layer fields. For max_tlayer_id == 0 the output is identical.
std::size_t write_sequence_header(BitWriter& writer, const SequenceConfig& c) {
  const std::size_t start = writer.bit_size();
  assert(c.order_hint_bits >= 1U && c.order_hint_bits <= 16U);
  assert(c.num_ref_frames >= 1U && c.num_ref_frames <= 16U);
  assert(c.max_drl_bits_minus_1 < 5U);
  assert(c.max_tlayer_id >= 0 && c.max_tlayer_id <= kMaxTemporalLayerId);

  put_uvlc(writer, 0);                   // seq_header_id
  writer.put_bits(0U, 5U);               // seq_profile_idc
  writer.put_bit(false);                 // single_picture_header_flag = 0
  writer.put_bits(c.seq_level_idx, 5U);  // seq_level_idx
  // §5.4.1: seq_tier is absent for seq_level_idx <= 3 (and this writer always
  // sets single_picture_header_flag = 0).
  if (c.seq_level_idx > 3U) {
    writer.put_bit(false);               // seq_tier = 0 (Main tier)
  }
  put_uvlc(writer, 0);                   // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                   // bit_depth_idc = 1 => 8-bit

  // §5.4.1 non-single-picture header.
  writer.put_bits(0U, 3U);  // seq_lcr_id
  writer.put_bit(false);    // still_picture
  writer.put_bits(static_cast<std::uint64_t>(c.max_tlayer_id), 2U);
  writer.put_bits(0U, 3U);  // max_mlayer_id
  // SeqMaxMlayerCnt = 1; the seq_max_mlayer_cnt_minus_1 syntax is absent.
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
  writer.put_bit(false);  // seq_initial_display_delay_present_flag
  writer.put_bit(false);  // decoder_model_info_present_flag

  // §5.4.1 temporal layers. TLayerDependencyMap and MLayerDependencyMap have
  // already been initialised by the spec to (ref <= curr). max_mlayer_id == 0,
  // so the MLayerDependencyMap override block is absent and
  // multi_tlayer_dependency_map_present_flag is inferred 0 (no bit).
  if (c.max_tlayer_id > 0) {
    writer.put_bit(c.tlayer_dependency_present_flag);
    if (c.tlayer_dependency_present_flag) {
      for (int curr = 1; curr <= c.max_tlayer_id; ++curr) {
        for (int ref = curr; ref >= 0; --ref) {
          writer.put_bit(
              c.tlayer_dependency[static_cast<std::size_t>(curr)]
                                  [static_cast<std::size_t>(ref)]);
        }
      }
    }
  }

  // §5.4.3 sequence_partition_config().
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
  writer.put_bit(c.enable_cfl_intra);
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);    // enable_mhccp
  writer.put_bit(false);    // enable_ibp

  // §5.4.6 sequence_inter_config().
  for (int mode = 1; mode < 5; ++mode) {  // INTERINTRA..EXTENDWARP
    writer.put_bit(false);                // seq_enabled_motion_modes[mode]
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
  writer.put_bit(false);   // enable_adaptive_mvd
  writer.put_bit(false);   // enable_mvd_sign_derive
  writer.put_bit(false);   // enable_flex_mvres
  writer.put_bit(false);   // enable_global_motion
  writer.put_bit(false);   // enable_short_refresh_frame_flags

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
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q
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

// §5.18.2 frame_header_info(). Mirrors m2's writer and adds the output and
// reference fields needed by reordered temporal-layer streams.
void write_frame_header_syntax(BitWriter& writer, const SequenceConfig& c,
                               const FrameHeader& h) {
  assert(c.base_q_idx >= 1U);
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header

  const bool key = h.keyframe;
  if (!key) {
    writer.put_bit(true);  // frame_is_inter = 1 -> INTER_FRAME
  }
  // long_term_id_plus_1 absent (long_term_frame_id_bits == 0).
  writer.put_bit(h.immediate_output_frame);
  if (!h.immediate_output_frame && !c.monotonic_output_order_flag) {
    writer.put_bit(h.implicit_output_frame);
  } else {
    assert(!h.implicit_output_frame ||
           (!h.immediate_output_frame && !c.monotonic_output_order_flag));
  }

  if (key) {
    // OBU_CLOSED_LOOP_KEY + max_mlayer_id == 0 -> refresh_frame_flags =
    // allFrames, no bits coded.
  }

  writer.put_bit(false);  // frame_size_override_flag = 0
  writer.put_bits(h.order_hint_lsbs, c.order_hint_bits);  // order_hint

  if (key) {
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
    // §5.18.2 refresh_frame_flags; enable_short_refresh_frame_flags == 0.
    writer.put_bits(h.refresh_frame_flags, c.num_ref_frames);
    if (h.frame_explicit_ref_frame_map) {
      writer.put_bit(true);  // frame_explicit_ref_frame_map
    } else {
      writer.put_bit(false);
    }
    assert(h.num_total_refs <= kMaxRefs);
    writer.put_bits(h.num_total_refs, 3U);
    const unsigned idx_bits =
        static_cast<unsigned>(ceil_log2(c.num_ref_frames));
    for (int r = 0; r < h.num_total_refs; ++r) {
      writer.put_bits(h.ref_frame_idx[static_cast<std::size_t>(r)], idx_bits);
    }
    writer.put_bit(false);  // allow_intrabc
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv -> 1/8 pel
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTTAP
  }

  writer.put_bit(c.disable_cdf_update);  // disable_cdf_update

  write_tile_info(writer, c);

  writer.put_bits(c.base_q_idx, 8U);  // base_q_idx
  writer.put_bit(false);              // segmentation_enabled
  writer.put_bit(false);              // using_qmatrix
  writer.put_bit(false);              // delta_q_present
  writer.put_bit(false);              // apply_deblocking_filter[0]
  writer.put_bit(false);              // apply_deblocking_filter[1]
  writer.put_bit(false);              // tx_mode_select = 0
  if (!key) {
    writer.put_bit(h.reference_select);
    writer.put_bit(false);  // skip_mode_present
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
}

// --- bit reader for the golden/validation parse ---------------------------

class BitReader {
 public:
  BitReader(std::span<const std::uint8_t> bytes, std::size_t byte_offset)
      : bytes_(bytes), bit_pos_(byte_offset * 8U) {}

  bool get_bit(bool& out) {
    if (bit_pos_ >= bytes_.size() * 8U) {
      return false;
    }
    const std::size_t byte = bit_pos_ >> 3U;
    const unsigned shift = 7U - static_cast<unsigned>(bit_pos_ & 7U);
    out = ((bytes_[byte] >> shift) & 1U) != 0U;
    ++bit_pos_;
    return true;
  }

  bool get_bits(unsigned count, std::uint32_t& out) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < count; ++i) {
      bool bit = false;
      if (!get_bit(bit)) {
        return false;
      }
      value = (value << 1U) | (bit ? 1U : 0U);
    }
    out = value;
    return true;
  }

  bool get_uvlc(std::uint32_t& out) {
    unsigned leading_zeros = 0;
    bool bit = false;
    while (true) {
      if (!get_bit(bit)) {
        return false;
      }
      if (bit) {
        break;
      }
      ++leading_zeros;
      if (leading_zeros > 31U) {
        return false;
      }
    }
    std::uint32_t payload = 0;
    if (leading_zeros > 0 && !get_bits(leading_zeros, payload)) {
      return false;
    }
    out = payload + ((1U << leading_zeros) - 1U);
    return true;
  }

  bool get_ns(std::uint32_t n, std::uint32_t& out) {
    if (n < 2U) {
      return false;
    }
    unsigned w = 0;
    while ((1U << w) < n) {
      ++w;
    }
    const std::uint32_t m = (1U << w) - n;
    std::uint32_t value = 0;
    if (!get_bits(w - 1U, value)) {
      return false;
    }
    if (value < m) {
      out = value;
      return true;
    }
    bool extra = false;
    if (!get_bit(extra)) {
      return false;
    }
    out = ((value << 1U) | (extra ? 1U : 0U)) - m;
    return true;
  }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_pos_ = 0;
};

bool read_leb128(std::span<const std::uint8_t> bytes, std::size_t& offset,
                 std::uint32_t& value) {
  value = 0;
  unsigned shift = 0;
  while (offset < bytes.size()) {
    const std::uint8_t byte = bytes[offset++];
    value |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
    if ((byte & 0x80U) == 0U) {
      return true;
    }
    shift += 7U;
    if (shift >= 32U) {
      return false;
    }
  }
  return false;
}

}  // namespace

void SequenceConfig::apply_plan(const Plan& plan) {
  max_tlayer_id = plan.max_tlayer_id;
  monotonic_output_order_flag = false;
  tlayer_dependency_present_flag = plan.tlayer_dependency_present_flag;
  tlayer_dependency = plan.tlayer_dependency;
  order_hint_bits = static_cast<std::uint8_t>(plan.config.order_hint_bits);
  num_ref_frames = static_cast<std::uint8_t>(plan.num_slots);
  base_q_idx = plan.config.base_q_idx;
}

SequenceConfig baseline_sequence_config(std::uint16_t width,
                                        std::uint16_t height,
                                        std::uint8_t base_q_idx,
                                        std::uint8_t num_ref_frames) {
  SequenceConfig c;
  c.max_frame_width = width;
  c.max_frame_height = height;
  c.base_q_idx = base_q_idx;
  c.num_ref_frames = num_ref_frames;
  c.max_tlayer_id = 0;
  c.monotonic_output_order_flag = true;
  c.tlayer_dependency_present_flag = false;
  for (auto& row : c.tlayer_dependency) {
    row.fill(false);
  }
  c.tlayer_dependency[0][0] = true;
  return c;
}

FrameHeader frame_header_for(const SequenceConfig& config, const Plan& plan,
                             int display_index) {
  const PlannedFrame& f =
      plan.frames[static_cast<std::size_t>(display_index)];
  FrameHeader h;
  h.keyframe = f.is_key;
  h.order_hint_lsbs = f.order_hint_lsbs;
  h.refresh_frame_flags = f.refresh_frame_flags;
  h.frame_explicit_ref_frame_map = true;
  h.num_total_refs = static_cast<std::uint8_t>(f.num_refs);
  for (int i = 0; i < f.num_refs; ++i) {
    h.ref_frame_idx[static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>(f.ref_slot[i]);
  }
  h.immediate_output_frame = f.immediate_output_frame;
  h.implicit_output_frame = f.implicit_output_frame;
  h.reference_select = f.num_refs >= 2;
  // The sequence config must describe the same reference count used by the
  // plan's slot indices.
  assert(config.num_ref_frames == plan.num_slots);
  return h;
}

std::uint8_t obu_header_byte(bool extension_flag, int obu_type,
                             int obu_tlayer_id) {
  assert(obu_type >= 0 && obu_type <= 31);
  assert(obu_tlayer_id >= 0 && obu_tlayer_id <= kMaxTemporalLayerId);
  return static_cast<std::uint8_t>(
      (extension_flag ? 0x80U : 0x00U) |
      ((static_cast<unsigned>(obu_type) & 0x1FU) << 2U) |
      (static_cast<unsigned>(obu_tlayer_id) & 0x03U));
}

std::vector<std::uint8_t> make_obu(std::uint8_t obu_type,
                                   std::uint8_t obu_tlayer_id,
                                   std::span<const std::uint8_t> payload) {
  assert(obu_type <= 31U);
  assert(obu_tlayer_id <= kMaxTemporalLayerId);
  const std::size_t obu_size = 1U + payload.size();
  std::vector<std::uint8_t> out;
  // Annex B.2 leb128(num_bytes_in_obu). 1 header byte + payload.
  std::uint32_t remaining = static_cast<std::uint32_t>(obu_size);
  do {
    std::uint8_t byte = static_cast<std::uint8_t>(remaining & 0x7FU);
    remaining >>= 7U;
    if (remaining != 0U) {
      byte = static_cast<std::uint8_t>(byte | 0x80U);
    }
    out.push_back(byte);
  } while (remaining != 0U);
  out.push_back(obu_header_byte(false, obu_type, obu_tlayer_id));
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

std::vector<std::uint8_t> make_sequence_header_obu(
    const SequenceConfig& config) {
  BitWriter writer;
  write_sequence_header(writer, config);
  writer.put_bit(false);  // obu_header_extension_flag
  put_trailing_bits(writer);
  return make_obu(static_cast<std::uint8_t>(kObuSequenceHeader), 0,
                  writer.take_bytes());
}

std::vector<std::uint8_t> make_temporal_delimiter_obu() {
  return make_obu(static_cast<std::uint8_t>(kObuTemporalDelimiter), 0, {});
}

std::vector<std::uint8_t> make_frame_obu(
    const SequenceConfig& config, const FrameHeader& header,
    std::uint8_t obu_tlayer_id, std::span<const std::uint8_t> tile_payload) {
  // §6.2.2: key OBUs (CLK/OLK/SWITCH/RAS) require obu_tlayer_id == 0.
  assert(!header.keyframe || obu_tlayer_id == 0U);
  assert(obu_tlayer_id <= static_cast<std::uint8_t>(config.max_tlayer_id));
  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_syntax(writer, config, header);
  writer.align_to_byte();  // byte_alignment()
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  const std::uint8_t type = header.keyframe
                                ? static_cast<std::uint8_t>(kObuClosedLoopKey)
                                : static_cast<std::uint8_t>(kObuRegularTileGroup);
  return make_obu(type, obu_tlayer_id, payload);
}

bool parse_obu(std::span<const std::uint8_t> bytes, std::size_t offset,
               ParsedObuHeader& out, std::string& error) {
  std::uint32_t obu_size = 0;
  std::size_t pos = offset;
  if (!read_leb128(bytes, pos, obu_size)) {
    error = "invalid OBU leb128 length";
    return false;
  }
  if (obu_size == 0 || pos >= bytes.size()) {
    error = "empty OBU";
    return false;
  }
  const std::uint8_t header = bytes[pos];
  out.extension_flag = (header & 0x80U) != 0U;
  out.obu_type = static_cast<int>((header >> 2U) & 0x1FU);
  out.obu_tlayer_id = static_cast<int>(header & 0x03U);
  out.obu_mlayer_id = 0;
  out.obu_xlayer_id = 0;
  std::size_t header_bytes = 1;
  if (out.extension_flag) {
    if (pos + 1 >= bytes.size()) {
      error = "truncated OBU extension header";
      return false;
    }
    const std::uint8_t ext = bytes[pos + 1];
    out.obu_mlayer_id = static_cast<int>((ext >> 5U) & 0x07U);
    out.obu_xlayer_id = static_cast<int>(ext & 0x1FU);
    header_bytes = 2;
  }
  const std::size_t total = static_cast<std::size_t>(obu_size);
  if (pos + total > bytes.size() || total < header_bytes) {
    error = "truncated OBU payload";
    return false;
  }
  out.header_bytes = header_bytes;
  out.total_bytes = (pos - offset) + total;
  out.payload = bytes.subspan(pos + header_bytes,
                              static_cast<std::size_t>(obu_size) - header_bytes);
  return true;
}

bool parse_sequence_header_obu(std::span<const std::uint8_t> obu,
                               ParsedSequenceHeader& out, std::string& error) {
  ParsedObuHeader header;
  if (!parse_obu(obu, 0, header, error)) {
    return false;
  }
  if (header.obu_type != kObuSequenceHeader) {
    error = "not a sequence header OBU";
    return false;
  }
  if (header.obu_tlayer_id != 0) {
    error = "OBU_SEQUENCE_HEADER must have obu_tlayer_id == 0 (§6.2.2)";
    return false;
  }
  BitReader br(header.payload, 0);
  std::uint32_t v = 0;
  if (!br.get_uvlc(v)) {
    error = "seq_header_id";
    return false;
  }
  out.seq_header_id = static_cast<int>(v);
  if (!br.get_bits(5, v)) {
    error = "seq_profile_idc";
    return false;
  }
  out.seq_profile_idc = static_cast<int>(v);
  bool bit = false;
  if (!br.get_bit(bit)) {
    error = "single_picture_header_flag";
    return false;
  }
  out.single_picture_header_flag = bit;
  if (!br.get_bits(5, v)) {
    error = "seq_level_idx";
    return false;
  }
  out.seq_level_idx = static_cast<int>(v);
  if (out.seq_level_idx > 3 && !out.single_picture_header_flag) {
    if (!br.get_bit(bit)) {
      error = "seq_tier";
      return false;
    }
  }
  if (!br.get_uvlc(v)) {
    error = "chroma_format_idc";
    return false;
  }
  if (!br.get_uvlc(v)) {
    error = "bit_depth_idc";
    return false;
  }
  if (out.single_picture_header_flag) {
    out.max_tlayer_id = 0;
    out.max_mlayer_id = 0;
    out.monotonic_output_order_flag = true;
  } else {
    if (!br.get_bits(3, v)) {
      error = "seq_lcr_id";
      return false;
    }
    if (!br.get_bit(bit)) {
      error = "still_picture";
      return false;
    }
    if (!br.get_bits(2, v)) {
      error = "max_tlayer_id";
      return false;
    }
    out.max_tlayer_id = static_cast<int>(v);
    if (!br.get_bits(3, v)) {
      error = "max_mlayer_id";
      return false;
    }
    out.max_mlayer_id = static_cast<int>(v);
    if (out.max_mlayer_id > 0) {
      const std::uint32_t n = ceil_log2(
          static_cast<std::uint32_t>(out.max_mlayer_id) + 1U);
      if (!br.get_bits(n, v)) {
        error = "seq_max_mlayer_cnt_minus_1";
        return false;
      }
    }
    if (!br.get_bit(bit)) {
      error = "monotonic_output_order_flag";
      return false;
    }
    out.monotonic_output_order_flag = bit;
  }

  std::uint32_t width_bits_minus_1 = 0;
  std::uint32_t height_bits_minus_1 = 0;
  if (!br.get_bits(4, width_bits_minus_1) ||
      !br.get_bits(4, height_bits_minus_1)) {
    error = "frame size bits";
    return false;
  }
  std::uint32_t width_minus_1 = 0;
  std::uint32_t height_minus_1 = 0;
  if (!br.get_bits(width_bits_minus_1 + 1U, width_minus_1) ||
      !br.get_bits(height_bits_minus_1 + 1U, height_minus_1)) {
    error = "frame size";
    return false;
  }
  out.max_frame_width = static_cast<int>(width_minus_1) + 1;
  out.max_frame_height = static_cast<int>(height_minus_1) + 1;

  if (!br.get_bit(bit)) {
    error = "seq_cropping_window_present_flag";
    return false;
  }
  if (bit) {
    for (int i = 0; i < 4; ++i) {
      if (!br.get_uvlc(v)) {
        error = "seq_cropping_window";
        return false;
      }
    }
  }
  if (!out.single_picture_header_flag) {
    if (!br.get_bit(bit)) {
      error = "seq_initial_display_delay_present_flag";
      return false;
    }
    if (bit) {
      if (!br.get_bits(4, v)) {
        error = "seq_initial_display_delay_minus_1";
        return false;
      }
    }
    if (!br.get_bit(bit)) {
      error = "decoder_model_info_present_flag";
      return false;
    }
    if (bit) {
      error = "decoder model info is not supported by this reader";
      return false;
    }
  }

  // §5.4.1 defaults: TLayerDependencyMap[0][curr][ref] = (ref <= curr).
  for (int curr = 0; curr < kMaxTLayers; ++curr) {
    for (int ref = 0; ref < kMaxTLayers; ++ref) {
      out.tlayer_dependency[static_cast<std::size_t>(curr)]
                           [static_cast<std::size_t>(ref)] =
          (ref <= curr && curr <= out.max_tlayer_id);
    }
  }
  out.tlayer_dependency[0][0] = true;

  if (out.max_mlayer_id > 0) {
    if (!br.get_bit(bit)) {
      error = "mlayer_dependency_present_flag";
      return false;
    }
    if (bit) {
      for (int curr = 1; curr <= out.max_mlayer_id; ++curr) {
        for (int ref = curr; ref >= 0; --ref) {
          bool dep = false;
          if (!br.get_bit(dep)) {
            error = "mlayer_dependency_map";
            return false;
          }
        }
      }
    }
  }

  if (out.max_tlayer_id > 0) {
    if (!br.get_bit(bit)) {
      error = "tlayer_dependency_present_flag";
      return false;
    }
    out.tlayer_dependency_present_flag = bit;
    if (bit) {
      if (out.max_mlayer_id > 0) {
        if (!br.get_bit(bit)) {
          error = "multi_tlayer_dependency_map_present_flag";
          return false;
        }
        out.multi_tlayer_dependency_map_present_flag = bit;
      }
      for (int mlayer = 0; mlayer <= out.max_mlayer_id; ++mlayer) {
        for (int curr = 1; curr <= out.max_tlayer_id; ++curr) {
          for (int ref = curr; ref >= 0; --ref) {
            bool dep = false;
            if (!br.get_bit(dep)) {
              error = "tlayer_dependency_map";
              return false;
            }
            const bool coded =
                out.multi_tlayer_dependency_map_present_flag || mlayer == 0;
            if (coded && mlayer == 0) {
              out.tlayer_dependency[static_cast<std::size_t>(curr)]
                                   [static_cast<std::size_t>(ref)] = dep;
              out.tlayer_dependency_coded[static_cast<std::size_t>(curr)]
                                        [static_cast<std::size_t>(ref)] = true;
            }
          }
        }
      }
    }
  }

  // Skip to OrderHintBits: partition, segment, intra configs.
  for (int i = 0; i < 5; ++i) {  // sequence_partition_config
    if (!br.get_bit(bit)) {
      error = "sequence_partition_config";
      return false;
    }
  }
  for (int i = 0; i < 2; ++i) {  // sequence_segment_config
    if (!br.get_bit(bit)) {
      error = "sequence_segment_config";
      return false;
    }
  }
  bool enable_cfl = false;
  if (!br.get_bit(bit) || !br.get_bit(bit) || !br.get_bit(bit) ||
      !br.get_bit(enable_cfl) || !br.get_bits(2, v) || !br.get_bit(bit) ||
      !br.get_bit(bit)) {
    error = "sequence_intra_config";
    return false;
  }
  (void)enable_cfl;

  bool motion_mode_enabled = false;
  for (int mode = 1; mode < 5; ++mode) {
    if (!br.get_bit(bit)) {
      error = "seq_enabled_motion_modes";
      return false;
    }
    motion_mode_enabled = motion_mode_enabled || bit;
  }
  if (motion_mode_enabled) {
    if (!br.get_bit(bit)) {
      error = "seq_frame_motion_modes_present_flag";
      return false;
    }
  }
  bool masked = false;
  bool ref_frame_mvs = false;
  if (!br.get_bit(masked) || !br.get_bit(ref_frame_mvs)) {
    error = "sequence_inter_config";
    return false;
  }
  (void)masked;
  if (ref_frame_mvs) {
    if (!br.get_bit(bit)) {
      error = "reduced_ref_frame_mvs_mode";
      return false;
    }
  }
  std::uint32_t order_hint_bits_minus_1 = 0;
  if (!br.get_bits(4, order_hint_bits_minus_1)) {
    error = "order_hint_bits_minus_1";
    return false;
  }
  out.order_hint_bits = static_cast<int>(order_hint_bits_minus_1) + 1;
  bool refmvbank = false;
  bool disable_drl = false;
  if (!br.get_bit(refmvbank) || !br.get_bit(disable_drl)) {
    error = "sequence_inter_config";
    return false;
  }
  (void)refmvbank;
  if (!disable_drl) {
    if (!br.get_bit(bit)) {
      error = "constrain_drl_reorder";
      return false;
    }
  }
  bool explicit_ref_frame_map = false;
  bool explicit_num_ref_frames = false;
  if (!br.get_bit(explicit_ref_frame_map) ||
      !br.get_bit(explicit_num_ref_frames)) {
    error = "explicit reference config";
    return false;
  }
  (void)explicit_ref_frame_map;
  if (explicit_num_ref_frames) {
    std::uint32_t num_ref_frames_minus_1 = 0;
    if (!br.get_bits(4, num_ref_frames_minus_1)) {
      error = "num_ref_frames_minus_1";
      return false;
    }
    out.num_ref_frames = static_cast<int>(num_ref_frames_minus_1) + 1;
  } else {
    out.num_ref_frames = 8;
  }
  // We stop here; enable_short_refresh_frame_flags follows much later and is
  // pinned to 0 by the writer. The golden test reads it from the plan.
  out.enable_short_refresh_frame_flags = false;
  return true;
}

const char* tlayer_role_name(int obu_type) {
  switch (obu_type) {
    case kObuTemporalDelimiter:
      return "temporal-delimiter";
    case kObuSequenceHeader:
      return "sequence-header";
    case kObuClosedLoopKey:
      return "closed-loop-key";
    case kObuRegularTileGroup:
      return "regular-tile-group";
    default:
      return "other";
  }
}

}  // namespace av2enc::inter::tlayers
