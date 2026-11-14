#include "codec/inter/random_access/random_access.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>
#include <utility>

#include "core/bitstream/basic_headers.h"

namespace av2enc::inter::random_access {
namespace {

// ---------------------------------------------------------------------------
// Bit helpers (§4.11.6 uvlc, §4.11.8 ns, §5.2.3 trailing_bits)
// ---------------------------------------------------------------------------

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

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

std::uint32_t mi_cols_for(std::uint32_t width) {
  return 2U * ((width + 7U) >> 3U);
}
std::uint32_t mi_rows_for(std::uint32_t height) {
  return 2U * ((height + 7U) >> 3U);
}

// ---------------------------------------------------------------------------
// Minimal MSB-first bit reader (§4.11)
// ---------------------------------------------------------------------------

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  bool read_bit(bool& value) {
    if (byte_index_ >= bytes_.size() || bit_index_ >= 8U) {
      if (byte_index_ >= bytes_.size()) {
        error_ = "truncated bitstream";
        return false;
      }
    }
    const std::uint8_t byte = bytes_[byte_index_];
    value = ((byte >> (7U - bit_index_)) & 1U) != 0U;
    ++bit_index_;
    if (bit_index_ == 8U) {
      bit_index_ = 0;
      ++byte_index_;
    }
    ++bits_read_;
    return true;
  }

  bool read_bits(unsigned count, std::uint64_t& value) {
    if (count > 64U) {
      error_ = "bit count exceeds 64";
      return false;
    }
    value = 0;
    for (unsigned i = 0; i < count; ++i) {
      bool bit = false;
      if (!read_bit(bit)) return false;
      value = (value << 1U) | (bit ? 1U : 0U);
    }
    return true;
  }

  bool read_uvlc(std::uint32_t& value) {
    unsigned leading_zeros = 0;
    bool bit = false;
    while (true) {
      if (!read_bit(bit)) return false;
      if (bit) break;
      ++leading_zeros;
      if (leading_zeros >= 32U) {
        error_ = "uvlc leading zeros exceed 31";
        return false;
      }
    }
    std::uint64_t payload = 0;
    if (!read_bits(leading_zeros, payload)) return false;
    const std::uint64_t decoded =
        ((1ULL << leading_zeros) - 1ULL) + payload;
    if (decoded > std::numeric_limits<std::uint32_t>::max()) {
      error_ = "uvlc value exceeds 32 bits";
      return false;
    }
    value = static_cast<std::uint32_t>(decoded);
    return true;
  }

  bool read_ns(std::uint32_t n, std::uint32_t& value) {
    if (n < 2U) {
      error_ = "ns() requires n >= 2";
      return false;
    }
    unsigned w = 0;
    while ((1U << w) < n) {
      ++w;
    }
    const std::uint32_t m = (1U << w) - n;
    std::uint64_t x = 0;
    if (!read_bits(w - 1U, x)) return false;
    if (x < m) {
      value = static_cast<std::uint32_t>(x);
      return true;
    }
    bool extra = false;
    if (!read_bit(extra)) return false;
    const std::uint64_t y = (x << 1U) | (extra ? 1U : 0U);
    const std::uint64_t decoded = y - m;
    if (decoded >= n) {
      error_ = "ns() value out of range";
      return false;
    }
    value = static_cast<std::uint32_t>(decoded);
    return true;
  }

  bool align_to_byte() {
    while ((bits_read_ & 7U) != 0U) {
      bool bit = false;
      if (!read_bit(bit)) return false;
      if (bit) {
        error_ = "non-zero byte_alignment bit";
        return false;
      }
    }
    return true;
  }

  std::size_t bits_read() const { return bits_read_; }
  const std::string& error() const { return error_; }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t byte_index_ = 0;
  unsigned bit_index_ = 0;
  std::size_t bits_read_ = 0;
  std::string error_;
};

bool read_flag(BitReader& reader, bool& value, const char* what) {
  bool bit = false;
  if (!reader.read_bit(bit)) return false;
  value = bit;
  (void)what;
  return true;
}

bool expect_flag(BitReader& reader, bool expected, const char* what,
                 bool& supported) {
  bool value = false;
  if (!reader.read_bit(value)) return false;
  (void)what;
  if (value != expected) supported = false;
  return true;
}

bool read_bits_u32(BitReader& reader, unsigned count, std::uint32_t& value) {
  std::uint64_t wide = 0;
  if (!reader.read_bits(count, wide)) return false;
  value = static_cast<std::uint32_t>(wide);
  return true;
}

// ---------------------------------------------------------------------------
// OBU framing (§5.2.1/§5.2.2, Annex B.2)
// ---------------------------------------------------------------------------

bool read_leb128(std::span<const std::uint8_t> bytes, std::size_t& offset,
                 std::uint32_t& value, std::string& error) {
  value = 0;
  unsigned shift = 0;
  while (true) {
    if (offset >= bytes.size()) {
      error = "truncated leb128";
      return false;
    }
    const std::uint8_t byte = bytes[offset++];
    if (shift >= 32U) {
      error = "leb128 exceeds 32 bits";
      return false;
    }
    value |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
    if ((byte & 0x80U) == 0) break;
    shift += 7U;
  }
  return true;
}

std::vector<std::uint8_t> make_obu(ObuType type,
                                   std::span<const std::uint8_t> payload) {
  // The M0 OBU writer codes obu_type as a raw 5-bit field, so the two values
  // AV2 adds beyond M0 (10 and 21) are valid inputs. Keep the cast local and
  // documented rather than duplicating the Annex B framing.
  av2enc::basic::ObuHeader header;
  header.type = static_cast<av2enc::basic::ObuType>(
      static_cast<std::uint8_t>(type));
  return av2enc::basic::make_obu(header, payload);
}

// ---------------------------------------------------------------------------
// Fixed sequence-header profile (§5.4.1-§5.4.10)
// ---------------------------------------------------------------------------

std::size_t write_sequence_header_bits(BitWriter& writer,
                                       const SequenceConfig& c) {
  assert(c.order_hint_bits >= 1U && c.order_hint_bits <= 16U);
  assert(c.num_ref_frames >= 1U && c.num_ref_frames <= kNumRefFrames);
  assert(c.max_drl_bits_minus_1 < 5U);
  assert(c.long_term_frame_id_bits <= 7U);
  assert(c.max_frame_width > 0 && c.max_frame_height > 0);

  const std::size_t start = writer.bit_size();
  put_uvlc(writer, 0);          // seq_header_id
  writer.put_bits(0U, 5U);      // seq_profile_idc
  writer.put_bit(false);        // single_picture_header_flag = 0
  writer.put_bits(c.seq_level_idx, 5U);
  if (c.seq_level_idx > 3U) {
    writer.put_bit(false);      // seq_tier (level > 3)
  }
  put_uvlc(writer, 0);          // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);          // bit_depth_idc = 1 => 8-bit

  // §5.4.1 non-single-picture header.
  writer.put_bits(0U, 3U);      // seq_lcr_id
  writer.put_bit(false);        // still_picture
  writer.put_bits(0U, 2U);      // max_tlayer_id
  writer.put_bits(0U, 3U);      // max_mlayer_id
  // max_mlayer_id == 0 -> SeqMaxMlayerCnt = 1, no seq_max_mlayer_cnt_minus_1.
  writer.put_bit(true);         // monotonic_output_order_flag

  const std::uint32_t width_bits = c.frame_width_bits();
  const std::uint32_t height_bits = c.frame_height_bits();
  writer.put_bits(width_bits - 1U, 4U);   // frame_width_bits_minus_1
  writer.put_bits(height_bits - 1U, 4U);  // frame_height_bits_minus_1
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
  writer.put_bit(c.enable_cfl_intra);  // enable_cfl_intra
  writer.put_bits(0U, 2U);             // cfl_ds_filter_index
  writer.put_bit(false);               // enable_mhccp
  writer.put_bit(false);               // enable_ibp

  // §5.4.6 sequence_inter_config().
  for (int mode = 1; mode < 5; ++mode) {  // INTERINTRA..EXTENDWARP
    writer.put_bit(false);                // seq_enabled_motion_modes[mode]
  }
  // motionModeEnabled == 0 -> seq_frame_motion_modes_present_flag = 0.
  writer.put_bit(false);  // enable_masked_compound
  writer.put_bit(false);  // enable_ref_frame_mvs (no reduced_ref_frame_mvs)
  writer.put_bits(static_cast<std::uint64_t>(c.order_hint_bits) - 1U, 4U);
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder -> DRL_REORDER_DISABLED
  writer.put_bit(true);   // explicit_ref_frame_map
  writer.put_bit(true);   // explicit_num_ref_frames
  writer.put_bits(static_cast<std::uint64_t>(c.num_ref_frames) - 1U, 4U);
  writer.put_bits(c.long_term_frame_id_bits, 3U);
  put_ns(writer, 5U, c.max_drl_bits_minus_1);  // MAX_REF_MV_STACK_SIZE - 1
  writer.put_bit(false);   // allow_frame_max_drl_bits
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
  writer.put_bit(c.enable_avg_cdf);  // enable_avg_cdf
  if (c.enable_avg_cdf) {
    writer.put_bit(c.avg_cdf_type);  // avg_cdf_type: 1 = average across tiles
  }
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

  writer.put_bit(c.film_grain_params_present);  // §5.4.2 film grain models

  return writer.bit_size() - start;
}

bool read_sequence_header_bits(std::span<const std::uint8_t> payload,
                               SequenceHeaderView& out, std::string& error) {
  BitReader reader(payload);
  bool supported = true;
  std::uint32_t u = 0;

  if (!reader.read_uvlc(u) || u != 0) {
    error = "seq_header_id must be 0 for this profile";
    return false;
  }
  if (!read_bits_u32(reader, 5, u) || u != 0) {
    error = "seq_profile_idc must be 0";
    return false;
  }
  if (!expect_flag(reader, false, "single_picture_header_flag", supported)) {
    error = reader.error();
    return false;
  }
  std::uint32_t level = 0;
  if (!read_bits_u32(reader, 5, level)) {
    error = reader.error();
    return false;
  }
  out.seq_level_idx = static_cast<std::uint8_t>(level);
  if (out.seq_level_idx > 3) {
    if (!expect_flag(reader, false, "seq_tier", supported)) {
      error = reader.error();
      return false;
    }
  }
  if (!reader.read_uvlc(u) || u != 0) {  // chroma_format_idc
    error = "chroma_format_idc must be 0";
    return false;
  }
  if (!reader.read_uvlc(u) || u != 1) {  // bit_depth_idc
    error = "bit_depth_idc must be 1 (8-bit)";
    return false;
  }

  if (!read_bits_u32(reader, 3, u) || u != 0) {  // seq_lcr_id
    error = "seq_lcr_id must be 0";
    return false;
  }
  if (!expect_flag(reader, false, "still_picture", supported)) {
    error = reader.error();
    return false;
  }
  if (!read_bits_u32(reader, 2, u) || u != 0) {  // max_tlayer_id
    error = "max_tlayer_id must be 0";
    return false;
  }
  if (!read_bits_u32(reader, 3, u) || u != 0) {  // max_mlayer_id
    error = "max_mlayer_id must be 0";
    return false;
  }
  if (!read_flag(reader, out.monotonic_output_order_flag,
                  "monotonic_output_order_flag")) {
    error = reader.error();
    return false;
  }
  if (!out.monotonic_output_order_flag) supported = false;

  std::uint32_t width_bits_minus_1 = 0;
  std::uint32_t height_bits_minus_1 = 0;
  if (!read_bits_u32(reader, 4, width_bits_minus_1) ||
      !read_bits_u32(reader, 4, height_bits_minus_1)) {
    error = reader.error();
    return false;
  }
  const std::uint32_t width_bits = width_bits_minus_1 + 1U;
  const std::uint32_t height_bits = height_bits_minus_1 + 1U;
  std::uint32_t width_minus_1 = 0;
  std::uint32_t height_minus_1 = 0;
  if (!read_bits_u32(reader, width_bits, width_minus_1) ||
      !read_bits_u32(reader, height_bits, height_minus_1)) {
    error = reader.error();
    return false;
  }
  out.max_frame_width = static_cast<std::uint16_t>(width_minus_1 + 1U);
  out.max_frame_height = static_cast<std::uint16_t>(height_minus_1 + 1U);

  if (!expect_flag(reader, false, "seq_cropping_window_present_flag",
                   supported) ||
      !expect_flag(reader, false, "seq_initial_display_delay_present_flag",
                   supported) ||
      !expect_flag(reader, false, "decoder_model_info_present_flag",
                   supported)) {
    error = reader.error();
    return false;
  }

  // §5.4.3 partition config.
  if (!expect_flag(reader, false, "use_256x256_superblock", supported) ||
      !expect_flag(reader, false, "use_128x128_superblock", supported) ||
      !expect_flag(reader, false, "enable_sdp", supported) ||
      !expect_flag(reader, false, "enable_ext_partitions", supported) ||
      !expect_flag(reader, false, "reduce_pb_aspect_ratio", supported)) {
    error = reader.error();
    return false;
  }
  // §5.4.4 segment config.
  if (!expect_flag(reader, false, "enable_ext_seg", supported) ||
      !expect_flag(reader, false, "seq_seg_info_present_flag", supported)) {
    error = reader.error();
    return false;
  }
  // §5.4.5 intra config.
  if (!expect_flag(reader, false, "enable_dip", supported) ||
      !expect_flag(reader, true, "enable_intra_edge_filter", supported) ||
      !expect_flag(reader, false, "enable_mrls", supported) ||
      !read_flag(reader, out.enable_cfl_intra, "enable_cfl_intra")) {
    error = reader.error();
    return false;
  }
  if (!read_bits_u32(reader, 2, u) || u != 0) {
    error = "cfl_ds_filter_index must be 0";
    return false;
  }
  if (!expect_flag(reader, false, "enable_mhccp", supported) ||
      !expect_flag(reader, false, "enable_ibp", supported)) {
    error = reader.error();
    return false;
  }
  // §5.4.6 inter config.
  for (int mode = 1; mode < 5; ++mode) {
    if (!expect_flag(reader, false, "seq_enabled_motion_modes", supported)) {
      error = reader.error();
      return false;
    }
  }
  // motionModeEnabled == 0 -> seq_frame_motion_modes_present_flag is not
  // coded (§5.4.6).
  if (!expect_flag(reader, false, "enable_masked_compound", supported) ||
      !expect_flag(reader, false, "enable_ref_frame_mvs", supported)) {
    error = reader.error();
    return false;
  }
  std::uint32_t order_hint_bits_minus_1 = 0;
  if (!read_bits_u32(reader, 4, order_hint_bits_minus_1)) {
    error = reader.error();
    return false;
  }
  out.order_hint_bits = static_cast<std::uint8_t>(order_hint_bits_minus_1 + 1U);
  if (!expect_flag(reader, false, "enable_refmvbank", supported) ||
      !expect_flag(reader, true, "disable_drl_reorder", supported) ||
      !read_flag(reader, out.explicit_ref_frame_map,
                 "explicit_ref_frame_map") ||
      !read_flag(reader, out.explicit_num_ref_frames,
                 "explicit_num_ref_frames")) {
    error = reader.error();
    return false;
  }
  if (out.explicit_num_ref_frames) {
    std::uint32_t num_ref_frames_minus_1 = 0;
    if (!read_bits_u32(reader, 4, num_ref_frames_minus_1)) {
      error = reader.error();
      return false;
    }
    out.num_ref_frames = static_cast<std::uint8_t>(num_ref_frames_minus_1 + 1U);
  } else {
    out.num_ref_frames = 8;
  }
  std::uint32_t long_term_bits = 0;
  if (!read_bits_u32(reader, 3, long_term_bits)) {
    error = reader.error();
    return false;
  }
  out.long_term_frame_id_bits = static_cast<std::uint8_t>(long_term_bits);
  std::uint32_t drl = 0;
  if (!reader.read_ns(5U, drl)) {
    error = reader.error();
    return false;
  }
  out.max_drl_bits_minus_1 = static_cast<std::uint8_t>(drl);
  if (!expect_flag(reader, false, "allow_frame_max_drl_bits", supported)) {
    error = reader.error();
    return false;
  }
  std::uint32_t bvp = 0;
  if (!reader.read_ns(3U, bvp) || bvp != 0) {
    error = "seq_max_bvp_drl_bits_minus_1 must be 0";
    return false;
  }
  if (!expect_flag(reader, false, "allow_frame_max_bvp_drl_bits", supported) ||
      !read_bits_u32(reader, 2, u) || u != 0) {
    error = "num_same_ref_compound must be 0";
    return false;
  }
  if (!expect_flag(reader, false, "enable_tip", supported) ||
      !expect_flag(reader, false, "enable_mv_traj", supported) ||
      !expect_flag(reader, false, "enable_bawp", supported) ||
      !expect_flag(reader, false, "enable_cwp", supported) ||
      !expect_flag(reader, false, "enable_imp_msk_bld", supported) ||
      !expect_flag(reader, false, "enable_df_sub_pu", supported)) {
    error = reader.error();
    return false;
  }
  if (!read_bits_u32(reader, 2, u) || u != 0) {
    error = "enable_opfl_refine must be REFINE_NONE";
    return false;
  }
  if (!expect_flag(reader, false, "enable_refinemv", supported) ||
      !expect_flag(reader, false, "enable_bru", supported) ||
      !expect_flag(reader, false, "enable_adaptive_mvd", supported) ||
      !expect_flag(reader, false, "enable_mvd_sign_derive", supported) ||
      !expect_flag(reader, false, "enable_flex_mvres", supported) ||
      !expect_flag(reader, false, "enable_global_motion", supported) ||
      !expect_flag(reader, false, "enable_short_refresh_frame_flags",
                   supported)) {
    error = reader.error();
    return false;
  }
  // §5.4.7 SCC.
  if (!expect_flag(reader, false, "seq_choose_screen_content_tools",
                   supported) ||
      !expect_flag(reader, false, "seq_force_screen_content_tools",
                   supported)) {
    error = reader.error();
    return false;
  }
  out.seq_force_screen_content_tools = false;
  // §5.4.8 transform/quant/entropy.
  if (!expect_flag(reader, false, "enable_fsc", supported) ||
      !expect_flag(reader, false, "enable_idtx_intra", supported) ||
      !expect_flag(reader, false, "enable_intra_ist", supported) ||
      !expect_flag(reader, false, "enable_inter_ist", supported) ||
      !expect_flag(reader, true, "enable_chroma_dctonly", supported) ||
      !expect_flag(reader, false, "enable_inter_ddt", supported) ||
      !expect_flag(reader, false, "reduced_tx_part_set", supported) ||
      !expect_flag(reader, false, "enable_cctx", supported) ||
      !expect_flag(reader, false, "enable_tcq", supported) ||
      !expect_flag(reader, false, "enable_parity_hiding", supported)) {
    error = reader.error();
    return false;
  }
  // §5.4.8: enable_avg_cdf gates the avg_cdf_type bit.
  bool avg_cdf = false;
  if (!read_flag(reader, avg_cdf, "enable_avg_cdf")) {
    error = reader.error();
    return false;
  }
  out.enable_avg_cdf = avg_cdf;
  if (avg_cdf) {
    bool avg_cdf_type = false;
    if (!read_flag(reader, avg_cdf_type, "avg_cdf_type")) {
      error = reader.error();
      return false;
    }
    out.avg_cdf_type = avg_cdf_type;
  }
  if (!expect_flag(reader, false, "separate_uv_delta_q", supported) ||
      !expect_flag(reader, true, "equal_ac_dc_q", supported)) {
    error = reader.error();
    return false;
  }
  std::uint32_t base_uv_ac = 0;
  if (!read_bits_u32(reader, 5, base_uv_ac) || base_uv_ac != 23) {
    error = "base_uv_ac_delta_q must be 23 (BaseUVAcDeltaQ = 0)";
    return false;
  }
  if (!expect_flag(reader, false, "uv_ac_delta_q_enabled", supported)) {
    error = reader.error();
    return false;
  }
  // §5.4.10 filter config.
  if (!expect_flag(reader, false, "disable_loopfilters_across_tiles",
                   supported) ||
      !expect_flag(reader, false, "enable_cdef", supported) ||
      !expect_flag(reader, false, "enable_gdf", supported) ||
      !expect_flag(reader, false, "enable_restoration", supported) ||
      !expect_flag(reader, false, "enable_ccso", supported) ||
      !expect_flag(reader, false, "cdef_on_skip_txfm_always_on", supported) ||
      !expect_flag(reader, true, "cdef_on_skip_txfm_disabled", supported)) {
    error = reader.error();
    return false;
  }
  if (!read_bits_u32(reader, 2, u) || u != 0) {
    error = "df_par_bits_minus_2 must be 0";
    return false;
  }
  // §5.4.2 tile config, §5.4.1 film grain.
  if (!expect_flag(reader, false, "seq_tile_info_present_flag", supported) ||
      !read_flag(reader, out.film_grain_params_present,
                 "film_grain_params_present")) {
    error = reader.error();
    return false;
  }
  if (error.empty()) error = reader.error();
  out.supported_profile = supported;
  return true;
}

// ---------------------------------------------------------------------------
// Frame header syntax bits (§5.18.2 fixed profile)
// ---------------------------------------------------------------------------

std::uint32_t ceil_log2_u32(std::uint32_t value) {
  if (value <= 1U) return 0;
  std::uint32_t bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

void write_tile_info(BitWriter& writer, std::uint32_t width,
                     std::uint32_t height) {
  const std::uint32_t mi_cols = mi_cols_for(width);
  const std::uint32_t mi_rows = mi_rows_for(height);
  const std::uint32_t sb_cols = (mi_cols + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows + 15U) >> 4U;
  assert(sb_cols >= 1U && sb_rows >= 1U);
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) writer.put_bit(false);  // increment_tile_cols_log2 = 0
  if (sb_rows > 1U) writer.put_bit(false);  // increment_tile_rows_log2 = 0
}

bool read_tile_info(BitReader& reader, std::uint32_t width,
                    std::uint32_t height, std::string& error) {
  bool uniform = false;
  if (!reader.read_bit(uniform) || !uniform) {
    error = "only uniform_tile_spacing_flag = 1 is supported";
    return false;
  }
  const std::uint32_t mi_cols = mi_cols_for(width);
  const std::uint32_t mi_rows = mi_rows_for(height);
  const std::uint32_t sb_cols = (mi_cols + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows + 15U) >> 4U;
  bool bit = false;
  if (sb_cols > 1U && !reader.read_bit(bit)) {
    error = reader.error();
    return false;
  }
  if (sb_rows > 1U && !reader.read_bit(bit)) {
    error = reader.error();
    return false;
  }
  return true;
}

std::size_t write_frame_header_bits(BitWriter& writer,
                                    const SequenceConfig& c,
                                    const FrameHeader& h) {
  const std::size_t start = writer.bit_size();
  const FrameTypeInfo info =
      derive_frame_type(h.obu_type, h.restricted_prediction_switch);
  assert(c.base_q_idx >= 1U);
  assert(c.long_term_frame_id_bits <= 7U);

  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header

  // §5.18.2: frame_type derivation.
  if (info.codes_restricted_prediction_switch) {
    writer.put_bit(h.restricted_prediction_switch);
  } else if (info.codes_frame_is_inter) {
    writer.put_bit(true);  // frame_is_inter -> INTER_FRAME
  }

  if (info.frame_type == FrameType::kKey && c.long_term_frame_id_bits > 0) {
    assert(h.long_term_id_plus_1 < (1U << c.long_term_frame_id_bits));
    writer.put_bits(h.long_term_id_plus_1, c.long_term_frame_id_bits);
  }

  if (info.codes_num_key_ref_frames && c.long_term_frame_id_bits > 0) {
    assert(h.num_key_ref_frames <= kRefsPerFrame);
    writer.put_bits(h.num_key_ref_frames, 3U);
    for (std::size_t i = 0; i < h.num_key_ref_frames; ++i) {
      assert(h.ref_long_term_id[i] < (1U << c.long_term_frame_id_bits));
      writer.put_bits(h.ref_long_term_id[i], c.long_term_frame_id_bits);
    }
  }

  // All three supported frame types are neither bridge nor OLK.
  writer.put_bit(h.immediate_output_frame);
  // monotonic_output_order_flag = 1 -> implicit_output_frame is not coded.

  if (info.forces_frame_size_override) {
    // frame_size_override_flag = 1 is inferred for SWITCH_FRAME.
  } else {
    writer.put_bit(false);  // frame_size_override_flag = 0
  }

  writer.put_bits(h.order_hint, c.order_hint_bits);

  if (!info.skips_primary_ref) {
    // Not reached by the supported types, but keep the syntax complete.
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
  }

  if (info.frame_type == FrameType::kKey) {
    // OBU_CLOSED_LOOP_KEY + max_mlayer_id == 0 -> refresh_frame_flags =
    // allFrames; no bits.
  } else if (info.derives_refresh_frame_flags) {
    // Normatively RAS derives refresh_frame_flags (§5.18.2) and codes no bits.
    // The pinned dav2d build reads the generic SWITCH_FRAME field instead; see
    // SequenceConfig::ras_codes_refresh_frame_flags.
    if (c.ras_codes_refresh_frame_flags) {
      writer.put_bits(h.refresh_frame_flags, c.num_ref_frames);
    }
  } else {
    writer.put_bits(h.refresh_frame_flags, c.num_ref_frames);
  }

  if (info.frame_type == FrameType::kKey) {
    // frame_size(): override = 0 -> no bits.
    // screen_content_params(): seq_force_screen_content_tools = 0 -> no bits.
    writer.put_bit(false);  // allow_intrabc
  } else {
    // SWITCH/RAS force explicitRefFrameMap = 1 (§5.18.2).
    assert(h.num_total_refs <= kRefsPerFrame);
    writer.put_bits(h.num_total_refs, 3U);
    const unsigned ref_idx_bits = ceil_log2_u32(c.num_ref_frames);
    for (std::size_t i = 0; i < h.num_total_refs; ++i) {
      writer.put_bits(h.ref_frame_idx[i], ref_idx_bits);
    }

    // frame_size(): forced override = 1.
    const std::uint32_t width = h.frame_width != 0 ? h.frame_width
                                                   : c.max_frame_width;
    const std::uint32_t height = h.frame_height != 0 ? h.frame_height
                                                     : c.max_frame_height;
    writer.put_bits(width - 1U, c.frame_width_bits());
    writer.put_bits(height - 1U, c.frame_height_bits());

    // screen_content_params(): seq_force_screen_content_tools = 0 -> no bits.
    writer.put_bit(false);  // allow_intrabc

    // max_drl_bits_minus_1 = seq_max_drl_bits_minus_1 -> no bits.
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv -> 1/8 pel
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTTAP
    // frame_enabled_motion_modes[] all zero -> no bits.
  }

  writer.put_bit(c.disable_cdf_update);  // disable_cdf_update

  write_tile_info(writer, h.frame_width != 0 ? h.frame_width : c.max_frame_width,
                  h.frame_height != 0 ? h.frame_height : c.max_frame_height);

  writer.put_bits(c.base_q_idx, 8U);  // base_q_idx
  writer.put_bit(false);              // segmentation_enabled
  writer.put_bit(false);              // using_qmatrix
  writer.put_bit(false);              // delta_q_present (base_q_idx > 0)
  writer.put_bit(false);              // apply_deblocking_filter[0]
  writer.put_bit(false);              // apply_deblocking_filter[1]
  writer.put_bit(false);              // tx_mode_select = 0 -> TX_MODE_LARGEST

  if (info.frame_type != FrameType::kKey) {
    writer.put_bit(false);  // reference_select
    if (info.frame_type != FrameType::kSwitch) {
      writer.put_bit(false);  // skip_mode_present (regular inter only)
    }
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set = TX_SET_DCTONLY

  // §5.18.10.1 film_grain_config(). monotonic_output_order_flag = 1 here, so
  // implicit_output_frame is always 0.
  if (c.film_grain_params_present) {
    filters::grain::FilmGrainConfigContext context;
    context.film_grain_params_present = true;
    context.immediate_output_frame = h.immediate_output_frame;
    std::string grain_error;
    filters::grain::WriteFilmGrainConfig(writer, h.grain, context,
                                         &grain_error);
    assert(grain_error.empty());
  }

  return writer.bit_size() - start;
}

}  // namespace

// ---------------------------------------------------------------------------
// Small public helpers
// ---------------------------------------------------------------------------

std::uint32_t SequenceConfig::frame_width_bits() const {
  unsigned bits = bit_width(static_cast<std::uint32_t>(max_frame_width) - 1U);
  return bits == 0 ? 1U : bits;
}

std::uint32_t SequenceConfig::frame_height_bits() const {
  unsigned bits = bit_width(static_cast<std::uint32_t>(max_frame_height) - 1U);
  return bits == 0 ? 1U : bits;
}

std::uint32_t ceil_log2(std::uint32_t value) { return ceil_log2_u32(value); }

std::uint32_t frame_width_bits(const SequenceConfig& config) {
  return config.frame_width_bits();
}

std::uint32_t frame_height_bits(const SequenceConfig& config) {
  return config.frame_height_bits();
}

const char* frame_type_name(FrameType type) {
  switch (type) {
    case FrameType::kKey:
      return "KEY_FRAME";
    case FrameType::kInter:
      return "INTER_FRAME";
    case FrameType::kIntraOnly:
      return "INTRA_ONLY_FRAME";
    case FrameType::kSwitch:
      return "SWITCH_FRAME";
  }
  return "UNKNOWN";
}

const char* obu_type_name(ObuType type) {
  switch (type) {
    case ObuType::kSequenceHeader:
      return "OBU_SEQUENCE_HEADER";
    case ObuType::kTemporalDelimiter:
      return "OBU_TEMPORAL_DELIMITER";
    case ObuType::kClosedLoopKey:
      return "OBU_CLOSED_LOOP_KEY";
    case ObuType::kRegularTileGroup:
      return "OBU_REGULAR_TILE_GROUP";
    case ObuType::kSwitch:
      return "OBU_SWITCH";
    case ObuType::kRas:
      return "OBU_RAS_FRAME";
  }
  return "UNKNOWN";
}

FrameTypeInfo derive_frame_type(ObuType obu_type,
                                bool restricted_prediction_switch) {
  FrameTypeInfo info;
  info.obu_type = obu_type;
  info.key_frame = obu_type == ObuType::kClosedLoopKey;
  info.is_regular = obu_type == ObuType::kRegularTileGroup ||
                    obu_type == ObuType::kSwitch ||
                    obu_type == ObuType::kRas;
  if (obu_type == ObuType::kSwitch || obu_type == ObuType::kRas) {
    info.codes_restricted_prediction_switch = true;
    info.frame_type = FrameType::kSwitch;
    info.frame_is_intra = false;
    info.forces_explicit_ref_map = true;
    info.forces_frame_size_override = true;
    info.derives_refresh_frame_flags = obu_type == ObuType::kRas;
    info.skips_primary_ref = true;
    info.codes_num_key_ref_frames = obu_type == ObuType::kRas;
    info.resets_qm = obu_type == ObuType::kRas || restricted_prediction_switch;
  } else if (obu_type == ObuType::kClosedLoopKey) {
    info.frame_type = FrameType::kKey;
    info.frame_is_intra = true;
    info.skips_primary_ref = true;
  } else {
    // Regular tile group: frame_is_inter = 1 in this profile.
    info.frame_type = FrameType::kInter;
    info.frame_is_intra = false;
    info.codes_frame_is_inter = true;
    info.skips_primary_ref = false;
  }
  return info;
}

// ---------------------------------------------------------------------------
// Reference state / conformance
// ---------------------------------------------------------------------------

bool long_term_id_in_use(std::int32_t long_term_id,
                         std::span<const std::uint8_t> ref_long_term_ids) {
  if (long_term_id < 0) return false;
  for (const std::uint8_t id : ref_long_term_ids) {
    if (static_cast<std::int32_t>(id) == long_term_id) return true;
  }
  return false;
}

std::uint16_t allowed_reference_mask(
    ObuType obu_type, const RefState& refs,
    std::span<const std::uint8_t> ref_long_term_ids) {
  std::uint16_t mask = 0;
  for (std::size_t i = 0; i < kNumRefFrames; ++i) {
    if (!refs.slots[i].valid) continue;
    if (obu_type == ObuType::kRas) {
      if (!long_term_id_in_use(refs.slots[i].long_term_id,
                               ref_long_term_ids)) {
        continue;
      }
    }
    mask = static_cast<std::uint16_t>(mask | (1U << i));
  }
  if (obu_type == ObuType::kClosedLoopKey) return 0;
  return mask;
}

std::uint16_t derive_ras_refresh_flags(
    const RefState& refs, std::span<const std::uint8_t> ref_long_term_ids,
    std::uint8_t num_ref_frames) {
  std::uint16_t flags = 0;
  const std::uint8_t limit =
      std::min<std::uint8_t>(num_ref_frames, kNumRefFrames);
  for (std::size_t i = 0; i < limit; ++i) {
    if (!refs.slots[i].valid ||
        !long_term_id_in_use(refs.slots[i].long_term_id,
                             ref_long_term_ids)) {
      flags = static_cast<std::uint16_t>(flags | (1U << i));
    }
  }
  return flags;
}

void update_reference_state(const SequenceConfig& config,
                            const FrameHeader& header,
                            std::uint16_t actual_refresh_flags,
                            RefState& refs) {
  const FrameTypeInfo info =
      derive_frame_type(header.obu_type, header.restricted_prediction_switch);
  // §5.18.2: a restricted SWITCH marks every available reference restricted
  // before the reference-frame update process stores the new frame.
  if (info.frame_type == FrameType::kSwitch &&
      header.restricted_prediction_switch) {
    for (std::size_t i = 0; i < kNumRefFrames; ++i) {
      if (refs.slots[i].valid) refs.slots[i].restricted = true;
    }
  }

  std::uint16_t refresh = actual_refresh_flags;
  if (info.frame_type == FrameType::kKey) {
    refresh = static_cast<std::uint16_t>((1U << config.num_ref_frames) - 1U);
  } else if (info.derives_refresh_frame_flags) {
    // Caller passes derive_ras_refresh_flags().
  } else {
    refresh = header.refresh_frame_flags;
  }

  const std::uint16_t limit =
      static_cast<std::uint16_t>((1U << config.num_ref_frames) - 1U);
  refresh = static_cast<std::uint16_t>(refresh & limit);
  const std::uint32_t width =
      header.frame_width != 0 ? header.frame_width : config.max_frame_width;
  const std::uint32_t height =
      header.frame_height != 0 ? header.frame_height : config.max_frame_height;
  const std::int32_t long_term_id =
      info.frame_type == FrameType::kKey
          ? static_cast<std::int32_t>(header.long_term_id_plus_1) - 1
          : -1;

  for (std::size_t i = 0; i < kNumRefFrames; ++i) {
    if ((refresh & (1U << i)) == 0) continue;
    refs.slots[i].valid = true;
    refs.slots[i].long_term_id = long_term_id;
    refs.slots[i].width = static_cast<std::uint16_t>(width);
    refs.slots[i].height = static_cast<std::uint16_t>(height);
    refs.slots[i].restricted = false;
  }
}

ValidationResult validate_header(const SequenceConfig& config,
                                 const FrameHeader& header, const RefState& refs,
                                 std::uint8_t tlayer_id) {
  ValidationResult result;
  const auto add = [&result](ViolationCode code, std::string message) {
    result.ok = false;
    result.violations.push_back(Violation{code, std::move(message)});
  };

  const FrameTypeInfo info =
      derive_frame_type(header.obu_type, header.restricted_prediction_switch);
  if (header.obu_type != ObuType::kClosedLoopKey &&
      header.obu_type != ObuType::kSwitch &&
      header.obu_type != ObuType::kRas) {
    add(ViolationCode::kUnsupportedObuType,
        "only CLK, SWITCH and RAS headers are supported");
    return result;
  }
  if ((header.obu_type == ObuType::kSwitch ||
       header.obu_type == ObuType::kRas) &&
      tlayer_id != 0) {
    add(ViolationCode::kTlayerNonZero,
        "Table 6.1 requires obu_tlayer_id == 0 for SWITCH/RAS");
  }
  if (header.obu_type == ObuType::kClosedLoopKey && tlayer_id != 0) {
    add(ViolationCode::kTlayerNonZero,
        "Table 6.1 requires obu_tlayer_id == 0 for CLK");
  }
  if (config.long_term_frame_id_bits > 7) {
    add(ViolationCode::kBadLongTermIdBits,
        "long_term_frame_id_bits does not fit the 3-bit field");
  }

  if (header.obu_type == ObuType::kClosedLoopKey) {
    if (config.long_term_frame_id_bits == 0) {
      if (header.long_term_id_plus_1 != 0) {
        add(ViolationCode::kReservedFieldSet,
            "long_term_id_plus_1 is not present when "
            "long_term_frame_id_bits == 0");
      }
    } else if (header.long_term_id_plus_1 >=
               (1U << config.long_term_frame_id_bits)) {
      add(ViolationCode::kKeyLongTermIdReserved,
          "long_term_id_plus_1 exceeds the field width");
    }
    return result;
  }

  if (!info.codes_restricted_prediction_switch &&
      header.restricted_prediction_switch) {
    add(ViolationCode::kSwitchRestrictedOnly,
        "restricted_prediction_switch is only present for SWITCH/RAS");
  }

  const std::uint32_t width =
      header.frame_width != 0 ? header.frame_width : config.max_frame_width;
  const std::uint32_t height =
      header.frame_height != 0 ? header.frame_height : config.max_frame_height;
  const std::uint32_t order_hint_limit =
      config.order_hint_bits >= 31 ? 0x7FFFFFFFU
                                   : (1U << config.order_hint_bits);

  if (header.obu_type == ObuType::kRas) {
    if (header.num_key_ref_frames > kRefsPerFrame) {
      add(ViolationCode::kTooManyKeyRefFrames,
          "num_key_ref_frames exceeds REFS_PER_FRAME");
    }
    if (config.long_term_frame_id_bits == 0) {
      if (header.num_key_ref_frames != 0) {
        add(ViolationCode::kRasWithoutLongTermIds,
            "num_key_ref_frames must be 0 when "
            "long_term_frame_id_bits == 0");
      }
      if (header.num_total_refs != 0) {
        add(ViolationCode::kRasWithoutLongTermIds,
            "a RAS without long-term ids cannot reference any frame; "
            "num_total_refs must be 0");
      }
    } else {
      if (header.num_total_refs > 0 && header.num_key_ref_frames == 0) {
        add(ViolationCode::kRasWithoutLongTermIds,
            "a referencing RAS must name at least one ref_long_term_id");
      }
      const std::uint8_t reserved_id = static_cast<std::uint8_t>(
          (1U << config.long_term_frame_id_bits) - 1U);
      for (std::size_t i = 0; i < header.num_key_ref_frames; ++i) {
        if (header.ref_long_term_id[i] == reserved_id) {
          add(ViolationCode::kBadRefLongTermId,
              "ref_long_term_id must not be (1 << long_term_frame_id_bits)-1");
        }
        bool present = false;
        for (std::size_t slot = 0; slot < kNumRefFrames; ++slot) {
          if (refs.slots[slot].valid &&
              refs.slots[slot].long_term_id ==
                  static_cast<std::int32_t>(header.ref_long_term_id[i])) {
            present = true;
            break;
          }
        }
        if (!present) {
          add(ViolationCode::kRasRefNotLongTerm,
              "a named long-term reference is not held in any reference slot "
              "(§7.4.5)");
        }
      }
      if (!header.restricted_prediction_switch &&
          header.order_hint >= order_hint_limit) {
        add(ViolationCode::kRasOrderHintOutOfRange,
            "OrderHint of an unrestricted RAS must be < (1 << "
            "OrderHintBits) when long_term_frame_id_bits > 0");
      }
    }
  }

  if (header.num_total_refs > kRefsPerFrame) {
    add(ViolationCode::kBadNumTotalRefs,
        "num_total_refs exceeds REFS_PER_FRAME");
  }
  const std::uint32_t ref_idx_limit = config.num_ref_frames;
  const std::uint16_t allowed =
      allowed_reference_mask(header.obu_type, refs,
                             std::span<const std::uint8_t>(
                                 header.ref_long_term_id.data(),
                                 header.num_key_ref_frames));
  for (std::size_t i = 0; i < header.num_total_refs; ++i) {
    const std::uint8_t slot = header.ref_frame_idx[i];
    if (slot >= ref_idx_limit) {
      add(ViolationCode::kBadRefFrameIdx,
          "ref_frame_idx[" + std::to_string(i) + "] = " +
              std::to_string(slot) + " exceeds NumRefFrames-1");
      continue;
    }
    if (!refs.slots[slot].valid) {
      add(ViolationCode::kInvalidReference,
          "ref_frame_idx[" + std::to_string(i) +
              "] names an invalid reference slot");
      continue;
    }
    if (header.obu_type == ObuType::kRas && (allowed & (1U << slot)) == 0) {
      add(ViolationCode::kRasRefNotLongTerm,
          "RAS reference slot " + std::to_string(slot) +
              " does not hold a ref_long_term_id from this frame header");
    }
    const RefSlot& ref = refs.slots[slot];
    if (2U * width < ref.width || 2U * height < ref.height ||
        width > 16U * ref.width || height > 16U * ref.height) {
      add(ViolationCode::kReferenceDimensions,
          "frame dimensions are outside the §6.17.4.3 1:2..16:1 reference "
          "ratio");
    }
  }

  if (header.obu_type == ObuType::kSwitch) {
    const std::uint16_t refresh_limit = static_cast<std::uint16_t>(
        (1U << config.num_ref_frames) - 1U);
    if ((header.refresh_frame_flags & ~refresh_limit) != 0) {
      add(ViolationCode::kReservedFieldSet,
          "refresh_frame_flags sets bits outside NumRefFrames");
    }
  }

  if (header.obu_type == ObuType::kRas &&
      header.refresh_frame_flags != 0) {
    const std::uint16_t derived = derive_ras_refresh_flags(
        refs, std::span<const std::uint8_t>(header.ref_long_term_id.data(),
                                           header.num_key_ref_frames),
        config.num_ref_frames);
    if (header.refresh_frame_flags != derived) {
      add(ViolationCode::kRasRefreshMismatch,
          "RAS refresh_frame_flags differs from the §5.18.2 derived mask");
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Encoder-side strategy
// ---------------------------------------------------------------------------

PayloadPlan plan_switch_frame(const SequenceConfig& config, const RefState& refs,
                              const PFrameCandidate& candidate,
                              bool restricted_prediction_switch) {
  PayloadPlan plan;
  plan.header.obu_type = ObuType::kSwitch;
  plan.header.order_hint = candidate.order_hint;
  plan.header.restricted_prediction_switch = restricted_prediction_switch;
  plan.header.num_total_refs = 1;
  plan.header.ref_frame_idx[0] = candidate.reference_slot;
  plan.header.refresh_frame_flags = candidate.refresh_frame_flags;
  plan.header.immediate_output_frame = true;

  const std::uint16_t allowed =
      allowed_reference_mask(ObuType::kSwitch, refs, {});
  if (candidate.reference_slot >= config.num_ref_frames ||
      (allowed & (1U << candidate.reference_slot)) == 0) {
    plan.decision = PayloadDecision::kReencodeRequired;
    plan.reason =
        "SWITCH payload references an invalid slot; re-encode the P frame";
    return plan;
  }
  const ValidationResult validation =
      validate_header(config, plan.header, refs);
  if (!validation.ok) {
    plan.decision = PayloadDecision::kReencodeRequired;
    plan.reason = validation.violations.front().message;
    return plan;
  }
  plan.decision = PayloadDecision::kReusePFrame;
  plan.reason = "P payload references a valid slot; only the header and OBU "
                "type change";
  return plan;
}

PayloadPlan plan_ras_frame(const SequenceConfig& config, const RefState& refs,
                           const PFrameCandidate& candidate,
                           std::span<const std::uint8_t> ras_long_term_ids) {
  PayloadPlan plan;
  plan.header.obu_type = ObuType::kRas;
  plan.header.order_hint = candidate.order_hint;
  plan.header.restricted_prediction_switch = false;
  plan.header.immediate_output_frame = true;
  plan.header.num_key_ref_frames =
      static_cast<std::uint8_t>(ras_long_term_ids.size());
  for (std::size_t i = 0; i < ras_long_term_ids.size(); ++i) {
    plan.header.ref_long_term_id[i] = ras_long_term_ids[i];
  }

  const bool slot_valid =
      candidate.reference_slot < config.num_ref_frames &&
      refs.slots[candidate.reference_slot].valid;
  const bool slot_is_long_term =
      slot_valid &&
      long_term_id_in_use(refs.slots[candidate.reference_slot].long_term_id,
                          ras_long_term_ids);

  // A RAS is an inter-predicted frame whose reference list may only contain
  // long-term frames named in ref_long_term_id (§6.17, §7.4.5). With
  // long_term_frame_id_bits == 0 the list is inferred empty, so no reference
  // can be declared; the pinned dav2d build also rejects n_ref_frames == 0.
  // The reusable path therefore always needs a named long-term slot.
  if (config.long_term_frame_id_bits == 0) {
    plan.decision = PayloadDecision::kReencodeRequired;
    plan.reason =
        "RAS requires long_term_frame_id_bits > 0 and at least one long-term "
        "reference slot; set it in the sequence header";
    return plan;
  }
  if (!slot_is_long_term) {
    plan.decision = PayloadDecision::kReencodeRequired;
    plan.reason =
        "the P payload does not predict from a long-term frame named in "
        "ref_long_term_id; re-encode against a long-term slot";
    return plan;
  }
  plan.header.num_total_refs = 1;
  plan.header.ref_frame_idx[0] = candidate.reference_slot;
  if (!candidate.has_inter_blocks) {
    plan.reason =
        "all-intra payload reused with a declared long-term reference "
        "(num_total_refs = 1)";
  } else {
    plan.reason =
        "P payload references a named long-term slot; only the header and OBU "
        "type change";
  }

  plan.header.refresh_frame_flags = derive_ras_refresh_flags(
      refs, std::span<const std::uint8_t>(plan.header.ref_long_term_id.data(),
                                         plan.header.num_key_ref_frames),
      config.num_ref_frames);
  const ValidationResult validation =
      validate_header(config, plan.header, refs);
  if (!validation.ok) {
    plan.decision = PayloadDecision::kReencodeRequired;
    plan.reason = validation.violations.front().message;
    return plan;
  }
  plan.decision = PayloadDecision::kReusePFrame;
  return plan;
}

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

std::size_t write_sequence_header(BitWriter& writer,
                                  const SequenceConfig& config) {
  return write_sequence_header_bits(writer, config);
}

std::vector<std::uint8_t> make_sequence_header_obu(
    const SequenceConfig& config) {
  BitWriter writer;
  write_sequence_header_bits(writer, config);
  writer.put_bit(false);  // obu_extension_flag (§5.2.1)
  put_trailing_bits(writer);
  return make_obu(ObuType::kSequenceHeader, writer.take_bytes());
}

std::vector<std::uint8_t> make_temporal_delimiter_obu() {
  // Reuse the M0 empty temporal-delimiter OBU (obu_type = 2, no payload).
  return av2enc::basic::make_temporal_delimiter_obu();
}

std::vector<std::uint8_t> make_frame_obu(
    const SequenceConfig& config, const FrameHeader& header,
    std::span<const std::uint8_t> tile_payload) {
  BitWriter writer;
  write_tile_group_header(writer, config, header);
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  return make_obu(header.obu_type, payload);
}

std::size_t write_frame_header_info(BitWriter& writer,
                                    const SequenceConfig& config,
                                    const FrameHeader& header) {
  return write_frame_header_bits(writer, config, header);
}

std::size_t write_tile_group_header(BitWriter& writer,
                                    const SequenceConfig& config,
                                    const FrameHeader& header) {
  const std::size_t start = writer.bit_size();
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_bits(writer, config, header);
  // tile_start_and_end_present_flag is absent (NumTiles == 1).
  writer.align_to_byte();
  return writer.bit_size() - start;
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

bool parse_obu(std::span<const std::uint8_t> bytes, ObuView& out,
               std::size_t& consumed, std::string& error) {
  consumed = 0;
  std::size_t offset = 0;
  std::uint32_t size = 0;
  if (!read_leb128(bytes, offset, size, error)) return false;
  if (size == 0) {
    error = "OBU size is zero";
    return false;
  }
  const std::size_t header_offset = offset;
  if (offset >= bytes.size()) {
    error = "truncated OBU header";
    return false;
  }
  const std::uint8_t first = bytes[offset++];
  const bool extension = (first & 0x80U) != 0U;
  out.obu_type = static_cast<std::uint8_t>((first >> 2U) & 0x1FU);
  out.tlayer_id = static_cast<std::uint8_t>(first & 0x03U);
  std::size_t header_bytes = 1;
  if (extension) {
    if (offset >= bytes.size()) {
      error = "truncated OBU extension byte";
      return false;
    }
    const std::uint8_t second = bytes[offset++];
    out.mlayer_id = static_cast<std::uint8_t>((second >> 5U) & 0x07U);
    out.xlayer_id = static_cast<std::uint8_t>(second & 0x1FU);
    header_bytes = 2;
  } else {
    out.mlayer_id = 0;
    out.xlayer_id = 0;
  }
  if (size < header_bytes) {
    error = "OBU size is smaller than its header";
    return false;
  }
  if (header_offset + size > bytes.size()) {
    error = "OBU size exceeds the available bytes";
    return false;
  }
  out.payload = bytes.subspan(header_offset + header_bytes,
                              size - header_bytes);
  consumed = header_offset + size;
  return true;
}

bool read_sequence_header(std::span<const std::uint8_t> payload,
                          SequenceHeaderView& out, std::string& error) {
  out = SequenceHeaderView{};
  return read_sequence_header_bits(payload, out, error);
}

bool read_frame_header(std::span<const std::uint8_t> payload, ObuType obu_type,
                       const SequenceHeaderView& sequence, FrameHeader& out,
                       std::size_t& bits_consumed, std::string& error,
                       bool ras_codes_refresh_frame_flags) {
  out = FrameHeader{};
  out.obu_type = obu_type;
  bits_consumed = 0;
  if (obu_type != ObuType::kClosedLoopKey && obu_type != ObuType::kSwitch &&
      obu_type != ObuType::kRas) {
    error = "read_frame_header supports CLK, SWITCH and RAS only";
    return false;
  }

  BitReader reader(payload);
  bool first_tile = false;
  if (!reader.read_bit(first_tile)) {
    error = reader.error();
    return false;
  }
  if (!first_tile) {
    error = "expected is_first_tile_group = 1";
    return false;
  }
  std::uint32_t u = 0;
  if (!reader.read_uvlc(u) || u != 0) {
    error = "cur_mfh_id must be 0";
    return false;
  }
  if (!reader.read_uvlc(u) || u != 0) {
    error = "seq_header_id_in_frame_header must be 0";
    return false;
  }

  const bool switch_or_ras =
      obu_type == ObuType::kSwitch || obu_type == ObuType::kRas;
  if (switch_or_ras) {
    if (!read_flag(reader, out.restricted_prediction_switch,
                   "restricted_prediction_switch")) {
      error = reader.error();
      return false;
    }
  }
  if (obu_type == ObuType::kClosedLoopKey &&
      sequence.long_term_frame_id_bits > 0) {
    std::uint32_t value = 0;
    if (!read_bits_u32(reader, sequence.long_term_frame_id_bits, value)) {
      error = reader.error();
      return false;
    }
    out.long_term_id_plus_1 = static_cast<std::uint8_t>(value);
  }
  if (obu_type == ObuType::kRas &&
      sequence.long_term_frame_id_bits > 0) {
    std::uint32_t count = 0;
    if (!read_bits_u32(reader, 3, count)) {
      error = reader.error();
      return false;
    }
    out.num_key_ref_frames = static_cast<std::uint8_t>(count);
    for (std::size_t i = 0; i < out.num_key_ref_frames; ++i) {
      std::uint32_t id = 0;
      if (!read_bits_u32(reader, sequence.long_term_frame_id_bits, id)) {
        error = reader.error();
        return false;
      }
      out.ref_long_term_id[i] = static_cast<std::uint8_t>(id);
    }
  }

  if (!read_flag(reader, out.immediate_output_frame,
                 "immediate_output_frame")) {
    error = reader.error();
    return false;
  }
  if (!out.immediate_output_frame &&
      !sequence.monotonic_output_order_flag) {
    bool implicit = false;
    if (!read_flag(reader, implicit, "implicit_output_frame")) {
      error = reader.error();
      return false;
    }
  }

  if (!switch_or_ras) {
    bool override_flag = false;
    if (!read_flag(reader, override_flag, "frame_size_override_flag")) {
      error = reader.error();
      return false;
    }
    if (override_flag) {
      error = "key frame_size_override_flag = 1 is not supported";
      return false;
    }
  }

  if (!read_bits_u32(reader, sequence.order_hint_bits, out.order_hint)) {
    error = reader.error();
    return false;
  }

  if (!switch_or_ras) {
    bool signal_primary = false;
    bool disable_cross = false;
    if (!read_flag(reader, signal_primary, "signal_primary_ref_frame") ||
        !read_flag(reader, disable_cross,
                   "disable_cross_frame_cdf_init")) {
      error = reader.error();
      return false;
    }
    if (signal_primary) {
      std::uint32_t primary = 0;
      if (!read_bits_u32(reader, 3, primary)) {
        error = reader.error();
        return false;
      }
    }
  }

  if (obu_type == ObuType::kClosedLoopKey) {
    // refresh_frame_flags = allFrames, no bits.
    out.frame_width = sequence.max_frame_width;
    out.frame_height = sequence.max_frame_height;
  } else if (obu_type == ObuType::kSwitch) {
    std::uint32_t flags = 0;
    if (!read_bits_u32(reader, sequence.num_ref_frames, flags)) {
      error = reader.error();
      return false;
    }
    out.refresh_frame_flags = static_cast<std::uint16_t>(flags);
  } else {
    // Normatively RAS refresh_frame_flags is derived. The pinned dav2d build
    // reads explicit bits (see SequenceConfig::ras_codes_refresh_frame_flags).
    if (ras_codes_refresh_frame_flags) {
      std::uint32_t flags = 0;
      if (!read_bits_u32(reader, sequence.num_ref_frames, flags)) {
        error = reader.error();
        return false;
      }
      out.refresh_frame_flags = static_cast<std::uint16_t>(flags);
    } else {
      out.refresh_frame_flags = 0;  // derive with derive_ras_refresh_flags()
    }
  }

  if (obu_type == ObuType::kClosedLoopKey) {
    bool allow_intrabc = false;
    if (!read_flag(reader, allow_intrabc, "allow_intrabc")) {
      error = reader.error();
      return false;
    }
  } else {
    std::uint32_t refs = 0;
    if (!read_bits_u32(reader, 3, refs)) {
      error = reader.error();
      return false;
    }
    out.num_total_refs = static_cast<std::uint8_t>(refs);
    const std::uint32_t idx_bits = ceil_log2_u32(sequence.num_ref_frames);
    for (std::size_t i = 0; i < out.num_total_refs; ++i) {
      std::uint32_t idx = 0;
      if (!read_bits_u32(reader, idx_bits, idx)) {
        error = reader.error();
        return false;
      }
      out.ref_frame_idx[i] = static_cast<std::uint8_t>(idx);
    }
    std::uint32_t width_minus_1 = 0;
    std::uint32_t height_minus_1 = 0;
    const std::uint32_t width_bits = bit_width(
        static_cast<std::uint32_t>(sequence.max_frame_width) - 1U);
    const std::uint32_t height_bits = bit_width(
        static_cast<std::uint32_t>(sequence.max_frame_height) - 1U);
    if (!read_bits_u32(reader, width_bits == 0 ? 1 : width_bits,
                       width_minus_1) ||
        !read_bits_u32(reader, height_bits == 0 ? 1 : height_bits,
                       height_minus_1)) {
      error = reader.error();
      return false;
    }
    out.frame_width = static_cast<std::uint16_t>(width_minus_1 + 1U);
    out.frame_height = static_cast<std::uint16_t>(height_minus_1 + 1U);
    bool allow_intrabc = false;
    if (!read_flag(reader, allow_intrabc, "allow_intrabc")) {
      error = reader.error();
      return false;
    }
    bool use_qtr = false;
    bool high_prec = false;
    bool filter_switchable = false;
    std::uint32_t filter = 0;
    if (!read_flag(reader, use_qtr, "use_qtr_precision_mv") ||
        !read_flag(reader, high_prec, "allow_high_precision_mv") ||
        !read_flag(reader, filter_switchable, "is_filter_switchable")) {
      error = reader.error();
      return false;
    }
    if (!filter_switchable && !read_bits_u32(reader, 2, filter)) {
      error = reader.error();
      return false;
    }
  }

  bool disable_cdf_update = false;
  if (!read_flag(reader, disable_cdf_update, "disable_cdf_update")) {
    error = reader.error();
    return false;
  }
  if (!read_tile_info(reader, out.frame_width, out.frame_height, error)) {
    return false;
  }
  std::uint32_t base_q_idx = 0;
  if (!read_bits_u32(reader, 8, base_q_idx)) {
    error = reader.error();
    return false;
  }
  bool segmentation = false;
  bool using_qmatrix = false;
  bool delta_q_present = false;
  bool dbf0 = false;
  bool dbf1 = false;
  bool tx_mode_select = false;
  if (!read_flag(reader, segmentation, "segmentation_enabled") ||
      !read_flag(reader, using_qmatrix, "using_qmatrix") ||
      !read_flag(reader, delta_q_present, "delta_q_present") ||
      !read_flag(reader, dbf0, "apply_deblocking_filter[0]") ||
      !read_flag(reader, dbf1, "apply_deblocking_filter[1]") ||
      !read_flag(reader, tx_mode_select, "tx_mode_select")) {
    error = reader.error();
    return false;
  }
  if (segmentation || using_qmatrix || delta_q_present || dbf0 || dbf1) {
    error = "unsupported optional frame-header field in this profile";
    return false;
  }
  if (obu_type != ObuType::kClosedLoopKey) {
    bool reference_select = false;
    if (!read_flag(reader, reference_select, "reference_select")) {
      error = reader.error();
      return false;
    }
    // skip_mode_params(): FrameIsIntra || FrameType == SWITCH_FRAME -> no bit.
    // Both OBU_SWITCH and OBU_RAS have FrameType = SWITCH_FRAME.
    if (obu_type != ObuType::kSwitch && obu_type != ObuType::kRas) {
      bool skip_mode_present = false;
      if (!read_flag(reader, skip_mode_present, "skip_mode_present")) {
        error = reader.error();
        return false;
      }
    }
  }
  std::uint32_t reduced_tx_set = 0;
  if (!read_bits_u32(reader, 2, reduced_tx_set)) {
    error = reader.error();
    return false;
  }
  // §5.18.10.1 film_grain_config(). The reader has no single_picture_header
  // form (this profile always codes 0) and monotonic_output_order_flag = 1, so
  // implicit_output_frame stays 0.
  if (sequence.film_grain_params_present) {
    filters::grain::FilmGrainConfigContext context;
    context.film_grain_params_present = true;
    context.immediate_output_frame = out.immediate_output_frame;
    std::size_t position = reader.bits_read();
    std::string grain_error;
    if (!filters::grain::ReadFilmGrainConfig(payload, &position, context,
                                             &out.grain, &grain_error)) {
      error = "film_grain_config: " + grain_error;
      return false;
    }
    while (reader.bits_read() < position) {
      bool bit = false;
      if (!reader.read_bit(bit)) {
        error = reader.error();
        return false;
      }
    }
  }
  if (error.empty()) error = reader.error();
  if (!error.empty()) return false;
  bits_consumed = reader.bits_read();
  return true;
}

}  // namespace av2enc::inter::random_access
