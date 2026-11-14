#include "codec/inter/leading/leading_reader.h"

#include <cassert>
#include <limits>

namespace av2enc::inter::leading {
namespace {

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  bool read_bit(bool& value) {
    if (bit_pos_ >= bytes_.size() * 8U) return false;
    const std::size_t byte_index = bit_pos_ / 8U;
    const unsigned bit = static_cast<unsigned>(bit_pos_ & 7U);
    value = ((bytes_[byte_index] >> (7U - bit)) & 1U) != 0U;
    ++bit_pos_;
    return true;
  }

  bool read_bits(unsigned num_bits, std::uint64_t& value) {
    if (num_bits > 64U) return false;
    value = 0;
    for (unsigned i = 0; i < num_bits; ++i) {
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
      if (leading_zeros >= 32U) return false;
    }
    std::uint64_t payload = 0;
    if (!read_bits(leading_zeros, payload)) return false;
    const std::uint64_t base =
        (leading_zeros == 0U) ? 0ULL : ((1ULL << leading_zeros) - 1ULL);
    const std::uint64_t result = base + payload;
    if (result > std::numeric_limits<std::uint32_t>::max()) return false;
    value = static_cast<std::uint32_t>(result);
    return true;
  }

  bool read_ns(std::uint32_t n, std::uint32_t& value) {
    if (n < 2U) return false;
    unsigned w = 0;
    while ((1U << w) < n) ++w;
    const std::uint32_t m = (1U << w) - n;
    std::uint64_t v = 0;
    if (!read_bits(w - 1U, v)) return false;
    if (v < m) {
      value = static_cast<std::uint32_t>(v);
      return true;
    }
    bool extra = false;
    if (!read_bit(extra)) return false;
    value = static_cast<std::uint32_t>((v << 1U) - m + (extra ? 1U : 0U));
    return true;
  }

  bool align_to_byte(std::string& error, const char* what) {
    while ((bit_pos_ & 7U) != 0U) {
      bool bit = false;
      if (!read_bit(bit)) {
        error = std::string("truncated byte_alignment in ") + what;
        return false;
      }
      if (bit) {
        error = std::string("non-zero byte_alignment bit in ") + what;
        return false;
      }
    }
    return true;
  }

  std::size_t bit_pos() const { return bit_pos_; }
  std::size_t bit_size() const { return bytes_.size() * 8U; }
  std::size_t bytes_consumed() const { return bit_pos_ / 8U; }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_pos_ = 0;
};

bool read_expect_bit(BitReader& reader, bool expected, const char* what,
                     std::string& error) {
  bool bit = false;
  if (!reader.read_bit(bit)) {
    error = std::string("truncated ") + what;
    return false;
  }
  if (bit != expected) {
    error = std::string("unexpected ") + what;
    return false;
  }
  return true;
}

bool read_expect_bits(BitReader& reader, unsigned num_bits,
                      std::uint64_t expected, const char* what,
                      std::string& error) {
  std::uint64_t value = 0;
  if (!reader.read_bits(num_bits, value)) {
    error = std::string("truncated ") + what;
    return false;
  }
  if (value != expected) {
    error = std::string("unexpected ") + what;
    return false;
  }
  return true;
}

FrameKind kind_from_obu(ObuType type) {
  switch (type) {
    case ObuType::kClosedLoopKey:
      return FrameKind::kClosedLoopKey;
    case ObuType::kOpenLoopKey:
      return FrameKind::kOpenLoopKey;
    case ObuType::kLeadingTileGroup:
      return FrameKind::kLeadingTileGroup;
    case ObuType::kRegularTileGroup:
      return FrameKind::kRegularTileGroup;
    default:
      return FrameKind::kRegularTileGroup;
  }
}

std::uint32_t mi_cols(const SequenceConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
}
std::uint32_t mi_rows(const SequenceConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
}

}  // namespace

bool parse_obu(std::span<const std::uint8_t> bytes, ParsedObu& obu,
               std::string& error) {
  error.clear();
  obu = ParsedObu{};
  if (bytes.empty()) {
    error = "empty OBU";
    return false;
  }

  std::size_t pos = 0;
  std::uint32_t size = 0;
  unsigned shift = 0;
  bool terminated = false;
  while (pos < bytes.size() && shift <= 28U) {
    const std::uint8_t byte = bytes[pos++];
    size |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
    shift += 7U;
    if ((byte & 0x80U) == 0U) {
      terminated = true;
      break;
    }
  }
  if (!terminated) {
    error = "truncated or oversized leb128 OBU size";
    return false;
  }
  if (size == 0U) {
    error = "zero-length OBU";
    return false;
  }
  if (pos + size > bytes.size()) {
    error = "OBU size exceeds the remaining bytes";
    return false;
  }

  const std::uint8_t header_byte = bytes[pos];
  obu.header_extension = (header_byte & 0x80U) != 0U;
  obu.type = static_cast<ObuType>((header_byte >> 2U) & 0x1FU);
  obu.tlayer_id = static_cast<std::uint8_t>(header_byte & 0x03U);
  std::size_t header_bytes = 1;
  if (obu.header_extension) {
    if (size < 2U) {
      error = "OBU too short for header extension";
      return false;
    }
    const std::uint8_t ext = bytes[pos + 1U];
    obu.mlayer_id = static_cast<std::uint8_t>((ext >> 5U) & 0x07U);
    obu.xlayer_id = static_cast<std::uint8_t>(ext & 0x1FU);
    header_bytes = 2;
  }
  if (size < header_bytes) {
    error = "OBU payload size is smaller than its header";
    return false;
  }

  // `pos` is the first header byte; the leb128 prefix starts at offset 0 of the
  // slice passed to this function.
  obu.offset = 0;
  obu.total_size = pos + size;
  obu.header_bytes = header_bytes;
  obu.payload_offset = pos + header_bytes;
  obu.payload_size = size - header_bytes;
  obu.payload = bytes.subspan(obu.payload_offset, obu.payload_size);
  return true;
}

bool parse_obu_sequence(std::span<const std::uint8_t> bytes,
                        std::vector<ParsedObu>& obus, std::string& error) {
  error.clear();
  obus.clear();
  std::size_t pos = 0;
  while (pos < bytes.size()) {
    ParsedObu obu;
    if (!parse_obu(bytes.subspan(pos), obu, error)) return false;
    // Rebase absolute offsets onto the full input.
    obu.offset += pos;
    obu.payload_offset += pos;
    obu.payload = bytes.subspan(obu.payload_offset, obu.payload_size);
    pos += obu.total_size;
    obus.push_back(obu);
  }
  if (obus.empty()) {
    error = "no OBUs";
    return false;
  }
  return true;
}

bool parse_frame_header(std::span<const std::uint8_t> payload, ObuType type,
                        const SequenceConfig& config,
                        ParsedFrameHeader& header, std::string& error) {
  error.clear();
  header = ParsedFrameHeader{};
  if (!is_tile_group_obu(type)) {
    error = "OBU is not a tile group";
    return false;
  }
  const bool key = is_key_obu(type);
  const bool olk = type == ObuType::kOpenLoopKey;
  const bool leading = type == ObuType::kLeadingTileGroup;
  header.kind = kind_from_obu(type);
  header.key_frame = key;
  header.is_regular = !leading;

  BitReader reader(payload);
  if (!read_expect_bit(reader, true, "is_first_tile_group", error)) {
    return false;
  }

  std::uint32_t value = 0;
  if (!reader.read_uvlc(value)) {
    error = "truncated cur_mfh_id";
    return false;
  }
  if (value != 0U) {
    error = "unsupported cur_mfh_id != 0";
    return false;
  }
  if (!reader.read_uvlc(value) || value != 0U) {
    error = "unsupported seq_header_id_in_frame_header != 0";
    return false;
  }

  if (!key) {
    if (!read_expect_bit(reader, true, "frame_is_inter", error)) return false;
  }

  if (olk) {
    // immediate_output_frame is inferred 0 for an OLK.
    header.immediate_output_frame = false;
    if (!config.monotonic_output_order_flag) {
      bool implicit = false;
      if (!reader.read_bit(implicit)) {
        error = "truncated implicit_output_frame";
        return false;
      }
      header.implicit_output_frame = implicit;
    }
  } else if (leading) {
    bool immediate = false;
    if (!reader.read_bit(immediate)) {
      error = "truncated immediate_output_frame";
      return false;
    }
    header.immediate_output_frame = immediate;
    if (!immediate && !config.monotonic_output_order_flag) {
      bool implicit = false;
      if (!reader.read_bit(implicit)) {
        error = "truncated implicit_output_frame";
        return false;
      }
      header.implicit_output_frame = implicit;
    }
  } else {
    if (!read_expect_bit(reader, true, "immediate_output_frame", error)) {
      return false;
    }
    header.immediate_output_frame = true;
  }

  if (!read_expect_bit(reader, false, "frame_size_override_flag", error)) {
    return false;
  }
  std::uint64_t order_hint = 0;
  if (!reader.read_bits(config.order_hint_bits, order_hint)) {
    error = "truncated order_hint";
    return false;
  }
  header.order_hint = static_cast<std::uint32_t>(order_hint);

  if (key) {
    if (olk) {
      if (!reader.read_bits(config.num_ref_frames, order_hint)) {
        error = "truncated refresh_frame_flags";
        return false;
      }
      header.refresh_frame_flags = static_cast<std::uint32_t>(order_hint);
    } else {
      const std::uint32_t all_slots =
          (config.num_ref_frames == 16U)
              ? 0xFFFFU
              : ((1U << config.num_ref_frames) - 1U);
      header.refresh_frame_flags = all_slots;
    }
    if (!read_expect_bit(reader, false, "allow_intrabc", error)) return false;
  } else {
    if (!read_expect_bit(reader, false, "signal_primary_ref_frame", error)) {
      return false;
    }
    if (!read_expect_bit(reader, true, "disable_cross_frame_cdf_init", error)) {
      return false;
    }
    if (!reader.read_bits(config.num_ref_frames, order_hint)) {
      error = "truncated refresh_frame_flags";
      return false;
    }
    header.refresh_frame_flags = static_cast<std::uint32_t>(order_hint);
    if (!read_expect_bit(reader, true, "frame_explicit_ref_frame_map",
                         error)) {
      return false;
    }
    if (!reader.read_bits(3U, order_hint)) {
      error = "truncated num_total_refs";
      return false;
    }
    header.num_total_refs = static_cast<std::uint8_t>(order_hint);
    if (header.num_total_refs > 8U) {
      error = "num_total_refs > 8";
      return false;
    }
    unsigned ref_bits = 0;
    while ((1U << ref_bits) < config.num_ref_frames) ++ref_bits;
    for (std::uint8_t r = 0; r < header.num_total_refs; ++r) {
      if (!reader.read_bits(ref_bits, order_hint)) {
        error = "truncated ref_frame_idx";
        return false;
      }
      header.ref_frame_idx[r] = static_cast<std::uint8_t>(order_hint);
      if (header.ref_frame_idx[r] >= config.num_ref_frames) {
        error = "ref_frame_idx >= NumRefFrames";
        return false;
      }
    }
    if (!read_expect_bit(reader, false, "allow_intrabc", error)) return false;
    if (!read_expect_bit(reader, false, "use_qtr_precision_mv", error)) {
      return false;
    }
    if (!read_expect_bit(reader, true, "allow_high_precision_mv", error)) {
      return false;
    }
    if (!read_expect_bit(reader, false, "is_filter_switchable", error)) {
      return false;
    }
    if (!read_expect_bits(reader, 2U, 0U, "interpolation_filter", error)) {
      return false;
    }
  }

  bool disable_cdf = false;
  if (!reader.read_bit(disable_cdf)) {
    error = "truncated disable_cdf_update";
    return false;
  }
  header.disable_cdf_update = disable_cdf;

  // tile_info() for one tile.
  if (!read_expect_bit(reader, true, "uniform_tile_spacing_flag", error)) {
    return false;
  }
  const std::uint32_t sb_cols = (mi_cols(config) + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows(config) + 15U) >> 4U;
  if (sb_cols > 1U &&
      !read_expect_bit(reader, false, "increment_tile_cols_log2", error)) {
    return false;
  }
  if (sb_rows > 1U &&
      !read_expect_bit(reader, false, "increment_tile_rows_log2", error)) {
    return false;
  }

  std::uint64_t base_q = 0;
  if (!reader.read_bits(8U, base_q)) {
    error = "truncated base_q_idx";
    return false;
  }
  if (base_q != config.base_q_idx) {
    error = "base_q_idx does not match the sequence configuration";
    return false;
  }
  if (!read_expect_bit(reader, false, "segmentation_enabled", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "using_qmatrix", error)) return false;
  if (!read_expect_bit(reader, false, "delta_q_present", error)) return false;
  if (!read_expect_bit(reader, false, "apply_deblocking_filter[0]", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "apply_deblocking_filter[1]", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "tx_mode_select", error)) return false;
  if (!key) {
    if (!read_expect_bit(reader, false, "reference_select", error)) {
      return false;
    }
    if (!read_expect_bit(reader, false, "skip_mode_present", error)) {
      return false;
    }
  }
  if (!read_expect_bits(reader, 2U, 0U, "reduced_tx_set", error)) {
    return false;
  }

  // §5.18.10.1 film_grain_config().
  if (config.film_grain_params_present) {
    filters::grain::FilmGrainConfigContext context;
    context.film_grain_params_present = true;
    context.immediate_output_frame = header.immediate_output_frame;
    context.implicit_output_frame = header.implicit_output_frame;
    std::size_t position = reader.bit_pos();
    std::string grain_error;
    if (!filters::grain::ReadFilmGrainConfig(payload, &position, context,
                                             &header.grain, &grain_error)) {
      error = "film_grain_config: " + grain_error;
      return false;
    }
    std::uint64_t skipped = 0;
    if (!reader.read_bits(static_cast<unsigned>(position - reader.bit_pos()),
                          skipped)) {
      error = "truncated film_grain_config";
      return false;
    }
  }

  if (!reader.align_to_byte(error, "tile_group_obu")) return false;
  header.header_bytes = reader.bytes_consumed();
  if (header.header_bytes == 0U || header.header_bytes > payload.size()) {
    error = "invalid tile group header size";
    return false;
  }
  return true;
}

bool parse_sequence_header(std::span<const std::uint8_t> payload,
                           SequenceConfig& config, std::string& error) {
  error.clear();
  BitReader reader(payload);
  std::uint32_t value = 0;
  if (!reader.read_uvlc(value) || value != 0U) {
    error = "seq_header_id != 0";
    return false;
  }
  if (!read_expect_bits(reader, 5U, 0U, "seq_profile_idc", error)) return false;
  if (!read_expect_bit(reader, false, "single_picture_header_flag", error)) {
    return false;
  }
  std::uint64_t level = 0;
  if (!reader.read_bits(5U, level)) {
    error = "truncated seq_level_idx";
    return false;
  }
  config.seq_level_idx = static_cast<std::uint8_t>(level);
  if (config.seq_level_idx > 3U &&
      !read_expect_bit(reader, false, "seq_tier", error)) {
    return false;
  }
  if (!reader.read_uvlc(value) || value != 0U) {
    error = "chroma_format_idc != 4:2:0";
    return false;
  }
  if (!reader.read_uvlc(value) || value != 1U) {
    error = "bit_depth_idc != 1";
    return false;
  }

  if (!read_expect_bits(reader, 3U, 0U, "seq_lcr_id", error)) return false;
  if (!read_expect_bit(reader, false, "still_picture", error)) return false;
  if (!read_expect_bits(reader, 2U, 0U, "max_tlayer_id", error)) return false;
  if (!read_expect_bits(reader, 3U, 0U, "max_mlayer_id", error)) return false;
  bool monotonic = false;
  if (!reader.read_bit(monotonic)) {
    error = "truncated monotonic_output_order_flag";
    return false;
  }
  config.monotonic_output_order_flag = monotonic;

  std::uint64_t width_bits_minus_1 = 0;
  std::uint64_t height_bits_minus_1 = 0;
  if (!reader.read_bits(4U, width_bits_minus_1) ||
      !reader.read_bits(4U, height_bits_minus_1)) {
    error = "truncated frame size bit widths";
    return false;
  }
  std::uint64_t width_minus_1 = 0;
  std::uint64_t height_minus_1 = 0;
  if (!reader.read_bits(static_cast<unsigned>(width_bits_minus_1) + 1U,
                        width_minus_1) ||
      !reader.read_bits(static_cast<unsigned>(height_bits_minus_1) + 1U,
                        height_minus_1)) {
    error = "truncated max frame size";
    return false;
  }
  if (width_minus_1 + 1U > 0xFFFFU || height_minus_1 + 1U > 0xFFFFU) {
    error = "max frame size exceeds uint16";
    return false;
  }
  config.max_frame_width = static_cast<std::uint16_t>(width_minus_1 + 1U);
  config.max_frame_height = static_cast<std::uint16_t>(height_minus_1 + 1U);

  if (!read_expect_bit(reader, false, "seq_cropping_window_present_flag",
                       error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "seq_initial_display_delay_present_flag",
                       error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "decoder_model_info_present_flag",
                       error)) {
    return false;
  }

  if (!read_expect_bit(reader, false, "use_256x256_superblock", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "use_128x128_superblock", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_sdp", error)) return false;
  if (!read_expect_bit(reader, false, "enable_ext_partitions", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "reduce_pb_aspect_ratio", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_ext_seg", error)) return false;
  if (!read_expect_bit(reader, false, "seq_seg_info_present_flag", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_dip", error)) return false;
  if (!read_expect_bit(reader, true, "enable_intra_edge_filter", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_mrls", error)) return false;
  if (!read_expect_bit(reader, true, "enable_cfl_intra", error)) return false;
  if (!read_expect_bits(reader, 2U, 0U, "cfl_ds_filter_index", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_mhccp", error)) return false;
  if (!read_expect_bit(reader, false, "enable_ibp", error)) return false;

  for (int mode = 1; mode < 5; ++mode) {
    if (!read_expect_bit(reader, false, "seq_enabled_motion_modes", error)) {
      return false;
    }
  }
  if (!read_expect_bit(reader, false, "enable_masked_compound", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_ref_frame_mvs", error)) {
    return false;
  }
  std::uint64_t order_hint_bits_minus_1 = 0;
  if (!reader.read_bits(4U, order_hint_bits_minus_1)) {
    error = "truncated order_hint_bits_minus_1";
    return false;
  }
  config.order_hint_bits =
      static_cast<std::uint8_t>(order_hint_bits_minus_1 + 1U);
  if (!read_expect_bit(reader, false, "enable_refmvbank", error)) return false;
  if (!read_expect_bit(reader, true, "disable_drl_reorder", error)) {
    return false;
  }
  if (!read_expect_bit(reader, true, "explicit_ref_frame_map", error)) {
    return false;
  }
  if (!read_expect_bit(reader, true, "explicit_num_ref_frames", error)) {
    return false;
  }
  std::uint64_t num_ref_frames_minus_1 = 0;
  if (!reader.read_bits(4U, num_ref_frames_minus_1)) {
    error = "truncated num_ref_frames_minus_1";
    return false;
  }
  config.num_ref_frames =
      static_cast<std::uint8_t>(num_ref_frames_minus_1 + 1U);
  if (!read_expect_bits(reader, 3U, 0U, "long_term_frame_id_bits", error)) {
    return false;
  }
  if (!reader.read_ns(5U, value)) {
    error = "truncated seq_max_drl_bits_minus_1";
    return false;
  }
  config.max_drl_bits_minus_1 = static_cast<std::uint8_t>(value);
  if (!read_expect_bit(reader, false, "allow_frame_max_drl_bits", error)) {
    return false;
  }
  if (!reader.read_ns(3U, value) || value != 0U) {
    error = "unexpected seq_max_bvp_drl_bits_minus_1";
    return false;
  }
  if (!read_expect_bit(reader, false, "allow_frame_max_bvp_drl_bits", error)) {
    return false;
  }
  if (!read_expect_bits(reader, 2U, 0U, "num_same_ref_compound", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_tip", error)) return false;
  if (!read_expect_bit(reader, false, "enable_mv_traj", error)) return false;
  if (!read_expect_bit(reader, false, "enable_bawp", error)) return false;
  if (!read_expect_bit(reader, false, "enable_cwp", error)) return false;
  if (!read_expect_bit(reader, false, "enable_imp_msk_bld", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_df_sub_pu", error)) return false;
  if (!read_expect_bits(reader, 2U, 0U, "enable_opfl_refine", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_refinemv", error)) return false;
  if (!read_expect_bit(reader, false, "enable_bru", error)) return false;
  if (!read_expect_bit(reader, false, "enable_adaptive_mvd", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_mvd_sign_derive", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_flex_mvres", error)) return false;
  if (!read_expect_bit(reader, false, "enable_global_motion", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_short_refresh_frame_flags",
                       error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "seq_choose_screen_content_tools",
                       error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "seq_force_screen_content_tools",
                       error)) {
    return false;
  }

  if (!read_expect_bit(reader, false, "enable_fsc", error)) return false;
  if (!read_expect_bit(reader, false, "enable_idtx_intra", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_intra_ist", error)) return false;
  if (!read_expect_bit(reader, false, "enable_inter_ist", error)) return false;
  if (!read_expect_bit(reader, true, "enable_chroma_dctonly", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_inter_ddt", error)) return false;
  if (!read_expect_bit(reader, false, "reduced_tx_part_set", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_cctx", error)) return false;
  if (!read_expect_bit(reader, false, "enable_tcq", error)) return false;
  if (!read_expect_bit(reader, false, "enable_parity_hiding", error)) {
    return false;
  }
  bool avg_cdf = false;
  if (!reader.read_bit(avg_cdf)) {
    error = "truncated enable_avg_cdf";
    return false;
  }
  config.enable_avg_cdf = avg_cdf;
  if (avg_cdf) {
    bool avg_cdf_type = false;
    if (!reader.read_bit(avg_cdf_type)) {
      error = "truncated avg_cdf_type";
      return false;
    }
    config.avg_cdf_type = avg_cdf_type;
  }
  if (!read_expect_bit(reader, false, "separate_uv_delta_q", error)) {
    return false;
  }
  if (!read_expect_bit(reader, true, "equal_ac_dc_q", error)) return false;
  if (!read_expect_bits(reader, 5U, 23U, "base_uv_ac_delta_q", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "uv_ac_delta_q_enabled", error)) {
    return false;
  }

  if (!read_expect_bit(reader, false, "disable_loopfilters_across_tiles",
                       error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_cdef", error)) return false;
  if (!read_expect_bit(reader, false, "enable_gdf", error)) return false;
  if (!read_expect_bit(reader, false, "enable_restoration", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "enable_ccso", error)) return false;
  if (!read_expect_bit(reader, false, "cdef_on_skip_txfm_always_on", error)) {
    return false;
  }
  if (!read_expect_bit(reader, true, "cdef_on_skip_txfm_disabled", error)) {
    return false;
  }
  if (!read_expect_bits(reader, 2U, 0U, "df_par_bits_minus_2", error)) {
    return false;
  }
  if (!read_expect_bit(reader, false, "seq_tile_info_present_flag", error)) {
    return false;
  }
  bool grain_present = false;
  if (!reader.read_bit(grain_present)) {
    error = "truncated film_grain_params_present";
    return false;
  }
  config.film_grain_params_present = grain_present;

  // §5.2.1: extensible OBU payloads end with obu_extension_flag then
  // trailing_bits().  The first trailing bit is 1 and every later bit is 0.
  if (!read_expect_bit(reader, false, "obu_extension_flag", error)) {
    return false;
  }
  if (!read_expect_bit(reader, true, "trailing_one_bit", error)) return false;
  while (reader.bit_pos() < reader.bit_size()) {
    if (!read_expect_bit(reader, false, "trailing_zero_bit", error)) {
      return false;
    }
  }
  return true;
}

}  // namespace av2enc::inter::leading
