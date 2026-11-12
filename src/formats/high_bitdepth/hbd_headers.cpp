#include "formats/high_bitdepth/hbd_headers.h"

#include <cassert>
#include <string>

#include "core/bitstream/basic_headers.h"
#include "formats/high_bitdepth/hbd_depth.h"

namespace av2enc::hbd {
namespace {

using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;

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

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

std::uint32_t mi_cols(const HbdConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
}

std::uint32_t mi_rows(const HbdConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
}

// tile_info() for one 64x64-superblock tile (§5.18.7.2, §5.18.7.3).
std::size_t write_tile_info(BitWriter& writer, const HbdConfig& c) {
  const std::size_t start = writer.bit_size();
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
  return writer.bit_size() - start;
}

std::size_t write_sequence_header_syntax(BitWriter& writer,
                                         const HbdConfig& c) {
  const std::size_t start = writer.bit_size();
  const int idc = fmt::bit_depth_idc(c.depth);
  assert(idc == 0);  // 10-bit only; 12-bit cannot be signalled
  assert(fmt::chroma_format_idc(fmt::ChromaFormat::k420) == 0);

  // §5.4.1 sequence_header_obu().
  put_uvlc(writer, 0);                        // seq_header_id
  writer.put_bits(0U, 5U);                    // seq_profile_idc = 0 (4:2:0)
  writer.put_bit(true);                       // single_picture_header_flag
  writer.put_bits(c.seq_level_idx, 5U);       // seq_level_idx
  put_uvlc(writer, 0);                        // chroma_format_idc = 0 (4:2:0)
  // §6.4.1 Table 6.3: bit_depth_idc is one bit, 0 = 10-bit, 1 = 8-bit.
  put_uvlc(writer, static_cast<std::uint32_t>(idc));

  unsigned width_bits =
      bit_width(static_cast<std::uint32_t>(c.max_frame_width) - 1U);
  if (width_bits == 0) width_bits = 1;
  unsigned height_bits =
      bit_width(static_cast<std::uint32_t>(c.max_frame_height) - 1U);
  if (height_bits == 0) height_bits = 1;
  assert(width_bits <= 16U && height_bits <= 16U);
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_width) - 1U,
                  width_bits);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_height) - 1U,
                  height_bits);

  writer.put_bit(false);  // seq_cropping_window_present_flag

  // §5.4.3 sequence_partition_config(): 64x64 SB, no SDP / extended parts.
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
  writer.put_bit(c.enable_intra_edge_filter);
  writer.put_bit(false);  // enable_mrls
  writer.put_bit(c.enable_cfl_intra);
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);    // enable_mhccp
  writer.put_bit(false);    // enable_ibp

  // §5.4.6 sequence_inter_config() (single_picture_header_flag forces the
  // reference subset; fields are still present and pinned).
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder
  put_ns(writer, 3U, 0U); // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);  // allow_frame_max_bvp_drl_bits
  writer.put_bit(false);  // enable_bawp

  // §5.4.8 sequence_transform_quant_entropy_config().
  writer.put_bit(false);  // enable_fsc
  writer.put_bit(false);  // enable_idtx_intra
  writer.put_bit(false);  // enable_intra_ist
  writer.put_bit(false);  // enable_inter_ist
  writer.put_bit(c.enable_chroma_dctonly);
  writer.put_bit(false);  // reduced_tx_part_set
  writer.put_bit(false);  // enable_cctx
  writer.put_bit(false);  // enable_tcq
  writer.put_bit(false);  // enable_parity_hiding
  writer.put_bit(false);  // separate_uv_delta_q
  writer.put_bit(true);   // equal_ac_dc_q
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q -> BaseUVAcDeltaQ = 0
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  // §5.4.10 sequence_filter_config().
  writer.put_bit(false);  // disable_loopfilters_across_tiles
  writer.put_bit(false);  // enable_cdef
  writer.put_bit(false);  // enable_gdf
  writer.put_bit(false);  // enable_restoration
  writer.put_bit(false);  // enable_ccso
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2

  // §5.4.2 sequence_tile_config().
  writer.put_bit(false);  // seq_tile_info_present_flag

  writer.put_bit(false);  // film_grain_params_present
  return writer.bit_size() - start;
}

std::size_t write_frame_header_syntax(BitWriter& writer, const HbdConfig& c) {
  const std::size_t start = writer.bit_size();
  const int qbits = depth::base_q_idx_bits(depth::kBitDepth);
  assert(qbits == 9);
  assert(c.base_q_idx >= 1U);
  assert(c.base_q_idx <=
         static_cast<unsigned>(depth::max_q(depth::kBitDepth)));

  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  writer.put_bit(false);  // allow_screen_content_tools
  writer.put_bit(false);  // allow_intrabc
  writer.put_bit(c.disable_cdf_update);
  write_tile_info(writer, c);

  // §5.18.6.1: base_q_idx is f(9) because BitDepth > 8.
  writer.put_bits(c.base_q_idx, static_cast<unsigned>(qbits));
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present

  // deblocking_filter_params() (§5.18.5.2).
  writer.put_bit(c.apply_deblock_y_vert);
  writer.put_bit(c.apply_deblock_y_horz);
  const bool have_chroma_deblock =
      c.apply_deblock_y_vert || c.apply_deblock_y_horz;
  if (have_chroma_deblock) {
    writer.put_bit(c.apply_deblock_u);
    writer.put_bit(c.apply_deblock_v);
  }
  // One df_delta_q_present f(1) per enabled plane (DfDeltaQ = 0).
  if (c.apply_deblock_y_vert) writer.put_bit(false);
  if (c.apply_deblock_y_horz) writer.put_bit(false);
  if (have_chroma_deblock && c.apply_deblock_u) writer.put_bit(false);
  if (have_chroma_deblock && c.apply_deblock_v) writer.put_bit(false);

  writer.put_bit(c.tx_mode_select);  // 0 = TX_MODE_LARGEST
  writer.put_bits(static_cast<std::uint32_t>(c.reduced_tx_set), 2U);
  return writer.bit_size() - start;
}

}  // namespace

bool validate_config(const HbdConfig& config, std::string* error) {
  const auto fail = [error](const std::string& text) {
    if (error != nullptr) *error = text;
    return false;
  };
  if (config.depth != fmt::BitDepth::k10) {
    return fail("hbd is 10-bit only; 8-bit and 12-bit are rejected");
  }
  if (fmt::bit_depth_idc(config.depth) != 0) {
    return fail("10-bit must serialize bit_depth_idc = 0");
  }
  if (config.max_frame_width == 0 || config.max_frame_height == 0) {
    return fail("hbd frame dimensions must be non-zero");
  }
  if (config.seq_level_idx > 31U) {
    return fail("seq_level_idx must fit f(5)");
  }
  if (config.base_q_idx == 0) {
    return fail("base_q_idx = 0 would enable lossless coding");
  }
  const int maxq = depth::max_q(depth::kBitDepth);
  if (config.base_q_idx > static_cast<unsigned>(maxq)) {
    return fail("base_q_idx exceeds MaxQ = 303 at 10-bit");
  }
  if (config.reduced_tx_set < 0 || config.reduced_tx_set > 3) {
    return fail("reduced_tx_set must fit f(2)");
  }
  // deblocking_filter_params() only codes apply_deblocking_filter[2..3] when
  // at least one luma direction is enabled.  A chroma-only request cannot be
  // signalled, so reject it instead of silently dropping it.
  const bool luma_deblock =
      config.apply_deblock_y_vert || config.apply_deblock_y_horz;
  if (!luma_deblock && (config.apply_deblock_u || config.apply_deblock_v)) {
    return fail("chroma deblock requires a luma deblock direction");
  }
  return true;
}

std::size_t write_sequence_header(BitWriter& writer,
                                  const HbdConfig& config) {
  return write_sequence_header_syntax(writer, config);
}

std::vector<std::uint8_t> make_sequence_header_obu(const HbdConfig& config) {
  BitWriter writer;
  write_sequence_header_syntax(writer, config);
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

std::size_t write_frame_header(BitWriter& writer, const HbdConfig& config) {
  return write_frame_header_syntax(writer, config);
}

std::size_t write_tile_group_header(BitWriter& writer,
                                    const HbdConfig& config) {
  const std::size_t start = writer.bit_size();
  assert(writer.byte_aligned());
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_syntax(writer, config);
  // tile_start_and_end_present_flag is absent because NumTiles == 1.
  writer.align_to_byte();  // byte_alignment() (§5.2.4)
  assert(writer.byte_aligned());
  return writer.bit_size() - start;
}

std::vector<std::uint8_t> assemble_still_picture(
    const HbdConfig& config, std::span<const std::uint8_t> tile_payload) {
  BitWriter frame_writer;
  write_tile_group_header(frame_writer, config);
  std::vector<std::uint8_t> payload = frame_writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  const std::vector<std::uint8_t> clk =
      av2enc::basic::make_obu(ObuHeader{ObuType::kClosedLoopKey}, payload);
  const std::vector<std::uint8_t> td =
      av2enc::basic::make_obu(ObuHeader{ObuType::kTemporalDelimiter},
                           std::span<const std::uint8_t>{});
  const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);

  std::vector<std::uint8_t> out;
  out.reserve(td.size() + seq.size() + clk.size());
  out.insert(out.end(), td.begin(), td.end());
  out.insert(out.end(), seq.begin(), seq.end());
  out.insert(out.end(), clk.begin(), clk.end());
  return out;
}

HeaderBitCounts header_bit_counts(const HbdConfig& config) {
  HeaderBitCounts counts{};
  {
    BitWriter writer;
    counts.sequence_syntax_bits = write_sequence_header_syntax(writer, config);
  }
  {
    BitWriter writer;
    counts.frame_syntax_bits = write_frame_header_syntax(writer, config);
  }
  {
    BitWriter writer;
    counts.tile_group_bits = write_tile_group_header(writer, config);
  }
  return counts;
}

}  // namespace av2enc::hbd
