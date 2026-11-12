#include "pipeline/still/high_bitdepth/high_bitdepth_headers.h"

#include <cassert>
#include <string>

#include "core/bitstream/basic_headers.h"

namespace av2enc::high_bitdepth {
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

std::uint32_t mi_cols(const HighBitDepthConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
}
std::uint32_t mi_rows(const HighBitDepthConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
}

// tile_info() for one 64x64-superblock tile (§5.18.7.2, §5.18.7.3).
std::size_t write_tile_info(BitWriter& writer, const HighBitDepthConfig& c) {
  const std::size_t start = writer.bit_size();
  const std::uint32_t sb_cols = (mi_cols(c) + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows(c) + 15U) >> 4U;
  assert(sb_cols >= 1U && sb_rows >= 1U);
  assert(sb_cols <= 64U && sb_rows <= 64U);   // MAX_TILE_COLS / MAX_TILE_ROWS
  assert(sb_cols * sb_rows <= 2304U);         // level 5.1 tile area
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
                                         const HighBitDepthConfig& c) {
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
  put_uvlc(writer, static_cast<std::uint32_t>(idc));  // bit_depth_idc = 0

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

  // §5.4.3 sequence_partition_config(): 64x64 SB, no SDP / ext partitions.
  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(false);  // use_128x128_superblock
  writer.put_bit(false);  // enable_sdp
  writer.put_bit(false);  // enable_ext_partitions
  writer.put_bit(false);  // reduce_pb_aspect_ratio

  // §5.4.4 sequence_segment_config().
  writer.put_bit(false);  // enable_ext_seg
  writer.put_bit(false);  // seq_seg_info_present_flag

  // §5.4.5 sequence_intra_config(): DC only.  M10 deliberately disables the
  // intra edge filter and CfL so the prediction path is the simple §7.13.2.10
  // DC process (no IDIF state to reproduce).
  writer.put_bit(false);  // enable_dip
  writer.put_bit(false);  // enable_intra_edge_filter
  writer.put_bit(false);  // enable_mrls
  writer.put_bit(false);  // enable_cfl_intra
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);  // enable_mhccp
  writer.put_bit(false);  // enable_ibp

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
  writer.put_bit(false);  // enable_chroma_dctonly
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

std::size_t write_frame_header_syntax(BitWriter& writer, const HighBitDepthConfig& c) {
  const std::size_t start = writer.bit_size();
  const int qbits = fmt::base_q_idx_bits(c.depth);
  assert(qbits == 9);
  assert(c.base_q_idx >= 1U);
  assert(c.base_q_idx <= static_cast<unsigned>(fmt::max_quantizer(c.depth)));

  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  writer.put_bit(false);  // allow_screen_content_tools
  writer.put_bit(false);  // allow_intrabc
  writer.put_bit(c.disable_cdf_update);
  write_tile_info(writer, c);

  writer.put_bits(c.base_q_idx, static_cast<unsigned>(qbits));  // f(9)
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present
  writer.put_bit(false);  // apply_deblocking_filter[0]
  writer.put_bit(false);  // apply_deblocking_filter[1]
  // apply_deblocking_filter[2..3] are absent (both luma flags 0).
  writer.put_bit(false);   // tx_mode_select = 0 -> TX_MODE_LARGEST
  writer.put_bits(0U, 2U); // reduced_tx_set

  return writer.bit_size() - start;
}

}  // namespace

bool validate_config(const HighBitDepthConfig& config, std::string* error) {
  const auto fail = [error](const std::string& text) {
    if (error != nullptr) *error = text;
    return false;
  };
  if (config.depth != fmt::BitDepth::k10) {
    return fail("M10 is 10-bit only; 8-bit and 12-bit are rejected");
  }
  if (fmt::bit_depth_idc(config.depth) != 0) {
    return fail("10-bit must serialize bit_depth_idc = 0");
  }
  if (config.max_frame_width == 0 || config.max_frame_height == 0) {
    return fail("M10 frame dimensions must be non-zero");
  }
  if (config.max_frame_width > 65535U || config.max_frame_height > 65535U) {
    return fail("M10 frame dimensions exceed the uvlc range");
  }
  if (config.seq_level_idx > 31U) {
    return fail("seq_level_idx must fit f(5)");
  }
  if (config.base_q_idx == 0) {
    return fail("base_q_idx = 0 would enable lossless coding");
  }
  const int max_q = fmt::max_quantizer(config.depth);
  if (config.base_q_idx > static_cast<unsigned>(max_q)) {
    return fail("base_q_idx exceeds MaxQ = 303 at 10-bit");
  }
  // The shipped coefficient CDF/quant tables are slice 0 (raw base_q_idx <= 90,
  // §8.3.1).  The forward quantizer is the fixed QP-90 reference (the M0/M1
  // path also pins it), so emitting another qindex would desynchronize the
  // decoder's dequantization.  Keep the raw value in the header.
  if (config.base_q_idx != 90U) {
    return fail("M10 pins raw base_q_idx = 90 (fixed quantizer); "
                "other qindexes need a q-dependent quantizer");
  }
  return true;
}

std::size_t write_sequence_header(BitWriter& writer,
                                  const HighBitDepthConfig& config) {
  return write_sequence_header_syntax(writer, config);
}

std::vector<std::uint8_t> make_sequence_header_obu(const HighBitDepthConfig& config) {
  BitWriter writer;
  write_sequence_header_syntax(writer, config);
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

std::size_t write_frame_header(BitWriter& writer, const HighBitDepthConfig& config) {
  return write_frame_header_syntax(writer, config);
}

std::size_t write_tile_group_header(BitWriter& writer,
                                    const HighBitDepthConfig& config) {
  const std::size_t start = writer.bit_size();
  assert(writer.byte_aligned());
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_syntax(writer, config);
  // tile_start_and_end_present_flag is absent because NumTiles == 1.
  writer.align_to_byte();  // byte_alignment() (§5.2.4)
  assert(writer.byte_aligned());
  return writer.bit_size() - start;
}

std::vector<std::uint8_t> assemble_high_bitdepth_still_picture(
    const HighBitDepthConfig& config, std::span<const std::uint8_t> tile_payload) {
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

HeaderBitCounts header_bit_counts(const HighBitDepthConfig& config) {
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

}  // namespace av2enc::high_bitdepth
