#include "bitstream/m0_headers.h"

#include <cassert>
#include <limits>

namespace av2enc::m0 {
namespace {

// uvlc() (§4.11.3): `leading_zeros` zero bits, a one bit, then
// `leading_zeros` payload bits.
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

// ns(n) (§4.11.8): non-symmetric range 0..n-1.
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

// §5.2.3 trailing_bits(): one 1 bit, then zeros to the byte boundary. The
// caller sizes the OBU payload to include this bit; if the syntax already ends
// on a byte boundary the trailing 1 starts a fresh 0x80 byte.
void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

// Superblock grid for M0: SbSize = BLOCK_64X64, so sb4x4 = 16 and sbShift = 4
// (§5.18.7.3).
struct TileGrid {
  std::uint32_t sb_cols = 0;
  std::uint32_t sb_rows = 0;
};

TileGrid m0_tile_grid(const StillPictureConfig& config) {
  const std::uint32_t frame_width = config.max_frame_width;
  const std::uint32_t frame_height = config.max_frame_height;
  // MiCols = 2 * ((frameWidth + 7) >> 3), MiRows likewise (§5.18.7.3).
  const std::uint32_t mi_cols = 2U * ((frame_width + 7U) >> 3U);
  const std::uint32_t mi_rows = 2U * ((frame_height + 7U) >> 3U);
  return TileGrid{(mi_cols + 15U) >> 4U, (mi_rows + 15U) >> 4U};
}

// tile_info() for the one-tile M0 case (§5.18.7.2, §5.18.7.3).
std::size_t write_tile_info(BitWriter& writer, const StillPictureConfig& config) {
  const std::size_t start = writer.bit_size();
  const TileGrid grid = m0_tile_grid(config);

  // One tile is legal for every M0 size. Level 5.1 has
  // Tile_Width_Scaling_Factor = 4 and Tile_Area_Scaling_Factor = 4, so with
  // SbSize = 64x64 (sbShift = 4):
  //   maxTileWidthSb = (4 * 4096) >> 8          = 64
  //   maxTileAreaSb  = (4 * 9437184) >> 14      = 2304
  // Both constraints put minLog2TileCols/minLog2TileRows at 0 for M0.
  assert(grid.sb_cols >= 1U && grid.sb_rows >= 1U);
  assert(grid.sb_cols <= 64U && grid.sb_rows <= 64U);  // MAX_TILE_COLS/ROWS
  assert(grid.sb_cols * grid.sb_rows <= 2304U);        // level 5.1 max area

  writer.put_bit(true);  // uniform_tile_spacing_flag

  // tile_params() still codes one increment_*_log2 flag per dimension while
  // the current log2 is below maxLog2Tile{Cols,Rows}. Writing 0 breaks out and
  // keeps tileColsLog2 = tileRowsLog2 = 0, i.e. one tile.
  if (grid.sb_cols > 1U) {
    writer.put_bit(false);  // increment_tile_cols_log2 = 0
  }
  if (grid.sb_rows > 1U) {
    writer.put_bit(false);  // increment_tile_rows_log2 = 0
  }
  return writer.bit_size() - start;
}

}  // namespace

std::size_t write_obu_header(BitWriter& writer, const ObuHeader& header) {
  const std::size_t start = writer.bit_size();
  const unsigned type = static_cast<unsigned>(header.type);
  assert(type <= 31U);
  assert(header.tlayer_id <= 3U);

  writer.put_bit(header.extension_flag);  // obu_header_extension_flag
  writer.put_bits(type, 5U);              // obu_type
  writer.put_bits(header.tlayer_id, 2U);  // obu_tlayer_id

  if (header.extension_flag) {
    assert(header.mlayer_id <= 7U);
    assert(header.xlayer_id <= 31U);
    writer.put_bits(header.mlayer_id, 3U);  // obu_mlayer_id
    writer.put_bits(header.xlayer_id, 5U);  // obu_xlayer_id
  }
  return writer.bit_size() - start;
}

void append_leb128(std::vector<std::uint8_t>& out, std::uint32_t value) {
  do {
    std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7FU);
    value >>= 7U;
    if (value != 0) {
      byte = static_cast<std::uint8_t>(byte | 0x80U);
    }
    out.push_back(byte);
  } while (value != 0);
}

std::size_t leb128_size(std::uint32_t value) {
  std::size_t bytes = 1;
  while (value >= 0x80U) {
    value >>= 7U;
    ++bytes;
  }
  return bytes;
}

std::vector<std::uint8_t> make_obu(const ObuHeader& header,
                                   std::span<const std::uint8_t> payload) {
  const std::size_t header_bytes =
      kObuHeaderBytes + (header.extension_flag ? kObuHeaderExtensionBytes : 0U);
  const std::size_t obu_size = header_bytes + payload.size();
  assert(obu_size <= std::numeric_limits<std::uint32_t>::max());

  std::vector<std::uint8_t> out;
  out.reserve(leb128_size(static_cast<std::uint32_t>(obu_size)) + obu_size);
  append_leb128(out, static_cast<std::uint32_t>(obu_size));

  BitWriter header_writer;
  const std::size_t header_bits = write_obu_header(header_writer, header);
  (void)header_bits;
  assert(header_bits == header_bytes * 8U);
  const std::vector<std::uint8_t> header_bytes_vec = header_writer.take_bytes();
  assert(header_bytes_vec.size() == header_bytes);
  out.insert(out.end(), header_bytes_vec.begin(), header_bytes_vec.end());
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

std::size_t write_sequence_header(BitWriter& writer,
                                  const StillPictureConfig& config) {
  const std::size_t start = writer.bit_size();

  assert(config.max_frame_width >= 1U);
  assert(config.max_frame_height >= 1U);
  assert(config.seq_level_idx <= 31U);

  // §5.4.1 sequence_header_obu().
  put_uvlc(writer, 0);                        // seq_header_id
  writer.put_bits(0U, 5U);                    // seq_profile_idc (Annex A.2)
  writer.put_bit(true);                       // single_picture_header_flag
  writer.put_bits(config.seq_level_idx, 5U);  // seq_level_idx
  // seq_tier is absent when single_picture_header_flag == 1 (§5.4.1).
  put_uvlc(writer, 0);  // chroma_format_idc = CHROMA_FORMAT_420
  put_uvlc(writer, 1);  // bit_depth_idc = 1 => 8-bit (§6.4.1, inverted!)

  // §5.4.1 frame size. n = frame_width_bits_minus_1 + 1 must cover
  // max_frame_width_minus_1.
  unsigned width_bits = bit_width(static_cast<std::uint32_t>(config.max_frame_width) - 1U);
  if (width_bits == 0) {
    width_bits = 1;
  }
  unsigned height_bits = bit_width(static_cast<std::uint32_t>(config.max_frame_height) - 1U);
  if (height_bits == 0) {
    height_bits = 1;
  }
  assert(width_bits <= 16U);
  assert(height_bits <= 16U);
  writer.put_bits(width_bits - 1U, 4U);   // frame_width_bits_minus_1
  writer.put_bits(height_bits - 1U, 4U);  // frame_height_bits_minus_1
  writer.put_bits(static_cast<std::uint64_t>(config.max_frame_width) - 1U, width_bits);
  writer.put_bits(static_cast<std::uint64_t>(config.max_frame_height) - 1U, height_bits);

  // §5.4.1 cropping and decoder model info (absent for single picture).
  writer.put_bit(false);  // seq_cropping_window_present_flag
  // decoder_model_info_present_flag = 0 and seq_initial_display_delay absent
  // are forced by single_picture_header_flag.

  // §5.4.3 sequence_partition_config(): 64x64 SB, no SDP/ext partitions.
  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(false);  // use_128x128_superblock
  writer.put_bit(false);  // enable_sdp
  writer.put_bit(false);  // enable_ext_partitions
  writer.put_bit(false);  // reduce_pb_aspect_ratio

  // §5.4.4 sequence_segment_config().
  writer.put_bit(false);  // enable_ext_seg
  writer.put_bit(false);  // seq_seg_info_present_flag

  // §5.4.5 sequence_intra_config(): no DIP, edge filter, MRL, CfL, MHCCP, IBP.
  writer.put_bit(false);  // enable_dip
  writer.put_bit(false);  // enable_intra_edge_filter
  writer.put_bit(false);  // enable_mrls
  writer.put_bit(false);  // enable_cfl_intra
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);  // enable_mhccp
  writer.put_bit(false);  // enable_ibp

  // §5.4.6 sequence_inter_config(): single picture forces OrderHintBits = 0,
  // NumRefFrames = 2, no motion modes, no global motion.
  writer.put_bit(false);       // enable_refmvbank
  writer.put_bit(true);        // disable_drl_reorder
  put_ns(writer, 3U, 0U);      // seq_max_bvp_drl_bits_minus_1 = ns(3)
  writer.put_bit(false);       // allow_frame_max_bvp_drl_bits
  writer.put_bit(false);       // enable_bawp

  // §5.4.7 screen content config is forced by single_picture_header_flag.

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
  // enable_avg_cdf = 1, avg_cdf_type = 1 are forced by single_picture_header_flag.
  writer.put_bit(false);  // separate_uv_delta_q
  writer.put_bit(true);   // equal_ac_dc_q
  // base_uv_ac_delta_q = 23 encodes BaseUVAcDeltaQ = 0 because
  // DELTA_DCQUANT_MIN = -23 (§9 constants; §5.4.8).
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  // §5.4.10 sequence_filter_config(): no loop filters.
  writer.put_bit(false);  // disable_loopfilters_across_tiles
  writer.put_bit(false);  // enable_cdef
  writer.put_bit(false);  // enable_gdf
  writer.put_bit(false);  // enable_restoration
  writer.put_bit(false);  // enable_ccso
  // CdefOnSkipTxfm is forced to ADAPTIVE, which codes no bits here.
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2

  // §5.4.2 sequence_tile_config(): tiles are described at frame level. With
  // one tile, tile_info() still writes uniform_tile_spacing_flag (§5.18.7.3).
  writer.put_bit(false);  // seq_tile_info_present_flag

  // §5.4.1.
  writer.put_bit(false);  // film_grain_params_present

  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_sequence_header_obu(
    const StillPictureConfig& config) {
  BitWriter writer;
  write_sequence_header(writer, config);

  // §5.2.1: OBU_SEQUENCE_HEADER is an extensible OBU (is_extensible_obu()),
  // so the payload continues with obu_extension_flag f(1). We do not use an
  // extension payload, so it is 0 and is followed by trailing_bits().
  // TODO(docs): minimal-bitstream.md §2.2 omits this bit before trailing_bits.
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);

  const std::vector<std::uint8_t> payload = writer.take_bytes();
  return make_obu(ObuHeader{ObuType::kSequenceHeader}, payload);
}

std::size_t write_frame_header(BitWriter& writer,
                               const StillPictureConfig& config) {
  const std::size_t start = writer.bit_size();
  assert(config.base_q_idx >= 1U);  // 0 would set CodedLossless = 1 (§5.18.2)

  // §5.18.2 frame_header_info(); keyFrame and FrameIsIntra come from
  // obu_type == OBU_CLOSED_LOOP_KEY. single_picture_header_flag forces
  // FrameType = KEY_FRAME, ShowExistingFrame = 0, immediate_output_frame = 1.
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header -> load_sequence_header()
  // frame_size_override_flag = 0; order_hint has OrderHintBits = 0 bits;
  // refresh_frame_flags = allFrames; primary_ref_frame = PRIMARY_REF_NONE.
  writer.put_bit(false);  // allow_screen_content_tools
  writer.put_bit(false);  // allow_intrabc
  writer.put_bit(config.disable_cdf_update);  // disable_cdf_update (§8.2.6)

  const std::size_t tile_info_bits = write_tile_info(writer, config);
  (void)tile_info_bits;

  writer.put_bits(config.base_q_idx, 8U);  // base_q_idx (8-bit stream)
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix (setup_qm_params())
  writer.put_bit(false);  // delta_q_present (base_q_idx > 0, §5.18.7.8)
  writer.put_bit(false);  // apply_deblocking_filter[0]
  writer.put_bit(false);  // apply_deblocking_filter[1]
  // apply_deblocking_filter[2..3] are absent because both luma flags are 0.
  writer.put_bit(false);   // tx_mode_select = 0 -> TX_MODE_LARGEST
  writer.put_bits(0U, 2U); // reduced_tx_set

  // Every remaining frame_header_info() call is a no-op for this config:
  // gdf/cdef/lr/ccso are disabled at sequence level, FrameIsIntra skips
  // reference_select/skip_mode_present/global_motion_params, and
  // film_grain_params_present = 0.
  assert(writer.bit_size() - start == 21U + tile_info_bits);
  return writer.bit_size() - start;
}

std::size_t write_tile_group_header(BitWriter& writer,
                                    const StillPictureConfig& config) {
  // §5.19 starts at the beginning of the OBU payload, which is byte-aligned.
  assert(writer.byte_aligned());
  const std::size_t start = writer.bit_size();

  writer.put_bit(true);  // is_first_tile_group
  write_frame_header(writer, config);
  // tile_start_and_end_present_flag is absent because NumTiles == 1.
  writer.align_to_byte();  // byte_alignment() (§5.2.4: zero bits)

  assert(writer.byte_aligned());
  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_closed_loop_key_obu(
    const StillPictureConfig& config, std::span<const std::uint8_t> tile_payload) {
  BitWriter writer;
  write_tile_group_header(writer, config);

  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  return make_obu(ObuHeader{ObuType::kClosedLoopKey}, payload);
}

std::vector<std::uint8_t> make_temporal_delimiter_obu() {
  // §5.5: empty payload; §5.2.1 adds trailing_bits only when payload size > 0.
  return make_obu(ObuHeader{ObuType::kTemporalDelimiter},
                  std::span<const std::uint8_t>{});
}

std::vector<std::uint8_t> assemble_m0_still_picture(
    const StillPictureConfig& config, std::span<const std::uint8_t> tile_payload) {
  const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
  const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);
  const std::vector<std::uint8_t> clk = make_closed_loop_key_obu(config, tile_payload);

  // §7.3.3: TD -> sequence header -> tile group (one frame).
  std::vector<std::uint8_t> out;
  out.reserve(td.size() + seq.size() + clk.size());
  out.insert(out.end(), td.begin(), td.end());
  out.insert(out.end(), seq.begin(), seq.end());
  out.insert(out.end(), clk.begin(), clk.end());
  return out;
}

std::vector<std::uint8_t> assemble_m0_still_picture_placeholder(
    const StillPictureConfig& config) {
  return assemble_m0_still_picture(
      config, std::span<const std::uint8_t>(kPlaceholderTilePayload));
}

}  // namespace av2enc::m0
