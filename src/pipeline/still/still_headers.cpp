#include "pipeline/still/still_headers.h"

#include "core/bitstream/basic_headers.h"

#include <cassert>

namespace av2enc::still {
namespace {

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

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

std::uint32_t mi_cols(const StillConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
}
std::uint32_t mi_rows(const StillConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
}

void write_tile_info(BitWriter& writer, const StillConfig& c) {
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

std::size_t write_sequence_header(BitWriter& writer, const StillConfig& c) {
  const std::size_t start = writer.bit_size();

  put_uvlc(writer, 0);                        // seq_header_id
  writer.put_bits(0U, 5U);                    // seq_profile_idc
  writer.put_bit(true);                       // single_picture_header_flag
  writer.put_bits(c.seq_level_idx, 5U);       // seq_level_idx
  put_uvlc(writer, 0);                        // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                        // bit_depth_idc = 1 => 8-bit

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

  // §5.4.6 sequence_inter_config(): forced by single_picture_header_flag.
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder
  put_ns(writer, 3U, 0U); // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);  // allow_frame_max_bvp_drl_bits
  writer.put_bit(false);  // enable_bawp

  // §5.4.8 sequence_transform_quant_entropy_config().
  writer.put_bit(false);  // enable_fsc
  writer.put_bit(c.enable_idtx_intra);  // enable_idtx_intra
  writer.put_bit(false);  // enable_intra_ist
  writer.put_bit(false);  // enable_inter_ist
  writer.put_bit(c.enable_chroma_dctonly);
  writer.put_bit(false);  // reduced_tx_part_set
  writer.put_bit(false);  // enable_cctx
  writer.put_bit(false);  // enable_tcq
  writer.put_bit(false);  // enable_parity_hiding
  writer.put_bit(false);  // separate_uv_delta_q
  writer.put_bit(true);   // equal_ac_dc_q
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q => BaseUVAcDeltaQ = 0
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  // §5.4.10 sequence_filter_config().
  writer.put_bit(false);  // disable_loopfilters_across_tiles
  writer.put_bit(c.cdef.enable);  // enable_cdef
  writer.put_bit(false);  // enable_gdf
  writer.put_bit(false);  // enable_restoration
  // §5.4.10 sequence_filter_config() CCSO row.
  writer.put_bit(c.ccso.enable);  // enable_ccso
  if (c.ccso.enable) {
    writer.put_bit(true);  // ccso_unit_matches_sb_size (64x64 units)
  }
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2

  // §5.4.2 sequence_tile_config().
  writer.put_bit(false);  // seq_tile_info_present_flag

  writer.put_bit(false);  // film_grain_params_present

  return writer.bit_size() - start;
}

std::size_t write_frame_header_syntax(BitWriter& writer, const StillConfig& c) {
  const std::size_t start = writer.bit_size();
  assert(c.base_q_idx >= 1U);

  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  writer.put_bit(false);  // allow_screen_content_tools
  writer.put_bit(false);  // allow_intrabc
  writer.put_bit(c.disable_cdf_update);
  write_tile_info(writer, c);

  writer.put_bits(c.base_q_idx, 8U);  // base_q_idx
  writer.put_bit(false);              // segmentation_enabled
  writer.put_bit(false);              // using_qmatrix
  writer.put_bit(false);              // delta_q_present

  // deblocking_filter_params() (§5.18.5.2). Chroma flags are only coded when
  // at least one luma direction is enabled.
  writer.put_bit(c.apply_deblock_y_vert);    // apply_deblocking_filter[0]
  writer.put_bit(c.apply_deblock_y_horz);    // apply_deblocking_filter[1]
  const bool have_chroma_deblock =
      c.apply_deblock_y_vert || c.apply_deblock_y_horz;
  if (have_chroma_deblock) {
    writer.put_bit(c.apply_deblock_u);       // apply_deblocking_filter[2]
    writer.put_bit(c.apply_deblock_v);       // apply_deblocking_filter[3]
  }

  // §5.18.5.2: one df_delta_q_present f(1) per enabled plane, in plane order,
  // after all apply_deblocking_filter[] bits and before tx_mode_select. M1
  // sends DfDeltaQ = 0, so the bit is 0. A disabled plane has no bit.
  if (c.apply_deblock_y_vert) writer.put_bit(false);
  if (c.apply_deblock_y_horz) writer.put_bit(false);
  if (have_chroma_deblock && c.apply_deblock_u) writer.put_bit(false);
  if (have_chroma_deblock && c.apply_deblock_v) writer.put_bit(false);

  // §5.18.7.10 cdef_params().  single_picture_header_flag implies
  // cdef_frame_enable = 1, so no bit is coded for it here.
  if (c.cdef.enable) {
    assert(c.cdef.damping >= 3 && c.cdef.damping <= 6);
    assert(c.cdef.num_strengths >= 1 && c.cdef.num_strengths <= 8);
    writer.put_bits(static_cast<std::uint32_t>(c.cdef.damping - 3), 2U);
    writer.put_bits(static_cast<std::uint32_t>(c.cdef.num_strengths - 1), 3U);
    writer.put_bit(c.cdef.on_skip_txfm);
    for (int i = 0; i < c.cdef.num_strengths; ++i) {
      const StillCdefStrength& s = c.cdef.strengths[i];
      assert(s.y_pri >= 0 && s.y_pri <= 15);
      assert(s.uv_pri >= 0 && s.uv_pri <= 15);
      assert(s.y_sec == 0 || s.y_sec == 1 || s.y_sec == 2 || s.y_sec == 4);
      assert(s.uv_sec == 0 || s.uv_sec == 1 || s.uv_sec == 2 || s.uv_sec == 4);
      writer.put_bit(s.y_pri == 0);
      if (s.y_pri != 0) {
        writer.put_bits(static_cast<std::uint32_t>(s.y_pri), 4U);
      }
      // The syntax value 3 decodes to 4; a decoded 4 is written as 3.
      writer.put_bits(static_cast<std::uint32_t>(s.y_sec == 4 ? 3 : s.y_sec),
                      2U);
      writer.put_bit(s.uv_pri == 0);
      if (s.uv_pri != 0) {
        writer.put_bits(static_cast<std::uint32_t>(s.uv_pri), 4U);
      }
      writer.put_bits(static_cast<std::uint32_t>(s.uv_sec == 4 ? 3 : s.uv_sec),
                      2U);
    }
  }

  // §5.18.7.12 ccso_params().  single_picture_header_flag implies
  // ccso_frame_flag = 1, so no bit is coded for it.  M1 only signals the
  // band-offset-only form.
  if (c.ccso.enable) {
    for (int plane = 0; plane < 3; ++plane) {
      const StillCcsoPlane& p = c.ccso.plane[plane];
      writer.put_bit(p.enabled);
      if (!p.enabled) continue;
      assert(p.bo_only);
      assert(p.scale_idx >= 0 && p.scale_idx <= 3);
      assert(p.max_band_log2 >= 0 && p.max_band_log2 <= 6);
      writer.put_bit(p.bo_only);  // ccso_bo_only
      writer.put_bits(static_cast<std::uint32_t>(p.scale_idx), 2U);
      // bo_only => quant/ext/edge are inferred to 0.
      const int band_bits = 2 + (p.bo_only ? 1 : 0);
      writer.put_bits(static_cast<std::uint32_t>(p.max_band_log2),
                      static_cast<unsigned>(band_bits));
      const int bands = 1 << p.max_band_log2;
      for (int band = 0; band < bands; ++band) {
        // ccso_offset_idx tu(7): ones until the value, then a terminating 0
        // unless the maximum (7) is reached.
        const int idx = p.offset_idx[band];
        assert(idx >= 0 && idx <= 7);
        for (int i = 0; i < idx; ++i) writer.put_bit(true);
        if (idx < 7) writer.put_bit(false);
      }
    }
  }

  writer.put_bit(c.tx_mode_select);  // 0 = TX_MODE_LARGEST
  writer.put_bits(static_cast<std::uint32_t>(c.reduced_tx_set), 2U);

  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_temporal_delimiter_obu() {
  return av2enc::basic::make_obu(ObuHeader{ObuType::kTemporalDelimiter},
                              std::span<const std::uint8_t>{});
}

}  // namespace

std::size_t write_frame_header(BitWriter& writer, const StillConfig& config) {
  return write_frame_header_syntax(writer, config);
}

std::vector<std::uint8_t> make_sequence_header_obu(const StillConfig& config) {
  BitWriter writer;
  write_sequence_header(writer, config);
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

std::vector<std::uint8_t> assemble_still_picture(
    const StillConfig& config, std::span<const std::uint8_t> tile_payload) {
  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_syntax(writer, config);
  writer.align_to_byte();

  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  const std::vector<std::uint8_t> clk =
      av2enc::basic::make_obu(ObuHeader{ObuType::kClosedLoopKey}, payload);
  const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
  const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);

  std::vector<std::uint8_t> out;
  out.reserve(td.size() + seq.size() + clk.size());
  out.insert(out.end(), td.begin(), td.end());
  out.insert(out.end(), seq.begin(), seq.end());
  out.insert(out.end(), clk.begin(), clk.end());
  return out;
}

}  // namespace av2enc::still
