#include "formats/chroma/chroma_headers.h"

#include "core/bitstream/basic_headers.h"
#include "codec/filters/gdf/gdf_syntax.h"
#include "formats/high_bitdepth/hbd_lr_syntax.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::chroma {
namespace {

using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;

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
  unsigned width = 0;
  while ((1U << width) < n) {
    ++width;
  }
  const std::uint32_t m = (1U << width) - n;
  if (value < m) {
    writer.put_bits(value, width - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, width - 1U);
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

// §5.2.3 trailing_bits(): one 1 bit then zeros to the byte boundary.
void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

// ---------------------------------------------------------------------------
// Bit reader (only enough syntax for the sequence header)
// ---------------------------------------------------------------------------

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> data) : data_(data) {}

  bool get_bit(bool& value) {
    if (position_ >= data_.size() * 8U) {
      return false;
    }
    const unsigned shift = 7U - static_cast<unsigned>(position_ & 7U);
    value = ((data_[position_ >> 3U] >> shift) & 1U) != 0U;
    ++position_;
    return true;
  }

  bool get_bits(unsigned count, std::uint64_t& value) {
    value = 0;
    for (unsigned i = 0; i < count; ++i) {
      bool bit = false;
      if (!get_bit(bit)) {
        return false;
      }
      value = (value << 1U) | (bit ? 1U : 0U);
    }
    return true;
  }

  bool get_uvlc(std::uint32_t& value) {
    unsigned leading_zeros = 0;
    bool bit = false;
    for (;;) {
      if (!get_bit(bit)) {
        return false;
      }
      if (bit) {
        break;
      }
      ++leading_zeros;
      if (leading_zeros >= 32U) {
        return false;
      }
    }
    std::uint64_t payload = 0;
    if (!get_bits(leading_zeros, payload)) {
      return false;
    }
    value = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(1) << leading_zeros) - 1U + payload);
    return true;
  }

  bool get_ns(std::uint32_t n, std::uint32_t& value) {
    if (n < 2) {
      return false;
    }
    unsigned width = 0;
    while ((1U << width) < n) {
      ++width;
    }
    const std::uint32_t m = (1U << width) - n;
    std::uint64_t x = 0;
    if (!get_bits(width - 1U, x)) {
      return false;
    }
    if (x < m) {
      value = static_cast<std::uint32_t>(x);
      return true;
    }
    bool bit = false;
    if (!get_bit(bit)) {
      return false;
    }
    value = static_cast<std::uint32_t>((x << 1U) + (bit ? 1U : 0U) - m);
    return value < n;
  }

  bool peek_bit(unsigned count, bool& value) const {
    if (position_ + count >= data_.size() * 8U) {
      return false;
    }
    const std::size_t at = position_ + count;
    const unsigned shift = 7U - static_cast<unsigned>(at & 7U);
    value = ((data_[at >> 3U] >> shift) & 1U) != 0U;
    return true;
  }

  std::size_t position() const { return position_; }
  std::size_t remaining_bits() const {
    const std::size_t total = data_.size() * 8U;
    return position_ >= total ? 0 : total - position_;
  }
  bool byte_aligned() const { return (position_ & 7U) == 0; }

  bool all_remaining_zero() const {
    for (std::size_t at = position_; at < data_.size() * 8U; ++at) {
      const unsigned shift = 7U - static_cast<unsigned>(at & 7U);
      if (((data_[at >> 3U] >> shift) & 1U) != 0U) {
        return false;
      }
    }
    return true;
  }

 private:
  std::span<const std::uint8_t> data_;
  std::size_t position_ = 0;
};

// ---------------------------------------------------------------------------
// Sequence-header syntax
// ---------------------------------------------------------------------------

bool read_flag(BitReader& reader, bool& value, const char* name,
               std::string& error) {
  if (!reader.get_bit(value)) {
    error = std::string("truncated sequence header at ") + name;
    return false;
  }
  return true;
}

std::uint32_t chroma_mi_cols(const ChromaSeqConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
}
std::uint32_t chroma_mi_rows(const ChromaSeqConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
}

// tile_info() for one 64x64-superblock tile (§5.18.7.2, §5.18.7.3).  This is
// the same row the M0 frame-header writer emits.
void write_chroma_tile_info(BitWriter& writer, const ChromaSeqConfig& c) {
  const std::uint32_t sb_cols = (chroma_mi_cols(c) + 15U) >> 4U;
  const std::uint32_t sb_rows = (chroma_mi_rows(c) + 15U) >> 4U;
  assert(sb_cols >= 1U && sb_rows >= 1U);
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) writer.put_bit(false);  // increment_tile_cols_log2 = 0
  if (sb_rows > 1U) writer.put_bit(false);  // increment_tile_rows_log2 = 0
}

// §5.19 tile group header + §5.18 frame header for the filter-enabled 8-bit
// format still.  With every filter flag off this is byte-identical to
// `basic::make_closed_loop_key_obu`.
std::vector<std::uint8_t> make_chroma_filtered_key_obu(
    const ChromaSeqConfig& c, const ChromaFilterHeaderConfig& f,
    std::span<const std::uint8_t> tile_payload) {
  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group

  // §5.18.2 frame_header_info().  single_picture_header_flag forces the KEY
  // frame / no-override shape used by the M0 writer.
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  writer.put_bit(false);  // allow_screen_content_tools
  writer.put_bit(false);  // allow_intrabc
  writer.put_bit(c.disable_cdf_update);
  write_chroma_tile_info(writer, c);
  writer.put_bits(c.base_q_idx, 8U);  // base_q_idx: 8-bit stream
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present

  // §5.18.5.2 deblocking_filter_params().  The chroma flags are coded only for
  // a non-monochrome stream with at least one luma direction; DfDeltaQ = 0.
  const bool has_chroma = fmt::has_chroma(c.chroma);
  writer.put_bit(f.apply_deblock_y_vert);
  writer.put_bit(f.apply_deblock_y_horz);
  const bool have_chroma_deblock =
      has_chroma && (f.apply_deblock_y_vert || f.apply_deblock_y_horz);
  if (have_chroma_deblock) {
    writer.put_bit(f.apply_deblock_u);
    writer.put_bit(f.apply_deblock_v);
  }
  if (f.apply_deblock_y_vert) writer.put_bit(false);
  if (f.apply_deblock_y_horz) writer.put_bit(false);
  if (have_chroma_deblock && f.apply_deblock_u) writer.put_bit(false);
  if (have_chroma_deblock && f.apply_deblock_v) writer.put_bit(false);

  // §5.18.7.9 gdf_params() (header-only mode).
  if (f.enable_gdf) {
    namespace gdf = av2enc::filters::gdf;
    gdf::FrameGdfContext context;
    context.enable_gdf = true;
    context.gdf_unit_matches_sb_size = c.gdf_unit_matches_sb_size;
    context.coded_lossless = false;
    context.single_picture_header_flag = true;
    context.sb_size = 64;
    context.width = c.max_frame_width;
    context.height = c.max_frame_height;
    context.mi_cols = static_cast<int>(chroma_mi_cols(c));
    context.mi_rows = static_cast<int>(chroma_mi_rows(c));
    context.disable_loopfilters_across_tiles = false;
    context.tile_cols = 1;
    context.tile_rows = 1;
    context.mi_col_starts.assign(1, 0);
    context.mi_row_starts.assign(1, 0);
    gdf::FrameGdfParams params;
    params.frame_enable = true;
    params.per_block = false;
    params.pic_qc_idx = f.gdf_pic_qc_idx;
    params.pic_scale_idx = f.gdf_pic_scale_idx;
    std::string gdf_error;
    gdf::WriteFrameGdfParams(writer, context, &params, &gdf_error);
    assert(gdf_error.empty());
    (void)gdf_error;
  }

  // §5.18.7.10 cdef_params().  single_picture_header_flag infers
  // cdef_frame_enable = 1, so only damping / strengths / skip / strength sets
  // are coded.  8-bit strengths are used as-is (coeffShift = 0).
  if (f.enable_cdef) {
    assert(f.cdef_damping >= 3 && f.cdef_damping <= 6);
    assert(f.cdef_num_strengths >= 1 && f.cdef_num_strengths <= 8);
    writer.put_bits(static_cast<std::uint64_t>(f.cdef_damping - 3), 2U);
    writer.put_bits(static_cast<std::uint64_t>(f.cdef_num_strengths - 1), 3U);
    writer.put_bit(false);  // on_skip_txfm_frame_enable (ADAPTIVE, no skips)
    for (int i = 0; i < f.cdef_num_strengths; ++i) {
      const cdef::CdefStrength& s = f.cdef_strengths[i];
      writer.put_bit(s.y_pri == 0);
      if (s.y_pri != 0) {
        writer.put_bits(static_cast<std::uint64_t>(s.y_pri), 4U);
      }
      writer.put_bits(static_cast<std::uint64_t>(s.y_sec == 4 ? 3 : s.y_sec),
                      2U);
      if (has_chroma) {
        writer.put_bit(s.uv_pri == 0);
        if (s.uv_pri != 0) {
          writer.put_bits(static_cast<std::uint64_t>(s.uv_pri), 4U);
        }
        writer.put_bits(
            static_cast<std::uint64_t>(s.uv_sec == 4 ? 3 : s.uv_sec), 2U);
      }
    }
  }

  // §5.18.7.11 lr_params().
  if (f.enable_restoration) {
    namespace lr_syntax = av2enc::hbd::lr_syntax;
    lr_syntax::LrParams params;
    av2enc::hbd::fill_lr_params(f.lr, fmt::num_planes(c.chroma), &params);
    lr_syntax::LrHeaderContext context;
    av2enc::hbd::fill_lr_header_context(
        f.lr, fmt::num_planes(c.chroma), fmt::subsampling_x(c.chroma),
        fmt::subsampling_y(c.chroma), static_cast<int>(c.base_q_idx),
        &context);
    std::string lr_error;
    lr_syntax::WriteLrParams(writer, &params, context, &lr_error);
    assert(lr_error.empty());
    (void)lr_error;
  }

  // §5.18.7.12 ccso_params().  single_picture_header_flag infers
  // ccso_frame_flag = 1; the offset table is per plane in (d0, d1, band) order.
  if (f.enable_ccso) {
    for (int plane = 0; plane < 3; ++plane) {
      if (plane > 0 && !has_chroma) break;
      const av2enc::hbd::HbdCcsoPlaneConfig& p = f.ccso.plane[plane];
      writer.put_bit(p.enabled);
      if (!p.enabled) continue;
      writer.put_bit(p.bo_only);
      writer.put_bits(static_cast<std::uint64_t>(p.scale_idx), 2U);
      if (!p.bo_only) {
        writer.put_bits(static_cast<std::uint64_t>(p.quant_idx), 2U);
        writer.put_bits(static_cast<std::uint64_t>(p.ext_filter), 3U);
        const int quant =
            restoration::kCcsoQuantSz[p.scale_idx & 3][p.quant_idx & 3];
        if (quant != 0) writer.put_bit(p.edge_clf != 0);
      }
      const int band_bits = 2 + (p.bo_only ? 1 : 0);
      writer.put_bits(static_cast<std::uint64_t>(p.max_band_log2),
                      static_cast<unsigned>(band_bits));
      const int interval = p.bo_only ? 1 : (3 - p.edge_clf);
      const std::size_t need =
          static_cast<std::size_t>(1 << p.max_band_log2) *
          static_cast<std::size_t>(interval) *
          static_cast<std::size_t>(interval);
      assert(p.offset_idx.size() == need);
      for (std::size_t k = 0; k < need; ++k) {
        const int idx = p.offset_idx[k];
        for (int i = 0; i < idx; ++i) writer.put_bit(true);
        if (idx < 7) writer.put_bit(false);
      }
    }
  }

  writer.put_bit(false);  // tx_mode_select = 0 -> TX_MODE_LARGEST
  writer.put_bits(0U, 2U);  // reduced_tx_set
  writer.align_to_byte();  // byte_alignment() (§5.2.4)

  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  return av2enc::basic::make_obu(ObuHeader{ObuType::kClosedLoopKey}, payload);
}

}  // namespace

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

bool validate_chroma_seq_config(const ChromaSeqConfig& config,
                                std::string& error) {
  if (config.max_frame_width == 0 || config.max_frame_height == 0) {
    error = "max_frame_width and max_frame_height must be non-zero";
    return false;
  }
  if (fmt::chroma_format_idc(config.chroma) < 0) {
    error = "invalid chroma format";
    return false;
  }
  if (config.seq_level_idx > 31) {
    error = "seq_level_idx must be in [0, 31]";
    return false;
  }
  if (config.base_q_idx == 0) {
    error = "base_q_idx 0 is lossless and not implemented";
    return false;
  }
  if (config.base_q_idx > 255) {
    error = "base_q_idx must fit f(8) for an 8-bit stream";
    return false;
  }

  const ToolLegality tools = tool_legality(config.chroma);
  if (tools.monochrome) {
    // These are inferred/absent, so accepting a contradictory config would
    // silently write a different stream than the caller asked for.
    if (config.enable_sdp || config.enable_chroma_dctonly ||
        config.enable_cctx || config.separate_uv_delta_q ||
        config.enable_cfl_intra || config.enable_mhccp) {
      error = "Monochrome forces enable_sdp, enable_cfl_intra, enable_mhccp, "
              "enable_chroma_dctonly, enable_cctx and separate_uv_delta_q to 0";
      return false;
    }
  } else {
    if (!tools.sdp_syntax_present && config.enable_sdp) {
      error = "enable_sdp is not codable for this format";
      return false;
    }
  }

  if (config.use_256x256_superblock || config.use_128x128_superblock) {
    error = "the minimal still writer uses 64x64 superblocks; 128x128 and "
            "256x256 superblocks are not implemented";
    return false;
  }
  if (config.enable_sdp) {
    error = "enable_sdp = 1 is not implemented by the minimal tile writer";
    return false;
  }
  if (config.enable_ext_partitions) {
    error = "enable_ext_partitions is not implemented by this writer";
    return false;
  }
  if (config.reduce_pb_aspect_ratio) {
    error = "reduce_pb_aspect_ratio is not implemented by this writer";
    return false;
  }
  if (config.enable_ext_seg) {
    error = "enable_ext_seg is not implemented by this writer";
    return false;
  }
  if (config.seq_seg_info_present_flag) {
    error = "seq_seg_info_present_flag is not implemented by this writer";
    return false;
  }
  if (config.seq_tile_info_present_flag) {
    error =
        "seq_tile_info_present_flag = 1 requires allow_tile_info_change and "
        "tile_params() (§5.4.2), which this writer does not implement";
    return false;
  }
  if (config.film_grain_params_present) {
    error =
        "film_grain_params_present = 1 requires film_grain_config() in every "
        "frame header (§5.18.10.1), which this writer does not implement";
    return false;
  }
  if (config.enable_restoration) {
    if (!config.lr_luma_pc && !config.lr_luma_ns && !config.lr_chroma_ns) {
      error = "enable_restoration requires at least one loop-restoration tool";
      return false;
    }
    if (config.lr_chroma_ns && !fmt::has_chroma(config.chroma)) {
      error = "chroma loop restoration is not codable for 4:0:0";
      return false;
    }
    if ((config.lr_luma_pc || config.lr_luma_ns) &&
        !fmt::has_chroma(config.chroma)) {
      // Monochrome still codes the luma tool list; no restriction.
    }
  }
  if (config.enable_fsc) {
    error = "enable_fsc is not implemented by this writer";
    return false;
  }
  if (config.enable_tcq) {
    error = "enable_tcq is not implemented by this writer";
    return false;
  }
  if (config.enable_dip || config.enable_intra_edge_filter ||
      config.enable_mrls || config.enable_ibp) {
    error = "the minimal tile writer codes DC_PRED only; enable_dip, "
            "enable_intra_edge_filter, enable_mrls and enable_ibp would add "
            "mode-info syntax it does not emit";
    return false;
  }
  if (config.enable_cfl_intra || config.enable_mhccp) {
    error = "the minimal tile writer does not code CfL or MHCCP";
    return false;
  }
  if (config.enable_cctx) {
    error = "enable_cctx is not implemented by this writer";
    return false;
  }
  if (config.separate_uv_delta_q) {
    error = "separate_uv_delta_q would add diff_uv_delta to the frame header";
    return false;
  }
  if (config.uv_ac_delta_q_enabled) {
    error = "uv_ac_delta_q_enabled would add delta-q syntax to the frame header";
    return false;
  }
  if (!config.equal_ac_dc_q) {
    error = "equal_ac_dc_q = 0 would add base_y_dc_delta_q/base_uv_dc_delta_q "
            "syntax to the sequence header";
    return false;
  }
  if (config.seq_cropping_window_present_flag) {
    if (config.seq_cropping_win_left_offset >= config.max_frame_width ||
        config.seq_cropping_win_right_offset >= config.max_frame_width ||
        config.seq_cropping_win_top_offset >= config.max_frame_height ||
        config.seq_cropping_win_bottom_offset >= config.max_frame_height) {
      error = "cropping offset exceeds the maximum frame dimension";
      return false;
    }
  }
  error.clear();
  return true;
}

std::size_t write_sequence_header_syntax(BitWriter& writer,
                                         const ChromaSeqConfig& config) {
  std::string validation_error;
  if (!validate_chroma_seq_config(config, validation_error)) {
    return 0;
  }
  const std::size_t start = writer.bit_size();
  const ChromaFormatInfo info = chroma_format_info(config.chroma);
  const bool monochrome = info.monochrome;

  // §5.4.1.
  put_uvlc(writer, 0);                        // seq_header_id
  writer.put_bits(static_cast<std::uint64_t>(info.profile_idc), 5U);
  writer.put_bit(true);                       // single_picture_header_flag
  writer.put_bits(config.seq_level_idx, 5U);  // seq_level_idx
  // seq_tier is absent when single_picture_header_flag == 1.
  put_uvlc(writer, static_cast<std::uint32_t>(info.idc));  // chroma_format_idc
  put_uvlc(writer, 1);                                     // bit_depth_idc = 1 (8-bit)

  // set_chroma_format_and_bit_depth() is inferred by the decoder.

  unsigned width_bits =
      bit_width(static_cast<std::uint32_t>(config.max_frame_width) - 1U);
  if (width_bits == 0) width_bits = 1;
  unsigned height_bits =
      bit_width(static_cast<std::uint32_t>(config.max_frame_height) - 1U);
  if (height_bits == 0) height_bits = 1;
  assert(width_bits <= 16U && height_bits <= 16U);
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(config.max_frame_width) - 1U,
                  width_bits);
  writer.put_bits(static_cast<std::uint64_t>(config.max_frame_height) - 1U,
                  height_bits);

  writer.put_bit(config.seq_cropping_window_present_flag);
  if (config.seq_cropping_window_present_flag) {
    put_uvlc(writer, config.seq_cropping_win_left_offset);
    put_uvlc(writer, config.seq_cropping_win_right_offset);
    put_uvlc(writer, config.seq_cropping_win_top_offset);
    put_uvlc(writer, config.seq_cropping_win_bottom_offset);
  }

  // single_picture_header_flag forces decoder_model_info_present_flag = 0, so
  // no display-delay or decoder-model syntax follows.

  // §5.4.3 sequence_partition_config().
  writer.put_bit(config.use_256x256_superblock);
  if (!config.use_256x256_superblock) {
    writer.put_bit(config.use_128x128_superblock);
  }
  if (monochrome) {
    // enable_sdp = 0, no bit.
  } else {
    writer.put_bit(config.enable_sdp);
  }
  // enable_extended_sdp is absent for single_picture_header_flag = 1 or
  // enable_sdp = 0.
  writer.put_bit(config.enable_ext_partitions);
  // enable_uneven_4way_partitions is absent because enable_ext_partitions = 0.
  writer.put_bit(config.reduce_pb_aspect_ratio);
  // max_pb_aspect_ratio_log2_minus_1 is absent because reduce_pb = 0.

  // §5.4.4 sequence_segment_config().
  writer.put_bit(config.enable_ext_seg);
  writer.put_bit(config.seq_seg_info_present_flag);

  // §5.4.5 sequence_intra_config().
  writer.put_bit(config.enable_dip);
  writer.put_bit(config.enable_intra_edge_filter);
  writer.put_bit(config.enable_mrls);
  writer.put_bit(config.enable_cfl_intra);
  if (monochrome) {
    // cfl_ds_filter_index = 0, no bits.
  } else {
    writer.put_bits(static_cast<std::uint64_t>(config.cfl_ds_filter_index), 2U);
  }
  writer.put_bit(config.enable_mhccp);
  writer.put_bit(config.enable_ibp);

  // §5.4.6 sequence_inter_config(): single-picture forced fields.
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder
  put_ns(writer, 3U, 0U); // seq_max_bvp_drl_bits_minus_1 = ns(3)
  writer.put_bit(false);  // allow_frame_max_bvp_drl_bits
  writer.put_bit(false);  // enable_bawp

  // §5.4.8 sequence_transform_quant_entropy_config().
  writer.put_bit(config.enable_fsc);
  if (config.enable_fsc) {
    // enable_idtx_intra = 1, no bit.
  } else {
    writer.put_bit(config.enable_idtx_intra);
  }
  writer.put_bit(config.enable_intra_ist);
  writer.put_bit(config.enable_inter_ist);
  if (monochrome) {
    // enable_chroma_dctonly = 0, no bit.
  } else {
    writer.put_bit(config.enable_chroma_dctonly);
  }
  // enable_inter_ddt is absent for single_picture_header_flag = 1.
  writer.put_bit(config.reduce_tx_part_set);
  if (monochrome) {
    // enable_cctx = 0, no bit.
  } else {
    writer.put_bit(config.enable_cctx);
  }
  writer.put_bit(config.enable_tcq);
  // choose_tcq_per_frame is absent for single_picture_header_flag = 1.
  if (config.enable_tcq) {
    // choose_tcq_per_frame = 0 forces enable_parity_hiding = 0, no bit.
  } else {
    writer.put_bit(config.enable_parity_hiding);
  }
  // enable_avg_cdf = 1 and avg_cdf_type = 1 are forced by
  // single_picture_header_flag = 1, no bits.
  if (monochrome) {
    // separate_uv_delta_q = 0, no bit.
  } else {
    writer.put_bit(config.separate_uv_delta_q);
  }
  writer.put_bit(config.equal_ac_dc_q);
  if (!config.equal_ac_dc_q) {
    // base_y_dc_delta_q / y_dc_delta_q_enabled are not implemented; the
    // validator pins equal_ac_dc_q = 1 for this writer.
    assert(false);
  }
  if (!monochrome) {
    // base_uv_dc_delta_q / uv_dc_delta_q_enabled are absent because
    // equal_ac_dc_q = 1.
    writer.put_bits(static_cast<std::uint64_t>(config.base_uv_ac_delta_q), 5U);
    writer.put_bit(config.uv_ac_delta_q_enabled);
  }

  // §5.4.10 sequence_filter_config().
  writer.put_bit(config.disable_loopfilters_across_tiles);
  writer.put_bit(config.enable_cdef);
  writer.put_bit(config.enable_gdf);
  if (config.enable_gdf) {
    writer.put_bit(config.gdf_unit_matches_sb_size);
  }
  writer.put_bit(config.enable_restoration);
  if (config.enable_restoration) {
    // §5.4.10 loop-restoration tool list.  lr_tools_uv_present is coded so a
    // luma-only signal does not inherit the luma tool list into planes 1/2.
    writer.put_bit(!config.lr_luma_pc);  // lr_tools_disable[0][PC_WIENER]
    writer.put_bit(!config.lr_luma_ns);  // lr_tools_disable[0][WIENER_NONSEP]
    const bool uv_present = config.lr_tools_uv_present && fmt::has_chroma(config.chroma);
    writer.put_bit(uv_present);  // lr_tools_uv_present
    if (uv_present) {
      writer.put_bit(!config.lr_chroma_ns);  // lr_tools_disable[1][WIENER_NONSEP]
    }
  }
  writer.put_bit(config.enable_ccso);
  if (config.enable_ccso) {
    writer.put_bit(config.ccso_unit_matches_sb_size);
  }
  // CdefOnSkipTxfm is forced by single_picture_header_flag = 1.
  writer.put_bits(config.df_par_bits_minus_2, 2U);

  // §5.4.2 sequence_tile_config().
  writer.put_bit(config.seq_tile_info_present_flag);

  // §5.4.1 tail.
  writer.put_bit(config.film_grain_params_present);

  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_sequence_header_obu(
    const ChromaSeqConfig& config) {
  std::string validation_error;
  if (!validate_chroma_seq_config(config, validation_error)) return {};
  BitWriter writer;
  write_sequence_header_syntax(writer, config);
  writer.put_bit(false);  // §5.2.1 obu_extension_flag
  put_trailing_bits(writer);

  const std::vector<std::uint8_t> payload = writer.take_bytes();
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader}, payload);
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

bool read_sequence_header_syntax(std::span<const std::uint8_t> payload,
                                 SequenceHeaderInfo& info, std::string& error) {
  info = SequenceHeaderInfo{};
  BitReader reader(payload);

  std::uint32_t uvlc_value = 0;
  if (!reader.get_uvlc(uvlc_value)) {
    error = "truncated seq_header_id";
    return false;
  }
  if (uvlc_value != 0) {
    error = "unsupported seq_header_id " + std::to_string(uvlc_value);
    return false;
  }

  std::uint64_t field = 0;
  if (!reader.get_bits(5, field)) {
    error = "truncated seq_profile_idc";
    return false;
  }
  info.seq_profile_idc = static_cast<int>(field);
  if (!read_flag(reader, info.single_picture_header_flag,
                 "single_picture_header_flag", error)) {
    return false;
  }
  if (!reader.get_bits(5, field)) {
    error = "truncated seq_level_idx";
    return false;
  }
  info.seq_level_idx = static_cast<int>(field);
  if (info.seq_level_idx > 3 && !info.single_picture_header_flag) {
    if (!read_flag(reader, info.seq_tier, "seq_tier", error)) {
      return false;
    }
  }
  if (!reader.get_uvlc(uvlc_value)) {
    error = "truncated chroma_format_idc";
    return false;
  }
  info.chroma_format_idc = static_cast<int>(uvlc_value);
  const auto format = chroma_format_from_idc(info.chroma_format_idc);
  if (!format) {
    error = "chroma_format_idc > 3 is reserved (§6.4.1 Table 6.2)";
    return false;
  }
  info.chroma = *format;
  if (!reader.get_uvlc(uvlc_value)) {
    error = "truncated bit_depth_idc";
    return false;
  }
  info.bit_depth_idc = static_cast<int>(uvlc_value);
  if (info.bit_depth_idc > 1) {
    error = "bit_depth_idc > 1 is reserved for AOMedia (§6.4.1 Table 6.3)";
    return false;
  }
  info.bit_depth = info.bit_depth_idc == 1 ? 8 : 10;
  info.max_quantizer = info.bit_depth_idc == 1 ? 255 : 303;
  const ChromaFormatInfo cinfo = chroma_format_info(info.chroma);
  info.monochrome = cinfo.monochrome;
  info.subsampling_x = cinfo.subsampling_x;
  info.subsampling_y = cinfo.subsampling_y;
  info.num_planes = cinfo.num_planes;

  if (!info.single_picture_header_flag) {
    error = "this reader only supports single_picture_header_flag = 1 "
            "(the still-picture profile used by the format gate)";
    return false;
  }

  unsigned width_bits = 0;
  unsigned height_bits = 0;
  if (!reader.get_bits(4, field)) {
    error = "truncated frame_width_bits_minus_1";
    return false;
  }
  width_bits = static_cast<unsigned>(field) + 1U;
  if (!reader.get_bits(4, field)) {
    error = "truncated frame_height_bits_minus_1";
    return false;
  }
  height_bits = static_cast<unsigned>(field) + 1U;
  if (width_bits > 16 || height_bits > 16) {
    error = "frame size bit count exceeds 16";
    return false;
  }
  if (!reader.get_bits(width_bits, field)) {
    error = "truncated max_frame_width_minus_1";
    return false;
  }
  info.max_frame_width = static_cast<std::uint32_t>(field) + 1U;
  if (!reader.get_bits(height_bits, field)) {
    error = "truncated max_frame_height_minus_1";
    return false;
  }
  info.max_frame_height = static_cast<std::uint32_t>(field) + 1U;

  if (!read_flag(reader, info.cropping_window_present,
                 "seq_cropping_window_present_flag", error)) {
    return false;
  }
  if (info.cropping_window_present) {
    std::uint32_t* values[4] = {&info.crop_left, &info.crop_right,
                                &info.crop_top, &info.crop_bottom};
    for (std::uint32_t* value : values) {
      if (!reader.get_uvlc(uvlc_value)) {
        error = "truncated cropping window offset";
        return false;
      }
      *value = uvlc_value;
    }
  }

  // single_picture_header_flag: decoder_model_info_present_flag = 0.

  // §5.4.3 sequence_partition_config().
  bool flag = false;
  if (!read_flag(reader, flag, "use_256x256_superblock", error)) return false;
  const bool use_256 = flag;
  bool use_128 = false;
  if (!use_256) {
    if (!read_flag(reader, use_128, "use_128x128_superblock", error)) {
      return false;
    }
  }
  if (info.monochrome) {
    info.enable_sdp = false;
  } else {
    if (!read_flag(reader, info.enable_sdp, "enable_sdp", error)) return false;
  }
  if (info.enable_sdp && !info.single_picture_header_flag) {
    if (!read_flag(reader, flag, "enable_extended_sdp", error)) return false;
  }
  if (!read_flag(reader, info.enable_ext_partitions,
                 "enable_ext_partitions", error)) {
    return false;
  }
  if (info.enable_ext_partitions) {
    if (!read_flag(reader, flag, "enable_uneven_4way_partitions", error)) {
      return false;
    }
  }
  if (!read_flag(reader, flag, "reduce_pb_aspect_ratio", error)) return false;
  if (flag) {
    if (!read_flag(reader, flag, "max_pb_aspect_ratio_log2_minus_1", error)) {
      return false;
    }
  }

  // §5.4.4 sequence_segment_config().
  if (!read_flag(reader, flag, "enable_ext_seg", error)) return false;
  if (!read_flag(reader, flag, "seq_seg_info_present_flag", error)) return false;
  if (flag) {
    error = "seq_seg_info_present_flag = 1 is not supported by this reader";
    return false;
  }

  // §5.4.5 sequence_intra_config().
  if (!read_flag(reader, flag, "enable_dip", error)) return false;
  if (!read_flag(reader, flag, "enable_intra_edge_filter", error)) return false;
  if (!read_flag(reader, flag, "enable_mrls", error)) return false;
  if (!read_flag(reader, info.enable_cfl_intra, "enable_cfl_intra", error)) {
    return false;
  }
  if (info.monochrome) {
    // cfl_ds_filter_index = 0, no bits.
  } else {
    if (!reader.get_bits(2, field)) {
      error = "truncated cfl_ds_filter_index";
      return false;
    }
  }
  if (!read_flag(reader, info.enable_mhccp, "enable_mhccp", error)) return false;
  if (!read_flag(reader, flag, "enable_ibp", error)) return false;

  // §5.4.6 sequence_inter_config(): single-picture forced fields.
  if (!read_flag(reader, flag, "enable_refmvbank", error)) return false;
  bool disable_drl_reorder = false;
  if (!read_flag(reader, disable_drl_reorder, "disable_drl_reorder", error)) {
    return false;
  }
  if (!disable_drl_reorder) {
    if (!read_flag(reader, flag, "constrain_drl_reorder", error)) return false;
  }
  std::uint32_t ns_value = 0;
  if (!reader.get_ns(3, ns_value)) {
    error = "truncated seq_max_bvp_drl_bits_minus_1";
    return false;
  }
  if (!read_flag(reader, flag, "allow_frame_max_bvp_drl_bits", error)) {
    return false;
  }
  if (!read_flag(reader, flag, "enable_bawp", error)) return false;

  // §5.4.7 sequence_scc_config(): forced by single_picture_header_flag.

  // §5.4.8 sequence_transform_quant_entropy_config().
  bool enable_fsc = false;
  if (!read_flag(reader, enable_fsc, "enable_fsc", error)) return false;
  if (!enable_fsc) {
    if (!read_flag(reader, flag, "enable_idtx_intra", error)) return false;
  }
  if (!read_flag(reader, flag, "enable_intra_ist", error)) return false;
  if (!read_flag(reader, flag, "enable_inter_ist", error)) return false;
  if (info.monochrome) {
    info.enable_chroma_dctonly = false;
  } else {
    if (!read_flag(reader, info.enable_chroma_dctonly,
                   "enable_chroma_dctonly", error)) {
      return false;
    }
  }
  // enable_inter_ddt is absent for single_picture_header_flag = 1.
  if (!read_flag(reader, flag, "reduced_tx_part_set", error)) return false;
  if (info.monochrome) {
    info.enable_cctx = false;
  } else {
    if (!read_flag(reader, info.enable_cctx, "enable_cctx", error)) {
      return false;
    }
  }
  bool enable_tcq = false;
  if (!read_flag(reader, enable_tcq, "enable_tcq", error)) return false;
  if (enable_tcq) {
    // choose_tcq_per_frame = 0 for single_picture_header_flag = 1, so
    // enable_parity_hiding is absent and 0.
  } else {
    if (!read_flag(reader, flag, "enable_parity_hiding", error)) return false;
  }
  // enable_avg_cdf = 1, avg_cdf_type = 1 are forced.
  if (info.monochrome) {
    info.separate_uv_delta_q = false;
  } else {
    if (!read_flag(reader, info.separate_uv_delta_q, "separate_uv_delta_q",
                   error)) {
      return false;
    }
  }
  bool equal_ac_dc_q = false;
  if (!read_flag(reader, equal_ac_dc_q, "equal_ac_dc_q", error)) return false;
  if (!equal_ac_dc_q) {
    if (!reader.get_bits(5, field)) {
      error = "truncated base_y_dc_delta_q";
      return false;
    }
    if (!read_flag(reader, flag, "y_dc_delta_q_enabled", error)) return false;
  }
  if (!info.monochrome) {
    if (!equal_ac_dc_q) {
      if (!reader.get_bits(5, field)) {
        error = "truncated base_uv_dc_delta_q";
        return false;
      }
      if (!read_flag(reader, flag, "uv_dc_delta_q_enabled", error)) {
        return false;
      }
    }
    if (!reader.get_bits(5, field)) {
      error = "truncated base_uv_ac_delta_q";
      return false;
    }
    if (!read_flag(reader, flag, "uv_ac_delta_q_enabled", error)) return false;
  }

  // §5.4.10 sequence_filter_config().
  if (!read_flag(reader, flag, "disable_loopfilters_across_tiles", error)) {
    return false;
  }
  if (!read_flag(reader, info.enable_cdef, "enable_cdef", error)) return false;
  bool enable_gdf = false;
  if (!read_flag(reader, enable_gdf, "enable_gdf", error)) return false;
  // gdf_unit_matches_sb_size is present only when the sequence uses 64x64
  // superblocks (§5.4.10).
  if (enable_gdf && !use_256 && !use_128) {
    if (!read_flag(reader, flag, "gdf_unit_matches_sb_size", error)) {
      return false;
    }
  }
  if (!read_flag(reader, info.enable_restoration, "enable_restoration",
                 error)) {
    return false;
  }
  if (info.enable_restoration) {
    if (!read_flag(reader, info.lr_tools_disable_pc, "lr_tools_disable_pc",
                   error)) {
      return false;
    }
    if (!read_flag(reader, info.lr_tools_disable_ns, "lr_tools_disable_ns",
                   error)) {
      return false;
    }
    if (!read_flag(reader, info.lr_tools_uv_present, "lr_tools_uv_present",
                   error)) {
      return false;
    }
    if (info.lr_tools_uv_present) {
      if (!read_flag(reader, info.lr_tools_disable_uv_ns,
                     "lr_tools_disable_uv", error)) {
        return false;
      }
    }
  }
  if (!read_flag(reader, info.enable_ccso, "enable_ccso", error)) return false;
  if (info.enable_ccso) {
    if (!read_flag(reader, flag, "ccso_unit_matches_sb_size", error)) {
      return false;
    }
  }
  // CdefOnSkipTxfm is forced by single_picture_header_flag = 1.
  if (!reader.get_bits(2, field)) {
    error = "truncated df_par_bits_minus_2";
    return false;
  }

  // §5.4.2 sequence_tile_config().
  bool seq_tile_info_present = false;
  if (!read_flag(reader, seq_tile_info_present, "seq_tile_info_present_flag",
                 error)) {
    return false;
  }
  if (seq_tile_info_present) {
    error = "seq_tile_info_present_flag = 1 is not supported by this reader";
    return false;
  }

  // §5.4.1 tail.
  if (!read_flag(reader, info.film_grain_params_present,
                 "film_grain_params_present", error)) {
    return false;
  }

  // OBU payloads continue with §5.2.1 obu_extension_flag and §5.2.3
  // trailing_bits().  A syntax-only buffer ends in zero padding instead.
  if (reader.remaining_bits() >= 2) {
    bool ext = true;
    bool trailing = false;
    reader.peek_bit(0, ext);
    reader.peek_bit(1, trailing);
    if (!ext && trailing) {
      if (!read_flag(reader, ext, "obu_extension_flag", error)) return false;
      if (ext) {
        error = "OBU extension flag is not supported";
        return false;
      }
      if (!read_flag(reader, trailing, "trailing_one_bit", error)) {
        return false;
      }
      if (!trailing) {
        error = "trailing_one_bit must be 1";
        return false;
      }
      if (!reader.all_remaining_zero()) {
        error = "non-zero padding after trailing_bits";
        return false;
      }
    } else if (!reader.all_remaining_zero()) {
      error = "unexpected bits after the sequence header syntax";
      return false;
    }
  } else if (!reader.all_remaining_zero()) {
    error = "unexpected bits after the sequence header syntax";
    return false;
  }

  error.clear();
  return true;
}

bool parse_sequence_header_obu(std::span<const std::uint8_t> obu,
                               SequenceHeaderInfo& info, std::string& error) {
  if (obu.empty()) {
    error = "empty OBU";
    return false;
  }
  std::size_t position = 0;
  std::uint32_t size = 0;
  unsigned shift = 0;
  for (;;) {
    if (position >= obu.size() || shift > 28) {
      error = "truncated or oversized leb128 OBU size";
      return false;
    }
    const std::uint8_t byte = obu[position++];
    size |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
    if ((byte & 0x80U) == 0) {
      break;
    }
    shift += 7;
  }
  if (size == 0 || size > obu.size() - position) {
    error = "OBU size exceeds the available bytes";
    return false;
  }
  const std::span<const std::uint8_t> payload = obu.subspan(position, size);
  const std::uint8_t header = payload[0];
  const bool extension_flag = (header & 0x80U) != 0;
  const unsigned type = (header >> 2U) & 0x1FU;
  if (type != static_cast<unsigned>(ObuType::kSequenceHeader)) {
    error = "OBU is not a sequence header (obu_type " +
            std::to_string(type) + ")";
    return false;
  }
  const std::size_t header_bytes = 1U + (extension_flag ? 1U : 0U);
  if (payload.size() <= header_bytes) {
    error = "sequence header OBU has no payload";
    return false;
  }
  return read_sequence_header_syntax(payload.subspan(header_bytes), info, error);
}

bool find_sequence_header(std::span<const std::uint8_t> temporal_unit,
                          SequenceHeaderInfo& info, std::string& error) {
  std::size_t position = 0;
  while (position < temporal_unit.size()) {
    const std::size_t start = position;
    std::uint32_t size = 0;
    unsigned shift = 0;
    for (;;) {
      if (position >= temporal_unit.size() || shift > 28) {
        error = "truncated leb128 OBU size in temporal unit";
        return false;
      }
      const std::uint8_t byte = temporal_unit[position++];
      size |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
      if ((byte & 0x80U) == 0) {
        break;
      }
      shift += 7;
    }
    if (size == 0 || size > temporal_unit.size() - position) {
      error = "OBU size exceeds the temporal unit";
      return false;
    }
    const std::span<const std::uint8_t> obu =
        temporal_unit.subspan(start, position - start + size);
    const unsigned type = (obu[position - start] >> 2U) & 0x1FU;
    if (type == static_cast<unsigned>(ObuType::kSequenceHeader)) {
      return parse_sequence_header_obu(obu, info, error);
    }
    position += size;
  }
  error = "no OBU_SEQUENCE_HEADER found in the temporal unit";
  return false;
}

std::vector<std::uint8_t> assemble_chroma_still(
    const ChromaSeqConfig& config, std::span<const std::uint8_t> tile_payload) {
  return assemble_chroma_still(config, tile_payload, nullptr);
}

std::vector<std::uint8_t> assemble_chroma_still(
    const ChromaSeqConfig& config, std::span<const std::uint8_t> tile_payload,
    const ChromaFilterHeaderConfig* filters) {
  av2enc::basic::StillPictureConfig basic_config;
  basic_config.max_frame_width = config.max_frame_width;
  basic_config.max_frame_height = config.max_frame_height;
  basic_config.seq_level_idx = config.seq_level_idx;
  basic_config.base_q_idx = config.base_q_idx;
  basic_config.disable_cdf_update = config.disable_cdf_update;
  basic_config.enable_cdf_adaptation = false;

  const std::vector<std::uint8_t> td = av2enc::basic::make_temporal_delimiter_obu();
  const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);
  std::vector<std::uint8_t> clk;
  if (filters == nullptr) {
    clk = av2enc::basic::make_closed_loop_key_obu(basic_config, tile_payload);
  } else {
    clk = make_chroma_filtered_key_obu(config, *filters, tile_payload);
  }

  std::vector<std::uint8_t> out;
  out.reserve(td.size() + seq.size() + clk.size());
  out.insert(out.end(), td.begin(), td.end());
  out.insert(out.end(), seq.begin(), seq.end());
  out.insert(out.end(), clk.begin(), clk.end());
  return out;
}

}  // namespace av2enc::chroma
