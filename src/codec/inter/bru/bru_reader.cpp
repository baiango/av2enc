#include "codec/inter/bru/bru.h"

#include "codec/inter/bru/bru_common.h"
#include "pipeline/still/basic_cdf_tables.h"
#include "pipeline/video/video_cdf_tables.h"
#include "codec/entropy/av2/av2_entropy.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace av2enc::inter::bru {
namespace {

using detail::ceil_log2;

class BitReader {
 public:
  BitReader(std::span<const std::uint8_t> bytes, std::size_t bit_offset = 0)
      : bytes_(bytes), bit_(bit_offset) {}

  bool ok() const { return ok_; }
  std::size_t bit_position() const { return bit_; }
  std::size_t bits_remaining() const {
    const std::size_t total = bytes_.size() * 8U;
    return bit_ <= total ? total - bit_ : 0U;
  }

  bool get_bit() {
    if (!ok_ || bit_ >= bytes_.size() * 8U) {
      ok_ = false;
      return false;
    }
    const std::uint8_t byte = bytes_[bit_ >> 3U];
    const bool value = ((byte >> (7U - (bit_ & 7U))) & 1U) != 0U;
    ++bit_;
    return value;
  }

  std::uint32_t get_bits(unsigned count) {
    if (count > 32U) {
      ok_ = false;
      return 0;
    }
    std::uint32_t value = 0;
    for (unsigned i = 0; i < count; ++i) {
      value = (value << 1U) | (get_bit() ? 1U : 0U);
    }
    return value;
  }

  void align_to_byte() {
    if (!ok_) return;
    const std::size_t rem = bit_ & 7U;
    if (rem != 0U) bit_ += 8U - rem;
    if (bit_ > bytes_.size() * 8U) ok_ = false;
  }

  std::uint32_t get_uvlc() {
    unsigned leading_zeros = 0;
    while (ok_ && !get_bit()) {
      ++leading_zeros;
      if (leading_zeros >= 32U) {
        ok_ = false;
        return 0;
      }
    }
    if (!ok_) return 0;
    const std::uint32_t payload =
        leading_zeros == 0 ? 0U : get_bits(leading_zeros);
    return payload + ((1U << leading_zeros) - 1U);
  }

  std::uint32_t get_le(int bytes) {
    std::uint32_t value = 0;
    for (int i = 0; i < bytes; ++i) {
      value |= static_cast<std::uint32_t>(get_bits(8U)) << (8 * i);
    }
    return value;
  }

  std::span<const std::uint8_t> bytes() const { return bytes_; }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_ = 0;
  bool ok_ = true;
};

bool read_leb128(std::span<const std::uint8_t> bytes, std::size_t& pos,
                 std::uint32_t& value) {
  value = 0;
  int shift = 0;
  while (pos < bytes.size()) {
    const std::uint8_t byte = bytes[pos++];
    value |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
    if ((byte & 0x80U) == 0U) return true;
    shift += 7;
    if (shift > 28) return false;
  }
  return false;
}

int compute_min_log2_cols(const SequenceConfig& c, int sb_cols) {
  return detail::min_log2_tile_cols(c, sb_cols);
}

int compute_min_log2_rows(const SequenceConfig& c, int sb_cols, int sb_rows,
                          int min_cols, int cols_log2) {
  const int min_tiles = detail::min_log2_tiles(c, sb_cols, sb_rows, min_cols);
  return std::max(min_tiles - cols_log2, 0);
}

bool parse_sequence_header(std::span<const std::uint8_t> payload,
                           SequenceConfig& config, std::string& error) {
  BitReader br(payload);
  (void)br.get_uvlc();  // seq_header_id
  (void)br.get_bits(5U);  // seq_profile_idc
  const bool single_picture = br.get_bit();
  if (single_picture) {
    error = "BRU reader: single-picture sequence has no BRU config";
    return false;
  }
  config.seq_level_idx = static_cast<std::uint8_t>(br.get_bits(5U));
  (void)br.get_bit();  // seq_tier
  (void)br.get_uvlc();  // chroma_format_idc
  (void)br.get_uvlc();  // bit_depth_idc

  (void)br.get_bits(3U);  // seq_lcr_id
  (void)br.get_bit();     // still_picture
  (void)br.get_bits(2U);  // max_tlayer_id
  const std::uint32_t max_mlayer_id = br.get_bits(3U);
  if (max_mlayer_id != 0) {
    error = "BRU reader: multi-layer sequence unsupported";
    return false;
  }
  (void)br.get_bit();  // monotonic_output_order_flag

  const unsigned width_bits = br.get_bits(4U) + 1U;
  const unsigned height_bits = br.get_bits(4U) + 1U;
  config.max_frame_width =
      static_cast<std::uint16_t>(br.get_bits(width_bits) + 1U);
  config.max_frame_height =
      static_cast<std::uint16_t>(br.get_bits(height_bits) + 1U);

  (void)br.get_bit();  // seq_cropping_window_present_flag
  (void)br.get_bit();  // seq_initial_display_delay_present_flag
  (void)br.get_bit();  // decoder_model_info_present_flag

  (void)br.get_bit();  // use_256x256_superblock
  (void)br.get_bit();  // use_128x128_superblock
  (void)br.get_bit();  // enable_sdp
  (void)br.get_bit();  // enable_ext_partitions
  (void)br.get_bit();  // reduce_pb_aspect_ratio

  (void)br.get_bit();  // enable_ext_seg
  (void)br.get_bit();  // seq_seg_info_present_flag

  (void)br.get_bit();  // enable_dip
  (void)br.get_bit();  // enable_intra_edge_filter
  (void)br.get_bit();  // enable_mrls
  config.enable_cfl_intra = br.get_bit();
  (void)br.get_bits(2U);  // cfl_ds_filter_index
  (void)br.get_bit();     // enable_mhccp
  (void)br.get_bit();     // enable_ibp

  for (int i = 1; i < 5; ++i) (void)br.get_bit();
  (void)br.get_bit();  // enable_masked_compound
  (void)br.get_bit();  // enable_ref_frame_mvs
  config.order_hint_bits = static_cast<std::uint8_t>(br.get_bits(4U) + 1U);
  (void)br.get_bit();  // enable_refmvbank
  (void)br.get_bit();  // disable_drl_reorder
  (void)br.get_bit();  // explicit_ref_frame_map
  (void)br.get_bit();  // explicit_num_ref_frames
  config.num_ref_frames = static_cast<std::uint8_t>(br.get_bits(4U) + 1U);
  (void)br.get_bits(3U);  // long_term_frame_id_bits

  // ns(5) max_drl_bits_minus_1.  The value only matters to the writer.
  {
    const unsigned w = 3;  // ceil_log2(5)
    const unsigned m = (1U << w) - 5U;
    const std::uint32_t v = br.get_bits(w - 1U);
    config.max_drl_bits_minus_1 =
        static_cast<std::uint8_t>(v < m ? v : (v << 1U) | br.get_bit());
  }
  (void)br.get_bit();  // allow_frame_max_drl_bits
  {
    const unsigned w = 2;  // ceil_log2(3)
    (void)br.get_bits(w - 1U);
  }
  (void)br.get_bit();  // allow_frame_max_bvp_drl_bits
  (void)br.get_bits(2U);  // num_same_ref_compound
  (void)br.get_bit();  // enable_tip
  (void)br.get_bit();  // enable_mv_traj
  (void)br.get_bit();  // enable_bawp
  (void)br.get_bit();  // enable_cwp
  (void)br.get_bit();  // enable_imp_msk_bld
  (void)br.get_bit();  // enable_df_sub_pu
  (void)br.get_bits(2U);  // enable_opfl_refine
  (void)br.get_bit();  // enable_refinemv
  config.enable_bru = br.get_bit();
  (void)br.get_bit();  // enable_adaptive_mvd
  (void)br.get_bit();  // enable_mvd_sign_derive
  (void)br.get_bit();  // enable_flex_mvres
  (void)br.get_bit();  // enable_global_motion
  (void)br.get_bit();  // enable_short_refresh_frame_flags

  (void)br.get_bit();  // seq_choose_screen_content_tools
  (void)br.get_bit();  // seq_force_screen_content_tools

  (void)br.get_bit();  // enable_fsc
  (void)br.get_bit();  // enable_idtx_intra
  (void)br.get_bit();  // enable_intra_ist
  (void)br.get_bit();  // enable_inter_ist
  (void)br.get_bit();  // enable_chroma_dctonly
  (void)br.get_bit();  // enable_inter_ddt
  (void)br.get_bit();  // reduced_tx_part_set
  (void)br.get_bit();  // enable_cctx
  (void)br.get_bit();  // enable_tcq
  (void)br.get_bit();  // enable_parity_hiding
  (void)br.get_bit();  // enable_avg_cdf
  (void)br.get_bit();  // separate_uv_delta_q
  (void)br.get_bit();  // equal_ac_dc_q
  (void)br.get_bits(5U);  // base_uv_ac_delta_q
  (void)br.get_bit();     // uv_ac_delta_q_enabled

  (void)br.get_bit();  // disable_loopfilters_across_tiles
  (void)br.get_bit();  // enable_cdef
  (void)br.get_bit();  // enable_gdf
  (void)br.get_bit();  // enable_restoration
  (void)br.get_bit();  // enable_ccso
  (void)br.get_bit();  // cdef_on_skip_txfm_always_on
  (void)br.get_bit();  // cdef_on_skip_txfm_disabled
  (void)br.get_bits(2U);  // df_par_bits_minus_2
  (void)br.get_bit();     // seq_tile_info_present_flag
  (void)br.get_bit();     // film_grain_params_present

  if (!br.ok()) {
    error = "BRU reader: truncated sequence header";
    return false;
  }
  return true;
}

struct ParsedTileInfo {
  int cols_log2 = 0;
  int rows_log2 = 0;
  int cols = 1;
  int rows = 1;
  int size_bytes = 1;
};

bool parse_tile_info(BitReader& br, const SequenceConfig& c, int sb_cols,
                     int sb_rows, ParsedTileInfo& out, std::string& error) {
  const bool uniform = br.get_bit();
  if (!uniform) {
    error = "BRU reader: non-uniform tiles unsupported";
    return false;
  }
  const int min_cols = compute_min_log2_cols(c, sb_cols);
  const int max_cols = detail::max_log2_tile_cols(sb_cols);
  int cols_log2 = min_cols;
  while (cols_log2 < max_cols) {
    if (!br.get_bit()) break;
    ++cols_log2;
  }
  const int min_rows = compute_min_log2_rows(c, sb_cols, sb_rows, min_cols,
                                             cols_log2);
  const int max_rows = detail::max_log2_tile_rows(sb_rows);
  int rows_log2 = min_rows;
  while (rows_log2 < max_rows) {
    if (!br.get_bit()) break;
    ++rows_log2;
  }
  out.cols_log2 = cols_log2;
  out.rows_log2 = rows_log2;
  out.cols = 1 << cols_log2;
  out.rows = 1 << rows_log2;
  if (out.cols > 1 || out.rows > 1) {
    const int n = cols_log2 + rows_log2;
    (void)br.get_bits(static_cast<unsigned>(n));  // context_update_tile_id
    out.size_bytes = static_cast<int>(br.get_bits(2U)) + 1;
  }
  if (!br.ok()) {
    error = "BRU reader: truncated tile_info";
    return false;
  }
  return true;
}

bool parse_frame_header(BitReader& br, const SequenceConfig& c, bool key_obu,
                        FrameConfig& frame, ParsedTileInfo& tile_info,
                        std::string& error) {
  (void)br.get_uvlc();  // cur_mfh_id
  (void)br.get_uvlc();  // seq_header_id_in_frame_header
  if (key_obu) {
    frame.keyframe = true;
  } else {
    const bool frame_is_inter = br.get_bit();
    frame.keyframe = !frame_is_inter;
  }
  frame.immediate_output_frame = br.get_bit();
  (void)br.get_bit();  // frame_size_override_flag
  frame.order_hint = br.get_bits(c.order_hint_bits);

  if (!frame.keyframe) {
    const bool signal_primary = br.get_bit();
    (void)br.get_bit();  // disable_cross_frame_cdf_init
    if (signal_primary) (void)br.get_bits(3U);
    frame.refresh_frame_flags = br.get_bits(c.num_ref_frames);
  }

  if (frame.keyframe) {
    (void)br.get_bit();  // allow_intrabc
  } else {
    (void)br.get_bit();  // frame_explicit_ref_frame_map (must be 1)
    frame.num_total_refs = static_cast<std::uint8_t>(br.get_bits(3U));
    if (frame.num_total_refs == 0 || frame.num_total_refs > 7) {
      error = "BRU reader: bad NumTotalRefs";
      return false;
    }
    const int n = ceil_log2(c.num_ref_frames);
    for (std::uint8_t i = 0; i < frame.num_total_refs; ++i) {
      frame.ref_frame_idx[i] = static_cast<std::uint8_t>(br.get_bits(n));
    }
    if (c.enable_bru) {
      frame.use_bru = br.get_bit();
      if (frame.use_bru) {
        const int nb = ceil_log2(frame.num_total_refs);
        frame.bru_ref = nb == 0
                            ? 0
                            : static_cast<std::uint8_t>(br.get_bits(nb));
        frame.bru_inactive = br.get_bit();
      }
    }
    if (frame.use_bru && frame.bru_ref >= frame.num_total_refs) {
      error = "BRU reader: bru_ref outside NumTotalRefs";
      return false;
    }
    if (frame.bru_inactive) {
      // §5.18.2 p105-107 early return: no tile_info / quant / filter syntax.
      if (!br.ok()) {
        error = "BRU reader: truncated inactive frame header";
        return false;
      }
      return true;
    }
    (void)br.get_bit();  // allow_intrabc
    (void)br.get_bit();  // use_qtr_precision_mv
    (void)br.get_bit();  // allow_high_precision_mv
    (void)br.get_bit();  // is_filter_switchable
    (void)br.get_bits(2U);  // interpolation_filter
  }

  (void)br.get_bit();  // disable_cdf_update
  if (!parse_tile_info(br, c, sb_cols_for(c.max_frame_width),
                       sb_rows_for(c.max_frame_height), tile_info, error)) {
    return false;
  }
  (void)br.get_bits(8U);  // base_q_idx
  (void)br.get_bit();     // segmentation_enabled
  (void)br.get_bit();     // using_qmatrix
  (void)br.get_bit();     // delta_q_present
  (void)br.get_bit();     // apply_deblocking_filter[0]
  (void)br.get_bit();     // apply_deblocking_filter[1]
  (void)br.get_bit();     // tx_mode_select
  if (!frame.keyframe) {
    (void)br.get_bit();  // reference_select
    (void)br.get_bit();  // skip_mode_present
  }
  (void)br.get_bits(2U);  // reduced_tx_set
  if (!br.ok()) {
    error = "BRU reader: truncated frame header";
    return false;
  }
  return true;
}

int neighbor_mode(const std::vector<BruMode>& modes, int cols, int rows,
                  int r, int c, int dr, int dc, BruMode& out) {
  const int nr = r + dr;
  const int nc = c + dc;
  if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) return 0;
  out = modes[static_cast<std::size_t>(nr) * cols + nc];
  return 1;
}

bool decode_tile_modes(std::span<const std::uint8_t> payload, int cols,
                       int rows, bool tile_active,
                       std::vector<BruMode>& modes, std::string& error) {
  modes.assign(static_cast<std::size_t>(cols) * rows, BruMode::kInactive);
  av2::Decoder decoder;
  decoder.disable_cdf_update = 1;
  av2::Status status = decoder.init_symbol(payload.data(), payload.size(),
                                           payload.size());
  if (status != av2::Status::kOk) {
    error = std::string("BRU reader: entropy init failed: ") +
            av2::status_name(status);
    return false;
  }
  std::uint16_t bru_cdf[4] = {4124, 16615, 0, 0};
  std::uint16_t do_split_cdf[2][64][3];
  std::memcpy(do_split_cdf, av2enc::basic::kDoSplitCdf, sizeof(do_split_cdf));
  std::uint16_t is_inter_cdf[4][3];
  std::memcpy(is_inter_cdf, av2enc::video::kIsInterCdf, sizeof(is_inter_cdf));
  std::uint16_t skip_cdf[6][3];
  std::memcpy(skip_cdf, av2enc::video::kSkipCdf, sizeof(skip_cdf));
  std::uint16_t single_ref_cdf[3][6][3];
  std::memcpy(single_ref_cdf, av2enc::video::kSingleRefCdf, sizeof(single_ref_cdf));
  std::uint16_t single_mode_cdf[5][4];
  std::memcpy(single_mode_cdf, av2enc::video::kSingleModeCdf, sizeof(single_mode_cdf));

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (!tile_active) continue;
      int symbol = 0;
      status = decoder.decode_symbol(bru_cdf, 3, &symbol);
      if (status != av2::Status::kOk || symbol < 0 || symbol >= 3) {
        error = "BRU reader: bru_mode decode failed";
        return false;
      }
      const BruMode mode = static_cast<BruMode>(symbol);
      modes[static_cast<std::size_t>(r) * cols + c] = mode;
      if (mode != BruMode::kActive) continue;

      int split_symbol = 0;
      status = decoder.decode_symbol(do_split_cdf[0][12], 2, &split_symbol);
      if (status != av2::Status::kOk || split_symbol != 0) {
        error = "BRU reader: do_split decode failed";
        return false;
      }
      BruMode n = BruMode::kInactive;
      const bool has_above =
          neighbor_mode(modes, cols, rows, r, c, -1, 0, n) != 0;
      const bool has_left =
          neighbor_mode(modes, cols, rows, r, c, 0, -1, n) != 0;
      const int skip_ctx = static_cast<int>(has_above) +
                           static_cast<int>(has_left);
      symbol = 0;
      status = decoder.decode_symbol(is_inter_cdf[0], 2, &symbol);
      if (status != av2::Status::kOk || symbol != 1) {
        error = "BRU reader: is_inter decode failed";
        return false;
      }
      symbol = 0;
      status = decoder.decode_symbol(skip_cdf[skip_ctx], 2, &symbol);
      if (status != av2::Status::kOk || symbol != 1) {
        error = "BRU reader: skip_flag decode failed";
        return false;
      }

      int ref_zero = 0;
      int ref_one = 0;
      const auto count = [&](int nr, int nc) {
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) return;
        const BruMode m = modes[static_cast<std::size_t>(nr) * cols + nc];
        if (m == BruMode::kActive) {
          ++ref_one;
        } else {
          ++ref_zero;
        }
      };
      count(r, c - 1);
      count(r - 1, c);
      const int ref_ctx = (ref_zero == ref_one)
                              ? 1
                              : (ref_zero < ref_one ? 0 : 2);
      symbol = 0;
      status = decoder.decode_symbol(single_ref_cdf[ref_ctx][0], 2, &symbol);
      if (status != av2::Status::kOk || symbol != 0) {
        error = "BRU reader: single_ref decode failed";
        return false;
      }

      symbol = 0;
      status = decoder.decode_symbol(single_mode_cdf[skip_ctx], 3, &symbol);
      if (status != av2::Status::kOk || symbol != 1) {
        error = "BRU reader: single_mode decode failed";
        return false;
      }
    }
  }
  status = decoder.exit_symbol();
  if (status != av2::Status::kOk) {
    error = std::string("BRU reader: entropy exit failed: ") +
            av2::status_name(status);
    return false;
  }
  return true;
}

bool parse_tile_group(std::span<const std::uint8_t> payload, bool key_obu,
                      const SequenceConfig& sequence, ParsedFrame& out,
                      std::string& error) {
  BitReader br(payload);
  const bool first = br.get_bit();
  if (!first) {
    error = "BRU reader: non-first tile group unsupported";
    return false;
  }
  const bool frame_header_present = true;
  (void)frame_header_present;
  ParsedTileInfo tile_info;
  if (!parse_frame_header(br, sequence, key_obu, out.header, tile_info,
                          error)) {
    return false;
  }
  out.sb_cols = sb_cols_for(sequence.max_frame_width);
  out.sb_rows = sb_rows_for(sequence.max_frame_height);
  out.tile_cols = tile_info.cols;
  out.tile_rows = tile_info.rows;
  out.sb_modes.assign(static_cast<std::size_t>(out.sb_cols) * out.sb_rows,
                      BruMode::kInactive);

  if (out.header.bru_inactive) {
    out.whole_frame_inactive = true;
    out.modes_reconstructed = true;
    out.region_valid = true;
    return true;
  }

  const int num_tiles = tile_info.cols * tile_info.rows;
  bool all_tiles = true;
  int tg_start = 0;
  int tg_end = num_tiles - 1;
  if (num_tiles > 1) {
    const bool present = br.get_bit();
    if (present) {
      const unsigned tile_bits =
          static_cast<unsigned>(tile_info.cols_log2 + tile_info.rows_log2);
      tg_start = static_cast<int>(br.get_bits(tile_bits));
      tg_end = static_cast<int>(br.get_bits(tile_bits));
      all_tiles = false;
    }
  }
  out.bru_tile_active.assign(static_cast<std::size_t>(num_tiles), 0);
  if (out.header.use_bru) {
    if (num_tiles > 1) {
      for (int i = 0; i < num_tiles; ++i) {
        out.bru_tile_active[static_cast<std::size_t>(i)] =
            br.get_bit() ? 1U : 0U;
      }
    } else {
      out.bru_tile_active[0] = 1;
    }
  }
  br.align_to_byte();
  if (!br.ok()) {
    error = "BRU reader: truncated tile group header";
    return false;
  }

  const std::size_t payload_start_bit = br.bit_position();
  const std::size_t payload_start = payload_start_bit / 8U;
  std::size_t cursor = payload_start;
  const detail::UniformSpacing col_spacing = detail::uniform_spacing(
      tile_info.cols_log2, mi_cols(sequence.max_frame_width), 64);
  const detail::UniformSpacing row_spacing = detail::uniform_spacing(
      tile_info.rows_log2, mi_rows(sequence.max_frame_height), 64);

  for (int tile = tg_start; tile <= tg_end; ++tile) {
    std::size_t tile_size = 0;
    if (tile == tg_end) {
      tile_size = payload.size() - cursor;
    } else {
      if (cursor + static_cast<std::size_t>(tile_info.size_bytes) >
          payload.size()) {
        error = "BRU reader: truncated tile size";
        return false;
      }
      std::uint32_t size_minus_1 = 0;
      for (int i = 0; i < tile_info.size_bytes; ++i) {
        size_minus_1 |=
            static_cast<std::uint32_t>(payload[cursor++]) << (8 * i);
      }
      tile_size = static_cast<std::size_t>(size_minus_1) + 1U;
    }
    if (cursor + tile_size > payload.size()) {
      error = "BRU reader: tile payload runs past the OBU";
      return false;
    }
    const auto tile_payload =
        payload.subspan(cursor, tile_size);
    cursor += tile_size;

    if (!out.header.use_bru) continue;

    const int tile_row = tile / tile_info.cols;
    const int tile_col = tile % tile_info.cols;
    const int sb_col_start = col_spacing.starts[static_cast<std::size_t>(tile_col)];
    const int sb_row_start = row_spacing.starts[static_cast<std::size_t>(tile_row)];
    const int sb_col_end =
        tile_col + 1 < tile_info.cols
            ? col_spacing.starts[static_cast<std::size_t>(tile_col + 1)]
            : out.sb_cols;
    const int sb_row_end =
        tile_row + 1 < tile_info.rows
            ? row_spacing.starts[static_cast<std::size_t>(tile_row + 1)]
            : out.sb_rows;
    const int tile_sb_cols = sb_col_end - sb_col_start;
    const int tile_sb_rows = sb_row_end - sb_row_start;
    const bool active = out.bru_tile_active[static_cast<std::size_t>(tile)] != 0U;
    std::vector<BruMode> tile_modes;
    if (!decode_tile_modes(tile_payload, tile_sb_cols, tile_sb_rows, active,
                           tile_modes, error)) {
      return false;
    }
    for (int r = 0; r < tile_sb_rows; ++r) {
      for (int c = 0; c < tile_sb_cols; ++c) {
        out.sb_modes[static_cast<std::size_t>(sb_row_start + r) *
                         out.sb_cols +
                     (sb_col_start + c)] =
            tile_modes[static_cast<std::size_t>(r) * tile_sb_cols + c];
      }
    }
  }
  out.modes_reconstructed = true;
  if (all_tiles && tg_end == num_tiles - 1 && out.header.use_bru) {
    out.region_valid = bru_region_valid(
        out.sb_cols, out.sb_rows, std::span<const BruMode>(out.sb_modes));
  }
  return true;
}

}  // namespace

bool parse_stream(std::span<const std::uint8_t> bytes, ParsedStream& out,
                  std::string& error) {
  out = ParsedStream{};
  SequenceConfig sequence;
  bool have_sequence = false;
  std::size_t pos = 0;
  while (pos < bytes.size()) {
    std::uint32_t obu_size = 0;
    if (!read_leb128(bytes, pos, obu_size)) {
      error = "BRU reader: bad OBU length";
      return false;
    }
    if (obu_size == 0 || pos + obu_size > bytes.size()) {
      error = "BRU reader: OBU length out of range";
      return false;
    }
    const std::size_t obu_start = pos;
    const std::size_t obu_end = pos + obu_size;
    std::uint8_t header = bytes[obu_start];
    pos = obu_start + 1;
    const bool extension = (header & 0x80U) != 0U;
    const std::uint8_t obu_type = (header >> 2U) & 0x1FU;
    if (extension) {
      if (pos >= obu_end) {
        error = "BRU reader: truncated OBU extension";
        return false;
      }
      ++pos;
    }
    const auto payload = bytes.subspan(pos, obu_end - pos);
    pos = obu_end;

    if (obu_type == 1U) {  // OBU_SEQUENCE_HEADER
      if (!parse_sequence_header(payload, sequence, error)) return false;
      have_sequence = true;
      out.sequence = sequence;
    } else if (obu_type == 4U || obu_type == 7U) {
      if (!have_sequence) {
        error = "BRU reader: frame precedes sequence header";
        return false;
      }
      ParsedFrame frame;
      if (!parse_tile_group(payload, obu_type == 4U, sequence, frame, error)) {
        return false;
      }
      out.frames.push_back(std::move(frame));
    } else if (obu_type == 2U) {
      // Temporal delimiter: empty payload.
    } else {
      error = "BRU reader: unsupported OBU type " +
              std::to_string(obu_type);
      return false;
    }
  }
  if (!have_sequence) {
    error = "BRU reader: no sequence header";
    return false;
  }
  return true;
}

}  // namespace av2enc::inter::bru
