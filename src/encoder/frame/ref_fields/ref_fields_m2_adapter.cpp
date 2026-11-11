#include "encoder/frame/ref_fields/ref_fields_m2_adapter.h"

#include "bitstream/m0_headers.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::frame::ref_fields {
namespace {

using av2enc::m0::ObuHeader;
using av2enc::m0::ObuType;

bool fail(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) ++leading_zeros;
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

bool put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value,
            std::string* error) {
  if (n < 2U || value >= n) {
    return fail(error, "ns(): value out of range");
  }
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
  return true;
}

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

std::uint32_t mi_cols(const M2RefConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
}
std::uint32_t mi_rows(const M2RefConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
}

unsigned size_bits(std::uint32_t maximum) {
  unsigned bits = bit_width(maximum - 1U);
  return bits == 0 ? 1U : bits;
}

std::size_t write_sequence_header(BitWriter& writer, const M2RefConfig& c,
                                  std::string* error) {
  const std::size_t start = writer.bit_size();
  if (c.order_hint_bits < 1U || c.order_hint_bits > 16U) {
    fail(error, "order_hint_bits must be in [1, 16]");
    return 0;
  }
  if (c.num_ref_frames < 1U || c.num_ref_frames > 16U) {
    fail(error, "num_ref_frames must be in [1, 16]");
    return 0;
  }
  if (c.seq_max_drl_bits_minus_1 >= 5U) {
    fail(error, "seq_max_drl_bits_minus_1 must be < MAX_REF_MV_STACK_SIZE - 1");
    return 0;
  }
  if (c.seq_max_bvp_drl_bits_minus_1 >= 3U) {
    fail(error,
         "seq_max_bvp_drl_bits_minus_1 must be < MAX_REF_BV_STACK_SIZE - 1");
    return 0;
  }

  put_uvlc(writer, 0);                  // seq_header_id
  writer.put_bits(0U, 5U);              // seq_profile_idc
  writer.put_bit(false);                // single_picture_header_flag = 0
  writer.put_bits(c.seq_level_idx, 5U);  // seq_level_idx
  writer.put_bit(false);                // seq_tier
  put_uvlc(writer, 0);                  // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                  // bit_depth_idc = 1 => 8-bit

  writer.put_bits(0U, 3U);  // seq_lcr_id
  writer.put_bit(false);    // still_picture
  writer.put_bits(0U, 2U);  // max_tlayer_id
  writer.put_bits(0U, 3U);  // max_mlayer_id
  writer.put_bit(true);     // monotonic_output_order_flag

  const unsigned width_bits =
      size_bits(static_cast<std::uint32_t>(c.max_frame_width));
  const unsigned height_bits =
      size_bits(static_cast<std::uint32_t>(c.max_frame_height));
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_width) - 1U,
                  width_bits);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_height) - 1U,
                  height_bits);

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
  writer.put_bit(c.enable_cfl_intra);  // enable_cfl_intra
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);  // enable_mhccp
  writer.put_bit(false);  // enable_ibp

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
  if (!put_ns(writer, 5U, c.seq_max_drl_bits_minus_1, error)) return 0;
  writer.put_bit(c.allow_frame_max_drl_bits);
  if (!put_ns(writer, 3U, c.seq_max_bvp_drl_bits_minus_1, error)) return 0;
  writer.put_bit(c.allow_frame_max_bvp_drl_bits);
  writer.put_bits(0U, 2U);  // num_same_ref_compound
  writer.put_bit(false);    // enable_tip
  writer.put_bit(false);    // enable_mv_traj
  writer.put_bit(false);    // enable_bawp
  writer.put_bit(false);    // enable_cwp
  writer.put_bit(false);    // enable_imp_msk_bld
  writer.put_bit(false);    // enable_df_sub_pu
  writer.put_bits(0U, 2U);  // enable_opfl_refine = REFINE_NONE
  writer.put_bit(false);    // enable_refinemv
  writer.put_bit(false);    // enable_bru
  writer.put_bit(false);    // enable_adaptive_mvd
  writer.put_bit(false);    // enable_mvd_sign_derive
  writer.put_bit(false);    // enable_flex_mvres
  writer.put_bit(false);    // enable_global_motion
  writer.put_bit(c.enable_short_refresh_frame_flags);

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
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q => BaseUVAcDeltaQ = 0
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  writer.put_bit(false);    // disable_loopfilters_across_tiles
  writer.put_bit(false);    // enable_cdef
  writer.put_bit(false);    // enable_gdf
  writer.put_bit(false);    // enable_restoration
  writer.put_bit(false);    // enable_ccso
  writer.put_bit(false);    // cdef_on_skip_txfm_always_on
  writer.put_bit(true);     // cdef_on_skip_txfm_disabled
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2

  writer.put_bit(false);    // seq_tile_info_present_flag
  writer.put_bit(false);    // film_grain_params_present

  return writer.bit_size() - start;
}

bool write_tile_info(BitWriter& writer, const M2RefConfig& config,
                     const M2TileInfo& tiles, std::string* error) {
  // Mirror of m2::post::write_m2_tile_info().
  const std::uint32_t sb_cols = (mi_cols(config) + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows(config) + 15U) >> 4U;
  if (sb_cols == 1U && tiles.cols_log2 != 0) {
    return fail(error, "tile_info: one superblock column needs cols_log2 = 0");
  }
  if (sb_rows == 1U && tiles.rows_log2 != 0) {
    return fail(error, "tile_info: one superblock row needs rows_log2 = 0");
  }
  writer.put_bit(true);  // uniform_tile_spacing_flag

  const int cols_log2 = tiles.cols_log2;
  const int rows_log2 = tiles.rows_log2;
  const int min_cols_log2 = tiles.min_cols_log2;
  const int max_cols_log2 = tiles.max_cols_log2;
  const int min_tiles_log2 = tiles.min_tiles_log2;
  const int max_rows_log2 = tiles.max_rows_log2;
  const int min_rows_log2 = std::max(min_tiles_log2 - cols_log2, 0);
  if (min_cols_log2 < 0 || cols_log2 < min_cols_log2 ||
      cols_log2 > max_cols_log2) {
    return fail(error, "tile_info: tile column log2 outside the level bounds");
  }
  for (int l = min_cols_log2; l < cols_log2; ++l) writer.put_bit(true);
  if (cols_log2 < max_cols_log2) writer.put_bit(false);

  if (min_rows_log2 < 0 || rows_log2 < min_rows_log2 ||
      rows_log2 > max_rows_log2) {
    return fail(error, "tile_info: tile row log2 outside the level bounds");
  }
  for (int l = min_rows_log2; l < rows_log2; ++l) writer.put_bit(true);
  if (rows_log2 < max_rows_log2) writer.put_bit(false);

  if (tiles.tile_count > 1) {
    if (tiles.tile_size_bytes < 1 || tiles.tile_size_bytes > 4) {
      return fail(error, "tile_info: TileSizeBytes must be in 1..4");
    }
    const unsigned n = static_cast<unsigned>(cols_log2 + rows_log2);
    writer.put_bits(0U, n);  // context_update_tile_id
    writer.put_bits(static_cast<std::uint64_t>(tiles.tile_size_bytes - 1), 2U);
  }
  return true;
}

bool read_tile_info(RefBitReader& reader, const M2RefConfig& config,
                    const M2TileInfo& tiles, std::string* error) {
  bool uniform = false;
  if (!reader.bit(&uniform) || !uniform) {
    return fail(error, "tile_info: uniform_tile_spacing_flag must be 1");
  }
  const int min_rows_log2 = std::max(tiles.min_tiles_log2 - tiles.cols_log2, 0);
  for (int l = tiles.min_cols_log2; l < tiles.cols_log2; ++l) {
    bool inc = false;
    if (!reader.bit(&inc) || !inc) {
      return fail(error, "tile_info: expected an increment_tile_cols_log2 bit");
    }
  }
  if (tiles.cols_log2 < tiles.max_cols_log2) {
    bool inc = true;
    if (!reader.bit(&inc) || inc) {
      return fail(error, "tile_info: expected the terminating cols bit");
    }
  }
  for (int l = min_rows_log2; l < tiles.rows_log2; ++l) {
    bool inc = false;
    if (!reader.bit(&inc) || !inc) {
      return fail(error, "tile_info: expected an increment_tile_rows_log2 bit");
    }
  }
  if (tiles.rows_log2 < tiles.max_rows_log2) {
    bool inc = true;
    if (!reader.bit(&inc) || inc) {
      return fail(error, "tile_info: expected the terminating rows bit");
    }
  }
  if (tiles.tile_count > 1) {
    const unsigned n = static_cast<unsigned>(tiles.cols_log2 + tiles.rows_log2);
    std::uint32_t context = 0;
    std::uint32_t size_minus_1 = 0;
    if (!reader.bits(n, &context) || !reader.bits(2U, &size_minus_1)) {
      return fail(error, "tile_info: truncated tile tail");
    }
    if (context != 0) {
      return fail(error,
                  "tile_info: context_update_tile_id must be 0 for these "
                  "streams");
    }
    if (static_cast<int>(size_minus_1) + 1 != tiles.tile_size_bytes) {
      return fail(error, "tile_info: TileSizeBytes mismatch");
    }
  }
  (void)config;
  return true;
}

}  // namespace

M2TileInfo make_m2_tile_info(const tiles::TileLayout& layout,
                             int tile_size_bytes) {
  M2TileInfo info;
  info.cols_log2 = layout.cols_log2;
  info.rows_log2 = layout.rows_log2;
  info.min_cols_log2 = layout.min_log2_tile_cols;
  info.max_cols_log2 = layout.max_log2_tile_cols;
  info.min_tiles_log2 = layout.min_log2_tiles;
  info.max_rows_log2 = layout.max_log2_tile_rows;
  info.tile_count = layout.tile_count();
  info.tile_size_bytes = tile_size_bytes;
  return info;
}

std::vector<std::uint8_t> make_m2_ref_sequence_header_obu(
    const M2RefConfig& config, std::string* error) {
  if (error != nullptr) error->clear();
  BitWriter writer;
  if (write_sequence_header(writer, config, error) == 0) return {};
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::m0::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

std::size_t write_m2_ref_frame_header(BitWriter& writer,
                                      const M2RefConfig& config,
                                      const M2RefFrameHeader& header,
                                      const RefList& refs,
                                      const FrameRefState& state,
                                      const M2TileInfo& tiles,
                                      std::size_t* header_bits,
                                      std::string* error) {
  if (error != nullptr) error->clear();
  if (header_bits != nullptr) *header_bits = 0;
  if (config.base_q_idx == 0) {
    fail(error, "M2 adapter does not support lossless q = 0");
    return 0;
  }
  if (config.num_ref_frames != state.num_ref_frames) {
    fail(error, "config.num_ref_frames disagrees with FrameRefState");
    return 0;
  }
  const bool key = header.keyframe;
  const FrameType frame_type = key ? FrameType::kKey : FrameType::kInter;
  const std::uint32_t frame_width_bits =
      size_bits(static_cast<std::uint32_t>(config.max_frame_width));
  const std::uint32_t frame_height_bits =
      size_bits(static_cast<std::uint32_t>(config.max_frame_height));

  // Validate before writing anything so a rejected header leaves no bits.
  PrimaryRefSignal primary;
  primary.signal_primary_ref_frame = header.signal_primary_ref_frame;
  primary.primary_ref_frame = header.primary_ref_frame;
  primary.disable_cross_frame_cdf_init =
      header.disable_cross_frame_cdf_init;
  PrimaryRefResolved primary_resolved;
  if (!resolve_primary_ref(frame_type, primary, refs, state,
                           config.base_q_idx, &primary_resolved, error)) {
    return 0;
  }
  if (!key && refs.num_total_refs != config.num_total_refs) {
    fail(error, "config.num_total_refs disagrees with RefList");
    return 0;
  }
  if (!key && refs.num_total_refs > 0) {
    for (std::uint32_t i = 0; i < refs.num_total_refs; ++i) {
      if (refs.index[i] >= kNumRefFrames) {
        fail(error, "ref_frame_idx is out of range");
        return 0;
      }
    }
  }

  const std::size_t start = writer.bit_size();
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  if (!key) writer.put_bit(header.frame_is_inter);
  writer.put_bit(true);  // immediate_output_frame

  writer.put_bit(header.frame_size_override);
  writer.put_bits(header.order_hint, config.order_hint_bits);

  if (!write_primary_ref_signal(writer, frame_type, /*is_tip=*/false, primary,
                                error)) {
    return 0;
  }

  // §5.18.2 refresh block (ordered before the size/reference-map block).
  RefreshConfig refresh_config;
  refresh_config.enable_short_refresh_frame_flags =
      config.enable_short_refresh_frame_flags;
  refresh_config.num_ref_frames = config.num_ref_frames;
  refresh_config.obu_is_closed_loop_key = key;
  refresh_config.max_mlayer_id_zero = true;
  RefreshSignal refresh;
  refresh.refresh_frame_flags = header.refresh_frame_flags;
  if (!write_refresh_frame_flags(writer, frame_type, refresh_config, refresh,
                                 /*immediate_output_frame=*/true, error)) {
    return 0;
  }

  if (key) {
    if (header.frame_size_override) {
      if (header.frame_width == 0 || header.frame_height == 0) {
        fail(error, "explicit key-frame size must be non-zero");
        return 0;
      }
      if (header.frame_width > config.max_frame_width ||
          header.frame_height > config.max_frame_height) {
        fail(error, "explicit key-frame size exceeds the sequence maximum");
        return 0;
      }
      writer.put_bits(header.frame_width - 1U, frame_width_bits);
      writer.put_bits(header.frame_height - 1U, frame_height_bits);
    }
    writer.put_bit(header.allow_intrabc);
    if (header.allow_intrabc) {
      writer.put_bit(header.allow_global_intrabc);
      if (header.allow_global_intrabc) {
        writer.put_bit(header.allow_local_intrabc);
      }
      BvDrlConfig bvp_config;
      bvp_config.allow_frame_max_bvp_drl_bits =
          config.allow_frame_max_bvp_drl_bits;
      bvp_config.seq_max_bvp_drl_bits_minus_1 =
          config.seq_max_bvp_drl_bits_minus_1;
      BvDrlSignal bvp;
      bvp.change_bvp_drl = config.allow_frame_max_bvp_drl_bits &&
                           header.change_bvp_drl;
      bvp.max_bvp_drl_bits_minus_1 =
          config.allow_frame_max_bvp_drl_bits
              ? header.max_bvp_drl_bits_minus_1
              : config.seq_max_bvp_drl_bits_minus_1;
      if (!write_bvp_drl(writer, bvp_config, bvp, error)) return 0;
    }
  } else {
    writer.put_bit(header.explicit_ref_frame_map);
    if (header.explicit_ref_frame_map) {
      writer.put_bits(config.num_total_refs, 3U);
      const unsigned index_bits = ceil_log2(config.num_ref_frames);
      for (std::uint32_t i = 0; i < config.num_total_refs; ++i) {
        writer.put_bits(refs.index[i], index_bits);
      }
    } else {
      fail(error,
           "the implicit get_ref_frames() map is out of scope for this "
           "adapter");
      return 0;
    }

    if (header.frame_size_override) {
      SizeConfig size_config;
      size_config.num_total_refs = refs.num_total_refs;
      size_config.max_frame_width = state.max_frame_width;
      size_config.max_frame_height = state.max_frame_height;
      size_config.frame_width_bits = frame_width_bits;
      size_config.frame_height_bits = frame_height_bits;
      FrameSizeSignal size_signal;
      size_signal.width = header.frame_width;
      size_signal.height = header.frame_height;
      size_signal.prefer_found_ref = header.prefer_found_ref;
      FrameSizeResolved resolved;
      if (!write_frame_size_with_refs(writer, size_config, refs, state,
                                      size_signal, &resolved, error)) {
        return 0;
      }
    } else if (state.max_frame_width == 0 || state.max_frame_height == 0 ||
               header.frame_width != state.max_frame_width ||
               header.frame_height != state.max_frame_height) {
      fail(error,
           "frame_size_override_flag = 0 inherits the sequence/MFH size; the "
           "requested size must match it");
      return 0;
    }

    writer.put_bit(header.allow_intrabc);
    if (header.allow_intrabc) {
      // Inter frames infer allow_global_intrabc = 0, allow_local_intrabc = 1.
      BvDrlConfig bvp_config;
      bvp_config.allow_frame_max_bvp_drl_bits =
          config.allow_frame_max_bvp_drl_bits;
      bvp_config.seq_max_bvp_drl_bits_minus_1 =
          config.seq_max_bvp_drl_bits_minus_1;
      BvDrlSignal bvp;
      bvp.change_bvp_drl = config.allow_frame_max_bvp_drl_bits &&
                           header.change_bvp_drl;
      bvp.max_bvp_drl_bits_minus_1 =
          config.allow_frame_max_bvp_drl_bits
              ? header.max_bvp_drl_bits_minus_1
              : config.seq_max_bvp_drl_bits_minus_1;
      if (!write_bvp_drl(writer, bvp_config, bvp, error)) return 0;
    }

    DrlConfig drl_config;
    drl_config.allow_frame_max_drl_bits = config.allow_frame_max_drl_bits;
    drl_config.seq_max_drl_bits_minus_1 = config.seq_max_drl_bits_minus_1;
    DrlSignal drl;
    // The header fields only matter when the sequence header allows a change;
    // otherwise §5.18.2 infers the sequence value and codes no bits.
    drl.change_drl = config.allow_frame_max_drl_bits && header.change_drl;
    drl.max_drl_bits_minus_1 = config.allow_frame_max_drl_bits
                                   ? header.max_drl_bits_minus_1
                                   : config.seq_max_drl_bits_minus_1;
    if (!write_drl(writer, drl_config, drl, error)) return 0;

    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv -> 1/8 pel
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTTAP
  }

  writer.put_bit(config.disable_cdf_update);

  if (!write_tile_info(writer, config, tiles, error)) return 0;

  writer.put_bits(config.base_q_idx, 8U);  // base_q_idx
  writer.put_bit(false);                   // segmentation_enabled
  writer.put_bit(false);                   // using_qmatrix
  writer.put_bit(false);                   // delta_q_present
  writer.put_bit(false);                   // apply_deblocking_filter[0]
  writer.put_bit(false);                   // apply_deblocking_filter[1]
  writer.put_bit(false);                   // tx_mode_select = 0
  if (!key) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set

  if (header_bits != nullptr) *header_bits = writer.bit_size() - start;
  return writer.bit_size() - start;
}

namespace {

// Writes the tile-group header for a tile subset [tg_start, tg_end].
bool write_tile_group_header_range(BitWriter& writer,
                                   const M2RefConfig& config,
                                   const M2RefFrameHeader& header,
                                   const RefList& refs,
                                   const FrameRefState& state,
                                   const M2TileInfo& tiles,
                                   int tg_start, int tg_end,
                                   FrameHeaderStore* store,
                                   std::string* error) {
  const std::size_t start = writer.bit_size();
  const bool first = tg_start == 0 && !store->present();
  writer.put_bit(first);
  std::size_t bits = 0;
  if (first) {
    if (write_m2_ref_frame_header(writer, config, header, refs, state, tiles,
                                  &bits, error) == 0) {
      return false;
    }
    store->record(writer.bytes(), start + 1, writer.bit_size());
  } else {
    if (!write_nonfirst_tile_group_header(writer, *store, true, error)) {
      return false;
    }
  }
  if (tiles.tile_count > 1) {
    writer.put_bit(true);  // tile_start_and_end_present_flag
    const unsigned tile_bits =
        static_cast<unsigned>(tiles.cols_log2 + tiles.rows_log2);
    writer.put_bits(static_cast<std::uint64_t>(tg_start), tile_bits);
    writer.put_bits(static_cast<std::uint64_t>(tg_end), tile_bits);
  } else if (tg_start != 0 || tg_end != 0) {
    fail(error, "single-tile frames cannot split a tile group");
    return false;
  }
  writer.align_to_byte();
  return true;
}

}  // namespace

std::vector<std::vector<std::uint8_t>> make_m2_ref_frame_obus(
    const M2RefConfig& config, const M2RefFrameHeader& header,
    const RefList& refs, const FrameRefState& state, const M2TileInfo& tiles,
    std::span<const std::vector<std::uint8_t>> tile_payloads,
    bool split_tile_groups, std::string* error) {
  if (error != nullptr) error->clear();
  if (tile_payloads.empty()) {
    fail(error, "frame OBU needs at least one tile payload");
    return {};
  }
  if (static_cast<int>(tile_payloads.size()) != tiles.tile_count) {
    fail(error, "tile payload count disagrees with tile_info");
    return {};
  }
  const ObuType type =
      header.keyframe ? ObuType::kClosedLoopKey : ObuType::kRegularTileGroup;
  std::vector<std::vector<std::uint8_t>> obus;

  if (!split_tile_groups) {
    // One tile group covering every tile (tile_start_and_end_present_flag=0).
    BitWriter writer;
    writer.put_bit(true);  // is_first_tile_group
    std::size_t bits = 0;
    if (write_m2_ref_frame_header(writer, config, header, refs, state, tiles,
                                  &bits, error) == 0) {
      return {};
    }
    if (tiles.tile_count > 1) writer.put_bit(false);
    writer.align_to_byte();
    std::vector<std::uint8_t> payload = writer.take_bytes();
    for (int tile = 0; tile < tiles.tile_count; ++tile) {
      if (tile + 1 < tiles.tile_count) {
        const std::uint32_t value = static_cast<std::uint32_t>(
                                        tile_payloads[static_cast<std::size_t>(tile)].size()) -
                                    1U;
        for (int b = 0; b < tiles.tile_size_bytes; ++b) {
          payload.push_back(
              static_cast<std::uint8_t>((value >> (8 * b)) & 0xFFU));
        }
      }
      const auto& bytes = tile_payloads[static_cast<std::size_t>(tile)];
      payload.insert(payload.end(), bytes.begin(), bytes.end());
    }
    obus.push_back(av2enc::m0::make_obu(ObuHeader{type}, payload));
    return obus;
  }

  // One OBU per tile group.  The first group codes the header; later groups
  // repeat it bit for bit through frame_header_copy().
  FrameHeaderStore store;
  for (int tile = 0; tile < tiles.tile_count; ++tile) {
    BitWriter writer;
    if (!write_tile_group_header_range(writer, config, header, refs, state,
                                       tiles, tile, tile, &store, error)) {
      return {};
    }
    std::vector<std::uint8_t> payload = writer.take_bytes();
    // A single tile is the last tile of its own group, so
    // tile_group_payload() codes no tile_size_minus_1 field.
    const auto& bytes = tile_payloads[static_cast<std::size_t>(tile)];
    payload.insert(payload.end(), bytes.begin(), bytes.end());
    obus.push_back(av2enc::m0::make_obu(ObuHeader{type}, payload));
  }
  return obus;
}

bool read_m2_ref_frame_header(RefBitReader& reader, bool keyframe,
                              const M2RefConfig& config, const RefList& refs,
                              const FrameRefState& state,
                              const M2TileInfo& tiles,
                              M2RefParsedHeader* out, std::string* error) {
  if (error != nullptr) error->clear();
  if (out == nullptr) return fail(error, "null parsed-header output");
  *out = M2RefParsedHeader{};
  if (config.num_ref_frames != state.num_ref_frames) {
    return fail(error, "config.num_ref_frames disagrees with FrameRefState");
  }
  const std::uint32_t frame_width_bits =
      size_bits(static_cast<std::uint32_t>(config.max_frame_width));
  const std::uint32_t frame_height_bits =
      size_bits(static_cast<std::uint32_t>(config.max_frame_height));

  std::uint32_t cur_mfh_id = 0;
  std::uint32_t seq_id = 0;
  if (!reader.uvlc(&cur_mfh_id) || cur_mfh_id != 0) {
    return fail(error, "expected cur_mfh_id = 0");
  }
  if (!reader.uvlc(&seq_id)) return fail(error, "truncated seq_header_id");

  M2RefFrameHeader& header = out->header;
  header.keyframe = keyframe;
  const bool is_key = keyframe;
  if (!is_key) {
    bool inter = false;
    if (!reader.bit(&inter)) return fail(error, "truncated frame_is_inter");
    header.frame_is_inter = inter;
  }
  bool immediate = false;
  if (!reader.bit(&immediate) || !immediate) {
    return fail(error, "these streams must code immediate_output_frame = 1");
  }

  bool override = false;
  if (!reader.bit(&override)) {
    return fail(error, "truncated frame_size_override_flag");
  }
  header.frame_size_override = override;
  std::uint32_t order_hint = 0;
  if (!reader.bits(config.order_hint_bits, &order_hint)) {
    return fail(error, "truncated order_hint");
  }
  header.order_hint = order_hint;

  const FrameType frame_type = is_key ? FrameType::kKey : FrameType::kInter;
  PrimaryRefSignal primary_signal;
  if (!read_primary_ref_signal(reader, frame_type, /*is_tip=*/false,
                               &primary_signal, error)) {
    return false;
  }
  header.signal_primary_ref_frame = primary_signal.signal_primary_ref_frame;
  header.primary_ref_frame = primary_signal.primary_ref_frame;
  header.disable_cross_frame_cdf_init =
      primary_signal.disable_cross_frame_cdf_init;

  RefreshConfig refresh_config;
  refresh_config.enable_short_refresh_frame_flags =
      config.enable_short_refresh_frame_flags;
  refresh_config.num_ref_frames = config.num_ref_frames;
  refresh_config.obu_is_closed_loop_key = is_key;
  refresh_config.max_mlayer_id_zero = true;
  if (!read_refresh_frame_flags(reader, frame_type, refresh_config,
                                /*immediate_output_frame=*/true,
                                &out->refresh, error)) {
    return false;
  }
  header.refresh_frame_flags = out->refresh.refresh_frame_flags;

  if (is_key) {
    if (override) {
      std::uint32_t width_minus_1 = 0;
      std::uint32_t height_minus_1 = 0;
      if (!reader.bits(frame_width_bits, &width_minus_1) ||
          !reader.bits(frame_height_bits, &height_minus_1)) {
        return fail(error, "truncated explicit key-frame size");
      }
      header.frame_width = width_minus_1 + 1U;
      header.frame_height = height_minus_1 + 1U;
      if (header.frame_width > config.max_frame_width ||
          header.frame_height > config.max_frame_height) {
        return fail(error, "explicit key-frame size exceeds the sequence max");
      }
      out->size.width = header.frame_width;
      out->size.height = header.frame_height;
      out->size.explicit_size = true;
    } else {
      out->size.width = config.max_frame_width;
      out->size.height = config.max_frame_height;
    }
    bool allow_intrabc = false;
    if (!reader.bit(&allow_intrabc)) {
      return fail(error, "truncated allow_intrabc");
    }
    header.allow_intrabc = allow_intrabc;
    if (allow_intrabc) {
      bool global = false;
      if (!reader.bit(&global)) {
        return fail(error, "truncated allow_global_intrabc");
      }
      header.allow_global_intrabc = global;
      if (global) {
        bool local = false;
        if (!reader.bit(&local) || !local) {
          return fail(error, "truncated allow_local_intrabc");
        }
      }
      BvDrlConfig bvp_config;
      bvp_config.allow_frame_max_bvp_drl_bits =
          config.allow_frame_max_bvp_drl_bits;
      bvp_config.seq_max_bvp_drl_bits_minus_1 =
          config.seq_max_bvp_drl_bits_minus_1;
      if (!read_bvp_drl(reader, bvp_config, &out->bvp_drl, error)) {
        return false;
      }
    }
  } else {
    bool explicit_map = false;
    if (!reader.bit(&explicit_map) || !explicit_map) {
      return fail(error,
                  "the implicit get_ref_frames() map is out of scope");
    }
    std::uint32_t num_total_refs = 0;
    if (!reader.bits(3U, &num_total_refs)) {
      return fail(error, "truncated num_total_refs");
    }
    if (num_total_refs != refs.num_total_refs) {
      return fail(error, "num_total_refs disagrees with the RefList");
    }
    const unsigned index_bits = ceil_log2(config.num_ref_frames);
    for (std::uint32_t i = 0; i < num_total_refs; ++i) {
      std::uint32_t idx = 0;
      if (!reader.bits(index_bits, &idx)) {
        return fail(error, "truncated ref_frame_idx");
      }
      if (idx != refs.index[i]) {
        return fail(error, "ref_frame_idx disagrees with the RefList");
      }
    }
    if (override) {
      SizeConfig size_config;
      size_config.num_total_refs = refs.num_total_refs;
      size_config.max_frame_width = state.max_frame_width;
      size_config.max_frame_height = state.max_frame_height;
      size_config.frame_width_bits = frame_width_bits;
      size_config.frame_height_bits = frame_height_bits;
      if (!read_frame_size_with_refs(reader, size_config, refs, state,
                                     &out->size, error)) {
        return false;
      }
      header.frame_width = out->size.width;
      header.frame_height = out->size.height;
      header.prefer_found_ref = out->size.inherited_from != kNoRef;
    } else {
      out->size.width = state.max_frame_width;
      out->size.height = state.max_frame_height;
    }

    bool allow_intrabc = false;
    if (!reader.bit(&allow_intrabc)) {
      return fail(error, "truncated allow_intrabc");
    }
    header.allow_intrabc = allow_intrabc;
    if (allow_intrabc) {
      BvDrlConfig bvp_config;
      bvp_config.allow_frame_max_bvp_drl_bits =
          config.allow_frame_max_bvp_drl_bits;
      bvp_config.seq_max_bvp_drl_bits_minus_1 =
          config.seq_max_bvp_drl_bits_minus_1;
      if (!read_bvp_drl(reader, bvp_config, &out->bvp_drl, error)) {
        return false;
      }
    }

    DrlConfig drl_config;
    drl_config.allow_frame_max_drl_bits = config.allow_frame_max_drl_bits;
    drl_config.seq_max_drl_bits_minus_1 = config.seq_max_drl_bits_minus_1;
    if (!read_drl(reader, drl_config, &out->drl, error)) return false;
    header.change_drl = out->drl.change_drl;
    header.max_drl_bits_minus_1 = out->drl.max_drl_bits_minus_1;

    bool qtr = true;
    bool high = false;
    if (!reader.bit(&qtr) || qtr) {
      return fail(error, "expected use_qtr_precision_mv = 0");
    }
    if (!reader.bit(&high) || !high) {
      return fail(error, "expected allow_high_precision_mv = 1");
    }
    bool switchable = true;
    if (!reader.bit(&switchable) || switchable) {
      return fail(error, "expected is_filter_switchable = 0");
    }
    std::uint32_t filter = 4;
    if (!reader.bits(2U, &filter) || filter != 0) {
      return fail(error, "expected interpolation_filter = EIGHTTAP");
    }
  }

  if (!resolve_primary_ref(frame_type, primary_signal, refs, state,
                           config.base_q_idx, &out->primary, error)) {
    return false;
  }

  bool disable_cdf_update = false;
  if (!reader.bit(&disable_cdf_update)) {
    return fail(error, "truncated disable_cdf_update");
  }
  if (disable_cdf_update != config.disable_cdf_update) {
    return fail(error, "disable_cdf_update disagrees with the config");
  }

  if (!read_tile_info(reader, config, tiles, error)) return false;

  std::uint32_t base_q_idx = 0;
  if (!reader.bits(8U, &base_q_idx)) return fail(error, "truncated base_q_idx");
  out->base_q_idx = base_q_idx;
  if (base_q_idx != config.base_q_idx) {
    return fail(error, "base_q_idx disagrees with the config");
  }
  bool segmentation = true;
  bool qmatrix = true;
  bool delta_q = true;
  bool deblock0 = true;
  bool deblock1 = true;
  if (!reader.bit(&segmentation) || segmentation) {
    return fail(error, "segmentation_enabled must be 0");
  }
  if (!reader.bit(&qmatrix) || qmatrix) {
    return fail(error, "using_qmatrix must be 0");
  }
  if (!reader.bit(&delta_q) || delta_q) {
    return fail(error, "delta_q_present must be 0");
  }
  if (!reader.bit(&deblock0) || deblock0) {
    return fail(error, "apply_deblocking_filter[0] must be 0");
  }
  if (!reader.bit(&deblock1) || deblock1) {
    return fail(error, "apply_deblocking_filter[1] must be 0");
  }
  bool tx_select = true;
  if (!reader.bit(&tx_select) || tx_select) {
    return fail(error, "tx_mode_select must be 0");
  }
  if (!is_key) {
    bool reference_select = true;
    bool skip_mode = true;
    if (!reader.bit(&reference_select) || reference_select) {
      return fail(error, "reference_select must be 0");
    }
    if (!reader.bit(&skip_mode) || skip_mode) {
      return fail(error, "skip_mode_present must be 0");
    }
  }
  std::uint32_t reduced = 3;
  if (!reader.bits(2U, &reduced) || reduced != 0) {
    return fail(error, "reduced_tx_set must be 0");
  }
  return true;
}

bool read_m2_ref_nonfirst_tile_group_header(RefBitReader& reader,
                                            const FrameHeaderStore& store,
                                            bool* frame_header_present,
                                            std::string* error) {
  return read_nonfirst_tile_group_header(reader, store, frame_header_present,
                                         error);
}

}  // namespace av2enc::frame::ref_fields
