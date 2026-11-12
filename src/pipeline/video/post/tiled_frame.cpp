#include "pipeline/video/post/tiled_frame.h"

#include "core/bitstream/basic_headers.h"
#include "pipeline/video/post/bit_depth_checklist.h"
#include "pipeline/video/post/rc_adapter.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>

namespace av2enc::video::post {
namespace {

using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;

bool fail(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

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
  while ((1U << w) < n) ++w;
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

std::uint32_t ceil_log2_u32(std::uint32_t value) {
  if (value <= 1U) return 0U;
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

bool validate_sequence_options(const VideoPostSequenceOptions& sequence,
                               std::string* error) {
  if (video_base_q_idx_bits(sequence.depth) < 0 ||
      video_max_quantizer(sequence.depth) < 0) {
    return fail(error, "m2 tiled: 12-bit is reserved in AV2 v1.0.0");
  }
  const auto& f = sequence.filters;
  if (f.enable_gdf) {
    return fail(error, "m2 tiled: GDF is not implemented");
  }
  if (f.cdef_on_skip_txfm_always_on && f.cdef_on_skip_txfm_disabled) {
    return fail(error,
                "m2 tiled: CDEF_ON_SKIP_TXFM cannot be both always-on and "
                "disabled");
  }
  return true;
}

std::size_t write_sequence_header(const VideoConfig& c,
                                  const VideoPostSequenceOptions& sequence,
                                  BitWriter& writer, std::string* error) {
  const std::size_t start = writer.bit_size();
  assert(c.order_hint_bits >= 1U && c.order_hint_bits <= 16U);
  assert(c.num_ref_frames >= 1U && c.num_ref_frames <= 16U);
  assert(c.max_drl_bits_minus_1 < 5U);

  const int idc = fmt::bit_depth_idc(sequence.depth);
  if (idc < 0) {
    fail(error, "m2 tiled: 12-bit cannot be signalled");
    return 0;
  }

  put_uvlc(writer, 0);                    // seq_header_id
  writer.put_bits(0U, 5U);                // seq_profile_idc
  writer.put_bit(false);                  // single_picture_header_flag = 0
  writer.put_bits(c.seq_level_idx, 5U);   // seq_level_idx
  writer.put_bit(false);                  // seq_tier
  put_uvlc(writer, 0);                    // chroma_format_idc = 4:2:0
  put_uvlc(writer, static_cast<std::uint32_t>(idc));  // bit_depth_idc

  writer.put_bits(0U, 3U);   // seq_lcr_id
  writer.put_bit(false);     // still_picture
  writer.put_bits(0U, 2U);   // max_tlayer_id
  writer.put_bits(0U, 3U);   // max_mlayer_id
  writer.put_bit(true);      // monotonic_output_order_flag

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
  writer.put_bit(c.enable_cfl_intra);  // enable_cfl_intra
  writer.put_bits(0U, 2U);             // cfl_ds_filter_index
  writer.put_bit(false);               // enable_mhccp
  writer.put_bit(false);               // enable_ibp

  // §5.4.6 sequence_inter_config() for single_picture_header_flag = 0.
  for (int mode = 1; mode < 5; ++mode) writer.put_bit(false);
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
  writer.put_bit(false);  // enable_avg_cdf
  writer.put_bit(false);  // separate_uv_delta_q
  writer.put_bit(true);   // equal_ac_dc_q
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  // §5.4.10 sequence_filter_config().
  std::string filter_error;
  const std::size_t filter_bits = filters_stream::write_sequence_filter_config(
      writer, sequence.filters, /*single_picture_header_flag=*/false,
      &filter_error);
  if (!filter_error.empty()) {
    fail(error, "m2 tiled: sequence_filter_config: " + filter_error);
    return 0;
  }
  if (filter_bits == 0) {
    fail(error, "m2 tiled: sequence_filter_config wrote no bits");
    return 0;
  }

  // §5.4.2 sequence_tile_config().
  writer.put_bit(false);  // seq_tile_info_present_flag
  writer.put_bit(false);  // film_grain_params_present

  return writer.bit_size() - start;
}

bool validate_layout(const VideoConfig& config, const tiles::TileLayout& layout,
                     std::string* error) {
  if (layout.tile_count() <= 0) return fail(error, "m2 tiled: empty layout");
  if (layout.width != static_cast<int>(config.max_frame_width) ||
      layout.height != static_cast<int>(config.max_frame_height)) {
    return fail(error, "m2 tiled: layout size does not match the sequence");
  }
  if (layout.col_starts_sb.size() !=
          static_cast<std::size_t>(layout.cols) ||
      layout.row_starts_sb.size() !=
          static_cast<std::size_t>(layout.rows)) {
    return fail(error, "m2 tiled: layout start arrays are inconsistent");
  }
  std::string layout_error;
  if (!tiles::validate_tile_layout(layout, layout_error)) {
    return fail(error, "m2 tiled: invalid layout: " + layout_error);
  }
  // tile_info() writes increments relative to the level-derived
  // minLog2TileCols/minLog2Tiles.  A layout planned for a different level (the
  // planner's default 31 is a planning-only value) would emit a different
  // number of increment bits than the decoder expects.
  const tiles::LayoutResult expected = tiles::plan_tile_layout(
      layout.width, layout.height, layout.cols, layout.rows, layout.sb_size,
      config.seq_level_idx, 0);
  if (!expected.ok()) {
    return fail(error,
                "m2 tiled: layout is not representable at the sequence level: " +
                    expected.error);
  }
  const tiles::TileLayout& e = expected.layout;
  if (e.cols_log2 != layout.cols_log2 || e.rows_log2 != layout.rows_log2 ||
      e.min_log2_tile_cols != layout.min_log2_tile_cols ||
      e.max_log2_tile_cols != layout.max_log2_tile_cols ||
      e.max_log2_tile_rows != layout.max_log2_tile_rows ||
      e.min_log2_tiles != layout.min_log2_tiles ||
      e.col_starts_sb != layout.col_starts_sb ||
      e.row_starts_sb != layout.row_starts_sb) {
    return fail(error,
                "m2 tiled: layout was planned for a different sequence level; "
                "pass config.seq_level_idx to plan_video_tiles()/the pipeline");
  }
  return true;
}

}  // namespace

VideoTilePlan plan_video_tiles(int width, int height, int cols, int rows,
                         int sb_size, int seq_level_idx, int seq_tier) {
  VideoTilePlan out;
  std::string geometry_error;
  if (!video_geometry_supported(width, height, sb_size, &geometry_error)) {
    out.error = geometry_error;
    return out;
  }
  tiles::LayoutResult result =
      tiles::plan_tile_layout(width, height, cols, rows, sb_size,
                              seq_level_idx, seq_tier);
  if (!result.ok()) {
    out.error = result.error;
    return out;
  }
  out.layout = std::move(result.layout);
  out.ok = true;
  return out;
}

bool adopt_pipeline_frame_plan(const pipeline::FramePipelinePlan& plan,
                               VideoTiledFrameHeader* header, VideoTilePlan* tiles,
                               std::string* error) {
  if (header == nullptr || tiles == nullptr) {
    return fail(error, "m2 tiled: null pipeline adoption output");
  }
  if (!plan.valid || plan.layout.empty()) {
    return fail(error, "m2 tiled: pipeline plan is not valid");
  }
  if (plan.base_q_idx < 1 || plan.base_q_idx > 255) {
    return fail(error, "m2 tiled: pipeline base_q_idx is outside [1, 255]");
  }
  std::string layout_error;
  if (!tiles::validate_tile_layout(plan.layout, layout_error)) {
    return fail(error, "m2 tiled: pipeline layout is invalid: " + layout_error);
  }
  for (const pipeline::TilePlan& tile : plan.tiles) {
    if (tile.qindex != plan.base_q_idx) {
      return fail(error,
                  "m2 tiled: per-tile QP requires delta-Q/segmentation syntax "
                  "that this writer does not emit; use kUniform");
    }
  }
  tiles->ok = true;
  tiles->layout = plan.layout;
  tiles->error.clear();
  header->base_q_idx = plan.base_q_idx;
  return true;
}

bool write_video_tile_info(BitWriter& writer, const tiles::TileLayout& layout,
                        int tile_size_bytes, std::string* error) {
  if (layout.tile_count() <= 0) {
    return fail(error, "m2 tiled: empty layout");
  }
  writer.put_bit(true);  // uniform_tile_spacing_flag

  const int cols_log2 = layout.cols_log2;
  const int rows_log2 = layout.rows_log2;
  const int min_cols_log2 = layout.min_log2_tile_cols;
  const int max_cols_log2 = layout.max_log2_tile_cols;
  const int min_tiles_log2 = layout.min_log2_tiles;
  const int max_rows_log2 = layout.max_log2_tile_rows;
  const int min_rows_log2 = std::max(min_tiles_log2 - cols_log2, 0);

  if (min_cols_log2 < 0 || cols_log2 < min_cols_log2 ||
      cols_log2 > max_cols_log2) {
    return fail(error, "m2 tiled: tile column log2 outside the level bounds");
  }
  for (int l = min_cols_log2; l < cols_log2; ++l) writer.put_bit(true);
  if (cols_log2 < max_cols_log2) writer.put_bit(false);

  if (min_rows_log2 < 0 || rows_log2 < min_rows_log2 ||
      rows_log2 > max_rows_log2) {
    return fail(error, "m2 tiled: tile row log2 outside the level bounds");
  }
  for (int l = min_rows_log2; l < rows_log2; ++l) writer.put_bit(true);
  if (rows_log2 < max_rows_log2) writer.put_bit(false);

  if (layout.tile_count() > 1) {
    if (tile_size_bytes < 1 || tile_size_bytes > 4) {
      return fail(error, "m2 tiled: TileSizeBytes must be in 1..4");
    }
    const unsigned n = static_cast<unsigned>(cols_log2 + rows_log2);
    // context_update_tile_id.  With disable_cdf_update == 1 it is ignored by
    // the decoder; 0 is the deterministic choice.  With adaptive CDFs a real
    // encoder would pick the tile whose updated CDFs become the frame's.
    writer.put_bits(0U, n);
    writer.put_bits(static_cast<std::uint64_t>(tile_size_bytes - 1), 2U);
  }
  return true;
}

bool write_video_frame_header_syntax(BitWriter& writer, const VideoConfig& config,
                                  const VideoPostSequenceOptions& sequence,
                                  const VideoTiledFrameHeader& header,
                                  const tiles::TileLayout& layout,
                                  int tile_size_bytes, std::size_t* bits,
                                  std::string* error) {
  if (!validate_sequence_options(sequence, error)) return false;
  if (!validate_layout(config, layout, error)) return false;
  if (layout.tile_count() > 1 && !config.disable_cdf_update) {
    return fail(error,
                "m2 tiled: multi-tile adaptive CDFs need per-tile CDF carry and "
                "context_update_tile_id; keep disable_cdf_update = 1");
  }
  if (sequence.filters.enable_restoration) {
    return fail(error,
                "m2 tiled: loop restoration (Wiener) tile syntax is not "
                "implemented in this layer");
  }

  const int max_q = video_max_quantizer(sequence.depth);
  const int base_q_idx =
      header.base_q_idx >= 0 ? header.base_q_idx : config.base_q_idx;
  if (max_q < 0) return fail(error, "m2 tiled: unsupported bit depth");
  if (base_q_idx < 1 || base_q_idx > max_q) {
    return fail(error, "m2 tiled: base_q_idx is outside [1, MaxQ]");
  }
  if (header.cdef_frame_enable && !sequence.filters.enable_cdef) {
    return fail(error,
                "m2 tiled: CDEF frame enable requested but the sequence header "
                "disables CDEF");
  }
  if (header.ccso_frame_flag && !sequence.filters.enable_ccso) {
    return fail(error,
                "m2 tiled: CCSO frame flag requested but the sequence header "
                "disables CCSO");
  }

  const std::size_t start = writer.bit_size();
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header

  const bool key = header.keyframe;
  if (!key) writer.put_bit(true);  // frame_is_inter = 1
  writer.put_bit(true);            // immediate_output_frame

  writer.put_bit(false);  // frame_size_override_flag = 0
  writer.put_bits(header.order_hint, config.order_hint_bits);  // order_hint

  if (key) {
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
    writer.put_bits(header.refresh_frame_flags, config.num_ref_frames);
    writer.put_bit(true);   // frame_explicit_ref_frame_map
    writer.put_bits(config.num_total_refs, 3U);
    writer.put_bits(header.ref_slot,
                    static_cast<unsigned>(ceil_log2_u32(config.num_ref_frames)));
    writer.put_bit(false);  // allow_intrabc
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv -> 1/8 pel
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTTAP
  }

  writer.put_bit(config.disable_cdf_update);  // disable_cdf_update

  if (!write_video_tile_info(writer, layout, tile_size_bytes, error)) {
    return false;
  }

  writer.put_bits(static_cast<std::uint64_t>(base_q_idx),
                  static_cast<unsigned>(video_base_q_idx_bits(sequence.depth)));
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present

  // deblocking_filter_params() (§5.18.7.9).
  if (!header.apply_deblocking_filter[0] &&
      !header.apply_deblocking_filter[1] &&
      (header.apply_deblocking_filter[2] ||
       header.apply_deblocking_filter[3])) {
    return fail(error,
                "m2 tiled: chroma deblock flags require a luma direction");
  }
  std::string deblock_error;
  write_video_deblock_params(writer, header.apply_deblocking_filter,
                          header.df_delta_q_present, header.df_delta_q,
                          sequence.filters.df_par_bits_minus_2,
                          &deblock_error);
  if (!deblock_error.empty()) {
    return fail(error, "m2 tiled: " + deblock_error);
  }

  // cdef_params() (§5.18.7.10).  With single_picture_header_flag = 0 the
  // frame-enable bit is present even for a keyframe.
  if (sequence.filters.enable_cdef) {
    filters_stream::CdefHeaderContext ctx;
    ctx.coded_lossless = false;
    ctx.enable_cdef = true;
    ctx.single_picture_header_flag = false;
    ctx.cdef_on_skip_txfm = sequence.filters.cdef_on_skip_txfm_always_on
                                ? 1
                                : (sequence.filters.cdef_on_skip_txfm_disabled
                                       ? 0
                                       : 2);
    ctx.num_planes = 3;
    filters_stream::CdefParams params = header.cdef;
    params.frame_enable = header.cdef_frame_enable;
    std::string cdef_error;
    filters_stream::write_cdef_params(writer, params, ctx, &cdef_error);
    if (!cdef_error.empty()) {
      return fail(error, "m2 tiled: cdef_params: " + cdef_error);
    }
  }

  // lr_params() would be here; restoration is rejected above.

  // ccso_params() (§5.18.7.12).
  if (sequence.filters.enable_ccso) {
    filters_stream::CcsoHeaderContext ctx;
    ctx.coded_lossless = false;
    ctx.enable_ccso = true;
    ctx.single_picture_header_flag = false;
    ctx.frame_is_intra = key;
    ctx.num_planes = 3;
    ctx.sb_size = layout.sb_size;
    ctx.ccso_unit_matches_sb_size =
        sequence.filters.ccso_unit_matches_sb_size;
    int tile_start_or = 0;
    const int sb_mi = layout.sb_size / 4;
    for (int sb : layout.col_starts_sb) tile_start_or |= sb * sb_mi;
    for (int sb : layout.row_starts_sb) tile_start_or |= sb * sb_mi;
    ctx.tile_start_or = tile_start_or;
    filters_stream::CcsoParams params = header.ccso;
    params.frame_flag = header.ccso_frame_flag;
    std::string ccso_error;
    filters_stream::write_ccso_params(writer, params, ctx, &ccso_error);
    if (!ccso_error.empty()) {
      return fail(error, "m2 tiled: ccso_params: " + ccso_error);
    }
  }

  writer.put_bit(false);  // tx_mode_select = 0 -> TX_MODE_LARGEST
  if (!key) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
  // global_motion_params() / film_grain_config() encode no bits.

  if (bits != nullptr) *bits = writer.bit_size() - start;
  return true;
}

bool split_video_blocks_for_tile(const tiles::TileLayout& layout, int tile_index,
                              std::span<const syntax::BlockSyntax> blocks,
                              VideoTileBlocks* out, std::string* error) {
  if (out == nullptr) return fail(error, "m2 tiled: null tile block output");
  const tiles::TileRect* tile = layout.at_index(tile_index);
  if (tile == nullptr) return fail(error, "m2 tiled: tile index out of range");
  const int sb_mi = layout.sb_size / 4;
  const int tile_mi_row = tile->sb_row_start * sb_mi;
  const int tile_mi_col = tile->sb_col_start * sb_mi;

  VideoTileBlocks result;
  result.tile_index = tile_index;
  result.width = tile->pixel_width();
  result.height = tile->pixel_height();
  result.sb_cols = tile->sb_cols;
  result.sb_rows = tile->sb_rows;

  std::size_t index = 0;
  int roots = 0;
  while (index < blocks.size()) {
    const syntax::BlockSyntax& root = blocks[index];
    if ((root.flags & syntax::kFlagSbRoot) == 0) {
      return fail(error, "m2 tiled: block list root missing kFlagSbRoot");
    }
    const std::size_t end = index + 1 + root.subtree;
    if (end > blocks.size()) {
      return fail(error, "m2 tiled: block subtree past the end");
    }
    const int root_sb_col = root.mi_col / sb_mi;
    const int root_sb_row = root.mi_row / sb_mi;
    const bool inside =
        root_sb_col >= tile->sb_col_start &&
        root_sb_col < tile->sb_col_start + tile->sb_cols &&
        root_sb_row >= tile->sb_row_start &&
        root_sb_row < tile->sb_row_start + tile->sb_rows;
    if (inside) {
      if (root.mi_col % sb_mi != 0 || root.mi_row % sb_mi != 0) {
        return fail(error, "m2 tiled: superblock root is not aligned");
      }
      for (std::size_t i = index; i < end; ++i) {
        syntax::BlockSyntax node = blocks[i];
        if (node.mi_col < tile->sb_col_start * sb_mi ||
            node.mi_row < tile->sb_row_start * sb_mi ||
            node.mi_col >= (tile->sb_col_start + tile->sb_cols) * sb_mi ||
            node.mi_row >= (tile->sb_row_start + tile->sb_rows) * sb_mi) {
          return fail(error,
                      "m2 tiled: a partition subtree crosses a tile boundary");
        }
        node.mi_col = static_cast<std::uint16_t>(node.mi_col - tile_mi_col);
        node.mi_row = static_cast<std::uint16_t>(node.mi_row - tile_mi_row);
        result.blocks.push_back(node);
      }
      ++roots;
    }
    index = end;
  }
  if (roots != tile->sb_cols * tile->sb_rows || index != blocks.size()) {
    std::ostringstream text;
    text << "m2 tiled: tile " << tile_index << " has " << roots
         << " superblock roots, expected " << tile->sb_cols * tile->sb_rows;
    return fail(error, text.str());
  }
  *out = std::move(result);
  return true;
}

bool assemble_video_tiled_frame(const VideoConfig& config,
                             const VideoPostSequenceOptions& sequence,
                             const VideoTiledFrameHeader& header,
                             const tiles::TileLayout& layout,
                             std::span<const syntax::BlockSyntax> blocks,
                             std::span<const std::int32_t> levels,
                             VideoTiledFrame* out, std::string* error) {
  if (out == nullptr) return fail(error, "m2 tiled: null output");
  if (sequence.depth != fmt::BitDepth::k8) {
    return fail(error,
                "m2 tiled: tile payload coding is 8-bit; 10-bit M2 needs the "
                "hbd transform/quant/recon path (see bit_depth_checklist.h)");
  }
  if (!validate_sequence_options(sequence, error)) return false;
  if (!validate_layout(config, layout, error)) return false;

  std::vector<std::vector<std::uint8_t>> payloads;
  payloads.reserve(static_cast<std::size_t>(layout.tile_count()));
  for (int tile_index = 0; tile_index < layout.tile_count(); ++tile_index) {
    VideoTileBlocks tile_blocks;
    if (!split_video_blocks_for_tile(layout, tile_index, blocks, &tile_blocks,
                                  error)) {
      return false;
    }
    syntax::FramePlan local_plan;
    local_plan.width = tile_blocks.width;
    local_plan.height = tile_blocks.height;
    local_plan.sb_size = layout.sb_size;
    local_plan.inter_frame = !header.keyframe;
    VideoConfig local_config = config;
    local_config.max_frame_width = static_cast<std::uint16_t>(tile_blocks.width);
    local_config.max_frame_height =
        static_cast<std::uint16_t>(tile_blocks.height);
    std::vector<std::uint8_t> payload;
    std::string assemble_error;
    if (!syntax::assemble_video_frame(local_plan, tile_blocks.blocks, levels,
                                   local_config, payload, assemble_error)) {
      std::ostringstream text;
      text << "m2 tiled: tile " << tile_index << ": " << assemble_error;
      return fail(error, text.str());
    }
    if (payload.size() < 2) {
      return fail(error, "m2 tiled: a tile payload is shorter than 2 bytes");
    }
    payloads.push_back(std::move(payload));
  }
  return assemble_video_tile_group_from_payloads(config, sequence, header, layout,
                                              payloads, out, error);
}

bool assemble_video_tile_group_from_payloads(
    const VideoConfig& config, const VideoPostSequenceOptions& sequence,
    const VideoTiledFrameHeader& header, const tiles::TileLayout& layout,
    std::span<const std::vector<std::uint8_t>> tile_payloads,
    VideoTiledFrame* out, std::string* error) {
  if (out == nullptr) return fail(error, "m2 tiled: null output");
  if (tile_payloads.size() != static_cast<std::size_t>(layout.tile_count()) ||
      tile_payloads.empty()) {
    return fail(error, "m2 tiled: expected one payload per tile");
  }
  if (layout.tile_count() > 1 && !config.disable_cdf_update) {
    return fail(error,
                "m2 tiled: multi-tile adaptive CDFs are not modelled");
  }

  VideoTiledFrame result;
  result.tile_payloads.reserve(tile_payloads.size());
  result.tile_bytes.reserve(tile_payloads.size());
  std::uint32_t largest = 0;
  for (const auto& payload : tile_payloads) {
    if (payload.size() < 2) {
      return fail(error, "m2 tiled: a tile payload is shorter than 2 bytes");
    }
    if (payload.size() > 0xFFFFFFFFull) {
      return fail(error, "m2 tiled: tile payload exceeds 4 GiB");
    }
    const std::uint32_t size = static_cast<std::uint32_t>(payload.size());
    result.tile_payloads.push_back(payload);
    result.tile_bytes.push_back(size);
    largest = std::max(largest, size);
  }
  result.largest_tile_bytes = largest;
  result.tile_size_bytes = tiles::tile_size_bytes_for(largest);
  if (result.tile_size_bytes == 0) {
    return fail(error, "m2 tiled: tile size cannot be represented in 1..4 bytes");
  }

  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  std::size_t frame_header_bits = 0;
  if (!write_video_frame_header_syntax(
          writer, config, sequence, header, layout, result.tile_size_bytes,
          &frame_header_bits, error)) {
    return false;
  }
  if (layout.tile_count() > 1) {
    writer.put_bit(false);  // tile_start_and_end_present_flag = 0
  }
  writer.align_to_byte();
  result.frame_header_bits = frame_header_bits;
  result.tile_group_header_bytes = writer.byte_size();

  std::vector<std::uint8_t> tile_group = writer.take_bytes();
  // tile_group_payload(): the size field for each tile precedes that tile's
  // payload.  For the last tile the size is implied by the remaining OBU
  // bytes, so there is no field.  Writing all size fields first is a classic
  // desync that only shows up at NumTiles >= 3 (with two tiles there is one
  // field, which happens to be in the right place).
  for (int i = 0; i < layout.tile_count(); ++i) {
    if (i + 1 < layout.tile_count()) {
      const std::uint32_t value =
          result.tile_bytes[static_cast<std::size_t>(i)] - 1;
      for (int b = 0; b < result.tile_size_bytes; ++b) {
        tile_group.push_back(
            static_cast<std::uint8_t>((value >> (8 * b)) & 0xFFU));
      }
    }
    const auto& payload = result.tile_payloads[static_cast<std::size_t>(i)];
    tile_group.insert(tile_group.end(), payload.begin(), payload.end());
  }

  const ObuType type =
      header.keyframe ? ObuType::kClosedLoopKey : ObuType::kRegularTileGroup;
  result.obu = av2enc::basic::make_obu(ObuHeader{type}, tile_group);
  *out = std::move(result);
  return true;
}

std::vector<std::uint8_t> make_video_sequence_header_obu(
    const VideoConfig& config, const VideoPostSequenceOptions& sequence,
    std::string* error) {
  if (error != nullptr) error->clear();
  if (!validate_sequence_options(sequence, error)) return {};
  BitWriter writer;
  if (write_sequence_header(config, sequence, writer, error) == 0) return {};
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

}  // namespace av2enc::video::post
