#include "codec/inter/bru/bru.h"

#include "core/bitstream/basic_headers.h"
#include "codec/inter/bru/bru_common.h"
#include "pipeline/still/basic_cdf_tables.h"
#include "pipeline/video/video_cdf_tables.h"
#include "codec/entropy/av2/av2_entropy.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>

namespace av2enc::inter::bru {
namespace {

using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;
using detail::ceil_log2;
using detail::put_ns;
using detail::put_trailing_bits;
using detail::put_uvlc;

constexpr std::uint16_t kDefaultBruModeCdf[4] = {4124, 16615, 0, 0};

std::size_t write_sequence_header(BitWriter& writer,
                                  const SequenceConfig& c) {
  const std::size_t start = writer.bit_size();
  assert(c.order_hint_bits >= 1U && c.order_hint_bits <= 16U);
  assert(c.num_ref_frames >= 1U && c.num_ref_frames <= 16U);
  assert(c.max_drl_bits_minus_1 < 5U);

  put_uvlc(writer, 0);                 // seq_header_id
  writer.put_bits(0U, 5U);             // seq_profile_idc = Main_420_10_IP0
  writer.put_bit(false);               // single_picture_header_flag = 0
  writer.put_bits(c.seq_level_idx, 5U);
  writer.put_bit(false);               // seq_tier (level > 3)
  put_uvlc(writer, 0);                 // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                 // bit_depth_idc = 1 => 8-bit

  // §5.4.1 non-single-picture header.
  writer.put_bits(0U, 3U);  // seq_lcr_id
  writer.put_bit(false);    // still_picture
  writer.put_bits(0U, 2U);  // max_tlayer_id
  writer.put_bits(0U, 3U);  // max_mlayer_id
  writer.put_bit(true);     // monotonic_output_order_flag

  int width_bits =
      detail::bit_width(static_cast<std::uint32_t>(c.max_frame_width) - 1U);
  if (width_bits == 0) width_bits = 1;
  int height_bits =
      detail::bit_width(static_cast<std::uint32_t>(c.max_frame_height) - 1U);
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(static_cast<std::uint64_t>(width_bits) - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(height_bits) - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_width) - 1U,
                  static_cast<unsigned>(width_bits));
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_height) - 1U,
                  static_cast<unsigned>(height_bits));

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
  writer.put_bit(false);          // enable_dip
  writer.put_bit(true);           // enable_intra_edge_filter
  writer.put_bit(false);          // enable_mrls
  writer.put_bit(c.enable_cfl_intra);
  writer.put_bits(0U, 2U);        // cfl_ds_filter_index
  writer.put_bit(false);          // enable_mhccp
  writer.put_bit(false);          // enable_ibp

  // §5.4.6 sequence_inter_config().
  for (int mode = 1; mode < 5; ++mode) {
    writer.put_bit(false);        // INTERINTRA..EXTENDWARP
  }
  writer.put_bit(false);          // enable_masked_compound
  writer.put_bit(false);          // enable_ref_frame_mvs
  writer.put_bits(static_cast<std::uint64_t>(c.order_hint_bits) - 1U, 4U);
  writer.put_bit(false);          // enable_refmvbank
  writer.put_bit(true);           // disable_drl_reorder
  writer.put_bit(true);           // explicit_ref_frame_map
  writer.put_bit(true);           // explicit_num_ref_frames
  writer.put_bits(static_cast<std::uint64_t>(c.num_ref_frames) - 1U, 4U);
  writer.put_bits(0U, 3U);        // long_term_frame_id_bits
  put_ns(writer, 5U, c.max_drl_bits_minus_1);
  writer.put_bit(false);          // allow_frame_max_drl_bits
  put_ns(writer, 3U, 0U);         // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);          // allow_frame_max_bvp_drl_bits
  writer.put_bits(0U, 2U);        // num_same_ref_compound
  writer.put_bit(false);          // enable_tip
  writer.put_bit(false);          // enable_mv_traj
  writer.put_bit(false);          // enable_bawp
  writer.put_bit(false);          // enable_cwp
  writer.put_bit(false);          // enable_imp_msk_bld
  writer.put_bit(false);          // enable_df_sub_pu
  writer.put_bits(0U, 2U);        // enable_opfl_refine = REFINE_NONE
  writer.put_bit(false);          // enable_refinemv
  writer.put_bit(c.enable_bru);   // §5.4.6 enable_bru
  writer.put_bit(false);          // enable_adaptive_mvd
  writer.put_bit(false);          // enable_mvd_sign_derive
  writer.put_bit(false);          // enable_flex_mvres
  writer.put_bit(false);          // enable_global_motion
  writer.put_bit(false);          // enable_short_refresh_frame_flags

  // §5.4.7 sequence_scc_config(): no screen content tools.
  writer.put_bit(false);          // seq_choose_screen_content_tools
  writer.put_bit(false);          // seq_force_screen_content_tools

  // §5.4.8 sequence_transform_quant_entropy_config().
  writer.put_bit(false);          // enable_fsc
  writer.put_bit(false);          // enable_idtx_intra
  writer.put_bit(false);          // enable_intra_ist
  writer.put_bit(false);          // enable_inter_ist
  writer.put_bit(true);           // enable_chroma_dctonly
  writer.put_bit(false);          // enable_inter_ddt
  writer.put_bit(false);          // reduced_tx_part_set
  writer.put_bit(false);          // enable_cctx
  writer.put_bit(false);          // enable_tcq
  writer.put_bit(false);          // enable_parity_hiding
  writer.put_bit(false);          // enable_avg_cdf
  writer.put_bit(false);          // separate_uv_delta_q
  writer.put_bit(true);           // equal_ac_dc_q
  writer.put_bits(23U, 5U);       // base_uv_ac_delta_q => 0
  writer.put_bit(false);          // uv_ac_delta_q_enabled

  // §5.4.10 sequence_filter_config().
  writer.put_bit(false);          // disable_loopfilters_across_tiles
  writer.put_bit(false);          // enable_cdef
  writer.put_bit(false);          // enable_gdf
  writer.put_bit(false);          // enable_restoration
  writer.put_bit(false);          // enable_ccso
  writer.put_bit(false);          // cdef_on_skip_txfm_always_on
  writer.put_bit(true);           // cdef_on_skip_txfm_disabled
  writer.put_bits(0U, 2U);        // df_par_bits_minus_2

  // §5.4.2 sequence_tile_config().
  writer.put_bit(false);          // seq_tile_info_present_flag
  writer.put_bit(false);          // film_grain_params_present

  return writer.bit_size() - start;
}

void write_tile_log2(BitWriter& writer, int min_log2, int desired,
                     int max_log2) {
  assert(desired >= min_log2 && desired <= max_log2);
  int current = min_log2;
  while (current < max_log2) {
    const bool increment = current < desired;
    writer.put_bit(increment);
    if (!increment) break;
    ++current;
  }
  assert(current == desired);
}

void write_tile_info(BitWriter& writer, const SequenceConfig& c,
                     const FrameConfig& f) {
  const int sb_cols = sb_cols_for(c.max_frame_width);
  const int sb_rows = sb_rows_for(c.max_frame_height);
  const int min_cols = detail::min_log2_tile_cols(c, sb_cols);
  const int max_cols = detail::max_log2_tile_cols(sb_cols);
  const int min_tiles = detail::min_log2_tiles(c, sb_cols, sb_rows, min_cols);
  const int min_rows = std::max(min_tiles - f.tile_cols_log2, 0);
  const int max_rows = detail::max_log2_tile_rows(sb_rows);
  assert(f.tile_cols_log2 >= min_cols && f.tile_cols_log2 <= max_cols);
  assert(f.tile_rows_log2 >= min_rows && f.tile_rows_log2 <= max_rows);

  writer.put_bit(true);  // uniform_tile_spacing_flag
  write_tile_log2(writer, min_cols, f.tile_cols_log2, max_cols);
  write_tile_log2(writer, min_rows, f.tile_rows_log2, max_rows);

  const int tile_cols = 1 << f.tile_cols_log2;
  const int tile_rows = 1 << f.tile_rows_log2;
  if (tile_cols > 1 || tile_rows > 1) {
    const int n = f.tile_cols_log2 + f.tile_rows_log2;
    writer.put_bits(f.context_update_tile_id, static_cast<unsigned>(n));
    writer.put_bits(f.tile_size_bytes_minus_1, 2U);
  }
}

std::size_t write_frame_header_syntax(BitWriter& writer,
                                      const SequenceConfig& c,
                                      const FrameConfig& f) {
  const std::size_t start = writer.bit_size();
  assert(c.base_q_idx >= 1U);
  assert(f.num_total_refs >= 1U && f.num_total_refs <= 7U);
  assert(f.bru_ref < f.num_total_refs);

  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header

  const bool key = f.keyframe;
  if (!key) writer.put_bit(true);  // frame_is_inter = 1
  writer.put_bit(f.immediate_output_frame);

  // Frame size inheritance from the sequence / MFH.
  writer.put_bit(false);  // frame_size_override_flag = 0
  writer.put_bits(f.order_hint, c.order_hint_bits);

  if (!key) {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
  }

  if (key) {
    // OBU_CLOSED_LOOP_KEY + max_mlayer_id == 0 -> allFrames, no bits.
  } else {
    writer.put_bits(f.refresh_frame_flags, c.num_ref_frames);
  }

  if (key) {
    // FrameIsIntra: primary_ref_frame = PRIMARY_REF_NONE.
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(true);   // frame_explicit_ref_frame_map
    writer.put_bits(f.num_total_refs, 3U);
    const int n = ceil_log2(c.num_ref_frames);
    for (std::uint8_t i = 0; i < f.num_total_refs; ++i) {
      writer.put_bits(f.ref_frame_idx[i], static_cast<unsigned>(n));
    }

    // §5.18.2 p102: the BRU fields sit between frame_size() and the
    // use_ref_frame_mvs branch.
    bool use_bru = false;
    if (c.enable_bru) {
      writer.put_bit(f.use_bru);
      use_bru = f.use_bru;
      if (use_bru) {
        const int nb = ceil_log2(f.num_total_refs);
        if (nb > 0) {
          writer.put_bits(f.bru_ref, static_cast<unsigned>(nb));
        }
        writer.put_bit(f.bru_inactive);
      }
    }

    // use_ref_frame_mvs is inferred 0 (enable_ref_frame_mvs == 0).
    if (f.bru_inactive) {
      // §5.18.2 p105-107: the inactive frame returns immediately after
      // film_grain_config(); no tile_info, quantization or filter syntax is
      // coded.  film_grain_params_present == 0 so nothing else is needed.
      return writer.bit_size() - start;
    }
    if (!use_bru) {
      // BruModes are all BRU_ACTIVE by inference.
    }

    writer.put_bit(false);  // allow_intrabc
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv -> 1/8 pel
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTAP
    // frame_enabled_motion_modes[] all zero -> no bits.
  }

  writer.put_bit(c.disable_cdf_update);

  write_tile_info(writer, c, f);

  writer.put_bits(c.base_q_idx, 8U);
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present (base_q_idx > 0)
  writer.put_bit(false);  // apply_deblocking_filter[0]
  writer.put_bit(false);  // apply_deblocking_filter[1]
  writer.put_bit(false);  // tx_mode_select = 0 -> TX_MODE_LARGEST
  if (!key) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
  // global_motion_params() / film_grain_config() code no bits.
  return writer.bit_size() - start;
}

int neighbor_present(int r, int c, int sb_cols, int sb_rows, int dr, int dc) {
  const int nr = r + dr;
  const int nc = c + dc;
  return (nr >= 0 && nr < sb_rows && nc >= 0 && nc < sb_cols) ? 1 : 0;
}

}  // namespace

const char* bru_mode_name(BruMode mode) {
  switch (mode) {
    case BruMode::kInactive:
      return "BRU_INACTIVE";
    case BruMode::kSupport:
      return "BRU_SUPPORT";
    case BruMode::kActive:
      return "BRU_ACTIVE";
  }
  return "?";
}

int mi_cols(int frame_width) { return 2 * ((frame_width + 7) >> 3); }
int mi_rows(int frame_height) { return 2 * ((frame_height + 7) >> 3); }
int sb_cols_for(int frame_width) { return (mi_cols(frame_width) + 15) >> 4; }
int sb_rows_for(int frame_height) { return (mi_rows(frame_height) + 15) >> 4; }

bool compute_uniform_tile_layout(const SequenceConfig& sequence,
                                 const FrameConfig& frame,
                                 UniformTileLayout& out, std::string& error) {
  out = UniformTileLayout{};
  out.sb_cols = sb_cols_for(sequence.max_frame_width);
  out.sb_rows = sb_rows_for(sequence.max_frame_height);
  out.tile_cols = 1 << frame.tile_cols_log2;
  out.tile_rows = 1 << frame.tile_rows_log2;
  if (out.tile_cols < 1 || out.tile_rows < 1) {
    error = "BRU: invalid tile grid";
    return false;
  }
  const auto cols = detail::uniform_spacing(frame.tile_cols_log2,
                                            mi_cols(sequence.max_frame_width),
                                            64);
  const auto rows = detail::uniform_spacing(frame.tile_rows_log2,
                                            mi_rows(sequence.max_frame_height),
                                            64);
  out.sb_col_starts = cols.starts;
  out.sb_row_starts = rows.starts;
  out.tile_cols = cols.count;
  out.tile_rows = rows.count;
  return true;
}

std::vector<std::uint8_t> make_sequence_header_obu(
    const SequenceConfig& config) {
  BitWriter writer;
  write_sequence_header(writer, config);
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

std::size_t write_frame_header(BitWriter& writer, const SequenceConfig& config,
                               const FrameConfig& frame) {
  return write_frame_header_syntax(writer, config, frame);
}

std::size_t write_tile_group_header(BitWriter& writer,
                                    const SequenceConfig& config,
                                    const FrameConfig& frame,
                                    const TilePlan& plan) {
  const std::size_t start = writer.bit_size();
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_syntax(writer, config, frame);

  if (frame.bru_inactive) {
    // §5.19 p144: remainingBits are filled with trailing_bits() and the tile
    // payload is omitted.
    put_trailing_bits(writer);
    return writer.bit_size() - start;
  }

  const int num_tiles = (1 << frame.tile_cols_log2) *
                        (1 << frame.tile_rows_log2);
  if (num_tiles > 1) {
    writer.put_bit(false);  // tile_start_and_end_present_flag -> all tiles
  }
  if (frame.use_bru) {
    if (num_tiles > 1) {
      assert(static_cast<int>(plan.bru_tile_active.size()) == num_tiles);
      for (int i = 0; i < num_tiles; ++i) {
        writer.put_bit(plan.bru_tile_active[static_cast<std::size_t>(i)] != 0U);
      }
    }
    // NumTiles == 1 -> BruTileActives[0][0] = 1, no bit.
  }
  writer.align_to_byte();
  return writer.bit_size() - start;
}

bool encode_tile_payload(std::span<const BruMode> tile_modes,
                         int tile_sb_cols, int tile_sb_rows,
                         std::uint8_t bru_ref, bool tile_active,
                         std::vector<std::uint8_t>& out, std::string& error) {
  (void)bru_ref;
  out.clear();
  if (tile_sb_cols <= 0 || tile_sb_rows <= 0) {
    error = "BRU: empty tile";
    return false;
  }
  if (tile_active &&
      tile_modes.size() !=
          static_cast<std::size_t>(tile_sb_cols) *
              static_cast<std::size_t>(tile_sb_rows)) {
    error = "BRU: tile mode count mismatch";
    return false;
  }

  std::vector<std::uint8_t> buffer(64U * 1024U);
  av2::Encoder encoder;
  encoder.disable_cdf_update = 1;
  av2::Status status = encoder.init_symbol(buffer.data(), buffer.size());
  if (status != av2::Status::kOk) {
    error = std::string("BRU: entropy init failed: ") +
            av2::status_name(status);
    return false;
  }

  std::uint16_t bru_cdf[4];
  std::memcpy(bru_cdf, kDefaultBruModeCdf, sizeof(bru_cdf));
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

  for (int r = 0; r < tile_sb_rows; ++r) {
    for (int c = 0; c < tile_sb_cols; ++c) {
      if (!tile_active) continue;
      const BruMode mode = tile_modes[static_cast<std::size_t>(r) *
                                          tile_sb_cols + c];
      status = encoder.encode_symbol(bru_cdf, 3, static_cast<int>(mode));
      if (status != av2::Status::kOk) {
        error = std::string("BRU: bru_mode symbol failed: ") +
                av2::status_name(status);
        return false;
      }
      if (mode != BruMode::kActive) continue;

      // Active block: PARTITION_NONE at the 64x64 SB root, GLOBALMV + skip.
      // read_partition() codes do_split for an active 64x64 superblock.  The
      // partition context for a NONE-coded 64x64 root is 12 (bsize_map 3 *
      // PARTITION_PLOFFSET 4 + zero above/left bits).
      status = encoder.encode_symbol(do_split_cdf[0][12], 2, 0);
      if (status != av2::Status::kOk) {
        error = std::string("BRU: do_split symbol failed: ") +
                av2::status_name(status);
        return false;
      }
      const bool has_above =
          neighbor_present(r, c, tile_sb_cols, tile_sb_rows, -1, 0) != 0;
      const bool has_left =
          neighbor_present(r, c, tile_sb_cols, tile_sb_rows, 0, -1) != 0;
      // MiSize == ChromaMiSize for a PARTITION_NONE superblock, so
      // read_is_inter() codes a bit.  Both BRU_ACTIVE and the inferred
      // bru_mode_info() blocks are inter, hence every neighbour is NIntra=0
      // and ctx is 0.
      status = encoder.encode_symbol(is_inter_cdf[0], 2, 1);
      if (status != av2::Status::kOk) {
        error = std::string("BRU: is_inter symbol failed: ") +
                av2::status_name(status);
        return false;
      }
      const int skip_ctx = static_cast<int>(has_above) +
                           static_cast<int>(has_left);
      status = encoder.encode_symbol(skip_cdf[skip_ctx], 2, 1);
      if (status != av2::Status::kOk) {
        error = std::string("BRU: skip_flag symbol failed: ") +
                av2::status_name(status);
        return false;
      }

      int ref_zero_neighbors = 0;
      int ref_one_neighbors = 0;
      const auto note_neighbor = [&](int nr, int nc) {
        if (nr < 0 || nr >= tile_sb_rows || nc < 0 || nc >= tile_sb_cols) {
          return;
        }
        const BruMode n = tile_modes[static_cast<std::size_t>(nr) *
                                         tile_sb_cols + nc];
        // Inactive and support blocks use bru_mode_info(), whose
        // RefFrame[0] is bru_ref (the gate uses bru_ref == 0).  Active blocks
        // select the last reference (index 1) so RefFrame[0] != bru_ref, as
        // required by §6.19.7.9.
        if (n == BruMode::kActive) {
          ++ref_one_neighbors;
        } else {
          ++ref_zero_neighbors;
        }
      };
      note_neighbor(r, c - 1);
      note_neighbor(r - 1, c);
      const int ref_ctx = (ref_zero_neighbors == ref_one_neighbors)
                              ? 1
                              : (ref_zero_neighbors < ref_one_neighbors ? 0
                                                                        : 2);
      status = encoder.encode_symbol(single_ref_cdf[ref_ctx][0], 2, 0);
      if (status != av2::Status::kOk) {
        error = std::string("BRU: single_ref symbol failed: ") +
                av2::status_name(status);
        return false;
      }

      const int new_mv_ctx = static_cast<int>(has_above) +
                             static_cast<int>(has_left);
      // YMode = NEARMV + single_mode; 1 -> GLOBALMV (the enum order is
      // NEARMV, GLOBALMV, NEWMV).
      status = encoder.encode_symbol(single_mode_cdf[new_mv_ctx], 3, 1);
      if (status != av2::Status::kOk) {
        error = std::string("BRU: single_mode symbol failed: ") +
                av2::status_name(status);
        return false;
      }
    }
  }

  std::size_t bytes = 0;
  status = encoder.exit_symbol(&bytes);
  if (status != av2::Status::kOk) {
    error = std::string("BRU: entropy exit failed: ") +
            av2::status_name(status);
    return false;
  }
  out.assign(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(bytes));
  return true;
}

std::vector<std::uint8_t> make_tile_group_obu(
    const SequenceConfig& config, const FrameConfig& frame,
    const TilePlan& plan,
    std::span<const std::vector<std::uint8_t>> tile_payloads,
    std::string& error) {
  BitWriter header;
  write_tile_group_header(header, config, frame, plan);
  std::vector<std::uint8_t> payload = header.take_bytes();

  const int num_tiles = (1 << frame.tile_cols_log2) *
                        (1 << frame.tile_rows_log2);
  if (!frame.bru_inactive) {
    if (static_cast<int>(tile_payloads.size()) != num_tiles) {
      error = "BRU: tile payload count mismatch";
      return {};
    }
    const int size_bytes = frame.tile_size_bytes_minus_1 + 1;
    for (int i = 0; i < num_tiles; ++i) {
      const auto& tile = tile_payloads[static_cast<std::size_t>(i)];
      if (tile.empty()) {
        error = "BRU: empty tile payload";
        return {};
      }
      if (i + 1 < num_tiles) {
        if (tile.size() - 1U >= (1ULL << (8 * size_bytes))) {
          error = "BRU: tile payload does not fit TileSizeBytes";
          return {};
        }
        BitWriter size_writer;
        detail::put_le(size_writer, static_cast<std::uint32_t>(tile.size() - 1U),
                       size_bytes);
        const auto size_bytes_out = size_writer.take_bytes();
        payload.insert(payload.end(), size_bytes_out.begin(),
                       size_bytes_out.end());
      }
      payload.insert(payload.end(), tile.begin(), tile.end());
    }
  }

  const ObuType type =
      frame.keyframe ? ObuType::kClosedLoopKey : ObuType::kRegularTileGroup;
  return av2enc::basic::make_obu(ObuHeader{type}, payload);
}

}  // namespace av2enc::inter::bru
