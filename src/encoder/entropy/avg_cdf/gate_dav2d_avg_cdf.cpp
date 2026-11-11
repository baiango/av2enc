// dav2d gate for a 2-tile / 2-frame M2 stream with enable_avg_cdf = 1 and
// avg_cdf_type = 1.
//
// The existing M2 tiled writer forces enable_avg_cdf = 0 and therefore codes
// context_update_tile_id in tile_info().  This gate owns a small header writer
// in this directory so the stream can exercise the other §5.4.8 branch without
// editing the M2 core:
//
//   sequence_header: enable_avg_cdf = 1, avg_cdf_type = 1
//   frame_header:    no context_update_tile_id (§5.18.7.2)
//   tile payloads:   produced by the existing M2 syntax writer, frozen CDFs
//                    (disable_cdf_update = 1) so the entropy bytes are
//                    independent of the header-only change.
//
// Key frame 0 is intra DC (constant 128 with no neighbours), frame 1 is an
// inter GLOBALMV skip (copy), so the decoded Y/U/V samples must all be 128.
// The gate fails on any sample difference with the pinned decoder:
//
//   dav2d --strict 1 --threads 1 -q -i out.ivf -o decoded.y4m
//
// Returns 77 when no decoder is available (CTest-style skip).

#include "bitstream/bitwriter.h"
#include "bitstream/m0_headers.h"
#include "encoder/entropy/avg_cdf/avg_cdf_bank.h"
#include "encoder/m1/filters_stream/filters_stream.h"
#include "encoder/m2/m2_headers.h"
#include "encoder/m2/m2_syntax.h"
#include "encoder/m2/post/loop_filters.h"
#include "encoder/m2/post/tiled_frame.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using av2enc::BitWriter;
using av2enc::m0::ObuHeader;
using av2enc::m0::ObuType;
using av2enc::m2::M2Config;
using av2enc::m2::syntax::BlockSyntax;
using av2enc::tiles::TileLayout;

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

// §5.4.8 sequence_transform_quant_entropy_config(), with the average-CDF
// branch set to 1/1.  All other fields mirror the all-filters-off M2 sequence
// header used by the existing tiled gate.
std::vector<std::uint8_t> make_avg_cdf_sequence_header(const M2Config& c) {
  BitWriter writer;
  put_uvlc(writer, 0);                 // seq_header_id
  writer.put_bits(0U, 5U);             // seq_profile_idc
  writer.put_bit(false);               // single_picture_header_flag = 0
  writer.put_bits(c.seq_level_idx, 5U);
  writer.put_bit(false);               // seq_tier
  put_uvlc(writer, 0);                 // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                 // bit_depth_idc = 1 => 8-bit

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
  writer.put_bit(true);   // enable_cfl_intra
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);  // enable_mhccp
  writer.put_bit(false);  // enable_ibp

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
  writer.put_bit(true);   // enable_avg_cdf
  writer.put_bit(true);   // avg_cdf_type = 1 (average across tiles)
  writer.put_bit(false);  // separate_uv_delta_q
  writer.put_bit(true);   // equal_ac_dc_q
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q => BaseUVAcDeltaQ = 0
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  av2enc::m1::filters_stream::SequenceFilterConfig filters;
  filters.cdef_on_skip_txfm_disabled = true;
  std::string filter_error;
  if (av2enc::m1::filters_stream::write_sequence_filter_config(
          writer, filters, /*single_picture_header_flag=*/false,
          &filter_error) == 0) {
    std::cerr << "sequence filter config failed: " << filter_error << "\n";
    return {};
  }

  writer.put_bit(false);  // seq_tile_info_present_flag
  writer.put_bit(false);  // film_grain_params_present

  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::m0::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

// §5.18.7.2 tile_info() tail for enable_avg_cdf && avg_cdf_type: the
// context_update_tile_id field is absent.
bool write_avg_tile_info(BitWriter& writer, const TileLayout& layout,
                         int tile_size_bytes, std::string* error) {
  if (layout.tile_count() <= 0) {
    *error = "empty layout";
    return false;
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
    *error = "tile column log2 outside level bounds";
    return false;
  }
  for (int l = min_cols_log2; l < cols_log2; ++l) writer.put_bit(true);
  if (cols_log2 < max_cols_log2) writer.put_bit(false);

  if (min_rows_log2 < 0 || rows_log2 < min_rows_log2 ||
      rows_log2 > max_rows_log2) {
    *error = "tile row log2 outside level bounds";
    return false;
  }
  for (int l = min_rows_log2; l < rows_log2; ++l) writer.put_bit(true);
  if (rows_log2 < max_rows_log2) writer.put_bit(false);

  if (layout.tile_count() > 1) {
    if (tile_size_bytes < 1 || tile_size_bytes > 4) {
      *error = "TileSizeBytes outside 1..4";
      return false;
    }
    // context_update_tile_id is intentionally absent: §5.18.7.2 codes it only
    // when !enable_avg_cdf || !avg_cdf_type.
    writer.put_bits(static_cast<std::uint64_t>(tile_size_bytes - 1), 2U);
  }
  return true;
}

// frame_header_info() for the all-filters-off path, with disable_cdf_update=1
// and the average-CDF tile_info tail.  Returns syntax bits excluding the
// is_first_tile_group bit and byte_alignment().
bool write_avg_frame_header(BitWriter& writer, const M2Config& config,
                            const TileLayout& layout, bool keyframe,
                            std::uint32_t order_hint,
                            std::uint8_t ref_slot,
                            std::uint32_t refresh_frame_flags,
                            int tile_size_bytes, std::size_t* bits,
                            std::string* error) {
  const std::size_t start = writer.bit_size();
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header

  if (!keyframe) writer.put_bit(true);  // frame_is_inter = 1
  writer.put_bit(true);                 // immediate_output_frame

  writer.put_bit(false);  // frame_size_override_flag = 0
  writer.put_bits(order_hint, config.order_hint_bits);

  if (keyframe) {
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
    writer.put_bits(refresh_frame_flags, config.num_ref_frames);
    writer.put_bit(true);   // frame_explicit_ref_frame_map
    writer.put_bits(config.num_total_refs, 3U);
    writer.put_bits(ref_slot,
                    static_cast<unsigned>(ceil_log2_u32(config.num_ref_frames)));
    writer.put_bit(false);  // allow_intrabc
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTTAP
  }

  writer.put_bit(true);  // disable_cdf_update = 1 (frozen tile CDFs)

  if (!write_avg_tile_info(writer, layout, tile_size_bytes, error)) {
    return false;
  }

  writer.put_bits(static_cast<std::uint64_t>(config.base_q_idx), 8U);
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present
  writer.put_bit(false);  // apply_deblocking_filter[0]
  writer.put_bit(false);  // apply_deblocking_filter[1]
  writer.put_bit(false);  // tx_mode_select = 0 -> TX_MODE_LARGEST
  if (!keyframe) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
  if (bits != nullptr) *bits = writer.bit_size() - start;
  return true;
}

BlockSyntax make_root(int mi_row, int mi_col, bool inter) {
  BlockSyntax node;
  node.mi_row = static_cast<std::uint16_t>(mi_row);
  node.mi_col = static_cast<std::uint16_t>(mi_col);
  node.bsize = 12;     // BLOCK_64X64
  node.partition = 0;  // PARTITION_NONE
  node.allowed_mask = 1U << 0;
  node.flags = av2enc::m2::syntax::kFlagSbRoot;
  node.subtree = 0;
  node.is_inter = inter ? 1 : 0;
  node.skip_flag = inter ? 1 : 0;
  node.inter_mode = av2enc::m2::syntax::kInterGlobalMv;
  node.mv_y = 0;
  node.mv_x = 0;
  node.y_mode = 0;  // DC_PRED
  node.uv_mode = 0;
  node.tx_size[0] = 4;  // 64x64
  node.tx_size[1] = 3;  // 32x32 chroma
  node.tx_size[2] = 3;
  node.eob[0] = node.eob[1] = node.eob[2] = 0;
  node.levels_offset = 0;
  return node;
}

bool build_tile_payloads(const M2Config& config, const TileLayout& layout,
                         bool keyframe,
                         std::span<const BlockSyntax> blocks,
                         std::span<const std::int32_t> levels,
                         std::vector<std::vector<std::uint8_t>>* payloads,
                         std::string* error) {
  payloads->clear();
  for (int tile_index = 0; tile_index < layout.tile_count(); ++tile_index) {
    av2enc::m2::post::M2TileBlocks tile_blocks;
    if (!av2enc::m2::post::split_m2_blocks_for_tile(
            layout, tile_index, blocks, &tile_blocks, error)) {
      return false;
    }
    av2enc::m2::syntax::FramePlan local_plan;
    local_plan.width = tile_blocks.width;
    local_plan.height = tile_blocks.height;
    local_plan.sb_size = layout.sb_size;
    local_plan.inter_frame = !keyframe;
    M2Config local_config = config;
    local_config.max_frame_width = static_cast<std::uint16_t>(tile_blocks.width);
    local_config.max_frame_height =
        static_cast<std::uint16_t>(tile_blocks.height);
    std::vector<std::uint8_t> payload;
    if (!av2enc::m2::syntax::assemble_m2_frame(
            local_plan, tile_blocks.blocks, levels, local_config, payload,
            *error)) {
      return false;
    }
    if (payload.size() < 2) {
      *error = "tile payload shorter than 2 bytes";
      return false;
    }
    payloads->push_back(std::move(payload));
  }
  return true;
}

bool assemble_avg_frame_obu(const M2Config& config, const TileLayout& layout,
                            bool keyframe, std::uint32_t order_hint,
                            std::uint8_t ref_slot,
                            std::uint32_t refresh_frame_flags,
                            const std::vector<std::vector<std::uint8_t>>& payloads,
                            std::vector<std::uint8_t>* obu,
                            std::vector<std::uint32_t>* tile_bytes,
                            int* tile_size_bytes, std::string* error) {
  if (payloads.size() != static_cast<std::size_t>(layout.tile_count())) {
    *error = "payload count does not match the tile layout";
    return false;
  }
  std::uint32_t largest = 0;
  tile_bytes->clear();
  for (const auto& payload : payloads) {
    if (payload.size() > 0xFFFFFFFFull) {
      *error = "tile payload exceeds 4 GiB";
      return false;
    }
    const std::uint32_t size = static_cast<std::uint32_t>(payload.size());
    tile_bytes->push_back(size);
    largest = std::max(largest, size);
  }
  *tile_size_bytes = av2enc::tiles::tile_size_bytes_for(largest);
  if (*tile_size_bytes == 0) {
    *error = "tile size cannot be represented in 1..4 bytes";
    return false;
  }

  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  std::size_t frame_header_bits = 0;
  if (!write_avg_frame_header(writer, config, layout, keyframe, order_hint,
                              ref_slot, refresh_frame_flags, *tile_size_bytes,
                              &frame_header_bits, error)) {
    return false;
  }
  if (layout.tile_count() > 1) {
    writer.put_bit(false);  // tile_start_and_end_present_flag = 0
  }
  writer.align_to_byte();

  std::vector<std::uint8_t> tile_group = writer.take_bytes();
  for (int i = 0; i < layout.tile_count(); ++i) {
    if (i + 1 < layout.tile_count()) {
      const std::uint32_t value =
          (*tile_bytes)[static_cast<std::size_t>(i)] - 1u;
      for (int b = 0; b < *tile_size_bytes; ++b) {
        tile_group.push_back(
            static_cast<std::uint8_t>((value >> (8 * b)) & 0xFFU));
      }
    }
    const auto& payload = payloads[static_cast<std::size_t>(i)];
    tile_group.insert(tile_group.end(), payload.begin(), payload.end());
  }

  const ObuType type =
      keyframe ? ObuType::kClosedLoopKey : ObuType::kRegularTileGroup;
  *obu = av2enc::m0::make_obu(ObuHeader{type}, tile_group);
  return true;
}

std::string find_dav2d() {
  if (const char* env = std::getenv("DAV2D")) {
    if (std::filesystem::exists(env)) return env;
  }
  for (const char* candidate : {"tools/bin/dav2d", "tools/bin/dav2d-macos",
                                "dav2d", "dav2d-macos"}) {
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

bool write_ivf(const std::string& path,
               const std::vector<std::vector<std::uint8_t>>& units,
               std::uint16_t width, std::uint16_t height, std::string* error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = width;
  header.height = height;
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = static_cast<std::uint32_t>(units.size());
  if (!writer.open(path, header, *error)) return false;
  for (std::size_t i = 0; i < units.size(); ++i) {
    if (!writer.write_frame(units[i], i, *error)) return false;
  }
  return writer.close(*error);
}

bool check_decoded(const std::string& path, int width, int height,
                   std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  int frames = 0;
  long mismatches = 0;
  int max_diff = 0;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, *error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    if (frame.width() != static_cast<std::uint32_t>(width) ||
        frame.height() != static_cast<std::uint32_t>(height)) {
      *error = "decoded frame has the wrong dimensions";
      return false;
    }
    ++frames;
    const auto check_plane = [&](std::span<const std::uint8_t> plane) {
      for (std::uint8_t sample : plane) {
        const int diff = std::abs(static_cast<int>(sample) - 128);
        if (diff != 0) {
          ++mismatches;
          max_diff = std::max(max_diff, diff);
        }
      }
    };
    check_plane(frame.y());
    check_plane(frame.u());
    check_plane(frame.v());
  }
  if (frames != 2) {
    *error = "decoder produced " + std::to_string(frames) +
             " frames, expected 2";
    return false;
  }
  if (mismatches != 0) {
    *error = "decoded samples differ from the DC/GLOBALMV constant: " +
             std::to_string(mismatches) + " mismatches, max_diff " +
             std::to_string(max_diff);
    return false;
  }
  return true;
}

bool run_case(int width, int height, int cols, int rows,
              const std::string& dav2d, const std::string& out_path) {
  std::string error;
  M2Config config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = 90;

  const auto tile_plan =
      av2enc::m2::post::plan_m2_tiles(width, height, cols, rows);
  if (!tile_plan.ok) {
    std::cerr << "TILE PLAN FAIL: " << tile_plan.error << "\n";
    return false;
  }
  const TileLayout& layout = tile_plan.layout;

  const auto seq_obu = make_avg_cdf_sequence_header(config);
  if (seq_obu.empty()) {
    std::cerr << "SEQ FAIL\n";
    return false;
  }

  // Two frames, every superblock one leaf.  The key frame is DC intra; the
  // P frame is GLOBALMV skip (zero MV, copy).  All loop filters are off.
  std::vector<std::int32_t> levels;
  std::vector<BlockSyntax> key_blocks;
  std::vector<BlockSyntax> inter_blocks;
  for (int sb_row = 0; sb_row < layout.sb_rows; ++sb_row) {
    for (int sb_col = 0; sb_col < layout.sb_cols; ++sb_col) {
      key_blocks.push_back(make_root(sb_row * 16, sb_col * 16, false));
      inter_blocks.push_back(make_root(sb_row * 16, sb_col * 16, true));
    }
  }

  std::vector<std::vector<std::uint8_t>> key_payloads;
  std::vector<std::vector<std::uint8_t>> inter_payloads;
  if (!build_tile_payloads(config, layout, /*keyframe=*/true, key_blocks,
                           levels, &key_payloads, &error)) {
    std::cerr << "KEY PAYLOAD FAIL: " << error << "\n";
    return false;
  }
  if (!build_tile_payloads(config, layout, /*keyframe=*/false, inter_blocks,
                           levels, &inter_payloads, &error)) {
    std::cerr << "INTER PAYLOAD FAIL: " << error << "\n";
    return false;
  }

  std::vector<std::uint8_t> key_obu;
  std::vector<std::uint8_t> inter_obu;
  std::vector<std::uint32_t> key_bytes;
  std::vector<std::uint32_t> inter_bytes;
  int key_tile_size_bytes = 0;
  int inter_tile_size_bytes = 0;
  if (!assemble_avg_frame_obu(config, layout, /*keyframe=*/true, 0, 0,
                              (1u << config.num_ref_frames) - 1u,
                              key_payloads, &key_obu, &key_bytes,
                              &key_tile_size_bytes, &error)) {
    std::cerr << "KEY OBU FAIL: " << error << "\n";
    return false;
  }
  if (!assemble_avg_frame_obu(config, layout, /*keyframe=*/false, 1, 0, 1u,
                              inter_payloads, &inter_obu, &inter_bytes,
                              &inter_tile_size_bytes, &error)) {
    std::cerr << "INTER OBU FAIL: " << error << "\n";
    return false;
  }

  std::vector<std::vector<std::uint8_t>> units;
  {
    const auto td = av2enc::m2::make_temporal_delimiter_obu();
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), seq_obu.begin(), seq_obu.end());
    unit.insert(unit.end(), key_obu.begin(), key_obu.end());
    units.push_back(std::move(unit));
  }
  {
    const auto td = av2enc::m2::make_temporal_delimiter_obu();
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), inter_obu.begin(), inter_obu.end());
    units.push_back(std::move(unit));
  }

  if (!write_ivf(out_path, units, static_cast<std::uint16_t>(width),
                 static_cast<std::uint16_t>(height), &error)) {
    std::cerr << "IVF FAIL: " << error << "\n";
    return false;
  }

  std::cout << "  " << width << "x" << height << " " << cols << "x" << rows
            << " tiles, enable_avg_cdf=1 avg_cdf_type=1 "
            << "(context_update_tile_id absent)\n";
  std::cout << "    key_tile_bytes=";
  for (std::size_t i = 0; i < key_bytes.size(); ++i) {
    std::cout << (i != 0 ? "/" : "") << key_bytes[i];
  }
  std::cout << " key_TileSizeBytes=" << key_tile_size_bytes
            << " inter_tile_bytes=";
  for (std::size_t i = 0; i < inter_bytes.size(); ++i) {
    std::cout << (i != 0 ? "/" : "") << inter_bytes[i];
  }
  std::cout << " inter_TileSizeBytes=" << inter_tile_size_bytes << "\n";

  if (dav2d.empty()) return true;
  const std::string decoded = out_path + ".decoded.y4m";
  const std::string command = dav2d + " --strict 1 --threads 1 -q -i " +
                              out_path + " -o " + decoded;
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::cerr << "DECODE FAIL: dav2d returned " << rc << "\n";
    return false;
  }
  if (!check_decoded(decoded, width, height, &error)) {
    std::cerr << "DECODE CHECK FAIL: " << error << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_path = "avg_cdf_gate.ivf";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: gate_dav2d_avg_cdf [--out out.ivf]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }

  const std::string dav2d = find_dav2d();
  if (dav2d.empty()) {
    std::cout << "SKIP: no dav2d binary found (set DAV2D)\n";
    return 77;
  }

  const bool one_sb_per_tile =
      run_case(128, 64, 2, 1, dav2d, out_path + ".2x1.ivf");
  const bool two_sb_per_tile =
      run_case(256, 64, 2, 1, dav2d, out_path + ".4sb.ivf");
  const bool four_tiles =
      run_case(256, 128, 2, 2, dav2d, out_path + ".2x2.ivf");

  if (!one_sb_per_tile || !two_sb_per_tile || !four_tiles) {
    std::cout << "AVG-CDF DAV2D GATE FAIL\n";
    return 1;
  }
  std::cout << "PASS: 3 tiled 2-frame streams with enable_avg_cdf=1 decoded "
               "with --strict 1 --threads 1, all samples 128 (recon 0 diffs)\n";
  return 0;
}
