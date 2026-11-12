#include "tiles_stream.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/bitstream/basic_headers.h"
#include "pipeline/still/still_headers.h"
#include "pipeline/tiles/tile_group.h"

namespace av2enc::still::tiles_stream {
namespace {

using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;
using av2enc::tiles::TileLayout;
using av2enc::tiles::TileRect;

// ---------------------------------------------------------------------------
// Bit syntax helpers (§4.11.5 le(n), §4.11.6 uvlc, §5.18.1-2, §5.18.7)
// ---------------------------------------------------------------------------

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
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

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

void append_le(std::vector<std::uint8_t>& out, std::uint32_t value, int bytes) {
  for (int i = 0; i < bytes; ++i) {
    out.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xFFU));
  }
}

// ---------------------------------------------------------------------------
// M1 core mirror
// ---------------------------------------------------------------------------

StillConfig to_still_config(const Config& config) {
  StillConfig m1;
  m1.max_frame_width = static_cast<std::uint16_t>(config.width);
  m1.max_frame_height = static_cast<std::uint16_t>(config.height);
  m1.seq_level_idx = config.seq_level_idx;
  m1.base_q_idx = config.base_q_idx;
  m1.disable_cdf_update = true;
  m1.enable_intra_edge_filter = config.enable_intra_edge_filter;
  m1.enable_cfl_intra = config.enable_cfl_intra;
  m1.enable_chroma_dctonly = config.enable_chroma_dctonly;
  m1.enable_idtx_intra = false;
  m1.tx_mode_select = false;
  m1.reduced_tx_set = 0;
  m1.apply_deblock_y_vert = config.enable_deblock;
  m1.apply_deblock_y_horz = config.enable_deblock;
  m1.apply_deblock_u = config.enable_deblock;
  m1.apply_deblock_v = config.enable_deblock;
  m1.cdef.enable = false;
  m1.ccso.enable = false;
  return m1;
}

// ---------------------------------------------------------------------------
// Sequence header (§5.4)
// ---------------------------------------------------------------------------

// Mirrors src/encoder/m1/still_headers.cpp write_sequence_header(). The only
// intentional difference from the core writer is that
// disable_loopfilters_across_tiles is configurable here.
void write_sequence_header_syntax(BitWriter& writer, const Config& config) {
  put_uvlc(writer, 0);                        // seq_header_id
  writer.put_bits(0U, 5U);                    // seq_profile_idc
  writer.put_bit(true);                       // single_picture_header_flag
  writer.put_bits(config.seq_level_idx, 5U);  // seq_level_idx
  put_uvlc(writer, 0);                        // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                        // bit_depth_idc = 1 => 8-bit

  const std::uint32_t width = static_cast<std::uint32_t>(config.width);
  const std::uint32_t height = static_cast<std::uint32_t>(config.height);
  unsigned width_bits = bit_width(width - 1U);
  if (width_bits == 0) width_bits = 1;
  unsigned height_bits = bit_width(height - 1U);
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(width) - 1U, width_bits);
  writer.put_bits(static_cast<std::uint64_t>(height) - 1U, height_bits);

  writer.put_bit(false);  // seq_cropping_window_present_flag
  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(false);  // use_128x128_superblock
  writer.put_bit(false);  // enable_sdp
  writer.put_bit(false);  // enable_ext_partitions
  writer.put_bit(false);  // reduce_pb_aspect_ratio

  writer.put_bit(false);  // enable_ext_seg
  writer.put_bit(false);  // seq_seg_info_present_flag

  writer.put_bit(false);  // enable_dip
  writer.put_bit(config.enable_intra_edge_filter);
  writer.put_bit(false);  // enable_mrls
  writer.put_bit(config.enable_cfl_intra);
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);    // enable_mhccp
  writer.put_bit(false);    // enable_ibp

  writer.put_bit(false);   // enable_refmvbank
  writer.put_bit(true);    // disable_drl_reorder
  put_ns(writer, 3U, 0U);  // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);   // allow_frame_max_bvp_drl_bits
  writer.put_bit(false);   // enable_bawp

  writer.put_bit(false);  // enable_fsc
  writer.put_bit(false);  // enable_idtx_intra
  writer.put_bit(false);  // enable_intra_ist
  writer.put_bit(false);  // enable_inter_ist
  writer.put_bit(config.enable_chroma_dctonly);
  writer.put_bit(false);     // reduced_tx_part_set
  writer.put_bit(false);     // enable_cctx
  writer.put_bit(false);     // enable_tcq
  writer.put_bit(false);     // enable_parity_hiding
  writer.put_bit(false);     // separate_uv_delta_q
  writer.put_bit(true);      // equal_ac_dc_q
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q => BaseUVAcDeltaQ = 0
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  writer.put_bit(config.disable_loopfilters_across_tiles);
  writer.put_bit(false);    // enable_cdef
  writer.put_bit(false);    // enable_gdf
  writer.put_bit(false);    // enable_restoration
  writer.put_bit(false);    // enable_ccso
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2

  writer.put_bit(false);  // seq_tile_info_present_flag
  writer.put_bit(false);  // film_grain_params_present
}

// ---------------------------------------------------------------------------
// Frame header (§5.18)
// ---------------------------------------------------------------------------

void write_frame_header_syntax(BitWriter& writer, const Config& config, const TileLayout& layout,
                               int tile_size_bytes) {
  put_uvlc(writer, 0);    // cur_mfh_id
  put_uvlc(writer, 0);    // seq_header_id_in_frame_header
  writer.put_bit(false);  // allow_screen_content_tools
  writer.put_bit(false);  // allow_intrabc
  writer.put_bit(true);   // disable_cdf_update: frozen per-tile CDFs
  write_tile_info(writer, layout, tile_size_bytes);

  writer.put_bits(config.base_q_idx, 8U);  // base_q_idx
  writer.put_bit(false);                   // segmentation_enabled
  writer.put_bit(false);                   // using_qmatrix
  writer.put_bit(false);                   // delta_q_present

  // deblocking_filter_params() (§5.18.5.2), same bit pattern as
  // StillConfig: chroma flags only appear when a luma direction is enabled.
  writer.put_bit(config.enable_deblock);  // apply_deblocking_filter[0]
  writer.put_bit(config.enable_deblock);  // apply_deblocking_filter[1]
  if (config.enable_deblock) {
    writer.put_bit(config.enable_deblock);  // apply_deblocking_filter[2]
    writer.put_bit(config.enable_deblock);  // apply_deblocking_filter[3]
    for (int i = 0; i < 4; ++i) {
      writer.put_bit(false);  // df_delta_q_present[i] = 0
    }
  }
  writer.put_bit(false);    // tx_mode_select = TX_MODE_LARGEST
  writer.put_bits(0U, 2U);  // reduced_tx_set
}

// ---------------------------------------------------------------------------
// Source helpers
// ---------------------------------------------------------------------------

bool crop_plane(std::span<const std::uint8_t> plane, int stride, int x0, int y0, int width,
                int height, std::vector<std::uint8_t>* out, std::string* error) {
  if (x0 < 0 || y0 < 0 || width < 0 || height < 0) {
    *error = "negative crop rectangle";
    return false;
  }
  const std::size_t need = static_cast<std::size_t>(y0 + height) * static_cast<std::size_t>(stride);
  if (plane.size() < need) {
    *error = "source plane is too small for a tile crop";
    return false;
  }
  out->resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    std::memcpy(out->data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(width),
                plane.data() + static_cast<std::size_t>(y0 + y) * static_cast<std::size_t>(stride) +
                    static_cast<std::size_t>(x0),
                static_cast<std::size_t>(width));
  }
  return true;
}

void paste_plane(const std::vector<std::uint8_t>& tile, int tile_w, int tile_h,
                 std::vector<std::uint8_t>* full, int full_w, int x0, int y0) {
  for (int y = 0; y < tile_h; ++y) {
    std::memcpy(full->data() + static_cast<std::size_t>(y0 + y) * static_cast<std::size_t>(full_w) +
                    static_cast<std::size_t>(x0),
                tile.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(tile_w),
                static_cast<std::size_t>(tile_w));
  }
}

// OBU splitting for a single-frame temporal unit: Annex B.2 leb128 length plus
// a 1-byte obu_header (no extension). Only the CLK payload is needed.
bool extract_clk_payload(std::span<const std::uint8_t> data, std::vector<std::uint8_t>* clk_payload,
                         std::string* error) {
  std::size_t off = 0;
  bool found = false;
  while (off < data.size()) {
    std::uint32_t size = 0;
    unsigned shift = 0;
    while (true) {
      if (off >= data.size() || shift > 28) {
        *error = "truncated OBU size";
        return false;
      }
      const std::uint8_t byte = data[off++];
      size |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
      shift += 7;
      if ((byte & 0x80U) == 0) break;
    }
    if (off + size > data.size() || size < 1) {
      *error = "truncated OBU payload";
      return false;
    }
    const std::uint8_t header = data[off];
    const int type = (header >> 2) & 0x1f;
    const std::span<const std::uint8_t> payload = data.subspan(off + 1, size - 1);
    if (type == static_cast<int>(ObuType::kClosedLoopKey)) {
      clk_payload->assign(payload.begin(), payload.end());
      found = true;
    }
    off += size;
  }
  if (!found) {
    *error = "M1 frame has no CLK OBU";
    return false;
  }
  return true;
}

bool validate_config(const Config& config, std::string* error) {
  if (config.width <= 0 || config.height <= 0) {
    *error = "frame dimensions must be positive";
    return false;
  }
  if (config.width > 65535 || config.height > 65535) {
    *error = "frame dimensions exceed max_frame_width/height f(16)";
    return false;
  }
  if ((config.width & 1) != 0 || (config.height & 1) != 0) {
    *error = "M1 4:2:0 requires even frame dimensions";
    return false;
  }
  if (config.base_q_idx == 0) {
    *error = "M1 does not support lossless base_q_idx = 0";
    return false;
  }
  if (config.sb_size != tiles::kSuperblockPixels64) {
    *error = "M1 core only implements 64x64 superblocks";
    return false;
  }
  if (!config.enable_chroma_dctonly) {
    *error = "M1 core always signals enable_chroma_dctonly = 1";
    return false;
  }
  if (config.seq_level_idx > 31U) {
    *error = "seq_level_idx outside 0..31";
    return false;
  }
  return true;
}

}  // namespace

bool build_layout(int width, int height, tiles::TileLayoutPreset preset, int sb_size,
                  int seq_level_idx, TileLayout* layout, std::string* error) {
  if (layout == nullptr || error == nullptr) return false;
  const tiles::TileLayoutDims dims = tiles::preset_dims(preset);
  const tiles::LayoutResult planned =
      tiles::plan_tile_layout(width, height, dims.cols, dims.rows, sb_size, seq_level_idx, 0);
  if (!planned.ok()) {
    *error = "layout " + std::string(tiles::preset_name(preset)) + " rejected: " + planned.error;
    return false;
  }
  *layout = planned.layout;
  return true;
}

int select_tile_size_bytes(std::span<const std::uint32_t> sizes, std::string* error) {
  if (sizes.empty()) {
    if (error != nullptr) *error = "a tile group needs at least one payload";
    return 0;
  }
  std::uint32_t largest = 0;
  for (const std::uint32_t size : sizes) {
    if (size < 2U) {
      if (error != nullptr) {
        *error = "every tile payload must be at least 2 bytes";
      }
      return 0;
    }
    largest = std::max(largest, size);
  }
  const int bytes = tiles::tile_size_bytes_for(largest);
  if (bytes == 0 && error != nullptr) {
    *error = "tile size cannot be represented in a 1..4 byte field";
  }
  return bytes;
}

void write_tile_info(BitWriter& writer, const TileLayout& layout, int tile_size_bytes) {
  writer.put_bit(true);  // uniform_tile_spacing_flag

  int cols_log2 = layout.min_log2_tile_cols;
  while (cols_log2 < layout.max_log2_tile_cols) {
    const bool increment = cols_log2 < layout.cols_log2;
    writer.put_bit(increment);  // increment_tile_cols_log2
    if (!increment) break;
    ++cols_log2;
  }

  // §5.18.7.3 derives minLog2TileRows from minLog2Tiles and the chosen column
  // count; the layout stores min_log2_tiles and max_log2_tile_rows.
  int rows_log2 = std::max(layout.min_log2_tiles - layout.cols_log2, 0);
  while (rows_log2 < layout.max_log2_tile_rows) {
    const bool increment = rows_log2 < layout.rows_log2;
    writer.put_bit(increment);  // increment_tile_rows_log2
    if (!increment) break;
    ++rows_log2;
  }

  if (layout.tile_count() > 1) {
    // off-by-one: TileSizeBytes = tile_size_bytes_minus_1 + 1.
    assert(tile_size_bytes >= 1 && tile_size_bytes <= 4);
    writer.put_bits(static_cast<std::uint64_t>(tile_size_bytes - 1), 2U);
  }
}

std::size_t write_frame_header(BitWriter& writer, const Config& config, const TileLayout& layout,
                               int tile_size_bytes) {
  const std::size_t start = writer.bit_size();
  write_frame_header_syntax(writer, config, layout, tile_size_bytes);
  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_sequence_header_obu(const Config& config) {
  BitWriter writer;
  write_sequence_header_syntax(writer, config);
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader}, writer.bytes());
}

bool make_tile_group_payload(const TileLayout& layout, const Config& config,
                             std::span<const TilePayloadView> payloads,
                             std::vector<std::uint8_t>* clk_payload, int* tile_size_bytes,
                             std::string* error) {
  if (clk_payload == nullptr || tile_size_bytes == nullptr || error == nullptr) {
    return false;
  }
  if (layout.empty()) {
    *error = "tile layout is empty";
    return false;
  }
  if (payloads.size() != static_cast<std::size_t>(layout.tile_count())) {
    *error = "tile payload count does not match the layout";
    return false;
  }

  std::vector<std::uint32_t> sizes;
  sizes.reserve(payloads.size());
  for (const TilePayloadView& payload : payloads) {
    if (payload.bytes.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      *error = "a tile payload exceeds the 32-bit size field";
      return false;
    }
    sizes.push_back(static_cast<std::uint32_t>(payload.bytes.size()));
  }
  const int bytes = select_tile_size_bytes(sizes, error);
  if (bytes == 0) return false;

  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_syntax(writer, config, layout, bytes);
  if (layout.tile_count() > 1) {
    writer.put_bit(false);  // tile_start_and_end_present_flag = 0
  }
  writer.align_to_byte();

  std::vector<std::uint8_t> payload = writer.take_bytes();
  for (int i = 0; i < layout.tile_count(); ++i) {
    const bool last = i == layout.tile_count() - 1;
    if (!last) {
      // §5.20.1: le(TileSizeBytes) then the tile payload, interleaved.
      append_le(payload, sizes[static_cast<std::size_t>(i)] - 1U, bytes);
    }
    const std::span<const std::uint8_t> tile = payloads[static_cast<std::size_t>(i)].bytes;
    payload.insert(payload.end(), tile.begin(), tile.end());
  }

  *clk_payload = std::move(payload);
  *tile_size_bytes = bytes;
  return true;
}

bool core_header_parity(const Config& config, std::string* error) {
  if (error == nullptr) return false;
  *error = "";

  TileLayout one_by_one;
  if (!build_layout(config.width, config.height, tiles::TileLayoutPreset::k1x1, config.sb_size,
                    config.seq_level_idx, &one_by_one, error)) {
    return false;
  }

  const StillConfig m1 = to_still_config(config);

  BitWriter reference;
  av2enc::still::write_frame_header(reference, m1);
  reference.align_to_byte();
  BitWriter writer;
  write_frame_header_syntax(writer, config, one_by_one, 0);
  writer.align_to_byte();
  if (reference.bit_size() != writer.bit_size() || reference.take_bytes() != writer.take_bytes()) {
    *error = "1x1 frame header does not match av2enc::still::write_frame_header()";
    return false;
  }

  Config sequence_config = config;
  sequence_config.disable_loopfilters_across_tiles = false;
  if (av2enc::still::make_sequence_header_obu(m1) != make_sequence_header_obu(sequence_config)) {
    *error = "sequence header does not match av2enc::still::make_sequence_header_obu()";
    return false;
  }
  return true;
}

bool encode(const StillPicture& source, const Config& config, Frame* frame, std::string* error) {
  if (frame == nullptr || error == nullptr) return false;
  error->clear();
  *frame = Frame{};

  if (!validate_config(config, error)) return false;
  if (source.width != static_cast<std::uint32_t>(config.width) ||
      source.height != static_cast<std::uint32_t>(config.height)) {
    *error = "Config dimensions do not match StillPicture";
    return false;
  }

  // Unit check first: a core header change must fail loudly, not emit a stream
  // the decoder disagrees with.
  if (!core_header_parity(config, error)) {
    *error = "tiled header self-check failed: " + *error;
    return false;
  }

  TileLayout layout;
  if (!build_layout(config.width, config.height, config.preset, config.sb_size,
                    config.seq_level_idx, &layout, error)) {
    return false;
  }
  if (layout.width != config.width || layout.height != config.height) {
    *error = "planned layout dimensions do not match the frame";
    return false;
  }

  const std::size_t luma_size =
      static_cast<std::size_t>(config.width) * static_cast<std::size_t>(config.height);
  const std::size_t chroma_size =
      static_cast<std::size_t>(config.width / 2) * static_cast<std::size_t>(config.height / 2);
  if (source.y.size() < luma_size || source.u.size() < chroma_size ||
      source.v.size() < chroma_size) {
    *error = "source planes are too small for the configured frame";
    return false;
  }

  StillOptions options;
  options.base_q_idx = config.base_q_idx;
  options.enable_cfl = config.enable_cfl_intra;
  options.enable_deblock = config.enable_deblock;
  options.enable_intra_edge_filter = config.enable_intra_edge_filter;
  options.adaptive_cdf = false;  // frozen per-tile CDF reset
  options.min_block_size = 8;
  options.max_partition_depth = 4;
  options.force_single_block = false;
  options.force_dc = false;
  options.luma_dc_only = true;
  options.uv_full = false;
  options.rdo_intra = false;
  options.rdoq = false;
  options.tx_type = false;
  options.enable_cdef = false;
  options.enable_ccso = false;

  std::vector<Tile> tiles;
  tiles.reserve(static_cast<std::size_t>(layout.tile_count()));
  for (const TileRect& rect : layout.tiles) {
    Tile tile;
    tile.index = rect.index;
    tile.row = rect.row;
    tile.col = rect.col;
    tile.x0 = rect.x0;
    tile.y0 = rect.y0;
    tile.x1 = rect.x1;
    tile.y1 = rect.y1;
    tile.sb_col_start = rect.sb_col_start;
    tile.sb_row_start = rect.sb_row_start;
    tile.sb_cols = rect.sb_cols;
    tile.sb_rows = rect.sb_rows;

    const int tile_w = rect.pixel_width();
    const int tile_h = rect.pixel_height();
    if (tile_w <= 0 || tile_h <= 0) {
      *error = "tile has empty visible geometry";
      return false;
    }
    if ((tile_w & 1) != 0 || (tile_h & 1) != 0) {
      *error = "tile dimensions must be even for 4:2:0";
      return false;
    }
    tile.local_pixel_width = tile_w;
    tile.local_pixel_height = tile_h;
    tile.local_sb_cols = tiles::sb_cols(tile_w, config.sb_size);
    tile.local_sb_rows = tiles::sb_rows(tile_h, config.sb_size);
    if (tile.local_sb_cols != rect.sb_cols || tile.local_sb_rows != rect.sb_rows) {
      *error = "tile-local superblock geometry disagrees with the plan";
      return false;
    }

    const int chroma_w = tile_w / 2;
    const int chroma_h = tile_h / 2;
    std::vector<std::uint8_t> y;
    std::vector<std::uint8_t> u;
    std::vector<std::uint8_t> v;
    if (!crop_plane(source.y, config.width, rect.x0, rect.y0, tile_w, tile_h, &y, error)) {
      return false;
    }
    if (!crop_plane(source.u, config.width / 2, rect.x0 / 2, rect.y0 / 2, chroma_w, chroma_h, &u,
                    error)) {
      return false;
    }
    if (!crop_plane(source.v, config.width / 2, rect.x0 / 2, rect.y0 / 2, chroma_w, chroma_h, &v,
                    error)) {
      return false;
    }

    StillPicture picture;
    picture.width = static_cast<std::uint32_t>(tile_w);
    picture.height = static_cast<std::uint32_t>(tile_h);
    picture.y = y;
    picture.u = u;
    picture.v = v;

    StillResult result;
    if (!av2enc::still::encode_still_picture(picture, options, result, *error)) {
      *error = "tile " + std::to_string(rect.index) + " encode failed: " + *error;
      return false;
    }
    if (result.tile_bytes == 0) {
      *error = "tile " + std::to_string(rect.index) + " produced no payload";
      return false;
    }

    std::vector<std::uint8_t> clk_payload;
    if (!extract_clk_payload(result.frame, &clk_payload, error)) {
      *error = "tile " + std::to_string(rect.index) + ": " + *error;
      return false;
    }
    if (clk_payload.size() < result.tile_bytes) {
      *error = "tile payload is larger than the CLK OBU";
      return false;
    }
    const std::size_t offset = clk_payload.size() - result.tile_bytes;
    tile.payload.assign(clk_payload.begin() + static_cast<std::ptrdiff_t>(offset),
                        clk_payload.end());

    const std::size_t expect_y =
        static_cast<std::size_t>(tile_w) * static_cast<std::size_t>(tile_h);
    const std::size_t expect_c =
        static_cast<std::size_t>(chroma_w) * static_cast<std::size_t>(chroma_h);
    if (result.recon_y.size() != expect_y || result.recon_u.size() != expect_c ||
        result.recon_v.size() != expect_c) {
      *error = "tile reconstruction size does not match the tile rectangle";
      return false;
    }
    tile.recon_y = std::move(result.recon_y);
    tile.recon_u = std::move(result.recon_u);
    tile.recon_v = std::move(result.recon_v);
    tiles.push_back(std::move(tile));
  }

  std::vector<TilePayloadView> views;
  views.reserve(tiles.size());
  for (const Tile& tile : tiles) {
    views.push_back(TilePayloadView{std::span<const std::uint8_t>(tile.payload)});
  }

  std::vector<std::uint8_t> clk_payload;
  int tile_size_bytes = 0;
  if (!make_tile_group_payload(layout, config, views, &clk_payload, &tile_size_bytes, error)) {
    return false;
  }

  const std::vector<std::uint8_t> clk =
      av2enc::basic::make_obu(ObuHeader{ObuType::kClosedLoopKey}, clk_payload);
  const std::vector<std::uint8_t> td =
      av2enc::basic::make_obu(ObuHeader{ObuType::kTemporalDelimiter}, std::span<const std::uint8_t>{});
  const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);

  std::vector<std::uint8_t> obus;
  obus.reserve(td.size() + seq.size() + clk.size());
  obus.insert(obus.end(), td.begin(), td.end());
  obus.insert(obus.end(), seq.begin(), seq.end());
  obus.insert(obus.end(), clk.begin(), clk.end());

  frame->layout = std::move(layout);
  frame->tiles = std::move(tiles);
  frame->tile_size_bytes = tile_size_bytes;
  frame->clk_payload = std::move(clk_payload);
  frame->frame = std::move(obus);
  return true;
}

bool assemble_reconstruction(const Frame& frame, Reconstruction* out, std::string* error) {
  if (out == nullptr || error == nullptr) return false;
  const TileLayout& layout = frame.layout;
  if (layout.width <= 0 || layout.height <= 0) {
    *error = "layout has no frame dimensions";
    return false;
  }
  if (layout.tiles.size() != frame.tiles.size()) {
    *error = "frame tiles do not match the layout";
    return false;
  }
  const int width = layout.width;
  const int height = layout.height;
  const int chroma_w = width / 2;
  const int chroma_h = height / 2;
  out->width = width;
  out->height = height;
  out->y.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
  out->u.assign(static_cast<std::size_t>(chroma_w) * static_cast<std::size_t>(chroma_h), 0);
  out->v.assign(out->u.size(), 0);

  for (const Tile& tile : frame.tiles) {
    const TileRect* rect = layout.at_index(tile.index);
    if (rect == nullptr) {
      *error = "tile index is outside the layout";
      return false;
    }
    const int tile_w = rect->pixel_width();
    const int tile_h = rect->pixel_height();
    const int tile_cw = tile_w / 2;
    const int tile_ch = tile_h / 2;
    if (tile.recon_y.size() !=
            static_cast<std::size_t>(tile_w) * static_cast<std::size_t>(tile_h) ||
        tile.recon_u.size() !=
            static_cast<std::size_t>(tile_cw) * static_cast<std::size_t>(tile_ch) ||
        tile.recon_v.size() != tile.recon_u.size()) {
      *error = "tile reconstruction size does not match the tile rectangle";
      return false;
    }
    paste_plane(tile.recon_y, tile_w, tile_h, &out->y, width, rect->x0, rect->y0);
    paste_plane(tile.recon_u, tile_cw, tile_ch, &out->u, chroma_w, rect->x0 / 2, rect->y0 / 2);
    paste_plane(tile.recon_v, tile_cw, tile_ch, &out->v, chroma_w, rect->x0 / 2, rect->y0 / 2);
  }
  return true;
}

}  // namespace av2enc::still::tiles_stream
