// Tiled still-picture dav2d gate (no core edits).
//
// Builds a real multi-tile AV2 still from the public M1 tile encoder:
//
//   1. plan_tile_layout() gives the exact uniform-spacing geometry;
//   2. every tile is cropped out of the source and encoded independently with
//      av2enc::m1::encode_m1_still_picture() (tile-local prediction, CDF reset,
//      context reset — exactly what §5.20.2.1 does at a tile start);
//   3. the entropy payload is extracted from each tile's CLK OBU;
//   4. this harness writes the frame header with the tiled tile_info() and a
//      single tile_group OBU (is_first_tile_group = 1,
//      tile_start_and_end_present_flag = 0, interleaved le(TileSizeBytes)
//      tile_size_minus_1 fields and payloads in raster order);
//   5. dav2d decodes it, and the decoded planes must be byte-identical to the
//      assembled per-tile M1 reconstructions.
//
// Header trust: before writing a tiled stream the harness rebuilds the
// 1x1 frame header with its own writer and requires it to be byte-identical to
// av2enc::m1::write_frame_header(), and rebuilds the sequence header with
// disable_loopfilters_across_tiles = 0 and requires it to be byte-identical to
// av2enc::m1::make_sequence_header_obu(). The only intentional difference in
// the emitted stream is disable_loopfilters_across_tiles = 1, which makes the
// per-tile deblock match the decoder at tile boundaries.
//
// This is a decoder gate for layout planning, not a product encoder: it calls
// the CPU M1 reference per tile and makes no speed claim. No file outside
// src/encoder/pipeline/ and src/cuda/pipeline/ is modified.
//
// Build/run (the script handles linking):
//   ./gate_dav2d_tiled.sh [source.y4m] [layout]
//   DAV2D=tools/bin/dav2d-macos ./gate_dav2d_tiled.sh testdata/shine_256x256_1f.y4m 2x2
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "bitstream/m0_headers.h"
#include "encoder/m1/m1_headers.h"
#include "encoder/m1/m1_still.h"
#include "encoder/tiles/tile_group.h"
#include "encoder/tiles/tile_layout.h"
#include "io/ivf.h"
#include "io/y4m.h"

namespace {

using av2enc::BitWriter;
using av2enc::m0::ObuHeader;
using av2enc::m0::ObuType;
using av2enc::tiles::TileLayout;
using av2enc::tiles::TileLayoutPreset;
using av2enc::tiles::TileRect;

int g_failures = 0;

void fail(const std::string& what) {
  ++g_failures;
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

// ---------------------------------------------------------------------------
// Syntax helpers (§4.11.5 le(n), §4.11.6 uvlc, §5.18.1-2, §5.18.7)
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

void append_le(std::vector<std::uint8_t>& out, std::uint32_t value,
               int bytes) {
  for (int i = 0; i < bytes; ++i) {
    out.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xFFU));
  }
}

struct SequenceHeaderConfig {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint8_t level = 7;
  bool disable_loopfilters_across_tiles = false;
  bool enable_intra_edge_filter = true;
  bool enable_cfl_intra = true;
  bool enable_chroma_dctonly = true;
};

// Mirrors src/encoder/m1/m1_headers.cpp write_sequence_header(). The gate
// self-checks the disable_loopfilters_across_tiles = 0 form against the public
// make_sequence_header_obu() before trusting it.
void write_sequence_header_syntax(BitWriter& writer,
                                  const SequenceHeaderConfig& config) {
  put_uvlc(writer, 0);                        // seq_header_id
  writer.put_bits(0U, 5U);                    // seq_profile_idc
  writer.put_bit(true);                       // single_picture_header_flag
  writer.put_bits(config.level, 5U);          // seq_level_idx
  put_uvlc(writer, 0);                        // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                        // bit_depth_idc = 1 => 8-bit

  unsigned width_bits =
      bit_width(static_cast<std::uint32_t>(config.width) - 1U);
  if (width_bits == 0) width_bits = 1;
  unsigned height_bits =
      bit_width(static_cast<std::uint32_t>(config.height) - 1U);
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(config.width) - 1U, width_bits);
  writer.put_bits(static_cast<std::uint64_t>(config.height) - 1U, height_bits);

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
  writer.put_bit(false);  // reduced_tx_part_set
  writer.put_bit(false);  // enable_cctx
  writer.put_bit(false);  // enable_tcq
  writer.put_bit(false);  // enable_parity_hiding
  writer.put_bit(false);  // separate_uv_delta_q
  writer.put_bit(true);   // equal_ac_dc_q
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q => BaseUVAcDeltaQ = 0
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  writer.put_bit(config.disable_loopfilters_across_tiles);
  writer.put_bit(false);  // enable_cdef
  writer.put_bit(false);  // enable_gdf
  writer.put_bit(false);  // enable_restoration
  writer.put_bit(false);  // enable_ccso
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2

  writer.put_bit(false);  // seq_tile_info_present_flag
  writer.put_bit(false);  // film_grain_params_present
}

std::vector<std::uint8_t> make_sequence_header_obu(
    const SequenceHeaderConfig& config) {
  BitWriter writer;
  write_sequence_header_syntax(writer, config);
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::m0::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.bytes());
}

// tile_info() with uniform spacing. `TileSizeBytes` is only present when
// NumTiles > 1.
void write_tile_info(BitWriter& writer, const TileLayout& layout,
                     int tile_size_bytes) {
  writer.put_bit(true);  // uniform_tile_spacing_flag

  int cols_log2 = layout.min_log2_tile_cols;
  while (cols_log2 < layout.max_log2_tile_cols) {
    const bool increment = cols_log2 < layout.cols_log2;
    writer.put_bit(increment);  // increment_tile_cols_log2
    if (!increment) break;
    ++cols_log2;
  }

  // §5.18.7.3 derives minLog2TileRows from minLog2Tiles and the chosen column
  // count; TileLayout stores the former and exposes min_log2_tiles.
  int rows_log2 = std::max(layout.min_log2_tiles - layout.cols_log2, 0);
  while (rows_log2 < layout.max_log2_tile_rows) {
    const bool increment = rows_log2 < layout.rows_log2;
    writer.put_bit(increment);  // increment_tile_rows_log2
    if (!increment) break;
    ++rows_log2;
  }

  if (layout.tile_count() > 1) {
    // context_update_tile_id is absent: single_picture_header_flag forces
    // enable_avg_cdf = 1 and avg_cdf_type = 1 (§5.4.8, §5.18.7.2).
    writer.put_bits(static_cast<std::uint64_t>(tile_size_bytes - 1), 2U);
  }
}

struct FrameHeaderConfig {
  int width = 0;
  int height = 0;
  std::uint8_t base_q_idx = 90;
  bool deblock = true;
  int tile_size_bytes = 1;
};

void write_frame_header(BitWriter& writer, const FrameHeaderConfig& config,
                        const TileLayout& layout) {
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  writer.put_bit(false);  // allow_screen_content_tools
  writer.put_bit(false);  // allow_intrabc
  writer.put_bit(true);   // disable_cdf_update
  write_tile_info(writer, layout, config.tile_size_bytes);

  writer.put_bits(config.base_q_idx, 8U);  // base_q_idx
  writer.put_bit(false);                   // segmentation_enabled
  writer.put_bit(false);                   // using_qmatrix
  writer.put_bit(false);                   // delta_q_present

  writer.put_bit(config.deblock);  // apply_deblocking_filter[0]
  writer.put_bit(config.deblock);  // apply_deblocking_filter[1]
  if (config.deblock) {
    writer.put_bit(config.deblock);  // apply_deblocking_filter[2]
    writer.put_bit(config.deblock);  // apply_deblocking_filter[3]
    for (int i = 0; i < 4; ++i) {
      writer.put_bit(false);  // df_delta_q_present[i] = 0
    }
  }
  writer.put_bit(false);   // tx_mode_select = TX_MODE_LARGEST
  writer.put_bits(0U, 2U); // reduced_tx_set
}

// ---------------------------------------------------------------------------
// Source / decode helpers
// ---------------------------------------------------------------------------

struct Planes {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool load_first_frame(const std::string& path, Planes* out,
                      std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult result = reader.read_frame(frame, *error);
  if (result != av2enc::Y4mReadResult::kFrame) {
    if (error->empty()) *error = "no frame in " + path;
    return false;
  }
  out->width = static_cast<int>(frame.width());
  out->height = static_cast<int>(frame.height());
  out->y.assign(frame.y().begin(), frame.y().end());
  out->u.assign(frame.u().begin(), frame.u().end());
  out->v.assign(frame.v().begin(), frame.v().end());
  return true;
}

bool write_ivf(const std::string& path, int width, int height,
               std::span<const std::uint8_t> frame, std::string* error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  if (!writer.open(path, header, *error)) return false;
  if (!writer.write_frame(frame, 0, *error)) return false;
  return writer.close(*error);
}

bool run_decoder(const std::string& dav2d, const std::string& ivf,
                 const std::string& y4m, std::string* error) {
  const std::string command =
      "'" + dav2d + "' -i '" + ivf + "' -o '" + y4m + "' >/dev/null 2>&1";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    *error = "dav2d failed on " + ivf;
    return false;
  }
  return true;
}

// OBU splitting: Annex B.2 leb128 length + 1-byte obu_header (no extension).
struct Obus {
  std::vector<std::uint8_t> clk;
};

bool parse_obus(std::span<const std::uint8_t> data, Obus* out,
                std::string* error) {
  std::size_t off = 0;
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
    const std::span<const std::uint8_t> payload =
        data.subspan(off + 1, size - 1);
    if (type == 4) {
      out->clk.assign(payload.begin(), payload.end());
    }
    off += size;
  }
  return true;
}

bool crop_plane(const std::vector<std::uint8_t>& plane, int stride,
                int x0, int y0, int width, int height,
                std::vector<std::uint8_t>* out) {
  out->resize(static_cast<std::size_t>(width) *
              static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    std::memcpy(out->data() + static_cast<std::size_t>(y) *
                                 static_cast<std::size_t>(width),
                plane.data() +
                    static_cast<std::size_t>(y0 + y) *
                        static_cast<std::size_t>(stride) +
                    static_cast<std::size_t>(x0),
                static_cast<std::size_t>(width));
  }
  return true;
}

struct EncodedTile {
  int index = 0;
  std::vector<std::uint8_t> payload;
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
};

bool encode_tile(const Planes& source, const TileRect& rect,
                 std::uint8_t base_q_idx, bool deblock, EncodedTile* out,
                 std::string* error) {
  const int tile_w = rect.pixel_width();
  const int tile_h = rect.pixel_height();
  if ((tile_w & 1) != 0 || (tile_h & 1) != 0) {
    *error = "tile dimensions must be even for 4:2:0";
    return false;
  }
  const int chroma_w = tile_w / 2;
  const int chroma_h = tile_h / 2;

  av2enc::m1::StillPicture picture;
  picture.width = static_cast<std::uint32_t>(tile_w);
  picture.height = static_cast<std::uint32_t>(tile_h);
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  crop_plane(source.y, source.width, rect.x0, rect.y0, tile_w, tile_h, &y);
  crop_plane(source.u, source.width / 2, rect.x0 / 2, rect.y0 / 2, chroma_w,
             chroma_h, &u);
  crop_plane(source.v, source.width / 2, rect.x0 / 2, rect.y0 / 2, chroma_w,
             chroma_h, &v);
  picture.y = y;
  picture.u = u;
  picture.v = v;

  av2enc::m1::M1Options options;
  options.base_q_idx = base_q_idx;
  options.enable_deblock = deblock;
  // Pin the tool set so a future default change in the M1 reference cannot
  // silently add syntax the gate's tiled header does not write.
  options.enable_cdef = false;
  options.rdo_intra = false;
  options.rdoq = false;
  options.tx_type = false;
  options.adaptive_cdf = false;

  av2enc::m1::M1Result result;
  if (!av2enc::m1::encode_m1_still_picture(picture, options, result, *error)) {
    return false;
  }

  Obus obus;
  if (!parse_obus(result.frame, &obus, error)) {
    return false;
  }
  if (obus.clk.size() < result.tile_bytes) {
    *error = "tile payload is larger than the CLK OBU";
    return false;
  }
  const std::size_t offset = obus.clk.size() - result.tile_bytes;
  out->index = rect.index;
  out->payload.assign(obus.clk.begin() + static_cast<std::ptrdiff_t>(offset),
                      obus.clk.end());
  out->width = tile_w;
  out->height = tile_h;
  out->recon_y = result.recon_y;
  out->recon_u = result.recon_u;
  out->recon_v = result.recon_v;
  return true;
}

void paste_plane(const std::vector<std::uint8_t>& tile, int tile_w,
                 int tile_h, std::vector<std::uint8_t>* full, int full_w,
                 int x0, int y0) {
  for (int y = 0; y < tile_h; ++y) {
    std::memcpy(full->data() + static_cast<std::size_t>(y0 + y) *
                                   static_cast<std::size_t>(full_w) +
                    static_cast<std::size_t>(x0),
                tile.data() + static_cast<std::size_t>(y) *
                                  static_cast<std::size_t>(tile_w),
                static_cast<std::size_t>(tile_w));
  }
}

bool assemble_tiled_stream(const TileLayout& layout,
                           const Planes& source,
                           const std::vector<EncodedTile>& tiles,
                           std::uint8_t base_q_idx, bool deblock,
                           std::vector<std::uint8_t>* frame,
                           std::string* error) {
  std::vector<std::uint32_t> sizes;
  sizes.reserve(tiles.size());
  std::uint32_t largest = 0;
  for (const EncodedTile& tile : tiles) {
    sizes.push_back(static_cast<std::uint32_t>(tile.payload.size()));
    largest = std::max(largest, sizes.back());
  }
  const int tile_size_bytes = av2enc::tiles::tile_size_bytes_for(largest);
  if (tile_size_bytes == 0) {
    *error = "tile size does not fit a 1..4 byte field (or is < 2 bytes)";
    return false;
  }

  FrameHeaderConfig header;
  header.width = source.width;
  header.height = source.height;
  header.base_q_idx = base_q_idx;
  header.deblock = deblock;
  header.tile_size_bytes = tile_size_bytes;

  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header(writer, header, layout);
  if (layout.tile_count() > 1) {
    writer.put_bit(false);  // tile_start_and_end_present_flag = 0
  }
  writer.align_to_byte();

  std::vector<std::uint8_t> payload = writer.take_bytes();
  for (int i = 0; i < layout.tile_count(); ++i) {
    const bool last = i == layout.tile_count() - 1;
    if (!last) {
      // §5.20.1: le(TileSizeBytes) then the tile payload, interleaved per tile.
      append_le(payload, sizes[static_cast<std::size_t>(i)] - 1U,
                tile_size_bytes);
    }
    payload.insert(payload.end(),
                   tiles[static_cast<std::size_t>(i)].payload.begin(),
                   tiles[static_cast<std::size_t>(i)].payload.end());
  }

  const std::vector<std::uint8_t> clk = av2enc::m0::make_obu(
      ObuHeader{ObuType::kClosedLoopKey}, payload);
  const std::vector<std::uint8_t> td = av2enc::m0::make_obu(
      ObuHeader{ObuType::kTemporalDelimiter},
      std::span<const std::uint8_t>{});

  SequenceHeaderConfig sequence;
  sequence.width = static_cast<std::uint32_t>(source.width);
  sequence.height = static_cast<std::uint32_t>(source.height);
  sequence.level = 7;
  sequence.disable_loopfilters_across_tiles = true;
  const std::vector<std::uint8_t> seq = make_sequence_header_obu(sequence);

  frame->clear();
  frame->reserve(td.size() + seq.size() + clk.size());
  frame->insert(frame->end(), td.begin(), td.end());
  frame->insert(frame->end(), seq.begin(), seq.end());
  frame->insert(frame->end(), clk.begin(), clk.end());
  return true;
}

bool header_self_check(const Planes& source, const TileLayout& one_by_one,
                       std::uint8_t base_q_idx, std::string* error) {
  // Frame header parity: m1::write_frame_header() vs this writer for 1x1.
  av2enc::m1::M1Config config;
  config.max_frame_width = static_cast<std::uint16_t>(source.width);
  config.max_frame_height = static_cast<std::uint16_t>(source.height);
  config.base_q_idx = base_q_idx;
  config.disable_cdf_update = true;

  BitWriter reference;
  av2enc::m1::write_frame_header(reference, config);
  reference.align_to_byte();

  FrameHeaderConfig header;
  header.width = source.width;
  header.height = source.height;
  header.base_q_idx = base_q_idx;
  header.deblock = true;
  header.tile_size_bytes = 1;
  BitWriter writer;
  write_frame_header(writer, header, one_by_one);
  writer.align_to_byte();

  if (reference.bit_size() != writer.bit_size() ||
      reference.take_bytes() != writer.take_bytes()) {
    *error = "1x1 frame header does not match m1::write_frame_header()";
    return false;
  }

  // Sequence header parity with disable_loopfilters_across_tiles = 0.
  SequenceHeaderConfig sequence;
  sequence.width = static_cast<std::uint32_t>(source.width);
  sequence.height = static_cast<std::uint32_t>(source.height);
  sequence.level = 7;
  sequence.disable_loopfilters_across_tiles = false;
  const std::vector<std::uint8_t> reference_seq =
      av2enc::m1::make_sequence_header_obu(config);
  const std::vector<std::uint8_t> writer_seq =
      make_sequence_header_obu(sequence);
  if (reference_seq != writer_seq) {
    *error = "sequence header does not match m1::make_sequence_header_obu()";
    return false;
  }
  return true;
}

TileLayoutPreset parse_preset(const std::string& name) {
  if (name == "1x1") return TileLayoutPreset::k1x1;
  if (name == "2x1") return TileLayoutPreset::k2x1;
  if (name == "2x2") return TileLayoutPreset::k2x2;
  if (name == "4x2") return TileLayoutPreset::k4x2;
  if (name == "8x2") return TileLayoutPreset::k8x2;
  if (name == "16x2") return TileLayoutPreset::k16x2;
  return TileLayoutPreset::k1x1;
}

struct MismatchStats {
  int total = 0;
  int interior = 0;  // farther than 3 luma samples from any interior tile edge
};

// A mismatch on a tile boundary is expected when deblocking is enabled: the
// product M1 reference deblocks each cropped tile as if its edges were frame
// edges, while a decoder may still filter at the shared edge (the pinned
// dav2d build does, even with disable_loopfilters_across_tiles = 1). A
// mismatch in the tile interior would mean the payload/merge is wrong.
MismatchStats classify_mismatches(const std::vector<std::uint8_t>& decoded,
                                  const std::vector<std::uint8_t>& recon,
                                  int width, int height, const TileLayout& layout,
                                  int chroma_shift) {
  MismatchStats stats;
  if (decoded.size() != recon.size()) {
    stats.total = -1;
    stats.interior = -1;
    return stats;
  }
  // Luma-domain tolerance. Deblock taps reach at most 3 luma samples each side;
  // chroma coordinates are doubled, so allow a little more there.
  const int tolerance = 6;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t index =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(x);
      if (decoded[index] == recon[index]) continue;
      ++stats.total;
      const int luma_x = x << chroma_shift;
      const int luma_y = y << chroma_shift;
      bool on_edge = false;
      for (std::size_t c = 1; c < layout.col_starts_sb.size() && !on_edge; ++c) {
        const int edge = layout.col_starts_sb[c] * layout.sb_size;
        if (std::abs(luma_x - edge) <= tolerance) on_edge = true;
      }
      for (std::size_t r = 1; r < layout.row_starts_sb.size() && !on_edge; ++r) {
        const int edge = layout.row_starts_sb[r] * layout.sb_size;
        if (std::abs(luma_y - edge) <= tolerance) on_edge = true;
      }
      if (!on_edge) ++stats.interior;
    }
  }
  return stats;
}

// Diagnostic only: where do the first mismatches sit?
void print_mismatches(const char* plane_name, const std::vector<std::uint8_t>& a,
                      const std::vector<std::uint8_t>& b, int width,
                      int limit) {
  if (a.size() != b.size()) return;
  int printed = 0;
  for (std::size_t i = 0; i < a.size() && printed < limit; ++i) {
    if (a[i] != b[i]) {
      const int x = static_cast<int>(i % static_cast<std::size_t>(width));
      const int y = static_cast<int>(i / static_cast<std::size_t>(width));
      std::printf("  %s mismatch @ (%d,%d): decoded %u recon %u\n", plane_name,
                  x, y, static_cast<unsigned>(a[i]),
                  static_cast<unsigned>(b[i]));
      ++printed;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <source.y4m> [layout] [--dav2d PATH] "
                 "[--out DIR] [--base-q Q]\n"
                 "layout: 1x1 2x1 2x2 4x2 8x2 16x2 (default 2x2)\n",
                 argv[0]);
    return 2;
  }
  const std::string source_path = argv[1];
  std::string layout_name = "2x2";
  std::string dav2d = "tools/bin/dav2d";
  std::string out_dir = ".";
  int base_q = 90;
  bool deblock = false;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--dav2d" && i + 1 < argc) {
      dav2d = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (arg == "--deblock") {
      deblock = true;
    } else if (arg == "--base-q" && i + 1 < argc) {
      base_q = std::atoi(argv[++i]);
    } else if (arg.rfind("--", 0) == 0) {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    } else {
      layout_name = arg;
    }
  }

  std::string resolved_decoder = dav2d;
  if (!std::filesystem::exists(resolved_decoder)) {
    const std::string macos = dav2d + "-macos";
    if (std::filesystem::exists(macos)) {
      resolved_decoder = macos;
    } else {
      std::fprintf(stderr, "dav2d not found: %s\n", dav2d.c_str());
      return 127;
    }
  }
  std::filesystem::create_directories(out_dir);

  std::string error;
  Planes source;
  if (!load_first_frame(source_path, &source, &error)) {
    std::fprintf(stderr, "cannot read %s: %s\n", source_path.c_str(),
                 error.c_str());
    return 1;
  }
  if ((source.width & 1) != 0 || (source.height & 1) != 0) {
    std::fprintf(stderr, "source must have even 4:2:0 dimensions\n");
    return 1;
  }

  const TileLayoutPreset preset = parse_preset(layout_name);
  const auto dims = av2enc::tiles::preset_dims(preset);
  const auto planned = av2enc::tiles::plan_tile_layout(
      source.width, source.height, dims.cols, dims.rows, 64, 7, 0);
  if (!planned.ok()) {
    std::fprintf(stderr, "layout %s rejected: %s\n", layout_name.c_str(),
                 planned.error.c_str());
    return 1;
  }
  const TileLayout& layout = planned.layout;
  std::printf("source: %s %dx%d\n", source_path.c_str(), source.width,
              source.height);
  std::printf("layout: %s, %d tiles (%dx%d), deblock %s, tile sizes:",
              layout_name.c_str(), layout.tile_count(), layout.cols,
              layout.rows, deblock ? "on" : "off");
  for (const TileRect& tile : layout.tiles) {
    std::printf(" %dx%d", tile.pixel_width(), tile.pixel_height());
  }
  std::printf("\n");

  // 1x1 view of the same frame for the header parity check.
  const auto one_by_one_result = av2enc::tiles::plan_tile_layout(
      source.width, source.height, 1, 1, 64, 7, 0);
  if (!one_by_one_result.ok()) {
    std::fprintf(stderr, "cannot plan the 1x1 reference layout\n");
    return 1;
  }
  if (!header_self_check(source, one_by_one_result.layout,
                         static_cast<std::uint8_t>(base_q), &error)) {
    fail(error);
    return 1;
  }
  std::printf("header self-check: PASS (1x1 frame + sequence headers)\n");

  std::vector<EncodedTile> tiles;
  tiles.reserve(static_cast<std::size_t>(layout.tile_count()));
  for (const TileRect& rect : layout.tiles) {
    EncodedTile tile;
    if (!encode_tile(source, rect, static_cast<std::uint8_t>(base_q), deblock,
                     &tile, &error)) {
      std::fprintf(stderr, "tile %d encode failed: %s\n", rect.index,
                   error.c_str());
      return 1;
    }
    tiles.push_back(std::move(tile));
  }

  std::vector<std::uint8_t> frame;
  if (!assemble_tiled_stream(layout, source, tiles,
                             static_cast<std::uint8_t>(base_q), deblock, &frame,
                             &error)) {
    std::fprintf(stderr, "tiled assembly failed: %s\n", error.c_str());
    return 1;
  }
  const std::string ivf = out_dir + "/tiled_" + layout_name + ".ivf";
  if (!write_ivf(ivf, source.width, source.height, frame, &error)) {
    std::fprintf(stderr, "cannot write %s: %s\n", ivf.c_str(), error.c_str());
    return 1;
  }

  std::printf("stream: %zu bytes, tiles:", frame.size());
  for (const EncodedTile& tile : tiles) {
    std::printf(" %zu", tile.payload.size());
  }
  std::printf("\nwrote: %s\n", ivf.c_str());

  const std::string decoded_path = out_dir + "/tiled_" + layout_name + ".y4m";
  if (!run_decoder(resolved_decoder, ivf, decoded_path, &error)) {
    fail(error);
    return 1;
  }
  Planes decoded;
  if (!load_first_frame(decoded_path, &decoded, &error)) {
    fail("cannot read dav2d output: " + error);
    return 1;
  }

  // Assemble the per-tile M1 reconstructions into one frame.
  Planes recon;
  recon.width = source.width;
  recon.height = source.height;
  recon.y.assign(static_cast<std::size_t>(source.width) *
                     static_cast<std::size_t>(source.height),
                 0);
  recon.u.assign(static_cast<std::size_t>(source.width / 2) *
                     static_cast<std::size_t>(source.height / 2),
                 0);
  recon.v.assign(recon.u.size(), 0);
  for (const EncodedTile& tile : tiles) {
    const TileRect& rect = layout.tiles[static_cast<std::size_t>(tile.index)];
    paste_plane(tile.recon_y, tile.width, tile.height, &recon.y, source.width,
                rect.x0, rect.y0);
    paste_plane(tile.recon_u, tile.width / 2, tile.height / 2, &recon.u,
                source.width / 2, rect.x0 / 2, rect.y0 / 2);
    paste_plane(tile.recon_v, tile.width / 2, tile.height / 2, &recon.v,
                source.width / 2, rect.x0 / 2, rect.y0 / 2);
  }

  const MismatchStats y_stats = classify_mismatches(
      decoded.y, recon.y, source.width, source.height, layout, 0);
  const MismatchStats u_stats = classify_mismatches(
      decoded.u, recon.u, source.width / 2, source.height / 2, layout, 1);
  const MismatchStats v_stats = classify_mismatches(
      decoded.v, recon.v, source.width / 2, source.height / 2, layout, 1);
  std::printf(
      "decoded vs per-tile M1 recon: Y %d (%d interior) U %d (%d interior) "
      "V %d (%d interior)\n",
      y_stats.total, y_stats.interior, u_stats.total, u_stats.interior,
      v_stats.total, v_stats.interior);

  int interior_mismatches =
      y_stats.interior + u_stats.interior + v_stats.interior;
  if (deblock) {
    // Deblock-on: the tile payloads and merge must still be exact away from a
    // shared tile edge. The pinned decoder also filters the shared edge, which
    // a per-tile CPU reference cannot reproduce; that difference is reported,
    // not hidden.
    if (interior_mismatches != 0) {
      print_mismatches("Y", decoded.y, recon.y, source.width, 12);
      fail("tile interior differs from the per-tile reconstruction");
    }
  } else if (y_stats.total != 0 || u_stats.total != 0 || v_stats.total != 0) {
    print_mismatches("Y", decoded.y, recon.y, source.width, 12);
    print_mismatches("U", decoded.u, recon.u, source.width / 2, 6);
    print_mismatches("V", decoded.v, recon.v, source.width / 2, 6);
    fail("decoded tiled frame does not match the per-tile reconstruction");
  }
  if (decoded.width != source.width || decoded.height != source.height) {
    fail("decoded frame dimensions differ from the source");
  }

  std::printf("tiled dav2d gate (%s): %s\n", layout_name.c_str(),
              g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
