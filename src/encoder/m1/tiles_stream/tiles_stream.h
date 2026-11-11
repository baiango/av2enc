// Reusable tiled writer for AV2 M1 still pictures.
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//   §5.4.2   sequence_tile_config() (seq_tile_info_present_flag = 0)
//   §5.4.10  disable_loopfilters_across_tiles
//   §5.18.7.2 / §5.18.7.3  tile_info() with uniform spacing
//   §5.18.7.5 uniform_spacing()
//   §5.19    tile_group_obu()
//   §5.20.1  tile_group_payload(): interleaved le(TileSizeBytes) / payload
//   §5.20.2.1 per-tile CDF/context initialisation
//
// What this module owns
// ---------------------
//   * uniform tile layout planning (through the already-verified
//     `av2enc::tiles::plan_tile_layout()`);
//   * the tiled frame/sequence headers (the 1x1 forms are byte-compared with
//     the M1 core writers before every encode);
//   * `tile_size_bytes_minus_1` selection;
//   * a single tile group with `tile_start_and_end_present_flag = 0` and
//     `le(TileSizeBytes) tile_size_minus_1` followed immediately by that tile's
//     payload, in raster `TileNum` order (the last size is implied);
//   * cropping each tile, encoding it with the public M1 reference
//     `encode_m1_still_picture()`, and collecting its payload and visible
//     reconstruction.
//
// Per-tile reset / tile-local origins
// -----------------------------------
// Each tile is encoded as a complete, standalone still picture. That makes the
// entropy coder start from `init_symbol()` on a fresh CDF copy, starts every
// tile at local superblock origin (0,0), and clears the above/left contexts at
// the tile start, exactly as §5.20.2.1 requires. `Tile` still records the
// frame-global superblock rectangle so callers can reason about boundaries and
// paste the reconstructions back without recomputing geometry.
//
// Deblocking across tile boundaries
// ---------------------------------
// The CPU M1 reference deblocks a cropped tile as if every tile edge were a
// frame edge. The pinned dav2d build filters the *shared* edge as well, even
// when the sequence header says `disable_loopfilters_across_tiles = 1`; the
// two outputs therefore differ by a few samples on interior tile edges when
// deblocking is enabled. The exactness gate uses `enable_deblock = false`.
// See README.md for the measured finding. Do not enable deblocking for a
// multi-tile stream and then claim bit-exact reconstruction across tile edges
// until a coordinator cross-tile filter pass exists.
//
// This module deliberately does not edit the M1/M2 core, `src/encoder/tiles/`,
// the pipeline, `src/main.cpp`, or CMake. It links the public core entry points.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "encoder/m1/m1_still.h"
#include "encoder/tiles/tile_layout.h"

namespace av2enc::m1::tiles_stream {

// Coding configuration for one tiled still picture. Defaults mirror the
// verified single-tile M1 reference (`M1Config`/`M1Options` defaults), with
// deblocking off because a per-tile CPU reconstruction cannot reproduce the
// pinned decoder's cross-tile deblock taps.
struct Config {
  // Visible luma size. Must match StillPicture::width/height in encode().
  // §5.18.1 max_frame_width/height.
  int width = 0;
  int height = 0;
  // Uniform raster tile grid. 2x1/2x2/4x2/8x2 are supported everywhere the
  // planner can represent them; 16x2 is supported where the frame has at least
  // 16 superblock columns (for example 1920x1080, not 640x360).
  tiles::TileLayoutPreset preset = tiles::TileLayoutPreset::k1x1;
  std::uint8_t base_q_idx = 90;
  // Deblock each tile before output. Exact multi-tile recon requires this to
  // stay false (see the deblock note above).
  bool enable_deblock = false;
  // §5.4.10. The gate writes 1 so the decoder is told not to filter across
  // tile edges. It does not make deblock-on output exact with pinned dav2d.
  bool disable_loopfilters_across_tiles = true;
  // 64 is the only superblock size the M1 core implements. 128 is accepted by
  // the geometry planner but rejected by encode().
  int sb_size = tiles::kSuperblockPixels64;
  // §5.18.1 seq_level_idx. Also feeds the geometry planner's level bounds.
  std::uint8_t seq_level_idx = 7;
  // Tool set pinned to the verified M1 reference. Changing these without a
  // matching core change makes the stream non-decodable.
  bool enable_intra_edge_filter = true;
  bool enable_cfl_intra = true;
  bool enable_chroma_dctonly = true;
};

// One encoded tile. `payload` is the §8.2 entropy payload only (no OBU framing
// and no size field). Reconstructions are cropped to the visible tile size.
struct Tile {
  int index = 0;
  int row = 0;
  int col = 0;
  int x0 = 0;  // visible luma pixels, left/top inclusive
  int y0 = 0;
  int x1 = 0;  // right/bottom exclusive
  int y1 = 0;
  // Frame-global superblock rectangle from the plan.
  int sb_col_start = 0;
  int sb_row_start = 0;
  int sb_cols = 0;
  int sb_rows = 0;
  // Tile-local geometry. Every tile is encoded from local origin (0,0), so the
  // syntax writer's superblock walk starts here, not at the frame-global SB.
  int local_sb_cols = 0;
  int local_sb_rows = 0;
  int local_pixel_width = 0;
  int local_pixel_height = 0;

  std::vector<std::uint8_t> payload;
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
};

// One complete still picture, in one tile group.
struct Frame {
  tiles::TileLayout layout;
  std::vector<Tile> tiles;
  // Minimal TileSizeBytes for the largest payload (1..4).
  int tile_size_bytes = 0;
  // §5.20.1 payload bytes: tiled frame header, then interleaved size/payload.
  std::vector<std::uint8_t> clk_payload;
  // TD + sequence-header + CLK OBUs, ready for an IVF frame.
  std::vector<std::uint8_t> frame;
};

// A full-frame reconstruction assembled from the per-tile visible planes.
struct Reconstruction {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

// Plan a uniform-spacing layout for `preset` using the same level/tier bounds
// as the header. Rejects non-representable grids (for example 16x2 at
// 640x360, whose 10 superblock columns cannot be split into 16 tiles).
bool build_layout(int width, int height, tiles::TileLayoutPreset preset, int sb_size,
                  int seq_level_idx, tiles::TileLayout* layout, std::string* error);

// Minimal TileSizeBytes that can hold the largest payload, or 0 with `error`
// when a payload is < 2 bytes (init_symbol() reads up to 15 bits) or does not
// fit in 4 bytes.
int select_tile_size_bytes(std::span<const std::uint32_t> sizes, std::string* error);

// §5.18.7.2/§5.18.7.3 tile_info() with uniform_tile_spacing_flag = 1. When
// `layout.tile_count() > 1` this also writes tile_size_bytes_minus_1 f(2).
// context_update_tile_id is absent: single_picture_header_flag forces
// enable_avg_cdf = 1 / avg_cdf_type = 1.
void write_tile_info(BitWriter& writer, const tiles::TileLayout& layout, int tile_size_bytes);

// Writes frame_header(isFirst = 1) syntax (no is_first_tile_group bit and no
// byte_alignment). `tile_size_bytes` is the §5.18.7.2 field width and is only
// coded when the layout has more than one tile. Returns the number of bits
// written.
std::size_t write_frame_header(BitWriter& writer, const Config& config,
                               const tiles::TileLayout& layout, int tile_size_bytes);

// Complete sequence_header_obu() with OBU header + Annex B.2 framing.
std::vector<std::uint8_t> make_sequence_header_obu(const Config& config);

// A non-owning view of one tile payload, used by the low-level assembler.
struct TilePayloadView {
  std::span<const std::uint8_t> bytes;
};

// Builds the §5.20.1 tile-group payload: is_first_tile_group, the tiled frame
// header, tile_start_and_end_present_flag = 0, byte_alignment, then raster
// order `le(TileSizeBytes) (size - 1)` + payload for every tile except the
// last (whose size is implied by the OBU length). On success `tile_size_bytes`
// receives the selected field width.
bool make_tile_group_payload(const tiles::TileLayout& layout, const Config& config,
                             std::span<const TilePayloadView> payloads,
                             std::vector<std::uint8_t>* clk_payload, int* tile_size_bytes,
                             std::string* error);

// Self-check: compares this writer's 1x1 frame header and its
// disable_loopfilters_across_tiles = 0 sequence header with the M1 core
// writers. encode() runs this before doing any work, so a core header change
// can never silently desynchronise the tiled stream.
bool core_header_parity(const Config& config, std::string* error);

// Encode a tiled still picture: plan, crop, encode every tile with the public
// M1 reference, assemble the tile group, and return the OBU stream plus the
// per-tile visible reconstructions.
bool encode(const StillPicture& source, const Config& config, Frame* frame, std::string* error);

// Paste the per-tile visible reconstructions into one full frame. Validates
// coverage and plane sizes; used by the gate and by any caller that wants the
// expected decoder output.
bool assemble_reconstruction(const Frame& frame, Reconstruction* out, std::string* error);

}  // namespace av2enc::m1::tiles_stream
