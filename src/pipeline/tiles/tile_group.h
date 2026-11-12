// Tile-group payload planning for AV2 multi-tile frames.
//
// Spec-only: AV2 v1.0.0 §5.19 (`tile_group_obu`), §6.18, §5.20.1
// (`tile_group_payload`), §6.19.1, and §4.11.5 (`le(n)` little-endian).
//
// This module does not write a bitstream. It answers the questions the
// coordinator needs before writing one:
//   * how many `tile_size_minus_1` fields are present (NumTiles - 1),
//   * how wide each field must be (1..4 bytes),
//   * where each tile's byte range lands in the payload,
//   * and whether the sizes are legal (>= 2 bytes, last size implied).
//
// The payload order is always raster TileNum, independent of which GPU or
// workgroup produced a tile. A caller that appends payloads in completion
// order is wrong even if every tile is individually valid.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "tile_layout.h"

namespace av2enc::tiles {

struct TileGroupPlan {
  bool valid = false;
  bool is_first_tile_group = true;
  bool all_tiles = true;  // tile_start_and_end_present_flag == 0
  int tile_count = 0;
  int tile_cols_log2 = 0;
  int tile_rows_log2 = 0;
  int tile_bits = 0;  // TileColsLog2 + TileRowsLog2, for tg_start/tg_end

  int tile_size_bytes = 0;     // TileSizeBytes = tile_size_bytes_minus_1 + 1
  int size_field_count = 0;    // NumTiles - 1
  std::uint64_t size_field_bytes = 0;
  std::uint64_t payload_bytes = 0;

  // Header bit accounting, including the caller-supplied frame_header_bits.
  // `header_bits` is byte-aligned by the trailing byte_alignment().
  std::uint64_t frame_header_bits = 0;
  std::uint64_t header_bits = 0;
  std::uint64_t header_bytes = 0;
  std::uint64_t obu_payload_bytes = 0;

  // Byte offsets relative to the first byte of the payload region.
  std::vector<std::uint64_t> tile_payload_offsets;
  // Only tiles 0..NumTiles-2 have a signalled size; the last size is implied.
  std::vector<std::uint32_t> tile_size_minus_1;

  std::uint32_t largest_tile_bytes = 0;
  std::uint32_t smallest_tile_bytes = 0;
  std::string error;
};

// Minimal TileSizeBytes that can represent `largest_tile_bytes` in a
// `tile_size_minus_1` field. Returns 0 when it cannot fit in 4 bytes or the
// size is below the 2-byte minimum.
int tile_size_bytes_for(std::uint32_t largest_tile_bytes);

// Plan one tile-group OBU containing every tile in raster order.
//
//   tile_sizes          one byte count per tile, raster TileNum order
//   frame_header_bits   exact bit length of frame_header() at its current
//                       position; 0 is legal only for the first group of a
//                       frame whose frame header is written elsewhere or is
//                       empty (the M0 writer writes it between is_first_tile_group
//                       and the tile-start fields)
//   is_first_tile_group true for the single group of a frame
//
// A non-first group with a frame header is not modelled; partial tile groups
// are deliberately unsupported here.
TileGroupPlan plan_single_tile_group(
    const TileLayout& layout, std::span<const std::uint32_t> tile_sizes,
    std::uint64_t frame_header_bits = 0, bool is_first_tile_group = true);

}  // namespace av2enc::tiles
