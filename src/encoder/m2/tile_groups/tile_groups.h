// Multi-tile-group framing for AV2 frames (spec-only component).
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit `38f28e7`):
//   * §5.19    tile_group_obu()            (pp. 144-145)
//   * §5.18.1  frame_header(isFirst)       (pp. 93-94)  -> frame_header_copy()
//   * §6.17.1  frame_header_copy semantics (p. 319)
//   * §5.20.1  tile_group_payload()        (pp. 145-146)
//   * §6.18    tile group OBU semantics    (p. 349)
//   * §6.19.1  tile size semantics         (p. 349)
//   * §5.20.2.1 BruTileActive / bru_mode   (p. 147)
//
// This module owns the *framing* of a frame that is split across two or more
// tile-group OBUs. It does not encode tile payloads and it does not write the
// frame header itself: the M2 header writer
// (`m2::post::write_m2_frame_header_syntax`) supplies the exact
// `frame_header_info()` bits once, and this module either emits them with the
// first group, omits them (`frame_header_present_flag = 0`), or re-emits a
// bit-identical copy (`frame_header_copy()` / `header_bit[i]`, §5.18.1).
//
// Current writers hardcode one all-tiles group:
//   * `m2::post::assemble_m2_tile_group_from_payloads`  is_first_tile_group = 1,
//     tile_start_and_end_present_flag = 0
//   * `m1::tiles_stream` / `tiles::plan_single_tile_group`
// Those remain the single-group fast path. A frame that must be split (GPU
// tile completion order, per-group robustness, partial decode) is assembled
// here instead; see README.md for the exact call sites.
//
// State that is not a syntax element but is required by the framing:
//   * `NumFrameHeaderBits` (§5.18.1) - needed by a reader to consume a
//     `frame_header_copy()`. It is produced when the first group's frame
//     header is parsed.
//   * `TileSizeBytes` (§5.18.7.2) - frame-level, so it must cover the largest
//     tile in the *whole frame*, not the largest tile in a group.
//   * `use_bru` / `bru_inactive` (§5.18.2) - frame-header flags that decide
//     whether `bru_tile_active` bits exist and whether a group carries any
//     tile payload at all.
#pragma once

#include "bitstream/bitwriter.h"
#include "encoder/tiles/tile_layout.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::m2::tile_groups {

// §5.18.1 `NumFrameHeaderBits` worth of bits, MSB first. Bit i lives at
// `bytes[i >> 3] >> (7 - (i & 7))`.
struct FrameHeaderBits {
  std::vector<std::uint8_t> bytes;
  std::size_t bit_count = 0;

  bool empty() const { return bit_count == 0; }
};

// Copies the first `bit_count` bits out of a writer/byte buffer. The writer
// may have zero padding in its final byte; only `bit_count` bits are kept.
bool capture_frame_header_bits(std::span<const std::uint8_t> bytes,
                               std::size_t bit_count, FrameHeaderBits* out,
                               std::string* error = nullptr);

// True when bit `index` is set. Returns false, leaving `*in_range` false, when
// the index is outside [0, bit_count).
bool frame_header_bit(const FrameHeaderBits& bits, std::size_t index,
                      bool* in_range);

// §5.19 / §5.18.7.2 state the tile-group syntax depends on.
struct TileGroupGeometry {
  int num_tiles = 0;        // NumTiles = TileCols * TileRows
  int tile_cols_log2 = 0;   // TileColsLog2
  int tile_rows_log2 = 0;   // TileRowsLog2
  int tile_size_bytes = 0;  // TileSizeBytes; 0 when NumTiles == 1
  bool use_bru = false;     // frame_header().use_bru
  // bru_inactive == 1 makes the whole frame a single header-only OBU
  // (§5.19: `if (bru_inactive) { trailing_bits(...); return; }`).
  bool bru_inactive = false;

  int tile_bits() const { return tile_cols_log2 + tile_rows_log2; }
};

// Builds the geometry from a validated tile layout. `tile_size_bytes` comes
// from the frame header (`tile_info()`), not from a group.
TileGroupGeometry geometry_from_layout(const tiles::TileLayout& layout,
                                       int tile_size_bytes, bool use_bru,
                                       bool bru_inactive);

bool validate_tile_group_geometry(const TileGroupGeometry& geometry,
                                  std::string* error = nullptr);

// One tile group's coverage. `tg_end` is inclusive.
struct TileGroupSpan {
  int tg_start = 0;
  int tg_end = -1;
  // Only consulted for a non-first group. The first group always has
  // `frame_header_present_flag = 1` by §5.19 (the bit is not coded).
  bool frame_header_present = false;
  // Per-tile `bru_tile_active` for the tiles in this group, raster order.
  // Required only when `use_bru && !bru_inactive && NumTiles > 1`. For
  // NumTiles == 1 the spec infers `BruTileActives[0][0] = 1` and codes no bit.
  std::vector<std::uint8_t> bru_tile_active;

  int tile_count() const { return tg_end - tg_start + 1; }
};

// A full frame's tile-group plan.
struct FrameTileGroupPlan {
  TileGroupGeometry geometry;
  FrameHeaderBits frame_header;
  // false: every non-first group has frame_header_present_flag = 0.
  // true:  every non-first group re-emits a bit-identical copy.
  bool copy_frame_header = false;
  std::vector<TileGroupSpan> groups;
};

// Splits NumTiles into contiguous groups of `tiles_per_group` tiles. Fills the
// per-tile BRU flags with `bru_active_default` when the geometry requires them.
bool plan_contiguous_tile_groups(const TileGroupGeometry& geometry,
                                 std::span<const int> tiles_per_group,
                                 bool copy_frame_header,
                                 bool bru_active_default,
                                 std::vector<TileGroupSpan>* groups,
                                 std::string* error = nullptr);

// §6.18 conformance of the whole frame:
//   * SeenFrameHeader != is_first_tile_group  (first group is first, rest are
//     not);
//   * the first group starts at TileNum 0;
//   * tg_end >= tg_start for every group;
//   * no gap and no overlap: group i starts at group i-1's tg_end + 1;
//   * the last group ends at NumTiles - 1.
// A bru_inactive frame is a single header-only group and has no coverage.
bool validate_frame_tile_groups(const FrameTileGroupPlan& plan,
                                std::string* error = nullptr);

// Per-OBU write parameters. The first group's `frame_header_present` is
// inferred to 1; a non-first group's bit is written from this struct.
struct TileGroupWriteParams {
  bool is_first_tile_group = false;
  bool frame_header_present = false;
  TileGroupSpan span;
};

// Writes one §5.19 `tile_group_obu()` payload (without the OBU header).
//
// `tile_payloads` is the whole frame in raster TileNum order; the group writes
// only `span.tg_start .. span.tg_end`. Every used payload must be at least 2
// bytes (init_symbol reads up to 15 bits). Size fields use `TileSizeBytes` and
// are interleaved with the payloads (the last tile of the group is implied).
bool write_tile_group(const TileGroupGeometry& geometry,
                      const FrameHeaderBits& frame_header,
                      const TileGroupWriteParams& params,
                      std::span<const std::vector<std::uint8_t>> tile_payloads,
                      std::vector<std::uint8_t>* out,
                      std::string* error = nullptr);

// Validated whole-frame writer. Produces one payload per group, in order.
bool write_frame_tile_groups(
    const FrameTileGroupPlan& plan,
    std::span<const std::vector<std::uint8_t>> tile_payloads,
    std::vector<std::vector<std::uint8_t>>* out,
    std::string* error = nullptr);

// --- independent reader -----------------------------------------------------

struct ParsedTileGroup {
  bool is_first_tile_group = false;
  bool frame_header_present = false;
  // A non-first group that carries `frame_header_copy()` bits.
  bool frame_header_is_copy = false;
  FrameHeaderBits frame_header;  // empty when not present
  bool bru_inactive = false;     // header-only group (§5.19 early return)
  bool tile_start_and_end_present = false;
  int tg_start = 0;
  int tg_end = -1;
  std::vector<std::uint8_t> bru_tile_active;
  // Header length before the first tile payload.
  std::size_t header_bits = 0;
  std::size_t header_bytes = 0;
  // One entry per tile in the group, and its byte offset in the payload.
  std::vector<std::uint32_t> tile_bytes;
  std::vector<std::size_t> tile_offsets;
};

struct TileGroupSequence {
  int num_tiles = 0;
  bool bru_inactive = false;
  int groups_with_header_copy = 0;
  int groups_with_shared_header = 0;
  std::vector<ParsedTileGroup> groups;
};

// Parses one payload. `num_frame_header_bits` is the §5.18.1
// `NumFrameHeaderBits` saved from the first group's frame header; it is
// required whenever a group carries a header. The reader never trusts writer
// state: every framing field is re-read from the bytes and the byte-alignment
// padding, size fields and payload extent are checked.
bool read_tile_group(std::span<const std::uint8_t> payload,
                     const TileGroupGeometry& geometry,
                     std::size_t num_frame_header_bits, ParsedTileGroup* out,
                     std::string* error = nullptr);

// Reads every group of one frame and enforces §6.18. This is the rejection
// point for gaps, overlap/reordering, a missing final tile, a first group that
// does not start at tile 0, duplicate `is_first_tile_group` bits, and a
// `frame_header_copy()` that differs from the first group's header.
bool read_tile_group_sequence(
    std::span<const std::vector<std::uint8_t>> payloads,
    const TileGroupGeometry& geometry, std::size_t num_frame_header_bits,
    TileGroupSequence* out, std::string* error = nullptr);

}  // namespace av2enc::m2::tile_groups
