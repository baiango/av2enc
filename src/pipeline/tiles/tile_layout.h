// AV2 v1.0.0 tile geometry planner (standalone component).
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//   * §5.18.7.2 / §6.17.7.2  tile_info()
//   * §5.18.7.3 / §6.17.7.3  tile_params()
//   * §5.18.7.5              uniform_spacing()
//   * §5.18.7.6 / §6.17.7.6  tile size in superblocks
//   * §5.18.7.7              tile_log2()
//   * §9.2                   MAX_TILE_WIDTH / MAX_TILE_AREA / tile scaling
//
// This module computes *geometry only*. It emits no bits, owns no entropy
// state, and is not wired into the M0 path. The integration pass consumes the
// `TileLayout` to drive per-tile prediction availability, loop-filter
// boundaries, and tile-group assembly (see tile_group.h).
//
// The widths are computed exactly as `uniform_spacing()` does: extra
// superblocks go to the *first* tiles, and the last tile extends to the end of
// the superblock grid. For 1920x1080 that means tile row heights 512 / 568,
// not an even 540/540 split. Do not "simplify" this; the decoder derives the
// same starts and a mismatch is an invalid stream.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::tiles {

// AV2 Table 9.x geometry constants.
inline constexpr int kSuperblockPixels64 = 64;
inline constexpr int kSuperblockPixels128 = 128;
inline constexpr int kMaxTileColumns = 64;      // MAX_TILE_COLS
inline constexpr int kMaxTileRows = 64;         // MAX_TILE_ROWS
inline constexpr int kMaxTileWidthPixels = 4096;  // MAX_TILE_WIDTH
inline constexpr int kMaxTileAreaPixels = 4096 * 2304;  // MAX_TILE_AREA

// Layout policies used by the multi-GPU planner. Kept here so both components
// name the same layouts.
enum class TileLayoutPreset : std::uint8_t {
  k1x1 = 0,
  k2x1,
  k2x2,
  k4x2,
  k8x2,
  k16x2,
};

struct TileLayoutDims {
  int cols = 1;
  int rows = 1;
};

const char* preset_name(TileLayoutPreset preset);
TileLayoutDims preset_dims(TileLayoutPreset preset);

// §9.2 Tile_Width_Scaling_Factor / Tile_Area_Scaling_Factor plus the Annex A
// Table A.9 MaxTiles / MaxTileCols bounds for the level. `seq_tier` is 0 or 1.
// A reserved level/tier combination sets `reserved`. The values are kept here
// so this component stays standalone; tiles_test.cpp cross-checks every row
// against core/conformance/conformance_tables.cpp so the two cannot drift.
struct LevelTileLimits {
  int width_scaling_factor = 4;
  int area_scaling_factor = 4;
  int max_tiles = 0;       // Table A.9 MaxTiles
  int max_tile_cols = 0;   // Table A.9 MaxTileCols
  bool reserved = false;
};

LevelTileLimits level_tile_limits(int seq_level_idx, int seq_tier);

// Lowest seq_level_idx in 0..21 whose Table A.9 MaxTiles / MaxTileCols admit a
// cols x rows grid. Returns 31 (maximum parameters) when no level admits it.
// Used by writers that must signal a high enough level instead of emitting a
// stream the conformance validator rejects.
int minimum_level_for_grid(int cols, int rows);

// §5.18.7.7:
//   tile_log2(blkSize, target) { for (k = 0; (blkSize << k) < target; k++) {} }
// Returns -1 for a non-positive blkSize.
int tile_log2(int blk_size, int target);

int mi_cols(int width);  // 2 * ((width + 7) >> 3)  (§5.18.4.4)
int mi_rows(int height);
int sb_cols(int width, int sb_size);
int sb_rows(int height, int sb_size);
bool is_power_of_two(int value);
int log2_exact(int value);  // -1 when value is not a positive power of two

// A tile edge is either a frame boundary or an interior tile boundary. Interior
// edges are where `is_inside()` (§5.20.9.1) changes ownership and where
// `disable_loopfilters_across_tiles` (§5.4.10) skips filtering.
enum class TileEdge : std::uint8_t {
  kFrameBoundary = 0,
  kInterior = 1,
};

struct TileRect {
  int index = 0;
  int row = 0;
  int col = 0;
  int x0 = 0;  // visible luma pixels, left/top inclusive
  int y0 = 0;
  int x1 = 0;  // right/bottom exclusive
  int y1 = 0;
  int sb_col_start = 0;
  int sb_row_start = 0;
  int sb_cols = 0;
  int sb_rows = 0;

  int pixel_width() const { return x1 - x0; }
  int pixel_height() const { return y1 - y0; }
  std::uint64_t pixels() const;
  std::uint64_t sb_area() const;
  bool contains(int x, int y) const;
};

struct TileBoundaries {
  TileEdge left = TileEdge::kFrameBoundary;
  TileEdge top = TileEdge::kFrameBoundary;
  TileEdge right = TileEdge::kFrameBoundary;
  TileEdge bottom = TileEdge::kFrameBoundary;

  // True when the tile touches the visible frame on that side.
  bool touches_frame_left = true;
  bool touches_frame_top = true;
  bool touches_frame_right = true;
  bool touches_frame_bottom = true;

  // The final tile row/column contains the partial superblock strip. The
  // visible height of a 1080p bottom tile is 568 px, of which the last 56 px
  // are a partial 64x64 superblock row.
  bool has_partial_superblock_column = false;
  bool has_partial_superblock_row = false;

  bool has_interior_vertical_edge() const { return left == TileEdge::kInterior; }
  bool has_interior_horizontal_edge() const { return top == TileEdge::kInterior; }
};

// Geometry derived from tile_info() plus the frame size. `tiles` is in raster
// order: row-major, tile index = row * cols + col.
struct TileLayout {
  int width = 0;
  int height = 0;
  int sb_size = kSuperblockPixels64;
  int sb_cols = 0;
  int sb_rows = 0;
  int cols = 0;
  int rows = 0;
  int cols_log2 = 0;
  int rows_log2 = 0;

  // Sequence level the layout was planned for. `seq_level_idx == 31` means
  // maximum parameters (no level limits). Exposed so validate_tile_layout()
  // can re-check the Table A.8/A.9 bounds on a hand-built layout.
  int seq_level_idx = 31;
  int seq_tier = 0;

  // Size `cols` (starts) and `rows`; the end is the next start or sb_cols/rows.
  std::vector<int> col_starts_sb;
  std::vector<int> row_starts_sb;

  std::vector<TileRect> tiles;
  std::vector<TileBoundaries> boundaries;

  // Spec-derived limits for the chosen level/tier. Exposed so the planner can
  // explain why a layout was rejected.
  int max_tile_width_sb = 0;
  int max_tile_area_sb = 0;
  int max_tiles = 0;       // Table A.9 MaxTiles (0 = unconstrained)
  int max_tile_cols = 0;   // Table A.9 MaxTileCols (0 = unconstrained)
  int min_log2_tile_cols = 0;
  int min_log2_tiles = 0;
  int max_log2_tile_cols = 0;
  int max_log2_tile_rows = 0;

  int tile_count() const { return static_cast<int>(tiles.size()); }
  bool empty() const { return tiles.empty(); }
  const TileRect* find(int row, int col) const;
  const TileRect* at_index(int index) const;
};

enum class LayoutStatus : std::uint8_t {
  kOk = 0,
  kInvalidGeometry,
  kInvalidLayout,
  kLevelConstraint,
  kNotRepresentable,
};

const char* layout_status_name(LayoutStatus status);

struct LayoutResult {
  LayoutStatus status = LayoutStatus::kInvalidGeometry;
  TileLayout layout;
  std::string error;

  bool ok() const { return status == LayoutStatus::kOk; }
};

// Plan a uniform-spacing layout.
//
//   width, height  visible luma samples, not padded
//   cols, rows     requested tile grid; both must be powers of two
//   sb_size        64 (primary) or 128
//   seq_level_idx  0..31; 31 means "unconstrained" for planning only
//   seq_tier       0 or 1 (main/high)
//
// Level checks model the video conformance bounds in §9.2 and Annex A
// (still_picture = 0): MAX_TILE_WIDTH / MAX_TILE_AREA scaling factors plus
// MaxTiles / MaxTileCols. A grid above either Table A.9 bound is rejected with
// kLevelConstraint; use minimum_level_for_grid() to pick a level that admits
// the grid. For an unconstrained still-picture plan, pass seq_level_idx = 31.
//
// Failure is explicit: a 64x64 frame cannot be 2x1, non-uniform grids are not
// representable with the uniform flag, and a level may force a minimum tile
// count. Never fall back silently to another layout.
LayoutResult plan_tile_layout(int width, int height, int cols, int rows,
                              int sb_size = kSuperblockPixels64,
                              int seq_level_idx = 31, int seq_tier = 0);

// Structural validation of an already-built layout: coverage, raster order,
// start monotonicity, boundary flags, and level limits. Does not consult a
// bitstream.
bool validate_tile_layout(const TileLayout& layout, std::string& error);

}  // namespace av2enc::tiles
