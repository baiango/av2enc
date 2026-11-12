#include "tile_layout.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::tiles {
namespace {

// Table A.9 MaxTiles / MaxTileCols, indexed by seq_level_idx 0..21. The
// conformance component owns the authoritative copy; tiles_test.cpp compares
// every row (and the §9.2 scaling factors below) against it.
struct LevelTileRow {
  int max_tiles;
  int max_tile_cols;
};

constexpr std::array<LevelTileRow, 22> kLevelTileRows = {{
    {8, 4},     {8, 4},     {16, 6},    {16, 6},    {32, 8},    {32, 8},
    {64, 8},    {64, 8},    {64, 8},    {64, 8},    {128, 16},  {128, 16},
    {128, 16},  {128, 16},  {256, 32},  {256, 32},  {256, 32},  {256, 32},
    {512, 64},  {512, 64},  {512, 64},  {512, 64},
}};

std::uint64_t pixel_area(int x0, int y0, int x1, int y1) {
  if (x1 <= x0 || y1 <= y0) {
    return 0;
  }
  return static_cast<std::uint64_t>(x1 - x0) *
         static_cast<std::uint64_t>(y1 - y0);
}

std::string describe_rect(const TileRect& tile) {
  std::ostringstream out;
  out << "tile " << tile.index << " (row " << tile.row << ", col " << tile.col
      << ") has negative geometry";
  return out.str();
}

// §5.18.7.5 uniform_spacing(). Returns the starts including the implicit end
// sentinel? No: it returns the actual starts, and the caller uses sb_cols as
// the final end. The number of starts is the real tile count, which may be
// smaller than 1<<tile_log2 for very small frames.
std::vector<int> uniform_starts(int mis, int sb_size, int tile_log2_value) {
  const int sb4x4 = sb_size / 4;
  const int sb_shift = log2_exact(sb4x4);
  const int sbs = (mis + sb4x4 - 1) >> sb_shift;
  const int full_sbs = mis >> sb_shift;
  const int tile_sb = full_sbs >> tile_log2_value;
  const int extra = (tile_sb == 0) ? sbs : full_sbs - (tile_sb << tile_log2_value);

  std::vector<int> starts;
  const int tile_count = 1 << tile_log2_value;
  int start = 0;
  for (int i = 0; i < tile_count && start < sbs; ++i) {
    starts.push_back(start);
    start += tile_sb;
    if (i < extra) {
      ++start;
    }
  }
  return starts;
}

int min_log2_for_area(int max_tile_area_sb, int sb_rows_value, int sb_cols_value) {
  if (max_tile_area_sb <= 0) {
    return 0;
  }
  return tile_log2(max_tile_area_sb, sb_rows_value * sb_cols_value);
}

}  // namespace

const char* preset_name(TileLayoutPreset preset) {
  switch (preset) {
    case TileLayoutPreset::k1x1:
      return "1x1";
    case TileLayoutPreset::k2x1:
      return "2x1";
    case TileLayoutPreset::k2x2:
      return "2x2";
    case TileLayoutPreset::k4x2:
      return "4x2";
    case TileLayoutPreset::k8x2:
      return "8x2";
    case TileLayoutPreset::k16x2:
      return "16x2";
  }
  return "?";
}

TileLayoutDims preset_dims(TileLayoutPreset preset) {
  switch (preset) {
    case TileLayoutPreset::k1x1:
      return {1, 1};
    case TileLayoutPreset::k2x1:
      return {2, 1};
    case TileLayoutPreset::k2x2:
      return {2, 2};
    case TileLayoutPreset::k4x2:
      return {4, 2};
    case TileLayoutPreset::k8x2:
      return {8, 2};
    case TileLayoutPreset::k16x2:
      return {16, 2};
  }
  return {1, 1};
}

LevelTileLimits level_tile_limits(int seq_level_idx, int seq_tier) {
  LevelTileLimits limits;
  if (seq_level_idx < 0 || seq_level_idx > 31 || seq_tier < 0 || seq_tier > 1) {
    limits.reserved = true;
    return limits;
  }
  if (seq_level_idx == 31) {
    // §5.18.7.3 special case: maxTileWidthSb = sbCols, maxTileAreaSb = total.
    // Table A.9 has no row for 31 ("maximum parameters").
    limits.width_scaling_factor = 1;
    limits.area_scaling_factor = 1;
    limits.max_tiles = kMaxTileColumns * kMaxTileRows;
    limits.max_tile_cols = kMaxTileColumns;
    return limits;
  }
  if (seq_level_idx >= 22) {
    limits.reserved = true;
    return limits;
  }
  // §9.2 Table: tier-0 width scaling is 4 for levels 2.0-6.3 (idx 0-13) and 8
  // for 7.0-8.3 (idx 14-21). The old (idx <= 17) ? 4 : 8 split gave levels
  // 7.0-7.3 a factor of 4, raising minLog2TileCols by one.
  if (seq_tier == 0) {
    limits.width_scaling_factor = (seq_level_idx <= 13) ? 4 : 8;
    limits.area_scaling_factor =
        (seq_level_idx <= 13) ? 4 : ((seq_level_idx <= 17) ? 8 : 16);
  } else {
    limits.width_scaling_factor =
        (seq_level_idx <= 13) ? 4 : ((seq_level_idx <= 17) ? 8 : 16);
    limits.area_scaling_factor =
        (seq_level_idx <= 13) ? 4 : ((seq_level_idx <= 17) ? 16 : 32);
  }
  const LevelTileRow& row =
      kLevelTileRows[static_cast<std::size_t>(seq_level_idx)];
  limits.max_tiles = row.max_tiles;
  limits.max_tile_cols = row.max_tile_cols;
  return limits;
}

int minimum_level_for_grid(int cols, int rows) {
  if (cols <= 0 || rows <= 0) {
    return 31;
  }
  const std::int64_t tiles =
      static_cast<std::int64_t>(cols) * static_cast<std::int64_t>(rows);
  for (int level = 0; level <= 21; ++level) {
    const LevelTileLimits limits = level_tile_limits(level, 0);
    if (limits.reserved) {
      continue;
    }
    if (limits.max_tile_cols >= cols && limits.max_tiles >= tiles) {
      return level;
    }
  }
  return 31;
}

int tile_log2(int blk_size, int target) {
  if (blk_size <= 0) {
    return -1;
  }
  int k = 0;
  // The spec loop is unbounded; bound it so malformed input cannot hang.
  while ((blk_size << k) < target) {
    ++k;
    if (k > 30) {
      return -1;
    }
  }
  return k;
}

int mi_cols(int width) { return 2 * ((width + 7) >> 3); }
int mi_rows(int height) { return 2 * ((height + 7) >> 3); }

int sb_cols(int width, int sb_size) {
  const int sb4x4 = sb_size / 4;
  const int sb_shift = log2_exact(sb4x4);
  if (sb_shift < 0) {
    return 0;
  }
  return (mi_cols(width) + sb4x4 - 1) >> sb_shift;
}

int sb_rows(int height, int sb_size) {
  const int sb4x4 = sb_size / 4;
  const int sb_shift = log2_exact(sb4x4);
  if (sb_shift < 0) {
    return 0;
  }
  return (mi_rows(height) + sb4x4 - 1) >> sb_shift;
}

bool is_power_of_two(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

int log2_exact(int value) {
  if (!is_power_of_two(value)) {
    return -1;
  }
  int bits = 0;
  while (value > 1) {
    value >>= 1;
    ++bits;
  }
  return bits;
}

std::uint64_t TileRect::pixels() const { return pixel_area(x0, y0, x1, y1); }

std::uint64_t TileRect::sb_area() const {
  return static_cast<std::uint64_t>(sb_cols) *
         static_cast<std::uint64_t>(sb_rows);
}

bool TileRect::contains(int x, int y) const {
  return x >= x0 && x < x1 && y >= y0 && y < y1;
}

const TileRect* TileLayout::find(int row, int col) const {
  if (row < 0 || row >= rows || col < 0 || col >= cols) {
    return nullptr;
  }
  const std::size_t index = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                            static_cast<std::size_t>(col);
  if (index >= tiles.size()) {
    return nullptr;
  }
  return &tiles[index];
}

const TileRect* TileLayout::at_index(int index) const {
  if (index < 0 || static_cast<std::size_t>(index) >= tiles.size()) {
    return nullptr;
  }
  return &tiles[static_cast<std::size_t>(index)];
}

const char* layout_status_name(LayoutStatus status) {
  switch (status) {
    case LayoutStatus::kOk:
      return "ok";
    case LayoutStatus::kInvalidGeometry:
      return "invalid-geometry";
    case LayoutStatus::kInvalidLayout:
      return "invalid-layout";
    case LayoutStatus::kLevelConstraint:
      return "level-constraint";
    case LayoutStatus::kNotRepresentable:
      return "not-representable";
  }
  return "?";
}

LayoutResult plan_tile_layout(int width, int height, int cols, int rows,
                              int sb_size, int seq_level_idx, int seq_tier) {
  LayoutResult result;
  auto fail = [&result](LayoutStatus status, std::string message) {
    result.status = status;
    result.error = std::move(message);
    return result;
  };

  if (width <= 0 || height <= 0) {
    return fail(LayoutStatus::kInvalidGeometry,
                "frame width and height must be positive");
  }
  if (sb_size != kSuperblockPixels64 && sb_size != kSuperblockPixels128) {
    return fail(LayoutStatus::kInvalidGeometry,
                "superblock size must be 64 or 128");
  }
  if (!is_power_of_two(cols) || !is_power_of_two(rows)) {
    return fail(LayoutStatus::kInvalidLayout,
                "tile columns and rows must be powers of two");
  }
  if (cols > kMaxTileColumns || rows > kMaxTileRows) {
    return fail(LayoutStatus::kInvalidLayout,
                "tile grid exceeds MAX_TILE_COLS / MAX_TILE_ROWS (64x64)");
  }

  TileLayout layout;
  layout.width = width;
  layout.height = height;
  layout.sb_size = sb_size;
  layout.sb_cols = sb_cols(width, sb_size);
  layout.sb_rows = sb_rows(height, sb_size);
  layout.cols = cols;
  layout.rows = rows;
  layout.cols_log2 = log2_exact(cols);
  layout.rows_log2 = log2_exact(rows);
  layout.seq_level_idx = seq_level_idx;
  layout.seq_tier = seq_tier;
  if (layout.sb_cols <= 0 || layout.sb_rows <= 0) {
    return fail(LayoutStatus::kInvalidGeometry, "empty superblock grid");
  }

  const int sb4x4 = sb_size / 4;
  const int sb_shift = log2_exact(sb4x4);
  if (seq_level_idx < 0 || seq_level_idx > 31 || seq_tier < 0 || seq_tier > 1) {
    return fail(LayoutStatus::kLevelConstraint, "invalid level or tier");
  }

  if (seq_level_idx == 31) {
    layout.max_tile_width_sb = layout.sb_cols;
    layout.max_tile_area_sb = layout.sb_cols * layout.sb_rows;
    layout.max_tiles = kMaxTileColumns * kMaxTileRows;
    layout.max_tile_cols = kMaxTileColumns;
  } else {
    const LevelTileLimits limits = level_tile_limits(seq_level_idx, seq_tier);
    if (limits.reserved) {
      return fail(LayoutStatus::kLevelConstraint,
                  "reserved sequence level / tier combination");
    }
    layout.max_tile_width_sb =
        (limits.width_scaling_factor * kMaxTileWidthPixels) >> (sb_shift + 4);
    layout.max_tile_area_sb =
        (limits.area_scaling_factor * kMaxTileAreaPixels) >>
        (2 * (sb_shift + 2) + 2);
    layout.max_tiles = limits.max_tiles;
    layout.max_tile_cols = limits.max_tile_cols;
  }
  if (layout.max_tile_width_sb <= 0 || layout.max_tile_area_sb <= 0 ||
      layout.max_tiles <= 0 || layout.max_tile_cols <= 0) {
    return fail(LayoutStatus::kLevelConstraint,
                "level yields a non-positive tile bound");
  }
  // Annex A Table A.9. These are absolute counts, independent of the §9.2
  // width/area scaling factors the existing planner already modelled.
  if (cols > layout.max_tile_cols) {
    std::ostringstream out;
    out << "tile column count " << cols << " exceeds the level's MaxTileCols "
        << layout.max_tile_cols;
    return fail(LayoutStatus::kLevelConstraint, out.str());
  }
  if (cols * rows > layout.max_tiles) {
    std::ostringstream out;
    out << "tile count " << cols * rows << " exceeds the level's MaxTiles "
        << layout.max_tiles;
    return fail(LayoutStatus::kLevelConstraint, out.str());
  }

  layout.min_log2_tile_cols =
      tile_log2(layout.max_tile_width_sb, layout.sb_cols);
  // maxLog2TileCols is §5.18.7.3's syntax bound (Min(sbCols, MAX_TILE_COLS)),
  // not Table A.9 MaxTileCols: the decoder derives the increment-bit count from
  // the same expression, so capping it would shift tile_info().
  layout.max_log2_tile_cols =
      tile_log2(1, std::min(layout.sb_cols, kMaxTileColumns));
  layout.max_log2_tile_rows =
      tile_log2(1, std::min(layout.sb_rows, kMaxTileRows));
  layout.min_log2_tiles =
      std::max(layout.min_log2_tile_cols,
               min_log2_for_area(layout.max_tile_area_sb, layout.sb_rows,
                                 layout.sb_cols));
  if (layout.min_log2_tile_cols < 0 || layout.max_log2_tile_cols < 0 ||
      layout.max_log2_tile_rows < 0 || layout.min_log2_tiles < 0) {
    return fail(LayoutStatus::kLevelConstraint, "failed to derive tile limits");
  }
  if (layout.cols_log2 > layout.max_log2_tile_cols) {
    return fail(LayoutStatus::kLevelConstraint,
                "tile column count exceeds the frame/level maximum");
  }
  if (layout.rows_log2 > layout.max_log2_tile_rows) {
    return fail(LayoutStatus::kLevelConstraint,
                "tile row count exceeds the frame/level maximum");
  }
  // §5.18.7.3: tileColsLog2 starts at minLog2TileCols, then
  // tileRowsLog2 starts at max(minLog2Tiles - tileColsLog2, 0).
  if (layout.cols_log2 < layout.min_log2_tile_cols) {
    return fail(LayoutStatus::kLevelConstraint,
                "tile column count is below the level's minimum");
  }
  const int min_rows_log2 =
      std::max(layout.min_log2_tiles - layout.cols_log2, 0);
  if (layout.rows_log2 < min_rows_log2) {
    return fail(LayoutStatus::kLevelConstraint,
                "tile row count is below the level's minimum");
  }
  const int requested_tiles = cols * rows;
  const int min_tiles = 1 << layout.min_log2_tiles;
  if (requested_tiles < min_tiles) {
    return fail(LayoutStatus::kLevelConstraint,
                "layout has fewer tiles than the level's minimum tile count");
  }
  const int max_tiles = (1 << layout.max_log2_tile_cols) *
                        (1 << layout.max_log2_tile_rows);
  if (min_tiles > max_tiles) {
    return fail(LayoutStatus::kLevelConstraint,
                "level minimum tile count exceeds the frame's maximum");
  }

  layout.col_starts_sb =
      uniform_starts(mi_cols(width), sb_size, layout.cols_log2);
  layout.row_starts_sb =
      uniform_starts(mi_rows(height), sb_size, layout.rows_log2);
  if (static_cast<int>(layout.col_starts_sb.size()) != cols ||
      static_cast<int>(layout.row_starts_sb.size()) != rows) {
    std::ostringstream out;
    out << "layout " << cols << "x" << rows << " is not representable at "
        << width << "x" << height << " (uniform spacing yields "
        << layout.col_starts_sb.size() << "x" << layout.row_starts_sb.size()
        << " tiles)";
    return fail(LayoutStatus::kNotRepresentable, out.str());
  }

  layout.tiles.reserve(static_cast<std::size_t>(requested_tiles));
  layout.boundaries.reserve(static_cast<std::size_t>(requested_tiles));

  int index = 0;
  for (int row = 0; row < rows; ++row) {
    const int sb_row_start = layout.row_starts_sb[static_cast<std::size_t>(row)];
    const int sb_row_end = (row + 1 < rows)
                               ? layout.row_starts_sb[static_cast<std::size_t>(row + 1)]
                               : layout.sb_rows;
    const int sb_rows_tile = sb_row_end - sb_row_start;
    const int y0 = sb_row_start * sb_size;
    const int y1 = std::min(sb_row_end * sb_size, height);
    for (int col = 0; col < cols; ++col) {
      const int sb_col_start =
          layout.col_starts_sb[static_cast<std::size_t>(col)];
      const int sb_col_end =
          (col + 1 < cols)
              ? layout.col_starts_sb[static_cast<std::size_t>(col + 1)]
              : layout.sb_cols;
      const int sb_cols_tile = sb_col_end - sb_col_start;
      const int x0 = sb_col_start * sb_size;
      const int x1 = std::min(sb_col_end * sb_size, width);

      TileRect tile;
      tile.index = index;
      tile.row = row;
      tile.col = col;
      tile.x0 = x0;
      tile.y0 = y0;
      tile.x1 = x1;
      tile.y1 = y1;
      tile.sb_col_start = sb_col_start;
      tile.sb_row_start = sb_row_start;
      tile.sb_cols = sb_cols_tile;
      tile.sb_rows = sb_rows_tile;

      if (x0 < 0 || y0 < 0 || x1 <= x0 || y1 <= y0) {
        return fail(LayoutStatus::kInvalidGeometry, describe_rect(tile));
      }
      if (tile.sb_cols <= 0 || tile.sb_rows <= 0) {
        return fail(LayoutStatus::kInvalidGeometry,
                    "tile has an empty superblock span");
      }
      if (sb_cols_tile > layout.max_tile_width_sb) {
        return fail(LayoutStatus::kLevelConstraint,
                    "tile width exceeds the level maximum");
      }
      if (tile.sb_area() >
          static_cast<std::uint64_t>(layout.max_tile_area_sb)) {
        return fail(LayoutStatus::kLevelConstraint,
                    "tile area exceeds the level maximum");
      }

      TileBoundaries boundary;
      boundary.left =
          (col == 0) ? TileEdge::kFrameBoundary : TileEdge::kInterior;
      boundary.top =
          (row == 0) ? TileEdge::kFrameBoundary : TileEdge::kInterior;
      boundary.right =
          (col == cols - 1) ? TileEdge::kFrameBoundary : TileEdge::kInterior;
      boundary.bottom =
          (row == rows - 1) ? TileEdge::kFrameBoundary : TileEdge::kInterior;
      boundary.touches_frame_left = (col == 0);
      boundary.touches_frame_top = (row == 0);
      boundary.touches_frame_right = (col == cols - 1);
      boundary.touches_frame_bottom = (row == rows - 1);
      boundary.has_partial_superblock_column =
          (col == cols - 1) && (width % sb_size != 0);
      boundary.has_partial_superblock_row =
          (row == rows - 1) && (height % sb_size != 0);

      layout.tiles.push_back(tile);
      layout.boundaries.push_back(boundary);
      ++index;
    }
  }

  if (!validate_tile_layout(layout, result.error)) {
    result.status = LayoutStatus::kInvalidLayout;
    return result;
  }
  result.layout = std::move(layout);
  result.status = LayoutStatus::kOk;
  return result;
}

bool validate_tile_layout(const TileLayout& layout, std::string& error) {
  error.clear();
  auto fail = [&error](std::string message) {
    error = std::move(message);
    return false;
  };

  if (layout.width <= 0 || layout.height <= 0) {
    return fail("layout has a non-positive frame size");
  }
  if (layout.sb_size != kSuperblockPixels64 &&
      layout.sb_size != kSuperblockPixels128) {
    return fail("layout has an invalid superblock size");
  }
  if (layout.cols <= 0 || layout.rows <= 0) {
    return fail("layout has a non-positive tile grid");
  }
  if (layout.cols_log2 != log2_exact(layout.cols) ||
      layout.rows_log2 != log2_exact(layout.rows)) {
    return fail("layout tile grid is not a power of two");
  }
  if (layout.seq_level_idx < 0 || layout.seq_level_idx > 31 ||
      layout.seq_tier < 0 || layout.seq_tier > 1) {
    return fail("layout has an invalid sequence level or tier");
  }
  if (layout.seq_level_idx != 31) {
    const LevelTileLimits limits =
        level_tile_limits(layout.seq_level_idx, layout.seq_tier);
    if (limits.reserved) {
      return fail("layout signals a reserved sequence level / tier");
    }
    if (layout.cols > limits.max_tile_cols) {
      return fail("layout tile column count exceeds the level MaxTileCols");
    }
    if (layout.tile_count() > limits.max_tiles) {
      return fail("layout tile count exceeds the level MaxTiles");
    }
  }
  if (layout.sb_cols != sb_cols(layout.width, layout.sb_size) ||
      layout.sb_rows != sb_rows(layout.height, layout.sb_size)) {
    return fail("layout superblock dimensions do not match the frame size");
  }
  if (static_cast<int>(layout.col_starts_sb.size()) != layout.cols ||
      static_cast<int>(layout.row_starts_sb.size()) != layout.rows) {
    return fail("layout start arrays do not match the tile grid");
  }
  if (layout.tile_count() != layout.cols * layout.rows) {
    return fail("layout tile count does not match cols * rows");
  }
  if (static_cast<int>(layout.boundaries.size()) != layout.tile_count()) {
    return fail("layout boundary array does not match the tile count");
  }

  int previous = -1;
  for (int i = 0; i < layout.cols; ++i) {
    const int start = layout.col_starts_sb[static_cast<std::size_t>(i)];
    if (start <= previous || start < 0 || start >= layout.sb_cols) {
      return fail("layout column starts are not strictly increasing");
    }
    previous = start;
  }
  previous = -1;
  for (int i = 0; i < layout.rows; ++i) {
    const int start = layout.row_starts_sb[static_cast<std::size_t>(i)];
    if (start <= previous || start < 0 || start >= layout.sb_rows) {
      return fail("layout row starts are not strictly increasing");
    }
    previous = start;
  }
  if (layout.col_starts_sb.front() != 0 ||
      layout.row_starts_sb.front() != 0) {
    return fail("layout starts must begin at superblock 0");
  }

  std::uint64_t pixels = 0;
  std::uint64_t sb_area = 0;
  for (int i = 0; i < layout.tile_count(); ++i) {
    const TileRect& tile = layout.tiles[static_cast<std::size_t>(i)];
    const TileBoundaries& boundary =
        layout.boundaries[static_cast<std::size_t>(i)];
    if (tile.index != i) {
      return fail("layout tile index is not raster order");
    }
    if (tile.row != i / layout.cols || tile.col != i % layout.cols) {
      return fail("layout tile row/column is not raster order");
    }
    if (tile.x0 < 0 || tile.y0 < 0 || tile.x1 > layout.width ||
        tile.y1 > layout.height || tile.x1 <= tile.x0 ||
        tile.y1 <= tile.y0) {
      return fail("layout tile rectangle is outside the visible frame");
    }
    if (tile.sb_col_start !=
            layout.col_starts_sb[static_cast<std::size_t>(tile.col)] ||
        tile.sb_row_start !=
            layout.row_starts_sb[static_cast<std::size_t>(tile.row)]) {
      return fail("layout tile superblock origin does not match the starts");
    }
    const int expected_cols =
        ((tile.col + 1 < layout.cols)
             ? layout.col_starts_sb[static_cast<std::size_t>(tile.col + 1)]
             : layout.sb_cols) -
        tile.sb_col_start;
    const int expected_rows =
        ((tile.row + 1 < layout.rows)
             ? layout.row_starts_sb[static_cast<std::size_t>(tile.row + 1)]
             : layout.sb_rows) -
        tile.sb_row_start;
    if (tile.sb_cols != expected_cols || tile.sb_rows != expected_rows) {
      return fail("layout tile superblock span does not match the starts");
    }
    if (tile.x0 != tile.sb_col_start * layout.sb_size ||
        tile.y0 != tile.sb_row_start * layout.sb_size) {
      return fail("layout tile pixel origin does not match the superblock start");
    }

    const TileEdge expected_left =
        (tile.col == 0) ? TileEdge::kFrameBoundary : TileEdge::kInterior;
    const TileEdge expected_top =
        (tile.row == 0) ? TileEdge::kFrameBoundary : TileEdge::kInterior;
    const TileEdge expected_right = (tile.col == layout.cols - 1)
                                        ? TileEdge::kFrameBoundary
                                        : TileEdge::kInterior;
    const TileEdge expected_bottom = (tile.row == layout.rows - 1)
                                         ? TileEdge::kFrameBoundary
                                         : TileEdge::kInterior;
    if (boundary.left != expected_left || boundary.top != expected_top ||
        boundary.right != expected_right || boundary.bottom != expected_bottom) {
      return fail("layout boundary edges do not match the tile grid");
    }
    if (boundary.has_partial_superblock_column !=
            ((tile.col == layout.cols - 1) &&
             (layout.width % layout.sb_size != 0)) ||
        boundary.has_partial_superblock_row !=
            ((tile.row == layout.rows - 1) &&
             (layout.height % layout.sb_size != 0))) {
      return fail("layout partial-superblock flags are inconsistent");
    }

    pixels += tile.pixels();
    sb_area += tile.sb_area();
  }

  const std::uint64_t expected_pixels =
      static_cast<std::uint64_t>(layout.width) *
      static_cast<std::uint64_t>(layout.height);
  if (pixels != expected_pixels) {
    return fail("layout tile pixel areas do not cover the frame exactly once");
  }
  const std::uint64_t expected_sb =
      static_cast<std::uint64_t>(layout.sb_cols) *
      static_cast<std::uint64_t>(layout.sb_rows);
  if (sb_area != expected_sb) {
    return fail("layout tile superblock areas do not cover the grid exactly once");
  }
  return true;
}

}  // namespace av2enc::tiles
