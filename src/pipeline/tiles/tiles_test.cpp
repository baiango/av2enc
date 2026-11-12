// Standalone tests for the tile geometry, tile-group, and SVC layer planners.
// No CUDA, no bitstream, no external dependencies.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "svc_layers.h"
#include "tile_group.h"
#include "tile_layout.h"
#include "core/conformance/conformance_tables.h"

namespace {

int g_failures = 0;

void check(bool condition, const char* expression, const char* file, int line) {
  if (!condition) {
    ++g_failures;
    std::cerr << file << ':' << line << ": CHECK failed: " << expression << '\n';
  }
}

#define CHECK(expression) \
  ::check((expression), #expression, __FILE__, __LINE__)

const av2enc::tiles::TileRect& tile(const av2enc::tiles::TileLayout& layout,
                                    int row, int col) {
  return layout.tiles[static_cast<std::size_t>(row * layout.cols + col)];
}

void expect_widths(const av2enc::tiles::TileLayout& layout,
                   const std::vector<int>& expected) {
  CHECK(static_cast<int>(expected.size()) == layout.cols);
  for (int col = 0; col < layout.cols && col < static_cast<int>(expected.size());
       ++col) {
    CHECK(tile(layout, 0, col).pixel_width() == expected[static_cast<std::size_t>(col)]);
  }
}

void test_geometry() {
  using av2enc::tiles::plan_tile_layout;

  {
    const auto result = plan_tile_layout(1920, 1080, 4, 2, 64, 31, 0);
    CHECK(result.ok());
    const auto& layout = result.layout;
    CHECK(layout.sb_cols == 30);
    CHECK(layout.sb_rows == 17);
    CHECK(layout.tile_count() == 8);
    expect_widths(layout, {512, 512, 448, 448});
    CHECK(tile(layout, 0, 0).pixel_height() == 512);
    CHECK(tile(layout, 1, 0).pixel_height() == 568);
    CHECK(tile(layout, 0, 0).pixel_width() == 512);
    CHECK(layout.tiles.front().x0 == 0 && layout.tiles.front().y0 == 0);
    CHECK(layout.tiles.back().x1 == 1920 && layout.tiles.back().y1 == 1080);
    CHECK(layout.boundaries.back().has_partial_superblock_row);
    CHECK(!layout.boundaries.back().has_partial_superblock_column);
    CHECK(layout.boundaries[2].left == av2enc::tiles::TileEdge::kInterior);
    CHECK(layout.boundaries[4].top == av2enc::tiles::TileEdge::kInterior);
    std::uint64_t pixels = 0;
    for (const auto& rect : layout.tiles) {
      pixels += rect.pixels();
    }
    CHECK(pixels == 1920ull * 1080ull);
  }
  {
    const auto result = plan_tile_layout(1920, 1080, 8, 2, 64, 31, 0);
    CHECK(result.ok());
    expect_widths(result.layout, {256, 256, 256, 256, 256, 256, 192, 192});
  }
  {
    const auto result = plan_tile_layout(1920, 1080, 16, 2, 64, 31, 0);
    CHECK(result.ok());
    std::vector<int> widths;
    for (int col = 0; col < 16; ++col) {
      widths.push_back(tile(result.layout, 0, col).pixel_width());
    }
    CHECK(widths == std::vector<int>({128, 128, 128, 128, 128, 128, 128, 128,
                                      128, 128, 128, 128, 128, 128, 64, 64}));
  }
  {
    const auto result = plan_tile_layout(3840, 2160, 4, 2, 64, 31, 0);
    CHECK(result.ok());
    CHECK(result.layout.sb_cols == 60);
    CHECK(result.layout.sb_rows == 34);
    expect_widths(result.layout, {960, 960, 960, 960});
    CHECK(tile(result.layout, 0, 0).pixel_height() == 1088);
    CHECK(tile(result.layout, 1, 0).pixel_height() == 1072);
  }
  {
    const auto result = plan_tile_layout(3840, 2160, 8, 2, 64, 31, 0);
    CHECK(result.ok());
    expect_widths(result.layout,
                  {512, 512, 512, 512, 448, 448, 448, 448});
  }
  {
    const auto result = plan_tile_layout(3840, 2160, 16, 2, 64, 31, 0);
    CHECK(result.ok());
    std::vector<int> widths;
    for (int col = 0; col < 16; ++col) {
      widths.push_back(tile(result.layout, 0, col).pixel_width());
    }
    CHECK(widths == std::vector<int>({256, 256, 256, 256, 256, 256, 256, 256,
                                      256, 256, 256, 256, 192, 192, 192, 192}));
  }
  {
    // 640x360 is exact on the right edge and partial on the bottom edge.
    const auto result = plan_tile_layout(640, 360, 4, 2, 64, 31, 0);
    CHECK(result.ok());
    CHECK(result.layout.sb_cols == 10);
    CHECK(result.layout.sb_rows == 6);
    CHECK(!result.layout.boundaries.back().has_partial_superblock_column);
    CHECK(result.layout.boundaries.back().has_partial_superblock_row);
    CHECK(tile(result.layout, 1, 0).pixel_height() == 168);
  }
  {
    // One tile covers the frame exactly, including a non-multiple-of-64 size.
    const auto result = plan_tile_layout(641, 361, 1, 1, 64, 31, 0);
    CHECK(result.ok());
    CHECK(result.layout.tile_count() == 1);
    CHECK(result.layout.tiles.front().x1 == 641);
    CHECK(result.layout.tiles.front().y1 == 361);
    CHECK(result.layout.tiles.front().pixels() == 641ull * 361ull);
  }
}

void test_geometry_rejections() {
  using av2enc::tiles::LayoutStatus;
  using av2enc::tiles::plan_tile_layout;

  CHECK(plan_tile_layout(0, 1080, 1, 1).status ==
        LayoutStatus::kInvalidGeometry);
  CHECK(plan_tile_layout(1920, 1080, 3, 2).status ==
        LayoutStatus::kInvalidLayout);
  CHECK(plan_tile_layout(1920, 1080, 1, 3).status ==
        LayoutStatus::kInvalidLayout);
  CHECK(plan_tile_layout(1920, 1080, 128, 2).status ==
        LayoutStatus::kInvalidLayout);
  // A 64x64 frame cannot be split: the frame is one superblock, so the
  // column/row log2 maximum is zero before representability is even reached.
  CHECK(plan_tile_layout(64, 64, 2, 1).status ==
        LayoutStatus::kLevelConstraint);
  CHECK(plan_tile_layout(64, 64, 2, 2).status ==
        LayoutStatus::kLevelConstraint);
  // A small frame that passes the level maximum but has no room for four tile
  // columns is reported as not representable.
  CHECK(plan_tile_layout(128, 64, 2, 1).ok());
  CHECK(plan_tile_layout(192, 64, 4, 1).status ==
        LayoutStatus::kNotRepresentable);
  // 1080p cannot have more than 32 columns (sbCols = 30 -> log2 max = 5).
  CHECK(plan_tile_layout(1920, 1080, 64, 2).status ==
        LayoutStatus::kLevelConstraint);
  // A level-5.1 8K frame needs at least four tiles by MAX_TILE_AREA.
  CHECK(plan_tile_layout(7680, 4320, 1, 1, 64, 13, 0).status ==
        LayoutStatus::kLevelConstraint);
  CHECK(plan_tile_layout(7680, 4320, 2, 2, 64, 13, 0).ok());
  // The same level also forces at least two tile columns at 8K (tile width
  // bound), so 1x4 is invalid even though it has four tiles.
  CHECK(plan_tile_layout(7680, 4320, 1, 4, 64, 13, 0).status ==
        LayoutStatus::kLevelConstraint);
  CHECK(plan_tile_layout(7680, 4320, 4, 1, 64, 13, 0).ok());
  // Non-multiple-of-8 dimensions are still legal geometry.
  const auto odd = plan_tile_layout(641, 361, 2, 2, 64, 31, 0);
  CHECK(odd.ok());
}

void test_validate_layout() {
  auto result = av2enc::tiles::plan_tile_layout(1920, 1080, 8, 2, 64, 31, 0);
  CHECK(result.ok());
  std::string error;
  CHECK(av2enc::tiles::validate_tile_layout(result.layout, error));

  auto broken = result.layout;
  broken.tiles[3].x1 -= 1;
  CHECK(!av2enc::tiles::validate_tile_layout(broken, error));

  broken = result.layout;
  broken.boundaries[1].right = av2enc::tiles::TileEdge::kFrameBoundary;
  CHECK(!av2enc::tiles::validate_tile_layout(broken, error));

  broken = result.layout;
  broken.tiles[5].index = 4;
  CHECK(!av2enc::tiles::validate_tile_layout(broken, error));
}

void test_tile_group_plan() {
  using av2enc::tiles::plan_single_tile_group;
  using av2enc::tiles::tile_size_bytes_for;

  CHECK(tile_size_bytes_for(2) == 1);
  CHECK(tile_size_bytes_for(256) == 1);
  CHECK(tile_size_bytes_for(257) == 2);
  CHECK(tile_size_bytes_for(65536) == 2);
  CHECK(tile_size_bytes_for(65537) == 3);
  CHECK(tile_size_bytes_for(16777216) == 3);
  CHECK(tile_size_bytes_for(16777217) == 4);
  CHECK(tile_size_bytes_for(1) == 0);
  CHECK(tile_size_bytes_for(0) == 0);

  const auto layout_result =
      av2enc::tiles::plan_tile_layout(1920, 1080, 8, 2, 64, 31, 0);
  CHECK(layout_result.ok());
  const auto& layout = layout_result.layout;

  std::vector<std::uint32_t> sizes(static_cast<std::size_t>(layout.tile_count()),
                                   1000);
  sizes.back() = 2000;
  const auto plan = plan_single_tile_group(layout, sizes, 24, true);
  CHECK(plan.valid);
  CHECK(plan.tile_count == 16);
  CHECK(plan.tile_size_bytes == 2);
  CHECK(plan.size_field_count == 15);
  CHECK(plan.size_field_bytes == 30);
  CHECK(plan.header_bits == 32);  // 1 + 24 + 1 -> 26 -> 32
  CHECK(plan.header_bytes == 4);
  CHECK(plan.payload_bytes == 15ull * 1000ull + 2000ull);
  CHECK(plan.obu_payload_bytes == 4 + 30 + plan.payload_bytes);
  CHECK(plan.tile_size_minus_1.size() == 15);
  CHECK(plan.tile_size_minus_1.front() == 999);
  CHECK(plan.tile_payload_offsets.front() == 0);
  CHECK(plan.tile_payload_offsets[1] == 1000);
  CHECK(plan.tile_payload_offsets.back() == 15ull * 1000ull);
  CHECK(plan.largest_tile_bytes == 2000);
  CHECK(plan.smallest_tile_bytes == 1000);

  std::vector<std::uint32_t> too_small(static_cast<std::size_t>(layout.tile_count()),
                                      1000);
  too_small[4] = 1;
  CHECK(!plan_single_tile_group(layout, too_small, 0, true).valid);

  std::vector<std::uint32_t> wrong_count(3, 1000);
  CHECK(!plan_single_tile_group(layout, wrong_count, 0, true).valid);
  CHECK(!plan_single_tile_group(layout, sizes, 8, false).valid);

  const auto one_tile_result =
      av2enc::tiles::plan_tile_layout(256, 256, 1, 1, 64, 31, 0);
  CHECK(one_tile_result.ok());
  std::vector<std::uint32_t> one_size = {500};
  const auto one = plan_single_tile_group(one_tile_result.layout, one_size, 0,
                                          true);
  CHECK(one.valid);
  CHECK(one.tile_count == 1);
  CHECK(one.size_field_count == 0);
  CHECK(one.size_field_bytes == 0);
  CHECK(one.tile_size_minus_1.empty());
  CHECK(one.header_bits == 8);  // one is_first bit plus alignment
  CHECK(one.obu_payload_bytes == 1 + 500);
}

void test_svc_sequence_and_dependencies() {
  using namespace av2enc::tiles::svc;

  LayerId id;
  id.xlayer_id = 0;
  id.mlayer_id = 2;
  id.tlayer_id = 1;
  std::string error;
  CHECK(validate_layer_id(id, error));
  id.xlayer_id = 32;
  CHECK(!validate_layer_id(id, error));
  id.xlayer_id = 0;
  id.mlayer_id = 8;
  CHECK(!validate_layer_id(id, error));
  id.mlayer_id = 0;
  id.tlayer_id = 4;
  CHECK(!validate_layer_id(id, error));

  const SequenceLayerConfig single =
      default_sequence_layer_config(3, 3, true);
  SequenceLayerConfig valid_single = single;
  valid_single.max_width = 1920;
  valid_single.max_height = 1080;
  CHECK(validate_sequence_layer_config(valid_single, error));

  SequenceLayerConfig bad_single = valid_single;
  bad_single.max_mlayer_id = 1;
  CHECK(!validate_sequence_layer_config(bad_single, error));

  SequenceLayerConfig config =
      default_sequence_layer_config(2, 2, false);
  config.max_width = 1920;
  config.max_height = 1080;
  CHECK(validate_sequence_layer_config(config, error));
  CHECK(config.dependencies.mlayer[2][0]);
  CHECK(config.dependencies.mlayer[2][2]);
  CHECK(!config.dependencies.mlayer[0][2]);

  LayerDependencyMaps broken = config.dependencies;
  broken.mlayer[2][1] = true;   // 2 -> 1
  broken.mlayer[1][0] = true;   // 1 -> 0
  broken.mlayer[2][0] = false;  // violates transitivity
  CHECK(!validate_dependency_maps(broken, 2, 2, error));

  CHECK(mlayer_dependencies_satisfied(0b111, config.dependencies, error));
  CHECK(!mlayer_dependencies_satisfied(0b100, config.dependencies, error));
  CHECK(!mlayer_dependencies_satisfied(0b110, config.dependencies, error));

  std::array<std::uint8_t, kEmbeddedLayerCount> tmaps{};
  tmaps[0] = 0b1111;
  tmaps[1] = 0b0111;
  tmaps[2] = 0b0011;
  CHECK(tlayer_dependencies_satisfied(0b111, tmaps, config.dependencies, error));
  tmaps[2] = 0b0100;  // t2 without t0/t1
  CHECK(!tlayer_dependencies_satisfied(0b111, tmaps, config.dependencies, error));
}

void test_svc_lcr_and_operating_points() {
  using namespace av2enc::tiles::svc;

  std::string error;
  SequenceLayerConfig sequence = default_sequence_layer_config(1, 1, false);
  sequence.max_width = 1920;
  sequence.max_height = 1080;
  CHECK(validate_sequence_layer_config(sequence, error));

  LayerConfigRecord record;
  record.is_global = true;
  record.xlayer_id = kGlobalXLayerId;
  record.xlayer_map = (1u << 0) | (1u << 1);
  for (int x = 0; x < 2; ++x) {
    ExtendedLayerConfig layer;
    layer.xlayer_id = static_cast<std::uint8_t>(x);
    layer.mlayer_map = 0b11;
    layer.tlayer_map[0] = 0b11;
    layer.tlayer_map[1] = 0b11;
    EmbeddedLayerConfig base;
    base.mlayer_id = 0;
    base.tlayer_map = 0b11;
    EmbeddedLayerConfig enhancement;
    enhancement.mlayer_id = 1;
    enhancement.tlayer_map = 0b11;
    layer.embedded = {base, enhancement};
    record.layers.push_back(layer);
  }
  CHECK(validate_layer_config_record(record, sequence, error));

  LayerConfigRecord missing_dep = record;
  missing_dep.layers[0].mlayer_map = 0b10;  // m1 without m0
  CHECK(!validate_layer_config_record(missing_dep, sequence, error));

  LayerConfigRecord missing_tdep = record;
  missing_tdep.layers[0].tlayer_map[0] = 0b10;  // t1 without t0
  missing_tdep.layers[0].embedded[0].tlayer_map = 0b10;
  CHECK(!validate_layer_config_record(missing_tdep, sequence, error));

  LayerConfigRecord map_mismatch = record;
  map_mismatch.xlayer_map = 1u << 0;  // layer 1 is listed but absent from map
  CHECK(!validate_layer_config_record(map_mismatch, sequence, error));

  LayerConfigRecord local = record;
  local.is_global = false;
  local.xlayer_id = 0;
  local.xlayer_map = 0;
  local.layers.resize(1);
  CHECK(validate_layer_config_record(local, sequence, error));
  LayerConfigRecord local_two = local;
  local_two.layers.push_back(record.layers[1]);
  CHECK(!validate_layer_config_record(local_two, sequence, error));

  OperatingPointSet set;
  set.xlayer_id = kGlobalXLayerId;
  set.ops_id = 0;
  OperatingPoint point;
  point.op_index = 0;
  point.xlayer_map = (1u << 0) | (1u << 1);
  for (int x = 0; x < 2; ++x) {
    point.mlayer_map[static_cast<std::size_t>(x)] = 0b11;
    point.tlayer_map[static_cast<std::size_t>(x)][0] = 0b11;
    point.tlayer_map[static_cast<std::size_t>(x)][1] = 0b11;
  }
  set.points.push_back(point);
  CHECK(validate_operating_point_set(set, sequence, &record, error));

  OperatingPointSet missing = set;
  missing.points[0].mlayer_map[0] = 0b10;
  CHECK(!validate_operating_point_set(missing, sequence, &record, error));

  OperatingPointSet unselected_info = set;
  unselected_info.points[0].mlayer_map[2] = 0b11;
  CHECK(!validate_operating_point_set(unselected_info, sequence, &record, error));

  OperatingPointSet local_set;
  local_set.xlayer_id = 0;
  local_set.ops_id = 1;
  OperatingPoint local_point;
  local_point.op_index = 0;
  local_point.xlayer_map = 1u << 0;
  local_point.mlayer_map[0] = 0b11;
  local_point.tlayer_map[0][0] = 0b11;
  local_point.tlayer_map[0][1] = 0b11;
  local_set.points.push_back(local_point);
  CHECK(validate_operating_point_set(local_set, sequence, &local, error));
  OperatingPointSet local_bad = local_set;
  local_bad.points[0].xlayer_map = (1u << 0) | (1u << 1);
  CHECK(!validate_operating_point_set(local_bad, sequence, &local, error));
  // A local OPS must not be validated against the global LCR, and vice versa.
  CHECK(!validate_operating_point_set(local_set, sequence, &record, error));
  CHECK(!validate_operating_point_set(set, sequence, &local, error));

  RetentionMap retention;
  CHECK(build_retention_map(set, 0, sequence.dependencies, retention, error));
  CHECK(retention.xlayer_selected[0]);
  CHECK(retention.xlayer_selected[1]);
  CHECK(retention.retain[0][0][0]);
  CHECK(retention.retain[0][0][1]);
  CHECK(retention.retain[0][1][0]);
  CHECK(retention.retain[0][1][1]);
  CHECK(retention.retain[1][1][1]);
  CHECK(!retention.retain[2][0][0]);

  // A point that selects m1/t1 without m0/t0 is rejected before retention.
  OperatingPointSet unclosed = set;
  unclosed.points[0].mlayer_map[1] = 0b10;
  unclosed.points[0].tlayer_map[1][1] = 0b10;
  CHECK(!validate_operating_point_set(unclosed, sequence, &record, error));
}

// §9.2 tile scaling factors and Annex A Table A.9 MaxTiles / MaxTileCols.
// level_tile_limits() keeps a standalone copy of the tables; this cross-check
// fails if it ever drifts from core/conformance/conformance_tables.cpp.
void test_level_tile_limits() {
  namespace conf = av2enc::conformance;
  using av2enc::tiles::level_tile_limits;
  for (int tier = 0; tier <= 1; ++tier) {
    for (int level = 0; level <= 21; ++level) {
      const av2enc::tiles::LevelTileLimits limits =
          level_tile_limits(level, tier);
      const conf::TileScaleFactor factor = conf::tile_scale_factor(level, tier);
      const conf::LevelInfo& info = conf::level_at(level);
      CHECK(!limits.reserved && !factor.reserved);
      CHECK(limits.width_scaling_factor == factor.width);
      CHECK(limits.area_scaling_factor == factor.area);
      CHECK(limits.max_tiles == info.max_tiles);
      CHECK(limits.max_tile_cols == info.max_tile_cols);
    }
  }
  // §9.2 regression: levels 7.0-7.3 tier 0 scale the width by 8, not 4.
  const auto l70 = level_tile_limits(14, 0);
  CHECK(l70.width_scaling_factor == 8 && l70.area_scaling_factor == 8);
  const auto l63 = level_tile_limits(13, 0);
  CHECK(l63.width_scaling_factor == 4 && l63.area_scaling_factor == 4);
  CHECK(level_tile_limits(22, 0).reserved);
  CHECK(level_tile_limits(14, 2).reserved);

  // A tier-0 level-7.0 8192x512 frame fits one tile by the width bound: the
  // old factor of 4 wrongly required two columns.
  {
    const auto one = av2enc::tiles::plan_tile_layout(8192, 512, 1, 1, 64, 14, 0);
    CHECK(one.ok());
    CHECK(one.layout.min_log2_tile_cols == 0);
    const auto two = av2enc::tiles::plan_tile_layout(8192, 512, 2, 1, 64, 14, 0);
    CHECK(two.ok());
    CHECK(two.layout.min_log2_tile_cols == 0);
  }
}

// Table A.9 MaxTiles / MaxTileCols must reject grids the validator rejects.
void test_level_tile_count_limits() {
  using av2enc::tiles::LayoutStatus;
  using av2enc::tiles::minimum_level_for_grid;
  using av2enc::tiles::plan_tile_layout;

  // Level 5.1 (idx 7): MaxTileCols = 8.
  {
    const auto ok = plan_tile_layout(1024, 128, 8, 2, 64, 7, 0);
    CHECK(ok.ok());
    const auto bad = plan_tile_layout(1024, 128, 16, 2, 64, 7, 0);
    CHECK(bad.status == LayoutStatus::kLevelConstraint);
    CHECK(bad.error.find("MaxTileCols") != std::string::npos);
    CHECK(plan_tile_layout(1024, 128, 16, 2, 64, 10, 0).ok());
  }
  // Level 3.1 (idx 3): MaxTiles = 16, MaxTileCols = 6.
  {
    const auto many = plan_tile_layout(1280, 720, 4, 8, 64, 3, 0);
    CHECK(many.status == LayoutStatus::kLevelConstraint);
    CHECK(many.error.find("MaxTiles") != std::string::npos);
    const auto cols = plan_tile_layout(1280, 720, 8, 2, 64, 3, 0);
    CHECK(cols.status == LayoutStatus::kLevelConstraint);
    CHECK(cols.error.find("MaxTileCols") != std::string::npos);
    CHECK(plan_tile_layout(1280, 720, 4, 2, 64, 3, 0).ok());
  }
  // validate_tile_layout() re-checks the level on an already-built layout.
  {
    const auto wide = plan_tile_layout(1024, 128, 16, 2, 64, 31, 0);
    CHECK(wide.ok());
    av2enc::tiles::TileLayout layout = wide.layout;
    layout.seq_level_idx = 7;
    std::string error;
    CHECK(!av2enc::tiles::validate_tile_layout(layout, error));
    CHECK(error.find("MaxTileCols") != std::string::npos);
  }
  // Writers can pick the lowest level that admits a grid.
  CHECK(minimum_level_for_grid(2, 2) == 0);
  CHECK(minimum_level_for_grid(8, 2) == 4);
  CHECK(minimum_level_for_grid(16, 2) == 10);
  CHECK(minimum_level_for_grid(1, 1) == 0);
}

void test_determinism() {
  const auto a = av2enc::tiles::plan_tile_layout(1920, 1080, 8, 2, 64, 13, 0);
  const auto b = av2enc::tiles::plan_tile_layout(1920, 1080, 8, 2, 64, 13, 0);
  CHECK(a.ok() && b.ok());
  CHECK(a.layout.tile_count() == b.layout.tile_count());
  CHECK(a.layout.col_starts_sb == b.layout.col_starts_sb);
  CHECK(a.layout.row_starts_sb == b.layout.row_starts_sb);
  for (int i = 0; i < a.layout.tile_count(); ++i) {
    const auto& ta = a.layout.tiles[static_cast<std::size_t>(i)];
    const auto& tb = b.layout.tiles[static_cast<std::size_t>(i)];
    CHECK(ta.index == tb.index && ta.row == tb.row && ta.col == tb.col);
    CHECK(ta.x0 == tb.x0 && ta.y0 == tb.y0 && ta.x1 == tb.x1 && ta.y1 == tb.y1);
  }

  std::vector<std::uint32_t> sizes(static_cast<std::size_t>(a.layout.tile_count()),
                                   777);
  const auto p1 = av2enc::tiles::plan_single_tile_group(a.layout, sizes, 16, true);
  const auto p2 = av2enc::tiles::plan_single_tile_group(b.layout, sizes, 16, true);
  CHECK(p1.valid && p2.valid);
  CHECK(p1.obu_payload_bytes == p2.obu_payload_bytes);
  CHECK(p1.tile_payload_offsets == p2.tile_payload_offsets);
  CHECK(p1.tile_size_minus_1 == p2.tile_size_minus_1);
  CHECK(p1.header_bits == p2.header_bits);
}

}  // namespace

int main() {
  test_geometry();
  test_geometry_rejections();
  test_validate_layout();
  test_tile_group_plan();
  test_svc_sequence_and_dependencies();
  test_svc_lcr_and_operating_points();
  test_level_tile_limits();
  test_level_tile_count_limits();
  test_determinism();

  if (g_failures != 0) {
    std::cerr << "tiles_test: " << g_failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "tiles_test: PASS\n";
  return EXIT_SUCCESS;
}
