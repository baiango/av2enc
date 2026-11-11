// Tests for the encoder-side Annex A conformance validator.
//
// Level boundary coverage: 3.1 (idx 3), 4.1 (idx 5), 5.1 (idx 7), 6.1 (idx 11).
// Table integrity: A.1/A.2 profiles, A.5/A.6 multi-sequence configurations,
// A.7/A.8/A.9 levels, §9.2 tile scaling factors, A.3/A.4 interoperability.
// Plan audit: the documented 1080p/4K matrix plus DCI-4K.
//
// This file is self-contained except for tests/test_common.h. It needs only
// the three conformance translation units; see README.md for the CMake lines.

#include "../../../tests/test_common.h"

#include "encoder/conformance/conformance.h"
#include "encoder/conformance/known_plans.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace av2enc::conformance;

Config video_config(int width, int height, int level, int tier = 0) {
  Config config;
  config.width = width;
  config.height = height;
  config.sb_size = 64;
  config.seq_profile_idc = 0;
  config.seq_level_idx = level;
  config.seq_tier = tier;
  config.chroma_format_idc = 0;
  config.bit_depth_idc = 1;
  config.still_picture = false;
  config.single_picture_header_flag = false;
  config.explicit_num_ref_frames = true;
  config.num_ref_frames = 4;
  config.tiles = TileGrid::uniform(width, height, 1, 1, config.sb_size);
  return config;
}

Result run(const Config& config) { return validate(config, StreamStats{}); }

bool has(const Result& result, const char* check) {
  return result.has_check(check);
}

StreamStats rate_stats(int width, int height, double fps) {
  StreamStats stats;
  stats.fps = fps;
  stats.output_frames_in_tu = 1;
  stats.tu_display_luma_samples =
      static_cast<int64_t>(width) * static_cast<int64_t>(height);
  stats.tu_duration_seconds = fps > 0.0 ? 1.0 / fps : 0.0;
  return stats;
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

void test_profile_and_level_tables() {
  CHECK(profile_table_size() == 32);
  for (int idc = 0; idc <= 4; ++idc) {
    const ProfileInfo* profile = find_profile(idc);
    CHECK(profile != nullptr);
    CHECK(!profile->reserved);
    CHECK(profile->interop_point >= 0);
    CHECK(profile->pic_size_profile_factor > 0);
    CHECK(profile->bitrate_profile_factor >= 1.0);
  }
  CHECK(find_profile(5) != nullptr && find_profile(5)->reserved);
  CHECK(find_profile(30) != nullptr && find_profile(30)->reserved);
  CHECK(find_profile(31) != nullptr && find_profile(31)->configurable);
  CHECK(find_profile(32) == nullptr);

  int profile_factor = 0;
  int pic_factor = 0;
  double bitrate_factor = 0.0;
  CHECK(profile_scaling(0, &profile_factor, &pic_factor, &bitrate_factor));
  CHECK(profile_factor == 0);
  CHECK(pic_factor == 15);
  CHECK(std::abs(bitrate_factor - 1.0) < 1e-9);
  CHECK(profile_scaling(3, &profile_factor, &pic_factor, &bitrate_factor));
  CHECK(profile_factor == 1);
  CHECK(pic_factor == 20);
  CHECK(std::abs(bitrate_factor - 1.667) < 1e-9);
  CHECK(profile_scaling(4, &profile_factor, &pic_factor, &bitrate_factor));
  CHECK(profile_factor == 2);
  CHECK(pic_factor == 30);
  CHECK(std::abs(bitrate_factor - 2.5) < 1e-9);
  CHECK(!profile_scaling(31, &profile_factor, &pic_factor, &bitrate_factor));

  CHECK(profile_allows_chroma(*find_profile(0), 0));
  CHECK(profile_allows_chroma(*find_profile(0), 1));
  CHECK(!profile_allows_chroma(*find_profile(0), 2));
  CHECK(!profile_allows_chroma(*find_profile(0), 3));
  CHECK(profile_allows_chroma(*find_profile(3), 3));
  CHECK(!profile_allows_chroma(*find_profile(3), 2));
  CHECK(profile_allows_chroma(*find_profile(4), 2));
  CHECK(!profile_allows_chroma(*find_profile(4), 3));

  // Table A.8/A.9 boundaries used by the tests.
  CHECK(level_table_size() == 22);
  const LevelInfo* l31 = find_level(3);
  CHECK(l31 != nullptr && std::string(l31->name) == "3.1");
  CHECK(l31->max_pic_size == 1065024);
  CHECK(l31->max_hsize == 1720 && l31->max_vsize == 1720);
  CHECK(l31->max_display_rate == 31950720);
  CHECK(l31->max_decode_rate == 39938400);
  CHECK(l31->max_header_rate == 150);
  CHECK(std::abs(l31->main_mbps - 10.0) < 1e-9);
  CHECK(l31->max_tiles == 16 && l31->max_tile_cols == 6);

  const LevelInfo* l41 = find_level(5);
  CHECK(l41 != nullptr && std::string(l41->name) == "4.1");
  CHECK(l41->max_pic_size == 2359296);
  CHECK(l41->max_hsize == 2560);
  CHECK(l41->max_display_rate == 141557760);
  CHECK(l41->max_decode_rate == 155713536);
  CHECK(std::abs(l41->main_mbps - 20.0) < 1e-9);
  CHECK(std::abs(l41->high_mbps - 50.0) < 1e-9);
  CHECK(l41->max_tiles == 32 && l41->max_tile_cols == 8);

  const LevelInfo* l51 = find_level(7);
  CHECK(l51 != nullptr && std::string(l51->name) == "5.1");
  CHECK(l51->max_pic_size == 8912896);
  CHECK(l51->max_hsize == 4975 && l51->max_vsize == 4975);
  CHECK(l51->max_display_rate == 534773760);
  CHECK(l51->max_decode_rate == 547430400);
  CHECK(l51->max_header_rate == 300);
  CHECK(std::abs(l51->main_mbps - 40.0) < 1e-9);
  CHECK(std::abs(l51->high_mbps - 160.0) < 1e-9);
  CHECK(l51->main_cr == 8 && l51->high_cr == 4);
  CHECK(l51->max_tiles == 64 && l51->max_tile_cols == 8);

  const LevelInfo* l61 = find_level(11);
  CHECK(l61 != nullptr && std::string(l61->name) == "6.1");
  CHECK(l61->max_pic_size == 35651584);
  CHECK(l61->max_hsize == 9951);
  CHECK(l61->max_display_rate == 2139095040);
  CHECK(l61->max_decode_rate == 2189721600);
  CHECK(l61->max_header_rate == 300);
  CHECK(std::abs(l61->main_mbps - 100.0) < 1e-9);
  CHECK(std::abs(l61->high_mbps - 480.0) < 1e-9);
  CHECK(l61->max_tiles == 128 && l61->max_tile_cols == 16);

  CHECK(level_is_reserved(22));
  CHECK(level_is_reserved(30));
  CHECK(!level_is_reserved(21));
  CHECK(level_is_maximum(31));

  // §9.2 tile scaling factors, p622.
  TileScaleFactor factor = tile_scale_factor(3, 0);
  CHECK(!factor.reserved && factor.width == 4 && factor.area == 4);
  factor = tile_scale_factor(14, 0);
  CHECK(factor.width == 8 && factor.area == 8);
  factor = tile_scale_factor(18, 0);
  CHECK(factor.width == 8 && factor.area == 16);
  factor = tile_scale_factor(5, 1);
  CHECK(factor.width == 4 && factor.area == 4);
  factor = tile_scale_factor(14, 1);
  CHECK(factor.width == 8 && factor.area == 16);
  factor = tile_scale_factor(18, 1);
  CHECK(factor.width == 16 && factor.area == 32);
  CHECK(tile_scale_factor(22, 0).reserved);
  CHECK(tile_scale_factor(30, 1).reserved);
}

void test_multi_seq_and_interop_tables() {
  CHECK(multi_seq_table_size() == 4);
  CHECK(find_multi_seq_config(0) != nullptr);
  CHECK(std::string(find_multi_seq_config(0)->label) == "C_Main_420_10");
  CHECK(multi_seq_allows_profile(*find_multi_seq_config(0), 2));
  CHECK(multi_seq_allows_profile(*find_multi_seq_config(0), 31));
  CHECK(!multi_seq_allows_profile(*find_multi_seq_config(0), 3));
  CHECK(multi_seq_allows_profile(*find_multi_seq_config(1), 3));
  CHECK(multi_seq_allows_chroma(*find_multi_seq_config(1), 3));
  CHECK(!multi_seq_allows_chroma(*find_multi_seq_config(0), 3));
  CHECK(multi_seq_allows_profile(*find_multi_seq_config(2), 4));
  CHECK(multi_seq_allows_chroma(*find_multi_seq_config(2), 2));
  CHECK(!multi_seq_allows_chroma(*find_multi_seq_config(2), 3));
  CHECK(find_multi_seq_config(3) == nullptr);
  CHECK(multi_seq_is_reserved(3));
  CHECK(multi_seq_is_reserved(63));
  CHECK(!multi_seq_is_reserved(2));

  CHECK(interop_table_size() == 5);
  CHECK(find_interop_point(0) != nullptr && find_interop_point(0)->max_layers == 4);
  CHECK(find_interop_point(2)->max_layers == 8);
  CHECK(find_interop_point(2)->combination_allowed);
  CHECK(find_interop_point(3)->reserved);
  CHECK(interop_is_reserved(14));
  CHECK(!interop_is_reserved(15));
  CHECK(find_interop_point(15)->max_layers == 248);

  InteropObuRule rule = interop_obu_rule(0, false, false);
  CHECK(rule.msdo == ObuRequirement::kProhibited);
  rule = interop_obu_rule(0, true, false);
  CHECK(rule.msdo == ObuRequirement::kRequired);
  rule = interop_obu_rule(1, false, true);
  CHECK(rule.lcr == ObuRequirement::kRequired);
  CHECK(!rule.lcr_global_or_local);
  rule = interop_obu_rule(2, false, true);
  CHECK(rule.lcr == ObuRequirement::kRequired);
  CHECK(rule.lcr_global_or_local);
}

// ---------------------------------------------------------------------------
// Level boundaries
// ---------------------------------------------------------------------------

void test_geometry_levels() {
  struct Case {
    int level;
    int width;
    int height;
  };
  const Case exact[] = {
      {3, 1032, 1032},   // 1,065,024 = MaxPicSize at 3.1
      {5, 2048, 1152},   // 2,359,296 = MaxPicSize at 4.1
      {7, 4096, 2176},   // 8,912,896 = MaxPicSize at 5.1
      {11, 8192, 4352},  // 35,651,584 = MaxPicSize at 6.1
  };
  for (const Case& item : exact) {
    Config config = video_config(item.width, item.height, item.level);
    config.tiles =
        TileGrid::uniform(item.width, item.height, 2, 2, config.sb_size);
    const Result result = run(config);
    CHECK(!has(result, "A.4.MaxPicSize"));
    CHECK(!has(result, "A.4.MaxHSize"));
    CHECK(!has(result, "A.4.MaxVSize"));
    CHECK(result.conformant());
  }

  for (const Case& item : exact) {
    Config config = video_config(item.width, item.height + 1, item.level);
    config.tiles =
        TileGrid::uniform(item.width, item.height + 1, 2, 2, config.sb_size);
    const Result result = run(config);
    CHECK(has(result, "A.4.MaxPicSize"));
  }

  // MaxHSize / MaxVSize are separate limits from MaxPicSize.
  {
    Config config = video_config(1722, 16, 3);
    config.tiles = TileGrid::uniform(1722, 16, 2, 1, config.sb_size);
    CHECK(has(run(config), "A.4.MaxHSize"));
    CHECK(!has(run(config), "A.4.MaxPicSize"));
  }
  {
    Config config = video_config(4976, 16, 7);
    config.tiles = TileGrid::uniform(4976, 16, 2, 1, config.sb_size);
    CHECK(has(run(config), "A.4.MaxHSize"));
  }
  {
    Config config = video_config(9952, 16, 11);
    config.tiles = TileGrid::uniform(9952, 16, 2, 1, config.sb_size);
    CHECK(has(run(config), "A.4.MaxHSize"));
  }
  {
    Config config = video_config(16, 1722, 3);
    config.tiles = TileGrid::uniform(16, 1722, 1, 2, config.sb_size);
    CHECK(has(run(config), "A.4.MaxVSize"));
  }

  // FrameWidth >= 16 and FrameHeight >= 16 are level requirements (p1110).
  {
    Config config = video_config(16, 16, 7);
    CHECK(run(config).conformant());
    CHECK(!has(run(config), "A.4.MinFrameWidth"));
  }
  {
    Config config = video_config(14, 16, 7);
    CHECK(has(run(config), "A.4.MinFrameWidth"));
  }
  {
    Config config = video_config(16, 14, 7);
    CHECK(has(run(config), "A.4.MinFrameHeight"));
  }
}

void test_tile_level_limits() {
  // MaxTiles = 16 at level 3.1. 4x8 = 32 tiles with 4 columns (<= 6).
  {
    Config config = video_config(1280, 720, 3);
    config.tiles = TileGrid::uniform(1280, 720, 4, 8, config.sb_size);
    CHECK(config.tiles.tile_count() == 32);
    const Result result = run(config);
    CHECK(has(result, "A.4.MaxTiles"));
    CHECK(!has(result, "A.4.MaxTileCols"));
    CHECK(result.error_count() == 1);
  }
  // MaxTileCols = 6 at level 3.1; 8x2 = 16 tiles (== MaxTiles).
  {
    Config config = video_config(1280, 720, 3);
    config.tiles = TileGrid::uniform(1280, 720, 8, 2, config.sb_size);
    const Result result = run(config);
    CHECK(has(result, "A.4.MaxTileCols"));
    CHECK(!has(result, "A.4.MaxTiles"));
    CHECK(result.error_count() == 1);
  }
  // MaxTileCols = 8 at level 5.1; the documented 16x2 plan is the known
  // violation.
  {
    Config config = video_config(1920, 1080, 7);
    config.tiles = TileGrid::uniform(1920, 1080, 8, 2, config.sb_size);
    CHECK(run(config).conformant());
    CHECK(!has(run(config), "A.4.MaxTileCols"));
  }
  {
    Config config = video_config(1920, 1080, 7);
    config.tiles = TileGrid::uniform(1920, 1080, 16, 2, config.sb_size);
    const Result result = run(config);
    CHECK(has(result, "A.4.MaxTileCols"));
    CHECK(!has(result, "A.4.MaxTiles"));
    CHECK(result.error_count() == 1);
  }
  // MaxTileCols = 16 at level 6.1; 16x2 passes, 32x2 fails.
  {
    Config config = video_config(3840, 2160, 11);
    config.tiles = TileGrid::uniform(3840, 2160, 16, 2, config.sb_size);
    CHECK(run(config).conformant());
    config.tiles = TileGrid::uniform(3840, 2160, 32, 2, config.sb_size);
    const Result result = run(config);
    CHECK(has(result, "A.4.MaxTileCols"));
    CHECK(!has(result, "A.4.MaxTiles"));
    CHECK(result.error_count() == 1);
  }
  // TileWidth >= 64 only applies when RightMostTile == 0. A non-uniform grid
  // with a 32-sample non-rightmost column must be rejected.
  {
    Config config = video_config(256, 256, 7);
    config.tiles.cols = 2;
    config.tiles.rows = 1;
    config.tiles.tile_widths = {32, 224};
    config.tiles.tile_heights = {256, 256};
    const Result result = run(config);
    CHECK(has(result, "A.4.MinTileWidth"));
    CHECK(result.error_count() == 1);
  }
  // The rightmost column may be narrower than 64 (it is clipped at MiCols).
  {
    Config config = video_config(200, 64, 7);
    config.tiles.cols = 2;
    config.tiles.rows = 1;
    config.tiles.tile_widths = {136, 64};
    config.tiles.tile_heights = {64, 64};
    CHECK(!has(run(config), "A.4.MinTileWidth"));
  }
  // TileWidth <= WSF * MAX_TILE_WIDTH / 4. At level 6.1 the tier-0 factor is
  // 8, so a 9000-sample tile width exceeds 8192.
  {
    Config config = video_config(9000, 16, 11);
    CHECK(has(run(config), "A.4.MaxTileWidth"));
  }
  // TileWidth * TileHeight <= ASF * MAX_TILE_AREA / 4. At level 6.1 a single
  // 8192x4352 tile is 35,651,584 samples, over the 9,437,184 limit.
  {
    Config config = video_config(8192, 4352, 11);
    const Result result = run(config);
    CHECK(has(result, "A.4.MaxTileArea"));
    CHECK(!has(result, "A.4.MaxPicSize"));
  }
}

void test_uniform_tile_grid() {
  const TileGrid grid = TileGrid::uniform(1920, 1080, 16, 2, 64);
  CHECK(grid.representable);
  CHECK(grid.cols == 16 && grid.rows == 2);
  CHECK(grid.tile_widths.size() == 32);
  CHECK(grid.tile_heights.size() == 32);
  int min_width = 1000000;
  int max_width = 0;
  for (int width : grid.tile_widths) {
    min_width = std::min(min_width, width);
    max_width = std::max(max_width, width);
  }
  CHECK(min_width == 64);
  CHECK(max_width == 128);
  CHECK(grid.tile_heights[0] == 512);
  CHECK(grid.tile_heights[31] == 568);

  // 16 columns over 640x360 is not representable by uniform spacing (10 SB
  // columns), which is why the planner rejects it rather than emitting 40 px
  // tiles.
  const TileGrid narrow = TileGrid::uniform(640, 360, 16, 2, 64);
  CHECK(!narrow.representable);
  CHECK(narrow.tile_widths.empty());

  // cols/rows without explicit extents mean "uniform": the validator derives
  // the geometry, so a plan-only config cannot bypass the tile checks.
  {
    Config config = video_config(1920, 1080, 7);
    config.tiles = TileGrid{};
    config.tiles.cols = 16;
    config.tiles.rows = 2;
    CHECK(has(run(config), "A.4.MaxTileCols"));
  }

  // The uniform syntax is 1 << tileColsLog2; a non-power-of-two explicit grid
  // is not representable.
  {
    Config config = video_config(1920, 1080, 7);
    config.tiles = TileGrid{};
    config.tiles.cols = 3;
    config.tiles.rows = 1;
    config.tiles.tile_widths = {640, 640, 640};
    config.tiles.tile_heights = {1080, 1080, 1080};
    CHECK(has(run(config), "A.4.TileGridInvalid"));
  }
}

void test_rates_and_refframes() {
  // TotalDisplayLumaSampleRate (p1110).
  {
    Config config = video_config(3840, 2160, 7);
    CHECK(run(config).conformant());
    CHECK(validate(config, rate_stats(3840, 2160, 60.0)).conformant());
    CHECK(has(validate(config, rate_stats(3840, 2160, 120.0)),
              "A.4.MaxDisplayRate"));
  }
  {
    Config config = video_config(1920, 1080, 5);
    CHECK(validate(config, rate_stats(1920, 1080, 60.0)).conformant());
    CHECK(has(validate(config, rate_stats(1920, 1080, 120.0)),
              "A.4.MaxDisplayRate"));
  }
  {
    Config config = video_config(1280, 720, 3);
    CHECK(validate(config, rate_stats(1280, 720, 30.0)).conformant());
    CHECK(has(validate(config, rate_stats(1280, 720, 60.0)),
              "A.4.MaxDisplayRate"));
  }
  {
    Config config = video_config(7680, 4320, 11);
    config.tiles = TileGrid::uniform(7680, 4320, 4, 2, config.sb_size);
    CHECK(validate(config, rate_stats(7680, 4320, 30.0)).conformant());
    CHECK(validate(config, rate_stats(7680, 4320, 60.0)).conformant());
    CHECK(has(validate(config, rate_stats(7680, 4320, 120.0)),
              "A.4.MaxDisplayRate"));
  }

  // NumFrameHeadersPerSec = MaxHeaderRate * (1 + (seq_tier << 1)).
  {
    Config config = video_config(1920, 1080, 7);
    config.tiles = TileGrid::uniform(1920, 1080, 4, 2, config.sb_size);
    StreamStats stats;
    stats.max_frame_headers_per_second = 300;
    CHECK(validate(config, stats).conformant());
    stats.max_frame_headers_per_second = 301;
    CHECK(has(validate(config, stats), "A.4.NumFrameHeadersPerSec"));

    Config high = video_config(1920, 1080, 7, 1);
    high.tiles = TileGrid::uniform(1920, 1080, 4, 2, config.sb_size);
    stats.max_frame_headers_per_second = 900;
    CHECK(validate(high, stats).conformant());
    stats.max_frame_headers_per_second = 901;
    CHECK(has(validate(high, stats), "A.4.NumFrameHeadersPerSec"));
  }

  // MaxLevelRefFrames (p1109). 4K at 5.1 grants 8; 1080p is capped at 16.
  {
    Config config = video_config(3840, 2160, 7);
    CHECK(resolve_level(config).max_level_ref_frames == 8);  // p1109, no -1
    CHECK(max_level_ref_frames(config) == 8);
    CHECK(max_level_ref_frames(config, true) == 7);
    config.num_ref_frames = 8;
    CHECK(run(config).conformant());
    config.num_ref_frames = 9;
    CHECK(has(run(config), "A.4.MaxLevelRefFrames"));
    config.num_ref_frames = 16;
    CHECK(has(run(config), "A.4.MaxLevelRefFrames"));

    Config explicit_off = video_config(3840, 2160, 7);
    explicit_off.explicit_num_ref_frames = false;
    explicit_off.num_ref_frames = 16;
    CHECK(max_level_ref_frames(explicit_off) == 8);
    // With explicit_num_ref_frames == 0 the decoder infers 8, so a config
    // field of 16 is not emitted; a measured 16 still exceeds the 8 cap.
    CHECK(!has(run(explicit_off), "A.4.MaxLevelRefFrames"));
    StreamStats off_stats;
    off_stats.num_ref_frames = 16;
    CHECK(has(validate(explicit_off, off_stats), "A.4.MaxLevelRefFrames"));

    Config hd = video_config(1920, 1080, 7);
    CHECK(max_level_ref_frames(hd) == 16);
    hd.num_ref_frames = 16;
    CHECK(run(hd).conformant());
    hd.num_ref_frames = 17;
    CHECK(has(run(hd), "A.4.MaxLevelRefFrames"));
  }
}

void test_frame_budget() {
  // LumaSampleCount <= FrameParsingTime * MaxDecodeRate (p1110).
  {
    Config config = video_config(1280, 720, 3);
    StreamStats stats;
    stats.frames.resize(1);
    stats.frames[0].luma_sample_count = 1280 * 720;
    stats.frames[0].frame_parsing_time_seconds = 1.0 / 120.0;
    const Result result = validate(config, stats);
    CHECK(has(result, "A.4.LumaSampleCount"));
    CHECK(max_luma_samples_for_parsing_time(config, 1.0 / 120.0) == 332820);
  }

  // NumTiles <= Min(MaxTiles, Max(1, MaxTiles * 120 * FrameParsingTime)).
  {
    Config config = video_config(3840, 2160, 7);
    config.tiles = TileGrid::uniform(3840, 2160, 8, 8, config.sb_size);
    CHECK(config.tiles.tile_count() == 64);
    CHECK(max_tiles_for_parsing_time(config, 1.0 / 60.0) == 64);
    CHECK(max_tiles_for_parsing_time(config, 1.0 / 240.0) == 32);

    StreamStats stats;
    stats.fps = 240.0;
    stats.frames.resize(1);
    stats.frames[0].luma_sample_count = 3840 * 2160;
    const Result result = validate(config, stats);
    CHECK(has(result, "A.4.NumTilesParsingTime"));
  }

  // CompressedSize with the profile-0 (PicSizeProfileFactor = 15) budget.
  {
    Config config = video_config(3840, 2160, 7);
    config.tiles = TileGrid::uniform(3840, 2160, 4, 2, config.sb_size);
    const double allowed =
        max_compressed_size_bytes(config, 3840LL * 2160, 1.0 / 60.0);
    CHECK(std::abs(allowed - 2138400.0) < 0.5);

    StreamStats stats;
    stats.frames.resize(1);
    stats.frames[0].luma_sample_count = 3840LL * 2160;
    stats.frames[0].frame_parsing_time_seconds = 1.0 / 60.0;
    stats.frames[0].frame_obu_bytes =
        static_cast<int64_t>(allowed) + 128;  // exactly at the limit after -128
    CHECK(!has(validate(config, stats), "A.4.CompressedSize"));
    stats.frames[0].frame_obu_bytes = static_cast<int64_t>(allowed) + 129;
    CHECK(has(validate(config, stats), "A.4.CompressedSize"));
  }

  // FrameSymbolCount budget.
  {
    Config config = video_config(3840, 2160, 7);
    config.tiles = TileGrid::uniform(3840, 2160, 4, 2, config.sb_size);
    const double allowed = max_frame_symbol_count(config, 1.0 / 60.0);
    CHECK(std::abs(allowed - 18057600.0) < 1.0);

    StreamStats stats;
    stats.frames.resize(1);
    stats.frames[0].luma_sample_count = 3840LL * 2160;
    stats.frames[0].frame_parsing_time_seconds = 1.0 / 60.0;
    stats.frames[0].symbol_count = static_cast<int64_t>(allowed);
    CHECK(!has(validate(config, stats), "A.4.FrameSymbolCount"));
    stats.frames[0].symbol_count = static_cast<int64_t>(allowed) + 1;
    CHECK(has(validate(config, stats), "A.4.FrameSymbolCount"));
  }

  // MaxTileSizeInLumaSamples * NumFrameHeadersPerSec (p1110). At 300 Hz and
  // level 5.1 the allowed tile area is 1,824,768; 2x2 at 4K is 2,073,600.
  {
    Config config = video_config(3840, 2160, 7);
    config.tiles = TileGrid::uniform(3840, 2160, 2, 2, config.sb_size);
    StreamStats stats;
    stats.max_frame_headers_per_second = 300;
    CHECK(max_tile_size_luma_samples(config, 300) == 1824768);
    CHECK(has(validate(config, stats), "A.4.MaxTileSizeHeaderRate"));

    config.tiles = TileGrid::uniform(3840, 2160, 4, 2, config.sb_size);
    CHECK(!has(validate(config, stats), "A.4.MaxTileSizeHeaderRate"));
  }

  // MaxBitrate / MaxBufferSize (p1110-1111, p1125).
  {
    Config config = video_config(3840, 2160, 7);
    StreamStats stats;
    stats.max_bitrate_bps_over_1s = 40.0e6;
    CHECK(validate(config, stats).conformant());
    stats.max_bitrate_bps_over_1s = 40.0e6 + 1.0;
    CHECK(has(validate(config, stats), "A.4.MaxBitrate"));

    stats.max_bitrate_bps_over_1s = 0.0;
    stats.peak_buffer_occupancy_bits = 40.0e6;
    CHECK(validate(config, stats).conformant());
    stats.peak_buffer_occupancy_bits = 40.0e6 + 1.0;
    CHECK(has(validate(config, stats), "A.4.MaxBufferSize"));

    // Profile 3 multiplies the level ceiling by BitrateProfileFactor = 1.667,
    // so 50 Mbit/s passes at 5.1 even though the profile-0 limit is 40.
    Config profile3 = config;
    profile3.seq_profile_idc = 3;
    profile3.chroma_format_idc = 3;
    profile3.bit_depth_idc = 0;
    StreamStats p3stats;
    p3stats.max_bitrate_bps_over_1s = 50.0e6;
    CHECK(validate(profile3, p3stats).conformant());
    p3stats.max_bitrate_bps_over_1s = 67.0e6;
    CHECK(has(validate(profile3, p3stats), "A.4.MaxBitrate"));
  }
}

void test_tier_and_level_legality() {
  {
    Config config = video_config(1280, 720, 3, 1);
    CHECK(has(run(config), "A.4.SeqTierLevel"));
  }
  {
    Config config = video_config(1920, 1080, 4, 1);
    CHECK(!has(run(config), "A.4.SeqTierLevel"));
  }
  {
    Config config = video_config(1920, 1080, 5, 1);
    CHECK(!has(run(config), "A.4.SeqTierLevel"));
  }
  {
    // single_picture_header_flag = 1 removes seq_tier from the syntax.
    Config config = video_config(1920, 1080, 7, 1);
    config.still_picture = true;
    config.single_picture_header_flag = true;
    CHECK(has(run(config), "A.4.SeqTierNotSignalled"));
  }
  {
    Config config = video_config(1280, 720, 22);
    CHECK(has(run(config), "A.4.ReservedLevel"));
  }
  {
    // Level 31 means no level-based constraints (p1111).
    Config config = video_config(8192, 8192, 31);
    config.tiles = TileGrid::uniform(8192, 8192, 1, 1, config.sb_size);
    const Result result = run(config);
    CHECK(result.conformant());
    CHECK(!has(result, "A.4.MaxPicSize"));
    CHECK(!has(result, "A.4.MaxTiles"));
  }
}

void test_profiles_and_interop() {
  {
    Config config = video_config(640, 360, 7);
    config.chroma_format_idc = 3;  // 4:2:2 under profile 0
    CHECK(has(run(config), "A.2.ProfileChroma"));
  }
  {
    Config config = video_config(640, 360, 7);
    config.seq_profile_idc = 3;
    config.chroma_format_idc = 3;
    config.bit_depth_idc = 0;
    CHECK(!has(run(config), "A.2.ProfileChroma"));
  }
  {
    Config config = video_config(640, 360, 7);
    config.seq_profile_idc = 4;
    config.chroma_format_idc = 2;  // 4:4:4
    CHECK(!has(run(config), "A.2.ProfileChroma"));
  }
  {
    Config config = video_config(640, 360, 7);
    config.seq_profile_idc = 2;
    config.chroma_format_idc = 2;  // profile 2 does not allow 4:4:4
    CHECK(has(run(config), "A.2.ProfileChroma"));
  }
  {
    Config config = video_config(640, 360, 7);
    config.seq_profile_idc = 5;
    CHECK(has(run(config), "A.2.ReservedProfile"));
  }
  {
    Config config = video_config(640, 360, 7);
    config.bit_depth_idc = 2;
    CHECK(has(run(config), "A.2.BitDepthIdc"));
  }
  {
    Config config = video_config(640, 360, 7);
    config.chroma_format_idc = 4;
    CHECK(has(run(config), "A.2.ChromaFormat"));
  }
  {
    // Configurable profile: no Table A.2 factors, constraints not evaluable.
    Config config = video_config(640, 360, 7);
    config.seq_profile_idc = 31;
    const Result result = run(config);
    CHECK(has(result, "A.2.ConfigurableProfile"));
    CHECK(result.conformant());  // warning only; no A.4 error at level 5.1
    CHECK(result.error_count() == 0);
  }

  // Table A.4 MSDO/LCR rules.
  {
    Config config = video_config(1920, 1080, 7);
    config.interop_point = 0;
    config.num_extended_layers = 2;
    CHECK(has(run(config), "A.3.Iop0MsdoRequired"));
    config.msdo_present = true;
    CHECK(run(config).conformant());
  }
  {
    Config config = video_config(1920, 1080, 7);
    config.seq_profile_idc = 1;  // Main_420_10_IP1
    config.interop_point = 1;
    config.num_extended_layers = 1;
    config.num_embedded_layers = 2;
    CHECK(has(run(config), "A.3.Iop1LcrRequired"));
    config.local_lcr_present = true;
    CHECK(run(config).conformant());
    // Table A.4 says "Required (Local)" for IOP 1, so a global LCR alone is
    // not sufficient.
    config.local_lcr_present = false;
    config.global_lcr_present = true;
    CHECK(has(run(config), "A.3.Iop1LcrRequired"));
  }
  {
    Config config = video_config(1920, 1080, 7);
    config.seq_profile_idc = 2;  // Main_420_10_IP2
    config.interop_point = 2;
    config.num_extended_layers = 2;
    config.num_embedded_layers = 2;
    config.msdo_present = true;
    CHECK(has(run(config), "A.3.Iop2Combination"));
    config.local_lcr_present = true;
    CHECK(run(config).conformant());
    config.local_lcr_present = false;
    config.global_lcr_present = true;
    CHECK(run(config).conformant());
  }
  {
    // Table A.6: C_Main_420_10 does not allow profile 3.
    Config config = video_config(640, 360, 7);
    config.seq_profile_idc = 3;
    config.chroma_format_idc = 3;
    config.bit_depth_idc = 0;
    config.lcr_config_idc = 0;
    config.local_lcr_present = true;
    CHECK(has(run(config), "A.3.MultiSeqProfile"));

    config.lcr_config_idc = 1;  // C_Main_422_10
    CHECK(!has(run(config), "A.3.MultiSeqProfile"));
  }
  {
    Config config = video_config(640, 360, 7);
    config.lcr_config_idc = 3;  // Reserved
    CHECK(has(run(config), "A.3.MultiSeqConfigReserved"));
  }
}

void test_verifiability_and_coverage() {
  {
    Config config = video_config(1920, 1080, 7);
    const Result result = run(config);
    CHECK(has(result, "E.4.4.LevelConformanceUnverifiable"));
    CHECK(result.conformant());  // warning, not an error
    CHECK(result.warning_count() == 1);
  }
  {
    Config config = video_config(1920, 1080, 7);
    config.ci_timing_info_present = true;
    config.decoder_model_info_present = true;
    StreamStats stats;
    stats.timing_info_signalled = true;
    stats.decoder_model_info_signalled = true;
    CHECK(!has(validate(config, stats), "E.4.4.LevelConformanceUnverifiable"));
  }

  const std::vector<CoverageEntry>& coverage = absent_constraint_coverage();
  CHECK(coverage.size() == 33);
  for (std::size_t i = 0; i < coverage.size(); ++i) {
    CHECK(coverage[i].audit_row == static_cast<int>(i) + 1);
    CHECK(std::string(coverage[i].constraint) != "");
    CHECK(std::string(coverage[i].spec_ref) != "");
    CHECK(std::string(coverage[i].disposition) != "");
  }
  int implemented = 0;
  int not_applicable = 0;
  int timing = 0;
  for (const CoverageEntry& entry : coverage) {
    const std::string disposition(entry.disposition);
    if (disposition.rfind("check:", 0) == 0 ||
        disposition.rfind("table:", 0) == 0) {
      ++implemented;
    } else if (disposition.rfind("N/A:", 0) == 0) {
      ++not_applicable;
    } else if (disposition.rfind("timing:", 0) == 0) {
      ++timing;
    }
  }
  CHECK(implemented == 22);
  CHECK(not_applicable == 7);
  CHECK(timing == 4);
}

// ---------------------------------------------------------------------------
// Documented plans
// ---------------------------------------------------------------------------

void test_documented_plans() {
  // 30 fps isolates the two geometry violations: at 60 fps DCI-4K would also
  // exceed MaxDisplayRate, which is a separate (true) finding.
  const std::vector<PlanAudit> audits = audit_documented_plans(30.0);
  CHECK(audits.size() == 8);

  int sixteen_violations = 0;
  int dci_violations = 0;
  for (const PlanAudit& audit : audits) {
    const std::string name(audit.plan.name);
    const bool is_16x2 = name.find("16x2") != std::string::npos;
    const bool is_dci = name.find("dci4k") != std::string::npos;
    const bool is_level6 = audit.plan.seq_level_idx == 10;

    if (is_16x2 && !is_level6) {
      ++sixteen_violations;
      CHECK(has(audit.result, "A.4.MaxTileCols"));
      CHECK(!has(audit.result, "A.4.MaxTiles"));
      CHECK(audit.result.error_count() == 1);
      CHECK(!audit.result.conformant());
      bool found = false;
      for (const Violation& violation : audit.result.violations) {
        if (violation.check == "A.4.MaxTileCols") {
          CHECK(violation.detail.find("TileCols = 16 > MaxTileCols 8") !=
                std::string::npos);
          CHECK(violation.section == "A.4");
          CHECK(violation.page == 1110);
          found = true;
        }
      }
      CHECK(found);
    } else if (is_dci) {
      ++dci_violations;
      CHECK(has(audit.result, "A.4.MaxPicSize"));
      CHECK(audit.result.error_count() == 1);
      CHECK(!audit.result.conformant());
      bool found = false;
      for (const Violation& violation : audit.result.violations) {
        if (violation.check == "A.4.MaxPicSize") {
          CHECK(violation.detail.find("9437184") != std::string::npos);
          CHECK(violation.detail.find("8912896") != std::string::npos);
          found = true;
        }
      }
      CHECK(found);
    } else {
      CHECK(audit.result.conformant());
      CHECK(audit.result.error_count() == 0);
    }
    // Every video plan without timing gets the E.4.4 warning.
    CHECK(has(audit.result, "E.4.4.LevelConformanceUnverifiable"));
  }
  CHECK(sixteen_violations == 2);  // 1080p/4 GPU and 4K/4 GPU
  CHECK(dci_violations == 1);

  const std::string matrix = render_plan_matrix();
  CHECK(matrix.find("1080p-4gpu-16x2") != std::string::npos);
  CHECK(matrix.find("FAIL") != std::string::npos);
  CHECK(matrix.find("A.4.MaxTileCols") != std::string::npos);
  CHECK(matrix.find("A.4.MaxPicSize") != std::string::npos);
  CHECK(matrix.find("PASS") != std::string::npos);
}

}  // namespace

int main() {
  test_profile_and_level_tables();
  test_multi_seq_and_interop_tables();
  test_geometry_levels();
  test_tile_level_limits();
  test_uniform_tile_grid();
  test_rates_and_refframes();
  test_frame_budget();
  test_tier_and_level_legality();
  test_profiles_and_interop();
  test_verifiability_and_coverage();
  test_documented_plans();
  return test::finish("conformance_test");
}
