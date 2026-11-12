// Deterministic tests for the combined rate-control / tile / multi-GPU
// pipeline planner. No CUDA, no decoder, no bitstream. Every number here is a
// planner decision or an arithmetic model result, not a measurement.

#include "pipeline/planner/pipeline.h"

#include "cuda/pipeline/planner/pipeline_gpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace av2enc;
using namespace av2enc::pipeline;
using av2enc::tiles::TileLayoutPreset;

int g_failures = 0;
int g_checks = 0;

void check_impl(bool condition, const char* expression, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::cerr << "line " << line << ": CHECK failed: " << expression << '\n';
  }
}

#define CHECK(expression) check_impl((expression), #expression, __LINE__)

bool near(double a, double b, double epsilon = 1e-9) {
  return std::fabs(a - b) <= epsilon;
}

constexpr std::uint64_t kMiB = 1024ull * 1024ull;

rc::FrameComplexity complexity(int width, int height, double mad) {
  rc::FrameComplexity value;
  value.width = width;
  value.height = height;
  value.residual_mad = mad;
  return value;
}

PipelineConfig base_config(int width, int height, int gpus,
                           TileLayoutPreset preset) {
  PipelineConfig config;
  config.width = width;
  config.height = height;
  config.gpu.gpu_count = gpus;
  config.gpu.reference_frames = 4;
  config.gpu.lookahead_frames = 4;
  config.layout.mode = LayoutPolicyMode::kExplicit;
  config.layout.explicit_preset = preset;
  config.device.total_bytes = 8192ull * kMiB;
  config.device.free_bytes = 7800ull * kMiB;
  config.rate_control.mode = rc::RateControlMode::kCqp;
  config.rate_control.cqp.base_q_idx = 90;
  return config;
}

FrameStats stats_for(int tile_count, double mad) {
  FrameStats stats;
  stats.frame_index = 0;
  stats.complexity = complexity(1920, 1080, mad);
  for (int i = 0; i < tile_count; ++i) {
    TileStats tile;
    tile.tile_index = -1;
    tile.complexity = complexity(64, 64, mad);
    stats.tiles.push_back(tile);
  }
  return stats;
}

bool configure_and_plan(const PipelineConfig& config, const FrameStats& stats,
                        FramePipelinePlan& plan, std::string* error) {
  PipelinePlanner planner;
  if (!planner.configure(config, error)) {
    return false;
  }
  if (!planner.plan_frame(stats, plan, error)) {
    return false;
  }
  return planner.end_frame(
      static_cast<std::int64_t>(std::llround(plan.estimated_frame_bits)),
      error);
}

bool tile_geometry_equal(const TilePlan& a, const TilePlan& b) {
  return a.index == b.index && a.row == b.row && a.col == b.col &&
         a.gpu_index == b.gpu_index && a.rect.x0 == b.rect.x0 &&
         a.rect.y0 == b.rect.y0 && a.rect.x1 == b.rect.x1 &&
         a.rect.y1 == b.rect.y1 && a.target_qindex == b.target_qindex &&
         a.qindex == b.qindex && a.segment_id == b.segment_id &&
         a.alt_q == b.alt_q && a.delta_q_reduced == b.delta_q_reduced;
}

bool frame_plan_equal(const FramePipelinePlan& a,
                      const FramePipelinePlan& b) {
  if (a.valid != b.valid || a.frame_index != b.frame_index ||
      a.base_q_idx != b.base_q_idx || a.layout.tile_count() !=
                                            b.layout.tile_count() ||
      a.tiles.size() != b.tiles.size() ||
      a.has_segment_map != b.has_segment_map ||
      a.tile_qp_mode != b.tile_qp_mode ||
      a.delta_q.present != b.delta_q.present ||
      a.delta_q.res != b.delta_q.res ||
      a.segmentation.enabled != b.segmentation.enabled ||
      a.gpu.gpus.size() != b.gpu.gpus.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.tiles.size(); ++i) {
    if (!tile_geometry_equal(a.tiles[i], b.tiles[i])) {
      return false;
    }
    if (a.tiles[i].delta_plan.steps.size() !=
        b.tiles[i].delta_plan.steps.size()) {
      return false;
    }
    for (std::size_t s = 0; s < a.tiles[i].delta_plan.steps.size(); ++s) {
      const auto& sa = a.tiles[i].delta_plan.steps[s];
      const auto& sb = b.tiles[i].delta_plan.steps[s];
      if (sa.sb_x != sb.sb_x || sa.sb_y != sb.sb_y ||
          sa.reduced_delta != sb.reduced_delta ||
          sa.qindex_before != sb.qindex_before ||
          sa.qindex_after != sb.qindex_after) {
        return false;
      }
    }
  }
  if (a.has_segment_map) {
    if (a.segment_map.mi_cols != b.segment_map.mi_cols ||
        a.segment_map.mi_rows != b.segment_map.mi_rows ||
        a.segment_map.segment_id != b.segment_map.segment_id) {
      return false;
    }
  }
  for (std::size_t g = 0; g < a.gpu.gpus.size(); ++g) {
    if (a.gpu.gpus[g].tile_indices != b.gpu.gpus[g].tile_indices ||
        !near(a.gpu.gpus[g].vram.live_mb, b.gpu.gpus[g].vram.live_mb) ||
        a.gpu.gpus[g].vram.fits != b.gpu.gpus[g].vram.fits) {
      return false;
    }
  }
  return true;
}

void test_quantizer_helpers() {
  TileQpConfig config;
  config.complexity_strength = 1.0;
  config.max_abs_delta = 24;

  CHECK(tile_qindex_delta(0.0, 10.0, config) == 0);
  CHECK(tile_qindex_delta(-1.0, 10.0, config) == 0);
  CHECK(tile_qindex_delta(10.0, 0.0, config) == 0);
  CHECK(tile_qindex_delta(10.0, 10.0, config) == 0);
  CHECK(tile_qindex_delta(20.0, 10.0, config) == 24);
  CHECK(tile_qindex_delta(5.0, 10.0, config) == -24);
  config.max_abs_delta = 6;
  CHECK(tile_qindex_delta(20.0, 10.0, config) == 6);

  config.complexity_strength = 0.0;
  CHECK(tile_qindex_delta(40.0, 10.0, config) == 0);

  TileQpConfig bad;
  bad.delta_q_res = 4;
  std::string error;
  CHECK(!tile_qp_config_valid(bad, &error));
  bad = TileQpConfig{};
  bad.mode = TileQpMode::kUniform;
  bad.delta_q_res = 1;
  CHECK(!tile_qp_config_valid(bad, &error));
}

void test_layout_policy() {
  LayoutPolicy policy;
  CHECK(layout_preset_allowed(TileLayoutPreset::k1x1, policy));
  CHECK(layout_preset_allowed(TileLayoutPreset::k8x2, policy));
  CHECK(!layout_preset_allowed(TileLayoutPreset::k16x2, policy));
  policy.allow_unverified = true;
  CHECK(layout_preset_allowed(TileLayoutPreset::k16x2, policy));
  policy.allow_unverified = false;
  policy.decoder_verified[5] = true;
  CHECK(layout_preset_allowed(TileLayoutPreset::k16x2, policy));
  CHECK(std::string(layout_preset_name(TileLayoutPreset::k4x2)) == "4x2");

  CHECK(default_layout_for_gpus(1, LayoutPolicy{}) ==
        TileLayoutPreset::k4x2);
  CHECK(default_layout_for_gpus(2, LayoutPolicy{}) ==
        TileLayoutPreset::k8x2);
  CHECK(default_layout_for_gpus(4, LayoutPolicy{}) ==
        TileLayoutPreset::k8x2);
  LayoutPolicy verified;
  verified.decoder_verified[5] = true;
  CHECK(default_layout_for_gpus(4, verified) ==
        TileLayoutPreset::k16x2);
}

void test_tile_geometry() {
  struct Case {
    int width;
    int height;
    TileLayoutPreset preset;
    int expected_tiles;
    int expected_cols;
    int expected_rows;
  };
  const std::array<Case, 5> cases{{
      {1920, 1080, TileLayoutPreset::k1x1, 1, 1, 1},
      {1920, 1080, TileLayoutPreset::k2x1, 2, 2, 1},
      {1920, 1080, TileLayoutPreset::k2x2, 4, 2, 2},
      {1920, 1080, TileLayoutPreset::k4x2, 8, 4, 2},
      {1920, 1080, TileLayoutPreset::k8x2, 16, 8, 2},
  }};

  for (const Case& item : cases) {
    PipelineConfig config =
        base_config(item.width, item.height, 1, item.preset);
    FrameStats stats = stats_for(item.expected_tiles, 16.0);
    FramePipelinePlan plan;
    std::string error;
    CHECK(configure_and_plan(config, stats, plan, &error));
    CHECK(plan.valid);
    CHECK(plan.layout.tile_count() == item.expected_tiles);
    CHECK(plan.layout.cols == item.expected_cols);
    CHECK(plan.layout.rows == item.expected_rows);
    CHECK(static_cast<int>(plan.tiles.size()) == item.expected_tiles);

    std::uint64_t pixels = 0;
    std::uint64_t sb_cells = 0;
    for (const TilePlan& tile : plan.tiles) {
      CHECK(tile.rect.pixel_width() > 0);
      CHECK(tile.rect.pixel_height() > 0);
      pixels += tile.rect.pixels();
      sb_cells += tile.rect.sb_area();
    }
    CHECK(pixels ==
          static_cast<std::uint64_t>(item.width) *
              static_cast<std::uint64_t>(item.height));
    CHECK(sb_cells ==
          static_cast<std::uint64_t>(plan.layout.sb_cols) *
              static_cast<std::uint64_t>(plan.layout.sb_rows));
  }

  // 256x256 has 4x4 superblocks: 1x1, 2x1, 2x2 and 4x2 are representable,
  // 8x2 is not (the uniform grid would need 8 columns).
  for (TileLayoutPreset preset :
       {TileLayoutPreset::k1x1, TileLayoutPreset::k2x1, TileLayoutPreset::k2x2,
        TileLayoutPreset::k4x2}) {
    PipelineConfig config = base_config(256, 256, 1, preset);
    const auto dims = tiles::preset_dims(preset);
    FrameStats stats = stats_for(dims.cols * dims.rows, 16.0);
    FramePipelinePlan plan;
    std::string error;
    CHECK(configure_and_plan(config, stats, plan, &error));
    CHECK(plan.layout.tile_count() == dims.cols * dims.rows);
  }
  {
    PipelineConfig config =
        base_config(256, 256, 1, TileLayoutPreset::k8x2);
    PipelinePlanner planner;
    std::string error;
    CHECK(!planner.configure(config, &error));
    CHECK(!error.empty());
  }

  // 16x2 is decoder-unverified by default and must be refused explicitly.
  {
    PipelineConfig config =
        base_config(1920, 1080, 4, TileLayoutPreset::k16x2);
    PipelinePlanner planner;
    std::string error;
    CHECK(!planner.configure(config, &error));
    CHECK(error.find("not decoder-verified") != std::string::npos);

    config.layout.allow_unverified = true;
    FrameStats stats = stats_for(32, 16.0);
    FramePipelinePlan plan;
    CHECK(configure_and_plan(config, stats, plan, &error));
    CHECK(plan.layout.tile_count() == 32);
    CHECK(plan.gpu.gpus.size() == 4);
  }

  // Auto layout follows the documented per-GPU defaults but keeps 16x2 off
  // until the gate table says otherwise.
  {
    PipelineConfig config = base_config(1920, 1080, 4, TileLayoutPreset::k4x2);
    config.layout.mode = LayoutPolicyMode::kAuto;
    FrameStats stats = stats_for(16, 16.0);
    FramePipelinePlan plan;
    std::string error;
    CHECK(configure_and_plan(config, stats, plan, &error));
    CHECK(plan.layout.tile_count() == 16);
    CHECK(plan.layout_reason.find("8x2") != std::string::npos);

    config.layout.decoder_verified[5] = true;
    FramePipelinePlan verified_plan;
    CHECK(configure_and_plan(config, stats_for(32, 16.0), verified_plan,
                             &error));
    CHECK(verified_plan.layout.tile_count() == 32);
    CHECK(verified_plan.layout_reason.find("16x2") != std::string::npos);
  }

  // Auto falls back to a representable layout for a small frame with 2 GPUs.
  {
    PipelineConfig config = base_config(256, 256, 2, TileLayoutPreset::k8x2);
    config.layout.mode = LayoutPolicyMode::kAuto;
    FrameStats stats = stats_for(8, 16.0);
    FramePipelinePlan plan;
    std::string error;
    CHECK(configure_and_plan(config, stats, plan, &error));
    CHECK(plan.layout.tile_count() >= 2);
    CHECK(plan.layout.tile_count() == 8);  // 4x2
  }
}

void test_delta_q_tile_qp() {
  PipelineConfig config =
      base_config(1920, 1080, 1, TileLayoutPreset::k4x2);
  config.tile_qp.mode = TileQpMode::kDeltaQ;
  config.tile_qp.complexity_strength = 1.0;
  config.tile_qp.max_abs_delta = 48;
  config.tile_qp.delta_q_res = 0;

  FrameStats stats;
  stats.complexity = complexity(1920, 1080, 20.0);
  const std::array<double, 8> mads{{10.0, 20.0, 40.0, 20.0,
                                    5.0, 80.0, 20.0, 20.0}};
  for (double mad : mads) {
    TileStats tile;
    tile.complexity = complexity(64, 64, mad);
    stats.tiles.push_back(tile);
  }

  FramePipelinePlan plan;
  std::string error;
  CHECK(configure_and_plan(config, stats, plan, &error));
  CHECK(plan.base_q_idx == 90);
  CHECK(plan.tile_qp_mode == TileQpMode::kDeltaQ);
  CHECK(plan.delta_q.present);
  CHECK(plan.delta_q.res == 0);
  CHECK(plan.tiles.size() == 8);

  for (std::size_t i = 0; i < plan.tiles.size(); ++i) {
    const TilePlan& tile = plan.tiles[i];
    const int expected = tile_qindex_delta(
        mads[i], 20.0, config.tile_qp);
    CHECK(tile.target_qindex == 90 + expected);
    CHECK(tile.qindex == tile.target_qindex);
    CHECK(tile.delta_q_reduced == expected);
    CHECK(!tile.delta_plan.empty());
    CHECK(tile.delta_plan.steps.front().qindex_before == 90);
    CHECK(tile.delta_plan.steps.front().reduced_delta == expected);
    for (std::size_t s = 1; s < tile.delta_plan.steps.size(); ++s) {
      CHECK(tile.delta_plan.steps[s].reduced_delta == 0);
      CHECK(tile.delta_plan.steps[s].qindex_before == tile.qindex);
    }
    CHECK(tile.delta_plan.final_qindex == tile.qindex);
    // The decoder resets CurrentQIndex at every tile; the first step of every
    // plan therefore starts from base_q_idx, never from the previous tile.
    CHECK(tile.delta_plan.steps.front().qindex_before == 90);
  }

  CHECK(plan.tiles[0].qindex < 90);    // low activity -> finer
  CHECK(plan.tiles[2].qindex > 90);    // high activity -> coarser
  CHECK(plan.tiles[5].qindex > plan.tiles[2].qindex);
  CHECK(plan.qp.coeff_cdf_qctx == rc::coeff_cdf_qctx(plan.base_q_idx));
  CHECK(plan.qp.qp_slice == plan.qp.coeff_cdf_qctx);

  // With no per-tile statistics every tile inherits the frame sigma and
  // delta-Q is not signalled at all.
  FrameStats inherited;
  inherited.complexity = complexity(1920, 1080, 20.0);
  FramePipelinePlan flat;
  CHECK(configure_and_plan(config, inherited, flat, &error));
  CHECK(!flat.delta_q.present);
  CHECK(flat.delta_q.res == 0);
  for (const TilePlan& tile : flat.tiles) {
    CHECK(tile.qindex == flat.base_q_idx);
    CHECK(tile.delta_q_reduced == 0);
    CHECK(tile.delta_plan.empty());
  }

  // Non-zero delta_q_res quantises the reachable qindex; the achieved value is
  // reported and the coded delta is a multiple of 2^res.
  config.tile_qp.delta_q_res = 2;
  FramePipelinePlan quantised;
  CHECK(configure_and_plan(config, stats, quantised, &error));
  CHECK(quantised.delta_q.present);
  CHECK(quantised.delta_q.res == 2);
  for (const TilePlan& tile : quantised.tiles) {
    CHECK(tile.delta_q_reduced * 4 == tile.qindex - 90);
    CHECK((tile.qindex - plan.base_q_idx) % 4 == 0);
  }
}

void test_segmentation_tile_qp() {
  PipelineConfig config =
      base_config(256, 256, 1, TileLayoutPreset::k2x2);
  config.tile_qp.mode = TileQpMode::kSegmentation;
  config.tile_qp.complexity_strength = 1.0;
  config.tile_qp.max_abs_delta = 48;

  FrameStats stats;
  stats.complexity = complexity(256, 256, 20.0);
  for (double mad : {10.0, 20.0, 40.0, 80.0}) {
    TileStats tile;
    tile.complexity = complexity(64, 64, mad);
    stats.tiles.push_back(tile);
  }

  FramePipelinePlan plan;
  std::string error;
  CHECK(configure_and_plan(config, stats, plan, &error));
  CHECK(plan.tile_qp_mode == TileQpMode::kSegmentation);
  CHECK(plan.segmentation.enabled);
  CHECK(plan.segmentation.update_map);
  CHECK(!plan.segmentation.temporal_update);
  CHECK(plan.has_segment_map);
  CHECK(!plan.delta_q.present);
  CHECK(plan.segment_map.mi_cols == tiles::mi_cols(256));
  CHECK(plan.segment_map.mi_rows == tiles::mi_rows(256));
  CHECK(plan.segment_map.segment_id.size() ==
        static_cast<std::size_t>(plan.segment_map.mi_cols) *
            static_cast<std::size_t>(plan.segment_map.mi_rows));

  std::vector<int> used;
  for (const TilePlan& tile : plan.tiles) {
    CHECK(tile.segment_id >= 0);
    CHECK(tile.segment_id < rc::kMaxSegments);
    CHECK(tile.alt_q == tile.target_qindex - plan.base_q_idx);
    CHECK(tile.qindex == tile.target_qindex);
    CHECK(effective_tile_qindex(plan.base_q_idx, plan.delta_q,
                                plan.segmentation, tile.segment_id,
                                plan.base_q_idx) == tile.qindex);
    if (std::find(used.begin(), used.end(), tile.segment_id) == used.end()) {
      used.push_back(tile.segment_id);
    }
    // Every MI cell inside the tile maps to the same segment id.
    const int sb_mi = config.sb_size / 4;
    for (int row = tile.rect.sb_row_start * sb_mi;
         row < std::min(plan.segment_map.mi_rows,
                        (tile.rect.sb_row_start + tile.rect.sb_rows) * sb_mi);
         ++row) {
      for (int col = tile.rect.sb_col_start * sb_mi;
           col < std::min(plan.segment_map.mi_cols,
                          (tile.rect.sb_col_start + tile.rect.sb_cols) * sb_mi);
           ++col) {
        CHECK(plan.segment_map.segment_id[
                  static_cast<std::size_t>(row) *
                      static_cast<std::size_t>(plan.segment_map.mi_cols) +
                  static_cast<std::size_t>(col)] ==
              static_cast<std::uint8_t>(tile.segment_id));
      }
    }
  }
  CHECK(used.size() == 4);

  // All tiles at base -> segmentation stays off rather than coding a
  // one-segment map for no reason.
  config.tile_qp.complexity_strength = 0.0;
  FramePipelinePlan flat;
  CHECK(configure_and_plan(config, stats, flat, &error));
  CHECK(!flat.segmentation.enabled);
  CHECK(!flat.has_segment_map);
  for (const TilePlan& tile : flat.tiles) {
    CHECK(tile.qindex == flat.base_q_idx);
    CHECK(tile.alt_q == 0);
  }
}

void test_segmentation_segment_limit() {
  PipelineConfig config =
      base_config(1920, 1080, 4, TileLayoutPreset::k16x2);
  config.layout.allow_unverified = true;
  config.tile_qp.mode = TileQpMode::kSegmentation;
  config.tile_qp.complexity_strength = 2.0;
  config.tile_qp.max_abs_delta = 24;

  FrameStats stats;
  stats.complexity = complexity(1920, 1080, 20.0);
  for (int i = 0; i < 32; ++i) {
    TileStats tile;
    // strength 2 * 24 * log2(mad/20) = i - 24, so each tile requests a
    // different integer delta before clamping (32 distinct qindexes).
    const double mad =
        20.0 * std::pow(2.0, (static_cast<double>(i) - 24.0) / 48.0);
    tile.complexity = complexity(64, 64, mad);
    stats.tiles.push_back(tile);
  }

  PipelinePlanner planner;
  std::string error;
  CHECK(planner.configure(config, &error));
  FramePipelinePlan plan;
  CHECK(!planner.plan_frame(stats, plan, &error));
  CHECK(error.find("MAX_SEGMENTS") != std::string::npos);
  CHECK(!planner.frame_open());
}

void test_frame_rate_control() {
  // CQP: fixed base qindex everywhere, no delta/segmentation signalling.
  PipelineConfig config =
      base_config(1920, 1080, 1, TileLayoutPreset::k4x2);
  config.rate_control.cqp.base_q_idx = 60;
  FrameStats stats = stats_for(8, 20.0);
  FramePipelinePlan plan;
  std::string error;
  CHECK(configure_and_plan(config, stats, plan, &error));
  CHECK(plan.base_q_idx == 60);
  CHECK(plan.qp.qp_slice == rc::coeff_cdf_qctx(60));
  CHECK(plan.qp.lambda_sse > 0.0);
  CHECK(plan.estimated_frame_bits > 0.0);
  for (const TilePlan& tile : plan.tiles) {
    CHECK(tile.qindex == 60);
  }

  // CRF: higher crf -> coarser qindex.
  auto crf_plan = [&](double crf, FramePipelinePlan& out) {
    PipelineConfig crf_config = config;
    crf_config.rate_control.mode = rc::RateControlMode::kCrf;
    crf_config.rate_control.crf.crf = crf;
    return configure_and_plan(crf_config, stats, out, &error);
  };
  FramePipelinePlan crf_low;
  FramePipelinePlan crf_high;
  CHECK(crf_plan(5.0, crf_low));
  CHECK(crf_plan(55.0, crf_high));
  CHECK(crf_low.base_q_idx < crf_high.base_q_idx);

  // VBR: a lower target bitrate chooses a coarser qindex.
  auto vbr_plan = [&](double bitrate, FramePipelinePlan& out) {
    PipelineConfig vbr_config = config;
    vbr_config.rate_control.mode = rc::RateControlMode::kVbr;
    vbr_config.rate_control.vbr.target_bitrate_bps = bitrate;
    vbr_config.rate_control.vbr.frame_rate_num = 30;
    vbr_config.rate_control.vbr.frame_rate_den = 1;
    vbr_config.rate_control.vbr.max_qindex_step = 255;
    return configure_and_plan(vbr_config, stats, out, &error);
  };
  FramePipelinePlan low_rate;
  FramePipelinePlan high_rate;
  CHECK(vbr_plan(100000.0, low_rate));
  CHECK(vbr_plan(4000000.0, high_rate));
  CHECK(low_rate.base_q_idx > high_rate.base_q_idx);
  CHECK(low_rate.qp.estimated_bits > 0.0);
  CHECK(high_rate.qp.estimated_bits > 0.0);

  // A writer range that excludes the RC decision is an error, not a silent
  // clamp. The failed decision closes its RC frame.
  PipelineConfig pinned =
      base_config(1920, 1080, 1, TileLayoutPreset::k4x2);
  pinned.rate_control.cqp.base_q_idx = 60;
  pinned.min_base_q_idx = 90;
  pinned.max_base_q_idx = 90;
  PipelinePlanner planner;
  CHECK(planner.configure(pinned, &error));
  CHECK(!planner.plan_frame(stats, plan, &error));
  CHECK(error.find("outside the writer range") != std::string::npos);
  CHECK(!planner.frame_open());

  pinned.rate_control.cqp.base_q_idx = 90;
  PipelinePlanner ok_planner;
  CHECK(ok_planner.configure(pinned, &error));
  CHECK(ok_planner.plan_frame(stats, plan, &error));
  CHECK(plan.base_q_idx == 90);
  CHECK(ok_planner.end_frame(1000, &error));
}

void test_determinism_and_independence() {
  PipelineConfig config =
      base_config(1920, 1080, 2, TileLayoutPreset::k8x2);
  config.rate_control.mode = rc::RateControlMode::kVbr;
  config.rate_control.vbr.target_bitrate_bps = 800000.0;
  config.rate_control.vbr.max_qindex_step = 16;
  config.tile_qp.mode = TileQpMode::kDeltaQ;
  config.tile_qp.delta_q_res = 1;
  config.tile_qp.max_abs_delta = 16;

  std::vector<FrameStats> frames;
  for (int f = 0; f < 4; ++f) {
    FrameStats stats;
    stats.frame_index = static_cast<std::uint64_t>(f);
    stats.complexity =
        complexity(1920, 1080, 10.0 + 5.0 * static_cast<double>(f));
    for (int t = 0; t < 16; ++t) {
      TileStats tile;
      tile.complexity = complexity(
          64, 64, 8.0 + 3.0 * static_cast<double>(t) +
                       2.0 * static_cast<double>(f));
      stats.tiles.push_back(tile);
    }
    frames.push_back(stats);
  }
  const std::array<std::int64_t, 4> actual{{120000, 90000, 130000, 110000}};

  SequencePlan first;
  std::string error;
  CHECK(plan_sequence(config, frames, actual, first, &error));
  CHECK(first.valid);
  CHECK(first.frames.size() == frames.size());

  SequencePlan second;
  CHECK(plan_sequence(config, frames, actual, second, &error));
  CHECK(second.valid);
  for (std::size_t i = 0; i < first.frames.size(); ++i) {
    CHECK(frame_plan_equal(first.frames[i], second.frames[i]));
  }

  // A permuted per-tile stats list must produce the same per-tile decisions.
  std::vector<FrameStats> permuted = frames;
  std::reverse(permuted[1].tiles.begin(), permuted[1].tiles.end());
  for (std::size_t i = 0; i < permuted[1].tiles.size(); ++i) {
    permuted[1].tiles[i].tile_index =
        static_cast<int>(permuted[1].tiles.size() - 1 - i);
  }
  SequencePlan permuted_plan;
  CHECK(plan_sequence(config, permuted, actual, permuted_plan, &error));
  CHECK(frame_plan_equal(first.frames[1], permuted_plan.frames[1]));

  // Assignment policy changes the GPU split, never the QP decision.
  PipelineConfig contiguous = config;
  contiguous.gpu.assignment = multigpu::AssignmentPolicy::kContiguousRanges;
  SequencePlan contiguous_plan;
  CHECK(plan_sequence(contiguous, frames, actual, contiguous_plan, &error));
  for (std::size_t f = 0; f < first.frames.size(); ++f) {
    CHECK(first.frames[f].base_q_idx == contiguous_plan.frames[f].base_q_idx);
    for (std::size_t t = 0; t < first.frames[f].tiles.size(); ++t) {
      CHECK(first.frames[f].tiles[t].qindex ==
            contiguous_plan.frames[f].tiles[t].qindex);
      CHECK(first.frames[f].tiles[t].target_qindex ==
            contiguous_plan.frames[f].tiles[t].target_qindex);
    }
  }
}

void test_arena_budget() {
  using av2enc::cuda::pipeline::compute_arena_budget;
  cuda::pipeline::ArenaBudget budget;
  std::string error;

  CHECK(compute_arena_budget(8192ull * kMiB, 7000ull * kMiB, 512ull * kMiB, 0,
                             budget, &error));
  CHECK(budget.device_known);
  CHECK(budget.capacity_bytes == (7000ull - 512ull) * kMiB);
  CHECK(!budget.capped_by_request);

  CHECK(compute_arena_budget(8192ull * kMiB, 7000ull * kMiB, 512ull * kMiB,
                             4000ull * kMiB, budget, &error));
  CHECK(budget.capacity_bytes == 4000ull * kMiB);
  CHECK(budget.capped_by_request);

  CHECK(compute_arena_budget(8192ull * kMiB, 7000ull * kMiB, 512ull * kMiB,
                             9000ull * kMiB, budget, &error));
  CHECK(budget.capacity_bytes == (7000ull - 512ull) * kMiB);
  CHECK(!budget.capped_by_request);

  CHECK(compute_arena_budget(0, 0, 512ull * kMiB, 3000ull * kMiB, budget,
                             &error));
  CHECK(!budget.device_known);
  CHECK(budget.capacity_bytes == 3000ull * kMiB);

  CHECK(!compute_arena_budget(0, 0, 512ull * kMiB, 0, budget, &error));
  CHECK(!error.empty());
  CHECK(!compute_arena_budget(8192ull * kMiB, 0, 512ull * kMiB, 0, budget,
                              &error));
  CHECK(!compute_arena_budget(8192ull * kMiB, 512ull * kMiB,
                              512ull * kMiB, 0, budget, &error));
}

void test_vram_and_dispatch() {
  // 4K / 4 GPUs / 16x2 under an 8 GB device arena. The default planner table
  // excludes 16x2, so the gate must be opted into here.
  PipelineConfig config =
      base_config(3840, 2160, 4, TileLayoutPreset::k16x2);
  config.layout.allow_unverified = true;
  config.device.total_bytes = 8192ull * kMiB;
  config.device.free_bytes = 7800ull * kMiB;
  FrameStats stats = stats_for(32, 20.0);

  PipelinePlanner planner;
  std::string error;
  CHECK(planner.configure(config, &error));
  const std::uint64_t expected_capacity = (7800ull - 512ull) * kMiB;
  CHECK(planner.arena_capacity_bytes() == expected_capacity);
  CHECK(near(planner.config().gpu.vram_cap_mb,
             static_cast<double>(expected_capacity) / static_cast<double>(kMiB)));

  FramePipelinePlan plan;
  CHECK(planner.plan_frame(stats, plan, &error));
  CHECK(plan.gpu.valid);
  CHECK(plan.gpu.vram_fits);
  CHECK(plan.gpu.gpus.size() == 4);
  CHECK(plan.tiles.size() == 32);
  for (const multigpu::GpuPlan& gpu : plan.gpu.gpus) {
    CHECK(gpu.vram.fits);
    CHECK(gpu.vram.live_mb <= gpu.vram.cap_mb);
    CHECK(!gpu.tile_indices.empty());
  }

  // Reference pools are duplicated per GPU: (G-1) * refs * frame_bytes.
  const double frame_mb = plan.gpu.memory.frame_mb;
  CHECK(near(plan.gpu.memory.ref_pool_mb_per_gpu, 4.0 * frame_mb));
  CHECK(near(plan.gpu.memory.ref_extra_mb, 3.0 * 4.0 * frame_mb));

  // Host-mediated transfer plan is carried through unchanged.
  CHECK(plan.gpu.transfer.effective_gbps > 0.0);
  CHECK(plan.gpu.transfer.total_mb > 0.0);
  CHECK(plan.gpu.transfer.pct_of_frame_budget > 0.0);
  CHECK(plan.gpu.transfer.input_upload_mb > 0.0);
  CHECK(plan.gpu.transfer.ref_fill_mb > 0.0);
  CHECK(!plan.gpu.transfer.adaptive_cdf);

  // Every tile is assigned exactly once and the dispatch is raster within a
  // worker.
  std::vector<int> seen(32, 0);
  for (const multigpu::GpuPlan& gpu : plan.gpu.gpus) {
    int previous = -1;
    for (int index : gpu.tile_indices) {
      ++seen[static_cast<std::size_t>(index)];
      CHECK(index > previous);
      previous = index;
      CHECK(plan.tiles[static_cast<std::size_t>(index)].gpu_index ==
            gpu.gpu_index);
    }
  }
  for (int count : seen) {
    CHECK(count == 1);
  }

  std::vector<cuda::pipeline::GpuWorkerPlan> workers;
  CHECK(cuda::pipeline::build_gpu_dispatch(plan, expected_capacity, workers,
                                           &error));
  CHECK(workers.size() == 4);
  std::vector<int> dispatched(32, 0);
  for (const auto& worker : workers) {
    CHECK(worker.arena_capacity_bytes == expected_capacity);
    CHECK(worker.vram_fits);
    CHECK(!worker.tiles.empty());
    int previous = -1;
    for (const auto& tile : worker.tiles) {
      ++dispatched[static_cast<std::size_t>(tile.tile_index)];
      CHECK(tile.tile_index > previous);
      previous = tile.tile_index;
      const TilePlan& planned =
          plan.tiles[static_cast<std::size_t>(tile.tile_index)];
      CHECK(tile.x0 == planned.rect.x0);
      CHECK(tile.x1 == planned.rect.x1);
      CHECK(tile.qindex == planned.qindex);
      CHECK(tile.delta_q_reduced == planned.delta_q_reduced);
    }
  }
  for (int count : dispatched) {
    CHECK(count == 1);
  }

  // A cap that cannot hold the estimate is rejected unless the caller asks to
  // inspect an invalid plan.
  PipelineConfig small = config;
  small.device.free_bytes = 1000ull * kMiB;  // 488 MiB arena
  PipelinePlanner rejected;
  CHECK(!rejected.configure(small, &error));
  CHECK(error.find("does not fit") != std::string::npos);

  small.require_vram_fit = false;
  PipelinePlanner inspect;
  CHECK(inspect.configure(small, &error));
  FramePipelinePlan small_plan;
  CHECK(inspect.plan_frame(stats, small_plan, &error));
  CHECK(!small_plan.gpu.vram_fits);

  // A zero arena is not a valid dispatch.
  std::vector<cuda::pipeline::GpuWorkerPlan> bad_workers;
  CHECK(!cuda::pipeline::build_gpu_dispatch(plan, 0, bad_workers, &error));
  CHECK(!error.empty());
}

void test_frame_stats_validation() {
  const int tiles = 8;
  FrameStats stats = stats_for(tiles, 10.0);
  std::string error;
  CHECK(frame_stats_valid(stats, tiles, &error));

  FrameStats missing = stats;
  missing.tiles.pop_back();
  CHECK(!frame_stats_valid(missing, tiles, &error));

  FrameStats duplicate = stats_for(tiles, 10.0);
  duplicate.tiles[1].tile_index = 0;
  for (int i = 0; i < tiles; ++i) {
    duplicate.tiles[static_cast<std::size_t>(i)].tile_index = i;
  }
  duplicate.tiles[1].tile_index = 0;
  CHECK(!frame_stats_valid(duplicate, tiles, &error));

  FrameStats negative = stats_for(tiles, 10.0);
  for (int i = 0; i < tiles; ++i) {
    negative.tiles[static_cast<std::size_t>(i)].tile_index = i;
  }
  negative.tiles[3].tile_index = -2;
  CHECK(!frame_stats_valid(negative, tiles, &error));

  FrameStats partial = stats_for(tiles, 10.0);
  partial.tiles[0].complexity = rc::FrameComplexity{};
  partial.tiles[0].complexity.width = 64;
  CHECK(!frame_stats_valid(partial, tiles, &error));

  FrameStats bad_global;
  bad_global.complexity.width = 0;
  bad_global.complexity.height = 1080;
  CHECK(!frame_stats_valid(bad_global, tiles, &error));
}

void test_sequence_preview() {
  PipelineConfig config =
      base_config(1920, 1080, 2, TileLayoutPreset::k8x2);
  config.rate_control.mode = rc::RateControlMode::kCrf;
  config.rate_control.crf.crf = 30.0;

  std::vector<FrameStats> frames;
  for (int i = 0; i < 3; ++i) {
    FrameStats stats;
    stats.frame_index = static_cast<std::uint64_t>(i);
    stats.complexity = complexity(1920, 1080, 12.0 + static_cast<double>(i));
    frames.push_back(stats);
  }

  SequencePlan preview;
  std::string error;
  CHECK(plan_sequence(config, frames, {}, preview, &error));
  CHECK(preview.valid);
  CHECK(preview.frames.size() == 3);
  CHECK(preview.arena_capacity_bytes == (7800ull - 512ull) * kMiB);
  for (const auto& frame : preview.frames) {
    CHECK(frame.valid);
    CHECK(frame.base_q_idx == frame.qp.base_q_idx);
    CHECK(frame.base_q_idx >= rc::kMinQIndex);
    CHECK(frame.tiles.size() == 16);
  }

  // A mismatched actual-bits span is an explicit error.
  const std::array<std::int64_t, 2> wrong{{1, 2}};
  SequencePlan bad;
  CHECK(!plan_sequence(config, frames, wrong, bad, &error));
  CHECK(!error.empty());
}

void test_null_error_paths() {
  // A null error pointer is legal on every public entry point.
  PipelineConfig config =
      base_config(1920, 1080, 1, TileLayoutPreset::k4x2);
  config.tile_qp.mode = TileQpMode::kDeltaQ;
  FrameStats stats = stats_for(8, 20.0);
  stats.tiles[0].complexity = complexity(64, 64, 80.0);
  PipelinePlanner planner;
  CHECK(planner.configure(config, nullptr));
  FramePipelinePlan plan;
  CHECK(planner.plan_frame(stats, plan, nullptr));
  CHECK(planner.end_frame(1000, nullptr));

  PipelineConfig segmentation =
      base_config(256, 256, 1, TileLayoutPreset::k2x2);
  segmentation.tile_qp.mode = TileQpMode::kSegmentation;
  FrameStats seg_stats = stats_for(4, 20.0);
  seg_stats.tiles[1].complexity = complexity(64, 64, 80.0);
  PipelinePlanner seg_planner;
  CHECK(seg_planner.configure(segmentation, nullptr));
  CHECK(seg_planner.plan_frame(seg_stats, plan, nullptr));
  CHECK(seg_planner.end_frame(1000, nullptr));

  std::vector<FrameStats> frames{stats, stats};
  SequencePlan sequence;
  CHECK(plan_sequence(config, frames, {}, sequence, nullptr));
  CHECK(sequence.valid);

  cuda::pipeline::ArenaBudget budget;
  CHECK(cuda::pipeline::compute_arena_budget(8192ull * kMiB, 7000ull * kMiB,
                                             512ull * kMiB, 0, budget,
                                             nullptr));
}

}  // namespace

int main() {
  test_quantizer_helpers();
  test_layout_policy();
  test_tile_geometry();
  test_delta_q_tile_qp();
  test_segmentation_tile_qp();
  test_segmentation_segment_limit();
  test_frame_rate_control();
  test_determinism_and_independence();
  test_arena_budget();
  test_vram_and_dispatch();
  test_frame_stats_validation();
  test_sequence_preview();
  test_null_error_paths();

  std::cout << (g_failures == 0 ? "PASS" : "FAIL") << ": " << g_checks
            << " checks, " << g_failures << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
