#include "pipeline/planner/pipeline.h"

#include "cuda/pipeline/planner/pipeline_gpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::pipeline {
namespace {

// Matches src/encoder/rc/: the normative quantizer step doubles roughly every
// 24 qindex steps, which is the conversion used by the CRF qcomp term.
constexpr double kQpOctave = 24.0;

void set_error(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

bool finite_nonnegative(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

int clamp_qindex_value(int qindex) noexcept {
  return std::clamp(qindex, rc::kMinQIndex, rc::kMaxQIndex);
}

// Largest first. Used only after the policy's preferred preset has been
// tried, so kAuto still follows the documented per-GPU default.
constexpr std::array<tiles::TileLayoutPreset, 6> kPresetsLargestFirst{
    tiles::TileLayoutPreset::k16x2, tiles::TileLayoutPreset::k8x2,
    tiles::TileLayoutPreset::k4x2,  tiles::TileLayoutPreset::k2x2,
    tiles::TileLayoutPreset::k2x1,  tiles::TileLayoutPreset::k1x1};

struct OrderedTileStats {
  std::vector<const TileStats*> by_index;
};

bool all_indices_unspecified(const std::vector<TileStats>& tiles) noexcept {
  for (const TileStats& tile : tiles) {
    if (tile.tile_index != -1) {
      return false;
    }
  }
  return true;
}

OrderedTileStats order_tile_stats(const std::vector<TileStats>& tiles,
                                  int tile_count) {
  OrderedTileStats ordered;
  ordered.by_index.assign(static_cast<std::size_t>(tile_count), nullptr);
  if (tiles.empty()) {
    return ordered;
  }
  if (all_indices_unspecified(tiles)) {
    for (std::size_t i = 0; i < tiles.size(); ++i) {
      ordered.by_index[i] = &tiles[i];
    }
    return ordered;
  }
  for (const TileStats& tile : tiles) {
    if (tile.tile_index >= 0 &&
        tile.tile_index < static_cast<int>(ordered.by_index.size())) {
      ordered.by_index[static_cast<std::size_t>(tile.tile_index)] = &tile;
    }
  }
  return ordered;
}

bool build_segment_map(const tiles::TileLayout& layout, int sb_size,
                       const std::vector<int>& segment_ids,
                       rc::SegmentationMap& map, std::string* error) {
  const int mi_cols = tiles::mi_cols(layout.width);
  const int mi_rows = tiles::mi_rows(layout.height);
  if (mi_cols <= 0 || mi_rows <= 0) {
    set_error(error, "segment map: invalid frame geometry");
    return false;
  }
  if (static_cast<int>(segment_ids.size()) != layout.tile_count()) {
    set_error(error, "segment map: segment id count does not match tile count");
    return false;
  }
  const int sb_mi = sb_size / 4;
  if (sb_mi <= 0) {
    set_error(error, "segment map: invalid superblock size");
    return false;
  }

  map.mi_cols = mi_cols;
  map.mi_rows = mi_rows;
  map.segment_id.assign(
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows), 0);

  for (const tiles::TileRect& tile : layout.tiles) {
    if (tile.index < 0 || tile.index >= static_cast<int>(segment_ids.size())) {
      set_error(error, "segment map: tile index out of range");
      return false;
    }
    const std::uint8_t id =
        static_cast<std::uint8_t>(segment_ids[static_cast<std::size_t>(
            tile.index)]);
    const int col_begin = tile.sb_col_start * sb_mi;
    const int row_begin = tile.sb_row_start * sb_mi;
    const int col_end =
        std::min(mi_cols, (tile.sb_col_start + tile.sb_cols) * sb_mi);
    const int row_end =
        std::min(mi_rows, (tile.sb_row_start + tile.sb_rows) * sb_mi);
    for (int row = row_begin; row < row_end; ++row) {
      const std::size_t base =
          static_cast<std::size_t>(row) * static_cast<std::size_t>(mi_cols);
      for (int col = col_begin; col < col_end; ++col) {
        map.segment_id[base + static_cast<std::size_t>(col)] = id;
      }
    }
  }
  return true;
}

bool select_layout(const PipelineConfig& config, tiles::TileLayout& layout,
                   int& selected_preset_index, std::string& reason,
                   std::string* error) {
  const int gpu_count = config.gpu.gpu_count;
  std::vector<tiles::TileLayoutPreset> candidates;
  const tiles::TileLayoutPreset preferred =
      config.layout.mode == LayoutPolicyMode::kExplicit
          ? config.layout.explicit_preset
          : default_layout_for_gpus(gpu_count, config.layout);
  candidates.push_back(preferred);
  for (tiles::TileLayoutPreset preset : kPresetsLargestFirst) {
    if (preset != preferred) {
      candidates.push_back(preset);
    }
  }

  std::string last_error;
  for (tiles::TileLayoutPreset preset : candidates) {
    if (!layout_preset_allowed(preset, config.layout)) {
      last_error = std::string("layout ") + tiles::preset_name(preset) +
                   " is not decoder-verified; run the tiled dav2d gate or set "
                   "allow_unverified";
      if (config.layout.mode == LayoutPolicyMode::kExplicit) {
        set_error(error, last_error);
        return false;
      }
      continue;
    }
    const tiles::TileLayoutDims dims = tiles::preset_dims(preset);
    const tiles::LayoutResult result = tiles::plan_tile_layout(
        config.width, config.height, dims.cols, dims.rows, config.sb_size,
        config.seq_level_idx, config.seq_tier);
    if (!result.ok()) {
      last_error = std::string("layout ") + tiles::preset_name(preset) +
                   ": " + result.error;
      if (config.layout.mode == LayoutPolicyMode::kExplicit) {
        set_error(error, last_error);
        return false;
      }
      continue;
    }
    if (result.layout.tile_count() < gpu_count) {
      std::ostringstream out;
      out << "layout " << tiles::preset_name(preset) << " has "
          << result.layout.tile_count() << " tiles for " << gpu_count
          << " GPUs";
      last_error = out.str();
      if (config.layout.mode == LayoutPolicyMode::kExplicit) {
        set_error(error, last_error);
        return false;
      }
      continue;
    }
    std::string validation_error;
    if (!tiles::validate_tile_layout(result.layout, validation_error)) {
      last_error = std::string("layout ") + tiles::preset_name(preset) +
                   ": " + validation_error;
      if (config.layout.mode == LayoutPolicyMode::kExplicit) {
        set_error(error, last_error);
        return false;
      }
      continue;
    }

    layout = result.layout;
    selected_preset_index = preset_index(preset);
    if (preset == preferred) {
      reason = std::string(config.layout.mode == LayoutPolicyMode::kAuto
                               ? "auto: "
                               : "explicit: ") +
               tiles::preset_name(preset);
    } else {
      reason = std::string("auto: preferred ") +
               tiles::preset_name(preferred) +
               " does not fit; selected " + tiles::preset_name(preset);
    }
    return true;
  }

  if (last_error.empty()) {
    last_error = "no decoder-verified tile layout fits this frame and GPU "
                 "count";
  }
  set_error(error, last_error);
  return false;
}

int tile_target_qindex(const TileStats* stats, double frame_sigma,
                       int base_q_idx, const TileQpConfig& config) {
  if (stats == nullptr || !rc::frame_complexity_valid(stats->complexity)) {
    return base_q_idx;
  }
  const double tile_sigma = rc::frame_complexity_sigma(stats->complexity);
  const int delta = tile_qindex_delta(tile_sigma, frame_sigma, config);
  return clamp_qindex_value(base_q_idx + delta);
}

bool vram_failure(const multigpu::MultiGpuPlan& plan, std::string* error) {
  std::ostringstream out;
  bool any = false;
  for (const multigpu::GpuPlan& gpu : plan.gpus) {
    if (!gpu.vram.fits) {
      if (any) {
        out << "; ";
      }
      out << "GPU " << gpu.gpu_index << " live " << gpu.vram.live_mb
          << " MiB > arena cap " << gpu.vram.cap_mb << " MiB";
      any = true;
    }
  }
  if (!any) {
    return false;
  }
  set_error(error, "per-GPU VRAM estimate does not fit the arena: " + out.str());
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Names and policies
// ---------------------------------------------------------------------------

const char* layout_policy_mode_name(LayoutPolicyMode mode) noexcept {
  switch (mode) {
    case LayoutPolicyMode::kAuto:
      return "auto";
    case LayoutPolicyMode::kExplicit:
      return "explicit";
  }
  return "?";
}

const char* layout_preset_name(tiles::TileLayoutPreset preset) noexcept {
  return tiles::preset_name(preset);
}

const char* tile_qp_mode_name(TileQpMode mode) noexcept {
  switch (mode) {
    case TileQpMode::kUniform:
      return "uniform";
    case TileQpMode::kDeltaQ:
      return "delta-q";
    case TileQpMode::kSegmentation:
      return "segmentation";
  }
  return "?";
}

int preset_index(tiles::TileLayoutPreset preset) noexcept {
  return static_cast<int>(preset);
}

bool layout_preset_allowed(tiles::TileLayoutPreset preset,
                           const LayoutPolicy& policy) noexcept {
  const int index = preset_index(preset);
  if (index < 0 || index >= static_cast<int>(policy.decoder_verified.size())) {
    return false;
  }
  return policy.decoder_verified[static_cast<std::size_t>(index)] ||
         policy.allow_unverified;
}

tiles::TileLayoutPreset default_layout_for_gpus(
    int gpu_count, const LayoutPolicy& policy) noexcept {
  if (gpu_count <= 1) {
    return tiles::TileLayoutPreset::k4x2;
  }
  if (gpu_count == 2) {
    return tiles::TileLayoutPreset::k8x2;
  }
  if (layout_preset_allowed(tiles::TileLayoutPreset::k16x2, policy)) {
    return tiles::TileLayoutPreset::k16x2;
  }
  return tiles::TileLayoutPreset::k8x2;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

bool tile_qp_config_valid(const TileQpConfig& config, std::string* error) {
  switch (config.mode) {
    case TileQpMode::kUniform:
    case TileQpMode::kDeltaQ:
    case TileQpMode::kSegmentation:
      break;
    default:
      set_error(error, "tile QP: unknown mode");
      return false;
  }
  if (!finite_nonnegative(config.complexity_strength)) {
    set_error(error, "tile QP: complexity_strength must be finite and >= 0");
    return false;
  }
  if (config.max_abs_delta < 0 || config.max_abs_delta > rc::kMaxQIndex) {
    set_error(error, "tile QP: max_abs_delta must be in [0, 255]");
    return false;
  }
  if (config.delta_q_res < 0 || config.delta_q_res > 3) {
    set_error(error, "tile QP: delta_q_res must be in [0, 3]");
    return false;
  }
  if (config.mode != TileQpMode::kDeltaQ && config.delta_q_res != 0) {
    set_error(error, "tile QP: delta_q_res is only used by delta-Q mode");
    return false;
  }
  return true;
}

bool pipeline_config_valid(const PipelineConfig& config, std::string* error) {
  if (config.width <= 0 || config.height <= 0) {
    set_error(error, "pipeline: width and height must be positive");
    return false;
  }
  if (config.width % 2 != 0 || config.height % 2 != 0) {
    set_error(error, "pipeline: 4:2:0 planning requires even dimensions");
    return false;
  }
  if (config.sb_size != 64 && config.sb_size != 128) {
    set_error(error, "pipeline: sb_size must be 64 or 128");
    return false;
  }
  if (config.seq_level_idx < 0 || config.seq_level_idx > 31) {
    set_error(error, "pipeline: seq_level_idx must be in [0, 31]");
    return false;
  }
  if (config.seq_tier < 0 || config.seq_tier > 1) {
    set_error(error, "pipeline: seq_tier must be 0 or 1");
    return false;
  }
  if (config.gpu.gpu_count < 1 || config.gpu.gpu_count > 8) {
    set_error(error, "pipeline: GPU count must be 1..8");
    return false;
  }
  if (!tile_qp_config_valid(config.tile_qp, error)) {
    return false;
  }
  if (!rc::rate_control_config_valid(config.rate_control, error)) {
    return false;
  }
  if (config.min_base_q_idx < rc::kMinQIndex ||
      config.max_base_q_idx > rc::kMaxQIndex ||
      config.min_base_q_idx > config.max_base_q_idx) {
    set_error(error,
              "pipeline: base_q_idx range must satisfy 1 <= min <= max <= 255");
    return false;
  }
  if (config.device.headroom_bytes == 0 &&
      config.device.total_bytes == 0 && config.device.free_bytes == 0 &&
      config.device.cap_bytes == 0) {
    set_error(error,
              "pipeline: provide a device budget or an explicit VRAM cap");
    return false;
  }
  return true;
}

bool frame_stats_valid(const FrameStats& stats, int tile_count,
                       std::string* error) {
  if (!rc::frame_complexity_valid(stats.complexity, error)) {
    return false;
  }
  if (stats.tiles.empty()) {
    return true;
  }
  if (tile_count <= 0) {
    set_error(error, "frame stats: tile count must be positive");
    return false;
  }
  if (static_cast<int>(stats.tiles.size()) != tile_count) {
    std::ostringstream out;
    out << "frame stats: " << stats.tiles.size() << " tile entries for "
        << tile_count << " tiles";
    set_error(error, out.str());
    return false;
  }

  const bool unspecified = all_indices_unspecified(stats.tiles);
  std::vector<bool> seen(static_cast<std::size_t>(tile_count), false);
  for (const TileStats& tile : stats.tiles) {
    int index = tile.tile_index;
    if (unspecified) {
      index = static_cast<int>(&tile - stats.tiles.data());
    } else {
      if (index < 0 || index >= tile_count) {
        set_error(error, "frame stats: tile index out of range");
        return false;
      }
      if (seen[static_cast<std::size_t>(index)]) {
        set_error(error, "frame stats: duplicate tile index");
        return false;
      }
      seen[static_cast<std::size_t>(index)] = true;
    }
    const rc::FrameComplexity& complexity = tile.complexity;
    if (complexity.width == 0 && complexity.height == 0) {
      continue;  // inherit the frame complexity
    }
    if (complexity.width <= 0 || complexity.height <= 0) {
      set_error(error,
                "frame stats: tile complexity must have both dimensions or "
                "neither");
      return false;
    }
    if (!rc::frame_complexity_valid(complexity, error)) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Deterministic helpers
// ---------------------------------------------------------------------------

int tile_qindex_delta(double tile_sigma, double frame_sigma,
                      const TileQpConfig& config) noexcept {
  if (!std::isfinite(tile_sigma) || !std::isfinite(frame_sigma) ||
      tile_sigma <= 0.0 || frame_sigma <= 0.0 || config.complexity_strength <= 0.0) {
    return 0;
  }
  const double octaves = std::log2(tile_sigma / frame_sigma);
  if (!std::isfinite(octaves)) {
    return 0;
  }
  const double raw = config.complexity_strength * kQpOctave * octaves;
  long long rounded = 0;
  if (raw >= static_cast<double>(std::numeric_limits<int>::max())) {
    rounded = std::numeric_limits<int>::max();
  } else if (raw <= static_cast<double>(std::numeric_limits<int>::min())) {
    rounded = std::numeric_limits<int>::min();
  } else {
    rounded = std::llround(raw);
  }
  const long long limit = config.max_abs_delta;
  const long long clamped = std::clamp(rounded, -limit, limit);
  return static_cast<int>(clamped);
}

int effective_tile_qindex(int base_q_idx, const rc::DeltaQParams& delta_q,
                          const rc::SegmentationParams& segmentation,
                          int segment_id, int current_q_index) noexcept {
  rc::QpContext context;
  context.base_q_idx = base_q_idx;
  context.delta_q = delta_q;
  context.current_q_index = current_q_index;
  context.segmentation = segmentation;
  int qindex = base_q_idx;
  if (!rc::effective_qindex(context, segment_id, /*ignore_delta_q=*/false,
                            qindex, nullptr)) {
    return clamp_qindex_value(base_q_idx);
  }
  return qindex;
}

// ---------------------------------------------------------------------------
// PipelinePlanner
// ---------------------------------------------------------------------------

bool PipelinePlanner::configure(const PipelineConfig& config,
                                std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  configured_ = false;
  frame_open_ = false;
  layout_ = tiles::TileLayout{};
  layout_reason_.clear();
  layout_preset_index_ = -1;
  arena_capacity_bytes_ = 0;
  gpu_plan_ = multigpu::MultiGpuPlan{};

  if (!pipeline_config_valid(config, error)) {
    return false;
  }

  cuda::pipeline::ArenaBudget budget;
  if (!cuda::pipeline::compute_arena_budget(
          config.device.total_bytes, config.device.free_bytes,
          config.device.headroom_bytes, config.device.cap_bytes, budget,
          error)) {
    return false;
  }
  arena_capacity_bytes_ = budget.capacity_bytes;

  PipelineConfig stored = config;
  stored.gpu.vram_cap_mb =
      static_cast<double>(arena_capacity_bytes_) /
      static_cast<double>(cuda::pipeline::kMiB);

  tiles::TileLayout layout;
  int preset_index_value = -1;
  std::string reason;
  if (!select_layout(stored, layout, preset_index_value, reason, error)) {
    return false;
  }

  multigpu::MultiGpuPlan gpu_plan;
  std::string gpu_error;
  if (!multigpu::plan_multi_gpu(layout, stored.gpu, gpu_plan, gpu_error)) {
    set_error(error, gpu_error);
    return false;
  }
  if (stored.require_vram_fit && vram_failure(gpu_plan, error)) {
    return false;
  }

  rc::RateController controller;
  if (!controller.configure(stored.rate_control, error)) {
    return false;
  }

  config_ = stored;
  layout_ = std::move(layout);
  layout_reason_ = std::move(reason);
  layout_preset_index_ = preset_index_value;
  gpu_plan_ = std::move(gpu_plan);
  rc_ = std::move(controller);
  configured_ = true;
  return true;
}

bool PipelinePlanner::plan_frame(const FrameStats& stats,
                                 FramePipelinePlan& plan, std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  plan = FramePipelinePlan{};
  if (!configured_) {
    set_error(error, "pipeline: configure() must be called first");
    return false;
  }
  if (frame_open_) {
    set_error(error,
              "pipeline: end_frame() must precede the next plan_frame()");
    return false;
  }
  if (!frame_stats_valid(stats, layout_.tile_count(), error)) {
    return false;
  }

  rc::QpDecision qp;
  if (!rc_.begin_frame(stats.complexity, qp, error)) {
    return false;
  }
  frame_open_ = true;

  const auto fail_after_begin = [this, &qp, error](const std::string& message) {
    // Keep the controller usable: a failed decision still closes its frame
    // with the deterministic model estimate. An encode loop that sees this
    // error must abort; it must not continue as if a frame had been planned.
    std::string ignored;
    (void)rc_.end_frame(static_cast<std::int64_t>(std::llround(
                            std::min(qp.estimated_bits, 1.0e12))),
                        &ignored);
    frame_open_ = false;
    set_error(error, message);
    return false;
  };

  if (qp.base_q_idx < config_.min_base_q_idx ||
      qp.base_q_idx > config_.max_base_q_idx) {
    std::ostringstream out;
    out << "pipeline: rate control chose base_q_idx " << qp.base_q_idx
        << " outside the writer range [" << config_.min_base_q_idx << ", "
        << config_.max_base_q_idx << "]";
    return fail_after_begin(out.str());
  }

  const int tile_count = layout_.tile_count();
  const OrderedTileStats ordered = order_tile_stats(stats.tiles, tile_count);
  const double frame_sigma = rc::frame_complexity_sigma(stats.complexity);

  std::vector<int> target(static_cast<std::size_t>(tile_count),
                          qp.base_q_idx);
  if (config_.tile_qp.mode != TileQpMode::kUniform) {
    for (int i = 0; i < tile_count; ++i) {
      target[static_cast<std::size_t>(i)] =
          tile_target_qindex(ordered.by_index[static_cast<std::size_t>(i)],
                             frame_sigma, qp.base_q_idx, config_.tile_qp);
    }
  }

  FramePipelinePlan out;
  out.frame_index = stats.frame_index;
  out.qp = qp;
  out.base_q_idx = qp.base_q_idx;
  out.tile_qp_mode = config_.tile_qp.mode;
  out.layout = layout_;
  out.layout_reason = layout_reason_;
  out.gpu = gpu_plan_;
  out.estimated_frame_bits = qp.estimated_bits;

  std::vector<int> gpu_of_tile(static_cast<std::size_t>(tile_count), -1);
  for (int g = 0; g < static_cast<int>(out.gpu.gpus.size()); ++g) {
    for (int index : out.gpu.gpus[static_cast<std::size_t>(g)].tile_indices) {
      if (index >= 0 && index < tile_count) {
        gpu_of_tile[static_cast<std::size_t>(index)] = g;
      }
    }
  }

  out.tiles.resize(static_cast<std::size_t>(tile_count));
  for (int i = 0; i < tile_count; ++i) {
    TilePlan& tile = out.tiles[static_cast<std::size_t>(i)];
    tile.index = i;
    tile.rect = layout_.tiles[static_cast<std::size_t>(i)];
    tile.row = tile.rect.row;
    tile.col = tile.rect.col;
    tile.gpu_index = gpu_of_tile[static_cast<std::size_t>(i)];
    tile.target_qindex = target[static_cast<std::size_t>(i)];
    tile.qindex = tile.target_qindex;
  }

  // Internal validators take a pointer; keep a local so a null caller error
  // cannot be dereferenced on the failure path.
  std::string detail;

  if (config_.tile_qp.mode == TileQpMode::kDeltaQ) {
    rc::DeltaQParams params;
    params.res = config_.tile_qp.delta_q_res;
    bool any_delta = false;
    for (int i = 0; i < tile_count; ++i) {
      TilePlan& tile = out.tiles[static_cast<std::size_t>(i)];
      rc::DeltaQPlanConfig plan_config;
      plan_config.base_q_idx = qp.base_q_idx;
      plan_config.delta_q_res = params.res;
      plan_config.cols = tile.rect.sb_cols;
      plan_config.rows = tile.rect.sb_rows;
      const std::size_t cells = static_cast<std::size_t>(plan_config.cols) *
                                static_cast<std::size_t>(plan_config.rows);
      std::vector<int> targets(cells, tile.target_qindex);
      if (!rc::plan_delta_q(plan_config, targets, tile.delta_plan, &detail)) {
        return fail_after_begin(detail);
      }
      if (!tile.delta_plan.empty()) {
        tile.delta_q_reduced = tile.delta_plan.steps.front().reduced_delta;
      }
      tile.qindex = tile.delta_plan.empty()
                        ? qp.base_q_idx
                        : tile.delta_plan.final_qindex;
      any_delta = any_delta || tile.delta_plan.max_abs_reduced_delta > 0;
    }
    params.present = any_delta;
    if (!any_delta) {
      params.res = 0;
      for (TilePlan& tile : out.tiles) {
        tile.qindex = qp.base_q_idx;
        tile.delta_plan.clear();
        tile.delta_q_reduced = 0;
      }
    }
    if (!rc::delta_q_params_valid(params, &detail)) {
      return fail_after_begin(detail);
    }
    out.delta_q = params;
  } else if (config_.tile_qp.mode == TileQpMode::kSegmentation) {
    std::vector<int> distinct = target;
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    if (static_cast<int>(distinct.size()) > rc::kMaxSegments) {
      std::ostringstream out_message;
      out_message << "pipeline: segmentation tile QP needs "
                  << distinct.size() << " segments; MAX_SEGMENTS is "
                  << rc::kMaxSegments
                  << " (use delta-Q or a coarser max_abs_delta)";
      return fail_after_begin(out_message.str());
    }

    rc::SegmentationParams segmentation;
    std::vector<int> segment_ids(static_cast<std::size_t>(tile_count), 0);
    const bool all_base =
        distinct.size() == 1 && distinct.front() == qp.base_q_idx;
    if (!all_base) {
      segmentation.enabled = true;
      segmentation.update_map = true;
      segmentation.temporal_update = false;
      for (int i = 0; i < static_cast<int>(distinct.size()); ++i) {
        const int alt_q = distinct[static_cast<std::size_t>(i)] - qp.base_q_idx;
        if (!rc::set_segment_alt_q(segmentation, i, alt_q, &detail)) {
          return fail_after_begin(detail);
        }
      }
      for (int i = 0; i < tile_count; ++i) {
        const int value = target[static_cast<std::size_t>(i)];
        const auto it =
            std::lower_bound(distinct.begin(), distinct.end(), value);
        segment_ids[static_cast<std::size_t>(i)] =
            static_cast<int>(it - distinct.begin());
      }
      rc::SegmentationMap map;
      if (!build_segment_map(layout_, config_.sb_size, segment_ids, map,
                             &detail)) {
        return fail_after_begin(detail);
      }
      if (!rc::segmentation_map_valid(map, &detail)) {
        return fail_after_begin(detail);
      }
      out.segment_map = std::move(map);
      out.has_segment_map = true;
    }
    if (!rc::segmentation_params_valid(segmentation, &detail)) {
      return fail_after_begin(detail);
    }
    out.segmentation = segmentation;
    for (int i = 0; i < tile_count; ++i) {
      TilePlan& tile = out.tiles[static_cast<std::size_t>(i)];
      tile.segment_id = static_cast<int>(segment_ids[static_cast<std::size_t>(
          i)]);
      tile.alt_q = tile.target_qindex - qp.base_q_idx;
      tile.qindex =
          clamp_qindex_value(qp.base_q_idx + tile.alt_q);
      tile.qindex = effective_tile_qindex(qp.base_q_idx, out.delta_q,
                                          out.segmentation, tile.segment_id,
                                          qp.base_q_idx);
    }
  }

  for (const TilePlan& tile : out.tiles) {
    if (tile.gpu_index < 0) {
      return fail_after_begin("pipeline: tile is not assigned to a GPU");
    }
  }

  out.valid = true;
  plan = std::move(out);
  return true;
}

bool PipelinePlanner::end_frame(std::int64_t actual_bits,
                                std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  if (!configured_) {
    set_error(error, "pipeline: configure() must be called first");
    return false;
  }
  if (!frame_open_) {
    set_error(error, "pipeline: end_frame() without plan_frame()");
    return false;
  }
  if (!rc_.end_frame(actual_bits, error)) {
    return false;
  }
  frame_open_ = false;
  return true;
}

void PipelinePlanner::reset() noexcept {
  rc_.reset();
  frame_open_ = false;
}

// ---------------------------------------------------------------------------
// Batch preview
// ---------------------------------------------------------------------------

bool plan_sequence(const PipelineConfig& config,
                   std::span<const FrameStats> stats,
                   std::span<const std::int64_t> actual_bits,
                   SequencePlan& plan, std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  plan = SequencePlan{};
  if (!actual_bits.empty() && actual_bits.size() != stats.size()) {
    set_error(error, "plan_sequence: actual_bits size does not match frames");
    return false;
  }
  if (stats.empty()) {
    set_error(error, "plan_sequence: no frames");
    return false;
  }

  PipelinePlanner planner;
  if (!planner.configure(config, error)) {
    return false;
  }
  plan.config = planner.config();
  plan.arena_capacity_bytes = planner.arena_capacity_bytes();
  plan.layout = planner.layout();
  plan.frames.resize(stats.size());

  for (std::size_t i = 0; i < stats.size(); ++i) {
    FramePipelinePlan frame;
    if (!planner.plan_frame(stats[i], frame, error)) {
      set_error(error, std::string("plan_sequence frame ") +
                           std::to_string(i) + ": " +
                           (error != nullptr ? *error : std::string()));
      return false;
    }
    std::int64_t bits = 0;
    if (actual_bits.empty()) {
      const double estimate =
          std::min(frame.estimated_frame_bits, 1.0e12);
      bits = static_cast<std::int64_t>(std::llround(estimate));
    } else {
      bits = actual_bits[i];
    }
    if (!planner.end_frame(bits, error)) {
      return false;
    }
    plan.frames[i] = std::move(frame);
  }
  plan.valid = true;
  return true;
}

}  // namespace av2enc::pipeline
