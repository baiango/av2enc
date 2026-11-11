// AV2 v1.0.0 tile/frame CDF continuity implementation.
//
// The integer arithmetic for averaging and blending lives in
// `src/encoder/entropy/avg_cdf/` and is reused verbatim.  This file only
// decides *which* of those operations runs and when.

#include "encoder/entropy/continuity/cdf_continuity.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

namespace av2enc::entropy::continuity {
namespace {

bool fail(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

bool mul_overflow(std::uint32_t a, std::uint32_t b, std::uint32_t* out) {
  const std::uint64_t value = static_cast<std::uint64_t>(a) * b;
  if (value > std::numeric_limits<std::uint32_t>::max()) return false;
  *out = static_cast<std::uint32_t>(value);
  return true;
}

}  // namespace

std::uint32_t ceil_log2_u32(std::uint32_t value) noexcept {
  if (value <= 1u) return 0u;
  std::uint32_t bits = 0u;
  std::uint32_t v = value - 1u;
  while (v != 0u) {
    v >>= 1u;
    ++bits;
  }
  return bits;
}

TileInfoCdfField tile_info_cdf_field(bool enable_avg_cdf, bool avg_cdf_type,
                                     std::uint32_t tile_cols,
                                     std::uint32_t tile_rows, bool is_bridge,
                                     bool tip_frame_as_output) noexcept {
  TileInfoCdfField field;
  field.tile_cols = tile_cols == 0u ? 1u : tile_cols;
  field.tile_rows = tile_rows == 0u ? 1u : tile_rows;
  field.tile_cols_log2 = ceil_log2_u32(field.tile_cols);
  field.tile_rows_log2 = ceil_log2_u32(field.tile_rows);
  field.cdf_tail_coded = (field.tile_cols > 1u || field.tile_rows > 1u) &&
                         !is_bridge && !tip_frame_as_output;
  field.context_update_present =
      field.cdf_tail_coded && (!enable_avg_cdf || !avg_cdf_type);
  if (field.context_update_present) {
    field.context_update_bits =
        field.tile_rows_log2 + field.tile_cols_log2;
    field.context_update_max =
        (field.context_update_bits >= 31u)
            ? 0x7FFFFFFFu
            : ((1u << field.context_update_bits) - 1u);
  }
  return field;
}

bool validate_context_update_tile_id(const TileInfoCdfField& field,
                                     std::uint32_t context_update_tile_id,
                                     std::string* error) {
  std::uint32_t tile_count = 0;
  if (!mul_overflow(field.tile_cols, field.tile_rows, &tile_count)) {
    return fail(error, "tile_info: TileCols * TileRows overflows");
  }
  if (tile_count == 0u) {
    return fail(error, "tile_info: zero tiles");
  }
  if (!field.context_update_present) {
    // §5.18.7.2 sets the variable to 0 when the field is not coded.
    if (context_update_tile_id != 0u) {
      return fail(error,
                  "tile_info: context_update_tile_id must be 0 when the "
                  "field is not coded");
    }
    return true;
  }
  if (context_update_tile_id >= tile_count) {
    return fail(error,
                "tile_info: context_update_tile_id " +
                    std::to_string(context_update_tile_id) + " >= TileCols*"
                    "TileRows " + std::to_string(tile_count));
  }
  if (context_update_tile_id > field.context_update_max) {
    return fail(error,
                "tile_info: context_update_tile_id does not fit f(" +
                    std::to_string(field.context_update_bits) + ")");
  }
  return true;
}

TileExitPlan plan_tile_exit(bool enable_avg_cdf, bool avg_cdf_type,
                            std::uint32_t tile_num, std::uint32_t tile_cols,
                            std::uint32_t tile_rows,
                            std::uint32_t context_update_tile_id) noexcept {
  TileExitPlan plan;
  plan.tile_num = tile_num;
  const std::uint32_t cols = tile_cols == 0u ? 1u : tile_cols;
  const std::uint32_t rows = tile_rows == 0u ? 1u : tile_rows;
  std::uint32_t tile_count = 0;
  if (!mul_overflow(cols, rows, &tile_count)) tile_count = 0xFFFFFFFFu;
  plan.tile_count = tile_count;
  plan.num_log2 = avg_cdf::average_log2(tile_count);
  plan.average_limit = 1u << plan.num_log2;

  if (enable_avg_cdf && avg_cdf_type) {
    plan.avg_cdf = tile_num < plan.average_limit;
  } else {
    plan.copy_cdf = tile_num == context_update_tile_id;
  }
  return plan;
}

const char* frame_cdf_source_name(FrameCdfSource source) noexcept {
  switch (source) {
    case FrameCdfSource::kDefaults:
      return "defaults";
    case FrameCdfSource::kPrimaryReference:
      return "primary-reference";
    case FrameCdfSource::kPrimaryReferenceBlended:
      return "primary-reference-blended";
  }
  return "unknown";
}

FrameCdfPlan plan_frame_cdfs(const FrameCdfInput& input) noexcept {
  FrameCdfPlan plan;
  plan.qctx_slice = qctx_slice_for_base_q_idx(input.raw_base_q_idx);
  plan.disable_cdf_update = input.disable_cdf_update;
  plan.tile_info = tile_info_cdf_field(
      input.enable_avg_cdf, input.avg_cdf_type, input.tile_cols,
      input.tile_rows, input.is_bridge, input.tip_frame_as_output);
  plan.enable_avg_cdf = input.enable_avg_cdf;
  plan.avg_cdf_type = input.avg_cdf_type;
  (void)mul_overflow(plan.tile_info.tile_cols, plan.tile_info.tile_rows,
                     &plan.tile_count);
  plan.context_update_tile_id =
      plan.tile_info.context_update_present ? input.context_update_tile_id
                                            : 0u;

  // §5.18.2 set_primary_ref_frame_and_ctx(): FrameIsIntra (CLK/OLK) and
  // SWITCH_FRAME make choose_primary_secondary_ref_frame() return NONE/NONE;
  // any NONE result forces primary_ref_frame = PRIMARY_REF_NONE and
  // disable_cross_frame_cdf_init = 1.
  plan.primary_ref_none = input.keyframe || input.primary_ref_none ||
                          input.derived_primary_ref_none;
  plan.reset = plan.primary_ref_none ||
               input.disable_cross_frame_cdf_init;
  if (plan.reset) {
    plan.source = FrameCdfSource::kDefaults;
    plan.primary_ref_slot = 0;
    plan.blend_ref_slot = 0;
    return plan;
  }

  plan.primary_ref_slot = input.primary_ref_slot;
  // §7.4.4: blend_cdfs(ref_frame_idx[blendFrame]) runs after
  // load_cdfs(ref_frame_idx[primary_ref_frame]) only for
  // enable_avg_cdf && !avg_cdf_type, with a real blend frame, outside
  // bru_inactive, and not for TIP_FRAME_AS_OUTPUT.
  plan.blend = input.enable_avg_cdf && !input.avg_cdf_type &&
               input.blend_ref_present && !input.bru_inactive &&
               !input.tip_frame_as_output;
  if (plan.blend) {
    plan.blend_ref_slot = input.blend_ref_slot;
    plan.source = FrameCdfSource::kPrimaryReferenceBlended;
  } else {
    plan.blend_ref_slot = 0;
    plan.source = FrameCdfSource::kPrimaryReference;
  }
  return plan;
}

bool validate_frame_cdfs(const FrameCdfInput& input, std::string* error) {
  std::uint32_t tile_count = 0;
  if (!mul_overflow(input.tile_cols == 0u ? 1u : input.tile_cols,
                    input.tile_rows == 0u ? 1u : input.tile_rows,
                    &tile_count)) {
    return fail(error, "continuity: TileCols * TileRows overflows");
  }
  if (tile_count == 0u) {
    return fail(error, "continuity: zero tiles");
  }
  const TileInfoCdfField field = tile_info_cdf_field(
      input.enable_avg_cdf, input.avg_cdf_type, input.tile_cols,
      input.tile_rows, input.is_bridge, input.tip_frame_as_output);
  if (!validate_context_update_tile_id(field, input.context_update_tile_id,
                                       error)) {
    return false;
  }
  if (input.raw_base_q_idx < 0) {
    return fail(error, "continuity: negative raw base_q_idx");
  }
  const FrameCdfPlan plan = plan_frame_cdfs(input);
  if (!plan.reset) {
    if (input.primary_ref_slot >= kNumRefFrames) {
      return fail(error, "continuity: primary_ref_slot >= NUM_REF_FRAMES");
    }
    if (plan.blend && input.blend_ref_slot >= kNumRefFrames) {
      return fail(error, "continuity: blend_ref_slot >= NUM_REF_FRAMES");
    }
  }
  return true;
}

const char* continuity_status_name(ContinuityStatus status) noexcept {
  switch (status) {
    case ContinuityStatus::kOk:
      return "ok";
    case ContinuityStatus::kInvalidArgument:
      return "invalid-argument";
    case ContinuityStatus::kNotInitialised:
      return "not-initialised";
    case ContinuityStatus::kFrameAlreadyOpen:
      return "frame-already-open";
    case ContinuityStatus::kFrameNotOpen:
      return "frame-not-open";
    case ContinuityStatus::kTileAlreadyOpen:
      return "tile-already-open";
    case ContinuityStatus::kTileNotOpen:
      return "tile-not-open";
    case ContinuityStatus::kTileOutOfRange:
      return "tile-out-of-range";
    case ContinuityStatus::kFrameIncomplete:
      return "frame-incomplete";
    case ContinuityStatus::kContextOutOfRange:
      return "context-out-of-range";
    case ContinuityStatus::kCdfBankError:
      return "cdf-bank-error";
  }
  return "unknown";
}

bool FrameCdfContinuity::init_owned(
    std::span<const avg_cdf::CdfSpec> specs,
    std::uint32_t reference_contexts, std::string* error) {
  if (reference_contexts > kNumRefFrames) {
    return fail(error, "continuity: reference_contexts > NUM_REF_FRAMES");
  }
  avg_cdf::CdfBankConfig config;
  // The bank's own copy/avg switch is unused: this controller selects the
  // §8.2.4 branch from the per-frame tile_info() values.
  config.enable_avg_cdf = false;
  config.avg_cdf_type = false;
  config.context_update_tile_id = 0;
  config.reference_contexts = reference_contexts;
  if (!bank_.init_owned(specs, config, error)) return false;
  initialised_ = true;
  frame_open_ = false;
  tile_open_ = false;
  next_tile_ = 0;
  tile_count_ = 0;
  return true;
}

bool FrameCdfContinuity::init(std::span<const avg_cdf::CdfTable> tables,
                              std::uint32_t reference_contexts,
                              std::string* error) {
  if (reference_contexts > kNumRefFrames) {
    return fail(error, "continuity: reference_contexts > NUM_REF_FRAMES");
  }
  avg_cdf::CdfBankConfig config;
  config.enable_avg_cdf = false;
  config.avg_cdf_type = false;
  config.context_update_tile_id = 0;
  config.reference_contexts = reference_contexts;
  if (!bank_.init(tables, config, error)) return false;
  initialised_ = true;
  frame_open_ = false;
  tile_open_ = false;
  next_tile_ = 0;
  tile_count_ = 0;
  return true;
}

ContinuityStatus FrameCdfContinuity::begin_frame(const FrameCdfInput& input,
                                                 std::string* error) {
  if (!initialised_) return ContinuityStatus::kNotInitialised;
  if (frame_open_) return ContinuityStatus::kFrameAlreadyOpen;
  if (!validate_frame_cdfs(input, error)) {
    return ContinuityStatus::kInvalidArgument;
  }
  plan_ = plan_frame_cdfs(input);
  tile_count_ = plan_.tile_count;
  next_tile_ = 0;
  tile_open_ = false;

  if (plan_.reset) {
    bank_.begin_frame(true);
  } else {
    if (!bank_.load_reference(plan_.primary_ref_slot)) {
      fail(error, "continuity: load_cdfs context is unavailable");
      return ContinuityStatus::kContextOutOfRange;
    }
    if (plan_.blend && !bank_.blend_reference(plan_.blend_ref_slot)) {
      fail(error, "continuity: blend_cdfs context is unavailable");
      return ContinuityStatus::kContextOutOfRange;
    }
    // §8.2.2: the frame array is the tile seed.  `begin_frame(false)` copies
    // it into `tile` and leaves the frame state intact.
    bank_.begin_frame(false);
  }
  frame_open_ = true;
  return ContinuityStatus::kOk;
}

ContinuityStatus FrameCdfContinuity::begin_tile(std::string* error) {
  if (!initialised_) return ContinuityStatus::kNotInitialised;
  if (!frame_open_) return ContinuityStatus::kFrameNotOpen;
  if (tile_open_) return ContinuityStatus::kTileAlreadyOpen;
  if (next_tile_ >= tile_count_) {
    fail(error, "continuity: all tiles of the frame are already coded");
    return ContinuityStatus::kTileOutOfRange;
  }
  // §8.2.2: fresh frame -> Tile* copy at init_symbol().
  bank_.load_tile_cdfs();
  tile_open_ = true;
  return ContinuityStatus::kOk;
}

ContinuityStatus FrameCdfContinuity::end_tile(std::uint32_t tile_num,
                                              std::string* error) {
  if (!initialised_) return ContinuityStatus::kNotInitialised;
  if (!frame_open_) return ContinuityStatus::kFrameNotOpen;
  if (!tile_open_) return ContinuityStatus::kTileNotOpen;
  if (tile_num != next_tile_ || tile_num >= tile_count_) {
    fail(error, "continuity: end_tile out of order");
    return ContinuityStatus::kTileOutOfRange;
  }

  const TileExitPlan exit_plan =
      plan_tile_exit(plan_.enable_avg_cdf, plan_.avg_cdf_type, tile_num,
                     plan_.tile_info.tile_cols, plan_.tile_info.tile_rows,
                     plan_.context_update_tile_id);

  for (std::size_t table_index = 0; table_index < bank_.table_count();
       ++table_index) {
    avg_cdf::CdfTable& table = bank_.table(table_index);
    const std::size_t entries =
        static_cast<std::size_t>(table.symbols) + 1u;
    if (exit_plan.avg_cdf) {
      // §8.2.4 avg_cdf(): tile 0 initialises the accumulator; every tile in
      // the first 1 << numLog2 subtracts its scaled probability and adds its
      // scaled count.
      avg_cdf::average_cdf(table.saved, table.tile, table.symbols,
                           exit_plan.num_log2, tile_num == 0u);
    } else if (exit_plan.copy_cdf) {
      std::memcpy(table.saved, table.tile,
                  entries * sizeof(std::uint16_t));
    }
    // Tiles outside 1 << numLog2 run neither branch (§8.2.4).
  }

  tile_open_ = false;
  ++next_tile_;
  return ContinuityStatus::kOk;
}

ContinuityStatus FrameCdfContinuity::end_frame(std::string* error) {
  if (!initialised_) return ContinuityStatus::kNotInitialised;
  if (!frame_open_) return ContinuityStatus::kFrameNotOpen;
  if (tile_open_) {
    fail(error, "continuity: end_frame while a tile is still open");
    return ContinuityStatus::kTileNotOpen;
  }
  if (next_tile_ != tile_count_) {
    fail(error, "continuity: end_frame before the last tile");
    return ContinuityStatus::kFrameIncomplete;
  }
  // §7.5: Saved* -> frame*, count = (3 * count) >> 2.
  bank_.average_at_frame_end();
  frame_open_ = false;
  next_tile_ = 0;
  tile_count_ = 0;
  return ContinuityStatus::kOk;
}

ContinuityStatus FrameCdfContinuity::save_reference(std::uint32_t slot,
                                                    std::string* error) {
  if (!initialised_) return ContinuityStatus::kNotInitialised;
  if (frame_open_) {
    fail(error, "continuity: save_cdfs before frame_end_update_cdf");
    return ContinuityStatus::kFrameNotOpen;
  }
  if (slot >= kNumRefFrames || !bank_.save_reference(slot)) {
    fail(error, "continuity: save_cdfs context is unavailable");
    return ContinuityStatus::kContextOutOfRange;
  }
  return ContinuityStatus::kOk;
}

TileExitPlan FrameCdfContinuity::tile_exit_plan(
    std::uint32_t tile_num) const noexcept {
  return plan_tile_exit(plan_.enable_avg_cdf, plan_.avg_cdf_type, tile_num,
                        plan_.tile_info.tile_cols, plan_.tile_info.tile_rows,
                        plan_.context_update_tile_id);
}

}  // namespace av2enc::entropy::continuity
