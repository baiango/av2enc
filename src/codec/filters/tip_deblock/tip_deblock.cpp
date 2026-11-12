#include "codec/filters/tip_deblock/tip_deblock.h"

#include <cstddef>

namespace av2enc::tip_deblock {
namespace {

int plane_shift_x(const TipDeblockFrame& frame, int plane) {
  return (plane == 0) ? 0 : frame.subsampling_x;
}

int plane_shift_y(const TipDeblockFrame& frame, int plane) {
  return (plane == 0) ? 0 : frame.subsampling_y;
}

TipDeblockStrength to_strength(const TipDeblockOptions& options) {
  TipDeblockStrength strength;
  strength.base_q_idx = options.base_q_idx;
  strength.delta_q_u_ac = options.delta_q_u_ac;
  strength.delta_q_v_ac = options.delta_q_v_ac;
  strength.base_uv_ac_delta_q = options.base_uv_ac_delta_q;
  strength.bit_depth = options.bit_depth;
  return strength;
}

}  // namespace

const char* tip_deblock_status_string(TipDeblockStatus status) {
  switch (status) {
    case TipDeblockStatus::kOk:
      return "ok";
    case TipDeblockStatus::kNotApplicable:
      return "not a TIP_FRAME_AS_OUTPUT frame with §7.16 enabled";
    case TipDeblockStatus::kInvalidArgument:
      return "invalid argument";
    case TipDeblockStatus::kUnsupportedBitDepth:
      return "unsupported bit depth (only 8-bit is implemented)";
    case TipDeblockStatus::kUnsupportedSubsampling:
      return "unsupported chroma subsampling (only 4:2:0 is implemented)";
    case TipDeblockStatus::kUnsupportedFlagCombination:
      return "apply_deblocking_filter_tip cannot be signalled for this frame";
  }
  return "unknown";
}

int tip_deblock_plane_width(const TipDeblockFrame& frame, int plane) {
  if (plane < 0 || plane > 2 || frame.mi_cols <= 0) {
    return 0;
  }
  return (frame.mi_cols * deblock::kMiSize) >> plane_shift_x(frame, plane);
}

int tip_deblock_plane_height(const TipDeblockFrame& frame, int plane) {
  if (plane < 0 || plane > 2 || frame.mi_rows <= 0) {
    return 0;
  }
  return (frame.mi_rows * deblock::kMiSize) >> plane_shift_y(frame, plane);
}

TipSize tip_deblock_tip_size(const TipDeblockOptions& options) {
  return select_tip_size(options.enable_tip_refinemv,
                         options.tip_interp_filter);
}

TipDeblockPlaneInfo tip_deblock_plane_info(const TipDeblockOptions& options,
                                           int plane) {
  if (plane < 0 || plane > 2) {
    return TipDeblockPlaneInfo{};
  }
  return derive_plane_info(plane, to_strength(options));
}

bool tip_deblock_flag_signallable(const TipDeblockOptions& options) {
  return options.tip_frame_mode == TipFrameMode::kAsOutput &&
         options.enable_df_sub_pu;
}

bool tip_deblock_is_active(const TipDeblockOptions& options) {
  return tip_deblock_flag_signallable(options) && options.allow_df_sub_pu &&
         options.apply_deblocking_filter_tip;
}

bool tip_deblock_supported(const TipDeblockFrame& frame,
                           const TipDeblockOptions& options,
                           TipDeblockStatus* status) {
  auto fail = [status](TipDeblockStatus value) {
    if (status != nullptr) {
      *status = value;
    }
    return false;
  };

  if (options.bit_depth != 8) {
    return fail(TipDeblockStatus::kUnsupportedBitDepth);
  }
  if (frame.subsampling_x != 1 || frame.subsampling_y != 1) {
    return fail(TipDeblockStatus::kUnsupportedSubsampling);
  }
  if (frame.mi_cols <= 0 || frame.mi_rows <= 0) {
    return fail(TipDeblockStatus::kInvalidArgument);
  }
  // An even MI grid keeps the 4:2:0 chroma grid integral and makes the 4-row /
  // 4-column §7.16 sample groups align with the plane extent.
  if ((frame.mi_cols & 1) != 0 || (frame.mi_rows & 1) != 0) {
    return fail(TipDeblockStatus::kInvalidArgument);
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (frame.planes[plane].data == nullptr) {
      return fail(TipDeblockStatus::kInvalidArgument);
    }
    const int width = tip_deblock_plane_width(frame, plane);
    const int height = tip_deblock_plane_height(frame, plane);
    if (height <= 0 || frame.planes[plane].stride < width) {
      return fail(TipDeblockStatus::kInvalidArgument);
    }
  }
  if ((frame.num_tile_cols > 0 && frame.tile_col_starts == nullptr) ||
      (frame.num_tile_rows > 0 && frame.tile_row_starts == nullptr)) {
    return fail(TipDeblockStatus::kInvalidArgument);
  }
  if (frame.num_tile_cols < 0 || frame.num_tile_rows < 0) {
    return fail(TipDeblockStatus::kInvalidArgument);
  }
  for (int i = 0; i < frame.num_tile_cols; ++i) {
    const int value = static_cast<int>(frame.tile_col_starts[i]);
    if (value < 0 || value >= frame.mi_cols) {
      return fail(TipDeblockStatus::kInvalidArgument);
    }
  }
  for (int i = 0; i < frame.num_tile_rows; ++i) {
    const int value = static_cast<int>(frame.tile_row_starts[i]);
    if (value < 0 || value >= frame.mi_rows) {
      return fail(TipDeblockStatus::kInvalidArgument);
    }
  }
  if (status != nullptr) {
    *status = TipDeblockStatus::kOk;
  }
  return true;
}

TipDeblockCoreFrame tip_deblock_to_core(const TipDeblockFrame& frame) {
  TipDeblockCoreFrame core{};
  for (int plane = 0; plane < 3; ++plane) {
    core.plane[plane] = frame.planes[plane].data;
    core.stride[plane] = frame.planes[plane].stride;
  }
  core.mi_cols = frame.mi_cols;
  core.mi_rows = frame.mi_rows;
  core.sub_x = frame.subsampling_x;
  core.sub_y = frame.subsampling_y;
  core.tile_col_starts = frame.tile_col_starts;
  core.num_tile_cols = frame.num_tile_cols;
  core.tile_row_starts = frame.tile_row_starts;
  core.num_tile_rows = frame.num_tile_rows;
  return core;
}

namespace {

// Shared entry validation: returns kOk with `active` set when the process must
// run, kNotApplicable when it legitimately does not, or an error status.
TipDeblockStatus check_invocation(const TipDeblockFrame& frame,
                                  const TipDeblockOptions& options,
                                  bool* active) {
  if (active != nullptr) {
    *active = false;
  }
  // apply_deblocking_filter_tip can only be coded after
  // enable_df_sub_pu && allow_df_sub_pu (and only in a TIP output frame).
  if (options.apply_deblocking_filter_tip &&
      !(tip_deblock_flag_signallable(options) && options.allow_df_sub_pu)) {
    return TipDeblockStatus::kUnsupportedFlagCombination;
  }
  TipDeblockStatus status = TipDeblockStatus::kOk;
  if (!tip_deblock_supported(frame, options, &status)) {
    return status;
  }
  if (active != nullptr) {
    *active = tip_deblock_is_active(options);
  }
  return TipDeblockStatus::kOk;
}

}  // namespace

TipDeblockStatus tip_deblock_frame(TipDeblockFrame& frame,
                                   const TipDeblockOptions& options) {
  bool active = false;
  const TipDeblockStatus status = check_invocation(frame, options, &active);
  if (status != TipDeblockStatus::kOk) {
    return status;
  }
  if (!active) {
    return TipDeblockStatus::kNotApplicable;
  }
  const TipDeblockCoreFrame core = tip_deblock_to_core(frame);
  tip_deblock_frame_core(core, tip_deblock_tip_size(options),
                         to_strength(options),
                         options.disable_loopfilters_across_tiles);
  return TipDeblockStatus::kOk;
}

TipDeblockStatus tip_deblock_plane_pass(TipDeblockFrame& frame,
                                        const TipDeblockOptions& options,
                                        int plane, int pass) {
  if (plane < 0 || plane > 2 || pass < 0 || pass > 1) {
    return TipDeblockStatus::kInvalidArgument;
  }
  bool active = false;
  const TipDeblockStatus status = check_invocation(frame, options, &active);
  if (status != TipDeblockStatus::kOk) {
    return status;
  }
  if (!active) {
    return TipDeblockStatus::kNotApplicable;
  }
  const TipDeblockCoreFrame core = tip_deblock_to_core(frame);
  tip_deblock_plane_pass_core(core, tip_deblock_tip_size(options),
                              to_strength(options), plane, pass,
                              options.disable_loopfilters_across_tiles);
  return TipDeblockStatus::kOk;
}

TipDeblockEdgeInfo tip_deblock_inspect_edge(const TipDeblockFrame& frame,
                                            const TipDeblockOptions& options,
                                            int plane, int pass, int x,
                                            int y) {
  TipDeblockEdgeInfo info{};
  TipDeblockStatus status = TipDeblockStatus::kOk;
  if (!tip_deblock_supported(frame, options, &status)) {
    return info;
  }
  const TipDeblockCoreFrame core = tip_deblock_to_core(frame);
  return tip_edge_core<false>(core, tip_deblock_tip_size(options),
                              to_strength(options), plane, pass, x, y,
                              options.disable_loopfilters_across_tiles);
}

}  // namespace av2enc::tip_deblock
