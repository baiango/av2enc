// AV2 §7.16 Deblocking filter for TIP — CPU reference API.
//
// Scope: 8-bit 4:2:0 planar, scalar, single-threaded (correctness oracle).
//
// ===========================================================================
// Where this runs relative to the regular deblocking filter (§7.17)
// ===========================================================================
// They are alternatives, never a pipeline.  Per §7.2 decode_frame_wrapup:
//
//   if (TipFrameMode == TIP_FRAME_AS_OUTPUT) {
//     if (apply_deblocking_filter_tip) §7.16;      // this component
//     LrFrame = CurrFrame;
//     reference_frame_update();                    // no §7.17/§7.18/§7.19/§7.20
//   } else if (bru_inactive) {
//     ...
//   } else if (!ShowExistingFrame) {
//     §7.17;  §7.18;  §7.19;  §7.20;
//   }
//
// Consequences for an encoder:
//   * A TIP_FRAME_AS_OUTPUT frame runs §7.16 only, and the §7.16 output is
//     what is stored as reference and emitted for display.  Do not run §7.17
//     on it, and do not run CDEF/CCSO/LR (frame_filters_on[] is forced to 0
//     for these frames).
//   * A TIP_FRAME_AS_REF frame (TIP blocks inside an ordinary inter frame) is
//     a normal reconstructed inter frame: §7.16 does not run at all; the
//     regular §7.17 -> §7.18 -> §7.19 -> §7.20 chain does.
//   * SEF/bridge/BRU-inactive frames run neither here.
//
// Call-site rule: branch on TipFrameMode *before* the §7.17 call.  If the
// frame is TIP_FRAME_AS_OUTPUT, call tip_deblock_frame() instead of
// deblock::deblock_frame(); otherwise call deblock::deblock_frame() and never
// this component.
//
// ===========================================================================
// §5.18.2 flag interaction (enable_df_sub_pu)
// ===========================================================================
// apply_deblocking_filter_tip is signalled only for TIP_FRAME_AS_OUTPUT and
// only inside `if (enable_df_sub_pu) { allow_df_sub_pu; if (allow_df_sub_pu)
// apply_deblocking_filter_tip; else apply_deblocking_filter_tip = 0; }`.
//
// Therefore, to emit a TIP output frame with §7.16 enabled the encoder must:
//   1. set sequence-level enable_df_sub_pu = 1,
//   2. set frame-level allow_df_sub_pu = 1 in the TIP frame header,
//   3. set apply_deblocking_filter_tip = 1.
// §7.16 itself does not use sub-PU edges: allow_df_sub_pu is a syntax
// precondition, not a filtering input.  A caller that sets
// apply_deblocking_filter_tip = 1 without (1) and (2) is rejected with
// kUnsupportedFlagCombination because that bit cannot be coded.
//
// ===========================================================================
// Quantizer state
// ===========================================================================
// For a TIP output frame, base_q_idx / DeltaQUAc / DeltaQVAc come from the
// TIP frame header when enable_tip_explicit_qp == 1; otherwise §5.18.2 sets
// them to Round2(RefX[slot0] + RefX[slot1], 1) over the closest past/future
// references.  BaseUVAcDeltaQ is the sequence-level value.  §7.16 does not
// consult segment ids, per-cell qindex, LosslessArray, Skips, TX sizes or
// sub-PU bases, so this API has no MI grid at all.

#ifndef AV2ENC_ENCODER_FILTERS_TIP_DEBLOCK_TIP_DEBLOCK_H_
#define AV2ENC_ENCODER_FILTERS_TIP_DEBLOCK_TIP_DEBLOCK_H_

#include <cstdint>

#include "encoder/filters/tip_deblock/tip_deblock_core.h"

namespace av2enc::tip_deblock {

using inter::temporal::TipFrameMode;

// A mutable reconstructed plane.  `stride` is in samples.  For 4:2:0 the
// required visible extent is
//   (mi_cols * 4) >> sub_x  by  (mi_rows * 4) >> sub_y,
// i.e. the frame rounded up to the MI grid, matching §7.16's planeWidth and
// planeHeight.  Buffers may be padded further; §7.16 never writes those guard
// samples.
struct TipDeblockPlane {
  std::uint8_t* data = nullptr;
  int stride = 0;
};

struct TipDeblockFrame {
  TipDeblockPlane planes[3];
  int mi_cols = 0;
  int mi_rows = 0;
  int subsampling_x = 1;
  int subsampling_y = 1;
  // Tile starts in luma MI units (MiColStarts / MiRowStarts).
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

enum class TipDeblockStatus {
  kOk = 0,
  // The process is not invoked for this frame: not a TIP_FRAME_AS_OUTPUT frame,
  // or apply_deblocking_filter_tip == 0.  Nothing was modified.
  kNotApplicable,
  kInvalidArgument,
  kUnsupportedBitDepth,
  kUnsupportedSubsampling,
  // apply_deblocking_filter_tip == 1 while the bit cannot be signalled
  // (TipFrameMode != TIP_FRAME_AS_OUTPUT, or enable_df_sub_pu/allow_df_sub_pu
  // is 0).  Nothing was modified.
  kUnsupportedFlagCombination,
};

const char* tip_deblock_status_string(TipDeblockStatus status);

// Frame-level §5.18.2 / §5.4.10 / §5.18.7 state needed by §7.16.
struct TipDeblockOptions {
  // §5.18.2 frame header state.
  TipFrameMode tip_frame_mode = TipFrameMode::kDisabled;
  bool apply_deblocking_filter_tip = false;
  // Sequence-level enable_df_sub_pu and frame-level allow_df_sub_pu.  Both
  // must be set for apply_deblocking_filter_tip to be signallable.
  bool enable_df_sub_pu = false;
  bool allow_df_sub_pu = false;
  // §5.4.10: skip filtering across tile boundaries.
  bool disable_loopfilters_across_tiles = false;
  // Quantizer state (see the header comment).
  int base_q_idx = 0;
  int delta_q_u_ac = 0;
  int delta_q_v_ac = 0;
  int base_uv_ac_delta_q = 0;
  int bit_depth = 8;
  // §7.16 tipSize inputs.
  bool enable_tip_refinemv = false;
  InterpFilter tip_interp_filter = InterpFilter::kEighttapSharp;
};

// Required plane extents, in samples, derived from the MI grid.
int tip_deblock_plane_width(const TipDeblockFrame& frame, int plane);
int tip_deblock_plane_height(const TipDeblockFrame& frame, int plane);

// §7.16 `tipSize` for this frame's options.
TipSize tip_deblock_tip_size(const TipDeblockOptions& options);

// §7.16 per-plane strength derivation, independent of whether the frame-level
// flag is set.  Returns a zeroed record for an invalid plane.
TipDeblockPlaneInfo tip_deblock_plane_info(const TipDeblockOptions& options,
                                           int plane);

// True when the bitstream state makes apply_deblocking_filter_tip signallable:
// TipFrameMode == TIP_FRAME_AS_OUTPUT && enable_df_sub_pu.
bool tip_deblock_flag_signallable(const TipDeblockOptions& options);

// True when §7.16 must be invoked: signallable && allow_df_sub_pu &&
// apply_deblocking_filter_tip.
bool tip_deblock_is_active(const TipDeblockOptions& options);

// Validates the frame/options and applies §7.16 in place.
TipDeblockStatus tip_deblock_frame(TipDeblockFrame& frame,
                                   const TipDeblockOptions& options);

// One plane/pass only (vertical pass 0 or horizontal pass 1).  Provided for
// integration staging and for tests that isolate a pass.  Returns
// kNotApplicable exactly like tip_deblock_frame() when §7.16 is not active.
TipDeblockStatus tip_deblock_plane_pass(TipDeblockFrame& frame,
                                        const TipDeblockOptions& options,
                                        int plane, int pass);

// Runs the full §7.16 decision path for one edge without writing pixels.
// `x`/`y` are plane coordinates and must be a position the §7.16 loops visit
// (vertical: x a positive multiple of sw, y a multiple of 4; horizontal: y a
// positive multiple of sh, x a multiple of 4).  Returns a zeroed record
// (on_screen == 0) otherwise or for an unsupported frame.
TipDeblockEdgeInfo tip_deblock_inspect_edge(const TipDeblockFrame& frame,
                                            const TipDeblockOptions& options,
                                            int plane, int pass, int x, int y);

// True when `options` + plane geometry are inside the supported subset.
bool tip_deblock_supported(const TipDeblockFrame& frame,
                           const TipDeblockOptions& options,
                           TipDeblockStatus* status);

// Converts the public frame description to the core form.
TipDeblockCoreFrame tip_deblock_to_core(const TipDeblockFrame& frame);

}  // namespace av2enc::tip_deblock

#endif  // AV2ENC_ENCODER_FILTERS_TIP_DEBLOCK_TIP_DEBLOCK_H_
