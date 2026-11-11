#ifndef AV2ENC_INTER_ADVANCED_WARP_H_
#define AV2ENC_INTER_ADVANCED_WARP_H_

// CPU reference for AV2 v1.0.0 warped motion compensation (§7.13.3.19–.24).
//
// This module predicts a block from a reference plane using a 6-parameter affine
// model. It does not run motion search: the caller supplies the model (global
// motion, a delta from the neighbouring model, or the output of
// `estimate_local_warp()`), and the reference plane.
//
// Two predictors exist in the spec:
//   * §7.13.3.19 block warp: 8x8 sections, per-section phase -> true affine.
//   * §7.13.3.20 extended warp: 4x4 sections, one translation per section.
//     Used when `!shearValid || w < 8 || h < 8 || is_scaled(ref)`, and as the
//     EXTENDWARP motion mode.
//
// Output precision follows §7.13.3.14: a single-reference prediction is rounded
// to integer samples (`output >> 11`), while a compound prediction stays at
// 1/16 sample precision (`>> 7`). The destination is therefore `int16_t`:
//
//   * compound == false: values are 0..255 (clip1 applied);
//   * compound == true : values are `sample << 4` and must be fed to
//                        `run_compound_job()` as `Preds[]`.
//
// Integration sketch:
//
//   adv::WarpParams params = adv::estimate_local_warp(samples, count, mid_x,
//                                                    mid_y, block_mv);
//   adv::ShearParams shear = adv::setup_shear(params);
//   adv::WarpBlockGeometry geom = ...;
//   if (adv::warp_needs_extended(params, geom)) {
//     adv::warp_predict_ext(params, ref, geom, compound, preds, stride);
//   } else {
//     adv::warp_predict(params, ref, geom, compound, preds, stride);
//   }
//
// `ref.max_x`/`ref.max_y` must be the spec's padded bounds
// `(RefMiCols * MI_SIZE >> subX) - 1`, not the visible crop.

#include <cstdint>
#include <optional>

#include "warp_common.h"

namespace av2enc::inter::adv {

// Geometry needed by the warp paths. The `rect` origin is the block position in
// the frame (used by the affine model); `dst_x`/`dst_y` say where the same block
// is written in the destination buffer, so a block-local prediction buffer is
// fine.
struct WarpBlockGeometry {
  int plane = 0;  // 0 = luma; >0 selects the chroma subsampling in the model
  int sub_x = 0;
  int sub_y = 0;
  BlockRect rect{};
  int dst_x = 0;
  int dst_y = 0;
  // Motion-vector clamp context (mi units, 4 luma samples each).
  int mi_col = 0;
  int mi_row = 0;
  int bw4 = 1;
  int bh4 = 1;
  int mi_cols = 0;
  int mi_rows = 0;
  // Reference/frame dimensions for the scaled-reference path.
  int ref_w = 0;
  int ref_h = 0;
  int frame_w = 0;
  int frame_h = 0;
  int bit_depth = 8;
};

// True when §7.13.3.14 must fall back from block warp to extended warp.
bool warp_needs_extended(const WarpParams& params, const WarpBlockGeometry& geom);

// §7.13.3.19 over a whole block: 8x8 sections.
bool warp_predict(const WarpParams& params, const PlaneView& ref, const WarpBlockGeometry& geom,
                  bool compound, std::int16_t* dst, int dst_stride);

// §7.13.3.20 over a whole block: 4x4 sections of one translation each.
bool warp_predict_ext(const WarpParams& params, const PlaneView& ref, const WarpBlockGeometry& geom,
                      bool compound, std::int16_t* dst, int dst_stride);

// Convenience for the single-reference path: writes clipped uint8 samples into
// `dst` at (geom.dst_x, geom.dst_y).
bool warp_predict_u8(const WarpParams& params, const PlaneView& ref, const WarpBlockGeometry& geom,
                     const MutablePlaneView& dst);

// Encoder-side: estimate the local warp model from candidate correspondences.
// `mid_x`/`mid_y` are the block centre in luma samples and `block_mv` the
// block's own MV in 1/8 sample units. Returns std::nullopt if the sample count
// is out of range (0..LEAST_SQUARES_SAMPLES_MAX).
std::optional<WarpParams> estimate_local_warp(const WarpSample* samples, int count, int mid_x,
                                              int mid_y, const int block_mv[2]);

// Convenience: build a Q16 affine model (encoder/test helper only; the
// bitstream signals warp_delta_param, not this form).
WarpParams make_affine_params(int m00_q16, int m01_q16, int m10_q16, int m11_q16, int trans_row_q16,
                              int trans_col_q16);

// SSE between a warped prediction and the source block. `scratch` is block-local
// (geom.rect.w x geom.rect.h). For `compound == true` the 1/16-precision
// prediction is rounded to an integer sample before comparison.
// Returns -1 on invalid input. Encoder-side helper; RDO owns lambda.
std::int64_t warp_block_sse(const WarpParams& params, const PlaneView& ref,
                            const WarpBlockGeometry& geom, bool compound,
                            const std::uint8_t* source, int source_stride, std::int16_t* scratch,
                            int scratch_stride);

// True if the reference is scaled relative to the current frame.
bool warp_ref_is_scaled(const WarpBlockGeometry& geom);

}  // namespace av2enc::inter::adv

#endif  // AV2ENC_INTER_ADVANCED_WARP_H_
