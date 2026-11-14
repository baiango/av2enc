#pragma once

// Encoder-side global-motion planning for the M2 video path (§5.18.9,
// §7.12.2, §7.13.3.23).
//
// The component in `src/codec/inter/gme/` owns the model fitting; the CUDA
// module in `src/cuda/codec/inter/gme/` ported the same integer core.  This
// file is the product bridge: it builds a deterministic integer ME grid over
// the current frame, turns the per-block MVs into correspondences, and turns
// the estimates into a `GmFramePlan` that both the CPU writer and the CUDA
// pipeline use.
//
// Collect and finalise are separate so the CUDA product path can run the fit
// on the device (the GPU backend's `gme_estimate()`) while the CPU path calls
// `inter::gme::estimate_gm_block_mvs()` directly.  Both consume the same
// correspondence vector, so the two paths are decision-identical.

#include <span>
#include <string>
#include <vector>

#include "codec/inter/gme/gme.h"
#include "codec/inter/mc/mc.h"
#include "pipeline/video/video_frame.h"
#include "pipeline/video/video_gm_types.h"

namespace av2enc::video {

// Deterministic 16x16-luma block-MV grid, one entry per reference.  Blocks are
// searched with the same integer+quarter-pel ME core the encoder uses, so the
// correspondences are identical on every platform.  Returns an empty vector
// for a reference with no valid blocks.
std::vector<std::vector<inter::gme::GmBlockMv>> collect_gm_block_mvs(
    const VideoInputFrame& source,
    std::span<const inter::mc::RefFrame> references,
    const VideoOptions& options, std::string& error);

// Options used for every fit (model, outlier reject, frame geometry).
inter::gme::GmOptions make_gm_options(const VideoInputFrame& source,
                                      const VideoOptions& options);

// Builds the per-frame plan from the per-reference estimates.  References at
// zero distance are forced to IDENTITY because §5.18.9.1 codes no model for
// them.  `use_global_motion` is set when at least one accepted model is
// non-identity.
GmFramePlan make_gm_frame_plan(
    std::span<const inter::gme::GmEstimate> estimates,
    std::span<const int> ref_order_hint,
    std::span<const std::uint8_t> ref_frame_idx,
    std::span<const int> frame_distance);

// CPU reference: collect + fit + plan for one frame.
bool estimate_gm_plan_cpu(
    const VideoInputFrame& source,
    std::span<const inter::mc::RefFrame> references,
    std::span<const int> ref_order_hint,
    std::span<const std::uint8_t> ref_frame_idx,
    std::span<const int> frame_distance, const VideoOptions& options,
    GmFramePlan& plan, std::string& error);

// ---------------------------------------------------------------------------
// §7.22 per-8x8 warp projection
// ---------------------------------------------------------------------------
//
// The §7.12.2.2 block-centre projection (`get_warp_motion_vector`) is not what
// the §7.22 store uses: the decoder projects the model at the centre of every
// 8x8 sub-block of a warped block.  `gm_model_uses_warp` reports the decoder's
// derived `TransformationType > TRANSLATION` (a model with identity linear
// part is stored as TRANSLATION and never warped), and
// `gm_sub_block_warp_mv` reproduces its `get_sub_block_warp_mv` for the 8x8
// whose top-left luma sample is (pixel_x, pixel_y) in MI/coded coordinates.

// True when the §5.18.9-quantised model derives a warp type (ROTZOOM/AFFINE).
bool gm_model_uses_warp(const inter::gme::GmParams& params);

// §7.22 per-8x8 warped MV (1/8 luma samples), evaluated at the 8x8 centre.
void gm_sub_block_warp_mv(const inter::gme::GmParams& params, int pixel_x,
                          int pixel_y, inter::mc::Mv* mv);

// Unit check for the §7.22 warp store: model classification (identity /
// translation / warp / invalid shear) and the per-8x8 projection against
// hand-computed values.  Returns false and fills `error` on a mismatch.
bool ref_mvs_field_selftest(std::string* error);

}  // namespace av2enc::video
