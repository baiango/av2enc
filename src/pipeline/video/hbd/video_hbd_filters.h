// §7.17-§7.20.5 in-loop filters for the format video core.
//
// The pixel arithmetic is the same depth-parameterised core the 10-bit still
// writer and the 8-bit chroma still writer use.  This module builds the
// frame-level `post::VideoFilterPlan` (the structure the shared M2 syntax
// writer already consumes) and applies the filter chain to the reconstructed
// uint16_t planes.
//
// Scope: deblock runs on every frame (including P frames; the metadata grid
// carries each leaf's skip flag).  CDEF/CCSO/LR/GDF are planned on the intra
// key frame only.  Every frame of the format video core is reconstructed in
// full before the tile payload is assembled, so the plan can be made from the
// completed reconstruction exactly like the still writer.  Inter frames signal
// those tools disabled (cdef_frame_enable = 0 / frame_restoration_type = NONE),
// which is why their skipped leaves need no CDEF/LR skip grid.
#ifndef AV2ENC_PIPELINE_VIDEO_HBD_VIDEO_HBD_FILTERS_H_
#define AV2ENC_PIPELINE_VIDEO_HBD_VIDEO_HBD_FILTERS_H_

#include <cstdint>
#include <span>
#include <string>

#include "formats/format.h"
#include "pipeline/video/hbd/video_hbd.h"
#include "pipeline/video/post/video_filter_plan.h"
#include "pipeline/video/post/video_loop_filters.h"
#include "pipeline/video/video_syntax.h"

namespace av2enc::video::hbd {

// Plans and applies deblock -> CDEF -> CCSO -> LR -> GDF to the reconstructed
// MI-aligned key frame and fills `plan` for the frame header and tile writer.
//
//   recon[p]        coded plane, `recon_stride[p]` samples per row
//   source          visible source picture (SSE-driven decisions)
//   blocks          the key frame's coded leaves (one entry per partition
//                   leaf) with per-plane tx sizes and effective qindex
//
// Returns false with a precise reason and leaves the planes unspecified on
// failure.  `plan` is all-disabled when every requested tool decides to do
// nothing, which is what makes a no-filter stream byte-identical.
bool apply_frame_filters(
    const fmt::Format& format, int mi_cols, int mi_rows,
    const int visible_w[3], const int visible_h[3],
    std::uint16_t* const recon[3], const int recon_stride[3],
    const FramePlanes& source, std::span<const syntax::BlockSyntax> blocks,
    int base_q_idx, bool frame_is_intra, bool enable_deblock,
    const post::VideoLoopFilterOptions& options,
    post::VideoFilterPlan* plan, std::string& error);

}  // namespace av2enc::video::hbd

#endif  // AV2ENC_PIPELINE_VIDEO_HBD_VIDEO_HBD_FILTERS_H_
