// Umbrella header for the M2 post-processing / rate-control layer.
//
// Nothing in this directory edits the M2 core.  It exposes:
//   * per-frame CQP/CRF/VBR decisions with the M2 coefficient-CDF constraint;
//   * multi-tile frame headers and tile groups built from
//     src/encoder/tiles/ geometry and the verified src/encoder/pipeline/ rules;
//   * the post-reconstruction deblock -> CDEF -> CCSO chain;
//   * a 10-bit geometry/header checklist (no 12-bit);
//   * deterministic tests and a dav2d tiled two-frame gate.
#pragma once

#include "pipeline/video/post/bit_depth_checklist.h"
#include "pipeline/video/post/loop_filters.h"
#include "pipeline/video/post/rc_adapter.h"
#include "pipeline/video/post/tiled_frame.h"
