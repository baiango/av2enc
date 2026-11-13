#pragma once

// Per-frame global-motion state shared by the CPU video writer and the CUDA
// product pipeline.
//
// §5.18.9 codes one model per explicit reference list entry.  The encoder
// estimates the model in current-frame -> reference-frame coordinates and
// stores the final §5.18.9-quantised parameters here.  The same struct is
// carried into the frame header so `global_motion_params()` sees exactly the
// model the block decision used.
//
// A default-constructed plan is the historical stream: no model, every
// reference IDENTITY, `use_global_motion = 0`.  Nothing in this header writes
// bits; `video_headers.cpp` consumes the plan.

#include <array>
#include <cstdint>

#include "codec/inter/gme/gme.h"

namespace av2enc::video {

// REFS_PER_FRAME (§9.2) is 7.
inline constexpr int kGmMaxRefs = 7;

struct GmFramePlan {
  // §5.18.9.1 use_global_motion.  True when at least one reference carries an
  // accepted non-identity model.
  bool use_global_motion = false;
  int num_refs = 0;
  std::array<inter::gme::GmParams, kGmMaxRefs> params{};
  // Absolute OrderHints[ref] for each reference list entry.  The syntax writer
  // derives the per-reference distance from `order_hint - ref_order_hint`.
  std::array<int, kGmMaxRefs> ref_order_hint{};
  // §5.18.2 ref_frame_idx[] slots, for the base-model history lookup.
  std::array<std::uint8_t, kGmMaxRefs> ref_frame_idx{};
  // get_relative_dist(order_hint, ref_order_hint) for each entry.
  std::array<int, kGmMaxRefs> frame_distance{};
};

}  // namespace av2enc::video
