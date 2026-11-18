// Frame-level subpel interpolation filter (§5.18.5.1 / §7.13.3.14).
//
// The historical stream signs `is_filter_switchable = 0` with
// `interpolation_filter = EIGHTTAP` and predicts with the regular 8-tap
// kernel.  The MC component and the frame-header writer already implement
// SMOOTH/SHARP/BILINEAR; this gate exposes the frame-level choice so the
// encoder can measure it.  No new syntax: only the existing two header bits
// and the existing MC kernels change.
//
// Default (unset, 0) is EIGHTTAP, so every pin and every historical stream is
// byte-identical.  Values are the §5.18.5.1 codes:
//   0 EIGHTTAP, 1 EIGHTTAP_SMOOTH, 2 EIGHTTAP_SHARP, 3 BILINEAR.
//
// This is a CPU-experiment gate.  The CUDA backend keeps EIGHTTAP until the
// filter is threaded through the device kernels and parity is proven.
#pragma once

#include <cstdlib>

#include "codec/inter/mc/mc_types.h"

namespace av2enc::video {

inline int frame_interp_filter_index() {
  const char* value = std::getenv("AV2ENC_INTERP_FILTER");
  if (value == nullptr || *value == '\0') return 0;
  const int filter = std::atoi(value);
  if (filter < 0 || filter > 3) return 0;
  return filter;
}

inline inter::mc::InterpFilter frame_interp_filter() {
  return static_cast<inter::mc::InterpFilter>(frame_interp_filter_index());
}

}  // namespace av2enc::video
