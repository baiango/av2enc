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

// Per-block switchable subpel filter (§5.18.5.1 + §5.20.7.6).  The frame
// header signs is_filter_switchable = 1 and every eligible inter leaf codes a
// TileInterpFilterCdf symbol; the ordinary §5.18.5.1 frame-global form stays
// the default.  This is a CPU experiment and is off unless explicit.
//
// The per-leaf decision is a perceptual RD rule: the filter that minimises
//   plain_Sse + w * (perceptual_Sse - plain_Sse) + lambda * (syntax + filter)
// wins, where the perceptual term is the AV2ENC_EFF_PERC distortion (source
// weight map + error-domain CSF) and `w` is AV2ENC_INTERP_BLOCK_CSF.  The
// weight is the sweep knob: w = 1 is the same D the leaf/TX search uses, and
// larger values price the low/mid-band error more aggressively.
inline bool interp_block_enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_INTERP_BLOCK");
    if (env == nullptr || *env == '\0') return false;
    return !(env[0] == '0' && env[1] == '\0');
  }();
  return value;
}

inline double interp_block_csf_weight() {
  static const double value = [] {
    const char* env = std::getenv("AV2ENC_INTERP_BLOCK_CSF");
    if (env == nullptr || *env == '\0') return 1.0;
    char* end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || parsed < 0.0) return 1.0;
    return parsed;
  }();
  return value;
}

// Current per-leaf prediction filter.  The decision path sets the override
// while it evaluates one filter hypothesis; MC then reads it through
// leaf_interp_filter().  -1 means "no override": the frame-global filter is
// used (EIGHTTAP when the switchable gate is on, because the frame header
// codes no frame-level filter value in that form).
inline int& interp_block_filter_override() {
  static thread_local int value = -1;
  return value;
}

inline inter::mc::InterpFilter leaf_interp_filter() {
  if (!interp_block_enabled()) return frame_interp_filter();
  const int filter = interp_block_filter_override();
  return static_cast<inter::mc::InterpFilter>(
      filter >= 0 && filter <= 2 ? filter : 0);
}

}  // namespace av2enc::video
