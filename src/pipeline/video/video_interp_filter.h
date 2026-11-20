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

#include <cstdint>
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

// AV2ENC_INTERP_BLOCK_SKIP: keep the per-block switchable filter from changing
// the leaf's skip/residual decision.  A switchable filter is a prediction
// refinement: when a leaf codes residual the filter error is corrected in the
// same picture, but a skip leaf installs the filtered prediction into the
// reference and the following frames copy it through MC (the drift mechanism
// in docs/luma-audit.md §4.5).  The per-block RD sees only this frame's
// distortion, so a filter that makes the skip copy look better can still
// reshape the reference and cost the tail.
//
//   0 (default)  the historical rule: compare all three filters and let the
//                winner decide skip/residual, as measured in
//                docs/interp-filter.md.
//   1            EIGHTTAP keeps its own skip/residual decision; a non-zero
//                filter can refine a leaf that already codes residual, but it
//                cannot turn an EIGHTTAP skip into a filtered skip (or an
//                EIGHTTAP residual into a filtered skip).
//   2            the same guard only for a non-zero-MV skip, whose subpel MC
//                is the measured drift path; a zero-MV skip may still choose
//                a non-zero filter.
inline int interp_block_skip_guard() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_INTERP_BLOCK_SKIP");
    if (env == nullptr || *env == '\0' || env[0] == '0') return 0;
    if (env[0] == 'n' || env[0] == 'N') return 0;
    const int parsed = std::atoi(env);
    return parsed >= 1 && parsed <= 2 ? parsed : 1;
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

// Integer-MV (zero displacement) prediction copy used by the lane-2
// frame-horizon pipeline (src/pipeline/video/video_pipeline_horizon.cpp).  The
// tool is a pure prediction primitive: it copies one reference rectangle into
// `out` with the reference extent clamp the decoder's MC applies, so the
// horizon driver can price skip/residual candidates without entering the
// existing P-frame decision path.  Defined in video_frame_me.cpp.
void horizon_predict_zero_mv(const std::uint8_t* ref, int ref_stride,
                             int ref_width, int ref_height, int x0, int y0,
                             int w, int h, std::uint8_t* out);

// Lane-2 frame-horizon MC primitive: the normative §7.13.3 prediction of one
// 4:2:0 block at an integer-pel displacement.  `mv_y`/`mv_x` are in 1/8 luma
// samples (the horizon search only produces multiples of 8).  The luma phase is
// then 0 and an 8-tap convolution is an exact copy, while the chroma planes run
// the half-pel phase the decoder derives for an integer luma displacement.  The
// primitive calls the shared `inter::mc` kernel instead of copying by hand, so
// the horizon reconstruction stays decoder-exact for a non-zero MV.
//
// `ref` planes are the previous reconstruction at its MI-aligned extent
// (4 * MiCols x 4 * MiRows).  `out[0]` receives w*h luma samples, `out[1]` and
// `out[2]` receive (w/2)*(h/2) chroma samples each; all are tightly packed (the
// destination stride is w or w/2).  Returns false when the rectangle or the
// plane extent is not representable.
struct HorizonRefPlanes {
  const std::uint8_t* data[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};
  int width = 0;   // luma MI-aligned width (4 * MiCols)
  int height = 0;  // luma MI-aligned height (4 * MiRows)
};

bool horizon_predict_mv(const HorizonRefPlanes& ref, int x0, int y0, int w,
                        int h, int mv_y, int mv_x, std::uint8_t* out[3]);

}  // namespace av2enc::video
