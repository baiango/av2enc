// Future-error (drift) surcharge for the inter leaf skip decision.
//
// The leaf RD in video_frame_decision.cpp prices a skip leaf with this
// frame's prediction SSE only.  A skipped leaf also installs that prediction
// error into the reference, where the following frames copy it through
// subpel MC.  On fractional-motion content the nightly leaf RD therefore
// skips 76-91% of inter leaves and the P chain loses 2.3-5.8 dB against its
// own key (docs/tool-quality.md §4.1); disabling --video-leaf-rd recovers
// +4.1 K on the probe and +2.3 K on the natural 240p ladder.
//
// This gate adds the missing future term to the skip cost only:
//
//   J_skip = D_pred * (1 + alpha * mv_factor) + lambda * R_skip
//
// D_pred is the prediction SSE the skip would freeze; mv_factor is 1 for the
// default and AV2ENC_SKIP_DRIFT_NZMV for leaves whose chosen MV is non-zero
// (the zero-MV skip is an exact copy, so it must stay free on static
// content).  Off by default: gate off leaves every decision and pin
// byte-identical.
#ifndef AV2ENC_PIPELINE_VIDEO_SKIP_DRIFT_H_
#define AV2ENC_PIPELINE_VIDEO_SKIP_DRIFT_H_

#include <cstdlib>

namespace av2enc::video::skip_drift {

inline bool parse_off(const char* value) {
  if (value == nullptr) return true;
  if (value[0] == '\0' || value[0] == '0') return true;
  if (value[0] == 'n' || value[0] == 'N') return true;
  return false;
}

inline double env_double(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value) return fallback;
  return parsed;
}

inline bool enabled() {
  static const bool value = !parse_off(std::getenv("AV2ENC_SKIP_DRIFT"));
  return value;
}

// Weight of the frozen prediction error in the skip cost. 1.0 is "charge the
// full committed error once"; the measured horizon is a small multiple.
inline double alpha() {
  static const double value = env_double("AV2ENC_SKIP_DRIFT_ALPHA", 4.0);
  return value;
}

// Extra factor for a non-zero-MV skip.  The zero-MV skip is an exact copy of
// the reference, so its frozen error does not accumulate through MC.
inline double nz_mv_factor() {
  static const double value = env_double("AV2ENC_SKIP_DRIFT_NZMV", 1.0);
  return value;
}

// Factor for a zero-MV skip.  Default 0.0 restricts the surcharge to the
// subpel-MC case that docs/tool-quality.md §4.1 identifies as the drift
// mechanism; 1.0 charges every skip.
inline double zero_mv_factor() {
  static const double value = env_double("AV2ENC_SKIP_DRIFT_ZEROMV", 0.0);
  return value;
}

// AV2ENC_SKIP_DRIFT_FADE: a zero-MV skip on an illumination ramp freezes the
// signed DC error in the reference just as a subpel skip freezes motion
// error, but the zero-MV factor keeps it free so exact static content stays
// skip-only.  The evidence is the leaf's mean signed prediction error: a fade
// leaves a bias, exact-static content and sensor noise do not.
inline bool fade_enabled() {
  const char* value = std::getenv("AV2ENC_SKIP_DRIFT_FADE");
  if (value == nullptr) return false;
  if (value[0] == '\0' || value[0] == '0') return false;
  if (value[0] == 'n' || value[0] == 'N') return false;
  return true;
}

inline double fade_threshold() {
  static const double value =
      env_double("AV2ENC_SKIP_DRIFT_FADE_T", 1.0);
  return value;
}

// AV2ENC_SKIP_DRIFT_ILLUM: frame-level illumination-drift latch.  The
// per-leaf fade test above only fires on a leaf whose *mean* error exceeds 1
// LSB, so a fade leaf whose local mean sits near the fade pivot shows no bias
// even though its texture is being rescaled, and it stays skip-only.  The
// frame-level evidence is cheaper and stronger: on a fade the source's luma
// mean and contrast both move every frame (the 240p clip moves ~1.0 LSB of
// mean and ~0.9 LSB of std per tail frame), while exact-static and
// sensor-noise content move neither (~0.01).  When the frame's own source
// statistics moved by at least the threshold, every zero-MV skip in that
// frame pays the same future-error surcharge a subpel skip pays.
inline bool illum_enabled() {
  const char* value = std::getenv("AV2ENC_SKIP_DRIFT_ILLUM");
  if (value == nullptr) return false;
  if (value[0] == '\0' || value[0] == '0') return false;
  if (value[0] == 'n' || value[0] == 'N') return false;
  return true;
}

// LSB of |mean delta| + |std delta| between consecutive source frames.
inline double illum_threshold() {
  static const double value =
      env_double("AV2ENC_SKIP_DRIFT_ILLUM_T", 0.35);
  return value;
}

// AV2ENC_SKIP_DRIFT_REFDEF: frame-level reference-deficit latch.  The fade and
// illumination gates above fire on a *known* drift cause; a P chain also adds
// error with no such signature (subpel MC on a moving reference, BAWP, loop
// filters).  The direct evidence is the measured coding error of the picture
// this frame predicts from: compare the primary reference's stored
// reconstruction against the source picture it was coded from, and the key's
// own reconstruction against its source as the anchor.  When the reference has
// lost at least the threshold against that anchor, the chain itself has added
// error beyond the key's deliberate QP choice, and a zero-MV skip would freeze
// it.  Exact-repeat content measures zero against the anchor; sensor-noise
// content measures the source noise floor (~1.3 dB at QP139, below the
// default threshold), so both stay skip-only.
inline bool refdef_enabled() {
  const char* value = std::getenv("AV2ENC_SKIP_DRIFT_REFDEF");
  if (value == nullptr) return false;
  if (value[0] == '\0' || value[0] == '0') return false;
  if (value[0] == 'n' || value[0] == 'N') return false;
  return true;
}

// dB of reference coding loss against the key anchor that arms the latch.
// 1.5 dB is above the near_static noise floor (1.27-1.32 dB at QP139) and
// below the natural head's first real drift (1.63 dB at f5), so exact/noise
// content stays untouched.
inline double refdef_threshold() {
  static const double value =
      env_double("AV2ENC_SKIP_DRIFT_REFDEF_T", 1.5);
  return value;
}

}  // namespace av2enc::video::skip_drift

#endif  // AV2ENC_PIPELINE_VIDEO_SKIP_DRIFT_H_
