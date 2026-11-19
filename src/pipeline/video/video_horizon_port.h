// Port of the lane-2 frame-horizon tool onto the nightly/eff-perc decision
// path (`AV2ENC_PORT_HORIZON`, default off).
//
// The lane-2 tool (`docs/agent-loop-ledger-2.md`, `video_pipeline_horizon.cpp`)
// prices a leaf against the error the choice installs in the *next* K source
// pictures, not just this one:
//
//   J_skip = sum_k w_k * SSE(source[i+k], prediction) + lambda * R_skip
//   J_res  = sum_k w_k * SSE(source[i+k], recon)      + lambda * R_res
//   w_0 = 1, w_k = 0.5^k, k = 0..K
//
// A skipped leaf copies its prediction into the reference, and every picture
// that keeps skipping keeps copying it, so the k>0 terms are the real future
// source frames the sequence already holds (the caller passes
// `frames.subspan(i + 1)` into InterFrameEncoder).  The prediction primitive
// itself is the lane's `horizon_predict_zero_mv` / `horizon_predict_mv` in
// `video_interp_filter.h`; the ported decision only needs the already-computed
// prediction/reconstruction, so the tool body is unchanged.
//
// Off by default: `enabled()` is false and the caller adds nothing, so every
// decision and every pinned stream stays byte-identical.
#ifndef AV2ENC_PIPELINE_VIDEO_HORIZON_PORT_H_
#define AV2ENC_PIPELINE_VIDEO_HORIZON_PORT_H_

#include <cstdint>
#include <cstdlib>
#include <span>

#include "pipeline/video/video_frame.h"

namespace av2enc::video::horizon_port {

inline bool enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_PORT_HORIZON");
    if (env == nullptr || env[0] == '\0') return false;
    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N') return false;
    return true;
  }();
  return value;
}

// K = 2 / 4 / 8; anything else falls back to 2 (the lane's envelope arm).
inline int horizon_k() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_PORT_HORIZON_K");
    if (env == nullptr || env[0] == '\0') return 2;
    const int parsed = std::atoi(env);
    return parsed == 4 || parsed == 8 || parsed == 2 ? parsed : 2;
  }();
  return value;
}

// The lane's stage-3 aggregation constant (0.5^k).
inline constexpr double kHorizonDecay = 0.5;

// SSE of one future source plane against a tightly packed block, with the
// source extent clamp `source_sample()` applies.  `samples` is w*h
// (luma) or (w/2)*(h/2) (chroma).
inline double future_plane_sse(const VideoInputFrame& frame, int plane,
                               int x0, int y0, int w, int h,
                               const std::uint8_t* samples, int stride) {
  if (samples == nullptr || w <= 0 || h <= 0) return 0.0;
  const int frame_w = plane == 0 ? static_cast<int>(frame.width)
                                 : (static_cast<int>(frame.width) + 1) / 2;
  const int frame_h = plane == 0 ? static_cast<int>(frame.height)
                                 : (static_cast<int>(frame.height) + 1) / 2;
  if (frame_w <= 0 || frame_h <= 0) return 0.0;
  const std::span<const std::uint8_t> data =
      plane == 0 ? frame.y : (plane == 1 ? frame.u : frame.v);
  if (data.empty()) return 0.0;
  double sum = 0.0;
  for (int y = 0; y < h; ++y) {
    int sy = y0 + y;
    if (sy < 0) sy = 0;
    if (sy > frame_h - 1) sy = frame_h - 1;
    const std::uint8_t* row =
        data.data() + static_cast<std::size_t>(sy) *
                          static_cast<std::size_t>(frame_w);
    const std::uint8_t* coded =
        samples + static_cast<std::size_t>(y) * static_cast<std::size_t>(stride);
    for (int x = 0; x < w; ++x) {
      int sx = x0 + x;
      if (sx < 0) sx = 0;
      if (sx > frame_w - 1) sx = frame_w - 1;
      const int d = static_cast<int>(row[sx]) - static_cast<int>(coded[x]);
      sum += static_cast<double>(d) * static_cast<double>(d);
    }
  }
  return sum;
}

// sum_{k=1..K} 0.5^k * SSE(source[i+k] block, coded block), for all three
// planes.  `future` is the span of frames after the current one.
inline double future_distortion(std::span<const VideoInputFrame> future,
                                int horizon, int x0, int y0, int w, int h,
                                const std::uint8_t* y, const std::uint8_t* u,
                                const std::uint8_t* v) {
  if (horizon <= 0 || future.empty() || y == nullptr) return 0.0;
  const int cw = w / 2;
  const int ch = h / 2;
  const int cx0 = x0 / 2;
  const int cy0 = y0 / 2;
  double value = 0.0;
  double weight = kHorizonDecay;  // k = 1
  for (int k = 0; k < horizon; ++k) {
    const std::size_t index = static_cast<std::size_t>(k);
    if (index >= future.size()) break;
    const VideoInputFrame& frame = future[index];
    value += weight * (future_plane_sse(frame, 0, x0, y0, w, h, y, w) +
                       future_plane_sse(frame, 1, cx0, cy0, cw, ch, u, cw) +
                       future_plane_sse(frame, 2, cx0, cy0, cw, ch, v, cw));
    weight *= kHorizonDecay;
  }
  return value;
}

}  // namespace av2enc::video::horizon_port

#endif  // AV2ENC_PIPELINE_VIDEO_HORIZON_PORT_H_
