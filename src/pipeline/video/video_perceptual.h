// Opt-in perceptual distortion proxy for the TX-rate-distortion decision.
//
// The encoder prices distortion with SSE (or SATD), which assumes every
// decoded sample error is equally visible.  Butteraugli does not: on this
// project's 240p clip the measured error mass is on the fine-detail / text
// band of the P frames, not on the flat background (the flat half is already
// coded at about 54 dB PSNR).  Plain SSE therefore under-prices the
// detail-rich blocks that dominate the metric, and the TX search prunes their
// coefficients first.  This proxy re-weights the TX/distortion term by local
// source activity:
//
//   activity = mean absolute deviation of the 8x8 source luma cell
//   raw      = (kReferenceActivity + kActivitySoftness) / (activity + kSoft)
//   weight   = 1 + strength * (1 - raw / frame_mean_raw)
//
// so a cell whose activity equals the frame mean (after the inverse-activity
// curve) gets weight 1.0, a detail-rich cell gets more and a flat cell less.
// The frame mean keeps the strength knob close to a redistribution: the
// average weight over the frame is 1, so it does not silently change the
// global lambda.  8x8 is the smallest P-frame leaf and the block the leaf
// decision already reasons about.
//
// The TX search prices candidates as `sse + lambda * rate`, so weighting its
// SSE by the leaf factor is exactly dividing lambda by that factor.  This
// header exposes it that way (`tx_lambda_scale`) so the shared TX-search
// module -- and the CUDA reference contract it defines -- is untouched.
// Weighting the leaf-level candidate / skip-residual SSE as well was measured
// and is a wash or worse: the extra leaf terms move bits between planes and
// candidates but the metric gain comes from the TX precision in detail-rich
// leaves.  `docs/perceptual-rd.md` has the smooth-versus-detail sweeps, the
// per-frame Butteraugli band, and the ablations.
//
// The environment variable is read once (a process encodes one configuration)
// and strength 0 disables everything, so every pinned stream keeps its bytes:
// no weight map is built, `sse_vis()` returns the historical SSE and the TX
// lambda is passed through unchanged.
//
// Usage (leaf decision):
//   const av2enc::video::perceptual::LeafWeights w =
//       av2enc::video::perceptual::make_leaf_weights(...);
//   lambda * av2enc::video::perceptual::tx_lambda_scale(w)
//
// The proxy is deterministic: cell bounds are integer, the activity sum order
// is fixed (row-major, one double accumulator), and the weight is computed
// from integers plus the fixed constants only.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace av2enc::video::perceptual {

// AV2ENC_PERCEPTUAL_RD=<strength>.  Unset, empty, non-numeric or <= 0 keeps
// the historical cost.  Parsed once per process: the value is configuration,
// not per-frame state.
inline double rd_strength() {
  static const double value = [] {
    const char* env = std::getenv("AV2ENC_PERCEPTUAL_RD");
    if (env == nullptr || *env == '\0') return 0.0;
    char* end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || !std::isfinite(parsed) || parsed <= 0.0) return 0.0;
    return parsed;
  }();
  return value;
}

// Activity reference and softness in 8-bit luma units.  kReferenceActivity is
// the 8x8 MAD at which the raw weight is 1.0; the clip's cells are bimodal
// (about half the 240p frame is the black background at MAD 0, the textured
// half sits around 12-40), so 12 puts the crossover between the two modes
// without making either side extreme.  kActivitySoftness bounds the slope near
// 0 and keeps a single dark cell from dominating.
inline constexpr double kReferenceActivity = 12.0;
inline constexpr double kActivitySoftness = 12.0;
// The weight is a multiplier on a non-negative cost; keep it positive and
// bounded so a strength sweep stays interpretable.
inline constexpr double kMinWeight = 0.2;
inline constexpr double kMaxWeight = 5.0;

inline double activity_weight(double activity) {
  return (kReferenceActivity + kActivitySoftness) /
         (activity + kActivitySoftness);
}

// Mean absolute deviation of one 8x8 luma cell.  The window is shifted inside
// the picture at the right/bottom edge, so the read is always in bounds and a
// partial leaf cell still sees real neighbours instead of a clamp plateau.
inline double cell_activity(const std::uint8_t* src, int stride, int width,
                            int height, int sx, int sy) {
  if (src == nullptr || width <= 0 || height <= 0) return kReferenceActivity;
  sx = std::clamp(sx, 0, std::max(0, width - 8));
  sy = std::clamp(sy, 0, std::max(0, height - 8));
  const int cols = std::min(8, width);
  const int rows = std::min(8, height);
  double sum = 0.0;
  for (int y = 0; y < rows; ++y) {
    const std::uint8_t* row =
        src + static_cast<std::size_t>(sy + y) * static_cast<std::size_t>(stride);
    for (int x = 0; x < cols; ++x) sum += row[sx + x];
  }
  const double count = static_cast<double>(cols) * static_cast<double>(rows);
  const double mean = sum / count;
  double mad = 0.0;
  for (int y = 0; y < rows; ++y) {
    const std::uint8_t* row =
        src + static_cast<std::size_t>(sy + y) * static_cast<std::size_t>(stride);
    for (int x = 0; x < cols; ++x) {
      mad += std::abs(static_cast<double>(row[sx + x]) - mean);
    }
  }
  return mad / count;
}

// Mean raw weight over a deterministic 8x8-cell subsample of the frame.  It
// only sets the strength knob's global scale, so a coarse stride is enough;
// every leaf in the frame computes the same value.
inline double frame_mean_raw(const std::uint8_t* src, int stride, int width,
                             int height) {
  const int cols = (width + 7) / 8;
  const int rows = (height + 7) / 8;
  constexpr int kCellStride = 4;
  double sum = 0.0;
  int count = 0;
  for (int cy = 0; cy < rows; cy += kCellStride) {
    for (int cx = 0; cx < cols; cx += kCellStride) {
      sum += activity_weight(cell_activity(src, stride, width, height, cx * 8,
                                           cy * 8));
      ++count;
    }
  }
  if (count == 0) return 1.0;
  const double mean = sum / static_cast<double>(count);
  return mean > 0.0 ? mean : 1.0;
}

// 8x8-cell weights for one leaf extent.  `ox`/`oy` are absolute luma
// coordinates.  Cells that extend past the leaf edge still get a weight; the
// area-weighted mean (`block`) is the leaf factor the TX lambda uses.
struct LeafWeights {
  static constexpr int kCellsPerAxis = 16;  // 128x128 in 8x8 cells

  int ox = 0;
  int oy = 0;
  int cols = 0;
  int rows = 0;
  std::array<double, kCellsPerAxis * kCellsPerAxis> cell{};
  double block = 1.0;
  bool valid = false;

  double at(int x, int y) const {
    if (!valid) return 1.0;
    const int cx = std::clamp((x - ox) >> 3, 0, cols - 1);
    const int cy = std::clamp((y - oy) >> 3, 0, rows - 1);
    return cell[static_cast<std::size_t>(cy) * kCellsPerAxis +
                static_cast<std::size_t>(cx)];
  }
};

inline LeafWeights make_leaf_weights(const std::uint8_t* src, int stride,
                                     int width, int height, int ox, int oy,
                                     int bw, int bh, double strength) {
  LeafWeights out;
  if (!(strength > 0.0) || src == nullptr || bw <= 0 || bh <= 0 || width <= 0 ||
      height <= 0) {
    return out;
  }
  out.ox = ox;
  out.oy = oy;
  out.cols = std::clamp((bw + 7) / 8, 1, LeafWeights::kCellsPerAxis);
  out.rows = std::clamp((bh + 7) / 8, 1, LeafWeights::kCellsPerAxis);
  const double mean_raw = frame_mean_raw(src, stride, width, height);
  double area_sum = 0.0;
  for (int cy = 0; cy < out.rows; ++cy) {
    const int cell_y = std::min(oy + cy * 8, height - 1);
    for (int cx = 0; cx < out.cols; ++cx) {
      const int cell_x = std::min(ox + cx * 8, width - 1);
      const double raw =
          activity_weight(cell_activity(src, stride, width, height, cell_x,
                                        cell_y));
      double weight = 1.0 + strength * (1.0 - raw / mean_raw);
      weight = std::clamp(weight, kMinWeight, kMaxWeight);
      const std::size_t index = static_cast<std::size_t>(cy) *
                                    LeafWeights::kCellsPerAxis +
                                static_cast<std::size_t>(cx);
      out.cell[index] = weight;
      const int cw = std::min(bw - cx * 8, 8);
      const int ch = std::min(bh - cy * 8, 8);
      area_sum += weight * static_cast<double>(cw) * static_cast<double>(ch);
    }
  }
  out.block = area_sum / (static_cast<double>(bw) * static_cast<double>(bh));
  out.valid = true;
  return out;
}

// The TX search prices candidates as `sse + lambda * rate`, so weighting its
// SSE by the leaf factor is exactly dividing lambda by that factor.
inline double tx_lambda_scale(const LeafWeights& weights) {
  return weights.valid ? 1.0 / weights.block : 1.0;
}

}  // namespace av2enc::video::perceptual
