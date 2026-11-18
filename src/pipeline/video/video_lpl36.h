// Self-contained lpl36 perceptual error for the refresh planner.
//
// This is first-party av2enc code, released under the repository's Unlicense
// (public domain).  The metric recipe and the reference implementation are by
// baiango (the lpl36 metric, 2026); this port keeps that attribution only.
//
// lpl36 measures the multi-scale Laplacian-of-Gaussian energy of the luma
// difference:
//
//   error = (mean + std)(|L * G2 (Y_ref - Y_dist)|)
//         + 0.5 * (mean + std)(|L * G4 (Y_ref - Y_dist)|)
//
//   G  = cv2.pyrDown 5x5 Gaussian [1 4 6 4 1]/16, reflect-101 border,
//        log2(scale) cascaded applications, sampled at even positions;
//   L  = 5x5 LoG: center 16, mid-ring -2, outer-ring -1 (sums to zero),
//        reflect-101 border;
//   mean/std are population statistics over the whole plane.
//
// Lower error is better; identical planes give exactly 0.0.  The public
// reference scores -error (higher is better); the planner consumes the error
// form directly.
//
// The per-frame call is allocation-free: Init() owns every scratch plane and
// Error() only overwrites them.  The arithmetic is scalar float32 for the
// filters and float64 for the two pooled sums, which reproduces OpenCV's
// float pipeline to well inside the parity gate's 1e-3 relative tolerance.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace av2enc::video::lpl36 {

// Mirror index for OpenCV's BORDER_REFLECT_101: -1 -> 1, -2 -> 2,
// n -> n-2, n+1 -> n-3.  Handles degenerate sizes (0, 1, 2) defensively.
inline int reflect101(int index, int size) {
  if (size <= 1) return 0;
  if (index >= 0 && index < size) return index;
  const int period = 2 * size - 2;
  index %= period;
  if (index < 0) index += period;
  if (index >= size) index = period - index;
  return index;
}

class Metric {
 public:
  Metric() = default;

  // Sizes the scratch planes.  Not called from the per-frame path.
  void Init(int width, int height) {
    if (width <= 0 || height <= 0) {
      width_ = 0;
      height_ = 0;
      return;
    }
    width_ = width;
    height_ = height;
    half_width_ = (width + 1) / 2;
    half_height_ = (height + 1) / 2;
    quarter_width_ = (half_width_ + 1) / 2;
    quarter_height_ = (half_height_ + 1) / 2;
    const std::size_t full = static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
    const std::size_t half =
        static_cast<std::size_t>(half_width_) * static_cast<std::size_t>(half_height_);
    const std::size_t row =
        static_cast<std::size_t>(half_width_) * static_cast<std::size_t>(height_);
    // diff_ doubles as the LoG output buffer at both scales (its first
    // half/quarter-sized region); tmp_ is the horizontal pass of pyrDown.
    diff_.assign(full, 0.0f);
    tmp_.assign(row, 0.0f);
    half_.assign(half, 0.0f);
  }

  bool ready() const { return width_ > 0 && height_ > 0; }
  int width() const { return width_; }
  int height() const { return height_; }

  // Lower is better.  0.0 means the planes are identical.
  double Error(const std::uint8_t* ref_y, int ref_stride, const std::uint8_t* dist_y,
               int dist_stride) {
    if (!ready() || ref_y == nullptr || dist_y == nullptr) return 0.0;
    // Delta = Y_ref - Y_dist, signed and exact for 8-bit inputs.
    for (int y = 0; y < height_; ++y) {
      const std::uint8_t* ref_row =
          ref_y + static_cast<std::size_t>(y) * static_cast<std::size_t>(ref_stride);
      const std::uint8_t* dist_row =
          dist_y + static_cast<std::size_t>(y) * static_cast<std::size_t>(dist_stride);
      float* out = diff_.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(width_);
      for (int x = 0; x < width_; ++x) {
        out[x] = static_cast<float>(ref_row[x]) - static_cast<float>(dist_row[x]);
      }
    }

    // Scale 2: one pyrDown, then the 5x5 LoG on the low-passed difference.
    PyrDown(diff_.data(), width_, height_, tmp_.data(), half_.data(), half_width_, half_height_);
    FilterLog(half_.data(), half_width_, half_height_, diff_.data());
    const double pool2 = Pool(diff_.data(), static_cast<std::size_t>(half_width_) *
                                                static_cast<std::size_t>(half_height_));

    // Scale 4: a second pyrDown.  The destination aliases half_ on purpose:
    // the horizontal pass has already copied every source row it needs into
    // tmp_, and the vertical pass reads only tmp_.
    PyrDown(half_.data(), half_width_, half_height_, tmp_.data(), half_.data(), quarter_width_,
            quarter_height_);
    FilterLog(half_.data(), quarter_width_, quarter_height_, diff_.data());
    const double pool4 = Pool(diff_.data(), static_cast<std::size_t>(quarter_width_) *
                                                static_cast<std::size_t>(quarter_height_));
    return pool2 + 0.5 * pool4;
  }

  // Higher is better, matching the lpl36 reference score.
  double Score(const std::uint8_t* ref_y, int ref_stride, const std::uint8_t* dist_y,
               int dist_stride) {
    return -Error(ref_y, ref_stride, dist_y, dist_stride);
  }

 private:
  // OpenCV pyrDown: horizontal 5-tap Gaussian at full height, then vertical
  // 5-tap on the decimated columns.  Both passes decimate by two and divide by
  // 16.  src and dst may alias when tmp is a distinct buffer (the scale-4
  // call relies on this).
  static void PyrDown(const float* src, int src_width, int src_height, float* tmp, float* dst,
                      int dst_width, int dst_height) {
    constexpr float kG[5] = {1.0f, 4.0f, 6.0f, 4.0f, 1.0f};
    constexpr float kInv16 = 1.0f / 16.0f;
    for (int y = 0; y < src_height; ++y) {
      const float* row = src + static_cast<std::size_t>(y) * static_cast<std::size_t>(src_width);
      float* out = tmp + static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_width);
      for (int x = 0; x < dst_width; ++x) {
        const int sx = 2 * x;
        float sum = kG[0] * row[reflect101(sx - 2, src_width)];
        sum += kG[1] * row[reflect101(sx - 1, src_width)];
        sum += kG[2] * row[reflect101(sx, src_width)];
        sum += kG[3] * row[reflect101(sx + 1, src_width)];
        sum += kG[4] * row[reflect101(sx + 2, src_width)];
        out[x] = sum * kInv16;
      }
    }
    for (int y = 0; y < dst_height; ++y) {
      const int sy = 2 * y;
      const float* r0 = tmp + static_cast<std::size_t>(reflect101(sy - 2, src_height)) *
                                  static_cast<std::size_t>(dst_width);
      const float* r1 = tmp + static_cast<std::size_t>(reflect101(sy - 1, src_height)) *
                                  static_cast<std::size_t>(dst_width);
      const float* r2 = tmp + static_cast<std::size_t>(sy) * static_cast<std::size_t>(dst_width);
      const float* r3 = tmp + static_cast<std::size_t>(reflect101(sy + 1, src_height)) *
                                  static_cast<std::size_t>(dst_width);
      const float* r4 = tmp + static_cast<std::size_t>(reflect101(sy + 2, src_height)) *
                                  static_cast<std::size_t>(dst_width);
      float* out = dst + static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_width);
      for (int x = 0; x < dst_width; ++x) {
        float sum = kG[0] * r0[x];
        sum += kG[1] * r1[x];
        sum += kG[2] * r2[x];
        sum += kG[3] * r3[x];
        sum += kG[4] * r4[x];
        out[x] = sum * kInv16;
      }
    }
  }

  // 5x5 LoG, reflect-101.  The kernel has 13 non-zero taps:
  //   16 at the centre; -2 on the four axial neighbours; -1 on the remaining
  //   axial distance-2 taps and the four diagonals.
  static void FilterLog(const float* src, int width, int height, float* dst) {
    for (int y = 0; y < height; ++y) {
      const float* r0 = src + static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
      const float* rm1 = src + static_cast<std::size_t>(reflect101(y - 1, height)) *
                                   static_cast<std::size_t>(width);
      const float* rm2 = src + static_cast<std::size_t>(reflect101(y - 2, height)) *
                                   static_cast<std::size_t>(width);
      const float* rp1 = src + static_cast<std::size_t>(reflect101(y + 1, height)) *
                                   static_cast<std::size_t>(width);
      const float* rp2 = src + static_cast<std::size_t>(reflect101(y + 2, height)) *
                                   static_cast<std::size_t>(width);
      float* out = dst + static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
      for (int x = 0; x < width; ++x) {
        const int xm1 = reflect101(x - 1, width);
        const int xp1 = reflect101(x + 1, width);
        const int xm2 = reflect101(x - 2, width);
        const int xp2 = reflect101(x + 2, width);
        float sum = 16.0f * r0[x];
        sum -= 2.0f * (r0[xm1] + r0[xp1] + rm1[x] + rp1[x]);
        sum -= r0[xm2] + r0[xp2] + rm2[x] + rp2[x] + rm1[xm1] + rm1[xp1] + rp1[xm1] + rp1[xp1];
        out[x] = sum;
      }
    }
  }

  // mean + population std of |samples|.  The two moments are accumulated in
  // float64, matching cv2.meanStdDev's double accumulators.
  static double Pool(const float* samples, std::size_t count) {
    if (count == 0) return 0.0;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      const double value = std::fabs(static_cast<double>(samples[i]));
      sum += value;
      sum_sq += value * value;
    }
    const double inv = 1.0 / static_cast<double>(count);
    const double mean = sum * inv;
    const double variance = std::fmax(0.0, sum_sq * inv - mean * mean);
    return mean + std::sqrt(variance);
  }

  int width_ = 0;
  int height_ = 0;
  int half_width_ = 0;
  int half_height_ = 0;
  int quarter_width_ = 0;
  int quarter_height_ = 0;
  std::vector<float> diff_;  // w*h; later reused for the LoG output
  std::vector<float> tmp_;   // half_width * h; pyrDown horizontal pass
  std::vector<float> half_;  // half_width * half_height; low-pass / quarter
};

}  // namespace av2enc::video::lpl36
