// Device-side helpers for the CUDA M2 P-frame baseline.
//
// These functions are the exact code the CUDA kernels run and the exact code
// the host-emulation parity test runs, so the arithmetic cannot drift between
// the GPU backend and the emulation backend.
//
// Everything here mirrors the CPU reference in src/encoder/m2/m2_frame.cpp:
//
//   * source_sample()  - clamp to the visible frame bounds
//   * block variance   - binary64 mean, then binary64 sum of squared
//                        deviations in the same row-major order
//   * SSE              - integer, exact
//   * residual / recon - integer, exact
//
// The variance expression is the only floating-point code.  The CUDA build
// compiles this file with --fmad=false (same rule as the M1 transform path)
// and uses explicit __dadd_rn/__dmul_rn so contraction cannot change the
// result.  See the README for the FP-contraction caveat.
#ifndef AV2ENC_CUDA_M2_M2_DEVICE_CUH_
#define AV2ENC_CUDA_M2_M2_DEVICE_CUH_

#include <cstdint>

#if defined(__CUDACC__)
#define AV2ENC_M2_HD __host__ __device__ __forceinline__
#else
#define AV2ENC_M2_HD inline
#endif
// __CUDA_ARCH__ is only defined in the device pass; the host pass of nvcc
// cannot call the device-only double intrinsics.
#if defined(__CUDA_ARCH__)
#define AV2ENC_M2_DADD(a, b) __dadd_rn((a), (b))
#define AV2ENC_M2_DMUL(a, b) __dmul_rn((a), (b))
#else
#define AV2ENC_M2_DADD(a, b) ((a) + (b))
#define AV2ENC_M2_DMUL(a, b) ((a) * (b))
#endif

namespace av2enc::cuda::m2::device {

AV2ENC_M2_HD int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

AV2ENC_M2_HD int clip1(int value) { return clip3(0, 255, value); }

// The CPU reference clamps every source read to the visible frame.  Keep the
// same argument order (x then y) so the sample sequence is identical.
AV2ENC_M2_HD int source_sample(const std::uint8_t* plane, int stride,
                               int frame_w, int frame_h, int x, int y) {
  const int cx = clip3(0, frame_w - 1, x);
  const int cy = clip3(0, frame_h - 1, y);
  return plane[static_cast<std::size_t>(cy) *
                   static_cast<std::size_t>(stride) +
               static_cast<std::size_t>(cx)];
}

// block_variance_cost_at() from src/encoder/m2/m2_frame.cpp.  `w`/`h` are the
// visible block dimensions and x0/y0 its origin.
AV2ENC_M2_HD std::int64_t block_variance_cost(const std::uint8_t* plane,
                                              int stride, int frame_w,
                                              int frame_h, int x0, int y0,
                                              int w, int h) {
  if (w <= 0 || h <= 0) {
    return 0;
  }
  std::int64_t sum = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      sum += source_sample(plane, stride, frame_w, frame_h, x0 + x, y0 + y);
    }
  }
  const double mean = static_cast<double>(sum) /
                      static_cast<double>(w * h);
  double sse = 0.0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const double d =
          static_cast<double>(
              source_sample(plane, stride, frame_w, frame_h, x0 + x, y0 + y)) -
          mean;
      sse = AV2ENC_M2_DADD(sse, AV2ENC_M2_DMUL(d, d));
    }
  }
  return static_cast<std::int64_t>(sse);
}

// sse_against_source() from src/encoder/m2/m2_frame.cpp.  Integer only.
AV2ENC_M2_HD std::int64_t sse_against_source(
    const std::uint8_t* src_plane, int src_stride, int frame_w, int frame_h,
    int x0, int y0, const std::uint8_t* pred, int pred_stride, int w, int h) {
  std::int64_t sse = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int src =
          source_sample(src_plane, src_stride, frame_w, frame_h, x0 + x,
                        y0 + y);
      const int p = pred[static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(pred_stride) +
                         static_cast<std::size_t>(x)];
      const int d = src - p;
      sse += static_cast<std::int64_t>(d) * d;
    }
  }
  return sse;
}

AV2ENC_M2_HD std::int16_t residual_sample(int src, int pred) {
  return static_cast<std::int16_t>(src - pred);
}

AV2ENC_M2_HD std::uint8_t recon_sample(int pred, int residual) {
  return static_cast<std::uint8_t>(clip1(pred + residual));
}

}  // namespace av2enc::cuda::m2::device

#endif  // AV2ENC_CUDA_M2_M2_DEVICE_CUH_
