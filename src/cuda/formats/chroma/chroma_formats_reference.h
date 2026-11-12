// Host-only reference for the chroma-format prediction / residual /
// reconstruction stages, shared by the emulation test and the real-GPU test.
//
// This transcribes the corresponding bodies of
// `TileWriter::encode_residual_plane()` in
// `src/encoder/format/chroma/chroma_still.cpp` and calls the CPU module's own
// arithmetic helpers (`av2enc::chroma::approx_divide`, `round2`), so the two
// GPU tests use one oracle.  The 10-bit clip bound is the only extension; the
// CPU still writer is 8-bit-only by design.
#ifndef AV2ENC_CUDA_CHROMA_FORMATS_CHROMA_FORMATS_REFERENCE_H_
#define AV2ENC_CUDA_CHROMA_FORMATS_CHROMA_FORMATS_REFERENCE_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "cuda/formats/chroma/chroma_formats.h"
#include "formats/chroma/chroma_format.h"
#include "formats/chroma/chroma_residual.h"

namespace av2enc::cuda::chroma::testref {

inline int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

inline int clip1(int value, int bit_depth) {
  const int high = (1 << bit_depth) - 1;
  return value < 0 ? 0 : (value > high ? high : value);
}

// §7.13.2.1 / §7.13.2.10 DC prediction.
inline int dc_predict(const std::vector<std::uint16_t>& recon, int stride,
                      int coded_w, int coded_h, int x0, int y0, int nominal_w,
                      int nominal_h, int have_left, int have_above,
                      int bit_depth) {
  const int max_x = coded_w - 1;
  const int max_y = coded_h - 1;
  int sum_left = 0;
  int sum_above = 0;
  if (have_left != 0) {
    for (int i = 0; i < nominal_h; ++i) {
      const int y = std::min(std::max(y0 + i, 0), max_y);
      sum_left += static_cast<int>(recon[static_cast<std::size_t>(y) * stride +
                                        static_cast<std::size_t>(x0 - 1)]);
    }
  }
  if (have_above != 0) {
    for (int j = 0; j < nominal_w; ++j) {
      const int x = std::min(std::max(x0 + j, 0), max_x);
      sum_above += static_cast<int>(recon[static_cast<std::size_t>(y0 - 1) *
                                              stride +
                                          static_cast<std::size_t>(x)]);
    }
  }
  int pred = 1 << (bit_depth - 1);
  if (have_left != 0 && have_above != 0) {
    pred = clip1(av2enc::chroma::approx_divide(sum_left + sum_above,
                                               nominal_w + nominal_h),
                 bit_depth);
  } else if (have_left != 0) {
    pred = av2enc::chroma::round2(sum_left, floor_log2(nominal_h));
  } else if (have_above != 0) {
    pred = av2enc::chroma::round2(sum_above, floor_log2(nominal_w));
  }
  return pred;
}

// Forward residual: average the upscale_x * upscale_y samples the decoder's
// §7.15.4 duplication expands into.
inline void make_residual(const std::vector<std::uint16_t>& src, int stride,
                          int max_x, int max_y, int x0, int y0, int pred,
                          const TxShape& shape,
                          std::vector<std::int16_t>& residual) {
  residual.assign(static_cast<std::size_t>(shape.adj_w) * shape.adj_h, 0);
  const int shift = floor_log2(shape.upscale_x * shape.upscale_y);
  for (int j = 0; j < shape.adj_h; ++j) {
    for (int i = 0; i < shape.adj_w; ++i) {
      int sum = 0;
      for (int dy = 0; dy < shape.upscale_y; ++dy) {
        const int sy =
            std::min(std::max(y0 + j * shape.upscale_y + dy, 0), max_y);
        for (int dx = 0; dx < shape.upscale_x; ++dx) {
          const int sx =
              std::min(std::max(x0 + i * shape.upscale_x + dx, 0), max_x);
          sum += static_cast<int>(
                     src[static_cast<std::size_t>(sy) * stride +
                         static_cast<std::size_t>(sx)]) -
                 pred;
        }
      }
      residual[static_cast<std::size_t>(j) * shape.adj_w + i] =
          static_cast<std::int16_t>(av2enc::chroma::round2(sum, shift));
    }
  }
}

// recon = Clip1(pred + residual), duplicated by upscale_x/upscale_y.
inline void reconstruct(const std::vector<std::int32_t>& residual_adj,
                        int pred, const TxShape& shape, int bit_depth,
                        std::vector<std::uint16_t>& dst, int stride, int max_x,
                        int max_y, int x0, int y0) {
  for (int j = 0; j < shape.adj_h; ++j) {
    for (int i = 0; i < shape.adj_w; ++i) {
      const std::uint16_t value = static_cast<std::uint16_t>(clip1(
          pred + residual_adj[static_cast<std::size_t>(j) * shape.adj_w + i],
          bit_depth));
      for (int dy = 0; dy < shape.upscale_y; ++dy) {
        const int py = y0 + j * shape.upscale_y + dy;
        if (py > max_y) continue;
        for (int dx = 0; dx < shape.upscale_x; ++dx) {
          const int px = x0 + i * shape.upscale_x + dx;
          if (px > max_x) continue;
          dst[static_cast<std::size_t>(py) * stride +
              static_cast<std::size_t>(px)] = value;
        }
      }
    }
  }
}

}  // namespace av2enc::cuda::chroma::testref

#endif  // AV2ENC_CUDA_CHROMA_FORMATS_CHROMA_FORMATS_REFERENCE_H_
