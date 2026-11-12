// Test helpers and independent spec-transcription oracles for the AV2
// restoration components.
//
// The oracles here are deliberately written as a second, naive transcription
// of the spec pseudocode (spec-order loops, std::min/max, separate sample
// getters) rather than by calling the core headers.  They are used to check
// both the CPU reference and the CUDA kernels, so a shared transcription bug
// in the production path is not hidden by a copy of itself.

#ifndef AV2ENC_ENCODER_FILTERS_RESTORATION_RESTORATION_TEST_COMMON_H_
#define AV2ENC_ENCODER_FILTERS_RESTORATION_RESTORATION_TEST_COMMON_H_

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "codec/filters/loop_restoration/restoration.h"

namespace av2enc::restoration::test {

constexpr std::uint8_t kGuard = 0xEE;
constexpr int kGuardCols = 8;
constexpr int kGuardRows = 2;

inline int clip3i(int lo, int hi, int v) {
  return v < lo ? lo : (v > hi ? hi : v);
}

struct TestFrame {
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;
  int width[3] = {0, 0, 0};
  int height[3] = {0, 0, 0};
  int stride[3] = {0, 0, 0};
  std::vector<std::uint8_t> curr[3];
  std::vector<std::uint8_t> cdef[3];
  std::vector<std::uint8_t> lr[3];

  int plane_width(int p) const {
    return (mi_cols * 4) >> (p == 0 ? 0 : sub_x);
  }
  int plane_height(int p) const {
    return (mi_rows * 4) >> (p == 0 ? 0 : sub_y);
  }
  std::uint8_t& at(std::vector<std::uint8_t>& v, int p, int y, int x) {
    return v[static_cast<std::size_t>(y * stride[p] + x)];
  }
  std::uint8_t at(const std::vector<std::uint8_t>& v, int p, int y,
                  int x) const {
    return v[static_cast<std::size_t>(y * stride[p] + x)];
  }

  void allocate() {
    for (int p = 0; p < 3; ++p) {
      width[p] = plane_width(p);
      height[p] = plane_height(p);
      stride[p] = width[p] + kGuardCols;
      const std::size_t n = static_cast<std::size_t>(stride[p]) *
                            static_cast<std::size_t>(height[p] + kGuardRows);
      curr[p].assign(n, kGuard);
      cdef[p].assign(n, kGuard);
      lr[p].assign(n, kGuard);
    }
  }

  RestorationFrame view() {
    RestorationFrame f;
    for (int p = 0; p < 3; ++p) {
      f.curr[p].data = curr[p].data();
      f.curr[p].stride = stride[p];
      f.cdef[p].data = cdef[p].data();
      f.cdef[p].stride = stride[p];
      f.lr[p].data = lr[p].data();
      f.lr[p].stride = stride[p];
    }
    f.mi_cols = mi_cols;
    f.mi_rows = mi_rows;
    f.subsampling_x = sub_x;
    f.subsampling_y = sub_y;
    f.bit_depth = 8;
    return f;
  }

  // Returns true when every guard sample is still kGuard.
  bool guard_intact() const {
    for (int p = 0; p < 3; ++p) {
      for (int y = 0; y < height[p] + kGuardRows; ++y) {
        for (int x = 0; x < stride[p]; ++x) {
          const bool visible = y < height[p] && x < width[p];
          if (visible) {
            continue;
          }
          const std::uint8_t v = at(cdef[p], p, y, x);
          if (v != kGuard) {
            return false;
          }
        }
      }
    }
    return true;
  }

  bool lr_guard_intact() const {
    for (int p = 0; p < 3; ++p) {
      for (int y = 0; y < height[p] + kGuardRows; ++y) {
        for (int x = 0; x < stride[p]; ++x) {
          const bool visible = y < height[p] && x < width[p];
          if (visible) {
            continue;
          }
          if (at(lr[p], p, y, x) != kGuard) {
            return false;
          }
        }
      }
    }
    return true;
  }
};

struct Rng {
  std::uint32_t s;
  explicit Rng(std::uint32_t seed) : s(seed ? seed : 1u) {}
  std::uint32_t next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  int range(int lo, int hi) {  // inclusive
    return lo + static_cast<int>(next() % static_cast<std::uint32_t>(hi - lo + 1));
  }
};

inline void fill_random(TestFrame& f, Rng& rng, int lo = 0, int hi = 255) {
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p]; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        f.at(f.curr[p], p, y, x) = static_cast<std::uint8_t>(rng.range(lo, hi));
        f.at(f.cdef[p], p, y, x) = static_cast<std::uint8_t>(rng.range(lo, hi));
      }
    }
  }
}

inline void fill_smooth(TestFrame& f) {
  for (int p = 0; p < 3; ++p) {
    for (int y = 0; y < f.height[p]; ++y) {
      for (int x = 0; x < f.width[p]; ++x) {
        const int v = clip3i(0, 255, 40 + 3 * x + 2 * y + (x * y) / 13);
        f.at(f.curr[p], p, y, x) = static_cast<std::uint8_t>(v);
        f.at(f.cdef[p], p, y, x) =
            static_cast<std::uint8_t>(clip3i(0, 255, v + 5));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Naive CCSO (§7.19.1) — spec-order transcription.
// ---------------------------------------------------------------------------

inline void naive_ccso(TestFrame& f, const CcsoFrameParams& params,
                       const std::uint8_t* segment_ids = nullptr,
                       const std::uint8_t* chroma_segment_ids = nullptr,
                       const std::uint8_t* lossless = nullptr,
                       const std::uint8_t* bru_modes = nullptr,
                       int disable_across_tiles = 0,
                       const std::uint16_t* mcs = nullptr,
                       const std::uint16_t* mce = nullptr,
                       const std::uint16_t* mrs = nullptr,
                       const std::uint16_t* mre = nullptr,
                       int sb_width = 64, int sb_height = 64) {
  const int mi_cols = f.mi_cols;
  const int mi_rows = f.mi_rows;
  for (int plane = 0; plane < 3; ++plane) {
    const CcsoParams& cp = params.plane[plane];
    if (!cp.enabled) {
      continue;
    }
    const int subX = (plane == 0) ? 0 : f.sub_x;
    const int subY = (plane == 0) ? 0 : f.sub_y;
    const int shiftY = params.ccso_luma_size_log2 - subY;
    const int shiftX = params.ccso_luma_size_log2 - subX;
    const int blkH = 1 << shiftY;
    const int blkW = 1 << shiftX;
    const int planeWidth = (mi_cols * 4) >> subX;
    const int planeHeight = (mi_rows * 4) >> subY;
    const int sbBlkW = (sb_width >> subX);
    const int sbBlkH = (sb_height >> subY);
    const int quantStep = kCcsoQuantSz[cp.scale_idx][cp.quant_idx];
    const int dy = kCcsoPos[cp.ext_filter][0];
    const int dx = kCcsoPos[cp.ext_filter][1];

    for (int y = 0; y < planeHeight; y += blkH) {
      for (int x = 0; x < planeWidth; x += blkW) {
        const int unitRow = y >> shiftY;
        const int unitCol = x >> shiftX;
        const bool useCcso = params.blks[plane] != nullptr &&
                             params.blks[plane][unitRow * params.blk_cols[plane] +
                                                unitCol] != 0;
        for (int y2 = y; y2 < std::min(planeHeight, y + blkH); y2 += sbBlkH) {
          for (int x2 = x; x2 < std::min(planeWidth, x + blkW);
               x2 += sbBlkW) {
            const int bruRow = y2 >> (2 - subY);
            const int bruCol = x2 >> (2 - subX);
            const bool bruActive =
                bru_modes == nullptr ||
                bru_modes[bruRow * mi_cols + bruCol] == kBruActive;
            if (!useCcso || !bruActive) {
              continue;
            }
            const int shift = 8 - cp.max_band_log2;
            for (int y3 = y2; y3 < std::min(planeHeight, y2 + sbBlkH); ++y3) {
              for (int x3 = x2; x3 < std::min(planeWidth, x2 + sbBlkW);
                   ++x3) {
                const int yLuma = y3 << subY;
                const int xLuma = x3 << subX;
                const int row = yLuma >> 2;
                const int col = xLuma >> 2;
                const int s = (plane > 0)
                                  ? (chroma_segment_ids != nullptr
                                         ? chroma_segment_ids[row * mi_cols + col]
                                         : 0)
                                  : (segment_ids != nullptr
                                         ? segment_ids[row * mi_cols + col]
                                         : 0);
                if (lossless != nullptr && lossless[s] != 0) {
                  continue;
                }
                int miColStart = 0;
                int miRowStart = 0;
                int miColEnd = mi_cols;
                int miRowEnd = mi_rows;
                if (disable_across_tiles != 0) {
                  miColStart = mcs[row * mi_cols + col];
                  miColEnd = mce[row * mi_cols + col];
                  miRowStart = mrs[row * mi_cols + col];
                  miRowEnd = mre[row * mi_cols + col];
                }
                const int LumaStartX = miColStart * 4;
                const int LumaStartY = miRowStart * 4;
                const int LumaEndX = miColEnd * 4 - 1;
                const int LumaEndY = miRowEnd * 4 - 1;
                auto getLuma = [&](int lx, int ly) {
                  const int cx = clip3i(LumaStartX, LumaEndX, lx);
                  const int cy = clip3i(LumaStartY, LumaEndY, ly);
                  return static_cast<int>(f.at(f.curr[0], 0, cy, cx));
                };
                const int c = getLuma(xLuma, yLuma);
                const int band = c >> shift;
                int cls0 = 0;
                int cls1 = 0;
                if (!cp.bo_only) {
                  cls0 = ccso_score(getLuma(xLuma + dx, yLuma + dy) - c,
                                    quantStep, cp.edge_clf);
                  cls1 = ccso_score(getLuma(xLuma - dx, yLuma - dy) - c,
                                    quantStep, cp.edge_clf);
                }
                const int off =
                    cp.offsets[ccso_offset_index(band, cls0, cls1)];
                const int v = static_cast<int>(f.at(f.cdef[plane], plane, y3, x3)) + off;
                f.at(f.cdef[plane], plane, y3, x3) =
                    static_cast<std::uint8_t>(clip3i(0, 255, v));
              }
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Naive loop restoration (§7.20.1–§7.20.3) — spec-order transcription.
// ---------------------------------------------------------------------------

struct NaiveLrFrame {
  const TestFrame* src = nullptr;
  const LoopRestorationParams* params = nullptr;
  const std::uint8_t* segment_ids = nullptr;
  const std::uint8_t* chroma_segment_ids = nullptr;
  const std::uint8_t* lossless = nullptr;
  const std::uint16_t* mcs = nullptr;
  const std::uint16_t* mce = nullptr;
  const std::uint16_t* mrs = nullptr;
  const std::uint16_t* mre = nullptr;
  int disable_across_tiles = 0;
  const std::uint8_t* filter_class_grid = nullptr;
  const std::uint8_t* subclass_lookup = nullptr;
  const std::uint8_t* subclass_grid = nullptr;
  int cfl_ds_filter_index = 0;
};

struct NaiveRegion {
  int LumaStartX = 0;
  int LumaStartY = 0;
  int LumaEndX = 0;
  int LumaEndY = 0;
  int LumaStripeStartY = 0;
  int LumaStripeEndY = 0;
};

inline int naive_source_sample(const NaiveLrFrame& n, const NaiveRegion& r,
                               int plane, int x, int y) {
  const TestFrame& f = *n.src;
  const int subX = (plane == 0) ? 0 : f.sub_x;
  const int subY = (plane == 0) ? 0 : f.sub_y;
  x = clip3i(r.LumaStartX >> subX, r.LumaEndX >> subX, x);
  y = clip3i(r.LumaStartY >> subY, r.LumaEndY >> subY, y);
  const int stripeStartY = r.LumaStripeStartY >> subY;
  const int stripeEndY = r.LumaStripeEndY >> subY;
  if (y < stripeStartY) {
    y = std::max(stripeStartY - 2, y);
    return f.at(f.curr[plane], plane, y, x);
  }
  if (y > stripeEndY) {
    y = std::min(stripeEndY + 2, y);
    return f.at(f.curr[plane], plane, y, x);
  }
  return f.at(f.cdef[plane], plane, y, x);
}

inline int naive_luma_sample(const NaiveLrFrame& n, const NaiveRegion& r,
                             int x, int y) {
  const TestFrame& f = *n.src;
  const int lastY = f.mi_rows * 4 - 1 - f.sub_y;
  const int lastX = r.LumaEndX - f.sub_x;
  x <<= f.sub_x;
  y <<= f.sub_y;
  y = clip3i(0, lastY, y);
  x = clip3i(r.LumaStartX, lastX, x);
  int idx = n.cfl_ds_filter_index;
  if (idx == 3) {
    idx = 0;
  }
  if (f.sub_x && f.sub_y && idx <= 1) {
    int t = 0;
    for (int dy = 0; dy < 2; ++dy) {
      for (int dx = 0; dx < 2; ++dx) {
        t += static_cast<int>(kWienerFilters420[idx][dy][dx]) *
             naive_source_sample(n, r, 0, x + dx, y + dy);
      }
    }
    return t >> 2;
  }
  return naive_source_sample(n, r, 0, x, y);
}

// Output is written into out[plane] (same stride as src).
inline void naive_loop_restoration(const NaiveLrFrame& n,
                                   std::vector<std::uint8_t>* out) {
  const TestFrame& f = *n.src;
  const LoopRestorationParams& p = *n.params;
  // LrFrame := CdefFrame
  for (int plane = 0; plane < 3; ++plane) {
    for (int y = 0; y < f.height[plane]; ++y) {
      for (int x = 0; x < f.width[plane]; ++x) {
        out[plane][static_cast<std::size_t>(y * f.stride[plane] + x)] =
            f.at(f.cdef[plane], plane, y, x);
      }
    }
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (p.frame_type[plane] == kRestoreNone) {
      continue;
    }
    const int subX = (plane == 0) ? 0 : f.sub_x;
    const int subY = (plane == 0) ? 0 : f.sub_y;
    const int unitSize = p.unit_size[plane];
    for (int row = 0; row < f.mi_rows; ++row) {
      for (int col = 0; col < f.mi_cols; ++col) {
        // lossless
        const int seg = (plane > 0)
                            ? (n.chroma_segment_ids != nullptr
                                   ? n.chroma_segment_ids[row * f.mi_cols + col]
                                   : 0)
                            : (n.segment_ids != nullptr
                                   ? n.segment_ids[row * f.mi_cols + col]
                                   : 0);
        if (n.lossless != nullptr && n.lossless[seg] != 0) {
          continue;
        }
        int miColStart = 0;
        int miRowStart = 0;
        int miColEnd = f.mi_cols;
        int miRowEnd = f.mi_rows;
        if (n.mcs != nullptr) {
          miColStart = n.mcs[row * f.mi_cols + col];
          miColEnd = n.mce[row * f.mi_cols + col];
          miRowStart = n.mrs[row * f.mi_cols + col];
          miRowEnd = n.mre[row * f.mi_cols + col];
        }
        const int lrRowOffset = (miRowStart * 4 >> subY) / unitSize;
        const int lrColOffset = (miColStart * 4 >> subX) / unitSize;
        const int stripeRow =
            std::min(miRowEnd - 1, ((row + 2) >> 4) << 4);
        (void)stripeRow;
        const int r = row - miRowStart;
        const int c = col - miColStart;
        const int miCols = miColEnd - miColStart;
        const int miRows = miRowEnd - miRowStart;
        const int lumaY = r * 4;
        const int stripeNum = (lumaY + 8) / 64;
        const int unitRows = std::max(
            ((miRows * 4 >> subY) + (unitSize >> 1)) / unitSize, 1);
        const int unitCols = std::max(
            ((miCols * 4 >> subX) + (unitSize >> 1)) / unitSize, 1);
        const int unitRow =
            lrRowOffset +
            std::min(unitRows - 1, ((r * 4 + 8) >> subY) / unitSize);
        const int unitCol =
            lrColOffset +
            std::min(unitCols - 1, (c * 4 >> subX) / unitSize);

        const int w = 4 >> subX;
        const int h = 4 >> subY;
        const int unclippedStart = miRowStart * 4 + stripeNum * 64 - 8;
        const int unclippedEnd = unclippedStart + 64;
        NaiveRegion reg;
        if (n.disable_across_tiles != 0) {
          reg.LumaStartX = miColStart * 4;
          reg.LumaEndX = miColEnd * 4 - 1;
          reg.LumaStartY = miRowStart * 4;
          reg.LumaEndY = miRowEnd * 4 - 1;
        } else {
          reg.LumaStartX = 0;
          reg.LumaEndX = f.mi_cols * 4 - 1;
          reg.LumaStartY = 0;
          reg.LumaEndY = f.mi_rows * 4 - 1;
        }
        reg.LumaStripeStartY = std::max(reg.LumaStartY, unclippedStart);
        reg.LumaStripeEndY = std::min(reg.LumaEndY, unclippedEnd - 1);

        const int rType = p.unit_type[plane][unitRow * p.unit_cols[plane] +
                                             unitCol];
        if (rType != kRestoreWienerNs) {
          continue;
        }
        const int x = (col * 4) >> subX;
        const int y = (row * 4) >> subY;
        const int nTaps = (plane == 0) ? kWienerNsTapsY : kWienerNsTapsUv;
        const std::int8_t(*cfg)[3] =
            (plane == 0) ? kWienerNsConfigY : kWienerNsConfigUv;
        const std::int16_t* unitCoeffs =
            p.unit_coeffs[plane] +
            (unitRow * p.unit_cols[plane] + unitCol) * kWienerNsCoeffStride;
        for (int rr = 0; rr < h; ++rr) {
          for (int cc = 0; cc < w; ++cc) {
            const int px = x + cc;
            const int py = y + rr;
            const int m = naive_source_sample(n, reg, plane, px, py);
            int s = m << 7;
            int subcls = 0;
            const bool frameFilters = p.frame_filters_on[plane] != 0;
            if (plane == 0 && p.frame_filters_on[0] && p.num_filter_classes > 1) {
              const int cell = (py >> 2) * f.mi_cols + (px >> 2);
              if (n.filter_class_grid != nullptr) {
                int cls = n.filter_class_grid[cell];
                if (cls < 0 || cls >= 16) {
                  cls = 0;
                }
                subcls = n.subclass_lookup != nullptr ? n.subclass_lookup[cls]
                                                      : cls;
              } else if (n.subclass_grid != nullptr) {
                subcls = n.subclass_grid[cell];
              }
              if (subcls < 0 || subcls >= p.num_filter_classes) {
                subcls = 0;
              }
            }
            const std::int16_t* frameCoeffs =
                frameFilters ? p.frame_coeffs[plane] +
                                   subcls * kWienerNsCoeffStride
                             : nullptr;
            for (int i = 0; i < nTaps; ++i) {
              const int diff =
                  naive_source_sample(n, reg, plane, px + cfg[i][1],
                                      py + cfg[i][0]) -
                  m;
              const int coeff =
                  frameFilters ? frameCoeffs[cfg[i][2]] : unitCoeffs[cfg[i][2]];
              s += diff * coeff;
            }
            if (plane > 0) {
              const int mLuma = naive_luma_sample(n, reg, px, py);
              for (int i = 0; i < nTaps; ++i) {
                const int coeff = frameFilters
                                      ? frameCoeffs[i + 6]
                                      : unitCoeffs[i + 6];
                if (coeff != 0) {
                  s += (naive_luma_sample(n, reg, px + cfg[i][1],
                                          py + cfg[i][0]) -
                        mLuma) *
                       coeff;
                }
              }
            }
            const int v = (s + (1 << (7 - 1))) >> 7;  // §4.8 Round2(s, 7)
            out[plane][static_cast<std::size_t>(py * f.stride[plane] + px)] =
                static_cast<std::uint8_t>(clip3i(0, 255, v));
          }
        }
      }
    }
  }
}

}  // namespace av2enc::restoration::test

#endif  // AV2ENC_ENCODER_FILTERS_RESTORATION_RESTORATION_TEST_COMMON_H_
