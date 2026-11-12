#include "codec/filters/tables/gdf.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "codec/filters/tables/gdf_tables.h"
#include "codec/filters/tables/spec_math.h"

namespace av2enc::filters::spec {
namespace {

using namespace av2enc::filters::spec_tables;

std::size_t PlaneIndex(int width, int row, int col) {
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
         static_cast<std::size_t>(col);
}

constexpr int kVer = 0;
constexpr int kHor = 1;
constexpr int kDiag0 = 2;
constexpr int kDiag1 = 3;

// §7.20.5 grad_sum(grad, i, j, down, across).
std::int64_t GradSum(const std::vector<int>& grad, int width, int i, int j,
                     int down, int across) {
  std::int64_t t = 0;
  for (int i2 = 0; i2 < down; ++i2) {
    for (int j2 = 0; j2 < across; ++j2) {
      t += grad[PlaneIndex(width, i + i2, j + j2)];
    }
  }
  return t;
}

}  // namespace

GdfBlockIndices GdfBlockTableIndices(const GdfParams& params,
                                     const int* order_hints,
                                     const int* frame_distance,
                                     int num_total_refs,
                                     int restricted_oh) {
  GdfBlockIndices out;
  if (params.frame_is_intra) {
    out.ref_dst_idx = 0;
  } else {
    int max_dist = 0;
    const int n = std::min(num_total_refs, 2);
    for (int i = 0; i < n; ++i) {
      if (order_hints[i] != restricted_oh) {
        max_dist = std::max(Abs(frame_distance[i]), max_dist);
      }
    }
    if (max_dist == 0) {
      out.ref_dst_idx = 5;
    } else if (max_dist < 2) {
      out.ref_dst_idx = 1;
    } else if (max_dist < 3) {
      out.ref_dst_idx = 2;
    } else if (max_dist < 6) {
      out.ref_dst_idx = 3;
    } else if (max_dist < 11) {
      out.ref_dst_idx = 4;
    } else {
      out.ref_dst_idx = 5;
    }
  }

  const int qp_base = params.frame_is_intra ? 85 : 110;
  const int qp_diff = params.base_q_idx - qp_base - 24 * (params.bit_depth - 8);
  out.qp_idx = Clip3(0, 2, (qp_diff - 37) / 25) + params.gdf_pic_qc_idx;
  return out;
}

void ApplyGdfFilter(const LumaSource& src, std::uint16_t* lr, int lr_stride,
                    int x, int y, int qp_idx, int ref_dst_idx, int w, int h,
                    int stripe_end_y, int gdf_pix_scale, int bit_depth) {
  // The v1.0.0 pseudocode does not read stripeEndY; keep the parameter honest.
  (void)stripe_end_y;

  const int gw = w + 2;
  const int gh = h + 2;
  std::vector<int> grad[4];
  for (int d = 0; d < 4; ++d) {
    grad[d].resize(static_cast<std::size_t>(gw) * static_cast<std::size_t>(gh));
  }

  for (int i = 0; i < gh; ++i) {
    for (int j = 0; j < gw; ++j) {
      for (int d = 0; d < 4; ++d) {
        int dx = 0;
        int dy = 0;
        if (d == kVer) {
          dy = 1;
        } else if (d == kHor) {
          dx = 1;
        } else if (d == kDiag0) {
          dx = 1;
          dy = 1;
        } else {
          dx = 1;
          dy = -1;
        }
        const int a = luma_source_sample(src, x - 1 + j - dx, y - 1 + i - dy);
        const int b = luma_source_sample(src, x - 1 + j, y - 1 + i);
        const int c = luma_source_sample(src, x - 1 + j + dx, y - 1 + i + dy);
        grad[d][PlaneIndex(gw, i, j)] = Abs(b * 2 - a - c);
      }
    }
  }

  const int cw = w >> 1;
  const int ch = h >> 1;
  std::vector<std::uint8_t> gdf_cls(static_cast<std::size_t>(cw) *
                                      static_cast<std::size_t>(ch));

  for (int i = ch - 1; i >= 0; --i) {
    for (int j = 0; j < cw; ++j) {
      std::int64_t str[4];
      for (int d = 0; d < 4; ++d) {
        str[d] = GradSum(grad[d], gw, i * 2, j * 2, 4, 4);
      }
      int cls = str[kVer] > str[kHor] ? 0 : 1;
      cls |= str[kDiag0] > str[kDiag1] ? 0 : 2;
      gdf_cls[PlaneIndex(cw, i, j)] = static_cast<std::uint8_t>(cls);
    }
  }

  const int scale = ref_dst_idx == 0 ? 8 : 5;
  const int grad_shift = bit_depth == 8 ? 2 : 4;
  const int sample_shift = 10 - std::min(10, bit_depth);

  for (int i = 0; i < h; ++i) {
    const int y2 = i + y;
    for (int j = 0; j < w; ++j) {
      const int x2 = x + j;
      const int cls = gdf_cls[PlaneIndex(cw, i >> 1, j >> 1)];
      std::int64_t gdf_idx[3] = {0, 0, 0};

      for (int k = 0; k < 18 + 4; ++k) {
        const int alpha = kGdfAlpha[ref_dst_idx][qp_idx][k][cls];
        int comb = 0;
        if (k < 18) {
          const int dy = kGdfCoords[k].dy;
          const int dx = kGdfCoords[k].dx;
          const int sample2 = luma_source_sample(src, x2, y2);
          const int sample3 = luma_source_sample(src, x2 - dx, y2 - dy);
          const int sample4 = luma_source_sample(src, x2 + dx, y2 + dy);
          const int above =
              Clip3(-alpha, alpha, (sample3 - sample2) << sample_shift);
          const int below =
              Clip3(-alpha, alpha, (sample4 - sample2) << sample_shift);
          comb = Clip3(-512, 511, above + below);
        } else {
          const int d = k - 18;
          const std::int64_t v =
              GradSum(grad[d], gw, (i >> 1) << 1, (j >> 1) << 1, 4, 4) >>
              grad_shift;
          comb = static_cast<int>(std::min<std::int64_t>(v, alpha));
        }
        for (int idx = 0; idx < 3; ++idx) {
          gdf_idx[idx] += static_cast<std::int64_t>(comb) *
                          kGdfWeight[ref_dst_idx][qp_idx][idx][k][cls];
        }
      }

      int pos = 0;
      for (int idx = 0; idx < 3; ++idx) {
        const std::int64_t v = Round2Signed(
            (gdf_idx[idx] + kGdfBias[ref_dst_idx][qp_idx][idx]) * scale, 15);
        pos = pos * scale * 2 +
              Clip3(-scale, scale - 1, static_cast<int>(v)) + scale;
      }

      const int err = ref_dst_idx == 0 ? kGdfIntraError[qp_idx][pos]
                                       : kGdfInterError[ref_dst_idx - 1][qp_idx][pos];
      const std::int64_t delta =
          Round2Signed(static_cast<std::int64_t>(err) * gdf_pix_scale,
                       12 - bit_depth);
      const int base = lr[static_cast<std::ptrdiff_t>(y2) * lr_stride + x2];
      lr[static_cast<std::ptrdiff_t>(y2) * lr_stride + x2] =
          static_cast<std::uint16_t>(Clip1(bit_depth, base + static_cast<int>(delta)));
    }
  }
}

}  // namespace av2enc::filters::spec
