// Device-side AV2 GDF filter core (§7.20.1, §7.20.2, §7.20.5).
//
// Shared by the CUDA kernel (gdf.cu), the host-emulation parity test
// (gdf_emulation_test.cpp) and, indirectly, the GPU test's expectations.  The
// arithmetic is a direct transcription of the CPU reference in
// src/encoder/filters/spec_tables/gdf.cpp, using the same normative tables:
//
//   * four 4x4 gradient sums per 2x2 output cell (vertical, horizontal,
//     diagonal +1, diagonal -1);
//   * the 2-bit direction class from those sums;
//   * the 22-tap filter (18 spatial + 4 gradient) for all 6x6
//     refDstIdx x qpIdx combinations;
//   * the intra/inter error lookup and GdfPixScale rounding.
//
// Both the CUDA kernel and the host emulation run these exact functions, so
// the only difference between them is where the loops execute.

#ifndef AV2ENC_CUDA_GDF_GDF_DEVICE_CUH_
#define AV2ENC_CUDA_GDF_GDF_DEVICE_CUH_

#include "cuda/codec/filters/gdf/gdf_types.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__CUDA__)
#define AV2ENC_GDF_HD __host__ __device__ __forceinline__
#else
#define AV2ENC_GDF_HD inline
#endif

namespace av2enc::cuda::gdf {

// Direction indices, in the order used by §7.20.5 grad_sum().  Enum constants
// are compile-time values, safe in device code without host storage.
enum : int {
  kGdfDirVer = 0,
  kGdfDirHor = 1,
  kGdfDirDiag0 = 2,
  kGdfDirDiag1 = 3,
};

// Typed accessors for the device-resident normative tables.  The kernels only
// see GdfTables (device pointers), never a namespace-scope host object.
AV2ENC_GDF_HD int gdf_alpha(const GdfTables& tables, int ref_dst_idx,
                            int qp_idx, int tap, int cls) {
  return tables.alpha[((ref_dst_idx * kGdfNumQpIdx + qp_idx) * kGdfNumTaps +
                       tap) * kGdfNumClasses + cls];
}

AV2ENC_GDF_HD int gdf_weight(const GdfTables& tables, int ref_dst_idx,
                             int qp_idx, int component, int tap, int cls) {
  return tables.weight[(((ref_dst_idx * kGdfNumQpIdx + qp_idx) *
                             kGdfNumComponents +
                         component) *
                            kGdfNumTaps +
                        tap) *
                           kGdfNumClasses +
                       cls];
}

AV2ENC_GDF_HD int gdf_bias(const GdfTables& tables, int ref_dst_idx,
                           int qp_idx, int component) {
  return tables.bias[(ref_dst_idx * kGdfNumQpIdx + qp_idx) *
                         kGdfNumComponents +
                     component];
}

AV2ENC_GDF_HD int gdf_intra_error(const GdfTables& tables, int qp_idx,
                                  int pos) {
  return tables.intra_error[qp_idx * kGdfIntraPositions + pos];
}

AV2ENC_GDF_HD int gdf_inter_error(const GdfTables& tables, int ref_dst_idx_m1,
                                  int qp_idx, int pos) {
  return tables.inter_error[(ref_dst_idx_m1 * kGdfNumQpIdx + qp_idx) *
                                kGdfInterPositions +
                            pos];
}

// §4.8 Clip3 / Clip1 / Round2Signed, integer only.
AV2ENC_GDF_HD int gdf_clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

AV2ENC_GDF_HD int gdf_clip1(int bit_depth, int value) {
  return gdf_clip3(0, (1 << bit_depth) - 1, value);
}

AV2ENC_GDF_HD std::int64_t gdf_round2(std::int64_t value, int shift) {
  return shift <= 0 ? value
                    : (value + (std::int64_t{1} << (shift - 1))) >> shift;
}

AV2ENC_GDF_HD std::int64_t gdf_round2_signed(std::int64_t value, int shift) {
  return value >= 0 ? gdf_round2(value, shift) : -gdf_round2(-value, shift);
}

// §7.20.1 GdfBlockTableIndices.
AV2ENC_GDF_HD GdfTableIndices gdf_derive_table_indices(
    const GdfIndexParams& params) {
  GdfTableIndices out;
  if (params.frame_is_intra != 0) {
    out.ref_dst_idx = 0;
  } else {
    int max_dist = 0;
    const int n = params.num_total_refs < 2 ? params.num_total_refs : 2;
    for (int i = 0; i < n; ++i) {
      if (params.order_hints[i] != params.restricted_oh) {
        int dist = params.frame_distance[i];
        if (dist < 0) {
          dist = -dist;
        }
        if (dist > max_dist) {
          max_dist = dist;
        }
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

  const int qp_base = params.frame_is_intra != 0 ? 85 : 110;
  const int qp_diff =
      params.base_q_idx - qp_base - 24 * (params.bit_depth - 8);
  // C++ integer division truncates toward zero, which is what the CPU
  // reference does for the negative qp_diff values.
  const int qp_class = gdf_clip3(0, 2, (qp_diff - 37) / 25);
  out.qp_idx = qp_class + params.gdf_pic_qc_idx;
  return out;
}

// §7.20.2 get_source_sample(0, x, y).
AV2ENC_GDF_HD int gdf_source_sample(const GdfFrameSource& src, int x, int y) {
  x = gdf_clip3(src.start_x, src.end_x, x);
  y = gdf_clip3(src.start_y, src.end_y, y);
  if (y < src.stripe_start_y) {
    y = (src.stripe_start_y - 2 > y) ? src.stripe_start_y - 2 : y;
    return static_cast<int>(
        src.curr[static_cast<std::ptrdiff_t>(y) * src.curr_stride + x]);
  }
  if (y > src.stripe_end_y) {
    y = (src.stripe_end_y + 2 < y) ? src.stripe_end_y + 2 : y;
    return static_cast<int>(
        src.curr[static_cast<std::ptrdiff_t>(y) * src.curr_stride + x]);
  }
  return static_cast<int>(
      src.cdef[static_cast<std::ptrdiff_t>(y) * src.cdef_stride + x]);
}

// One §7.20.5 gradient sample: |2*b - a - c| at unit-local position
// (gi, gj) of the (w+2)x(h+2) gradient plane, direction d.
AV2ENC_GDF_HD int gdf_gradient_sample(const GdfFrameSource& src, int x, int y,
                                      int gi, int gj, int d) {
  int dx = 0;
  int dy = 0;
  if (d == kGdfDirVer) {
    dy = 1;
  } else if (d == kGdfDirHor) {
    dx = 1;
  } else if (d == kGdfDirDiag0) {
    dx = 1;
    dy = 1;
  } else {
    dx = 1;
    dy = -1;
  }
  const int sx = x - 1 + gj;
  const int sy = y - 1 + gi;
  const int a = gdf_source_sample(src, sx - dx, sy - dy);
  const int b = gdf_source_sample(src, sx, sy);
  const int c = gdf_source_sample(src, sx + dx, sy + dy);
  const int v = b * 2 - a - c;
  return v < 0 ? -v : v;
}

// §7.20.5 grad_sum over the 4x4 window at the origin of cell (ci, cj).  This
// is the only state the filter needs: the class decision and the four
// gradient taps both use exactly this sum (see the CPU reference, which stores
// the full gradient plane but reads only these windows).
AV2ENC_GDF_HD void gdf_cell_sums(const GdfFrameSource& src, int x, int y,
                                 int ci, int cj, int sums[4]) {
  const int gi = ci * 2;
  const int gj = cj * 2;
  for (int d = 0; d < 4; ++d) {
    int sum = 0;
    for (int i2 = 0; i2 < 4; ++i2) {
      for (int j2 = 0; j2 < 4; ++j2) {
        sum += gdf_gradient_sample(src, x, y, gi + i2, gj + j2, d);
      }
    }
    sums[d] = sum;
  }
}

// §7.20.5 2-bit direction class from the four gradient sums.
AV2ENC_GDF_HD int gdf_class_from_sums(const int sums[4]) {
  int cls = sums[kGdfDirVer] > sums[kGdfDirHor] ? 0 : 1;
  cls |= sums[kGdfDirDiag0] > sums[kGdfDirDiag1] ? 0 : 2;
  return cls;
}

// Gradient-tap inputs.  `sum >> grad_shift` is what §7.20.5 compares against
// alpha; alpha is at most 511 and the shifted sums fit uint16 for 8/10/12-bit
// (8-bit <= 2040, 10/12-bit <= 8190), so the cell table stores them directly.
AV2ENC_GDF_HD void gdf_taps_from_sums(const int sums[4], int bit_depth,
                                      std::uint16_t taps[4]) {
  const int grad_shift = bit_depth == 8 ? 2 : 4;
  for (int d = 0; d < 4; ++d) {
    taps[d] = static_cast<std::uint16_t>(sums[d] >> grad_shift);
  }
}

// §7.20.5 filter for one output sample.  Returns the signed delta applied to
// the LrFrame sample.  `taps` are the four shifted gradient sums of the 2x2
// cell that owns the sample; `cls` is that cell's direction class.
//
// The 22-term accumulation is bounded: |comb| <= 512, |weight| <= 512 and
// there are 22 taps, so |acc| < 5.8e6.  The (acc + bias) * scale step is done
// in int64 to match the CPU reference exactly.
AV2ENC_GDF_HD int gdf_filter_delta(const GdfFrameSource& src,
                                   const GdfTables& tables, int x2, int y2,
                                   int cls, const std::uint16_t taps[4],
                                   int qp_idx, int ref_dst_idx, int pix_scale,
                                   int bit_depth) {
  const int scale = ref_dst_idx == 0 ? 8 : 5;
  const int sample_shift = 10 - (bit_depth < 10 ? bit_depth : 10);
  const int shift = sample_shift;

  int acc[3] = {0, 0, 0};
  for (int k = 0; k < kGdfNumTaps; ++k) {
    const int alpha = gdf_alpha(tables, ref_dst_idx, qp_idx, k, cls);
    int comb = 0;
    if (k < 18) {
      const int dy = tables.coords[k].dy;
      const int dx = tables.coords[k].dx;
      const int sample2 = gdf_source_sample(src, x2, y2);
      const int sample3 = gdf_source_sample(src, x2 - dx, y2 - dy);
      const int sample4 = gdf_source_sample(src, x2 + dx, y2 + dy);
      // Multiplication rather than a left shift of a possibly negative
      // value; bit-identical to the CPU reference's << and free of UB.
      const int above =
          gdf_clip3(-alpha, alpha, (sample3 - sample2) * (1 << shift));
      const int below =
          gdf_clip3(-alpha, alpha, (sample4 - sample2) * (1 << shift));
      comb = gdf_clip3(-512, 511, above + below);
    } else {
      const int v = taps[k - 18];
      comb = v < alpha ? v : alpha;
    }
    for (int idx = 0; idx < 3; ++idx) {
      acc[idx] += comb * gdf_weight(tables, ref_dst_idx, qp_idx, idx, k, cls);
    }
  }

  int pos = 0;
  for (int idx = 0; idx < 3; ++idx) {
    const std::int64_t v = gdf_round2_signed(
        static_cast<std::int64_t>(
            acc[idx] + gdf_bias(tables, ref_dst_idx, qp_idx, idx)) *
            scale,
        15);
    pos = pos * scale * 2 + gdf_clip3(-scale, scale - 1, static_cast<int>(v)) +
          scale;
  }

  const int err =
      ref_dst_idx == 0 ? gdf_intra_error(tables, qp_idx, pos)
                       : gdf_inter_error(tables, ref_dst_idx - 1, qp_idx, pos);
  return static_cast<int>(gdf_round2_signed(
      static_cast<std::int64_t>(err) * pix_scale, 12 - bit_depth));
}

// §7.20.1/§7.20.5 stripe-region decomposition of one GdfBlks unit, matching
// av2enc::filters::gdf::GdfUnitRegions for a single-tile frame.  Stripes are
// offset upwards by 8 luma samples; the last unit row is clamped to MiRows-1.
// Writes at most kGdfMaxRegionsPerUnit regions and returns the count.
AV2ENC_GDF_HD int gdf_unit_regions(int unit_size, int row, int col, int cols,
                                   int rows, int mi_rows, int width,
                                   int height, GdfRegion out[3]) {
  if (out == nullptr || unit_size <= 0 || row < 0 || row >= rows || col < 0 ||
      col >= cols || mi_rows <= 0 || width <= 0 || height <= 0) {
    return 0;
  }
  const int unit_mi = unit_size / 4;
  int start_mi = unit_mi * row - 2;
  int end_mi = start_mi + unit_mi - 1;
  if (row == rows - 1) {
    end_mi = mi_rows - 1;
  }
  if (end_mi >= mi_rows) {
    end_mi = mi_rows - 1;
  }
  if (start_mi > end_mi) {
    return 0;
  }

  int ys = start_mi * 4;
  if (ys < 0) {
    ys = 0;
  }
  int ye = end_mi * 4 + 3;
  if (ye >= height) {
    ye = height - 1;
  }
  const int xs = col * unit_size;
  if (xs >= width) {
    return 0;
  }
  int xe = xs + unit_size - 1;
  if (xe >= width) {
    xe = width - 1;
  }

  int count = 0;
  int stripe = (ys + 8) / 64;
  while (stripe * 64 - 8 <= ye && count < kGdfMaxRegionsPerUnit) {
    const int unclipped_start = stripe * 64 - 8;
    const int stripe_start = unclipped_start > 0 ? unclipped_start : 0;
    const int stripe_end_candidate = unclipped_start + 63;
    const int stripe_end =
        stripe_end_candidate < height - 1 ? stripe_end_candidate : height - 1;
    const int region_y = ys > stripe_start ? ys : stripe_start;
    const int region_end = ye < stripe_end ? ye : stripe_end;
    if (region_y <= region_end) {
      GdfRegion& region = out[count];
      region.x = xs;
      region.y = region_y;
      region.w = xe - xs + 1;
      region.h = region_end - region_y + 1;
      region.stripe_start_y = stripe_start;
      region.stripe_end_y = stripe_end;
      ++count;
    }
    ++stripe;
  }
  return count;
}

// Complete §7.20.5 unit, used by the host-emulation parity test.  Cell-major
// order is safe because every output sample is written once and all source
// reads come from CurrFrame/CdefFrame, never from LrFrame.
AV2ENC_GDF_HD void gdf_filter_unit(const GdfFrameSource& src,
                                   const GdfTables& tables, std::uint16_t* lr,
                                   int lr_stride, int x, int y, int w, int h,
                                   int qp_idx, int ref_dst_idx, int pix_scale,
                                   int bit_depth) {
  const int cw = w >> 1;
  const int ch = h >> 1;
  for (int ci = 0; ci < ch; ++ci) {
    for (int cj = 0; cj < cw; ++cj) {
      int sums[4];
      gdf_cell_sums(src, x, y, ci, cj, sums);
      std::uint16_t taps[4];
      gdf_taps_from_sums(sums, bit_depth, taps);
      const int cls = gdf_class_from_sums(sums);
      for (int di = 0; di < 2; ++di) {
        const int i = ci * 2 + di;
        const int y2 = y + i;
        for (int dj = 0; dj < 2; ++dj) {
          const int j = cj * 2 + dj;
          const int x2 = x + j;
          const int delta = gdf_filter_delta(
              src, tables, x2, y2, cls, taps, qp_idx, ref_dst_idx, pix_scale,
              bit_depth);
          const std::ptrdiff_t off =
              static_cast<std::ptrdiff_t>(y2) * lr_stride + x2;
          lr[off] = static_cast<std::uint16_t>(
              gdf_clip1(bit_depth, static_cast<int>(lr[off]) + delta));
        }
      }
    }
  }
}

}  // namespace av2enc::cuda::gdf

#endif  // AV2ENC_CUDA_GDF_GDF_DEVICE_CUH_
