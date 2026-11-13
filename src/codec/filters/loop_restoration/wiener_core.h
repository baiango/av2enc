// AV2 §7.20 loop restoration — shared host/device core.
//
// Implements the loop restoration framework (§7.20, §7.20.1, §7.20.2) and the
// non-separable Wiener filter (§7.20.3) for luma and chroma (including the
// chroma-from-luma taps).  The sample container is templated: `uint8_t` for
// the 8-bit product paths and `uint16_t` for the 10/12-bit format writer.
// PC-Wiener (§7.20.4) and GDF (§7.20.5) are separate normative modules
// (`codec/filters/tables/`, `codec/filters/gdf/`).
//
// The CPU reference (restoration.cpp) and the CUDA kernel
// (src/cuda/restoration/wiener.cu) call the same inline functions, so results
// are bit-exact by construction.
//
// Sample sourcing (§7.20.2) is the subtle part:
//   * Samples inside the current 64-luma-line stripe (offset up by 8 lines)
//     come from CdefFrame (post-CDEF/CCSO).
//   * Samples outside the stripe come from CurrFrame (deblocked only), with
//     vertical fetch clamped to at most 2 lines away from the stripe.  Results
//     are therefore independent of how far outside the stripe a tap reaches.
//   * Horizontal/vertical access is clipped to the tile extent when
//     disable_loopfilters_across_tiles is set, else to the frame.
//
// The CfL luma downsampling table is 4:2:0-only (§7.20.3); for 4:2:2/4:4:4 the
// luma sample is fetched directly, which the condition `sub_x && sub_y` below
// expresses.

#ifndef AV2ENC_ENCODER_FILTERS_RESTORATION_WIENER_CORE_H_
#define AV2ENC_ENCODER_FILTERS_RESTORATION_WIENER_CORE_H_

#include <cstddef>
#include <cstdint>

#include "codec/filters/loop_restoration/restoration_common.h"

namespace av2enc::restoration {

template <typename Sample>
struct LrCorePlaneT {
  const Sample* curr = nullptr;  // CurrFrame[plane], pre-CDEF
  int curr_stride = 0;
  const Sample* cdef = nullptr;  // CdefFrame[plane], post-CDEF/CCSO
  int cdef_stride = 0;
  Sample* dst = nullptr;  // LrFrame[plane], output (must not alias)
  int dst_stride = 0;

  // LrType[plane]: unit_rows x unit_cols entries, row-major.
  const std::uint8_t* unit_type = nullptr;
  int unit_cols = 0;
  int unit_rows = 0;

  // LrWienerNs[plane][unitRow][unitCol][coeff], stride kWienerNsCoeffStride.
  const std::int16_t* unit_coeffs = nullptr;

  // FrameLrWienerNs[plane][class][coeff], stride kWienerNsCoeffStride, used
  // when frame_filters_on[plane] is set.
  const std::int16_t* frame_coeffs = nullptr;

  // Optional §7.20.3/§7.20.4 class map.  `filter_class_grid` is FilterClass on
  // the luma MI grid; `subclass_lookup` maps it to a coefficient class.
  // `subclass_grid` is the legacy shortcut: already-looked-up subclasses.
  const std::uint8_t* filter_class_grid = nullptr;
  const std::uint8_t* subclass_lookup = nullptr;
};

template <typename Sample>
struct LrCoreFrameT {
  LrCorePlaneT<Sample> plane[3];
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;
  int bit_depth = 8;
  int num_planes = 3;

  // LoopRestorationSize[plane] and FrameRestorationType[plane].
  int unit_size[3] = {64, 64, 64};
  int frame_type[3] = {kRestoreNone, kRestoreNone, kRestoreNone};
  int frame_filters_on[3] = {0, 0, 0};
  int num_filter_classes = 1;

  // FilterClass / SubclassLookup inputs, luma MI grid.  For frame-level luma
  // filters with num_filter_classes > 1:
  //   * if filter_class_grid != nullptr, subcls = subclass_lookup[cls];
  //   * else the legacy subclass_grid is used directly.
  // `filter_class_count` is the number of distinct FilterClass values the grid
  // can hold (256 for the §7.20.4 classifier; 16 is the historical default).
  const std::uint8_t* filter_class_grid = nullptr;
  const std::uint8_t* subclass_lookup = nullptr;
  const std::uint8_t* subclass_grid = nullptr;
  int filter_class_count = kWienerNsClasses;

  const std::uint8_t* segment_ids = nullptr;
  const std::uint8_t* chroma_segment_ids = nullptr;
  const std::uint8_t* lossless = nullptr;

  const std::uint16_t* mi_col_start = nullptr;
  const std::uint16_t* mi_col_end = nullptr;
  const std::uint16_t* mi_row_start = nullptr;
  const std::uint16_t* mi_row_end = nullptr;
  int disable_loopfilters_across_tiles = 0;

  int use_bru = 0;
  const std::uint8_t* bru_modes = nullptr;
  int sb_shift = 4;  // Mi_Width_Log2[SbSize]

  int cfl_ds_filter_index = 0;
};

using LrCorePlane = LrCorePlaneT<std::uint8_t>;
using LrCoreFrame = LrCoreFrameT<std::uint8_t>;
using LrCorePlane16 = LrCorePlaneT<std::uint16_t>;
using LrCoreFrame16 = LrCoreFrameT<std::uint16_t>;

// Geometry helpers -----------------------------------------------------------

template <typename Sample>
AV2ENC_LR_HD int lr_plane_width_t(const LrCoreFrameT<Sample>& f, int plane) {
  return (f.mi_cols * kMiSize) >> (plane == 0 ? 0 : f.sub_x);
}

template <typename Sample>
AV2ENC_LR_HD int lr_plane_height_t(const LrCoreFrameT<Sample>& f, int plane) {
  return (f.mi_rows * kMiSize) >> (plane == 0 ? 0 : f.sub_y);
}

AV2ENC_LR_HD int lr_plane_width(const LrCoreFrame& f, int plane) {
  return lr_plane_width_t(f, plane);
}

AV2ENC_LR_HD int lr_plane_height(const LrCoreFrame& f, int plane) {
  return lr_plane_height_t(f, plane);
}

// §5.20.10.4 count_units_in_frame().
AV2ENC_LR_HD int lr_count_units(int unit_size, int frame_size) {
  if (unit_size <= 0) {
    return 1;
  }
  const int n = (frame_size + (unit_size >> 1)) / unit_size;
  return n > 1 ? n : 1;
}

// Per-4x4-block context produced by §7.20.1 loop_restore_block_process.
struct LrBlock {
  int x = 0;               // block origin in plane samples
  int y = 0;
  int w = 0;               // block extent in plane samples (2 for 4:2:0 chroma)
  int h = 0;
  int unit_row = 0;        // frame-global restoration unit index
  int unit_col = 0;
  int type = kRestoreNone; // LrType[plane][unitRow][unitCol]
  int luma_start_x = 0;
  int luma_start_y = 0;
  int luma_end_x = 0;
  int luma_end_y = 0;
  int stripe_start_y = 0;  // LumaStripeStartY (luma samples)
  int stripe_end_y = 0;    // LumaStripeEndY
};

// Computes the §7.20.1 context for one 4x4 luma block of one plane.  Returns
// false when the block is skipped entirely (lossless segment, BRU-inactive
// region, or FrameRestorationType == RESTORE_NONE for the plane).
template <typename Sample>
AV2ENC_LR_HD bool lr_block_context_t(const LrCoreFrameT<Sample>& f, int plane,
                                    int row, int col, LrBlock* out) {
  const int sub_x = (plane == 0) ? 0 : f.sub_x;
  const int sub_y = (plane == 0) ? 0 : f.sub_y;
  const int frame_type = f.frame_type[plane];
  if (out == nullptr) {
    return false;
  }
  *out = LrBlock{};
  if (frame_type == kRestoreNone) {
    return false;
  }

  const int unit_size = f.unit_size[plane];
  if (unit_size <= 0) {
    return false;
  }

  // Lossless segments are not restored.
  if (f.lossless != nullptr) {
    const int seg_id =
        (plane > 0) ? (f.chroma_segment_ids != nullptr
                           ? f.chroma_segment_ids[row * f.mi_cols + col]
                           : 0)
                    : (f.segment_ids != nullptr
                           ? f.segment_ids[row * f.mi_cols + col]
                           : 0);
    if (seg_id < kMaxSegments && f.lossless[seg_id] != 0) {
      return false;
    }
  }

  int mi_col_start = 0;
  int mi_row_start = 0;
  int mi_col_end = f.mi_cols;
  int mi_row_end = f.mi_rows;
  if (f.mi_col_start != nullptr && f.mi_col_end != nullptr &&
      f.mi_row_start != nullptr && f.mi_row_end != nullptr) {
    const int idx = row * f.mi_cols + col;
    mi_col_start = static_cast<int>(f.mi_col_start[idx]);
    mi_col_end = static_cast<int>(f.mi_col_end[idx]);
    mi_row_start = static_cast<int>(f.mi_row_start[idx]);
    mi_row_end = static_cast<int>(f.mi_row_end[idx]);
  }
  if (mi_col_end <= mi_col_start || mi_row_end <= mi_row_start) {
    return false;
  }

  const int lr_row_offset = (mi_row_start * kMiSize >> sub_y) / unit_size;
  const int lr_col_offset = (mi_col_start * kMiSize >> sub_x) / unit_size;

  const int stripe_row_a = ((row + 2) >> 4) << 4;
  const int stripe_row_b = mi_row_end - 1;
  const int stripe_row =
      stripe_row_a < stripe_row_b ? stripe_row_a : stripe_row_b;

  // §7.20.1: skip BRU-inactive superblock regions when use_bru is set.
  if (f.use_bru != 0 && f.bru_modes != nullptr) {
    const int bru_row = (stripe_row >> f.sb_shift) << f.sb_shift;
    const int bru_col = (col >> f.sb_shift) << f.sb_shift;
    if (f.bru_modes[bru_row * f.mi_cols + bru_col] != kBruActive) {
      return false;
    }
  }

  const int rel_row = row - mi_row_start;
  const int rel_col = col - mi_col_start;
  const int mi_cols = mi_col_end - mi_col_start;
  const int mi_rows = mi_row_end - mi_row_start;
  const int luma_y = rel_row * kMiSize;
  const int stripe_num = (luma_y + 8) / 64;

  const int unit_rows = lr_count_units(unit_size, (mi_rows * kMiSize) >> sub_y);
  const int unit_cols = lr_count_units(unit_size, (mi_cols * kMiSize) >> sub_x);

  const int unit_row_local = (rel_row * kMiSize + 8) >> sub_y;
  const int unit_col_local = (rel_col * kMiSize) >> sub_x;
  const int unit_row =
      lr_row_offset + ((unit_row_local / unit_size) < unit_rows - 1
                           ? (unit_row_local / unit_size)
                           : unit_rows - 1);
  const int unit_col =
      lr_col_offset + ((unit_col_local / unit_size) < unit_cols - 1
                           ? (unit_col_local / unit_size)
                           : unit_cols - 1);

  // Allowed filtering extent.
  int luma_start_x = 0;
  int luma_start_y = 0;
  int luma_end_x = f.mi_cols * kMiSize - 1;
  int luma_end_y = f.mi_rows * kMiSize - 1;
  if (f.disable_loopfilters_across_tiles != 0) {
    luma_start_x = mi_col_start * kMiSize;
    luma_end_x = mi_col_end * kMiSize - 1;
    luma_start_y = mi_row_start * kMiSize;
    luma_end_y = mi_row_end * kMiSize - 1;
  }

  const int unclipped_stripe_start =
      mi_row_start * kMiSize + stripe_num * 64 - 8;
  const int unclipped_stripe_end = unclipped_stripe_start + 64;

  out->x = (col * kMiSize) >> sub_x;
  out->y = (row * kMiSize) >> sub_y;
  out->w = kMiSize >> sub_x;
  out->h = kMiSize >> sub_y;
  out->unit_row = unit_row;
  out->unit_col = unit_col;
  out->luma_start_x = luma_start_x;
  out->luma_start_y = luma_start_y;
  out->luma_end_x = luma_end_x;
  out->luma_end_y = luma_end_y;
  out->stripe_start_y = luma_start_y > unclipped_stripe_start
                            ? luma_start_y
                            : unclipped_stripe_start;
  out->stripe_end_y = luma_end_y < unclipped_stripe_end - 1
                          ? luma_end_y
                          : unclipped_stripe_end - 1;

  if (f.plane[plane].unit_type != nullptr) {
    const int rows = f.plane[plane].unit_rows;
    const int cols = f.plane[plane].unit_cols;
    if (rows <= 0 || cols <= 0) {
      return false;
    }
    if (unit_row < 0 || unit_row >= rows || unit_col < 0 || unit_col >= cols) {
      return false;
    }
    out->type = static_cast<int>(f.plane[plane].unit_type[unit_row * cols +
                                                         unit_col]);
  } else {
    // No per-unit grid supplied: the frame type applies to every unit.
    out->type = (frame_type == kRestoreSwitchable) ? kRestoreNone : frame_type;
  }
  return true;
}

AV2ENC_LR_HD bool lr_block_context(const LrCoreFrame& f, int plane, int row,
                                  int col, LrBlock* out) {
  return lr_block_context_t(f, plane, row, col, out);
}

// §7.20.2 get_source_sample().
template <typename Sample>
AV2ENC_LR_HD int lr_get_source_sample_t(const LrCoreFrameT<Sample>& f,
                                        const LrBlock& b, int plane, int x,
                                        int y) {
  const int sub_x = (plane == 0) ? 0 : f.sub_x;
  const int sub_y = (plane == 0) ? 0 : f.sub_y;
  const LrCorePlaneT<Sample>& p = f.plane[plane];

  x = clip3(b.luma_start_x >> sub_x, b.luma_end_x >> sub_x, x);
  y = clip3(b.luma_start_y >> sub_y, b.luma_end_y >> sub_y, y);

  const int stripe_start = b.stripe_start_y >> sub_y;
  const int stripe_end = b.stripe_end_y >> sub_y;
  if (y < stripe_start) {
    const int yy = (stripe_start - 2 > y) ? stripe_start - 2 : y;
    return static_cast<int>(p.curr[yy * p.curr_stride + x]);
  }
  if (y > stripe_end) {
    const int yy = (stripe_end + 2 < y) ? stripe_end + 2 : y;
    return static_cast<int>(p.curr[yy * p.curr_stride + x]);
  }
  return static_cast<int>(p.cdef[y * p.cdef_stride + x]);
}

AV2ENC_LR_HD int lr_get_source_sample(const LrCoreFrame& f, const LrBlock& b,
                                      int plane, int x, int y) {
  return lr_get_source_sample_t(f, b, plane, x, y);
}

// §7.20.3 get_luma_source_sample() — the §7.20.2 source fetch for the luma
// plane, used only by the chroma-from-luma taps.
//
// One pinned-decoder detail is load-bearing here.  §7.20.2 clamps an
// above-stripe request with Max(LumaStripeStartY - 2, y), so the first line
// above the stripe is returned unchanged.  dav2d f9dafbc instead reads
// LumaStripeStartY - 2 for *every* above-stripe luma request when the chroma
// plane is not vertically subsampled (SubsamplingY == 0, i.e. 4:2:2 and
// 4:4:4): its CfL row loop backs up both "lines above" through a helper that
// ignores the source stride for that layout, so both rows alias the same
// line.  The two forms only differ for the first line above the stripe, and a
// 4:2:0 chroma plane never asks for it (its vertical downsampling pins the
// taps to the stripe start), which is why 4:2:0 was already bit-exact.
// Matching the decoder here keeps the CfL tap set exact at 4:2:2/4:4:4.
template <typename Sample>
AV2ENC_LR_HD int lr_get_luma_source_sample_t(const LrCoreFrameT<Sample>& f,
                                             const LrBlock& b, int x, int y) {
  const LrCorePlaneT<Sample>& p = f.plane[0];
  x = clip3(b.luma_start_x, b.luma_end_x, x);
  y = clip3(b.luma_start_y, b.luma_end_y, y);
  if (y < b.stripe_start_y) {
    int yy = b.stripe_start_y - 2;
    if (f.sub_y == 0) {
      // Both lines above the stripe alias the second one.
      if (yy < b.luma_start_y) yy = b.luma_start_y;
    } else if (yy < y) {
      yy = y;
    }
    return static_cast<int>(p.curr[yy * p.curr_stride + x]);
  }
  if (y > b.stripe_end_y) {
    const int yy = (b.stripe_end_y + 2 < y) ? b.stripe_end_y + 2 : y;
    return static_cast<int>(p.curr[yy * p.curr_stride + x]);
  }
  return static_cast<int>(p.cdef[y * p.cdef_stride + x]);
}

// §7.20.3 get_luma_sample() — subsampled luma fetch with the CfL downsampling
// filter selected by cfl_ds_filter_index (3 aliases to 0).
template <typename Sample>
AV2ENC_LR_HD int lr_get_luma_sample_t(const LrCoreFrameT<Sample>& f,
                                      const LrBlock& b, int x, int y) {
  const int sub_x = f.sub_x;
  const int sub_y = f.sub_y;
  const int last_y = f.mi_rows * kMiSize - 1 - sub_y;
  const int last_x = b.luma_end_x - sub_x;
  x <<= sub_x;
  y <<= sub_y;
  y = clip3(0, last_y, y);
  x = clip3(b.luma_start_x, last_x, x);
  int filter_idx = f.cfl_ds_filter_index;
  if (filter_idx == 3) {
    filter_idx = 0;
  }
  if (sub_x != 0 && sub_y != 0 && filter_idx >= 0 && filter_idx <= 1) {
    int t = 0;
    for (int dy = 0; dy < 2; ++dy) {
      for (int dx = 0; dx < 2; ++dx) {
        t += static_cast<int>(kWienerFilters420[filter_idx][dy][dx]) *
             lr_get_luma_source_sample_t(f, b, x + dx, y + dy);
      }
    }
    return t >> 2;
  }
  return lr_get_luma_source_sample_t(f, b, x, y);
}

AV2ENC_LR_HD int lr_get_luma_sample(const LrCoreFrame& f, const LrBlock& b,
                                    int x, int y) {
  return lr_get_luma_sample_t(f, b, x, y);
}

// §7.20.3 non-separable Wiener filter for one block.
template <typename Sample>
AV2ENC_LR_HD void wiener_ns_block_t(const LrCoreFrameT<Sample>& f,
                                    const LrBlock& b, int plane) {
  const LrCorePlaneT<Sample>& p = f.plane[plane];
  const int n_taps = (plane == 0) ? kWienerNsTapsY : kWienerNsTapsUv;
  const std::int8_t(*config)[3] =
      (plane == 0) ? kWienerNsConfigY : kWienerNsConfigUv;
  const bool frame_filters = f.frame_filters_on[plane] != 0;
  const int unit_index =
      b.unit_row * (p.unit_cols > 0 ? p.unit_cols : 1) + b.unit_col;
  const std::int16_t* unit_off =
      (p.unit_coeffs != nullptr && p.unit_cols > 0)
          ? p.unit_coeffs + unit_index * kWienerNsCoeffStride
          : nullptr;

  for (int r = 0; r < b.h; ++r) {
    for (int c = 0; c < b.w; ++c) {
      const int sx = b.x + c;
      const int sy = b.y + r;
      const int m = lr_get_source_sample_t(f, b, plane, sx, sy);
      int s = m << kWienerNsPrecBits;

      int subcls = 0;
      if (plane == 0 && f.frame_filters_on[0] != 0 &&
          f.num_filter_classes > 1) {
        const int cell = (sy >> kMiSizeLog2) * f.mi_cols +
                         (sx >> kMiSizeLog2);
        if (f.filter_class_grid != nullptr) {
          int cls = f.filter_class_grid[cell];
          const int ncls = f.filter_class_count > 0 ? f.filter_class_count
                                                    : kWienerNsClasses;
          if (cls < 0 || cls >= ncls) {
            cls = 0;
          }
          subcls = f.subclass_lookup != nullptr ? f.subclass_lookup[cls] : cls;
        } else if (f.subclass_grid != nullptr) {
          subcls = f.subclass_grid[cell];
        }
        if (subcls < 0 || subcls >= f.num_filter_classes) {
          subcls = 0;
        }
      }
      const std::int16_t* frame_off =
          p.frame_coeffs != nullptr
              ? p.frame_coeffs + subcls * kWienerNsCoeffStride
              : nullptr;

      for (int i = 0; i < n_taps; ++i) {
        const int dy = config[i][0];
        const int dx = config[i][1];
        const int idx = config[i][2];
        const int diff =
            lr_get_source_sample_t(f, b, plane, sx + dx, sy + dy) - m;
        const int coeff = frame_filters
                              ? (frame_off != nullptr ? frame_off[idx] : 0)
                              : (unit_off != nullptr ? unit_off[idx] : 0);
        s += diff * coeff;
      }

      if (plane > 0) {
        const int m_luma = lr_get_luma_sample_t(f, b, sx, sy);
        for (int i = 0; i < n_taps; ++i) {
          const int coeff =
              frame_filters
                  ? (frame_off != nullptr
                         ? frame_off[i + kWienerNsShortCoeffs]
                         : 0)
                  : (unit_off != nullptr
                         ? unit_off[i + kWienerNsShortCoeffs]
                         : 0);
          if (coeff != 0) {
            const int dy = config[i][0];
            const int dx = config[i][1];
            const int luma_diff =
                lr_get_luma_sample_t(f, b, sx + dx, sy + dy) - m_luma;
            s += luma_diff * coeff;
          }
        }
      }

      const int v = round2(s, kWienerNsPrecBits);
      p.dst[sy * p.dst_stride + sx] =
          static_cast<Sample>(clip3(0, (1 << f.bit_depth) - 1, v));
    }
  }
}

AV2ENC_LR_HD void wiener_ns_block(const LrCoreFrame& f, const LrBlock& b,
                                  int plane) {
  wiener_ns_block_t(f, b, plane);
}

// Copies CdefFrame into LrFrame for one plane (§7.20 first step).
template <typename Sample>
inline void lr_copy_plane_t(const LrCoreFrameT<Sample>& f, int plane) {
  const LrCorePlaneT<Sample>& p = f.plane[plane];
  const int width = lr_plane_width_t(f, plane);
  const int height = lr_plane_height_t(f, plane);
  for (int y = 0; y < height; ++y) {
    const Sample* src =
        p.cdef + static_cast<std::ptrdiff_t>(y) * p.cdef_stride;
    Sample* dst = p.dst + static_cast<std::ptrdiff_t>(y) * p.dst_stride;
    for (int x = 0; x < width; ++x) {
      dst[x] = src[x];
    }
  }
}

inline void lr_copy_plane(const LrCoreFrame& f, int plane) {
  lr_copy_plane_t(f, plane);
}

// Runs the §7.20 block loop for one plane (CPU reference path).  Returns the
// number of blocks that were filtered.
template <typename Sample>
inline int lr_plane_t(const LrCoreFrameT<Sample>& f, int plane) {
  if (f.frame_type[plane] == kRestoreNone) {
    return 0;
  }
  int filtered = 0;
  for (int row = 0; row < f.mi_rows; ++row) {
    for (int col = 0; col < f.mi_cols; ++col) {
      LrBlock b;
      if (!lr_block_context_t(f, plane, row, col, &b)) {
        continue;
      }
      if (b.type == kRestoreWienerNs) {
        wiener_ns_block_t(f, b, plane);
        ++filtered;
      }
    }
  }
  return filtered;
}

inline int lr_plane(const LrCoreFrame& f, int plane) {
  return lr_plane_t(f, plane);
}

}  // namespace av2enc::restoration

#endif  // AV2ENC_ENCODER_FILTERS_RESTORATION_WIENER_CORE_H_
