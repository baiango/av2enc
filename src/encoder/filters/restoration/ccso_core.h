// AV2 §7.19 CCSO (cross-component sample offset) — shared host/device core.
//
// The CPU reference (restoration.cpp) and the CUDA kernel
// (src/cuda/restoration/ccso.cu) call the same inline functions in this
// header, so results are bit-exact by construction for identical inputs.
//
// Normative process (§7.19 / §7.19.1):
//
//   for plane in {Y,U,V}:
//     if ccso_planes[plane]:
//       for every (1<<CcsoLumaSizeLog2) luma-aligned block in the plane:
//         if CcsoBlks[plane][unitRow][unitCol]:
//           for every sample in the block, unless the containing superblock
//           region is a BRU-inactive region or the segment is lossless:
//             band = CurrFrame[0][luma] >> (BitDepth - maxBandLog2)
//             cls0 = ccso_score(luma(+dy,+dx) - c)
//             cls1 = ccso_score(luma(-dy,-dx) - c)
//             CdefFrame[plane][y][x] += CcsoFilterOffset[plane][band][cls0][cls1]
//
// Notes on the reference frame:
//   * `get_ccso_luma()` always reads CurrFrame (pre-CDEF, deblocked) luma,
//     never CdefFrame, and clamps to the allowed (tile) extent.
//   * CdefFrame is modified in place and is the only written array.
//
// Two spec details that are easy to get wrong:
//   * The positive/negative neighbourhood taps are taken in *luma* samples
//     (yLuma = y3 << subY) even when filtering chroma.
//   * When ccso_bo_only is set the filter is band-offset only: both classes
//     are forced to 0 (and only offsets[band][0][0] is signalled).

#ifndef AV2ENC_ENCODER_FILTERS_RESTORATION_CCSO_CORE_H_
#define AV2ENC_ENCODER_FILTERS_RESTORATION_CCSO_CORE_H_

#include <cstdint>

#include "encoder/filters/restoration/restoration_common.h"

namespace av2enc::restoration {

// One plane's CCSO parameters (parsed by §5.18.7.12) plus its plane views.
struct CcsoCorePlane {
  bool enabled = false;  // ccso_planes[plane]
  const std::uint8_t* curr_luma = nullptr;  // CurrFrame[0], pre-CDEF
  int curr_luma_stride = 0;
  std::uint8_t* dst = nullptr;  // CdefFrame[plane], post-CDEF, modified
  int dst_stride = 0;

  // CcsoBlks[plane]: one byte per luma CCSO unit, row-major,
  // blk_rows x blk_cols entries.
  const std::uint8_t* blks = nullptr;
  int blk_cols = 0;
  int blk_rows = 0;

  int scale_idx = 0;      // ccso_scale_idx[plane]
  int quant_idx = 0;      // ccso_quant_idx[plane]
  int ext_filter = 0;     // ccso_ext_filter[plane]
  int edge_clf = 0;       // ccso_edge_clf[plane]
  int max_band_log2 = 0;  // ccso_max_band_log2[plane]
  bool bo_only = false;   // ccso_bo_only[plane]

  // CcsoFilterOffset[plane][band][cls0][cls1] flattened by
  // ccso_offset_index(); kCcsoOffsetEntries entries.
  const std::int16_t* offsets = nullptr;
};

struct CcsoCoreFrame {
  CcsoCorePlane plane[3];
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;
  int bit_depth = 8;
  int ccso_luma_size_log2 = 6;
  // Superblock size in luma samples; used only to re-derive the BruModes
  // lookup position (sbBlkW/sbBlkH in the spec pseudocode).
  int sb_width = 64;
  int sb_height = 64;

  // Optional per-frame metadata, all luma MI grid (mi_cols x mi_rows).
  const std::uint8_t* bru_modes = nullptr;  // null => all BRU_ACTIVE
  const std::uint8_t* segment_ids = nullptr;
  const std::uint8_t* chroma_segment_ids = nullptr;
  const std::uint8_t* lossless = nullptr;  // kMaxSegments entries

  // Optional MiColStart/End, MiRowStart/End grids (luma MI units).  Null means
  // "single tile / whole frame".  Only consulted when
  // disable_loopfilters_across_tiles is set.
  const std::uint16_t* mi_col_start = nullptr;
  const std::uint16_t* mi_col_end = nullptr;
  const std::uint16_t* mi_row_start = nullptr;
  const std::uint16_t* mi_row_end = nullptr;
  int disable_loopfilters_across_tiles = 0;
};

// Plane extents in samples, matching the spec's planeWidth/planeHeight.
AV2ENC_LR_HD int ccso_plane_width(const CcsoCoreFrame& f, int plane) {
  return (f.mi_cols * kMiSize) >> (plane == 0 ? 0 : f.sub_x);
}

AV2ENC_LR_HD int ccso_plane_height(const CcsoCoreFrame& f, int plane) {
  return (f.mi_rows * kMiSize) >> (plane == 0 ? 0 : f.sub_y);
}

// §7.19.1 get_ccso_luma(): clamp into the allowed extent, then read CurrFrame.
AV2ENC_LR_HD int ccso_get_luma(const CcsoCoreFrame& f, int x, int y,
                               int luma_start_x, int luma_end_x,
                               int luma_start_y, int luma_end_y) {
  x = clip3(luma_start_x, luma_end_x, x);
  y = clip3(luma_start_y, luma_end_y, y);
  return f.plane[0].curr_luma[y * f.plane[0].curr_luma_stride + x];
}

// Applies CCSO to one sample of one plane.  Returns true when the sample was
// visited by the normative process (i.e. inside an enabled, non-lossless,
// non-BRU-inactive area); the caller can use the return value for coverage
// diagnostics.
AV2ENC_LR_HD bool ccso_apply_sample(const CcsoCoreFrame& f, int plane, int y3,
                                    int x3) {
  const CcsoCorePlane& p = f.plane[plane];
  if (!p.enabled || p.offsets == nullptr || p.blks == nullptr) {
    return false;
  }

  const int sub_x = (plane == 0) ? 0 : f.sub_x;
  const int sub_y = (plane == 0) ? 0 : f.sub_y;
  const int shift_x = f.ccso_luma_size_log2 - sub_x;
  const int shift_y = f.ccso_luma_size_log2 - sub_y;
  if (shift_x < 0 || shift_y < 0) {
    return false;
  }
  const int blk_w = 1 << shift_x;
  const int blk_h = 1 << shift_y;

  const int x = (x3 / blk_w) * blk_w;
  const int y = (y3 / blk_h) * blk_h;
  const int unit_col = x >> shift_x;
  const int unit_row = y >> shift_y;
  if (unit_col < 0 || unit_row < 0 || unit_col >= p.blk_cols ||
      unit_row >= p.blk_rows) {
    return false;
  }
  if (p.blks[unit_row * p.blk_cols + unit_col] == 0) {
    return false;
  }

  // The spec re-tests BruModes once per superblock-aligned sub-block of the
  // CCSO unit (y2/x2).  BruModes is indexed on the luma MI grid.
  const int sb_blk_w = f.sb_width >> sub_x;
  const int sb_blk_h = f.sb_height >> sub_y;
  if (sb_blk_w <= 0 || sb_blk_h <= 0) {
    return false;
  }
  const int x2 = x + ((x3 - x) / sb_blk_w) * sb_blk_w;
  const int y2 = y + ((y3 - y) / sb_blk_h) * sb_blk_h;
  const int mi_row = (y2 << sub_y) >> kMiSizeLog2;
  const int mi_col = (x2 << sub_x) >> kMiSizeLog2;
  if (f.bru_modes != nullptr &&
      f.bru_modes[mi_row * f.mi_cols + mi_col] != kBruActive) {
    return false;
  }

  const int row = (y3 << sub_y) >> kMiSizeLog2;
  const int col = (x3 << sub_x) >> kMiSizeLog2;
  if (row < 0 || col < 0 || row >= f.mi_rows || col >= f.mi_cols) {
    return false;
  }
  const int seg_id =
      (plane > 0) ? (f.chroma_segment_ids != nullptr
                         ? f.chroma_segment_ids[row * f.mi_cols + col]
                         : 0)
                  : (f.segment_ids != nullptr
                         ? f.segment_ids[row * f.mi_cols + col]
                         : 0);
  if (f.lossless != nullptr && seg_id < kMaxSegments &&
      f.lossless[seg_id] != 0) {
    return false;
  }

  // Allowed filtering extent (§7.19.1): tile extents when the stream disables
  // loop filters across tiles, otherwise the whole frame.
  int luma_start_x = 0;
  int luma_start_y = 0;
  int luma_end_x = f.mi_cols * kMiSize - 1;
  int luma_end_y = f.mi_rows * kMiSize - 1;
  if (f.disable_loopfilters_across_tiles != 0 &&
      f.mi_col_start != nullptr && f.mi_col_end != nullptr &&
      f.mi_row_start != nullptr && f.mi_row_end != nullptr) {
    const int idx = row * f.mi_cols + col;
    luma_start_x = static_cast<int>(f.mi_col_start[idx]) * kMiSize;
    luma_end_x = static_cast<int>(f.mi_col_end[idx]) * kMiSize - 1;
    luma_start_y = static_cast<int>(f.mi_row_start[idx]) * kMiSize;
    luma_end_y = static_cast<int>(f.mi_row_end[idx]) * kMiSize - 1;
  }

  const int x_luma = x3 << sub_x;
  const int y_luma = y3 << sub_y;
  const int c = ccso_get_luma(f, x_luma, y_luma, luma_start_x, luma_end_x,
                              luma_start_y, luma_end_y);
  const int shift = f.bit_depth - p.max_band_log2;
  if (shift < 0 || shift > 24) {
    return false;
  }
  const int band = c >> shift;

  int cls0 = 0;
  int cls1 = 0;
  if (!p.bo_only) {
    const int quant_step = kCcsoQuantSz[p.scale_idx & 3][p.quant_idx & 3];
    const int ext = (p.ext_filter >= 0 && p.ext_filter < 7) ? p.ext_filter : 0;
    const int ey = kCcsoPos[ext][0];
    const int ex = kCcsoPos[ext][1];
    const int plus = ccso_get_luma(f, x_luma + ex, y_luma + ey, luma_start_x,
                                   luma_end_x, luma_start_y, luma_end_y);
    const int minus = ccso_get_luma(f, x_luma - ex, y_luma - ey, luma_start_x,
                                    luma_end_x, luma_start_y, luma_end_y);
    cls0 = ccso_score(plus - c, quant_step, p.edge_clf);
    cls1 = ccso_score(minus - c, quant_step, p.edge_clf);
  }

  const int idx = ccso_offset_index(band, cls0, cls1);
  const int base = static_cast<int>(p.dst[y3 * p.dst_stride + x3]);
  const int out = base + static_cast<int>(p.offsets[idx]);
  p.dst[y3 * p.dst_stride + x3] =
      static_cast<std::uint8_t>(clip3(0, (1 << f.bit_depth) - 1, out));
  return true;
}

// Runs the whole normative per-plane loop for one plane.  The CPU reference
// calls this; the CUDA kernel maps a thread to each (y3,x3) instead, using
// ccso_apply_sample directly.
inline int ccso_plane(const CcsoCoreFrame& f, int plane) {
  const int width = ccso_plane_width(f, plane);
  const int height = ccso_plane_height(f, plane);
  int visited = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (ccso_apply_sample(f, plane, y, x)) {
        ++visited;
      }
    }
  }
  return visited;
}

}  // namespace av2enc::restoration

#endif  // AV2ENC_ENCODER_FILTERS_RESTORATION_CCSO_CORE_H_
