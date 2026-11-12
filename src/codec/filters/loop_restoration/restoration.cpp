// AV2 §7.19 CCSO and §7.20 loop restoration — CPU reference implementation.
//
// This is the correctness oracle: scalar, single-threaded, no SIMD.  It shares
// the normative arithmetic with the CUDA kernels through ccso_core.h and
// wiener_core.h.

#include "codec/filters/loop_restoration/restoration.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace av2enc::restoration {
namespace {

bool plane_valid(const RestorationPlane& p, int width, int height) {
  return p.data != nullptr && p.stride >= width && height > 0;
}

RestorationStatus validate_frame(const RestorationFrame& f) {
  if (f.mi_cols <= 0 || f.mi_rows <= 0) {
    return RestorationStatus::kInvalidArgument;
  }
  if (f.bit_depth != 8) {
    return RestorationStatus::kUnsupportedBitDepth;
  }
  if (f.subsampling_x != 1 || f.subsampling_y != 1) {
    return RestorationStatus::kUnsupportedSubsampling;
  }
  if (f.sb_width <= 0 || f.sb_height <= 0 || f.sb_shift < 0 || f.sb_shift > 8) {
    return RestorationStatus::kInvalidArgument;
  }
  if (f.cfl_ds_filter_index < 0 || f.cfl_ds_filter_index > 3) {
    return RestorationStatus::kInvalidArgument;
  }
  for (int p = 0; p < 3; ++p) {
    const int w = restoration_plane_width(f, p);
    const int h = restoration_plane_height(f, p);
    if (!plane_valid(f.curr[p], w, h) || !plane_valid(f.cdef[p], w, h)) {
      return RestorationStatus::kInvalidArgument;
    }
    if (f.disable_loopfilters_across_tiles != 0) {
      if (f.mi_col_start == nullptr || f.mi_col_end == nullptr ||
          f.mi_row_start == nullptr || f.mi_row_end == nullptr) {
        return RestorationStatus::kInvalidArgument;
      }
    }
  }
  if (f.lossless != nullptr) {
    // kMaxSegments entries are read; nothing to validate beyond the pointer.
  }
  return RestorationStatus::kOk;
}

bool ccso_params_valid(const CcsoParams& p) {
  if (p.enabled && p.bo_only) {
    // bo_only forces quant_idx/ext_filter/edge_clf to zero, but callers may
    // leave either value; only the ranges that would be dereferenced matter.
    return p.scale_idx >= 0 && p.scale_idx <= 3 && p.max_band_log2 >= 0 &&
           p.max_band_log2 <= 6;
  }
  if (!p.enabled) {
    return true;
  }
  return p.scale_idx >= 0 && p.scale_idx <= 3 && p.quant_idx >= 0 &&
         p.quant_idx <= 3 && p.ext_filter >= 0 && p.ext_filter < 7 &&
         p.edge_clf >= 0 && p.edge_clf <= 2 && p.max_band_log2 >= 0 &&
         p.max_band_log2 <= 6;
}

}  // namespace

const char* restoration_status_string(RestorationStatus status) {
  switch (status) {
    case RestorationStatus::kOk:
      return "ok";
    case RestorationStatus::kInvalidArgument:
      return "invalid argument";
    case RestorationStatus::kUnsupportedBitDepth:
      return "unsupported bit depth (8-bit only)";
    case RestorationStatus::kUnsupportedSubsampling:
      return "unsupported chroma subsampling (4:2:0 only)";
    case RestorationStatus::kUnsupportedPcWiener:
      return "PC-Wiener needs pc_wiener_*.h tables (not in the spec text)";
    case RestorationStatus::kUnsupportedGdf:
      return "GDF needs gdf_*.h tables (not in the spec text)";
    case RestorationStatus::kUnsupportedCcsoUnitSize:
      return "reserved (CCSO units smaller than the superblock are legal)";
    case RestorationStatus::kMissingSubclassGrid:
      return "frame_filters_on needs a §7.20.4 subclass grid";
    case RestorationStatus::kNonConformingUnitSize:
      return "LoopRestorationSize violates §6.17.7.7 (power of two, >= stripe, "
             "tile-aligned)";
  }
  return "unknown";
}

int restoration_plane_width(const RestorationFrame& frame, int plane) {
  return (frame.mi_cols * kMiSize) >> (plane == 0 ? 0 : frame.subsampling_x);
}

int restoration_plane_height(const RestorationFrame& frame, int plane) {
  return (frame.mi_rows * kMiSize) >> (plane == 0 ? 0 : frame.subsampling_y);
}

int restoration_count_units(int unit_size, int frame_size) {
  return lr_count_units(unit_size, frame_size);
}

// ---------------------------------------------------------------------------
// CCSO
// ---------------------------------------------------------------------------

int ccso_offset_count(const CcsoParams& params) {
  if (params.max_band_log2 < 0 || params.max_band_log2 > 6) {
    return 0;
  }
  const int bands = 1 << params.max_band_log2;
  const int interval =
      params.bo_only ? 1 : (kCcsoInputInterval - params.edge_clf);
  if (interval <= 0) {
    return 0;
  }
  return bands * interval * interval;
}

bool ccso_header_conforms(const CcsoParams& params) {
  if (!params.enabled) {
    return true;
  }
  const int max_band =
      params.bo_only ? kCcsoMaxBandLog2BoOnly : kCcsoMaxBandLog2Normal;
  if (params.max_band_log2 < 0 || params.max_band_log2 > max_band) {
    return false;
  }
  if (!params.bo_only && params.edge_clf > kCcsoMaxEdgeClf) {
    return false;
  }
  if (!params.bo_only &&
      kCcsoQuantSz[params.scale_idx & 3][params.quant_idx & 3] == 0 &&
      params.edge_clf != 0) {
    // §5.18.7.12: quantStep == 0 forces ccso_edge_clf to 0.
    return false;
  }
  return true;
}

bool ccso_set_offsets(CcsoParams& params, const std::uint8_t* indices,
                      std::size_t count) {
  if (indices == nullptr || !ccso_params_valid(params)) {
    return false;
  }
  const int need = ccso_offset_count(params);
  if (need <= 0 || count != static_cast<std::size_t>(need)) {
    return false;
  }
  for (int i = 0; i < kCcsoOffsetEntries; ++i) {
    params.offsets[i] = 0;
  }
  const int bands = 1 << params.max_band_log2;
  const int interval =
      params.bo_only ? 1 : (kCcsoInputInterval - params.edge_clf);
  const int scale = params.scale_idx + 1;
  std::size_t k = 0;
  for (int d0 = 0; d0 < interval; ++d0) {
    for (int d1 = 0; d1 < interval; ++d1) {
      for (int band = 0; band < bands; ++band) {
        const std::uint8_t idx = indices[k++];
        if (idx >= kCcsoOffsetRange) {
          return false;
        }
        params.offsets[ccso_offset_index(band, d0, d1)] =
            static_cast<std::int16_t>(
                static_cast<int>(kCcsoOffset[idx]) * scale);
      }
    }
  }
  return true;
}

CcsoCoreFrame ccso_to_core(const RestorationFrame& frame,
                           const CcsoFrameParams& params) {
  CcsoCoreFrame core{};
  for (int p = 0; p < 3; ++p) {
    CcsoCorePlane& c = core.plane[p];
    c.curr_luma = frame.curr[0].data;
    c.curr_luma_stride = frame.curr[0].stride;
    c.dst = frame.cdef[p].data;
    c.dst_stride = frame.cdef[p].stride;
    c.blks = params.blks[p];
    c.blk_cols = params.blk_cols[p];
    c.blk_rows = params.blk_rows[p];
    c.enabled = params.plane[p].enabled;
    c.scale_idx = params.plane[p].scale_idx;
    c.quant_idx = params.plane[p].quant_idx;
    c.ext_filter = params.plane[p].ext_filter;
    c.edge_clf = params.plane[p].edge_clf;
    c.max_band_log2 = params.plane[p].max_band_log2;
    c.bo_only = params.plane[p].bo_only;
    c.offsets = params.plane[p].offsets;
  }
  core.mi_cols = frame.mi_cols;
  core.mi_rows = frame.mi_rows;
  core.sub_x = frame.subsampling_x;
  core.sub_y = frame.subsampling_y;
  core.bit_depth = frame.bit_depth;
  core.ccso_luma_size_log2 = params.ccso_luma_size_log2;
  core.sb_width = frame.sb_width;
  core.sb_height = frame.sb_height;
  core.bru_modes = frame.bru_modes;
  core.segment_ids = frame.segment_ids;
  core.chroma_segment_ids = frame.chroma_segment_ids;
  core.lossless = frame.lossless;
  core.mi_col_start = frame.mi_col_start;
  core.mi_col_end = frame.mi_col_end;
  core.mi_row_start = frame.mi_row_start;
  core.mi_row_end = frame.mi_row_end;
  core.disable_loopfilters_across_tiles =
      frame.disable_loopfilters_across_tiles;
  return core;
}

namespace {

// The CcsoBlks grid must cover every CCSO block of the plane, otherwise the
// filter would silently skip samples.
bool ccso_grid_covers(const RestorationFrame& frame,
                      const CcsoFrameParams& params, int plane) {
  if (!params.plane[plane].enabled) {
    return true;
  }
  const int sub_x = (plane == 0) ? 0 : frame.subsampling_x;
  const int sub_y = (plane == 0) ? 0 : frame.subsampling_y;
  int cols = 0;
  int rows = 0;
  ccso_blk_grid_size(restoration_plane_width(frame, plane),
                     restoration_plane_height(frame, plane),
                     params.ccso_luma_size_log2, sub_x, sub_y, &cols, &rows);
  return params.blk_cols[plane] >= cols && params.blk_rows[plane] >= rows;
}

}  // namespace

RestorationStatus ccso_validate(const RestorationFrame& frame,
                                const CcsoFrameParams& params) {
  const RestorationStatus st = validate_frame(frame);
  if (st != RestorationStatus::kOk) {
    return st;
  }
  if (params.ccso_luma_size_log2 < 6 || params.ccso_luma_size_log2 > 8) {
    return RestorationStatus::kInvalidArgument;
  }
  for (int p = 0; p < 3; ++p) {
    if (!ccso_params_valid(params.plane[p])) {
      return RestorationStatus::kInvalidArgument;
    }
    if (params.plane[p].enabled) {
      if (params.blks[p] == nullptr || params.blk_cols[p] <= 0 ||
          params.blk_rows[p] <= 0) {
        return RestorationStatus::kInvalidArgument;
      }
      if (!ccso_grid_covers(frame, params, p)) {
        return RestorationStatus::kInvalidArgument;
      }
    }
  }
  return RestorationStatus::kOk;
}

RestorationStatus ccso_plane_frame(const RestorationFrame& frame,
                                   const CcsoFrameParams& params, int plane) {
  if (plane < 0 || plane > 2) {
    return RestorationStatus::kInvalidArgument;
  }
  const RestorationStatus st = ccso_validate(frame, params);
  if (st != RestorationStatus::kOk) {
    return st;
  }
  const CcsoCoreFrame core = ccso_to_core(frame, params);
  ccso_plane(core, plane);
  return RestorationStatus::kOk;
}

RestorationStatus ccso_frame(const RestorationFrame& frame,
                             const CcsoFrameParams& params) {
  const RestorationStatus st = ccso_validate(frame, params);
  if (st != RestorationStatus::kOk) {
    return st;
  }
  const CcsoCoreFrame core = ccso_to_core(frame, params);
  for (int p = 0; p < 3; ++p) {
    if (params.plane[p].enabled) {
      ccso_plane(core, p);
    }
  }
  return RestorationStatus::kOk;
}

// ---------------------------------------------------------------------------
// Loop restoration
// ---------------------------------------------------------------------------

LrCoreFrame loop_restoration_to_core(const RestorationFrame& frame,
                                     const LoopRestorationParams& params) {
  LrCoreFrame core{};
  for (int p = 0; p < 3; ++p) {
    LrCorePlane& c = core.plane[p];
    c.curr = frame.curr[p].data;
    c.curr_stride = frame.curr[p].stride;
    c.cdef = frame.cdef[p].data;
    c.cdef_stride = frame.cdef[p].stride;
    c.dst = frame.lr[p].data;
    c.dst_stride = frame.lr[p].stride;
    c.unit_type = params.unit_type[p];
    c.unit_cols = params.unit_cols[p];
    c.unit_rows = params.unit_rows[p];
    c.unit_coeffs = params.unit_coeffs[p];
    c.frame_coeffs = params.frame_coeffs[p];
    c.filter_class_grid = params.filter_class_grid;
    c.subclass_lookup = params.subclass_lookup;
  }
  core.mi_cols = frame.mi_cols;
  core.mi_rows = frame.mi_rows;
  core.sub_x = frame.subsampling_x;
  core.sub_y = frame.subsampling_y;
  core.bit_depth = frame.bit_depth;
  for (int p = 0; p < 3; ++p) {
    core.unit_size[p] = params.unit_size[p];
    core.frame_type[p] = static_cast<int>(params.frame_type[p]);
    core.frame_filters_on[p] = params.frame_filters_on[p];
  }
  core.num_filter_classes = params.num_filter_classes;
  core.filter_class_grid = params.filter_class_grid;
  core.subclass_lookup = params.subclass_lookup;
  core.subclass_grid = params.subclass_grid;
  core.filter_class_count = params.filter_class_count;
  core.segment_ids = frame.segment_ids;
  core.chroma_segment_ids = frame.chroma_segment_ids;
  core.lossless = frame.lossless;
  core.mi_col_start = frame.mi_col_start;
  core.mi_col_end = frame.mi_col_end;
  core.mi_row_start = frame.mi_row_start;
  core.mi_row_end = frame.mi_row_end;
  core.disable_loopfilters_across_tiles =
      frame.disable_loopfilters_across_tiles;
  core.use_bru = frame.use_bru;
  core.bru_modes = frame.bru_modes;
  core.sb_shift = frame.sb_shift;
  core.cfl_ds_filter_index = frame.cfl_ds_filter_index;
  return core;
}

namespace {

bool frame_type_in_range(RestorationType t) {
  const int v = static_cast<int>(t);
  return v >= kRestoreNone && v <= kRestoreSwitchable;
}

bool grid_has_pc_wiener(const std::uint8_t* grid, int rows, int cols) {
  if (grid == nullptr) {
    return false;
  }
  const int n = rows * cols;
  for (int i = 0; i < n; ++i) {
    if (grid[i] == static_cast<std::uint8_t>(kRestorePcWiener)) {
      return true;
    }
  }
  return false;
}

bool is_power_of_two(int v) { return v > 0 && (v & (v - 1)) == 0; }

bool num_filter_classes_valid(int n) {
  for (int i = 0; i < 8; ++i) {
    if (kDecodeNumFilterClasses[i] == n) {
      return true;
    }
  }
  return false;
}

// §6.17.7.7 check_ru_size(): restoration units must not cross internal tile
// boundaries.  Tile boundaries are recovered from the per-MI tile grids; a
// null grid is a single tile.
bool check_ru_size(const RestorationFrame& frame,
                   const LoopRestorationParams& params) {
  const int sub = (frame.subsampling_x > frame.subsampling_y)
                      ? frame.subsampling_x
                      : frame.subsampling_y;
  const bool uses_chroma = params.frame_type[1] != kRestoreNone ||
                           params.frame_type[2] != kRestoreNone;
  const int chroma_scaled =
      uses_chroma ? (params.unit_size[1] << sub) : 0;
  const int max_plane_ru =
      params.unit_size[0] > chroma_scaled ? params.unit_size[0]
                                          : chroma_scaled;
  if (max_plane_ru <= 0) {
    return false;
  }
  if (frame.mi_col_start == nullptr || frame.mi_col_end == nullptr ||
      frame.mi_row_start == nullptr || frame.mi_row_end == nullptr) {
    return true;  // single tile
  }
  const int n = frame.mi_cols * frame.mi_rows;
  std::vector<int> cols;
  std::vector<int> rows;
  cols.reserve(static_cast<std::size_t>(frame.mi_cols));
  rows.reserve(static_cast<std::size_t>(frame.mi_rows));
  for (int i = 0; i < n; ++i) {
    cols.push_back(static_cast<int>(frame.mi_col_start[i]));
    rows.push_back(static_cast<int>(frame.mi_row_start[i]));
  }
  auto check_starts = [max_plane_ru](std::vector<int>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    for (std::size_t i = 1; i < v.size(); ++i) {
      const int width = (v[i] - v[i - 1]) * kMiSize;
      if (width <= 0 || width % max_plane_ru != 0) {
        return false;
      }
    }
    return true;
  };
  return check_starts(cols) && check_starts(rows);
}

RestorationStatus validate_lr(const RestorationFrame& frame,
                              const LoopRestorationParams& params) {
  if (params.gdf_frame_enable != 0) {
    return RestorationStatus::kUnsupportedGdf;
  }
  if (params.num_filter_classes < 1 ||
      params.num_filter_classes > kWienerNsClasses ||
      !num_filter_classes_valid(params.num_filter_classes)) {
    return RestorationStatus::kInvalidArgument;
  }
  if (params.filter_class_count < 1 || params.filter_class_count > 256) {
    // §7.20.4 FilterClass comes from Pc_Wiener_Lut_To_Class, whose range is
    // NUM_PC_WIENER_LUT_CLASSES (256).  The legacy 16-entry form stays valid.
    return RestorationStatus::kInvalidArgument;
  }
  if (params.unit_size[1] != params.unit_size[2]) {
    // §5.18.7.11 sets LoopRestorationSize[2] = LoopRestorationSize[1].
    return RestorationStatus::kNonConformingUnitSize;
  }
  for (int p = 0; p < 3; ++p) {
    if (!frame_type_in_range(params.frame_type[p])) {
      return RestorationStatus::kInvalidArgument;
    }
    if (params.frame_type[p] == kRestorePcWiener) {
      return RestorationStatus::kUnsupportedPcWiener;
    }
    const int w = restoration_plane_width(frame, p);
    if (frame.lr[p].data == nullptr || frame.lr[p].stride < w) {
      return RestorationStatus::kInvalidArgument;
    }
    if (frame.lr[p].data == frame.cdef[p].data) {
      // Loop restoration reads CdefFrame while writing LrFrame.
      return RestorationStatus::kInvalidArgument;
    }
    if (params.frame_type[p] == kRestoreNone) {
      continue;
    }
    if (params.unit_size[p] <= 0) {
      return RestorationStatus::kInvalidArgument;
    }
    const int sub_x = (p == 0) ? 0 : frame.subsampling_x;
    const int sub_y = (p == 0) ? 0 : frame.subsampling_y;
    // §5.18.7.11/§6.17.7.7: LoopRestorationSize is a power of two no larger
    // than RESTORATION_TILESIZE_MAX; chroma is at least one stripe high
    // (64 >> SubsamplingY), luma at least 64.
    if (!is_power_of_two(params.unit_size[p]) ||
        params.unit_size[p] >
            (kRestorationTileSizeMax >> ((p == 0) ? 0 : sub_y))) {
      return RestorationStatus::kNonConformingUnitSize;
    }
    if (p == 0) {
      if (params.unit_size[0] < 64) {
        return RestorationStatus::kNonConformingUnitSize;
      }
    } else if (params.unit_size[p] < (64 >> sub_y)) {
      return RestorationStatus::kNonConformingUnitSize;
    }
    const int need_rows = restoration_count_units(
        params.unit_size[p], (frame.mi_rows * kMiSize) >> sub_y);
    const int need_cols = restoration_count_units(
        params.unit_size[p], (frame.mi_cols * kMiSize) >> sub_x);
    if (params.unit_type[p] == nullptr || params.unit_rows[p] < need_rows ||
        params.unit_cols[p] < need_cols) {
      return RestorationStatus::kInvalidArgument;
    }
    // For RESTORE_SWITCHABLE, the per-unit LrType grid may be host or device
    // memory; this validator never dereferences it.  The CPU path checks host
    // grids in loop_restoration_frame(); the CUDA kernel leaves a PC-Wiener
    // unit as a copy of CdefFrame and callers must not signal one.
    if (params.frame_filters_on[p] != 0) {
      if (params.frame_coeffs[p] == nullptr) {
        return RestorationStatus::kInvalidArgument;
      }
    } else if (params.unit_coeffs[p] == nullptr) {
      return RestorationStatus::kInvalidArgument;
    }
    if (p == 0 && params.frame_filters_on[0] != 0 &&
        params.num_filter_classes > 1) {
      if (params.filter_class_grid != nullptr) {
        if (params.subclass_lookup == nullptr) {
          return RestorationStatus::kMissingSubclassGrid;
        }
      } else if (params.subclass_grid == nullptr) {
        return RestorationStatus::kMissingSubclassGrid;
      }
    }
  }
  return RestorationStatus::kOk;
}

}  // namespace

// Checks that need to dereference host metadata grids.  The CUDA path cannot
// use these (the grids are device memory), so they are applied only by the CPU
// reference entry points.
namespace {

RestorationStatus validate_lr_host_grids(const RestorationFrame& frame,
                                         const LoopRestorationParams& params) {
  for (int p = 0; p < 3; ++p) {
    if (params.frame_type[p] == kRestoreSwitchable &&
        grid_has_pc_wiener(params.unit_type[p], params.unit_rows[p],
                           params.unit_cols[p])) {
      return RestorationStatus::kUnsupportedPcWiener;
    }
  }
  if (!check_ru_size(frame, params)) {
    return RestorationStatus::kNonConformingUnitSize;
  }
  return RestorationStatus::kOk;
}

}  // namespace

RestorationStatus loop_restoration_validate(
    const RestorationFrame& frame, const LoopRestorationParams& params) {
  const RestorationStatus st = validate_frame(frame);
  if (st != RestorationStatus::kOk) {
    return st;
  }
  return validate_lr(frame, params);
}

RestorationStatus loop_restoration_plane(const RestorationFrame& frame,
                                         const LoopRestorationParams& params,
                                         int plane) {
  if (plane < 0 || plane > 2) {
    return RestorationStatus::kInvalidArgument;
  }
  const RestorationStatus lr = loop_restoration_validate(frame, params);
  if (lr != RestorationStatus::kOk) {
    return lr;
  }
  const RestorationStatus grids = validate_lr_host_grids(frame, params);
  if (grids != RestorationStatus::kOk) {
    return grids;
  }
  LrCoreFrame core = loop_restoration_to_core(frame, params);
  lr_copy_plane(core, plane);
  if (params.frame_type[plane] != kRestoreNone) {
    lr_plane(core, plane);
  }
  return RestorationStatus::kOk;
}

RestorationStatus loop_restoration_frame(const RestorationFrame& frame,
                                         const LoopRestorationParams& params) {
  const RestorationStatus lr = loop_restoration_validate(frame, params);
  if (lr != RestorationStatus::kOk) {
    return lr;
  }
  const RestorationStatus grids = validate_lr_host_grids(frame, params);
  if (grids != RestorationStatus::kOk) {
    return grids;
  }
  LrCoreFrame core = loop_restoration_to_core(frame, params);
  // §7.20: LrFrame := CdefFrame before any filtering.
  for (int p = 0; p < 3; ++p) {
    lr_copy_plane(core, p);
  }
  bool any = false;
  for (int p = 0; p < 3; ++p) {
    if (params.frame_type[p] != kRestoreNone) {
      any = true;
    }
  }
  if (!any) {
    return RestorationStatus::kOk;
  }
  for (int p = 0; p < 3; ++p) {
    if (params.frame_type[p] != kRestoreNone) {
      lr_plane(core, p);
    }
  }
  return RestorationStatus::kOk;
}

}  // namespace av2enc::restoration
