#include "codec/filters/cdef/cdef.h"

#include <cstddef>
#include <cstdint>

namespace av2enc::cdef {
namespace {

int plane_horizontal_shift(const CdefFrame& frame, int plane) {
  return (plane == 0) ? 0 : frame.subsampling_x;
}

int plane_vertical_shift(const CdefFrame& frame, int plane) {
  return (plane == 0) ? 0 : frame.subsampling_y;
}

int derived_n64_cols(const CdefFrame& frame) {
  return (frame.mi_cols + kCdefSize4 - 1) / kCdefSize4;
}

int derived_n64_rows(const CdefFrame& frame) {
  return (frame.mi_rows + kCdefSize4 - 1) / kCdefSize4;
}

int idx_grid_cols(const CdefFrame& frame) {
  return frame.n64_cols > 0 ? frame.n64_cols : derived_n64_cols(frame);
}

int idx_grid_rows(const CdefFrame& frame) {
  return frame.n64_rows > 0 ? frame.n64_rows : derived_n64_rows(frame);
}

// Filter region for the CDEF block whose origin is (r, c) in luma MI units.
// §5.20.9.3: the tile bounds when loop filters are disabled across tiles, the
// whole frame otherwise.
void tile_region(const CdefFrame& frame, const CdefOptions& options, int r,
                 int c, CdefRegion* region) {
  *region = cdef_filter_region(
      frame.mi_cols, frame.mi_rows, frame.tile_col_starts, frame.num_tile_cols,
      frame.tile_row_starts, frame.num_tile_rows,
      options.disable_loopfilters_across_tiles, r, c);
}

// §7.18.1 copy step: the block is always copied from CurrFrame to CdefFrame.
void copy_block(const CdefFrame& frame, int r, int c) {
  const int y0 = r * kMiSize;
  const int x0 = c * kMiSize;
  for (int plane = 0; plane < 3; ++plane) {
    const CdefPlane& p = frame.planes[plane];
    const int sub_x = plane_horizontal_shift(frame, plane);
    const int sub_y = plane_vertical_shift(frame, plane);
    const int px0 = x0 >> sub_x;
    const int py0 = y0 >> sub_y;
    const int w = 8 >> sub_x;
    const int h = 8 >> sub_y;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const std::size_t offset =
            static_cast<std::size_t>(py0 + i) *
                static_cast<std::size_t>(p.stride) +
            static_cast<std::size_t>(px0 + j);
        p.dst[offset] = p.src[offset];
      }
    }
  }
}

// Fills idx, skip flags and, when filtering runs, the derived strengths and
// direction.  Does not touch pixels.
void analyze_block(const CdefFrame& frame, const CdefOptions& options, int r,
                   int c, CdefBlockInfo* info) {
  *info = CdefBlockInfo{};
  info->r = r;
  info->c = c;

  const int base_r = (r / kCdefSize4) * kCdefSize4;
  const int base_c = (c / kCdefSize4) * kCdefSize4;
  const int index =
      (base_r / kCdefSize4) * idx_grid_cols(frame) +
      (base_c / kCdefSize4);
  info->idx = static_cast<int>(frame.cdef_idx[index]);

  tile_region(frame, options, r, c, &info->region);

  bool skip = false;
  bool skip_chroma = false;
  cdef_block_skip_flags(frame.skip, frame.lossless, frame.lossless_chroma,
                        frame.mi_cols, r, c, options.on_skip_txfm, &skip,
                        &skip_chroma);
  info->skip = skip;
  info->skip_chroma = skip_chroma;

  if (info->idx < 0 || skip) {
    return;
  }

  const int coeff_shift = options.bit_depth - 8;
  const int y_pri_raw = options.strengths[info->idx].y_pri << coeff_shift;
  const int y_sec_raw = options.strengths[info->idx].y_sec << coeff_shift;

  int y_dir = 0;
  std::int64_t var = 0;
  cdef_direction(frame.planes[0].src,
                 static_cast<std::size_t>(frame.planes[0].stride),
                 c * kMiSize, r * kMiSize, options.bit_depth, &y_dir, &var);

  const CdefFilterParams luma = cdef_luma_params(
      y_pri_raw, y_sec_raw, y_dir, var, options.damping + coeff_shift);

  info->filtered = true;
  info->y_dir = y_dir;
  info->var = var;
  info->var_str = cdef_variance_strength(var);
  info->y_pri = luma.pri;
  info->y_sec = luma.sec;
  info->y_dir_used = luma.dir;
  info->y_damping = luma.damping;

  const int uv_pri = options.strengths[info->idx].uv_pri << coeff_shift;
  const int uv_sec = options.strengths[info->idx].uv_sec << coeff_shift;
  const CdefFilterParams chroma = cdef_chroma_params(
      uv_pri, uv_sec, y_dir, frame.subsampling_x, frame.subsampling_y,
      options.damping + coeff_shift);
  info->uv_pri = chroma.pri;
  info->uv_sec = chroma.sec;
  info->uv_dir = chroma.dir;
  info->uv_damping = chroma.damping;
}

void filter_plane(const CdefFrame& frame, int plane, int r, int c,
                  const CdefBlockInfo& info) {
  const CdefPlane& p = frame.planes[plane];
  const int sub_x = plane_horizontal_shift(frame, plane);
  const int sub_y = plane_vertical_shift(frame, plane);
  const int x0 = (c * kMiSize) >> sub_x;
  const int y0 = (r * kMiSize) >> sub_y;
  const int w = 8 >> sub_x;
  const int h = 8 >> sub_y;
  const int pri = (plane == 0) ? info.y_pri : info.uv_pri;
  const int sec = (plane == 0) ? info.y_sec : info.uv_sec;
  const int dir = (plane == 0) ? info.y_dir_used : info.uv_dir;
  const int damping = (plane == 0) ? info.y_damping : info.uv_damping;
  const std::size_t stride = static_cast<std::size_t>(p.stride);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int value = cdef_filter_sample(p.src, stride, x0, y0, i, j, dir,
                                           pri, sec, damping, sub_x, sub_y,
                                           info.region);
      p.dst[static_cast<std::size_t>(y0 + i) * stride +
            static_cast<std::size_t>(x0 + j)] = static_cast<std::uint8_t>(value);
    }
  }
}

void process_block(const CdefFrame& frame, const CdefOptions& options, int r,
                   int c) {
  copy_block(frame, r, c);
  CdefBlockInfo info;
  analyze_block(frame, options, r, c, &info);
  if (!info.filtered) {
    return;
  }
  filter_plane(frame, 0, r, c, info);
  if (info.skip_chroma) {
    return;
  }
  filter_plane(frame, 1, r, c, info);
  filter_plane(frame, 2, r, c, info);
}

bool valid_sec_strength(int value) {
  return value == 0 || value == 1 || value == 2 || value == 4;
}

}  // namespace

const char* cdef_status_string(CdefStatus status) {
  switch (status) {
    case CdefStatus::kOk:
      return "ok";
    case CdefStatus::kInvalidArgument:
      return "invalid argument";
    case CdefStatus::kUnsupportedBitDepth:
      return "unsupported bit depth (only 8-bit is implemented)";
    case CdefStatus::kUnsupportedSubsampling:
      return "unsupported chroma subsampling (only 4:2:0 is implemented)";
  }
  return "unknown";
}

int cdef_plane_width(const CdefFrame& frame, int plane) {
  if (plane < 0 || plane > 2 || frame.mi_cols <= 0) {
    return 0;
  }
  return (frame.mi_cols * kMiSize) >> plane_horizontal_shift(frame, plane);
}

int cdef_plane_height(const CdefFrame& frame, int plane) {
  if (plane < 0 || plane > 2 || frame.mi_rows <= 0) {
    return 0;
  }
  return (frame.mi_rows * kMiSize) >> plane_vertical_shift(frame, plane);
}

bool cdef_supported(const CdefFrame& frame, const CdefOptions& options,
                    CdefStatus* status) {
  auto fail = [status](CdefStatus s) {
    if (status != nullptr) {
      *status = s;
    }
    return false;
  };

  if (options.bit_depth != 8) {
    return fail(CdefStatus::kUnsupportedBitDepth);
  }
  if (frame.subsampling_x != 1 || frame.subsampling_y != 1) {
    return fail(CdefStatus::kUnsupportedSubsampling);
  }
  if (frame.mi_cols <= 0 || frame.mi_rows <= 0) {
    return fail(CdefStatus::kInvalidArgument);
  }
  // The 8x8 luma block grid must tile the MI-aligned plane exactly.
  if ((frame.mi_cols & 1) != 0 || (frame.mi_rows & 1) != 0) {
    return fail(CdefStatus::kInvalidArgument);
  }
  for (int plane = 0; plane < 3; ++plane) {
    const CdefPlane& p = frame.planes[plane];
    if (p.src == nullptr || p.dst == nullptr ||
        p.src == p.dst) {  // CDEF reads CurrFrame and writes CdefFrame
      return fail(CdefStatus::kInvalidArgument);
    }
    const int width = cdef_plane_width(frame, plane);
    if (p.stride < width) {
      return fail(CdefStatus::kInvalidArgument);
    }
  }
  if (frame.skip == nullptr || frame.lossless == nullptr ||
      frame.cdef_idx == nullptr) {
    return fail(CdefStatus::kInvalidArgument);
  }

  if (options.damping < 3 || options.damping > 6) {
    return fail(CdefStatus::kInvalidArgument);
  }
  if (options.num_strengths < 1 || options.num_strengths > kMaxStrengths) {
    return fail(CdefStatus::kInvalidArgument);
  }
  for (int i = 0; i < options.num_strengths; ++i) {
    const CdefStrength& s = options.strengths[i];
    if (s.y_pri < 0 || s.y_pri > 15 || s.uv_pri < 0 || s.uv_pri > 15 ||
        !valid_sec_strength(s.y_sec) || !valid_sec_strength(s.uv_sec)) {
      return fail(CdefStatus::kInvalidArgument);
    }
  }

  const int need_cols = derived_n64_cols(frame);
  const int need_rows = derived_n64_rows(frame);
  if (frame.n64_cols < 0 || frame.n64_rows < 0) {
    return fail(CdefStatus::kInvalidArgument);
  }
  if ((frame.n64_cols != 0 && frame.n64_cols < need_cols) ||
      (frame.n64_rows != 0 && frame.n64_rows < need_rows)) {
    return fail(CdefStatus::kInvalidArgument);
  }
  const int grid_cols = idx_grid_cols(frame);
  const int grid_rows = idx_grid_rows(frame);
  for (int row = 0; row < grid_rows; ++row) {
    for (int col = 0; col < grid_cols; ++col) {
      const int value = static_cast<int>(
          frame.cdef_idx[static_cast<std::size_t>(row) *
                             static_cast<std::size_t>(grid_cols) +
                         static_cast<std::size_t>(col)]);
      if (value < -1 || value >= options.num_strengths) {
        return fail(CdefStatus::kInvalidArgument);
      }
    }
  }

  if ((frame.num_tile_cols > 0 && frame.tile_col_starts == nullptr) ||
      (frame.num_tile_rows > 0 && frame.tile_row_starts == nullptr)) {
    return fail(CdefStatus::kInvalidArgument);
  }
  if (frame.num_tile_cols < 0 || frame.num_tile_rows < 0) {
    return fail(CdefStatus::kInvalidArgument);
  }
  for (int i = 0; i < frame.num_tile_cols; ++i) {
    const int value = static_cast<int>(frame.tile_col_starts[i]);
    if (value < 0 || value >= frame.mi_cols) {
      return fail(CdefStatus::kInvalidArgument);
    }
    if (i > 0 &&
        value <= static_cast<int>(frame.tile_col_starts[i - 1])) {
      return fail(CdefStatus::kInvalidArgument);
    }
  }
  for (int i = 0; i < frame.num_tile_rows; ++i) {
    const int value = static_cast<int>(frame.tile_row_starts[i]);
    if (value < 0 || value >= frame.mi_rows) {
      return fail(CdefStatus::kInvalidArgument);
    }
    if (i > 0 &&
        value <= static_cast<int>(frame.tile_row_starts[i - 1])) {
      return fail(CdefStatus::kInvalidArgument);
    }
  }

  if (status != nullptr) {
    *status = CdefStatus::kOk;
  }
  return true;
}

CdefStatus cdef_frame(const CdefFrame& frame, const CdefOptions& options) {
  CdefStatus status = CdefStatus::kOk;
  if (!cdef_supported(frame, options, &status)) {
    return status;
  }
  for (int r = 0; r < frame.mi_rows; r += kCdefBlock4) {
    for (int c = 0; c < frame.mi_cols; c += kCdefBlock4) {
      process_block(frame, options, r, c);
    }
  }
  return CdefStatus::kOk;
}

CdefStatus cdef_inspect_block(const CdefFrame& frame,
                              const CdefOptions& options, int r, int c,
                              CdefBlockInfo* info) {
  CdefStatus status = CdefStatus::kOk;
  if (!cdef_supported(frame, options, &status)) {
    return status;
  }
  if (info == nullptr || r < 0 || c < 0 || (r & 1) != 0 || (c & 1) != 0 ||
      r + kCdefBlock4 > frame.mi_rows || c + kCdefBlock4 > frame.mi_cols) {
    return CdefStatus::kInvalidArgument;
  }
  analyze_block(frame, options, r, c, info);
  return CdefStatus::kOk;
}

}  // namespace av2enc::cdef
