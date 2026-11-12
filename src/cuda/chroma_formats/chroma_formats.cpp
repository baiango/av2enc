// Host half of the CUDA chroma-format component: Table 6.2, plane geometry,
// Subsampled_Size and the nominal -> adjusted transform shape.
//
// Everything here is a deliberate line-for-line twin of
// `src/encoder/format/chroma/chroma_format.cpp` and
// `src/encoder/format/chroma/chroma_residual.cpp`; the emulation test pins
// every field against those functions.
#include "cuda/chroma_formats/chroma_formats.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "encoder/format/format_tables.h"

namespace av2enc::cuda::chroma {
namespace {

constexpr int kInvalidBlockIndex = fmt::kInvalidBlockIndex;

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

void transform_shifts(int nominal_width, int nominal_height, int* row_shift,
                      int* col_shift) {
  // Transform_Shift[TX_SIZES_ALL][2] (§7.15.4.1), keyed by the nominal size.
  const bool square = nominal_width == nominal_height;
  if (nominal_width == 4 && nominal_height == 4) {
    *row_shift = 7; *col_shift = 10;
  } else if (nominal_width == 8 && nominal_height == 8) {
    *row_shift = 7; *col_shift = 11;
  } else if (nominal_width == 16 && nominal_height == 16) {
    *row_shift = 6; *col_shift = 13;
  } else if (nominal_width == 32 && nominal_height == 32) {
    *row_shift = 6; *col_shift = 13;
  } else if (nominal_width == 64 && nominal_height == 64) {
    *row_shift = 6; *col_shift = 13;
  } else if (nominal_width == 4 &&
             (nominal_height == 8 || nominal_height == 4)) {
    *row_shift = 7; *col_shift = 10;
  } else if (nominal_height == 4 && nominal_width == 8) {
    *row_shift = 7; *col_shift = 10;
  } else if ((nominal_width == 8 && nominal_height == 16) ||
             (nominal_width == 16 && nominal_height == 8)) {
    *row_shift = 7; *col_shift = 11;
  } else if ((nominal_width == 16 && nominal_height == 32) ||
             (nominal_width == 32 && nominal_height == 16)) {
    *row_shift = 6; *col_shift = 12;
  } else if ((nominal_width == 32 && nominal_height == 64) ||
             (nominal_width == 64 && nominal_height == 32)) {
    *row_shift = 6; *col_shift = 12;
  } else if ((nominal_width == 4 && nominal_height == 16) ||
             (nominal_width == 16 && nominal_height == 4)) {
    *row_shift = 6; *col_shift = 12;
  } else if ((nominal_width == 8 && nominal_height == 32) ||
             (nominal_width == 32 && nominal_height == 8)) {
    *row_shift = 6; *col_shift = 13;
  } else if ((nominal_width == 16 && nominal_height == 64) ||
             (nominal_width == 64 && nominal_height == 16)) {
    *row_shift = 6; *col_shift = 13;
  } else if ((nominal_width == 4 && nominal_height == 32) ||
             (nominal_width == 32 && nominal_height == 4)) {
    *row_shift = 7; *col_shift = 11;
  } else if ((nominal_width == 8 && nominal_height == 64) ||
             (nominal_width == 64 && nominal_height == 8)) {
    *row_shift = 6; *col_shift = 12;
  } else if ((nominal_width == 4 && nominal_height == 64) ||
             (nominal_width == 64 && nominal_height == 4)) {
    *row_shift = 6; *col_shift = 13;
  } else if (square) {
    *row_shift = 6; *col_shift = 13;
  } else {
    *row_shift = 6; *col_shift = 13;
  }
}

}  // namespace

bool format_valid(Format format) noexcept {
  switch (format) {
    case Format::k400:
    case Format::k420:
    case Format::k422:
    case Format::k444:
      return true;
  }
  return false;
}

FormatInfo format_info(Format format) noexcept {
  FormatInfo info;
  info.format = format;
  switch (format) {
    case Format::k400:
      info.idc = 1;
      info.name = "4:0:0";
      info.sub_x = 1;
      info.sub_y = 1;
      info.monochrome = true;
      info.num_planes = 1;
      info.profile_idc = 0;
      break;
    case Format::k420:
      info.idc = 0;
      info.name = "4:2:0";
      info.sub_x = 1;
      info.sub_y = 1;
      info.monochrome = false;
      info.num_planes = 3;
      info.profile_idc = 0;
      break;
    case Format::k422:
      info.idc = 3;
      info.name = "4:2:2";
      info.sub_x = 1;
      info.sub_y = 0;
      info.monochrome = false;
      info.num_planes = 3;
      info.profile_idc = 3;
      break;
    case Format::k444:
      info.idc = 2;
      info.name = "4:4:4";
      info.sub_x = 0;
      info.sub_y = 0;
      info.monochrome = false;
      info.num_planes = 3;
      info.profile_idc = 4;
      break;
  }
  return info;
}

const char* format_name(Format format) noexcept {
  return format_valid(format) ? format_info(format).name : "invalid";
}

FrameGeom make_frame_geometry(int width, int height, Format format) noexcept {
  return make_frame_geometry(format_info(format), width, height);
}

FrameGeom make_frame_geometry(const FormatInfo& info, int width,
                              int height) noexcept {
  FrameGeom shape;
  shape.width = width;
  shape.height = height;
  shape.format = info.format;
  shape.info = info;
  shape.monochrome = info.monochrome;
  shape.num_planes = info.num_planes;

  // §5.18.4.4 compute_image_size().
  shape.mi_cols = 2 * ((width + 7) >> 3);
  shape.mi_rows = 2 * ((height + 7) >> 3);
  shape.coded_w = shape.mi_cols * kMiSize;
  shape.coded_h = shape.mi_rows * kMiSize;
  shape.padded_w = ((width + 63) / 64) * 64;
  shape.padded_h = ((height + 63) / 64) * 64;

  for (int plane = 0; plane < kMaxPlanes; ++plane) {
    PlaneGeom& p = shape.planes[plane];
    p.present = plane == 0 || !shape.monochrome;
    if (!p.present) {
      continue;
    }
    p.sub_x = plane == 0 ? 0 : info.sub_x;
    p.sub_y = plane == 0 ? 0 : info.sub_y;
    if (plane == 0) {
      p.visible_w = width;
      p.visible_h = height;
      p.coded_w = shape.coded_w;
      p.coded_h = shape.coded_h;
      p.padded_w = shape.padded_w;
      p.padded_h = shape.padded_h;
    } else {
      // §7.21.2: visible chroma = ceil(width / 2^SubsamplingX).
      p.visible_w = (width + ((1 << p.sub_x) - 1)) >> p.sub_x;
      p.visible_h = (height + ((1 << p.sub_y) - 1)) >> p.sub_y;
      p.coded_w = shape.coded_w >> p.sub_x;
      p.coded_h = shape.coded_h >> p.sub_y;
      p.padded_w = shape.padded_w >> p.sub_x;
      p.padded_h = shape.padded_h >> p.sub_y;
    }
  }
  return shape;
}

int block_width(int block_index) noexcept {
  if (block_index < 0 || block_index >= fmt::kBlockSizeCount) return 0;
  return fmt::kBlockWidth[block_index];
}

int block_height(int block_index) noexcept {
  if (block_index < 0 || block_index >= fmt::kBlockSizeCount) return 0;
  return fmt::kBlockHeight[block_index];
}

int block_size_index(int width, int height) noexcept {
  for (int index = 0; index < fmt::kBlockSizeCount; ++index) {
    if (fmt::kBlockWidth[index] == width &&
        fmt::kBlockHeight[index] == height) {
      return index;
    }
  }
  return -1;
}

ResidualSize chroma_residual_size_block(int block_index,
                                        Format format) noexcept {
  ResidualSize result;
  if (!format_valid(format)) return result;
  const FormatInfo info = format_info(format);
  if (info.monochrome) return result;  // no chroma residual plane
  if (block_index < 0 || block_index >= fmt::kBlockSizeCount) return result;

  result.block_index = block_index;
  result.luma_w = fmt::kBlockWidth[block_index];
  result.luma_h = fmt::kBlockHeight[block_index];
  const int value =
      fmt::kSubsampledSize[block_index][info.sub_x][info.sub_y];
  if (value == kInvalidBlockIndex) {
    return result;  // BLOCK_INVALID: no legal chroma residual
  }
  result.valid = true;
  result.width = fmt::kBlockWidth[value];
  result.height = fmt::kBlockHeight[value];
  return result;
}

ResidualSize chroma_residual_size(int luma_w, int luma_h,
                                  Format format) noexcept {
  ResidualSize result;
  const int index = block_size_index(luma_w, luma_h);
  if (index < 0) {
    result.luma_w = luma_w;
    result.luma_h = luma_h;
    return result;
  }
  return chroma_residual_size_block(index, format);
}

bool is_chroma_offset_for_subsize(int luma_w, int luma_h,
                                  Format format) noexcept {
  if (!format_valid(format)) return false;
  const FormatInfo info = format_info(format);
  if (info.monochrome) return false;
  const int w4 = luma_w / kMiSize;
  const int h4 = luma_h / kMiSize;
  if (info.sub_y != 0 && h4 == 1) return true;
  if (info.sub_x != 0 && w4 == 1) return true;
  return false;
}

int tx_size_context(int nominal_w, int nominal_h) noexcept {
  // Tx_Size_Sqr / Tx_Size_Sqr_Up are enum indices: TX_4X4 = 0, ..., TX_64X64
  // = 4.  They are not raw log2 values.
  const int sqr = std::max(0, floor_log2(nominal_w) - 2);
  const int sqr_up = std::max(0, floor_log2(nominal_h) - 2);
  const int lo = std::min(sqr, sqr_up);
  const int hi = std::max(sqr, sqr_up);
  return (lo + hi + 1) >> 1;
}

bool odd_rect_transform(int nominal_w, int nominal_h) noexcept {
  return ((floor_log2(nominal_w) ^ floor_log2(nominal_h)) & 1) != 0;
}

bool make_tx_shape(int nominal_w, int nominal_h, TxShape* shape) noexcept {
  if (shape == nullptr) return false;
  if (nominal_w < 4 || nominal_h < 4 || nominal_w > 64 || nominal_h > 64) {
    return false;
  }
  if ((nominal_w & (nominal_w - 1)) != 0 ||
      (nominal_h & (nominal_h - 1)) != 0) {
    return false;
  }
  TxShape out;
  out.nominal_w = nominal_w;
  out.nominal_h = nominal_h;
  out.adj_w = std::min(nominal_w, 32);
  out.adj_h = std::min(nominal_h, 32);
  out.upscale_x = nominal_w / out.adj_w;
  out.upscale_y = nominal_h / out.adj_h;
  out.tx_size_ctx = tx_size_context(nominal_w, nominal_h);
  out.eob_multisize = std::min(5, floor_log2(nominal_w)) +
                      std::min(5, floor_log2(nominal_h)) - 4;
  out.seg_eob = out.adj_w * out.adj_h;
  out.tx_pels = nominal_w * nominal_h;
  out.dq_denom = 1 << ((out.tx_pels > 256) + (out.tx_pels > 1024));
  transform_shifts(nominal_w, nominal_h, &out.row_shift, &out.col_shift);
  out.odd_rect = odd_rect_transform(nominal_w, nominal_h);
  out.scale_w = 1.0 / (64.0 * std::sqrt(static_cast<double>(out.adj_w)));
  out.scale_h = 1.0 / (64.0 * std::sqrt(static_cast<double>(out.adj_h)));
  out.rect_prescale = out.odd_rect ? 4096.0 / 2896.0 : 1.0;
  // Same expression and order as chroma::quant_factor_for().
  const double inv_scale =
      (64.0 * std::sqrt(static_cast<double>(out.adj_w)) /
       static_cast<double>(1 << out.row_shift)) *
      (64.0 * std::sqrt(static_cast<double>(out.adj_h)) /
       static_cast<double>(1 << out.col_shift));
  out.quant_factor =
      8.0 * static_cast<double>(out.dq_denom) / (inv_scale * 528.0);
  *shape = out;
  return true;
}

bool config_valid(const Config& config) noexcept {
  if (!format_valid(config.format)) return false;
  if (config.width <= 0 || config.height <= 0) return false;
  if (config.width > kMaxFrameDimension ||
      config.height > kMaxFrameDimension) {
    return false;
  }
  if (config.bit_depth != 8 && config.bit_depth != 10) return false;
  if (config.max_blocks < 1 || config.max_blocks > 4096) return false;
  return true;
}

}  // namespace av2enc::cuda::chroma
