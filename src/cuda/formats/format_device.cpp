#include "cuda/formats/format_device.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::fmtdev {
namespace {

int padded_extent(int coded) { return (coded + 63) & ~63; }

int depth_value(av2enc::fmt::BitDepth depth) {
  switch (depth) {
    case av2enc::fmt::BitDepth::k8:
      return 8;
    case av2enc::fmt::BitDepth::k10:
      return 10;
    case av2enc::fmt::BitDepth::k12:
      return 12;
  }
  return 0;
}

}  // namespace

FrameLayout make_frame_layout(const av2enc::fmt::Format& format) {
  FrameLayout layout;
  layout.format = format;
  layout.bit_depth = depth_value(format.depth);
  const auto geometry = av2enc::fmt::make_frame_geometry(format);
  if (!geometry.has_value()) {
    return layout;
  }
  const av2enc::fmt::FrameGeometry& g = *geometry;
  layout.mi_cols = static_cast<int>(g.mi_cols);
  layout.mi_rows = static_cast<int>(g.mi_rows);
  layout.coded_w = static_cast<int>(g.coded_width);
  layout.coded_h = static_cast<int>(g.coded_height);
  layout.storage_w = padded_extent(layout.coded_w);
  layout.storage_h = padded_extent(layout.coded_h);
  layout.padded_w = static_cast<int>(g.planes[0].stride);
  layout.padded_h = 0;
  // `fmt::make_frame_geometry()` does not carry padded height explicitly; the
  // format helper uses 64-aligned visible dimensions, so derive it the same
  // way (chroma is the luma extent shifted right).
  layout.padded_w = ((static_cast<int>(format.width) + 63) / 64) * 64;
  layout.padded_h = ((static_cast<int>(format.height) + 63) / 64) * 64;
  layout.num_planes = g.num_planes;
  layout.sub_x = g.sub_x;
  layout.sub_y = g.sub_y;

  std::size_t coded_offset = 0;
  std::size_t storage_offset = 0;
  for (int plane = 0; plane < 3; ++plane) {
    PlaneLayout& p = layout.planes[plane];
    p = PlaneLayout{};
    if (plane >= g.num_planes) {
      continue;
    }
    p.present = true;
    p.sub_x = plane == 0 ? 0 : g.sub_x;
    p.sub_y = plane == 0 ? 0 : g.sub_y;
    p.visible_w = static_cast<int>(g.planes[plane].visible_width);
    p.visible_h = static_cast<int>(g.planes[plane].visible_height);
    p.coded_w = static_cast<int>(g.planes[plane].coded_width);
    p.coded_h = static_cast<int>(g.planes[plane].coded_height);
    p.storage_w = padded_extent(layout.coded_w) >> p.sub_x;
    p.storage_h = padded_extent(layout.coded_h) >> p.sub_y;
    p.padded_w = layout.padded_w >> p.sub_x;
    p.padded_h = layout.padded_h >> p.sub_y;
    p.stride = p.storage_w;
    p.coded_samples = static_cast<std::size_t>(p.coded_w) * p.coded_h;
    p.storage_samples =
        static_cast<std::size_t>(p.storage_w) * p.storage_h;
    p.coded_offset = coded_offset;
    p.storage_offset = storage_offset;
    p.sample_bytes = layout.bit_depth <= 8 ? 1 : 2;
    coded_offset += p.coded_samples;
    storage_offset += p.storage_samples;
  }
  layout.coded_samples = coded_offset;
  layout.storage_samples = storage_offset;
  layout.coded_bytes = coded_offset * sizeof(std::uint16_t);
  layout.storage_bytes = storage_offset * sizeof(std::uint16_t);
  return layout;
}

FrameLayout make_frame_layout(std::uint32_t width, std::uint32_t height,
                              av2enc::fmt::ChromaFormat chroma,
                              av2enc::fmt::BitDepth depth) {
  av2enc::fmt::Format format;
  format.width = width;
  format.height = height;
  format.chroma = chroma;
  format.depth = depth;
  return make_frame_layout(format);
}

FilterGeometry make_filter_geometry(const av2enc::fmt::Format& format) {
  const FrameLayout layout = make_frame_layout(format);
  return make_filter_geometry(layout);
}

FilterGeometry make_filter_geometry(const FrameLayout& layout) {
  FilterGeometry g;
  g.format = layout.format;
  g.bit_depth = layout.bit_depth;
  g.luma_mi_cols = layout.mi_cols;
  g.luma_mi_rows = layout.mi_rows;
  g.num_planes = layout.num_planes;
  g.sub_x = layout.sub_x;
  g.sub_y = layout.sub_y;
  for (int plane = 0; plane < 3; ++plane) {
    FilterPlaneLayout& p = g.planes[plane];
    p = FilterPlaneLayout{};
    if (plane >= layout.num_planes) continue;
    const PlaneLayout& src = layout.planes[plane];
    p.present = true;
    p.sub_x = src.sub_x;
    p.sub_y = src.sub_y;
    p.visible_w = src.visible_w;
    p.visible_h = src.visible_h;
    p.coded_w = src.coded_w;
    p.coded_h = src.coded_h;
    p.stride = src.coded_w;
    p.sample_bytes = src.sample_bytes;
    p.coded_mi_cols = src.coded_w / 4;
    p.coded_mi_rows = src.coded_h / 4;
    p.luma_mi_cols = layout.mi_cols;
    p.luma_mi_rows = layout.mi_rows;
    // Interior edge counts: a plane with N 4x4 columns has N-1 interior
    // vertical edges (the plane border is not filtered).
    p.edge4_cols = p.coded_mi_cols > 0 ? p.coded_mi_cols - 1 : 0;
    p.edge4_rows = p.coded_mi_rows > 0 ? p.coded_mi_rows - 1 : 0;
    p.edge8_cols = p.coded_w >= 8 ? p.coded_w / 8 - 1 : 0;
    p.edge8_rows = p.coded_h >= 8 ? p.coded_h / 8 - 1 : 0;
  }
  return g;
}

int filter_plane_width(const FilterGeometry& geometry, int plane) {
  if (plane < 0 || plane >= 3 || !geometry.planes[plane].present) return 0;
  return geometry.planes[plane].coded_w;
}

int filter_plane_height(const FilterGeometry& geometry, int plane) {
  if (plane < 0 || plane >= 3 || !geometry.planes[plane].present) return 0;
  return geometry.planes[plane].coded_h;
}

bool plane_block_rect(const FrameLayout& layout, int plane, int luma_x,
                      int luma_y, int luma_w, int luma_h,
                      PlaneBlockRect* out) {
  if (out == nullptr || plane < 0 || plane >= layout.num_planes) {
    return false;
  }
  const auto block = av2enc::fmt::find_block_size(luma_w, luma_h);
  if (!block.has_value()) return false;
  int width = luma_w;
  int height = luma_h;
  if (plane != 0) {
    const auto residual = av2enc::fmt::chroma_residual_size(
        *block, layout.format.chroma);
    if (!residual.has_value()) return false;
    width = av2enc::fmt::block_width(*residual);
    height = av2enc::fmt::block_height(*residual);
  }
  const PlaneLayout& p = layout.planes[plane];
  const int x = luma_x >> p.sub_x;
  const int y = luma_y >> p.sub_y;
  if (x < 0 || y < 0) return false;
  PlaneBlockRect rect;
  rect.x = x;
  rect.y = y;
  rect.width = width;
  rect.height = height;
  rect.visible_width = std::max(0, std::min(width, p.visible_w - x));
  rect.visible_height = std::max(0, std::min(height, p.visible_h - y));
  rect.inside = rect.visible_width > 0 && rect.visible_height > 0;
  rect.partial = rect.visible_width < width || rect.visible_height < height;
  *out = rect;
  return true;
}

BitstreamLayout make_bitstream_layout(int max_blocks) {
  BitstreamLayout layout;
  layout.max_blocks = max_blocks > 0 ? max_blocks : 1;
  layout.max_tx_elems = 32 * 32;
  layout.level_capacity =
      static_cast<std::size_t>(layout.max_blocks) * layout.max_tx_elems;
  layout.eob_capacity = static_cast<std::size_t>(layout.max_blocks) * 3;
  layout.block_capacity = static_cast<std::size_t>(layout.max_blocks);
  return layout;
}

}  // namespace av2enc::cuda::fmtdev
