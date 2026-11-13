// Host parity test for the shared device frame/filter geometry.
//
// Cross-checks `format_device.*` against the product format reference:
//   * every plane field of `fmt::make_frame_geometry()`,
//   * `fmt::block_rect()` over exact and partial blocks (all 29 BLOCK_SIZE
//     entries and both chroma subsampling modes),
//   * the 8-bit 4:2:0 deblock plane extents in `src/codec/filters/deblock/`,
//   * the bitstream-facing level/EOB capacity contract.
//
// No GPU or nvcc is needed.
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "codec/filters/deblock/deblock.h"
#include "cuda/formats/format_device.h"
#include "formats/format.h"

namespace {

int failures = 0;
long long checks = 0;

void check(bool condition, const char* what) {
  ++checks;
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

void check_int(long long actual, long long expected, const char* what) {
  ++checks;
  if (actual != expected) {
    std::fprintf(stderr, "FAIL: %s (actual %lld expected %lld)\n", what,
                 actual, expected);
    ++failures;
  }
}

void check_geometry(const av2enc::fmt::Format& format) {
  const auto reference = av2enc::fmt::make_frame_geometry(format);
  check(reference.has_value(), "reference geometry");
  if (!reference.has_value()) return;
  const av2enc::cuda::fmtdev::FrameLayout layout =
      av2enc::cuda::fmtdev::make_frame_layout(format);
  check(layout.valid(), "layout valid");
  check_int(layout.mi_cols, reference->mi_cols, "mi_cols");
  check_int(layout.mi_rows, reference->mi_rows, "mi_rows");
  check_int(layout.coded_w, reference->coded_width, "coded_w");
  check_int(layout.coded_h, reference->coded_height, "coded_h");
  check_int(layout.num_planes, reference->num_planes, "num_planes");
  check_int(layout.sub_x, reference->sub_x, "sub_x");
  check_int(layout.sub_y, reference->sub_y, "sub_y");

  std::size_t coded_offset = 0;
  std::size_t storage_offset = 0;
  const int depth = av2enc::fmt::bit_depth_value(format.depth);
  for (int plane = 0; plane < 3; ++plane) {
    const auto& p = layout.planes[plane];
    const auto& r = reference->planes[plane];
    if (plane >= reference->num_planes) {
      check(!p.present, "absent plane");
      continue;
    }
    check(p.present, "present plane");
    check_int(p.visible_w, r.visible_width, "visible_w");
    check_int(p.visible_h, r.visible_height, "visible_h");
    check_int(p.coded_w, r.coded_width, "plane coded_w");
    check_int(p.coded_h, r.coded_height, "plane coded_h");
    check_int(p.coded_samples,
              static_cast<long long>(r.coded_width) * r.coded_height,
              "plane coded_samples");
    check_int(p.coded_offset, static_cast<long long>(coded_offset),
              "plane coded_offset");
    check_int(p.storage_offset, static_cast<long long>(storage_offset),
              "plane storage_offset");
    check_int(p.stride, p.storage_w, "plane stride");
    check_int(p.sample_bytes, depth <= 8 ? 1 : 2, "plane sample_bytes");
    const int padded_luma_w =
        ((static_cast<int>(format.width) + 63) / 64) * 64;
    const int padded_luma_h =
        ((static_cast<int>(format.height) + 63) / 64) * 64;
    check_int(p.padded_w, padded_luma_w >> p.sub_x, "plane padded_w");
    check_int(p.padded_h, padded_luma_h >> p.sub_y, "plane padded_h");
    const int storage_w = ((p.coded_w + 63) & ~63);
    const int storage_h = ((p.coded_h + 63) & ~63);
    // The storage extent is 64-aligned in luma coordinates and then shifted.
    const int expected_storage_w =
        (((static_cast<int>(reference->coded_width) + 63) & ~63) >> p.sub_x);
    const int expected_storage_h =
        (((static_cast<int>(reference->coded_height) + 63) & ~63) >> p.sub_y);
    check_int(p.storage_w, expected_storage_w, "plane storage_w");
    check_int(p.storage_h, expected_storage_h, "plane storage_h");
    (void)storage_w;
    (void)storage_h;
    coded_offset += p.coded_samples;
    storage_offset += p.storage_samples;
  }
  check_int(layout.coded_samples, static_cast<long long>(coded_offset),
            "coded_samples total");
  check_int(layout.storage_samples, static_cast<long long>(storage_offset),
            "storage_samples total");
  check_int(layout.source_device_bytes(),
            static_cast<long long>(coded_offset) * 2,
            "source_device_bytes");
  check_int(layout.recon_device_bytes(),
            static_cast<long long>(storage_offset) * 2,
            "recon_device_bytes");

  const av2enc::cuda::fmtdev::FilterGeometry filter =
      av2enc::cuda::fmtdev::make_filter_geometry(layout);
  check(filter.valid(), "filter geometry valid");
  for (int plane = 0; plane < 3; ++plane) {
    const auto& f = filter.planes[plane];
    if (plane >= reference->num_planes) {
      check(!f.present, "absent filter plane");
      continue;
    }
    check(f.present, "present filter plane");
    check_int(av2enc::cuda::fmtdev::filter_plane_width(filter, plane),
              reference->planes[plane].coded_width, "filter plane width");
    check_int(av2enc::cuda::fmtdev::filter_plane_height(filter, plane),
              reference->planes[plane].coded_height, "filter plane height");
    check_int(f.coded_mi_cols, f.coded_w / 4, "filter coded_mi_cols");
    check_int(f.coded_mi_rows, f.coded_h / 4, "filter coded_mi_rows");
    check_int(f.edge4_cols, f.coded_mi_cols > 0 ? f.coded_mi_cols - 1 : 0,
              "filter edge4_cols");
    check_int(f.edge4_rows, f.coded_mi_rows > 0 ? f.coded_mi_rows - 1 : 0,
              "filter edge4_rows");
  }
}

void check_block_rects(const av2enc::fmt::Format& format) {
  const av2enc::cuda::fmtdev::FrameLayout layout =
      av2enc::cuda::fmtdev::make_frame_layout(format);
  const auto reference = av2enc::fmt::make_frame_geometry(format);
  if (!reference.has_value()) return;
  const av2enc::fmt::BlockSize sizes[] = {
      av2enc::fmt::BlockSize::k4x4,   av2enc::fmt::BlockSize::k4x8,
      av2enc::fmt::BlockSize::k8x4,   av2enc::fmt::BlockSize::k8x8,
      av2enc::fmt::BlockSize::k8x16,  av2enc::fmt::BlockSize::k16x8,
      av2enc::fmt::BlockSize::k16x16, av2enc::fmt::BlockSize::k16x32,
      av2enc::fmt::BlockSize::k32x16, av2enc::fmt::BlockSize::k32x32,
      av2enc::fmt::BlockSize::k32x64, av2enc::fmt::BlockSize::k64x32,
      av2enc::fmt::BlockSize::k64x64, av2enc::fmt::BlockSize::k64x128,
      av2enc::fmt::BlockSize::k128x64, av2enc::fmt::BlockSize::k128x128,
      av2enc::fmt::BlockSize::k128x256, av2enc::fmt::BlockSize::k256x128,
      av2enc::fmt::BlockSize::k256x256, av2enc::fmt::BlockSize::k4x16,
      av2enc::fmt::BlockSize::k16x4,  av2enc::fmt::BlockSize::k8x32,
      av2enc::fmt::BlockSize::k32x8,  av2enc::fmt::BlockSize::k16x64,
      av2enc::fmt::BlockSize::k64x16, av2enc::fmt::BlockSize::k4x32,
      av2enc::fmt::BlockSize::k32x4,  av2enc::fmt::BlockSize::k8x64,
      av2enc::fmt::BlockSize::k64x8,
  };
  const int max_mi_cols = static_cast<int>(reference->mi_cols);
  const int max_mi_rows = static_cast<int>(reference->mi_rows);
  const auto positions = [](int limit) {
    std::vector<int> out = {0};
    if (limit > 1) out.push_back(1);
    if (limit > 2) out.push_back(limit / 2);
    if (limit > 1) out.push_back(limit - 1);
    return out;
  };
  const std::vector<int> cols = positions(max_mi_cols);
  const std::vector<int> rows = positions(max_mi_rows);
  for (const av2enc::fmt::BlockSize size : sizes) {
    const int luma_w = av2enc::fmt::block_width(size);
    const int luma_h = av2enc::fmt::block_height(size);
    for (int mi_col : cols) {
      for (int mi_row : rows) {
        for (int plane = 0; plane < 3; ++plane) {
          const auto expected = av2enc::fmt::block_rect(
              *reference, plane, static_cast<std::uint32_t>(mi_col * 4),
              static_cast<std::uint32_t>(mi_row * 4), size);
          av2enc::cuda::fmtdev::PlaneBlockRect actual;
          const bool ok = av2enc::cuda::fmtdev::plane_block_rect(
              layout, plane, mi_col * 4, mi_row * 4, luma_w, luma_h, &actual);
          check_int(ok, expected.has_value(), "block rect availability");
          if (!ok || !expected.has_value()) continue;
          check_int(actual.x, expected->x, "block rect x");
          check_int(actual.y, expected->y, "block rect y");
          check_int(actual.width, expected->width, "block rect width");
          check_int(actual.height, expected->height, "block rect height");
          check_int(actual.visible_width, expected->visible_width,
                    "block rect visible_width");
          check_int(actual.visible_height, expected->visible_height,
                    "block rect visible_height");
          check_int(actual.inside, expected->inside, "block rect inside");
          check_int(actual.partial, expected->partial, "block rect partial");
        }
      }
    }
  }
}

void check_deblock_reference() {
  // The CPU deblock component is 8-bit 4:2:0 only; its plane extents must agree
  // with the shared filter geometry.
  for (const auto [w, h] : {std::pair<int, int>{64, 64}, {100, 100},
                            {640, 360}, {1920, 1080}}) {
    av2enc::fmt::Format format;
    format.width = static_cast<std::uint32_t>(w);
    format.height = static_cast<std::uint32_t>(h);
    format.chroma = av2enc::fmt::ChromaFormat::k420;
    format.depth = av2enc::fmt::BitDepth::k8;
    const auto geometry = av2enc::fmt::make_frame_geometry(format);
    if (!geometry.has_value()) continue;
    av2enc::deblock::DeblockFrame frame;
    frame.mi_cols = static_cast<int>(geometry->mi_cols);
    frame.mi_rows = static_cast<int>(geometry->mi_rows);
    frame.subsampling_x = 1;
    frame.subsampling_y = 1;
    const auto filter = av2enc::cuda::fmtdev::make_filter_geometry(format);
    for (int plane = 0; plane < 3; ++plane) {
      check_int(av2enc::deblock::deblock_plane_width(frame, plane),
                av2enc::cuda::fmtdev::filter_plane_width(filter, plane),
                "deblock plane width");
      check_int(av2enc::deblock::deblock_plane_height(frame, plane),
                av2enc::cuda::fmtdev::filter_plane_height(filter, plane),
                "deblock plane height");
    }
  }
}

void check_bitstream_layout() {
  for (int blocks : {1, 4, 16, 256}) {
    const auto layout = av2enc::cuda::fmtdev::make_bitstream_layout(blocks);
    check_int(layout.level_capacity, static_cast<long long>(blocks) * 1024,
              "bitstream level capacity");
    check_int(layout.eob_capacity, static_cast<long long>(blocks) * 3,
              "bitstream eob capacity");
    check_int(layout.block_capacity, blocks, "bitstream block capacity");
    check_int(layout.level_bytes(),
              static_cast<long long>(blocks) * 1024 * 4, "bitstream level bytes");
    check_int(layout.eob_bytes(), static_cast<long long>(blocks) * 3 * 4,
              "bitstream eob bytes");
  }
}

}  // namespace

int main() {
  const av2enc::fmt::ChromaFormat formats[] = {
      av2enc::fmt::ChromaFormat::k400, av2enc::fmt::ChromaFormat::k420,
      av2enc::fmt::ChromaFormat::k422, av2enc::fmt::ChromaFormat::k444};
  const av2enc::fmt::BitDepth depths[] = {av2enc::fmt::BitDepth::k8,
                                          av2enc::fmt::BitDepth::k10,
                                          av2enc::fmt::BitDepth::k12};
  const std::pair<int, int> sizes[] = {
      {1, 1},     {63, 65},   {64, 64},   {65, 63},   {100, 100},
      {101, 99},  {128, 64},  {640, 360}, {1920, 1080}};
  for (const auto chroma : formats) {
    for (const auto depth : depths) {
      for (const auto [w, h] : sizes) {
        av2enc::fmt::Format format;
        format.width = static_cast<std::uint32_t>(w);
        format.height = static_cast<std::uint32_t>(h);
        format.chroma = chroma;
        format.depth = depth;
        check_geometry(format);
        check_block_rects(format);
      }
    }
  }
  check_deblock_reference();
  check_bitstream_layout();
  if (failures == 0) {
    std::printf("format_device_test: PASS (%lld checks)\n", checks);
    return 0;
  }
  std::fprintf(stderr, "format_device_test: %d failure(s)\n", failures);
  return 1;
}
