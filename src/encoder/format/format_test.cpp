// Tests for the standalone AV2 format / high-bit-depth layer.
//
// Run with ./build_and_test.sh.  Nothing here writes a bitstream or invokes a
// decoder; the tests cover geometry, chroma residual sizes, edge rules, sample
// range/clipping, conversion round-trips, and the per-format requirements
// checklist.

#include "format.h"

#include "format_tables.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace f = av2enc::fmt;

namespace {

int g_failures = 0;
long long g_checks = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(condition)) {                                                      \
      ++g_failures;                                                          \
      std::cerr << __FILE__ << ':' << __LINE__                               \
                << ": CHECK failed: " #condition << '\n';                    \
    }                                                                        \
  } while (false)

#define CHECK_EQ_INT(actual, expected)                                       \
  do {                                                                       \
    const long long actual_value = static_cast<long long>(actual);           \
    const long long expected_value = static_cast<long long>(expected);       \
    ++g_checks;                                                              \
    if (actual_value != expected_value) {                                    \
      ++g_failures;                                                          \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK_EQ failed: "      \
                << #actual << " = " << actual_value << ", expected "         \
                << expected_value << '\n';                                   \
    }                                                                        \
  } while (false)

#define CHECK_STREQ(actual, expected)                                        \
  do {                                                                       \
    const std::string actual_value = (actual);                               \
    const std::string expected_value = (expected);                           \
    ++g_checks;                                                              \
    if (actual_value != expected_value) {                                    \
      ++g_failures;                                                          \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK_STREQ failed: "   \
                << #actual << " = \"" << actual_value << "\", expected \""   \
                << expected_value << "\"\n";                                 \
    }                                                                        \
  } while (false)

// ---------------------------------------------------------------------------
// Small fixtures
// ---------------------------------------------------------------------------

struct SourceFixture {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  f::Source420_8 view;
};

SourceFixture make_source(std::uint32_t width, std::uint32_t height) {
  SourceFixture fixture;
  fixture.width = width;
  fixture.height = height;
  const std::uint32_t chroma_width = (width + 1U) / 2U;
  const std::uint32_t chroma_height = (height + 1U) / 2U;
  fixture.y.resize(static_cast<std::size_t>(width) * height);
  fixture.u.resize(static_cast<std::size_t>(chroma_width) * chroma_height);
  fixture.v.resize(static_cast<std::size_t>(chroma_width) * chroma_height);

  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      fixture.y[static_cast<std::size_t>(y) * width + x] =
          static_cast<std::uint8_t>((x * 5U + y * 3U + 7U) & 0xFFU);
    }
  }
  for (std::uint32_t y = 0; y < chroma_height; ++y) {
    for (std::uint32_t x = 0; x < chroma_width; ++x) {
      const std::size_t index = static_cast<std::size_t>(y) * chroma_width + x;
      fixture.u[index] = static_cast<std::uint8_t>((x * 11U + y * 7U + 29U) & 0xFFU);
      fixture.v[index] = static_cast<std::uint8_t>((x * 3U + y * 13U + 61U) & 0xFFU);
    }
  }

  // Pin the extremes so the round-trip test covers 0 and 255.
  fixture.y[0] = 0;
  fixture.y[1] = 255;
  fixture.u[0] = 0;
  fixture.u[1 % fixture.u.size()] = 255;
  fixture.v[0] = 255;
  fixture.v[1 % fixture.v.size()] = 0;

  fixture.view.width = width;
  fixture.view.height = height;
  fixture.view.y = {fixture.y.data(), width};
  fixture.view.u = {fixture.u.data(), chroma_width};
  fixture.view.v = {fixture.v.data(), chroma_width};
  return fixture;
}

bool plane8_equals(const f::Plane8& plane, const std::vector<std::uint8_t>& data,
                   std::uint32_t width, std::uint32_t height) {
  if (plane.width != width || plane.height != height || plane.stride < width ||
      data.size() != static_cast<std::size_t>(width) * height) {
    return false;
  }
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      if (plane.data[static_cast<std::size_t>(y) * plane.stride + x] !=
          data[static_cast<std::size_t>(y) * width + x]) {
        return false;
      }
    }
  }
  return true;
}

const f::ChecklistItem* find_item(const std::vector<f::ChecklistItem>& items,
                                  const std::string& id) {
  for (const f::ChecklistItem& item : items) {
    if (item.id == id) {
      return &item;
    }
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Bit depth
// ---------------------------------------------------------------------------

void test_depth_helpers() {
  CHECK_EQ_INT(f::bit_depth_value(f::BitDepth::k8), 8);
  CHECK_EQ_INT(f::bit_depth_value(f::BitDepth::k10), 10);
  CHECK_EQ_INT(f::bit_depth_value(f::BitDepth::k12), 12);

  CHECK_EQ_INT(f::sample_max(f::BitDepth::k8), 255);
  CHECK_EQ_INT(f::sample_max(f::BitDepth::k10), 1023);
  CHECK_EQ_INT(f::sample_max(f::BitDepth::k12), 4095);

  // §6.4.1 Table 6.3: 0 = 10-bit, 1 = 8-bit; > 1 is reserved.
  CHECK_EQ_INT(f::bit_depth_idc(f::BitDepth::k10), 0);
  CHECK_EQ_INT(f::bit_depth_idc(f::BitDepth::k8), 1);
  CHECK_EQ_INT(f::bit_depth_idc(f::BitDepth::k12), -1);
  CHECK(f::bit_depth_from_idc(0) == f::BitDepth::k10);
  CHECK(f::bit_depth_from_idc(1) == f::BitDepth::k8);
  CHECK(!f::bit_depth_from_idc(2).has_value());

  CHECK_EQ_INT(f::max_quantizer(f::BitDepth::k8), 255);
  CHECK_EQ_INT(f::max_quantizer(f::BitDepth::k10), 303);
  CHECK_EQ_INT(f::max_quantizer(f::BitDepth::k12), 351);
  CHECK_EQ_INT(f::base_q_idx_bits(f::BitDepth::k8), 8);
  CHECK_EQ_INT(f::base_q_idx_bits(f::BitDepth::k10), 9);
  CHECK_EQ_INT(f::base_q_idx_bits(f::BitDepth::k12), 9);

  // Clip1 / §4.8.
  CHECK_EQ_INT(f::clip_sample(-1, f::BitDepth::k8), 0);
  CHECK_EQ_INT(f::clip_sample(0, f::BitDepth::k8), 0);
  CHECK_EQ_INT(f::clip_sample(255, f::BitDepth::k8), 255);
  CHECK_EQ_INT(f::clip_sample(256, f::BitDepth::k8), 255);
  CHECK_EQ_INT(f::clip_sample(-9999, f::BitDepth::k10), 0);
  CHECK_EQ_INT(f::clip_sample(1023, f::BitDepth::k10), 1023);
  CHECK_EQ_INT(f::clip_sample(1024, f::BitDepth::k10), 1023);
  CHECK_EQ_INT(f::clip_sample(4095, f::BitDepth::k12), 4095);
  CHECK_EQ_INT(f::clip_sample(4096, f::BitDepth::k12), 4095);

  // 8 -> high depth is a left shift; high -> 8 rounds to nearest.
  CHECK_EQ_INT(f::expand_8_to_depth(0, f::BitDepth::k10), 0);
  CHECK_EQ_INT(f::expand_8_to_depth(255, f::BitDepth::k10), 1020);
  CHECK_EQ_INT(f::expand_8_to_depth(255, f::BitDepth::k12), 4080);
  CHECK_EQ_INT(f::reduce_to_8(1020, f::BitDepth::k10), 255);
  CHECK_EQ_INT(f::reduce_to_8(1023, f::BitDepth::k10), 255);
  CHECK_EQ_INT(f::reduce_to_8(2, f::BitDepth::k10), 1);
  CHECK_EQ_INT(f::reduce_to_8(1, f::BitDepth::k10), 0);
  CHECK_EQ_INT(f::reduce_to_8(4080, f::BitDepth::k12), 255);
  CHECK_EQ_INT(f::reduce_to_8(4095, f::BitDepth::k12), 255);
  CHECK_EQ_INT(f::reduce_to_8(7, f::BitDepth::k12), 0);

  // 8 -> 10 -> 8 and 10 -> 12 -> 10 are exact for every value.
  for (int value = 0; value <= 255; ++value) {
    const auto sample = static_cast<std::uint16_t>(value);
    const auto high = f::expand_8_to_depth(static_cast<std::uint8_t>(value),
                                           f::BitDepth::k10);
    CHECK_EQ_INT(f::reduce_to_8(high, f::BitDepth::k10), value);
    CHECK_EQ_INT(f::convert_sample_depth(
                     f::convert_sample_depth(sample, f::BitDepth::k8,
                                             f::BitDepth::k12),
                     f::BitDepth::k12, f::BitDepth::k8),
                 value);
  }
  for (int value = 0; value <= 1023; ++value) {
    const auto sample = static_cast<std::uint16_t>(value);
    CHECK_EQ_INT(f::convert_sample_depth(
                     f::convert_sample_depth(sample, f::BitDepth::k10,
                                             f::BitDepth::k12),
                     f::BitDepth::k12, f::BitDepth::k10),
                 value);
  }
  // 12 -> 10 -> 12 is deliberately lossy.
  CHECK_EQ_INT(f::convert_sample_depth(4095, f::BitDepth::k12, f::BitDepth::k10),
               1023);
  CHECK_EQ_INT(f::convert_sample_depth(1023, f::BitDepth::k10, f::BitDepth::k12),
               4092);
}

// ---------------------------------------------------------------------------
// Chroma formats
// ---------------------------------------------------------------------------

void test_chroma_mapping() {
  CHECK_EQ_INT(f::chroma_format_idc(f::ChromaFormat::k420), 0);
  CHECK_EQ_INT(f::chroma_format_idc(f::ChromaFormat::k400), 1);
  CHECK_EQ_INT(f::chroma_format_idc(f::ChromaFormat::k444), 2);
  CHECK_EQ_INT(f::chroma_format_idc(f::ChromaFormat::k422), 3);

  for (int idc = 0; idc <= 3; ++idc) {
    const auto chroma = f::chroma_format_from_idc(idc);
    CHECK(chroma.has_value());
    if (chroma) {
      CHECK_EQ_INT(f::chroma_format_idc(*chroma), idc);
    }
  }
  CHECK(!f::chroma_format_from_idc(4).has_value());
  CHECK(!f::chroma_format_from_idc(-1).has_value());

  CHECK_EQ_INT(f::subsampling_x(f::ChromaFormat::k420), 1);
  CHECK_EQ_INT(f::subsampling_y(f::ChromaFormat::k420), 1);
  CHECK_EQ_INT(f::subsampling_x(f::ChromaFormat::k422), 1);
  CHECK_EQ_INT(f::subsampling_y(f::ChromaFormat::k422), 0);
  CHECK_EQ_INT(f::subsampling_x(f::ChromaFormat::k444), 0);
  CHECK_EQ_INT(f::subsampling_y(f::ChromaFormat::k444), 0);
  CHECK_EQ_INT(f::subsampling_x(f::ChromaFormat::k400), 1);
  CHECK_EQ_INT(f::subsampling_y(f::ChromaFormat::k400), 1);

  CHECK_EQ_INT(f::num_planes(f::ChromaFormat::k400), 1);
  CHECK_EQ_INT(f::num_planes(f::ChromaFormat::k420), 3);
  CHECK_EQ_INT(f::num_planes(f::ChromaFormat::k422), 3);
  CHECK_EQ_INT(f::num_planes(f::ChromaFormat::k444), 3);
  CHECK(!f::has_chroma(f::ChromaFormat::k400));
  CHECK(f::has_chroma(f::ChromaFormat::k444));

  CHECK_STREQ(f::chroma_format_name(f::ChromaFormat::k422), "4:2:2");
  CHECK_STREQ(f::chroma_format_name(f::ChromaFormat::k444), "4:4:4");
}

// ---------------------------------------------------------------------------
// Blocks and Subsampled_Size
// ---------------------------------------------------------------------------

void test_block_tables() {
  CHECK_EQ_INT(f::kBlockSizeCount, 29);
  CHECK_EQ_INT(f::block_width(f::BlockSize::k4x4), 4);
  CHECK_EQ_INT(f::block_height(f::BlockSize::k4x4), 4);
  CHECK_EQ_INT(f::block_width(f::BlockSize::k64x64), 64);
  CHECK_EQ_INT(f::block_height(f::BlockSize::k64x64), 64);
  CHECK_EQ_INT(f::block_width(f::BlockSize::k8x64), 8);
  CHECK_EQ_INT(f::block_height(f::BlockSize::k8x64), 64);
  CHECK_STREQ(f::block_size_name(f::BlockSize::k32x16), "BLOCK_32X16");

  CHECK(f::find_block_size(64, 64) == f::BlockSize::k64x64);
  CHECK(f::find_block_size(4, 16) == f::BlockSize::k4x16);
  CHECK(!f::find_block_size(12, 8).has_value());
  CHECK(!f::find_block_size(0, 4).has_value());

  auto size = [](int w, int h, f::ChromaFormat chroma) {
    return f::chroma_residual_size(w, h, chroma);
  };

  // 4:2:0 halves both dimensions, but Subsampled_Size has explicit entries.
  CHECK(size(4, 4, f::ChromaFormat::k420) == f::BlockSize::k4x4);
  CHECK(size(4, 8, f::ChromaFormat::k420) == f::BlockSize::k4x4);
  CHECK(size(8, 4, f::ChromaFormat::k420) == f::BlockSize::k4x4);
  CHECK(size(8, 8, f::ChromaFormat::k420) == f::BlockSize::k4x4);
  CHECK(size(64, 64, f::ChromaFormat::k420) == f::BlockSize::k32x32);
  CHECK(size(16, 32, f::ChromaFormat::k420) == f::BlockSize::k8x16);

  // 4:2:2 halves width only; width-4 luma blocks have no legal chroma size.
  CHECK(size(4, 4, f::ChromaFormat::k422) == f::BlockSize::k4x4);
  CHECK(size(8, 8, f::ChromaFormat::k422) == f::BlockSize::k4x8);
  CHECK(size(8, 4, f::ChromaFormat::k422) == f::BlockSize::k4x4);
  CHECK(size(16, 8, f::ChromaFormat::k422) == f::BlockSize::k8x8);
  CHECK(!size(4, 8, f::ChromaFormat::k422).has_value());
  CHECK(!size(4, 16, f::ChromaFormat::k422).has_value());
  CHECK(!size(64, 128, f::ChromaFormat::k422).has_value());
  CHECK(size(128, 128, f::ChromaFormat::k422) == f::BlockSize::k64x128);

  // 4:4:4 uses the luma block size directly.
  CHECK(size(8, 8, f::ChromaFormat::k444) == f::BlockSize::k8x8);
  CHECK(size(16, 16, f::ChromaFormat::k444) == f::BlockSize::k16x16);
  CHECK(size(64, 64, f::ChromaFormat::k444) == f::BlockSize::k64x64);

  // Monochrome has no chroma residual.
  CHECK(!size(8, 8, f::ChromaFormat::k400).has_value());
  CHECK(!f::chroma_residual_size(f::BlockSize::k8x8,
                                 f::ChromaFormat::k400)
             .has_value());
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

struct GeometryExpectation {
  std::uint32_t width;
  std::uint32_t height;
  f::ChromaFormat chroma;
  std::uint32_t mi_cols;
  std::uint32_t mi_rows;
  std::uint32_t coded_width;
  std::uint32_t coded_height;
  int num_planes;
  std::uint32_t luma_visible_w;
  std::uint32_t luma_visible_h;
  std::uint32_t chroma_coded_w;
  std::uint32_t chroma_coded_h;
  std::uint32_t chroma_visible_w;
  std::uint32_t chroma_visible_h;
  std::size_t total_samples;
};

void test_geometry_case(const GeometryExpectation& expected) {
  const f::Format format{expected.width, expected.height, expected.chroma,
                         f::BitDepth::k10};
  const auto geometry = f::make_frame_geometry(format);
  CHECK(geometry.has_value());
  if (!geometry) {
    return;
  }

  CHECK_EQ_INT(geometry->mi_cols, expected.mi_cols);
  CHECK_EQ_INT(geometry->mi_rows, expected.mi_rows);
  CHECK_EQ_INT(geometry->coded_width, expected.coded_width);
  CHECK_EQ_INT(geometry->coded_height, expected.coded_height);
  CHECK_EQ_INT(geometry->num_planes, expected.num_planes);
  CHECK_EQ_INT(geometry->planes[0].visible_width, expected.luma_visible_w);
  CHECK_EQ_INT(geometry->planes[0].visible_height, expected.luma_visible_h);
  CHECK_EQ_INT(geometry->planes[0].coded_width, expected.coded_width);
  CHECK_EQ_INT(geometry->planes[0].coded_height, expected.coded_height);
  CHECK_EQ_INT(geometry->planes[0].stride, expected.coded_width);
  CHECK_EQ_INT(geometry->planes[0].offset, 0);
  CHECK_EQ_INT(geometry->total_samples, expected.total_samples);
  CHECK_EQ_INT(geometry->frame_bytes, expected.total_samples * 2U);

  if (expected.num_planes > 1) {
    CHECK_EQ_INT(geometry->planes[1].coded_width, expected.chroma_coded_w);
    CHECK_EQ_INT(geometry->planes[1].coded_height, expected.chroma_coded_h);
    CHECK_EQ_INT(geometry->planes[1].visible_width, expected.chroma_visible_w);
    CHECK_EQ_INT(geometry->planes[1].visible_height, expected.chroma_visible_h);
    CHECK_EQ_INT(geometry->planes[2].coded_width, expected.chroma_coded_w);
    CHECK_EQ_INT(geometry->planes[2].coded_height, expected.chroma_coded_h);
    CHECK_EQ_INT(geometry->planes[1].offset, geometry->planes[0].sample_count);
    CHECK_EQ_INT(geometry->planes[2].offset,
                 geometry->planes[0].sample_count +
                     geometry->planes[1].sample_count);
  }
}

void test_geometry() {
  test_geometry_case({64, 64, f::ChromaFormat::k420, 16, 16, 64, 64, 3,
                      64, 64, 32, 32, 32, 32, 6144});
  test_geometry_case({640, 360, f::ChromaFormat::k420, 160, 90, 640, 360, 3,
                      640, 360, 320, 180, 320, 180, 345600});
  // Odd dimensions round the coded luma extent up to 8, then subsample.
  test_geometry_case({641, 361, f::ChromaFormat::k420, 162, 92, 648, 368, 3,
                      641, 361, 324, 184, 321, 181, 357696});
  test_geometry_case({640, 360, f::ChromaFormat::k422, 160, 90, 640, 360, 3,
                      640, 360, 320, 360, 320, 360, 460800});
  test_geometry_case({641, 361, f::ChromaFormat::k422, 162, 92, 648, 368, 3,
                      641, 361, 324, 368, 321, 361, 476928});
  test_geometry_case({640, 360, f::ChromaFormat::k444, 160, 90, 640, 360, 3,
                      640, 360, 640, 360, 640, 360, 691200});
  test_geometry_case({641, 361, f::ChromaFormat::k444, 162, 92, 648, 368, 3,
                      641, 361, 648, 368, 641, 361, 715392});
  test_geometry_case({640, 360, f::ChromaFormat::k400, 160, 90, 640, 360, 1,
                      640, 360, 0, 0, 0, 0, 230400});

  // 10-bit and 12-bit use the same geometry but a wider container.
  const f::Format ten{641, 361, f::ChromaFormat::k444, f::BitDepth::k10};
  const f::Format twelve{641, 361, f::ChromaFormat::k444, f::BitDepth::k12};
  const auto ten_geometry = f::make_frame_geometry(ten);
  const auto twelve_geometry = f::make_frame_geometry(twelve);
  CHECK(ten_geometry.has_value());
  CHECK(twelve_geometry.has_value());
  if (ten_geometry && twelve_geometry) {
    CHECK_EQ_INT(ten_geometry->total_samples, twelve_geometry->total_samples);
    CHECK_EQ_INT(ten_geometry->frame_bytes, twelve_geometry->frame_bytes);
  }

  // Invalid formats produce no geometry.
  CHECK(!f::make_frame_geometry({0, 64, f::ChromaFormat::k420, f::BitDepth::k8})
             .has_value());
  CHECK(!f::make_frame_geometry({64, 0, f::ChromaFormat::k420, f::BitDepth::k8})
             .has_value());
  CHECK(!f::make_frame_geometry({f::kMaxDimension + 1U, 64,
                                 f::ChromaFormat::k420, f::BitDepth::k8})
             .has_value());
}

// ---------------------------------------------------------------------------
// Format validation and profiles
// ---------------------------------------------------------------------------

void test_format_validation() {
  const f::Format eight_420{64, 64, f::ChromaFormat::k420, f::BitDepth::k8};
  const f::Format ten_420{64, 64, f::ChromaFormat::k420, f::BitDepth::k10};
  const f::Format ten_422{64, 64, f::ChromaFormat::k422, f::BitDepth::k10};
  const f::Format ten_444{64, 64, f::ChromaFormat::k444, f::BitDepth::k10};
  const f::Format twelve_444{64, 64, f::ChromaFormat::k444, f::BitDepth::k12};

  CHECK(f::validate_format(eight_420).ok());
  CHECK(f::validate_format(ten_420).ok());
  CHECK(f::validate_format(ten_422).ok());
  CHECK(f::validate_format(ten_444).ok());

  // 12-bit is internally supported but cannot be signalled.
  CHECK(!f::validate_format(twelve_444).ok());
  CHECK(f::validate_format(twelve_444).status ==
        f::FormatStatus::kReservedBitDepth);
  CHECK(f::is_internally_supported(twelve_444));
  CHECK(!f::is_signalable(twelve_444));

  CHECK_EQ_INT(f::profile_idc(eight_420), 0);
  CHECK_EQ_INT(f::profile_idc(ten_420), 0);
  CHECK_EQ_INT(f::profile_idc(ten_422), 3);
  CHECK_EQ_INT(f::profile_idc(ten_444), 4);
  CHECK_EQ_INT(f::profile_idc(twelve_444), -1);
  CHECK(f::is_signalable(eight_420));
  CHECK(f::is_signalable(ten_422));
  CHECK(f::is_signalable(ten_444));
  CHECK(f::is_signalable({64, 64, f::ChromaFormat::k400, f::BitDepth::k10}));

  // 8-bit 4:2:2 / 4:4:4 are allowed by the same profiles.
  CHECK(f::is_signalable({64, 64, f::ChromaFormat::k422, f::BitDepth::k8}));
  CHECK(f::is_signalable({64, 64, f::ChromaFormat::k444, f::BitDepth::k8}));

  CHECK(f::validate_format({0, 64, f::ChromaFormat::k420, f::BitDepth::k8})
            .status == f::FormatStatus::kZeroDimension);
  CHECK(f::validate_format({64, 64, f::ChromaFormat::k420,
                            static_cast<f::BitDepth>(9)})
            .status == f::FormatStatus::kInvalidBitDepth);
  CHECK(f::validate_format({64, 64, static_cast<f::ChromaFormat>(9),
                            f::BitDepth::k8})
            .status == f::FormatStatus::kInvalidChromaFormat);
  CHECK(f::validate_format({f::kMaxDimension + 1U, 64,
                            f::ChromaFormat::k420, f::BitDepth::k8})
            .status == f::FormatStatus::kDimensionTooLarge);
}

// ---------------------------------------------------------------------------
// Range / clipping
// ---------------------------------------------------------------------------

void test_range_and_buffers() {
  f::Picture picture;
  std::string error;
  CHECK(f::allocate_picture({8, 8, f::ChromaFormat::k420, f::BitDepth::k10},
                            picture, error));
  CHECK(error.empty());
  CHECK_EQ_INT(picture.planes[0].width, 8);
  CHECK_EQ_INT(picture.planes[0].stride, 8);
  CHECK_EQ_INT(picture.planes[1].width, 4);
  CHECK_EQ_INT(picture.planes[1].height, 4);
  CHECK(f::samples_in_range(picture));

  // Edge-clamped reads never leave the visible rectangle.
  f::PlaneBuffer plane;
  plane.width = 4;
  plane.height = 3;
  plane.visible_width = 4;
  plane.visible_height = 3;
  plane.stride = 4;
  plane.samples = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  CHECK_EQ_INT(f::sample_clamped(plane, -1, -1), 1);
  CHECK_EQ_INT(f::sample_clamped(plane, 99, 99), 12);
  CHECK_EQ_INT(f::sample_clamped(plane, 2, 1), 7);
  CHECK_EQ_INT(f::sample_clamped(plane, -5, 1), 5);

  // Out-of-range high-bit-depth samples are reported and clipped.
  picture.planes[0].at(0, 0) = 1024;
  picture.planes[0].at(1, 0) = 5000;
  picture.planes[1].at(0, 0) = 2048;
  f::RangeReport report = f::check_sample_range(picture);
  CHECK_EQ_INT(report.num_planes, 3);
  CHECK_EQ_INT(report.planes[0].out_of_range, 2);
  CHECK_EQ_INT(report.planes[1].out_of_range, 1);
  CHECK_EQ_INT(report.planes[2].out_of_range, 0);
  CHECK_EQ_INT(report.total_out_of_range, 3);
  CHECK_EQ_INT(report.planes[0].max_value, 5000);
  CHECK(!f::samples_in_range(picture));
  CHECK_EQ_INT(f::clamp_samples(picture), 3);
  CHECK_EQ_INT(picture.planes[0].at(0, 0), 1023);
  CHECK_EQ_INT(picture.planes[0].at(1, 0), 1023);
  CHECK_EQ_INT(picture.planes[1].at(0, 0), 1023);
  CHECK(f::samples_in_range(picture));

  // Replicated padding: coded 8x8 from visible 7x7.
  f::Picture padded;
  CHECK(f::allocate_picture({7, 7, f::ChromaFormat::k444, f::BitDepth::k12},
                            padded, error));
  for (std::uint32_t y = 0; y < 7; ++y) {
    for (std::uint32_t x = 0; x < 7; ++x) {
      padded.planes[0].at(x, y) = static_cast<std::uint16_t>(x + y * 100);
    }
  }
  f::replicate_borders(padded.planes[0]);
  CHECK_EQ_INT(padded.planes[0].at(7, 0), padded.planes[0].at(6, 0));
  CHECK_EQ_INT(padded.planes[0].at(0, 7), padded.planes[0].at(0, 6));
  CHECK_EQ_INT(padded.planes[0].at(7, 7), padded.planes[0].at(6, 6));
  CHECK_EQ_INT(padded.planes[0].at(6, 0), 6);
  CHECK_EQ_INT(padded.planes[0].at(0, 6), 600);
}

// ---------------------------------------------------------------------------
// Conversion round-trips
// ---------------------------------------------------------------------------

void check_conversion(const SourceFixture& source, f::ChromaFormat chroma,
                      f::BitDepth depth) {
  const int shift = f::bit_depth_value(depth) - 8;
  const std::uint32_t source_chroma_width = (source.width + 1U) / 2U;

  f::Picture picture;
  const f::ConvertResult converted =
      f::convert_420_8_to_picture(source.view, chroma, depth, picture);
  CHECK(converted.ok());
  if (!converted.ok()) {
    std::cerr << "  conversion failed: " << converted.error << '\n';
    return;
  }

  const auto& geometry = picture.geometry;
  CHECK_EQ_INT(geometry.num_planes,
               chroma == f::ChromaFormat::k400 ? 1 : 3);
  CHECK_EQ_INT(picture.planes[0].visible_width, source.width);
  CHECK_EQ_INT(picture.planes[0].visible_height, source.height);
  CHECK(f::samples_in_range(picture));

  // Luma is an exact left shift of the source.
  for (std::uint32_t y = 0; y < source.height; ++y) {
    for (std::uint32_t x = 0; x < source.width; ++x) {
      const auto expected = static_cast<std::uint16_t>(
          static_cast<std::uint32_t>(
              source.y[static_cast<std::size_t>(y) * source.width + x])
          << static_cast<unsigned>(shift));
      CHECK_EQ_INT(picture.planes[0].at(x, y), expected);
    }
  }

  // Chroma replication follows the destination subsampling.
  for (int plane = 1; plane < geometry.num_planes; ++plane) {
    const std::vector<std::uint8_t>& src_plane = plane == 1 ? source.u : source.v;
    const f::PlaneBuffer& dst = picture.planes[plane];
    for (std::uint32_t y = 0; y < dst.visible_height; ++y) {
      for (std::uint32_t x = 0; x < dst.visible_width; ++x) {
        const std::uint32_t sx =
            std::min((x << static_cast<unsigned>(geometry.sub_x)) >> 1,
                     source_chroma_width - 1U);
        const std::uint32_t sy =
            std::min((y << static_cast<unsigned>(geometry.sub_y)) >> 1,
                     (source.height + 1U) / 2U - 1U);
        const auto expected = static_cast<std::uint16_t>(
            static_cast<std::uint32_t>(
                src_plane[static_cast<std::size_t>(sy) * source_chroma_width + sx])
            << static_cast<unsigned>(shift));
        CHECK_EQ_INT(dst.at(x, y), expected);
      }
    }
    // Padding mirrors the visible edge.
    if (dst.width > dst.visible_width) {
      CHECK_EQ_INT(dst.at(dst.width - 1U, 0),
                   dst.at(dst.visible_width - 1U, 0));
    }
    if (dst.height > dst.visible_height) {
      CHECK_EQ_INT(dst.at(0, dst.height - 1U),
                   dst.at(0, dst.visible_height - 1U));
    }
  }

  // 4:0:0 has no chroma to convert back.
  f::Picture420_8 round_trip;
  const f::ConvertResult back = f::convert_to_420_8(picture, round_trip);
  if (chroma == f::ChromaFormat::k400) {
    CHECK(!back.ok());
    CHECK(back.status == f::ConvertStatus::kUnsupportedFormat);
    return;
  }
  CHECK(back.ok());
  if (!back.ok()) {
    std::cerr << "  reverse conversion failed: " << back.error << '\n';
    return;
  }

  CHECK_EQ_INT(round_trip.width, source.width);
  CHECK_EQ_INT(round_trip.height, source.height);
  CHECK(plane8_equals(round_trip.y, source.y, source.width, source.height));
  CHECK(plane8_equals(round_trip.u, source.u, source_chroma_width,
                      (source.height + 1U) / 2U));
  CHECK(plane8_equals(round_trip.v, source.v, source_chroma_width,
                      (source.height + 1U) / 2U));
}

void test_conversion_round_trips() {
  const SourceFixture even = make_source(64, 48);
  const SourceFixture odd = make_source(65, 49);

  const f::ChromaFormat formats[] = {f::ChromaFormat::k400,
                                     f::ChromaFormat::k420,
                                     f::ChromaFormat::k422,
                                     f::ChromaFormat::k444};
  const f::BitDepth depths[] = {f::BitDepth::k8, f::BitDepth::k10,
                                f::BitDepth::k12};

  for (const f::ChromaFormat chroma : formats) {
    for (const f::BitDepth depth : depths) {
      check_conversion(even, chroma, depth);
      check_conversion(odd, chroma, depth);
    }
  }
}

void test_conversion_errors() {
  const SourceFixture source = make_source(16, 16);
  f::Picture picture;
  std::string error;

  f::Source420_8 bad = source.view;
  bad.y.data = nullptr;
  CHECK(f::convert_420_8_to_picture(bad, f::ChromaFormat::k420,
                                    f::BitDepth::k10, picture)
            .status == f::ConvertStatus::kBadSource);

  bad = source.view;
  bad.y.stride = source.width - 1U;
  CHECK(f::convert_420_8_to_picture(bad, f::ChromaFormat::k420,
                                    f::BitDepth::k10, picture)
            .status == f::ConvertStatus::kBadSource);

  bad = source.view;
  bad.width = 0;
  CHECK(f::convert_420_8_to_picture(bad, f::ChromaFormat::k420,
                                    f::BitDepth::k10, picture)
            .status == f::ConvertStatus::kBadSource);

  CHECK(f::convert_420_8_to_picture(source.view,
                                    static_cast<f::ChromaFormat>(9),
                                    f::BitDepth::k10, picture)
            .status == f::ConvertStatus::kUnsupportedFormat);
  CHECK(f::convert_420_8_to_picture(source.view, f::ChromaFormat::k420,
                                    static_cast<f::BitDepth>(9), picture)
            .status == f::ConvertStatus::kUnsupportedFormat);

  // Monochrome cannot be turned back into 4:2:0.
  CHECK(f::convert_420_8_to_picture(source.view, f::ChromaFormat::k400,
                                    f::BitDepth::k10, picture)
            .ok());
  f::Picture420_8 out;
  CHECK(f::convert_to_420_8(picture, out).status ==
        f::ConvertStatus::kUnsupportedFormat);

  // Allocation error text is populated for a bad geometry.
  CHECK(!f::allocate_picture({0, 4, f::ChromaFormat::k420, f::BitDepth::k8},
                             picture, error));
  CHECK(!error.empty());
}

// ---------------------------------------------------------------------------
// Block rectangles / edge rules
// ---------------------------------------------------------------------------

void test_block_rects() {
  const auto geom_420 =
      f::make_frame_geometry({640, 360, f::ChromaFormat::k420, f::BitDepth::k10});
  CHECK(geom_420.has_value());
  if (geom_420) {
    const auto luma = f::block_rect(*geom_420, 0, 576, 320, f::BlockSize::k64x64);
    CHECK(luma.has_value());
    if (luma) {
      CHECK_EQ_INT(luma->x, 576);
      CHECK_EQ_INT(luma->y, 320);
      CHECK_EQ_INT(luma->visible_width, 64);
      CHECK_EQ_INT(luma->visible_height, 40);
      CHECK(luma->partial);
      CHECK(luma->inside);
    }

    const auto chroma =
        f::block_rect(*geom_420, 1, 576, 320, f::BlockSize::k64x64);
    CHECK(chroma.has_value());
    if (chroma) {
      CHECK_EQ_INT(chroma->x, 288);
      CHECK_EQ_INT(chroma->y, 160);
      CHECK_EQ_INT(chroma->width, 32);
      CHECK_EQ_INT(chroma->height, 32);
      CHECK_EQ_INT(chroma->visible_width, 32);
      CHECK_EQ_INT(chroma->visible_height, 20);
      CHECK(chroma->partial);
    }

    // A 4x4 luma block at the right/bottom edge: chroma residual is 4x4 but
    // only 2x2 samples are visible (the min-4 chroma block overlaps).
    const auto edge = f::block_rect(*geom_420, 0, 636, 356, f::BlockSize::k4x4);
    CHECK(edge.has_value());
    if (edge) {
      CHECK_EQ_INT(edge->visible_width, 4);
      CHECK_EQ_INT(edge->visible_height, 4);
      CHECK(!edge->partial);
    }
    const auto edge_chroma =
        f::block_rect(*geom_420, 1, 636, 356, f::BlockSize::k4x4);
    CHECK(edge_chroma.has_value());
    if (edge_chroma) {
      CHECK_EQ_INT(edge_chroma->x, 318);
      CHECK_EQ_INT(edge_chroma->y, 178);
      CHECK_EQ_INT(edge_chroma->width, 4);
      CHECK_EQ_INT(edge_chroma->height, 4);
      CHECK_EQ_INT(edge_chroma->visible_width, 2);
      CHECK_EQ_INT(edge_chroma->visible_height, 2);
      CHECK(edge_chroma->partial);
    }

    // Completely outside the visible plane.
    const auto outside =
        f::block_rect(*geom_420, 0, 640, 360, f::BlockSize::k4x4);
    CHECK(outside.has_value());
    if (outside) {
      CHECK(!outside->inside);
      CHECK_EQ_INT(outside->visible_width, 0);
      CHECK_EQ_INT(outside->visible_height, 0);
    }
  }

  const auto geom_422 =
      f::make_frame_geometry({640, 360, f::ChromaFormat::k422, f::BitDepth::k10});
  CHECK(geom_422.has_value());
  if (geom_422) {
    const auto chroma =
        f::block_rect(*geom_422, 1, 576, 320, f::BlockSize::k64x64);
    CHECK(chroma.has_value());
    if (chroma) {
      CHECK_EQ_INT(chroma->x, 288);
      CHECK_EQ_INT(chroma->y, 320);
      CHECK_EQ_INT(chroma->width, 32);
      CHECK_EQ_INT(chroma->height, 64);
      CHECK_EQ_INT(chroma->visible_height, 40);
    }
    // 4x8 luma has no legal 4:2:2 chroma residual size.
    CHECK(!f::block_rect(*geom_422, 1, 0, 0, f::BlockSize::k4x8).has_value());
  }

  const auto geom_444 =
      f::make_frame_geometry({640, 360, f::ChromaFormat::k444, f::BitDepth::k10});
  CHECK(geom_444.has_value());
  if (geom_444) {
    const auto chroma =
        f::block_rect(*geom_444, 1, 576, 320, f::BlockSize::k64x64);
    CHECK(chroma.has_value());
    if (chroma) {
      CHECK_EQ_INT(chroma->x, 576);
      CHECK_EQ_INT(chroma->y, 320);
      CHECK_EQ_INT(chroma->width, 64);
      CHECK_EQ_INT(chroma->height, 64);
      CHECK_EQ_INT(chroma->visible_width, 64);
      CHECK_EQ_INT(chroma->visible_height, 40);
      CHECK(chroma->partial);
    }
  }

  const auto geom_400 =
      f::make_frame_geometry({64, 64, f::ChromaFormat::k400, f::BitDepth::k10});
  CHECK(geom_400.has_value());
  if (geom_400) {
    CHECK(!f::block_rect(*geom_400, 1, 0, 0, f::BlockSize::k8x8).has_value());
  }

  CHECK_EQ_INT(f::visible_extent(0, 64, 640), 64);
  CHECK_EQ_INT(f::visible_extent(600, 64, 640), 40);
  CHECK_EQ_INT(f::visible_extent(640, 64, 640), 0);
}

// ---------------------------------------------------------------------------
// Requirements / checklist
// ---------------------------------------------------------------------------

void test_requirements() {
  const f::Format ten_422{640, 360, f::ChromaFormat::k422, f::BitDepth::k10};
  const f::FormatRequirements ten = f::format_requirements(ten_422);
  CHECK_EQ_INT(ten.chroma_format_idc, 3);
  CHECK_EQ_INT(ten.bit_depth_idc, 0);
  CHECK_EQ_INT(ten.profile_idc, 3);
  CHECK_EQ_INT(ten.max_q, 303);
  CHECK_EQ_INT(ten.base_q_idx_bits, 9);
  CHECK(ten.signalable);
  CHECK_EQ_INT(ten.dequant_clip_bits, 17);
  CHECK_EQ_INT(ten.inverse_row_clip_bits, 17);
  CHECK_EQ_INT(ten.inverse_col_clip_bits, 10);
  CHECK_EQ_INT(ten.dc_pred_bias, 512);
  CHECK_EQ_INT(ten.residual_clip_max, 1023);
  CHECK_EQ_INT(ten.cdf_q_ctx_count, 4);
  CHECK_EQ_INT(ten.cdf_q_ctx_threshold[0], 90);
  CHECK_EQ_INT(ten.cdf_q_ctx_threshold[1], 140);
  CHECK_EQ_INT(ten.cdf_q_ctx_threshold[2], 190);
  CHECK_EQ_INT(ten.cdef_coeff_shift, 2);
  CHECK_EQ_INT(ten.deblock_qindex_scale, 24);

  const f::Format eight_444{640, 360, f::ChromaFormat::k444, f::BitDepth::k8};
  const f::FormatRequirements eight = f::format_requirements(eight_444);
  CHECK_EQ_INT(eight.chroma_format_idc, 2);
  CHECK_EQ_INT(eight.bit_depth_idc, 1);
  CHECK_EQ_INT(eight.profile_idc, 4);
  CHECK_EQ_INT(eight.max_q, 255);
  CHECK_EQ_INT(eight.base_q_idx_bits, 8);
  CHECK_EQ_INT(eight.dequant_clip_bits, 15);
  CHECK_EQ_INT(eight.inverse_row_clip_bits, 15);
  CHECK_EQ_INT(eight.inverse_col_clip_bits, 8);
  CHECK_EQ_INT(eight.dc_pred_bias, 128);
  CHECK_EQ_INT(eight.cdef_coeff_shift, 0);

  const f::Format twelve_420{640, 360, f::ChromaFormat::k420,
                             f::BitDepth::k12};
  const f::FormatRequirements twelve = f::format_requirements(twelve_420);
  CHECK_EQ_INT(twelve.bit_depth_idc, -1);
  CHECK_EQ_INT(twelve.profile_idc, -1);
  CHECK_EQ_INT(twelve.max_q, 351);
  CHECK_EQ_INT(twelve.dequant_clip_bits, 19);
  CHECK_EQ_INT(twelve.dc_pred_bias, 2048);
  CHECK_EQ_INT(twelve.cdef_coeff_shift, 4);
  CHECK(!twelve.signalable);

  // §8.3.1 thresholds, on the raw base_q_idx.
  CHECK_EQ_INT(f::cdf_q_context(0), 0);
  CHECK_EQ_INT(f::cdf_q_context(90), 0);
  CHECK_EQ_INT(f::cdf_q_context(91), 1);
  CHECK_EQ_INT(f::cdf_q_context(140), 1);
  CHECK_EQ_INT(f::cdf_q_context(141), 2);
  CHECK_EQ_INT(f::cdf_q_context(190), 2);
  CHECK_EQ_INT(f::cdf_q_context(191), 3);
  CHECK_EQ_INT(f::cdf_q_context(303), 3);

  // Checklist carries the values an integrator must act on.
  const auto checklist = f::format_checklist(ten_422);
  CHECK(checklist.size() >= 12);
  const f::ChecklistItem* profile = find_item(checklist, "seq_profile_idc");
  CHECK(profile != nullptr);
  if (profile) {
    CHECK(profile->requirement.find("= 3") != std::string::npos);
  }
  const f::ChecklistItem* chroma = find_item(checklist, "chroma_format_idc");
  CHECK(chroma != nullptr);
  if (chroma) {
    CHECK(chroma->requirement.find("= 3") != std::string::npos);
  }
  const f::ChecklistItem* depth = find_item(checklist, "bit_depth_idc");
  CHECK(depth != nullptr);
  if (depth) {
    CHECK(depth->requirement.find("= 0") != std::string::npos);
  }
  const f::ChecklistItem* dequant = find_item(checklist, "dequant_clip");
  CHECK(dequant != nullptr);
  if (dequant) {
    CHECK(dequant->requirement.find("17") != std::string::npos);
  }
  const f::ChecklistItem* signalable = find_item(checklist, "signalable");
  CHECK(signalable != nullptr);
  if (signalable) {
    CHECK(signalable->requirement.find("yes") != std::string::npos);
  }

  const auto twelve_checklist = f::format_checklist(twelve_420);
  const f::ChecklistItem* reserved = find_item(twelve_checklist, "bit_depth_idc");
  CHECK(reserved != nullptr);
  if (reserved) {
    CHECK(reserved->requirement.find("reserved") != std::string::npos);
  }
  const f::ChecklistItem* not_signalable =
      find_item(twelve_checklist, "signalable");
  CHECK(not_signalable != nullptr);
  if (not_signalable) {
    CHECK(not_signalable->requirement.find("no") != std::string::npos);
  }
}

}  // namespace

int main() {
  test_depth_helpers();
  test_chroma_mapping();
  test_block_tables();
  test_geometry();
  test_format_validation();
  test_range_and_buffers();
  test_conversion_round_trips();
  test_conversion_errors();
  test_block_rects();
  test_requirements();

  if (g_failures != 0) {
    std::cerr << "format_test: " << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "format_test: PASS (" << g_checks << " checks)\n";
  return 0;
}
