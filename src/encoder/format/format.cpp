// Implementation of the standalone AV2 format / high-bit-depth layer.
//
// See format.h for the normative clause map and the 12-bit caveat.  Nothing in
// this file writes bits or touches entropy/header code; it only manipulates
// planar samples and geometry.

#include "format.h"

#include "format_tables.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::fmt {
namespace {

static_assert(kBlockSizeCount == 29, "AV2 has 29 block sizes (§6.19.3.1)");
static_assert(static_cast<int>(BlockSize::k64x8) + 1 == kBlockSizeCount,
              "BlockSize enum must match the generated table order");

bool valid_depth(BitDepth depth) {
  switch (depth) {
    case BitDepth::k8:
    case BitDepth::k10:
    case BitDepth::k12:
      return true;
  }
  return false;
}

bool valid_chroma(ChromaFormat chroma) {
  switch (chroma) {
    case ChromaFormat::k400:
    case ChromaFormat::k420:
    case ChromaFormat::k422:
    case ChromaFormat::k444:
      return true;
  }
  return false;
}

std::uint32_t clip_visible(std::uint32_t origin, std::uint32_t extent,
                           std::uint32_t limit) {
  if (origin >= limit) {
    return 0;
  }
  return std::min(extent, limit - origin);
}

}  // namespace

bool operator==(const Format& a, const Format& b) {
  return a.width == b.width && a.height == b.height && a.chroma == b.chroma &&
         a.depth == b.depth;
}

bool operator!=(const Format& a, const Format& b) { return !(a == b); }

// ---------------------------------------------------------------------------
// Bit depth
// ---------------------------------------------------------------------------

int bit_depth_value(BitDepth depth) {
  switch (depth) {
    case BitDepth::k8:
      return 8;
    case BitDepth::k10:
      return 10;
    case BitDepth::k12:
      return 12;
  }
  return 0;
}

const char* bit_depth_name(BitDepth depth) {
  switch (depth) {
    case BitDepth::k8:
      return "8-bit";
    case BitDepth::k10:
      return "10-bit";
    case BitDepth::k12:
      return "12-bit";
  }
  return "invalid";
}

int sample_max(BitDepth depth) {
  const int bits = bit_depth_value(depth);
  return bits == 0 ? 0 : (1 << bits) - 1;
}

int bit_depth_idc(BitDepth depth) {
  switch (depth) {
    case BitDepth::k8:
      return 1;
    case BitDepth::k10:
      return 0;
    case BitDepth::k12:
      // §6.4.1: bit_depth_idc > 1 is reserved for AOMedia.
      return -1;
  }
  return -1;
}

std::optional<BitDepth> bit_depth_from_idc(int idc) {
  switch (idc) {
    case 0:
      return BitDepth::k10;
    case 1:
      return BitDepth::k8;
    default:
      return std::nullopt;
  }
}

int max_quantizer(BitDepth depth) {
  const int bits = bit_depth_value(depth);
  if (bits == 0) {
    return 0;
  }
  // MAXQ_8_BITS + 2 * MAXQ_OFFSET per extra bit-depth step, where
  // MAXQ_OFFSET = 24.  8 -> 255, 10 -> 303, 12 -> 351 (MAXQ_BITS).
  return 255 + 24 * (bits - 8);
}

int base_q_idx_bits(BitDepth depth) {
  // §5.18.6.1: n = BitDepth == 8 ? 8 : 9.
  return bit_depth_value(depth) == 8 ? 8 : 9;
}

// ---------------------------------------------------------------------------
// Chroma formats
// ---------------------------------------------------------------------------

int chroma_format_idc(ChromaFormat chroma) {
  switch (chroma) {
    case ChromaFormat::k420:
      return 0;  // CHROMA_FORMAT_420
    case ChromaFormat::k400:
      return 1;  // CHROMA_FORMAT_400
    case ChromaFormat::k444:
      return 2;  // CHROMA_FORMAT_444
    case ChromaFormat::k422:
      return 3;  // CHROMA_FORMAT_422
  }
  return -1;
}

std::optional<ChromaFormat> chroma_format_from_idc(int idc) {
  switch (idc) {
    case 0:
      return ChromaFormat::k420;
    case 1:
      return ChromaFormat::k400;
    case 2:
      return ChromaFormat::k444;
    case 3:
      return ChromaFormat::k422;
    default:
      return std::nullopt;
  }
}

int subsampling_x(ChromaFormat chroma) {
  switch (chroma) {
    case ChromaFormat::k420:
    case ChromaFormat::k422:
      return 1;
    case ChromaFormat::k444:
      return 0;
    case ChromaFormat::k400:
      // §6.4.1 table uses SubsamplingX = 1 for 4:0:0 even though there is no
      // chroma plane.  Keep the spec values; num_planes() reports 1.
      return 1;
  }
  return 0;
}

int subsampling_y(ChromaFormat chroma) {
  switch (chroma) {
    case ChromaFormat::k420:
      return 1;
    case ChromaFormat::k422:
    case ChromaFormat::k444:
      return 0;
    case ChromaFormat::k400:
      return 1;
  }
  return 0;
}

int num_planes(ChromaFormat chroma) { return chroma == ChromaFormat::k400 ? 1 : 3; }

bool has_chroma(ChromaFormat chroma) { return chroma != ChromaFormat::k400; }

const char* chroma_format_name(ChromaFormat chroma) {
  switch (chroma) {
    case ChromaFormat::k400:
      return "4:0:0";
    case ChromaFormat::k420:
      return "4:2:0";
    case ChromaFormat::k422:
      return "4:2:2";
    case ChromaFormat::k444:
      return "4:4:4";
  }
  return "invalid";
}

// ---------------------------------------------------------------------------
// Blocks and chroma residual sizes
// ---------------------------------------------------------------------------

int block_width(BlockSize size) {
  const auto index = static_cast<std::size_t>(size);
  return index < static_cast<std::size_t>(kBlockSizeCount) ? kBlockWidth[index] : 0;
}

int block_height(BlockSize size) {
  const auto index = static_cast<std::size_t>(size);
  return index < static_cast<std::size_t>(kBlockSizeCount) ? kBlockHeight[index] : 0;
}

const char* block_size_name(BlockSize size) {
  const auto index = static_cast<std::size_t>(size);
  return index < static_cast<std::size_t>(kBlockSizeCount) ? kBlockSizeNames[index] : "invalid";
}

std::optional<BlockSize> find_block_size(int width, int height) {
  for (int i = 0; i < kBlockSizeCount; ++i) {
    if (kBlockWidth[static_cast<std::size_t>(i)] == width &&
        kBlockHeight[static_cast<std::size_t>(i)] == height) {
      return static_cast<BlockSize>(i);
    }
  }
  return std::nullopt;
}

std::optional<BlockSize> chroma_residual_size(BlockSize luma,
                                              ChromaFormat chroma) {
  if (!valid_chroma(chroma) || !has_chroma(chroma)) {
    return std::nullopt;
  }
  const auto index = static_cast<std::size_t>(luma);
  if (index >= static_cast<std::size_t>(kBlockSizeCount)) {
    return std::nullopt;
  }
  const int sub_x = subsampling_x(chroma);
  const int sub_y = subsampling_y(chroma);
  const int value = kSubsampledSize[index][static_cast<std::size_t>(sub_x)]
                                   [static_cast<std::size_t>(sub_y)];
  if (value == kInvalidBlockIndex) {
    return std::nullopt;
  }
  return static_cast<BlockSize>(value);
}

std::optional<BlockSize> chroma_residual_size(int luma_width, int luma_height,
                                              ChromaFormat chroma) {
  const auto luma = find_block_size(luma_width, luma_height);
  if (!luma) {
    return std::nullopt;
  }
  return chroma_residual_size(*luma, chroma);
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

std::uint32_t mi_cols(std::uint32_t width) {
  // §5.18.4.4: MiCols = 2 * ((FrameWidth + 7) >> 3).
  return 2U * ((width + 7U) >> 3);
}

std::uint32_t mi_rows(std::uint32_t height) {
  return 2U * ((height + 7U) >> 3);
}

std::uint32_t coded_luma_width(std::uint32_t width) { return mi_cols(width) * 4U; }

std::uint32_t coded_luma_height(std::uint32_t height) {
  return mi_rows(height) * 4U;
}

bool is_internally_supported(const Format& format) {
  if (!valid_depth(format.depth) || !valid_chroma(format.chroma)) {
    return false;
  }
  if (format.width == 0 || format.height == 0) {
    return false;
  }
  if (format.width > kMaxDimension || format.height > kMaxDimension) {
    return false;
  }
  return true;
}

FormatCheck validate_format(const Format& format) {
  if (!valid_depth(format.depth)) {
    return {FormatStatus::kInvalidBitDepth,
            "bit depth is not 8, 10 or 12"};
  }
  if (!valid_chroma(format.chroma)) {
    return {FormatStatus::kInvalidChromaFormat,
            "chroma format is not 4:0:0, 4:2:0, 4:2:2 or 4:4:4"};
  }
  if (format.width == 0 || format.height == 0) {
    return {FormatStatus::kZeroDimension,
            "width and height must be non-zero"};
  }
  if (format.width > kMaxDimension || format.height > kMaxDimension) {
    return {FormatStatus::kDimensionTooLarge,
            "width or height exceeds the " + std::to_string(kMaxDimension) +
                " implementation limit"};
  }
  if (format.depth == BitDepth::k12) {
    return {FormatStatus::kReservedBitDepth,
            "AV2 v1.0.0 bit_depth_idc > 1 is reserved; 12-bit samples are "
            "internal-only and cannot be signalled"};
  }
  return {FormatStatus::kOk, {}};
}

int profile_idc(const Format& format) {
  if (!valid_depth(format.depth) || !valid_chroma(format.chroma) ||
      format.depth == BitDepth::k12) {
    return -1;
  }
  switch (format.chroma) {
    case ChromaFormat::k400:
    case ChromaFormat::k420:
      return 0;  // Main_420_10_IP0
    case ChromaFormat::k422:
      return 3;  // Main_422_10_IP1
    case ChromaFormat::k444:
      return 4;  // Main_444_10_IP1
  }
  return -1;
}

bool is_signalable(const Format& format) {
  return validate_format(format).ok() && bit_depth_idc(format.depth) >= 0 &&
         profile_idc(format) >= 0;
}

std::optional<FrameGeometry> make_frame_geometry(const Format& format) {
  if (!is_internally_supported(format)) {
    return std::nullopt;
  }

  FrameGeometry geometry;
  geometry.format = format;
  geometry.mi_cols = mi_cols(format.width);
  geometry.mi_rows = mi_rows(format.height);
  geometry.coded_width = geometry.mi_cols * 4U;
  geometry.coded_height = geometry.mi_rows * 4U;
  geometry.num_planes = num_planes(format.chroma);
  geometry.sub_x = subsampling_x(format.chroma);
  geometry.sub_y = subsampling_y(format.chroma);

  std::size_t offset = 0;
  for (int plane = 0; plane < geometry.num_planes; ++plane) {
    PlaneGeometry& pg = geometry.planes[plane];
    pg.coded_width = geometry.coded_width >> (plane == 0 ? 0 : geometry.sub_x);
    pg.coded_height = geometry.coded_height >> (plane == 0 ? 0 : geometry.sub_y);
    pg.visible_width = plane == 0
                           ? format.width
                           : (format.width +
                              ((1U << geometry.sub_x) - 1U)) >>
                                 geometry.sub_x;
    pg.visible_height = plane == 0
                            ? format.height
                            : (format.height +
                               ((1U << geometry.sub_y) - 1U)) >>
                                  geometry.sub_y;
    pg.stride = pg.coded_width;
    pg.offset = offset;

    const std::size_t width = static_cast<std::size_t>(pg.coded_width);
    const std::size_t height = static_cast<std::size_t>(pg.coded_height);
    if (width != 0 && height > std::numeric_limits<std::size_t>::max() / width) {
      return std::nullopt;
    }
    pg.sample_count = width * height;
    if (offset > std::numeric_limits<std::size_t>::max() - pg.sample_count) {
      return std::nullopt;
    }
    offset += pg.sample_count;
  }

  geometry.total_samples = offset;
  if (offset > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
    return std::nullopt;
  }
  geometry.frame_bytes = offset * sizeof(std::uint16_t);
  return geometry;
}

// ---------------------------------------------------------------------------
// Sample buffers
// ---------------------------------------------------------------------------

std::uint16_t& PlaneBuffer::at(std::uint32_t x, std::uint32_t y) {
  return samples[static_cast<std::size_t>(y) * stride + x];
}

const std::uint16_t& PlaneBuffer::at(std::uint32_t x, std::uint32_t y) const {
  return samples[static_cast<std::size_t>(y) * stride + x];
}

bool allocate_picture(const Format& format, Picture& picture,
                      std::string& error) {
  const auto geometry = make_frame_geometry(format);
  if (!geometry) {
    error = "unsupported format: " + std::to_string(format.width) + "x" +
            std::to_string(format.height) + " " +
            chroma_format_name(format.chroma) + " " +
            bit_depth_name(format.depth);
    return false;
  }

  Picture allocated;
  allocated.format = format;
  allocated.geometry = *geometry;

  try {
    for (int plane = 0; plane < geometry->num_planes; ++plane) {
      const PlaneGeometry& pg = geometry->planes[plane];
      PlaneBuffer& buffer = allocated.planes[plane];
      buffer.width = pg.coded_width;
      buffer.height = pg.coded_height;
      buffer.visible_width = pg.visible_width;
      buffer.visible_height = pg.visible_height;
      buffer.stride = pg.stride;
      buffer.samples.assign(pg.sample_count, 0);
    }
  } catch (const std::bad_alloc&) {
    error = "out of memory allocating " +
            std::to_string(geometry->total_samples) + " samples";
    return false;
  }

  picture = std::move(allocated);
  error.clear();
  return true;
}

std::uint16_t sample_clamped(const PlaneBuffer& plane, int x, int y) {
  const std::uint32_t visible_width =
      std::min(plane.visible_width, plane.width);
  const std::uint32_t visible_height =
      std::min(plane.visible_height, plane.height);
  if (visible_width == 0 || visible_height == 0 || plane.samples.empty()) {
    return 0;
  }
  const int max_x = static_cast<int>(visible_width) - 1;
  const int max_y = static_cast<int>(visible_height) - 1;
  const int cx = std::clamp(x, 0, max_x);
  const int cy = std::clamp(y, 0, max_y);
  return plane.at(static_cast<std::uint32_t>(cx),
                  static_cast<std::uint32_t>(cy));
}

void replicate_borders(PlaneBuffer& plane) {
  const std::uint32_t visible_width =
      std::min(plane.visible_width, plane.width);
  const std::uint32_t visible_height =
      std::min(plane.visible_height, plane.height);
  if (visible_width == 0 || visible_height == 0 || plane.samples.empty()) {
    return;
  }

  // First copy the last visible column into the right padding.
  for (std::uint32_t y = 0; y < visible_height; ++y) {
    const std::uint16_t value = plane.at(visible_width - 1U, y);
    for (std::uint32_t x = visible_width; x < plane.width; ++x) {
      plane.at(x, y) = value;
    }
  }
  // Then copy the last visible row (which already has padded columns) down.
  for (std::uint32_t x = 0; x < plane.width; ++x) {
    const std::uint16_t value = plane.at(x, visible_height - 1U);
    for (std::uint32_t y = visible_height; y < plane.height; ++y) {
      plane.at(x, y) = value;
    }
  }
}

// ---------------------------------------------------------------------------
// Range and clipping
// ---------------------------------------------------------------------------

std::uint16_t clip_sample(std::int64_t value, BitDepth depth) {
  const std::int64_t high = sample_max(depth);
  if (value < 0) {
    return 0;
  }
  if (value > high) {
    return static_cast<std::uint16_t>(high);
  }
  return static_cast<std::uint16_t>(value);
}

std::uint16_t expand_8_to_depth(std::uint8_t value, BitDepth depth) {
  const int shift = bit_depth_value(depth) - 8;
  if (shift <= 0) {
    return value;
  }
  return static_cast<std::uint16_t>(static_cast<std::uint32_t>(value)
                                    << static_cast<unsigned>(shift));
}

std::uint8_t reduce_to_8(std::uint16_t value, BitDepth depth) {
  const int bits = bit_depth_value(depth);
  if (bits <= 8) {
    return static_cast<std::uint8_t>(std::min<std::uint32_t>(value, 255U));
  }
  const int shift = bits - 8;
  const std::uint32_t bias = 1U << static_cast<unsigned>(shift - 1);
  const std::uint32_t rounded =
      (static_cast<std::uint32_t>(value) + bias) >> static_cast<unsigned>(shift);
  return static_cast<std::uint8_t>(std::min<std::uint32_t>(rounded, 255U));
}

std::uint16_t convert_sample_depth(std::uint16_t value, BitDepth from,
                                   BitDepth to) {
  const int from_bits = bit_depth_value(from);
  const int to_bits = bit_depth_value(to);
  if (from_bits == 0 || to_bits == 0) {
    return 0;
  }
  std::int64_t result = value;
  if (to_bits >= from_bits) {
    result <<= static_cast<unsigned>(to_bits - from_bits);
  } else {
    const int shift = from_bits - to_bits;
    const std::int64_t bias = std::int64_t{1} << static_cast<unsigned>(shift - 1);
    result = (result + bias) >> static_cast<unsigned>(shift);
  }
  return clip_sample(result, to);
}

RangeReport check_sample_range(const Picture& picture) {
  RangeReport report;
  report.num_planes = picture.geometry.num_planes;
  const std::uint32_t high =
      static_cast<std::uint32_t>(sample_max(picture.format.depth));

  for (int plane = 0; plane < report.num_planes; ++plane) {
    const PlaneBuffer& buffer = picture.planes[plane];
    if (buffer.samples.empty()) {
      continue;
    }
    PlaneRange& range = report.planes[plane];
    range.min_value = std::numeric_limits<std::uint32_t>::max();
    range.max_value = 0;
    for (const std::uint16_t sample : buffer.samples) {
      const std::uint32_t value = sample;
      range.min_value = std::min(range.min_value, value);
      range.max_value = std::max(range.max_value, value);
      if (value > high) {
        ++range.out_of_range;
      }
    }
    report.total_out_of_range += range.out_of_range;
  }
  return report;
}

std::size_t clamp_samples(Picture& picture) {
  std::size_t changed = 0;
  for (int plane = 0; plane < picture.geometry.num_planes; ++plane) {
    PlaneBuffer& buffer = picture.planes[plane];
    for (std::uint16_t& sample : buffer.samples) {
      const std::uint16_t clipped = clip_sample(sample, picture.format.depth);
      if (clipped != sample) {
        sample = clipped;
        ++changed;
      }
    }
  }
  return changed;
}

bool samples_in_range(const Picture& picture) {
  return check_sample_range(picture).total_out_of_range == 0;
}

// ---------------------------------------------------------------------------
// Conversion from / to the M0 8-bit 4:2:0 input
// ---------------------------------------------------------------------------

namespace {

bool valid_source_plane(const ConstPlane8& plane, std::uint32_t width) {
  return plane.data != nullptr && plane.stride >= width;
}

}  // namespace

ConvertResult convert_420_8_to_picture(const Source420_8& source,
                                       ChromaFormat chroma, BitDepth depth,
                                       Picture& picture) {
  if (source.width == 0 || source.height == 0) {
    return {ConvertStatus::kBadSource, "source width and height must be non-zero"};
  }
  if (source.width > kMaxDimension || source.height > kMaxDimension) {
    return {ConvertStatus::kBadSource, "source dimensions exceed implementation limit"};
  }

  const std::uint32_t chroma_width = (source.width + 1U) / 2U;
  if (!valid_source_plane(source.y, source.width) ||
      !valid_source_plane(source.u, chroma_width) ||
      !valid_source_plane(source.v, chroma_width)) {
    return {ConvertStatus::kBadSource,
            "source plane is null or stride is smaller than the plane width"};
  }

  const Format format{source.width, source.height, chroma, depth};
  if (!is_internally_supported(format)) {
    return {ConvertStatus::kUnsupportedFormat,
            "unsupported destination format"};
  }

  std::string error;
  if (!allocate_picture(format, picture, error)) {
    return {ConvertStatus::kAllocationFailed, error};
  }

  PlaneBuffer& luma = picture.planes[0];
  for (std::uint32_t y = 0; y < luma.visible_height; ++y) {
    const std::uint8_t* row =
        source.y.data + static_cast<std::size_t>(y) * source.y.stride;
    for (std::uint32_t x = 0; x < luma.visible_width; ++x) {
      luma.at(x, y) = expand_8_to_depth(row[x], depth);
    }
  }

  if (picture.geometry.num_planes > 1) {
    const int sub_x = picture.geometry.sub_x;
    const int sub_y = picture.geometry.sub_y;
    const std::uint32_t source_chroma_height = (source.height + 1U) / 2U;

    for (int plane = 1; plane < picture.geometry.num_planes; ++plane) {
      PlaneBuffer& dst = picture.planes[plane];
      const ConstPlane8& src_plane = plane == 1 ? source.u : source.v;
      for (std::uint32_t y = 0; y < dst.visible_height; ++y) {
        for (std::uint32_t x = 0; x < dst.visible_width; ++x) {
          // Destination chroma sample (x, y) covers luma (x << sub_x,
          // y << sub_y); the source is 4:2:0, so source chroma is luma >> 1.
          const std::uint32_t source_x = (x << static_cast<unsigned>(sub_x)) >> 1;
          const std::uint32_t source_y = (y << static_cast<unsigned>(sub_y)) >> 1;
          const std::uint32_t clamped_x =
              std::min(source_x, chroma_width - 1U);
          const std::uint32_t clamped_y =
              std::min(source_y, source_chroma_height - 1U);
          const std::uint8_t value =
              src_plane.data[static_cast<std::size_t>(clamped_y) *
                                 src_plane.stride +
                             clamped_x];
          dst.at(x, y) = expand_8_to_depth(value, depth);
        }
      }
    }
  }

  for (int plane = 0; plane < picture.geometry.num_planes; ++plane) {
    replicate_borders(picture.planes[plane]);
  }
  return {ConvertStatus::kOk, {}};
}

ConvertResult convert_to_420_8(const Picture& picture, Picture420_8& out) {
  if (!is_internally_supported(picture.format)) {
    return {ConvertStatus::kUnsupportedFormat, "unsupported source format"};
  }
  if (!has_chroma(picture.format.chroma)) {
    return {ConvertStatus::kUnsupportedFormat,
            "cannot convert monochrome 4:0:0 to 4:2:0 without chroma"};
  }

  const std::uint32_t width = picture.format.width;
  const std::uint32_t height = picture.format.height;
  const std::uint32_t chroma_width = (width + 1U) / 2U;
  const std::uint32_t chroma_height = (height + 1U) / 2U;
  const int src_sub_x = picture.geometry.sub_x;
  const int src_sub_y = picture.geometry.sub_y;
  // Destination is 4:2:0.  A source 4:2:0 chroma plane needs no scaling, a
  // 4:2:2 plane doubles vertically, and 4:4:4 doubles both axes.
  const int scale_x = 1 - src_sub_x;
  const int scale_y = 1 - src_sub_y;

  Picture420_8 converted;
  converted.width = width;
  converted.height = height;

  try {
    converted.y.width = width;
    converted.y.height = height;
    converted.y.stride = width;
    converted.y.data.assign(static_cast<std::size_t>(width) * height, 0);

    converted.u.width = chroma_width;
    converted.u.height = chroma_height;
    converted.u.stride = chroma_width;
    converted.u.data.assign(
        static_cast<std::size_t>(chroma_width) * chroma_height, 0);

    converted.v.width = chroma_width;
    converted.v.height = chroma_height;
    converted.v.stride = chroma_width;
    converted.v.data.assign(
        static_cast<std::size_t>(chroma_width) * chroma_height, 0);
  } catch (const std::bad_alloc&) {
    return {ConvertStatus::kAllocationFailed, "out of memory"};
  }

  const PlaneBuffer& luma = picture.planes[0];
  for (std::uint32_t y = 0; y < height; ++y) {
    std::uint8_t* row =
        converted.y.data.data() + static_cast<std::size_t>(y) * converted.y.stride;
    for (std::uint32_t x = 0; x < width; ++x) {
      row[x] = reduce_to_8(luma.at(x, y), picture.format.depth);
    }
  }

  for (int plane = 1; plane < 3; ++plane) {
    Plane8& dst = plane == 1 ? converted.u : converted.v;
    const PlaneBuffer& src = picture.planes[plane];
    for (std::uint32_t y = 0; y < chroma_height; ++y) {
      std::uint8_t* row =
          dst.data.data() + static_cast<std::size_t>(y) * dst.stride;
      for (std::uint32_t x = 0; x < chroma_width; ++x) {
        const int source_x = static_cast<int>(x << static_cast<unsigned>(scale_x));
        const int source_y = static_cast<int>(y << static_cast<unsigned>(scale_y));
        row[x] = reduce_to_8(sample_clamped(src, source_x, source_y),
                             picture.format.depth);
      }
    }
  }

  out = std::move(converted);
  return {ConvertStatus::kOk, {}};
}

// ---------------------------------------------------------------------------
// Block rectangles and edge rules
// ---------------------------------------------------------------------------

std::uint32_t visible_extent(std::uint32_t origin, std::uint32_t extent,
                             std::uint32_t visible_limit) {
  return clip_visible(origin, extent, visible_limit);
}

std::optional<BlockRect> block_rect(const FrameGeometry& geometry, int plane,
                                    std::uint32_t luma_x, std::uint32_t luma_y,
                                    BlockSize luma_size) {
  if (plane < 0 || plane >= geometry.num_planes) {
    return std::nullopt;
  }

  const PlaneGeometry* plane_geometry = nullptr;
  std::uint32_t x = 0;
  std::uint32_t y = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;

  if (plane == 0) {
    plane_geometry = &geometry.planes[0];
    x = luma_x;
    y = luma_y;
    width = static_cast<std::uint32_t>(block_width(luma_size));
    height = static_cast<std::uint32_t>(block_height(luma_size));
  } else {
    const auto chroma_size = chroma_residual_size(luma_size, geometry.format.chroma);
    if (!chroma_size) {
      return std::nullopt;
    }
    plane_geometry = &geometry.planes[plane];
    x = luma_x >> static_cast<unsigned>(geometry.sub_x);
    y = luma_y >> static_cast<unsigned>(geometry.sub_y);
    width = static_cast<std::uint32_t>(block_width(*chroma_size));
    height = static_cast<std::uint32_t>(block_height(*chroma_size));
  }

  if (width == 0 || height == 0) {
    return std::nullopt;
  }

  BlockRect rect;
  rect.x = x;
  rect.y = y;
  rect.width = width;
  rect.height = height;
  rect.visible_width =
      visible_extent(x, width, plane_geometry->visible_width);
  rect.visible_height =
      visible_extent(y, height, plane_geometry->visible_height);
  rect.inside = rect.visible_width > 0 && rect.visible_height > 0;
  rect.partial = rect.visible_width != width || rect.visible_height != height;
  return rect;
}

// ---------------------------------------------------------------------------
// Header / transform / entropy / filter requirements
// ---------------------------------------------------------------------------

FormatRequirements format_requirements(const Format& format) {
  FormatRequirements requirements;
  const int depth = bit_depth_value(format.depth);
  requirements.chroma_format_idc = chroma_format_idc(format.chroma);
  requirements.bit_depth_idc = bit_depth_idc(format.depth);
  requirements.profile_idc = profile_idc(format);
  requirements.max_q = max_quantizer(format.depth);
  requirements.base_q_idx_bits = base_q_idx_bits(format.depth);
  requirements.signalable = is_signalable(format);

  requirements.dequant_clip_bits = 7 + depth;
  requirements.inverse_row_clip_bits = depth + 7;
  requirements.inverse_col_clip_bits = depth;
  requirements.dc_pred_bias = depth > 0 ? (1 << (depth - 1)) : 0;
  requirements.residual_clip_max = sample_max(format.depth);

  requirements.cdf_q_ctx_threshold[0] = 90;
  requirements.cdf_q_ctx_threshold[1] = 140;
  requirements.cdf_q_ctx_threshold[2] = 190;
  requirements.cdf_q_ctx_count = 4;

  requirements.cdef_coeff_shift = depth > 8 ? depth - 8 : 0;
  requirements.deblock_qindex_scale = 24;
  return requirements;
}

BitDepthLimits bit_depth_limits(BitDepth depth) {
  BitDepthLimits limits;
  const int bits = bit_depth_value(depth);
  if (bits <= 0) {
    return limits;
  }
  limits.bit_depth = bits;
  limits.sample_max = (1 << bits) - 1;
  limits.dc_bias = 1 << (bits - 1);
  limits.dequant_min = -(1 << (7 + bits));
  limits.dequant_max = (1 << (7 + bits)) - 1;
  limits.inverse_row_min = -(1 << (bits + 7));
  limits.inverse_row_max = (1 << (bits + 7)) - 1;
  limits.inverse_col_min = -(1 << bits);
  limits.inverse_col_max = (1 << bits) - 1;
  limits.residual_min = -(1 << bits);
  limits.residual_max = (1 << bits) - 1;
  return limits;
}

namespace {

int clip_int64(std::int64_t value, std::int64_t low, std::int64_t high) {
  if (value < low) {
    return static_cast<int>(low);
  }
  if (value > high) {
    return static_cast<int>(high);
  }
  return static_cast<int>(value);
}

}  // namespace

int clip1_sample(std::int64_t value, BitDepth depth) {
  const BitDepthLimits limits = bit_depth_limits(depth);
  return clip_int64(value, 0, limits.sample_max);
}

int clip_dequant(std::int64_t value, BitDepth depth) {
  const BitDepthLimits limits = bit_depth_limits(depth);
  return clip_int64(value, limits.dequant_min, limits.dequant_max);
}

int clip_inverse_row(std::int64_t value, BitDepth depth) {
  const BitDepthLimits limits = bit_depth_limits(depth);
  return clip_int64(value, limits.inverse_row_min, limits.inverse_row_max);
}

int clip_inverse_col(std::int64_t value, BitDepth depth) {
  const BitDepthLimits limits = bit_depth_limits(depth);
  return clip_int64(value, limits.inverse_col_min, limits.inverse_col_max);
}

int cdf_q_context(int base_q_idx) {
  if (base_q_idx <= 90) {
    return 0;
  }
  if (base_q_idx <= 140) {
    return 1;
  }
  if (base_q_idx <= 190) {
    return 2;
  }
  return 3;
}

std::vector<ChecklistItem> format_checklist(const Format& format) {
  std::vector<ChecklistItem> items;
  const FormatRequirements req = format_requirements(format);

  auto add = [&items](ChecklistArea area, std::string id,
                      std::string section, std::string requirement) {
    items.push_back({area, std::move(id), std::move(section),
                     std::move(requirement)});
  };

  // Header --------------------------------------------------------------
  if (req.profile_idc >= 0) {
    add(ChecklistArea::kHeader, "seq_profile_idc", "§5.4.1, Annex A.2",
        "seq_profile_idc = " + std::to_string(req.profile_idc) + " (" +
            chroma_format_name(format.chroma) + ", " +
            bit_depth_name(format.depth) + ")");
  } else {
    add(ChecklistArea::kHeader, "seq_profile_idc", "§5.4.1, Annex A.2",
        "no named profile; AV2 v1.0.0 defines no 12-bit profile");
  }

  add(ChecklistArea::kHeader, "chroma_format_idc", "§5.4.1, §6.4.1 Table 6.2",
      "chroma_format_idc = " + std::to_string(req.chroma_format_idc) + " (" +
          chroma_format_name(format.chroma) + "); note 4:4:4 = 2, 4:2:2 = 3");

  if (req.bit_depth_idc >= 0) {
    add(ChecklistArea::kHeader, "bit_depth_idc", "§5.4.1, §6.4.1 Table 6.3",
        "bit_depth_idc = " + std::to_string(req.bit_depth_idc) + " (" +
            bit_depth_name(format.depth) + "; 0 = 10-bit, 1 = 8-bit)");
  } else {
    add(ChecklistArea::kHeader, "bit_depth_idc", "§5.4.1, §6.4.1 Table 6.3",
        "reserved: bit_depth_idc > 1 is not a conforming AV2 v1.0.0 stream; "
        "12-bit support is internal/experimental only");
  }

  add(ChecklistArea::kHeader, "MaxQ", "§6.4.1 Table 6.3",
      "MaxQ = " + std::to_string(req.max_q));
  add(ChecklistArea::kHeader, "base_q_idx", "§5.18.6.1",
      "base_q_idx coded with f(" + std::to_string(req.base_q_idx_bits) + ")");
  add(ChecklistArea::kHeader, "signalable", "Annex A.2",
      req.signalable ? "yes: format can be signalled by a named profile"
                     : "no: internal sample format only");

  // Transform / reconstruction -----------------------------------------
  add(ChecklistArea::kTransform, "dequant_clip", "§7.14.3",
      "clip dequantized coefficients to [-(1<<" +
          std::to_string(req.dequant_clip_bits) + "), (1<<" +
          std::to_string(req.dequant_clip_bits) + ")-1]");
  add(ChecklistArea::kTransform, "inverse_transform_clip", "§7.15.2.1",
      "row pass clips to ±(1<<" + std::to_string(req.inverse_row_clip_bits) +
          ")-1; column pass clips to ±(1<<" +
          std::to_string(req.inverse_col_clip_bits) + ")-1");
  add(ChecklistArea::kTransform, "reconstruction_clip", "§4.8, §7.14.3",
      "Clip1 to [0, " + std::to_string(req.residual_clip_max) + "]");
  add(ChecklistArea::kTransform, "dc_prediction_bias", "§7.13.2.10",
      "DC predictor neutral value is 1 << (BitDepth-1) = " +
          std::to_string(req.dc_pred_bias) +
          "; do not keep the 8-bit constant 128");
  add(ChecklistArea::kTransform, "chroma_residual_size", "§5.20.7.26",
      "use Subsampled_Size for chroma TX geometry; 4:2:2 halves width only, "
      "4:4:4 uses the luma block size");
  add(ChecklistArea::kTransform, "transform_shift", "§7.15.4",
      "Transform_Shift is bit-depth independent; only the clip bounds change");

  // Entropy -------------------------------------------------------------
  add(ChecklistArea::kEntropy, "coeff_cdf_q_context", "§8.3.1",
      "init_coeff_cdfs() selects context by raw base_q_idx: <=90 -> 0, "
      "<=140 -> 1, <=190 -> 2, else 3; no bit-depth normalization");
  add(ChecklistArea::kEntropy, "coeff_symbols", "§5.20.7.27",
      "coefficient symbol alphabet/CDF layout is bit-depth independent; "
      "10-bit levels grow by up to 4x and use the existing escape path");
  add(ChecklistArea::kEntropy, "base_q_idx_syntax", "§5.18.6.1",
      "quantization_params() writes f(9) base_q_idx for non-8-bit; update the "
      "M0 header writer, do not reuse its 8-bit path");
  add(ChecklistArea::kEntropy, "cdf_reset", "§7.5, §8.3.1",
      "CDF reset/adaptation is unchanged; reset per tile as today");

  // Filters -------------------------------------------------------------
  add(ChecklistArea::kFilter, "deblock_level", "§7.17",
      "subtract 24 * (BitDepth-8) from the q index before Side_Thresholds; "
      "chroma edges use subsampled coordinates");
  add(ChecklistArea::kFilter, "cdef_shift", "§7.18",
      "coeffShift = BitDepth-8 = " + std::to_string(req.cdef_coeff_shift) +
          "; direction search uses (sample >> (BitDepth-8)) - 128");
  add(ChecklistArea::kFilter, "ccso_lr", "§7.19, §7.20",
      "CCSO band offsets and restoration rounding are bit-depth scaled; "
      "LoopRestorationSize is per plane and constrained by 64 >> SubsamplingY");
  add(ChecklistArea::kFilter, "chroma_filter_planes", "§5.18.5.2, §7.17",
      format.chroma == ChromaFormat::k400
          ? "monochrome: no chroma filter planes"
          : "filter chroma planes at their subsampled origin; 4:2:2 has "
            "full-height chroma, 4:4:4 has full-resolution chroma");

  return items;
}

}  // namespace av2enc::fmt
