#include "formats/format_y4m.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace av2enc::fmt {
namespace {

bool read_line(std::ifstream& file, std::string* line) {
  line->clear();
  char c = 0;
  while (file.get(c)) {
    if (c == '\n') {
      return true;
    }
    if (c != '\r') {
      line->push_back(c);
    }
  }
  return false;
}

bool parse_uint(std::string_view text, std::uint32_t* value) {
  if (text.empty()) {
    return false;
  }
  std::uint64_t parsed = 0;
  for (const char c : text) {
    if (c < '0' || c > '9') {
      return false;
    }
    parsed = parsed * 10U + static_cast<std::uint64_t>(c - '0');
    if (parsed > 0xFFFFFFFFULL) {
      return false;
    }
  }
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}

struct Colorspace {
  ChromaFormat chroma = ChromaFormat::k420;
  BitDepth depth = BitDepth::k8;
};

bool parse_colorspace(std::string_view tag, Colorspace* out) {
  if (tag == "420" || tag == "420jpeg" || tag == "420mpeg2" ||
      tag == "420paldv") {
    out->chroma = ChromaFormat::k420;
    out->depth = BitDepth::k8;
    return true;
  }
  if (tag == "420p10") {
    out->chroma = ChromaFormat::k420;
    out->depth = BitDepth::k10;
    return true;
  }
  if (tag == "422") {
    out->chroma = ChromaFormat::k422;
    out->depth = BitDepth::k8;
    return true;
  }
  if (tag == "422p10") {
    out->chroma = ChromaFormat::k422;
    out->depth = BitDepth::k10;
    return true;
  }
  if (tag == "444") {
    out->chroma = ChromaFormat::k444;
    out->depth = BitDepth::k8;
    return true;
  }
  if (tag == "444p10") {
    out->chroma = ChromaFormat::k444;
    out->depth = BitDepth::k10;
    return true;
  }
  if (tag == "mono" || tag == "monochrome" || tag == "400") {
    out->chroma = ChromaFormat::k400;
    out->depth = BitDepth::k8;
    return true;
  }
  if (tag == "mono10" || tag == "400p10" || tag == "monochrome10") {
    out->chroma = ChromaFormat::k400;
    out->depth = BitDepth::k10;
    return true;
  }
  return false;
}

int bytes_per_sample(BitDepth depth) {
  return depth == BitDepth::k8 ? 1 : 2;
}

std::uint32_t ceil_shift(std::uint32_t value, int shift) {
  if (shift == 0) {
    return value;
  }
  return (value + ((1U << shift) - 1U)) >> shift;
}

}  // namespace

std::string y4m_colorspace_tag(ChromaFormat chroma, BitDepth depth) {
  const char* base = "420";
  switch (chroma) {
    case ChromaFormat::k400:
      base = "mono";
      break;
    case ChromaFormat::k420:
      base = "420";
      break;
    case ChromaFormat::k422:
      base = "422";
      break;
    case ChromaFormat::k444:
      base = "444";
      break;
  }
  std::string tag = base;
  if (depth == BitDepth::k10) {
    // Y4M spells monochrome 10-bit "mono10"; other formats use "p10".
    tag += chroma == ChromaFormat::k400 ? "10" : "p10";
  }
  return "C" + tag;
}

bool read_y4m_sequence(const std::filesystem::path& path,
                       std::vector<Picture>* pictures, Y4mFormatInfo* info,
                       std::uint64_t max_frames, std::string* error) {
  if (error != nullptr) error->clear();
  if (pictures == nullptr) {
    if (error != nullptr) *error = "null picture list";
    return false;
  }
  pictures->clear();
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    if (error != nullptr) *error = "cannot open " + path.string();
    return false;
  }
  std::string header;
  if (!read_line(file, &header)) {
    if (error != nullptr) *error = "truncated Y4M header";
    return false;
  }
  std::istringstream tokens(header);
  std::string token;
  if (!(tokens >> token) || token != "YUV4MPEG2") {
    if (error != nullptr) *error = "not a Y4M file (expected YUV4MPEG2)";
    return false;
  }
  bool have_width = false;
  bool have_height = false;
  bool have_fps = false;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t fps_num = 1;
  std::uint32_t fps_den = 1;
  Colorspace colorspace;
  while (tokens >> token) {
    if (token.empty()) continue;
    const char key = token.front();
    const std::string_view rest(token.data() + 1, token.size() - 1);
    if (key == 'W') {
      if (!parse_uint(rest, &width)) {
        if (error != nullptr) *error = "invalid Y4M width";
        return false;
      }
      have_width = true;
    } else if (key == 'H') {
      if (!parse_uint(rest, &height)) {
        if (error != nullptr) *error = "invalid Y4M height";
        return false;
      }
      have_height = true;
    } else if (key == 'F') {
      const std::size_t colon = rest.find(':');
      if (colon == std::string_view::npos ||
          !parse_uint(rest.substr(0, colon), &fps_num) ||
          !parse_uint(rest.substr(colon + 1), &fps_den) || fps_num == 0 ||
          fps_den == 0) {
        if (error != nullptr) *error = "invalid Y4M frame rate";
        return false;
      }
      have_fps = true;
    } else if (key == 'C') {
      if (!parse_colorspace(rest, &colorspace)) {
        if (error != nullptr) {
          *error = "unsupported Y4M colorspace C" + std::string(rest);
        }
        return false;
      }
    }
  }
  if (!have_width || !have_height || width == 0 || height == 0) {
    if (error != nullptr) *error = "Y4M header is missing W/H";
    return false;
  }
  if (!have_fps) {
    if (error != nullptr) *error = "Y4M header is missing F";
    return false;
  }
  if (width > kMaxDimension || height > kMaxDimension) {
    if (error != nullptr) *error = "Y4M dimensions exceed the implementation limit";
    return false;
  }

  Format format;
  format.width = width;
  format.height = height;
  format.chroma = colorspace.chroma;
  format.depth = colorspace.depth;
  const int bps = bytes_per_sample(colorspace.depth);
  const int sx = subsampling_x(colorspace.chroma);
  const int sy = subsampling_y(colorspace.chroma);
  const std::uint32_t visible[3] = {width, ceil_shift(width, sx),
                                    ceil_shift(width, sx)};
  const std::uint32_t visible_h[3] = {height, ceil_shift(height, sy),
                                      ceil_shift(height, sy)};
  const int plane_count = num_planes(colorspace.chroma);
  std::size_t total_samples =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  for (int plane = 1; plane < plane_count; ++plane) {
    total_samples += static_cast<std::size_t>(visible[plane]) *
                     static_cast<std::size_t>(visible_h[plane]);
  }
  std::vector<std::uint8_t> raw(total_samples * static_cast<std::size_t>(bps));

  while (max_frames == 0 || pictures->size() < max_frames) {
    std::string frame_tag;
    if (!read_line(file, &frame_tag)) break;
    if (frame_tag.empty()) break;
    if (frame_tag.rfind("FRAME", 0) != 0) {
      if (error != nullptr) *error = "Y4M is missing the FRAME tag";
      return false;
    }
    file.read(reinterpret_cast<char*>(raw.data()),
              static_cast<std::streamsize>(raw.size()));
    if (static_cast<std::size_t>(file.gcount()) != raw.size()) {
      if (error != nullptr) *error = "truncated Y4M frame";
      return false;
    }
    Picture picture;
    if (!allocate_picture(format, picture, *error)) return false;
    std::size_t cursor = 0;
    for (int plane = 0; plane < plane_count; ++plane) {
      PlaneBuffer& dst = picture.planes[plane];
      for (std::uint32_t y = 0; y < visible_h[plane]; ++y) {
        for (std::uint32_t x = 0; x < visible[plane]; ++x) {
          std::uint16_t sample = 0;
          if (bps == 1) {
            sample = raw[cursor++];
          } else {
            sample = static_cast<std::uint16_t>(
                raw[cursor] |
                (static_cast<std::uint16_t>(raw[cursor + 1]) << 8));
            cursor += 2;
          }
          if (colorspace.depth == BitDepth::k10 && sample > 1023) {
            if (error != nullptr) *error = "10-bit Y4M sample exceeds 1023";
            return false;
          }
          dst.at(x, y) = sample;
        }
      }
    }
    for (int plane = 0; plane < plane_count; ++plane) {
      replicate_borders(picture.planes[plane]);
    }
    pictures->push_back(std::move(picture));
  }
  if (pictures->empty()) {
    if (error != nullptr) *error = "Y4M file has no frames";
    return false;
  }
  if (info != nullptr) {
    info->width = width;
    info->height = height;
    info->fps_num = fps_num;
    info->fps_den = fps_den;
    info->chroma = colorspace.chroma;
    info->depth = colorspace.depth;
    info->colorspace =
        y4m_colorspace_tag(colorspace.chroma, colorspace.depth).substr(1);
  }
  return true;
}

bool read_y4m_picture(const std::filesystem::path& path, Picture* picture,
                      Y4mFormatInfo* info, std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  if (picture == nullptr) {
    if (error != nullptr) {
      *error = "null picture";
    }
    return false;
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    if (error != nullptr) {
      *error = "cannot open " + path.string();
    }
    return false;
  }

  std::string header;
  if (!read_line(file, &header)) {
    if (error != nullptr) {
      *error = "truncated Y4M header";
    }
    return false;
  }
  std::istringstream tokens(header);
  std::string token;
  if (!(tokens >> token) || token != "YUV4MPEG2") {
    if (error != nullptr) {
      *error = "not a Y4M file (expected YUV4MPEG2)";
    }
    return false;
  }

  bool have_width = false;
  bool have_height = false;
  bool have_fps = false;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t fps_num = 1;
  std::uint32_t fps_den = 1;
  Colorspace colorspace;

  while (tokens >> token) {
    if (token.empty()) {
      continue;
    }
    const char key = token.front();
    const std::string_view rest(token.data() + 1, token.size() - 1);
    if (key == 'W') {
      if (!parse_uint(rest, &width)) {
        if (error != nullptr) {
          *error = "invalid Y4M width";
        }
        return false;
      }
      have_width = true;
    } else if (key == 'H') {
      if (!parse_uint(rest, &height)) {
        if (error != nullptr) {
          *error = "invalid Y4M height";
        }
        return false;
      }
      have_height = true;
    } else if (key == 'F') {
      const std::size_t colon = rest.find(':');
      if (colon == std::string_view::npos ||
          !parse_uint(rest.substr(0, colon), &fps_num) ||
          !parse_uint(rest.substr(colon + 1), &fps_den) || fps_num == 0 ||
          fps_den == 0) {
        if (error != nullptr) {
          *error = "invalid Y4M frame rate";
        }
        return false;
      }
      have_fps = true;
    } else if (key == 'C') {
      if (!parse_colorspace(rest, &colorspace)) {
        if (error != nullptr) {
          *error = "unsupported Y4M colorspace C" + std::string(rest) +
                   " (supported: C420, C420p10, C422, C422p10, C444, "
                   "C444p10, Cmono, Cmono10)";
        }
        return false;
      }
    }
  }

  if (!have_width || !have_height || width == 0 || height == 0) {
    if (error != nullptr) {
      *error = "Y4M header is missing W/H";
    }
    return false;
  }
  if (!have_fps) {
    if (error != nullptr) {
      *error = "Y4M header is missing F";
    }
    return false;
  }
  if (width > kMaxDimension || height > kMaxDimension) {
    if (error != nullptr) {
      *error = "Y4M dimensions exceed the implementation limit";
    }
    return false;
  }

  std::string frame_tag;
  if (!read_line(file, &frame_tag)) {
    if (error != nullptr) {
      *error = "Y4M file has no frame";
    }
    return false;
  }
  if (frame_tag.rfind("FRAME", 0) != 0) {
    if (error != nullptr) {
      *error = "Y4M is missing the FRAME tag";
    }
    return false;
  }

  Format format;
  format.width = width;
  format.height = height;
  format.chroma = colorspace.chroma;
  format.depth = colorspace.depth;
  if (!allocate_picture(format, *picture, *error)) {
    return false;
  }

  const int bps = bytes_per_sample(colorspace.depth);
  const int sx = subsampling_x(colorspace.chroma);
  const int sy = subsampling_y(colorspace.chroma);
  const std::uint32_t visible[3] = {
      width, ceil_shift(width, sx), ceil_shift(width, sx)};
  const std::uint32_t visible_h[3] = {
      height, ceil_shift(height, sy), ceil_shift(height, sy)};
  const int plane_count = num_planes(colorspace.chroma);

  std::size_t total_samples =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  for (int plane = 1; plane < plane_count; ++plane) {
    total_samples += static_cast<std::size_t>(visible[plane]) *
                     static_cast<std::size_t>(visible_h[plane]);
  }
  std::vector<std::uint8_t> raw(total_samples * static_cast<std::size_t>(bps));
  file.read(reinterpret_cast<char*>(raw.data()),
            static_cast<std::streamsize>(raw.size()));
  if (static_cast<std::size_t>(file.gcount()) != raw.size()) {
    if (error != nullptr) {
      *error = "truncated Y4M frame";
    }
    return false;
  }

  // A frame body is exactly `total_samples * bytes_per_sample` bytes, so the
  // file must end here.  Without this the reader silently accepted a body that
  // the W/H/C tag under-declares (e.g. a 4:4:4 might be tagged C420) and
  // encoded the prefix as the smaller format.  The format still paths read one
  // frame; a multi-frame input or appended bytes are a frame-size error.
  std::error_code size_error;
  const std::uintmax_t file_bytes = std::filesystem::file_size(path, size_error);
  const std::streampos body_end = file.tellg();
  if (!size_error && body_end >= 0 &&
      static_cast<std::uintmax_t>(body_end) != file_bytes) {
    const std::uintmax_t trailing =
        file_bytes > static_cast<std::uintmax_t>(body_end)
            ? file_bytes - static_cast<std::uintmax_t>(body_end)
            : 0;
    if (error != nullptr) {
      *error = "oversized Y4M frame: " + std::to_string(trailing) +
               " trailing byte(s) after the " + std::to_string(width) + "x" +
               std::to_string(height) + " " +
               y4m_colorspace_tag(colorspace.chroma, colorspace.depth) +
               " frame body (the format paths read exactly one frame; check "
               "for a mismatched C tag or a multi-frame input)";
    }
    return false;
  }

  std::size_t cursor = 0;
  for (int plane = 0; plane < plane_count; ++plane) {
    PlaneBuffer& dst = picture->planes[plane];
    for (std::uint32_t y = 0; y < visible_h[plane]; ++y) {
      for (std::uint32_t x = 0; x < visible[plane]; ++x) {
        std::uint16_t sample = 0;
        if (bps == 1) {
          sample = raw[cursor++];
        } else {
          sample = static_cast<std::uint16_t>(
              raw[cursor] | (static_cast<std::uint16_t>(raw[cursor + 1]) << 8));
          cursor += 2;
        }
        if (colorspace.depth == BitDepth::k10 && sample > 1023) {
          if (error != nullptr) {
            *error = "10-bit Y4M sample exceeds 1023";
          }
          return false;
        }
        dst.at(x, y) = sample;
      }
    }
  }
  for (int plane = 0; plane < plane_count; ++plane) {
    replicate_borders(picture->planes[plane]);
  }

  if (info != nullptr) {
    info->width = width;
    info->height = height;
    info->fps_num = fps_num;
    info->fps_den = fps_den;
    info->chroma = colorspace.chroma;
    info->depth = colorspace.depth;
    info->colorspace = y4m_colorspace_tag(colorspace.chroma, colorspace.depth)
                           .substr(1);
  }
  return true;
}

bool write_y4m_picture(const std::filesystem::path& path, const Picture& picture,
                       std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    if (error != nullptr) {
      *error = "cannot write " + path.string();
    }
    return false;
  }
  const Format& format = picture.format;
  file << "YUV4MPEG2 W" << format.width << " H" << format.height
       << " F1:1 Ip A0:0 " << y4m_colorspace_tag(format.chroma, format.depth)
       << "\n";
  file << "FRAME\n";

  const int bps = bytes_per_sample(format.depth);
  const int plane_count = num_planes(format.chroma);
  for (int plane = 0; plane < plane_count; ++plane) {
    const PlaneBuffer& src = picture.planes[plane];
    std::vector<std::uint8_t> row(static_cast<std::size_t>(src.visible_width) *
                                  static_cast<std::size_t>(bps));
    for (std::uint32_t y = 0; y < src.visible_height; ++y) {
      for (std::uint32_t x = 0; x < src.visible_width; ++x) {
        const std::uint16_t sample = src.at(x, y);
        if (bps == 1) {
          row[x] = static_cast<std::uint8_t>(sample);
        } else {
          row[static_cast<std::size_t>(x) * 2U] =
              static_cast<std::uint8_t>(sample & 0xFFU);
          row[static_cast<std::size_t>(x) * 2U + 1U] =
              static_cast<std::uint8_t>((sample >> 8) & 0x03U);
        }
      }
      file.write(reinterpret_cast<const char*>(row.data()),
                 static_cast<std::streamsize>(row.size()));
    }
  }
  if (!file) {
    if (error != nullptr) {
      *error = "write failed: " + path.string();
    }
    return false;
  }
  return true;
}

}  // namespace av2enc::fmt
