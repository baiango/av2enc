#include "encoder/m10/m10_y4m.h"

#include "encoder/m10/m10_still.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace av2enc::m10 {
namespace {

bool read_header_line(std::ifstream& file, std::string* line, std::string* error) {
  line->clear();
  char c = 0;
  while (file.get(c)) {
    if (c == '\n') return true;
    if (c == '\r') continue;
    line->push_back(c);
  }
  if (error != nullptr) *error = "truncated Y4M header";
  return false;
}

bool parse_uint(const std::string& token, std::uint32_t* out) {
  if (token.empty()) return false;
  try {
    *out = static_cast<std::uint32_t>(std::stoul(token));
  } catch (...) {
    return false;
  }
  return true;
}

bool fill_10bit_visible(const std::vector<std::uint16_t>& y,
                        const std::vector<std::uint16_t>& u,
                        const std::vector<std::uint16_t>& v, std::uint32_t w,
                        std::uint32_t h, fmt::Picture* picture,
                        std::string* error) {
  fmt::Format format;
  format.width = w;
  format.height = h;
  format.chroma = fmt::ChromaFormat::k420;
  format.depth = fmt::BitDepth::k10;
  if (!fmt::allocate_picture(format, *picture, *error)) return false;
  const std::uint32_t cw = (w + 1U) / 2U;
  const std::uint32_t ch = (h + 1U) / 2U;
  for (std::uint32_t yy = 0; yy < h; ++yy) {
    for (std::uint32_t xx = 0; xx < w; ++xx) {
      picture->planes[0].at(xx, yy) = y[static_cast<std::size_t>(yy) * w + xx];
    }
  }
  for (std::uint32_t yy = 0; yy < ch; ++yy) {
    for (std::uint32_t xx = 0; xx < cw; ++xx) {
      const std::size_t i = static_cast<std::size_t>(yy) * cw + xx;
      picture->planes[1].at(xx, yy) = u[i];
      picture->planes[2].at(xx, yy) = v[i];
    }
  }
  fmt::replicate_borders(picture->planes[0]);
  fmt::replicate_borders(picture->planes[1]);
  fmt::replicate_borders(picture->planes[2]);
  return true;
}

}  // namespace

bool read_y4m_10bit(const std::string& path, fmt::Picture* picture,
                    Y4mFrameInfo* info, std::string* error) {
  if (error != nullptr) error->clear();
  if (picture == nullptr) {
    if (error != nullptr) *error = "null picture";
    return false;
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    if (error != nullptr) *error = "cannot open " + path;
    return false;
  }
  std::string header;
  if (!read_header_line(file, &header, error)) return false;
  if (header.rfind("YUV4MPEG2", 0) != 0) {
    if (error != nullptr) *error = "not a Y4M file";
    return false;
  }
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::string colorspace = "C420";
  std::istringstream tokens(header);
  std::string token;
  while (tokens >> token) {
    if (token[0] == 'W') {
      if (!parse_uint(token.substr(1), &width)) width = 0;
    } else if (token[0] == 'H') {
      if (!parse_uint(token.substr(1), &height)) height = 0;
    } else if (token[0] == 'C') {
      colorspace = token.substr(1);
    }
  }
  if (width == 0 || height == 0) {
    if (error != nullptr) *error = "missing Y4M dimensions";
    return false;
  }
  std::string frame_tag;
  if (!read_header_line(file, &frame_tag, error)) return false;
  if (frame_tag.rfind("FRAME", 0) != 0) {
    if (error != nullptr) *error = "missing Y4M FRAME tag";
    return false;
  }

  const std::uint32_t cw = (width + 1U) / 2U;
  const std::uint32_t ch = (height + 1U) / 2U;
  const bool ten_bit =
      colorspace == "420p10" || colorspace == "C420p10" ||
      colorspace.rfind("420p10", 0) == 0;
  const std::size_t y_count = static_cast<std::size_t>(width) * height;
  const std::size_t c_count = static_cast<std::size_t>(cw) * ch;

  if (ten_bit) {
    std::vector<std::uint8_t> raw((y_count + 2U * c_count) * 2U);
    file.read(reinterpret_cast<char*>(raw.data()),
              static_cast<std::streamsize>(raw.size()));
    if (static_cast<std::size_t>(file.gcount()) != raw.size()) {
      if (error != nullptr) *error = "truncated 10-bit Y4M frame";
      return false;
    }
    std::vector<std::uint16_t> y(y_count);
    std::vector<std::uint16_t> u(c_count);
    std::vector<std::uint16_t> v(c_count);
    const auto load = [&raw](std::size_t index) {
      return static_cast<std::uint16_t>(
          raw[index * 2] | (static_cast<std::uint16_t>(raw[index * 2 + 1]) << 8));
    };
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < y_count; ++i) y[i] = load(cursor++);
    for (std::size_t i = 0; i < c_count; ++i) u[i] = load(cursor++);
    for (std::size_t i = 0; i < c_count; ++i) v[i] = load(cursor++);
    for (std::uint16_t sample : y) {
      if (sample > 1023) {
        if (error != nullptr) *error = "10-bit Y4M luma exceeds 1023";
        return false;
      }
    }
    for (std::size_t i = 0; i < c_count; ++i) {
      if (u[i] > 1023 || v[i] > 1023) {
        if (error != nullptr) *error = "10-bit Y4M chroma exceeds 1023";
        return false;
      }
    }
    if (!fill_10bit_visible(y, u, v, width, height, picture, error)) {
      return false;
    }
  } else {
    if (colorspace != "420" && colorspace != "420jpeg" &&
        colorspace != "420mpeg2" && colorspace != "420paldv") {
      if (error != nullptr) {
        *error = "unsupported Y4M colorspace C" + colorspace +
                 " (M10 accepts C420 and C420p10)";
      }
      return false;
    }
    std::vector<std::uint8_t> raw(y_count + 2U * c_count);
    file.read(reinterpret_cast<char*>(raw.data()),
              static_cast<std::streamsize>(raw.size()));
    if (static_cast<std::size_t>(file.gcount()) != raw.size()) {
      if (error != nullptr) *error = "truncated 8-bit Y4M frame";
      return false;
    }
    if (!convert_8bit_420_to_picture(raw.data(), raw.data() + y_count,
                                     raw.data() + y_count + c_count, width,
                                     height, width, picture, error)) {
      return false;
    }
  }

  if (info != nullptr) {
    info->width = width;
    info->height = height;
    info->bit_depth = ten_bit ? 10U : 8U;
    info->chroma_width = cw;
    info->chroma_height = ch;
  }
  return true;
}

bool write_y4m_10bit(const std::string& path, const fmt::Picture& picture,
                     std::string* error) {
  return write_y4m_10bit_planes(
      path, picture.format.width, picture.format.height,
      picture.planes[0].samples.data(), static_cast<int>(picture.planes[0].stride),
      picture.planes[1].samples.data(), static_cast<int>(picture.planes[1].stride),
      picture.planes[2].samples.data(), static_cast<int>(picture.planes[2].stride),
      static_cast<int>((picture.format.width + 1U) / 2U),
      static_cast<int>((picture.format.height + 1U) / 2U), error);
}

bool write_y4m_10bit_planes(const std::string& path, std::uint32_t width,
                            std::uint32_t height, const std::uint16_t* y,
                            int y_stride, const std::uint16_t* u, int u_stride,
                            const std::uint16_t* v, int v_stride,
                            int chroma_width, int chroma_height,
                            std::string* error) {
  if (y == nullptr || u == nullptr || v == nullptr) {
    if (error != nullptr) *error = "null reconstruction plane";
    return false;
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    if (error != nullptr) *error = "cannot write " + path;
    return false;
  }
  file << "YUV4MPEG2 W" << width << " H" << height << " F1:1 Ip A0:0 C420p10\n";
  file << "FRAME\n";
  const auto put_plane = [&file](const std::uint16_t* plane, int stride,
                                 int w, int h) {
    std::vector<std::uint8_t> row(static_cast<std::size_t>(w) * 2U);
    for (int yy = 0; yy < h; ++yy) {
      for (int xx = 0; xx < w; ++xx) {
        const std::uint16_t value =
            plane[static_cast<std::size_t>(yy) * stride + xx];
        row[static_cast<std::size_t>(xx) * 2U] =
            static_cast<std::uint8_t>(value & 0xFFU);
        row[static_cast<std::size_t>(xx) * 2U + 1U] =
            static_cast<std::uint8_t>((value >> 8) & 0x03U);
      }
      file.write(reinterpret_cast<const char*>(row.data()),
                 static_cast<std::streamsize>(row.size()));
    }
  };
  put_plane(y, y_stride, static_cast<int>(width), static_cast<int>(height));
  put_plane(u, u_stride, chroma_width, chroma_height);
  put_plane(v, v_stride, chroma_width, chroma_height);
  if (!file) {
    if (error != nullptr) *error = "write failed: " + path;
    return false;
  }
  return true;
}

}  // namespace av2enc::m10
