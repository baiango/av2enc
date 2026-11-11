// Minimal Y4M I/O for the M10 tool.
//
// Input: 8-bit C420 (converted with fmt::convert_420_8_to_picture) or 10-bit
// C420p10 (loaded directly, MSB zero / LSB aligned).  Output: C420p10, which
// is what the dav2d gate compares against the encoder reconstruction.
#pragma once

#include <cstdint>
#include <string>

#include "encoder/format/format.h"

namespace av2enc::m10 {

struct Y4mFrameInfo {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t bit_depth = 0;
  std::uint32_t chroma_width = 0;
  std::uint32_t chroma_height = 0;
};

// Reads the first frame of `path`.  Accepts C420/C420jpeg/C420mpeg2/C420paldv
// (8-bit) and C420p10.
bool read_y4m_10bit(const std::string& path, fmt::Picture* picture,
                    Y4mFrameInfo* info, std::string* error);

// Writes one C420p10 frame using the visible rectangle of `picture`.
bool write_y4m_10bit(const std::string& path, const fmt::Picture& picture,
                     std::string* error);

// Writes the visible 10-bit planes of an M10 reconstruction.
bool write_y4m_10bit_planes(const std::string& path, std::uint32_t width,
                            std::uint32_t height,
                            const std::uint16_t* y, int y_stride,
                            const std::uint16_t* u, int u_stride,
                            const std::uint16_t* v, int v_stride,
                            int chroma_width, int chroma_height,
                            std::string* error);

}  // namespace av2enc::m10
