// Strict Y4M I/O for the format layer (8/10-bit, 4:2:0/4:2:2/4:4:4/4:0:0).
//
// The product input reader (`src/core/io/y4m.h`) is deliberately 8-bit 4:2:0.
// The format still paths need the other chromatic samplings and 10-bit planar
// samples (two bytes per component, little-endian, LSB aligned).  This reader
// is a *strict* parser: the Y4M coloursapce tag must match the requested
// format exactly, no implicit up/down sampling or depth conversion is done.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "formats/format.h"

namespace av2enc::fmt {

struct Y4mFormatInfo {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t fps_num = 0;
  std::uint32_t fps_den = 0;
  ChromaFormat chroma = ChromaFormat::k420;
  BitDepth depth = BitDepth::k8;
  std::string colorspace;  // tag without the leading 'C', e.g. "420p10"
};

// Parses the header and reads exactly one frame. `picture` is allocated with
// coded-size planes and edge-replicated padding.  Returns false with a precise
// message for an unsupported coloursapce, a truncated/oversized frame, a
// sample outside [0, (1<<depth)-1], or a malformed header.
bool read_y4m_picture(const std::filesystem::path& path, Picture* picture,
                      Y4mFormatInfo* info, std::string* error);

// Writes one frame using the visible rectangle of `picture`.  The coloursapce
// tag is derived from the format (C420/C422/C444/Cmono and the *p10 variants).
bool write_y4m_picture(const std::filesystem::path& path, const Picture& picture,
                       std::string* error);

// Canonical Y4M coloursapce tag (with the leading 'C').
std::string y4m_colorspace_tag(ChromaFormat chroma, BitDepth depth);

}  // namespace av2enc::fmt
