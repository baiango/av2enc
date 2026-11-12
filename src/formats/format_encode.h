// Composition root for the format still paths (`--bit-depth 10`, `--chroma`).
//
// The format components own their own header writer, per-plane syntax, entropy
// and reconstruction; this file is the product-facing adapter that reads the
// strict Y4M input, selects the matching component, writes the IVF and (when
// requested) dumps the encoder reconstruction as a Y4M of the same format.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "formats/format.h"

namespace av2enc::formats {

using av2enc::fmt::BitDepth;
using av2enc::fmt::ChromaFormat;

struct StillFormatOptions {
  ChromaFormat chroma = ChromaFormat::k420;
  BitDepth depth = BitDepth::k8;
  std::uint16_t base_q_idx = 90;
  std::filesystem::path recon_out;  // empty = do not dump
};

struct StillFormatStats {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t fps_num = 0;
  std::uint32_t fps_den = 0;
  ChromaFormat chroma = ChromaFormat::k420;
  BitDepth depth = BitDepth::k8;
  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;
  std::uint64_t ivf_bytes = 0;
};

// True when the requested format is served by a wired format component.
// 10-bit is 4:2:0 only (M10-derived path); 4:2:2/4:4:4/4:0:0 are 8-bit only.
bool format_still_supported(ChromaFormat chroma, BitDepth depth,
                            std::string* reason);

// Reads `input`, encodes one still picture in the requested format and writes
// `output` (IVF).  The input coloursapce must match the requested format; no
// implicit conversion is done.  Returns false with a precise error.
bool encode_still_format(const std::filesystem::path& input,
                         const std::filesystem::path& output,
                         const StillFormatOptions& options,
                         StillFormatStats* stats, std::string* error);

}  // namespace av2enc::formats
