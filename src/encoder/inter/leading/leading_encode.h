#pragma once

#include "encoder/m2/m2_frame.h"

#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::inter::leading {

// A planar 8-bit 4:2:0 picture owned by the caller.  Leading.cpp only frames
// pictures; this helper drives the existing M2 intra/inter picture encoder so
// the gate can produce a real leading-picture stream without duplicating the
// M1/M2 entropy path.
struct Picture {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

struct EncodedPicture {
  // §5.20 tile payload including §8.2.4 exit_symbol().
  std::vector<std::uint8_t> tile_payload;
  // Encoder reconstruction, used by the gate to compare decoder output.
  Picture recon;
};

// Encodes one intra picture (the tile payload is the same for a CLK and an
// OLK; only the frame/OBU header differs).
bool encode_intra_picture(const Picture& source, const m2::M2Options& options,
                          EncodedPicture& out, std::string& error);

// Encodes `source` as a single-reference P picture against `reference`.
bool encode_inter_picture(const Picture& reference, const Picture& source,
                          const m2::M2Options& options, EncodedPicture& out,
                          std::string& error);

}  // namespace av2enc::inter::leading
