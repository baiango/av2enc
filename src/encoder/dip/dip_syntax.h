// AV2 v1.0.0 DIP mode info syntax codec (§5.20.7.9, §6.19.7.8).
//
// The syntax is:
//
//   dip_mode_info( ) {
//     use_dip = 0
//     if ( enable_dip &&
//          YMode == DC_PRED && PaletteSizeY == 0 &&
//          Block_Width > 4 && Block_Height > 4 &&
//          Block_Width * Block_Height >= 128 ) {
//       use_dip        S()
//       if ( use_dip ) {
//         dip_transpose L(1)
//         dip_mode      S()
//       }
//     }
//   }
//
// The caller must place the codec at the point where `dip_mode_info()` is
// invoked: after the intra Y/UV mode and palette syntax, and before the
// residual/transform syntax.  TreeType == CHROMA_PART never invokes it.
//
// This wrapper uses the repository's normative §8.2 range coder.  It owns the
// two Tile*Cdf copies (one per tile) and updates them exactly like the decoder.

#pragma once

#include "encoder/dip/dip.h"
#include "entropy/av2/av2_entropy.h"

#include <array>
#include <cstdint>
#include <string>

namespace av2enc::dip {

// Per-tile copies of TileUseDipCdf and TileDipModeCdf.  The default values are
// the §9.3 Default_* arrays.
struct DipCdfState {
  std::array<std::array<std::uint16_t, 3>, kUseDipContexts> use_dip{};
  std::array<std::uint16_t, kNumModes + 1> dip_mode{};

  DipCdfState() noexcept { reset(); }
  void reset() noexcept;
};

// Writes one dip_mode_info() syntax structure.  When the context makes the
// structure absent, `info` must be the default {false,false,0}; the function
// succeeds without writing bits.  Returns false and sets `error` on an invalid
// combination or coder failure.
bool write_dip_mode_info(const SyntaxContext& context,
                         const DipModeInfo& info, av2::Encoder& encoder,
                         DipCdfState& cdfs,
                         std::string* error = nullptr) noexcept;

// Reads one dip_mode_info() syntax structure.  Missing (not signalled) fields
// are left at their inferred defaults, exactly as in the syntax pseudocode.
bool read_dip_mode_info(const SyntaxContext& context, av2::Decoder& decoder,
                        DipCdfState& cdfs, DipModeInfo& info,
                        std::string* error = nullptr) noexcept;

}  // namespace av2enc::dip
