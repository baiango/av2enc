#pragma once

#include "bitstream/m0_headers.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace av2enc::m0 {

// One 8-bit 4:2:0 still picture. Planes are tightly packed, Y then U then V.
struct StillPicture {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::span<const std::uint8_t> y;
  std::span<const std::uint8_t> u;
  std::span<const std::uint8_t> v;
};

struct StillEncodeResult {
  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;  // one temporal unit (TD + SEQ + CLK)
  // One IVF frame payload: TD + SEQ + CLK, ready for IvfWriter.
  std::vector<std::uint8_t> frame;
  // Reconstructed luma/chroma, for encoder-side PSNR and debugging.
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
};

// Encodes the M0 still picture: one key frame, one tile, 8-bit 4:2:0,
// DC intra only, TX_MODE_LARGEST, no loop filters / palette / CfL / film grain.
//
// `config.max_frame_width/height` must equal picture.width/height and
// `config.base_q_idx` must be 90 (the coefficient CDF tables are generated for
// the base_q_idx <= 90 quantiser-context slice).
//
// Debug/integration partition modes. `k32x32` is the M0 target
// (64x64 SB -> 32x64 -> 32x32). `k64x64Single` codes PARTITION_NONE for the
// whole superblock, which is the smallest possible M0-like stream and is used
// to isolate entropy-coder bugs from partition-tree bugs.
enum class M0PartitionMode {
  k32x32 = 0,
  k64x64Single = 1,
  k64x64VertNone = 2,
  k64x64HorzNone = 3,
  k32x32HorzFirst = 4,
};

// Returns false and fills `error` on invalid input or an internal coder error.
bool encode_still_picture(const StillPicture& picture,
                          const StillPictureConfig& config,
                          StillEncodeResult& result,
                          std::string& error,
                          M0PartitionMode partition_mode = M0PartitionMode::k32x32);

}  // namespace av2enc::m0
