// Minimal AV2 still-picture stream carrying use_256x256_superblock = 1.
//
// §5.18.1: an intra frame always has SbSize = BLOCK_128X128, even when
// use_256x256_superblock = 1.  The flag therefore changes the sequence header
// (and get_seq_sb_size()/sequence-level unit sizing), while the intra tile is
// a 128x128-superblock tree.  This writer exercises exactly that path:
//
//   * use_256x256_superblock = 1, use_128x128_superblock is *not present*
//     and is inferred to 0 (§5.4.3);
//   * one tile, frozen default CDFs (disable_cdf_update = 1);
//   * DC_PRED luma/chroma, no residual, no CDEF/CCSO/deblock/restoration;
//   * a deterministic square-leaf partition tree (8x8..64x64).
//
// `make_use128_control_stream()` writes the same tile under the equivalent
// use_256=0/use_128=1 header.  Byte-identical tiles prove the 256 flag does
// not alter the intra partition tree, and the gate uses that as a control.
//
// This is not a production encoder.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "codec/partition/sb256/sb256.h"

namespace av2enc::sb256 {

struct StreamConfig {
  std::uint16_t width = 256;
  std::uint16_t height = 256;
  std::uint8_t base_q_idx = 90;
  std::uint8_t seq_level_idx = 7;  // 7 = level 5.1
  bool disable_cdf_update = true;
  bool enable_cfl_intra = true;
};

struct StreamResult {
  std::vector<std::uint8_t> frame;  // TD + SEQ + CLK OBUs, as IVF payload
  std::vector<std::uint8_t> tile;
  int leaves = 0;
  int superblocks = 0;
  std::size_t symbols = 0;
};

// The two sequence-header flag combinations this module writes.  They are
// mutually exclusive: use_128 is not parsed when use_256 is 1.
struct SbFlags {
  bool use_256x256_superblock = false;
  bool use_128x128_superblock = false;
};

// The reference tile uses square leaves from 8x8 to 64x64.  Reject geometries
// whose boundary recursion forces anything else.
bool stream_geometry_supported(const Geometry& geometry, std::string& error);

// use_256x256_superblock = 1 (the §5.4.3 flag under test).
bool make_sb256_stream(const StreamConfig& config, StreamResult& out,
                       std::string& error);

// use_256x256_superblock = 0, use_128x128_superblock = 1.  Same intra tile as
// make_sb256_stream(); used by tests and the gate as a positive control.
bool make_use128_control_stream(const StreamConfig& config, StreamResult& out,
                                std::string& error);

// Header pieces, exported for tests and for a future CLI/GPU integration.
std::size_t write_sequence_header(BitWriter& writer, const StreamConfig& config,
                                  const SbFlags& flags);
std::size_t write_frame_header(BitWriter& writer, const StreamConfig& config);
std::vector<std::uint8_t> make_sequence_header_obu(const StreamConfig& config,
                                                   const SbFlags& flags);
std::vector<std::uint8_t> make_temporal_delimiter_obu();

}  // namespace av2enc::sb256
