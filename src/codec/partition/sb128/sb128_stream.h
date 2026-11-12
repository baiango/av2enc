// Minimal 128x128-superblock AV2 still-picture stream.
//
// Emits one intra frame with:
//   * use_128x128_superblock = 1 (§5.4.3)
//   * one tile, frozen default CDFs (disable_cdf_update = 1)
//   * DC_PRED luma/chroma, no residual, no CDEF/CCSO/deblock/restoration
//   * a reference partition tree whose leaves are all 64x64
//
// The point is to exercise the parts the M1 writer cannot currently express:
// the 128 SB root's do_split -> do_square_split syntax, the different
// Partition_Size_Adjust context row, boundary-forced partitions and the
// 128-sized tile/CDEF geometry.  It is not a production encoder.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "codec/partition/sb128/sb128.h"

namespace av2enc::sb128 {

struct StreamConfig {
  std::uint16_t width = 128;
  std::uint16_t height = 128;
  std::uint8_t base_q_idx = 90;
  std::uint8_t seq_level_idx = 7;  // 7 = level 5.1
  bool disable_cdf_update = true;
  bool enable_cfl_intra = true;
  // Kept as a field so tests can compare the header with the flag clear.  The
  // stream builder requires true.
  bool use_128x128_superblock = true;
};

struct StreamResult {
  std::vector<std::uint8_t> frame;  // TD + SEQ + CLK OBUs, as IVF payload
  std::vector<std::uint8_t> tile;
  int leaves = 0;
  std::size_t symbols = 0;
};

// The reference tree writes 64x64 leaves.  Reject geometries whose right or
// bottom partial superblock must recurse below 64x64.
bool stream_geometry_supported(const Geometry& geometry, std::string& error);

// Build the complete IVF frame payload.  Returns false with `error` on any
// spec or plan inconsistency.
bool make_sb128_stream(const StreamConfig& config, StreamResult& out,
                       std::string& error);

// Header pieces, exported for tests and for a future CLI/GPU integration.
std::size_t write_sequence_header(BitWriter& writer,
                                  const StreamConfig& config);
std::size_t write_frame_header(BitWriter& writer,
                               const StreamConfig& config);
std::vector<std::uint8_t> make_sequence_header_obu(
    const StreamConfig& config);
std::vector<std::uint8_t> make_temporal_delimiter_obu();

}  // namespace av2enc::sb128
