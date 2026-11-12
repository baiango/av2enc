// AV2 v1.0.0 OBU/header writer for the standalone 10-bit (M10) still picture.
//
// This writer is deliberately independent of the M1 core (`still_headers.*`,
// `still_syntax.*`, `still_encoder.*`) and of the M0 writer in `bitstream/`.  M10 is
// the minimal 10-bit 4:2:0 key-frame stream: one tile, 64x64 superblocks,
// DC intra, DCT only, no CfL / deblock / CDEF / CCSO / GDF / restoration /
// film grain.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (commit 38f28e7).
//
//   10-bit fields that differ from M0:
//     §6.4.1 Table 6.3  bit_depth_idc = 0 (0 = 10-bit, 1 = 8-bit)
//     §6.4.1 Table 6.3  MaxQ = MAXQ_8_BITS + 2*MAXQ_OFFSET = 303
//     §5.18.6.1         base_q_idx is f(9), not f(8)
//     §8.3.1            the coefficient CDF slice is chosen from the *raw*
//                       base_q_idx (<=90 -> slice 0), never by bit depth
//
// 12-bit is intentionally not signalable.  `bit_depth_idc` is one bit and
// values greater than 1 are reserved; this header only accepts 10-bit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "formats/format.h"

namespace av2enc::high_bitdepth {

struct HighBitDepthConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // level 5.1 (Annex A.4)

  // M10 is 10-bit only; stored as the format enum so the MaxQ/f(9) helpers are
  // the single source of truth.
  fmt::BitDepth depth = fmt::BitDepth::k10;

  // Raw qindex, not depth-normalized.  Must satisfy 1 <= base_q_idx <= MaxQ
  // (303 at 10-bit).  M10 pins 90 because its forward quantizer is the fixed
  // QP-90 reference; the writer rejects every other value instead of emitting
  // a stream the decoder would dequantize differently.
  std::uint16_t base_q_idx = 90;

  // Frozen CDFs keep the regression bitstream deterministic (§8.2.6).
  bool disable_cdf_update = true;
};

// Writes sequence_header_obu() syntax bits only: no §5.2.1 extension flag and
// no trailing_bits.  `bit_depth_idc` comes from fmt::bit_depth_idc().
std::size_t write_sequence_header(BitWriter& writer, const HighBitDepthConfig& config);

// sequence_header_obu() + obu_extension_flag + trailing_bits + OBU framing.
std::vector<std::uint8_t> make_sequence_header_obu(const HighBitDepthConfig& config);

// frame_header(isFirst = 1) -> frame_header_info() (§5.18.1-§5.18.2), without
// the leading is_first_tile_group bit and without byte_alignment().
std::size_t write_frame_header(BitWriter& writer, const HighBitDepthConfig& config);

// tile_group_obu() header: is_first_tile_group + frame_header + byte_alignment.
std::size_t write_tile_group_header(BitWriter& writer, const HighBitDepthConfig& config);

// TD + sequence header + OBU_CLOSED_LOOP_KEY (header + tile payload), in the
// §7.3.3 order.  Ready for the IVF muxer.
std::vector<std::uint8_t> assemble_high_bitdepth_still_picture(
    const HighBitDepthConfig& config, std::span<const std::uint8_t> tile_payload);

// Returns true when the config is a legal M10 stream configuration:
// 10-bit 4:2:0 geometry, 1 <= base_q_idx <= 90 (slice-0 tables) and
// base_q_idx <= fmt::max_quantizer(10) == 303.
bool validate_config(const HighBitDepthConfig& config, std::string* error);

// Exact syntax-bit counts for the sizes exercised by the M10 test/gate.
struct HeaderBitCounts {
  std::size_t sequence_syntax_bits;
  std::size_t frame_syntax_bits;   // excluding is_first_tile_group
  std::size_t tile_group_bits;     // including is_first_tile_group
};
HeaderBitCounts header_bit_counts(const HighBitDepthConfig& config);

}  // namespace av2enc::high_bitdepth
