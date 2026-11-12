// AV2 v1.0.0 10-bit sequence/frame/tile-group header writer.
//
// This is the 10-bit header half of `src/encoder/hbd/`.  It writes the same
// OBU order as the verified M10 writer (TD -> SEQ -> CLK, §7.3.3) but makes the
// bit-depth-dependent fields explicit and switches on the *raw* frame qindex:
//
//   §5.4.1      uvlc(bit_depth_idc), 0 = 10-bit, 1 = 8-bit
//   §6.4.1      MaxQ = 303 at 10-bit (255 at 8-bit)
//   §5.18.6.1   base_q_idx f(9) at 10-bit (f(8) at 8-bit)
//   §8.3.1.     the coefficient-CDF slice is NOT written; the decoder derives
//               it from the raw base_q_idx (see depth::qctx_slice)
//
// The writer accepts any qindex in [1, 303].  The *tile payload* writer reused
// from M10 only has the frozen slice-0 coefficient CDFs, so
// `hbd::encode_hbd_still()` restricts encoding to q <= 90 even though this
// header writer (and the reader) handle the full 10-bit MaxQ range.  See
// README.md "qindex seam".
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "formats/format.h"

namespace av2enc::hbd {

struct HbdConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // level 5.1 (Annex A.4)

  // 10-bit only.  Kept as the format enum so MaxQ/f(9) have one source.
  fmt::BitDepth depth = fmt::BitDepth::k10;

  // §6.4.1 Table 6.2 chroma_format_idc source (idc is not the enum order).
  // The default 4:2:0 keeps the M10 header bytes unchanged.
  fmt::ChromaFormat chroma = fmt::ChromaFormat::k420;

  // Raw frame qindex.  Must satisfy 1 <= base_q_idx <= MaxQ (303 at 10-bit).
  std::uint16_t base_q_idx = 90;

  // §5.18.2 / §8.2.6.  Frozen CDFs keep regression streams deterministic.
  bool disable_cdf_update = true;

  // Intra tool flags.  The M10-compatible still path keeps these off; the
  // writer exposes them for the integrated M1/M2 stream once the tile syntax
  // codes the matching symbols.
  bool enable_intra_edge_filter = false;
  bool enable_cfl_intra = false;
  bool enable_chroma_dctonly = false;

  // §5.18.2 tx_mode_select: 0 = TX_MODE_LARGEST.
  bool tx_mode_select = false;
  int reduced_tx_set = 0;

  // §5.18.2 deblocking_filter_params().  The M10-compatible path keeps every
  // flag off (the M10 tile syntax codes no deblock metadata).
  bool apply_deblock_y_vert = false;
  bool apply_deblock_y_horz = false;
  bool apply_deblock_u = false;
  bool apply_deblock_v = false;
};

// True when the config is a legal 10-bit still configuration.  Rejects 8-bit
// and 12-bit (only 10-bit is implemented here), zero/oversized dimensions,
// base_q_idx 0 (lossless) and base_q_idx > 303.
bool validate_config(const HbdConfig& config, std::string* error);

// sequence_header_obu() syntax bits only (no obu_extension_flag, no trailing
// bits).  bit_depth_idc comes from fmt::bit_depth_idc(k10) == 0.
std::size_t write_sequence_header(BitWriter& writer, const HbdConfig& config);

// sequence_header_obu() + obu_extension_flag + trailing_bits + OBU framing.
std::vector<std::uint8_t> make_sequence_header_obu(const HbdConfig& config);

// frame_header(isFirst = 1) -> frame_header_info() (§5.18.1-§5.18.2), without
// the leading is_first_tile_group bit and without byte_alignment().
std::size_t write_frame_header(BitWriter& writer, const HbdConfig& config);

// tile_group_obu() header: is_first_tile_group + frame_header +
// byte_alignment().
std::size_t write_tile_group_header(BitWriter& writer, const HbdConfig& config);

// TD + sequence header + OBU_CLOSED_LOOP_KEY (header + tile payload).
std::vector<std::uint8_t> assemble_still_picture(
    const HbdConfig& config, std::span<const std::uint8_t> tile_payload);

struct HeaderBitCounts {
  std::size_t sequence_syntax_bits = 0;
  std::size_t frame_syntax_bits = 0;   // excluding is_first_tile_group
  std::size_t tile_group_bits = 0;     // including is_first_tile_group
};

HeaderBitCounts header_bit_counts(const HbdConfig& config);

}  // namespace av2enc::hbd
