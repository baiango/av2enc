#pragma once

#include "bitstream/bitwriter.h"

#include <cstdint>
#include <span>
#include <vector>

// AV2 v1.0.0 OBU/header writer for the integrated M1 still picture.
//
// The M0 writer in bitstream/m0_headers.* is intentionally left untouched. M1
// needs a different sequence-header tool set (intra edge filter, CfL, chroma
// DCT-only transforms) and configurable deblocking flags, so the few extra
// bits live here rather than changing the pinned M0 bytes.
namespace av2enc::m1 {

struct M1Config {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // level 5.1
  std::uint8_t base_q_idx = 90;
  bool disable_cdf_update = true;
  bool enable_intra_edge_filter = true;
  bool enable_cfl_intra = true;
  // Keeps chroma on DCT_DCT, so the DCT-only coefficient scan/context path is
  // used for every chroma block.
  bool enable_chroma_dctonly = true;
  // M1 starts with TX_MODE_LARGEST (one transform per coding block). Set true
  // for TX_MODE_SELECT once TX partition syntax is wired.
  bool tx_mode_select = false;
  // apply_deblocking_filter[0..3].
  bool apply_deblock_y_vert = true;
  bool apply_deblock_y_horz = true;
  bool apply_deblock_u = true;
  bool apply_deblock_v = true;
};

std::vector<std::uint8_t> make_sequence_header_obu(const M1Config& config);

// Writes frame_header(isFirst = 1) -> frame_header_info() (§5.18.1-§5.18.2),
// without the leading is_first_tile_group bit and without byte_alignment().
// Returns the number of syntax bits written.
//
// Deblocking planes follow deblocking_filter_params() (§5.18.5.2): after
// apply_deblocking_filter[0..3], each enabled plane writes one
// df_delta_q_present f(1) before tx_mode_select. M1 always sends delta 0, so
// an enabled plane contributes exactly one zero bit. When deblocking is off no
// delta bit is present.
std::size_t write_frame_header(BitWriter& writer, const M1Config& config);

std::vector<std::uint8_t> assemble_m1_still_picture(
    const M1Config& config, std::span<const std::uint8_t> tile_payload);

}  // namespace av2enc::m1
