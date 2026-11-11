#include "encoder/m1/sub8/sub8_minimal.h"

#include <algorithm>

namespace av2enc::m1::sub8 {

M1Config minimal_config(const FrameGeometry& geometry,
                        const SyntaxOptions& options) {
  M1Config config;
  config.max_frame_width = static_cast<std::uint16_t>(geometry.width);
  config.max_frame_height = static_cast<std::uint16_t>(geometry.height);
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = options.disable_cdf_update;
  config.enable_intra_edge_filter = true;
  config.enable_cfl_intra = options.enable_cfl;
  config.enable_chroma_dctonly = true;
  config.enable_idtx_intra = false;
  config.tx_mode_select = false;
  config.reduced_tx_set = 0;
  config.apply_deblock_y_vert = true;
  config.apply_deblock_y_horz = true;
  config.apply_deblock_u = true;
  config.apply_deblock_v = true;
  config.cdef.enable = false;
  config.ccso.enable = false;
  return config;
}

bool make_minimal_frame(const FrameGeometry& geometry,
                        std::span<const Node> superblocks,
                        const SyntaxOptions& options,
                        std::vector<std::uint8_t>& frame,
                        std::vector<DecodedLeaf>* round_trip, std::string& error) {
  if (geometry.width <= 0 || geometry.height <= 0) {
    error = "frame geometry is empty";
    return false;
  }
  if (geometry.width > 65536 || geometry.height > 65536) {
    error = "frame dimensions exceed the header range";
    return false;
  }
  std::vector<std::uint8_t> tile;
  if (!write_tile(geometry, superblocks, options, tile, error)) return false;

  std::vector<DecodedLeaf> leaves;
  if (!read_tile(geometry, tile, options, leaves, error)) return false;
  if (round_trip != nullptr) *round_trip = leaves;

  const M1Config config = minimal_config(geometry, options);
  frame = assemble_m1_still_picture(config, tile);
  return true;
}

}  // namespace av2enc::m1::sub8
