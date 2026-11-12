// Three-frame M2 compound stream builder (gate + unit-test driver).
//
//   frame 0  OBU_CLOSED_LOOP_KEY    order_hint 0, refresh all slots
//   frame 1  OBU_REGULAR_TILE_GROUP order_hint 4, single ref {0} -> slot 1
//   frame 2  OBU_REGULAR_TILE_GROUP order_hint 2, refs {0,1}, reference_select,
//            skip_mode_present and the compound tile from
//            compound_syntax.cpp
//
// Frames 0 and 1 are encoded by the M2 core; this builder receives their tile
// payloads and reconstructions and re-wraps them with a sequence header that
// reflects the compound configuration.  Frame 2 is coded here.
//
// B-frame interaction: frame 1 is decoded before frame 2 but carries the
// larger order hint (4).  Frame 2's order hint (2) sits between the two
// references, which is what §5.20.7.6 opfl_allowed_for_refs() and the is_joint
// context require: one past (d>0) and one future (d<0) reference.  With
// monotonic_output_order_flag = 1 the output order stays the decode order, so
// the future reference is a prediction-distance relationship, not an output
// reorder.  This is the smallest legal way to exercise `use_optflow S()`
// without a display-order reordering feature.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "encoder/m2/compound_stream/compound_syntax.h"

namespace av2enc::m2::compound {

struct RefPlaneView {
  const std::uint8_t* y = nullptr;
  const std::uint8_t* u = nullptr;
  const std::uint8_t* v = nullptr;
  int stride_y = 0;
  int stride_uv = 0;
};

struct CompoundStreamInput {
  int width = 0;
  int height = 0;
  std::uint8_t base_q_idx = 90;
  std::span<const std::uint8_t> key_tile;
  std::span<const std::uint8_t> future_tile;
  RefPlaneView key_recon;
  RefPlaneView future_recon;
  // One entry per 64x64 superblock, raster order.
  std::vector<CompoundBlockPlan> leaves;
  CompoundFrameConfig config;
};

struct CompoundStreamOutput {
  std::vector<std::vector<std::uint8_t>> units;
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  std::uint32_t tile_bytes = 0;
  int order_hint = 2;
  int future_order_hint = 4;
};

// Builds the three temporal units.  `config.ref_map` must already describe
// frame 2: NumTotalRefs = 2, ref_frame_idx = {0,1}, the order hints and
// reference_select.  Every other sequence-header flag is derived from config.
bool build_compound_stream(const CompoundStreamInput& input,
                           CompoundStreamOutput& output, std::string& error);

// Re-encodes frame 2's tile with the given features forced off and compares it
// with the M2 core single-reference writer for the same all-GLOBALMV plan.
// This is the "all disabled" byte-identity control: it proves the new writer
// does not change the pre-existing single-reference syntax.  Returns the tile
// byte count on success.
bool check_disabled_byte_identity(int width, int height,
                                  std::uint8_t base_q_idx,
                                  bool* identical, std::size_t* bytes,
                                  std::string& error);

}  // namespace av2enc::m2::compound
