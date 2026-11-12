// Minimal compound-P-frame stream builder (decoder gate only).
//
// Produces a three-frame IVF payload:
//
//   frame 0  OBU_CLOSED_LOOP_KEY   intra, order_hint 0, refresh all slots
//   frame 1  OBU_REGULAR_TILE_GROUP single-ref P, order_hint 1, slot 0 -> slot 1
//   frame 2  OBU_REGULAR_TILE_GROUP compound P, order_hint 2,
//            NumTotalRefs = 2, ref_frame_idx = {0, 1}, reference_select = 1
//
// Frames 0 and 1 are encoded by the M2 core; this module receives their tile
// payloads and their reconstructions and re-wraps them with a sequence header
// that enables masked compound / CWP / implicit-mask blending and with frame
// headers that carry the reference map.  Frame 2 is written by
// gate/compound_tile.cpp.
//
// This path is deliberately separate from the M2 integration path.  See
// README.md for the exact syntax/order notes and the reason the M2 core cannot
// emit this stream itself.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "codec/inter/multiref/gate/compound_tile.h"

namespace av2enc::video::multiref::gate {

struct CompoundStreamRefFrame {
  const std::uint8_t* y = nullptr;
  const std::uint8_t* u = nullptr;
  const std::uint8_t* v = nullptr;
  int stride_y = 0;
  int stride_uv = 0;
};

struct CompoundStreamInput {
  int width = 0;
  int height = 0;
  std::uint32_t order_hint = 2;
  std::uint8_t base_q_idx = 90;
  // M2 core outputs.
  std::span<const std::uint8_t> key_tile;
  std::span<const std::uint8_t> p1_tile;
  CompoundStreamRefFrame key_recon;
  CompoundStreamRefFrame p1_recon;
  // Compound choices, one per superblock, raster order.
  std::vector<GateLeaf> leaves;
  bool enable_masked_compound = true;
  bool enable_cwp = true;
  bool enable_imp_msk_bld = true;
};

struct CompoundStreamOutput {
  // One temporal unit per frame: TD + [SEQ] + frame OBU.
  std::vector<std::vector<std::uint8_t>> units;
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  std::uint32_t p2_tile_bytes = 0;
  RefFrameMap ref_map;
};

// Default mode assignment for a gate stream: cycles
// average / CWP(12) / wedge / diffwtd over the superblocks.
std::vector<GateLeaf> make_default_gate_leaves(int superblock_count);

bool build_compound_stream(const CompoundStreamInput& input,
                           CompoundStreamOutput& output, std::string& error);

}  // namespace av2enc::video::multiref::gate
