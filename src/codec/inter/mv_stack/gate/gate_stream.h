// Decoder gate for the complete §7.12.2 MV stack, `allow_warpmv_mode` and the
// §7.12.2.20 extra search.
//
// Stream shape (four temporal units):
//   frame 0  CLK  intra key                 (M2 core tile)
//   frame 1  P    single ref slot 0         (M2 core tile)
//   frame 2  P    single ref slot 1         (this writer, allow_warpmv_mode = 0)
//   frame 3  P    NumTotalRefs = 2,         (this writer, allow_warpmv_mode = 1)
//                 ref_frame_idx = {1, 0}, reference_select = 0,
//                 seq_enabled_motion_modes[DELTAWARP] = 1
//
// Frame 3 is a real multi-reference M2 frame.  Every block is a 64x64
// PARTITION_NONE leaf, which is the only block size where the §7.12.2.20
// `Block_Width > 32 && Block_Height > 32` MVP-combination gate fires.  The leaf
// pattern includes:
//   * NEARMV whose DRL index names a §7.12.2.20 candidate (an MVP combination or
//     the global-MV candidate), i.e. an index >= the stack size before the extra
//     search;
//   * GLOBALMV (the global-MV candidate path);
//   * WARPMV (is_warp = 1, warp_mv = 1, warp_idx >= 2) selecting an identity
//     model from the DeriveWrl warp stack.
//
// The reconstruction is produced from the same stack this module builds, with
// `mc::predict_block_420`; the gate then requires dav2d to reproduce it
// sample-for-sample.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::inter::mv_stack_full::gate {

struct PlaneRef {
  const std::uint8_t* y = nullptr;
  const std::uint8_t* u = nullptr;
  const std::uint8_t* v = nullptr;
  int stride_y = 0;
  int stride_uv = 0;
};

struct StreamInput {
  int width = 0;
  int height = 0;
  std::uint32_t order_hint = 3;
  std::span<const std::uint8_t> key_tile;
  std::span<const std::uint8_t> p1_tile;
  PlaneRef key_recon;  // slot 0
  PlaneRef p1_recon;   // slot 1
};

struct StreamCoverage {
  int near_extra = 0;          // NEARMV indices that only exist after §7.12.2.20
  int near_neighbour = 0;      // NEARMV indices inside the scanned neighbourhood
  int global_mv = 0;
  int warpmv = 0;
  int is_warp_bits = 0;  // leaves that coded the §5.20.7.6 is_warp symbol
  int mvp_combinations = 0;    // leaves whose final stack grew past the scan
  int max_stack = 0;
  int max_warp_found = 0;
  int ref_index_bits = 0;
};

struct StreamOutput {
  std::vector<std::vector<std::uint8_t>> units;
  std::vector<std::uint8_t> recon2_y;
  std::vector<std::uint8_t> recon2_u;
  std::vector<std::uint8_t> recon2_v;
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  std::uint32_t p2_tile_bytes = 0;
  std::uint32_t p3_tile_bytes = 0;
  StreamCoverage coverage;
};

bool build_stream(const StreamInput& input, StreamOutput& output,
                  std::string& error);

}  // namespace av2enc::inter::mv_stack_full::gate
