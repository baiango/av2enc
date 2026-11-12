#ifndef AV2ENC_INTER_WARP_STACK_GATE_WARP_GATE_TILE_H_
#define AV2ENC_INTER_WARP_STACK_GATE_WARP_GATE_TILE_H_

// Minimal single-reference inter tile writer used only by the LOCALWARP
// decoder gate.  It is not part of the encoder integration path.
//
// Supported subset (the smallest stream that exercises §7.11.4, §7.12.2.3/.4/
// .9/.11, §7.12.3 and §7.13.3.19/.20 on a real decoder):
//   * one 64x64 PARTITION_NONE leaf per superblock, all inter, skip_flag = 1
//     (no residual);
//   * NumTotalRefs = 1, reference_select = 0 (RefFrame = {0, NONE});
//   * YMode NEWMV or WARP_NEWMV, motion mode SIMPLE, LOCALWARP or EXTENDWARP
//     (the gate uses LOCALWARP);
//   * DRL index and shell-coded MVD.
//
// Context derivation is deliberately normative:
//   * is_inter/skip/single_mode/drl contexts use the neighbour grids;
//   * the `is_warp` CDF row is selected by §7.11.4 `WarpMvCount` computed by
//     warp_stack.h, so a wrong warp context desynchronises the gate;
//   * `use_local_warp` uses the §8.3.2 NPos context.

#include <cstdint>
#include <string>
#include <vector>

#include "encoder/inter/warp_stack/warp_stack.h"

namespace av2enc::inter::warp::gate {

struct GateLeaf {
  bool is_warp = false;
  bool warp_mv = false;
  int single_mode = 1;  // 0 NEARMV, 1 GLOBALMV, 2 NEWMV (when !is_warp)
  bool use_local_warp = false;
  bool use_extend_warp = false;
  int drl_index = 0;
  // Either a literal MVD or an absolute target MV resolved against the
  // decoder's stack (pred = RefStackMv[drl_index][0]).
  bool use_target_mv = false;
  int target_mv_y = 0;
  int target_mv_x = 0;
  int mvd_y = 0;
  int mvd_x = 0;
  // Expected derived values, checked after the bits are written.
  int y_mode = kYModeGlobalMv;
  int motion_mode = kMotionSimple;
};

struct GateTilePlan {
  int width = 0;
  int height = 0;
  int sb_size = 64;
  std::vector<GateLeaf> leaves;  // one per superblock, raster order
};

struct GateTileConfig {
  bool allow_warpmv_mode = true;
  bool frame_enable_local_warp = true;
  bool frame_enable_extend_warp = false;
  bool force_integer_mv = false;
  bool disable_cdf_update = true;
  int max_drl_bits_minus_1 = 4;
  // Test-only: >= 0 forces the TileIsWarpCdf row.  The gate's
  // --negative-context mode uses 0 to demonstrate the §7.11.4 desync.
  int force_warp_mv_ctx = -1;
};

// Writes the range-coded tile payload including exit_symbol().  `grid_out`
// receives the final MI grid (block MVs, modes and warp models) so the gate
// reconstructs from exactly the decoded state.
bool write_gate_warp_tile(const GateTilePlan& plan, const GateTileConfig& config,
                          std::vector<std::uint8_t>& payload,
                          std::vector<WarpMi>* grid_out, int* mi_cols_out,
                          int* mi_rows_out, std::string& error);

}  // namespace av2enc::inter::warp::gate

#endif  // AV2ENC_INTER_WARP_STACK_GATE_WARP_GATE_TILE_H_
