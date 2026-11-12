// Minimal compound P-frame tile writer (gate only).
//
// The M2 core tile writer (video_syntax.cpp) codes single-reference inter blocks
// and the M2 frame/sequence headers pin reference_select = 0, NumTotalRefs = 1,
// enable_masked_compound = 0 and enable_cwp = 0.  A compound stream therefore
// cannot be produced without touching the M2 core, which this agent does not
// own.  Instead this file contains an independent, self-contained tile writer
// used only by the decoder gate; it is not part of the M2 integration path.
//
// Supported subset (the smallest stream that exercises compound prediction):
//   * one 64x64 PARTITION_NONE leaf per superblock, all blocks inter,
//     skip_flag = 1 (no residual), frame interpolation_filter = EIGHTAP;
//   * NumTotalRefs = 2 (and, for diagnostics, the general map path);
//   * YMode GLOBAL_GLOBALMV or NEAR_NEARMV; for NEAR_NEARMV the DRL indices are
//     supplied by the caller and must be below the decoder stack size;
//   * COMPOUND_AVERAGE (+CWP), COMPOUND_WEDGE, COMPOUND_DIFFWTD;
//   * no intra blocks, no skip_mode, no motion modes, no warp/BAWP/OPFL.
//
// Symbol order (§5.20.7.6 / .10 / .11 / .16 / .18):
//   is_inter, skip_flag, comp_mode, [comp_ref], is_joint,
//   compound_mode_non_joint, drl_idx(0), drl_idx(1), comp_group_idx,
//   [compound_type, {wedge_quad, wedge_angle, wedge_dist1|2, wedge_sign} |
//    mask_type], [cwp_idx].
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "codec/inter/multiref/compound_predict.h"
#include "codec/inter/multiref/ref_map.h"

namespace av2enc::video::multiref::gate {

// One 64x64 inter leaf.  The tile writer supports exactly one leaf per
// superblock so the partition syntax is a single do_split = 0.
struct GateLeaf {
  bool compound = true;
  // 19 = NEAR_NEARMV, 22 = GLOBAL_GLOBALMV (both non-joint).
  int y_mode = 22;
  int drl_index[2] = {0, 0};
  CompoundModeParams mode;
};

struct GateTilePlan {
  int width = 0;
  int height = 0;
  int sb_size = 64;
  // One entry per superblock, raster order.
  std::vector<GateLeaf> leaves;
};

struct GateTileConfig {
  bool enable_masked_compound = true;
  bool enable_cwp = true;
  bool disable_cdf_update = true;
  int max_drl_bits_minus_1 = 4;
};

// Writes the range-coded tile payload including exit_symbol().
bool write_gate_compound_tile(const GateTilePlan& plan,
                              const RefFrameMap& map,
                              const GateTileConfig& config,
                              std::vector<std::uint8_t>& payload,
                              std::string& error);

}  // namespace av2enc::video::multiref::gate
