// Self-contained P-frame tile writer for the DRL reorder decoder gate.
//
// The M2 core tile writer hard-codes single-reference mode info and the M2
// sequence header pins disable_drl_reorder = 1.  This agent does not own that
// code, so the gate writes its own P frame:
//
//   * 128x128 (or any 64-multiple) frame, 64x64 superblocks;
//   * uniform 4x4 PARTITION_SPLIT down from 64x64, so each superblock has 256
//     inter leaves and the DRL stack can reach numNearest >= 4 (the
//     DRL_REORDER_CONSTRAINT trigger);
//   * two reference slots (NumTotalRefs = 2) with reference_select = 0, so
//     every leaf codes one `single_ref` symbol;
//   * every leaf is NEARMV (single_mode = 0, skip_flag = 1, no residual, no
//     MVD), which is the minimal syntax that carries a DRL index;
//   * `drl_idx` is written by inter::drl::write_drl_index() from the stack
//     order produced by inter::drl::reorder_video_stack() (or, with
//     legacy_sort = true, by the historical full selection sort used only to
//     reproduce the desync).
//
// Symbol order (§5.20.7.6 / .9 / .10 / .12):
//   is_inter, skip_flag, single_ref, single_mode, drl_idx.
// `skip_flag = 1` means no residual and no further mode syntax.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/inter/drl_reorder/drl_reorder.h"
#include "codec/inter/mc/mc.h"

namespace av2enc::inter::drl::gate {

struct LeafPlan {
  // RefFrame label 0 or 1 (NumTotalRefs = 2).
  std::int8_t ref = 0;
  // NEWMV seed block: code single_mode = 2 and an MVD so the leaf can
  // introduce a new MV into the spatial stack.  Non-seed blocks are NEARMV
  // and take their MV from the stack.
  bool newmv = false;
  // Block extent in 4x4 units: 4x8 = {1,2}, 8x4 = {2,1}.
  int w4 = 1;
  int h4 = 1;
  // Target MV (NEWMV) or preferred stack MV (NEARMV, advisory).
  mc::Mv mv{};
};

struct LeafStats {
  int mi_row = 0;
  int mi_col = 0;
  std::int8_t ref = 0;
  bool signalable = false;
  int stack_count = 0;
  int num_nearest = 0;
  int pre_index = -1;
  int spec_post = -1;
  int legacy_post = -1;
  bool divergent = false;
  int signalled_index = -1;
  int new_mv_context = 0;
  mc::Mv chosen_mv{};
  mc::Mv pred_mv{};
  mc::Mv mvd{};
  std::string weights;
  std::string stack_mvs;
  int w4 = 1;
  int h4 = 1;
};

struct TileConfig {
  // Smallest leaf: 64, 32, 16, 8, or 4 (4 selects the 4x8/8x4 split).
  int leaf_size = 8;
  // 0 = square leaf; 1 = split 16x16 into 8x16 leaves (HORZ); 2 = 16x8 (VERT).
  int leaf_rect_16 = 0;
  int num_total_refs = 2;
  int max_drl_bits_minus_1 = 4;
  Mode mode = Mode::kConstraint;
  bool use_temporal_first = false;
  // Reproduce the historical encoder: full selection sort + its own index
  // mapping.  The decoder still applies the normative transform.
  bool legacy_sort = false;
  // Diagnostic mode: code GLOBALMV (no DRL, no MVD) to isolate the header
  // and reference syntax from the DRL syntax.
  bool force_global_mv = false;
  // Negative control: signal (index+1) % m for every block.
  bool corrupt_index = false;
  bool disable_cdf_update = true;
};

struct TileOutput {
  std::vector<std::uint8_t> payload;
  std::vector<LeafStats> leaves;
  int divergent_blocks = 0;
  int reordered_blocks = 0;
  int distinct_mvs = 0;
};

// `leaves` is raster order, one per 4x4 block.  On success `stats[i].chosen_mv`
// is the MV the encoder reconstructs with and the tile writer used for the
// MiInfo grid.
bool write_drl_tile(int width, int height, const std::vector<LeafPlan>& leaves,
                    const TileConfig& config, TileOutput* out,
                    std::string* error);

}  // namespace av2enc::inter::drl::gate
