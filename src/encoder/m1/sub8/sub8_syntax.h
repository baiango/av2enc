// Minimal all-DC / zero-residual tile writer+reader for sub-8 coding blocks.
//
// This is the executable oracle for the rules in sub8.h.  It writes a complete
// §8.2 tile payload (partition symbols + intra mode info + all_zero flags) for
// an intra frame whose leaves are all DC_PRED with EOB 0.  That subset is
// enough to exercise every syntax element that changes when a coding block is
// 4xN / Nx4:
//
//   * which leaf codes the chroma mode info and chroma residuals
//     (HasChroma / ChromaMi* inheritance, §5.20.3.1);
//   * the §8.3.2 all_zero context of the U and V planes (V depends on EobU
//     and on whether the chroma block is larger than its transform);
//   * the y_mode_index / is_cfl neighbour contexts at the chroma block origin;
//   * the do_split / rect_type contexts after a sub-8 leaf updated
//     LeftMiSizes / AboveMiSizes.
//
// The reader is an independent implementation of the same subset: it derives
// the tree from the symbols instead of consuming the plant, and it re-derives
// all contexts.  It is used for round-trip tests and as a safety net in the
// gate.  It deliberately refuses streams it cannot verify (non-DC mode,
// non-zero EOB, 128x128 superblocks, SPLIT).
#pragma once

#include "encoder/m1/sub8/sub8.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::m1::sub8 {

struct SyntaxOptions {
  std::uint8_t base_q_idx = 90;
  bool enable_cfl = true;          // sequence-header enable_cfl_intra
  bool disable_cdf_update = true;  // frame-header disable_cdf_update
};

// Independent decoder result for one leaf.
struct DecodedLeaf {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  bool has_chroma = false;
  ChromaInfo chroma;
  int y_mode = 0;  // AV2 YMode; 0 == DC_PRED
  int uv_mode = 0; // AV2 UV mode; 0 == DC_PRED
  bool is_cfl = false;
  bool luma_all_zero = false;
  bool u_all_zero = false;
  bool v_all_zero = false;
};

// §9.3 CDFs plus the two coefficient-skip arrays selected by §8.3.1 qctx.
struct CdfState {
  std::uint16_t do_split[2][64][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t uv_mode[2][9];
  std::uint16_t is_cfl[3][3];
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
};

void init_cdfs(CdfState& cdfs, const SyntaxOptions& options);

// Writes the tile payload for all superblocks (raster order).  Each Node must
// be a root built by build_superblock().  Leaves are coded DC_PRED with EOB 0.
bool write_tile(const FrameGeometry& geometry,
                std::span<const Node> superblocks,
                const SyntaxOptions& options, std::vector<std::uint8_t>& tile,
                std::string& error);

// Parses a tile payload and returns leaves in decode order.  Verifies that
// every leaf is DC_PRED with all_zero = 1 in all three planes.
bool read_tile(const FrameGeometry& geometry,
               std::span<const std::uint8_t> tile,
               const SyntaxOptions& options,
               std::vector<DecodedLeaf>& leaves, std::string& error);

}  // namespace av2enc::m1::sub8
