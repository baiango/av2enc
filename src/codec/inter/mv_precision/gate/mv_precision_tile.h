// P2 tile writer for the MV-precision/AMVD decoder gate.
//
// The tile is a 16x16-leaf P frame written directly with §5.20.7 syntax so the
// AMVD and per-block precision tools can be exercised without editing the M2
// core.  Every leaf is NEWMV with skip_flag = 0 and an all-zero residual, so
// the decoder's reconstruction is exactly the inter prediction:
//
//   is_inter, skip_flag, single_mode(NEWMV), use_amvd,
//   read_drl_idx(=0), use_most_probable_precision/pb_mv_precision,
//   read_mv (AMVD or shell), mv_sign, interp_filter, all_zero x3.
//
// The M2 merge stack supplies the NEWMV predictor and the neighbour state is
// kept in the same `MiInfo` layout M2 uses.  Blocks alternate between:
//   0  AMVD (mv_joint + amvd_index, IsAdaptiveMvd on)
//   1  flex precision (use_most_probable_precision = 0, pb = 0 -> ONE_PEL)
//   2  frame precision (use_most_probable_precision = 1 -> QUARTER)

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/inter/mc/mc.h"
#include "codec/inter/mv_precision/mv_precision.h"

namespace av2enc::inter::mvprec::gate {

enum class LeafMode : std::uint8_t {
  kFramePrecision = 0,
  kAdaptiveMvd = 1,
  kFlexPrecision = 2,
};

struct TileConfig {
  bool enable_adaptive_mvd = true;
  bool enable_mvd_sign_derive = true;
  bool enable_flex_mvres = true;
  int frame_precision = me2::kMvPrecisionQuarterPel;
  // Desired per-block precision for kFlexPrecision leaves.
  int flex_target_precision = me2::kMvPrecisionOnePel;
  int leaf_size = 16;
};

struct LeafStats {
  int mi_row = 0;
  int mi_col = 0;
  int w4 = 0;
  int h4 = 0;
  LeafMode mode = LeafMode::kFramePrecision;
  int mv_precision = me2::kMvPrecisionEighthPel;
  bool use_amvd = false;
  int amvd_joint = 0;
  int amvd_index[2] = {0, 0};
  bool use_most_probable_precision = true;
  int pb_mv_precision = -1;
  int interp_filter = 0;
  int new_mv_context = 0;
  int amvd_context = 0;
  int precision_context = 0;
  int interp_context = 0;
  mc::Mv pred{};
  mc::Mv diff{};
  mc::Mv mv{};
  bool derived_sign = false;
};

struct TileOutput {
  std::vector<std::uint8_t> payload;
  std::vector<LeafStats> leaves;
  int amvd_blocks = 0;
  int flex_blocks = 0;
  int frame_precision_blocks = 0;
  int derived_sign_blocks = 0;
};

bool write_tile(int width, int height, const TileConfig& config,
                TileOutput* output, std::string* error);

}  // namespace av2enc::inter::mvprec::gate
