// AV2 §7.16 Deblocking filter for TIP process — shared reference core.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §7.16 (pages 528–529), invoking the §7.17.3
// filter maximum width, §7.17.7.1 sample filtering and §7.17.7.2 filter choice
// processes.
//
// This is deliberately *not* §7.17.  §7.16 is a separate, much simpler filter:
//
//   * It is invoked only for TipFrameMode == TIP_FRAME_AS_OUTPUT, in place of
//     the regular §7.17 deblocking filter (see tip_deblock.h for the ordering).
//   * The edge lattice is the TIP block grid, not the coding/transform grid.
//     There is no mode-info metadata, no block/TX/sub-PU edge decision, no
//     skip, no lossless and no segmentation.
//   * One strength pair (qThr, side) is derived per plane from baseFilterLevel,
//     outside the traversal.  DfDeltaQ is not used.
//   * A single frame-level flag gates all three planes and both passes;
//     there are no per-plane apply flags.
//
// Geometry is 8-bit 4:2:0 planar.  The MI grid must be even in both directions
// (frame dimensions padded to a multiple of 8), exactly as for §7.17, so the
// 4:2:0 chroma grid and the 4-row/4-column sample groups are integral.
//
// All arithmetic and tables are shared with the §7.17 core so the two filters
// cannot drift: filter_max_width(), filter_choice(), sample_filtering(),
// get_q(), round2(), clip3(), kSideThresholds and friends come from
// encoder/filters/deblock/deblock_core.h.

#ifndef AV2ENC_ENCODER_FILTERS_TIP_DEBLOCK_TIP_DEBLOCK_CORE_H_
#define AV2ENC_ENCODER_FILTERS_TIP_DEBLOCK_TIP_DEBLOCK_CORE_H_

#include <cstddef>
#include <cstdint>

#include "codec/filters/deblock/deblock_core.h"
#include "codec/inter/temporal/temporal_types.h"

namespace av2enc::tip_deblock {

using inter::temporal::InterpFilter;

// QUANT_TABLE_BITS (§9.2 constant table, value 3).  This is the shift used by
// both §7.17.6 and §7.16 for qThr.
inline constexpr int kQuantTableBits = 3;

// MI_SIZE_LOG2 = 2.  §7.16 converts plane coordinates to luma MI coordinates
// with `lumaX >> MI_SIZE_LOG2` for the vertical tile-edge test.
inline constexpr int kMiSizeLog2 = 2;

// `tipSize` is the TIP build block size (§7.10.6).  Only these two values are
// reachable: BLOCK_8X8 and BLOCK_16X16.
enum class TipSize : std::int32_t {
  kBlock8x8 = 8,
  kBlock16x16 = 16,
};

inline constexpr int tip_width(TipSize size) {
  return static_cast<int>(size);
}
inline constexpr int tip_height(TipSize size) {
  return static_cast<int>(size);
}

// §7.16:
//   tipSize = (enable_tip_refinemv && TipInterpFilter == EIGHTTAP_SHARP)
//                 ? BLOCK_8X8 : BLOCK_16X16
inline TipSize select_tip_size(bool enable_tip_refinemv,
                               InterpFilter interp_filter) {
  return (enable_tip_refinemv &&
          interp_filter == InterpFilter::kEighttapSharp)
             ? TipSize::kBlock8x8
             : TipSize::kBlock16x16;
}

// Strength inputs.  For a TIP output frame, §5.18.2 derives base_q_idx,
// DeltaQUAc and DeltaQVAc from the closest past/future references (rounded
// averages) unless enable_tip_explicit_qp signals them in the TIP frame
// header.  BaseUVAcDeltaQ is a sequence-level quantizer value and is not
// averaged.
struct TipDeblockStrength {
  int base_q_idx = 0;
  int delta_q_u_ac = 0;
  int delta_q_v_ac = 0;
  int base_uv_ac_delta_q = 0;
  int bit_depth = 8;
};

// Per-plane §7.16 outputs, derived once per plane:
//   baseFilterLevel = base_q_idx (+ DeltaQUAc/DeltaQVAc + BaseUVAcDeltaQ)
//   qThr = Round2(get_q(baseFilterLevel, 0), QUANT_TABLE_BITS) >> 6
//   qInd = Clip3(0, MAX_SIDE_TABLE - 1, baseFilterLevel - 24*(BitDepth-8))
//   side = Max(Side_Thresholds[qInd] + (1 << (12 - BitDepth)), 0)
//              >> (13 - BitDepth)
struct TipDeblockPlaneInfo {
  int base_filter_level = 0;
  int q_thr = 0;
  int side = 0;
};

inline TipDeblockPlaneInfo derive_plane_info(
    int plane, const TipDeblockStrength& strength) {
  int base_filter_level = strength.base_q_idx;
  if (plane == 1) {
    base_filter_level += strength.delta_q_u_ac + strength.base_uv_ac_delta_q;
  } else if (plane == 2) {
    base_filter_level += strength.delta_q_v_ac + strength.base_uv_ac_delta_q;
  }

  // Note: no q_clamped() special case and no DfDeltaQ term.  With
  // baseFilterLevel == 0, get_q(0, 0) returns Ac_Qlookup[0] (64), so qThr is
  // 1; `side` becomes 0 and filter_choice() returns 0 anyway.
  const int q_thr =
      deblock::round2(deblock::get_q(base_filter_level, 0),
                      kQuantTableBits) >>
      6;
  const int q_ind = deblock::clip3(
      0, deblock::kMaxSideTable - 1,
      base_filter_level - 24 * (strength.bit_depth - 8));
  const int side =
      deblock::imax(deblock::kSideThresholds[q_ind] +
                        (1 << (12 - strength.bit_depth)),
                    0) >>
      (13 - strength.bit_depth);

  TipDeblockPlaneInfo info;
  info.base_filter_level = base_filter_level;
  info.q_thr = q_thr;
  info.side = side;
  return info;
}

// Frame description used by the core.  The caller owns all memory.
struct TipDeblockCoreFrame {
  std::uint8_t* plane[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;
  // Tile starts in luma MI units.  A count of zero means one implicit tile.
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

// Diagnostic record for one §7.16 edge.  Populated even when the edge is not
// filtered so integration code and tests can compare decisions.
struct TipDeblockEdgeInfo {
  int on_screen = 0;
  int tile_vert_edge = 0;
  int tile_horz_edge = 0;
  // sbEdge passed to filter_maximum_width(): the vertical tile edge for the
  // vertical pass, the horizontal 64x64-grid edge for the horizontal pass.
  int sb_edge = 0;
  // 1 when disable_loopfilters_across_tiles suppressed this tile edge.
  int tile_skipped = 0;
  int filter_size = 0;
  int max_width_pos = 0;
  int max_width_neg = 0;
  int q_thr = 0;
  int side = 0;
  int width = 0;
  int apply_filter = 0;
};

// §7.16 is_vert_tile_edge(x, subX):
//   col = (x << subX) >> MI_SIZE_LOG2
//   returns 1 when col is one of MiColStarts.
inline bool is_vert_tile_edge(const TipDeblockCoreFrame& frame, int x,
                              int sub_x) {
  const int col = (x << sub_x) >> kMiSizeLog2;
  return deblock::is_tile_start(frame.tile_col_starts, frame.num_tile_cols,
                                col);
}

// §7.16 is_horz_tile_edge(y, subY):
//   row = (y << subY) >> MI_SIZE_LOG2
inline bool is_horz_tile_edge(const TipDeblockCoreFrame& frame, int y,
                              int sub_y) {
  const int row = (y << sub_y) >> kMiSizeLog2;
  return deblock::is_tile_start(frame.tile_row_starts, frame.num_tile_rows,
                                row);
}

// One §7.16 edge.
//
// `x`/`y` are plane coordinates of the edge (the spec's x/y inside the plane
// loops).  `pass == 0` is the vertical-boundary pass (dx = 1, dy = 0);
// `pass == 1` is the horizontal-boundary pass (dx = 0, dy = 1).
//
// The template parameter selects the modifying or inspection path.  The
// decision record is identical for both.
template <bool Apply>
TipDeblockEdgeInfo tip_edge_core(const TipDeblockCoreFrame& frame,
                                 TipSize tip_size,
                                 const TipDeblockStrength& strength, int plane,
                                 int pass, int x, int y,
                                 bool disable_loopfilters_across_tiles) {
  TipDeblockEdgeInfo info{};
  if (plane < 0 || plane > 2 || pass < 0 || pass > 1) {
    return info;
  }

  const int sub_x = (plane != 0) ? frame.sub_x : 0;
  const int sub_y = (plane != 0) ? frame.sub_y : 0;
  const int dx = (pass == 0) ? 1 : 0;
  const int dy = (pass == 0) ? 0 : 1;
  const int sw = tip_width(tip_size) >> sub_x;
  const int sh = tip_height(tip_size) >> sub_y;
  const int w = (frame.mi_cols * deblock::kMiSize) >> sub_x;
  const int h = (frame.mi_rows * deblock::kMiSize) >> sub_y;

  // Validate that (x, y) is one of the positions the §7.16 loops visit.
  // The vertical pass visits x = k*sw (k > 0) and y = 4*m, operating on the
  // four rows y..y+3.  The horizontal pass visits y = k*sh (k > 0) and
  // x = 4*m, operating on the four columns x..x+3.
  if (pass == 0) {
    if (x <= 0 || x >= w || (x % sw) != 0 || y < 0 || (y % 4) != 0 ||
        y + 3 >= h) {
      return info;
    }
  } else {
    if (y <= 0 || y >= h || (y % sh) != 0 || x < 0 || (x % 4) != 0 ||
        x + 3 >= w) {
      return info;
    }
  }
  info.on_screen = 1;

  const TipDeblockPlaneInfo plane_info = derive_plane_info(plane, strength);
  info.q_thr = plane_info.q_thr;
  info.side = plane_info.side;

  if (pass == 0) {
    info.tile_vert_edge = is_vert_tile_edge(frame, x, sub_x) ? 1 : 0;
    // The vertical pass passes vertTileEdge as sbEdge (§7.16 page 529).
    info.sb_edge = info.tile_vert_edge;
    info.filter_size = sw;
    info.tile_skipped =
        (disable_loopfilters_across_tiles && info.tile_vert_edge) ? 1 : 0;
  } else {
    info.tile_horz_edge = is_horz_tile_edge(frame, y, sub_y) ? 1 : 0;
    // The horizontal pass passes horz64Edge as sbEdge; a horizontal *tile*
    // edge only gates the early return and does not widen the clamp.
    info.sb_edge = ((((y << sub_y) % 64) == 0)) ? 1 : 0;
    info.filter_size = sh;
    info.tile_skipped =
        (disable_loopfilters_across_tiles && info.tile_horz_edge) ? 1 : 0;
  }

  // filterSize is sw/sh, unclipped: §7.16 has no screen-edge clip of
  // filterSize (that clip belongs to the §7.17.2 edge process).
  deblock::filter_max_width(plane, info.filter_size, info.sb_edge,
                            &info.max_width_pos, &info.max_width_neg);

  if (info.tile_skipped) {
    return info;
  }

  std::uint8_t* data = frame.plane[plane];
  const int stride = frame.stride[plane];
  // §7.17.7.2 with count = MI_SIZE and the plane-coordinate (x, y).
  info.width = deblock::filter_choice(
      data, stride, x, y, info.q_thr, info.side, dx, dy, info.max_width_neg,
      info.max_width_pos);
  if (info.width <= 0) {
    return info;
  }
  info.apply_filter = 1;

  if (Apply) {
    const int neg = deblock::imin(info.width, info.max_width_neg);
    const int pos = deblock::imin(info.width, info.max_width_pos);
    // §7.16 invokes sample_filtering with prevLossless = currLossless = 0.
    for (int i = 0; i < deblock::kMiSize; ++i) {
      // The spec calls sample_filtering(x, y + i) for the vertical pass and
      // sample_filtering(x + i, y) for the horizontal pass.
      const int sx = x + dy * i;
      const int sy = y + dx * i;
      deblock::sample_filtering(data, stride, sx, sy, info.q_thr, dx, dy, neg,
                                pos, 0, 0);
    }
  }
  return info;
}

// One plane/pass traversal, exactly the §7.16 loop nest.
inline void tip_deblock_plane_pass_core(
    const TipDeblockCoreFrame& frame, TipSize tip_size,
    const TipDeblockStrength& strength, int plane, int pass,
    bool disable_loopfilters_across_tiles) {
  if (plane < 0 || plane > 2 || pass < 0 || pass > 1) {
    return;
  }
  const int sub_x = (plane != 0) ? frame.sub_x : 0;
  const int sub_y = (plane != 0) ? frame.sub_y : 0;
  const int sw = tip_width(tip_size) >> sub_x;
  const int sh = tip_height(tip_size) >> sub_y;
  const int w = (frame.mi_cols * deblock::kMiSize) >> sub_x;
  const int h = (frame.mi_rows * deblock::kMiSize) >> sub_y;

  if (pass == 0) {
    // Literal §7.16: y += 4 (not 4 << subY), x += sw.
    for (int y = 0; y < h; y += 4) {
      for (int x = 0; x < w; x += sw) {
        if (x > 0) {
          (void)tip_edge_core<true>(frame, tip_size, strength, plane, pass, x,
                                    y, disable_loopfilters_across_tiles);
        }
      }
    }
  } else {
    // Literal §7.16: x += 4 (not 4 << subX), y += sh.
    for (int x = 0; x < w; x += 4) {
      for (int y = 0; y < h; y += sh) {
        if (y > 0) {
          (void)tip_edge_core<true>(frame, tip_size, strength, plane, pass, x,
                                    y, disable_loopfilters_across_tiles);
        }
      }
    }
  }
}

// Full §7.16 traversal: plane outer, then vertical boundaries, then horizontal
// boundaries.  There is one strength pair per plane and a single frame-level
// enable flag (checked by the caller).
inline void tip_deblock_frame_core(const TipDeblockCoreFrame& frame,
                                   TipSize tip_size,
                                   const TipDeblockStrength& strength,
                                   bool disable_loopfilters_across_tiles) {
  for (int plane = 0; plane < 3; ++plane) {
    tip_deblock_plane_pass_core(frame, tip_size, strength, plane, 0,
                                disable_loopfilters_across_tiles);
    tip_deblock_plane_pass_core(frame, tip_size, strength, plane, 1,
                                disable_loopfilters_across_tiles);
  }
}

}  // namespace av2enc::tip_deblock

#endif  // AV2ENC_ENCODER_FILTERS_TIP_DEBLOCK_TIP_DEBLOCK_CORE_H_
