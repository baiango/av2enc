// Deblocking metadata grid for the M2 video path.
//
// The CPU M2 encoder and the CUDA M2 orchestration both build the §7.17
// deblocking metadata from the real per-leaf decisions (intra/inter, skip,
// transform size, qindex, plane geometry) instead of the conservative
// every-4x4-is-its-own-intra-block default in loop_filters.h.  This header is
// header-only so both translation units share exactly one implementation and
// no build-system change is needed.
//
// Only the entries the deblock core consumes are filled:
//   * coding-block origin/size (MiColBase/MiRowBase, Num_4x4_Blocks_Wide/High)
//   * SubPuSize = Max_Tx_Size_Rect[MiSize] (the sequence disables df_sub_pu, so
//     sub-PU edges are never filtered; the field still matches the decoder)
//   * DeblockingTxSizes[plane] and TxColBase/TxRowBase (TX_MODE_LARGEST: one
//     transform per plane at the block origin)
//   * Skips (inter skip only; intra leaves never skip)
//   * RegionTypes (frame-level INTRA_REGION / INTER_REGION)
//   * the effective qindex (no delta-Q/segmentation in the M2 header, so it is
//     the frame base_q_idx)
// Chroma grids are indexed on the luma MI grid and replicated over the luma
// extent of the owning block, matching the decoder's bottom-right mode-info
// rule.
#pragma once

#include "codec/filters/deblock/deblock.h"
#include "codec/partition/tree/partition.h"
#include "pipeline/still/still_tables.h"
#include "pipeline/video/video_syntax.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::video::post {

struct VideoDeblockGrid {
  int mi_cols = 0;
  int mi_rows = 0;
  std::vector<deblock::DeblockMi> mi[3];
  std::vector<std::uint16_t> tile_col_starts;
  std::vector<std::uint16_t> tile_row_starts;
  deblock::DeblockOptions options{};
};

// Max_Tx_Size_Rect[MiSize].  The table lives in still_tables.h, generated from
// the same §9.2 data this path already trusts; taking it directly avoids a link
// dependency on tx_partition.cpp in the standalone video gates.
inline int video_deblock_max_tx(int bsize) {
  if (bsize < 0 || bsize >= 29) return -1;
  return still::kMaxTxSizeRect[bsize];
}

// §5.20.6.3 find_tx_size(): the TX_SIZES_ALL entry whose Tx_Width/Tx_Height
// match (w, h) exactly, or -1.
inline int video_find_tx_size(int w, int h) {
  for (int i = 0; i < deblock::kTxSizesAll; ++i) {
    const auto tx = static_cast<deblock::TxSize>(i);
    if (deblock::tx_width(tx) == w && deblock::tx_height(tx) == h) return i;
  }
  return -1;
}

// §7.13.3.15 save subpu size process.  Invoked per plane from
// §7.13.3.14 predict_subblock() with the predicted subblock origin/size and the
// whole prediction unit size.  The whole-PU case (ordinary P/B leaves, which
// predict once with the coding-block MV) returns immediately, leaving the
// §6.19.3 `Max_Tx_Size_Rect[MiSize]` value in place; TIP/OPFL/refine-MV
// subblocks replace it with `find_tx_size(w, h)`.
inline bool video_save_subpu(int plane, int x, int y, int w, int h, int pu_w,
                             int pu_h, VideoDeblockGrid* grid,
                             std::string* error) {
  if (grid == nullptr || plane < 0 || plane > 2) {
    if (error != nullptr) *error = "save_subpu: invalid plane/grid";
    return false;
  }
  if (w == pu_w && h == pu_h) return true;  // normative early return
  const int sub_pu = video_find_tx_size(w, h);
  if (sub_pu < 0) {
    if (error != nullptr) {
      *error = "save_subpu: no TX size for " + std::to_string(w) + "x" +
               std::to_string(h);
    }
    return false;
  }
  const int sub_x = plane > 0 ? 1 : 0;
  const int sub_y = plane > 0 ? 1 : 0;
  const int luma_row = y >> (2 - sub_y);
  const int luma_col = x >> (2 - sub_x);
  const int rows = h >> (2 - sub_y);
  const int cols = w >> (2 - sub_x);
  for (int r = 0; r < rows; ++r) {
    const int row = luma_row + r;
    if (row < 0 || row >= grid->mi_rows) continue;
    for (int c = 0; c < cols; ++c) {
      const int col = luma_col + c;
      if (col < 0 || col >= grid->mi_cols) continue;
      deblock::DeblockMi& mi =
          grid->mi[plane][static_cast<std::size_t>(row) *
                              static_cast<std::size_t>(grid->mi_cols) +
                          static_cast<std::size_t>(col)];
      mi.sub_pu_col_base = static_cast<std::uint16_t>(luma_col);
      mi.sub_pu_row_base = static_cast<std::uint16_t>(luma_row);
      mi.sub_pu_size = static_cast<deblock::TxSize>(sub_pu);
    }
  }
  return true;
}

// Chroma residual block size for the 8-bit 4:2:0 video path.
inline partition::BlockSize video_deblock_chroma_size(int bsize) {
  partition::FrameGeometry geometry;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;
  return partition::chroma_plane_residual_size(
      geometry, static_cast<partition::BlockSize>(bsize));
}

// Fills the grid with the conservative per-4x4 default.  Every cell covered by
// a coding leaf is replaced by add_video_deblock_blocks().
inline void init_video_deblock_grid(int mi_cols, int mi_rows, int base_q_idx,
                                    bool frame_is_intra,
                                    VideoDeblockGrid* grid) {
  grid->mi_cols = mi_cols;
  grid->mi_rows = mi_rows;
  grid->tile_col_starts.clear();
  grid->tile_row_starts.clear();
  const std::size_t cells =
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows);
  const int q = std::min(std::max(base_q_idx, 0), 255);
  for (int plane = 0; plane < 3; ++plane) {
    grid->mi[plane].assign(cells, deblock::DeblockMi{});
    for (int r = 0; r < mi_rows; ++r) {
      for (int c = 0; c < mi_cols; ++c) {
        deblock::DeblockMi& mi =
            grid->mi[plane][static_cast<std::size_t>(r) *
                                static_cast<std::size_t>(mi_cols) +
                            static_cast<std::size_t>(c)];
        mi.block_col_base = static_cast<std::uint16_t>(c);
        mi.block_row_base = static_cast<std::uint16_t>(r);
        mi.block_w4 = 1;
        mi.block_h4 = 1;
        mi.sub_pu_size = deblock::TxSize::k4x4;
        mi.tx_size = deblock::TxSize::k4x4;
        mi.tx_col_base = static_cast<std::uint16_t>(c);
        mi.tx_row_base = static_cast<std::uint16_t>(r);
        mi.sub_pu_col_base = static_cast<std::uint16_t>(c);
        mi.sub_pu_row_base = static_cast<std::uint16_t>(r);
        mi.skip = 0;
        mi.region_intra = frame_is_intra ? 1 : 0;
        mi.lossless = 0;
        mi.qindex = static_cast<std::uint8_t>(q);
      }
    }
  }
  grid->options = deblock::DeblockOptions{};
  for (int i = 0; i < 4; ++i) grid->options.apply_deblocking_filter[i] = true;
  grid->options.frame_is_intra = frame_is_intra;
  grid->options.enable_sdp = false;
  grid->options.allow_df_sub_pu = false;
  grid->options.disable_loopfilters_across_tiles = false;
  grid->options.delta_q_u_ac = 0;
  grid->options.delta_q_v_ac = 0;
  grid->options.base_uv_ac_delta_q = 0;
  for (int i = 0; i < 4; ++i) grid->options.df_delta_q[i] = 0;
  grid->options.bit_depth = 8;
}

// A predicted subblock (§7.13.3.14 predict_subblock) for §7.13.3.15.
struct VideoSubPuRect {
  int x = 0;      // origin in samples of `plane` (-1 = luma/all planes)
  int y = 0;
  int w = 0;
  int h = 0;
  int pu_w = 0;   // whole prediction unit extent, same plane
  int pu_h = 0;
  // Plane this rect belongs to (-1 = apply to all three, the historical
  // whole-PU form).  OPFL/refine-MV split luma and chroma differently, so
  // each plane's predict_subblock() call contributes its own rect.
  int plane = -1;
};

// Applies the leaves of one coded tile (tile-local MI coordinates) to the
// frame-wide grid.  `blocks` is the complete partition list produced by the
// syntax collector, including internal nodes; only partition == kNone nodes are
// coding leaves.  `sub_pus` optionally lists the predicted subblocks of the
// frame (TIP/OPFL/refine-MV); ordinary P/B leaves predict the whole PU and
// take the §7.13.3.15 early return.
inline bool add_video_deblock_blocks(
    const std::vector<syntax::BlockSyntax>& blocks, int tile_mi_row,
    int tile_mi_col, int base_q_idx, bool frame_is_intra,
    VideoDeblockGrid* grid, std::string* error,
    const VideoSubPuRect* sub_pus = nullptr, int num_sub_pus = 0) {
  const int q = std::min(std::max(base_q_idx, 0), 255);
  for (const syntax::BlockSyntax& block : blocks) {
    if (block.partition !=
        static_cast<std::uint8_t>(partition::PartitionType::kNone)) {
      continue;
    }
    const int bsize = static_cast<int>(block.bsize);
    const int w4 = partition::num_4x4_blocks_wide(
        static_cast<partition::BlockSize>(bsize));
    const int h4 = partition::num_4x4_blocks_high(
        static_cast<partition::BlockSize>(bsize));
    if (w4 <= 0 || h4 <= 0) {
      if (error != nullptr) {
        *error = "video deblock: invalid leaf block size " +
                 std::to_string(bsize);
      }
      return false;
    }
    const int r = tile_mi_row + static_cast<int>(block.mi_row);
    const int c = tile_mi_col + static_cast<int>(block.mi_col);
    if (r < 0 || c < 0 || r >= grid->mi_rows || c >= grid->mi_cols) {
      if (error != nullptr) {
        *error = "video deblock: leaf outside the MI grid";
      }
      return false;
    }
    const int clamped_w = std::min(w4, grid->mi_cols - c);
    const int clamped_h = std::min(h4, grid->mi_rows - r);
    // §5.20.3.1: chroma metadata belongs to the owned ChromaMi block, not to
    // the luma leaf.  A sub-8 leaf group shares one chroma block at the
    // ChromaMi origin, so the U/V grids are replicated over the ChromaMi luma
    // extent (the decoder's bottom-right mode-info rule) and only the owning
    // leaf writes them.  Without kFlagHasChromaInfo the historical leaf-sized
    // chroma block is used.
    const bool chroma_info = (block.flags & syntax::kFlagHasChromaInfo) != 0;
    const bool owns_chroma = !chroma_info || block.has_chroma != 0;
    int chroma_r = r;
    int chroma_c = c;
    int chroma_w4 = clamped_w;
    int chroma_h4 = clamped_h;
    int chroma_bsize = bsize;
    if (chroma_info && owns_chroma) {
      chroma_r = tile_mi_row + static_cast<int>(block.chroma_mi_row);
      chroma_c = tile_mi_col + static_cast<int>(block.chroma_mi_col);
      chroma_bsize = static_cast<int>(block.chroma_bsize);
      if (!partition::is_valid_block_size(
              static_cast<partition::BlockSize>(chroma_bsize))) {
        if (error != nullptr) {
          *error = "video deblock: invalid ChromaMi size";
        }
        return false;
      }
      chroma_w4 = std::min(
          partition::num_4x4_blocks_wide(
              static_cast<partition::BlockSize>(chroma_bsize)),
          grid->mi_cols - chroma_c);
      chroma_h4 = std::min(
          partition::num_4x4_blocks_high(
              static_cast<partition::BlockSize>(chroma_bsize)),
          grid->mi_rows - chroma_r);
    }
    // §7.14.2 effective qindex for this leaf.  The producer fills it when the
    // frame codes segmentation / delta-Q; -1 keeps the frame base_q_idx.
    const int leaf_q =
        block.qindex >= 0 ? std::min(static_cast<int>(block.qindex), 255) : q;
    const bool inter_skip = block.is_inter != 0 && block.skip_flag != 0;
    const int luma_max_tx = video_deblock_max_tx(bsize);
    if (luma_max_tx < 0) {
      if (error != nullptr) {
        *error = "video deblock: no Max_Tx_Size_Rect for leaf size " +
                 std::to_string(bsize);
      }
      return false;
    }
    int tx_size[3];
    const partition::BlockSize chroma_tx_bsize =
        video_deblock_chroma_size(bsize);
    const int chroma_max_tx =
        partition::is_valid_block_size(chroma_tx_bsize)
            ? video_deblock_max_tx(static_cast<int>(chroma_tx_bsize))
            : luma_max_tx;
    if (!inter_skip) {
      for (int plane = 0; plane < 3; ++plane) {
        tx_size[plane] = static_cast<int>(block.tx_size[plane]);
      }
    } else {
      const partition::BlockSize chroma_residual =
          video_deblock_chroma_size(chroma_bsize);
      const int chroma_skip_max_tx =
          partition::is_valid_block_size(chroma_residual)
              ? video_deblock_max_tx(static_cast<int>(chroma_residual))
              : luma_max_tx;
      tx_size[0] = luma_max_tx;
      tx_size[1] = chroma_skip_max_tx;
      tx_size[2] = chroma_skip_max_tx;
    }
    // §7.13.3.15 / page 163: SubPuSize[plane] is Max_Tx_Size_Rect[MiSize] on
    // luma and Max_Tx_Size_Rect[ChromaMiSize] on chroma, independent of the
    // actual DeblockingTxSize.  allow_df_sub_pu = 0 leaves it inert.
    const int sub_pu_size[3] = {luma_max_tx, chroma_max_tx, chroma_max_tx};
    for (int plane = 0; plane < 3; ++plane) {
      if (tx_size[plane] < 0 || tx_size[plane] >= deblock::kTxSizesAll) {
        if (error != nullptr) {
          *error = "video deblock: invalid transform size on plane " +
                   std::to_string(plane);
        }
        return false;
      }
    }
    for (int plane = 0; plane < 3; ++plane) {
      // A non-owning sub-8 leaf codes no chroma; the owner filled the shared
      // ChromaMi rectangle (it follows the sibling in decode order).
      if (plane > 0 && !owns_chroma) continue;
      const int fill_r = plane == 0 ? r : chroma_r;
      const int fill_c = plane == 0 ? c : chroma_c;
      const int fill_w = plane == 0 ? clamped_w : chroma_w4;
      const int fill_h = plane == 0 ? clamped_h : chroma_h4;
      for (int yy = 0; yy < fill_h; ++yy) {
        for (int xx = 0; xx < fill_w; ++xx) {
          deblock::DeblockMi& mi =
              grid->mi[plane][static_cast<std::size_t>(fill_r + yy) *
                                  static_cast<std::size_t>(grid->mi_cols) +
                              static_cast<std::size_t>(fill_c + xx)];
          mi.block_col_base = static_cast<std::uint16_t>(fill_c);
          mi.block_row_base = static_cast<std::uint16_t>(fill_r);
          mi.block_w4 = static_cast<std::uint8_t>(fill_w);
          mi.block_h4 = static_cast<std::uint8_t>(fill_h);
          mi.sub_pu_size =
              static_cast<deblock::TxSize>(sub_pu_size[plane]);
          mi.tx_size = static_cast<deblock::TxSize>(tx_size[plane]);
          mi.tx_col_base = static_cast<std::uint16_t>(fill_c);
          mi.tx_row_base = static_cast<std::uint16_t>(fill_r);
          mi.sub_pu_col_base = static_cast<std::uint16_t>(fill_c);
          mi.sub_pu_row_base = static_cast<std::uint16_t>(fill_r);
          mi.skip = inter_skip ? 1 : 0;
          mi.region_intra = frame_is_intra ? 1 : 0;
          mi.lossless = 0;
          mi.qindex = static_cast<std::uint8_t>(leaf_q);
        }
      }
    }
    // §5.20.6.3 TX_MODE_SELECT: the decoder stores every transform part's own
    // origin in TxColBase/TxRowBase, so the §7.17 grid must carry each part and
    // filter the interior part edges too.  Chroma follows the historical
    // single-transform-per-plane rule (the luma search does not partition UV).
    if (block.tx_part_count > 1) {
      for (int i = 0; i < static_cast<int>(block.tx_part_count); ++i) {
        const int part_tx = static_cast<int>(block.tx_part_tx_size[i]);
        if (part_tx < 0 || part_tx >= deblock::kTxSizesAll) {
          if (error != nullptr) {
            *error = "video deblock: invalid TX-part size";
          }
          return false;
        }
        const int part_r = r + static_cast<int>(block.tx_part_mi_row[i]);
        const int part_c = c + static_cast<int>(block.tx_part_mi_col[i]);
        if (part_r < 0 || part_c < 0 || part_r >= grid->mi_rows ||
            part_c >= grid->mi_cols) {
          if (error != nullptr) {
            *error = "video deblock: TX part outside the MI grid";
          }
          return false;
        }
        const int pw4 =
            std::max(1, deblock::tx_width(static_cast<deblock::TxSize>(
                            part_tx)) /
                             4);
        const int ph4 =
            std::max(1, deblock::tx_height(static_cast<deblock::TxSize>(
                            part_tx)) /
                             4);
        for (int yy = 0; yy < ph4 && part_r + yy < grid->mi_rows; ++yy) {
          for (int xx = 0; xx < pw4 && part_c + xx < grid->mi_cols; ++xx) {
            deblock::DeblockMi& mi =
                grid->mi[0][static_cast<std::size_t>(part_r + yy) *
                                static_cast<std::size_t>(grid->mi_cols) +
                            static_cast<std::size_t>(part_c + xx)];
            mi.tx_size = static_cast<deblock::TxSize>(part_tx);
            mi.tx_col_base = static_cast<std::uint16_t>(part_c);
            mi.tx_row_base = static_cast<std::uint16_t>(part_r);
          }
        }
      }
    }
  }
  // §7.13.3.15: sub-PU prediction overrides SubPuSize/SubPuRowBase/
  // SubPuColBase for the predicted subblocks.  With no sub-PU list (ordinary
  // P/B leaves) every rect is the whole PU and the process early-returns.
  if (sub_pus != nullptr) {
    for (int i = 0; i < num_sub_pus; ++i) {
      for (int plane = 0; plane < 3; ++plane) {
        if (sub_pus[i].plane >= 0 && sub_pus[i].plane != plane) continue;
        if (!video_save_subpu(plane, sub_pus[i].x, sub_pus[i].y, sub_pus[i].w,
                              sub_pus[i].h, sub_pus[i].pu_w, sub_pus[i].pu_h,
                              grid, error)) {
          return false;
        }
      }
    }
  }
  return true;
}

// Self-check for §5.20.6.3/§7.13.3.15 used by the video gate.
inline bool video_save_subpu_selftest(std::string* error) {
  if (video_find_tx_size(8, 8) != static_cast<int>(deblock::TxSize::k8x8) ||
      video_find_tx_size(16, 8) != static_cast<int>(deblock::TxSize::k16x8) ||
      video_find_tx_size(8, 32) != static_cast<int>(deblock::TxSize::k8x32) ||
      video_find_tx_size(48, 8) >= 0) {
    if (error != nullptr) *error = "find_tx_size table mismatch";
    return false;
  }
  VideoDeblockGrid grid;
  init_video_deblock_grid(16, 16, 90, false, &grid);
  // Whole-PU prediction takes the early return (Max_Tx_Size_Rect stays).
  if (!video_save_subpu(0, 0, 0, 16, 16, 16, 16, &grid, error)) return false;
  for (int p = 0; p < 3; ++p) {
    for (int r = 0; r < 16; ++r) {
      for (int c = 0; c < 16; ++c) {
        const deblock::DeblockMi& mi =
            grid.mi[p][static_cast<std::size_t>(r) * 16U +
                       static_cast<std::size_t>(c)];
        if (mi.sub_pu_size != deblock::TxSize::k4x4) {
          if (error != nullptr) *error = "whole-PU early return changed sub-PU";
          return false;
        }
      }
    }
  }
  // An 8x8 luma sub-PU of a 16x16 PU: one 8x8 TX over the bottom-right
  // 4x4-block region (lumaRow = 4/4 = 1, lumaCol = 2).
  if (!video_save_subpu(0, 8, 4, 8, 8, 16, 16, &grid, error)) return false;
  for (int r = 1; r <= 2; ++r) {
    for (int c = 2; c <= 3; ++c) {
      const deblock::DeblockMi& mi =
          grid.mi[0][static_cast<std::size_t>(r) * 16U +
                     static_cast<std::size_t>(c)];
      if (mi.sub_pu_size != deblock::TxSize::k8x8 ||
          mi.sub_pu_row_base != 1 || mi.sub_pu_col_base != 2) {
        if (error != nullptr) *error = "luma sub-PU rect mismatch";
        return false;
      }
    }
  }
  // Chroma uses the subsampled origin and extent: an 8x8 chroma sub-PU of a
  // 16x16 PU at (4,0) covers luma rows 0..3 and columns 2..5.
  if (!video_save_subpu(1, 4, 0, 8, 8, 16, 16, &grid, error)) return false;
  for (int r = 0; r <= 3; ++r) {
    for (int c = 2; c <= 5; ++c) {
      const deblock::DeblockMi& mi =
          grid.mi[1][static_cast<std::size_t>(r) * 16U +
                     static_cast<std::size_t>(c)];
      if (mi.sub_pu_size != deblock::TxSize::k8x8 ||
          mi.sub_pu_row_base != 0 || mi.sub_pu_col_base != 2) {
        if (error != nullptr) *error = "chroma sub-PU rect mismatch";
        return false;
      }
    }
  }
  return true;
}

// Convenience wrapper for a single-tile frame.
inline bool build_video_deblock_grid(    const std::vector<syntax::BlockSyntax>& blocks, int mi_cols, int mi_rows,
    int base_q_idx, bool frame_is_intra, VideoDeblockGrid* grid,
    std::string* error,
    const VideoSubPuRect* sub_pus = nullptr, int num_sub_pus = 0) {
  init_video_deblock_grid(mi_cols, mi_rows, base_q_idx, frame_is_intra, grid);
  return add_video_deblock_blocks(blocks, 0, 0, base_q_idx, frame_is_intra,
                                  grid, error, sub_pus, num_sub_pus);
}

// Runs the CPU reference deblock over caller-owned MI-aligned planes.  The
// CUDA product path uses the CUDA kernel in src/cuda/codec/filters/deblock/
// instead; this helper is for --video and the CPU-emulation backend.
inline bool apply_video_deblock_host(const VideoDeblockGrid& grid,
                                     std::uint8_t* const plane[3],
                                     const int stride[3],
                                     std::string* error) {
  deblock::DeblockFrame frame;
  for (int p = 0; p < 3; ++p) {
    frame.planes[p].data = plane[p];
    frame.planes[p].stride = stride[p];
    frame.mi[p] = grid.mi[p].data();
  }
  frame.mi_cols = grid.mi_cols;
  frame.mi_rows = grid.mi_rows;
  frame.subsampling_x = 1;
  frame.subsampling_y = 1;
  if (!grid.tile_col_starts.empty()) {
    frame.tile_col_starts = grid.tile_col_starts.data();
    frame.num_tile_cols = static_cast<int>(grid.tile_col_starts.size());
  }
  if (!grid.tile_row_starts.empty()) {
    frame.tile_row_starts = grid.tile_row_starts.data();
    frame.num_tile_rows = static_cast<int>(grid.tile_row_starts.size());
  }
  const deblock::DeblockStatus status =
      deblock::deblock_frame(frame, grid.options);
  if (status != deblock::DeblockStatus::kOk) {
    if (error != nullptr) {
      *error = std::string("video deblock failed: ") +
               deblock::deblock_status_string(status);
    }
    return false;
  }
  return true;
}

}  // namespace av2enc::video::post
