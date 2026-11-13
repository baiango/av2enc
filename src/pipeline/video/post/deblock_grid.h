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

// Applies the leaves of one coded tile (tile-local MI coordinates) to the
// frame-wide grid.  `blocks` is the complete partition list produced by the
// syntax collector, including internal nodes; only partition == kNone nodes are
// coding leaves.
inline bool add_video_deblock_blocks(
    const std::vector<syntax::BlockSyntax>& blocks, int tile_mi_row,
    int tile_mi_col, int base_q_idx, bool frame_is_intra,
    VideoDeblockGrid* grid, std::string* error) {
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
    if (!inter_skip) {
      for (int plane = 0; plane < 3; ++plane) {
        tx_size[plane] = static_cast<int>(block.tx_size[plane]);
      }
    } else {
      const partition::BlockSize chroma_bsize =
          video_deblock_chroma_size(bsize);
      const int chroma_max_tx =
          partition::is_valid_block_size(chroma_bsize)
              ? video_deblock_max_tx(static_cast<int>(chroma_bsize))
              : luma_max_tx;
      tx_size[0] = luma_max_tx;
      tx_size[1] = chroma_max_tx;
      tx_size[2] = chroma_max_tx;
    }
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
      for (int yy = 0; yy < clamped_h; ++yy) {
        for (int xx = 0; xx < clamped_w; ++xx) {
          deblock::DeblockMi& mi =
              grid->mi[plane][static_cast<std::size_t>(r + yy) *
                                  static_cast<std::size_t>(grid->mi_cols) +
                              static_cast<std::size_t>(c + xx)];
          mi.block_col_base = static_cast<std::uint16_t>(c);
          mi.block_row_base = static_cast<std::uint16_t>(r);
          mi.block_w4 = static_cast<std::uint8_t>(clamped_w);
          mi.block_h4 = static_cast<std::uint8_t>(clamped_h);
          mi.sub_pu_size =
              static_cast<deblock::TxSize>(luma_max_tx);
          mi.tx_size = static_cast<deblock::TxSize>(tx_size[plane]);
          mi.tx_col_base = static_cast<std::uint16_t>(c);
          mi.tx_row_base = static_cast<std::uint16_t>(r);
          mi.sub_pu_col_base = static_cast<std::uint16_t>(c);
          mi.sub_pu_row_base = static_cast<std::uint16_t>(r);
          mi.skip = inter_skip ? 1 : 0;
          mi.region_intra = frame_is_intra ? 1 : 0;
          mi.lossless = 0;
          mi.qindex = static_cast<std::uint8_t>(q);
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
  return true;
}

// Convenience wrapper for a single-tile frame.
inline bool build_video_deblock_grid(
    const std::vector<syntax::BlockSyntax>& blocks, int mi_cols, int mi_rows,
    int base_q_idx, bool frame_is_intra, VideoDeblockGrid* grid,
    std::string* error) {
  init_video_deblock_grid(mi_cols, mi_rows, base_q_idx, frame_is_intra, grid);
  return add_video_deblock_blocks(blocks, 0, 0, base_q_idx, frame_is_intra,
                                  grid, error);
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
