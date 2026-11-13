// §7.17 deblocking metadata for the fixed M10 10-bit still block plan.
//
// The CPU product writer (`hbd::encode_still`) and the GPU block stage
// (`cuda::high_bitdepth::encode_high_bitdepth_gpu`) must feed the filter the
// exact same per-MI grid, so it is built in one place.  The plan codes one
// transform per block and never skips an intra block, so every cell carries
// the block origin, the coded transform size for that plane, region_intra = 1
// and the frame qindex.
#ifndef AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_DEBLOCK_H_
#define AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_DEBLOCK_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/filters/deblock/deblock.h"
#include "formats/format.h"
#include "pipeline/still/high_bitdepth/high_bitdepth_still.h"

namespace av2enc::hbd {

struct DeblockGrids {
  std::vector<deblock::DeblockMi> mi[3];
  int mi_cols = 0;
  int mi_rows = 0;
  int num_planes = 0;
  int sub_x = 0;
  int sub_y = 0;
};

// Builds one MI grid per coded plane for the block plan.  Returns false with
// `error` on an invalid plane count or grid.
inline bool build_deblock_grids(
    const av2enc::fmt::Format& format,
    const std::vector<av2enc::high_bitdepth::HighBitDepthBlock>& blocks,
    int base_q_idx, DeblockGrids* out, std::string* error) {
  if (out == nullptr) {
    if (error != nullptr) *error = "deblock grid: null output";
    return false;
  }
  const int mi_cols = static_cast<int>(av2enc::fmt::mi_cols(format.width));
  const int mi_rows = static_cast<int>(av2enc::fmt::mi_rows(format.height));
  const int num_planes = av2enc::fmt::num_planes(format.chroma);
  if (mi_cols <= 0 || mi_rows <= 0 || num_planes < 1 || num_planes > 3) {
    if (error != nullptr) *error = "deblock grid: invalid geometry";
    return false;
  }
  out->mi_cols = mi_cols;
  out->mi_rows = mi_rows;
  out->num_planes = num_planes;
  out->sub_x = av2enc::fmt::subsampling_x(format.chroma);
  out->sub_y = av2enc::fmt::subsampling_y(format.chroma);
  const std::size_t count =
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows);
  for (int p = 0; p < 3; ++p) {
    out->mi[p].assign(p < num_planes ? count : 0, deblock::DeblockMi{});
  }
  for (const av2enc::high_bitdepth::HighBitDepthBlock& block : blocks) {
    const int bx = static_cast<int>(block.mi_col);
    const int by = static_cast<int>(block.mi_row);
    const int w4 = block.luma_size / 4;
    if (w4 <= 0) {
      if (error != nullptr) *error = "deblock grid: empty block";
      return false;
    }
    // The fixed M10 plan keeps a 64x64 edge block even where the MI grid ends
    // early (a 640x360 bottom row has 10 of 16 rows).  The decoder's metadata
    // uses the clamped Num_4x4_Blocks_* the M1 writer records.
    const int clamped_w4 = std::min(w4, mi_cols - bx);
    const int clamped_h4 = std::min(w4, mi_rows - by);
    for (int p = 0; p < num_planes; ++p) {
      deblock::DeblockMi entry;
      entry.block_col_base = static_cast<std::uint16_t>(bx);
      entry.block_row_base = static_cast<std::uint16_t>(by);
      entry.block_w4 = static_cast<std::uint8_t>(clamped_w4);
      entry.block_h4 = static_cast<std::uint8_t>(clamped_h4);
      entry.tx_size = static_cast<deblock::TxSize>(block.tx_size[p]);
      entry.sub_pu_size = entry.tx_size;
      entry.tx_col_base = static_cast<std::uint16_t>(bx);
      entry.tx_row_base = static_cast<std::uint16_t>(by);
      entry.sub_pu_col_base = static_cast<std::uint16_t>(bx);
      entry.sub_pu_row_base = static_cast<std::uint16_t>(by);
      entry.skip = 0;
      entry.region_intra = 1;
      entry.lossless = 0;
      entry.qindex = static_cast<std::uint16_t>(base_q_idx);
      for (int yy = 0; yy < w4 && by + yy < mi_rows; ++yy) {
        for (int xx = 0; xx < w4 && bx + xx < mi_cols; ++xx) {
          out->mi[p][static_cast<std::size_t>(by + yy) *
                         static_cast<std::size_t>(mi_cols) +
                     static_cast<std::size_t>(bx + xx)] = entry;
        }
      }
    }
  }
  return true;
}

// Filters host reconstruction planes with the 10-bit core.
inline bool apply_deblock_16(
    const DeblockGrids& grids,
    av2enc::high_bitdepth::HighBitDepthPlane recon[3], std::string* error) {
  deblock::DeblockFrame16 frame{};
  for (int p = 0; p < grids.num_planes; ++p) {
    frame.planes[p].data = recon[p].samples.data();
    frame.planes[p].stride = recon[p].width;
    frame.mi[p] = grids.mi[p].data();
  }
  frame.mi_cols = grids.mi_cols;
  frame.mi_rows = grids.mi_rows;
  frame.subsampling_x = grids.sub_x;
  frame.subsampling_y = grids.sub_y;
  frame.num_planes = grids.num_planes;
  deblock::DeblockOptions options;
  for (int i = 0; i < 4; ++i) {
    options.apply_deblocking_filter[i] = true;
  }
  options.frame_is_intra = true;
  options.bit_depth = 10;
  if (deblock::deblock_frame16(frame, options) != deblock::DeblockStatus::kOk) {
    if (error != nullptr) *error = "hbd deblock failed";
    return false;
  }
  return true;
}

}  // namespace av2enc::hbd

#endif  // AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_DEBLOCK_H_
