// Test/harness helper: the M0 default (k32x32) partition plan as a
// sched::CodingBlock list.
//
// This is deliberately *not* part of the scheduler's public API: the production
// integration passes the real partition tree (sched::blocks_from), and this
// helper exists only so the plan unit test and the benchmark harness can build a
// plan without editing src/cuda/basic_pipeline.cpp (which owns PartitionPlanner).
//
// It mirrors PartitionPlanner for BasicPartitionMode::k32x32:
//   * a 64x64 superblock that fits inside the coded picture is split into four
//     32x32 luma blocks (16x16 chroma) in coding order TL, BL, TR, BR;
//   * an edge superblock is a single 64x64 luma block (32x32 chroma) using the
//     adjusted 32x32 transform (upscale 2);
//   * coefficients accumulate in coding order: luma, U, V per block.
//
// The harness compares the resulting reconstruction against the real pipeline
// (encode_still_picture_cuda) bit-exactly, which validates this helper.

#pragma once

#include "cuda/pipeline/dispatch/scheduler/scheduler.h"

#include <cstdint>
#include <vector>

namespace av2enc::cuda::sched::testutil {

inline FrameGeometry make_geometry(int width, int height) {
  FrameGeometry g;
  g.width = width;
  g.height = height;
  g.chroma_width = (width + 1) / 2;
  g.chroma_height = (height + 1) / 2;
  g.padded_width = ((width + 63) / 64) * 64;
  g.padded_height = ((height + 63) / 64) * 64;
  g.chroma_padded_width = (g.padded_width + 1) / 2;
  g.chroma_padded_height = (g.padded_height + 1) / 2;
  g.mi_cols = 2 * ((width + 7) >> 3);
  g.mi_rows = 2 * ((height + 7) >> 3);
  return g;
}

inline std::vector<CodingBlock> make_basic_k32x32_blocks(const FrameGeometry& g,
                                                      std::int64_t* total) {
  std::vector<CodingBlock> blocks;
  std::int64_t coeffs = 0;
  const auto add = [&](int mi_row, int mi_col, int luma_w, int luma_h,
                       int chroma_w, int chroma_h, int upscale) {
    CodingBlock b;
    b.mi_row = mi_row;
    b.mi_col = mi_col;
    b.luma_w = luma_w;
    b.luma_h = luma_h;
    b.chroma_w = chroma_w;
    b.chroma_h = chroma_h;
    b.upscale = upscale;
    b.luma_pels = (luma_w / upscale) * (luma_h / upscale);
    b.chroma_pels = chroma_w * chroma_h;
    b.coeff_offset = static_cast<int>(coeffs);
    b.real_coeffs = 1;
    coeffs += static_cast<std::int64_t>(b.luma_pels) + 2 * b.chroma_pels;
    blocks.push_back(b);
  };
  for (int r = 0; r < g.mi_rows; r += 16) {
    for (int c = 0; c < g.mi_cols; c += 16) {
      const bool edge = (r + 16 > g.mi_rows) || (c + 16 > g.mi_cols);
      if (edge) {
        add(r, c, 64, 64, 32, 32, 2);
      } else {
        add(r, c, 32, 32, 16, 16, 1);      // TL
        add(r + 8, c, 32, 32, 16, 16, 1);  // BL
        add(r, c + 8, 32, 32, 16, 16, 1);  // TR
        add(r + 8, c + 8, 32, 32, 16, 16, 1);  // BR
      }
    }
  }
  if (total != nullptr) {
    *total = coeffs;
  }
  return blocks;
}

}  // namespace av2enc::cuda::sched::testutil
