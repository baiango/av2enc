// Shared §7.13.3.25 fitted-BAWP helper for the M2 CPU reference and the CUDA
// orchestration.
//
// Both product paths must make the same on/off decision and produce the same
// adapted prediction, so the arithmetic lives in this one header and calls the
// shared component fit (`inter::adv::bawp_derive_block`) exactly once.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "codec/inter/advanced/bawp.h"

namespace av2enc::video {

// Adapts one luma prediction block in place.
//
//   cur/cur_stride  current reconstructed plane (above/left template)
//   ref/ref_stride  decoded reference plane (co-located template)
//   mi_cols/rows    coded MI extent the decoder uses for planeWidth/Height
//   mi_row/mi_col   block origin in MI units
//   x/y/w/h         block origin and size in luma samples
//   mv_y/mv_x       1/8-luma MV of the prediction
//
// Returns false when §7.13.3.25's conformance box or neighbour reads make the
// process illegal for this block; the caller then leaves the prediction
// untouched and does not signal use_bawp.
inline bool bawp_adapt_luma(const std::uint8_t* cur, int cur_stride,
                            const std::uint8_t* ref, int ref_stride,
                            int mi_cols, int mi_rows, int mi_row, int mi_col,
                            int x, int y, int w, int h, int mv_y, int mv_x,
                            std::vector<std::uint8_t>& pred,
                            inter::adv::BawpFit* fit_out) {
  if (cur == nullptr || ref == nullptr || w <= 0 || h <= 0 ||
      pred.size() != static_cast<std::size_t>(w) *
                         static_cast<std::size_t>(h)) {
    return false;
  }
  inter::adv::BawpJob job{};
  job.ctx.plane = 0;
  job.ctx.sub_x = 0;
  job.ctx.sub_y = 0;
  job.ctx.mi_col = mi_col;
  job.ctx.mi_row = mi_row;
  job.ctx.mi_cols = mi_cols;
  job.ctx.mi_rows = mi_rows;
  job.ctx.avail_u = mi_row > 0;
  job.ctx.avail_l = mi_col > 0;
  job.ctx.bit_depth = 8;
  job.rect = inter::adv::BlockRect{x, y, w, h};
  job.mv[0] = mv_y;
  job.mv[1] = mv_x;
  job.cur = inter::adv::PlaneView{cur, cur_stride};
  job.prev = inter::adv::PlaneView{ref, ref_stride};
  job.dst = inter::adv::MutablePlaneView{pred.data(), w};
  job.morph_pred = false;
  job.explicit_bawp = false;
  inter::adv::BawpFit fit{};
  if (!inter::adv::bawp_derive_block(job, &fit)) return false;
  for (std::size_t i = 0; i < pred.size(); ++i) {
    pred[i] = static_cast<std::uint8_t>(
        inter::adv::bawp_apply_sample(pred[i], fit.alpha, fit.beta, 8));
  }
  if (fit_out != nullptr) *fit_out = fit;
  return true;
}

}  // namespace av2enc::video
