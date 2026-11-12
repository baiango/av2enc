#ifndef AV2ENC_INTER_ADVANCED_BAWP_COMMON_H_
#define AV2ENC_INTER_ADVANCED_BAWP_COMMON_H_

// Block adaptive weighted prediction — AV2 v1.0.0 §7.13.3.25 (and the
// morphological prediction wrapper §7.13.3.26, which is BAWP with
// `morphPred = 1`).
//
// BAWP refits `pred' = (alpha * pred + beta) >> 8` to the reconstructed
// above/left neighbours of the current block, using the co-located reference
// samples as the regressor. The fit is a least-squares slope through the
// template samples with a fixed 8-bit precision.
//
// This is one transcription shared by the CPU reference and the CUDA kernel.

#include "advanced_common.h"

namespace av2enc::inter::adv {

struct BawpContext {
  int plane = 0;
  int sub_x = 0;
  int sub_y = 0;
  int mi_col = 0;
  int mi_row = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  bool avail_u = false;
  bool avail_l = false;
  int bit_depth = 8;
};

struct BawpFit {
  int alpha = 256;  // Q8
  int beta = -128;
  int count = 0;
};

// Number of template samples for the fit. Split out so tests can check the
// AvailU/AvailL matrix without running the whole process.
struct BawpTemplatePlan {
  int width = 0;
  int height = 0;
  int num_up = 0;
  int num_left = 0;
};

AV2ENC_ADV_HD inline BawpTemplatePlan bawp_template_plan(const BawpContext& ctx, int bw, int bh) {
  BawpTemplatePlan plan{};
  const int limit = ctx.plane == 0 ? 16 : 8;
  int bw2 = bw < limit ? bw : limit;
  int bh2 = bh < limit ? bh : limit;
  plan.width = bw2 == 12 ? 8 : bw2;
  plan.height = bh2 == 12 ? 8 : bh2;
  if (ctx.avail_u && ctx.avail_l) {
    if (plan.width == 16 && plan.height == 16) {
      plan.num_up = 16;
      plan.num_left = 16;
    } else if (plan.width > 4 && plan.height > 4) {
      plan.num_up = 8;
      plan.num_left = 8;
    } else if (plan.width < 16 && plan.height < 16) {
      plan.num_up = 4;
      plan.num_left = 4;
    } else if (plan.width == 16) {
      plan.num_up = 16;
    } else {
      plan.num_left = 16;
    }
  } else if (ctx.avail_u) {
    plan.num_up = plan.width;
  } else if (ctx.avail_l) {
    plan.num_left = plan.height;
  }
  return plan;
}

// Gathers the template and derives alpha/beta.
//
//   `cur`      — current frame plane; above/left reconstructed samples are read
//                at (x, y-1) and (x-1, y).
//   `prev`     — reference plane (or `cur` itself for morphPred).
//   `ref_x``ref_y` — co-located reference sample of the block origin.
//   `bawp_alpha` — the luma alpha, used for chroma. Ignored for plane 0.
//
// `explicit_bawp` / `explicit_bawp_scale` / `list_index` / `first_ref_dist`
// implement the luma fast path in §7.13.3.25. Pass explicit_bawp = false for
// the fitted path.
AV2ENC_ADV_HD inline BawpFit bawp_fit(const BawpContext& ctx, const PlaneView& cur,
                                      const PlaneView& prev, int x, int y, int w, int h, int ref_x,
                                      int ref_y, int bawp_alpha, bool explicit_bawp,
                                      bool explicit_bawp_scale, int list_index,
                                      int first_ref_dist) {
  constexpr int shift = 8;
  BawpFit fit{};
  fit.alpha = 1 << shift;
  fit.beta = -(1 << (shift - 1));

  const int bw = w;
  const int bh = h;
  const BawpTemplatePlan plan = bawp_template_plan(ctx, bw, bh);
  int sum_x = 0;
  int sum_y = 0;
  int sum_xx = 0;
  int sum_xy = 0;
  int count = 0;

  if (plan.num_up > 0) {
    const int step = plan.width / plan.num_up;
    for (int i = step >> 1; i < plan.width; i += step) {
      const int recon = cur.data[(y - 1) * cur.stride + x + i];
      const int ref = prev.data[(ref_y - 1) * prev.stride + ref_x + i];
      sum_x += ref;
      sum_y += recon;
      sum_xy += ref * recon;
      sum_xx += ref * ref;
    }
    count += plan.num_up;
  }
  if (plan.num_left > 0) {
    const int step = plan.height / plan.num_left;
    for (int i = step >> 1; i < plan.height; i += step) {
      const int recon = cur.data[(y + i) * cur.stride + x - 1];
      const int ref = prev.data[(ref_y + i) * prev.stride + ref_x - 1];
      sum_x += ref;
      sum_y += recon;
      sum_xy += ref * recon;
      sum_xx += ref * ref;
    }
    count += plan.num_left;
  }

  if (ctx.plane > 0) {
    fit.alpha = bawp_alpha;
    if (count == 0) {
      fit.alpha = 1 << shift;
    }
  } else if (explicit_bawp) {
    fit.alpha = bawp_explicit_alpha(list_index, first_ref_dist, explicit_bawp_scale);
  } else if (count > 0) {
    const int nor = sum_xy - sum_x * sum_y / count;
    const int der = sum_xx - sum_x * sum_x / count;
    if (der != 0 && nor != 0) {
      fit.alpha = resolve_division(nor, der, shift);
      if (fit.alpha == 0) {
        fit.alpha = 1 << shift;
      }
    } else {
      fit.alpha = 1 << shift;
    }
  }

  if (count > 0) {
    fit.beta = ((sum_y << shift) - sum_x * fit.alpha) / count;
  }
  fit.count = count;
  return fit;
}

// Applies the fit to all w*h samples of the block (§7.13.3.25 final loop; note
// it covers the full block, not just the in-frame part).
AV2ENC_ADV_HD inline void bawp_apply(const BawpFit& fit, const MutablePlaneView& dst, int x, int y,
                                     int w, int h) {
  for (int i = 0; i < h; ++i) {
    std::uint8_t* row = dst.data + static_cast<std::ptrdiff_t>(y + i) * dst.stride;
    for (int j = 0; j < w; ++j) {
      row[x + j] = static_cast<std::uint8_t>(bawp_apply_sample(row[x + j], fit.alpha, fit.beta, 8));
    }
  }
}

}  // namespace av2enc::inter::adv

#endif  // AV2ENC_INTER_ADVANCED_BAWP_COMMON_H_
