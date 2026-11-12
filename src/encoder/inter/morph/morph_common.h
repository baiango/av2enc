#ifndef AV2ENC_INTER_MORPH_MORPH_COMMON_H_
#define AV2ENC_INTER_MORPH_MORPH_COMMON_H_

// Morphological prediction reference — AV2 v1.0.0 §7.13.3.26.
//
// §7.13.3.26 has no body of its own: it invokes the §7.13.3.25 block adaptive
// weighted prediction process with
//
//     plane = 0, x, y, w, h, mv, morphPred = 1
//
// so this header transcribes §7.13.3.25 with the morphPred branch selected:
//
//   * prevFrame = CurrFrame (the IBC reference block is in the current frame);
//   * the template regressor is the already-reconstructed source block at
//     refX/refY derived from the IntraBC vector;
//   * explicit_bawp and the BawpAlpha carry-over are disabled for morphPred;
//   * the final adjustment covers the full w*h block.
//
// The process is luma only. The caller must have written the IBC prediction
// into `frame` before calling morph_predict(), because the spec adapts
// CurrFrame[0] in place.
//
// Conformance conditions enforced by morph_reference_geometry():
//   refX >= 1, refY >= 1,
//   refX + bw <= planeWidth, refY + bh <= planeHeight.
// The separate §6.19.5.4 requirement (is_offset_mv_valid(-1,-1) == 1 when
// morph_pred == 1) is a syntax/eligibility condition; see morph_syntax.h.

#include <cstddef>
#include <cstdint>

namespace av2enc::inter::morph {

inline constexpr int kMiSize = 4;
inline constexpr int kDivLutBits = 7;
inline constexpr int kDivLutPrecBits = 9;

// §4.8 Clip3/Clip1/FloorLog2/Round2, integer forms.
inline constexpr int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}
inline constexpr int clip1(int value, int bit_depth) {
  return clip3(0, (1 << bit_depth) - 1, value);
}
inline constexpr int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}
inline constexpr int round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

// §7.13.3.25 helper: 1/8-pel to integer luma samples. The spec uses a bias of
// 3 plus an extra 1 for non-negative values (round half up).
inline constexpr int to_fullmv(int mv) {
  return (mv + 3 + ((mv >= 0) ? 1 : 0)) >> 3;
}

// §7.13.3.22: Div_Lut entry. The PDF table is the closed form below; the
// advanced-inter component already pins the same formula.
inline constexpr int div_lut_entry(int f) {
  const int den = 128 + f;
  return (65536 + den / 2) / den;
}

// §7.13.3.25 resolve_division(N, D, shift) == floor((N << shift) / D) with the
// spec's Div_Lut approximation. D is always > 0 at the BAWP call site.
inline int resolve_division(int n_value, int d_value, int shift) {
  if (d_value <= 0) {
    return 1 << shift;
  }
  const bool sign_n = n_value < 0;
  int n = sign_n ? -n_value : n_value;
  const int shift_n = floor_log2(n);
  const int shift_d = floor_log2(d_value);
  const int e_d = d_value - (1 << shift_d);
  const int f_d = (shift_d > kDivLutBits)
                      ? round2(e_d, shift_d - kDivLutBits)
                      : (e_d << (kDivLutBits - shift_d));
  const int f_n = (shift_n > kDivLutBits)
                      ? round2(n, shift_n - kDivLutBits)
                      : (n << (kDivLutBits - shift_n));
  const int shift_add = shift_d - shift_n - shift;
  int ret = 0;
  if (shift_add <= 1) {
    const int shift0 = kDivLutPrecBits + kDivLutBits + shift_add;
    if (shift0 >= 0) {
      ret = (div_lut_entry(f_d) * f_n) >> shift0;
    } else {
      ret = (2 << shift) - 1;
    }
  }
  ret = ret < ((2 << shift) - 1) ? ret : ((2 << shift) - 1);
  if (sign_n) {
    ret = -ret;
  }
  return ret;
}

// Inputs mirror the current-block state used by §7.13.3.25. MiCol/MiRow are
// the top-left 4x4 coordinates of the block; MiCols/MiRows size the luma plane.
struct MorphContext {
  int mi_col = 0;
  int mi_row = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  bool avail_u = false;
  bool avail_l = false;
  int bit_depth = 8;
};

struct MorphRect {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

struct MorphPlane {
  std::uint8_t* data = nullptr;
  int stride = 0;
};

// Template sampling plan from §7.13.3.25. `width`/`height` are the clamped
// template extents (after the 12 -> 8 rule); num_up/num_left are the sample
// counts actually accumulated on each side.
struct MorphTemplatePlan {
  int width = 0;
  int height = 0;
  int num_up = 0;
  int num_left = 0;
  int up_step = 0;
  int left_step = 0;
};

struct MorphFit {
  int alpha = 256;  // Q8
  int beta = -128;
  int count = 0;
  int ref_x = 0;
  int ref_y = 0;
  int bw = 0;
  int bh = 0;
  MorphTemplatePlan plan{};
};

// §7.13.3.25 geometry: refX/refY from the block's MI origin and the 1/8-pel
// MV, bw/bh clipped to the luma plane. Returns false for a degenerate block.
inline bool morph_reference_geometry(const MorphContext& ctx, const MorphRect& rect,
                                     int mv_row, int mv_col, int* ref_x, int* ref_y,
                                     int* bw, int* bh) {
  if (ctx.bit_depth != 8 || ctx.mi_cols <= 0 || ctx.mi_rows <= 0 || rect.w <= 0 ||
      rect.h <= 0 || rect.x < 0 || rect.y < 0) {
    return false;
  }
  const int plane_w = ctx.mi_cols * kMiSize;
  const int plane_h = ctx.mi_rows * kMiSize;
  const int dy = to_fullmv(mv_row);
  const int dx = to_fullmv(mv_col);
  *ref_y = ctx.mi_row * kMiSize + dy;
  *ref_x = ctx.mi_col * kMiSize + dx;
  *bw = plane_w - rect.x;
  if (rect.w < *bw) {
    *bw = rect.w;
  }
  *bh = plane_h - rect.y;
  if (rect.h < *bh) {
    *bh = rect.h;
  }
  return *bw > 0 && *bh > 0;
}

// The §7.13.3.25 conformance box. The template reads (refX-1, refY-1) and up
// to (refX+bw-1, refY+bh-1), so this is exactly the spec's four requirements.
inline bool morph_reference_in_bounds(int ref_x, int ref_y, int bw, int bh,
                                      int plane_w, int plane_h) {
  return ref_x >= 1 && ref_y >= 1 && ref_x + bw <= plane_w && ref_y + bh <= plane_h;
}

// Template plan for luma (plane 0).
inline MorphTemplatePlan morph_template_plan(int bw, int bh, bool avail_u, bool avail_l) {
  MorphTemplatePlan plan{};
  const int bw2 = bw < 16 ? bw : 16;
  const int bh2 = bh < 16 ? bh : 16;
  plan.width = bw2 == 12 ? 8 : bw2;
  plan.height = bh2 == 12 ? 8 : bh2;
  if (avail_u && avail_l) {
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
  } else if (avail_u) {
    plan.num_up = plan.width;
  } else if (avail_l) {
    plan.num_left = plan.height;
  }
  if (plan.num_up > 0) {
    plan.up_step = plan.width / plan.num_up;
  }
  if (plan.num_left > 0) {
    plan.left_step = plan.height / plan.num_left;
  }
  return plan;
}

// Derives the fit only. `frame` must already contain the IBC prediction for
// the block, because the template above/left samples are reconstructed values
// and the current block is the prediction being adapted (the spec's
// prevFrame = CurrFrame makes the source and destination the same array).
inline bool morph_derive(const MorphContext& ctx, const MorphPlane& frame,
                         const MorphRect& rect, int mv_row, int mv_col, MorphFit* fit_out) {
  if (fit_out == nullptr || frame.data == nullptr || frame.stride <= 0) {
    return false;
  }
  if (ctx.avail_u && rect.y < 1) {
    return false;
  }
  if (ctx.avail_l && rect.x < 1) {
    return false;
  }
  MorphFit fit{};
  if (!morph_reference_geometry(ctx, rect, mv_row, mv_col, &fit.ref_x, &fit.ref_y, &fit.bw,
                                &fit.bh)) {
    return false;
  }
  const int plane_w = ctx.mi_cols * kMiSize;
  const int plane_h = ctx.mi_rows * kMiSize;
  if (!morph_reference_in_bounds(fit.ref_x, fit.ref_y, fit.bw, fit.bh, plane_w, plane_h)) {
    return false;
  }

  constexpr int kShift = 8;
  fit.alpha = 1 << kShift;
  fit.beta = -(1 << (kShift - 1));
  fit.plan = morph_template_plan(fit.bw, fit.bh, ctx.avail_u, ctx.avail_l);

  int sum_x = 0;
  int sum_y = 0;
  int sum_xx = 0;
  int sum_xy = 0;
  int count = 0;
  const int x = rect.x;
  const int y = rect.y;
  if (fit.plan.num_up > 0) {
    const int step = fit.plan.up_step;
    for (int i = step >> 1; i < fit.plan.width; i += step) {
      const int recon = frame.data[(y - 1) * frame.stride + x + i];
      const int ref =
          frame.data[(fit.ref_y - 1) * frame.stride + fit.ref_x + i];
      sum_x += ref;
      sum_y += recon;
      sum_xy += ref * recon;
      sum_xx += ref * ref;
    }
    count += fit.plan.num_up;
  }
  if (fit.plan.num_left > 0) {
    const int step = fit.plan.left_step;
    for (int i = step >> 1; i < fit.plan.height; i += step) {
      const int recon = frame.data[(y + i) * frame.stride + x - 1];
      const int ref =
          frame.data[(fit.ref_y + i) * frame.stride + fit.ref_x - 1];
      sum_x += ref;
      sum_y += recon;
      sum_xy += ref * recon;
      sum_xx += ref * ref;
    }
    count += fit.plan.num_left;
  }

  // morphPred == 1, plane == 0: neither the explicit-alpha branch nor the
  // BawpAlpha branch applies.
  if (count > 0) {
    const int nor = sum_xy - sum_x * sum_y / count;
    const int der = sum_xx - sum_x * sum_x / count;
    if (der != 0 && nor != 0) {
      fit.alpha = resolve_division(nor, der, kShift);
      if (fit.alpha == 0) {
        fit.alpha = 1 << kShift;
      }
    } else {
      fit.alpha = 1 << kShift;
    }
    fit.beta = ((sum_y << kShift) - sum_x * fit.alpha) / count;
  }
  fit.count = count;
  *fit_out = fit;
  return true;
}

// Applies the fit in place to all w*h samples. §7.13.3.25 note: this covers
// the full block, not only the part inside planeWidth/planeHeight.
inline void morph_apply_fit(const MorphFit& fit, const MorphPlane& frame, const MorphRect& rect) {
  constexpr int kShift = 8;
  for (int i = 0; i < rect.h; ++i) {
    std::uint8_t* row =
        frame.data + static_cast<std::ptrdiff_t>(rect.y + i) * frame.stride;
    for (int j = 0; j < rect.w; ++j) {
      const int orig = row[rect.x + j];
      row[rect.x + j] = static_cast<std::uint8_t>(
          clip1((orig * fit.alpha + fit.beta) >> kShift, 8));
    }
  }
}

// Full §7.13.3.26 process: derive then adapt. On failure the plane is
// untouched.
inline bool morph_predict(const MorphContext& ctx, const MorphPlane& frame,
                          const MorphRect& rect, int mv_row, int mv_col,
                          MorphFit* fit_out = nullptr) {
  MorphFit fit{};
  if (!morph_derive(ctx, frame, rect, mv_row, mv_col, &fit)) {
    return false;
  }
  morph_apply_fit(fit, frame, rect);
  if (fit_out != nullptr) {
    *fit_out = fit;
  }
  return true;
}

}  // namespace av2enc::inter::morph

#endif  // AV2ENC_INTER_MORPH_MORPH_COMMON_H_
