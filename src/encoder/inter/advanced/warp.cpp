#include "warp.h"

#include <cstring>

namespace av2enc::inter::adv {
namespace {

bool geom_ok(const WarpBlockGeometry& g, const PlaneView& ref, const std::int16_t* dst) {
  if (ref.data == nullptr || dst == nullptr) {
    return false;
  }
  if (g.rect.w <= 0 || g.rect.h <= 0 || g.rect.w > 64 || g.rect.h > 64) {
    return false;
  }
  if (ref.stride <= 0 || ref.max_x < 0 || ref.max_y < 0) {
    return false;
  }
  return g.frame_w > 0 && g.frame_h > 0;
}

}  // namespace

bool warp_ref_is_scaled(const WarpBlockGeometry& geom) {
  return is_scaled_ref(geom.ref_w, geom.ref_h, geom.frame_w, geom.frame_h);
}

bool warp_needs_extended(const WarpParams& params, const WarpBlockGeometry& geom) {
  return !setup_shear(params).valid || geom.rect.w < 8 || geom.rect.h < 8 ||
         warp_ref_is_scaled(geom);
}

bool warp_predict(const WarpParams& params, const PlaneView& ref, const WarpBlockGeometry& geom,
                  bool compound, std::int16_t* dst, int dst_stride) {
  if (!geom_ok(geom, ref, dst) || dst_stride <= 0) {
    return false;
  }
  const int inter_round1 = compound ? kInterRound1Compound : kInterRound1Single;
  std::int32_t intermediate[kBlockWarpRows * kBlockWarpCols];
  for (int i8 = 0; i8 <= (geom.rect.h - 1) >> 3; ++i8) {
    for (int j8 = 0; j8 <= (geom.rect.w - 1) >> 3; ++j8) {
      block_warp_predict(params, ref, geom.rect.x, geom.rect.y, i8, j8, geom.rect.w, geom.rect.h,
                         geom.sub_x, geom.sub_y, inter_round1, !compound, geom.dst_x, geom.dst_y,
                         intermediate, dst, dst_stride);
    }
  }
  return true;
}

bool warp_predict_ext(const WarpParams& params, const PlaneView& ref, const WarpBlockGeometry& geom,
                      bool compound, std::int16_t* dst, int dst_stride) {
  if (!geom_ok(geom, ref, dst) || dst_stride <= 0) {
    return false;
  }
  ExtWarpContext ctx{};
  ctx.plane = geom.plane;
  ctx.sub_x = geom.sub_x;
  ctx.sub_y = geom.sub_y;
  ctx.block_w = geom.rect.w;
  ctx.block_h = geom.rect.h;
  ctx.mi_col = geom.mi_col;
  ctx.mi_row = geom.mi_row;
  ctx.bw4 = geom.bw4;
  ctx.bh4 = geom.bh4;
  ctx.mi_cols = geom.mi_cols;
  ctx.mi_rows = geom.mi_rows;
  ctx.scaled = warp_ref_is_scaled(geom);
  ctx.ref_w = geom.ref_w;
  ctx.ref_h = geom.ref_h;
  ctx.frame_w = geom.frame_w;
  ctx.frame_h = geom.frame_h;
  ctx.inter_round1 = compound ? kInterRound1Compound : kInterRound1Single;
  ctx.dst_x = geom.dst_x;
  ctx.dst_y = geom.dst_y;
  ctx.clip_output = !compound;

  std::int32_t intermediate[kExtWarpRows * kExtWarpCols];
  for (int i4 = 0; i4 < (geom.rect.h >> 2); ++i4) {
    for (int j4 = 0; j4 < (geom.rect.w >> 2); ++j4) {
      if (!ext_block_warp_predict(params, ref, ctx, geom.rect.x, geom.rect.y, i4, j4, intermediate,
                                  dst, dst_stride)) {
        return false;
      }
    }
  }
  return true;
}

bool warp_predict_u8(const WarpParams& params, const PlaneView& ref, const WarpBlockGeometry& geom,
                     const MutablePlaneView& dst) {
  if (dst.data == nullptr || dst.stride <= 0) {
    return false;
  }
  std::int16_t scratch[64 * 64];
  WarpBlockGeometry local = geom;
  local.dst_x = 0;
  local.dst_y = 0;
  if (!warp_predict(params, ref, local, false, scratch, geom.rect.w)) {
    return false;
  }
  for (int i = 0; i < geom.rect.h; ++i) {
    for (int j = 0; j < geom.rect.w; ++j) {
      dst.data[static_cast<std::ptrdiff_t>(geom.dst_y + i) * dst.stride + geom.dst_x + j] =
          static_cast<std::uint8_t>(scratch[i * geom.rect.w + j]);
    }
  }
  return true;
}

std::optional<WarpParams> estimate_local_warp(const WarpSample* samples, int count, int mid_x,
                                              int mid_y, const int block_mv[2]) {
  if (samples == nullptr || block_mv == nullptr || count < 0 || count > kLeastSquaresSamplesMax) {
    return std::nullopt;
  }
  return warp_estimate(samples, count, mid_x, mid_y, block_mv);
}

WarpParams make_affine_params(int m00_q16, int m01_q16, int m10_q16, int m11_q16, int trans_row_q16,
                              int trans_col_q16) {
  WarpParams params{};
  params.p[0] = trans_row_q16;
  params.p[1] = trans_col_q16;
  params.p[2] = m00_q16;
  params.p[3] = m01_q16;
  params.p[4] = m10_q16;
  params.p[5] = m11_q16;
  return params;
}

std::int64_t warp_block_sse(const WarpParams& params, const PlaneView& ref,
                            const WarpBlockGeometry& geom, bool compound,
                            const std::uint8_t* source, int source_stride, std::int16_t* scratch,
                            int scratch_stride) {
  if (source == nullptr || scratch == nullptr || source_stride <= 0 || scratch_stride <= 0) {
    return -1;
  }
  if (!warp_predict(params, ref, geom, compound, scratch, scratch_stride)) {
    return -1;
  }
  std::int64_t sse = 0;
  for (int i = 0; i < geom.rect.h; ++i) {
    for (int j = 0; j < geom.rect.w; ++j) {
      const int raw = scratch[i * scratch_stride + j];
      const int a = compound ? clip1(round2(raw, 4), 8) : raw;
      const int b = source[i * source_stride + j];
      sse += static_cast<std::int64_t>((a - b) * (a - b));
    }
  }
  return sse;
}

}  // namespace av2enc::inter::adv
