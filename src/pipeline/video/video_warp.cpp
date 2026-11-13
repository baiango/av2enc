#include "pipeline/video/video_warp.h"

#include <algorithm>
#include <cstddef>

#include "codec/inter/advanced/warp.h"

namespace av2enc::video {
namespace {

using inter::adv::PlaneView;
using inter::adv::WarpBlockGeometry;
using inter::adv::WarpParams;

WarpParams to_warp_params(const inter::gme::GmParams& params) {
  WarpParams warp;
  for (int i = 0; i < 6; ++i) warp.p[i] = params.p[static_cast<std::size_t>(i)];
  return warp;
}

}  // namespace

bool warp_predict_plane(const inter::gme::GmParams& params,
                        const inter::mc::RefFrame& reference, int plane, int x,
                        int y, int w, int h, int mi_row, int mi_col, int bw4,
                        int bh4, std::vector<std::uint8_t>& out,
                        std::string& error) {
  if (plane < 0 || plane > 2) {
    error = "warp: plane out of range";
    return false;
  }
  if (w <= 0 || h <= 0 || w > 64 || h > 64) {
    error = "warp: unsupported block size";
    return false;
  }
  if (params.type == inter::gme::GmType::kIdentity) {
    error = "warp: identity model";
    return false;
  }
  const int sub_x = plane == 0 ? 0 : 1;
  const int sub_y = plane == 0 ? 0 : 1;
  const int plane_w = (reference.mi_cols * 4) >> sub_x;
  const int plane_h = (reference.mi_rows * 4) >> sub_y;
  if (reference.plane[plane].data == nullptr || reference.plane[plane].stride <= 0) {
    error = "warp: missing reference plane";
    return false;
  }
  const PlaneView view{reference.plane[plane].data,
                       reference.plane[plane].stride, plane_w - 1,
                       plane_h - 1};
  WarpBlockGeometry geom;
  geom.plane = plane;
  geom.sub_x = sub_x;
  geom.sub_y = sub_y;
  geom.rect = inter::adv::BlockRect{x, y, w, h};
  geom.dst_x = 0;
  geom.dst_y = 0;
  geom.mi_col = mi_col;
  geom.mi_row = mi_row;
  geom.bw4 = bw4;
  geom.bh4 = bh4;
  geom.mi_cols = reference.mi_cols;
  geom.mi_rows = reference.mi_rows;
  geom.ref_w = plane_w;
  geom.ref_h = plane_h;
  geom.frame_w = plane_w;
  geom.frame_h = plane_h;
  geom.bit_depth = 8;

  out.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  std::vector<std::int16_t> scratch(static_cast<std::size_t>(w) *
                                    static_cast<std::size_t>(h));
  const WarpParams warp = to_warp_params(params);
  bool ok = false;
  if (inter::adv::warp_needs_extended(warp, geom)) {
    ok = inter::adv::warp_predict_ext(warp, view, geom, /*compound=*/false,
                                      scratch.data(), w);
  } else {
    ok = inter::adv::warp_predict(warp, view, geom, /*compound=*/false,
                                  scratch.data(), w);
  }
  if (!ok) {
    error = "warp: prediction failed";
    return false;
  }
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<std::uint8_t>(std::clamp<int>(scratch[i], 0, 255));
  }
  return true;
}

bool warp_predict_420(const inter::gme::GmParams& params,
                      const inter::mc::RefFrame& reference, int x, int y, int w,
                      int h, int mi_row, int mi_col, int bw4, int bh4,
                      std::vector<std::uint8_t>& y_out,
                      std::vector<std::uint8_t>& u_out,
                      std::vector<std::uint8_t>& v_out, std::string& error) {
  const int cw = w / 2;
  const int ch = h / 2;
  if (!warp_predict_plane(params, reference, 0, x, y, w, h, mi_row, mi_col,
                          bw4, bh4, y_out, error)) {
    return false;
  }
  if (!warp_predict_plane(params, reference, 1, x / 2, y / 2, cw, ch, mi_row,
                          mi_col, bw4, bh4, u_out, error)) {
    return false;
  }
  return warp_predict_plane(params, reference, 2, x / 2, y / 2, cw, ch, mi_row,
                            mi_col, bw4, bh4, v_out, error);
}

}  // namespace av2enc::video
