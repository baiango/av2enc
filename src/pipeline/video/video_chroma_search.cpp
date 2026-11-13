#include "pipeline/video/video_chroma_search.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

#include "codec/intra/cfl/cfl.h"
#include "codec/intra/cfl/mhccp/mhccp.h"

namespace av2enc::video::chroma_search {

namespace mhccp = ::av2enc::cfl::mhccp;
namespace {

void write_block(std::uint8_t* plane, int stride, int x, int y, int w, int h,
                 const std::vector<std::uint8_t>& samples) {
  for (int yy = 0; yy < h; ++yy) {
    for (int xx = 0; xx < w; ++xx) {
      plane[static_cast<std::size_t>(y + yy) * static_cast<std::size_t>(stride) +
            static_cast<std::size_t>(x + xx)] =
          samples[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) +
                  static_cast<std::size_t>(xx)];
    }
  }
}

int source_sample(const PlaneView& plane, int x, int y) {
  const int cx = x < 0 ? 0 : (x > plane.source_max_x ? plane.source_max_x : x);
  const int cy = y < 0 ? 0 : (y > plane.source_max_y ? plane.source_max_y : y);
  return plane.source[static_cast<std::size_t>(cy) *
                          static_cast<std::size_t>(plane.source_stride) +
                      static_cast<std::size_t>(cx)];
}

cfl::Block make_block(const PlaneView& plane, const std::uint8_t* luma,
                      int luma_stride, int luma_max_x, int luma_max_y,
                      int sb_top_y, int x, int y, int w, int h,
                      int coded_width, int coded_height, int ds_filter) {
  cfl::Block block;
  block.luma = luma;
  block.luma_stride = luma_stride;
  block.luma_max_x = luma_max_x;
  block.luma_max_y = luma_max_y;
  block.sb_top_y = sb_top_y;
  block.chroma = plane.recon;
  block.chroma_stride = plane.recon_stride;
  block.x = x;
  block.y = y;
  block.w = w;
  block.h = h;
  block.coded_width = coded_width;
  block.coded_height = coded_height;
  block.have_above = cfl::cfl_avail_above(y, 1) ? 1 : 0;
  block.have_left = cfl::cfl_avail_left(x, 1) ? 1 : 0;
  block.bit_depth = 8;
  block.ds_filter = ds_filter;
  block.sub_x = 1;
  block.sub_y = 1;
  return block;
}

}  // namespace

bool choose(const PlaneView& u_plane, const PlaneView& v_plane,
            const std::uint8_t* recon_luma, int luma_stride,
            int luma_coded_max_x, int luma_coded_max_y, int sb_top_y,
            int x, int y, int w, int h, int coded_width, int coded_height,
            int num4_above_right, int num4_below_left,
            const std::vector<std::uint8_t>& dc_u,
            const std::vector<std::uint8_t>& dc_v, std::int64_t dc_sse_u,
            const Options& options, Choice& out, std::string& error) {
  const std::size_t count =
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  out = Choice{};
  out.u = dc_u;
  out.v = dc_v;
  out.sse = dc_sse_u;
  if (!options.enable_cfl) return true;
  if (u_plane.recon == nullptr || v_plane.recon == nullptr ||
      recon_luma == nullptr || u_plane.source == nullptr ||
      v_plane.source == nullptr || dc_u.size() != count ||
      dc_v.size() != count) {
    error = "video CfL: invalid block descriptor";
    return false;
  }
  if (x < 0 || y < 0 || u_plane.recon_width < 0 || u_plane.recon_height < 0 ||
      u_plane.recon_width != v_plane.recon_width ||
      u_plane.recon_height != v_plane.recon_height) {
    error = "video CfL: inconsistent reconstruction plane";
    return false;
  }
  // A coding block may extend past the MI-aligned recon plane at the
  // bottom/right edge (PARTITION_NONE is legal while has_rows is true).  The
  // reconstruction writes clip, but the CfL/MHCCP prediction needs all w*h
  // samples for the transform, so compute on a padded copy in that case.
  const bool block_fits = x + w <= u_plane.recon_width &&
                          y + h <= u_plane.recon_height;
  std::vector<std::uint8_t> local_u;
  std::vector<std::uint8_t> local_v;
  PlaneView u_view = u_plane;
  PlaneView v_view = v_plane;
  int local_stride = 0;
  if (!block_fits) {
    const int local_w = std::max(u_plane.recon_width, x + w);
    const int local_h = std::max(u_plane.recon_height, y + h);
    local_stride = local_w;
    local_u.assign(static_cast<std::size_t>(local_w) *
                       static_cast<std::size_t>(local_h),
                   0);
    local_v.assign(local_u.size(), 0);
    for (int row = 0; row < u_plane.recon_height; ++row) {
      std::memcpy(local_u.data() + static_cast<std::size_t>(row) * local_w,
                  u_plane.recon + static_cast<std::size_t>(row) *
                                      u_plane.recon_stride,
                  static_cast<std::size_t>(u_plane.recon_width));
      std::memcpy(local_v.data() + static_cast<std::size_t>(row) * local_w,
                  v_plane.recon + static_cast<std::size_t>(row) *
                                      v_plane.recon_stride,
                  static_cast<std::size_t>(v_plane.recon_width));
    }
    u_view.recon = local_u.data();
    u_view.recon_stride = local_stride;
    v_view.recon = local_v.data();
    v_view.recon_stride = local_stride;
  }

  if (w > 32 || h > 32) {
    error =
        "video CfL: chroma block exceeds 32 samples; the §7.13.2.11 "
        "DC-subsampled prediction is not implemented";
    return false;
  }

  // The CfL base is the DC prediction.  Write it into the reconstructed
  // planes exactly like the M1 encoder does before deriving the alpha.
  write_block(u_view.recon, u_view.recon_stride, x, y, w, h, dc_u);
  write_block(v_view.recon, v_view.recon_stride, x, y, w, h, dc_v);

  cfl::Block ub = make_block(u_view, recon_luma, luma_stride, luma_coded_max_x,
                             luma_coded_max_y, sb_top_y, x, y, w, h,
                             coded_width, coded_height, options.ds_filter);
  cfl::Block vb = make_block(v_view, recon_luma, luma_stride, luma_coded_max_x,
                             luma_coded_max_y, sb_top_y, x, y, w, h,
                             coded_width, coded_height, options.ds_filter);
  cfl::Scratch scratch;
  cfl::compute_luma(ub, scratch);
  const int alpha_u = cfl::derive_implicit_alpha(ub, scratch);
  cfl::compute_luma(vb, scratch);
  const int alpha_v = cfl::derive_implicit_alpha(vb, scratch);
  cfl::apply_prediction(ub, alpha_u, scratch);
  cfl::apply_prediction(vb, alpha_v, scratch);

  std::int64_t cfl_sse = 0;
  for (int yy = 0; yy < h; ++yy) {
    for (int xx = 0; xx < w; ++xx) {
      const std::size_t index = static_cast<std::size_t>(yy) *
                                    static_cast<std::size_t>(w) +
                                static_cast<std::size_t>(xx);
      const int pu = u_view.recon[static_cast<std::size_t>(y + yy) *
                                      static_cast<std::size_t>(
                                          u_view.recon_stride) +
                                  static_cast<std::size_t>(x + xx)];
      const int pv = v_view.recon[static_cast<std::size_t>(y + yy) *
                                      static_cast<std::size_t>(
                                          v_view.recon_stride) +
                                  static_cast<std::size_t>(x + xx)];
      out.u[index] = static_cast<std::uint8_t>(pu);
      out.v[index] = static_cast<std::uint8_t>(pv);
      const int du = source_sample(u_plane, x + xx, y + yy) - pu;
      const int dv = source_sample(v_plane, x + xx, y + yy) - pv;
      cfl_sse += static_cast<std::int64_t>(du) * du;
      cfl_sse += static_cast<std::int64_t>(dv) * dv;
    }
  }
  // §7.13.6-.8 MHCCP: the three CFL_MULTI directions compete with the derived
  // hypothesis first (strictly better wins), then the winner competes with DC.
  bool mh_wins = false;
  int mh_dir = 0;
  std::int64_t best_cfl = cfl_sse;
  std::uint8_t* u_recon = u_view.recon;
  std::uint8_t* v_recon = v_view.recon;
  const int u_stride = u_view.recon_stride;
  const int v_stride = v_view.recon_stride;
  if (options.enable_mhccp &&
      mhccp::allowed(w, h, /*enable_mhccp=*/true, /*is_chroma_part=*/false,
                     /*frame_is_intra=*/true, /*cfl_allowed_in_sdp=*/false,
                     /*lossless=*/false)) {
    mhccp::Block mh_u =
        mhccp::make_block(ub, num4_above_right, num4_below_left,
                          luma_coded_max_x + 1, luma_coded_max_y + 1);
    mhccp::Block mh_v =
        mhccp::make_block(vb, num4_above_right, num4_below_left,
                          luma_coded_max_x + 1, luma_coded_max_y + 1);
    mhccp::SourcePlane su;
    su.plane = u_plane.source;
    su.stride = u_plane.source_stride;
    su.max_x = u_plane.source_max_x;
    su.max_y = u_plane.source_max_y;
    mhccp::SourcePlane sv;
    sv.plane = v_plane.source;
    sv.stride = v_plane.source_stride;
    sv.max_x = v_plane.source_max_x;
    sv.max_y = v_plane.source_max_y;
    const std::optional<mhccp::Candidate> multi =
        mhccp::search_mhccp(mh_u, mh_v, su, sv);
    if (multi.has_value() && multi->total_sse() < best_cfl) {
      mhccp::apply_mhccp_pair(mh_u, mh_v, multi->mh_dir, multi->params_u,
                              multi->params_v);
      best_cfl = multi->total_sse();
      mh_wins = true;
      mh_dir = multi->mh_dir;
      for (int yy = 0; yy < h; ++yy) {
        for (int xx = 0; xx < w; ++xx) {
          const std::size_t index = static_cast<std::size_t>(yy) *
                                        static_cast<std::size_t>(w) +
                                    static_cast<std::size_t>(xx);
          out.u[index] = u_recon[static_cast<std::size_t>(y + yy) *
                                     static_cast<std::size_t>(u_stride) +
                                 static_cast<std::size_t>(x + xx)];
          out.v[index] = v_recon[static_cast<std::size_t>(y + yy) *
                                     static_cast<std::size_t>(v_stride) +
                                 static_cast<std::size_t>(x + xx)];
        }
      }
    } else {
      // Restore the derived prediction that won the comparison.
      cfl::compute_luma(ub, scratch);
      cfl::apply_prediction(ub, alpha_u, scratch);
      cfl::apply_prediction(vb, alpha_v, scratch);
    }
  }
  // Same comparison as the M1 encoder: the DC candidate is scored on U only.
  if (best_cfl < dc_sse_u) {
    out.is_cfl = true;
    out.cfl_index = mh_wins ? 2 : 1;
    out.cfl_mh_dir = mh_dir;
    out.alpha_u_q8 = mh_wins ? 0 : alpha_u;
    out.alpha_v_q8 = mh_wins ? 0 : alpha_v;
    out.sse = best_cfl;
    return true;
  }
  // DC wins: restore its prediction into the planes so the caller's later
  // reconstruction write sees the same transient state as the M1 path.
  write_block(u_view.recon, u_view.recon_stride, x, y, w, h, dc_u);
  write_block(v_view.recon, v_view.recon_stride, x, y, w, h, dc_v);
  out.u = dc_u;
  out.v = dc_v;
  out.sse = dc_sse_u;
  return true;
}

}  // namespace av2enc::video::chroma_search
