#include "bawp.h"

namespace av2enc::inter::adv {
namespace {

bool resolve_ref_origin(const BawpJob& job, int* ref_x, int* ref_y, int* bw, int* bh) {
  const BawpContext& ctx = job.ctx;
  const int plane_w = (ctx.mi_cols * kMiSize) >> ctx.sub_x;
  const int plane_h = (ctx.mi_rows * kMiSize) >> ctx.sub_y;
  const int dy = to_fullmv(job.mv[0]);
  const int dx = to_fullmv(job.mv[1]);
  *ref_y = (ctx.mi_row * kMiSize + dy) >> ctx.sub_y;
  *ref_x = (ctx.mi_col * kMiSize + dx) >> ctx.sub_x;
  *bw = plane_w - job.rect.x;
  if (job.rect.w < *bw) {
    *bw = job.rect.w;
  }
  *bh = plane_h - job.rect.y;
  if (job.rect.h < *bh) {
    *bh = job.rect.h;
  }
  return *bw > 0 && *bh > 0;
}

bool job_ok(const BawpJob& job) {
  if (job.cur.data == nullptr || job.prev.data == nullptr || job.dst.data == nullptr) {
    return false;
  }
  if (job.cur.stride <= 0 || job.prev.stride <= 0 || job.dst.stride <= 0) {
    return false;
  }
  if (job.rect.w <= 0 || job.rect.h <= 0 || job.rect.x < 0 || job.rect.y < 0) {
    return false;
  }
  if (job.ctx.mi_cols <= 0 || job.ctx.mi_rows <= 0) {
    return false;
  }
  // The template reads (x-1) / (y-1), so the block must not be at the plane
  // origin unless the corresponding neighbour is unavailable.
  if (job.ctx.avail_u && job.rect.y < 1) {
    return false;
  }
  if (job.ctx.avail_l && job.rect.x < 1) {
    return false;
  }
  return true;
}

bool derive(const BawpJob& job, BawpFit* fit_out) {
  if (!job_ok(job) || fit_out == nullptr) {
    return false;
  }
  int ref_x = 0;
  int ref_y = 0;
  int bw = 0;
  int bh = 0;
  if (!resolve_ref_origin(job, &ref_x, &ref_y, &bw, &bh)) {
    return false;
  }
  // Conformance: the reference block must be at least one sample inside.
  if (ref_x < 1 || ref_y < 1 || ref_x + bw > (job.ctx.mi_cols * kMiSize) >> job.ctx.sub_x ||
      ref_y + bh > (job.ctx.mi_rows * kMiSize) >> job.ctx.sub_y) {
    return false;
  }

  const PlaneView& template_ref = job.morph_pred ? job.cur : job.prev;
  *fit_out =
      bawp_fit(job.ctx, job.cur, template_ref, job.rect.x, job.rect.y, job.rect.w, job.rect.h,
               ref_x, ref_y, job.bawp_alpha, job.explicit_bawp && !job.morph_pred,
               job.explicit_bawp_scale, job.list_index, job.first_ref_dist);
  return true;
}

}  // namespace

bool bawp_derive_block(const BawpJob& job, BawpFit* fit_out) { return derive(job, fit_out); }

bool bawp_apply_block(const BawpJob& job, BawpFit* fit_out) {
  BawpFit fit{};
  if (!derive(job, &fit)) {
    return false;
  }
  bawp_apply(fit, job.dst, job.rect.x, job.rect.y, job.rect.w, job.rect.h);
  if (fit_out != nullptr) {
    *fit_out = fit;
  }
  return true;
}

int bawp_list_index(bool is_near_mv, bool is_new_mv, bool use_amvd) {
  if (is_near_mv) {
    return 0;
  }
  if (is_new_mv && use_amvd) {
    return 1;
  }
  return 2;
}

}  // namespace av2enc::inter::adv
