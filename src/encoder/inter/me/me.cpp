// AV2 inter motion estimation — CPU reference implementation.
//
// The search itself lives in me_core.h so the CUDA kernel can instantiate the
// identical state machine.  This file only validates the host API and loops
// over jobs with the scalar cost functor.

#include "encoder/inter/me/me.h"

#include <cstddef>

namespace av2enc::me {
namespace {

bool plane_valid(const MeCorePlane& p) {
  return p.data != nullptr && p.width > 0 && p.height > 0 &&
         p.stride >= p.width;
}

bool frame_valid(const MeCoreFrame& f) {
  return plane_valid(f.plane[0]) && f.width == f.plane[0].width &&
         f.height == f.plane[0].height;
}

bool ref_valid(const MeCoreRef& r, const MeCoreFrame& cur) {
  return plane_valid(r.plane[0]) && r.plane[0].width >= cur.width &&
         r.plane[0].height >= cur.height;
}

bool options_valid(const MeCoreOptions& o) {
  if (o.method < kMeFull || o.method > kMeTestZone) {
    return false;
  }
  if (o.metric != kMeSad && o.metric != kMeSatd) {
    return false;
  }
  if (o.range < 0 || o.range > 4096 || o.integer_step < 1 ||
      o.max_iterations < 1 || o.subpel_iterations < 0 ||
      o.max_candidates < 0 || o.diamond_step < 0 || o.tz_start_step < 0) {
    return false;
  }
  if ((o.subpel_stages & ~(kMeSubpelHalf | kMeSubpelQuarter |
                           kMeSubpelEighth | kMeSubpelSixteenth)) != 0) {
    return false;
  }
  if (o.subpel_interp < 0 || o.subpel_interp > kMeMaxInterpFilter) {
    return false;
  }
  if (o.use_predictors < 0 || o.use_predictors > 1 ||
      o.use_merge_candidates < 0 || o.use_merge_candidates > 1) {
    return false;
  }
  if (o.max_merge_candidates < 0 || o.max_merge_candidates > kMeMaxMerge) {
    return false;
  }
  return true;
}

bool mv_in_bounds(const MeCorePlane& ref, int x, int y, int w, int h,
                  std::int32_t mv_x, std::int32_t mv_y) {
  if (w <= 0 || h <= 0 || x < 0 || y < 0 || w > ref.width - x ||
      h > ref.height - y) {
    return false;
  }
  const std::int32_t lo_x = -static_cast<std::int32_t>(x) * kSubpelOne -
                            kSubpelMask;
  const std::int32_t hi_x =
      static_cast<std::int32_t>(ref.width - x - w) * kSubpelOne + kSubpelMask;
  const std::int32_t lo_y = -static_cast<std::int32_t>(y) * kSubpelOne -
                            kSubpelMask;
  const std::int32_t hi_y =
      static_cast<std::int32_t>(ref.height - y - h) * kSubpelOne + kSubpelMask;
  return mv_x >= lo_x && mv_x <= hi_x && mv_y >= lo_y && mv_y <= hi_y;
}

}  // namespace

const char* me_status_string(MeStatus status) {
  switch (status) {
    case MeStatus::kOk:
      return "ok";
    case MeStatus::kInvalidArgument:
      return "invalid argument";
    case MeStatus::kUnsupportedFormat:
      return "unsupported format";
  }
  return "unknown";
}

MeStatus me_search_batch(const MeCoreFrame& cur, const MeCoreRef* refs,
                         int num_refs, const MeCoreJob* jobs, int num_jobs,
                         const MeCoreOptions& options,
                         const MeCoreMvField* field, MeCoreResult* results) {
  if (!frame_valid(cur)) {
    return MeStatus::kInvalidArgument;
  }
  if (num_refs <= 0 || num_refs > kMeMaxRefs || refs == nullptr) {
    return MeStatus::kInvalidArgument;
  }
  for (int i = 0; i < num_refs; ++i) {
    if (!ref_valid(refs[i], cur)) {
      return MeStatus::kInvalidArgument;
    }
  }
  if (!options_valid(options)) {
    return MeStatus::kInvalidArgument;
  }
  if (num_jobs < 0 || (num_jobs > 0 && (jobs == nullptr || results == nullptr))) {
    return MeStatus::kInvalidArgument;
  }
  if (field != nullptr) {
    if (field->cols < 0 || field->rows < 0 ||
        (field->cols > 0 && (field->mv_x == nullptr || field->mv_y == nullptr)) ||
        (field->stride > 0 && field->stride < field->cols)) {
      return MeStatus::kInvalidArgument;
    }
  }
  for (int j = 0; j < num_jobs; ++j) {
    const MeCoreJob& job = jobs[j];
    if (job.x < 0 || job.y < 0 || job.x >= cur.width ||
        job.y >= cur.height || job.w < 1 || job.h < 1 ||
        job.w > kMeMaxBlockDim || job.h > kMeMaxBlockDim) {
      return MeStatus::kInvalidArgument;
    }
    if (job.num_refs < 1 || job.num_refs > kMeMaxRefs) {
      return MeStatus::kInvalidArgument;
    }
    for (int i = 0; i < job.num_refs; ++i) {
      const int ri = job.ref_indices[i];
      if (ri < 0 || ri >= num_refs) {
        return MeStatus::kInvalidArgument;
      }
    }
    if (job.num_pred < 0 || job.num_pred > kMeMaxPredictors) {
      return MeStatus::kInvalidArgument;
    }
  }

  const MeScalarCost cost;
  for (int j = 0; j < num_jobs; ++j) {
    results[j] =
        me_search_core(cur, refs, jobs[j], options, field, cost);
  }
  return MeStatus::kOk;
}

MeStatus me_cost_block(const MeCoreFrame& cur, const MeCoreRef& ref,
                       const MeCostJob& job, std::int32_t* out_cost) {
  if (out_cost == nullptr || job.plane < 0 || job.plane > 2 ||
      (job.metric != kMeSad && job.metric != kMeSatd) ||
      job.interp < 0 || job.interp > kMeMaxInterpFilter) {
    return MeStatus::kInvalidArgument;
  }
  const MeCorePlane& cur_p = cur.plane[job.plane];
  const MeCorePlane& ref_p = ref.plane[job.plane];
  if (!plane_valid(cur_p) || !plane_valid(ref_p)) {
    return MeStatus::kInvalidArgument;
  }
  if (job.x < 0 || job.y < 0 || job.w < 1 || job.h < 1 ||
      job.w > kMeMaxBlockDim || job.h > kMeMaxBlockDim ||
      job.x + job.w > cur_p.width || job.y + job.h > cur_p.height) {
    return MeStatus::kInvalidArgument;
  }
  if (!mv_in_bounds(ref_p, job.x, job.y, job.w, job.h, job.mv_x, job.mv_y)) {
    return MeStatus::kInvalidArgument;
  }
  *out_cost = me_cost_plane(cur_p, ref_p, job.x, job.y, job.w, job.h, job.mv_x,
                            job.mv_y, job.metric, job.interp);
  return MeStatus::kOk;
}

MeStatus me_interp_block(const MeCoreRef& ref, const MeInterpJob& job,
                         std::uint8_t* out) {
  if (out == nullptr || job.plane < 0 || job.plane > 2 || job.x < 0 ||
      job.y < 0 || job.w < 1 || job.h < 1 || job.w > kMeMaxBlockDim ||
      job.h > kMeMaxBlockDim || job.interp < 0 ||
      job.interp > kMeMaxInterpFilter) {
    return MeStatus::kInvalidArgument;
  }
  const MeCorePlane& ref_p = ref.plane[job.plane];
  if (!plane_valid(ref_p)) {
    return MeStatus::kInvalidArgument;
  }
  if (job.x + job.w > ref_p.width || job.y + job.h > ref_p.height) {
    return MeStatus::kInvalidArgument;
  }
  if (!mv_in_bounds(ref_p, job.x, job.y, job.w, job.h, job.mv_x, job.mv_y)) {
    return MeStatus::kInvalidArgument;
  }
  for (int r = 0; r < job.h; ++r) {
    for (int c = 0; c < job.w; ++c) {
      out[static_cast<std::size_t>(r) * static_cast<std::size_t>(job.w) +
          static_cast<std::size_t>(c)] =
          static_cast<std::uint8_t>(me_sample_at(
              ref_p, job.x + c, job.y + r, job.mv_x, job.mv_y, job.interp));
    }
  }
  return MeStatus::kOk;
}

int me_collect_mv_candidates(const MeCoreMvField& field, int x, int y, int w,
                             int h, int default_ref, MeMvCandidate* out,
                             int max_out) {
  if (out == nullptr || max_out <= 0 || x < 0 || y < 0 || w <= 0 || h <= 0) {
    return 0;
  }
  if (field.cols <= 0 || field.rows <= 0 || field.mv_x == nullptr ||
      field.mv_y == nullptr ||
      (field.stride > 0 && field.stride < field.cols)) {
    out[0].mv_x = 0;
    out[0].mv_y = 0;
    out[0].ref_index = static_cast<std::int16_t>(default_ref);
    return 1;
  }
  return me_collect_mv_candidates_core(field, x, y, w, h, default_ref, out,
                                       max_out);
}

}  // namespace av2enc::me
