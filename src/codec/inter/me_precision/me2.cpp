// AV2 inter motion estimation, extended — CPU reference implementation.
//
// The search lives in me2_core.h (shared with the CUDA kernel).  This file
// validates the host API and instantiates the core with the scalar cost
// functor from the original ME component.

#include "codec/inter/me_precision/me2.h"

#include <cstddef>

#include "codec/inter/me/me.h"

namespace av2enc::me2 {
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

bool search_options_valid(const me::MeCoreOptions& o) {
  if (o.method < me::kMeFull || o.method > me::kMeTestZone) {
    return false;
  }
  if (o.metric != me::kMeSad && o.metric != me::kMeSatd) {
    return false;
  }
  if (o.range < 0 || o.range > 4096 || o.integer_step < 1 ||
      o.max_iterations < 1 || o.subpel_iterations < 0 ||
      o.max_candidates < 0 || o.diamond_step < 0 || o.tz_start_step < 0) {
    return false;
  }
  if ((o.subpel_stages & ~(me::kMeSubpelHalf | me::kMeSubpelQuarter |
                           me::kMeSubpelEighth |
                           me::kMeSubpelSixteenth)) != 0) {
    return false;
  }
  if (o.subpel_interp < 0 || o.subpel_interp > me::kMeMaxInterpFilter) {
    return false;
  }
  if (o.use_predictors < 0 || o.use_predictors > 1 ||
      o.use_merge_candidates < 0 || o.use_merge_candidates > 1) {
    return false;
  }
  if (o.max_merge_candidates < 0 ||
      o.max_merge_candidates > me::kMeMaxMerge) {
    return false;
  }
  return true;
}

bool options_valid(const Me2Options& o) {
  if (!search_options_valid(o.search)) {
    return false;
  }
  if (!me2_precision_valid(o.frame_precision)) {
    return false;
  }
  if (o.enable_flex_mvres < 0 || o.enable_flex_mvres > 1) {
    return false;
  }
  if (o.precision_mask < 0 || o.precision_mask >= (1 << kMvPrecisionCount)) {
    return false;
  }
  if (o.refine_radius < 0 || o.refine_radius > 64 ||
      o.refine_step < 0 || o.refine_step > 16) {
    return false;
  }
  if (o.mv_rate_lambda < 0 || o.mv_rate_shift < 0 ||
      o.mv_rate_shift > 24) {
    return false;
  }
  if (o.count_precision_bins < 0 || o.count_precision_bins > 1) {
    return false;
  }
  return true;
}

bool mv_at_precision_bounds(const MeCorePlane& ref, int x, int y, int w,
                            int h, std::int32_t mv_x, std::int32_t mv_y) {
  if (w <= 0 || h <= 0 || x < 0 || y < 0 || w > ref.width - x ||
      h > ref.height - y) {
    return false;
  }
  const std::int32_t lo_x =
      -static_cast<std::int32_t>(x) * me::kSubpelOne - me::kSubpelMask;
  const std::int32_t hi_x =
      static_cast<std::int32_t>(ref.width - x - w) * me::kSubpelOne +
      me::kSubpelMask;
  const std::int32_t lo_y =
      -static_cast<std::int32_t>(y) * me::kSubpelOne - me::kSubpelMask;
  const std::int32_t hi_y =
      static_cast<std::int32_t>(ref.height - y - h) * me::kSubpelOne +
      me::kSubpelMask;
  return mv_x >= lo_x && mv_x <= hi_x && mv_y >= lo_y && mv_y <= hi_y;
}

}  // namespace

const char* me2_status_string(Me2Status status) {
  switch (status) {
    case Me2Status::kOk:
      return "ok";
    case Me2Status::kInvalidArgument:
      return "invalid argument";
    case Me2Status::kUnsupportedFormat:
      return "unsupported format";
  }
  return "unknown";
}

Me2Status me2_search_batch(const MeCoreFrame& cur, const MeCoreRef* refs,
                           int num_refs, const MeCoreJob* jobs, int num_jobs,
                           const Me2Options& options,
                           const MeCoreMvField* field, Me2Result* results) {
  if (!frame_valid(cur)) {
    return Me2Status::kInvalidArgument;
  }
  if (num_refs <= 0 || num_refs > me::kMeMaxRefs || refs == nullptr) {
    return Me2Status::kInvalidArgument;
  }
  for (int i = 0; i < num_refs; ++i) {
    if (!ref_valid(refs[i], cur)) {
      return Me2Status::kInvalidArgument;
    }
  }
  if (!options_valid(options)) {
    return Me2Status::kInvalidArgument;
  }
  if (num_jobs < 0 ||
      (num_jobs > 0 && (jobs == nullptr || results == nullptr))) {
    return Me2Status::kInvalidArgument;
  }
  if (field != nullptr) {
    if (field->cols < 0 || field->rows < 0 ||
        (field->cols > 0 &&
         (field->mv_x == nullptr || field->mv_y == nullptr)) ||
        (field->stride > 0 && field->stride < field->cols)) {
      return Me2Status::kInvalidArgument;
    }
  }
  for (int j = 0; j < num_jobs; ++j) {
    const MeCoreJob& job = jobs[j];
    if (job.x < 0 || job.y < 0 || job.x >= cur.width ||
        job.y >= cur.height || job.w < 1 || job.h < 1 ||
        job.w > me::kMeMaxBlockDim || job.h > me::kMeMaxBlockDim) {
      return Me2Status::kInvalidArgument;
    }
    if (job.num_refs < 1 || job.num_refs > me::kMeMaxRefs) {
      return Me2Status::kInvalidArgument;
    }
    for (int i = 0; i < job.num_refs; ++i) {
      const int ri = job.ref_indices[i];
      if (ri < 0 || ri >= num_refs) {
        return Me2Status::kInvalidArgument;
      }
    }
    if (job.num_pred < 0 || job.num_pred > me::kMeMaxPredictors) {
      return Me2Status::kInvalidArgument;
    }
  }

  const me::MeScalarCost cost;
  for (int j = 0; j < num_jobs; ++j) {
    results[j] = me2_search_core(cur, refs, jobs[j], options, field, cost);
  }
  return Me2Status::kOk;
}

Me2Status me2_cost_block(const MeCoreFrame& cur, const MeCoreRef& ref,
                         const Me2CostJob& job, std::int32_t* out_cost) {
  if (out_cost == nullptr || !me2_precision_valid(job.precision)) {
    return Me2Status::kInvalidArgument;
  }
  if (job.base.plane < 0 || job.base.plane > 2) {
    return Me2Status::kInvalidArgument;
  }
  if (!me2_mv_is_at_precision(job.base.mv_x, job.precision) ||
      !me2_mv_is_at_precision(job.base.mv_y, job.precision)) {
    return Me2Status::kInvalidArgument;
  }
  if (!mv_at_precision_bounds(ref.plane[job.base.plane], job.base.x,
                              job.base.y, job.base.w, job.base.h,
                              job.base.mv_x, job.base.mv_y)) {
    return Me2Status::kInvalidArgument;
  }
  const me::MeStatus status = me::me_cost_block(cur, ref, job.base, out_cost);
  return status == me::MeStatus::kOk ? Me2Status::kOk
                                     : Me2Status::kInvalidArgument;
}

Me2Status me2_interp_block(const MeCoreRef& ref, const me::MeInterpJob& job,
                           std::uint8_t* out) {
  const me::MeStatus status = me::me_interp_block(ref, job, out);
  return status == me::MeStatus::kOk ? Me2Status::kOk
                                     : Me2Status::kInvalidArgument;
}

}  // namespace av2enc::me2
