// AV2 inter motion estimation — CPU reference implementation.
//
// The search itself lives in me_core.h so the CUDA kernel can instantiate the
// identical state machine.  This file only validates the host API and loops
// over jobs with the scalar cost functor.

#include "codec/inter/me/me.h"

#include <cstddef>

#if defined(AV2ENC_PRIMITIVES)
#include "primitives/kernels.h"
#endif

namespace av2enc::me {
namespace {

// Host cost backed by the pixel-kernel layer (src/primitives/kernels.h).
//
// The integer-MV path is exactly the kernels' `sad` and `satd(k8x8)`: they are
// transcriptions of me_sad_rows() and me_cost_plane(kMeSatd) with the same
// accumulation order, tile order and (sum + 8) >> 4 rounding, so candidate
// costs, the search order and the tie-breaking are unchanged.  Fractional MVs
// keep the inline me_cost_plane() path: the layer has no exact slot for a
// subpel SAD (it would have to materialise the prediction first) and none of
// the subpel variants are interchangeable with the integer kernels.
//
// `AV2ENC_PRIMITIVES` is defined for the product av2enc_core target, which
// links src/primitives/kernels.cpp.  The per-component build_and_test.sh gates
// compile this TU on its own, so they keep the scalar MeScalarCost functor and
// stay bit-identical to the kernel table (the table is a transcription).
#if defined(AV2ENC_PRIMITIVES)
struct MeKernelCost {
  primitives::SadFn sad = nullptr;
  primitives::SatdFn satd = nullptr;

  explicit MeKernelCost(const primitives::KernelTable& kernels)
      : sad(kernels.sad), satd(kernels.satd) {}

  std::int32_t operator()(const MeCorePlane& cur, const MeCorePlane& ref,
                          int x, int y, int w, int h, std::int32_t mv_x,
                          std::int32_t mv_y, int metric, int filter) const {
    if (me_is_integer_mv(mv_x, mv_y)) {
      const int ix = x + (mv_x >> kSubpelBits);
      const int iy = y + (mv_y >> kSubpelBits);
      const std::uint8_t* cur_p =
          cur.data + static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(cur.stride) +
          static_cast<std::size_t>(x);
      const std::uint8_t* ref_p =
          ref.data + static_cast<std::size_t>(iy) *
                         static_cast<std::size_t>(ref.stride) +
          static_cast<std::size_t>(ix);
      if (metric == kMeSad) {
        return static_cast<std::int32_t>(
            sad(cur_p, static_cast<std::size_t>(cur.stride), ref_p,
                static_cast<std::size_t>(ref.stride), w, h));
      }
      return static_cast<std::int32_t>(
          satd(cur_p, static_cast<std::size_t>(cur.stride), ref_p,
               static_cast<std::size_t>(ref.stride), w, h,
               primitives::SatdTile::k8x8));
    }
    return me_cost_plane(cur, ref, x, y, w, h, mv_x, mv_y, metric, filter);
  }
};
#endif

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

#if defined(AV2ENC_PRIMITIVES)
  const MeKernelCost cost{primitives::kernels()};
#else
  const MeScalarCost cost;
#endif
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
#if defined(AV2ENC_PRIMITIVES)
  const MeKernelCost cost{primitives::kernels()};
#else
  const MeScalarCost cost;
#endif
  *out_cost = cost(cur_p, ref_p, job.x, job.y, job.w, job.h, job.mv_x,
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
