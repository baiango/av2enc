#include "compound.h"

#include <cstring>
#include <limits>

namespace av2enc::inter::adv {
namespace {

bool rect_ok(const BlockRect& r) { return r.w > 0 && r.h > 0 && r.w <= 64 && r.h <= 64; }

}  // namespace

void build_wedge_mask(int bsize, int wedge_sign, int wedge_index, std::uint8_t* out, int stride) {
  if (out == nullptr || !bsize_has_wedge(bsize)) {
    return;
  }
  const int w = tables::kWedgeBlockWidth[bsize];
  const int h = tables::kWedgeBlockHeight[bsize];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      out[i * stride + j] =
          static_cast<std::uint8_t>(wedge_mask_value(bsize, wedge_sign, wedge_index, i, j));
    }
  }
}

int wedge_index_from_angle_dist(int wedge_angle, int wedge_dist) {
  if (wedge_angle < 0 || wedge_angle >= kWedgeAngles || wedge_dist < 0 ||
      wedge_dist >= kNumWedgeDist) {
    return -1;
  }
  return tables::kWedgeAngleDist2Index[wedge_angle][wedge_dist];
}

bool wedge_index_signalable(int wedge_index) {
  for (int angle = 0; angle < kWedgeAngles; ++angle) {
    for (int dist = 0; dist < kNumWedgeDist; ++dist) {
      if (tables::kWedgeAngleDist2Index[angle][dist] == wedge_index) {
        return true;
      }
    }
  }
  return false;
}

bool apply_compound(const CompoundJob& job) {
  if (!rect_ok(job.rect) || job.dst == nullptr || job.dst_stride <= 0) {
    return false;
  }
  const bool needs_preds = !job.is_inter_intra;
  if (needs_preds &&
      (job.p0 == nullptr || job.p1 == nullptr || job.p0_stride <= 0 || job.p1_stride <= 0)) {
    return false;
  }
  if (job.is_inter_intra && (job.intra == nullptr || job.intra_stride <= 0)) {
    return false;
  }
  if (job.plane > 0 && job.sub_x == 0 && job.sub_y == 0) {
    return false;
  }
  if (!compound_type_valid(job.type, job.is_inter_intra != 0, job.type == CompoundType::kWedge,
                           job.bsize)) {
    return false;
  }
  if (job.type == CompoundType::kAverage && (job.cwp_weight < -4 || job.cwp_weight > 20)) {
    return false;
  }
  if ((job.type == CompoundType::kWedge) && !wedge_index_signalable(job.wedge_index)) {
    return false;
  }
  run_compound_job(job);
  return true;
}

std::int64_t block_sse(const std::uint8_t* a, int a_stride, int a_x, int a_y, const std::uint8_t* b,
                       int b_stride, int b_x, int b_y, int w, int h) {
  std::int64_t sse = 0;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int d = static_cast<int>(a[(a_y + i) * a_stride + a_x + j]) -
                    static_cast<int>(b[(b_y + i) * b_stride + b_x + j]);
      sse += static_cast<std::int64_t>(d * d);
    }
  }
  return sse;
}

std::optional<CompoundDecision> search_compound(const CompoundSearchInput& in) {
  if (!rect_ok(in.rect) || in.p0 == nullptr || in.p1 == nullptr || in.source == nullptr ||
      in.scratch == nullptr || in.p0_stride <= 0 || in.p1_stride <= 0 || in.source_stride <= 0 ||
      in.scratch_stride <= 0) {
    return std::nullopt;
  }

  CompoundJob job{};
  job.plane = 0;
  job.sub_x = 0;
  job.sub_y = 0;
  job.rect = in.rect;
  job.bsize = in.bsize;
  job.bit_depth = in.bit_depth;
  job.p0 = in.p0;
  job.p0_stride = in.p0_stride;
  job.p1 = in.p1;
  job.p1_stride = in.p1_stride;
  job.dst = in.scratch;
  job.dst_stride = in.scratch_stride;
  job.ref_start_x[0] = in.ref_start_x[0];
  job.ref_start_x[1] = in.ref_start_x[1];
  job.ref_start_y[0] = in.ref_start_y[0];
  job.ref_start_y[1] = in.ref_start_y[1];
  job.mi_cols = in.mi_cols;
  job.mi_rows = in.mi_rows;

  std::int64_t best_sse = std::numeric_limits<std::int64_t>::max();
  CompoundDecision best{};
  bool found = false;

  auto evaluate = [&](const CompoundDecision& candidate) {
    const std::int64_t sse = block_sse(in.scratch, in.scratch_stride, in.rect.x, in.rect.y,
                                       in.source, in.source_stride, 0, 0, in.rect.w, in.rect.h);
    if (!found || sse < best_sse) {
      best_sse = sse;
      best = candidate;
      best.sse = sse;
      found = true;
    }
  };

  if (in.enable_average) {
    if (in.imp_mask_eligible) {
      job.type = CompoundType::kAverage;
      job.imp_mask_eligible = 1;
      job.cwp_weight = kCwpEqual;
      run_compound_job(job);
      CompoundDecision d{};
      d.type = CompoundType::kAverage;
      d.cwp_weight = kCwpEqual;
      d.use_imp_mask = true;
      evaluate(d);
      job.imp_mask_eligible = 0;
    }
    const int count = in.enable_cwp ? in.cwp_count : 0;
    if (count == 0) {
      job.type = CompoundType::kAverage;
      job.cwp_weight = kCwpEqual;
      run_compound_job(job);
      CompoundDecision d{};
      d.type = CompoundType::kAverage;
      d.cwp_weight = kCwpEqual;
      evaluate(d);
    }
    for (int c = 0; c < count; ++c) {
      const int cw = in.cwp_count > 0 ? in.cwp_candidates[c] : kCwpEqual;
      if (cw == kCwpEqual && in.imp_mask_eligible) {
        continue;  // already evaluated with the implicit mask
      }
      if (cw < -4 || cw > 20) {
        continue;
      }
      job.type = CompoundType::kAverage;
      job.cwp_weight = cw;
      run_compound_job(job);
      CompoundDecision d{};
      d.type = CompoundType::kAverage;
      d.cwp_weight = cw;
      evaluate(d);
    }
  }

  if (in.enable_wedge && bsize_has_wedge(in.bsize)) {
    job.type = CompoundType::kWedge;
    const int sign_begin = in.wedge_sign < 0 ? 0 : in.wedge_sign;
    const int sign_end = in.wedge_sign < 0 ? 1 : in.wedge_sign;
    for (int sign = sign_begin; sign <= sign_end; ++sign) {
      for (int index = 0; index < kWedgeTypes; ++index) {
        if (!wedge_index_signalable(index)) {
          continue;
        }
        job.wedge_sign = sign;
        job.wedge_index = index;
        run_compound_job(job);
        CompoundDecision d{};
        d.type = CompoundType::kWedge;
        d.wedge_sign = sign;
        d.wedge_index = index;
        evaluate(d);
      }
    }
  }

  if (in.enable_diffwtd) {
    job.type = CompoundType::kDiffWtd;
    for (int mask_type = 0; mask_type < 2; ++mask_type) {
      job.mask_type = mask_type;
      run_compound_job(job);
      CompoundDecision d{};
      d.type = CompoundType::kDiffWtd;
      d.mask_type = mask_type;
      evaluate(d);
    }
  }

  if (!found) {
    return std::nullopt;
  }
  return best;
}

bool apply_inter_intra(const InterIntraSearchInput& in, InterIntraMode mode, bool wedge_interintra,
                       int wedge_index) {
  if (!rect_ok(in.rect) || in.inter_pred == nullptr || in.scratch == nullptr ||
      in.inter_pred_stride <= 0 || in.scratch_stride <= 0) {
    return false;
  }
  const std::uint8_t* intra = in.intra[static_cast<int>(mode)];
  if (intra == nullptr || in.intra_stride <= 0) {
    return false;
  }
  if (wedge_interintra && (!bsize_has_wedge(in.bsize) || in.wedge_sign < 0 || in.wedge_sign > 1)) {
    return false;
  }

  // The blend consumes the inter prediction in place.
  for (int i = 0; i < in.rect.h; ++i) {
    std::memcpy(
        in.scratch + static_cast<std::ptrdiff_t>(in.rect.y + i) * in.scratch_stride + in.rect.x,
        in.inter_pred + static_cast<std::ptrdiff_t>(i) * in.inter_pred_stride,
        static_cast<std::size_t>(in.rect.w));
  }

  CompoundJob job{};
  job.plane = 0;
  job.rect = in.rect;
  job.bsize = in.bsize;
  job.is_inter_intra = 1;
  job.interintra_mode = mode;
  job.type = wedge_interintra ? CompoundType::kWedge : CompoundType::kIntra;
  job.wedge_index = wedge_index;
  job.wedge_sign = in.wedge_sign;
  job.intra = intra;
  job.intra_stride = in.intra_stride;
  job.dst = in.scratch;
  job.dst_stride = in.scratch_stride;
  run_compound_job(job);
  return true;
}

std::optional<InterIntraDecision> search_inter_intra(const InterIntraSearchInput& in) {
  if (!rect_ok(in.rect) || in.source == nullptr || in.source_stride <= 0) {
    return std::nullopt;
  }
  std::int64_t best_sse = std::numeric_limits<std::int64_t>::max();
  InterIntraDecision best{};
  bool found = false;

  auto evaluate = [&](InterIntraMode mode, bool wedge, int index) {
    if (!apply_inter_intra(in, mode, wedge, index)) {
      return;
    }
    const std::int64_t sse = block_sse(in.scratch, in.scratch_stride, in.rect.x, in.rect.y,
                                       in.source, in.source_stride, 0, 0, in.rect.w, in.rect.h);
    if (!found || sse < best_sse) {
      best_sse = sse;
      best.mode = mode;
      best.wedge_interintra = wedge;
      best.wedge_index = index;
      best.sse = sse;
      found = true;
    }
  };

  for (int m = 0; m < kInterIntraModes; ++m) {
    evaluate(static_cast<InterIntraMode>(m), false, 0);
  }
  if (in.enable_wedge_interintra && bsize_has_wedge(in.bsize)) {
    for (int index = 0; index < kWedgeTypes; ++index) {
      if (!wedge_index_signalable(index)) {
        continue;
      }
      for (int m = 0; m < kInterIntraModes; ++m) {
        evaluate(static_cast<InterIntraMode>(m), true, index);
      }
    }
  }
  if (!found) {
    return std::nullopt;
  }
  return best;
}

}  // namespace av2enc::inter::adv
