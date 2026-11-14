#include "pipeline/video/video_gme.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "codec/inter/me/me.h"

namespace av2enc::video {
namespace {

namespace me = ::av2enc::me;
using inter::gme::GmBlockMv;
using inter::gme::GmEstimate;
using inter::gme::GmOptions;
using inter::gme::GmParams;
using inter::gme::GmType;
using inter::mc::RefFrame;

// 16x16 luma blocks give a dense, deterministic correspondence set at 360p
// (920 blocks) and stay well below GmOptions::max_points at 1080p.
constexpr int kGmBlockSize = 16;
constexpr int kGmGridStep = 16;

bool make_me_ref(const RefFrame& reference, me::MeCoreRef& out) {
  if (reference.plane[0].data == nullptr || reference.plane[1].data == nullptr ||
      reference.plane[2].data == nullptr) {
    return false;
  }
  const int luma_w = reference.mi_cols * 4;
  const int luma_h = reference.mi_rows * 4;
  out.plane[0] = me::MeCorePlane{reference.plane[0].data,
                                 reference.plane[0].stride, luma_w, luma_h};
  out.plane[1] = me::MeCorePlane{reference.plane[1].data,
                                 reference.plane[1].stride, luma_w / 2,
                                 luma_h / 2};
  out.plane[2] = me::MeCorePlane{reference.plane[2].data,
                                 reference.plane[2].stride, luma_w / 2,
                                 luma_h / 2};
  return true;
}

}  // namespace

GmOptions make_gm_options(const VideoInputFrame& source,
                          const VideoOptions& options) {
  GmOptions gm;
  gm.model = inter::gme::GmFitModel::kAffine;
  gm.min_points = 8;
  gm.max_points = 16384;
  gm.outlier_passes = 2;
  gm.outlier_threshold_q8 = 32;  // 4 luma samples, the component default
  gm.allow_rotzoom_degrade = true;
  gm.require_shear_valid = false;  // extended warp covers an invalid shear
  gm.quantize = true;
  gm.frame_width = static_cast<int>(source.width);
  gm.frame_height = static_cast<int>(source.height);
  // Reject a model that does not measurably beat identity on the integer
  // correspondences.  The estimator's own default is 1%; 2% keeps marginal
  // fits out of the header without hiding a real global pan.
  gm.min_gain_percent = 2;
  // The ME grid is spatially uniform, so weight each block equally.
  (void)options;
  return gm;
}

std::vector<std::vector<GmBlockMv>> collect_gm_block_mvs(
    const VideoInputFrame& source,
    std::span<const inter::mc::RefFrame> references,
    const VideoOptions& options, std::string& error) {
  std::vector<std::vector<GmBlockMv>> out;
  out.reserve(references.size());
  if (references.empty()) return out;

  const int frame_w = static_cast<int>(source.width);
  const int frame_h = static_cast<int>(source.height);
  const int chroma_w = frame_w / 2;
  const int chroma_h = frame_h / 2;
  const int mi_cols = references[0].mi_cols;
  const int mi_rows = references[0].mi_rows;
  const int ref_w = mi_cols * 4;
  const int ref_h = mi_rows * 4;

  me::MeCoreFrame cur;
  cur.width = frame_w;
  cur.height = frame_h;
  cur.plane[0] = me::MeCorePlane{source.y.data(), frame_w, frame_w, frame_h};
  cur.plane[1] =
      me::MeCorePlane{source.u.data(), chroma_w, chroma_w, chroma_h};
  cur.plane[2] =
      me::MeCorePlane{source.v.data(), chroma_w, chroma_w, chroma_h};

  std::vector<me::MeCoreJob> jobs;
  std::vector<std::array<int, 2>> centres;
  for (int y = 0; y + kGmBlockSize <= frame_h; y += kGmGridStep) {
    for (int x = 0; x + kGmBlockSize <= frame_w; x += kGmGridStep) {
      if (x + kGmBlockSize > ref_w || y + kGmBlockSize > ref_h) continue;
      me::MeCoreJob job;
      job.x = x;
      job.y = y;
      job.w = kGmBlockSize;
      job.h = kGmBlockSize;
      job.num_refs = 1;
      job.ref_indices[0] = 0;
      jobs.push_back(job);
      centres.push_back({x + kGmBlockSize / 2, y + kGmBlockSize / 2});
    }
  }
  if (jobs.size() < 8) {
    error = "global motion: too few 16x16 blocks for a fit";
    return out;
  }

  me::MeCoreOptions me_options;
  me_options.method = me::kMeDiamond;
  me_options.metric = me::kMeSad;
  me_options.range = std::max(1, std::min(options.me_range, 64));
  me_options.subpel_stages = me::kMeSubpelHalf | me::kMeSubpelQuarter;
  me_options.subpel_interp = me::kMeInterpRegular;
  me_options.use_predictors = 1;
  me_options.use_merge_candidates = 0;

  std::vector<me::MeCoreResult> results(jobs.size());
  for (const RefFrame& reference : references) {
    me::MeCoreRef ref;
    if (!make_me_ref(reference, ref)) {
      error = "global motion: reference plane is missing";
      return out;
    }
    if (me::me_search_batch(cur, &ref, 1, jobs.data(),
                            static_cast<int>(jobs.size()), me_options, nullptr,
                            results.data()) != me::MeStatus::kOk) {
      error = "global motion: ME block grid failed";
      return out;
    }
    std::vector<GmBlockMv> points;
    points.reserve(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
      const inter::mc::Mv mv =
          inter::mc::mv_from_me_1_16(results[i].mv_y, results[i].mv_x);
      GmBlockMv point;
      point.x = centres[i][0] * 8;
      point.y = centres[i][1] * 8;
      // mc::Mv is already in 1/8 luma samples, which is the GmBlockMv q8 unit.
      point.mv_x = mv.x;
      point.mv_y = mv.y;
      point.weight = 1;
      points.push_back(point);
    }
    out.push_back(std::move(points));
  }
  return out;
}

GmFramePlan make_gm_frame_plan(std::span<const GmEstimate> estimates,
                               std::span<const int> ref_order_hint,
                               std::span<const std::uint8_t> ref_frame_idx,
                               std::span<const int> frame_distance) {
  GmFramePlan plan;
  plan.num_refs = static_cast<int>(
      std::min<std::size_t>(estimates.size(), static_cast<std::size_t>(kGmMaxRefs)));
  for (int r = 0; r < plan.num_refs; ++r) {
    const std::size_t i = static_cast<std::size_t>(r);
    plan.params[i] = estimates[i].params;
    const bool zero_distance =
        i < frame_distance.size() && frame_distance[i] == 0;
    if (zero_distance || !estimates[i].diag.accepted) {
      plan.params[i] = inter::gme::make_identity_gm();
    }
    if (i < ref_order_hint.size()) plan.ref_order_hint[i] = ref_order_hint[i];
    if (i < ref_frame_idx.size()) plan.ref_frame_idx[i] = ref_frame_idx[i];
    if (i < frame_distance.size()) plan.frame_distance[i] = frame_distance[i];
    if (plan.params[i].type != GmType::kIdentity) {
      plan.use_global_motion = true;
    }
  }
  return plan;
}

bool estimate_gm_plan_cpu(const VideoInputFrame& source,
                          std::span<const inter::mc::RefFrame> references,
                          std::span<const int> ref_order_hint,
                          std::span<const std::uint8_t> ref_frame_idx,
                          std::span<const int> frame_distance,
                          const VideoOptions& options, GmFramePlan& plan,
                          std::string& error) {
  plan = GmFramePlan{};
  if (!options.enable_gme || references.empty()) return true;
  const std::vector<std::vector<GmBlockMv>> points =
      collect_gm_block_mvs(source, references, options, error);
  if (points.empty()) return error.empty() ? true : false;
  const GmOptions gm_options = make_gm_options(source, options);
  std::vector<GmEstimate> estimates;
  estimates.reserve(points.size());
  for (const std::vector<GmBlockMv>& ref_points : points) {
    estimates.push_back(
        inter::gme::estimate_gm_block_mvs(ref_points, gm_options));
  }
  plan = make_gm_frame_plan(estimates, ref_order_hint, ref_frame_idx,
                            frame_distance);
  return true;
}

bool gm_model_uses_warp(const inter::gme::GmParams& params) {
  // The decoder derives TransformationType from the coded model, not from the
  // signalled GmType (§6.19.7.x get_wmtype): a model whose linear part is the
  // identity is TRANSLATION (or IDENTITY) and is never warped.  A model whose
  // shear fails §7.13.3.21 has gm_params.invalid set, so is_warp_affine_block()
  // also declines it even though prediction still runs the extended warp.
  const bool identity_linear =
      params.p[2] == inter::gme::kGmOne && params.p[3] == 0 &&
      params.p[4] == 0 && params.p[5] == inter::gme::kGmOne;
  if (identity_linear) return false;
  return inter::gme::setup_shear_valid(params.p.data());
}

void gm_sub_block_warp_mv(const inter::gme::GmParams& params, int pixel_x,
                          int pixel_y, inter::mc::Mv* mv) {
  if (mv == nullptr) return;
  // get_sub_block_warp_mv(): centre of the 8x8 (bw = bh = TMVP_MI_SIZE = 8).
  const int center_x = pixel_x + 4;
  const int center_y = pixel_y + 4;
  const std::int64_t xc =
      static_cast<std::int64_t>(params.p[2] - inter::gme::kGmOne) * center_x +
      static_cast<std::int64_t>(params.p[3]) * center_y + params.p[0];
  const std::int64_t yc =
      static_cast<std::int64_t>(params.p[4]) * center_x +
      static_cast<std::int64_t>(params.p[5] - inter::gme::kGmOne) * center_y +
      params.p[1];
  // WARPEDMODEL_PREC_BITS (16) -> 1/8 pel (3 fractional bits), rounding half
  // away from zero exactly like ROUND_POWER_OF_TWO_SIGNED.
  mv->x = static_cast<int>(inter::gme::round2_signed64(
      std::clamp<std::int64_t>(xc, INT32_MIN, INT32_MAX), 13));
  mv->y = static_cast<int>(inter::gme::round2_signed64(
      std::clamp<std::int64_t>(yc, INT32_MIN, INT32_MAX), 13));
}

bool ref_mvs_field_selftest(std::string* error) {
  using inter::gme::GmParams;
  using inter::gme::GmType;
  using inter::gme::kGmOne;
  const auto fail = [error](const char* msg) {
    if (error != nullptr) *error = msg;
    return false;
  };
  const auto model = [](GmType type, std::array<std::int32_t, 6> p) {
    GmParams out;
    out.type = type;
    out.p = p;
    return out;
  };
  // Model classification: §6.19.7.x get_wmtype over the coded parameters.
  if (gm_model_uses_warp(model(GmType::kIdentity,
                               {0, 0, kGmOne, 0, 0, kGmOne}))) {
    return fail("identity model classified as warp");
  }
  if (gm_model_uses_warp(model(GmType::kRotZoom,
                               {100, -200, kGmOne, 0, 0, kGmOne}))) {
    return fail("translation-only model classified as warp");
  }
  const GmParams zoom =
      model(GmType::kRotZoom, {0, 0, kGmOne + 8192, 0, 0, kGmOne + 8192});
  if (!gm_model_uses_warp(zoom)) {
    return fail("zoom model classified as non-warp");
  }

  // Per-8x8 projection.  A pure +1/8 scale step (8192 = kGmOne / 8) makes the
  // expected 1/8-pel value of every 8x8 centre exact.
  inter::mc::Mv mv;
  gm_sub_block_warp_mv(zoom, 0, 0, &mv);  // centre (4, 4)
  if (mv.y != 4 || mv.x != 4) {
    return fail("zoom projection at origin is not (4, 4)");
  }
  gm_sub_block_warp_mv(zoom, 56, 32, &mv);  // centre (60, 36)
  if (mv.y != 36 || mv.x != 60) {
    return fail("zoom projection at (56, 32) is not (36, 60)");
  }
  // Translation is added after the linear part.
  const GmParams shifted =
      model(GmType::kRotZoom, {7 * 8192, -3 * 8192, kGmOne + 8192, 0, 0,
                               kGmOne + 8192});
  gm_sub_block_warp_mv(shifted, 0, 0, &mv);  // centre (4, 4)
  if (mv.y != 4 - 3 || mv.x != 4 + 7) {
    return fail("translated zoom projection is not (1, 11)");
  }
  // Rotation: p3 = +1/16, p4 = -1/16; the centre is (4, 4).
  const GmParams rot =
      model(GmType::kRotZoom, {0, 0, kGmOne, kGmOne / 16, -(kGmOne / 16),
                               kGmOne});
  gm_sub_block_warp_mv(rot, 0, 0, &mv);  // xc = 4096*4 = 16384 -> 2
  if (mv.x != 2 || mv.y != -2) {
    return fail("rotation projection is not (row -2, col 2)");
  }
  // Rounding is half away from zero: xc = 8192*1.5 = 12288 -> 1.5 -> 2.
  const GmParams half =
      model(GmType::kRotZoom, {0, 0, kGmOne, 12288, 0, kGmOne});
  gm_sub_block_warp_mv(half, 0, 0, &mv);  // centre (4, 4): 12288*4 = 49152
  if (mv.x != 6) {
    return fail("warp projection rounding is not half-away-from-zero");
  }
  return true;
}

}  // namespace av2enc::video
