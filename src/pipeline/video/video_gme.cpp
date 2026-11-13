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

}  // namespace av2enc::video
