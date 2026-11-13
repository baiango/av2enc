// CPU-emulation backend for the CUDA M2 P-frame pipeline.
//
// Used on machines without nvcc (macOS dev hosts) and as a decision oracle.
// It implements the same backend contract as video_backend_cuda.cu using the CPU
// component references, and the shared inline code in video_device.cuh for the
// M2-owned arithmetic.  Launch counters report the number of *equivalent*
// device calls so a harness can compare pipeline shape, not speed.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "codec/inter/mc/mc.h"
#include "codec/inter/me/me.h"
#include "codec/inter/multiref/compound_predict.h"
#include "codec/intra/modes/intra_modes.h"
#include "pipeline/video/video_frame.h"
#include "codec/transform/variants/tx_variants.h"

#include "video_backend.h"
#include "video_device.cuh"

namespace av2enc::cuda::video::detail {
namespace {

using av2enc::inter::mc::BlockRect;
using av2enc::inter::mc::InterpFilter;
using av2enc::inter::mc::McStatus;
using av2enc::inter::mc::MutablePlane;
using av2enc::inter::mc::Mv;
using av2enc::inter::mc::RefFrame;

namespace multiref = ::av2enc::video::multiref;

int source_sample(const av2enc::video::VideoInputFrame& frame, int plane, int x,
                  int y) {
  const int w = plane == 0 ? static_cast<int>(frame.width)
                           : static_cast<int>((frame.width + 1U) / 2U);
  const int h = plane == 0 ? static_cast<int>(frame.height)
                           : static_cast<int>((frame.height + 1U) / 2U);
  const int cx = std::clamp(x, 0, w - 1);
  const int cy = std::clamp(y, 0, h - 1);
  const std::span<const std::uint8_t> data =
      plane == 0 ? frame.y : (plane == 1 ? frame.u : frame.v);
  return data[static_cast<std::size_t>(cy) * static_cast<std::size_t>(w) +
              static_cast<std::size_t>(cx)];
}

struct EmuPlane {
  int width = 0;
  int height = 0;
  int stride = 0;
  std::vector<std::uint8_t> samples;

  void resize(int w, int h, std::uint8_t value) {
    width = w;
    height = h;
    stride = w;
    samples.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h),
                   value);
  }
  std::uint8_t* row(int y) {
    return samples.data() + static_cast<std::size_t>(y) *
                                static_cast<std::size_t>(stride);
  }
  const std::uint8_t* row(int y) const {
    return samples.data() + static_cast<std::size_t>(y) *
                                static_cast<std::size_t>(stride);
  }
};

bool tx_plane_cpu(const av2enc::video::VideoInputFrame& source, int plane, int x0,
                  int y0, int w, int h,
                  const std::vector<std::uint8_t>& pred, PlaneTx& out,
                  std::string& error) {
  const auto params_opt =
      av2enc::txv::make_tx_params(w, h, av2enc::txv::TxType::kDctDct);
  if (!params_opt.has_value()) {
    error = "unsupported transform size";
    return false;
  }
  const av2enc::txv::TxParams& params = *params_opt;
  const std::size_t coded =
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  std::vector<std::int16_t> residual_coded(coded);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int src = source_sample(source, plane, x0 + x, y0 + y);
      const int p = pred[static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(w) +
                         static_cast<std::size_t>(x)];
      residual_coded[static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(w) +
                     static_cast<std::size_t>(x)] =
          device::residual_sample(src, p);
    }
  }
  std::vector<std::int16_t> residual_adj(
      static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
  av2enc::txv::downsample_residual(residual_coded.data(), params,
                                   residual_adj.data());
  std::vector<double> coeff(
      static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
  out.levels.assign(coeff.size(), 0);
  av2enc::txv::forward_transform(residual_adj.data(), params, coeff.data(),
                                 out.levels.data());
  std::vector<std::int32_t> dequant(coeff.size());
  std::vector<std::int16_t> recon_residual(coded);
  av2enc::txv::inverse_transform(out.levels.data(), params, dequant.data(),
                                 recon_residual.data());
  out.recon.resize(coded);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int p = pred[static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(w) +
                         static_cast<std::size_t>(x)];
      const int r = recon_residual[static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(w) +
                                   static_cast<std::size_t>(x)];
      out.recon[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(x)] = device::recon_sample(p, r);
    }
  }
  return true;
}

class CpuBackend final : public Backend {
 public:
  const char* name() const override { return "cpu-emulation"; }

  bool begin_frame(
      const av2enc::video::VideoInputFrame& source, int mi_cols, int mi_rows,
      std::span<const av2enc::video::VideoReconFrame* const> references,
      std::string& error) override {
    (void)error;
    begin_time_ = std::chrono::steady_clock::now();
    source_ = source;
    mi_cols_ = mi_cols;
    mi_rows_ = mi_rows;
    width_ = static_cast<int>(source.width);
    height_ = static_cast<int>(source.height);
    const int cw = width_ / 2;
    const int ch = height_ / 2;
    recon_[0].resize(mi_cols_ * 4, mi_rows_ * 4, 0);
    recon_[1].resize(mi_cols_ * 2, mi_rows_ * 2, 0);
    recon_[2].resize(mi_cols_ * 2, mi_rows_ * 2, 0);
    stats_.bytes_uploaded +=
        static_cast<std::uint64_t>(width_) * static_cast<std::uint64_t>(height_) +
        2ULL * static_cast<std::uint64_t>(cw) * static_cast<std::uint64_t>(ch);
    references_.clear();
    ref_frames_.clear();
    references_.reserve(references.size());
    ref_frames_.reserve(references.size());
    for (const av2enc::video::VideoReconFrame* reference : references) {
      if (reference == nullptr) {
        error = "CPU emulation reference list contains null";
        return false;
      }
      references_.push_back(*reference);
      RefFrame ref;
      ref.plane[0] = av2enc::inter::mc::PlaneView{
          references_.back().y.data(),
          static_cast<int>(references_.back().width)};
      ref.plane[1] = av2enc::inter::mc::PlaneView{
          references_.back().u.data(),
          static_cast<int>((references_.back().width + 1U) / 2U)};
      ref.plane[2] = av2enc::inter::mc::PlaneView{
          references_.back().v.data(),
          static_cast<int>((references_.back().width + 1U) / 2U)};
      ref.mi_cols = mi_cols_;
      ref.mi_rows = mi_rows_;
      ref_frames_.push_back(ref);
    }
    return true;
  }

  bool end_frame(std::string& error) override {
    (void)error;
    const auto now = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(now - begin_time_).count();
    stats_.device_ms += ms;
    stats_.frame_ms.push_back(ms);
    return true;
  }

  bool variance_costs(const std::vector<VarianceNode>& nodes,
                      std::vector<std::int64_t>& costs,
                      std::string& error) override {
    (void)error;
    costs.resize(nodes.size());
    const std::uint8_t* luma = source_.y.data();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      costs[i] = device::block_variance_cost(luma, width_, width_, height_,
                                             nodes[i].x, nodes[i].y,
                                             nodes[i].w, nodes[i].h);
    }
    ++stats_.variance_launches;
    return true;
  }

  bool me_batch(const std::vector<MeRequest>& jobs,
                const me::MeCoreOptions& options,
                std::vector<MeOutcome>& results, std::string& error) override {
    results.assign(jobs.size(), MeOutcome{});
    if (jobs.empty()) return true;
    me::MeCoreFrame cur;
    cur.width = width_;
    cur.height = height_;
    cur.plane[0] = me::MeCorePlane{source_.y.data(), width_, width_, height_};
    cur.plane[1] = me::MeCorePlane{source_.u.data(), width_ / 2, width_ / 2,
                                   height_ / 2};
    cur.plane[2] = me::MeCorePlane{source_.v.data(), width_ / 2, width_ / 2,
                                   height_ / 2};
    std::vector<me::MeCoreRef> refs(ref_frames_.size());
    for (std::size_t r = 0; r < ref_frames_.size(); ++r) {
      refs[r].plane[0] =
          me::MeCorePlane{ref_frames_[r].plane[0].data,
                          ref_frames_[r].plane[0].stride, mi_cols_ * 4,
                          mi_rows_ * 4};
      refs[r].plane[1] =
          me::MeCorePlane{ref_frames_[r].plane[1].data,
                          ref_frames_[r].plane[1].stride, mi_cols_ * 2,
                          mi_rows_ * 2};
      refs[r].plane[2] =
          me::MeCorePlane{ref_frames_[r].plane[2].data,
                          ref_frames_[r].plane[2].stride, mi_cols_ * 2,
                          mi_rows_ * 2};
    }
    if (!jobs.empty() && refs.empty()) {
      error = "CPU emulation ME batch has no reference";
      return false;
    }

    std::vector<me::MeCoreJob> me_jobs(jobs.size());
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      me_jobs[i] = me::MeCoreJob{};
      me_jobs[i].x = jobs[i].x;
      me_jobs[i].y = jobs[i].y;
      me_jobs[i].w = jobs[i].w;
      me_jobs[i].h = jobs[i].h;
      me_jobs[i].num_refs = 1;
      me_jobs[i].ref_indices[0] = static_cast<std::int8_t>(jobs[i].ref_index);
    }
    std::vector<me::MeCoreResult> me_results(jobs.size());
    const me::MeStatus status =
        me::me_search_batch(cur, refs.data(), static_cast<int>(refs.size()),
                            me_jobs.data(), static_cast<int>(me_jobs.size()),
                            options, nullptr, me_results.data());
    if (status != me::MeStatus::kOk) {
      error = "CPU emulation ME batch failed";
      return false;
    }
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      results[i].mv_y = me_results[i].mv_y;
      results[i].mv_x = me_results[i].mv_x;
    }
    ++stats_.me_launches;
    stats_.me_jobs += jobs.size();
    return true;
  }

  bool inter_sse_batch(const std::vector<InterJob>& jobs,
                       std::vector<std::int64_t>& sse,
                       std::string& error) override {
    sse.resize(jobs.size());
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      const InterJob& job = jobs[i];
      std::vector<std::uint8_t> pred(
          static_cast<std::size_t>(job.w) * static_cast<std::size_t>(job.h));
      const BlockRect rect{job.x, job.y, job.w, job.h};
      if (job.ref_index < 0 ||
          static_cast<std::size_t>(job.ref_index) >= ref_frames_.size()) {
        error = "CPU emulation MC ref_index out of range";
        return false;
      }
      const McStatus status = av2enc::inter::mc::predict_plane(
          ref_frames_[static_cast<std::size_t>(job.ref_index)], 0, rect,
          Mv{job.mv_y, job.mv_x}, InterpFilter::kEighttap, pred.data(), job.w);
      if (status != McStatus::kOk) {
        error = "CPU emulation MC luma failed";
        return false;
      }
      sse[i] = device::sse_against_source(
          source_.y.data(), width_, width_, height_, job.x, job.y, pred.data(),
          job.w, job.w, job.h);
    }
    ++stats_.mc_launches;
    ++stats_.sse_launches;
    stats_.inter_candidates += jobs.size();
    return true;
  }

  bool mc_420(const InterJob& luma_job, std::vector<std::uint8_t>& y,
              std::vector<std::uint8_t>& u, std::vector<std::uint8_t>& v,
              std::string& error) override {
    const int w = luma_job.w;
    const int h = luma_job.h;
    const int cw = w / 2;
    const int ch = h / 2;
    y.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    MutablePlane dst[3];
    dst[0] = MutablePlane{y.data(), w};
    dst[1] = MutablePlane{u.data(), cw};
    dst[2] = MutablePlane{v.data(), cw};
    const BlockRect rect{luma_job.x, luma_job.y, w, h};
    if (luma_job.ref_index < 0 ||
        static_cast<std::size_t>(luma_job.ref_index) >= ref_frames_.size()) {
      error = "CPU emulation MC ref_index out of range";
      return false;
    }
    if (av2enc::inter::mc::predict_block_420(
            ref_frames_[static_cast<std::size_t>(luma_job.ref_index)], rect,
            Mv{luma_job.mv_y, luma_job.mv_x},
            InterpFilter::kEighttap, dst) != McStatus::kOk) {
      error = "CPU emulation MC 4:2:0 failed";
      return false;
    }
    write_plane_block(recon_[0], luma_job.x, luma_job.y, w, h, y);
    write_plane_block(recon_[1], luma_job.x / 2, luma_job.y / 2, cw, ch, u);
    write_plane_block(recon_[2], luma_job.x / 2, luma_job.y / 2, cw, ch, v);
    ++stats_.mc_launches;
    stats_.bytes_downloaded +=
        static_cast<std::uint64_t>(y.size() + u.size() + v.size());
    return true;
  }

  bool compound_sse_batch(const std::vector<InterJob>& jobs,
                          std::vector<std::int64_t>& sse,
                          std::string& error) override {
    sse.assign(jobs.size(), 0);
    if (jobs.empty()) return true;
    if (ref_frames_.size() < 2) {
      error = "CPU emulation compound prediction needs two references";
      return false;
    }
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      const InterJob& job = jobs[i];
      multiref::CompoundPrediction prediction;
      const BlockRect rect{job.x, job.y, job.w, job.h};
      if (!multiref::build_compound_prediction(
              ref_frames_[0], ref_frames_[1], rect, 0, Mv{0, 0}, Mv{0, 0},
              InterpFilter::kEighttap, mi_cols_, mi_rows_, prediction,
              error)) {
        return false;
      }
      multiref::CompoundModeParams params;
      params.type = av2enc::inter::adv::CompoundType::kAverage;
      params.cwp_weight = multiref::kCwpEqual;
      std::vector<std::uint8_t> scratch(
          static_cast<std::size_t>(job.w) * static_cast<std::size_t>(job.h));
      sse[i] = multiref::compound_luma_sse(
          prediction, params,
          source_.y.data() + static_cast<std::size_t>(job.y) *
                                 static_cast<std::size_t>(width_) +
              static_cast<std::size_t>(job.x),
          width_, scratch.data(), job.w);
    }
    ++stats_.mc_launches;
    ++stats_.sse_launches;
    stats_.inter_candidates += jobs.size();
    return true;
  }

  bool compound_420(const InterJob& luma_job, std::vector<std::uint8_t>& y,
                    std::vector<std::uint8_t>& u, std::vector<std::uint8_t>& v,
                    std::string& error) override {
    if (ref_frames_.size() < 2) {
      error = "CPU emulation compound prediction needs two references";
      return false;
    }
    const int w = luma_job.w;
    const int h = luma_job.h;
    const int cw = w / 2;
    const int ch = h / 2;
    multiref::CompoundPrediction prediction;
    const BlockRect rect{luma_job.x, luma_job.y, w, h};
    if (!multiref::build_compound_prediction(
            ref_frames_[0], ref_frames_[1], rect, 0, Mv{0, 0}, Mv{0, 0},
            InterpFilter::kEighttap, mi_cols_, mi_rows_, prediction, error)) {
      return false;
    }
    multiref::CompoundModeParams params;
    params.type = av2enc::inter::adv::CompoundType::kAverage;
    params.cwp_weight = multiref::kCwpEqual;
    std::uint8_t* dst[3] = {recon_[0].samples.data(), recon_[1].samples.data(),
                            recon_[2].samples.data()};
    int dst_stride[3] = {recon_[0].stride, recon_[1].stride, recon_[2].stride};
    if (!multiref::apply_compound_prediction(prediction, params, dst,
                                             dst_stride, error)) {
      return false;
    }
    y.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    v.resize(u.size());
    for (int row = 0; row < h; ++row) {
      std::memcpy(y.data() + static_cast<std::size_t>(row) *
                                 static_cast<std::size_t>(w),
                  recon_[0].samples.data() +
                      static_cast<std::size_t>(luma_job.y + row) *
                          static_cast<std::size_t>(recon_[0].stride) +
                      static_cast<std::size_t>(luma_job.x),
                  static_cast<std::size_t>(w));
    }
    for (int row = 0; row < ch; ++row) {
      std::memcpy(u.data() + static_cast<std::size_t>(row) *
                                 static_cast<std::size_t>(cw),
                  recon_[1].samples.data() +
                      static_cast<std::size_t>(luma_job.y / 2 + row) *
                          static_cast<std::size_t>(recon_[1].stride) +
                      static_cast<std::size_t>(luma_job.x / 2),
                  static_cast<std::size_t>(cw));
      std::memcpy(v.data() + static_cast<std::size_t>(row) *
                                 static_cast<std::size_t>(cw),
                  recon_[2].samples.data() +
                      static_cast<std::size_t>(luma_job.y / 2 + row) *
                          static_cast<std::size_t>(recon_[2].stride) +
                      static_cast<std::size_t>(luma_job.x / 2),
                  static_cast<std::size_t>(cw));
    }
    ++stats_.mc_launches;
    stats_.bytes_downloaded +=
        static_cast<std::uint64_t>(y.size() + u.size() + v.size());
    return true;
  }

  bool intra_candidate(int x0, int y0, int w, int h,
                       std::array<PlaneTx, 3>& planes,
                       std::string& error) override {
    const int coords[3][4] = {
        {x0, y0, w, h},
        {x0 / 2, y0 / 2, w / 2, h / 2},
        {x0 / 2, y0 / 2, w / 2, h / 2}};
    for (int plane = 0; plane < 3; ++plane) {
      const EmuPlane& recon = recon_[plane];
      std::vector<std::uint8_t> pred(
          static_cast<std::size_t>(coords[plane][2]) *
          static_cast<std::size_t>(coords[plane][3]));
      av2enc::intra::NeighborRequest request;
      request.recon =
          av2enc::intra::PlaneView{recon.samples.data(), recon.stride};
      request.x = coords[plane][0];
      request.y = coords[plane][1];
      request.w = coords[plane][2];
      request.h = coords[plane][3];
      request.max_x = recon.width - 1;
      request.max_y = recon.height - 1;
      request.have_left = request.x > 0;
      request.have_above = request.y > 0;
      request.num4_above_right = 0;
      request.num4_below_left = 0;
      request.color_plane = plane;
      av2enc::intra::PredictParams params;
      params.mode = av2enc::intra::Mode::kDc;
      params.angle_delta = 0;
      params.enable_edge_filter = false;
      if (av2enc::intra::predict_block(
              request, 8, params, pred.data(),
              static_cast<std::size_t>(coords[plane][2])) !=
          av2enc::intra::IntraStatus::kOk) {
        error = "CPU emulation intra DC prediction failed";
        return false;
      }
      if (!tx_plane_cpu(source_, plane, coords[plane][0], coords[plane][1],
                        coords[plane][2], coords[plane][3], pred,
                        planes[static_cast<std::size_t>(plane)], error)) {
        return false;
      }
    }
    stats_.intra_launches += 1;
    stats_.tx_launches += 2;  // forward + inverse, batched per frame in CUDA
    return true;
  }

  bool write_recon_420(int x0, int y0, int w, int h,
                       const std::vector<std::uint8_t>& y,
                       const std::vector<std::uint8_t>& u,
                       const std::vector<std::uint8_t>& v,
                       std::string& error) override {
    (void)error;
    write_plane_block(recon_[0], x0, y0, w, h, y);
    write_plane_block(recon_[1], x0 / 2, y0 / 2, w / 2, h / 2, u);
    write_plane_block(recon_[2], x0 / 2, y0 / 2, w / 2, h / 2, v);
    ++stats_.recon_launches;
    stats_.bytes_uploaded +=
        static_cast<std::uint64_t>(y.size() + u.size() + v.size());
    return true;
  }

  bool inter_residual_candidate(const InterJob& job, PlaneTx& out,
                                std::string& error) override {
    std::vector<std::uint8_t> pred(
        static_cast<std::size_t>(job.w) * static_cast<std::size_t>(job.h));
    const BlockRect rect{job.x, job.y, job.w, job.h};
    if (job.ref_index < 0 ||
        static_cast<std::size_t>(job.ref_index) >= ref_frames_.size()) {
      error = "CPU emulation inter residual ref_index out of range";
      return false;
    }
    if (av2enc::inter::mc::predict_plane(
            ref_frames_[static_cast<std::size_t>(job.ref_index)], 0, rect,
            Mv{job.mv_y, job.mv_x}, InterpFilter::kEighttap, pred.data(),
            job.w) != McStatus::kOk) {
      error = "CPU emulation inter residual MC failed";
      return false;
    }
    if (!tx_plane_cpu(source_, 0, job.x, job.y, job.w, job.h, pred, out,
                      error)) {
      return false;
    }
    ++stats_.mc_launches;
    ++stats_.tx_launches;
    return true;
  }

  bool inter_residual_420(const InterJob& luma_job,
                          std::array<PlaneTx, 3>& planes,
                          std::string& error) override {
    const int w = luma_job.w;
    const int h = luma_job.h;
    const int cw = w / 2;
    const int ch = h / 2;
    std::vector<std::uint8_t> y(static_cast<std::size_t>(w) *
                                static_cast<std::size_t>(h));
    std::vector<std::uint8_t> u(static_cast<std::size_t>(cw) *
                                static_cast<std::size_t>(ch));
    std::vector<std::uint8_t> v(u.size());
    MutablePlane dst[3];
    dst[0] = MutablePlane{y.data(), w};
    dst[1] = MutablePlane{u.data(), cw};
    dst[2] = MutablePlane{v.data(), cw};
    const BlockRect rect{luma_job.x, luma_job.y, w, h};
    if (luma_job.ref_index < 0 ||
        static_cast<std::size_t>(luma_job.ref_index) >= ref_frames_.size()) {
      error = "CPU emulation MC ref_index out of range";
      return false;
    }
    if (av2enc::inter::mc::predict_block_420(
            ref_frames_[static_cast<std::size_t>(luma_job.ref_index)], rect,
            Mv{luma_job.mv_y, luma_job.mv_x},
            InterpFilter::kEighttap, dst) != McStatus::kOk) {
      error = "CPU emulation inter residual MC 4:2:0 failed";
      return false;
    }
    const std::vector<std::uint8_t>* preds[3] = {&y, &u, &v};
    const int coords[3][4] = {
        {luma_job.x, luma_job.y, w, h},
        {luma_job.x / 2, luma_job.y / 2, cw, ch},
        {luma_job.x / 2, luma_job.y / 2, cw, ch}};
    for (int plane = 0; plane < 3; ++plane) {
      if (!tx_plane_cpu(source_, plane, coords[plane][0], coords[plane][1],
                        coords[plane][2], coords[plane][3],
                        *preds[static_cast<std::size_t>(plane)],
                        planes[static_cast<std::size_t>(plane)], error)) {
        return false;
      }
    }
    ++stats_.mc_launches;
    stats_.tx_launches += 3;
    return true;
  }

  bool debug_set_recon(int plane, const std::vector<std::uint8_t>& data,
                       std::string& error) override {
    if (plane < 0 || plane > 2 ||
        data.size() != recon_[plane].samples.size()) {
      error = "debug_set_recon: bad plane or size";
      return false;
    }
    recon_[plane].samples = data;
    return true;
  }

  const VideoGpuStats& stats() const override { return stats_; }
  void reset_stats() override {
    stats_ = VideoGpuStats{};
    stats_.frame_bytes_from_cpu = true;
  }

 private:
  static void write_plane_block(EmuPlane& plane, int x0, int y0, int w, int h,
                                const std::vector<std::uint8_t>& samples) {
    for (int y = 0; y < h; ++y) {
      const int py = y0 + y;
      if (py < 0 || py >= plane.height) continue;
      std::uint8_t* dst = plane.row(py);
      for (int x = 0; x < w; ++x) {
        const int px = x0 + x;
        if (px < 0 || px >= plane.width) continue;
        dst[px] = samples[static_cast<std::size_t>(y) *
                              static_cast<std::size_t>(w) +
                          static_cast<std::size_t>(x)];
      }
    }
  }

  av2enc::video::VideoInputFrame source_{};
  std::vector<av2enc::video::VideoReconFrame> references_;
  std::vector<RefFrame> ref_frames_;
  EmuPlane recon_[3];
  int width_ = 0;
  int height_ = 0;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  std::chrono::steady_clock::time_point begin_time_{};
  VideoGpuStats stats_{};
};

}  // namespace

std::unique_ptr<Backend> make_backend(av2enc::cuda::Arena* /*arena*/) {
  return std::unique_ptr<Backend>(new CpuBackend());
}

const char* make_backend_name() { return "cpu-emulation"; }

}  // namespace av2enc::cuda::video::detail
