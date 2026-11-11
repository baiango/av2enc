// CUDA backend for the M2 P-frame pipeline.
//
// Reuses the validated device components:
//   src/cuda/inter/me        integer + subpel motion estimation
//   src/cuda/inter/mc        §7.13.3 translational motion compensation
//   src/cuda/intra_modes     DC intra prediction + residual
//   src/cuda/tx_variants     residual downsample / forward / quant / inverse
//
// Only four small kernels are M2-owned (block variance, SSE, residual and
// reconstruction); they call the inline code in m2_device.cuh, shared with the
// host-emulation backend so the arithmetic cannot drift.
//
// Allocation policy: device scratch is reserved once per frame size at
// begin_frame (a frame boundary, not a block loop) and reused.  A product
// integration should carve these buffers from src/cuda/arena.*; the standalone
// gate uses cudaMalloc so it does not depend on the CLI runtime.
//
// Synchronisation: every backend call returns host-visible data, so the stream
// is synchronized at each download.  One CUDA event pair per frame reports the
// GPU interval.  There is no hidden synchronisation anywhere else.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/inter/mc/mc.h"
#include "cuda/inter/me/me.h"
#include "cuda/intra_modes/intra_modes.h"
#include "cuda/tx_variants/tx_variants.h"
#include "encoder/m2/m2_frame.h"

#include "m2_backend.h"
#include "m2_device.cuh"

namespace av2enc::cuda::m2::detail {
namespace {

constexpr int kMaxBlockDim = 64;
constexpr int kMaxBlockPixels = kMaxBlockDim * kMaxBlockDim;
constexpr int kMaxPlaneTx = 1024;  // adjusted transform is at most 32x32
constexpr int kMaxCodedPixels = 4096;
constexpr int kPredSlots = 12;  // global + 6 near + new + final 4:2:0

bool check(cudaError_t err, const char* what, std::string& error) {
  if (err == cudaSuccess) return true;
  error = std::string(what) + ": " + cudaGetErrorString(err);
  return false;
}

int div_up(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

// Small POD job structs shared with the kernels.
struct SseJob {
  const std::uint8_t* src;
  int src_stride;
  int frame_w;
  int frame_h;
  const std::uint8_t* pred;
  int pred_stride;
  int x;
  int y;
  int w;
  int h;
  std::int64_t* out;
};

struct ReconJob {
  const std::uint8_t* pred;
  const std::int16_t* residual;
  std::uint8_t* out;
  int w;
  int h;
};

__global__ void m2_variance_kernel(const VarianceNode* __restrict__ nodes,
                                   int count,
                                   const std::uint8_t* __restrict__ src,
                                   int stride, int frame_w, int frame_h,
                                   std::int64_t* __restrict__ costs) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  const VarianceNode node = nodes[i];
  costs[i] = device::block_variance_cost(src, stride, frame_w, frame_h,
                                         node.x, node.y, node.w, node.h);
}

__global__ void m2_sse_kernel(const SseJob* __restrict__ jobs, int count) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  const SseJob job = jobs[i];
  *job.out = device::sse_against_source(
      job.src, job.src_stride, job.frame_w, job.frame_h, job.x, job.y,
      job.pred, job.pred_stride, job.w, job.h);
}

__global__ void m2_residual_kernel(const std::uint8_t* __restrict__ src,
                                   int src_stride, int frame_w, int frame_h,
                                   int x0, int y0,
                                   const std::uint8_t* __restrict__ pred,
                                   int pred_stride,
                                   std::int16_t* __restrict__ out, int w,
                                   int h) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= w * h) return;
  const int x = i % w;
  const int y = i / w;
  const int s = device::source_sample(src, src_stride, frame_w, frame_h,
                                      x0 + x, y0 + y);
  const int p = pred[static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(pred_stride) +
                     static_cast<std::size_t>(x)];
  out[i] = device::residual_sample(s, p);
}

__global__ void m2_recon_kernel(const ReconJob* __restrict__ jobs, int count) {
  const int job_index = static_cast<int>(blockIdx.x);
  if (job_index >= count) return;
  const ReconJob job = jobs[job_index];
  for (int i = static_cast<int>(threadIdx.x); i < job.w * job.h;
       i += static_cast<int>(blockDim.x)) {
    const int x = i % job.w;
    const int y = i / job.w;
    const int p = job.pred[static_cast<std::size_t>(y) *
                               static_cast<std::size_t>(job.w) +
                           static_cast<std::size_t>(x)];
    const int r = job.residual[static_cast<std::size_t>(y) *
                                   static_cast<std::size_t>(job.w) +
                               static_cast<std::size_t>(x)];
    job.out[i] = device::recon_sample(p, r);
  }
}

template <typename T>
bool ensure_device(T** ptr, std::size_t& capacity, std::size_t count,
                   std::string& error) {
  const std::size_t bytes = count * sizeof(T);
  if (bytes <= capacity) return true;
  if (*ptr != nullptr) {
    cudaFree(*ptr);
    *ptr = nullptr;
    capacity = 0;
  }
  void* raw = nullptr;
  if (!check(cudaMalloc(&raw, bytes), "cudaMalloc", error)) return false;
  *ptr = static_cast<T*>(raw);
  capacity = bytes;
  return true;
}

class CudaBackend final : public Backend {
 public:
  CudaBackend() {
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) ==
        cudaSuccess) {
      stream_ok_ = true;
    }
    cudaEventCreate(&start_event_);
    cudaEventCreate(&end_event_);
  }

  ~CudaBackend() override {
    if (stream_ok_) cudaStreamDestroy(stream_);
    if (start_event_ != nullptr) cudaEventDestroy(start_event_);
    if (end_event_ != nullptr) cudaEventDestroy(end_event_);
    free_all();
  }

  const char* name() const override { return "cuda"; }

  bool begin_frame(const av2enc::m2::M2InputFrame& source, int mi_cols,
                   int mi_rows, const av2enc::m2::M2ReconFrame* reference,
                   std::string& error) override {
    if (!stream_ok_) {
      error = "CUDA stream creation failed";
      return false;
    }
    width_ = static_cast<int>(source.width);
    height_ = static_cast<int>(source.height);
    chroma_w_ = width_ / 2;
    chroma_h_ = height_ / 2;
    mi_cols_ = mi_cols;
    mi_rows_ = mi_rows;
    luma_w_ = mi_cols_ * 4;
    luma_h_ = mi_rows_ * 4;
    chroma_mi_w_ = mi_cols_ * 2;
    chroma_mi_h_ = mi_rows_ * 2;

    if (luma_w_ != width_ || luma_h_ != height_ ||
        chroma_mi_w_ != chroma_w_ || chroma_mi_h_ != chroma_h_) {
      error = "MI-aligned planes differ from the visible frame; M2 requires "
              "dimensions divisible by 8";
      return false;
    }
    if (!ensure_buffers(error)) return false;

    if (!upload_2d(src_planes_[0], width_, source.y.data(), width_, width_,
                   height_, error) ||
        !upload_2d(src_planes_[1], chroma_w_, source.u.data(), chroma_w_,
                   chroma_w_, chroma_h_, error) ||
        !upload_2d(src_planes_[2], chroma_w_, source.v.data(), chroma_w_,
                   chroma_w_, chroma_h_, error)) {
      return false;
    }

    if (reference != nullptr) {
      if (reference->width != source.width ||
          reference->height != source.height) {
        error = "reference dimensions differ from the current frame";
        return false;
      }
      if (!upload_2d(ref_planes_[0], luma_w_, reference->y.data(), width_,
                     luma_w_, height_, error) ||
          !upload_2d(ref_planes_[1], chroma_mi_w_, reference->u.data(),
                     chroma_w_, chroma_mi_w_, chroma_h_, error) ||
          !upload_2d(ref_planes_[2], chroma_mi_w_, reference->v.data(),
                     chroma_w_, chroma_mi_w_, chroma_h_, error)) {
        return false;
      }
    }

    if (!check(cudaMemsetAsync(recon_planes_[0], 0,
                               static_cast<std::size_t>(luma_w_) *
                                   static_cast<std::size_t>(luma_h_),
                               stream_),
               "cudaMemsetAsync", error) ||
        !check(cudaMemsetAsync(recon_planes_[1], 0,
                               static_cast<std::size_t>(chroma_mi_w_) *
                                   static_cast<std::size_t>(chroma_mi_h_),
                               stream_),
               "cudaMemsetAsync", error) ||
        !check(cudaMemsetAsync(recon_planes_[2], 0,
                               static_cast<std::size_t>(chroma_mi_w_) *
                                   static_cast<std::size_t>(chroma_mi_h_),
                               stream_),
               "cudaMemsetAsync", error)) {
      return false;
    }
    if (start_event_ != nullptr) {
      cudaEventRecord(start_event_, stream_);
    }
    return true;
  }

  bool end_frame(std::string& error) override {
    if (end_event_ != nullptr && start_event_ != nullptr) {
      if (!check(cudaEventRecord(end_event_, stream_), "cudaEventRecord",
                 error)) {
        return false;
      }
      if (!check(cudaEventSynchronize(end_event_), "cudaEventSynchronize",
                 error)) {
        return false;
      }
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start_event_, end_event_);
      stats_.device_ms += static_cast<double>(ms);
      stats_.frame_ms.push_back(static_cast<double>(ms));
      ++stats_.syncs;
    }
    return true;
  }

  bool variance_costs(const std::vector<VarianceNode>& nodes,
                      std::vector<std::int64_t>& costs,
                      std::string& error) override {
    costs.assign(nodes.size(), 0);
    if (nodes.empty()) return true;
    if (!ensure_device(&d_variance_nodes_, variance_nodes_bytes_,
                       nodes.size(), error) ||
        !ensure_device(&d_variance_costs_, variance_costs_bytes_,
                       nodes.size(), error)) {
      return false;
    }
    upload(d_variance_nodes_, nodes.data(),
           nodes.size() * sizeof(VarianceNode));
    const int threads = 128;
    const int blocks = div_up(static_cast<int>(nodes.size()), threads);
    m2_variance_kernel<<<blocks, threads, 0, stream_>>>(
        d_variance_nodes_, static_cast<int>(nodes.size()), src_planes_[0],
        width_, width_, height_, d_variance_costs_);
    if (!check(cudaGetLastError(), "m2_variance_kernel", error)) return false;
    ++stats_.variance_launches;
    return check(download(costs.data(), d_variance_costs_,
                          nodes.size() * sizeof(std::int64_t)),
                 "variance download", error);
  }

  bool me_batch(const std::vector<MeRequest>& jobs,
                const me::MeCoreOptions& options,
                std::vector<MeOutcome>& results, std::string& error) override {
    results.assign(jobs.size(), MeOutcome{});
    if (jobs.empty()) return true;
    if (!ensure_device(&d_me_jobs_, me_jobs_bytes_, jobs.size(), error) ||
        !ensure_device(&d_me_results_, me_results_bytes_, jobs.size(), error) ||
        !ensure_device(&d_me_refs_, me_refs_bytes_, 1, error)) {
      return false;
    }
    std::vector<me::MeCoreJob> host_jobs(jobs.size());
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      host_jobs[i] = me::MeCoreJob{};
      host_jobs[i].x = jobs[i].x;
      host_jobs[i].y = jobs[i].y;
      host_jobs[i].w = jobs[i].w;
      host_jobs[i].h = jobs[i].h;
      host_jobs[i].num_refs = 1;
      host_jobs[i].ref_indices[0] = 0;
    }
    me::MeCoreRef host_ref;
    host_ref.plane[0] = me::MeCorePlane{ref_planes_[0], luma_w_, luma_w_,
                                        luma_h_};
    host_ref.plane[1] = me::MeCorePlane{ref_planes_[1], chroma_mi_w_,
                                        chroma_mi_w_, chroma_mi_h_};
    host_ref.plane[2] = me::MeCorePlane{ref_planes_[2], chroma_mi_w_,
                                        chroma_mi_w_, chroma_mi_h_};
    upload(d_me_jobs_, host_jobs.data(),
           host_jobs.size() * sizeof(me::MeCoreJob));
    upload(d_me_refs_, &host_ref, sizeof(me::MeCoreRef));

    me::MeCoreFrame cur;
    cur.width = width_;
    cur.height = height_;
    cur.plane[0] = me::MeCorePlane{src_planes_[0], width_, width_, height_};
    cur.plane[1] =
        me::MeCorePlane{src_planes_[1], chroma_w_, chroma_w_, chroma_h_};
    cur.plane[2] =
        me::MeCorePlane{src_planes_[2], chroma_w_, chroma_w_, chroma_h_};

    if (!check(av2enc::cuda::me_search_batch(
                   cur, d_me_refs_, 1, d_me_jobs_,
                   static_cast<int>(jobs.size()), options, nullptr,
                   d_me_results_, stream_),
               "me_search_batch", error)) {
      return false;
    }
    ++stats_.me_launches;
    stats_.me_jobs += jobs.size();
    std::vector<me::MeCoreResult> host_results(jobs.size());
    if (!check(download(host_results.data(), d_me_results_,
                        jobs.size() * sizeof(me::MeCoreResult)),
               "ME download", error)) {
      return false;
    }
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      results[i].mv_y = host_results[i].mv_y;
      results[i].mv_x = host_results[i].mv_x;
    }
    return true;
  }

  bool inter_sse_batch(const std::vector<InterJob>& jobs,
                       std::vector<std::int64_t>& sse,
                       std::string& error) override {
    sse.assign(jobs.size(), 0);
    if (jobs.empty()) return true;
    if (!ensure_device(&d_mc_jobs_, mc_jobs_bytes_, jobs.size(), error) ||
        !ensure_device(&d_sse_jobs_, sse_jobs_bytes_, jobs.size(), error) ||
        !ensure_device(&d_sse_out_, sse_out_bytes_, jobs.size(), error) ||
        !ensure_device(&d_pred_, pred_bytes_,
                       std::max<std::size_t>(jobs.size(), kPredSlots) *
                           kMaxBlockPixels,
                       error)) {
      return false;
    }
    std::vector<McBlockJob> mc_jobs(jobs.size());
    std::vector<SseJob> sse_jobs(jobs.size());
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      std::uint8_t* dst =
          d_pred_ + i * static_cast<std::size_t>(kMaxBlockPixels);
      mc_jobs[i] = McBlockJob{};
      mc_jobs[i].dst = dst;
      mc_jobs[i].dst_stride = jobs[i].w;
      mc_jobs[i].x = jobs[i].x;
      mc_jobs[i].y = jobs[i].y;
      mc_jobs[i].w = jobs[i].w;
      mc_jobs[i].h = jobs[i].h;
      mc_jobs[i].plane = 0;
      mc_jobs[i].mv = inter::mc::Mv{jobs[i].mv_y, jobs[i].mv_x};
      mc_jobs[i].filter = inter::mc::InterpFilter::kEighttap;
      sse_jobs[i].src = src_planes_[0];
      sse_jobs[i].src_stride = width_;
      sse_jobs[i].frame_w = width_;
      sse_jobs[i].frame_h = height_;
      sse_jobs[i].pred = dst;
      sse_jobs[i].pred_stride = jobs[i].w;
      sse_jobs[i].x = jobs[i].x;
      sse_jobs[i].y = jobs[i].y;
      sse_jobs[i].w = jobs[i].w;
      sse_jobs[i].h = jobs[i].h;
      sse_jobs[i].out = d_sse_out_ + i;
    }
    upload(d_mc_jobs_, mc_jobs.data(),
           mc_jobs.size() * sizeof(McBlockJob));
    if (!check(av2enc::cuda::mc_predict_batch(
                   ref_frame(), d_mc_jobs_, static_cast<int>(jobs.size()),
                   stream_),
               "mc_predict_batch", error)) {
      return false;
    }
    ++stats_.mc_launches;
    stats_.inter_candidates += jobs.size();

    upload(d_sse_jobs_, sse_jobs.data(),
           sse_jobs.size() * sizeof(SseJob));
    const int threads = 128;
    const int blocks = div_up(static_cast<int>(jobs.size()), threads);
    m2_sse_kernel<<<blocks, threads, 0, stream_>>>(
        d_sse_jobs_, static_cast<int>(jobs.size()));
    if (!check(cudaGetLastError(), "m2_sse_kernel", error)) return false;
    ++stats_.sse_launches;
    return check(download(sse.data(), d_sse_out_,
                          jobs.size() * sizeof(std::int64_t)),
                 "SSE download", error);
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
    if (!ensure_device(&d_mc_jobs_, mc_jobs_bytes_, 3, error)) return false;

    McBlockJob jobs[3];
    jobs[0].dst = recon_planes_[0] +
                  static_cast<std::size_t>(luma_job.y) *
                      static_cast<std::size_t>(luma_w_) +
                  static_cast<std::size_t>(luma_job.x);
    jobs[0].dst_stride = luma_w_;
    jobs[0].x = luma_job.x;
    jobs[0].y = luma_job.y;
    jobs[0].w = w;
    jobs[0].h = h;
    jobs[0].plane = 0;
    jobs[0].mv = inter::mc::Mv{luma_job.mv_y, luma_job.mv_x};
    jobs[0].filter = inter::mc::InterpFilter::kEighttap;
    for (int plane = 1; plane <= 2; ++plane) {
      jobs[plane].dst = recon_planes_[plane] +
                        static_cast<std::size_t>(luma_job.y / 2) *
                            static_cast<std::size_t>(chroma_mi_w_) +
                        static_cast<std::size_t>(luma_job.x / 2);
      jobs[plane].dst_stride = chroma_mi_w_;
      jobs[plane].x = luma_job.x / 2;
      jobs[plane].y = luma_job.y / 2;
      jobs[plane].w = cw;
      jobs[plane].h = ch;
      jobs[plane].plane = plane;
      jobs[plane].mv = inter::mc::Mv{luma_job.mv_y, luma_job.mv_x};
      jobs[plane].filter = inter::mc::InterpFilter::kEighttap;
    }
    upload(d_mc_jobs_, jobs, sizeof(jobs));
    if (!check(av2enc::cuda::mc_predict_batch(ref_frame(), d_mc_jobs_, 3,
                                              stream_),
               "mc_predict_batch 4:2:0", error)) {
      return false;
    }
    ++stats_.mc_launches;
    if (!download_2d(y.data(), w, recon_planes_[0] +
                                     static_cast<std::size_t>(luma_job.y) *
                                         static_cast<std::size_t>(luma_w_) +
                                     static_cast<std::size_t>(luma_job.x),
                     luma_w_, w, h, error) ||
        !download_2d(u.data(), cw,
                     recon_planes_[1] +
                         static_cast<std::size_t>(luma_job.y / 2) *
                             static_cast<std::size_t>(chroma_mi_w_) +
                         static_cast<std::size_t>(luma_job.x / 2),
                     chroma_mi_w_, cw, ch, error) ||
        !download_2d(v.data(), cw,
                     recon_planes_[2] +
                         static_cast<std::size_t>(luma_job.y / 2) *
                             static_cast<std::size_t>(chroma_mi_w_) +
                         static_cast<std::size_t>(luma_job.x / 2),
                     chroma_mi_w_, cw, ch, error)) {
      return false;
    }
    return true;
  }

  bool intra_candidate(int x0, int y0, int w, int h,
                       std::array<PlaneTx, 3>& planes,
                       std::string& error) override {
    const int coded_w[3] = {w, w / 2, w / 2};
    const int coded_h[3] = {h, h / 2, h / 2};
    const int ox[3] = {x0, x0 / 2, x0 / 2};
    const int oy[3] = {y0, y0 / 2, y0 / 2};

    if (!ensure_intra_buffers(error)) return false;

    std::vector<intra_modes::IntraJob> jobs(3);
    for (int plane = 0; plane < 3; ++plane) {
      intra_modes::IntraBlock block;
      block.recon_plane = recon_planes_[plane];
      block.recon_stride = plane == 0 ? luma_w_ : chroma_mi_w_;
      block.x0 = ox[plane];
      block.y0 = oy[plane];
      block.width = coded_w[plane];
      block.height = coded_h[plane];
      block.max_x = (plane == 0 ? luma_w_ : chroma_mi_w_) - 1;
      block.max_y = (plane == 0 ? luma_h_ : chroma_mi_h_) - 1;
      block.have_left = block.x0 > 0 ? 1 : 0;
      block.have_above = block.y0 > 0 ? 1 : 0;
      block.num4_above_right = 0;
      block.num4_below_left = 0;
      block.plane = plane;
      block.mode = intra_modes::kDcPred;
      block.angle_delta = 0;
      block.filter_type_above = 0;
      block.filter_type_left = 0;
      block.enable_edge_filter = 0;
      block.source_plane = src_planes_[plane];
      block.source_stride = plane == 0 ? width_ : chroma_w_;
      block.source_max_x = (plane == 0 ? width_ : chroma_w_) - 1;
      block.source_max_y = (plane == 0 ? height_ : chroma_h_) - 1;
      jobs[static_cast<std::size_t>(plane)].block = block;
      jobs[static_cast<std::size_t>(plane)].pred =
          d_pred_ + static_cast<std::size_t>(plane) * kMaxBlockPixels;
      jobs[static_cast<std::size_t>(plane)].residual =
          d_residual_ + static_cast<std::size_t>(plane) * kMaxCodedPixels;
    }

    if (!ensure_device(&d_intra_jobs_, intra_jobs_bytes_, 3, error)) {
      return false;
    }
    upload(d_intra_jobs_, jobs.data(), jobs.size() * sizeof(intra_modes::IntraJob));
    if (!check(intra_modes::intra_predict_batch(
                   d_intra_jobs_, 3, stream_),
               "intra_predict_batch", error)) {
      return false;
    }
    ++stats_.intra_launches;

    return tx_and_recon_420(coded_w, coded_h, planes, error);
  }

  // Runs the mixed-size transform chain on three planes already staged in
  // d_pred_ / d_residual_ (coded residual), reconstructs, and downloads the
  // levels + recon.  Used by the intra candidate and the inter residual path.
  bool tx_and_recon_420(const int coded_w[3], const int coded_h[3],
                        std::array<PlaneTx, 3>& planes, std::string& error) {
    const std::size_t need_levels =
        static_cast<std::size_t>(3 * kMaxPlaneTx) * sizeof(std::int32_t);
    const std::size_t need_recon =
        static_cast<std::size_t>(3 * kMaxCodedPixels);
    const std::size_t need_residual =
        static_cast<std::size_t>(3 * kMaxCodedPixels) * sizeof(std::int16_t);
    const std::size_t need_adjusted =
        static_cast<std::size_t>(3 * kMaxPlaneTx) * sizeof(std::int16_t);
    const std::size_t need_coeff =
        static_cast<std::size_t>(3 * kMaxPlaneTx) * sizeof(double);
    const std::size_t need_pred =
        static_cast<std::size_t>(3 * kMaxBlockPixels);
    if (levels_bytes_ < need_levels || recon_out_bytes_ < need_recon ||
        recon_residual_bytes_ < need_residual ||
        residual_adj_bytes_ < need_adjusted || coeff_bytes_ < need_coeff ||
        pred_bytes_ < need_pred ||
        tx_jobs_bytes_ < 3 * sizeof(av2enc::cuda::TxBlockJob) ||
        recon_jobs_bytes_ < 3 * sizeof(ReconJob)) {
      error = "M2 TX scratch is not allocated";
      return false;
    }
    std::vector<av2enc::cuda::TxBlockJob> tx_jobs(3);
    for (int plane = 0; plane < 3; ++plane) {
      const auto params_opt = av2enc::txv::make_tx_params(
          coded_w[plane], coded_h[plane], av2enc::txv::TxType::kDctDct);
      if (!params_opt.has_value()) {
        error = "unsupported transform size";
        return false;
      }
      const av2enc::txv::TxParams& params = *params_opt;
      const std::int16_t* coded = d_residual_ + plane * kMaxCodedPixels;
      std::int16_t* adjusted = d_residual_adj_ + plane * kMaxPlaneTx;
      const int coded_elems = coded_w[plane] * coded_h[plane];
      if (params.w == coded_w[plane] && params.h == coded_h[plane]) {
        if (!check(cudaMemcpyAsync(adjusted, coded,
                                   static_cast<std::size_t>(coded_elems) *
                                       sizeof(std::int16_t),
                                   cudaMemcpyDeviceToDevice, stream_),
                   "residual copy", error)) {
          return false;
        }
      } else if (!check(av2enc::cuda::txv_downsample_residual(
                            coded, adjusted, params, 1, stream_),
                        "txv_downsample_residual", error)) {
        return false;
      }
      tx_jobs[static_cast<std::size_t>(plane)].params = params;
      tx_jobs[static_cast<std::size_t>(plane)].in_offset =
          plane * kMaxPlaneTx;
      tx_jobs[static_cast<std::size_t>(plane)].out_offset =
          plane * kMaxCodedPixels;
    }
    upload(d_tx_jobs_, tx_jobs.data(),
           tx_jobs.size() * sizeof(av2enc::cuda::TxBlockJob));
    if (!check(av2enc::cuda::txv_forward_transform_multi(
                   d_residual_adj_, d_coeff_, d_levels_, d_tx_jobs_, 3,
                   stream_),
               "txv_forward_transform_multi", error)) {
      return false;
    }
    if (!check(av2enc::cuda::txv_inverse_transform_multi(
                   d_levels_, d_recon_residual_, d_tx_jobs_, 3, stream_),
               "txv_inverse_transform_multi", error)) {
      return false;
    }
    stats_.tx_launches += 2;

    // d_tx_jobs_, d_recon_jobs_ and d_recon_out_ are allocated by
    // ensure_intra_buffers() before the first TX call.
    std::vector<ReconJob> recon_jobs(3);
    for (int plane = 0; plane < 3; ++plane) {
      recon_jobs[static_cast<std::size_t>(plane)].pred =
          d_pred_ + static_cast<std::size_t>(plane) * kMaxBlockPixels;
      recon_jobs[static_cast<std::size_t>(plane)].residual =
          d_recon_residual_ + static_cast<std::size_t>(plane) * kMaxCodedPixels;
      recon_jobs[static_cast<std::size_t>(plane)].out =
          d_recon_out_ + static_cast<std::size_t>(plane) * kMaxCodedPixels;
      recon_jobs[static_cast<std::size_t>(plane)].w = coded_w[plane];
      recon_jobs[static_cast<std::size_t>(plane)].h = coded_h[plane];
    }
    upload(d_recon_jobs_, recon_jobs.data(),
           recon_jobs.size() * sizeof(ReconJob));
    constexpr int kReconThreads = 256;
    m2_recon_kernel<<<3, kReconThreads, 0, stream_>>>(d_recon_jobs_, 3);
    if (!check(cudaGetLastError(), "m2_recon_kernel", error)) return false;
    ++stats_.recon_launches;

    std::vector<std::int32_t> host_levels(3 * kMaxPlaneTx);
    std::vector<std::uint8_t> host_recon(3 * kMaxCodedPixels);
    if (!check(download(host_levels.data(), d_levels_,
                        host_levels.size() * sizeof(std::int32_t)),
               "tx levels download", error) ||
        !check(download(host_recon.data(), d_recon_out_,
                        host_recon.size() * sizeof(std::uint8_t)),
               "tx recon download", error)) {
      return false;
    }

    for (int plane = 0; plane < 3; ++plane) {
      const auto params_opt = av2enc::txv::make_tx_params(
          coded_w[plane], coded_h[plane], av2enc::txv::TxType::kDctDct);
      const av2enc::txv::TxParams& params = *params_opt;
      const int corners = params.w * params.h;
      planes[static_cast<std::size_t>(plane)].levels.assign(
          host_levels.begin() + plane * kMaxPlaneTx,
          host_levels.begin() + plane * kMaxPlaneTx + corners);
      const int coded_elems = coded_w[plane] * coded_h[plane];
      planes[static_cast<std::size_t>(plane)].recon.assign(
          host_recon.begin() + plane * kMaxCodedPixels,
          host_recon.begin() + plane * kMaxCodedPixels + coded_elems);
    }
    return true;
  }

  bool write_recon_420(int x0, int y0, int w, int h,
                       const std::vector<std::uint8_t>& y,
                       const std::vector<std::uint8_t>& u,
                       const std::vector<std::uint8_t>& v,
                       std::string& error) override {
    if (!upload_2d(recon_planes_[0] +
                       static_cast<std::size_t>(y0) *
                           static_cast<std::size_t>(luma_w_) +
                       static_cast<std::size_t>(x0),
                   luma_w_, y.data(), w, w, h, error) ||
        !upload_2d(recon_planes_[1] +
                       static_cast<std::size_t>(y0 / 2) *
                           static_cast<std::size_t>(chroma_mi_w_) +
                       static_cast<std::size_t>(x0 / 2),
                   chroma_mi_w_, u.data(), w / 2, w / 2, h / 2, error) ||
        !upload_2d(recon_planes_[2] +
                       static_cast<std::size_t>(y0 / 2) *
                           static_cast<std::size_t>(chroma_mi_w_) +
                       static_cast<std::size_t>(x0 / 2),
                   chroma_mi_w_, v.data(), w / 2, w / 2, h / 2, error)) {
      return false;
    }
    // The caller's host vectors die at the end of the block decision, so the
    // copies must complete before returning.
    if (!check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize",
               error)) {
      return false;
    }
    ++stats_.syncs;
    ++stats_.recon_launches;
    return true;
  }

  bool inter_residual_420(const InterJob& luma_job,
                          std::array<PlaneTx, 3>& planes,
                          std::string& error) override {
    const int w = luma_job.w;
    const int h = luma_job.h;
    const int cw = w / 2;
    const int ch = h / 2;
    const int coded_w[3] = {w, cw, cw};
    const int coded_h[3] = {h, ch, ch};
    const int ox[3] = {luma_job.x, luma_job.x / 2, luma_job.x / 2};
    const int oy[3] = {luma_job.y, luma_job.y / 2, luma_job.y / 2};
    const int src_stride[3] = {width_, chroma_w_, chroma_w_};
    const int src_w[3] = {width_, chroma_w_, chroma_w_};
    const int src_h[3] = {height_, chroma_h_, chroma_h_};

    if (!ensure_intra_buffers(error) ||
        !ensure_device(&d_mc_jobs_, mc_jobs_bytes_, 3, error)) {
      return false;
    }
    McBlockJob mc_jobs[3];
    for (int plane = 0; plane < 3; ++plane) {
      mc_jobs[static_cast<std::size_t>(plane)].dst = d_pred_ +
                           static_cast<std::size_t>(plane) * kMaxBlockPixels;
      mc_jobs[static_cast<std::size_t>(plane)].dst_stride = coded_w[plane];
      mc_jobs[static_cast<std::size_t>(plane)].x = ox[plane];
      mc_jobs[static_cast<std::size_t>(plane)].y = oy[plane];
      mc_jobs[static_cast<std::size_t>(plane)].w = coded_w[plane];
      mc_jobs[static_cast<std::size_t>(plane)].h = coded_h[plane];
      mc_jobs[static_cast<std::size_t>(plane)].plane = plane;
      mc_jobs[static_cast<std::size_t>(plane)].mv =
          inter::mc::Mv{luma_job.mv_y, luma_job.mv_x};
      mc_jobs[static_cast<std::size_t>(plane)].filter =
          inter::mc::InterpFilter::kEighttap;
    }
    upload(d_mc_jobs_, mc_jobs, sizeof(mc_jobs));
    if (!check(av2enc::cuda::mc_predict_batch(ref_frame(), d_mc_jobs_, 3,
                                              stream_),
               "mc_predict_batch residual 4:2:0", error)) {
      return false;
    }
    ++stats_.mc_launches;

    constexpr int kResidualThreads = 256;
    for (int plane = 0; plane < 3; ++plane) {
      const int pixels = coded_w[plane] * coded_h[plane];
      m2_residual_kernel<<<div_up(pixels, kResidualThreads),
                           kResidualThreads, 0, stream_>>>(
          src_planes_[plane], src_stride[plane], src_w[plane], src_h[plane],
          ox[plane], oy[plane],
          d_pred_ + static_cast<std::size_t>(plane) * kMaxBlockPixels,
          coded_w[plane],
          d_residual_ + static_cast<std::size_t>(plane) * kMaxCodedPixels,
          coded_w[plane], coded_h[plane]);
      if (!check(cudaGetLastError(), "m2_residual_kernel", error)) {
        return false;
      }
    }
    stats_.residual_launches += 3;
    return tx_and_recon_420(coded_w, coded_h, planes, error);
  }

  bool inter_residual_candidate(const InterJob& job, PlaneTx& out,
                                std::string& error) override {
    if (!ensure_intra_buffers(error) ||
        !ensure_device(&d_mc_jobs_, mc_jobs_bytes_, 1, error)) {
      return false;
    }
    McBlockJob mc_job;
    mc_job.dst = d_pred_;
    mc_job.dst_stride = job.w;
    mc_job.x = job.x;
    mc_job.y = job.y;
    mc_job.w = job.w;
    mc_job.h = job.h;
    mc_job.plane = 0;
    mc_job.mv = inter::mc::Mv{job.mv_y, job.mv_x};
    mc_job.filter = inter::mc::InterpFilter::kEighttap;
    upload(d_mc_jobs_, &mc_job, sizeof(mc_job));
    if (!check(av2enc::cuda::mc_predict_batch(ref_frame(), d_mc_jobs_, 1,
                                              stream_),
               "mc_predict_batch residual", error)) {
      return false;
    }
    ++stats_.mc_launches;

    const int pixels = job.w * job.h;
    const int threads = 256;
    m2_residual_kernel<<<div_up(pixels, threads), threads, 0, stream_>>>(
        src_planes_[0], width_, width_, height_, job.x, job.y, d_pred_, job.w,
        d_residual_, job.w, job.h);
    if (!check(cudaGetLastError(), "m2_residual_kernel", error)) return false;
    ++stats_.residual_launches;

    const auto params_opt =
        av2enc::txv::make_tx_params(job.w, job.h, av2enc::txv::TxType::kDctDct);
    if (!params_opt.has_value()) {
      error = "selftest unsupported transform size";
      return false;
    }
    const av2enc::txv::TxParams& params = *params_opt;
    std::int16_t* adjusted = d_residual_adj_;
    if (params.w == job.w && params.h == job.h) {
      if (!check(cudaMemcpyAsync(adjusted, d_residual_,
                                 static_cast<std::size_t>(pixels) *
                                     sizeof(std::int16_t),
                                 cudaMemcpyDeviceToDevice, stream_),
                 "selftest residual copy", error)) {
        return false;
      }
    } else if (!check(av2enc::cuda::txv_downsample_residual(d_residual_, adjusted,
                                                   params, 1, stream_),
                      "selftest downsample", error)) {
      return false;
    }
    av2enc::cuda::TxBlockJob tx_job;
    tx_job.params = params;
    tx_job.in_offset = 0;
    tx_job.out_offset = 0;
    upload(d_tx_jobs_, &tx_job, sizeof(tx_job));
    if (!check(av2enc::cuda::txv_forward_transform_multi(d_residual_adj_, d_coeff_,
                                                d_levels_, d_tx_jobs_, 1,
                                                stream_),
               "selftest forward", error) ||
        !check(av2enc::cuda::txv_inverse_transform_multi(d_levels_, d_recon_residual_,
                                                d_tx_jobs_, 1, stream_),
               "selftest inverse", error)) {
      return false;
    }
    stats_.tx_launches += 2;

    ReconJob recon_job;
    recon_job.pred = d_pred_;
    recon_job.residual = d_recon_residual_;
    recon_job.out = d_recon_out_;
    recon_job.w = job.w;
    recon_job.h = job.h;
    upload(d_recon_jobs_, &recon_job, sizeof(recon_job));
    m2_recon_kernel<<<1, 256, 0, stream_>>>(d_recon_jobs_, 1);
    if (!check(cudaGetLastError(), "selftest recon", error)) return false;
    ++stats_.recon_launches;

    const int adjusted_elems = params.w * params.h;
    if (levels_bytes_ <
            static_cast<std::size_t>(adjusted_elems) * sizeof(std::int32_t) ||
        recon_out_bytes_ < static_cast<std::size_t>(pixels) ||
        recon_residual_bytes_ <
            static_cast<std::size_t>(pixels) * sizeof(std::int16_t) ||
        residual_adj_bytes_ <
            static_cast<std::size_t>(adjusted_elems) * sizeof(std::int16_t) ||
        coeff_bytes_ <
            static_cast<std::size_t>(adjusted_elems) * sizeof(double) ||
        pred_bytes_ < static_cast<std::size_t>(pixels) ||
        tx_jobs_bytes_ < sizeof(av2enc::cuda::TxBlockJob) ||
        recon_jobs_bytes_ < sizeof(ReconJob)) {
      error = "M2 selftest TX scratch is not allocated";
      return false;
    }
    out.levels.resize(static_cast<std::size_t>(adjusted_elems));
    out.recon.resize(static_cast<std::size_t>(pixels));
    if (!check(download(out.levels.data(), d_levels_,
                        static_cast<std::size_t>(adjusted_elems) *
                            sizeof(std::int32_t)),
               "selftest levels download", error) ||
        !check(download(out.recon.data(), d_recon_out_,
                        static_cast<std::size_t>(pixels) *
                            sizeof(std::uint8_t)),
               "selftest recon download", error)) {
      return false;
    }
    return true;
  }

  bool debug_set_recon(int plane, const std::vector<std::uint8_t>& data,
                       std::string& error) override {
    if (plane < 0 || plane > 2) {
      error = "debug_set_recon: bad plane";
      return false;
    }
    const int stride = plane == 0 ? luma_w_ : chroma_mi_w_;
    const int rows = plane == 0 ? luma_h_ : chroma_mi_h_;
    const std::size_t expected = static_cast<std::size_t>(stride) *
                                 static_cast<std::size_t>(rows);
    if (data.size() != expected) {
      error = "debug_set_recon: bad size";
      return false;
    }
    if (!upload_2d(recon_planes_[plane], stride, data.data(), stride, stride,
                   rows, error)) {
      return false;
    }
    return check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize",
                 error);
  }

  const M2GpuStats& stats() const override { return stats_; }
  void reset_stats() override { stats_ = M2GpuStats{}; }

 private:
  McDeviceFrame ref_frame() const {
    McDeviceFrame frame;
    frame.plane[0] = ref_planes_[0];
    frame.plane[1] = ref_planes_[1];
    frame.plane[2] = ref_planes_[2];
    frame.stride[0] = luma_w_;
    frame.stride[1] = chroma_mi_w_;
    frame.stride[2] = chroma_mi_w_;
    frame.mi_cols = mi_cols_;
    frame.mi_rows = mi_rows_;
    return frame;
  }

  bool ensure_buffers(std::string& error) {
    const std::size_t luma =
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
    const std::size_t chroma = static_cast<std::size_t>(chroma_w_) *
                               static_cast<std::size_t>(chroma_h_);
    const std::size_t mi_luma = static_cast<std::size_t>(luma_w_) *
                                static_cast<std::size_t>(luma_h_);
    const std::size_t mi_chroma = static_cast<std::size_t>(chroma_mi_w_) *
                                  static_cast<std::size_t>(chroma_mi_h_);
    for (int plane = 0; plane < 3; ++plane) {
      const std::size_t src_bytes = plane == 0 ? luma : chroma;
      const std::size_t mi_bytes = plane == 0 ? mi_luma : mi_chroma;
      if (!ensure_device(&src_planes_[plane], src_bytes_[plane], src_bytes,
                         error) ||
          !ensure_device(&ref_planes_[plane], ref_bytes_[plane], mi_bytes,
                         error) ||
          !ensure_device(&recon_planes_[plane], recon_bytes_[plane], mi_bytes,
                         error)) {
        return false;
      }
    }
    return true;
  }

  bool ensure_intra_buffers(std::string& error) {
    if (!ensure_device(&d_pred_, pred_bytes_,
                       static_cast<std::size_t>(kPredSlots) * kMaxBlockPixels,
                       error) ||
        !ensure_device(&d_residual_, residual_bytes_,
                       3 * kMaxCodedPixels, error) ||
        !ensure_device(&d_residual_adj_, residual_adj_bytes_,
                       3 * kMaxPlaneTx, error) ||
        !ensure_device(&d_coeff_, coeff_bytes_, 3 * kMaxPlaneTx, error) ||
        !ensure_device(&d_levels_, levels_bytes_, 3 * kMaxPlaneTx, error) ||
        !ensure_device(&d_recon_residual_, recon_residual_bytes_,
                       3 * kMaxCodedPixels, error) ||
        !ensure_device(&d_tx_jobs_, tx_jobs_bytes_, 3, error) ||
        !ensure_device(&d_recon_jobs_, recon_jobs_bytes_, 3, error) ||
        // uint8_t buffer: the count is the number of *samples*, not planes.
        !ensure_device(&d_recon_out_, recon_out_bytes_,
                       3 * kMaxCodedPixels, error)) {
      return false;
    }
    return true;
  }

  bool upload_2d(std::uint8_t* dst, int dst_stride, const std::uint8_t* src,
                 int src_stride, int width, int rows, std::string& error) {
    if (width <= 0 || rows <= 0) return true;
    if (!check(cudaMemcpy2DAsync(
                   dst, static_cast<std::size_t>(dst_stride), src,
                   static_cast<std::size_t>(src_stride),
                   static_cast<std::size_t>(width), static_cast<std::size_t>(rows),
                   cudaMemcpyHostToDevice, stream_),
               "cudaMemcpy2DAsync upload", error)) {
      return false;
    }
    stats_.bytes_uploaded += static_cast<std::uint64_t>(width) *
                             static_cast<std::uint64_t>(rows);
    return true;
  }

  bool download_2d(std::uint8_t* dst, int dst_stride,
                   const std::uint8_t* src, int src_stride, int width,
                   int rows, std::string& error) {
    if (width <= 0 || rows <= 0) return true;
    if (!check(cudaMemcpy2DAsync(
                   dst, static_cast<std::size_t>(dst_stride), src,
                   static_cast<std::size_t>(src_stride),
                   static_cast<std::size_t>(width), static_cast<std::size_t>(rows),
                   cudaMemcpyDeviceToHost, stream_),
               "cudaMemcpy2DAsync download", error)) {
      return false;
    }
    if (!check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize",
               error)) {
      return false;
    }
    ++stats_.syncs;
    stats_.bytes_downloaded += static_cast<std::uint64_t>(width) *
                               static_cast<std::uint64_t>(rows);
    return true;
  }

  void upload(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_);
    stats_.bytes_uploaded += bytes;
  }

  cudaError_t download(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return cudaSuccess;
    cudaError_t err =
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) return err;
    ++stats_.syncs;
    stats_.bytes_downloaded += bytes;
    return cudaSuccess;
  }

  void free_all() {
    for (int plane = 0; plane < 3; ++plane) {
      if (src_planes_[plane] != nullptr) cudaFree(src_planes_[plane]);
      if (ref_planes_[plane] != nullptr) cudaFree(ref_planes_[plane]);
      if (recon_planes_[plane] != nullptr) cudaFree(recon_planes_[plane]);
    }
    auto free_one = [](auto*& ptr) {
      if (ptr != nullptr) {
        cudaFree(static_cast<void*>(ptr));
        ptr = nullptr;
      }
    };
    free_one(d_variance_nodes_);
    free_one(d_variance_costs_);
    free_one(d_me_jobs_);
    free_one(d_me_results_);
    free_one(d_me_refs_);
    free_one(d_mc_jobs_);
    free_one(d_sse_jobs_);
    free_one(d_sse_out_);
    free_one(d_pred_);
    free_one(d_intra_jobs_);
    free_one(d_residual_);
    free_one(d_residual_adj_);
    free_one(d_coeff_);
    free_one(d_levels_);
    free_one(d_tx_jobs_);
    free_one(d_recon_residual_);
    free_one(d_recon_out_);
    free_one(d_recon_jobs_);
  }

  cudaStream_t stream_ = nullptr;
  bool stream_ok_ = false;
  cudaEvent_t start_event_ = nullptr;
  cudaEvent_t end_event_ = nullptr;

  int width_ = 0;
  int height_ = 0;
  int chroma_w_ = 0;
  int chroma_h_ = 0;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  int luma_w_ = 0;
  int luma_h_ = 0;
  int chroma_mi_w_ = 0;
  int chroma_mi_h_ = 0;

  std::uint8_t* src_planes_[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* ref_planes_[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* recon_planes_[3] = {nullptr, nullptr, nullptr};
  std::size_t src_bytes_[3] = {0, 0, 0};
  std::size_t ref_bytes_[3] = {0, 0, 0};
  std::size_t recon_bytes_[3] = {0, 0, 0};

  VarianceNode* d_variance_nodes_ = nullptr;
  std::int64_t* d_variance_costs_ = nullptr;
  std::size_t variance_nodes_bytes_ = 0;
  std::size_t variance_costs_bytes_ = 0;

  me::MeCoreJob* d_me_jobs_ = nullptr;
  me::MeCoreResult* d_me_results_ = nullptr;
  me::MeCoreRef* d_me_refs_ = nullptr;
  std::size_t me_jobs_bytes_ = 0;
  std::size_t me_results_bytes_ = 0;
  std::size_t me_refs_bytes_ = 0;

  McBlockJob* d_mc_jobs_ = nullptr;
  SseJob* d_sse_jobs_ = nullptr;
  std::int64_t* d_sse_out_ = nullptr;
  std::size_t mc_jobs_bytes_ = 0;
  std::size_t sse_jobs_bytes_ = 0;
  std::size_t sse_out_bytes_ = 0;

  std::uint8_t* d_pred_ = nullptr;
  std::size_t pred_bytes_ = 0;
  intra_modes::IntraJob* d_intra_jobs_ = nullptr;
  std::size_t intra_jobs_bytes_ = 0;
  std::int16_t* d_residual_ = nullptr;
  std::size_t residual_bytes_ = 0;
  std::int16_t* d_residual_adj_ = nullptr;
  std::size_t residual_adj_bytes_ = 0;
  double* d_coeff_ = nullptr;
  std::size_t coeff_bytes_ = 0;
  std::int32_t* d_levels_ = nullptr;
  std::size_t levels_bytes_ = 0;
  av2enc::cuda::TxBlockJob* d_tx_jobs_ = nullptr;
  std::size_t tx_jobs_bytes_ = 0;
  std::int16_t* d_recon_residual_ = nullptr;
  std::size_t recon_residual_bytes_ = 0;
  std::uint8_t* d_recon_out_ = nullptr;
  std::size_t recon_out_bytes_ = 0;
  ReconJob* d_recon_jobs_ = nullptr;
  std::size_t recon_jobs_bytes_ = 0;

  M2GpuStats stats_{};
};

}  // namespace

std::unique_ptr<Backend> make_backend() {
  return std::unique_ptr<Backend>(new CudaBackend());
}

const char* make_backend_name() { return "cuda"; }

}  // namespace av2enc::cuda::m2::detail
