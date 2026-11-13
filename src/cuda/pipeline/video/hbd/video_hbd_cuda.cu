// CUDA backend for the format video core (10-bit and 8-bit chroma formats).
//
// The orchestration and every decision live in
// `src/pipeline/video/hbd/video_hbd.cpp`; this file only performs pixel work on
// the device.  The CPU reference backend and this backend call the same inline
// arithmetic (`video_hbd_ops.h`), so their predictions and transforms are
// identical by construction:
//
//   * DC intra prediction            (§7.13.2.10)
//   * two-pass subpel MC + filter downgrade (§7.13.3.18)
//   * integer diamond + subpel ME    (search heuristic; same code on both)
//   * forward quantise / dequantise / inverse via `cuda::hbd::hbd_tx_batch`
//
// Allocation policy: one arena for every device buffer, no cudaMalloc after
// init, frame-boundary growth only.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/pipeline/still/hbd/hbd_tx.h"
#include "formats/high_bitdepth/hbd_depth.h"
#include "cuda/pipeline/video/hbd/video_hbd_cuda.h"
#include "pipeline/video/hbd/video_hbd.h"
#include "pipeline/video/hbd/video_hbd_ops.h"

namespace av2enc::cuda::video::hbd_gpu {
namespace {

using av2enc::video::hbd::Backend;
using av2enc::video::hbd::BeginFrame;
using av2enc::video::hbd::FramePlanes;
using av2enc::video::hbd::InterJob;
using av2enc::video::hbd::MeJob;
using av2enc::video::hbd::Mv;
using av2enc::video::hbd::Options;
using av2enc::video::hbd::PlaneTx;
using av2enc::video::hbd::PlaneView;

constexpr std::uint64_t kMiB = 1024ull * 1024ull;

bool check(cudaError_t err, const char* what, std::string& error) {
  if (err == cudaSuccess) return true;
  error = std::string(what) + ": " + cudaGetErrorString(err);
  return false;
}

struct DcJob {
  const std::uint16_t* recon = nullptr;
  int stride = 0;
  int max_x = 0;
  int max_y = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  bool have_above = false;
  bool have_left = false;
  std::uint16_t* pred = nullptr;
  int pred_stride = 0;
};

__global__ void dc_predict_kernel(DcJob job, int depth) {
  std::uint16_t above[64];
  std::uint16_t left[64];
  for (int i = 0; i < job.w; ++i) {
    const int sx = min(job.max_x, job.x + i);
    above[i] = job.have_above
                   ? job.recon[static_cast<std::size_t>(job.y - 1) *
                                   static_cast<std::size_t>(job.stride) +
                               static_cast<std::size_t>(sx)]
                   : 0;
  }
  for (int i = 0; i < job.h; ++i) {
    const int sy = min(job.max_y, job.y + i);
    left[i] = job.have_left
                  ? job.recon[static_cast<std::size_t>(sy) *
                                  static_cast<std::size_t>(job.stride) +
                              static_cast<std::size_t>(job.x - 1)]
                  : 0;
  }
  av2enc::video::hbd::hbd_dc_predict(above, left, job.w, job.h,
                                     job.have_above, job.have_left, depth,
                                     job.pred, job.pred_stride);
}

struct McJob {
  const std::uint16_t* ref = nullptr;
  int ref_stride = 0;
  int ref_w = 0;
  int ref_h = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int mv_y = 0;
  int mv_x = 0;
  int sub_x = 0;
  int sub_y = 0;
  std::uint16_t* dst = nullptr;
  int dst_stride = 0;
};

__global__ void mc_kernel(McJob job, int depth) {
  av2enc::video::hbd::hbd_mc_block(
      job.ref, job.ref_stride, 0, 0, job.ref_w - 1, job.ref_h - 1, job.mv_y,
      job.mv_x, job.sub_x, job.sub_y, depth, job.x, job.y, job.w, job.h,
      job.dst, job.dst_stride);
}

__global__ void sse_kernel(McJob job, const std::uint16_t* src, int src_stride,
                           int depth, std::int64_t* out) {
  std::int64_t total = 0;
  for (int r = 0; r < job.h; ++r) {
    for (int c = 0; c < job.w; ++c) {
      const int pred = av2enc::video::hbd::hbd_mc_sample(
          job.ref, job.ref_stride, 0, 0, job.ref_w - 1, job.ref_h - 1,
          job.mv_y, job.mv_x, job.sub_x, job.sub_y, job.w, job.h, depth,
          job.y + r, job.x + c);
      const int s = static_cast<int>(
          src[static_cast<std::size_t>(job.y + r) *
                  static_cast<std::size_t>(src_stride) +
              static_cast<std::size_t>(job.x + c)]);
      const int d = s - pred;
      total += static_cast<std::int64_t>(d) * d;
    }
  }
  out[0] = total;
}

struct MeJobDevice {
  const std::uint16_t* cur = nullptr;
  int cur_stride = 0;
  const std::uint16_t* ref = nullptr;
  int ref_stride = 0;
  int ref_w = 0;
  int ref_h = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int range = 16;
  int stages = 3;
  int* out = nullptr;
};

__global__ void me_kernel(MeJobDevice job, int depth) {
  int mv_y = 0;
  int mv_x = 0;
  av2enc::video::hbd::hbd_me_search(
      job.cur, job.cur_stride, job.ref, job.ref_stride, job.x, job.y, job.w,
      job.h, job.ref_w, job.ref_h, job.range, job.stages, depth, &mv_y, &mv_x);
  job.out[0] = mv_y;
  job.out[1] = mv_x;
}

struct PlaneBuffers {
  std::uint16_t* source = nullptr;  // visible plane, stride = visible width
  int source_stride = 0;
  int source_w = 0;
  int source_h = 0;
  std::uint16_t* recon = nullptr;  // coded plane
  int recon_stride = 0;
  int recon_w = 0;
  int recon_h = 0;
  std::uint16_t* refs[8] = {};
};

class CudaBackend final : public Backend {
 public:
  explicit CudaBackend(av2enc::cuda::Arena* arena) : external_arena_(arena) {}
  ~CudaBackend() override = default;

  const char* name() const override { return "cuda"; }

  void set_options(const Options& options) override { options_ = options; }

  bool begin_frame(const BeginFrame& frame, std::string& error) override {
    if (!ensure_arena(error)) return false;
    frame_ = frame;
    format_ = frame.format;
    depth_ = static_cast<int>(format_.depth);
    quant_step_ = ::av2enc::hbd::depth::qlookup(frame.base_q_idx);
    if (!allocate_planes(frame, error)) return false;
    if (!upload_source(frame, error)) return false;
    if (!zero_recon(error)) return false;
    if (!upload_references(frame, error)) return false;
    return true;
  }

  bool end_frame(std::string& error) override {
    return check(cudaStreamSynchronize(stream_), "stream sync", error);
  }

  bool variance_costs(const std::vector<av2enc::video::hbd::VarianceNode>&,
                      std::vector<std::int64_t>&, std::string&) override {
    return false;  // the orchestration evaluates variance on the host
  }

  bool me_batch(const std::vector<MeJob>& jobs, std::vector<Mv>& mvs,
                std::string& error) override {
    mvs.resize(jobs.size());
    if (jobs.empty()) return true;
    const int count = static_cast<int>(
        std::min<std::size_t>(jobs.size(), kMaxBatch));
    for (int i = 0; i < count; ++i) {
      const MeJob& job = jobs[static_cast<std::size_t>(i)];
      if (job.ref_index < 0 || job.ref_index >= frame_.num_refs) {
        error = "ME reference index out of range";
        return false;
      }
      MeJobDevice d;
      d.cur = planes_[0].source;
      d.cur_stride = planes_[0].source_stride;
      d.ref = planes_[0].refs[job.ref_index];
      d.ref_stride = planes_[0].recon_stride;
      d.ref_w = planes_[0].recon_w;
      d.ref_h = planes_[0].recon_h;
      d.x = job.x;
      d.y = job.y;
      d.w = job.w;
      d.h = job.h;
      d.range = options_.me_range;
      d.stages = 1 | 2 | (options_.me_subpel_eighth ? 4 : 0);
      d.out = d_mv_ + static_cast<std::size_t>(i) * 2;
      me_kernel<<<1, 1, 0, stream_>>>(d, depth_);
    }
    if (!check(cudaGetLastError(), "me kernel", error)) return false;
    std::array<int, kMaxBatch * 2> host{};
    if (!check(cudaMemcpyAsync(host.data(), d_mv_,
                               static_cast<std::size_t>(count) * 2 *
                                   sizeof(int),
                               cudaMemcpyDeviceToHost, stream_),
               "me download", error)) {
      return false;
    }
    if (!check(cudaStreamSynchronize(stream_), "me sync", error)) return false;
    for (int i = 0; i < count; ++i) {
      mvs[static_cast<std::size_t>(i)] =
          Mv{host[static_cast<std::size_t>(i) * 2],
             host[static_cast<std::size_t>(i) * 2 + 1]};
    }
    return true;
  }

  bool inter_sse_batch(const std::vector<InterJob>& jobs,
                       std::vector<std::int64_t>& sse,
                       std::string& error) override {
    sse.resize(jobs.size());
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      McJob job;
      if (!make_mc_job(jobs[i], 0, &job, error)) return false;
      sse_kernel<<<1, 1, 0, stream_>>>(job, planes_[0].source,
                                       planes_[0].source_stride, depth_,
                                       d_sse_);
      std::int64_t value = 0;
      if (!check(cudaMemcpyAsync(&value, d_sse_, sizeof(value),
                                 cudaMemcpyDeviceToHost, stream_),
                 "sse download", error)) {
        return false;
      }
      if (!check(cudaStreamSynchronize(stream_), "sse sync", error)) {
        return false;
      }
      sse[i] = value;
    }
    return true;
  }

  bool mc_420(const InterJob& job, std::vector<std::uint16_t> out[3],
              std::string& error) override {
    const int plane_count = av2enc::fmt::has_chroma(format_.chroma) ? 3 : 1;
    int cw = 0;
    int ch = 0;
    if (!block_chroma(job.w, job.h, &cw, &ch, error)) return false;
    const int sizes[3][2] = {{job.w, job.h}, {cw, ch}, {cw, ch}};
    for (int plane = 0; plane < plane_count; ++plane) {
      McJob mc;
      if (!make_mc_job(job, plane, &mc, error)) return false;
      mc.dst = d_pred_[plane];
      mc.dst_stride = sizes[plane][0];
      mc_kernel<<<1, 1, 0, stream_>>>(mc, depth_);
      out[plane].resize(static_cast<std::size_t>(sizes[plane][0]) *
                        static_cast<std::size_t>(sizes[plane][1]));
      if (!check(cudaMemcpyAsync(out[plane].data(), d_pred_[plane],
                                 out[plane].size() * sizeof(std::uint16_t),
                                 cudaMemcpyDeviceToHost, stream_),
                 "mc download", error)) {
        return false;
      }
    }
    return check(cudaStreamSynchronize(stream_), "mc sync", error);
  }

  bool intra_candidate(int x, int y, int w, int h,
                       std::array<PlaneTx, 3>& planes,
                       std::string& error) override {
    return predict_and_transform(x, y, w, h, /*use_mc=*/false, InterJob{},
                                 planes, error);
  }

  bool inter_residual(const InterJob& job,
                      std::array<PlaneTx, 3>& planes,
                      std::string& error) override {
    return predict_and_transform(job.x, job.y, job.w, job.h, /*use_mc=*/true,
                                 job, planes, error);
  }

  bool write_recon(int x, int y, int w, int h,
                   const std::vector<std::uint16_t>& y_plane,
                   const std::vector<std::uint16_t>& u_plane,
                   const std::vector<std::uint16_t>& v_plane,
                   std::string& error) override {
    int cw = 0;
    int ch = 0;
    if (!block_chroma(w, h, &cw, &ch, error)) return false;
    const std::vector<std::uint16_t>* src[3] = {&y_plane, &u_plane, &v_plane};
    const int sizes[3][2] = {{w, h}, {cw, ch}, {cw, ch}};
    const int plane_count = av2enc::fmt::has_chroma(format_.chroma) ? 3 : 1;
    for (int plane = 0; plane < plane_count; ++plane) {
      const int px = x >> (plane == 0 ? 0 : fmt::subsampling_x(format_.chroma));
      const int py = y >> (plane == 0 ? 0 : fmt::subsampling_y(format_.chroma));
      const int pw = sizes[plane][0];
      const int ph = sizes[plane][1];
      if (src[plane]->empty()) continue;
      const int rows = std::min(ph, planes_[plane].recon_h - py);
      const int cols = std::min(pw, planes_[plane].recon_w - px);
      if (rows <= 0 || cols <= 0) continue;
      // Mirror the reconstruction into the host planes the orchestration
      // keeps (references, result dump and the next frame's `BeginFrame`).
      const PlaneView& host = frame_.recon.plane[plane];
      if (host.data != nullptr) {
        std::uint16_t* dst = const_cast<std::uint16_t*>(host.data) +
                             static_cast<std::size_t>(py) *
                                 static_cast<std::size_t>(host.stride) +
                             static_cast<std::size_t>(px);
        for (int r = 0; r < rows; ++r) {
          std::memcpy(dst + static_cast<std::size_t>(r) * host.stride,
                      src[plane]->data() +
                          static_cast<std::size_t>(r) * pw,
                      static_cast<std::size_t>(cols) * sizeof(std::uint16_t));
        }
      }
      if (!check(cudaMemcpy2DAsync(
                     planes_[plane].recon +
                         static_cast<std::size_t>(py) *
                             static_cast<std::size_t>(
                                 planes_[plane].recon_stride) +
                         static_cast<std::size_t>(px),
                     static_cast<std::size_t>(planes_[plane].recon_stride) *
                         sizeof(std::uint16_t),
                     src[plane]->data(),
                     static_cast<std::size_t>(pw) * sizeof(std::uint16_t),
                     static_cast<std::size_t>(cols) * sizeof(std::uint16_t),
                     static_cast<std::size_t>(rows), cudaMemcpyHostToDevice,
                     stream_),
                 "recon upload", error)) {
        return false;
      }
    }
    return true;
  }

  std::uint64_t required_device_bytes() const override {
    return required_bytes_;
  }
  std::uint64_t last_required_device_bytes() const override {
    return required_bytes_;
  }

 private:
  static constexpr std::size_t kMaxBatch = 256;
  static constexpr int kMaxBlock = 64;

  bool ensure_arena(std::string& error) {
    if (external_arena_ != nullptr) {
      arena_ = external_arena_;
      owns_arena_ = false;
      if (!arena_->initialized()) {
        error = "format video CUDA: the supplied arena is not initialized";
        return false;
      }
      return true;
    }
    if (arena_ != nullptr) return true;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::uint64_t budget = 256 * kMiB;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
      const std::uint64_t available =
          std::min<std::uint64_t>(total_bytes, free_bytes);
      if (available > av2enc::cuda::Arena::kDeviceHeadroomBytes) {
        budget = available - av2enc::cuda::Arena::kDeviceHeadroomBytes;
      }
    }
    const std::uint64_t cap = av2enc::cuda::Arena::capacity_cap_bytes();
    if (cap != 0) budget = std::min(budget, cap);
    budget = std::max<std::uint64_t>(budget, 256 * kMiB);
    owned_arena_ = std::make_unique<av2enc::cuda::Arena>();
    const av2enc::cuda::Status status =
        owned_arena_->init(budget, av2enc::cuda::Arena::kAlignment, &error);
    if (!av2enc::cuda::is_ok(status)) {
      error = "format video CUDA: arena init failed: " + error;
      return false;
    }
    arena_ = owned_arena_.get();
    owns_arena_ = true;
    return true;
  }

  bool block_chroma(int w, int h, int* cw, int* ch, std::string& error) {
    if (!av2enc::fmt::has_chroma(format_.chroma)) {
      *cw = 0;
      *ch = 0;
      return true;
    }
    const auto size =
        av2enc::fmt::chroma_residual_size(w, h, format_.chroma);
    if (!size.has_value()) {
      error = "invalid chroma residual size";
      return false;
    }
    *cw = av2enc::fmt::block_width(*size);
    *ch = av2enc::fmt::block_height(*size);
    return true;
  }

  bool allocate_planes(const BeginFrame& frame, std::string& error) {
    const int mi_cols = frame.mi_cols;
    const int mi_rows = frame.mi_rows;
    const int sub_x = av2enc::fmt::subsampling_x(format_.chroma);
    const int sub_y = av2enc::fmt::subsampling_y(format_.chroma);
    const bool has_chroma = av2enc::fmt::has_chroma(format_.chroma);
    const int plane_count = has_chroma ? 3 : 1;
    std::uint64_t bytes = 0;
    for (int plane = 0; plane < plane_count; ++plane) {
      const int sw = plane == 0 ? frame.width : (frame.width >> sub_x);
      const int sh = plane == 0 ? frame.height : (frame.height >> sub_y);
      const int cwp = plane == 0 ? mi_cols * 4 : ((mi_cols * 4) >> sub_x);
      const int chp = plane == 0 ? mi_rows * 4 : ((mi_rows * 4) >> sub_y);
      bytes += static_cast<std::uint64_t>(sw) * sh * sizeof(std::uint16_t);
      bytes += static_cast<std::uint64_t>(cwp) * chp * sizeof(std::uint16_t);
      bytes += static_cast<std::uint64_t>(cwp) * chp * sizeof(std::uint16_t) *
               kMaxRefs;
      bytes += static_cast<std::uint64_t>(kMaxBlock) * kMaxBlock *
               sizeof(std::uint16_t);
    }
    bytes += 4 * 1024 * 1024;  // TX scratch, descriptors, MV/SSE outputs
    required_bytes_ = bytes;
    if (arena_mark_ == 0) {
      arena_mark_ = arena_->mark();
    } else if (!arena_->rewind(arena_mark_)) {
      error = "format video CUDA: arena rewind failed";
      return false;
    }
    for (int plane = 0; plane < plane_count; ++plane) {
      const int sw = plane == 0 ? frame.width : (frame.width >> sub_x);
      const int sh = plane == 0 ? frame.height : (frame.height >> sub_y);
      const int cwp = plane == 0 ? mi_cols * 4 : ((mi_cols * 4) >> sub_x);
      const int chp = plane == 0 ? mi_rows * 4 : ((mi_rows * 4) >> sub_y);
      std::uint16_t* source = arena_->allocate_array<std::uint16_t>(
          static_cast<std::size_t>(sw) * sh);
      std::uint16_t* recon = arena_->allocate_array<std::uint16_t>(
          static_cast<std::size_t>(cwp) * chp);
      std::uint16_t* pred = arena_->allocate_array<std::uint16_t>(
          static_cast<std::size_t>(kMaxBlock) * kMaxBlock);
      if (source == nullptr || recon == nullptr || pred == nullptr) {
        error = "format video CUDA: arena too small for the frame planes";
        return false;
      }
      planes_[plane].source = source;
      planes_[plane].source_stride = sw;
      planes_[plane].source_w = sw;
      planes_[plane].source_h = sh;
      planes_[plane].recon = recon;
      planes_[plane].recon_stride = cwp;
      planes_[plane].recon_w = cwp;
      planes_[plane].recon_h = chp;
      d_pred_[plane] = pred;
      for (int r = 0; r < kMaxRefs; ++r) {
        std::uint16_t* ref = arena_->allocate_array<std::uint16_t>(
            static_cast<std::size_t>(cwp) * chp);
        if (ref == nullptr) {
          error = "format video CUDA: arena too small for the reference pool";
          return false;
        }
        planes_[plane].refs[r] = ref;
      }
    }
    d_tx_jobs_ = arena_->allocate_array<av2enc::cuda::hbd::HbdTxJob>(3);
    d_tx.pred = arena_->allocate_array<std::uint16_t>(
        static_cast<std::size_t>(kMaxBlock) * kMaxBlock);
    d_tx.residual_adj = arena_->allocate_array<std::int16_t>(
        av2enc::cuda::hbd::kHbdTxMaxAdjElems);
    d_tx.coeff = arena_->allocate_array<double>(
        av2enc::cuda::hbd::kHbdTxMaxAdjElems);
    d_tx.levels = arena_->allocate_array<std::int32_t>(
        av2enc::cuda::hbd::kHbdTxMaxAdjElems);
    d_tx.dequant = arena_->allocate_array<std::int32_t>(
        av2enc::cuda::hbd::kHbdTxMaxAdjElems);
    d_tx.residual_out = arena_->allocate_array<std::int16_t>(
        static_cast<std::size_t>(kMaxBlock) * kMaxBlock);
    d_tx.eob = arena_->allocate_array<std::int32_t>(3);
    d_tx_scratch_ = arena_->allocate_array<double>(
        av2enc::cuda::hbd::kHbdTxMaxAdjElems);
    d_mv_ = arena_->allocate_array<int>(kMaxBatch * 2);
    d_sse_ = arena_->allocate_array<std::int64_t>(1);
    if (d_tx_jobs_ == nullptr || d_tx.pred == nullptr || d_tx.residual_adj == nullptr ||
        d_tx.coeff == nullptr || d_tx.levels == nullptr ||
        d_tx.dequant == nullptr || d_tx.residual_out == nullptr ||
        d_tx.eob == nullptr || d_tx_scratch_ == nullptr ||
        d_mv_ == nullptr || d_sse_ == nullptr) {
      error = "format video CUDA: arena too small for the TX scratch";
      return false;
    }
    d_tx.residual_in = nullptr;
    d_tx.error_count = nullptr;
    return true;
  }

  bool upload_source(const BeginFrame& frame, std::string& error) {
    for (int plane = 0; plane < 3; ++plane) {
      const PlaneView& view = frame.source.plane[plane];
      if (view.data == nullptr) continue;
      const int rows = std::min(view.height, planes_[plane].source_h);
      const int cols = std::min(view.width, planes_[plane].source_w);
      if (!check(cudaMemcpy2DAsync(
                     planes_[plane].source,
                     static_cast<std::size_t>(planes_[plane].source_stride) *
                         sizeof(std::uint16_t),
                     view.data,
                     static_cast<std::size_t>(view.stride) *
                         sizeof(std::uint16_t),
                     static_cast<std::size_t>(cols) * sizeof(std::uint16_t),
                     static_cast<std::size_t>(rows), cudaMemcpyHostToDevice,
                     stream_),
                 "source upload", error)) {
        return false;
      }
    }
    return true;
  }

  bool zero_recon(std::string& error) {
    for (int plane = 0; plane < 3; ++plane) {
      if (planes_[plane].recon == nullptr) continue;
      const std::size_t count = static_cast<std::size_t>(
          planes_[plane].recon_stride) * planes_[plane].recon_h;
      if (!check(cudaMemsetAsync(planes_[plane].recon, 0,
                                 count * sizeof(std::uint16_t), stream_),
                 "recon clear", error)) {
        return false;
      }
    }
    return true;
  }

  bool upload_references(const BeginFrame& frame, std::string& error) {
    const int count = std::min(frame.num_refs, kMaxRefs);
    for (int r = 0; r < count; ++r) {
      for (int plane = 0; plane < 3; ++plane) {
        const PlaneView& view = frame.refs[r].plane[plane];
        if (view.data == nullptr) continue;
        const int rows = std::min(view.height, planes_[plane].recon_h);
        const int cols = std::min(view.width, planes_[plane].recon_w);
        if (!check(cudaMemcpy2DAsync(
                       planes_[plane].refs[r],
                       static_cast<std::size_t>(planes_[plane].recon_stride) *
                           sizeof(std::uint16_t),
                       view.data,
                       static_cast<std::size_t>(view.stride) *
                           sizeof(std::uint16_t),
                       static_cast<std::size_t>(cols) * sizeof(std::uint16_t),
                       static_cast<std::size_t>(rows), cudaMemcpyHostToDevice,
                       stream_),
                   "reference upload", error)) {
          return false;
        }
      }
    }
    return true;
  }

  bool make_mc_job(const InterJob& in, int plane, McJob* out,
                   std::string& error) {
    if (in.ref_index < 0 || in.ref_index >= frame_.num_refs) {
      error = "inter reference index out of range";
      return false;
    }
    int cw = 0;
    int ch = 0;
    if (!block_chroma(in.w, in.h, &cw, &ch, error)) return false;
    const int pw = plane == 0 ? in.w : cw;
    const int ph = plane == 0 ? in.h : ch;
    out->ref = planes_[plane].refs[in.ref_index];
    out->ref_stride = planes_[plane].recon_stride;
    out->ref_w = planes_[plane].recon_w;
    out->ref_h = planes_[plane].recon_h;
    out->x = in.x >> (plane == 0 ? 0 : av2enc::fmt::subsampling_x(format_.chroma));
    out->y = in.y >> (plane == 0 ? 0 : av2enc::fmt::subsampling_y(format_.chroma));
    out->w = pw;
    out->h = ph;
    out->mv_y = in.mv_y;
    out->mv_x = in.mv_x;
    out->sub_x = plane == 0 ? 0 : av2enc::fmt::subsampling_x(format_.chroma);
    out->sub_y = plane == 0 ? 0 : av2enc::fmt::subsampling_y(format_.chroma);
    out->dst = d_pred_[plane];
    out->dst_stride = pw;
    return true;
  }

  bool predict_and_transform(int x, int y, int w, int h, bool use_mc,
                             const InterJob& job,
                             std::array<PlaneTx, 3>& planes,
                             std::string& error) {
    int cw = 0;
    int ch = 0;
    if (!block_chroma(w, h, &cw, &ch, error)) return false;
    const int sizes[3][2] = {{w, h}, {cw, ch}, {cw, ch}};
    const int plane_count = av2enc::fmt::has_chroma(format_.chroma) ? 3 : 1;
    for (int plane = 0; plane < plane_count; ++plane) {
      const int pw = sizes[plane][0];
      const int ph = sizes[plane][1];
      const int px = x >> (plane == 0 ? 0 : av2enc::fmt::subsampling_x(format_.chroma));
      const int py = y >> (plane == 0 ? 0 : av2enc::fmt::subsampling_y(format_.chroma));
      if (use_mc) {
        McJob mc;
        if (!make_mc_job(job, plane, &mc, error)) return false;
        mc_kernel<<<1, 1, 0, stream_>>>(mc, depth_);
      } else {
        DcJob dc;
        dc.recon = planes_[plane].recon;
        dc.stride = planes_[plane].recon_stride;
        dc.max_x = planes_[plane].recon_w - 1;
        dc.max_y = planes_[plane].recon_h - 1;
        dc.x = px;
        dc.y = py;
        dc.w = pw;
        dc.h = ph;
        dc.have_above = py > 0;
        dc.have_left = px > 0;
        dc.pred = d_pred_[plane];
        dc.pred_stride = pw;
        dc_predict_kernel<<<1, 1, 0, stream_>>>(dc, depth_);
      }
      av2enc::cuda::hbd::HbdTxJob tx;
      if (!av2enc::cuda::hbd::hbd_tx_init_job(&tx, pw, ph,
                                              av2enc::txv::TxType::kDctDct)) {
        error = "GPU TX: unsupported transform size";
        return false;
      }
      tx.input = static_cast<std::uint8_t>(
          av2enc::cuda::hbd::HbdTxInput::kSourcePred);
      tx.src_packed = 0;
      tx.src_x = static_cast<std::int16_t>(px);
      tx.src_y = static_cast<std::int16_t>(py);
      tx.src_width = static_cast<std::int16_t>(planes_[plane].source_w);
      tx.src_height = static_cast<std::int16_t>(planes_[plane].source_h);
      tx.src_offset = 0;
      tx.src_stride =
          static_cast<std::int32_t>(planes_[plane].source_stride);
      tx.pred_offset = 0;
      tx.residual_adj_offset = 0;
      tx.coeff_offset = 0;
      tx.levels_offset = 0;
      tx.dequant_offset = 0;
      tx.residual_out_offset = 0;
      tx.eob_offset = plane;
      tx.quant_step = quant_step_;
      tx.params.quant_factor *=
          static_cast<double>(av2enc::txv::kQuantStep) /
          static_cast<double>(quant_step_);
      if (!check(cudaMemcpyAsync(d_tx_jobs_ + plane, &tx, sizeof(tx),
                                 cudaMemcpyHostToDevice, stream_),
                 "tx job upload", error)) {
        return false;
      }
    }
    if (!check(cudaGetLastError(), "prediction kernel", error)) return false;
    // One batch for all planes: each plane uses its own source pointer, so run
    // them individually with the shared scratch buffers.
    for (int plane = 0; plane < plane_count; ++plane) {
      const int pw = sizes[plane][0];
      const int ph = sizes[plane][1];
      av2enc::cuda::hbd::HbdTxBuffers buffers;
      buffers.src = planes_[plane].source;
      buffers.pred = d_pred_[plane];
      buffers.residual_adj = d_tx.residual_adj;
      buffers.coeff = d_tx.coeff;
      buffers.levels = d_tx.levels;
      buffers.dequant = d_tx.dequant;
      buffers.residual_out = d_tx.residual_out;
      buffers.eob = d_tx.eob;
      if (!check(av2enc::cuda::hbd::hbd_tx_batch(d_tx_jobs_ + plane, 1,
                                                 buffers, depth_, 4096,
                                                 stream_),
                 "hbd_tx_batch", error)) {
        return false;
      }
      const auto params_opt =
          av2enc::txv::make_tx_params(pw, ph, av2enc::txv::TxType::kDctDct);
      if (!params_opt.has_value()) {
        error = "GPU TX: unsupported transform size";
        return false;
      }
      const int adj_area = params_opt->w * params_opt->h;
      PlaneTx& out = planes[static_cast<std::size_t>(plane)];
      out.levels.resize(static_cast<std::size_t>(adj_area));
      out.recon.resize(static_cast<std::size_t>(pw) * ph);
      std::vector<std::int16_t> residual_out(
          static_cast<std::size_t>(pw) * ph);
      std::vector<std::uint16_t> pred(static_cast<std::size_t>(pw) * ph);
      std::int32_t eob = 0;
      if (!check(cudaMemcpyAsync(out.levels.data(), d_tx.levels,
                                 out.levels.size() * sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost, stream_),
                 "tx levels download", error)) {
        return false;
      }
      if (!check(cudaMemcpyAsync(residual_out.data(), d_tx.residual_out,
                                 residual_out.size() * sizeof(std::int16_t),
                                 cudaMemcpyDeviceToHost, stream_),
                 "tx residual download", error)) {
        return false;
      }
      if (!check(cudaMemcpyAsync(pred.data(), d_pred_[plane],
                                 pred.size() * sizeof(std::uint16_t),
                                 cudaMemcpyDeviceToHost, stream_),
                 "tx pred download", error)) {
        return false;
      }
      if (!check(cudaMemcpyAsync(&eob, d_tx.eob + plane, sizeof(eob),
                                 cudaMemcpyDeviceToHost, stream_),
                 "tx eob download", error)) {
        return false;
      }
      if (!check(cudaStreamSynchronize(stream_), "tx sync", error)) return false;
      out.eob = eob;
      out.tx_size = params_opt->tx_size;
      out.adj_w = params_opt->w;
      out.adj_h = params_opt->h;
      for (std::size_t i = 0; i < out.recon.size(); ++i) {
        out.recon[i] = static_cast<std::uint16_t>(
            av2enc::video::hbd::clip1_value(
                static_cast<int>(pred[i]) + residual_out[i], depth_));
      }
    }
    return true;
  }

  Options options_{};
  BeginFrame frame_{};
  fmt::Format format_{};
  int depth_ = 8;
  int quant_step_ = av2enc::txv::kQuantStep;
  av2enc::cuda::Arena* external_arena_ = nullptr;
  av2enc::cuda::Arena* arena_ = nullptr;
  std::unique_ptr<av2enc::cuda::Arena> owned_arena_;
  bool owns_arena_ = false;
  std::uint64_t arena_mark_ = 0;
  std::uint64_t required_bytes_ = 0;
  cudaStream_t stream_ = nullptr;
  PlaneBuffers planes_[3];
  std::uint16_t* d_pred_[3] = {};
  av2enc::cuda::hbd::HbdTxJob* d_tx_jobs_ = nullptr;
  av2enc::cuda::hbd::HbdTxBuffers d_tx{};
  double* d_tx_scratch_ = nullptr;
  int* d_mv_ = nullptr;
  std::int64_t* d_sse_ = nullptr;
  static constexpr int kMaxRefs = 8;
};

}  // namespace

bool encode_video_sequence_format_gpu(
    std::span<const av2enc::video::VideoInputFrame> frames,
    const av2enc::video::VideoOptions& options,
    av2enc::video::VideoSequenceResult& result, std::string& error,
    av2enc::cuda::Arena* arena) {
  av2enc::video::hbd::Options format_options;
  format_options.format = options.format;
  format_options.base_q_idx = options.base_q_idx;
  format_options.num_total_refs = options.num_total_refs;
  format_options.num_ref_frames = options.num_ref_frames;
  format_options.mv_precision = options.mv_precision;
  format_options.me_subpel_eighth = options.me_subpel_eighth;
  format_options.min_block_size = options.min_block_size;
  format_options.max_partition_depth = options.max_partition_depth;
  format_options.me_range = options.me_range;
  format_options.force_single_block = options.force_single_block;
  format_options.force_intra = options.force_intra;
  format_options.enable_intra_fallback = options.enable_intra_fallback;
  format_options.enable_newmv = options.enable_newmv;
  format_options.enable_nearmv = options.enable_nearmv;
  format_options.enable_inter_residual = options.enable_inter_residual;
  format_options.force_inter_residual = options.force_inter_residual;
  format_options.inter_rd_lambda = options.inter_rd_lambda;
  format_options.trace = options.trace;
  if (options.enable_deblock || options.loop_filters.any() ||
      options.tile_cols > 1 || options.tile_rows > 1 || options.b_frames > 0 ||
      options.temporal_layers > 0 || options.long_term || options.olk ||
      options.leading_frames > 0 || options.sef || options.ra_switch ||
      options.ras || options.enable_intra_full || options.enable_tx_search ||
      options.enable_gme || options.enable_warp || options.compound ||
      options.drl_reorder || options.avg_cdf || options.adaptive_cdf ||
      options.cdf_continuity || options.mfh || options.video_timing) {
    error = "this M2 tool is not wired for the --bit-depth/--chroma video "
            "formats yet";
    return false;
  }
  CudaBackend backend(arena);
  return av2enc::video::hbd::encode_sequence(frames, format_options, backend,
                                             result, error);
}

}  // namespace av2enc::cuda::video::hbd_gpu
