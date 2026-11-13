// CUDA backend for the M2 P-frame pipeline.
//
// Reuses the validated device components:
//   src/cuda/inter/me        integer + subpel motion estimation
//   src/cuda/inter/mc        §7.13.3 translational motion compensation
//   src/cuda/intra_modes     DC intra prediction + residual
//   src/cuda/tx_variants     residual downsample / forward / quant / inverse
//
// Only four small kernels are M2-owned (block variance, SSE, residual and
// reconstruction); they call the inline code in video_device.cuh, shared with the
// host-emulation backend so the arithmetic cannot drift.
//
// Allocation policy: one av2enc::cuda::Arena per backend.  Persistent frame
// planes and per-block scratch are allocated once in begin_frame(); per-batch
// job arrays are marked and rewound inside each call.  There is no cudaMalloc
// and no per-frame free.  The backend can use an external Arena supplied by
// the caller, or own one sized from the device (min(total, free) - 512 MiB,
// optionally bounded by Arena::capacity_cap_bytes()).
//
// Synchronisation: every backend call returns host-visible data, so the stream
// is synchronized at each download.  One CUDA event pair per frame reports the
// GPU interval.  There is no hidden synchronisation anywhere else.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "cuda/error.h"
#include "cuda/codec/inter/mc/mc.h"
#include "cuda/codec/inter/me/me.h"
#include "cuda/codec/inter/compound/compound_full.h"
#include "cuda/codec/intra/modes/intra_modes.h"
#include "cuda/codec/transform/variants/tx_variants.h"
#include "pipeline/video/video_frame.h"

#include "video_backend.h"
#include "video_device.cuh"

namespace av2enc::cuda::video::detail {
namespace {

namespace cf = ::av2enc::cuda::compound_full;

constexpr int kMaxBlockDim = 64;
constexpr int kMaxBlockPixels = kMaxBlockDim * kMaxBlockDim;
constexpr int kMaxPlaneTx = 1024;  // adjusted transform is at most 32x32
constexpr int kMaxCodedPixels = 4096;
constexpr int kPredSlots = 12;  // global + 6 near + new + final 4:2:0
// Explicit per-frame reference planes: 8 slots covers the CPU `kMaxRefSlots`
// rotating P pool and every B/temporal plan (at most two refs per frame).
constexpr int kMaxRefSlots = 8;

constexpr std::uint64_t kMiB = 1024ull * 1024ull;constexpr std::uint64_t kArenaFloorBytes = 256ull * kMiB;
constexpr std::uint64_t kArenaSlackBytes = 16ull * kMiB;
// Conservative default for the largest variance/ME/SSE batch.  A caller that
// knows its batch size can pass a smaller or larger value through
// required_device_bytes(); exceeding it returns ArenaStatus::kArenaTooSmall.
constexpr std::size_t kDefaultMaxBatchJobs = 8192;

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

__global__ void video_variance_kernel(const VarianceNode* __restrict__ nodes,
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

__global__ void video_sse_kernel(const SseJob* __restrict__ jobs, int count) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  const SseJob job = jobs[i];
  *job.out = device::sse_against_source(
      job.src, job.src_stride, job.frame_w, job.frame_h, job.x, job.y,
      job.pred, job.pred_stride, job.w, job.h);
}

__global__ void video_residual_kernel(const std::uint8_t* __restrict__ src,
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

__global__ void video_recon_kernel(const ReconJob* __restrict__ jobs, int count) {
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

std::uint64_t align_up_u64(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

// Rewinds the arena on every return path.  Used for per-batch job arrays: the
// persistent frame buffers live below the mark.
class ScratchScope {
 public:
  explicit ScratchScope(Arena& arena) : arena_(arena), mark_(arena.mark()) {}
  ~ScratchScope() { arena_.rewind(mark_); }
  ScratchScope(const ScratchScope&) = delete;
  ScratchScope& operator=(const ScratchScope&) = delete;

 private:
  Arena& arena_;
  std::uint64_t mark_;
};

template <typename T>
bool arena_alloc_array(Arena& arena, std::size_t count, T** out,
                       const char* what, ArenaAllocation& alloc,
                       std::string& error) {
  *out = nullptr;
  if (count == 0) return true;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    alloc.status = ArenaStatus::kInvalidArgument;
    error = std::string("M2 CUDA: oversized ") + what;
    return false;
  }
  const std::size_t bytes = count * sizeof(T);
  void* raw = nullptr;
  const Status status = arena.allocate_checked(bytes, &raw);
  if (status == Status::kArenaExhausted) {
    alloc.status = ArenaStatus::kArenaTooSmall;
    alloc.required_bytes = bytes;
    error = std::string("M2 CUDA: arena too small for ") + what + " (need " +
            std::to_string(bytes) + " bytes)";
    return false;
  }
  if (!is_ok(status)) {
    alloc.status = ArenaStatus::kInvalidArgument;
    error = std::string("M2 CUDA: arena allocation failed for ") + what + ": " +
            status_name(status);
    return false;
  }
  *out = static_cast<T*>(raw);
  return true;
}

std::uint64_t required_device_bytes_impl(const VideoDeviceConfig& config) {
  if (config.width == 0 || config.height == 0 || config.mi_cols == 0 ||
      config.mi_rows == 0) {
    // Enough for one aligned allocation after base-alignment padding.
    return 2u * Arena::kAlignment - 1u;
  }
  const std::uint64_t width = config.width;
  const std::uint64_t height = config.height;
  const std::uint64_t mi_cols = config.mi_cols;
  const std::uint64_t mi_rows = config.mi_rows;
  const std::uint64_t luma = width * height;
  const std::uint64_t chroma = (width / 2) * (height / 2);
  const std::uint64_t mi_luma = (mi_cols * 4) * (mi_rows * 4);
  const std::uint64_t mi_chroma = (mi_cols * 2) * (mi_rows * 2);
  std::uint64_t bytes = 0;
  const auto add = [&bytes](std::uint64_t count, std::uint64_t element) {
    bytes += align_up_u64(count * element, Arena::kAlignment);
  };

  // Persistent frame planes (source, per-slot references, reconstruction).
  add(luma, 1);
  add(chroma, 1);
  add(chroma, 1);
  for (int slot = 0; slot < kMaxRefSlots; ++slot) {
    add(mi_luma, 1);
    add(mi_chroma, 1);
    add(mi_chroma, 1);
  }
  add(mi_luma, 1);
  add(mi_chroma, 1);
  add(mi_chroma, 1);

  // Persistent per-block scratch.
  add(static_cast<std::uint64_t>(kPredSlots) * kMaxBlockPixels, 1);
  add(3 * kMaxCodedPixels, sizeof(std::int16_t));
  add(3 * kMaxPlaneTx, sizeof(std::int16_t));
  add(3 * kMaxPlaneTx, sizeof(double));
  add(3 * kMaxPlaneTx, sizeof(std::int32_t));
  add(3 * kMaxCodedPixels, sizeof(std::int16_t));
  add(3 * kMaxCodedPixels, 1);
  add(3, sizeof(av2enc::cuda::TxBlockJob));
  add(3, sizeof(ReconJob));

  // Transient batch scratch.  Each path is marked/rewound inside its call;
  // summing the paths is conservative and keeps the sizing API simple.
  const std::uint64_t jobs = config.max_batch_jobs != 0
                                 ? config.max_batch_jobs
                                 : kDefaultMaxBatchJobs;
  add(jobs, sizeof(VarianceNode));
  add(jobs, sizeof(std::int64_t));
  add(jobs, sizeof(me::MeCoreJob));
  add(jobs, sizeof(me::MeCoreResult));
  add(kMaxRefSlots, sizeof(me::MeCoreRef));
  add(jobs, sizeof(McBlockJob));
  add(jobs, sizeof(cf::SkipJob));
  add(jobs, sizeof(SseJob));
  add(jobs, sizeof(std::int64_t));
  // inter_sse_batch() keeps at least kPredSlots prediction blocks, even for a
  // one-job batch, so size the pool from the larger of the two.
  add(std::max<std::uint64_t>(jobs, kPredSlots) *
          static_cast<std::uint64_t>(kMaxBlockPixels),
      1);
  add(3, sizeof(McBlockJob));
  add(3, sizeof(intra_modes::IntraJob));

  // Arena::init() aligns the usable base inside the reservation.
  bytes += Arena::kAlignment - 1u;
  return bytes;
}

class CudaBackend final : public Backend {
 public:
  CudaBackend() : owned_arena_(), arena_(owned_arena_), owns_arena_(true) {
    init_runtime();
  }

  explicit CudaBackend(Arena& arena)
      : owned_arena_(), arena_(arena), owns_arena_(false) {
    init_runtime();
  }

  ~CudaBackend() override {
    if (stream_ok_) cudaStreamDestroy(stream_);
    if (start_event_ != nullptr) cudaEventDestroy(start_event_);
    if (end_event_ != nullptr) cudaEventDestroy(end_event_);
  }

  const char* name() const override { return "cuda"; }

  bool begin_frame(
      const av2enc::video::VideoInputFrame& source, int mi_cols, int mi_rows,
      std::span<const av2enc::video::VideoReconFrame* const> references,
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
    if (!ensure_arena(error)) return false;
    if (!ensure_persistent(error)) return false;

    if (!upload_2d(src_planes_[0], width_, source.y.data(), width_, width_,
                   height_, error) ||
        !upload_2d(src_planes_[1], chroma_w_, source.u.data(), chroma_w_,
                   chroma_w_, chroma_h_, error) ||
        !upload_2d(src_planes_[2], chroma_w_, source.v.data(), chroma_w_,
                   chroma_w_, chroma_h_, error)) {
      return false;
    }

    ref_count_ = static_cast<int>(references.size());
    if (ref_count_ > kMaxRefSlots) {
      error = "too many references for the M2 CUDA backend";
      return false;
    }
    for (int slot = 0; slot < ref_count_; ++slot) {
      const av2enc::video::VideoReconFrame* reference = references[slot];
      if (reference == nullptr) {
        error = "reference list contains null";
        return false;
      }
      if (reference->width != source.width ||
          reference->height != source.height) {
        error = "reference dimensions differ from the current frame";
        return false;
      }
      if (!upload_2d(ref_planes_[slot][0], luma_w_, reference->y.data(),
                     width_, luma_w_, height_, error) ||
          !upload_2d(ref_planes_[slot][1], chroma_mi_w_, reference->u.data(),
                     chroma_w_, chroma_mi_w_, chroma_h_, error) ||
          !upload_2d(ref_planes_[slot][2], chroma_mi_w_, reference->v.data(),
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
    ScratchScope scratch(arena_);
    ArenaAllocation alloc;
    VarianceNode* d_nodes = nullptr;
    std::int64_t* d_costs = nullptr;
    if (!arena_alloc_array(arena_, nodes.size(), &d_nodes, "variance nodes",
                           alloc, error) ||
        !arena_alloc_array(arena_, nodes.size(), &d_costs, "variance costs",
                           alloc, error)) {
      return record_arena_failure(alloc);
    }
    upload(d_nodes, nodes.data(), nodes.size() * sizeof(VarianceNode));
    const int threads = 128;
    const int blocks = div_up(static_cast<int>(nodes.size()), threads);
    video_variance_kernel<<<blocks, threads, 0, stream_>>>(
        d_nodes, static_cast<int>(nodes.size()), src_planes_[0], width_,
        width_, height_, d_costs);
    if (!check(cudaGetLastError(), "video_variance_kernel", error)) return false;
    ++stats_.variance_launches;
    return check(download(costs.data(), d_costs,
                          nodes.size() * sizeof(std::int64_t)),
                 "variance download", error);
  }

  bool me_batch(const std::vector<MeRequest>& jobs,
                const me::MeCoreOptions& options,
                std::vector<MeOutcome>& results, std::string& error) override {
    results.assign(jobs.size(), MeOutcome{});
    if (jobs.empty()) return true;
    ScratchScope scratch(arena_);
    ArenaAllocation alloc;
    me::MeCoreJob* d_jobs = nullptr;
    me::MeCoreResult* d_results = nullptr;
    me::MeCoreRef* d_refs = nullptr;
    const std::size_t ref_count =
        static_cast<std::size_t>(std::max(ref_count_, 1));
    if (!arena_alloc_array(arena_, jobs.size(), &d_jobs, "ME jobs", alloc,
                           error) ||
        !arena_alloc_array(arena_, jobs.size(), &d_results, "ME results", alloc,
                           error) ||
        !arena_alloc_array(arena_, ref_count, &d_refs, "ME references", alloc,
                           error)) {
      return record_arena_failure(alloc);
    }
    std::vector<me::MeCoreJob> host_jobs(jobs.size());
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      host_jobs[i] = me::MeCoreJob{};
      host_jobs[i].x = jobs[i].x;
      host_jobs[i].y = jobs[i].y;
      host_jobs[i].w = jobs[i].w;
      host_jobs[i].h = jobs[i].h;
      host_jobs[i].num_refs = 1;
      host_jobs[i].ref_indices[0] = static_cast<std::int8_t>(jobs[i].ref_index);
    }
    std::vector<me::MeCoreRef> host_refs(ref_count);
    for (std::size_t slot = 0; slot < ref_count; ++slot) {
      const int s = static_cast<int>(slot);
      host_refs[slot].plane[0] =
          me::MeCorePlane{ref_planes_[s][0], luma_w_, luma_w_, luma_h_};
      host_refs[slot].plane[1] =
          me::MeCorePlane{ref_planes_[s][1], chroma_mi_w_, chroma_mi_w_,
                          chroma_mi_h_};
      host_refs[slot].plane[2] =
          me::MeCorePlane{ref_planes_[s][2], chroma_mi_w_, chroma_mi_w_,
                          chroma_mi_h_};
    }
    upload(d_jobs, host_jobs.data(),
           host_jobs.size() * sizeof(me::MeCoreJob));
    upload(d_refs, host_refs.data(), host_refs.size() * sizeof(me::MeCoreRef));

    me::MeCoreFrame cur;
    cur.width = width_;
    cur.height = height_;
    cur.plane[0] = me::MeCorePlane{src_planes_[0], width_, width_, height_};
    cur.plane[1] =
        me::MeCorePlane{src_planes_[1], chroma_w_, chroma_w_, chroma_h_};
    cur.plane[2] =
        me::MeCorePlane{src_planes_[2], chroma_w_, chroma_w_, chroma_h_};

    if (!check(av2enc::cuda::me_search_batch(
                   cur, d_refs, static_cast<int>(ref_count), d_jobs,
                   static_cast<int>(jobs.size()), options, nullptr, d_results,
                   stream_),
               "me_search_batch", error)) {
      return false;
    }
    ++stats_.me_launches;
    stats_.me_jobs += jobs.size();
    std::vector<me::MeCoreResult> host_results(jobs.size());
    if (!check(download(host_results.data(), d_results,
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
    ScratchScope scratch(arena_);
    ArenaAllocation alloc;
    McBlockJob* d_mc_jobs = nullptr;
    SseJob* d_sse_jobs = nullptr;
    std::int64_t* d_sse_out = nullptr;
    std::uint8_t* d_pred_batch = nullptr;
    const std::size_t pred_pixels =
        std::max<std::size_t>(jobs.size(), kPredSlots) * kMaxBlockPixels;
    if (!arena_alloc_array(arena_, jobs.size(), &d_mc_jobs, "MC jobs", alloc,
                           error) ||
        !arena_alloc_array(arena_, jobs.size(), &d_sse_jobs, "SSE jobs", alloc,
                           error) ||
        !arena_alloc_array(arena_, jobs.size(), &d_sse_out, "SSE output",
                           alloc, error) ||
        !arena_alloc_array(arena_, pred_pixels, &d_pred_batch,
                           "SSE prediction pool", alloc, error)) {
      return record_arena_failure(alloc);
    }
    std::vector<McBlockJob> mc_jobs(jobs.size());
    std::vector<SseJob> sse_jobs(jobs.size());
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      std::uint8_t* dst =
          d_pred_batch + i * static_cast<std::size_t>(kMaxBlockPixels);
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
      sse_jobs[i].out = d_sse_out + i;
    }
    // mc_predict_batch() takes one reference frame, so group the jobs by their
    // ref_index and launch once per reference.  The prediction destinations and
    // the SSE job array stay in the caller's one-slot-per-job order.
    for (int slot = 0; slot < ref_count_; ++slot) {
      std::vector<McBlockJob> subset;
      for (std::size_t i = 0; i < jobs.size(); ++i) {
        if (jobs[i].ref_index == slot) subset.push_back(mc_jobs[i]);
      }
      if (subset.empty()) continue;
      upload(d_mc_jobs, subset.data(), subset.size() * sizeof(McBlockJob));
      if (!check(av2enc::cuda::mc_predict_batch(
                     ref_frame(slot), d_mc_jobs, static_cast<int>(subset.size()),
                     stream_),
                 "mc_predict_batch", error)) {
        return false;
      }
      ++stats_.mc_launches;
    }
    stats_.inter_candidates += jobs.size();

    upload(d_sse_jobs, sse_jobs.data(), sse_jobs.size() * sizeof(SseJob));
    const int threads = 128;
    const int blocks = div_up(static_cast<int>(jobs.size()), threads);
    video_sse_kernel<<<blocks, threads, 0, stream_>>>(d_sse_jobs,
                                                   static_cast<int>(jobs.size()));
    if (!check(cudaGetLastError(), "video_sse_kernel", error)) return false;
    ++stats_.sse_launches;
    return check(download(sse.data(), d_sse_out,
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
    ScratchScope scratch(arena_);
    ArenaAllocation alloc;
    McBlockJob* d_jobs = nullptr;
    if (!arena_alloc_array(arena_, 3, &d_jobs, "MC 4:2:0 jobs", alloc, error)) {
      return record_arena_failure(alloc);
    }

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
    upload(d_jobs, jobs, sizeof(jobs));
    if (!check(av2enc::cuda::mc_predict_batch(ref_frame(luma_job.ref_index), d_jobs, 3,
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

  bool compound_sse_batch(const std::vector<InterJob>& jobs,
                          std::vector<std::int64_t>& sse,
                          std::string& error) override {
    sse.assign(jobs.size(), 0);
    if (jobs.empty()) return true;
    if (ref_count_ < 2) {
      error = "M2 CUDA compound prediction needs two references";
      return false;
    }
    ScratchScope scratch(arena_);
    ArenaAllocation alloc;
    cf::SkipJob* d_jobs = nullptr;
    SseJob* d_sse_jobs = nullptr;
    std::int64_t* d_sse_out = nullptr;
    if (!arena_alloc_array(arena_, jobs.size(), &d_jobs, "compound jobs",
                           alloc, error) ||
        !arena_alloc_array(arena_, jobs.size(), &d_sse_jobs, "compound SSE jobs",
                           alloc, error) ||
        !arena_alloc_array(arena_, jobs.size(), &d_sse_out, "compound SSE out",
                           alloc, error)) {
      return record_arena_failure(alloc);
    }
    std::vector<cf::SkipJob> host_jobs(jobs.size());
    std::vector<SseJob> sse_jobs(jobs.size());
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      const InterJob& job = jobs[i];
      std::uint8_t* luma = recon_planes_[0];
      std::uint8_t* u = recon_planes_[1];
      std::uint8_t* v = recon_planes_[2];
      cf::SkipJob& out = host_jobs[i];
      out = cf::SkipJob{};
      out.bit_depth = 8;
      out.frame_w = width_;
      out.frame_h = height_;
      out.mi_cols = mi_cols_;
      out.mi_rows = mi_rows_;
      out.x = job.x;
      out.y = job.y;
      out.w = job.w;
      out.h = job.h;
      out.bsize = 0;
      out.mv_row = job.mv_y;
      out.mv_col = job.mv_x;
      out.filter = static_cast<int>(inter::mc::InterpFilter::kEighttap);
      out.cwp_weight = 8;
      const int plane_w[3] = {luma_w_, chroma_mi_w_, chroma_mi_w_};
      const int plane_h[3] = {luma_h_, chroma_mi_h_, chroma_mi_h_};
      const int strides[3] = {luma_w_, chroma_mi_w_, chroma_mi_w_};
      std::uint8_t* dsts[3] = {luma, u, v};
      for (int plane = 0; plane < 3; ++plane) {
        for (int list = 0; list < 2; ++list) {
          cf::RefPlane& ref = out.ref[list][plane];
          ref.data = ref_planes_[list][plane];
          ref.stride = strides[plane];
          ref.width = plane_w[plane];
          ref.height = plane_h[plane];
          ref.is_16bit = 0;
        }
        out.dst[plane] = dsts[plane];
        out.dst_stride[plane] = strides[plane];
        out.dst_16bit[plane] = 0;
      }
      sse_jobs[i].src = src_planes_[0];
      sse_jobs[i].src_stride = width_;
      sse_jobs[i].frame_w = width_;
      sse_jobs[i].frame_h = height_;
      sse_jobs[i].pred = recon_planes_[0] +
                         static_cast<std::size_t>(job.y) *
                             static_cast<std::size_t>(luma_w_) +
                         static_cast<std::size_t>(job.x);
      sse_jobs[i].pred_stride = luma_w_;
      sse_jobs[i].x = job.x;
      sse_jobs[i].y = job.y;
      sse_jobs[i].w = job.w;
      sse_jobs[i].h = job.h;
      sse_jobs[i].out = d_sse_out + i;
    }
    upload(d_jobs, host_jobs.data(), host_jobs.size() * sizeof(cf::SkipJob));
    if (!check(cf::skip_batch(d_jobs, static_cast<int>(jobs.size()), stream_),
               "compound skip_batch", error)) {
      return false;
    }
    ++stats_.mc_launches;
    stats_.inter_candidates += jobs.size();

    upload(d_sse_jobs, sse_jobs.data(), sse_jobs.size() * sizeof(SseJob));
    const int threads = 128;
    const int blocks = div_up(static_cast<int>(jobs.size()), threads);
    video_sse_kernel<<<blocks, threads, 0, stream_>>>(d_sse_jobs,
                                                   static_cast<int>(jobs.size()));
    if (!check(cudaGetLastError(), "video_sse_kernel", error)) return false;
    ++stats_.sse_launches;
    return check(download(sse.data(), d_sse_out,
                          jobs.size() * sizeof(std::int64_t)),
                 "compound SSE download", error);
  }

  bool compound_420(const InterJob& luma_job, std::vector<std::uint8_t>& y,
                    std::vector<std::uint8_t>& u,
                    std::vector<std::uint8_t>& v, std::string& error) override {
    if (ref_count_ < 2) {
      error = "M2 CUDA compound prediction needs two references";
      return false;
    }
    const int w = luma_job.w;
    const int h = luma_job.h;
    const int cw = w / 2;
    const int ch = h / 2;
    ScratchScope scratch(arena_);
    ArenaAllocation alloc;
    cf::SkipJob* d_job = nullptr;
    if (!arena_alloc_array(arena_, 1, &d_job, "compound 4:2:0 job", alloc,
                           error)) {
      return record_arena_failure(alloc);
    }
    cf::SkipJob job;
    job.bit_depth = 8;
    job.frame_w = width_;
    job.frame_h = height_;
    job.mi_cols = mi_cols_;
    job.mi_rows = mi_rows_;
    job.x = luma_job.x;
    job.y = luma_job.y;
    job.w = w;
    job.h = h;
    job.bsize = 0;
    job.mv_row = luma_job.mv_y;
    job.mv_col = luma_job.mv_x;
    job.filter = static_cast<int>(inter::mc::InterpFilter::kEighttap);
    job.cwp_weight = 8;
    const int plane_w[3] = {luma_w_, chroma_mi_w_, chroma_mi_w_};
    const int plane_h[3] = {luma_h_, chroma_mi_h_, chroma_mi_h_};
    const int strides[3] = {luma_w_, chroma_mi_w_, chroma_mi_w_};
    std::uint8_t* dsts[3] = {recon_planes_[0], recon_planes_[1],
                             recon_planes_[2]};
    const int dst_strides[3] = {luma_w_, chroma_mi_w_, chroma_mi_w_};
    for (int plane = 0; plane < 3; ++plane) {
      for (int list = 0; list < 2; ++list) {
        cf::RefPlane& ref = job.ref[list][plane];
        ref.data = ref_planes_[list][plane];
        ref.stride = strides[plane];
        ref.width = plane_w[plane];
        ref.height = plane_h[plane];
        ref.is_16bit = 0;
      }
      job.dst[plane] = dsts[plane];
      job.dst_stride[plane] = dst_strides[plane];
      job.dst_16bit[plane] = 0;
    }
    upload(d_job, &job, sizeof(job));
    if (!check(cf::skip_batch(d_job, 1, stream_), "compound skip_batch 4:2:0",
               error)) {
      return false;
    }
    ++stats_.mc_launches;
    y.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    v.resize(u.size());
    if (!download_2d(y.data(), w,
                     recon_planes_[0] +
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

    if (!persistent_ready_) {
      error = "M2 CUDA: begin_frame() has not allocated the frame buffers";
      return false;
    }

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

    ScratchScope scratch(arena_);
    ArenaAllocation alloc;
    intra_modes::IntraJob* d_jobs = nullptr;
    if (!arena_alloc_array(arena_, 3, &d_jobs, "intra jobs", alloc, error)) {
      return record_arena_failure(alloc);
    }
    upload(d_jobs, jobs.data(), jobs.size() * sizeof(intra_modes::IntraJob));
    if (!check(intra_modes::intra_predict_batch(d_jobs, 3, stream_),
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
    if (!persistent_ready_ || d_levels_ == nullptr || d_recon_out_ == nullptr ||
        d_recon_residual_ == nullptr || d_residual_adj_ == nullptr ||
        d_coeff_ == nullptr || d_pred_ == nullptr || d_tx_jobs_ == nullptr ||
        d_recon_jobs_ == nullptr) {
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

    // d_tx_jobs_, d_recon_jobs_ and d_recon_out_ are persistent and were
    // allocated by begin_frame().
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
    video_recon_kernel<<<3, kReconThreads, 0, stream_>>>(d_recon_jobs_, 3);
    if (!check(cudaGetLastError(), "video_recon_kernel", error)) return false;
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

    if (!persistent_ready_) {
      error = "M2 CUDA: begin_frame() has not allocated the frame buffers";
      return false;
    }
    ScratchScope scratch(arena_);
    ArenaAllocation alloc;
    McBlockJob* d_mc_jobs = nullptr;
    if (!arena_alloc_array(arena_, 3, &d_mc_jobs, "residual MC jobs", alloc,
                           error)) {
      return record_arena_failure(alloc);
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
    upload(d_mc_jobs, mc_jobs, sizeof(mc_jobs));
    if (!check(av2enc::cuda::mc_predict_batch(ref_frame(luma_job.ref_index), d_mc_jobs, 3,
                                              stream_),
               "mc_predict_batch residual 4:2:0", error)) {
      return false;
    }
    ++stats_.mc_launches;

    constexpr int kResidualThreads = 256;
    for (int plane = 0; plane < 3; ++plane) {
      const int pixels = coded_w[plane] * coded_h[plane];
      video_residual_kernel<<<div_up(pixels, kResidualThreads),
                           kResidualThreads, 0, stream_>>>(
          src_planes_[plane], src_stride[plane], src_w[plane], src_h[plane],
          ox[plane], oy[plane],
          d_pred_ + static_cast<std::size_t>(plane) * kMaxBlockPixels,
          coded_w[plane],
          d_residual_ + static_cast<std::size_t>(plane) * kMaxCodedPixels,
          coded_w[plane], coded_h[plane]);
      if (!check(cudaGetLastError(), "video_residual_kernel", error)) {
        return false;
      }
    }
    stats_.residual_launches += 3;
    return tx_and_recon_420(coded_w, coded_h, planes, error);
  }

  bool inter_residual_candidate(const InterJob& job, PlaneTx& out,
                                std::string& error) override {
    if (!persistent_ready_) {
      error = "M2 CUDA: begin_frame() has not allocated the frame buffers";
      return false;
    }
    ScratchScope scratch(arena_);
    ArenaAllocation alloc;
    McBlockJob* d_mc_job = nullptr;
    if (!arena_alloc_array(arena_, 1, &d_mc_job, "selftest MC job", alloc,
                           error)) {
      return record_arena_failure(alloc);
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
    upload(d_mc_job, &mc_job, sizeof(mc_job));
    if (!check(av2enc::cuda::mc_predict_batch(ref_frame(job.ref_index), d_mc_job, 1,
                                              stream_),
               "mc_predict_batch residual", error)) {
      return false;
    }
    ++stats_.mc_launches;

    const int pixels = job.w * job.h;
    const int threads = 256;
    video_residual_kernel<<<div_up(pixels, threads), threads, 0, stream_>>>(
        src_planes_[0], width_, width_, height_, job.x, job.y, d_pred_, job.w,
        d_residual_, job.w, job.h);
    if (!check(cudaGetLastError(), "video_residual_kernel", error)) return false;
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
    video_recon_kernel<<<1, 256, 0, stream_>>>(d_recon_jobs_, 1);
    if (!check(cudaGetLastError(), "selftest recon", error)) return false;
    ++stats_.recon_launches;

    const int adjusted_elems = params.w * params.h;
    if (d_levels_ == nullptr || d_recon_out_ == nullptr) {
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

  const VideoGpuStats& stats() const override { return stats_; }
  void reset_stats() override { stats_ = VideoGpuStats{}; }

 private:
  McDeviceFrame ref_frame(int slot) const {
    const int index = std::clamp(slot, 0, kMaxRefSlots - 1);
    McDeviceFrame frame;
    frame.plane[0] = ref_planes_[index][0];
    frame.plane[1] = ref_planes_[index][1];
    frame.plane[2] = ref_planes_[index][2];
    frame.stride[0] = luma_w_;
    frame.stride[1] = chroma_mi_w_;
    frame.stride[2] = chroma_mi_w_;
    frame.mi_cols = mi_cols_;
    frame.mi_rows = mi_rows_;
    return frame;
  }

  VideoDeviceConfig device_config() const {
    return VideoDeviceConfig{static_cast<std::uint32_t>(width_),
                          static_cast<std::uint32_t>(height_),
                          static_cast<std::uint32_t>(mi_cols_),
                          static_cast<std::uint32_t>(mi_rows_),
                          kDefaultMaxBatchJobs};
  }

  bool record_arena_failure(const ArenaAllocation& alloc) {
    last_arena_required_ = alloc.required_bytes;
    last_arena_status_ = alloc.status;
    return false;
  }

  void init_runtime() {
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) ==
        cudaSuccess) {
      stream_ok_ = true;
    }
    cudaEventCreate(&start_event_);
    cudaEventCreate(&end_event_);
  }

  // Initialise the private arena once, from the device budget.  An external
  // arena is the caller's responsibility; this backend only checks that it is
  // initialized.
  bool ensure_arena(std::string& error) {
    required_device_bytes_ = required_device_bytes_impl(device_config());
    if (arena_.initialized()) return true;
    if (!owns_arena_) {
      error = "M2 CUDA: the supplied Arena is not initialized";
      return false;
    }
    std::uint64_t budget = 0;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
      const std::uint64_t available =
          std::min(static_cast<std::uint64_t>(total_bytes),
                   static_cast<std::uint64_t>(free_bytes));
      budget = available > Arena::kDeviceHeadroomBytes
                   ? available - Arena::kDeviceHeadroomBytes
                   : 0;
    }
    const std::uint64_t cap = Arena::capacity_cap_bytes();
    if (cap != 0 && cap < required_device_bytes_) {
      error = "M2 CUDA: arena cap (" + std::to_string(cap) +
              " bytes) is below the required working set (" +
              std::to_string(required_device_bytes_) + " bytes)";
      last_arena_required_ = required_device_bytes_;
      last_arena_status_ = ArenaStatus::kArenaTooSmall;
      return false;
    }
    std::uint64_t desired = std::max(
        kArenaFloorBytes,
        required_device_bytes_ + required_device_bytes_ / 2u + kArenaSlackBytes);
    if (cap != 0) desired = std::min(desired, cap);
    if (budget != 0) {
      desired = std::min(desired, std::max(budget, required_device_bytes_));
    }
    std::string detail;
    const Status status = arena_.init(desired, Arena::kAlignment, &detail);
    if (!is_ok(status)) {
      last_arena_required_ = required_device_bytes_;
      last_arena_status_ = ArenaStatus::kCudaError;
      error = "M2 CUDA: arena init failed: " +
              (detail.empty() ? std::string(status_name(status)) : detail);
      return false;
    }
    return true;
  }

  void clear_persistent() {
    for (int plane = 0; plane < 3; ++plane) {
      src_planes_[plane] = nullptr;
      recon_planes_[plane] = nullptr;
    }
    for (int slot = 0; slot < kMaxRefSlots; ++slot) {
      for (int plane = 0; plane < 3; ++plane) {
        ref_planes_[slot][plane] = nullptr;
      }
    }
    ref_count_ = 0;
    d_pred_ = nullptr;
    d_residual_ = nullptr;
    d_residual_adj_ = nullptr;
    d_coeff_ = nullptr;
    d_levels_ = nullptr;
    d_recon_residual_ = nullptr;
    d_recon_out_ = nullptr;
    d_tx_jobs_ = nullptr;
    d_recon_jobs_ = nullptr;
    persistent_ready_ = false;
  }

  // Persistent buffers are allocated once per frame geometry.  Reallocating
  // rewinds to the persistent mark, so it is only legal at a frame boundary:
  // every pointer handed out earlier is invalid afterwards.
  bool ensure_persistent(std::string& error) {
    required_device_bytes_ = required_device_bytes_impl(device_config());
    if (persistent_ready_ && persistent_width_ == width_ &&
        persistent_height_ == height_ && persistent_mi_cols_ == mi_cols_ &&
        persistent_mi_rows_ == mi_rows_) {
      return true;
    }
    if (persistent_ready_) {
      arena_.rewind(persistent_mark_);
      clear_persistent();
    }
    persistent_mark_ = arena_.mark();
    ArenaAllocation alloc;
    const std::size_t luma =
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
    const std::size_t chroma = static_cast<std::size_t>(chroma_w_) *
                               static_cast<std::size_t>(chroma_h_);
    const std::size_t mi_luma = static_cast<std::size_t>(luma_w_) *
                                static_cast<std::size_t>(luma_h_);
    const std::size_t mi_chroma = static_cast<std::size_t>(chroma_mi_w_) *
                                  static_cast<std::size_t>(chroma_mi_h_);
    bool ok = true;
    for (int plane = 0; plane < 3 && ok; ++plane) {
      const std::size_t src_count = plane == 0 ? luma : chroma;
      const std::size_t mi_count = plane == 0 ? mi_luma : mi_chroma;
      ok = arena_alloc_array(arena_, src_count, &src_planes_[plane],
                             "source plane", alloc, error) &&
           arena_alloc_array(arena_, mi_count, &recon_planes_[plane],
                             "recon plane", alloc, error);
    }
    for (int slot = 0; slot < kMaxRefSlots && ok; ++slot) {
      for (int plane = 0; plane < 3 && ok; ++plane) {
        const std::size_t mi_count = plane == 0 ? mi_luma : mi_chroma;
        ok = arena_alloc_array(arena_, mi_count, &ref_planes_[slot][plane],
                               "reference plane", alloc, error);
      }
    }
    ok = ok &&
         arena_alloc_array(arena_,
                           static_cast<std::size_t>(kPredSlots) *
                               kMaxBlockPixels,
                           &d_pred_, "prediction pool", alloc, error) &&
         arena_alloc_array(arena_, 3 * kMaxCodedPixels, &d_residual_,
                           "residual", alloc, error) &&
         arena_alloc_array(arena_, 3 * kMaxPlaneTx, &d_residual_adj_,
                           "adjusted residual", alloc, error) &&
         arena_alloc_array(arena_, 3 * kMaxPlaneTx, &d_coeff_, "coeff", alloc,
                           error) &&
         arena_alloc_array(arena_, 3 * kMaxPlaneTx, &d_levels_, "levels", alloc,
                           error) &&
         arena_alloc_array(arena_, 3 * kMaxCodedPixels, &d_recon_residual_,
                           "recon residual", alloc, error) &&
         arena_alloc_array(arena_, 3, &d_tx_jobs_, "tx jobs", alloc, error) &&
         arena_alloc_array(arena_, 3, &d_recon_jobs_, "recon jobs", alloc,
                           error) &&
         arena_alloc_array(arena_, 3 * kMaxCodedPixels, &d_recon_out_,
                           "recon output", alloc, error);
    if (!ok) {
      arena_.rewind(persistent_mark_);
      clear_persistent();
      last_arena_required_ = required_device_bytes_;
      last_arena_status_ = ArenaStatus::kArenaTooSmall;
      return false;
    }
    persistent_ready_ = true;
    persistent_width_ = width_;
    persistent_height_ = height_;
    persistent_mi_cols_ = mi_cols_;
    persistent_mi_rows_ = mi_rows_;
    last_arena_status_ = ArenaStatus::kOk;
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

  std::uint64_t required_device_bytes() const override {
    return required_device_bytes_;
  }

  std::uint64_t last_arena_required_bytes() const override {
    return last_arena_required_;
  }

  ArenaStatus last_arena_status() const override { return last_arena_status_; }

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

  Arena owned_arena_;
  Arena& arena_;
  bool owns_arena_ = false;
  std::uint64_t persistent_mark_ = 0;
  bool persistent_ready_ = false;
  int persistent_width_ = 0;
  int persistent_height_ = 0;
  int persistent_mi_cols_ = 0;
  int persistent_mi_rows_ = 0;
  std::uint64_t required_device_bytes_ = 0;
  std::uint64_t last_arena_required_ = 0;
  ArenaStatus last_arena_status_ = ArenaStatus::kOk;

  std::uint8_t* src_planes_[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* ref_planes_[kMaxRefSlots][3] = {};
  int ref_count_ = 0;
  std::uint8_t* recon_planes_[3] = {nullptr, nullptr, nullptr};

  std::uint8_t* d_pred_ = nullptr;
  std::int16_t* d_residual_ = nullptr;
  std::int16_t* d_residual_adj_ = nullptr;
  double* d_coeff_ = nullptr;
  std::int32_t* d_levels_ = nullptr;
  std::int16_t* d_recon_residual_ = nullptr;
  std::uint8_t* d_recon_out_ = nullptr;
  av2enc::cuda::TxBlockJob* d_tx_jobs_ = nullptr;
  ReconJob* d_recon_jobs_ = nullptr;

  VideoGpuStats stats_{};
};

}  // namespace

std::unique_ptr<Backend> make_backend(av2enc::cuda::Arena* arena) {
  if (arena != nullptr) {
    return std::unique_ptr<Backend>(new CudaBackend(*arena));
  }
  return std::unique_ptr<Backend>(new CudaBackend());
}

const char* make_backend_name() { return "cuda"; }

}  // namespace av2enc::cuda::video::detail

namespace av2enc::cuda::video {

std::uint64_t required_device_bytes(const VideoDeviceConfig& config) {
  return detail::required_device_bytes_impl(config);
}

}  // namespace av2enc::cuda::video
