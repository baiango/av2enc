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
#include "cuda/codec/filters/deblock/deblock.h"
#include "cuda/codec/filters/cdef/cdef.h"
#include "cuda/codec/filters/loop_restoration/ccso.h"
#include "cuda/codec/filters/loop_restoration/wiener.h"
#include "cuda/codec/filters/gdf/gdf.h"
#include "cuda/codec/inter/mc/mc.h"
#include "cuda/codec/inter/me/me.h"
#include "cuda/codec/inter/compound/compound_full.h"
#include "cuda/codec/inter/gme/gme.h"
#include "cuda/codec/inter/warp_stack/warp_stack.h"
#include "cuda/codec/intra/modes/intra_modes.h"
#include "cuda/codec/transform/variants/tx_variants.h"
#include "codec/rate/rc/rc_tables.h"
#include "cuda/pipeline/video/tx/video_tx.h"
#include "codec/transform/tx_partition/tx_scan.h"
#include "pipeline/video/video_frame.h"
#include "pipeline/video/post/video_loop_filters.h"
#include "cuda/pipeline/video/intra/video_intra.h"
#include "pipeline/video/video_frame.h"
#include "pipeline/video/video_intra_search.h"

#include "video_backend.h"
#include "video_device.cuh"

namespace av2enc::cuda::video::detail {
namespace {

namespace cf = ::av2enc::cuda::compound_full;
namespace vi = ::av2enc::cuda::video::intra;
namespace vintra = ::av2enc::video::intra_search;
namespace videotx = ::av2enc::cuda::videotx;

// One residual block at a time: the module's plan builder caps a job at 8 plans
// and 16 candidates per leaf; every M2 coding block resolves to <= 4 plans.
inline constexpr videotx::VideoTxConfig kTxSearchWorkspaceConfig{1, 8, 16};
constexpr int kMaxBlockDim = 64;
constexpr int kMaxBlockPixels = kMaxBlockDim * kMaxBlockDim;
constexpr int kMaxPlaneTx = 1024;  // adjusted transform is at most 32x32
constexpr int kMaxCodedPixels = 4096;
constexpr int kPredSlots = 12;  // global + 6 near + new + final 4:2:0
// One warp job per scored GLOBALMV candidate.  The pipeline scores at most one
// warped candidate per block, so 64 covers a whole superblock batch with room
// to spare.
constexpr int kWarpBatchJobs = 64;
// Explicit per-frame reference planes: 8 slots covers the CPU `kMaxRefSlots`
// rotating P pool and every B/temporal plan (at most two refs per frame).
constexpr int kMaxRefSlots = 8;

constexpr std::uint64_t kMiB = 1024ull * 1024ull;constexpr std::uint64_t kArenaFloorBytes = 256ull * kMiB;
constexpr std::uint64_t kArenaSlackBytes = 16ull * kMiB;
// Conservative default for the largest variance/ME/SSE batch.  A caller that
// knows its batch size can pass a smaller or larger value through
// required_device_bytes(); exceeding it returns ArenaStatus::kArenaTooSmall.
constexpr std::size_t kDefaultMaxBatchJobs = 8192;

// Full-intra candidate workspace (--video-intra-full).  One search batch is
// three jobs (luma 61 candidates, U/V DC); the TX batch repeats them with only
// the winning luma candidate group selected.
vi::Config intra_full_config() {
  vi::Config config;
  config.max_blocks = 3;
  config.max_candidates = vi::kMaxCandidates;
  config.max_block_area = kMaxBlockPixels;
  config.enable_tx = true;
  config.max_tx_area = kMaxPlaneTx;
  return config;
}

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
  // Stride of `pred`.  A block-local predictor uses w; a sub-block of a larger
  // packed prediction (the TX search parts) uses the parent block width.
  int pred_stride;
};

// §5.18.9 fit inputs.  One device buffer set covers a whole frame's 16x16 grid
// (at most ~8200 points at 1080p, comfortably inside 16384).
gme::GmeConfig gme_config() {
  gme::GmeConfig config;
  config.max_points = 16384;
  config.max_blocks = 16384;
  config.max_warps = 64;
  config.max_jobs = 1;
  return config;
}

// Warped-prediction scoring: the warp kernel writes clipped 8-bit samples into
// an int16 buffer (compound = 0), so SSE is a direct int16 -> source compare.
struct WarpSseJob {
  const std::int16_t* pred;
  int pred_stride;
  const std::uint8_t* src;
  int src_stride;
  int frame_w;
  int frame_h;
  int x;
  int y;
  int w;
  int h;
  std::int64_t* out;
};

// Warped-prediction store into the reconstructed plane.
struct WarpStoreJob {
  const std::int16_t* pred;
  int pred_stride;
  std::uint8_t* dst;
  int dst_stride;
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
    const int p =
        job.pred[static_cast<std::size_t>(y) *
                     static_cast<std::size_t>(job.pred_stride != 0
                                                  ? job.pred_stride
                                                  : job.w) +
                 static_cast<std::size_t>(x)];
    const int r = job.residual[static_cast<std::size_t>(y) *
                                   static_cast<std::size_t>(job.w) +
                               static_cast<std::size_t>(x)];
    job.out[i] = device::recon_sample(p, r);
  }
}

__global__ void video_warp_sse_kernel(const WarpSseJob* __restrict__ jobs,
                                      int count) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  const WarpSseJob job = jobs[i];
  std::int64_t sse = 0;
  for (int row = 0; row < job.h; ++row) {
    const int py = job.y + row;
    const int cy = py < 0 ? 0 : (py >= job.frame_h ? job.frame_h - 1 : py);
    for (int col = 0; col < job.w; ++col) {
      const int px = job.x + col;
      const int cx = px < 0 ? 0 : (px >= job.frame_w ? job.frame_w - 1 : px);
      const int s = job.src[static_cast<std::size_t>(cy) *
                                static_cast<std::size_t>(job.src_stride) +
                            static_cast<std::size_t>(cx)];
      const int p = job.pred[static_cast<std::size_t>(row) *
                                 static_cast<std::size_t>(job.pred_stride) +
                             static_cast<std::size_t>(col)];
      const int d = s - p;
      sse += static_cast<std::int64_t>(d) * d;
    }
  }
  *job.out = sse;
}

__global__ void video_warp_store_kernel(const WarpStoreJob* __restrict__ jobs,
                                        int count) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  const WarpStoreJob job = jobs[i];
  for (int row = 0; row < job.h; ++row) {
    for (int col = 0; col < job.w; ++col) {
      const int v = job.pred[static_cast<std::size_t>(row) *
                                 static_cast<std::size_t>(job.pred_stride) +
                             static_cast<std::size_t>(col)];
      job.dst[static_cast<std::size_t>(row) *
                  static_cast<std::size_t>(job.dst_stride) +
              static_cast<std::size_t>(col)] = static_cast<std::uint8_t>(v);
    }
  }
}

__global__ void u8_to_u16_kernel(const std::uint8_t* __restrict__ src,
                              int src_stride, std::uint16_t* __restrict__ dst,
                              int dst_stride, int w, int h) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h) return;
  dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_stride) +
      static_cast<std::size_t>(x)] =
      src[static_cast<std::size_t>(y) * static_cast<std::size_t>(src_stride) +
          static_cast<std::size_t>(x)];
}

__global__ void u16_to_u8_kernel(const std::uint16_t* __restrict__ src,
                              int src_stride, std::uint8_t* __restrict__ dst,
                              int dst_stride, int w, int h) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h) return;
  const std::uint16_t v =
      src[static_cast<std::size_t>(y) * static_cast<std::size_t>(src_stride) +
          static_cast<std::size_t>(x)];
  dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_stride) +
      static_cast<std::size_t>(x)] =
      static_cast<std::uint8_t>(v > 255 ? 255 : v);
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

  // Loop-filter stage (§7.18 CDEF / §7.19 CCSO): a staging input frame, a
  // separate filtered output frame, the §7.18.1 skip/lossless MI grids, the
  // per-64x64 cdef_idx grid and the CCSO metadata.
  add(mi_luma, 1);
  add(mi_chroma, 1);
  add(mi_chroma, 1);
  add(mi_luma, 1);
  add(mi_chroma, 1);
  add(mi_chroma, 1);
  add(mi_luma, 1);  // post-deblock luma for CCSO
  add(mi_luma, 1);  // skip
  add(mi_luma, 1);  // lossless
  add(mi_luma, 1);  // lossless_chroma
  add(((mi_cols + 15) / 16) * ((mi_rows + 15) / 16), 1);  // cdef_idx
  add(3ull * av2enc::restoration::kCcsoOffsetEntries,
      sizeof(std::int16_t));  // CCSO offset tables
  add(3ull * mi_luma, 1);                   // CCSO unit grids
  // §7.20 loop restoration: pre-CDEF planes, per-unit LrType and coefficients.
  add(mi_luma, 1);
  add(mi_chroma, 1);
  add(mi_chroma, 1);
  const std::uint64_t lr_units_max = (mi_cols / 8 + 2) * (mi_rows / 8 + 2);
  add(3ull * lr_units_max, 1);
  add(3ull * lr_units_max * av2enc::restoration::kWienerNsCoeffStride,
      sizeof(std::int16_t));
  // §7.20.5 GDF: three 16-bit luma planes plus the reference-order arrays.
  add(3ull * mi_luma, sizeof(std::uint16_t));
  add(mi_luma, 1);
  add(2, sizeof(int));
  add(2, sizeof(int));
  {
    const int gdf_cols = static_cast<int>((mi_cols * 4 + 63) / 64);
    const int gdf_rows = static_cast<int>((mi_rows * 4 + 63) / 64);
    gdf::GdfConfig gdf_config;
    gdf_config.max_units = gdf::gdf_grid_slots(gdf_cols, gdf_rows);
    bytes += gdf::required_device_bytes(gdf_config);
  }

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
  // Deblock metadata scratch: one MI grid per plane plus the tile-start lists.
  add(3 * mi_cols * mi_rows, sizeof(deblock::DeblockMi));
  add(2 * (mi_cols + mi_rows), sizeof(std::uint16_t));

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
  // Opt-in luma TX search: one device workspace (persistent, one job at a time).
  bytes += videotx::required_device_bytes(kTxSearchWorkspaceConfig);

  // §7.13.3.19/.20 warp batch + §5.18.9 estimator.  Both modules report their
  // own arena footprint; the int16 prediction pool and job tables are owned by
  // this backend.
  add(static_cast<std::uint64_t>(kWarpBatchJobs) * kMaxBlockPixels,
      sizeof(std::int16_t));
  add(kWarpBatchJobs, sizeof(WarpSseJob));
  add(kWarpBatchJobs, sizeof(WarpStoreJob));
  add(kWarpBatchJobs, sizeof(std::int64_t));
  warpstack::WarpStackConfig ws_config;
  ws_config.max_predict_jobs = kWarpBatchJobs;
  bytes += align_up_u64(warpstack::required_device_bytes(ws_config),
                        Arena::kAlignment);
  bytes += align_up_u64(gme::required_device_bytes(gme_config()),
                        Arena::kAlignment);
  // Full-intra candidate workspace (--video-intra-full).  Initialised once per
  // frame geometry from this same arena; see intra_full_config().
  bytes += align_up_u64(vi::required_device_bytes(intra_full_config()),
                        Arena::kAlignment);

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
      const av2enc::video::VideoInputFrame& source, int frame_mi_cols,
      int frame_mi_rows, int tile_x, int tile_y, int mi_cols, int mi_rows,
      std::span<const av2enc::video::VideoReconFrame* const> references,
      int base_q_idx, std::string& error) override {
    if (!stream_ok_) {
      error = "CUDA stream creation failed";
      return false;
    }
    quant_step_ = static_cast<int>(rc::kQlookupQ0To255[
        std::clamp(base_q_idx, 1, 255)]);
    frame_width_ = static_cast<int>(source.width);
    frame_height_ = static_cast<int>(source.height);
    frame_chroma_w_ = frame_width_ / 2;
    frame_chroma_h_ = frame_height_ / 2;
    frame_mi_cols_ = frame_mi_cols;
    frame_mi_rows_ = frame_mi_rows;
    tile_x_ = tile_x;
    tile_y_ = tile_y;
    width_ = mi_cols * 4;
    height_ = mi_rows * 4;
    chroma_w_ = mi_cols * 2;
    chroma_h_ = mi_rows * 2;
    mi_cols_ = mi_cols;
    mi_rows_ = mi_rows;
    luma_w_ = width_;
    luma_h_ = height_;
    chroma_mi_w_ = chroma_w_;
    chroma_mi_h_ = chroma_h_;
    tiled_ = tile_x != 0 || tile_y != 0 || mi_cols != frame_mi_cols_ ||
             mi_rows != frame_mi_rows_;

    if (width_ <= 0 || height_ <= 0 || frame_mi_cols_ * 4 != frame_width_ ||
        frame_mi_rows_ * 4 != frame_height_ ||
        (tile_x_ & 1) != 0 || (tile_y_ & 1) != 0 ||
        tile_x_ + width_ > frame_width_ || tile_y_ + height_ > frame_height_) {
      error = "MI-aligned planes differ from the visible frame; M2 requires "
              "dimensions divisible by 8 and superblock-aligned tiles";
      return false;
    }
    if (!ensure_arena(error)) return false;
    if (!ensure_persistent(error)) return false;

    if (!upload_2d(src_planes_[0], frame_width_, source.y.data(), frame_width_,
                   frame_width_, frame_height_, error) ||
        !upload_2d(src_planes_[1], frame_chroma_w_, source.u.data(),
                   frame_chroma_w_, frame_chroma_w_, frame_chroma_h_, error) ||
        !upload_2d(src_planes_[2], frame_chroma_w_, source.v.data(),
                   frame_chroma_w_, frame_chroma_w_, frame_chroma_h_, error)) {
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
      if (!upload_2d(ref_planes_[slot][0], frame_width_, reference->y.data(),
                     frame_width_, frame_width_, frame_height_, error) ||
          !upload_2d(ref_planes_[slot][1], frame_chroma_w_,
                     reference->u.data(), frame_chroma_w_, frame_chroma_w_,
                     frame_chroma_h_, error) ||
          !upload_2d(ref_planes_[slot][2], frame_chroma_w_,
                     reference->v.data(), frame_chroma_w_, frame_chroma_w_,
                     frame_chroma_h_, error)) {
        return false;
      }
    }

    // Reconstruction and its §5.20.9.1 availability are tile-local: only the
    // tile-shaped prefix is used, with the tile stride.  The allocation is
    // frame-sized so every tile of a frame reuses it without a rewind.
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
        d_nodes, static_cast<int>(nodes.size()), src_planes_[0], frame_width_,
        frame_width_, frame_height_, d_costs);
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
      host_refs[slot].plane[0] = me::MeCorePlane{
          ref_planes_[s][0], frame_width_, frame_width_, frame_height_};
      host_refs[slot].plane[1] = me::MeCorePlane{
          ref_planes_[s][1], frame_chroma_w_, frame_chroma_w_,
          frame_chroma_h_};
      host_refs[slot].plane[2] = me::MeCorePlane{
          ref_planes_[s][2], frame_chroma_w_, frame_chroma_w_,
          frame_chroma_h_};
    }
    upload(d_jobs, host_jobs.data(),
           host_jobs.size() * sizeof(me::MeCoreJob));
    upload(d_refs, host_refs.data(), host_refs.size() * sizeof(me::MeCoreRef));

    me::MeCoreFrame cur;
    cur.width = frame_width_;
    cur.height = frame_height_;
    cur.plane[0] = me::MeCorePlane{src_planes_[0], frame_width_, frame_width_,
                                   frame_height_};
    cur.plane[1] = me::MeCorePlane{src_planes_[1], frame_chroma_w_,
                                   frame_chroma_w_, frame_chroma_h_};
    cur.plane[2] = me::MeCorePlane{src_planes_[2], frame_chroma_w_,
                                   frame_chroma_w_, frame_chroma_h_};

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
      sse_jobs[i].src_stride = frame_width_;
      sse_jobs[i].frame_w = frame_width_;
      sse_jobs[i].frame_h = frame_height_;
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
    const int lx = luma_job.x - tile_x_;
    const int ly = luma_job.y - tile_y_;
    const int lcx = luma_job.x / 2 - tile_x_ / 2;
    const int lcy = luma_job.y / 2 - tile_y_ / 2;
    jobs[0].dst = recon_planes_[0] +
                  static_cast<std::size_t>(ly) *
                      static_cast<std::size_t>(luma_w_) +
                  static_cast<std::size_t>(lx);
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
                        static_cast<std::size_t>(lcy) *
                            static_cast<std::size_t>(chroma_mi_w_) +
                        static_cast<std::size_t>(lcx);
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
                                     static_cast<std::size_t>(ly) *
                                         static_cast<std::size_t>(luma_w_) +
                                     static_cast<std::size_t>(lx),
                     luma_w_, w, h, error) ||
        !download_2d(u.data(), cw,
                     recon_planes_[1] +
                         static_cast<std::size_t>(lcy) *
                             static_cast<std::size_t>(chroma_mi_w_) +
                         static_cast<std::size_t>(lcx),
                     chroma_mi_w_, cw, ch, error) ||
        !download_2d(v.data(), cw,
                     recon_planes_[2] +
                         static_cast<std::size_t>(lcy) *
                             static_cast<std::size_t>(chroma_mi_w_) +
                         static_cast<std::size_t>(lcx),
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
    if (tiled_) {
      error = "M2 CUDA compound prediction is not implemented for multi-tile "
              "frames";
      return false;
    }
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
      out.frame_w = frame_width_;
      out.frame_h = frame_height_;
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
      const int ref_w[3] = {frame_width_, frame_chroma_w_, frame_chroma_w_};
      const int ref_h[3] = {frame_height_, frame_chroma_h_, frame_chroma_h_};
      const int ref_stride[3] = {frame_width_, frame_chroma_w_,
                                 frame_chroma_w_};
      const int dst_stride[3] = {luma_w_, chroma_mi_w_, chroma_mi_w_};
      std::uint8_t* dsts[3] = {luma, u, v};
      for (int plane = 0; plane < 3; ++plane) {
        for (int list = 0; list < 2; ++list) {
          cf::RefPlane& ref = out.ref[list][plane];
          ref.data = ref_planes_[list][plane];
          ref.stride = ref_stride[plane];
          ref.width = ref_w[plane];
          ref.height = ref_h[plane];
          ref.is_16bit = 0;
        }
        out.dst[plane] = dsts[plane];
        out.dst_stride[plane] = dst_stride[plane];
        out.dst_16bit[plane] = 0;
      }
      sse_jobs[i].src = src_planes_[0];
      sse_jobs[i].src_stride = frame_width_;
      sse_jobs[i].frame_w = frame_width_;
      sse_jobs[i].frame_h = frame_height_;
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
    if (tiled_) {
      error = "M2 CUDA compound prediction is not implemented for multi-tile "
              "frames";
      return false;
    }
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
    job.frame_w = frame_width_;
    job.frame_h = frame_height_;
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
    const int ref_w[3] = {frame_width_, frame_chroma_w_, frame_chroma_w_};
    const int ref_h[3] = {frame_height_, frame_chroma_h_, frame_chroma_h_};
    const int ref_stride[3] = {frame_width_, frame_chroma_w_,
                               frame_chroma_w_};
    std::uint8_t* dsts[3] = {recon_planes_[0], recon_planes_[1],
                             recon_planes_[2]};
    const int dst_strides[3] = {luma_w_, chroma_mi_w_, chroma_mi_w_};
    for (int plane = 0; plane < 3; ++plane) {
      for (int list = 0; list < 2; ++list) {
        cf::RefPlane& ref = job.ref[list][plane];
        ref.data = ref_planes_[list][plane];
        ref.stride = ref_stride[plane];
        ref.width = ref_w[plane];
        ref.height = ref_h[plane];
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

  // Fills one device warp-prediction job for `job` and `plane`.  The block
  // origin and the destination are the plane's own coordinates; the MI
  // geometry and the model stay in luma units (§7.13.3.19).
  bool fill_warp_job(const WarpJob& job, int plane, std::int16_t* dst,
                     warpstack::WarpPredictJob* out, std::string& error) const {
    if (plane < 0 || plane > 2) {
      error = "M2 CUDA: warp plane out of range";
      return false;
    }
    if (job.ref_index < 0 || job.ref_index >= ref_count_) {
      error = "M2 CUDA: warp ref_index out of range";
      return false;
    }
    const int sub_x = plane == 0 ? 0 : 1;
    const int sub_y = plane == 0 ? 0 : 1;
    const int plane_w = (plane == 0 ? luma_w_ : chroma_mi_w_);
    const int plane_h = (plane == 0 ? luma_h_ : chroma_mi_h_);
    *out = warpstack::WarpPredictJob{};
    for (int i = 0; i < 6; ++i) {
      out->params.p[i] = job.params.p[static_cast<std::size_t>(i)];
    }
    out->ref = ref_planes_[job.ref_index][plane];
    out->ref_stride = plane_w;
    out->ref_max_x = plane_w - 1;
    out->ref_max_y = plane_h - 1;
    out->geom.plane = plane;
    out->geom.sub_x = sub_x;
    out->geom.sub_y = sub_y;
    out->geom.rect.x = job.x >> sub_x;
    out->geom.rect.y = job.y >> sub_y;
    out->geom.rect.w = job.w >> sub_x;
    out->geom.rect.h = job.h >> sub_y;
    out->geom.dst_x = 0;
    out->geom.dst_y = 0;
    out->geom.mi_col = job.mi_col;
    out->geom.mi_row = job.mi_row;
    out->geom.bw4 = job.bw4;
    out->geom.bh4 = job.bh4;
    out->geom.mi_cols = mi_cols_;
    out->geom.mi_rows = mi_rows_;
    out->geom.ref_w = plane_w;
    out->geom.ref_h = plane_h;
    out->geom.frame_w = plane_w;
    out->geom.frame_h = plane_h;
    out->geom.bit_depth = 8;
    out->compound = 0;
    out->path = warpstack::kWarpPathAuto;
    out->dst = dst;
    out->dst_stride = job.w >> sub_x;
    return true;
  }

  bool warp_sse_batch(const std::vector<WarpJob>& jobs,
                      std::vector<std::int64_t>& sse,
                      std::string& error) override {
    sse.assign(jobs.size(), 0);
    if (jobs.empty()) return true;
    if (jobs.size() > static_cast<std::size_t>(kWarpBatchJobs)) {
      error = "M2 CUDA: warp batch exceeds the workspace capacity";
      return false;
    }
    if (!persistent_ready_ || !warp_workspace_.created()) {
      error = "M2 CUDA: warp workspace is not initialized";
      return false;
    }
    std::vector<warpstack::WarpPredictJob> predict(jobs.size());
    std::vector<WarpSseJob> host_sse(jobs.size());
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      std::int16_t* dst =
          d_warp_pred_ + i * static_cast<std::size_t>(kMaxBlockPixels);
      if (!fill_warp_job(jobs[i], 0, dst, &predict[i], error)) return false;
      host_sse[i] = WarpSseJob{};
      host_sse[i].pred = dst;
      host_sse[i].pred_stride = jobs[i].w;
      host_sse[i].src = src_planes_[0];
      host_sse[i].src_stride = width_;
      host_sse[i].frame_w = width_;
      host_sse[i].frame_h = height_;
      host_sse[i].x = jobs[i].x;
      host_sse[i].y = jobs[i].y;
      host_sse[i].w = jobs[i].w;
      host_sse[i].h = jobs[i].h;
      host_sse[i].out = d_warp_sse_out_ + i;
    }
    if (warp_workspace_.submit_predict(predict.data(),
                                       static_cast<int>(predict.size()),
                                       stream_) != Status::kOk) {
      error = "M2 CUDA: warp predict launch failed";
      return false;
    }
    upload(d_warp_sse_jobs_, host_sse.data(),
           host_sse.size() * sizeof(WarpSseJob));
    video_warp_sse_kernel<<<1, static_cast<unsigned>(host_sse.size()), 0,
                            stream_>>>(d_warp_sse_jobs_,
                                       static_cast<int>(host_sse.size()));
    if (!check(cudaGetLastError(), "warp SSE kernel", error)) return false;
    if (!check(download(sse.data(), d_warp_sse_out_,
                        jobs.size() * sizeof(std::int64_t)),
               "warp SSE download", error)) {
      return false;
    }
    ++stats_.mc_launches;
    ++stats_.sse_launches;
    stats_.inter_candidates += jobs.size();
    return true;
  }

  bool warp_420(const WarpJob& luma_job, std::vector<std::uint8_t>& y,
                std::vector<std::uint8_t>& u, std::vector<std::uint8_t>& v,
                std::string& error) override {
    if (!persistent_ready_ || !warp_workspace_.created()) {
      error = "M2 CUDA: warp workspace is not initialized";
      return false;
    }
    const int w = luma_job.w;
    const int h = luma_job.h;
    const int cw = w / 2;
    const int ch = h / 2;
    std::vector<warpstack::WarpPredictJob> predict(3);
    for (int plane = 0; plane < 3; ++plane) {
      std::int16_t* dst =
          d_warp_pred_ + static_cast<std::size_t>(plane) * kMaxBlockPixels;
      if (!fill_warp_job(luma_job, plane, dst, &predict[plane], error)) {
        return false;
      }
    }
    if (warp_workspace_.submit_predict(predict.data(), 3, stream_) !=
        Status::kOk) {
      error = "M2 CUDA: warp predict launch failed";
      return false;
    }
    const int plane_w[3] = {w, cw, cw};
    const int plane_h[3] = {h, ch, ch};
    std::uint8_t* recon[3] = {recon_planes_[0], recon_planes_[1],
                              recon_planes_[2]};
    const int strides[3] = {luma_w_, chroma_mi_w_, chroma_mi_w_};
    const int origins_x[3] = {luma_job.x, luma_job.x / 2, luma_job.x / 2};
    const int origins_y[3] = {luma_job.y, luma_job.y / 2, luma_job.y / 2};
    std::vector<WarpStoreJob> store(3);
    for (int plane = 0; plane < 3; ++plane) {
      store[static_cast<std::size_t>(plane)] = WarpStoreJob{};
      WarpStoreJob& job = store[static_cast<std::size_t>(plane)];
      job.pred = d_warp_pred_ +
                 static_cast<std::size_t>(plane) * kMaxBlockPixels;
      job.pred_stride = plane_w[plane];
      job.dst = recon[plane] +
                static_cast<std::size_t>(origins_y[plane]) *
                    static_cast<std::size_t>(strides[plane]) +
                static_cast<std::size_t>(origins_x[plane]);
      job.dst_stride = strides[plane];
      job.w = plane_w[plane];
      job.h = plane_h[plane];
    }
    upload(d_warp_store_jobs_, store.data(), store.size() * sizeof(WarpStoreJob));
    video_warp_store_kernel<<<1, 3, 0, stream_>>>(d_warp_store_jobs_, 3);
    if (!check(cudaGetLastError(), "warp store kernel", error)) return false;

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
    ++stats_.mc_launches;
    stats_.bytes_downloaded +=
        static_cast<std::uint64_t>(y.size() + u.size() + v.size());
    return true;
  }

  bool gme_estimate(const std::vector<inter::gme::GmBlockMv>& blocks,
                    const inter::gme::GmOptions& options,
                    inter::gme::GmEstimate& estimate,
                    std::string& error) override {
    estimate = inter::gme::GmEstimate{};
    if (!persistent_ready_) {
      error = "M2 CUDA: global-motion buffers are not initialized";
      return false;
    }
    if (blocks.size() > static_cast<std::size_t>(gme_config().max_blocks)) {
      error = "M2 CUDA: too many block correspondences for the estimator";
      return false;
    }
    std::vector<gme::GmeBlockMv> device_blocks(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      device_blocks[i].x = blocks[i].x;
      device_blocks[i].y = blocks[i].y;
      device_blocks[i].mv_x = blocks[i].mv_x;
      device_blocks[i].mv_y = blocks[i].mv_y;
      device_blocks[i].weight = blocks[i].weight;
    }
    if (gme::stage_blocks(gme_buffers_, 0, device_blocks.data(),
                          static_cast<int>(device_blocks.size()), stream_,
                          &error) != gme::GmeStatus::kOk) {
      return false;
    }
    gme::GmeOptions device_options{};
    device_options.model = static_cast<gme::GmeI32>(options.model);
    device_options.min_points = options.min_points;
    device_options.max_points = options.max_points;
    device_options.outlier_passes = options.outlier_passes;
    device_options.outlier_threshold_q8 = options.outlier_threshold_q8;
    device_options.allow_rotzoom_degrade = options.allow_rotzoom_degrade ? 1 : 0;
    device_options.require_shear_valid = options.require_shear_valid ? 1 : 0;
    device_options.rotzoom_mismatch_q16 = options.rotzoom_mismatch_q16;
    device_options.min_scale_q16 = options.min_scale_q16;
    device_options.max_scale_q16 = options.max_scale_q16;
    device_options.max_corner_displacement_q8 =
        options.max_corner_displacement_q8;
    device_options.force_integer_mv = options.force_integer_mv ? 1 : 0;
    device_options.quantize = options.quantize ? 1 : 0;
    device_options.min_gain_percent = options.min_gain_percent;
    device_options.frame_width = options.frame_width;
    device_options.frame_height = options.frame_height;
    if (gme::estimate_blocks(gme_buffers_, 0,
                             static_cast<int>(device_blocks.size()),
                             device_options, stream_, &error) !=
        gme::GmeStatus::kOk) {
      return false;
    }
    gme::GmeEstimate device_estimate{};
    if (gme::download_estimate(gme_buffers_, 0, &device_estimate, stream_,
                               &error) != gme::GmeStatus::kOk) {
      return false;
    }
    estimate.params.type =
        static_cast<inter::gme::GmType>(device_estimate.params.type);
    for (int i = 0; i < 6; ++i) {
      estimate.params.p[static_cast<std::size_t>(i)] =
          device_estimate.params.p[i];
    }
    estimate.diag.ok = device_estimate.diag.ok != 0;
    estimate.diag.accepted = device_estimate.diag.accepted != 0;
    estimate.diag.codable = device_estimate.diag.codable != 0;
    estimate.diag.shear_valid = device_estimate.diag.shear_valid != 0;
    estimate.diag.degraded = device_estimate.diag.degraded != 0;
    estimate.diag.identity = device_estimate.diag.identity != 0;
    estimate.diag.points_in = device_estimate.diag.points_in;
    estimate.diag.points_used = device_estimate.diag.points_used;
    estimate.diag.outlier_passes_run = device_estimate.diag.outlier_passes_run;
    estimate.diag.identity_sse_q8 = device_estimate.diag.identity_sse_q8;
    estimate.diag.model_sse_q8 = device_estimate.diag.model_sse_q8;
    estimate.diag.gain_db = device_estimate.diag.gain_db;
    estimate.diag.model_rmse_q8 = device_estimate.diag.model_rmse_q8;
    estimate.diag.max_residual_q8 = device_estimate.diag.max_residual_q8;
    estimate.diag.shear_mismatch_q16 = device_estimate.diag.shear_mismatch_q16;
    estimate.diag.candidate_gain_db = device_estimate.diag.candidate_gain_db;
    estimate.diag.candidate_model_sse_q8 =
        device_estimate.diag.candidate_model_sse_q8;
    estimate.diag.reason = gme::gme_reason_name(device_estimate.diag.reason);
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
      block.source_plane = src_planes_[plane] +
                           static_cast<std::size_t>(plane == 0 ? tile_y_
                                                              : tile_y_ / 2) *
                               static_cast<std::size_t>(
                                   plane == 0 ? frame_width_
                                              : frame_chroma_w_) +
                           static_cast<std::size_t>(plane == 0 ? tile_x_
                                                              : tile_x_ / 2);
      block.source_stride = plane == 0 ? frame_width_ : frame_chroma_w_;
      block.source_max_x =
          (plane == 0 ? frame_width_ : frame_chroma_w_) - 1 -
          (plane == 0 ? tile_x_ : tile_x_ / 2);
      block.source_max_y =
          (plane == 0 ? frame_height_ : frame_chroma_h_) - 1 -
          (plane == 0 ? tile_y_ : tile_y_ / 2);
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

  // Full luma intra search (--video-intra-full) through the verified device
  // candidate module (`src/cuda/pipeline/video/intra/`), then the winner's TX
  // plan from the same module.  The orchestrator owns the mode decision inputs
  // (availability is derived here from the tile-local plane geometry, the
  // filter flags come from its IntraJointMode grid), so this is a device mirror
  // of `src/pipeline/video/video_intra_search.h`.
  bool intra_full_candidate(
      int x0, int y0, int w, int h,
      const vintra::Availability& luma_avail,
      const vintra::Availability& chroma_avail, bool above_smooth,
      bool left_smooth, std::array<PlaneTx, 3>& planes, int& y_mode,
      int& y_angle_delta, std::string& error) override {
    if (!persistent_ready_ || !intra_ws_ready_) {
      error = "M2 CUDA: begin_frame() has not allocated the intra workspace";
      return false;
    }
    const int coded_w[3] = {w, w / 2, w / 2};
    const int coded_h[3] = {h, h / 2, h / 2};
    const int ox[3] = {x0, x0 / 2, x0 / 2};
    const int oy[3] = {y0, y0 / 2, y0 / 2};
    const int src_stride[3] = {width_, chroma_w_, chroma_w_};
    const int src_w[3] = {width_, chroma_w_, chroma_w_};
    const int src_h[3] = {height_, chroma_h_, chroma_h_};

    std::array<vi::BlockJob, 3> jobs{};
    for (int plane = 0; plane < 3; ++plane) {
      const vintra::Availability& avail =
          plane == 0 ? luma_avail : chroma_avail;
      vi::BlockJob& job = jobs[static_cast<std::size_t>(plane)];
      job.recon_plane = recon_planes_[plane];
      job.recon_stride = plane == 0 ? luma_w_ : chroma_mi_w_;
      job.source_plane = src_planes_[plane];
      job.source_stride = src_stride[plane];
      job.x = ox[plane];
      job.y = oy[plane];
      job.w = coded_w[plane];
      job.h = coded_h[plane];
      job.plane = plane;
      job.max_x = avail.max_x;
      job.max_y = avail.max_y;
      job.source_max_x = src_w[plane] - 1;
      job.source_max_y = src_h[plane] - 1;
      job.have_left = avail.have_left;
      job.have_above = avail.have_above;
      job.num4_above_right = avail.num4_above_right;
      job.num4_below_left = avail.num4_below_left;
      job.filter_type_above = (plane == 0 && above_smooth) ? 1 : 0;
      job.filter_type_left = (plane == 0 && left_smooth) ? 1 : 0;
      job.enable_edge_filter = 1;
      job.mode_mask = vi::kAllModesMask;
      job.include_angle_deltas = 1;
    }

    // Batch 1: every candidate's predictor + SSE (no TX).
    vi::EvaluateOptions search_options;
    if (!is_ok(intra_ws_.evaluate(jobs.data(), 3, search_options, stream_,
                                  &error))) {
      error = "M2 CUDA: intra search batch failed: " + error;
      return false;
    }
    ++stats_.intra_launches;
    std::array<vi::BestResult, 3> best{};
    if (!is_ok(intra_ws_.download_best(best.data(), 3, stream_))) {
      error = "M2 CUDA: intra search download failed";
      return false;
    }
    ++stats_.syncs;
    y_mode = best[0].mode;
    y_angle_delta = best[0].angle_delta;
    // §7.13.2.7 only directional modes carry an angle delta.
    const bool luma_directional = y_mode >= 1 && y_mode <= 8;

    // Batch 2: the winning luma candidate group + DC chroma, with the module's
    // shared still TX plan.
    std::array<vi::BlockJob, 3> tx_jobs = jobs;
    tx_jobs[0].mode_mask = 1u << static_cast<unsigned>(y_mode);
    tx_jobs[0].include_angle_deltas = luma_directional ? 1 : 0;
    vi::EvaluateOptions tx_options;
    tx_options.run_tx = true;
    if (!is_ok(intra_ws_.evaluate(tx_jobs.data(), 3, tx_options, stream_,
                                  &error))) {
      error = "M2 CUDA: intra TX batch failed: " + error;
      return false;
    }
    ++stats_.intra_launches;
    ++stats_.tx_launches;

    const int candidate_total = intra_ws_.candidate_total();
    std::vector<vi::CandidateResult> results(
        static_cast<std::size_t>(candidate_total));
    std::vector<vi::TxCandidateResult> tx_results(
        static_cast<std::size_t>(candidate_total));
    if (!is_ok(intra_ws_.download_results(results.data(), candidate_total,
                                          stream_)) ||
        !is_ok(intra_ws_.download_tx(tx_results.data(), candidate_total,
                                     stream_))) {
      error = "M2 CUDA: intra TX download failed";
      return false;
    }
    stats_.syncs += 2;

    int chosen[3] = {-1, -1, -1};
    for (int plane = 0; plane < 3; ++plane) {
      const int begin = intra_ws_.candidate_offset(plane);
      const int count = intra_ws_.candidate_count(plane);
      for (int i = 0; i < count; ++i) {
        const vi::CandidateResult& candidate =
            results[static_cast<std::size_t>(begin + i)];
        if (plane != 0 || (candidate.mode == y_mode &&
                           candidate.angle_delta == y_angle_delta)) {
          chosen[plane] = begin + i;
          break;
        }
      }
      if (chosen[plane] < 0) {
        error = "M2 CUDA: intra TX batch lost the winning candidate";
        return false;
      }
    }

    // §7.14.4 inverse + the shared reconstruction kernel on the winning
    // prediction.  `PlaneTx::levels` are the natural-order adjusted-size
    // coefficients the syntax writer scans.
    std::array<av2enc::cuda::TxBlockJob, 3> inverse_jobs{};
    for (int plane = 0; plane < 3; ++plane) {
      const auto params_opt = av2enc::txv::make_tx_params(
          coded_w[plane], coded_h[plane], av2enc::txv::TxType::kDctDct);
      if (!params_opt.has_value()) {
        error = "unsupported transform size";
        return false;
      }
      const av2enc::txv::TxParams& params = *params_opt;
      const int elems = params.w * params.h;
      const std::int32_t levels_offset =
          tx_results[static_cast<std::size_t>(chosen[plane])].levels_offset;
      if (levels_offset < 0) {
        error = "M2 CUDA: intra TX stage rejected the winning candidate";
        return false;
      }
      planes[static_cast<std::size_t>(plane)].levels.resize(
          static_cast<std::size_t>(elems));
      const std::int32_t* src_levels =
          intra_ws_.levels_device() + levels_offset;
      if (!check(cudaMemcpyAsync(
                     planes[static_cast<std::size_t>(plane)].levels.data(),
                     src_levels,
                     static_cast<std::size_t>(elems) * sizeof(std::int32_t),
                     cudaMemcpyDeviceToHost, stream_),
                 "intra levels download", error)) {
        return false;
      }
      stats_.bytes_downloaded +=
          static_cast<std::uint64_t>(elems) * sizeof(std::int32_t);
      av2enc::cuda::TxBlockJob& inverse =
          inverse_jobs[static_cast<std::size_t>(plane)];
      inverse.params = params;
      inverse.in_offset = levels_offset;
      inverse.out_offset = plane * kMaxCodedPixels;
    }
    if (!check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize",
               error)) {
      return false;
    }
    ++stats_.syncs;
    upload(d_tx_jobs_, inverse_jobs.data(), sizeof(inverse_jobs));
    if (!check(av2enc::cuda::txv_inverse_transform_multi(
                   intra_ws_.levels_device(), d_recon_residual_, d_tx_jobs_, 3,
                   stream_),
               "txv_inverse_transform_multi", error)) {
      return false;
    }
    ++stats_.tx_launches;

    std::array<ReconJob, 3> recon_jobs{};
    for (int plane = 0; plane < 3; ++plane) {
      ReconJob& job = recon_jobs[static_cast<std::size_t>(plane)];
      job.pred = intra_ws_.prediction_device() +
                 results[static_cast<std::size_t>(chosen[plane])].pred_offset;
      job.residual = d_recon_residual_ + plane * kMaxCodedPixels;
      job.out = d_recon_out_ + plane * kMaxCodedPixels;
      job.w = coded_w[plane];
      job.h = coded_h[plane];
    }
    upload(d_recon_jobs_, recon_jobs.data(), sizeof(recon_jobs));
    constexpr int kReconThreads = 256;
    video_recon_kernel<<<3, kReconThreads, 0, stream_>>>(d_recon_jobs_, 3);
    if (!check(cudaGetLastError(), "video_recon_kernel", error)) return false;
    ++stats_.recon_launches;

    std::vector<std::uint8_t> host_recon(3 * kMaxCodedPixels);
    if (!check(download(host_recon.data(), d_recon_out_, host_recon.size()),
               "intra recon download", error)) {
      return false;
    }
    for (int plane = 0; plane < 3; ++plane) {
      const int coded_elems = coded_w[plane] * coded_h[plane];
      planes[static_cast<std::size_t>(plane)].recon.assign(
          host_recon.begin() + plane * kMaxCodedPixels,
          host_recon.begin() + plane * kMaxCodedPixels + coded_elems);
    }
    return true;
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
      if (quant_step_ != av2enc::txv::kQuantStep) {
        // CPU compute_tx(): scale the forward quant_factor by 528/qstep; the
        // inverse dequantizes with qlookup(base_q_idx) instead of 528.
        tx_jobs[static_cast<std::size_t>(plane)].params.quant_factor *=
            static_cast<double>(av2enc::txv::kQuantStep) /
            static_cast<double>(quant_step_);
      }
      tx_jobs[static_cast<std::size_t>(plane)].quant_step = quant_step_;
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
      recon_jobs[static_cast<std::size_t>(plane)].pred_stride = coded_w[plane];
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

  bool loop_filter_stage(
      av2enc::video::post::VideoFilterStage stage,
      const av2enc::video::post::VideoFilterPlan& plan,
      const av2enc::video::post::VideoLoopFilterDeviceFrame& frame,
      std::string& error) override {
    using av2enc::video::post::VideoFilterStage;
    // Every CUDA call in this function is checked directly.  Clear the sticky
    // per-thread error so a launcher that inspects cudaGetLastError() (GDF)
    // does not attribute an unrelated earlier probe to its own launch.
    (void)cudaGetLastError();
    if (!persistent_ready_) {
      error = "M2 CUDA: begin_frame() has not allocated the frame buffers";
      return false;
    }
    const std::size_t luma =
        static_cast<std::size_t>(luma_w_) * static_cast<std::size_t>(luma_h_);
    const std::size_t chroma = static_cast<std::size_t>(chroma_mi_w_) *
                               static_cast<std::size_t>(chroma_mi_h_);
    for (int plane = 0; plane < 3; ++plane) {
      const int width = plane == 0 ? luma_w_ : chroma_mi_w_;
      if (frame.in_stride[plane] < width ||
          frame.out_stride[plane] < width) {
        error = "M2 CUDA loop filter: short plane stride";
        return false;
      }
    }

    // Stage input: upload the host planes into the device staging buffer.
    for (int plane = 0; plane < 3; ++plane) {
      const int width = plane == 0 ? luma_w_ : chroma_mi_w_;
      const int rows = plane == 0 ? luma_h_ : chroma_mi_h_;
      if (!upload_2d(filter_in_planes_[plane], width, frame.in[plane],
                     frame.in_stride[plane], width, rows, error)) {
        return false;
      }
    }

    switch (stage) {
      case VideoFilterStage::kCdef: {
        const std::size_t mi_luma = static_cast<std::size_t>(mi_cols_) *
                                    static_cast<std::size_t>(mi_rows_);
        if (plan.cdef.skip.size() != mi_luma ||
            plan.cdef.index.size() !=
                static_cast<std::size_t>(d_cdef_idx_count_)) {
          error = "M2 CUDA loop filter: plan grid size mismatch";
          return false;
        }
        upload(d_filter_skip_, plan.cdef.skip.data(), mi_luma);
        upload(d_cdef_idx_, plan.cdef.index.data(),
               plan.cdef.index.size() * sizeof(std::int8_t));
        av2enc::cuda::CdefDeviceFrame cdef;
        for (int plane = 0; plane < 3; ++plane) {
          cdef.src[plane] = filter_in_planes_[plane];
          cdef.dst[plane] = filter_planes_[plane];
          cdef.stride[plane] = plane == 0 ? luma_w_ : chroma_mi_w_;
        }
        cdef.mi_cols = mi_cols_;
        cdef.mi_rows = mi_rows_;
        cdef.subsampling_x = 1;
        cdef.subsampling_y = 1;
        cdef.skip = d_filter_skip_;
        cdef.lossless = d_filter_lossless_;
        cdef.lossless_chroma = d_filter_lossless_chroma_;
        cdef.cdef_idx = d_cdef_idx_;
        cdef.n64_cols = (mi_cols_ + 15) / 16;
        cdef.n64_rows = (mi_rows_ + 15) / 16;
        const av2enc::video::post::CdefFilterOptions derived =
            av2enc::video::post::cdef_filter_options(plan.cdef);
        av2enc::cdef::CdefOptions options;
        options.damping = derived.damping;
        options.num_strengths = derived.num_strengths;
        options.on_skip_txfm = derived.on_skip_txfm;
        options.disable_loopfilters_across_tiles = false;
        options.bit_depth = 8;
        for (int i = 0; i < derived.num_strengths && i < 8; ++i) {
          options.strengths[i].y_pri = derived.strength[i].y_pri;
          options.strengths[i].y_sec = derived.strength[i].y_sec;
          options.strengths[i].uv_pri = derived.strength[i].uv_pri;
          options.strengths[i].uv_sec = derived.strength[i].uv_sec;
        }
        if (!check(av2enc::cuda::cdef_frame(cdef, options, stream_),
                   "cdef_frame", error)) {
          return false;
        }
        break;
      }
      case VideoFilterStage::kCcso: {
        if (!plan.ccso.enabled) break;
        // Prefill the destination with the stage input: CCSO adds offsets to
        // the post-CDEF samples in place.
        for (int plane = 0; plane < 3; ++plane) {
          if (!check(cudaMemcpyAsync(
                         filter_planes_[plane], filter_in_planes_[plane],
                         (plane == 0 ? luma : chroma), cudaMemcpyDeviceToDevice,
                         stream_),
                     "cudaMemcpyAsync cdef->ccso", error)) {
            return false;
          }
        }
        const std::uint8_t* curr_luma = filter_in_planes_[0];
        if (frame.curr[0] != nullptr) {
          if (!upload_2d(d_filter_curr_luma_, luma_w_, frame.curr[0],
                         frame.curr_stride[0], luma_w_, luma_h_, error)) {
            return false;
          }
          curr_luma = d_filter_curr_luma_;
        }
        const av2enc::video::post::CcsoFilterOptions derived =
            av2enc::video::post::ccso_filter_options(plan.ccso);
        for (int plane = 0; plane < 3; ++plane) {
          upload(d_ccso_offsets_ +
                     static_cast<std::size_t>(plane) *
                         av2enc::restoration::kCcsoOffsetEntries,
                 derived.plane[plane].offsets,
                 av2enc::restoration::kCcsoOffsetEntries *
                     sizeof(std::int16_t));
          const av2enc::video::post::VideoCcsoPlane& src =
              plan.ccso.plane[plane];
          if (src.enabled && !src.blks.empty()) {
            upload(d_ccso_blks_[plane], src.blks.data(), src.blks.size());
          }
        }
        av2enc::restoration::CcsoCoreFrame ccso;
        for (int plane = 0; plane < 3; ++plane) {
          const av2enc::video::post::VideoCcsoPlane& src =
              plan.ccso.plane[plane];
          av2enc::restoration::CcsoCorePlane& dst = ccso.plane[plane];
          dst.enabled = derived.plane[plane].enabled;
          dst.curr_luma = curr_luma;
          dst.curr_luma_stride = luma_w_;
          dst.dst = filter_planes_[plane];
          dst.dst_stride = plane == 0 ? luma_w_ : chroma_mi_w_;
          dst.blks = d_ccso_blks_[plane];
          dst.blk_cols = src.blk_cols;
          dst.blk_rows = src.blk_rows;
          dst.scale_idx = derived.plane[plane].scale_idx;
          dst.quant_idx = derived.plane[plane].quant_idx;
          dst.ext_filter = derived.plane[plane].ext_filter;
          dst.edge_clf = derived.plane[plane].edge_clf;
          dst.max_band_log2 = derived.plane[plane].max_band_log2;
          dst.bo_only = derived.plane[plane].bo_only;
          dst.offsets =
              d_ccso_offsets_ +
              static_cast<std::size_t>(plane) *
                  av2enc::restoration::kCcsoOffsetEntries;
        }
        ccso.mi_cols = mi_cols_;
        ccso.mi_rows = mi_rows_;
        ccso.sub_x = 1;
        ccso.sub_y = 1;
        ccso.bit_depth = 8;
        ccso.ccso_luma_size_log2 = plan.ccso.ccso_luma_size_log2;
        ccso.sb_width = 64;
        ccso.sb_height = 64;
        if (!check(av2enc::cuda::ccso_frame(ccso, stream_), "ccso_frame",
                   error)) {
          return false;
        }
        break;
      }
      case VideoFilterStage::kLr: {
        if (!plan.lr.enabled) break;
        // Pre-CDEF CurrFrame is required by §7.20.1; fail loudly if absent.
        for (int plane = 0; plane < 3; ++plane) {
          if (frame.curr[plane] == nullptr) {
            error = "M2 CUDA loop filter: LR needs the pre-CDEF frame";
            return false;
          }
          const int width = plane == 0 ? luma_w_ : chroma_mi_w_;
          const int rows = plane == 0 ? luma_h_ : chroma_mi_h_;
          if (!upload_2d(d_filter_curr_[plane], width, frame.curr[plane],
                         frame.curr_stride[plane], width, rows, error)) {
            return false;
          }
        }
        const av2enc::video::post::LrFilterOptions derived =
            av2enc::video::post::lr_filter_options(plan.lr);
        for (int plane = 0; plane < 3; ++plane) {
          if (!plan.lr.plane[plane].enabled) continue;
          const std::size_t units =
              static_cast<std::size_t>(plan.lr.plane[plane].unit_rows) *
              static_cast<std::size_t>(plan.lr.plane[plane].unit_cols);
          upload(d_lr_unit_type_[plane], derived.unit_type[plane], units);
          upload(d_lr_unit_coeffs_[plane], derived.unit_coeffs[plane],
                 units * av2enc::restoration::kWienerNsCoeffStride *
                     sizeof(std::int16_t));
        }
        av2enc::restoration::RestorationFrame lr_frame{};
        for (int plane = 0; plane < 3; ++plane) {
          lr_frame.curr[plane].data = d_filter_curr_[plane];
          lr_frame.curr[plane].stride = plane == 0 ? luma_w_ : chroma_mi_w_;
          lr_frame.cdef[plane].data = filter_in_planes_[plane];
          lr_frame.cdef[plane].stride = plane == 0 ? luma_w_ : chroma_mi_w_;
          lr_frame.lr[plane].data = filter_planes_[plane];
          lr_frame.lr[plane].stride = plane == 0 ? luma_w_ : chroma_mi_w_;
        }
        lr_frame.mi_cols = mi_cols_;
        lr_frame.mi_rows = mi_rows_;
        lr_frame.subsampling_x = 1;
        lr_frame.subsampling_y = 1;
        lr_frame.bit_depth = 8;
        lr_frame.sb_width = 64;
        lr_frame.sb_height = 64;
        lr_frame.sb_shift = 4;
        av2enc::restoration::LoopRestorationParams params;
        for (int plane = 0; plane < 3; ++plane) {
          params.frame_type[plane] = static_cast<
              av2enc::restoration::RestorationType>(derived.frame_type[plane]);
          params.unit_size[plane] = derived.unit_size[plane];
          params.unit_type[plane] = d_lr_unit_type_[plane];
          params.unit_rows[plane] = derived.unit_rows[plane];
          params.unit_cols[plane] = derived.unit_cols[plane];
          params.unit_coeffs[plane] = d_lr_unit_coeffs_[plane];
          params.frame_filters_on[plane] = 0;
        }
        if (!check(av2enc::cuda::loop_restoration_frame(lr_frame, params,
                                                        stream_),
                   "loop_restoration_frame", error)) {
          return false;
        }
        break;
      }
      case VideoFilterStage::kGdf: {
        if (!plan.gdf.enabled) break;
        if (!gdf_ready_) {
          error = "M2 CUDA loop filter: GDF workspace is not initialized";
          return false;
        }
        const dim3 threads(16, 16);
        const dim3 blocks(static_cast<unsigned>((luma_w_ + 15) / 16),
                          static_cast<unsigned>((luma_h_ + 15) / 16));
        // CurrFrame (pre-CDEF) -> device u8, then widen to u16.
        if (frame.curr[0] != nullptr) {
          if (!upload_2d(d_filter_curr_[0], luma_w_, frame.curr[0],
                         frame.curr_stride[0], luma_w_, luma_h_, error)) {
            return false;
          }
        } else {
          if (!check(cudaMemcpyAsync(d_filter_curr_[0], filter_in_planes_[0],
                                     luma, cudaMemcpyDeviceToDevice, stream_),
                     "cudaMemcpyAsync gdf curr", error)) {
            return false;
          }
        }
        // CdefFrame (post-CCSO, pre-LR) -> device u8.
        if (frame.cdef[0] != nullptr) {
          if (!upload_2d(d_gdf_cdef_u8_, luma_w_, frame.cdef[0],
                         frame.cdef_stride[0], luma_w_, luma_h_, error)) {
            return false;
          }
        } else {
          if (!check(cudaMemcpyAsync(d_gdf_cdef_u8_, filter_in_planes_[0], luma,
                                     cudaMemcpyDeviceToDevice, stream_),
                     "cudaMemcpyAsync gdf cdef", error)) {
            return false;
          }
        }
        u8_to_u16_kernel<<<blocks, threads, 0, stream_>>>(
            d_filter_curr_[0], luma_w_, d_gdf_curr_, luma_w_, luma_w_,
            luma_h_);
        u8_to_u16_kernel<<<blocks, threads, 0, stream_>>>(
            d_gdf_cdef_u8_, luma_w_, d_gdf_cdef_, luma_w_, luma_w_, luma_h_);
        u8_to_u16_kernel<<<blocks, threads, 0, stream_>>>(
            filter_in_planes_[0], luma_w_, d_gdf_lr_, luma_w_, luma_w_,
            luma_h_);
        if (!check(cudaGetLastError(), "gdf u8->u16", error)) return false;

        // §7.20.1 reference order hints / distances (device arrays).
        upload(d_gdf_order_hints_, plan.gdf.order_hints,
               sizeof(plan.gdf.order_hints));
        upload(d_gdf_frame_distance_, plan.gdf.frame_distance,
               sizeof(plan.gdf.frame_distance));

        const int gdf_cols = (luma_w_ + 63) / 64;
        const int gdf_rows = (luma_h_ + 63) / 64;
        const int units = gdf_cols * gdf_rows;
        std::vector<std::uint8_t> use_gdf(static_cast<std::size_t>(units), 1);
        if (!is_ok(gdf::gdf_upload_use_gdf(gdf_, use_gdf.data(), units,
                                           stream_, &error))) {
          return false;
        }
        gdf::GdfFrameGrid grid{};
        grid.src.curr = d_gdf_curr_;
        grid.src.curr_stride = luma_w_;
        grid.src.cdef = d_gdf_cdef_;
        grid.src.cdef_stride = luma_w_;
        grid.src.start_x = 0;
        grid.src.end_x = luma_w_ - 1;
        grid.src.start_y = 0;
        grid.src.end_y = luma_h_ - 1;
        grid.src.stripe_start_y = 0;
        grid.src.stripe_end_y = luma_h_ - 1;
        grid.lr = d_gdf_lr_;
        grid.lr_stride = luma_w_;
        grid.bit_depth = 8;
        grid.pix_scale = 1 + plan.gdf.scale_idx;
        grid.unit_size = 64;
        grid.cols = gdf_cols;
        grid.rows = gdf_rows;
        grid.width = luma_w_;
        grid.height = luma_h_;
        grid.mi_rows = mi_rows_;
        grid.index.frame_is_intra = plan.gdf.frame_is_intra ? 1 : 0;
        grid.index.base_q_idx = plan.gdf.base_q_idx;
        grid.index.bit_depth = 8;
        grid.index.gdf_pic_qc_idx = plan.gdf.qc_idx;
        grid.index.order_hints = d_gdf_order_hints_;
        grid.index.frame_distance = d_gdf_frame_distance_;
        grid.index.num_total_refs = plan.gdf.num_total_refs;
        grid.index.restricted_oh = plan.gdf.restricted_oh;
        if (!is_ok(gdf::gdf_apply_grid(grid, gdf_, stream_, &error))) {
          return false;
        }
        u16_to_u8_kernel<<<blocks, threads, 0, stream_>>>(
            d_gdf_lr_, luma_w_, filter_planes_[0], luma_w_, luma_w_, luma_h_);
        if (!check(cudaGetLastError(), "gdf u16->u8", error)) return false;
        // Chroma is untouched by GDF.
        for (int plane = 1; plane < 3; ++plane) {
          if (!check(cudaMemcpyAsync(
                         filter_planes_[plane], filter_in_planes_[plane],
                         chroma, cudaMemcpyDeviceToDevice, stream_),
                     "cudaMemcpyAsync gdf chroma", error)) {
            return false;
          }
        }
        break;
      }
    }

    for (int plane = 0; plane < 3; ++plane) {
      const int width = plane == 0 ? luma_w_ : chroma_mi_w_;
      const int rows = plane == 0 ? luma_h_ : chroma_mi_h_;
      if (!download_2d(frame.out[plane], frame.out_stride[plane],
                       filter_planes_[plane], width, width, rows, error)) {
        return false;
      }
    }
    ++stats_.filter_launches;
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
    const int src_stride[3] = {frame_width_, frame_chroma_w_,
                               frame_chroma_w_};
    const int src_w[3] = {frame_width_, frame_chroma_w_, frame_chroma_w_};
    const int src_h[3] = {frame_height_, frame_chroma_h_, frame_chroma_h_};

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

  bool tx_search_luma(bool inter, const InterJob& job, int x0, int y0, int w,
                      int h, int qindex, double lambda, bool partition,
                      TxSearchPlan& out, std::string& error) override {
    if (!persistent_ready_ || !tx_ws_.valid()) {
      error = "M2 CUDA: TX-search workspace is not allocated";
      return false;
    }
    out = TxSearchPlan{};
    out.recon.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h),
                     0);

    // 1. Predictor + coded residual into d_pred_[0] / d_residual_[0].
    if (inter) {
      if (job.ref_index < 0 || job.ref_index >= ref_count_) {
        error = "M2 CUDA TX search: ref_index out of range";
        return false;
      }
      ScratchScope scratch(arena_);
      ArenaAllocation alloc;
      McBlockJob* d_mc_job = nullptr;
      if (!arena_alloc_array(arena_, 1, &d_mc_job, "tx search MC job", alloc,
                             error)) {
        return record_arena_failure(alloc);
      }
      McBlockJob mc_job;
      mc_job.dst = d_pred_;
      mc_job.dst_stride = w;
      mc_job.x = x0;
      mc_job.y = y0;
      mc_job.w = w;
      mc_job.h = h;
      mc_job.plane = 0;
      mc_job.mv = inter::mc::Mv{job.mv_y, job.mv_x};
      mc_job.filter = inter::mc::InterpFilter::kEighttap;
      upload(d_mc_job, &mc_job, sizeof(mc_job));
      if (!check(av2enc::cuda::mc_predict_batch(ref_frame(job.ref_index),
                                                d_mc_job, 1, stream_),
                 "mc_predict_batch tx search", error)) {
        return false;
      }
      ++stats_.mc_launches;
      const int pixels = w * h;
      video_residual_kernel<<<div_up(pixels, 256), 256, 0, stream_>>>(
          src_planes_[0], width_, width_, height_, x0, y0, d_pred_, w,
          d_residual_, w, h);
      if (!check(cudaGetLastError(), "video_residual_kernel tx search", error)) {
        return false;
      }
      ++stats_.residual_launches;
    } else {
      ScratchScope scratch(arena_);
      ArenaAllocation alloc;
      intra_modes::IntraJob* d_job = nullptr;
      if (!arena_alloc_array(arena_, 1, &d_job, "tx search intra job", alloc,
                             error)) {
        return record_arena_failure(alloc);
      }
      intra_modes::IntraJob intra_job;
      intra_modes::IntraBlock& block = intra_job.block;
      block.recon_plane = recon_planes_[0];
      block.recon_stride = luma_w_;
      block.x0 = x0;
      block.y0 = y0;
      block.width = w;
      block.height = h;
      block.max_x = luma_w_ - 1;
      block.max_y = luma_h_ - 1;
      block.have_left = x0 > 0 ? 1 : 0;
      block.have_above = y0 > 0 ? 1 : 0;
      block.num4_above_right = 0;
      block.num4_below_left = 0;
      block.plane = 0;
      block.mode = intra_modes::kDcPred;
      block.angle_delta = 0;
      block.filter_type_above = 0;
      block.filter_type_left = 0;
      block.enable_edge_filter = 0;
      block.source_plane = src_planes_[0];
      block.source_stride = width_;
      block.source_max_x = width_ - 1;
      block.source_max_y = height_ - 1;
      intra_job.pred = d_pred_;
      intra_job.residual = d_residual_;
      upload(d_job, &intra_job, sizeof(intra_job));
      if (!check(intra_modes::intra_predict_batch(d_job, 1, stream_),
                 "intra_predict_batch tx search", error)) {
        return false;
      }
      ++stats_.intra_launches;
    }

    // 2. Device TX search (one job, one launch inside the module).
    videotx::VideoTxJobRequest request;
    request.plane = 0;
    request.mode = inter ? videotx::VideoTxMode::kInter
                         : videotx::VideoTxMode::kIntra;
    request.coded_w = w;
    request.coded_h = h;
    request.qindex = qindex;
    request.delta_dc = 0;
    request.delta_ac = 0;
    request.residual_offset = 0;
    request.search.search_none = 1;
    request.search.search_partition = partition ? 1 : 0;
    request.search.search_tx_type = 1;
    request.search.include_1d_types = 0;
    request.search.partition_mask = 0x3f;
    request.search.reduced_tx_set = 0;
    request.search.reduced_tx_part_set = 0;
    request.search.enable_inter_ddt = 0;
    request.search.dct_only = 0;
    request.search.intra_mapped_mode = 0;
    request.search.lambda = lambda;
    request.search.coefficient_bits = 1.0;
    request.search.skip_bits = 1.2;
    request.search.use_type_bits = 1;
    request.search.max_tx_size = 64;

    videotx::VideoTxPlanData plan;
    if (!videotx::video_tx_build_plan(&request, 1, &plan, &error)) {
      return false;
    }
    videotx::VideoTxBuffers buffers;
    buffers.residual = d_residual_;
    const Status status =
        videotx::video_tx_select_batch(plan, buffers, tx_ws_, stream_);
    if (status != Status::kOk) {
      error = std::string("video_tx_select_batch: ") + status_name(status);
      return false;
    }
    ++stats_.tx_launches;

    // 3. Read the winning plan back.
    videotx::VideoTxJobResult job_result;
    if (!check(download(&job_result, tx_ws_.d_job_results,
                        sizeof(job_result)),
               "TX search job result", error)) {
      return false;
    }
    if (job_result.status != 0 || job_result.leaf_count <= 0) {
      error = "video TX search produced no codable plan";
      return false;
    }
    const videotx::VideoTxPlanDesc& plan_desc = plan.plans[job_result.plan_index];
    std::vector<videotx::VideoTxLeafResult> leaf_results(
        static_cast<std::size_t>(job_result.leaf_count));
    if (!check(download(leaf_results.data(),
                        tx_ws_.d_leaf_results + job_result.leaf_result_begin,
                        leaf_results.size() * sizeof(leaf_results[0])),
               "TX search leaf results", error)) {
      return false;
    }
    out.partition = job_result.partition;
    out.rate_bits = job_result.rate_bits;
    out.parts.reserve(leaf_results.size());

    for (int k = 0; k < job_result.leaf_count; ++k) {
      const videotx::VideoTxLeafDesc& leaf = plan.leaves[plan_desc.leaf_begin + k];
      const videotx::VideoTxLeafResult& result = leaf_results[k];
      const auto params_opt = av2enc::txv::make_tx_params(
          static_cast<av2enc::txv::TxSize>(result.tx_size),
          static_cast<av2enc::txv::TxType>(result.tx_type));
      if (!params_opt.has_value()) {
        error = "TX search returned an unsupported transform";
        return false;
      }
      const av2enc::txv::TxParams& params = *params_opt;
      const int area = params.w * params.h;
      std::vector<std::int32_t> scan_levels(static_cast<std::size_t>(area));
      const std::int32_t* device_levels =
          tx_ws_.d_levels +
          static_cast<std::size_t>(leaf.level_offset);
      if (!check(download(scan_levels.data(), device_levels,
                          scan_levels.size() * sizeof(std::int32_t)),
                 "TX search levels", error)) {
        return false;
      }
      TxSearchPart part;
      part.x = leaf.x;
      part.y = leaf.y;
      part.coded_w = leaf.coded_w;
      part.coded_h = leaf.coded_h;
      part.tx_size = result.tx_size;
      part.tx_type = result.tx_type;
      part.mi_row = leaf.y / 4;
      part.mi_col = leaf.x / 4;
      part.eob = result.eob;
      part.levels = scan_levels;

      // Reconstruct this leaf on the device: undo the §5.20.7.30 scan order on
      // the host (index permutation only), then run the device inverse
      // transform and reconstruction kernels.  There is no CPU coefficient
      // fallback: the dequantise/inverse arithmetic is the shared device path.
      std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
      if (!av2enc::video::txpartition::get_scan(
              result.tx_size, av2enc::video::txpartition::TxScanClass::k2D,
              scan)) {
        error = "TX search scan table failed";
        return false;
      }
      std::vector<std::int32_t> natural(static_cast<std::size_t>(area), 0);
      for (int c = 0; c < area; ++c) {
        natural[scan[c]] = scan_levels[c];
      }
      if (!check(cudaMemcpyAsync(d_levels_, natural.data(),
                                 natural.size() * sizeof(std::int32_t),
                                 cudaMemcpyHostToDevice, stream_),
                 "TX search levels upload", error)) {
        return false;
      }
      av2enc::cuda::TxBlockJob tx_job;
      tx_job.params = params;
      tx_job.in_offset = 0;
      tx_job.out_offset = 0;
      upload(d_tx_jobs_, &tx_job, sizeof(tx_job));
      if (!check(av2enc::cuda::txv_inverse_transform_multi(
                     d_levels_, d_recon_residual_, d_tx_jobs_, 1, stream_),
                 "txv_inverse_transform_multi TX search", error)) {
        return false;
      }
      ReconJob recon_job;
      recon_job.pred = d_pred_ + static_cast<std::size_t>(leaf.y) * w +
                       static_cast<std::size_t>(leaf.x);
      recon_job.residual = d_recon_residual_;
      recon_job.out = d_recon_out_;
      recon_job.w = leaf.coded_w;
      recon_job.h = leaf.coded_h;
      recon_job.pred_stride = w;
      upload(d_recon_jobs_, &recon_job, sizeof(recon_job));
      video_recon_kernel<<<1, 256, 0, stream_>>>(d_recon_jobs_, 1);
      if (!check(cudaGetLastError(), "video_recon_kernel TX search", error)) {
        return false;
      }
      std::vector<std::uint8_t> leaf_recon(
          static_cast<std::size_t>(leaf.coded_w) *
          static_cast<std::size_t>(leaf.coded_h));
      if (!check(download(leaf_recon.data(), d_recon_out_,
                          leaf_recon.size() * sizeof(std::uint8_t)),
                 "TX search recon", error)) {
        return false;
      }
      for (int y = 0; y < leaf.coded_h; ++y) {
        std::memcpy(
            out.recon.data() +
                static_cast<std::size_t>(leaf.y + y) *
                    static_cast<std::size_t>(w) +
                static_cast<std::size_t>(leaf.x),
            leaf_recon.data() +
                static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(leaf.coded_w),
            static_cast<std::size_t>(leaf.coded_w));
      }
      out.parts.push_back(std::move(part));
    }
    ++stats_.tx_launches;
    ++stats_.recon_launches;
    return true;
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
        src_planes_[0], frame_width_, frame_width_, frame_height_, job.x,
        job.y, d_pred_, job.w, d_residual_, job.w, job.h);
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
    if (quant_step_ != av2enc::txv::kQuantStep) {
      tx_job.params.quant_factor *=
          static_cast<double>(av2enc::txv::kQuantStep) /
          static_cast<double>(quant_step_);
    }
    tx_job.quant_step = quant_step_;
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
    recon_job.pred_stride = job.w;
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

  bool deblock_frame(const av2enc::video::post::VideoDeblockGrid& grid,
                     std::vector<std::uint8_t>& y, std::vector<std::uint8_t>& u,
                     std::vector<std::uint8_t>& v,
                     std::string& error) override {
    if (!persistent_ready_ || recon_planes_[0] == nullptr) {
      error = "M2 CUDA: deblock_frame() has not allocated the frame buffers";
      return false;
    }
    if (grid.mi_cols != mi_cols_ || grid.mi_rows != mi_rows_) {
      error = "M2 CUDA: deblock MI grid does not match the frame";
      return false;
    }
    const std::size_t cells =
        static_cast<std::size_t>(mi_cols_) * static_cast<std::size_t>(mi_rows_);
    if (cells == 0 || grid.mi[0].size() != cells ||
        grid.mi[1].size() != cells || grid.mi[2].size() != cells) {
      error = "M2 CUDA: deblock MI grid is incomplete";
      return false;
    }
    ScratchScope scratch(arena_);
    ArenaAllocation alloc;
    deblock::DeblockMi* d_mi[3] = {nullptr, nullptr, nullptr};
    for (int plane = 0; plane < 3; ++plane) {
      if (!arena_alloc_array(arena_, cells, &d_mi[plane], "deblock MI grid",
                             alloc, error)) {
        return record_arena_failure(alloc);
      }
      upload(d_mi[plane], grid.mi[plane].data(),
             cells * sizeof(deblock::DeblockMi));
    }
    std::uint16_t* d_tile_cols = nullptr;
    std::uint16_t* d_tile_rows = nullptr;
    if (!grid.tile_col_starts.empty()) {
      if (!arena_alloc_array(arena_, grid.tile_col_starts.size(), &d_tile_cols,
                             "deblock tile columns", alloc, error)) {
        return record_arena_failure(alloc);
      }
      upload(d_tile_cols, grid.tile_col_starts.data(),
             grid.tile_col_starts.size() * sizeof(std::uint16_t));
    }
    if (!grid.tile_row_starts.empty()) {
      if (!arena_alloc_array(arena_, grid.tile_row_starts.size(), &d_tile_rows,
                             "deblock tile rows", alloc, error)) {
        return record_arena_failure(alloc);
      }
      upload(d_tile_rows, grid.tile_row_starts.data(),
             grid.tile_row_starts.size() * sizeof(std::uint16_t));
    }

    av2enc::cuda::DeblockDeviceFrame device_frame;
    for (int plane = 0; plane < 3; ++plane) {
      device_frame.plane[plane] = recon_planes_[plane];
      device_frame.stride[plane] = plane == 0 ? luma_w_ : chroma_mi_w_;
      device_frame.mi[plane] = d_mi[plane];
    }
    device_frame.mi_cols = mi_cols_;
    device_frame.mi_rows = mi_rows_;
    device_frame.subsampling_x = 1;
    device_frame.subsampling_y = 1;
    if (d_tile_cols != nullptr) {
      device_frame.tile_col_starts = d_tile_cols;
      device_frame.num_tile_cols =
          static_cast<int>(grid.tile_col_starts.size());
    }
    if (d_tile_rows != nullptr) {
      device_frame.tile_row_starts = d_tile_rows;
      device_frame.num_tile_rows =
          static_cast<int>(grid.tile_row_starts.size());
    }
    if (!check(::av2enc::cuda::deblock_frame(device_frame, grid.options,
                                             stream_),
               "CUDA deblock_frame", error)) {
      return false;
    }
    stats_.deblock_launches += 2;  // vertical edges, then horizontal edges

    y.resize(static_cast<std::size_t>(luma_w_) *
             static_cast<std::size_t>(luma_h_));
    u.resize(static_cast<std::size_t>(chroma_mi_w_) *
             static_cast<std::size_t>(chroma_mi_h_));
    v.resize(u.size());
    return download_2d(y.data(), luma_w_, recon_planes_[0], luma_w_, luma_w_,
                       luma_h_, error) &&
           download_2d(u.data(), chroma_mi_w_, recon_planes_[1], chroma_mi_w_,
                       chroma_mi_w_, chroma_mi_h_, error) &&
           download_2d(v.data(), chroma_mi_w_, recon_planes_[2], chroma_mi_w_,
                       chroma_mi_w_, chroma_mi_h_, error);
  }

  bool debug_set_recon(int plane, const std::vector<std::uint8_t>& data,
                       std::string& error) override {
    if (plane < 0 || plane > 2) {
      error = "debug_set_recon: bad plane";
      return false;
    }
    const int stride = plane == 0 ? frame_width_ : frame_chroma_w_;
    const int rows = plane == 0 ? frame_height_ : frame_chroma_h_;
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
    frame.stride[0] = frame_width_;
    frame.stride[1] = frame_chroma_w_;
    frame.stride[2] = frame_chroma_w_;
    frame.mi_cols = frame_mi_cols_;
    frame.mi_rows = frame_mi_rows_;
    return frame;
  }

  VideoDeviceConfig device_config() const {
    return VideoDeviceConfig{static_cast<std::uint32_t>(frame_width_),
                          static_cast<std::uint32_t>(frame_height_),
                          static_cast<std::uint32_t>(frame_mi_cols_),
                          static_cast<std::uint32_t>(frame_mi_rows_),
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
    d_warp_pred_ = nullptr;
    d_warp_sse_jobs_ = nullptr;
    d_warp_store_jobs_ = nullptr;
    d_warp_sse_out_ = nullptr;
    warp_workspace_.reset();
    gme_buffers_ = gme::GmeBuffers{};
    for (int plane = 0; plane < 3; ++plane) {
      filter_planes_[plane] = nullptr;
      filter_in_planes_[plane] = nullptr;
      d_ccso_blks_[plane] = nullptr;
      d_filter_curr_[plane] = nullptr;
      d_lr_unit_type_[plane] = nullptr;
      d_lr_unit_coeffs_[plane] = nullptr;
    }
    d_filter_curr_luma_ = nullptr;
    d_filter_skip_ = nullptr;
    d_filter_lossless_ = nullptr;
    d_filter_lossless_chroma_ = nullptr;
    d_cdef_idx_ = nullptr;
    d_cdef_idx_count_ = 0;
    d_ccso_offsets_ = nullptr;
    d_gdf_cdef_u8_ = nullptr;
    d_gdf_curr_ = nullptr;
    d_gdf_cdef_ = nullptr;
    d_gdf_lr_ = nullptr;
    d_gdf_order_hints_ = nullptr;
    d_gdf_frame_distance_ = nullptr;
    gdf_ = gdf::GdfDeviceBuffers{};
    gdf_ready_ = false;
    tx_ws_.reset();    persistent_ready_ = false;
    intra_ws_ready_ = false;
  }

  // Persistent buffers are allocated once per frame geometry.  Reallocating
  // rewinds to the persistent mark, so it is only legal at a frame boundary:
  // every pointer handed out earlier is invalid afterwards.
  bool ensure_persistent(std::string& error) {
    required_device_bytes_ = required_device_bytes_impl(device_config());
    if (persistent_ready_ && persistent_width_ == frame_width_ &&
        persistent_height_ == frame_height_ &&
        persistent_mi_cols_ == frame_mi_cols_ &&
        persistent_mi_rows_ == frame_mi_rows_) {
      return true;
    }
    if (persistent_ready_) {
      arena_.rewind(persistent_mark_);
      clear_persistent();
    }
    persistent_mark_ = arena_.mark();
    ArenaAllocation alloc;
    const std::size_t luma = static_cast<std::size_t>(frame_width_) *
                             static_cast<std::size_t>(frame_height_);
    const std::size_t chroma = static_cast<std::size_t>(frame_chroma_w_) *
                               static_cast<std::size_t>(frame_chroma_h_);
    const std::size_t mi_luma =
        static_cast<std::size_t>(frame_mi_cols_ * 4) *
        static_cast<std::size_t>(frame_mi_rows_ * 4);
    const std::size_t mi_chroma =
        static_cast<std::size_t>(frame_mi_cols_ * 2) *
        static_cast<std::size_t>(frame_mi_rows_ * 2);
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
    for (int plane = 0; plane < 3 && ok; ++plane) {
      const std::size_t mi_count = plane == 0 ? mi_luma : mi_chroma;
      ok = arena_alloc_array(arena_, mi_count, &filter_planes_[plane],
                             "filter plane", alloc, error) &&
           arena_alloc_array(arena_, mi_count, &filter_in_planes_[plane],
                             "filter input plane", alloc, error);
    }
    if (ok) {
      d_cdef_idx_count_ =
          ((mi_cols_ + 15) / 16) * ((mi_rows_ + 15) / 16);
      ok = arena_alloc_array(arena_, mi_luma, &d_filter_curr_luma_,
                             "filter curr luma", alloc, error) &&
           arena_alloc_array(arena_, mi_luma, &d_filter_skip_,
                             "filter skip grid", alloc, error) &&
           arena_alloc_array(arena_, mi_luma, &d_filter_lossless_,
                             "filter lossless grid", alloc, error) &&
           arena_alloc_array(arena_, mi_luma, &d_filter_lossless_chroma_,
                             "filter chroma lossless grid", alloc, error) &&
           arena_alloc_array(arena_, static_cast<std::size_t>(d_cdef_idx_count_),
                             &d_cdef_idx_, "cdef index grid", alloc, error) &&
           arena_alloc_array(
               arena_, 3ull * av2enc::restoration::kCcsoOffsetEntries,
               &d_ccso_offsets_, "ccso offsets", alloc, error);
      const std::size_t lr_units_max =
          static_cast<std::size_t>(mi_cols_ / 8 + 2) *
          static_cast<std::size_t>(mi_rows_ / 8 + 2);
      for (int plane = 0; plane < 3 && ok; ++plane) {
        const std::size_t mi_count = plane == 0 ? mi_luma : mi_chroma;
        ok = arena_alloc_array(arena_, mi_luma, &d_ccso_blks_[plane],
                               "ccso unit grid", alloc, error) &&
             arena_alloc_array(arena_, mi_count, &d_filter_curr_[plane],
                               "filter curr plane", alloc, error) &&
             arena_alloc_array(arena_, lr_units_max, &d_lr_unit_type_[plane],
                               "lr unit type", alloc, error) &&
             arena_alloc_array(
                 arena_,
                 lr_units_max * av2enc::restoration::kWienerNsCoeffStride,
                 &d_lr_unit_coeffs_[plane], "lr unit coeffs", alloc, error);
      }
      ok = ok &&
           arena_alloc_array(arena_, mi_luma, &d_gdf_cdef_u8_, "gdf cdef8",
                             alloc, error) &&
           arena_alloc_array(arena_, mi_luma, &d_gdf_curr_, "gdf curr16",
                             alloc, error) &&
           arena_alloc_array(arena_, mi_luma, &d_gdf_cdef_, "gdf cdef16",
                             alloc, error) &&
           arena_alloc_array(arena_, mi_luma, &d_gdf_lr_, "gdf lr16", alloc,
                             error) &&
           arena_alloc_array(arena_, 2, &d_gdf_order_hints_,
                             "gdf order hints", alloc, error) &&
           arena_alloc_array(arena_, 2, &d_gdf_frame_distance_,
                             "gdf frame distance", alloc, error);
      if (ok) {
        const int gdf_cols = static_cast<int>((mi_cols_ * 4 + 63) / 64);
        const int gdf_rows = static_cast<int>((mi_rows_ * 4 + 63) / 64);
        gdf::GdfConfig gdf_config;
        gdf_config.max_units = gdf::gdf_grid_slots(gdf_cols, gdf_rows);
        std::string gdf_detail;
        if (gdf::gdf_buffers_init(arena_, gdf_config, &gdf_, nullptr,
                                  &gdf_detail) != Status::kOk) {
          ok = false;
          error = "M2 CUDA: GDF workspace init failed: " + gdf_detail;
        } else {
          gdf_ready_ = true;
        }
      }
      if (ok) {
        // No lossless segments in the M2 baseline; zero once per geometry.
        ok = check(cudaMemsetAsync(d_filter_lossless_, 0, mi_luma, stream_),
                   "cudaMemsetAsync lossless", error) &&
             check(cudaMemsetAsync(d_filter_lossless_chroma_, 0, mi_luma,
                                   stream_),
                   "cudaMemsetAsync chroma lossless", error);
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
                           "recon output", alloc, error) &&
         arena_alloc_array(arena_,
                           static_cast<std::size_t>(kWarpBatchJobs) *
                               kMaxBlockPixels,
                           &d_warp_pred_, "warp prediction pool", alloc,
                           error) &&
         arena_alloc_array(arena_, kWarpBatchJobs, &d_warp_sse_jobs_,
                           "warp SSE jobs", alloc, error) &&
         arena_alloc_array(arena_, kWarpBatchJobs, &d_warp_store_jobs_,
                           "warp store jobs", alloc, error) &&
         arena_alloc_array(arena_, kWarpBatchJobs, &d_warp_sse_out_,
                           "warp SSE output", alloc, error);
    if (!ok) {
      arena_.rewind(persistent_mark_);
      clear_persistent();
      last_arena_required_ = required_device_bytes_;
      last_arena_status_ = ArenaStatus::kArenaTooSmall;
      return false;
    }

    // §5.18.9 estimator + §7.13.3.19/.20 warp workspace.  Both allocate once
    // from the same arena and are released by the persistent rewind above.
    warp_workspace_.reset();
    warpstack::WarpStackConfig ws_config;
    ws_config.max_predict_jobs = kWarpBatchJobs;
    if (warp_workspace_.create(arena_, ws_config, &error) != Status::kOk) {
      arena_.rewind(persistent_mark_);
      clear_persistent();
      last_arena_required_ = required_device_bytes_;
      last_arena_status_ = ArenaStatus::kArenaTooSmall;
      return false;
    }
    gme_buffers_ = gme::GmeBuffers{};
    std::uint64_t gme_required = 0;
    if (gme::allocate_checked(arena_, gme_config(), &gme_buffers_,
                              &gme_required, &error) != gme::GmeStatus::kOk) {
      arena_.rewind(persistent_mark_);
      clear_persistent();
      last_arena_required_ = required_device_bytes_;
      last_arena_status_ = ArenaStatus::kArenaTooSmall;
      return false;
    }

    // Full-intra candidate workspace (--video-intra-full).  Persistent for the
    // frame geometry, from the same arena, above the other frame buffers.
    std::uint64_t intra_required = 0;
    std::string intra_error;
    const Status intra_status = intra_ws_.init(
        arena_, intra_full_config(), &intra_error, &intra_required);
    if (!is_ok(intra_status)) {
      arena_.rewind(persistent_mark_);
      clear_persistent();
      last_arena_required_ = intra_required;
      last_arena_status_ =
          intra_status == Status::kArenaExhausted ? ArenaStatus::kArenaTooSmall
                                                  : ArenaStatus::kCudaError;
      error = "M2 CUDA: intra workspace init failed: " + intra_error;
      return false;
    }
    intra_ws_ready_ = true;
    // Opt-in luma TX-search workspace: one persistent module allocation.  It is
    // part of the same frame-boundary mark, so a geometry change rewinds both.
    std::uint64_t tx_bytes = 0;
    std::string tx_error;
    const av2enc::cuda::Status tx_status =
        tx_ws_.init(arena_, kTxSearchWorkspaceConfig, &tx_bytes, &tx_error);
    if (tx_status != av2enc::cuda::Status::kOk) {
      arena_.rewind(persistent_mark_);
      clear_persistent();
      last_arena_required_ = required_device_bytes_ + tx_bytes;
      last_arena_status_ = ArenaStatus::kArenaTooSmall;
      error = std::string("M2 CUDA: TX-search workspace init failed: ") +
              (tx_error.empty() ? av2enc::cuda::status_name(tx_status)
                                : tx_error);
      return false;
    }    persistent_ready_ = true;
    persistent_width_ = frame_width_;
    persistent_height_ = frame_height_;
    persistent_mi_cols_ = frame_mi_cols_;
    persistent_mi_rows_ = frame_mi_rows_;
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

  int width_ = 0;   // coded tile width (MI-aligned)
  int height_ = 0;  // coded tile height
  int chroma_w_ = 0;
  int chroma_h_ = 0;
  int frame_width_ = 0;
  int frame_height_ = 0;
  int frame_chroma_w_ = 0;
  int frame_chroma_h_ = 0;
  int frame_mi_cols_ = 0;
  int frame_mi_rows_ = 0;
  int tile_x_ = 0;
  int tile_y_ = 0;
  bool tiled_ = false;
  int quant_step_ = av2enc::txv::kQuantStep;  // §7.14.2 qlookup(base_q_idx)
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
  // Loop-filter stage: staging input, filtered output, CCSO metadata.
  std::uint8_t* filter_in_planes_[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* filter_planes_[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* d_filter_curr_luma_ = nullptr;
  std::uint8_t* d_filter_skip_ = nullptr;
  std::uint8_t* d_filter_lossless_ = nullptr;
  std::uint8_t* d_filter_lossless_chroma_ = nullptr;
  std::int8_t* d_cdef_idx_ = nullptr;
  int d_cdef_idx_count_ = 0;
  std::int16_t* d_ccso_offsets_ = nullptr;
  std::uint8_t* d_ccso_blks_[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* d_filter_curr_[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* d_lr_unit_type_[3] = {nullptr, nullptr, nullptr};
  std::int16_t* d_lr_unit_coeffs_[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* d_gdf_cdef_u8_ = nullptr;
  std::uint16_t* d_gdf_curr_ = nullptr;
  std::uint16_t* d_gdf_cdef_ = nullptr;
  std::uint16_t* d_gdf_lr_ = nullptr;
  int* d_gdf_order_hints_ = nullptr;
  int* d_gdf_frame_distance_ = nullptr;
  gdf::GdfDeviceBuffers gdf_{};
  bool gdf_ready_ = false;

  std::uint8_t* d_pred_ = nullptr;
  std::int16_t* d_residual_ = nullptr;
  std::int16_t* d_residual_adj_ = nullptr;
  double* d_coeff_ = nullptr;
  std::int32_t* d_levels_ = nullptr;
  std::int16_t* d_recon_residual_ = nullptr;
  std::uint8_t* d_recon_out_ = nullptr;
  av2enc::cuda::TxBlockJob* d_tx_jobs_ = nullptr;
  videotx::VideoTxWorkspace tx_ws_;
  ReconJob* d_recon_jobs_ = nullptr;
  vi::Workspace intra_ws_;
  bool intra_ws_ready_ = false;

  std::int16_t* d_warp_pred_ = nullptr;
  WarpSseJob* d_warp_sse_jobs_ = nullptr;
  WarpStoreJob* d_warp_store_jobs_ = nullptr;
  std::int64_t* d_warp_sse_out_ = nullptr;
  warpstack::Workspace warp_workspace_;
  gme::GmeBuffers gme_buffers_{};

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
