// CUDA execution for the wavefront plan.
//
// The per-frame sequence is:
//
//   upload source planes (pinned staging, 3 copies)
//   for each wavefront level:
//       one batched dc_predict_residual for every independent plane block
//       for each (level, transform class) group:
//           [per 64x64 luma block: downsample the adjusted 32x32 residual]
//           forward_transform   (forward DCT + quantise)
//           inverse_transform   (dequantise + inverse DCT)
//           recon               (batched clip / 2x2 duplication)
//   scatter canonical coefficient layout
//   download recon planes + coefficients + predictions
//
// prepare() allocates one persistent arena, builds the device job descriptors,
// runs the sequence once to warm the CUDA modules, and captures it into a CUDA
// graph.  run() then uploads, replays the graph (one host launch) and downloads.
//
// Bit-exactness notes:
//   * prediction/residual and transform/quantise/dequantise/inverse are the
//     shared components, invoked with the same arguments as the M0 pipeline;
//   * the reconstruction launcher below applies the exact expressions of
//     intra::add_residual_clip_to_plane and transform::reconstruct_64x64 (the
//     M0 pipeline has its own 20-line adapter for the latter); it exists only
//     because the components expose no batched reconstruction entry point.  The
//     harness checks all three output buffers bit-exactly against the per-block
//     path, so this glue cannot drift silently.

#include "cuda/pipeline/dispatch/scheduler/scheduler.h"

#include "cuda/arena.h"
#include "cuda/context.h"
#include "cuda/intra.h"
#include "cuda/transform.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::cuda::sched {
namespace {

constexpr int kThreads = 256;
constexpr int kAdjBlockPels = 64 * 64;  // full 64x64 luma residual/prediction
constexpr int kMiB = 1024 * 1024;

bool check_cuda(cudaError_t status, const char* what, std::string& error) {
  if (status == cudaSuccess) {
    return true;
  }
  error = std::string(what) + " failed: " + cudaGetErrorString(status);
  (void)cudaGetLastError();
  return false;
}

bool check_status(Status status, const char* what, std::string& error) {
  if (is_ok(status)) {
    return true;
  }
  error = std::string(what) + " failed: " + status_name(status);
  return false;
}

// One batched reconstruction step.  `upscale == 2` means the block is a 64x64
// luma block whose 32x32 inverse transform output is duplicated 2x2, matching
// transform::reconstruct_64x64.  Otherwise the block is written sample by
// sample, matching intra::add_residual_clip_to_plane.
struct ReconJob {
  const std::uint8_t* pred = nullptr;
  const std::int16_t* residual = nullptr;
  std::uint8_t* plane = nullptr;
  int stride = 0;
  int x0 = 0;
  int y0 = 0;
  int w = 0;
  int h = 0;
  int max_x = 0;
  int max_y = 0;
  int upscale = 1;
  int pad = 0;
  std::int32_t* pred_out = nullptr;
};

// Canonical coefficient layout scatter: class-order slot -> per-block offset.
struct CoeffCopy {
  const std::int32_t* src = nullptr;
  std::int32_t* dst = nullptr;
  int count = 0;
  int pad = 0;
};

__device__ __forceinline__ int clip1_dev(int value) {
  return value < 0 ? 0 : (value > 255 ? 255 : value);
}

__global__ void recon_batch_kernel(const ReconJob* __restrict__ jobs) {
  const ReconJob job = jobs[blockIdx.x];
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  const int pred = static_cast<int>(job.pred[0]);
  if (tid == 0 && job.pred_out != nullptr) {
    *job.pred_out = pred;
  }
  const std::size_t stride = static_cast<std::size_t>(job.stride);
  if (job.upscale == 2) {
    for (int idx = tid; idx < kAdjBlockPels / 4; idx += nthr) {
      const int i = idx & 31;
      const int j = idx >> 5;
      const int value =
          clip1_dev(pred + static_cast<int>(job.residual[idx]));
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const int x = job.x0 + 2 * i + dx;
          const int y = job.y0 + 2 * j + dy;
          if (x <= job.max_x && y <= job.max_y) {
            job.plane[static_cast<std::size_t>(y) * stride +
                      static_cast<std::size_t>(x)] =
                static_cast<std::uint8_t>(value);
          }
        }
      }
    }
    return;
  }
  const int n = job.w * job.h;
  for (int idx = tid; idx < n; idx += nthr) {
    const int x = job.x0 + idx % job.w;
    const int y = job.y0 + idx / job.w;
    if (x > job.max_x || y > job.max_y) {
      continue;
    }
    const int value = clip1_dev(pred + static_cast<int>(job.residual[idx]));
    job.plane[static_cast<std::size_t>(y) * stride +
              static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(value);
  }
}

__global__ void scatter_coeffs_kernel(const CoeffCopy* __restrict__ copies) {
  const CoeffCopy copy = copies[blockIdx.x];
  for (int i = static_cast<int>(threadIdx.x); i < copy.count;
       i += static_cast<int>(blockDim.x)) {
    copy.dst[i] = copy.src[i];
  }
}

template <typename T>
bool arena_alloc(Arena& arena, std::size_t count, T** out, const char* what,
                 std::string& error) {
  *out = nullptr;
  if (count == 0) {
    return true;
  }
  const std::size_t bytes = count * sizeof(T);
  void* pointer = nullptr;
  const Status status = arena.allocate_checked(bytes, &pointer);
  if (!is_ok(status)) {
    error = std::string("arena allocation for ") + what + " (" +
            std::to_string(bytes) + " bytes) failed: " + status_name(status);
    return false;
  }
  *out = static_cast<T*>(pointer);
  return true;
}

struct ClassBuffers {
  std::int16_t* residual = nullptr;   // batch input (int16)
  std::uint8_t* pred = nullptr;       // per-sample prediction block
  double* coeff = nullptr;            // forward DCT scratch
  std::int32_t* levels = nullptr;     // quantised coefficients (batched)
  std::int32_t* dequant = nullptr;    // inverse DCT scratch
  std::int16_t* inv = nullptr;        // inverse transform residual
};

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct ScheduledFrame::Impl {
  Context context;
  Arena arena;
  StagingBuffer upload_staging;
  StagingBuffer download_staging;
  Plan plan;
  bool ready = false;
  int device = -1;
  std::uint64_t prepared_signature = 0;

  std::size_t src_y_bytes = 0;
  std::size_t src_c_bytes = 0;
  std::size_t rec_y_bytes = 0;
  std::size_t rec_c_bytes = 0;
  std::size_t coeff_bytes = 0;
  std::size_t pred_bytes = 0;
  std::size_t total_coeffs = 0;
  std::size_t workspace_bytes = 0;

  std::uint8_t* d_y = nullptr;
  std::uint8_t* d_u = nullptr;
  std::uint8_t* d_v = nullptr;
  std::uint8_t* d_rec_y = nullptr;
  std::uint8_t* d_rec_u = nullptr;
  std::uint8_t* d_rec_v = nullptr;
  std::int32_t* d_coeffs = nullptr;
  std::int32_t* d_preds = nullptr;

  std::vector<ClassBuffers> class_buffers;
  std::vector<TxParams> class_params;
  std::int16_t* d_adj_residual = nullptr;
  std::uint8_t* d_adj_pred = nullptr;
  int adj_jobs = 0;

  DcBlockJob* d_jobs = nullptr;
  int job_count = 0;
  ReconJob* d_recon_jobs = nullptr;
  CoeffCopy* d_coeff_copies = nullptr;

  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_uploaded = nullptr;
  cudaEvent_t ev_kernel = nullptr;
  cudaEvent_t ev_download = nullptr;

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  int graph_nodes = 0;
  bool graph_available = false;
  std::string graph_error;

  bool prepare(const FrameJob& job, int device_index, std::string& error);
  bool enqueue(cudaStream_t stream, std::string& error);
  bool capture(std::string& error);
  bool upload(const FrameJob& job, cudaStream_t stream, std::string& error);
  bool download(const FrameJob& job, cudaStream_t stream, std::string& error);
  void teardown();

  std::uint8_t* plane_ptr(int plane) {
    return plane == 0 ? d_rec_y : (plane == 1 ? d_rec_u : d_rec_v);
  }
  int plane_stride(int plane) const {
    return plane == 0 ? plan.geometry().padded_width
                      : plan.geometry().chroma_padded_width;
  }
  const std::uint8_t* source_ptr(int plane) const {
    return plane == 0 ? d_y : (plane == 1 ? d_u : d_v);
  }
  int source_stride(int plane) const {
    return plane == 0 ? plan.geometry().width : plan.geometry().chroma_width;
  }
  int source_max_x(int plane) const {
    return (plane == 0 ? plan.geometry().width
                       : plan.geometry().chroma_width) - 1;
  }
  int source_max_y(int plane) const {
    return (plane == 0 ? plan.geometry().height
                       : plan.geometry().chroma_height) - 1;
  }
};

void ScheduledFrame::Impl::teardown() {
  if (graph_exec != nullptr) {
    cudaGraphExecDestroy(graph_exec);
    graph_exec = nullptr;
  }
  if (graph != nullptr) {
    cudaGraphDestroy(graph);
    graph = nullptr;
  }
  graph_available = false;
  graph_nodes = 0;
  graph_error.clear();
  for (cudaEvent_t* event :
       {&ev_start, &ev_uploaded, &ev_kernel, &ev_download}) {
    if (*event != nullptr) {
      cudaEventDestroy(*event);
      *event = nullptr;
    }
  }
  upload_staging.release();
  download_staging.release();
  arena.release();
  context.shutdown();
  class_buffers.clear();
  class_params.clear();
  plan.clear();
  d_y = nullptr;
  d_u = nullptr;
  d_v = nullptr;
  d_rec_y = nullptr;
  d_rec_u = nullptr;
  d_rec_v = nullptr;
  d_coeffs = nullptr;
  d_preds = nullptr;
  d_adj_residual = nullptr;
  d_adj_pred = nullptr;
  d_jobs = nullptr;
  d_recon_jobs = nullptr;
  d_coeff_copies = nullptr;
  adj_jobs = 0;
  job_count = 0;
  total_coeffs = 0;
  workspace_bytes = 0;
  src_y_bytes = 0;
  src_c_bytes = 0;
  rec_y_bytes = 0;
  rec_c_bytes = 0;
  coeff_bytes = 0;
  pred_bytes = 0;
  prepared_signature = 0;
  ready = false;
  device = -1;
}

bool ScheduledFrame::Impl::prepare(const FrameJob& job, int device_index,
                                   std::string& error) {
  teardown();
  error.clear();
  if (!plan.build(job, error)) {
    return false;
  }
  if (plan.jobs().empty()) {
    error = "plan has no real coding blocks; nothing to schedule";
    plan.clear();
    return false;
  }
  const FrameGeometry& g = plan.geometry();

  std::string detail;
  if (!check_status(context.init(device_index, &detail), "Context::init", error)) {
    if (!detail.empty()) {
      error += ": " + detail;
    }
    return false;
  }
  device = device_index;
  cudaStream_t stream = context.stream();

  // -------------------------------------------------------------------------
  // Sizes
  // -------------------------------------------------------------------------
  src_y_bytes = static_cast<std::size_t>(g.width) * g.height;
  src_c_bytes = static_cast<std::size_t>(g.chroma_width) * g.chroma_height;
  rec_y_bytes = static_cast<std::size_t>(g.padded_width) * g.padded_height;
  rec_c_bytes = static_cast<std::size_t>(g.chroma_padded_width) *
                g.chroma_padded_height;

  std::int64_t coeff_end = 0;
  for (const PlaneJob& j : plan.jobs()) {
    const std::int64_t end =
        static_cast<std::int64_t>(j.coeff_offset) + j.coeff_count;
    if (end > coeff_end) {
      coeff_end = end;
    }
  }
  total_coeffs = static_cast<std::size_t>(
      std::max<std::int64_t>(job.total_coeffs, coeff_end));
  if (total_coeffs == 0) {
    error = "plan produced no coefficients";
    teardown();
    return false;
  }
  coeff_bytes = total_coeffs * sizeof(std::int32_t);
  pred_bytes = static_cast<std::size_t>(job.block_count) * 3u *
               sizeof(std::int32_t);

  // -------------------------------------------------------------------------
  // Arena sized to this plan (no cudaMalloc after prepare).
  // -------------------------------------------------------------------------
  std::size_t required = src_y_bytes + 2 * src_c_bytes + rec_y_bytes +
                         2 * rec_c_bytes + coeff_bytes + pred_bytes;
  required += plan.jobs().size() * (sizeof(DcBlockJob) + sizeof(ReconJob) +
                                    sizeof(CoeffCopy));
  for (const TransformClass& cls : plan.classes()) {
    // residual(2) + pred(1) + coeff(8) + levels(4) + dequant(4) + inv(2)
    required += static_cast<std::size_t>(cls.pels) *
                static_cast<std::size_t>(cls.job_count) * 21u;
  }
  adj_jobs = plan.counters().luma_64_jobs;
  required += static_cast<std::size_t>(adj_jobs) * kAdjBlockPels *
              (sizeof(std::int16_t) + sizeof(std::uint8_t));
  required += 4 * kMiB;
  std::size_t capacity = ((required + 16 * kMiB - 1) / (16 * kMiB)) * 16 * kMiB;
  if (capacity < static_cast<std::size_t>(64 * kMiB)) {
    capacity = static_cast<std::size_t>(64 * kMiB);
  }
  if (!check_status(arena.init(capacity, Arena::kAlignment, &detail),
                    "Arena::init", error)) {
    if (!detail.empty()) {
      error += ": " + detail;
    }
    teardown();
    return false;
  }

  // -------------------------------------------------------------------------
  // Persistent buffers
  // -------------------------------------------------------------------------
  bool ok = arena_alloc(arena, src_y_bytes, &d_y, "source Y", error) &&
            arena_alloc(arena, src_c_bytes, &d_u, "source U", error) &&
            arena_alloc(arena, src_c_bytes, &d_v, "source V", error) &&
            arena_alloc(arena, rec_y_bytes, &d_rec_y, "recon Y", error) &&
            arena_alloc(arena, rec_c_bytes, &d_rec_u, "recon U", error) &&
            arena_alloc(arena, rec_c_bytes, &d_rec_v, "recon V", error) &&
            arena_alloc(arena, total_coeffs, &d_coeffs, "coefficients", error) &&
            arena_alloc(arena, static_cast<std::size_t>(job.block_count) * 3u,
                        &d_preds, "predictions", error);
  if (!ok) {
    teardown();
    return false;
  }

  class_buffers.resize(plan.classes().size());
  class_params.resize(plan.classes().size());
  for (int c = 0; c < static_cast<int>(plan.classes().size()); ++c) {
    const TransformClass& cls = plan.classes()[static_cast<std::size_t>(c)];
    const int coded_w = cls.upscale == 2 ? 2 * cls.tx_w : cls.tx_w;
    const int coded_h = cls.upscale == 2 ? 2 * cls.tx_h : cls.tx_h;
    const std::optional<TxParams> params =
        make_tx_params(coded_w, coded_h, cls.upscale == 2);
    if (!params.has_value() || params->w != cls.tx_w || params->h != cls.tx_h) {
      error = "transform component does not support the " +
              std::to_string(cls.tx_w) + "x" + std::to_string(cls.tx_h) +
              " class (coded " + std::to_string(coded_w) + "x" +
              std::to_string(coded_h) + ")";
      teardown();
      return false;
    }
    class_params[static_cast<std::size_t>(c)] = *params;
    const std::size_t count = static_cast<std::size_t>(cls.pels) *
                              static_cast<std::size_t>(cls.job_count);
    ClassBuffers& cb = class_buffers[static_cast<std::size_t>(c)];
    ok = arena_alloc(arena, count, &cb.residual, "class residual", error) &&
         arena_alloc(arena, count, &cb.pred, "class prediction", error) &&
         arena_alloc(arena, count, &cb.coeff, "class DCT scratch", error) &&
         arena_alloc(arena, count, &cb.levels, "class coefficients", error) &&
         arena_alloc(arena, count, &cb.dequant, "class dequant scratch", error) &&
         arena_alloc(arena, count, &cb.inv, "class inverse residual", error);
    if (!ok) {
      teardown();
      return false;
    }
  }

  if (adj_jobs > 0) {
    ok = arena_alloc(arena,
                     static_cast<std::size_t>(adj_jobs) * kAdjBlockPels,
                     &d_adj_residual, "64x64 residual scratch", error) &&
         arena_alloc(arena,
                     static_cast<std::size_t>(adj_jobs) * kAdjBlockPels,
                     &d_adj_pred, "64x64 prediction scratch", error);
    if (!ok) {
      teardown();
      return false;
    }
  }

  // -------------------------------------------------------------------------
  // Device descriptors (uploaded once; block positions never change)
  // -------------------------------------------------------------------------
  const std::vector<PlaneJob>& jobs = plan.jobs();
  job_count = static_cast<int>(jobs.size());
  std::vector<DcBlockJob> host_jobs(jobs.size());
  std::vector<ReconJob> host_recon(jobs.size());
  std::vector<CoeffCopy> host_copies(jobs.size());
  for (int i = 0; i < job_count; ++i) {
    const PlaneJob& j = jobs[static_cast<std::size_t>(i)];
    const TransformClass& cls = plan.classes()[static_cast<std::size_t>(j.cls)];
    ClassBuffers& cb = class_buffers[static_cast<std::size_t>(j.cls)];
    const std::size_t pels = static_cast<std::size_t>(cls.pels);
    const std::size_t slot = static_cast<std::size_t>(j.slot) * pels;

    DcBlockJob& dj = host_jobs[static_cast<std::size_t>(i)];
    dj.block.recon_plane = plane_ptr(j.plane);
    dj.block.recon_stride = plane_stride(j.plane);
    dj.block.x0 = j.x;
    dj.block.y0 = j.y;
    dj.block.block_w = j.w;
    dj.block.block_h = j.h;
    dj.block.plane_max_x = j.max_x;
    dj.block.plane_max_y = j.max_y;
    dj.block.have_left = (j.x > 0) ? 1 : 0;
    dj.block.have_above = (j.y > 0) ? 1 : 0;
    dj.block.source_plane = source_ptr(j.plane);
    dj.block.source_stride = source_stride(j.plane);
    dj.block.source_max_x = source_max_x(j.plane);
    dj.block.source_max_y = source_max_y(j.plane);
    if (j.upscale == 2) {
      const std::size_t scratch =
          static_cast<std::size_t>(j.scratch_slot) * kAdjBlockPels;
      dj.pred_block = d_adj_pred + scratch;
      dj.residual_block = d_adj_residual + scratch;
    } else {
      dj.pred_block = cb.pred + slot;
      dj.residual_block = cb.residual + slot;
    }

    ReconJob& rj = host_recon[static_cast<std::size_t>(i)];
    rj.pred = dj.pred_block;
    rj.residual = cb.inv + slot;
    rj.plane = plane_ptr(j.plane);
    rj.stride = plane_stride(j.plane);
    rj.x0 = j.x;
    rj.y0 = j.y;
    rj.w = j.w;
    rj.h = j.h;
    rj.max_x = j.max_x;
    rj.max_y = j.max_y;
    rj.upscale = j.upscale;
    rj.pred_out = d_preds + j.pred_index;

    CoeffCopy& cc = host_copies[static_cast<std::size_t>(i)];
    cc.src = cb.levels + slot;
    cc.dst = d_coeffs + j.coeff_offset;
    cc.count = cls.pels;
  }

  ok = arena_alloc(arena, jobs.size(), &d_jobs, "block job descriptors", error) &&
       arena_alloc(arena, jobs.size(), &d_recon_jobs, "reconstruction jobs",
                   error) &&
       arena_alloc(arena, jobs.size(), &d_coeff_copies, "coefficient copies",
                   error);
  if (!ok) {
    teardown();
    return false;
  }
  cudaError_t copy_status =
      cudaMemcpy(d_jobs, host_jobs.data(),
                 jobs.size() * sizeof(DcBlockJob), cudaMemcpyHostToDevice);
  if (copy_status == cudaSuccess) {
    copy_status = cudaMemcpy(d_recon_jobs, host_recon.data(),
                             jobs.size() * sizeof(ReconJob),
                             cudaMemcpyHostToDevice);
  }
  if (copy_status == cudaSuccess) {
    copy_status = cudaMemcpy(d_coeff_copies, host_copies.data(),
                             jobs.size() * sizeof(CoeffCopy),
                             cudaMemcpyHostToDevice);
  }
  if (!check_cuda(copy_status, "descriptor upload", error)) {
    teardown();
    return false;
  }

  // The padded recon margins and the unused prediction slots stay at their M0
  // initial values for every frame; the visible samples are rewritten each run.
  // Ordered on the scheduler stream (the stream is non-blocking, so it does not
  // implicitly interlock with the legacy default stream).
  if (!check_cuda(cudaMemsetAsync(d_rec_y, 128, rec_y_bytes, stream),
                  "memset recon Y", error) ||
      !check_cuda(cudaMemsetAsync(d_rec_u, 128, rec_c_bytes, stream),
                  "memset recon U", error) ||
      !check_cuda(cudaMemsetAsync(d_rec_v, 128, rec_c_bytes, stream),
                  "memset recon V", error) ||
      !check_cuda(cudaMemsetAsync(d_preds, 0, pred_bytes, stream),
                  "memset predictions", error) ||
      !check_cuda(cudaMemsetAsync(d_coeffs, 0, coeff_bytes, stream),
                  "memset coefficients", error)) {
    teardown();
    return false;
  }

  // Pinned staging for the per-frame copies.
  const std::size_t upload_bytes = src_y_bytes + 2 * src_c_bytes;
  const std::size_t download_bytes =
      rec_y_bytes + 2 * rec_c_bytes + coeff_bytes + pred_bytes;
  if (!check_status(upload_staging.init(upload_bytes, &detail),
                    "upload StagingBuffer::init", error) ||
      !check_status(download_staging.init(download_bytes, &detail),
                    "download StagingBuffer::init", error)) {
    if (!detail.empty()) {
      error += ": " + detail;
    }
    teardown();
    return false;
  }

  for (cudaEvent_t* event :
       {&ev_start, &ev_uploaded, &ev_kernel, &ev_download}) {
    if (!check_cuda(cudaEventCreate(event), "cudaEventCreate", error)) {
      teardown();
      return false;
    }
  }

  // Warm-up: loads the CUDA modules before capture, and catches component
  // argument errors early.  The source planes are uninitialised here; the
  // arithmetic does not care and the first real run overwrites every output.
  if (!enqueue(stream, error)) {
    teardown();
    return false;
  }
  if (!check_status(context.synchronize(), "Context::synchronize", error)) {
    teardown();
    return false;
  }

  // Capture is best-effort: if the driver refuses, run() falls back to the
  // batched (still much cheaper than per-block) launch sequence.
  if (!capture(error)) {
    teardown();
    return false;
  }

  workspace_bytes = static_cast<std::size_t>(arena.used());
  prepared_signature = plan.blocks_signature();
  ready = true;
  return true;
}

bool ScheduledFrame::Impl::capture(std::string& error) {
  cudaStream_t stream = context.stream();
  const cudaError_t begin =
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (begin != cudaSuccess) {
    graph_error = std::string("cudaStreamBeginCapture: ") +
                  cudaGetErrorString(begin);
    (void)cudaGetLastError();
    return true;  // not fatal
  }
  std::string enqueue_error;
  const bool enqueued = enqueue(stream, enqueue_error);
  cudaGraph_t captured = nullptr;
  const cudaError_t end = cudaStreamEndCapture(stream, &captured);
  if (!enqueued) {
    if (captured != nullptr) {
      cudaGraphDestroy(captured);
    }
    error = "CUDA graph capture failed: " + enqueue_error;
    return false;
  }
  if (end != cudaSuccess) {
    graph_error = std::string("cudaStreamEndCapture: ") +
                  cudaGetErrorString(end);
    (void)cudaGetLastError();
    if (captured != nullptr) {
      cudaGraphDestroy(captured);
    }
    return true;  // not fatal
  }
  cudaGraphExec_t exec = nullptr;
  const cudaError_t instantiate = cudaGraphInstantiate(&exec, captured, 0);
  if (instantiate != cudaSuccess) {
    graph_error = std::string("cudaGraphInstantiate: ") +
                  cudaGetErrorString(instantiate);
    (void)cudaGetLastError();
    cudaGraphDestroy(captured);
    return true;  // not fatal
  }
  std::size_t nodes = 0;
  if (cudaGraphGetNodes(captured, nullptr, &nodes) != cudaSuccess) {
    nodes = 0;
    (void)cudaGetLastError();
  }
  graph = captured;
  graph_exec = exec;
  graph_nodes = static_cast<int>(nodes);
  graph_available = true;

  // Upload the graph and run it once: the first cudaGraphLaunch pays a one-off
  // device upload for a large graph, which must not land in the first measured
  // frame (and this proves the instantiated graph replays).
  if (!check_cuda(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch(warm)",
                  error)) {
    return false;
  }
  return check_status(context.synchronize(), "Context::synchronize", error);
}

bool ScheduledFrame::Impl::enqueue(cudaStream_t stream, std::string& error) {
  const std::vector<PlaneJob>& jobs = plan.jobs();
  const std::vector<WaveGroup>& groups = plan.groups();
  int group_cursor = 0;
  for (int level = 0; level < plan.levels(); ++level) {
    const int begin = plan.level_begin(level);
    const int end = plan.level_end(level);
    if (begin >= end) {
      continue;
    }
    if (!check_cuda(dc_predict_residual_batch(d_jobs + begin, end - begin,
                                              stream),
                    "intra::dc_predict_residual_batch", error)) {
      return false;
    }
    while (group_cursor < static_cast<int>(groups.size()) &&
           groups[static_cast<std::size_t>(group_cursor)].level == level) {
      const WaveGroup& group = groups[static_cast<std::size_t>(group_cursor++)];
      const TransformClass& cls =
          plan.classes()[static_cast<std::size_t>(group.cls)];
      const TxParams& params = class_params[static_cast<std::size_t>(group.cls)];
      ClassBuffers& cb = class_buffers[static_cast<std::size_t>(group.cls)];
      const std::size_t pels = static_cast<std::size_t>(cls.pels);
      const std::size_t offset = static_cast<std::size_t>(group.slot_begin) * pels;
      const int count = group.job_end - group.job_begin;
      if (cls.upscale == 2) {
        for (int i = group.job_begin; i < group.job_end; ++i) {
          const PlaneJob& j = jobs[static_cast<std::size_t>(i)];
          const std::size_t scratch =
              static_cast<std::size_t>(j.scratch_slot) * kAdjBlockPels;
          const std::size_t dst = static_cast<std::size_t>(j.slot) * pels;
          if (!check_cuda(
                  downsample_residual_64x64(d_adj_residual + scratch,
                                            cb.residual + dst, stream),
                  "transform::downsample_residual_64x64", error)) {
            return false;
          }
        }
      }
      if (!check_cuda(forward_transform(cb.residual + offset, cb.coeff + offset,
                                        cb.levels + offset, params, count,
                                        stream),
                      "transform::forward_transform", error) ||
          !check_cuda(inverse_transform(cb.levels + offset,
                                        cb.dequant + offset, cb.inv + offset,
                                        params, count, stream),
                      "transform::inverse_transform", error)) {
        return false;
      }
      recon_batch_kernel<<<count, kThreads, 0, stream>>>(d_recon_jobs +
                                                         group.job_begin);
      if (!check_cuda(cudaGetLastError(), "recon_batch_kernel", error)) {
        return false;
      }
    }
  }
  if (group_cursor != static_cast<int>(groups.size())) {
    error = "internal: wave groups left unexecuted";
    return false;
  }
  scatter_coeffs_kernel<<<job_count, kThreads, 0, stream>>>(d_coeff_copies);
  return check_cuda(cudaGetLastError(), "scatter_coeffs_kernel", error);
}

bool ScheduledFrame::Impl::upload(const FrameJob& job, cudaStream_t stream,
                                  std::string& error) {
  if (src_y_bytes == 0 || src_c_bytes == 0) {
    return true;
  }
  unsigned char* staging = static_cast<unsigned char*>(upload_staging.data());
  std::memcpy(staging, job.y, src_y_bytes);
  std::memcpy(staging + src_y_bytes, job.u, src_c_bytes);
  std::memcpy(staging + src_y_bytes + src_c_bytes, job.v, src_c_bytes);
  if (!check_status(upload_async(stream, d_y, staging, src_y_bytes),
                    "upload source Y", error) ||
      !check_status(upload_async(stream, d_u, staging + src_y_bytes, src_c_bytes),
                    "upload source U", error) ||
      !check_status(upload_async(stream, d_v,
                                 staging + src_y_bytes + src_c_bytes,
                                 src_c_bytes),
                    "upload source V", error)) {
    return false;
  }
  return true;
}

bool ScheduledFrame::Impl::download(const FrameJob& job, cudaStream_t stream,
                                    std::string& error) {
  unsigned char* staging = static_cast<unsigned char*>(download_staging.data());
  std::size_t offset = 0;
  const auto enqueue = [&](const void* device_source, std::size_t bytes) {
    unsigned char* destination = staging + offset;
    offset += bytes;
    return download_async(stream, destination, device_source, bytes);
  };
  if (!check_status(enqueue(d_rec_y, rec_y_bytes), "download recon Y", error) ||
      !check_status(enqueue(d_rec_u, rec_c_bytes), "download recon U", error) ||
      !check_status(enqueue(d_rec_v, rec_c_bytes), "download recon V", error) ||
      !check_status(enqueue(d_coeffs, coeff_bytes), "download coefficients",
                    error) ||
      !check_status(enqueue(d_preds, pred_bytes), "download predictions",
                    error)) {
    return false;
  }
  (void)job;  // host copies happen after synchronisation in run()
  return true;
}

// ---------------------------------------------------------------------------
// Public wrapper
// ---------------------------------------------------------------------------

ScheduledFrame::ScheduledFrame() = default;

ScheduledFrame::~ScheduledFrame() {
  if (impl_) {
    impl_->teardown();
  }
}

bool ScheduledFrame::prepare(const FrameJob& job, int device, std::string& error) {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return impl_->prepare(job, device, error);
}

void ScheduledFrame::release() {
  if (impl_) {
    impl_->teardown();
  }
}

bool ScheduledFrame::prepared() const { return impl_ && impl_->ready; }

bool ScheduledFrame::graph_available() const {
  return impl_ && impl_->graph_available;
}

const std::string& ScheduledFrame::graph_error() const {
  static const std::string empty;
  return impl_ ? impl_->graph_error : empty;
}

const Plan& ScheduledFrame::plan() const {
  static const Plan empty;
  return impl_ ? impl_->plan : empty;
}

bool ScheduledFrame::run(const FrameJob& job, const RunOptions& options,
                         RunStats* stats, std::string& error) {
  error.clear();
  if (!impl_ || !impl_->ready) {
    error = "ScheduledFrame::run called before prepare";
    return false;
  }
  Impl& s = *impl_;
  const FrameGeometry& g = s.plan.geometry();
  if (job.geometry.width != g.width || job.geometry.height != g.height ||
      job.geometry.padded_width != g.padded_width ||
      job.geometry.padded_height != g.padded_height) {
    error = "frame geometry differs from the prepared plan";
    return false;
  }
  if (job.block_count != s.plan.counters().coding_blocks) {
    error = "block count differs from the prepared plan";
    return false;
  }
  if (coding_blocks_signature(job.blocks, job.block_count) !=
      s.prepared_signature) {
    error = "block list differs from the prepared plan";
    return false;
  }
  if (job.total_coeffs > 0 &&
      static_cast<std::size_t>(job.total_coeffs) != s.total_coeffs) {
    error = "total_coeffs differs from the prepared plan";
    return false;
  }
  if (job.y == nullptr || job.u == nullptr || job.v == nullptr) {
    error = "missing host source planes";
    return false;
  }
  if (job.recon_y == nullptr || job.recon_u == nullptr ||
      job.recon_v == nullptr || job.coeffs == nullptr || job.preds == nullptr) {
    error = "missing host output buffers";
    return false;
  }
  if (s.upload_staging.capacity() == 0 || s.download_staging.capacity() == 0) {
    error = "staging buffers are not initialised";
    return false;
  }
  cudaStream_t stream = s.context.stream();
  if (!check_cuda(cudaEventRecord(s.ev_start, stream), "cudaEventRecord",
                  error)) {
    return false;
  }
  if (!s.upload(job, stream, error)) {
    return false;
  }
  if (!check_cuda(cudaEventRecord(s.ev_uploaded, stream), "cudaEventRecord",
                  error)) {
    return false;
  }

  bool used_graph = false;
  int host_launches = 0;
  int gpu_nodes = 0;
  const auto submit_start = std::chrono::steady_clock::now();
  if (options.use_cuda_graph && s.graph_available) {
    if (!check_cuda(cudaGraphLaunch(s.graph_exec, stream), "cudaGraphLaunch",
                    error)) {
      return false;
    }
    used_graph = true;
    host_launches = 1;
    gpu_nodes = s.graph_nodes;
  } else {
    if (!s.enqueue(stream, error)) {
      return false;
    }
    host_launches = static_cast<int>(s.plan.counters().scheduled_kernel_launches);
    gpu_nodes = host_launches;
  }
  const auto submit_end = std::chrono::steady_clock::now();
  if (!check_cuda(cudaEventRecord(s.ev_kernel, stream), "cudaEventRecord",
                  error)) {
    return false;
  }
  if (!s.download(job, stream, error)) {
    return false;
  }
  if (!check_cuda(cudaEventRecord(s.ev_download, stream), "cudaEventRecord",
                  error)) {
    return false;
  }

  // One frame-boundary synchronisation, exactly as the M0 pipeline does.
  if (!check_status(s.context.synchronize(), "Context::synchronize", error)) {
    return false;
  }

  float upload_ms = 0.0f;
  float kernel_ms = 0.0f;
  float download_ms = 0.0f;
  cudaEventElapsedTime(&upload_ms, s.ev_start, s.ev_uploaded);
  cudaEventElapsedTime(&kernel_ms, s.ev_uploaded, s.ev_kernel);
  cudaEventElapsedTime(&download_ms, s.ev_kernel, s.ev_download);

  // Host copies out of the pinned staging.
  unsigned char* staging = static_cast<unsigned char*>(s.download_staging.data());
  std::size_t offset = 0;
  std::memcpy(job.recon_y, staging + offset, s.rec_y_bytes);
  offset += s.rec_y_bytes;
  std::memcpy(job.recon_u, staging + offset, s.rec_c_bytes);
  offset += s.rec_c_bytes;
  std::memcpy(job.recon_v, staging + offset, s.rec_c_bytes);
  offset += s.rec_c_bytes;
  std::memcpy(job.coeffs, staging + offset, s.coeff_bytes);
  offset += s.coeff_bytes;
  std::memcpy(job.preds, staging + offset, s.pred_bytes);

  if (stats != nullptr) {
    stats->used_graph = used_graph;
    stats->host_launches = host_launches;
    stats->gpu_nodes = gpu_nodes;
    stats->workspace_bytes = s.workspace_bytes;
    stats->upload_ms = static_cast<double>(upload_ms);
    stats->kernel_ms = static_cast<double>(kernel_ms);
    stats->download_ms = static_cast<double>(download_ms);
    stats->submit_ms = std::chrono::duration<double, std::milli>(
                           submit_end - submit_start)
                           .count();
  }
  return true;
}

}  // namespace av2enc::cuda::sched
