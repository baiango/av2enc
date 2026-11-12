// CUDA kernels for the AV2 global motion estimator.
//
// The arithmetic is entirely in `gme_device.cuh`, which the host-emulation
// test compiles too.  Extraction and the image quality proxy are
// data-parallel.  The fit itself runs one job per block with one active thread
// so the accumulation order, outlier selection and degradation path are
// exactly the shared core's order; that is what makes the result bit-exact
// with `gme::estimate_*` for the same input.
#include "cuda/gme/gme.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace av2enc::cuda::gme {
namespace {

constexpr int kGmeExtractThreads = 256;
constexpr int kGmeImageThreads = 256;

void set_error(std::string* error, const char* context,
               const std::string& detail) {
  if (error != nullptr) {
    *error = std::string(context) + ": " + detail;
  }
}

bool buffers_ready(const GmeBuffers& buffers) noexcept {
  return buffers.d_points != nullptr && buffers.d_active != nullptr &&
         buffers.d_residuals != nullptr && buffers.d_keep != nullptr &&
         buffers.d_blocks != nullptr && buffers.d_warps != nullptr &&
         buffers.d_results != nullptr && buffers.d_control != nullptr &&
         buffers.max_points > 0 && buffers.max_blocks > 0 &&
         buffers.max_warps > 0 && buffers.max_jobs > 0;
}

bool valid_job(const GmeBuffers& buffers, int job) noexcept {
  return job >= 0 && job < buffers.max_jobs;
}

GmeStatus from_cuda(cudaError_t result) noexcept {
  return status_from_cuda(result) == Status::kOk ? GmeStatus::kOk
                                                 : GmeStatus::kCudaError;
}

GmePoint* points_for(const GmeBuffers& buffers, int job) noexcept {
  return buffers.d_points + static_cast<std::size_t>(job) *
                                static_cast<std::size_t>(buffers.max_points);
}

GmePoint* active_for(const GmeBuffers& buffers, int job) noexcept {
  return buffers.d_active + static_cast<std::size_t>(job) *
                                static_cast<std::size_t>(buffers.max_points);
}

GmeI64* residuals_for(const GmeBuffers& buffers, int job) noexcept {
  return buffers.d_residuals + static_cast<std::size_t>(job) *
                                   static_cast<std::size_t>(buffers.max_points);
}

GmeU8* keep_for(const GmeBuffers& buffers, int job) noexcept {
  return buffers.d_keep + static_cast<std::size_t>(job) *
                              static_cast<std::size_t>(buffers.max_points);
}

GmeBlockMv* blocks_for(const GmeBuffers& buffers, int job) noexcept {
  return buffers.d_blocks + static_cast<std::size_t>(job) *
                                static_cast<std::size_t>(buffers.max_blocks);
}

GmeWarpSample* warps_for(const GmeBuffers& buffers, int job) noexcept {
  return buffers.d_warps + static_cast<std::size_t>(job) *
                               static_cast<std::size_t>(buffers.max_warps);
}

// Inclusive Hillis-Steele scan of one shared-memory array.  `s_scan` has
// blockDim.x entries; all threads participate.
__device__ __forceinline__ void scan_shared(int* s_scan, int tid,
                                            int nthreads) {
  for (int offset = 1; offset < nthreads; offset <<= 1) {
    const int add = (tid >= offset) ? s_scan[tid - offset] : 0;
    __syncthreads();
    s_scan[tid] += add;
    __syncthreads();
  }
}

__global__ void gme_extract_blocks_kernel(const GmeBlockMv* blocks,
                                          int count, GmePoint* out,
                                          int capacity,
                                          GmeJobControl* control) {
  __shared__ int s_scan[kGmeExtractThreads];
  __shared__ int s_running;
  __shared__ int s_overflow;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  if (tid == 0) {
    s_running = 0;
    s_overflow = 0;
  }
  __syncthreads();

  for (int base = 0; base < count; base += nthreads) {
    const int index = base + tid;
    int valid = 0;
    GmePoint point;
    if (index < count) {
      const int result = gme_extract_point_from_block(blocks[index], &point);
      if (result < 0) atomicOr(&s_overflow, 1);
      valid = result > 0 ? 1 : 0;
    }
    s_scan[tid] = valid;
    __syncthreads();
    scan_shared(s_scan, tid, nthreads);
    const int my_offset = s_running + (tid > 0 ? s_scan[tid - 1] : 0);
    if (index < count && valid) {
      if (my_offset < capacity) out[my_offset] = point;
    }
    const int chunk_total = s_scan[nthreads - 1];
    __syncthreads();
    if (tid == nthreads - 1) s_running += chunk_total;
    __syncthreads();
  }
  __syncthreads();
  if (tid == 0) {
    control->point_count = s_running;
    control->overflow_mask = s_overflow;
    control->too_many_mask = s_running > capacity ? 1 : 0;
    control->reason_mask = kGmeReasonBlockMvOverflow;
  }
}

__global__ void gme_extract_warps_kernel(const GmeWarpSample* warps,
                                         int count, GmePoint* out,
                                         int capacity,
                                         GmeJobControl* control) {
  __shared__ int s_scan[kGmeExtractThreads];
  __shared__ int s_running;
  __shared__ int s_overflow;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  if (tid == 0) {
    s_running = 0;
    s_overflow = 0;
  }
  __syncthreads();

  for (int base = 0; base < count; base += nthreads) {
    const int index = base + tid;
    int valid = 0;
    GmePoint samples[5];
    if (index < count) {
      const int result = gme_extract_points_from_warp(warps[index], samples);
      if (result < 0) atomicOr(&s_overflow, 1);
      valid = result > 0 ? result : 0;
    }
    s_scan[tid] = valid;
    __syncthreads();
    scan_shared(s_scan, tid, nthreads);
    const int my_offset = s_running + (tid > 0 ? s_scan[tid - 1] : 0);
    if (index < count && valid == 5) {
      for (int i = 0; i < 5; ++i) {
        if (my_offset + i < capacity) out[my_offset + i] = samples[i];
      }
    }
    const int chunk_total = s_scan[nthreads - 1];
    __syncthreads();
    if (tid == nthreads - 1) s_running += chunk_total;
    __syncthreads();
  }
  __syncthreads();
  if (tid == 0) {
    control->point_count = s_running;
    control->overflow_mask = s_overflow;
    control->too_many_mask = s_running > capacity ? 1 : 0;
    control->reason_mask = kGmeReasonWarpOverflow;
  }
}

__global__ void gme_fit_kernel(GmePoint* points, int count,
                               GmeOptions options,
                               const GmeJobControl* control,
                               GmePoint* active, GmeI64* residuals,
                               GmeU8* keep, GmeEstimate* results,
                               int extracted) {
  const int job = static_cast<int>(blockIdx.x);
  if (threadIdx.x != 0) return;
  GmeJobControl state;
  state.point_count = 0;
  state.overflow_mask = 0;
  state.too_many_mask = 0;
  state.reason_mask = kGmeReasonNone;
  if (extracted) {
    state = control[job];
    if (state.overflow_mask) {
      results[job] = gme_make_identity_estimate_reason(state.reason_mask);
      return;
    }
    if (state.too_many_mask) {
      GmeEstimate out = gme_make_identity_estimate();
      out.diag.reason = kGmeReasonTooManyCorrespondences;
      results[job] = out;
      return;
    }
    count = state.point_count;
  }
  results[job] =
      gme_estimate_points_core(points, count, options, active, residuals, keep);
}

__global__ void gme_warp_plane_kernel(GmeParams params, const GmeU8* src,
                                      int src_w, int src_h, int src_stride,
                                      GmeU8* dst, int dst_w, int dst_h,
                                      int dst_stride, int bit_depth) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = dst_w * dst_h;
  if (index >= total) return;
  const int y = index / dst_w;
  const int x = index - y * dst_w;
  dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_stride) +
      static_cast<std::size_t>(x)] =
      gme_warp_sample(params, src, src_w, src_h, src_stride, x, y, bit_depth);
}

__global__ void gme_warp_sse_kernel(GmeParams params, const GmeU8* ref,
                                    int ref_w, int ref_h, int ref_stride,
                                    const GmeU8* cur, int cur_w, int cur_h,
                                    int cur_stride, int bit_depth,
                                    unsigned long long* sse_out) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = cur_w * cur_h;
  if (index >= total) return;
  const int y = index / cur_w;
  const int x = index - y * cur_w;
  const int predicted = static_cast<int>(gme_warp_sample(
      params, ref, ref_w, ref_h, ref_stride, x, y, bit_depth));
  const int source =
      cur[static_cast<std::size_t>(y) * static_cast<std::size_t>(cur_stride) +
          static_cast<std::size_t>(x)];
  const int delta = predicted - source;
  atomicAdd(sse_out, static_cast<unsigned long long>(delta * delta));
}

}  // namespace

GmeStatus estimate_points(const GmeBuffers& buffers, int job, int count,
                          const GmeOptions& options, cudaStream_t stream,
                          std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "estimate_points", "buffers are not initialized");
    return GmeStatus::kNotInitialized;
  }
  if (!valid_job(buffers, job)) {
    set_error(error, "estimate_points", "job index out of range");
    return GmeStatus::kInvalidArgument;
  }
  if (count < 0 || count > buffers.max_points) {
    set_error(error, "estimate_points",
              "count must be in [0, max_points=" +
                  std::to_string(buffers.max_points) + "]");
    return GmeStatus::kInvalidArgument;
  }
  gme_fit_kernel<<<1, 1, 0, stream>>>(
      points_for(buffers, job), count, options, nullptr,
      active_for(buffers, job), residuals_for(buffers, job),
      keep_for(buffers, job), buffers.d_results + job, 0);
  const GmeStatus status = from_cuda(cudaGetLastError());
  if (status != GmeStatus::kOk) {
    set_error(error, "estimate_points", "kernel launch failed");
  }
  return status;
}

GmeStatus estimate_blocks(const GmeBuffers& buffers, int job, int count,
                          const GmeOptions& options, cudaStream_t stream,
                          std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "estimate_blocks", "buffers are not initialized");
    return GmeStatus::kNotInitialized;
  }
  if (!valid_job(buffers, job)) {
    set_error(error, "estimate_blocks", "job index out of range");
    return GmeStatus::kInvalidArgument;
  }
  if (count < 0 || count > buffers.max_blocks) {
    set_error(error, "estimate_blocks",
              "count must be in [0, max_blocks=" +
                  std::to_string(buffers.max_blocks) + "]");
    return GmeStatus::kInvalidArgument;
  }
  GmeJobControl* control = buffers.d_control + job;
  gme_extract_blocks_kernel<<<1, kGmeExtractThreads, 0, stream>>>(
      blocks_for(buffers, job), count, points_for(buffers, job),
      buffers.max_points, control);
  GmeStatus status = from_cuda(cudaGetLastError());
  if (status != GmeStatus::kOk) {
    set_error(error, "estimate_blocks", "extraction launch failed");
    return status;
  }
  gme_fit_kernel<<<1, 1, 0, stream>>>(
      points_for(buffers, job), 0, options, control,
      active_for(buffers, job), residuals_for(buffers, job),
      keep_for(buffers, job), buffers.d_results + job, 1);
  status = from_cuda(cudaGetLastError());
  if (status != GmeStatus::kOk) {
    set_error(error, "estimate_blocks", "fit launch failed");
  }
  return status;
}

GmeStatus estimate_warps(const GmeBuffers& buffers, int job, int count,
                         const GmeOptions& options, cudaStream_t stream,
                         std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "estimate_warps", "buffers are not initialized");
    return GmeStatus::kNotInitialized;
  }
  if (!valid_job(buffers, job)) {
    set_error(error, "estimate_warps", "job index out of range");
    return GmeStatus::kInvalidArgument;
  }
  if (count < 0 || count > buffers.max_warps) {
    set_error(error, "estimate_warps",
              "count must be in [0, max_warps=" +
                  std::to_string(buffers.max_warps) + "]");
    return GmeStatus::kInvalidArgument;
  }
  GmeJobControl* control = buffers.d_control + job;
  gme_extract_warps_kernel<<<1, kGmeExtractThreads, 0, stream>>>(
      warps_for(buffers, job), count, points_for(buffers, job),
      buffers.max_points, control);
  GmeStatus status = from_cuda(cudaGetLastError());
  if (status != GmeStatus::kOk) {
    set_error(error, "estimate_warps", "extraction launch failed");
    return status;
  }
  gme_fit_kernel<<<1, 1, 0, stream>>>(
      points_for(buffers, job), 0, options, control,
      active_for(buffers, job), residuals_for(buffers, job),
      keep_for(buffers, job), buffers.d_results + job, 1);
  status = from_cuda(cudaGetLastError());
  if (status != GmeStatus::kOk) {
    set_error(error, "estimate_warps", "fit launch failed");
  }
  return status;
}

GmeStatus warp_plane_bilinear(const GmeBuffers& buffers,
                              const GmeParams& params, const GmeU8* d_src,
                              int src_w, int src_h, int src_stride,
                              GmeU8* d_dst, int dst_w, int dst_h,
                              int dst_stride, int bit_depth,
                              cudaStream_t stream, std::string* error) {
  (void)buffers;
  if (d_src == nullptr || d_dst == nullptr || src_w <= 0 || src_h <= 0 ||
      dst_w <= 0 || dst_h <= 0 || src_stride < src_w ||
      dst_stride < dst_w || bit_depth != 8) {
    set_error(error, "warp_plane_bilinear", "invalid plane arguments");
    return GmeStatus::kInvalidArgument;
  }
  const int total = dst_w * dst_h;
  const int blocks = (total + kGmeImageThreads - 1) / kGmeImageThreads;
  gme_warp_plane_kernel<<<blocks, kGmeImageThreads, 0, stream>>>(
      params, d_src, src_w, src_h, src_stride, d_dst, dst_w, dst_h,
      dst_stride, bit_depth);
  const GmeStatus status = from_cuda(cudaGetLastError());
  if (status != GmeStatus::kOk) {
    set_error(error, "warp_plane_bilinear", "kernel launch failed");
  }
  return status;
}

GmeStatus warp_plane_sse(const GmeBuffers& buffers, const GmeParams& params,
                         const GmeU8* d_ref, int ref_w, int ref_h,
                         int ref_stride, const GmeU8* d_cur, int cur_w,
                         int cur_h, int cur_stride, int bit_depth,
                         GmeI64* d_sse, cudaStream_t stream,
                         std::string* error) {
  (void)buffers;
  if (d_ref == nullptr || d_cur == nullptr || d_sse == nullptr ||
      ref_w <= 0 || ref_h <= 0 || cur_w <= 0 || cur_h <= 0 ||
      ref_stride < ref_w || cur_stride < cur_w || bit_depth != 8) {
    set_error(error, "warp_plane_sse", "invalid plane arguments");
    return GmeStatus::kInvalidArgument;
  }
  cudaError_t result = cudaMemsetAsync(d_sse, 0, sizeof(GmeI64), stream);
  if (result != cudaSuccess) {
    set_error(error, "warp_plane_sse", "SSE reset failed");
    return GmeStatus::kCudaError;
  }
  const int total = cur_w * cur_h;
  const int blocks = (total + kGmeImageThreads - 1) / kGmeImageThreads;
  gme_warp_sse_kernel<<<blocks, kGmeImageThreads, 0, stream>>>(
      params, d_ref, ref_w, ref_h, ref_stride, d_cur, cur_w, cur_h,
      cur_stride, bit_depth,
      reinterpret_cast<unsigned long long*>(d_sse));
  const GmeStatus status = from_cuda(cudaGetLastError());
  if (status != GmeStatus::kOk) {
    set_error(error, "warp_plane_sse", "kernel launch failed");
  }
  return status;
}

}  // namespace av2enc::cuda::gme
