// AV2 §7.13.3.18 inter motion compensation — CUDA kernel.

#include "mc.h"

namespace av2enc::cuda {
namespace {

using inter::mc::BlockRect;
using inter::mc::InterpFilter;
using inter::mc::Mv;

__global__ void mc_predict_plane_kernel(
    const std::uint8_t* __restrict__ ref, int ref_stride, int first_x,
    int first_y, int last_x, int last_y, int start_x, int start_y, int step_x,
    int step_y, int filter_h, int filter_v, int w, int h,
    std::uint8_t* __restrict__ dst, int dst_stride) {
  const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int r = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (c >= w || r >= h) {
    return;
  }
  const std::int16_t* taps_h =
      &inter::mc::kSubpelFilters[filter_h][0][0];
  const std::int16_t* taps_v =
      &inter::mc::kSubpelFilters[filter_v][0][0];
  dst[static_cast<std::size_t>(r) * static_cast<std::size_t>(dst_stride) +
      static_cast<std::size_t>(c)] =
      inter::mc::mc_predict_sample(ref, ref_stride, first_x, first_y, last_x,
                                   last_y, start_x, start_y, step_x, step_y,
                                   taps_h, taps_v, r, c);
}

__global__ void mc_predict_batch_kernel(McDeviceFrame ref,
                                        const McBlockJob* __restrict__ jobs) {
  const McBlockJob job = jobs[blockIdx.x];
  const int sub_x = job.plane > 0 ? 1 : 0;
  const int sub_y = job.plane > 0 ? 1 : 0;
  const inter::mc::McSampleLocation loc = inter::mc::mc_derive_sample_location(
      job.x, job.y, job.mv, sub_x, sub_y);
  const int last_x =
      (job.plane == 0 ? ref.mi_cols * inter::mc::kMiSize
                      : (ref.mi_cols * inter::mc::kMiSize) >> 1) -
      1;
  const int last_y =
      (job.plane == 0 ? ref.mi_rows * inter::mc::kMiSize
                      : (ref.mi_rows * inter::mc::kMiSize) >> 1) -
      1;
  const inter::mc::InterpFilter filter_h =
      inter::mc::mc_pass_filter(job.filter, job.w <= 4);
  const inter::mc::InterpFilter filter_v =
      inter::mc::mc_pass_filter(job.filter, job.h <= 4);
  const std::int16_t* taps_h =
      &inter::mc::kSubpelFilters[static_cast<int>(filter_h)][0][0];
  const std::int16_t* taps_v =
      &inter::mc::kSubpelFilters[static_cast<int>(filter_v)][0][0];

  for (int r = static_cast<int>(threadIdx.y); r < job.h;
       r += static_cast<int>(blockDim.y)) {
    for (int c = static_cast<int>(threadIdx.x); c < job.w;
         c += static_cast<int>(blockDim.x)) {
      job.dst[static_cast<std::size_t>(r) *
                  static_cast<std::size_t>(job.dst_stride) +
              static_cast<std::size_t>(c)] =
          inter::mc::mc_predict_sample(ref.plane[job.plane],
                                       ref.stride[job.plane], 0, 0, last_x,
                                       last_y, loc.start_x, loc.start_y,
                                       loc.step_x, loc.step_y, taps_h, taps_v,
                                       r, c);
    }
  }
}

bool valid_plane(const McDeviceFrame& ref, int plane, const BlockRect& rect,
                 InterpFilter filter, const std::uint8_t* dst,
                 int dst_stride) {
  if (plane < 0 || plane > 2) {
    return false;
  }
  const int filter_index = static_cast<int>(filter);
  if (filter_index < 0 || filter_index >= inter::mc::kSubpelFilterTypes) {
    return false;
  }
  if (ref.mi_cols <= 0 || ref.mi_rows <= 0) {
    return false;
  }
  if (ref.plane[plane] == nullptr || dst == nullptr) {
    return false;
  }
  const int plane_w = (plane == 0 ? ref.mi_cols * inter::mc::kMiSize
                                  : (ref.mi_cols * inter::mc::kMiSize) >> 1);
  const int plane_h = (plane == 0 ? ref.mi_rows * inter::mc::kMiSize
                                  : (ref.mi_rows * inter::mc::kMiSize) >> 1);
  if (ref.stride[plane] < plane_w) {
    return false;
  }
  if (rect.w <= 0 || rect.h <= 0 || rect.x < 0 || rect.y < 0 ||
      rect.x + rect.w > plane_w || rect.y + rect.h > plane_h) {
    return false;
  }
  if (rect.w > inter::mc::kMaxBlockDim || rect.h > inter::mc::kMaxBlockDim) {
    return false;
  }
  if (dst_stride < rect.w) {
    return false;
  }
  return true;
}

cudaError_t launch_plane(const McDeviceFrame& ref, int plane,
                         const BlockRect& rect, Mv mv, InterpFilter filter,
                         std::uint8_t* dst, int dst_stride,
                         cudaStream_t stream) {
  if (!valid_plane(ref, plane, rect, filter, dst, dst_stride)) {
    return cudaErrorInvalidValue;
  }
  const int sub_x = plane > 0 ? 1 : 0;
  const int sub_y = plane > 0 ? 1 : 0;
  const inter::mc::McSampleLocation loc =
      inter::mc::mc_derive_sample_location(rect.x, rect.y, mv, sub_x, sub_y);
  const int first_x = 0;
  const int first_y = 0;
  const int last_x =
      (plane == 0 ? ref.mi_cols * inter::mc::kMiSize
                  : (ref.mi_cols * inter::mc::kMiSize) >> 1) -
      1;
  const int last_y =
      (plane == 0 ? ref.mi_rows * inter::mc::kMiSize
                  : (ref.mi_rows * inter::mc::kMiSize) >> 1) -
      1;
  const InterpFilter filter_h =
      inter::mc::mc_pass_filter(filter, rect.w <= 4);
  const InterpFilter filter_v =
      inter::mc::mc_pass_filter(filter, rect.h <= 4);

  constexpr int kThreadsX = 32;
  constexpr int kThreadsY = 8;
  const dim3 block(kThreadsX, kThreadsY);
  const dim3 grid(static_cast<unsigned>((rect.w + kThreadsX - 1) / kThreadsX),
                  static_cast<unsigned>((rect.h + kThreadsY - 1) / kThreadsY));

  mc_predict_plane_kernel<<<grid, block, 0, stream>>>(
      ref.plane[plane], ref.stride[plane], first_x, first_y, last_x, last_y,
      loc.start_x, loc.start_y, loc.step_x, loc.step_y,
      static_cast<int>(filter_h), static_cast<int>(filter_v), rect.w, rect.h,
      dst, dst_stride);
  return cudaGetLastError();
}

}  // namespace

cudaError_t mc_predict_plane(const McDeviceFrame& ref, int plane,
                             const BlockRect& rect, Mv mv,
                             InterpFilter filter, std::uint8_t* dst,
                             int dst_stride, cudaStream_t stream) {
  return launch_plane(ref, plane, rect, mv, filter, dst, dst_stride, stream);
}

cudaError_t mc_predict_block_420(const McDeviceFrame& ref,
                                 const BlockRect& luma_rect, Mv mv,
                                 InterpFilter filter, std::uint8_t* dst[3],
                                 int dst_stride[3], cudaStream_t stream) {
  if (dst == nullptr || dst_stride == nullptr) {
    return cudaErrorInvalidValue;
  }
  if ((luma_rect.x & 1) != 0 || (luma_rect.y & 1) != 0 ||
      (luma_rect.w & 1) != 0 || (luma_rect.h & 1) != 0) {
    return cudaErrorInvalidValue;
  }
  const BlockRect chroma_rect{luma_rect.x >> 1, luma_rect.y >> 1,
                              luma_rect.w >> 1, luma_rect.h >> 1};
  cudaError_t err =
      launch_plane(ref, 0, luma_rect, mv, filter, dst[0], dst_stride[0],
                   stream);
  if (err != cudaSuccess) {
    return err;
  }
  for (int plane = 1; plane <= 2; ++plane) {
    err = launch_plane(ref, plane, chroma_rect, mv, filter, dst[plane],
                       dst_stride[plane], stream);
    if (err != cudaSuccess) {
      return err;
    }
  }
  return cudaSuccess;
}

cudaError_t mc_predict_batch(const McDeviceFrame& ref,
                             const McBlockJob* device_jobs, int count,
                             cudaStream_t stream) {
  if (count <= 0) {
    return count == 0 ? cudaSuccess : cudaErrorInvalidValue;
  }
  if (device_jobs == nullptr) {
    return cudaErrorInvalidValue;
  }
  constexpr int kThreadsX = 32;
  constexpr int kThreadsY = 8;
  const dim3 block(kThreadsX, kThreadsY);
  const dim3 grid(static_cast<unsigned>(count));
  mc_predict_batch_kernel<<<grid, block, 0, stream>>>(ref, device_jobs);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
