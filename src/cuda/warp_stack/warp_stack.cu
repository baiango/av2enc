// CUDA warp stack: kernels and arena-backed batched launchers.
//
// There is no allocation, no host copy and no synchronisation in the launch
// paths. `Workspace::create()` is the only place that touches the Arena; upload
// helpers use cudaMemcpyAsync into the persistent descriptor tables.

#include "warp_stack.h"

#include "warp_stack_device.cuh"

namespace av2enc::cuda::warpstack {
namespace {

__global__ void predict_kernel(const WarpPredictJob* jobs, WarpPredictResult* results) {
  __shared__ std::int32_t intermediate[kWarpIntermediateInts];
  const int index = static_cast<int>(blockIdx.x);
  const WarpPredictJob& job = jobs[index];
  WarpPredictResult* out = (results != nullptr) ? &results[index] : nullptr;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthr = static_cast<int>(blockDim.x);
  if (job.geom.bit_depth > 8) {
    ws_predict_job_device<std::uint16_t>(job, tid, nthr, intermediate, out);
  } else {
    ws_predict_job_device<std::uint8_t>(job, tid, nthr, intermediate, out);
  }
}

__global__ void corner_kernel(const WarpCornerJob* jobs, WarpCornerResult* results) {
  const int index = static_cast<int>(blockIdx.x);
  const WarpCornerJob& job = jobs[index];
  WarpCornerResult result{};
  result.model_valid =
      ws_generate_points_from_corners(job.grid, job.sb_h4, job.mi_row, job.mi_col, job.bw4,
                                      job.bh4, job.ref_frame0, job.iter, &result.corner)
          ? 1
          : 0;
  results[index] = result;
}

__global__ void sample_kernel(const WarpSampleJob* jobs, inter::warp::WarpSampleSet* results) {
  const int index = static_cast<int>(blockIdx.x);
  (void)ws_find_warp_samples(jobs[index].params, &results[index]);
}

__global__ void estimate_kernel(const WarpEstimateJob* jobs, ia::WarpParams* results) {
  const int index = static_cast<int>(blockIdx.x);
  const WarpEstimateJob& job = jobs[index];
  results[index] = ws_estimate_warp_from_samples(job.samples, job.mid_x, job.mid_y,
                                                 job.block_mv);
}

}  // namespace

// ---------------------------------------------------------------------------
// Raw launchers
// ---------------------------------------------------------------------------

Status predict_launch(const WarpPredictJob* device_jobs, WarpPredictResult* results, int count,
                      cudaStream_t stream) {
  if (count < 0 || (count > 0 && device_jobs == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  predict_kernel<<<count, kPredictThreads, 0, stream>>>(device_jobs, results);
  return status_from_cuda(cudaGetLastError());
}

Status corner_launch(const WarpCornerJob* device_jobs, WarpCornerResult* results, int count,
                     cudaStream_t stream) {
  if (count < 0 || (count > 0 && (device_jobs == nullptr || results == nullptr))) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  corner_kernel<<<count, 1, 0, stream>>>(device_jobs, results);
  return status_from_cuda(cudaGetLastError());
}

Status sample_launch(const WarpSampleJob* device_jobs, inter::warp::WarpSampleSet* results,
                     int count, cudaStream_t stream) {
  if (count < 0 || (count > 0 && (device_jobs == nullptr || results == nullptr))) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  sample_kernel<<<count, 1, 0, stream>>>(device_jobs, results);
  return status_from_cuda(cudaGetLastError());
}

Status estimate_launch(const WarpEstimateJob* device_jobs, ia::WarpParams* results, int count,
                       cudaStream_t stream) {
  if (count < 0 || (count > 0 && (device_jobs == nullptr || results == nullptr))) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  estimate_kernel<<<count, 1, 0, stream>>>(device_jobs, results);
  return status_from_cuda(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

namespace {

template <typename T>
Status copy_out_impl(const T* device_src, T* host_dst, int count, int capacity,
                     cudaStream_t stream) {
  if (count < 0 || count > capacity || (count > 0 && host_dst == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  return status_from_cuda(cudaMemcpyAsync(host_dst, device_src,
                                          static_cast<std::size_t>(count) * sizeof(T),
                                          cudaMemcpyDeviceToHost, stream));
}

}  // namespace

template <typename T>
Status Workspace::copy_in(T* device_dst, const T* host_src, int count, int capacity,
                          cudaStream_t stream) {
  if (!created_) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > capacity || (count > 0 && host_src == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  return status_from_cuda(cudaMemcpyAsync(device_dst, host_src,
                                          static_cast<std::size_t>(count) * sizeof(T),
                                          cudaMemcpyHostToDevice, stream));
}

Status Workspace::upload_predict(const WarpPredictJob* host_jobs, int count,
                                 cudaStream_t stream) {
  return copy_in(predict_jobs_, host_jobs, count, config_.max_predict_jobs, stream);
}

Status Workspace::upload_corner(const WarpCornerJob* host_jobs, int count,
                                cudaStream_t stream) {
  return copy_in(corner_jobs_, host_jobs, count, config_.max_corner_jobs, stream);
}

Status Workspace::upload_sample(const WarpSampleJob* host_jobs, int count,
                                cudaStream_t stream) {
  return copy_in(sample_jobs_, host_jobs, count, config_.max_sample_jobs, stream);
}

Status Workspace::upload_estimate(const WarpEstimateJob* host_jobs, int count,
                                  cudaStream_t stream) {
  return copy_in(estimate_jobs_, host_jobs, count, config_.max_estimate_jobs, stream);
}

Status Workspace::download_predict(WarpPredictResult* host_results, int count,
                                   cudaStream_t stream) const {
  if (!created_) {
    return Status::kNotInitialized;
  }
  return copy_out_impl(predict_results_, host_results, count, config_.max_predict_jobs, stream);
}

Status Workspace::download_corner(WarpCornerResult* host_results, int count,
                                  cudaStream_t stream) const {
  if (!created_) {
    return Status::kNotInitialized;
  }
  return copy_out_impl(corner_results_, host_results, count, config_.max_corner_jobs, stream);
}

Status Workspace::download_sample(inter::warp::WarpSampleSet* host_results, int count,
                                  cudaStream_t stream) const {
  if (!created_) {
    return Status::kNotInitialized;
  }
  return copy_out_impl(sample_sets_, host_results, count, config_.max_sample_jobs, stream);
}

Status Workspace::download_estimate(ia::WarpParams* host_results, int count,
                                    cudaStream_t stream) const {
  if (!created_) {
    return Status::kNotInitialized;
  }
  return copy_out_impl(estimate_params_, host_results, count, config_.max_estimate_jobs, stream);
}

Status Workspace::submit_predict(const WarpPredictJob* host_jobs, int count,
                                 cudaStream_t stream) {
  const Status status = upload_predict(host_jobs, count, stream);
  if (status != Status::kOk) {
    return status;
  }
  return run_predict(count, stream);
}

Status Workspace::submit_corner(const WarpCornerJob* host_jobs, int count,
                                cudaStream_t stream) {
  const Status status = upload_corner(host_jobs, count, stream);
  if (status != Status::kOk) {
    return status;
  }
  return run_corner(count, stream);
}

Status Workspace::submit_sample(const WarpSampleJob* host_jobs, int count,
                                cudaStream_t stream) {
  const Status status = upload_sample(host_jobs, count, stream);
  if (status != Status::kOk) {
    return status;
  }
  return run_sample(count, stream);
}

Status Workspace::submit_estimate(const WarpEstimateJob* host_jobs, int count,
                                  cudaStream_t stream) {
  const Status status = upload_estimate(host_jobs, count, stream);
  if (status != Status::kOk) {
    return status;
  }
  return run_estimate(count, stream);
}

Status Workspace::run_predict(int count, cudaStream_t stream) {
  if (!created_) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_predict_jobs) {
    return Status::kInvalidArgument;
  }
  return predict_launch(predict_jobs_, predict_results_, count, stream);
}

Status Workspace::run_corner(int count, cudaStream_t stream) {
  if (!created_) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_corner_jobs) {
    return Status::kInvalidArgument;
  }
  return corner_launch(corner_jobs_, corner_results_, count, stream);
}

Status Workspace::run_sample(int count, cudaStream_t stream) {
  if (!created_) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_sample_jobs) {
    return Status::kInvalidArgument;
  }
  return sample_launch(sample_jobs_, sample_sets_, count, stream);
}

Status Workspace::run_estimate(int count, cudaStream_t stream) {
  if (!created_) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_estimate_jobs) {
    return Status::kInvalidArgument;
  }
  return estimate_launch(estimate_jobs_, estimate_params_, count, stream);
}

}  // namespace av2enc::cuda::warpstack
