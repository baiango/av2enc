// CUDA kernels for the AV2 §7.13.3.15 sub-PU grid.
//
// One thread executes one descriptor through the shared `subpu_device.cuh`
// core, so the device arithmetic is literally the same code the host
// emulation test compares against the CPU reference.  Descriptors in one
// batch must target disjoint MI cells (partition leaves are disjoint); the
// kernel does not serialise overlapping writes.
//
// No allocation, no copies outside the persistent staging arrays, and no
// hidden work beyond the one launch per entry point.  `save_batch()` owns its
// stream synchronisation so its result array can be read on the host.
#include "cuda/codec/filters/deblock/subpu/subpu.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace av2enc::cuda::subpu {
namespace {

constexpr int kSubPuThreads = 256;

void set_error(std::string* error, const char* context,
               const std::string& detail) {
  if (error != nullptr) {
    *error = std::string(context) + ": " + detail;
  }
}

bool buffers_ready(const SubPuBuffers& buffers) noexcept {
  if (buffers.grid.mi_rows <= 0 || buffers.grid.mi_cols <= 0 ||
      buffers.max_jobs <= 0 || buffers.d_jobs == nullptr ||
      buffers.d_codes == nullptr || buffers.d_tx_table == nullptr) {
    return false;
  }
  for (int plane = 0; plane < kSubPuPlaneCount; ++plane) {
    if (buffers.grid.size[plane] == nullptr ||
        buffers.grid.col_base[plane] == nullptr ||
        buffers.grid.row_base[plane] == nullptr) {
      return false;
    }
  }
  return true;
}

SubPuStatus from_cuda(cudaError_t result) noexcept {
  return status_from_cuda(result) == Status::kOk ? SubPuStatus::kOk
                                                 : SubPuStatus::kCudaError;
}

__global__ void reset_grid_kernel(SubPuGridView grid, int cells) {
  const int stride = static_cast<int>(blockDim.x * gridDim.x);
  for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
       index < cells; index += stride) {
    for (int plane = 0; plane < kSubPuPlaneCount; ++plane) {
      reset_grid_cell(grid, plane, index);
    }
  }
}

__global__ void save_subpu_kernel(const SubPuJob* jobs, int count,
                                  SubPuGridView grid, SubPuResult* codes) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) {
    return;
  }
  const SubPuJob job = jobs[index];
  codes[index].code = save_subpu(job.plane, job.x, job.y, job.w, job.h,
                                 job.subsampling_x, job.subsampling_y,
                                 job.pu_width, job.pu_height, grid);
}

__global__ void tx_table_kernel(SubPuTxEntry* out) {
  const int index = static_cast<int>(threadIdx.x);
  if (index >= kTxSizeCount) {
    return;
  }
  out[index] = make_tx_entry(index);
}

int grid_block_count(int cells) noexcept {
  const int blocks = (cells + kSubPuThreads - 1) / kSubPuThreads;
  return blocks < 1024 ? blocks : 1024;
}

}  // namespace

SubPuStatus reset_grid(const SubPuBuffers& buffers, cudaStream_t stream,
                       std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "reset_grid", "buffers are not initialized");
    return SubPuStatus::kNotInitialized;
  }
  const int cells = buffers.grid.mi_rows * buffers.grid.mi_cols;
  if (cells <= 0) {
    set_error(error, "reset_grid", "empty grid");
    return SubPuStatus::kInvalidArgument;
  }
  reset_grid_kernel<<<grid_block_count(cells), kSubPuThreads, 0, stream>>>(
      buffers.grid.view(), cells);
  const SubPuStatus status = from_cuda(cudaGetLastError());
  if (status != SubPuStatus::kOk) {
    set_error(error, "reset_grid", "kernel launch failed");
  }
  return status;
}

SubPuStatus save_batch(const SubPuBuffers& buffers, const SubPuJob* host_jobs,
                       int count, SubPuResult* host_results,
                       cudaStream_t stream, std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "save_batch", "buffers are not initialized");
    return SubPuStatus::kNotInitialized;
  }
  if (count < 0 || count > buffers.max_jobs) {
    set_error(error, "save_batch",
              "count must be in [0, max_jobs=" +
                  std::to_string(buffers.max_jobs) + "]");
    return SubPuStatus::kInvalidArgument;
  }
  if (count == 0) {
    return SubPuStatus::kOk;
  }
  if (host_jobs == nullptr) {
    set_error(error, "save_batch", "null host jobs");
    return SubPuStatus::kInvalidArgument;
  }

  cudaError_t result = cudaMemcpyAsync(
      buffers.d_jobs, host_jobs,
      static_cast<std::size_t>(count) * sizeof(SubPuJob),
      cudaMemcpyHostToDevice, stream);
  if (result != cudaSuccess) {
    set_error(error, "save_batch", "host-to-device job copy failed");
    return SubPuStatus::kCudaError;
  }

  const int blocks = (count + kSubPuThreads - 1) / kSubPuThreads;
  save_subpu_kernel<<<blocks, kSubPuThreads, 0, stream>>>(
      buffers.d_jobs, count, buffers.grid.view(), buffers.d_codes);
  result = cudaGetLastError();
  if (result != cudaSuccess) {
    set_error(error, "save_batch", "kernel launch failed");
    return SubPuStatus::kCudaError;
  }

  if (host_results != nullptr) {
    result = cudaMemcpyAsync(
        host_results, buffers.d_codes,
        static_cast<std::size_t>(count) * sizeof(SubPuResult),
        cudaMemcpyDeviceToHost, stream);
    if (result != cudaSuccess) {
      set_error(error, "save_batch", "device-to-host result copy failed");
      return SubPuStatus::kCudaError;
    }
  }

  result = cudaStreamSynchronize(stream);
  if (result != cudaSuccess) {
    set_error(error, "save_batch", "stream synchronisation failed");
    return SubPuStatus::kCudaError;
  }
  return SubPuStatus::kOk;
}

SubPuStatus save_one(const SubPuBuffers& buffers, const SubPuJob& job,
                     SubPuResult* result, cudaStream_t stream,
                     std::string* error) {
  return save_batch(buffers, &job, 1, result, stream, error);
}

SubPuStatus build_tx_table(const SubPuBuffers& buffers, cudaStream_t stream,
                           std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "build_tx_table", "buffers are not initialized");
    return SubPuStatus::kNotInitialized;
  }
  tx_table_kernel<<<1, 32, 0, stream>>>(buffers.d_tx_table);
  const SubPuStatus status = from_cuda(cudaGetLastError());
  if (status != SubPuStatus::kOk) {
    set_error(error, "build_tx_table", "kernel launch failed");
    return status;
  }
  const cudaError_t sync = cudaStreamSynchronize(stream);
  if (sync != cudaSuccess) {
    set_error(error, "build_tx_table", "stream synchronisation failed");
    return SubPuStatus::kCudaError;
  }
  return SubPuStatus::kOk;
}

}  // namespace av2enc::cuda::subpu
