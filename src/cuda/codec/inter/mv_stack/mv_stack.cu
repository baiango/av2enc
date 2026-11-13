// CUDA kernels and launchers for the AV2 §7.12.2 spatial MV stack.
//
// One thread per job: the stack scan is a short serial algorithm with a handful
// of neighbours, so a thread-per-block launch is the natural batching.  A frame
// at 1080p has ~32k 8x8 blocks; 128 threads/block keeps the scheduler busy.
//
// No allocation, no copy and no synchronisation in `*_batch`.  The Workspace
// helpers own the only device buffers and the only copies.

#include "mv_stack.h"

namespace av2enc::cuda::mvstack {
namespace {

constexpr int kThreads = 128;

__global__ void stack_build_kernel(const StackJob* jobs, StackResult* results,
                                   int count) {
  const int index = static_cast<int>(blockIdx.x) * kThreads +
                    static_cast<int>(threadIdx.x);
  if (index >= count) {
    return;
  }
  StackResult& out = results[index];
  out.status = static_cast<int>(mvs_build_stack(jobs[index], &out.stack));
}

__global__ void pair_build_kernel(const PairJob* jobs, PairStacks* results,
                                  int count) {
  const int index = static_cast<int>(blockIdx.x) * kThreads +
                    static_cast<int>(threadIdx.x);
  if (index >= count) {
    return;
  }
  mvs_build_pair_stacks(jobs[index], &results[index]);
}

__global__ void merge_set_kernel(const PairStacks* pairs, MergeSet* sets,
                                 int count) {
  const int index = static_cast<int>(blockIdx.x) * kThreads +
                    static_cast<int>(threadIdx.x);
  if (index >= count) {
    return;
  }
  mvs_build_merge_set(pairs[index], &sets[index]);
}

inline int grid_for(int count) { return (count + kThreads - 1) / kThreads; }

std::size_t round_up(std::size_t bytes) {
  const std::size_t alignment = Arena::kAlignment;
  return (bytes + alignment - 1) / alignment * alignment;
}

}  // namespace

cudaError_t stack_build_batch(const StackJob* d_jobs, StackResult* d_results,
                              int count, cudaStream_t stream) {
  if (count < 0) {
    return cudaErrorInvalidValue;
  }
  if (count == 0) {
    return cudaSuccess;
  }
  if (d_jobs == nullptr || d_results == nullptr) {
    return cudaErrorInvalidValue;
  }
  stack_build_kernel<<<grid_for(count), kThreads, 0, stream>>>(d_jobs,
                                                              d_results, count);
  return cudaGetLastError();
}

cudaError_t pair_build_batch(const PairJob* d_jobs, PairStacks* d_results,
                             int count, cudaStream_t stream) {
  if (count < 0) {
    return cudaErrorInvalidValue;
  }
  if (count == 0) {
    return cudaSuccess;
  }
  if (d_jobs == nullptr || d_results == nullptr) {
    return cudaErrorInvalidValue;
  }
  pair_build_kernel<<<grid_for(count), kThreads, 0, stream>>>(d_jobs,
                                                             d_results, count);
  return cudaGetLastError();
}

cudaError_t merge_set_batch(const PairStacks* d_pairs, MergeSet* d_sets,
                            int count, cudaStream_t stream) {
  if (count < 0) {
    return cudaErrorInvalidValue;
  }
  if (count == 0) {
    return cudaSuccess;
  }
  if (d_pairs == nullptr || d_sets == nullptr) {
    return cudaErrorInvalidValue;
  }
  merge_set_kernel<<<grid_for(count), kThreads, 0, stream>>>(d_pairs, d_sets,
                                                            count);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Arena workspace
// ---------------------------------------------------------------------------

std::size_t required_device_bytes(const MvStackConfig& config) {
  if (config.max_jobs <= 0) {
    return 0;
  }
  const std::size_t n = static_cast<std::size_t>(config.max_jobs);
  return round_up(sizeof(StackJob) * n) + round_up(sizeof(StackResult) * n) +
         round_up(sizeof(PairJob) * n) + round_up(sizeof(PairStacks) * n) +
         round_up(sizeof(MergeSet) * n);
}

Status Workspace::create(Arena& arena, const MvStackConfig& config,
                         std::string* error, std::size_t* out_required_bytes) {
  if (config.max_jobs < 0) {
    if (error != nullptr) {
      *error = "mv_stack: max_jobs must be non-negative";
    }
    return Status::kInvalidArgument;
  }
  config_ = config;
  required_bytes_ = required_device_bytes(config);
  created_ = false;
  if (out_required_bytes != nullptr) {
    *out_required_bytes = required_bytes_;
  }
  if (config.max_jobs == 0) {
    created_ = true;
    persistent_mark_ = arena.mark();
    return Status::kOk;
  }

  const std::uint64_t mark = arena.mark();
  const std::size_t n = static_cast<std::size_t>(config.max_jobs);
  void* p = nullptr;
  const auto alloc = [&](std::size_t bytes, void** out) {
    return arena.allocate_checked(bytes, out);
  };
  Status status = alloc(round_up(sizeof(StackJob) * n), &p);
  if (status != Status::kOk) {
    arena.rewind(mark);
    return status;
  }
  stack_jobs_ = static_cast<StackJob*>(p);
  status = alloc(round_up(sizeof(StackResult) * n), &p);
  if (status != Status::kOk) {
    arena.rewind(mark);
    return status;
  }
  stack_results_ = static_cast<StackResult*>(p);
  status = alloc(round_up(sizeof(PairJob) * n), &p);
  if (status != Status::kOk) {
    arena.rewind(mark);
    return status;
  }
  pair_jobs_ = static_cast<PairJob*>(p);
  status = alloc(round_up(sizeof(PairStacks) * n), &p);
  if (status != Status::kOk) {
    arena.rewind(mark);
    return status;
  }
  pair_results_ = static_cast<PairStacks*>(p);
  status = alloc(round_up(sizeof(MergeSet) * n), &p);
  if (status != Status::kOk) {
    arena.rewind(mark);
    return status;
  }
  merge_sets_ = static_cast<MergeSet*>(p);

  persistent_mark_ = arena.mark();
  created_ = true;
  if (error != nullptr) {
    error->clear();
  }
  return Status::kOk;
}

Status Workspace::upload_stack_jobs(const StackJob* host_jobs, int count,
                                    cudaStream_t stream) {
  if (!created_ || host_jobs == nullptr || count < 0 ||
      count > config_.max_jobs) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  return status_from_cuda(cudaMemcpyAsync(
      stack_jobs_, host_jobs, sizeof(StackJob) * static_cast<std::size_t>(count),
      cudaMemcpyHostToDevice, stream));
}

Status Workspace::run_stacks(int count, cudaStream_t stream) {
  if (!created_ || count < 0 || count > config_.max_jobs) {
    return Status::kInvalidArgument;
  }
  return status_from_cuda(
      stack_build_batch(stack_jobs_, stack_results_, count, stream));
}

Status Workspace::download_stacks(StackResult* host_results, int count,
                                  cudaStream_t stream) {
  if (!created_ || host_results == nullptr || count < 0 ||
      count > config_.max_jobs) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  const Status copy = status_from_cuda(cudaMemcpyAsync(
      host_results, stack_results_,
      sizeof(StackResult) * static_cast<std::size_t>(count),
      cudaMemcpyDeviceToHost, stream));
  if (copy != Status::kOk) {
    return copy;
  }
  return status_from_cuda(cudaStreamSynchronize(stream));
}

Status Workspace::upload_pair_jobs(const PairJob* host_jobs, int count,
                                   cudaStream_t stream) {
  if (!created_ || host_jobs == nullptr || count < 0 ||
      count > config_.max_jobs) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  return status_from_cuda(cudaMemcpyAsync(
      pair_jobs_, host_jobs, sizeof(PairJob) * static_cast<std::size_t>(count),
      cudaMemcpyHostToDevice, stream));
}

Status Workspace::run_pairs(int count, cudaStream_t stream) {
  if (!created_ || count < 0 || count > config_.max_jobs) {
    return Status::kInvalidArgument;
  }
  return status_from_cuda(
      pair_build_batch(pair_jobs_, pair_results_, count, stream));
}

Status Workspace::download_pairs(PairStacks* host_results, int count,
                                 cudaStream_t stream) {
  if (!created_ || host_results == nullptr || count < 0 ||
      count > config_.max_jobs) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  const Status copy = status_from_cuda(cudaMemcpyAsync(
      host_results, pair_results_,
      sizeof(PairStacks) * static_cast<std::size_t>(count),
      cudaMemcpyDeviceToHost, stream));
  if (copy != Status::kOk) {
    return copy;
  }
  return status_from_cuda(cudaStreamSynchronize(stream));
}

Status Workspace::run_merge_sets(int count, cudaStream_t stream) {
  if (!created_ || count < 0 || count > config_.max_jobs) {
    return Status::kInvalidArgument;
  }
  return status_from_cuda(
      merge_set_batch(pair_results_, merge_sets_, count, stream));
}

Status Workspace::download_merge_sets(MergeSet* host_sets, int count,
                                      cudaStream_t stream) {
  if (!created_ || host_sets == nullptr || count < 0 ||
      count > config_.max_jobs) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  const Status copy = status_from_cuda(cudaMemcpyAsync(
      host_sets, merge_sets_,
      sizeof(MergeSet) * static_cast<std::size_t>(count),
      cudaMemcpyDeviceToHost, stream));
  if (copy != Status::kOk) {
    return copy;
  }
  return status_from_cuda(cudaStreamSynchronize(stream));
}

}  // namespace av2enc::cuda::mvstack
