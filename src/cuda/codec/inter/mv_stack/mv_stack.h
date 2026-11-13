#ifndef AV2ENC_CUDA_INTER_MV_STACK_MV_STACK_H_
#define AV2ENC_CUDA_INTER_MV_STACK_MV_STACK_H_

// CUDA side of the AV2 v1.0.0 §7.12.2 spatial MV stack, DRL sort, multi-
// reference pair stacks and the compound merge-candidate set.
//
// The CPU reference lives in `src/codec/inter/merge/merge.cpp` and
// `src/codec/inter/multiref/mvstack_multi.cpp`; the parity tests compare every
// field.  This is the piece the video orchestrator needs when it moves block
// mode decisions off the host: one launch builds the stack for a whole frame's
// blocks, including §7.12.2.19 DRL reordering and the §7.12.2.16/.14 derived
// candidates.
//
// Ownership: `src/cuda/codec/inter/mv_stack/**`.  No CMake, pipeline or other
// module is touched.
//
// Arena policy: every device buffer comes from the caller's `Arena`.  There is
// no `cudaMalloc`/`cudaFree`/`cudaMallocHost` here or in the launchers.  The
// `Workspace` allocates once from `required_device_bytes()`; the raw
// `*_batch` entry points allocate nothing and take caller-owned device arrays.

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "cuda/error.h"
#include "cuda/codec/inter/mv_stack/mv_stack_device.cuh"
#include "codec/inter/merge/merge.h"

namespace av2enc::cuda::mvstack {

// ---------------------------------------------------------------------------
// POD batch descriptors
// ---------------------------------------------------------------------------

// One single-reference or compound stack job.  `params.mi` and
// `params.frame_distance` must be device pointers; everything else is a value.
using StackJob = video::merge::StackParams;

struct StackResult {
  video::merge::MvStack stack;
  // video::merge::Status as an int; 0 == kOk.
  int status = 0;
};

// Two single-list stacks plus the two-list (skip_mode) stack.
using PairJob = MvsPairJob;
using PairStacks = MvsPairStacks;
using MergeCandidate = MvsMergeCandidate;
using MergeSet = MvsMergeSet;

// ---------------------------------------------------------------------------
// Raw launchers (device arrays in, device arrays out)
// ---------------------------------------------------------------------------

// One thread per job.  Invalid jobs write `status != 0` and a zeroed stack.
cudaError_t stack_build_batch(const StackJob* d_jobs, StackResult* d_results,
                              int count, cudaStream_t stream = nullptr);

// Builds list-0, list-1 and the compound stack for each job.
cudaError_t pair_build_batch(const PairJob* d_jobs, PairStacks* d_results,
                             int count, cudaStream_t stream = nullptr);

// Fills the deterministic compound merge candidate set from a pair result.
cudaError_t merge_set_batch(const PairStacks* d_pairs, MergeSet* d_sets,
                            int count, cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Arena workspace
// ---------------------------------------------------------------------------

struct MvStackConfig {
  int max_jobs = 0;
};

// Exact persistent size, including the arena's 256-byte rounding of every
// allocation.  Zero bytes for `max_jobs == 0`.
std::size_t required_device_bytes(const MvStackConfig& config);

class Workspace {
 public:
  Workspace() = default;
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  // Allocates the descriptor/result tables once.  A short arena returns
  // `Status::kArenaExhausted`, writes the requirement to `*out_required_bytes`
  // and leaves the bump pointer untouched.
  Status create(Arena& arena, const MvStackConfig& config,
                std::string* error = nullptr,
                std::size_t* out_required_bytes = nullptr);

  bool created() const noexcept { return created_; }
  int max_jobs() const noexcept { return config_.max_jobs; }
  std::size_t required_bytes() const noexcept { return required_bytes_; }
  std::uint64_t persistent_mark() const noexcept { return persistent_mark_; }

  StackJob* stack_jobs() noexcept { return stack_jobs_; }
  StackResult* stack_results() noexcept { return stack_results_; }
  PairJob* pair_jobs() noexcept { return pair_jobs_; }
  PairStacks* pair_results() noexcept { return pair_results_; }
  MergeSet* merge_sets() noexcept { return merge_sets_; }

  // Upload `count` host jobs, launch, and (for the download forms) synchronize
  // the stream and copy results back.  `count <= max_jobs()`.
  Status upload_stack_jobs(const StackJob* host_jobs, int count,
                           cudaStream_t stream = nullptr);
  Status run_stacks(int count, cudaStream_t stream = nullptr);
  Status download_stacks(StackResult* host_results, int count,
                         cudaStream_t stream = nullptr);

  Status upload_pair_jobs(const PairJob* host_jobs, int count,
                          cudaStream_t stream = nullptr);
  Status run_pairs(int count, cudaStream_t stream = nullptr);
  Status download_pairs(PairStacks* host_results, int count,
                        cudaStream_t stream = nullptr);

  Status run_merge_sets(int count, cudaStream_t stream = nullptr);
  Status download_merge_sets(MergeSet* host_sets, int count,
                             cudaStream_t stream = nullptr);

 private:
  MvStackConfig config_{};
  bool created_ = false;
  std::size_t required_bytes_ = 0;
  std::uint64_t persistent_mark_ = 0;
  StackJob* stack_jobs_ = nullptr;
  StackResult* stack_results_ = nullptr;
  PairJob* pair_jobs_ = nullptr;
  PairStacks* pair_results_ = nullptr;
  MergeSet* merge_sets_ = nullptr;
};

}  // namespace av2enc::cuda::mvstack

#endif  // AV2ENC_CUDA_INTER_MV_STACK_MV_STACK_H_
