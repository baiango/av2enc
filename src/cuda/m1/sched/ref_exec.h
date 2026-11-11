// Reference M1 stage executor for scheduler verification.
//
// This is the "stage 1" block pipeline driven by a scheduler plan: it runs
// prediction (intra_modes), CfL, forward/inverse transform (tx_variants) and
// reconstruction with the same shared components and the same arguments as
// src/encoder/m1/m1_still.cpp.  It exists so the scheduler can be tested for
// bit-exact reconstruction against the CPU M1 with the same decisions; the
// production orchestrator supplies its own LevelOps.
//
// Not part of the product build (file name ends in _bench.cu so the root CMake
// M1 glob skips it).

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <cuda_runtime.h>

#include "cuda/m1/sched/sched.h"
#include "cuda/m1/sched/sched_exec.h"

namespace av2enc::cuda::m1::sched {

struct RefExecOptions {
  bool enable_edge_filter = true;  // M1Options::enable_intra_edge_filter
};

// Owns one persistent arena with the recon planes, source planes and per-job
// scratch.  prepare() is the only place that allocates.
class RefM1Executor {
 public:
  RefM1Executor();
  ~RefM1Executor();
  RefM1Executor(const RefM1Executor&) = delete;
  RefM1Executor& operator=(const RefM1Executor&) = delete;

  bool prepare(const Plan& plan, const FramePlan& frame,
               const RefExecOptions& options, cudaStream_t stream,
               std::string& error);

  LevelOps ops();

  bool upload_source(const std::uint8_t* y, const std::uint8_t* u,
                     const std::uint8_t* v, cudaStream_t stream,
                     std::string& error);
  bool zero_recon(cudaStream_t stream, std::string& error);
  bool download_recon(std::uint8_t* y, std::uint8_t* u, std::uint8_t* v,
                      cudaStream_t stream, std::string& error);

  int recon_stride(int plane) const;
  int recon_height(int plane) const;
  std::size_t workspace_bytes() const;
  int launch_count() const;
  void reset_launch_count();

 public:
  struct Impl;  // opaque; defined in sched_ref_bench.cu

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace av2enc::cuda::m1::sched
