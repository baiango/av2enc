// Execution half of the M1 wavefront scheduler: level dispatch, eager
// fallback and CUDA-graph capture/replay.
//
// The scheduler owns no arithmetic.  A caller supplies three stage callbacks
// (`LevelOps`) that enqueue the real pipeline kernels for one independent
// level:
//
//   predict     -> intra prediction (luma + chroma, CfL) and residual
//   transform   -> downsample / forward DCT / quantise / dequantise / inverse
//   reconstruct -> clip1(pred + residual) into the reconstruction planes
//   post        -> once per frame (deblock, coefficient scatter, ...)
//
// The callbacks run on the supplied stream and must not allocate, block, or
// touch host-side mutable state that changes between frames: those are exactly
// the conditions for capture.  Everything the callbacks read must already be
// in device memory (descriptors are built once in prepare()).
//
// `prepare()` runs the sequence once eagerly (module warm-up), captures it with
// cudaStreamBeginCapture, instantiates the graph and performs one warm replay.
// `run()` is then one cudaGraphLaunch for the whole frame.  If the driver
// refuses capture, graph_available() is false and run() falls back to the eager
// batched sequence.

#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "cuda/pipeline/still/sched/sched.h"

namespace av2enc::cuda::still::sched {

struct LevelOps {
  void* user = nullptr;
  bool (*predict)(void* user, const Plan& plan, const LevelBatch& batch,
                  cudaStream_t stream, std::string& error) = nullptr;
  bool (*transform)(void* user, const Plan& plan, const LevelBatch& batch,
                    cudaStream_t stream, std::string& error) = nullptr;
  bool (*reconstruct)(void* user, const Plan& plan, const LevelBatch& batch,
                      cudaStream_t stream, std::string& error) = nullptr;
  // Optional: called once after the last level (deblock, coefficient scatter).
  bool (*post)(void* user, const Plan& plan, cudaStream_t stream,
               std::string& error) = nullptr;
  // Optional: number of kernels enqueued by the whole eager sequence.  When
  // null, stats fall back to the plan's launch estimate.
  int (*launch_count)(void* user) = nullptr;
};

// Dispatches one level: predict -> transform -> reconstruct for its batch.
bool run_level(const Plan& plan, int level, const LevelOps& ops,
               cudaStream_t stream, std::string& error);

// Enqueues the full per-frame sequence without a graph.
bool run_levels_eager(const Plan& plan, const LevelOps& ops,
                      cudaStream_t stream, RunStats* stats,
                      std::string& error);

// Persistent captured graph for one plan + one LevelOps instance.  The caller
// owns the device, stream and all device buffers; `stream` is the stream the
// callbacks capture on and the stream run() replays on.
class LevelGraph {
 public:
  LevelGraph();
  ~LevelGraph();
  LevelGraph(const LevelGraph&) = delete;
  LevelGraph& operator=(const LevelGraph&) = delete;

  bool prepare(const Plan& plan, const LevelOps& ops, cudaStream_t stream,
               std::string& error);
  bool run(const LevelOps& ops, const RunOptions& options, RunStats* stats,
           std::string& error);
  void release();

  bool prepared() const;
  bool graph_available() const;
  const std::string& graph_error() const;
  int graph_nodes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace av2enc::cuda::still::sched
