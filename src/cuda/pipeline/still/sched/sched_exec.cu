// CUDA execution for the M1 wavefront plan.  No normative arithmetic lives
// here; the stage callbacks do the work and this file only orders and captures
// them.

#include "cuda/pipeline/still/sched/sched_exec.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <string>

namespace av2enc::cuda::still::sched {
namespace {

bool check_cuda(cudaError_t status, const char* what, std::string& error) {
  if (status == cudaSuccess) {
    return true;
  }
  error = std::string(what) + " failed: " + cudaGetErrorString(status);
  return false;
}

bool check_stage(bool ok, const char* stage, int level, std::string& error) {
  if (ok) {
    return true;
  }
  if (error.empty()) {
    error = std::string(stage) + " failed at level " + std::to_string(level);
  } else {
    error = std::string(stage) + " (level " + std::to_string(level) +
            "): " + error;
  }
  return false;
}

bool run_sequence(const Plan& plan, const LevelOps& ops, cudaStream_t stream,
                  std::string& error) {
  for (int level = 0; level < plan.levels(); ++level) {
    if (!run_level(plan, level, ops, stream, error)) {
      return false;
    }
  }
  if (ops.post != nullptr) {
    std::string post_error;
    if (!ops.post(ops.user, plan, stream, post_error)) {
      error = post_error.empty() ? "post stage failed" : post_error;
      return false;
    }
  }
  return true;
}

double ms_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

bool run_level(const Plan& plan, int level, const LevelOps& ops,
               cudaStream_t stream, std::string& error) {
  if (level < 0 || level >= plan.levels()) {
    error = "level out of range";
    return false;
  }
  const LevelBatch batch = plan.level_batch(level);
  if (batch.job_begin == batch.job_end) {
    return true;
  }
  if (ops.predict != nullptr &&
      !check_stage(ops.predict(ops.user, plan, batch, stream, error),
                   "predict", level, error)) {
    return false;
  }
  if (ops.transform != nullptr &&
      !check_stage(ops.transform(ops.user, plan, batch, stream, error),
                   "transform", level, error)) {
    return false;
  }
  if (ops.reconstruct != nullptr &&
      !check_stage(ops.reconstruct(ops.user, plan, batch, stream, error),
                   "reconstruct", level, error)) {
    return false;
  }
  return true;
}

bool run_levels_eager(const Plan& plan, const LevelOps& ops,
                      cudaStream_t stream, RunStats* stats,
                      std::string& error) {
  const auto start = std::chrono::steady_clock::now();
  if (!run_sequence(plan, ops, stream, error)) {
    return false;
  }
  if (stats != nullptr) {
    stats->used_graph = false;
    stats->levels = plan.levels();
    stats->gpu_nodes = (ops.launch_count != nullptr)
                           ? ops.launch_count(ops.user)
                           : static_cast<int>(
                                 plan.counters().scheduled_kernel_launches);
    stats->host_launches = stats->gpu_nodes;
    stats->submit_ms = ms_since(start);
  }
  return true;
}

struct LevelGraph::Impl {
  bool ready = false;
  bool graph_available = false;
  std::string graph_error;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  int nodes = 0;
  int levels = 0;
  const Plan* plan = nullptr;
  cudaStream_t stream = nullptr;

  void destroy() {
    if (exec != nullptr) {
      cudaGraphExecDestroy(exec);
      exec = nullptr;
    }
    if (graph != nullptr) {
      cudaGraphDestroy(graph);
      graph = nullptr;
    }
    nodes = 0;
    graph_available = false;
    ready = false;
  }
};

LevelGraph::LevelGraph() : impl_(std::make_unique<Impl>()) {}
LevelGraph::~LevelGraph() { release(); }

void LevelGraph::release() {
  if (impl_) {
    impl_->destroy();
  }
}

bool LevelGraph::prepared() const { return impl_ && impl_->ready; }
bool LevelGraph::graph_available() const {
  return impl_ && impl_->graph_available;
}
const std::string& LevelGraph::graph_error() const {
  static const std::string kEmpty;
  return impl_ ? impl_->graph_error : kEmpty;
}
int LevelGraph::graph_nodes() const { return impl_ ? impl_->nodes : 0; }

bool LevelGraph::prepare(const Plan& plan, const LevelOps& ops,
                         cudaStream_t stream, std::string& error) {
  if (impl_->ready) {
    release();
  }
  impl_->levels = plan.levels();
  impl_->plan = &plan;
  impl_->stream = stream;
  impl_->graph_error.clear();
  if (plan.empty()) {
    error = "cannot prepare a graph for an empty plan";
    return false;
  }

  // Module warm-up: the first launch of a kernel can trigger lazy loading and
  // one-off driver work, which must not happen inside capture.
  {
    std::string warm_error;
    if (!run_sequence(plan, ops, stream, warm_error)) {
      error = "warm-up: " + warm_error;
      return false;
    }
    if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                    error)) {
      return false;
    }
  }

  const cudaError_t begin =
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (begin != cudaSuccess) {
    impl_->graph_error = std::string("cudaStreamBeginCapture: ") +
                         cudaGetErrorString(begin);
    impl_->ready = true;  // eager fallback is still usable
    return true;
  }
  std::string capture_error;
  const bool captured_ok = run_sequence(plan, ops, stream, capture_error);
  cudaGraph_t captured = nullptr;
  const cudaError_t end = cudaStreamEndCapture(stream, &captured);
  if (!captured_ok) {
    // A callback failed inside capture.  CUDA invalidates the capture stream in
    // this situation, so an eager fallback on it is not trustworthy: report
    // the failure and let the caller rebuild the stream.
    if (end == cudaSuccess && captured != nullptr) {
      cudaGraphDestroy(captured);
    }
    cudaGetLastError();
    impl_->ready = false;
    error = "capture sequence failed: " + capture_error;
    return false;
  }
  if (end != cudaSuccess || captured == nullptr) {
    impl_->ready = false;
    error = std::string("cudaStreamEndCapture: ") + cudaGetErrorString(end);
    return false;
  }

  cudaGraphExec_t exec = nullptr;
  const cudaError_t instantiate = cudaGraphInstantiate(&exec, captured, 0);
  if (instantiate != cudaSuccess) {
    impl_->graph_error = std::string("cudaGraphInstantiate: ") +
                         cudaGetErrorString(instantiate);
    cudaGraphDestroy(captured);
    impl_->ready = true;
    return true;
  }

  std::size_t nodes = 0;
  if (cudaGraphGetNodes(captured, nullptr, &nodes) != cudaSuccess) {
    nodes = 0;
  }
  impl_->graph = captured;
  impl_->exec = exec;
  impl_->nodes = static_cast<int>(nodes);
  impl_->graph_available = true;
  impl_->graph_error.clear();
  impl_->ready = true;

  // Upload the graph once so the first measured run does not pay for it.
  {
    std::string replay_error;
    if (!check_cuda(cudaGraphLaunch(exec, stream), "cudaGraphLaunch(warm)",
                    replay_error) ||
        !check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                    replay_error)) {
      impl_->graph_error = "warm replay: " + replay_error;
      impl_->graph_available = false;
    }
  }
  return true;
}

bool LevelGraph::run(const LevelOps& ops, const RunOptions& options,
                     RunStats* stats, std::string& error) {
  if (!impl_ || !impl_->ready) {
    error = "LevelGraph::run called before prepare";
    return false;
  }
  if (impl_->plan == nullptr || impl_->stream == nullptr) {
    error = "LevelGraph has no plan/stream";
    return false;
  }
  if (!options.use_cuda_graph || !impl_->graph_available) {
    return run_levels_eager(*impl_->plan, ops, impl_->stream, stats, error);
  }
  const auto start = std::chrono::steady_clock::now();
  const cudaError_t launch = cudaGraphLaunch(impl_->exec, impl_->stream);
  if (launch != cudaSuccess) {
    error = std::string("cudaGraphLaunch: ") + cudaGetErrorString(launch);
    return false;
  }
  if (stats != nullptr) {
    stats->used_graph = true;
    stats->levels = impl_->levels;
    stats->gpu_nodes = impl_->nodes;
    stats->host_launches = 1;
    stats->submit_ms = ms_since(start);
  }
  return true;
}

}  // namespace av2enc::cuda::still::sched
