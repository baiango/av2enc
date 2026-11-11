// GPU verification and benchmark for the M1 scheduler adapter.
//
// For each input Y4M frame:
//   1. run the CPU M1 encoder with AV2ENC_M1_TRACE enabled (deblock off, so the
//      reported reconstruction is the pre-deblock reference) and parse the
//      per-leaf decisions;
//   2. build the scheduler plan from those leaves;
//   3. run the reference GPU stage executor through the plan once eagerly and
//      once from the captured CUDA graph;
//   4. require bit-exact reconstructed Y/U/V against the CPU result.
//
// It then builds synthetic all-8x8 1920x1080 plans (DC and all-CfL) and reports
// the measured block-stage time as the 1080p projection.
//
// File name ends in _bench.cpp so the root CMake M1 glob skips it.

#include "cuda/m1/sched/ref_exec.h"
#include "cuda/m1/sched/sched.h"
#include "cuda/m1/sched/sched_exec.h"

#include "cuda/context.h"
#include "encoder/m1/m1_still.h"
#include "encoder/partition/partition.h"
#include "io/y4m.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace partition = av2enc::partition;

using av2enc::cuda::m1::sched::BlockPlan;
using av2enc::cuda::m1::sched::FramePlan;
using av2enc::cuda::m1::sched::LevelGraph;
using av2enc::cuda::m1::sched::LevelOps;
using av2enc::cuda::m1::sched::Plan;
using av2enc::cuda::m1::sched::RefExecOptions;
using av2enc::cuda::m1::sched::RefM1Executor;
using av2enc::cuda::m1::sched::RunOptions;
using av2enc::cuda::m1::sched::RunStats;

int failures = 0;

void fail(const std::string& what) {
  std::printf("  FAIL  %s\n", what.c_str());
  ++failures;
}

void ok(const std::string& what) {
  std::printf("  ok    %s\n", what.c_str());
}

struct TraceLeaf {
  int mi_row = 0;
  int mi_col = 0;
  std::string size;
  int y_mode = 0;
  int y_angle = 0;
  int uv_mode = 0;
  int uv_angle = 0;
  int cfl = 0;
  int tx_w = 0;
  int tx_h = 0;
};

bool parse_trace(const std::string& text, std::vector<TraceLeaf>& leaves,
                 std::string& error) {
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("m1 block ", 0) != 0) {
      continue;
    }
    TraceLeaf leaf;
    char size[32] = {};
    const int n = std::sscanf(
        line.c_str(),
        "m1 block r=%d c=%d size=%31s y=%d/%d uv=%d/%d cfl=%d tx=%dx%d",
        &leaf.mi_row, &leaf.mi_col, size, &leaf.y_mode, &leaf.y_angle,
        &leaf.uv_mode, &leaf.uv_angle, &leaf.cfl, &leaf.tx_w, &leaf.tx_h);
    if (n != 10) {
      error = "could not parse trace line: " + line;
      return false;
    }
    leaf.size = size;
    leaves.push_back(leaf);
  }
  if (leaves.empty()) {
    error = "trace contained no 'm1 block' lines";
    return false;
  }
  return true;
}

partition::BlockSize size_from_name(const std::string& name) {
  for (int i = 0; i < partition::kBlockSizeCount; ++i) {
    const auto size = static_cast<partition::BlockSize>(i);
    if (name == partition::block_size_name(size)) {
      return size;
    }
  }
  return partition::BlockSize::kInvalid;
}

FramePlan make_frame_plan(int width, int height) {
  FramePlan f;
  f.width = width;
  f.height = height;
  f.chroma_width = (width + 1) / 2;
  f.chroma_height = (height + 1) / 2;
  f.mi_cols = 2 * ((width + 7) >> 3);
  f.mi_rows = 2 * ((height + 7) >> 3);
  const int coded_w = f.mi_cols * 4;
  const int coded_h = f.mi_rows * 4;
  f.padded_luma_width = (coded_w + 63) & ~63;
  f.padded_luma_height = (coded_h + 63) & ~63;
  f.padded_chroma_width = f.padded_luma_width / 2;
  f.padded_chroma_height = f.padded_luma_height / 2;
  return f;
}

bool build_blocks_from_trace(const FramePlan& frame,
                             const std::vector<TraceLeaf>& trace,
                             std::vector<BlockPlan>& blocks,
                             std::string& error) {
  partition::FrameGeometry geometry;
  geometry.width = frame.width;
  geometry.height = frame.height;
  geometry.sb_size = 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;
  blocks.clear();
  blocks.reserve(trace.size());
  for (const TraceLeaf& t : trace) {
    const partition::BlockSize size = size_from_name(t.size);
    if (size == partition::BlockSize::kInvalid) {
      error = "unknown block size '" + t.size + "' in trace";
      return false;
    }
    const partition::BlockSize chroma_size =
        partition::chroma_plane_residual_size(geometry, size);
    BlockPlan b;
    b.mi_row = t.mi_row;
    b.mi_col = t.mi_col;
    b.luma_w = partition::block_width(size);
    b.luma_h = partition::block_height(size);
    b.has_chroma = chroma_size != partition::BlockSize::kInvalid;
    if (b.has_chroma) {
      b.chroma_w = partition::block_width(chroma_size);
      b.chroma_h = partition::block_height(chroma_size);
    }
    b.luma_mode = t.y_mode;
    b.luma_angle = t.y_angle;
    b.chroma_mode = t.cfl ? 0 : t.uv_mode;
    b.chroma_angle = t.cfl ? 0 : t.uv_angle;
    b.is_cfl = t.cfl;
    if (b.luma_w != t.tx_w || b.luma_h != t.tx_h) {
      error = "trace TX size does not match the luma leaf size";
      return false;
    }
    blocks.push_back(b);
  }
  return true;
}

bool read_y4m(const std::string& path, std::vector<std::uint8_t>& y,
              std::vector<std::uint8_t>& u, std::vector<std::uint8_t>& v,
              int& width, int& height, std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) {
    return false;
  }
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult result = reader.read_frame(frame, error);
  if (result != av2enc::Y4mReadResult::kFrame) {
    if (error.empty()) {
      error = "Y4M contains no frame";
    }
    return false;
  }
  width = static_cast<int>(frame.width());
  height = static_cast<int>(frame.height());
  y.assign(frame.y().begin(), frame.y().end());
  u.assign(frame.u().begin(), frame.u().end());
  v.assign(frame.v().begin(), frame.v().end());
  return true;
}

std::int64_t compare_plane(const std::vector<std::uint8_t>& got,
                           const std::vector<std::uint8_t>& want, int width,
                           int height, int got_stride, const char* name,
                           int* max_diff, int* first_x, int* first_y) {
  (void)name;
  std::int64_t mismatches = 0;
  *max_diff = 0;
  *first_x = -1;
  *first_y = -1;
  for (int yy = 0; yy < height; ++yy) {
    for (int xx = 0; xx < width; ++xx) {
      const int a = got[static_cast<std::size_t>(yy) *
                           static_cast<std::size_t>(got_stride) +
                       static_cast<std::size_t>(xx)];
      const int b = want[static_cast<std::size_t>(yy) *
                             static_cast<std::size_t>(width) +
                         static_cast<std::size_t>(xx)];
      if (a != b) {
        if (*first_x < 0) {
          *first_x = xx;
          *first_y = yy;
        }
        const int d = a > b ? a - b : b - a;
        *max_diff = std::max(*max_diff, d);
        ++mismatches;
      }
    }
  }
  return mismatches;
}

struct Timing {
  double upload_ms = 0;
  double kernel_ms = 0;
  double download_ms = 0;
  double submit_ms = 0;
  bool used_graph = false;
  int host_launches = 0;
  int gpu_nodes = 0;
};

class Bench {
 public:
  bool init(std::string& error) {
    const av2enc::cuda::Status status = ctx_.init(0, &error);
    if (!av2enc::cuda::is_ok(status)) {
      error = std::string("context init: ") +
              av2enc::cuda::status_name(status) + " " + error;
      return false;
    }
    const auto create = [&](cudaEvent_t* ev) {
      return cudaEventCreate(ev) == cudaSuccess;
    };
    if (!create(&up0_) || !create(&up1_) || !create(&k1_) ||
        !create(&dl0_)) {
      error = "cudaEventCreate failed";
      return false;
    }
    return true;
  }

  ~Bench() {
    cudaEventDestroy(up0_);
    cudaEventDestroy(up1_);
    cudaEventDestroy(k1_);
    cudaEventDestroy(dl0_);
    ctx_.shutdown();
  }

  av2enc::cuda::Context& context() { return ctx_; }

  Timing run(RefM1Executor& executor, LevelGraph& graph, const LevelOps& ops,
             const RunOptions& options, const std::uint8_t* y,
             const std::uint8_t* u, const std::uint8_t* v, std::uint8_t* ry,
             std::uint8_t* ru, std::uint8_t* rv, std::string& error) {
    Timing t;
    cudaStream_t stream = ctx_.stream();
    executor.reset_launch_count();
    cudaEventRecord(up0_, stream);
    if (!executor.zero_recon(stream, error) ||
        !executor.upload_source(y, u, v, stream, error)) {
      return t;
    }
    cudaEventRecord(up1_, stream);
    RunStats stats;
    if (!graph.run(ops, options, &stats, error)) {
      return t;
    }
    cudaEventRecord(k1_, stream);
    if (!executor.download_recon(ry, ru, rv, stream, error)) {
      return t;
    }
    cudaEventRecord(dl0_, stream);
    if (!av2enc::cuda::is_ok(ctx_.synchronize())) {
      error = "context synchronize failed";
      return t;
    }
    float ms = 0;
    cudaEventElapsedTime(&ms, up0_, up1_);
    t.upload_ms = ms;
    cudaEventElapsedTime(&ms, up1_, k1_);
    t.kernel_ms = ms;
    cudaEventElapsedTime(&ms, k1_, dl0_);
    t.download_ms = ms;
    t.submit_ms = stats.submit_ms;
    t.used_graph = stats.used_graph;
    t.host_launches = stats.host_launches;
    t.gpu_nodes = stats.gpu_nodes;
    if (t.gpu_nodes == 0 && !stats.used_graph) {
      t.gpu_nodes = executor.launch_count();
    }
    return t;
  }

 private:
  av2enc::cuda::Context ctx_;
  cudaEvent_t up0_ = nullptr;
  cudaEvent_t up1_ = nullptr;
  cudaEvent_t k1_ = nullptr;
  cudaEvent_t dl0_ = nullptr;
};

void print_plan(const Plan& plan) {
  const auto& c = plan.counters();
  std::printf(
      "  plan: blocks=%d jobs=%d (Y %d, C %d, CfL %d) levels=%d classes=%d "
      "groups=%d edges=%lld dep_viol=%lld syntax_viol=%lld coeffs=%lld\n",
      c.coding_blocks, c.plane_jobs, c.luma_jobs, c.chroma_jobs, c.cfl_jobs,
      c.levels, c.transform_classes, c.groups, c.dependency_edges,
      c.dependency_violations, c.syntax_order_violations,
      static_cast<long long>(plan.total_coeffs()));
  std::printf("  launches: per-block estimate=%lld batched estimate=%lld\n",
              c.naive_block_calls, c.scheduled_kernel_launches);
  std::vector<int> class_jobs(static_cast<std::size_t>(c.transform_classes), 0);
  for (int i = 0; i < c.plane_jobs; ++i) {
    ++class_jobs[static_cast<std::size_t>(plan.job(i).cls)];
  }
  std::printf("  classes:");
  for (int cls = 0; cls < c.transform_classes; ++cls) {
    int w = 0;
    int h = 0;
    for (int i = 0; i < c.plane_jobs; ++i) {
      if (plan.job(i).cls == cls) {
        w = plan.job(i).w;
        h = plan.job(i).h;
        break;
      }
    }
    std::printf(" %dx%d:%d", w, h, class_jobs[static_cast<std::size_t>(cls)]);
  }
  std::printf("\n");
}

struct CaseResult {
  int width = 0;
  int height = 0;
  int leaves = 0;
  int jobs = 0;
  int levels = 0;
  int graph_nodes = 0;
  int eager_launches = 0;
  double graph_kernel_ms = 0;
  double eager_kernel_ms = 0;
  double graph_submit_ms = 0;
  double eager_submit_ms = 0;
  bool exact = false;
};

bool run_real_case(Bench& bench, const std::string& path, int iterations,
                   bool full_intra, CaseResult* out) {
  std::printf("\n== CPU M1 vs scheduled GPU: %s ==\n", path.c_str());
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  int width = 0;
  int height = 0;
  std::string error;
  if (!read_y4m(path, y, u, v, width, height, error)) {
    fail("read Y4M: " + error);
    return false;
  }

  av2enc::m1::StillPicture picture;
  picture.width = static_cast<std::uint32_t>(width);
  picture.height = static_cast<std::uint32_t>(height);
  picture.y = y;
  picture.u = u;
  picture.v = v;
  av2enc::m1::M1Options options;
  options.enable_deblock = false;  // compare the pre-deblock reconstruction
  options.luma_dc_only = !full_intra;

  av2enc::m1::M1Result cpu;
  {
    std::ostringstream trace;
    std::streambuf* old = std::cerr.rdbuf(trace.rdbuf());
    setenv("AV2ENC_M1_TRACE", "1", 1);
    const bool encoded =
        av2enc::m1::encode_m1_still_picture(picture, options, cpu, error);
    unsetenv("AV2ENC_M1_TRACE");
    std::cerr.rdbuf(old);
    if (!encoded) {
      fail("CPU M1 encode: " + error);
      return false;
    }
    std::vector<TraceLeaf> trace_leaves;
    if (!parse_trace(trace.str(), trace_leaves, error)) {
      fail("trace: " + error);
      return false;
    }
    if (static_cast<int>(trace_leaves.size()) != cpu.leaves) {
      fail("trace leaf count does not match M1Result.leaves");
      return false;
    }
    std::printf("  cpu: %dx%d leaves=%d frame=%u B deblock=off\n", width,
                height, cpu.leaves, cpu.frame_bytes);

    FramePlan frame = make_frame_plan(width, height);
    std::vector<BlockPlan> blocks;
    if (!build_blocks_from_trace(frame, trace_leaves, blocks, error)) {
      fail("blocks: " + error);
      return false;
    }
    frame.blocks = blocks.data();
    frame.block_count = static_cast<int>(blocks.size());
    Plan plan;
    if (!plan.build(frame, error)) {
      fail("plan: " + error);
      return false;
    }
    print_plan(plan);

    RefM1Executor executor;
    RefExecOptions exec_options;
    exec_options.enable_edge_filter = options.enable_intra_edge_filter;
    if (!executor.prepare(plan, frame, exec_options, bench.context().stream(),
                          error)) {
      fail("executor prepare: " + error);
      return false;
    }
    const LevelOps ops = executor.ops();
    LevelGraph graph;
    if (!graph.prepare(plan, ops, bench.context().stream(), error)) {
      fail("graph prepare: " + error);
      return false;
    }
    if (!graph.graph_available()) {
      std::printf("  graph unavailable (%s); using eager fallback\n",
                  graph.graph_error().c_str());
    } else {
      std::printf("  graph: %d nodes, workspace %.1f MiB\n", graph.graph_nodes(),
                  static_cast<double>(executor.workspace_bytes()) /
                      (1024.0 * 1024.0));
    }

    const int luma_stride = executor.recon_stride(0);
    const int luma_height = executor.recon_height(0);
    const int chroma_stride = executor.recon_stride(1);
    const int chroma_height = executor.recon_height(1);
    std::vector<std::uint8_t> hy(
        static_cast<std::size_t>(luma_stride) * luma_height);
    std::vector<std::uint8_t> hu(
        static_cast<std::size_t>(chroma_stride) * chroma_height);
    std::vector<std::uint8_t> hv(hu.size());
    std::vector<std::uint8_t> eager_y(hy.size());
    std::vector<std::uint8_t> eager_u(hu.size());
    std::vector<std::uint8_t> eager_v(hu.size());

    RunOptions graph_options;
    Timing last_graph;
    double best_graph_ms = 1.0e30;
    for (int it = 0; it < iterations; ++it) {
      const Timing t = bench.run(executor, graph, ops, graph_options, y.data(),
                                 u.data(), v.data(), hy.data(), hu.data(),
                                 hv.data(), error);
      if (!error.empty()) {
        fail("graph run: " + error);
        return false;
      }
      if (t.kernel_ms < best_graph_ms) {
        best_graph_ms = t.kernel_ms;
        last_graph = t;
      }
    }
    int maxd = 0;
    int fx = 0;
    int fy = 0;
    const std::int64_t my =
        compare_plane(hy, cpu.recon_y, width, height, luma_stride, "Y", &maxd,
                      &fx, &fy);
    const std::int64_t mu =
        compare_plane(hu, cpu.recon_u, width / 2, height / 2, chroma_stride,
                      "U", &maxd, &fx, &fy);
    const std::int64_t mv =
        compare_plane(hv, cpu.recon_v, width / 2, height / 2, chroma_stride,
                      "V", &maxd, &fx, &fy);
    if (my == 0 && mu == 0 && mv == 0) {
      ok("CUDA-graph recon is bit-exact against CPU M1 (Y/U/V)");
    } else {
      fail("graph recon mismatch Y=" + std::to_string(my) +
           " U=" + std::to_string(mu) + " V=" + std::to_string(mv) +
           " first=(" + std::to_string(fx) + "," + std::to_string(fy) + ")");
    }

    // Eager sequence (still batched, one host call per kernel).
    RunOptions eager_options;
    eager_options.use_cuda_graph = false;
    Timing last_eager;
    double best_eager_ms = 1.0e30;
    for (int it = 0; it < iterations; ++it) {
      const Timing t =
          bench.run(executor, graph, ops, eager_options, y.data(), u.data(),
                    v.data(), eager_y.data(), eager_u.data(), eager_v.data(),
                    error);
      if (!error.empty()) {
        fail("eager run: " + error);
        return false;
      }
      if (t.kernel_ms < best_eager_ms) {
        best_eager_ms = t.kernel_ms;
        last_eager = t;
      }
    }
    if (!graph.graph_available()) {
      last_graph = last_eager;  // graph path did not exist
    }
    const std::int64_t ey =
        compare_plane(eager_y, cpu.recon_y, width, height, luma_stride, "Y",
                      &maxd, &fx, &fy);
    const std::int64_t eu =
        compare_plane(eager_u, cpu.recon_u, width / 2, height / 2,
                      chroma_stride, "U", &maxd, &fx, &fy);
    const std::int64_t ev =
        compare_plane(eager_v, cpu.recon_v, width / 2, height / 2,
                      chroma_stride, "V", &maxd, &fx, &fy);
    if (ey == 0 && eu == 0 && ev == 0) {
      ok("eager batched recon is bit-exact against CPU M1 (Y/U/V)");
    } else {
      fail("eager recon mismatch Y=" + std::to_string(ey) +
           " U=" + std::to_string(eu) + " V=" + std::to_string(ev));
    }

    out->width = width;
    out->height = height;
    out->leaves = cpu.leaves;
    out->jobs = plan.counters().plane_jobs;
    out->levels = plan.levels();
    out->graph_nodes = graph.graph_nodes();
    out->eager_launches = last_eager.gpu_nodes;
    out->graph_kernel_ms = last_graph.kernel_ms;
    out->eager_kernel_ms = last_eager.kernel_ms;
    out->graph_submit_ms = last_graph.submit_ms;
    out->eager_submit_ms = last_eager.submit_ms;
    out->exact = (my == 0 && mu == 0 && mv == 0 && ey == 0 && eu == 0 &&
                  ev == 0);
    std::printf(
        "  graph: levels=%d nodes=%d host=%d best kernel=%.3f ms submit=%.3f ms\n",
        last_graph.used_graph ? plan.levels() : 0, last_graph.gpu_nodes,
        last_graph.host_launches, last_graph.kernel_ms,
        last_graph.submit_ms);
    std::printf(
        "  eager: launches=%d best kernel=%.3f ms submit=%.3f ms (same geometry)\n",
        last_eager.gpu_nodes, last_eager.kernel_ms, last_eager.submit_ms);
  }
  return true;
}

// Synthetic all-8x8 plan: worst-case job/level count for a picture size.
std::vector<BlockPlan> synthetic_blocks(const FramePlan& frame, bool all_cfl) {
  std::vector<BlockPlan> out;
  int seed = 0;
  for (int sbr = 0; sbr < frame.mi_rows; sbr += 16) {
    for (int sbc = 0; sbc < frame.mi_cols; sbc += 16) {
      const int r_end = std::min(sbr + 16, frame.mi_rows);
      const int c_end = std::min(sbc + 16, frame.mi_cols);
      for (int r = sbr; r < r_end; r += 2) {
        for (int c = sbc; c < c_end; c += 2) {
          BlockPlan b;
          b.mi_row = r;
          b.mi_col = c;
          b.luma_w = 8;
          b.luma_h = 8;
          b.has_chroma = true;
          b.chroma_w = 4;
          b.chroma_h = 4;
          b.luma_mode = 0;
          b.luma_angle = 0;
          b.chroma_mode = 0;
          b.chroma_angle = 0;
          b.is_cfl = all_cfl ? 1 : ((seed++ % 2) == 0 ? 1 : 0);
          out.push_back(b);
        }
      }
    }
  }
  return out;
}

bool run_synthetic_case(Bench& bench, int width, int height, bool all_cfl,
                        int iterations, CaseResult* out) {
  std::printf("\n== synthetic %dx%d all 8x8 (%s) ==\n", width, height,
              all_cfl ? "all CfL" : "mixed CfL/DC");
  FramePlan frame = make_frame_plan(width, height);
  std::vector<BlockPlan> blocks = synthetic_blocks(frame, all_cfl);
  frame.blocks = blocks.data();
  frame.block_count = static_cast<int>(blocks.size());
  Plan plan;
  std::string error;
  if (!plan.build(frame, error)) {
    fail("plan: " + error);
    return false;
  }
  print_plan(plan);

  RefM1Executor executor;
  RefExecOptions exec_options;
  if (!executor.prepare(plan, frame, exec_options, bench.context().stream(),
                        error)) {
    fail("executor prepare: " + error);
    return false;
  }
  const LevelOps ops = executor.ops();
  LevelGraph graph;
  if (!graph.prepare(plan, ops, bench.context().stream(), error)) {
    fail("graph prepare: " + error);
    return false;
  }
  std::printf("  graph: %d nodes, workspace %.1f MiB\n", graph.graph_nodes(),
              static_cast<double>(executor.workspace_bytes()) /
                  (1024.0 * 1024.0));

  std::vector<std::uint8_t> y(static_cast<std::size_t>(width) * height);
  std::vector<std::uint8_t> u(static_cast<std::size_t>(width / 2) * (height / 2));
  std::vector<std::uint8_t> v(u.size());
  for (int yy = 0; yy < height; ++yy) {
    for (int xx = 0; xx < width; ++xx) {
      y[static_cast<std::size_t>(yy) * width + xx] =
          static_cast<std::uint8_t>((xx * 3 + yy * 5) & 0xff);
    }
  }
  for (int yy = 0; yy < height / 2; ++yy) {
    for (int xx = 0; xx < width / 2; ++xx) {
      u[static_cast<std::size_t>(yy) * (width / 2) + xx] =
          static_cast<std::uint8_t>((xx * 7 + yy * 11) & 0xff);
      v[static_cast<std::size_t>(yy) * (width / 2) + xx] =
          static_cast<std::uint8_t>((xx * 13 + yy * 3) & 0xff);
    }
  }
  const int luma_stride = executor.recon_stride(0);
  const int luma_height = executor.recon_height(0);
  const int chroma_stride = executor.recon_stride(1);
  const int chroma_height = executor.recon_height(1);
  std::vector<std::uint8_t> hy(
      static_cast<std::size_t>(luma_stride) * luma_height);
  std::vector<std::uint8_t> hu(
      static_cast<std::size_t>(chroma_stride) * chroma_height);
  std::vector<std::uint8_t> hv(hu.size());
  RunOptions graph_options;
  Timing last_graph;
  double best_graph_ms = 1.0e30;
  for (int it = 0; it < iterations; ++it) {
    const Timing t = bench.run(executor, graph, ops, graph_options, y.data(),
                               u.data(), v.data(), hy.data(), hu.data(),
                               hv.data(), error);
    if (!error.empty()) {
      fail("graph run: " + error);
      return false;
    }
    if (t.kernel_ms < best_graph_ms) {
      best_graph_ms = t.kernel_ms;
      last_graph = t;
    }
  }
  RunOptions eager_options;
  eager_options.use_cuda_graph = false;
  Timing last_eager;
  double best_eager_ms = 1.0e30;
  for (int it = 0; it < iterations; ++it) {
    const Timing t =
        bench.run(executor, graph, ops, eager_options, y.data(), u.data(),
                  v.data(), hy.data(), hu.data(), hv.data(), error);
    if (!error.empty()) {
      fail("eager run: " + error);
      return false;
    }
    if (t.kernel_ms < best_eager_ms) {
      best_eager_ms = t.kernel_ms;
      last_eager = t;
    }
  }
  std::printf(
      "  graph: levels=%d nodes=%d best kernel=%.3f ms submit=%.3f ms\n",
      plan.levels(), last_graph.gpu_nodes, last_graph.kernel_ms,
      last_graph.submit_ms);
  std::printf("  eager: launches=%d best kernel=%.3f ms submit=%.3f ms\n",
              last_eager.gpu_nodes, last_eager.kernel_ms,
              last_eager.submit_ms);
  if (out != nullptr) {
    out->width = width;
    out->height = height;
    out->leaves = static_cast<int>(blocks.size());
    out->jobs = plan.counters().plane_jobs;
    out->levels = plan.levels();
    out->graph_nodes = last_graph.gpu_nodes;
    out->eager_launches = last_eager.gpu_nodes;
    out->graph_kernel_ms = last_graph.kernel_ms;
    out->eager_kernel_ms = last_eager.kernel_ms;
    out->graph_submit_ms = last_graph.submit_ms;
    out->eager_submit_ms = last_eager.submit_ms;
    out->exact = true;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> inputs;
  int iterations = 5;
  bool projection = true;
  bool quick = false;
  bool full_intra = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--y4m" && i + 1 < argc) {
      inputs.push_back(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      iterations = std::atoi(argv[++i]);
    } else if (arg == "--no-projection") {
      projection = false;
    } else if (arg == "--quick") {
      quick = true;
    } else if (arg == "--full-intra") {
      full_intra = true;
    } else if (arg == "-h" || arg == "--help") {
      std::printf(
          "usage: sched_bench [--y4m FILE]... [--iterations N] "
          "[--no-projection] [--quick] [--full-intra]\n");
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }
  if (inputs.empty()) {
    inputs.push_back("tests/data/m0_64x64_1f.y4m");
    inputs.push_back("tests/data/m0_640x360_1f.y4m");
  }
  if (iterations < 1) {
    iterations = 1;
  }

  // Decision-affecting environment knobs must be clear for both paths.
  static const char* kEnv[] = {
      "AV2ENC_M1_FORCE_Y",   "AV2ENC_M1_FORCE_Y2", "AV2ENC_M1_BASE_ONLY",
      "AV2ENC_M1_NO_DIR",    "AV2ENC_M1_NO_EDGE",  "AV2ENC_M1_UV_DC",
      "AV2ENC_M1_LUMA_DC",   "AV2ENC_M1_SQUARE_ONLY", "AV2ENC_M1_LEAF32",
      "AV2ENC_M1_ZERO_DC",   "AV2ENC_M1_YMODE_CTX0", "AV2ENC_M1_YMODE_CTX2",
      "AV2ENC_M1_DCSIGN_CTX"};
  for (const char* name : kEnv) {
    unsetenv(name);
  }

  Bench bench;
  std::string error;
  if (!bench.init(error)) {
    std::printf("SKIP: %s\n", error.c_str());
    return 77;
  }
  std::printf("device: %s (cc %d.%d, %d SMs)\n",
              bench.context().device().name, bench.context().device().cc_major,
              bench.context().device().cc_minor,
              bench.context().device().sm_count);

  std::vector<CaseResult> real_cases;
  for (const std::string& input : inputs) {
    CaseResult result;
    if (!run_real_case(bench, input, iterations, full_intra, &result)) {
      continue;
    }
    real_cases.push_back(result);
  }

  if (projection) {
    const int synth_iters = quick ? 2 : std::max(2, iterations);
    run_synthetic_case(bench, 640, 360, false, synth_iters, nullptr);
    run_synthetic_case(bench, 1920, 1080, false, synth_iters, nullptr);
    run_synthetic_case(bench, 1920, 1080, true, synth_iters, nullptr);
  }

  std::printf("\n== summary ==\n");
  std::printf(
      "%-12s %6s %6s %6s %8s %10s %10s %10s\n", "case", "leaves", "jobs",
      "levels", "nodes", "graph ms", "eager ms", "exact");
  for (const CaseResult& c : real_cases) {
    char name[32];
    std::snprintf(name, sizeof(name), "%dx%d", c.width, c.height);
    std::printf("%-12s %6d %6d %6d %8d %10.3f %10.3f %10s\n", name, c.leaves,
                c.jobs, c.levels, c.graph_nodes, c.graph_kernel_ms,
                c.eager_kernel_ms, c.exact ? "yes" : "NO");
  }

  if (failures == 0) {
    std::printf("\nPASS: M1 scheduler adapter\n");
    return 0;
  }
  std::printf("\nFAIL: %d check(s) failed\n", failures);
  return 1;
}
