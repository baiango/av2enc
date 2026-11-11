// Dependency-aware wavefront scheduler for the CUDA M1 block pipeline.
//
// M1 codes a still with a quadtree of leaves down to 8x8.  Every leaf has one
// luma block and (for 4:2:0, with the M1 minimum size) one U and one V block.
// The CPU path walks the leaves in syntax order and runs prediction, forward
// transform, quantisation, inverse transform and reconstruction per block, so
// the block pipeline is a serial chain of tiny GPU launches.
//
// This module computes the real data-dependency graph of that pipeline and
// turns it into wavefront levels:
//
//   * a luma block reads the reconstructed left column, above row, the
//     above-left corner and the directional-prediction above-right / below-left
//     extensions (num4AboveRight / num4BelowLeft, §7.13.2.1);
//   * a chroma block reads the same neighbour lines in its own plane and, for
//     CfL (§7.13.5) and the UV mode decision, the reconstructed luma of the
//     co-located leaf (sample_luma plus the filtered above/left lines used by
//     the derived-alpha correlation);
//   * a block's level is 1 + max(level of its prerequisites), so every edge
//     crosses a level boundary and all blocks in a level are independent.
//
// Availability of the directional extensions is not arbitrary: it is exactly
// what `clear_block_decoded_flags` / `count_top_right` / `count_bottom_left`
// in src/encoder/m1/m1_still.cpp see at the CPU's coding time.  The plan
// replays that bookkeeping on the coverage map (4-sample cells for every
// plane, matching the encoder's 4x4 / 4-chroma-sample MI accounting) so the
// recorded edges are the reads the kernels will actually perform.
//
// The header is deliberately free of CUDA types so the plan can be built and
// tested on a host without a GPU.  Execution (run_level, CUDA graph capture and
// replay) lives in sched_exec.h.
//
// M1 geometry contract: 8-bit 4:2:0, one tile, one keyframe, variable TX where
// the transform block is the leaf size.  Each leaf may carry a different coded
// size, so the plan records the TX class per job rather than assuming a single
// transform size.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::cuda::m1::sched {

// Plane indices used everywhere in this module.
enum PlaneIndex : int {
  kPlaneY = 0,
  kPlaneU = 1,
  kPlaneV = 2,
};

// ---------------------------------------------------------------------------
// Input: one planned frame.
// ---------------------------------------------------------------------------

// One leaf of the M1 partition tree, in syntax order.  This mirrors the
// orchestrator's BlockPlan (docs/m1-gpu.h); the decision fields are optional
// and only used to classify jobs (CfL, modes) and to drive the verification
// executor.  Geometry alone is enough to build the dependency graph.
struct BlockPlan {
  int mi_row = 0;  // luma position in 4x4 units
  int mi_col = 0;
  int luma_w = 0;  // luma block width / height in samples (>= 4, 4-aligned)
  int luma_h = 0;
  int chroma_w = 0;  // 4:2:0 chroma residual size; ignored if !has_chroma
  int chroma_h = 0;
  bool has_chroma = true;  // §5.20.5 hasChroma for this leaf

  // Decided syntax (optional).  Modes are AV2 YMode/UVMode values, which are
  // numerically identical to intra::Mode (src/encoder/intra_modes).
  int luma_mode = 0;  // 0 = DC
  int luma_angle = 0;
  int chroma_mode = 0;  // 0 = DC
  int chroma_angle = 0;
  int is_cfl = 0;  // chroma uses CfL; chroma_mode is then DC for the base pred
};

// Frame geometry for one planned still.
//
// `padded_*` describe the reconstruction planes the CPU M1 allocates:
// luma_stride = (mi_cols * 4 + 63) & ~63, chroma = luma / 2.  Blocks may extend
// past the visible picture up to the coded (MiCols*4 x MiRows*4) plane, exactly
// like the decoder.
struct FramePlan {
  int width = 0;
  int height = 0;
  int chroma_width = 0;
  int chroma_height = 0;
  int padded_luma_width = 0;
  int padded_luma_height = 0;
  int padded_chroma_width = 0;
  int padded_chroma_height = 0;
  int mi_cols = 0;  // 2 * ((width + 7) >> 3)
  int mi_rows = 0;
  const BlockPlan* blocks = nullptr;
  int block_count = 0;
};

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

// One plane block (Y, U or V) of one leaf.
struct PlaneJob {
  int plane = 0;   // PlaneIndex
  int block = 0;   // index into FramePlan::blocks
  int syntax = 0;  // CPU coding order (= block index)
  int x = 0;       // plane sample origin
  int y = 0;
  int w = 0;  // coded block size; M1 transform == block size
  int h = 0;
  int tx_w = 0;  // nominal transform size (== w, h)
  int tx_h = 0;
  int cls = 0;      // TX class: (tx_w, tx_h)
  int level = 0;    // wavefront level
  int n4_above_right = 0;  // available 4-sample units, §7.13.2.1
  int n4_below_left = 0;
  int max_x = 0;  // coded plane bounds, inclusive (MiCols*4-1 / MiCols*2-1)
  int max_y = 0;
  int pred_index = 0;  // 3 * block + plane
  int coeff_offset = -1;  // canonical per-frame coefficient element offset
  int buf_offset = -1;    // per-job element offset in execution buffers
  int adj_offset = -1;    // per-job adjusted-TX element offset
  int is_cfl = 0;
  int mode = 0;   // plane decision (AV2/intra mode value)
  int angle = 0;
  int luma_mode = 0;   // co-located luma decision (chroma jobs)
  int luma_angle = 0;
};

// A contiguous run of jobs in one level that share a TX class.  Jobs in a span
// also have contiguous `buf_offset` / `adj_offset` ranges, so a uniform-params
// component call can be issued with one base pointer and a count.
struct ClassSpan {
  int level = 0;
  int cls = 0;
  int job_begin = 0;
  int job_end = 0;
};

// A view of one wavefront level.  Contains no storage; the ranges refer to the
// plan's job array and class-span array, which outlive the batch.
struct LevelBatch {
  int level = 0;
  int job_begin = 0;
  int job_end = 0;
  int class_begin = 0;  // into Plan::class_spans()
  int class_end = 0;
  int luma_jobs = 0;
  int chroma_jobs = 0;
  int cfl_jobs = 0;
  int first_syntax = -1;
  int last_syntax = -1;
};

struct PlanCounters {
  int coding_blocks = 0;
  int plane_jobs = 0;
  int luma_jobs = 0;
  int chroma_jobs = 0;
  int cfl_jobs = 0;
  int levels = 0;
  int transform_classes = 0;
  int groups = 0;  // (level, class) spans
  int max_jobs_per_level = 0;
  long long dependency_edges = 0;
  long long dependency_violations = 0;  // must be 0
  long long syntax_order_violations = 0;  // must be 0 (CPU order is topological)
  bool cyclic = false;
  // Estimate of the batched launch sequence this module drives (one launch
  // per stage class per level).  The harness reports the measured count.
  long long scheduled_kernel_launches = 0;
  // Per-block component-call count of the CPU-shaped pipeline (3 stages per
  // plane block + 2 extra CfL passes per chroma block using CfL).
  long long naive_block_calls = 0;
};

class Plan {
 public:
  Plan() = default;
  // Move-only: input().blocks points into an owned copy of the caller's leaf
  // list, so a copy would leave the pointer aliasing the source plan.
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
  Plan(Plan&&) noexcept = default;
  Plan& operator=(Plan&&) noexcept = default;

  // Builds the wavefront plan.  Returns false and fills `error` on invalid
  // input; the plan is left empty.
  bool build(const FramePlan& plan, std::string& error);
  void clear();

  bool empty() const { return jobs_.empty(); }
  const std::string& error() const { return error_; }
  const FramePlan& input() const { return input_; }
  const std::vector<PlaneJob>& jobs() const { return jobs_; }
  const PlaneJob& job(int index) const { return jobs_[static_cast<std::size_t>(index)]; }
  const std::vector<ClassSpan>& class_spans() const { return spans_; }
  const std::vector<ClassSpan>& groups() const { return spans_; }
  const PlanCounters& counters() const { return counters_; }
  int levels() const { return levels_; }
  int level_count() const { return levels_; }
  int level_begin(int level) const;
  int level_end(int level) const;
  LevelBatch level_batch(int level) const;

  // Dependency graph (CSR).  Dependencies of job i are the job indices in
  // [dependency_begin(i), dependency_end(i)).  Used by tests and integrators.
  int dependency_begin(int job) const;
  int dependency_end(int job) const;
  const std::vector<int>& dependency_indices() const { return dep_indices_; }

  // FNV-1a over the block geometry/decisions; run-time layouts can be checked
  // against the plan that produced the device buffers.
  std::uint64_t signature() const { return signature_; }

  // Total elements in the canonical coefficient layout.
  std::int64_t total_coeffs() const { return total_coeffs_; }

 private:
  FramePlan input_{};
  std::vector<BlockPlan> owned_blocks_;  // copy of the caller's leaf list
  std::vector<PlaneJob> jobs_;
  std::vector<ClassSpan> spans_;
  std::vector<int> level_begin_;
  std::vector<int> span_begin_;
  std::vector<int> dep_begin_;
  std::vector<int> dep_indices_;
  int levels_ = 0;
  std::int64_t total_coeffs_ = 0;
  std::uint64_t signature_ = 0;
  PlanCounters counters_{};
  std::string error_;
};

// Convenience wrapper for the documented API.  Returns an empty plan on
// failure; inspect plan.error().
Plan build_m1_plan(const FramePlan& plan);

const char* plane_name(int plane);

// ---------------------------------------------------------------------------
// Run options/stats shared by the eager and graph executors.
// ---------------------------------------------------------------------------

struct RunOptions {
  bool use_cuda_graph = true;  // replay the captured graph when available
};

struct RunStats {
  bool used_graph = false;
  int host_launches = 0;  // host submission calls (1 with a graph)
  int gpu_nodes = 0;      // graph nodes, or eager kernel launches
  int levels = 0;
  std::size_t workspace_bytes = 0;
  double submit_ms = 0.0;  // host wall time to enqueue the block sequence
};

}  // namespace av2enc::cuda::m1::sched
