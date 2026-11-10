// Dependency-aware wavefront scheduler for the CUDA M0 DC-intra block path.
//
// The M0 pipeline (src/cuda/m0_pipeline.*) walks the coding blocks in raster
// order and issues ~13 component calls per 32x32 block: 2,740 component calls
// (3,580 kernel launches) for one 640x360 frame.  That is launch-bound
// (~34 ms/frame measured) and extrapolates to tens of thousands of launches at
// 1080p.
//
// This module replaces the per-block call sequence with a plan:
//
//   1. every plane block (Y, U, V) becomes an independent node, with edges to
//      the blocks that own its DC-prediction left column and above row;
//   2. nodes are assigned a wavefront level = 1 + max(level of prerequisites),
//      so all nodes in a level are independent;
//   3. per level, the prediction + residual work is one batched launch, and all
//      transform/quantise/dequantise/inverse/reconstruct work is batched per
//      transform class (nodes that share TxParams) into five launches;
//   4. the whole per-frame sequence is captured into a CUDA graph once and
//      replayed with a single cudaGraphLaunch per frame.
//
// Bit-exactness: the module never re-implements normative arithmetic.  It calls
// the shared components (intra.h, transform.h) exactly like the M0 pipeline and
// only batches *independent* invocations of them.  The one piece of glue is the
// batched reconstruction launcher, which applies the same
// `clip1(pred + residual)` / 2x2-duplication expression as
// intra::add_residual_clip_to_plane / transform::reconstruct_64x64; the test
// harness compares every reconstructed plane, coefficient and predictor against
// the per-block M0 path.
//
// Channel-level dependency rules (normative §7.13.2 DC prediction):
//   * a luma block at plane (x0, y0) reads the reconstructed column x0-1 over
//     rows [y0, y0+h) and the reconstructed row y0-1 over columns [x0, x0+w);
//   * luma, U and V are separate planes, so their chains are independent;
//   * blocks on the same wavefront anti-diagonal never depend on each other.
//
// The module is deliberately free of CUDA types in its header so the plan can
// be unit-tested on a host without a GPU (see scheduler_plan_test.cpp).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace av2enc::cuda::sched {

// ---------------------------------------------------------------------------
// Input description (M0 coding blocks)
// ---------------------------------------------------------------------------

// One M0 coding block.  Field-for-field the same as the M0 pipeline's
// detail::BlockDesc; use blocks_from() to convert without this module including
// m0_pipeline.h.
struct CodingBlock {
  int mi_row = 0;  // luma position in 4x4 units
  int mi_col = 0;
  int luma_w = 0;   // 32x32 / 64x64 (M0 target tree)
  int luma_h = 0;
  int chroma_w = 0;  // half the luma size
  int chroma_h = 0;
  int upscale = 1;   // 2 for the 64x64 -> adjusted 32x32 luma path
  int luma_pels = 0;    // quantised luma coefficients (tx_w * tx_h)
  int chroma_pels = 0;  // quantised coefficients per chroma plane
  int coeff_offset = 0;  // canonical layout: Y, U, V per block
  int real_coeffs = 0;   // 0 = M0 diagnostic block; skipped by the GPU
};

struct FrameGeometry {
  int width = 0;
  int height = 0;
  int chroma_width = 0;
  int chroma_height = 0;
  int padded_width = 0;
  int padded_height = 0;
  int chroma_padded_width = 0;
  int chroma_padded_height = 0;
  int mi_cols = 0;  // 2 * ((width + 7) >> 3)
  int mi_rows = 0;
};

// Host-side in/out description.  `y/u/v` are read (uploaded) on every run;
// `recon_*`, `coeffs` and `preds` are written (downloaded) after the run.  The
// layouts are the M0 pipeline's: padded recon planes, canonical per-block
// coefficient blocks, and three int32 predictions per coding block.
struct FrameJob {
  const std::uint8_t* y = nullptr;
  const std::uint8_t* u = nullptr;
  const std::uint8_t* v = nullptr;
  FrameGeometry geometry{};
  const CodingBlock* blocks = nullptr;
  int block_count = 0;
  std::int64_t total_coeffs = 0;  // int32 entries in `coeffs`
  std::uint8_t* recon_y = nullptr;
  std::uint8_t* recon_u = nullptr;
  std::uint8_t* recon_v = nullptr;
  std::int32_t* coeffs = nullptr;
  std::int32_t* preds = nullptr;
};

// Adapter for the M0 pipeline's detail::BlockDesc (same member names).
template <typename BlockDesc>
std::vector<CodingBlock> blocks_from(const BlockDesc* blocks, int count) {
  std::vector<CodingBlock> out;
  if (blocks == nullptr || count <= 0) {
    return out;
  }
  out.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const BlockDesc& b = blocks[i];
    CodingBlock c;
    c.mi_row = b.mi_row;
    c.mi_col = b.mi_col;
    c.luma_w = b.luma_w;
    c.luma_h = b.luma_h;
    c.chroma_w = b.chroma_w;
    c.chroma_h = b.chroma_h;
    c.upscale = b.upscale;
    c.luma_pels = b.luma_pels;
    c.chroma_pels = b.chroma_pels;
    c.coeff_offset = b.coeff_offset;
    c.real_coeffs = b.real_coeffs;
    out.push_back(c);
  }
  return out;
}

// FNV-1a over the block list.  run() uses it to reject a frame whose coding
// layout differs from the one prepare() built descriptors for.
std::uint64_t coding_blocks_signature(const CodingBlock* blocks, int count);

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

// One plane block.  `plane` is 0 = Y, 1 = U, 2 = V.
struct PlaneJob {
  int plane = 0;
  int block = 0;  // index into the CodingBlock list (for preds[] and diagnostics)
  int x = 0;      // plane sample origin
  int y = 0;
  int w = 0;      // plane block size (64x64 / 32x32 / 16x16)
  int h = 0;
  int tx_w = 0;   // transform size (32x32 for an adjusted 64x64 luma block)
  int tx_h = 0;
  int upscale = 1;
  int coeff_count = 0;  // tx_w * tx_h; == luma_pels / chroma_pels
  int max_x = 0;        // visible plane bounds (inclusive)
  int max_y = 0;
  int level = 0;        // wavefront level
  int cls = 0;          // transform class id
  int slot = 0;         // block slot inside the class buffers
  int scratch_slot = -1;  // 64x64 residual/pred scratch slot (upscale == 2)
  int pred_index = 0;     // 3 * block + plane
  int coeff_offset = 0;   // canonical coefficient offset
};

// A contiguous run of PlaneJob entries whose slots are also contiguous.
struct WaveGroup {
  int level = 0;
  int cls = 0;
  int job_begin = 0;
  int job_end = 0;
  int slot_begin = 0;
  int slot_end = 0;
};

struct TransformClass {
  int tx_w = 0;
  int tx_h = 0;
  int upscale = 1;
  int pels = 0;       // tx_w * tx_h
  int job_count = 0;  // plane blocks in this class
};

struct PlanCounters {
  int coding_blocks = 0;
  int real_blocks = 0;
  int plane_jobs = 0;
  int levels = 0;  // wavefront depth (independent waves)
  int groups = 0;
  int transform_classes = 0;
  int luma_64_jobs = 0;
  long long dependency_edges = 0;
  long long dependency_violations = 0;  // must be 0
  bool cyclic = false;
  long long naive_component_calls = 0;  // matches the M0 pipeline's counter
  long long naive_kernel_launches = 0;  // actual kernels on the naive path
  long long scheduled_kernel_launches = 0;  // eager wavefront kernels
};

class Plan {
 public:
  Plan() = default;

  // Builds the wavefront plan.  Idempotent; on failure returns false with a
  // reason and leaves the plan empty.
  bool build(const FrameJob& job, std::string& error);
  void clear();

  bool empty() const { return jobs_.empty(); }
  const FrameGeometry& geometry() const { return geometry_; }
  const std::vector<PlaneJob>& jobs() const { return jobs_; }
  const std::vector<WaveGroup>& groups() const { return groups_; }
  const std::vector<TransformClass>& classes() const { return classes_; }
  int levels() const { return levels_; }
  int level_begin(int level) const;
  int level_end(int level) const;
  const PlanCounters& counters() const { return counters_; }
  std::uint64_t blocks_signature() const { return blocks_signature_; }

 private:
  FrameGeometry geometry_{};
  std::vector<PlaneJob> jobs_;
  std::vector<WaveGroup> groups_;
  std::vector<TransformClass> classes_;
  std::vector<int> level_begin_;
  int levels_ = 0;
  PlanCounters counters_{};
  std::uint64_t blocks_signature_ = 0;
};

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

struct RunOptions {
  // Replay the captured graph (one host launch) instead of enqueueing the
  // planned launches (still batched, but one host call per kernel).
  bool use_cuda_graph = true;
};

struct RunStats {
  bool used_graph = false;
  int host_launches = 0;  // cudaGraphLaunch count + copies, or eager kernel count
  int gpu_nodes = 0;      // graph nodes executed, or eager kernel launches
  std::size_t workspace_bytes = 0;  // prepared device arena footprint
  double upload_ms = 0.0;
  double kernel_ms = 0.0;  // GPU time between the upload and download events
  double download_ms = 0.0;
  double submit_ms = 0.0;  // host wall time to enqueue the frame
};

// Owns the persistent device workspace and (optionally) one captured graph for
// a fixed frame layout.  prepare() allocates once and captures; run() may be
// called for every frame with the same geometry and block list.
class ScheduledFrame {
 public:
  ScheduledFrame();
  ~ScheduledFrame();
  ScheduledFrame(const ScheduledFrame&) = delete;
  ScheduledFrame& operator=(const ScheduledFrame&) = delete;

  bool prepare(const FrameJob& job, int device, std::string& error);
  bool run(const FrameJob& job, const RunOptions& options, RunStats* stats,
           std::string& error);
  void release();

  bool prepared() const;
  bool graph_available() const;
  const std::string& graph_error() const;
  const Plan& plan() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

const char* plane_name(int plane);

}  // namespace av2enc::cuda::sched
