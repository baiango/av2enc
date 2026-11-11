#ifndef AV2ENC_CUDA_M1_CHROMA_H_
#define AV2ENC_CUDA_M1_CHROMA_H_

// M1 GPU chroma decision + prediction — AV2 v1.0.0 §7.13.5 / §7.13.2.2.
//
// Bit-exact port of the chroma half of the CPU M1 block coder in
// src/encoder/m1/m1_still.cpp (`choose_uv_mode`, `make_cfl_block`, the DC
// prediction used for both the DC candidate and the CfL base). "Bit-exact"
// here means byte-identical predictions, identical per-plane SSE, identical
// `is_cfl`/alpha, and therefore identical mode decisions for the same input
// state.
//
// Default policy (CPU default since ff254d0): the chroma candidate set is
//   { DC, CfL }.
// Directional/smooth UV prediction is NOT bit-exact against the pinned
// decoder yet (docs/negative-results.md), so it is never selected. The
// candidate plumbing (`M1ChromaCandidate`, the scoring switch, the reserved
// availability fields) is kept so that re-enabling it later does not change
// this ABI.
//
// Inputs are device pointers. The caller owns all memory (arena), the stream,
// and synchronization; these entry points never allocate or synchronize.
//
// The reconstructed chroma planes must contain the already-coded neighbours
// (above row / left column). The current block area is NOT read: this module
// derives the DC prediction itself and computes CfL on top of it, exactly like
// `choose_uv_mode()` does before it writes the DC prediction into the planes.
// The winning prediction is returned in caller-provided `pred_u`/`pred_v`
// planes, so the input chroma planes are not modified (the CPU version only
// writes them as transient scratch that the reconstruction pass overwrites).

#include <cstdint>

#include <cuda_runtime.h>

namespace av2enc::cuda::m1 {

// Candidate identifiers. `winner` in M1ChromaResult uses these values.
enum M1ChromaCandidate : int {
  kM1ChromaDc = 0,
  kM1ChromaCfl = 1,
  // Reserved: not implemented, rejected by the entry points below.
  kM1ChromaUvDirectional = 2,
};

// Frame-level chroma policy. M1 defaults: enable_cfl = 1, everything else 0/8.
struct M1ChromaOptions {
  // CPU `options_.enable_cfl` (and force_dc / AV2ENC_M1_UV_DC folded in by the
  // caller as enable_cfl = 0). When 0 the CPU returns DC immediately and sets
  // UvChoice.sse = 0; this module reproduces that.
  int enable_cfl = 1;
  // Must remain 0. Directional/smooth UV candidates are not bit-exact yet; a
  // non-zero value makes the entry points return cudaErrorNotSupported instead
  // of silently producing a divergent decision.
  int enable_uv_directional = 0;
  // GPU counterpart of the CPU M1Options::uv_full / --m1-uv-full flag. The
  // full candidate loop is not implemented in this module yet; non-zero also
  // returns cudaErrorNotSupported until the orchestrator wires the job fields
  // below. The CPU path is verified; this keeps the GPU path from silently
  // selecting a different UV mode when the flag is enabled.
  int enable_uv_full = 0;
  // `cfl_ds_filter_index`. M1's make_cfl_block() hardcodes 0; 3 behaves as 0.
  int ds_filter = 0;
  // Clamp applied to the derived (implicit) alpha, in Q8 units. §7.13.5
  // resolve_division() returns a signed value with magnitude <= 511 (chroma
  // can correlate negatively with luma), so the default range is a no-op and
  // the CPU parity path is exact. A narrower range is only for encoder
  // experiments; it changes decisions by construction.
  int alpha_q8_min = -511;
  int alpha_q8_max = 511;
};

// One chroma residual block plus the reconstruction/source state it needs.
// All pointers are device pointers. Coordinates are chroma samples; the luma
// origin is (x*2, y*2) (M1 is 8-bit 4:2:0).
struct M1ChromaJob {
  // Reconstructed luma plane (padded stride) and its inclusive coded bounds
  // (MiCols*MI_SIZE - 1 / MiRows*MI_SIZE - 1), exactly like cfl::Block.
  const std::uint8_t* luma = nullptr;
  int luma_stride = 0;
  int luma_max_x = 0;
  int luma_max_y = 0;
  int sb_top_y = 0;

  // Reconstructed chroma planes (padded stride). Only the above row and left
  // column of the current block are read (CfL derived alpha); the block itself
  // is ignored.
  const std::uint8_t* chroma_u = nullptr;
  const std::uint8_t* chroma_v = nullptr;
  int chroma_stride = 0;

  // Destination prediction planes, tightly packed rows of `w` samples each,
  // or with `pred_stride` >= w. Either may be null to skip that plane.
  std::uint8_t* pred_u = nullptr;
  std::uint8_t* pred_v = nullptr;
  int pred_stride = 0;

  // Source chroma (visible frame), used only for SSE scoring.
  const std::uint8_t* src_u = nullptr;
  const std::uint8_t* src_v = nullptr;
  int src_stride_u = 0;
  int src_stride_v = 0;
  int src_max_x = 0;  // inclusive, visible chroma width - 1
  int src_max_y = 0;  // inclusive, visible chroma height - 1

  // Chroma block rect.
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;

  // Coded chroma bounds: MiCols*MI_SIZE>>1, MiRows*MI_SIZE>>1.
  int coded_width = 0;
  int coded_height = 0;

  // Reserved for the directional/smooth candidates (§7.13.2 availability).
  // DC and CfL do not read them; the CPU derives DC availability from x/y and
  // §5.20.4.1 chroma-mi availability from x/4, y/4.
  int num4_above_right = 0;
  int num4_below_left = 0;
  // Full-UV candidate inputs (CPU M1Options::uv_full). Unused by DC/CfL.
  // y_mode/y_angle_delta drive the §6.19.5.6 derived AngleDeltaUV when the
  // chosen UV mode equals a directional YMode. filter_type_* are the OR-ed
  // §7.13.2.15/.16 smooth-neighbour flags for the chroma grid; enable_edge_filter
  // mirrors the sequence flag.
  int y_mode = 0;
  int y_angle_delta = 0;
  int filter_type_above = 0;
  int filter_type_left = 0;
  int enable_edge_filter = 1;
};

struct M1ChromaResult {
  int winner = kM1ChromaDc;  // M1ChromaCandidate
  int uv_mode = 0;           // AV2 YMode; 0 (DC) for both DC and CfL
  int angle_delta = 0;       // 0 for both DC and CfL
  int is_cfl = 0;
  int alpha_u = 0;  // Q8, only meaningful when is_cfl
  int alpha_v = 0;  // Q8, only meaningful when is_cfl
  // The DC prediction value the kernel used as the CfL base (diagnostic).
  int dc_u = 0;
  int dc_v = 0;

  // Per-plane SSE. The CPU scores DC on U only and CfL on U+V; those are the
  // values that drive the decision. The V-plane DC SSE is filled as a
  // diagnostic (the CPU does not score it).
  long long sse_dc_u = 0;
  long long sse_dc_v = 0;
  long long sse_cfl_u = 0;
  long long sse_cfl_v = 0;

  // CPU-comparable winning score: `UvChoice.sse`. For enable_cfl = 0 this is
  // 0 (the CPU returns before scoring); otherwise it is sse_dc_u or
  // sse_cfl_u + sse_cfl_v.
  long long sse = 0;
};

// Single block. `d_result` is device memory. The job is passed by value; no
// host-to-device copy or allocation is performed.
cudaError_t chroma_decide(const M1ChromaJob& job, const M1ChromaOptions& options,
                          M1ChromaResult* d_result,
                          cudaStream_t stream = nullptr);

// Batched form: one CUDA block per job. `d_jobs` and `d_results` are device
// arrays of `job_count` entries. job_count <= 0 is a no-op.
cudaError_t chroma_decide_batch(const M1ChromaJob* d_jobs, int job_count,
                                const M1ChromaOptions& options,
                                M1ChromaResult* d_results,
                                cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::m1

#endif  // AV2ENC_CUDA_M1_CHROMA_H_
