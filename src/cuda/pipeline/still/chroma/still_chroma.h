#ifndef AV2ENC_CUDA_STILL_CHROMA_H_
#define AV2ENC_CUDA_STILL_CHROMA_H_

// M1 GPU chroma decision + prediction — AV2 v1.0.0 §7.13.5 / §7.13.2.2.
//
// Bit-exact port of the chroma half of the CPU M1 block coder in
// src/encoder/m1/still_encoder.cpp (`choose_uv_mode`, `make_cfl_block`, the DC
// prediction used for both the DC candidate and the CfL base). "Bit-exact"
// here means byte-identical predictions, identical per-plane SSE, identical
// `is_cfl`/alpha, and therefore identical mode decisions for the same input
// state.
//
// Default policy (CPU default since ff254d0): the chroma candidate set is
//   { DC, CfL }.
// `StillChromaOptions::enable_uv_full` (CPU StillOptions::uv_full / the
// AV2ENC_STILL_UV_FULL gate) enables the CPU full-UV search: all 13 base modes
// with U+V SSE, the §6.19.5.6 derived AngleDeltaUV, edge filtering, and CfL
// last. It uses the same device intra predictor as src/cuda/intra_modes/.
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

namespace av2enc::cuda::still {

// Candidate identifiers. `winner` in StillChromaResult uses these values.
enum StillChromaCandidate : int {
  kM1ChromaDc = 0,
  kM1ChromaCfl = 1,
  // A non-DC directional/smooth/Paeth UV mode (only with enable_uv_full).
  kM1ChromaUvDirectional = 2,
};

// Frame-level chroma policy. M1 defaults: enable_cfl = 1, everything else 0/8.
struct StillChromaOptions {
  // CPU `options_.enable_cfl` (and force_dc / AV2ENC_STILL_UV_DC folded in by the
  // caller as enable_cfl = 0). When 0 the CPU returns DC immediately and sets
  // UvChoice.sse = 0; this module reproduces that.
  int enable_cfl = 1;
  // Legacy alias for enable_uv_full: runs the same 13-mode full search
  // (including DC and, when enable_cfl != 0, CfL). Directional/smooth UV is
  // now bit-exact against the CPU path, so neither flag is rejected.
  int enable_uv_directional = 0;
  // GPU counterpart of the CPU StillOptions::uv_full / AV2ENC_STILL_UV_FULL gate.
  // Non-zero runs the exact CPU choose_uv_mode() full candidate loop. The
  // caller must fill y_mode, y_angle_delta, filter_type_above/left,
  // enable_edge_filter, and num4_above_right/num4_below_left.
  int enable_uv_full = 0;
  // `cfl_ds_filter_index`. M1's make_cfl_block() hardcodes 0; 3 behaves as 0.
  int ds_filter = 0;
  // §5.4.5 enable_ibp.  When set the DC candidate uses the §7.13.2.12 IBP DC
  // post-filter; the CfL base prediction excludes it (uv_is_cfl).
  int enable_ibp = 0;
  // CPU --still-cfl-explicit: also score CFL_EXPLICIT (17 sign/magnitude
  // values per plane) and keep it when its U+V SSE beats the derived alpha.
  int cfl_explicit = 0;
  // CPU --still-mhccp: score the three §7.13.7 CFL_MULTI directions.
  int enable_mhccp = 0;
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
struct StillChromaJob {
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

  // §7.13.2.1 above-right / below-left availability in chroma 4x4 units.
  // DC and CfL do not read them; they only feed directional/smooth candidates.
  // The device predictor applies the chroma >32 rule itself.
  int num4_above_right = 0;
  int num4_below_left = 0;
  // Full-UV candidate inputs (enable_uv_full / enable_uv_directional).
  // y_mode/y_angle_delta drive the §6.19.5.6 derived AngleDeltaUV when the
  // chosen UV mode equals a directional YMode. filter_type_above/left are the
  // §7.13.2.15/.16 smooth-neighbour booleans from the chroma UVSmooth grid
  // (0/1); enable_edge_filter mirrors the sequence flag.
  int y_mode = 0;
  int y_angle_delta = 0;
  int filter_type_above = 0;
  int filter_type_left = 0;
  int enable_edge_filter = 1;
};

struct StillChromaResult {
  int winner = kM1ChromaDc;  // StillChromaCandidate
  int uv_mode = 0;           // AV2 YMode; 0 (DC) for both DC and CfL
  int angle_delta = 0;       // 0 for both DC and CfL
  int is_cfl = 0;
  int alpha_u = 0;  // Q8, only meaningful when is_cfl
  int alpha_v = 0;  // Q8, only meaningful when is_cfl
  // §5.20.7.32 signalling for the winning CfL hypothesis: 1 = derived,
  // 0 = explicit.  `alpha_signs`/`alpha_u`/`alpha_v` are the syntax symbols
  // (magnitude minus one), unused when cfl_index != 0.
  int cfl_index = 1;
  int alpha_signs = 0;
  int alpha_u_syntax = 0;
  int alpha_v_syntax = 0;
  // §7.13.7 CFL_MULTI direction when cfl_index == 2.
  int cfl_mh_dir = 0;
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
cudaError_t chroma_decide(const StillChromaJob& job, const StillChromaOptions& options,
                          StillChromaResult* d_result,
                          cudaStream_t stream = nullptr);

// Batched form: one CUDA block per job. `d_jobs` and `d_results` are device
// arrays of `job_count` entries. job_count <= 0 is a no-op.
cudaError_t chroma_decide_batch(const StillChromaJob* d_jobs, int job_count,
                                const StillChromaOptions& options,
                                StillChromaResult* d_results,
                                cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::still

#endif  // AV2ENC_CUDA_STILL_CHROMA_H_
