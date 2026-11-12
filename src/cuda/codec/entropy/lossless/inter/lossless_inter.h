// AV2 v1.0.0 lossless inter residual path — CUDA component.
//
// Scope matches the CPU reference in `src/encoder/lossless/inter/`:
//
//   * §7.15.2.2  reversible 4x4 WHT (forward and inverse, exact integer
//                rounding including the odd (a - d) lifting case);
//   * §7.15.4    lossless 2D residual transform at 4x4..32x32 (DCT_DCT = WHT)
//                and the identity / FSC path (IDTX = level copy);
//   * DPCM first-difference / cumulative sum for luma and chroma residual
//                blocks (4x4, 8x8, rectangular, both V and H modes);
//   * §5.20.6.1 / §5.20.8.2 lossless inter transform-size selection;
//   * §5.18.2    per-segment qindex / LosslessArray derivation and
//                segment-aware lossless reconstruction.
//
// Every device buffer is allocated from the caller's `av2enc::cuda::Arena`.
// This component never calls cudaMalloc/cudaFree/cudaMallocHost and never
// allocates inside a launcher.  A `Workspace` holds the persistent per-batch
// descriptor staging buffers; `required_device_bytes()` reports the exact
// arena requirement before `Workspace::init()` is called.  An undersized arena
// yields `Status::kArenaExhausted` with the required size returned.
//
// All arithmetic is integer and bit-exact with the CPU module; entry points
// return a `Status` and never synchronise implicitly.

#ifndef AV2ENC_CUDA_LOSSLESS_INTER_LOSSLESS_INTER_H_
#define AV2ENC_CUDA_LOSSLESS_INTER_LOSSLESS_INTER_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "cuda/error.h"

namespace av2enc::cuda::lossless_inter {

// Mirrors av2enc::lossless::kMaxSegments / kBaseSegments.
inline constexpr int kMaxSegments = 16;
inline constexpr int kBaseSegments = 8;

// Largest transform extent supported by the lossless module (32x32).
inline constexpr int kMaxTransformDim = 32;
inline constexpr int kMaxTransformSamples = kMaxTransformDim * kMaxTransformDim;

// ---------------------------------------------------------------------------
// §5.18.2 lossless flags and per-segment qindex
// ---------------------------------------------------------------------------

// Same fields and semantics as
// `av2enc::lossless::inter::FrameQuantInputs`.
struct FrameQuantInputs {
  int base_q_idx = 0;
  int delta_q_present = 0;

  int delta_q_y_dc = 0;
  int base_y_dc_delta_q = 0;
  int delta_q_u_dc = 0;
  int delta_q_v_dc = 0;
  int delta_q_u_ac = 0;
  int delta_q_v_ac = 0;
  int base_uv_dc_delta_q = 0;
  int base_uv_ac_delta_q = 0;

  bool alt_q_enabled[kMaxSegments] = {};
  int alt_q[kMaxSegments] = {};
  int num_segments = kBaseSegments;
};

struct FrameLossless {
  bool coded_lossless = false;
  bool has_lossless_segment = false;
  bool lossless[kMaxSegments] = {};
  int qindex[kMaxSegments] = {};
  int num_segments = 0;
};

// Host mirror of the CPU §5.18.2 derivation (`derive_lossless`).
FrameLossless derive_lossless(const FrameQuantInputs& inputs);

// ---------------------------------------------------------------------------
// §5.20.6.1 / §5.20.8.2 lossless transform-size selection
// ---------------------------------------------------------------------------

struct TxSizeRequest {
  int block_w = 4;
  int block_h = 4;
  // 1 = inter coding block, 0 = intra.
  int is_inter = 0;
  // Intra FSC mode (only meaningful when is_inter == 0).
  int fsc_mode = 0;
  // read_tx_size() saw an explicit transform-size choice.
  int allow_select = 1;
  // Value of the lossless_tx_size symbol when it is present.
  int want_large = 0;
};

// Mirrors `av2enc::lossless::inter::TxSizeDecision`.
struct TxSizeDecision {
  int coded_w = 4;
  int coded_h = 4;
  bool lossless_tx_size = false;
  bool signal_lossless_tx_size = false;
};

bool find_tx_size(int coded_w, int coded_h, int* tx_w, int* tx_h);
TxSizeDecision select_tx_size(const TxSizeRequest& request);

// ---------------------------------------------------------------------------
// Block jobs (all pointers are device pointers, all buffers arena-resident)
// ---------------------------------------------------------------------------

// One 4-sample WHT.  `in`/`out` point to 4 consecutive int32 values.
struct WhtJob {
  const int* in = nullptr;
  int* out = nullptr;
  // inverse only: pre-scale shift (3 for the row pass, 0 for the column pass)
  int shift = 0;
};

// Encoder residual: optional DPCM first difference, then §7.15.4 forward.
//   levels = WHT( difference( residual ) )          (dct_dct != 0)
//   levels =       difference( residual )           (dct_dct == 0, IDTX/FSC)
struct ForwardJob {
  const int* residual = nullptr;
  int* levels = nullptr;
  int w = 4;
  int h = 4;
  int dct_dct = 1;
  int dpcm = 0;
  int vertical = 0;
};

// Decoder residual before prediction is added: inverse transform, then the
// optional DPCM cumulative sum.  Matches `inverse_lossless_block` +
// `dpcm_cumsum_block` in the CPU reference (no clipping, so the exact residual
// integers are observable).
struct InverseJob {
  const int* levels = nullptr;
  int* residual = nullptr;
  int w = 4;
  int h = 4;
  int dct_dct = 1;
  int dpcm = 0;
  int vertical = 0;
};

// Standalone DPCM conversion, out-of-place (`in` and `out` must not overlap
// for the difference direction; the cumulative sum also accepts in == out).
struct DpcmJob {
  const int* in = nullptr;
  int* out = nullptr;
  int w = 4;
  int h = 4;
  int vertical = 0;
};

// Decoder reconstruction: inverse transform, optional DPCM cumulative sum,
// then Clip1(pred + residual) written with `recon_stride`.
// `frame_index` selects the per-frame `FrameLossless` record for the segmented
// entry point; `segment_id` is the block's segment.
struct ReconJob {
  const int* levels = nullptr;
  const std::uint8_t* pred = nullptr;
  std::uint8_t* recon = nullptr;
  int pred_stride = 0;
  int recon_stride = 0;
  int w = 4;
  int h = 4;
  int dct_dct = 1;
  int dpcm = 0;
  int vertical = 0;
  int frame_index = 0;
  int segment_id = 0;
};

// ---------------------------------------------------------------------------
// Persistent device workspace
// ---------------------------------------------------------------------------

// Sizing request for one workspace.  `max_jobs` is the largest batch any entry
// point will be asked to run; `max_block_w/h` bound the transform extent used
// by the caller (validation only, not allocation).
struct LaunchConfig {
  int max_jobs = 1;
  int max_block_w = kMaxTransformDim;
  int max_block_h = kMaxTransformDim;
};

// Exact number of device bytes `Workspace::init()` will take from the arena for
// this config (all arrays 256-byte aligned, matching Arena's bump allocator).
std::uint64_t required_device_bytes(const LaunchConfig& config);

// Deviates from the usual `class` because the device pointers are the contract
// with the launchers.  Treat every field as read-only after init().
struct Workspace {
  WhtJob* wht_jobs = nullptr;
  ForwardJob* forward_jobs = nullptr;
  InverseJob* inverse_jobs = nullptr;
  DpcmJob* dpcm_jobs = nullptr;
  ReconJob* recon_jobs = nullptr;
  TxSizeRequest* tx_requests = nullptr;
  TxSizeDecision* tx_decisions = nullptr;
  FrameQuantInputs* quant_inputs = nullptr;
  FrameLossless* lossless = nullptr;
  std::uint8_t* recon_status = nullptr;

  int max_jobs = 0;
  int max_w = 0;
  int max_h = 0;
  std::uint64_t bytes = 0;

  // Allocates every persistent buffer exactly once from `arena`.  On
  // `Status::kArenaExhausted` *required_bytes receives the exact size the
  // config needs so the caller can grow the arena.  Never allocates per
  // launch.  `error` is optional.
  Status init(Arena& arena, const LaunchConfig& config,
              std::uint64_t* required_bytes = nullptr,
              std::string* error = nullptr);

  // Forgets the pointers without freeing (the arena owns the memory).
  void reset() noexcept;

  bool valid() const noexcept { return max_jobs > 0 && wht_jobs != nullptr; }
};

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------
//
// `jobs` is host memory; the launcher copies it to the workspace staging array
// on `stream`.  `count` must be in [1, workspace.max_jobs].  All data pointers
// inside the jobs are device pointers.  Outputs written to workspace arrays can
// be downloaded by the caller; `host_out` (when non-null) is copied back
// asynchronously on `stream`.

Status wht4_forward(Workspace& workspace, const WhtJob* jobs, int count,
                    cudaStream_t stream = nullptr);
Status wht4_inverse(Workspace& workspace, const WhtJob* jobs, int count,
                    cudaStream_t stream = nullptr);

Status dpcm_difference(Workspace& workspace, const DpcmJob* jobs, int count,
                       cudaStream_t stream = nullptr);
Status dpcm_cumsum(Workspace& workspace, const DpcmJob* jobs, int count,
                   cudaStream_t stream = nullptr);

Status forward_blocks(Workspace& workspace, const ForwardJob* jobs, int count,
                      cudaStream_t stream = nullptr);

Status inverse_blocks(Workspace& workspace, const InverseJob* jobs, int count,
                      cudaStream_t stream = nullptr);

Status reconstruct_blocks(Workspace& workspace, const ReconJob* jobs,
                          int count, cudaStream_t stream = nullptr);

// Segment-aware reconstruction.  `segment_qp` is a device array indexed by
// `ReconJob::frame_index`; when null the workspace's `lossless` array is used.
// Blocks whose segment is not lossless are skipped and reported as 0 in
// workspace.recon_status; reconstructed blocks report 1.
Status reconstruct_blocks_segmented(Workspace& workspace, const ReconJob* jobs,
                                    int count,
                                    const FrameLossless* segment_qp,
                                    cudaStream_t stream = nullptr);

// Device-side §5.20.6.1 selection.  Results are staged in the workspace and
// copied to `host_out` when non-null.
Status select_tx_size_batch(Workspace& workspace, const TxSizeRequest* requests,
                            int count, TxSizeDecision* host_out,
                            cudaStream_t stream = nullptr);

// Device-side §5.18.2 derivation.  Results are staged in the workspace and
// copied to `host_out` when non-null.
Status derive_lossless_batch(Workspace& workspace,
                             const FrameQuantInputs* inputs, int count,
                             FrameLossless* host_out,
                             cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::lossless_inter

#endif  // AV2ENC_CUDA_LOSSLESS_INTER_LOSSLESS_INTER_H_
