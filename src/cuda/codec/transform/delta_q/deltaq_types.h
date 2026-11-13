// CUDA-free POD descriptors for the §7.14.4 DC/AC delta-Q quantizer.
//
// The CPU reference is `src/codec/transform/delta_q/` (`QuantBlock` +
// `quantize_block` / `dequantize_block`).  This header is deliberately free of
// CUDA and of the CPU reference headers so the same descriptors are used by
// device code, the real-GPU test, and the host-emulation parity test.
#pragma once

#include <cstdint>

namespace av2enc::cuda::deltaq {

// §7.14.2 qindex domain constants (8-bit).
inline constexpr int kMaxQ = 255;
inline constexpr int kMaxAdjustedDim = 32;
inline constexpr int kMaxQuantElems = kMaxAdjustedDim * kMaxAdjustedDim;
inline constexpr int kQuantStep = 528;
inline constexpr int kQuantTableBits = 3;

// Per-plane residual coefficient task.  A job is one transform block; the
// buffers in `Batch` are shared and addressed by element offsets so a batch
// can mix transform sizes.
//
// `delta_dc` / `delta_ac` are the *resolved* totals for this plane, i.e.
// `DeltaQ* + Base*` exactly as `dcdeltaq::resolve_delta_q()` returns them.
// The device derives the §7.14.2 DC and AC quantizers from `base_q_idx` and
// these deltas (`get_q()`), so the per-plane delta-Q decision stays on device.
// Plane 0 (luma) AC is always the no-delta quantizer, as on the CPU.
struct Job {
  std::int32_t in_offset = 0;   // element offset into the input buffer
  std::int32_t out_offset = 0;  // element offset into the output buffer
  std::int32_t plane = 0;       // 0 = Y, 1 = U, 2 = V
  std::int32_t coded_w = 4;     // coded block width (4..64; 64 -> adjusted 32)
  std::int32_t coded_h = 4;     // coded block height
  // §7.14.2 `qindex` for this block: the §7.14.2 get_qindex() result
  // (equals base_q_idx when delta_q_present = 0 and no segment ALT_Q applies).
  std::int32_t qindex = 0;
  std::int32_t delta_dc = 0;    // resolved delta for the DC quantizer
  std::int32_t delta_ac = 0;    // resolved delta for the AC quantizer
  std::int32_t tx_type = 0;     // TX_TYPE 0..15; IDTX (9) is TX_CLASS_2D
  // -1: derive the §7.14.4 q2 matrix from the DC/AC quantizers.
  // >=0: element offset of a prebuilt `w*h` q2 matrix in `Batch::q2`
  //      (quantizer-matrix interplay; the host builds it with the CPU
  //      `build_q2_matrix()` so the QM search stays off the device).
  std::int32_t q2_offset = -1;
  std::int32_t lossless = 0;    // §5.18.2 Lossless for this block
  std::int32_t allow_tcq = 0;   // §5.20.7.27 allow_tcq
  std::int32_t use_fsc = 0;     // luma IDTX FSC path (disables the TCQ shift)
  // Forward transform coefficient multiplier at q2 = 528 (the value the CPU
  // `QuantBlock::coefficient_scale` carries).
  double coefficient_scale = 1.0;
};

struct Batch {
  // quantize_batch: input coefficients.  dequantize_batch: input levels.
  const double* coeff = nullptr;
  const std::int32_t* levels = nullptr;
  // quantize_batch: output levels.  dequantize_batch: output dequant.
  std::int32_t* out = nullptr;
  const Job* jobs = nullptr;
  std::int32_t num_jobs = 0;
  // Optional prebuilt q2 matrices (see Job::q2_offset).
  const std::int32_t* q2 = nullptr;
  // Optional per-job result: 0 = ok, -1 = rejected.  When null, invalid jobs
  // are skipped silently (their output is left untouched).
  std::int32_t* status = nullptr;
};

}  // namespace av2enc::cuda::deltaq
