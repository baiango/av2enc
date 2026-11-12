// Pure integer math shared by the CUDA kernels, the host wrappers and the
// host-emulation parity test.
//
// These functions are the single source of truth for the reversible arithmetic
// and the §5.18.2 / §5.20.6.1 derivations.  They are `__host__ __device__`
// under nvcc and plain `inline` under a host compiler, so the exact same code
// is compiled into the GPU test and the no-CUDA emulation test.
//
// No CUDA runtime allocation, shared memory or synchronisation appears here:
// every function is reentrant and works on caller-provided buffers.

#ifndef AV2ENC_CUDA_LOSSLESS_INTER_LOSSLESS_INTER_MATH_H_
#define AV2ENC_CUDA_LOSSLESS_INTER_LOSSLESS_INTER_MATH_H_

#include "lossless_inter.h"

#if defined(__CUDACC__) || defined(__CUDA__)
#define AV2ENC_LI_HD __host__ __device__
#else
#define AV2ENC_LI_HD
#endif

namespace av2enc::cuda::lossless_inter {

AV2ENC_LI_HD inline int li_clip3(int low, int high, int value) noexcept {
  return value < low ? low : (value > high ? high : value);
}

AV2ENC_LI_HD inline bool li_pow2_dim(int v) noexcept {
  return v == 4 || v == 8 || v == 16 || v == 32;
}

// §7.15.2.2 inverse WHT on four samples.  `shift` is 3 for the row pass of a
// lossless 4x4 transform and 0 for the column pass.  The `>> 1` lifting step
// floors, so the transform is not self-inverse for odd (a - d); the forward
// function below is the algebraic inverse the encoder uses.
AV2ENC_LI_HD inline void li_wht4_inverse(const int in[4], int shift,
                                         int out[4]) noexcept {
  int a = in[0] >> shift;
  int c = in[1] >> shift;
  int d = in[2] >> shift;
  int b = in[3] >> shift;
  a += c;
  d -= b;
  const int e = (a - d) >> 1;
  b = e - b;
  c = e - c;
  a -= b;
  d += c;
  out[0] = a;
  out[1] = b;
  out[2] = c;
  out[3] = d;
}

// Encoder-side forward WHT, exactly inverting li_wht4_inverse(..., 0, ...).
AV2ENC_LI_HD inline void li_wht4_forward(const int in[4],
                                         int out[4]) noexcept {
  const int a = in[0];
  const int b = in[1];
  const int c = in[2];
  const int d = in[3];
  const int a1 = a + b;
  const int d1 = d - c;
  const int e = (a1 - d1) >> 1;
  const int b0 = e - b;
  const int c0 = e - c;
  const int a0 = a1 - c0;
  const int d0 = d1 + b0;
  // Input order is src[0]=a, src[1]=c, src[2]=d, src[3]=b.
  out[0] = a0;
  out[1] = c0;
  out[2] = d0;
  out[3] = b0;
}

// §5.20.6.1 find_tx_size(): Min(32, coded dimension) when it is a legal
// lossless transform extent (4/8/16/32).
AV2ENC_LI_HD inline bool li_find_tx_size(int coded_w, int coded_h, int* tx_w,
                                         int* tx_h) noexcept {
  const int w = coded_w < 32 ? coded_w : 32;
  const int h = coded_h < 32 ? coded_h : 32;
  if (!li_pow2_dim(w) || !li_pow2_dim(h)) return false;
  *tx_w = w;
  *tx_h = h;
  return true;
}

// §5.20.6.1 read_tx_size() lossless branch, identical to the CPU reference.
AV2ENC_LI_HD inline TxSizeDecision li_select_tx_size(
    const TxSizeRequest& request) noexcept {
  TxSizeDecision d;
  const bool blk4 = request.block_w == 4 && request.block_h == 4;
  if (blk4 || (request.is_inter == 0 && request.fsc_mode == 0) ||
      request.allow_select == 0) {
    d.coded_w = 4;
    d.coded_h = 4;
    return d;
  }
  d.signal_lossless_tx_size = true;
  d.lossless_tx_size = (request.want_large != 0);
  if (d.lossless_tx_size) {
    int w = 4;
    int h = 4;
    if (li_find_tx_size(request.block_w, request.block_h, &w, &h)) {
      d.coded_w = w;
      d.coded_h = h;
      return d;
    }
  }
  d.coded_w = 4;
  d.coded_h = 4;
  d.lossless_tx_size = false;
  return d;
}

// §5.18.2 LosslessArray / CodedLossless with the per-segment qindex.  The
// derivation is frame-type independent, exactly like the CPU reference.
AV2ENC_LI_HD inline FrameLossless li_derive_lossless(
    const FrameQuantInputs& inputs) noexcept {
  FrameLossless out;
  out.coded_lossless = true;
  int segments = inputs.num_segments;
  if (segments <= 0) segments = kBaseSegments;
  if (segments > kMaxSegments) segments = kMaxSegments;
  if (segments < 1) segments = 1;
  out.num_segments = segments;
  for (int s = 0; s < segments; ++s) {
    int qindex = inputs.base_q_idx;
    if (inputs.alt_q_enabled[s]) qindex += inputs.alt_q[s];
    qindex = li_clip3(0, 255, qindex);
    out.qindex[s] = qindex;
    const bool lossless =
        qindex == 0 && inputs.delta_q_present == 0 &&
        inputs.delta_q_y_dc + inputs.base_y_dc_delta_q <= 0 &&
        inputs.delta_q_u_dc + inputs.base_uv_dc_delta_q <= 0 &&
        inputs.delta_q_v_dc + inputs.base_uv_dc_delta_q <= 0 &&
        inputs.delta_q_u_ac + inputs.base_uv_ac_delta_q <= 0 &&
        inputs.delta_q_v_ac + inputs.base_uv_ac_delta_q <= 0;
    out.lossless[s] = lossless;
    if (lossless) {
      out.has_lossless_segment = true;
    } else {
      out.coded_lossless = false;
    }
  }
  return out;
}

}  // namespace av2enc::cuda::lossless_inter

#endif  // AV2ENC_CUDA_LOSSLESS_INTER_LOSSLESS_INTER_MATH_H_
