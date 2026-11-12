#ifndef AV2ENC_CUDA_IBP_IBP_H_
#define AV2ENC_CUDA_IBP_IBP_H_

// AV2 intra bi-prediction (IBP) — CUDA API.
//
// Normative source: AV2 v1.0.0 §7.13.2.7/.8/.9/.12/.14/.17/.18.
//
// The component owns the *directional* and *DC* prediction paths, because
// those are the only intra modes IBP modifies:
//   * directional modes: §7.13.2.7 edge-filter changes and, when useIBP is
//     true, the §7.13.2.8/§7.13.2.9 two-angle blend;
//   * DC: §7.13.2.12 IBP DC post-filter.
// Smooth, smooth-V, smooth-H and Paeth are unchanged by IBP.  For those modes
// the host entry point delegates to intra_modes::intra_predict().
//
// The device body is in ibp_predict_device.cuh; it is also compiled by the
// host-emulation parity test (same integer code, one emulated thread).

#include "../modes/intra_modes.h"

#include <cstdint>

#include <cuda_runtime.h>

#if defined(__CUDACC__)
#define AV2ENC_IBP_HD __host__ __device__
#else
#define AV2ENC_IBP_HD
#endif

namespace av2enc::cuda::ibp {

using intra_modes::IntraBlock;
using intra_modes::IntraMode;

// ---------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------

// All fields of the wrapped IntraBlock are used.  `enable_ibp` is the
// sequence-level §5.4.5 flag.  IBP itself has no per-block syntax bit.
struct IbpBlock {
  IntraBlock intra{};
  int enable_ibp = 0;
  int is_inter = 0;   // inter-intra prediction: §7.13.2.7 skips WAIP
  int use_dip = 0;    // only consulted by the DC applicability predicate
  int uv_is_cfl = 0;  // only consulted for chroma DC
};

struct IbpDecision {
  int valid = 0;
  int apply_ibp_edge_rules = 0;  // applyIbp = enable_ibp && not4x4
  int use_ibp = 0;               // two-angle blend
  int p_angle = 0;
  int second_angle = 0;
};

// Host/device mirror of the CPU decision.  `b.intra.mode` must be a valid
// IntraMode and `b.intra.angle_delta` in [-3, 3].
AV2ENC_IBP_HD inline IbpDecision ibp_decide(const IbpBlock& b) {
  IbpDecision d;
  const IntraBlock& in = b.intra;
  const int w = in.width;
  const int h = in.height;
  if (w < 4 || w > 64 || h < 4 || h > 64) {
    return d;
  }
  const int mode = static_cast<int>(in.mode);
  if (mode < 0 || mode >= intra_modes::kIntraModes) {
    return d;
  }
  if (in.angle_delta < -intra_modes::kMaxAngleDelta ||
      in.angle_delta > intra_modes::kMaxAngleDelta) {
    return d;
  }
  if (!intra_modes::is_directional_mode(mode)) {
    if (in.angle_delta != 0) {
      return d;
    }
    d.valid = 1;
    return d;
  }
  int p_angle = 0;
  if (b.is_inter != 0) {
    p_angle = intra_modes::mode_to_angle(mode) +
              in.angle_delta * intra_modes::kAngleStep;
  } else {
    p_angle = intra_modes::intra_pangle(mode, in.angle_delta, w, h);
  }
  if (p_angle <= 0 || p_angle >= 270) {
    return d;
  }
  const int not4x4 = !(w == 4 && h == 4);
  d.valid = 1;
  d.p_angle = p_angle;
  d.apply_ibp_edge_rules = (b.enable_ibp != 0) && not4x4;
  d.use_ibp = d.apply_ibp_edge_rules && ((in.angle_delta & 1) == 0) &&
              in.plane == 0 && (p_angle < 90 || p_angle > 180);
  if (d.use_ibp) {
    d.second_angle = p_angle < 90 ? p_angle + 180 : p_angle - 180;
  }
  return d;
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

// Predicts one block.  For DC and the eight directional modes this runs the
// IBP-aware kernel; for SMOOTH/SMOOTH_V/SMOOTH_H/PAETH it calls
// intra_modes::intra_predict() unchanged.
cudaError_t ibp_predict(const IbpBlock& block, std::uint8_t* pred,
                        cudaStream_t stream = nullptr);

// Device-resident batch.  One CUDA block per job.  Every job must be DC or
// directional (the modes IBP can affect); other modes return
// cudaErrorInvalidValue.  Jobs are independent.
struct IbpJob {
  IbpBlock block{};
  std::uint8_t* pred = nullptr;
};

cudaError_t ibp_predict_batch(const IbpJob* d_jobs, int job_count,
                              cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::ibp

#endif  // AV2ENC_CUDA_IBP_IBP_H_
