// Shared host/device arithmetic for the §7.14.4 DC/AC delta-Q quantizer.
//
// This is a transcription of `src/codec/transform/delta_q/quant.cpp`
// (`get_q`, `build_q2_matrix`, `quantize_block`, `dequantize_block`) with the
// same integer rounding and the same floating-point operation order.  The CPU
// reference remains the oracle; the emulation test compiles this header with
// the host compiler and compares every output.
//
// Device-visibility rules:
//   * every free function here is DQ_HD (__host__ __device__ under nvcc);
//   * no namespace-scope host object is read from device code (the §7.14.2
//     table is a function-local switch);
//   * no CUDA runtime or CPU-reference header is included.
#pragma once

#include <cmath>
#include <cstdint>

#include "deltaq_types.h"

#if defined(__CUDACC__)
#define DQ_HD __host__ __device__ __forceinline__
#else
#define DQ_HD inline
#endif

namespace av2enc::cuda::deltaq {
namespace core {

// §7.14.2 Ac_Qlookup[25].
DQ_HD int ac_qlookup(int index) {
  switch (index) {
    case 0: return 64;
    case 1: return 40;
    case 2: return 41;
    case 3: return 43;
    case 4: return 44;
    case 5: return 45;
    case 6: return 47;
    case 7: return 48;
    case 8: return 49;
    case 9: return 51;
    case 10: return 52;
    case 11: return 54;
    case 12: return 55;
    case 13: return 57;
    case 14: return 59;
    case 15: return 60;
    case 16: return 62;
    case 17: return 64;
    case 18: return 66;
    case 19: return 68;
    case 20: return 70;
    case 21: return 72;
    case 22: return 74;
    case 23: return 76;
    default: return 78;  // index 24
  }
}

DQ_HD int clip3(int lo, int hi, int value) {
  return value < lo ? lo : (value > hi ? hi : value);
}

// §7.14.2 qlookup().
DQ_HD int qlookup(int q) {
  if (q < 25) return ac_qlookup(q < 0 ? 0 : q);
  return ac_qlookup(((q - 1) % 24) + 1) << ((q - 1) / 24);
}

// §7.14.2 get_q().
DQ_HD int get_q(int qindex, int delta, int max_q = kMaxQ) {
  if (qindex == 0 && delta <= 0) return ac_qlookup(0);
  const int clamped = clip3(1, max_q, qindex + delta);
  return qlookup(clamped);
}

// §5.20.7.29 get_tx_class(): V_* and H_* are the 1-D classes; every other
// type (including IDTX == 9) is TX_CLASS_2D.
DQ_HD int tx_class_2d(int tx_type) { return tx_type >= 10 ? 0 : 1; }

DQ_HD int adjusted_width(int coded_w) {
  return coded_w < kMaxAdjustedDim ? coded_w : kMaxAdjustedDim;
}

DQ_HD int adjusted_height(int coded_h) {
  return coded_h < kMaxAdjustedDim ? coded_h : kMaxAdjustedDim;
}

DQ_HD int extra_tcq_shift(const Job& job) {
  return (job.allow_tcq != 0 && job.plane == 0 && job.lossless == 0 &&
          tx_class_2d(job.tx_type) != 0 && job.use_fsc == 0)
             ? 1
             : 0;
}

// §7.14.4 dqDenom shift: (coded pels > 256) + (coded pels > 1024) [+ TCQ].
DQ_HD int dequant_shift(const Job& job) {
  const int pels = job.coded_w * job.coded_h;
  return (pels > 256 ? 1 : 0) + (pels > 1024 ? 1 : 0) +
         extra_tcq_shift(job);
}

// §4.8 Round2 for the non-negative domain used by dequantization.
DQ_HD std::int32_t round2(std::int32_t value, int n) {
  if (n <= 0) return value;
  return static_cast<std::int32_t>(
      (static_cast<std::int64_t>(value) + (1LL << (n - 1))) >> n);
}

// std::llround is round-half-away-from-zero on both paths.
DQ_HD std::int32_t round_half_away(double value) {
#if defined(__CUDA_ARCH__)
  return static_cast<std::int32_t>(llround(value));
#else
  return static_cast<std::int32_t>(std::llround(value));
#endif
}

// §7.14.2 DC and AC quantizers for one plane from the effective qindex and the
// resolved (sequence base + frame) delta totals.
DQ_HD void plane_quant(const Job& job, int* dc_quant, int* ac_quant) {
  *dc_quant = get_q(job.qindex, job.delta_dc);
  *ac_quant = get_q(job.qindex, job.delta_ac);
}

// One §7.14.4 q2 element.  `qm_value` is the quantizer-matrix entry for this
// position, or -1 when no matrix is active.
DQ_HD std::int32_t q2_value(int quant, int qm_value) {
  if (qm_value < 0) return quant;
  return round2(quant * qm_value, 5);
}

DQ_HD bool valid_job(const Job& job) {
  if (job.plane < 0 || job.plane > 2) return false;
  if (job.coded_w <= 0 || job.coded_w > 64) return false;
  if (job.coded_h <= 0 || job.coded_h > 64) return false;
  if (job.tx_type < 0 || job.tx_type > 15) return false;
  if (job.qindex < 0 || job.qindex > kMaxQ) return false;
  return true;
}

// Returns false when the job's q2 matrix contains a non-positive entry (the
// CPU reference rejects that too).
DQ_HD bool q2_ready(const Job& job, const std::int32_t* batch_q2,
                    int dc_quant, int ac_quant) {
  if (job.lossless != 0) return true;
  if (job.q2_offset < 0) {
    return dc_quant > 0 && ac_quant > 0;
  }
  if (batch_q2 == nullptr) return false;
  const int tw = adjusted_width(job.coded_w);
  const int th = adjusted_height(job.coded_h);
  const std::int32_t* matrix = batch_q2 + job.q2_offset;
  for (int i = 0; i < tw * th; ++i) {
    if (matrix[i] <= 0) return false;
  }
  return true;
}

DQ_HD std::int32_t q2_at(const Job& job, const std::int32_t* batch_q2,
                         int dc_quant, int ac_quant, int index) {
  if (job.q2_offset >= 0) return batch_q2[job.q2_offset + index];
  const int q = (index == 0) ? dc_quant : ac_quant;  // i == 0 && j == 0
  return q2_value(q, -1);
}

// `coeff` -> `levels` for one job.  `batch_q2` may be null when
// `job.q2_offset < 0`.
DQ_HD void quantize_job(const Job& job, const double* coeff,
                        std::int32_t* levels, const std::int32_t* batch_q2) {
  const int tw = adjusted_width(job.coded_w);
  const int th = adjusted_height(job.coded_h);
  const int n = tw * th;
  if (job.lossless != 0) {
    for (int i = 0; i < n; ++i) levels[i] = round_half_away(coeff[i]);
    return;
  }
  int dc_quant = 0;
  int ac_quant = 0;
  plane_quant(job, &dc_quant, &ac_quant);
  // Mirrors quantize_block()'s expression order exactly:
  //   factor = coefficient_scale * 528 * (1 << extra_shift)
  //   level  = llround(coeff * factor / static_cast<double>(q2))
  const double factor =
      job.coefficient_scale * static_cast<double>(kQuantStep) *
      static_cast<double>(1 << extra_tcq_shift(job));
  for (int i = 0; i < n; ++i) {
    const std::int32_t q2 = q2_at(job, batch_q2, dc_quant, ac_quant, i);
    const double denom = static_cast<double>(q2);
    levels[i] = round_half_away(coeff[i] * factor / denom);
  }
}

// `levels` -> `dequant` for one job.
DQ_HD void dequantize_job(const Job& job, const std::int32_t* levels,
                          std::int32_t* dequant,
                          const std::int32_t* batch_q2) {
  const int tw = adjusted_width(job.coded_w);
  const int th = adjusted_height(job.coded_h);
  const int n = tw * th;
  if (job.lossless != 0) {
    for (int i = 0; i < n; ++i) dequant[i] = levels[i];
    return;
  }
  int dc_quant = 0;
  int ac_quant = 0;
  plane_quant(job, &dc_quant, &ac_quant);
  const int shift = dequant_shift(job);
  const std::int32_t denom = static_cast<std::int32_t>(1 << shift);
  const std::int32_t lo = -(1 << 15);
  const std::int32_t hi = (1 << 15) - 1;
  for (int i = 0; i < n; ++i) {
    const std::int32_t level = levels[i];
    const std::int64_t abs_level =
        level < 0 ? -static_cast<std::int64_t>(level) : level;
    const std::int32_t q2 = q2_at(job, batch_q2, dc_quant, ac_quant, i);
    const std::int64_t high = (abs_level * q2) & 0xFFFFFF;
    const std::int32_t dq =
        round2(static_cast<std::int32_t>(high), kQuantTableBits) / denom;
    const std::int32_t signed_dq = level < 0 ? -dq : dq;
    dequant[i] = clip3(lo, hi, signed_dq);
  }
}

}  // namespace core
}  // namespace av2enc::cuda::deltaq
