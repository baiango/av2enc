// Shared device core for CUDA TCQ / CCTX / parity hiding.
//
// This is the single transcription of the arithmetic that both the real
// kernels (tcq.cu, compiled by nvcc with -fmad=false) and the host-emulation
// parity test (tcq_emulation_test.cpp, compiled by the host compiler with
// -ffp-contract=off) execute.  It is deliberately:
//
//   * CUDA-intrinsic-free: no shared memory, no atomics, no warp primitives,
//     so the same code compiles as host C++20;
//   * table-free: the §5.20.7.27 state map and the CCTX matrix are switch
//     expressions, and every other constant is an enum value.  No host
//     `constexpr` array is odr-used from a kernel, which is what keeps the
//     device pass free of host-symbol references;
//   * allocation-free: all scratch (path choices, tail energy) is either an
//     automatic array or a pointer supplied by the batch descriptor, which
//     the workspace carved from the single av2enc::cuda::Arena.
//
// Every free function is TCQ_HD (__host__ __device__ under nvcc, `inline`
// otherwise).  build_gpu_test.sh audits that property.
//
// Numeric contract
// ----------------
// The DP is `tcq::quantize()` from src/encoder/tcq/tcq.cpp, line for line:
// same reverse-scan recurrence, same derived level bound, same tie-break
// order (cost, distortion, rate, |level|, signed quant), same EOB rule, same
// arithmetic order.  The rate callback is replaced by a caller-filled table
// (tcq_host.h) so an arbitrary CPU rate model is reproduced exactly instead
// of approximated by a device log2.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#include "cuda/tcq/tcq_types.h"

#if defined(__CUDACC__)
#include <math_constants.h>
#define TCQ_HD __host__ __device__ inline
#else
#define TCQ_HD inline
#endif

namespace av2enc::cuda::tcq::detail {

enum : std::int32_t {
  kStateCount = 8,
  kCctxTypeCount = 7,
  kCctxPrecBits = 8,
  kPhThresh = 4,
  kNumBaseLevels = 2,
  kCoeffBaseRange = 3,
  kMaxBaseBrRange = 6,
  kLfNumBaseLevels = 4,
};

// ---------------------------------------------------------------------------
// Portable exact math helpers
// ---------------------------------------------------------------------------

TCQ_HD double tcq_infinite() {
#if defined(__CUDA_ARCH__)
  return CUDART_INF;
#else
  return std::numeric_limits<double>::infinity();
#endif
}

TCQ_HD double tcq_fabs(double value) {
#if defined(__CUDACC__)
  return ::fabs(value);
#else
  return std::fabs(value);
#endif
}

TCQ_HD double tcq_floor(double value) {
#if defined(__CUDACC__)
  return ::floor(value);
#else
  return std::floor(value);
#endif
}

TCQ_HD double tcq_round(double value) {
#if defined(__CUDACC__)
  return ::round(value);
#else
  return std::round(value);
#endif
}

TCQ_HD long long tcq_llround(double value) {
#if defined(__CUDACC__)
  return ::llround(value);
#else
  return std::llround(value);
#endif
}

TCQ_HD bool tcq_isfinite(double value) {
#if defined(__CUDACC__)
  return isfinite(value) != 0;
#else
  return std::isfinite(value);
#endif
}

// tcq.cpp's abs_i32 clamps INT32_MIN instead of overflowing.
TCQ_HD std::int32_t tcq_abs(std::int32_t value) {
  if (value == std::numeric_limits<std::int32_t>::min()) {
    return std::numeric_limits<std::int32_t>::max();
  }
  return value < 0 ? -value : value;
}

TCQ_HD bool tcq_rate_valid(double rate) {
  return tcq_isfinite(rate) && rate >= 0.0;
}

// ---------------------------------------------------------------------------
// §5.20.7.27 TCQ state machine
// ---------------------------------------------------------------------------

// Tcq_Next_State[8][2] as a switch: no array storage in either pass.
TCQ_HD std::int32_t tcq_next_state(std::int32_t state, std::int32_t parity) {
  switch (state & 7) {
    case 0:
      return (parity & 1) != 0 ? 4 : 0;
    case 1:
      return (parity & 1) != 0 ? 0 : 4;
    case 2:
      return (parity & 1) != 0 ? 5 : 1;
    case 3:
      return (parity & 1) != 0 ? 1 : 5;
    case 4:
      return (parity & 1) != 0 ? 2 : 6;
    case 5:
      return (parity & 1) != 0 ? 6 : 2;
    case 6:
      return (parity & 1) != 0 ? 3 : 7;
    default:
      return (parity & 1) != 0 ? 7 : 3;
  }
}

TCQ_HD std::int32_t tcq_state_q0(std::int32_t state) {
  return (state >> 1) & 1;
}

TCQ_HD std::int32_t tcq_state_cdf_index(std::int32_t state) {
  return (state >> 1) & 1;
}

// Syntax base level for a full level, including the parity-dependent choice
// between `pre_max` and `pre_max - 1` above the escape threshold.
TCQ_HD std::int32_t tcq_syntax_level_for(std::int32_t full_level,
                                         std::int32_t pre_max) {
  if (full_level < pre_max - 1) return full_level;
  return ((full_level ^ pre_max) & 1) != 0 ? pre_max - 1 : pre_max;
}

// `read_quant` escape increment x for a full level.
TCQ_HD std::int32_t tcq_syntax_escape_for(std::int32_t full_level,
                                          std::int32_t pre_max) {
  const std::int32_t level = tcq_syntax_level_for(full_level, pre_max);
  if (level < pre_max - 1) return 0;
  return (full_level - level) / 2;
}

// §7.14.4 dequant shift interaction.
TCQ_HD std::int32_t tcq_dequant_shift_extra(bool allow_tcq, std::int32_t plane,
                                            bool lossless, bool tx_class_2d,
                                            bool use_fsc) {
  return (allow_tcq && plane == 0 && !lossless && tx_class_2d && !use_fsc)
             ? 1
             : 0;
}

TCQ_HD std::int32_t tcq_dequant_shift(std::int32_t coded_pels, bool allow_tcq,
                                      std::int32_t plane, bool lossless,
                                      bool tx_class_2d, bool use_fsc) {
  return (coded_pels > 256 ? 1 : 0) + (coded_pels > 1024 ? 1 : 0) +
         tcq_dequant_shift_extra(allow_tcq, plane, lossless, tx_class_2d,
                                 use_fsc);
}

// ---------------------------------------------------------------------------
// §5.20.7.27 level limits / pre_max
// ---------------------------------------------------------------------------
//
// Mirrors tcq::parity::level_limits().  `tx_class` is 0 = TX_CLASS_2D,
// 1 = TX_CLASS_HORIZ, 2 = TX_CLASS_VERT.  For luma 2D this reduces to the
// `tcq.h` convention: pre_max = 8 for (row + col) < 4, 6 otherwise.

struct TcQLevelLimits {
  std::int32_t base_levels;
  std::int32_t base_max;
  std::int32_t br_max;
  std::int32_t pre_max;
  bool is_lf;
};

TCQ_HD bool tcq_is_low_frequency(std::int32_t row, std::int32_t col,
                                 std::int32_t tx_class, std::int32_t plane) {
  if (tx_class == 0) {
    return plane == 0 ? ((row + col) < 4) : ((row + col) < 1);
  }
  if (tx_class == 1) {
    return plane == 0 ? (col < 2) : (col < 1);
  }
  return plane == 0 ? (row < 2) : (row < 1);
}

TCQ_HD TcQLevelLimits tcq_level_limits(std::int32_t row, std::int32_t col,
                                       std::int32_t tx_class,
                                       std::int32_t plane) {
  TcQLevelLimits out;
  out.is_lf = tcq_is_low_frequency(row, col, tx_class, plane);
  out.base_levels = out.is_lf ? kLfNumBaseLevels : kNumBaseLevels;
  out.base_max = out.is_lf ? 5 : 3;
  out.br_max = (out.is_lf && plane > 0) ? 0 : kCoeffBaseRange;
  out.pre_max = out.base_max + out.br_max;
  return out;
}

// ---------------------------------------------------------------------------
// tcq.cpp bound_for()/level_bound()
// ---------------------------------------------------------------------------

// Returns false when the ratio is not finite (tcq.cpp rejects that block).
TCQ_HD bool tcq_level_bound(double original, double quant_step,
                            std::int32_t level_cap, bool allow_zero_to_one,
                            std::int32_t* bound, bool* truncated) {
  const double ratio = tcq_fabs(original) / quant_step;
  if (!tcq_isfinite(ratio)) return false;
  double raw = tcq_floor(2.0 * ratio) + 2.0;
  if (original == 0.0 && !allow_zero_to_one) {
    raw = 0.0;
  }
  const double max_level =
      static_cast<double>(std::numeric_limits<std::int32_t>::max() - 1);
  if (raw > max_level) raw = max_level;
  std::int64_t value = static_cast<std::int64_t>(raw);
  if (!(original == 0.0 && !allow_zero_to_one)) {
    value = value > 1 ? value : 1;
    const std::int64_t with_margin = value + 64;
    const std::int64_t cap =
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max() - 1);
    value = with_margin < cap ? with_margin : cap;
  }
  if (value < 0) value = 0;
  bool clipped = false;
  if (level_cap > 0 && value > static_cast<std::int64_t>(level_cap)) {
    value = level_cap;
    clipped = true;
  }
  *bound = static_cast<std::int32_t>(value);
  *truncated = clipped;
  return true;
}

// ---------------------------------------------------------------------------
// Encoder trellis (tcq::quantize)
// ---------------------------------------------------------------------------

struct TcQChoice {
  double cost;
  double distortion;
  double rate_bits;
  std::int32_t base_level;
  std::int32_t quant_level;
  std::int32_t next_state;
  std::int32_t abs_level;
};

TCQ_HD TcQChoice tcq_choice_infinite() {
  TcQChoice choice;
  choice.cost = tcq_infinite();
  choice.distortion = 0.0;
  choice.rate_bits = 0.0;
  choice.base_level = 0;
  choice.quant_level = 0;
  choice.next_state = 0;
  choice.abs_level = 0;
  return choice;
}

TCQ_HD TcQChoice tcq_choice_zero() {
  TcQChoice choice;
  choice.cost = 0.0;
  choice.distortion = 0.0;
  choice.rate_bits = 0.0;
  choice.base_level = 0;
  choice.quant_level = 0;
  choice.next_state = 0;
  choice.abs_level = 0;
  return choice;
}

// tcq.cpp choice_better(): a total order over the chosen suffix.
TCQ_HD bool tcq_choice_better(const TcQChoice& a, const TcQChoice& b) {
  if (a.cost != b.cost) return a.cost < b.cost;
  if (a.distortion != b.distortion) return a.distortion < b.distortion;
  if (a.rate_bits != b.rate_bits) return a.rate_bits < b.rate_bits;
  if (a.abs_level != b.abs_level) return a.abs_level < b.abs_level;
  return a.quant_level < b.quant_level;
}

// One complete block.  Returns 0 on success and -1 for a rejected job (the
// per-job analogue of `tcq::quantize()` returning false).
TCQ_HD std::int32_t tcq_run_job(const TcqDeviceBatch& batch,
                                std::int32_t job_index) {
  const TcqJob job = batch.jobs[job_index];
  const std::int32_t count = job.num_coefficients;
  std::int32_t* base_out = batch.base_levels + job.output_offset;
  std::int32_t* quant_out = batch.quant_levels + job.output_offset;
  std::int32_t* states_out = batch.states + job.output_offset;
  batch.eob[job_index] = 0;
  batch.distortion[job_index] = 0.0;
  batch.rate_bits[job_index] = 0.0;
  batch.cost[job_index] = 0.0;
  batch.max_level_searched[job_index] = 0;
  batch.level_range_truncated[job_index] = 0;
  batch.status[job_index] = -1;
  if (count < 0 || job.table_levels <= 0 ||
      job.table_levels > kMaxLevelsCap) {
    return -1;
  }
  for (std::int32_t c = 0; c < count; ++c) {
    base_out[c] = 0;
    quant_out[c] = 0;
    states_out[c] = 0;
  }
  if (count == 0) {
    batch.status[job_index] = 0;
    return 0;
  }
  if (!tcq_isfinite(job.quant_step) || !(job.quant_step > 0.0) ||
      !tcq_isfinite(job.lambda) || job.lambda < 0.0) {
    return -1;
  }

  const double* original = batch.original + job.coefficient_offset;
  const std::int32_t* pre_max = batch.pre_max + job.coefficient_offset;
  const double* normal_rate = batch.normal_rate + job.rate_offset;
  const double* eob_coeff_rate = batch.eob_coeff_rate + job.rate_offset;
  const double* block_eob_rate = batch.block_eob_rate + job.eob_table_offset;
  const std::int32_t levels_per_coefficient = job.table_levels;
  std::int32_t* path_base = batch.path_base + job.path_offset;
  std::int32_t* path_quant = batch.path_quant + job.path_offset;
  std::int32_t* path_next = batch.path_next + job.path_offset;
  double* tail = batch.tail + job.tail_offset;
  const double lambda = job.lambda;
  const double half_step = job.quant_step * 0.5;

  for (std::int32_t c = 0; c < count; ++c) {
    if (!tcq_isfinite(original[c])) return -1;
  }
  // tcq.cpp computes tail[] in reverse; the arithmetic order is preserved.
  tail[count] = 0.0;
  for (std::int32_t c = count - 1; c >= 0; --c) {
    tail[c] = original[c] * original[c] + tail[c + 1];
  }

  std::int32_t max_level_searched = 0;
  for (std::int32_t c = 0; c < count; ++c) {
    std::int32_t bound = 0;
    bool clipped = false;
    if (!tcq_level_bound(original[c], job.quant_step, job.level_cap,
                         job.allow_zero_to_one != 0, &bound, &clipped)) {
      return -1;
    }
    if (bound >= levels_per_coefficient) {
      // The rate table does not cover the derived range.  The caller sizes it
      // from tcq_level_bound(); reject instead of clamping silently.
      return -1;
    }
    if (bound > max_level_searched) max_level_searched = bound;
    if (clipped) batch.level_range_truncated[job_index] = 1;
  }

  const double all_zero_rate = block_eob_rate[0];
  if (!tcq_rate_valid(all_zero_rate)) return -1;
  TcQChoice best = tcq_choice_zero();
  best.cost = tail[0] + all_zero_rate * lambda;
  best.distortion = tail[0];
  best.rate_bits = all_zero_rate;
  std::int32_t start_position = -1;

  TcQChoice prev[kStateCount];
  TcQChoice cur[kStateCount];
  for (std::int32_t s = 0; s < kStateCount; ++s) {
    prev[s] = tcq_choice_zero();
    cur[s] = tcq_choice_infinite();
  }

  for (std::int32_t c = 0; c < count; ++c) {
    const double original_value = original[c];
    const std::int32_t sign = original_value < 0.0 ? -1 : 1;
    std::int32_t bound = 0;
    bool bound_clipped = false;
    (void)tcq_level_bound(original_value, job.quant_step, job.level_cap,
                          job.allow_zero_to_one != 0, &bound, &bound_clipped);
    const std::int32_t coefficient_pre_max = pre_max[c] > 0 ? pre_max[c] : 6;

    for (std::int32_t s = 0; s < kStateCount; ++s) {
      TcQChoice best_choice = tcq_choice_infinite();
      for (std::int32_t q = 0; q <= bound; ++q) {
        const bool nonzero = q > 0;
        const std::int32_t syntax_level =
            q == 0 ? 0 : tcq_syntax_level_for(q, coefficient_pre_max);
        const std::int32_t q0 = nonzero ? tcq_state_q0(s) : 0;
        const std::int32_t magnitude = nonzero ? 2 * q - q0 : 0;
        const std::int32_t base = nonzero ? sign * q : 0;
        const std::int32_t quant = nonzero ? sign * magnitude : 0;
        const std::int32_t next = tcq_next_state(s, syntax_level & 1);
        const double difference =
            original_value - static_cast<double>(quant) * half_step;
        const double local_dist = difference * difference;
        const double local_rate =
            normal_rate[(c * kStateCount + s) * levels_per_coefficient + q];
        if (!tcq_rate_valid(local_rate)) return -1;
        const TcQChoice& suffix = prev[next];
        TcQChoice candidate;
        candidate.cost = local_dist + lambda * local_rate + suffix.cost;
        candidate.distortion = local_dist + suffix.distortion;
        candidate.rate_bits = local_rate + suffix.rate_bits;
        candidate.base_level = base;
        candidate.quant_level = quant;
        candidate.next_state = next;
        candidate.abs_level = q;
        if (tcq_choice_better(candidate, best_choice)) best_choice = candidate;
      }
      cur[s] = best_choice;
    }

    // Start-of-EOB pass.  Running it here (with dp[c-1] still in `prev`) is
    // value-identical to tcq.cpp's separate loop and keeps the suffix
    // arithmetic order: local + tail + lambda*(local_rate + eob_rate) + suffix.
    for (std::int32_t q = 1; q <= bound; ++q) {
      const std::int32_t syntax_level =
          tcq_syntax_level_for(q, coefficient_pre_max);
      const std::int32_t q0 = tcq_state_q0(0);
      const std::int32_t magnitude = 2 * q - q0;
      const std::int32_t base = sign * q;
      const std::int32_t quant = sign * magnitude;
      const std::int32_t next = tcq_next_state(0, syntax_level & 1);
      const double difference =
          original_value - static_cast<double>(quant) * half_step;
      const double local_dist = difference * difference;
      const double local_rate =
          eob_coeff_rate[(c * kStateCount) * levels_per_coefficient + q];
      if (!tcq_rate_valid(local_rate)) return -1;
      const double eob_rate = block_eob_rate[c + 1];
      if (!tcq_rate_valid(eob_rate)) return -1;
      const TcQChoice& suffix = prev[next];
      const double tail_distortion = tail[c + 1];
      TcQChoice candidate;
      candidate.cost = local_dist + tail_distortion +
                       lambda * (local_rate + eob_rate) + suffix.cost;
      candidate.distortion = local_dist + tail_distortion + suffix.distortion;
      candidate.rate_bits = local_rate + eob_rate + suffix.rate_bits;
      candidate.base_level = base;
      candidate.quant_level = quant;
      candidate.next_state = next;
      candidate.abs_level = q;
      const bool better =
          tcq_choice_better(candidate, best) ||
          (start_position >= 0 && !tcq_choice_better(best, candidate) &&
           c < start_position);
      if (better) {
        best = candidate;
        start_position = c;
      }
    }

    for (std::int32_t s = 0; s < kStateCount; ++s) {
      const std::int32_t slot = c * kStateCount + s;
      path_base[slot] = cur[s].base_level;
      path_quant[slot] = cur[s].quant_level;
      path_next[slot] = cur[s].next_state;
      prev[s] = cur[s];
    }
  }

  batch.distortion[job_index] = best.distortion;
  batch.rate_bits[job_index] = best.rate_bits;
  batch.cost[job_index] = best.cost;
  batch.max_level_searched[job_index] = max_level_searched;
  if (start_position < 0) {
    batch.eob[job_index] = 0;
    batch.status[job_index] = 0;
    return 0;
  }

  batch.eob[job_index] = start_position + 1;
  base_out[start_position] = best.base_level;
  quant_out[start_position] = best.quant_level;
  states_out[start_position] = 0;
  std::int32_t state = best.next_state;
  for (std::int32_t c = start_position - 1; c >= 0; --c) {
    const std::int32_t slot = c * kStateCount + state;
    base_out[c] = path_base[slot];
    quant_out[c] = path_quant[slot];
    states_out[c] = state;
    state = path_next[slot];
  }
  batch.status[job_index] = 0;
  return 0;
}

// ---------------------------------------------------------------------------
// CCTX (§5.4.8, §7.14.6) and the seven-candidate scalar search
// ---------------------------------------------------------------------------

TCQ_HD std::int32_t tcq_clip3(std::int32_t low, std::int32_t high,
                              std::int32_t value) {
  return value < low ? low : (value > high ? high : value);
}

// §4.8 Round2Signed: symmetric rounding, not an arithmetic shift.
TCQ_HD std::int32_t tcq_round2_signed(std::int64_t value, int n) {
  if (n <= 0) return static_cast<std::int32_t>(value);
  if (value >= 0) {
    return static_cast<std::int32_t>((value + (1LL << (n - 1))) >> n);
  }
  return static_cast<std::int32_t>(-((-value + (1LL << (n - 1))) >> n));
}

// Cctx_Mtx[CCTX_TYPES - 1][2] as a switch (no array in either pass).
TCQ_HD bool tcq_cctx_matrix(std::int32_t angle, std::int32_t* c,
                            std::int32_t* s) {
  switch (angle) {
    case 0:
      *c = 181;
      *s = 181;
      return true;
    case 1:
      *c = 222;
      *s = 128;
      return true;
    case 2:
      *c = 128;
      *s = 222;
      return true;
    case 3:
      *c = 181;
      *s = -181;
      return true;
    case 4:
      *c = 222;
      *s = -128;
      return true;
    case 5:
      *c = 128;
      *s = -222;
      return true;
    default:
      return false;
  }
}

TCQ_HD std::int32_t tcq_cctx_dequant_component(std::int32_t u, std::int32_t v,
                                               std::int32_t plane,
                                               std::int32_t type,
                                               std::int32_t bitdepth) {
  if (type == 0) return plane == 1 ? u : v;
  std::int32_t c = 0;
  std::int32_t s = 0;
  if (!tcq_cctx_matrix(type - 1, &c, &s)) return plane == 1 ? u : v;
  const std::int32_t c_u = plane == 1 ? c : s;
  const std::int32_t c_v = plane == 1 ? -s : c;
  const std::int64_t value =
      static_cast<std::int64_t>(u) * c_u + static_cast<std::int64_t>(v) * c_v;
  const std::int32_t rounded = tcq_round2_signed(value, kCctxPrecBits);
  const std::int32_t low = -(1 << (bitdepth + 7));
  const std::int32_t high = (1 << (bitdepth + 7)) - 1;
  return tcq_clip3(low, high, rounded);
}

struct TcQCctxPair {
  std::int32_t u;
  std::int32_t v;
};

TCQ_HD TcQCctxPair tcq_cctx_dequant_pair(std::int32_t u, std::int32_t v,
                                         std::int32_t type,
                                         std::int32_t bitdepth) {
  TcQCctxPair out;
  out.u = tcq_cctx_dequant_component(u, v, 1, type, bitdepth);
  out.v = tcq_cctx_dequant_component(u, v, 2, type, bitdepth);
  return out;
}

// Exact inverse of the §7.14.6 matrix, including the 256/(c^2+s^2) scale.
TCQ_HD void tcq_cctx_forward_rotate(double u, double v, std::int32_t type,
                                    double* out_u, double* out_v) {
  if (type == 0) {
    *out_u = u;
    *out_v = v;
    return;
  }
  std::int32_t ci = 0;
  std::int32_t si = 0;
  if (!tcq_cctx_matrix(type - 1, &ci, &si)) {
    *out_u = u;
    *out_v = v;
    return;
  }
  const double c = static_cast<double>(ci);
  const double s = static_cast<double>(si);
  const double denom = c * c + s * s;
  const double scale = 256.0 / denom;
  *out_u = (c * u + s * v) * scale;
  *out_v = (-s * u + c * v) * scale;
}

TCQ_HD std::int32_t tcq_run_cctx_job(const CctxDeviceBatch& batch,
                                     std::int32_t job_index) {
  const CctxJob job = batch.jobs[job_index];
  const std::int32_t count = job.num_coefficients;
  const std::int32_t types = job.enabled != 0 ? kCctxTypeCount : 1;
  batch.status[job_index] = -1;
  batch.best_type[job_index] = 0;
  batch.best_distortion[job_index] = 0.0;
  batch.best_eob_u[job_index] = 0;
  batch.best_eob_v[job_index] = 0;
  if (count < 0 || job.bitdepth < 8 || job.bitdepth > 16 ||
      !tcq_isfinite(job.quant_step_u) || !(job.quant_step_u > 0.0) ||
      !tcq_isfinite(job.quant_step_v) || !(job.quant_step_v > 0.0)) {
    return -1;
  }
  const double* u_coeff = batch.u_coeff + job.coefficient_offset;
  const double* v_coeff = batch.v_coeff + job.coefficient_offset;
  for (std::int32_t c = 0; c < count; ++c) {
    if (!tcq_isfinite(u_coeff[c]) || !tcq_isfinite(v_coeff[c])) return -1;
  }

  for (std::int32_t t = 0; t < kCctxTypeCount; ++t) {
    batch.candidate_type[job_index * kCctxTypeCount + t] = -1;
    batch.candidate_distortion[job_index * kCctxTypeCount + t] = 0.0;
    batch.candidate_eob_u[job_index * kCctxTypeCount + t] = 0;
    batch.candidate_eob_v[job_index * kCctxTypeCount + t] = 0;
    std::int32_t* u_levels =
        batch.u_levels + job.candidate_offset + t * count;
    std::int32_t* v_levels =
        batch.v_levels + job.candidate_offset + t * count;
    for (std::int32_t c = 0; c < count; ++c) {
      u_levels[c] = 0;
      v_levels[c] = 0;
    }
  }

  const std::int32_t int32_min_plus_one =
      std::numeric_limits<std::int32_t>::min() + 1;
  const std::int32_t int32_max = std::numeric_limits<std::int32_t>::max();
  std::int32_t best_index = 0;
  double best_distortion = 0.0;
  for (std::int32_t t = 0; t < types; ++t) {
    std::int32_t* u_levels =
        batch.u_levels + job.candidate_offset + t * count;
    std::int32_t* v_levels =
        batch.v_levels + job.candidate_offset + t * count;
    double distortion = 0.0;
    std::int32_t eob_u = 0;
    std::int32_t eob_v = 0;
    for (std::int32_t i = 0; i < count; ++i) {
      double rot_u = 0.0;
      double rot_v = 0.0;
      tcq_cctx_forward_rotate(u_coeff[i], v_coeff[i], t, &rot_u, &rot_v);
      const double u_q = tcq_round(rot_u / job.quant_step_u);
      const double v_q = tcq_round(rot_v / job.quant_step_v);
      const double u_clamped =
          u_q < static_cast<double>(int32_min_plus_one)
              ? static_cast<double>(int32_min_plus_one)
              : (u_q > static_cast<double>(int32_max)
                     ? static_cast<double>(int32_max)
                     : u_q);
      const double v_clamped =
          v_q < static_cast<double>(int32_min_plus_one)
              ? static_cast<double>(int32_min_plus_one)
              : (v_q > static_cast<double>(int32_max)
                     ? static_cast<double>(int32_max)
                     : v_q);
      const std::int32_t u_level = static_cast<std::int32_t>(u_clamped);
      const std::int32_t v_level = static_cast<std::int32_t>(v_clamped);
      u_levels[i] = u_level;
      v_levels[i] = v_level;
      if (u_level != 0) eob_u = i + 1;
      if (v_level != 0) eob_v = i + 1;

      // Decoder arithmetic: integer dequant, then the §7.14.6 rotation.
      const std::int32_t du = static_cast<std::int32_t>(
          tcq_llround(static_cast<double>(u_level) * job.quant_step_u));
      const std::int32_t dv = static_cast<std::int32_t>(
          tcq_llround(static_cast<double>(v_level) * job.quant_step_v));
      const TcQCctxPair out = tcq_cctx_dequant_pair(du, dv, t, job.bitdepth);
      const double eu = u_coeff[i] - static_cast<double>(out.u);
      const double ev = v_coeff[i] - static_cast<double>(out.v);
      distortion += eu * eu + ev * ev;
    }
    batch.candidate_type[job_index * kCctxTypeCount + t] = t;
    batch.candidate_distortion[job_index * kCctxTypeCount + t] = distortion;
    batch.candidate_eob_u[job_index * kCctxTypeCount + t] = eob_u;
    batch.candidate_eob_v[job_index * kCctxTypeCount + t] = eob_v;
    if (t == 0 || distortion < best_distortion) {
      best_index = t;
      best_distortion = distortion;
    }
  }
  batch.best_type[job_index] = best_index;
  batch.best_distortion[job_index] = best_distortion;
  batch.best_eob_u[job_index] =
      batch.candidate_eob_u[job_index * kCctxTypeCount + best_index];
  batch.best_eob_v[job_index] =
      batch.candidate_eob_v[job_index * kCctxTypeCount + best_index];
  batch.status[job_index] = 0;
  return 0;
}

// ---------------------------------------------------------------------------
// Parity hiding: minimum-distortion level choice
// ---------------------------------------------------------------------------
//
// This transcribes the non-normative candidate set of
// tcq::parity::quantize() for QuantizeMode::kMinDistortion:
//
//   * seed unchanged when the block is not hidable or already satisfies
//     |DC| & 1 == sumAbs1;
//   * otherwise DC +/-1 (both signs when the DC is zero) and each AC +/-1 that
//     flips the capped parity min(level, 6) & 1, plus the fourth-nonzero-to-
//     zero case that legally turns hiding off;
//   * blocks above `exhaustive_max_area` keep the best `pruned_ac_candidates`
//     AC moves ordered by (local distortion delta, scan index descending),
//     exactly the CPU stable_sort + keep rule;
//   * first candidate wins exact distortion ties.
//
// The §9.3 static rate model and hence QuantizeMode::kRateDistortion stay
// host-side; see README.md for that seam.

struct TcQParityEval {
  double distortion;
  std::int32_t eob;
  std::int32_t hidden;
  bool feasible;
};

TCQ_HD std::int32_t tcq_parity_level_at(const std::int32_t* seed,
                                        std::int32_t change_index,
                                        std::int32_t change_level,
                                        std::int32_t index) {
  return index == change_index ? change_level : seed[index];
}

// Recomputes distortion, EOB, isHidden and the hidden-DC parity constraint for
// the seed with one signed level replaced (change_index < 0 means the seed).
TCQ_HD TcQParityEval tcq_parity_evaluate(
    const double* original, const std::int32_t* seed, const std::int32_t* scan,
    std::int32_t count, std::int32_t w, std::int32_t tx_class,
    std::int32_t plane, bool header_ok, double quant_step,
    std::int32_t change_index, std::int32_t change_level) {
  TcQParityEval out;
  out.distortion = 0.0;
  out.eob = 0;
  out.hidden = 0;
  out.feasible = true;

  for (std::int32_t c = count - 1; c >= 0; --c) {
    if (tcq_parity_level_at(seed, change_index, change_level, c) != 0) {
      out.eob = c + 1;
      break;
    }
  }
  for (std::int32_t i = 0; i < count; ++i) {
    const std::int32_t level =
        tcq_parity_level_at(seed, change_index, change_level, i);
    const double error =
        original[i] - static_cast<double>(level) * quant_step;
    out.distortion += error * error;
  }

  std::int32_t sum_abs1 = 0;
  std::int32_t num_nz = 0;
  if (header_ok && out.eob > kPhThresh) {
    for (std::int32_t c = out.eob - 1; c >= 1; --c) {
      const std::int32_t pos = scan[c];
      const std::int32_t row = pos / w;
      const std::int32_t col = pos % w;
      const TcQLevelLimits limits =
          tcq_level_limits(row, col, tx_class, plane);
      const std::int32_t level =
          tcq_abs(tcq_parity_level_at(seed, change_index, change_level, c));
      const std::int32_t syntax_level =
          level < limits.pre_max ? level : limits.pre_max;
      const std::int32_t capped =
          syntax_level < kMaxBaseBrRange ? syntax_level : kMaxBaseBrRange;
      sum_abs1 ^= capped & 1;
      if (level != 0) ++num_nz;
    }
    out.hidden = num_nz >= kPhThresh ? 1 : 0;
    if (out.hidden != 0) {
      const std::int32_t dc =
          tcq_abs(tcq_parity_level_at(seed, change_index, change_level, 0));
      if ((dc & 1) != sum_abs1) out.feasible = false;
    }
  }

  // derive_syntax() rejects an escape value above 1 << 20.  That is the only
  // other way the CPU candidate evaluation can fail for finite levels.
  if (out.feasible) {
    for (std::int32_t c = 0; c < out.eob; ++c) {
      const std::int32_t pos = scan[c];
      const std::int32_t row = pos / w;
      const std::int32_t col = pos % w;
      const TcQLevelLimits limits =
          tcq_level_limits(row, col, tx_class, plane);
      const std::int32_t level =
          tcq_abs(tcq_parity_level_at(seed, change_index, change_level, c));
      std::int32_t escape_x = 0;
      if (out.hidden != 0 && c == 0) {
        const std::int32_t q_index = level >> 1;
        escape_x = q_index > kNumBaseLevels ? q_index - 3 : 0;
      } else {
        const std::int32_t syntax_level =
            level < limits.pre_max ? level : limits.pre_max;
        escape_x =
            syntax_level >= limits.pre_max ? level - syntax_level : 0;
      }
      if (escape_x > (1 << 20)) {
        out.feasible = false;
        break;
      }
    }
  }
  return out;
}

struct TcQParityCandidate {
  double local_delta;
  std::int32_t scan_index;
  std::int32_t level;
  std::int32_t delta;
};

TCQ_HD bool tcq_parity_candidate_less(const TcQParityCandidate& a,
                                      const TcQParityCandidate& b) {
  if (a.local_delta != b.local_delta) return a.local_delta < b.local_delta;
  return a.scan_index > b.scan_index;
}

// tcq.cpp make_delta_level().
TCQ_HD bool tcq_make_delta_level(std::int32_t level, std::int32_t delta,
                                 std::int32_t* out) {
  const std::int32_t magnitude = tcq_abs(level);
  const std::int32_t next = magnitude + delta;
  if (next < 0) return false;
  *out = level < 0 ? -next : next;
  return true;
}

struct TcQParitySearch {
  const double* original;
  const std::int32_t* seed;
  const std::int32_t* scan;
  std::int32_t count;
  std::int32_t w;
  std::int32_t tx_class;
  std::int32_t plane;
  bool header_ok;
  double quant_step;
  bool have_best;
  double best_distortion;
  std::int32_t best_eob;
  std::int32_t best_hidden;
  std::int32_t best_index;
  std::int32_t best_level;
  std::int32_t best_delta;
};

TCQ_HD void tcq_parity_consider(TcQParitySearch* search, std::int32_t index,
                                std::int32_t level, std::int32_t delta) {
  const TcQParityEval ev = tcq_parity_evaluate(
      search->original, search->seed, search->scan, search->count, search->w,
      search->tx_class, search->plane, search->header_ok, search->quant_step,
      index, level);
  if (!ev.feasible) return;
  // tcq::parity::quantize() keeps the first candidate on an exact tie.
  if (!search->have_best || ev.distortion < search->best_distortion) {
    search->have_best = true;
    search->best_distortion = ev.distortion;
    search->best_eob = ev.eob;
    search->best_hidden = ev.hidden;
    search->best_index = index;
    search->best_level = level;
    search->best_delta = delta;
  }
}

TCQ_HD std::int32_t tcq_run_parity_job(const ParityDeviceBatch& batch,
                                       std::int32_t job_index) {
  const ParityJob job = batch.jobs[job_index];
  const std::int32_t count = job.num_coefficients;
  std::int32_t* levels_out = batch.levels + job.output_offset;
  batch.eob[job_index] = 0;
  batch.hidden[job_index] = 0;
  batch.hidable[job_index] = 0;
  batch.hiding_considered[job_index] = 0;
  batch.hiding_turned_off[job_index] = 0;
  batch.adjusted_scan_index[job_index] = -1;
  batch.adjusted_delta[job_index] = 0;
  batch.distortion[job_index] = 0.0;
  batch.status[job_index] = -1;
  if (count <= 0 || job.w <= 0 || job.tx_class < 0 || job.tx_class > 2 ||
      job.plane < 0 || job.plane > 2 || !tcq_isfinite(job.quant_step) ||
      !(job.quant_step > 0.0) || job.exhaustive_max_area < 0 ||
      job.pruned_ac_candidates < 0 || job.pruned_ac_candidates > 8) {
    return -1;
  }
  const double* original = batch.original + job.coefficient_offset;
  const std::int32_t* seed = batch.seed_levels + job.coefficient_offset;
  const std::int32_t* scan = batch.scan + job.scan_offset;
  for (std::int32_t c = 0; c < count; ++c) {
    if (!tcq_isfinite(original[c])) return -1;
    levels_out[c] = seed[c];
  }

  const bool header_ok = job.allow_parity_hiding != 0 &&
                         job.lossless == 0 && job.allow_tcq == 0 &&
                         job.tx_type_is_idtx == 0 && job.use_fsc == 0 &&
                         job.plane == 0;

  // tcq::parity::analyze_ac() over the seed.
  std::int32_t eob = 0;
  for (std::int32_t c = count - 1; c >= 0; --c) {
    if (seed[c] != 0) {
      eob = c + 1;
      break;
    }
  }
  std::int32_t sum_abs1 = 0;
  std::int32_t num_nz = 0;
  for (std::int32_t c = eob - 1; c >= 1; --c) {
    const std::int32_t pos = scan[c];
    const std::int32_t row = pos / job.w;
    const std::int32_t col = pos % job.w;
    const TcQLevelLimits limits =
        tcq_level_limits(row, col, job.tx_class, job.plane);
    const std::int32_t level = tcq_abs(seed[c]);
    const std::int32_t syntax_level =
        level < limits.pre_max ? level : limits.pre_max;
    const std::int32_t capped =
        syntax_level < kMaxBaseBrRange ? syntax_level : kMaxBaseBrRange;
    sum_abs1 ^= capped & 1;
    if (level != 0) ++num_nz;
  }
  const std::int32_t dc_parity = tcq_abs(seed[0]) & 1;
  const bool hidable = header_ok && eob > kPhThresh && num_nz >= kPhThresh;
  batch.hidable[job_index] = hidable ? 1 : 0;
  batch.hiding_considered[job_index] = hidable ? 1 : 0;

  TcQParitySearch search;
  search.original = original;
  search.seed = seed;
  search.scan = scan;
  search.count = count;
  search.w = job.w;
  search.tx_class = job.tx_class;
  search.plane = job.plane;
  search.header_ok = header_ok;
  search.quant_step = job.quant_step;
  search.have_best = false;
  search.best_distortion = 0.0;
  search.best_eob = 0;
  search.best_hidden = 0;
  search.best_index = -1;
  search.best_level = 0;
  search.best_delta = 0;

  if (!hidable || dc_parity == sum_abs1) {
    tcq_parity_consider(&search, -1, 0, 0);
  } else {
    // DC candidates: delta +1 then -1 (both signs when the seed DC is zero).
    for (std::int32_t d = 0; d < 2; ++d) {
      const std::int32_t delta = d == 0 ? 1 : -1;
      std::int32_t next = 0;
      if (!tcq_make_delta_level(seed[0], delta, &next)) continue;
      if ((tcq_abs(next) & 1) != sum_abs1) continue;
      tcq_parity_consider(&search, 0, next, delta);
      if (seed[0] == 0) tcq_parity_consider(&search, 0, -next, delta);
    }

    // AC candidates in descending scan order, delta +1 then -1.
    const bool exhaustive = count <= job.exhaustive_max_area;
    TcQParityCandidate top[8];
    std::int32_t kept = 0;
    const std::int32_t keep_limit =
        job.pruned_ac_candidates < 8 ? job.pruned_ac_candidates : 8;
    for (std::int32_t c = eob - 1; c >= 1; --c) {
      const std::int32_t old_level = seed[c];
      if (old_level == 0) continue;
      const std::int32_t pos = scan[c];
      const std::int32_t row = pos / job.w;
      const std::int32_t col = pos % job.w;
      const TcQLevelLimits limits =
          tcq_level_limits(row, col, job.tx_class, job.plane);
      const std::int32_t old_abs = tcq_abs(old_level);
      const std::int32_t old_syntax =
          old_abs < limits.pre_max ? old_abs : limits.pre_max;
      const std::int32_t old_contribution =
          (old_syntax < kMaxBaseBrRange ? old_syntax : kMaxBaseBrRange) & 1;
      const double step = job.quant_step;
      const double old_recon = static_cast<double>(old_level) * step;
      const double old_err = original[c] - old_recon;
      const double old_dist = old_err * old_err;
      for (std::int32_t d = 0; d < 2; ++d) {
        const std::int32_t delta = d == 0 ? 1 : -1;
        std::int32_t next = 0;
        if (!tcq_make_delta_level(old_level, delta, &next)) continue;
        const std::int32_t next_abs = tcq_abs(next);
        const std::int32_t next_syntax =
            next_abs < limits.pre_max ? next_abs : limits.pre_max;
        const std::int32_t next_contribution =
            (next_syntax < kMaxBaseBrRange ? next_syntax : kMaxBaseBrRange) & 1;
        const bool flips = next_contribution != old_contribution;
        const bool drops_hiding = next == 0 && num_nz == kPhThresh;
        if (!flips && !drops_hiding) continue;
        if (exhaustive) {
          tcq_parity_consider(&search, c, next, delta);
          continue;
        }
        const double new_recon = static_cast<double>(next) * step;
        const double new_err = original[c] - new_recon;
        TcQParityCandidate candidate;
        candidate.local_delta = new_err * new_err - old_dist;
        candidate.scan_index = c;
        candidate.level = next;
        candidate.delta = delta;
        if (keep_limit == 0) continue;
        // Stable insertion into the best `keep_limit` by
        // (local_delta, scan_index desc); equal keys keep generation order.
        std::int32_t position = kept;
        for (std::int32_t i = 0; i < kept; ++i) {
          if (tcq_parity_candidate_less(candidate, top[i])) {
            position = i;
            break;
          }
        }
        if (position >= keep_limit) continue;
        const std::int32_t last =
            kept < keep_limit ? kept : keep_limit - 1;
        for (std::int32_t i = last; i > position; --i) {
          top[i] = top[i - 1];
        }
        top[position] = candidate;
        if (kept < keep_limit) ++kept;
      }
    }
    if (!exhaustive) {
      for (std::int32_t i = 0; i < kept; ++i) {
        tcq_parity_consider(&search, top[i].scan_index, top[i].level,
                            top[i].delta);
      }
    }
  }

  if (!search.have_best) return -1;
  if (search.best_index >= 0) {
    levels_out[search.best_index] = search.best_level;
  }
  batch.eob[job_index] = search.best_eob;
  batch.hidden[job_index] = search.best_hidden;
  batch.hiding_turned_off[job_index] =
      (hidable && search.best_hidden == 0) ? 1 : 0;
  batch.adjusted_scan_index[job_index] = search.best_index;
  batch.adjusted_delta[job_index] = search.best_delta;
  batch.distortion[job_index] = search.best_distortion;
  batch.status[job_index] = 0;
  return 0;
}

}  // namespace av2enc::cuda::tcq::detail
