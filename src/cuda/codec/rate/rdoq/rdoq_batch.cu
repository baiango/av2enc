// CUDA full-trellis RDOQ batch kernel.
//
// Mirrors `av2enc::rdoq::rdoq_quantize` (src/encoder/rdoq/rdoq.cpp) exactly:
// same DP recurrence, same derived level range supplied by the host, same
// tie-break order, same final cost recomputation.  Build with -fmad=false and
// the CPU reference with -ffp-contract=off for bit-exact decisions.

#include "rdoq_batch.h"

#include <cmath>
#include <cstdint>
#include <math_constants.h>

namespace av2enc::cuda::rdoq {
namespace {

struct Choice {
  double cost;
  double distortion;
  double rate_bits;
  std::int32_t signed_level;
  std::int32_t abs_level;
  std::int32_t mode;  // 0 terminate-all-zero, 1 continue, 2 EOB here
  int terminal;
};

__device__ __forceinline__ bool choice_better(const Choice& a,
                                              const Choice& b) {
  if (a.cost != b.cost) return a.cost < b.cost;
  if (a.distortion != b.distortion) return a.distortion < b.distortion;
  if (a.rate_bits != b.rate_bits) return a.rate_bits < b.rate_bits;
  if (a.abs_level != b.abs_level) return a.abs_level < b.abs_level;
  if (a.terminal != b.terminal) return a.terminal != 0;
  return a.signed_level < b.signed_level;
}

__device__ __forceinline__ std::int32_t abs_i32(std::int32_t value) {
  return value < 0 ? -value : value;
}

__device__ __forceinline__ bool finite_d(double value) {
  return isfinite(value) != 0;
}

__global__ void rdoq_batch_kernel(Batch batch) {
  const std::int32_t job_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (job_index >= batch.num_jobs) {
    return;
  }
  const BlockJob job = batch.jobs[job_index];
  const std::int32_t count = job.num_coefficients;
  const std::int32_t cap = job.prev_level_cap;
  const std::int32_t states = cap + 1;
  const std::int32_t levels_per_coefficient = job.table_levels;
  std::int32_t* level_out = batch.levels + job.level_offset;
  for (std::int32_t i = 0; i < count; ++i) {
    level_out[i] = 0;
  }
  batch.eob[job_index] = -1;
  batch.status[job_index] = -1;
  if (count < 0 || cap < 0 || cap >= kMaxStates ||
      levels_per_coefficient <= 0) {
    return;
  }
  if (count == 0) {
    const double bits = batch.block_eob_rate[job.eob_table_offset];
    batch.eob[job_index] = 0;
    batch.distortion[job_index] = 0.0;
    batch.rate_bits[job_index] = bits;
    batch.cost[job_index] = job.lambda * bits;
    batch.status[job_index] = 0;
    return;
  }

  const double* original = batch.original + job.coefficient_offset;
  const std::int32_t* sign = batch.sign + job.coefficient_offset;
  const std::int32_t* bound = batch.bound + job.coefficient_offset;
  const double* normal_rate = batch.normal_rate + job.table_offset;
  const double* eob_coeff_rate = batch.eob_rate + job.table_offset;
  const double* block_eob_rate = batch.block_eob_rate + job.eob_table_offset;
  std::int32_t* path_levels = batch.path_levels + job.path_offset;
  std::uint8_t* path_modes = batch.path_modes + job.path_offset;
  const double lambda = job.lambda;
  const double quant_step = job.quant_step;

  double total_zero_distortion = 0.0;
  for (std::int32_t i = 0; i < count; ++i) {
    total_zero_distortion += original[i] * original[i];
  }

  double dp_next[kMaxStates];
  double dd_next[kMaxStates];
  double dr_next[kMaxStates];
  double dp_cur[kMaxStates];
  double dd_cur[kMaxStates];
  double dr_cur[kMaxStates];
  for (std::int32_t s = 0; s < kMaxStates; ++s) {
    dp_next[s] = CUDART_INF;
    dd_next[s] = 0.0;
    dr_next[s] = 0.0;
    dp_cur[s] = CUDART_INF;
    dd_cur[s] = 0.0;
    dr_cur[s] = 0.0;
  }

  double suffix_after = 0.0;  // sum of original[j]^2 for j > i
  for (std::int32_t i = count; i-- > 0;) {
    const double orig = original[i];
    const std::int32_t level_bound = bound[i];
    const std::int32_t coefficient_sign = sign[i];
    std::int32_t closest = 0;
    const double ratio = fabs(orig) / quant_step;
    if (isfinite(ratio) != 0) {
      const double rounded = round(ratio);
      if (rounded <= 0.0) {
        closest = 0;
      } else if (rounded >= static_cast<double>(level_bound)) {
        closest = level_bound;
      } else {
        closest = static_cast<std::int32_t>(rounded);
      }
    }
    for (std::int32_t prev = 0; prev < states; ++prev) {
      Choice best;
      best.cost = CUDART_INF;
      best.distortion = 0.0;
      best.rate_bits = 0.0;
      best.signed_level = 0;
      best.abs_level = 0;
      best.mode = 0;
      best.terminal = 0;

      if (i == 0) {
        best.distortion = total_zero_distortion;
        best.rate_bits = block_eob_rate[0];
        best.cost = best.distortion + lambda * best.rate_bits;
        best.signed_level = 0;
        best.abs_level = 0;
        best.mode = 0;
        best.terminal = 1;
      }

      for (std::int32_t magnitude = 0; magnitude <= level_bound; ++magnitude) {
        const std::int32_t signed_level =
            magnitude == 0 ? 0 : coefficient_sign * magnitude;
        const double reconstructed =
            static_cast<double>(signed_level) * quant_step;
        const double difference = orig - reconstructed;
        const double distortion = difference * difference;
        const std::int32_t next_state =
            magnitude < cap ? magnitude : cap;
        const std::int32_t rate_offset =
            (i * states + prev) * levels_per_coefficient + magnitude;

        const double normal_bits = normal_rate[rate_offset];
        if (finite_d(dp_next[next_state])) {
          Choice candidate;
          candidate.cost =
              distortion + lambda * normal_bits + dp_next[next_state];
          candidate.distortion = distortion + dd_next[next_state];
          candidate.rate_bits = normal_bits + dr_next[next_state];
          candidate.signed_level = signed_level;
          candidate.abs_level = magnitude;
          candidate.mode = 1;
          candidate.terminal = 0;
          if (choice_better(candidate, best)) best = candidate;
        }

        if (magnitude != 0) {
          const double eob_bits = eob_coeff_rate[rate_offset];
          Choice candidate;
          candidate.distortion = distortion + suffix_after;
          candidate.rate_bits = eob_bits + block_eob_rate[i + 1];
          candidate.cost =
              candidate.distortion + lambda * candidate.rate_bits;
          candidate.signed_level = signed_level;
          candidate.abs_level = magnitude;
          candidate.mode = 2;
          candidate.terminal = 1;
          if (choice_better(candidate, best)) best = candidate;
        }
        // Same safe stop rule as the CPU: above the distortion minimum, once
        // the local distortion alone exceeds the best total cost, no higher
        // magnitude can win.
        if (magnitude >= closest && finite_d(best.cost) &&
            distortion >= best.cost) {
          break;
        }
      }

      dp_cur[prev] = best.cost;
      dd_cur[prev] = best.distortion;
      dr_cur[prev] = best.rate_bits;
      const std::int32_t slot = i * states + prev;
      path_levels[slot] = best.signed_level;
      path_modes[slot] = static_cast<std::uint8_t>(best.mode);
    }
    for (std::int32_t s = 0; s < states; ++s) {
      const double next_cost = dp_next[s];
      dp_next[s] = dp_cur[s];
      dp_cur[s] = next_cost;
      const double next_distortion = dd_next[s];
      dd_next[s] = dd_cur[s];
      dd_cur[s] = next_distortion;
      const double next_rate = dr_next[s];
      dr_next[s] = dr_cur[s];
      dr_cur[s] = next_rate;
    }
    suffix_after += orig * orig;
  }

  if (!finite_d(dp_next[0])) {
    return;
  }

  std::int32_t state = 0;
  std::int32_t eob_value = count;
  for (std::int32_t i = 0; i < count; ++i) {
    const std::int32_t slot = i * states + state;
    const std::int32_t level = path_levels[slot];
    const std::int32_t mode = path_modes[slot];
    if (mode == 0) {
      eob_value = i;
      break;
    }
    level_out[i] = level;
    state = abs_i32(level) < cap ? abs_i32(level) : cap;
    if (mode == 2) {
      eob_value = i + 1;
      break;
    }
  }

  double distortion = 0.0;
  for (std::int32_t i = 0; i < count; ++i) {
    const double reconstructed =
        static_cast<double>(level_out[i]) * quant_step;
    const double difference = original[i] - reconstructed;
    distortion += difference * difference;
  }

  double rate_bits = block_eob_rate[eob_value];
  std::int32_t previous = 0;
  for (std::int32_t i = 0; i < eob_value; ++i) {
    const std::int32_t level = level_out[i];
    const std::int32_t is_eob = (i == eob_value - 1 && level != 0) ? 1 : 0;
    const std::int32_t rate_offset =
        (i * states + previous) * levels_per_coefficient + abs_i32(level);
    rate_bits += is_eob != 0 ? eob_coeff_rate[rate_offset]
                             : normal_rate[rate_offset];
    const std::int32_t magnitude = abs_i32(level);
    previous = magnitude < cap ? magnitude : cap;
  }

  batch.eob[job_index] = eob_value;
  batch.distortion[job_index] = distortion;
  batch.rate_bits[job_index] = rate_bits;
  batch.cost[job_index] = distortion + lambda * rate_bits;
  batch.status[job_index] = 0;
}

}  // namespace

cudaError_t quantize_batch(const Batch& batch, cudaStream_t stream) {
  if (batch.num_jobs <= 0) {
    return cudaSuccess;
  }
  if (batch.original == nullptr || batch.sign == nullptr ||
      batch.bound == nullptr || batch.normal_rate == nullptr ||
      batch.eob_rate == nullptr || batch.block_eob_rate == nullptr ||
      batch.jobs == nullptr || batch.levels == nullptr || batch.eob == nullptr ||
      batch.distortion == nullptr || batch.rate_bits == nullptr ||
      batch.cost == nullptr || batch.status == nullptr ||
      batch.path_levels == nullptr || batch.path_modes == nullptr) {
    return cudaErrorInvalidValue;
  }
  constexpr int kThreads = 128;
  const int blocks = (batch.num_jobs + kThreads - 1) / kThreads;
  rdoq_batch_kernel<<<blocks, kThreads, 0, stream>>>(batch);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::rdoq
