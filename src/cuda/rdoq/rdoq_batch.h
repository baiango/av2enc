// CUDA batch entry point for the full coefficient-level RDOQ trellis.
//
// The CPU reference is `src/encoder/rdoq/`.  This layer does not invent a
// device rate model: the caller supplies a precomputed per-coefficient rate
// table.  That keeps the device DP free of transcendental functions and lets
// the entropy owner generate costs however it likes (CDF model, measured
// table, exact coder probe) while CPU and GPU run the *same* DP.
//
// Layout
// ------
// For one job:
//   S      = prev_level_cap + 1
//   L      = table_levels (uniform level count; bound[i] <= L - 1)
//   normal rate index(i, s, m) = table_offset + (i * S + s) * L + m
//   eob    rate index(i, s, m) = same layout in `eob_rate`
//   block EOB rate index(e)    = eob_table_offset + e, e = 0..N
//   path storage               = path_offset + i * S + s
//
// All arrays are caller-owned.  The launcher allocates nothing, frees nothing
// and does not synchronize; use the arena for long-lived buffers.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace av2enc::cuda::rdoq {

// Matches av2enc::rdoq::Config::prev_level_cap <= 63.
inline constexpr int kMaxStates = 64;

struct BlockJob {
  std::int32_t coefficient_offset = 0;  // into original/sign/bound
  std::int32_t num_coefficients = 0;
  std::int32_t level_offset = 0;      // into levels
  std::int32_t path_offset = 0;       // into path_levels/path_modes
  std::int32_t table_offset = 0;      // into normal_rate/eob_rate
  std::int32_t eob_table_offset = 0;  // into block_eob_rate
  std::int32_t prev_level_cap = 2;
  std::int32_t table_levels = 1;  // L
  double quant_step = 1.0;
  double lambda = 0.0;
};

struct Batch {
  const double* original = nullptr;
  const std::int32_t* sign = nullptr;   // +1 / -1 per coefficient
  const std::int32_t* bound = nullptr;  // derived bound per coefficient
  const double* normal_rate = nullptr;
  const double* eob_rate = nullptr;
  const double* block_eob_rate = nullptr;
  const BlockJob* jobs = nullptr;
  std::int32_t num_jobs = 0;

  std::int32_t* levels = nullptr;
  std::int32_t* eob = nullptr;
  double* distortion = nullptr;
  double* rate_bits = nullptr;
  double* cost = nullptr;
  std::int32_t* status = nullptr;  // 0 = ok, -1 = rejected job

  // Per-job scratch, num_coefficients * S entries.  The caller sizes it.
  std::int32_t* path_levels = nullptr;
  std::uint8_t* path_modes = nullptr;
};

// Launches one thread per job.  Returns the launch error; per-job validation
// failures are reported in `status` (a rejected job writes eob = -1).
cudaError_t quantize_batch(const Batch& batch, cudaStream_t stream);

}  // namespace av2enc::cuda::rdoq
