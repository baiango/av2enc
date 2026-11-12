// Host-side helpers for the CUDA TCQ component.
//
// These functions run on the host, need no CUDA runtime, and are shared by the
// emulation test and the real GPU test:
//
//   * tcq_scan_and_pre_max(): §5.20.7.30 scan + §5.20.7.27 pre_max for one TX
//     size / scan class / plane, derived from the parity-hiding reference
//     geometry so the device sees exactly `parity::level_limits()`.
//   * tcq_fill_rate_tables(): invokes the caller's tcq::RateModel callbacks for
//     every (coefficient, state, level) and both is_eob values, producing the
//     tables the device DP consumes.  This is what makes an arbitrary CPU rate
//     model bit-exact on the GPU instead of a device log2 approximation.
//
// Both write into caller-owned host staging; tcq.h exposes the matching device
// workspace slots.
#pragma once

#include <cstdint>
#include <string>

#include "cuda/tcq/tcq_types.h"
#include "encoder/tcq/tcq.h"

namespace av2enc::cuda::tcq {

// Fills `scan` and `pre_max` for one block.  `max_coefficients` is the caller's
// buffer capacity; the adjusted block area must fit.  `tx_class` is
// 0 = TX_CLASS_2D, 1 = TX_CLASS_HORIZ, 2 = TX_CLASS_VERT; `plane` is 0..2.
// `count`, `w` and `h` receive the adjusted area and dimensions.
bool tcq_scan_and_pre_max(int tx_size, int tx_class, int plane,
                          int max_coefficients, int* count, int* w, int* h,
                          std::int32_t* scan, std::int32_t* pre_max,
                          std::string* error = nullptr);

// Fills `normal_rate`, `eob_coeff_rate` (both table_levels * 8 * count doubles)
// and `block_eob_rate` (count + 1 doubles) for `job`.
//
// The (coefficient, state, level) decomposition is the same one the CPU DP
// evaluates, so the stored doubles are the exact callback results.  Returns
// false (with a diagnostic) if a callback returns a non-finite or negative
// rate, or if a derived level bound does not fit `job.table_levels`.
bool tcq_fill_rate_tables(const TcqJob& job, const double* original,
                          const std::int32_t* pre_max,
                          const std::int32_t* level_context,
                          const av2enc::tcq::RateModel& model,
                          double* normal_rate, double* eob_coeff_rate,
                          double* block_eob_rate,
                          std::string* error = nullptr);

}  // namespace av2enc::cuda::tcq
