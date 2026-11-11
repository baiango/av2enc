// GPU batch counterpart of `transform_stream.h`.
//
// The heavy arithmetic is borrowed from the two verified CUDA components:
//
//   * `src/cuda/tx_type/`  - candidate forward transforms + scalar levels
//   * `src/cuda/rdoq/`     - full coefficient-level trellis, one thread/block
//
// The adapter owns the glue: candidate enumeration, per-candidate rate-table
// materialisation, RDOQ, inverse transform, coded-domain SSE and the final
// deterministic TX-type decision.
//
// Current contract (correctness first)
// ------------------------------------
// `src/cuda/rdoq/` consumes a caller-built per-coefficient rate table, so the
// adapter builds those tables on the host from the same `RateModelHooks` /
// AV2 §9.3 model the CPU path uses, then uploads them.  The device still runs
// every transform, every DP and every inverse transform; the host only:
//
//   * enumerates candidates and packs descriptors,
//   * materialises the rate table (no device callback exists yet),
//   * compares the per-candidate costs with the exact CPU tie-break.
//
// Decisions are therefore bit-exact with `ts::quantize_block()`: the DP is the
// verified CUDA kernel, the transform is the verified CUDA kernel, and the
// cost/tie-break code is shared.  A future device-side rate model can replace
// the table build without changing this API.
//
// No hidden allocations in the steady state: pass a `DeviceAllocator` backed by
// `src/cuda/arena.*`.  The default `cudaMalloc`/`cudaFree` pair exists for
// tests and bring-up only.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "encoder/m1/transform_stream/transform_stream.h"

namespace av2enc::m1::ts::gpu {

struct DeviceAllocator {
  void* (*allocate)(void* user, std::size_t bytes) = nullptr;
  void (*deallocate)(void* user, void* pointer) = nullptr;
  void* user = nullptr;
};

struct BatchOptions {
  // Null -> cudaMalloc/cudaFree (tests/bring-up only).
  DeviceAllocator allocator{};
  // Hard cap on the per-coefficient rate tables.  Exceeding it returns an
  // error before allocating; split the block batch and retry.  The table size
  // is dominated by the largest coefficient bound in the batch.
  std::size_t table_budget_bytes = 512ull << 20;
  // Shared by every request; individual requests may override with their own
  // non-null hooks.  Null -> AV2 §9.3 CDF model.
  const RateModelHooks* hooks = nullptr;
  cudaStream_t stream = nullptr;
};

// One evaluated candidate.  Filled for every candidate, including ones that
// are not codable (their `codable` flag is false).  This is the parity surface:
// compare it against a per-candidate CPU evaluation.
struct CandidateOutcome {
  int block = -1;
  int candidate_index = -1;
  int type = 0;  // TX_TYPE numeric value
  TxTypeSymbol symbol;
  bool codable = false;
  std::int32_t eob = 0;
  std::int32_t nonzero = 0;
  double sse = 0.0;
  double coeff_rate_bits = 0.0;
  double type_bits = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  double rdoq_distortion = 0.0;
  double rdoq_rate_bits = 0.0;
  double rdoq_cost = 0.0;
  // 0 = accepted; -1 = the CUDA RDOQ kernel rejected the job.
  std::int32_t status = 0;
  std::vector<std::int32_t> levels_scan;
  std::vector<std::int16_t> recon_coded;
};

// Evaluates every candidate of every request.  `requests[i].residual_coded`
// points at host memory.  Outcomes are grouped by block in candidate order.
bool evaluate_batch(const Request* requests, int count,
                    const BatchOptions& options,
                    std::vector<CandidateOutcome>* outcomes,
                    std::string* error = nullptr);

// Full decision batch.  Fills `decisions[0..count)`; on failure returns false
// and leaves decisions reset (`valid == false`).
bool quantize_batch(const Request* requests, int count,
                    const BatchOptions& options, Decision* decisions,
                    std::string* error = nullptr);

}  // namespace av2enc::m1::ts::gpu
