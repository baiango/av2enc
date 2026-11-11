// GPU-ready candidate batch.
//
// A batch is one decision group: all luma modes for a block, all chroma modes,
// all TX types for a transform, or all partition shapes for a node.  Every
// field is a plain value; there are no pointers, spans, callbacks or hidden
// allocations.  A CUDA kernel can fill `records[i].distortion_sse` and the rate
// components in any order, then run the same deterministic reduction this CPU
// reference uses.  That is the contract the GPU port has to match bit-for-bit
// (build both with -ffp-contract=off).
//
// The reduction is stable and total:
//   1. lower cost = distortion + lambda * total_bits
//   2. lower distortion
//   3. lower total bits
//   4. lower caller index (AV2/search order)
//   5. lower array position
//
// Invalid records (non-finite/negative terms, bad lambda, duplicate index) are
// rejected and counted; they never influence the winner.

#pragma once

#include "cost.h"

#include <cstdint>
#include <string>
#include <type_traits>

namespace av2enc::m1::rdo_loop {

enum class CandidateDomain : std::uint8_t {
  kLumaMode = 0,
  kChromaMode = 1,
  kTxType = 2,
  kPartition = 3,
};

inline constexpr int kMaxBatchCandidates = 96;

struct CandidateRecord {
  // Group id (block/node) and the caller's candidate index.  Both are
  // diagnostic: one batch is one group, so `index` is the tie-break key.
  std::uint32_t group = 0;
  std::uint16_t index = 0;
  std::uint8_t domain = 0;     // CandidateDomain
  std::uint8_t partition = 0;  // partition::PartitionType for kPartition
  std::uint8_t plane = 0;      // 0 luma, 1 U, 2 V
  std::uint8_t tx_size = 0;    // AV2 TX_SIZE index
  std::int16_t mode = 0;       // AV2 mode (YMode or UVMode)
  std::int8_t angle_delta = 0;
  std::uint8_t is_cfl = 0;
  std::uint8_t flags = 0;      // caller-defined (e.g. pruned/forced)

  // Measurements written by the encoder (or a GPU kernel).
  double distortion_sse = 0.0;
  double mode_bits = 0.0;
  double coefficient_bits = 0.0;
  double tx_type_bits = 0.0;
  double signal_bits = 0.0;

  // Filled by finalize_batch().
  double cost = 0.0;
  std::uint8_t valid = 1;
};

struct CandidateBatch {
  std::uint32_t batch_id = 0;
  std::int32_t domain = 0;  // CandidateDomain
  std::int32_t count = 0;
  double lambda = 0.0;
  std::uint8_t rate_aware = 1;
  std::uint8_t reserved[3] = {0, 0, 0};
  CandidateRecord records[kMaxBatchCandidates]{};

  // Results.
  std::int32_t best_index = -1;  // index into records[]
  double best_cost = 0.0;
  std::int32_t evaluated = 0;
  std::int32_t rejected = 0;
};

static_assert(std::is_trivially_copyable_v<CandidateRecord>);
static_assert(std::is_trivially_copyable_v<CandidateBatch>);

// Comparator usable from host or device code.  `position` is the array slot.
inline bool batch_record_better(const CandidateRecord& a, int position_a,
                                const CandidateRecord& b, int position_b,
                                bool rate_aware) noexcept {
  (void)rate_aware;  // cost already encodes rate awareness
  if (a.cost != b.cost) {
    return a.cost < b.cost;
  }
  if (a.distortion_sse != b.distortion_sse) {
    return a.distortion_sse < b.distortion_sse;
  }
  const double rate_a = a.mode_bits + a.coefficient_bits + a.tx_type_bits +
                        a.signal_bits;
  const double rate_b = b.mode_bits + b.coefficient_bits + b.tx_type_bits +
                        b.signal_bits;
  if (rate_a != rate_b) {
    return rate_a < rate_b;
  }
  if (a.index != b.index) {
    return a.index < b.index;
  }
  return position_a < position_b;
}

// Adds one record.  Returns false and increments rejected on overflow/invalid
// identity fields.  Measurement validity is checked by finalize_batch().
bool append_candidate(CandidateBatch& batch, const CandidateRecord& record,
                      std::string* error = nullptr);

// Computes every cost and the winner.  Allocation-free and deterministic.
// Returns false on an invalid batch (no candidates, invalid lambda).
bool finalize_batch(CandidateBatch& batch, std::string* error = nullptr);

// Convenience for a caller that only has aggregate terms.
CandidateRecord make_candidate_record(std::uint32_t group, std::uint16_t index,
                                      CandidateDomain domain,
                                      const CostTerms& terms) noexcept;

}  // namespace av2enc::m1::rdo_loop
