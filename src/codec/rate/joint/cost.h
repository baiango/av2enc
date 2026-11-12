// Per-node cost arithmetic and the CPU reference cost trace.
//
// Everything in the unified loop is expressed with one breakdown so an
// experiment can attribute a decision to distortion, mode syntax, coefficient
// rate, transform-type syntax or partition syntax:
//
//     distortion_sse    sum of squared error over the coded planes
//     mode_bits         luma mode + angle, chroma mode + CfL syntax
//     coefficient_bits  transform-stream rate (txb_skip + EOB + coefficients)
//     tx_type_bits      §5.20.8.2 transform-type symbol (luma only)
//     signal_bits       partition do_split / rect_type / square-split syntax
//
// Rate is always in bits.  A caller holding bytes must multiply by 8.
//
// The trace is a fixed-capacity, caller-owned recorder.  It never allocates
// and never changes a decision; it exists for experiment reports and for the
// GPU planner to compare per-node costs without recomputing them.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "codec/partition/tree/partition.h"

namespace av2enc::still::rdo_loop {

struct CostTerms {
  double distortion_sse = 0.0;
  double mode_bits = 0.0;
  double coefficient_bits = 0.0;
  double tx_type_bits = 0.0;
  double signal_bits = 0.0;
};

double total_bits(const CostTerms& terms) noexcept;
bool cost_terms_valid(const CostTerms& terms) noexcept;

struct NodeCost {
  CostTerms terms;
  double lambda = 0.0;
  bool rate_aware = true;
  double cost = 0.0;
  int leaves = 0;
};

bool node_cost_valid(const NodeCost& cost) noexcept;

// Builds the Lagrangian cost from the breakdown.  Invalid input returns a cost
// with +inf cost so a deterministic selector rejects it instead of propagating
// NaN.
NodeCost make_node_cost(const CostTerms& terms, double lambda,
                        bool rate_aware, int leaves = 0) noexcept;

// Sums child/leaf terms into an aggregate.  `target` must already carry the
// intended lambda and rate_aware flag; the cost is recomputed after each add.
// Invalid terms make the aggregate infinite.
void accumulate_node_cost(NodeCost& target, const NodeCost& addend) noexcept;

// Adds signal bits (partition syntax) to an existing node cost.  Forced
// partitions pass 0.
void add_signal_bits(NodeCost& target, double bits) noexcept;

// Deterministic total order for candidates at one node:
//   1. lower cost
//   2. lower distortion SSE
//   3. lower total rate
//   4. fewer leaves (a coarser tree on an exact tie)
//   5. lower partition ordinal
//   6. lower insertion index
// Exact comparisons only: no epsilon, no tolerance.
bool node_cost_better(const NodeCost& a, partition::PartitionType part_a,
                      int index_a, const NodeCost& b,
                      partition::PartitionType part_b, int index_b) noexcept;

// One node's decision, including the winner's full breakdown and every
// candidate's aggregate.  The fixed arrays keep the hot loop allocation-free.
inline constexpr int kMaxNodeCandidates = 4;

struct NodeCandidate {
  partition::PartitionType partition = partition::PartitionType::kNone;
  NodeCost cost;
  CostTerms terms[kMaxNodeCandidates];  // per-child breakdown, unused entries 0
  int child_count = 0;
  bool valid = false;
};

struct NodeDecision {
  bool found = false;
  partition::PartitionType partition = partition::PartitionType::kNone;
  NodeCost cost;
  int candidate_index = 0;
  int candidate_count = 0;
  int rejected_count = 0;
  NodeCandidate candidates[kMaxNodeCandidates];
};

// Per-node trace entry.  This is deliberately a plain POD so a GPU planner can
// write it directly.
struct NodeTraceEntry {
  std::int32_t mi_row = 0;
  std::int32_t mi_col = 0;
  std::uint16_t size = 0;       // partition::BlockSize
  std::int8_t depth = 0;
  std::uint8_t partition = 0;   // partition::PartitionType
  std::uint8_t forced = 0;
  std::uint8_t rate_aware = 1;
  std::uint8_t child_count = 0;
  double lambda = 0.0;
  CostTerms terms;
  double cost = 0.0;
  std::int32_t leaves = 0;
};

inline constexpr std::size_t kTraceCapacity = std::size_t{1} << 15;

class CostTrace {
 public:
  void clear() noexcept {
    size_ = 0;
    dropped_ = 0;
    enabled_ = true;
  }
  void set_enabled(bool enabled) noexcept { enabled_ = enabled; }
  bool enabled() const noexcept { return enabled_; }

  bool record(const NodeTraceEntry& entry) noexcept {
    if (!enabled_) {
      return false;
    }
    if (size_ >= kTraceCapacity) {
      ++dropped_;
      return false;
    }
    entries_[size_++] = entry;
    return true;
  }

  std::size_t size() const noexcept { return size_; }
  std::size_t dropped() const noexcept { return dropped_; }
  const NodeTraceEntry& at(std::size_t index) const noexcept {
    return entries_[index];
  }
  std::span<const NodeTraceEntry> entries() const noexcept {
    return std::span<const NodeTraceEntry>(entries_, size_);
  }

 private:
  NodeTraceEntry entries_[kTraceCapacity]{};
  std::size_t size_ = 0;
  std::size_t dropped_ = 0;
  bool enabled_ = true;
};

}  // namespace av2enc::still::rdo_loop
