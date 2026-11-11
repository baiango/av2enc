// Unified partition / mode / transform decision loop (CPU reference).
//
// The loop replaces the M1 encoder's variance-only partition search with one
// cost comparison over every legal decision that changes the coded block:
//
//   * partition NONE / HORZ / VERT / SPLIT (SPLIT only where §9.2 allows it),
//   * luma mode and angle (candidate pruning before the rate probe),
//   * chroma DC / CfL / full UV,
//   * TX type and RDOQ (the caller supplies the measured terms),
//   * a per-CTU lambda and a fixed candidate order.
//
// This header owns the *policy* half of the loop:
//   * the exact spec/legality query for one node (delegated to partition/ and
//     partition_ext/),
//   * the deterministic candidate order and tie-break,
//   * exact branch-and-bound on partially evaluated partitions,
//   * the per-node cost breakdown.
//
// The encoder adapter owns the *measurement* half: it trial-encodes a leaf or a
// partition shape with the real prediction/transform/RDOQ path and hands the
// breakdown back.  Keeping the two halves separate is what makes the loop
// GPU-ready: a CUDA kernel can fill the same CostTerms arrays, and the
// selector below is a pointer-free reduction.
//
// No function here allocates.  NodeDecision holds a fixed four-candidate
// array; pruning scratch is caller-owned; CostTrace is caller-owned.

#pragma once

#include "cost.h"
#include "encoder/partition/partition.h"
#include "encoder/partition_ext/partition_ext.h"

#include <cstdint>
#include <span>
#include <string>

namespace av2enc::m1::rdo_loop {

// Search order is the task's order, not the numeric PartitionType order:
// NONE, HORZ, VERT, SPLIT.  Ties therefore prefer NONE, then horizontal
// splits, then vertical splits, then the square split.
inline constexpr int kSearchOrderCount = 4;
inline constexpr partition::PartitionType
    kSearchOrder[kSearchOrderCount] = {
        partition::PartitionType::kNone,
        partition::PartitionType::kHorz,
        partition::PartitionType::kVert,
        partition::PartitionType::kSplit,
};

struct SearchOptions {
  int min_block_size_px = 8;
  int max_depth = 4;
  bool allow_rect = true;
  bool allow_split = false;
  bool rate_aware = true;
  // Candidate pruning.  <= 0 means "probe every candidate".  Pruning happens
  // after SSE prediction and before transform/RDOQ, so it removes work, not
  // accuracy of the cost model.  The SSE margin is relative to the best SSE.
  int max_luma_candidates = 6;
  int max_chroma_candidates = 4;
  double candidate_sse_margin = 1.35;
  // Exact branch-and-bound: a partition candidate is abandoned when the
  // accumulated child cost already exceeds the best complete candidate and the
  // remaining children cannot have negative cost.  This never prunes the true
  // winner under non-negative distortion and rate.
  bool exact_branch_and_bound = true;
  // Optional hard cap on leaf probes per superblock.  Probe order is fixed, so
  // a capped run is still deterministic.  0 = unlimited.
  int max_leaf_probes_per_superblock = 0;
};

bool search_options_valid(const SearchOptions& options) noexcept;
bool validate_search_options(const SearchOptions& options,
                             std::string* error = nullptr);

// One node's spec-derived state.  `allowed` is the four-value M1 subset; the
// extended partition family is intentionally not probed by this loop.
struct NodeInfo {
  int mi_row = 0;
  int mi_col = 0;
  partition::BlockSize size = partition::BlockSize::kInvalid;
  int depth = 0;
  bool has_chroma = false;
  bool chroma_offset = false;
  partition::PartitionBoundary boundary;
  partition::PartitionSet allowed;
  bool forced = false;
  partition::PartitionType forced_partition =
      partition::PartitionType::kNone;
};

// Bits charged for the partition symbol when the caller's measurement does not
// include it.  Forced partitions cost 0 (no syntax is coded).  The values are
// deterministic estimates; the caller can override them in CandidateCost.
double default_partition_signal_bits(
    const NodeInfo& info, partition::PartitionType partition) noexcept;

// Fills NodeInfo from partition::* and applies the search policy.  A forced or
// implied partition is reported in `forced`; callers must not probe anything
// else in that case.
bool describe_node(const partition::FrameGeometry& geometry,
                   int mi_row, int mi_col, partition::BlockSize size,
                   int depth, bool has_chroma, bool chroma_offset,
                   NodeInfo* out, std::string* error = nullptr);

// True when the encoder policy allows `partition` at this node (depth and
// minimum block size).  Spec legality is separate: check NodeInfo::allowed.
bool policy_allows(const SearchOptions& options, int depth,
                   partition::BlockSize size,
                   partition::PartitionType partition) noexcept;

// One child produced by decode_partition() in decode order.  Delegates to
// partition_ext::enumerate_children(), so the loop and the entropy writer share
// one authority for child geometry and chroma bookkeeping.
struct ChildRef {
  int mi_row = 0;
  int mi_col = 0;
  partition::BlockSize size = partition::BlockSize::kInvalid;
  bool has_chroma = false;
  bool chroma_offset = false;
};

// Returns the number of in-frame children (0 for NONE or an invalid subsize).
// `slots` receives up to four entries in decode order.
int enumerate_node_children(const partition::FrameGeometry& geometry,
                            const NodeInfo& info,
                            partition::PartitionType partition,
                            ChildRef slots[4], std::string* error = nullptr);

struct CandidateCost {
  partition::PartitionType partition = partition::PartitionType::kNone;
  NodeCost cost;
  // Per-child breakdown in decode order.  `child_count` entries are used.
  CostTerms child_terms[kMaxNodeCandidates]{};
  int child_count = 0;
  bool valid = false;
};

// Incremental node selector.  The adapter calls reset(), then add() once per
// fully measured candidate.  add() is O(1) and allocation-free.
class NodeDecisionLoop {
 public:
  void reset(const NodeInfo& info, const SearchOptions& options) noexcept;
  bool add(const CandidateCost& candidate, std::string* error = nullptr);
  bool add_terms(partition::PartitionType partition, const CostTerms& terms,
                 int leaves, std::string* error = nullptr);
  bool add_partition(partition::PartitionType partition,
                     std::span<const NodeCost> children,
                     std::string* error = nullptr);

  // Exact branch-and-bound check.  Returns true when a partially evaluated
  // partition candidate can no longer beat the current best, whatever the
  // remaining children cost.  `remaining_children` is the number not measured
  // yet; their distortion and rate are >= 0 for a valid encoder.
  bool cannot_beat_best(const NodeCost& accumulated) const noexcept;

  const NodeDecision& decision() const noexcept { return decision_; }
  const NodeInfo& info() const noexcept { return info_; }
  void set_lambda(double lambda) noexcept { lambda_ = lambda; }

 private:
  NodeInfo info_{};
  SearchOptions options_{};
  double lambda_ = 0.0;
  NodeDecision decision_{};
};

// Fills the trace entry for one completed node.  Returns false (and leaves the
// trace untouched) if the decision has no valid winner.
bool record_node_trace(const NodeInfo& info, const NodeDecision& decision,
                       CostTrace& trace) noexcept;

// ---------------------------------------------------------------------------
// Candidate pruning
// ---------------------------------------------------------------------------

struct CandidateScore {
  int index = 0;   // caller's candidate index (stable, deterministic)
  double sse = 0.0;
};

struct PruneConfig {
  // Keep at most this many candidates.  <= 0 keeps all.
  int max_keep = 0;
  // Keep candidates whose SSE is <= best_sse * margin.  <= 0 disables.
  double sse_margin = 0.0;
  // Always keep this caller index when >= 0 (e.g. DC for luma, DC for chroma,
  // DCT_DCT for TX type), even when it is outside the margin/top-K.
  int keep_index = -1;
};

// Writes the kept caller indices in their original order into `keep` and
// returns the count.  Deterministic: selection is by (sse, index), output is
// original order.  `scratch` must hold at least max_keep entries; this
// function performs no allocation.
int prune_candidates(std::span<const CandidateScore> candidates,
                     std::span<int> keep, std::span<int> scratch,
                     const PruneConfig& config) noexcept;

// ---------------------------------------------------------------------------
// Per-superblock probe budget
// ---------------------------------------------------------------------------

class ProbeBudget {
 public:
  void reset(int limit) noexcept {
    limit_ = limit;
    used_ = 0;
    exhausted_ = false;
  }
  bool take() noexcept {
    if (limit_ <= 0) {
      return true;
    }
    if (used_ >= limit_) {
      exhausted_ = true;
      return false;
    }
    ++used_;
    return true;
  }
  int used() const noexcept { return used_; }
  bool exhausted() const noexcept { return exhausted_; }

 private:
  int limit_ = 0;
  int used_ = 0;
  bool exhausted_ = false;
};

}  // namespace av2enc::m1::rdo_loop
