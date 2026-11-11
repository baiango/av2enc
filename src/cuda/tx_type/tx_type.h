// GPU batch evaluation of AV2 intra transform-type candidates.
//
// CPU reference: `src/encoder/tx_type/` (candidate sets, symbols, rate model,
// decision) and `src/encoder/tx_variants/` (the transform itself).
//
// This layer does not re-implement any transform arithmetic.  It reuses the
// bit-exact batch kernels in `src/cuda/tx_variants/`:
//
//   1. duplicate_adjusted  copies each block's adjusted residual into a
//                          per-candidate scratch slot (the tx_variants batch
//                          kernels key their outputs by the input offset),
//   2. txv_forward_transform_multi   forward + quantize for every candidate,
//   3. txv_inverse_transform_multi   dequantize + inverse for every candidate,
//   4. tx_type_metrics               SSE / nonzero / sum|level| / eob==1 per
//                                    candidate,
//   5. tx_type_decide                per block: cost + deterministic tie-break.
//
// Five launches per batch, any mix of sizes and types.  All buffers are
// caller-owned device memory (arena or test scaffolding); nothing here
// allocates, frees or synchronizes.
//
// Arithmetic contract: levels and reconstructed residuals are bit-identical to
// the CPU reference by construction (same kernels), and decisions match
// `txtype::choose_intra_tx_type()` when the same cost model and candidate list
// are used.  The test compares every candidate, not just the winner.
//
// The GPU supports the default cost model:
//
//   rate = coeff_bits * nonzero [+ type_bits when use_type_bits]
//   cost = sse + lambda * rate
//
// `type_bits` is computed on the host by `txtype::static_type_bits()` so both
// sides see exactly the same §9.3 symbol cost.  A custom CPU `rate_fn` cannot
// be uploaded; use the per-candidate metrics to re-cost on the host.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cuda/tx_variants/tx_variants.h"
#include "encoder/tx_type/tx_type.h"
#include "encoder/tx_variants/tx_variants.h"

namespace av2enc::cuda {

using av2enc::txv::TxParams;

// Fixed candidate slots per block (the largest intra set is WIDE_32/HIGH_32
// with 8 candidates).
inline constexpr int kTxTypeMaxCandidates = txtype::kMaxTxTypeCandidates;
// Scratch stride per candidate.  The tx_variants `_multi` kernels write their
// outputs at the input offset, so every candidate needs its own slot of the
// maximum adjusted block (32x32) / coded block (64x64) size.
inline constexpr int kTxTypeAdjElems = av2enc::txv::kMaxAdjTxElems;
inline constexpr int kTxTypeCodedElems =
    av2enc::txv::kMaxCodedTxDim * av2enc::txv::kMaxCodedTxDim;

// One candidate.  `tx` is passed to the tx_variants batch kernels; the rest is
// bookkeeping for the duplicate/metrics/decide stages.
struct TxTypeCandJob {
  TxBlockJob tx;           // params + adj_dst (in_offset) + recon (out_offset)
  std::int32_t adj_src = 0;  // block's adjusted residual in d_residual_adj_block
  std::int32_t block = 0;    // owning source block index
  std::int32_t type = 0;     // TX_TYPE value (report + tie-break)
  std::int32_t candidate = 0;  // position in the block's candidate list
  double type_bits = 0.0;    // txtype::static_type_bits() for this candidate
};

struct TxTypeBlockJob {
  std::int32_t cand_begin = 0;
  std::int32_t cand_count = 0;
  std::int32_t coded = 0;  // block residual offset in d_residual_coded
  std::int32_t coded_w = 0;
  std::int32_t coded_h = 0;
};

struct TxTypeCostModel {
  double lambda = 0.0;
  double coeff_bits = 1.0;
  bool use_type_bits = true;
};

struct TxTypeCandidateMetrics {
  double sse = 0.0;
  std::int32_t nonzero = 0;
  std::int64_t sum_abs = 0;
  // eob == 1: exactly one nonzero level and it is the DC (scan position 0).
  // §5.20.8.2 forces DCT_DCT in that case.
  std::int32_t eob_one = 0;
};

struct TxTypeDecision {
  std::int32_t type = 0;
  std::int32_t candidate_index = -1;  // position in the block's list
  std::int32_t nonzero = 0;
  std::int32_t eob_one = 0;
  double cost = 0.0;
  double sse = 0.0;
};

// ---------------------------------------------------------------------------
// Host-side plan construction
// ---------------------------------------------------------------------------

struct TxTypeBlockRequest {
  std::int32_t coded_w = 0;  // 4..64
  std::int32_t coded_h = 0;  // 4..64
  std::int32_t mapped_mode = 0;
  std::int32_t reduced_tx_set = 0;
  std::int32_t dct_only = 0;
  std::int32_t coded_offset = 0;  // element offset in d_residual_coded
  std::int32_t adj_offset = 0;    // element offset in d_residual_adj_block
};

struct TxTypePlan {
  std::vector<TxTypeBlockJob> blocks;
  std::vector<TxTypeCandJob> candidates;
  std::vector<TxBlockJob> tx_jobs;  // same order as candidates
  // Required scratch sizes, in elements.
  std::size_t adj_scratch_elems = 0;
  std::size_t coeff_scratch_elems = 0;
  std::size_t level_scratch_elems = 0;
  std::size_t recon_scratch_elems = 0;
};

// Packs a block list into the two linear input buffers:
//   coded: block b at requests[b].coded_offset, coded_w*coded_h elements
//   adj:   block b at requests[b].adj_offset, adjusted w*h elements
// Writes the offsets back and returns the total element counts.
void tx_type_pack_layout(TxTypeBlockRequest* requests, int nblocks,
                         std::size_t* coded_elems, std::size_t* adj_elems);

// Fills the plan with the §5.20.8.2 candidate list of every block (via
// txtype::intra_tx_candidates), the per-candidate TxBlockJob offsets and the
// static §9.3 type cost.  `type_bits` is always populated; the device decide
// kernel adds it only when TxTypeCostModel::use_type_bits is set.  Returns
// false and sets *error for an unsupported size, invalid mode, or out-of-range
// reduced_tx_set.
//
// The returned offsets are trusted by the device entry point (validating them
// on the device would need a sync), so build plans with this function or
// reproduce its layout exactly.
bool tx_type_build_plan(const TxTypeBlockRequest* requests, int nblocks,
                        TxTypePlan* plan, std::string* error);

// ---------------------------------------------------------------------------
// Device entry point
// ---------------------------------------------------------------------------

// Evaluates every candidate of every block and writes one decision per block.
//
//   d_residual_coded      input, packed coded-size residuals (int16)
//   d_residual_adj_block  input, one adjusted residual per block (int16)
//   d_blocks              device array of nblocks TxTypeBlockJob
//   d_candidates          device array of ncandidates TxTypeCandJob
//   d_tx_jobs             device array of ncandidates TxBlockJob (in the same
//                         order as d_candidates; pass plan.tx_jobs)
//   d_adj_scratch         ncandidates * kTxTypeAdjElems int16
//   d_coeff_scratch       ncandidates * kTxTypeAdjElems double
//   d_levels              ncandidates * kTxTypeAdjElems int32
//   d_recon_scratch       ncandidates * kTxTypeCodedElems int16
//   d_metrics             ncandidates TxTypeCandidateMetrics (required; also
//                         what a host-side custom rate model can re-cost)
//   d_decisions           nblocks TxTypeDecision
//
// All pointers must be device pointers on `stream`.  No allocation, no sync.
cudaError_t tx_type_choose_batch(
    const std::int16_t* d_residual_coded,
    const std::int16_t* d_residual_adj_block,
    const TxTypeBlockJob* d_blocks, int nblocks,
    const TxTypeCandJob* d_candidates, int ncandidates,
    const TxBlockJob* d_tx_jobs, std::int16_t* d_adj_scratch,
    double* d_coeff_scratch, std::int32_t* d_levels,
    std::int16_t* d_recon_scratch, TxTypeCandidateMetrics* d_metrics,
    TxTypeDecision* d_decisions, const TxTypeCostModel& cost,
    cudaStream_t stream);

}  // namespace av2enc::cuda
