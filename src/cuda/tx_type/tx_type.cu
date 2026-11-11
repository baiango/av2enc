// CUDA batch evaluation of AV2 intra transform-type candidates.
// See tx_type.h for the contract; the transform arithmetic lives in
// src/cuda/tx_variants/ and is reused unchanged.
#include "cuda/tx_type/tx_type.h"

#include <algorithm>

namespace av2enc::cuda {
namespace {

constexpr int kThreads = 128;

// One CUDA block per candidate.  Copies the block's adjusted residual into the
// candidate's slot so the tx_variants batch kernels can key outputs by offset.
__global__ void duplicate_adj_kernel(const std::int16_t* __restrict__ adj_block,
                                     std::int16_t* __restrict__ adj_scratch,
                                     const TxTypeCandJob* __restrict__ cands) {
  const TxTypeCandJob c = cands[blockIdx.x];
  const int elems = c.tx.params.w * c.tx.params.h;
  for (int i = static_cast<int>(threadIdx.x); i < elems;
       i += static_cast<int>(blockDim.x)) {
    adj_scratch[c.tx.in_offset + i] = adj_block[c.adj_src + i];
  }
}

// One CUDA block per candidate: coded-domain SSE, nonzero count, |level| sum
// and the eob == 1 flag.  The int64 sums are exact, so the block reduction
// order does not affect the result.
__global__ void metrics_kernel(const std::int16_t* __restrict__ coded,
                               const std::int32_t* __restrict__ levels,
                               const std::int16_t* __restrict__ recon,
                               const TxTypeBlockJob* __restrict__ blocks,
                               const TxTypeCandJob* __restrict__ cands,
                               TxTypeCandidateMetrics* __restrict__ metrics) {
  const int cand_index = static_cast<int>(blockIdx.x);
  const TxTypeCandJob c = cands[cand_index];
  const TxTypeBlockJob b = blocks[c.block];
  const int adj_n = c.tx.params.w * c.tx.params.h;
  const int coded_n = b.coded_w * b.coded_h;

  __shared__ long long s_err[kThreads];
  __shared__ long long s_abs[kThreads];
  __shared__ int s_nz[kThreads];

  const int tid = static_cast<int>(threadIdx.x);
  long long err = 0;
  long long abs_sum = 0;
  int nz = 0;
  for (int i = tid; i < coded_n; i += kThreads) {
    const long long d = static_cast<long long>(recon[c.tx.out_offset + i]) -
                        static_cast<long long>(coded[b.coded + i]);
    err += d * d;
  }
  for (int i = tid; i < adj_n; i += kThreads) {
    const std::int32_t level = levels[c.tx.in_offset + i];
    if (level != 0) {
      ++nz;
      abs_sum += level < 0 ? -static_cast<long long>(level) : level;
    }
  }
  s_err[tid] = err;
  s_abs[tid] = abs_sum;
  s_nz[tid] = nz;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_err[tid] += s_err[tid + stride];
      s_abs[tid] += s_abs[tid + stride];
      s_nz[tid] += s_nz[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    TxTypeCandidateMetrics m;
    m.sse = static_cast<double>(s_err[0]);
    m.nonzero = s_nz[0];
    m.sum_abs = s_abs[0];
    m.eob_one =
        (m.nonzero == 1 && levels[c.tx.in_offset] != 0) ? 1 : 0;
    metrics[cand_index] = m;
  }
}

// One thread per source block.  Applies the CPU default rate model and the
// deterministic tie-break (cost, then SSE, then TX_TYPE value).
__global__ void decide_kernel(const TxTypeCandJob* __restrict__ cands,
                              const TxTypeCandidateMetrics* __restrict__ metrics,
                              const TxTypeBlockJob* __restrict__ blocks,
                              TxTypeCostModel cost, int nblocks,
                              TxTypeDecision* __restrict__ decisions) {
  const int b = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (b >= nblocks) {
    return;
  }
  const TxTypeBlockJob job = blocks[b];
  bool have = false;
  double best_cost = 0.0;
  double best_sse = 0.0;
  int best_type = 0;
  int best_index = -1;
  for (int i = 0; i < job.cand_count; ++i) {
    const int index = job.cand_begin + i;
    const TxTypeCandJob cand = cands[index];
    const TxTypeCandidateMetrics m = metrics[index];
    // §5.20.8.2: eob <= 1 forces DCT_DCT.  eob == 0 means every level is zero,
    // eob == 1 means the only nonzero level is DC.
    const bool codable =
        cand.type == 0 || m.nonzero >= 2 || (m.nonzero == 1 && m.eob_one == 0);
    if (!codable) {
      continue;
    }
    double bits = cost.coeff_bits * static_cast<double>(m.nonzero);
    if (cost.use_type_bits) {
      bits += cand.type_bits;
    }
    const double candidate_cost = m.sse + cost.lambda * bits;
    const bool better =
        !have || candidate_cost < best_cost ||
        (candidate_cost == best_cost &&
         (m.sse < best_sse ||
          (m.sse == best_sse && cand.type < best_type)));
    if (better) {
      have = true;
      best_cost = candidate_cost;
      best_sse = m.sse;
      best_type = cand.type;
      best_index = i;
    }
  }
  TxTypeDecision decision;
  decision.type = best_type;
  decision.candidate_index = best_index;
  decision.cost = best_cost;
  decision.sse = best_sse;
  decision.nonzero = 0;
  decision.eob_one = 0;
  if (best_index >= 0) {
    const TxTypeCandidateMetrics m = metrics[job.cand_begin + best_index];
    decision.nonzero = m.nonzero;
    decision.eob_one = m.eob_one;
  }
  decisions[b] = decision;
}

}  // namespace

void tx_type_pack_layout(TxTypeBlockRequest* requests, int nblocks,
                         std::size_t* coded_elems, std::size_t* adj_elems) {
  std::size_t coded = 0;
  std::size_t adj = 0;
  for (int i = 0; i < nblocks; ++i) {
    TxTypeBlockRequest& r = requests[i];
    r.coded_offset = static_cast<std::int32_t>(coded);
    r.adj_offset = static_cast<std::int32_t>(adj);
    coded += static_cast<std::size_t>(r.coded_w) *
             static_cast<std::size_t>(r.coded_h);
    const auto params = av2enc::txv::make_tx_params(r.coded_w, r.coded_h);
    if (params.has_value()) {
      adj += static_cast<std::size_t>(params->w) *
             static_cast<std::size_t>(params->h);
    }
  }
  if (coded_elems != nullptr) {
    *coded_elems = coded;
  }
  if (adj_elems != nullptr) {
    *adj_elems = adj;
  }
}

bool tx_type_build_plan(const TxTypeBlockRequest* requests, int nblocks,
                        TxTypePlan* plan, std::string* error) {
  if (requests == nullptr || plan == nullptr || nblocks < 0) {
    if (error != nullptr) {
      *error = "invalid arguments";
    }
    return false;
  }
  plan->blocks.clear();
  plan->candidates.clear();
  plan->tx_jobs.clear();
  plan->blocks.reserve(static_cast<std::size_t>(nblocks));
  int candidate_slot = 0;
  for (int i = 0; i < nblocks; ++i) {
    const TxTypeBlockRequest& r = requests[i];
    txtype::TxTypeQuery query;
    query.coded_w = r.coded_w;
    query.coded_h = r.coded_h;
    query.mapped_mode = r.mapped_mode;
    query.reduced_tx_set = r.reduced_tx_set;
    query.dct_only = r.dct_only != 0;
    txtype::TxTypeSet set;
    if (!txtype::intra_tx_candidates(query, &set)) {
      if (error != nullptr) {
        *error = "unsupported TX size or intra mode";
      }
      return false;
    }
    TxTypeBlockJob block;
    block.cand_begin = static_cast<std::int32_t>(plan->candidates.size());
    block.cand_count = set.count;
    block.coded = r.coded_offset;
    block.coded_w = r.coded_w;
    block.coded_h = r.coded_h;
    plan->blocks.push_back(block);
    for (int j = 0; j < set.count; ++j) {
      const txtype::TxTypeCandidate& candidate = set.candidates[j];
      TxTypeCandJob job;
      job.tx.params = candidate.params;
      job.tx.in_offset =
          static_cast<std::int32_t>(candidate_slot) * kTxTypeAdjElems;
      job.tx.out_offset =
          static_cast<std::int32_t>(candidate_slot) * kTxTypeCodedElems;
      job.adj_src = r.adj_offset;
      job.block = i;
      job.type = static_cast<std::int32_t>(candidate.type);
      job.candidate = j;
      job.type_bits = txtype::static_type_bits(query, candidate);
      plan->candidates.push_back(job);
      plan->tx_jobs.push_back(job.tx);
      ++candidate_slot;
    }
  }
  const std::size_t ncandidates = plan->candidates.size();
  plan->adj_scratch_elems = ncandidates * kTxTypeAdjElems;
  plan->coeff_scratch_elems = ncandidates * kTxTypeAdjElems;
  plan->level_scratch_elems = ncandidates * kTxTypeAdjElems;
  plan->recon_scratch_elems = ncandidates * kTxTypeCodedElems;
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

cudaError_t tx_type_choose_batch(
    const std::int16_t* d_residual_coded,
    const std::int16_t* d_residual_adj_block,
    const TxTypeBlockJob* d_blocks, int nblocks,
    const TxTypeCandJob* d_candidates, int ncandidates,
    const TxBlockJob* d_tx_jobs, std::int16_t* d_adj_scratch,
    double* d_coeff_scratch, std::int32_t* d_levels,
    std::int16_t* d_recon_scratch, TxTypeCandidateMetrics* d_metrics,
    TxTypeDecision* d_decisions, const TxTypeCostModel& cost,
    cudaStream_t stream) {
  if (nblocks < 0 || ncandidates < 0) {
    return cudaErrorInvalidValue;
  }
  if (nblocks == 0 || ncandidates == 0) {
    return cudaSuccess;
  }
  if (d_residual_coded == nullptr || d_residual_adj_block == nullptr ||
      d_blocks == nullptr || d_candidates == nullptr || d_tx_jobs == nullptr ||
      d_adj_scratch == nullptr || d_coeff_scratch == nullptr ||
      d_levels == nullptr || d_recon_scratch == nullptr || d_metrics == nullptr ||
      d_decisions == nullptr) {
    return cudaErrorInvalidValue;
  }

  duplicate_adj_kernel<<<static_cast<unsigned>(ncandidates), kThreads, 0,
                         stream>>>(d_residual_adj_block, d_adj_scratch,
                                   d_candidates);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  status = txv_forward_transform_multi(d_adj_scratch, d_coeff_scratch, d_levels,
                                       d_tx_jobs, ncandidates, stream);
  if (status != cudaSuccess) {
    return status;
  }
  status = txv_inverse_transform_multi(d_levels, d_recon_scratch, d_tx_jobs,
                                       ncandidates, stream);
  if (status != cudaSuccess) {
    return status;
  }

  metrics_kernel<<<static_cast<unsigned>(ncandidates), kThreads, 0, stream>>>(
      d_residual_coded, d_levels, d_recon_scratch, d_blocks, d_candidates,
      d_metrics);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  const int decision_blocks = (nblocks + kThreads - 1) / kThreads;
  decide_kernel<<<static_cast<unsigned>(decision_blocks), kThreads, 0,
                  stream>>>(d_candidates, d_metrics, d_blocks, cost, nblocks,
                            d_decisions);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
