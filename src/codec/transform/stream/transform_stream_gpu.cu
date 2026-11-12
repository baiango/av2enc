#include "codec/transform/stream/transform_stream_gpu.h"

#include "cuda/codec/rate/rdoq/rdoq_batch.h"
#include "cuda/codec/transform/tx_type/tx_type.h"
#include "cuda/codec/transform/variants/tx_variants.h"
#include "codec/rate/rdoq/rdoq_av2_model.h"
#include "codec/transform/tx_type/tx_type.h"
#include "codec/transform/variants/tx_variants.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::still::ts::gpu {
namespace {

using av2enc::cuda::TxBlockJob;
using av2enc::cuda::TxTypeBlockJob;
using av2enc::cuda::TxTypeBlockRequest;
using av2enc::cuda::TxTypeCandJob;
using av2enc::cuda::TxTypeCandidateMetrics;
using av2enc::cuda::TxTypeDecision;
using av2enc::cuda::TxTypePlan;
using txtype::TxType;
using txtype::TxTypeQuery;

constexpr std::size_t kAdjElems =
    static_cast<std::size_t>(cuda::kTxTypeAdjElems);
constexpr std::size_t kCodedElems =
    static_cast<std::size_t>(cuda::kTxTypeCodedElems);

bool cuda_ok(cudaError_t status, const char* what, std::string* error) {
  if (status == cudaSuccess) return true;
  if (error != nullptr) {
    *error = std::string(what) + ": " + cudaGetErrorString(status);
  }
  return false;
}

bool sync_stream(cudaStream_t stream, const char* what, std::string* error) {
  return cuda_ok(cudaStreamSynchronize(stream), what, error);
}

// The adapter owns no device memory of its own.  Every buffer is carved from
// the caller's persistent av2enc::cuda::Arena through this callback pair; there
// is deliberately no cudaMalloc/cudaFree fallback.  Callers that want zero-copy
// scratch should size their arena with the same geometry they pass here.
class DeviceArena {
 public:
  explicit DeviceArena(const DeviceAllocator& allocator)
      : allocator_(allocator) {}
  ~DeviceArena() { reset(); }

  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;

  void reset() {
    for (void* pointer : pointers_) {
      allocator_.deallocate(allocator_.user, pointer);
    }
    pointers_.clear();
  }

  void* allocate(std::size_t bytes, const char* what, std::string* error) {
    if (bytes == 0) bytes = 1;
    void* pointer = allocator_.allocate(allocator_.user, bytes);
    if (pointer == nullptr) {
      if (error != nullptr) {
        *error = std::string("device allocation failed: ") + what;
      }
      return nullptr;
    }
    pointers_.push_back(pointer);
    return pointer;
  }

  template <typename T>
  T* allocate_array(std::size_t count, const char* what, std::string* error) {
    const std::size_t bytes = count * sizeof(T);
    return static_cast<T*>(allocate(bytes, what, error));
  }

 private:
  DeviceAllocator allocator_{};
  std::vector<void*> pointers_;
};

bool upload(cudaStream_t stream, void* device, const void* host,
            std::size_t bytes, const char* what, std::string* error) {
  return cuda_ok(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice,
                                 stream),
                 what, error);
}

bool download(cudaStream_t stream, void* host, const void* device,
              std::size_t bytes, const char* what, std::string* error) {
  return cuda_ok(cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost,
                                 stream),
                 what, error);
}

TxTypeQuery make_query(const Request& request, int mapped_mode) {
  TxTypeQuery query;
  query.coded_w = request.coded_w;
  query.coded_h = request.coded_h;
  query.mapped_mode = mapped_mode;
  query.reduced_tx_set = request.reduced_tx_set;
  query.dct_only = request.dct_only;
  return query;
}

bool validate_request(const Request& request, int* mapped_mode,
                      std::string* error) {
  if (request.residual_coded == nullptr) {
    if (error != nullptr) *error = "null residual";
    return false;
  }
  if (!txv::tx_size_supported(request.coded_w, request.coded_h)) {
    if (error != nullptr) *error = "unsupported coded TX size";
    return false;
  }
  if (!std::isfinite(request.lambda) || request.lambda < 0.0) {
    if (error != nullptr) *error = "lambda must be finite and >= 0";
    return false;
  }
  if (request.prev_level_cap < 0 || request.prev_level_cap > 63) {
    if (error != nullptr) *error = "prev_level_cap must be in [0, 63]";
    return false;
  }
  if (request.level_cap < 0) {
    if (error != nullptr) *error = "level_cap must be >= 0";
    return false;
  }
  if (request.reduced_tx_set < 0 || request.reduced_tx_set > 3) {
    if (error != nullptr) *error = "reduced_tx_set must be in [0, 3]";
    return false;
  }
  if (request.plane < 0 || request.plane > 2) {
    if (error != nullptr) *error = "plane must be 0, 1 or 2";
    return false;
  }
  const int mode = txtype::map_intra_mode(
      request.intra_mode, request.angle_delta, request.coded_w,
      request.coded_h);
  if (mode < 0) {
    if (error != nullptr) *error = "invalid intra mode, angle delta or size";
    return false;
  }
  *mapped_mode = mode;
  return true;
}

struct RpdoqPlan {
  std::vector<double> original;  // scan order, ncand * kAdjElems
  std::vector<std::int32_t> sign;
  std::vector<std::int32_t> bound;
  std::vector<double> normal_rate;
  std::vector<double> eob_rate;
  std::vector<double> block_eob_rate;
  std::vector<cuda::rdoq::BlockJob> jobs;
  std::size_t path_total = 0;
};

bool build_rdoq_plan(const Request* requests,
                     const TxTypePlan& tx_plan,
                     const std::vector<double>& coeff_scan,  // raster slots
                     const std::vector<std::int32_t>& scalar_raster,
                     const RateModelHooks* global_hooks,
                     std::size_t table_budget, RpdoqPlan* plan,
                     std::string* error) {
  const std::size_t candidates = tx_plan.candidates.size();
  plan->original.assign(candidates * kAdjElems, 0.0);
  plan->sign.assign(candidates * kAdjElems, 1);
  plan->bound.assign(candidates * kAdjElems, 0);
  plan->jobs.clear();
  plan->jobs.reserve(candidates);
  plan->normal_rate.clear();
  plan->eob_rate.clear();
  plan->block_eob_rate.clear();
  plan->path_total = 0;

  std::size_t normal_count = 0;
  std::size_t block_eob_count = 0;

  for (std::size_t c = 0; c < candidates; ++c) {
    const TxTypeCandJob& candidate = tx_plan.candidates[c];
    const Request& request = requests[candidate.block];
    const txv::TxParams& params = candidate.tx.params;
    const int area = params.w * params.h;
    if (area <= 0) {
      if (error != nullptr) *error = "invalid candidate geometry";
      return false;
    }

    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    txtype::diagonal_scan(params.w, params.h, scan.data());

    std::vector<double> coeff(static_cast<std::size_t>(area));
    std::vector<std::int32_t> scalar(static_cast<std::size_t>(area));
    std::int32_t scalar_eob = 0;
    const std::size_t slot = c * kAdjElems;
    for (int i = 0; i < area; ++i) {
      const std::size_t raster =
          slot + static_cast<std::size_t>(scan[static_cast<std::size_t>(i)]);
      coeff[static_cast<std::size_t>(i)] = coeff_scan[raster];
      scalar[static_cast<std::size_t>(i)] = scalar_raster[raster];
      if (scalar_raster[raster] != 0) scalar_eob = i + 1;
    }

    rdoq::Config config;
    config.lambda = request.lambda;
    config.quant_step = 1.0 / params.quant_factor;
    config.prev_level_cap = request.prev_level_cap;
    config.level_cap = request.level_cap;
    config.allow_zero_to_one = request.allow_zero_to_one;
    config.max_coefficients = 4096;

    std::int32_t max_bound = 0;
    for (int i = 0; i < area; ++i) {
      rdoq::Coefficient coefficient;
      coefficient.original = coeff[static_cast<std::size_t>(i)];
      coefficient.scalar_level = scalar[static_cast<std::size_t>(i)];
      coefficient.level_context = i;
      const std::int32_t bound = rdoq::rdoq_level_bound(coefficient, config);
      if (bound < 0) {
        if (error != nullptr) *error = "invalid coefficient level bound";
        return false;
      }
      const std::size_t index = slot + static_cast<std::size_t>(i);
      plan->original[index] = coefficient.original;
      plan->bound[index] = bound;
      max_bound = std::max(max_bound, bound);
      if (coefficient.original < 0.0) {
        plan->sign[index] = -1;
      } else if (coefficient.original > 0.0) {
        plan->sign[index] = 1;
      } else {
        plan->sign[index] = coefficient.scalar_level < 0 ? -1 : 1;
      }
    }

    const std::int32_t states = request.prev_level_cap + 1;
    const std::int32_t table_levels = max_bound + 1;
    const std::size_t table_size =
        static_cast<std::size_t>(area) * static_cast<std::size_t>(states) *
        static_cast<std::size_t>(table_levels);

    RateModelRequest model_request;
    model_request.params = params;
    model_request.plane = request.plane;
    model_request.txb_skip_context = request.txb_skip_context;
    model_request.dc_sign_context = request.dc_sign_context;
    model_request.scalar_levels_scan =
        std::span<const std::int32_t>(scalar);
    model_request.scalar_eob = scalar_eob;
    model_request.quant_step = config.quant_step;
    model_request.lambda = request.lambda;

    const RateModelHooks* hooks =
        request.hooks != nullptr ? request.hooks : global_hooks;
    rdoq::av2::Model default_model;
    rdoq::RateModel rate_model;
    bool custom_model = false;
    if (hooks != nullptr && hooks->build != nullptr) {
      std::string build_error;
      if (!hooks->build(hooks->user, model_request, &rate_model,
                        &build_error)) {
        if (error != nullptr) {
          *error = build_error.empty() ? "rate-model hook failed" : build_error;
        }
        return false;
      }
      if (rate_model.coefficient_rate == nullptr ||
          rate_model.eob_rate == nullptr) {
        if (error != nullptr) *error = "rate-model hook returned no model";
        return false;
      }
      custom_model = true;
    } else {
      rdoq::av2::BlockInfo info;
      info.plane = request.plane;
      info.params = params;
      info.txb_skip_context = request.txb_skip_context;
      info.dc_sign_context = request.dc_sign_context;
      std::string init_error;
      if (!default_model.init(info, model_request.scalar_levels_scan,
                              &init_error)) {
        if (error != nullptr) {
          *error = init_error.empty() ? "AV2 rate-model init failed"
                                      : init_error;
        }
        return false;
      }
      rate_model = default_model.rate_model();
    }

    plan->normal_rate.resize(normal_count + table_size, 0.0);
    plan->eob_rate.resize(normal_count + table_size, 0.0);
    for (int i = 0; i < area; ++i) {
      const std::int32_t bound = plan->bound[slot + static_cast<std::size_t>(i)];
      const std::int32_t coefficient_sign =
          plan->sign[slot + static_cast<std::size_t>(i)];
      for (std::int32_t s = 0; s < states; ++s) {
        const std::size_t row =
            normal_count +
            ((static_cast<std::size_t>(i) * static_cast<std::size_t>(states) +
              static_cast<std::size_t>(s)) *
             static_cast<std::size_t>(table_levels));
        const std::size_t first_row =
            normal_count +
            (static_cast<std::size_t>(i) * static_cast<std::size_t>(states) *
             static_cast<std::size_t>(table_levels));
        if (!custom_model && s > 0) {
          // The AV2 model ignores prev_abs_state; copy the state-0 row
          // instead of repeating the transcendental work.
          std::memcpy(&plan->normal_rate[row],
                      &plan->normal_rate[first_row],
                      static_cast<std::size_t>(table_levels) * sizeof(double));
          std::memcpy(&plan->eob_rate[row], &plan->eob_rate[first_row],
                      static_cast<std::size_t>(table_levels) * sizeof(double));
          continue;
        }
        // m == 0 is only queried for a normal (non-EOB) coefficient.
        plan->normal_rate[row] = rate_model.coefficient_rate(
            rate_model.context, i, s, 0, i, 0);
        for (std::int32_t m = 1; m <= bound; ++m) {
          const std::int32_t signed_level = coefficient_sign * m;
          const std::size_t index = row + static_cast<std::size_t>(m);
          plan->normal_rate[index] = rate_model.coefficient_rate(
              rate_model.context, i, s, signed_level, i, 0);
          plan->eob_rate[index] = rate_model.coefficient_rate(
              rate_model.context, i, s, signed_level, i, 1);
        }
      }
    }

    const std::size_t eob_count =
        static_cast<std::size_t>(area) + 1;
    plan->block_eob_rate.resize(block_eob_count + eob_count, 0.0);
    for (std::int32_t e = 0; e <= area; ++e) {
      plan->block_eob_rate[block_eob_count + static_cast<std::size_t>(e)] =
          rate_model.eob_rate(rate_model.context, e, area);
    }

    if (custom_model && hooks != nullptr && hooks->release != nullptr) {
      hooks->release(hooks->user, &rate_model);
    }

    cuda::rdoq::BlockJob job;
    job.coefficient_offset = static_cast<std::int32_t>(slot);
    job.num_coefficients = area;
    job.level_offset = static_cast<std::int32_t>(slot);
    job.path_offset = static_cast<std::int32_t>(plan->path_total);
    job.table_offset = static_cast<std::int32_t>(normal_count);
    job.eob_table_offset = static_cast<std::int32_t>(block_eob_count);
    job.prev_level_cap = request.prev_level_cap;
    job.table_levels = table_levels;
    job.quant_step = config.quant_step;
    job.lambda = request.lambda;
    plan->jobs.push_back(job);

    normal_count += table_size;
    block_eob_count += eob_count;
    plan->path_total += static_cast<std::size_t>(area) *
                        static_cast<std::size_t>(states);

    const std::size_t bytes =
        plan->normal_rate.size() * sizeof(double) * 2 +
        plan->block_eob_rate.size() * sizeof(double) +
        plan->path_total * (sizeof(std::int32_t) + sizeof(std::uint8_t)) +
        plan->original.size() * (sizeof(double) + 2 * sizeof(std::int32_t));
    if (bytes > table_budget) {
      if (error != nullptr) {
        *error = "RDOQ rate tables exceed BatchOptions::table_budget_bytes (" +
                 std::to_string(bytes) + " > " +
                 std::to_string(table_budget) + "); split the block batch";
      }
      return false;
    }

  }
  return true;
}

bool candidate_allowed(const Request& request, int type) {
  const TxType tx_type = static_cast<TxType>(type);
  if (!request.candidate_types.empty()) {
    bool found = false;
    for (const TxType allowed : request.candidate_types) {
      if (allowed == tx_type) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  if (!request.enable_idtx_intra) {
    txv::Tx1dType row = txv::Tx1dType::kDct;
    txv::Tx1dType col = txv::Tx1dType::kDct;
    txv::tx_1d_types(tx_type, &row, &col);
    if (row == txv::Tx1dType::kIdt || col == txv::Tx1dType::kIdt) return false;
  }
  return true;
}

// Keeps the original candidate slots (so the existing scratch offsets remain
// valid) but compacts the per-block candidate ranges to the caller's legal
// subset.  This is how `Request::candidate_types` and enable_idtx_intra = 0
// reach the GPU path.
bool filter_plan(const Request* requests, const TxTypePlan& input,
                 TxTypePlan* output, std::string* error) {
  *output = input;
  output->candidates.clear();
  output->tx_jobs.clear();
  output->candidates.reserve(input.candidates.size());
  output->tx_jobs.reserve(input.tx_jobs.size());
  for (std::size_t b = 0; b < input.blocks.size(); ++b) {
    TxTypeBlockJob& block = output->blocks[b];
    const TxTypeBlockJob& source = input.blocks[b];
    const Request& request = requests[b];
    if (!request.candidate_types.empty()) {
      for (const TxType requested : request.candidate_types) {
        bool legal = false;
        for (int j = 0; j < source.cand_count; ++j) {
          const TxTypeCandJob& candidate =
              input.candidates[static_cast<std::size_t>(source.cand_begin + j)];
          if (static_cast<TxType>(candidate.type) == requested) {
            legal = true;
            break;
          }
        }
        if (!legal) {
          if (error != nullptr) {
            *error = "block " + std::to_string(b) +
                     ": candidate TX type is not legal for this block";
          }
          return false;
        }
      }
    }
    block.cand_begin = static_cast<std::int32_t>(output->candidates.size());
    block.cand_count = 0;
    for (int j = 0; j < source.cand_count; ++j) {
      const int index = source.cand_begin + j;
      const TxTypeCandJob& candidate = input.candidates[static_cast<std::size_t>(index)];
      if (!candidate_allowed(requests[candidate.block], candidate.type)) {
        continue;
      }
      output->candidates.push_back(candidate);
      output->tx_jobs.push_back(input.tx_jobs[static_cast<std::size_t>(index)]);
      ++block.cand_count;
    }
    if (block.cand_count == 0) {
      if (error != nullptr) {
        *error = "block " + std::to_string(b) +
                 " has no candidate after Request filtering";
      }
      return false;
    }
  }
  // Re-pack the surviving candidates into dense slots so every downstream
  // stage can index by candidate number.
  for (std::size_t c = 0; c < output->candidates.size(); ++c) {
    TxBlockJob& job = output->tx_jobs[c];
    job.in_offset = static_cast<std::int32_t>(c) * cuda::kTxTypeAdjElems;
    job.out_offset = static_cast<std::int32_t>(c) * cuda::kTxTypeCodedElems;
    output->candidates[c].tx = job;
  }
  const std::size_t ncandidates = output->candidates.size();
  output->adj_scratch_elems = ncandidates * kAdjElems;
  output->coeff_scratch_elems = ncandidates * kAdjElems;
  output->level_scratch_elems = ncandidates * kAdjElems;
  output->recon_scratch_elems = ncandidates * kCodedElems;
  return true;
}

// Same total order as ts::quantize_block().
bool better(const CandidateOutcome& candidate, const CandidateOutcome& best) {
  if (candidate.cost != best.cost) return candidate.cost < best.cost;
  if (candidate.sse != best.sse) return candidate.sse < best.sse;
  return candidate.type < best.type;
}

}  // namespace

bool evaluate_batch(const Request* requests, int count,
                    const BatchOptions& options,
                    std::vector<CandidateOutcome>* outcomes,
                    std::string* error) {
  if (outcomes == nullptr) {
    if (error != nullptr) *error = "null outcomes";
    return false;
  }
  outcomes->clear();
  if (count < 0 || (count > 0 && requests == nullptr)) {
    if (error != nullptr) *error = "invalid request batch";
    return false;
  }
  if (count == 0) return true;

  // A DeviceAllocator over the caller's arena is mandatory: this adapter has
  // no allocator of its own and no cudaMalloc fallback.
  if (options.allocator.allocate == nullptr ||
      options.allocator.deallocate == nullptr) {
    if (error != nullptr) {
      *error = "BatchOptions::allocator is required and must be backed by "
               "av2enc::cuda::Arena (both allocate and deallocate set)";
    }
    return false;
  }

  std::vector<int> mapped_modes(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    if (!validate_request(requests[i], &mapped_modes[static_cast<std::size_t>(i)],
                          error)) {
      return false;
    }
  }

  std::vector<TxTypeBlockRequest> block_requests(
      static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const Request& request = requests[i];
    TxTypeBlockRequest& block = block_requests[static_cast<std::size_t>(i)];
    block.coded_w = request.coded_w;
    block.coded_h = request.coded_h;
    block.mapped_mode = mapped_modes[static_cast<std::size_t>(i)];
    block.reduced_tx_set = request.reduced_tx_set;
    block.dct_only = request.dct_only ? 1 : 0;
  }

  std::size_t coded_elems = 0;
  std::size_t adj_elems = 0;
  av2enc::cuda::tx_type_pack_layout(block_requests.data(), count, &coded_elems,
                                    &adj_elems);

  std::vector<std::int16_t> residual_coded(coded_elems);
  std::vector<std::int16_t> residual_adj(adj_elems);
  for (int i = 0; i < count; ++i) {
    const TxTypeBlockRequest& block = block_requests[static_cast<std::size_t>(i)];
    const Request& request = requests[i];
    const std::size_t coded_count =
        static_cast<std::size_t>(request.coded_w) *
        static_cast<std::size_t>(request.coded_h);
    std::memcpy(&residual_coded[static_cast<std::size_t>(block.coded_offset)],
                request.residual_coded, coded_count * sizeof(std::int16_t));
    const auto params =
        txv::make_tx_params(request.coded_w, request.coded_h);
    if (!params.has_value()) {
      if (error != nullptr) *error = "unsupported TX size";
      return false;
    }
    txv::downsample_residual(
        request.residual_coded, *params,
        &residual_adj[static_cast<std::size_t>(block.adj_offset)]);
  }

  TxTypePlan raw_plan;
  std::string plan_error;
  if (!av2enc::cuda::tx_type_build_plan(block_requests.data(), count, &raw_plan,
                                        &plan_error)) {
    if (error != nullptr) *error = plan_error.empty() ? "TX-type plan failed"
                                                      : plan_error;
    return false;
  }
  TxTypePlan tx_plan;
  if (!filter_plan(requests, raw_plan, &tx_plan, error)) {
    return false;
  }
  const int candidates = static_cast<int>(tx_plan.candidates.size());

  DeviceArena arena(options.allocator);
  const cudaStream_t stream = options.stream;

  std::int16_t* d_coded =
      arena.allocate_array<std::int16_t>(coded_elems, "d_coded", error);
  std::int16_t* d_adj =
      arena.allocate_array<std::int16_t>(adj_elems, "d_adj", error);
  TxTypeBlockJob* d_blocks = arena.allocate_array<TxTypeBlockJob>(
      static_cast<std::size_t>(count), "d_blocks", error);
  TxTypeCandJob* d_candidates = arena.allocate_array<TxTypeCandJob>(
      static_cast<std::size_t>(candidates), "d_candidates", error);
  TxBlockJob* d_tx_jobs = arena.allocate_array<TxBlockJob>(
      static_cast<std::size_t>(candidates), "d_tx_jobs", error);
  std::int16_t* d_adj_scratch = arena.allocate_array<std::int16_t>(
      tx_plan.adj_scratch_elems, "d_adj_scratch", error);
  double* d_coeff = arena.allocate_array<double>(tx_plan.coeff_scratch_elems,
                                                 "d_coeff", error);
  std::int32_t* d_levels = arena.allocate_array<std::int32_t>(
      tx_plan.level_scratch_elems, "d_levels", error);
  std::int16_t* d_recon = arena.allocate_array<std::int16_t>(
      tx_plan.recon_scratch_elems, "d_recon", error);
  TxTypeCandidateMetrics* d_metrics =
      arena.allocate_array<TxTypeCandidateMetrics>(
          static_cast<std::size_t>(candidates), "d_metrics", error);
  TxTypeDecision* d_decisions = arena.allocate_array<TxTypeDecision>(
      static_cast<std::size_t>(count), "d_decisions", error);
  if (d_coded == nullptr || d_adj == nullptr || d_blocks == nullptr ||
      d_candidates == nullptr || d_tx_jobs == nullptr ||
      d_adj_scratch == nullptr || d_coeff == nullptr || d_levels == nullptr ||
      d_recon == nullptr || d_metrics == nullptr || d_decisions == nullptr) {
    return false;
  }

  if (!upload(stream, d_coded, residual_coded.data(),
              residual_coded.size() * sizeof(std::int16_t), "upload coded",
              error) ||
      !upload(stream, d_adj, residual_adj.data(),
              residual_adj.size() * sizeof(std::int16_t), "upload adj", error) ||
      !upload(stream, d_blocks, tx_plan.blocks.data(),
              tx_plan.blocks.size() * sizeof(TxTypeBlockJob), "upload blocks",
              error) ||
      !upload(stream, d_candidates, tx_plan.candidates.data(),
              tx_plan.candidates.size() * sizeof(TxTypeCandJob),
              "upload candidates", error) ||
      !upload(stream, d_tx_jobs, tx_plan.tx_jobs.data(),
              tx_plan.tx_jobs.size() * sizeof(TxBlockJob), "upload tx jobs",
              error)) {
    return false;
  }

  av2enc::cuda::TxTypeCostModel cost_model;
  cost_model.lambda = 0.0;
  cost_model.coeff_bits = 0.0;
  cost_model.use_type_bits = false;
  if (!cuda_ok(av2enc::cuda::tx_type_choose_batch(
                   d_coded, d_adj, d_blocks, count, d_candidates, candidates,
                   d_tx_jobs, d_adj_scratch, d_coeff, d_levels, d_recon,
                   d_metrics, d_decisions, cost_model, stream),
               "tx_type_choose_batch", error)) {
    return false;
  }
  if (!sync_stream(stream, "tx_type_choose_batch sync", error)) return false;

  std::vector<double> coeff(tx_plan.coeff_scratch_elems, 0.0);
  std::vector<std::int32_t> scalar(tx_plan.level_scratch_elems, 0);
  if (!download(stream, coeff.data(), d_coeff,
                coeff.size() * sizeof(double), "download coeff", error) ||
      !download(stream, scalar.data(), d_levels,
                scalar.size() * sizeof(std::int32_t), "download levels",
                error) ||
      !sync_stream(stream, "download transforms sync", error)) {
    return false;
  }

  RpdoqPlan rdoq_plan;
  if (!build_rdoq_plan(requests, tx_plan, coeff, scalar, options.hooks,
                       options.table_budget_bytes, &rdoq_plan, error)) {
    return false;
  }

  const std::size_t table_doubles = rdoq_plan.normal_rate.size();
  double* d_original = nullptr;
  // d_coeff already has the right layout (candidate slot * kAdjElems), so it
  // can be reused as the scan-order RDOQ input.
  d_original = d_coeff;
  if (!upload(stream, d_original, rdoq_plan.original.data(),
              rdoq_plan.original.size() * sizeof(double), "upload original",
              error)) {
    return false;
  }

  std::int32_t* d_sign = arena.allocate_array<std::int32_t>(
      rdoq_plan.sign.size(), "d_sign", error);
  std::int32_t* d_bound = arena.allocate_array<std::int32_t>(
      rdoq_plan.bound.size(), "d_bound", error);
  double* d_normal = arena.allocate_array<double>(
      table_doubles, "d_normal", error);
  double* d_eob = arena.allocate_array<double>(table_doubles, "d_eob", error);
  double* d_block_eob = arena.allocate_array<double>(
      rdoq_plan.block_eob_rate.size(), "d_block_eob", error);
  cuda::rdoq::BlockJob* d_rdoq_jobs = arena.allocate_array<cuda::rdoq::BlockJob>(
      rdoq_plan.jobs.size(), "d_rdoq_jobs", error);
  std::int32_t* d_path_levels = arena.allocate_array<std::int32_t>(
      rdoq_plan.path_total, "d_path_levels", error);
  std::uint8_t* d_path_modes = arena.allocate_array<std::uint8_t>(
      rdoq_plan.path_total, "d_path_modes", error);
  std::int32_t* d_rdoq_eob = arena.allocate_array<std::int32_t>(
      static_cast<std::size_t>(candidates), "d_rdoq_eob", error);
  double* d_distortion = arena.allocate_array<double>(
      static_cast<std::size_t>(candidates), "d_distortion", error);
  double* d_rate = arena.allocate_array<double>(
      static_cast<std::size_t>(candidates), "d_rate", error);
  double* d_rdoq_cost = arena.allocate_array<double>(
      static_cast<std::size_t>(candidates), "d_rdoq_cost", error);
  std::int32_t* d_status = arena.allocate_array<std::int32_t>(
      static_cast<std::size_t>(candidates), "d_status", error);
  if (d_sign == nullptr || d_bound == nullptr || d_normal == nullptr ||
      d_eob == nullptr || d_block_eob == nullptr || d_rdoq_jobs == nullptr ||
      d_path_levels == nullptr || d_path_modes == nullptr ||
      d_rdoq_eob == nullptr || d_distortion == nullptr || d_rate == nullptr ||
      d_rdoq_cost == nullptr || d_status == nullptr) {
    return false;
  }

  if (!upload(stream, d_sign, rdoq_plan.sign.data(),
              rdoq_plan.sign.size() * sizeof(std::int32_t), "upload sign",
              error) ||
      !upload(stream, d_bound, rdoq_plan.bound.data(),
              rdoq_plan.bound.size() * sizeof(std::int32_t), "upload bound",
              error) ||
      !upload(stream, d_normal, rdoq_plan.normal_rate.data(),
              rdoq_plan.normal_rate.size() * sizeof(double),
              "upload normal rate", error) ||
      !upload(stream, d_eob, rdoq_plan.eob_rate.data(),
              rdoq_plan.eob_rate.size() * sizeof(double), "upload eob rate",
              error) ||
      !upload(stream, d_block_eob, rdoq_plan.block_eob_rate.data(),
              rdoq_plan.block_eob_rate.size() * sizeof(double),
              "upload block eob rate", error) ||
      !upload(stream, d_rdoq_jobs, rdoq_plan.jobs.data(),
              rdoq_plan.jobs.size() * sizeof(cuda::rdoq::BlockJob),
              "upload rdoq jobs", error)) {
    return false;
  }

  cuda::rdoq::Batch batch;
  batch.original = d_original;
  batch.sign = d_sign;
  batch.bound = d_bound;
  batch.normal_rate = d_normal;
  batch.eob_rate = d_eob;
  batch.block_eob_rate = d_block_eob;
  batch.jobs = d_rdoq_jobs;
  batch.num_jobs = candidates;
  batch.levels = d_levels;
  batch.eob = d_rdoq_eob;
  batch.distortion = d_distortion;
  batch.rate_bits = d_rate;
  batch.cost = d_rdoq_cost;
  batch.status = d_status;
  batch.path_levels = d_path_levels;
  batch.path_modes = d_path_modes;
  if (!cuda_ok(cuda::rdoq::quantize_batch(batch, stream), "rdoq_quantize_batch",
               error) ||
      !sync_stream(stream, "rdoq sync", error)) {
    return false;
  }

  std::vector<std::int32_t> levels_scan(scalar.size(), 0);
  std::vector<std::int32_t> eob(static_cast<std::size_t>(candidates), 0);
  std::vector<double> distortion(static_cast<std::size_t>(candidates), 0.0);
  std::vector<double> rate(static_cast<std::size_t>(candidates), 0.0);
  std::vector<double> rdoq_cost(static_cast<std::size_t>(candidates), 0.0);
  std::vector<std::int32_t> status(static_cast<std::size_t>(candidates), -1);
  if (!download(stream, levels_scan.data(), d_levels,
                levels_scan.size() * sizeof(std::int32_t), "download rdoq levels",
                error) ||
      !download(stream, eob.data(), d_rdoq_eob,
                eob.size() * sizeof(std::int32_t), "download eob", error) ||
      !download(stream, distortion.data(), d_distortion,
                distortion.size() * sizeof(double), "download distortion",
                error) ||
      !download(stream, rate.data(), d_rate, rate.size() * sizeof(double),
                "download rate", error) ||
      !download(stream, rdoq_cost.data(), d_rdoq_cost,
                rdoq_cost.size() * sizeof(double), "download rdoq cost",
                error) ||
      !download(stream, status.data(), d_status,
                status.size() * sizeof(std::int32_t), "download status",
                error) ||
      !sync_stream(stream, "download rdoq sync", error)) {
    return false;
  }

  // RDOQ levels are scan order.  The inverse transform expects raster order.
  std::vector<std::int32_t> raster_levels(levels_scan.size(), 0);
  for (int c = 0; c < candidates; ++c) {
    const TxTypeCandJob& candidate = tx_plan.candidates[static_cast<std::size_t>(c)];
    const txv::TxParams& params = candidate.tx.params;
    const int area = params.w * params.h;
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    txtype::diagonal_scan(params.w, params.h, scan.data());
    const std::size_t slot = static_cast<std::size_t>(c) * kAdjElems;
    const int count_coded = std::min(eob[static_cast<std::size_t>(c)], area);
    for (int i = 0; i < count_coded; ++i) {
      raster_levels[slot +
                    static_cast<std::size_t>(scan[static_cast<std::size_t>(i)])] =
          levels_scan[slot + static_cast<std::size_t>(i)];
    }
  }
  if (!upload(stream, d_levels, raster_levels.data(),
              raster_levels.size() * sizeof(std::int32_t), "upload raster",
              error) ||
      !cuda_ok(av2enc::cuda::txv_inverse_transform_multi(
                   d_levels, d_recon, d_tx_jobs, candidates, stream),
               "txv_inverse_transform_multi", error) ||
      !sync_stream(stream, "inverse sync", error)) {
    return false;
  }

  std::vector<std::int16_t> recon(tx_plan.recon_scratch_elems, 0);
  if (!download(stream, recon.data(), d_recon,
                recon.size() * sizeof(std::int16_t), "download recon",
                error) ||
      !sync_stream(stream, "download recon sync", error)) {
    return false;
  }

  outcomes->resize(static_cast<std::size_t>(candidates));
  for (int c = 0; c < candidates; ++c) {
    const TxTypeCandJob& candidate = tx_plan.candidates[static_cast<std::size_t>(c)];
    const Request& request = requests[candidate.block];
    const txv::TxParams& params = candidate.tx.params;
    const int area = params.w * params.h;
    const int coded_area = params.coded_w * params.coded_h;
    const std::size_t slot = static_cast<std::size_t>(c) * kAdjElems;
    const std::size_t coded_slot =
        static_cast<std::size_t>(c) * kCodedElems;

    CandidateOutcome& outcome = (*outcomes)[static_cast<std::size_t>(c)];
    outcome.block = candidate.block;
    outcome.candidate_index = candidate.candidate;
    outcome.type = candidate.type;
    outcome.eob = eob[static_cast<std::size_t>(c)];
    outcome.status = status[static_cast<std::size_t>(c)];
    outcome.rdoq_distortion = distortion[static_cast<std::size_t>(c)];
    outcome.rdoq_rate_bits = rate[static_cast<std::size_t>(c)];
    outcome.rdoq_cost = rdoq_cost[static_cast<std::size_t>(c)];
    outcome.levels_scan.assign(
        levels_scan.begin() + static_cast<std::ptrdiff_t>(slot),
        levels_scan.begin() + static_cast<std::ptrdiff_t>(slot) +
            static_cast<std::ptrdiff_t>(area));
    outcome.recon_coded.assign(
        recon.begin() + static_cast<std::ptrdiff_t>(coded_slot),
        recon.begin() + static_cast<std::ptrdiff_t>(coded_slot) +
            static_cast<std::ptrdiff_t>(coded_area));

    const TxTypeQuery query =
        make_query(request, mapped_modes[static_cast<std::size_t>(candidate.block)]);
    outcome.symbol = classify_tx_type(
        query, static_cast<TxType>(outcome.type), outcome.eob);
    outcome.codable = outcome.symbol.valid &&
                      txtype::candidate_codable(
                          static_cast<TxType>(outcome.type), outcome.eob);

    std::int32_t nonzero = 0;
    for (int i = 0; i < area; ++i) {
      if (outcome.levels_scan[static_cast<std::size_t>(i)] != 0) ++nonzero;
    }
    outcome.nonzero = nonzero;

    std::int64_t sse = 0;
    for (int i = 0; i < coded_area; ++i) {
      const std::int64_t difference =
          static_cast<std::int64_t>(
              outcome.recon_coded[static_cast<std::size_t>(i)]) -
          request.residual_coded[i];
      sse += difference * difference;
    }
    outcome.sse = static_cast<double>(sse);

    double type_bits = 0.0;
    const RateModelHooks* hooks =
        request.hooks != nullptr ? request.hooks : options.hooks;
    if (outcome.symbol.valid &&
        outcome.symbol.kind != TxTypeSyntaxKind::kNone) {
      if (hooks != nullptr && hooks->type_bits != nullptr) {
        txtype::TxTypeCandidate candidate_record;
        candidate_record.type = static_cast<TxType>(outcome.type);
        candidate_record.symbol = outcome.symbol.symbol;
        candidate_record.is_long_side_dct = outcome.symbol.is_long_side_dct;
        candidate_record.codes_long_flag = outcome.symbol.has_long_flag;
        candidate_record.params = params;
        type_bits =
            hooks->type_bits(hooks->user, query, candidate_record);
      } else {
        type_bits = outcome.symbol.bits;
      }
    }
    outcome.type_bits = type_bits;
    outcome.coeff_rate_bits = outcome.rdoq_rate_bits;
    outcome.rate_bits = outcome.rdoq_rate_bits + type_bits;
    outcome.cost =
        outcome.sse + request.lambda * outcome.rate_bits;
  }

  return true;
}

bool quantize_batch(const Request* requests, int count,
                    const BatchOptions& options, Decision* decisions,
                    std::string* error) {
  if (decisions == nullptr) {
    if (error != nullptr) *error = "null decisions";
    return false;
  }
  std::vector<CandidateOutcome> outcomes;
  if (!evaluate_batch(requests, count, options, &outcomes, error)) {
    for (int i = 0; i < count; ++i) decisions[i] = Decision{};
    return false;
  }

  int index = 0;
  for (int block = 0; block < count; ++block) {
    Decision& decision = decisions[block];
    decision = Decision{};
    const int begin = index;
    while (index < static_cast<int>(outcomes.size()) &&
           outcomes[static_cast<std::size_t>(index)].block == block) {
      ++index;
    }
    const int end = index;
    const CandidateOutcome* best = nullptr;
    for (int i = begin; i < end; ++i) {
      const CandidateOutcome& outcome = outcomes[static_cast<std::size_t>(i)];
      if (!outcome.codable || outcome.status != 0) continue;
      if (best == nullptr || better(outcome, *best)) best = &outcome;
    }
    if (best == nullptr) {
      if (error != nullptr) {
        *error = "block " + std::to_string(block) +
                 " has no codable TX-type candidate";
      }
      return false;
    }

    const auto params = txv::make_tx_params(
        requests[block].coded_w, requests[block].coded_h,
        static_cast<TxType>(best->type));
    if (!params.has_value()) {
      if (error != nullptr) *error = "winner has an unsupported TX type";
      return false;
    }
    decision.valid = true;
    decision.codable = true;
    decision.type = static_cast<TxType>(best->type);
    decision.params = *params;
    decision.symbol = best->symbol;
    decision.eob = best->eob;
    decision.nonzero = best->nonzero;
    decision.sse = best->sse;
    decision.coeff_rate_bits = best->coeff_rate_bits;
    decision.type_bits = best->type_bits;
    decision.rate_bits = best->rate_bits;
    decision.cost = best->cost;
    decision.rdoq_distortion = best->rdoq_distortion;
    decision.rdoq_rate_bits = best->rdoq_rate_bits;
    decision.rdoq_cost = best->rdoq_cost;
    decision.candidates_considered = end - begin;
    decision.levels = best->levels_scan;
    decision.recon_coded = best->recon_coded;
  }
  return true;
}

}  // namespace av2enc::still::ts::gpu
