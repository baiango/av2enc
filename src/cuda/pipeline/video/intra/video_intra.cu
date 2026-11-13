// M2 video intra candidate generator: kernels + arena workspace.
//
// See video_intra.h for the contract.  Prediction is delegated wholesale to
// the shared device body in
// `src/cuda/codec/intra/modes/intra_predict_device.cuh`; this translation unit
// only expands candidates, stages descriptors, and launches the batch.
#include "cuda/pipeline/video/intra/video_intra.h"

#include "cuda/pipeline/video/intra/video_intra_device.cuh"
#include "cuda/pipeline/video/intra/video_intra_host.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace av2enc::cuda::video::intra {
namespace {

namespace detail_v = detail;

constexpr std::uint64_t kAlign = Arena::kAlignment;

std::uint64_t align_up(std::uint64_t value) noexcept {
  return ((value + kAlign - 1) / kAlign) * kAlign;
}

bool valid_dim(int value) noexcept {
  return value == 4 || value == 8 || value == 16 || value == 32 || value == 64;
}

bool valid_plane(int plane) noexcept { return plane >= 0 && plane <= 2; }

// Capacity derived from one Config.  Every region is sized independently and
// rounded exactly like Arena::allocate_array, so required_device_bytes() is
// the exact byte count init() consumes.
struct Caps {
  bool valid = false;
  int max_blocks = 0;
  int max_candidates = 0;
  int cand_total = 0;
  std::uint64_t jobs_bytes = 0;
  std::uint64_t descs_bytes = 0;
  std::uint64_t results_bytes = 0;
  std::uint64_t pred_bytes = 0;
  std::uint64_t tx_jobs_bytes = 0;
  std::uint64_t tx_results_bytes = 0;
  std::uint64_t levels_bytes = 0;
  std::uint64_t residual_bytes = 0;
  std::uint64_t staging_bytes = 0;
};

Caps make_caps(const Config& config) noexcept {
  Caps caps;
  if (config.max_blocks < 1 || config.max_blocks > (1 << 20) ||
      config.max_candidates < 1 || config.max_candidates > kMaxCandidates ||
      config.max_block_area < 16 || config.max_block_area > 64 * 64 ||
      config.max_tx_area < 16 || config.max_tx_area > 32 * 32) {
    return caps;
  }
  const std::uint64_t blocks = static_cast<std::uint64_t>(config.max_blocks);
  const std::uint64_t cands =
      static_cast<std::uint64_t>(config.max_candidates);
  if (blocks * cands > static_cast<std::uint64_t>(INT_MAX)) {
    return caps;
  }
  caps.max_blocks = config.max_blocks;
  caps.max_candidates = config.max_candidates;
  caps.cand_total = static_cast<int>(blocks * cands);

  caps.jobs_bytes = align_up(blocks * sizeof(BlockJob));
  caps.descs_bytes =
      align_up(static_cast<std::uint64_t>(caps.cand_total) *
               sizeof(detail_v::CandidateDesc));
  caps.results_bytes =
      align_up(static_cast<std::uint64_t>(caps.cand_total) *
               sizeof(CandidateResult));
  caps.pred_bytes = align_up(static_cast<std::uint64_t>(caps.cand_total) *
                             static_cast<std::uint64_t>(config.max_block_area));

  const std::uint64_t tx_stage = align_up(
      static_cast<std::uint64_t>(caps.cand_total) * sizeof(StillTxJob));
  if (config.enable_tx) {
    caps.tx_jobs_bytes = tx_stage;
    caps.tx_results_bytes =
        align_up(static_cast<std::uint64_t>(caps.cand_total) *
                 sizeof(TxCandidateResult));
    caps.levels_bytes = align_up(static_cast<std::uint64_t>(caps.cand_total) *
                                 static_cast<std::uint64_t>(config.max_tx_area) *
                                 sizeof(std::int32_t));
    caps.residual_bytes =
        align_up(static_cast<std::uint64_t>(caps.cand_total) *
                 static_cast<std::uint64_t>(config.max_block_area) *
                 sizeof(std::int16_t));
  }

  caps.staging_bytes =
      std::max(caps.jobs_bytes + caps.descs_bytes,
               caps.tx_jobs_bytes + caps.tx_results_bytes);
  caps.valid = true;
  return caps;
}

// -------------------------------------------------------------------------
// Kernels
// -------------------------------------------------------------------------

__global__ void vintra_candidates_kernel(
    const BlockJob* __restrict__ jobs,
    const detail_v::CandidateDesc* __restrict__ descs,
    std::uint8_t* __restrict__ pred, std::int16_t* __restrict__ residual,
    CandidateResult* __restrict__ results) {
  detail_v::run_candidate(jobs, descs, static_cast<int>(blockIdx.x), pred,
                          residual, results);
}

}  // namespace

// ---------------------------------------------------------------------------
// Arena sizing
// ---------------------------------------------------------------------------

std::uint64_t required_device_bytes(const Config& config) noexcept {
  const Caps caps = make_caps(config);
  if (!caps.valid) {
    return 0;
  }
  std::uint64_t total = caps.staging_bytes + caps.results_bytes +
                        caps.pred_bytes;
  if (config.enable_tx) {
    total += caps.tx_results_bytes + caps.levels_bytes + caps.residual_bytes;
  }
  return total;
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

Status Workspace::release_and_restore(std::uint64_t mark, std::string* error,
                                      const char* what) {
  if (arena_ != nullptr) {
    arena_->rewind(mark);
  }
  if (error != nullptr) {
    *error = std::string("video_intra: ") + what;
  }
  return Status::kArenaExhausted;
}

Status Workspace::init(Arena& arena, const Config& config, std::string* error,
                       std::uint64_t* required_out) {
  arena_ = nullptr;
  config_ = Config{};
  staging_ = nullptr;
  staging_bytes_ = 0;
  d_results_ = nullptr;
  d_pred_ = nullptr;
  d_tx_results_ = nullptr;
  d_levels_ = nullptr;
  d_residual_ = nullptr;
  block_count_ = 0;
  candidate_total_ = 0;
  pred_total_ = 0;
  tx_level_total_ = 0;
  host_plan_.clear();
  host_results_.clear();

  const Caps caps = make_caps(config);
  const std::uint64_t need = required_device_bytes(config);
  if (required_out != nullptr) {
    *required_out = need;
  }
  if (!caps.valid) {
    if (error != nullptr) {
      *error = "video_intra: invalid configuration";
    }
    return Status::kInvalidArgument;
  }
  if (!arena.initialized()) {
    if (error != nullptr) {
      *error = "video_intra: arena is not initialized";
    }
    return Status::kNotInitialized;
  }
  if (arena.remaining() < need) {
    if (error != nullptr) {
      *error = "video_intra: arena needs " + std::to_string(need) + " bytes, " +
               std::to_string(arena.remaining()) + " available";
    }
    return Status::kArenaExhausted;
  }

  const std::uint64_t mark = arena.mark();
  staging_ = arena.allocate_array<std::uint8_t>(
      static_cast<std::size_t>(caps.staging_bytes));
  if (staging_ == nullptr) {
    return release_and_restore(mark, error, "staging allocation failed");
  }
  staging_bytes_ = caps.staging_bytes;
  d_results_ = arena.allocate_array<CandidateResult>(
      static_cast<std::size_t>(caps.cand_total));
  d_pred_ = arena.allocate_array<std::uint8_t>(
      static_cast<std::size_t>(caps.pred_bytes));
  if (d_results_ == nullptr || d_pred_ == nullptr) {
    return release_and_restore(mark, error, "result/prediction allocation failed");
  }
  if (config.enable_tx) {
    d_tx_results_ = arena.allocate_array<TxCandidateResult>(
        static_cast<std::size_t>(caps.cand_total));
    d_levels_ = arena.allocate_array<std::int32_t>(
        static_cast<std::size_t>(caps.levels_bytes / sizeof(std::int32_t)));
    d_residual_ = arena.allocate_array<std::int16_t>(
        static_cast<std::size_t>(caps.residual_bytes / sizeof(std::int16_t)));
    if (d_tx_results_ == nullptr || d_levels_ == nullptr ||
        d_residual_ == nullptr) {
      return release_and_restore(mark, error, "TX allocation failed");
    }
  }
  config_ = config;
  arena_ = &arena;
  return Status::kOk;
}

Status Workspace::evaluate(const BlockJob* jobs, int count,
                           const EvaluateOptions& options, cudaStream_t stream,
                           std::string* error) {
  if (arena_ == nullptr) {
    if (error != nullptr) {
      *error = "video_intra: workspace is not initialized";
    }
    return Status::kNotInitialized;
  }
  if (count < 0 || (count > 0 && jobs == nullptr)) {
    if (error != nullptr) {
      *error = "video_intra: bad job array";
    }
    return Status::kInvalidArgument;
  }
  if (options.run_tx && !config_.enable_tx) {
    if (error != nullptr) {
      *error = "video_intra: TX requested but Config::enable_tx is false";
    }
    return Status::kInvalidArgument;
  }

  block_count_ = 0;
  candidate_total_ = 0;
  pred_total_ = 0;
  tx_level_total_ = 0;
  host_plan_.clear();
  host_results_.clear();
  if (count == 0) {
    return Status::kOk;
  }

  const Caps caps = make_caps(config_);
  std::vector<detail_v::CandidateDesc> descs;
  std::vector<StillTxJob> tx_jobs;
  std::vector<TxCandidateResult> tx_results;
  descs.reserve(static_cast<std::size_t>(count) * 4);
  if (options.run_tx) {
    tx_jobs.reserve(static_cast<std::size_t>(count) * 4);
  }

  int candidate_total = 0;
  std::uint64_t pred_total = 0;
  std::uint64_t tx_level_total = 0;
  int max_tx_elems = 0;
  host_plan_.resize(static_cast<std::size_t>(count));

  Candidate candidates[kMaxCandidates];
  for (int i = 0; i < count; ++i) {
    const BlockJob& job = jobs[i];
    const char* reason = nullptr;
    if (!valid_dim(job.w) || !valid_dim(job.h)) {
      reason = "block dimensions must be 4/8/16/32/64";
    } else if (static_cast<std::int64_t>(job.w) * job.h >
               config_.max_block_area) {
      reason = "block area exceeds Config::max_block_area";
    } else if (!valid_plane(job.plane)) {
      reason = "plane must be 0, 1 or 2";
    } else if (job.recon_plane == nullptr || job.recon_stride <= 0) {
      reason = "missing reconstructed plane";
    } else if (job.source_plane == nullptr || job.source_stride <= 0) {
      reason = "missing source plane";
    } else if (job.x < 0 || job.y < 0 || job.max_x < job.x ||
               job.max_y < job.y) {
      reason = "bad origin/visible bounds";
    } else if (job.max_x >= job.recon_stride) {
      reason = "reconstructed stride is smaller than max_x + 1";
    } else if (job.source_max_x < 0 ||
               job.source_max_x >= job.source_stride) {
      reason = "bad source visible bounds / stride";
    } else if (job.have_left < 0 || job.have_left > 1 || job.have_above < 0 ||
               job.have_above > 1) {
      reason = "availability flags must be 0 or 1";
    } else if ((job.have_left != 0 && job.x < 1) ||
               (job.have_above != 0 && job.y < 1)) {
      reason = "contradictory availability (left/above at origin 0)";
    } else if (job.num4_above_right < 0 || job.num4_below_left < 0) {
      reason = "negative extension count";
    } else if (job.filter_type_above < 0 || job.filter_type_above > 1 ||
               job.filter_type_left < 0 || job.filter_type_left > 1) {
      reason = "filter type flags must be 0 or 1";
    } else if (job.enable_edge_filter < 0 || job.enable_edge_filter > 1) {
      reason = "enable_edge_filter must be 0 or 1";
    }
    if (reason != nullptr) {
      if (error != nullptr) {
        *error = "video_intra: job " + std::to_string(i) + ": " + reason;
      }
      return Status::kInvalidArgument;
    }

    const int n = build_candidates(job.plane, job.mode_mask,
                                   job.include_angle_deltas, candidates,
                                   kMaxCandidates);
    if (n <= 0) {
      if (error != nullptr) {
        *error = "video_intra: job " + std::to_string(i) +
                 ": empty or invalid candidate set";
      }
      return Status::kInvalidArgument;
    }
    if (n > config_.max_candidates) {
      if (error != nullptr) {
        *error = "video_intra: job " + std::to_string(i) + ": " +
                 std::to_string(n) + " candidates exceed Config::max_candidates";
      }
      return Status::kArenaExhausted;
    }
    if (candidate_total + n > caps.cand_total) {
      if (error != nullptr) {
        *error = "video_intra: candidate total exceeds capacity " +
                 std::to_string(caps.cand_total) +
                 " (raise Config::max_blocks/max_candidates)";
      }
      return Status::kArenaExhausted;
    }
    host_plan_[static_cast<std::size_t>(i)].candidate_offset = candidate_total;
    host_plan_[static_cast<std::size_t>(i)].candidate_count = n;

    for (int c = 0; c < n; ++c) {
      detail_v::CandidateDesc desc{};
      desc.job_index = i;
      desc.result_index = candidate_total + c;
      desc.pred_offset = static_cast<std::int32_t>(pred_total);
      desc.mode = candidates[c].mode;
      desc.angle_delta = candidates[c].angle_delta;
      descs.push_back(desc);
      pred_total += static_cast<std::uint64_t>(job.w) *
                    static_cast<std::uint64_t>(job.h);
    }
    candidate_total += n;
  }
  if (pred_total > caps.pred_bytes) {
    if (error != nullptr) {
      *error = "video_intra: prediction area exceeds capacity " +
               std::to_string(caps.pred_bytes) + " bytes";
    }
    return Status::kArenaExhausted;
  }
  if (pred_total > static_cast<std::uint64_t>(INT_MAX)) {
    if (error != nullptr) {
      *error = "video_intra: prediction area exceeds int32 offsets";
    }
    return Status::kInvalidArgument;
  }

  // Build the optional TX plan before staging so a bad size fails fast.
  if (options.run_tx) {
    tx_jobs.resize(static_cast<std::size_t>(candidate_total));
    tx_results.resize(static_cast<std::size_t>(candidate_total));
    int tx_index = 0;
    for (int i = 0; i < count; ++i) {
      const BlockJob& job = jobs[i];
      const auto& plan = host_plan_[static_cast<std::size_t>(i)];
      av2enc::txv::TxType type = job.plane == 0 ? options.luma_tx_type
                                                : options.chroma_tx_type;
      for (int c = 0; c < plan.candidate_count; ++c) {
        const int cand = plan.candidate_offset + c;
        StillTxJob tx{};
        if (!still_tx_init_job(&tx, job.w, job.h, type)) {
          if (error != nullptr) {
            *error = "video_intra: no TX size for " + std::to_string(job.w) +
                     "x" + std::to_string(job.h);
          }
          return Status::kInvalidArgument;
        }
        const int adj_elems = tx.params.w * tx.params.h;
        if (adj_elems > config_.max_tx_area) {
          if (error != nullptr) {
            *error = "video_intra: adjusted TX area exceeds "
                     "Config::max_tx_area";
          }
          return Status::kInvalidArgument;
        }
        tx.input =
            static_cast<std::uint8_t>(StillTxInput::kResidual);
        tx.src_packed = 0;
        // Residual offsets share the packed prediction prefix layout.
        const std::int32_t coded_offset =
            descs[static_cast<std::size_t>(cand)].pred_offset;
        tx.residual_in_offset = coded_offset;
        tx.levels_offset = static_cast<std::int32_t>(tx_level_total);
        tx.eob_offset = 2 * cand;  // TxCandidateResult is two int32 fields
        tx_jobs[static_cast<std::size_t>(tx_index)] = tx;
        tx_results[static_cast<std::size_t>(cand)].eob = 0;
        tx_results[static_cast<std::size_t>(cand)].levels_offset =
            tx.levels_offset;
        tx_level_total += static_cast<std::uint64_t>(adj_elems);
        if (adj_elems > max_tx_elems) {
          max_tx_elems = adj_elems;
        }
        ++tx_index;
      }
    }
    if (tx_level_total > caps.levels_bytes / sizeof(std::int32_t)) {
      if (error != nullptr) {
        *error = "video_intra: TX level area exceeds capacity";
      }
      return Status::kArenaExhausted;
    }
  }

  // ---------------------------------------------------------------------
  // Stage + launch (stream-ordered; one candidate launch for the batch).
  // ---------------------------------------------------------------------
  auto* staging = static_cast<std::uint8_t*>(staging_);
  const std::size_t jobs_bytes = static_cast<std::size_t>(count) *
                                 sizeof(BlockJob);
  const std::size_t descs_bytes =
      descs.size() * sizeof(detail_v::CandidateDesc);
  if (align_up(jobs_bytes) + align_up(descs_bytes) > staging_bytes_) {
    if (error != nullptr) {
      *error = "video_intra: descriptor staging exceeds capacity";
    }
    return Status::kArenaExhausted;
  }
  Status status = status_from_cuda(cudaMemcpyAsync(
      staging, jobs, jobs_bytes, cudaMemcpyHostToDevice, stream));
  if (!is_ok(status)) {
    if (error != nullptr) {
      *error = "video_intra: staging jobs failed";
    }
    return status;
  }
  auto* d_descs = reinterpret_cast<detail_v::CandidateDesc*>(
      staging + align_up(jobs_bytes));
  status = status_from_cuda(cudaMemcpyAsync(
      d_descs, descs.data(), descs_bytes, cudaMemcpyHostToDevice, stream));
  if (!is_ok(status)) {
    if (error != nullptr) {
      *error = "video_intra: staging descriptors failed";
    }
    return status;
  }

  auto* d_jobs = reinterpret_cast<BlockJob*>(staging);
  std::int16_t* d_residual = options.run_tx ? d_residual_ : nullptr;
  vintra_candidates_kernel<<<static_cast<unsigned>(candidate_total),
                             detail_v::kCandidateThreads, 0, stream>>>(
      d_jobs, d_descs, d_pred_, d_residual, d_results_);
  status = last_error();
  if (!is_ok(status)) {
    if (error != nullptr) {
      *error = "video_intra: candidate kernel launch failed";
    }
    return status;
  }

  if (options.run_tx) {
    const std::size_t tx_jobs_bytes =
        tx_jobs.size() * sizeof(StillTxJob);
    const std::size_t tx_res_bytes =
        tx_results.size() * sizeof(TxCandidateResult);
    if (align_up(tx_jobs_bytes) + align_up(tx_res_bytes) > staging_bytes_) {
      if (error != nullptr) {
        *error = "video_intra: TX staging exceeds capacity";
      }
      return Status::kArenaExhausted;
    }
    auto* d_tx_jobs = reinterpret_cast<StillTxJob*>(staging);
    status = status_from_cuda(cudaMemcpyAsync(
        d_tx_jobs, tx_jobs.data(), tx_jobs_bytes, cudaMemcpyHostToDevice,
        stream));
    if (!is_ok(status)) {
      if (error != nullptr) {
        *error = "video_intra: staging TX jobs failed";
      }
      return status;
    }
    status = status_from_cuda(cudaMemcpyAsync(
        d_tx_results_, tx_results.data(), tx_res_bytes,
        cudaMemcpyHostToDevice, stream));
    if (!is_ok(status)) {
      if (error != nullptr) {
        *error = "video_intra: staging TX results failed";
      }
      return status;
    }

    StillTxBuffers buffers{};
    buffers.residual_in = d_residual_;
    buffers.levels = d_levels_;
    // `eob` aliases the first int32 of each TxCandidateResult; still_tx treats
    // it as a flat int32 array and writes eob_offset = 2 * candidate.
    buffers.eob = reinterpret_cast<std::int32_t*>(d_tx_results_);
    const int smem_elems =
        options.tx_smem_elems > 0 ? options.tx_smem_elems : max_tx_elems;
    const cudaError_t launch =
        still_tx_batch(d_tx_jobs, candidate_total, buffers, smem_elems, stream);
    status = status_from_cuda(launch);
    if (!is_ok(status)) {
      if (error != nullptr) {
        *error = "video_intra: still_tx_batch launch failed";
      }
      return status;
    }
  }

  block_count_ = count;
  candidate_total_ = candidate_total;
  pred_total_ = static_cast<std::size_t>(pred_total);
  tx_level_total_ = static_cast<std::size_t>(tx_level_total);
  return Status::kOk;
}

Status Workspace::synchronize(cudaStream_t stream) {
  if (arena_ == nullptr) {
    return Status::kNotInitialized;
  }
  return status_from_cuda(cudaStreamSynchronize(stream));
}

int Workspace::candidate_offset(int block_index) const noexcept {
  if (block_index < 0 ||
      static_cast<std::size_t>(block_index) >= host_plan_.size()) {
    return 0;
  }
  return host_plan_[static_cast<std::size_t>(block_index)].candidate_offset;
}

int Workspace::candidate_count(int block_index) const noexcept {
  if (block_index < 0 ||
      static_cast<std::size_t>(block_index) >= host_plan_.size()) {
    return 0;
  }
  return host_plan_[static_cast<std::size_t>(block_index)].candidate_count;
}

namespace {

// Blocking device-to-host copy.  Downloads are host transfers, not the frame
// synchronisation: the caller still calls synchronize() once for the batch.
Status copy_to_host(void* dst, const void* src, std::size_t bytes,
                    cudaStream_t stream) {
  if (bytes == 0) {
    return Status::kOk;
  }
  Status status = status_from_cuda(cudaMemcpyAsync(
      dst, src, bytes, cudaMemcpyDeviceToHost, stream));
  if (!is_ok(status)) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(stream));
}

}  // namespace

Status Workspace::download_results(CandidateResult* out, int capacity,
                                   cudaStream_t stream) const {
  if (arena_ == nullptr) {
    return Status::kNotInitialized;
  }
  if (out == nullptr || capacity < candidate_total_) {
    return Status::kInvalidArgument;
  }
  return copy_to_host(
      out, d_results_,
      static_cast<std::size_t>(candidate_total_) * sizeof(CandidateResult),
      stream);
}

Status Workspace::download_prediction(std::uint8_t* out, std::size_t capacity,
                                      cudaStream_t stream) const {
  if (arena_ == nullptr) {
    return Status::kNotInitialized;
  }
  if (out == nullptr || capacity < pred_total_) {
    return Status::kInvalidArgument;
  }
  return copy_to_host(out, d_pred_, pred_total_, stream);
}

Status Workspace::download_tx(TxCandidateResult* out, int capacity,
                              cudaStream_t stream) {
  if (arena_ == nullptr || d_tx_results_ == nullptr) {
    return Status::kNotInitialized;
  }
  if (out == nullptr || capacity < candidate_total_) {
    return Status::kInvalidArgument;
  }
  return copy_to_host(
      out, d_tx_results_,
      static_cast<std::size_t>(candidate_total_) * sizeof(TxCandidateResult),
      stream);
}

Status Workspace::download_residual(std::int16_t* out, std::size_t capacity,
                                    cudaStream_t stream) const {
  if (arena_ == nullptr || d_residual_ == nullptr) {
    return Status::kNotInitialized;
  }
  if (out == nullptr || capacity < pred_total_) {
    return Status::kInvalidArgument;
  }
  return copy_to_host(out, d_residual_, pred_total_ * sizeof(std::int16_t),
                      stream);
}

Status Workspace::download_best(BestResult* out, int capacity,
                                cudaStream_t stream) {
  if (arena_ == nullptr) {
    return Status::kNotInitialized;
  }
  if (out == nullptr || capacity < block_count_ || block_count_ <= 0) {
    return Status::kInvalidArgument;
  }
  host_results_.resize(static_cast<std::size_t>(candidate_total_));
  Status status = download_results(host_results_.data(), candidate_total_,
                                   stream);
  if (!is_ok(status)) {
    return status;
  }
  for (int block = 0; block < block_count_; ++block) {
    const int begin = candidate_offset(block);
    const int count = candidate_count(block);
    BestResult best{};
    best.block = block;
    best.candidate_index = -1;
    best.sse = INT64_MAX;
    for (int i = 0; i < count; ++i) {
      const CandidateResult& candidate =
          host_results_[static_cast<std::size_t>(begin + i)];
      if (candidate.sse < best.sse) {
        best.sse = candidate.sse;
        best.candidate_index = begin + i;
        best.mode = candidate.mode;
        best.angle_delta = candidate.angle_delta;
        best.plane = candidate.plane;
      }
    }
    out[block] = best;
  }
  return Status::kOk;
}

}  // namespace av2enc::cuda::video::intra
