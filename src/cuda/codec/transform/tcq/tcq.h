// CUDA TCQ / CCTX / parity-hiding batch API.
//
// Scope
// -----
// This module is the GPU implementation of the encoder-side quantisers in
// src/encoder/tcq/:
//
//   * trellis-coded quantisation: the 8-state DP over the reverse scan with
//     the exact `tcq::quantize()` cost model (SSE + lambda * bits, identical
//     tie-breaks and arithmetic order);
//   * the §5.20.7.27 `syntax_level_for` / `syntax_escape_for` mapping,
//     including the parity-dependent `pre_max` / `pre_max - 1` choice;
//   * CCTX: the six §7.14.6 rotations, joint U/V dequant, and the
//     seven-candidate (CCTX_NONE + six angles) scalar search;
//   * parity-hiding level choice in minimum-distortion mode
//     (`tcq::parity::QuantizeMode::kMinDistortion`);
//   * 8/10-bit coefficient magnitudes, all 25 TX sizes, and 2D/HORIZ/VERT
//     scan classes (the scan and `pre_max` are caller-supplied, see
//     tcq_scan_and_pre_max()).
//
// Memory contract
// ---------------
// Every device buffer this module needs is allocated once from the caller's
// persistent av2enc::cuda::Arena by `workspace_create()`.  There is no
// cudaMalloc/cudaFree here, no implicit synchronization, and no per-launch
// allocation.  `required_device_bytes(config)` is the exact persistent size,
// including the Arena's 256-byte rounding of every buffer.  If the arena is
// too small, `workspace_create()` returns Status::kArenaExhausted (the
// repository's name for the docs' "kArenaTooSmall") and writes that exact size
// to *out_required_bytes; the arena is rewound, so nothing leaks.  Launchers
// take no mark and allocate nothing, so repeated launches cannot grow the
// arena.  `persistent_mark` records the watermark for callers that want to
// take scratch of their own.
//
// Rate model
// ----------
// The device DP does not call function pointers and has no transcendentals.
// The caller fills the rate tables from the *CPU* tcq::RateModel with
// `tcq_fill_rate_tables()` (tcq_host.h), then uploads them.  That keeps the
// GPU decisions bit-identical to `tcq::quantize()` for an arbitrary rate
// model, including the non-monotone test model used by the CPU suite.
//
// Seams (documented in README.md)
// -------------------------------
//   * parity hiding: only kMinDistortion is implemented on device; the §9.3
//     static-CDF kRateDistortion search stays on the host;
//   * parity hiding: per-position `dequant_step` (QM) is not part of the batch
//     (use a uniform `quant_step`);
//   * parity hiding outputs `rate_bits = 0` / `cost = distortion` because the
//     device path has no CDF tables.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "cuda/arena.h"
#include "cuda/error.h"
#include "cuda/codec/transform/tcq/tcq_types.h"

namespace av2enc::cuda::tcq {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct TcqConfig {
  // Slots in one workspace; each slot is one coefficient block.
  int max_jobs = 8;
  // Largest block area (adjusted w*h).  32x32 = 1024 covers every TX size.
  int max_coefficients = 1024;
  // Largest rate-table stride L.  A job's `table_levels` must be <= this and
  // > every derived coefficient bound (`detail::tcq_level_bound`).
  int max_levels = 32;
};

inline constexpr std::uint64_t kTcqAlignment = 256;

inline std::uint64_t tcq_align_up(std::uint64_t value) {
  return (value + (kTcqAlignment - 1)) & ~(kTcqAlignment - 1);
}

inline bool tcq_config_valid(const TcqConfig& config) {
  if (config.max_jobs <= 0 || config.max_coefficients <= 0 ||
      config.max_levels <= 0 || config.max_levels > kMaxLevelsCap) {
    return false;
  }
  const std::int64_t jobs = config.max_jobs;
  const std::int64_t coeff = config.max_coefficients;
  const std::int64_t levels = config.max_levels;
  const std::int64_t rate_elements = jobs * coeff * kStateCount * levels;
  const std::int64_t cctx_elements = jobs * kCctxTypeCount * coeff;
  constexpr std::int64_t kInt32Max = 2147483647;
  return rate_elements <= kInt32Max && cctx_elements <= kInt32Max &&
         jobs * coeff <= kInt32Max;
}

// One entry per Arena allocation.  The same list drives the byte total, the
// allocation loop and the split report, so they cannot drift apart.  The last
// field is the diagnostic group (0 = TCQ, 1 = CCTX, 2 = parity hiding).
#define TCQ_WORKSPACE_BUFFERS(X)                                      \
  X(TcqJob, jobs, jobs, 0)                                            \
  X(double, original, jobs * coeff, 0)                                \
  X(std::int32_t, pre_max, jobs * coeff, 0)                           \
  X(std::int32_t, level_context, jobs * coeff, 0)                     \
  X(double, normal_rate, jobs * coeff * kStateCount * levels, 0)      \
  X(double, eob_coeff_rate, jobs * coeff * kStateCount * levels, 0)   \
  X(double, block_eob_rate, jobs * (coeff + 1), 0)                    \
  X(double, tail, jobs * (coeff + 1), 0)                              \
  X(std::int32_t, path_base, jobs * coeff * kStateCount, 0)           \
  X(std::int32_t, path_quant, jobs * coeff * kStateCount, 0)          \
  X(std::int32_t, path_next, jobs * coeff * kStateCount, 0)           \
  X(std::int32_t, base_levels, jobs * coeff, 0)                       \
  X(std::int32_t, quant_levels, jobs * coeff, 0)                      \
  X(std::int32_t, states, jobs * coeff, 0)                            \
  X(std::int32_t, eob, jobs, 0)                                       \
  X(std::int32_t, max_level_searched, jobs, 0)                        \
  X(std::uint8_t, level_range_truncated, jobs, 0)                     \
  X(double, distortion, jobs, 0)                                      \
  X(double, rate_bits, jobs, 0)                                       \
  X(double, cost, jobs, 0)                                            \
  X(std::int32_t, status, jobs, 0)                                    \
  X(CctxJob, cctx_jobs, jobs, 1)                                      \
  X(double, u_coeff, jobs * coeff, 1)                                 \
  X(double, v_coeff, jobs * coeff, 1)                                 \
  X(std::int32_t, cctx_u_levels, jobs * kCctxTypeCount * coeff, 1)    \
  X(std::int32_t, cctx_v_levels, jobs * kCctxTypeCount * coeff, 1)    \
  X(std::int32_t, cctx_candidate_type, jobs * kCctxTypeCount, 1)      \
  X(double, cctx_candidate_distortion, jobs * kCctxTypeCount, 1)      \
  X(std::int32_t, cctx_candidate_eob_u, jobs * kCctxTypeCount, 1)     \
  X(std::int32_t, cctx_candidate_eob_v, jobs * kCctxTypeCount, 1)     \
  X(std::int32_t, cctx_best_type, jobs, 1)                            \
  X(double, cctx_best_distortion, jobs, 1)                            \
  X(std::int32_t, cctx_best_eob_u, jobs, 1)                           \
  X(std::int32_t, cctx_best_eob_v, jobs, 1)                           \
  X(std::int32_t, cctx_status, jobs, 1)                               \
  X(ParityJob, parity_jobs, jobs, 2)                                  \
  X(std::int32_t, scan, jobs * coeff, 2)                              \
  X(std::int32_t, parity_levels, jobs * coeff, 2)                     \
  X(std::int32_t, parity_eob, jobs, 2)                                \
  X(std::int32_t, parity_hidden, jobs, 2)                             \
  X(std::int32_t, parity_hidable, jobs, 2)                            \
  X(std::int32_t, parity_hiding_considered, jobs, 2)                  \
  X(std::int32_t, parity_hiding_turned_off, jobs, 2)                  \
  X(std::int32_t, parity_adjusted_scan_index, jobs, 2)                \
  X(std::int32_t, parity_adjusted_delta, jobs, 2)                     \
  X(double, parity_distortion, jobs, 2)                               \
  X(std::int32_t, parity_status, jobs, 2)

struct TcqBytes {
  std::uint64_t tcq = 0;
  std::uint64_t cctx = 0;
  std::uint64_t parity = 0;
  std::uint64_t total = 0;
};

// Exact persistent footprint in bytes, including Arena rounding.  Zero for an
// invalid configuration.
inline std::uint64_t required_device_bytes(const TcqConfig& config) {
  if (!tcq_config_valid(config)) return 0;
  const std::uint64_t jobs = static_cast<std::uint64_t>(config.max_jobs);
  const std::uint64_t coeff =
      static_cast<std::uint64_t>(config.max_coefficients);
  const std::uint64_t levels = static_cast<std::uint64_t>(config.max_levels);
  std::uint64_t total = 0;
#define TCQ_ADD_BYTES(T, name, n, group) \
  total += tcq_align_up(sizeof(T) * static_cast<std::uint64_t>(n));
  TCQ_WORKSPACE_BUFFERS(TCQ_ADD_BYTES)
#undef TCQ_ADD_BYTES
  return total;
}

// Same numbers, grouped for diagnostics.  `total == required_device_bytes()`.
inline TcqBytes tcq_bytes(const TcqConfig& config) {
  TcqBytes out;
  if (!tcq_config_valid(config)) return out;
  const std::uint64_t jobs = static_cast<std::uint64_t>(config.max_jobs);
  const std::uint64_t coeff =
      static_cast<std::uint64_t>(config.max_coefficients);
  const std::uint64_t levels = static_cast<std::uint64_t>(config.max_levels);
#define TCQ_ADD_GROUP(T, name, n, group)                                  \
  {                                                                       \
    const std::uint64_t bytes =                                           \
        tcq_align_up(sizeof(T) * static_cast<std::uint64_t>(n));          \
    if ((group) == 0) {                                                   \
      out.tcq += bytes;                                                   \
    } else if ((group) == 1) {                                            \
      out.cctx += bytes;                                                  \
    } else {                                                              \
      out.parity += bytes;                                                \
    }                                                                     \
  }
  TCQ_WORKSPACE_BUFFERS(TCQ_ADD_GROUP)
#undef TCQ_ADD_GROUP
  out.total = out.tcq + out.cctx + out.parity;
  return out;
}

// ---------------------------------------------------------------------------
// Arena workspace
// ---------------------------------------------------------------------------

struct TcqWorkspace {
  // TCQ
  TcqJob* jobs = nullptr;
  double* original = nullptr;
  std::int32_t* pre_max = nullptr;
  std::int32_t* level_context = nullptr;
  double* normal_rate = nullptr;
  double* eob_coeff_rate = nullptr;
  double* block_eob_rate = nullptr;
  double* tail = nullptr;
  std::int32_t* path_base = nullptr;
  std::int32_t* path_quant = nullptr;
  std::int32_t* path_next = nullptr;
  std::int32_t* base_levels = nullptr;
  std::int32_t* quant_levels = nullptr;
  std::int32_t* states = nullptr;
  std::int32_t* eob = nullptr;
  std::int32_t* max_level_searched = nullptr;
  std::uint8_t* level_range_truncated = nullptr;
  double* distortion = nullptr;
  double* rate_bits = nullptr;
  double* cost = nullptr;
  std::int32_t* status = nullptr;
  // CCTX
  CctxJob* cctx_jobs = nullptr;
  double* u_coeff = nullptr;
  double* v_coeff = nullptr;
  std::int32_t* cctx_u_levels = nullptr;
  std::int32_t* cctx_v_levels = nullptr;
  std::int32_t* cctx_candidate_type = nullptr;
  double* cctx_candidate_distortion = nullptr;
  std::int32_t* cctx_candidate_eob_u = nullptr;
  std::int32_t* cctx_candidate_eob_v = nullptr;
  std::int32_t* cctx_best_type = nullptr;
  double* cctx_best_distortion = nullptr;
  std::int32_t* cctx_best_eob_u = nullptr;
  std::int32_t* cctx_best_eob_v = nullptr;
  std::int32_t* cctx_status = nullptr;
  // Parity hiding
  ParityJob* parity_jobs = nullptr;
  std::int32_t* scan = nullptr;
  std::int32_t* parity_levels = nullptr;
  std::int32_t* parity_eob = nullptr;
  std::int32_t* parity_hidden = nullptr;
  std::int32_t* parity_hidable = nullptr;
  std::int32_t* parity_hiding_considered = nullptr;
  std::int32_t* parity_hiding_turned_off = nullptr;
  std::int32_t* parity_adjusted_scan_index = nullptr;
  std::int32_t* parity_adjusted_delta = nullptr;
  double* parity_distortion = nullptr;
  std::int32_t* parity_status = nullptr;

  int max_jobs = 0;
  int max_coefficients = 0;
  int max_levels = 0;
  std::uint64_t required_bytes = 0;
  // Arena watermark after the persistent allocations.  Launchers never move
  // it; a caller that takes its own scratch should rewind to this value.
  std::uint64_t persistent_mark = 0;
};

// Allocates the persistent set once.  Returns:
//   kOk              workspace ready
//   kInvalidArgument bad config / null output
//   kNotInitialized  arena not initialised
//   kArenaExhausted  arena too small; *out_required_bytes is the exact size
//                    and the arena is left untouched
inline Status workspace_create(Arena& arena, const TcqConfig& config,
                               TcqWorkspace* out,
                               std::uint64_t* out_required_bytes = nullptr,
                               std::string* error = nullptr) {
  const std::uint64_t required = required_device_bytes(config);
  if (out_required_bytes != nullptr) *out_required_bytes = required;
  if (out == nullptr || required == 0) {
    if (error != nullptr) {
      *error = "tcq: invalid workspace configuration";
    }
    return Status::kInvalidArgument;
  }
  *out = TcqWorkspace{};
  if (!arena.initialized()) {
    if (error != nullptr) *error = "tcq: arena not initialized";
    return Status::kNotInitialized;
  }
  if (arena.remaining() < required) {
    if (error != nullptr) {
      *error = "tcq: arena too small: need " + std::to_string(required) +
               " bytes, have " + std::to_string(arena.remaining());
    }
    return Status::kArenaExhausted;
  }

  const std::uint64_t start_mark = arena.mark();
  const std::uint64_t jobs = static_cast<std::uint64_t>(config.max_jobs);
  const std::uint64_t coeff =
      static_cast<std::uint64_t>(config.max_coefficients);
  const std::uint64_t levels =
      static_cast<std::uint64_t>(config.max_levels);
#define TCQ_ALLOC(T, name, n, group)                                       \
  {                                                                        \
    const std::uint64_t bytes =                                            \
        sizeof(T) * static_cast<std::uint64_t>(n);                         \
    const Status status = arena.allocate_checked(                          \
        static_cast<std::size_t>(bytes),                                   \
        reinterpret_cast<void**>(&out->name));                             \
    if (status != Status::kOk) {                                           \
      arena.rewind(start_mark);                                            \
      if (error != nullptr) {                                              \
        *error = "tcq: arena allocation failed for " #name;                \
      }                                                                    \
      return status;                                                       \
    }                                                                      \
  }
  TCQ_WORKSPACE_BUFFERS(TCQ_ALLOC)
#undef TCQ_ALLOC

  out->max_jobs = config.max_jobs;
  out->max_coefficients = config.max_coefficients;
  out->max_levels = config.max_levels;
  out->required_bytes = required;
  out->persistent_mark = arena.mark();
  if (out->persistent_mark != required) {
    // The allocator rounds every request to the 256-byte alignment, so the
    // watermark must equal the reported total.  Treat a mismatch as a bug.
    if (error != nullptr) {
      *error = "tcq: workspace footprint mismatch (required " +
               std::to_string(required) + ", used " +
               std::to_string(out->persistent_mark) + ")";
    }
    arena.rewind(start_mark);
    *out = TcqWorkspace{};
    return Status::kInvalidArgument;
  }
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// Slot helpers.  Every offset is a multiple of the slot capacity, so kernels
// only see element offsets.
// ---------------------------------------------------------------------------

inline TcqJob tcq_make_job(const TcqConfig& config, int slot,
                           std::int32_t count, std::int32_t table_levels,
                           double quant_step, double lambda,
                           std::int32_t level_cap = 0,
                           bool allow_zero_to_one = false) {
  TcqJob job;
  job.coefficient_offset = slot * config.max_coefficients;
  job.num_coefficients = count;
  job.rate_offset =
      slot * config.max_coefficients * kStateCount * config.max_levels;
  job.table_levels = table_levels;
  job.eob_table_offset = slot * (config.max_coefficients + 1);
  job.path_offset = slot * config.max_coefficients * kStateCount;
  job.output_offset = slot * config.max_coefficients;
  job.tail_offset = slot * (config.max_coefficients + 1);
  job.level_cap = level_cap;
  job.allow_zero_to_one = allow_zero_to_one ? 1 : 0;
  job.quant_step = quant_step;
  job.lambda = lambda;
  return job;
}

inline CctxJob tcq_make_cctx_job(const TcqConfig& config, int slot,
                                 std::int32_t count, double quant_step_u,
                                 double quant_step_v, std::int32_t bitdepth,
                                 bool enabled = true) {
  CctxJob job;
  job.coefficient_offset = slot * config.max_coefficients;
  job.candidate_offset =
      slot * kCctxTypeCount * config.max_coefficients;
  job.num_coefficients = count;
  job.enabled = enabled ? 1 : 0;
  job.bitdepth = bitdepth;
  job.quant_step_u = quant_step_u;
  job.quant_step_v = quant_step_v;
  return job;
}

inline ParityJob tcq_make_parity_job(const TcqConfig& config, int slot,
                                     std::int32_t count, std::int32_t w,
                                     std::int32_t tx_class,
                                     std::int32_t plane, double quant_step,
                                     bool allow_parity_hiding) {
  ParityJob job;
  job.coefficient_offset = slot * config.max_coefficients;
  job.output_offset = slot * config.max_coefficients;
  job.scan_offset = slot * config.max_coefficients;
  job.num_coefficients = count;
  job.w = w;
  job.tx_class = tx_class;
  job.plane = plane;
  job.allow_parity_hiding = allow_parity_hiding ? 1 : 0;
  job.quant_step = quant_step;
  return job;
}

// ---------------------------------------------------------------------------
// Device batch views
// ---------------------------------------------------------------------------

inline TcqDeviceBatch tcq_batch(TcqWorkspace& ws, int job_count) {
  TcqDeviceBatch batch;
  batch.jobs = ws.jobs;
  batch.num_jobs = job_count;
  batch.original = ws.original;
  batch.pre_max = ws.pre_max;
  batch.level_context = ws.level_context;
  batch.normal_rate = ws.normal_rate;
  batch.eob_coeff_rate = ws.eob_coeff_rate;
  batch.block_eob_rate = ws.block_eob_rate;
  batch.path_base = ws.path_base;
  batch.path_quant = ws.path_quant;
  batch.path_next = ws.path_next;
  batch.tail = ws.tail;
  batch.base_levels = ws.base_levels;
  batch.quant_levels = ws.quant_levels;
  batch.states = ws.states;
  batch.eob = ws.eob;
  batch.distortion = ws.distortion;
  batch.rate_bits = ws.rate_bits;
  batch.cost = ws.cost;
  batch.max_level_searched = ws.max_level_searched;
  batch.level_range_truncated = ws.level_range_truncated;
  batch.status = ws.status;
  return batch;
}

inline CctxDeviceBatch cctx_batch(TcqWorkspace& ws, int job_count) {
  CctxDeviceBatch batch;
  batch.jobs = ws.cctx_jobs;
  batch.num_jobs = job_count;
  batch.u_coeff = ws.u_coeff;
  batch.v_coeff = ws.v_coeff;
  batch.u_levels = ws.cctx_u_levels;
  batch.v_levels = ws.cctx_v_levels;
  batch.candidate_type = ws.cctx_candidate_type;
  batch.candidate_distortion = ws.cctx_candidate_distortion;
  batch.candidate_eob_u = ws.cctx_candidate_eob_u;
  batch.candidate_eob_v = ws.cctx_candidate_eob_v;
  batch.best_type = ws.cctx_best_type;
  batch.best_distortion = ws.cctx_best_distortion;
  batch.best_eob_u = ws.cctx_best_eob_u;
  batch.best_eob_v = ws.cctx_best_eob_v;
  batch.status = ws.cctx_status;
  return batch;
}

inline ParityDeviceBatch parity_batch(TcqWorkspace& ws, int job_count) {
  ParityDeviceBatch batch;
  batch.jobs = ws.parity_jobs;
  batch.num_jobs = job_count;
  batch.original = ws.original;
  // The seed is read and updated in place: candidate evaluation always finishes
  // before the chosen level is written back.
  batch.seed_levels = ws.parity_levels;
  batch.scan = ws.scan;
  batch.levels = ws.parity_levels;
  batch.eob = ws.parity_eob;
  batch.hidden = ws.parity_hidden;
  batch.hidable = ws.parity_hidable;
  batch.hiding_considered = ws.parity_hiding_considered;
  batch.hiding_turned_off = ws.parity_hiding_turned_off;
  batch.adjusted_scan_index = ws.parity_adjusted_scan_index;
  batch.adjusted_delta = ws.parity_adjusted_delta;
  batch.distortion = ws.parity_distortion;
  batch.status = ws.parity_status;
  return batch;
}

// ---------------------------------------------------------------------------
// Launchers (defined in tcq.cu)
// ---------------------------------------------------------------------------
//
// `host_jobs` is copied into the workspace slot array before the launch.  The
// caller fills the coefficient / pre_max / rate buffers through the workspace
// pointers above (typically with cudaMemcpy).  No launcher allocates, frees or
// synchronizes.

cudaError_t tcq_quantize_batch(TcqWorkspace& workspace,
                               const TcqJob* host_jobs, int job_count,
                               cudaStream_t stream = nullptr);

cudaError_t cctx_search_batch(TcqWorkspace& workspace,
                              const CctxJob* host_jobs, int job_count,
                              cudaStream_t stream = nullptr);

cudaError_t parity_hide_batch(TcqWorkspace& workspace,
                              const ParityJob* host_jobs, int job_count,
                              cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::tcq
