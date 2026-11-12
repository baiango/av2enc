// Shared POD descriptors for the CUDA TCQ / CCTX / parity-hiding batch.
//
// This header is deliberately CUDA-free so the device core (tcq_device.cuh)
// and the host emulation test can include it with any compiler.  The CUDA API
// (tcq.h) and the kernels (tcq.cu) use the same structures.
//
// Layout
// ------
// Every descriptor addresses the workspace by element offset.  A slot's base
// offsets are `slot * <per-slot capacity>`, so kernels never know how many
// slots exist; they only see the offsets.  See tcq.h for the per-slot
// capacities and tcq_workspace_batch() for a convenience builder.
//
// The CPU TCQ input is a coefficient block already in §5.20.7.30 scan order.
// `pre_max` is per coefficient (`tcq::Coefficient::pre_max`); the host helper
// `tcq_scan_and_pre_max()` derives it from TX size / scan class / plane with
// the same `level_limits()` rule the parity-hiding reference uses.
//
// The rate tables are caller-filled (host side) by invoking the exact
// `tcq::RateModel` callbacks.  That keeps the device DP free of transcendentals
// and makes the GPU decisions bit-identical to `tcq::quantize()` for an
// arbitrary CPU rate model, not just the built-in one.  Index layout:
//
//   normal / eob_coeff:  rate_offset + ((c * 8) + state) * table_levels + q
//   block EOB:           eob_table_offset + eob
//   path choice:         path_offset + c * 8 + state
//   outputs:             output_offset + c
//
// `table_levels` (L) must be > every derived coefficient bound.
#pragma once

#include <cstdint>

namespace av2enc::cuda::tcq {

// ---------------------------------------------------------------------------
// Numeric contract (mirrors src/encoder/tcq/tcq.h)
// ---------------------------------------------------------------------------

enum : std::int32_t {
  kStateCount = 8,        // §5.20.7.27 Tcq_Next_State rows
  kCctxTypeCount = 7,     // CCTX_NONE + six angles
  kMaxLevelsCap = 4096,   // sanity bound on TcqJob::table_levels
};

// One trellis-coded quantisation job.  `coefficient_offset` addresses
// original/pre_max/level_context; the rate/path/output offsets address the
// buffers documented in tcq.h.
struct TcqJob {
  std::int32_t coefficient_offset = 0;
  std::int32_t num_coefficients = 0;
  std::int32_t rate_offset = 0;
  std::int32_t table_levels = 1;  // L
  std::int32_t eob_table_offset = 0;
  std::int32_t path_offset = 0;
  std::int32_t output_offset = 0;
  std::int32_t tail_offset = 0;  // count+1 doubles per job
  std::int32_t level_cap = 0;          // 0 = derived bound + 64
  std::int32_t allow_zero_to_one = 0;
  double quant_step = 1.0;
  double lambda = 0.0;
};

// CCTX seven-candidate scalar search job.  U and V are parallel arrays of
// `num_coefficients` doubles.  `candidate_offset` addresses the per-candidate
// level arrays: candidate t is at candidate_offset + t * num_coefficients.
struct CctxJob {
  std::int32_t coefficient_offset = 0;  // into u_coeff / v_coeff
  std::int32_t candidate_offset = 0;    // into u_levels / v_levels
  std::int32_t num_coefficients = 0;
  std::int32_t enabled = 1;             // 0 = CCTX_NONE only
  std::int32_t bitdepth = 8;            // 8..16
  double quant_step_u = 1.0;
  double quant_step_v = 1.0;
};

// Parity-hiding encoder level choice (minimum-distortion mode).
//
// `tx_class` matches av2enc::tcq::parity::TxClass (0 = 2D, 1 = HORIZ,
// 2 = VERT).  `scan` is the §5.20.7.30 scan of the adjusted block; `w` is the
// adjusted width so row = scan[c] / w.  Eligibility mirrors
// parity::Eligibility / parity::parity_hiding_eligible().
struct ParityJob {
  std::int32_t coefficient_offset = 0;  // into original and seed_levels
  std::int32_t output_offset = 0;       // into levels
  std::int32_t scan_offset = 0;
  std::int32_t num_coefficients = 0;
  std::int32_t w = 0;                   // adjusted width
  std::int32_t tx_class = 0;
  std::int32_t plane = 0;
  std::int32_t allow_parity_hiding = 0;
  std::int32_t lossless = 0;
  std::int32_t allow_tcq = 0;
  std::int32_t tx_type_is_idtx = 0;
  std::int32_t use_fsc = 0;
  std::int32_t exhaustive_max_area = 64;
  std::int32_t pruned_ac_candidates = 8;
  double quant_step = 1.0;
};

// ---------------------------------------------------------------------------
// Device batches.  `tcq_run_*_job(batch, index)` in tcq_device.cuh is the whole
// per-job body; the kernels are thin thread-index wrappers around it.
// ---------------------------------------------------------------------------

struct TcqDeviceBatch {
  const TcqJob* jobs = nullptr;
  std::int32_t num_jobs = 0;
  const double* original = nullptr;
  const std::int32_t* pre_max = nullptr;
  const std::int32_t* level_context = nullptr;
  const double* normal_rate = nullptr;
  const double* eob_coeff_rate = nullptr;
  const double* block_eob_rate = nullptr;
  std::int32_t* path_base = nullptr;
  std::int32_t* path_quant = nullptr;
  std::int32_t* path_next = nullptr;
  double* tail = nullptr;
  std::int32_t* base_levels = nullptr;
  std::int32_t* quant_levels = nullptr;
  std::int32_t* states = nullptr;
  std::int32_t* eob = nullptr;
  double* distortion = nullptr;
  double* rate_bits = nullptr;
  double* cost = nullptr;
  std::int32_t* max_level_searched = nullptr;
  std::uint8_t* level_range_truncated = nullptr;
  std::int32_t* status = nullptr;  // 0 = ok, -1 = rejected job
};

struct CctxDeviceBatch {
  const CctxJob* jobs = nullptr;
  std::int32_t num_jobs = 0;
  const double* u_coeff = nullptr;
  const double* v_coeff = nullptr;
  std::int32_t* u_levels = nullptr;
  std::int32_t* v_levels = nullptr;
  std::int32_t* candidate_type = nullptr;
  double* candidate_distortion = nullptr;
  std::int32_t* candidate_eob_u = nullptr;
  std::int32_t* candidate_eob_v = nullptr;
  std::int32_t* best_type = nullptr;
  double* best_distortion = nullptr;
  std::int32_t* best_eob_u = nullptr;
  std::int32_t* best_eob_v = nullptr;
  std::int32_t* status = nullptr;
};

struct ParityDeviceBatch {
  const ParityJob* jobs = nullptr;
  std::int32_t num_jobs = 0;
  const double* original = nullptr;
  const std::int32_t* seed_levels = nullptr;
  const std::int32_t* scan = nullptr;
  std::int32_t* levels = nullptr;
  std::int32_t* eob = nullptr;
  std::int32_t* hidden = nullptr;
  std::int32_t* hidable = nullptr;
  std::int32_t* hiding_considered = nullptr;
  std::int32_t* hiding_turned_off = nullptr;
  std::int32_t* adjusted_scan_index = nullptr;
  std::int32_t* adjusted_delta = nullptr;
  double* distortion = nullptr;
  std::int32_t* status = nullptr;
};

}  // namespace av2enc::cuda::tcq
